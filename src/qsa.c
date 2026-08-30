#include "fg_qsa.h"
#include "fg_qsa_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FG_QSA_OWNER_LAYERS 6u
#define FG_QSA_MAX_LAYERS 12u
#define FG_QSA_SELECTED_TOKENS (FG_Q38_INDEX_BUDGET+FG_Q38_QSA_COMPRESS_RATIO-1u)
#define FG_QSA_HOT_LIMIT_BYTES (UINT64_C(768)*1024u*1024u)

struct fg_qsa_session {
    fg_model *model;
    fg_qsa_state *state;
    uint32_t max_context,max_blocks,max_tokens,layer_count,hot_capacity,committed[FG_QSA_MAX_LAYERS];
    uint8_t layers[FG_QSA_MAX_LAYERS];
    fg_vk_tensor *positions,*index_history,*hot_records;
    fg_vk_tensor *index_keys[FG_QSA_MAX_LAYERS];
    uint8_t partial[FG_QSA_MAX_LAYERS][FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES];
    fg_vk_tensor *raw_query_gate,*raw_key,*raw_value,*query,*gate,*key;
    fg_vk_tensor *raw_index_query,*raw_index_key,*index_query;
    fg_vk_tensor *key_q8,*value_q4,*index_key_q8;
    fg_vk_tensor *scores[2],*ids[2],*selected_records,*attention,*output;
    uint8_t *read_records;
};

static fg_status make_tensor(fg_qsa_session *s,uint64_t bytes,fg_vk_tensor **out,fg_error *err){return fg_vk_tensor_create(fg_model_vk(s->model),bytes,out,err);}
static uint32_t get_u32_le(const uint8_t *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8u)|((uint32_t)p[2]<<16u)|((uint32_t)p[3]<<24u);}
static void put_u32_le(uint8_t *p,uint32_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8u);p[2]=(uint8_t)(v>>16u);p[3]=(uint8_t)(v>>24u);}

static int layer_slot(const fg_qsa_session *s,uint32_t layer){for(uint32_t i=0;i<s->layer_count;i++)if(s->layers[i]==layer)return (int)i;return -1;}
static fg_vk_tensor *layer_weight(fg_qsa_session *s,uint32_t layer,const char *suffix,fg_error *err){char name[FG_TENSOR_NAME_MAX];int n=snprintf(name,sizeof(name),"blk.%u.%s",layer,suffix);if(n<0||(uint32_t)n>=sizeof(name)){fg_error_set(err,FG_ERR_LIMIT,"QSA tensor name overflow");return NULL;}fg_vk_tensor *tensor=fg_model_tensor(s->model,name);if(!tensor)fg_error_set(err,FG_ERR_MISMATCH,"QSA owner is missing %s",name);return tensor;}

static fg_status restore_state(fg_qsa_session *s,fg_error *err){
    uint32_t common=fg_qsa_state_layer_tokens(s->state,0);for(uint32_t slot=1;slot<s->layer_count;slot++)if(fg_qsa_state_layer_tokens(s->state,slot)!=common){fg_error_set(err,FG_ERR_MISMATCH,"QSA layer checkpoints stop at different token boundaries");return FG_ERR_MISMATCH;}
    uint32_t *positions=fg_vk_tensor_map(s->positions);uint32_t blocks=(common+3u)/4u;uint32_t block_ids[FG_QSA_MAX_SELECTED_BLOCKS],committed[FG_QSA_MAX_SELECTED_BLOCKS];
    for(uint32_t slot=0;slot<s->layer_count;slot++){
        uint8_t *resident=fg_vk_tensor_map(s->index_keys[slot]);
        for(uint32_t first=0;first<blocks;first+=FG_QSA_MAX_SELECTED_BLOCKS){uint32_t count=blocks-first;if(count>FG_QSA_MAX_SELECTED_BLOCKS)count=FG_QSA_MAX_SELECTED_BLOCKS;for(uint32_t i=0;i<count;i++)block_ids[i]=first+i;fg_status status=fg_qsa_state_read_blocks(s->state,slot,block_ids,count,s->read_records,committed,err);if(status!=FG_OK)return status;for(uint32_t i=0;i<count;i++){uint32_t expected=block_ids[i]+1u==blocks&&common%4u?common%4u:4u;if(committed[i]!=expected){fg_error_set(err,FG_ERR_MISMATCH,"QSA checkpoint page length does not match header");return FG_ERR_MISMATCH;}for(uint32_t inside=0;inside<committed[i];inside++){uint32_t token=block_ids[i]*4u+inside;const uint8_t *record=s->read_records+((uint64_t)i*4u+inside)*FG_Q38_QSA_TOKEN_RECORD_BYTES;const uint8_t *metadata=record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES;memcpy(resident+(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES,metadata,FG_Q38_QSA_INDEX_KEY_BYTES);for(uint32_t axis=0;axis<3u;axis++){uint32_t value=get_u32_le(metadata+FG_Q38_QSA_INDEX_KEY_BYTES+axis*4u);if(slot&&positions[(uint64_t)token*3u+axis]!=value){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers disagree on persisted MRoPE positions");return FG_ERR_MISMATCH;}positions[(uint64_t)token*3u+axis]=value;}}} }
        if(common%4u){uint32_t committed_tokens=0;fg_status status=fg_qsa_state_read_block(s->state,slot,blocks-1u,s->partial[slot],&committed_tokens,err);if(status!=FG_OK)return status;if(committed_tokens!=common%4u){fg_error_set(err,FG_ERR_MISMATCH,"QSA partial checkpoint length mismatch");return FG_ERR_MISMATCH;}}
    }
    return FG_OK;
}

fg_status fg_qsa_session_open(fg_qsa_session **out,fg_model *model,const char *path,bool create,fg_error *err){
    if(!out||!model||!path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA session arguments");return FG_ERR_ARGUMENT;}*out=NULL;const fg_manifest *manifest=fg_model_manifest(model);uint32_t rank=fg_model_rank(model);fg_qsa_session *s=calloc(1,sizeof(*s));if(!s){fg_error_set(err,FG_ERR_OOM,"allocate QSA session");return FG_ERR_OOM;}s->model=model;s->max_context=manifest->max_context;s->max_blocks=(s->max_context+3u)/4u;s->max_tokens=manifest->prefill_microbatch;if(!s->max_tokens||s->max_tokens>512u){fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds QSA session limit");fg_qsa_session_close(s);return FG_ERR_MISMATCH;}
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u){
        if(rank==0u||manifest->layer_owner[layer]==rank)s->layers[s->layer_count++]=(uint8_t)layer;
    }
    uint32_t expected_layers=rank==0u?FG_QSA_MAX_LAYERS:FG_QSA_OWNER_LAYERS;
    if(s->layer_count!=expected_layers){fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA layers, expected %u",rank,s->layer_count,expected_layers);fg_qsa_session_close(s);return FG_ERR_MISMATCH;}
    fg_status status=fg_qsa_state_open(&s->state,path,s->layers,s->layer_count,s->max_context,create,err);if(status==FG_OK)status=make_tensor(s,(uint64_t)s->max_context*FG_Q38_QSA_POSITION_BYTES,&s->positions,err);for(uint32_t i=0;status==FG_OK&&i<s->layer_count;i++)status=make_tensor(s,(uint64_t)s->max_context*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_keys[i],err);
    uint64_t batch=s->max_tokens;if(status==FG_OK)status=make_tensor(s,batch*12288u*4u,&s->raw_query_gate,err);
    if(status==FG_OK)status=make_tensor(s,batch*512u*4u,&s->raw_key,err);
    if(status==FG_OK)status=make_tensor(s,batch*512u*4u,&s->raw_value,err);
    if(status==FG_OK)status=make_tensor(s,batch*6144u*4u,&s->query,err);
    if(status==FG_OK)status=make_tensor(s,batch*6144u*4u,&s->gate,err);
    if(status==FG_OK)status=make_tensor(s,batch*512u*4u,&s->key,err);
    if(status==FG_OK)status=make_tensor(s,batch*512u*4u,&s->raw_index_query,err);
    if(status==FG_OK)status=make_tensor(s,batch*128u*4u,&s->raw_index_key,err);
    if(status==FG_OK)status=make_tensor(s,batch*512u*4u,&s->index_query,err);
    if(status==FG_OK)status=make_tensor(s,batch*FG_Q38_QSA_KEY_BYTES,&s->key_q8,err);
    if(status==FG_OK)status=make_tensor(s,batch*FG_Q38_QSA_VALUE_BYTES,&s->value_q4,err);
    if(status==FG_OK)status=make_tensor(s,batch*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_key_q8,err);
    for(uint32_t i=0;status==FG_OK&&i<2u;i++){status=make_tensor(s,(uint64_t)s->max_blocks*4u,&s->scores[i],err);if(status==FG_OK)status=make_tensor(s,(uint64_t)s->max_blocks*4u,&s->ids[i],err);}
    if(status==FG_OK)status=make_tensor(s,(uint64_t)FG_QSA_SELECTED_TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES,&s->selected_records,err);
    if(status==FG_OK)status=make_tensor(s,batch*6144u*4u,&s->attention,err);
    if(status==FG_OK)status=make_tensor(s,batch*2560u*4u,&s->output,err);
    s->read_records=status==FG_OK?malloc((uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*4u*FG_Q38_QSA_TOKEN_RECORD_BYTES):NULL;if(status==FG_OK&&!s->read_records){fg_error_set(err,FG_ERR_OOM,"allocate QSA state staging");status=FG_ERR_OOM;}if(status==FG_OK&&!create)status=restore_state(s,err);if(status==FG_OK)for(uint32_t slot=0;slot<s->layer_count;slot++)s->committed[slot]=fg_qsa_state_layer_tokens(s->state,slot);if(status!=FG_OK){fg_qsa_session_close(s);return status;}*out=s;return FG_OK;
}

/* Decode-only QSA session: uses resident_tokens for GPU index allocation (not max_context),
   batch=1 for scratch.  Saves ~1.6 GiB GPU memory for the coordinator. */
fg_status fg_qsa_session_open_decode(fg_qsa_session **out,fg_model *model,const char *state_path,uint32_t resident_tokens,uint32_t batch_size,fg_error *err){
    if(!out||!model||!state_path||!resident_tokens||!batch_size){fg_error_set(err,FG_ERR_ARGUMENT,"invalid decode QSA session arguments");return FG_ERR_ARGUMENT;}*out=NULL;const fg_manifest *manifest=fg_model_manifest(model);uint32_t rank=fg_model_rank(model);fg_qsa_session *s=calloc(1,sizeof(*s));if(!s){fg_error_set(err,FG_ERR_OOM,"allocate QSA session");return FG_ERR_OOM;}s->model=model;s->max_context=resident_tokens;s->max_blocks=(s->max_context+3u)/4u;s->max_tokens=batch_size;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u){if(rank==0u||manifest->layer_owner[layer]==rank)s->layers[s->layer_count++]=(uint8_t)layer;}
    uint32_t expected_layers=rank==0u?FG_QSA_MAX_LAYERS:FG_QSA_OWNER_LAYERS;if(s->layer_count!=expected_layers){fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA layers, expected %u",rank,s->layer_count,expected_layers);fg_qsa_session_close(s);return FG_ERR_MISMATCH;}
    uint64_t hot_bytes=(uint64_t)resident_tokens*s->layer_count*FG_Q38_QSA_TOKEN_RECORD_BYTES;if(hot_bytes<=FG_QSA_HOT_LIMIT_BYTES)s->hot_capacity=resident_tokens;
    fg_status status=fg_qsa_state_open(&s->state,state_path,s->layers,s->layer_count,manifest->max_context,true,err);if(status==FG_OK){status=make_tensor(s,(uint64_t)resident_tokens*FG_Q38_QSA_POSITION_BYTES,&s->positions,err);}if(status==FG_OK&&s->hot_capacity){status=make_tensor(s,(uint64_t)s->layer_count*s->hot_capacity*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_history,err);if(status==FG_OK)status=make_tensor(s,hot_bytes,&s->hot_records,err);if(status==FG_OK){memset(fg_vk_tensor_map(s->index_history),0,(size_t)fg_vk_tensor_bytes(s->index_history));memset(fg_vk_tensor_map(s->hot_records),0,(size_t)fg_vk_tensor_bytes(s->hot_records));}for(uint32_t i=0;status==FG_OK&&i<s->layer_count;i++)status=fg_vk_tensor_view(s->index_history,(uint64_t)i*s->hot_capacity*FG_Q38_QSA_INDEX_KEY_BYTES,(uint64_t)s->hot_capacity*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_keys[i],err);}else for(uint32_t i=0;status==FG_OK&&i<s->layer_count;i++)status=make_tensor(s,(uint64_t)resident_tokens*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_keys[i],err);
    if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*12288u*4u,&s->raw_query_gate,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->raw_key,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->raw_value,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->query,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->gate,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->key,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->raw_index_query,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*128u*4u,&s->raw_index_key,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->index_query,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*FG_Q38_QSA_KEY_BYTES,&s->key_q8,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*FG_Q38_QSA_VALUE_BYTES,&s->value_q4,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_key_q8,err);}
    for(uint32_t i=0;status==FG_OK&&i<2u;i++){status=make_tensor(s,(uint64_t)s->max_blocks*4u,&s->scores[i],err);if(status==FG_OK)status=make_tensor(s,(uint64_t)s->max_blocks*4u,&s->ids[i],err);}
    if(status==FG_OK){status=make_tensor(s,(uint64_t)FG_QSA_SELECTED_TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES,&s->selected_records,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->attention,err);}if(status==FG_OK){status=make_tensor(s,(uint64_t)batch_size*2560u*4u,&s->output,err);}
    s->read_records=status==FG_OK?malloc((uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*4u*FG_Q38_QSA_TOKEN_RECORD_BYTES):NULL;if(status==FG_OK&&!s->read_records){fg_error_set(err,FG_ERR_OOM,"allocate QSA decode staging");status=FG_ERR_OOM;}
    fprintf(stderr,"[rank %u] QSA decode session: %u tokens, %u layers, %.1f MiB index, %.1f MiB hot records\n",rank,resident_tokens,s->layer_count,(double)((uint64_t)resident_tokens*(FG_Q38_QSA_POSITION_BYTES+s->layer_count*FG_Q38_QSA_INDEX_KEY_BYTES))/(1024.0*1024.0),s->hot_capacity?(double)hot_bytes/(1024.0*1024.0):0.0);
    if(status!=FG_OK){fg_qsa_session_close(s);return status;}*out=s;return FG_OK;
}

void fg_qsa_session_close(fg_qsa_session *s){if(!s)return;free(s->read_records);fg_vk_tensor_destroy(s->output);fg_vk_tensor_destroy(s->attention);fg_vk_tensor_destroy(s->selected_records);for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(s->ids[i]);fg_vk_tensor_destroy(s->scores[i]);}fg_vk_tensor_destroy(s->index_key_q8);fg_vk_tensor_destroy(s->value_q4);fg_vk_tensor_destroy(s->key_q8);fg_vk_tensor_destroy(s->index_query);fg_vk_tensor_destroy(s->raw_index_key);fg_vk_tensor_destroy(s->raw_index_query);fg_vk_tensor_destroy(s->key);fg_vk_tensor_destroy(s->gate);fg_vk_tensor_destroy(s->query);fg_vk_tensor_destroy(s->raw_value);fg_vk_tensor_destroy(s->raw_key);fg_vk_tensor_destroy(s->raw_query_gate);for(uint32_t i=0;i<FG_QSA_MAX_LAYERS;i++)fg_vk_tensor_destroy(s->index_keys[i]);fg_vk_tensor_destroy(s->hot_records);fg_vk_tensor_destroy(s->index_history);fg_vk_tensor_destroy(s->positions);fg_qsa_state_close(s->state);free(s);}

fg_status fg_qsa_session_reset(fg_qsa_session *s,fg_error *err){if(!s){fg_error_set(err,FG_ERR_ARGUMENT,"QSA session reset is null");return FG_ERR_ARGUMENT;}memset(s->committed,0,sizeof(s->committed));memset(s->partial,0,sizeof(s->partial));return fg_qsa_state_reset(s->state,err);}

fg_status fg_qsa_session_checkpoint(fg_qsa_session *s,fg_error *err){
    if(!s){fg_error_set(err,FG_ERR_ARGUMENT,"QSA checkpoint session is null");return FG_ERR_ARGUMENT;}if(!s->hot_capacity)return FG_OK;if(fg_vk_batch_active(fg_model_vk(s->model))){fg_error_set(err,FG_ERR_ARGUMENT,"QSA checkpoint cannot run inside a Vulkan batch");return FG_ERR_ARGUMENT;}
    const uint8_t *records=fg_vk_tensor_map(s->hot_records);fg_status status=FG_OK;
    for(uint32_t slot=0;status==FG_OK&&slot<s->layer_count;slot++){
        uint32_t persisted=fg_qsa_state_layer_tokens(s->state,slot),committed=s->committed[slot];if(persisted>committed||committed>s->hot_capacity){fg_error_set(err,FG_ERR_MISMATCH,"QSA checkpoint token range is invalid");status=FG_ERR_MISMATCH;break;}
        while(status==FG_OK&&persisted<committed){uint32_t block=persisted/FG_Q38_QSA_COMPRESS_RATIO,first=block*FG_Q38_QSA_COMPRESS_RATIO,count=committed-first;if(count>FG_Q38_QSA_COMPRESS_RATIO)count=FG_Q38_QSA_COMPRESS_RATIO;const uint8_t *page_records=records+((uint64_t)slot*s->hot_capacity+first)*FG_Q38_QSA_TOKEN_RECORD_BYTES;status=fg_qsa_state_write_block(s->state,slot,block,page_records,count,err);persisted=first+count;}
    }
    return status;
}

uint32_t fg_qsa_session_tokens(const fg_qsa_session *s,uint32_t layer){int slot=s?layer_slot(s,layer):-1;return slot<0?0:s->committed[slot];}
void fg_qsa_session_set_tokens(fg_qsa_session *s,uint32_t tokens){if(!s)return;for(uint32_t i=0;i<s->layer_count;i++){s->committed[i]=tokens;if(!s->hot_capacity)fg_qsa_state_set_layer_tokens(s->state,i,tokens);}}

static fg_status select_blocks(fg_qsa_session *s,uint32_t slot,const fg_vk_tensor *index_query,uint32_t tokens,uint32_t *selected,uint32_t *selected_count,fg_error *err){uint32_t count=tokens/4u;if(!count){*selected_count=0;return FG_OK;}fg_vk_tensor *key_norm=layer_weight(s,s->layers[slot],"indexer.k_norm.weight",err);if(!key_norm)return FG_ERR_MISMATCH;fg_status status=fg_vk_qsa_index_score(fg_model_vk(s->model),s->scores[0],s->ids[0],index_query,s->index_keys[slot],key_norm,s->positions,tokens,err);uint32_t side=0;while(status==FG_OK&&count>512u){uint32_t next=0;status=fg_vk_topk_reduce(fg_model_vk(s->model),s->scores[side^1u],s->ids[side^1u],s->scores[side],s->ids[side],count,&next,err);count=next;side^=1u;}if(status==FG_OK){uint32_t final_count=0;status=fg_vk_topk_reduce(fg_model_vk(s->model),s->scores[side^1u],s->ids[side^1u],s->scores[side],s->ids[side],count,&final_count,err);if(status==FG_OK){count=final_count;side^=1u;status=fg_vk_tensor_read(s->ids[side],0,selected,(uint64_t)count*4u,err);}}if(status==FG_OK)*selected_count=count;return status;}

static fg_status select_blocks_hot(fg_qsa_session *s,uint32_t slot,const fg_vk_tensor *index_query,uint32_t tokens,const fg_vk_tensor **selected_ids,uint32_t *selected_count,fg_error *err){
    uint32_t count=tokens/FG_Q38_QSA_COMPRESS_RATIO;if(count<=FG_QSA_MAX_SELECTED_BLOCKS){*selected_ids=NULL;*selected_count=count;return FG_OK;}
    fg_vk_tensor *key_norm=layer_weight(s,s->layers[slot],"indexer.k_norm.weight",err);if(!key_norm)return FG_ERR_MISMATCH;
    fg_status status=fg_vk_qsa_index_score(fg_model_vk(s->model),s->scores[0],s->ids[0],index_query,s->index_keys[slot],key_norm,s->positions,tokens,err);uint32_t side=0;
    while(status==FG_OK&&count>FG_QSA_MAX_SELECTED_BLOCKS){uint32_t next=0;status=fg_vk_topk_reduce(fg_model_vk(s->model),s->scores[side^1u],s->ids[side^1u],s->scores[side],s->ids[side],count,&next,err);count=next;side^=1u;}
    if(status==FG_OK){*selected_ids=s->ids[side];*selected_count=count;}return status;
}

static fg_status commit_and_attend_hot(fg_qsa_session *s,uint32_t slot,uint32_t token,const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,const fg_vk_tensor *index_query,const fg_vk_tensor *query,const fg_vk_tensor *gate,fg_vk_tensor *attention,fg_error *err){
    if(token!=s->committed[slot]||token>=s->hot_capacity){fg_error_set(err,FG_ERR_MISMATCH,"QSA hot token position does not match committed state");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(s->model);fg_status status=fg_vk_qsa_record_commit(vk,s->hot_records,s->index_history,key_q8,value_q8,index_key_q8,position,slot,token,s->hot_capacity,err);uint32_t tokens=token+1u;if(status==FG_OK)s->committed[slot]=tokens;
    const fg_vk_tensor *selected_ids=NULL;uint32_t block_count=0;if(status==FG_OK)status=select_blocks_hot(s,slot,index_query,tokens,&selected_ids,&block_count,err);uint32_t tail=tokens%FG_Q38_QSA_COMPRESS_RATIO;
    if(status==FG_OK&&!selected_ids){fg_vk_tensor *records=NULL;status=fg_vk_tensor_view(s->hot_records,(uint64_t)slot*s->hot_capacity*FG_Q38_QSA_TOKEN_RECORD_BYTES,(uint64_t)tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,&records,err);if(status==FG_OK)status=fg_vk_qsa_attention(vk,attention,records,query,gate,tokens,err);fg_vk_tensor_destroy(records);}
    else if(status==FG_OK){uint32_t tail_start=tokens-tail;status=fg_vk_qsa_record_gather(vk,s->selected_records,s->hot_records,selected_ids,slot,s->hot_capacity,block_count,tail_start,tail,err);if(status==FG_OK)status=fg_vk_qsa_attention(vk,attention,s->selected_records,query,gate,block_count*FG_Q38_QSA_COMPRESS_RATIO+tail,err);}
    return status;
}

static fg_status commit_and_attend(fg_qsa_session *s,uint32_t slot,uint32_t token,const uint32_t position[3],const uint8_t *key,const uint8_t *value,const uint8_t *index_key,const fg_vk_tensor *index_query,const fg_vk_tensor *query,const fg_vk_tensor *gate,fg_vk_tensor *attention,fg_error *err){
    if(token!=fg_qsa_state_layer_tokens(s->state,slot)||token>=s->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA token position does not match committed state");return FG_ERR_MISMATCH;}
    uint32_t inside=token%FG_Q38_QSA_COMPRESS_RATIO;if(!inside)memset(s->partial[slot],0,sizeof(s->partial[slot]));uint8_t *record=s->partial[slot]+(uint64_t)inside*FG_Q38_QSA_TOKEN_RECORD_BYTES;memcpy(record,key,FG_Q38_QSA_KEY_BYTES);memcpy(record+FG_Q38_QSA_KEY_BYTES,value,FG_Q38_QSA_VALUE_BYTES);memcpy(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,index_key,FG_Q38_QSA_INDEX_KEY_BYTES);for(uint32_t axis=0;axis<3u;axis++)put_u32_le(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES+axis*4u,position[axis]);
    fg_status status=fg_qsa_state_write_block(s->state,slot,token/FG_Q38_QSA_COMPRESS_RATIO,s->partial[slot],inside+1u,err);if(status!=FG_OK)return status;s->committed[slot]=token+1u;memcpy((uint8_t *)fg_vk_tensor_map(s->index_keys[slot])+(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES,index_key,FG_Q38_QSA_INDEX_KEY_BYTES);
    uint32_t selected_blocks[FG_QSA_MAX_SELECTED_BLOCKS],block_count=0,tokens=token+1u;status=select_blocks(s,slot,index_query,tokens,selected_blocks,&block_count,err);uint32_t selected_tokens=0;if(status==FG_OK&&block_count){uint32_t committed[FG_QSA_MAX_SELECTED_BLOCKS];status=fg_qsa_state_read_blocks(s->state,slot,selected_blocks,block_count,s->read_records,committed,err);for(uint32_t i=0;status==FG_OK&&i<block_count;i++){if(committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){fg_error_set(err,FG_ERR_MISMATCH,"selected QSA block is not complete");status=FG_ERR_MISMATCH;break;}memcpy((uint8_t *)fg_vk_tensor_map(s->selected_records)+(uint64_t)selected_tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,s->read_records+(uint64_t)i*FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES,(uint64_t)FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES);selected_tokens+=FG_Q38_QSA_COMPRESS_RATIO;}}
    uint32_t tail=tokens%FG_Q38_QSA_COMPRESS_RATIO;if(status==FG_OK&&tail){memcpy((uint8_t *)fg_vk_tensor_map(s->selected_records)+(uint64_t)selected_tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,s->partial[slot],(uint64_t)tail*FG_Q38_QSA_TOKEN_RECORD_BYTES);selected_tokens+=tail;}if(status==FG_OK)status=fg_vk_qsa_attention(fg_model_vk(s->model),attention,s->selected_records,query,gate,selected_tokens,err);return status;
}

fg_status fg_qsa_session_decode(fg_qsa_session *s,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;if(!s||signed_slot<0||!position||!hidden||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA decode arguments");return FG_ERR_ARGUMENT;}uint32_t slot=(uint32_t)signed_slot;if(token!=s->committed[slot]||token>=s->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA token position does not match committed state");return FG_ERR_MISMATCH;}fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err),*kw=layer_weight(s,layer,"attn_k.weight",err),*vw=layer_weight(s,layer,"attn_v.weight",err),*qn=layer_weight(s,layer,"attn_q_norm.weight",err),*kn=layer_weight(s,layer,"attn_k_norm.weight",err),*ow=layer_weight(s,layer,"attn_output.weight",err),*iqw=layer_weight(s,layer,"indexer.q_proj.weight",err),*ikw=layer_weight(s,layer,"indexer.k_proj.weight",err),*iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn)return FG_ERR_MISMATCH;
    uint32_t *resident_positions=fg_vk_tensor_map(s->positions);if(slot==0)memcpy(resident_positions+(uint64_t)token*3u,position,12u);else if(memcmp(resident_positions+(uint64_t)token*3u,position,12u)!=0){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers received inconsistent MRoPE positions");return FG_ERR_MISMATCH;}fg_vk_tensor *position_view=NULL;fg_status status=fg_vk_tensor_view(s->positions,(uint64_t)token*12u,12u,&position_view,err);fg_vk_context *vk=fg_model_vk(s->model);if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_projection",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_query_gate,qw,hidden,2560u,12288u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_key,kw,hidden,2560u,512u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_value,vw,hidden,2560u,512u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare(vk,s->query,s->gate,s->key,s->raw_query_gate,s->raw_key,qn,kn,position_view,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_query,iqw,hidden,2560u,512u,1u,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_key,ikw,hidden,2560u,128u,1u,err);
    if(status==FG_OK)status=fg_vk_qsa_index_prepare(vk,s->index_query,s->raw_index_query,iqn,position_view,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->key_q8,s->key,512u,1u,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->value_q4,s->raw_value,512u,1u,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->index_key_q8,s->raw_index_key,128u,1u,err);
    if(status!=FG_OK){fg_vk_tensor_destroy(position_view);return status;}
    if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_state_attention",err);
    if(status==FG_OK&&s->hot_capacity)status=commit_and_attend_hot(s,slot,token,s->key_q8,s->value_q4,s->index_key_q8,position_view,s->index_query,s->query,s->gate,s->attention,err);
    else if(status==FG_OK)status=commit_and_attend(s,slot,token,position,fg_vk_tensor_map(s->key_q8),fg_vk_tensor_map(s->value_q4),fg_vk_tensor_map(s->index_key_q8),s->index_query,s->query,s->gate,s->attention,err);
    fg_vk_tensor_destroy(position_view);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_output",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->output,ow,s->attention,6144u,2560u,1u,1.0f,err);
    if(status==FG_OK&&token<30u){const float *ap=fg_vk_tensor_map(s->attention),*op=fg_vk_tensor_map(s->output);double a2=0.0,o2=0.0;for(uint32_t i=0;i<6144u;i++)a2+=(double)ap[i]*ap[i];for(uint32_t i=0;i<2560u;i++)o2+=(double)op[i]*op[i];fprintf(stderr,"qsa[%u] t=%u selected=%u attn_rms=%.6f out_rms=%.6f\n",layer,token,token+1u,sqrt(a2/6144.0),sqrt(o2/2560.0));}
    if(status==FG_OK){*output=s->output;}return status;
}

fg_status fg_qsa_session_prefill(fg_qsa_session *s,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint32_t token_count,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;if(!s||signed_slot<0||!positions||!hidden||!output||!token_count||token_count>s->max_tokens||token_count>s->max_context||first_token>s->max_context-token_count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prefill arguments");return FG_ERR_ARGUMENT;}uint32_t slot=(uint32_t)signed_slot;if(first_token!=s->committed[slot]){fg_error_set(err,FG_ERR_MISMATCH,"QSA prefill range does not start at committed state");return FG_ERR_MISMATCH;}
    fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err),*kw=layer_weight(s,layer,"attn_k.weight",err),*vw=layer_weight(s,layer,"attn_v.weight",err),*qn=layer_weight(s,layer,"attn_q_norm.weight",err),*kn=layer_weight(s,layer,"attn_k_norm.weight",err),*ow=layer_weight(s,layer,"attn_output.weight",err),*iqw=layer_weight(s,layer,"indexer.q_proj.weight",err),*ikw=layer_weight(s,layer,"indexer.k_proj.weight",err),*iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn)return FG_ERR_MISMATCH;
    uint32_t *resident_positions=fg_vk_tensor_map(s->positions);for(uint32_t i=0;i<token_count;i++){uint32_t token=first_token+i;uint32_t *resident=resident_positions+(uint64_t)token*3u;const uint32_t *position=positions+(uint64_t)i*3u;if(slot==0u)memcpy(resident,position,FG_Q38_QSA_POSITION_BYTES);else if(memcmp(resident,position,FG_Q38_QSA_POSITION_BYTES)!=0){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers received inconsistent MRoPE positions");return FG_ERR_MISMATCH;}}
    fg_vk_tensor *position_view=NULL;fg_status status=fg_vk_tensor_view(s->positions,(uint64_t)first_token*FG_Q38_QSA_POSITION_BYTES,(uint64_t)token_count*FG_Q38_QSA_POSITION_BYTES,&position_view,err);fg_vk_context *vk=fg_model_vk(s->model);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_query_gate,qw,hidden,FG_HIDDEN_SIZE,12288u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_key,kw,hidden,FG_HIDDEN_SIZE,512u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_value,vw,hidden,FG_HIDDEN_SIZE,512u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare_prefill(vk,s->query,s->gate,s->key,s->raw_query_gate,s->raw_key,qn,kn,position_view,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_query,iqw,hidden,FG_HIDDEN_SIZE,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_key,ikw,hidden,FG_HIDDEN_SIZE,128u,token_count,err);
    if(status==FG_OK)status=fg_vk_qsa_index_prepare_prefill(vk,s->index_query,s->raw_index_query,iqn,position_view,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->key_q8,s->key,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->value_q4,s->raw_value,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->index_key_q8,s->raw_index_key,128u,token_count,err);
    fg_vk_tensor_destroy(position_view);if(status!=FG_OK)return status;
    const uint8_t *keys=s->hot_capacity?NULL:fg_vk_tensor_map(s->key_q8),*values=s->hot_capacity?NULL:fg_vk_tensor_map(s->value_q4),*index_keys=s->hot_capacity?NULL:fg_vk_tensor_map(s->index_key_q8);
    for(uint32_t i=0;status==FG_OK&&i<token_count;i++){
        fg_vk_tensor *index_query=NULL,*query=NULL,*gate=NULL,*attention=NULL,*key_q8=NULL,*value_q8=NULL,*index_key_q8=NULL,*token_position=NULL;
        status=fg_vk_tensor_view(s->index_query,(uint64_t)i*512u*4u,512u*4u,&index_query,err);
        if(status==FG_OK)status=fg_vk_tensor_view(s->query,(uint64_t)i*6144u*4u,6144u*4u,&query,err);
        if(status==FG_OK)status=fg_vk_tensor_view(s->gate,(uint64_t)i*6144u*4u,6144u*4u,&gate,err);
        if(status==FG_OK)status=fg_vk_tensor_view(s->attention,(uint64_t)i*6144u*4u,6144u*4u,&attention,err);
        if(status==FG_OK&&s->hot_capacity)status=fg_vk_tensor_view(s->key_q8,(uint64_t)i*FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_KEY_BYTES,&key_q8,err);
        if(status==FG_OK&&s->hot_capacity)status=fg_vk_tensor_view(s->value_q4,(uint64_t)i*FG_Q38_QSA_VALUE_BYTES,FG_Q38_QSA_VALUE_BYTES,&value_q8,err);
        if(status==FG_OK&&s->hot_capacity)status=fg_vk_tensor_view(s->index_key_q8,(uint64_t)i*FG_Q38_QSA_INDEX_KEY_BYTES,FG_Q38_QSA_INDEX_KEY_BYTES,&index_key_q8,err);
        if(status==FG_OK&&s->hot_capacity)status=fg_vk_tensor_view(s->positions,(uint64_t)(first_token+i)*FG_Q38_QSA_POSITION_BYTES,FG_Q38_QSA_POSITION_BYTES,&token_position,err);
        if(status==FG_OK&&s->hot_capacity)status=commit_and_attend_hot(s,slot,first_token+i,key_q8,value_q8,index_key_q8,token_position,index_query,query,gate,attention,err);
        else if(status==FG_OK)status=commit_and_attend(s,slot,first_token+i,positions+(uint64_t)i*3u,keys+(uint64_t)i*FG_Q38_QSA_KEY_BYTES,values+(uint64_t)i*FG_Q38_QSA_VALUE_BYTES,index_keys+(uint64_t)i*FG_Q38_QSA_INDEX_KEY_BYTES,index_query,query,gate,attention,err);
        fg_vk_tensor_destroy(token_position);fg_vk_tensor_destroy(index_key_q8);fg_vk_tensor_destroy(value_q8);fg_vk_tensor_destroy(key_q8);fg_vk_tensor_destroy(attention);fg_vk_tensor_destroy(gate);fg_vk_tensor_destroy(query);fg_vk_tensor_destroy(index_query);
    }
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->output,ow,s->attention,6144u,FG_HIDDEN_SIZE,token_count,1.0f,err);
    if(status==FG_OK)*output=s->output;
    return status;
}
