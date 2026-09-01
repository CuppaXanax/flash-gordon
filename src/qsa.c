#include "fg_qsa.h"
#include "fg_qsa_cache.h"
#include "fg_qsa_locality.h"
#include "fg_qsa_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#define FG_QSA_OWNER_LAYERS 6u
#define FG_QSA_MAX_LAYERS 12u
struct fg_qsa_session {
    fg_model *model;
    fg_qsa_state *state;
    uint32_t max_context,max_blocks,max_tokens,layer_count;
    uint32_t cache_pages;
    uint32_t index_segment_count,index_segment_tokens[FG_QSA_INDEX_MAX_SEGMENTS];
    uint32_t committed[FG_QSA_MAX_LAYERS];
    uint8_t layers[FG_QSA_MAX_LAYERS];
    fg_vk_tensor *positions,*cache_records;
    fg_vk_tensor *index_keys[FG_QSA_MAX_LAYERS][FG_QSA_INDEX_MAX_SEGMENTS];
    fg_vk_tensor *records[FG_QSA_MAX_LAYERS][FG_QSA_RESIDENT_MAX_SEGMENTS];
    uint8_t partial[FG_QSA_MAX_LAYERS][FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES];
    fg_vk_tensor *raw_query_gate,*raw_key,*raw_value,*query,*gate,*key;
    fg_vk_tensor *raw_index_query,*raw_index_key,*index_query;
    fg_vk_tensor *key_q8,*value_q4,*index_key_q8;
    fg_vk_tensor *scores[2],*ids[2],*selected_records,*attention,*output;
    fg_vk_tensor *position_view,*token_position_view,*index_query_view,*query_view,*gate_view;
    fg_vk_tensor *attention_view,*key_q8_view,*value_q4_view,*index_key_q8_view;
    uint8_t *read_records;
    fg_qsa_page_cache *cache;
    fg_qsa_page_fetch_fn fetch_pages;
    void *fetch_opaque;
    fg_qsa_locality *locality;
    bool resident;
};

static fg_status make_tensor(fg_qsa_session *s,uint64_t bytes,fg_vk_tensor **out,fg_error *err){return fg_vk_tensor_create(fg_model_vk(s->model),bytes,out,err);}

fg_status fg_qsa_submit_host_reads(fg_vk_context *vk,fg_error *err){
    if(!vk){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA host-read barrier has no Vulkan context");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=FG_OK;
    while(status==FG_OK&&fg_vk_batch_active(vk))status=fg_vk_end(vk,err);
    if(status!=FG_OK&&fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    return status;
}

uint32_t fg_qsa_index_segment_count(uint32_t logical_context){
    if(!logical_context)return 0u;
    return logical_context/FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY+
        (logical_context%FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY!=0u);
}

uint32_t fg_qsa_index_segment_first(uint32_t logical_context,uint32_t segment){
    if(segment>=fg_qsa_index_segment_count(logical_context))return 0u;
    return segment*FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY;
}

uint32_t fg_qsa_index_segment_tokens(uint32_t logical_context,uint32_t segment){
    uint32_t first=fg_qsa_index_segment_first(logical_context,segment);
    if(!logical_context||segment>=fg_qsa_index_segment_count(logical_context))return 0u;
    uint32_t remaining=logical_context-first;
    return remaining<FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY?remaining:
        FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY;
}

uint64_t fg_qsa_index_segment_bytes(uint32_t logical_context,uint32_t segment){
    return (uint64_t)fg_qsa_index_segment_tokens(logical_context,segment)*
        FG_Q38_QSA_INDEX_KEY_BYTES;
}

uint64_t fg_qsa_record_segment_bytes(uint32_t logical_context,uint32_t segment){
    return (uint64_t)fg_qsa_index_segment_tokens(logical_context,segment)*
        FG_Q38_QSA_TOKEN_RECORD_BYTES;
}

uint64_t fg_qsa_resident_layer_bytes(uint32_t logical_context){
    return (uint64_t)logical_context*
        (FG_Q38_QSA_TOKEN_RECORD_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES);
}

bool fg_qsa_index_token_location(uint32_t logical_context,uint32_t token,
                                 uint32_t *segment,uint32_t *offset){
    if(!logical_context||token>=logical_context||!segment||!offset||
       fg_qsa_index_segment_count(logical_context)>FG_QSA_INDEX_MAX_SEGMENTS)
        return false;
    *segment=token/FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY;
    *offset=token%FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY;
    return *segment<FG_QSA_INDEX_MAX_SEGMENTS;
}

static fg_status create_index_segments(fg_qsa_session *s,fg_error *err){
    s->index_segment_count=fg_qsa_index_segment_count(s->max_context);
    if(!s->index_segment_count||s->index_segment_count>FG_QSA_INDEX_MAX_SEGMENTS){
        fg_error_set(err,FG_ERR_LIMIT,"QSA index segment geometry exceeds bounded capacity");
        return FG_ERR_LIMIT;
    }
    for(uint32_t segment=0;segment<s->index_segment_count;segment++)
        s->index_segment_tokens[segment]=
            fg_qsa_index_segment_tokens(s->max_context,segment);
    for(uint32_t slot=0;slot<s->layer_count;slot++)
        for(uint32_t segment=0;segment<s->index_segment_count;segment++){
            fg_status status=make_tensor(s,fg_qsa_index_segment_bytes(s->max_context,segment),
                                         &s->index_keys[slot][segment],err);
            if(status!=FG_OK)return status;
            uint64_t touched=0;
            status=fg_vk_tensor_residency_canary(s->index_keys[slot][segment],
                                                  &touched,err);
            if(status==FG_OK&&touched!=fg_vk_tensor_bytes(s->index_keys[slot][segment])){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "QSA index segment residency canary touched %llu of %llu bytes",
                             (unsigned long long)touched,
                             (unsigned long long)fg_vk_tensor_bytes(
                                 s->index_keys[slot][segment]));
                status=FG_ERR_MISMATCH;
            }
            if(status!=FG_OK)return status;
        }
    return FG_OK;
}

static fg_status fail_created_session(fg_qsa_session *session,const char *path,
                                      fg_status status,fg_error *err){
    fg_error original={.code=status};if(err)original=*err;
    fg_qsa_session_close(session);unlink(path);if(err)*err=original;return status;
}
static uint32_t get_u32_le(const uint8_t *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8u)|((uint32_t)p[2]<<16u)|((uint32_t)p[3]<<24u);}
static void put_u32_le(uint8_t *p,uint32_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8u);p[2]=(uint8_t)(v>>16u);p[3]=(uint8_t)(v>>24u);}

static int layer_slot(const fg_qsa_session *s,uint32_t layer){for(uint32_t i=0;i<s->layer_count;i++)if(s->layers[i]==layer)return (int)i;return -1;}
static fg_vk_tensor *layer_weight(fg_qsa_session *s,uint32_t layer,const char *suffix,fg_error *err){char name[FG_TENSOR_NAME_MAX];int n=snprintf(name,sizeof(name),"blk.%u.%s",layer,suffix);if(n<0||(uint32_t)n>=sizeof(name)){fg_error_set(err,FG_ERR_LIMIT,"QSA tensor name overflow");return NULL;}fg_vk_tensor *tensor=fg_model_tensor(s->model,name);if(!tensor)fg_error_set(err,FG_ERR_MISMATCH,"QSA owner is missing %s",name);return tensor;}

static fg_status create_reusable_views(fg_qsa_session *s,fg_error *err){
    fg_status status=fg_vk_tensor_view(s->positions,0,FG_Q38_QSA_POSITION_BYTES,&s->position_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->positions,0,FG_Q38_QSA_POSITION_BYTES,&s->token_position_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->index_query,0,512u*4u,&s->index_query_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->query,0,6144u*4u,&s->query_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->gate,0,6144u*4u,&s->gate_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->attention,0,6144u*4u,&s->attention_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->key_q8,0,FG_Q38_QSA_KEY_BYTES,&s->key_q8_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->value_q4,0,FG_Q38_QSA_VALUE_BYTES,&s->value_q4_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(s->index_key_q8,0,FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_key_q8_view,err);
    return status;
}

static fg_status create_attention_views(fg_qsa_session *s,fg_vk_tensor *scratch,
                                         uint32_t batch_size,fg_error *err){
    uint64_t offset=0;
    const uint64_t bytes[]={
        (uint64_t)batch_size*12288u*4u,
        (uint64_t)batch_size*512u*4u,
        (uint64_t)batch_size*512u*4u,
        (uint64_t)batch_size*6144u*4u,
        (uint64_t)batch_size*6144u*4u,
        (uint64_t)batch_size*512u*4u,
        (uint64_t)batch_size*512u*4u,
        (uint64_t)batch_size*128u*4u,
        (uint64_t)batch_size*512u*4u,
        (uint64_t)batch_size*FG_Q38_QSA_KEY_BYTES,
        (uint64_t)batch_size*FG_Q38_QSA_VALUE_BYTES,
        (uint64_t)batch_size*FG_Q38_QSA_INDEX_KEY_BYTES,
        (uint64_t)batch_size*6144u*4u,
        (uint64_t)batch_size*2560u*4u
    };
    fg_vk_tensor **fields[]={
        &s->raw_query_gate,&s->raw_key,&s->raw_value,&s->query,&s->gate,
        &s->key,&s->raw_index_query,&s->raw_index_key,&s->index_query,
        &s->key_q8,&s->value_q4,&s->index_key_q8,&s->attention,&s->output
    };
    for(uint32_t i=0;i<sizeof(bytes)/sizeof(bytes[0]);i++){
        fg_status status=fg_vk_tensor_view(scratch,offset,bytes[i],fields[i],err);
        if(status!=FG_OK)return status;
        offset+=bytes[i];
    }
    if(offset!=fg_qsa_attention_scratch_bytes(batch_size)){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA shared attention scratch geometry mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

static fg_status create_selection_views(fg_qsa_session *s,fg_vk_tensor *scratch,
                                         uint32_t batch_size,fg_error *err){
    uint64_t blocks=s->max_blocks,score_bytes=blocks*sizeof(uint32_t);
    uint64_t offset=fg_align_up_u64(fg_qsa_attention_scratch_bytes(batch_size),
                                    FG_ALIGNMENT);
    fg_vk_tensor **scores[]={&s->scores[0],&s->scores[1],
                             &s->ids[0],&s->ids[1]};
    for(uint32_t i=0;i<4u;i++){
        fg_status status=fg_vk_tensor_view(scratch,offset,score_bytes,scores[i],err);
        if(status!=FG_OK)return status;
        offset=fg_align_up_u64(offset+score_bytes,FG_ALIGNMENT);
    }
    return fg_vk_tensor_view(scratch,offset,
                             (uint64_t)FG_QSA_SELECTED_TOKENS*
                                 FG_Q38_QSA_TOKEN_RECORD_BYTES,
                             &s->selected_records,err);
}

static fg_status create_resident_selection_views(fg_qsa_session *s,
                                                  fg_vk_tensor *scratch,
                                                  uint32_t batch_size,
                                                  fg_error *err){
    uint64_t entries=fg_qsa_resident_candidate_entries(s->max_context,batch_size);
    uint64_t bytes=entries*sizeof(uint32_t);
    uint64_t offset=fg_align_up_u64(fg_qsa_attention_scratch_bytes(batch_size),
                                    FG_ALIGNMENT);
    fg_vk_tensor **fields[]={&s->scores[0],&s->ids[0],&s->scores[1],&s->ids[1]};
    for(uint32_t i=0;i<4u;i++){
        fg_status status=fg_vk_tensor_view(scratch,offset,bytes,fields[i],err);
        if(status!=FG_OK)return status;
        offset=fg_align_up_u64(offset+bytes,FG_ALIGNMENT);
    }
    uint64_t required=fg_align_up_u64(fg_qsa_attention_scratch_bytes(batch_size),
                                      FG_ALIGNMENT)+
        fg_qsa_resident_selection_scratch_bytes(s->max_context,batch_size);
    if(offset!=required||required>fg_vk_tensor_bytes(scratch)){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "resident QSA shared selection scratch geometry mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

static fg_status ensure_read_records(fg_qsa_session *s,fg_error *err){
    if(s->read_records)return FG_OK;
    s->read_records=malloc((uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*
                           FG_QSA_PAGE_RECORD_BYTES);
    if(!s->read_records){
        fg_error_set(err,FG_ERR_OOM,"allocate QSA page staging");
        return FG_ERR_OOM;
    }

    return FG_OK;
}

static fg_status ensure_page_cache(fg_qsa_session *s,fg_error *err){
    if(!s||!s->cache_pages||s->cache)return FG_OK;
    fg_status status=fg_qsa_page_cache_create(&s->cache,s->cache_pages,err);
    if(status==FG_OK)status=make_tensor(s,(uint64_t)s->cache_pages*
                                        FG_QSA_PAGE_RECORD_BYTES,
                                        &s->cache_records,err);
    if(status!=FG_OK){
        fg_qsa_page_cache_destroy(s->cache);s->cache=NULL;
        fg_vk_tensor_destroy(s->cache_records);s->cache_records=NULL;
    }
    return status;
}

static fg_status restore_state(fg_qsa_session *s,fg_error *err){
    uint32_t common=fg_qsa_state_layer_tokens(s->state,0);
    for(uint32_t slot=1;slot<s->layer_count;slot++)
        if(fg_qsa_state_layer_tokens(s->state,slot)!=common){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "QSA layer checkpoints stop at different token boundaries");
            return FG_ERR_MISMATCH;
        }
    uint32_t *positions=fg_vk_tensor_map(s->positions);
    uint32_t blocks=(common+3u)/4u;
    uint32_t block_ids[FG_QSA_MAX_SELECTED_BLOCKS],
        committed[FG_QSA_MAX_SELECTED_BLOCKS];
    for(uint32_t slot=0;slot<s->layer_count;slot++){
        uint8_t *resident[FG_QSA_INDEX_MAX_SEGMENTS]={0};
        for(uint32_t segment=0;segment<s->index_segment_count;segment++)
            resident[segment]=fg_vk_tensor_map(s->index_keys[slot][segment]);
        for(uint32_t first=0;first<blocks;first+=FG_QSA_MAX_SELECTED_BLOCKS){
            uint32_t count=blocks-first;
            if(count>FG_QSA_MAX_SELECTED_BLOCKS)count=FG_QSA_MAX_SELECTED_BLOCKS;
            for(uint32_t i=0;i<count;i++)block_ids[i]=first+i;
            fg_status status=fg_qsa_state_read_blocks(s->state,slot,block_ids,count,
                                                      s->read_records,committed,err);
            if(status!=FG_OK)return status;
            for(uint32_t i=0;i<count;i++){
                uint32_t expected=block_ids[i]+1u==blocks&&common%4u?
                    common%4u:4u;
                if(committed[i]!=expected){
                    fg_error_set(err,FG_ERR_MISMATCH,
                                 "QSA checkpoint page length does not match header");
                    return FG_ERR_MISMATCH;
                }
                for(uint32_t inside=0;inside<committed[i];inside++){
                    uint32_t token=block_ids[i]*4u+inside,segment,offset;
                    if(!fg_qsa_index_token_location(s->max_context,token,
                                                    &segment,&offset)){
                        fg_error_set(err,FG_ERR_LIMIT,
                                     "QSA checkpoint token has no index segment");
                        return FG_ERR_LIMIT;
                    }
                    const uint8_t *record=s->read_records+
                        ((uint64_t)i*4u+inside)*FG_Q38_QSA_TOKEN_RECORD_BYTES;
                    const uint8_t *metadata=record+FG_Q38_QSA_KEY_BYTES+
                        FG_Q38_QSA_VALUE_BYTES;
                    memcpy(resident[segment]+(uint64_t)offset*
                           FG_Q38_QSA_INDEX_KEY_BYTES,metadata,
                           FG_Q38_QSA_INDEX_KEY_BYTES);
                    for(uint32_t axis=0;axis<3u;axis++){
                        uint32_t value=get_u32_le(metadata+
                            FG_Q38_QSA_INDEX_KEY_BYTES+axis*4u);
                        if(slot&&positions[(uint64_t)token*3u+axis]!=value){
                            fg_error_set(err,FG_ERR_MISMATCH,
                                         "QSA layers disagree on persisted MRoPE positions");
                            return FG_ERR_MISMATCH;
                        }
                        positions[(uint64_t)token*3u+axis]=value;
                    }
                }
            }
        }
        if(common%4u){
            uint32_t committed_tokens=0;
            fg_status status=fg_qsa_state_read_block(s->state,slot,blocks-1u,
                                                      s->partial[slot],
                                                      &committed_tokens,err);
            if(status!=FG_OK)return status;
            if(committed_tokens!=common%4u){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA partial checkpoint length mismatch");
                return FG_ERR_MISMATCH;
            }
        }
    }
    return FG_OK;
}

fg_status fg_qsa_session_open(fg_qsa_session **out,fg_model *model,const char *path,bool create,fg_error *err){
    if(!out||!model||!path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA session arguments");return FG_ERR_ARGUMENT;}*out=NULL;const fg_manifest *manifest=fg_model_manifest(model);uint32_t rank=fg_model_rank(model);fg_qsa_session *s=calloc(1,sizeof(*s));if(!s){fg_error_set(err,FG_ERR_OOM,"allocate QSA session");return FG_ERR_OOM;}s->model=model;s->max_context=manifest->session.logical_context_tokens;s->max_blocks=(s->max_context+3u)/4u;s->max_tokens=manifest->prefill_microbatch;if(!s->max_context||s->max_context>manifest->native_context||s->max_context>manifest->max_context){fg_error_set(err,FG_ERR_MISMATCH,"manifest QSA logical context is invalid");fg_qsa_session_close(s);return FG_ERR_MISMATCH;}if(!s->max_tokens||s->max_tokens>512u){fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds QSA session limit");fg_qsa_session_close(s);return FG_ERR_MISMATCH;}
    bool coordinator=manifest->execution_mode!=FG_EXECUTION_PIPELINE&&rank==0u;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u){
        if(coordinator||manifest->layer_owner[layer]==rank)
            s->layers[s->layer_count++]=(uint8_t)layer;
    }
    uint32_t expected_layers=coordinator?FG_QSA_MAX_LAYERS:
        manifest->execution_mode==FG_EXECUTION_PIPELINE?s->layer_count:
        FG_QSA_OWNER_LAYERS;
    if(!s->layer_count||s->layer_count!=expected_layers){fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA layers, expected %u",rank,s->layer_count,expected_layers);fg_qsa_session_close(s);return FG_ERR_MISMATCH;}
    fg_status status=fg_qsa_state_open(&s->state,path,s->layers,s->layer_count,
                                       s->max_context,create,err);
    bool created_state=status==FG_OK&&create;
    if(status==FG_OK)status=make_tensor(s,(uint64_t)s->max_context*
                                         FG_Q38_QSA_POSITION_BYTES,&s->positions,err);
    if(status==FG_OK)status=create_index_segments(s,err);
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
    if(status==FG_OK)status=create_reusable_views(s,err);
    s->read_records=status==FG_OK?malloc((uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*4u*FG_Q38_QSA_TOKEN_RECORD_BYTES):NULL;if(status==FG_OK&&!s->read_records){fg_error_set(err,FG_ERR_OOM,"allocate QSA state staging");status=FG_ERR_OOM;}if(status==FG_OK&&!create)status=restore_state(s,err);if(status==FG_OK)for(uint32_t slot=0;slot<s->layer_count;slot++)s->committed[slot]=fg_qsa_state_layer_tokens(s->state,slot);if(status!=FG_OK){if(created_state)return fail_created_session(s,path,status,err);fg_qsa_session_close(s);return status;}*out=s;return FG_OK;
}

static fg_status open_decode_config(fg_qsa_session **out,fg_model *model,const char *state_path,
                                    uint32_t logical_context,uint32_t hot_tokens,
                                    uint32_t cache_pages,uint32_t batch_size,
                                    fg_vk_tensor *shared_scratch,
                                    fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                    fg_error *err){
    (void)hot_tokens;
    if(!out||!model||!logical_context||!cache_pages||!batch_size){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid decode QSA session arguments");
        return FG_ERR_ARGUMENT;
    }
    if(fg_qsa_index_segment_count(logical_context)>FG_QSA_INDEX_MAX_SEGMENTS){
        fg_error_set(err,FG_ERR_LIMIT,
                     "QSA logical context exceeds segmented index capacity");
        return FG_ERR_LIMIT;
    }
    if(cache_pages>(UINT32_C(1)<<30u)){
        fg_error_set(err,FG_ERR_LIMIT,"QSA page cache exceeds hash address space");
        return FG_ERR_LIMIT;
    }
    if(shared_scratch&&fg_vk_tensor_bytes(shared_scratch)<
       fg_qsa_attention_scratch_bytes(batch_size)){
        fg_error_set(err,FG_ERR_MISMATCH,"shared QSA attention scratch is too small");
        return FG_ERR_MISMATCH;
    }
    *out=NULL;const fg_manifest *manifest=fg_model_manifest(model);uint32_t rank=fg_model_rank(model);
    bool coordinator=manifest->execution_mode!=FG_EXECUTION_PIPELINE&&rank==0u;
    if(!state_path&&!coordinator){
        fg_error_set(err,FG_ERR_ARGUMENT,"only the coordinator may open a fileless QSA mirror");
        return FG_ERR_ARGUMENT;
    }
    fg_qsa_session *s=calloc(1,sizeof(*s));
    if(!s){fg_error_set(err,FG_ERR_OOM,"allocate QSA session");return FG_ERR_OOM;}
    s->model=model;s->max_context=logical_context;s->max_blocks=(logical_context+3u)/4u;
    s->max_tokens=batch_size;s->fetch_pages=fetch_pages;
    s->fetch_opaque=fetch_opaque;
    if(coordinator)s->locality=fg_qsa_locality_create_from_env(s->max_blocks,0u);
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(coordinator||manifest->layer_owner[layer]==rank)
            s->layers[s->layer_count++]=(uint8_t)layer;
    uint32_t expected_layers=coordinator?FG_QSA_MAX_LAYERS:
        manifest->execution_mode==FG_EXECUTION_PIPELINE?s->layer_count:
        FG_QSA_OWNER_LAYERS;
    if(!s->layer_count||s->layer_count!=expected_layers){
        fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA layers, expected %u",
                     rank,s->layer_count,expected_layers);
        fg_qsa_session_close(s);return FG_ERR_MISMATCH;
    }
    fg_status status=FG_OK;bool created_state=false;
    uint64_t index_requested=0,index_allocated=0,index_touched=0;
    if(state_path){
        status=fg_qsa_state_open(&s->state,state_path,s->layers,s->layer_count,
                                 logical_context,true,err);
        created_state=status==FG_OK;
    }
    if(status==FG_OK)status=make_tensor(s,(uint64_t)logical_context*
                                        FG_Q38_QSA_POSITION_BYTES,&s->positions,err);
    if(status==FG_OK){
        s->index_segment_count=fg_qsa_index_segment_count(logical_context);
        if(!s->index_segment_count||s->index_segment_count>FG_QSA_INDEX_MAX_SEGMENTS){
            fg_error_set(err,FG_ERR_LIMIT,
                         "QSA index segment geometry exceeds bounded capacity");
            status=FG_ERR_LIMIT;
        }
    }
    for(uint32_t segment=0;status==FG_OK&&segment<s->index_segment_count;segment++)
        s->index_segment_tokens[segment]=
            fg_qsa_index_segment_tokens(logical_context,segment);
    for(uint32_t slot=0;status==FG_OK&&slot<s->layer_count;slot++)
        for(uint32_t segment=0;status==FG_OK&&segment<s->index_segment_count;segment++){
            status=make_tensor(s,fg_qsa_index_segment_bytes(logical_context,segment),
                               &s->index_keys[slot][segment],err);
            if(status!=FG_OK)break;
            uint64_t touched=0;
            index_requested+=fg_vk_tensor_bytes(s->index_keys[slot][segment]);
            index_allocated+=fg_vk_tensor_allocation_bytes(s->index_keys[slot][segment]);
            status=fg_vk_tensor_residency_canary(s->index_keys[slot][segment],
                                                 &touched,err);
            if(status==FG_OK&&touched!=fg_vk_tensor_bytes(s->index_keys[slot][segment])){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "QSA index segment residency canary touched %llu of %llu bytes",
                             (unsigned long long)touched,
                             (unsigned long long)fg_vk_tensor_bytes(
                                 s->index_keys[slot][segment]));
                status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK)index_touched+=touched;
            if(status==FG_OK)memset(fg_vk_tensor_map(s->index_keys[slot][segment]),0,
                                    (size_t)fg_vk_tensor_bytes(s->index_keys[slot][segment]));
        }
    s->cache_pages=cache_pages;
    if(status==FG_OK&&!state_path)status=ensure_page_cache(s,err);
    if(status==FG_OK&&index_requested)
        fprintf(stderr,"[rank %u] QSA Vulkan index canary: segments=%u max_segment=%llu "
                       "requested=%llu allocated=%llu touched=%llu bytes\n",rank,
                s->layer_count*s->index_segment_count,
                (unsigned long long)fg_qsa_index_segment_bytes(logical_context,0u),
                (unsigned long long)index_requested,(unsigned long long)index_allocated,
                (unsigned long long)index_touched);
    if(status==FG_OK&&shared_scratch)
        status=create_attention_views(s,shared_scratch,batch_size,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*12288u*4u,&s->raw_query_gate,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->raw_key,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->raw_value,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->query,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->gate,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->key,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->raw_index_query,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*128u*4u,&s->raw_index_key,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*512u*4u,&s->index_query,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*FG_Q38_QSA_KEY_BYTES,&s->key_q8,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*FG_Q38_QSA_VALUE_BYTES,&s->value_q4,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*FG_Q38_QSA_INDEX_KEY_BYTES,&s->index_key_q8,err);
    if(status==FG_OK&&shared_scratch)
        status=create_selection_views(s,shared_scratch,batch_size,err);
    for(uint32_t i=0;status==FG_OK&&!shared_scratch&&i<2u;i++){
        status=make_tensor(s,(uint64_t)s->max_blocks*4u,&s->scores[i],err);
        if(status==FG_OK)status=make_tensor(s,(uint64_t)s->max_blocks*4u,&s->ids[i],err);
    }
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)FG_QSA_SELECTED_TOKENS*
                                        FG_Q38_QSA_TOKEN_RECORD_BYTES,&s->selected_records,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->attention,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*2560u*4u,&s->output,err);
    if(status==FG_OK)status=create_reusable_views(s,err);
    if(status==FG_OK&&state_path)status=ensure_read_records(s,err);
    fg_vk_memory_stats memory_stats={0};fg_vk_get_memory_stats(fg_model_vk(model),&memory_stats);
    fprintf(stderr,"[rank %u] QSA decode session: %u logical, %u cache pages, %u layers, "
                   "%.1f MiB index, %.1f MiB record cache\n",rank,logical_context,
            s->cache_pages,s->layer_count,
            (double)((uint64_t)logical_context*(FG_Q38_QSA_POSITION_BYTES+
                     s->layer_count*FG_Q38_QSA_INDEX_KEY_BYTES))/(1024.0*1024.0),
            (double)((uint64_t)s->cache_pages*FG_QSA_PAGE_RECORD_BYTES)/
                (1024.0*1024.0));
    fprintf(stderr,"[rank %u] Vulkan memory: live_requested=%llu live_allocated=%llu "
                  "peak_requested=%llu peak_allocated=%llu live_allocations=%llu\n",rank,
            (unsigned long long)memory_stats.requested_live_bytes,
            (unsigned long long)memory_stats.allocated_live_bytes,
            (unsigned long long)memory_stats.requested_peak_bytes,
            (unsigned long long)memory_stats.allocated_peak_bytes,
            (unsigned long long)memory_stats.live_allocations);
    if(status!=FG_OK){
        if(created_state)return fail_created_session(s,state_path,status,err);
        fg_qsa_session_close(s);return status;
    }
    *out=s;return FG_OK;
}

fg_status fg_qsa_session_open_decode(fg_qsa_session **out,fg_model *model,const char *state_path,
                                     uint32_t resident_tokens,uint32_t batch_size,fg_error *err){
    if(!state_path){
        fg_error_set(err,FG_ERR_ARGUMENT,"decode QSA state path is null");return FG_ERR_ARGUMENT;
    }
    return open_decode_config(out,model,state_path,resident_tokens,resident_tokens,0u,batch_size,
                              NULL,NULL,NULL,err);
}

fg_status fg_qsa_session_open_mirror(fg_qsa_session **out,fg_model *model,
                                     uint32_t logical_context,uint32_t hot_tokens,
                                     uint32_t cache_pages,uint32_t batch_size,
                                     fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                     fg_error *err){
    if(logical_context>hot_tokens&&!fetch_pages){
        fg_error_set(err,FG_ERR_ARGUMENT,"tiered QSA mirror requires a cold-page fetch callback");
        return FG_ERR_ARGUMENT;
    }
    return open_decode_config(out,model,NULL,logical_context,hot_tokens,cache_pages,batch_size,
                              NULL,fetch_pages,fetch_opaque,err);
}

fg_status fg_qsa_session_open_mirror_with_scratch(
    fg_qsa_session **out,fg_model *model,uint32_t logical_context,uint32_t hot_tokens,
    uint32_t cache_pages,uint32_t batch_size,fg_vk_tensor *scratch,
    fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,fg_error *err){
    if(!scratch){
        fg_error_set(err,FG_ERR_ARGUMENT,"shared QSA attention scratch is null");
        return FG_ERR_ARGUMENT;
    }
    if(logical_context>hot_tokens&&!fetch_pages){
        fg_error_set(err,FG_ERR_ARGUMENT,"tiered QSA mirror requires a cold-page fetch callback");
        return FG_ERR_ARGUMENT;
    }
    return open_decode_config(out,model,NULL,logical_context,hot_tokens,cache_pages,batch_size,
                              scratch,fetch_pages,fetch_opaque,err);
}

fg_status fg_qsa_session_open_resident(fg_qsa_session **out,fg_model *model,
                                      uint32_t batch_size,fg_vk_tensor *scratch,
                                      fg_error *err){
    if(!out||!model||!batch_size||!scratch){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident QSA session arguments");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    const fg_manifest *manifest=fg_model_manifest(model);
    uint32_t rank=fg_model_rank(model);
    if(manifest->execution_mode!=FG_EXECUTION_PIPELINE||
       batch_size!=manifest->prefill_microbatch){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "resident QSA requires the sealed pipeline microbatch");
        return FG_ERR_MISMATCH;
    }
    uint32_t logical_context=manifest->session.logical_context_tokens;
    uint64_t required=fg_align_up_u64(fg_qsa_attention_scratch_bytes(batch_size),
                                     FG_ALIGNMENT)+
        fg_qsa_resident_selection_scratch_bytes(logical_context,batch_size);
    if(required==UINT64_MAX||fg_vk_tensor_bytes(scratch)<required||
       fg_qsa_index_segment_count(logical_context)>FG_QSA_RESIDENT_MAX_SEGMENTS){
        fg_error_set(err,FG_ERR_LIMIT,
                     "resident QSA geometry exceeds the sealed scratch or segment limit");
        return FG_ERR_LIMIT;
    }
    fg_qsa_session *s=calloc(1,sizeof(*s));
    if(!s){
        fg_error_set(err,FG_ERR_OOM,"allocate resident QSA session");
        return FG_ERR_OOM;
    }
    s->model=model;s->resident=true;s->max_context=logical_context;
    s->max_blocks=(logical_context+FG_Q38_QSA_COMPRESS_RATIO-1u)/
        FG_Q38_QSA_COMPRESS_RATIO;
    s->max_tokens=batch_size;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(manifest->layer_owner[layer]==rank)
            s->layers[s->layer_count++]=(uint8_t)layer;
    if(!s->layer_count){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline rank %u owns no resident QSA layers",rank);
        fg_qsa_session_close(s);
        return FG_ERR_MISMATCH;
    }
    s->index_segment_count=fg_qsa_index_segment_count(logical_context);
    for(uint32_t segment=0;segment<s->index_segment_count;segment++)
        s->index_segment_tokens[segment]=
            fg_qsa_index_segment_tokens(logical_context,segment);
    fg_status status=make_tensor(s,(uint64_t)logical_context*
                                FG_Q38_QSA_POSITION_BYTES,&s->positions,err);
    uint64_t requested=fg_vk_tensor_bytes(s->positions),touched=0u;
    if(status==FG_OK){
        uint64_t bytes=0u;
        status=fg_vk_tensor_residency_canary(s->positions,&bytes,err);
        if(status==FG_OK&&bytes!=fg_vk_tensor_bytes(s->positions)){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "resident QSA position canary did not touch the segment");
            status=FG_ERR_MISMATCH;
        }
        touched+=bytes;
    }
    for(uint32_t slot=0;status==FG_OK&&slot<s->layer_count;slot++)
        for(uint32_t segment=0;status==FG_OK&&segment<s->index_segment_count;
            segment++){
            status=make_tensor(s,fg_qsa_record_segment_bytes(logical_context,segment),
                              &s->records[slot][segment],err);
            if(status==FG_OK)status=make_tensor(
                s,fg_qsa_index_segment_bytes(logical_context,segment),
                &s->index_keys[slot][segment],err);
            fg_vk_tensor *resident[]={s->records[slot][segment],
                                     s->index_keys[slot][segment]};
            for(uint32_t i=0;status==FG_OK&&i<2u;i++){
                uint64_t bytes=0u;
                requested+=fg_vk_tensor_bytes(resident[i]);
                status=fg_vk_tensor_residency_canary(resident[i],&bytes,err);
                if(status==FG_OK&&bytes!=fg_vk_tensor_bytes(resident[i])){
                    fg_error_set(err,FG_ERR_MISMATCH,
                                "resident QSA canary did not touch a complete segment");
                    status=FG_ERR_MISMATCH;
                }
                touched+=bytes;
            }
        }
    if(status==FG_OK)status=create_attention_views(s,scratch,batch_size,err);
    if(status==FG_OK)status=create_resident_selection_views(s,scratch,batch_size,err);
    if(status==FG_OK)status=create_reusable_views(s,err);
    if(status!=FG_OK){
        fg_qsa_session_close(s);
        return status;
    }
    fprintf(stderr,"[rank %u] resident QSA: layers=%u segments=%u "
                   "requested=%llu touched=%llu scratch=%llu bytes\n",
            rank,s->layer_count,s->layer_count*s->index_segment_count,
            (unsigned long long)requested,(unsigned long long)touched,
            (unsigned long long)required);
    *out=s;
    return FG_OK;
}

void fg_qsa_session_close(fg_qsa_session *s){if(!s)return;fg_qsa_locality_destroy(s->locality,"close");fg_qsa_page_cache_destroy(s->cache);free(s->read_records);fg_vk_tensor_destroy(s->index_key_q8_view);fg_vk_tensor_destroy(s->value_q4_view);fg_vk_tensor_destroy(s->key_q8_view);fg_vk_tensor_destroy(s->attention_view);fg_vk_tensor_destroy(s->gate_view);fg_vk_tensor_destroy(s->query_view);fg_vk_tensor_destroy(s->index_query_view);fg_vk_tensor_destroy(s->token_position_view);fg_vk_tensor_destroy(s->position_view);fg_vk_tensor_destroy(s->output);fg_vk_tensor_destroy(s->attention);fg_vk_tensor_destroy(s->selected_records);for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(s->ids[i]);fg_vk_tensor_destroy(s->scores[i]);}fg_vk_tensor_destroy(s->index_key_q8);fg_vk_tensor_destroy(s->value_q4);fg_vk_tensor_destroy(s->key_q8);fg_vk_tensor_destroy(s->index_query);fg_vk_tensor_destroy(s->raw_index_key);fg_vk_tensor_destroy(s->raw_index_query);fg_vk_tensor_destroy(s->key);fg_vk_tensor_destroy(s->gate);fg_vk_tensor_destroy(s->query);fg_vk_tensor_destroy(s->raw_value);fg_vk_tensor_destroy(s->raw_key);fg_vk_tensor_destroy(s->raw_query_gate);for(uint32_t i=0;i<FG_QSA_MAX_LAYERS;i++)for(uint32_t segment=0;segment<FG_QSA_INDEX_MAX_SEGMENTS;segment++){fg_vk_tensor_destroy(s->records[i][segment]);fg_vk_tensor_destroy(s->index_keys[i][segment]);}fg_vk_tensor_destroy(s->cache_records);fg_vk_tensor_destroy(s->positions);fg_qsa_state_close(s->state);free(s);}

fg_status fg_qsa_session_reset(fg_qsa_session *s,fg_error *err){if(!s){fg_error_set(err,FG_ERR_ARGUMENT,"QSA session reset is null");return FG_ERR_ARGUMENT;}fg_qsa_locality_reset(s->locality,"reset");memset(s->committed,0,sizeof(s->committed));memset(s->partial,0,sizeof(s->partial));fg_qsa_page_cache_reset(s->cache);return s->state?fg_qsa_state_reset(s->state,err):FG_OK;}

fg_status fg_qsa_session_checkpoint(fg_qsa_session *s,fg_error *err){
    if(!s){fg_error_set(err,FG_ERR_ARGUMENT,"QSA checkpoint session is null");return FG_ERR_ARGUMENT;}
    return FG_OK;
}

uint32_t fg_qsa_session_tokens(const fg_qsa_session *s,uint32_t layer){int slot=s?layer_slot(s,layer):-1;return slot<0?0:s->committed[slot];}
void fg_qsa_session_set_tokens(fg_qsa_session *s,uint32_t tokens){if(!s)return;for(uint32_t i=0;i<s->layer_count;i++){s->committed[i]=tokens;if(s->state)fg_qsa_state_set_layer_tokens(s->state,i,tokens);}}

uint64_t fg_qsa_session_host_bytes(const fg_qsa_session *s){
    if(!s)return 0;
    return (s->read_records?(uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*
           FG_QSA_PAGE_RECORD_BYTES:0u)+
           fg_qsa_page_cache_memory_bytes(s->cache);
}

static fg_status score_index_segments(fg_qsa_session *s,uint32_t slot,
                                      const fg_vk_tensor *index_query,
                                      uint32_t tokens,fg_error *err){
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_vk_tensor *key_norm=layer_weight(s,s->layers[slot],
                                        "indexer.k_norm.weight",err);
    if(!key_norm)return FG_ERR_MISMATCH;
    if(tokens<=FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY)
        return fg_vk_qsa_index_score(vk,s->scores[0],s->ids[0],index_query,
                                     s->index_keys[slot][0],key_norm,
                                     s->positions,tokens,err);
    fg_status status=FG_OK;
    for(uint32_t segment=0;status==FG_OK&&segment<s->index_segment_count;segment++){
        uint32_t first=fg_qsa_index_segment_first(s->max_context,segment);
        if(first>=tokens)break;
        uint32_t segment_tokens=tokens-first;
        if(segment_tokens>s->index_segment_tokens[segment])
            segment_tokens=s->index_segment_tokens[segment];
        uint32_t blocks=segment_tokens/FG_Q38_QSA_COMPRESS_RATIO;
        if(!blocks)continue;
        fg_vk_tensor *position_view=NULL,*score_view=NULL,*id_view=NULL;
        status=fg_vk_tensor_view(s->positions,(uint64_t)first*
                                 FG_Q38_QSA_POSITION_BYTES,
                                 (uint64_t)segment_tokens*
                                 FG_Q38_QSA_POSITION_BYTES,&position_view,err);
        if(status==FG_OK)status=fg_vk_tensor_view(s->scores[0],
            (uint64_t)(first/FG_Q38_QSA_COMPRESS_RATIO)*4u,
            (uint64_t)blocks*4u,&score_view,err);
        if(status==FG_OK)status=fg_vk_tensor_view(s->ids[0],
            (uint64_t)(first/FG_Q38_QSA_COMPRESS_RATIO)*4u,
            (uint64_t)blocks*4u,&id_view,err);
        if(status==FG_OK)status=fg_vk_qsa_index_score_segment(
            vk,score_view,id_view,index_query,s->index_keys[slot][segment],
            key_norm,position_view,segment_tokens,
            first/FG_Q38_QSA_COMPRESS_RATIO,err);
        fg_vk_tensor_destroy(id_view);
        fg_vk_tensor_destroy(score_view);
        fg_vk_tensor_destroy(position_view);
    }
    return status;
}

static fg_status select_blocks(fg_qsa_session *s,uint32_t slot,const fg_vk_tensor *index_query,
                               uint32_t tokens,uint32_t *selected,uint32_t *selected_count,
                               bool restart_batch,fg_error *err){
    uint32_t count=tokens/4u;if(!count){*selected_count=0;return FG_OK;}
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_status status=score_index_segments(s,slot,index_query,tokens,err);
    uint32_t side=0;
    while(status==FG_OK&&count>512u){
        uint32_t next=0;status=fg_vk_topk_reduce(vk,s->scores[side^1u],s->ids[side^1u],
            s->scores[side],s->ids[side],count,&next,err);count=next;side^=1u;
    }
    if(status==FG_OK){
        uint32_t final_count=0;status=fg_vk_topk_reduce(vk,s->scores[side^1u],
            s->ids[side^1u],s->scores[side],s->ids[side],count,&final_count,err);
        count=final_count;side^=1u;
    }
    bool ended=status==FG_OK&&fg_vk_batch_active(vk);
    if(ended)status=fg_vk_end(vk,err);
    if(status==FG_OK)status=fg_vk_tensor_read(s->ids[side],0,selected,
                                              (uint64_t)count*4u,err);
    if(status==FG_OK&&s->locality)
        fg_qsa_locality_record_selection(s->locality,s->layers[slot],tokens,selected,count);
    if(status==FG_OK&&ended&&restart_batch)status=fg_vk_begin(vk,err);
    if(status==FG_OK)*selected_count=count;
    return status;
}

static fg_status attend_cache(fg_qsa_session *s,uint32_t slot,uint32_t tokens,
                              const fg_vk_tensor *index_query,const fg_vk_tensor *query,
                              const fg_vk_tensor *gate,fg_vk_tensor *attention,fg_error *err){
    if(!s->cache||!s->cache_records){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA record cache is not initialized");
        return FG_ERR_MISMATCH;
    }
    uint32_t selected[FG_QSA_MAX_SELECTED_BLOCKS],selected_count=0;
    fg_vk_context *vk=fg_model_vk(s->model);
    bool resume_batch=fg_vk_batch_active(vk),restart_batch=false;
    uint32_t complete_blocks=tokens/FG_Q38_QSA_COMPRESS_RATIO;
    fg_status status=FG_OK;
    if(complete_blocks<=FG_QSA_MAX_SELECTED_BLOCKS){
        selected_count=complete_blocks;
        for(uint32_t i=0;i<selected_count;i++)selected[i]=i;
        if(s->locality)
            fg_qsa_locality_record_selection(s->locality,s->layers[slot],tokens,
                                             selected,selected_count);
    }else{
        status=select_blocks(s,slot,index_query,tokens,selected,&selected_count,false,err);
        restart_batch=resume_batch;
    }
    uint32_t missing[FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t fetch_blocks[FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t cache_slots[FG_QSA_MAX_SELECTED_BLOCKS],missing_count=0;
    for(uint32_t i=0;status==FG_OK&&i<selected_count;i++){
        if(fg_qsa_page_cache_lookup(s->cache,s->layers[slot],selected[i],
                                    &cache_slots[i])){
            if(s->locality)fg_qsa_locality_record_cache(s->locality,s->layers[slot],true);
        }else{
            if(s->locality)fg_qsa_locality_record_cache(s->locality,s->layers[slot],false);
            fprintf(stderr,"QSA_CACHE_MISS layer=%u block=%u tokens=%u cache_pages=%u\n",
                    s->layers[slot],selected[i],tokens,s->cache_pages);
            missing[missing_count++]=selected[i];
        }
    }
    if(status==FG_OK&&missing_count)status=ensure_read_records(s,err);
    if(status==FG_OK&&missing_count){
        uint32_t fetch_count=0;
        status=fg_qsa_page_cache_plan_fetch(s->cache,s->layers[slot],missing,missing_count,
            complete_blocks,UINT32_MAX,fetch_blocks,FG_QSA_MAX_SELECTED_BLOCKS,
            &fetch_count,err);
        if(status==FG_OK&&!s->fetch_pages){
            fg_error_set(err,FG_ERR_UNAVAILABLE,"QSA cold page miss has no owner fetch service");
            status=FG_ERR_UNAVAILABLE;
        }else if(status==FG_OK)
            status=s->fetch_pages(s->fetch_opaque,s->layers[slot],fetch_blocks,fetch_count,
                                  s->read_records,err);
        for(uint32_t i=0;status==FG_OK&&i<fetch_count;i++){
            const uint8_t *page=s->read_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES;
            uint32_t cache_slot=0;bool hit=false;
            status=fg_qsa_page_cache_acquire(s->cache,s->layers[slot],fetch_blocks[i],
                                             &cache_slot,&hit,err);
            if(status==FG_OK)status=fg_vk_tensor_write(
                s->cache_records,(uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,
                page,FG_QSA_PAGE_RECORD_BYTES,err);
        }
    }
    for(uint32_t i=0;status==FG_OK&&i<selected_count;i++)
        if(!fg_qsa_page_cache_lookup(s->cache,s->layers[slot],selected[i],
                                     &cache_slots[i])){
            fg_error_set(err,FG_ERR_MISMATCH,"selected QSA page was not cached");
            status=FG_ERR_MISMATCH;
        }
    uint32_t tail=tokens%FG_Q38_QSA_COMPRESS_RATIO;
    uint32_t tail_start=0;
    if(status==FG_OK&&tail){
        uint32_t tail_slot=0;
        if(!fg_qsa_page_cache_lookup(s->cache,s->layers[slot],
                                     tokens/FG_Q38_QSA_COMPRESS_RATIO,&tail_slot)){
            fg_error_set(err,FG_ERR_MISMATCH,"partial QSA page was not cached");
            status=FG_ERR_MISMATCH;
        }
        tail_start=tail_slot*FG_Q38_QSA_COMPRESS_RATIO;
    }
    if(status==FG_OK&&selected_count)status=fg_vk_tensor_write(
        s->ids[0],0,cache_slots,(uint64_t)selected_count*sizeof(*cache_slots),err);
    if(status==FG_OK&&restart_batch)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_qsa_record_gather(
        vk,s->selected_records,s->cache_records,s->ids[0],0u,
        s->cache_pages*FG_Q38_QSA_COMPRESS_RATIO,selected_count,tail_start,tail,err);
    uint32_t selected_tokens=selected_count*FG_Q38_QSA_COMPRESS_RATIO+tail;
    if(status==FG_OK)status=fg_vk_qsa_attention(
        vk,attention,s->selected_records,query,gate,selected_tokens,err);
    return status;
}

static fg_status commit_and_attend_cache(fg_qsa_session *s,uint32_t slot,uint32_t token,const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,const fg_vk_tensor *index_query,const fg_vk_tensor *query,const fg_vk_tensor *gate,fg_vk_tensor *attention,fg_error *err){
    if(token!=s->committed[slot]||token>=s->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA cache token position does not match committed state");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(s->model);
    uint32_t segment=0,index_token=0;
    if(!fg_qsa_index_token_location(s->max_context,token,&segment,&index_token)){
        fg_error_set(err,FG_ERR_LIMIT,"QSA hot token has no index segment");
        return FG_ERR_LIMIT;
    }
    uint32_t cache_slot=0;bool hit=false;
    fg_status status=fg_qsa_page_cache_acquire(
        s->cache,s->layers[slot],token/FG_Q38_QSA_COMPRESS_RATIO,
        &cache_slot,&hit,err);
    if(status==FG_OK)status=fg_qsa_page_cache_pin(
        s->cache,s->layers[slot],token/FG_Q38_QSA_COMPRESS_RATIO,err);
    if(status==FG_OK)status=fg_vk_qsa_record_commit_segmented(
        vk,s->cache_records,s->index_keys[slot][segment],key_q8,value_q8,
        index_key_q8,position,0u,token,index_token,
        s->index_segment_tokens[segment],
        cache_slot*FG_Q38_QSA_COMPRESS_RATIO+
            token%FG_Q38_QSA_COMPRESS_RATIO,
        s->cache_pages*FG_Q38_QSA_COMPRESS_RATIO,err);
    uint32_t tokens=token+1u;if(status==FG_OK)s->committed[slot]=tokens;
    if(status==FG_OK)status=attend_cache(s,slot,tokens,index_query,query,gate,
                                        attention,err);
    return status;
}

static fg_status commit_and_attend(fg_qsa_session *s,uint32_t slot,uint32_t token,const uint32_t position[3],const uint8_t *key,const uint8_t *value,const uint8_t *index_key,const fg_vk_tensor *index_query,const fg_vk_tensor *query,const fg_vk_tensor *gate,fg_vk_tensor *attention,fg_error *err){
    if(token!=fg_qsa_state_layer_tokens(s->state,slot)||token>=s->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA token position does not match committed state");return FG_ERR_MISMATCH;}
    uint32_t inside=token%FG_Q38_QSA_COMPRESS_RATIO;if(!inside)memset(s->partial[slot],0,sizeof(s->partial[slot]));uint8_t *record=s->partial[slot]+(uint64_t)inside*FG_Q38_QSA_TOKEN_RECORD_BYTES;memcpy(record,key,FG_Q38_QSA_KEY_BYTES);memcpy(record+FG_Q38_QSA_KEY_BYTES,value,FG_Q38_QSA_VALUE_BYTES);memcpy(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,index_key,FG_Q38_QSA_INDEX_KEY_BYTES);for(uint32_t axis=0;axis<3u;axis++)put_u32_le(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES+axis*4u,position[axis]);
    fg_status status=fg_qsa_state_write_block(s->state,slot,token/FG_Q38_QSA_COMPRESS_RATIO,s->partial[slot],inside+1u,err);if(status!=FG_OK)return status;s->committed[slot]=token+1u;uint32_t segment=0,offset=0;if(!fg_qsa_index_token_location(s->max_context,token,&segment,&offset)){fg_error_set(err,FG_ERR_LIMIT,"QSA token has no index segment");return FG_ERR_LIMIT;}memcpy((uint8_t *)fg_vk_tensor_map(s->index_keys[slot][segment])+(uint64_t)offset*FG_Q38_QSA_INDEX_KEY_BYTES,index_key,FG_Q38_QSA_INDEX_KEY_BYTES);
    uint32_t selected_blocks[FG_QSA_MAX_SELECTED_BLOCKS],block_count=0,tokens=token+1u;status=select_blocks(s,slot,index_query,tokens,selected_blocks,&block_count,true,err);uint32_t selected_tokens=0;if(status==FG_OK&&block_count){uint32_t committed[FG_QSA_MAX_SELECTED_BLOCKS];status=fg_qsa_state_read_blocks(s->state,slot,selected_blocks,block_count,s->read_records,committed,err);for(uint32_t i=0;status==FG_OK&&i<block_count;i++){if(committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){fg_error_set(err,FG_ERR_MISMATCH,"selected QSA block is not complete");status=FG_ERR_MISMATCH;break;}memcpy((uint8_t *)fg_vk_tensor_map(s->selected_records)+(uint64_t)selected_tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,s->read_records+(uint64_t)i*FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES,(uint64_t)FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES);selected_tokens+=FG_Q38_QSA_COMPRESS_RATIO;}}
    uint32_t tail=tokens%FG_Q38_QSA_COMPRESS_RATIO;if(status==FG_OK&&tail){memcpy((uint8_t *)fg_vk_tensor_map(s->selected_records)+(uint64_t)selected_tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,s->partial[slot],(uint64_t)tail*FG_Q38_QSA_TOKEN_RECORD_BYTES);selected_tokens+=tail;}if(status==FG_OK)status=fg_vk_qsa_attention(fg_model_vk(s->model),attention,s->selected_records,query,gate,selected_tokens,err);return status;
}

static fg_status resident_prefill(fg_qsa_session *s,uint32_t layer,uint32_t slot,
                                  uint32_t first_token,const uint32_t *positions,
                                  uint32_t token_count,const fg_vk_tensor *hidden,
                                  bool submit,fg_vk_tensor **output,
                                  fg_error *err){
    if(first_token!=s->committed[slot]||
       first_token>s->max_context-token_count){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "resident QSA range does not start at committed state");
        return FG_ERR_MISMATCH;
    }
    fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err);
    fg_vk_tensor *kw=layer_weight(s,layer,"attn_k.weight",err);
    fg_vk_tensor *vw=layer_weight(s,layer,"attn_v.weight",err);
    fg_vk_tensor *qn=layer_weight(s,layer,"attn_q_norm.weight",err);
    fg_vk_tensor *kn=layer_weight(s,layer,"attn_k_norm.weight",err);
    fg_vk_tensor *ow=layer_weight(s,layer,"attn_output.weight",err);
    fg_vk_tensor *iqw=layer_weight(s,layer,"indexer.q_proj.weight",err);
    fg_vk_tensor *ikw=layer_weight(s,layer,"indexer.k_proj.weight",err);
    fg_vk_tensor *iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);
    fg_vk_tensor *ikn=layer_weight(s,layer,"indexer.k_norm.weight",err);
    if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn||!ikn)
        return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_status status=FG_OK;
    if(submit)status=fg_qsa_submit_host_reads(vk,err);
    else if(!fg_vk_batch_active(vk)){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "resident pipeline QSA requires an active stage batch");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_vk_tensor_write(
        s->positions,(uint64_t)first_token*FG_Q38_QSA_POSITION_BYTES,
        positions,(uint64_t)token_count*FG_Q38_QSA_POSITION_BYTES,err);
    if(status==FG_OK)status=fg_vk_tensor_view_rebind(
        s->position_view,s->positions,
        (uint64_t)first_token*FG_Q38_QSA_POSITION_BYTES,
        (uint64_t)token_count*FG_Q38_QSA_POSITION_BYTES,err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(
        vk,s->raw_query_gate,qw,hidden,FG_HIDDEN_SIZE,12288u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(
        vk,s->raw_key,kw,hidden,FG_HIDDEN_SIZE,512u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(
        vk,s->raw_value,vw,hidden,FG_HIDDEN_SIZE,512u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare_prefill(
        vk,s->query,s->gate,s->key,s->raw_query_gate,s->raw_key,qn,kn,
        s->position_view,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(
        vk,s->raw_index_query,iqw,hidden,FG_HIDDEN_SIZE,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(
        vk,s->raw_index_key,ikw,hidden,FG_HIDDEN_SIZE,128u,token_count,err);
    if(status==FG_OK)status=fg_vk_qsa_index_prepare_prefill(
        vk,s->index_query,s->raw_index_query,iqn,s->position_view,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(
        vk,s->key_q8,s->key,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(
        vk,s->value_q4,s->raw_value,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(
        vk,s->index_key_q8,s->raw_index_key,128u,token_count,err);
    fg_vk_tensor *record_1=s->index_segment_count>1u?s->records[slot][1]:NULL;
    fg_vk_tensor *index_1=s->index_segment_count>1u?s->index_keys[slot][1]:NULL;
    if(status==FG_OK)status=fg_vk_qsa_resident_record_commit(
        vk,s->records[slot][0],record_1,s->index_keys[slot][0],index_1,
        s->key_q8,s->value_q4,s->index_key_q8,s->position_view,first_token,
        token_count,s->max_context,FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,err);
    uint32_t final_side=0u;
    if(status==FG_OK)status=fg_vk_qsa_resident_select(
        vk,s->scores[0],s->ids[0],s->scores[1],s->ids[1],s->index_query,
        s->index_keys[slot][0],index_1,ikn,s->positions,first_token,token_count,
        s->max_context,FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,&final_side,err);
    if(status==FG_OK)status=fg_vk_qsa_resident_attention(
        vk,s->attention,s->records[slot][0],record_1,s->ids[final_side],
        s->query,s->gate,first_token,token_count,s->max_context,
        FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,FG_QSA_TOPK_CANDIDATES,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(
        vk,s->output,ow,s->attention,6144u,FG_HIDDEN_SIZE,token_count,1.0f,err);
    if(status==FG_OK)status=submit?fg_qsa_submit_host_reads(vk,err):
                                      fg_vk_end(vk,err);
    else if(fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    if(status==FG_OK){
        s->committed[slot]=first_token+token_count;
        *output=s->output;
    }
    return status;
}

fg_status fg_qsa_session_decode(fg_qsa_session *s,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;if(!s||signed_slot<0||!position||!hidden||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA decode arguments");return FG_ERR_ARGUMENT;}uint32_t slot=(uint32_t)signed_slot;if(token!=s->committed[slot]||token>=s->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA token position does not match committed state");return FG_ERR_MISMATCH;}fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err),*kw=layer_weight(s,layer,"attn_k.weight",err),*vw=layer_weight(s,layer,"attn_v.weight",err),*qn=layer_weight(s,layer,"attn_q_norm.weight",err),*kn=layer_weight(s,layer,"attn_k_norm.weight",err),*ow=layer_weight(s,layer,"attn_output.weight",err),*iqw=layer_weight(s,layer,"indexer.q_proj.weight",err),*ikw=layer_weight(s,layer,"indexer.k_proj.weight",err),*iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn)return FG_ERR_MISMATCH;
    if(s->resident)
        return resident_prefill(s,layer,slot,token,position,1u,hidden,true,
                                output,err);
    fg_vk_context *vk=fg_model_vk(s->model);fg_status status=fg_qsa_submit_host_reads(vk,err);
    uint32_t *resident_positions=status==FG_OK?fg_vk_tensor_map(s->positions):NULL;if(status==FG_OK&&slot==0)memcpy(resident_positions+(uint64_t)token*3u,position,12u);else if(status==FG_OK&&memcmp(resident_positions+(uint64_t)token*3u,position,12u)!=0){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers received inconsistent MRoPE positions");return FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->position_view,s->positions,(uint64_t)token*12u,12u,err);if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_projection",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_query_gate,qw,hidden,2560u,12288u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_key,kw,hidden,2560u,512u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_value,vw,hidden,2560u,512u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare(vk,s->query,s->gate,s->key,s->raw_query_gate,s->raw_key,qn,kn,s->position_view,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_query,iqw,hidden,2560u,512u,1u,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_key,ikw,hidden,2560u,128u,1u,err);
    if(status==FG_OK)status=fg_vk_qsa_index_prepare(vk,s->index_query,s->raw_index_query,iqn,s->position_view,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->key_q8,s->key,512u,1u,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->value_q4,s->raw_value,512u,1u,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->index_key_q8,s->raw_index_key,128u,1u,err);
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status!=FG_OK)return status;
    if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_state_attention",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&s->cache)status=commit_and_attend_cache(s,slot,token,s->key_q8,s->value_q4,s->index_key_q8,s->position_view,s->index_query,s->query,s->gate,s->attention,err);
    else if(status==FG_OK)status=commit_and_attend(s,slot,token,position,fg_vk_tensor_map(s->key_q8),fg_vk_tensor_map(s->value_q4),fg_vk_tensor_map(s->index_key_q8),s->index_query,s->query,s->gate,s->attention,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_output",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->output,ow,s->attention,6144u,2560u,1u,1.0f,err);
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status==FG_OK&&token<30u){const float *ap=fg_vk_tensor_map(s->attention),*op=fg_vk_tensor_map(s->output);double a2=0.0,o2=0.0;for(uint32_t i=0;i<6144u;i++)a2+=(double)ap[i]*ap[i];for(uint32_t i=0;i<2560u;i++)o2+=(double)op[i]*op[i];fprintf(stderr,"qsa[%u] t=%u selected=%u attn_rms=%.6f out_rms=%.6f\n",layer,token,token+1u,sqrt(a2/6144.0),sqrt(o2/2560.0));}
    if(status==FG_OK){*output=s->output;}return status;
}

fg_status fg_qsa_session_decode_pipeline(fg_qsa_session *s,uint32_t layer,
                                         uint32_t token,
                                         const uint32_t position[3],
                                         const fg_vk_tensor *hidden,
                                         fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;
    if(!s||signed_slot<0||!position||!hidden||!output||!s->resident){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline QSA decode requires a resident owned layer");
        return FG_ERR_MISMATCH;
    }
    uint32_t slot=(uint32_t)signed_slot;
    return resident_prefill(s,layer,slot,token,position,1u,hidden,false,
                            output,err);
}

fg_status fg_qsa_session_prefill(fg_qsa_session *s,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint32_t token_count,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;if(!s||signed_slot<0||!positions||!hidden||!output||!token_count||token_count>s->max_tokens||token_count>s->max_context||first_token>s->max_context-token_count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prefill arguments");return FG_ERR_ARGUMENT;}uint32_t slot=(uint32_t)signed_slot;if(first_token!=s->committed[slot]){fg_error_set(err,FG_ERR_MISMATCH,"QSA prefill range does not start at committed state");return FG_ERR_MISMATCH;}
    if(s->resident)
        return resident_prefill(s,layer,slot,first_token,positions,token_count,
                                hidden,true,output,err);
    fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err),*kw=layer_weight(s,layer,"attn_k.weight",err),*vw=layer_weight(s,layer,"attn_v.weight",err),*qn=layer_weight(s,layer,"attn_q_norm.weight",err),*kn=layer_weight(s,layer,"attn_k_norm.weight",err),*ow=layer_weight(s,layer,"attn_output.weight",err),*iqw=layer_weight(s,layer,"indexer.q_proj.weight",err),*ikw=layer_weight(s,layer,"indexer.k_proj.weight",err),*iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(s->model);fg_status status=fg_qsa_submit_host_reads(vk,err);
    uint32_t *resident_positions=status==FG_OK?fg_vk_tensor_map(s->positions):NULL;for(uint32_t i=0;status==FG_OK&&i<token_count;i++){uint32_t token=first_token+i;uint32_t *resident=resident_positions+(uint64_t)token*3u;const uint32_t *position=positions+(uint64_t)i*3u;if(slot==0u)memcpy(resident,position,FG_Q38_QSA_POSITION_BYTES);else if(memcmp(resident,position,FG_Q38_QSA_POSITION_BYTES)!=0){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers received inconsistent MRoPE positions");return FG_ERR_MISMATCH;}}
    if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->position_view,s->positions,(uint64_t)first_token*FG_Q38_QSA_POSITION_BYTES,(uint64_t)token_count*FG_Q38_QSA_POSITION_BYTES,err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_query_gate,qw,hidden,FG_HIDDEN_SIZE,12288u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_key,kw,hidden,FG_HIDDEN_SIZE,512u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->raw_value,vw,hidden,FG_HIDDEN_SIZE,512u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_qsa_prepare_prefill(vk,s->query,s->gate,s->key,s->raw_query_gate,s->raw_key,qn,kn,s->position_view,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_query,iqw,hidden,FG_HIDDEN_SIZE,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_bf16_f32(vk,s->raw_index_key,ikw,hidden,FG_HIDDEN_SIZE,128u,token_count,err);
    if(status==FG_OK)status=fg_vk_qsa_index_prepare_prefill(vk,s->index_query,s->raw_index_query,iqn,s->position_view,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->key_q8,s->key,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->value_q4,s->raw_value,512u,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_0(vk,s->index_key_q8,s->raw_index_key,128u,token_count,err);
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status!=FG_OK)return status;
    const uint8_t *keys=s->cache?NULL:fg_vk_tensor_map(s->key_q8),*values=s->cache?NULL:fg_vk_tensor_map(s->value_q4),*index_keys=s->cache?NULL:fg_vk_tensor_map(s->index_key_q8);
    status=fg_vk_begin(vk,err);
    for(uint32_t i=0;status==FG_OK&&i<token_count;i++){
        status=fg_vk_tensor_view_rebind(s->index_query_view,s->index_query,(uint64_t)i*512u*4u,512u*4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->query_view,s->query,(uint64_t)i*6144u*4u,6144u*4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->gate_view,s->gate,(uint64_t)i*6144u*4u,6144u*4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->attention_view,s->attention,(uint64_t)i*6144u*4u,6144u*4u,err);
        if(status==FG_OK&&s->cache)status=fg_vk_tensor_view_rebind(s->key_q8_view,s->key_q8,(uint64_t)i*FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_KEY_BYTES,err);
        if(status==FG_OK&&s->cache)status=fg_vk_tensor_view_rebind(s->value_q4_view,s->value_q4,(uint64_t)i*FG_Q38_QSA_VALUE_BYTES,FG_Q38_QSA_VALUE_BYTES,err);
        if(status==FG_OK&&s->cache)status=fg_vk_tensor_view_rebind(s->index_key_q8_view,s->index_key_q8,(uint64_t)i*FG_Q38_QSA_INDEX_KEY_BYTES,FG_Q38_QSA_INDEX_KEY_BYTES,err);
        if(status==FG_OK&&s->cache)status=fg_vk_tensor_view_rebind(s->token_position_view,s->positions,(uint64_t)(first_token+i)*FG_Q38_QSA_POSITION_BYTES,FG_Q38_QSA_POSITION_BYTES,err);
        if(status==FG_OK&&s->cache)status=commit_and_attend_cache(s,slot,first_token+i,s->key_q8_view,s->value_q4_view,s->index_key_q8_view,s->token_position_view,s->index_query_view,s->query_view,s->gate_view,s->attention_view,err);
        else if(status==FG_OK)status=commit_and_attend(s,slot,first_token+i,positions+(uint64_t)i*3u,keys+(uint64_t)i*FG_Q38_QSA_KEY_BYTES,values+(uint64_t)i*FG_Q38_QSA_VALUE_BYTES,index_keys+(uint64_t)i*FG_Q38_QSA_INDEX_KEY_BYTES,s->index_query_view,s->query_view,s->gate_view,s->attention_view,err);
    }
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->output,ow,s->attention,6144u,FG_HIDDEN_SIZE,token_count,1.0f,err);
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status==FG_OK)*output=s->output;
    return status;
}

fg_status fg_qsa_session_page_records(const fg_qsa_session *s,uint32_t layer,
                                      uint32_t block,const uint8_t **records,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;
    if(!s||signed_slot<0||!records||!s->cache_records||!s->cache){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA complete-page lookup");
        return FG_ERR_ARGUMENT;
    }
    if(fg_vk_batch_active(fg_model_vk(s->model))){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA page lookup requires a completed Vulkan batch");
        return FG_ERR_ARGUMENT;
    }
    uint32_t slot=(uint32_t)signed_slot;
    uint64_t first=(uint64_t)block*FG_Q38_QSA_COMPRESS_RATIO;
    uint64_t end=first+FG_Q38_QSA_COMPRESS_RATIO;
    uint32_t committed=s->committed[slot];
    if(end>committed){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page is incomplete");
        return FG_ERR_MISMATCH;
    }
    uint32_t cache_slot=0;
    if(!fg_qsa_page_cache_lookup(s->cache,s->layers[slot],block,&cache_slot)){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page is not resident in the record cache");
        return FG_ERR_MISMATCH;
    }
    *records=(const uint8_t *)fg_vk_tensor_map(s->cache_records)+
        (uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES;
    return FG_OK;
}

void fg_qsa_session_page_published(fg_qsa_session *s,uint32_t layer,uint32_t block){
    if(s&&s->cache)fg_qsa_page_cache_unpin(s->cache,layer,block);
}
