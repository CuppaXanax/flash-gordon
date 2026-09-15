#include "fg_qsa.h"
#include "fg_qsa_cache.h"
#include "fg_qsa_locality.h"
#include "fg_qsa_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#define FG_QSA_OWNER_LAYERS 6u
#define FG_QSA_MAX_LAYERS 12u

static double qsa_now_ms(void){struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return (double)value.tv_sec*1000.0+(double)value.tv_nsec/1000000.0;}
static bool qsa_trace_enabled(void){const char *value=getenv("FG_FRAME_TRACE");return value&&*value&&strcmp(value,"0")!=0;}
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
    fg_vk_tensor *slot_table,*select_resolved,*select_flags;
    fg_vk_tensor *tile_scores[FG_QSA_PREFILL_QUERY_TILE][2];
    fg_vk_tensor *tile_ids[FG_QSA_PREFILL_QUERY_TILE][2];
    fg_vk_tensor *tile_records[FG_QSA_PREFILL_QUERY_TILE];
    fg_vk_tensor *attn_partials;
    /* Batched prefill selection and attention scratch. */
    fg_vk_tensor *sel_scores[2],*sel_ids[2],*sel_result_ids;
    fg_vk_tensor *batch_records,*batch_partials,*batch_slots,*batch_counts;
    uint32_t *select_ids,select_chunk;
    fg_vk_tensor *position_view,*token_position_view,*index_query_view,*query_view,*gate_view;
    fg_vk_tensor *attention_view,*key_q8_view,*value_q4_view,*index_key_q8_view;
    uint8_t *read_records,*position_written;
    uint32_t state_committed[FG_QSA_MAX_SELECTED_BLOCKS];
    fg_qsa_page_cache *cache;
    fg_qsa_page_fetch_fn fetch_pages;
    void *fetch_opaque;
    fg_qsa_locality *locality;
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

/* Decode-only flush: a cache-backed selection drains at its own fence, so the
 * projection and commit recordings can stay queued behind the previous layer
 * instead of draining the block at every QSA layer. */
static fg_status qsa_decode_flush(fg_vk_context *vk,bool pipelined,fg_error *err){
    if(!fg_vk_batch_active(vk))return FG_OK;
    return pipelined?fg_vk_flush(vk,err):fg_qsa_submit_host_reads(vk,err);
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

/* Index segments are lazy after the first one: segment 0 is materialized at
 * open, and every later segment only allocates the first time the committed
 * context reaches it.  The coordinator mirror therefore carries one segment
 * per layer instead of the full logical-context index. */
static fg_status ensure_index_segment(fg_qsa_session *s,uint32_t slot,
                                      uint32_t segment,fg_error *err){
    if(!s||slot>=s->layer_count||segment>=s->index_segment_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA index segment request");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor **tensor=&s->index_keys[slot][segment];
    if(*tensor)return FG_OK;
    fg_status status=make_tensor(s,fg_qsa_index_segment_bytes(s->max_context,segment),
                                 tensor,err);
    if(status!=FG_OK)return status;
    uint64_t bytes=fg_vk_tensor_bytes(*tensor),touched=0;
    /* The canary is a host-side probe: it cannot run while a Vulkan batch is
     * recording, so batch entry points pre-ensure their segments.  A late
     * allocation under an active batch is still zeroed and usable. */
    if(!fg_vk_batch_active(fg_model_vk(s->model))){
        status=fg_vk_tensor_residency_canary(*tensor,&touched,err);
        if(status==FG_OK&&touched!=bytes){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "QSA index segment residency canary touched %llu of %llu bytes",
                         (unsigned long long)touched,(unsigned long long)bytes);
            status=FG_ERR_MISMATCH;
        }
    }
    if(status==FG_OK)memset(fg_vk_tensor_map(*tensor),0,(size_t)bytes);
    if(status!=FG_OK){fg_vk_tensor_destroy(*tensor);*tensor=NULL;}
    return status;
}

static fg_status ensure_index_segments_for_range(fg_qsa_session *s,uint32_t slot,
                                                 uint32_t first_token,
                                                 uint32_t token_count,fg_error *err){
    uint32_t last_segment=
        (first_token+token_count-1u)/FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY;
    for(uint32_t segment=0;segment<=last_segment;segment++){
        fg_status status=ensure_index_segment(s,slot,segment,err);
        if(status!=FG_OK)return status;
    }
    return FG_OK;
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
    for(uint32_t slot=0;slot<s->layer_count;slot++){
        fg_status status=ensure_index_segment(s,slot,0u,err);
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
    uint64_t queries=fg_qsa_query_tile_size(batch_size);
    uint64_t blocks=s->max_blocks,score_bytes=queries*blocks*sizeof(uint32_t);
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
                             queries*FG_QSA_SELECTED_TOKENS*
                                 FG_Q38_QSA_TOKEN_RECORD_BYTES,
                             &s->selected_records,err);
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

/* A state-backed owner serves its own cold pages from its state file. The
 * page cache is a compute shortcut, not the authority: every gather miss is
 * satisfied here after the block has been persisted. */
static fg_status state_fetch_pages(void *opaque,uint32_t layer,
                                   const uint32_t *blocks,uint32_t block_count,
                                   uint8_t *records,fg_error *err){
    fg_qsa_session *s=opaque;
    int signed_slot=s?layer_slot(s,layer):-1;
    if(!s||signed_slot<0||!s->state){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA state fetch is not on a state-backed session");
        return FG_ERR_MISMATCH;
    }
    fg_status status=fg_qsa_state_read_blocks(s->state,(uint32_t)signed_slot,blocks,
        block_count,records,s->state_committed,err);
    for(uint32_t i=0;status==FG_OK&&i<block_count;i++)
        if(s->state_committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA state fetch returned an incomplete page");
            status=FG_ERR_MISMATCH;
        }
    return status;
}

static void qsa_slot_store(fg_qsa_session *s,uint32_t layer,uint32_t block,uint32_t slot){
    if(!s||!s->slot_table)return;
    int index=layer_slot(s,layer);
    if(index<0||block>=s->max_blocks)return;
    ((uint32_t *)fg_vk_tensor_map(s->slot_table))[(uint64_t)index*
        s->max_blocks+block]=slot;
}
static void qsa_slot_evict(void *opaque,uint32_t layer,uint32_t block){
    qsa_slot_store(opaque,layer,block,UINT32_MAX);
}
static fg_status qsa_cache_acquire(fg_qsa_session *s,uint32_t layer,uint32_t block,
                                   uint32_t *slot,bool *hit,fg_error *err){
    fg_status status=fg_qsa_page_cache_acquire(s->cache,layer,block,slot,hit,err);
    if(status==FG_OK)qsa_slot_store(s,layer,block,*slot);
    return status;
}
static fg_status qsa_cache_acquire_soft(fg_qsa_session *s,uint32_t layer,uint32_t block,
                                        uint32_t *slot,bool *hit,fg_error *err){
    fg_status status=fg_qsa_page_cache_acquire_soft(s->cache,layer,block,slot,hit,err);
    if(status==FG_OK&&*slot!=UINT32_MAX)qsa_slot_store(s,layer,block,*slot);
    return status;
}
static void qsa_cache_reset(fg_qsa_session *s){
    fg_qsa_page_cache_reset(s->cache);
    if(s->slot_table)
        memset(fg_vk_tensor_map(s->slot_table),0xff,
               (size_t)fg_vk_tensor_bytes(s->slot_table));
}
static fg_status ensure_select_resolve(fg_qsa_session *s,fg_error *err){
    if(!s->slot_table){
        fg_status status=make_tensor(s,(uint64_t)s->layer_count*s->max_blocks*4u,
                                     &s->slot_table,err);
        if(status!=FG_OK)return status;
        memset(fg_vk_tensor_map(s->slot_table),0xff,
               (size_t)fg_vk_tensor_bytes(s->slot_table));
    }
    if(!s->select_resolved){
        fg_status status=make_tensor(s,512u*4u,&s->select_resolved,err);
        if(status!=FG_OK)return status;
    }
    if(!s->select_flags){
        fg_status status=make_tensor(s,2u*4u,&s->select_flags,err);
        if(status!=FG_OK)return status;
    }
    if(s->cache)fg_qsa_page_cache_set_evict_hook(s->cache,qsa_slot_evict,s);
    return FG_OK;
}
static fg_status ensure_page_cache(fg_qsa_session *s,fg_error *err){
    if(!s||!s->cache_pages||s->cache)return FG_OK;
    fg_status status=fg_qsa_page_cache_create(&s->cache,s->cache_pages,err);
    if(status==FG_OK)status=make_tensor(s,(uint64_t)s->cache_pages*
                                        FG_QSA_PAGE_RECORD_BYTES,
                                        &s->cache_records,err);
    if(status==FG_OK)status=ensure_select_resolve(s,err);
    if(status!=FG_OK){
        fg_qsa_page_cache_destroy(s->cache);s->cache=NULL;
        fg_vk_tensor_destroy(s->cache_records);s->cache_records=NULL;
        fg_vk_tensor_destroy(s->slot_table);s->slot_table=NULL;
        fg_vk_tensor_destroy(s->select_resolved);s->select_resolved=NULL;
        fg_vk_tensor_destroy(s->select_flags);s->select_flags=NULL;
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
        uint32_t needed_segments=common?
            (common-1u)/FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY+1u:0u;
        for(uint32_t segment=0;segment<needed_segments;segment++){
            fg_status ensure=ensure_index_segment(s,slot,segment,err);
            if(ensure!=FG_OK)return ensure;
            resident[segment]=fg_vk_tensor_map(s->index_keys[slot][segment]);
        }
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
    bool coordinator=rank==0u;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u){
        if(coordinator||manifest->layer_owner[layer]==rank)
            s->layers[s->layer_count++]=(uint8_t)layer;
    }
    uint32_t expected_layers=coordinator?FG_QSA_MAX_LAYERS:
                FG_QSA_OWNER_LAYERS;
    if(!s->layer_count||s->layer_count>expected_layers){fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA layers, expected %u",rank,s->layer_count,expected_layers);fg_qsa_session_close(s);return FG_ERR_MISMATCH;}
    fg_status status=fg_qsa_state_open(&s->state,path,s->layers,s->layer_count,
                                       s->max_context,create,err);
    bool created_state=status==FG_OK&&create;
    if(status==FG_OK)status=make_tensor(s,(uint64_t)s->max_context*
                                         FG_Q38_QSA_POSITION_BYTES,&s->positions,err);
    if(status==FG_OK){s->position_written=calloc(1,s->max_context);
        if(!s->position_written){fg_error_set(err,FG_ERR_OOM,"allocate QSA position map");status=FG_ERR_OOM;}}
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

static fg_status create_tile_views(fg_qsa_session *s,uint32_t batch_size,fg_error *err){
    fg_status status=FG_OK;
    for(uint32_t q=0;status==FG_OK&&q<fg_qsa_query_tile_size(batch_size);q++){
        for(uint32_t side=0;status==FG_OK&&side<2u;side++){
            uint64_t bytes=(uint64_t)s->max_blocks*4u;
            status=fg_vk_tensor_view(s->scores[side],q*bytes,bytes,&s->tile_scores[q][side],err);
            if(status==FG_OK)status=fg_vk_tensor_view(s->ids[side],q*bytes,bytes,&s->tile_ids[q][side],err);
        }
        uint64_t bytes=(uint64_t)FG_QSA_SELECTED_TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES;
        if(status==FG_OK)status=fg_vk_tensor_view(s->selected_records,q*bytes,bytes,&s->tile_records[q],err);
    }
    return status;
}

static fg_status open_decode_config(fg_qsa_session **out,fg_model *model,const char *state_path,
                                    uint32_t logical_context,uint32_t hot_tokens,
                                    uint32_t cache_pages,uint32_t batch_size,
                                    fg_vk_tensor *shared_scratch,
                                    bool owned_only,
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
    bool coordinator=rank==0u;
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
        if((coordinator&&!owned_only)||manifest->layer_owner[layer]==rank)
            s->layers[s->layer_count++]=(uint8_t)layer;
    uint32_t expected_layers=coordinator?FG_QSA_MAX_LAYERS:
                FG_QSA_OWNER_LAYERS;
    if(!s->layer_count||s->layer_count>expected_layers){
        fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA layers, expected %u",
                     rank,s->layer_count,expected_layers);
        fg_qsa_session_close(s);return FG_ERR_MISMATCH;
    }
    fg_status status=FG_OK;bool created_state=false;
    uint64_t index_requested=0,index_allocated=0,index_touched=0;
    if(state_path){
        /* The owner runtime creates the worker state at SESSION_BEGIN; the
         * session attaches to the existing file rather than recreating it. */
        status=fg_qsa_state_open(&s->state,state_path,s->layers,s->layer_count,
                                 logical_context,false,err);
        created_state=false;
    }
    if(status==FG_OK)status=make_tensor(s,(uint64_t)logical_context*
                                        FG_Q38_QSA_POSITION_BYTES,&s->positions,err);
    if(status==FG_OK){s->position_written=calloc(1,s->max_context);
        if(!s->position_written){fg_error_set(err,FG_ERR_OOM,"allocate QSA position map");status=FG_ERR_OOM;}}
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
    for(uint32_t slot=0;status==FG_OK&&slot<s->layer_count;slot++){
        status=ensure_index_segment(s,slot,0u,err);
        if(status==FG_OK){
            index_requested+=fg_vk_tensor_bytes(s->index_keys[slot][0]);
            index_allocated+=fg_vk_tensor_allocation_bytes(s->index_keys[slot][0]);
            index_touched+=fg_vk_tensor_bytes(s->index_keys[slot][0]);
        }
    }
    s->cache_pages=cache_pages;
    if(status==FG_OK)status=ensure_page_cache(s,err);
    if(status==FG_OK&&index_requested)
        fprintf(stderr,"[rank %u] QSA Vulkan index canary: eager_segments=%u of=%u "
                       "max_segment=%llu requested=%llu allocated=%llu touched=%llu bytes\n",rank,
                s->layer_count,s->layer_count*s->index_segment_count,
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
        status=make_tensor(s,(uint64_t)fg_qsa_query_tile_size(batch_size)*s->max_blocks*4u,&s->scores[i],err);
        if(status==FG_OK)status=make_tensor(s,(uint64_t)fg_qsa_query_tile_size(batch_size)*s->max_blocks*4u,&s->ids[i],err);
    }
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)fg_qsa_query_tile_size(batch_size)*FG_QSA_SELECTED_TOKENS*
                                        FG_Q38_QSA_TOKEN_RECORD_BYTES,&s->selected_records,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*6144u*4u,&s->attention,err);
    if(status==FG_OK&&!shared_scratch)
        status=make_tensor(s,(uint64_t)batch_size*2560u*4u,&s->output,err);
    if(status==FG_OK)status=create_reusable_views(s,err);
    if(status==FG_OK&&state_path)status=ensure_read_records(s,err);
    if(status==FG_OK&&state_path&&!s->fetch_pages){
        s->fetch_pages=state_fetch_pages;s->fetch_opaque=s;
    }
    if(status==FG_OK)status=create_tile_views(s,batch_size,err);
    if(status==FG_OK)status=make_tensor(s,(uint64_t)FG_QSA_ATTENTION_SPLITS*24u*258u*4u,&s->attn_partials,err);
    fg_vk_memory_stats memory_stats={0};fg_vk_get_memory_stats(fg_model_vk(model),&memory_stats);
    fprintf(stderr,"[rank %u] QSA decode session: %u logical, %u cache pages, %u layers, "
                   "%.1f MiB eager index, %.1f MiB record cache\n",rank,logical_context,
            s->cache_pages,s->layer_count,
            (double)((uint64_t)logical_context*FG_Q38_QSA_POSITION_BYTES+
                     (uint64_t)s->layer_count*
                         fg_qsa_index_segment_bytes(logical_context,0u))/
                (1024.0*1024.0),
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
                              NULL,false,NULL,NULL,err);
}

/* State-backed worker session: the owner computes its QSA layers and persists
 * every completed block into its own state file; selected blocks are read back
 * from the same state, so no page-fetch callback or network transport is
 * required inside the ring. */
fg_status fg_qsa_session_open_state(fg_qsa_session **out,fg_model *model,
                                    const char *state_path,uint32_t logical_context,
                                    uint32_t hot_tokens,uint32_t cache_pages,
                                    uint32_t batch_size,fg_error *err){
    if(!state_path){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA state session path is null");
        return FG_ERR_ARGUMENT;
    }
    return open_decode_config(out,model,state_path,logical_context,hot_tokens,
                              cache_pages,batch_size,NULL,false,NULL,NULL,err);
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
                              NULL,false,fetch_pages,fetch_opaque,err);
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
                              scratch,false,fetch_pages,fetch_opaque,err);
}

/* Coordinator variant: the session keeps a state file for the layers this rank
 * executes while still mirroring remote-owned pages through the fetch callback.
 * Persist and the state-file read path make its own pages recoverable after
 * cache eviction, so pins no longer have to hold a whole context in memory. */
fg_status fg_qsa_session_open_state_mirror_with_scratch(
    fg_qsa_session **out,fg_model *model,const char *state_path,uint32_t logical_context,
    uint32_t hot_tokens,uint32_t cache_pages,uint32_t batch_size,fg_vk_tensor *scratch,
    bool owned_only,fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,fg_error *err){
    if(!state_path){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA state mirror path is null");
        return FG_ERR_ARGUMENT;
    }
    if(!scratch){
        fg_error_set(err,FG_ERR_ARGUMENT,"shared QSA attention scratch is null");
        return FG_ERR_ARGUMENT;
    }
    if(logical_context>hot_tokens&&!fetch_pages){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "tiered QSA state mirror requires a cold-page fetch callback");
        return FG_ERR_ARGUMENT;
    }
    return open_decode_config(out,model,state_path,logical_context,hot_tokens,cache_pages,
                              batch_size,scratch,owned_only,fetch_pages,fetch_opaque,err);
}

void fg_qsa_session_close(fg_qsa_session *s){if(!s)return;fg_vk_tensor_destroy(s->attn_partials);for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(s->sel_scores[i]);fg_vk_tensor_destroy(s->sel_ids[i]);}fg_vk_tensor_destroy(s->sel_result_ids);fg_vk_tensor_destroy(s->batch_records);fg_vk_tensor_destroy(s->batch_partials);fg_vk_tensor_destroy(s->batch_slots);fg_vk_tensor_destroy(s->batch_counts);for(uint32_t q=0;q<FG_QSA_PREFILL_QUERY_TILE;q++){fg_vk_tensor_destroy(s->tile_records[q]);for(uint32_t side=0;side<2u;side++){fg_vk_tensor_destroy(s->tile_scores[q][side]);fg_vk_tensor_destroy(s->tile_ids[q][side]);}}fg_qsa_locality_destroy(s->locality,"close");fg_qsa_page_cache_destroy(s->cache);free(s->select_ids);free(s->read_records);free(s->position_written);fg_vk_tensor_destroy(s->index_key_q8_view);fg_vk_tensor_destroy(s->value_q4_view);fg_vk_tensor_destroy(s->key_q8_view);fg_vk_tensor_destroy(s->attention_view);fg_vk_tensor_destroy(s->gate_view);fg_vk_tensor_destroy(s->query_view);fg_vk_tensor_destroy(s->index_query_view);fg_vk_tensor_destroy(s->token_position_view);fg_vk_tensor_destroy(s->position_view);fg_vk_tensor_destroy(s->output);fg_vk_tensor_destroy(s->attention);fg_vk_tensor_destroy(s->selected_records);fg_vk_tensor_destroy(s->select_flags);fg_vk_tensor_destroy(s->select_resolved);fg_vk_tensor_destroy(s->slot_table);for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(s->ids[i]);fg_vk_tensor_destroy(s->scores[i]);}fg_vk_tensor_destroy(s->index_key_q8);fg_vk_tensor_destroy(s->value_q4);fg_vk_tensor_destroy(s->key_q8);fg_vk_tensor_destroy(s->index_query);fg_vk_tensor_destroy(s->raw_index_key);fg_vk_tensor_destroy(s->raw_index_query);fg_vk_tensor_destroy(s->key);fg_vk_tensor_destroy(s->gate);fg_vk_tensor_destroy(s->query);fg_vk_tensor_destroy(s->raw_value);fg_vk_tensor_destroy(s->raw_key);fg_vk_tensor_destroy(s->raw_query_gate);for(uint32_t i=0;i<FG_QSA_MAX_LAYERS;i++)for(uint32_t segment=0;segment<FG_QSA_INDEX_MAX_SEGMENTS;segment++){fg_vk_tensor_destroy(s->records[i][segment]);fg_vk_tensor_destroy(s->index_keys[i][segment]);}fg_vk_tensor_destroy(s->cache_records);fg_vk_tensor_destroy(s->positions);fg_qsa_state_close(s->state);free(s);}

fg_status fg_qsa_session_reset(fg_qsa_session *s,fg_error *err){if(!s){fg_error_set(err,FG_ERR_ARGUMENT,"QSA session reset is null");return FG_ERR_ARGUMENT;}if(s->position_written)memset(s->position_written,0,s->max_context);fg_qsa_locality_reset(s->locality,"reset");memset(s->committed,0,sizeof(s->committed));memset(s->partial,0,sizeof(s->partial));qsa_cache_reset(s);return s->state?fg_qsa_state_reset(s->state,err):FG_OK;}

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
           (s->select_ids?(uint64_t)s->max_tokens*FG_QSA_MAX_SELECTED_BLOCKS*
            sizeof(uint32_t):0u)+
           fg_qsa_page_cache_memory_bytes(s->cache);
}

static fg_status score_index_segments(fg_qsa_session *s,uint32_t slot,
                                      const fg_vk_tensor *index_query,
                                      uint32_t tokens,fg_error *err){
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_vk_tensor *key_norm=layer_weight(s,s->layers[slot],
                                        "indexer.k_norm.weight",err);
    if(!key_norm)return FG_ERR_MISMATCH;
    if(tokens<=FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY){
        fg_status status=ensure_index_segment(s,slot,0u,err);
        if(status!=FG_OK)return status;
        return fg_vk_qsa_index_score(vk,s->scores[0],s->ids[0],index_query,
                                     s->index_keys[slot][0],key_norm,
                                     s->positions,tokens,err);
    }
    fg_status status=FG_OK;
    for(uint32_t segment=0;status==FG_OK&&segment<s->index_segment_count;segment++){
        uint32_t first=fg_qsa_index_segment_first(s->max_context,segment);
        if(first>=tokens)break;
        uint32_t segment_tokens=tokens-first;
        if(segment_tokens>s->index_segment_tokens[segment])
            segment_tokens=s->index_segment_tokens[segment];
        uint32_t blocks=segment_tokens/FG_Q38_QSA_COMPRESS_RATIO;
        if(!blocks)continue;
        status=ensure_index_segment(s,slot,segment,err);
        if(status!=FG_OK)break;
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

static fg_status qsa_select_scan(fg_qsa_session *s,uint32_t slot,
                                 const fg_vk_tensor *index_query,uint32_t tokens,
                                 uint32_t *side_out,uint32_t *count_out,fg_error *err){
    uint32_t count=tokens/4u;
    if(!count){*side_out=0;*count_out=0;return FG_OK;}
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_status status=score_index_segments(s,slot,index_query,tokens,err);
    uint32_t side=0;bool reduced=false;
    while(status==FG_OK&&count>512u){
        uint32_t next=0;status=fg_vk_topk_reduce(vk,s->scores[side^1u],s->ids[side^1u],
            s->scores[side],s->ids[side],count,&next,err);count=next;side^=1u;reduced=true;
    }
    /* The reduction already emits a sorted top-512 once the window is trimmed,
     * so re-sorting exactly 512 selected blocks would only rewrite the same
     * values; short windows still need the single pass to keep selection order. */
    if(status==FG_OK&&!(reduced&&count==512u)){
        uint32_t final_count=0;status=fg_vk_topk_reduce(vk,s->scores[side^1u],
            s->ids[side^1u],s->scores[side],s->ids[side],count,&final_count,err);
        count=final_count;side^=1u;
    }
    if(status==FG_OK){*side_out=side;*count_out=count;}
    return status;
}

static fg_status select_blocks(fg_qsa_session *s,uint32_t slot,const fg_vk_tensor *index_query,
                               uint32_t tokens,uint32_t *selected,uint32_t *selected_count,
                               bool restart_batch,fg_error *err){
    uint32_t side=0,count=0;
    fg_status status=qsa_select_scan(s,slot,index_query,tokens,&side,&count,err);
    if(status==FG_OK&&count){
        fg_vk_context *vk=fg_model_vk(s->model);
        bool ended=fg_vk_batch_active(vk);
        if(ended)status=fg_vk_end(vk,err);
        if(status==FG_OK)status=fg_vk_tensor_read(s->ids[side],0,selected,
                                                  (uint64_t)count*4u,err);
        if(status==FG_OK&&s->locality)
            fg_qsa_locality_record_selection(s->locality,s->layers[slot],tokens,selected,count);
        if(status==FG_OK&&ended&&restart_batch)status=fg_vk_begin(vk,err);
    }
    if(status==FG_OK)*selected_count=count;
    return status;
}

static bool qsa_select_resolve_requested(void){
    const char *value=getenv("FG_QSA_SELECT_GPU");
    return !(value&&*value&&strcmp(value,"0")==0);
}

static fg_status select_blocks_resolve(fg_qsa_session *s,uint32_t slot,
                                       const fg_vk_tensor *index_query,uint32_t tokens,
                                       uint32_t *selected,uint32_t *selected_count,
                                       bool *fallback,fg_error *err){
    uint32_t side=0,count=0;
    fg_status status=qsa_select_scan(s,slot,index_query,tokens,&side,&count,err);
    fg_vk_context *vk=fg_model_vk(s->model);
    bool trace=qsa_trace_enabled();
    double scan_done=trace?qsa_now_ms():0.0,end_done=0.0,read_done=0.0;
    uint32_t groups=(count+255u)/256u;
    if(status==FG_OK&&groups){
        fg_vk_host_write_visible(vk);
        status=fg_vk_qsa_select_resolve(vk,s->select_resolved,s->select_flags,
            s->ids[side],s->slot_table,count,s->max_blocks,
            (uint32_t)slot*s->max_blocks,err);
    }
    bool ended=status==FG_OK&&fg_vk_batch_active(vk);
    if(ended)status=fg_vk_end(vk,err);
    if(trace)end_done=qsa_now_ms();
    uint32_t misses=0;
    if(status==FG_OK&&groups){
        uint32_t flags[2]={0u,0u};
        status=fg_vk_tensor_read(s->select_flags,0,flags,groups*4u,err);
        for(uint32_t i=0;i<groups;i++)misses+=flags[i];
    }
    if(status==FG_OK&&misses)
        status=fg_vk_tensor_read(s->ids[side],0,selected,(uint64_t)count*4u,err);
    if(trace){
        read_done=qsa_now_ms();
        fprintf(stderr,"QSA_SELECT_TRACE layer=%u tokens=%u count=%u misses=%u "
            "wait_ms=%.3f read_ms=%.3f\n",s->layers[slot],tokens,count,misses,
            end_done-scan_done,read_done-end_done);
    }
    if(status==FG_OK){*selected_count=count;*fallback=misses!=0;}
    return status;
}

/* Single-token decode attention: the serial kernel scans every selected
 * record under a per-token workgroup barrier, so long contexts leave the GPU
 * latency-bound.  Above a small window, split the record range across
 * (head, split) workgroups and merge the partial online-softmax states with
 * the deterministic merge kernel instead. */
#define FG_QSA_DECODE_SPLIT_TOKENS 256u
/* Single-token decode attention: the serial kernel scans every selected
 * record under a per-token workgroup barrier, so long contexts leave the GPU
 * latency-bound.  Above a small window, the shared-tile split kernel scans the
 * record range with one workgroup per (kv head, split), decodes each record
 * once for the twelve heads that share the kv head, and merges the partial
 * online-softmax states with the deterministic merge kernel instead. */
#define FG_QSA_DECODE_SPLIT_TOKENS 256u
static fg_status decode_attention(fg_qsa_session *s,fg_vk_tensor *attention,
                                  const fg_vk_tensor *query,const fg_vk_tensor *gate,
                                  uint32_t selected_tokens,fg_error *err){
    fg_vk_context *vk=fg_model_vk(s->model);
    uint32_t splits=(selected_tokens+FG_QSA_DECODE_SPLIT_TOKENS-1u)/
        FG_QSA_DECODE_SPLIT_TOKENS;
    if(splits>FG_QSA_ATTENTION_SPLITS)splits=FG_QSA_ATTENTION_SPLITS;
    if(splits<2u)
        return fg_vk_qsa_attention(vk,attention,s->selected_records,query,gate,
                                   selected_tokens,err);
    fg_status status=fg_vk_qsa_decode_attention_split(vk,s->attn_partials,
        s->selected_records,query,selected_tokens,splits,err);
    if(status==FG_OK)status=fg_vk_qsa_attention_merge(vk,attention,
        s->attn_partials,gate,splits,err);
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
    bool trace=qsa_trace_enabled();double t0=0,t_select=0,t_lookup=0,t_fetch=0,t_gather=0;uint32_t fetched=0;
    bool use_resolve=complete_blocks>FG_QSA_MAX_SELECTED_BLOCKS&&!s->locality&&
        s->slot_table&&s->select_resolved&&s->select_flags&&
        qsa_select_resolve_requested();
    bool fallback=false,resolved=false;
    if(trace)t0=qsa_now_ms();
    if(complete_blocks<=FG_QSA_MAX_SELECTED_BLOCKS){
        selected_count=complete_blocks;
        for(uint32_t i=0;i<selected_count;i++)selected[i]=i;
        if(s->locality)
            fg_qsa_locality_record_selection(s->locality,s->layers[slot],tokens,
                                             selected,selected_count);
    }else if(use_resolve){
        status=select_blocks_resolve(s,slot,index_query,tokens,selected,
                                     &selected_count,&fallback,err);
        restart_batch=resume_batch;
        resolved=status==FG_OK&&!fallback;
    }else{
        status=select_blocks(s,slot,index_query,tokens,selected,&selected_count,false,err);
        restart_batch=resume_batch;
    }
    if(trace)t_select=qsa_now_ms();
    uint32_t missing[FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t fetch_blocks[FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t cache_slots[FG_QSA_MAX_SELECTED_BLOCKS],missing_count=0;
    if(!resolved)for(uint32_t i=0;status==FG_OK&&i<selected_count;i++){
        if(fg_qsa_page_cache_lookup(s->cache,s->layers[slot],selected[i],
                                    &cache_slots[i])){
            if(s->locality)fg_qsa_locality_record_cache(s->locality,s->layers[slot],true);
        }else{
            if(s->locality)fg_qsa_locality_record_cache(s->locality,s->layers[slot],false);
            missing[missing_count++]=selected[i];
        }
    }
    if(trace)t_lookup=qsa_now_ms();
    if(status==FG_OK&&!resolved&&missing_count)status=ensure_read_records(s,err);
    if(status==FG_OK&&!resolved&&missing_count){
        uint32_t fetch_count=0;
        status=fg_qsa_page_cache_plan_fetch(s->cache,s->layers[slot],missing,missing_count,
            complete_blocks,UINT32_MAX,fetch_blocks,FG_QSA_MAX_SELECTED_BLOCKS,
            &fetch_count,err);
        if(trace)fetched=fetch_count;
        if(status==FG_OK&&!s->fetch_pages){
            fg_error_set(err,FG_ERR_UNAVAILABLE,"QSA cold page miss has no owner fetch service");
            status=FG_ERR_UNAVAILABLE;
        }else if(status==FG_OK)
            status=s->fetch_pages(s->fetch_opaque,s->layers[slot],fetch_blocks,fetch_count,
                                  s->read_records,err);
        for(uint32_t i=0;status==FG_OK&&i<fetch_count;i++){
            const uint8_t *page=s->read_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES;
            uint32_t cache_slot=0;bool hit=false;
            status=qsa_cache_acquire(s,s->layers[slot],fetch_blocks[i],
                                     &cache_slot,&hit,err);
            if(status==FG_OK)status=fg_vk_tensor_write(
                s->cache_records,(uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,
                page,FG_QSA_PAGE_RECORD_BYTES,err);
        }
    }
    if(trace)t_fetch=qsa_now_ms();
    if(!resolved)for(uint32_t i=0;status==FG_OK&&i<selected_count;i++){
        if(!fg_qsa_page_cache_lookup(s->cache,s->layers[slot],selected[i],
                                     &cache_slots[i])){
            fg_error_set(err,FG_ERR_MISMATCH,"selected QSA page was not cached");
            status=FG_ERR_MISMATCH;
        }else qsa_slot_store(s,s->layers[slot],selected[i],cache_slots[i]);
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
    if(status==FG_OK&&!resolved&&selected_count)status=fg_vk_tensor_write(
        s->ids[0],0,cache_slots,(uint64_t)selected_count*sizeof(*cache_slots),err);
    if(status==FG_OK&&restart_batch)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_qsa_record_gather(
        vk,s->selected_records,s->cache_records,
        resolved?s->select_resolved:s->ids[0],0u,
        s->cache_pages*FG_Q38_QSA_COMPRESS_RATIO,selected_count,tail_start,tail,err);
    if(trace)t_gather=qsa_now_ms();
    uint32_t selected_tokens=selected_count*FG_Q38_QSA_COMPRESS_RATIO+tail;
    if(status==FG_OK)status=decode_attention(s,attention,query,gate,
                                             selected_tokens,err);
    if(trace){double t_end=qsa_now_ms();fprintf(stderr,"QSA_ATTEND_TRACE layer=%u tokens=%u selected=%u missing=%u fetched=%u select_ms=%.3f lookup_ms=%.3f fetch_ms=%.3f gather_ms=%.3f attn_ms=%.3f total_ms=%.3f\n",s->layers[slot],tokens,selected_count,missing_count,fetched,t_select-t0,t_lookup-t_select,t_fetch-t_lookup,t_gather-t_fetch,t_end-t_gather,t_end-t0);}
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
    fg_status status=ensure_index_segment(s,slot,segment,err);
    if(status==FG_OK)status=qsa_cache_acquire(
        s,s->layers[slot],token/FG_Q38_QSA_COMPRESS_RATIO,
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
    uint32_t segment=0,offset=0;if(!fg_qsa_index_token_location(s->max_context,token,&segment,&offset)){fg_error_set(err,FG_ERR_LIMIT,"QSA token has no index segment");return FG_ERR_LIMIT;}fg_status status=ensure_index_segment(s,slot,segment,err);if(status!=FG_OK)return status;status=fg_qsa_state_write_block(s->state,slot,token/FG_Q38_QSA_COMPRESS_RATIO,s->partial[slot],inside+1u,err);if(status!=FG_OK)return status;s->committed[slot]=token+1u;memcpy((uint8_t *)fg_vk_tensor_map(s->index_keys[slot][segment])+(uint64_t)offset*FG_Q38_QSA_INDEX_KEY_BYTES,index_key,FG_Q38_QSA_INDEX_KEY_BYTES);
    uint32_t selected_blocks[FG_QSA_MAX_SELECTED_BLOCKS],block_count=0,tokens=token+1u;status=select_blocks(s,slot,index_query,tokens,selected_blocks,&block_count,true,err);uint32_t selected_tokens=0;if(status==FG_OK&&block_count){uint32_t committed[FG_QSA_MAX_SELECTED_BLOCKS];status=fg_qsa_state_read_blocks(s->state,slot,selected_blocks,block_count,s->read_records,committed,err);for(uint32_t i=0;status==FG_OK&&i<block_count;i++){if(committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){fg_error_set(err,FG_ERR_MISMATCH,"selected QSA block is not complete");status=FG_ERR_MISMATCH;break;}memcpy((uint8_t *)fg_vk_tensor_map(s->selected_records)+(uint64_t)selected_tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,s->read_records+(uint64_t)i*FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES,(uint64_t)FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES);selected_tokens+=FG_Q38_QSA_COMPRESS_RATIO;}}
    uint32_t tail=tokens%FG_Q38_QSA_COMPRESS_RATIO;if(status==FG_OK&&tail){memcpy((uint8_t *)fg_vk_tensor_map(s->selected_records)+(uint64_t)selected_tokens*FG_Q38_QSA_TOKEN_RECORD_BYTES,s->partial[slot],(uint64_t)tail*FG_Q38_QSA_TOKEN_RECORD_BYTES);selected_tokens+=tail;}if(status==FG_OK)status=decode_attention(s,attention,query,gate,selected_tokens,err);return status;
}

fg_status fg_qsa_session_decode(fg_qsa_session *s,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;if(!s||signed_slot<0||!position||!hidden||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA decode arguments");return FG_ERR_ARGUMENT;}uint32_t slot=(uint32_t)signed_slot;if(token!=s->committed[slot]||token>=s->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA token position does not match committed state");return FG_ERR_MISMATCH;}fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err),*kw=layer_weight(s,layer,"attn_k.weight",err),*vw=layer_weight(s,layer,"attn_v.weight",err),*qn=layer_weight(s,layer,"attn_q_norm.weight",err),*kn=layer_weight(s,layer,"attn_k_norm.weight",err),*ow=layer_weight(s,layer,"attn_output.weight",err),*iqw=layer_weight(s,layer,"indexer.q_proj.weight",err),*ikw=layer_weight(s,layer,"indexer.k_proj.weight",err),*iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn)return FG_ERR_MISMATCH;
    bool trace=qsa_trace_enabled();double t0=0,t_proj=0,t_commit=0,t_flush=0;if(trace)t0=qsa_now_ms();
    fg_vk_context *vk=fg_model_vk(s->model);double flush_begin=trace?qsa_now_ms():0.0;fg_status status=qsa_decode_flush(vk,s->cache!=NULL,err);if(trace)t_flush+=qsa_now_ms()-flush_begin;
    /* Materialize the token's index segment before any batch records; the
     * residency canary cannot run under an active batch. */
    if(status==FG_OK)status=ensure_index_segments_for_range(s,slot,token,1u,err);
    if(status!=FG_OK)return status;
    uint32_t *resident_positions=status==FG_OK?fg_vk_tensor_map(s->positions):NULL;if(status==FG_OK&&!s->position_written[token]){memcpy(resident_positions+(uint64_t)token*3u,position,12u);s->position_written[token]=1u;}else if(status==FG_OK&&memcmp(resident_positions+(uint64_t)token*3u,position,12u)!=0){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers received inconsistent MRoPE positions");return FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->position_view,s->positions,(uint64_t)token*12u,12u,err);if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_projection",err);
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
    if(status==FG_OK){flush_begin=trace?qsa_now_ms():0.0;status=qsa_decode_flush(vk,s->cache!=NULL,err);if(trace)t_flush+=qsa_now_ms()-flush_begin;}
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status!=FG_OK)return status;
    if(trace)t_proj=qsa_now_ms();
    if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_state_attention",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&s->cache)status=commit_and_attend_cache(s,slot,token,s->key_q8,s->value_q4,s->index_key_q8,s->position_view,s->index_query,s->query,s->gate,s->attention,err);
    else if(status==FG_OK)status=commit_and_attend(s,slot,token,position,fg_vk_tensor_map(s->key_q8),fg_vk_tensor_map(s->value_q4),fg_vk_tensor_map(s->index_key_q8),s->index_query,s->query,s->gate,s->attention,err);
    if(trace)t_commit=qsa_now_ms();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"qsa_output",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->output,ow,s->attention,6144u,2560u,1u,1.0f,err);
    if(status!=FG_OK&&fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(trace)fprintf(stderr,"QSA_TRACE layer=%u token=%u status=%d proj_ms=%.3f flush_ms=%.3f commit_attend_ms=%.3f out_ms=%.3f total_ms=%.3f\n",layer,token,(int)status,t_proj-t0,t_flush,t_commit-t_proj,qsa_now_ms()-t_commit,qsa_now_ms()-t0);
    if(status==FG_OK){*output=s->output;}return status;
}

/* Commit projected records independently of attention. New pages stay pinned
 * until the coordinator publishes them to their storage owners. */
static fg_status commit_prefill_records(fg_qsa_session *s,uint32_t slot,
    uint32_t first,uint32_t count,fg_error *err){
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_status status=fg_vk_begin(vk,err);
    for(uint32_t i=0;status==FG_OK&&i<count;i++){
        uint32_t token=first+i,segment=0,offset=0,cache_slot=0;bool hit=false;
        if(!fg_qsa_index_token_location(s->max_context,token,&segment,&offset)){
            fg_error_set(err,FG_ERR_LIMIT,"QSA prefill token has no index segment");
            status=FG_ERR_LIMIT;break;
        }
        status=ensure_index_segment(s,slot,segment,err);
        if(status!=FG_OK)break;
        status=qsa_cache_acquire(s,s->layers[slot],token/4u,&cache_slot,&hit,err);
        if(status==FG_OK)status=fg_qsa_page_cache_pin(s->cache,s->layers[slot],token/4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->key_q8_view,s->key_q8,
            (uint64_t)i*FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_KEY_BYTES,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->value_q4_view,s->value_q4,
            (uint64_t)i*FG_Q38_QSA_VALUE_BYTES,FG_Q38_QSA_VALUE_BYTES,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->index_key_q8_view,s->index_key_q8,
            (uint64_t)i*FG_Q38_QSA_INDEX_KEY_BYTES,FG_Q38_QSA_INDEX_KEY_BYTES,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->token_position_view,s->positions,
            (uint64_t)token*FG_Q38_QSA_POSITION_BYTES,FG_Q38_QSA_POSITION_BYTES,err);
        if(status==FG_OK)status=fg_vk_qsa_record_commit_segmented(vk,s->cache_records,
            s->index_keys[slot][segment],s->key_q8_view,s->value_q4_view,
            s->index_key_q8_view,s->token_position_view,0u,token,offset,
            s->index_segment_tokens[segment],cache_slot*4u+token%4u,s->cache_pages*4u,err);
    }
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    return status;
}

static void qsa_align_host_select(fg_qsa_session *s){
    if(s&&!s->select_ids){
        s->select_ids=malloc((uint64_t)s->max_tokens*FG_QSA_MAX_SELECTED_BLOCKS*4u);
    }
}

static fg_status ensure_select_scratch(fg_qsa_session *s,fg_error *err){
    if(s->sel_scores[0]&&s->sel_ids[0]&&s->sel_scores[1]&&s->sel_ids[1]&&
       s->sel_result_ids&&s->select_ids)return FG_OK;
    for(uint32_t i=0;i<2u;i++){
        fg_vk_tensor_destroy(s->sel_scores[i]);s->sel_scores[i]=NULL;
        fg_vk_tensor_destroy(s->sel_ids[i]);s->sel_ids[i]=NULL;
    }
    fg_vk_tensor_destroy(s->sel_result_ids);s->sel_result_ids=NULL;
    /* select_ids is host staging whose size only depends on max_tokens; the
     * selection callers hold a pointer to it across this initialization, so
     * it must never be freed here. */
    uint64_t stride=(uint64_t)s->max_blocks*4u;
    /* Rank 0's mirror has only ~150 MB of headroom; keep the lazy selection
     * scratch at 4 MiB per side. */
    uint64_t budget=UINT64_C(4)*1024u*1024u;
    uint64_t chunk=stride?budget/stride:1u;
    if(chunk>s->max_tokens)chunk=s->max_tokens;
    if(!chunk)chunk=1u;
    s->select_chunk=(uint32_t)chunk;
    uint64_t bytes=chunk*stride;
    for(uint32_t i=0;i<2u;i++){
        fg_status status=make_tensor(s,bytes,&s->sel_scores[i],err);
        if(status==FG_OK)status=make_tensor(s,bytes,&s->sel_ids[i],err);
        if(status!=FG_OK)return status;
    }
    fg_status status=make_tensor(s,(uint64_t)s->max_tokens*
        FG_QSA_MAX_SELECTED_BLOCKS*4u,&s->sel_result_ids,err);
    if(status!=FG_OK)return status;
    qsa_align_host_select(s);
    if(!s->select_ids){fg_error_set(err,FG_ERR_OOM,"allocate QSA selection staging");return FG_ERR_OOM;}
    return FG_OK;
}

static fg_status ensure_prefill_batch(fg_qsa_session *s,fg_error *err){
    if(s->batch_records&&s->batch_partials&&s->batch_slots&&s->batch_counts&&
       s->select_ids)return FG_OK;
    fg_vk_tensor_destroy(s->batch_records);s->batch_records=NULL;
    fg_vk_tensor_destroy(s->batch_partials);s->batch_partials=NULL;
    fg_vk_tensor_destroy(s->batch_slots);s->batch_slots=NULL;
    fg_vk_tensor_destroy(s->batch_counts);s->batch_counts=NULL;
    /* select_ids is deliberately preserved: callers capture it before the
     * scratch tensors above are (re)created and it never depends on them. */
    const uint32_t tile=FG_QSA_PREFILL_BATCH_QUERIES;
    fg_status status=make_tensor(s,(uint64_t)tile*FG_QSA_SELECTED_TOKENS*
        FG_Q38_QSA_TOKEN_RECORD_BYTES,&s->batch_records,err);
    if(status==FG_OK)status=make_tensor(s,(uint64_t)tile*24u*
        FG_QSA_ATTENTION_SPLITS*258u*4u,&s->batch_partials,err);
    if(status==FG_OK)status=make_tensor(s,(uint64_t)tile*
        FG_QSA_MAX_SELECTED_BLOCKS*4u,&s->batch_slots,err);
    if(status==FG_OK)status=make_tensor(s,(uint64_t)tile*4u,&s->batch_counts,err);
    if(status!=FG_OK)return status;
    qsa_align_host_select(s);
    if(!s->select_ids){fg_error_set(err,FG_ERR_OOM,"allocate QSA selection staging");return FG_ERR_OOM;}
    return FG_OK;
}

/* Causal top-512 selection for a contiguous query range, batched into one
 * score sweep plus per-query top-k merges.  The fast path (all complete
 * blocks visible) keeps selecting every block in order without GPU work. */
static fg_status select_prefill_batch(fg_qsa_session *s,uint32_t slot,
    uint32_t first_visible,uint32_t first_query,uint32_t queries,uint32_t *selected,
    uint32_t select_stride,uint32_t *counts,fg_error *err){
    if(!queries||!selected||!counts||select_stride<FG_QSA_MAX_SELECTED_BLOCKS){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched QSA selection arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_context *vk=fg_model_vk(s->model);
    uint32_t blocks_total=(first_visible+queries-1u)/FG_Q38_QSA_COMPRESS_RATIO;
    for(uint32_t q=0;q<queries;q++){
        uint32_t complete=(first_visible+q)/FG_Q38_QSA_COMPRESS_RATIO;
        counts[q]=complete<FG_QSA_MAX_SELECTED_BLOCKS?complete:FG_QSA_MAX_SELECTED_BLOCKS;
    }
    if(blocks_total<=FG_QSA_MAX_SELECTED_BLOCKS){
        for(uint32_t q=0;q<queries;q++)
            for(uint32_t i=0;i<counts[q];i++)selected[(uint64_t)q*select_stride+i]=i;
        return fg_qsa_submit_host_reads(vk,err);
    }
    fg_vk_tensor *norm=layer_weight(s,s->layers[slot],"indexer.k_norm.weight",err);
    if(!norm)return FG_ERR_MISMATCH;
    fg_status status=ensure_select_scratch(s,err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    uint32_t chunk=s->select_chunk;
    for(uint32_t base=0;status==FG_OK&&base<queries;base+=chunk){
        uint32_t batch=queries-base;
        if(batch>chunk)batch=chunk;
        uint64_t entries=(uint64_t)batch*blocks_total*4u;
        fg_vk_tensor *score_view=NULL,*id_view=NULL;
        status=fg_vk_tensor_view(s->sel_scores[0],0,entries,&score_view,err);
        if(status==FG_OK)status=fg_vk_tensor_view(s->sel_ids[0],0,entries,&id_view,err);
        for(uint32_t segment=0;status==FG_OK&&segment<s->index_segment_count;segment++){
            uint32_t first=fg_qsa_index_segment_first(s->max_context,segment);
        uint32_t visible=first_visible+queries-1u;
        if(first>visible)break;
        /* visible is an exclusive count: it is the number of tokens the last
         * query of the batch can see (first_visible already includes the
         * first query's own token).  Adding one here used to score a fourth
         * token of the block after the last visible one, so chunks whose
         * endpoint landed on a block boundary requested blocks_end one past
         * blocks_total and the score-tile validation rejected the dispatch. */
        uint32_t count=visible-first;
        if(count>s->index_segment_tokens[segment])count=s->index_segment_tokens[segment];
            if(count<FG_Q38_QSA_COMPRESS_RATIO)continue;
            uint32_t blocks_end=count/FG_Q38_QSA_COMPRESS_RATIO;
            status=ensure_index_segment(s,slot,segment,err);
            for(uint32_t window=0;status==FG_OK&&window<blocks_end;
                window+=FG_QSA_SELECT_WINDOW_BLOCKS){
                uint32_t span=blocks_end-window;
                if(span>FG_QSA_SELECT_WINDOW_BLOCKS)span=FG_QSA_SELECT_WINDOW_BLOCKS;
                status=fg_vk_tensor_view_rebind(s->index_query_view,s->index_query,
                    (uint64_t)(first_query+base)*512u*4u,(uint64_t)batch*512u*4u,err);
                if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->token_position_view,
                    s->positions,(uint64_t)(first+window*FG_Q38_QSA_COMPRESS_RATIO)*
                    FG_Q38_QSA_POSITION_BYTES,(uint64_t)span*
                    FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_POSITION_BYTES,err);
                if(status==FG_OK){
                    fg_vk_tensor *key_view=NULL;
                    status=fg_vk_tensor_view(s->index_keys[slot][segment],
                        (uint64_t)window*FG_Q38_QSA_COMPRESS_RATIO*
                        FG_Q38_QSA_INDEX_KEY_BYTES,(uint64_t)span*
                        FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_INDEX_KEY_BYTES,
                        &key_view,err);
                    if(status==FG_OK)status=fg_vk_qsa_index_score_batch(vk,score_view,id_view,
                        s->index_query_view,key_view,norm,s->token_position_view,
                        span*FG_Q38_QSA_COMPRESS_RATIO,
                        first/FG_Q38_QSA_COMPRESS_RATIO+window,blocks_total,
                        first_visible+base,batch,err);
                    fg_vk_tensor_destroy(key_view);
                }
            }
        }
        fg_vk_tensor_destroy(id_view);
        fg_vk_tensor_destroy(score_view);
        if(status==FG_OK){
            fg_vk_tensor *result_view=NULL;
            status=fg_vk_tensor_view(s->sel_result_ids,(uint64_t)base*
                FG_QSA_MAX_SELECTED_BLOCKS*4u,(uint64_t)batch*
                FG_QSA_MAX_SELECTED_BLOCKS*4u,&result_view,err);
            if(status==FG_OK)status=fg_vk_qsa_select_merge(vk,s->sel_scores[0],
                s->sel_ids[0],s->sel_scores[1],s->sel_ids[1],result_view,
                blocks_total,blocks_total,batch,err);
            fg_vk_tensor_destroy(result_view);
        }
    }
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    if(status==FG_OK)status=fg_vk_tensor_read(s->sel_result_ids,0,s->select_ids,
        (uint64_t)queries*FG_QSA_MAX_SELECTED_BLOCKS*4u,err);
    if(status==FG_OK)for(uint32_t q=0;q<queries;q++)
        memcpy(selected+(uint64_t)q*select_stride,
            s->select_ids+(uint64_t)q*FG_QSA_MAX_SELECTED_BLOCKS,
            (uint64_t)counts[q]*4u);
    return status;
}

static fg_status __attribute__((unused)) select_prefill_tile(fg_qsa_session *s,uint32_t slot,
    uint32_t first_token,uint32_t first_query,uint32_t queries,
    uint32_t selected[FG_QSA_PREFILL_QUERY_TILE][FG_QSA_MAX_SELECTED_BLOCKS],
    uint32_t *counts,fg_error *err){
    /* The tile path is the parity reference for the batched selector. */
    return select_prefill_batch(s,slot,first_token+first_query+1u,first_query,queries,
        &selected[0][0],FG_QSA_MAX_SELECTED_BLOCKS,counts,err);
}

typedef struct qsa_missing_page {uint32_t block,query,index;} qsa_missing_page;
static int missing_page_order(const void *a,const void *b){
    uint32_t left=((const qsa_missing_page *)a)->block,right=((const qsa_missing_page *)b)->block;
    return (left>right)-(left<right);
}

static fg_status __attribute__((unused)) gather_prefill_tile(fg_qsa_session *s,uint32_t slot,
    uint32_t first_visible,uint32_t queries,
    uint32_t selected[FG_QSA_PREFILL_QUERY_TILE][FG_QSA_MAX_SELECTED_BLOCKS],
    const uint32_t *counts,fg_error *err){
    qsa_missing_page missing[FG_QSA_PREFILL_QUERY_TILE*FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t missing_count=0,layer=s->layers[slot];
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_status status=fg_vk_begin(vk,err);
    /* Gather cache hits before any eviction. Each query then owns a private,
     * bounded record slice, so even a union larger than the cache is safe. */
    for(uint32_t q=0;status==FG_OK&&q<queries;q++){
        uint32_t cache_slots[FG_QSA_MAX_SELECTED_BLOCKS],tokens=first_visible+q;
        if(s->locality)fg_qsa_locality_record_selection(s->locality,layer,tokens,selected[q],counts[q]);
        for(uint32_t i=0;i<counts[q];i++){
            if(selected[q][i]>=tokens/4u){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA tile selected a future block");
                status=FG_ERR_MISMATCH;break;
            }
            bool hit=fg_qsa_page_cache_lookup(s->cache,layer,selected[q][i],&cache_slots[i]);
            if(s->locality)fg_qsa_locality_record_cache(s->locality,layer,hit);
            if(!hit){
                cache_slots[i]=UINT32_MAX;
                missing[missing_count++]=(qsa_missing_page){selected[q][i],q,i};
            }
        }
        uint32_t tail=tokens%4u,tail_slot=0;
        if(status==FG_OK&&tail&&!fg_qsa_page_cache_lookup(s->cache,layer,tokens/4u,&tail_slot)){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA tile lost its causal tail");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK&&counts[q])status=fg_vk_tensor_write(s->tile_ids[q][0],0,
            cache_slots,(uint64_t)counts[q]*4u,err);
        if(status==FG_OK)status=fg_vk_qsa_record_gather(vk,s->tile_records[q],s->cache_records,
            s->tile_ids[q][0],0u,s->cache_pages*4u,counts[q],tail_slot*4u,tail,err);
    }
    if(status!=FG_OK)return status;
    /* A hit-only gather stays in the shared batch; the next selection fence
     * consumes it before any host write touches the same staging buffers. */
    if(!missing_count)return status;
    status=fg_qsa_submit_host_reads(vk,err);
    if(status!=FG_OK)return status;
    if(!s->fetch_pages){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"QSA cold page miss has no owner fetch service");
        return FG_ERR_UNAVAILABLE;
    }
    status=ensure_read_records(s,err);
    qsort(missing,missing_count,sizeof(*missing),missing_page_order);
    uint32_t at=0;
    while(status==FG_OK&&at<missing_count){
        uint32_t fetch[FG_QSA_MAX_SELECTED_BLOCKS],starts[FG_QSA_MAX_SELECTED_BLOCKS+1u];
        uint32_t count=0,end=at;
        while(end<missing_count&&count<FG_QSA_MAX_SELECTED_BLOCKS){
            starts[count]=end;fetch[count]=missing[end].block;
            do{end++;}while(end<missing_count&&missing[end].block==fetch[count]);
            count++;
        }
        starts[count]=end;
        status=s->fetch_pages(s->fetch_opaque,layer,fetch,count,s->read_records,err);
        for(uint32_t i=0;status==FG_OK&&i<count;i++){
            const uint8_t *page=s->read_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES;
            for(uint32_t j=starts[i];status==FG_OK&&j<starts[i+1u];j++)
                status=fg_vk_tensor_write(s->tile_records[missing[j].query],
                    (uint64_t)missing[j].index*FG_QSA_PAGE_RECORD_BYTES,page,FG_QSA_PAGE_RECORD_BYTES,err);
            if(status!=FG_OK)break;
            /* Cache insertion is optional; unpublished pages may pin all slots. */
            uint32_t cache_slot=0;bool hit=false;fg_error cache_error={0};
            fg_status cached=qsa_cache_acquire_soft(s,layer,fetch[i],
                                                    &cache_slot,&hit,&cache_error);
            if(status==FG_OK&&cached==FG_OK&&cache_slot!=UINT32_MAX)
                status=fg_vk_tensor_write(s->cache_records,
                    (uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,page,
                    FG_QSA_PAGE_RECORD_BYTES,err);
            else if(status==FG_OK&&cached!=FG_OK){status=cached;if(err)*err=cache_error;}
        }
        at=end;
    }
    return status;
}

/* Batched tile gather + attention: stage the tile's selected records into a
 * private per-query arena (GPU copies for cache hits, host copies for cold
 * pages and the causal tail), then run one split and one merge dispatch for
 * every query of the tile.  A fence at entry keeps the previous tile's
 * dispatches from reading the staging buffers while the host rewrites them. */
static fg_status prefill_tile_attention(fg_qsa_session *s,uint32_t slot,
    uint32_t first_query,uint32_t first_visible,uint32_t queries,
    const uint32_t *selected,uint32_t select_stride,const uint32_t *counts,fg_error *err){
    if(!queries||queries>FG_QSA_PREFILL_BATCH_QUERIES){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched QSA attention tile");
        return FG_ERR_ARGUMENT;
    }
    uint32_t layer=s->layers[slot];
    fg_vk_context *vk=fg_model_vk(s->model);
    fg_status status=fg_qsa_submit_host_reads(vk,err);
    uint32_t cache_slots[FG_QSA_PREFILL_BATCH_QUERIES][FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t tail_slots[FG_QSA_PREFILL_BATCH_QUERIES],tails[FG_QSA_PREFILL_BATCH_QUERIES];
    uint32_t selected_tokens[FG_QSA_PREFILL_BATCH_QUERIES];
    qsa_missing_page missing[FG_QSA_PREFILL_BATCH_QUERIES*FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t missing_count=0;
    for(uint32_t q=0;status==FG_OK&&q<queries;q++){
        uint32_t tokens=first_visible+q;
        if(s->locality)fg_qsa_locality_record_selection(s->locality,layer,tokens,
            selected+(uint64_t)q*select_stride,counts[q]);
        for(uint32_t i=0;i<counts[q];i++){
            bool hit=fg_qsa_page_cache_lookup(s->cache,layer,
                selected[(uint64_t)q*select_stride+i],&cache_slots[q][i]);
            if(s->locality)fg_qsa_locality_record_cache(s->locality,layer,hit);
            if(!hit){
                cache_slots[q][i]=UINT32_MAX;
                missing[missing_count++]=(qsa_missing_page){
                    selected[(uint64_t)q*select_stride+i],q,i};
            }
        }
        for(uint32_t i=counts[q];i<FG_QSA_MAX_SELECTED_BLOCKS;i++)
            cache_slots[q][i]=UINT32_MAX;
        uint32_t tail=tokens%FG_Q38_QSA_COMPRESS_RATIO,tail_slot=0;
        if(tail&&!fg_qsa_page_cache_lookup(s->cache,layer,
                tokens/FG_Q38_QSA_COMPRESS_RATIO,&tail_slot)){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA tile lost its causal tail");
            status=FG_ERR_MISMATCH;
            break;
        }
        tail_slots[q]=tail?tail_slot:UINT32_MAX;
        tails[q]=tail;
        selected_tokens[q]=counts[q]*FG_Q38_QSA_COMPRESS_RATIO+tail;
    }
    if(status==FG_OK)status=fg_vk_tensor_write(s->batch_slots,0,cache_slots,
        (uint64_t)queries*FG_QSA_MAX_SELECTED_BLOCKS*4u,err);
    if(status==FG_OK)status=fg_vk_tensor_write(s->batch_counts,0,selected_tokens,
        (uint64_t)queries*4u,err);
    uint8_t *arena=status==FG_OK?(uint8_t *)fg_vk_tensor_map(s->batch_records):NULL;
    const uint8_t *pages=status==FG_OK?(const uint8_t *)fg_vk_tensor_map(s->cache_records):NULL;
    for(uint32_t q=0;status==FG_OK&&q<queries;q++)if(tails[q]){
        const uint8_t *tail_page=pages+(uint64_t)tail_slots[q]*FG_QSA_PAGE_RECORD_BYTES;
        uint8_t *destination=arena+((uint64_t)q*FG_QSA_SELECTED_TOKENS+
            (uint64_t)counts[q]*FG_Q38_QSA_COMPRESS_RATIO)*FG_Q38_QSA_TOKEN_RECORD_BYTES;
        memcpy(destination,tail_page,(uint64_t)tails[q]*FG_Q38_QSA_TOKEN_RECORD_BYTES);
    }
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK){
        fg_vk_tensor *slot_view=NULL;
        status=fg_vk_tensor_view(s->batch_slots,0,
            (uint64_t)queries*FG_QSA_MAX_SELECTED_BLOCKS*4u,&slot_view,err);
        if(status==FG_OK)status=fg_vk_qsa_record_gather_batch(vk,s->batch_records,
            s->cache_records,slot_view,queries,FG_QSA_MAX_SELECTED_BLOCKS,
            FG_QSA_SELECTED_TOKENS,s->cache_pages*FG_Q38_QSA_COMPRESS_RATIO,err);
        fg_vk_tensor_destroy(slot_view);
    }
    if(status==FG_OK&&missing_count){
        /* Cache hits are materialized before the host mutates cache slots. */
        status=fg_qsa_submit_host_reads(vk,err);
        if(status==FG_OK&&!s->fetch_pages){
            fg_error_set(err,FG_ERR_UNAVAILABLE,"QSA cold page miss has no owner fetch service");
            status=FG_ERR_UNAVAILABLE;
        }
        if(status==FG_OK)status=ensure_read_records(s,err);
        qsort(missing,missing_count,sizeof(*missing),missing_page_order);
        uint32_t at=0;
        while(status==FG_OK&&at<missing_count){
            uint32_t fetch[FG_QSA_MAX_SELECTED_BLOCKS],starts[FG_QSA_MAX_SELECTED_BLOCKS+1u];
            uint32_t count=0,end=at;
            while(end<missing_count&&count<FG_QSA_MAX_SELECTED_BLOCKS){
                starts[count]=end;fetch[count]=missing[end].block;
                do{end++;}while(end<missing_count&&missing[end].block==fetch[count]);
                count++;
            }
            starts[count]=end;
            status=s->fetch_pages(s->fetch_opaque,layer,fetch,count,s->read_records,err);
            for(uint32_t i=0;status==FG_OK&&i<count;i++){
                const uint8_t *page=s->read_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES;
                for(uint32_t j=starts[i];status==FG_OK&&j<starts[i+1u];j++){
                    uint8_t *destination=arena+((uint64_t)missing[j].query*
                        FG_QSA_SELECTED_TOKENS+(uint64_t)missing[j].index*
                        FG_Q38_QSA_COMPRESS_RATIO)*FG_Q38_QSA_TOKEN_RECORD_BYTES;
                    memcpy(destination,page,FG_QSA_PAGE_RECORD_BYTES);
                }
                uint32_t cache_slot=0;bool hit=false;fg_error cache_error={0};
                fg_status cached=qsa_cache_acquire_soft(s,layer,fetch[i],
                    &cache_slot,&hit,&cache_error);
                if(status==FG_OK&&cached==FG_OK&&cache_slot!=UINT32_MAX)
                    status=fg_vk_tensor_write(s->cache_records,
                        (uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,page,
                        FG_QSA_PAGE_RECORD_BYTES,err);
                else if(status==FG_OK&&cached!=FG_OK){status=cached;if(err)*err=cache_error;}
            }
            at=end;
        }
        if(status==FG_OK)status=fg_vk_begin(vk,err);
    }
    if(status==FG_OK){
        uint64_t offset=(uint64_t)first_query*6144u*4u;
        status=fg_vk_tensor_view_rebind(s->query_view,s->query,offset,
            (uint64_t)queries*6144u*4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->gate_view,s->gate,offset,
            (uint64_t)queries*6144u*4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->attention_view,s->attention,
            offset,(uint64_t)queries*6144u*4u,err);
        if(status==FG_OK)status=fg_vk_qsa_attention_split_batch(vk,s->batch_partials,
            s->batch_records,s->query_view,s->batch_counts,queries,
            FG_QSA_SELECTED_TOKENS,6144u,FG_QSA_ATTENTION_SPLITS,err);
        if(status==FG_OK)status=fg_vk_qsa_attention_merge_batch(vk,s->attention_view,
            s->batch_partials,s->gate_view,queries,6144u,FG_QSA_ATTENTION_SPLITS,err);
    }
    return status;
}

static bool qsa_prefill_profile_enabled(void){
    const char *value=getenv("FG_PREFILL_PROFILE");
    return value&&*value&&strcmp(value,"0")!=0;
}

static fg_status attend_prefill_tiles(fg_qsa_session *s,uint32_t slot,
    uint32_t first_token,uint32_t token_count,fg_error *err){
    fg_vk_context *vk=fg_model_vk(s->model);
    bool profile=qsa_prefill_profile_enabled();
    double t_begin=0,t_commit=0,t_select=0,t_gather=0,t_attend=0;
    double select_ms=0.0,gather_ms=0.0,attend_ms=0.0;
    if(profile)t_begin=qsa_now_ms();
    fg_status status=commit_prefill_records(s,slot,first_token,token_count,err);
    if(profile)t_commit=qsa_now_ms();
    uint32_t counts[FG_PREFILL_MAX_TOKENS];
    if(status==FG_OK)status=ensure_prefill_batch(s,err);
    if(profile)t_select=qsa_now_ms();
    /* One selection sweep for the whole microbatch; ids cross to the host once
     * so every tile resolves its page-cache slots without further GPU reads. */
    if(status==FG_OK)status=select_prefill_batch(s,slot,first_token+1u,0u,token_count,
        s->select_ids,FG_QSA_MAX_SELECTED_BLOCKS,counts,err);
    if(profile)t_gather=qsa_now_ms();
    for(uint32_t first=0;status==FG_OK&&first<token_count;
        first+=FG_QSA_PREFILL_BATCH_QUERIES){
        uint32_t queries=token_count-first;
        if(queries>FG_QSA_PREFILL_BATCH_QUERIES)queries=FG_QSA_PREFILL_BATCH_QUERIES;
        status=prefill_tile_attention(s,slot,first,first_token+first+1u,
            queries,s->select_ids+(uint64_t)first*FG_QSA_MAX_SELECTED_BLOCKS,
            FG_QSA_MAX_SELECTED_BLOCKS,counts+first,err);
    }
    if(profile)t_attend=qsa_now_ms();
    /* The per-tile helpers fence their own staging writes; one final fence
     * hands the completed attention back to the caller. */
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    if(profile){double t_end=qsa_now_ms();
        select_ms=t_gather-t_select;gather_ms=0.0;attend_ms=t_attend-t_gather;
        fprintf(stderr,"QSA_PREFILL_TRACE layer=%u first=%u tokens=%u commit_ms=%.1f "
            "select_ms=%.1f gather_ms=%.1f attend_ms=%.1f total_ms=%.1f\n",
            s->layers[slot],first_token,token_count,t_commit-t_begin,
            select_ms,gather_ms,attend_ms,t_end-t_begin);}
    return status;
}

/* The ring's state-backed owner computes prefill through the page cache, then
 * persists newly completed pages once per layer instead of once per token.
 * Page order is contiguous from the layer frontier, so complete runs go out as
 * one batched uring write; only the boundary pages need a single-page write. */
static fg_status persist_prefill_state(fg_qsa_session *s,uint32_t slot,
    uint32_t first_token,uint32_t token_count,fg_error *err){
    if(!s->cache||!s->state)return FG_OK;
    /* Decode commits records into the page cache without write-through, so a
     * continuation prefill starts with the state file behind the session
     * frontier.  Persist that decoded range from the cache first; the pages
     * are pinned, so the lookup below cannot miss them. */
    uint32_t persisted=fg_qsa_state_layer_tokens(s->state,slot);
    if(persisted>first_token){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "QSA state frontier is ahead of the prefill range");
        return FG_ERR_MISMATCH;
    }
    while(persisted<first_token){
        uint32_t step=first_token-persisted;
        if(step>FG_QSA_MAX_SELECTED_BLOCKS*FG_Q38_QSA_COMPRESS_RATIO)
            step=FG_QSA_MAX_SELECTED_BLOCKS*FG_Q38_QSA_COMPRESS_RATIO;
        fg_status flush=persist_prefill_state(s,slot,persisted,step,err);
        if(flush!=FG_OK)return flush;
        persisted+=step;
    }
    uint32_t layer=s->layers[slot],end=first_token+token_count;
    const uint8_t *cache=(const uint8_t *)fg_vk_tensor_map(s->cache_records);
    if(!cache){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"QSA record cache is not host-mapped");
        return FG_ERR_UNAVAILABLE;
    }
    fg_status status=FG_OK;
    uint32_t block=first_token/FG_Q38_QSA_COMPRESS_RATIO;
    uint32_t leading=first_token%FG_Q38_QSA_COMPRESS_RATIO;
    if(leading){
        uint32_t committed=end-block*FG_Q38_QSA_COMPRESS_RATIO;
        if(committed>FG_Q38_QSA_COMPRESS_RATIO)committed=FG_Q38_QSA_COMPRESS_RATIO;
        uint32_t cache_slot=0;
        if(!fg_qsa_page_cache_lookup(s->cache,layer,block,&cache_slot)){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA persist lost the leading cache page");
            return FG_ERR_MISMATCH;
        }
        status=fg_qsa_state_write_block(s->state,slot,block,
            cache+(uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,committed,err);
        fg_qsa_session_page_published(s,layer,block);
        if(status!=FG_OK||committed<FG_Q38_QSA_COMPRESS_RATIO)return status;
        block++;
    }
    uint32_t complete=end/FG_Q38_QSA_COMPRESS_RATIO;
    if(complete>block){
        uint32_t count=complete-block;
        if(count>FG_QSA_MAX_SELECTED_BLOCKS){
            fg_error_set(err,FG_ERR_LIMIT,"QSA persist batch exceeds staging capacity");
            return FG_ERR_LIMIT;
        }
        uint32_t blocks[FG_QSA_MAX_SELECTED_BLOCKS];
        for(uint32_t i=0;i<count;i++){
            uint32_t cache_slot=0;
            if(!fg_qsa_page_cache_lookup(s->cache,layer,block+i,&cache_slot)){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA persist lost a complete cache page");
                return FG_ERR_MISMATCH;
            }
            memcpy(s->read_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,
                cache+(uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,
                FG_QSA_PAGE_RECORD_BYTES);
            blocks[i]=block+i;
        }
        status=fg_qsa_state_write_blocks(s->state,slot,blocks,count,s->read_records,err);
        for(uint32_t i=0;i<count;i++)fg_qsa_session_page_published(s,layer,block+i);
        if(status!=FG_OK)return status;
        block+=count;
    }
    uint32_t tail=end%FG_Q38_QSA_COMPRESS_RATIO;
    if(tail){
        uint32_t cache_slot=0;
        if(!fg_qsa_page_cache_lookup(s->cache,layer,block,&cache_slot)){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA persist lost the trailing cache page");
            return FG_ERR_MISMATCH;
        }
        status=fg_qsa_state_write_block(s->state,slot,block,
            cache+(uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,tail,err);
        fg_qsa_session_page_published(s,layer,block);
    }
    return status;
}

fg_status fg_qsa_session_prefill(fg_qsa_session *s,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint32_t token_count,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;if(!s||signed_slot<0||!positions||!hidden||!output||!token_count||token_count>s->max_tokens||token_count>s->max_context||first_token>s->max_context-token_count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prefill arguments");return FG_ERR_ARGUMENT;}uint32_t slot=(uint32_t)signed_slot;if(first_token!=s->committed[slot]){fg_error_set(err,FG_ERR_MISMATCH,"QSA prefill range does not start at committed state");return FG_ERR_MISMATCH;}
    fg_vk_tensor *qw=layer_weight(s,layer,"attn_q.weight",err),*kw=layer_weight(s,layer,"attn_k.weight",err),*vw=layer_weight(s,layer,"attn_v.weight",err),*qn=layer_weight(s,layer,"attn_q_norm.weight",err),*kn=layer_weight(s,layer,"attn_k_norm.weight",err),*ow=layer_weight(s,layer,"attn_output.weight",err),*iqw=layer_weight(s,layer,"indexer.q_proj.weight",err),*ikw=layer_weight(s,layer,"indexer.k_proj.weight",err),*iqn=layer_weight(s,layer,"indexer.q_norm.weight",err);if(!qw||!kw||!vw||!qn||!kn||!ow||!iqw||!ikw||!iqn)return FG_ERR_MISMATCH;
    /* Materialize every index segment the range will touch before the first
     * batch records; lazy allocation must not race the residency canary. */
    fg_status ensure_status=ensure_index_segments_for_range(s,slot,first_token,
                                                            token_count,err);
    if(ensure_status!=FG_OK)return ensure_status;
    fg_vk_context *vk=fg_model_vk(s->model);fg_status status=fg_qsa_submit_host_reads(vk,err);
    uint32_t *resident_positions=status==FG_OK?fg_vk_tensor_map(s->positions):NULL;for(uint32_t i=0;status==FG_OK&&i<token_count;i++){uint32_t token=first_token+i;uint32_t *resident=resident_positions+(uint64_t)token*3u;const uint32_t *position=positions+(uint64_t)i*3u;if(!s->position_written[token]){memcpy(resident,position,FG_Q38_QSA_POSITION_BYTES);s->position_written[token]=1u;}else if(memcmp(resident,position,FG_Q38_QSA_POSITION_BYTES)!=0){fg_error_set(err,FG_ERR_MISMATCH,"QSA layers received inconsistent MRoPE positions");return FG_ERR_MISMATCH;}}
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
    if(s->cache){
        status=attend_prefill_tiles(s,slot,first_token,token_count,err);
    }else{
        const uint8_t *keys=fg_vk_tensor_map(s->key_q8),*values=fg_vk_tensor_map(s->value_q4),*index_keys=fg_vk_tensor_map(s->index_key_q8);
        status=fg_vk_begin(vk,err);
        for(uint32_t i=0;status==FG_OK&&i<token_count;i++){
            status=fg_vk_tensor_view_rebind(s->index_query_view,s->index_query,(uint64_t)i*512u*4u,512u*4u,err);
            if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->query_view,s->query,(uint64_t)i*6144u*4u,6144u*4u,err);
            if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->gate_view,s->gate,(uint64_t)i*6144u*4u,6144u*4u,err);
            if(status==FG_OK)status=fg_vk_tensor_view_rebind(s->attention_view,s->attention,(uint64_t)i*6144u*4u,6144u*4u,err);
            if(status==FG_OK)status=commit_and_attend(s,slot,first_token+i,positions+(uint64_t)i*3u,keys+(uint64_t)i*FG_Q38_QSA_KEY_BYTES,values+(uint64_t)i*FG_Q38_QSA_VALUE_BYTES,index_keys+(uint64_t)i*FG_Q38_QSA_INDEX_KEY_BYTES,s->index_query_view,s->query_view,s->gate_view,s->attention_view,err);
        }
    }
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,s->output,ow,s->attention,6144u,FG_HIDDEN_SIZE,token_count,1.0f,err);
    if(status==FG_OK)status=fg_qsa_submit_host_reads(vk,err);
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status==FG_OK)status=persist_prefill_state(s,slot,first_token,token_count,err);
    if(status==FG_OK){s->committed[slot]=first_token+token_count;*output=s->output;}
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

/* Mirror warm insertion: host-write fetched complete pages into the record
 * cache.  Called between Vulkan batches from the coordinator loop, so no
 * gather may be reading cache_records at the same time. */
fg_status fg_qsa_session_warm_pages(fg_qsa_session *s,uint32_t layer,
                                    const uint32_t *blocks,const uint8_t *records,
                                    uint32_t page_count,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;
    if(!s||signed_slot<0||!blocks||!records||!page_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA mirror warm request");
        return FG_ERR_ARGUMENT;
    }
    if(!s->cache||!s->cache_records){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"QSA mirror warm requires a record cache");
        return FG_ERR_UNAVAILABLE;
    }
    fg_status status=FG_OK;
    for(uint32_t i=0;status==FG_OK&&i<page_count;i++){
        uint32_t cache_slot=0;bool hit=false;
        status=qsa_cache_acquire_soft(s,layer,blocks[i],&cache_slot,&hit,err);
        if(status==FG_OK&&cache_slot!=UINT32_MAX)
            status=fg_vk_tensor_write(s->cache_records,
                (uint64_t)cache_slot*FG_QSA_PAGE_RECORD_BYTES,
                records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,FG_QSA_PAGE_RECORD_BYTES,err);
    }
    return status;
}

bool fg_qsa_session_page_cached(fg_qsa_session *s,uint32_t layer,uint32_t block){
    uint32_t cache_slot=0;
    return s&&s->cache&&layer_slot(s,layer)>=0&&
        fg_qsa_page_cache_lookup(s->cache,layer,block,&cache_slot);
}

/* Authoritative read for state-backed sessions: the block records come from
 * the session's own state file rather than a pinned page-cache entry. */
fg_status fg_qsa_session_state_records(fg_qsa_session *s,uint32_t layer,uint32_t block,
                                       uint8_t *records,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;
    if(!s||signed_slot<0||!records||!s->state){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state page lookup");
        return FG_ERR_ARGUMENT;
    }
    uint32_t committed=0;
    fg_status status=fg_qsa_state_read_block(s->state,(uint32_t)signed_slot,block,
                                             records,&committed,err);
    if(status==FG_OK&&committed!=FG_Q38_QSA_COMPRESS_RATIO){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA state page is incomplete");
        status=FG_ERR_MISMATCH;
    }
    return status;
}

/* Batched form of the same read: one uring submission for up to
 * FG_QSA_MAX_SELECTED_BLOCKS pages keeps a decode fetch off the per-page pread
 * path. */
fg_status fg_qsa_session_state_records_batch(fg_qsa_session *s,uint32_t layer,
    const uint32_t *blocks,uint32_t page_count,uint8_t *records,fg_error *err){
    int signed_slot=s?layer_slot(s,layer):-1;
    if(!s||signed_slot<0||!blocks||!records||!page_count||!s->state){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state page batch lookup");
        return FG_ERR_ARGUMENT;
    }
    if(page_count>FG_QSA_MAX_SELECTED_BLOCKS){
        fg_error_set(err,FG_ERR_LIMIT,"QSA state page batch exceeds staging capacity");
        return FG_ERR_LIMIT;
    }
    fg_status status=fg_qsa_state_read_blocks(s->state,(uint32_t)signed_slot,blocks,
        page_count,records,s->state_committed,err);
    for(uint32_t i=0;status==FG_OK&&i<page_count;i++)
        if(s->state_committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA state page is incomplete");
            status=FG_ERR_MISMATCH;
        }
    return status;
}
