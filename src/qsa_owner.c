#include "fg_qsa_owner.h"
#include "fg_runtime.h"

#include <string.h>

static bool owned_qsa_layer(const fg_manifest *manifest,uint32_t rank,uint32_t layer){
    return manifest&&rank<FG_RANK_COUNT&&layer<FG_LAYER_COUNT&&(layer&3u)==3u&&
           manifest->layer_owner[layer]==rank;
}

void fg_qsa_owner_guard_init(fg_qsa_owner_guard *guard,uint32_t rank){
    if(!guard)return;
    memset(guard,0,sizeof(*guard));guard->rank=(uint8_t)rank;
}

fg_status fg_qsa_owner_guard_begin(fg_qsa_owner_guard *guard,const fg_manifest *manifest,
                                   const fg_session_identity *identity,uint64_t request_id,
                                   const fg_owner_session_control *control,fg_error *err){
    if(!guard||!manifest||!identity||guard->rank>=FG_RANK_COUNT||!request_id){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA owner session begin");
        return FG_ERR_ARGUMENT;
    }
    uint32_t layers=0;for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        layers+=manifest->layer_owner[layer]==guard->rank;
    if(layers!=6u){
        fg_error_set(err,FG_ERR_MISMATCH,"rank %u owns %u QSA layers, expected 6",
                     guard->rank,layers);return FG_ERR_MISMATCH;
    }
    if(guard->active&&request_id<=guard->session_nonce){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or duplicate QSA owner session begin");
        return FG_ERR_MISMATCH;
    }
    if(manifest->protocol_version==FG_PROTOCOL_MIN_VERSION){
        if(control){fg_error_set(err,FG_ERR_FORMAT,"legacy QSA owner begin has a control payload");return FG_ERR_FORMAT;}
    }else{
        if(!control||control->version!=FG_OWNER_SESSION_CONTROL_VERSION||
           control->operation!=FG_OWNER_SESSION_BEGIN||control->rank!=guard->rank||
           control->position_mode!=(fg_position_mode)manifest->session.position_mode||
           control->session_nonce!=request_id||
           memcmp(control->identity_sha256,identity->identity_sha256,32u)||
           memcmp(control->state_format_sha256,
                  manifest->session.rank_state_format_sha256[guard->rank],32u)){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA owner session identity or rank mismatch");
            return FG_ERR_MISMATCH;
        }
        fg_runtime_options profile={
            .logical_context_tokens=control->logical_context_tokens,
            .gpu_index_tokens=control->gpu_index_tokens,
            .qsa_hot_tokens=control->qsa_hot_tokens,
            .qsa_page_cache_bytes=control->qsa_page_cache_bytes
        };
        fg_status status=fg_runtime_profile_validate(&profile,manifest->native_context,err);
        if(status!=FG_OK)return status;
    }
    uint64_t generation=guard->generation+1u;uint8_t rank=guard->rank;
    memset(guard,0,sizeof(*guard));guard->rank=rank;guard->session_nonce=request_id;
    guard->generation=generation;guard->context_limit=control?
        control->logical_context_tokens:manifest->session.logical_context_tokens;
    guard->position_mode=(fg_position_mode)manifest->session.position_mode;guard->active=true;
    return FG_OK;
}

static fg_status validate_route(const fg_qsa_owner_guard *guard,const fg_manifest *manifest,
                                uint64_t request_id,uint32_t sequence,uint32_t layer,
                                uint32_t first_token,uint32_t token_count,uint8_t source_rank,
                                uint8_t destination_rank,fg_position_mode position_mode,
                                const float *hidden,fg_error *err){
    if(!guard||!manifest||!hidden||!token_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA owner work validation");
        return FG_ERR_ARGUMENT;
    }
    if(!guard->active||request_id!=guard->session_nonce){
        fg_error_set(err,FG_ERR_MISMATCH,"stale QSA owner work session");
        return FG_ERR_MISMATCH;
    }
    if(source_rank!=0u||destination_rank!=guard->rank||
       !owned_qsa_layer(manifest,guard->rank,layer)){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA work is not routed to its manifest owner");
        return FG_ERR_MISMATCH;
    }
    if(position_mode!=guard->position_mode||position_mode!=FG_POSITION_TEXT){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA work position mode mismatch");
        return FG_ERR_MISMATCH;
    }
    if(first_token!=guard->next_token[layer]||first_token>guard->context_limit||
       token_count>guard->context_limit-first_token||
       sequence!=first_token*FG_LAYER_COUNT+layer){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or out-of-order QSA work for layer %u",layer);
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_qsa_owner_guard_validate_decode(const fg_qsa_owner_guard *guard,
                                             const fg_manifest *manifest,uint64_t request_id,
                                             uint32_t sequence,const fg_qsa_block_work *work,
                                             fg_error *err){
    if(!work){
        fg_error_set(err,FG_ERR_FORMAT,"invalid QSA decode work");
        return FG_ERR_FORMAT;
    }
    fg_status status=validate_route(guard,manifest,request_id,sequence,work->layer,
        work->token_index,1u,work->source_rank,work->destination_rank,work->position_mode,
        work->hidden,err);
    for(uint32_t axis=0;status==FG_OK&&axis<3u;axis++)if(work->position[axis]!=work->token_index){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA decode text position does not match token index");
        status=FG_ERR_MISMATCH;
    }
    return status;
}

fg_status fg_qsa_owner_guard_validate_prefill(const fg_qsa_owner_guard *guard,
                                              const fg_manifest *manifest,uint64_t request_id,
                                              uint32_t sequence,
                                              const fg_qsa_block_prefill_work *work,
                                              fg_error *err){
    if(!work||!work->positions){
        fg_error_set(err,FG_ERR_FORMAT,"invalid QSA prefill work");
        return FG_ERR_FORMAT;
    }
    fg_status status=validate_route(guard,manifest,request_id,sequence,work->layer,
        work->first_token,work->token_count,work->source_rank,work->destination_rank,
        work->position_mode,work->hidden,err);
    for(uint32_t offset=0;status==FG_OK&&offset<work->token_count;offset++){
        uint32_t expected=work->first_token+offset;
        for(uint32_t axis=0;axis<3u;axis++)if(work->positions[(uint64_t)offset*3u+axis]!=expected){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "QSA prefill text position does not match contiguous token index");
            status=FG_ERR_MISMATCH;break;
        }
    }
    return status;
}

fg_status fg_qsa_owner_guard_commit(fg_qsa_owner_guard *guard,uint32_t layer,
                                    uint32_t first_token,uint32_t token_count,
                                    fg_error *err){
    if(!guard||!guard->active||layer>=FG_LAYER_COUNT||!token_count||
       first_token!=guard->next_token[layer]||first_token>guard->context_limit||
       token_count>guard->context_limit-first_token){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid QSA owner progress commit");
        return FG_ERR_MISMATCH;
    }
    guard->next_token[layer]=first_token+token_count;return FG_OK;
}

fg_status fg_qsa_owner_validate_decode_result(const fg_manifest *manifest,uint32_t owner_rank,
                                              uint32_t sequence,
                                              const fg_qsa_block_result *result,fg_error *err){
    if(!result||!owned_qsa_layer(manifest,owner_rank,result->layer)||
       result->source_rank!=owner_rank||result->destination_rank!=0u||
       !result->hidden||
       sequence!=result->token_index*FG_LAYER_COUNT+result->layer){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted QSA decode result");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_qsa_owner_validate_prefill_result(const fg_manifest *manifest,uint32_t owner_rank,
                                               uint32_t sequence,uint16_t token_count,
                                               const fg_qsa_block_prefill_result *result,
                                               fg_error *err){
    if(!result||!owned_qsa_layer(manifest,owner_rank,result->layer)||
       result->source_rank!=owner_rank||result->destination_rank!=0u||
       result->token_count!=token_count||!result->hidden||
       sequence!=result->first_token*FG_LAYER_COUNT+result->layer){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted QSA prefill result");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

static fg_status validate_page_route(const fg_qsa_owner_guard *guard,const fg_manifest *manifest,
                                     uint64_t request_id,uint8_t source,uint8_t destination,
                                     fg_error *err){
    if(!guard||!manifest||!guard->active||request_id!=guard->session_nonce){
        fg_error_set(err,FG_ERR_MISMATCH,"stale QSA page-service session");
        return FG_ERR_MISMATCH;
    }
    if(source!=0u||destination!=guard->rank){
        fg_error_set(err,FG_ERR_MISMATCH,"misrouted QSA page-service request");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_qsa_owner_guard_accept_append(fg_qsa_owner_guard *guard,
                                           const fg_manifest *manifest,uint64_t request_id,
                                           const fg_qsa_page_batch *batch,fg_error *err){
    if(!batch||!batch->pages){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA page append guard input");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=validate_page_route(guard,manifest,request_id,batch->source_rank,
                                         batch->destination_rank,err);
    if(status!=FG_OK)return status;
    if(batch->batch_id!=guard->next_append_batch){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or out-of-order QSA page append batch");
        return FG_ERR_MISMATCH;
    }
    uint32_t frontier[FG_LAYER_COUNT];uint16_t pages_per_layer[FG_LAYER_COUNT]={0};
    memcpy(frontier,guard->next_token,sizeof(frontier));
    for(uint32_t i=0;i<batch->page_count;i++){
        const fg_qsa_page *page=&batch->pages[i];
        uint64_t first=(uint64_t)page->block*FG_Q38_QSA_COMPRESS_RATIO;
        if(!page->records||!owned_qsa_layer(manifest,guard->rank,page->layer)||
           ++pages_per_layer[page->layer]>FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES||
           first!=frontier[page->layer]||
           first+FG_Q38_QSA_COMPRESS_RATIO>guard->context_limit){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "non-contiguous or misowned QSA page append entry %u",i);
            return FG_ERR_MISMATCH;
        }
        frontier[page->layer]+=(uint32_t)FG_Q38_QSA_COMPRESS_RATIO;
    }
    memcpy(guard->next_token,frontier,sizeof(frontier));guard->next_append_batch++;
    return FG_OK;
}

fg_status fg_qsa_owner_guard_accept_fetch(fg_qsa_owner_guard *guard,
                                          const fg_manifest *manifest,uint64_t request_id,
                                          const fg_qsa_page_batch *batch,fg_error *err){
    if(!batch||!batch->pages){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA page fetch guard input");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=validate_page_route(guard,manifest,request_id,batch->source_rank,
                                         batch->destination_rank,err);
    if(status!=FG_OK)return status;
    if(batch->batch_id!=guard->next_fetch_batch){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or out-of-order QSA page fetch batch");
        return FG_ERR_MISMATCH;
    }
    uint32_t layer=batch->pages[0].layer;
    for(uint32_t i=0;i<batch->page_count;i++){
        const fg_qsa_page *page=&batch->pages[i];
        uint64_t end=((uint64_t)page->block+1u)*FG_Q38_QSA_COMPRESS_RATIO;
        if(page->layer!=layer||!owned_qsa_layer(manifest,guard->rank,page->layer)||
           end>guard->next_token[page->layer]){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "future or misowned QSA page fetch entry %u",i);
            return FG_ERR_MISMATCH;
        }
    }
    guard->next_fetch_batch++;return FG_OK;
}

fg_status fg_qsa_owner_guard_accept_barrier(const fg_qsa_owner_guard *guard,
                                            uint64_t request_id,
                                            const fg_qsa_page_barrier *barrier,fg_error *err){
    if(!barrier){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA page barrier guard input");
        return FG_ERR_ARGUMENT;
    }
    if(!guard||!guard->active||request_id!=guard->session_nonce||
       barrier->source_rank!=0u||barrier->destination_rank!=guard->rank||
       barrier->batch_id!=guard->next_append_batch){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or out-of-order QSA page barrier");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}
