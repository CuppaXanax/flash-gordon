#include "fg_mtp_native.h"
#include "fg_mtp_embedding.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct native_session {
    fg_mtp_head *head;
    fg_vk_tensor *input,*embedding;
    const fg_vk_tensor *draft_hyper;
    uint32_t last_token,previous_token,verify_first,verify_count,draft_first,drafted;
    uint32_t warm_start,prefix_end;
    bool last_valid,previous_valid;
    float last[FG_HYPER_WIDTH],previous[FG_HYPER_WIDTH];
    float verify[FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH];
} native_session;

struct fg_mtp_native {
    fg_vk_context *vk;
    const fg_manifest *manifest;
    uint32_t rank,native_owner,context_tokens,window_tokens;
    bool enabled,final_owner;
    char overlay[FG_MTP_OVERLAY_PATH_MAX+1u];
    char embedding_file[FG_MTP_OVERLAY_PATH_MAX+1u];
    fg_mtp_embedding *embedding;
    native_session session[2];
};

static fg_status mismatch(fg_error *err,const char *what){
    fg_error_set(err,FG_ERR_MISMATCH,"native MTP %s",what);return FG_ERR_MISMATCH;
}
static fg_status warm(fg_mtp_native *n,native_session *s,uint32_t cell,
    const float *hyper,uint32_t token,fg_error *err);
static fg_status ensure_session(fg_mtp_native *n,uint32_t slot,fg_error *err){
    native_session *s=&n->session[slot];
    fg_status status=FG_OK;
    if(!s->head){
        if(slot)status=fg_mtp_head_fork(&s->head,n->session[0].head,slot,err);
        else if(n->native_owner==n->manifest->layer_owner[FG_LAYER_COUNT-1u])
            status=fg_mtp_head_open_windowed(&s->head,n->vk,n->manifest,n->rank,
                n->overlay,n->context_tokens,n->window_tokens,err);
        else status=fg_mtp_head_open_owner_windowed(&s->head,n->vk,n->manifest,n->rank,
            n->native_owner,n->overlay,n->context_tokens,n->window_tokens,err);
    }
    if(status==FG_OK&&!s->input)
        status=fg_vk_tensor_create(n->vk,FG_HYPER_WIDTH*4u,&s->input,err);
    if(status==FG_OK&&!s->embedding)
        status=fg_vk_tensor_create(n->vk,FG_HIDDEN_SIZE*4u,&s->embedding,err);
    return status;
}
fg_status fg_mtp_native_create(fg_mtp_native **out,fg_vk_context *vk,
    const fg_manifest *manifest,uint32_t rank,fg_error *err){
    if(!out||!vk||!manifest||rank>=FG_RANK_COUNT)
        return mismatch(err,"create arguments");
    *out=calloc(1,sizeof(**out));
    if(!*out){fg_error_set(err,FG_ERR_OOM,"allocate native MTP state");return FG_ERR_OOM;}
    (*out)->vk=vk;(*out)->manifest=manifest;(*out)->rank=rank;
    (*out)->native_owner=manifest->layer_owner[FG_LAYER_COUNT-1u];
    (*out)->final_owner=manifest->layer_owner[FG_LAYER_COUNT-1u]==rank;
    return FG_OK;
}
void fg_mtp_native_close(fg_mtp_native *n){
    if(!n)return;
    for(uint32_t slot=0;slot<2u;slot++){
        native_session *s=&n->session[slot];
        fg_mtp_head_close(s->head);fg_vk_tensor_destroy(s->input);
        fg_vk_tensor_destroy(s->embedding);
    }
    fg_mtp_embedding_close(n->embedding);free(n);
}
bool fg_mtp_native_enabled(const fg_mtp_native *n){return n&&n->enabled;}
uint32_t fg_mtp_native_owner(const fg_mtp_native *n){return n?n->native_owner:UINT32_MAX;}
fg_status fg_mtp_native_reset(fg_mtp_native *n,uint32_t slot,fg_error *err){
    if(!n||slot>=2u)return mismatch(err,"reset slot");
    native_session *s=&n->session[slot];
    s->last_valid=false;s->previous_valid=false;
    s->verify_count=0u;s->drafted=0u;s->draft_hyper=NULL;s->prefix_end=0u;
    return s->head?fg_mtp_head_reset(s->head,0u,err):FG_OK;
}
static fg_status observe(fg_mtp_native *n,uint32_t slot,uint32_t first,uint32_t count,
    const float *hyper,fg_error *err){
    if(!n||!n->enabled||!n->final_owner)return FG_OK;
    if(slot>=2u||!hyper||!count||count>n->context_tokens||
       first>n->context_tokens-count)
        return mismatch(err,"target observation range");
    return FG_OK;
}
fg_status fg_mtp_native_observe_prefill(fg_mtp_native *n,uint32_t slot,
    uint32_t first,uint32_t count,const float *hyper,const uint32_t *next_tokens,
    fg_error *err){
    fg_status status=observe(n,slot,first,count,hyper,err);
    if(status!=FG_OK||!n||!n->enabled||!n->final_owner)return status;
    native_session *s=&n->session[slot];
    if(!s->prefix_end||first+count>s->prefix_end||!next_tokens)
        return mismatch(err,"prefill without prefix plan");
    uint32_t lo=first>s->warm_start?first:s->warm_start;
    uint32_t hi=first+count;
    if(hi>=s->prefix_end)hi=s->prefix_end-1u;
    for(uint32_t cell=lo;status==FG_OK&&cell<hi;cell++)
        status=warm(n,s,cell,hyper+(uint64_t)(cell-first)*FG_HYPER_WIDTH,
                    next_tokens[cell-first],err);
    if(status!=FG_OK)return status;
    memcpy(s->last,hyper+(uint64_t)(count-1u)*FG_HYPER_WIDTH,sizeof(s->last));
    s->last_token=first+count-1u;s->last_valid=true;s->previous_valid=false;
    return FG_OK;
}
fg_status fg_mtp_native_observe_decode(fg_mtp_native *n,uint32_t slot,
    uint32_t first,const float *hyper,fg_error *err){
    fg_status status=observe(n,slot,first,1u,hyper,err);
    if(status!=FG_OK||!n||!n->enabled||!n->final_owner)return status;
    native_session *s=&n->session[slot];
    s->previous_valid=s->last_valid&&s->last_token+1u==first;
    if(s->previous_valid){
        memcpy(s->previous,s->last,sizeof(s->last));s->previous_token=s->last_token;
    }
    memcpy(s->last,hyper,sizeof(s->last));s->last_token=first;s->last_valid=true;
    return FG_OK;
}
fg_status fg_mtp_native_observe_verify(fg_mtp_native *n,uint32_t slot,
    uint32_t first,uint32_t count,const float *hyper,fg_error *err){
    fg_status status=observe(n,slot,first,count,hyper,err);
    if(status!=FG_OK||!n||!n->enabled||!n->final_owner)return status;
    if(count>FG_MTP_MAX_VERIFY)return mismatch(err,"verification width");
    native_session *s=&n->session[slot];
    memcpy(s->verify,hyper,(size_t)count*FG_HYPER_WIDTH*sizeof(float));
    s->verify_first=first;s->verify_count=count;return FG_OK;
}
static fg_status warm(fg_mtp_native *n,native_session *s,uint32_t cell,
    const float *hyper,uint32_t token,fg_error *err){
    float embedding[FG_HIDDEN_SIZE];
    fg_status status=fg_mtp_embedding_row(n->embedding,token,embedding,err);
    if(status==FG_OK)status=fg_vk_tensor_write(s->input,0u,hyper,FG_HYPER_WIDTH*4u,err);
    if(status==FG_OK)status=fg_vk_tensor_write(s->embedding,0u,embedding,sizeof(embedding),err);
    if(status==FG_OK)status=fg_mtp_head_warm_replay(s->head,s->input,s->embedding,cell,err);
    return status;
}
static fg_status configure(fg_mtp_native *n,const fg_mtp_native_work *w,fg_error *err){
    uint32_t selected=0u;
    fg_status placement=fg_mtp_native_owner_resolve(n->manifest,w->owner_rank_plus_one,
                                                  &selected,err);
    if(placement!=FG_OK)return placement;
    if(n->enabled){
        if(strcmp(n->overlay,w->overlay)||strcmp(n->embedding_file,w->embedding_file)||
           n->context_tokens!=w->argument||
           n->window_tokens!=w->limit||n->native_owner!=selected)
            return mismatch(err,"configuration changed");
        return FG_OK;
    }
    snprintf(n->overlay,sizeof(n->overlay),"%s",w->overlay);
    snprintf(n->embedding_file,sizeof(n->embedding_file),"%s",w->embedding_file);
    n->context_tokens=w->argument;n->window_tokens=w->limit;
    n->native_owner=selected;n->final_owner=selected==n->rank;
    fg_status status=FG_OK;
    if(n->final_owner)status=fg_mtp_embedding_open(&n->embedding,n->manifest,n->embedding_file,err);
    if(status==FG_OK&&n->final_owner)status=ensure_session(n,0u,err);
    if(status==FG_OK)n->enabled=true;
    return status;
}
fg_status fg_mtp_native_run(fg_mtp_native *n,uint32_t slot,
    const fg_mtp_native_work *w,float hidden[FG_HIDDEN_SIZE],
    fg_mtp_native_ack *ack,fg_error *err){
    if(!n||slot>=2u||!w||!ack)return mismatch(err,"operation arguments");
    uint8_t validated[FG_MTP_NATIVE_WORK_MAX_BYTES];uint32_t bytes=0u;
    fg_status status=fg_mtp_native_work_encode(validated,sizeof(validated),&bytes,w,err);
    if(status!=FG_OK)return status;
    if(w->operation!=FG_MTP_NATIVE_CONFIG&&
       (w->first_token>=n->context_tokens||
        (w->operation==FG_MTP_NATIVE_PREFIX&&w->limit>n->context_tokens)||
        (w->operation==FG_MTP_NATIVE_FINISH&&w->count>n->context_tokens-w->first_token)))
        return mismatch(err,"operation exceeds configured context");
    native_session *s=&n->session[slot];
    if(w->operation==FG_MTP_NATIVE_CONFIG)status=configure(n,w,err);
    else if(!n->enabled||!n->final_owner)return mismatch(err,"operation on inactive owner");
    else status=ensure_session(n,slot,err);
    if(status!=FG_OK)return status;
    switch(w->operation){
        case FG_MTP_NATIVE_CONFIG:break;
        case FG_MTP_NATIVE_PREPARE:
            for(uint32_t i=0;status==FG_OK&&i<w->argument;i++)
                status=ensure_session(n,i,err);
            if(status==FG_OK)status=fg_mtp_head_check_residency(s->head,err);
            break;
        case FG_MTP_NATIVE_PREFIX:{
            s->draft_hyper=NULL;s->drafted=0u;s->verify_count=0u;
            s->warm_start=w->argument;s->prefix_end=w->limit;
            bool reuse=w->count&&s->last_valid&&s->last_token+1u==w->first_token&&
                fg_mtp_head_frontier(s->head)==w->first_token-1u&&
                w->first_token-1u>=s->warm_start;
            if(reuse)status=warm(n,s,w->first_token-1u,s->last,w->tokens[0],err);
            else{
                uint32_t start=w->first_token>s->warm_start?w->first_token:s->warm_start;
                status=fg_mtp_head_reset(s->head,start,err);
            }
            break;
        }
        case FG_MTP_NATIVE_DRAFT:{
            if(!hidden)return mismatch(err,"draft output storage");
            if(w->argument==0u){
                if(!s->last_valid||s->last_token+1u!=w->first_token)
                    return mismatch(err,"draft target frontier");
                if(fg_mtp_head_frontier(s->head)!=w->first_token-1u)
                    return mismatch(err,"draft KV frontier");
                status=fg_vk_tensor_write(s->input,0u,s->last,sizeof(s->last),err);
                s->draft_hyper=s->input;s->draft_first=w->first_token;s->drafted=0u;
            }else if(!s->draft_hyper||s->drafted!=w->argument||
                     w->first_token!=s->draft_first+w->argument)
                return mismatch(err,"draft chain frontier");
            fg_vk_tensor *next=NULL,*contracted=NULL;
            float embedding[FG_HIDDEN_SIZE];
            if(status==FG_OK)status=fg_mtp_embedding_row(n->embedding,w->tokens[0],embedding,err);
            if(status==FG_OK)status=fg_vk_tensor_write(s->embedding,0u,embedding,sizeof(embedding),err);
            if(status==FG_OK)status=fg_mtp_head_step_replay(s->head,s->draft_hyper,
                s->embedding,w->first_token-1u,&next,&contracted,err);
            if(status==FG_OK)status=fg_vk_tensor_read(contracted,0u,hidden,FG_HIDDEN_SIZE*4u,err);
            if(status==FG_OK){s->draft_hyper=next;s->drafted++;}
            break;
        }
        case FG_MTP_NATIVE_FINISH:{
            if(s->verify_first!=w->first_token||s->verify_count<w->count||
               !s->drafted||s->draft_first!=w->first_token)
                return mismatch(err,"repair without matching verification");
            status=fg_mtp_head_truncate(s->head,w->first_token,err);
            for(uint32_t i=0;status==FG_OK&&i+1u<w->count;i++)
                status=warm(n,s,w->first_token+i,s->verify+(uint64_t)i*FG_HYPER_WIDTH,
                    w->tokens[i],err);
            if(status==FG_OK){
                memcpy(s->last,s->verify+(uint64_t)(w->count-1u)*FG_HYPER_WIDTH,sizeof(s->last));
                s->last_token=w->first_token+w->count-1u;s->last_valid=true;
                s->draft_hyper=NULL;s->drafted=0u;s->verify_count=0u;s->previous_valid=false;
            }
            break;
        }
        case FG_MTP_NATIVE_ADVANCE:
            if(!s->previous_valid||s->previous_token+1u!=w->first_token||
               !s->last_valid||s->last_token!=w->first_token)
                return mismatch(err,"ordinary decode repair frontier");
            if(fg_mtp_head_frontier(s->head)!=w->first_token-1u)
                status=fg_mtp_head_reset(s->head,w->first_token-1u,err);
            if(status==FG_OK)status=warm(n,s,w->first_token-1u,s->previous,w->tokens[0],err);
            if(status==FG_OK)s->previous_valid=false;
            break;
        default:return mismatch(err,"unknown operation");
    }
    if(status==FG_OK)*ack=(fg_mtp_native_ack){.operation=w->operation,
        .first_token=w->first_token,.count=w->count,
        .frontier=s->head?fg_mtp_head_frontier(s->head):0u};
    return status;
}
