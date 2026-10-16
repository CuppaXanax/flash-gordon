/* Real runtime/fabric orchestration with deterministic, CPU-only compute
 * doubles. GPU arithmetic and trained-head quality are separate gates. */
#include "fg_runtime.h"
struct fg_coordinator;
static fg_status fixture_serial(struct fg_coordinator *,const int32_t *,size_t,
    uint32_t,uint32_t,uint32_t *,float *,fg_error *);
#define FG_RUNTIME_TEST_SERIAL_DECODE fixture_serial
#include "../src/runtime.c"
#include "fg_mtp_embedding.h"
#include "mtp_ring_sse.h"

#include <sys/wait.h>
#include <signal.h>

struct fg_vk_tensor {uint64_t bytes;float *data;};
struct fg_model {uint32_t rank;};
struct fg_owner_executor {
    uint32_t slot,rank,pending_layer,pending_first,pending_count;
    uint32_t frontier[2][FG_LAYER_COUNT];
    fg_vk_tensor *input,*output;
    fg_expert_result results[FG_GROUP_SIZE];
};
struct fg_output_slice {uint32_t rank;};
struct fg_mtp_embedding {int unused;};
struct fg_mtp_head {uint32_t frontier,step;fg_vk_tensor *next,*hidden;};
static uint32_t rejection=FG_MTP_MAX_DRAFT;
static uint32_t fixture_eos=FG_Q38_VOCAB_SIZE-1u;
static fg_vk_tensor *zero_ngram;
static void require(fg_status status,fg_error *e){
    if(status!=FG_OK){fprintf(stderr,"MTP_RING_FAIL %d %s\n",status,e->message);exit(1);}
}
#define VERIFY(x) do{if(!(x)){fprintf(stderr,"MTP_RING_FAIL line=%d %s\n",__LINE__,#x);exit(1);}}while(0)

fg_status __wrap_fg_vk_tensor_create(fg_vk_context *v,uint64_t n,fg_vk_tensor **t,fg_error *e){
    (void)v;(void)e;*t=calloc(1,sizeof(**t));if(!*t)return FG_ERR_OOM;
    (*t)->bytes=n;(*t)->data=calloc(1,(size_t)n);return (*t)->data?FG_OK:FG_ERR_OOM;
}
void __wrap_fg_vk_tensor_destroy(fg_vk_tensor *t){if(t){free(t->data);free(t);}}
fg_status __wrap_fg_vk_tensor_write(fg_vk_tensor *t,uint64_t at,const void *p,uint64_t n,fg_error *e){
    (void)e;if(!t||at>t->bytes||n>t->bytes-at)return FG_ERR_LIMIT;
    memcpy((uint8_t *)t->data+at,p,(size_t)n);return FG_OK;
}
fg_status __wrap_fg_vk_tensor_read(const fg_vk_tensor *t,uint64_t at,void *p,uint64_t n,fg_error *e){
    (void)e;if(!t||at>t->bytes||n>t->bytes-at)return FG_ERR_LIMIT;
    memcpy(p,(uint8_t *)t->data+at,(size_t)n);return FG_OK;
}
fg_vk_context *__wrap_fg_model_vk(const fg_model *m){(void)m;return (fg_vk_context *)(uintptr_t)1u;}
uint32_t __wrap_fg_model_rank(const fg_model *m){return m->rank;}
fg_vk_tensor *__wrap_fg_model_tensor(const fg_model *m,const char *name){
    (void)m;(void)name;return zero_ngram;
}
fg_status __wrap_fg_vk_embedding_q8_0_batch(fg_vk_context *v,fg_vk_tensor *out,
    const fg_vk_tensor *w,const fg_vk_tensor *ids,uint32_t n,uint32_t width,
    uint32_t rows,uint32_t copies,fg_error *e){
    (void)v;(void)w;(void)width;(void)rows;(void)copies;(void)e;
    memset(out->data,0,(size_t)out->bytes);
    const uint32_t *tokens=(const uint32_t *)ids->data;
    for(uint32_t i=0;i<n;i++)out->data[(uint64_t)i*FG_HYPER_WIDTH]=(float)tokens[i]*1000.0f;
    return FG_OK;
}
fg_status __wrap_fg_ngram_store_lookup_prefill(fg_ngram_store *s,const int32_t *h,
    size_t hn,uint32_t first,uint32_t n,fg_vk_tensor **out,fg_error *e){
    (void)s;(void)h;(void)hn;(void)first;(void)n;(void)e;*out=zero_ngram;return FG_OK;
}
uint32_t __wrap_fg_tokenizer_eos(const fg_tokenizer *t){(void)t;return fixture_eos;}
fg_status __wrap_fg_tokenizer_decode_token(const fg_tokenizer *t,uint32_t id,char *out,
    size_t capacity,size_t *bytes,fg_error *e){
    (void)t;(void)e;if(!capacity)return FG_ERR_LIMIT;out[0]=(char)('a'+id%26u);*bytes=1u;return FG_OK;
}
fg_status __wrap_fg_tokenizer_token(const fg_tokenizer *t,uint32_t id,const char **text,
    size_t *bytes,uint32_t *type,fg_error *e){
    (void)t;(void)id;(void)e;*text="<|im_end|>";*bytes=10u;if(type)*type=0u;return FG_OK;
}
uint32_t __wrap_fg_output_slice_ways(const fg_output_slice *s){(void)s;return 4u;}
fg_status __wrap_fg_output_hc_run(fg_output_hc *h,const void *p,float *out,fg_error *e){
    (void)h;(void)e;memset(out,0,FG_HIDDEN_SIZE*4u);
    out[0]=(float)((uint32_t)((const float *)p)[0]/1000u+1u);return FG_OK;
}
fg_status __wrap_fg_output_slice_run_hidden(fg_output_slice *s,const void *p,
    float *score,uint32_t *id,fg_error *e){
    (void)e;uint32_t way=0u,first=0u,rows=0u;
    VERIFY(fg_output_split_way_for_rank(4u,s->rank,&way));
    fg_output_split_span(4u,way,&first,&rows);
    uint32_t wanted=(uint32_t)((const float *)p)[0];
    *score=wanted>=first&&wanted-first<rows?1.0f:-1.0f;
    *id=*score>0.0f?wanted:first;return FG_OK;
}
uint32_t __wrap_fg_owner_session_count(const fg_owner_executor *o){(void)o;return 2u;}
uint32_t __wrap_fg_owner_active_session(const fg_owner_executor *o){return o->slot;}
fg_status __wrap_fg_owner_set_active_session(fg_owner_executor *o,uint32_t s,fg_error *e){
    (void)e;if(s>=2u)return FG_ERR_ARGUMENT;o->slot=s;return FG_OK;
}
fg_vk_tensor *__wrap_fg_owner_prefill_input(fg_owner_executor *o){return o->input;}
fg_vk_tensor *__wrap_fg_owner_prefill_input_slot(fg_owner_executor *o,uint32_t s){
    (void)s;return o->input;
}
fg_expert_result *__wrap_fg_owner_decode_results(fg_owner_executor *o){return o->results;}
fg_status __wrap_fg_owner_qsa_frontier(const fg_owner_executor *o,uint32_t s,
    uint32_t tokens[FG_LAYER_COUNT]){memcpy(tokens,o->frontier[s],FG_LAYER_COUNT*4u);return FG_OK;}
fg_status __wrap_fg_owner_session_device_snapshot(fg_owner_executor *o,uint32_t s,
    fg_owner_session_checkpoint *c,fg_error *e){
    (void)e;memset(c,0,sizeof(*c));c->valid=true;c->session=s;
    memcpy(c->qsa_tokens,o->frontier[s],sizeof(c->qsa_tokens));return FG_OK;
}
fg_status __wrap_fg_owner_session_rollback(fg_owner_executor *o,
    const fg_owner_session_checkpoint *c,fg_error *e){
    (void)e;if(!c->valid)return FG_ERR_MISMATCH;
    memcpy(o->frontier[c->session],c->qsa_tokens,sizeof(c->qsa_tokens));return FG_OK;
}
void __wrap_fg_owner_session_snapshot_release(fg_owner_session_checkpoint *c){c->valid=false;}
fg_status __wrap_fg_expert_decode_chain_ready(fg_expert_executor *x,uint32_t l,fg_error *e){
    (void)x;(void)l;(void)e;return FG_OK;
}
static fg_status toy_block(fg_owner_executor *o,uint32_t first,uint32_t last,
    uint32_t token,uint32_t n,const fg_vk_tensor *in,fg_vk_tensor **out,fg_error *e){
    (void)e;
    for(uint32_t l=first;l<=last;l++){
        if(o->frontier[o->slot][l]!=token)return FG_ERR_MISMATCH;
        o->frontier[o->slot][l]+=n;
    }
    if(o->output!=in)memcpy(o->output->data,in->data,(size_t)n*FG_HYPER_WIDTH*4u);
    for(uint32_t i=0;i<n;i++)o->output->data[(uint64_t)i*FG_HYPER_WIDTH]+=(float)(last-first+1u);
    *out=o->output;return FG_OK;
}
fg_status __wrap_fg_owner_decode_small_n(fg_owner_executor *o,uint32_t first,
    uint32_t last,uint32_t token,const uint32_t *p,uint32_t n,const fg_vk_tensor *in,
    const fg_vk_tensor *ng,fg_owner_expert_small_fn fn,void *ctx,fg_vk_tensor **out,fg_error *e){
    (void)p;(void)ng;(void)fn;(void)ctx;return toy_block(o,first,last,token,n,in,out,e);
}
fg_status __wrap_fg_owner_decode_block_chained(fg_owner_executor *o,uint32_t first,
    uint32_t last,uint32_t token,const uint32_t p[3],const fg_vk_tensor *in,
    const fg_vk_tensor *ng,fg_owner_expert_inline_fn fn,void *ctx,fg_vk_tensor **out,fg_error *e){
    (void)p;(void)ng;(void)fn;(void)ctx;return toy_block(o,first,last,token,1u,in,out,e);
}
fg_status __wrap_fg_owner_prefill_layer_begin(fg_owner_executor *o,uint32_t slot,
    uint32_t l,uint32_t first,const uint32_t *p,uint16_t n,const fg_vk_tensor *in,
    const fg_vk_tensor *ng,fg_owner_prefill_fire_fn f,void *fc,
    fg_owner_qsa_prefill_dispatch_fn q,void *qc,fg_error *e){
    (void)slot;(void)p;(void)ng;(void)f;(void)fc;(void)q;(void)qc;
    fg_vk_tensor *out=NULL;return toy_block(o,l,l,first,n,in,&out,e);
}
fg_status __wrap_fg_owner_prefill_layer_finish(fg_owner_executor *o,uint32_t slot,
    fg_owner_prefill_collect_fn f,void *ctx,fg_vk_tensor **out,fg_error *e){
    (void)slot;(void)f;(void)ctx;(void)e;*out=o->output;return FG_OK;
}
fg_status __wrap_fg_mtp_embedding_open(fg_mtp_embedding **out,const fg_manifest *m,
    const char *p,fg_error *e){(void)m;(void)p;(void)e;*out=calloc(1,sizeof(**out));return *out?FG_OK:FG_ERR_OOM;}
void __wrap_fg_mtp_embedding_close(fg_mtp_embedding *x){free(x);}
fg_status __wrap_fg_mtp_embedding_row(fg_mtp_embedding *x,uint32_t id,float *out,fg_error *e){
    (void)x;(void)e;memset(out,0,FG_HIDDEN_SIZE*4u);out[0]=(float)id;return FG_OK;
}
fg_status __wrap_fg_mtp_head_open_windowed(fg_mtp_head **h,fg_vk_context *v,
    const fg_manifest *m,uint32_t rank,const char *p,uint32_t ctx,uint32_t win,fg_error *e){
    (void)m;(void)rank;(void)p;(void)ctx;(void)win;*h=calloc(1,sizeof(**h));
    if(!*h)return FG_ERR_OOM;
    require(fg_vk_tensor_create(v,FG_HYPER_WIDTH*4u,&(*h)->next,e),e);
    return fg_vk_tensor_create(v,FG_HIDDEN_SIZE*4u,&(*h)->hidden,e);
}
fg_status __wrap_fg_mtp_head_open_owner_windowed(fg_mtp_head **h,fg_vk_context *v,
    const fg_manifest *m,uint32_t rank,uint32_t selected,const char *p,
    uint32_t ctx,uint32_t win,fg_error *e){
    VERIFY(m&&rank==selected&&selected==6u);
    return __wrap_fg_mtp_head_open_windowed(h,v,m,rank,p,ctx,win,e);
}
fg_status __wrap_fg_mtp_head_fork(fg_mtp_head **out,fg_mtp_head *h,uint32_t s,fg_error *e){
    (void)h;(void)s;return __wrap_fg_mtp_head_open_windowed(out,NULL,NULL,0,NULL,0,0,e);
}
fg_status __wrap_fg_mtp_head_check_residency(const fg_mtp_head *head,fg_error *err){
    static uint32_t checks;
    VERIFY(head);
    if(!checks++){
        fg_error_set(err,FG_ERR_LIMIT,"injected startup residency rejection");
        return FG_ERR_LIMIT;
    }
    return FG_OK;
}
fg_status __wrap_fg_expert_prepare_small_n_scratch(fg_expert_executor *expert,fg_error *err){
    (void)expert;(void)err;return FG_OK;
}
void __wrap_fg_vk_get_memory_stats(const fg_vk_context *vk,fg_vk_memory_stats *stats){
    (void)vk;memset(stats,0,sizeof(*stats));
}
void __wrap_fg_mtp_head_close(fg_mtp_head *h){
    if(h){fg_vk_tensor_destroy(h->next);fg_vk_tensor_destroy(h->hidden);free(h);}
}
uint32_t __wrap_fg_mtp_head_frontier(const fg_mtp_head *h){return h->frontier;}
fg_status __wrap_fg_mtp_head_reset(fg_mtp_head *h,uint32_t n,fg_error *e){
    (void)e;h->frontier=n;h->step=0u;return FG_OK;
}
fg_status __wrap_fg_mtp_head_truncate(fg_mtp_head *h,uint32_t n,fg_error *e){
    (void)e;if(n>h->frontier)return FG_ERR_MISMATCH;h->frontier=n;h->step=0u;return FG_OK;
}
fg_status __wrap_fg_mtp_head_warm_replay(fg_mtp_head *h,const fg_vk_tensor *a,
    const fg_vk_tensor *b,uint32_t n,fg_error *e){
    (void)a;(void)b;(void)e;if(n!=h->frontier)return FG_ERR_MISMATCH;
    h->frontier++;h->step=0u;return FG_OK;
}
fg_status __wrap_fg_mtp_head_step_replay(fg_mtp_head *h,const fg_vk_tensor *a,
    const fg_vk_tensor *b,uint32_t n,fg_vk_tensor **next,fg_vk_tensor **hidden,fg_error *e){
    (void)b;(void)e;if(n!=h->frontier)return FG_ERR_MISMATCH;
    h->hidden->data[0]=(float)((uint32_t)a->data[0]/1000u+2u+(h->step==rejection?5u:0u));
    h->next->data[0]=a->data[0]+1000.0f;h->frontier++;h->step++;
    *next=h->next;*hidden=h->hidden;return FG_OK;
}

static fg_status fixture_serial(fg_coordinator *c,const int32_t *h,size_t n,
    uint32_t token,uint32_t position,uint32_t *next,float *logit,fg_error *e){
    (void)token;(void)position;
    fg_runtime *r=(fg_runtime *)((uint8_t *)c - offsetof(fg_runtime,coordinator));
    fg_mtp_result result;float logits[4];
    fg_status status=runtime_mtp_round(r,(uint32_t)h[n-1u],1u,&result,logits,e);
    if(status==FG_OK){VERIFY(result.count==1u&&result.consumed==1u);*next=result.tokens[0];*logit=logits[0];}
    return status;
}

typedef struct capture_tokens {
    uint32_t ids[64],count,cancel_after;
    fixture_sse *sse;
} capture_tokens;
static fg_status capture(void *p,uint32_t id,const char *s,size_t n,fg_error *e){
    capture_tokens *c=p;VERIFY(c->count<64u);c->ids[c->count++]=id;
    return c->sse?fixture_sse_token(c->sse,id,s,n,e):FG_OK;
}
static bool cancel(void *p){
    capture_tokens *c=p;return (c->cancel_after&&c->count>=c->cancel_after)||
        (c->sse&&fixture_sse_stopped(c->sse));
}
static void generation_cases(fg_runtime *r,fg_error *e){
    r->manifest=(fg_manifest *)r->coordinator.manifest;
    r->context_limit=8192u;r->state_ready=true;r->next_token_valid=true;
    r->prefix_continuation=true;r->history_capacity=64u;
    r->state_frontier=r->state_position=(uint32_t)r->history_count;
    r->next_token=(uint32_t)r->history[r->history_count-1u]+1u;
    r->sampler.repetition_penalty=1.0f;
    for(uint32_t limit=1u;limit<=8u;limit++){
        uint32_t first=r->next_token;capture_tokens out={0};fg_generation_stats stats;
        fg_tokens prompt={.data=(uint32_t *)r->history,.count=r->history_count};
        require(runtime_generate_tokens(r,"fixture",&prompt,true,NULL,NULL,false,0u,
            limit,capture,&out,cancel,&out,&stats,e),e);
        VERIFY(out.count==limit&&r->state_position==r->history_count);
        for(uint32_t i=0;i<out.count;i++)VERIFY(out.ids[i]==first+i);
    }
    capture_tokens out={.cancel_after=2u};fg_generation_stats stats;
    fg_tokens prompt={.data=(uint32_t *)r->history,.count=r->history_count};
    require(runtime_generate_tokens(r,"fixture",&prompt,true,NULL,NULL,false,0u,
        8u,capture,&out,cancel,&out,&stats,e),e);
    VERIFY(out.count>=2u&&out.count<=4u&&r->state_position==r->history_count);
    for(uint32_t distance=0u;distance<=4u;distance++){
        fixture_eos=r->next_token+distance;out=(capture_tokens){0};
        prompt=(fg_tokens){.data=(uint32_t *)r->history,.count=r->history_count};
        require(runtime_generate_tokens(r,"fixture",&prompt,true,NULL,NULL,false,0u,
            8u,capture,&out,cancel,&out,&stats,e),e);
        VERIFY(out.count==distance&&r->state_position==r->history_count&&r->pending_eos_valid);
    }
    fixture_eos=FG_Q38_VOCAB_SIZE-1u;
    out=(capture_tokens){0};require(fixture_sse_open(&out.sse,NULL,e),e);
    prompt=(fg_tokens){.data=(uint32_t *)r->history,.count=r->history_count};
    require(runtime_generate_tokens(r,"fixture",&prompt,true,NULL,NULL,false,0u,
        8u,capture,&out,cancel,&out,&stats,e),e);
    VERIFY(out.count==8u&&r->state_position==r->history_count);
    require(fixture_sse_finish(out.sse,e),e);
    char stop[3]={(char)('a'+(r->next_token+1u)%26u),
        (char)('a'+(r->next_token+2u)%26u),0};
    out=(capture_tokens){0};require(fixture_sse_open(&out.sse,stop,e),e);
    prompt=(fg_tokens){.data=(uint32_t *)r->history,.count=r->history_count};
    require(runtime_generate_tokens(r,"fixture",&prompt,true,NULL,NULL,false,0u,
        8u,capture,&out,cancel,&out,&stats,e),e);
    VERIFY(out.count>=3u&&out.count<=6u&&r->state_position==r->history_count);
    require(fixture_sse_finish(out.sse,e),e);
    puts("Actual generation loop: limits1..8, final one-input verifier seam, cancel drain and EOS PASS");
    puts("Actual API SSE sink: multi-token commits, split-stop suppression, completion framing and DONE PASS");
}

typedef struct coexist_job {
    fg_runtime *runtime;
    capture_tokens capture;
    fg_generation_stats stats;
    fg_error error;
    fg_status status;
    uint32_t first;
} coexist_job;
static void coexist_run(void *opaque){
    coexist_job *job=opaque;fg_runtime *r=job->runtime;
    fg_tokens prompt={.data=(uint32_t *)r->history,.count=r->history_count};
    job->status=runtime_generate_tokens(r,"fixture",&prompt,true,NULL,NULL,false,0u,
        12u,capture,&job->capture,cancel,&job->capture,&job->stats,&job->error);
}
static void coexist_cases(fg_runtime runtime[2],fg_owner_executor *owner,
    fg_fabric *fabric,fg_error *err){
    fg_pipeline pipeline;require(fg_pipeline_init(&pipeline,fabric,err),err);
    coexist_job jobs[2]={{.runtime=&runtime[0],.first=runtime[0].next_token},
        {.runtime=&runtime[1],.first=runtime[1].next_token}};
    for(uint32_t slot=0;slot<2u;slot++){
        require(fixture_sse_open(&jobs[slot].capture.sse,NULL,err),err);
        require(fg_pipeline_start(&pipeline,slot,coexist_run,&jobs[slot],err),err);
    }
    bool overlapped=false;
    while(pipeline.slots[0].active||pipeline.slots[1].active){
        uint32_t slot=fg_pipeline_pick(&pipeline);
        if(slot>=2u){usleep(100u);continue;}
        fg_fabric_set_slot(fabric,slot);
        require(fg_owner_set_active_session(owner,slot,err),err);
        fg_pipeline_turn(&pipeline,slot);
        if(pipeline.slots[0].active&&pipeline.slots[1].active&&
            jobs[0].capture.count&&jobs[1].capture.count)overlapped=true;
    }
    VERIFY(overlapped);
    for(uint32_t slot=0;slot<2u;slot++){
        require(jobs[slot].status,&jobs[slot].error);
        VERIFY(jobs[slot].capture.count==12u);
        VERIFY(runtime[slot].history_count==runtime[slot].state_position);
        for(uint32_t i=0;i<12u;i++)VERIFY(jobs[slot].capture.ids[i]==jobs[slot].first+i);
        for(uint32_t l=0;l<FG_LAYER_COUNT;l++)if(runtime[slot].manifest->layer_owner[l]==0u)
            VERIFY(owner->frontier[slot][l]==runtime[slot].history_count);
        require(fixture_sse_finish(jobs[slot].capture.sse,err),err);
    }
    fg_pipeline_destroy(&pipeline);
    puts("Real coroutine coexistence: two active sessions, private policy/native state, causal commit and SSE PASS");
}

static void large_credit(fg_fabric *fabric,fg_owner_executor *owner,
    const fg_manifest *m,uint32_t rank,layer_work_context *base,fg_error *err){
    layer_work_context context=*base;
    context.tokens=FG_PREFILL_MAX_TOKENS;
    context.work_capacity=FG_MTP_TARGET_HEADER_BYTES+
        context.tokens*(FG_HYPER_WIDTH+1u)*4u;
    context.work_wire=calloc(1,context.work_capacity);
    context.hyper_in=calloc((size_t)context.tokens*FG_HYPER_WIDTH,4u);
    context.hyper_out=calloc((size_t)context.tokens*FG_HYPER_WIDTH,4u);
    VERIFY(context.work_wire&&context.hyper_in&&context.hyper_out);
    for(uint32_t turn=0u;turn<4u;turn++){
        uint32_t slot=turn%2u,first=turn<2u?0u:context.tokens;
        uint64_t session=context.session_ids[slot];
        fg_fabric_set_slot(fabric,slot);
        require(fg_owner_set_active_session(owner,slot,err),err);
        if(rank==6u){
            require(fg_fabric_send(fabric,7u,FG_FABRIC_BULK,FG_MSG_PREFILL_LAYER_WORK,
                session,turn,0u,context.work_wire,context.work_capacity,err),err);
            require(mtp_target_credit(fabric,owner,m,rank,session,first,context.tokens,
                FG_MTP_TARGET_PREFILL,&context,err),err);
        }else{
            fg_frame_header header;uint32_t bytes=0u;
            require(fg_fabric_recv(fabric,6u,FG_FABRIC_BULK,&header,context.work_wire,
                context.work_capacity,&bytes,err),err);
            VERIFY(bytes==context.work_capacity&&fg_frame_request_id(&header)==session&&
                   fg_frame_sequence(&header)==turn&&fg_frame_session_slot(&header)==slot);
            for(uint32_t i=0;i<context.tokens;i++){
                context.hyper_out[(uint64_t)i*FG_HYPER_WIDTH]=(float)(first+i+slot);
                context.mtp_tokens[i]=first+i+1u;
            }
            if(turn>=2u)context.mtp_tokens[context.tokens-1u]=UINT32_MAX;
            require(mtp_target_observe(fabric,m,rank,session,first,context.tokens,
                FG_MTP_TARGET_PREFILL,&context,err),err);
        }
    }
    free(context.hyper_out);free(context.hyper_in);free(context.work_wire);
    puts("MTP large-frame two-session target credit: PASS");
}

static int run_rank(fg_manifest *m,uint32_t rank,uint32_t native_owner){
    fg_error e={0};fg_fabric *f=NULL;
    require(fg_fabric_open(&f,m,rank,&e),&e);
    fg_model model={.rank=rank};fg_owner_executor owner={.rank=rank};
    require(fg_vk_tensor_create(NULL,FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH*4u,&owner.input,&e),&e);
    require(fg_vk_tensor_create(NULL,FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH*4u,&owner.output,&e),&e);
    require(fg_vk_tensor_create(NULL,FG_MTP_MAX_VERIFY*FG_NGRAM_EMBED_VALUES*4u,&zero_ngram,&e),&e);
    layer_work_context context={0};
    require(layer_work_context_create(&context,&model,&owner,m,
        (fg_expert_executor *)(uintptr_t)1u,NULL,&e),&e);
    fg_output_slice slice={.rank=rank};context.output_slice=&slice;
    context.output_hc=(fg_output_hc *)(uintptr_t)1u;
    fg_mtp_native *native=NULL;
    require(fg_mtp_native_create(&native,context.vk,m,rank,&e),&e);
    context.mtp_native=native;
    uint32_t wire_frontiers[2][FG_LAYER_COUNT]={{0}};
    depthb_owner_runtime depthb={.owner=&owner,.manifest=m,.self=rank,
        .logical_context=8192u,.wire_frontiers=wire_frontiers};
    uint64_t nonce[2]={71u,72u};
    context.session_ids=nonce;
    fg_runtime runtimes[2]={0};
    fg_runtime *runtime=&runtimes[0];
    fg_coordinator *c=&runtime->coordinator;
    if(!rank){
        c->manifest=m;c->model=&model;c->owner=&owner;c->fabric=f;
        c->output_slice=&slice;c->depthb=depthb;c->depthb_work=context;
        c->ring_prefill=c->ring_decode=true;
        c->options=(fg_runtime_options){.experimental_flags=FG_RUNTIME_EXPERIMENTAL_MTP,
            .logical_context_tokens=8192u,.gpu_index_tokens=8192u,.qsa_hot_tokens=8192u,
            .mtp_overlay="fixture.gguf",.mtp_embedding="fixture.fgme",
            .mtp_native_rank_plus_one=native_owner==7u?0u:native_owner+1u,
            .mtp_window_tokens=64u,.mtp_max_draft=3u};
        memset(c->identity.identity_sha256,1,32u);
        atomic_init(&c->transport_state,FG_TRANSPORT_READY);
        require(coordinator_mtp_create(c,true,&e),&e);
        for(uint32_t slot=0;slot<2u;slot++){
            runtime=&runtimes[slot];
            if(slot)runtime->coordinator=runtimes[0].coordinator;
            c=&runtime->coordinator;
            c->output_session=slot;c->session_id=nonce[slot];fg_fabric_set_slot(f,slot);
            require(fg_owner_set_active_session(&owner,slot,&e),&e);
            if(!slot){
                VERIFY(coordinator_mtp_prepare(c,2u,&e)==FG_ERR_LIMIT);
                VERIFY(transport_ready(&c->transport_state));
                require(coordinator_mtp_prepare(c,2u,&e),&e);
                require(coordinator_mtp_prepare(c,2u,&e),&e);
            }
            if(slot){c->mtp=NULL;require(coordinator_mtp_create(c,false,&e),&e);}
            uint32_t seed=slot?101u:1u;
            uint32_t prefix[2]={seed,seed+1u},next[2]={seed+1u,UINT32_MAX},pos[6]={0,0,0,1,1,1};
            int32_t *history=calloc(128u,sizeof(*history));VERIFY(history);
            history[0]=(int32_t)seed;history[1]=(int32_t)seed+1;
            runtime->history=history;runtime->history_count=2u;
            runtime->history_capacity=128u;
            require(coordinator_mtp_prefix(c,history,0u,2u,true,&e),&e);
            require(fg_vk_tensor_create(NULL,8u,&c->prefill_layer[0].token_tensor,&e),&e);
            require(fg_vk_tensor_write(c->prefill_layer[0].token_tensor,0u,prefix,8u,&e),&e);
            require(fg_vk_embedding_q8_0_batch(context.vk,owner.input,zero_ngram,
                c->prefill_layer[0].token_tensor,2u,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,4u,&e),&e);
            float hc[2u*FG_HYPER_WIDTH];
            require(fg_vk_tensor_read(owner.input,0u,hc,sizeof(hc),&e),&e);
            fg_prefill_layer_work work={.layer=0u,.source_rank=0u,
                .destination_rank=m->layer_owner[0],.flags=FG_LAYER_WORK_HAS_NGRAM,
                .first_token=0u,.token_count=2u,.positions=pos,.hyper=hc,
                .ngram_embeddings=zero_ngram->data};
            uint32_t bytes=0u;
            require(fg_prefill_layer_work_encode(context.work_wire,context.work_capacity,
                &bytes,m->protocol_version,&work,&e),&e);
            require(fg_mtp_prefill_tokens_encode(context.work_wire,context.work_capacity,
                &bytes,next,2u,&e),&e);
            require(fg_fabric_send(f,work.destination_rank,FG_FABRIC_BULK,
                FG_MSG_MTP_PREFILL_WORK,c->session_id,0u,0u,context.work_wire,bytes,&e),&e);
            for(;;){
                uint32_t peer=0u,got=0u;fg_frame_header header;
                require(fg_fabric_recv_any(f,FG_FABRIC_BULK,&peer,&header,
                    context.result_wire,context.result_capacity,&got,&e),&e);
                if(fg_frame_type(&header)==FG_MSG_PREFILL_LAYER_RESULT)break;
                VERIFY(fg_frame_type(&header)==FG_MSG_MTP_PREFILL_WORK);
                require(handle_prefill_layer_work(f,&owner,m,0u,c->session_id,peer,
                    &header,context.result_wire,got,&c->depthb_work,&e),&e);
            }
            fg_vk_tensor_destroy(c->prefill_layer[0].token_tensor);
            require(fg_vk_tensor_create(NULL,FG_MTP_MAX_VERIFY*4u,
                &c->prefill_layer[0].token_tensor,&e),&e);
            for(uint32_t fixed=0u;fixed<2u;fixed++)
            for(uint32_t k=1u;k<=3u;k++){
                uint32_t next_pending=(uint32_t)history[runtime->history_count-1u]+1u;
                history[runtime->history_count++]=(int32_t)next_pending;
                uint32_t pending=(uint32_t)history[runtime->history_count-1u];
                require(runtime_mtp_policy_init(runtime,&e),&e);
                runtime->mtp_policy.rounds=1u;c->options.mtp_max_draft=k;
                c->options.mtp_fixed_draft=fixed?k:0u;
                for(uint32_t j=0;j<k;j++)runtime->mtp_policy.round_ms[j]=j+1u==k?1.0:1000.0;
                if(fixed)runtime->mtp_policy.rounds=0u;
                fg_mtp_result result;float logits[4];
                require(runtime_mtp_round(runtime,pending,4u,&result,logits,&e),&e);
                VERIFY(result.accepted==(rejection<k?rejection:k));
                VERIFY(result.count==result.accepted+1u);
                for(uint32_t i=0;i<result.count;i++)VERIFY(result.tokens[i]==pending+i+1u);
                for(uint32_t i=0;i+1u<result.count;i++)
                    history[runtime->history_count++]=(int32_t)result.tokens[i];
                VERIFY(transport_ready(&c->transport_state));
            }
            c->options.mtp_fixed_draft=0u;
            generation_cases(runtime,&e);
        }
        coexist_cases(runtimes,&owner,f,&e);
        if(native_owner==6u){
            int32_t prefix[2u*FG_PREFILL_MAX_TOKENS];
            for(uint32_t i=0;i<2u*FG_PREFILL_MAX_TOKENS;i++)prefix[i]=(int32_t)i+1;
            for(uint32_t slot=0;slot<2u;slot++){
                c=&runtimes[slot].coordinator;fg_fabric_set_slot(f,slot);
                require(coordinator_mtp_prefix(c,prefix,0u,
                    2u*FG_PREFILL_MAX_TOKENS,true,&e),&e);
            }
        }
        for(uint32_t slot=0;slot<2u;slot++){
            free(runtimes[slot].history);free(runtimes[slot].rendered_history);
        }
        for(uint32_t peer=1u;peer<FG_RANK_COUNT;peer++)
            require(fg_fabric_send(f,peer,FG_FABRIC_CONTROL,FG_MSG_ERROR,99u,0u,0u,NULL,0u,&e),&e);
        puts("MTP synthetic 8-rank runtime: prefill/warmfill, fanout, k1..3, rollback, sessions2 PASS");
    }else{
        for(;;){
            uint32_t peer=0u,bytes=0u;fg_frame_header header;fg_fabric_class cls;
            require(fg_fabric_wait_ready(f,3u,&peer,&cls,&e),&e);
            require(fg_fabric_recv(f,peer,cls,&header,context.result_wire,
                context.result_capacity,&bytes,&e),&e);
            fg_message_type type=fg_frame_type(&header);
            if(type==FG_MSG_ERROR){
                if(native_owner==6u&&(rank==6u||rank==7u))
                    large_credit(f,&owner,m,rank,&context,&e);
                break;
            }
            uint32_t slot=fg_frame_session_slot(&header);VERIFY(slot<2u);
            fg_fabric_set_slot(f,slot);require(fg_owner_set_active_session(&owner,slot,&e),&e);
            fg_status status=FG_ERR_FORMAT;
            if(type==FG_MSG_MTP_NATIVE_WORK)
                status=handle_mtp_native_work(f,&owner,m,rank,nonce[slot],peer,&header,
                    context.result_wire,bytes,&context,&e);
            else if(type==FG_MSG_MTP_PREFILL_WORK)
                status=handle_prefill_layer_work(f,&owner,m,rank,nonce[slot],peer,
                    &header,context.result_wire,bytes,&context,&e);
            else if(type==FG_MSG_MTP_VERIFY_WORK)
                status=handle_mtp_verify_work(f,&owner,m,rank,nonce[slot],peer,&header,
                    context.result_wire,bytes,&context,&depthb,&e);
            else if(type==FG_MSG_MTP_HEAD_WORK)
                status=handle_mtp_head_work(f,m,rank,nonce[slot],peer,&header,
                    context.result_wire,bytes,&context,&e);
            else if(type==FG_MSG_SESSION_PREPARE||type==FG_MSG_SESSION_RESTORE||
                    type==FG_MSG_SESSION_COMMIT)
                status=handle_owner_session_transaction(f,&depthb,nonce[slot],peer,
                    &header,context.result_wire,bytes,&e);
            require(status,&e);
        }
    }
    fg_fabric_close(f);return 0;
}

static pid_t children[FG_RANK_COUNT];
static volatile sig_atomic_t child_count,interrupted;
static void stop_children(int signal){
    interrupted=signal;
    for(sig_atomic_t i=0;i<child_count;i++)if(children[i]>0)kill(children[i],SIGTERM);
}
int main(int argc,char **argv){
    if(argc!=6&&argc!=7){fprintf(stderr,"usage: test_mtp_ring HOST0 HOST1 BASEPORT REJECT(0..3) PARITY(0..1|2=local) [NATIVE_OWNER]\n");return 2;}
    uint32_t base=(uint32_t)strtoul(argv[3],NULL,10);
    rejection=(uint32_t)strtoul(argv[4],NULL,10);
    uint32_t parity=(uint32_t)strtoul(argv[5],NULL,10);
    uint32_t native_owner=argc==7?(uint32_t)strtoul(argv[6],NULL,10):7u;
    VERIFY(native_owner==7u||native_owner==6u);
    VERIFY(base>1024u&&base<65000u&&rejection<=3u&&parity<=2u);
    fg_manifest *m=malloc(sizeof(*m));VERIFY(m);fg_manifest_init(m);
    m->prefill_microbatch=4u;m->protocol_version=6u;
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){
        memset(m->session.rank_state_format_sha256[r],(int)r+1,32u);
        snprintf(m->ranks[r].endpoint,sizeof(m->ranks[r].endpoint),"%s:%u",
            argv[1u+(r%2u)],base+r*2u);
    }
    static const uint8_t order[]={1,0,2,3,4,5,6,7};
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        m->layer_owner[l]=order[l/6u];
        for(uint32_t x=0;x<FG_EXPERT_COUNT;x++)m->expert_rank[l][x]=m->layer_owner[l];
    }
    for(uint32_t r=0;r<FG_RANK_COUNT;r++)if(parity==2u||r%2u==parity){
        pid_t p=fork();
        if(p<0){perror("fork");stop_children(SIGTERM);break;}
        if(!p)return run_rank(m,r,native_owner);
        children[child_count++]=p;
    }
    struct sigaction action={.sa_handler=stop_children};
    sigemptyset(&action.sa_mask);
    VERIFY(!sigaction(SIGTERM,&action,NULL)&&!sigaction(SIGINT,&action,NULL)&&
        !sigaction(SIGALRM,&action,NULL));
    alarm(90u);
    int failed=interrupted?1:0;uint32_t remaining=(uint32_t)child_count;
    while(remaining){
        int status=0;pid_t pid=waitpid(-1,&status,0);
        if(pid<0&&errno==EINTR)continue;
        if(pid<0){failed=1;stop_children(SIGTERM);break;}
        remaining--;
        for(sig_atomic_t i=0;i<child_count;i++)if(children[i]==pid)children[i]=0;
        if(!WIFEXITED(status)||WEXITSTATUS(status)){
            failed=1;stop_children(SIGTERM);
        }
    }
    alarm(0u);if(interrupted)failed=1;
    free(m);return failed;
}
