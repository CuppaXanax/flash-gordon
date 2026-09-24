#include "fg_owner.h"
#include "fg_q38_math.h"
#include "fg_qsa.h"
#include "fg_runtime.h"
#include "fg_topology.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ts_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e3+(double)t.tv_nsec*1e-6;}
static uint64_t wall_ns(void){struct timespec value;clock_gettime(CLOCK_REALTIME,&value);return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;}
static bool gdn_diag_enabled(void){const char *value=getenv("FG_GDN_DIAG");return value&&*value&&strcmp(value,"0")!=0;}
static bool frame_trace_enabled(void){const char *value=getenv("FG_FRAME_TRACE");return value&&*value&&strcmp(value,"0")!=0;}
static bool decode_ms_enabled(void){const char *value=getenv("FG_DECODE_MS");return value&&*value&&strcmp(value,"0")!=0;}
static bool numerics_trace_enabled(void){const char *value=getenv("FG_NUMERICS_TRACE");return value&&*value&&strcmp(value,"0")!=0;}
static uint64_t numerics_hash_bytes(const void *data,size_t bytes){
    const uint8_t *p=data;uint64_t hash=UINT64_C(1469598103934665603);
    for(size_t i=0;i<bytes;i++){hash^=p[i];hash*=UINT64_C(1099511628211);}
    return hash;
}
/* Env-gated per-layer hyper-state digest used to bisect the ring numerics
 * defect against the single-owner reference: identical inputs must produce
 * identical OUT hashes on every rank for both topologies. */
static void numerics_trace_tensor(const char *phase,uint32_t rank,uint32_t layer,
    uint32_t first_token,uint16_t token_count,const fg_vk_tensor *tensor,fg_error *err){
    (void)err;
    if(!numerics_trace_enabled()||!tensor)return;
    uint64_t values=(uint64_t)token_count*FG_HYPER_WIDTH,bytes=values*4u;
    if(bytes>fg_vk_tensor_bytes(tensor))bytes=fg_vk_tensor_bytes(tensor);
    if(!bytes)return;
    float *buffer=malloc((size_t)bytes);
    if(!buffer)return;
    fg_error local={0};
    if(fg_vk_tensor_read(tensor,0,buffer,bytes,&local)==FG_OK){
        const float *f=buffer;double sum=0.0;float min=f[0],max=f[0];
        uint64_t count=bytes/4u;
        for(uint64_t i=0;i<count;i++){float v=f[i];sum+=v;if(v<min)min=v;if(v>max)max=v;}
        fprintf(stderr,"FG_NUMERICS rank=%u layer=%u phase=%s first=%u tokens=%u hash=%016llx "
            "sum=%.6f min=%.6g max=%.6g f0=%.6g fl=%.6g\n",rank,layer,phase,first_token,
            (unsigned)token_count,(unsigned long long)numerics_hash_bytes(buffer,(size_t)bytes),
            sum,min,max,f[0],f[count-1u]);
    }
    free(buffer);
}
static void numerics_trace_values_local(const char *phase,uint32_t rank,uint32_t layer,
    const float *values,uint64_t count){
    if(!numerics_trace_enabled()||!values||!count)return;
    double sum=0.0;float min=values[0],max=values[0];
    for(uint64_t i=0;i<count;i++){float v=values[i];sum+=v;if(v<min)min=v;if(v>max)max=v;}
    fprintf(stderr,"FG_NUMERICS rank=%u layer=%u phase=%s first=0 tokens=%llu hash=%016llx "
        "sum=%.6f min=%.6g max=%.6g f0=%.6g fl=%.6g\n",rank,layer,phase,
        (unsigned long long)count,(unsigned long long)numerics_hash_bytes(values,count*4u),
        sum,min,max,values[0],values[count-1u]);
}
static fg_status chained_layer_trace(uint32_t rank,fg_vk_context *vk,uint32_t layer,
    uint32_t token,const fg_vk_tensor *state,fg_error *err){
    if(!numerics_trace_enabled())return FG_OK;
    while(fg_vk_batch_active(vk)){fg_status pending=fg_vk_end(vk,err);if(pending!=FG_OK)return pending;}
    fg_status status=fg_vk_static_drain(vk,err);
    if(status!=FG_OK)return status;
    const float *values=fg_vk_tensor_map((fg_vk_tensor *)state);
    int32_t first_bad=-1;
    if(values)for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)if(!isfinite(values[i])){first_bad=(int32_t)i;break;}
    fprintf(stderr,"FG_NUMERICS_LAYER rank=%u token=%u layer=%u finite=%d first_bad=%d f0=%g\n",
        rank,token,layer,first_bad<0,first_bad,values?values[0]:0.0f);
    if(first_bad>=0){
        fg_error_set(err,FG_ERR_FORMAT,
            "rank %u decode layer %u token %u produced non-finite hidden at element %d value=%g",
            rank,layer,token,first_bad,values[first_bad]);
        return FG_ERR_FORMAT;
    }
    return FG_OK;
}

static fg_status finish_batch(fg_vk_context *vk,fg_status status,fg_error *err){
    if(status==FG_OK)status=fg_vk_end(vk,err);
    if(status!=FG_OK&&fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    return status;
}

#define FG_HC_DOWN_SPLITS 8u
_Static_assert(FG_Q38_PREFILL_TILE_WORDS==FG_VK_PREFILL_TILE_WORDS,
               "owner grouped prefill tile geometry");

/* Per-frame decode/prefill state.  Slot 0 aliases the executor's existing
 * transient tensors; slots 1..FG_OWNER_SLOT_COUNT-1 own dedicated storage so
 * that many frames can be in flight one layer apart.  Only the state consumed
 * after the routed fire is duplicated: the front-half transients are dead for
 * a frame once its begin returns.  Worker owners only ever execute their own
 * block on slot 0, so their extra slots are not allocated. */
#define FG_OWNER_SLOT_COUNT 4u
typedef struct fg_owner_pending_write {bool active,skip;uint32_t layer,token;const fg_vk_tensor *hyper,*block,*injection;fg_vk_tensor *output;} fg_owner_pending_write;

typedef struct fg_owner_prefill_slot {
    bool active;
    uint32_t layer,first_token;
    uint16_t token_count;
    const fg_vk_tensor *residual;
    uint16_t expert_ids[FG_PREFILL_MAX_PAIRS];
    float gates[FG_PREFILL_MAX_PAIRS];
} fg_owner_prefill_slot;

typedef struct fg_owner_decode_slot {
    bool active,fire_called,ep_trace;
    uint32_t layer,token;
    const fg_vk_tensor *residual;
    fg_vk_tensor *injection,*shared_output,*shared_scalar,*reduced;
    fg_vk_tensor *ping[2];
    uint16_t expert_ids[FG_TOP_K];
    float gates[FG_TOP_K];
    fg_owner_pending_write pending_write;
    fg_owner_expert_collect_fn collect;
    void *dispatch_context;
    double t_begin,t_sync1,t_fire,t_sync2,t_topk,t_write;
    uint64_t trace_start,trace_sync1,trace_fire,trace_sync2;
    fg_vk_counters counters_before;
} fg_owner_decode_slot;

struct fg_owner_executor {
    fg_model *model;
    uint32_t max_tokens;
    uint32_t hc_inject_pieces;
    bool replicated;
    uint32_t session_count;
    uint32_t active_session;
    fg_vk_tensor *hyper_norm,*low,*hc_down_partials,*low_active,*up_logits,*inject_partials,*mixed,*injection,*hyper_output,*hyper_output_b;
    fg_vk_tensor *router_logits,*activation_q8k,*shared_gate,*shared_up,*shared_mid,*shared_output,*shared_scalar,*reduced;
    fg_vk_tensor *prefill_experts,*prefill_gates;
    fg_vk_tensor *gdn_qkv,*gdn_conv_output,*gdn_z,*gdn_alpha,*gdn_beta,*gdn_core,*gdn_output;
    fg_vk_tensor *ple_key,*ple_value,*ple_key_norm,*ple_query_norm,*ple_gated,*ple_gated_norm,*ple_output,*ple_added;
    struct {fg_vk_tensor *conv_state,*recurrent_state;} gdn_state[FG_OWNER_SESSION_MAX][FG_LAYER_COUNT];
    /* Device-side checkpoint shadows, allocated lazily by the depth-B session
     * transaction so the snapshot is a GPU copy, not an uncached host read. */
    struct {fg_vk_tensor *conv_state,*recurrent_state;} gdn_shadow[FG_OWNER_SESSION_MAX][FG_LAYER_COUNT];
    fg_vk_tensor *ples_shadow[FG_OWNER_SESSION_MAX];
    fg_vk_tensor *ples_state[FG_OWNER_SESSION_MAX];
    fg_vk_tensor *session_input[FG_OWNER_SESSION_MAX];
    /* Depth-B batch-2 block storage: a two-row residual ping-pong and the
     * two-token HC down partials (the shared transient is one row).  Allocated
     * lazily on the first batch block so single-token owners stay unchanged. */
    fg_vk_tensor *batch_ping[2];
    fg_vk_tensor *batch_hc_down_partials;
    fg_vk_tensor *attention_family_scratch;
    fg_vk_tensor *reduce_experts,*reduce_gates,*reduce_shared,*reduce_logits,*reduce_output;
    uint32_t reduce_tile_tokens;
    const float **prefill_slot_outputs;
    fg_expert_result decode_results[FG_GROUP_SIZE];
    fg_prefill_result prefill_results[FG_GROUP_SIZE];
    fg_owner_pending_write pending_write;
    fg_owner_decode_slot decode_slots[FG_OWNER_SLOT_COUNT];
    fg_owner_prefill_slot prefill_slots[FG_OWNER_SLOT_COUNT];
    fg_vk_tensor *static_run_output[FG_VK_STATIC_SLOTS];
    uint32_t static_run_session[FG_VK_STATIC_SLOTS];
    bool static_run_session_valid[FG_VK_STATIC_SLOTS];
    fg_qsa_session *qsa[FG_OWNER_SESSION_MAX];
};

/* Active-session state selectors.  Every stateful owner entry point resolves
 * GDN/PLE/QSA through these so a batch step's slot switch is one field. */
#define OWNER_GDN(e,layer) (&(e)->gdn_state[(e)->active_session][(layer)])
#define OWNER_PLE(e) ((e)->ples_state[(e)->active_session])
#define OWNER_QSA(e) ((e)->qsa[(e)->active_session])
static bool session_allocated(const fg_owner_executor *e,uint32_t session){
    return e&&session<e->session_count&&session<FG_OWNER_SESSION_MAX;
}
uint32_t fg_owner_session_count(const fg_owner_executor *executor){
    return executor?executor->session_count:0u;
}
uint32_t fg_owner_active_session(const fg_owner_executor *executor){
    return executor?executor->active_session:0u;
}
fg_status fg_owner_set_active_session(fg_owner_executor *executor,uint32_t session,
                                      fg_error *err){
    if(!executor||!session_allocated(executor,session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner session %u is not allocated",session);
        return FG_ERR_ARGUMENT;
    }
    executor->active_session=session;
    return FG_OK;
}
fg_vk_tensor *fg_owner_session_input(fg_owner_executor *executor,uint32_t session){
    if(!executor||!session_allocated(executor,session))return NULL;
    return executor->session_input[session];
}

static fg_status scratch(fg_vk_context *vk,uint64_t values,fg_vk_tensor **out,fg_error *err){return fg_vk_tensor_create(vk,values*4u,out,err);}
static fg_status family_view(fg_owner_executor *executor,uint64_t *offset,uint64_t bytes,
                             fg_vk_tensor **out,fg_error *err){
    fg_vk_tensor *arena=executor?executor->attention_family_scratch:NULL;
    if(!arena||!offset||!out||!bytes||*offset>fg_vk_tensor_bytes(arena)||
       bytes>fg_vk_tensor_bytes(arena)-*offset){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid shared attention scratch view");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_tensor_view(arena,*offset,bytes,out,err);
    if(status==FG_OK)*offset+=bytes;
    return status;
}
static fg_status create_attention_family_views(fg_owner_executor *executor,fg_error *err){
    uint32_t tokens=executor->max_tokens;uint64_t offset=0;fg_status status=FG_OK;
    const uint64_t gdn[]={
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u,
        (uint64_t)tokens*6144u*4u,(uint64_t)tokens*48u*4u,
        (uint64_t)tokens*48u*4u,(uint64_t)tokens*6144u*4u,
        (uint64_t)tokens*2560u*4u
    };
    fg_vk_tensor **gdn_fields[]={
        &executor->gdn_qkv,&executor->gdn_conv_output,&executor->gdn_z,
        &executor->gdn_alpha,&executor->gdn_beta,&executor->gdn_core,
        &executor->gdn_output
    };
    for(uint32_t i=0;status==FG_OK&&i<sizeof(gdn)/sizeof(gdn[0]);i++)
        status=family_view(executor,&offset,gdn[i],gdn_fields[i],err);
    if(status==FG_OK&&offset!=fg_qsa_gdn_scratch_bytes(tokens)){
        fg_error_set(err,FG_ERR_MISMATCH,"shared GDN scratch geometry mismatch");
        status=FG_ERR_MISMATCH;
    }
    offset=0;
    const uint64_t ple[]={
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*2560u*4u,
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u,
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u,
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u
    };
    fg_vk_tensor **ple_fields[]={
        &executor->ple_key,&executor->ple_value,&executor->ple_key_norm,
        &executor->ple_query_norm,&executor->ple_gated,&executor->ple_gated_norm,
        &executor->ple_output,&executor->ple_added
    };
    for(uint32_t i=0;status==FG_OK&&i<sizeof(ple)/sizeof(ple[0]);i++)
        status=family_view(executor,&offset,ple[i],ple_fields[i],err);
    if(status==FG_OK&&offset!=fg_qsa_ple_scratch_bytes(tokens)){
        fg_error_set(err,FG_ERR_MISMATCH,"shared PLE scratch geometry mismatch");
        status=FG_ERR_MISMATCH;
    }
    return status;
}

static fg_status create_transient_views(fg_owner_executor *executor,fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t tokens=executor->max_tokens;

    fg_vk_tensor *arena=executor->attention_family_scratch;
    uint64_t offset=0;
    fg_status status=FG_OK;
#define TRANSIENT(field,bytes) do{ \
        status=fg_vk_tensor_view(arena,offset,(bytes),&executor->field,err); \
        if(status==FG_OK)offset=fg_align_up_u64(offset+(bytes),FG_ALIGNMENT); \
    }while(0)
    TRANSIENT(hyper_norm,(uint64_t)tokens*10240u*4u);
    TRANSIENT(up_logits,(uint64_t)tokens*10240u*4u);
    TRANSIENT(low,(uint64_t)tokens*320u*4u);
    TRANSIENT(low_active,(uint64_t)tokens*320u*4u);
    TRANSIENT(inject_partials,(uint64_t)tokens*executor->hc_inject_pieces*4u*4u);
    TRANSIENT(hc_down_partials,(uint64_t)FG_HC_DOWN_SPLITS*320u*4u);
    TRANSIENT(router_logits,(uint64_t)tokens*FG_EXPERT_COUNT*4u);
    TRANSIENT(shared_gate,(uint64_t)tokens*640u*4u);
    TRANSIENT(shared_up,(uint64_t)tokens*640u*4u);
    TRANSIENT(shared_mid,(uint64_t)tokens*640u*4u);
    TRANSIENT(reduced,(uint64_t)tokens*FG_HIDDEN_SIZE*4u);
    if(status==FG_OK){
        offset=fg_align_up_u64(fg_qsa_attention_scratch_bytes(tokens),FG_ALIGNMENT)+
            fg_qsa_selection_scratch_bytes(manifest->native_context,tokens);
        TRANSIENT(mixed,(uint64_t)tokens*2560u*4u);
        /* GR injection has one gate value for every group and token. */
        TRANSIENT(injection,(uint64_t)tokens*FG_GROUP_SIZE*4u);
        TRANSIENT(shared_output,(uint64_t)tokens*FG_HIDDEN_SIZE*4u);
        TRANSIENT(shared_scalar,(uint64_t)tokens*4u);
    }
#undef TRANSIENT
    /* After experts finish, the GR/attention prefix is dead. Stage reduction
     * tiles there, stopping before the live reduction output and shared inputs. */
    if(status==FG_OK){
        uint64_t available=(const uint8_t *)fg_vk_tensor_const_map(executor->reduced)-
                           (const uint8_t *)fg_vk_tensor_const_map(arena);
        uint32_t tile=tokens<16u?tokens:16u;
        uint64_t gate_offset=0;
        while(tile){
            gate_offset=fg_align_up_u64((uint64_t)tile*FG_TOP_K*FG_HIDDEN_SIZE*4u,FG_ALIGNMENT);
            if(gate_offset+(uint64_t)tile*FG_TOP_K*4u<=available)break;
            tile--;
        }
        if(!tile){fg_error_set(err,FG_ERR_LIMIT,"no disjoint prefill reduction scratch");return FG_ERR_LIMIT;}
        executor->reduce_tile_tokens=tile;
        status=fg_vk_tensor_view(arena,0,(uint64_t)tile*FG_TOP_K*FG_HIDDEN_SIZE*4u,&executor->reduce_experts,err);
        if(status==FG_OK)status=fg_vk_tensor_view(arena,gate_offset,(uint64_t)tile*FG_TOP_K*4u,&executor->reduce_gates,err);
        if(status==FG_OK)status=fg_vk_tensor_view(executor->shared_output,0,FG_HIDDEN_SIZE*4u,&executor->reduce_shared,err);
        if(status==FG_OK)status=fg_vk_tensor_view(executor->shared_scalar,0,4u,&executor->reduce_logits,err);
        if(status==FG_OK)status=fg_vk_tensor_view(executor->reduced,0,FG_HIDDEN_SIZE*4u,&executor->reduce_output,err);
    }
    return status;
}

static fg_status create_decode_slots(fg_owner_executor *executor,fg_error *err){
    fg_owner_decode_slot *slot0=&executor->decode_slots[0];
    slot0->injection=executor->injection;slot0->shared_output=executor->shared_output;
    slot0->shared_scalar=executor->shared_scalar;slot0->reduced=executor->reduced;
    slot0->ping[0]=executor->hyper_output;slot0->ping[1]=executor->hyper_output_b;
    fg_vk_context *vk=fg_model_vk(executor->model);uint32_t tokens=executor->max_tokens;
    fg_status status=FG_OK;
    bool ring=executor->replicated&&fg_runtime_ring_enabled();
    uint32_t slot_count=executor->replicated&&!ring?FG_OWNER_SLOT_COUNT:1u;
    for(uint32_t slot=1u;status==FG_OK&&slot<slot_count;slot++){
        fg_owner_decode_slot *frame=&executor->decode_slots[slot];
        status=fg_vk_tensor_create(vk,(uint64_t)tokens*FG_GROUP_SIZE*4u,&frame->injection,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)tokens*FG_HIDDEN_SIZE*4u,&frame->shared_output,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)tokens*4u,&frame->shared_scalar,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)tokens*FG_HIDDEN_SIZE*4u,&frame->reduced,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)tokens*10240u*4u,&frame->ping[0],err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)tokens*10240u*4u,&frame->ping[1],err);
    }
    return status;
}

static fg_status owner_executor_create_impl(fg_owner_executor **out,fg_model *model,
                                            bool replicated,uint32_t sessions,fg_error *err){
    if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner executor arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    if(!sessions||sessions>FG_OWNER_SESSION_MAX){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner session count %u is outside 1..%u",
                     sessions,FG_OWNER_SESSION_MAX);
        return FG_ERR_ARGUMENT;
    }
    fg_owner_executor *executor=calloc(1,sizeof(*executor));if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate owner executor");return FG_ERR_OOM;}executor->model=model;fg_vk_context *vk=fg_model_vk(model);
    executor->session_count=sessions;executor->active_session=0;
    executor->hc_inject_pieces=fg_vk_hc_inject_pieces(vk);
    const fg_manifest *manifest=fg_model_manifest(model);executor->max_tokens=manifest->prefill_microbatch;executor->replicated=replicated;if(!executor->max_tokens||executor->max_tokens>FG_PREFILL_MAX_TOKENS){fg_owner_executor_destroy(executor);fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds owner executor limit");return FG_ERR_MISMATCH;}uint64_t tokens=executor->max_tokens;
    fg_status status=fg_vk_tensor_create(vk,(uint64_t)10240u*tokens*4u,&executor->hyper_output,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)10240u*tokens*4u,
                                                  &executor->hyper_output_b,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,tokens*FG_Q8K_ACTIVATION_BYTES,
                                                  &executor->activation_q8k,err);
    /* Prefill routing scratch is needed on every executor: worker owners run
     * their own prefill blocks with local routers. */
    if(status==FG_OK)status=fg_vk_tensor_create(
        vk,tokens*FG_TOP_K*4u,&executor->prefill_experts,err);
    if(status==FG_OK)status=fg_vk_tensor_create(
        vk,tokens*FG_TOP_K*4u,&executor->prefill_gates,err);
    if(status==FG_OK)status=fg_vk_tensor_create(
        vk,fg_qsa_attention_family_scratch_bytes(executor->max_tokens),
        &executor->attention_family_scratch,err);

    if(status==FG_OK)status=create_attention_family_views(executor,err);
    if(status==FG_OK)status=create_transient_views(executor,err);
    if(status==FG_OK)status=create_decode_slots(executor,err);
    bool ring=executor->replicated&&fg_runtime_ring_enabled();
    for(uint32_t session=0;status==FG_OK&&session<executor->session_count;session++){
        for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
            bool owned=manifest->layer_owner[layer]==fg_model_rank(model)||
                       (executor->replicated&&!ring);
            if(owned&&(layer&3u)!=3u){
                status=scratch(vk,10240u*4u,&executor->gdn_state[session][layer].conv_state,err);
                if(status==FG_OK)status=scratch(vk,48u*128u*128u,
                                                &executor->gdn_state[session][layer].recurrent_state,err);
                if(status==FG_OK){
                    memset(fg_vk_tensor_map(executor->gdn_state[session][layer].conv_state),0,
                           10240u*4u*4u);
                    memset(fg_vk_tensor_map(executor->gdn_state[session][layer].recurrent_state),0,
                           48u*128u*128u*4u);
                }
            }
        }
        if(status==FG_OK&&(manifest->layer_owner[1u]==fg_model_rank(model)||
                           (executor->replicated&&!ring)))
            status=scratch(vk,10240u*9u,&executor->ples_state[session],err);
        if(status==FG_OK&&executor->ples_state[session])
            memset(fg_vk_tensor_map(executor->ples_state[session]),0,10240u*9u*4u);
        if(status==FG_OK)
            status=scratch(vk,(uint64_t)FG_HYPER_WIDTH*4u,&executor->session_input[session],err);
    }
    if(status==FG_OK){
        executor->prefill_slot_outputs=calloc(
            (size_t)executor->max_tokens*FG_TOP_K,
            sizeof(*executor->prefill_slot_outputs));
        if(!executor->prefill_slot_outputs){
            fg_error_set(err,FG_ERR_OOM,
                         "allocate canonical prefill reduction slots");
            status=FG_ERR_OOM;
        }
    }
    if(status!=FG_OK){fg_owner_executor_destroy(executor);return status;}*out=executor;return FG_OK;
}
fg_status fg_owner_executor_create(fg_owner_executor **out,fg_model *model,fg_error *err){
    return owner_executor_create_impl(out,model,true,1u,err);
}
fg_status fg_owner_executor_create_worker(fg_owner_executor **out,fg_model *model,fg_error *err){
    return owner_executor_create_impl(out,model,false,1u,err);
}
fg_status fg_owner_executor_create_slots(fg_owner_executor **out,fg_model *model,
                                         uint32_t sessions,fg_error *err){
    return owner_executor_create_impl(out,model,true,sessions,err);
}
fg_status fg_owner_executor_create_worker_slots(fg_owner_executor **out,fg_model *model,
                                                uint32_t sessions,fg_error *err){
    return owner_executor_create_impl(out,model,false,sessions,err);
}
fg_vk_tensor *fg_owner_prefill_input(fg_owner_executor *executor){
    /*
     * GR read writes up_logits, so the embedding input must stay in the
     * ping-pong buffer rather than aliasing that transient output.
     */
    return executor?executor->hyper_output_b:NULL;
}
fg_vk_tensor *fg_owner_prefill_input_slot(fg_owner_executor *executor,uint32_t slot){
    if(!executor||slot>=FG_OWNER_SLOT_COUNT)return NULL;
    return executor->decode_slots[slot].ping[1];
}
uint64_t fg_owner_qsa_host_bytes(const fg_owner_executor *executor){
    if(!executor)return 0u;
    uint64_t bytes=0u;
    for(uint32_t session=0;session<executor->session_count;session++)
        bytes+=fg_qsa_session_host_bytes(executor->qsa[session]);
    return bytes;
}
void fg_owner_executor_destroy(fg_owner_executor *e){
    if(!e)return;
    for(uint32_t session=0;session<e->session_count;session++)
        fg_qsa_session_close(e->qsa[session]);
    free(e->prefill_slot_outputs);
    for(uint32_t session=0;session<e->session_count;session++){
        for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
            fg_vk_tensor_destroy(e->gdn_state[session][layer].recurrent_state);
            fg_vk_tensor_destroy(e->gdn_state[session][layer].conv_state);
            fg_vk_tensor_destroy(e->gdn_shadow[session][layer].recurrent_state);
            fg_vk_tensor_destroy(e->gdn_shadow[session][layer].conv_state);
        }
        fg_vk_tensor_destroy(e->ples_shadow[session]);
        fg_vk_tensor_destroy(e->ples_state[session]);
        fg_vk_tensor_destroy(e->session_input[session]);
    }
    fg_vk_tensor_destroy(e->batch_hc_down_partials);
    fg_vk_tensor_destroy(e->batch_ping[1]);
    fg_vk_tensor_destroy(e->batch_ping[0]);
    fg_vk_tensor_destroy(e->ple_added);
    fg_vk_tensor_destroy(e->ple_output);
    fg_vk_tensor_destroy(e->ple_gated_norm);
    fg_vk_tensor_destroy(e->ple_gated);
    fg_vk_tensor_destroy(e->ple_query_norm);
    fg_vk_tensor_destroy(e->ple_key_norm);
    fg_vk_tensor_destroy(e->ple_value);
    fg_vk_tensor_destroy(e->ple_key);
    fg_vk_tensor_destroy(e->gdn_output);
    fg_vk_tensor_destroy(e->gdn_core);
    fg_vk_tensor_destroy(e->gdn_beta);
    fg_vk_tensor_destroy(e->gdn_alpha);
    fg_vk_tensor_destroy(e->gdn_z);
    fg_vk_tensor_destroy(e->gdn_conv_output);
    fg_vk_tensor_destroy(e->gdn_qkv);
    for(uint32_t slot=1u;slot<FG_OWNER_SLOT_COUNT;slot++){
        fg_vk_tensor_destroy(e->decode_slots[slot].ping[1]);
        fg_vk_tensor_destroy(e->decode_slots[slot].ping[0]);
        fg_vk_tensor_destroy(e->decode_slots[slot].reduced);
        fg_vk_tensor_destroy(e->decode_slots[slot].shared_scalar);
        fg_vk_tensor_destroy(e->decode_slots[slot].shared_output);
        fg_vk_tensor_destroy(e->decode_slots[slot].injection);
    }
    fg_vk_tensor_destroy(e->reduce_output);
    fg_vk_tensor_destroy(e->reduce_logits);
    fg_vk_tensor_destroy(e->reduce_shared);
    fg_vk_tensor_destroy(e->reduce_gates);
    fg_vk_tensor_destroy(e->reduce_experts);
    fg_vk_tensor_destroy(e->reduced);
    fg_vk_tensor_destroy(e->shared_scalar);
    fg_vk_tensor_destroy(e->shared_output);
    fg_vk_tensor_destroy(e->shared_mid);
    fg_vk_tensor_destroy(e->shared_up);
    fg_vk_tensor_destroy(e->shared_gate);
    fg_vk_tensor_destroy(e->router_logits);
    fg_vk_tensor_destroy(e->injection);
    fg_vk_tensor_destroy(e->mixed);
    fg_vk_tensor_destroy(e->inject_partials);
    fg_vk_tensor_destroy(e->up_logits);
    fg_vk_tensor_destroy(e->low_active);
    fg_vk_tensor_destroy(e->hc_down_partials);
    fg_vk_tensor_destroy(e->low);
    fg_vk_tensor_destroy(e->hyper_norm);
    fg_vk_tensor_destroy(e->attention_family_scratch);
    fg_vk_tensor_destroy(e->prefill_gates);
    fg_vk_tensor_destroy(e->prefill_experts);
    fg_vk_tensor_destroy(e->activation_q8k);
    fg_vk_tensor_destroy(e->hyper_output_b);
    fg_vk_tensor_destroy(e->hyper_output);
    free(e);
}
fg_status fg_owner_reset_state(fg_owner_executor *e,fg_error *err){
    if(!e){fg_error_set(err,FG_ERR_ARGUMENT,"owner state reset is null");return FG_ERR_ARGUMENT;}
    for(uint32_t session=0;session<e->session_count;session++)for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        if(e->gdn_state[session][layer].conv_state)memset(fg_vk_tensor_map(e->gdn_state[session][layer].conv_state),0,(size_t)fg_vk_tensor_bytes(e->gdn_state[session][layer].conv_state));
        if(e->gdn_state[session][layer].recurrent_state)memset(fg_vk_tensor_map(e->gdn_state[session][layer].recurrent_state),0,(size_t)fg_vk_tensor_bytes(e->gdn_state[session][layer].recurrent_state));
    }
    for(uint32_t session=0;session<e->session_count;session++)
        if(e->ples_state[session])memset(fg_vk_tensor_map(e->ples_state[session]),0,(size_t)fg_vk_tensor_bytes(e->ples_state[session]));
    e->active_session=0;
    memset(&e->pending_write,0,sizeof(e->pending_write));
    for(uint32_t slot=0;slot<FG_OWNER_SLOT_COUNT;slot++){memset(&e->decode_slots[slot].pending_write,0,sizeof(e->decode_slots[slot].pending_write));e->decode_slots[slot].active=false;e->prefill_slots[slot].active=false;}
    for(uint32_t static_slot=0;static_slot<FG_VK_STATIC_SLOTS;static_slot++)
        e->static_run_session_valid[static_slot]=false;
    fg_status status=FG_OK;
    for(uint32_t session=0;status==FG_OK&&session<e->session_count;session++)
        if(e->qsa[session])status=fg_qsa_session_reset(e->qsa[session],err);
    return status;
}
fg_status fg_owner_qsa_checkpoint(fg_owner_executor *executor,fg_error *err){if(!executor||!OWNER_QSA(executor)){fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA checkpoint is unavailable");return FG_ERR_ARGUMENT;}return fg_qsa_session_checkpoint(OWNER_QSA(executor),err);}

static fg_vk_tensor *weight(fg_owner_executor *executor,uint32_t layer,const char *suffix,fg_error *err){char name[FG_TENSOR_NAME_MAX];int length=snprintf(name,sizeof(name),"blk.%u.%s",layer,suffix);if(length<0||(uint32_t)length>=sizeof(name)){fg_error_set(err,FG_ERR_LIMIT,"owner tensor name overflow");return NULL;}fg_vk_tensor *tensor=fg_model_tensor(executor->model,name);if(!tensor)fg_error_set(err,FG_ERR_MISMATCH,"owner rank is missing %s",name);return tensor;}
static fg_status dense_prefill(fg_owner_executor *executor,fg_vk_tensor *output,
                               const fg_vk_tensor *weights,
                               const fg_vk_tensor *input,uint32_t input_width,
                               uint32_t output_width,uint32_t tokens,
                               float scale,fg_error *err){

    fg_vk_context *vk=fg_model_vk(executor->model);
    if(fg_vk_tensor_get_format(weights)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED)
        return fg_vk_dense_q8_0_cooked(vk,output,weights,input,input_width,
                                       output_width,tokens,scale,err);
    return fg_vk_dense_q8_0_f32(vk,output,weights,input,input_width,output_width,
                                tokens,scale,err);
}
bool fg_owner_owns_layer(const fg_owner_executor *executor,uint32_t layer){if(!executor)return false;if(executor->replicated)return layer<FG_LAYER_COUNT;const fg_manifest *manifest=fg_model_manifest(executor->model);return layer<FG_LAYER_COUNT&&manifest->layer_owner[layer]==fg_model_rank(executor->model);}
static bool owns_layer(const fg_owner_executor *executor,uint32_t layer){
    return fg_owner_owns_layer(executor,layer);
}

static fg_status gr_read_batch_into(fg_owner_executor *e,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,uint32_t token_count,fg_vk_tensor *injection_tensor,fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err){
    if(!e||!hyper_input||!injection_tensor||!mixed||!residual||!injection||!token_count||token_count>e->max_tokens||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"gated residual batch is not on the layer owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}const char *prefix=ffn?"hc_ffn":"hc_attn";char suffix[48];fg_vk_tensor *norm_weight,*down_weight,*up_weight,*inject_weight;snprintf(suffix,sizeof(suffix),"%s_norm.weight",prefix);norm_weight=weight(e,layer,suffix,err);snprintf(suffix,sizeof(suffix),"%s_down.weight",prefix);down_weight=weight(e,layer,suffix,err);snprintf(suffix,sizeof(suffix),"%s_up.weight",prefix);up_weight=weight(e,layer,suffix,err);snprintf(suffix,sizeof(suffix),"%s_inject.weight",prefix);inject_weight=weight(e,layer,suffix,err);if(!norm_weight||!down_weight||!up_weight||!inject_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);    if(layer<=1u&&numerics_trace_enabled()){
        uint32_t rank=fg_model_rank(e->model);
        numerics_trace_values_local("W_HC_NORM",rank,layer,fg_vk_tensor_map((fg_vk_tensor *)norm_weight),fg_vk_tensor_bytes(norm_weight)/4u);
        numerics_trace_values_local("W_HC_DOWN",rank,layer,fg_vk_tensor_map((fg_vk_tensor *)down_weight),fg_vk_tensor_bytes(down_weight)/4u);
        numerics_trace_values_local("W_HC_UP",rank,layer,fg_vk_tensor_map((fg_vk_tensor *)up_weight),fg_vk_tensor_bytes(up_weight)/4u);
        numerics_trace_values_local("W_HC_INJ",rank,layer,fg_vk_tensor_map((fg_vk_tensor *)inject_weight),fg_vk_tensor_bytes(inject_weight)/4u);
        fprintf(stderr,"FG_NUMERICS_FMT rank=%u layer=%u hc_down_fmt=%u hc_up_fmt=%u attn_read_tokens=%u\n",
            rank,layer,(unsigned)fg_vk_tensor_get_format(down_weight),
            (unsigned)fg_vk_tensor_get_format(up_weight),(unsigned)token_count);
    }
    bool fused_down=token_count==1u&&fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
    fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->hyper_norm,hyper_input,norm_weight,FG_HIDDEN_SIZE,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_hc_inject_partial(vk,e->inject_partials,e->hyper_norm,inject_weight,FG_HIDDEN_SIZE,4u,token_count,e->hc_inject_pieces,err);if(status==FG_OK&&fused_down){fg_vk_next_dispatch_independent(vk);status=fg_vk_dense_q8_0_cooked_split_silu(vk,e->low,e->low_active,e->hc_down_partials,down_weight,e->hyper_norm,10240u,320u,1u,FG_HC_DOWN_SPLITS,1.0f,0.25f,err);}else if(status==FG_OK){status=dense_prefill(e,e->low,down_weight,e->hyper_norm,10240u,320u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_silu_scaled(vk,e->low_active,e->low,token_count*320u,0.25f,err);}if(status==FG_OK)status=dense_prefill(e,e->up_logits,up_weight,e->low_active,320u,10240u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_gr_mix_partial(vk,e->mixed,injection_tensor,e->hyper_norm,e->up_logits,e->inject_partials,FG_HIDDEN_SIZE,4u,token_count,e->hc_inject_pieces,err);status=finish_batch(vk,status,err);if(status==FG_OK){*mixed=e->mixed;*residual=hyper_input;*injection=injection_tensor;}return status;
}
fg_status fg_owner_gr_read_batch(fg_owner_executor *e,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,uint32_t token_count,fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err){return gr_read_batch_into(e,layer,ffn,hyper_input,token_count,e?e->injection:NULL,mixed,residual,injection,err);}

fg_status fg_owner_gr_read(fg_owner_executor *e,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err){return fg_owner_gr_read_batch(e,layer,ffn,hyper_input,1u,mixed,residual,injection,err);}

static fg_status owner_moe_routes(fg_owner_executor *e,uint32_t layer,
    const fg_vk_tensor *hidden,uint16_t token_count,uint16_t *expert_ids,
    float *gates,const uint8_t **activation,fg_error *err){
    if(!e||!hidden||!token_count||token_count>e->max_tokens||!expert_ids||
       !gates||!activation||!owns_layer(e,layer)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MoE routing arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor *router=weight(e,layer,"ffn_gate_inp.weight",err);
    if(!router)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router,
        hidden,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,token_count,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,
        hidden,FG_HIDDEN_SIZE,token_count,err);
    status=finish_batch(vk,status,err);
    if(status!=FG_OK)return status;
    const float *logits=fg_vk_tensor_map(e->router_logits);
    for(uint32_t token=0;status==FG_OK&&token<token_count;token++){
        uint32_t ids[FG_TOP_K];
        status=fg_q38_router_topk(logits+(uint64_t)token*FG_EXPERT_COUNT,
            FG_EXPERT_COUNT,FG_TOP_K,ids,gates+(uint64_t)token*FG_TOP_K,err);
        for(uint32_t slot=0;status==FG_OK&&slot<FG_TOP_K;slot++)
            expert_ids[(uint64_t)token*FG_TOP_K+slot]=(uint16_t)ids[slot];
    }
    if(status==FG_OK)*activation=fg_vk_tensor_map(e->activation_q8k);
    return status;
}

typedef struct shared_expert_work {
    fg_owner_executor *owner;
    const fg_vk_tensor *hidden;
    fg_vk_tensor *output,*scalar;
    uint32_t layer;
    uint16_t tokens;
} shared_expert_work;

static fg_status owner_prefill_routes(fg_owner_executor *e,uint32_t layer,
    const fg_vk_tensor *hidden,uint16_t tokens,uint16_t *expert_ids,
    float *gates,const uint8_t **activation,fg_error *err){
    fg_vk_tensor *router=weight(e,layer,"ffn_gate_inp.weight",err);
    if(!router)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router,
        hidden,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,tokens,err);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,
        hidden,FG_HIDDEN_SIZE,tokens,err);
    if(status==FG_OK)status=fg_vk_router_top10(vk,e->prefill_experts,
        e->prefill_gates,e->router_logits,FG_EXPERT_COUNT,tokens,err);
    status=finish_batch(vk,status,err);
    if(status!=FG_OK)return status;
    const uint32_t *ids=fg_vk_tensor_map(e->prefill_experts);
    const float *weights=fg_vk_tensor_map(e->prefill_gates);
    for(uint32_t i=0;i<(uint32_t)tokens*FG_TOP_K;i++){
        if(ids[i]>=FG_EXPERT_COUNT||!isfinite(weights[i])){
            fg_error_set(err,FG_ERR_FORMAT,"prefill router probabilities are not finite");
            return FG_ERR_FORMAT;
        }
        expert_ids[i]=(uint16_t)ids[i];gates[i]=weights[i];
    }
    *activation=fg_vk_tensor_map(e->activation_q8k);
    return FG_OK;
}

static fg_status owner_shared_expert(void *opaque,fg_error *err){
    const shared_expert_work *work=opaque;
    fg_owner_executor *e=work->owner;
    const fg_vk_tensor *hidden=work->hidden;
    uint32_t layer=work->layer,tokens=work->tokens;
    fg_vk_tensor *scalar=weight(e,layer,"ffn_gate_inp_shexp.weight",err);
    fg_vk_tensor *gate=weight(e,layer,"ffn_gate_shexp.weight",err);
    fg_vk_tensor *up=weight(e,layer,"ffn_up_shexp.weight",err);
    fg_vk_tensor *down=weight(e,layer,"ffn_down_shexp.weight",err);
    if(!scalar||!gate||!up||!down)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate,
        hidden,FG_HIDDEN_SIZE,640u,tokens,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up,
        hidden,FG_HIDDEN_SIZE,640u,tokens,1.0f,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,
        e->shared_up,tokens*640u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,work->output,down,
        e->shared_mid,640u,FG_HIDDEN_SIZE,tokens,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,work->scalar,scalar,
        hidden,FG_HIDDEN_SIZE,1u,tokens,err);
    return finish_batch(vk,status,err);
}

fg_status fg_owner_moe_prepare_batch(fg_owner_executor *e,uint32_t layer,
    const fg_vk_tensor *hidden,uint16_t tokens,uint16_t *expert_ids,float *gates,
    const uint8_t **activation,fg_error *err){
    fg_status status=owner_moe_routes(e,layer,hidden,tokens,expert_ids,gates,
                                    activation,err);
    shared_expert_work work={e,hidden,e->shared_output,e->shared_scalar,layer,tokens};
    if(status==FG_OK)status=owner_shared_expert(&work,err);
    return status;
}

fg_status fg_owner_moe_prepare(fg_owner_executor *e,uint32_t layer,
    const fg_vk_tensor *hidden,uint16_t expert_ids[FG_TOP_K],float gates[FG_TOP_K],
    const uint8_t **activation,fg_error *err){
    return fg_owner_moe_prepare_batch(e,layer,hidden,1u,expert_ids,gates,activation,err);
}

static fg_status moe_reduce_into(fg_owner_executor *e,uint32_t layer,uint32_t position,const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],const fg_expert_result *results,uint32_t result_count,fg_vk_tensor *shared_output,fg_vk_tensor *shared_scalar,fg_vk_tensor *reduced,fg_vk_tensor **output,fg_error *err){
    if(!e||!shared_output||!shared_scalar||!reduced||!expert_ids||!gates||!results||!result_count||result_count>FG_GROUP_SIZE||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid MoE reduction arguments");return FG_ERR_ARGUMENT;}
    if(!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"MoE reduction is not on the layer owner");return FG_ERR_MISMATCH;}
    const fg_manifest *manifest=fg_model_manifest(e->model);uint32_t owner=fg_model_rank(e->model);const float *slot_output[FG_TOP_K]={0};bool seen_expert[FG_EXPERT_COUNT]={0};
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){if(expert_ids[slot]>=FG_EXPERT_COUNT||seen_expert[expert_ids[slot]]||!isfinite(gates[slot])){fg_error_set(err,FG_ERR_FORMAT,"invalid canonical route slot %u",slot);return FG_ERR_FORMAT;}seen_expert[expert_ids[slot]]=true;}
    fg_status route_status=fg_expert_results_validate_route(manifest,layer,position,owner,expert_ids,results,result_count,err);if(route_status!=FG_OK)return route_status;
    /* Check for pre-reduced results (routing_slot 0xFF = worker already applied gates) */
    bool has_prereduced=false;
    const float *prereduced[FG_RANK_COUNT]={0};
    for(uint32_t r=0;r<result_count;r++)for(uint32_t i=0;i<results[r].selected_count;i++){if(results[r].routing_slots[i]==0xFFu){uint32_t source=results[r].source_rank;if(prereduced[source]){fg_error_set(err,FG_ERR_MISMATCH,"duplicate pre-reduced expert result from rank %u",source);return FG_ERR_MISMATCH;}has_prereduced=true;prereduced[source]=results[r].outputs[i];}else{slot_output[results[r].routing_slots[i]]=results[r].outputs[i];}}
    if(!has_prereduced){for(uint32_t slot=0;slot<FG_TOP_K;slot++){if(!slot_output[slot]){fg_error_set(err,FG_ERR_MISMATCH,"missing expert result slot %u",slot);return FG_ERR_MISMATCH;}}}
    float shared_scale=1.0f/(1.0f+expf(-*(const float *)fg_vk_tensor_map(shared_scalar)));
    /* Cache-friendly reduction: copy GPU-mapped (write-combining) data to stack,
       reduce in L1, then write result back.  The WC mapping makes scattered reads
       ~20x slower than cached DRAM; this copy+reduce pattern eliminates that. */
    float shared_local[FG_HIDDEN_SIZE],result_local[FG_HIDDEN_SIZE];
    fg_status read_status=fg_vk_tensor_read(shared_output,0,shared_local,
        FG_HIDDEN_SIZE*sizeof(float),err);
    if(read_status!=FG_OK)return read_status;
    for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]=shared_scale*shared_local[element];
    if(has_prereduced){for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(prereduced[rank])for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]+=prereduced[rank][element];
    /* Add any non-pre-reduced slot outputs (e.g., local experts on coordinator) */
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){if(slot_output[slot]){float g=gates[slot];const float *out=slot_output[slot];for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]=fmaf(g,out[element],result_local[element]);}}
    }else{for(uint32_t slot=0;slot<FG_TOP_K;slot++){float g=gates[slot];const float *out=slot_output[slot];for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]=fmaf(g,out[element],result_local[element]);}}
    fg_status write_status=fg_vk_tensor_write(reduced,0,result_local,
        FG_HIDDEN_SIZE*sizeof(float),err);
    if(write_status!=FG_OK)return write_status;
    *output=reduced;return FG_OK;
}
fg_status fg_owner_moe_reduce(fg_owner_executor *e,uint32_t layer,uint32_t position,const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],const fg_expert_result *results,uint32_t result_count,fg_vk_tensor **output,fg_error *err){return moe_reduce_into(e,layer,position,expert_ids,gates,results,result_count,e?e->shared_output:NULL,e?e->shared_scalar:NULL,e?e->reduced:NULL,output,err);}

static fg_status moe_reduce_batch_into(fg_owner_executor *e,uint32_t layer,uint32_t first_position,uint16_t token_count,const uint16_t *expert_ids,const float *gates,const fg_prefill_result *results,uint32_t result_count,fg_vk_tensor *shared_output,fg_vk_tensor *shared_scalar,fg_vk_tensor *reduced,fg_vk_tensor **output,fg_error *err){
    if(!e||!token_count||token_count>e->max_tokens||!expert_ids||!gates||!results||!result_count||result_count>FG_GROUP_SIZE||!shared_output||!shared_scalar||!reduced||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner MoE batch reduction arguments");return FG_ERR_ARGUMENT;}fg_status status=fg_prefill_results_validate_route(fg_model_manifest(e->model),layer,first_position,fg_model_rank(e->model),token_count,expert_ids,results,result_count,err);if(status!=FG_OK)return status;
    const float *rank_outputs[FG_RANK_COUNT]={0};
    for(uint32_t r=0;r<result_count;r++)rank_outputs[results[r].source_rank]=results[r].outputs;
    fg_vk_context *vk=fg_model_vk(e->model);
    if(fg_vk_batch_active(vk)){
        fg_error_set(err,FG_ERR_MISMATCH,"prefill reduction requires completed expert work");
        return FG_ERR_MISMATCH;
    }
    fg_vk_tensor *shared_view=NULL,*logits_view=NULL,*output_view=NULL;
    if(status==FG_OK)status=fg_vk_tensor_view(shared_output,0,fg_vk_tensor_bytes(shared_output),&shared_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(shared_scalar,0,fg_vk_tensor_bytes(shared_scalar),&logits_view,err);
    if(status==FG_OK)status=fg_vk_tensor_view(reduced,0,fg_vk_tensor_bytes(reduced),&output_view,err);
    float *staged=fg_vk_tensor_map(e->reduce_experts);
    for(uint32_t first=0;status==FG_OK&&first<token_count;first+=e->reduce_tile_tokens){
        uint32_t count=token_count-first;
        if(count>e->reduce_tile_tokens)count=e->reduce_tile_tokens;
        float rank_gates[16u*FG_TOP_K]={0};
        memset(staged,0,(size_t)count*FG_TOP_K*FG_HIDDEN_SIZE*4u);
        /* Fixed ascending-rank order, independent of response arrival order. */
        for(uint32_t t=0;t<count;t++)for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            if(!rank_outputs[rank])continue;
            memcpy(staged+((uint64_t)t*FG_TOP_K+rank)*FG_HIDDEN_SIZE,
                   rank_outputs[rank]+(uint64_t)(first+t)*FG_HIDDEN_SIZE,FG_HIDDEN_SIZE*4u);
            rank_gates[t*FG_TOP_K+rank]=1.0f;
        }
        status=fg_vk_tensor_write(e->reduce_gates,0,rank_gates,(uint64_t)count*FG_TOP_K*4u,err);
        uint64_t offset=(uint64_t)first*FG_HIDDEN_SIZE*4u,bytes=(uint64_t)count*FG_HIDDEN_SIZE*4u;
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(shared_view,shared_output,offset,bytes,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(logits_view,shared_scalar,(uint64_t)first*4u,(uint64_t)count*4u,err);
        if(status==FG_OK)status=fg_vk_tensor_view_rebind(output_view,reduced,offset,bytes,err);
        /* Each dispatch completes before its staging tile is overwritten. */
        if(status==FG_OK)status=fg_vk_moe_prefill_reduce(vk,output_view,
            e->reduce_experts,e->reduce_gates,shared_view,logits_view,
            FG_HIDDEN_SIZE,count,err);
    }
    fg_vk_tensor_destroy(shared_view);fg_vk_tensor_destroy(logits_view);fg_vk_tensor_destroy(output_view);
    if(status==FG_OK)*output=reduced;
    return status;
}
fg_status fg_owner_moe_reduce_batch(fg_owner_executor *e,uint32_t layer,uint32_t first_position,uint16_t token_count,const uint16_t *expert_ids,const float *gates,const fg_prefill_result *results,uint32_t result_count,fg_vk_tensor **output,fg_error *err){return moe_reduce_batch_into(e,layer,first_position,token_count,expert_ids,gates,results,result_count,e?e->shared_output:NULL,e?e->shared_scalar:NULL,e?e->reduced:NULL,output,err);}

static fg_status gr_write_batch_into(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,uint32_t token_count,fg_vk_tensor *ping_a,fg_vk_tensor *ping_b,fg_vk_tensor **output,fg_error *err){if(!executor||!hyper_input||!block_output||!injection||!ping_a||!ping_b||!token_count||token_count>executor->max_tokens||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual batch write arguments");return FG_ERR_ARGUMENT;}fg_vk_tensor *dst=hyper_input!=ping_a?ping_a:ping_b;fg_status status=fg_vk_gr_write(fg_model_vk(executor->model),dst,hyper_input,block_output,injection,FG_HIDDEN_SIZE,4u,token_count,err);if(status==FG_OK)*output=dst;return status;}

fg_status fg_owner_gr_write_batch(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,uint32_t token_count,fg_vk_tensor **output,fg_error *err){return gr_write_batch_into(executor,hyper_input,block_output,injection,token_count,executor?executor->hyper_output:NULL,executor?executor->hyper_output_b:NULL,output,err);}

fg_status fg_owner_gr_write(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,fg_vk_tensor **output,fg_error *err){return fg_owner_gr_write_batch(executor,hyper_input,block_output,injection,1u,output,err);}

static fg_status defer_gr_write(fg_owner_executor *executor,fg_owner_pending_write *pending,uint32_t layer,uint32_t token,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,fg_vk_tensor *ping_a,fg_vk_tensor *ping_b,fg_vk_tensor **output,fg_error *err){if(!executor||!pending||!hyper_input||!block_output||!injection||!output||pending->active){fg_error_set(err,FG_ERR_MISMATCH,"invalid or duplicate deferred residual write");return FG_ERR_MISMATCH;}fg_vk_tensor *destination=hyper_input!=ping_a?ping_a:ping_b;pending->active=true;pending->layer=layer;pending->token=token;pending->hyper=hyper_input;pending->block=block_output;pending->injection=injection;pending->output=destination;*output=destination;return FG_OK;}

static fg_status flush_gr_write(fg_owner_executor *executor,fg_owner_pending_write *pending,fg_error *err){if(!executor||!pending||!pending->active){fg_error_set(err,FG_ERR_MISMATCH,"deferred residual write is unavailable");return FG_ERR_MISMATCH;}fg_status status=FG_OK;if(!pending->skip){status=fg_vk_gr_write(fg_model_vk(executor->model),pending->output,pending->hyper,pending->block,pending->injection,FG_HIDDEN_SIZE,4u,1u,err);if(status==FG_OK)numerics_trace_tensor("DOUT_FLUSH",fg_model_rank(executor->model),pending->layer,pending->token,1u,pending->output,err);}if(status==FG_OK)memset(pending,0,sizeof(*pending));return status;}

fg_status fg_owner_gdn_decode(fg_owner_executor *executor,uint32_t layer,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    if(!executor||!hidden||!output||!owns_layer(executor,layer)||(layer&3u)==3u||!OWNER_GDN(executor,layer)->conv_state||!OWNER_GDN(executor,layer)->recurrent_state){fg_error_set(err,FG_ERR_MISMATCH,"GDN decode is not on an owned linear-attention layer");return FG_ERR_MISMATCH;}
    fg_vk_tensor *qkv_weight=weight(executor,layer,"attn_qkv.weight",err),*z_weight=weight(executor,layer,"attn_gate.weight",err),*alpha_weight=weight(executor,layer,"ssm_alpha.weight",err),*beta_weight=weight(executor,layer,"ssm_beta.weight",err),*conv_weight=weight(executor,layer,"ssm_conv1d.weight",err),*a_decay=weight(executor,layer,"ssm_a",err),*dt_bias=weight(executor,layer,"ssm_dt.bias",err),*norm_weight=weight(executor,layer,"ssm_norm.weight",err),*out_weight=weight(executor,layer,"ssm_out.weight",err);if(!qkv_weight||!z_weight||!alpha_weight||!beta_weight||!conv_weight||!a_decay||!dt_bias||!norm_weight||!out_weight)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(executor->model);
    if(layer<=1u&&numerics_trace_enabled()){
        uint32_t rank=fg_model_rank(executor->model);
        numerics_trace_values_local("W_GDN_QKV",rank,layer,fg_vk_tensor_map(qkv_weight),fg_vk_tensor_bytes(qkv_weight)/4u);
        numerics_trace_values_local("W_GDN_GATE",rank,layer,fg_vk_tensor_map(z_weight),fg_vk_tensor_bytes(z_weight)/4u);
        numerics_trace_values_local("W_GDN_CONV",rank,layer,fg_vk_tensor_map(conv_weight),fg_vk_tensor_bytes(conv_weight)/4u);
        numerics_trace_values_local("W_GDN_OUT",rank,layer,fg_vk_tensor_map(out_weight),fg_vk_tensor_bytes(out_weight)/4u);
        numerics_trace_values_local("W_GDN_ALPHA",rank,layer,fg_vk_tensor_map(alpha_weight),fg_vk_tensor_bytes(alpha_weight)/4u);
        numerics_trace_values_local("W_GDN_BETA",rank,layer,fg_vk_tensor_map(beta_weight),fg_vk_tensor_bytes(beta_weight)/4u);
        numerics_trace_values_local("W_GDN_A",rank,layer,fg_vk_tensor_map(a_decay),fg_vk_tensor_bytes(a_decay)/4u);
        numerics_trace_values_local("W_GDN_DT",rank,layer,fg_vk_tensor_map(dt_bias),fg_vk_tensor_bytes(dt_bias)/4u);
        numerics_trace_values_local("W_GDN_NORM",rank,layer,fg_vk_tensor_map(norm_weight),fg_vk_tensor_bytes(norm_weight)/4u);
        fprintf(stderr,"FG_NUMERICS_FMT rank=%u layer=%u qkv_fmt=%u gate_fmt=%u\n",
            rank,layer,(unsigned)fg_vk_tensor_get_format(qkv_weight),
            (unsigned)fg_vk_tensor_get_format(z_weight));
    }
    if(layer==0u)numerics_trace_values_local("GDN_IN",fg_model_rank(executor->model),
        layer,fg_vk_tensor_map((fg_vk_tensor *)hidden),FG_HIDDEN_SIZE);
    bool diagnostics=gdn_diag_enabled()&&!fg_vk_batch_active(vk);
    static atomic_int gdn_diag_done=0;
    if(diagnostics&&!gdn_diag_done){gdn_diag_done=1;const float *a_vals=fg_vk_tensor_map(a_decay),*dt_vals=fg_vk_tensor_map(dt_bias);fprintf(stderr,"GDN_DIAG layer=%u ssm_a[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f dt_bias[0..3]=%.4f,%.4f,%.4f,%.4f\n",layer,a_vals[0],a_vals[1],a_vals[2],a_vals[3],a_vals[4],a_vals[5],a_vals[6],a_vals[7],dt_vals[0],dt_vals[1],dt_vals[2],dt_vals[3]);}
    fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"gdn_projection",err):FG_OK;
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_gdn_project_decode(vk,executor->gdn_qkv,executor->gdn_z,executor->gdn_alpha,executor->gdn_beta,qkv_weight,z_weight,alpha_weight,beta_weight,hidden,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gdn_recurrent",err);
    if(status==FG_OK)status=fg_vk_gdn_conv_decode(vk,executor->gdn_conv_output,OWNER_GDN(executor,layer)->conv_state,executor->gdn_qkv,conv_weight,10240u,err);
    if(status==FG_OK)status=fg_vk_gdn_recurrent_algebraic(vk,executor->gdn_core,OWNER_GDN(executor,layer)->recurrent_state,executor->gdn_conv_output,executor->gdn_z,executor->gdn_alpha,executor->gdn_beta,a_decay,dt_bias,norm_weight,48u,16u,128u,1e-6f,err);
    status=finish_batch(vk,status,err);
    if(status==FG_OK&&layer==0u&&numerics_trace_enabled()){
        numerics_trace_values_local("GDN_QKV",fg_model_rank(executor->model),layer,
            fg_vk_tensor_map(executor->gdn_qkv),10240u);
        numerics_trace_values_local("GDN_CONV",fg_model_rank(executor->model),layer,
            fg_vk_tensor_map(executor->gdn_conv_output),10240u);
        numerics_trace_values_local("GDN_CORE",fg_model_rank(executor->model),layer,
            fg_vk_tensor_map(executor->gdn_core),6144u);
    }
    if(status==FG_OK&&layer==0u&&diagnostics){const float *alpha_vals=fg_vk_tensor_map(executor->gdn_alpha),*a_vals2=fg_vk_tensor_map(a_decay),*dt_vals2=fg_vk_tensor_map(dt_bias);float sp0=alpha_vals[0]+dt_vals2[0];sp0=sp0>0.0f?sp0+logf(1.0f+expf(-sp0)):logf(1.0f+expf(sp0));float example_decay=expf(a_vals2[0]*sp0);fprintf(stderr,"GDN_DECAY layer=0 alpha[0]=%.4f softplus=%.4f a[0]=%.4f decay[0]=%.6f\n",alpha_vals[0],sp0,a_vals2[0],example_decay);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gdn_output",err);
    if(status==FG_OK){status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_output,out_weight,executor->gdn_core,6144u,2560u,1u,1.0f,err);status=finish_batch(vk,status,err);}if(status==FG_OK&&layer==0u)numerics_trace_values_local("GDN_OUT",fg_model_rank(executor->model),layer,fg_vk_tensor_map(executor->gdn_output),FG_HIDDEN_SIZE);if(status==FG_OK){*output=executor->gdn_output;}return status;
}

static fg_status owner_gdn_prefill(fg_owner_executor *executor,uint32_t layer,
                                   uint32_t token_count,const fg_vk_tensor *hidden,
                                   fg_vk_tensor **output,
                                   fg_error *err){
    if(!executor||!hidden||!output||!token_count||token_count>executor->max_tokens||
       !owns_layer(executor,layer)||(layer&3u)==3u||
       !OWNER_GDN(executor,layer)->conv_state||
       !OWNER_GDN(executor,layer)->recurrent_state){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "GDN prefill is not on an owned linear-attention layer or exceeds the sealed microbatch");
        return FG_ERR_MISMATCH;
    }
    fg_vk_tensor *qkv_weight=weight(executor,layer,"attn_qkv.weight",err),*z_weight=weight(executor,layer,"attn_gate.weight",err),*alpha_weight=weight(executor,layer,"ssm_alpha.weight",err),*beta_weight=weight(executor,layer,"ssm_beta.weight",err),*conv_weight=weight(executor,layer,"ssm_conv1d.weight",err),*a_decay=weight(executor,layer,"ssm_a",err),*dt_bias=weight(executor,layer,"ssm_dt.bias",err),*norm_weight=weight(executor,layer,"ssm_norm.weight",err),*out_weight=weight(executor,layer,"ssm_out.weight",err);if(!qkv_weight||!z_weight||!alpha_weight||!beta_weight||!conv_weight||!a_decay||!dt_bias||!norm_weight||!out_weight)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(executor->model);
    bool profiling=fg_vk_profile_active(vk);
    fg_status status=profiling?
        fg_vk_profile_set_scope(vk,"gdn_prefill_projection",err):FG_OK;
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_qkv,qkv_weight,hidden,2560u,10240u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_z,z_weight,hidden,2560u,6144u,token_count,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,executor->gdn_alpha,alpha_weight,hidden,2560u,48u,token_count,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,executor->gdn_beta,beta_weight,hidden,2560u,48u,token_count,err);
    if(status==FG_OK&&profiling)
        status=fg_vk_profile_set_scope(vk,"gdn_prefill_convolution",err);
    if(status==FG_OK)status=fg_vk_gdn_conv_prefill(vk,executor->gdn_conv_output,OWNER_GDN(executor,layer)->conv_state,executor->gdn_qkv,conv_weight,10240u,token_count,err);
    if(status==FG_OK&&profiling)
        status=fg_vk_profile_set_scope(vk,"gdn_recurrent_prefill",err);
    if(status==FG_OK)status=fg_vk_gdn_recurrent_prefill_chunked(vk,executor->gdn_core,
            OWNER_GDN(executor,layer)->recurrent_state,
            executor->gdn_conv_output,executor->gdn_z,executor->gdn_alpha,
            executor->gdn_beta,a_decay,dt_bias,norm_weight,
            token_count,1e-6f,err);
    if(status==FG_OK&&profiling)
        status=fg_vk_profile_set_scope(vk,"gdn_prefill_output",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_output,out_weight,executor->gdn_core,6144u,2560u,token_count,1.0f,err);
    status=finish_batch(vk,status,err);
    if(status==FG_OK)*output=executor->gdn_output;
    return status;
}

fg_status fg_owner_gdn_prefill(fg_owner_executor *executor,uint32_t layer,
                               uint32_t token_count,const fg_vk_tensor *hidden,
                               fg_vk_tensor **output,fg_error *err){
    return owner_gdn_prefill(executor,layer,token_count,hidden,output,err);
}

/* PLE residuals remain live through GR read and attention. Keep them in the
 * protected ping-pong storage, outside the aliased attention scratch arena. */
static fg_status ple_decode_into(fg_owner_executor *e,const fg_vk_tensor *hyper,const fg_vk_tensor *embedding,fg_vk_tensor *ping_a,fg_vk_tensor *ping_b,fg_vk_tensor **output,fg_error *err){
    if(!e||!hyper||!embedding||!ping_a||!ping_b||!output||!owns_layer(e,1u)||!OWNER_PLE(e)){fg_error_set(err,FG_ERR_MISMATCH,"PLE decode is not on the layer-1 owner");return FG_ERR_MISMATCH;}fg_vk_tensor *key_weight=weight(e,1u,"ple_key.weight",err),*value_weight=weight(e,1u,"ple_value.weight",err),*key_norm=weight(e,1u,"ple_norm_key.weight",err),*query_norm=weight(e,1u,"ple_norm_query.weight",err),*conv_norm=weight(e,1u,"ple_norm_conv.weight",err),*conv_weight=weight(e,1u,"ple_conv1d.weight",err);if(!key_weight||!value_weight||!key_norm||!query_norm||!conv_norm||!conv_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);fg_vk_tensor *destination=hyper!=ping_a?ping_a:ping_b;fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_key,key_weight,embedding,2560u,10240u,1u,1.0f,err);if(status==FG_OK)fg_vk_next_dispatch_independent(vk);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_value,value_weight,embedding,2560u,2560u,1u,1.0f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_key_norm,e->ple_key,key_norm,2560u,4u,1u,1e-6f,err);if(status==FG_OK)fg_vk_next_dispatch_independent(vk);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_query_norm,hyper,query_norm,2560u,4u,1u,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_gate(vk,e->ple_gated,e->ple_key_norm,e->ple_query_norm,e->ple_value,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_gated_norm,e->ple_gated,conv_norm,2560u,4u,1u,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_conv_decode_add(vk,destination,OWNER_PLE(e),e->ple_gated,e->ple_gated_norm,conv_weight,hyper,err);status=finish_batch(vk,status,err);if(status==FG_OK)*output=destination;return status;
}
fg_status fg_owner_ple_decode(fg_owner_executor *e,const fg_vk_tensor *hyper,const fg_vk_tensor *embedding,fg_vk_tensor **output,fg_error *err){return ple_decode_into(e,hyper,embedding,e?e->hyper_output:NULL,e?e->hyper_output_b:NULL,output,err);}

static fg_status ple_prefill_into(fg_owner_executor *e,const fg_vk_tensor *hyper,const fg_vk_tensor *embedding,fg_vk_tensor *ping_a,fg_vk_tensor *ping_b,uint32_t token_count,fg_vk_tensor **output,fg_error *err){
    if(!e||!hyper||!embedding||!ping_a||!ping_b||!output||!token_count||token_count>e->max_tokens||!owns_layer(e,1u)||!OWNER_PLE(e)){fg_error_set(err,FG_ERR_MISMATCH,"PLE prefill is not on the layer-1 owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}fg_vk_tensor *key_weight=weight(e,1u,"ple_key.weight",err),*value_weight=weight(e,1u,"ple_value.weight",err),*key_norm=weight(e,1u,"ple_norm_key.weight",err),*query_norm=weight(e,1u,"ple_norm_query.weight",err),*conv_norm=weight(e,1u,"ple_norm_conv.weight",err),*conv_weight=weight(e,1u,"ple_conv1d.weight",err);if(!key_weight||!value_weight||!key_norm||!query_norm||!conv_norm||!conv_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);fg_vk_tensor *destination=hyper!=ping_a?ping_a:ping_b;fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_key,key_weight,embedding,2560u,10240u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_value,value_weight,embedding,2560u,2560u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_key_norm,e->ple_key,key_norm,2560u,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_query_norm,hyper,query_norm,2560u,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_gate_prefill(vk,e->ple_gated,e->ple_key_norm,e->ple_query_norm,e->ple_value,token_count,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_gated_norm,e->ple_gated,conv_norm,2560u,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_conv_prefill(vk,e->ple_output,OWNER_PLE(e),e->ple_gated,e->ple_gated_norm,conv_weight,token_count,err);if(status==FG_OK)status=fg_vk_add_f32(vk,destination,hyper,e->ple_output,token_count*10240u,err);status=finish_batch(vk,status,err);if(status==FG_OK)*output=destination;return status;
}
fg_status fg_owner_ple_prefill(fg_owner_executor *e,const fg_vk_tensor *hyper,const fg_vk_tensor *embedding,uint32_t token_count,fg_vk_tensor **output,fg_error *err){return ple_prefill_into(e,hyper,embedding,e?e->hyper_output:NULL,e?e->hyper_output_b:NULL,token_count,output,err);}

static fg_status owner_qsa_open_check(fg_owner_executor *executor,uint32_t session,
                                      const char *state_path,fg_error *err){
    if(!executor||!state_path){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner QSA open arguments");
        return FG_ERR_ARGUMENT;
    }
    if(!session_allocated(executor,session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA session %u is not allocated",session);
        return FG_ERR_ARGUMENT;
    }
    if(executor->qsa[session]){
        fg_error_set(err,FG_ERR_MISMATCH,"owner QSA session is already open");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}
static fg_status qsa_open_plain_slot(fg_owner_executor *executor,uint32_t session,
                                     const char *state_path,bool create,fg_error *err){
    fg_status status=owner_qsa_open_check(executor,session,state_path,err);
    if(status!=FG_OK)return status;
    return fg_qsa_session_open(&executor->qsa[session],executor->model,state_path,create,err);
}
static fg_status qsa_open_state_slot_impl(fg_owner_executor *executor,uint32_t session,
    const char *state_path,uint32_t logical_context,uint32_t hot_tokens,uint32_t cache_pages,
    uint32_t batch_size,bool mirror,bool owned_only,fg_qsa_page_fetch_fn fetch_pages,
    void *fetch_opaque,fg_error *err){
    fg_status status=owner_qsa_open_check(executor,session,state_path,err);
    if(status!=FG_OK)return status;
    if(!mirror)
        return fg_qsa_session_open_state(&executor->qsa[session],executor->model,state_path,
                                         logical_context,hot_tokens,cache_pages,batch_size,err);
    if(executor->session_count>1u&&session!=0u){
        /* Session 1 gets its own attention scratch region view over the shared
         * arena: sessions execute sequentially, so aliasing session 0's scratch
         * is safe, but the open call builds its own views and must not disturb
         * the live session. */
        return fg_qsa_session_open_state_mirror_with_scratch(
            &executor->qsa[session],executor->model,state_path,logical_context,hot_tokens,
            cache_pages,batch_size,executor->attention_family_scratch,owned_only,
            fetch_pages,fetch_opaque,err);
    }
    return fg_qsa_session_open_state_mirror_with_scratch(
        &executor->qsa[session],executor->model,state_path,logical_context,hot_tokens,
        cache_pages,batch_size,executor->attention_family_scratch,owned_only,
        fetch_pages,fetch_opaque,err);
}
fg_status fg_owner_qsa_open(fg_owner_executor *executor,const char *state_path,bool create,fg_error *err){
    return qsa_open_plain_slot(executor,0u,state_path,create,err);
}
fg_status fg_owner_qsa_open_decode(fg_owner_executor *executor,const char *state_path,uint32_t resident_tokens,uint32_t batch_size,fg_error *err){
    if(!state_path){fg_error_set(err,FG_ERR_ARGUMENT,"decode QSA state path is null");return FG_ERR_ARGUMENT;}
    fg_status status=owner_qsa_open_check(executor,0u,state_path,err);
    if(status!=FG_OK)return status;
    return fg_qsa_session_open_decode(&executor->qsa[0u],executor->model,state_path,
                                      resident_tokens,batch_size,err);
}
fg_status fg_owner_qsa_open_state(fg_owner_executor *executor,const char *state_path,
                                  uint32_t logical_context,uint32_t hot_tokens,
                                  uint32_t cache_pages,uint32_t batch_size,fg_error *err){
    return fg_owner_qsa_open_state_slot(executor,0u,state_path,logical_context,hot_tokens,
                                        cache_pages,batch_size,err);
}
fg_status fg_owner_qsa_open_state_slot(fg_owner_executor *executor,uint32_t session,
                                       const char *state_path,uint32_t logical_context,
                                       uint32_t hot_tokens,uint32_t cache_pages,
                                       uint32_t batch_size,fg_error *err){
    return qsa_open_state_slot_impl(executor,session,state_path,logical_context,hot_tokens,
                                    cache_pages,batch_size,false,false,NULL,NULL,err);
}
fg_status fg_owner_qsa_open_state_mirror(fg_owner_executor *executor,const char *state_path,
                                         uint32_t logical_context,uint32_t hot_tokens,
                                         uint32_t cache_pages,uint32_t batch_size,
                                         bool owned_only,
                                         fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                         fg_error *err){
    return fg_owner_qsa_open_state_mirror_slot(executor,0u,state_path,logical_context,
                                               hot_tokens,cache_pages,batch_size,owned_only,
                                               fetch_pages,fetch_opaque,err);
}
fg_status fg_owner_qsa_open_state_mirror_slot(fg_owner_executor *executor,uint32_t session,
                                              const char *state_path,uint32_t logical_context,
                                              uint32_t hot_tokens,uint32_t cache_pages,
                                              uint32_t batch_size,bool owned_only,
                                              fg_qsa_page_fetch_fn fetch_pages,
                                              void *fetch_opaque,fg_error *err){
    return qsa_open_state_slot_impl(executor,session,state_path,logical_context,hot_tokens,
                                    cache_pages,batch_size,true,owned_only,fetch_pages,
                                    fetch_opaque,err);
}
bool fg_owner_qsa_ready(const fg_owner_executor *executor){
    return executor&&executor->qsa[0u];
}
bool fg_owner_qsa_ready_slot(const fg_owner_executor *executor,uint32_t session){
    return executor&&session_allocated(executor,session)&&executor->qsa[session];
}
fg_status fg_owner_qsa_open_mirror(fg_owner_executor *executor,uint32_t logical_context,
                                   uint32_t hot_tokens,uint32_t cache_pages,uint32_t batch_size,
                                   fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                   fg_error *err){
    return fg_owner_qsa_open_mirror_slot(executor,0u,logical_context,hot_tokens,cache_pages,
                                         batch_size,fetch_pages,fetch_opaque,err);
}
fg_status fg_owner_qsa_open_mirror_slot(fg_owner_executor *executor,uint32_t session,
                                        uint32_t logical_context,uint32_t hot_tokens,
                                        uint32_t cache_pages,uint32_t batch_size,
                                        fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                        fg_error *err){
    if(!executor){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner QSA mirror open");return FG_ERR_ARGUMENT;}
    if(!session_allocated(executor,session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA session %u is not allocated",session);
        return FG_ERR_ARGUMENT;
    }
    if(executor->qsa[session]){fg_error_set(err,FG_ERR_MISMATCH,"owner QSA session is already open");return FG_ERR_MISMATCH;}
    return fg_qsa_session_open_mirror_with_scratch(
        &executor->qsa[session],executor->model,logical_context,hot_tokens,cache_pages,batch_size,
        executor->attention_family_scratch,fetch_pages,fetch_opaque,err);
}
void fg_owner_qsa_set_tokens(fg_owner_executor *executor,uint32_t tokens){if(executor&&OWNER_QSA(executor))fg_qsa_session_set_tokens(OWNER_QSA(executor),tokens);}
fg_status fg_owner_qsa_frontier(const fg_owner_executor *executor,uint32_t session,
                                uint32_t tokens[FG_LAYER_COUNT]){
    if(!executor||!tokens||!session_allocated(executor,session))return FG_ERR_UNAVAILABLE;
    fg_qsa_session *qsa=executor->qsa[session];
    if(!qsa){memset(tokens,0,FG_LAYER_COUNT*sizeof(*tokens));return FG_OK;}
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
        tokens[layer]=fg_qsa_session_tokens(qsa,layer);
    return FG_OK;
}
fg_status fg_owner_qsa_rollback(fg_owner_executor *executor,uint32_t session,
                                const uint32_t tokens[FG_LAYER_COUNT],fg_error *err){
    if(!executor||!tokens||!session_allocated(executor,session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner QSA rollback");
        return FG_ERR_ARGUMENT;
    }
    fg_qsa_session *qsa=executor->qsa[session];
    if(!qsa)return FG_OK;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
        fg_qsa_session_set_layer_tokens(qsa,layer,tokens[layer]);
    return FG_OK;
}
void fg_owner_session_snapshot_release(fg_owner_session_checkpoint *snapshot){
    if(!snapshot)return;
    for(uint32_t i=0;i<snapshot->gdn_count;i++){
        free(snapshot->gdn[i].data);
        snapshot->gdn[i].data=NULL;
    }
    free(snapshot->ple_data);
    snapshot->ple_data=NULL;
    snapshot->ple_values=0u;
    snapshot->gdn_count=0u;
    snapshot->qsa_valid=false;
    snapshot->device=false;
    snapshot->valid=false;
}
fg_status fg_owner_session_snapshot(fg_owner_executor *executor,uint32_t session,
                                    fg_owner_session_checkpoint *snapshot,fg_error *err){
    if(!executor||!snapshot||!session_allocated(executor,session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner session snapshot");
        return FG_ERR_ARGUMENT;
    }
    memset(snapshot,0,sizeof(*snapshot));
    snapshot->session=session;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        fg_vk_tensor *tensors[2u]={executor->gdn_state[session][layer].conv_state,
                                   executor->gdn_state[session][layer].recurrent_state};
        for(uint32_t which=0u;which<2u;which++){
            fg_vk_tensor *tensor=tensors[which];
            if(!tensor)continue;
            if(snapshot->gdn_count>=FG_LAYER_COUNT*2u){
                fg_owner_session_snapshot_release(snapshot);
                fg_error_set(err,FG_ERR_LIMIT,"owner session snapshot overflow");
                return FG_ERR_LIMIT;
            }
            uint64_t values=fg_vk_tensor_bytes(tensor)/4u;
            float *data=malloc((size_t)(values*4u));
            if(!data){
                fg_owner_session_snapshot_release(snapshot);
                fg_error_set(err,FG_ERR_OOM,"allocate owner session snapshot");
                return FG_ERR_OOM;
            }
            fg_status status=fg_vk_tensor_read(tensor,0,data,values*4u,err);
            if(status!=FG_OK){
                free(data);
                fg_owner_session_snapshot_release(snapshot);
                return status;
            }
            snapshot->gdn[snapshot->gdn_count].tensor=tensor;
            snapshot->gdn[snapshot->gdn_count].data=data;
            snapshot->gdn[snapshot->gdn_count].values=values;
            snapshot->gdn_count++;
        }
    }
    fg_vk_tensor *ple=executor->ples_state[session];
    if(ple){
        uint64_t values=fg_vk_tensor_bytes(ple)/4u;
        snapshot->ple_data=malloc((size_t)(values*4u));
        if(!snapshot->ple_data){
            fg_owner_session_snapshot_release(snapshot);
            fg_error_set(err,FG_ERR_OOM,"allocate owner PLE session snapshot");
            return FG_ERR_OOM;
        }
        snapshot->ple=ple;
        snapshot->ple_values=values;
        fg_status status=fg_vk_tensor_read(ple,0,snapshot->ple_data,values*4u,err);
        if(status!=FG_OK){
            fg_owner_session_snapshot_release(snapshot);
            return status;
        }
    }
    snapshot->qsa_valid=fg_owner_qsa_frontier(executor,session,snapshot->qsa_tokens)==FG_OK;
    snapshot->valid=true;
    return FG_OK;
}
/* Device-side checkpoint: grow the per-session shadow tensors on first use and
 * copy GDN conv/recurrent + PLE state into them with one GPU transfer, so the
 * transaction never reads uncached device memory from the host. */
fg_status fg_owner_session_device_snapshot(fg_owner_executor *executor,uint32_t session,
                                           fg_owner_session_checkpoint *snapshot,
                                           fg_error *err){
    if(!executor||!snapshot||!session_allocated(executor,session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner session device snapshot");
        return FG_ERR_ARGUMENT;
    }
    memset(snapshot,0,sizeof(*snapshot));
    snapshot->session=session;
    snapshot->device=true;
    fg_vk_context *vk=fg_model_vk(executor->model);
    fg_vk_tensor *dst[FG_LAYER_COUNT*2u+1u];const fg_vk_tensor *src[FG_LAYER_COUNT*2u+1u];
    uint32_t count=0;
    fg_status status=FG_OK;
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        fg_vk_tensor *states[2u]={executor->gdn_state[session][layer].conv_state,
                                  executor->gdn_state[session][layer].recurrent_state};
        fg_vk_tensor **shadows[2u]={&executor->gdn_shadow[session][layer].conv_state,
                                    &executor->gdn_shadow[session][layer].recurrent_state};
        for(uint32_t which=0u;status==FG_OK&&which<2u;which++){
            if(!states[which])continue;
            if(!*shadows[which])
                status=fg_vk_tensor_create(vk,fg_vk_tensor_bytes(states[which]),
                                           shadows[which],err);
            if(status==FG_OK){
                src[count]=states[which];
                dst[count]=*shadows[which];
                count++;
            }
        }
    }
    if(status==FG_OK&&executor->ples_state[session]){
        if(!executor->ples_shadow[session])
            status=fg_vk_tensor_create(vk,fg_vk_tensor_bytes(executor->ples_state[session]),
                                       &executor->ples_shadow[session],err);
        if(status==FG_OK){
            src[count]=executor->ples_state[session];
            dst[count]=executor->ples_shadow[session];
            count++;
        }
    }
    if(status==FG_OK)status=fg_vk_copy_tensors(vk,dst,src,count,err);
    if(status!=FG_OK)return status;
    snapshot->qsa_valid=fg_owner_qsa_frontier(executor,session,snapshot->qsa_tokens)==FG_OK;
    snapshot->valid=true;
    return FG_OK;
}
fg_status fg_owner_session_rollback(fg_owner_executor *executor,
                                    const fg_owner_session_checkpoint *snapshot,fg_error *err){
    if(!executor||!snapshot||!snapshot->valid||
       !session_allocated(executor,snapshot->session)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner session rollback");
        return FG_ERR_ARGUMENT;
    }
    if(snapshot->device){
        fg_vk_context *vk=fg_model_vk(executor->model);
        fg_vk_tensor *dst[FG_LAYER_COUNT*2u+1u];const fg_vk_tensor *src[FG_LAYER_COUNT*2u+1u];
        uint32_t count=0;
        for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
            fg_vk_tensor *states[2u]={executor->gdn_state[snapshot->session][layer].conv_state,
                                      executor->gdn_state[snapshot->session][layer].recurrent_state};
            fg_vk_tensor *shadows[2u]={executor->gdn_shadow[snapshot->session][layer].conv_state,
                                       executor->gdn_shadow[snapshot->session][layer].recurrent_state};
            for(uint32_t which=0u;which<2u;which++){
                if(!states[which]||!shadows[which])continue;
                dst[count]=states[which];
                src[count]=shadows[which];
                count++;
            }
        }
        if(executor->ples_state[snapshot->session]&&
           executor->ples_shadow[snapshot->session]){
            dst[count]=executor->ples_state[snapshot->session];
            src[count]=executor->ples_shadow[snapshot->session];
            count++;
        }
        if(count){
            fg_status status=fg_vk_copy_tensors(vk,dst,src,count,err);
            if(status!=FG_OK)return status;
        }
        if(snapshot->qsa_valid)
            return fg_owner_qsa_rollback(executor,snapshot->session,
                                         snapshot->qsa_tokens,err);
        return FG_OK;
    }
    for(uint32_t i=0;i<snapshot->gdn_count;i++){
        fg_status status=fg_vk_tensor_write(snapshot->gdn[i].tensor,0,
            snapshot->gdn[i].data,snapshot->gdn[i].values*4u,err);
        if(status!=FG_OK)return status;
    }
    if(snapshot->ple_data){
        fg_status status=fg_vk_tensor_write(snapshot->ple,0,snapshot->ple_data,
                                            snapshot->ple_values*4u,err);
        if(status!=FG_OK)return status;
    }
    if(snapshot->qsa_valid)
        return fg_owner_qsa_rollback(executor,snapshot->session,snapshot->qsa_tokens,err);
    return FG_OK;
}
fg_expert_result *fg_owner_decode_results(fg_owner_executor *executor){
    return executor?executor->decode_results:NULL;
}
fg_vk_tensor *fg_owner_gdn_state_tensor(fg_owner_executor *executor,uint32_t layer,
                                        uint32_t slot){
    if(!executor||layer>=FG_LAYER_COUNT||slot>1u)return NULL;
    return slot==0u?OWNER_GDN(executor,layer)->conv_state:
                    OWNER_GDN(executor,layer)->recurrent_state;
}
fg_vk_tensor *fg_owner_ple_state_tensor(fg_owner_executor *executor){
    return executor?OWNER_PLE(executor):NULL;
}
uint32_t fg_owner_gdn_layers(const fg_owner_executor *executor,uint8_t *layers,
                             uint32_t capacity,fg_error *err){
    if(!executor||!layers){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN layer query");return 0u;}
    uint32_t count=0;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        if((layer&3u)==3u||!executor->gdn_state[0u][layer].conv_state)continue;
        if(count>=capacity){
            fg_error_set(err,FG_ERR_LIMIT,"GDN layer list exceeds capacity");
            return 0u;
        }
        layers[count++]=(uint8_t)layer;
    }
    return count;
}

fg_status fg_owner_qsa_decode(fg_owner_executor *executor,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!executor||!qsa||!owns_layer(executor,layer)||(layer&3u)!=3u){fg_error_set(err,FG_ERR_MISMATCH,"QSA decode is not on an initialized QSA layer owner");return FG_ERR_MISMATCH;}
    return fg_qsa_session_decode(qsa,layer,token,position,hidden,output,err);
}

fg_status fg_owner_qsa_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint32_t token_count,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!executor||!qsa||!owns_layer(executor,layer)||(layer&3u)!=3u||!token_count||token_count>executor->max_tokens){fg_error_set(err,FG_ERR_MISMATCH,"QSA prefill is not on an initialized QSA layer owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}
    return fg_qsa_session_prefill(qsa,layer,first_token,positions,token_count,hidden,output,err);
}
fg_status fg_owner_qsa_page_records(const fg_owner_executor *executor,uint32_t layer,
                                    uint32_t block,const uint8_t **records,fg_error *err){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!executor||!qsa){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA page lookup is unavailable");
        return FG_ERR_ARGUMENT;
    }
    return fg_qsa_session_page_records(qsa,layer,block,records,err);
}
fg_status fg_owner_qsa_warm_pages(fg_owner_executor *executor,uint32_t layer,
                                  const uint32_t *blocks,const uint8_t *records,
                                  uint32_t page_count,fg_error *err){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!executor||!qsa){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA mirror warm is unavailable");
        return FG_ERR_ARGUMENT;
    }
    return fg_qsa_session_warm_pages(qsa,layer,blocks,records,page_count,err);
}
bool fg_owner_qsa_page_cached(fg_owner_executor *executor,uint32_t layer,uint32_t block){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!qsa)return false;
    return fg_qsa_session_page_cached(qsa,layer,block);
}
void fg_owner_qsa_page_published(fg_owner_executor *executor,uint32_t layer,uint32_t block){
    if(executor&&OWNER_QSA(executor))fg_qsa_session_page_published(OWNER_QSA(executor),layer,block);
}
fg_status fg_owner_qsa_state_records(fg_owner_executor *executor,uint32_t layer,
                                     uint32_t block,uint8_t *records,fg_error *err){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!executor||!qsa){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"owner QSA state records are unavailable");
        return FG_ERR_UNAVAILABLE;
    }
    return fg_qsa_session_state_records(qsa,layer,block,records,err);
}
fg_status fg_owner_qsa_state_records_batch(fg_owner_executor *executor,uint32_t layer,
    const uint32_t *blocks,uint32_t page_count,uint8_t *records,fg_error *err){
    fg_qsa_session *qsa=executor?OWNER_QSA(executor):NULL;
    if(!executor||!qsa){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"owner QSA state records are unavailable");
        return FG_ERR_UNAVAILABLE;
    }
    return fg_qsa_session_state_records_batch(qsa,layer,blocks,page_count,records,err);
}

static float tensor_l2(const fg_vk_tensor *t,uint32_t n){const float *p=fg_vk_tensor_map((fg_vk_tensor *)t);if(!p)return -1.0f;double s=0.0;for(uint32_t i=0;i<n;i++)s+=(double)p[i]*p[i];return (float)sqrt(s/n);}

fg_status fg_owner_decode_layer(fg_owner_executor *e,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,fg_owner_expert_dispatch_fn dispatch,void *dispatch_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!position||!hyper_input||!dispatch||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"text layer decode is not on its owner");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embedding!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 PLE embedding presence mismatch");return FG_ERR_MISMATCH;}double t0=ts_ms();fg_vk_context *vk=fg_model_vk(e->model);const fg_vk_tensor *layer_input=hyper_input;if(layer==1u){fg_vk_tensor *ple_input=NULL;fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"ple",err):FG_OK;if(status==FG_OK)status=fg_owner_ple_decode(e,hyper_input,ngram_embedding,&ple_input,err);if(status!=FG_OK)return status;layer_input=ple_input;}double t_ple=ts_ms();
    int diag=token<30u; /* first ~11 decode tokens */
    /* ---- FUSED BATCH 1: gr_read(attn) + attention + gr_write + gr_read(FFN) + router ---- */
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;
    fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"gr_attn_read",err):FG_OK;if(status==FG_OK)status=fg_vk_begin(vk,err);if(status==FG_OK){status=fg_owner_gr_read(e,layer,false,layer_input,&mixed,&residual,&injection,err);}double t_gr1=ts_ms();
    if(status==FG_OK&&(layer&3u)==3u){
        status=fg_owner_qsa_decode(e,layer,token,position,mixed,&block,err);
    }else if(status==FG_OK)
        status=fg_owner_gdn_decode(e,layer,mixed,&block,err);
    double t_attn=ts_ms();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_write",err);
    if(status==FG_OK){status=fg_owner_gr_write(e,residual,block,injection,&after_attention,err);}double t_grw1=ts_ms();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_read",err);
    if(status==FG_OK){status=fg_owner_gr_read(e,layer,true,after_attention,&mixed,&residual,&injection,err);}double t_gr2=ts_ms();
    /* router matmul — stays in the same fused batch */
    fg_vk_tensor *router_w=status==FG_OK?weight(e,layer,"ffn_gate_inp.weight",err):NULL;
    if(status==FG_OK&&!router_w)status=FG_ERR_MISMATCH;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router",err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router_w,mixed,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,1u,err);
    status=finish_batch(vk,status,err); /* SYNC 1: read router logits for CPU top-K */
    double t_router=ts_ms();
    /* CPU top-K routing */
    uint16_t expert_ids[FG_TOP_K];float gates[FG_TOP_K];const uint8_t *activation=NULL;
    if(status==FG_OK){float router_local[FG_EXPERT_COUNT];memcpy(router_local,fg_vk_tensor_map(e->router_logits),sizeof(router_local));uint32_t ids[FG_TOP_K];status=fg_q38_router_topk(router_local,FG_EXPERT_COUNT,FG_TOP_K,ids,gates,err);for(uint32_t s=0;status==FG_OK&&s<FG_TOP_K;s++)expert_ids[s]=(uint16_t)ids[s];}
    /* ---- FUSED BATCH 2: quantize + shared expert ---- */
    fg_vk_tensor *shared_gate_w=NULL,*gate_w=NULL,*up_w=NULL,*down_w=NULL;
    if(status==FG_OK){shared_gate_w=weight(e,layer,"ffn_gate_inp_shexp.weight",err);gate_w=weight(e,layer,"ffn_gate_shexp.weight",err);up_w=weight(e,layer,"ffn_up_shexp.weight",err);down_w=weight(e,layer,"ffn_down_shexp.weight",err);if(!shared_gate_w||!gate_w||!up_w||!down_w)status=FG_ERR_MISMATCH;}
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router_quantization",err);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,mixed,FG_HIDDEN_SIZE,1u,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"shared_expert",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->shared_scalar,shared_gate_w,mixed,FG_HIDDEN_SIZE,1u,1u,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,e->shared_up,640u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_output,down_w,e->shared_mid,640u,FG_HIDDEN_SIZE,1u,1.0f,err);
    status=finish_batch(vk,status,err); /* SYNC 2: read activation + shared output */
    if(status==FG_OK)activation=fg_vk_tensor_map(e->activation_q8k);
    double t_mprep=ts_ms();
    /* expert dispatch + reduce + final gr_write (unchanged) */
    memset(e->decode_results,0,sizeof(e->decode_results));uint32_t result_count=0;if(status==FG_OK)status=dispatch(dispatch_context,layer,token,expert_ids,gates,activation,e->decode_results,&result_count,err);double t_exp=ts_ms();if(status==FG_OK)status=fg_owner_moe_reduce(e,layer,token,expert_ids,gates,e->decode_results,result_count,&block,err);double t_red=ts_ms();if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);if(status==FG_OK)status=fg_owner_gr_write(e,residual,block,injection,output,err);double t_end=ts_ms();
    float attn_block_l2=diag&&status==FG_OK?tensor_l2(block,FG_HIDDEN_SIZE):0.0f;
    if(diag&&status==FG_OK){fprintf(stderr,"layer[%u] t=%u in=%.4f attn_blk=%.4f attn=%.4f moe=%.4f out=%.4f exp=%u,%u,%u",layer,token,tensor_l2(layer_input,FG_HYPER_WIDTH),attn_block_l2,tensor_l2(after_attention,FG_HYPER_WIDTH),tensor_l2(block,FG_HIDDEN_SIZE),tensor_l2(*output,FG_HYPER_WIDTH),expert_ids[0],expert_ids[1],expert_ids[2]);if(layer==0u&&(layer&3u)!=3u)fprintf(stderr," gdn_state=%.6f",tensor_l2(OWNER_GDN(e,layer)->recurrent_state,48u*128u*128u));fprintf(stderr,"\n");}
    if(token>=26u&&token<32u){fprintf(stderr,"TIMING layer[%u] t=%u total=%.1f ple=%.1f gr_read=%.1f attn=%.1f gr_write=%.1f gr_read2=%.1f router=%.1f moe_prep=%.1f expert=%.1f moe_red=%.1f gr_write2=%.1f\n",layer,token,t_end-t0,t_ple-t0,t_gr1-t_ple,t_attn-t_gr1,t_grw1-t_attn,t_gr2-t_grw1,t_router-t_gr2,t_mprep-t_router,t_exp-t_mprep,t_red-t_exp,t_end-t_red);}
    return status;
}

static fg_status decode_layer_begin_impl(fg_owner_executor *e,uint32_t slot,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,void *dispatch_context,fg_owner_qsa_decode_dispatch_fn qsa_dispatch,void *qsa_context,fg_error *err){
    if(!e||slot>=FG_OWNER_SLOT_COUNT||!position||!hyper_input||!fire||!collect||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"async decode layer precondition");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embedding!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 PLE embedding presence mismatch");return FG_ERR_MISMATCH;}fg_owner_decode_slot *frame=&e->decode_slots[slot];fg_owner_pending_write *pending=&frame->pending_write;if((layer==0u&&pending->active)||(layer>0u&&(!pending->active||pending->layer+1u!=layer||pending->token!=token||pending->output!=hyper_input))){fg_error_set(err,FG_ERR_MISMATCH,"deferred residual write does not match successor layer");return FG_ERR_MISMATCH;}double t0=ts_ms();fg_vk_context *vk=fg_model_vk(e->model);bool ep_trace=fg_vk_profile_active(vk);uint64_t trace_start=ep_trace?wall_ns():0;fg_vk_counters counters_before={0};if(ep_trace)fg_vk_get_counters(vk,&counters_before);fg_status status=FG_OK;const fg_vk_tensor *layer_input=hyper_input;frame->active=true;frame->ep_trace=ep_trace;frame->fire_called=false;frame->residual=NULL;frame->layer=layer;frame->token=token;frame->collect=collect;frame->dispatch_context=dispatch_context;frame->t_begin=t0;frame->trace_start=trace_start;frame->counters_before=counters_before;if(layer==1u){fg_vk_tensor *ple_input=NULL;if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);if(status==FG_OK)status=fg_vk_begin(vk,err);if(status==FG_OK)status=flush_gr_write(e,pending,err);if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"ple",err);if(status==FG_OK)status=ple_decode_into(e,hyper_input,ngram_embedding,frame->ping[0],frame->ping[1],&ple_input,err);status=finish_batch(vk,status,err);if(status!=FG_OK){frame->active=false;return status;}layer_input=ple_input;}
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;
    /* Quantization joins Batch 1 so routed work can fire before shared expert compute. */
    if(status==FG_OK&&layer>1u&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&layer>1u)status=flush_gr_write(e,pending,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_read",err);
    if(status==FG_OK){status=gr_read_batch_into(e,layer,false,layer_input,1u,frame->injection,&mixed,&residual,&injection,err);}
    bool dbg_flush=layer==0u&&numerics_trace_enabled();
    if(status==FG_OK&&dbg_flush){
        status=finish_batch(vk,status,err);
        if(status==FG_OK)status=fg_vk_begin(vk,err);
        uint32_t rank=fg_model_rank(e->model);
        numerics_trace_values_local("A_NORM",rank,layer,fg_vk_tensor_map(e->hyper_norm),10240u);
        numerics_trace_values_local("A_LOW",rank,layer,fg_vk_tensor_map(e->low),320u);
        numerics_trace_values_local("A_UP",rank,layer,fg_vk_tensor_map(e->up_logits),10240u);
        numerics_trace_values_local("A_PART",rank,layer,fg_vk_tensor_map(e->inject_partials),e->hc_inject_pieces*4u);
        numerics_trace_values_local("A_MIX",rank,layer,fg_vk_tensor_map(mixed),FG_HIDDEN_SIZE);
        numerics_trace_values_local("A_INJ",rank,layer,fg_vk_tensor_map(injection),FG_GROUP_SIZE);
    }
    bool remote_qsa=(layer&3u)==3u&&qsa_dispatch;
    if(status==FG_OK&&remote_qsa){status=finish_batch(vk,status,err);if(status==FG_OK)status=qsa_dispatch(qsa_context,layer,token,position,mixed,&block,err);if(status==FG_OK)status=fg_vk_begin(vk,err);}
    else     if(status==FG_OK&&(layer&3u)==3u){
        status=fg_owner_qsa_decode(e,layer,token,position,mixed,&block,err);
    }else if(status==FG_OK)status=fg_owner_gdn_decode(e,layer,mixed,&block,err);
    if(status==FG_OK&&dbg_flush){
        status=finish_batch(vk,status,err);
        if(status==FG_OK)status=fg_vk_begin(vk,err);
        uint32_t rank=fg_model_rank(e->model);
        numerics_trace_values_local("B_QKV",rank,layer,fg_vk_tensor_map(e->gdn_qkv),10240u);
        numerics_trace_values_local("B_CONV",rank,layer,fg_vk_tensor_map(e->gdn_conv_output),10240u);
        numerics_trace_values_local("B_CORE",rank,layer,fg_vk_tensor_map(e->gdn_core),6144u);
        numerics_trace_values_local("B_OUT",rank,layer,fg_vk_tensor_map(e->gdn_output),FG_HIDDEN_SIZE);
    }
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_write",err);
    if(status==FG_OK){status=gr_write_batch_into(e,residual,block,injection,1u,frame->ping[0],frame->ping[1],&after_attention,err);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_read",err);
    if(status==FG_OK){status=gr_read_batch_into(e,layer,true,after_attention,1u,frame->injection,&mixed,&residual,&injection,err);}
    fg_vk_tensor *router_w=status==FG_OK?weight(e,layer,"ffn_gate_inp.weight",err):NULL;
    if(status==FG_OK&&!router_w)status=FG_ERR_MISMATCH;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router",err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router_w,mixed,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,1u,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router_quantization",err);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,mixed,FG_HIDDEN_SIZE,1u,err);
    status=finish_batch(vk,status,err); /* SYNC 1: router + activation */
    double t_sync1=ts_ms();uint64_t trace_sync1=ep_trace?wall_ns():0;
    if(status==FG_OK&&layer<=1u&&numerics_trace_enabled()){
        numerics_trace_values_local("SYNC1_NORM",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->hyper_norm),FG_HIDDEN_SIZE*4u);
        numerics_trace_values_local("SYNC1_LOW",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->low),320u);
        numerics_trace_values_local("SYNC1_UP",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->up_logits),FG_HIDDEN_SIZE*4u);
        numerics_trace_values_local("SYNC1_PART",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->inject_partials),e->hc_inject_pieces*4u);
        numerics_trace_values_local("SYNC1_QKV",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->gdn_qkv),10240u);
        numerics_trace_values_local("SYNC1_CONV",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->gdn_conv_output),10240u);
        numerics_trace_values_local("SYNC1_CORE",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->gdn_core),6144u);
        numerics_trace_values_local("SYNC1_GDNOUT",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(e->gdn_output),FG_HIDDEN_SIZE);
        numerics_trace_values_local("SYNC1_ATTN",fg_model_rank(e->model),layer,
            fg_vk_tensor_map((fg_vk_tensor *)after_attention),FG_HYPER_WIDTH);
        numerics_trace_values_local("SYNC1_MIX",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(mixed),FG_HIDDEN_SIZE);
        numerics_trace_values_local("SYNC1_INJ",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(frame->injection),FG_GROUP_SIZE);
    }
    float router_local[FG_EXPERT_COUNT];
    if(status==FG_OK){fg_status read_status=fg_vk_tensor_read(e->router_logits,0,router_local,sizeof(router_local),err);if(read_status!=FG_OK)status=read_status;}
    if(status==FG_OK){uint32_t ids[FG_TOP_K];status=fg_q38_router_topk(router_local,FG_EXPERT_COUNT,FG_TOP_K,ids,frame->gates,err);for(uint32_t s=0;status==FG_OK&&s<FG_TOP_K;s++)frame->expert_ids[s]=(uint16_t)ids[s];}
    double t_topk=ts_ms();
    if(status==FG_OK&&layer<=1u&&numerics_trace_enabled())
        numerics_trace_values_local("SYNC1_GATES",fg_model_rank(e->model),layer,
            frame->gates,FG_TOP_K);
    const uint8_t *activation=status==FG_OK?fg_vk_tensor_map(e->activation_q8k):NULL;
    fg_vk_tensor *shared_gate_w=NULL,*gate_w=NULL,*up_w=NULL,*down_w=NULL;
    if(status==FG_OK){shared_gate_w=weight(e,layer,"ffn_gate_inp_shexp.weight",err);gate_w=weight(e,layer,"ffn_gate_shexp.weight",err);up_w=weight(e,layer,"ffn_up_shexp.weight",err);down_w=weight(e,layer,"ffn_down_shexp.weight",err);if(!shared_gate_w||!gate_w||!up_w||!down_w)status=FG_ERR_MISMATCH;}
    if(status==FG_OK){frame->fire_called=true;status=fire(dispatch_context,layer,token,frame->expert_ids,frame->gates,activation,err);}
    double t_fire=ts_ms();uint64_t trace_fire=ep_trace?wall_ns():0;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"shared_expert",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,frame->shared_scalar,shared_gate_w,mixed,FG_HIDDEN_SIZE,1u,1u,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,e->shared_up,640u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,frame->shared_output,down_w,e->shared_mid,640u,FG_HIDDEN_SIZE,1u,1.0f,err);
    status=finish_batch(vk,status,err);
    frame->residual=residual;frame->t_sync1=t_sync1;frame->trace_sync1=trace_sync1;frame->t_topk=t_topk;frame->t_fire=t_fire;frame->trace_fire=trace_fire;frame->t_sync2=ts_ms();frame->trace_sync2=ep_trace?wall_ns():0;
    if(status!=FG_OK){
        if(frame->fire_called){fg_expert_result drain[FG_GROUP_SIZE];uint32_t drain_count=0;fg_error ignored={0};fg_status drained=collect(dispatch_context,layer,token,drain,&drain_count,&ignored);(void)drained;}
        frame->active=false;
        return status;
    }
    /* The shared-expert batch is drained by now, so the predecessor output that
     * this begin flushed into hyper_input is materialized for a valid digest. */
    numerics_trace_tensor(layer==0u?"BIN":"DOUT_PREV",fg_model_rank(e->model),
        layer==0u?0u:layer-1u,token,1u,hyper_input,err);
    if(layer==1u&&numerics_trace_enabled()){
        numerics_trace_values_local("PLE_OUT",fg_model_rank(e->model),layer,
            fg_vk_tensor_map((fg_vk_tensor *)layer_input),FG_HYPER_WIDTH);
    }
    if(layer<=1u&&numerics_trace_enabled()){
        numerics_trace_values_local("SYNC2_MIX",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(mixed),FG_HIDDEN_SIZE);
        numerics_trace_values_local("SYNC2_SHARED",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(frame->shared_output),FG_HIDDEN_SIZE);
        numerics_trace_values_local("SYNC2_SCALAR",fg_model_rank(e->model),layer,
            fg_vk_tensor_map(frame->shared_scalar),1u);
    }
    return FG_OK;
}

static fg_status decode_layer_finish_impl(fg_owner_executor *e,uint32_t slot,fg_vk_tensor **output,fg_error *err){
    if(!e||slot>=FG_OWNER_SLOT_COUNT||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid decode layer finish");return FG_ERR_ARGUMENT;}
    fg_owner_decode_slot *frame=&e->decode_slots[slot];
    if(!frame->active){fg_error_set(err,FG_ERR_MISMATCH,"decode layer finish does not match an active begin");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(e->model);fg_status status=FG_OK;
    fg_expert_result results[FG_GROUP_SIZE];uint32_t result_count=0;
    if(frame->fire_called){
        fg_error collect_error={0};
        fg_status collect_status=frame->collect(frame->dispatch_context,frame->layer,frame->token,results,&result_count,status==FG_OK?err:&collect_error);
        if(status==FG_OK)status=collect_status;
    }
    double t_collect=ts_ms();uint64_t trace_collect=frame->ep_trace?wall_ns():0;
    fg_vk_tensor *block=NULL;
    if(status==FG_OK)status=moe_reduce_into(e,frame->layer,frame->token,frame->expert_ids,frame->gates,results,result_count,frame->shared_output,frame->shared_scalar,frame->reduced,&block,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);
    if(status==FG_OK)status=frame->layer+1u<FG_LAYER_COUNT?defer_gr_write(e,&frame->pending_write,frame->layer,frame->token,frame->residual,block,frame->injection,frame->ping[0],frame->ping[1],output,err):gr_write_batch_into(e,frame->residual,block,frame->injection,1u,frame->ping[0],frame->ping[1],output,err);
    if(status==FG_OK&&frame->layer+1u>=FG_LAYER_COUNT)
        numerics_trace_tensor("DOUT",fg_model_rank(e->model),frame->layer,
            frame->token,1u,*output,err);
    double t_end=ts_ms();uint64_t trace_end=frame->ep_trace?wall_ns():0;
    frame->active=false;
    if(frame->ep_trace){fg_vk_counters counters_after={0};fg_vk_get_counters(vk,&counters_after);fprintf(stderr,"EP_LAYER_TRACE token=%u layer=%u status=%d total_ms=%.3f sync1_ms=%.3f fire_ms=%.3f shared_ms=%.3f collect_ms=%.3f finish_ms=%.3f submissions=%llu dispatches=%llu start_ns=%llu router_ready_ns=%llu fire_end_ns=%llu shared_end_ns=%llu collect_end_ns=%llu finish_end_ns=%llu\n",frame->token,frame->layer,(int)status,t_end-frame->t_begin,frame->t_sync1-frame->t_begin,frame->t_fire-frame->t_sync1,frame->t_sync2-frame->t_fire,t_collect-frame->t_sync2,t_end-t_collect,(unsigned long long)(counters_after.submissions-frame->counters_before.submissions),(unsigned long long)(counters_after.dispatches-frame->counters_before.dispatches),(unsigned long long)frame->trace_start,(unsigned long long)frame->trace_sync1,(unsigned long long)frame->trace_fire,(unsigned long long)frame->trace_sync2,(unsigned long long)trace_collect,(unsigned long long)trace_end);}
    if(frame_trace_enabled()||(frame->token>=26u&&frame->token<32u)){fprintf(stderr,"OVERLAP_TIMING layer[%u] t=%u total=%.1f sync1=%.1f fire=%.1f shared=%.1f collect=%.1f reduce+grw=%.1f\n",frame->layer,frame->token,t_end-frame->t_begin,frame->t_sync1-frame->t_begin,frame->t_fire-frame->t_sync1,frame->t_sync2-frame->t_fire,t_collect-frame->t_sync2,t_end-t_collect);}
    if(decode_ms_enabled())fprintf(stderr,"DECODE_MS rank=%u token=%u layer=%u batch1_ms=%.3f topk_ms=%.3f fire_ms=%.3f shared_ms=%.3f collect_ms=%.3f reduce_ms=%.3f total_ms=%.3f\n",fg_model_rank(e->model),frame->token,frame->layer,frame->t_sync1-frame->t_begin,frame->t_topk-frame->t_sync1,frame->t_fire-frame->t_topk,frame->t_sync2-frame->t_fire,t_collect-frame->t_sync2,t_end-t_collect,t_end-frame->t_begin);
    return status;
}

fg_status fg_owner_decode_layer_begin(fg_owner_executor *e,uint32_t slot,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,void *dispatch_context,fg_owner_qsa_decode_dispatch_fn qsa_dispatch,void *qsa_context,fg_error *err){
    return decode_layer_begin_impl(e,slot,layer,token,position,hyper_input,ngram_embedding,fire,collect,dispatch_context,qsa_dispatch,qsa_context,err);
}

fg_status fg_owner_decode_layer_finish(fg_owner_executor *e,uint32_t slot,fg_vk_tensor **output,fg_error *err){
    return decode_layer_finish_impl(e,slot,output,err);
}

fg_status fg_owner_decode_layer_async(fg_owner_executor *e,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,void *dispatch_context,fg_owner_qsa_decode_dispatch_fn qsa_dispatch,void *qsa_context,fg_vk_tensor **output,fg_error *err){
    if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"async decode layer output is null");return FG_ERR_ARGUMENT;}
    fg_status status=decode_layer_begin_impl(e,0u,layer,token,position,hyper_input,ngram_embedding,fire,collect,dispatch_context,qsa_dispatch,qsa_context,err);
    if(status!=FG_OK)return status;
    return decode_layer_finish_impl(e,0u,output,err);
}

/* Execute a contiguous layer block of one decode token on its owner.  A
 * mid-model block has no predecessor on this rank, so the deferred residual
 * chain is seeded with a skip marker rather than flushed, and the final
 * layer's deferred write is materialized before returning so the caller can
 * read the block result.  Ring decode runs every block through this path. */
fg_status fg_owner_decode_block(fg_owner_executor *e,uint32_t first_layer,
    uint32_t last_layer,uint32_t token,const uint32_t position[3],
    const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,
    fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,
    void *dispatch_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!position||!hyper_input||!fire||!collect||!output||
       first_layer>last_layer||last_layer>=FG_LAYER_COUNT||
       (first_layer<=1u)!=(ngram_embedding!=NULL)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner decode block arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_owner_pending_write *pending=&e->decode_slots[0].pending_write;
    memset(pending,0,sizeof(*pending));
    if(first_layer>0u){
        pending->active=true;pending->skip=true;pending->layer=first_layer-1u;
        pending->token=token;pending->hyper=hyper_input;pending->block=hyper_input;
        pending->injection=hyper_input;pending->output=(fg_vk_tensor *)hyper_input;
    }
    fg_vk_tensor *current=(fg_vk_tensor *)hyper_input;
    for(uint32_t layer=first_layer;layer<=last_layer;layer++){
        fg_status status=decode_layer_begin_impl(e,0u,layer,token,position,current,
            layer==1u?ngram_embedding:NULL,fire,collect,dispatch_context,NULL,NULL,err);
        if(status!=FG_OK){memset(pending,0,sizeof(*pending));return status;}
        status=decode_layer_finish_impl(e,0u,&current,err);
        if(status!=FG_OK){memset(pending,0,sizeof(*pending));return status;}
    }
    if(pending->active){
        fg_vk_tensor *materialized=pending->output;
        fg_status status=flush_gr_write(e,pending,err);
        if(status!=FG_OK)return status;
        current=materialized;
    }
    *output=current;
    return FG_OK;
}

static fg_status ensure_decode_batch(fg_vk_context *vk,fg_error *err){
    return fg_vk_batch_active(vk)?FG_OK:fg_vk_begin(vk,err);
}

/* The residual stream lives in the executor ping-pong; the tensor a recorded
 * run leaves current is captured when the run is recorded and reused verbatim
 * on replay, so this rule only feeds the STATIC_PING diagnostic. */
static fg_vk_tensor *chained_ping_next(fg_owner_executor *e,const fg_vk_tensor *residual){
    return residual!=e->hyper_output?e->hyper_output:e->hyper_output_b;
}

static uint64_t state_hash_f32(const float *values,uint64_t count){
    uint64_t hash=UINT64_C(1469598103934665603);
    for(uint64_t i=0;i<count;i++){uint32_t bits;memcpy(&bits,&values[i],4u);hash^=bits;hash*=UINT64_C(1099511628211);}
    return hash;
}

static void gdn_state_trace(fg_owner_executor *e,uint32_t layer,uint32_t token,
    const char *phase,const fg_vk_tensor *input){
    if(!numerics_trace_enabled())return;
    if(!e||layer>=FG_LAYER_COUNT||(layer&3u)==3u)return;
    const fg_vk_tensor *conv=OWNER_GDN(e,layer)->conv_state;
    const fg_vk_tensor *recur=OWNER_GDN(e,layer)->recurrent_state;
    if(!conv||!recur)return;
    const float *c=fg_vk_tensor_map((fg_vk_tensor *)conv);
    const float *r=fg_vk_tensor_map((fg_vk_tensor *)recur);
    const float *in=input?fg_vk_tensor_map((fg_vk_tensor *)input):NULL;
    fprintf(stderr,"FG_STATE rank=%u token=%u layer=%u phase=%s input=%016llx conv=%016llx recur=%016llx c0=%g r0=%g i0=%g\n",
        fg_model_rank(e->model),token,layer,phase,
        (unsigned long long)(in?state_hash_f32(in,FG_HIDDEN_SIZE):0u),
        (unsigned long long)state_hash_f32(c,10240u),
        (unsigned long long)state_hash_f32(r,48u*128u*128u),c[0],r[0],in?in[0]:0.0f);
}

/* Static replay stays enabled unless explicitly disabled (FG_DECODE_STATIC=0)
 * or the Vulkan timeline profiler is active.  A recorded run must execute
 * exactly the work of a fresh recording; the run output pointer captured at
 * record time is reused on replay. */
static bool chained_static_allowed(fg_vk_context *vk){
    const char *value=getenv("FG_DECODE_STATIC");
    if(value&&*value&&strcmp(value,"0")==0)return false;
    if(vk&&fg_vk_profile_active(vk))return false;
    return true;
}

/* Diagnostic: re-record every N tokens (FG_STATIC_RERECORD=N) to bisect the
 * staleness horizon; 0 or unset keeps the recorded run. */
static uint32_t static_rerecord_period(void){
    static int period=-1;
    if(period<0){
        const char *value=getenv("FG_STATIC_RERECORD");
        period=(value&&*value)?atoi(value):0;
        if(period<=0)period=0;
    }
    return (uint32_t)period;
}
static bool static_check_enabled(void){
    const char *value=getenv("FG_STATIC_CHECK");
    return value&&*value&&strcmp(value,"0")!=0;
}
static fg_status static_run_check(fg_owner_executor *e,fg_vk_context *vk,uint32_t slot,
    uint32_t token,fg_error *err){
    fg_status status=fg_vk_static_wait(vk,slot,err);
    if(status!=FG_OK)return status;
    const fg_vk_tensor *output=e->static_run_output[slot];
    const float *values=output?fg_vk_tensor_map((fg_vk_tensor *)output):NULL;
    int32_t first_bad=-1;
    if(values)for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
        if(!isfinite(values[i])){first_bad=(int32_t)i;break;}
    fprintf(stderr,"FG_STATIC_RUN rank=%u token=%u slot=%u finite=%d first_bad=%d f0=%g\n",
        fg_model_rank(e->model),token,slot,first_bad<0,first_bad,values?values[0]:0.0f);
    if(first_bad>=0){
        fg_error_set(err,FG_ERR_FORMAT,
            "rank %u static run slot %u token %u produced non-finite hidden at element %d value=%g",
            fg_model_rank(e->model),slot,token,first_bad,values[first_bad]);
        return FG_ERR_FORMAT;
    }
    return FG_OK;
}

/* ---------------------------------------------------------------------------
 * Depth-B batch-2 block: both slots' tokens through one layer pass.
 *
 * Dense passes read each weight once for the token pair: the b2 MMV where the
 * B=1 path takes the r8 decode kernel (same per-token arithmetic, measured
 * bit-identical) and the token-grid cooked kernel for the remaining
 * non-r8-eligible projections.  Row-wise passes (RMS, HC injection, GR mix,
 * router, shared expert, shared fold) take both rows in one dispatch.  The
 * stateful work stays per token: GDN conv/recurrent and PLE conv under the
 * token's owner session, and the whole QSA session (projections, selection,
 * records, attention) is literally the B=1 code path.  Routed experts run as
 * one union work: GPU top-10 for the pair, one token-tagged schedule, one
 * fused gate/up and one down/reduce dispatch.
 *
 * Bit-exactness: every batched pass keeps each token's arithmetic sequence
 * (operand order, fma chain, subgroup reduction) identical to its B=1
 * dispatch; no partial sum crosses tokens.  The block never records into the
 * static replay (session 0's recorded run stays bound to the single-token
 * path), so a batch step is always a fresh dynamic recording.
 * ------------------------------------------------------------------------- */

static fg_status ensure_batch_storage(fg_owner_executor *e,fg_error *err){
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=FG_OK;
    if(!e->batch_ping[0])status=fg_vk_tensor_create(vk,
        (uint64_t)2u*FG_HYPER_WIDTH*4u,&e->batch_ping[0],err);
    if(status==FG_OK&&!e->batch_ping[1])status=fg_vk_tensor_create(vk,
        (uint64_t)2u*FG_HYPER_WIDTH*4u,&e->batch_ping[1],err);
    if(status==FG_OK&&!e->batch_hc_down_partials)status=fg_vk_tensor_create(vk,
        (uint64_t)2u*FG_HC_DOWN_SPLITS*320u*4u,&e->batch_hc_down_partials,err);
    return status;
}

static fg_status row_slice(fg_vk_tensor *base,uint32_t row,uint32_t width,
                           fg_vk_tensor **out,fg_error *err){
    if(!base||!out){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner row slice");
        return FG_ERR_ARGUMENT;
    }
    return fg_vk_tensor_view(base,(uint64_t)row*width*4u,(uint64_t)width*4u,out,err);
}

/* GR read for the token pair: identical layout to gr_read_batch_into with
 * token_count=2, but the HC down projection goes through the split+SiLU
 * kernel (the exact arithmetic the single-token decode uses) with two token
 * rows, and the up projection through the b2 MMV (the single-token r8
 * arithmetic for each row). */
static fg_status gr_read_batch_into_b2(fg_owner_executor *e,uint32_t layer,bool ffn,
    const fg_vk_tensor *hyper_input,fg_vk_tensor **mixed,const fg_vk_tensor **residual,
    fg_vk_tensor **injection,fg_error *err){
    if(!e||!hyper_input||!mixed||!residual||!injection||!owns_layer(e,layer)){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "batch-2 gated residual read is not on the layer owner");
        return FG_ERR_MISMATCH;
    }
    const char *prefix=ffn?"hc_ffn":"hc_attn";
    char suffix[48];
    fg_vk_tensor *norm_weight,*down_weight,*up_weight,*inject_weight;
    snprintf(suffix,sizeof(suffix),"%s_norm.weight",prefix);norm_weight=weight(e,layer,suffix,err);
    snprintf(suffix,sizeof(suffix),"%s_down.weight",prefix);down_weight=weight(e,layer,suffix,err);
    snprintf(suffix,sizeof(suffix),"%s_up.weight",prefix);up_weight=weight(e,layer,suffix,err);
    snprintf(suffix,sizeof(suffix),"%s_inject.weight",prefix);inject_weight=weight(e,layer,suffix,err);
    if(!norm_weight||!down_weight||!up_weight||!inject_weight)return FG_ERR_MISMATCH;
    if(fg_vk_tensor_get_format(down_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
       fg_vk_tensor_get_format(up_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "batch-2 GR read requires cooked HC projections on layer %u",layer);
        return FG_ERR_UNAVAILABLE;
    }
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->hyper_norm,hyper_input,norm_weight,
        FG_HIDDEN_SIZE,4u,2u,1e-6f,err);
    if(status==FG_OK)status=fg_vk_hc_inject_partial(vk,e->inject_partials,e->hyper_norm,
        inject_weight,FG_HIDDEN_SIZE,4u,2u,e->hc_inject_pieces,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_q8_0_cooked_split_silu(vk,e->low,e->low_active,
        e->batch_hc_down_partials,down_weight,e->hyper_norm,10240u,320u,2u,
        FG_HC_DOWN_SPLITS,1.0f,0.25f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->up_logits,up_weight,
        e->low_active,320u,10240u,1.0f,err);
    if(status==FG_OK)status=fg_vk_gr_mix_partial(vk,e->mixed,e->injection,e->hyper_norm,
        e->up_logits,e->inject_partials,FG_HIDDEN_SIZE,4u,2u,e->hc_inject_pieces,err);
    status=finish_batch(vk,status,err);
    if(status==FG_OK){*mixed=e->mixed;*residual=hyper_input;*injection=e->injection;}
    return status;
}

/* PLE for the token pair: b2 key/value projections, two-token norms and gate,
 * then the per-token convolution under the token's owner session (the PLE
 * state is session-scoped). */
static fg_status ple_decode_into_b2(fg_owner_executor *e,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,fg_vk_tensor *destination,const uint32_t slots[2],
    fg_vk_tensor **output,fg_error *err){
    if(!e||!hyper||!embedding||!destination||!slots||!output||
       !owns_layer(e,1u)||!OWNER_PLE(e)){
        fg_error_set(err,FG_ERR_MISMATCH,"batch-2 PLE decode is not on the layer-1 owner");
        return FG_ERR_MISMATCH;
    }
    fg_vk_tensor *key_weight=weight(e,1u,"ple_key.weight",err),*value_weight=weight(e,1u,"ple_value.weight",err),*key_norm=weight(e,1u,"ple_norm_key.weight",err),*query_norm=weight(e,1u,"ple_norm_query.weight",err),*conv_norm=weight(e,1u,"ple_norm_conv.weight",err),*conv_weight=weight(e,1u,"ple_conv1d.weight",err);
    if(!key_weight||!value_weight||!key_norm||!query_norm||!conv_norm||!conv_weight)return FG_ERR_MISMATCH;
    if(fg_vk_tensor_get_format(key_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
       fg_vk_tensor_get_format(value_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"batch-2 PLE requires cooked key/value");
        return FG_ERR_UNAVAILABLE;
    }
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->ple_key,key_weight,embedding,
        2560u,10240u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->ple_value,value_weight,embedding,
        2560u,2560u,1.0f,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_key_norm,e->ple_key,key_norm,
        2560u,4u,2u,1e-6f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_query_norm,hyper,query_norm,
        2560u,4u,2u,1e-6f,err);
    if(status==FG_OK)status=fg_vk_ple_gate_prefill(vk,e->ple_gated,e->ple_key_norm,
        e->ple_query_norm,e->ple_value,2u,err);
    if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_gated_norm,e->ple_gated,
        conv_norm,2560u,4u,2u,1e-6f,err);
    for(uint32_t t=0;status==FG_OK&&t<2u;t++){
        fg_vk_tensor *gated_row=NULL,*norm_row=NULL,*dst_row=NULL,*hyper_row=NULL;
        status=row_slice(e->ple_gated,t,10240u,&gated_row,err);
        if(status==FG_OK)status=row_slice(e->ple_gated_norm,t,10240u,&norm_row,err);
        if(status==FG_OK)status=row_slice(destination,t,10240u,&dst_row,err);
        if(status==FG_OK)status=row_slice((fg_vk_tensor *)hyper,t,10240u,&hyper_row,err);
        if(status==FG_OK)status=fg_owner_set_active_session(e,slots[t],err);
        if(status==FG_OK)status=fg_vk_ple_conv_decode_add(vk,dst_row,OWNER_PLE(e),
            gated_row,norm_row,conv_weight,hyper_row,err);
        fg_vk_tensor_destroy(hyper_row);fg_vk_tensor_destroy(dst_row);
        fg_vk_tensor_destroy(norm_row);fg_vk_tensor_destroy(gated_row);
    }
    status=finish_batch(vk,status,err);
    if(status==FG_OK)*output=destination;
    return status;
}

/* GDN for the token pair: b2 qkv, token-grid z, f32 alpha/beta, then the
 * per-token convolution and algebraic recurrence under the token's owner
 * session, and the b2 output projection into the two-row block tensor. */
static fg_status gdn_decode_pair_into(fg_owner_executor *e,uint32_t layer,
    const fg_vk_tensor *hidden,const uint32_t slots[2],fg_vk_tensor **output,
    fg_error *err){
    if(!e||!hidden||!slots||!output||!owns_layer(e,layer)||(layer&3u)==3u||
       !OWNER_GDN(e,layer)->conv_state||!OWNER_GDN(e,layer)->recurrent_state){
        fg_error_set(err,FG_ERR_MISMATCH,"batch-2 GDN decode is not on an owned linear-attention layer");
        return FG_ERR_MISMATCH;
    }
    fg_vk_tensor *qkv_weight=weight(e,layer,"attn_qkv.weight",err),*z_weight=weight(e,layer,"attn_gate.weight",err),*alpha_weight=weight(e,layer,"ssm_alpha.weight",err),*beta_weight=weight(e,layer,"ssm_beta.weight",err),*conv_weight=weight(e,layer,"ssm_conv1d.weight",err),*a_decay=weight(e,layer,"ssm_a",err),*dt_bias=weight(e,layer,"ssm_dt.bias",err),*norm_weight=weight(e,layer,"ssm_norm.weight",err),*out_weight=weight(e,layer,"ssm_out.weight",err);
    if(!qkv_weight||!z_weight||!alpha_weight||!beta_weight||!conv_weight||!a_decay||!dt_bias||!norm_weight||!out_weight)return FG_ERR_MISMATCH;
    if(fg_vk_tensor_get_format(qkv_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
       fg_vk_tensor_get_format(z_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
       fg_vk_tensor_get_format(out_weight)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"batch-2 GDN requires cooked projections");
        return FG_ERR_UNAVAILABLE;
    }
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"gdn_projection",err):FG_OK;
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->gdn_qkv,qkv_weight,hidden,
        2560u,10240u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_cooked_pair(vk,e->gdn_z,z_weight,hidden,
        2560u,6144u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->gdn_alpha,alpha_weight,hidden,
        2560u,48u,2u,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->gdn_beta,beta_weight,hidden,
        2560u,48u,2u,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gdn_recurrent",err);
    for(uint32_t t=0;status==FG_OK&&t<2u;t++){
        fg_vk_tensor *qkv_row=NULL,*conv_row=NULL,*core_row=NULL,*z_row=NULL,*alpha_row=NULL,*beta_row=NULL;
        status=row_slice(e->gdn_qkv,t,10240u,&qkv_row,err);
        if(status==FG_OK)status=row_slice(e->gdn_conv_output,t,10240u,&conv_row,err);
        if(status==FG_OK)status=row_slice(e->gdn_core,t,6144u,&core_row,err);
        if(status==FG_OK)status=row_slice(e->gdn_z,t,6144u,&z_row,err);
        if(status==FG_OK)status=row_slice(e->gdn_alpha,t,48u,&alpha_row,err);
        if(status==FG_OK)status=row_slice(e->gdn_beta,t,48u,&beta_row,err);
        if(status==FG_OK)status=fg_owner_set_active_session(e,slots[t],err);
        if(status==FG_OK)status=fg_vk_gdn_conv_decode(vk,conv_row,
            OWNER_GDN(e,layer)->conv_state,qkv_row,conv_weight,10240u,err);
        if(status==FG_OK)status=fg_vk_gdn_recurrent_algebraic(vk,core_row,
            OWNER_GDN(e,layer)->recurrent_state,conv_row,z_row,alpha_row,beta_row,
            a_decay,dt_bias,norm_weight,48u,16u,128u,1e-6f,err);
        fg_vk_tensor_destroy(beta_row);fg_vk_tensor_destroy(alpha_row);
        fg_vk_tensor_destroy(z_row);fg_vk_tensor_destroy(core_row);
        fg_vk_tensor_destroy(conv_row);fg_vk_tensor_destroy(qkv_row);
    }
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gdn_output",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->gdn_output,out_weight,
        e->gdn_core,6144u,2560u,1.0f,err);
    status=finish_batch(vk,status,err);
    if(status==FG_OK)*output=e->gdn_output;
    return status;
}

/* Per-token GR write used on QSA layers: the QSA session output is a
 * per-session tensor, so the residual write stays per token exactly as the
 * single-token path writes it (the row views alias the two-row ping). */
static fg_status gr_write_pair_into(fg_owner_executor *e,const fg_vk_tensor *residual2,
    fg_vk_tensor *const block[2],const fg_vk_tensor *injection2,fg_vk_tensor *destination,
    fg_error *err){
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=FG_OK;
    for(uint32_t t=0;status==FG_OK&&t<2u;t++){
        fg_vk_tensor *residual_row=NULL,*injection_row=NULL,*dst_row=NULL;
        /* The residual and the destination are full hyper rows (hidden x
         * groups); only the block is hidden-wide. */
        status=row_slice((fg_vk_tensor *)residual2,t,FG_HYPER_WIDTH,&residual_row,err);
        if(status==FG_OK)status=row_slice(destination,t,FG_HYPER_WIDTH,&dst_row,err);
        if(status==FG_OK)status=row_slice((fg_vk_tensor *)injection2,t,FG_GROUP_SIZE,&injection_row,err);
        if(status==FG_OK&&!block[t])status=FG_ERR_MISMATCH;
        if(status==FG_OK)status=fg_vk_gr_write(vk,dst_row,residual_row,block[t],
            injection_row,FG_HIDDEN_SIZE,4u,1u,err);
        fg_vk_tensor_destroy(injection_row);fg_vk_tensor_destroy(dst_row);
        fg_vk_tensor_destroy(residual_row);
    }
    return status;
}

/* Readiness check for the batch block: every weight it dispatches must be in
 * the cooked ring layout (otherwise the block would silently change a token's
 * arithmetic).  Returns FG_ERR_UNAVAILABLE so the caller can fall back to the
 * serial slot loop without touching state. */
static bool batch_cooked(fg_owner_executor *e,uint32_t layer,const char *name,
                         fg_error *err){
    fg_error probe={0};
    fg_vk_tensor *t=weight(e,layer,name,&probe);
    if(!t||fg_vk_tensor_get_format(t)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"batch-2 block layer %u %s is not cooked",
                     layer,name);
        return false;
    }
    return true;
}

bool fg_owner_decode_block_batch_ready(fg_owner_executor *e,uint32_t first,uint32_t last,
    fg_error *err){
    if(!e||first>last||last>=FG_LAYER_COUNT||e->session_count<2u||e->max_tokens<2u){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"batch-2 owner block is not constructible");
        return false;
    }
    for(uint32_t layer=first;layer<=last;layer++){
        if(!owns_layer(e,layer)){
            fg_error_set(err,FG_ERR_UNAVAILABLE,"batch-2 block layer %u is not owned",layer);
            return false;
        }
        const char *hc=(layer&3u)==3u?"hc_ffn":"hc_attn";
        char hc_down[48],hc_up[48];
        snprintf(hc_down,sizeof(hc_down),"%s_down.weight",hc);
        snprintf(hc_up,sizeof(hc_up),"%s_up.weight",hc);
        if(layer==1u&&(!batch_cooked(e,layer,"ple_key.weight",err)||
                       !batch_cooked(e,layer,"ple_value.weight",err)))return false;
        if((layer&3u)!=3u&&(!batch_cooked(e,layer,"attn_qkv.weight",err)||
                            !batch_cooked(e,layer,"attn_gate.weight",err)||
                            !batch_cooked(e,layer,"ssm_out.weight",err)))return false;
        if(!batch_cooked(e,layer,hc_down,err)||!batch_cooked(e,layer,hc_up,err)||
           !batch_cooked(e,layer,"ffn_gate_shexp.weight",err)||
           !batch_cooked(e,layer,"ffn_up_shexp.weight",err)||
           !batch_cooked(e,layer,"ffn_down_shexp.weight",err))return false;
    }
    return true;
}

fg_status fg_owner_decode_block_batch(fg_owner_executor *e,uint32_t first_layer,
    uint32_t last_layer,const uint32_t token_index[2],const uint32_t state_slot[2],
    const uint32_t positions[6],const fg_vk_tensor *hyper_input,
    const fg_vk_tensor *ngram_embedding,fg_owner_expert_inline_fn expert,
    void *expert_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!token_index||!state_slot||!positions||!hyper_input||!expert||!output||
       first_layer>last_layer||last_layer>=FG_LAYER_COUNT||
       (first_layer<=1u)!=(ngram_embedding!=NULL)||!owns_layer(e,first_layer)||
       e->session_count<2u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batch-2 owner decode block arguments");
        return FG_ERR_ARGUMENT;
    }
    for(uint32_t t=0;t<2u;t++)
        if(state_slot[t]>=e->session_count||!session_allocated(e,state_slot[t])){
            fg_error_set(err,FG_ERR_ARGUMENT,"batch-2 block session %u is not allocated",
                         state_slot[t]);
            return FG_ERR_ARGUMENT;
        }
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=ensure_batch_storage(e,err);
    const fg_vk_tensor *cur=hyper_input;
    uint32_t cur_ping=UINT32_MAX;
#define BATCH_NEXT() (cur_ping==UINT32_MAX?(cur_ping=0u,e->batch_ping[0]): \
    (cur_ping^=1u,e->batch_ping[cur_ping]))
    for(uint32_t layer=first_layer;status==FG_OK&&layer<=last_layer;layer++){
        bool qsa=(layer&3u)==3u;
        const fg_vk_tensor *layer_input=cur;
        if(layer==1u){
            if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"ple",err);
            if(status==FG_OK)status=ensure_decode_batch(vk,err);
            fg_vk_tensor *destination=BATCH_NEXT(),*ple_input=NULL;
            if(status==FG_OK)status=ple_decode_into_b2(e,layer_input,ngram_embedding,
                destination,state_slot,&ple_input,err);
            if(status==FG_OK){layer_input=ple_input;cur=ple_input;}
        }
        if(status==FG_OK&&fg_vk_profile_active(vk))
            status=fg_vk_profile_set_scope(vk,"gr_attn_read",err);
        if(status==FG_OK)status=ensure_decode_batch(vk,err);
        fg_vk_tensor *mixed=NULL,*injection=NULL;const fg_vk_tensor *residual=NULL;
        if(status==FG_OK)status=gr_read_batch_into_b2(e,layer,false,layer_input,&mixed,
            &residual,&injection,err);
        fg_vk_tensor *qsa_block[2]={NULL,NULL};
        if(status==FG_OK&&fg_vk_profile_active(vk))
            status=fg_vk_profile_set_scope(vk,qsa?"qsa":"gdn",err);
        if(status==FG_OK&&qsa){
            for(uint32_t t=0;status==FG_OK&&t<2u;t++){
                fg_vk_tensor *hidden_row=NULL;
                const uint32_t position[3]={positions[t*3u],positions[t*3u+1u],
                                            positions[t*3u+2u]};
                status=row_slice(e->mixed,t,FG_HIDDEN_SIZE,&hidden_row,err);
                if(status==FG_OK)status=fg_owner_set_active_session(e,state_slot[t],err);
                if(status==FG_OK)status=fg_owner_qsa_decode(e,layer,token_index[t],
                    position,hidden_row,&qsa_block[t],err);
                fg_vk_tensor_destroy(hidden_row);
            }
        }else if(status==FG_OK){
            fg_vk_tensor *block=NULL;
            status=gdn_decode_pair_into(e,layer,layer_input,state_slot,&block,err);
        }
        if(status==FG_OK&&fg_vk_profile_active(vk))
            status=fg_vk_profile_set_scope(vk,"gr_attn_write",err);
        if(status==FG_OK)status=ensure_decode_batch(vk,err);
        if(status==FG_OK){
            fg_vk_tensor *destination=BATCH_NEXT();
            if(qsa){
                status=gr_write_pair_into(e,cur,qsa_block,injection,destination,err);
                if(status==FG_OK)cur=destination;
            }else{
                /* gdn pair wrote the two-row block into gdn_output. */
                status=fg_vk_gr_write(vk,destination,cur,e->gdn_output,injection,
                    FG_HIDDEN_SIZE,4u,2u,err);
                if(status==FG_OK)cur=destination;
            }
        }
        if(status==FG_OK&&fg_vk_profile_active(vk))
            status=fg_vk_profile_set_scope(vk,"gr_ffn_read",err);
        fg_vk_tensor *mixed2=NULL,*injection2=NULL;const fg_vk_tensor *residual2=NULL;
        if(status==FG_OK)status=gr_read_batch_into_b2(e,layer,true,cur,&mixed2,&residual2,
            &injection2,err);
        fg_vk_tensor *router_w=status==FG_OK?weight(e,layer,"ffn_gate_inp.weight",err):NULL;
        if(status==FG_OK&&!router_w)status=FG_ERR_MISMATCH;
        if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router",err);
        if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router_w,mixed2,
            FG_HIDDEN_SIZE,FG_EXPERT_COUNT,2u,err);
        if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router_quantization",err);
        if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
        if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,mixed2,
            FG_HIDDEN_SIZE,2u,err);
        fg_vk_tensor *shared_gate_w=NULL,*gate_w=NULL,*up_w=NULL,*down_w=NULL;
        if(status==FG_OK){
            shared_gate_w=weight(e,layer,"ffn_gate_inp_shexp.weight",err);
            gate_w=weight(e,layer,"ffn_gate_shexp.weight",err);
            up_w=weight(e,layer,"ffn_up_shexp.weight",err);
            down_w=weight(e,layer,"ffn_down_shexp.weight",err);
            if(!shared_gate_w||!gate_w||!up_w||!down_w)status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"shared_expert",err);
        if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->shared_gate,gate_w,mixed2,
            FG_HIDDEN_SIZE,640u,1.0f,err);
        if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
        if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->shared_up,up_w,mixed2,
            FG_HIDDEN_SIZE,640u,1.0f,err);
        if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
        if(status==FG_OK)status=fg_vk_dense_f32(vk,e->shared_scalar,shared_gate_w,mixed2,
            FG_HIDDEN_SIZE,1u,2u,err);
        if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,e->shared_up,
            2u*640u,err);
        if(status==FG_OK)status=fg_vk_dense_q8_0_b2(vk,e->shared_output,down_w,
            e->shared_mid,640u,FG_HIDDEN_SIZE,1.0f,err);
        fg_vk_tensor *expert_output=NULL;
        if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"expert_decode",err);
        if(status==FG_OK)status=expert(expert_context,layer,e->activation_q8k,
            e->router_logits,&expert_output,err);
        if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"moe_reduce",err);
        if(status==FG_OK)status=fg_vk_moe_decode_shared_add(vk,e->reduced,expert_output,
            e->shared_output,e->shared_scalar,FG_HIDDEN_SIZE,2u,err);
        if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);
        if(status==FG_OK)status=ensure_decode_batch(vk,err);
        if(status==FG_OK){
            fg_vk_tensor *destination=BATCH_NEXT();
            status=fg_vk_gr_write(vk,destination,cur,e->reduced,injection2,
                FG_HIDDEN_SIZE,4u,2u,err);
            if(status==FG_OK)cur=destination;
        }
    }
#undef BATCH_NEXT
    if(status==FG_OK){
        if(fg_vk_batch_active(vk))status=finish_batch(vk,status,err);
    }else{
        fprintf(stderr,"BATCH_BLOCK_FAIL rank=%u first=%u last=%u status=%d msg=%s\n",
                fg_model_rank(e->model),first_layer,last_layer,(int)status,
                err?err->message:"");
        if(fg_vk_batch_active(vk)){
            fg_error ignored={0};
            fg_vk_abort(vk,&ignored);
        }
    }
    if(status==FG_OK)*output=(fg_vk_tensor *)cur;
    return status;
}

/* One text layer: PLE, GR read, GDN or QSA, GR write, GR read, router, shared
 * expert, GPU-routed expert pair, reduction and the final GR write.  Every
 * non-QSA layer is token-invariant, so a run of them is recorded once into a
 * static batch and replayed for later tokens. */
static fg_status owner_record_layer(fg_owner_executor *e,uint32_t layer,
    uint32_t token,const uint32_t position[3],const fg_vk_tensor *block_input,
    const fg_vk_tensor *ngram_embedding,fg_owner_expert_inline_fn expert,
    void *expert_context,fg_vk_tensor **current,fg_error *err){
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_status status=FG_OK;
    (void)block_input;
    const fg_vk_tensor *layer_input=*current;
    if(layer==1u){
        fg_vk_tensor *ple_input=NULL;
        if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"ple",err);
        if(status==FG_OK)status=ensure_decode_batch(vk,err);
        /* The PLE query is the current residual (layer 0's flushed output).
         * On the single-token ring the block input aliases that ping tensor, so
         * this used to pass `block_input`; a batch slot whose input lives in a
         * dedicated tensor would then read the pre-layer-0 hidden instead. */
        if(status==FG_OK)status=ple_decode_into(e,layer_input,ngram_embedding,
            e->hyper_output,e->hyper_output_b,&ple_input,err);
        if(status!=FG_OK)return status;
        layer_input=ple_input;
    }
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;
    const fg_vk_tensor *residual=NULL;
    if(status==FG_OK)status=ensure_decode_batch(vk,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_read",err);
    if(status==FG_OK)status=gr_read_batch_into(e,layer,false,layer_input,1u,
        e->injection,&mixed,&residual,&injection,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,
        (layer&3u)==3u?"qsa":"gdn",err);
    if(status==FG_OK&&(layer&3u)==3u)
        status=fg_owner_qsa_decode(e,layer,token,position,mixed,&block,err);
    else if(status==FG_OK)status=fg_owner_gdn_decode(e,layer,mixed,&block,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_write",err);
    if(status==FG_OK)status=gr_write_batch_into(e,residual,block,injection,1u,
        e->hyper_output,e->hyper_output_b,&after_attention,err);
                              after_attention,err);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_read",err);
    if(status==FG_OK)status=gr_read_batch_into(e,layer,true,after_attention,1u,
        e->injection,&mixed,&residual,&injection,err);
    fg_vk_tensor *router_w=status==FG_OK?weight(e,layer,"ffn_gate_inp.weight",err):NULL;
    if(status==FG_OK&&!router_w)status=FG_ERR_MISMATCH;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router",err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router_w,mixed,
        FG_HIDDEN_SIZE,FG_EXPERT_COUNT,1u,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router_quantization",err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,mixed,
        FG_HIDDEN_SIZE,1u,err);
    fg_vk_tensor *shared_gate_w=NULL,*gate_w=NULL,*up_w=NULL,*down_w=NULL;
    if(status==FG_OK){
        shared_gate_w=weight(e,layer,"ffn_gate_inp_shexp.weight",err);
        gate_w=weight(e,layer,"ffn_gate_shexp.weight",err);
        up_w=weight(e,layer,"ffn_up_shexp.weight",err);
        down_w=weight(e,layer,"ffn_down_shexp.weight",err);
        if(!shared_gate_w||!gate_w||!up_w||!down_w)status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"shared_expert",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate_w,mixed,
        FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up_w,mixed,
        FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)fg_vk_next_dispatch_independent(vk);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->shared_scalar,shared_gate_w,
        mixed,FG_HIDDEN_SIZE,1u,1u,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,
        e->shared_up,640u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_output,down_w,
        e->shared_mid,640u,FG_HIDDEN_SIZE,1u,1.0f,err);
    fg_vk_tensor *expert_output=NULL;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"expert_decode",err);
    if(status==FG_OK)status=expert(expert_context,layer,e->activation_q8k,
        e->router_logits,&expert_output,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"moe_reduce",err);
    if(status==FG_OK)status=fg_vk_moe_decode_shared_add(vk,e->reduced,expert_output,
        e->shared_output,e->shared_scalar,FG_HIDDEN_SIZE,1u,err);
                              e->reduced,err);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);
    if(status==FG_OK)status=gr_write_batch_into(e,residual,e->reduced,injection,1u,
        e->hyper_output,e->hyper_output_b,current,err);
                              *current,err);}
    return status;
}

/* Chained ring block.  Every layer's common path, shared expert, GPU routing
 * and fused expert pair are recorded into one submission; QSA selection may
 * still flush the recording when it needs a host readback, but nothing in the
 * block needs a per-layer fence, host top-K, or expert readback.  Runs of
 * non-QSA layers are token-invariant and are recorded once into static command
 * buffers, then replayed; the token-dependent QSA layers stay dynamic. */
fg_status fg_owner_decode_block_chained(fg_owner_executor *e,uint32_t first_layer,
    uint32_t last_layer,uint32_t token,const uint32_t position[3],
    const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,
    fg_owner_expert_inline_fn expert,void *expert_context,
    fg_vk_tensor **output,fg_error *err){
    if(!e||!position||!hyper_input||!expert||!output||
       first_layer>last_layer||last_layer>=FG_LAYER_COUNT||
       (first_layer<=1u)!=(ngram_embedding!=NULL)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid chained owner decode block arguments");
        return FG_ERR_ARGUMENT;
    }
    memset(&e->decode_slots[0].pending_write,0,sizeof(e->decode_slots[0].pending_write));
    fg_vk_context *vk=fg_model_vk(e->model);
    fg_vk_tensor *current=(fg_vk_tensor *)hyper_input;
    fg_status status=FG_OK;
    /* Static replay records slot-0 transients and advances slot-0 state; a
     * recorded run replayed against another session's state is silent
     * corruption.  Session 0 keeps the validated replay; other sessions run
     * dynamically and any recorded run bound to a different session fails
     * closed instead of replaying. */
    bool static_allowed=chained_static_allowed(vk)&&e->active_session==0u;
    uint32_t layer=first_layer;
    while(status==FG_OK&&layer<=last_layer){
        if((layer&3u)==3u){
            status=owner_record_layer(e,layer,token,position,hyper_input,
                ngram_embedding,expert,expert_context,&current,err);
            if(status==FG_OK&&layer<last_layer)status=fg_vk_flush(vk,err);
            if(status==FG_OK)status=chained_layer_trace(fg_model_rank(e->model),vk,layer,token,current,err);
            layer++;
            continue;
        }
        uint32_t run_last=layer;
        while(run_last+1u<=last_layer&&((run_last+1u)&3u)!=3u)run_last++;
        uint32_t slot=0u;
        for(uint32_t l=first_layer;l<layer;l++)if((l&3u)==3u)slot++;
        uint32_t rerecord=static_rerecord_period();
        if(status==FG_OK&&static_allowed&&slot<FG_VK_STATIC_SLOTS&&rerecord&&
           (token%rerecord)==0u)
            status=fg_vk_static_reset(vk,slot,err);
        if(status==FG_OK&&static_allowed&&slot<FG_VK_STATIC_SLOTS&&fg_vk_static_recorded(vk,slot)){
            if(e->static_run_session_valid[slot]&&
               e->static_run_session[slot]!=e->active_session){
                fg_error_set(err,FG_ERR_MISMATCH,
                    "static replay is bound to owner session %u, not %u",
                    e->static_run_session[slot],e->active_session);
                status=FG_ERR_MISMATCH;
            }
            if(status!=FG_OK)break;
            if(frame_trace_enabled()){
                const fg_vk_tensor *cur=current;
                for(uint32_t l=layer;l<=run_last;l++){
                    const fg_vk_tensor *run_input=cur;
                    if(l==1u)run_input=chained_ping_next(e,hyper_input);
                    cur=chained_ping_next(e,run_input);
                }
                fprintf(stderr,"STATIC_PING rank=%u token=%u slot=%u sim=%p recorded=%p\n",
                    fg_model_rank(e->model),token,slot,(const void *)cur,
                    (const void *)e->static_run_output[slot]);
            }
            if(e->static_run_output[slot])current=e->static_run_output[slot];
            status=fg_vk_static_submit(vk,slot,err);
            if(status==FG_OK&&static_check_enabled())
                status=static_run_check(e,vk,slot,token,err);
        }else if(status==FG_OK&&static_allowed&&slot<FG_VK_STATIC_SLOTS){
            status=fg_vk_static_begin(vk,slot,err);
            for(uint32_t l=layer;status==FG_OK&&l<=run_last;l++)
                status=owner_record_layer(e,l,token,position,hyper_input,
                    ngram_embedding,expert,expert_context,&current,err);
            if(status==FG_OK)status=fg_vk_static_end(vk,slot,err);
            if(status==FG_OK)status=fg_vk_static_submit(vk,slot,err);
            if(status==FG_OK){
                e->static_run_output[slot]=current;
                e->static_run_session[slot]=e->active_session;
                e->static_run_session_valid[slot]=true;
            }
            if(status==FG_OK&&static_check_enabled())
                status=static_run_check(e,vk,slot,token,err);
        }else{
            for(uint32_t l=layer;status==FG_OK&&l<=run_last;l++){
                status=owner_record_layer(e,l,token,position,hyper_input,
                    ngram_embedding,expert,expert_context,&current,err);
                if(status==FG_OK&&l<last_layer)status=fg_vk_flush(vk,err);
                if(status==FG_OK)status=chained_layer_trace(fg_model_rank(e->model),vk,l,token,current,err);
                if(status==FG_OK)gdn_state_trace(e,l,token,"post",current);
            }
        }
        layer=run_last+1u;
    }
    if(status==FG_OK){
        if(fg_vk_batch_active(vk))status=finish_batch(vk,status,err);
        if(status==FG_OK)status=fg_vk_static_drain(vk,err);
    }
    if(status==FG_OK)*output=current;
    return status;
}

fg_status fg_owner_prefill_layer(fg_owner_executor *e,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint16_t token_count,const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embeddings,fg_owner_prefill_dispatch_fn dispatch,void *dispatch_context,fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,void *qsa_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!positions||!token_count||token_count>e->max_tokens||!hyper_input||!dispatch||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"text layer prefill is not on its owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embeddings!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 batched PLE embedding presence mismatch");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(e->model);bool profiling=fg_vk_profile_active(vk);double t0=profiling?ts_ms():0.0;const fg_vk_tensor *layer_input=hyper_input;
    if(layer==1u){fg_vk_tensor *ple_input=NULL;fg_status status=profiling?fg_vk_profile_set_scope(vk,"ple_prefill",err):FG_OK;if(status==FG_OK)status=fg_owner_ple_prefill(e,hyper_input,ngram_embeddings,token_count,&ple_input,err);if(status!=FG_OK)return status;layer_input=ple_input;}
    double t_ple=profiling?ts_ms():0.0;
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;fg_status status=profiling?fg_vk_profile_set_scope(vk,"gr_attn_read_prefill",err):FG_OK;
    if(status==FG_OK)status=fg_owner_gr_read_batch(e,layer,false,layer_input,token_count,&mixed,&residual,&injection,err);
    double t_attn_read=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,(layer&3u)==3u?"qsa_prefill":"gdn_prefill",err);
    if(status==FG_OK&&(layer&3u)==3u&&qsa_dispatch)
        status=qsa_dispatch(qsa_context,layer,first_token,positions,token_count,mixed,&block,err);
    else if(status==FG_OK&&(layer&3u)==3u)
        status=fg_owner_qsa_prefill(e,layer,first_token,positions,
                                    token_count,mixed,&block,err);
    else if(status==FG_OK)
        status=fg_owner_gdn_prefill(e,layer,token_count,mixed,&block,err);
    double t_attention=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_attn_write_prefill",err);
    if(status==FG_OK)status=fg_owner_gr_write_batch(e,residual,block,injection,token_count,&after_attention,err);
    double t_attn_write=profiling?ts_ms():0.0;
    if(status!=FG_OK)return status;
    if(profiling)status=fg_vk_profile_set_scope(vk,"gr_ffn_read_prefill",err);
    if(status==FG_OK)status=fg_owner_gr_read_batch(e,layer,true,after_attention,token_count,&mixed,&residual,&injection,err);
    double t_ffn_read=profiling?ts_ms():0.0;
    uint16_t expert_ids[FG_PREFILL_MAX_PAIRS];float gates[FG_PREFILL_MAX_PAIRS];const uint8_t *activations=NULL;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"router_prefill",err);
    if(status==FG_OK)status=owner_prefill_routes(e,layer,mixed,token_count,expert_ids,gates,&activations,err);
    shared_expert_work shared={e,mixed,e->shared_output,e->shared_scalar,layer,token_count};
    double t_router=profiling?ts_ms():0.0;
    memset(e->prefill_results,0,sizeof(e->prefill_results));uint32_t result_count=0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"expert_dispatch_prefill",err);
    if(status==FG_OK)status=dispatch(dispatch_context,layer,first_token,token_count,expert_ids,gates,activations,owner_shared_expert,&shared,e->prefill_results,&result_count,err);
    double t_dispatch=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"expert_reduce_prefill",err);
    if(status==FG_OK)status=fg_owner_moe_reduce_batch(e,layer,first_token,token_count,expert_ids,gates,e->prefill_results,result_count,&block,err);
    double t_reduce=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_ffn_write_prefill",err);
    if(status==FG_OK)status=fg_owner_gr_write_batch(e,residual,block,injection,token_count,output,err);
    if(profiling){double t_end=ts_ms();fprintf(stderr,"PREFILL_PROFILE_STAGE first=%u tokens=%u layer=%u ple_ms=%.3f attn_read_ms=%.3f attention_ms=%.3f attn_write_ms=%.3f ffn_read_ms=%.3f router_ms=%.3f dispatch_ms=%.3f reduce_ms=%.3f final_write_ms=%.3f total_ms=%.3f\n",first_token,token_count,layer,t_ple-t0,t_attn_read-t_ple,t_attention-t_attn_read,t_attn_write-t_attention,t_ffn_read-t_attn_write,t_router-t_ffn_read,t_dispatch-t_router,t_reduce-t_dispatch,t_end-t_reduce,t_end-t0);}
    return status;
}

static fg_status prefill_layer_begin_impl(fg_owner_executor *e,uint32_t slot,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint16_t token_count,const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embeddings,fg_owner_prefill_fire_fn fire,void *fire_context,fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,void *qsa_context,fg_error *err){
    if(!e||slot>=FG_OWNER_SLOT_COUNT||!positions||!token_count||token_count>e->max_tokens||!hyper_input||!fire||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"text layer prefill is not on its owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embeddings!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 batched PLE embedding presence mismatch");return FG_ERR_MISMATCH;}
    fg_owner_decode_slot *dslot=&e->decode_slots[slot];fg_owner_prefill_slot *frame=&e->prefill_slots[slot];
    if(frame->active){fg_error_set(err,FG_ERR_MISMATCH,"prefill layer begin slot is already active");return FG_ERR_MISMATCH;}
    numerics_trace_tensor("IN",fg_model_rank(e->model),layer,first_token,token_count,hyper_input,err);
    frame->active=true;frame->layer=layer;frame->first_token=first_token;frame->token_count=token_count;frame->residual=NULL;
    fg_vk_context *vk=fg_model_vk(e->model);bool profiling=fg_vk_profile_active(vk);const fg_vk_tensor *layer_input=hyper_input;
    bool step_trace=frame_trace_enabled();
#define PREFILL_STEP(name) do{if(step_trace)fprintf(stderr,"PREFILL_STEP slot=%u layer=%u %s status=%d msg=%s\n",slot,layer,(name),(int)status,err?err->message:"-");}while(0)
    fg_status status=profiling?fg_vk_profile_set_scope(vk,"ple_prefill",err):FG_OK;
    if(status==FG_OK&&layer==1u){fg_vk_tensor *ple_input=NULL;if(status==FG_OK)status=ple_prefill_into(e,hyper_input,ngram_embeddings,dslot->ping[0],dslot->ping[1],token_count,&ple_input,err);PREFILL_STEP("ple");if(status!=FG_OK){frame->active=false;return status;}layer_input=ple_input;}
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_attn_read_prefill",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err); /* fuse the common path into one submission */
    if(status==FG_OK)status=gr_read_batch_into(e,layer,false,layer_input,token_count,dslot->injection,&mixed,&residual,&injection,err);
    PREFILL_STEP("attn_read");
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,(layer&3u)==3u?"qsa_prefill":"gdn_prefill",err);
    if(status==FG_OK&&(layer&3u)==3u&&qsa_dispatch){
        status=finish_batch(vk,status,err);
        if(status==FG_OK)status=qsa_dispatch(qsa_context,layer,first_token,positions,token_count,mixed,&block,err);
        if(status==FG_OK)status=fg_vk_begin(vk,err);
    }else if(status==FG_OK&&(layer&3u)==3u){
        status=finish_batch(vk,status,err);
        if(status==FG_OK)status=fg_owner_qsa_prefill(e,layer,first_token,positions,token_count,mixed,&block,err);
        if(status==FG_OK)status=fg_vk_begin(vk,err);
    }else if(status==FG_OK)status=fg_owner_gdn_prefill(e,layer,token_count,mixed,&block,err);
    PREFILL_STEP("attention");
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_attn_write_prefill",err);
    if(status==FG_OK)status=gr_write_batch_into(e,residual,block,injection,token_count,dslot->ping[0],dslot->ping[1],&after_attention,err);
    PREFILL_STEP("attn_write");
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_ffn_read_prefill",err);
    if(status==FG_OK)status=gr_read_batch_into(e,layer,true,after_attention,token_count,dslot->injection,&mixed,&residual,&injection,err);
    status=finish_batch(vk,status,err);
    PREFILL_STEP("ffn_read");
    const uint8_t *activations=NULL;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"router_prefill",err);
    if(status==FG_OK)status=owner_prefill_routes(e,layer,mixed,token_count,frame->expert_ids,frame->gates,&activations,err);
    PREFILL_STEP("routes");
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"expert_dispatch_prefill",err);
    if(status==FG_OK)status=fire(fire_context,layer,first_token,token_count,frame->expert_ids,frame->gates,activations,err);
    PREFILL_STEP("fire");
    shared_expert_work shared={e,mixed,dslot->shared_output,dslot->shared_scalar,layer,token_count};
    if(status==FG_OK)status=owner_shared_expert(&shared,err);
    PREFILL_STEP("shared");
#undef PREFILL_STEP
    if(status!=FG_OK){frame->active=false;return status;}
    frame->residual=residual;
    return status;
}

static fg_status prefill_layer_finish_impl(fg_owner_executor *e,uint32_t slot,fg_owner_prefill_collect_fn collect,void *collect_context,fg_vk_tensor **output,fg_error *err){
    if(!e||slot>=FG_OWNER_SLOT_COUNT||!collect||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill layer finish");return FG_ERR_ARGUMENT;}
    fg_owner_decode_slot *dslot=&e->decode_slots[slot];fg_owner_prefill_slot *frame=&e->prefill_slots[slot];
    if(!frame->active){fg_error_set(err,FG_ERR_MISMATCH,"prefill layer finish does not match an active begin");return FG_ERR_MISMATCH;}
    fg_prefill_result results[FG_GROUP_SIZE]={0};uint32_t result_count=0;
    fg_status status=collect(collect_context,frame->layer,frame->first_token,frame->token_count,results,&result_count,err);
    fg_vk_tensor *block=NULL;
    if(status==FG_OK)status=moe_reduce_batch_into(e,frame->layer,frame->first_token,frame->token_count,frame->expert_ids,frame->gates,results,result_count,dslot->shared_output,dslot->shared_scalar,dslot->reduced,&block,err);
    if(status==FG_OK)status=gr_write_batch_into(e,frame->residual,block,dslot->injection,frame->token_count,dslot->ping[0],dslot->ping[1],output,err);
    if(status==FG_OK)numerics_trace_tensor("OUT",fg_model_rank(e->model),frame->layer,
        frame->first_token,frame->token_count,*output,err);
    frame->active=false;
    return status;
}

fg_status fg_owner_prefill_layer_begin(fg_owner_executor *e,uint32_t slot,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint16_t token_count,const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embeddings,fg_owner_prefill_fire_fn fire,void *fire_context,fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,void *qsa_context,fg_error *err){
    return prefill_layer_begin_impl(e,slot,layer,first_token,positions,token_count,hyper_input,ngram_embeddings,fire,fire_context,qsa_dispatch,qsa_context,err);
}

fg_status fg_owner_prefill_layer_finish(fg_owner_executor *e,uint32_t slot,fg_owner_prefill_collect_fn collect,void *collect_context,fg_vk_tensor **output,fg_error *err){
    return prefill_layer_finish_impl(e,slot,collect,collect_context,output,err);
}