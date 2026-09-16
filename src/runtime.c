#include "fg_runtime.h"
#include "fg_topology.h"
#include "fg_expert.h"
#include "fg_fabric.h"
#include "fg_model.h"
#include "fg_ngram.h"
#include "fg_owner.h"
#include "fg_output.h"
#include "fg_q38_schema.h"
#include "fg_qsa.h"
#include "fg_qsa_cache.h"
#include "fg_qsa_owner.h"
#include "fg_qsa_replica.h"
#include "fg_qsa_state.h"
#include "fg_tokenizer.h"
#include "fg_uring.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static fg_status load_checked(const char *path,fg_manifest **out,fg_error *err){fg_manifest *m=malloc(sizeof(*m));if(!m){fg_error_set(err,FG_ERR_OOM,"allocate manifest");return FG_ERR_OOM;}fg_status rc=fg_manifest_read(path,m,err);if(rc==FG_OK)rc=fg_manifest_validate_deployment(m,err);if(rc!=FG_OK){free(m);return rc;}*out=m;return FG_OK;}
static fg_status manifest_directory(const char *path,char output[1024],fg_error *err){size_t length=strlen(path);if(!length||length>=1024u){fg_error_set(err,FG_ERR_ARGUMENT,"manifest path is invalid");return FG_ERR_ARGUMENT;}memcpy(output,path,length+1u);char *slash=strrchr(output,'/');if(!slash){snprintf(output,1024,".");return FG_OK;}if(slash==output)slash[1]=0;else *slash=0;return FG_OK;}

typedef struct token_profile_capture {fg_vk_context *vk;struct timespec start;bool owned;} token_profile_capture;
static double elapsed_seconds(const struct timespec *start,const struct timespec *end);
static uint32_t output_split_timeout_ms(void);
static bool output_split_trace_enabled(void);
static void output_split_trace(const char *what,uint32_t rank,uint32_t token,uint32_t detail);

static bool token_profile_requested(uint32_t token){const char *requested=getenv("FG_PROFILE_TOKEN");char value[16];if(!requested||!*requested)return false;snprintf(value,sizeof(value),"%u",token);return strcmp(requested,value)==0;}
static bool prefill_profile_requested(void){const char *enabled=getenv("FG_PREFILL_PROFILE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool prefill_ring_requested(void){const char *enabled=getenv("FG_PREFILL_RING");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
/* Ring decode is the default whenever ring prefill is active: it beats the
 * legacy expert-parallel decode on sustained 4K and short prompts.  Set
 * FG_DECODE_RING=0 to opt back into the legacy replay. */
static bool decode_ring_requested(void){
    const char *disabled=getenv("FG_DECODE_RING");
    return !(disabled&&*disabled&&strcmp(disabled,"0")==0);
}
static bool decode_ring_trace_enabled(void){const char *enabled=getenv("FG_DECODE_RING_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
/* Direct final-block -> output-owner handoff: rank 0 ships the sampler config
 * for a decode token up front and the last block owner ships the hidden
 * straight to the output owner, so the 40 KiB never round-trips through rank 0.
 * Both the coordinator and the last block owner derive the route from the
 * manifest, and the rank-0 relay remains the fallback for topologies where the
 * output owner or the last block owner is rank 0. */
static bool decode_direct_output_eligible(const fg_manifest *manifest){
    if(!manifest)return false;
    uint32_t owner=fg_output_owner_rank(manifest);
    uint32_t last=manifest->layer_owner[FG_LAYER_COUNT-1u];
    return owner!=0u&&last!=0u&&owner!=last;
}
static bool decode_profile_enabled(void){const char *enabled=getenv("FG_DECODE_PROFILE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool frame_trace_enabled(void){const char *enabled=getenv("FG_FRAME_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool route_trace_enabled(void){const char *enabled=getenv("FG_TRACE_ROUTES");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool prefix_trace_enabled(void){const char *enabled=getenv("FG_PREFIX_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool numerics_trace_enabled(void){const char *enabled=getenv("FG_NUMERICS_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static void numerics_trace_values(const char *phase,uint32_t rank,uint32_t layer,
    uint32_t first_token,const float *values,uint64_t count){
    if(!numerics_trace_enabled()||!values||!count)return;
    const uint8_t *bytes=(const uint8_t *)values;uint64_t hash=UINT64_C(1469598103934665603);
    for(uint64_t i=0;i<count*4u;i++){hash^=bytes[i];hash*=UINT64_C(1099511628211);}
    double sum=0.0;float min=values[0],max=values[0];
    for(uint64_t i=0;i<count;i++){float v=values[i];sum+=v;if(v<min)min=v;if(v>max)max=v;}
    fprintf(stderr,"FG_NUMERICS rank=%u layer=%u phase=%s first=%u tokens=%llu hash=%016llx "
        "sum=%.6f min=%.6g max=%.6g f0=%.6g fl=%.6g\n",rank,layer,phase,first_token,
        (unsigned long long)count,(unsigned long long)hash,sum,min,max,values[0],
        values[count-1u]);
}
static void numerics_trace_host(const char *phase,uint32_t rank,uint32_t layer,
    uint32_t first_token,uint32_t tokens,const float *hyper){
    numerics_trace_values(phase,rank,layer,first_token,hyper,
        (uint64_t)tokens*FG_HYPER_WIDTH);
}
static uint64_t critical_ns(void){struct timespec value;clock_gettime(CLOCK_REALTIME,&value);return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;}
static uint64_t monotonic_ms(void){struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return (uint64_t)value.tv_sec*1000u+(uint64_t)value.tv_nsec/1000000u;}
_Static_assert(FG_NGRAM_ROW_BYTES==FG_NGRAM_WIRE_ROW_BYTES,"n-gram row wire size mismatch");
static uint64_t coordinator_prefill_work_wire_bytes(uint32_t tokens);

enum {
    FG_TRANSPORT_READY=0u,
    FG_TRANSPORT_PENDING=1u,
    FG_TRANSPORT_POISONED=2u
};

static bool transport_ready(const atomic_uint *state){
    return atomic_load(state)==FG_TRANSPORT_READY;
}

static void transport_pending(atomic_uint *state){
    unsigned expected=FG_TRANSPORT_READY;
    atomic_compare_exchange_strong(state,&expected,FG_TRANSPORT_PENDING);
}

static void transport_complete(atomic_uint *state){
    unsigned expected=FG_TRANSPORT_PENDING;
    atomic_compare_exchange_strong(state,&expected,FG_TRANSPORT_READY);
}

static void transport_poison(atomic_uint *state){
    atomic_store(state,FG_TRANSPORT_POISONED);
}

static fg_status token_profile_prepare(fg_vk_context *vk,fg_error *err){
    if(!getenv("FG_PROFILE_TOKEN"))return FG_OK;
    fg_vk_profile profile={0};
    fg_status status=fg_vk_profile_begin(vk,err);
    if(status==FG_OK)status=fg_vk_profile_end(vk,&profile,err);
    return status;
}

static fg_status token_profile_begin(token_profile_capture *capture,fg_vk_context *vk,uint32_t token,fg_error *err){
    memset(capture,0,sizeof(*capture));capture->vk=vk;if(!token_profile_requested(token)||fg_vk_profile_active(vk))return FG_OK;fg_status status=fg_vk_profile_begin(vk,err);if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&capture->start);capture->owned=true;}return status;
}

static fg_status token_profile_end(token_profile_capture *capture,uint32_t rank,const char *kind,uint32_t token,uint32_t layer,fg_status status,fg_error *err){
    if(!capture->owned)return status;
    fg_vk_profile profile={0};fg_error profile_error={0};fg_status profile_status=fg_vk_profile_end(capture->vk,&profile,status==FG_OK?err:&profile_error);
    if(profile_status!=FG_OK)return status==FG_OK?profile_status:status;
    struct timespec end;clock_gettime(CLOCK_MONOTONIC,&end);double wall_ms=(double)(end.tv_sec-capture->start.tv_sec)*1000.0+(double)(end.tv_nsec-capture->start.tv_nsec)*1e-6;
    fprintf(stderr,"TOKEN_PROFILE rank=%u token=%u kind=%s layer=%u wall_ms=%.3f gpu_ms=%.3f kernel_ms=%.3f vk_overhead_ms=%.3f wall_residual_ms=%.3f submissions=%llu dispatches=%llu\n",rank,token,kind,layer,wall_ms,profile.gpu_ms,profile.kernel_ms,profile.gpu_ms-profile.kernel_ms,wall_ms-profile.gpu_ms,(unsigned long long)profile.submissions,(unsigned long long)profile.dispatches);
    for(uint32_t i=0;i<profile.kernel_count;i++){const fg_vk_profile_kernel *kernel=&profile.kernels[i];fprintf(stderr,"TOKEN_PROFILE_KERNEL rank=%u token=%u kind=%s layer=%u scope=%s kernel=%s calls=%llu gpu_ms=%.3f\n",rank,token,kind,layer,kernel->scope,kernel->name,(unsigned long long)kernel->invocations,kernel->gpu_ms);}
    return status;
}

static fg_status rank_ready(fg_fabric *fabric,uint32_t self,fg_error *err){
    /* Send READY to every peer first (fire-and-forget into TCP). */
    for(uint32_t peer=0;peer<FG_RANK_COUNT;peer++){
        if(peer!=self){
            fg_status status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_READY,0,0,0,NULL,0,err);
            if(status!=FG_OK)return status;
        }
    }
    /* Receive READY from each peer individually.  Using per-peer recv
       (not recv_any) guarantees we read exactly one frame per peer and
       never accidentally consume a later message — such as
       SESSION_BEGIN — that a faster peer already queued. */
    for(uint32_t peer=0;peer<FG_RANK_COUNT;peer++){
        if(peer==self)continue;
        uint32_t bytes=0;
        fg_frame_header header;
        fg_status status=fg_fabric_recv(fabric,peer,FG_FABRIC_CONTROL,&header,NULL,0,&bytes,err);
        if(status!=FG_OK)return status;
        if(fg_frame_type(&header)!=FG_MSG_READY||bytes){
            fg_error_set(err,FG_ERR_MISMATCH,"invalid READY from rank %u",peer);
            return FG_ERR_MISMATCH;
        }
    }
    return FG_OK;
}

/* Async expert dispatch context — tracks in-flight header/payload recvs */
typedef struct async_expert_context {
    fg_fabric *fabric;fg_expert_executor *expert;const fg_manifest *manifest;uint32_t self;
    uint64_t request_id;uint32_t sequence;
    atomic_uint *transport_state;
    bool send_failed,discard_results;
    uint32_t expected_peer_mask,seen_peer_mask;
    /* Saved routes for deferred local expert compute */
    fg_expert_route routes[FG_GROUP_SIZE];uint32_t route_count;
    uint8_t activation_copy[FG_Q8K_ACTIVATION_BYTES];
    /* Per-peer recv state */
    uint32_t remote_count;
    uint32_t recv_peers[FG_GROUP_SIZE];
    fg_frame_header recv_headers[FG_GROUP_SIZE];
    uint8_t *recv_payloads[FG_GROUP_SIZE]; /* each malloc'd single-result payload */
    bool critical_trace;
    struct {uint32_t peer;uint64_t start_ns,end_ns;} send_trace[FG_LAYER_COUNT][FG_GROUP_SIZE];
    struct {uint32_t peer,ready_mask,bytes;uint64_t poll_start_ns,ready_ns,header_end_ns,payload_end_ns,validate_end_ns,decode_end_ns;} recv_trace[FG_LAYER_COUNT][FG_GROUP_SIZE];
    uint8_t send_trace_count[FG_LAYER_COUNT],recv_trace_count[FG_LAYER_COUNT];
} async_expert_context;

static double dispatch_ts(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e3+(double)t.tv_nsec*1e-6;}
/* Fire all remote work and return; collection stays on the fast poll/recv_any path. */
static fg_status fire_experts(void *opaque,uint32_t layer,uint32_t token,const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],const uint8_t *activation,fg_error *err){
    async_expert_context *ctx=opaque;fg_expert_route routes[FG_GROUP_SIZE];uint32_t route_count=0;
    fg_status status=fg_partition_route(ctx->manifest,layer,expert_ids,gates,routes,&route_count,err);
    uint8_t work_wire[FG_GROUP_SIZE][FG_DECODE_WORK_BYTES];ctx->remote_count=0;ctx->send_failed=false;ctx->discard_results=false;ctx->expected_peer_mask=0;ctx->seen_peer_mask=0;
    /* Save local routes for deferred compute in collect */
    ctx->route_count=route_count;
    memcpy(ctx->routes,routes,sizeof(routes));
    memcpy(ctx->activation_copy,activation,FG_Q8K_ACTIVATION_BYTES);
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){
        if(routes[r].destination_rank==ctx->self)continue;
        fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)ctx->self,.destination_rank=routes[r].destination_rank,.selected_count=routes[r].selected_count,.position=token};
        for(uint32_t i=0;i<routes[r].selected_count;i++){work.expert_ids[i]=routes[r].global_expert_ids[i];work.routing_slots[i]=routes[r].routing_slots[i];work.gates[i]=routes[r].gates[i];}
        memcpy(work.activation_q8k,activation,FG_Q8K_ACTIVATION_BYTES);
        uint32_t send_index=ctx->remote_count;status=fg_decode_work_encode(work_wire[send_index],&work,err);
        uint32_t trace_index=ctx->send_trace_count[layer];if(ctx->critical_trace&&trace_index<FG_GROUP_SIZE){ctx->send_trace[layer][trace_index].peer=work.destination_rank;ctx->send_trace[layer][trace_index].start_ns=critical_ns();}
        if(status==FG_OK){
            status=fg_fabric_send(ctx->fabric,work.destination_rank,FG_FABRIC_CONTROL,
                                  FG_MSG_DECODE_WORK,ctx->request_id,ctx->sequence,0,
                                  work_wire[send_index],FG_DECODE_WORK_BYTES,err);
            if(status==FG_OK){
                ctx->remote_count++;
                ctx->expected_peer_mask|=1u<<work.destination_rank;
                transport_pending(ctx->transport_state);
            }else{
                ctx->send_failed=true;
                transport_poison(ctx->transport_state);
            }
        }
        if(ctx->critical_trace&&trace_index<FG_GROUP_SIZE){ctx->send_trace[layer][trace_index].end_ns=critical_ns();ctx->send_trace_count[layer]++;}
    }
    ctx->discard_results=status!=FG_OK;
    if(status==FG_OK&&(token_profile_requested(token)||route_trace_enabled())){uint32_t local=0,local_selected=0,selected=0,rank_mask=0;for(uint32_t r=0;r<route_count;r++){bool is_local=routes[r].destination_rank==ctx->self;local+=is_local;local_selected+=is_local?routes[r].selected_count:0u;selected+=routes[r].selected_count;rank_mask|=1u<<routes[r].destination_rank;}fprintf(stderr,"EP_ROUTE_TRACE token=%u layer=%u routes=%u remotes=%u local=%u local_selected=%u selected=%u rank_mask=%u",token,layer,route_count,ctx->remote_count,local,local_selected,selected,rank_mask);if(route_trace_enabled()){fprintf(stderr," expert_ids=");for(uint32_t slot=0;slot<FG_TOP_K;slot++)fprintf(stderr,"%s%u",slot?",":"",expert_ids[slot]);fprintf(stderr," expert_ranks=");for(uint32_t slot=0;slot<FG_TOP_K;slot++)fprintf(stderr,"%s%u",slot?",":"",ctx->manifest->expert_rank[layer][expert_ids[slot]]);}fputc('\n',stderr);}
    return status;
}

/* Compute local experts, then collect remote results in arrival order. */
static fg_status collect_experts(void *opaque,uint32_t layer,uint32_t token,fg_expert_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    async_expert_context *ctx=opaque;*result_count=0;fg_status status=FG_OK;
    bool transport_failed=ctx->send_failed;
    /* Compute local experts while remote results are in flight */
    for(uint32_t r=0;status==FG_OK&&!ctx->discard_results&&r<ctx->route_count;r++){
        if(ctx->routes[r].destination_rank!=ctx->self)continue;
        fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)ctx->self,.destination_rank=(uint8_t)ctx->self,.selected_count=ctx->routes[r].selected_count,.position=token};
        for(uint32_t i=0;i<ctx->routes[r].selected_count;i++){work.expert_ids[i]=ctx->routes[r].global_expert_ids[i];work.routing_slots[i]=ctx->routes[r].routing_slots[i];work.gates[i]=ctx->routes[r].gates[i];}
        memcpy(work.activation_q8k,ctx->activation_copy,FG_Q8K_ACTIVATION_BYTES);
        status=fg_expert_decode(ctx->expert,&work,&results[*result_count],err);
        if(status==FG_OK)(*result_count)++;
    }
    uint32_t received=0;
    for(;received<ctx->remote_count;received++){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        fg_error receive_error={0};
        fg_fabric_recv_timing timing={0};
        fg_status receive_status=ctx->critical_trace?
            fg_fabric_recv_any_timed(ctx->fabric,FG_FABRIC_BULK,&peer,&header,
                ctx->recv_payloads[0],FG_EXPERT_RESULT_SINGLE_BYTES,&bytes,&timing,
                status==FG_OK?err:&receive_error):
            fg_fabric_recv_any(ctx->fabric,FG_FABRIC_BULK,&peer,&header,
                ctx->recv_payloads[0],FG_EXPERT_RESULT_SINGLE_BYTES,&bytes,
                status==FG_OK?err:&receive_error);
        if(receive_status!=FG_OK){
            transport_failed=true;
            if(status==FG_OK)status=receive_status;
            break;
        }
        uint32_t peer_bit=peer<FG_RANK_COUNT?1u<<peer:0u;
        if(!peer_bit||!(ctx->expected_peer_mask&peer_bit)||
           (ctx->seen_peer_mask&peer_bit)||
           fg_frame_type(&header)!=FG_MSG_EXPERT_RESULT||
           fg_frame_request_id(&header)!=ctx->request_id||
           fg_frame_sequence(&header)!=ctx->sequence){
            transport_failed=true;
            if(status==FG_OK){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "stale split expert result from peer %u",peer);
                status=FG_ERR_MISMATCH;
            }
        }else{
            ctx->seen_peer_mask|=peer_bit;
        }
        if(status==FG_OK){
            status=fg_expert_result_decode(&results[*result_count],
                                           ctx->recv_payloads[0],bytes,err);
            if(status!=FG_OK)transport_failed=true;
        }
        if(ctx->critical_trace&&received<FG_GROUP_SIZE){ctx->recv_trace[layer][received].peer=peer;ctx->recv_trace[layer][received].ready_mask=timing.ready_mask;ctx->recv_trace[layer][received].bytes=bytes;ctx->recv_trace[layer][received].poll_start_ns=timing.poll_start_ns;ctx->recv_trace[layer][received].ready_ns=timing.ready_ns;ctx->recv_trace[layer][received].header_end_ns=timing.header_end_ns;ctx->recv_trace[layer][received].payload_end_ns=timing.payload_end_ns;ctx->recv_trace[layer][received].validate_end_ns=timing.validate_end_ns;ctx->recv_trace[layer][received].decode_end_ns=critical_ns();ctx->recv_trace_count[layer]++;}
        if(status==FG_OK)(*result_count)++;
    }
    if(received==ctx->remote_count&&!transport_failed&&
       ctx->seen_peer_mask==ctx->expected_peer_mask)
        transport_complete(ctx->transport_state);
    else if(transport_failed||ctx->seen_peer_mask!=ctx->expected_peer_mask)
        transport_poison(ctx->transport_state);
    return status;
}

static fg_status handle_expert_work(fg_fabric *fabric,fg_expert_executor *expert,fg_vk_context *vk,uint32_t self,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,const fg_fabric_recv_timing *receive_timing,fg_expert_result *result,uint8_t *wire,fg_error *err){
    double tw0=dispatch_ts();
    fg_decode_work work;fg_status status=fg_decode_work_decode(&work,payload,bytes,err);if(status==FG_OK&&(peer!=work.source_rank||work.destination_rank!=self)){fg_error_set(err,FG_ERR_MISMATCH,"decode work peer/rank mismatch");status=FG_ERR_MISMATCH;}
    double tw_decode=dispatch_ts();uint64_t trace_decode=critical_ns();
    token_profile_capture capture={0};if(status==FG_OK)status=token_profile_begin(&capture,vk,work.position,err);
    uint64_t trace_gpu_begin=critical_ns();
    uint32_t result_bytes=0;if(status==FG_OK)status=fg_expert_decode(expert,&work,result,err);
    double tw_gpu=dispatch_ts();uint64_t trace_gpu_end=critical_ns();
    double tw_reduce=dispatch_ts();uint64_t trace_reduce=critical_ns();
    if(status==FG_OK)status=fg_expert_result_encode(wire,FG_EXPERT_RESULT_SINGLE_BYTES,&result_bytes,result,err);
    double tw_encode=dispatch_ts();uint64_t trace_encode=critical_ns(),trace_send_start=trace_encode;
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_EXPERT_RESULT,fg_frame_request_id(header),fg_frame_sequence(header),0,wire,result_bytes,err);
    double tw_send=dispatch_ts();uint64_t trace_send_end=critical_ns();
    status=token_profile_end(&capture,self,"routed_expert",work.position,work.layer,status,err);
    if(status==FG_OK&&((work.position>=26u&&work.position<28u)||token_profile_requested(work.position))){fprintf(stderr,"WORKER_EXPERT layer[%u] t=%u rank=%u sel=%u decode=%.2f gpu=%.2f reduce=%.2f encode=%.2f send=%.2f total=%.2f ready_ns=%llu header_end_ns=%llu recv_end_ns=%llu validate_end_ns=%llu decode_end_ns=%llu gpu_begin_ns=%llu gpu_end_ns=%llu reduce_end_ns=%llu encode_end_ns=%llu send_start_ns=%llu send_end_ns=%llu\n",work.layer,work.position,self,work.selected_count,tw_decode-tw0,tw_gpu-tw_decode,tw_reduce-tw_gpu,tw_encode-tw_reduce,tw_send-tw_encode,tw_send-tw0,(unsigned long long)(receive_timing?receive_timing->ready_ns:0),(unsigned long long)(receive_timing?receive_timing->header_end_ns:0),(unsigned long long)(receive_timing?receive_timing->payload_end_ns:0),(unsigned long long)(receive_timing?receive_timing->validate_end_ns:0),(unsigned long long)trace_decode,(unsigned long long)trace_gpu_begin,(unsigned long long)trace_gpu_end,(unsigned long long)trace_reduce,(unsigned long long)trace_encode,(unsigned long long)trace_send_start,(unsigned long long)trace_send_end);}
    return status;
}

typedef struct prefill_worker_buffers {
    uint8_t *receive,*activations,*result_wire,*pair_storage;
    fg_prefill_pair *pairs;
    fg_prefill_result_pair *result_pairs;
    float *outputs;
    bool pair_storage_owned,outputs_owned;
    uint32_t token_capacity,pair_capacity,receive_capacity,result_capacity;
} prefill_worker_buffers;

_Static_assert(sizeof(fg_prefill_pair)==FG_PREFILL_PAIR_BYTES,
               "prefill pair storage geometry must match its wire geometry");
_Static_assert(sizeof(fg_prefill_result_pair)==4u,
               "prefill result pair storage geometry must remain four bytes");
_Static_assert(sizeof(fg_prefill_pair)%_Alignof(fg_prefill_result_pair)==0u,
               "prefill result pair arena region must be naturally aligned");

static uint64_t prefill_pair_storage_bytes(uint32_t pair_capacity){
    return (uint64_t)pair_capacity*
        (sizeof(fg_prefill_pair)+sizeof(fg_prefill_result_pair));
}

static void prefill_worker_buffers_destroy(prefill_worker_buffers *buffers){if(!buffers)return;if(buffers->outputs_owned)free(buffers->outputs);if(buffers->pair_storage_owned)free(buffers->pair_storage);free(buffers->result_wire);free(buffers->activations);free(buffers->receive);memset(buffers,0,sizeof(*buffers));}

static fg_status prefill_worker_buffers_create(prefill_worker_buffers *buffers,uint32_t tokens,
                                               bool coordinator,fg_error *err){
    memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid manifest prefill microbatch");return FG_ERR_MISMATCH;}buffers->token_capacity=tokens;buffers->pair_capacity=tokens*FG_TOP_K;buffers->receive_capacity=FG_PREFILL_WORK_HEADER_BYTES+tokens*FG_Q8K_ACTIVATION_BYTES+buffers->pair_capacity*FG_PREFILL_PAIR_BYTES;buffers->result_capacity=FG_PREFILL_RESULT_HEADER_BYTES+buffers->pair_capacity*FG_PREFILL_RESULT_PAIR_BYTES+tokens*FG_HIDDEN_SIZE*4u;
    if(!coordinator)buffers->receive=malloc(buffers->receive_capacity);
    if(!coordinator)buffers->activations=malloc((size_t)tokens*FG_Q8K_ACTIVATION_BYTES);
    if(!coordinator)buffers->result_wire=malloc(buffers->result_capacity);
    uint64_t pair_storage_bytes=prefill_pair_storage_bytes(buffers->pair_capacity);
    if(pair_storage_bytes>SIZE_MAX){
        prefill_worker_buffers_destroy(buffers);
        fg_error_set(err,FG_ERR_LIMIT,"prefill pair storage exceeds address space");
        return FG_ERR_LIMIT;
    }
    buffers->pair_storage=malloc((size_t)pair_storage_bytes);
    if(buffers->pair_storage){
        size_t pair_bytes=(size_t)buffers->pair_capacity*sizeof(*buffers->pairs);
        buffers->pairs=(fg_prefill_pair *)buffers->pair_storage;
        buffers->result_pairs=(fg_prefill_result_pair *)(buffers->pair_storage+pair_bytes);
        buffers->pair_storage_owned=true;
    }
    buffers->outputs=malloc((size_t)buffers->pair_capacity*FG_HIDDEN_SIZE*
                            sizeof(*buffers->outputs));
    buffers->outputs_owned=true;
    if((!coordinator&&!buffers->result_wire)||!buffers->pair_storage||
       !buffers->outputs||
       (!coordinator&&(!buffers->receive||!buffers->activations||
                       !buffers->result_pairs))){
        prefill_worker_buffers_destroy(buffers);fg_error_set(err,FG_ERR_OOM,
            "allocate bounded prefill worker buffers");return FG_ERR_OOM;
    }
    return FG_OK;
}

static fg_status prefill_worker_buffers_ensure_result_wire(prefill_worker_buffers *buffers,
                                                           fg_error *err){
    if(!buffers)return FG_ERR_ARGUMENT;
    if(buffers->result_wire)return FG_OK;
    buffers->result_wire=malloc(buffers->result_capacity);
    if(!buffers->result_wire){
        fg_error_set(err,FG_ERR_OOM,"allocate coordinator prefill result wire");
        return FG_ERR_OOM;
    }
    return FG_OK;
}

static void prefill_worker_buffers_release_result_wire(prefill_worker_buffers *buffers){
    if(!buffers)return;
    free(buffers->result_wire);buffers->result_wire=NULL;
}

/* ---------------------------------------------------------------------------
 * Sealed single-owner worker prefill.  The worker owns every expert of the
 * layers in its block, so fire enqueues the complete routing batch into the
 * local expert executor and collect finishes it.  Mirrors the coordinator's
 * self-route path without the tree.
 * ------------------------------------------------------------------------- */
typedef struct worker_prefill_dispatch {
    fg_expert_executor *expert;
    const fg_manifest *manifest;
    prefill_worker_buffers *buffers;
    fg_prefill_work work;
    fg_prefill_result result;
    fg_prefill_route routes[FG_GROUP_SIZE];
    uint32_t route_count,self;
    bool enqueued;
} worker_prefill_dispatch;

static fg_status worker_prefill_fire(void *opaque,uint32_t layer,uint32_t first_token,
    uint16_t token_count,const uint16_t *expert_ids,const float *gates,
    const uint8_t *activations,fg_error *err){
    worker_prefill_dispatch *context=opaque;prefill_worker_buffers *buffers=context->buffers;
    context->enqueued=false;context->route_count=0;
    fg_status status=fg_partition_prefill_routes(context->manifest,layer,token_count,
        expert_ids,gates,context->routes,&context->route_count,buffers->pairs,
        buffers->pair_capacity,err);
    for(uint32_t r=0;status==FG_OK&&r<context->route_count;r++){
        if(context->routes[r].destination_rank!=context->self){
            fg_error_set(err,FG_ERR_MISMATCH,
                "worker prefill block layer %u routed to remote rank %u",layer,
                context->routes[r].destination_rank);
            return FG_ERR_MISMATCH;
        }
        context->work=(fg_prefill_work){.layer=(uint8_t)layer,
            .source_rank=(uint8_t)context->self,.destination_rank=(uint8_t)context->self,
            .first_position=first_token,.token_count=token_count,
            .pair_count=context->routes[r].pair_count,
            .activations_q8k=(uint8_t *)activations,.pairs=context->routes[r].pairs};
        status=fg_expert_prefill_enqueue(context->expert,&context->work,&context->result,
            buffers->result_pairs,buffers->pair_capacity,buffers->outputs,
            (uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE,err);
        if(status==FG_OK)context->enqueued=true;
    }
    return status;
}

static fg_status worker_prefill_collect(void *opaque,uint32_t layer,uint32_t first_token,
    uint16_t token_count,fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,
    fg_error *err){
    worker_prefill_dispatch *context=opaque;(void)layer;(void)first_token;(void)token_count;
    *result_count=0;
    if(!context->enqueued)return FG_OK;
    fg_status status=fg_expert_prefill_finish(context->expert,&context->work,&context->result,
        context->buffers->result_pairs,context->buffers->outputs,err);
    if(status==FG_OK){results[0]=context->result;*result_count=1u;}
    context->enqueued=false;
    return status;
}

/* ---------------------------------------------------------------------------
 * Sealed single-owner worker decode.  The worker owns every expert of the
 * layers in its block, so a routed layer dispatch partitions into exactly one
 * local route and completes synchronously, exactly like the coordinator's
 * local-route path in collect_experts.
 * ------------------------------------------------------------------------- */
typedef struct worker_decode_dispatch {
    fg_expert_executor *expert;
    const fg_manifest *manifest;
    uint32_t self;
    fg_expert_result *results;
    uint32_t result_count;
    bool expert_pending;
} worker_decode_dispatch;

static fg_status worker_decode_experts(void *opaque,uint32_t layer,uint32_t token,
    const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],
    const uint8_t *activation,fg_error *err){
    worker_decode_dispatch *context=opaque;
    fg_expert_route routes[FG_GROUP_SIZE];uint32_t route_count=0;
    context->result_count=0;
    fg_status status=fg_partition_route(context->manifest,layer,expert_ids,gates,
        routes,&route_count,err);
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){
        if(routes[r].destination_rank!=context->self){
            fg_error_set(err,FG_ERR_MISMATCH,
                "worker decode block layer %u routed to remote rank %u",layer,
                routes[r].destination_rank);
            return FG_ERR_MISMATCH;
        }
        fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,
            .destination_rank=(uint8_t)context->self,
            .selected_count=routes[r].selected_count,.position=token};
        for(uint32_t i=0;i<routes[r].selected_count;i++){
            work.expert_ids[i]=routes[r].global_expert_ids[i];
            work.routing_slots[i]=routes[r].routing_slots[i];
            work.gates[i]=routes[r].gates[i];
        }
        memcpy(work.activation_q8k,activation,FG_Q8K_ACTIVATION_BYTES);
        if(route_count==1u&&!context->expert_pending){
            /* Single-owner blocks (the ring layout) defer the readback until
             * collect, after the shared expert batch has synced. */
            status=fg_expert_decode_submit(context->expert,&work,
                &context->results[0],err);
            if(status==FG_OK){context->result_count=1u;context->expert_pending=true;}
        }else{
            status=fg_expert_decode(context->expert,&work,
                &context->results[context->result_count],err);
            if(status==FG_OK)context->result_count++;
        }
    }
    return status;
}

/* Local fire/collect pair for the owner async decode machine: the sealed block
 * owns every routed expert, so compute completes in fire and collect just
 * hands the canonical results to the shared reduce.  A deferred expert graph
 * is collected here once the shared expert batch has already synced. */
static fg_status worker_decode_fire(void *opaque,uint32_t layer,uint32_t token,
    const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],
    const uint8_t *activation,fg_error *err){
    return worker_decode_experts(opaque,layer,token,expert_ids,gates,activation,err);
}

/* Inline expert hook for chained ring blocks: routing stays on the GPU, so
 * there is no fire/collect pair and no per-layer expert fence. */
static fg_status chained_decode_expert(void *opaque,uint32_t layer,
    const fg_vk_tensor *activation,const fg_vk_tensor *router_logits,
    fg_vk_tensor **expert_output,fg_error *err){
    worker_decode_dispatch *context=opaque;
    if(!context||!context->expert){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"chained decode has no expert executor");
        return FG_ERR_UNAVAILABLE;
    }
    return fg_expert_decode_chain(context->expert,layer,activation,router_logits,
        expert_output,err);
}

/* Chaining needs every layer's whole routed expert slab on this rank (the
 * ring pack) and a fusable gate/up/down pair. */
static bool decode_block_chain_eligible(fg_expert_executor *expert,
    const fg_manifest *manifest,uint32_t rank,uint32_t first,uint32_t last){
    if(!expert)return false;
    for(uint32_t layer=first;layer<=last;layer++){
        if(fg_expert_local_count(manifest,layer,rank)!=FG_EXPERT_COUNT)return false;
        fg_error probe={0};
        if(fg_expert_decode_chain_ready(expert,layer,&probe)!=FG_OK)return false;
    }
    return true;
}

static fg_status worker_decode_collect(void *opaque,uint32_t layer,uint32_t token,
    fg_expert_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    (void)layer;(void)token;
    worker_decode_dispatch *context=opaque;
    fg_status status=FG_OK;
    if(context->expert_pending){
        status=fg_expert_decode_finish(context->expert,&context->results[0],err);
        context->expert_pending=false;
    }
    if(status==FG_OK){
        for(uint32_t i=0;i<context->result_count;i++)results[i]=context->results[i];
        *result_count=context->result_count;
    }else *result_count=0;
    context->result_count=0;
    return status;
}

typedef struct layer_work_context {
    uint8_t *work_wire,*result_wire;
    uint32_t work_capacity,result_capacity,tokens;
    uint32_t *positions;
    float *hyper_in,*hyper_out,*ngram;
    fg_vk_tensor *ngram_tensor;
    fg_vk_context *vk;
    worker_prefill_dispatch dispatch;
    worker_decode_dispatch decode_dispatch;
    fg_layer_work decode_work;
    fg_layer_result decode_result;
    void *qsa_owner;
    fg_output_hc *output_hc;
    uint32_t output_split_ways;
} layer_work_context;

static void layer_work_context_destroy(layer_work_context *context){
    if(!context)return;
    fg_vk_tensor_destroy(context->ngram_tensor);
    free(context->ngram);free(context->hyper_out);free(context->hyper_in);
    free(context->positions);free(context->result_wire);free(context->work_wire);
    memset(context,0,sizeof(*context));
}

static fg_status layer_work_context_create(layer_work_context *context,fg_model *model,
    fg_owner_executor *owner,const fg_manifest *manifest,fg_expert_executor *expert,
    prefill_worker_buffers *buffers,fg_error *err){
    memset(context,0,sizeof(*context));
    uint32_t tokens=manifest->prefill_microbatch;
    context->tokens=tokens;
    context->work_capacity=FG_PREFILL_LAYER_HEADER_BYTES+tokens*4u*4u+
        tokens*FG_HYPER_WIDTH*4u+tokens*FG_NGRAM_EMBED_VALUES*4u;
    context->result_capacity=FG_PREFILL_LAYER_RESULT_MAX_BYTES;
    context->work_wire=malloc(context->work_capacity);
    context->result_wire=malloc(context->result_capacity);
    context->positions=malloc((size_t)tokens*3u*sizeof(*context->positions));
    context->hyper_in=malloc((size_t)tokens*FG_HYPER_WIDTH*sizeof(float));
    context->hyper_out=malloc((size_t)tokens*FG_HYPER_WIDTH*sizeof(float));
    if(manifest->layer_owner[1u]==(uint8_t)fg_model_rank(model))
        context->ngram=malloc((size_t)tokens*FG_NGRAM_EMBED_VALUES*sizeof(float));
    if(!context->work_wire||!context->result_wire||!context->positions||
       !context->hyper_in||!context->hyper_out||
       (manifest->layer_owner[1u]==(uint8_t)fg_model_rank(model)&&!context->ngram)){
        layer_work_context_destroy(context);
        fg_error_set(err,FG_ERR_OOM,"allocate worker layer-work buffers");
        return FG_ERR_OOM;
    }
    if(manifest->layer_owner[1u]==(uint8_t)fg_model_rank(model)){
        fg_status status=fg_vk_tensor_create(fg_model_vk(model),
            (uint64_t)tokens*FG_NGRAM_EMBED_VALUES*4u,&context->ngram_tensor,err);
        if(status!=FG_OK){layer_work_context_destroy(context);return status;}
    }
    context->dispatch.expert=expert;context->dispatch.manifest=manifest;
    context->dispatch.buffers=buffers;context->dispatch.self=fg_model_rank(model);
    context->decode_dispatch.expert=expert;context->decode_dispatch.manifest=manifest;
    context->decode_dispatch.self=fg_model_rank(model);
    context->decode_dispatch.results=fg_owner_decode_results(owner);
    if(!context->decode_dispatch.results){
        layer_work_context_destroy(context);
        fg_error_set(err,FG_ERR_ARGUMENT,"worker decode result arena is unavailable");
        return FG_ERR_ARGUMENT;
    }
    context->vk=fg_model_vk(model);
    return FG_OK;
}

static fg_status worker_publish_qsa_pages(void *opaque,fg_owner_executor *owner,
    uint32_t self,uint32_t first_token,uint16_t token_count,fg_error *err);

/* Execute this rank's whole layer block for one chunk and hand the hyper state
 * to the next block owner (or back to rank 0 as the final result). */
static fg_status handle_prefill_layer_work(fg_fabric *fabric,fg_owner_executor *owner,
    const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    layer_work_context *context,fg_error *err){
    if(!owner){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"worker layer work requires an owner executor");
        return FG_ERR_UNAVAILABLE;
    }
    fg_prefill_layer_work work={0};
    fg_status status=fg_prefill_layer_work_decode(&work,manifest->protocol_version,
        context->positions,context->tokens*3u,context->hyper_in,
        (uint64_t)context->tokens*FG_HYPER_WIDTH,context->ngram,
        (uint64_t)context->tokens*FG_NGRAM_EMBED_VALUES,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work.source_rank||
       work.destination_rank!=self||!work.token_count||
       work.token_count>context->tokens||work.layer>=FG_LAYER_COUNT||
       manifest->layer_owner[work.layer]!=self||
       fg_frame_sequence(header)!=work.first_token*FG_LAYER_COUNT+work.layer)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted prefill layer work");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK&&work.layer>0u&&manifest->layer_owner[work.layer-1u]==self){
        fg_error_set(err,FG_ERR_MISMATCH,"prefill layer work does not start a block");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)numerics_trace_host("RECV_WORK",self,work.layer,work.first_token,
        work.token_count,context->hyper_in);
    uint32_t last=work.layer;
    while(status==FG_OK&&last+1u<FG_LAYER_COUNT&&manifest->layer_owner[last+1u]==self)last++;
    bool profiling=prefill_profile_requested();struct timespec t_begin={0};
    if(profiling)clock_gettime(CLOCK_MONOTONIC,&t_begin);
    fg_vk_tensor *current=status==FG_OK?fg_owner_prefill_input_slot(owner,0u):NULL;
    if(status==FG_OK&&!current){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"worker layer work prefill input is unavailable");
        status=FG_ERR_UNAVAILABLE;
    }
    if(status==FG_OK)status=fg_vk_tensor_write(current,0,work.hyper,
        (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
    for(uint32_t layer=work.layer;status==FG_OK&&layer<=last;layer++){
        struct timespec t_layer_begin={0},t_layer_end={0};
        if(profiling)clock_gettime(CLOCK_MONOTONIC,&t_layer_begin);
        const fg_vk_tensor *ngram=layer==1u?context->ngram_tensor:NULL;
        if(layer==1u){
            if(!(work.flags&FG_LAYER_WORK_HAS_NGRAM)||!context->ngram_tensor){
                fg_error_set(err,FG_ERR_MISMATCH,
                    "layer-1 worker work is missing n-gram embeddings");
                status=FG_ERR_MISMATCH;
                break;
            }
            status=fg_vk_tensor_write(context->ngram_tensor,0,work.ngram_embeddings,
                (uint64_t)work.token_count*FG_NGRAM_EMBED_VALUES*4u,err);
            if(status!=FG_OK)break;
        }
        bool capture=profiling&&context->vk&&!fg_vk_batch_active(context->vk);
        if(capture){fg_error profile_error={0};
            if(fg_vk_profile_begin(context->vk,&profile_error)!=FG_OK)capture=false;}
        status=fg_owner_prefill_layer_begin(owner,0u,layer,work.first_token,
            context->positions,(uint16_t)work.token_count,current,ngram,
            worker_prefill_fire,&context->dispatch,NULL,NULL,err);
        if(status==FG_OK)status=fg_owner_prefill_layer_finish(owner,0u,
            worker_prefill_collect,&context->dispatch,&current,err);
        if(profiling){clock_gettime(CLOCK_MONOTONIC,&t_layer_end);
            fprintf(stderr,"PREFILL_BLOCK_LAYER rank=%u layer=%u ms=%.1f\n",
                self,layer,elapsed_seconds(&t_layer_begin,&t_layer_end)*1000.0);}
        if(capture){fg_vk_profile layer_profile={0};fg_error profile_error={0};
            if(fg_vk_profile_end(context->vk,&layer_profile,
                    status==FG_OK?err:&profile_error)==FG_OK){
                fprintf(stderr,"PREFILL_LAYER_PROFILE rank=%u layer=%u gpu_ms=%.3f "
                    "kernel_ms=%.3f submissions=%llu dispatches=%llu\n",self,layer,
                    layer_profile.gpu_ms,layer_profile.kernel_ms,
                    (unsigned long long)layer_profile.submissions,
                    (unsigned long long)layer_profile.dispatches);
                for(uint32_t k=0;k<layer_profile.kernel_count;k++)
                    fprintf(stderr,"PREFILL_LAYER_KERNEL rank=%u layer=%u scope=%s "
                        "kernel=%s calls=%llu gpu_ms=%.3f\n",self,layer,
                        layer_profile.kernels[k].scope,layer_profile.kernels[k].name,
                        (unsigned long long)layer_profile.kernels[k].invocations,
                        layer_profile.kernels[k].gpu_ms);
            }
        }
    }
    if(status==FG_OK)status=worker_publish_qsa_pages(context->qsa_owner,owner,self,
        work.first_token,work.token_count,err);
    if(status==FG_OK)status=fg_vk_tensor_read(current,0,context->hyper_out,
        (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
    if(status==FG_OK)numerics_trace_host(last+1u<FG_LAYER_COUNT?"SEND_NEXT":"SEND_RESULT",
        self,last,work.first_token,work.token_count,context->hyper_out);
    if(status==FG_OK&&last+1u<FG_LAYER_COUNT){
        uint32_t next_layer=last+1u,next_rank=manifest->layer_owner[next_layer];
        fg_prefill_layer_work next={.layer=(uint8_t)next_layer,.source_rank=(uint8_t)self,
            .destination_rank=(uint8_t)next_rank,.first_token=work.first_token,
            .token_count=work.token_count,.positions=context->positions,
            .hyper=context->hyper_out};
        uint32_t wire_bytes=0;
        status=fg_prefill_layer_work_encode(context->work_wire,context->work_capacity,
            &wire_bytes,manifest->protocol_version,&next,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,next_rank,FG_FABRIC_BULK,
            FG_MSG_PREFILL_LAYER_WORK,request,
            next_layer+work.first_token*FG_LAYER_COUNT,0,context->work_wire,wire_bytes,err);
    }else if(status==FG_OK){
        fg_prefill_layer_result result={.layer=(uint8_t)last,.source_rank=(uint8_t)self,
            .destination_rank=0u,.first_token=work.first_token,.token_count=work.token_count,
            .hyper=context->hyper_out};
        uint32_t wire_bytes=0;
        status=fg_prefill_layer_result_encode(context->result_wire,context->result_capacity,
            &wire_bytes,&result,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
            FG_MSG_PREFILL_LAYER_RESULT,request,
            last+work.first_token*FG_LAYER_COUNT,0,context->result_wire,wire_bytes,err);
    }
    if(profiling){struct timespec t_end;clock_gettime(CLOCK_MONOTONIC,&t_end);
        fprintf(stderr,"PREFILL_LAYER_WORK rank=%u layer=%u last=%u tokens=%u forward=%u total_ms=%.3f\n",
            self,(unsigned)work.layer,last,(unsigned)work.token_count,
            last+1u<FG_LAYER_COUNT?1u:0u,elapsed_seconds(&t_begin,&t_end)*1000.0);}
    return status;
}

/* Execute this rank's whole layer block for one decode token and hand the hyper
 * state to the next block owner (or back to rank 0 as the final result).  The
 * block owner's own GDN/QSA/PLE state was advanced by ring prefill, so no
 * state is shipped: only the 40 KiB hyper vector moves. */
static fg_status handle_decode_layer_work(fg_fabric *fabric,fg_owner_executor *owner,
    const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    layer_work_context *context,fg_error *err){
    if(!owner){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"worker decode work requires an owner executor");
        return FG_ERR_UNAVAILABLE;
    }
    fg_layer_work *work=&context->decode_work;
    fg_status status=fg_decode_layer_work_decode(work,manifest->protocol_version,
        payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    bool has_ngram=(work->flags&FG_LAYER_WORK_HAS_NGRAM)!=0u;
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work->source_rank||
       work->destination_rank!=self||work->layer>=FG_LAYER_COUNT||
       manifest->layer_owner[work->layer]!=self||
       (work->layer>0u&&manifest->layer_owner[work->layer-1u]==self)||
       ((work->layer<=1u)!=has_ngram)||
       fg_frame_sequence(header)!=work->token_index*FG_LAYER_COUNT+work->layer)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted decode layer work");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK&&has_ngram&&!context->ngram_tensor){
        fg_error_set(err,FG_ERR_MISMATCH,"rank %u has no decode n-gram storage",self);
        status=FG_ERR_MISMATCH;
    }
    bool trace=decode_ring_trace_enabled();struct timespec t_begin={0};
    if(trace)clock_gettime(CLOCK_MONOTONIC,&t_begin);
    fg_vk_tensor *input=status==FG_OK?fg_owner_prefill_input(owner):NULL;
    if(status==FG_OK&&!input){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"worker decode input storage is unavailable");
        status=FG_ERR_UNAVAILABLE;
    }
    if(status==FG_OK)status=fg_vk_tensor_write(input,0,work->hyper,
        (uint64_t)FG_HYPER_WIDTH*4u,err);
    struct timespec t_written={0};if(trace)clock_gettime(CLOCK_MONOTONIC,&t_written);
    numerics_trace_host("FB_IN",self,work->layer,work->token_index,1u,work->hyper);
    if(status==FG_OK&&has_ngram)status=fg_vk_tensor_write(context->ngram_tensor,0,
        work->ngram_embedding,(uint64_t)FG_NGRAM_EMBED_VALUES*4u,err);
    if(status==FG_OK&&numerics_trace_enabled()&&work->layer==0u){
        fg_vk_tensor *ple=fg_owner_ple_state_tensor(owner);
        float *ple_host=ple?malloc(FG_GDN_STATE_PLE_BYTES):NULL;
        if(ple_host){
            if(fg_vk_tensor_read(ple,0,ple_host,FG_GDN_STATE_PLE_BYTES,err)==FG_OK)
                numerics_trace_values("PLE_STATE_LOCAL",self,1u,work->token_index,
                    ple_host,(uint64_t)FG_HYPER_WIDTH*9u);
            free(ple_host);
        }
        for(uint32_t dl=0u;dl<=1u;dl++){
            fg_vk_tensor *conv=fg_owner_gdn_state_tensor(owner,dl,0u);
            fg_vk_tensor *recurrent=fg_owner_gdn_state_tensor(owner,dl,1u);
            if(conv){
                const float *map=fg_vk_tensor_map(conv);
                for(uint32_t q=0;q<4u;q++)
                    numerics_trace_values("STATE_LOCAL_CONV",self,dl,work->token_index,
                        map+(uint64_t)q*FG_HYPER_WIDTH,(uint64_t)FG_HYPER_WIDTH);
            }
            if(recurrent)numerics_trace_values("STATE_LOCAL_RECUR",self,dl,work->token_index,
                fg_vk_tensor_map(recurrent),(uint64_t)48u*128u*128u);
        }
    }
    uint32_t last=work->layer;
    while(status==FG_OK&&last+1u<FG_LAYER_COUNT&&manifest->layer_owner[last+1u]==self)last++;
    fg_vk_tensor *current=NULL;
    bool decode_profile=decode_profile_enabled();
    if(status==FG_OK&&decode_profile){
        fg_error profile_error={0};
        if(fg_vk_profile_begin(context->vk,&profile_error)!=FG_OK)decode_profile=false;
    }
    if(status==FG_OK&&decode_block_chain_eligible(context->decode_dispatch.expert,
        manifest,self,work->layer,last)){
        status=fg_owner_decode_block_chained(owner,work->layer,last,work->token_index,
            work->position,input,has_ngram?context->ngram_tensor:NULL,
            chained_decode_expert,&context->decode_dispatch,&current,err);
    }else if(status==FG_OK)status=fg_owner_decode_block(owner,work->layer,last,work->token_index,
        work->position,input,has_ngram?context->ngram_tensor:NULL,worker_decode_fire,
        worker_decode_collect,&context->decode_dispatch,&current,err);
    if(decode_profile){
        fg_vk_profile profile={0};fg_error profile_error={0};
        fg_status profile_status=fg_vk_profile_end(context->vk,&profile,
            status==FG_OK?err:&profile_error);
        if(profile_status==FG_OK){
            fprintf(stderr,"DECODE_PROFILE rank=%u token=%u layers=%u..%u gpu_ms=%.3f "
                "kernel_ms=%.3f wait_ms=%.3f record_ms=%.3f op_ms=%.3f barrier_ms=%.3f "
                "submit_ms=%.3f gap_ms=%.3f gap_max_ms=%.3f gap_max=%s submissions=%llu dispatches=%llu\n",self,
                work->token_index,(unsigned)work->layer,last,profile.gpu_ms,
                profile.kernel_ms,profile.wait_ms,profile.record_ms,profile.op_ms,
                profile.barrier_ms,profile.submit_ms,profile.gap_ms,profile.gap_max_ms,
                profile.gap_max_kernel?profile.gap_max_kernel:"-",
                (unsigned long long)profile.submissions,
                (unsigned long long)profile.dispatches);
            for(uint32_t k=0;k<profile.kernel_count;k++)
                fprintf(stderr,"DECODE_PROFILE_KERNEL rank=%u scope=%s kernel=%s "
                    "calls=%llu gpu_ms=%.3f\n",self,profile.kernels[k].scope,
                    profile.kernels[k].name,
                    (unsigned long long)profile.kernels[k].invocations,
                    profile.kernels[k].gpu_ms);
        }
    }
    struct timespec t_block={0};if(trace)clock_gettime(CLOCK_MONOTONIC,&t_block);
    if(status==FG_OK)status=fg_vk_tensor_read(current,0,context->hyper_out,
        (uint64_t)FG_HYPER_WIDTH*4u,err);
    struct timespec t_read={0};if(trace)clock_gettime(CLOCK_MONOTONIC,&t_read);
    numerics_trace_host("FB_OUT",self,last,work->token_index,1u,context->hyper_out);
    if(status==FG_OK)status=worker_publish_qsa_pages(context->qsa_owner,owner,self,
        work->token_index,1u,err);
    struct timespec t_publish={0};if(trace)clock_gettime(CLOCK_MONOTONIC,&t_publish);
    if(status==FG_OK&&last+1u<FG_LAYER_COUNT){
        fg_layer_work next={.layer=(uint8_t)(last+1u),.source_rank=(uint8_t)self,
            .destination_rank=manifest->layer_owner[last+1u],
            .token_index=work->token_index,.position_mode=FG_POSITION_TEXT,
            .flags=(uint8_t)(work->flags&FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)};
        for(uint32_t axis=0;axis<3u;axis++)next.position[axis]=work->position[axis];
        memcpy(next.hyper,context->hyper_out,(uint64_t)FG_HYPER_WIDTH*4u);
        uint32_t wire_bytes=0;
        status=fg_decode_layer_work_encode(context->work_wire,context->work_capacity,
            &wire_bytes,manifest->protocol_version,&next,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,next.destination_rank,FG_FABRIC_BULK,
            FG_MSG_DECODE_LAYER_WORK,request,
            work->token_index*FG_LAYER_COUNT+next.layer,0,context->work_wire,wire_bytes,err);
    }else if(status==FG_OK){
        uint32_t output_owner=fg_output_owner_rank(manifest);
        bool direct=decode_direct_output_eligible(manifest)&&output_owner!=self;
        bool skip_hidden=direct&&context->output_split_ways==FG_OUTPUT_SPLIT_WAYS_MAX&&
            (work->flags&FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)!=0u;
        fg_layer_result *result=&context->decode_result;
        memset(result,0,sizeof(*result));
        result->layer=(uint8_t)last;result->source_rank=(uint8_t)self;
        result->destination_rank=(uint8_t)(direct?output_owner:0u);result->token_index=work->token_index;
        memcpy(result->hyper,context->hyper_out,sizeof(result->hyper));
        if(skip_hidden)output_split_trace("hidden-suppressed",self,work->token_index,
            context->output_split_ways);
        if(!skip_hidden){
            uint32_t result_bytes=0u;
            status=fg_decode_layer_result_encode(context->result_wire,result,&result_bytes,err);
            if(status==FG_OK)status=fg_fabric_send(fabric,direct?output_owner:0u,FG_FABRIC_BULK,
                direct?FG_MSG_OUTPUT_HIDDEN:FG_MSG_DECODE_LAYER_RESULT,request,
                work->token_index*FG_LAYER_COUNT+last,0,context->result_wire,
                result_bytes,err);
        }
        if(status==FG_OK&&direct&&context->output_split_ways==FG_OUTPUT_SPLIT_WAYS_MIN){
            uint32_t slice_bytes=0u;
            status=fg_output_slice_encode(context->result_wire,result,&slice_bytes,err);
            if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
                FG_MSG_OUTPUT_SLICE,request,
                work->token_index*FG_LAYER_COUNT+last,0,context->result_wire,
                slice_bytes,err);
        }else if(status==FG_OK&&direct&&context->output_split_ways==FG_OUTPUT_SPLIT_WAYS_MAX){
            float hidden[FG_HIDDEN_SIZE];
            if(!context->output_hc){
                fg_error_set(err,FG_ERR_MISMATCH,
                    "rank %u is the final block owner without a 4-way output HC executor",self);
                status=FG_ERR_MISMATCH;
            }else status=fg_output_hc_run(context->output_hc,result->hyper,hidden,err);
            if(status==FG_OK)output_split_trace("hc",self,work->token_index,0u);
            for(uint32_t way=0;status==FG_OK&&way<FG_OUTPUT_SPLIT_WAYS_MAX;way++){
                uint32_t destination=fg_output_split_rank(context->output_split_ways,way);
                if(destination==UINT32_MAX||destination==self)continue;
                fg_output_slice_hidden slice={.source_rank=(uint8_t)self,
                    .destination_rank=(uint8_t)destination,.token_index=work->token_index};
                memcpy(slice.hidden,hidden,sizeof(slice.hidden));
                uint32_t hidden_bytes=0u;
                status=fg_output_slice_hidden_encode(context->result_wire,&slice,&hidden_bytes,err);
                if(status==FG_OK)status=fg_fabric_send(fabric,destination,FG_FABRIC_BULK,
                    FG_MSG_OUTPUT_SLICE_HIDDEN,request,
                    work->token_index*FG_LAYER_COUNT+last,0,context->result_wire,
                    hidden_bytes,err);
            }
        }
        if(direct&&trace&&status==FG_OK)
            fprintf(stderr,"RING_DECODE_HANDOFF rank=%u token=%u output_rank=%u\n",
                self,work->token_index,output_owner);
    }
    if(trace&&status==FG_OK){struct timespec t_end;clock_gettime(CLOCK_MONOTONIC,&t_end);
        fprintf(stderr,"RING_DECODE_BLOCK rank=%u token=%u layers=%u..%u write_ms=%.3f "
            "block_ms=%.3f read_ms=%.3f publish_ms=%.3f egress_ms=%.3f total_ms=%.3f\n",
            self,work->token_index,(unsigned)work->layer,last,
            elapsed_seconds(&t_begin,&t_written)*1000.0,
            elapsed_seconds(&t_written,&t_block)*1000.0,
            elapsed_seconds(&t_block,&t_read)*1000.0,
            elapsed_seconds(&t_read,&t_publish)*1000.0,
            elapsed_seconds(&t_publish,&t_end)*1000.0,
            elapsed_seconds(&t_begin,&t_end)*1000.0);}
    return status;
}

/* Ring prefill leaves each block owner's GDN/PLE state at the prefill frontier
 * while rank 0 decodes all 48 layers locally.  Serve rank 0 one layer's state
 * on demand so its decode executor starts from the distributed frontier. */
static fg_status handle_gdn_state_fetch(fg_fabric *fabric,fg_owner_executor *owner,
    const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_error *err){
    if(!owner||peer!=0u||fg_frame_request_id(header)!=session_id){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted GDN state fetch");
        return FG_ERR_MISMATCH;
    }
    fg_gdn_state_fetch fetch={0};
    fg_status status=fg_gdn_state_fetch_decode(&fetch,payload,bytes,err);
    if(status==FG_OK&&(fetch.layer>=FG_LAYER_COUNT||(fetch.layer&3u)==3u||
       manifest->layer_owner[fetch.layer]!=self)){
        fg_error_set(err,FG_ERR_MISMATCH,"GDN state fetch layer is not owned by rank %u",self);
        status=FG_ERR_MISMATCH;
    }
    uint8_t *wire=status==FG_OK?malloc(FG_GDN_STATE_RESULT_MAX_BYTES):NULL;
    if(status==FG_OK&&!wire){fg_error_set(err,FG_ERR_OOM,"allocate GDN state result wire");status=FG_ERR_OOM;}
    if(status==FG_OK){
        fg_vk_tensor *conv=fg_owner_gdn_state_tensor(owner,fetch.layer,0u);
        fg_vk_tensor *recurrent=fg_owner_gdn_state_tensor(owner,fetch.layer,1u);
        if(!conv||!recurrent){
            fg_error_set(err,FG_ERR_MISMATCH,"rank %u has no GDN state for layer %u",self,fetch.layer);
            status=FG_ERR_MISMATCH;
        }else{
            fg_vk_tensor *ple=fetch.layer==1u?fg_owner_ple_state_tensor(owner):NULL;
            fg_gdn_state_result result={.source_rank=(uint8_t)self,.layer=fetch.layer,
                .frontier=fetch.frontier,.conv=fg_vk_tensor_map(conv),
                .recurrent=fg_vk_tensor_map(recurrent),
                .ple=ple?fg_vk_tensor_map(ple):NULL};
            if(fetch.layer<=1u&&numerics_trace_enabled()){
                if(fetch.layer==0u)
                    fprintf(stderr,"FG_NUMERICS_SIZE rank=%u conv_bytes=%llu recurrent_bytes=%llu conv_const=%u\n",
                        self,(unsigned long long)fg_vk_tensor_bytes(conv),
                        (unsigned long long)fg_vk_tensor_bytes(recurrent),
                        (unsigned)FG_GDN_STATE_CONV_BYTES);
                const float *conv_map=fg_vk_tensor_map(conv);
                for(uint32_t q=0;q<4u;q++)
                    numerics_trace_values("STATE_SEND_CONV",self,(uint32_t)fetch.layer,fetch.frontier,
                        conv_map+(uint64_t)q*FG_HYPER_WIDTH,(uint64_t)FG_HYPER_WIDTH);
                numerics_trace_values("STATE_SEND_RECUR",self,(uint32_t)fetch.layer,fetch.frontier,
                    result.recurrent,(uint64_t)48u*128u*128u);
            }
            uint32_t result_bytes=0;
            status=fg_gdn_state_result_encode(wire,FG_GDN_STATE_RESULT_MAX_BYTES,
                &result_bytes,&result,err);
            if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
                FG_MSG_GDN_STATE_RESULT,fg_frame_request_id(header),fetch.layer,0,
                wire,result_bytes,err);
        }
    }
    free(wire);
    return status;
}

static bool block_bench_requested(void){
    const char *value=getenv("FG_BLOCK_BENCH");
    return value&&*value&&strcmp(value,"0")!=0;
}

static bool worker_owner_enabled(void){
    const char *value=getenv("FG_WORKER_OWNER");
    return value&&*value&&strcmp(value,"0")!=0;
}

/* Offline block-service bench: run this rank's owned GDN layers over a sealed
 * microbatch and report per-layer and per-block wall time.  This is the
 * measurement gate for the layer-ring throughput model. */
static fg_status run_block_bench(fg_model *model,fg_expert_executor *expert,
    fg_owner_executor *owner,const fg_manifest *manifest,fg_error *err){
    uint32_t rank=fg_model_rank(model),tokens=manifest->prefill_microbatch;
    if(!owner){fg_error_set(err,FG_ERR_ARGUMENT,"block bench requires an owner executor");return FG_ERR_ARGUMENT;}
    prefill_worker_buffers buffers={0};
    fg_status status=prefill_worker_buffers_create(&buffers,tokens,false,err);
    uint32_t owned[FG_LAYER_COUNT],owned_count=0,qsa_skipped=0;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        if(!fg_owner_owns_layer(owner,layer))continue;
        if((layer&3u)==3u){qsa_skipped++;continue;}
        owned[owned_count++]=layer;
    }
    if(status==FG_OK&&!owned_count){
        fg_error_set(err,FG_ERR_MISMATCH,"block bench rank %u owns no GDN layers",rank);
        status=FG_ERR_MISMATCH;
    }
    fg_vk_tensor *input=status==FG_OK?fg_owner_prefill_input_slot(owner,0u):NULL;
    if(status==FG_OK&&!input){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"block bench prefill input is unavailable");
        status=FG_ERR_UNAVAILABLE;
    }
    uint32_t *positions=status==FG_OK?malloc((size_t)tokens*3u*4u):NULL;
    fg_vk_tensor *ngram_tensor=NULL;
    if(status==FG_OK&&!positions){fg_error_set(err,FG_ERR_OOM,"block bench positions");status=FG_ERR_OOM;}
    if(status==FG_OK&&manifest->layer_owner[1u]==(uint8_t)rank)
        status=fg_vk_tensor_create(fg_model_vk(model),
            (uint64_t)tokens*FG_NGRAM_EMBED_VALUES*4u,&ngram_tensor,err);
    if(status==FG_OK){
        float *mapped=fg_vk_tensor_map(input);
        for(uint32_t i=0;i<tokens*FG_HYPER_WIDTH;i++)
            mapped[i]=(float)((int32_t)(i%37u)-18)*(1.0f/64.0f);
        for(uint32_t i=0;i<tokens;i++)
            for(uint32_t axis=0;axis<3u;axis++)positions[i*3u+axis]=1024u+i;
        if(ngram_tensor)memset(fg_vk_tensor_map(ngram_tensor),0,
            (size_t)tokens*FG_NGRAM_EMBED_VALUES*4u);
    }
    worker_prefill_dispatch dispatch={.expert=expert,.manifest=manifest,
        .buffers=&buffers,.self=rank};
    const uint32_t measured=5u;
    double layer_ms[FG_LAYER_COUNT]={0};
    double block_sum=0.0,block_min=0.0,block_max=0.0;
    if(status==FG_OK)fprintf(stderr,
        "BLOCK_BENCH rank=%u tokens=%u gdn_layers=%u qsa_skipped=%u ngram=%s\n",
        rank,tokens,owned_count,qsa_skipped,ngram_tensor?"attached":"absent");
    for(uint32_t iteration=0;status==FG_OK&&iteration<=measured;iteration++){
        fg_vk_tensor *current=input;
        struct timespec block_start;clock_gettime(CLOCK_MONOTONIC,&block_start);
        for(uint32_t index=0;status==FG_OK&&index<owned_count;index++){
            uint32_t layer=owned[index];
            const fg_vk_tensor *ngram=layer==1u?ngram_tensor:NULL;
            struct timespec layer_start,layer_end;
            clock_gettime(CLOCK_MONOTONIC,&layer_start);
            status=fg_owner_prefill_layer_begin(owner,0u,layer,1024u,positions,
                (uint16_t)tokens,current,ngram,worker_prefill_fire,&dispatch,NULL,NULL,err);
            if(status==FG_OK)status=fg_owner_prefill_layer_finish(owner,0u,
                worker_prefill_collect,&dispatch,&current,err);
            clock_gettime(CLOCK_MONOTONIC,&layer_end);
            if(status==FG_OK&&iteration>0u)
                layer_ms[layer]+=elapsed_seconds(&layer_start,&layer_end)*1000.0;
        }
        struct timespec block_end;clock_gettime(CLOCK_MONOTONIC,&block_end);
        if(status==FG_OK&&iteration>0u){
            double block_ms=elapsed_seconds(&block_start,&block_end)*1000.0;
            block_sum+=block_ms;
            if(block_min==0.0||block_ms<block_min)block_min=block_ms;
            if(block_ms>block_max)block_max=block_ms;
            fprintf(stderr,"BLOCK_BENCH_ITER rank=%u iteration=%u block_ms=%.2f\n",
                rank,iteration-1u,block_ms);
        }
    }
    for(uint32_t index=0;index<owned_count;index++)
        fprintf(stderr,"BLOCK_BENCH_LAYER rank=%u layer=%u mean_ms=%.2f\n",
            rank,owned[index],layer_ms[owned[index]]/(double)measured);
    if(status==FG_OK)fprintf(stderr,
        "BLOCK_BENCH_SUMMARY rank=%u tokens=%u layers=%u measured=%u "
        "mean_block_ms=%.2f min_block_ms=%.2f max_block_ms=%.2f\n",
        rank,tokens,owned_count,measured,block_sum/(double)measured,block_min,block_max);
    fg_vk_tensor_destroy(ngram_tensor);
    free(positions);
    prefill_worker_buffers_destroy(&buffers);
    return status;
}

/* Active remote ranks form a binary heap in ascending rank order. The
 * coordinator is outside this tree even when it owns some selected experts. */
static fg_status prefill_tree_ranks(const fg_manifest *manifest,const fg_prefill_work *work,
    uint8_t ranks[FG_GROUP_SIZE],uint32_t *count,fg_error *err){
    uint8_t mask=0;*count=0;
    if(work->pair_count!=(uint32_t)work->token_count*FG_TOP_K){
        fg_error_set(err,FG_ERR_MISMATCH,"broadcast requires the complete routing list");return FG_ERR_MISMATCH;
    }
    for(uint32_t i=0;i<work->pair_count;i++){
        uint32_t rank=manifest->expert_rank[work->layer][work->pairs[i].expert_id];
        if(rank>=FG_RANK_COUNT||!fg_topology_rank_in_layer(manifest,work->layer,rank)){
            fg_error_set(err,FG_ERR_MISMATCH,"broadcast route outside layer group");return FG_ERR_MISMATCH;
        }
        if(rank!=work->source_rank)mask|=(uint8_t)(1u<<rank);
    }
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(mask&(1u<<rank)){
        if(*count==FG_GROUP_SIZE){fg_error_set(err,FG_ERR_MISMATCH,"too many broadcast participants");return FG_ERR_MISMATCH;}
        ranks[(*count)++]=(uint8_t)rank;
    }
    return FG_OK;
}

static uint8_t prefill_subtree_mask(const uint8_t *ranks,uint32_t count,uint32_t node){
    if(node>=count)return 0;
    return (uint8_t)((1u<<ranks[node])|prefill_subtree_mask(ranks,count,node*2u+1u)|
                     prefill_subtree_mask(ranks,count,node*2u+2u));
}

/* Both arrays are disjoint slices of the cached malloc-backed receive arena,
 * not Vulkan mappings. Adding here avoids uploading and reading back each tile. */
static void prefill_merge(float *restrict sum,const float *restrict child,uint32_t tokens){
    for(uint32_t i=0;i<tokens*FG_HIDDEN_SIZE;i++)sum[i]+=child[i];
}

static fg_status handle_prefill_expert_work(fg_fabric *fabric,fg_expert_executor *expert,
    const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    prefill_worker_buffers *buffers,fg_error *err){
    bool prof=prefill_profile_requested();struct timespec t_begin={0},t_decoded={0},t_forward={0},t_compute={0},t_send={0};if(prof)clock_gettime(CLOCK_MONOTONIC,&t_begin);
    fg_prefill_work work={0};
    fg_status status=fg_prefill_work_decode(&work,buffers->activations,
        buffers->token_capacity*FG_Q8K_ACTIVATION_BYTES,buffers->pairs,
        buffers->pair_capacity,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);uint8_t ranks[FG_GROUP_SIZE];uint32_t count=0,node=0;
    if(status==FG_OK&&(!session_id||request!=session_id||work.destination_rank!=self||
       (work.source_rank!=0u&&manifest->layer_owner[work.layer]!=work.source_rank)||
       fg_frame_sequence(header)!=work.first_position*FG_LAYER_COUNT+work.layer)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted prefill broadcast");status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=prefill_tree_ranks(manifest,&work,ranks,&count,err);
    if(status==FG_OK){
        while(node<count&&ranks[node]!=self)node++;
        uint32_t parent=node==0u?work.source_rank:ranks[(node-1u)/2u];
        if(node==count||peer!=parent){fg_error_set(err,FG_ERR_MISMATCH,"prefill broadcast from wrong parent");status=FG_ERR_MISMATCH;}
    }
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_decoded);
    /* Forward before GPU execution. All children receive the same immutable
     * activation block and route list; only the destination header changes. */
    for(uint32_t child=node*2u+1u;status==FG_OK&&child<count&&child<=node*2u+2u;child++){
        work.destination_rank=ranks[child];uint32_t work_bytes=0;
        status=fg_prefill_work_encode(buffers->receive,buffers->receive_capacity,&work_bytes,&work,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,ranks[child],FG_FABRIC_BULK,
            FG_MSG_PREFILL_WORK,request,fg_frame_sequence(header),0,buffers->receive,work_bytes,err);
    }
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_forward);
    work.destination_rank=(uint8_t)self;
    fg_prefill_pair local_pairs[FG_PREFILL_MAX_PAIRS];uint32_t local_count=0;
    if(status==FG_OK)for(uint32_t i=0;i<work.pair_count;i++)
        if(manifest->expert_rank[work.layer][work.pairs[i].expert_id]==self)local_pairs[local_count++]=work.pairs[i];
    fg_prefill_work local=work;local.pairs=local_pairs;local.pair_count=(uint16_t)local_count;
    fg_prefill_result result={0};
    if(status==FG_OK)status=fg_expert_prefill(expert,&local,&result,buffers->result_pairs,
        buffers->pair_capacity,buffers->outputs,(uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE,err);
    if(status==FG_OK)status=fg_prefill_result_validate_subset(manifest,&work,
        (uint8_t)(1u<<self),&result,err);
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_compute);
    uint8_t mask=prefill_subtree_mask(ranks,count,node);
    /* Forward each reduced tile immediately. Full-batch store-and-forward
     * multiplies serialization latency by the depth of the tree. */
    for(uint32_t first=0;status==FG_OK&&first<work.token_count;first+=FG_PREFILL_REDUCE_TILE_TOKENS){
        uint16_t tokens=(uint16_t)(work.token_count-first);
        if(tokens>FG_PREFILL_REDUCE_TILE_TOKENS)tokens=FG_PREFILL_REDUCE_TILE_TOKENS;
        float *sum=result.outputs+(uint64_t)first*FG_HIDDEN_SIZE;
        for(uint32_t child=node*2u+1u;status==FG_OK&&child<count&&child<=node*2u+2u;child++){
            fg_frame_header reply;uint32_t reply_bytes=0;
            status=fg_fabric_recv(fabric,ranks[child],FG_FABRIC_BULK,&reply,
                buffers->result_wire,buffers->result_capacity,&reply_bytes,err);
            if(status==FG_OK&&(fg_frame_type(&reply)!=FG_MSG_PREFILL_RESULT||
               fg_frame_request_id(&reply)!=request||fg_frame_sequence(&reply)!=fg_frame_sequence(header))){
                fg_error_set(err,FG_ERR_MISMATCH,"stale prefill subtree frame");status=FG_ERR_MISMATCH;
            }
            fg_prefill_result incoming={0};
            if(status==FG_OK)status=fg_prefill_result_decode(&incoming,
                buffers->result_pairs,buffers->pair_capacity,
                buffers->outputs+(uint64_t)work.token_count*FG_HIDDEN_SIZE,
                (uint64_t)(buffers->pair_capacity-work.token_count)*FG_HIDDEN_SIZE,
                buffers->result_wire,reply_bytes,err);
            if(status==FG_OK&&(incoming.source_rank!=ranks[child]||
               incoming.first_position!=work.first_position+first||incoming.token_count!=tokens)){
                fg_error_set(err,FG_ERR_MISMATCH,"wrong prefill subtree sender or tile");status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK)status=fg_prefill_result_validate_subset(manifest,&work,
                prefill_subtree_mask(ranks,count,child),&incoming,err);
            if(status==FG_OK)prefill_merge(sum,incoming.outputs,tokens);
        }
        fg_prefill_result tile={.layer=work.layer,.source_rank=(uint8_t)self,
            .destination_rank=work.source_rank,.contributor_mask=mask,
            .first_position=work.first_position+first,.token_count=tokens,
            .pairs=buffers->result_pairs,.outputs=sum};
        /* Child coverage was checked before it entered the sum. Reconstruct
         * this tile's union from the verified original routing list. */
        for(uint32_t i=0;status==FG_OK&&i<work.pair_count;i++){
            const fg_prefill_pair *pair=&work.pairs[i];
            uint32_t rank=manifest->expert_rank[work.layer][pair->expert_id];
            if((mask&(1u<<rank))&&pair->token_slot>=first&&pair->token_slot<first+tokens)
                tile.pairs[tile.pair_count++]=(fg_prefill_result_pair){(uint16_t)(pair->token_slot-first),pair->routing_slot};
        }
        uint32_t result_bytes=0;
        if(status==FG_OK)status=fg_prefill_result_encode(buffers->result_wire,buffers->result_capacity,&result_bytes,&tile,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_PREFILL_RESULT,
            request,fg_frame_sequence(header),0,buffers->result_wire,result_bytes,err);
    }
    /* On any error the worker loop exits and closes every fabric channel.
     * Parents/children cannot mistake unfinished work for a later request. */
    if(prof){clock_gettime(CLOCK_MONOTONIC,&t_send);fprintf(stderr,"PREFILL_WORKER rank=%u layer=%u tokens=%u decode_ms=%.3f forward_ms=%.3f compute_ms=%.3f send_ms=%.3f total_ms=%.3f\n",self,(unsigned)work.layer,(unsigned)work.token_count,elapsed_seconds(&t_begin,&t_decoded)*1000.0,elapsed_seconds(&t_decoded,&t_forward)*1000.0,elapsed_seconds(&t_forward,&t_compute)*1000.0,elapsed_seconds(&t_compute,&t_send)*1000.0,elapsed_seconds(&t_begin,&t_send)*1000.0);}
    return status;
}

typedef struct prefill_dispatch_context {fg_fabric *fabric;fg_expert_executor *expert;const fg_manifest *manifest;uint32_t self;uint64_t request_id;uint32_t sequence;prefill_worker_buffers *buffers;atomic_uint *transport_state;} prefill_dispatch_context;

static fg_status dispatch_prefill_experts(void *opaque,uint32_t layer,
    uint32_t first_token,uint16_t token_count,const uint16_t *expert_ids,
    const float *gates,const uint8_t *activations,
    fg_owner_prefill_shared_fn shared_work,void *shared_context,
    fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    prefill_dispatch_context *context=opaque;prefill_worker_buffers *buffers=context->buffers;
    bool prof=prefill_profile_requested();struct timespec t_begin={0},t_send={0},t_local={0},t_recv={0};if(prof)clock_gettime(CLOCK_MONOTONIC,&t_begin);
    fg_prefill_route routes[FG_GROUP_SIZE];uint32_t route_count=0;*result_count=0;
    fg_status status=fg_partition_prefill_routes(context->manifest,layer,token_count,
        expert_ids,gates,routes,&route_count,buffers->pairs,buffers->pair_capacity,err);
    fg_prefill_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,
        .destination_rank=(uint8_t)context->self,.first_position=first_token,
        .token_count=token_count,.pair_count=(uint16_t)(token_count*FG_TOP_K),
        .activations_q8k=(uint8_t *)activations,.pairs=buffers->pairs};
    uint8_t ranks[FG_GROUP_SIZE];uint32_t count=0;
    if(status==FG_OK)status=prefill_tree_ranks(context->manifest,&work,ranks,&count,err);
    uint32_t capacity=(uint32_t)coordinator_prefill_work_wire_bytes(token_count);
    uint8_t *wire=status==FG_OK&&count?malloc(capacity):NULL;
    if(status==FG_OK&&count&&!wire){fg_error_set(err,FG_ERR_OOM,"allocate prefill broadcast wire");status=FG_ERR_OOM;}
    if(status==FG_OK&&count)status=prefill_worker_buffers_ensure_result_wire(buffers,err);
    bool sent=false;
    if(status==FG_OK&&count){
        work.destination_rank=ranks[0];uint32_t bytes=0;
        status=fg_prefill_work_encode(wire,capacity,&bytes,&work,err);
        if(status==FG_OK){
            transport_pending(context->transport_state);
            status=fg_fabric_send(context->fabric,ranks[0],FG_FABRIC_BULK,FG_MSG_PREFILL_WORK,
                context->request_id,context->sequence,0,wire,bytes,err);
            sent=status==FG_OK;
            if(!sent)transport_poison(context->transport_state);
        }
    }
    free(wire);
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_send);
    if(status==FG_OK)status=shared_work(shared_context,err);
    uint32_t used_pairs=0,used_tokens=0;fg_prefill_work local={0};bool local_enqueued=false;
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++)if(routes[r].destination_rank==context->self){
        local=work;local.destination_rank=(uint8_t)context->self;
        local.pairs=routes[r].pairs;local.pair_count=routes[r].pair_count;
        status=fg_expert_prefill_enqueue(context->expert,&local,&results[0],buffers->result_pairs,
            buffers->pair_capacity,buffers->outputs,(uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE,err);
        if(status==FG_OK){used_pairs=local.pair_count;used_tokens=token_count;*result_count=1u;local_enqueued=true;}
    }
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_local);
    if(sent){
        /* Drain all expected tiles even after shared/local or tile validation
         * failure. A disconnected subtree poisons transport immediately. */
        fg_status received=FG_OK;fg_error receive_error={0};
        fg_prefill_result *result=&results[*result_count];
        *result=(fg_prefill_result){.layer=work.layer,.source_rank=ranks[0],
            .destination_rank=work.source_rank,.contributor_mask=prefill_subtree_mask(ranks,count,0u),
            .first_position=first_token,.token_count=token_count,
            .pairs=buffers->result_pairs+used_pairs,
            .outputs=buffers->outputs+(uint64_t)used_tokens*FG_HIDDEN_SIZE};
        for(uint32_t first=0;first<token_count;first+=FG_PREFILL_REDUCE_TILE_TOKENS){
            uint16_t tokens=(uint16_t)(token_count-first);
            if(tokens>FG_PREFILL_REDUCE_TILE_TOKENS)tokens=FG_PREFILL_REDUCE_TILE_TOKENS;
            fg_error tile_error={0};fg_frame_header reply;uint32_t bytes=0;
            fg_status tile_status=fg_fabric_recv(context->fabric,ranks[0],FG_FABRIC_BULK,
                &reply,buffers->result_wire,buffers->result_capacity,&bytes,&tile_error);
            if(tile_status!=FG_OK){if(received==FG_OK){received=tile_status;receive_error=tile_error;}break;}
            if(received!=FG_OK)continue;
            if(fg_frame_type(&reply)!=FG_MSG_PREFILL_RESULT||
               fg_frame_request_id(&reply)!=context->request_id||fg_frame_sequence(&reply)!=context->sequence){
                fg_error_set(&tile_error,FG_ERR_MISMATCH,"stale prefill root frame");tile_status=FG_ERR_MISMATCH;
            }
            fg_prefill_result tile={0};
            if(tile_status==FG_OK)tile_status=fg_prefill_result_decode(&tile,
                result->pairs+result->pair_count,buffers->pair_capacity-used_pairs-result->pair_count,
                result->outputs+(uint64_t)first*FG_HIDDEN_SIZE,
                (uint64_t)(token_count-first)*FG_HIDDEN_SIZE,
                buffers->result_wire,bytes,&tile_error);
            if(tile_status==FG_OK&&(tile.source_rank!=ranks[0]||
               tile.first_position!=first_token+first||tile.token_count!=tokens)){
                fg_error_set(&tile_error,FG_ERR_MISMATCH,"wrong prefill root sender or tile");tile_status=FG_ERR_MISMATCH;
            }
            if(tile_status==FG_OK)tile_status=fg_prefill_result_validate_subset(context->manifest,&work,
                result->contributor_mask,&tile,&tile_error);
            if(tile_status==FG_OK){
                for(uint32_t i=0;i<tile.pair_count;i++)tile.pairs[i].token_slot+=(uint16_t)first;
                result->pair_count+=tile.pair_count;
            }else{received=tile_status;receive_error=tile_error;}
        }
        if(received==FG_OK)received=fg_prefill_result_validate_subset(context->manifest,&work,
            result->contributor_mask,result,&receive_error);
        if(received==FG_OK){transport_complete(context->transport_state);(*result_count)++;}
        else{transport_poison(context->transport_state);if(status==FG_OK){status=received;if(err)*err=receive_error;}}
    }
    if(local_enqueued){
        fg_status local_status=fg_expert_prefill_finish(context->expert,&local,&results[0],
            buffers->result_pairs,buffers->outputs,err);
        if(status==FG_OK)status=local_status;
    }
    if(prof){clock_gettime(CLOCK_MONOTONIC,&t_recv);fprintf(stderr,"PREFILL_DISPATCH layer=%u tokens=%u remote=%u encode_send_ms=%.3f shared_local_ms=%.3f recv_ms=%.3f total_ms=%.3f\n",layer,token_count,count,elapsed_seconds(&t_begin,&t_send)*1000.0,elapsed_seconds(&t_send,&t_local)*1000.0,elapsed_seconds(&t_local,&t_recv)*1000.0,elapsed_seconds(&t_begin,&t_recv)*1000.0);}
    return status;
}

/* Pipelined prefill dispatch: fire sends the broadcast and computes local
 * experts eagerly; collect drains the remote subtree later, while another
 * chunk's layer runs on the owner.  The single local expert executor is fully
 * drained inside fire because dc87d08 measured its overlap as E2E-neutral. */
typedef struct prefill_frame {
    prefill_dispatch_context *dispatch;
    prefill_worker_buffers *buffers;
    uint32_t slot,sequence;
    fg_prefill_route routes[FG_GROUP_SIZE];uint32_t route_count;
    uint8_t ranks[FG_GROUP_SIZE];uint32_t rank_count;
    bool sent,local_enqueued,collected;
    fg_prefill_work work,local;
    uint32_t used_pairs,used_tokens;
    fg_prefill_result results[FG_GROUP_SIZE];
    uint32_t result_count;
} prefill_frame;

static fg_status prefill_fire(void *opaque,uint32_t layer,uint32_t first_token,
    uint16_t token_count,const uint16_t *expert_ids,const float *gates,
    const uint8_t *activations,fg_error *err){
    prefill_frame *f=opaque;prefill_dispatch_context *context=f->dispatch;
    prefill_worker_buffers *buffers=f->buffers;
    bool prof=prefill_profile_requested();struct timespec t_begin={0},t_send={0},t_local={0};
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_begin);
    f->sent=false;f->local_enqueued=false;f->collected=false;f->result_count=0;f->used_pairs=0;f->used_tokens=0;
    fg_status status=fg_partition_prefill_routes(context->manifest,layer,token_count,
        expert_ids,gates,f->routes,&f->route_count,buffers->pairs,buffers->pair_capacity,err);
    f->work=(fg_prefill_work){.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,
        .destination_rank=(uint8_t)context->self,.first_position=first_token,
        .token_count=token_count,.pair_count=(uint16_t)(token_count*FG_TOP_K),
        .activations_q8k=(uint8_t *)activations,.pairs=buffers->pairs};
    if(status==FG_OK)status=prefill_tree_ranks(context->manifest,&f->work,f->ranks,&f->rank_count,err);
    uint32_t capacity=(uint32_t)coordinator_prefill_work_wire_bytes(token_count);
    uint8_t *wire=status==FG_OK&&f->rank_count?malloc(capacity):NULL;
    if(status==FG_OK&&f->rank_count&&!wire){fg_error_set(err,FG_ERR_OOM,"allocate prefill broadcast wire");status=FG_ERR_OOM;}
    if(status==FG_OK&&f->rank_count)status=prefill_worker_buffers_ensure_result_wire(buffers,err);
    if(status==FG_OK&&f->rank_count){
        f->work.destination_rank=f->ranks[0];uint32_t bytes=0;
        status=fg_prefill_work_encode(wire,capacity,&bytes,&f->work,err);
        if(status==FG_OK){
            transport_pending(context->transport_state);
            status=fg_fabric_send(context->fabric,f->ranks[0],FG_FABRIC_BULK,
                FG_MSG_PREFILL_WORK,context->request_id,f->sequence,0,wire,bytes,err);
            f->sent=status==FG_OK;
            if(!f->sent)transport_poison(context->transport_state);
        }
    }
    free(wire);
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_send);
    for(uint32_t r=0;status==FG_OK&&r<f->route_count;r++)if(f->routes[r].destination_rank==context->self){
        f->local=f->work;f->local.destination_rank=(uint8_t)context->self;
        f->local.pairs=f->routes[r].pairs;f->local.pair_count=f->routes[r].pair_count;
        status=fg_expert_prefill_enqueue(context->expert,&f->local,&f->results[0],
            buffers->result_pairs,buffers->pair_capacity,buffers->outputs,
            (uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE,err);
        if(status==FG_OK){f->used_pairs=f->local.pair_count;f->used_tokens=token_count;
            f->result_count=1u;f->local_enqueued=true;}
    }
    if(f->local_enqueued){
        fg_status local_status=fg_expert_prefill_finish(context->expert,&f->local,
            &f->results[0],buffers->result_pairs,buffers->outputs,err);
        if(status==FG_OK)status=local_status;
    }
    if(prof){clock_gettime(CLOCK_MONOTONIC,&t_local);fprintf(stderr,"PREFILL_FIRE slot=%u layer=%u tokens=%u remote=%u send_ms=%.3f local_ms=%.3f total_ms=%.3f\n",f->slot,layer,token_count,f->rank_count,elapsed_seconds(&t_begin,&t_send)*1000.0,elapsed_seconds(&t_send,&t_local)*1000.0,elapsed_seconds(&t_begin,&t_local)*1000.0);}
    return status;
}

static fg_status prefill_collect(void *opaque,uint32_t layer,uint32_t first_token,
    uint16_t token_count,fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,
    fg_error *err){
    prefill_frame *f=opaque;prefill_dispatch_context *context=f->dispatch;
    prefill_worker_buffers *buffers=f->buffers;
    (void)layer;(void)first_token;
    bool prof=prefill_profile_requested();struct timespec t_begin={0},t_recv={0};
    if(prof)clock_gettime(CLOCK_MONOTONIC,&t_begin);
    uint32_t count=0;
    if(f->local_enqueued){results[0]=f->results[0];count=1u;}
    fg_status status=FG_OK;
    if(f->sent){
        fprintf(stderr,"PREFILL_RECV_WAIT slot=%u layer=%u first=%u seq=%u root=%u\n",f->slot,f->work.layer,f->work.first_position,f->sequence,f->ranks[0]);
        fg_status received=FG_OK;fg_error receive_error={0};
        fg_prefill_result *result=&results[count];
        *result=(fg_prefill_result){.layer=f->work.layer,.source_rank=f->ranks[0],
            .destination_rank=f->work.source_rank,
            .contributor_mask=prefill_subtree_mask(f->ranks,f->rank_count,0u),
            .first_position=f->work.first_position,.token_count=f->work.token_count,
            .pairs=buffers->result_pairs+f->used_pairs,
            .outputs=buffers->outputs+(uint64_t)f->used_tokens*FG_HIDDEN_SIZE};
        for(uint32_t first=0;first<token_count;first+=FG_PREFILL_REDUCE_TILE_TOKENS){
            uint16_t tokens=(uint16_t)(token_count-first);
            if(tokens>FG_PREFILL_REDUCE_TILE_TOKENS)tokens=FG_PREFILL_REDUCE_TILE_TOKENS;
            fg_error tile_error={0};fg_frame_header reply;uint32_t bytes=0;
            fg_status tile_status=fg_fabric_recv(context->fabric,f->ranks[0],FG_FABRIC_BULK,
                &reply,buffers->result_wire,buffers->result_capacity,&bytes,&tile_error);
            if(tile_status!=FG_OK){if(received==FG_OK){received=tile_status;receive_error=tile_error;}break;}
            if(received!=FG_OK)continue;
            if(fg_frame_type(&reply)!=FG_MSG_PREFILL_RESULT||
               fg_frame_request_id(&reply)!=context->request_id||
               fg_frame_sequence(&reply)!=f->sequence){
                fg_error_set(&tile_error,FG_ERR_MISMATCH,"stale prefill root frame");tile_status=FG_ERR_MISMATCH;
            }
            fg_prefill_result tile={0};
            if(tile_status==FG_OK)tile_status=fg_prefill_result_decode(&tile,
                result->pairs+result->pair_count,
                buffers->pair_capacity-f->used_pairs-result->pair_count,
                result->outputs+(uint64_t)first*FG_HIDDEN_SIZE,
                (uint64_t)(token_count-first)*FG_HIDDEN_SIZE,
                buffers->result_wire,bytes,&tile_error);
            if(tile_status==FG_OK&&(tile.source_rank!=f->ranks[0]||
               tile.first_position!=f->work.first_position+first||tile.token_count!=tokens)){
                fg_error_set(&tile_error,FG_ERR_MISMATCH,"wrong prefill root sender or tile");tile_status=FG_ERR_MISMATCH;
            }
            if(tile_status==FG_OK)tile_status=fg_prefill_result_validate_subset(context->manifest,&f->work,
                result->contributor_mask,&tile,&tile_error);
            if(tile_status==FG_OK){
                for(uint32_t i=0;i<tile.pair_count;i++)tile.pairs[i].token_slot+=(uint16_t)first;
                result->pair_count+=tile.pair_count;
            }else{received=tile_status;receive_error=tile_error;}
        }
        if(received==FG_OK)received=fg_prefill_result_validate_subset(context->manifest,&f->work,
            result->contributor_mask,result,&receive_error);
        if(received==FG_OK){transport_complete(context->transport_state);count++;}
        else{transport_poison(context->transport_state);if(status==FG_OK){status=received;if(err)*err=receive_error;}}
    }
    *result_count=count;
    f->result_count=count;f->collected=true;
    if(prof){clock_gettime(CLOCK_MONOTONIC,&t_recv);fprintf(stderr,"PREFILL_COLLECT slot=%u layer=%u tokens=%u recv_ms=%.3f\n",f->slot,f->work.layer,token_count,elapsed_seconds(&t_begin,&t_recv)*1000.0);}
    return status;
}

typedef struct prefill_layer_buffers {
    uint32_t *positions;
    fg_vk_tensor *token_tensor;
    uint32_t tokens;
} prefill_layer_buffers;

static void prefill_layer_buffers_destroy(prefill_layer_buffers *buffers){if(!buffers)return;fg_vk_tensor_destroy(buffers->token_tensor);free(buffers->positions);memset(buffers,0,sizeof(*buffers));}

static fg_status prefill_layer_buffers_create(prefill_layer_buffers *buffers,fg_model *model,uint32_t tokens,fg_error *err){memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid prefill layer buffer token count");return FG_ERR_MISMATCH;}buffers->tokens=tokens;buffers->positions=malloc((size_t)tokens*3u*4u);if(!buffers->positions){prefill_layer_buffers_destroy(buffers);fg_error_set(err,FG_ERR_OOM,"allocate bounded prefill layer buffers");return FG_ERR_OOM;}fg_status status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*4u,&buffers->token_tensor,err);if(status!=FG_OK)prefill_layer_buffers_destroy(buffers);return status;}

#define FG_QSA_OWNER_WRITE_DEPTH 8u

typedef struct qsa_owner_runtime {
    fg_qsa_owner_guard guard;
    fg_qsa_state *state;
    fg_qsa_page *pages;
    uint8_t *receive_wire,*result_wire,*read_records;
    uint32_t *blocks,*committed;
    uint8_t layers[FG_QSA_OWNER_LAYER_COUNT];
    uint32_t receive_capacity,result_capacity,layer_count;
    struct {
        uint8_t layers[FG_QSA_PAGE_APPEND_MAX_PAGES];
        uint32_t blocks[FG_QSA_PAGE_APPEND_MAX_PAGES];
        uint8_t *records;
        uint32_t page_count;
    } write_queue[FG_QSA_OWNER_WRITE_DEPTH];
    pthread_t writer;
    pthread_mutex_t writer_mutex;
    pthread_cond_t writer_ready,writer_space;
    fg_status writer_status;
    fg_error writer_error;
    uint32_t write_head,write_count;
    bool writer_started,writer_stop;
    bool enabled;
    char state_path[1200];
} qsa_owner_runtime;

static int qsa_owner_layer_slot(const qsa_owner_runtime *runtime,uint32_t layer);

static void *qsa_owner_writer_main(void *opaque){
    qsa_owner_runtime *runtime=opaque;
    pthread_mutex_lock(&runtime->writer_mutex);
    while(!runtime->writer_stop||runtime->write_count){
        while(!runtime->write_count&&!runtime->writer_stop)
            pthread_cond_wait(&runtime->writer_ready,&runtime->writer_mutex);
        if(!runtime->write_count)continue;
        uint32_t queue_index=runtime->write_head;
        uint32_t page_count=runtime->write_queue[queue_index].page_count;
        pthread_mutex_unlock(&runtime->writer_mutex);
        fg_status status=FG_OK;fg_error error={0};
        for(uint32_t slot=0;status==FG_OK&&slot<runtime->layer_count;slot++){
            uint32_t count=0;
            for(uint32_t i=0;i<page_count;i++)
                if(runtime->write_queue[queue_index].layers[i]==runtime->layers[slot]){
                    if(count>=FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES){
                        fg_error_set(&error,FG_ERR_LIMIT,
                                     "QSA owner writer layer batch exceeds capacity");
                        status=FG_ERR_LIMIT;
                        break;
                    }
                    runtime->blocks[count]=runtime->write_queue[queue_index].blocks[i];
                    memcpy(runtime->read_records+(uint64_t)count*FG_QSA_PAGE_RECORD_BYTES,
                           runtime->write_queue[queue_index].records+
                               (uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,
                           FG_QSA_PAGE_RECORD_BYTES);
                    count++;
                }
            if(count)status=fg_qsa_state_write_blocks(runtime->state,slot,runtime->blocks,
                count,runtime->read_records,&error);
        }
        pthread_mutex_lock(&runtime->writer_mutex);
        if(status!=FG_OK&&runtime->writer_status==FG_OK){
            runtime->writer_status=status;runtime->writer_error=error;
        }
        if(status!=FG_OK)runtime->write_count=0;
        else{
            runtime->write_head=(runtime->write_head+1u)%FG_QSA_OWNER_WRITE_DEPTH;runtime->write_count--;
        }
        pthread_cond_broadcast(&runtime->writer_space);
    }
    pthread_mutex_unlock(&runtime->writer_mutex);return NULL;
}

static fg_status qsa_owner_writer_drain(qsa_owner_runtime *runtime,fg_error *err){
    if(!runtime->writer_started)return FG_OK;
    pthread_mutex_lock(&runtime->writer_mutex);
    while(runtime->write_count)pthread_cond_wait(&runtime->writer_space,&runtime->writer_mutex);
    fg_status status=runtime->writer_status;
    if(status!=FG_OK&&err)*err=runtime->writer_error;
    pthread_mutex_unlock(&runtime->writer_mutex);return status;
}

static fg_status qsa_owner_writer_enqueue(qsa_owner_runtime *runtime,
                                          const fg_qsa_page_batch *batch,fg_error *err){
    pthread_mutex_lock(&runtime->writer_mutex);
    if(runtime->writer_status!=FG_OK){
        fg_status status=runtime->writer_status;if(err)*err=runtime->writer_error;
        pthread_mutex_unlock(&runtime->writer_mutex);return status;
    }
    if(runtime->write_count==FG_QSA_OWNER_WRITE_DEPTH){
        pthread_mutex_unlock(&runtime->writer_mutex);
        fg_error_set(err,FG_ERR_LIMIT,
                     "QSA owner writer queue is backpressured; session must reset");
        return FG_ERR_LIMIT;
    }
    uint32_t queue_index=(runtime->write_head+runtime->write_count)%FG_QSA_OWNER_WRITE_DEPTH;
    runtime->write_queue[queue_index].page_count=batch->page_count;
    for(uint32_t i=0;i<batch->page_count;i++){
        runtime->write_queue[queue_index].layers[i]=batch->pages[i].layer;
        runtime->write_queue[queue_index].blocks[i]=batch->pages[i].block;
        memcpy(runtime->write_queue[queue_index].records+
                   (uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,
               batch->pages[i].records,FG_QSA_PAGE_RECORD_BYTES);
    }
    runtime->write_count++;pthread_cond_signal(&runtime->writer_ready);
    pthread_mutex_unlock(&runtime->writer_mutex);return FG_OK;
}

static void qsa_owner_runtime_destroy(qsa_owner_runtime *runtime){
    if(!runtime)return;
    if(runtime->writer_started){
        pthread_mutex_lock(&runtime->writer_mutex);runtime->writer_stop=true;
        pthread_cond_signal(&runtime->writer_ready);pthread_mutex_unlock(&runtime->writer_mutex);
        pthread_join(runtime->writer,NULL);
        pthread_cond_destroy(&runtime->writer_space);
        pthread_cond_destroy(&runtime->writer_ready);
        pthread_mutex_destroy(&runtime->writer_mutex);
    }
    fg_qsa_state_close(runtime->state);
    if(runtime->state_path[0])unlink(runtime->state_path);
    free(runtime->committed);free(runtime->blocks);free(runtime->read_records);
    for(uint32_t i=0;i<FG_QSA_OWNER_WRITE_DEPTH;i++)free(runtime->write_queue[i].records);
    free(runtime->result_wire);free(runtime->receive_wire);free(runtime->pages);
    memset(runtime,0,sizeof(*runtime));
}

static fg_status qsa_owner_runtime_create(qsa_owner_runtime *runtime,
                                          const fg_manifest *manifest,uint32_t rank,
                                          fg_error *err){
    memset(runtime,0,sizeof(*runtime));fg_qsa_owner_guard_init(&runtime->guard,rank);
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(manifest->layer_owner[layer]==rank)
            runtime->layers[runtime->layer_count++]=(uint8_t)layer;
    if(!runtime->layer_count)return FG_OK;
    if(runtime->layer_count>FG_QSA_OWNER_LAYER_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,"rank %u has %u QSA owner layers",
                     rank,runtime->layer_count);return FG_ERR_MISMATCH;
    }
    runtime->enabled=true;runtime->receive_capacity=FG_QSA_PAGE_APPEND_MAX_BYTES;
    runtime->result_capacity=FG_QSA_PAGE_RESULT_MAX_BYTES;
    runtime->pages=calloc(FG_QSA_PAGE_APPEND_MAX_PAGES,sizeof(*runtime->pages));
    runtime->receive_wire=malloc(runtime->receive_capacity);
    runtime->result_wire=malloc(runtime->result_capacity);
    runtime->read_records=malloc((uint64_t)FG_QSA_PAGE_FETCH_MAX_PAGES*
                                  FG_QSA_PAGE_RECORD_BYTES);
    runtime->blocks=malloc((size_t)FG_QSA_PAGE_FETCH_MAX_PAGES*sizeof(*runtime->blocks));
    runtime->committed=malloc((size_t)FG_QSA_PAGE_FETCH_MAX_PAGES*
                              sizeof(*runtime->committed));
    if(!runtime->pages||!runtime->receive_wire||!runtime->result_wire||
       !runtime->read_records||!runtime->blocks||!runtime->committed){
        qsa_owner_runtime_destroy(runtime);fg_error_set(err,FG_ERR_OOM,"allocate QSA owner buffers");
        return FG_ERR_OOM;
    }
    for(uint32_t i=0;i<FG_QSA_OWNER_WRITE_DEPTH;i++){
        runtime->write_queue[i].records=malloc((uint64_t)FG_QSA_PAGE_APPEND_MAX_PAGES*
                                               FG_QSA_PAGE_RECORD_BYTES);
        if(!runtime->write_queue[i].records){
            qsa_owner_runtime_destroy(runtime);
            fg_error_set(err,FG_ERR_OOM,"allocate asynchronous QSA page writer queue");
            return FG_ERR_OOM;
        }
    }
    if(pthread_mutex_init(&runtime->writer_mutex,NULL)!=0){
        qsa_owner_runtime_destroy(runtime);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"initialize QSA page writer synchronization");
        return FG_ERR_UNAVAILABLE;
    }
    if(pthread_cond_init(&runtime->writer_ready,NULL)!=0){
        pthread_mutex_destroy(&runtime->writer_mutex);qsa_owner_runtime_destroy(runtime);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"initialize QSA page writer synchronization");
        return FG_ERR_UNAVAILABLE;
    }
    if(pthread_cond_init(&runtime->writer_space,NULL)!=0){
        pthread_cond_destroy(&runtime->writer_ready);pthread_mutex_destroy(&runtime->writer_mutex);
        qsa_owner_runtime_destroy(runtime);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"initialize QSA page writer synchronization");
        return FG_ERR_UNAVAILABLE;
    }
    runtime->writer_status=FG_OK;
    if(pthread_create(&runtime->writer,NULL,qsa_owner_writer_main,runtime)!=0){
        pthread_cond_destroy(&runtime->writer_space);pthread_cond_destroy(&runtime->writer_ready);
        pthread_mutex_destroy(&runtime->writer_mutex);memset(&runtime->writer_mutex,0,
                                                              sizeof(runtime->writer_mutex));
        qsa_owner_runtime_destroy(runtime);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"start asynchronous QSA page writer");
        return FG_ERR_UNAVAILABLE;
    }
    runtime->writer_started=true;
    return FG_OK;
}

static fg_status qsa_owner_open_session(qsa_owner_runtime *runtime,
                                        const fg_manifest *manifest,const char *directory,
                                        const fg_session_identity *identity,uint64_t request,
                                        const fg_owner_session_control *control,fg_error *err){
    if(!runtime->enabled)return FG_OK;
    fg_qsa_owner_guard next=runtime->guard;
    fg_status status=fg_qsa_owner_guard_begin(&next,manifest,identity,request,control,err);
    if(status!=FG_OK)return status;
    status=qsa_owner_writer_drain(runtime,err);
    if(status!=FG_OK)return status;
    char path[sizeof(runtime->state_path)];int length=snprintf(path,sizeof(path),
        "%s/qsa-owner-rank-%02u.state",directory,runtime->guard.rank);
    if(length<0||(size_t)length>=sizeof(path)){fg_error_set(err,FG_ERR_LIMIT,"QSA owner state path overflow");return FG_ERR_LIMIT;}
    fg_qsa_state_close(runtime->state);runtime->state=NULL;
    if(runtime->state_path[0])unlink(runtime->state_path);
    if(unlink(path)!=0&&errno!=ENOENT){fg_error_set(err,FG_ERR_IO,"remove stale QSA owner state: %s",strerror(errno));return FG_ERR_IO;}
    uint32_t logical_context=control?control->logical_context_tokens:
        manifest->session.logical_context_tokens;
    status=fg_qsa_state_open(&runtime->state,path,runtime->layers,runtime->layer_count,
                             logical_context,true,err);
    if(status==FG_OK){
        pthread_mutex_lock(&runtime->writer_mutex);runtime->writer_status=FG_OK;
        memset(&runtime->writer_error,0,sizeof(runtime->writer_error));
        pthread_mutex_unlock(&runtime->writer_mutex);
        runtime->guard=next;snprintf(runtime->state_path,sizeof(runtime->state_path),"%s",path);
    }
    return status;
}

static int qsa_owner_layer_slot(const qsa_owner_runtime *runtime,uint32_t layer){
    for(uint32_t i=0;i<runtime->layer_count;i++)if(runtime->layers[i]==layer)return (int)i;
    return -1;
}

static uint32_t owned_qsa_layers(const fg_manifest *manifest,uint32_t rank){
    uint32_t count=0;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(manifest->layer_owner[layer]==rank)count++;
    return count;
}

static fg_status worker_open_qsa_state(fg_owner_executor *owner,qsa_owner_runtime *runtime,
    const fg_manifest *manifest,const char *directory,uint32_t self,uint32_t logical,
    fg_error *err){
    if(!owner||!runtime->enabled||fg_owner_qsa_ready(owner))return FG_OK;
    char path[1200];
    if(snprintf(path,sizeof(path),"%s/qsa-owner-rank-%02u.state",directory,self)>=
       (int)sizeof(path)){
        fg_error_set(err,FG_ERR_LIMIT,"worker QSA state path overflow");
        return FG_ERR_LIMIT;
    }
    uint32_t layers=owned_qsa_layers(manifest,self);
    uint32_t full_pages=logical/FG_Q38_QSA_COMPRESS_RATIO;
    if(!full_pages)full_pages=1u;
    uint32_t capped_pages=full_pages<FG_QSA_WORKER_CACHE_PAGES?
        full_pages:FG_QSA_WORKER_CACHE_PAGES;
    uint32_t cache_pages=capped_pages*layers;
    return fg_owner_qsa_open_state(owner,path,logical,0u,cache_pages,
                                   manifest->prefill_microbatch,err);
}

/* The block owner's state-backed QSA session persists every completed block.
 * Advance the runtime guard frontier so decode cold fetches from rank 0 are
 * accepted for the tokens this block produced. */
static fg_status worker_publish_qsa_pages(void *opaque,fg_owner_executor *owner,
    uint32_t self,uint32_t first_token,uint16_t token_count,fg_error *err){
    (void)owner;(void)self;(void)err;
    qsa_owner_runtime *runtime=opaque;
    if(!runtime||!runtime->enabled)return FG_OK;
    uint32_t frontier=first_token+token_count;
    for(uint32_t slot=0;slot<runtime->layer_count;slot++){
        uint32_t layer=runtime->layers[slot];
        if(frontier>runtime->guard.next_token[layer])
            runtime->guard.next_token[layer]=frontier;
    }
    return FG_OK;
}

static fg_status handle_qsa_page_append(qsa_owner_runtime *runtime,
                                        const fg_manifest *manifest,uint32_t peer,
                                        const fg_frame_header *header,const uint8_t *payload,
                                        uint32_t bytes,fg_error *err){
    if(!runtime->enabled||!runtime->state||peer!=0u){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page append reached an inactive owner");
        return FG_ERR_MISMATCH;
    }
    fg_qsa_page_batch batch={0};fg_status status=fg_qsa_page_append_decode(&batch,
        runtime->pages,FG_QSA_PAGE_APPEND_MAX_PAGES,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&fg_frame_sequence(header)!=batch.batch_id){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page append frame sequence mismatch");
        status=FG_ERR_MISMATCH;
    }
    fg_qsa_owner_guard next=runtime->guard;
    if(status==FG_OK)status=fg_qsa_owner_guard_accept_append(&next,manifest,request,
                                                            &batch,err);
    if(status==FG_OK)status=qsa_owner_writer_enqueue(runtime,&batch,err);
    if(status==FG_OK)runtime->guard=next;
    return status;
}

static fg_status handle_qsa_page_barrier(fg_fabric *fabric,qsa_owner_runtime *runtime,
                                        uint32_t self,uint32_t peer,
                                        const fg_frame_header *header,const uint8_t *payload,
                                        uint32_t bytes,fg_error *err){
    fg_qsa_page_barrier barrier={0};fg_status status=fg_qsa_page_barrier_decode(
        &barrier,payload,bytes,err);uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(peer!=0u||fg_frame_sequence(header)!=barrier.batch_id)){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page barrier frame mismatch");status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_qsa_owner_guard_accept_barrier(&runtime->guard,request,
                                                             &barrier,err);
    if(status==FG_OK)status=qsa_owner_writer_drain(runtime,err);
    uint8_t wire[FG_QSA_PAGE_BARRIER_BYTES];
    if(status==FG_OK){
        barrier.source_rank=(uint8_t)self;barrier.destination_rank=0u;
        status=fg_qsa_page_barrier_encode(wire,&barrier,err);
    }
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
        FG_MSG_QSA_PAGE_BARRIER_ACK,request,barrier.batch_id,0,wire,sizeof(wire),err);
    return status;
}

static fg_status handle_qsa_page_fetch(fg_fabric *fabric,qsa_owner_runtime *runtime,
                                       fg_owner_executor *owner,
                                       const fg_manifest *manifest,uint32_t self,uint32_t peer,
                                       const fg_frame_header *header,const uint8_t *payload,
                                       uint32_t bytes,fg_error *err){
    if(!runtime->enabled||!runtime->state||peer!=0u){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page fetch reached an inactive owner");
        return FG_ERR_MISMATCH;
    }
    fg_qsa_page_batch batch={0};fg_status status=fg_qsa_page_fetch_decode(&batch,
        runtime->pages,FG_QSA_PAGE_FETCH_MAX_PAGES,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&fg_frame_sequence(header)!=batch.batch_id){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page fetch frame sequence mismatch");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_qsa_owner_guard_accept_fetch(&runtime->guard,manifest,
                                                           request,&batch,err);
    if(status==FG_OK)status=qsa_owner_writer_drain(runtime,err);
    int slot=status==FG_OK?qsa_owner_layer_slot(runtime,batch.pages[0].layer):-1;
    if(status==FG_OK&&slot<0){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA page fetch layer is not local");
        status=FG_ERR_MISMATCH;
    }
    for(uint32_t i=0;status==FG_OK&&i<batch.page_count;i++){
        runtime->blocks[i]=batch.pages[i].block;
        runtime->pages[i].records=runtime->read_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES;
    }
    if(status==FG_OK&&owner&&fg_owner_qsa_ready(owner)){
        /* The state-backed ring session is authoritative: read its records
         * directly instead of a second, stale state handle or a cache entry. */
        for(uint32_t i=0;status==FG_OK&&i<batch.page_count;i++)
            status=fg_owner_qsa_state_records(owner,batch.pages[i].layer,
                batch.pages[i].block,runtime->read_records+
                    (uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,err);
    }else if(status==FG_OK){
        status=fg_qsa_state_read_blocks(runtime->state,(uint32_t)slot,
            runtime->blocks,batch.page_count,runtime->read_records,runtime->committed,err);
        for(uint32_t i=0;status==FG_OK&&i<batch.page_count;i++){
            if(runtime->committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA owner returned an incomplete page");
                status=FG_ERR_MISMATCH;
            }
        }
    }
    fg_qsa_page_batch result={.source_rank=(uint8_t)self,.destination_rank=0u,
        .batch_id=batch.batch_id,.page_count=batch.page_count,.pages=runtime->pages};
    uint32_t result_bytes=0;
    if(status==FG_OK)status=fg_qsa_page_result_encode(runtime->result_wire,
        runtime->result_capacity,&result_bytes,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
        FG_MSG_QSA_PAGE_RESULT,request,batch.batch_id,0,runtime->result_wire,result_bytes,err);
    return status;
}

static fg_status begin_session(fg_fabric *fabric,const fg_manifest *manifest,
                               const char *directory,qsa_owner_runtime *qsa,
                               fg_owner_executor *owner,uint32_t self,
                               uint32_t peer,const fg_frame_header *header,const uint8_t *payload,
                               uint32_t bytes,uint64_t *session_id,
                               fg_output_executor *output,fg_error *err){
    uint64_t request=fg_frame_request_id(header);
    if(peer!=0u||!request||(*session_id&&request<=*session_id)){fg_error_set(err,FG_ERR_MISMATCH,"invalid, stale, or duplicate session begin");return FG_ERR_MISMATCH;}
    fg_session_identity identity;fg_status status=fg_session_identity_from_manifest(manifest,&identity,err);
    if(manifest->protocol_version==FG_PROTOCOL_MIN_VERSION){
        if(status==FG_OK&&bytes){fg_error_set(err,FG_ERR_FORMAT,"legacy session begin payload must be empty");status=FG_ERR_FORMAT;}
        if(status==FG_OK)status=qsa_owner_open_session(qsa,manifest,directory,&identity,
                                                       request,NULL,err);
        if(status==FG_OK)status=worker_open_qsa_state(owner,qsa,manifest,directory,self,
                                                      manifest->session.logical_context_tokens,err);
        if(status==FG_OK&&owner)status=fg_owner_reset_state(owner,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,
                                               FG_MSG_SESSION_READY,request,0,0,NULL,0,err);
        if(status==FG_OK&&output)status=fg_output_history_reset(output,NULL,0u,err);
        if(status==FG_OK)*session_id=request;
        return status;
    }
    fg_owner_session_control control;if(status==FG_OK)status=fg_owner_session_control_decode(
        &control,payload,bytes,err);
    if(status==FG_OK&&(control.operation!=FG_OWNER_SESSION_BEGIN||control.rank!=self||
       control.session_nonce!=request||
       control.position_mode!=(fg_position_mode)manifest->session.position_mode||
       memcmp(control.identity_sha256,identity.identity_sha256,32u)||
       memcmp(control.state_format_sha256,
              manifest->session.rank_state_format_sha256[self],32u))){
        fg_error_set(err,FG_ERR_MISMATCH,"owner session begin identity or rank mismatch");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=qsa_owner_open_session(qsa,manifest,directory,&identity,
                                                   request,&control,err);
    if(status==FG_OK)status=worker_open_qsa_state(owner,qsa,manifest,directory,self,
                                                  control.logical_context_tokens,err);
    if(status==FG_OK&&owner)status=fg_owner_reset_state(owner,err);
    if(status==FG_OK&&output)status=fg_output_history_reset(output,NULL,0u,err);
    uint8_t wire[FG_OWNER_SESSION_CONTROL_BYTES];
    if(status==FG_OK){
        control.operation=FG_OWNER_SESSION_READY;
        status=fg_owner_session_control_encode(wire,&control,err);
    }
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,
                                           FG_MSG_SESSION_READY,request,0,0,wire,sizeof(wire),err);
    if(status==FG_OK)*session_id=request;
    return status;
}

static fg_status handle_ngram_work(fg_fabric *fabric,
                                   const fg_ngram_resident *resident,
                                   fg_fabric_class result_class,uint32_t self,
                                   uint64_t session_id,uint32_t peer,
                                   const fg_frame_header *header,
                                   const uint8_t *payload,uint32_t bytes,
                                   fg_error *err){
    fg_ngram_work work;
    fg_status status=fg_ngram_work_decode(&work,payload,bytes,err);
    uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=0u||
       work.source_rank!=0u||work.destination_rank!=self||
       fg_frame_sequence(header)!=work.token_index)){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "stale or misrouted resident n-gram work");
        status=FG_ERR_MISMATCH;
    }
    fg_ngram_result result={.source_rank=(uint8_t)self,.destination_rank=0u,
        .item_count=work.item_count,.token_index=work.token_index};
    if(status==FG_OK){
        memcpy(result.heads,work.heads,work.item_count);
        status=fg_ngram_resident_read(resident,work.rows,work.item_count,
                                      result.packed,sizeof(result.packed),err);
    }
    uint8_t wire[FG_NGRAM_RESULT_MAX_BYTES];uint32_t result_bytes=0;
    if(status==FG_OK)status=fg_ngram_result_encode(
        wire,sizeof(wire),&result_bytes,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,result_class,
        FG_MSG_NGRAM_RESULT,request,work.token_index,0,wire,result_bytes,err);
    return status;
}

static fg_status handle_output_work(fg_fabric *fabric,fg_output_executor *output,fg_vk_context *vk,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_error *err){
    if(self!=4u||!output){fg_error_set(err,FG_ERR_MISMATCH,"output work reached a non-output rank");return FG_ERR_MISMATCH;}
    fg_output_work *work=calloc(1,sizeof(*work));if(!work){fg_error_set(err,FG_ERR_OOM,"allocate output work");return FG_ERR_OOM;}
    fg_status status=fg_output_work_decode(work,payload,bytes,err);uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work->source_rank||work->destination_rank!=self)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output work");status=FG_ERR_MISMATCH;}
    token_profile_capture capture={0};if(status==FG_OK)status=token_profile_begin(&capture,vk,work->token_index,err);
    if(status==FG_OK)status=fg_vk_tensor_write(hyper_tensor,0,work->hyper,sizeof(work->hyper),err);
    fg_output_result result={.source_rank=(uint8_t)self,.destination_rank=work->source_rank,.token_index=work->token_index};
    if(status==FG_OK)status=fg_output_sample(output,hyper_tensor,&work->sampler,
        work->uniform,&result.token,&result.logit,err);
    uint8_t wire[FG_OUTPUT_RESULT_BYTES];if(status==FG_OK)status=fg_output_result_encode(wire,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,request,fg_frame_sequence(header),0,wire,sizeof(wire),err);
    status=token_profile_end(&capture,self,"output",work->token_index,UINT32_MAX,status,err);
    free(work);return status;
}

/* Direct output handoff on the output owner.  The sampler config (control
 * channel) and the 40 KiB hidden (bulk channel) are buffered until both name
 * the same token, then the head runs and the 16-byte result returns to rank 0
 * on the bulk channel exactly like the relay route. */
#define FG_OUTPUT_SPLIT_TIMEOUT_DEFAULT_MS 4000u
#define FG_OUTPUT_SPLIT_TIMEOUT_MAX_MS 60000u
#define FG_OUTPUT_SPLIT_TIMEOUT_GRACE_MS 4000u

static uint32_t output_split_timeout_ms(void){
    const char *value=getenv("FG_OUTPUT_SPLIT_TIMEOUT_MS");
    if(!value||!*value)return FG_OUTPUT_SPLIT_TIMEOUT_DEFAULT_MS;
    long parsed=strtol(value,NULL,10);
    if(parsed<(long)100u)parsed=100;
    if(parsed>(long)FG_OUTPUT_SPLIT_TIMEOUT_MAX_MS)parsed=(long)FG_OUTPUT_SPLIT_TIMEOUT_MAX_MS;
    return (uint32_t)parsed;
}

static bool output_split_trace_enabled(void){
    const char *value=getenv("FG_OUTPUT_SPLIT_TRACE");
    return value&&*value&&strcmp(value,"0")!=0;
}

static void output_split_trace(const char *what,uint32_t rank,uint32_t token,uint32_t detail){
    if(output_split_trace_enabled())
        fprintf(stderr,"OUTPUT_SPLIT_STATE rank=%u token=%u what=%s detail=%u\n",
                rank,token,what,detail);
}

static bool worker_output_split_active(const fg_output_handoff *state){
    return state&&(state->config.flags&FG_OUTPUT_CONFIG_FLAG_SPLIT)!=0u&&
        state->config.sampler.temperature==0.0f&&
        !fg_sampler_penalties_active(&state->config.sampler);
}

static uint32_t worker_output_split_ways(const fg_output_handoff *state){
    return state->config.flags&FG_OUTPUT_CONFIG_FLAG_SPLIT_4?
        FG_OUTPUT_SPLIT_WAYS_MAX:FG_OUTPUT_SPLIT_WAYS_MIN;
}

static bool worker_output_split_pending(const fg_output_handoff *state){
    if(!state||!worker_output_split_active(state)||!state->have_local)return false;
    if(worker_output_split_ways(state)!=FG_OUTPUT_SPLIT_WAYS_MAX)return false;
    return state->remote_count<worker_output_split_ways(state)-1u;
}

static int32_t worker_output_split_remaining_ms(const fg_output_handoff *state){
    if(!state||!state->wait_start_ms)return -1;
    return fg_output_split_wait_remaining_ms(state->wait_start_ms,monotonic_ms(),
                                             output_split_timeout_ms());
}

static fg_status worker_output_split_timeout(fg_output_handoff *state,uint32_t self,
                                             fg_error *err){
    uint32_t ways=worker_output_split_ways(state);
    uint64_t now=monotonic_ms();
    uint32_t waited_ms=state->wait_start_ms&&now>=state->wait_start_ms?
        (uint32_t)(now-state->wait_start_ms):0u;
    fg_status status=fg_output_split_timeout_error(state,ways,waited_ms,err);
    fprintf(stderr,"OUTPUT_SPLIT_TIMEOUT rank=%u token=%u waited_ms=%u ways=%u %s\n",
            self,state->config.token_index,waited_ms,ways,err->message);
    return status;
}

static fg_status worker_output_split_combine(fg_output_handoff *state,uint32_t ways,
    uint32_t *token,float *logit,fg_error *err){
    uint32_t combined_id=state->local_id;
    float combined_value=state->local_value;
    for(uint32_t i=0u;i<state->remote_count;i++){
        uint32_t way=0u;
        if(!fg_output_split_way_for_rank(ways,state->remote_rank[i],&way)){
            fg_error_set(err,FG_ERR_MISMATCH,
                "output partial from rank %u which owns no slice in the %u-way split",
                state->remote_rank[i],ways);
            return FG_ERR_MISMATCH;
        }
        uint32_t first_row=0u,rows=0u;
        fg_output_split_span(ways,way,&first_row,&rows);
        if(state->remote_id[i]<first_row||state->remote_id[i]>=first_row+rows){
            fg_error_set(err,FG_ERR_MISMATCH,
                "output partial id %u is outside rank %u slice %u..%u",
                state->remote_id[i],state->remote_rank[i],first_row,first_row+rows);
            return FG_ERR_MISMATCH;
        }
        fg_output_combine(state->remote_value[i],state->remote_id[i],
            combined_value,combined_id,&combined_value,&combined_id);
    }
    if(combined_id>=FG_Q38_VOCAB_SIZE||!isfinite(combined_value)){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid combined output finalist");
        return FG_ERR_MISMATCH;
    }
    *token=combined_id;if(logit)*logit=combined_value;
    return FG_OK;
}

static fg_status worker_output_handoff_flush(fg_fabric *fabric,fg_output_executor *output,
    fg_output_slice *output_slice,fg_vk_context *vk,uint32_t self,uint64_t session_id,
    fg_vk_tensor *hyper_tensor,fg_output_handoff *state,fg_error *err){
    bool split=worker_output_split_active(state);
    if(split?!fg_output_handoff_ready(state):!fg_output_handoff_sample_ready(state))
        return FG_OK;
    const fg_output_config *config=&state->config;
    const fg_layer_result *hidden=&state->hidden;
    fg_output_result result={.source_rank=(uint8_t)self,.destination_rank=0u,
        .token_index=config->token_index};
    if(split){
        uint32_t ways=worker_output_split_ways(state);
        fg_status missing=fg_output_split_require_slice(output_slice!=NULL,ways,self,err);
        if(missing!=FG_OK)return missing;
        if(fg_output_slice_ways(output_slice)!=ways){
            fg_error_set(err,FG_ERR_MISMATCH,
                "output config declares a %u-way split but this rank holds a %u-way slice executor (FG_OUTPUT_SPLIT must match on all ranks)",
                ways,fg_output_slice_ways(output_slice));
            return FG_ERR_MISMATCH;
        }
        token_profile_capture capture={0};
        fg_status status=token_profile_begin(&capture,vk,config->token_index,err);
        if(status==FG_OK&&!state->have_local){
            if(ways==FG_OUTPUT_SPLIT_WAYS_MAX)
                status=fg_output_slice_run_hidden(output_slice,state->hidden_slice.hyper,
                    &state->local_value,&state->local_id,err);
            else
                status=fg_output_slice_run(output_slice,hidden->hyper,
                    &state->local_value,&state->local_id,err);
            if(status==FG_OK)state->have_local=true;
            output_split_trace("local",self,config->token_index,ways);
        }
        if(status==FG_OK&&state->remote_count<ways-1u){
            if(!state->wait_start_ms)state->wait_start_ms=monotonic_ms();
            status=token_profile_end(&capture,self,"output",config->token_index,
                                     UINT32_MAX,status,err);
            return status;
        }
        uint32_t token=0u;float logit=0.0f;
        if(status==FG_OK)status=worker_output_split_combine(state,ways,&token,&logit,err);
        result.token=token;result.logit=logit;
        uint8_t wire[FG_OUTPUT_RESULT_BYTES];
        if(status==FG_OK)status=fg_output_result_encode(wire,&result,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,
            session_id,config->token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,sizeof(wire),err);
        status=token_profile_end(&capture,self,"output",config->token_index,UINT32_MAX,status,err);
        output_split_trace("combined",self,config->token_index,state->remote_count);
        fg_output_handoff_reset(state);
        return status;
    }
    token_profile_capture capture={0};
    fg_status status=token_profile_begin(&capture,vk,config->token_index,err);
    if(status==FG_OK)status=fg_vk_tensor_write(hyper_tensor,0,hidden->hyper,
        (uint64_t)FG_HYPER_WIDTH*4u,err);
    if(status==FG_OK)status=fg_output_sample(output,hyper_tensor,&config->sampler,
        config->uniform,&result.token,&result.logit,err);
    uint8_t wire[FG_OUTPUT_RESULT_BYTES];
    if(status==FG_OK)status=fg_output_result_encode(wire,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,
        session_id,config->token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,sizeof(wire),err);
    status=token_profile_end(&capture,self,"output",config->token_index,UINT32_MAX,status,err);
    fg_output_handoff_reset(state);
    return status;
}

static fg_status handle_output_partial(fg_fabric *fabric,fg_output_executor *output,
    fg_output_slice *output_slice,fg_vk_context *vk,uint32_t self,uint64_t session_id,
    uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    fg_vk_tensor *hyper_tensor,fg_output_handoff *state,fg_error *err){
    if(self!=4u||!state){
        fg_error_set(err,FG_ERR_MISMATCH,"output partial reached a rank without the split head");
        return FG_ERR_MISMATCH;
    }
    fg_status head=fg_output_split_require_slice(output_slice!=NULL,0u,self,err);
    if(head!=FG_OK)return head;
    fg_output_partial partial;
    fg_status status=fg_output_partial_decode(&partial,payload,bytes,err);
    uint32_t ways=fg_output_slice_ways(output_slice),way=0u;
    if(status==FG_OK&&(!session_id||peer==self||
       !fg_output_split_way_for_rank(ways,peer,&way)||
       fg_frame_request_id(header)!=session_id||
       fg_frame_sequence(header)!=partial.token_index*FG_LAYER_COUNT+FG_LAYER_COUNT)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output partial");
        status=FG_ERR_MISMATCH;
    }
        if(status==FG_OK&&!state->have_config)return FG_OK;
    if(status==FG_OK)status=fg_output_handoff_partial(state,partial.token_index,(uint8_t)peer,
        partial.value,partial.id,err);
    if(status==FG_OK)output_split_trace("partial",self,partial.token_index,peer);
    if(status==FG_OK)status=worker_output_handoff_flush(fabric,output,output_slice,vk,self,
        session_id,hyper_tensor,state,err);
    return status;
}

static fg_status handle_output_slice_hidden(fg_fabric *fabric,fg_output_executor *output,
    fg_output_slice *output_slice,fg_vk_context *vk,const fg_manifest *manifest,uint32_t self,
    uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,
    uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_output_handoff *state,fg_error *err){
    if(!decode_direct_output_eligible(manifest)){
        fg_error_set(err,FG_ERR_MISMATCH,
            "rank %u received a 4-way output slice without the direct output handoff",self);
        return FG_ERR_MISMATCH;
    }
    fg_status status=fg_output_split_require_slice(output_slice!=NULL,
        FG_OUTPUT_SPLIT_WAYS_MAX,self,err);
    if(status!=FG_OK)return status;
    if(fg_output_slice_ways(output_slice)!=FG_OUTPUT_SPLIT_WAYS_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,
            "rank %u holds a %u-way slice executor for a 4-way output slice (FG_OUTPUT_SPLIT must match on all ranks)",
            self,fg_output_slice_ways(output_slice));
        return FG_ERR_MISMATCH;
    }
    fg_output_slice_hidden slice;
    status=fg_output_slice_hidden_decode(&slice,payload,bytes,err);
    if(status==FG_OK&&(!session_id||fg_frame_request_id(header)!=session_id||
       slice.destination_rank!=self||
       slice.source_rank!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
       peer!=slice.source_rank||
       fg_frame_sequence(header)!=slice.token_index*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output slice hidden");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK&&self==4u){
        if(!state){
            fg_error_set(err,FG_ERR_MISMATCH,"output owner has no handoff state");
            return FG_ERR_MISMATCH;
        }
        fg_layer_result stored={.layer=(uint8_t)(FG_LAYER_COUNT-1u),
            .source_rank=slice.source_rank,.destination_rank=(uint8_t)self,
            .token_index=slice.token_index};
        memcpy(stored.hyper,slice.hidden,sizeof(slice.hidden));
        status=fg_output_handoff_hidden_slice(state,&stored,err);
        if(status==FG_OK)status=worker_output_handoff_flush(fabric,output,output_slice,vk,
            self,session_id,hyper_tensor,state,err);
        return status;
    }
    if(status==FG_OK){
        float value=0.0f;uint32_t id=0u;
        status=fg_output_slice_run_hidden(output_slice,slice.hidden,&value,&id,err);
        if(status==FG_OK)output_split_trace("slice",self,slice.token_index,id);
        fg_output_partial partial={.token_index=slice.token_index,.value=value,.id=id};
        uint8_t wire[FG_OUTPUT_PARTIAL_BYTES];
        if(status==FG_OK)status=fg_output_partial_encode(wire,&partial,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_PARTIAL,session_id,
            slice.token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,sizeof(wire),err);
    }
    return status;
}

static fg_status handle_output_config(fg_fabric *fabric,fg_output_executor *output,
    fg_output_slice *output_slice,fg_vk_context *vk,const fg_manifest *manifest,uint32_t self,
    uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,
    uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_output_handoff *state,fg_error *err){
    if(self!=4u||!output||!state||!decode_direct_output_eligible(manifest)){
        fg_error_set(err,FG_ERR_MISMATCH,"output config reached a rank without the direct handoff");
        return FG_ERR_MISMATCH;
    }
    fg_output_config config;
    fg_status status=fg_output_config_decode(&config,payload,bytes,err);
    if(status==FG_OK&&(!session_id||fg_frame_request_id(header)!=session_id||peer!=0u||
       config.source_rank!=0u||config.destination_rank!=self||
       fg_frame_sequence(header)!=config.token_index*FG_LAYER_COUNT+FG_LAYER_COUNT)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output config");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_handoff_config(state,&config,err);
    if(status==FG_OK)status=worker_output_handoff_flush(fabric,output,output_slice,vk,self,
        session_id,hyper_tensor,state,err);
    return status;
}

static fg_status handle_output_hidden(fg_fabric *fabric,fg_output_executor *output,
    fg_output_slice *output_slice,fg_vk_context *vk,const fg_manifest *manifest,uint32_t self,
    uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,
    uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_output_handoff *state,fg_error *err){
    if(self!=4u||!output||!state||!decode_direct_output_eligible(manifest)){
        fg_error_set(err,FG_ERR_MISMATCH,"output hidden reached a rank without the direct handoff");
        return FG_ERR_MISMATCH;
    }
    fg_layer_result hidden;
    fg_status status=fg_decode_layer_result_decode(&hidden,payload,bytes,err);
    if(status==FG_OK&&(!session_id||fg_frame_request_id(header)!=session_id||
       hidden.destination_rank!=self||
       hidden.source_rank!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
       peer!=hidden.source_rank||hidden.layer!=FG_LAYER_COUNT-1u||
       fg_frame_sequence(header)!=hidden.token_index*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output hidden");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_handoff_hidden(state,&hidden,err);
    if(status==FG_OK)status=worker_output_handoff_flush(fabric,output,output_slice,vk,self,
        session_id,hyper_tensor,state,err);
    return status;
}

static bool output_history_count(const uint8_t *payload,uint32_t bytes,
                                 uint32_t *count){
    if(!payload||!count||bytes<FG_OUTPUT_HISTORY_HEADER_BYTES)return false;
    uint32_t value=((uint32_t)payload[0u]<<24u)|((uint32_t)payload[1u]<<16u)|
                   ((uint32_t)payload[2u]<<8u)|payload[3u];
    uint32_t reserved=((uint32_t)payload[4u]<<24u)|((uint32_t)payload[5u]<<16u)|
                      ((uint32_t)payload[6u]<<8u)|payload[7u];
    if(reserved||value>FG_NATIVE_CONTEXT||
       (uint64_t)FG_OUTPUT_HISTORY_HEADER_BYTES+(uint64_t)value*4u!=bytes)
        return false;
    *count=value;return true;
}

static fg_status handle_output_history(fg_fabric *fabric,fg_output_executor *output,
                                       uint32_t self,uint64_t session_id,uint32_t peer,
                                       const fg_frame_header *header,const uint8_t *payload,
                                       uint32_t bytes,fg_error *err){
    if(self!=4u||!output||peer!=0u||fg_frame_request_id(header)!=session_id){
        fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output history");return FG_ERR_MISMATCH;
    }
    uint32_t count=0u;
    if(!output_history_count(payload,bytes,&count)){
        fg_error_set(err,FG_ERR_FORMAT,"invalid output history size");return FG_ERR_FORMAT;
    }
    uint32_t *tokens=count?malloc((size_t)count*sizeof(*tokens)):NULL;
    if(count&&!tokens){fg_error_set(err,FG_ERR_OOM,"allocate output history decode");return FG_ERR_OOM;}
    fg_output_history history={0};fg_status status=fg_output_history_decode(
        &history,tokens,count,payload,bytes,err);
    if(status==FG_OK)status=fg_output_history_reset(output,history.tokens,history.count,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,
        FG_MSG_OUTPUT_HISTORY_ACK,fg_frame_request_id(header),fg_frame_sequence(header),0u,
        NULL,0u,err);
    free(tokens);return status;
}

static fg_status rank_worker_loop(fg_fabric *fabric,fg_owner_executor *owner,fg_expert_executor *expert,fg_output_executor *output,fg_output_slice *output_slice,fg_output_hc *output_hc,uint32_t output_split_ways,const fg_ngram_resident *ngram,fg_model *model,const fg_manifest *manifest,const char *directory,uint32_t self,fg_error *err){
    uint32_t control_capacity=FG_LAYER_WORK_FOUR_AXIS_BASE_BYTES;if(control_capacity<FG_OUTPUT_WORK_BYTES)control_capacity=FG_OUTPUT_WORK_BYTES;if(control_capacity<FG_OUTPUT_HISTORY_MAX_BYTES)control_capacity=FG_OUTPUT_HISTORY_MAX_BYTES;if(control_capacity<FG_DECODE_WORK_BYTES)control_capacity=FG_DECODE_WORK_BYTES;if(control_capacity<FG_QSA_BLOCK_WORK_MAX_BYTES)control_capacity=FG_QSA_BLOCK_WORK_MAX_BYTES;uint8_t *control=malloc(control_capacity);prefill_worker_buffers prefill={0};qsa_owner_runtime qsa={0};layer_work_context layer_work={0};fg_vk_tensor *hyper=NULL;fg_output_handoff *handoff=NULL;
    /* Pre-allocate expert work buffers — eliminates ~200 KB malloc/free per expert request */
    fg_expert_result *ew_result=malloc(sizeof(*ew_result));uint8_t *ew_wire=malloc(FG_EXPERT_RESULT_SINGLE_BYTES);
    if(!control||!ew_result||!ew_wire){free(ew_wire);free(ew_result);free(control);fg_error_set(err,FG_ERR_OOM,"allocate rank worker buffers");return FG_ERR_OOM;}
    fg_status status=prefill_worker_buffers_create(&prefill,manifest->prefill_microbatch,
                                                   false,err);if(status==FG_OK)status=qsa_owner_runtime_create(&qsa,manifest,self,err);if(status==FG_OK&&owner)status=layer_work_context_create(&layer_work,model,owner,manifest,expert,&prefill,err);layer_work.qsa_owner=&qsa;layer_work.output_hc=output_hc;layer_work.output_split_ways=output_split_ways;
    uint32_t worker_qsa_layers=owned_qsa_layers(manifest,self);
    (void)worker_qsa_layers;
    if(status==FG_OK&&output)status=fg_vk_tensor_create(fg_model_vk(model),FG_HYPER_WIDTH*4u,&hyper,err);
    if(status==FG_OK&&output){handoff=calloc(1,sizeof(*handoff));if(!handoff){fg_error_set(err,FG_ERR_OOM,"allocate output handoff state");status=FG_ERR_OOM;}}
    if(status==FG_OK)status=token_profile_prepare(fg_model_vk(model),err);
    uint8_t *bulk_receive=prefill.receive;uint32_t bulk_capacity=prefill.receive_capacity;if(qsa.enabled&&qsa.receive_capacity>bulk_capacity){bulk_receive=qsa.receive_wire;bulk_capacity=qsa.receive_capacity;}if(layer_work.work_capacity>bulk_capacity){bulk_receive=layer_work.work_wire;bulk_capacity=layer_work.work_capacity;}uint64_t session_id=0;
    while(status==FG_OK){uint32_t peer=0,bytes=0;fg_frame_header header;fg_fabric_class ready_class;fg_fabric_recv_timing receive_timing={0};receive_timing.poll_start_ns=critical_ns();int32_t split_wait_ms=worker_output_split_pending(handoff)?worker_output_split_remaining_ms(handoff):-1;if(split_wait_ms==0){status=worker_output_split_timeout(handoff,self,err);break;}status=split_wait_ms>0?fg_fabric_wait_ready_timeout(fabric,3u,split_wait_ms,&peer,&ready_class,err):fg_fabric_wait_ready(fabric,3u,&peer,&ready_class,err);if(status==FG_ERR_LIMIT){if(worker_output_split_pending(handoff)&&worker_output_split_remaining_ms(handoff)==0)status=worker_output_split_timeout(handoff,self,err);else status=FG_OK;}receive_timing.ready_ns=critical_ns();if(status!=FG_OK)break;if(ready_class==FG_FABRIC_BULK){status=fg_fabric_recv_timed(fabric,peer,FG_FABRIC_BULK,&header,bulk_receive,bulk_capacity,&bytes,&receive_timing,err);fg_message_type type=status==FG_OK?fg_frame_type(&header):0;if(status==FG_OK&&type==FG_MSG_PREFILL_LAYER_WORK)status=handle_prefill_layer_work(fabric,owner,manifest,self,session_id,peer,&header,bulk_receive,bytes,&layer_work,err);else if(status==FG_OK&&type==FG_MSG_PREFILL_WORK)status=handle_prefill_expert_work(fabric,expert,manifest,self,session_id,peer,&header,bulk_receive,bytes,&prefill,err);else if(status==FG_OK&&type==FG_MSG_QSA_PAGE_APPEND)status=handle_qsa_page_append(&qsa,manifest,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK&&type==FG_MSG_QSA_PAGE_BARRIER)status=handle_qsa_page_barrier(fabric,&qsa,self,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK&&type==FG_MSG_QSA_PAGE_FETCH)status=handle_qsa_page_fetch(fabric,&qsa,owner,manifest,self,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK&&type==FG_MSG_GDN_STATE_FETCH)status=handle_gdn_state_fetch(fabric,owner,manifest,self,session_id,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK&&type==FG_MSG_DECODE_LAYER_WORK)status=handle_decode_layer_work(fabric,owner,manifest,self,session_id,peer,&header,bulk_receive,bytes,&layer_work,err);else if(status==FG_OK&&type==FG_MSG_OUTPUT_HIDDEN)status=handle_output_hidden(fabric,output,output_slice,fg_model_vk(model),manifest,self,session_id,peer,&header,bulk_receive,bytes,hyper,handoff,err);else if(status==FG_OK&&type==FG_MSG_OUTPUT_SLICE_HIDDEN)status=handle_output_slice_hidden(fabric,output,output_slice,fg_model_vk(model),manifest,self,session_id,peer,&header,bulk_receive,bytes,hyper,handoff,err);else if(status==FG_OK){fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported bulk message %u",self,type);status=FG_ERR_FORMAT;}continue;}status=fg_fabric_recv_timed(fabric,peer,FG_FABRIC_CONTROL,&header,control,control_capacity,&bytes,&receive_timing,err);if(status!=FG_OK)break;fg_message_type type=fg_frame_type(&header);if(type==FG_MSG_DECODE_WORK){if(!session_id||fg_frame_request_id(&header)!=session_id){fg_error_set(err,FG_ERR_MISMATCH,"stale expert work request");status=FG_ERR_MISMATCH;}else status=handle_expert_work(fabric,expert,fg_model_vk(model),self,peer,&header,control,bytes,&receive_timing,ew_result,ew_wire,err);    }else if(type==FG_MSG_NGRAM_WORK)status=handle_ngram_work(fabric,ngram,FG_FABRIC_BULK,self,session_id,peer,&header,control,bytes,err);else if(type==FG_MSG_SESSION_BEGIN){fg_output_handoff_reset(handoff);status=begin_session(fabric,manifest,directory,&qsa,owner,self,peer,&header,control,bytes,&session_id,output,err);}else if(type==FG_MSG_OUTPUT_HISTORY)status=handle_output_history(fabric,output,self,session_id,peer,&header,control,bytes,err);else if(type==FG_MSG_OUTPUT_WORK)status=handle_output_work(fabric,output,fg_model_vk(model),self,session_id,peer,&header,control,bytes,hyper,err);else if(type==FG_MSG_OUTPUT_CONFIG)status=handle_output_config(fabric,output,output_slice,fg_model_vk(model),manifest,self,session_id,peer,&header,control,bytes,hyper,handoff,err);else if(type==FG_MSG_OUTPUT_PARTIAL)status=handle_output_partial(fabric,output,output_slice,fg_model_vk(model),self,session_id,peer,&header,control,bytes,hyper,handoff,err);else{fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported control message %u",self,type);status=FG_ERR_FORMAT;}}
    fg_vk_tensor_destroy(hyper);free(handoff);layer_work_context_destroy(&layer_work);qsa_owner_runtime_destroy(&qsa);
    prefill_worker_buffers_destroy(&prefill);free(ew_wire);free(ew_result);free(control);return status;
}

fg_status fg_rank_main(const char *path,uint32_t rank,fg_error *err){
    if(rank>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,"rank must be 0..7");
        return FG_ERR_ARGUMENT;
    }
    fg_manifest *manifest=NULL;
    fg_status status=load_checked(path,&manifest,err);
    char directory[1024];
    if(status==FG_OK)status=manifest_directory(path,directory,err);
    if(status==FG_OK){
        fg_model *model=NULL;fg_expert_executor *expert=NULL;
        fg_owner_executor *owner=NULL;
        fg_output_executor *output=NULL;fg_ngram_resident *ngram=NULL;
        fg_fabric *fabric=NULL;uint64_t row_begin=0,row_count=0;fg_output_slice *output_slice=NULL;
        fg_output_hc *output_hc=NULL;uint32_t output_split_ways=0u;
        char ngram_path[1200];
        bool bench=block_bench_requested();
        status=fg_model_open(&model,manifest,directory,rank,err);
        if(status==FG_OK)status=fg_expert_executor_create(&expert,model,err);
        if(status==FG_OK&&(bench||worker_owner_enabled()))
            status=fg_owner_executor_create_worker(&owner,model,err);
        if(status==FG_OK&&bench){
            status=run_block_bench(model,expert,owner,manifest,err);
            fg_owner_executor_destroy(owner);
            fg_output_executor_destroy(output);fg_expert_executor_destroy(expert);
            fg_model_close(model);free(manifest);
            return status;
        }
        if(status==FG_OK)status=fg_output_split_mode(&output_split_ways,err);
        if(status==FG_OK&&rank==4u)
            status=fg_output_executor_create(&output,model,err);
        if(status==FG_OK&&output_split_ways){
            uint32_t way=0u,first_row=0u,rows=0u;
            if(rank==4u){
                if(!fg_output_split_way_for_rank(output_split_ways,rank,&way)){
                    fg_error_set(err,FG_ERR_MISMATCH,
                                 "rank %u owns no slice in the %u-way output split",rank,output_split_ways);
                    status=FG_ERR_MISMATCH;
                }else{
                    fg_output_split_span(output_split_ways,way,&first_row,&rows);
                    status=fg_output_slice_create(&output_slice,model,output_split_ways,
                                                  first_row,rows,err);
                }
            }else if(rank!=0u&&fg_output_split_way_for_rank(output_split_ways,rank,&way)){
                fg_output_split_span(output_split_ways,way,&first_row,&rows);
                status=fg_output_slice_create_foreign(&output_slice,model,directory,
                                                      output_split_ways,first_row,rows,err);
            }
            if(status==FG_OK&&output_split_ways==FG_OUTPUT_SPLIT_WAYS_MAX&&
               manifest->layer_owner[FG_LAYER_COUNT-1u]==rank)
                status=fg_output_hc_create(&output_hc,model,directory,true,err);
            if(status==FG_OK)
                fprintf(stderr,"OUTPUT_SPLIT rank=%u ways=%u slice=%u hc=%u\n",rank,
                        output_split_ways,output_slice!=NULL,output_hc!=NULL);
        }
        if(status==FG_OK)
            status=fg_q38_ngram_rank_range(rank,&row_begin,&row_count,err);
        if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),
           "%s/ngram-rank-%02u.iq4nl",directory,rank)>=(int)sizeof(ngram_path)){
            fg_error_set(err,FG_ERR_LIMIT,
                         "resident n-gram shard path is too long");
            status=FG_ERR_LIMIT;
        }
        if(status==FG_OK)
            status=fg_ngram_resident_open(&ngram,ngram_path,row_begin,row_count,err);
        if(status==FG_OK)status=fg_fabric_open(&fabric,manifest,rank,err);
        if(status==FG_OK)status=rank_ready(fabric,rank,err);
        if(status==FG_OK){
            printf("rank %u READY: %.3f GiB sealed weights, %.3f GiB n-gram "
                   "rows resident on %s\n",rank,
                   (double)fg_model_weight_bytes(model)/(1024.0*1024.0*1024.0),
                   (double)(row_count*FG_NGRAM_ROW_BYTES)/(1024.0*1024.0*1024.0),
                   fg_vk_device_name(fg_model_vk(model)));
            fflush(stdout);
            status=rank_worker_loop(fabric,owner,expert,output,output_slice,output_hc,
                                    output_split_ways,ngram,model,manifest,
                                    directory,rank,err);
        }
        fg_fabric_close(fabric);fg_ngram_resident_close(ngram);
        fg_owner_executor_destroy(owner);
        fg_output_hc_destroy(output_hc);
        fg_output_slice_destroy(output_slice);
        fg_output_executor_destroy(output);fg_expert_executor_destroy(expert);
        fg_model_close(model);
    }
    free(manifest);
    return status;
}

typedef struct qsa_page_transport {
    fg_qsa_page *append_pages[FG_RANK_COUNT],*fetch_pages,*result_pages;
    uint8_t *fetch_wire,*result_wire;
    fg_qsa_replica *replica;
    fg_fabric *fabric;
    atomic_uint *transport_state;
    uint32_t append_sequence[FG_RANK_COUNT],fetch_sequence[FG_RANK_COUNT],
             append_count[FG_RANK_COUNT],append_owner_mask,append_payload_bytes;
    uint32_t warm_outstanding;
} qsa_page_transport;

static void qsa_page_transport_destroy(qsa_page_transport *transport){
    if(!transport)return;
    fg_qsa_replica_destroy(transport->replica);
    free(transport->result_wire);free(transport->fetch_wire);free(transport->result_pages);
    free(transport->fetch_pages);
    for(uint32_t i=0;i<FG_RANK_COUNT;i++)free(transport->append_pages[i]);
    memset(transport,0,sizeof(*transport));
}

static fg_status qsa_replica_send(void *opaque,uint32_t owner,uint64_t session_id,
                                  uint32_t batch_id,const void *payload,uint32_t bytes,
                                  fg_error *err){
    qsa_page_transport *transport=opaque;
    fg_status status=fg_fabric_send_direct(transport->fabric,owner,FG_FABRIC_BULK,
        FG_MSG_QSA_PAGE_APPEND,session_id,batch_id,0,payload,bytes,err);
    if(status!=FG_OK)transport_poison(transport->transport_state);
    return status;
}

static fg_status qsa_page_transport_create(qsa_page_transport *transport,fg_fabric *fabric,
                                           atomic_uint *transport_state,
                                           fg_error *err){
    (void)err;
    memset(transport,0,sizeof(*transport));transport->fabric=fabric;
    transport->transport_state=transport_state;
    return FG_OK;
}

static fg_status qsa_page_transport_ensure(qsa_page_transport *transport,fg_error *err){
    if(!transport){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA page transport is null");
        return FG_ERR_ARGUMENT;
    }
    if(transport->replica)return FG_OK;
    for(uint32_t i=0;i<FG_RANK_COUNT;i++)
        transport->append_pages[i]=calloc(FG_QSA_PAGE_APPEND_MAX_PAGES,
                                          sizeof(*transport->append_pages[i]));
    transport->fetch_pages=calloc(FG_QSA_PAGE_FETCH_MAX_PAGES,
                                  sizeof(*transport->fetch_pages));
    transport->result_pages=calloc(FG_QSA_PAGE_FETCH_MAX_PAGES,
                                   sizeof(*transport->result_pages));
    transport->fetch_wire=malloc(FG_QSA_PAGE_FETCH_MAX_BYTES);
    transport->result_wire=malloc(FG_QSA_PAGE_RESULT_MAX_BYTES);
    bool append_pages_ready=true;
    for(uint32_t i=0;i<FG_RANK_COUNT;i++)
        if(!transport->append_pages[i])append_pages_ready=false;
    if(!append_pages_ready||!transport->fetch_pages||!transport->result_pages||
       !transport->fetch_wire||!transport->result_wire){
        qsa_page_transport_destroy(transport);
        fg_error_set(err,FG_ERR_OOM,"allocate fixed QSA page transport buffers");
        return FG_ERR_OOM;
    }
    uint32_t payload=transport->append_payload_bytes;
    if(!payload)payload=FG_QSA_PAGE_APPEND_MAX_BYTES;
    fg_status status=fg_qsa_replica_create(&transport->replica,qsa_replica_send,
                                           transport,payload,err);
    if(status!=FG_OK)qsa_page_transport_destroy(transport);
    return status;
}

#define FG_PREFILL_FRAMES 8u
typedef struct fg_coordinator {const fg_manifest *manifest;fg_runtime_options options;fg_session_identity identity;fg_model *model;fg_expert_executor *expert;fg_owner_executor *owner;fg_fabric *fabric;fg_ngram_store *ngram;fg_tokenizer *tokenizer;prefill_worker_buffers prefill_expert[FG_PREFILL_FRAMES];prefill_layer_buffers prefill_layer[FG_PREFILL_FRAMES];qsa_page_transport qsa_pages;uint64_t session_id;uint8_t *async_recv_payloads[FG_GROUP_SIZE];const char *directory;atomic_uint transport_state;fg_sampler_config sampler;fg_sampler_state sampler_state;bool ring_prefill;fg_vk_tensor *ring_output[FG_PREFILL_FRAMES];bool ring_decode;uint8_t *decode_work_wire,*decode_result_wire;fg_layer_work decode_work;fg_layer_result decode_result;fg_output_slice *output_slice;} fg_coordinator;

static uint64_t coordinator_prefill_host_bytes(const prefill_worker_buffers *buffers){
    if(!buffers)return 0;
    uint64_t bytes=buffers->receive?buffers->receive_capacity:0;
    if(buffers->activations)
        bytes+=(uint64_t)buffers->token_capacity*FG_Q8K_ACTIVATION_BYTES;
    bytes+=buffers->result_wire?buffers->result_capacity:0;
    if(buffers->pair_storage_owned)
        bytes+=prefill_pair_storage_bytes(buffers->pair_capacity);
    if(buffers->outputs_owned)
        bytes+=(uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE*sizeof(*buffers->outputs);
    return bytes;
}

static uint64_t coordinator_transport_host_bytes(const qsa_page_transport *transport){
    if(!transport||!transport->replica)return 0;
    return (uint64_t)FG_RANK_COUNT*FG_QSA_PAGE_APPEND_MAX_PAGES*sizeof(fg_qsa_page)+
           2u*(uint64_t)FG_QSA_PAGE_FETCH_MAX_PAGES*sizeof(fg_qsa_page)+
           FG_QSA_PAGE_FETCH_MAX_BYTES+FG_QSA_PAGE_RESULT_MAX_BYTES+
           fg_qsa_replica_host_bytes(transport->replica);
}

/* One publish batch carries at most one page run per QSA layer a single owner
 * can hold, so the replica slot payload only needs the sealed microbatch's
 * page count instead of the 512-token protocol ceiling. */
static uint32_t coordinator_qsa_append_payload_bytes(uint32_t tokens){
    uint64_t blocks=((uint64_t)tokens+FG_Q38_QSA_COMPRESS_RATIO-1u)/
        FG_Q38_QSA_COMPRESS_RATIO;
    uint64_t bytes=FG_QSA_PAGE_BATCH_HEADER_BYTES+
        (uint64_t)FG_QSA_OWNER_LAYER_COUNT*blocks*FG_QSA_PAGE_ENTRY_BYTES;
    if(bytes>FG_QSA_PAGE_APPEND_MAX_BYTES)bytes=FG_QSA_PAGE_APPEND_MAX_BYTES;
    if(!bytes)bytes=1u;
    return (uint32_t)bytes;
}

static uint64_t coordinator_transport_capacity_bytes(uint32_t append_payload_bytes){
    return (uint64_t)FG_RANK_COUNT*FG_QSA_PAGE_APPEND_MAX_PAGES*sizeof(fg_qsa_page)+
           2u*(uint64_t)FG_QSA_PAGE_FETCH_MAX_PAGES*sizeof(fg_qsa_page)+
           FG_QSA_PAGE_FETCH_MAX_BYTES+FG_QSA_PAGE_RESULT_MAX_BYTES+
           fg_qsa_replica_host_bytes_for_capacity(append_payload_bytes);
}

static uint64_t coordinator_prefill_work_wire_bytes(uint32_t tokens){
    return FG_PREFILL_WORK_HEADER_BYTES+
        (uint64_t)tokens*FG_Q8K_ACTIVATION_BYTES+
        (uint64_t)tokens*FG_TOP_K*FG_PREFILL_PAIR_BYTES;
}

static uint32_t coordinator_qsa_cache_pages(const fg_runtime_options *options){
    if(!options)return 0u;
    if(options->qsa_page_cache_bytes)
        return (uint32_t)(options->qsa_page_cache_bytes/FG_QSA_PAGE_RECORD_BYTES);
    return options->qsa_hot_tokens/FG_Q38_QSA_COMPRESS_RATIO*
        (FG_LAYER_COUNT/4u);
}

#define FG_COORDINATOR_HC_DOWN_SPLITS 8u

static uint64_t coordinator_physical_memory_bytes(void){
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages=sysconf(_SC_PHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
    if(pages>0&&page_size>0&&
       (uint64_t)pages<=UINT64_MAX/(uint64_t)page_size)
        return (uint64_t)pages*(uint64_t)page_size;
#endif
    return 0;
}

static uint64_t coordinator_saturating_add(uint64_t left,uint64_t right){
    return left>UINT64_MAX-right?UINT64_MAX:left+right;
}

static int64_t coordinator_margin(uint64_t physical,uint64_t used){
    if(physical>=used){
        uint64_t value=physical-used;
        return value>INT64_MAX?INT64_MAX:(int64_t)value;
    }
    uint64_t value=used-physical;
    return value>INT64_MAX?INT64_MIN:-(int64_t)value;
}

static uint64_t coordinator_process_rss_bytes(void){
#if defined(__linux__)
    FILE *file=fopen("/proc/self/statm","r");
    unsigned long long pages=0,resident=0;
    if(file&&fscanf(file,"%llu %llu",&pages,&resident)==2){
        fclose(file);
        long page_size=sysconf(_SC_PAGESIZE);
        if(page_size>0&&resident<=UINT64_MAX/(uint64_t)page_size)
            return (uint64_t)resident*(uint64_t)page_size;
    }else if(file)fclose(file);
#endif
    return 0;
}

static uint64_t coordinator_available_memory_bytes(void){
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages=sysconf(_SC_AVPHYS_PAGES),page_size=sysconf(_SC_PAGESIZE);
    if(pages>0&&page_size>0&&
       (uint64_t)pages<=UINT64_MAX/(uint64_t)page_size)
        return (uint64_t)pages*(uint64_t)page_size;
#endif
    return 0;
}

static uint64_t coordinator_owner_transient_reclaim(uint32_t tokens,
                                                    uint32_t hc_inject_pieces){
    return (uint64_t)tokens*10240u*4u+(uint64_t)tokens*320u*4u+
           (uint64_t)FG_COORDINATOR_HC_DOWN_SPLITS*320u*4u+
               (uint64_t)tokens*320u*4u+(uint64_t)tokens*10240u*4u+
           (uint64_t)hc_inject_pieces*4u*
               tokens*4u+(uint64_t)tokens*2560u*4u+(uint64_t)tokens*4u+
           (uint64_t)tokens*FG_EXPERT_COUNT*4u+(uint64_t)tokens*640u*4u*3u+
           (uint64_t)tokens*FG_HIDDEN_SIZE*4u+(uint64_t)tokens*4u+
           (uint64_t)tokens*FG_HIDDEN_SIZE*4u;
}

static uint64_t coordinator_qsa_aux_reclaim(uint32_t logical_context){
    uint64_t blocks=((uint64_t)logical_context+FG_Q38_QSA_COMPRESS_RATIO-1u)/
                    FG_Q38_QSA_COMPRESS_RATIO;
    return 4u*blocks*sizeof(uint32_t)+
           (uint64_t)FG_QSA_SELECTED_TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES;
}

static uint64_t coordinator_qsa_deferred_host_bytes(const fg_coordinator *coordinator,
                                                    uint32_t cache_pages){
    if(!coordinator)return 0;
    uint64_t expected=(uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*
        FG_QSA_PAGE_RECORD_BYTES+
        fg_qsa_page_cache_memory_bytes_for_pages(cache_pages);
    uint64_t actual=fg_owner_qsa_host_bytes(coordinator->owner);
    return expected>actual?expected-actual:0;
}

static uint64_t coordinator_ngram_deferred_host_bytes(const fg_coordinator *coordinator){
    if(!coordinator||!coordinator->ngram)return 0;
    uint64_t allocated=fg_ngram_store_cache_host_bytes(coordinator->ngram);
    uint64_t expected=fg_ngram_cache_memory_bytes();
    return expected>allocated?expected-allocated:0;
}

static void coordinator_memory_report(const fg_coordinator *coordinator){
    if(!coordinator||!coordinator->model)return;
    fg_vk_memory_stats vk={0};
    fg_vk_get_memory_stats(fg_model_vk(coordinator->model),&vk);
    uint64_t prefill_host=0;
    for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++)prefill_host+=coordinator_prefill_host_bytes(&coordinator->prefill_expert[f]);
    uint64_t async_host=(uint64_t)FG_GROUP_SIZE*FG_EXPERT_RESULT_SINGLE_BYTES;
    uint64_t positions_host=0;
    for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++)positions_host+=(uint64_t)coordinator->prefill_layer[f].tokens*3u*sizeof(uint32_t);
    uint64_t transport_host=coordinator_transport_host_bytes(&coordinator->qsa_pages);
    uint64_t fabric_host=fg_fabric_host_bytes(coordinator->fabric);
    uint64_t ngram_host=fg_ngram_store_host_bytes(coordinator->ngram);
    uint64_t ngram_io=fg_ngram_store_io_host_bytes(coordinator->ngram);
    uint64_t ngram_cache=fg_ngram_store_cache_host_bytes(coordinator->ngram);
    uint64_t qsa_host=fg_owner_qsa_host_bytes(coordinator->owner);
    uint64_t host=prefill_host+async_host+positions_host+transport_host+
        fabric_host+ngram_host+qsa_host;
    uint64_t physical=coordinator_physical_memory_bytes();
    uint64_t logical=coordinator->options.logical_context_tokens;
    uint32_t cache_page_count=coordinator_qsa_cache_pages(&coordinator->options);
    bool ring=coordinator->ring_prefill&&coordinator->ring_decode;
    uint32_t qsa_layers=0,gdn_layers=0;bool ple_state=false;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        bool owned=coordinator->manifest->layer_owner[layer]==0u||!ring;
        if((layer&3u)==3u){if(owned)qsa_layers++;}
        else if(owned)gdn_layers++;
        if(layer==1u&&owned)ple_state=true;
    }
    /* Index segments after the first are lazy, so the startup ledger counts
     * only the eager first segment per QSA layer. */
    uint64_t index=(uint64_t)fg_qsa_index_segment_tokens(logical,0u)*qsa_layers*
        FG_Q38_QSA_INDEX_KEY_BYTES;
    uint64_t record_cache=(uint64_t)cache_page_count*FG_QSA_PAGE_RECORD_BYTES;
    uint32_t ngram_store_tokens=coordinator->manifest->prefill_microbatch<=FG_NGRAM_PREFILL_MAX_TOKENS/FG_PREFILL_FRAMES?FG_PREFILL_FRAMES*coordinator->manifest->prefill_microbatch:FG_NGRAM_PREFILL_MAX_TOKENS;
    uint64_t ngram_vk_capacity=(uint64_t)ngram_store_tokens*
        FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES+
        (uint64_t)ngram_store_tokens*
            FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH*4u;
    uint64_t ngram_vk=fg_ngram_store_vk_bytes(coordinator->ngram);
    uint64_t qsa_aux=fg_qsa_selection_scratch_bytes(
        logical,coordinator->manifest->prefill_microbatch);
    uint64_t owner_family=fg_qsa_attention_family_scratch_bytes(
        coordinator->manifest->prefill_microbatch);
    uint64_t owner_state=(uint64_t)gdn_layers*
        (10240u*4u*4u+48u*128u*128u*4u)+(ple_state?10240u*9u*4u:0u);
    uint64_t owner_pingpong=(uint64_t)coordinator->manifest->prefill_microbatch*
        10240u*4u*2u;
    uint64_t owner_activation=(uint64_t)coordinator->manifest->prefill_microbatch*
        FG_Q8K_ACTIVATION_BYTES;
    uint64_t expert_activation=owner_activation;
    uint64_t qsa_positions=logical*FG_Q38_QSA_POSITION_BYTES;
    uint64_t qsa_index=(uint64_t)fg_qsa_index_segment_tokens(logical,0u)*
        qsa_layers*FG_Q38_QSA_INDEX_KEY_BYTES;
    uint64_t qsa_record_cache=record_cache;
    uint64_t prefill_token=(uint64_t)coordinator->prefill_layer[0].tokens*
        sizeof(uint32_t)*FG_PREFILL_FRAMES;
    uint64_t known=fg_model_weight_bytes(coordinator->model)+owner_family+
        owner_state+owner_pingpong+owner_activation+expert_activation+
        qsa_positions+qsa_index+qsa_record_cache+ngram_vk+prefill_token;
    fprintf(stderr,
        "COORDINATOR_VK_COMPONENTS model=%llu family=%llu owner_state=%llu "
        "pingpong=%llu owner_activation=%llu expert_activation=%llu "
        "qsa_positions=%llu qsa_index=%llu qsa_record_cache=%llu ngram=%llu "
        "prefill_token=%llu known_total=%llu unclassified=%lld\n",
        (unsigned long long)fg_model_weight_bytes(coordinator->model),
        (unsigned long long)owner_family,(unsigned long long)owner_state,
        (unsigned long long)owner_pingpong,(unsigned long long)owner_activation,
        (unsigned long long)expert_activation,(unsigned long long)qsa_positions,
        (unsigned long long)qsa_index,(unsigned long long)qsa_record_cache,
        (unsigned long long)ngram_vk,(unsigned long long)prefill_token,
        (unsigned long long)known,
        (long long)vk.requested_live_bytes-(long long)known);
    fprintf(stderr,
        "COORDINATOR_VK_LEDGER model_arena=%llu qsa_index=%llu qsa_record_cache=%llu "
        "ngram_payload=%llu qsa_aux_shared=%llu final_requested=%llu "
        "final_allocated=%llu live_allocations=%llu\n",
        (unsigned long long)fg_model_weight_bytes(coordinator->model),
        (unsigned long long)index,(unsigned long long)record_cache,
        (unsigned long long)ngram_vk,(unsigned long long)qsa_aux,
        (unsigned long long)vk.requested_live_bytes,
        (unsigned long long)vk.allocated_live_bytes,
        (unsigned long long)vk.live_allocations);
    fprintf(stderr,
        "COORDINATOR_HOST_LEDGER prefill=%llu async=%llu positions=%llu "
        "transport=%llu fabric=%llu ngram=%llu ngram_io=%llu ngram_cache=%llu qsa=%llu "
        "startup_total=%llu\n",
        (unsigned long long)prefill_host,(unsigned long long)async_host,
        (unsigned long long)positions_host,(unsigned long long)transport_host,
        (unsigned long long)fabric_host,(unsigned long long)ngram_host,
        (unsigned long long)ngram_io,
        (unsigned long long)ngram_cache,(unsigned long long)qsa_host,
        (unsigned long long)host);
    uint64_t deferred_ngram_vk=ngram_vk_capacity>ngram_vk?
        ngram_vk_capacity-ngram_vk:0u;
    uint64_t deferred_qsa_host=coordinator_qsa_deferred_host_bytes(
        coordinator,cache_page_count);
    uint64_t deferred_ngram_host=coordinator_ngram_deferred_host_bytes(coordinator);
    uint64_t deferred_prefill_wire=0;
    for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++)deferred_prefill_wire+=
        coordinator->prefill_expert[f].result_wire?0u:coordinator->prefill_expert[f].result_capacity;
    uint64_t deferred_prefill_work=coordinator_prefill_work_wire_bytes(
        coordinator->manifest->prefill_microbatch);
    uint64_t deferred_transport=(coordinator->qsa_pages.replica||
        (coordinator->ring_prefill&&coordinator->ring_decode))?
        0u:coordinator_transport_capacity_bytes(
            coordinator_qsa_append_payload_bytes(
                coordinator->manifest->prefill_microbatch));
    uint64_t deferred_host=0;
    deferred_host=coordinator_saturating_add(deferred_host,deferred_qsa_host);
    deferred_host=coordinator_saturating_add(deferred_host,deferred_ngram_host);
    deferred_host=coordinator_saturating_add(deferred_host,deferred_prefill_wire);
    deferred_host=coordinator_saturating_add(deferred_host,deferred_prefill_work);
    deferred_host=coordinator_saturating_add(deferred_host,deferred_transport);
    uint64_t deferred_peak=coordinator_saturating_add(deferred_host,
                                                      deferred_ngram_vk);
    uint64_t deferred_peak_with_persistent_io=coordinator_saturating_add(
        deferred_peak,ngram_io);
    uint64_t startup_used=coordinator_saturating_add(vk.allocated_live_bytes,host);
    uint64_t projected_peak=coordinator_saturating_add(startup_used,deferred_peak);
    uint64_t startup_requested=coordinator_saturating_add(vk.requested_live_bytes,host);
    uint64_t projected_requested=coordinator_saturating_add(startup_requested,
                                                            deferred_peak);
    uint64_t owner_transient=coordinator_owner_transient_reclaim(
        coordinator->manifest->prefill_microbatch,
        fg_vk_hc_inject_pieces(fg_model_vk(coordinator->model)));
    uint64_t qsa_reclaim=coordinator_qsa_aux_reclaim(logical);
    uint64_t layer_reclaim=(uint64_t)coordinator->manifest->prefill_microbatch*
        (FG_HYPER_WIDTH*4u+FG_NGRAM_EMBED_VALUES*4u);
    uint64_t vulkan_reclaim=owner_transient+qsa_reclaim+layer_reclaim+
        FG_HYPER_WIDTH*sizeof(float)+deferred_ngram_vk;
    uint64_t prefill_host_reclaim=(uint64_t)FG_PREFILL_FRAMES*(FG_PREFILL_WORK_HEADER_BYTES+
        (uint64_t)coordinator->prefill_expert[0].token_capacity*
            (2u*FG_Q8K_ACTIVATION_BYTES+
             FG_TOP_K*sizeof(fg_prefill_pair)+
             FG_TOP_K*sizeof(fg_prefill_result_pair)+
             FG_TOP_K*FG_HIDDEN_SIZE*sizeof(float)));
    uint64_t expert_recv_reclaim=FG_EXPERT_RESULT_MAX_BYTES;
    uint64_t async_host_reclaim=(uint64_t)FG_GROUP_SIZE*
        (FG_EXPERT_RESULT_MAX_BYTES-FG_EXPERT_RESULT_SINGLE_BYTES);
    uint64_t positions_host_reclaim=0;
    for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++)positions_host_reclaim+=(uint64_t)coordinator->prefill_layer[f].tokens*sizeof(uint32_t);
    fprintf(stderr,
        "COORDINATOR_DEFERRED_LEDGER qsa_staging_cache=%llu ngram_cache=%llu "
        "prefill_result_wire=%llu prefill_work_wire=%llu transport=%llu "
        "ngram_io_persistent=%llu ngram_vk=%llu host_total=%llu "
        "peak_with_persistent_io=%llu "
        "peak_total=%llu\n",
        (unsigned long long)deferred_qsa_host,
        (unsigned long long)deferred_ngram_host,
        (unsigned long long)deferred_prefill_wire,
        (unsigned long long)deferred_prefill_work,
        (unsigned long long)deferred_transport,
        (unsigned long long)ngram_io,
        (unsigned long long)deferred_ngram_vk,
        (unsigned long long)deferred_host,
        (unsigned long long)deferred_peak_with_persistent_io,
        (unsigned long long)deferred_peak);
    fprintf(stderr,
        "COORDINATOR_RECLAIM vulkan=%llu owner_transient=%llu qsa_aux=%llu "
        "prefill_layer=%llu ngram_lazy=%llu positions_host=%llu\n",
        (unsigned long long)vulkan_reclaim,
        (unsigned long long)owner_transient,
        (unsigned long long)qsa_reclaim,(unsigned long long)layer_reclaim,
        (unsigned long long)deferred_ngram_vk,
        (unsigned long long)positions_host_reclaim);
    fprintf(stderr,
        "COORDINATOR_HOST_RECLAIM prefill=%llu expert_recv=%llu "
        "async_compact=%llu positions=%llu\n",
        (unsigned long long)prefill_host_reclaim,
        (unsigned long long)expert_recv_reclaim,
        (unsigned long long)async_host_reclaim,
        (unsigned long long)positions_host_reclaim);
    uint64_t driver_reserve=coordinator->manifest->ranks[0].driver_reserve_bytes;
    uint64_t projected_peak_with_driver=coordinator_saturating_add(
        projected_peak,driver_reserve);
    uint64_t projected_requested_with_driver=coordinator_saturating_add(
        projected_requested,driver_reserve);
    uint64_t readiness_peak=projected_peak_with_driver>
        projected_requested_with_driver?projected_peak_with_driver:
        projected_requested_with_driver;
    uint64_t available=coordinator_available_memory_bytes();
    uint64_t process_rss=coordinator_process_rss_bytes();
    const char *readiness=!physical?"unknown-physical":
        readiness_peak>physical?"insufficient":"unknown-os-overhead";
    fprintf(stderr,
        "COORDINATOR_MEMORY_LEDGER host_startup=%llu host_peak=%llu "
        "startup_used=%llu deferred_peak=%llu projected_peak=%llu "
        "driver_reserve_bytes=%llu projected_peak_with_driver=%llu "
        "projected_requested=%llu projected_requested_with_driver=%llu "
        "physical_bytes=%llu available_bytes=%llu "
        "process_rss_bytes=%llu conservative_peak_margin=%lld readiness=%s "
        "unknown_os_overhead=unmeasured uma_vulkan_counted_once=1\n",
        (unsigned long long)host,
        (unsigned long long)coordinator_saturating_add(host,deferred_host),
        (unsigned long long)startup_used,(unsigned long long)deferred_peak,
        (unsigned long long)projected_peak,
        (unsigned long long)driver_reserve,
        (unsigned long long)projected_peak_with_driver,
        (unsigned long long)projected_requested,
        (unsigned long long)projected_requested_with_driver,
        (unsigned long long)physical,(unsigned long long)available,
        (unsigned long long)process_rss,
        (long long)coordinator_margin(physical,readiness_peak),
        readiness);
}

static fg_status coordinator_publish_qsa_pages(fg_coordinator *coordinator,uint32_t first_token,
                                                uint32_t token_count,fg_error *err){
    qsa_page_transport *transport=&coordinator->qsa_pages;
    fg_vk_context *vk=fg_model_vk(coordinator->model);
    fg_status status=FG_OK;
    while(status==FG_OK&&fg_vk_batch_active(vk))status=fg_vk_end(vk,err);
    /* Ring prefill commits every complete page in the block that owns it, so
     * rank 0 never publishes or mirrors remote pages here; leave the transport
     * unallocated until a non-ring path actually needs it. */
    transport->append_payload_bytes=coordinator_qsa_append_payload_bytes(
        coordinator->manifest->prefill_microbatch);
    if(status==FG_OK&&coordinator->ring_prefill)return status;
    if(status==FG_OK)status=qsa_page_transport_ensure(transport,err);
    if(status!=FG_OK)return status;
    status=fg_qsa_replica_status(transport->replica,err);
    if(status!=FG_OK)return status;
    memset(transport->append_count,0,sizeof(transport->append_count));
    uint32_t first_block=0,block_count=0;
    status=fg_qsa_completed_page_range(first_token,token_count,&first_block,&block_count,err);
    if(status!=FG_OK)return status;
    for(uint32_t offset=0;offset<block_count;offset++){
        uint32_t block=first_block+offset;
        for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u){
            uint32_t owner=coordinator->manifest->layer_owner[layer];
            if(owner>=FG_RANK_COUNT){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA page owner rank %u is invalid",owner);
                return FG_ERR_MISMATCH;
            }
            if(owner==0u){
                /* Self-owned pages stay in the local mirror session for decode. */
                continue;
            }
            if(coordinator->ring_prefill)continue; /* block owners publish their own pages */
            uint32_t index=transport->append_count[owner]++;
            if(index>=FG_QSA_PAGE_APPEND_MAX_PAGES){
                fg_error_set(err,FG_ERR_LIMIT,"QSA complete-page append batch overflow");
                return FG_ERR_LIMIT;
            }
            const uint8_t *records=NULL;
            status=fg_owner_qsa_page_records(coordinator->owner,layer,block,&records,err);
            if(status!=FG_OK)return status;
            transport->append_pages[owner][index]=(fg_qsa_page){
                .layer=(uint8_t)layer,.block=block,.records=records};
        }
    }
    uint32_t send_count=0,owner_mask=0;
    for(uint32_t owner=1u;owner<FG_RANK_COUNT;owner++)
        if(transport->append_count[owner]){send_count++;owner_mask|=1u<<owner;}
    if(!send_count)return FG_OK;
    uint8_t *buffers[FG_RANK_COUNT]={0};
    status=fg_qsa_replica_reserve(transport->replica,send_count,buffers,err);
    if(status!=FG_OK)return status;
    fg_qsa_replica_item items[FG_RANK_COUNT]={0};uint32_t send_index=0;
    for(uint32_t owner=1u;owner<FG_RANK_COUNT;owner++){
        uint32_t count=transport->append_count[owner];
        if(!count)continue;
        fg_qsa_page_batch batch={.source_rank=0u,.destination_rank=(uint8_t)owner,
            .batch_id=transport->append_sequence[owner],.page_count=(uint16_t)count,
            .pages=transport->append_pages[owner]};
        uint32_t bytes=0;status=fg_qsa_page_append_encode(buffers[send_index],
            transport->append_payload_bytes,&bytes,&batch,err);
        if(status!=FG_OK){fg_qsa_replica_cancel(transport->replica);return status;}
        items[send_index++]=(fg_qsa_replica_item){.owner=owner,.batch_id=batch.batch_id,
            .bytes=bytes,.session_id=coordinator->session_id};
    }
    status=fg_qsa_replica_commit(transport->replica,items,send_count,err);
    if(status!=FG_OK){fg_qsa_replica_cancel(transport->replica);return status;}
    /* The queue now owns copies of every page. Failed reservation/encoding/
     * commit must leave the source pages pinned for reset or retry. */
    for(uint32_t owner=1u;owner<FG_RANK_COUNT;owner++)
        for(uint32_t i=0;i<transport->append_count[owner];i++){
            const fg_qsa_page *page=&transport->append_pages[owner][i];
            fg_owner_qsa_page_published(coordinator->owner,page->layer,page->block);
        }
    for(uint32_t owner=1u;owner<FG_RANK_COUNT;owner++)
        if(transport->append_count[owner])transport->append_sequence[owner]++;
    transport->append_owner_mask|=owner_mask;
    return status;
}

static fg_status coordinator_fetch_qsa_pages(void *opaque,uint32_t layer,
                                             const uint32_t *blocks,uint32_t block_count,
                                             uint8_t *records,fg_error *err){
    fg_coordinator *coordinator=opaque;qsa_page_transport *transport=&coordinator->qsa_pages;
    if(!blocks||!records||!block_count||block_count>FG_QSA_PAGE_FETCH_MAX_PAGES||
       layer>=FG_LAYER_COUNT||(layer&3u)!=3u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid coordinator QSA page fetch");
        return FG_ERR_ARGUMENT;
    }
    transport->append_payload_bytes=coordinator_qsa_append_payload_bytes(
        coordinator->manifest->prefill_microbatch);
    fg_status status=qsa_page_transport_ensure(transport,err);
    if(status!=FG_OK)return status;
    uint32_t owner=coordinator->manifest->layer_owner[layer];
    if(owner==0u){
        /* Rank 0 owns the block: its own state file is authoritative for the
         * complete pages that already missed the record cache.  One batched
         * read replaces a synchronous pread per page. */
        return fg_owner_qsa_state_records_batch(coordinator->owner,layer,blocks,
                                                block_count,records,err);
    }
    if(owner>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA cold page has no authoritative owner");
        return FG_ERR_MISMATCH;
    }
    status=fg_qsa_replica_drain(transport->replica,err);
    if(status!=FG_OK)return status;
    for(uint32_t i=0;i<block_count;i++)transport->fetch_pages[i]=(fg_qsa_page){
        .layer=(uint8_t)layer,.block=blocks[i],.records=NULL};
    fg_qsa_page_batch request={.source_rank=0u,.destination_rank=(uint8_t)owner,
        .batch_id=transport->fetch_sequence[owner],.page_count=(uint16_t)block_count,
        .pages=transport->fetch_pages};
    uint32_t bytes=0;status=fg_qsa_page_fetch_encode(transport->fetch_wire,
        FG_QSA_PAGE_FETCH_MAX_BYTES,&bytes,&request,err);
    if(status==FG_OK){
        status=fg_fabric_send(coordinator->fabric,owner,FG_FABRIC_BULK,
            FG_MSG_QSA_PAGE_FETCH,coordinator->session_id,request.batch_id,0,
            transport->fetch_wire,bytes,err);
        if(status==FG_OK)transport_pending(&coordinator->transport_state);
        else transport_poison(&coordinator->transport_state);
    }
    fg_frame_header header;uint32_t result_bytes=0;
    if(status==FG_OK)status=fg_fabric_recv(coordinator->fabric,owner,FG_FABRIC_BULK,
        &header,transport->result_wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&result_bytes,err);
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_QSA_PAGE_RESULT||
       fg_frame_request_id(&header)!=coordinator->session_id||
       fg_frame_sequence(&header)!=request.batch_id)){
        fg_error_set(err,FG_ERR_MISMATCH,"stale QSA cold-page result from rank %u",owner);
        status=FG_ERR_MISMATCH;
    }
    fg_qsa_page_batch result={0};
    if(status==FG_OK)status=fg_qsa_page_result_decode(&result,transport->result_pages,
        FG_QSA_PAGE_FETCH_MAX_PAGES,transport->result_wire,result_bytes,err);
    if(status==FG_OK&&(result.source_rank!=owner||result.destination_rank!=0u||
       result.batch_id!=request.batch_id||result.page_count!=block_count)){
        fg_error_set(err,FG_ERR_MISMATCH,"misrouted QSA cold-page result");
        status=FG_ERR_MISMATCH;
    }
    for(uint32_t i=0;status==FG_OK&&i<block_count;i++){
        if(result.pages[i].layer!=layer||result.pages[i].block!=blocks[i]){
            fg_error_set(err,FG_ERR_MISMATCH,"out-of-order QSA cold-page result entry");
            status=FG_ERR_MISMATCH;break;
        }
        memcpy(records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,result.pages[i].records,
               FG_QSA_PAGE_RECORD_BYTES);
    }
    if(status==FG_OK){
        transport->fetch_sequence[owner]++;
        transport_complete(&coordinator->transport_state);
    }else if(!transport_ready(&coordinator->transport_state)){
        transport_poison(&coordinator->transport_state);
    }
    return status;
}

static bool qsa_warm_trace_enabled(void){
    const char *value=getenv("FG_QSA_WARM_TRACE");
    return value&&*value&&strcmp(value,"0")!=0;
}

static void coordinator_trace_qsa_cache(fg_coordinator *coordinator,const char *tag,
                                        uint32_t complete_blocks){
    if(!coordinator||!coordinator->owner||!qsa_warm_trace_enabled())return;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u){
        uint32_t resident=0;
        for(uint32_t block=0;block<complete_blocks;block++)
            if(fg_owner_qsa_page_cached(coordinator->owner,layer,block))resident++;
        fprintf(stderr,"QSA_CACHE_TRACE tag=%s layer=%u owner=%u resident=%u of=%u\n",
            tag,layer,(unsigned)coordinator->manifest->layer_owner[layer],
            resident,complete_blocks);
    }
}

static void coordinator_warm_qsa_block_range(uint32_t first_token,uint32_t token_count,
                                             uint32_t *first_block,uint32_t *block_count){
    uint64_t begin=((uint64_t)first_token+FG_Q38_QSA_COMPRESS_RATIO-1u)/
        FG_Q38_QSA_COMPRESS_RATIO;
    uint64_t end=((uint64_t)first_token+token_count)/FG_Q38_QSA_COMPRESS_RATIO;
    if(end<begin)end=begin;
    *first_block=(uint32_t)begin;
    *block_count=(uint32_t)(end-begin);
}

/* Mirror warm: request every worker-owned QSA layer's newly committed complete
 * pages as ring chunks retire.  Requests are fire-and-forget, so the network
 * fetches overlap the remaining chain instead of serializing on the first
 * decode token; results are handled inline by the ring receive loop. */
static fg_status coordinator_warm_qsa_issue(fg_coordinator *coordinator,
                                            uint32_t first_token,uint32_t token_count,
                                            fg_error *err){
    if(!coordinator->ring_prefill||!token_count)return FG_OK;
    /* Ring decode runs every QSA layer on its owner, so the rank-0 mirror is
     * never read for remote-owned pages.  Warming it only pulls fabric traffic
     * and evicts rank 0's own prefill pages, which decode then reads back from
     * the state file. */
    if(coordinator->ring_decode)return FG_OK;
    qsa_page_transport *transport=&coordinator->qsa_pages;
    transport->append_payload_bytes=coordinator_qsa_append_payload_bytes(
        coordinator->manifest->prefill_microbatch);
    fg_status status=qsa_page_transport_ensure(transport,err);
    if(status!=FG_OK)return status;
    uint32_t first_block=0,block_count=0;
    coordinator_warm_qsa_block_range(first_token,token_count,&first_block,&block_count);
    if(!block_count)return FG_OK;
    for(uint32_t layer=3u;status==FG_OK&&layer<FG_LAYER_COUNT;layer+=4u){
        uint32_t owner=coordinator->manifest->layer_owner[layer];
        if(owner==0u)continue; /* rank-0 pages stay in the local mirror session */
        if(owner>=FG_RANK_COUNT){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA warm layer %u has an invalid owner",layer);
            return FG_ERR_MISMATCH;
        }
        for(uint32_t offset=0;status==FG_OK&&offset<block_count;){
            uint32_t count=block_count-offset;
            if(count>FG_QSA_PAGE_FETCH_MAX_PAGES)count=FG_QSA_PAGE_FETCH_MAX_PAGES;
            for(uint32_t i=0;i<count;i++)
                transport->fetch_pages[i]=(fg_qsa_page){
                    .layer=(uint8_t)layer,.block=first_block+offset+i,.records=NULL};
            fg_qsa_page_batch request={.source_rank=0u,.destination_rank=(uint8_t)owner,
                .batch_id=transport->fetch_sequence[owner],.page_count=(uint16_t)count,
                .pages=transport->fetch_pages};
            uint32_t bytes=0;
            status=fg_qsa_page_fetch_encode(transport->fetch_wire,
                FG_QSA_PAGE_FETCH_MAX_BYTES,&bytes,&request,err);
            if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,owner,FG_FABRIC_BULK,
                FG_MSG_QSA_PAGE_FETCH,coordinator->session_id,request.batch_id,0,
                transport->fetch_wire,bytes,err);
            if(status==FG_OK){
                transport_pending(&coordinator->transport_state);
                transport->fetch_sequence[owner]++;
                transport->warm_outstanding++;
            }else transport_poison(&coordinator->transport_state);
            offset+=count;
        }
    }
    if(status==FG_OK&&qsa_warm_trace_enabled())
        fprintf(stderr,"QSA_WARM_ISSUE first=%u tokens=%u blocks=%u..%u outstanding=%u t=%.3f\n",
            first_token,token_count,first_block,first_block+block_count-1u,
            transport->warm_outstanding,dispatch_ts());
    return status;
}

static fg_status coordinator_warm_qsa_result(fg_coordinator *coordinator,uint32_t peer,
                                             const fg_frame_header *header,
                                             const uint8_t *wire,uint32_t bytes,
                                             fg_error *err){
    qsa_page_transport *transport=&coordinator->qsa_pages;
    /* A warm fetch that was issued by an earlier session can still be in the
     * fabric when the next request starts its barrier; dropping it here keeps
     * the stale frame from aborting the new session.  An unsolicited result
     * inside the current session is also dropped: every issue increments
     * warm_outstanding before the result can arrive. */
    if(!transport->warm_outstanding||
       fg_frame_request_id(header)!=coordinator->session_id)
        return FG_OK;
    fg_qsa_page_batch result={0};
    fg_status status=fg_qsa_page_result_decode(&result,transport->result_pages,
        FG_QSA_PAGE_FETCH_MAX_PAGES,wire,bytes,err);
    if(status==FG_OK&&(result.source_rank!=peer||result.destination_rank!=0u||
       !result.page_count)){
        fg_error_set(err,FG_ERR_MISMATCH,"misrouted QSA mirror warm result");
        status=FG_ERR_MISMATCH;
    }
    for(uint32_t i=0;status==FG_OK&&i<result.page_count;i++){
        const fg_qsa_page *page=&result.pages[i];
        if(page->layer>=FG_LAYER_COUNT||(page->layer&3u)!=3u||
           coordinator->manifest->layer_owner[page->layer]!=peer){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA mirror warm result has the wrong owner");
            status=FG_ERR_MISMATCH;
            break;
        }
        status=fg_owner_qsa_warm_pages(coordinator->owner,page->layer,&page->block,
                                       page->records,1u,err);
    }
    if(status==FG_OK){
        transport->warm_outstanding--;
        transport_complete(&coordinator->transport_state);
        if(qsa_warm_trace_enabled())
            fprintf(stderr,"QSA_WARM_RESULT peer=%u pages=%u outstanding=%u t=%.3f\n",
                peer,result.page_count,transport->warm_outstanding,dispatch_ts());
    }else if(!transport_ready(&coordinator->transport_state))
        transport_poison(&coordinator->transport_state);
    return status;
}

static fg_status coordinator_warm_qsa_drain(fg_coordinator *coordinator,fg_error *err){
    qsa_page_transport *transport=&coordinator->qsa_pages;
    while(transport->warm_outstanding){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        fg_status status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_BULK,&peer,&header,
            transport->result_wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&bytes,err);
        if(status!=FG_OK)return status;
        if(fg_frame_type(&header)!=FG_MSG_QSA_PAGE_RESULT){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "unexpected message %u while draining QSA mirror warm",
                         fg_frame_type(&header));
            return FG_ERR_MISMATCH;
        }
        status=coordinator_warm_qsa_result(coordinator,peer,&header,
                                           transport->result_wire,bytes,err);
        if(status!=FG_OK)return status;
    }
    return FG_OK;
}

static fg_status coordinator_qsa_barrier(fg_coordinator *coordinator,fg_error *err){
    if(!coordinator->session_id)return FG_OK;
    fg_status drain_status=fg_qsa_replica_drain_if_present(
        coordinator->qsa_pages.replica,err);
    if(drain_status!=FG_OK)return drain_status;
    uint32_t owner_mask=coordinator->qsa_pages.append_owner_mask;
    uint8_t wire[FG_RANK_COUNT][FG_QSA_PAGE_BARRIER_BYTES];
    for(uint32_t owner=1u;owner<FG_RANK_COUNT;owner++){
        if(!(owner_mask&(1u<<owner)))continue;
        fg_qsa_page_barrier barrier={.source_rank=0u,.destination_rank=(uint8_t)owner,
            .batch_id=coordinator->qsa_pages.append_sequence[owner]};
        fg_status status=fg_qsa_page_barrier_encode(wire[owner],&barrier,err);
        if(status==FG_OK){
            status=fg_fabric_send(coordinator->fabric,owner,FG_FABRIC_BULK,
                FG_MSG_QSA_PAGE_BARRIER,coordinator->session_id,barrier.batch_id,0,
                wire[owner],sizeof(wire[owner]),err);
            if(status==FG_OK)transport_pending(&coordinator->transport_state);
            else transport_poison(&coordinator->transport_state);
        }
        if(status!=FG_OK){
            if(!transport_ready(&coordinator->transport_state))
                transport_poison(&coordinator->transport_state);
            return status;
        }
    }
    for(uint32_t owner=1u;owner<FG_RANK_COUNT;owner++){
        if(!(owner_mask&(1u<<owner)))continue;
        uint32_t bytes=0;fg_frame_header header;fg_status status=FG_OK;
        /* The owner's channel may still carry a mirror-warm result from the
         * previous session; discard anything that is not this barrier ack. */
        for(;;){
            status=fg_fabric_recv(coordinator->fabric,owner,FG_FABRIC_BULK,&header,
                wire[owner],sizeof(wire[owner]),&bytes,err);
            if(status!=FG_OK)break;
            if(fg_frame_type(&header)!=FG_MSG_QSA_PAGE_RESULT)break;
        }
        fg_qsa_page_barrier ack={0};
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_QSA_PAGE_BARRIER_ACK||
           fg_frame_request_id(&header)!=coordinator->session_id||
           fg_frame_sequence(&header)!=coordinator->qsa_pages.append_sequence[owner])){
            fg_error_set(err,FG_ERR_MISMATCH,"stale QSA page barrier acknowledgement");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_qsa_page_barrier_decode(&ack,wire[owner],bytes,err);
        if(status==FG_OK&&(ack.source_rank!=owner||ack.destination_rank!=0u||
           ack.batch_id!=coordinator->qsa_pages.append_sequence[owner])){
            fg_error_set(err,FG_ERR_MISMATCH,"misrouted QSA page barrier acknowledgement");
            status=FG_ERR_MISMATCH;
        }
        if(status!=FG_OK){
            transport_poison(&coordinator->transport_state);
            return status;
        }
    }
    transport_complete(&coordinator->transport_state);
    return FG_OK;
}

struct fg_runtime {
    fg_manifest *manifest;
    fg_coordinator coordinator;
    int32_t *history;
    size_t history_count,history_capacity;
    /* Owner state frontier: the number of tokens the block owners (and this
     * rank's own owner executor) have processed.  Equals history_count after a
     * successful generation; zero after any reset.  Ring continuation resumes
     * only when the prefix plan's offset matches exactly. */
    uint32_t state_frontier;
    char *rendered_history;
    size_t rendered_history_length;
    size_t pending_boundary_bytes;
    uint32_t pending_eos_token;
    bool pending_eos_valid;
    bool session_started;
    bool prefill_profiled;
    bool state_ready;
    bool next_token_valid;
    uint32_t next_token;
    float next_logit;
    fg_prefix_reset_reason empty_reason;
    fg_runtime_options options;
    uint32_t context_limit;
    char directory[1024];
    fg_sampler_config sampler;
    fg_mtp_capability mtp_capability;
    /* Ring prefix continuation: resume owner state across sequential requests
     * on an exact token-prefix extension. */
    bool prefix_continuation;
};

static fg_status coordinator_begin_session(fg_coordinator *coordinator,fg_error *err){
    fg_status barrier_status=coordinator_qsa_barrier(coordinator,err);
    if(barrier_status!=FG_OK)return barrier_status;
    struct timespec now;if(clock_gettime(CLOCK_REALTIME,&now)!=0){fg_error_set(err,FG_ERR_IO,"read session clock");return FG_ERR_IO;}
    uint64_t request=((uint64_t)(uint32_t)now.tv_sec<<32u)^(uint32_t)now.tv_nsec^(uint64_t)(uint32_t)getpid();if(!request)request=1u;if(request<=coordinator->session_id){if(coordinator->session_id==UINT64_MAX){fg_error_set(err,FG_ERR_LIMIT,"session nonce space is exhausted");return FG_ERR_LIMIT;}request=coordinator->session_id+1u;}
    bool legacy=coordinator->manifest->protocol_version==FG_PROTOCOL_MIN_VERSION;
    for(uint32_t peer=1;peer<FG_RANK_COUNT;peer++){
        uint8_t wire[FG_OWNER_SESSION_CONTROL_BYTES];const void *payload=NULL;uint32_t bytes=0;
        if(!legacy){
            fg_owner_session_control control={
                .version=FG_OWNER_SESSION_CONTROL_VERSION,
                .operation=FG_OWNER_SESSION_BEGIN,.rank=(uint8_t)peer,
                .position_mode=(fg_position_mode)coordinator->manifest->session.position_mode,
                .session_nonce=request,
                .logical_context_tokens=coordinator->options.logical_context_tokens,
                .gpu_index_tokens=coordinator->options.gpu_index_tokens,
                .qsa_hot_tokens=coordinator->options.qsa_hot_tokens,
                .qsa_page_cache_bytes=coordinator->options.qsa_page_cache_bytes
            };
            memcpy(control.identity_sha256,coordinator->identity.identity_sha256,32u);
            memcpy(control.state_format_sha256,
                   coordinator->manifest->session.rank_state_format_sha256[peer],32u);
            fg_status status=fg_owner_session_control_encode(wire,&control,err);
            if(status!=FG_OK){
                if(!transport_ready(&coordinator->transport_state))
                    transport_poison(&coordinator->transport_state);
                return status;
            }
            payload=wire;bytes=sizeof(wire);
        }
        fg_status status=fg_fabric_send(coordinator->fabric,peer,FG_FABRIC_CONTROL,
                                        FG_MSG_SESSION_BEGIN,request,0,0,payload,bytes,err);
        if(status!=FG_OK){
            transport_poison(&coordinator->transport_state);
            return status;
        }
        transport_pending(&coordinator->transport_state);
    }
    bool ready[FG_RANK_COUNT]={0};uint8_t wire[FG_OWNER_SESSION_CONTROL_BYTES];
    for(uint32_t received=1;received<FG_RANK_COUNT;received++){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        fg_status status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_CONTROL,&peer,
                                            &header,legacy?NULL:wire,legacy?0u:sizeof(wire),
                                            &bytes,err);
        if(status!=FG_OK){
            transport_poison(&coordinator->transport_state);
            return status;
        }
        if(fg_frame_type(&header)!=FG_MSG_SESSION_READY||
           fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=0u||
           peer==0u||ready[peer]||(legacy&&bytes)||(!legacy&&bytes!=sizeof(wire))){
            fg_error_set(err,FG_ERR_MISMATCH,"invalid session readiness from rank %u",peer);
            transport_poison(&coordinator->transport_state);
            return FG_ERR_MISMATCH;
        }
        if(!legacy){
            fg_owner_session_control control;
            status=fg_owner_session_control_decode(&control,wire,bytes,err);
            if(status!=FG_OK){
                transport_poison(&coordinator->transport_state);
                return status;
            }
            if(control.operation!=FG_OWNER_SESSION_READY||control.rank!=peer||
               control.session_nonce!=request||
               control.position_mode!=(fg_position_mode)coordinator->manifest->session.position_mode||
               control.logical_context_tokens!=coordinator->options.logical_context_tokens||
               control.gpu_index_tokens!=coordinator->options.gpu_index_tokens||
               control.qsa_hot_tokens!=coordinator->options.qsa_hot_tokens||
               control.qsa_page_cache_bytes!=coordinator->options.qsa_page_cache_bytes||
               memcmp(control.identity_sha256,coordinator->identity.identity_sha256,32u)||
               memcmp(control.state_format_sha256,
                      coordinator->manifest->session.rank_state_format_sha256[peer],32u)){
                fg_error_set(err,FG_ERR_MISMATCH,"rank %u session readiness fingerprint mismatch",peer);
                transport_poison(&coordinator->transport_state);
                return FG_ERR_MISMATCH;
            }
        }
        ready[peer]=true;
    }
    coordinator->session_id=request;
    memset(coordinator->qsa_pages.append_sequence,0,
           sizeof(coordinator->qsa_pages.append_sequence));
    memset(coordinator->qsa_pages.fetch_sequence,0,
           sizeof(coordinator->qsa_pages.fetch_sequence));
    coordinator->qsa_pages.append_owner_mask=0u;
    coordinator->qsa_pages.warm_outstanding=0u;
    transport_complete(&coordinator->transport_state);
    return FG_OK;
}

/* All-local prefill: process all 48 layers on the coordinator */
static fg_status coordinator_prefill_microbatch(fg_coordinator *coordinator,const uint32_t *token_ids,uint32_t first_token,uint16_t token_count,const fg_vk_tensor *ngram_embeddings,fg_vk_tensor **output,fg_error *err){
    if(!coordinator||!token_ids||!token_count||token_count>coordinator->manifest->prefill_microbatch||!ngram_embeddings||!output||token_count>coordinator->manifest->max_context||first_token>coordinator->manifest->max_context-token_count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid coordinator prefill microbatch");return FG_ERR_ARGUMENT;}
    prefill_layer_buffers *buffers=&coordinator->prefill_layer[0];
    for(uint32_t i=0;i<token_count;i++){if(token_ids[i]>=FG_Q38_VOCAB_SIZE){fg_error_set(err,FG_ERR_FORMAT,"prefill token %u is outside Qwen3.8 vocabulary",i);return FG_ERR_FORMAT;}buffers->positions[i]=token_ids[i];}
    fg_status status=fg_vk_tensor_write(buffers->token_tensor,0,buffers->positions,(uint64_t)token_count*4u,err);
    for(uint32_t i=0;i<token_count;i++){for(uint32_t axis=0;axis<3u;axis++){buffers->positions[(uint64_t)i*3u+axis]=first_token+i;}}
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    if(status==FG_OK&&!embedding){fg_error_set(err,FG_ERR_MISMATCH,"coordinator is missing token_embd.weight");status=FG_ERR_MISMATCH;}
    fg_vk_tensor *initial=fg_owner_prefill_input(coordinator->owner);
    if(status==FG_OK&&!initial){
        fg_error_set(err,FG_ERR_MISMATCH,"coordinator prefill input storage is unavailable");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK){status=fg_vk_embedding_q8_0_batch(fg_model_vk(coordinator->model),initial,embedding,buffers->token_tensor,token_count,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);}
    fg_vk_tensor *current=initial;
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        struct timespec layer_start,layer_end;bool profiling=fg_vk_profile_active(fg_model_vk(coordinator->model));if(profiling)clock_gettime(CLOCK_MONOTONIC,&layer_start);
        const fg_vk_tensor *layer_ngram=(layer==1u)?ngram_embeddings:NULL;
        prefill_dispatch_context dispatch={.fabric=coordinator->fabric,
            .expert=coordinator->expert,.manifest=coordinator->manifest,.self=0u,
            .request_id=coordinator->session_id,
            .sequence=first_token*FG_LAYER_COUNT+layer,
            .buffers=&coordinator->prefill_expert[0],
            .transport_state=&coordinator->transport_state};
        fg_vk_tensor *layer_out=NULL;
        status=fg_owner_prefill_layer(coordinator->owner,layer,first_token,buffers->positions,
            token_count,current,layer_ngram,dispatch_prefill_experts,&dispatch,
            NULL,NULL,&layer_out,err);
        if(status==FG_OK){current=layer_out;}
        if(profiling){clock_gettime(CLOCK_MONOTONIC,&layer_end);fprintf(stderr,"PREFILL_PROFILE_LAYER first=%u tokens=%u layer=%u wall_ms=%.3f\n",first_token,token_count,layer,elapsed_seconds(&layer_start,&layer_end)*1000.0);}
    }
    if(status==FG_OK)status=coordinator_publish_qsa_pages(coordinator,first_token,token_count,err);
    if(status==FG_OK){*output=current;}
    return status;
}

/* Pairwise chunk skew: chunk C at layer L and chunk C+1 at layer L-1 are in
 * flight together, so the remote expert wait of one chunk overlaps the owner's
 * attention/router/local work for the other.  Chunk C+1 only needs chunk C's
 * layer-L records, which are complete once C has passed layer L. */
typedef struct ring_slot {
    bool active,block_done;
    uint32_t first_token,chunk_index;
    uint16_t token_count;
    uint32_t *positions;
    prefill_frame frame;
    fg_vk_tensor *output;
} ring_slot;

static fg_status ring_send_block_work(fg_fabric *fabric,const fg_manifest *manifest,
    uint32_t self,uint64_t session_id,uint32_t destination,uint32_t layer,
    uint32_t first_token,uint16_t token_count,const uint32_t *positions,
    const float *hyper,const float *ngram,uint8_t *wire,uint32_t capacity,fg_error *err){
    fg_prefill_layer_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)self,
        .destination_rank=(uint8_t)destination,
        .flags=ngram?FG_LAYER_WORK_HAS_NGRAM:0u,.first_token=first_token,
        .token_count=token_count,.positions=(uint32_t *)positions,
        .hyper=(float *)hyper,.ngram_embeddings=(float *)ngram};
    uint32_t bytes=0;
    fg_status status=fg_prefill_layer_work_encode(wire,capacity,&bytes,
        manifest->protocol_version,&work,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,destination,FG_FABRIC_BULK,
        FG_MSG_PREFILL_LAYER_WORK,session_id,first_token*FG_LAYER_COUNT+layer,0,
        wire,bytes,err);
    return status;
}

/* Pull every remote block owner's GDN/PLE state for the layers this rank 0
 * decode executor will replay locally.  Requests are issued layer by layer
 * (round trip bounded by a single layer payload) and then drained in arrival
 * order so the fabric pipelines across owners. */
static fg_status coordinator_sync_gdn_state(fg_coordinator *coordinator,
    uint32_t frontier,fg_error *err){
    const fg_manifest *manifest=coordinator->manifest;
    uint8_t *result_wire=malloc(FG_GDN_STATE_RESULT_MAX_BYTES);
    if(!result_wire){fg_error_set(err,FG_ERR_OOM,"allocate GDN state receive wire");return FG_ERR_OOM;}
    uint32_t requested=0;fg_status status=FG_OK;
    uint8_t fetch_wire[FG_GDN_STATE_FETCH_BYTES];
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        if((layer&3u)==3u)continue;
        uint32_t owner=manifest->layer_owner[layer];
        if(owner==0u)continue;
        fg_gdn_state_fetch fetch={.layer=layer,.frontier=frontier};
        status=fg_gdn_state_fetch_encode(fetch_wire,&fetch,err);
        if(status!=FG_OK)break;
        status=fg_fabric_send(coordinator->fabric,owner,FG_FABRIC_BULK,
            FG_MSG_GDN_STATE_FETCH,coordinator->session_id,layer,0,
            fetch_wire,sizeof(fetch_wire),err);
        if(status!=FG_OK)break;
        requested++;
    }
    for(uint32_t received=0;status==FG_OK&&received<requested;received++){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_BULK,&peer,&header,
            result_wire,FG_GDN_STATE_RESULT_MAX_BYTES,&bytes,err);
        if(status!=FG_OK)break;
        fg_gdn_state_result result={0};
        if(fg_frame_type(&header)!=FG_MSG_GDN_STATE_RESULT||
           fg_frame_request_id(&header)!=coordinator->session_id){
            fg_error_set(err,FG_ERR_MISMATCH,"stale GDN state result");
            status=FG_ERR_MISMATCH;
            break;
        }
        status=fg_gdn_state_result_decode(&result,result_wire,bytes,err);
        if(status!=FG_OK)break;
        if(result.source_rank!=peer||result.frontier!=frontier||
           manifest->layer_owner[result.layer]!=peer||(result.layer&3u)==3u){
            fg_error_set(err,FG_ERR_MISMATCH,"misrouted GDN state result");
            status=FG_ERR_MISMATCH;
            break;
        }
        fg_vk_tensor *conv=fg_owner_gdn_state_tensor(coordinator->owner,result.layer,0u);
        fg_vk_tensor *recurrent=fg_owner_gdn_state_tensor(coordinator->owner,result.layer,1u);
        if(!conv||!recurrent){
            fg_error_set(err,FG_ERR_MISMATCH,"rank 0 has no local GDN slot for layer %u",
                         result.layer);
            status=FG_ERR_MISMATCH;
            break;
        }
        status=fg_vk_tensor_write(conv,0,result.conv,FG_GDN_STATE_CONV_BYTES,err);
        if(status==FG_OK)status=fg_vk_tensor_write(recurrent,0,result.recurrent,
            FG_GDN_STATE_RECURRENT_BYTES,err);
        if(status==FG_OK&&result.ple){
            fg_vk_tensor *ple=fg_owner_ple_state_tensor(coordinator->owner);
            if(!ple){
                fg_error_set(err,FG_ERR_MISMATCH,"rank 0 has no local PLE state slot");
                status=FG_ERR_MISMATCH;
            }else status=fg_vk_tensor_write(ple,0,result.ple,FG_GDN_STATE_PLE_BYTES,err);
            if(status==FG_OK&&numerics_trace_enabled()){
                float *ple_host=malloc(FG_GDN_STATE_PLE_BYTES);
                if(ple_host){
                    if(fg_vk_tensor_read(ple,0,ple_host,FG_GDN_STATE_PLE_BYTES,err)==FG_OK)
                        numerics_trace_values("PLE_STATE_IMPORT",0u,1u,frontier,ple_host,
                            (uint64_t)FG_HYPER_WIDTH*9u);
                    free(ple_host);
                }
            }
        }
        if(status==FG_OK&&result.layer<=1u&&numerics_trace_enabled()){
            float *full=malloc((uint64_t)FG_HYPER_WIDTH*4u*4u);
            if(full){
                if(fg_vk_tensor_read(conv,0,full,(uint64_t)FG_HYPER_WIDTH*4u*4u,err)==FG_OK)
                    for(uint32_t q=0;q<4u;q++)
                        numerics_trace_values("STATE_GOT_CONV",0u,result.layer,frontier,
                            full+(uint64_t)q*FG_HYPER_WIDTH,(uint64_t)FG_HYPER_WIDTH);
                free(full);
            }
        }
    }
    free(result_wire);
    return status;
}

/* Layer-ring prefill.  Rank 0 embeds each chunk and ships it to the first
 * block owner; every block owner runs its layers and forwards the hyper state
 * along the chain.  Rank 0 executes its own block when the chain reaches it
 * and receives the final result.  A new chunk is issued only after a chain
 * message arrived, so no rank can stall waiting for a peer that is itself
 * blocked sending to it. */
static fg_status coordinator_prefill_pipeline_ring(fg_coordinator *coordinator,
    const int32_t *history,size_t history_count,const uint32_t *token_ids,
    uint32_t first_token,uint32_t token_count,bool *profiled,
    fg_vk_tensor **output,fg_error *err){
    (void)profiled;
    if(!coordinator||!history||!token_ids||!token_count||!output||
       token_count>coordinator->manifest->max_context||
       first_token>coordinator->manifest->max_context-token_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid coordinator ring prefill");return FG_ERR_ARGUMENT;}
    const fg_manifest *manifest=coordinator->manifest;
    if(manifest->layer_owner[0u]==0u){
        fg_error_set(err,FG_ERR_MISMATCH,"ring prefill requires a remote first block");
        return FG_ERR_MISMATCH;
    }
    uint32_t microbatch=manifest->prefill_microbatch;
    fg_vk_context *vk=fg_model_vk(coordinator->model);
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    if(!embedding){fg_error_set(err,FG_ERR_MISMATCH,"coordinator is missing token_embd.weight");return FG_ERR_MISMATCH;}
    uint32_t work_capacity=FG_PREFILL_LAYER_HEADER_BYTES+microbatch*4u*4u+
        microbatch*FG_HYPER_WIDTH*4u+microbatch*FG_NGRAM_EMBED_VALUES*4u;
    uint32_t result_capacity=FG_PREFILL_LAYER_RESULT_MAX_BYTES;
    uint32_t receive_capacity=work_capacity>result_capacity?work_capacity:result_capacity;
    uint8_t *receive_wire=malloc(receive_capacity);
    uint8_t *work_wire=malloc(work_capacity);
    uint8_t *result_wire=malloc(result_capacity);
    float *hyper_host=malloc((size_t)microbatch*FG_HYPER_WIDTH*sizeof(float));
    float *ngram_host=malloc((size_t)microbatch*FG_NGRAM_EMBED_VALUES*sizeof(float));
    uint32_t *positions=calloc((size_t)FG_PREFILL_FRAMES*microbatch*3u,sizeof(uint32_t));
    uint32_t *positions_scratch=malloc((size_t)microbatch*3u*sizeof(uint32_t));
    fg_status status=FG_OK;
    if(!receive_wire||!work_wire||!result_wire||!hyper_host||!ngram_host||
       !positions||!positions_scratch){
        fg_error_set(err,FG_ERR_OOM,"allocate ring prefill buffers");status=FG_ERR_OOM;
    }
    prefill_dispatch_context contexts[FG_PREFILL_FRAMES];
    ring_slot slots[FG_PREFILL_FRAMES];
    memset(slots,0,sizeof(slots));
    for(uint32_t f=0;status==FG_OK&&f<FG_PREFILL_FRAMES;f++){
        contexts[f]=(prefill_dispatch_context){.fabric=coordinator->fabric,
            .expert=coordinator->expert,.manifest=manifest,.self=0u,
            .request_id=coordinator->session_id,
            .buffers=&coordinator->prefill_expert[f],
            .transport_state=&coordinator->transport_state};
        slots[f].frame=(prefill_frame){.dispatch=&contexts[f],
            .buffers=&coordinator->prefill_expert[f],.slot=f};
        slots[f].positions=positions+(size_t)f*microbatch*3u;
        /* Chunks complete in order, so the newest result is always in the
         * ping-pong output slot; only two result tensors are ever needed. */
        slots[f].output=coordinator->ring_output[f%2u];
        if(!slots[f].output){
            fg_error_set(err,FG_ERR_UNAVAILABLE,"ring output tensor is unavailable");
            status=FG_ERR_UNAVAILABLE;
        }
    }
    uint32_t total_chunks=(token_count+microbatch-1u)/microbatch;
    uint32_t next_chunk=0,in_flight=0,completed=0;
    fg_vk_tensor *last_output=NULL;
    bool ring_trace=getenv("FG_RING_TRACE")!=NULL;
    double ring_t0=dispatch_ts();
    while(status==FG_OK&&completed<total_chunks){
        if(next_chunk<total_chunks&&in_flight<FG_PREFILL_FRAMES){
            uint32_t chunk=next_chunk,f=chunk%FG_PREFILL_FRAMES;
            ring_slot *slot=&slots[f];
            if(slot->active){
                fg_error_set(err,FG_ERR_MISMATCH,"ring slot %u is still active",f);
                status=FG_ERR_MISMATCH;
                break;
            }
            struct timespec issue_start={0};
            if(ring_trace)clock_gettime(CLOCK_MONOTONIC,&issue_start);
            uint32_t chunk_first=first_token+chunk*microbatch;
            uint32_t chunk_tokens=token_count-chunk*microbatch;
            if(chunk_tokens>microbatch)chunk_tokens=microbatch;
            slot->active=true;slot->block_done=false;
            slot->first_token=chunk_first;slot->chunk_index=chunk;
            slot->token_count=(uint16_t)chunk_tokens;
            prefill_layer_buffers *layer=&coordinator->prefill_layer[f];
            status=fg_vk_tensor_write(layer->token_tensor,0,token_ids+chunk*microbatch,
                (uint64_t)chunk_tokens*4u,err);
            for(uint32_t i=0;status==FG_OK&&i<chunk_tokens;i++)
                for(uint32_t axis=0;axis<3u;axis++)
                    slot->positions[i*3u+axis]=chunk_first+i;
            fg_vk_tensor *input=fg_owner_prefill_input_slot(coordinator->owner,0u);
            if(status==FG_OK&&!input){
                fg_error_set(err,FG_ERR_UNAVAILABLE,"ring prefill input slot is unavailable");
                status=FG_ERR_UNAVAILABLE;
            }
            if(status==FG_OK)status=fg_vk_embedding_q8_0_batch(vk,input,embedding,
                layer->token_tensor,chunk_tokens,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,
                FG_Q38_HYPER_COUNT,err);
            fg_vk_tensor *ngram_view=NULL;
            if(status==FG_OK)status=fg_ngram_store_lookup_prefill(coordinator->ngram,
                history,history_count,chunk_first,chunk_tokens,&ngram_view,err);
            if(status==FG_OK)status=fg_vk_tensor_read(input,0,hyper_host,
                (uint64_t)chunk_tokens*FG_HYPER_WIDTH*4u,err);
            if(status==FG_OK)numerics_trace_host("ISSUE",0u,0u,chunk_first,chunk_tokens,hyper_host);
            if(status==FG_OK&&ngram_view)status=fg_vk_tensor_read(ngram_view,0,ngram_host,
                (uint64_t)chunk_tokens*FG_NGRAM_EMBED_VALUES*4u,err);
            if(status==FG_OK)status=ring_send_block_work(coordinator->fabric,manifest,0u,
                coordinator->session_id,manifest->layer_owner[0u],0u,chunk_first,
                (uint16_t)chunk_tokens,slot->positions,hyper_host,
                ngram_view?ngram_host:NULL,work_wire,work_capacity,err);
            if(status==FG_OK){
                next_chunk++;in_flight++;
                if(ring_trace){struct timespec issue_end;clock_gettime(CLOCK_MONOTONIC,&issue_end);
                    fprintf(stderr,
                    "RING_ISSUE chunk=%u first=%u in_flight=%u issue_ms=%.1f t=%.3f\n",
                    chunk,chunk_first,in_flight,
                    elapsed_seconds(&issue_start,&issue_end)*1000.0,dispatch_ts()-ring_t0);}
            }
            continue;
        }
        uint32_t peer=0,bytes=0;fg_frame_header header;
        status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_BULK,&peer,&header,
            receive_wire,receive_capacity,&bytes,err);
        if(status!=FG_OK)break;
        fg_message_type type=fg_frame_type(&header);
        if(type==FG_MSG_PREFILL_LAYER_WORK){
            fg_prefill_layer_work work={0};
            fg_status decode_status=fg_prefill_layer_work_decode(&work,
                manifest->protocol_version,positions_scratch,microbatch*3u,hyper_host,
                (uint64_t)microbatch*FG_HYPER_WIDTH,ngram_host,
                (uint64_t)microbatch*FG_NGRAM_EMBED_VALUES,receive_wire,bytes,err);
            if(decode_status!=FG_OK){status=decode_status;break;}
            if(work.destination_rank!=0u||manifest->layer_owner[work.layer]!=0u||
               (work.layer>0u&&peer!=manifest->layer_owner[work.layer-1u])||
               fg_frame_request_id(&header)!=coordinator->session_id||
               fg_frame_sequence(&header)!=work.first_token*FG_LAYER_COUNT+work.layer||
               work.first_token<first_token||
               work.first_token+work.token_count>first_token+token_count){
                fg_error_set(err,FG_ERR_MISMATCH,"misrouted ring layer work");
                status=FG_ERR_MISMATCH;
                break;
            }
            numerics_trace_host("RECV_WORK",0u,work.layer,work.first_token,
                work.token_count,hyper_host);
            uint32_t chunk=(work.first_token-first_token)/microbatch,f=chunk%FG_PREFILL_FRAMES;
            ring_slot *slot=&slots[f];
            if(!slot->active||slot->chunk_index!=chunk||
               slot->first_token!=work.first_token||slot->token_count!=work.token_count){
                fg_error_set(err,FG_ERR_MISMATCH,"ring work does not match an in-flight chunk");
                status=FG_ERR_MISMATCH;
                break;
            }
            if(ring_trace)fprintf(stderr,
                "RING_STAGE chunk=%u layer=%u in_flight=%u t=%.3f\n",
                chunk,work.layer,in_flight,dispatch_ts()-ring_t0);
            struct timespec own_start={0};
            if(ring_trace)clock_gettime(CLOCK_MONOTONIC,&own_start);
            memcpy(slot->positions,positions_scratch,(size_t)work.token_count*3u*4u);
            /* The coordinator executes its own block inline, one chunk at a
             * time, so every frame shares the base owner slot. */
            fg_vk_tensor *input=fg_owner_prefill_input_slot(coordinator->owner,0u);
            status=fg_vk_tensor_write(input,0,hyper_host,
                (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
            fg_vk_tensor *cur=input;
            uint32_t last=work.layer;
            while(last+1u<FG_LAYER_COUNT&&manifest->layer_owner[last+1u]==0u)last++;
            for(uint32_t layer=work.layer;status==FG_OK&&layer<=last;layer++){
                slot->frame.sequence=work.first_token*FG_LAYER_COUNT+layer;
                status=fg_owner_prefill_layer_begin(coordinator->owner,0u,layer,
                    work.first_token,slot->positions,(uint16_t)work.token_count,cur,
                    NULL,prefill_fire,&slot->frame,NULL,NULL,err);
                if(status==FG_OK)status=fg_owner_prefill_layer_finish(coordinator->owner,0u,
                    prefill_collect,&slot->frame,&cur,err);
            }
            if(status==FG_OK)status=fg_vk_tensor_read(cur,0,hyper_host,
                (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
            if(status==FG_OK){
                slot->block_done=true;
                if(last+1u<FG_LAYER_COUNT){
                    status=ring_send_block_work(coordinator->fabric,manifest,0u,
                        coordinator->session_id,manifest->layer_owner[last+1u],last+1u,
                        work.first_token,(uint16_t)work.token_count,slot->positions,
                        hyper_host,NULL,work_wire,work_capacity,err);
                }else{
                    bool final=completed+1u==total_chunks;
                    if(final)status=fg_vk_tensor_write(slot->output,0,hyper_host,
                        (uint64_t)work.token_count*FG_HYPER_WIDTH*4u,err);
                    slot->active=false;in_flight--;completed++;
                    if(final)last_output=slot->output;
                    if(status==FG_OK)status=coordinator_publish_qsa_pages(coordinator,
                        work.first_token,(uint16_t)work.token_count,err);
                    if(status==FG_OK)status=coordinator_warm_qsa_issue(coordinator,
                        work.first_token,(uint16_t)work.token_count,err);
                }
            }
            if(ring_trace){struct timespec own_end;clock_gettime(CLOCK_MONOTONIC,&own_end);
                fprintf(stderr,"RING_OWN_BLOCK chunk=%u layers=%u..%u ms=%.1f\n",
                    chunk,work.layer,last,elapsed_seconds(&own_start,&own_end)*1000.0);}
        }else if(type==FG_MSG_PREFILL_LAYER_RESULT){
            fg_prefill_layer_result result={0};
            fg_status decode_status=fg_prefill_layer_result_decode(&result,hyper_host,
                (uint64_t)microbatch*FG_HYPER_WIDTH,receive_wire,bytes,err);
            if(decode_status!=FG_OK){status=decode_status;break;}
            if(result.destination_rank!=0u||
               result.source_rank!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
               fg_frame_request_id(&header)!=coordinator->session_id||
               result.first_token<first_token||
               result.first_token+result.token_count>first_token+token_count){
                fg_error_set(err,FG_ERR_MISMATCH,"misrouted ring result");
                status=FG_ERR_MISMATCH;
                break;
            }
            numerics_trace_host("RECV_RESULT",0u,FG_LAYER_COUNT-1u,result.first_token,
                result.token_count,hyper_host);
            uint32_t chunk=(result.first_token-first_token)/microbatch,f=chunk%FG_PREFILL_FRAMES;
            ring_slot *slot=&slots[f];
            if(!slot->active||slot->chunk_index!=chunk||
               slot->first_token!=result.first_token||slot->token_count!=result.token_count){
                fg_error_set(err,FG_ERR_MISMATCH,"ring result does not match an in-flight chunk");
                status=FG_ERR_MISMATCH;
                break;
            }
            struct timespec result_start={0};
            if(ring_trace)clock_gettime(CLOCK_MONOTONIC,&result_start);
            bool final=completed+1u==total_chunks;
            if(final)status=fg_vk_tensor_write(slot->output,0,result.hyper,
                (uint64_t)result.token_count*FG_HYPER_WIDTH*4u,err);
            slot->active=false;in_flight--;completed++;
            if(final)last_output=slot->output;
            if(status==FG_OK)status=coordinator_publish_qsa_pages(coordinator,
                result.first_token,result.token_count,err);
            if(status==FG_OK)status=coordinator_warm_qsa_issue(coordinator,
                result.first_token,result.token_count,err);
            if(ring_trace){struct timespec result_end;clock_gettime(CLOCK_MONOTONIC,&result_end);
                fprintf(stderr,
                "RING_DONE chunk=%u in_flight=%u completed=%u handle_ms=%.1f t=%.3f\n",
                chunk,in_flight,completed,
                elapsed_seconds(&result_start,&result_end)*1000.0,dispatch_ts()-ring_t0);}
        }else if(type==FG_MSG_QSA_PAGE_RESULT){
            status=coordinator_warm_qsa_result(coordinator,peer,&header,receive_wire,bytes,err);
        }else{
            fg_error_set(err,FG_ERR_MISMATCH,"unexpected ring message type %u",type);
            status=FG_ERR_MISMATCH;
            break;
        }
    }
    if(status==FG_OK)status=coordinator_warm_qsa_drain(coordinator,err);
    if(status==FG_OK)*output=last_output;
    /* Decode on rank 0 attaches to owners' committed state; the mirror did not
     * compute most QSA layers, so advance its committed counters to the
     * prefilled context before decoding. */
    if(status==FG_OK)fg_owner_qsa_set_tokens(coordinator->owner,first_token+token_count);
    if(status==FG_OK&&qsa_warm_trace_enabled())
        coordinator_trace_qsa_cache(coordinator,"prefill_end",
            (first_token+token_count)/FG_Q38_QSA_COMPRESS_RATIO);
    /* Legacy decode runs all 48 layers on rank 0, so import every owner's
     * stateful layer state that ring prefill advanced remotely.  Ring decode
     * leaves each owner decoding its own block from its authoritative state. */
    if(status==FG_OK&&!coordinator->ring_decode)status=coordinator_sync_gdn_state(coordinator,
        first_token+token_count,err);
    free(positions_scratch);free(positions);free(ngram_host);free(hyper_host);
    free(result_wire);free(work_wire);free(receive_wire);
    return status;
}

static fg_status coordinator_prefill_pipeline(fg_coordinator *coordinator,
    const int32_t *history,size_t history_count,const uint32_t *token_ids,
    uint32_t first_token,uint32_t token_count,bool *profiled,
    fg_vk_tensor **output,fg_error *err){
    if(coordinator&&coordinator->ring_prefill)
        return coordinator_prefill_pipeline_ring(coordinator,history,history_count,token_ids,
            first_token,token_count,profiled,output,err);
    if(!coordinator||!history||!token_ids||!token_count||!output||
       token_count>coordinator->manifest->max_context||
       first_token>coordinator->manifest->max_context-token_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid coordinator prefill pipeline");return FG_ERR_ARGUMENT;}
    uint32_t microbatch=coordinator->manifest->prefill_microbatch;
    fg_vk_context *vk=fg_model_vk(coordinator->model);
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    if(!embedding){fg_error_set(err,FG_ERR_MISMATCH,"coordinator is missing token_embd.weight");return FG_ERR_MISMATCH;}
    fg_vk_tensor *last=NULL;fg_status status=FG_OK;
    for(uint32_t base=0;status==FG_OK&&base<token_count;base+=FG_PREFILL_FRAMES*microbatch){
        uint32_t counts[FG_PREFILL_FRAMES]={0},offsets[FG_PREFILL_FRAMES]={0},consumed=0;
        for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++){
            uint32_t remaining=token_count-base-consumed;
            offsets[f]=consumed;
            counts[f]=remaining>microbatch?microbatch:remaining;
            consumed+=counts[f];
        }
        uint32_t first=first_token+base;
        fg_vk_tensor *ngram_all=NULL,*ngram_views[FG_PREFILL_FRAMES]={NULL,NULL,NULL};
        status=fg_ngram_store_lookup_prefill(coordinator->ngram,history,history_count,
            first,consumed,&ngram_all,err);
        if(status==FG_OK)ngram_views[0]=ngram_all;
        for(uint32_t f=1;status==FG_OK&&f<FG_PREFILL_FRAMES;f++)
            if(counts[f])status=fg_vk_tensor_view(ngram_all,
                (uint64_t)offsets[f]*FG_NGRAM_EMBED_VALUES*4u,
                (uint64_t)counts[f]*FG_NGRAM_EMBED_VALUES*4u,&ngram_views[f],err);
        fg_vk_tensor *inputs[FG_PREFILL_FRAMES]={NULL,NULL,NULL};
        for(uint32_t f=0;status==FG_OK&&f<FG_PREFILL_FRAMES;f++){
            if(!counts[f])continue;
            prefill_layer_buffers *layer=&coordinator->prefill_layer[f];
            status=fg_vk_tensor_write(layer->token_tensor,0,token_ids+base+offsets[f],(uint64_t)counts[f]*4u,err);
            if(status==FG_OK){for(uint32_t i=0;i<counts[f];i++)for(uint32_t axis=0;axis<3u;axis++)layer->positions[(uint64_t)i*3u+axis]=first+offsets[f]+i;}
            if(status==FG_OK)inputs[f]=fg_owner_prefill_input_slot(coordinator->owner,f);
            if(status==FG_OK)status=fg_vk_embedding_q8_0_batch(vk,inputs[f],embedding,layer->token_tensor,counts[f],FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);
        }
        prefill_dispatch_context contexts[FG_PREFILL_FRAMES];
        prefill_frame frames[FG_PREFILL_FRAMES];
        for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++){
            contexts[f]=(prefill_dispatch_context){.fabric=coordinator->fabric,.expert=coordinator->expert,
                .manifest=coordinator->manifest,.self=0u,.request_id=coordinator->session_id,
                .buffers=&coordinator->prefill_expert[f],.transport_state=&coordinator->transport_state};
            frames[f]=(prefill_frame){.dispatch=&contexts[f],.buffers=&coordinator->prefill_expert[f],.slot=f};
        }
        fg_vk_tensor *cur[FG_PREFILL_FRAMES];
        for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++)cur[f]=inputs[f];
        bool capture=status==FG_OK&&profiled&&!*profiled&&prefill_profile_requested();
        const char *profile_chunk=getenv("FG_PREFILL_PROFILE_CHUNK");
        if(capture&&profile_chunk&&*profile_chunk)capture=(base/microbatch)==(uint32_t)strtoul(profile_chunk,NULL,10);
        bool capture_active=false;struct timespec capture_start={0},capture_end={0};
        struct timespec pair_start={0};if(frame_trace_enabled())clock_gettime(CLOCK_MONOTONIC,&pair_start);
        if(capture){status=fg_vk_profile_begin(vk,err);if(status==FG_OK){status=fg_vk_profile_set_scope(vk,"ngram_prefill",err);clock_gettime(CLOCK_MONOTONIC,&capture_start);capture_active=true;}}
        for(uint32_t i=0;status==FG_OK&&i<=FG_LAYER_COUNT+FG_PREFILL_FRAMES-1u;i++){
            for(uint32_t f=0;status==FG_OK&&f<FG_PREFILL_FRAMES;f++){
                if(!counts[f])continue;
                uint32_t layer=i-f;
                prefill_frame *frame=&frames[f];
                if(layer>=1u&&layer<=FG_LAYER_COUNT){
                    if(frame_trace_enabled())fprintf(stderr,"PREFILL_FINISH slot=%u layer=%u\n",f,layer-1u);
                    status=fg_owner_prefill_layer_finish(coordinator->owner,f,prefill_collect,frame,&cur[f],err);
                }
                if(status==FG_OK&&layer<FG_LAYER_COUNT){
                    frame->sequence=(first+offsets[f])*FG_LAYER_COUNT+layer;
                    if(frame_trace_enabled())fprintf(stderr,"PREFILL_BEGIN slot=%u layer=%u\n",f,layer);
                    status=fg_owner_prefill_layer_begin(coordinator->owner,f,layer,first+offsets[f],
                        coordinator->prefill_layer[f].positions,(uint16_t)counts[f],cur[f],
                        layer==1u?ngram_views[f]:NULL,prefill_fire,frame,NULL,NULL,err);
                }
            }
            if(status!=FG_OK){
                fprintf(stderr,"PREFILL_ERROR i=%u status=%d msg=%s\n",i,(int)status,err?err->message:"(none)");
                fg_prefill_result drain[FG_GROUP_SIZE];uint32_t drain_count=0;fg_error ignored={0};
                for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++){
                    if(!counts[f])continue;
                    if((frames[f].sent||frames[f].local_enqueued)&&!frames[f].collected){
                        uint32_t layer=(i>=f&&i-f<=FG_LAYER_COUNT)?i-f:0u;
                        prefill_collect(&frames[f],layer,first+offsets[f],(uint16_t)counts[f],drain,&drain_count,&ignored);
                    }
                }
                break;
            }
            if(status==FG_OK&&capture_active&&i==FG_LAYER_COUNT){
                fg_vk_profile capture_profile={0};fg_error profile_error={0};
                clock_gettime(CLOCK_MONOTONIC,&capture_end);
                fg_status profile_status=fg_vk_profile_end(vk,&capture_profile,status==FG_OK?err:&profile_error);
                capture_active=false;*profiled=true;
                if(status==FG_OK&&profile_status!=FG_OK)status=profile_status;
                fprintf(stderr,"PREFILL_PROFILE first=%u tokens=%u wall_ms=%.3f gpu_ms=%.3f kernel_ms=%.3f submissions=%llu dispatches=%llu\n",first,counts[0],elapsed_seconds(&capture_start,&capture_end)*1000.0,capture_profile.gpu_ms,capture_profile.kernel_ms,(unsigned long long)capture_profile.submissions,(unsigned long long)capture_profile.dispatches);
                for(uint32_t k=0;k<capture_profile.kernel_count;k++){const fg_vk_profile_kernel *kernel=&capture_profile.kernels[k];fprintf(stderr,"PREFILL_PROFILE_KERNEL scope=%s kernel=%s calls=%llu gpu_ms=%.3f\n",kernel->scope,kernel->name,(unsigned long long)kernel->invocations,kernel->gpu_ms);}
            }
        }
        if(capture_active){fg_vk_profile capture_profile={0};fg_error profile_error={0};clock_gettime(CLOCK_MONOTONIC,&capture_end);fg_status profile_status=fg_vk_profile_end(vk,&capture_profile,status==FG_OK?err:&profile_error);(void)profile_status;*profiled=true;}
        for(uint32_t f=1;f<FG_PREFILL_FRAMES;f++){fg_vk_tensor_destroy(ngram_views[f]);ngram_views[f]=NULL;}
        if(frame_trace_enabled()){struct timespec pair_end;clock_gettime(CLOCK_MONOTONIC,&pair_end);fprintf(stderr,"PREFILL_PAIR base=%u tokens=%u wall_ms=%.1f\n",base,consumed,elapsed_seconds(&pair_start,&pair_end)*1000.0);}
        if(status==FG_OK)status=coordinator_publish_qsa_pages(coordinator,first,counts[0],err);
        for(uint32_t f=1;status==FG_OK&&f<FG_PREFILL_FRAMES;f++)
            if(counts[f])status=coordinator_publish_qsa_pages(coordinator,first+offsets[f],counts[f],err);
        if(status==FG_OK){for(uint32_t f=0;f<FG_PREFILL_FRAMES;f++)if(counts[f])last=cur[f];}
    }
    if(status==FG_OK)*output=last;
    return status;
}

static fg_status coordinator_output(fg_coordinator *coordinator,uint32_t token_index,
                                    const fg_vk_tensor *hyper,uint32_t *next_token,
                                    float *logit,fg_error *err){
    fg_output_work *work=calloc(1,sizeof(*work));
    uint8_t *wire=malloc(FG_OUTPUT_WORK_BYTES);
    if(!work||!wire){
        free(wire);free(work);
        fg_error_set(err,FG_ERR_OOM,"allocate coordinator output exchange");
        return FG_ERR_OOM;
    }
    work->source_rank=0u;work->destination_rank=4u;work->token_index=token_index;
    work->sampler=coordinator->sampler;
    work->uniform=work->sampler.temperature>0.0f?
        fg_sampler_uniform(&coordinator->sampler_state):0.0f;
    fg_status status=fg_vk_tensor_read(hyper,0,work->hyper,sizeof(work->hyper),err);
    if(status==FG_OK)status=fg_output_work_encode(wire,work,err);
    uint32_t sequence=token_index*FG_LAYER_COUNT+FG_LAYER_COUNT;
    if(status==FG_OK){
        status=fg_fabric_send(coordinator->fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_WORK,coordinator->session_id,sequence,0,wire,
            FG_OUTPUT_WORK_BYTES,err);
        if(status==FG_OK)transport_pending(&coordinator->transport_state);
        else transport_poison(&coordinator->transport_state);
    }
    if(status==FG_OK){
        fg_frame_header header;uint32_t bytes=0;
        status=fg_fabric_recv(coordinator->fabric,4u,FG_FABRIC_BULK,&header,wire,
                              FG_OUTPUT_RESULT_BYTES,&bytes,err);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_RESULT||
           fg_frame_request_id(&header)!=coordinator->session_id||
           fg_frame_sequence(&header)!=sequence)){
            fg_error_set(err,FG_ERR_MISMATCH,"stale output result");
            status=FG_ERR_MISMATCH;
        }
        fg_output_result result;
        if(status==FG_OK)status=fg_output_result_decode(&result,wire,bytes,err);
        if(status==FG_OK&&(result.source_rank!=4u||result.destination_rank!=0u||
           result.token_index!=token_index)){
            fg_error_set(err,FG_ERR_MISMATCH,"misrouted output result");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK){
            *next_token=result.token;if(logit)*logit=result.logit;
            transport_complete(&coordinator->transport_state);
        }else transport_poison(&coordinator->transport_state);
    }
    free(wire);free(work);return status;
}

/* Direct handoff: rank 0 ships the sampler route for the token before the chain
 * runs and later accepts the 16-byte result from the output owner. */
static fg_status coordinator_output_config(fg_coordinator *coordinator,uint32_t token_index,
                                           fg_error *err){
    uint8_t flags=0u;
    if(coordinator->output_slice)
        flags=fg_output_slice_ways(coordinator->output_slice)==FG_OUTPUT_SPLIT_WAYS_MAX?
            (FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4):
            FG_OUTPUT_CONFIG_FLAG_SPLIT;
    fg_output_config config={.source_rank=0u,.destination_rank=4u,
        .flags=flags,
        .token_index=token_index,.sampler=coordinator->sampler,
        .uniform=coordinator->sampler.temperature>0.0f?
            fg_sampler_uniform(&coordinator->sampler_state):0.0f};
    uint8_t wire[FG_OUTPUT_CONFIG_BYTES];
    fg_status status=fg_output_config_encode(wire,&config,err);
    if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,4u,FG_FABRIC_CONTROL,
        FG_MSG_OUTPUT_CONFIG,coordinator->session_id,
        token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,sizeof(wire),err);
    if(status==FG_OK)transport_pending(&coordinator->transport_state);
    else transport_poison(&coordinator->transport_state);
    return status;
}

static fg_status coordinator_output_result_accept(fg_coordinator *coordinator,
    const fg_frame_header *header,const uint8_t *wire,uint32_t bytes,uint32_t token_index,
    uint32_t *next_token,float *logit,fg_error *err){
    if(fg_frame_type(header)!=FG_MSG_OUTPUT_RESULT||
       fg_frame_request_id(header)!=coordinator->session_id||
       fg_frame_sequence(header)!=token_index*FG_LAYER_COUNT+FG_LAYER_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,"stale output result");
        return FG_ERR_MISMATCH;
    }
    fg_output_result result;
    fg_status status=fg_output_result_decode(&result,wire,bytes,err);
    if(status==FG_OK&&(result.source_rank!=4u||result.destination_rank!=0u||
       result.token_index!=token_index)){
        fg_error_set(err,FG_ERR_MISMATCH,"misrouted output result");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK){
        *next_token=result.token;if(logit)*logit=result.logit;
        transport_complete(&coordinator->transport_state);
    }else transport_poison(&coordinator->transport_state);
    return status;
}

static fg_status ngram_rank_for_row(uint64_t row,uint32_t *owner,fg_error *err){if(!owner){fg_error_set(err,FG_ERR_ARGUMENT,"n-gram row owner output is null");return FG_ERR_ARGUMENT;}for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){uint64_t begin=0,count=0;fg_status status=fg_q38_ngram_rank_range(rank,&begin,&count,err);if(status!=FG_OK)return status;if(row>=begin&&row-begin<count){*owner=rank;return FG_OK;}}fg_error_set(err,FG_ERR_MISMATCH,"n-gram row %llu has no resident owner",(unsigned long long)row);return FG_ERR_MISMATCH;}

static fg_status resident_ngram_lookup(
    fg_fabric *fabric,fg_ngram_store *ngram,uint64_t session_id,
    atomic_uint *transport_state,fg_fabric_class result_class,
    const int32_t *history,size_t history_count,uint32_t token_index,
    fg_vk_tensor **embedding,fg_error *err){
    uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];
    fg_status status=fg_q38_ngram_lookup_range(history,history_count,
        history_count?history_count-1u:0u,1u,rows,addresses,err);
    fg_ngram_work work[FG_RANK_COUNT]={0};
    for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++)
        work[rank]=(fg_ngram_work){.source_rank=0u,.destination_rank=(uint8_t)rank,
            .token_index=token_index};
    for(uint32_t head=0;status==FG_OK&&head<FG_NGRAM_HEAD_COUNT;head++){
        uint32_t rank=0;status=ngram_rank_for_row(rows[head],&rank,err);
        uint32_t item=work[rank].item_count;
        if(status==FG_OK&&item>=FG_NGRAM_SHARD_MAX_ITEMS){
            fg_error_set(err,FG_ERR_LIMIT,"resident n-gram rank %u item overflow",rank);
            status=FG_ERR_LIMIT;
        }
        if(status==FG_OK){
            work[rank].heads[item]=(uint8_t)head;
            work[rank].rows[item]=rows[head];
            work[rank].item_count++;
        }
    }
    uint8_t work_wire[FG_NGRAM_WORK_MAX_BYTES],result_wire[FG_NGRAM_RESULT_MAX_BYTES];
    uint8_t packed[FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES];
    bool sent[FG_RANK_COUNT]={0},received[FG_RANK_COUNT]={0};
    bool seen_head[FG_NGRAM_HEAD_COUNT]={0},transport_failed=false;
    uint32_t sent_count=0;
    double start=dispatch_ts();
    for(uint32_t rank=1u;status==FG_OK&&rank<FG_RANK_COUNT;rank++)if(work[rank].item_count){
        uint32_t bytes=0;
        status=fg_ngram_work_encode(work_wire,sizeof(work_wire),&bytes,&work[rank],err);
        if(status==FG_OK){
            status=fg_fabric_send(fabric,rank,FG_FABRIC_CONTROL,
                FG_MSG_NGRAM_WORK,session_id,token_index,0,work_wire,bytes,err);
            if(status==FG_OK){
                sent[rank]=true;sent_count++;
                if(transport_state)transport_pending(transport_state);
            }else{
                transport_failed=true;
                if(transport_state)transport_poison(transport_state);
            }
        }
    }
    double fired=dispatch_ts();
    uint32_t consumed=0;
    while(consumed<sent_count){
        uint32_t peer=0,bytes=0;fg_frame_header header;fg_error receive_error={0};
        fg_status receive_status=fg_fabric_recv_any(fabric,result_class,
            &peer,&header,result_wire,sizeof(result_wire),&bytes,
            status==FG_OK?err:&receive_error);
        if(receive_status!=FG_OK){
            transport_failed=true;
            if(status==FG_OK)status=receive_status;
            break;
        }
        consumed++;
        bool valid=peer>0u&&peer<FG_RANK_COUNT&&sent[peer]&&!received[peer]&&
            fg_frame_type(&header)==FG_MSG_NGRAM_RESULT&&
            fg_frame_request_id(&header)==session_id&&
            fg_frame_sequence(&header)==token_index;
        if(!valid){
            transport_failed=true;
            if(status==FG_OK){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "stale or duplicate resident n-gram result from rank %u",peer);
                status=FG_ERR_MISMATCH;
            }
            continue;
        }
        received[peer]=true;
        if(status!=FG_OK)continue;
        fg_ngram_result result={0};
        status=fg_ngram_result_decode(&result,result_wire,bytes,err);
        if(status!=FG_OK)transport_failed=true;
        if(status==FG_OK&&(result.source_rank!=peer||result.destination_rank!=0u||
           result.token_index!=token_index||result.item_count!=work[peer].item_count)){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "misrouted resident n-gram result from rank %u",peer);
            status=FG_ERR_MISMATCH;
            transport_failed=true;
        }
        for(uint32_t i=0;status==FG_OK&&i<result.item_count;i++){
            uint32_t head=result.heads[i];bool expected=false;
            for(uint32_t j=0;j<work[peer].item_count;j++)
                if(work[peer].heads[j]==head){expected=true;break;}
            if(!expected||seen_head[head]){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "unexpected resident n-gram head %u from rank %u",head,peer);
                status=FG_ERR_MISMATCH;
                transport_failed=true;
            }else{
                memcpy(packed+(uint64_t)head*FG_NGRAM_ROW_BYTES,
                       result.packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,
                       FG_NGRAM_ROW_BYTES);
                seen_head[head]=true;
            }
        }
    }
    bool all_received=true;
    for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++)
        if(sent[rank]!=received[rank]){all_received=false;break;}
    if(transport_state){
        if(consumed==sent_count&&all_received&&!transport_failed)
            transport_complete(transport_state);
        else
            transport_poison(transport_state);
    }
    for(uint32_t head=0;status==FG_OK&&head<FG_NGRAM_HEAD_COUNT;head++)
        if(!seen_head[head]){
            fg_error_set(err,FG_ERR_MISMATCH,"missing resident n-gram head %u",head);
            status=FG_ERR_MISMATCH;
        }
    double collected=dispatch_ts();const char *verify=getenv("FG_NGRAM_VERIFY");
    if(status==FG_OK&&token_profile_requested(token_index)&&verify&&*verify&&
       strcmp(verify,"0")!=0){
        uint32_t mismatch=UINT32_MAX;
        status=fg_ngram_store_verify_packed(ngram,addresses,
            FG_NGRAM_HEAD_COUNT,packed,&mismatch,err);
        if(status==FG_OK)
            fprintf(stderr,"NGRAM_RESIDENT_VERIFY token=%u rows=%u status=exact\n",
                    token_index,FG_NGRAM_HEAD_COUNT);
    }
    if(status==FG_OK)status=fg_ngram_store_decode_packed(ngram,packed,
                                                         FG_NGRAM_HEAD_COUNT,embedding,err);
    double decoded=dispatch_ts();
    if(frame_trace_enabled())
        fprintf(stderr,"NGRAM_RESIDENT_TRACE token=%u workers=%u fire_ms=%.3f collect_ms=%.3f dequant_ms=%.3f total_ms=%.3f\n",
                token_index,sent_count,fired-start,collected-fired,decoded-collected,
                decoded-start);
    return status;
}

static fg_status coordinator_ngram_resident(
    fg_coordinator *coordinator,const int32_t *history,size_t history_count,
    uint32_t token_index,fg_vk_tensor **embedding,fg_error *err){
    return resident_ngram_lookup(coordinator->fabric,coordinator->ngram,
        coordinator->session_id,&coordinator->transport_state,FG_FABRIC_BULK,history,
        history_count,token_index,embedding,err);
}

/* Ring decode.  Rank 0 embeds the token and ships the 40 KiB hyper state to
 * the first block owner; every owner executes its six layers from its own
 * authoritative GDN/QSA/PLE state and forwards the state along the chain.
 * Rank 0 executes its own block when the chain reaches it, then samples the
 * token from the final block owner's result exactly like the legacy path. */
static fg_status coordinator_output_slice(fg_coordinator *coordinator,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    uint32_t token_index,fg_error *err){
    const fg_manifest *manifest=coordinator->manifest;
    fg_status status=fg_output_split_require_slice(coordinator->output_slice!=NULL,
        FG_OUTPUT_SPLIT_WAYS_MIN,0u,err);
    if(status!=FG_OK)return status;
    if(fg_output_slice_ways(coordinator->output_slice)!=FG_OUTPUT_SPLIT_WAYS_MIN){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "rank 0 holds a %u-way slice executor for a 2-way output slice (FG_OUTPUT_SPLIT must match on all ranks)",
                     fg_output_slice_ways(coordinator->output_slice));
        return FG_ERR_MISMATCH;
    }
    fg_layer_result slice;
    status=fg_decode_layer_result_decode(&slice,payload,bytes,err);
    if(status==FG_OK&&(slice.destination_rank!=0u||slice.layer!=FG_LAYER_COUNT-1u||
       slice.source_rank!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
       slice.source_rank!=peer||slice.token_index!=token_index||
       fg_frame_request_id(header)!=coordinator->session_id||
       fg_frame_sequence(header)!=token_index*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(err,FG_ERR_MISMATCH,"misrouted ring decode output slice");
        status=FG_ERR_MISMATCH;
    }
    fg_output_partial partial={.token_index=token_index};
    if(status==FG_OK)status=fg_output_slice_run(coordinator->output_slice,slice.hyper,
        &partial.value,&partial.id,err);
    uint8_t wire[FG_OUTPUT_PARTIAL_BYTES];
    if(status==FG_OK)status=fg_output_partial_encode(wire,&partial,err);
    if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,4u,FG_FABRIC_CONTROL,
        FG_MSG_OUTPUT_PARTIAL,coordinator->session_id,
        token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,sizeof(wire),err);
    return status;
}

static fg_status coordinator_output_slice_hidden(fg_coordinator *coordinator,uint32_t peer,
    const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,
    uint32_t token_index,fg_error *err){
    const fg_manifest *manifest=coordinator->manifest;
    fg_status status=fg_output_split_require_slice(coordinator->output_slice!=NULL,
        FG_OUTPUT_SPLIT_WAYS_MAX,0u,err);
    if(status!=FG_OK)return status;
    if(fg_output_slice_ways(coordinator->output_slice)!=FG_OUTPUT_SPLIT_WAYS_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "rank 0 holds a %u-way slice executor for a 4-way output slice (FG_OUTPUT_SPLIT must match on all ranks)",
                     fg_output_slice_ways(coordinator->output_slice));
        return FG_ERR_MISMATCH;
    }
    fg_output_slice_hidden slice;
    status=fg_output_slice_hidden_decode(&slice,payload,bytes,err);
    if(status==FG_OK&&(slice.destination_rank!=0u||
       slice.source_rank!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
       slice.source_rank!=peer||slice.token_index!=token_index||
       fg_frame_request_id(header)!=coordinator->session_id||
       fg_frame_sequence(header)!=token_index*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(err,FG_ERR_MISMATCH,"misrouted ring decode output slice");
        status=FG_ERR_MISMATCH;
    }
    fg_output_partial partial={.token_index=token_index};
    if(status==FG_OK)status=fg_output_slice_run_hidden(coordinator->output_slice,
        slice.hidden,&partial.value,&partial.id,err);
    if(status==FG_OK)output_split_trace("slice",0u,token_index,partial.id);
    uint8_t wire[FG_OUTPUT_PARTIAL_BYTES];
    if(status==FG_OK)status=fg_output_partial_encode(wire,&partial,err);
    if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,4u,FG_FABRIC_CONTROL,
        FG_MSG_OUTPUT_PARTIAL,coordinator->session_id,
        token_index*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,sizeof(wire),err);
    return status;
}

static fg_status coordinator_decode_token_ring(fg_coordinator *coordinator,
    const int32_t *history,size_t history_count,uint32_t token_index,
    uint32_t *next_token,float *logit,fg_error *err){
    const fg_manifest *manifest=coordinator->manifest;
    if(!history||!history_count||(uint32_t)history[history_count-1u]>=FG_Q38_VOCAB_SIZE){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid ring decode token history");
        return FG_ERR_ARGUMENT;
    }
    if(manifest->layer_owner[0u]==0u){
        fg_error_set(err,FG_ERR_MISMATCH,"ring decode requires a remote first block");
        return FG_ERR_MISMATCH;
    }
    fg_status status=FG_OK;
    fg_vk_context *vk=fg_model_vk(coordinator->model);
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    fg_vk_tensor *input=fg_owner_prefill_input(coordinator->owner);
    if(!embedding){
        fg_error_set(err,FG_ERR_MISMATCH,"coordinator is missing token_embd.weight");
        return FG_ERR_MISMATCH;
    }
    if(!input){
        fg_error_set(err,FG_ERR_MISMATCH,"coordinator ring decode input storage is unavailable");
        return FG_ERR_MISMATCH;
    }
    bool direct=decode_direct_output_eligible(manifest);
    bool trace=decode_ring_trace_enabled();double t0=trace?dispatch_ts():0.0;
    double t_embed=0.0,t_ngram=0.0;
    fg_vk_tensor *ngram_view=NULL;
    if(status==FG_OK)status=fg_vk_embedding_q8_0(vk,input,embedding,
        (uint32_t)history[history_count-1u],FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,
        FG_Q38_HYPER_COUNT,err);
    if(trace)t_embed=dispatch_ts();
    if(status==FG_OK)status=fg_ngram_store_lookup_prefill(coordinator->ngram,history,
        history_count,token_index,1u,&ngram_view,err);
    if(trace)t_ngram=dispatch_ts();
    fg_layer_work *work=&coordinator->decode_work;
    memset(work,0,sizeof(*work));
    work->layer=0u;work->source_rank=0u;work->destination_rank=manifest->layer_owner[0u];
    work->flags=ngram_view?FG_LAYER_WORK_HAS_NGRAM:0u;work->position_mode=FG_POSITION_TEXT;
    if(direct&&coordinator->output_slice&&
       fg_output_slice_ways(coordinator->output_slice)==FG_OUTPUT_SPLIT_WAYS_MAX&&
       coordinator->sampler.temperature==0.0f&&
       !fg_sampler_penalties_active(&coordinator->sampler))
        work->flags|=FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY;
    work->token_index=token_index;
    for(uint32_t axis=0;axis<3u;axis++)work->position[axis]=token_index;
    if(status==FG_OK)status=fg_vk_tensor_read(input,0,work->hyper,
        (uint64_t)FG_HYPER_WIDTH*4u,err);
    if(status==FG_OK&&ngram_view){
        status=fg_vk_tensor_read(ngram_view,0,work->ngram_embedding,
            (uint64_t)FG_NGRAM_EMBED_VALUES*4u,err);
        numerics_trace_values("NGRAM_DECODE",0u,1u,token_index,work->ngram_embedding,
            FG_NGRAM_EMBED_VALUES);
    }
    uint32_t wire_bytes=0;
    if(status==FG_OK)status=fg_decode_layer_work_encode(coordinator->decode_work_wire,
        FG_DECODE_LAYER_WORK_MAX_BYTES,&wire_bytes,manifest->protocol_version,work,err);
    if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,work->destination_rank,
        FG_FABRIC_BULK,FG_MSG_DECODE_LAYER_WORK,coordinator->session_id,
        token_index*FG_LAYER_COUNT+0u,0,coordinator->decode_work_wire,wire_bytes,err);
    if(status==FG_OK&&direct)status=coordinator_output_config(coordinator,token_index,err);
    double t_sent=trace?dispatch_ts():0.0;
    worker_decode_dispatch dispatch={.expert=coordinator->expert,
        .manifest=manifest,.self=0u,
        .results=fg_owner_decode_results(coordinator->owner)};
    bool have_result=false;
    double t_own_recv=0.0,t_own_end=0.0,t_final=0.0,t_own_run=0.0,t_own_read=0.0;
    uint32_t own_first=0u,own_last=0u;
    bool split_timeout=direct&&coordinator->output_slice&&
        fg_output_slice_ways(coordinator->output_slice)==FG_OUTPUT_SPLIT_WAYS_MAX;
    uint32_t split_timeout_ms=split_timeout?
        output_split_timeout_ms()+FG_OUTPUT_SPLIT_TIMEOUT_GRACE_MS:0u;
    uint64_t split_deadline=split_timeout?monotonic_ms()+split_timeout_ms:0u;
    while(status==FG_OK&&!have_result){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        if(split_timeout){
            uint64_t now=monotonic_ms();
            int32_t remaining=now<split_deadline?(int32_t)(split_deadline-now):0;
            status=fg_fabric_recv_any_timeout(coordinator->fabric,FG_FABRIC_BULK,remaining,
                &peer,&header,coordinator->decode_work_wire,
                FG_DECODE_LAYER_WORK_MAX_BYTES,&bytes,err);
            if(status==FG_ERR_LIMIT){
                fg_error_set(err,FG_ERR_LIMIT,
                    "ring decode timed out after %u ms waiting for the 4-way result from rank 4",
                    split_timeout_ms);
                fprintf(stderr,"RING_DECODE_TIMEOUT rank=0 token=%u waited_ms=%u expected=OUTPUT_RESULT peer=4\n",
                        token_index,split_timeout_ms);
            }
        }else status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_BULK,&peer,&header,
            coordinator->decode_work_wire,FG_DECODE_LAYER_WORK_MAX_BYTES,&bytes,err);
        if(status!=FG_OK)break;
        fg_message_type type=fg_frame_type(&header);
        if(type==FG_MSG_DECODE_LAYER_WORK){
            fg_status decode_status=fg_decode_layer_work_decode(work,
                manifest->protocol_version,coordinator->decode_work_wire,bytes,err);
            if(decode_status!=FG_OK){status=decode_status;break;}
            if(work->destination_rank!=0u||manifest->layer_owner[work->layer]!=0u||
               peer!=work->source_rank||
               (work->layer>0u&&peer!=manifest->layer_owner[work->layer-1u])||
               (work->layer>0u&&manifest->layer_owner[work->layer-1u]==0u)||
               fg_frame_request_id(&header)!=coordinator->session_id||
               fg_frame_sequence(&header)!=work->token_index*FG_LAYER_COUNT+work->layer||
               work->token_index!=token_index){
                fg_error_set(err,FG_ERR_MISMATCH,"misrouted ring decode layer work");
                status=FG_ERR_MISMATCH;
                break;
            }
            if(trace)t_own_recv=dispatch_ts();
            numerics_trace_host("FB_IN",0u,work->layer,work->token_index,1u,work->hyper);
            own_first=work->layer;
            uint32_t last=work->layer;
            while(last+1u<FG_LAYER_COUNT&&manifest->layer_owner[last+1u]==0u)last++;
            own_last=last;
            /* The coordinator executes its own block inline on the base owner
             * slot; the block state was advanced by this rank in ring prefill. */
            status=fg_vk_tensor_write(input,0,work->hyper,(uint64_t)FG_HYPER_WIDTH*4u,err);
            fg_vk_tensor *current=NULL;
            bool own_profile=decode_profile_enabled();
            if(status==FG_OK&&own_profile){
                fg_error profile_error={0};
                if(fg_vk_profile_begin(vk,&profile_error)!=FG_OK)own_profile=false;
            }
            if(status==FG_OK&&decode_block_chain_eligible(dispatch.expert,manifest,0u,
                own_first,last)){
                status=fg_owner_decode_block_chained(coordinator->owner,own_first,last,
                    work->token_index,work->position,input,NULL,chained_decode_expert,
                    &dispatch,&current,err);
            }else if(status==FG_OK)status=fg_owner_decode_block(coordinator->owner,own_first,last,
                work->token_index,work->position,input,NULL,worker_decode_fire,
                worker_decode_collect,&dispatch,&current,err);
            if(own_profile){
                fg_vk_profile profile={0};fg_error profile_error={0};
                fg_status profile_status=fg_vk_profile_end(vk,&profile,
                    status==FG_OK?err:&profile_error);
                if(profile_status==FG_OK){
                    fprintf(stderr,"DECODE_PROFILE rank=0 token=%u layers=%u..%u gpu_ms=%.3f "
                        "kernel_ms=%.3f wait_ms=%.3f record_ms=%.3f op_ms=%.3f barrier_ms=%.3f "
                        "submit_ms=%.3f gap_ms=%.3f gap_max_ms=%.3f gap_max=%s submissions=%llu dispatches=%llu\n",
                        work->token_index,own_first,last,profile.gpu_ms,profile.kernel_ms,
                        profile.wait_ms,profile.record_ms,profile.op_ms,profile.barrier_ms,
                        profile.submit_ms,profile.gap_ms,profile.gap_max_ms,
                        profile.gap_max_kernel?profile.gap_max_kernel:"-",
                        (unsigned long long)profile.submissions,
                        (unsigned long long)profile.dispatches);
                    for(uint32_t k=0;k<profile.kernel_count;k++)
                        fprintf(stderr,"DECODE_PROFILE_KERNEL rank=0 scope=%s kernel=%s "
                            "calls=%llu gpu_ms=%.3f\n",profile.kernels[k].scope,
                            profile.kernels[k].name,
                            (unsigned long long)profile.kernels[k].invocations,
                            profile.kernels[k].gpu_ms);
                }
            }
            if(trace)t_own_run=dispatch_ts();
            if(status==FG_OK)status=fg_vk_tensor_read(current,0,work->hyper,
                (uint64_t)FG_HYPER_WIDTH*4u,err);
            if(trace)t_own_read=dispatch_ts();
            numerics_trace_host("FB_OUT",0u,last,work->token_index,1u,work->hyper);
            if(trace)t_own_end=dispatch_ts();
            if(status==FG_OK&&last+1u<FG_LAYER_COUNT){
                fg_layer_work next={.layer=(uint8_t)(last+1u),.source_rank=0u,
                    .destination_rank=manifest->layer_owner[last+1u],
                    .token_index=work->token_index,.position_mode=FG_POSITION_TEXT,
                    .flags=(uint8_t)(work->flags&FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)};
                for(uint32_t axis=0;axis<3u;axis++)next.position[axis]=work->position[axis];
                memcpy(next.hyper,work->hyper,(uint64_t)FG_HYPER_WIDTH*4u);
                uint32_t next_bytes=0;
                status=fg_decode_layer_work_encode(coordinator->decode_work_wire,
                    FG_DECODE_LAYER_WORK_MAX_BYTES,&next_bytes,
                    manifest->protocol_version,&next,err);
                if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,
                    next.destination_rank,FG_FABRIC_BULK,FG_MSG_DECODE_LAYER_WORK,
                    coordinator->session_id,work->token_index*FG_LAYER_COUNT+next.layer,
                    0,coordinator->decode_work_wire,next_bytes,err);
            }else if(status==FG_OK){
                fg_layer_result *result=&coordinator->decode_result;
                memset(result,0,sizeof(*result));
                result->layer=(uint8_t)last;result->source_rank=0u;
                result->destination_rank=0u;result->token_index=work->token_index;
                memcpy(result->hyper,work->hyper,sizeof(result->hyper));
                uint32_t result_bytes=0u;
                status=fg_decode_layer_result_encode(coordinator->decode_work_wire,
                    result,&result_bytes,err);
                if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,0u,
                    FG_FABRIC_BULK,FG_MSG_DECODE_LAYER_RESULT,coordinator->session_id,
                    work->token_index*FG_LAYER_COUNT+last,0,coordinator->decode_work_wire,
                    result_bytes,err);
            }
            if(trace)t_final=dispatch_ts();
            if(status==FG_OK&&last+1u==FG_LAYER_COUNT)have_result=true;
        }else if(type==FG_MSG_DECODE_LAYER_RESULT&&!direct){
            fg_layer_result *result=&coordinator->decode_result;
            fg_status decode_status=fg_decode_layer_result_decode(result,
                coordinator->decode_work_wire,bytes,err);
            if(decode_status!=FG_OK){status=decode_status;break;}
            if(result->destination_rank!=0u||
               result->source_rank!=manifest->layer_owner[FG_LAYER_COUNT-1u]||
               result->source_rank!=peer||
               fg_frame_request_id(&header)!=coordinator->session_id||
               fg_frame_sequence(&header)!=result->token_index*FG_LAYER_COUNT+result->layer||
               result->token_index!=token_index||
               result->layer!=FG_LAYER_COUNT-1u){
                 fg_error_set(err,FG_ERR_MISMATCH,"misrouted ring decode result");
                 status=FG_ERR_MISMATCH;
                 break;
            }
            have_result=true;
        }else if(type==FG_MSG_OUTPUT_RESULT&&direct){
            if(peer!=4u){
                fg_error_set(err,FG_ERR_MISMATCH,"misrouted direct output result peer");
                status=FG_ERR_MISMATCH;
                break;
            }
            status=coordinator_output_result_accept(coordinator,&header,
                coordinator->decode_work_wire,bytes,token_index,next_token,logit,err);
            if(status!=FG_OK)break;
            have_result=true;
        }else if(type==FG_MSG_OUTPUT_SLICE&&direct){
            status=coordinator_output_slice(coordinator,peer,&header,
                coordinator->decode_work_wire,bytes,token_index,err);
        }else if(type==FG_MSG_OUTPUT_SLICE_HIDDEN&&direct){
            status=coordinator_output_slice_hidden(coordinator,peer,&header,
                coordinator->decode_work_wire,bytes,token_index,err);
        }else if(type==FG_MSG_QSA_PAGE_RESULT){
            status=coordinator_warm_qsa_result(coordinator,peer,&header,
                coordinator->decode_work_wire,bytes,err);
        }else{
            fg_error_set(err,FG_ERR_MISMATCH,"unexpected ring decode message type %u",type);
            status=FG_ERR_MISMATCH;
            break;
        }
        if(trace&&have_result)t_final=dispatch_ts();
    }
    double t_output=t_final;
    if(status==FG_OK&&!direct){
        status=fg_vk_tensor_write(input,0,coordinator->decode_result.hyper,
            (uint64_t)FG_HYPER_WIDTH*4u,err);
        if(status==FG_OK)status=coordinator_output(coordinator,token_index,input,next_token,
            logit,err);
    }
    if(trace){t_output=dispatch_ts();
        fg_vk_counters decode_counters={0};fg_vk_get_counters(vk,&decode_counters);
        fprintf(stderr,"RING_DECODE token=%u embed_ms=%.3f ngram_ms=%.3f send_ms=%.3f "
            "first_hop_ms=%.3f own_layers=%u..%u own_ms=%.3f own_run_ms=%.3f "
            "own_read_ms=%.3f tail_ms=%.3f output_ms=%.3f total_ms=%.3f handoff=%u "
            "submissions_total=%llu dispatches_total=%llu\n",
            token_index,t_sent-t0,t_ngram-t_embed,t_sent-t_ngram,t_own_recv-t_sent,
            own_first,own_last,
            t_own_end-t_own_recv,t_own_run-t_own_recv,t_own_read-t_own_run,
            t_final-t_own_end,t_output-t_final,t_output-t0,direct?1u:0u,
            (unsigned long long)decode_counters.submissions,
            (unsigned long long)decode_counters.dispatches);}
    return status;
}

/* Expert-parallel decode: all 48 layers on the coordinator, MoE dispatched to workers */
static fg_status coordinator_decode_token_local(fg_coordinator *coordinator,const int32_t *history,size_t history_count,uint32_t token_index,uint32_t *next_token,float *logit,fg_error *err){
    if(!history||!history_count||(uint32_t)history[history_count-1u]>=FG_Q38_VOCAB_SIZE){fg_error_set(err,FG_ERR_ARGUMENT,"invalid local decode token history");return FG_ERR_ARGUMENT;}
    fg_vk_context *vk=fg_model_vk(coordinator->model);token_profile_capture capture={0};fg_status status=token_profile_begin(&capture,vk,token_index,err);double frame_start=dispatch_ts();
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    fg_vk_tensor *decode_input=fg_owner_prefill_input(coordinator->owner);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"embedding",err);
    if(status==FG_OK&&!decode_input){fg_error_set(err,FG_ERR_MISMATCH,"coordinator decode input storage is unavailable");status=FG_ERR_MISMATCH;}
    if(status==FG_OK)status=fg_vk_embedding_q8_0(vk,decode_input,embedding,(uint32_t)history[history_count-1u],FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);
    double frame_embedding=dispatch_ts();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"ngram",err);
    fg_vk_tensor *ngram=NULL;if(status==FG_OK)status=coordinator_ngram_resident(coordinator,history,history_count,token_index,&ngram,err);
    if(status==FG_OK&&ngram&&numerics_trace_enabled()){
        float ngram_host[FG_NGRAM_EMBED_VALUES];
        if(fg_vk_tensor_read(ngram,0,ngram_host,sizeof(ngram_host),err)==FG_OK)
            numerics_trace_values("NGRAM_DECODE",0u,1u,token_index,ngram_host,
                FG_NGRAM_EMBED_VALUES);
    }
    double frame_ngram=dispatch_ts();
    uint32_t position[3]={token_index,token_index,token_index};
    fg_vk_tensor *current=decode_input;
    async_expert_context async_ctx={.fabric=coordinator->fabric,.expert=coordinator->expert,
        .manifest=coordinator->manifest,.self=0u,.request_id=coordinator->session_id,
        .transport_state=&coordinator->transport_state,
        .critical_trace=token_profile_requested(token_index)};
    for(uint32_t i=0;i<FG_GROUP_SIZE;i++)async_ctx.recv_payloads[i]=coordinator->async_recv_payloads[i];
    /* Process all 48 layers locally while overlapping shared and routed experts. */
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        const fg_vk_tensor *layer_ngram=(layer==1u)?ngram:NULL;
        async_ctx.sequence=token_index*FG_LAYER_COUNT+layer;
        fg_vk_tensor *layer_out=NULL;
        status=fg_owner_decode_layer_begin(coordinator->owner,0u,layer,token_index,position,current,
            layer_ngram,fire_experts,collect_experts,&async_ctx,NULL,NULL,err);
        if(status==FG_OK)status=fg_owner_decode_layer_finish(coordinator->owner,0u,&layer_out,err);
        if(status==FG_OK)current=layer_out;
    }
    double frame_layers=dispatch_ts();
    if(status==FG_OK)status=coordinator_output(coordinator,token_index,current,next_token,logit,err);
    if(status==FG_OK)status=coordinator_publish_qsa_pages(coordinator,token_index,1u,err);
    double frame_output=dispatch_ts();
    if(token_profile_requested(token_index)||frame_trace_enabled())fprintf(stderr,"TOKEN_FRAME_TRACE token=%u status=%d embedding_ms=%.3f ngram_ms=%.3f layers_ms=%.3f output_ms=%.3f total_ms=%.3f\n",token_index,(int)status,frame_embedding-frame_start,frame_ngram-frame_embedding,frame_layers-frame_ngram,frame_output-frame_layers,frame_output-frame_start);
    status=token_profile_end(&capture,0u,"token",token_index,UINT32_MAX,status,err);
    if(async_ctx.critical_trace)for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){for(uint32_t i=0;i<async_ctx.send_trace_count[layer];i++)fprintf(stderr,"EXPERT_COORD_SEND token=%u layer=%u peer=%u start_ns=%llu end_ns=%llu\n",token_index,layer,async_ctx.send_trace[layer][i].peer,(unsigned long long)async_ctx.send_trace[layer][i].start_ns,(unsigned long long)async_ctx.send_trace[layer][i].end_ns);for(uint32_t i=0;i<async_ctx.recv_trace_count[layer];i++)fprintf(stderr,"EXPERT_COORD_RECV token=%u layer=%u peer=%u bytes=%u ready_mask=%u poll_start_ns=%llu ready_ns=%llu header_end_ns=%llu payload_end_ns=%llu validate_end_ns=%llu decode_end_ns=%llu\n",token_index,layer,async_ctx.recv_trace[layer][i].peer,async_ctx.recv_trace[layer][i].bytes,async_ctx.recv_trace[layer][i].ready_mask,(unsigned long long)async_ctx.recv_trace[layer][i].poll_start_ns,(unsigned long long)async_ctx.recv_trace[layer][i].ready_ns,(unsigned long long)async_ctx.recv_trace[layer][i].header_end_ns,(unsigned long long)async_ctx.recv_trace[layer][i].payload_end_ns,(unsigned long long)async_ctx.recv_trace[layer][i].validate_end_ns,(unsigned long long)async_ctx.recv_trace[layer][i].decode_end_ns);}
    return status;
}

/* Env-gated decode selector: the legacy expert-parallel replay stays the
 * default until the ring path keeps the accuracy gates green. */
static fg_status coordinator_decode_token(fg_coordinator *coordinator,const int32_t *history,
    size_t history_count,uint32_t token_index,uint32_t *next_token,float *logit,
    fg_error *err){
    if(coordinator->ring_decode)
        return coordinator_decode_token_ring(coordinator,history,history_count,token_index,
            next_token,logit,err);
    return coordinator_decode_token_local(coordinator,history,history_count,token_index,
        next_token,logit,err);
}

static void coordinator_close(fg_coordinator *coordinator){if(!coordinator)return;free(coordinator->decode_result_wire);free(coordinator->decode_work_wire);for(uint32_t i=0;i<FG_GROUP_SIZE;i++)free(coordinator->async_recv_payloads[i]);qsa_page_transport_destroy(&coordinator->qsa_pages);for(uint32_t slot=0;slot<FG_PREFILL_FRAMES;slot++){fg_vk_tensor_destroy(coordinator->ring_output[slot]);prefill_layer_buffers_destroy(&coordinator->prefill_layer[slot]);prefill_worker_buffers_destroy(&coordinator->prefill_expert[slot]);}fg_ngram_store_close(coordinator->ngram);fg_tokenizer_close(coordinator->tokenizer);fg_fabric_close(coordinator->fabric);fg_output_slice_destroy(coordinator->output_slice);fg_owner_executor_destroy(coordinator->owner);fg_expert_executor_destroy(coordinator->expert);fg_model_close(coordinator->model);memset(coordinator,0,sizeof(*coordinator));}

/* The coordinator executes its own block's QSA layers in the ring.  A
 * state-backed mirror keeps those pages recoverable after cache eviction
 * instead of pinning them resident for the whole context. */
static fg_status coordinator_open_qsa(fg_coordinator *coordinator,const char *directory,
                                      uint32_t logical_context,uint32_t cache_pages,
                                      fg_error *err){
    const fg_manifest *manifest=coordinator->manifest;
    bool ring=fg_runtime_ring_enabled();
    bool owns_qsa=false;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(manifest->layer_owner[layer]==0u)owns_qsa=true;
    if(!owns_qsa)
        return fg_owner_qsa_open_mirror(coordinator->owner,logical_context,
            coordinator->options.qsa_hot_tokens,cache_pages,manifest->prefill_microbatch,
            coordinator_fetch_qsa_pages,coordinator,err);
    char path[1200];
    if(snprintf(path,sizeof(path),"%s/qsa-owner-rank-00.state",directory)>=
       (int)sizeof(path)){
        fg_error_set(err,FG_ERR_LIMIT,"QSA coordinator state path overflow");
        return FG_ERR_LIMIT;
    }
    unlink(path);
    uint8_t layers[FG_LAYER_COUNT/4u];uint32_t layer_count=0;
    /* Ring decode executes each QSA layer on its block owner, so the state
     * mirror only needs the layers this rank actually runs. */
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(!ring||manifest->layer_owner[layer]==0u)
            layers[layer_count++]=(uint8_t)layer;
    fg_qsa_state *state=NULL;
    fg_status status=fg_qsa_state_open(&state,path,layers,layer_count,logical_context,
                                       true,err);
    fg_qsa_state_close(state);
    if(status!=FG_OK)return status;
    return fg_owner_qsa_open_state_mirror(coordinator->owner,path,logical_context,
        coordinator->options.qsa_hot_tokens,cache_pages,manifest->prefill_microbatch,
        ring,coordinator_fetch_qsa_pages,coordinator,err);
}

static fg_status coordinator_output_split_open(fg_coordinator *coordinator,fg_error *err){
    uint32_t ways=0u;
    fg_status status=fg_output_split_mode(&ways,err);
    if(status!=FG_OK||!ways)return status;
    uint32_t way=0u;
    if(!fg_output_split_way_for_rank(ways,0u,&way)){
        fg_error_set(err,FG_ERR_MISMATCH,"rank 0 owns no slice in the %u-way output split",ways);
        return FG_ERR_MISMATCH;
    }
    uint32_t first_row=0u,rows=0u;
    fg_output_split_span(ways,way,&first_row,&rows);
    status=fg_output_slice_create(&coordinator->output_slice,coordinator->model,ways,
                                  first_row,rows,err);
    if(status==FG_OK)
        fprintf(stderr,"OUTPUT_SPLIT rank=0 ways=%u way=%u rows=%u..%u\n",ways,way,
                first_row,first_row+rows);
    return status;
}

static fg_status coordinator_open(fg_coordinator *coordinator,const fg_manifest *manifest,const char *directory,const fg_runtime_options *options,fg_error *err){memset(coordinator,0,sizeof(*coordinator));coordinator->manifest=manifest;coordinator->options=*options;fg_status status=fg_session_identity_from_manifest(manifest,&coordinator->identity,err);if(status==FG_OK&&manifest->protocol_version<6u){fg_error_set(err,FG_ERR_MISMATCH,"QSA page ownership requires protocol version 6");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_model_open_coordinator(&coordinator->model,manifest,directory,0u,err);if(status==FG_OK)status=coordinator_output_split_open(coordinator,err);if(status==FG_OK)status=fg_owner_executor_create(&coordinator->owner,coordinator->model,err);if(status==FG_OK)status=fg_expert_executor_create(&coordinator->expert,coordinator->model,err);uint32_t cache_page_count=coordinator_qsa_cache_pages(options);if(status==FG_OK&&!cache_page_count){fg_error_set(err,FG_ERR_LIMIT,"QSA record cache has no capacity");status=FG_ERR_LIMIT;}if(status==FG_OK)status=coordinator_open_qsa(coordinator,directory,options->logical_context_tokens,cache_page_count,err);if(status==FG_OK)status=fg_tokenizer_open(&coordinator->tokenizer,directory,manifest,err);if(status==FG_OK)status=fg_tokenizer_validate_qwen38(coordinator->tokenizer,err);const fg_tensor_record *ngram_record=NULL;for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)if(manifest->tensors[i].kind==FG_TENSOR_NGRAM){if(ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"multiple n-gram tensors in deployment manifest");status=FG_ERR_MISMATCH;}else ngram_record=&manifest->tensors[i];}char ngram_path[1200];if(status==FG_OK&&!ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"deployment manifest has no n-gram tensor");status=FG_ERR_MISMATCH;}if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),"%s/ngram.iq4nl",directory)>=(int)sizeof(ngram_path)){fg_error_set(err,FG_ERR_LIMIT,"n-gram path is too long");status=FG_ERR_LIMIT;}uint32_t ngram_store_tokens=manifest->prefill_microbatch<=FG_NGRAM_PREFILL_MAX_TOKENS/FG_PREFILL_FRAMES?FG_PREFILL_FRAMES*manifest->prefill_microbatch:FG_NGRAM_PREFILL_MAX_TOKENS;if(status==FG_OK)status=fg_ngram_store_open(&coordinator->ngram,fg_model_vk(coordinator->model),ngram_path,ngram_record->bytes,ngram_store_tokens,err);
    /* Allocate only coordinator-side asynchronous receive payloads. */
    for(uint32_t i=0;status==FG_OK&&i<FG_GROUP_SIZE;i++){coordinator->async_recv_payloads[i]=malloc(FG_EXPERT_RESULT_SINGLE_BYTES);if(!coordinator->async_recv_payloads[i]){fg_error_set(err,FG_ERR_OOM,"allocate async expert recv buffer %u",i);status=FG_ERR_OOM;}}for(uint32_t slot=0;status==FG_OK&&slot<FG_PREFILL_FRAMES;slot++){status=prefill_worker_buffers_create(&coordinator->prefill_expert[slot],manifest->prefill_microbatch,true,err);if(status==FG_OK)status=prefill_layer_buffers_create(&coordinator->prefill_layer[slot],coordinator->model,manifest->prefill_microbatch,err);if(status==FG_OK&&slot<2u)status=fg_vk_tensor_create(fg_model_vk(coordinator->model),(uint64_t)manifest->prefill_microbatch*FG_HYPER_WIDTH*4u,&coordinator->ring_output[slot],err);}if(status==FG_OK)coordinator->ring_prefill=prefill_ring_requested();if(status==FG_OK)coordinator->ring_decode=coordinator->ring_prefill&&decode_ring_requested();if(status==FG_OK&&coordinator->ring_decode){coordinator->decode_work_wire=malloc(FG_DECODE_LAYER_WORK_MAX_BYTES);coordinator->decode_result_wire=malloc(FG_DECODE_LAYER_RESULT_BYTES);if(!coordinator->decode_work_wire||!coordinator->decode_result_wire){fg_error_set(err,FG_ERR_OOM,"allocate ring decode exchange buffers");status=FG_ERR_OOM;}}if(status==FG_OK)status=fg_fabric_open(&coordinator->fabric,manifest,0u,err);if(status==FG_OK)atomic_init(&coordinator->transport_state,FG_TRANSPORT_READY);if(status==FG_OK)status=qsa_page_transport_create(&coordinator->qsa_pages,coordinator->fabric,&coordinator->transport_state,err);if(status==FG_OK)status=rank_ready(coordinator->fabric,0u,err);if(status==FG_OK)status=token_profile_prepare(fg_model_vk(coordinator->model),err);if(status==FG_OK)status=coordinator_begin_session(coordinator,err);if(status==FG_OK)coordinator_memory_report(coordinator);if(status!=FG_OK)coordinator_close(coordinator);coordinator->directory=directory;return status;}

static double elapsed_seconds(const struct timespec *start,const struct timespec *end){return (double)(end->tv_sec-start->tv_sec)+(double)(end->tv_nsec-start->tv_nsec)*1e-9;}

static fg_status runtime_reserve_history(fg_runtime *runtime,size_t count,fg_error *err){
    if(count<=runtime->history_capacity)return FG_OK;
    size_t capacity=runtime->history_capacity?runtime->history_capacity:1024u;
    while(capacity<count){if(capacity>SIZE_MAX/2u){fg_error_set(err,FG_ERR_LIMIT,"session token history exceeds address space");return FG_ERR_LIMIT;}capacity*=2u;}
    int32_t *history=realloc(runtime->history,capacity*sizeof(*history));
    if(!history){fg_error_set(err,FG_ERR_OOM,"grow session token history");return FG_ERR_OOM;}
    runtime->history=history;runtime->history_capacity=capacity;return FG_OK;
}

static fg_status runtime_reset_state(fg_runtime *runtime,fg_prefix_reset_reason reason,
                                     fg_error *err){
    fg_status status=FG_OK;

    if(!transport_ready(&runtime->coordinator.transport_state)&&
       runtime->coordinator.qsa_pages.warm_outstanding){
        fg_error drain_error={0};
        /* Ring prefill can complete with mirror-warm fetches still in flight if
         * a request aborted before its drain; collect them so the next session
         * does not inherit a pending transport. */
        status=coordinator_warm_qsa_drain(&runtime->coordinator,&drain_error);
        if(status!=FG_OK){
            runtime->state_ready=false;
            if(err)*err=drain_error;
            return status;
        }
    }
    if(!transport_ready(&runtime->coordinator.transport_state)){
        runtime->state_ready=false;
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "distributed transport is not reusable; reopen the runtime");
        return FG_ERR_UNAVAILABLE;
    }
    status=fg_owner_reset_state(runtime->coordinator.owner,err);
    if(status==FG_OK&&runtime->session_started)
        status=coordinator_begin_session(&runtime->coordinator,err);

    if(status!=FG_OK){
        runtime->state_ready=false;
        return status;
    }
    runtime->state_ready=false;
    runtime->history_count=0;
    runtime->state_frontier=0;
    runtime->next_token_valid=false;
    runtime->next_token=0;
    runtime->next_logit=0.0f;
    runtime->empty_reason=reason;
    free(runtime->rendered_history);
    runtime->rendered_history=NULL;
    runtime->rendered_history_length=0;
    runtime->pending_boundary_bytes=0;
    runtime->pending_eos_token=0;
    runtime->pending_eos_valid=false;
    if(status==FG_OK)runtime->session_started=true;
    if(status!=FG_OK)return status;
    runtime->state_ready=true;
    return FG_OK;
}

fg_status fg_runtime_open_with_options(fg_runtime **out,const char *path,
                                       const fg_runtime_options *requested,fg_error *err){
    if(!out||!path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime open arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    fg_runtime *runtime=calloc(1,sizeof(*runtime));if(!runtime){fg_error_set(err,FG_ERR_OOM,"allocate resident runtime");return FG_ERR_OOM;}
    fg_sampler_config_greedy(&runtime->sampler);
    fg_status status=load_checked(path,&runtime->manifest,err);
    if(status==FG_OK)status=fg_runtime_options_resolve(&runtime->options,runtime->manifest,
                                                       requested,err);
    if(status==FG_OK){
        bool requested_mtp=
            (runtime->options.experimental_flags&FG_RUNTIME_EXPERIMENTAL_MTP)!=0;
        bool sealed_mtp=(runtime->manifest->flags&FG_MANIFEST_HAS_MTP)!=0;
        runtime->mtp_capability=sealed_mtp?FG_MTP_CAPABILITY_WEIGHTS_SEALED:
            FG_MTP_CAPABILITY_UNSUPPORTED;
        if(requested_mtp){
            fg_error_set(err,FG_ERR_UNAVAILABLE,
                         "sealed MTP weights require the MTP head kernels described in "
                         "MTP_FEASIBILITY_2026-09-13.md; unset --experimental-mtp");
            status=FG_ERR_UNAVAILABLE;
        }
    }
    if(status==FG_OK)runtime->context_limit=runtime->options.logical_context_tokens;
    if(status==FG_OK)runtime->prefix_continuation=!runtime->options.no_prefix_continuation;
    if(status==FG_OK)status=manifest_directory(path,runtime->directory,err);
    if(status==FG_OK)
        status=coordinator_open(&runtime->coordinator,runtime->manifest,
                                runtime->directory,&runtime->options,err);
    if(status==FG_OK)status=runtime_reset_state(runtime,FG_PREFIX_RESET_COLD_START,err);
    if(status!=FG_OK){fg_runtime_close(runtime);return status;}*out=runtime;return FG_OK;
}

fg_status fg_runtime_open(fg_runtime **out,const char *path,fg_error *err){
    return fg_runtime_open_with_options(out,path,NULL,err);
}

static fg_status sync_output_history(fg_fabric *fabric,uint32_t owner,uint64_t session_id,
                                     const uint32_t *tokens,uint32_t count,fg_error *err){
    if(!fabric||owner>=FG_RANK_COUNT||!session_id||count>FG_NATIVE_CONTEXT||
       (count&&!tokens)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid output history sync");return FG_ERR_ARGUMENT;}
    uint8_t *wire=malloc(FG_OUTPUT_HISTORY_MAX_BYTES);
    if(!wire){fg_error_set(err,FG_ERR_OOM,"allocate output history wire");return FG_ERR_OOM;}
    fg_output_history history={.tokens=tokens,.count=count};uint32_t bytes=0;
    fg_status status=fg_output_history_encode(wire,FG_OUTPUT_HISTORY_MAX_BYTES,&bytes,&history,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,owner,FG_FABRIC_CONTROL,
        FG_MSG_OUTPUT_HISTORY,session_id,0u,0u,wire,bytes,err);
    uint8_t ack[1];fg_frame_header header;uint32_t ack_bytes=0;
    if(status==FG_OK)status=fg_fabric_recv(fabric,owner,FG_FABRIC_CONTROL,&header,
        ack,sizeof(ack),&ack_bytes,err);
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_HISTORY_ACK||
        fg_frame_request_id(&header)!=session_id||fg_frame_sequence(&header)!=0u||ack_bytes!=0u)){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid output history acknowledgement");status=FG_ERR_MISMATCH;
    }
    free(wire);return status;
}

fg_status fg_runtime_set_sampler(fg_runtime *runtime,const fg_sampler_config *config,
                                 fg_error *err){
    if(!runtime||!config){fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime sampler");return FG_ERR_ARGUMENT;}
    fg_status status=fg_sampler_config_validate(config,err);
    if(status!=FG_OK)return status;
    runtime->sampler=*config;
    return FG_OK;
}

void fg_runtime_close(fg_runtime *runtime){
    if(!runtime)return;
    coordinator_close(&runtime->coordinator);
    free(runtime->rendered_history);free(runtime->history);free(runtime->manifest);free(runtime);
}

fg_status fg_runtime_reset(fg_runtime *runtime,fg_error *err){
    if(!runtime){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
    return runtime_reset_state(runtime,FG_PREFIX_RESET_EXPLICIT,err);
}

fg_status fg_runtime_reset_public_history(fg_runtime *runtime,fg_error *err){
    if(!runtime){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
    return runtime_reset_state(runtime,FG_PREFIX_RESET_PUBLIC_MISMATCH,err);
}

fg_status fg_runtime_reset_failure(fg_runtime *runtime,fg_error *err){
    if(!runtime){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
    return runtime_reset_state(runtime,FG_PREFIX_RESET_FAILURE,err);
}

static fg_status runtime_render_append(char **rendered,size_t *length,size_t *capacity,
                                       const char *text,size_t bytes,fg_error *err){
    if(bytes>SIZE_MAX-*length-1u){
        fg_error_set(err,FG_ERR_LIMIT,"rendered runtime transcript exceeds address space");
        return FG_ERR_LIMIT;
    }
    size_t required=*length+bytes+1u;
    if(required>*capacity){
        size_t grown=*capacity?*capacity:1024u;
        while(grown<required){
            if(grown>SIZE_MAX/2u){grown=required;break;}
            grown*=2u;
        }
        char *data=realloc(*rendered,grown);
        if(!data){fg_error_set(err,FG_ERR_OOM,"grow rendered runtime transcript");return FG_ERR_OOM;}
        *rendered=data;*capacity=grown;
    }
    if(bytes)memcpy(*rendered+*length,text,bytes);
    *length+=bytes;(*rendered)[*length]=0;
    return FG_OK;
}

static fg_tokenizer *runtime_tokenizer(const fg_runtime *runtime){
    return runtime->coordinator.tokenizer;
}

static fg_status runtime_generate_tokens(
    fg_runtime *runtime,const char *transcript,const fg_tokens *prompt,
    bool require_prefix_hit,bool *prefix_miss,
    uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err){
    if(!runtime)return FG_ERR_ARGUMENT;
    runtime->coordinator.sampler=runtime->sampler;
    fg_sampler_state_init(&runtime->coordinator.sampler_state,runtime->sampler.seed);
    if(prefix_miss)*prefix_miss=false;
    if(!runtime||!transcript||!prompt||(!prompt->data&&prompt->count)||!callback||
       !max_tokens){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident generation arguments");return FG_ERR_ARGUMENT;}
    if(!runtime->state_ready){fg_error_set(err,FG_ERR_MISMATCH,"resident runtime requires a successful reset");return FG_ERR_MISMATCH;}
    if(stats){
        memset(stats,0,sizeof(*stats));
        stats->execution_mode=FG_EXECUTION_EXPERT_PARALLEL;
    }
    fg_status status=FG_OK;
    fg_prefix_plan plan={0};
    if(status==FG_OK)status=fg_prefix_plan_tokens(
        runtime->history,runtime->history_count,runtime->next_token_valid,
        prompt->data,prompt->count,runtime->empty_reason,&plan,err);
    /* Ring prefill advances GDN/PLE/QSA state on the block owners.  With ring
     * decode those owners remain authoritative across requests, so a request
     * whose history extends the previous one can resume from the recorded
     * frontier and prefill only the appended tokens.  Without ring decode a
     * reused prefix would resume owners from a stale frontier, so those
     * configurations keep the cold reset.  The frontier guard is belt and
     * braces: if rank 0's token history and the owners' state ever disagree,
     * demote to a cold reset instead of prefilling from the wrong offset. */
    if(status==FG_OK&&runtime->coordinator.ring_prefill&&plan.hit){
        bool resumable=runtime->coordinator.ring_decode&&
            runtime->prefix_continuation&&
            plan.prefill_offset==(size_t)runtime->state_frontier;
        if(!resumable){
            plan.hit=false;plan.exact_frontier=false;plan.reused_tokens=0;
            plan.prefill_offset=0;plan.prefill_tokens=prompt->count;
            plan.reset_reason=runtime->coordinator.ring_decode?
                FG_PREFIX_RESET_FRONTIER_UNAVAILABLE:FG_PREFIX_RESET_COLD_START;
        }
    }
    if(status==FG_OK&&require_prefix_hit&&!plan.hit){
        if(prefix_miss)*prefix_miss=true;
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "runtime-owned continuation is not an exact token-prefix hit");
        status=FG_ERR_UNAVAILABLE;
    }
    if(status==FG_OK&&(!prompt->count||prompt->count+(size_t)max_tokens>runtime->context_limit)){
        fg_error_set(err,FG_ERR_LIMIT,"prompt plus generation would use %zu of %u context tokens",
                     prompt->count+(size_t)max_tokens,runtime->context_limit);
        status=FG_ERR_LIMIT;
    }
    if(status==FG_OK)status=runtime_reserve_history(
        runtime,prompt->count+(size_t)max_tokens,err);
    if(status!=FG_OK)return status;
    size_t candidate_length=strlen(transcript);
    size_t candidate_capacity=candidate_length+1u;
    char *candidate=malloc(candidate_capacity);
    if(!candidate){fg_error_set(err,FG_ERR_OOM,"copy rendered runtime transcript");return FG_ERR_OOM;}
    memcpy(candidate,transcript,candidate_capacity);

    size_t old_count=runtime->history_count;
    bool old_next_valid=runtime->next_token_valid;
    uint32_t old_next=runtime->next_token;
    float old_logit=runtime->next_logit;
    bool state_mutated=false;
    if(!plan.hit&&old_count){
        status=runtime_reset_state(runtime,plan.reset_reason,err);
        if(status!=FG_OK){free(candidate);return status;}
    }
    size_t prefill_offset=plan.hit?plan.prefill_offset:0u;
    for(size_t i=prefill_offset;i<prompt->count;i++)
        runtime->history[i]=(int32_t)prompt->data[i];
    runtime->history_count=prompt->count;

    if(status==FG_OK&&fg_sampler_penalties_active(&runtime->sampler))
        status=sync_output_history(runtime->coordinator.fabric,4u,
            runtime->coordinator.session_id,prompt->data,(uint32_t)prompt->count,err);

    if(stats){
        stats->prompt_tokens=(uint32_t)prompt->count;
        stats->prefilled_tokens=(uint32_t)(prompt->count-prefill_offset);
        stats->reused_tokens=(uint32_t)plan.reused_tokens;
        stats->prefix_cache_hit=plan.hit;
        stats->exact_frontier=plan.exact_frontier;
        stats->reset_reason=plan.reset_reason;
    }

    struct timespec prefill_start={0},prefill_end={0},decode_start={0},decode_end={0};
    uint32_t next=runtime->next_token;
    float logit=runtime->next_logit;
    fg_vk_tensor *prefill_output=NULL;
    if(prefill_offset<prompt->count){
        state_mutated=true;
        clock_gettime(CLOCK_MONOTONIC,&prefill_start);
    }
    if(status==FG_OK&&prefill_offset<prompt->count)
        status=coordinator_prefill_pipeline(&runtime->coordinator,runtime->history,
            runtime->history_count,prompt->data+(size_t)prefill_offset,
            (uint32_t)prefill_offset,(uint32_t)(prompt->count-prefill_offset),
            &runtime->prefill_profiled,&prefill_output,err);
    prefill_worker_buffers_release_result_wire(&runtime->coordinator.prefill_expert[0]);
    prefill_worker_buffers_release_result_wire(&runtime->coordinator.prefill_expert[1]);
    prefill_worker_buffers_release_result_wire(&runtime->coordinator.prefill_expert[2]);
    fg_vk_tensor *last_hyper=NULL;
    if(status==FG_OK&&prefill_offset<prompt->count){
        uint32_t prefilled=(uint32_t)(prompt->count-prefill_offset);
        uint32_t final_count=prefilled%runtime->manifest->prefill_microbatch;
        if(!final_count)final_count=runtime->manifest->prefill_microbatch;
        status=fg_vk_tensor_view(prefill_output,(uint64_t)(final_count-1u)*FG_HYPER_WIDTH*4u,
                                 FG_HYPER_WIDTH*4u,&last_hyper,err);
    }
    if(status==FG_OK&&last_hyper)
        status=coordinator_output(&runtime->coordinator,(uint32_t)runtime->history_count-1u,
                                  last_hyper,&next,&logit,err);
    fg_vk_tensor_destroy(last_hyper);
    if(status==FG_OK&&prefill_offset<prompt->count){
        runtime->next_token=next;
        runtime->next_logit=logit;
        runtime->next_token_valid=true;
        clock_gettime(CLOCK_MONOTONIC,&prefill_end);
        if(stats)stats->prefill_seconds=elapsed_seconds(&prefill_start,&prefill_end);
    }
    if(status==FG_OK)clock_gettime(CLOCK_MONOTONIC,&decode_start);
    uint32_t generated=0;
    bool stopped_on_eos=false;
    size_t pending_boundary_bytes=0;
    uint32_t pending_eos=0;
    bool generate_trace=getenv("FG_GENERATE_TRACE")!=NULL;
    double gt_interrupt=0.0,gt_tokenizer=0.0,gt_callback=0.0,gt_render=0.0,gt_decode=0.0;
    double gt_loop=generate_trace?dispatch_ts():0.0;
    while(status==FG_OK&&generated<max_tokens){
        double t_mark=generate_trace?dispatch_ts():0.0;
        if(interrupted&&interrupted(interrupt_context))break;
        if(generate_trace){gt_interrupt+=dispatch_ts()-t_mark;t_mark=dispatch_ts();}
        if(next==fg_tokenizer_eos(runtime->coordinator.tokenizer)){
            const char *eos_text=NULL;size_t eos_bytes=0;
            status=fg_tokenizer_token(runtime->coordinator.tokenizer,next,&eos_text,
                                      &eos_bytes,NULL,err);
            if(status==FG_OK)status=runtime_render_append(&candidate,&candidate_length,
                                                          &candidate_capacity,eos_text,
                                                          eos_bytes,err);
            if(status==FG_OK)status=runtime_render_append(&candidate,&candidate_length,
                                                          &candidate_capacity,"\n",1u,err);
            if(status==FG_OK){
                stopped_on_eos=true;
                pending_boundary_bytes=eos_bytes+1u;
                pending_eos=next;
            }
            break;
        }
        char decoded[4096];size_t bytes=0;status=fg_tokenizer_decode_token(runtime->coordinator.tokenizer,next,decoded,sizeof(decoded),&bytes,err);
        if(generate_trace){gt_tokenizer+=dispatch_ts()-t_mark;t_mark=dispatch_ts();}
        if(status==FG_OK)status=callback(callback_context,next,decoded,bytes,err);
        if(generate_trace){gt_callback+=dispatch_ts()-t_mark;t_mark=dispatch_ts();}
        if(status==FG_OK)status=runtime_render_append(&candidate,&candidate_length,
                                                      &candidate_capacity,decoded,bytes,err);
        if(generate_trace){gt_render+=dispatch_ts()-t_mark;t_mark=dispatch_ts();}
        if(status!=FG_OK)break;
        runtime->history[runtime->history_count++]=(int32_t)next;generated++;
        state_mutated=true;
        status=coordinator_decode_token(&runtime->coordinator,runtime->history,
            runtime->history_count,(uint32_t)runtime->history_count-1u,&next,&logit,err);
        if(generate_trace)gt_decode+=dispatch_ts()-t_mark;
        if(status==FG_OK){
            runtime->next_token=next;
            runtime->next_logit=logit;
            runtime->next_token_valid=true;
        }
    }
    if(status==FG_OK){
        clock_gettime(CLOCK_MONOTONIC,&decode_end);
        runtime->empty_reason=FG_PREFIX_RESET_NONE;
        runtime->state_frontier=(uint32_t)runtime->history_count;
        free(runtime->rendered_history);
        runtime->rendered_history=candidate;
        runtime->rendered_history_length=candidate_length;
        runtime->pending_boundary_bytes=pending_boundary_bytes;
        runtime->pending_eos_token=pending_eos;
        runtime->pending_eos_valid=stopped_on_eos;
        candidate=NULL;
        if(stats){
            stats->generated_tokens=generated;
            stats->context_tokens=(uint32_t)runtime->history_count;
            stats->decode_seconds=elapsed_seconds(&decode_start,&decode_end);
        }
    }else if(state_mutated&&transport_ready(&runtime->coordinator.transport_state)){
        fg_error reset_error={0};
        fg_status reset_status=runtime_reset_state(runtime,FG_PREFIX_RESET_FAILURE,&reset_error);
        if(reset_status!=FG_OK){*err=reset_error;status=reset_status;}
    }else if(state_mutated){
        runtime->state_ready=false;
    }else{
        runtime->history_count=old_count;
        runtime->next_token_valid=old_next_valid;
        runtime->next_token=old_next;
        runtime->next_logit=old_logit;
    }
    if(generate_trace)fprintf(stderr,"GENERATE_TRACE tokens=%u interrupt_ms=%.3f "
        "tokenizer_ms=%.3f callback_ms=%.3f render_ms=%.3f decode_ms=%.3f total_ms=%.3f\n",
        generated,gt_interrupt,gt_tokenizer,gt_callback,gt_render,gt_decode,
        dispatch_ts()-gt_loop);
    free(candidate);return status;
}

fg_status fg_runtime_generate(fg_runtime *runtime,const char *transcript,uint32_t max_tokens,
                              fg_token_callback callback,void *callback_context,
                              fg_interrupt_fn interrupted,void *interrupt_context,
                              fg_generation_stats *stats,fg_error *err){
    fg_tokens prompt={0};
    fg_status status=runtime&&transcript?
        fg_tokenizer_encode(runtime_tokenizer(runtime),transcript,true,&prompt,err):
        FG_ERR_ARGUMENT;
    if(status==FG_ERR_ARGUMENT)
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident generation arguments");
    if(status==FG_OK)
        status=runtime_generate_tokens(runtime,transcript,&prompt,false,NULL,max_tokens,
                                       callback,callback_context,interrupted,
                                       interrupt_context,stats,err);
    fg_tokens_free(&prompt);
    return status;
}

static size_t runtime_first_token_mismatch(const fg_tokens *left,const fg_tokens *right){
    size_t common=left->count<right->count?left->count:right->count;
    size_t index=0;
    while(index<common&&left->data[index]==right->data[index])index++;
    return index;
}

static const char *runtime_token_relation(const fg_tokens *left,const fg_tokens *right,
                                          size_t mismatch){
    if(mismatch<left->count&&mismatch<right->count)return "token-mismatch";
    if(left->count!=right->count)return "length-mismatch";
    return "exact";
}

static void runtime_trace_continuation(const fg_runtime *runtime,
                                       const char *public_transcript,
                                       const char *private_transcript,
                                       const fg_tokens *suffix_tokens,
                                       const fg_tokens *constructed){
    if(!prefix_trace_enabled())return;
    fg_tokens public_tokens={0},private_tokens={0};
    fg_error ignored={0};
    fg_status public_status=fg_tokenizer_encode(runtime_tokenizer(runtime),
                                                public_transcript,true,
                                                &public_tokens,&ignored);
    memset(&ignored,0,sizeof(ignored));
    fg_status private_status=fg_tokenizer_encode(runtime_tokenizer(runtime),
                                                 private_transcript,true,
                                                 &private_tokens,&ignored);
    if(public_status==FG_OK&&private_status==FG_OK){
        size_t public_mismatch=runtime_first_token_mismatch(constructed,&public_tokens);
        size_t private_mismatch=runtime_first_token_mismatch(constructed,&private_tokens);
        fprintf(stderr,
                "PREFIX_TOKEN_TRACE raw_history_tokens=%zu suffix_tokens=%zu "
                "constructed_tokens=%zu pending_boundary=valid "
                "public_relation=%s public_first_mismatch=%zu "
                "private_relation=%s private_first_mismatch=%zu\n",
                runtime->history_count,suffix_tokens->count,constructed->count,
                runtime_token_relation(constructed,&public_tokens,public_mismatch),
                public_mismatch,
                runtime_token_relation(constructed,&private_tokens,private_mismatch),
                private_mismatch);
    }else{
        fprintf(stderr,
                "PREFIX_TOKEN_TRACE raw_history_tokens=%zu suffix_tokens=%zu "
                "constructed_tokens=%zu pending_boundary=valid trace=unavailable\n",
                runtime->history_count,suffix_tokens->count,constructed->count);
    }
    fg_tokens_free(&private_tokens);
    fg_tokens_free(&public_tokens);
}

fg_status fg_runtime_generate_continuation(
    fg_runtime *runtime,const char *public_transcript,const char *continuation,
    bool *prefix_miss,uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err){
    if(prefix_miss)*prefix_miss=false;
    if(!runtime||!public_transcript||!continuation||!runtime->rendered_history||
       !runtime->pending_eos_valid||!runtime->next_token_valid||
       runtime->pending_eos_token!=runtime->next_token||
       runtime->pending_eos_token!=fg_tokenizer_eos(runtime_tokenizer(runtime))||
       !runtime->pending_boundary_bytes||
       runtime->pending_boundary_bytes>runtime->rendered_history_length){
        if(prefix_miss)*prefix_miss=true;
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "runtime has no reusable pending EOS continuation frontier");
        return FG_ERR_UNAVAILABLE;
    }
    const char *eos_text=NULL;
    size_t eos_bytes=0;
    fg_status status=fg_tokenizer_token(runtime_tokenizer(runtime),
                                        runtime->pending_eos_token,&eos_text,
                                        &eos_bytes,NULL,err);
    const char *boundary=runtime->rendered_history+
        runtime->rendered_history_length-runtime->pending_boundary_bytes;
    if(status==FG_OK&&
       (runtime->pending_boundary_bytes!=eos_bytes+1u||
        memcmp(boundary,eos_text,eos_bytes)||boundary[eos_bytes]!='\n')){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime pending EOS transcript boundary is inconsistent");
        status=FG_ERR_MISMATCH;
    }
    size_t continuation_length=strlen(continuation);
    if(status==FG_OK&&
       continuation_length>SIZE_MAX-runtime->rendered_history_length-1u){
        fg_error_set(err,FG_ERR_LIMIT,"runtime transcript continuation exceeds address space");
        status=FG_ERR_LIMIT;
    }
    size_t combined_length=0;
    if(status==FG_OK)
        combined_length=runtime->rendered_history_length+continuation_length;
    char *combined=status==FG_OK?malloc(combined_length+1u):NULL;
    if(status==FG_OK&&!combined){
        fg_error_set(err,FG_ERR_OOM,"build runtime transcript continuation");
        status=FG_ERR_OOM;
    }
    if(status==FG_OK){
        memcpy(combined,runtime->rendered_history,runtime->rendered_history_length);
        memcpy(combined+runtime->rendered_history_length,continuation,
               continuation_length+1u);
    }

    size_t suffix_length=0;
    if(status==FG_OK)
        suffix_length=runtime->pending_boundary_bytes+continuation_length;
    char *suffix=status==FG_OK?malloc(suffix_length+1u):NULL;
    if(status==FG_OK&&!suffix){
        fg_error_set(err,FG_ERR_OOM,"build pending EOS continuation suffix");
        status=FG_ERR_OOM;
    }
    if(status==FG_OK){
        memcpy(suffix,boundary,runtime->pending_boundary_bytes);
        memcpy(suffix+runtime->pending_boundary_bytes,continuation,
               continuation_length+1u);
    }

    fg_tokens suffix_tokens={0},prompt={0};
    if(status==FG_OK)
        status=fg_tokenizer_encode(runtime_tokenizer(runtime),suffix,true,
                                   &suffix_tokens,err);
    if(status==FG_OK)
        status=fg_prefix_build_continuation_tokens(
            runtime->history,runtime->history_count,runtime->pending_eos_token,
            suffix_tokens.data,suffix_tokens.count,&prompt.data,&prompt.count,err);
    prompt.capacity=prompt.count;
    if(status==FG_OK){
        runtime_trace_continuation(runtime,public_transcript,combined,
                                   &suffix_tokens,&prompt);
        status=runtime_generate_tokens(
            runtime,combined,&prompt,true,prefix_miss,max_tokens,callback,
            callback_context,interrupted,interrupt_context,stats,err);
    }
    free(suffix);
    free(combined);
    fg_tokens_free(&prompt);
    fg_tokens_free(&suffix_tokens);
    return status;
}

uint32_t fg_runtime_context_tokens(const fg_runtime *runtime){return runtime?(uint32_t)runtime->history_count:0u;}
uint32_t fg_runtime_context_limit(const fg_runtime *runtime){return runtime?runtime->context_limit:0u;}
const char *fg_runtime_model_name(const fg_runtime *runtime){return runtime?"Qwen3.8-Flash-Next":NULL;}
fg_mtp_capability fg_runtime_mtp_capability(const fg_runtime *runtime){
    return runtime?runtime->mtp_capability:FG_MTP_CAPABILITY_UNSUPPORTED;
}
fg_execution_mode fg_runtime_execution_mode(const fg_runtime *runtime){
    (void)runtime;
    return FG_EXECUTION_EXPERT_PARALLEL;
}
const char *fg_execution_mode_name(fg_execution_mode mode){
    return mode==FG_EXECUTION_EXPERT_PARALLEL?"expert-parallel":"unsupported";
}

fg_status fg_serve_main(const char *path,fg_error *err){fg_manifest *m=NULL;fg_status rc=load_checked(path,&m,err);if(rc==FG_OK){fg_manifest_print(m);fg_error_set(err,FG_ERR_UNAVAILABLE,"HTTP serving is not enabled until the owned request path is qualified");rc=FG_ERR_UNAVAILABLE;}free(m);return rc;}
fg_status fg_bench_main(const char *path,fg_error *err){
    (void)path;
    fg_vk_context *vk=NULL;fg_status status=fg_vk_open(&vk,err);
    if(status!=FG_OK)return status;

    /* GPU-timestamped benchmark: measures device-side kernel execution */
    status=fg_vk_bench_dense_q8(vk,err);
    if(status!=FG_OK){fg_vk_close(vk);return status;}

    /* Decomposition benchmark: stream / dequant / dot-no-reduce / full */
    status=fg_vk_bench_decompose(vk,err);
    if(status!=FG_OK){fg_vk_close(vk);return status;}

    /* Layout vs access experiment: A (current) / B (wide scalar) / C (vec4) */
    status=fg_vk_bench_stream_abc(vk,err);
    if(status!=FG_OK){fg_vk_close(vk);return status;}
    status=fg_vk_bench_cooked_layout(vk,err);
    if(status!=FG_OK){fg_vk_close(vk);return status;}

    fprintf(stderr,"\n--- Wall-clock reference (includes CPU overhead) ---\n");
    /* Production decode matmul dimensions: {in_dim, out_dim, label} */
    struct {uint32_t in,out;const char *name;} shapes[]={
        {10240,320, "hc_down (10240→320)"},
        {320,10240, "hc_up   (320→10240)"},
        {2560,640,  "shexp_gate (2560→640)"},
        {640,2560,  "shexp_down (640→2560)"},
        {2560,512,  "qsa_attn_q (2560→512)"},
        {2560,10240,"ple_key (2560→10240)"},
    };
    uint32_t warmup=50,iters=200;
    fprintf(stderr,"=== Q8_0→F32 dense matvec kernel benchmark (GFX1013) ===\n");
    fprintf(stderr,"warmup=%u  iterations=%u  tokens=1\n\n",warmup,iters);
    fprintf(stderr,"%-25s %8s %8s %10s %10s %6s\n","shape","weight","wall","eff.BW","roofline","util");
    fprintf(stderr,"%-25s %8s %8s %10s %10s %6s\n","","(MB)","(us)","(GB/s)","(GB/s)","(%)");
    fprintf(stderr,"----------------------------------------------------------------------\n");
    for(uint32_t s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++){
        uint32_t in_dim=shapes[s].in,out_dim=shapes[s].out;
        uint32_t blocks=in_dim/32u;uint64_t row_bytes=(uint64_t)blocks*34u;
        uint64_t weight_bytes=row_bytes*out_dim;
        uint64_t input_bytes=(uint64_t)in_dim*4u;
        uint64_t output_bytes=(uint64_t)out_dim*4u;
        fg_vk_tensor *w=NULL,*x=NULL,*y=NULL;
        status=fg_vk_tensor_create(vk,weight_bytes,&w,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,input_bytes,&x,err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,output_bytes,&y,err);
        if(status!=FG_OK){fg_vk_tensor_destroy(y);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);break;}
        /* Fill with test pattern */
        memset(fg_vk_tensor_map(w),0x42,weight_bytes);
        float *xp=fg_vk_tensor_map(x);for(uint32_t i=0;i<in_dim;i++)xp[i]=1.0f/(float)(i+1);
        /* Warmup: standalone dispatches (not batched) to include full Vulkan overhead */
        for(uint32_t i=0;i<warmup;i++){
            status=fg_vk_dense_q8_0_f32(vk,y,w,x,in_dim,out_dim,1u,1.0f,err);
            if(status!=FG_OK)break;
        }
        /* Timed iterations — measure EACH dispatch independently (standalone, not batched) */
        struct timespec ts0,ts1;
        clock_gettime(CLOCK_MONOTONIC,&ts0);
        for(uint32_t i=0;status==FG_OK&&i<iters;i++){
            status=fg_vk_dense_q8_0_f32(vk,y,w,x,in_dim,out_dim,1u,1.0f,err);
        }
        clock_gettime(CLOCK_MONOTONIC,&ts1);
        if(status==FG_OK){
            double elapsed_s=(double)(ts1.tv_sec-ts0.tv_sec)+(double)(ts1.tv_nsec-ts0.tv_nsec)*1e-9;
            double per_call_us=elapsed_s*1e6/(double)iters;
            double total_bytes=(double)(weight_bytes+input_bytes+output_bytes);
            double eff_gbps=total_bytes/(per_call_us*1e-6)/1e9;
            double roofline=357.0;
            double util=eff_gbps/roofline*100.0;
            fprintf(stderr,"%-25s %7.2f %7.1f %9.1f %9.1f %5.1f%%\n",
                shapes[s].name,(double)weight_bytes/1e6,per_call_us,eff_gbps,roofline,util);
        }
        double standalone_us=0;
        if(status==FG_OK){
            double elapsed_s=(double)(ts1.tv_sec-ts0.tv_sec)+(double)(ts1.tv_nsec-ts0.tv_nsec)*1e-9;
            standalone_us=elapsed_s*1e6/(double)iters;
        }
        /* Now measure batched dispatch (N dispatches in one command buffer) to isolate kernel vs overhead */
        if(status==FG_OK){
            uint32_t batch_iters=50;
            status=fg_vk_begin(vk,err);
            clock_gettime(CLOCK_MONOTONIC,&ts0);
            for(uint32_t i=0;status==FG_OK&&i<batch_iters;i++){
                status=fg_vk_dense_q8_0_f32(vk,y,w,x,in_dim,out_dim,1u,1.0f,err);
            }
            if(status==FG_OK){fg_status es=fg_vk_end(vk,err);if(es!=FG_OK)status=es;}
            else if(fg_vk_batch_active(vk))fg_vk_end(vk,err);
            clock_gettime(CLOCK_MONOTONIC,&ts1);
            if(status==FG_OK){
                double elapsed_s=(double)(ts1.tv_sec-ts0.tv_sec)+(double)(ts1.tv_nsec-ts0.tv_nsec)*1e-9;
                double per_call_us=elapsed_s*1e6/(double)batch_iters;
                double total_bytes=(double)(weight_bytes+input_bytes+output_bytes);
                double eff_gbps=total_bytes/(per_call_us*1e-6)/1e9;
                double roofline=357.0;
                double util=eff_gbps/roofline*100.0;
                fprintf(stderr,"  └ batched (%u in 1 CB)  %7s %7.1f %9.1f %9.1f %5.1f%%\n",
                    batch_iters,""  ,per_call_us,eff_gbps,roofline,util);
                double dispatch_overhead=standalone_us-per_call_us;
                double theoretical_us=(double)(weight_bytes+input_bytes+output_bytes)/(357.0*1e3);
                fprintf(stderr,"  └ dispatch overhead     %7s %7.1f   (theoretical minimum: %.1f μs)\n",
                    "",dispatch_overhead,theoretical_us);
            }
        }
        fg_vk_tensor_destroy(y);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);
        if(status!=FG_OK)break;
    }
    fg_vk_close(vk);
    return status;
}

fg_status fg_eval_main(const char *path,const char *prompt,uint32_t generate,fg_error *err){
    if(!prompt||generate>4096u){fg_error_set(err,FG_ERR_ARGUMENT,"eval prompt is null or generation exceeds 4096 tokens");return FG_ERR_ARGUMENT;}
    /* Wrap in Qwen chat template if the user passed raw text. */
    char *wrapped=NULL;
    if(strstr(prompt,"<|im_start|>")==NULL){
        size_t n=strlen(prompt);
        wrapped=malloc(n+64u);
        if(!wrapped){fg_error_set(err,FG_ERR_OOM,"allocate chat template");return FG_ERR_OOM;}
        snprintf(wrapped,n+64u,"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",prompt);
        prompt=wrapped;
    }
    fg_manifest *manifest=NULL;fg_runtime_options options;uint32_t qsa_capacity=0u;
    fg_status status=load_checked(path,&manifest,err);

    if(status==FG_OK)status=fg_runtime_options_resolve(&options,manifest,NULL,err);
    char directory[1024];if(status==FG_OK)status=manifest_directory(path,directory,err);
    fg_coordinator coordinator={0};
    if(status==FG_OK)status=coordinator_open(&coordinator,manifest,directory,&options,err);
    fg_tokens prompt_tokens={0};
    if(status==FG_OK)
        status=fg_tokenizer_encode(coordinator.tokenizer,prompt,true,&prompt_tokens,err);
    if(status==FG_OK)
        status=fg_runtime_eval_capacity(&qsa_capacity,&options,prompt_tokens.count,generate,err);
    size_t capacity=status==FG_OK?prompt_tokens.count+(size_t)generate:0u;
    int32_t *history=status==FG_OK?malloc(capacity*sizeof(*history)):NULL;
    if(status==FG_OK&&!history){
        fg_error_set(err,FG_ERR_OOM,"allocate eval token history");status=FG_ERR_OOM;
    }
    for(size_t i=0;status==FG_OK&&i<prompt_tokens.count;i++){history[i]=(int32_t)prompt_tokens.data[i];}uint32_t next=0;float logit=0.0f;struct timespec start,end;fg_vk_tensor *prefill_output=NULL;if(status==FG_OK)clock_gettime(CLOCK_MONOTONIC,&start);for(uint32_t first=0;status==FG_OK&&first<prompt_tokens.count;){uint32_t count=(uint32_t)(prompt_tokens.count-first);if(count>manifest->prefill_microbatch)count=manifest->prefill_microbatch;fg_vk_tensor *ngram_batch=NULL;status=fg_ngram_store_lookup_prefill(coordinator.ngram,history,prompt_tokens.count,first,count,&ngram_batch,err);if(status==FG_OK)status=coordinator_prefill_microbatch(&coordinator,prompt_tokens.data+first,first,(uint16_t)count,ngram_batch,&prefill_output,err);first+=count;}prefill_worker_buffers_release_result_wire(&coordinator.prefill_expert[0]);prefill_worker_buffers_release_result_wire(&coordinator.prefill_expert[1]);prefill_worker_buffers_release_result_wire(&coordinator.prefill_expert[2]);fg_vk_tensor *last_hyper=NULL;if(status==FG_OK){uint32_t final_count=(uint32_t)(prompt_tokens.count%manifest->prefill_microbatch);if(!final_count)final_count=manifest->prefill_microbatch;status=fg_vk_tensor_view(prefill_output,(uint64_t)(final_count-1u)*FG_HYPER_WIDTH*4u,FG_HYPER_WIDTH*4u,&last_hyper,err);}if(status==FG_OK)status=coordinator_output(&coordinator,(uint32_t)prompt_tokens.count-1u,last_hyper,&next,&logit,err);fg_vk_tensor_destroy(last_hyper);if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&end);double seconds=(double)(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)*1e-9;fprintf(stderr,"prefill: %zu tokens in %.3f s (%.2f tok/s), next=%u logit=%g\n",prompt_tokens.count,seconds,(double)prompt_tokens.count/seconds,next,logit);}
    size_t history_count=prompt_tokens.count;struct timespec decode_start,decode_tok;clock_gettime(CLOCK_MONOTONIC,&decode_start);for(uint32_t generated=0;status==FG_OK&&generated<generate;generated++){char decoded[4096];size_t bytes=0;status=fg_tokenizer_decode_token(coordinator.tokenizer,next,decoded,sizeof(decoded),&bytes,err);if(status!=FG_OK)break;clock_gettime(CLOCK_MONOTONIC,&decode_tok);double tok_elapsed=(double)(decode_tok.tv_sec-decode_start.tv_sec)+(double)(decode_tok.tv_nsec-decode_start.tv_nsec)*1e-9;double tok_per_sec=generated>0?(double)generated/tok_elapsed:0.0;fprintf(stderr,"decode[%u]: token=%u logit=%.4f (%.3f s, avg %.2f tok/s)\n",generated,next,logit,tok_elapsed,tok_per_sec);fwrite(decoded,1,bytes,stdout);fflush(stdout);if(next==fg_tokenizer_eos(coordinator.tokenizer)||generated+1u==generate)break;history[history_count++]=(int32_t)next;status=coordinator_decode_token_local(&coordinator,history,history_count,(uint32_t)(history_count-1u),&next,&logit,err);}if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&decode_tok);double total=(double)(decode_tok.tv_sec-decode_start.tv_sec)+(double)(decode_tok.tv_nsec-decode_start.tv_nsec)*1e-9;fprintf(stderr,"decode complete: %.2f tok/s avg\n",total>0?(double)(generate)/total:0.0);fputc('\n',stdout);}
    free(history);
    fg_tokens_free(&prompt_tokens);
    if(manifest)coordinator_close(&coordinator);
    free(manifest);
    free(wrapped);
    return status;
}
