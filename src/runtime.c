#include "fg_runtime.h"
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

static bool token_profile_requested(uint32_t token){const char *requested=getenv("FG_PROFILE_TOKEN");char value[16];if(!requested||!*requested)return false;snprintf(value,sizeof(value),"%u",token);return strcmp(requested,value)==0;}
static bool prefill_profile_requested(void){const char *enabled=getenv("FG_PREFILL_PROFILE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool frame_trace_enabled(void){const char *enabled=getenv("FG_FRAME_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool route_trace_enabled(void){const char *enabled=getenv("FG_TRACE_ROUTES");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool expert_batch_send_enabled(void){const char *enabled=getenv("FG_EXPERT_BATCH_SEND");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool prefix_trace_enabled(void){const char *enabled=getenv("FG_PREFIX_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static uint64_t critical_ns(void){struct timespec value;clock_gettime(CLOCK_REALTIME,&value);return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;}
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
    uint8_t work_wire[FG_GROUP_SIZE][FG_DECODE_WORK_BYTES];fg_fabric_send_item send_items[FG_GROUP_SIZE];ctx->remote_count=0;ctx->send_failed=false;ctx->discard_results=false;ctx->expected_peer_mask=0;ctx->seen_peer_mask=0;bool batch_send=expert_batch_send_enabled()&&!ctx->critical_trace;
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
        if(status==FG_OK&&batch_send){send_items[send_index]=(fg_fabric_send_item){.peer=work.destination_rank,.cls=FG_FABRIC_CONTROL,.type=FG_MSG_DECODE_WORK,.request_id=ctx->request_id,.sequence=ctx->sequence,.payload=work_wire[send_index],.bytes=FG_DECODE_WORK_BYTES};ctx->remote_count++;}
        else if(status==FG_OK){
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
    if(status!=FG_OK&&batch_send)ctx->remote_count=0;
    if(status==FG_OK&&batch_send&&ctx->remote_count){
        status=fg_fabric_send_batch(ctx->fabric,send_items,ctx->remote_count,err);
        if(status!=FG_OK){
            ctx->send_failed=true;
            ctx->remote_count=0;
            transport_poison(ctx->transport_state);
        }else{
            for(uint32_t i=0;i<ctx->remote_count;i++)
                ctx->expected_peer_mask|=1u<<send_items[i].peer;
            transport_pending(ctx->transport_state);
        }
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
    memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid manifest prefill microbatch");return FG_ERR_MISMATCH;}buffers->token_capacity=tokens;buffers->pair_capacity=tokens*FG_TOP_K;buffers->receive_capacity=FG_PREFILL_WORK_HEADER_BYTES+tokens*FG_Q8K_ACTIVATION_BYTES+buffers->pair_capacity*FG_PREFILL_PAIR_BYTES;buffers->result_capacity=FG_PREFILL_RESULT_HEADER_BYTES+buffers->pair_capacity*FG_PREFILL_RESULT_PAIR_BYTES;
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

static fg_status handle_prefill_expert_work(fg_fabric *fabric,fg_expert_executor *expert,const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,prefill_worker_buffers *buffers,fg_error *err){
    fg_prefill_work work={0};fg_status status=fg_prefill_work_decode(&work,buffers->activations,buffers->token_capacity*FG_Q8K_ACTIVATION_BYTES,buffers->pairs,buffers->pair_capacity,payload,bytes,err);uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work.source_rank||work.destination_rank!=self||(work.source_rank!=0u&&manifest->layer_owner[work.layer]!=work.source_rank)||fg_frame_sequence(header)!=work.first_position*FG_LAYER_COUNT+work.layer)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted prefill expert work");status=FG_ERR_MISMATCH;}
    fg_prefill_result result={0};if(status==FG_OK)status=fg_expert_prefill(expert,&work,&result,buffers->result_pairs,buffers->pair_capacity,buffers->outputs,(uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE,err);uint32_t result_bytes=0;
    if(status==FG_OK)status=fg_prefill_result_encode(buffers->result_wire,buffers->result_capacity,&result_bytes,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_PREFILL_RESULT,request,fg_frame_sequence(header),0,buffers->result_wire,result_bytes,err);
    return status;
}

typedef struct prefill_dispatch_context {fg_fabric *fabric;fg_expert_executor *expert;const fg_manifest *manifest;uint32_t self;uint64_t request_id;uint32_t sequence;prefill_worker_buffers *buffers;atomic_uint *transport_state;} prefill_dispatch_context;

static fg_status dispatch_prefill_experts(void *opaque,uint32_t layer,uint32_t first_token,uint16_t token_count,const uint16_t *expert_ids,const float *gates,const uint8_t *activations,fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    prefill_dispatch_context *context=opaque;prefill_worker_buffers *buffers=context->buffers;
    fg_prefill_route routes[FG_GROUP_SIZE];uint32_t route_count=0;
    fg_status status=fg_partition_prefill_routes(context->manifest,layer,token_count,
        expert_ids,gates,routes,&route_count,buffers->pairs,buffers->pair_capacity,err);
    uint32_t work_capacity=(uint32_t)coordinator_prefill_work_wire_bytes(token_count);
    uint8_t *work_wire=status==FG_OK?malloc(work_capacity):NULL;
    if(status==FG_OK&&!work_wire){
        fg_error_set(err,FG_ERR_OOM,"allocate prefill expert work wire");
        status=FG_ERR_OOM;
    }
    bool remote[FG_GROUP_SIZE]={0},transport_failed=false;
    uint32_t expected_peer_mask=0,seen_peer_mask=0;
    uint32_t used_pairs=0;*result_count=0;
    bool needs_result_wire=false;
    for(uint32_t r=0;r<route_count;r++)
        if(routes[r].destination_rank!=context->self)needs_result_wire=true;
    if(status==FG_OK&&needs_result_wire)
        status=prefill_worker_buffers_ensure_result_wire(buffers,err);
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){
        fg_prefill_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,
            .destination_rank=routes[r].destination_rank,.first_position=first_token,
            .token_count=token_count,.pair_count=routes[r].pair_count,
            .activations_q8k=(uint8_t *)activations,.pairs=routes[r].pairs};
        if(work.destination_rank==context->self){
            status=fg_expert_prefill(context->expert,&work,&results[*result_count],
                buffers->result_pairs+used_pairs,buffers->pair_capacity-used_pairs,
                buffers->outputs+(uint64_t)used_pairs*FG_HIDDEN_SIZE,
                (uint64_t)(buffers->pair_capacity-used_pairs)*FG_HIDDEN_SIZE,err);
            if(status==FG_OK){
                used_pairs+=results[*result_count].pair_count;
                (*result_count)++;
            }
        }else{
            uint32_t work_bytes=0;
            status=fg_prefill_work_encode(work_wire,work_capacity,&work_bytes,&work,err);
            if(status==FG_OK){
                status=fg_fabric_send(context->fabric,work.destination_rank,FG_FABRIC_BULK,
                    FG_MSG_PREFILL_WORK,context->request_id,context->sequence,0,
                    work_wire,work_bytes,err);
                if(status==FG_OK){
                    remote[r]=true;
                    expected_peer_mask|=1u<<work.destination_rank;
                    transport_pending(context->transport_state);
                }else{
                    transport_failed=true;
                    transport_poison(context->transport_state);
                }
            }
        }
    }
    for(uint32_t r=0;r<route_count;r++)if(remote[r]){
        fg_frame_header header;uint32_t bytes=0,peer=routes[r].destination_rank;
        fg_error receive_error={0};
        fg_status receive_status=fg_fabric_recv(context->fabric,peer,FG_FABRIC_BULK,
            &header,buffers->result_wire,buffers->result_capacity,&bytes,
            status==FG_OK?err:&receive_error);
        if(receive_status!=FG_OK){
            transport_failed=true;
            if(status==FG_OK)status=receive_status;
            continue;
        }
        uint32_t peer_bit=peer<FG_RANK_COUNT?1u<<peer:0u;
        if(!peer_bit||!(expected_peer_mask&peer_bit)||(seen_peer_mask&peer_bit)||
           fg_frame_type(&header)!=FG_MSG_PREFILL_RESULT||
           fg_frame_request_id(&header)!=context->request_id||
           fg_frame_sequence(&header)!=context->sequence){
            transport_failed=true;
            if(status==FG_OK){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "stale prefill expert result frame from rank %u",peer);
                status=FG_ERR_MISMATCH;
            }
        }else{
            seen_peer_mask|=peer_bit;
        }
        if(status==FG_OK){
            status=fg_prefill_result_decode(&results[*result_count],
                buffers->result_pairs+used_pairs,buffers->pair_capacity-used_pairs,
                buffers->outputs+(uint64_t)used_pairs*FG_HIDDEN_SIZE,
                (uint64_t)(buffers->pair_capacity-used_pairs)*FG_HIDDEN_SIZE,
                buffers->result_wire,bytes,err);
            if(status!=FG_OK)transport_failed=true;
        }
        if(status==FG_OK){
            used_pairs+=results[*result_count].pair_count;
            (*result_count)++;
        }
    }
    if(!transport_failed&&seen_peer_mask==expected_peer_mask)
        transport_complete(context->transport_state);
    else
        transport_poison(context->transport_state);
    free(work_wire);return status;
}

typedef struct prefill_layer_buffers {
    uint32_t *positions;
    fg_vk_tensor *token_tensor;
    uint32_t tokens;
} prefill_layer_buffers;

static void prefill_layer_buffers_destroy(prefill_layer_buffers *buffers){if(!buffers)return;fg_vk_tensor_destroy(buffers->token_tensor);free(buffers->positions);memset(buffers,0,sizeof(*buffers));}

static fg_status prefill_layer_buffers_create(prefill_layer_buffers *buffers,fg_model *model,uint32_t tokens,fg_error *err){memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid prefill layer buffer token count");return FG_ERR_MISMATCH;}buffers->tokens=tokens;buffers->positions=malloc((size_t)tokens*3u*4u);if(!buffers->positions){prefill_layer_buffers_destroy(buffers);fg_error_set(err,FG_ERR_OOM,"allocate bounded prefill layer buffers");return FG_ERR_OOM;}fg_status status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*4u,&buffers->token_tensor,err);if(status!=FG_OK)prefill_layer_buffers_destroy(buffers);return status;}

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
    } write_queue[4];
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
            runtime->write_head=(runtime->write_head+1u)%4u;runtime->write_count--;
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
    if(runtime->write_count==4u){
        pthread_mutex_unlock(&runtime->writer_mutex);
        fg_error_set(err,FG_ERR_LIMIT,
                     "QSA owner writer queue is backpressured; session must reset");
        return FG_ERR_LIMIT;
    }
    uint32_t queue_index=(runtime->write_head+runtime->write_count)%4u;
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
    for(uint32_t i=0;i<4u;i++)free(runtime->write_queue[i].records);
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
    if(runtime->layer_count!=FG_QSA_OWNER_LAYER_COUNT){
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
    for(uint32_t i=0;i<4u;i++){
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
    if(status==FG_OK)status=fg_qsa_state_read_blocks(runtime->state,(uint32_t)slot,
        runtime->blocks,batch.page_count,runtime->read_records,runtime->committed,err);
    for(uint32_t i=0;status==FG_OK&&i<batch.page_count;i++){
        if(runtime->committed[i]!=FG_Q38_QSA_COMPRESS_RATIO){
            fg_error_set(err,FG_ERR_MISMATCH,"QSA owner returned an incomplete page");
            status=FG_ERR_MISMATCH;
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
                               const char *directory,qsa_owner_runtime *qsa,uint32_t self,
                               uint32_t peer,const fg_frame_header *header,const uint8_t *payload,
                               uint32_t bytes,uint64_t *session_id,fg_error *err){
    uint64_t request=fg_frame_request_id(header);
    if(peer!=0u||!request||(*session_id&&request<=*session_id)){fg_error_set(err,FG_ERR_MISMATCH,"invalid, stale, or duplicate session begin");return FG_ERR_MISMATCH;}
    fg_session_identity identity;fg_status status=fg_session_identity_from_manifest(manifest,&identity,err);
    if(manifest->protocol_version==FG_PROTOCOL_MIN_VERSION){
        if(status==FG_OK&&bytes){fg_error_set(err,FG_ERR_FORMAT,"legacy session begin payload must be empty");status=FG_ERR_FORMAT;}
        if(status==FG_OK)status=qsa_owner_open_session(qsa,manifest,directory,&identity,
                                                       request,NULL,err);
        if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,
                                               FG_MSG_SESSION_READY,request,0,0,NULL,0,err);
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

static fg_status handle_ngram_work(fg_fabric *fabric,const fg_ngram_resident *resident,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_error *err){fg_ngram_work work;fg_status status=fg_ngram_work_decode(&work,payload,bytes,err);uint64_t request=fg_frame_request_id(header);if(status==FG_OK&&(!session_id||request!=session_id||peer!=0u||work.source_rank!=0u||work.destination_rank!=self||fg_frame_sequence(header)!=work.token_index)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted resident n-gram work");status=FG_ERR_MISMATCH;}fg_ngram_result result={.source_rank=(uint8_t)self,.destination_rank=0u,.item_count=work.item_count,.token_index=work.token_index};if(status==FG_OK){memcpy(result.heads,work.heads,work.item_count);status=fg_ngram_resident_read(resident,work.rows,work.item_count,result.packed,sizeof(result.packed),err);}uint8_t wire[FG_NGRAM_RESULT_MAX_BYTES];uint32_t result_bytes=0;if(status==FG_OK)status=fg_ngram_result_encode(wire,sizeof(wire),&result_bytes,&result,err);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_NGRAM_RESULT,request,work.token_index,0,wire,result_bytes,err);return status;}

static fg_status handle_output_work(fg_fabric *fabric,fg_output_executor *output,fg_vk_context *vk,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_error *err){
    if(self!=4u||!output){fg_error_set(err,FG_ERR_MISMATCH,"output work reached a non-output rank");return FG_ERR_MISMATCH;}
    fg_output_work *work=calloc(1,sizeof(*work));if(!work){fg_error_set(err,FG_ERR_OOM,"allocate output work");return FG_ERR_OOM;}
    fg_status status=fg_output_work_decode(work,payload,bytes,err);uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work->source_rank||work->destination_rank!=self)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output work");status=FG_ERR_MISMATCH;}
    token_profile_capture capture={0};if(status==FG_OK)status=token_profile_begin(&capture,vk,work->token_index,err);
    if(status==FG_OK)status=fg_vk_tensor_write(hyper_tensor,0,work->hyper,sizeof(work->hyper),err);
    fg_output_result result={.source_rank=(uint8_t)self,.destination_rank=work->source_rank,.token_index=work->token_index};
    if(status==FG_OK)status=fg_output_greedy(output,hyper_tensor,&result.token,&result.logit,err);
    uint8_t wire[FG_OUTPUT_RESULT_BYTES];if(status==FG_OK)status=fg_output_result_encode(wire,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,request,fg_frame_sequence(header),0,wire,sizeof(wire),err);
    status=token_profile_end(&capture,self,"output",work->token_index,UINT32_MAX,status,err);
    free(work);return status;
}

static fg_status rank_worker_loop(fg_fabric *fabric,fg_expert_executor *expert,fg_output_executor *output,const fg_ngram_resident *ngram,fg_model *model,const fg_manifest *manifest,const char *directory,uint32_t self,fg_error *err){
    uint32_t control_capacity=FG_LAYER_WORK_FOUR_AXIS_BASE_BYTES;if(control_capacity<FG_OUTPUT_WORK_BYTES)control_capacity=FG_OUTPUT_WORK_BYTES;if(control_capacity<FG_DECODE_WORK_BYTES)control_capacity=FG_DECODE_WORK_BYTES;if(control_capacity<FG_QSA_BLOCK_WORK_MAX_BYTES)control_capacity=FG_QSA_BLOCK_WORK_MAX_BYTES;uint8_t *control=malloc(control_capacity);prefill_worker_buffers prefill={0};qsa_owner_runtime qsa={0};fg_vk_tensor *hyper=NULL;
    /* Pre-allocate expert work buffers — eliminates ~200 KB malloc/free per expert request */
    fg_expert_result *ew_result=malloc(sizeof(*ew_result));uint8_t *ew_wire=malloc(FG_EXPERT_RESULT_SINGLE_BYTES);
    if(!control||!ew_result||!ew_wire){free(ew_wire);free(ew_result);free(control);fg_error_set(err,FG_ERR_OOM,"allocate rank worker buffers");return FG_ERR_OOM;}
    fg_status status=prefill_worker_buffers_create(&prefill,manifest->prefill_microbatch,
                                                   false,err);if(status==FG_OK)status=qsa_owner_runtime_create(&qsa,manifest,self,err);if(status==FG_OK&&output)status=fg_vk_tensor_create(fg_model_vk(model),FG_HYPER_WIDTH*4u,&hyper,err);if(status==FG_OK)status=token_profile_prepare(fg_model_vk(model),err);uint8_t *bulk_receive=prefill.receive;uint32_t bulk_capacity=prefill.receive_capacity;if(qsa.enabled&&qsa.receive_capacity>bulk_capacity){bulk_receive=qsa.receive_wire;bulk_capacity=qsa.receive_capacity;}uint64_t session_id=0;
    while(status==FG_OK){uint32_t peer=0,bytes=0;fg_frame_header header;fg_fabric_class ready_class;fg_fabric_recv_timing receive_timing={0};receive_timing.poll_start_ns=critical_ns();status=fg_fabric_wait_ready(fabric,3u,&peer,&ready_class,err);receive_timing.ready_ns=critical_ns();if(status!=FG_OK)break;if(ready_class==FG_FABRIC_BULK){status=fg_fabric_recv_timed(fabric,peer,FG_FABRIC_BULK,&header,bulk_receive,bulk_capacity,&bytes,&receive_timing,err);fg_message_type type=status==FG_OK?fg_frame_type(&header):0;if(status==FG_OK&&type==FG_MSG_PREFILL_WORK)status=handle_prefill_expert_work(fabric,expert,manifest,self,session_id,peer,&header,bulk_receive,bytes,&prefill,err);else if(status==FG_OK&&type==FG_MSG_QSA_PAGE_APPEND)status=handle_qsa_page_append(&qsa,manifest,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK&&type==FG_MSG_QSA_PAGE_BARRIER)status=handle_qsa_page_barrier(fabric,&qsa,self,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK&&type==FG_MSG_QSA_PAGE_FETCH)status=handle_qsa_page_fetch(fabric,&qsa,manifest,self,peer,&header,bulk_receive,bytes,err);else if(status==FG_OK){fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported bulk message %u",self,type);status=FG_ERR_FORMAT;}continue;}status=fg_fabric_recv_timed(fabric,peer,FG_FABRIC_CONTROL,&header,control,control_capacity,&bytes,&receive_timing,err);if(status!=FG_OK)break;fg_message_type type=fg_frame_type(&header);if(type==FG_MSG_DECODE_WORK){if(!session_id||fg_frame_request_id(&header)!=session_id){fg_error_set(err,FG_ERR_MISMATCH,"stale expert work request");status=FG_ERR_MISMATCH;}else status=handle_expert_work(fabric,expert,fg_model_vk(model),self,peer,&header,control,bytes,&receive_timing,ew_result,ew_wire,err);}else if(type==FG_MSG_NGRAM_WORK)status=handle_ngram_work(fabric,ngram,self,session_id,peer,&header,control,bytes,err);else if(type==FG_MSG_SESSION_BEGIN)status=begin_session(fabric,manifest,directory,&qsa,self,peer,&header,control,bytes,&session_id,err);else if(type==FG_MSG_OUTPUT_WORK)status=handle_output_work(fabric,output,fg_model_vk(model),self,session_id,peer,&header,control,bytes,hyper,err);else{fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported control message %u",self,type);status=FG_ERR_FORMAT;}}
    fg_vk_tensor_destroy(hyper);qsa_owner_runtime_destroy(&qsa);prefill_worker_buffers_destroy(&prefill);free(ew_wire);free(ew_result);free(control);return status;
}

fg_status fg_rank_main(const char *path,uint32_t rank,fg_error *err){if(rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"rank must be 0..7");return FG_ERR_ARGUMENT;}fg_manifest *manifest=NULL;fg_status status=load_checked(path,&manifest,err);char directory[1024];if(status==FG_OK)status=manifest_directory(path,directory,err);fg_model *model=NULL;fg_expert_executor *expert=NULL;fg_output_executor *output=NULL;fg_ngram_resident *ngram=NULL;fg_fabric *fabric=NULL;if(status==FG_OK)status=fg_model_open(&model,manifest,directory,rank,err);if(status==FG_OK)status=fg_expert_executor_create(&expert,model,err);if(status==FG_OK&&rank==4u)status=fg_output_executor_create(&output,model,err);uint64_t row_begin=0,row_count=0;char ngram_path[1200];if(status==FG_OK)status=fg_q38_ngram_rank_range(rank,&row_begin,&row_count,err);if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),"%s/ngram-rank-%02u.iq4nl",directory,rank)>=(int)sizeof(ngram_path)){fg_error_set(err,FG_ERR_LIMIT,"resident n-gram shard path is too long");status=FG_ERR_LIMIT;}if(status==FG_OK)status=fg_ngram_resident_open(&ngram,ngram_path,row_begin,row_count,err);if(status==FG_OK)status=fg_fabric_open(&fabric,manifest,rank,err);if(status==FG_OK)status=rank_ready(fabric,rank,err);if(status==FG_OK){printf("rank %u READY: %.3f GiB sealed weights, %.3f GiB n-gram rows resident on %s\n",rank,(double)fg_model_weight_bytes(model)/(1024.0*1024.0*1024.0),(double)(row_count*FG_NGRAM_ROW_BYTES)/(1024.0*1024.0*1024.0),fg_vk_device_name(fg_model_vk(model)));fflush(stdout);status=rank_worker_loop(fabric,expert,output,ngram,model,manifest,directory,rank,err);}fg_fabric_close(fabric);fg_ngram_resident_close(ngram);fg_output_executor_destroy(output);fg_expert_executor_destroy(expert);fg_model_close(model);free(manifest);return status;}

typedef struct qsa_page_transport {
    fg_qsa_page *append_pages[2],*fetch_pages,*result_pages;
    uint8_t *fetch_wire,*result_wire;
    fg_qsa_replica *replica;
    fg_fabric *fabric;
    atomic_uint *transport_state;
    uint32_t append_sequence[2],fetch_sequence[2],append_count[2];
} qsa_page_transport;

static uint32_t qsa_owner_index(uint32_t rank){return rank==3u?0u:rank==7u?1u:UINT32_MAX;}

static void qsa_page_transport_destroy(qsa_page_transport *transport){
    if(!transport)return;
    fg_qsa_replica_destroy(transport->replica);
    free(transport->result_wire);free(transport->fetch_wire);free(transport->result_pages);
    free(transport->fetch_pages);
    for(uint32_t i=0;i<2u;i++)free(transport->append_pages[i]);
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
    for(uint32_t i=0;i<2u;i++){
        transport->append_pages[i]=calloc(FG_QSA_PAGE_APPEND_MAX_PAGES,
                                           sizeof(*transport->append_pages[i]));
    }
    transport->fetch_pages=calloc(FG_QSA_PAGE_FETCH_MAX_PAGES,
                                  sizeof(*transport->fetch_pages));
    transport->result_pages=calloc(FG_QSA_PAGE_FETCH_MAX_PAGES,
                                   sizeof(*transport->result_pages));
    transport->fetch_wire=malloc(FG_QSA_PAGE_FETCH_MAX_BYTES);
    transport->result_wire=malloc(FG_QSA_PAGE_RESULT_MAX_BYTES);
    if(!transport->append_pages[0]||!transport->append_pages[1]||
       !transport->fetch_pages||!transport->result_pages||
       !transport->fetch_wire||!transport->result_wire){
        qsa_page_transport_destroy(transport);
        fg_error_set(err,FG_ERR_OOM,"allocate fixed QSA page transport buffers");
        return FG_ERR_OOM;
    }
    fg_status status=fg_qsa_replica_create(&transport->replica,qsa_replica_send,transport,err);
    if(status!=FG_OK)qsa_page_transport_destroy(transport);
    return status;
}

typedef struct fg_coordinator {const fg_manifest *manifest;fg_runtime_options options;fg_session_identity identity;fg_model *model;fg_expert_executor *expert;fg_owner_executor *owner;fg_fabric *fabric;fg_ngram_store *ngram;fg_tokenizer *tokenizer;prefill_worker_buffers prefill_expert;prefill_layer_buffers prefill_layer;qsa_page_transport qsa_pages;uint64_t session_id;uint8_t *async_recv_payloads[FG_GROUP_SIZE];const char *directory;atomic_uint transport_state;} fg_coordinator;

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
    return 2u*(uint64_t)FG_QSA_PAGE_APPEND_MAX_PAGES*sizeof(fg_qsa_page)+
           2u*(uint64_t)FG_QSA_PAGE_FETCH_MAX_PAGES*sizeof(fg_qsa_page)+
           FG_QSA_PAGE_FETCH_MAX_BYTES+FG_QSA_PAGE_RESULT_MAX_BYTES+
           fg_qsa_replica_host_bytes(transport->replica);
}

static uint64_t coordinator_transport_capacity_bytes(void){
    return 2u*(uint64_t)FG_QSA_PAGE_APPEND_MAX_PAGES*sizeof(fg_qsa_page)+
           2u*(uint64_t)FG_QSA_PAGE_FETCH_MAX_PAGES*sizeof(fg_qsa_page)+
           FG_QSA_PAGE_FETCH_MAX_BYTES+FG_QSA_PAGE_RESULT_MAX_BYTES+
           fg_qsa_replica_host_bytes_for_capacity();
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

#define FG_COORDINATOR_HC_INJECT_PIECES 24u
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

static uint64_t coordinator_owner_transient_reclaim(uint32_t tokens){
    return (uint64_t)tokens*10240u*4u+(uint64_t)tokens*320u*4u+
           (uint64_t)FG_COORDINATOR_HC_DOWN_SPLITS*320u*4u+
               (uint64_t)tokens*320u*4u+(uint64_t)tokens*10240u*4u+
           (uint64_t)FG_COORDINATOR_HC_INJECT_PIECES*4u*
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
    uint64_t prefill_host=coordinator_prefill_host_bytes(&coordinator->prefill_expert);
    uint64_t async_host=(uint64_t)FG_GROUP_SIZE*FG_EXPERT_RESULT_SINGLE_BYTES;
    uint64_t positions_host=(uint64_t)coordinator->prefill_layer.tokens*3u*
        sizeof(uint32_t);
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
    uint64_t index=(uint64_t)logical*12u*FG_Q38_QSA_INDEX_KEY_BYTES;
    uint64_t record_cache=(uint64_t)cache_page_count*FG_QSA_PAGE_RECORD_BYTES;
    uint64_t ngram_vk_capacity=(uint64_t)coordinator->manifest->prefill_microbatch*
        FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES+
        (uint64_t)coordinator->manifest->prefill_microbatch*
            FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH*4u;
    uint64_t ngram_vk=fg_ngram_store_vk_bytes(coordinator->ngram);
    uint64_t qsa_aux=fg_qsa_selection_scratch_bytes(
        logical,coordinator->manifest->prefill_microbatch);
    uint64_t owner_family=fg_qsa_attention_family_scratch_bytes(
        coordinator->manifest->prefill_microbatch);
    uint64_t owner_state=(uint64_t)(FG_LAYER_COUNT-FG_LAYER_COUNT/4u)*
        (10240u*4u*4u+48u*128u*128u*4u)+10240u*9u*4u;
    uint64_t owner_pingpong=(uint64_t)coordinator->manifest->prefill_microbatch*
        10240u*4u*2u;
    uint64_t owner_activation=(uint64_t)coordinator->manifest->prefill_microbatch*
        FG_Q8K_ACTIVATION_BYTES;
    uint64_t expert_activation=owner_activation;
    uint64_t qsa_positions=logical*FG_Q38_QSA_POSITION_BYTES;
    uint64_t qsa_index=(uint64_t)logical*(FG_LAYER_COUNT/4u)*
        FG_Q38_QSA_INDEX_KEY_BYTES;
    uint64_t qsa_record_cache=record_cache;
    uint64_t prefill_token=(uint64_t)coordinator->prefill_layer.tokens*
        sizeof(uint32_t);
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
    uint64_t deferred_prefill_wire=coordinator->prefill_expert.result_wire?
        0u:coordinator->prefill_expert.result_capacity;
    uint64_t deferred_prefill_work=coordinator_prefill_work_wire_bytes(
        coordinator->manifest->prefill_microbatch);
    uint64_t deferred_transport=coordinator->qsa_pages.replica?
        0u:coordinator_transport_capacity_bytes();
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
        coordinator->manifest->prefill_microbatch);
    uint64_t qsa_reclaim=coordinator_qsa_aux_reclaim(logical);
    uint64_t layer_reclaim=(uint64_t)coordinator->manifest->prefill_microbatch*
        (FG_HYPER_WIDTH*4u+FG_NGRAM_EMBED_VALUES*4u);
    uint64_t vulkan_reclaim=owner_transient+qsa_reclaim+layer_reclaim+
        FG_HYPER_WIDTH*sizeof(float)+deferred_ngram_vk;
    uint64_t prefill_host_reclaim=FG_PREFILL_WORK_HEADER_BYTES+
        (uint64_t)coordinator->prefill_expert.token_capacity*
            (2u*FG_Q8K_ACTIVATION_BYTES+
             FG_TOP_K*sizeof(fg_prefill_pair)+
             FG_TOP_K*sizeof(fg_prefill_result_pair)+
             FG_TOP_K*FG_HIDDEN_SIZE*sizeof(float));
    uint64_t expert_recv_reclaim=FG_EXPERT_RESULT_MAX_BYTES;
    uint64_t async_host_reclaim=(uint64_t)FG_GROUP_SIZE*
        (FG_EXPERT_RESULT_MAX_BYTES-FG_EXPERT_RESULT_SINGLE_BYTES);
    uint64_t positions_host_reclaim=(uint64_t)coordinator->prefill_layer.tokens*
        sizeof(uint32_t);
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
            uint32_t owner_index=qsa_owner_index(owner);
            if(owner_index==UINT32_MAX){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA page owner is not rank 3 or 7");
                return FG_ERR_MISMATCH;
            }
            uint32_t index=transport->append_count[owner_index]++;
            if(index>=FG_QSA_PAGE_APPEND_MAX_PAGES){
                fg_error_set(err,FG_ERR_LIMIT,"QSA complete-page append batch overflow");
                return FG_ERR_LIMIT;
            }
            const uint8_t *records=NULL;
            status=fg_owner_qsa_page_records(coordinator->owner,layer,block,&records,err);
            if(status!=FG_OK)return status;
            transport->append_pages[owner_index][index]=(fg_qsa_page){
                .layer=(uint8_t)layer,.block=block,.records=records};
        }
    }
    uint32_t send_count=0;
    for(uint32_t i=0;i<2u;i++)send_count+=transport->append_count[i]!=0u;
    if(!send_count)return FG_OK;
    uint8_t *buffers[2]={0};
    status=fg_qsa_replica_reserve(transport->replica,send_count,buffers,err);
    if(status!=FG_OK)return status;
    fg_qsa_replica_item items[2]={0};uint32_t send_index=0;
    for(uint32_t owner_index=0;owner_index<2u;owner_index++){
        uint32_t count=transport->append_count[owner_index];
        if(!count)continue;
        uint32_t owner=owner_index?7u:3u;
        fg_qsa_page_batch batch={.source_rank=0u,.destination_rank=(uint8_t)owner,
            .batch_id=transport->append_sequence[owner_index],.page_count=(uint16_t)count,
            .pages=transport->append_pages[owner_index]};
        uint32_t bytes=0;status=fg_qsa_page_append_encode(buffers[send_index],
            FG_QSA_PAGE_APPEND_MAX_BYTES,&bytes,&batch,err);
        if(status!=FG_OK){fg_qsa_replica_cancel(transport->replica);return status;}
        for(uint32_t i=0;i<count;i++)
            fg_owner_qsa_page_published(coordinator->owner,batch.pages[i].layer,
                                        batch.pages[i].block);
        items[send_index++]=(fg_qsa_replica_item){.owner=owner,.batch_id=batch.batch_id,
            .bytes=bytes,.session_id=coordinator->session_id};
    }
    status=fg_qsa_replica_commit(transport->replica,items,send_count,err);
    if(status!=FG_OK){fg_qsa_replica_cancel(transport->replica);return status;}
    for(uint32_t owner_index=0;owner_index<2u;owner_index++)
        if(transport->append_count[owner_index])transport->append_sequence[owner_index]++;
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
    fg_status status=qsa_page_transport_ensure(transport,err);
    if(status!=FG_OK)return status;
    uint32_t owner=coordinator->manifest->layer_owner[layer],owner_index=qsa_owner_index(owner);
    if(owner_index==UINT32_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA cold page has no authoritative owner");
        return FG_ERR_MISMATCH;
    }
    status=fg_qsa_replica_drain(transport->replica,err);
    if(status!=FG_OK)return status;
    for(uint32_t i=0;i<block_count;i++)transport->fetch_pages[i]=(fg_qsa_page){
        .layer=(uint8_t)layer,.block=blocks[i],.records=NULL};
    fg_qsa_page_batch request={.source_rank=0u,.destination_rank=(uint8_t)owner,
        .batch_id=transport->fetch_sequence[owner_index],.page_count=(uint16_t)block_count,
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
        transport->fetch_sequence[owner_index]++;
        transport_complete(&coordinator->transport_state);
    }else if(!transport_ready(&coordinator->transport_state)){
        transport_poison(&coordinator->transport_state);
    }
    return status;
}

static fg_status coordinator_qsa_barrier(fg_coordinator *coordinator,fg_error *err){
    if(!coordinator->session_id)return FG_OK;
    fg_status drain_status=fg_qsa_replica_drain_if_present(
        coordinator->qsa_pages.replica,err);
    if(drain_status!=FG_OK)return drain_status;
    uint8_t wire[2][FG_QSA_PAGE_BARRIER_BYTES];
    for(uint32_t owner_index=0;owner_index<2u;owner_index++){
        uint32_t owner=owner_index?7u:3u;
        fg_qsa_page_barrier barrier={.source_rank=0u,.destination_rank=(uint8_t)owner,
            .batch_id=coordinator->qsa_pages.append_sequence[owner_index]};
        fg_status status=fg_qsa_page_barrier_encode(wire[owner_index],&barrier,err);
        if(status==FG_OK){
            status=fg_fabric_send(coordinator->fabric,owner,FG_FABRIC_BULK,
                FG_MSG_QSA_PAGE_BARRIER,coordinator->session_id,barrier.batch_id,0,
                wire[owner_index],sizeof(wire[owner_index]),err);
            if(status==FG_OK)transport_pending(&coordinator->transport_state);
            else transport_poison(&coordinator->transport_state);
        }
        if(status!=FG_OK){
            if(!transport_ready(&coordinator->transport_state))
                transport_poison(&coordinator->transport_state);
            return status;
        }
    }
    for(uint32_t owner_index=0;owner_index<2u;owner_index++){
        uint32_t owner=owner_index?7u:3u,bytes=0;fg_frame_header header;
        fg_status status=fg_fabric_recv(coordinator->fabric,owner,FG_FABRIC_BULK,&header,
            wire[owner_index],sizeof(wire[owner_index]),&bytes,err);
        fg_qsa_page_barrier ack={0};
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_QSA_PAGE_BARRIER_ACK||
           fg_frame_request_id(&header)!=coordinator->session_id||
           fg_frame_sequence(&header)!=coordinator->qsa_pages.append_sequence[owner_index])){
            fg_error_set(err,FG_ERR_MISMATCH,"stale QSA page barrier acknowledgement");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_qsa_page_barrier_decode(&ack,wire[owner_index],bytes,err);
        if(status==FG_OK&&(ack.source_rank!=owner||ack.destination_rank!=0u||
           ack.batch_id!=coordinator->qsa_pages.append_sequence[owner_index])){
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
    transport_complete(&coordinator->transport_state);
    return FG_OK;
}

/* All-local prefill: process all 48 layers on the coordinator */
static fg_status coordinator_prefill_microbatch(fg_coordinator *coordinator,const uint32_t *token_ids,uint32_t first_token,uint16_t token_count,const fg_vk_tensor *ngram_embeddings,fg_vk_tensor **output,fg_error *err){
    if(!coordinator||!token_ids||!token_count||token_count>coordinator->manifest->prefill_microbatch||!ngram_embeddings||!output||token_count>coordinator->manifest->max_context||first_token>coordinator->manifest->max_context-token_count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid coordinator prefill microbatch");return FG_ERR_ARGUMENT;}
    prefill_layer_buffers *buffers=&coordinator->prefill_layer;
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
            .buffers=&coordinator->prefill_expert,
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

static fg_status ngram_rank_for_row(uint64_t row,uint32_t *owner,fg_error *err){if(!owner){fg_error_set(err,FG_ERR_ARGUMENT,"n-gram row owner output is null");return FG_ERR_ARGUMENT;}for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){uint64_t begin=0,count=0;fg_status status=fg_q38_ngram_rank_range(rank,&begin,&count,err);if(status!=FG_OK)return status;if(row>=begin&&row-begin<count){*owner=rank;return FG_OK;}}fg_error_set(err,FG_ERR_MISMATCH,"n-gram row %llu has no resident owner",(unsigned long long)row);return FG_ERR_MISMATCH;}

static fg_status coordinator_ngram_resident(fg_coordinator *coordinator,const int32_t *history,size_t history_count,uint32_t token_index,fg_vk_tensor **embedding,fg_error *err){
    uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];
    fg_status status=fg_q38_ngram_lookup(history,history_count,rows,addresses,err);
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
            status=fg_fabric_send(coordinator->fabric,rank,FG_FABRIC_CONTROL,
                FG_MSG_NGRAM_WORK,coordinator->session_id,token_index,0,work_wire,bytes,err);
            if(status==FG_OK){
                sent[rank]=true;sent_count++;
                transport_pending(&coordinator->transport_state);
            }else{
                transport_failed=true;
                transport_poison(&coordinator->transport_state);
            }
        }
    }
    double fired=dispatch_ts();
    uint32_t consumed=0;
    while(consumed<sent_count){
        uint32_t peer=0,bytes=0;fg_frame_header header;fg_error receive_error={0};
        fg_status receive_status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_BULK,
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
            fg_frame_request_id(&header)==coordinator->session_id&&
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
    if(consumed==sent_count&&all_received&&!transport_failed)
        transport_complete(&coordinator->transport_state);
    else
        transport_poison(&coordinator->transport_state);
    for(uint32_t head=0;status==FG_OK&&head<FG_NGRAM_HEAD_COUNT;head++)
        if(!seen_head[head]){
            fg_error_set(err,FG_ERR_MISMATCH,"missing resident n-gram head %u",head);
            status=FG_ERR_MISMATCH;
        }
    double collected=dispatch_ts();const char *verify=getenv("FG_NGRAM_VERIFY");
    if(status==FG_OK&&token_profile_requested(token_index)&&verify&&*verify&&
       strcmp(verify,"0")!=0){
        uint32_t mismatch=UINT32_MAX;
        status=fg_ngram_store_verify_packed(coordinator->ngram,addresses,
            FG_NGRAM_HEAD_COUNT,packed,&mismatch,err);
        if(status==FG_OK)
            fprintf(stderr,"NGRAM_RESIDENT_VERIFY token=%u rows=%u status=exact\n",
                    token_index,FG_NGRAM_HEAD_COUNT);
    }
    if(status==FG_OK)status=fg_ngram_store_decode_packed(coordinator->ngram,packed,
                                                         FG_NGRAM_HEAD_COUNT,embedding,err);
    double decoded=dispatch_ts();
    if(frame_trace_enabled())
        fprintf(stderr,"NGRAM_RESIDENT_TRACE token=%u workers=%u fire_ms=%.3f collect_ms=%.3f dequant_ms=%.3f total_ms=%.3f\n",
                token_index,sent_count,fired-start,collected-fired,decoded-collected,
                decoded-start);
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
        status=fg_owner_decode_layer_async(coordinator->owner,layer,token_index,position,current,
            layer_ngram,fire_experts,collect_experts,&async_ctx,NULL,NULL,
            &layer_out,err);
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

static void coordinator_close(fg_coordinator *coordinator){if(!coordinator)return;for(uint32_t i=0;i<FG_GROUP_SIZE;i++)free(coordinator->async_recv_payloads[i]);qsa_page_transport_destroy(&coordinator->qsa_pages);prefill_layer_buffers_destroy(&coordinator->prefill_layer);prefill_worker_buffers_destroy(&coordinator->prefill_expert);fg_ngram_store_close(coordinator->ngram);fg_tokenizer_close(coordinator->tokenizer);fg_fabric_close(coordinator->fabric);fg_owner_executor_destroy(coordinator->owner);fg_expert_executor_destroy(coordinator->expert);fg_model_close(coordinator->model);memset(coordinator,0,sizeof(*coordinator));}

static fg_status coordinator_open(fg_coordinator *coordinator,const fg_manifest *manifest,const char *directory,const fg_runtime_options *options,fg_error *err){memset(coordinator,0,sizeof(*coordinator));coordinator->manifest=manifest;coordinator->options=*options;fg_status status=fg_session_identity_from_manifest(manifest,&coordinator->identity,err);if(status==FG_OK&&manifest->protocol_version<6u){fg_error_set(err,FG_ERR_MISMATCH,"QSA page ownership requires protocol version 6");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_model_open_coordinator(&coordinator->model,manifest,directory,0u,err);if(status==FG_OK)status=fg_owner_executor_create(&coordinator->owner,coordinator->model,err);if(status==FG_OK)status=fg_expert_executor_create(&coordinator->expert,coordinator->model,err);uint32_t cache_page_count=coordinator_qsa_cache_pages(options);if(status==FG_OK&&!cache_page_count){fg_error_set(err,FG_ERR_LIMIT,"QSA record cache has no capacity");status=FG_ERR_LIMIT;}if(status==FG_OK)status=fg_owner_qsa_open_mirror(coordinator->owner,options->logical_context_tokens,options->qsa_hot_tokens,cache_page_count,manifest->prefill_microbatch,coordinator_fetch_qsa_pages,coordinator,err);if(status==FG_OK)status=fg_tokenizer_open(&coordinator->tokenizer,directory,manifest,err);if(status==FG_OK)status=fg_tokenizer_validate_qwen38(coordinator->tokenizer,err);const fg_tensor_record *ngram_record=NULL;for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)if(manifest->tensors[i].kind==FG_TENSOR_NGRAM){if(ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"multiple n-gram tensors in deployment manifest");status=FG_ERR_MISMATCH;}else ngram_record=&manifest->tensors[i];}char ngram_path[1200];if(status==FG_OK&&!ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"deployment manifest has no n-gram tensor");status=FG_ERR_MISMATCH;}if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),"%s/ngram.iq4nl",directory)>=(int)sizeof(ngram_path)){fg_error_set(err,FG_ERR_LIMIT,"n-gram path is too long");status=FG_ERR_LIMIT;}if(status==FG_OK)status=fg_ngram_store_open(&coordinator->ngram,fg_model_vk(coordinator->model),ngram_path,ngram_record->bytes,manifest->prefill_microbatch,err);
    /* Allocate only coordinator-side asynchronous receive payloads. */
    for(uint32_t i=0;status==FG_OK&&i<FG_GROUP_SIZE;i++){coordinator->async_recv_payloads[i]=malloc(FG_EXPERT_RESULT_SINGLE_BYTES);if(!coordinator->async_recv_payloads[i]){fg_error_set(err,FG_ERR_OOM,"allocate async expert recv buffer %u",i);status=FG_ERR_OOM;}}if(status==FG_OK)status=prefill_worker_buffers_create(&coordinator->prefill_expert,manifest->prefill_microbatch,true,err);if(status==FG_OK)status=prefill_layer_buffers_create(&coordinator->prefill_layer,coordinator->model,manifest->prefill_microbatch,err);if(status==FG_OK)status=fg_fabric_open(&coordinator->fabric,manifest,0u,err);if(status==FG_OK)atomic_init(&coordinator->transport_state,FG_TRANSPORT_READY);if(status==FG_OK)status=qsa_page_transport_create(&coordinator->qsa_pages,coordinator->fabric,&coordinator->transport_state,err);if(status==FG_OK)status=rank_ready(coordinator->fabric,0u,err);if(status==FG_OK)status=token_profile_prepare(fg_model_vk(coordinator->model),err);if(status==FG_OK)status=coordinator_begin_session(coordinator,err);if(status==FG_OK)coordinator_memory_report(coordinator);if(status!=FG_OK)coordinator_close(coordinator);coordinator->directory=directory;return status;}

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
    if(!transport_ready(&runtime->coordinator.transport_state)){
        runtime->state_ready=false;
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "distributed transport is not reusable; reopen the runtime");
        return FG_ERR_UNAVAILABLE;
    }
    runtime->state_ready=false;
    runtime->history_count=0;
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
    fg_status status=fg_owner_reset_state(runtime->coordinator.owner,err);
    if(status==FG_OK&&runtime->session_started)
        status=coordinator_begin_session(&runtime->coordinator,err);
    if(status==FG_OK)runtime->session_started=true;
    if(status!=FG_OK)return status;
    runtime->state_ready=true;
    return FG_OK;
}

fg_status fg_runtime_open_with_options(fg_runtime **out,const char *path,
                                       const fg_runtime_options *requested,fg_error *err){
    if(!out||!path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime open arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    fg_runtime *runtime=calloc(1,sizeof(*runtime));if(!runtime){fg_error_set(err,FG_ERR_OOM,"allocate resident runtime");return FG_ERR_OOM;}
    fg_status status=load_checked(path,&runtime->manifest,err);
    if(status==FG_OK)status=fg_runtime_options_resolve(&runtime->options,runtime->manifest,
                                                       requested,err);
    if(status==FG_OK)runtime->context_limit=runtime->options.logical_context_tokens;
    if(status==FG_OK)status=manifest_directory(path,runtime->directory,err);
    if(status==FG_OK)status=coordinator_open(&runtime->coordinator,runtime->manifest,
                                             runtime->directory,&runtime->options,err);
    if(status==FG_OK)status=runtime_reset_state(runtime,FG_PREFIX_RESET_COLD_START,err);
    if(status!=FG_OK){fg_runtime_close(runtime);return status;}*out=runtime;return FG_OK;
}

fg_status fg_runtime_open(fg_runtime **out,const char *path,fg_error *err){
    return fg_runtime_open_with_options(out,path,NULL,err);
}

void fg_runtime_close(fg_runtime *runtime){
    if(!runtime)return;
    coordinator_close(&runtime->coordinator);
    free(runtime->rendered_history);free(runtime->history);free(runtime->manifest);free(runtime);
}

fg_status fg_runtime_reset(fg_runtime *runtime,fg_error *err){
    if(!runtime||!runtime->coordinator.owner){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
    return runtime_reset_state(runtime,FG_PREFIX_RESET_EXPLICIT,err);
}

fg_status fg_runtime_reset_public_history(fg_runtime *runtime,fg_error *err){
    if(!runtime||!runtime->coordinator.owner){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
    return runtime_reset_state(runtime,FG_PREFIX_RESET_PUBLIC_MISMATCH,err);
}

fg_status fg_runtime_reset_failure(fg_runtime *runtime,fg_error *err){
    if(!runtime||!runtime->coordinator.owner){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
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

static fg_status runtime_generate_tokens(
    fg_runtime *runtime,const char *transcript,const fg_tokens *prompt,
    bool require_prefix_hit,bool *prefix_miss,
    uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err){
    if(prefix_miss)*prefix_miss=false;
    if(!runtime||!transcript||!prompt||(!prompt->data&&prompt->count)||!callback||
       !max_tokens||max_tokens>4096u){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident generation arguments");return FG_ERR_ARGUMENT;}
    if(!runtime->state_ready){fg_error_set(err,FG_ERR_MISMATCH,"resident runtime requires a successful reset");return FG_ERR_MISMATCH;}
    if(stats)memset(stats,0,sizeof(*stats));
    fg_status status=FG_OK;
    fg_prefix_plan plan={0};
    if(status==FG_OK)status=fg_prefix_plan_tokens(
        runtime->history,runtime->history_count,runtime->next_token_valid,
        prompt->data,prompt->count,runtime->empty_reason,&plan,err);
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
    for(size_t offset=prefill_offset;status==FG_OK&&offset<prompt->count;){
        uint32_t count=(uint32_t)(prompt->count-offset);if(count>runtime->manifest->prefill_microbatch)count=runtime->manifest->prefill_microbatch;
        uint32_t first=(uint32_t)offset;fg_vk_tensor *ngram_batch=NULL;
        fg_vk_context *vk=fg_model_vk(runtime->coordinator.model);bool capture=!runtime->prefill_profiled&&prefill_profile_requested(),capture_active=false;struct timespec capture_start,capture_end;
        if(capture){status=fg_vk_profile_begin(vk,err);if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"ngram_prefill",err);if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&capture_start);capture_active=true;}}
        if(status==FG_OK)status=fg_ngram_store_lookup_prefill(runtime->coordinator.ngram,runtime->history,runtime->history_count,first,count,&ngram_batch,err);
        if(status==FG_OK)status=coordinator_prefill_microbatch(&runtime->coordinator,
            prompt->data+offset,first,(uint16_t)count,ngram_batch,&prefill_output,err);
        if(capture_active){fg_vk_profile profile={0};fg_error profile_error={0};clock_gettime(CLOCK_MONOTONIC,&capture_end);fg_status profile_status=fg_vk_profile_end(vk,&profile,status==FG_OK?err:&profile_error);runtime->prefill_profiled=true;if(status==FG_OK&&profile_status!=FG_OK)status=profile_status;fprintf(stderr,"PREFILL_PROFILE first=%u tokens=%u wall_ms=%.3f gpu_ms=%.3f kernel_ms=%.3f submissions=%llu dispatches=%llu\n",first,count,elapsed_seconds(&capture_start,&capture_end)*1000.0,profile.gpu_ms,profile.kernel_ms,(unsigned long long)profile.submissions,(unsigned long long)profile.dispatches);for(uint32_t i=0;i<profile.kernel_count;i++){const fg_vk_profile_kernel *kernel=&profile.kernels[i];fprintf(stderr,"PREFILL_PROFILE_KERNEL scope=%s kernel=%s calls=%llu gpu_ms=%.3f\n",kernel->scope,kernel->name,(unsigned long long)kernel->invocations,kernel->gpu_ms);}}
        offset+=count;
    }
    prefill_worker_buffers_release_result_wire(&runtime->coordinator.prefill_expert);
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
    while(status==FG_OK&&generated<max_tokens){
        if(interrupted&&interrupted(interrupt_context))break;
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
        if(status==FG_OK)status=callback(callback_context,next,decoded,bytes,err);
        if(status==FG_OK)status=runtime_render_append(&candidate,&candidate_length,
                                                      &candidate_capacity,decoded,bytes,err);
        if(status!=FG_OK)break;
        runtime->history[runtime->history_count++]=(int32_t)next;generated++;
        state_mutated=true;
        status=coordinator_decode_token_local(&runtime->coordinator,runtime->history,runtime->history_count,(uint32_t)runtime->history_count-1u,&next,&logit,err);
        if(status==FG_OK){
            runtime->next_token=next;
            runtime->next_logit=logit;
            runtime->next_token_valid=true;
        }
    }
    if(status==FG_OK){
        clock_gettime(CLOCK_MONOTONIC,&decode_end);
        runtime->empty_reason=FG_PREFIX_RESET_NONE;
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
    free(candidate);return status;
}

fg_status fg_runtime_generate(fg_runtime *runtime,const char *transcript,uint32_t max_tokens,
                              fg_token_callback callback,void *callback_context,
                              fg_interrupt_fn interrupted,void *interrupt_context,
                              fg_generation_stats *stats,fg_error *err){
    fg_tokens prompt={0};
    fg_status status=runtime&&transcript?
        fg_tokenizer_encode(runtime->coordinator.tokenizer,transcript,true,&prompt,err):
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
    fg_status public_status=fg_tokenizer_encode(runtime->coordinator.tokenizer,
                                                public_transcript,true,
                                                &public_tokens,&ignored);
    memset(&ignored,0,sizeof(ignored));
    fg_status private_status=fg_tokenizer_encode(runtime->coordinator.tokenizer,
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
       runtime->pending_eos_token!=fg_tokenizer_eos(runtime->coordinator.tokenizer)||
       !runtime->pending_boundary_bytes||
       runtime->pending_boundary_bytes>runtime->rendered_history_length){
        if(prefix_miss)*prefix_miss=true;
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "runtime has no reusable pending EOS continuation frontier");
        return FG_ERR_UNAVAILABLE;
    }
    const char *eos_text=NULL;
    size_t eos_bytes=0;
    fg_status status=fg_tokenizer_token(runtime->coordinator.tokenizer,
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
        status=fg_tokenizer_encode(runtime->coordinator.tokenizer,suffix,true,
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
    for(size_t i=0;status==FG_OK&&i<prompt_tokens.count;i++){history[i]=(int32_t)prompt_tokens.data[i];}uint32_t next=0;float logit=0.0f;struct timespec start,end;fg_vk_tensor *prefill_output=NULL;if(status==FG_OK)clock_gettime(CLOCK_MONOTONIC,&start);for(uint32_t first=0;status==FG_OK&&first<prompt_tokens.count;){uint32_t count=(uint32_t)(prompt_tokens.count-first);if(count>manifest->prefill_microbatch)count=manifest->prefill_microbatch;fg_vk_tensor *ngram_batch=NULL;status=fg_ngram_store_lookup_prefill(coordinator.ngram,history,prompt_tokens.count,first,count,&ngram_batch,err);if(status==FG_OK)status=coordinator_prefill_microbatch(&coordinator,prompt_tokens.data+first,first,(uint16_t)count,ngram_batch,&prefill_output,err);first+=count;}prefill_worker_buffers_release_result_wire(&coordinator.prefill_expert);fg_vk_tensor *last_hyper=NULL;if(status==FG_OK){uint32_t final_count=(uint32_t)(prompt_tokens.count%manifest->prefill_microbatch);if(!final_count)final_count=manifest->prefill_microbatch;status=fg_vk_tensor_view(prefill_output,(uint64_t)(final_count-1u)*FG_HYPER_WIDTH*4u,FG_HYPER_WIDTH*4u,&last_hyper,err);}if(status==FG_OK)status=coordinator_output(&coordinator,(uint32_t)prompt_tokens.count-1u,last_hyper,&next,&logit,err);fg_vk_tensor_destroy(last_hyper);if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&end);double seconds=(double)(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)*1e-9;fprintf(stderr,"prefill: %zu tokens in %.3f s (%.2f tok/s), next=%u logit=%g\n",prompt_tokens.count,seconds,(double)prompt_tokens.count/seconds,next,logit);}
    size_t history_count=prompt_tokens.count;struct timespec decode_start,decode_tok;clock_gettime(CLOCK_MONOTONIC,&decode_start);for(uint32_t generated=0;status==FG_OK&&generated<generate;generated++){char decoded[4096];size_t bytes=0;status=fg_tokenizer_decode_token(coordinator.tokenizer,next,decoded,sizeof(decoded),&bytes,err);if(status!=FG_OK)break;clock_gettime(CLOCK_MONOTONIC,&decode_tok);double tok_elapsed=(double)(decode_tok.tv_sec-decode_start.tv_sec)+(double)(decode_tok.tv_nsec-decode_start.tv_nsec)*1e-9;double tok_per_sec=generated>0?(double)generated/tok_elapsed:0.0;fprintf(stderr,"decode[%u]: token=%u logit=%.4f (%.3f s, avg %.2f tok/s)\n",generated,next,logit,tok_elapsed,tok_per_sec);fwrite(decoded,1,bytes,stdout);fflush(stdout);if(next==fg_tokenizer_eos(coordinator.tokenizer)||generated+1u==generate)break;history[history_count++]=(int32_t)next;status=coordinator_decode_token_local(&coordinator,history,history_count,(uint32_t)(history_count-1u),&next,&logit,err);}if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&decode_tok);double total=(double)(decode_tok.tv_sec-decode_start.tv_sec)+(double)(decode_tok.tv_nsec-decode_start.tv_nsec)*1e-9;fprintf(stderr,"decode complete: %.2f tok/s avg\n",total>0?(double)(generate)/total:0.0);fputc('\n',stdout);}
    free(history);
    fg_tokens_free(&prompt_tokens);
    if(manifest)coordinator_close(&coordinator);
    free(manifest);
    free(wrapped);
    return status;
}
