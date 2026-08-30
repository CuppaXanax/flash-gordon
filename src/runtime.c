#include "fg_runtime.h"
#include "fg_expert.h"
#include "fg_fabric.h"
#include "fg_model.h"
#include "fg_ngram.h"
#include "fg_owner.h"
#include "fg_output.h"
#include "fg_q38_schema.h"
#include "fg_tokenizer.h"
#include "fg_uring.h"

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>

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
static uint64_t critical_ns(void){struct timespec value;clock_gettime(CLOCK_REALTIME,&value);return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;}
_Static_assert(FG_NGRAM_ROW_BYTES==FG_NGRAM_WIRE_ROW_BYTES,"n-gram row wire size mismatch");

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
    /* Saved routes for deferred local expert compute */
    fg_expert_route routes[FG_GROUP_SIZE];uint32_t route_count;
    uint8_t activation_copy[FG_Q8K_ACTIVATION_BYTES];
    /* Per-peer recv state */
    uint32_t remote_count;
    uint32_t recv_peers[FG_GROUP_SIZE];
    fg_frame_header recv_headers[FG_GROUP_SIZE];
    uint8_t *recv_payloads[FG_GROUP_SIZE]; /* each malloc'd FG_EXPERT_RESULT_MAX_BYTES */
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
    uint8_t work_wire[FG_GROUP_SIZE][FG_DECODE_WORK_BYTES];fg_fabric_send_item send_items[FG_GROUP_SIZE];ctx->remote_count=0;bool batch_send=expert_batch_send_enabled()&&!ctx->critical_trace;
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
        else if(status==FG_OK){status=fg_fabric_send(ctx->fabric,work.destination_rank,FG_FABRIC_CONTROL,FG_MSG_DECODE_WORK,ctx->request_id,ctx->sequence,0,work_wire[send_index],FG_DECODE_WORK_BYTES,err);if(status==FG_OK)ctx->remote_count++;}
        if(ctx->critical_trace&&trace_index<FG_GROUP_SIZE){ctx->send_trace[layer][trace_index].end_ns=critical_ns();ctx->send_trace_count[layer]++;}
    }
    if(status==FG_OK&&batch_send&&ctx->remote_count)status=fg_fabric_send_batch(ctx->fabric,send_items,ctx->remote_count,err);
    if(status==FG_OK&&(token_profile_requested(token)||route_trace_enabled())){uint32_t local=0,local_selected=0,selected=0,rank_mask=0;for(uint32_t r=0;r<route_count;r++){bool is_local=routes[r].destination_rank==ctx->self;local+=is_local;local_selected+=is_local?routes[r].selected_count:0u;selected+=routes[r].selected_count;rank_mask|=1u<<routes[r].destination_rank;}fprintf(stderr,"EP_ROUTE_TRACE token=%u layer=%u routes=%u remotes=%u local=%u local_selected=%u selected=%u rank_mask=%u",token,layer,route_count,ctx->remote_count,local,local_selected,selected,rank_mask);if(route_trace_enabled()){fprintf(stderr," expert_ids=");for(uint32_t slot=0;slot<FG_TOP_K;slot++)fprintf(stderr,"%s%u",slot?",":"",expert_ids[slot]);fprintf(stderr," expert_ranks=");for(uint32_t slot=0;slot<FG_TOP_K;slot++)fprintf(stderr,"%s%u",slot?",":"",ctx->manifest->expert_rank[layer][expert_ids[slot]]);}fputc('\n',stderr);}
    return status;
}

/* Compute local experts, then collect remote results in arrival order. */
static fg_status collect_experts(void *opaque,uint32_t layer,uint32_t token,fg_expert_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    async_expert_context *ctx=opaque;*result_count=0;fg_status status=FG_OK;
    /* Compute local experts while remote results are in flight */
    for(uint32_t r=0;status==FG_OK&&r<ctx->route_count;r++){
        if(ctx->routes[r].destination_rank!=ctx->self)continue;
        fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)ctx->self,.destination_rank=(uint8_t)ctx->self,.selected_count=ctx->routes[r].selected_count,.position=token};
        for(uint32_t i=0;i<ctx->routes[r].selected_count;i++){work.expert_ids[i]=ctx->routes[r].global_expert_ids[i];work.routing_slots[i]=ctx->routes[r].routing_slots[i];work.gates[i]=ctx->routes[r].gates[i];}
        memcpy(work.activation_q8k,ctx->activation_copy,FG_Q8K_ACTIVATION_BYTES);
        status=fg_expert_decode(ctx->expert,&work,&results[*result_count],err);
        if(status==FG_OK)(*result_count)++;
    }
    for(uint32_t received=0;status==FG_OK&&received<ctx->remote_count;received++){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        fg_fabric_recv_timing timing={0};status=ctx->critical_trace?fg_fabric_recv_any_timed(ctx->fabric,FG_FABRIC_BULK,&peer,&header,ctx->recv_payloads[0],FG_EXPERT_RESULT_MAX_BYTES,&bytes,&timing,err):fg_fabric_recv_any(ctx->fabric,FG_FABRIC_BULK,&peer,&header,ctx->recv_payloads[0],FG_EXPERT_RESULT_MAX_BYTES,&bytes,err);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_EXPERT_RESULT||fg_frame_request_id(&header)!=ctx->request_id||fg_frame_sequence(&header)!=ctx->sequence)){fg_error_set(err,FG_ERR_MISMATCH,"stale split expert result from peer %u",peer);status=FG_ERR_MISMATCH;}
        if(status==FG_OK)status=fg_expert_result_decode(&results[*result_count],ctx->recv_payloads[0],bytes,err);
        if(ctx->critical_trace&&received<FG_GROUP_SIZE){ctx->recv_trace[layer][received].peer=peer;ctx->recv_trace[layer][received].ready_mask=timing.ready_mask;ctx->recv_trace[layer][received].bytes=bytes;ctx->recv_trace[layer][received].poll_start_ns=timing.poll_start_ns;ctx->recv_trace[layer][received].ready_ns=timing.ready_ns;ctx->recv_trace[layer][received].header_end_ns=timing.header_end_ns;ctx->recv_trace[layer][received].payload_end_ns=timing.payload_end_ns;ctx->recv_trace[layer][received].validate_end_ns=timing.validate_end_ns;ctx->recv_trace[layer][received].decode_end_ns=critical_ns();ctx->recv_trace_count[layer]++;}
        if(status==FG_OK)(*result_count)++;
    }
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
    if(status==FG_OK)status=fg_expert_result_encode(wire,FG_EXPERT_RESULT_MAX_BYTES,&result_bytes,result,err);
    double tw_encode=dispatch_ts();uint64_t trace_encode=critical_ns(),trace_send_start=trace_encode;
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_EXPERT_RESULT,fg_frame_request_id(header),fg_frame_sequence(header),0,wire,result_bytes,err);
    double tw_send=dispatch_ts();uint64_t trace_send_end=critical_ns();
    status=token_profile_end(&capture,self,"routed_expert",work.position,work.layer,status,err);
    if(status==FG_OK&&((work.position>=26u&&work.position<28u)||token_profile_requested(work.position))){fprintf(stderr,"WORKER_EXPERT layer[%u] t=%u rank=%u sel=%u decode=%.2f gpu=%.2f reduce=%.2f encode=%.2f send=%.2f total=%.2f ready_ns=%llu header_end_ns=%llu recv_end_ns=%llu validate_end_ns=%llu decode_end_ns=%llu gpu_begin_ns=%llu gpu_end_ns=%llu reduce_end_ns=%llu encode_end_ns=%llu send_start_ns=%llu send_end_ns=%llu\n",work.layer,work.position,self,work.selected_count,tw_decode-tw0,tw_gpu-tw_decode,tw_reduce-tw_gpu,tw_encode-tw_reduce,tw_send-tw_encode,tw_send-tw0,(unsigned long long)(receive_timing?receive_timing->ready_ns:0),(unsigned long long)(receive_timing?receive_timing->header_end_ns:0),(unsigned long long)(receive_timing?receive_timing->payload_end_ns:0),(unsigned long long)(receive_timing?receive_timing->validate_end_ns:0),(unsigned long long)trace_decode,(unsigned long long)trace_gpu_begin,(unsigned long long)trace_gpu_end,(unsigned long long)trace_reduce,(unsigned long long)trace_encode,(unsigned long long)trace_send_start,(unsigned long long)trace_send_end);}
    return status;
}

typedef struct prefill_worker_buffers {
    uint8_t *receive,*activations,*result_wire;
    fg_prefill_pair *pairs;
    fg_prefill_result_pair *result_pairs;
    float *outputs;
    uint32_t token_capacity,pair_capacity,receive_capacity,result_capacity;
} prefill_worker_buffers;

static void prefill_worker_buffers_destroy(prefill_worker_buffers *buffers){if(!buffers)return;free(buffers->outputs);free(buffers->result_pairs);free(buffers->pairs);free(buffers->result_wire);free(buffers->activations);free(buffers->receive);memset(buffers,0,sizeof(*buffers));}

static fg_status prefill_worker_buffers_create(prefill_worker_buffers *buffers,uint32_t tokens,fg_error *err){
    memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid manifest prefill microbatch");return FG_ERR_MISMATCH;}buffers->token_capacity=tokens;buffers->pair_capacity=tokens*FG_TOP_K;buffers->receive_capacity=FG_PREFILL_WORK_HEADER_BYTES+tokens*FG_Q8K_ACTIVATION_BYTES+buffers->pair_capacity*FG_PREFILL_PAIR_BYTES;buffers->result_capacity=FG_PREFILL_RESULT_HEADER_BYTES+buffers->pair_capacity*FG_PREFILL_RESULT_PAIR_BYTES;
    buffers->receive=malloc(buffers->receive_capacity);buffers->activations=malloc((size_t)tokens*FG_Q8K_ACTIVATION_BYTES);buffers->result_wire=malloc(buffers->result_capacity);buffers->pairs=malloc((size_t)buffers->pair_capacity*sizeof(*buffers->pairs));buffers->result_pairs=malloc((size_t)buffers->pair_capacity*sizeof(*buffers->result_pairs));buffers->outputs=malloc((size_t)buffers->pair_capacity*FG_HIDDEN_SIZE*sizeof(*buffers->outputs));
    if(!buffers->receive||!buffers->activations||!buffers->result_wire||!buffers->pairs||!buffers->result_pairs||!buffers->outputs){prefill_worker_buffers_destroy(buffers);fg_error_set(err,FG_ERR_OOM,"allocate bounded prefill worker buffers");return FG_ERR_OOM;}return FG_OK;
}

static fg_status handle_prefill_expert_work(fg_fabric *fabric,fg_expert_executor *expert,const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,prefill_worker_buffers *buffers,fg_error *err){
    fg_prefill_work work={0};fg_status status=fg_prefill_work_decode(&work,buffers->activations,buffers->token_capacity*FG_Q8K_ACTIVATION_BYTES,buffers->pairs,buffers->pair_capacity,payload,bytes,err);uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work.source_rank||work.destination_rank!=self||(work.source_rank!=0u&&manifest->layer_owner[work.layer]!=work.source_rank)||fg_frame_sequence(header)!=work.first_position*FG_LAYER_COUNT+work.layer)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted prefill expert work");status=FG_ERR_MISMATCH;}
    fg_prefill_result result={0};if(status==FG_OK)status=fg_expert_prefill(expert,&work,&result,buffers->result_pairs,buffers->pair_capacity,buffers->outputs,(uint64_t)buffers->pair_capacity*FG_HIDDEN_SIZE,err);uint32_t result_bytes=0;
    if(status==FG_OK)status=fg_prefill_result_encode(buffers->result_wire,buffers->result_capacity,&result_bytes,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_PREFILL_RESULT,request,fg_frame_sequence(header),0,buffers->result_wire,result_bytes,err);
    return status;
}

typedef struct prefill_dispatch_context {fg_fabric *fabric;fg_expert_executor *expert;const fg_manifest *manifest;uint32_t self;uint64_t request_id;uint32_t sequence;prefill_worker_buffers *buffers;} prefill_dispatch_context;

static fg_status dispatch_prefill_experts(void *opaque,uint32_t layer,uint32_t first_token,uint16_t token_count,const uint16_t *expert_ids,const float *gates,const uint8_t *activations,fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    prefill_dispatch_context *context=opaque;prefill_worker_buffers *buffers=context->buffers;fg_prefill_route routes[FG_GROUP_SIZE];uint32_t route_count=0;fg_status status=fg_partition_prefill_routes(context->manifest,layer,token_count,expert_ids,gates,routes,&route_count,buffers->pairs,buffers->pair_capacity,err);uint32_t work_capacity=FG_PREFILL_WORK_HEADER_BYTES+(uint32_t)token_count*FG_Q8K_ACTIVATION_BYTES+(uint32_t)token_count*FG_TOP_K*FG_PREFILL_PAIR_BYTES;uint8_t *work_wire=status==FG_OK?malloc(work_capacity):NULL;if(status==FG_OK&&!work_wire){fg_error_set(err,FG_ERR_OOM,"allocate prefill expert work wire");status=FG_ERR_OOM;}bool remote[FG_GROUP_SIZE]={0};uint32_t used_pairs=0;*result_count=0;
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){fg_prefill_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,.destination_rank=routes[r].destination_rank,.first_position=first_token,.token_count=token_count,.pair_count=routes[r].pair_count,.activations_q8k=(uint8_t *)activations,.pairs=routes[r].pairs};if(work.destination_rank==context->self){status=fg_expert_prefill(context->expert,&work,&results[*result_count],buffers->result_pairs+used_pairs,buffers->pair_capacity-used_pairs,buffers->outputs+(uint64_t)used_pairs*FG_HIDDEN_SIZE,(uint64_t)(buffers->pair_capacity-used_pairs)*FG_HIDDEN_SIZE,err);if(status==FG_OK){used_pairs+=results[*result_count].pair_count;(*result_count)++;}}else{uint32_t work_bytes=0;status=fg_prefill_work_encode(work_wire,work_capacity,&work_bytes,&work,err);if(status==FG_OK)status=fg_fabric_send(context->fabric,work.destination_rank,FG_FABRIC_BULK,FG_MSG_PREFILL_WORK,context->request_id,context->sequence,0,work_wire,work_bytes,err);remote[r]=status==FG_OK;}}
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++)if(remote[r]){fg_frame_header header;uint32_t bytes=0,peer=routes[r].destination_rank;status=fg_fabric_recv(context->fabric,peer,FG_FABRIC_BULK,&header,buffers->result_wire,buffers->result_capacity,&bytes,err);if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_RESULT||fg_frame_request_id(&header)!=context->request_id||fg_frame_sequence(&header)!=context->sequence)){fg_error_set(err,FG_ERR_MISMATCH,"stale prefill expert result frame from rank %u",peer);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_result_decode(&results[*result_count],buffers->result_pairs+used_pairs,buffers->pair_capacity-used_pairs,buffers->outputs+(uint64_t)used_pairs*FG_HIDDEN_SIZE,(uint64_t)(buffers->pair_capacity-used_pairs)*FG_HIDDEN_SIZE,buffers->result_wire,bytes,err);if(status==FG_OK){used_pairs+=results[*result_count].pair_count;(*result_count)++;}}
    free(work_wire);return status;
}

typedef struct prefill_layer_buffers {
    uint8_t *receive,*result_wire;
    uint32_t *positions;
    float *hyper,*ngram,*output;
    fg_vk_tensor *hyper_tensor,*ngram_tensor,*token_tensor;
    uint32_t tokens,receive_capacity,result_capacity;
} prefill_layer_buffers;

static void prefill_layer_buffers_destroy(prefill_layer_buffers *buffers){if(!buffers)return;fg_vk_tensor_destroy(buffers->token_tensor);fg_vk_tensor_destroy(buffers->ngram_tensor);fg_vk_tensor_destroy(buffers->hyper_tensor);free(buffers->output);free(buffers->ngram);free(buffers->hyper);free(buffers->positions);free(buffers->result_wire);free(buffers->receive);memset(buffers,0,sizeof(*buffers));}

static fg_status prefill_layer_buffers_create(prefill_layer_buffers *buffers,fg_model *model,uint32_t tokens,fg_error *err){memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid prefill layer buffer token count");return FG_ERR_MISMATCH;}buffers->tokens=tokens;buffers->receive_capacity=FG_PREFILL_LAYER_HEADER_BYTES+tokens*4u*4u+tokens*FG_HYPER_WIDTH*4u+tokens*FG_NGRAM_EMBED_VALUES*4u;buffers->result_capacity=FG_PREFILL_LAYER_HEADER_BYTES+tokens*FG_HYPER_WIDTH*4u;buffers->receive=malloc(buffers->receive_capacity);buffers->result_wire=malloc(buffers->result_capacity);buffers->positions=malloc((size_t)tokens*4u*4u);buffers->hyper=malloc((size_t)tokens*FG_HYPER_WIDTH*4u);buffers->ngram=malloc((size_t)tokens*FG_NGRAM_EMBED_VALUES*4u);buffers->output=malloc((size_t)tokens*FG_HYPER_WIDTH*4u);if(!buffers->receive||!buffers->result_wire||!buffers->positions||!buffers->hyper||!buffers->ngram||!buffers->output){prefill_layer_buffers_destroy(buffers);fg_error_set(err,FG_ERR_OOM,"allocate bounded prefill layer buffers");return FG_ERR_OOM;}fg_status status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*FG_HYPER_WIDTH*4u,&buffers->hyper_tensor,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*FG_NGRAM_EMBED_VALUES*4u,&buffers->ngram_tensor,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*4u,&buffers->token_tensor,err);if(status!=FG_OK)prefill_layer_buffers_destroy(buffers);return status;}

static fg_status begin_session(fg_fabric *fabric,const fg_manifest *manifest,uint32_t self,
                               uint32_t peer,const fg_frame_header *header,const uint8_t *payload,
                               uint32_t bytes,uint64_t *session_id,fg_error *err){
    uint64_t request=fg_frame_request_id(header);
    if(peer!=0u||!request||*session_id){fg_error_set(err,FG_ERR_MISMATCH,"invalid or duplicate session begin");return FG_ERR_MISMATCH;}
    if(manifest->protocol_version==FG_PROTOCOL_MIN_VERSION){
        if(bytes){fg_error_set(err,FG_ERR_FORMAT,"legacy session begin payload must be empty");return FG_ERR_FORMAT;}
        *session_id=request;
        return fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_SESSION_READY,
                              request,0,0,NULL,0,err);
    }
    fg_owner_session_control control;fg_status status=fg_owner_session_control_decode(
        &control,payload,bytes,err);
    fg_session_identity identity;
    if(status==FG_OK)status=fg_session_identity_from_manifest(manifest,&identity,err);
    if(status==FG_OK&&(control.operation!=FG_OWNER_SESSION_BEGIN||control.rank!=self||
       control.session_nonce!=request||
       control.position_mode!=(fg_position_mode)manifest->session.position_mode||
       memcmp(control.identity_sha256,identity.identity_sha256,32u)||
       memcmp(control.state_format_sha256,
              manifest->session.rank_state_format_sha256[self],32u))){
        fg_error_set(err,FG_ERR_MISMATCH,"owner session begin identity or rank mismatch");
        status=FG_ERR_MISMATCH;
    }
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

static fg_status rank_worker_loop(fg_fabric *fabric,fg_expert_executor *expert,fg_output_executor *output,const fg_ngram_resident *ngram,fg_model *model,const fg_manifest *manifest,uint32_t self,fg_error *err){
    uint32_t control_capacity=FG_OUTPUT_WORK_BYTES>FG_DECODE_WORK_BYTES?FG_OUTPUT_WORK_BYTES:FG_DECODE_WORK_BYTES;uint8_t *control=malloc(control_capacity);prefill_worker_buffers prefill={0};fg_vk_tensor *hyper=NULL;
    /* Pre-allocate expert work buffers — eliminates ~200 KB malloc/free per expert request */
    fg_expert_result *ew_result=malloc(sizeof(*ew_result));uint8_t *ew_wire=malloc(FG_EXPERT_RESULT_MAX_BYTES);
    if(!control||!ew_result||!ew_wire){free(ew_wire);free(ew_result);free(control);fg_error_set(err,FG_ERR_OOM,"allocate rank worker buffers");return FG_ERR_OOM;}
    fg_status status=prefill_worker_buffers_create(&prefill,manifest->prefill_microbatch,err);if(status==FG_OK&&output)status=fg_vk_tensor_create(fg_model_vk(model),FG_HYPER_WIDTH*4u,&hyper,err);if(status==FG_OK)status=token_profile_prepare(fg_model_vk(model),err);uint64_t session_id=0;
    while(status==FG_OK){uint32_t peer=0,bytes=0;fg_frame_header header;fg_fabric_class ready_class;fg_fabric_recv_timing receive_timing={0};receive_timing.poll_start_ns=critical_ns();status=fg_fabric_wait_ready(fabric,3u,&peer,&ready_class,err);receive_timing.ready_ns=critical_ns();if(status!=FG_OK)break;if(ready_class==FG_FABRIC_BULK){status=fg_fabric_recv_timed(fabric,peer,FG_FABRIC_BULK,&header,prefill.receive,prefill.receive_capacity,&bytes,&receive_timing,err);fg_message_type type=status==FG_OK?fg_frame_type(&header):0;if(status==FG_OK&&type==FG_MSG_PREFILL_WORK)status=handle_prefill_expert_work(fabric,expert,manifest,self,session_id,peer,&header,prefill.receive,bytes,&prefill,err);else if(status==FG_OK){fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported bulk message %u",self,type);status=FG_ERR_FORMAT;}continue;}status=fg_fabric_recv_timed(fabric,peer,FG_FABRIC_CONTROL,&header,control,control_capacity,&bytes,&receive_timing,err);if(status!=FG_OK)break;fg_message_type type=fg_frame_type(&header);if(type==FG_MSG_DECODE_WORK){if(!session_id||fg_frame_request_id(&header)!=session_id){fg_error_set(err,FG_ERR_MISMATCH,"stale expert work request");status=FG_ERR_MISMATCH;}else status=handle_expert_work(fabric,expert,fg_model_vk(model),self,peer,&header,control,bytes,&receive_timing,ew_result,ew_wire,err);}else if(type==FG_MSG_NGRAM_WORK)status=handle_ngram_work(fabric,ngram,self,session_id,peer,&header,control,bytes,err);else if(type==FG_MSG_SESSION_BEGIN)status=begin_session(fabric,manifest,self,peer,&header,control,bytes,&session_id,err);else if(type==FG_MSG_OUTPUT_WORK)status=handle_output_work(fabric,output,fg_model_vk(model),self,session_id,peer,&header,control,bytes,hyper,err);else{fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported control message %u",self,type);status=FG_ERR_FORMAT;}}
    fg_vk_tensor_destroy(hyper);prefill_worker_buffers_destroy(&prefill);free(ew_wire);free(ew_result);free(control);return status;
}

fg_status fg_rank_main(const char *path,uint32_t rank,fg_error *err){if(rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"rank must be 0..7");return FG_ERR_ARGUMENT;}fg_manifest *manifest=NULL;fg_status status=load_checked(path,&manifest,err);char directory[1024];if(status==FG_OK)status=manifest_directory(path,directory,err);fg_model *model=NULL;fg_expert_executor *expert=NULL;fg_output_executor *output=NULL;fg_ngram_resident *ngram=NULL;fg_fabric *fabric=NULL;if(status==FG_OK)status=fg_model_open(&model,manifest,directory,rank,err);if(status==FG_OK)status=fg_expert_executor_create(&expert,model,err);if(status==FG_OK&&rank==4u)status=fg_output_executor_create(&output,model,err);uint64_t row_begin=0,row_count=0;char ngram_path[1200];if(status==FG_OK)status=fg_q38_ngram_rank_range(rank,&row_begin,&row_count,err);if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),"%s/ngram-rank-%02u.iq4nl",directory,rank)>=(int)sizeof(ngram_path)){fg_error_set(err,FG_ERR_LIMIT,"resident n-gram shard path is too long");status=FG_ERR_LIMIT;}if(status==FG_OK)status=fg_ngram_resident_open(&ngram,ngram_path,row_begin,row_count,err);if(status==FG_OK)status=fg_fabric_open(&fabric,manifest,rank,err);if(status==FG_OK)status=rank_ready(fabric,rank,err);if(status==FG_OK){printf("rank %u READY: %.3f GiB sealed weights, %.3f GiB n-gram rows resident on %s\n",rank,(double)fg_model_weight_bytes(model)/(1024.0*1024.0*1024.0),(double)(row_count*FG_NGRAM_ROW_BYTES)/(1024.0*1024.0*1024.0),fg_vk_device_name(fg_model_vk(model)));fflush(stdout);status=rank_worker_loop(fabric,expert,output,ngram,model,manifest,rank,err);}fg_fabric_close(fabric);fg_ngram_resident_close(ngram);fg_output_executor_destroy(output);fg_expert_executor_destroy(expert);fg_model_close(model);free(manifest);return status;}

typedef struct fg_coordinator {const fg_manifest *manifest;fg_session_identity identity;fg_model *model;fg_expert_executor *expert;fg_owner_executor *owner;fg_fabric *fabric;fg_ngram_store *ngram;fg_tokenizer *tokenizer;fg_vk_tensor *hyper;prefill_worker_buffers prefill_expert;prefill_layer_buffers prefill_layer;uint64_t session_id;uint8_t *expert_recv;uint8_t *async_recv_payloads[FG_GROUP_SIZE];const char *directory;} fg_coordinator;

struct fg_runtime {
    fg_manifest *manifest;
    fg_coordinator coordinator;
    int32_t *history;
    size_t history_count,history_capacity;
    bool qsa_open;
    bool prefill_profiled;
    fg_runtime_options options;
    uint32_t context_limit;
    char directory[1024],qsa_path[1200];
};

static fg_status coordinator_begin_session(fg_coordinator *coordinator,fg_error *err){
    struct timespec now;if(clock_gettime(CLOCK_REALTIME,&now)!=0){fg_error_set(err,FG_ERR_IO,"read session clock");return FG_ERR_IO;}
    uint64_t request=((uint64_t)(uint32_t)now.tv_sec<<32u)^(uint32_t)now.tv_nsec^(uint64_t)(uint32_t)getpid();if(!request)request=1u;
    bool legacy=coordinator->manifest->protocol_version==FG_PROTOCOL_MIN_VERSION;
    for(uint32_t peer=1;peer<FG_RANK_COUNT;peer++){
        uint8_t wire[FG_OWNER_SESSION_CONTROL_BYTES];const void *payload=NULL;uint32_t bytes=0;
        if(!legacy){
            fg_owner_session_control control={
                .version=FG_OWNER_SESSION_CONTROL_VERSION,
                .operation=FG_OWNER_SESSION_BEGIN,.rank=(uint8_t)peer,
                .position_mode=(fg_position_mode)coordinator->manifest->session.position_mode,
                .session_nonce=request
            };
            memcpy(control.identity_sha256,coordinator->identity.identity_sha256,32u);
            memcpy(control.state_format_sha256,
                   coordinator->manifest->session.rank_state_format_sha256[peer],32u);
            fg_status status=fg_owner_session_control_encode(wire,&control,err);
            if(status!=FG_OK)return status;
            payload=wire;bytes=sizeof(wire);
        }
        fg_status status=fg_fabric_send(coordinator->fabric,peer,FG_FABRIC_CONTROL,
                                        FG_MSG_SESSION_BEGIN,request,0,0,payload,bytes,err);
        if(status!=FG_OK)return status;
    }
    bool ready[FG_RANK_COUNT]={0};uint8_t wire[FG_OWNER_SESSION_CONTROL_BYTES];
    for(uint32_t received=1;received<FG_RANK_COUNT;received++){
        uint32_t peer=0,bytes=0;fg_frame_header header;
        fg_status status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_CONTROL,&peer,
                                            &header,legacy?NULL:wire,legacy?0u:sizeof(wire),
                                            &bytes,err);
        if(status!=FG_OK)return status;
        if(fg_frame_type(&header)!=FG_MSG_SESSION_READY||
           fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=0u||
           peer==0u||ready[peer]||(legacy&&bytes)||(!legacy&&bytes!=sizeof(wire))){
            fg_error_set(err,FG_ERR_MISMATCH,"invalid session readiness from rank %u",peer);
            return FG_ERR_MISMATCH;
        }
        if(!legacy){
            fg_owner_session_control control;
            status=fg_owner_session_control_decode(&control,wire,bytes,err);
            if(status!=FG_OK)return status;
            if(control.operation!=FG_OWNER_SESSION_READY||control.rank!=peer||
               control.session_nonce!=request||
               control.position_mode!=(fg_position_mode)coordinator->manifest->session.position_mode||
               memcmp(control.identity_sha256,coordinator->identity.identity_sha256,32u)||
               memcmp(control.state_format_sha256,
                      coordinator->manifest->session.rank_state_format_sha256[peer],32u)){
                fg_error_set(err,FG_ERR_MISMATCH,"rank %u session readiness fingerprint mismatch",peer);
                return FG_ERR_MISMATCH;
            }
        }
        ready[peer]=true;
    }
    coordinator->session_id=request;return FG_OK;
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
    if(status==FG_OK){status=fg_vk_embedding_q8_0_batch(fg_model_vk(coordinator->model),buffers->hyper_tensor,embedding,buffers->token_tensor,token_count,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);}
    fg_vk_tensor *current=buffers->hyper_tensor;
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        struct timespec layer_start,layer_end;bool profiling=fg_vk_profile_active(fg_model_vk(coordinator->model));if(profiling)clock_gettime(CLOCK_MONOTONIC,&layer_start);
        const fg_vk_tensor *layer_ngram=(layer==1u)?ngram_embeddings:NULL;
        prefill_dispatch_context dispatch={coordinator->fabric,coordinator->expert,coordinator->manifest,0u,coordinator->session_id,first_token*FG_LAYER_COUNT+layer,&coordinator->prefill_expert};
        fg_vk_tensor *layer_out=NULL;
        status=fg_owner_prefill_layer(coordinator->owner,layer,first_token,buffers->positions,token_count,current,layer_ngram,dispatch_prefill_experts,&dispatch,&layer_out,err);
        if(status==FG_OK){current=layer_out;}
        if(profiling){clock_gettime(CLOCK_MONOTONIC,&layer_end);fprintf(stderr,"PREFILL_PROFILE_LAYER first=%u tokens=%u layer=%u wall_ms=%.3f\n",first_token,token_count,layer,elapsed_seconds(&layer_start,&layer_end)*1000.0);}
    }
    if(status==FG_OK){*output=current;}
    return status;
}

static fg_status coordinator_output(fg_coordinator *coordinator,uint32_t token_index,const fg_vk_tensor *hyper,uint32_t *next_token,float *logit,fg_error *err){fg_output_work *work=calloc(1,sizeof(*work));uint8_t *wire=malloc(FG_OUTPUT_WORK_BYTES);if(!work||!wire){free(wire);free(work);fg_error_set(err,FG_ERR_OOM,"allocate coordinator output exchange");return FG_ERR_OOM;}work->source_rank=0u;work->destination_rank=4u;work->token_index=token_index;fg_status status=fg_vk_tensor_read(hyper,0,work->hyper,sizeof(work->hyper),err);if(status==FG_OK)status=fg_output_work_encode(wire,work,err);uint32_t sequence=token_index*FG_LAYER_COUNT+FG_LAYER_COUNT;if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,4u,FG_FABRIC_CONTROL,FG_MSG_OUTPUT_WORK,coordinator->session_id,sequence,0,wire,FG_OUTPUT_WORK_BYTES,err);if(status==FG_OK){fg_frame_header header;uint32_t bytes=0;status=fg_fabric_recv(coordinator->fabric,4u,FG_FABRIC_BULK,&header,wire,FG_OUTPUT_RESULT_BYTES,&bytes,err);if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_RESULT||fg_frame_request_id(&header)!=coordinator->session_id||fg_frame_sequence(&header)!=sequence)){fg_error_set(err,FG_ERR_MISMATCH,"stale output result");status=FG_ERR_MISMATCH;}fg_output_result result;if(status==FG_OK)status=fg_output_result_decode(&result,wire,bytes,err);if(status==FG_OK&&(result.source_rank!=4u||result.destination_rank!=0u||result.token_index!=token_index)){fg_error_set(err,FG_ERR_MISMATCH,"misrouted output result");status=FG_ERR_MISMATCH;}if(status==FG_OK){*next_token=result.token;if(logit)*logit=result.logit;}}free(wire);free(work);return status;}

static fg_status ngram_rank_for_row(uint64_t row,uint32_t *owner,fg_error *err){if(!owner){fg_error_set(err,FG_ERR_ARGUMENT,"n-gram row owner output is null");return FG_ERR_ARGUMENT;}for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){uint64_t begin=0,count=0;fg_status status=fg_q38_ngram_rank_range(rank,&begin,&count,err);if(status!=FG_OK)return status;if(row>=begin&&row-begin<count){*owner=rank;return FG_OK;}}fg_error_set(err,FG_ERR_MISMATCH,"n-gram row %llu has no resident owner",(unsigned long long)row);return FG_ERR_MISMATCH;}

static fg_status coordinator_ngram_resident(fg_coordinator *coordinator,const int32_t *history,size_t history_count,uint32_t token_index,fg_vk_tensor **embedding,fg_error *err){uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];fg_status status=fg_q38_ngram_lookup(history,history_count,rows,addresses,err);fg_ngram_work work[FG_RANK_COUNT]={0};for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++)work[rank]=(fg_ngram_work){.source_rank=0u,.destination_rank=(uint8_t)rank,.token_index=token_index};for(uint32_t head=0;status==FG_OK&&head<FG_NGRAM_HEAD_COUNT;head++){uint32_t rank=0;status=ngram_rank_for_row(rows[head],&rank,err);uint32_t item=work[rank].item_count;if(status==FG_OK&&item>=FG_NGRAM_SHARD_MAX_ITEMS){fg_error_set(err,FG_ERR_LIMIT,"resident n-gram rank %u item overflow",rank);status=FG_ERR_LIMIT;}if(status==FG_OK){work[rank].heads[item]=(uint8_t)head;work[rank].rows[item]=rows[head];work[rank].item_count++;}}uint8_t work_wire[FG_NGRAM_WORK_MAX_BYTES],result_wire[FG_NGRAM_RESULT_MAX_BYTES],packed[FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES];uint32_t sent_count=0;double start=dispatch_ts();for(uint32_t rank=1u;status==FG_OK&&rank<FG_RANK_COUNT;rank++)if(work[rank].item_count){uint32_t bytes=0;status=fg_ngram_work_encode(work_wire,sizeof(work_wire),&bytes,&work[rank],err);if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,rank,FG_FABRIC_CONTROL,FG_MSG_NGRAM_WORK,coordinator->session_id,token_index,0,work_wire,bytes,err);if(status==FG_OK)sent_count++;}double fired=dispatch_ts();bool received[FG_RANK_COUNT]={0},seen_head[FG_NGRAM_HEAD_COUNT]={0};for(uint32_t count=0;status==FG_OK&&count<sent_count;count++){uint32_t peer=0,bytes=0;fg_frame_header header;status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_BULK,&peer,&header,result_wire,sizeof(result_wire),&bytes,err);if(status==FG_OK&&(peer==0u||peer>=FG_RANK_COUNT||received[peer]||fg_frame_type(&header)!=FG_MSG_NGRAM_RESULT||fg_frame_request_id(&header)!=coordinator->session_id||fg_frame_sequence(&header)!=token_index)){fg_error_set(err,FG_ERR_MISMATCH,"stale or duplicate resident n-gram result from rank %u",peer);status=FG_ERR_MISMATCH;}fg_ngram_result result;if(status==FG_OK)status=fg_ngram_result_decode(&result,result_wire,bytes,err);if(status==FG_OK&&(result.source_rank!=peer||result.destination_rank!=0u||result.token_index!=token_index||result.item_count!=work[peer].item_count)){fg_error_set(err,FG_ERR_MISMATCH,"misrouted resident n-gram result from rank %u",peer);status=FG_ERR_MISMATCH;}for(uint32_t i=0;status==FG_OK&&i<result.item_count;i++){uint32_t head=result.heads[i];bool expected=false;for(uint32_t j=0;j<work[peer].item_count;j++)if(work[peer].heads[j]==head){expected=true;break;}if(!expected||seen_head[head]){fg_error_set(err,FG_ERR_MISMATCH,"unexpected resident n-gram head %u from rank %u",head,peer);status=FG_ERR_MISMATCH;}else{memcpy(packed+(uint64_t)head*FG_NGRAM_ROW_BYTES,result.packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,FG_NGRAM_ROW_BYTES);seen_head[head]=true;}}if(status==FG_OK)received[peer]=true;}for(uint32_t head=0;status==FG_OK&&head<FG_NGRAM_HEAD_COUNT;head++)if(!seen_head[head]){fg_error_set(err,FG_ERR_MISMATCH,"missing resident n-gram head %u",head);status=FG_ERR_MISMATCH;}double collected=dispatch_ts();const char *verify=getenv("FG_NGRAM_VERIFY");if(status==FG_OK&&token_profile_requested(token_index)&&verify&&*verify&&strcmp(verify,"0")!=0){uint32_t mismatch=UINT32_MAX;status=fg_ngram_store_verify_packed(coordinator->ngram,addresses,FG_NGRAM_HEAD_COUNT,packed,&mismatch,err);if(status==FG_OK)fprintf(stderr,"NGRAM_RESIDENT_VERIFY token=%u rows=%u status=exact\n",token_index,FG_NGRAM_HEAD_COUNT);}if(status==FG_OK)status=fg_ngram_store_decode_packed(coordinator->ngram,packed,FG_NGRAM_HEAD_COUNT,embedding,err);double decoded=dispatch_ts();if(frame_trace_enabled())fprintf(stderr,"NGRAM_RESIDENT_TRACE token=%u workers=%u fire_ms=%.3f collect_ms=%.3f dequant_ms=%.3f total_ms=%.3f\n",token_index,sent_count,fired-start,collected-fired,decoded-collected,decoded-start);return status;}

/* Expert-parallel decode: all 48 layers on the coordinator, MoE dispatched to workers */
static fg_status coordinator_decode_token_local(fg_coordinator *coordinator,const int32_t *history,size_t history_count,uint32_t token_index,uint32_t *next_token,float *logit,fg_error *err){
    if(!history||!history_count||(uint32_t)history[history_count-1u]>=FG_Q38_VOCAB_SIZE){fg_error_set(err,FG_ERR_ARGUMENT,"invalid local decode token history");return FG_ERR_ARGUMENT;}
    fg_vk_context *vk=fg_model_vk(coordinator->model);token_profile_capture capture={0};fg_status status=token_profile_begin(&capture,vk,token_index,err);double frame_start=dispatch_ts();
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"embedding",err);
    if(status==FG_OK)status=fg_vk_embedding_q8_0(vk,coordinator->hyper,embedding,(uint32_t)history[history_count-1u],FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);
    double frame_embedding=dispatch_ts();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"ngram",err);
    fg_vk_tensor *ngram=NULL;if(status==FG_OK)status=coordinator_ngram_resident(coordinator,history,history_count,token_index,&ngram,err);
    double frame_ngram=dispatch_ts();
    uint32_t position[3]={token_index,token_index,token_index};
    fg_vk_tensor *current=coordinator->hyper;
    async_expert_context async_ctx={.fabric=coordinator->fabric,.expert=coordinator->expert,.manifest=coordinator->manifest,.self=0u,.request_id=coordinator->session_id,.critical_trace=token_profile_requested(token_index)};
    for(uint32_t i=0;i<FG_GROUP_SIZE;i++)async_ctx.recv_payloads[i]=coordinator->async_recv_payloads[i];
    /* Process all 48 layers locally while overlapping shared and routed experts. */
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        const fg_vk_tensor *layer_ngram=(layer==1u)?ngram:NULL;
        async_ctx.sequence=token_index*FG_LAYER_COUNT+layer;
        fg_vk_tensor *layer_out=NULL;
        status=fg_owner_decode_layer_async(coordinator->owner,layer,token_index,position,current,layer_ngram,fire_experts,collect_experts,&async_ctx,&layer_out,err);
        if(status==FG_OK)current=layer_out;
    }
    double frame_layers=dispatch_ts();
    if(status==FG_OK)status=coordinator_output(coordinator,token_index,current,next_token,logit,err);
    double frame_output=dispatch_ts();
    if(token_profile_requested(token_index)||frame_trace_enabled())fprintf(stderr,"TOKEN_FRAME_TRACE token=%u status=%d embedding_ms=%.3f ngram_ms=%.3f layers_ms=%.3f output_ms=%.3f total_ms=%.3f\n",token_index,(int)status,frame_embedding-frame_start,frame_ngram-frame_embedding,frame_layers-frame_ngram,frame_output-frame_layers,frame_output-frame_start);
    status=token_profile_end(&capture,0u,"token",token_index,UINT32_MAX,status,err);
    if(async_ctx.critical_trace)for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){for(uint32_t i=0;i<async_ctx.send_trace_count[layer];i++)fprintf(stderr,"EXPERT_COORD_SEND token=%u layer=%u peer=%u start_ns=%llu end_ns=%llu\n",token_index,layer,async_ctx.send_trace[layer][i].peer,(unsigned long long)async_ctx.send_trace[layer][i].start_ns,(unsigned long long)async_ctx.send_trace[layer][i].end_ns);for(uint32_t i=0;i<async_ctx.recv_trace_count[layer];i++)fprintf(stderr,"EXPERT_COORD_RECV token=%u layer=%u peer=%u bytes=%u ready_mask=%u poll_start_ns=%llu ready_ns=%llu header_end_ns=%llu payload_end_ns=%llu validate_end_ns=%llu decode_end_ns=%llu\n",token_index,layer,async_ctx.recv_trace[layer][i].peer,async_ctx.recv_trace[layer][i].bytes,async_ctx.recv_trace[layer][i].ready_mask,(unsigned long long)async_ctx.recv_trace[layer][i].poll_start_ns,(unsigned long long)async_ctx.recv_trace[layer][i].ready_ns,(unsigned long long)async_ctx.recv_trace[layer][i].header_end_ns,(unsigned long long)async_ctx.recv_trace[layer][i].payload_end_ns,(unsigned long long)async_ctx.recv_trace[layer][i].validate_end_ns,(unsigned long long)async_ctx.recv_trace[layer][i].decode_end_ns);}
    return status;
}

static void coordinator_close(fg_coordinator *coordinator){if(!coordinator)return;for(uint32_t i=0;i<FG_GROUP_SIZE;i++)free(coordinator->async_recv_payloads[i]);free(coordinator->expert_recv);prefill_layer_buffers_destroy(&coordinator->prefill_layer);prefill_worker_buffers_destroy(&coordinator->prefill_expert);fg_vk_tensor_destroy(coordinator->hyper);fg_ngram_store_close(coordinator->ngram);fg_tokenizer_close(coordinator->tokenizer);fg_fabric_close(coordinator->fabric);fg_owner_executor_destroy(coordinator->owner);fg_expert_executor_destroy(coordinator->expert);fg_model_close(coordinator->model);memset(coordinator,0,sizeof(*coordinator));}

static fg_status coordinator_open(fg_coordinator *coordinator,const fg_manifest *manifest,const char *directory,fg_error *err){memset(coordinator,0,sizeof(*coordinator));coordinator->manifest=manifest;fg_status status=fg_session_identity_from_manifest(manifest,&coordinator->identity,err);if(status==FG_OK)status=fg_model_open_replicated(&coordinator->model,manifest,directory,0u,err);if(status==FG_OK)status=fg_expert_executor_create(&coordinator->expert,coordinator->model,err);if(status==FG_OK)status=fg_owner_executor_create(&coordinator->owner,coordinator->model,err);if(status==FG_OK)status=fg_tokenizer_open(&coordinator->tokenizer,directory,manifest,err);if(status==FG_OK)status=fg_tokenizer_validate_qwen38(coordinator->tokenizer,err);const fg_tensor_record *ngram_record=NULL;for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)if(manifest->tensors[i].kind==FG_TENSOR_NGRAM){if(ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"multiple n-gram tensors in deployment manifest");status=FG_ERR_MISMATCH;}else ngram_record=&manifest->tensors[i];}char ngram_path[1200];if(status==FG_OK&&!ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"deployment manifest has no n-gram tensor");status=FG_ERR_MISMATCH;}if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),"%s/ngram.iq4nl",directory)>=(int)sizeof(ngram_path)){fg_error_set(err,FG_ERR_LIMIT,"n-gram path is too long");status=FG_ERR_LIMIT;}if(status==FG_OK)status=fg_ngram_store_open(&coordinator->ngram,fg_model_vk(coordinator->model),ngram_path,ngram_record->bytes,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(coordinator->model),FG_HYPER_WIDTH*sizeof(float),&coordinator->hyper,err);if(status==FG_OK){coordinator->expert_recv=malloc(FG_EXPERT_RESULT_MAX_BYTES);if(!coordinator->expert_recv){fg_error_set(err,FG_ERR_OOM,"allocate coordinator expert recv buffer");status=FG_ERR_OOM;}}
    /* Allocate per-group async recv payload buffers */
    for(uint32_t i=0;status==FG_OK&&i<FG_GROUP_SIZE;i++){coordinator->async_recv_payloads[i]=malloc(FG_EXPERT_RESULT_MAX_BYTES);if(!coordinator->async_recv_payloads[i]){fg_error_set(err,FG_ERR_OOM,"allocate async expert recv buffer %u",i);status=FG_ERR_OOM;}}if(status==FG_OK)status=prefill_worker_buffers_create(&coordinator->prefill_expert,manifest->prefill_microbatch,err);if(status==FG_OK)status=prefill_layer_buffers_create(&coordinator->prefill_layer,coordinator->model,manifest->prefill_microbatch,err);if(status==FG_OK)status=fg_fabric_open(&coordinator->fabric,manifest,0u,err);if(status==FG_OK)status=rank_ready(coordinator->fabric,0u,err);if(status==FG_OK)status=token_profile_prepare(fg_model_vk(coordinator->model),err);if(status==FG_OK)status=coordinator_begin_session(coordinator,err);if(status!=FG_OK)coordinator_close(coordinator);coordinator->directory=directory;return status;}

static double elapsed_seconds(const struct timespec *start,const struct timespec *end){return (double)(end->tv_sec-start->tv_sec)+(double)(end->tv_nsec-start->tv_nsec)*1e-9;}

static fg_status runtime_reserve_history(fg_runtime *runtime,size_t count,fg_error *err){
    if(count<=runtime->history_capacity)return FG_OK;
    size_t capacity=runtime->history_capacity?runtime->history_capacity:1024u;
    while(capacity<count){if(capacity>SIZE_MAX/2u){fg_error_set(err,FG_ERR_LIMIT,"session token history exceeds address space");return FG_ERR_LIMIT;}capacity*=2u;}
    int32_t *history=realloc(runtime->history,capacity*sizeof(*history));
    if(!history){fg_error_set(err,FG_ERR_OOM,"grow session token history");return FG_ERR_OOM;}
    runtime->history=history;runtime->history_capacity=capacity;return FG_OK;
}

void fg_runtime_options_init(fg_runtime_options *options){
    if(!options)return;
    memset(options,0,sizeof(*options));
}

static fg_status validate_runtime_options(fg_runtime_options *options,
                                          const fg_manifest *manifest,fg_error *err){
    uint32_t boot_context=manifest->native_context<FG_RUNTIME_BOOT_CONTEXT_TOKENS?
        manifest->native_context:FG_RUNTIME_BOOT_CONTEXT_TOKENS;
    if(!options->logical_context_tokens)options->logical_context_tokens=boot_context;
    if(!options->gpu_index_tokens)options->gpu_index_tokens=manifest->session.gpu_index_tokens;
    if(!options->qsa_hot_tokens)options->qsa_hot_tokens=options->logical_context_tokens;
    if(!options->prefill_microbatch)options->prefill_microbatch=manifest->prefill_microbatch;
    if(!options->prefill_window)options->prefill_window=manifest->prefill_window;
    if(options->prefill_microbatch!=manifest->prefill_microbatch||
       options->prefill_window!=manifest->prefill_window){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime prefill %ux%u does not match sealed manifest %ux%u",
                     options->prefill_microbatch,options->prefill_window,
                     manifest->prefill_microbatch,manifest->prefill_window);
        return FG_ERR_MISMATCH;
    }
    if(options->experimental_flags){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "experimental context, MTP, and vision are not enabled in this runtime");
        return FG_ERR_UNAVAILABLE;
    }
    if(options->logical_context_tokens>manifest->native_context){
        fg_error_set(err,FG_ERR_LIMIT,"logical context %u exceeds native context %u",
                     options->logical_context_tokens,manifest->native_context);
        return FG_ERR_LIMIT;
    }
    if(options->qsa_hot_tokens>options->logical_context_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA hot tokens %u exceed logical context %u",
                     options->qsa_hot_tokens,options->logical_context_tokens);
        return FG_ERR_ARGUMENT;
    }
    if(options->gpu_index_tokens>options->logical_context_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"GPU index tokens %u exceed logical context %u",
                     options->gpu_index_tokens,options->logical_context_tokens);
        return FG_ERR_ARGUMENT;
    }
    if(options->logical_context_tokens!=boot_context||
       options->gpu_index_tokens!=boot_context||
       options->qsa_hot_tokens!=boot_context||
       options->qsa_page_cache_bytes){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "tiered QSA is not enabled; use logical/index/hot context 8192 and page cache 0");
        return FG_ERR_UNAVAILABLE;
    }
    return FG_OK;
}

fg_status fg_runtime_open_with_options(fg_runtime **out,const char *path,
                                       const fg_runtime_options *requested,fg_error *err){
    if(!out||!path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime open arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    fg_runtime *runtime=calloc(1,sizeof(*runtime));if(!runtime){fg_error_set(err,FG_ERR_OOM,"allocate resident runtime");return FG_ERR_OOM;}
    fg_runtime_options_init(&runtime->options);
    if(requested)runtime->options=*requested;
    fg_status status=load_checked(path,&runtime->manifest,err);
    if(status==FG_OK)status=validate_runtime_options(&runtime->options,runtime->manifest,err);
    if(status==FG_OK)runtime->context_limit=runtime->options.logical_context_tokens;
    if(status==FG_OK)status=manifest_directory(path,runtime->directory,err);
    if(status==FG_OK)status=coordinator_open(&runtime->coordinator,runtime->manifest,runtime->directory,err);
    if(status==FG_OK){int n=snprintf(runtime->qsa_path,sizeof(runtime->qsa_path),"%s/session-%016llx-rank-00.qsa",runtime->directory,(unsigned long long)runtime->coordinator.session_id);if(n<0||(size_t)n>=sizeof(runtime->qsa_path)){fg_error_set(err,FG_ERR_LIMIT,"coordinator QSA path overflow");status=FG_ERR_LIMIT;}}
    if(status==FG_OK)status=fg_runtime_reset(runtime,err);
    if(status!=FG_OK){fg_runtime_close(runtime);return status;}*out=runtime;return FG_OK;
}

fg_status fg_runtime_open(fg_runtime **out,const char *path,fg_error *err){
    return fg_runtime_open_with_options(out,path,NULL,err);
}

void fg_runtime_close(fg_runtime *runtime){
    if(!runtime)return;
    coordinator_close(&runtime->coordinator);
    if(runtime->qsa_path[0])unlink(runtime->qsa_path);
    free(runtime->history);free(runtime->manifest);free(runtime);
}

fg_status fg_runtime_reset(fg_runtime *runtime,fg_error *err){
    if(!runtime||!runtime->coordinator.owner||!runtime->qsa_path[0]){fg_error_set(err,FG_ERR_ARGUMENT,"resident runtime is not open");return FG_ERR_ARGUMENT;}
    fg_status status=fg_owner_reset_state(runtime->coordinator.owner,err);if(status!=FG_OK)return status;
    runtime->history_count=0;
    if(runtime->qsa_open)return FG_OK;
    if(unlink(runtime->qsa_path)!=0&&errno!=ENOENT){fg_error_set(err,FG_ERR_IO,"remove stale QSA session: %s",strerror(errno));return FG_ERR_IO;}
    status=fg_owner_qsa_open_decode(runtime->coordinator.owner,runtime->qsa_path,
                                    runtime->context_limit,
                                    runtime->manifest->prefill_microbatch,err);
    if(status==FG_OK)runtime->qsa_open=true;
    return status;
}

fg_status fg_runtime_generate(fg_runtime *runtime,const char *suffix,uint32_t max_tokens,
                              fg_token_callback callback,void *callback_context,
                              fg_interrupt_fn interrupted,void *interrupt_context,
                              fg_generation_stats *stats,fg_error *err){
    if(!runtime||!suffix||!callback||!max_tokens||max_tokens>4096u){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident generation arguments");return FG_ERR_ARGUMENT;}
    if(stats)memset(stats,0,sizeof(*stats));
    fg_tokens prompt={0};fg_status status=fg_tokenizer_encode(runtime->coordinator.tokenizer,suffix,true,&prompt,err);
    size_t old_count=runtime->history_count,total=old_count+prompt.count;
    if(status==FG_OK&&(!prompt.count||total+(size_t)max_tokens>runtime->context_limit)){fg_error_set(err,FG_ERR_LIMIT,"prompt plus generation would use %zu of %u context tokens",total+(size_t)max_tokens,runtime->context_limit);status=FG_ERR_LIMIT;}
    if(status==FG_OK)status=runtime_reserve_history(runtime,total+(size_t)max_tokens,err);
    for(size_t i=0;status==FG_OK&&i<prompt.count;i++)runtime->history[old_count+i]=(int32_t)prompt.data[i];
    runtime->history_count=status==FG_OK?total:old_count;
    struct timespec prefill_start,prefill_end,decode_start,decode_end;
    uint32_t next=0;float logit=0.0f;fg_vk_tensor *prefill_output=NULL;
    if(status==FG_OK)clock_gettime(CLOCK_MONOTONIC,&prefill_start);
    for(size_t offset=0;status==FG_OK&&offset<prompt.count;){
        uint32_t count=(uint32_t)(prompt.count-offset);if(count>runtime->manifest->prefill_microbatch)count=runtime->manifest->prefill_microbatch;
        uint32_t first=(uint32_t)(old_count+offset);fg_vk_tensor *ngram_batch=NULL;
        fg_vk_context *vk=fg_model_vk(runtime->coordinator.model);bool capture=!runtime->prefill_profiled&&prefill_profile_requested(),capture_active=false;struct timespec capture_start,capture_end;
        if(capture){status=fg_vk_profile_begin(vk,err);if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"ngram_prefill",err);if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&capture_start);capture_active=true;}}
        if(status==FG_OK)status=fg_ngram_store_lookup_prefill(runtime->coordinator.ngram,runtime->history,runtime->history_count,first,count,&ngram_batch,err);
        if(status==FG_OK)status=coordinator_prefill_microbatch(&runtime->coordinator,prompt.data+offset,first,(uint16_t)count,ngram_batch,&prefill_output,err);
        if(capture_active){fg_vk_profile profile={0};fg_error profile_error={0};clock_gettime(CLOCK_MONOTONIC,&capture_end);fg_status profile_status=fg_vk_profile_end(vk,&profile,status==FG_OK?err:&profile_error);runtime->prefill_profiled=true;if(status==FG_OK&&profile_status!=FG_OK)status=profile_status;fprintf(stderr,"PREFILL_PROFILE first=%u tokens=%u wall_ms=%.3f gpu_ms=%.3f kernel_ms=%.3f submissions=%llu dispatches=%llu\n",first,count,elapsed_seconds(&capture_start,&capture_end)*1000.0,profile.gpu_ms,profile.kernel_ms,(unsigned long long)profile.submissions,(unsigned long long)profile.dispatches);for(uint32_t i=0;i<profile.kernel_count;i++){const fg_vk_profile_kernel *kernel=&profile.kernels[i];fprintf(stderr,"PREFILL_PROFILE_KERNEL scope=%s kernel=%s calls=%llu gpu_ms=%.3f\n",kernel->scope,kernel->name,(unsigned long long)kernel->invocations,kernel->gpu_ms);}}
        offset+=count;
    }
    fg_vk_tensor *last_hyper=NULL;
    if(status==FG_OK){uint32_t final_count=(uint32_t)(prompt.count%runtime->manifest->prefill_microbatch);if(!final_count)final_count=runtime->manifest->prefill_microbatch;status=fg_vk_tensor_view(prefill_output,(uint64_t)(final_count-1u)*FG_HYPER_WIDTH*4u,FG_HYPER_WIDTH*4u,&last_hyper,err);}
    if(status==FG_OK)status=coordinator_output(&runtime->coordinator,(uint32_t)runtime->history_count-1u,last_hyper,&next,&logit,err);
    fg_vk_tensor_destroy(last_hyper);
    if(status==FG_OK){fg_owner_qsa_set_tokens(runtime->coordinator.owner,(uint32_t)runtime->history_count);clock_gettime(CLOCK_MONOTONIC,&prefill_end);clock_gettime(CLOCK_MONOTONIC,&decode_start);}
    uint32_t generated=0;
    while(status==FG_OK&&generated<max_tokens){
        if(interrupted&&interrupted(interrupt_context))break;
        if(next==fg_tokenizer_eos(runtime->coordinator.tokenizer))break;
        char decoded[4096];size_t bytes=0;status=fg_tokenizer_decode_token(runtime->coordinator.tokenizer,next,decoded,sizeof(decoded),&bytes,err);
        if(status==FG_OK)status=callback(callback_context,next,decoded,bytes,err);
        if(status!=FG_OK)break;
        runtime->history[runtime->history_count++]=(int32_t)next;generated++;
        status=coordinator_decode_token_local(&runtime->coordinator,runtime->history,runtime->history_count,(uint32_t)runtime->history_count-1u,&next,&logit,err);
    }
    if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&decode_end);if(stats){stats->prompt_tokens=(uint32_t)prompt.count;stats->generated_tokens=generated;stats->context_tokens=(uint32_t)runtime->history_count;stats->prefill_seconds=elapsed_seconds(&prefill_start,&prefill_end);stats->decode_seconds=elapsed_seconds(&decode_start,&decode_end);}}
    if(status!=FG_OK&&runtime->history_count==total)runtime->history_count=old_count;
    fg_tokens_free(&prompt);return status;
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
    fg_manifest *manifest=NULL;fg_status status=load_checked(path,&manifest,err);char directory[1024];if(status==FG_OK)status=manifest_directory(path,directory,err);fg_coordinator coordinator={0};if(status==FG_OK)status=coordinator_open(&coordinator,manifest,directory,err);fg_tokens prompt_tokens={0};if(status==FG_OK)status=fg_tokenizer_encode(coordinator.tokenizer,prompt,true,&prompt_tokens,err);if(status==FG_OK&&(!prompt_tokens.count||prompt_tokens.count>manifest->native_context)){fg_error_set(err,FG_ERR_LIMIT,"eval prompt token count %zu is outside native context",prompt_tokens.count);status=FG_ERR_LIMIT;}
    size_t capacity=prompt_tokens.count+(size_t)generate;int32_t *history=status==FG_OK?malloc(capacity*sizeof(*history)):NULL;if(status==FG_OK&&!history){fg_error_set(err,FG_ERR_OOM,"allocate eval token history");status=FG_ERR_OOM;}
    /* Open coordinator QSA decode session with actual token capacity */
    if(status==FG_OK){char qsa_path[1200];int n=snprintf(qsa_path,sizeof(qsa_path),"%s/session-%016llx-rank-00.qsa",directory,(unsigned long long)coordinator.session_id);if(n<0||(uint32_t)n>=sizeof(qsa_path)){fg_error_set(err,FG_ERR_LIMIT,"coordinator QSA path overflow");status=FG_ERR_LIMIT;}else status=fg_owner_qsa_open_decode(coordinator.owner,qsa_path,(uint32_t)capacity,manifest->prefill_microbatch,err);}
    for(size_t i=0;status==FG_OK&&i<prompt_tokens.count;i++){history[i]=(int32_t)prompt_tokens.data[i];}uint32_t next=0;float logit=0.0f;struct timespec start,end;fg_vk_tensor *prefill_output=NULL;if(status==FG_OK)clock_gettime(CLOCK_MONOTONIC,&start);for(uint32_t first=0;status==FG_OK&&first<prompt_tokens.count;){uint32_t count=(uint32_t)(prompt_tokens.count-first);if(count>manifest->prefill_microbatch)count=manifest->prefill_microbatch;fg_vk_tensor *ngram_batch=NULL;status=fg_ngram_store_lookup_prefill(coordinator.ngram,history,prompt_tokens.count,first,count,&ngram_batch,err);if(status==FG_OK)status=coordinator_prefill_microbatch(&coordinator,prompt_tokens.data+first,first,(uint16_t)count,ngram_batch,&prefill_output,err);first+=count;}fg_vk_tensor *last_hyper=NULL;if(status==FG_OK){uint32_t final_count=(uint32_t)(prompt_tokens.count%manifest->prefill_microbatch);if(!final_count)final_count=manifest->prefill_microbatch;status=fg_vk_tensor_view(prefill_output,(uint64_t)(final_count-1u)*FG_HYPER_WIDTH*4u,FG_HYPER_WIDTH*4u,&last_hyper,err);}if(status==FG_OK)status=coordinator_output(&coordinator,(uint32_t)prompt_tokens.count-1u,last_hyper,&next,&logit,err);fg_vk_tensor_destroy(last_hyper);if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&end);double seconds=(double)(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)*1e-9;fprintf(stderr,"prefill: %zu tokens in %.3f s (%.2f tok/s), next=%u logit=%g\n",prompt_tokens.count,seconds,(double)prompt_tokens.count/seconds,next,logit);}
    if(status==FG_OK)fg_owner_qsa_set_tokens(coordinator.owner,(uint32_t)prompt_tokens.count);
    size_t history_count=prompt_tokens.count;struct timespec decode_start,decode_tok;clock_gettime(CLOCK_MONOTONIC,&decode_start);for(uint32_t generated=0;status==FG_OK&&generated<generate;generated++){char decoded[4096];size_t bytes=0;status=fg_tokenizer_decode_token(coordinator.tokenizer,next,decoded,sizeof(decoded),&bytes,err);if(status!=FG_OK)break;clock_gettime(CLOCK_MONOTONIC,&decode_tok);double tok_elapsed=(double)(decode_tok.tv_sec-decode_start.tv_sec)+(double)(decode_tok.tv_nsec-decode_start.tv_nsec)*1e-9;double tok_per_sec=generated>0?(double)generated/tok_elapsed:0.0;fprintf(stderr,"decode[%u]: token=%u logit=%.4f (%.3f s, avg %.2f tok/s)\n",generated,next,logit,tok_elapsed,tok_per_sec);fwrite(decoded,1,bytes,stdout);fflush(stdout);if(next==fg_tokenizer_eos(coordinator.tokenizer)||generated+1u==generate)break;history[history_count++]=(int32_t)next;status=coordinator_decode_token_local(&coordinator,history,history_count,(uint32_t)(history_count-1u),&next,&logit,err);}if(status==FG_OK){clock_gettime(CLOCK_MONOTONIC,&decode_tok);double total=(double)(decode_tok.tv_sec-decode_start.tv_sec)+(double)(decode_tok.tv_nsec-decode_start.tv_nsec)*1e-9;fprintf(stderr,"decode complete: %.2f tok/s avg\n",total>0?(double)(generate)/total:0.0);fputc('\n',stdout);}
    if(status==FG_OK)status=fg_owner_qsa_checkpoint(coordinator.owner,err);
    free(history);
    fg_tokens_free(&prompt_tokens);
    if(manifest)coordinator_close(&coordinator);
    free(manifest);
    free(wrapped);
    return status;
}
