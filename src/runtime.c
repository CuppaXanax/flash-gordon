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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static fg_status load_checked(const char *path,fg_manifest **out,fg_error *err){fg_manifest *m=malloc(sizeof(*m));if(!m){fg_error_set(err,FG_ERR_OOM,"allocate manifest");return FG_ERR_OOM;}fg_status rc=fg_manifest_read(path,m,err);if(rc==FG_OK)rc=fg_manifest_validate_deployment(m,err);if(rc!=FG_OK){free(m);return rc;}*out=m;return FG_OK;}
static fg_status manifest_directory(const char *path,char output[1024],fg_error *err){size_t length=strlen(path);if(!length||length>=1024u){fg_error_set(err,FG_ERR_ARGUMENT,"manifest path is invalid");return FG_ERR_ARGUMENT;}memcpy(output,path,length+1u);char *slash=strrchr(output,'/');if(!slash){snprintf(output,1024,".");return FG_OK;}if(slash==output)slash[1]=0;else *slash=0;return FG_OK;}

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

typedef struct expert_dispatch_context {fg_fabric *fabric;fg_expert_executor *expert;const fg_manifest *manifest;uint32_t self;uint64_t request_id;uint32_t sequence;uint8_t *recv_wire;} expert_dispatch_context;

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
} async_expert_context;

static double dispatch_ts(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e3+(double)t.tv_nsec*1e-6;}
static fg_status dispatch_experts(void *opaque,uint32_t layer,uint32_t token,const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],const uint8_t *activation,fg_expert_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    expert_dispatch_context *context=opaque;fg_expert_route routes[FG_GROUP_SIZE];uint32_t route_count=0;double t0=dispatch_ts();fg_status status=fg_partition_route(context->manifest,layer,expert_ids,gates,routes,&route_count,err);uint8_t work_wire[FG_DECODE_WORK_BYTES];uint32_t remote_count=0;*result_count=0;double t_route=dispatch_ts();
    /* Phase 1: fire ALL remote requests first so they compute while we do local work */
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){if(routes[r].destination_rank==context->self)continue;fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,.destination_rank=routes[r].destination_rank,.selected_count=routes[r].selected_count,.position=token};for(uint32_t i=0;i<routes[r].selected_count;i++){work.expert_ids[i]=routes[r].global_expert_ids[i];work.routing_slots[i]=routes[r].routing_slots[i];work.gates[i]=routes[r].gates[i];}memcpy(work.activation_q8k,activation,FG_Q8K_ACTIVATION_BYTES);status=fg_decode_work_encode(work_wire,&work,err);if(status==FG_OK)status=fg_fabric_send(context->fabric,work.destination_rank,FG_FABRIC_CONTROL,FG_MSG_DECODE_WORK,context->request_id,context->sequence,0,work_wire,sizeof(work_wire),err);if(status==FG_OK)remote_count++;}double t_send=dispatch_ts();
    /* Phase 2: compute local experts while remote requests are in flight */
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){if(routes[r].destination_rank!=context->self)continue;fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)context->self,.destination_rank=(uint8_t)context->self,.selected_count=routes[r].selected_count,.position=token};for(uint32_t i=0;i<routes[r].selected_count;i++){work.expert_ids[i]=routes[r].global_expert_ids[i];work.routing_slots[i]=routes[r].routing_slots[i];work.gates[i]=routes[r].gates[i];}memcpy(work.activation_q8k,activation,FG_Q8K_ACTIVATION_BYTES);status=fg_expert_decode(context->expert,&work,&results[*result_count],err);if(status==FG_OK)(*result_count)++;}double t_local=dispatch_ts();
    /* Phase 3: collect remote results in arrival order (not route order) */
    uint8_t *result_wire=context->recv_wire;if(remote_count&&status==FG_OK&&!result_wire){fg_error_set(err,FG_ERR_OOM,"pre-allocated expert recv buffer is null");status=FG_ERR_OOM;}double t_recv[4]={0};for(uint32_t received=0;status==FG_OK&&received<remote_count;received++){uint32_t peer=0;fg_frame_header header;uint32_t bytes=0;status=fg_fabric_recv_any(context->fabric,FG_FABRIC_BULK,&peer,&header,result_wire,FG_EXPERT_RESULT_MAX_BYTES,&bytes,err);t_recv[received]=dispatch_ts();if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_EXPERT_RESULT||fg_frame_request_id(&header)!=context->request_id||fg_frame_sequence(&header)!=context->sequence)){fg_error_set(err,FG_ERR_MISMATCH,"stale expert result frame from rank %u",peer);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_expert_result_decode(&results[*result_count],result_wire,bytes,err);if(status==FG_OK)(*result_count)++;}double t_end=dispatch_ts();
    if(token>=26u&&token<28u&&status==FG_OK){fprintf(stderr,"EXPERT_DIAG layer[%u] t=%u route=%.2f send=%.2f(%u) local=%.2f recv1=%.2f recv2=%.2f recv3=%.2f total=%.2f\n",layer,token,t_route-t0,t_send-t_route,remote_count,t_local-t_send,remote_count>=1?t_recv[0]-t_local:0.0,remote_count>=2?t_recv[1]-t_recv[0]:0.0,remote_count>=3?t_recv[2]-t_recv[1]:0.0,t_end-t0);}
    return status;
}

/* Async dispatch functions — reserved for future multi-token batching.
   Single-token decode uses sync dispatch (poll-based recv is faster). */
#if 0
/* Async fire: send remote expert work + submit header recv SQEs, return immediately.
   Local experts are computed in collect to overlap with network wait. */
static fg_status fire_experts(void *opaque,uint32_t layer,uint32_t token,const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],const uint8_t *activation,fg_error *err){
    async_expert_context *ctx=opaque;fg_expert_route routes[FG_GROUP_SIZE];uint32_t route_count=0;
    fg_status status=fg_partition_route(ctx->manifest,layer,expert_ids,gates,routes,&route_count,err);
    uint8_t work_wire[FG_DECODE_WORK_BYTES];ctx->remote_count=0;
    /* Save local routes for deferred compute in collect */
    ctx->route_count=route_count;
    memcpy(ctx->routes,routes,sizeof(routes));
    memcpy(ctx->activation_copy,activation,FG_Q8K_ACTIVATION_BYTES);
    /* Send to all remote peers (sync — sends are fast ~0.17ms total) */
    for(uint32_t r=0;status==FG_OK&&r<route_count;r++){
        if(routes[r].destination_rank==ctx->self)continue;
        fg_decode_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)ctx->self,.destination_rank=routes[r].destination_rank,.selected_count=routes[r].selected_count,.position=token};
        for(uint32_t i=0;i<routes[r].selected_count;i++){work.expert_ids[i]=routes[r].global_expert_ids[i];work.routing_slots[i]=routes[r].routing_slots[i];work.gates[i]=routes[r].gates[i];}
        memcpy(work.activation_q8k,activation,FG_Q8K_ACTIVATION_BYTES);
        status=fg_decode_work_encode(work_wire,&work,err);
        if(status==FG_OK)status=fg_fabric_send(ctx->fabric,work.destination_rank,FG_FABRIC_CONTROL,FG_MSG_DECODE_WORK,ctx->request_id,ctx->sequence,0,work_wire,sizeof(work_wire),err);
        if(status==FG_OK){ctx->recv_peers[ctx->remote_count]=(uint32_t)work.destination_rank;ctx->remote_count++;}
    }
    /* Submit async header recv SQEs for all expected responses */
    for(uint32_t i=0;status==FG_OK&&i<ctx->remote_count;i++){
        uint64_t tag=(uint64_t)i+1u;
        status=fg_fabric_prep_header_recv(ctx->fabric,ctx->recv_peers[i],FG_FABRIC_BULK,&ctx->recv_headers[i],tag,err);
    }
    if(status==FG_OK&&ctx->remote_count)status=fg_fabric_io_flush(ctx->fabric,ctx->remote_count,err);
    return status;
}

/* Async collect: compute local experts (overlaps with network), then reap remote results */
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
    if(!ctx->remote_count)return FG_OK;
    /* Reap header CQEs */
    fg_uring_cqe cqes[FG_GROUP_SIZE];uint32_t completed=0;
    status=fg_fabric_io_reap(ctx->fabric,ctx->remote_count,cqes,FG_GROUP_SIZE,&completed,err);
    /* Validate headers and submit payload recvs */
    for(uint32_t i=0;status==FG_OK&&i<completed;i++){
        uint32_t idx=(uint32_t)(cqes[i].tag-1u);
        if(idx>=ctx->remote_count){fg_error_set(err,FG_ERR_MISMATCH,"invalid async recv tag");return FG_ERR_MISMATCH;}
        if(cqes[i].result<0){fg_error_set(err,FG_ERR_IO,"async header recv: %s",strerror(-cqes[i].result));return FG_ERR_IO;}
        if((uint32_t)cqes[i].result!=sizeof(fg_frame_header)){fg_error_set(err,FG_ERR_IO,"short async header recv: %d",cqes[i].result);return FG_ERR_IO;}
        fg_frame_header *h=&ctx->recv_headers[idx];
        if(fg_frame_type(h)!=FG_MSG_EXPERT_RESULT||fg_frame_request_id(h)!=ctx->request_id||fg_frame_sequence(h)!=ctx->sequence){
            fg_error_set(err,FG_ERR_MISMATCH,"stale async expert result from peer %u",ctx->recv_peers[idx]);return FG_ERR_MISMATCH;}
        uint32_t payload_bytes=ntohl(h->bytes_be);
        if(payload_bytes>FG_EXPERT_RESULT_MAX_BYTES){fg_error_set(err,FG_ERR_LIMIT,"async expert payload too large");return FG_ERR_LIMIT;}
        uint64_t ptag=((uint64_t)idx+1u)|0x80000000u; /* payload tag */
        status=fg_fabric_prep_payload_recv(ctx->fabric,ctx->recv_peers[idx],FG_FABRIC_BULK,ctx->recv_payloads[idx],payload_bytes,ptag,err);
    }
    if(status==FG_OK)status=fg_fabric_io_flush(ctx->fabric,ctx->remote_count,err);
    /* Reap payload CQEs */
    completed=0;
    if(status==FG_OK)status=fg_fabric_io_reap(ctx->fabric,ctx->remote_count,cqes,FG_GROUP_SIZE,&completed,err);
    for(uint32_t i=0;status==FG_OK&&i<completed;i++){
        uint32_t idx=(uint32_t)((cqes[i].tag&0x7FFFFFFFu)-1u);
        if(idx>=ctx->remote_count){fg_error_set(err,FG_ERR_MISMATCH,"invalid async payload tag");return FG_ERR_MISMATCH;}
        if(cqes[i].result<0){fg_error_set(err,FG_ERR_IO,"async payload recv: %s",strerror(-cqes[i].result));return FG_ERR_IO;}
        fg_frame_header *h=&ctx->recv_headers[idx];uint32_t pbytes=0;
        status=fg_frame_validate(h,ctx->recv_payloads[idx],&pbytes,err);
        if(status==FG_OK)status=fg_expert_result_decode(&results[*result_count],ctx->recv_payloads[idx],pbytes,err);
        if(status==FG_OK)(*result_count)++;
    }
    return status;
}
#endif /* async dispatch — reserved for multi-token batching */

static fg_status handle_expert_work(fg_fabric *fabric,fg_expert_executor *expert,uint32_t self,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_expert_result *result,uint8_t *wire,fg_error *err){fg_decode_work work;fg_status status=fg_decode_work_decode(&work,payload,bytes,err);if(status==FG_OK&&(peer!=work.source_rank||work.destination_rank!=self)){fg_error_set(err,FG_ERR_MISMATCH,"decode work peer/rank mismatch");status=FG_ERR_MISMATCH;}uint32_t result_bytes=0;if(status==FG_OK)status=fg_expert_decode(expert,&work,result,err);if(status==FG_OK)status=fg_expert_result_encode(wire,FG_EXPERT_RESULT_MAX_BYTES,&result_bytes,result,err);if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_EXPERT_RESULT,fg_frame_request_id(header),fg_frame_sequence(header),0,wire,result_bytes,err);return status;}

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

static fg_status prefill_layer_buffers_create(prefill_layer_buffers *buffers,fg_model *model,uint32_t tokens,fg_error *err){memset(buffers,0,sizeof(*buffers));if(!tokens||tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"invalid prefill layer buffer token count");return FG_ERR_MISMATCH;}buffers->tokens=tokens;buffers->receive_capacity=FG_PREFILL_LAYER_HEADER_BYTES+tokens*3u*4u+tokens*FG_HYPER_WIDTH*4u+tokens*FG_NGRAM_EMBED_VALUES*4u;buffers->result_capacity=FG_PREFILL_LAYER_HEADER_BYTES+tokens*FG_HYPER_WIDTH*4u;buffers->receive=malloc(buffers->receive_capacity);buffers->result_wire=malloc(buffers->result_capacity);buffers->positions=malloc((size_t)tokens*3u*4u);buffers->hyper=malloc((size_t)tokens*FG_HYPER_WIDTH*4u);buffers->ngram=malloc((size_t)tokens*FG_NGRAM_EMBED_VALUES*4u);buffers->output=malloc((size_t)tokens*FG_HYPER_WIDTH*4u);if(!buffers->receive||!buffers->result_wire||!buffers->positions||!buffers->hyper||!buffers->ngram||!buffers->output){prefill_layer_buffers_destroy(buffers);fg_error_set(err,FG_ERR_OOM,"allocate bounded prefill layer buffers");return FG_ERR_OOM;}fg_status status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*FG_HYPER_WIDTH*4u,&buffers->hyper_tensor,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*FG_NGRAM_EMBED_VALUES*4u,&buffers->ngram_tensor,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(model),(uint64_t)tokens*4u,&buffers->token_tensor,err);if(status!=FG_OK)prefill_layer_buffers_destroy(buffers);return status;}

static fg_status begin_session(fg_fabric *fabric,fg_owner_executor *owner,const char *directory,uint32_t self,uint32_t peer,const fg_frame_header *header,uint64_t *session_id,fg_error *err){uint64_t request=fg_frame_request_id(header);if(peer!=0u||!request||*session_id){fg_error_set(err,FG_ERR_MISMATCH,"invalid or duplicate session begin");return FG_ERR_MISMATCH;}fg_status status=FG_OK;if(self==3u||self==7u){char path[1200];int n=snprintf(path,sizeof(path),"%s/session-%016llx-rank-%02u.qsa",directory,(unsigned long long)request,self);if(n<0||(uint32_t)n>=sizeof(path)){fg_error_set(err,FG_ERR_LIMIT,"QSA session path overflow");status=FG_ERR_LIMIT;}else status=fg_owner_qsa_open(owner,path,true,err);}if(status==FG_OK){*session_id=request;status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_SESSION_READY,request,0,0,NULL,0,err);}return status;}

static fg_status handle_layer_work(fg_fabric *fabric,fg_expert_executor *expert,fg_owner_executor *owner,const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_vk_tensor *ngram_tensor,fg_layer_work *work,fg_layer_result *result,uint8_t *wire,uint8_t *next_wire,uint8_t *expert_recv,fg_error *err){
    fg_status status=fg_layer_work_decode(work,payload,bytes,err);uint64_t request=fg_frame_request_id(header);uint64_t expected_sequence=(uint64_t)work->token_index*FG_LAYER_COUNT+work->layer;
    if(status==FG_OK&&(!session_id||request!=session_id||work->layer==0u||peer!=work->source_rank||work->source_rank!=manifest->layer_owner[work->layer-1u]||work->destination_rank!=self||manifest->layer_owner[work->layer]!=self||((work->layer==1u)!=((work->flags&FG_LAYER_WORK_HAS_NGRAM)!=0))||expected_sequence>UINT32_MAX||fg_frame_sequence(header)!=(uint32_t)expected_sequence)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted layer work");status=FG_ERR_MISMATCH;}
    if(status==FG_OK)status=fg_vk_tensor_write(hyper_tensor,0,work->hyper,sizeof(work->hyper),err);
    const fg_vk_tensor *ngram=NULL;
    if(status==FG_OK&&(work->flags&FG_LAYER_WORK_HAS_NGRAM)){if(!ngram_tensor){fg_error_set(err,FG_ERR_MISMATCH,"layer work carries n-gram data without a tensor");status=FG_ERR_MISMATCH;}else{status=fg_vk_tensor_write(ngram_tensor,0,work->ngram_embedding,sizeof(work->ngram_embedding),err);ngram=ngram_tensor;}}
    expert_dispatch_context dispatch={fabric,expert,manifest,self,request,fg_frame_sequence(header),expert_recv};fg_vk_tensor *output=NULL;
    if(status==FG_OK)status=fg_owner_decode_layer(owner,work->layer,work->token_index,work->position,hyper_tensor,ngram,dispatch_experts,&dispatch,&output,err);
    if(status==FG_OK){memset(result,0,sizeof(*result));result->layer=work->layer;result->source_rank=(uint8_t)self;result->destination_rank=0u;result->token_index=work->token_index;status=fg_vk_tensor_read(output,0,result->hyper,sizeof(result->hyper),err);}
    uint32_t send_bytes=0;
    if(status==FG_OK&&work->layer+1u<FG_LAYER_COUNT){uint32_t next_layer=(uint32_t)work->layer+1u,next_owner=manifest->layer_owner[next_layer];uint64_t next_sequence=expected_sequence+1u;if(next_sequence>UINT32_MAX){fg_error_set(err,FG_ERR_LIMIT,"decode sequence exceeds protocol range");status=FG_ERR_LIMIT;}else{fg_layer_work next={.layer=(uint8_t)next_layer,.source_rank=(uint8_t)self,.destination_rank=(uint8_t)next_owner,.token_index=work->token_index};memcpy(next.position,work->position,sizeof(next.position));memcpy(next.hyper,result->hyper,sizeof(next.hyper));status=fg_layer_work_encode(next_wire,FG_LAYER_WORK_MAX_BYTES,&send_bytes,&next,err);if(status==FG_OK)status=fg_fabric_send(fabric,next_owner,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,(uint32_t)next_sequence,0,next_wire,send_bytes,err);}}else if(status==FG_OK){status=fg_layer_result_encode(wire,result,err);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_LAYER_RESULT,request,fg_frame_sequence(header),0,wire,FG_LAYER_RESULT_BYTES,err);}
    return status;
}

static fg_status handle_prefill_layer_work(fg_fabric *fabric,fg_expert_executor *expert,fg_owner_executor *owner,const fg_manifest *manifest,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,prefill_worker_buffers *expert_buffers,prefill_layer_buffers *layer_buffers,fg_error *err){
    fg_prefill_layer_work work={0};fg_status status=fg_prefill_layer_work_decode(&work,layer_buffers->positions,layer_buffers->tokens*3u,layer_buffers->hyper,(uint64_t)layer_buffers->tokens*FG_HYPER_WIDTH,layer_buffers->ngram,(uint64_t)layer_buffers->tokens*FG_NGRAM_EMBED_VALUES,payload,bytes,err);uint64_t request=fg_frame_request_id(header);if(status==FG_OK&&(!session_id||request!=session_id||peer!=work.source_rank||work.destination_rank!=self||manifest->layer_owner[work.layer]!=self||work.layer==0u||work.source_rank!=manifest->layer_owner[work.layer-1u]||work.token_count>manifest->prefill_microbatch||work.token_count>manifest->max_context||work.first_token>manifest->max_context-work.token_count||fg_frame_sequence(header)!=work.first_token*FG_LAYER_COUNT+work.layer)){fg_error_set(err,FG_ERR_MISMATCH,"stale, oversized, or misrouted prefill layer work");status=FG_ERR_MISMATCH;}uint64_t hyper_bytes=(uint64_t)work.token_count*FG_HYPER_WIDTH*4u,ngram_bytes=(uint64_t)work.token_count*FG_NGRAM_EMBED_VALUES*4u;if(status==FG_OK)status=fg_vk_tensor_write(layer_buffers->hyper_tensor,0,work.hyper,hyper_bytes,err);const fg_vk_tensor *ngram=NULL;if(status==FG_OK&&(work.flags&FG_LAYER_WORK_HAS_NGRAM)){status=fg_vk_tensor_write(layer_buffers->ngram_tensor,0,work.ngram_embeddings,ngram_bytes,err);ngram=layer_buffers->ngram_tensor;}prefill_dispatch_context dispatch={fabric,expert,manifest,self,request,fg_frame_sequence(header),expert_buffers};fg_vk_tensor *output=NULL;if(status==FG_OK)status=fg_owner_prefill_layer(owner,work.layer,work.first_token,work.positions,work.token_count,layer_buffers->hyper_tensor,ngram,dispatch_prefill_experts,&dispatch,&output,err);if(status==FG_OK)status=fg_vk_tensor_read(output,0,layer_buffers->output,hyper_bytes,err);uint32_t send_bytes=0;if(status==FG_OK&&work.layer+1u<FG_LAYER_COUNT){uint32_t next_layer=work.layer+1u,next_owner=manifest->layer_owner[next_layer];fg_prefill_layer_work next={.layer=(uint8_t)next_layer,.source_rank=(uint8_t)self,.destination_rank=(uint8_t)next_owner,.first_token=work.first_token,.token_count=work.token_count,.positions=work.positions,.hyper=layer_buffers->output};status=fg_prefill_layer_work_encode(layer_buffers->receive,layer_buffers->receive_capacity,&send_bytes,&next,err);if(status==FG_OK)status=fg_fabric_send(fabric,next_owner,FG_FABRIC_BULK,FG_MSG_PREFILL_LAYER_WORK,request,work.first_token*FG_LAYER_COUNT+next_layer,0,layer_buffers->receive,send_bytes,err);}else if(status==FG_OK){fg_prefill_layer_result result={.layer=work.layer,.source_rank=(uint8_t)self,.destination_rank=0u,.first_token=work.first_token,.token_count=work.token_count,.hyper=layer_buffers->output};status=fg_prefill_layer_result_encode(layer_buffers->result_wire,layer_buffers->result_capacity,&send_bytes,&result,err);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_PREFILL_LAYER_RESULT,request,work.first_token*FG_LAYER_COUNT+FG_LAYER_COUNT,0,layer_buffers->result_wire,send_bytes,err);}return status;
}

static fg_status handle_output_work(fg_fabric *fabric,fg_output_executor *output,uint32_t self,uint64_t session_id,uint32_t peer,const fg_frame_header *header,const uint8_t *payload,uint32_t bytes,fg_vk_tensor *hyper_tensor,fg_error *err){
    if(self!=4u||!output){fg_error_set(err,FG_ERR_MISMATCH,"output work reached a non-output rank");return FG_ERR_MISMATCH;}
    fg_output_work *work=calloc(1,sizeof(*work));if(!work){fg_error_set(err,FG_ERR_OOM,"allocate output work");return FG_ERR_OOM;}
    fg_status status=fg_output_work_decode(work,payload,bytes,err);uint64_t request=fg_frame_request_id(header);
    if(status==FG_OK&&(!session_id||request!=session_id||peer!=work->source_rank||work->destination_rank!=self)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted output work");status=FG_ERR_MISMATCH;}
    if(status==FG_OK)status=fg_vk_tensor_write(hyper_tensor,0,work->hyper,sizeof(work->hyper),err);
    fg_output_result result={.source_rank=(uint8_t)self,.destination_rank=work->source_rank,.token_index=work->token_index};
    if(status==FG_OK)status=fg_output_greedy(output,hyper_tensor,&result.token,&result.logit,err);
    uint8_t wire[FG_OUTPUT_RESULT_BYTES];if(status==FG_OK)status=fg_output_result_encode(wire,&result,err);
    if(status==FG_OK)status=fg_fabric_send(fabric,peer,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,request,fg_frame_sequence(header),0,wire,sizeof(wire),err);
    free(work);return status;
}

static fg_status rank_worker_loop(fg_fabric *fabric,fg_expert_executor *expert,fg_owner_executor *owner,fg_output_executor *output,fg_model *model,const fg_manifest *manifest,const char *directory,uint32_t self,fg_error *err){
    uint8_t *control=malloc(FG_LAYER_WORK_MAX_BYTES);prefill_worker_buffers prefill={0};prefill_layer_buffers layer_prefill={0};fg_vk_tensor *hyper=NULL,*ngram=NULL;
    /* Pre-allocate decode layer buffers — eliminates ~240 KB malloc/free per layer call */
    fg_layer_work *lw_buf=malloc(sizeof(*lw_buf));fg_layer_result *lr_buf=malloc(sizeof(*lr_buf));uint8_t *lr_wire=malloc(FG_LAYER_RESULT_BYTES);uint8_t *lw_wire=malloc(FG_LAYER_WORK_MAX_BYTES);
    /* Pre-allocate expert work buffers — eliminates ~200 KB malloc/free per expert request */
    fg_expert_result *ew_result=malloc(sizeof(*ew_result));uint8_t *ew_wire=malloc(FG_EXPERT_RESULT_MAX_BYTES);uint8_t *expert_recv=malloc(FG_EXPERT_RESULT_MAX_BYTES);
    if(!control||!lw_buf||!lr_buf||!lr_wire||!lw_wire||!ew_result||!ew_wire||!expert_recv){free(expert_recv);free(ew_wire);free(ew_result);free(lw_wire);free(lr_wire);free(lr_buf);free(lw_buf);free(control);fg_error_set(err,FG_ERR_OOM,"allocate rank worker buffers");return FG_ERR_OOM;}
    fg_status status=prefill_worker_buffers_create(&prefill,manifest->prefill_microbatch,err);if(status==FG_OK)status=prefill_layer_buffers_create(&layer_prefill,model,manifest->prefill_microbatch,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(model),FG_HYPER_WIDTH*4u,&hyper,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(model),FG_NGRAM_EMBED_VALUES*4u,&ngram,err);uint64_t session_id=0;
    while(status==FG_OK){uint32_t peer=0,bytes=0;fg_frame_header header;fg_fabric_class ready_class;status=fg_fabric_wait_ready(fabric,3u,&peer,&ready_class,err);if(status!=FG_OK)break;if(ready_class==FG_FABRIC_BULK){status=fg_fabric_recv(fabric,peer,FG_FABRIC_BULK,&header,layer_prefill.receive,layer_prefill.receive_capacity,&bytes,err);fg_message_type type=status==FG_OK?fg_frame_type(&header):0;if(status==FG_OK&&type==FG_MSG_PREFILL_WORK)status=handle_prefill_expert_work(fabric,expert,manifest,self,session_id,peer,&header,layer_prefill.receive,bytes,&prefill,err);else if(status==FG_OK&&type==FG_MSG_PREFILL_LAYER_WORK)status=handle_prefill_layer_work(fabric,expert,owner,manifest,self,session_id,peer,&header,layer_prefill.receive,bytes,&prefill,&layer_prefill,err);else if(status==FG_OK){fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported bulk message %u",self,type);status=FG_ERR_FORMAT;}continue;}status=fg_fabric_recv(fabric,peer,FG_FABRIC_CONTROL,&header,control,FG_LAYER_WORK_MAX_BYTES,&bytes,err);if(status!=FG_OK)break;fg_message_type type=fg_frame_type(&header);if(type==FG_MSG_DECODE_WORK){if(!session_id||fg_frame_request_id(&header)!=session_id){fg_error_set(err,FG_ERR_MISMATCH,"stale expert work request");status=FG_ERR_MISMATCH;}else status=handle_expert_work(fabric,expert,self,peer,&header,control,bytes,ew_result,ew_wire,err);}else if(type==FG_MSG_SESSION_BEGIN){if(bytes){fg_error_set(err,FG_ERR_FORMAT,"session begin payload must be empty");status=FG_ERR_FORMAT;}else status=begin_session(fabric,owner,directory,self,peer,&header,&session_id,err);}else if(type==FG_MSG_LAYER_WORK)status=handle_layer_work(fabric,expert,owner,manifest,self,session_id,peer,&header,control,bytes,hyper,ngram,lw_buf,lr_buf,lr_wire,lw_wire,expert_recv,err);else if(type==FG_MSG_OUTPUT_WORK)status=handle_output_work(fabric,output,self,session_id,peer,&header,control,bytes,hyper,err);else{fg_error_set(err,FG_ERR_FORMAT,"rank %u received unsupported control message %u",self,type);status=FG_ERR_FORMAT;}}
    fg_vk_tensor_destroy(ngram);fg_vk_tensor_destroy(hyper);prefill_layer_buffers_destroy(&layer_prefill);prefill_worker_buffers_destroy(&prefill);free(expert_recv);free(ew_wire);free(ew_result);free(lw_wire);free(lr_wire);free(lr_buf);free(lw_buf);free(control);return status;
}

fg_status fg_rank_main(const char *path,uint32_t rank,fg_error *err){if(rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"rank must be 0..7");return FG_ERR_ARGUMENT;}fg_manifest *manifest=NULL;fg_status status=load_checked(path,&manifest,err);char directory[1024];if(status==FG_OK)status=manifest_directory(path,directory,err);fg_model *model=NULL;fg_expert_executor *expert=NULL;fg_owner_executor *owner=NULL;fg_output_executor *output=NULL;fg_fabric *fabric=NULL;if(status==FG_OK)status=fg_model_open(&model,manifest,directory,rank,err);if(status==FG_OK)status=fg_expert_executor_create(&expert,model,err);if(status==FG_OK)status=fg_owner_executor_create(&owner,model,err);if(status==FG_OK&&rank==4u)status=fg_output_executor_create(&output,model,err);if(status==FG_OK)status=fg_fabric_open(&fabric,manifest,rank,err);if(status==FG_OK)status=rank_ready(fabric,rank,err);if(status==FG_OK){printf("rank %u READY: %.3f GiB sealed weights on %s\n",rank,(double)fg_model_weight_bytes(model)/(1024.0*1024.0*1024.0),fg_vk_device_name(fg_model_vk(model)));fflush(stdout);status=rank_worker_loop(fabric,expert,owner,output,model,manifest,directory,rank,err);}fg_fabric_close(fabric);fg_output_executor_destroy(output);fg_owner_executor_destroy(owner);fg_expert_executor_destroy(expert);fg_model_close(model);free(manifest);return status;}

typedef struct fg_coordinator {const fg_manifest *manifest;fg_model *model;fg_expert_executor *expert;fg_owner_executor *owner;fg_fabric *fabric;fg_ngram_store *ngram;fg_tokenizer *tokenizer;fg_vk_tensor *hyper;prefill_worker_buffers prefill_expert;prefill_layer_buffers prefill_layer;uint64_t session_id;uint8_t *expert_recv;uint8_t *async_recv_payloads[FG_GROUP_SIZE];const char *directory;} fg_coordinator;

static fg_status coordinator_begin_session(fg_coordinator *coordinator,fg_error *err){struct timespec now;if(clock_gettime(CLOCK_REALTIME,&now)!=0){fg_error_set(err,FG_ERR_IO,"read session clock");return FG_ERR_IO;}uint64_t request=((uint64_t)(uint32_t)now.tv_sec<<32u)^(uint32_t)now.tv_nsec^(uint64_t)(uint32_t)getpid();if(!request)request=1u;for(uint32_t peer=1;peer<FG_RANK_COUNT;peer++){fg_status status=fg_fabric_send(coordinator->fabric,peer,FG_FABRIC_CONTROL,FG_MSG_SESSION_BEGIN,request,0,0,NULL,0,err);if(status!=FG_OK)return status;}bool ready[FG_RANK_COUNT]={0};for(uint32_t received=1;received<FG_RANK_COUNT;received++){uint32_t peer=0,bytes=0;fg_frame_header header;fg_status status=fg_fabric_recv_any(coordinator->fabric,FG_FABRIC_CONTROL,&peer,&header,NULL,0,&bytes,err);if(status!=FG_OK)return status;if(fg_frame_type(&header)!=FG_MSG_SESSION_READY||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=0u||bytes||peer==0u||ready[peer]){fg_error_set(err,FG_ERR_MISMATCH,"invalid session readiness from rank %u",peer);return FG_ERR_MISMATCH;}ready[peer]=true;}coordinator->session_id=request;return FG_OK;}


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
        const fg_vk_tensor *layer_ngram=(layer==1u)?ngram_embeddings:NULL;
        prefill_dispatch_context dispatch={coordinator->fabric,coordinator->expert,coordinator->manifest,0u,coordinator->session_id,first_token*FG_LAYER_COUNT+layer,&coordinator->prefill_expert};
        fg_vk_tensor *layer_out=NULL;
        status=fg_owner_prefill_layer(coordinator->owner,layer,first_token,buffers->positions,token_count,current,layer_ngram,dispatch_prefill_experts,&dispatch,&layer_out,err);
        if(status==FG_OK){current=layer_out;}
    }
    if(status==FG_OK){*output=current;}
    return status;
}

static fg_status coordinator_output(fg_coordinator *coordinator,uint32_t token_index,const fg_vk_tensor *hyper,uint32_t *next_token,float *logit,fg_error *err){fg_output_work *work=calloc(1,sizeof(*work));uint8_t *wire=malloc(FG_OUTPUT_WORK_BYTES);if(!work||!wire){free(wire);free(work);fg_error_set(err,FG_ERR_OOM,"allocate coordinator output exchange");return FG_ERR_OOM;}work->source_rank=0u;work->destination_rank=4u;work->token_index=token_index;fg_status status=fg_vk_tensor_read(hyper,0,work->hyper,sizeof(work->hyper),err);if(status==FG_OK)status=fg_output_work_encode(wire,work,err);uint32_t sequence=token_index*FG_LAYER_COUNT+FG_LAYER_COUNT;if(status==FG_OK)status=fg_fabric_send(coordinator->fabric,4u,FG_FABRIC_CONTROL,FG_MSG_OUTPUT_WORK,coordinator->session_id,sequence,0,wire,FG_OUTPUT_WORK_BYTES,err);if(status==FG_OK){fg_frame_header header;uint32_t bytes=0;status=fg_fabric_recv(coordinator->fabric,4u,FG_FABRIC_BULK,&header,wire,FG_OUTPUT_RESULT_BYTES,&bytes,err);if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_RESULT||fg_frame_request_id(&header)!=coordinator->session_id||fg_frame_sequence(&header)!=sequence)){fg_error_set(err,FG_ERR_MISMATCH,"stale output result");status=FG_ERR_MISMATCH;}fg_output_result result;if(status==FG_OK)status=fg_output_result_decode(&result,wire,bytes,err);if(status==FG_OK&&(result.source_rank!=4u||result.destination_rank!=0u||result.token_index!=token_index)){fg_error_set(err,FG_ERR_MISMATCH,"misrouted output result");status=FG_ERR_MISMATCH;}if(status==FG_OK){*next_token=result.token;if(logit)*logit=result.logit;}}free(wire);free(work);return status;}

/* Expert-parallel decode: all 48 layers on the coordinator, MoE dispatched to workers */
static fg_status coordinator_decode_token_local(fg_coordinator *coordinator,const int32_t *history,size_t history_count,uint32_t token_index,uint32_t *next_token,float *logit,fg_error *err){
    if(!history||!history_count||(uint32_t)history[history_count-1u]>=FG_Q38_VOCAB_SIZE){fg_error_set(err,FG_ERR_ARGUMENT,"invalid local decode token history");return FG_ERR_ARGUMENT;}
    fg_vk_tensor *embedding=fg_model_tensor(coordinator->model,"token_embd.weight");
    fg_status status=fg_vk_embedding_q8_0(fg_model_vk(coordinator->model),coordinator->hyper,embedding,(uint32_t)history[history_count-1u],FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,err);
    fg_vk_tensor *ngram=NULL;if(status==FG_OK)status=fg_ngram_store_lookup(coordinator->ngram,history,history_count,&ngram,err);
    uint32_t position[3]={token_index,token_index,token_index};
    fg_vk_tensor *current=coordinator->hyper;
    /* Process all 48 layers locally — sync dispatch (poll-based recv is faster than async for serial decode) */
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){
        const fg_vk_tensor *layer_ngram=(layer==1u)?ngram:NULL;
        expert_dispatch_context dispatch={coordinator->fabric,coordinator->expert,coordinator->manifest,0u,coordinator->session_id,token_index*FG_LAYER_COUNT+layer,coordinator->expert_recv};
        fg_vk_tensor *layer_out=NULL;
        status=fg_owner_decode_layer(coordinator->owner,layer,token_index,position,current,layer_ngram,dispatch_experts,&dispatch,&layer_out,err);
        if(status==FG_OK)current=layer_out;
    }
    if(status==FG_OK)status=coordinator_output(coordinator,token_index,current,next_token,logit,err);
    return status;
}

static void coordinator_close(fg_coordinator *coordinator){if(!coordinator)return;for(uint32_t i=0;i<FG_GROUP_SIZE;i++)free(coordinator->async_recv_payloads[i]);free(coordinator->expert_recv);prefill_layer_buffers_destroy(&coordinator->prefill_layer);prefill_worker_buffers_destroy(&coordinator->prefill_expert);fg_vk_tensor_destroy(coordinator->hyper);fg_ngram_store_close(coordinator->ngram);fg_tokenizer_close(coordinator->tokenizer);fg_fabric_close(coordinator->fabric);fg_owner_executor_destroy(coordinator->owner);fg_expert_executor_destroy(coordinator->expert);fg_model_close(coordinator->model);memset(coordinator,0,sizeof(*coordinator));}

static fg_status coordinator_open(fg_coordinator *coordinator,const fg_manifest *manifest,const char *directory,fg_error *err){memset(coordinator,0,sizeof(*coordinator));coordinator->manifest=manifest;fg_status status=fg_model_open_replicated(&coordinator->model,manifest,directory,0u,err);if(status==FG_OK)status=fg_expert_executor_create(&coordinator->expert,coordinator->model,err);if(status==FG_OK)status=fg_owner_executor_create(&coordinator->owner,coordinator->model,err);if(status==FG_OK)status=fg_tokenizer_open(&coordinator->tokenizer,directory,manifest,err);if(status==FG_OK)status=fg_tokenizer_validate_qwen38(coordinator->tokenizer,err);const fg_tensor_record *ngram_record=NULL;for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)if(manifest->tensors[i].kind==FG_TENSOR_NGRAM){if(ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"multiple n-gram tensors in deployment manifest");status=FG_ERR_MISMATCH;}else ngram_record=&manifest->tensors[i];}char ngram_path[1200];if(status==FG_OK&&!ngram_record){fg_error_set(err,FG_ERR_MISMATCH,"deployment manifest has no n-gram tensor");status=FG_ERR_MISMATCH;}if(status==FG_OK&&snprintf(ngram_path,sizeof(ngram_path),"%s/ngram.iq4nl",directory)>=(int)sizeof(ngram_path)){fg_error_set(err,FG_ERR_LIMIT,"n-gram path is too long");status=FG_ERR_LIMIT;}if(status==FG_OK)status=fg_ngram_store_open(&coordinator->ngram,fg_model_vk(coordinator->model),ngram_path,ngram_record->bytes,err);if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(coordinator->model),FG_HYPER_WIDTH*sizeof(float),&coordinator->hyper,err);if(status==FG_OK){coordinator->expert_recv=malloc(FG_EXPERT_RESULT_MAX_BYTES);if(!coordinator->expert_recv){fg_error_set(err,FG_ERR_OOM,"allocate coordinator expert recv buffer");status=FG_ERR_OOM;}}
    /* Allocate per-group async recv payload buffers */
    for(uint32_t i=0;status==FG_OK&&i<FG_GROUP_SIZE;i++){coordinator->async_recv_payloads[i]=malloc(FG_EXPERT_RESULT_MAX_BYTES);if(!coordinator->async_recv_payloads[i]){fg_error_set(err,FG_ERR_OOM,"allocate async expert recv buffer %u",i);status=FG_ERR_OOM;}}if(status==FG_OK)status=prefill_worker_buffers_create(&coordinator->prefill_expert,manifest->prefill_microbatch,err);if(status==FG_OK)status=prefill_layer_buffers_create(&coordinator->prefill_layer,coordinator->model,manifest->prefill_microbatch,err);if(status==FG_OK)status=fg_fabric_open(&coordinator->fabric,manifest,0u,err);if(status==FG_OK)status=rank_ready(coordinator->fabric,0u,err);if(status==FG_OK)status=coordinator_begin_session(coordinator,err);if(status!=FG_OK)coordinator_close(coordinator);coordinator->directory=directory;return status;}
fg_status fg_serve_main(const char *path,fg_error *err){fg_manifest *m=NULL;fg_status rc=load_checked(path,&m,err);if(rc==FG_OK){fg_manifest_print(m);fg_error_set(err,FG_ERR_UNAVAILABLE,"HTTP serving is not enabled until the owned request path is qualified");rc=FG_ERR_UNAVAILABLE;}free(m);return rc;}
fg_status fg_bench_main(const char *path,fg_error *err){fg_manifest *m=NULL;fg_status rc=load_checked(path,&m,err);if(rc==FG_OK){fg_manifest_print(m);puts("qualification matrix: microbatch={128,256,512} window={1,2,3,4}; canonical prompt=32768 tokens; repetitions=3");}free(m);return rc;}
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
    free(history);fg_tokens_free(&prompt_tokens);if(manifest)coordinator_close(&coordinator);free(manifest);free(wrapped);return status;
}
