#include "fg_pipeline.h"
#include "fg_topology.h"

#include <arpa/inet.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct pipeline_slot {
    uint32_t *positions;
    float *boundary;
    fg_pipeline_activation activation;
    uint32_t sequence;
    uint8_t upstream_slot;
    bool occupied;
    bool ready;
    bool has_upstream;
} pipeline_slot;

typedef struct pipeline_batch {
    uint32_t sequence;
    uint32_t first_token;
    uint32_t admitted_frontier;
    uint16_t token_count;
    bool request_output;
    fg_pipeline_result result;
    bool active;
    bool complete;
} pipeline_batch;

struct fg_pipeline {
    const fg_manifest *manifest;
    fg_pipeline_transport transport;
    fg_pipeline_execute_fn execute;
    void *execute_context;
    pipeline_slot *slots;
    pipeline_batch *batches;
    uint8_t *wire;
    uint32_t wire_capacity;
    uint32_t rank;
    uint32_t stage;
    uint32_t slot_count;
    uint32_t batch_capacity;
    uint32_t next_submit_sequence;
    uint32_t next_receive_sequence;
    uint32_t next_credit_sequence;
    uint32_t next_result_sequence;
    uint32_t next_take_sequence;
    uint32_t drain_sequence;
    uint32_t active_batches;
    uint32_t admission_frontier;
    uint32_t publication_cursor;
    uint32_t published_frontier;
    uint32_t operation_sequence;
    uint32_t abort_sequence;
    uint64_t request_id;
    pthread_t io_owner;
    fg_status poison_status;
    fg_error poison_error;
    bool io_owner_set;
    bool begun;
    bool admission_started;
    bool drain_sent;
    bool drain_received;
    bool drain_forwarded;
    bool drained_received;
    bool chain_drained;
    bool aborted;
    bool abort_complete;
    bool abort_propagated;
    bool abort_discarded;
    uint8_t abort_origin_stage;
    uint16_t abort_required_mask;
    uint16_t abort_notified_mask;
    bool *outbound_available;
    bool *outbound_active;
    uint32_t *outbound_sequence;
};

static uint32_t stage_rank(const fg_pipeline *pipeline,uint32_t stage){
    return pipeline->manifest->stage_ranks[stage];
}

static void cancel_pipeline_ledgers(fg_pipeline *pipeline){
    for(uint32_t i=0;i<pipeline->slot_count;i++){
        pipeline->slots[i].occupied=false;
        pipeline->slots[i].ready=false;
        pipeline->slots[i].has_upstream=false;
        pipeline->outbound_available[i]=false;
        pipeline->outbound_active[i]=false;
        pipeline->outbound_sequence[i]=0u;
    }
    uint32_t unread=0u;
    for(uint32_t i=0;i<pipeline->batch_capacity;i++){
        pipeline_batch *batch=&pipeline->batches[i];
        if(!pipeline->stage&&batch->active&&batch->complete)unread++;
        else memset(batch,0,sizeof(*batch));
    }
    pipeline->active_batches=unread;
}

static void propagate_abort_best_effort(fg_pipeline *pipeline,uint32_t incoming_peer){
    if(pipeline->abort_propagated||!pipeline->request_id)return;
    pipeline->abort_propagated=true;
    if(pipeline->stage)
        pipeline->abort_required_mask|=(uint16_t)(1u<<(pipeline->stage-1u));
    if(pipeline->stage+1u<pipeline->manifest->stage_count)
        pipeline->abort_required_mask|=(uint16_t)(1u<<(pipeline->stage+1u));
    fg_pipeline_abort abort={.origin_stage=pipeline->abort_origin_stage,
        .status=pipeline->poison_status,.failing_sequence=pipeline->abort_sequence};
    uint8_t payload[FG_PIPELINE_ABORT_BYTES];fg_error ignored={0};
    if(pipeline->stage&&stage_rank(pipeline,pipeline->stage-1u)==incoming_peer)
        pipeline->abort_notified_mask|=(uint16_t)(1u<<(pipeline->stage-1u));
    if(pipeline->stage+1u<pipeline->manifest->stage_count&&
       stage_rank(pipeline,pipeline->stage+1u)==incoming_peer)
        pipeline->abort_notified_mask|=(uint16_t)(1u<<(pipeline->stage+1u));
    if(fg_pipeline_abort_encode(payload,&abort,&ignored)!=FG_OK)return;
    if(pipeline->stage){
        uint32_t peer=stage_rank(pipeline,pipeline->stage-1u);
        if(peer!=incoming_peer&&pipeline->transport.send(
                pipeline->transport.context,peer,
                FG_MSG_PIPELINE_ABORT,pipeline->request_id,pipeline->abort_sequence,
                payload,sizeof(payload),&ignored)==FG_OK)
            pipeline->abort_notified_mask|=(uint16_t)(1u<<(pipeline->stage-1u));
    }
    if(pipeline->stage+1u<pipeline->manifest->stage_count){
        uint32_t peer=stage_rank(pipeline,pipeline->stage+1u);
        if(peer!=incoming_peer&&pipeline->transport.send(
                pipeline->transport.context,peer,
                FG_MSG_PIPELINE_ABORT,pipeline->request_id,pipeline->abort_sequence,
                payload,sizeof(payload),&ignored)==FG_OK)
            pipeline->abort_notified_mask|=(uint16_t)(1u<<(pipeline->stage+1u));
    }
    pipeline->abort_complete=
        (pipeline->abort_notified_mask&pipeline->abort_required_mask)==
        pipeline->abort_required_mask;
}

static fg_status fabric_pipeline_send(void *context,uint32_t peer,fg_message_type type,
                                      uint64_t request_id,uint32_t sequence,
                                      const void *payload,uint32_t bytes,fg_error *err){
    return fg_fabric_send(context,peer,FG_FABRIC_BULK,type,request_id,sequence,0u,
                          payload,bytes,err);
}

static fg_status fabric_pipeline_receive(void *context,uint32_t *peer,
                                         fg_frame_header *header,void *payload,
                                         uint32_t capacity,uint32_t *bytes,
                                         fg_error *err){
    return fg_fabric_recv_any(context,FG_FABRIC_BULK,peer,header,payload,capacity,
                              bytes,err);
}

static fg_status fabric_pipeline_receive_peer(
    void *context,uint32_t peer,fg_frame_header *header,void *payload,
    uint32_t capacity,uint32_t *bytes,fg_error *err){
    return fg_fabric_recv(context,peer,FG_FABRIC_BULK,header,payload,capacity,
                          bytes,err);
}

void fg_pipeline_transport_init_fabric(fg_pipeline_transport *transport,
                                       fg_fabric *fabric){
    if(!transport)return;
    *transport=(fg_pipeline_transport){.context=fabric,.send=fabric_pipeline_send,
        .receive=fabric_pipeline_receive,
        .receive_peer=fabric_pipeline_receive_peer};
}

static fg_status pipeline_poison(fg_pipeline *pipeline,fg_status status,
                                 const fg_error *cause,fg_error *err){
    if(pipeline->poison_status==FG_OK){
        pipeline->poison_status=status==FG_OK?FG_ERR_MISMATCH:status;
        if(cause)pipeline->poison_error=*cause;
        else fg_error_set(&pipeline->poison_error,pipeline->poison_status,
                          "pipeline scheduler poisoned");
        pipeline->poison_error.code=pipeline->poison_status;
    }
    if(pipeline->begun&&!pipeline->aborted){
        pipeline->aborted=true;
        pipeline->abort_origin_stage=(uint8_t)pipeline->stage;
        pipeline->abort_sequence=pipeline->operation_sequence;
        cancel_pipeline_ledgers(pipeline);
        propagate_abort_best_effort(pipeline,UINT32_MAX);
    }
    if(err)*err=pipeline->poison_error;
    return pipeline->poison_status;
}

static fg_status pipeline_owner(fg_pipeline *pipeline,fg_error *err){
    if(!pipeline){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline scheduler is null");
        return FG_ERR_ARGUMENT;
    }
    if(pipeline->poison_status!=FG_OK){
        if(err)*err=pipeline->poison_error;
        return pipeline->poison_status;
    }
    pthread_t self=pthread_self();
    if(!pipeline->io_owner_set){
        pipeline->io_owner=self;
        pipeline->io_owner_set=true;
    }else if(!pthread_equal(pipeline->io_owner,self)){
        fg_error local={0};
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline fabric I/O must remain on one owner thread");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    return FG_OK;
}

static fg_status pipeline_send(fg_pipeline *pipeline,uint32_t peer,
                               fg_message_type type,uint32_t sequence,
                               const void *payload,uint32_t bytes,fg_error *err){
    fg_error local={0};
    fg_status status=pipeline->transport.send(pipeline->transport.context,peer,type,
        pipeline->request_id,sequence,payload,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    return FG_OK;
}

static pipeline_batch *find_batch(fg_pipeline *pipeline,uint32_t sequence){
    for(uint32_t i=0;i<pipeline->batch_capacity;i++)
        if(pipeline->batches[i].active&&pipeline->batches[i].sequence==sequence)
            return &pipeline->batches[i];
    return NULL;
}

static pipeline_batch *free_batch(fg_pipeline *pipeline){
    for(uint32_t i=0;i<pipeline->batch_capacity;i++)
        if(!pipeline->batches[i].active)return &pipeline->batches[i];
    return NULL;
}

static pipeline_slot *oldest_ready_slot(fg_pipeline *pipeline){
    pipeline_slot *oldest=NULL;
    for(uint32_t i=0;i<pipeline->slot_count;i++){
        pipeline_slot *slot=&pipeline->slots[i];
        if(slot->occupied&&slot->ready&&(!oldest||slot->sequence<oldest->sequence))
            oldest=slot;
    }
    return oldest;
}

static pipeline_slot *free_local_slot(fg_pipeline *pipeline){
    for(uint32_t i=0;i<pipeline->slot_count;i++)
        if(!pipeline->slots[i].occupied)return &pipeline->slots[i];
    return NULL;
}

static int available_outbound_slot(const fg_pipeline *pipeline){
    for(uint32_t i=0;i<pipeline->slot_count;i++)
        if(pipeline->outbound_available[i])return (int)i;
    return -1;
}

static fg_status execute_slot(fg_pipeline *pipeline,pipeline_slot *slot,
                              fg_pipeline_result *terminal,fg_error *err){
    fg_pipeline_activation before=slot->activation;
    fg_error local={0};
    struct timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
    /* Deliberately synchronous: stages overlap chunks, not receive and Vulkan compute. */
    fg_status status=pipeline->execute(pipeline->execute_context,pipeline->stage,
        pipeline->request_id,slot->sequence,&slot->activation,slot->boundary,
        terminal,&local);
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(slot->activation.execution_kind!=before.execution_kind||
       slot->activation.slot!=before.slot||
       slot->activation.source_stage!=before.source_stage||
       slot->activation.destination_stage!=before.destination_stage||
       slot->activation.first_token!=before.first_token||
       slot->activation.token_count!=before.token_count||
       slot->activation.request_output!=before.request_output||
       slot->activation.positions!=before.positions||
       slot->activation.boundary!=before.boundary){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline execution callback changed activation identity");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    slot->activation.stage_seconds[pipeline->stage]+=
        (float)((double)(end.tv_sec-start.tv_sec)+
                (double)(end.tv_nsec-start.tv_nsec)*1e-9);
    status=fg_pipeline_activation_validate(&slot->activation,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    return FG_OK;
}

static fg_status send_credit(fg_pipeline *pipeline,uint8_t upstream_slot,
                             uint32_t sequence,fg_error *err){
    fg_pipeline_credit credit={.source_stage=(uint8_t)pipeline->stage,
        .destination_stage=(uint8_t)(pipeline->stage-1u),.slot=upstream_slot};
    uint8_t payload[FG_PIPELINE_CREDIT_BYTES];
    fg_error local={0};
    fg_status status=fg_pipeline_credit_encode(payload,&credit,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    return pipeline_send(pipeline,stage_rank(pipeline,pipeline->stage-1u),
        FG_MSG_PIPELINE_CREDIT,sequence,payload,sizeof(payload),err);
}

static fg_status release_local_slot(fg_pipeline *pipeline,pipeline_slot *slot,
                                    fg_error *err){
    bool return_credit=slot->has_upstream;
    uint8_t upstream_slot=slot->upstream_slot;
    uint32_t sequence=slot->sequence;
    slot->occupied=false;
    slot->ready=false;
    slot->has_upstream=false;
    if(return_credit)return send_credit(pipeline,upstream_slot,sequence,err);
    return FG_OK;
}

static fg_status send_activation(fg_pipeline *pipeline,pipeline_slot *slot,
                                 uint32_t outbound_slot,fg_error *err){
    slot->activation.slot=(uint8_t)outbound_slot;
    slot->activation.source_stage=(uint8_t)pipeline->stage;
    slot->activation.destination_stage=(uint8_t)(pipeline->stage+1u);
    uint32_t bytes=0;
    fg_error local={0};
    fg_status status=fg_pipeline_activation_encode(pipeline->wire,
        pipeline->wire_capacity,&bytes,&slot->activation,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    status=pipeline_send(pipeline,stage_rank(pipeline,pipeline->stage+1u),
        FG_MSG_PIPELINE_ACTIVATION,slot->sequence,pipeline->wire,bytes,err);
    if(status!=FG_OK)return status;
    pipeline->outbound_available[outbound_slot]=false;
    pipeline->outbound_active[outbound_slot]=true;
    pipeline->outbound_sequence[outbound_slot]=slot->sequence;
    return release_local_slot(pipeline,slot,err);
}

static fg_status forward_drain(fg_pipeline *pipeline,fg_error *err);

static fg_status flush_ready(fg_pipeline *pipeline,fg_error *err){
    while(pipeline->stage+1u<pipeline->manifest->stage_count){
        pipeline_slot *slot=oldest_ready_slot(pipeline);
        int outbound=available_outbound_slot(pipeline);
        if(!slot||outbound<0)break;
        fg_status status=send_activation(pipeline,slot,(uint32_t)outbound,err);
        if(status!=FG_OK)return status;
    }
    if(pipeline->drain_received&&!pipeline->drain_forwarded&&
       !oldest_ready_slot(pipeline))
        return forward_drain(pipeline,err);
    return FG_OK;
}

static fg_status send_terminal_result(fg_pipeline *pipeline,pipeline_slot *slot,
                                      fg_error *err){
    fg_pipeline_result result={.completed_first_token=slot->activation.first_token,
        .completed_token_count=slot->activation.token_count,
        .completed_frontier=slot->activation.first_token+slot->activation.token_count,
        .has_output=slot->activation.request_output,
        .final_token=FG_Q38_VOCAB_SIZE,
        .final_logit=slot->activation.request_output?NAN:0.0f};
    fg_status status=execute_slot(pipeline,slot,&result,err);
    if(status!=FG_OK)return status;
    memcpy(result.stage_seconds,slot->activation.stage_seconds,
           sizeof(result.stage_seconds));
    if(result.completed_first_token!=slot->activation.first_token||
       result.completed_token_count!=slot->activation.token_count||
       result.completed_frontier!=
           slot->activation.first_token+slot->activation.token_count||
       result.has_output!=slot->activation.request_output){
        fg_error local={0};
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "terminal callback changed completion identity");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    uint8_t payload[FG_PIPELINE_RESULT_BYTES];
    fg_error local={0};
    status=fg_pipeline_result_encode(payload,&result,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    status=pipeline_send(pipeline,stage_rank(pipeline,0u),FG_MSG_PIPELINE_RESULT,
                         slot->sequence,payload,sizeof(payload),err);
    if(status!=FG_OK)return status;
    return release_local_slot(pipeline,slot,err);
}

static fg_status receive_activation(fg_pipeline *pipeline,uint32_t peer,
                                    const fg_frame_header *header,uint32_t bytes,
                                    fg_error *err){
    fg_error local={0};
    if(pipeline->drain_received||pipeline->drain_forwarded||pipeline->chain_drained){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline activation arrived after terminal drain state");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    fg_status status=fg_pipeline_frame_validate_sequence(header,
        FG_MSG_PIPELINE_ACTIVATION,pipeline->request_id,
        pipeline->next_receive_sequence,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(peer!=stage_rank(pipeline,pipeline->stage-1u)){
        fg_error_set(&local,FG_ERR_MISMATCH,"pipeline activation arrived from wrong peer");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    if(bytes<FG_PIPELINE_ACTIVATION_HEADER_BYTES){
        fg_error_set(&local,FG_ERR_FORMAT,"pipeline activation header is truncated");
        return pipeline_poison(pipeline,FG_ERR_FORMAT,&local,err);
    }
    uint8_t slot_index=pipeline->wire[1];
    pipeline_slot *slot=free_local_slot(pipeline);
    if(slot_index>=pipeline->slot_count||!slot){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline activation has no available receiver slot");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    status=fg_pipeline_activation_decode(&slot->activation,slot->positions,
        pipeline->manifest->prefill_microbatch*FG_PIPELINE_POSITION_AXES,
        slot->boundary,
        (uint64_t)pipeline->manifest->prefill_microbatch*FG_PIPELINE_BOUNDARY_WIDTH,
        pipeline->wire,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(slot->activation.source_stage+1u!=pipeline->stage||
       slot->activation.destination_stage!=pipeline->stage||
       slot->activation.token_count>pipeline->manifest->prefill_microbatch){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline activation does not match local stage geometry");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    slot->sequence=pipeline->next_receive_sequence++;
    slot->upstream_slot=slot_index;
    slot->has_upstream=true;
    slot->occupied=true;
    if(pipeline->stage+1u==pipeline->manifest->stage_count)
        return send_terminal_result(pipeline,slot,err);
    status=execute_slot(pipeline,slot,NULL,err);
    if(status!=FG_OK)return status;
    slot->ready=true;
    return flush_ready(pipeline,err);
}

static fg_status receive_credit(fg_pipeline *pipeline,uint32_t peer,
                                const fg_frame_header *header,uint32_t bytes,
                                fg_error *err){
    fg_pipeline_credit credit={0};
    fg_error local={0};
    fg_status status=fg_pipeline_credit_decode(&credit,pipeline->wire,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    status=fg_pipeline_frame_validate_sequence(header,FG_MSG_PIPELINE_CREDIT,
        pipeline->request_id,pipeline->next_credit_sequence,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(pipeline->stage+1u>=pipeline->manifest->stage_count||
       peer!=stage_rank(pipeline,pipeline->stage+1u)||
       credit.source_stage!=pipeline->stage+1u||
       credit.destination_stage!=pipeline->stage||
       credit.slot>=pipeline->slot_count||
       !pipeline->outbound_active[credit.slot]||
       pipeline->outbound_sequence[credit.slot]!=pipeline->next_credit_sequence){
        fg_error_set(&local,FG_ERR_MISMATCH,"stale or misrouted pipeline credit");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    pipeline->outbound_active[credit.slot]=false;
    pipeline->outbound_available[credit.slot]=true;
    pipeline->next_credit_sequence++;
    return flush_ready(pipeline,err);
}

static fg_status receive_result(fg_pipeline *pipeline,uint32_t peer,
                                const fg_frame_header *header,uint32_t bytes,
                                fg_error *err){
    fg_error local={0};
    fg_status status=fg_pipeline_frame_validate_sequence(header,FG_MSG_PIPELINE_RESULT,
        pipeline->request_id,pipeline->next_result_sequence,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(peer!=stage_rank(pipeline,pipeline->manifest->stage_count-1u)){
        fg_error_set(&local,FG_ERR_MISMATCH,"pipeline result arrived from wrong peer");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    fg_pipeline_result result={0};
    status=fg_pipeline_result_decode(&result,pipeline->wire,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    pipeline_batch *batch=find_batch(pipeline,pipeline->next_result_sequence);
    if(!batch||batch->complete||
       result.completed_first_token!=batch->first_token||
       result.completed_token_count!=batch->token_count||
       result.completed_frontier!=batch->admitted_frontier||
       result.has_output!=batch->request_output||
       result.completed_first_token!=pipeline->publication_cursor||
       result.completed_frontier<=pipeline->publication_cursor){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline result regresses or skips the admitted frontier");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    batch->result=result;
    batch->complete=true;
    pipeline->publication_cursor=result.completed_frontier;
    pipeline->published_frontier=result.completed_frontier;
    pipeline->next_result_sequence++;
    if(pipeline->drained_received&&
       pipeline->next_result_sequence==pipeline->drain_sequence)
        pipeline->chain_drained=true;
    return FG_OK;
}

static fg_status forward_drain(fg_pipeline *pipeline,fg_error *err){
    fg_error local={0};
    if(pipeline->stage+1u<pipeline->manifest->stage_count){
        fg_pipeline_drain drain={.source_stage=(uint8_t)pipeline->stage,
            .destination_stage=(uint8_t)(pipeline->stage+1u)};
        uint8_t payload[FG_PIPELINE_DRAIN_BYTES];
        fg_status status=fg_pipeline_drain_encode(payload,&drain,&local);
        if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
        status=pipeline_send(pipeline,stage_rank(pipeline,pipeline->stage+1u),
            FG_MSG_PIPELINE_DRAIN,pipeline->drain_sequence,payload,sizeof(payload),err);
        if(status!=FG_OK)return status;
    }else{
        fg_pipeline_drained drained={.source_stage=(uint8_t)pipeline->stage,
            .destination_stage=(uint8_t)(pipeline->stage-1u)};
        uint8_t payload[FG_PIPELINE_DRAINED_BYTES];
        fg_status status=fg_pipeline_drained_encode(payload,&drained,&local);
        if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
        status=pipeline_send(pipeline,stage_rank(pipeline,pipeline->stage-1u),
            FG_MSG_PIPELINE_DRAINED,pipeline->drain_sequence,payload,sizeof(payload),err);
        if(status!=FG_OK)return status;
        pipeline->chain_drained=true;
    }
    pipeline->drain_forwarded=true;
    return FG_OK;
}

static fg_status receive_drain(fg_pipeline *pipeline,uint32_t peer,
                               const fg_frame_header *header,uint32_t bytes,
                               fg_error *err){
    fg_error local={0};
    fg_status status=fg_pipeline_frame_validate_sequence(header,FG_MSG_PIPELINE_DRAIN,
        pipeline->request_id,pipeline->next_receive_sequence,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    fg_pipeline_drain drain={0};
    status=fg_pipeline_drain_decode(&drain,pipeline->wire,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(!pipeline->stage||peer!=stage_rank(pipeline,pipeline->stage-1u)||
       drain.source_stage+1u!=pipeline->stage||
       drain.destination_stage!=pipeline->stage||pipeline->drain_received){
        fg_error_set(&local,FG_ERR_MISMATCH,"stale or misrouted pipeline drain");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    pipeline->drain_received=true;
    pipeline->drain_sequence=pipeline->next_receive_sequence;
    return flush_ready(pipeline,err);
}

static fg_status receive_drained(fg_pipeline *pipeline,uint32_t peer,
                                 const fg_frame_header *header,uint32_t bytes,
                                 fg_error *err){
    fg_error local={0};
    fg_status status=fg_pipeline_frame_validate_sequence(header,FG_MSG_PIPELINE_DRAINED,
        pipeline->request_id,pipeline->drain_sequence,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    fg_pipeline_drained drained={0};
    status=fg_pipeline_drained_decode(&drained,pipeline->wire,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(pipeline->stage+1u>=pipeline->manifest->stage_count||
       peer!=stage_rank(pipeline,pipeline->stage+1u)||
       drained.source_stage!=pipeline->stage+1u||
       drained.destination_stage!=pipeline->stage||!pipeline->drain_forwarded){
        fg_error_set(&local,FG_ERR_MISMATCH,"stale or misrouted pipeline drained");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    if(pipeline->stage){
        fg_pipeline_drained upstream={.source_stage=(uint8_t)pipeline->stage,
            .destination_stage=(uint8_t)(pipeline->stage-1u)};
        uint8_t payload[FG_PIPELINE_DRAINED_BYTES];
        status=fg_pipeline_drained_encode(payload,&upstream,&local);
        if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
        status=pipeline_send(pipeline,stage_rank(pipeline,pipeline->stage-1u),
            FG_MSG_PIPELINE_DRAINED,pipeline->drain_sequence,payload,sizeof(payload),err);
        if(status!=FG_OK)return status;
    }
    pipeline->drained_received=true;
    pipeline->chain_drained=pipeline->stage||
        pipeline->next_result_sequence==pipeline->drain_sequence;
    return FG_OK;
}

static fg_status receive_abort(fg_pipeline *pipeline,uint32_t peer,
                               const fg_frame_header *header,uint32_t bytes,
                               fg_error *err){
    fg_error local={0};fg_pipeline_abort abort={0};
    fg_status status=fg_pipeline_abort_decode(&abort,pipeline->wire,bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    status=fg_pipeline_frame_validate_sequence(header,FG_MSG_PIPELINE_ABORT,
        pipeline->request_id,abort.failing_sequence,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    bool from_previous=pipeline->stage&&
        peer==stage_rank(pipeline,pipeline->stage-1u);
    bool from_next=pipeline->stage+1u<pipeline->manifest->stage_count&&
        peer==stage_rank(pipeline,pipeline->stage+1u);
    if((!from_previous&&!from_next)||
       (from_previous&&abort.origin_stage>=pipeline->stage)||
       (from_next&&abort.origin_stage<=pipeline->stage)){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline abort origin does not match its adjacent path");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    pipeline->aborted=true;
    pipeline->abort_origin_stage=abort.origin_stage;
    pipeline->abort_sequence=abort.failing_sequence;
    pipeline->poison_status=abort.status;
    fg_error_set(&pipeline->poison_error,abort.status,
                 "pipeline request aborted by stage %u at sequence %u",
                 abort.origin_stage,abort.failing_sequence);
    cancel_pipeline_ledgers(pipeline);
    propagate_abort_best_effort(pipeline,peer);
    if(err)*err=pipeline->poison_error;
    return pipeline->poison_status;
}

fg_status fg_pipeline_create(fg_pipeline **out,const fg_pipeline_config *config,
                             fg_error *err){
    if(!out||!config||!config->manifest||!config->transport.send||
       !config->transport.receive||!config->transport.receive_peer||
       !config->execute){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline scheduler configuration");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    const fg_manifest *manifest=config->manifest;
    fg_status status=fg_topology_validate(manifest,err);
    if(status!=FG_OK)return status;
    if(manifest->execution_mode!=FG_EXECUTION_PIPELINE||
       manifest->protocol_version!=FG_PIPELINE_PROTOCOL_VERSION||
       !manifest->slot_count||manifest->slot_count>FG_PIPELINE_DEFAULT_SLOT_COUNT||
       !manifest->prefill_microbatch||
       manifest->prefill_microbatch>FG_PREFILL_MAX_TOKENS||
       manifest->native_context!=FG_NATIVE_CONTEXT||
       config->rank>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "manifest does not define a supported protocol 7 pipeline");
        return FG_ERR_MISMATCH;
    }
    uint32_t stage=manifest->stage_count;
    for(uint32_t i=0;i<manifest->stage_count;i++)
        if(manifest->stage_ranks[i]==config->rank){stage=i;break;}
    if(stage==manifest->stage_count){
        fg_error_set(err,FG_ERR_MISMATCH,"rank is not assigned to a pipeline stage");
        return FG_ERR_MISMATCH;
    }
    fg_pipeline *pipeline=calloc(1,sizeof(*pipeline));
    if(!pipeline){
        fg_error_set(err,FG_ERR_OOM,"allocate pipeline scheduler");
        return FG_ERR_OOM;
    }
    pipeline->manifest=manifest;
    pipeline->transport=config->transport;
    pipeline->execute=config->execute;
    pipeline->execute_context=config->execute_context;
    pipeline->rank=config->rank;
    pipeline->stage=stage;
    pipeline->slot_count=manifest->slot_count;
    if(manifest->stage_count>UINT32_MAX/pipeline->slot_count){
        fg_error_set(err,FG_ERR_LIMIT,
                     "pipeline in-flight ledger capacity overflows");
        free(pipeline);return FG_ERR_LIMIT;
    }
    pipeline->batch_capacity=manifest->stage_count*pipeline->slot_count;
    if(!pipeline->batch_capacity||
       pipeline->batch_capacity>FG_PIPELINE_MAX_INFLIGHT_CHUNKS){
        fg_error_set(err,FG_ERR_LIMIT,
                     "pipeline in-flight ledger exceeds bounded capacity");
        free(pipeline);return FG_ERR_LIMIT;
    }
    pipeline->poison_status=FG_OK;
    uint64_t tensor_values=(uint64_t)manifest->prefill_microbatch*
        FG_PIPELINE_BOUNDARY_WIDTH;
    uint64_t wire_capacity=FG_PIPELINE_ACTIVATION_HEADER_BYTES+
        (uint64_t)manifest->prefill_microbatch*FG_PIPELINE_POSITION_AXES*4u+
        tensor_values*FG_PIPELINE_BOUNDARY_FP32_BYTES;
    if(wire_capacity>UINT32_MAX){
        fg_error_set(err,FG_ERR_LIMIT,"pipeline activation capacity exceeds wire limit");
        free(pipeline);return FG_ERR_LIMIT;
    }
    pipeline->wire_capacity=(uint32_t)wire_capacity;
    pipeline->slots=calloc(pipeline->slot_count,sizeof(*pipeline->slots));
    pipeline->batches=calloc(pipeline->batch_capacity,sizeof(*pipeline->batches));
    pipeline->outbound_available=calloc(pipeline->slot_count,sizeof(bool));
    pipeline->outbound_active=calloc(pipeline->slot_count,sizeof(bool));
    pipeline->outbound_sequence=calloc(pipeline->slot_count,sizeof(uint32_t));
    pipeline->wire=malloc(pipeline->wire_capacity);
    if(!pipeline->slots||!pipeline->batches||!pipeline->outbound_available||
       !pipeline->outbound_active||!pipeline->outbound_sequence||!pipeline->wire){
        fg_error_set(err,FG_ERR_OOM,"allocate bounded pipeline scheduler buffers");
        fg_pipeline_destroy(pipeline);return FG_ERR_OOM;
    }
    for(uint32_t i=0;i<pipeline->slot_count;i++){
        pipeline->slots[i].positions=malloc((size_t)manifest->prefill_microbatch*
            FG_PIPELINE_POSITION_AXES*sizeof(uint32_t));
        pipeline->slots[i].boundary=malloc((size_t)tensor_values*sizeof(float));
        if(!pipeline->slots[i].positions||!pipeline->slots[i].boundary){
            fg_error_set(err,FG_ERR_OOM,"allocate fixed pipeline activation slot");
            fg_pipeline_destroy(pipeline);return FG_ERR_OOM;
        }
    }
    *out=pipeline;
    return FG_OK;
}

void fg_pipeline_destroy(fg_pipeline *pipeline){
    if(!pipeline)return;
    if(pipeline->slots)for(uint32_t i=0;i<pipeline->slot_count;i++){
        free(pipeline->slots[i].boundary);
        free(pipeline->slots[i].positions);
    }
    free(pipeline->wire);
    free(pipeline->outbound_sequence);
    free(pipeline->outbound_active);
    free(pipeline->outbound_available);
    free(pipeline->batches);
    free(pipeline->slots);
    free(pipeline);
}

fg_status fg_pipeline_begin(fg_pipeline *pipeline,uint64_t request_id,
                            uint32_t first_sequence,fg_error *err){
    fg_status status=pipeline_owner(pipeline,err);
    if(status!=FG_OK)return status;
    if(!request_id||(pipeline->begun&&!pipeline->chain_drained)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid or overlapping pipeline session");
        return FG_ERR_ARGUMENT;
    }
    if(pipeline->begun&&!pipeline->stage&&pipeline->active_batches){
        fg_error_set(err,FG_ERR_LIMIT,
                     "pipeline restart would discard unread completed results");
        return FG_ERR_LIMIT;
    }
    memset(pipeline->batches,0,
           pipeline->batch_capacity*sizeof(*pipeline->batches));
    memset(pipeline->outbound_active,0,pipeline->slot_count*sizeof(bool));
    memset(pipeline->outbound_sequence,0,pipeline->slot_count*sizeof(uint32_t));
    for(uint32_t i=0;i<pipeline->slot_count;i++){
        pipeline->slots[i].occupied=false;
        pipeline->slots[i].ready=false;
        pipeline->slots[i].has_upstream=false;
        pipeline->outbound_available[i]=true;
    }
    pipeline->request_id=request_id;
    pipeline->next_submit_sequence=first_sequence;
    pipeline->next_receive_sequence=first_sequence;
    pipeline->next_credit_sequence=first_sequence;
    pipeline->next_result_sequence=first_sequence;
    pipeline->next_take_sequence=first_sequence;
    pipeline->operation_sequence=first_sequence;
    pipeline->drain_sequence=0u;
    pipeline->active_batches=0u;
    pipeline->admission_frontier=0u;
    pipeline->publication_cursor=0u;
    pipeline->published_frontier=0u;
    pipeline->drain_sent=false;
    pipeline->drain_received=false;
    pipeline->drain_forwarded=false;
    pipeline->drained_received=false;
    pipeline->chain_drained=false;
    pipeline->aborted=false;
    pipeline->abort_complete=false;
    pipeline->abort_propagated=false;
    pipeline->abort_discarded=false;
    pipeline->abort_origin_stage=0u;
    pipeline->abort_sequence=0u;
    pipeline->abort_required_mask=0u;
    pipeline->abort_notified_mask=0u;
    pipeline->admission_started=false;
    pipeline->begun=true;
    return FG_OK;
}

fg_status fg_pipeline_session_begin_validate(const fg_frame_header *header,
                                             uint64_t current_session_id,
                                             fg_error *err){
    if(!header){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "pipeline session header is null");
        return FG_ERR_ARGUMENT;
    }
    uint64_t request=fg_frame_request_id(header);
    uint32_t flags=ntohl(header->flags_be);
    if(fg_frame_type(header)!=FG_MSG_SESSION_BEGIN||!request||
       request<=current_session_id||fg_frame_sequence(header)!=0u||
       flags&~FG_PIPELINE_SESSION_RESET_FLAG){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "invalid pipeline session nonce, sequence, or flags");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_pipeline_submit_with_sampler(fg_pipeline *pipeline,
                             fg_pipeline_execution_kind execution_kind,
                             uint32_t first_token,uint16_t token_count,
                             bool request_output,
                             const uint32_t *positions,const float *boundary,
                             const fg_sampler_config *sampler,float uniform,
                             uint32_t *sequence,fg_error *err){
    fg_status status=pipeline_owner(pipeline,err);
    if(status!=FG_OK)return status;
    if(!pipeline->begun||pipeline->stage||pipeline->drain_sent){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline submit is unavailable in this state");
        return FG_ERR_ARGUMENT;
    }
    if(pipeline->active_batches>=pipeline->batch_capacity){
        fg_error_set(err,FG_ERR_LIMIT,"pipeline result window is backpressured");
        return FG_ERR_LIMIT;
    }
    int slot_index=available_outbound_slot(pipeline);
    if(slot_index<0){
        fg_error_set(err,FG_ERR_LIMIT,"pipeline activation slots are backpressured");
        return FG_ERR_LIMIT;
    }
    pipeline_batch *batch=free_batch(pipeline);
    if(!batch){
        fg_error_set(err,FG_ERR_LIMIT,"pipeline batch ledger is full");
        return FG_ERR_LIMIT;
    }
    pipeline_slot *slot=&pipeline->slots[slot_index];
    pipeline->operation_sequence=pipeline->next_submit_sequence;
    fg_sampler_config valid_sampler;fg_sampler_config_greedy(&valid_sampler);
    fg_pipeline_activation input={.execution_kind=execution_kind,
        .slot=(uint8_t)slot_index,.source_stage=0u,.destination_stage=1u,
        .first_token=first_token,.token_count=token_count,
        .request_output=request_output,.sampler=sampler?*sampler:valid_sampler,
        .uniform=uniform,.positions=positions,.boundary=boundary};
    fg_error local={0};
    status=fg_pipeline_activation_validate(&input,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(token_count>pipeline->manifest->prefill_microbatch){
        fg_error_set(&local,FG_ERR_LIMIT,
                     "pipeline submission exceeds manifest prefill chunk");
        return pipeline_poison(pipeline,FG_ERR_LIMIT,&local,err);
    }
    if(pipeline->admission_started&&first_token!=pipeline->admission_frontier){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline submission is not contiguous with admission frontier");
        return FG_ERR_MISMATCH;
    }
    uint64_t position_values=(uint64_t)token_count*FG_PIPELINE_POSITION_AXES;
    uint64_t boundary_values=(uint64_t)token_count*FG_PIPELINE_BOUNDARY_WIDTH;
    memcpy(slot->positions,positions,(size_t)position_values*sizeof(uint32_t));
    memcpy(slot->boundary,boundary,(size_t)boundary_values*sizeof(float));
    slot->activation=(fg_pipeline_activation){.execution_kind=input.execution_kind,
        .slot=input.slot,.source_stage=input.source_stage,
        .destination_stage=input.destination_stage,
        .first_token=input.first_token,.token_count=input.token_count,
        .request_output=input.request_output,.sampler=input.sampler,
        .uniform=input.uniform,.positions=slot->positions,.boundary=slot->boundary};
    slot->sequence=pipeline->next_submit_sequence;
    slot->occupied=true;
    status=execute_slot(pipeline,slot,NULL,err);
    if(status!=FG_OK)return status;
    status=send_activation(pipeline,slot,(uint32_t)slot_index,err);
    if(status!=FG_OK)return status;
    uint32_t admitted_frontier=first_token+token_count;
    *batch=(pipeline_batch){.sequence=pipeline->next_submit_sequence,
        .first_token=first_token,.admitted_frontier=admitted_frontier,
        .token_count=token_count,.request_output=request_output,.active=true};
    if(!pipeline->admission_started){
        pipeline->admission_started=true;
        pipeline->publication_cursor=first_token;
    }
    pipeline->admission_frontier=admitted_frontier;
    pipeline->active_batches++;
    if(sequence)*sequence=pipeline->next_submit_sequence;
    pipeline->next_submit_sequence++;
    return FG_OK;
}

fg_status fg_pipeline_submit(fg_pipeline *pipeline,
                             fg_pipeline_execution_kind execution_kind,
                             uint32_t first_token,uint16_t token_count,
                             bool request_output,const uint32_t *positions,
                             const float *boundary,uint32_t *sequence,fg_error *err){
    return fg_pipeline_submit_with_sampler(pipeline,execution_kind,first_token,
        token_count,request_output,positions,boundary,NULL,0.0f,sequence,err);
}

fg_status fg_pipeline_step(fg_pipeline *pipeline,fg_error *err){
    fg_status status=pipeline_owner(pipeline,err);
    if(status!=FG_OK)return status;
    if(!pipeline->begun){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline session has not begun");
        return FG_ERR_ARGUMENT;
    }
    uint32_t peer=0u,bytes=0u;
    fg_frame_header header;
    fg_error local={0};
    pipeline_slot *free_slot=free_local_slot(pipeline);
    if(pipeline->stage&&
       pipeline->stage+1u<pipeline->manifest->stage_count&&!free_slot){
        peer=stage_rank(pipeline,pipeline->stage+1u);
        status=pipeline->transport.receive_peer(
            pipeline->transport.context,peer,&header,pipeline->wire,
            pipeline->wire_capacity,&bytes,&local);
    }else{
        status=pipeline->transport.receive(pipeline->transport.context,&peer,
            &header,pipeline->wire,pipeline->wire_capacity,&bytes,&local);
    }
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    pipeline->operation_sequence=fg_frame_sequence(&header);
    uint32_t frame_bytes=0u;
    status=fg_frame_validate_version(&header,FG_PIPELINE_PROTOCOL_VERSION,
                                     pipeline->wire,&frame_bytes,&local);
    if(status!=FG_OK)return pipeline_poison(pipeline,status,&local,err);
    if(frame_bytes!=bytes){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline transport returned inconsistent payload length");
        return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
    switch(fg_frame_type(&header)){
        case FG_MSG_PIPELINE_ACTIVATION:
            if(!pipeline->stage){
                fg_error_set(&local,FG_ERR_MISMATCH,
                             "stage zero cannot receive pipeline activation");
                return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
            }
            return receive_activation(pipeline,peer,&header,bytes,err);
        case FG_MSG_PIPELINE_CREDIT:
            return receive_credit(pipeline,peer,&header,bytes,err);
        case FG_MSG_PIPELINE_RESULT:
            if(pipeline->stage){
                fg_error_set(&local,FG_ERR_MISMATCH,
                             "only stage zero can receive pipeline result");
                return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
            }
            return receive_result(pipeline,peer,&header,bytes,err);
        case FG_MSG_PIPELINE_DRAIN:
            return receive_drain(pipeline,peer,&header,bytes,err);
        case FG_MSG_PIPELINE_DRAINED:
            return receive_drained(pipeline,peer,&header,bytes,err);
        case FG_MSG_PIPELINE_ABORT:
            return receive_abort(pipeline,peer,&header,bytes,err);
        default:
            fg_error_set(&local,FG_ERR_MISMATCH,
                         "non-pipeline message reached pipeline I/O owner");
            return pipeline_poison(pipeline,FG_ERR_MISMATCH,&local,err);
    }
}

fg_status fg_pipeline_request_drain(fg_pipeline *pipeline,fg_error *err){
    fg_status status=pipeline_owner(pipeline,err);
    if(status!=FG_OK)return status;
    if(!pipeline->begun||pipeline->stage||pipeline->drain_sent){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline drain is unavailable in this state");
        return FG_ERR_ARGUMENT;
    }
    pipeline->drain_sequence=pipeline->next_submit_sequence;
    pipeline->operation_sequence=pipeline->drain_sequence;
    pipeline->drain_sent=true;
    pipeline->drain_received=true;
    return forward_drain(pipeline,err);
}

fg_status fg_pipeline_take_result(fg_pipeline *pipeline,fg_pipeline_result *result,
                                  uint32_t *sequence,fg_error *err){
    fg_status status=pipeline_owner(pipeline,err);
    if(status!=FG_OK)return status;
    if(!result||pipeline->stage){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline result consumer");
        return FG_ERR_ARGUMENT;
    }
    pipeline_batch *batch=find_batch(pipeline,pipeline->next_take_sequence);
    if(!batch||!batch->complete){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"next pipeline result is not available");
        return FG_ERR_UNAVAILABLE;
    }
    *result=batch->result;
    if(sequence)*sequence=batch->sequence;
    memset(batch,0,sizeof(*batch));
    pipeline->active_batches--;
    pipeline->next_take_sequence++;
    return FG_OK;
}

fg_status fg_pipeline_discard_aborted(fg_pipeline *pipeline,fg_error *err){
    if(!pipeline){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline scheduler is null");
        return FG_ERR_ARGUMENT;
    }
    pthread_t self=pthread_self();
    if(pipeline->io_owner_set&&!pthread_equal(pipeline->io_owner,self)){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline abort reset must run on the I/O owner thread");
        return FG_ERR_MISMATCH;
    }
    if(!pipeline->aborted){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline request is not aborted");
        return FG_ERR_ARGUMENT;
    }
    if(pipeline->abort_discarded)return FG_OK;
    if(pipeline->slots)for(uint32_t i=0;i<pipeline->slot_count;i++){
        free(pipeline->slots[i].boundary);
        free(pipeline->slots[i].positions);
        pipeline->slots[i].boundary=NULL;
        pipeline->slots[i].positions=NULL;
    }
    free(pipeline->wire);pipeline->wire=NULL;pipeline->wire_capacity=0u;
    free(pipeline->outbound_sequence);pipeline->outbound_sequence=NULL;
    free(pipeline->outbound_active);pipeline->outbound_active=NULL;
    free(pipeline->outbound_available);pipeline->outbound_available=NULL;
    free(pipeline->batches);pipeline->batches=NULL;
    free(pipeline->slots);pipeline->slots=NULL;
    pipeline->active_batches=0u;
    pipeline->abort_discarded=true;
    return FG_OK;
}

fg_status fg_pipeline_status(const fg_pipeline *pipeline,fg_error *err){
    if(!pipeline){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline scheduler is null");
        return FG_ERR_ARGUMENT;
    }
    if(pipeline->poison_status!=FG_OK&&err)*err=pipeline->poison_error;
    return pipeline->poison_status;
}

uint32_t fg_pipeline_stage(const fg_pipeline *pipeline){
    return pipeline?pipeline->stage:UINT32_MAX;
}

uint32_t fg_pipeline_slot_count(const fg_pipeline *pipeline){
    return pipeline?pipeline->slot_count:0u;
}

uint32_t fg_pipeline_available_slots(const fg_pipeline *pipeline){
    if(!pipeline||pipeline->poison_status!=FG_OK)return 0u;
    uint32_t count=0u;
    for(uint32_t i=0;i<pipeline->slot_count;i++)
        if(pipeline->outbound_available[i])count++;
    return count;
}

uint32_t fg_pipeline_max_inflight(const fg_pipeline *pipeline){
    return pipeline?pipeline->batch_capacity:0u;
}

uint32_t fg_pipeline_available_inflight(const fg_pipeline *pipeline){
    if(!pipeline||pipeline->poison_status!=FG_OK||
       pipeline->active_batches>pipeline->batch_capacity)
        return 0u;
    return pipeline->batch_capacity-pipeline->active_batches;
}

uint32_t fg_pipeline_admission_frontier(const fg_pipeline *pipeline){
    return pipeline&&pipeline->admission_started?pipeline->admission_frontier:0u;
}

uint32_t fg_pipeline_published_frontier(const fg_pipeline *pipeline){
    return pipeline?pipeline->published_frontier:0u;
}

bool fg_pipeline_is_drained(const fg_pipeline *pipeline){
    return pipeline&&pipeline->chain_drained;
}

bool fg_pipeline_abort_complete(const fg_pipeline *pipeline){
    return pipeline&&pipeline->aborted&&pipeline->abort_complete;
}

uint64_t fg_pipeline_host_bytes(const fg_pipeline *pipeline){
    if(!pipeline)return 0u;
    if(pipeline->abort_discarded)return sizeof(*pipeline);
    uint64_t per_slot=(uint64_t)pipeline->manifest->prefill_microbatch*
        (FG_PIPELINE_POSITION_AXES*sizeof(uint32_t)+
         FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float));
    return sizeof(*pipeline)+(uint64_t)pipeline->slot_count*
        (sizeof(pipeline_slot)+sizeof(bool)*2u+sizeof(uint32_t)+per_slot)+
        (uint64_t)pipeline->batch_capacity*sizeof(pipeline_batch)+
        pipeline->wire_capacity;
}
