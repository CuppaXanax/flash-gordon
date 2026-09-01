#ifndef FLASH_GORDON_PIPELINE_H
#define FLASH_GORDON_PIPELINE_H

#include "fg_fabric.h"

typedef struct fg_pipeline fg_pipeline;

#define FG_PIPELINE_SESSION_RESET_FLAG 1u

typedef fg_status (*fg_pipeline_send_fn)(void *context,uint32_t peer,
                                         fg_message_type type,uint64_t request_id,
                                         uint32_t sequence,const void *payload,
                                         uint32_t bytes,fg_error *err);
typedef fg_status (*fg_pipeline_receive_fn)(void *context,uint32_t *peer,
                                            fg_frame_header *header,void *payload,
                                            uint32_t capacity,uint32_t *bytes,
                                            fg_error *err);
typedef fg_status (*fg_pipeline_receive_peer_fn)(void *context,uint32_t peer,
                                                fg_frame_header *header,
                                                void *payload,uint32_t capacity,
                                                uint32_t *bytes,fg_error *err);

typedef struct fg_pipeline_transport {
    void *context;
    fg_pipeline_send_fn send;
    fg_pipeline_receive_fn receive;
    fg_pipeline_receive_peer_fn receive_peer;
} fg_pipeline_transport;

/*
 * Execution callbacks run synchronously on the scheduler's single I/O owner.
 * This provides bounded inter-stage chunk pipelining; receive/compute overlap
 * within one stage remains a later asynchronous Vulkan integration.
 */
typedef fg_status (*fg_pipeline_execute_fn)(void *context,uint32_t stage,
                                            uint64_t request_id,uint32_t sequence,
                                            fg_pipeline_activation *activation,
                                            float *boundary,
                                            fg_pipeline_result *terminal_result,
                                            fg_error *err);

typedef struct fg_pipeline_config {
    const fg_manifest *manifest;
    uint32_t rank;
    fg_pipeline_transport transport;
    fg_pipeline_execute_fn execute;
    void *execute_context;
} fg_pipeline_config;

fg_status fg_pipeline_create(fg_pipeline **out,const fg_pipeline_config *config,
                             fg_error *err);
void fg_pipeline_destroy(fg_pipeline *pipeline);
void fg_pipeline_transport_init_fabric(fg_pipeline_transport *transport,
                                       fg_fabric *fabric);

fg_status fg_pipeline_begin(fg_pipeline *pipeline,uint64_t request_id,
                            uint32_t first_sequence,fg_error *err);
fg_status fg_pipeline_session_begin_validate(const fg_frame_header *header,
                                             uint64_t current_session_id,
                                             fg_error *err);
/* Decode always requires output; prefill callers select only chunks needing logits. */
fg_status fg_pipeline_submit(fg_pipeline *pipeline,
                             fg_pipeline_execution_kind execution_kind,
                             uint32_t first_token,uint16_t token_count,
                             bool request_output,
                             const uint32_t *positions,const float *boundary,
                             uint32_t *sequence,fg_error *err);
fg_status fg_pipeline_step(fg_pipeline *pipeline,fg_error *err);
fg_status fg_pipeline_request_drain(fg_pipeline *pipeline,fg_error *err);
fg_status fg_pipeline_take_result(fg_pipeline *pipeline,fg_pipeline_result *result,
                                  uint32_t *sequence,fg_error *err);
/*
 * Releases retained aborted-request buffers and unread results. The scheduler
 * remains permanently poisoned; restart requires destroying scheduler/fabric.
 */
fg_status fg_pipeline_discard_aborted(fg_pipeline *pipeline,fg_error *err);

fg_status fg_pipeline_status(const fg_pipeline *pipeline,fg_error *err);
uint32_t fg_pipeline_stage(const fg_pipeline *pipeline);
uint32_t fg_pipeline_slot_count(const fg_pipeline *pipeline);
uint32_t fg_pipeline_available_slots(const fg_pipeline *pipeline);
uint32_t fg_pipeline_max_inflight(const fg_pipeline *pipeline);
uint32_t fg_pipeline_available_inflight(const fg_pipeline *pipeline);
uint32_t fg_pipeline_admission_frontier(const fg_pipeline *pipeline);
uint32_t fg_pipeline_published_frontier(const fg_pipeline *pipeline);
bool fg_pipeline_is_drained(const fg_pipeline *pipeline);
bool fg_pipeline_abort_complete(const fg_pipeline *pipeline);
uint64_t fg_pipeline_host_bytes(const fg_pipeline *pipeline);

#endif
