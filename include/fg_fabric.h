#ifndef FLASH_GORDON_FABRIC_H
#define FLASH_GORDON_FABRIC_H

#include "fg_manifest.h"
#include "fg_protocol.h"
#include "fg_uring.h"

typedef enum fg_fabric_class { FG_FABRIC_CONTROL=0, FG_FABRIC_BULK=1 } fg_fabric_class;
typedef struct fg_fabric fg_fabric;
typedef fg_status (*fg_fabric_receiver)(void *,uint32_t,fg_fabric_class,int32_t,
    uint32_t *,fg_frame_header *,void *,uint32_t,uint32_t *,fg_error *);
void fg_fabric_set_receiver(fg_fabric *,fg_fabric_receiver,void *);
void fg_fabric_set_slot(fg_fabric *,uint32_t);
uint32_t fg_fabric_slot(const fg_fabric *);
bool fg_fabric_multiplexed(const fg_fabric *);
bool fg_fabric_connected(const fg_fabric *);
uint32_t fg_frame_session_slot(const fg_frame_header *);
fg_status fg_fabric_poll_slot(fg_fabric *,uint32_t,uint32_t,fg_fabric_class,
    uint32_t *,fg_frame_header *,void *,uint32_t,uint32_t *,fg_error *);
typedef struct fg_fabric_recv_timing {
    uint64_t poll_start_ns,ready_ns,header_end_ns,payload_end_ns,validate_end_ns;
    /* Monotonic counterparts used only by FG_FABRIC_PROFILE instrumentation. */
    uint64_t profile_poll_start_ns,profile_ready_ns,profile_header_end_ns,
             profile_payload_end_ns,profile_validate_end_ns;
    uint32_t ready_mask;
} fg_fabric_recv_timing;

fg_status fg_fabric_open(fg_fabric **out,const fg_manifest *manifest,uint32_t rank,fg_error *err);
void fg_fabric_close(fg_fabric *fabric);
uint16_t fg_fabric_protocol_version(const fg_fabric *fabric);
uint64_t fg_fabric_host_bytes(const fg_fabric *fabric);
fg_status fg_fabric_validate_frame(const fg_fabric *fabric,const fg_frame_header *header,
                                   const void *payload,uint32_t *payload_bytes,
                                   fg_error *err);
fg_status fg_fabric_send(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,fg_message_type type,uint64_t request_id,uint32_t sequence,uint32_t flags,const void *payload,uint32_t bytes,fg_error *err);
fg_status fg_fabric_send_direct(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
                                fg_message_type type,uint64_t request_id,uint32_t sequence,
                                uint32_t flags,const void *payload,uint32_t bytes,
                                fg_error *err);
fg_status fg_fabric_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,fg_frame_header *header,void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err);
fg_status fg_fabric_recv_timed(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,fg_frame_header *header,
                               void *payload,uint32_t capacity,uint32_t *bytes,
                               fg_fabric_recv_timing *timing,fg_error *err);
fg_status fg_fabric_recv_any(fg_fabric *fabric,fg_fabric_class cls,uint32_t *peer,fg_frame_header *header,
                             void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err);
fg_status fg_fabric_recv_any_timed(fg_fabric *fabric,fg_fabric_class cls,uint32_t *peer,
                                   fg_frame_header *header,void *payload,uint32_t capacity,
                                   uint32_t *bytes,fg_fabric_recv_timing *timing,fg_error *err);
fg_status fg_fabric_wait_ready(fg_fabric *fabric,uint32_t class_mask,uint32_t *peer,
                               fg_fabric_class *ready_class,fg_error *err);
/* Timeout variants return FG_ERR_LIMIT when timeout_ms elapses with nothing
 * ready; -1 waits forever exactly like the untimed calls. */
fg_status fg_fabric_wait_ready_timeout(fg_fabric *fabric,uint32_t class_mask,
                                       int32_t timeout_ms,uint32_t *peer,
                                       fg_fabric_class *ready_class,fg_error *err);
fg_status fg_fabric_recv_any_timeout(fg_fabric *fabric,fg_fabric_class cls,
                                     int32_t timeout_ms,uint32_t *peer,
                                     fg_frame_header *header,void *payload,
                                     uint32_t capacity,uint32_t *bytes,fg_error *err);
/* Async recv: prep header recv SQEs, flush, reap, then prep payload recvs. */
fg_status fg_fabric_prep_header_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
                                     fg_frame_header *header,uint64_t tag,fg_error *err);
fg_status fg_fabric_prep_payload_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
                                      void *payload,uint32_t bytes,uint64_t tag,fg_error *err);
fg_status fg_fabric_io_flush(fg_fabric *fabric,uint32_t count,fg_error *err);
fg_status fg_fabric_io_reap(fg_fabric *fabric,uint32_t min_count,fg_uring_cqe *out,
                            uint32_t capacity,uint32_t *completed,fg_error *err);

#endif
