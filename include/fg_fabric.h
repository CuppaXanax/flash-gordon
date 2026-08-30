#ifndef FLASH_GORDON_FABRIC_H
#define FLASH_GORDON_FABRIC_H

#include "fg_manifest.h"
#include "fg_protocol.h"
#include "fg_uring.h"

typedef enum fg_fabric_class { FG_FABRIC_CONTROL=0, FG_FABRIC_BULK=1 } fg_fabric_class;
typedef struct fg_fabric fg_fabric;
typedef struct fg_fabric_recv_timing {
    uint64_t poll_start_ns,ready_ns,header_end_ns,payload_end_ns,validate_end_ns;
    uint32_t ready_mask;
} fg_fabric_recv_timing;
typedef struct fg_fabric_send_item {uint32_t peer;fg_fabric_class cls;fg_message_type type;uint64_t request_id;uint32_t sequence,flags;const void *payload;uint32_t bytes;} fg_fabric_send_item;

fg_status fg_fabric_open(fg_fabric **out,const fg_manifest *manifest,uint32_t rank,fg_error *err);
void fg_fabric_close(fg_fabric *fabric);
fg_status fg_fabric_send(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,fg_message_type type,uint64_t request_id,uint32_t sequence,uint32_t flags,const void *payload,uint32_t bytes,fg_error *err);
fg_status fg_fabric_send_batch(fg_fabric *fabric,const fg_fabric_send_item *items,uint32_t count,fg_error *err);
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
/* Async recv: prep header recv SQEs, flush, reap, then prep payload recvs. */
fg_status fg_fabric_prep_header_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
                                     fg_frame_header *header,uint64_t tag,fg_error *err);
fg_status fg_fabric_prep_payload_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,
                                      void *payload,uint32_t bytes,uint64_t tag,fg_error *err);
fg_status fg_fabric_io_flush(fg_fabric *fabric,uint32_t count,fg_error *err);
fg_status fg_fabric_io_reap(fg_fabric *fabric,uint32_t min_count,fg_uring_cqe *out,
                            uint32_t capacity,uint32_t *completed,fg_error *err);

#endif
