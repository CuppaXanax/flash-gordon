#ifndef FLASH_GORDON_FABRIC_H
#define FLASH_GORDON_FABRIC_H

#include "fg_manifest.h"
#include "fg_protocol.h"

typedef enum fg_fabric_class { FG_FABRIC_CONTROL=0, FG_FABRIC_BULK=1 } fg_fabric_class;
typedef struct fg_fabric fg_fabric;

fg_status fg_fabric_open(fg_fabric **out,const fg_manifest *manifest,uint32_t rank,fg_error *err);
void fg_fabric_close(fg_fabric *fabric);
fg_status fg_fabric_send(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,fg_message_type type,uint64_t request_id,uint32_t sequence,uint32_t flags,const void *payload,uint32_t bytes,fg_error *err);
fg_status fg_fabric_recv(fg_fabric *fabric,uint32_t peer,fg_fabric_class cls,fg_frame_header *header,void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err);
fg_status fg_fabric_recv_any(fg_fabric *fabric,fg_fabric_class cls,uint32_t *peer,fg_frame_header *header,
                             void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err);
fg_status fg_fabric_wait_ready(fg_fabric *fabric,uint32_t class_mask,uint32_t *peer,
                               fg_fabric_class *ready_class,fg_error *err);

#endif
