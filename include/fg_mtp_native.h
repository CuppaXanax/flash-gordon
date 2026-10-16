#ifndef FLASH_GORDON_MTP_NATIVE_H
#define FLASH_GORDON_MTP_NATIVE_H

#include "fg_mtp_head.h"
#include "fg_mtp_wire.h"

typedef struct fg_mtp_native fg_mtp_native;

fg_status fg_mtp_native_create(fg_mtp_native **out,fg_vk_context *vk,
    const fg_manifest *manifest,uint32_t rank,fg_error *err);
void fg_mtp_native_close(fg_mtp_native *native);
bool fg_mtp_native_enabled(const fg_mtp_native *native);
uint32_t fg_mtp_native_owner(const fg_mtp_native *native);
fg_status fg_mtp_native_reset(fg_mtp_native *native,uint32_t slot,fg_error *err);
/* Warm only the trailing native window before returning the target frame.
 * next_tokens[c] supplies the next real input for authoritative HC[c]. */
fg_status fg_mtp_native_observe_prefill(fg_mtp_native *native,uint32_t slot,
    uint32_t first,uint32_t count,const float *hyper,const uint32_t *next_tokens,
    fg_error *err);
fg_status fg_mtp_native_observe_decode(fg_mtp_native *native,uint32_t slot,
    uint32_t first,const float *hyper,fg_error *err);
fg_status fg_mtp_native_observe_verify(fg_mtp_native *native,uint32_t slot,
    uint32_t first,uint32_t count,const float *hyper,fg_error *err);
/* DRAFT returns one already-contracted hidden vector. Every other operation
 * returns only an acknowledgement; target state is never modified here. */
fg_status fg_mtp_native_run(fg_mtp_native *native,uint32_t slot,
    const fg_mtp_native_work *work,float hidden[FG_HIDDEN_SIZE],
    fg_mtp_native_ack *ack,fg_error *err);

#endif
