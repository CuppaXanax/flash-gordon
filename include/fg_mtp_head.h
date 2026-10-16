#ifndef FLASH_GORDON_MTP_HEAD_H
#define FLASH_GORDON_MTP_HEAD_H

#include "fg_model.h"

typedef struct fg_mtp_head fg_mtp_head;
/* Optional read-only diagnostics. Tensor contents are valid only in the
 * callback; enabling this hook adds synchronous readbacks in the caller. */
typedef fg_status (*fg_mtp_head_trace_fn)(void *context,const char *stage,
    const fg_vk_tensor *values,uint32_t count,fg_error *err);
void fg_mtp_head_set_trace(fg_mtp_head *head,fg_mtp_head_trace_fn trace,void *context);

/* Independent dense draft KV; no target model state is modified. Only the
 * manifest owner of the last target layer may construct a native head. */
fg_status fg_mtp_head_open(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,const char *overlay,
    uint32_t context_tokens,fg_error *err);
void fg_mtp_head_close(fg_mtp_head *head);
uint32_t fg_mtp_head_frontier(const fg_mtp_head *head);
uint64_t fg_mtp_head_weight_bytes(const fg_mtp_head *head);
fg_status fg_mtp_head_truncate(fg_mtp_head *head,uint32_t cells,fg_error *err);
/* Cell c consumes target HC[c] and embedding(token[c+1]), predicts token[c+2].
 * Warmfill and repair must use authoritative target HC, not rejected drafts.
 * Output tensors are head-owned scratch, valid until the next step. */
fg_status fg_mtp_head_step(fg_mtp_head *head,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_vk_tensor **next_hyper,
    fg_vk_tensor **contracted,fg_error *err);

#endif
