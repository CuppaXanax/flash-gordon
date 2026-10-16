#ifndef FLASH_GORDON_MTP_HEAD_H
#define FLASH_GORDON_MTP_HEAD_H

#include "fg_model.h"

typedef struct fg_mtp_head fg_mtp_head;
typedef struct fg_mtp_vocab fg_mtp_vocab;
/* Explicit global token map: a production shard, or a separately qualified
 * draft-only vocabulary. NULL selects the entire shared vocabulary. */
fg_status fg_mtp_vocab_open(fg_mtp_vocab **out,fg_vk_context *vk,const char *head_q8,
    const uint32_t *token_ids,uint32_t rows,fg_error *err);
void fg_mtp_vocab_close(fg_mtp_vocab *vocab);
fg_status fg_mtp_vocab_argmax(fg_mtp_vocab *vocab,const fg_vk_tensor *hidden,
    uint32_t *token,float *score,fg_error *err);
/* Optional read-only diagnostics. Tensor contents are valid only in the
 * callback; enabling this hook adds synchronous readbacks in the caller. */
typedef fg_status (*fg_mtp_head_trace_fn)(void *context,const char *stage,
    const fg_vk_tensor *values,uint32_t count,fg_error *err);
void fg_mtp_head_set_trace(fg_mtp_head *head,fg_mtp_head_trace_fn trace,void *context);

/* Independent dense draft KV; no target model state is modified. Default
 * constructors require the manifest owner of the last target layer. */
fg_status fg_mtp_head_open(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,const char *overlay,
    uint32_t context_tokens,fg_error *err);
/* A bounded draft-only attention window, with four extra ring cells so a
 * rejected k<=3 round cannot evict keys needed after rollback. RoPE remains
 * absolute. This approximation needs a separate real-text acceptance gate. */
fg_status fg_mtp_head_open_windowed(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,const char *overlay,
    uint32_t context_tokens,uint32_t window_tokens,fg_error *err);
/* Explicit placement also permits the final block's contiguous predecessor. */
fg_status fg_mtp_head_open_owner_windowed(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *target,uint32_t rank,uint32_t native_owner,const char *overlay,
    uint32_t context_tokens,uint32_t window_tokens,fg_error *err);
/* Share only immutable weights. KV, replay descriptors and all scratch remain
 * private to state_slot; either instance may be closed first. */
fg_status fg_mtp_head_fork(fg_mtp_head **out,fg_mtp_head *source,
    uint32_t state_slot,fg_error *err);
void fg_mtp_head_close(fg_mtp_head *head);
uint32_t fg_mtp_head_frontier(const fg_mtp_head *head);
uint64_t fg_mtp_head_weight_bytes(const fg_mtp_head *head);
fg_status fg_mtp_head_check_residency(const fg_mtp_head *head,fg_error *err);
fg_status fg_mtp_head_truncate(fg_mtp_head *head,uint32_t cells,fg_error *err);
fg_status fg_mtp_head_reset(fg_mtp_head *head,uint32_t first_cell,fg_error *err);
/* Cell c consumes target HC[c] and embedding(token[c+1]), predicts token[c+2].
 * Warmfill and repair must use authoritative target HC, not rejected drafts.
 * Output tensors are head-owned scratch, valid until the next step. */
fg_status fg_mtp_head_step(fg_mtp_head *head,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_vk_tensor **next_hyper,
    fg_vk_tensor **contracted,fg_error *err);
/* GPU-routed single-submission core, replayed with indirect KV/RoPE position.
 * The first call records; subsequent cells reuse the same command buffer. */
fg_status fg_mtp_head_step_replay(fg_mtp_head *head,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_vk_tensor **next_hyper,
    fg_vk_tensor **contracted,fg_error *err);
/* Authoritative prefix/repair needs only the input projection and K/V, not
 * attention, Q/output projections, experts or the vocabulary head. */
fg_status fg_mtp_head_warm_replay(fg_mtp_head *head,const fg_vk_tensor *hyper,
    const fg_vk_tensor *embedding,uint32_t cell,fg_error *err);

#endif
