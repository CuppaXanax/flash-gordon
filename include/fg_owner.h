#ifndef FLASH_GORDON_OWNER_H
#define FLASH_GORDON_OWNER_H

#include "fg_expert.h"
#include "fg_qsa.h"

typedef struct fg_owner_executor fg_owner_executor;
typedef fg_status (*fg_owner_expert_dispatch_fn)(void *context,uint32_t layer,uint32_t token_index,
                                                  const uint16_t expert_ids[FG_TOP_K],
                                                  const float gates[FG_TOP_K],const uint8_t *activation_q8k,
                                                  fg_expert_result results[FG_GROUP_SIZE],
                                                  uint32_t *result_count,fg_error *err);
/* Async expert dispatch: fire sends + submit recvs, then collect results later. */
typedef fg_status (*fg_owner_expert_fire_fn)(void *context,uint32_t layer,uint32_t token_index,
                                             const uint16_t expert_ids[FG_TOP_K],
                                             const float gates[FG_TOP_K],const uint8_t *activation_q8k,
                                             fg_error *err);
typedef fg_status (*fg_owner_expert_collect_fn)(void *context,uint32_t layer,uint32_t token_index,
                                                fg_expert_result results[FG_GROUP_SIZE],
                                                uint32_t *result_count,fg_error *err);
/* Dispatch calls shared_work once after sending remote routes and drains every
 * issued result even if shared or local computation fails. */
typedef fg_status (*fg_owner_prefill_shared_fn)(void *context,fg_error *err);
typedef fg_status (*fg_owner_prefill_dispatch_fn)(void *context,uint32_t layer,
                                                   uint32_t first_token,uint16_t token_count,
                                                   const uint16_t *expert_ids,const float *gates,
                                                   const uint8_t *activations_q8k,
                                                   fg_owner_prefill_shared_fn shared_work,
                                                   void *shared_context,
                                                   fg_prefill_result results[FG_GROUP_SIZE],
                                                   uint32_t *result_count,fg_error *err);
typedef fg_status (*fg_owner_qsa_decode_dispatch_fn)(void *context,uint32_t layer,
                                                     uint32_t token_index,
                                                     const uint32_t position[3],
                                                     const fg_vk_tensor *hidden,
                                                     fg_vk_tensor **output,fg_error *err);
typedef fg_status (*fg_owner_qsa_prefill_dispatch_fn)(void *context,uint32_t layer,
                                                      uint32_t first_token,
                                                      const uint32_t *positions,
                                                      uint16_t token_count,
                                                      const fg_vk_tensor *hidden,
                                                      fg_vk_tensor **output,fg_error *err);
/* Pipelined prefill: fire sends remote work and computes local experts; collect
 * drains remote tiles.  Shared expert work stays on the owner between them. */
typedef fg_status (*fg_owner_prefill_fire_fn)(void *context,uint32_t layer,
                                              uint32_t first_token,uint16_t token_count,
                                              const uint16_t *expert_ids,const float *gates,
                                              const uint8_t *activations_q8k,fg_error *err);
typedef fg_status (*fg_owner_prefill_collect_fn)(void *context,uint32_t layer,
                                                 uint32_t first_token,uint16_t token_count,
                                                 fg_prefill_result results[FG_GROUP_SIZE],
                                                 uint32_t *result_count,fg_error *err);

/* Depth-B owner state namespaces.  Session 0 is the legacy single-session
 * state; session 1 is allocated only by the slot-count constructors and is
 * what a depth-2 batch step advances.  Weights, arena and command
 * infrastructure stay singular. */
#define FG_OWNER_SESSION_MAX 2u

fg_status fg_owner_executor_create(fg_owner_executor **out,fg_model *model,fg_error *err);
/* Sealed single-owner worker: owns only this rank's layers (manifest
 * layer_owner), allocates per-layer state for owned layers only. */
fg_status fg_owner_executor_create_worker(fg_owner_executor **out,fg_model *model,fg_error *err);
/* Slot-count variants: `sessions` in 1..FG_OWNER_SESSION_MAX.  Existing
 * callers keep one session; the depth-B batch path allocates two. */
fg_status fg_owner_executor_create_slots(fg_owner_executor **out,fg_model *model,
                                         uint32_t sessions,fg_error *err);
fg_status fg_owner_executor_create_worker_slots(fg_owner_executor **out,fg_model *model,
                                                uint32_t sessions,fg_error *err);
void fg_owner_executor_destroy(fg_owner_executor *executor);
uint32_t fg_owner_session_count(const fg_owner_executor *executor);
/* Select the owner state namespace every stateful call below operates on.
 * Session 0 is the default; selecting an unallocated session fails closed. */
fg_status fg_owner_set_active_session(fg_owner_executor *executor,uint32_t session,
                                      fg_error *err);
uint32_t fg_owner_active_session(const fg_owner_executor *executor);
/* Per-session transaction hook for the depth-B batch step: snapshot copies the
 * session's authoritative GDN/PLE device state and QSA frontier to the host;
 * rollback restores both.  Release frees the host staging. */
typedef struct fg_owner_session_checkpoint {
    bool valid;
    /* Device-side checkpoint: GDN/PLE state is shadowed by the executor's
     * per-session shadow tensors (a GPU copy) instead of host staging. */
    bool device;
    uint32_t session;
    uint32_t gdn_count;
    struct {
        fg_vk_tensor *tensor;
        float *data;
        uint64_t values;
    } gdn[FG_LAYER_COUNT*2u];
    fg_vk_tensor *ple;
    float *ple_data;
    uint64_t ple_values;
    bool qsa_valid;
    uint32_t qsa_tokens[FG_LAYER_COUNT];
} fg_owner_session_checkpoint;
fg_status fg_owner_session_snapshot(fg_owner_executor *executor,uint32_t session,
                                    fg_owner_session_checkpoint *snapshot,fg_error *err);
/* Device-side variant: copies the session's GDN conv/recurrent and PLE state
 * into per-session shadow tensors with vkCmdCopyBuffer (no host readback), and
 * records the QSA frontier.  Rollback copies the shadows back. */
fg_status fg_owner_session_device_snapshot(fg_owner_executor *executor,uint32_t session,
                                           fg_owner_session_checkpoint *snapshot,
                                           fg_error *err);
fg_status fg_owner_session_rollback(fg_owner_executor *executor,
                                    const fg_owner_session_checkpoint *snapshot,
                                    fg_error *err);
void fg_owner_session_snapshot_release(fg_owner_session_checkpoint *snapshot);
bool fg_owner_owns_layer(const fg_owner_executor *executor,uint32_t layer);
fg_vk_tensor *fg_owner_prefill_input(fg_owner_executor *executor);
uint64_t fg_owner_qsa_host_bytes(const fg_owner_executor *executor);
fg_status fg_owner_reset_state(fg_owner_executor *executor,fg_error *err);
fg_status fg_owner_qsa_checkpoint(fg_owner_executor *executor,fg_error *err);
fg_status fg_owner_gr_read(fg_owner_executor *executor,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,
                           fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err);
fg_status fg_owner_gr_read_batch(fg_owner_executor *executor,uint32_t layer,bool ffn,
                                 const fg_vk_tensor *hyper_input,uint32_t token_count,
                                 fg_vk_tensor **mixed,const fg_vk_tensor **residual,
                                 fg_vk_tensor **injection,fg_error *err);
fg_status fg_owner_moe_prepare(fg_owner_executor *executor,uint32_t layer,const fg_vk_tensor *hidden,
                               uint16_t expert_ids[FG_TOP_K],float gates[FG_TOP_K],
                               const uint8_t **activation_q8k,fg_error *err);
fg_status fg_owner_moe_prepare_batch(fg_owner_executor *executor,uint32_t layer,
                                     const fg_vk_tensor *hidden,uint16_t token_count,
                                     uint16_t *expert_ids,float *gates,
                                     const uint8_t **activation_q8k,fg_error *err);
fg_status fg_owner_moe_reduce(fg_owner_executor *executor,uint32_t layer,uint32_t position,
                              const uint16_t expert_ids[FG_TOP_K],
                              const float gates[FG_TOP_K],const fg_expert_result *results,
                              uint32_t result_count,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_moe_reduce_batch(fg_owner_executor *executor,uint32_t layer,
                                    uint32_t first_position,uint16_t token_count,
                                    const uint16_t *expert_ids,const float *gates,
                                    const fg_prefill_result *results,uint32_t result_count,
                                    fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gr_write(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                            const fg_vk_tensor *block_output,const fg_vk_tensor *injection,
                            fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gr_write_batch(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                                  const fg_vk_tensor *block_output,const fg_vk_tensor *injection,
                                  uint32_t token_count,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gdn_decode(fg_owner_executor *executor,uint32_t layer,const fg_vk_tensor *hidden,
                              fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gdn_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t token_count,
                               const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_ple_decode(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                              const fg_vk_tensor *ngram_embedding,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_ple_prefill(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                               const fg_vk_tensor *ngram_embeddings,uint32_t token_count,
                               fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_qsa_open(fg_owner_executor *executor,const char *state_path,bool create,
                            fg_error *err);
fg_status fg_owner_qsa_open_decode(fg_owner_executor *executor,const char *state_path,
                                   uint32_t resident_tokens,uint32_t batch_size,fg_error *err);
fg_status fg_owner_qsa_open_mirror(fg_owner_executor *executor,uint32_t logical_context,
                                   uint32_t hot_tokens,uint32_t cache_pages,uint32_t batch_size,
                                   fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                   fg_error *err);
fg_status fg_owner_qsa_open_state(fg_owner_executor *executor,const char *state_path,
                                  uint32_t logical_context,uint32_t hot_tokens,
                                  uint32_t cache_pages,uint32_t batch_size,fg_error *err);
fg_status fg_owner_qsa_open_state_mirror(fg_owner_executor *executor,const char *state_path,
                                         uint32_t logical_context,uint32_t hot_tokens,
                                         uint32_t cache_pages,uint32_t batch_size,
                                         bool owned_only,
                                         fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                         fg_error *err);
/* Per-session QSA open: same contracts as the calls above, but the session is
 * explicit so slot 1 gets its own state file and index/cache allocation. */
fg_status fg_owner_qsa_open_state_slot(fg_owner_executor *executor,uint32_t session,
                                       const char *state_path,uint32_t logical_context,
                                       uint32_t hot_tokens,uint32_t cache_pages,
                                       uint32_t batch_size,fg_error *err);
fg_status fg_owner_qsa_open_state_mirror_slot(fg_owner_executor *executor,uint32_t session,
                                              const char *state_path,uint32_t logical_context,
                                              uint32_t hot_tokens,uint32_t cache_pages,
                                              uint32_t batch_size,bool owned_only,
                                              fg_qsa_page_fetch_fn fetch_pages,
                                              void *fetch_opaque,fg_error *err);
fg_status fg_owner_qsa_open_mirror_slot(fg_owner_executor *executor,uint32_t session,
                                        uint32_t logical_context,uint32_t hot_tokens,
                                        uint32_t cache_pages,uint32_t batch_size,
                                        fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                        fg_error *err);
bool fg_owner_qsa_ready_slot(const fg_owner_executor *executor,uint32_t session);
fg_status fg_owner_qsa_frontier(const fg_owner_executor *executor,uint32_t session,
                                uint32_t tokens[FG_LAYER_COUNT]);
fg_status fg_owner_qsa_rollback(fg_owner_executor *executor,uint32_t session,
                                const uint32_t tokens[FG_LAYER_COUNT],fg_error *err);
/* Per-session embedding input storage for the depth-B batch step (one 40 KiB
 * hyper vector per session; allocated with the session). */
fg_vk_tensor *fg_owner_session_input(fg_owner_executor *executor,uint32_t session);
bool fg_owner_qsa_ready(const fg_owner_executor *executor);
void fg_owner_qsa_set_tokens(fg_owner_executor *executor,uint32_t tokens);
/* Ring decode-state handoff accessors: slot 0 is the GDN conv state, slot 1 the
 * recurrent state; layer 1 additionally carries the PLE convolution state. */
fg_vk_tensor *fg_owner_gdn_state_tensor(fg_owner_executor *executor,uint32_t layer,
                                        uint32_t slot);
fg_vk_tensor *fg_owner_ple_state_tensor(fg_owner_executor *executor);
uint32_t fg_owner_gdn_layers(const fg_owner_executor *executor,uint8_t *layers,
                             uint32_t capacity,fg_error *err);
fg_status fg_owner_qsa_decode(fg_owner_executor *executor,uint32_t layer,uint32_t token_index,
                              const uint32_t position[3],const fg_vk_tensor *hidden,
                              fg_vk_tensor **output,fg_error *err);
/* Canonical routed-expert result arena; the sealed block owner fills it during
 * the decode fire callback and hands it to the shared reduce on collect. */
fg_expert_result *fg_owner_decode_results(fg_owner_executor *executor);
fg_status fg_owner_qsa_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t first_token,
                               const uint32_t *positions,uint32_t token_count,
                               const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_qsa_page_records(const fg_owner_executor *executor,uint32_t layer,
                                    uint32_t block,const uint8_t **records,fg_error *err);
fg_status fg_owner_qsa_warm_pages(fg_owner_executor *executor,uint32_t layer,
                                  const uint32_t *blocks,const uint8_t *records,
                                  uint32_t page_count,fg_error *err);
bool fg_owner_qsa_page_cached(fg_owner_executor *executor,uint32_t layer,uint32_t block);
void fg_owner_qsa_page_published(fg_owner_executor *executor,uint32_t layer,uint32_t block);
fg_status fg_owner_qsa_state_records(fg_owner_executor *executor,uint32_t layer,
                                     uint32_t block,uint8_t *records,fg_error *err);
fg_status fg_owner_qsa_state_records_batch(fg_owner_executor *executor,uint32_t layer,
                                           const uint32_t *blocks,uint32_t page_count,
                                           uint8_t *records,fg_error *err);
fg_status fg_owner_decode_layer(fg_owner_executor *executor,uint32_t layer,uint32_t token_index,
                                const uint32_t position[3],const fg_vk_tensor *hyper_input,
                                const fg_vk_tensor *ngram_embedding,
                                fg_owner_expert_dispatch_fn dispatch,void *dispatch_context,
                                fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_decode_layer_async(fg_owner_executor *executor,uint32_t layer,uint32_t token_index,
                                      const uint32_t position[3],const fg_vk_tensor *hyper_input,
                                      const fg_vk_tensor *ngram_embedding,
                                      fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,
                                      void *dispatch_context,
                                      fg_owner_qsa_decode_dispatch_fn qsa_dispatch,
                                      void *qsa_context,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_decode_layer_begin(fg_owner_executor *executor,uint32_t slot,uint32_t layer,
                                      uint32_t token_index,const uint32_t position[3],
                                      const fg_vk_tensor *hyper_input,
                                      const fg_vk_tensor *ngram_embedding,
                                      fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,
                                      void *dispatch_context,
                                      fg_owner_qsa_decode_dispatch_fn qsa_dispatch,
                                      void *qsa_context,fg_error *err);
/* Ring decode block: run layers [first,last] of one token from the owner's
 * authoritative state, including a mid-model block start with no local
 * predecessor. */
fg_status fg_owner_decode_block(fg_owner_executor *executor,uint32_t first_layer,
                                uint32_t last_layer,uint32_t token_index,
                                const uint32_t position[3],const fg_vk_tensor *hyper_input,
                                const fg_vk_tensor *ngram_embedding,
                                fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,
                                void *dispatch_context,fg_vk_tensor **output,fg_error *err);
/* Chained ring block: the common path, shared expert, GPU routing and fused
 * expert pair for every layer are recorded into one submission so a block
 * costs a single fence.  The expert callback runs inside the active batch and
 * must leave its gate-weighted rank-local sum in an own tensor. */
typedef fg_status (*fg_owner_expert_inline_fn)(void *context,uint32_t layer,
    const fg_vk_tensor *activation_q8k,const fg_vk_tensor *router_logits,
    fg_vk_tensor **expert_output,fg_error *err);
fg_status fg_owner_decode_block_chained(fg_owner_executor *executor,uint32_t first_layer,
                                        uint32_t last_layer,uint32_t token_index,
                                        const uint32_t position[3],const fg_vk_tensor *hyper_input,
                                        const fg_vk_tensor *ngram_embedding,
                                        fg_owner_expert_inline_fn expert,void *expert_context,
                                        fg_vk_tensor **output,fg_error *err);
/* Depth-B batch-2 block: two slots' tokens through one layer pass per layer.
 * `hyper_input` is a two-row residual (token-major), `ngram_embedding` is a
 * two-row n-gram embedding when the range includes layer 1, and `output` is
 * the two-row residual after `last_layer`.  The expert callback receives the
 * two-row activation/router tensors and must leave two gate-weighted hidden
 * rows (the fg_expert_decode_chain_b2 contract).  Eligibility is
 * fg_owner_decode_block_batch_ready; an ineligible range must fall back to the
 * per-slot chained block. */
bool fg_owner_decode_block_batch_ready(fg_owner_executor *executor,uint32_t first_layer,
                                       uint32_t last_layer,fg_error *err);
fg_status fg_owner_decode_block_batch(fg_owner_executor *executor,uint32_t first_layer,
                                      uint32_t last_layer,const uint32_t token_index[2],
                                      const uint32_t state_slot[2],
                                      const uint32_t positions[6],
                                      const fg_vk_tensor *hyper_input,
                                      const fg_vk_tensor *ngram_embedding,
                                      fg_owner_expert_inline_fn expert,void *expert_context,
                                      fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_decode_layer_finish(fg_owner_executor *executor,uint32_t slot,
                                       fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_prefill_layer(fg_owner_executor *executor,uint32_t layer,
                                 uint32_t first_token,const uint32_t *positions,
                                 uint16_t token_count,const fg_vk_tensor *hyper_input,
                                 const fg_vk_tensor *ngram_embeddings,
                                 fg_owner_prefill_dispatch_fn dispatch,void *dispatch_context,
                                 fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,
                                 void *qsa_context,
                                 fg_vk_tensor **output,fg_error *err);
fg_vk_tensor *fg_owner_prefill_input_slot(fg_owner_executor *executor,uint32_t slot);
fg_status fg_owner_prefill_layer_begin(fg_owner_executor *executor,uint32_t slot,uint32_t layer,
                                       uint32_t first_token,const uint32_t *positions,
                                       uint16_t token_count,const fg_vk_tensor *hyper_input,
                                       const fg_vk_tensor *ngram_embeddings,
                                       fg_owner_prefill_fire_fn fire,void *fire_context,
                                       fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,
                                       void *qsa_context,fg_error *err);
fg_status fg_owner_prefill_layer_finish(fg_owner_executor *executor,uint32_t slot,
                                        fg_owner_prefill_collect_fn collect,void *collect_context,
                                        fg_vk_tensor **output,fg_error *err);

#endif
