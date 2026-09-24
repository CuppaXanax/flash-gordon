#ifndef FLASH_GORDON_DECODE_BATCH_H
#define FLASH_GORDON_DECODE_BATCH_H

#include "fg_protocol.h"
#include "fg_sampler.h"

/* Depth-B ring decode: the host-side batch table, scheduler and transactional
 * step.  A ring step carries one slot per ready sequence so the weights are
 * read once and B tokens are emitted.  Every sequence owns its frontier,
 * positions, QSA record cursor, sampling state and owner state slot; the
 * weights/arena/fabric stay singular.
 *
 * B=1 is not routed through this module in the runtime: the single-token ring
 * path remains the B=1 code path.  The module exists to assemble, transact and
 * account for B>=2 steps. */

#define FG_DECODE_BATCH_MAX FG_DECODE_BATCH_MAX_SLOTS
#define FG_DECODE_BATCH_SEQUENCE_MAX 8u
#define FG_DECODE_BATCH_INVALID_SLOT UINT32_MAX
/* Per-step aggregate QSA record traffic cap.  The scan is per-sequence (each
 * sequence scans its own committed index), so it scales with B and context:
 * ~95 MB/token at 4K and ~553 MB/token at 262K for the deployed 12 QSA layers.
 * 640 MiB admits B=2 at short context and shrinks to B=1 by 262K, which is the
 * graceful degradation the depth-B milestone requires. */
#define FG_DECODE_BATCH_QSA_SCAN_BUDGET_BYTES (UINT64_C(640) << 20u)
#define FG_DECODE_BATCH_INDEX_BYTES_PER_TOKEN 148u
#define FG_DECODE_BATCH_SELECT_BLOCKS 512u
#define FG_DECODE_BATCH_ATTENTION_BYTES_PER_TOKEN 1088u

typedef struct fg_decode_batch_sequence {
    uint64_t sequence_id;
    /* Bumped on every committed step; a step snapshot must still match it at
     * commit time, so an interleaved mutation fails closed instead of
     * committing stale state. */
    uint64_t state_generation;
    uint64_t sampler_generation;
    /* Owner-side per-session state namespace; stable for the sequence's life
     * and unique inside a batch. */
    uint32_t state_slot;
    uint32_t token_index;
    uint32_t position[4];
    uint32_t committed_tokens;
    uint32_t context_tokens;
    uint64_t qsa_records[FG_LAYER_COUNT];
    fg_sampler_state sampler;
    bool ready;
    bool in_flight;
    uint64_t ready_since;
    uint64_t steps;
} fg_decode_batch_sequence;

typedef struct fg_decode_batch_slot {
    uint32_t sequence;
    uint32_t state_slot;
    uint32_t token_index;
    uint32_t position[4];
} fg_decode_batch_slot;

typedef struct fg_decode_batch {
    uint32_t slot_count;
    fg_decode_batch_slot slots[FG_DECODE_BATCH_MAX];
} fg_decode_batch;

typedef struct fg_decode_batch_table {
    fg_decode_batch_sequence sequences[FG_DECODE_BATCH_SEQUENCE_MAX];
    uint32_t count;
    uint32_t depth;
    uint64_t step;
    bool restored;
} fg_decode_batch_table;

typedef struct fg_decode_batch_policy {
    uint32_t max_batch;
    uint64_t qsa_scan_budget_bytes;
    uint64_t weight_bytes;
    uint64_t state_bytes_per_token;
    uint64_t wire_bytes_per_token;
    uint32_t qsa_layer_count;
} fg_decode_batch_policy;

void fg_decode_batch_policy_default(fg_decode_batch_policy *policy);
uint64_t fg_decode_batch_qsa_bytes_per_token(const fg_decode_batch_policy *policy,
                                             uint32_t context_tokens);
uint32_t fg_decode_batch_choose_batch(const fg_decode_batch_policy *policy,
                                      uint32_t context_tokens,uint32_t depth);
double fg_decode_batch_bytes_per_token(const fg_decode_batch_policy *policy,
                                       uint32_t batch,uint32_t context_tokens);
double fg_decode_batch_speedup(const fg_decode_batch_policy *policy,uint32_t batch,
                               uint32_t context_tokens);

fg_status fg_decode_batch_table_init(fg_decode_batch_table *table,uint32_t depth,
                                     fg_error *err);
uint32_t fg_decode_batch_table_find(const fg_decode_batch_table *table,
                                    uint64_t sequence_id);
/* Bind a sequence to an owner state slot.  The slot must be free; a sequence
 * that is still in flight cannot be rebound or removed. */
fg_status fg_decode_batch_sequence_enter(fg_decode_batch_table *table,
                                         uint64_t sequence_id,uint32_t state_slot,
                                         fg_error *err);
fg_status fg_decode_batch_sequence_leave(fg_decode_batch_table *table,
                                         uint64_t sequence_id,fg_error *err);
fg_status fg_decode_batch_sequence_ready(fg_decode_batch_table *table,
                                         uint64_t sequence_id,uint64_t now,
                                         fg_error *err);
/* Record a committed frontier advance (prefill or external state restore). */
fg_status fg_decode_batch_sequence_frontier(fg_decode_batch_table *table,
                                            uint64_t sequence_id,
                                            uint32_t committed_tokens,
                                            uint32_t token_index,
                                            const uint32_t position[4],
                                            fg_error *err);
fg_status fg_decode_batch_schedule(fg_decode_batch_table *table,
                                   const fg_decode_batch_policy *policy,uint64_t now,
                                   fg_decode_batch *batch,fg_error *err);

typedef struct fg_decode_batch_outcome {
    bool valid;
    uint32_t next_token;
    float logit;
    uint32_t position[4];
    uint64_t qsa_records[FG_LAYER_COUNT];
    fg_sampler_state sampler;
} fg_decode_batch_outcome;

typedef struct fg_decode_batch_snapshot {
    uint32_t sequence;
    uint64_t state_generation;
    uint64_t sampler_generation;
    uint32_t token_index;
    uint32_t position[4];
    uint32_t committed_tokens;
    uint32_t context_tokens;
    uint64_t qsa_records[FG_LAYER_COUNT];
    fg_sampler_state sampler;
} fg_decode_batch_snapshot;

/* Device-side half of the transaction.  The runtime supplies the existing
 * SESSION_PREPARE/COMMIT/RESTORE calls; the host table mirrors their outcome.
 * Every hook is optional for local tests and pure-host use. */
typedef struct fg_decode_batch_ops {
    fg_status (*prepare)(void *context,uint64_t sequence_id,uint32_t state_slot,
                         fg_error *err);
    fg_status (*commit)(void *context,uint64_t sequence_id,uint32_t state_slot,
                        fg_error *err);
    fg_status (*restore)(void *context,uint64_t sequence_id,uint32_t state_slot,
                         fg_error *err);
    void *context;
} fg_decode_batch_ops;

typedef struct fg_decode_batch_step {
    bool active;
    fg_decode_batch batch;
    fg_decode_batch_outcome outcomes[FG_DECODE_BATCH_MAX];
    fg_decode_batch_snapshot snapshots[FG_DECODE_BATCH_MAX];
    const fg_decode_batch_ops *ops;
    uint32_t prepared;
} fg_decode_batch_step;

fg_status fg_decode_batch_step_begin(fg_decode_batch_table *table,
                                     const fg_decode_batch_policy *policy,
                                     const fg_decode_batch_ops *ops,uint64_t now,
                                     fg_decode_batch_step *step,fg_error *err);
/* Record one slot's sampled outcome.  Every slot in the step must be advanced
 * before a commit; a missing outcome fails the commit without mutating state. */
fg_status fg_decode_batch_step_advance(fg_decode_batch_table *table,
                                       fg_decode_batch_step *step,uint32_t slot,
                                       const fg_decode_batch_outcome *outcome,
                                       fg_error *err);
fg_status fg_decode_batch_step_commit(fg_decode_batch_table *table,
                                      fg_decode_batch_step *step,fg_error *err);
fg_status fg_decode_batch_step_restore(fg_decode_batch_table *table,
                                       fg_decode_batch_step *step,fg_error *err);

#endif
