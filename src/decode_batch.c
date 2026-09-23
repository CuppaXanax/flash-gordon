#include "fg_decode_batch.h"

#include <string.h>

/* Documented byte model of the depth-B ring step (PERFORMANCE_BYTE_BUDGET):
 * weights 6.34 GB read once per step, 0.36 GB of per-sequence state read and
 * written, 379,532 B of ring wire per sequence.  The QSA record scan is
 * per-sequence and grows with context, which is why it gates the batch
 * choice. */
#define FG_DECODE_BATCH_WEIGHT_BYTES UINT64_C(6340000000)
#define FG_DECODE_BATCH_STATE_BYTES UINT64_C(360000000)
#define FG_DECODE_BATCH_WIRE_BYTES UINT64_C(379532)
#define FG_DECODE_BATCH_QSA_LAYERS 12u
#define FG_DECODE_BATCH_QSA_PAGE_RECORD_BYTES 4944u
#define FG_DECODE_BATCH_QSA_COMMIT_BYTES 5080u

void fg_decode_batch_policy_default(fg_decode_batch_policy *policy) {
    if (!policy) return;
    memset(policy, 0, sizeof(*policy));
    policy->max_batch = FG_DECODE_BATCH_MAX;
    policy->qsa_scan_budget_bytes = FG_DECODE_BATCH_QSA_SCAN_BUDGET_BYTES;
    policy->weight_bytes = FG_DECODE_BATCH_WEIGHT_BYTES;
    policy->state_bytes_per_token = FG_DECODE_BATCH_STATE_BYTES;
    policy->wire_bytes_per_token = FG_DECODE_BATCH_WIRE_BYTES;
    policy->qsa_layer_count = FG_DECODE_BATCH_QSA_LAYERS;
}

uint64_t fg_decode_batch_qsa_bytes_per_token(const fg_decode_batch_policy *policy,
                                             uint32_t context_tokens) {
    uint64_t layers = policy && policy->qsa_layer_count ?
        policy->qsa_layer_count : FG_DECODE_BATCH_QSA_LAYERS;
    uint64_t index_scan = (uint64_t)context_tokens * FG_DECODE_BATCH_INDEX_BYTES_PER_TOKEN;
    uint64_t blocks = (uint64_t)context_tokens / 4u;
    if (blocks > FG_DECODE_BATCH_SELECT_BLOCKS) blocks = FG_DECODE_BATCH_SELECT_BLOCKS;
    uint64_t selection = blocks * FG_DECODE_BATCH_QSA_PAGE_RECORD_BYTES * 2u;
    uint64_t attention_tokens = blocks * 4u;
    if (attention_tokens > context_tokens) attention_tokens = context_tokens;
    uint64_t attention = attention_tokens * FG_DECODE_BATCH_ATTENTION_BYTES_PER_TOKEN;
    uint64_t per_layer = index_scan + selection + attention +
        FG_DECODE_BATCH_QSA_COMMIT_BYTES;
    return per_layer * layers;
}

uint32_t fg_decode_batch_choose_batch(const fg_decode_batch_policy *policy,
                                      uint32_t context_tokens, uint32_t depth) {
    uint32_t cap = policy ? policy->max_batch : FG_DECODE_BATCH_MAX;
    if (!cap || cap > FG_DECODE_BATCH_MAX) cap = FG_DECODE_BATCH_MAX;
    if (depth && depth < cap) cap = depth;
    if (!cap) cap = 1u;
    uint64_t scan = fg_decode_batch_qsa_bytes_per_token(policy, context_tokens);
    uint64_t budget = policy && policy->qsa_scan_budget_bytes ?
        policy->qsa_scan_budget_bytes : FG_DECODE_BATCH_QSA_SCAN_BUDGET_BYTES;
    for (uint32_t batch = cap; batch > 1u; batch--)
        if ((uint64_t)batch * scan <= budget) return batch;
    return 1u;
}

double fg_decode_batch_bytes_per_token(const fg_decode_batch_policy *policy,
                                       uint32_t batch, uint32_t context_tokens) {
    if (!batch) return 0.0;
    uint64_t weights = policy ? policy->weight_bytes : FG_DECODE_BATCH_WEIGHT_BYTES;
    uint64_t state = policy ? policy->state_bytes_per_token : FG_DECODE_BATCH_STATE_BYTES;
    uint64_t wire = policy ? policy->wire_bytes_per_token : FG_DECODE_BATCH_WIRE_BYTES;
    return (double)weights / (double)batch + (double)state +
        (double)wire / (double)batch +
        (double)fg_decode_batch_qsa_bytes_per_token(policy, context_tokens);
}

double fg_decode_batch_speedup(const fg_decode_batch_policy *policy, uint32_t batch,
                               uint32_t context_tokens) {
    double single = fg_decode_batch_bytes_per_token(policy, 1u, context_tokens);
    double batched = fg_decode_batch_bytes_per_token(policy, batch, context_tokens);
    if (batched <= 0.0) return 0.0;
    return single / batched;
}

static fg_decode_batch_sequence *find_sequence(fg_decode_batch_table *table,
                                               uint64_t sequence_id) {
    if (!table || !sequence_id) return NULL;
    for (uint32_t i = 0; i < table->count; i++)
        if (table->sequences[i].sequence_id == sequence_id) return &table->sequences[i];
    return NULL;
}

fg_status fg_decode_batch_table_init(fg_decode_batch_table *table, uint32_t depth,
                                     fg_error *err) {
    if (!table || !depth || depth > FG_DECODE_BATCH_MAX) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch table depth");
        return FG_ERR_ARGUMENT;
    }
    memset(table, 0, sizeof(*table));
    table->depth = depth;
    return FG_OK;
}

uint32_t fg_decode_batch_table_find(const fg_decode_batch_table *table,
                                    uint64_t sequence_id) {
    if (!table || !sequence_id) return FG_DECODE_BATCH_INVALID_SLOT;
    for (uint32_t i = 0; i < table->count; i++)
        if (table->sequences[i].sequence_id == sequence_id) return i;
    return FG_DECODE_BATCH_INVALID_SLOT;
}

static bool state_slot_used(const fg_decode_batch_table *table, uint32_t state_slot,
                            uint32_t except) {
    for (uint32_t i = 0; i < table->count; i++) {
        if (i == except) continue;
        if (table->sequences[i].state_slot == state_slot) return true;
    }
    return false;
}

fg_status fg_decode_batch_sequence_enter(fg_decode_batch_table *table,
                                         uint64_t sequence_id,uint32_t state_slot,
                                         fg_error *err) {
    if (!table || !table->depth || !sequence_id) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch sequence");
        return FG_ERR_ARGUMENT;
    }
    if (state_slot >= table->depth) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "owner state slot %u is outside the batch depth %u",
                     state_slot, table->depth);
        return FG_ERR_ARGUMENT;
    }
    if (find_sequence(table, sequence_id)) {
        fg_error_set(err, FG_ERR_MISMATCH, "decode batch sequence %llu already exists",
                     (unsigned long long)sequence_id);
        return FG_ERR_MISMATCH;
    }
    if (state_slot_used(table, state_slot, FG_DECODE_BATCH_SEQUENCE_MAX)) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "owner state slot %u is already owned by another sequence",
                     state_slot);
        return FG_ERR_MISMATCH;
    }
    if (table->count >= FG_DECODE_BATCH_SEQUENCE_MAX) {
        fg_error_set(err, FG_ERR_LIMIT, "decode batch table is full");
        return FG_ERR_LIMIT;
    }
    fg_decode_batch_sequence *sequence = &table->sequences[table->count++];
    memset(sequence, 0, sizeof(*sequence));
    sequence->sequence_id = sequence_id;
    sequence->state_slot = state_slot;
    /* Deterministic per-sequence stream; the runtime may re-seed it when the
     * request's sampler config arrives. */
    fg_sampler_state_init(&sequence->sampler, sequence_id);
    return FG_OK;
}

fg_status fg_decode_batch_sequence_leave(fg_decode_batch_table *table,
                                         uint64_t sequence_id,fg_error *err) {
    if (!table || !sequence_id) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch sequence");
        return FG_ERR_ARGUMENT;
    }
    for (uint32_t i = 0; i < table->count; i++) {
        if (table->sequences[i].sequence_id != sequence_id) continue;
        if (table->sequences[i].in_flight) {
            fg_error_set(err, FG_ERR_MISMATCH,
                         "decode batch sequence %llu is in flight",
                         (unsigned long long)sequence_id);
            return FG_ERR_MISMATCH;
        }
        if (i + 1u < table->count)
            table->sequences[i] = table->sequences[table->count - 1u];
        table->count--;
        return FG_OK;
    }
    fg_error_set(err, FG_ERR_MISMATCH, "decode batch sequence %llu is not bound",
                 (unsigned long long)sequence_id);
    return FG_ERR_MISMATCH;
}

fg_status fg_decode_batch_sequence_ready(fg_decode_batch_table *table,
                                         uint64_t sequence_id,uint64_t now,
                                         fg_error *err) {
    fg_decode_batch_sequence *sequence = find_sequence(table, sequence_id);
    if (!sequence) {
        fg_error_set(err, FG_ERR_MISMATCH, "decode batch sequence %llu is not bound",
                     (unsigned long long)sequence_id);
        return FG_ERR_MISMATCH;
    }
    if (sequence->in_flight) {
        fg_error_set(err, FG_ERR_MISMATCH, "decode batch sequence %llu is in flight",
                     (unsigned long long)sequence_id);
        return FG_ERR_MISMATCH;
    }
    sequence->ready = true;
    sequence->ready_since = now;
    return FG_OK;
}

fg_status fg_decode_batch_sequence_frontier(fg_decode_batch_table *table,
                                            uint64_t sequence_id,
                                            uint32_t committed_tokens,
                                            uint32_t token_index,
                                            const uint32_t position[4],
                                            fg_error *err) {
    fg_decode_batch_sequence *sequence = find_sequence(table, sequence_id);
    if (!sequence || !position) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch frontier");
        return FG_ERR_ARGUMENT;
    }
    if (sequence->in_flight) {
        fg_error_set(err, FG_ERR_MISMATCH, "decode batch sequence %llu is in flight",
                     (unsigned long long)sequence_id);
        return FG_ERR_MISMATCH;
    }
    if (committed_tokens > FG_MAX_CONTEXT || token_index >= FG_MAX_CONTEXT) {
        fg_error_set(err, FG_ERR_LIMIT, "decode batch frontier exceeds the context");
        return FG_ERR_LIMIT;
    }
    sequence->committed_tokens = committed_tokens;
    sequence->context_tokens = committed_tokens;
    sequence->token_index = token_index;
    for (uint32_t axis = 0; axis < 4u; axis++) sequence->position[axis] = position[axis];
    for (uint32_t layer = 0; layer < FG_LAYER_COUNT; layer++)
        if ((layer & 3u) == 3u) sequence->qsa_records[layer] = committed_tokens;
    return FG_OK;
}

fg_status fg_decode_batch_schedule(fg_decode_batch_table *table,
                                   const fg_decode_batch_policy *policy,uint64_t now,
                                   fg_decode_batch *batch,fg_error *err) {
    if (!table || !table->depth || !batch) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch schedule arguments");
        return FG_ERR_ARGUMENT;
    }
    memset(batch, 0, sizeof(*batch));
    uint32_t candidates = 0, worst_context = 0;
    for (uint32_t i = 0; i < table->count; i++) {
        const fg_decode_batch_sequence *sequence = &table->sequences[i];
        if (!sequence->ready || sequence->in_flight) continue;
        candidates++;
        uint32_t context = sequence->context_tokens;
        if (sequence->committed_tokens > context) context = sequence->committed_tokens;
        if (sequence->token_index > context) context = sequence->token_index;
        if (context > worst_context) worst_context = context;
    }
    if (!candidates) {
        fg_error_set(err, FG_ERR_UNAVAILABLE, "decode batch has no ready sequence");
        return FG_ERR_UNAVAILABLE;
    }
    uint32_t limit = fg_decode_batch_choose_batch(policy, worst_context, table->depth);
    if (limit > candidates) limit = candidates;
    for (uint32_t chosen = 0; chosen < limit; chosen++) {
        uint32_t best = FG_DECODE_BATCH_INVALID_SLOT;
        for (uint32_t i = 0; i < table->count; i++) {
            const fg_decode_batch_sequence *sequence = &table->sequences[i];
            if (!sequence->ready || sequence->in_flight) continue;
            if (best == FG_DECODE_BATCH_INVALID_SLOT ||
                sequence->ready_since < table->sequences[best].ready_since ||
                (sequence->ready_since == table->sequences[best].ready_since && i < best))
                best = i;
        }
        if (best == FG_DECODE_BATCH_INVALID_SLOT) break;
        fg_decode_batch_sequence *sequence = &table->sequences[best];
        fg_decode_batch_slot *slot = &batch->slots[batch->slot_count++];
        slot->sequence = best;
        slot->state_slot = sequence->state_slot;
        slot->token_index = sequence->token_index;
        for (uint32_t axis = 0; axis < 4u; axis++) slot->position[axis] = sequence->position[axis];
        sequence->ready = false;
        sequence->in_flight = true;
    }
    if (!batch->slot_count) {
        fg_error_set(err, FG_ERR_UNAVAILABLE, "decode batch assembly produced no slot");
        return FG_ERR_UNAVAILABLE;
    }
    (void)now;
    table->restored = false;
    return FG_OK;
}

static void step_release_slots(fg_decode_batch_table *table,
                               const fg_decode_batch *batch, bool ready) {
    for (uint32_t slot = 0; slot < batch->slot_count; slot++) {
        fg_decode_batch_sequence *sequence = &table->sequences[batch->slots[slot].sequence];
        sequence->in_flight = false;
        sequence->ready = ready;
    }
}

static void step_snapshot(fg_decode_batch_step *step, uint32_t slot,
                          const fg_decode_batch_sequence *sequence) {
    fg_decode_batch_snapshot *snapshot = &step->snapshots[slot];
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->sequence = step->batch.slots[slot].sequence;
    snapshot->state_generation = sequence->state_generation;
    snapshot->sampler_generation = sequence->sampler_generation;
    snapshot->token_index = sequence->token_index;
    snapshot->committed_tokens = sequence->committed_tokens;
    snapshot->context_tokens = sequence->context_tokens;
    for (uint32_t axis = 0; axis < 4u; axis++)
        snapshot->position[axis] = sequence->position[axis];
    memcpy(snapshot->qsa_records, sequence->qsa_records, sizeof(snapshot->qsa_records));
    snapshot->sampler = sequence->sampler;
}

fg_status fg_decode_batch_step_begin(fg_decode_batch_table *table,
                                     const fg_decode_batch_policy *policy,
                                     const fg_decode_batch_ops *ops,uint64_t now,
                                     fg_decode_batch_step *step,fg_error *err) {
    if (!table || !step) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch step");
        return FG_ERR_ARGUMENT;
    }
    if (step->active) {
        fg_error_set(err, FG_ERR_MISMATCH, "decode batch step is already active");
        return FG_ERR_MISMATCH;
    }
    memset(step, 0, sizeof(*step));
    step->ops = ops;
    fg_status status = fg_decode_batch_schedule(table, policy, now, &step->batch, err);
    if (status != FG_OK) return status;
    for (uint32_t slot = 0; slot < step->batch.slot_count; slot++)
        step_snapshot(step, slot, &table->sequences[step->batch.slots[slot].sequence]);
    if (ops && ops->prepare) {
        for (uint32_t slot = 0; slot < step->batch.slot_count; slot++) {
            const fg_decode_batch_slot *entry = &step->batch.slots[slot];
            fg_decode_batch_sequence *sequence = &table->sequences[entry->sequence];
            status = ops->prepare(ops->context, sequence->sequence_id, entry->state_slot,
                                  err);
            if (status != FG_OK) {
                for (uint32_t done = 0; done < slot; done++) {
                    fg_decode_batch_sequence *other =
                        &table->sequences[step->batch.slots[done].sequence];
                    fg_error ignored = {0};
                    ops->restore(ops->context, other->sequence_id,
                                 step->batch.slots[done].state_slot, &ignored);
                }
                step_release_slots(table, &step->batch, true);
                memset(step, 0, sizeof(*step));
                return status;
            }
            step->prepared = slot + 1u;
        }
    }
    step->active = true;
    return FG_OK;
}

fg_status fg_decode_batch_step_advance(fg_decode_batch_table *table,
                                       fg_decode_batch_step *step,uint32_t slot,
                                       const fg_decode_batch_outcome *outcome,
                                       fg_error *err) {
    if (!table || !step || !step->active || !outcome) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch advance");
        return FG_ERR_ARGUMENT;
    }
    if (slot >= step->batch.slot_count) {
        fg_error_set(err, FG_ERR_ARGUMENT, "decode batch slot %u is outside the step",
                     slot);
        return FG_ERR_ARGUMENT;
    }
    if (outcome->next_token >= FG_Q38_VOCAB_SIZE) {
        fg_error_set(err, FG_ERR_FORMAT, "decode batch slot %u token %u is outside "
                     "the vocabulary", slot, outcome->next_token);
        return FG_ERR_FORMAT;
    }
    step->outcomes[slot] = *outcome;
    step->outcomes[slot].valid = true;
    return FG_OK;
}

fg_status fg_decode_batch_step_commit(fg_decode_batch_table *table,
                                      fg_decode_batch_step *step,fg_error *err) {
    if (!table || !step || !step->active) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch commit");
        return FG_ERR_ARGUMENT;
    }
    for (uint32_t slot = 0; slot < step->batch.slot_count; slot++) {
        const fg_decode_batch_snapshot *snapshot = &step->snapshots[slot];
        const fg_decode_batch_sequence *sequence =
            &table->sequences[snapshot->sequence];
        if (!step->outcomes[slot].valid) {
            fg_error_set(err, FG_ERR_MISMATCH,
                         "decode batch slot %u has no outcome", slot);
            return FG_ERR_MISMATCH;
        }
        if (sequence->state_generation != snapshot->state_generation ||
            sequence->sampler_generation != snapshot->sampler_generation) {
            fg_error_set(err, FG_ERR_MISMATCH,
                         "decode batch sequence %llu moved during the step",
                         (unsigned long long)sequence->sequence_id);
            return FG_ERR_MISMATCH;
        }
    }
    const fg_decode_batch_ops *ops = step->ops;
    if (ops && ops->commit) {
        for (uint32_t slot = 0; slot < step->batch.slot_count; slot++) {
            const fg_decode_batch_snapshot *snapshot = &step->snapshots[slot];
            const fg_decode_batch_sequence *sequence =
                &table->sequences[snapshot->sequence];
            fg_status status = ops->commit(ops->context, sequence->sequence_id,
                                           step->batch.slots[slot].state_slot, err);
            if (status != FG_OK) {
                fg_status restored = fg_decode_batch_step_restore(table, step, err);
                return restored == FG_OK ? status : restored;
            }
        }
    }
    for (uint32_t slot = 0; slot < step->batch.slot_count; slot++) {
        fg_decode_batch_sequence *sequence =
            &table->sequences[step->snapshots[slot].sequence];
        const fg_decode_batch_outcome *outcome = &step->outcomes[slot];
        sequence->committed_tokens++;
        sequence->context_tokens++;
        sequence->token_index++;
        for (uint32_t axis = 0; axis < 4u; axis++)
            sequence->position[axis] = outcome->position[axis];
        memcpy(sequence->qsa_records, outcome->qsa_records,
               sizeof(sequence->qsa_records));
        sequence->sampler = outcome->sampler;
        sequence->sampler_generation++;
        sequence->state_generation++;
        sequence->steps++;
        sequence->ready = false;
        sequence->in_flight = false;
    }
    table->step++;
    table->restored = false;
    step->active = false;
    return FG_OK;
}

fg_status fg_decode_batch_step_restore(fg_decode_batch_table *table,
                                       fg_decode_batch_step *step,fg_error *err) {
    if (!table || !step) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid decode batch restore");
        return FG_ERR_ARGUMENT;
    }
    if (!step->active) return FG_OK;
    const fg_decode_batch_ops *ops = step->ops;
    fg_status first_error = FG_OK;
    if (ops && ops->restore) {
        for (uint32_t slot = 0; slot < step->batch.slot_count; slot++) {
            const fg_decode_batch_snapshot *snapshot = &step->snapshots[slot];
            const fg_decode_batch_sequence *sequence =
                &table->sequences[snapshot->sequence];
            fg_error hook_error = {0};
            fg_status status = ops->restore(ops->context, sequence->sequence_id,
                                            step->batch.slots[slot].state_slot,
                                            &hook_error);
            if (status != FG_OK && first_error == FG_OK) {
                first_error = status;
                if (err) *err = hook_error;
            }
        }
    }
    for (uint32_t slot = 0; slot < step->batch.slot_count; slot++) {
        const fg_decode_batch_snapshot *snapshot = &step->snapshots[slot];
        fg_decode_batch_sequence *sequence = &table->sequences[snapshot->sequence];
        sequence->state_generation = snapshot->state_generation;
        sequence->sampler_generation = snapshot->sampler_generation;
        sequence->token_index = snapshot->token_index;
        sequence->committed_tokens = snapshot->committed_tokens;
        sequence->context_tokens = snapshot->context_tokens;
        for (uint32_t axis = 0; axis < 4u; axis++)
            sequence->position[axis] = snapshot->position[axis];
        memcpy(sequence->qsa_records, snapshot->qsa_records,
               sizeof(sequence->qsa_records));
        sequence->sampler = snapshot->sampler;
        sequence->ready = true;
        sequence->in_flight = false;
    }
    memset(step->outcomes, 0, sizeof(step->outcomes));
    table->restored = true;
    step->active = false;
    return first_error;
}
