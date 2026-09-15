#ifndef FLASH_GORDON_PROTOCOL_H
#define FLASH_GORDON_PROTOCOL_H

#include "fg_manifest.h"
#include "fg_q38_schema.h"
#include "fg_session.h"
#include "fg_sampler.h"

#define FG_FRAME_MAGIC UINT32_C(0x31474646) /* FFG1 */
#define FG_MAX_FRAME_BYTES (64u * 1024u * 1024u)
#define FG_Q8K_BLOCK_BYTES 296u
#define FG_Q8K_BLOCK_COUNT (FG_HIDDEN_SIZE / 256u)
#define FG_Q8K_ACTIVATION_BYTES (FG_Q8K_BLOCK_BYTES * FG_Q8K_BLOCK_COUNT)
#define FG_DECODE_WORK_BYTES (8u + FG_TOP_K * 2u + FG_TOP_K + FG_TOP_K * 4u + FG_Q8K_ACTIVATION_BYTES)
#define FG_EXPERT_RESULT_ENTRY_BYTES (4u + FG_HIDDEN_SIZE * 4u)
#define FG_EXPERT_RESULT_MAX_BYTES (8u + FG_TOP_K * FG_EXPERT_RESULT_ENTRY_BYTES)
#define FG_EXPERT_RESULT_SINGLE_BYTES (8u + FG_EXPERT_RESULT_ENTRY_BYTES)
#define FG_HYPER_WIDTH (FG_HIDDEN_SIZE*4u)
#define FG_NGRAM_EMBED_VALUES (FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH)
#define FG_LAYER_WORK_LEGACY_HEADER_BYTES 20u
#define FG_LAYER_WORK_TEXT_HEADER_BYTES 24u
#define FG_LAYER_WORK_FOUR_AXIS_HEADER_BYTES 28u
#define FG_LAYER_WORK_HEADER_BYTES FG_LAYER_WORK_TEXT_HEADER_BYTES
#define FG_LAYER_WORK_BASE_BYTES (FG_LAYER_WORK_TEXT_HEADER_BYTES+FG_HYPER_WIDTH*4u)
#define FG_LAYER_WORK_FOUR_AXIS_BASE_BYTES (FG_LAYER_WORK_FOUR_AXIS_HEADER_BYTES+FG_HYPER_WIDTH*4u)
#define FG_LAYER_WORK_TEXT_MAX_BYTES (FG_LAYER_WORK_BASE_BYTES+FG_NGRAM_EMBED_VALUES*4u)
#define FG_LAYER_WORK_MAX_BYTES (FG_LAYER_WORK_FOUR_AXIS_BASE_BYTES+FG_NGRAM_EMBED_VALUES*4u)
#define FG_LAYER_RESULT_BYTES (8u+FG_HYPER_WIDTH*4u)
#define FG_LAYER_RESULT_BF16_BYTES (8u+FG_HYPER_WIDTH*2u)
#define FG_OUTPUT_WORK_HEADER_BYTES 40u
#define FG_OUTPUT_WORK_BYTES (FG_OUTPUT_WORK_HEADER_BYTES+FG_HYPER_WIDTH*4u)
#define FG_OUTPUT_RESULT_BYTES 16u
/* Sampler-only preamble for the direct final-block -> output-owner handoff:
 * the output owner runs the head as soon as the matching 40 KiB hidden message
 * arrives, so the sampler route rides the control channel and the bulk hop
 * never touches rank 0. */
#define FG_OUTPUT_CONFIG_BYTES FG_OUTPUT_WORK_HEADER_BYTES
#define FG_OUTPUT_HISTORY_HEADER_BYTES 8u
#define FG_OUTPUT_HISTORY_MAX_BYTES (FG_OUTPUT_HISTORY_HEADER_BYTES+FG_NATIVE_CONTEXT*4u)
#define FG_LAYER_WORK_HAS_NGRAM 1u
#define FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY 2u
#define FG_LAYER_WORK_FLAG_BF16_HYPER 4u
#define FG_PREFILL_MAX_TOKENS 512u
#define FG_PREFILL_MAX_PAIRS (FG_PREFILL_MAX_TOKENS*FG_TOP_K)
#define FG_PREFILL_WORK_HEADER_BYTES 16u
#define FG_PREFILL_PAIR_BYTES 12u
#define FG_PREFILL_RESULT_HEADER_BYTES 16u
#define FG_PREFILL_RESULT_PAIR_BYTES 4u
#define FG_PREFILL_RESULT_VERSION 3u
#define FG_PREFILL_REDUCE_TILE_TOKENS 16u
#define FG_PREFILL_WORK_VERSION 1u
#define FG_PREFILL_WORK_MAX_BYTES (FG_PREFILL_WORK_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*FG_Q8K_ACTIVATION_BYTES+FG_PREFILL_MAX_PAIRS*FG_PREFILL_PAIR_BYTES)
#define FG_PREFILL_RESULT_MAX_BYTES (FG_PREFILL_RESULT_HEADER_BYTES+FG_PREFILL_MAX_PAIRS*FG_PREFILL_RESULT_PAIR_BYTES+FG_PREFILL_MAX_TOKENS*FG_HIDDEN_SIZE*4u)
#define FG_PREFILL_LAYER_HEADER_BYTES 16u
#define FG_PREFILL_LAYER_WORK_MAX_BYTES (FG_PREFILL_LAYER_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*4u*4u+FG_PREFILL_MAX_TOKENS*FG_HYPER_WIDTH*4u+FG_PREFILL_MAX_TOKENS*FG_NGRAM_EMBED_VALUES*4u)
#define FG_PREFILL_LAYER_RESULT_MAX_BYTES (FG_PREFILL_LAYER_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*FG_HYPER_WIDTH*4u)
#define FG_QSA_BLOCK_WORK_LEGACY_HEADER_BYTES 20u
#define FG_QSA_BLOCK_WORK_TEXT_HEADER_BYTES 24u
#define FG_QSA_BLOCK_WORK_FOUR_AXIS_HEADER_BYTES 28u
#define FG_QSA_BLOCK_WORK_TEXT_BYTES (FG_QSA_BLOCK_WORK_TEXT_HEADER_BYTES+FG_HIDDEN_SIZE*4u)
#define FG_QSA_BLOCK_WORK_MAX_BYTES (FG_QSA_BLOCK_WORK_FOUR_AXIS_HEADER_BYTES+FG_HIDDEN_SIZE*4u)
#define FG_QSA_BLOCK_RESULT_BYTES (8u+FG_HIDDEN_SIZE*4u)
#define FG_QSA_BLOCK_PREFILL_HEADER_BYTES 16u
#define FG_QSA_BLOCK_PREFILL_WORK_MAX_BYTES (FG_QSA_BLOCK_PREFILL_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*4u*4u+FG_PREFILL_MAX_TOKENS*FG_HIDDEN_SIZE*4u)
#define FG_QSA_BLOCK_PREFILL_RESULT_MAX_BYTES (FG_QSA_BLOCK_PREFILL_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*FG_HIDDEN_SIZE*4u)
#define FG_QSA_PAGE_PROTOCOL_VERSION 1u
#define FG_QSA_PAGE_RECORD_BYTES (FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES)
#define FG_QSA_PAGE_ENTRY_HEADER_BYTES 8u
#define FG_QSA_PAGE_ENTRY_BYTES (FG_QSA_PAGE_ENTRY_HEADER_BYTES+FG_QSA_PAGE_RECORD_BYTES)
#define FG_QSA_PAGE_BATCH_HEADER_BYTES 12u
#define FG_QSA_OWNER_LAYER_COUNT 6u
#define FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES (FG_PREFILL_MAX_TOKENS/FG_Q38_QSA_COMPRESS_RATIO)
#define FG_QSA_PAGE_APPEND_MAX_PAGES (FG_QSA_OWNER_LAYER_COUNT*(FG_PREFILL_MAX_TOKENS/FG_Q38_QSA_COMPRESS_RATIO))
#define FG_QSA_PAGE_APPEND_MAX_BYTES (FG_QSA_PAGE_BATCH_HEADER_BYTES+FG_QSA_PAGE_APPEND_MAX_PAGES*FG_QSA_PAGE_ENTRY_BYTES)
#define FG_QSA_PAGE_FETCH_MAX_PAGES (FG_Q38_INDEX_BUDGET/FG_Q38_QSA_COMPRESS_RATIO)
#define FG_QSA_PAGE_FETCH_MAX_BYTES (FG_QSA_PAGE_BATCH_HEADER_BYTES+FG_QSA_PAGE_FETCH_MAX_PAGES*FG_QSA_PAGE_ENTRY_HEADER_BYTES)
#define FG_QSA_PAGE_RESULT_MAX_BYTES (FG_QSA_PAGE_BATCH_HEADER_BYTES+FG_QSA_PAGE_FETCH_MAX_PAGES*FG_QSA_PAGE_ENTRY_BYTES)
#define FG_QSA_PAGE_BARRIER_BYTES 8u
/* Ring decode-state handoff: GDN conv/recurrent state plus the layer-1 PLE
 * convolution state, fetched per layer from the block owner after ring prefill
 * so rank 0's local decode executor starts from the distributed frontier. */
/* The GDN convolution state is four taps per channel over the full 10240-wide
 * qkv projection: FG_HYPER_WIDTH floats per tap. */
#define FG_GDN_STATE_CONV_BYTES (FG_HYPER_WIDTH*4u*4u)
#define FG_GDN_STATE_RECURRENT_BYTES (48u*128u*128u*4u)
#define FG_GDN_STATE_PLE_BYTES (FG_HYPER_WIDTH*9u*4u)
#define FG_GDN_STATE_FETCH_BYTES 8u
#define FG_GDN_STATE_RESULT_BASE_BYTES (16u+FG_GDN_STATE_CONV_BYTES+FG_GDN_STATE_RECURRENT_BYTES)
#define FG_GDN_STATE_RESULT_MAX_BYTES (FG_GDN_STATE_RESULT_BASE_BYTES+FG_GDN_STATE_PLE_BYTES)
#define FG_GDN_STATE_RESULT_HAS_PLE 1u
#define FG_NGRAM_SHARD_MAX_ITEMS FG_NGRAM_HEAD_COUNT
#define FG_NGRAM_WIRE_ROW_BYTES 90u
#define FG_NGRAM_WORK_MAX_BYTES (8u+FG_NGRAM_SHARD_MAX_ITEMS*9u)
#define FG_NGRAM_RESULT_MAX_BYTES (8u+FG_NGRAM_SHARD_MAX_ITEMS*(1u+FG_NGRAM_WIRE_ROW_BYTES))
#define FG_OWNER_SESSION_CONTROL_VERSION 2u
#define FG_OWNER_SESSION_CONTROL_BYTES 184u

typedef enum fg_message_type {
    FG_MSG_HELLO = 1,
    FG_MSG_READY = 2,
    FG_MSG_DECODE_WORK = 3,
    FG_MSG_PREFILL_WORK = 4,
    FG_MSG_EXPERT_RESULT = 5,
    FG_MSG_CONTROL = 6,
    FG_MSG_ERROR = 7,
    FG_MSG_SESSION_BEGIN = 8,
    FG_MSG_SESSION_READY = 9,
    FG_MSG_LAYER_WORK = 10,
    FG_MSG_LAYER_RESULT = 11,
    FG_MSG_OUTPUT_WORK = 12,
    FG_MSG_OUTPUT_RESULT = 13,
    FG_MSG_PREFILL_RESULT = 14,
    FG_MSG_PREFILL_LAYER_WORK = 15,
    FG_MSG_PREFILL_LAYER_RESULT = 16,
    FG_MSG_NGRAM_WORK = 17,
    FG_MSG_NGRAM_RESULT = 18,
    FG_MSG_SESSION_PREPARE = 19,
    FG_MSG_SESSION_PREPARED = 20,
    FG_MSG_SESSION_COMMIT = 21,
    FG_MSG_SESSION_COMMITTED = 22,
    FG_MSG_SESSION_RESTORE = 23,
    FG_MSG_SESSION_RESTORED = 24,
    FG_MSG_QSA_BLOCK_WORK = 25,
    FG_MSG_QSA_BLOCK_RESULT = 26,
    FG_MSG_QSA_BLOCK_PREFILL_WORK = 27,
    FG_MSG_QSA_BLOCK_PREFILL_RESULT = 28,
    FG_MSG_QSA_PAGE_APPEND = 29,
    FG_MSG_QSA_PAGE_BARRIER = 30,
    FG_MSG_QSA_PAGE_BARRIER_ACK = 31,
    FG_MSG_QSA_PAGE_FETCH = 32,
    FG_MSG_QSA_PAGE_RESULT = 33,
    /* IDs 34–39 are retired and must not be reused. */
    FG_MSG_OUTPUT_HISTORY = 40,
    FG_MSG_OUTPUT_HISTORY_ACK = 41,
    FG_MSG_GDN_STATE_FETCH = 42,
    FG_MSG_GDN_STATE_RESULT = 43,
    /* Ring decode: one token's hyper state is handed along the block chain.
     * The payload reuses the single-token fg_layer_work/fg_layer_result wire
     * contract (40 KiB hyper plus optional layer-1 n-gram embedding). */
    FG_MSG_DECODE_LAYER_WORK = 44,
    FG_MSG_DECODE_LAYER_RESULT = 45,
    /* Direct output handoff: rank 0 ships only the sampler config for a decode
     * token on the control channel, and the final block owner ships the 40 KiB
     * hyper state straight to the output owner under FG_MSG_OUTPUT_HIDDEN.
     * Both sides must agree, so the direct route is selected from the manifest
     * (final owner != output owner) plus the FG_DECODE_DIRECT_OUTPUT env. */
    FG_MSG_OUTPUT_CONFIG = 46,
    FG_MSG_OUTPUT_HIDDEN = 47,
    /* Split output head: the final block owner also ships the same 40 KiB hyper
     * state to rank 0 under FG_MSG_OUTPUT_SLICE, rank 0 reduces its vocabulary
     * slice and returns the per-slice argmax under FG_MSG_OUTPUT_PARTIAL. */
    FG_MSG_OUTPUT_SLICE = 48,
    FG_MSG_OUTPUT_PARTIAL = 49,
    FG_MSG_OUTPUT_SLICE_HIDDEN = 50
} fg_message_type;

typedef struct fg_gdn_state_fetch {
    uint32_t layer;
    uint32_t frontier;
} fg_gdn_state_fetch;

typedef struct fg_gdn_state_result {
    uint8_t source_rank;
    uint32_t layer;
    uint32_t frontier;
    const float *conv;
    const float *recurrent;
    const float *ple;
} fg_gdn_state_result;

fg_status fg_gdn_state_fetch_encode(uint8_t output[FG_GDN_STATE_FETCH_BYTES],
                                    const fg_gdn_state_fetch *fetch,fg_error *err);
fg_status fg_gdn_state_fetch_decode(fg_gdn_state_fetch *fetch,const uint8_t *payload,
                                    uint32_t bytes,fg_error *err);
fg_status fg_gdn_state_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                     const fg_gdn_state_result *result,fg_error *err);
fg_status fg_gdn_state_result_decode(fg_gdn_state_result *result,const uint8_t *payload,
                                     uint32_t bytes,fg_error *err);

typedef enum fg_owner_session_operation {
    FG_OWNER_SESSION_BEGIN = 1,
    FG_OWNER_SESSION_READY = 2,
    FG_OWNER_SESSION_PREPARE = 3,
    FG_OWNER_SESSION_PREPARED = 4,
    FG_OWNER_SESSION_COMMIT = 5,
    FG_OWNER_SESSION_COMMITTED = 6,
    FG_OWNER_SESSION_RESTORE = 7,
    FG_OWNER_SESSION_RESTORED = 8
} fg_owner_session_operation;

typedef struct fg_owner_session_control {
    uint16_t version;
    uint8_t operation;
    uint8_t rank;
    fg_position_mode position_mode;
    uint8_t flags;
    uint64_t session_nonce;
    uint64_t generation;
    uint64_t committed_tokens;
    uint8_t identity_sha256[32];
    uint8_t frontier_sha256[32];
    uint8_t state_format_sha256[32];
    uint8_t state_sha256[32];
    uint32_t logical_context_tokens;
    uint32_t gpu_index_tokens;
    uint32_t qsa_hot_tokens;
    uint64_t qsa_page_cache_bytes;
} fg_owner_session_control;

typedef struct fg_frame_header {
    uint32_t magic_be;
    uint16_t version_be;
    uint16_t type_be;
    uint32_t bytes_be;
    uint32_t request_hi_be;
    uint32_t request_lo_be;
    uint32_t sequence_be;
    uint32_t flags_be;
    uint32_t crc32c_be;
} fg_frame_header;

typedef struct fg_decode_work {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t selected_count;
    uint32_t position;
    uint16_t expert_ids[FG_TOP_K];
    uint8_t routing_slots[FG_TOP_K];
    float gates[FG_TOP_K];
    uint8_t activation_q8k[FG_Q8K_ACTIVATION_BYTES];
} fg_decode_work;

typedef struct fg_expert_route {
    uint8_t destination_rank;
    uint8_t selected_count;
    uint16_t global_expert_ids[FG_TOP_K];
    uint16_t local_expert_ids[FG_TOP_K];
    uint8_t routing_slots[FG_TOP_K];
    float gates[FG_TOP_K];
} fg_expert_route;

typedef struct fg_expert_result {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t selected_count;
    uint32_t position;
    uint8_t routing_slots[FG_TOP_K];
    float outputs[FG_TOP_K][FG_HIDDEN_SIZE];
} fg_expert_result;

typedef struct fg_prefill_pair {
    uint16_t token_slot;
    uint16_t expert_id;
    uint8_t routing_slot;
    float gate;
} fg_prefill_pair;

typedef struct fg_prefill_work {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t first_position;
    uint16_t token_count;
    uint16_t pair_count;
    uint8_t *activations_q8k;
    fg_prefill_pair *pairs;
} fg_prefill_work;

typedef struct fg_prefill_route {
    uint8_t destination_rank;
    uint16_t pair_count;
    fg_prefill_pair *pairs;
} fg_prefill_route;

typedef struct fg_prefill_result_pair {
    uint16_t token_slot;
    uint8_t routing_slot;
} fg_prefill_result_pair;

typedef struct fg_prefill_result {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t contributor_mask;
    uint32_t first_position;
    uint16_t token_count;
    uint16_t pair_count;
    fg_prefill_result_pair *pairs;
    /* Weighted subtree sum, token-major; mask and pairs prove coverage. */
    float *outputs;
} fg_prefill_result;

typedef struct fg_prefill_layer_work {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t flags;
    fg_position_mode position_mode;
    uint32_t first_token;
    uint16_t token_count;
    uint32_t *positions;
    float *hyper;
    float *ngram_embeddings;
} fg_prefill_layer_work;

typedef struct fg_prefill_layer_result {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t first_token;
    uint16_t token_count;
    float *hyper;
} fg_prefill_layer_result;

typedef struct fg_layer_work {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t flags;
    fg_position_mode position_mode;
    uint32_t token_index;
    uint32_t position[4];
    float hyper[FG_HYPER_WIDTH];
    float ngram_embedding[FG_NGRAM_EMBED_VALUES];
} fg_layer_work;

typedef struct fg_layer_result {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t token_index;
    float hyper[FG_HYPER_WIDTH];
} fg_layer_result;

typedef struct fg_qsa_block_work {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    fg_position_mode position_mode;
    uint32_t token_index;
    uint32_t position[4];
    const float *hidden;
} fg_qsa_block_work;

typedef struct fg_qsa_block_result {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t token_index;
    const float *hidden;
} fg_qsa_block_result;

typedef struct fg_qsa_block_prefill_work {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    fg_position_mode position_mode;
    uint32_t first_token;
    uint16_t token_count;
    const uint32_t *positions;
    const float *hidden;
} fg_qsa_block_prefill_work;

typedef struct fg_qsa_block_prefill_result {
    uint8_t layer;
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t first_token;
    uint16_t token_count;
    const float *hidden;
} fg_qsa_block_prefill_result;

typedef struct fg_qsa_page {
    uint8_t layer;
    uint32_t block;
    const uint8_t *records;
} fg_qsa_page;

typedef struct fg_qsa_page_batch {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t batch_id;
    uint16_t page_count;
    const fg_qsa_page *pages;
} fg_qsa_page_batch;

typedef struct fg_qsa_page_barrier {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t batch_id;
} fg_qsa_page_barrier;

typedef struct fg_output_work {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t token_index;
    fg_sampler_config sampler;
    float uniform;
    float hyper[FG_HYPER_WIDTH];
} fg_output_work;

typedef struct fg_output_result {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t token_index;
    uint32_t token;
    float logit;
} fg_output_result;

typedef struct fg_output_config {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t flags;
    uint32_t token_index;
    fg_sampler_config sampler;
    float uniform;
} fg_output_config;

#define FG_OUTPUT_CONFIG_FLAG_SPLIT 1u
#define FG_OUTPUT_CONFIG_FLAG_SPLIT_4 2u

#define FG_OUTPUT_PARTIAL_BYTES 12u
#define FG_OUTPUT_SPLIT_FIRST_ROWS 140048u
#define FG_OUTPUT_SPLIT_WAYS_MIN 2u
#define FG_OUTPUT_SPLIT_WAYS_MAX 4u

fg_status fg_output_split_mode(uint32_t *ways,fg_error *err);
bool fg_output_split_requested(void);
bool fg_output_split_way_for_rank(uint32_t ways,uint32_t rank,uint32_t *way);
uint32_t fg_output_split_rank(uint32_t ways,uint32_t way);
void fg_output_split_span(uint32_t ways,uint32_t way,uint32_t *first_row,uint32_t *rows);
fg_status fg_output_split_require_slice(bool have_slice,uint32_t ways,uint32_t rank,
                                        fg_error *err);

typedef struct fg_output_partial {
    uint32_t token_index;
    float value;
    uint32_t id;
} fg_output_partial;

fg_status fg_output_partial_encode(uint8_t output[FG_OUTPUT_PARTIAL_BYTES],
                                   const fg_output_partial *partial,fg_error *err);
fg_status fg_output_partial_decode(fg_output_partial *partial,const uint8_t *payload,
                                   uint32_t bytes,fg_error *err);

#define FG_OUTPUT_SLICE_HIDDEN_BYTES (8u+FG_HIDDEN_SIZE*4u)
#define FG_OUTPUT_SLICE_HIDDEN_BF16_BYTES (8u+FG_HIDDEN_SIZE*2u)

typedef struct fg_output_slice_hidden {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint32_t token_index;
    float hidden[FG_HIDDEN_SIZE];
} fg_output_slice_hidden;

fg_status fg_output_slice_hidden_encode(
    uint8_t output[FG_OUTPUT_SLICE_HIDDEN_BYTES],
    const fg_output_slice_hidden *slice,uint32_t *bytes,fg_error *err);
fg_status fg_output_slice_hidden_decode(fg_output_slice_hidden *slice,
                                        const uint8_t *payload,uint32_t bytes,
                                        fg_error *err);

/* One-deep matcher for the direct output handoff.  Rank 0 emits exactly one
 * config per decode token and the final block owner emits exactly one hidden
 * result; either can be the first to arrive.  Newer token indexes replace
 * older pending ones, duplicates refresh in place, and stale arrivals are
 * dropped without error so a late frame from an aborted token cannot poison
 * the next one.  Ready means both halves name the same token. */
typedef struct fg_output_handoff {
    bool have_config;
    bool have_hidden;
    bool have_hidden_slice;
    bool have_local;
    fg_output_config config;
    fg_layer_result hidden;
    fg_layer_result hidden_slice;
    float local_value;
    uint32_t local_id;
    float remote_value[FG_OUTPUT_SPLIT_WAYS_MAX];
    uint32_t remote_id[FG_OUTPUT_SPLIT_WAYS_MAX];
    uint8_t remote_rank[FG_OUTPUT_SPLIT_WAYS_MAX];
    uint32_t remote_count;
    uint64_t wait_start_ms;
} fg_output_handoff;
void fg_output_handoff_reset(fg_output_handoff *state);
fg_status fg_output_handoff_config(fg_output_handoff *state,
                                   const fg_output_config *config,fg_error *err);
fg_status fg_output_handoff_hidden(fg_output_handoff *state,
                                   const fg_layer_result *hidden,fg_error *err);
fg_status fg_output_handoff_hidden_slice(fg_output_handoff *state,
                                         const fg_layer_result *slice,fg_error *err);
fg_status fg_output_handoff_partial(fg_output_handoff *state,uint32_t token_index,
                                    uint8_t source_rank,float value,uint32_t id,
                                    fg_error *err);
bool fg_output_handoff_ready(const fg_output_handoff *state);
bool fg_output_handoff_sample_ready(const fg_output_handoff *state);
void fg_output_handoff_take(fg_output_handoff *state,fg_output_config *config,
                            fg_layer_result *hidden);
/* Split liveness: -1 before the wait starts, 0 at/after the deadline, else the
 * remaining budget in milliseconds. */
int32_t fg_output_split_wait_remaining_ms(uint64_t start_ms,uint64_t now_ms,
                                          uint32_t timeout_ms);
/* Builds the loud timeout error naming every way rank whose partial the output
 * owner is still missing. */
fg_status fg_output_split_timeout_error(const fg_output_handoff *state,uint32_t ways,
                                        uint32_t waited_ms,fg_error *err);

typedef struct fg_output_history {
    const uint32_t *tokens;
    uint32_t count;
} fg_output_history;

typedef struct fg_ngram_work {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t item_count;
    uint32_t token_index;
    uint8_t heads[FG_NGRAM_SHARD_MAX_ITEMS];
    uint64_t rows[FG_NGRAM_SHARD_MAX_ITEMS];
} fg_ngram_work;

typedef struct fg_ngram_result {
    uint8_t source_rank;
    uint8_t destination_rank;
    uint8_t item_count;
    uint32_t token_index;
    uint8_t heads[FG_NGRAM_SHARD_MAX_ITEMS];
    uint8_t packed[FG_NGRAM_SHARD_MAX_ITEMS*FG_NGRAM_WIRE_ROW_BYTES];
} fg_ngram_result;

uint64_t fg_token_hash_update(uint64_t hash, const int32_t *tokens, size_t count);
uint32_t fg_crc32c(const void *data, size_t bytes);
bool fg_protocol_version_supported(uint16_t version);
fg_status fg_frame_encode_version(fg_frame_header *header, uint16_t version,
                                  fg_message_type type, uint64_t request_id,
                                  uint32_t sequence, uint32_t flags, const void *payload,
                                  uint32_t bytes, fg_error *err);
fg_status fg_frame_encode(fg_frame_header *header, fg_message_type type, uint64_t request_id, uint32_t sequence, uint32_t flags, const void *payload, uint32_t bytes, fg_error *err);
fg_status fg_frame_validate(const fg_frame_header *header, const void *payload, uint32_t *payload_bytes, fg_error *err);
fg_status fg_frame_validate_version(const fg_frame_header *header,uint16_t protocol_version,
                                    const void *payload,uint32_t *payload_bytes,
                                    fg_error *err);
uint16_t fg_frame_version(const fg_frame_header *header);
fg_message_type fg_frame_type(const fg_frame_header *header);
uint64_t fg_frame_request_id(const fg_frame_header *header);
uint32_t fg_frame_sequence(const fg_frame_header *header);
fg_status fg_decode_work_encode(uint8_t out[FG_DECODE_WORK_BYTES],const fg_decode_work *work,fg_error *err);
fg_status fg_decode_work_decode(fg_decode_work *work,const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_expert_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_expert_result *result,fg_error *err);
fg_status fg_expert_result_decode(fg_expert_result *result,const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_prefill_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                 const fg_prefill_work *work,fg_error *err);
fg_status fg_prefill_work_decode(fg_prefill_work *work,uint8_t *activation_storage,
                                 uint32_t activation_capacity,fg_prefill_pair *pair_storage,
                                 uint32_t pair_capacity,const uint8_t *payload,uint32_t bytes,
                                 fg_error *err);
fg_status fg_prefill_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                   const fg_prefill_result *result,fg_error *err);
fg_status fg_prefill_result_decode(fg_prefill_result *result,fg_prefill_result_pair *pair_storage,
                                   uint32_t pair_capacity,float *output_storage,
                                   uint64_t output_capacity_values,const uint8_t *payload,
                                   uint32_t bytes,fg_error *err);
fg_status fg_prefill_layer_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                       uint16_t protocol_version,
                                       const fg_prefill_layer_work *work,fg_error *err);
fg_status fg_prefill_layer_work_decode(fg_prefill_layer_work *work,uint16_t protocol_version,
                                       uint32_t *position_storage,
                                       uint32_t position_capacity,float *hyper_storage,
                                       uint64_t hyper_capacity_values,float *ngram_storage,
                                       uint64_t ngram_capacity_values,const uint8_t *payload,
                                       uint32_t bytes,fg_error *err);
fg_status fg_prefill_layer_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                         const fg_prefill_layer_result *result,fg_error *err);
fg_status fg_prefill_layer_result_decode(fg_prefill_layer_result *result,float *hyper_storage,
                                         uint64_t hyper_capacity_values,const uint8_t *payload,
                                         uint32_t bytes,fg_error *err);
fg_status fg_partition_prefill_routes(const fg_manifest *manifest,uint32_t layer,
                                      uint16_t token_count,const uint16_t *expert_ids,
                                      const float *gates,fg_prefill_route routes[FG_GROUP_SIZE],
                                      uint32_t *route_count,fg_prefill_pair *pair_storage,
                                      uint32_t pair_capacity,fg_error *err);
fg_status fg_prefill_result_validate_subset(const fg_manifest *manifest,
    const fg_prefill_work *work,uint8_t contributor_mask,
    const fg_prefill_result *result,fg_error *err);
fg_status fg_prefill_results_validate_route(const fg_manifest *manifest,uint32_t layer,
                                            uint32_t first_position,uint32_t owner_rank,
                                            uint16_t token_count,const uint16_t *expert_ids,
                                            const fg_prefill_result *results,uint32_t result_count,
                                            fg_error *err);
fg_status fg_expert_results_validate_route(const fg_manifest *manifest,uint32_t layer,uint32_t position,
                                           uint32_t owner_rank,const uint16_t expert_ids[FG_TOP_K],
                                           const fg_expert_result *results,uint32_t result_count,
                                           fg_error *err);
fg_status fg_partition_route(const fg_manifest *manifest,uint32_t layer,const uint16_t expert_ids[FG_TOP_K],
                             const float gates[FG_TOP_K],fg_expert_route routes[FG_GROUP_SIZE],
                             uint32_t *route_count,fg_error *err);
fg_status fg_layer_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                               uint16_t protocol_version,const fg_layer_work *work,
                               fg_error *err);
fg_status fg_layer_work_decode(fg_layer_work *work,uint16_t protocol_version,
                               const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_layer_result_encode(uint8_t output[FG_LAYER_RESULT_BYTES],const fg_layer_result *result,
                                 uint32_t *bytes,fg_error *err);
fg_status fg_layer_result_decode(fg_layer_result *result,const uint8_t *payload,uint32_t bytes,
                                 fg_error *err);
/* Ring decode uses the single-token layer work/result wire contract under its
 * own message ids; the work carries the first layer of the receiving block. */
#define FG_DECODE_LAYER_WORK_MAX_BYTES FG_LAYER_WORK_MAX_BYTES
#define FG_DECODE_LAYER_RESULT_BYTES FG_LAYER_RESULT_BYTES
fg_status fg_decode_layer_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                      uint16_t protocol_version,const fg_layer_work *work,
                                      fg_error *err);
fg_status fg_decode_layer_work_decode(fg_layer_work *work,uint16_t protocol_version,
                                      const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_decode_layer_result_encode(uint8_t output[FG_DECODE_LAYER_RESULT_BYTES],
                                        const fg_layer_result *result,uint32_t *bytes,fg_error *err);
fg_status fg_decode_layer_result_decode(fg_layer_result *result,const uint8_t *payload,
                                        uint32_t bytes,fg_error *err);
fg_status fg_output_slice_encode(uint8_t output[FG_DECODE_LAYER_RESULT_BYTES],
                                 const fg_layer_result *result,uint32_t *bytes,fg_error *err);
fg_status fg_qsa_block_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                   uint16_t protocol_version,const fg_qsa_block_work *work,
                                   fg_error *err);
fg_status fg_qsa_block_work_decode(fg_qsa_block_work *work,uint16_t protocol_version,
                                   float *hidden_storage,uint64_t hidden_capacity_values,
                                   const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_qsa_block_result_encode(uint8_t output[FG_QSA_BLOCK_RESULT_BYTES],
                                     const fg_qsa_block_result *result,fg_error *err);
fg_status fg_qsa_block_result_decode(fg_qsa_block_result *result,float *hidden_storage,
                                     uint64_t hidden_capacity_values,const uint8_t *payload,
                                     uint32_t bytes,fg_error *err);
fg_status fg_qsa_block_prefill_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                           uint16_t protocol_version,
                                           const fg_qsa_block_prefill_work *work,fg_error *err);
fg_status fg_qsa_block_prefill_work_decode(fg_qsa_block_prefill_work *work,
                                           uint16_t protocol_version,uint32_t *position_storage,
                                           uint32_t position_capacity,float *hidden_storage,
                                           uint64_t hidden_capacity_values,
                                           const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_qsa_block_prefill_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                             const fg_qsa_block_prefill_result *result,
                                             fg_error *err);
fg_status fg_qsa_block_prefill_result_decode(fg_qsa_block_prefill_result *result,
                                             float *hidden_storage,
                                             uint64_t hidden_capacity_values,
                                             const uint8_t *payload,uint32_t bytes,
                                             fg_error *err);
fg_status fg_qsa_completed_page_range(uint32_t first_token,uint32_t token_count,
                                      uint32_t *first_block,uint32_t *block_count,
                                      fg_error *err);
fg_status fg_qsa_page_append_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                    const fg_qsa_page_batch *batch,fg_error *err);
fg_status fg_qsa_page_append_decode(fg_qsa_page_batch *batch,fg_qsa_page *page_storage,
                                    uint32_t page_capacity,const uint8_t *payload,
                                    uint32_t bytes,fg_error *err);
fg_status fg_qsa_page_fetch_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                   const fg_qsa_page_batch *batch,fg_error *err);
fg_status fg_qsa_page_fetch_decode(fg_qsa_page_batch *batch,fg_qsa_page *page_storage,
                                   uint32_t page_capacity,const uint8_t *payload,
                                   uint32_t bytes,fg_error *err);
fg_status fg_qsa_page_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                    const fg_qsa_page_batch *batch,fg_error *err);
fg_status fg_qsa_page_result_decode(fg_qsa_page_batch *batch,fg_qsa_page *page_storage,
                                    uint32_t page_capacity,const uint8_t *payload,
                                    uint32_t bytes,fg_error *err);
fg_status fg_qsa_page_barrier_encode(uint8_t output[FG_QSA_PAGE_BARRIER_BYTES],
                                     const fg_qsa_page_barrier *barrier,fg_error *err);
fg_status fg_qsa_page_barrier_decode(fg_qsa_page_barrier *barrier,
                                     const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_output_work_encode(uint8_t output[FG_OUTPUT_WORK_BYTES],const fg_output_work *work,
                                fg_error *err);
fg_status fg_output_work_decode(fg_output_work *work,const uint8_t *payload,uint32_t bytes,
                                fg_error *err);
fg_status fg_output_config_encode(uint8_t output[FG_OUTPUT_CONFIG_BYTES],
                                  const fg_output_config *config,fg_error *err);
fg_status fg_output_config_decode(fg_output_config *config,const uint8_t *payload,
                                  uint32_t bytes,fg_error *err);
fg_status fg_output_result_encode(uint8_t output[FG_OUTPUT_RESULT_BYTES],const fg_output_result *result,
                                  fg_error *err);
fg_status fg_output_result_decode(fg_output_result *result,const uint8_t *payload,uint32_t bytes,
                                  fg_error *err);
fg_status fg_output_history_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                   const fg_output_history *history,fg_error *err);
fg_status fg_output_history_decode(fg_output_history *history,uint32_t *storage,
                                   uint32_t capacity,const uint8_t *payload,
                                   uint32_t bytes,fg_error *err);
fg_status fg_ngram_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                               const fg_ngram_work *work,fg_error *err);
fg_status fg_ngram_work_decode(fg_ngram_work *work,const uint8_t *payload,uint32_t bytes,
                               fg_error *err);
fg_status fg_ngram_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                 const fg_ngram_result *result,fg_error *err);
fg_status fg_ngram_result_decode(fg_ngram_result *result,const uint8_t *payload,uint32_t bytes,
                                 fg_error *err);
fg_status fg_owner_session_control_encode(uint8_t output[FG_OWNER_SESSION_CONTROL_BYTES],
                                         const fg_owner_session_control *control,
                                         fg_error *err);
fg_status fg_owner_session_control_decode(fg_owner_session_control *control,
                                         const uint8_t *payload, uint32_t bytes,
                                         fg_error *err);

#endif
