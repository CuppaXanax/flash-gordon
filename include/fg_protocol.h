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
#define FG_OUTPUT_WORK_HEADER_BYTES 24u
#define FG_OUTPUT_WORK_BYTES (FG_OUTPUT_WORK_HEADER_BYTES+FG_HYPER_WIDTH*4u)
#define FG_OUTPUT_RESULT_BYTES 16u
#define FG_LAYER_WORK_HAS_NGRAM 1u
#define FG_PREFILL_MAX_TOKENS 512u
#define FG_PREFILL_MAX_PAIRS (FG_PREFILL_MAX_TOKENS*FG_TOP_K)
#define FG_PREFILL_WORK_HEADER_BYTES 16u
#define FG_PREFILL_PAIR_BYTES 12u
#define FG_PREFILL_RESULT_HEADER_BYTES 16u
#define FG_PREFILL_RESULT_PAIR_BYTES (4u+FG_HIDDEN_SIZE*4u)
#define FG_PREFILL_WORK_MAX_BYTES (FG_PREFILL_WORK_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*FG_Q8K_ACTIVATION_BYTES+FG_PREFILL_MAX_PAIRS*FG_PREFILL_PAIR_BYTES)
#define FG_PREFILL_RESULT_MAX_BYTES (FG_PREFILL_RESULT_HEADER_BYTES+FG_PREFILL_MAX_PAIRS*FG_PREFILL_RESULT_PAIR_BYTES)
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
#define FG_NGRAM_SHARD_MAX_ITEMS FG_NGRAM_HEAD_COUNT
#define FG_NGRAM_WIRE_ROW_BYTES 90u
#define FG_NGRAM_WORK_MAX_BYTES (8u+FG_NGRAM_SHARD_MAX_ITEMS*9u)
#define FG_NGRAM_RESULT_MAX_BYTES (8u+FG_NGRAM_SHARD_MAX_ITEMS*(1u+FG_NGRAM_WIRE_ROW_BYTES))
#define FG_OWNER_SESSION_CONTROL_VERSION 2u
#define FG_OWNER_SESSION_CONTROL_BYTES 184u
#define FG_PIPELINE_POSITION_AXES 3u
#define FG_PIPELINE_BOUNDARY_WIDTH FG_Q38_HYPER_WIDTH
#define FG_PIPELINE_BOUNDARY_FP32_BYTES 4u
#define FG_PIPELINE_STAGE_TIMING_BYTES (FG_PIPELINE_STAGE_COUNT*4u)
#define FG_PIPELINE_ACTIVATION_HEADER_BYTES (32u+FG_PIPELINE_STAGE_TIMING_BYTES)
/* Protocol 7 boundary tensors are native little-endian IEEE-754 FP32 byte arrays. */
#define FG_PIPELINE_ACTIVATION_MAX_BYTES \
    (FG_PIPELINE_ACTIVATION_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*FG_PIPELINE_POSITION_AXES*4u+ \
     FG_PREFILL_MAX_TOKENS*FG_PIPELINE_BOUNDARY_WIDTH*FG_PIPELINE_BOUNDARY_FP32_BYTES)
#define FG_PIPELINE_CREDIT_BYTES 4u
#define FG_PIPELINE_RESULT_BYTES (20u+FG_PIPELINE_STAGE_TIMING_BYTES)
#define FG_PIPELINE_DRAIN_BYTES 4u
#define FG_PIPELINE_DRAINED_BYTES 4u
#define FG_PIPELINE_ABORT_BYTES 12u

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
    FG_MSG_PIPELINE_ACTIVATION = 34,
    FG_MSG_PIPELINE_CREDIT = 35,
    FG_MSG_PIPELINE_RESULT = 36,
    FG_MSG_PIPELINE_DRAIN = 37,
    FG_MSG_PIPELINE_DRAINED = 38,
    FG_MSG_PIPELINE_ABORT = 39
} fg_message_type;

typedef enum fg_pipeline_execution_kind {
    FG_PIPELINE_EXECUTION_PREFILL = 1,
    FG_PIPELINE_EXECUTION_DECODE = 2
} fg_pipeline_execution_kind;

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
    uint32_t first_position;
    uint16_t token_count;
    uint16_t pair_count;
    fg_prefill_result_pair *pairs;
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

typedef struct fg_pipeline_activation {
    fg_pipeline_execution_kind execution_kind;
    uint8_t slot;
    uint8_t source_stage;
    uint8_t destination_stage;
    uint32_t first_token;
    uint16_t token_count;
    /* Required for decode; optional for caller-selected prefill completions. */
    bool request_output;
    float stage_seconds[FG_PIPELINE_STAGE_COUNT];
    fg_sampler_config sampler;
    float uniform;
    const uint32_t *positions;
    const float *boundary;
} fg_pipeline_activation;

typedef struct fg_pipeline_credit {
    uint8_t source_stage;
    uint8_t destination_stage;
    uint8_t slot;
} fg_pipeline_credit;

typedef struct fg_pipeline_result {
    uint32_t completed_first_token;
    uint16_t completed_token_count;
    uint32_t completed_frontier;
    /* False requires final_token == FG_Q38_VOCAB_SIZE and +0.0f final_logit. */
    bool has_output;
    uint32_t final_token;
    float final_logit;
    float stage_seconds[FG_PIPELINE_STAGE_COUNT];
} fg_pipeline_result;

typedef struct fg_pipeline_drain {
    uint8_t source_stage;
    uint8_t destination_stage;
} fg_pipeline_drain;

typedef struct fg_pipeline_drained {
    uint8_t source_stage;
    uint8_t destination_stage;
} fg_pipeline_drained;

typedef struct fg_pipeline_abort {
    uint8_t origin_stage;
    fg_status status;
    uint32_t failing_sequence;
} fg_pipeline_abort;

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
                                 fg_error *err);
fg_status fg_layer_result_decode(fg_layer_result *result,const uint8_t *payload,uint32_t bytes,
                                 fg_error *err);
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
fg_status fg_output_result_encode(uint8_t output[FG_OUTPUT_RESULT_BYTES],const fg_output_result *result,
                                  fg_error *err);
fg_status fg_output_result_decode(fg_output_result *result,const uint8_t *payload,uint32_t bytes,
                                  fg_error *err);
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
fg_status fg_pipeline_activation_validate(const fg_pipeline_activation *activation,
                                         fg_error *err);
fg_status fg_pipeline_activation_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                        const fg_pipeline_activation *activation,
                                        fg_error *err);
fg_status fg_pipeline_activation_decode(fg_pipeline_activation *activation,
                                        uint32_t *position_storage,
                                        uint32_t position_capacity,float *boundary_storage,
                                        uint64_t boundary_capacity_values,
                                        const uint8_t *payload,uint32_t bytes,
                                        fg_error *err);
fg_status fg_pipeline_credit_encode(uint8_t output[FG_PIPELINE_CREDIT_BYTES],
                                    const fg_pipeline_credit *credit,fg_error *err);
fg_status fg_pipeline_credit_decode(fg_pipeline_credit *credit,const uint8_t *payload,
                                    uint32_t bytes,fg_error *err);
fg_status fg_pipeline_result_encode(uint8_t output[FG_PIPELINE_RESULT_BYTES],
                                    const fg_pipeline_result *result,fg_error *err);
fg_status fg_pipeline_result_decode(fg_pipeline_result *result,const uint8_t *payload,
                                    uint32_t bytes,fg_error *err);
fg_status fg_pipeline_drain_encode(uint8_t output[FG_PIPELINE_DRAIN_BYTES],
                                   const fg_pipeline_drain *drain,fg_error *err);
fg_status fg_pipeline_drain_decode(fg_pipeline_drain *drain,const uint8_t *payload,
                                   uint32_t bytes,fg_error *err);
fg_status fg_pipeline_drained_encode(uint8_t output[FG_PIPELINE_DRAINED_BYTES],
                                     const fg_pipeline_drained *drained,fg_error *err);
fg_status fg_pipeline_drained_decode(fg_pipeline_drained *drained,const uint8_t *payload,
                                     uint32_t bytes,fg_error *err);
fg_status fg_pipeline_abort_encode(uint8_t output[FG_PIPELINE_ABORT_BYTES],
                                   const fg_pipeline_abort *abort,fg_error *err);
fg_status fg_pipeline_abort_decode(fg_pipeline_abort *abort,const uint8_t *payload,
                                   uint32_t bytes,fg_error *err);
fg_status fg_pipeline_frame_validate_sequence(const fg_frame_header *header,
                                             fg_message_type expected_type,
                                             uint64_t expected_request_id,
                                             uint32_t expected_sequence,
                                             fg_error *err);

#endif
