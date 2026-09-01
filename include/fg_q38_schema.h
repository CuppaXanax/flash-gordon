#ifndef FLASH_GORDON_Q38_SCHEMA_H
#define FLASH_GORDON_Q38_SCHEMA_H

#include "fg_gguf.h"

#define FG_Q38_VOCAB_SIZE 248320u
#define FG_Q38_HYPER_COUNT 4u
#define FG_Q38_HYPER_WIDTH 10240u
#define FG_Q38_HYPER_RANK 320u
#define FG_Q38_EXPERT_WIDTH 640u
#define FG_Q38_GDN_KEY_WIDTH 2048u
#define FG_Q38_GDN_VALUE_WIDTH 6144u
#define FG_Q38_GDN_CONV_WIDTH 10240u
#define FG_Q38_GDN_HEADS 48u
#define FG_Q38_ATTN_QUERY_WIDTH 12288u
#define FG_Q38_ATTN_KV_WIDTH 512u
#define FG_Q38_ATTN_HEAD_WIDTH 256u
#define FG_Q38_ATTN_QUERY_HEADS 24u
#define FG_Q38_ATTN_KV_HEADS 2u
#define FG_Q38_ROTARY_WIDTH 64u
#define FG_Q38_ROPE_THETA 10000000.0f
#define FG_Q38_INDEX_WIDTH 128u
#define FG_Q38_INDEX_QUERY_WIDTH 512u
#define FG_Q38_INDEX_HEADS 4u
#define FG_Q38_INDEX_BUDGET 2048u
#define FG_Q38_Q8_0_BLOCK_BYTES 34u
#define FG_Q38_Q4_0_BLOCK_BYTES 18u
#define FG_Q38_QSA_COMPRESS_RATIO 4u
#define FG_Q38_QSA_STATE_PAGE_BYTES 8192u
#define FG_Q38_QSA_KEY_BYTES ((FG_Q38_ATTN_KV_WIDTH/32u)*FG_Q38_Q8_0_BLOCK_BYTES)
#define FG_Q38_QSA_VALUE_BYTES ((FG_Q38_ATTN_KV_WIDTH/32u)*FG_Q38_Q8_0_BLOCK_BYTES)
#define FG_Q38_QSA_INDEX_KEY_BYTES ((FG_Q38_INDEX_WIDTH/32u)*FG_Q38_Q8_0_BLOCK_BYTES)
#define FG_Q38_QSA_POSITION_BYTES 12u
#define FG_Q38_QSA_TOKEN_RECORD_BYTES (FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES+FG_Q38_QSA_POSITION_BYTES)
#define FG_Q38_DECODE_TILE_WORDS 9u
#define FG_Q38_PREFILL_TILE_WORDS 17u
#define FG_Q38_PIPELINE_DECODE_EXTRA_BYTES \
    ((((uint64_t)FG_TOP_K-1u)*FG_HIDDEN_SIZE+FG_TOP_K)*4u)

typedef enum fg_q38_pipeline_owner_transient {
    FG_Q38_PIPELINE_TRANSIENT_HYPER_NORM,
    FG_Q38_PIPELINE_TRANSIENT_LOW,
    FG_Q38_PIPELINE_TRANSIENT_HC_DOWN_PARTIALS,
    FG_Q38_PIPELINE_TRANSIENT_LOW_ACTIVE,
    FG_Q38_PIPELINE_TRANSIENT_UP_LOGITS,
    FG_Q38_PIPELINE_TRANSIENT_INJECT_PARTIALS,
    FG_Q38_PIPELINE_TRANSIENT_MIXED,
    FG_Q38_PIPELINE_TRANSIENT_INJECTION,
    FG_Q38_PIPELINE_TRANSIENT_ROUTER_LOGITS,
    FG_Q38_PIPELINE_TRANSIENT_SHARED_GATE,
    FG_Q38_PIPELINE_TRANSIENT_SHARED_UP,
    FG_Q38_PIPELINE_TRANSIENT_SHARED_MID,
    FG_Q38_PIPELINE_TRANSIENT_SHARED_OUTPUT,
    FG_Q38_PIPELINE_TRANSIENT_SHARED_SCALAR,
    FG_Q38_PIPELINE_TRANSIENT_REDUCED,
    FG_Q38_PIPELINE_TRANSIENT_SELECTED,
    FG_Q38_PIPELINE_TRANSIENT_GATES,
    FG_Q38_PIPELINE_TRANSIENT_PREFILL_TILES,
    FG_Q38_PIPELINE_TRANSIENT_COUNT
} fg_q38_pipeline_owner_transient;

typedef struct fg_q38_pipeline_owner_transient_layout {
    uint64_t offsets[FG_Q38_PIPELINE_TRANSIENT_COUNT];
    uint64_t bytes[FG_Q38_PIPELINE_TRANSIENT_COUNT];
    uint64_t total_bytes;
} fg_q38_pipeline_owner_transient_layout;

static inline bool fg_q38_pipeline_owner_transient_layout_get(
    uint32_t batch_size,fg_q38_pipeline_owner_transient_layout *layout){
    if(!layout||!batch_size||batch_size>512u)return false;
    const uint64_t tokens=batch_size;
    const uint64_t bytes[FG_Q38_PIPELINE_TRANSIENT_COUNT]={
        tokens*FG_Q38_HYPER_WIDTH*4u,
        tokens*FG_Q38_HYPER_RANK*4u,
        8u*FG_Q38_HYPER_RANK*4u,
        tokens*FG_Q38_HYPER_RANK*4u,
        tokens*FG_Q38_HYPER_WIDTH*4u,
        tokens*24u*FG_Q38_HYPER_COUNT*4u,
        tokens*FG_HIDDEN_SIZE*4u,
        tokens*FG_Q38_HYPER_COUNT*4u,
        tokens*FG_EXPERT_COUNT*4u,
        tokens*FG_Q38_EXPERT_WIDTH*4u,
        tokens*FG_Q38_EXPERT_WIDTH*4u,
        tokens*FG_Q38_EXPERT_WIDTH*4u,
        tokens*FG_HIDDEN_SIZE*4u,
        tokens*4u,
        tokens*FG_HIDDEN_SIZE*4u,
        tokens*FG_TOP_K*4u,
        tokens*FG_TOP_K*4u,
        tokens*FG_TOP_K*FG_Q38_PREFILL_TILE_WORDS*4u
    };
    uint64_t offset=0u;
    for(uint32_t i=0;i<FG_Q38_PIPELINE_TRANSIENT_COUNT;i++){
        offset=fg_align_up_u64(offset,FG_ALIGNMENT);
        layout->offsets[i]=offset;
        layout->bytes[i]=bytes[i];
        offset+=bytes[i];
    }
    layout->total_bytes=fg_align_up_u64(offset,FG_ALIGNMENT);
    return true;
}

static inline uint64_t fg_q38_pipeline_owner_transient_bytes(uint32_t batch_size){
    fg_q38_pipeline_owner_transient_layout layout;
    return fg_q38_pipeline_owner_transient_layout_get(batch_size,&layout)?
        layout.total_bytes:UINT64_MAX;
}

fg_status fg_q38_validate_gguf(const fg_gguf *gguf, fg_error *err);
fg_status fg_q38_validate_packed_manifest(const fg_manifest *manifest, fg_error *err);
fg_status fg_tensor_record_expected_bytes(const fg_tensor_record *record,uint64_t *bytes,
                                          fg_error *err);
fg_status fg_manifest_validate_tensor_storage(const fg_manifest *manifest,fg_error *err);
const fg_tensor_record *fg_q38_find_tensor(const fg_manifest *manifest, const char *name,
                                           uint32_t rank);
const fg_ngram_shard_record *fg_q38_find_ngram_shard(const fg_manifest *manifest,
                                                     uint32_t rank);
fg_status fg_q38_validate_ngram_shards(const fg_manifest *manifest,fg_error *err);
fg_status fg_q38_rank_residency_bytes(const fg_manifest *manifest,uint32_t rank,
                                      uint64_t *bytes,fg_error *err);
void fg_q38_account_session_state(fg_manifest *manifest);
void fg_q38_session_state_bytes_for_rank(const fg_manifest *manifest,uint32_t rank,
                                         uint64_t *kv_bytes,uint64_t *state_file_bytes);
uint64_t fg_q38_pipeline_activation_slot_bytes(uint32_t microbatch);
uint64_t fg_q38_runtime_scratch_bytes(uint32_t rank,uint32_t microbatch,uint32_t window,
                                      uint32_t max_context);
uint64_t fg_q38_runtime_scratch_bytes_for_manifest(const fg_manifest *manifest,
                                                   uint32_t rank,uint32_t microbatch,
                                                   uint32_t window,uint32_t max_context);

#endif
