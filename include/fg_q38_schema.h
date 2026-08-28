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

fg_status fg_q38_validate_gguf(const fg_gguf *gguf, fg_error *err);
fg_status fg_q38_validate_packed_manifest(const fg_manifest *manifest, fg_error *err);
const fg_tensor_record *fg_q38_find_tensor(const fg_manifest *manifest, const char *name,
                                           uint32_t rank);
void fg_q38_account_session_state(fg_manifest *manifest);
uint64_t fg_q38_runtime_scratch_bytes(uint32_t rank,uint32_t microbatch,uint32_t window,
                                      uint32_t max_context);

#endif
