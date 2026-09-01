#ifndef FLASH_GORDON_H
#define FLASH_GORDON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FG_VERSION_MAJOR 0u
#define FG_VERSION_MINOR 1u
#define FG_PROTOCOL_MIN_VERSION 5u
#define FG_PROTOCOL_VERSION 6u
#define FG_PIPELINE_PROTOCOL_VERSION 7u
#define FG_PROTOCOL_MAX_VERSION FG_PIPELINE_PROTOCOL_VERSION
#define FG_RANK_COUNT 8u
#define FG_LAYER_COUNT 48u
#define FG_EXPERT_COUNT 512u
#define FG_EXPERTS_PER_RANK 128u
#define FG_GROUP_SIZE 4u
#define FG_TOP_K 10u
#define FG_HIDDEN_SIZE 2560u
#define FG_NGRAM_HEAD_COUNT 16u
#define FG_NGRAM_EMBED_WIDTH 160u
#define FG_ALIGNMENT 4096u
#define FG_DEFAULT_MICROBATCH 256u
#define FG_DEFAULT_WINDOW 2u
#define FG_NATIVE_CONTEXT 262144u
#define FG_MAX_CONTEXT 1048576u
#define FG_REQUIRED_CU 24u
#define FG_RUNTIME_PROFILE_NONE 0u
#define FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128 1u
#define FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128_NAME "native-262k-microbatch-128"
#define FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES (UINT64_C(16) << 20u)
#define FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K 2u
#define FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K_NAME "pipeline-8stage-262k"
#define FG_PIPELINE_STAGE_COUNT FG_RANK_COUNT
#define FG_PIPELINE_LAYER_OFFSETS (FG_PIPELINE_STAGE_COUNT + 1u)
#define FG_PIPELINE_DEFAULT_LAYERS_PER_STAGE 6u
#define FG_PIPELINE_DEFAULT_SLOT_COUNT 2u
#define FG_PIPELINE_DEFAULT_MICROBATCH 128u
#define FG_PIPELINE_MAX_INFLIGHT_CHUNKS \
    (FG_PIPELINE_STAGE_COUNT * FG_PIPELINE_DEFAULT_SLOT_COUNT)
#define FG_PACK_RANK_TRANSIENT_BYTES (UINT64_C(512) << 20u)
#define FG_PACK_DRIVER_RESERVE_BYTES (UINT64_C(512) << 20u)
#define FG_PERSISTENT_CAP_BYTES UINT64_C(11166914969) /* 10.4 GiB */
#define FG_RESIDENCY_CAP_BYTES UINT64_C(14495514624)  /* 13.5 GiB */

typedef enum fg_position_mode {
    FG_POSITION_TEXT = 0,
    FG_POSITION_FOUR_AXIS = 1
} fg_position_mode;

typedef enum fg_status {
    FG_OK = 0,
    FG_ERR_ARGUMENT = 2,
    FG_ERR_IO = 3,
    FG_ERR_FORMAT = 4,
    FG_ERR_MISMATCH = 5,
    FG_ERR_OOM = 6,
    FG_ERR_UNAVAILABLE = 7,
    FG_ERR_LIMIT = 8
} fg_status;

typedef struct fg_error {
    fg_status code;
    char message[512];
} fg_error;

void fg_error_set(fg_error *err, fg_status code, const char *fmt, ...);
uint64_t fg_align_up_u64(uint64_t value, uint64_t alignment);
bool fg_is_aligned_u64(uint64_t value, uint64_t alignment);

#endif
