#ifndef FLASH_GORDON_MANIFEST_H
#define FLASH_GORDON_MANIFEST_H

#include "fg.h"

#define FG_MANIFEST_MAGIC UINT64_C(0x314d47464e574f51) /* QOWNFGM1 */
#define FG_MANIFEST_FORMAT_VERSION 3u
#define FG_MANIFEST_SIZE 65536u
#define FG_MAX_TENSORS 4096u
#define FG_TENSOR_NAME_MAX 96u
#define FG_MAX_ENDPOINT 64u

typedef enum fg_tensor_kind {
    FG_TENSOR_COMMON = 1,
    FG_TENSOR_ROUTED_EXPERT = 2,
    FG_TENSOR_NGRAM = 3,
    FG_TENSOR_VISION = 4,
    FG_TENSOR_MTP = 5,
    FG_TENSOR_TOKENIZER = 6
} fg_tensor_kind;

enum {
    FG_MANIFEST_HAS_TEXT = 1u << 0,
    FG_MANIFEST_HAS_NGRAM = 1u << 1,
    FG_MANIFEST_HAS_VISION = 1u << 2,
    FG_MANIFEST_HAS_MTP = 1u << 3,
    FG_MANIFEST_HAS_TOKENIZER = 1u << 4,
    FG_MANIFEST_COMPONENTS_TEXT_REQUIRED = FG_MANIFEST_HAS_TEXT | FG_MANIFEST_HAS_NGRAM |
        FG_MANIFEST_HAS_TOKENIZER,
    FG_MANIFEST_COMPONENTS_MULTIMODAL = FG_MANIFEST_HAS_VISION | FG_MANIFEST_HAS_MTP
};

typedef struct fg_tensor_record {
    char name[FG_TENSOR_NAME_MAX];
    uint64_t offset;
    uint64_t bytes;
    uint32_t ggml_type;
    uint32_t dims;
    uint64_t shape[4];
    uint16_t rank;
    uint16_t layer;
    uint16_t expert;
    uint8_t kind;
    uint8_t reserved[5];
    uint8_t sha256[32];
} fg_tensor_record;

typedef struct fg_rank_record {
    char endpoint[FG_MAX_ENDPOINT];
    uint64_t persistent_bytes;
    uint64_t transient_bytes;
    uint64_t kv_bytes;
    uint64_t state_file_bytes;
    uint64_t scratch_bytes;
    uint64_t driver_reserve_bytes;
    uint32_t tensor_begin;
    uint32_t tensor_count;
} fg_rank_record;

typedef struct fg_manifest {
    uint64_t magic;
    uint32_t format_version;
    uint32_t protocol_version;
    uint32_t header_bytes;
    uint32_t rank_count;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t hidden_size;
    uint32_t top_k;
    uint32_t required_cu;
    uint32_t native_context;
    uint32_t max_context;
    uint32_t prefill_microbatch;
    uint32_t prefill_window;
    uint32_t tensor_count;
    uint32_t flags;
    uint64_t persistent_cap_bytes;
    uint64_t residency_cap_bytes;
    uint8_t source_sha256[32];
    uint8_t quant_profile_sha256[32];
    uint8_t manifest_sha256[32];
    fg_rank_record ranks[FG_RANK_COUNT];
    uint8_t layer_owner[FG_LAYER_COUNT];
    uint8_t layer_groups[FG_LAYER_COUNT][FG_GROUP_SIZE];
    uint16_t expert_rank[FG_LAYER_COUNT][FG_EXPERT_COUNT];
    fg_tensor_record tensors[FG_MAX_TENSORS];
} fg_manifest;

void fg_manifest_init(fg_manifest *manifest);
fg_status fg_manifest_validate(const fg_manifest *manifest, fg_error *err);
fg_status fg_manifest_validate_deployment(const fg_manifest *manifest, fg_error *err);
fg_status fg_manifest_read(const char *path, fg_manifest *manifest, fg_error *err);
fg_status fg_manifest_write(const char *path, fg_manifest *manifest, fg_error *err);
fg_status fg_manifest_add_tensor(fg_manifest *manifest, const fg_tensor_record *record, fg_error *err);
void fg_manifest_print(const fg_manifest *manifest);

#endif
