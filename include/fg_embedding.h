#ifndef FLASH_GORDON_EMBEDDING_H
#define FLASH_GORDON_EMBEDDING_H

#include "fg_manifest.h"
#include "fg_quant.h"

#define FG_EMBEDDING_VOCAB_SIZE 248320u
#define FG_EMBEDDING_GROUP_COUNT 4u
#define FG_EMBEDDING_BOUNDARY_WIDTH (FG_HIDDEN_SIZE * FG_EMBEDDING_GROUP_COUNT)
#define FG_EMBEDDING_ROW_BYTES \
    ((FG_HIDDEN_SIZE / FG_QK8_0) * FG_Q8_0_BLOCK_BYTES)
#define FG_EMBEDDING_ARTIFACT_BYTES \
    ((uint64_t)FG_EMBEDDING_ROW_BYTES * FG_EMBEDDING_VOCAB_SIZE)

typedef struct fg_embedding fg_embedding;

static inline bool fg_embedding_record_metadata_valid(
    const fg_tensor_record *record,uint32_t rank){
    if(!record)return false;
    uint8_t digest=0u;
    for(uint32_t i=0;i<sizeof(record->sha256);i++)digest|=record->sha256[i];
    return digest&&record->kind==FG_TENSOR_HOST_CACHE&&
        record->layout==FG_TENSOR_LAYOUT_HOST_Q8_0&&record->rank==rank&&
        record->offset==0u&&record->ggml_type==8u&&record->dims==2u&&
        record->shape[0]==FG_HIDDEN_SIZE&&
        record->shape[1]==FG_EMBEDDING_VOCAB_SIZE&&
        record->shape[2]==0u&&record->shape[3]==0u&&
        record->layer==UINT16_MAX&&record->expert==UINT16_MAX&&
        record->bytes==FG_EMBEDDING_ARTIFACT_BYTES;
}

fg_status fg_embedding_open(fg_embedding **out, const fg_manifest *manifest,
                            const char *pack_dir, uint32_t rank, fg_error *err);
void fg_embedding_close(fg_embedding *embedding);

/*
 * Produces token-major [token_count, 10240] FP32 boundaries. Capacity is
 * expressed in float values, not bytes.
 */
fg_status fg_embedding_gather(const fg_embedding *embedding,
                              const uint32_t *token_ids, size_t token_count,
                              float *boundary, size_t boundary_capacity,
                              fg_error *err);

#endif
