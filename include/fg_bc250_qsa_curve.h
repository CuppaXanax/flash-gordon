#ifndef FLASH_GORDON_BC250_QSA_CURVE_H
#define FLASH_GORDON_BC250_QSA_CURVE_H

#include <stddef.h>
#include <stdint.h>

#define FG_BC250_QSA_CURVE_SCHEMA "fg.bc250.qsa_curve.v1"
#define FG_BC250_QSA_CAPACITY 262144u
#define FG_BC250_QSA_SEGMENT_CAPACITY 131072u
#define FG_BC250_QSA_COMPRESS_RATIO 4u
#define FG_BC250_QSA_GROUP_BLOCKS 4096u
#define FG_BC250_QSA_SELECTED_STRIDE 512u

typedef struct fg_bc250_qsa_geometry {
    uint32_t visible_tokens;
    uint32_t query_tokens;
    uint32_t complete_blocks;
    uint32_t selector_groups;
    uint32_t candidate_count;
    uint32_t merge_passes;
    uint32_t selected_stride;
    uint32_t capacity;
    uint32_t segment_capacity;
    uint64_t scratch_bytes;
} fg_bc250_qsa_geometry;

/* Returns zero for an invalid range or arithmetic overflow. */
int fg_bc250_qsa_geometry_make(uint32_t visible_tokens,uint32_t query_tokens,
                               uint32_t capacity,uint32_t segment_capacity,
                               fg_bc250_qsa_geometry *geometry);
int fg_bc250_qsa_curve_boundaries_valid(const uint32_t *points,size_t count,
                                        uint32_t capacity);

#endif
