#ifndef FLASH_GORDON_TOWER_H
#define FLASH_GORDON_TOWER_H

#include "fg.h"

#define FG_TOWER_PATCH_SIZE 16u
#define FG_TOWER_TEMPORAL 2u
#define FG_TOWER_MERGE 2u
#define FG_TOWER_HIDDEN 1152u
#define FG_TOWER_HEADS 16u
#define FG_TOWER_HEAD_DIM 72u
#define FG_TOWER_LAYERS 27u
#define FG_TOWER_MLP 4304u
#define FG_TOWER_OUT_HIDDEN 2560u
#define FG_TOWER_POS_GRID 48u
#define FG_TOWER_PATCH_VALUES (3u*FG_TOWER_PATCH_SIZE*FG_TOWER_PATCH_SIZE)
#define FG_TOWER_TOKEN_VALUES (FG_TOWER_TEMPORAL*FG_TOWER_PATCH_VALUES)
#define FG_TOWER_QKV_WIDTH (3u*FG_TOWER_HIDDEN)
#define FG_TOWER_MERGE_FACTOR (FG_TOWER_MERGE*FG_TOWER_MERGE)
#define FG_TOWER_MERGED_WIDTH (FG_TOWER_MERGE_FACTOR*FG_TOWER_HIDDEN)
#define FG_TOWER_ROPE_PAIRS (FG_TOWER_HEAD_DIM/2u)
#define FG_TOWER_ROPE_SECTION (FG_TOWER_HEAD_DIM/4u)
#define FG_TOWER_ROPE_BASE 10000.0f
#define FG_TOWER_LN_EPS 1e-6f
#define FG_TOWER_ALIGN 32u
#define FG_TOWER_IMAGE_MIN_PIXELS UINT64_C(65536)
#define FG_TOWER_IMAGE_MAX_PIXELS UINT64_C(16777216)
#define FG_TOWER_VIDEO_MIN_PIXELS UINT64_C(4096)
#define FG_TOWER_VIDEO_MAX_PIXELS UINT64_C(25228800)

typedef struct fg_tower_geometry {
    uint32_t image_width;
    uint32_t image_height;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t tokens;
    uint32_t merged_tokens;
} fg_tower_geometry;

typedef struct fg_tower_block_weights {
    const float *ln1_weight;
    const float *ln1_bias;
    const float *ln2_weight;
    const float *ln2_bias;
    const float *qkv_weight;
    const float *qkv_bias;
    const float *attn_out_weight;
    const float *attn_out_bias;
    const float *ffn_up_weight;
    const float *ffn_up_bias;
    const float *ffn_down_weight;
    const float *ffn_down_bias;
} fg_tower_block_weights;

typedef struct fg_tower_weights {
    const float *patch_weight;
    const float *patch_bias;
    const float *position_weight;
    fg_tower_block_weights blocks[FG_TOWER_LAYERS];
    const float *post_ln_weight;
    const float *post_ln_bias;
    const float *merger_fc1_weight;
    const float *merger_fc1_bias;
    const float *merger_fc2_weight;
    const float *merger_fc2_bias;
} fg_tower_weights;

fg_status fg_tower_smart_resize(uint32_t width,uint32_t height,uint32_t align,
                                uint64_t min_pixels,uint64_t max_pixels,
                                uint32_t *out_width,uint32_t *out_height,fg_error *err);
fg_status fg_tower_image_geometry(uint32_t width,uint32_t height,
                                  fg_tower_geometry *geometry,fg_error *err);
fg_status fg_tower_normalize_image(const uint8_t *rgb,uint32_t width,uint32_t height,
                                   float *out,fg_error *err);
fg_status fg_tower_resize_bicubic(const float *source,uint32_t source_width,
                                  uint32_t source_height,float *destination,
                                  uint32_t width,uint32_t height,fg_error *err);
fg_status fg_tower_patchify(const float *image,uint32_t width,uint32_t height,
                            float *tokens,const fg_tower_geometry *geometry,fg_error *err);
fg_status fg_tower_vision_positions(const fg_tower_geometry *geometry,int32_t *positions,
                                    fg_error *err);
fg_status fg_tower_position_embeddings(const float *position_weight,
                                       const fg_tower_geometry *geometry,float *tokens,
                                       fg_error *err);
fg_status fg_tower_forward_cpu(const fg_tower_weights *weights,const float *tokens,
                               const fg_tower_geometry *geometry,float *embeddings,
                               fg_error *err);
fg_status fg_tower_dequantize(const uint8_t *data,uint64_t bytes,uint32_t ggml_type,
                              uint64_t values,float *out,fg_error *err);
fg_status fg_tower_image_decode(const uint8_t *data,uint64_t bytes,uint8_t **rgb,
                                uint32_t *width,uint32_t *height,fg_error *err);
fg_status fg_tower_embed_cpu(const fg_tower_weights *weights,const float *tokens,
                             const fg_tower_geometry *geometry,float *hidden,fg_error *err);
fg_status fg_tower_block_cpu(const fg_tower_block_weights *weights,const float *input,
                             uint32_t tokens,uint32_t grid_width,float *output,fg_error *err);
fg_status fg_tower_merger_cpu(const fg_tower_weights *weights,const float *input,
                              uint32_t merged_tokens,float *embeddings,fg_error *err);

#endif
