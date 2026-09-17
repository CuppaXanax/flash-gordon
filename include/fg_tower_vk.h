#ifndef FLASH_GORDON_TOWER_VK_H
#define FLASH_GORDON_TOWER_VK_H

#include "fg.h"
#include "fg_tower.h"

typedef struct fg_tower_vk fg_tower_vk;
typedef struct fg_tower_vk_weights fg_tower_vk_weights;

typedef struct fg_tower_vk_stats {
    uint32_t dispatches;
    double upload_ms;
    double forward_ms;
    double preprocess_ms;
    double setup_ms;
    double host_ms;
    double compute_ms;
    uint64_t weight_bytes;
    uint64_t resident_bytes;
    uint64_t resident_hit_bytes;
} fg_tower_vk_stats;

fg_status fg_tower_vk_open(fg_tower_vk **out,fg_error *err);
void fg_tower_vk_close(fg_tower_vk *tower);
const char *fg_tower_vk_device_name(const fg_tower_vk *tower);
fg_status fg_tower_vk_weights_upload(fg_tower_vk *tower,const fg_tower_weights *weights,
                                     fg_tower_vk_weights **out,fg_error *err);
void fg_tower_vk_weights_destroy(fg_tower_vk *tower,fg_tower_vk_weights *weights);
fg_status fg_tower_vk_run(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                          const float *tokens,const fg_tower_geometry *geometry,
                          float *embeddings,fg_tower_vk_stats *stats,fg_error *err);
fg_status fg_tower_vk_run_patch(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                const float *tokens,const fg_tower_geometry *geometry,
                                float *hidden,fg_tower_vk_stats *stats,fg_error *err);
fg_status fg_tower_vk_run_block(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                uint32_t layer,const float *input,uint32_t tokens,
                                uint32_t grid_width,float *output,fg_tower_vk_stats *stats,
                                fg_error *err);
fg_status fg_tower_vk_run_merger(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                 const float *input,uint32_t merged_tokens,float *embeddings,
                                 fg_tower_vk_stats *stats,fg_error *err);
fg_status fg_tower_vk_debug_layernorm(fg_tower_vk *tower,const float *input,const float *weight,
                                      const float *bias,uint32_t rows,uint32_t width,
                                      float *output,fg_error *err);
fg_status fg_tower_vk_debug_rope(fg_tower_vk *tower,float *qkv,uint32_t tokens,
                                 uint32_t grid_width,fg_error *err);
fg_status fg_tower_vk_debug_attention(fg_tower_vk *tower,const float *qkv,uint32_t tokens,
                                      float *output,fg_error *err);

fg_status fg_tower_vision_forward(const char *tower_dir,const uint8_t *image_bytes,
                                  uint64_t image_length,float **embeddings,
                                  uint32_t *merged_tokens,uint32_t *grid_width,
                                  uint32_t *grid_height,fg_tower_vk_stats *stats,
                                  fg_error *err);

#endif
