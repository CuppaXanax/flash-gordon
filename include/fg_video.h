#ifndef FLASH_GORDON_VIDEO_H
#define FLASH_GORDON_VIDEO_H

#include "fg.h"
#include "fg_tower.h"
#include "fg_tower_vk.h"

#define FG_VIDEO_FPS_TARGET 2.0
#define FG_VIDEO_MAX_FRAMES 8u
#define FG_VIDEO_MAX_PIXELS UINT64_C(262144)
#define FG_VIDEO_MAX_TOKENS 4096u
#define FG_VIDEO_MAX_BYTES (UINT64_C(64) << 20u)
#define FG_VIDEO_MAX_INPUT_FRAMES 256u

typedef struct fg_video_source {
    uint32_t width;
    uint32_t height;
    double fps;
    double duration_seconds;
} fg_video_source;

typedef struct fg_video_options {
    double fps;
    uint32_t max_frames;
    uint64_t max_pixels;
    uint32_t max_tokens;
} fg_video_options;

typedef struct fg_video_clip {
    float *embeddings;
    uint32_t token_count;
    uint32_t tokens_per_pair;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t pair_count;
    uint32_t frame_count;
} fg_video_clip;

void fg_video_options_defaults(fg_video_options *options);

uint32_t fg_video_select_frames(uint32_t frame_count,double source_fps,double target_fps,
                                uint32_t max_frames,uint32_t *indices);

fg_status fg_video_frames_preprocess(const uint8_t *const *frames,const size_t *lengths,
                                     uint32_t frame_count,const uint32_t *indices,
                                     uint32_t selected_count,const fg_video_options *options,
                                     float **tokens,uint32_t *pair_count,
                                     fg_tower_geometry *geometry,fg_error *err);

fg_status fg_video_frames_forward(const char *tower_dir,const uint8_t *const *frames,
                                  const size_t *lengths,uint32_t frame_count,double source_fps,
                                  const fg_video_options *options,fg_video_clip *clip,
                                  fg_tower_vk_stats *stats,fg_error *err);

fg_status fg_video_superframes(const float *frames,uint32_t frame_count,
                               uint32_t width,uint32_t height,float **tokens,
                               uint32_t *pair_count,fg_tower_geometry *geometry,
                               fg_error *err);

uint64_t fg_video_token_count(uint32_t frame_count,uint32_t tokens_per_pair);

uint32_t fg_video_positions(uint32_t *positions,uint32_t token_count,uint32_t grid_width,
                            uint32_t grid_height,uint32_t temporal_groups,
                            uint32_t position);

fg_status fg_video_probe(const char *path,const char *tool_dir,fg_video_source *source,
                         fg_error *err);

fg_status fg_video_extract(const char *path,const char *tool_dir,
                           const fg_video_options *options,float **frames,
                           uint32_t *frame_count,uint32_t *frame_width,uint32_t *frame_height,
                           fg_error *err);

fg_status fg_video_forward(const char *tower_dir,const uint8_t *bytes,size_t length,
                           const fg_video_options *options,fg_video_clip *clip,
                           fg_tower_vk_stats *stats,fg_error *err);

bool fg_video_available(void);
bool fg_video_mp4_available(const char *tool_dir);

fg_status fg_tower_vision_forward_tokens(const char *tower_dir,const float *tokens,
                                         uint32_t tokens_per_pair,uint32_t pair_count,
                                         const fg_tower_geometry *geometry,
                                         float **embeddings,fg_tower_vk_stats *stats,
                                         fg_error *err) __attribute__((weak));

#endif
