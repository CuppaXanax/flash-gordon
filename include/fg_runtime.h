#ifndef FLASH_GORDON_RUNTIME_H
#define FLASH_GORDON_RUNTIME_H

#include "fg_manifest.h"
#include "fg_prefix.h"

typedef struct fg_runtime fg_runtime;
#define FG_RUNTIME_BOOT_CONTEXT_TOKENS FG_MANIFEST_DEFAULT_CONTEXT_TOKENS

enum {
    FG_RUNTIME_EXPERIMENTAL_CONTEXT = 1u << 0,
    FG_RUNTIME_EXPERIMENTAL_MTP = 1u << 1,
    FG_RUNTIME_EXPERIMENTAL_VISION = 1u << 2
};

enum {
    FG_RUNTIME_OPTION_LOGICAL_CONTEXT = 1u << 0,
    FG_RUNTIME_OPTION_GPU_INDEX = 1u << 1,
    FG_RUNTIME_OPTION_QSA_HOT = 1u << 2,
    FG_RUNTIME_OPTION_PAGE_CACHE = 1u << 3,
    FG_RUNTIME_OPTION_PREFILL_MICROBATCH = 1u << 4,
    FG_RUNTIME_OPTION_PREFILL_WINDOW = 1u << 5
};

typedef struct fg_runtime_options {
    uint32_t logical_context_tokens;
    uint32_t gpu_index_tokens;
    uint32_t qsa_hot_tokens;
    uint64_t qsa_page_cache_bytes;
    uint32_t prefill_microbatch;
    uint32_t prefill_window;
    uint32_t experimental_flags;
    uint32_t specified;
} fg_runtime_options;

typedef struct fg_generation_stats {
    uint32_t prompt_tokens;
    uint32_t prefilled_tokens;
    uint32_t reused_tokens;
    uint32_t generated_tokens;
    uint32_t context_tokens;
    bool prefix_cache_hit;
    bool exact_frontier;
    fg_prefix_reset_reason reset_reason;
    double prefill_seconds;
    double decode_seconds;
} fg_generation_stats;
typedef fg_status (*fg_token_callback)(void *context,uint32_t token,const char *text,
                                      size_t bytes,fg_error *err);
typedef bool (*fg_interrupt_fn)(void *context);

void fg_runtime_options_init(fg_runtime_options *options);
fg_status fg_runtime_options_resolve(fg_runtime_options *resolved,
                                     const fg_manifest *manifest,
                                     const fg_runtime_options *requested,
                                     fg_error *err);
fg_status fg_runtime_eval_capacity(uint32_t *qsa_capacity,
                                   const fg_runtime_options *options,
                                   size_t prompt_tokens,uint32_t generation_tokens,
                                   fg_error *err);
fg_status fg_runtime_open(fg_runtime **out,const char *manifest_path,fg_error *err);
fg_status fg_runtime_open_with_options(fg_runtime **out,const char *manifest_path,
                                       const fg_runtime_options *options,fg_error *err);
void fg_runtime_close(fg_runtime *runtime);
fg_status fg_runtime_reset(fg_runtime *runtime,fg_error *err);
fg_status fg_runtime_reset_public_history(fg_runtime *runtime,fg_error *err);
fg_status fg_runtime_generate(fg_runtime *runtime,const char *rendered_transcript,
                              uint32_t max_tokens,
                              fg_token_callback callback,void *callback_context,
                              fg_interrupt_fn interrupted,void *interrupt_context,
                              fg_generation_stats *stats,fg_error *err);
fg_status fg_runtime_generate_continuation(
    fg_runtime *runtime,const char *rendered_continuation,bool *prefix_miss,
    uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err);
uint32_t fg_runtime_context_tokens(const fg_runtime *runtime);
uint32_t fg_runtime_context_limit(const fg_runtime *runtime);
const char *fg_runtime_model_name(const fg_runtime *runtime);

fg_status fg_rank_main(const char *manifest_path, uint32_t rank, fg_error *err);
fg_status fg_serve_main(const char *manifest_path, fg_error *err);
fg_status fg_bench_main(const char *manifest_path, fg_error *err);
fg_status fg_eval_main(const char *manifest_path,const char *prompt,uint32_t generate,fg_error *err);

#endif
