#ifndef FLASH_GORDON_PIPELINE_RUNTIME_H
#define FLASH_GORDON_PIPELINE_RUNTIME_H

#include "fg_pipeline.h"

typedef struct fg_pipeline_runtime fg_pipeline_runtime;

typedef fg_status (*fg_pipeline_prepare_fn)(
    void *context,fg_pipeline_execution_kind kind,const uint32_t *token_ids,
    uint32_t first_token,uint16_t token_count,uint32_t *positions,float *boundary,
    fg_error *err);
typedef fg_status (*fg_pipeline_progress_fn)(
    void *context,fg_pipeline *pipeline,fg_error *err);

typedef struct fg_pipeline_runtime_config {
    const fg_manifest *manifest;
    fg_pipeline *pipeline;
    fg_pipeline_prepare_fn prepare;
    fg_pipeline_progress_fn progress;
    void *context;
} fg_pipeline_runtime_config;

fg_status fg_pipeline_runtime_create(fg_pipeline_runtime **out,
                                     const fg_pipeline_runtime_config *config,
                                     fg_error *err);
void fg_pipeline_runtime_destroy(fg_pipeline_runtime *runtime);
fg_status fg_pipeline_runtime_begin(fg_pipeline_runtime *runtime,
                                    uint64_t request_id,uint32_t first_sequence,
                                    fg_error *err);
fg_status fg_pipeline_runtime_prefill(fg_pipeline_runtime *runtime,
                                      const uint32_t *token_ids,
                                      uint32_t first_token,uint32_t token_count,
                                      fg_pipeline_result *terminal,
                                      double *seconds,fg_error *err);
fg_status fg_pipeline_runtime_decode(fg_pipeline_runtime *runtime,
                                     uint32_t token_id,uint32_t token_index,
                                     fg_pipeline_result *terminal,
                                     fg_error *err);
fg_status fg_pipeline_runtime_finish(fg_pipeline_runtime *runtime,fg_error *err);
bool fg_pipeline_runtime_reopen_required(const fg_pipeline_runtime *runtime);

#endif
