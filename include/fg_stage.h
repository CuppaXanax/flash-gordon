#ifndef FLASH_GORDON_STAGE_H
#define FLASH_GORDON_STAGE_H

#include "fg_output.h"
#include "fg_owner.h"
#include "fg_pipeline.h"

typedef struct fg_stage_executor fg_stage_executor;

typedef fg_status (*fg_stage_ngram_decode_fn)(
    void *context,uint64_t request_id,uint32_t sequence,uint32_t token_index,
    fg_vk_tensor **embedding,fg_error *err);
typedef fg_status (*fg_stage_ngram_prefill_fn)(
    void *context,uint64_t request_id,uint32_t sequence,uint32_t first_token,
    uint16_t token_count,fg_vk_tensor **embeddings,fg_error *err);

typedef struct fg_stage_config {
    fg_model *model;
    const char *qsa_state_path;
    bool create_qsa_state;
    fg_stage_ngram_decode_fn ngram_decode;
    fg_stage_ngram_prefill_fn ngram_prefill;
    void *ngram_context;
} fg_stage_config;

fg_status fg_stage_validate_manifest(const fg_manifest *manifest,uint32_t rank,
                                     uint32_t *stage,uint32_t *layer_begin,
                                     uint32_t *layer_end,fg_error *err);
fg_status fg_stage_executor_create(fg_stage_executor **out,
                                   const fg_stage_config *config,fg_error *err);
void fg_stage_executor_close(fg_stage_executor *executor);
fg_status fg_stage_executor_reset(fg_stage_executor *executor,fg_error *err);

fg_status fg_stage_decode(fg_stage_executor *executor,uint32_t token_index,
                          const uint32_t position[FG_PIPELINE_POSITION_AXES],
                          const fg_vk_tensor *input,
                          const fg_vk_tensor *ngram_embedding,
                          fg_vk_tensor **output,fg_pipeline_result *terminal_result,
                          fg_error *err);
/* request_output is forwarded on every stage and only materialized by the terminal stage. */
fg_status fg_stage_prefill(fg_stage_executor *executor,uint32_t first_token,
                           const uint32_t *positions,uint16_t token_count,
                           bool request_output,
                           const fg_vk_tensor *input,
                           const fg_vk_tensor *ngram_embeddings,
                           fg_vk_tensor **output,fg_pipeline_result *terminal_result,
                           fg_error *err);

fg_status fg_stage_pipeline_execute(void *context,uint32_t stage,
                                    uint64_t request_id,uint32_t sequence,
                                    fg_pipeline_activation *activation,
                                    float *boundary,
                                    fg_pipeline_result *terminal_result,
                                    fg_error *err);

#endif
