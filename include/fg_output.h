#ifndef FLASH_GORDON_OUTPUT_H
#define FLASH_GORDON_OUTPUT_H

#include "fg_model.h"
#include "fg_sampler.h"

typedef struct fg_output_executor fg_output_executor;

static inline uint32_t fg_output_owner_rank(const fg_manifest *manifest){
    return manifest&&manifest->execution_mode==FG_EXECUTION_PIPELINE&&
        manifest->stage_count?manifest->stage_ranks[manifest->stage_count-1u]:4u;
}

fg_status fg_output_executor_create(fg_output_executor **out,fg_model *model,fg_error *err);
void fg_output_executor_destroy(fg_output_executor *executor);
fg_status fg_output_history_reset(fg_output_executor *executor,
                                  const uint32_t *history,uint32_t count,
                                  fg_error *err);
fg_status fg_output_history_increment(fg_output_executor *executor,uint32_t token,
                                       fg_error *err);
fg_status fg_output_logits(fg_output_executor *executor,const fg_vk_tensor *hyper,
                           fg_vk_tensor **logits,fg_error *err);
fg_status fg_output_greedy(fg_output_executor *executor,const fg_vk_tensor *hyper,
                           uint32_t *token,float *logit,fg_error *err);
/* Produce exactly K finite vocabulary candidates. Candidate order is
   unspecified; callers sort before applying additional sampling filters.
   The returned tensor views are executor-owned scratch and remain valid until
   the next output operation or executor destruction. */
fg_status fg_output_topk(fg_output_executor *executor,const fg_vk_tensor *hyper,
                         uint32_t k,fg_vk_tensor **scores,fg_vk_tensor **ids,
                         uint32_t *count,fg_error *err);
fg_status fg_output_sample(fg_output_executor *executor,const fg_vk_tensor *hyper,
                           const fg_sampler_config *config,
                           float uniform,uint32_t *token,float *logit,fg_error *err);

#endif
