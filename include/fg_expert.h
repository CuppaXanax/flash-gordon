#ifndef FLASH_GORDON_EXPERT_H
#define FLASH_GORDON_EXPERT_H

#include "fg_model.h"
#include "fg_protocol.h"

typedef struct fg_expert_executor fg_expert_executor;

uint32_t fg_expert_local_count(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank);
uint32_t fg_expert_local_index(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank,uint32_t global_expert);
bool fg_expert_uses_rank_suffix(const fg_manifest *manifest);
fg_status fg_expert_executor_create(fg_expert_executor **out,fg_model *model,fg_error *err);
void fg_expert_executor_destroy(fg_expert_executor *executor);
fg_status fg_expert_decode(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err);
fg_status fg_expert_prefill(fg_expert_executor *executor,const fg_prefill_work *work,
                            fg_prefill_result *result,fg_prefill_result_pair *pair_storage,
                            uint32_t pair_capacity,float *output_storage,
                            uint64_t output_capacity_values,fg_error *err);
fg_status fg_expert_prefill_pipeline(fg_expert_executor *executor,uint32_t layer,
                                    uint16_t token_count,
                                    const fg_vk_tensor *activation_q8k,
                                    const fg_vk_tensor *tiles,
                                    fg_vk_tensor **output,fg_error *err);
fg_status fg_expert_decode_pipeline(fg_expert_executor *executor,uint32_t layer,
                                    const fg_vk_tensor *activation_q8k,
                                    const fg_vk_tensor *selected,
                                    const fg_vk_tensor *gates,
                                    const fg_vk_tensor *shared_output,
                                    const fg_vk_tensor *shared_logit,
                                    fg_vk_tensor **output,fg_error *err);

#endif
