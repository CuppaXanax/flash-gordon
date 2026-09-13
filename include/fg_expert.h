#ifndef FLASH_GORDON_EXPERT_H
#define FLASH_GORDON_EXPERT_H

#include "fg_model.h"
#include "fg_protocol.h"

typedef struct fg_expert_executor fg_expert_executor;

uint32_t fg_expert_local_count(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank);
uint32_t fg_expert_local_index(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank,uint32_t global_expert);
fg_status fg_expert_executor_create(fg_expert_executor **out,fg_model *model,fg_error *err);
void fg_expert_executor_destroy(fg_expert_executor *executor);
fg_status fg_expert_decode(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err);
fg_status fg_expert_decode_submit(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err);
fg_status fg_expert_decode_finish(fg_expert_executor *executor,fg_expert_result *result,fg_error *err);
/* Chained decode: record router top-10 and the fused expert pair into the
 * caller's active Vulkan batch; no host routing round trip or fence. */
fg_status fg_expert_decode_chain_ready(fg_expert_executor *executor,uint32_t layer,fg_error *err);
fg_status fg_expert_decode_chain(fg_expert_executor *executor,uint32_t layer,
                                 const fg_vk_tensor *activation,const fg_vk_tensor *router_logits,
                                 fg_vk_tensor **reduced,fg_error *err);
fg_status fg_expert_prefill(fg_expert_executor *executor,const fg_prefill_work *work,
                            fg_prefill_result *result,fg_prefill_result_pair *pair_storage,
                            uint32_t pair_capacity,float *output_storage,
                            uint64_t output_capacity_values,fg_error *err);
fg_status fg_expert_prefill_enqueue(fg_expert_executor *executor,const fg_prefill_work *work,
                                    fg_prefill_result *result,fg_prefill_result_pair *pair_storage,
                                    uint32_t pair_capacity,float *output_storage,
                                    uint64_t output_capacity_values,fg_error *err);
fg_status fg_expert_prefill_finish(fg_expert_executor *executor,const fg_prefill_work *work,
                                   fg_prefill_result *result,fg_prefill_result_pair *pair_storage,
                                   float *output_storage,fg_error *err);

#endif
