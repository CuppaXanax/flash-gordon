#ifndef FLASH_GORDON_OWNER_H
#define FLASH_GORDON_OWNER_H

#include "fg_expert.h"

typedef struct fg_owner_executor fg_owner_executor;
typedef fg_status (*fg_owner_expert_dispatch_fn)(void *context,uint32_t layer,uint32_t token_index,
                                                  const uint16_t expert_ids[FG_TOP_K],
                                                  const float gates[FG_TOP_K],const uint8_t *activation_q8k,
                                                  fg_expert_result results[FG_GROUP_SIZE],
                                                  uint32_t *result_count,fg_error *err);
typedef fg_status (*fg_owner_prefill_dispatch_fn)(void *context,uint32_t layer,
                                                   uint32_t first_token,uint16_t token_count,
                                                   const uint16_t *expert_ids,const float *gates,
                                                   const uint8_t *activations_q8k,
                                                   fg_prefill_result results[FG_GROUP_SIZE],
                                                   uint32_t *result_count,fg_error *err);

fg_status fg_owner_executor_create(fg_owner_executor **out,fg_model *model,fg_error *err);
void fg_owner_executor_destroy(fg_owner_executor *executor);
fg_status fg_owner_gr_read(fg_owner_executor *executor,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,
                           fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err);
fg_status fg_owner_gr_read_batch(fg_owner_executor *executor,uint32_t layer,bool ffn,
                                 const fg_vk_tensor *hyper_input,uint32_t token_count,
                                 fg_vk_tensor **mixed,const fg_vk_tensor **residual,
                                 fg_vk_tensor **injection,fg_error *err);
fg_status fg_owner_moe_prepare(fg_owner_executor *executor,uint32_t layer,const fg_vk_tensor *hidden,
                               uint16_t expert_ids[FG_TOP_K],float gates[FG_TOP_K],
                               const uint8_t **activation_q8k,fg_error *err);
fg_status fg_owner_moe_prepare_batch(fg_owner_executor *executor,uint32_t layer,
                                     const fg_vk_tensor *hidden,uint16_t token_count,
                                     uint16_t *expert_ids,float *gates,
                                     const uint8_t **activation_q8k,fg_error *err);
fg_status fg_owner_moe_reduce(fg_owner_executor *executor,uint32_t layer,uint32_t position,
                              const uint16_t expert_ids[FG_TOP_K],
                              const float gates[FG_TOP_K],const fg_expert_result *results,
                              uint32_t result_count,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_moe_reduce_batch(fg_owner_executor *executor,uint32_t layer,
                                    uint32_t first_position,uint16_t token_count,
                                    const uint16_t *expert_ids,const float *gates,
                                    const fg_prefill_result *results,uint32_t result_count,
                                    fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gr_write(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                            const fg_vk_tensor *block_output,const fg_vk_tensor *injection,
                            fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gr_write_batch(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                                  const fg_vk_tensor *block_output,const fg_vk_tensor *injection,
                                  uint32_t token_count,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gdn_decode(fg_owner_executor *executor,uint32_t layer,const fg_vk_tensor *hidden,
                              fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_gdn_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t token_count,
                               const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_ple_decode(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                              const fg_vk_tensor *ngram_embedding,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_ple_prefill(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,
                               const fg_vk_tensor *ngram_embeddings,uint32_t token_count,
                               fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_qsa_open(fg_owner_executor *executor,const char *state_path,bool create,
                            fg_error *err);
fg_status fg_owner_qsa_decode(fg_owner_executor *executor,uint32_t layer,uint32_t token_index,
                              const uint32_t position[3],const fg_vk_tensor *hidden,
                              fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_qsa_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t first_token,
                               const uint32_t *positions,uint32_t token_count,
                               const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_decode_layer(fg_owner_executor *executor,uint32_t layer,uint32_t token_index,
                                const uint32_t position[3],const fg_vk_tensor *hyper_input,
                                const fg_vk_tensor *ngram_embedding,
                                fg_owner_expert_dispatch_fn dispatch,void *dispatch_context,
                                fg_vk_tensor **output,fg_error *err);
fg_status fg_owner_prefill_layer(fg_owner_executor *executor,uint32_t layer,
                                 uint32_t first_token,const uint32_t *positions,
                                 uint16_t token_count,const fg_vk_tensor *hyper_input,
                                 const fg_vk_tensor *ngram_embeddings,
                                 fg_owner_prefill_dispatch_fn dispatch,void *dispatch_context,
                                 fg_vk_tensor **output,fg_error *err);

#endif
