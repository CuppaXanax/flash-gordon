#ifndef FLASH_GORDON_OUTPUT_H
#define FLASH_GORDON_OUTPUT_H

#include "fg_model.h"

typedef struct fg_output_executor fg_output_executor;

fg_status fg_output_executor_create(fg_output_executor **out,fg_model *model,fg_error *err);
void fg_output_executor_destroy(fg_output_executor *executor);
fg_status fg_output_logits(fg_output_executor *executor,const fg_vk_tensor *hyper,
                           fg_vk_tensor **logits,fg_error *err);
fg_status fg_output_greedy(fg_output_executor *executor,const fg_vk_tensor *hyper,
                           uint32_t *token,float *logit,fg_error *err);

#endif
