#ifndef FLASH_GORDON_Q38_MATH_H
#define FLASH_GORDON_Q38_MATH_H

#include "fg.h"

float fg_q5_1_dot_f32(const uint8_t *row,const float *x,uint32_t n);
void fg_q38_group_rms_norm(float *out,const float *x,const float *zero_centered_weight,
                           uint32_t groups,uint32_t width,float eps);
fg_status fg_q38_router_topk(const float *logits,uint32_t count,uint32_t k,
                             uint32_t *ids,float *weights,fg_error *err);
fg_status fg_q38_rms_mrope(float *vector,uint32_t heads,uint32_t head_width,
                           const float *zero_centered_weight,const uint32_t position[3],
                           fg_error *err);
fg_status fg_q38_qsa_index_select_reference(const float query[512],const float *raw_keys,
                                            uint32_t token_count,const float q_norm[128],
                                            const float k_norm[128],uint32_t *selected_tokens,
                                            uint32_t capacity,uint32_t *selected_count,
                                            fg_error *err);

#endif
