#ifndef FLASH_GORDON_OUTPUT_H
#define FLASH_GORDON_OUTPUT_H

#include "fg_model.h"
#include "fg_protocol.h"
#include "fg_sampler.h"

typedef struct fg_output_executor fg_output_executor;
typedef struct fg_output_slice fg_output_slice;
typedef struct fg_output_hc fg_output_hc;

/* Vocabulary rows [0, FG_OUTPUT_SPLIT_FIRST_ROWS) stay on the output owner;
 * rank 0 reduces the remaining rows and returns its per-slice argmax.  The
 * split boundary must stay a multiple of FG_Q8_0_COOK_ROWS. */
static inline uint32_t fg_output_owner_rank(const fg_manifest *manifest){
    (void)manifest;
    return 4u;
}

fg_status fg_output_slice_create(fg_output_slice **out,fg_model *model,uint32_t ways,
                                 uint32_t first_row,uint32_t rows,fg_error *err);
fg_status fg_output_slice_create_foreign(fg_output_slice **out,fg_model *model,
                                         const char *pack_dir,uint32_t ways,
                                         uint32_t first_row,uint32_t rows,fg_error *err);
void fg_output_slice_destroy(fg_output_slice *slice);
uint32_t fg_output_slice_ways(const fg_output_slice *slice);
fg_status fg_output_slice_run(fg_output_slice *slice,const void *hyper,
                              float *value,uint32_t *id,fg_error *err);
fg_status fg_output_slice_run_hidden(fg_output_slice *slice,const void *hidden,
                                     float *value,uint32_t *id,fg_error *err);

fg_status fg_output_hc_create(fg_output_hc **out,fg_model *model,const char *pack_dir,
                              bool foreign,fg_error *err);
void fg_output_hc_destroy(fg_output_hc *hc);
fg_status fg_output_hc_run(fg_output_hc *hc,const void *hyper,float *hidden,
                           fg_error *err);

bool fg_output_better(float left,uint32_t left_id,float right,uint32_t right_id);
void fg_output_combine(float left,uint32_t left_id,float right,uint32_t right_id,
                       float *value,uint32_t *id);

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
