#include "fg_output.h"
#include "fg_q38_schema.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct fg_output_executor {
    fg_model *model;
    fg_vk_tensor *normalized;
    fg_vk_tensor *down;
    fg_vk_tensor *activated;
    fg_vk_tensor *up;
    fg_vk_tensor *hidden;
    fg_vk_tensor *logits;
};

static fg_status scratch(fg_vk_context *vk,uint64_t values,fg_vk_tensor **out,fg_error *err){
    return fg_vk_tensor_create(vk,values*sizeof(float),out,err);
}

fg_status fg_output_executor_create(fg_output_executor **out,fg_model *model,fg_error *err){
    if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid output executor arguments");return FG_ERR_ARGUMENT;}
    *out=NULL;
    if(fg_model_rank(model)!=4u){fg_error_set(err,FG_ERR_MISMATCH,"Qwen output executor must run on rank 4");return FG_ERR_MISMATCH;}
    static const char *required[]={"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"};
    for(uint32_t i=0;i<sizeof(required)/sizeof(required[0]);i++)if(!fg_model_tensor(model,required[i])){fg_error_set(err,FG_ERR_MISMATCH,"rank 4 is missing %s",required[i]);return FG_ERR_MISMATCH;}
    fg_output_executor *executor=calloc(1,sizeof(*executor));
    if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate output executor");return FG_ERR_OOM;}
    executor->model=model;fg_vk_context *vk=fg_model_vk(model);
    fg_status status=scratch(vk,FG_Q38_HYPER_WIDTH,&executor->normalized,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_RANK,&executor->down,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_RANK,&executor->activated,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_WIDTH,&executor->up,err);
    if(status==FG_OK)status=scratch(vk,FG_HIDDEN_SIZE,&executor->hidden,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_VOCAB_SIZE,&executor->logits,err);
    if(status!=FG_OK){fg_output_executor_destroy(executor);return status;}
    *out=executor;return FG_OK;
}

void fg_output_executor_destroy(fg_output_executor *executor){
    if(!executor)return;
    fg_vk_tensor_destroy(executor->logits);fg_vk_tensor_destroy(executor->hidden);
    fg_vk_tensor_destroy(executor->up);fg_vk_tensor_destroy(executor->activated);
    fg_vk_tensor_destroy(executor->down);fg_vk_tensor_destroy(executor->normalized);free(executor);
}

fg_status fg_output_logits(fg_output_executor *executor,const fg_vk_tensor *hyper,fg_vk_tensor **logits,fg_error *err){
    if(!executor||!hyper||!logits||fg_vk_tensor_bytes(hyper)<FG_Q38_HYPER_WIDTH*sizeof(float)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Qwen output arguments");return FG_ERR_ARGUMENT;}
    fg_vk_context *vk=fg_model_vk(executor->model);
    fg_status status=fg_vk_group_rms_norm(vk,executor->normalized,hyper,fg_model_tensor(executor->model,"output_hc_norm.weight"),FG_HIDDEN_SIZE,FG_Q38_HYPER_COUNT,1u,1e-6f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->down,fg_model_tensor(executor->model,"output_hc_down.weight"),executor->normalized,FG_Q38_HYPER_WIDTH,FG_Q38_HYPER_RANK,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_silu_scaled(vk,executor->activated,executor->down,FG_Q38_HYPER_RANK,1.0f/(float)FG_Q38_HYPER_COUNT,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->up,fg_model_tensor(executor->model,"output_hc_up.weight"),executor->activated,FG_Q38_HYPER_RANK,FG_Q38_HYPER_WIDTH,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_hc_finalize(vk,executor->hidden,executor->normalized,executor->up,FG_HIDDEN_SIZE,FG_Q38_HYPER_COUNT,1u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->logits,fg_model_tensor(executor->model,"output.weight"),executor->hidden,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,1u,1.0f,err);
    if(status==FG_OK)*logits=executor->logits;
    return status;
}

fg_status fg_output_greedy(fg_output_executor *executor,const fg_vk_tensor *hyper,uint32_t *token,float *logit,fg_error *err){
    if(!token){fg_error_set(err,FG_ERR_ARGUMENT,"greedy token output is null");return FG_ERR_ARGUMENT;}
    fg_vk_tensor *logits=NULL;fg_status status=fg_output_logits(executor,hyper,&logits,err);if(status!=FG_OK)return status;
    const float *values=fg_vk_tensor_map(logits);uint32_t best=0u;float best_value=values[0];
    if(!isfinite(best_value)){fg_error_set(err,FG_ERR_MISMATCH,"non-finite output logit at token 0");return FG_ERR_MISMATCH;}
    uint32_t second=0u,third=0u;float second_value=-1e30f,third_value=-1e30f;
    for(uint32_t i=1;i<FG_Q38_VOCAB_SIZE;i++){if(!isfinite(values[i])){fg_error_set(err,FG_ERR_MISMATCH,"non-finite output logit at token %u",i);return FG_ERR_MISMATCH;}if(values[i]>best_value){third=second;third_value=second_value;second=best;second_value=best_value;best=i;best_value=values[i];}else if(values[i]>second_value){third=second;third_value=second_value;second=i;second_value=values[i];}else if(values[i]>third_value){third=i;third_value=values[i];}}
    const float *hyper_raw=fg_vk_tensor_map((fg_vk_tensor *)hyper);const float *hidden_raw=fg_vk_tensor_map(executor->hidden);
    fprintf(stderr,"greedy: top3 %u=%.4f %u=%.4f %u=%.4f hyper[0:4]=%.4f,%.4f,%.4f,%.4f hidden[0:4]=%.4f,%.4f,%.4f,%.4f\n",best,best_value,second,second_value,third,third_value,hyper_raw[0],hyper_raw[1],hyper_raw[2],hyper_raw[3],hidden_raw[0],hidden_raw[1],hidden_raw[2],hidden_raw[3]);
    *token=best;if(logit)*logit=best_value;return FG_OK;
}
