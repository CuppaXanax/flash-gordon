#include "fg_output.h"
#include "fg_q38_schema.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FG_OUTPUT_TOPK_CAPACITY (((FG_Q38_VOCAB_SIZE+4095u)/4096u)*512u)

struct fg_output_executor {
    fg_model *model;
    fg_vk_tensor *normalized;
    fg_vk_tensor *down;
    fg_vk_tensor *activated;
    fg_vk_tensor *up;
    fg_vk_tensor *hidden;
    fg_vk_tensor *logits;
    fg_vk_tensor *history_counts;
    fg_vk_tensor *vocabulary_ids;
    fg_vk_tensor *topk_scores[2];
    fg_vk_tensor *topk_ids[2];
};

static fg_status scratch(fg_vk_context *vk,uint64_t values,fg_vk_tensor **out,fg_error *err){
    return fg_vk_tensor_create(vk,values*sizeof(float),out,err);
}

fg_status fg_output_executor_create(fg_output_executor **out,fg_model *model,fg_error *err){
    if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid output executor arguments");return FG_ERR_ARGUMENT;}
    *out=NULL;
    uint32_t owner=fg_output_owner_rank(fg_model_manifest(model));
    if(fg_model_rank(model)!=owner){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "Qwen output executor must run on rank %u",owner);
        return FG_ERR_MISMATCH;
    }
    static const char *required[]={"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"};
    for(uint32_t i=0;i<sizeof(required)/sizeof(required[0]);i++)
        if(!fg_model_tensor(model,required[i])){
            fg_error_set(err,FG_ERR_MISMATCH,"rank %u is missing %s",
                         owner,required[i]);
            return FG_ERR_MISMATCH;
        }
    fg_output_executor *executor=calloc(1,sizeof(*executor));
    if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate output executor");return FG_ERR_OOM;}
    executor->model=model;fg_vk_context *vk=fg_model_vk(model);
    fg_status status=scratch(vk,FG_Q38_HYPER_WIDTH,&executor->normalized,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_RANK,&executor->down,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_RANK,&executor->activated,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_WIDTH,&executor->up,err);
    if(status==FG_OK)status=scratch(vk,FG_HIDDEN_SIZE,&executor->hidden,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_VOCAB_SIZE,&executor->logits,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)FG_Q38_VOCAB_SIZE*4u,&executor->history_counts,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)FG_Q38_VOCAB_SIZE*4u,&executor->vocabulary_ids,err);
    for(uint32_t i=0;status==FG_OK&&i<2u;i++){status=scratch(vk,FG_OUTPUT_TOPK_CAPACITY,&executor->topk_scores[i],err);if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)FG_OUTPUT_TOPK_CAPACITY*4u,&executor->topk_ids[i],err);}
    if(status==FG_OK){uint32_t *ids=fg_vk_tensor_map(executor->vocabulary_ids);for(uint32_t i=0;i<FG_Q38_VOCAB_SIZE;i++)ids[i]=i;}
    if(status!=FG_OK){fg_output_executor_destroy(executor);return status;}
    memset(fg_vk_tensor_map(executor->history_counts),0,
           (size_t)fg_vk_tensor_bytes(executor->history_counts));
    *out=executor;return FG_OK;
}

void fg_output_executor_destroy(fg_output_executor *executor){
    if(!executor)return;
    for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(executor->topk_ids[i]);fg_vk_tensor_destroy(executor->topk_scores[i]);}
    fg_vk_tensor_destroy(executor->vocabulary_ids);
    fg_vk_tensor_destroy(executor->history_counts);
    fg_vk_tensor_destroy(executor->logits);fg_vk_tensor_destroy(executor->hidden);
    fg_vk_tensor_destroy(executor->up);fg_vk_tensor_destroy(executor->activated);
    fg_vk_tensor_destroy(executor->down);fg_vk_tensor_destroy(executor->normalized);free(executor);
}

fg_status fg_output_logits(fg_output_executor *executor,const fg_vk_tensor *hyper,fg_vk_tensor **logits,fg_error *err){
    if(!executor||!hyper||!logits||fg_vk_tensor_bytes(hyper)<FG_Q38_HYPER_WIDTH*sizeof(float)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Qwen output arguments");return FG_ERR_ARGUMENT;}
    fg_vk_context *vk=fg_model_vk(executor->model);
    fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"output",err):FG_OK;
    if(status==FG_OK)status=fg_vk_group_rms_norm(vk,executor->normalized,hyper,fg_model_tensor(executor->model,"output_hc_norm.weight"),FG_HIDDEN_SIZE,FG_Q38_HYPER_COUNT,1u,1e-6f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->down,fg_model_tensor(executor->model,"output_hc_down.weight"),executor->normalized,FG_Q38_HYPER_WIDTH,FG_Q38_HYPER_RANK,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_silu_scaled(vk,executor->activated,executor->down,FG_Q38_HYPER_RANK,1.0f/(float)FG_Q38_HYPER_COUNT,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->up,fg_model_tensor(executor->model,"output_hc_up.weight"),executor->activated,FG_Q38_HYPER_RANK,FG_Q38_HYPER_WIDTH,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_hc_finalize(vk,executor->hidden,executor->normalized,executor->up,FG_HIDDEN_SIZE,FG_Q38_HYPER_COUNT,1u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->logits,fg_model_tensor(executor->model,"output.weight"),executor->hidden,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,1u,1.0f,err);
    if(status==FG_OK)*logits=executor->logits;
    return status;
}

fg_status fg_output_greedy(fg_output_executor *executor,const fg_vk_tensor *hyper,uint32_t *token,float *logit,fg_error *err){
    if(!executor||!token){fg_error_set(err,FG_ERR_ARGUMENT,"invalid greedy output arguments");return FG_ERR_ARGUMENT;}
    fg_vk_context *vk=fg_model_vk(executor->model);if(fg_vk_batch_active(vk)){fg_error_set(err,FG_ERR_ARGUMENT,"greedy output cannot run inside a Vulkan batch");return FG_ERR_ARGUMENT;}
    fg_vk_tensor *logits=NULL;fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_output_logits(executor,hyper,&logits,err);const fg_vk_tensor *scores=logits,*ids=executor->vocabulary_ids;uint32_t count=FG_Q38_VOCAB_SIZE,slot=0;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"output_argmax",err);
    while(status==FG_OK&&count>1u){uint32_t next=0;status=fg_vk_argmax_reduce(vk,executor->topk_scores[slot],executor->topk_ids[slot],scores,ids,count,&next,err);scores=executor->topk_scores[slot];ids=executor->topk_ids[slot];count=next;slot^=1u;}
    if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status!=FG_OK&&fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status!=FG_OK)return status;
    const float *values=fg_vk_tensor_map((fg_vk_tensor *)scores);const uint32_t *indices=fg_vk_tensor_map((fg_vk_tensor *)ids);uint32_t best=indices[0];float best_value=values[0];
    if(best>=FG_Q38_VOCAB_SIZE||!isfinite(best_value)){fg_error_set(err,FG_ERR_MISMATCH,"invalid output finalist at token %u",best);return FG_ERR_MISMATCH;}
    const float *hyper_raw=fg_vk_tensor_map((fg_vk_tensor *)hyper);const float *hidden_raw=fg_vk_tensor_map(executor->hidden);
    fprintf(stderr,"greedy: best %u=%.4f hyper[0:4]=%.4f,%.4f,%.4f,%.4f hidden[0:4]=%.4f,%.4f,%.4f,%.4f\n",best,best_value,hyper_raw[0],hyper_raw[1],hyper_raw[2],hyper_raw[3],hidden_raw[0],hidden_raw[1],hidden_raw[2],hidden_raw[3]);
    *token=best;if(logit)*logit=best_value;return FG_OK;
}

fg_status fg_output_history_reset(fg_output_executor *executor,
                                  const uint32_t *history,uint32_t count,
                                  fg_error *err){
    if(!executor||count>FG_NATIVE_CONTEXT||
       (count&&!history)){fg_error_set(err,FG_ERR_ARGUMENT,
           "invalid output history");return FG_ERR_ARGUMENT;}
    uint32_t *counts=fg_vk_tensor_map(executor->history_counts);
    if(!counts){fg_error_set(err,FG_ERR_UNAVAILABLE,
        "output history storage is unavailable");return FG_ERR_UNAVAILABLE;}
    memset(counts,0,(size_t)fg_vk_tensor_bytes(executor->history_counts));
    for(uint32_t i=0;i<count;i++){
        if(history[i]>=FG_Q38_VOCAB_SIZE){fg_error_set(err,FG_ERR_FORMAT,
            "history token %u is outside vocabulary",i);return FG_ERR_FORMAT;}
        if(counts[history[i]]!=UINT32_MAX)counts[history[i]]++;
    }
    return FG_OK;
}

fg_status fg_output_history_increment(fg_output_executor *executor,uint32_t token,
                                       fg_error *err){
    if(!executor||token>=FG_Q38_VOCAB_SIZE){fg_error_set(err,FG_ERR_ARGUMENT,
        "invalid output history token");return FG_ERR_ARGUMENT;}
    uint32_t *counts=fg_vk_tensor_map(executor->history_counts);
    if(!counts){fg_error_set(err,FG_ERR_UNAVAILABLE,
        "output history storage is unavailable");return FG_ERR_UNAVAILABLE;}
    if(counts[token]!=UINT32_MAX)counts[token]++;
    return FG_OK;
}

fg_status fg_output_topk(fg_output_executor *executor,const fg_vk_tensor *hyper,
                         uint32_t k,fg_vk_tensor **scores,fg_vk_tensor **ids,
                         uint32_t *count,fg_error *err){
    if(!executor||!hyper||!scores||!ids||!count||k<1u||k>64u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid generation top-k arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_context *vk=fg_model_vk(executor->model);
    if(fg_vk_batch_active(vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"generation top-k cannot run inside a Vulkan batch");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor *logits=NULL;
    fg_status status=fg_vk_begin(vk,err);
    uint32_t produced=FG_Q38_VOCAB_SIZE,slot=0u;
    const fg_vk_tensor *candidate_scores=NULL,*candidate_ids=NULL;
    if(status==FG_OK)status=fg_output_logits(executor,hyper,&logits,err);
    candidate_scores=logits;candidate_ids=executor->vocabulary_ids;
    do {
        if(status!=FG_OK)break;
        status=fg_vk_topk_select(vk,executor->topk_scores[slot],
            executor->topk_ids[slot],candidate_scores,candidate_ids,
            produced,k,&produced,err);
        if(status!=FG_OK)break;
        candidate_scores=executor->topk_scores[slot];
        candidate_ids=executor->topk_ids[slot];
        slot^=1u;
    } while(produced>k);
    if(status==FG_OK){
        fg_status end_status=fg_vk_end(vk,err);
        if(end_status!=FG_OK)status=end_status;
    }
    if(status!=FG_OK&&fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    if(status!=FG_OK)return status;
    const float *values=fg_vk_tensor_map((fg_vk_tensor *)candidate_scores);
    const uint32_t *indices=fg_vk_tensor_map((fg_vk_tensor *)candidate_ids);
    for(uint32_t i=0u;i<k;i++){
        if(indices[i]>=FG_Q38_VOCAB_SIZE||!isfinite(values[i])){
            fg_error_set(err,FG_ERR_MISMATCH,"invalid generation finalist at rank %u",i);
            return FG_ERR_MISMATCH;
        }
    }
    *scores=(fg_vk_tensor *)candidate_scores;
    *ids=(fg_vk_tensor *)candidate_ids;
    *count=k;
    return FG_OK;
}

static fg_status output_topk_in_batch(fg_output_executor *executor,
                                      const fg_vk_tensor *logits,uint32_t k,
                                      fg_vk_tensor **scores,fg_vk_tensor **ids,
                                      uint32_t *count,fg_error *err){
    fg_vk_context *vk=fg_model_vk(executor->model);
    uint32_t produced=FG_Q38_VOCAB_SIZE,slot=0u;
    const fg_vk_tensor *candidate_scores=logits,*candidate_ids=executor->vocabulary_ids;
    fg_status status=FG_OK;
    do{
        status=fg_vk_topk_select(vk,executor->topk_scores[slot],
            executor->topk_ids[slot],candidate_scores,candidate_ids,
            produced,k,&produced,err);
        if(status!=FG_OK)break;
        candidate_scores=executor->topk_scores[slot];
        candidate_ids=executor->topk_ids[slot];slot^=1u;
    }while(produced>k);
    if(status!=FG_OK)return status;
    *scores=(fg_vk_tensor *)candidate_scores;*ids=(fg_vk_tensor *)candidate_ids;
    *count=k;return FG_OK;
}

fg_status fg_output_sample(fg_output_executor *executor,const fg_vk_tensor *hyper,
                           const fg_sampler_config *config,
                           float uniform,uint32_t *token,float *logit,fg_error *err){
    if(!executor||!hyper||!config||!token||!logit){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid sampled output arguments");return FG_ERR_ARGUMENT;
    }
    bool penalties=fg_sampler_penalties_active(config);
    if(!penalties&&config->temperature==0.0f)
        return fg_output_greedy(executor,hyper,token,logit,err);
    fg_vk_context *vk=fg_model_vk(executor->model);
    if(fg_vk_batch_active(vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"sampled output cannot run inside a Vulkan batch");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_begin(vk,err);fg_vk_tensor *logits=NULL;
    if(status==FG_OK)status=fg_output_logits(executor,hyper,&logits,err);
    if(status==FG_OK&&penalties)status=fg_vk_apply_penalties(vk,logits,
        executor->history_counts,FG_Q38_VOCAB_SIZE,config->presence_penalty,
        config->frequency_penalty,config->repetition_penalty,err);
    fg_vk_tensor *scores=NULL,*ids=NULL;uint32_t count=0u;
    if(status==FG_OK&&config->temperature==0.0f){
        const fg_vk_tensor *current_scores=logits,*current_ids=executor->vocabulary_ids;
        uint32_t current=FG_Q38_VOCAB_SIZE,slot=0u;
        while(status==FG_OK&&current>1u){uint32_t next=0u;status=fg_vk_argmax_reduce(
            vk,executor->topk_scores[slot],executor->topk_ids[slot],current_scores,
            current_ids,current,&next,err);current_scores=executor->topk_scores[slot];
            current_ids=executor->topk_ids[slot];current=next;slot^=1u;}
        if(status==FG_OK){status=fg_vk_end(vk,err);if(status==FG_OK){
            const float *values=fg_vk_tensor_map((fg_vk_tensor *)current_scores);
            const uint32_t *indices=fg_vk_tensor_map((fg_vk_tensor *)current_ids);
            if(indices[0]>=FG_Q38_VOCAB_SIZE||!isfinite(values[0])){
                fg_error_set(err,FG_ERR_FORMAT,"invalid penalized output finalist");
                status=FG_ERR_FORMAT;
            }
            else {*token=indices[0];*logit=values[0];
                status=fg_output_history_increment(executor,*token,err);}
        }}
        if(status!=FG_OK&&fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}return status;
    }
    if(status==FG_OK)status=output_topk_in_batch(executor,logits,config->top_k,&scores,&ids,&count,err);
    float values[64];uint32_t tokens[64];
    if(status==FG_OK){status=fg_vk_end(vk,err);}
    if(status==FG_OK)status=fg_vk_tensor_read(scores,0,values,(uint64_t)count*sizeof(float),err);
    if(status==FG_OK)status=fg_vk_tensor_read(ids,0,tokens,(uint64_t)count*sizeof(uint32_t),err);
    if(status==FG_OK)status=fg_sampler_select(config,values,tokens,count,uniform,token,logit,err);
    if(status==FG_OK&&penalties)status=fg_output_history_increment(executor,*token,err);
    if(status!=FG_OK&&fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    return status;
}
