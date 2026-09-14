#include "fg_output.h"
#include "fg_q38_schema.h"
#include "fg_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FG_OUTPUT_TOPK_CAPACITY (((FG_Q38_VOCAB_SIZE+4095u)/4096u)*512u)

/* The per-token greedy trace was an unbuffered stderr write on rank 4's output
 * critical path.  Keep it available for A/B logs behind an env switch. */
static bool output_trace_enabled(void){
    const char *enabled=getenv("FG_OUTPUT_TRACE");
    return enabled&&*enabled&&strcmp(enabled,"0")!=0;
}

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

bool fg_output_split_requested(void){
    const char *enabled=getenv("FG_OUTPUT_SPLIT");
    return enabled&&*enabled&&strcmp(enabled,"0")!=0;
}

static fg_status output_hc_chain(fg_vk_context *vk,fg_model *model,fg_vk_tensor *normalized,
    fg_vk_tensor *down,fg_vk_tensor *activated,fg_vk_tensor *up,fg_vk_tensor *hidden,
    const fg_vk_tensor *hyper,fg_error *err);

struct fg_output_slice {
    fg_model *model;
    fg_vk_tensor *weight;
    fg_vk_tensor *normalized;
    fg_vk_tensor *down;
    fg_vk_tensor *activated;
    fg_vk_tensor *up;
    fg_vk_tensor *hidden;
    fg_vk_tensor *hyper;
    fg_vk_tensor *logits;
    fg_vk_tensor *ids;
    fg_vk_tensor *topk_scores[2];
    fg_vk_tensor *topk_ids[2];
    uint32_t first_row;
    uint32_t rows;
    uint32_t groups;
};

static fg_status output_slice_require(fg_model *model,fg_error *err){
    static const char *required[]={"output_hc_norm.weight","output_hc_down.weight","output_hc_up.weight","output.weight"};
    for(uint32_t i=0;i<sizeof(required)/sizeof(required[0]);i++)
        if(!fg_model_tensor(model,required[i])){
            fg_error_set(err,FG_ERR_MISMATCH,"rank %u is missing %s",
                         fg_model_rank(model),required[i]);
            return FG_ERR_MISMATCH;
        }
    return FG_OK;
}

fg_status fg_output_slice_create(fg_output_slice **out,fg_model *model,
                                 uint32_t first_row,uint32_t rows,fg_error *err){
    if(!out||!model||!rows||rows>FG_Q38_VOCAB_SIZE||first_row>=FG_Q38_VOCAB_SIZE||
       first_row+rows>FG_Q38_VOCAB_SIZE){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid output slice arguments");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    fg_status status=output_slice_require(model,err);
    if(status!=FG_OK)return status;
    fg_vk_tensor *base=fg_model_tensor(model,"output.weight");
    uint64_t offset=0,bytes=0;
    if(fg_vk_tensor_get_format(base)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED){
        if(first_row%FG_Q8_0_COOK_ROWS){
            fg_error_set(err,FG_ERR_MISMATCH,"output slice row %u is not tile aligned",first_row);
            return FG_ERR_MISMATCH;
        }
        uint64_t tile=fg_q8_0_cooked_tile_bytes(FG_HIDDEN_SIZE);
        uint64_t tiles=((uint64_t)first_row+rows+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS-
                       first_row/FG_Q8_0_COOK_ROWS;
        offset=((uint64_t)first_row/FG_Q8_0_COOK_ROWS)*tile;
        bytes=tiles*tile;
    }else{
        uint64_t row_bytes=(uint64_t)(FG_HIDDEN_SIZE/32u)*FG_Q8_0_BLOCK_BYTES;
        offset=(uint64_t)first_row*row_bytes;
        bytes=(uint64_t)rows*row_bytes;
    }
    if(!bytes){fg_error_set(err,FG_ERR_FORMAT,"output slice weight span is empty");return FG_ERR_FORMAT;}
    fg_output_slice *slice=calloc(1,sizeof(*slice));
    if(!slice){fg_error_set(err,FG_ERR_OOM,"allocate output slice");return FG_ERR_OOM;}
    slice->model=model;slice->first_row=first_row;slice->rows=rows;
    slice->groups=(rows+4095u)/4096u;if(!slice->groups)slice->groups=1u;
    fg_vk_context *vk=fg_model_vk(model);
    status=fg_vk_tensor_view(base,offset,bytes,&slice->weight,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_WIDTH,&slice->normalized,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_RANK,&slice->down,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_RANK,&slice->activated,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_WIDTH,&slice->up,err);
    if(status==FG_OK)status=scratch(vk,FG_HIDDEN_SIZE,&slice->hidden,err);
    if(status==FG_OK)status=scratch(vk,FG_Q38_HYPER_WIDTH,&slice->hyper,err);
    if(status==FG_OK)status=scratch(vk,rows,&slice->logits,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)rows*4u,&slice->ids,err);
    for(uint32_t i=0;status==FG_OK&&i<2u;i++){
        status=scratch(vk,slice->groups,&slice->topk_scores[i],err);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)slice->groups*4u,
                                                    &slice->topk_ids[i],err);
    }
    if(status==FG_OK){
        uint32_t *ids=fg_vk_tensor_map(slice->ids);
        for(uint32_t i=0;i<rows;i++)ids[i]=first_row+i;
    }
    if(status!=FG_OK){fg_output_slice_destroy(slice);return status;}
    *out=slice;return FG_OK;
}

void fg_output_slice_destroy(fg_output_slice *slice){
    if(!slice)return;
    for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(slice->topk_ids[i]);fg_vk_tensor_destroy(slice->topk_scores[i]);}
    fg_vk_tensor_destroy(slice->ids);fg_vk_tensor_destroy(slice->logits);
    fg_vk_tensor_destroy(slice->hyper);fg_vk_tensor_destroy(slice->hidden);
    fg_vk_tensor_destroy(slice->up);fg_vk_tensor_destroy(slice->activated);
    fg_vk_tensor_destroy(slice->down);fg_vk_tensor_destroy(slice->normalized);
    fg_vk_tensor_destroy(slice->weight);free(slice);
}

fg_status fg_output_slice_run(fg_output_slice *slice,const void *hyper,
                              float *value,uint32_t *id,fg_error *err){
    if(!slice||!hyper||!value||!id){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid output slice run arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_context *vk=fg_model_vk(slice->model);
    if(fg_vk_batch_active(vk)){
        fg_error_set(err,FG_ERR_ARGUMENT,"output slice cannot run inside a Vulkan batch");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_tensor_write(slice->hyper,0,hyper,
        (uint64_t)FG_Q38_HYPER_WIDTH*4u,err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"output_slice",err);
    if(status==FG_OK)status=output_hc_chain(vk,slice->model,slice->normalized,slice->down,
        slice->activated,slice->up,slice->hidden,slice->hyper,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,slice->logits,slice->weight,slice->hidden,
        FG_HIDDEN_SIZE,slice->rows,1u,1.0f,err);
    uint32_t count=slice->rows,slot=0u;
    const fg_vk_tensor *scores=slice->logits,*ids=slice->ids;
    while(status==FG_OK&&count>1u){
        uint32_t next=0u;
        status=fg_vk_argmax_reduce(vk,slice->topk_scores[slot],slice->topk_ids[slot],
            scores,ids,count,&next,err);
        scores=slice->topk_scores[slot];ids=slice->topk_ids[slot];count=next;slot^=1u;
    }
    if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}
    if(status!=FG_OK&&fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    if(status!=FG_OK)return status;
    const float *values=fg_vk_tensor_map((fg_vk_tensor *)scores);
    const uint32_t *indices=fg_vk_tensor_map((fg_vk_tensor *)ids);
    uint32_t best=indices[0];
    if(count!=1u||best<slice->first_row||best>=slice->first_row+slice->rows||!isfinite(values[0])){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid output slice finalist at token %u",best);
        return FG_ERR_MISMATCH;
    }
    *value=values[0];*id=best;return FG_OK;
}

static fg_status output_hc_chain(fg_vk_context *vk,fg_model *model,fg_vk_tensor *normalized,
    fg_vk_tensor *down,fg_vk_tensor *activated,fg_vk_tensor *up,fg_vk_tensor *hidden,
    const fg_vk_tensor *hyper,fg_error *err){
    fg_status status=fg_vk_group_rms_norm(vk,normalized,hyper,
        fg_model_tensor(model,"output_hc_norm.weight"),FG_HIDDEN_SIZE,FG_Q38_HYPER_COUNT,1u,1e-6f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,down,
        fg_model_tensor(model,"output_hc_down.weight"),normalized,FG_Q38_HYPER_WIDTH,FG_Q38_HYPER_RANK,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_silu_scaled(vk,activated,down,FG_Q38_HYPER_RANK,1.0f/(float)FG_Q38_HYPER_COUNT,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,up,
        fg_model_tensor(model,"output_hc_up.weight"),activated,FG_Q38_HYPER_RANK,FG_Q38_HYPER_WIDTH,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_hc_finalize(vk,hidden,normalized,up,FG_HIDDEN_SIZE,FG_Q38_HYPER_COUNT,1u,err);
    return status;
}

bool fg_output_better(float left,uint32_t left_id,float right,uint32_t right_id){
    bool left_padding=left_id==0xffffffffu,right_padding=right_id==0xffffffffu;
    if(left_padding||right_padding)return !left_padding&&right_padding;
    bool left_nonfinite=isnan(left)||isinf(left);
    bool right_nonfinite=isnan(right)||isinf(right);
    if(left_nonfinite||right_nonfinite)
        return left_nonfinite!=right_nonfinite?left_nonfinite:left_id<right_id;
    return left>right||(left==right&&left_id<right_id);
}

void fg_output_combine(float left,uint32_t left_id,float right,uint32_t right_id,
                       float *value,uint32_t *id){
    if(fg_output_better(left,left_id,right,right_id)){if(value)*value=left;if(id)*id=left_id;}
    else{if(value)*value=right;if(id)*id=right_id;}
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
    if(status==FG_OK)status=output_hc_chain(vk,executor->model,executor->normalized,
        executor->down,executor->activated,executor->up,executor->hidden,hyper,err);
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
    if(output_trace_enabled()){
        const float *hyper_raw=fg_vk_tensor_map((fg_vk_tensor *)hyper);const float *hidden_raw=fg_vk_tensor_map(executor->hidden);
        fprintf(stderr,"greedy: best %u=%.4f hyper[0:4]=%.4f,%.4f,%.4f,%.4f hidden[0:4]=%.4f,%.4f,%.4f,%.4f\n",best,best_value,hyper_raw[0],hyper_raw[1],hyper_raw[2],hyper_raw[3],hidden_raw[0],hidden_raw[1],hidden_raw[2],hidden_raw[3]);
    }
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
