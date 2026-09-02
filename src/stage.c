#include "fg_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool stage_profile_requested(uint32_t sequence){
    const char *value=getenv("FG_PIPELINE_PROFILE_SEQUENCE");
    if(!value||!*value)return false;
    char *end=NULL;unsigned long requested=strtoul(value,&end,10);
    return end&&!*end&&requested<=UINT32_MAX&&(uint32_t)requested==sequence;
}

struct fg_stage_executor {
    fg_model *model;
    fg_owner_executor *owner;
    fg_expert_executor *expert;
    fg_output_executor *output;
    fg_vk_tensor *terminal_input;
    fg_prefill_pair *work_pairs;
    fg_prefill_result_pair *result_pairs;
    float *result_outputs;
    fg_decode_work decode_work;
    fg_prefill_work prefill_work;
    fg_stage_ngram_decode_fn ngram_decode;
    fg_stage_ngram_prefill_fn ngram_prefill;
    void *ngram_context;
    fg_error failure;
    uint32_t rank,stage,layer_begin,layer_end,max_pairs;
    bool failed;
};

static fg_status stage_fail(fg_stage_executor *executor,fg_status status,
                            const fg_error *source,fg_error *err){
    if(executor&&!executor->failed){
        executor->failed=true;
        if(source)executor->failure=*source;
        else{
            memset(&executor->failure,0,sizeof(executor->failure));
            executor->failure.code=status;
        }
        if(executor->failure.code==FG_OK)executor->failure.code=status;
    }
    if(err&&executor)*err=executor->failure;
    return status;
}

static fg_status stage_ready(fg_stage_executor *executor,fg_error *err){
    if(!executor){
        fg_error_set(err,FG_ERR_ARGUMENT,"stage executor is null");
        return FG_ERR_ARGUMENT;
    }
    if(executor->failed){
        if(err)*err=executor->failure;
        return executor->failure.code?executor->failure.code:FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_stage_validate_manifest(const fg_manifest *manifest,uint32_t rank,
                                     uint32_t *stage,uint32_t *layer_begin,
                                     uint32_t *layer_end,fg_error *err){
    if(!manifest||rank>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid stage manifest arguments");
        return FG_ERR_ARGUMENT;
    }
    if(manifest->format_version!=FG_MANIFEST_FORMAT_VERSION||
       manifest->protocol_version!=FG_PIPELINE_PROTOCOL_VERSION||
       manifest->execution_mode!=FG_EXECUTION_PIPELINE||
       !manifest->prefill_microbatch||
       manifest->prefill_microbatch>FG_PREFILL_MAX_TOKENS||
       !manifest->stage_count||manifest->stage_count>FG_RANK_COUNT||
       manifest->layer_offsets[0]!=0u||
       manifest->layer_offsets[manifest->stage_count]!=FG_LAYER_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "stage executor requires a bounded pipeline manifest");
        return FG_ERR_MISMATCH;
    }
    uint32_t found=UINT32_MAX;
    for(uint32_t i=0;i<manifest->stage_count;i++){
        if(manifest->stage_ranks[i]==rank){
            if(found!=UINT32_MAX){
                fg_error_set(err,FG_ERR_FORMAT,
                             "rank %u appears in multiple pipeline stages",rank);
                return FG_ERR_FORMAT;
            }
            found=i;
        }
    }
    if(found==UINT32_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "rank %u is not a pipeline stage",rank);
        return FG_ERR_MISMATCH;
    }
    uint32_t begin=manifest->layer_offsets[found];
    uint32_t end=manifest->layer_offsets[found+1u];
    if(begin>=end||end>FG_LAYER_COUNT){
        fg_error_set(err,FG_ERR_FORMAT,
                     "pipeline stage %u has an invalid layer range",found);
        return FG_ERR_FORMAT;
    }
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        bool in_range=layer>=begin&&layer<end;
        if((manifest->layer_owner[layer]==rank)!=in_range){
            fg_error_set(err,FG_ERR_FORMAT,
                         "rank %u does not own one contiguous layer range",rank);
            return FG_ERR_FORMAT;
        }
        if(in_range)for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
            if(manifest->expert_rank[layer][expert]!=rank){
                fg_error_set(err,FG_ERR_FORMAT,
                             "layer %u expert %u is not stage-local",
                             layer,expert);
                return FG_ERR_FORMAT;
            }
    }
    if(stage)*stage=found;
    if(layer_begin)*layer_begin=begin;
    if(layer_end)*layer_end=end;
    return FG_OK;
}

static bool stage_has_qsa(const fg_stage_executor *executor){
    for(uint32_t layer=executor->layer_begin;layer<executor->layer_end;layer++)
        if((layer&3u)==3u)return true;
    return false;
}

fg_status fg_stage_executor_create(fg_stage_executor **out,
                                   const fg_stage_config *config,fg_error *err){
    if(!out||!config||!config->model){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid stage executor arguments");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    const fg_manifest *manifest=fg_model_manifest(config->model);
    uint32_t rank=fg_model_rank(config->model),stage=0u,begin=0u,end=0u;
    fg_status status=fg_stage_validate_manifest(manifest,rank,&stage,&begin,&end,err);
    if(status!=FG_OK)return status;
    fg_stage_executor *executor=calloc(1,sizeof(*executor));
    if(!executor){
        fg_error_set(err,FG_ERR_OOM,"allocate stage executor");
        return FG_ERR_OOM;
    }
    executor->model=config->model;
    executor->rank=rank;executor->stage=stage;
    executor->layer_begin=begin;executor->layer_end=end;
    executor->ngram_decode=config->ngram_decode;
    executor->ngram_prefill=config->ngram_prefill;
    executor->ngram_context=config->ngram_context;
    executor->max_pairs=manifest->prefill_microbatch*FG_TOP_K;
    executor->work_pairs=calloc(executor->max_pairs,
                                sizeof(*executor->work_pairs));
    executor->result_pairs=calloc(executor->max_pairs,
                                  sizeof(*executor->result_pairs));
    executor->result_outputs=calloc(
        (uint64_t)executor->max_pairs*FG_HIDDEN_SIZE,
        sizeof(*executor->result_outputs));
    if(!executor->work_pairs||!executor->result_pairs||
       !executor->result_outputs){
        fg_error_set(err,FG_ERR_OOM,"allocate stage-local expert results");
        status=FG_ERR_OOM;
    }
    if(status==FG_OK)status=fg_owner_executor_create(
        &executor->owner,config->model,err);
    if(status==FG_OK)status=fg_expert_executor_create(
        &executor->expert,config->model,err);
    if(status==FG_OK&&stage_has_qsa(executor))
        status=fg_owner_qsa_open_resident(executor->owner,err);
    if(status==FG_OK&&stage+1u==manifest->stage_count){
        status=fg_output_executor_create(&executor->output,config->model,err);
        if(status==FG_OK)status=fg_vk_tensor_create(fg_model_vk(config->model),
            (uint64_t)FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float),
            &executor->terminal_input,err);
    }
    if(status!=FG_OK){
        fg_stage_executor_close(executor);
        return status;
    }
    *out=executor;
    return FG_OK;
}

void fg_stage_executor_close(fg_stage_executor *executor){
    if(!executor)return;
    fg_vk_tensor_destroy(executor->terminal_input);
    fg_output_executor_destroy(executor->output);
    fg_expert_executor_destroy(executor->expert);
    fg_owner_executor_destroy(executor->owner);
    free(executor->result_outputs);
    free(executor->result_pairs);
    free(executor->work_pairs);
    free(executor);
}

fg_status fg_stage_executor_reset(fg_stage_executor *executor,fg_error *err){
    if(!executor){
        fg_error_set(err,FG_ERR_ARGUMENT,"stage executor reset is null");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_context *vk=fg_model_vk(executor->model);
    fg_status status=FG_OK;
    if(fg_vk_batch_active(vk))status=fg_vk_abort(vk,err);
    if(status==FG_OK)status=fg_owner_reset_state(executor->owner,err);
    if(status==FG_OK){
        executor->failed=false;
        memset(&executor->failure,0,sizeof(executor->failure));
    }
    return status;
}

static fg_status local_prefill_dispatch(
    void *context,uint32_t layer,uint32_t first_token,uint16_t token_count,
    const uint16_t *expert_ids,const float *gates,
    const uint8_t *activations_q8k,
    fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,
    fg_error *err){
    fg_stage_executor *executor=context;
    uint32_t pair_count=(uint32_t)token_count*FG_TOP_K;
    if(!executor||!token_count||pair_count>executor->max_pairs||
       !expert_ids||!gates||!activations_q8k||!results||!result_count||
       layer<executor->layer_begin||layer>=executor->layer_end){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid stage-local prefill dispatch");
        return FG_ERR_ARGUMENT;
    }
    for(uint32_t token=0;token<token_count;token++)
        for(uint32_t slot=0;slot<FG_TOP_K;slot++){
            uint32_t at=token*FG_TOP_K+slot;
            executor->work_pairs[at]=(fg_prefill_pair){
                .token_slot=(uint16_t)token,
                .expert_id=expert_ids[at],
                .routing_slot=(uint8_t)slot,
                .gate=gates[at]
            };
        }
    executor->prefill_work=(fg_prefill_work){
        .layer=(uint8_t)layer,
        .source_rank=(uint8_t)executor->rank,
        .destination_rank=(uint8_t)executor->rank,
        .first_position=first_token,
        .token_count=token_count,
        .pair_count=(uint16_t)pair_count,
        .activations_q8k=(uint8_t *)activations_q8k,
        .pairs=executor->work_pairs
    };
    fg_status status=fg_expert_prefill(executor->expert,
        &executor->prefill_work,&results[0],executor->result_pairs,
        executor->max_pairs,executor->result_outputs,
        (uint64_t)executor->max_pairs*FG_HIDDEN_SIZE,err);
    if(status==FG_OK)*result_count=1u;
    return status;
}

static fg_status finish_terminal(fg_stage_executor *executor,
                                 const fg_vk_tensor *output,
                                 uint16_t token_count,bool request_output,
                                 fg_pipeline_result *terminal,fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    bool final=executor->stage+1u==manifest->stage_count;
    if(!final||!request_output)return FG_OK;
    if(!terminal){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "terminal pipeline stage requires result storage");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=FG_OK;
    const fg_vk_tensor *final_input=output;
    if(token_count!=1u){
        uint64_t offset=(uint64_t)(token_count-1u)*
            FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float);
        status=fg_vk_tensor_write(executor->terminal_input,0,
            (const uint8_t *)fg_vk_tensor_map((fg_vk_tensor *)output)+offset,
            (uint64_t)FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float),err);
        final_input=executor->terminal_input;
    }
    if(status==FG_OK)status=fg_output_greedy(executor->output,final_input,
        &terminal->final_token,&terminal->final_logit,err);
    if(status==FG_OK)terminal->has_output=true;
    return status;
}

fg_status fg_stage_decode(fg_stage_executor *executor,uint32_t token_index,
                          const uint32_t position[FG_PIPELINE_POSITION_AXES],
                          const fg_vk_tensor *input,
                          const fg_vk_tensor *ngram_embedding,
                          fg_vk_tensor **output,fg_pipeline_result *terminal,
                          fg_error *err){
    fg_status status=stage_ready(executor,err);
    if(status!=FG_OK)return status;
    fg_error local={0};
    if(!position||!input||!output||
       fg_vk_tensor_bytes(input)<(uint64_t)FG_PIPELINE_BOUNDARY_WIDTH*4u||
       ((executor->layer_begin<=1u&&executor->layer_end>1u)!=
        (ngram_embedding!=NULL))){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "invalid stage decode boundary or PLE embedding");
        return stage_fail(executor,FG_ERR_MISMATCH,&local,err);
    }
    fg_vk_context *vk=fg_model_vk(executor->model);
    status=fg_vk_begin(vk,&local);
    const fg_vk_tensor *current=input;
    for(uint32_t layer=executor->layer_begin;
        status==FG_OK&&layer<executor->layer_end;layer++){
        fg_vk_tensor *next=NULL;
        status=fg_owner_decode_layer_pipeline(executor->owner,executor->expert,
            layer,token_index,position,current,
            layer==1u?ngram_embedding:NULL,&next,&local);
        if(status==FG_OK)current=next;
    }
    if(status==FG_OK)status=fg_vk_end(vk,&local);
    else if(fg_vk_batch_active(vk)){
        fg_error abort_error={0};
        fg_status abort_status=fg_vk_abort(vk,&abort_error);
        if(abort_status!=FG_OK){
            status=abort_status;
            local=abort_error;
        }
    }
    if(status==FG_OK)status=finish_terminal(executor,current,1u,true,terminal,
                                             &local);
    if(status!=FG_OK)return stage_fail(executor,status,&local,err);
    *output=(fg_vk_tensor *)current;
    return FG_OK;
}

fg_status fg_stage_prefill(fg_stage_executor *executor,uint32_t first_token,
                           const uint32_t *positions,uint16_t token_count,
                           bool request_output,
                           const fg_vk_tensor *input,
                           const fg_vk_tensor *ngram_embeddings,
                           fg_vk_tensor **output,fg_pipeline_result *terminal,
                           fg_error *err){
    fg_status status=stage_ready(executor,err);
    if(status!=FG_OK)return status;
    fg_error local={0};
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint64_t bytes=(uint64_t)token_count*FG_PIPELINE_BOUNDARY_WIDTH*4u;
    if(!positions||!token_count||token_count>manifest->prefill_microbatch||
       !input||!output||fg_vk_tensor_bytes(input)<bytes||
       ((executor->layer_begin<=1u&&executor->layer_end>1u)!=
        (ngram_embeddings!=NULL))){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "invalid stage prefill boundary or PLE embeddings");
        return stage_fail(executor,FG_ERR_MISMATCH,&local,err);
    }
    const fg_vk_tensor *current=input;
    for(uint32_t layer=executor->layer_begin;
        status==FG_OK&&layer<executor->layer_end;layer++){
        fg_vk_tensor *next=NULL;
        if(token_count>1u)
            status=fg_owner_prefill_layer_pipeline(executor->owner,
                executor->expert,layer,first_token,positions,token_count,
                current,layer==1u?ngram_embeddings:NULL,NULL,NULL,&next,&local);
        else status=fg_owner_prefill_layer(executor->owner,layer,first_token,
            positions,token_count,current,layer==1u?ngram_embeddings:NULL,
            local_prefill_dispatch,executor,NULL,NULL,&next,&local);
        current=next;
    }
    if(status==FG_OK)status=finish_terminal(executor,current,token_count,
                                            request_output,terminal,&local);
    if(status!=FG_OK)return stage_fail(executor,status,&local,err);
    *output=(fg_vk_tensor *)current;
    return FG_OK;
}

fg_status fg_stage_pipeline_execute(void *context,uint32_t stage,
                                    uint64_t request_id,uint32_t sequence,
                                    fg_pipeline_activation *activation,
                                    float *boundary,
                                    fg_pipeline_result *terminal,
                                    fg_error *err){
    fg_stage_executor *executor=context;
    fg_status status=stage_ready(executor,err);
    if(status!=FG_OK)return status;
    fg_error local={0};
    if(stage!=executor->stage||!request_id||!activation||!boundary||
       !activation->token_count||
       activation->token_count>
           fg_model_manifest(executor->model)->prefill_microbatch||
       (activation->execution_kind==FG_PIPELINE_EXECUTION_DECODE&&
        !activation->request_output)){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "pipeline callback does not match stage executor");
        return stage_fail(executor,FG_ERR_MISMATCH,&local,err);
    }
    uint64_t bytes=(uint64_t)activation->token_count*
        FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float);
    fg_vk_tensor *input=fg_owner_prefill_input(executor->owner);
    status=fg_vk_tensor_write(input,0,boundary,bytes,&local);
    fg_vk_context *vk=fg_model_vk(executor->model);
    bool profiling=status==FG_OK&&stage_profile_requested(sequence);
    if(profiling)status=fg_vk_profile_begin(vk,&local);
    fg_vk_tensor *ngram=NULL,*result=NULL;
    bool has_ple=executor->layer_begin<=1u&&executor->layer_end>1u;
    if(status==FG_OK&&has_ple){
        if(activation->execution_kind==FG_PIPELINE_EXECUTION_DECODE&&
           executor->ngram_decode)
            status=executor->ngram_decode(executor->ngram_context,request_id,
                sequence,activation->first_token,&ngram,&local);
        else if(activation->execution_kind==FG_PIPELINE_EXECUTION_PREFILL&&
                executor->ngram_prefill)
            status=executor->ngram_prefill(executor->ngram_context,request_id,
                sequence,activation->first_token,activation->token_count,
                &ngram,&local);
        else{
            fg_error_set(&local,FG_ERR_UNAVAILABLE,
                         "stage 0 requires prepared n-gram embeddings");
            status=FG_ERR_UNAVAILABLE;
        }
    }
    if(status==FG_OK&&activation->execution_kind==
                         FG_PIPELINE_EXECUTION_DECODE){
        if(activation->token_count!=1u){
            fg_error_set(&local,FG_ERR_MISMATCH,
                         "pipeline decode activation must contain one token");
            status=FG_ERR_MISMATCH;
        }else status=fg_stage_decode(executor,activation->first_token,
            activation->positions,input,ngram,&result,terminal,&local);
    }else if(status==FG_OK&&activation->execution_kind==
                              FG_PIPELINE_EXECUTION_PREFILL)
        status=fg_stage_prefill(executor,activation->first_token,
            activation->positions,activation->token_count,
            activation->request_output,input,ngram,
            &result,terminal,&local);
    else if(status==FG_OK){
        fg_error_set(&local,FG_ERR_MISMATCH,
                     "unsupported pipeline execution kind");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_vk_tensor_read(result,0,boundary,bytes,&local);
    if(profiling){
        fg_vk_profile profile={0};fg_error profile_error={0};
        fg_status profile_status=fg_vk_profile_end(
            vk,&profile,status==FG_OK?&local:&profile_error);
        if(status==FG_OK&&profile_status!=FG_OK)status=profile_status;
        fprintf(stderr,
            "PIPELINE_PROFILE rank=%u stage=%u sequence=%u tokens=%u "
            "gpu_ms=%.3f kernel_ms=%.3f submissions=%llu dispatches=%llu\n",
            executor->rank,executor->stage,sequence,activation->token_count,
            profile.gpu_ms,profile.kernel_ms,
            (unsigned long long)profile.submissions,
            (unsigned long long)profile.dispatches);
        for(uint32_t i=0;i<profile.kernel_count;i++)
            fprintf(stderr,
                "PIPELINE_PROFILE_KERNEL rank=%u stage=%u sequence=%u "
                "scope=%s kernel=%s calls=%llu gpu_ms=%.3f\n",
                executor->rank,executor->stage,sequence,profile.kernels[i].scope,
                profile.kernels[i].name,
                (unsigned long long)profile.kernels[i].invocations,
                profile.kernels[i].gpu_ms);
    }
    if(status!=FG_OK)return stage_fail(executor,status,&local,err);
    return FG_OK;
}
