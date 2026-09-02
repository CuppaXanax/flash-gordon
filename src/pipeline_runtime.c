#include "fg_pipeline_runtime.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

struct fg_pipeline_runtime {
    const fg_manifest *manifest;
    fg_pipeline *pipeline;
    fg_pipeline_prepare_fn prepare;
    fg_pipeline_progress_fn progress;
    void *context;
    uint32_t *positions[FG_PIPELINE_DEFAULT_SLOT_COUNT];
    float *boundary[FG_PIPELINE_DEFAULT_SLOT_COUNT];
    bool active;
    bool retired;
    fg_sampler_config sampler;
    fg_sampler_state sampler_state;
};

static double elapsed(const struct timespec *start,const struct timespec *end){
    return (double)(end->tv_sec-start->tv_sec)+
           (double)(end->tv_nsec-start->tv_nsec)*1e-9;
}

static fg_status retire(fg_pipeline_runtime *runtime,fg_status status,
                        fg_error *err){
    runtime->retired=true;
    if(err&&!err->message[0])
        fg_error_set(err,status,"pipeline runtime aborted; reopen is required");
    return status;
}

static fg_status ready(fg_pipeline_runtime *runtime,fg_error *err){
    if(!runtime){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline runtime is null");
        return FG_ERR_ARGUMENT;
    }
    if(runtime->retired){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "pipeline runtime is retired; reopen is required");
        return FG_ERR_UNAVAILABLE;
    }
    return FG_OK;
}

fg_status fg_pipeline_runtime_create(fg_pipeline_runtime **out,
                                     const fg_pipeline_runtime_config *config,
                                     fg_error *err){
    if(!out||!config||!config->manifest||!config->pipeline||
       !config->prepare||!config->progress||
       config->manifest->execution_mode!=FG_EXECUTION_PIPELINE||
       config->manifest->prefill_microbatch!=FG_PIPELINE_DEFAULT_MICROBATCH||
       config->manifest->slot_count!=FG_PIPELINE_DEFAULT_SLOT_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline runtime configuration");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    fg_pipeline_runtime *runtime=calloc(1,sizeof(*runtime));
    if(!runtime){
        fg_error_set(err,FG_ERR_OOM,"allocate pipeline runtime");
        return FG_ERR_OOM;
    }
    runtime->manifest=config->manifest;
    runtime->pipeline=config->pipeline;
    runtime->prepare=config->prepare;
    runtime->progress=config->progress;
    runtime->context=config->context;
    fg_sampler_config_greedy(&runtime->sampler);
    fg_sampler_state_init(&runtime->sampler_state,runtime->sampler.seed);
    uint64_t position_values=(uint64_t)config->manifest->prefill_microbatch*
        FG_PIPELINE_POSITION_AXES;
    uint64_t boundary_values=(uint64_t)config->manifest->prefill_microbatch*
        FG_PIPELINE_BOUNDARY_WIDTH;
    for(uint32_t slot=0;slot<FG_PIPELINE_DEFAULT_SLOT_COUNT;slot++){
        runtime->positions[slot]=malloc((size_t)position_values*
                                        sizeof(*runtime->positions[slot]));
        runtime->boundary[slot]=malloc((size_t)boundary_values*
                                       sizeof(*runtime->boundary[slot]));
        if(!runtime->positions[slot]||!runtime->boundary[slot]){
            fg_pipeline_runtime_destroy(runtime);
            fg_error_set(err,FG_ERR_OOM,
                         "allocate fixed pipeline request boundary slots");
            return FG_ERR_OOM;
        }
    }
    *out=runtime;
    return FG_OK;
}

fg_status fg_pipeline_runtime_set_sampler(fg_pipeline_runtime *runtime,
                                          const fg_sampler_config *sampler,
                                          fg_error *err){
    fg_status status=ready(runtime,err);
    if(status!=FG_OK)return status;
    status=fg_sampler_config_validate(sampler,err);
    if(status!=FG_OK)return status;
    runtime->sampler=*sampler;
    fg_sampler_state_init(&runtime->sampler_state,sampler->seed);
    return FG_OK;
}

static float sampler_draw(fg_pipeline_runtime *runtime){
    return runtime->sampler.temperature>0.0f?
        fg_sampler_uniform(&runtime->sampler_state):0.0f;
}

void fg_pipeline_runtime_destroy(fg_pipeline_runtime *runtime){
    if(!runtime)return;
    for(uint32_t slot=0;slot<FG_PIPELINE_DEFAULT_SLOT_COUNT;slot++){
        free(runtime->boundary[slot]);
        free(runtime->positions[slot]);
    }
    free(runtime);
}

fg_status fg_pipeline_runtime_begin(fg_pipeline_runtime *runtime,
                                    uint64_t request_id,uint32_t first_sequence,
                                    fg_error *err){
    fg_status status=ready(runtime,err);
    if(status!=FG_OK)return status;
    if(runtime->active){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline request is already active");
        return FG_ERR_ARGUMENT;
    }
    status=fg_pipeline_begin(runtime->pipeline,request_id,first_sequence,err);
    if(status!=FG_OK)return retire(runtime,status,err);
    runtime->active=true;
    return FG_OK;
}

static fg_status progress_until_result(fg_pipeline_runtime *runtime,
                                       fg_pipeline_result *result,fg_error *err){
    uint32_t sequence=0u;
    for(;;){
        fg_status status=fg_pipeline_take_result(runtime->pipeline,result,
                                                 &sequence,err);
        if(status==FG_OK)return FG_OK;
        if(status!=FG_ERR_UNAVAILABLE)return retire(runtime,status,err);
        memset(err,0,sizeof(*err));
        status=runtime->progress(runtime->context,runtime->pipeline,err);
        if(status!=FG_OK)return retire(runtime,status,err);
    }
}

static fg_status progress_once(fg_pipeline_runtime *runtime,fg_error *err){
    fg_status status=runtime->progress(
        runtime->context,runtime->pipeline,err);
    return status==FG_OK?FG_OK:retire(runtime,status,err);
}

static fg_status progress_until_admission(fg_pipeline_runtime *runtime,
                                          fg_error *err){
    while(!fg_pipeline_available_slots(runtime->pipeline)||
          !fg_pipeline_available_inflight(runtime->pipeline)){
        fg_status status=runtime->progress(runtime->context,runtime->pipeline,err);
        if(status!=FG_OK)return retire(runtime,status,err);
    }
    return FG_OK;
}

static bool admission_backpressured(fg_pipeline_runtime *runtime,
                                    fg_status status){
    if(status!=FG_ERR_LIMIT)return false;
    fg_error ignored={0};
    return fg_pipeline_status(runtime->pipeline,&ignored)==FG_OK&&
        (!fg_pipeline_available_slots(runtime->pipeline)||
         !fg_pipeline_available_inflight(runtime->pipeline));
}

fg_status fg_pipeline_runtime_prefill(fg_pipeline_runtime *runtime,
                                      const uint32_t *token_ids,
                                      uint32_t first_token,uint32_t token_count,
                                      fg_pipeline_result *terminal,
                                      double *seconds,fg_error *err){
    fg_status status=ready(runtime,err);
    if(status!=FG_OK)return status;
    if(!runtime->active||!token_ids||!token_count||!terminal||
       first_token>=runtime->manifest->max_context||
       token_count>runtime->manifest->max_context-first_token){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline prefill request");
        return FG_ERR_ARGUMENT;
    }
    struct timespec start={0},end={0};
    uint32_t submitted=0u,completed=0u;
    uint32_t submitted_chunks=0u,outstanding_chunks=0u;
    bool timer_started=false;
    float pending_uniform=0.0f;
    bool pending_uniform_valid=false;
    double stage_seconds[FG_PIPELINE_STAGE_COUNT]={0};
    while(completed<token_count){
        bool backpressured=false;
        while(submitted<token_count&&
              fg_pipeline_available_slots(runtime->pipeline)&&
              fg_pipeline_available_inflight(runtime->pipeline)){
            uint32_t count=token_count-submitted;
            if(count>runtime->manifest->prefill_microbatch)
                count=runtime->manifest->prefill_microbatch;
            uint32_t slot=submitted_chunks%runtime->manifest->slot_count;
            bool final=submitted+count==token_count;
            status=runtime->prepare(runtime->context,
                FG_PIPELINE_EXECUTION_PREFILL,token_ids+submitted,
                first_token+submitted,(uint16_t)count,runtime->positions[slot],
                runtime->boundary[slot],err);
            if(status!=FG_OK)return retire(runtime,status,err);
            if(!timer_started){
                clock_gettime(CLOCK_MONOTONIC,&start);
                timer_started=true;
            }
            if(final&&!pending_uniform_valid){
                pending_uniform=sampler_draw(runtime);
                pending_uniform_valid=true;
            }
            status=fg_pipeline_submit_with_sampler(runtime->pipeline,
                FG_PIPELINE_EXECUTION_PREFILL,first_token+submitted,
                (uint16_t)count,final,runtime->positions[slot],
                runtime->boundary[slot],&runtime->sampler,
                final?pending_uniform:0.0f,NULL,err);
            if(admission_backpressured(runtime,status)){
                memset(err,0,sizeof(*err));
                backpressured=true;
                break;
            }
            if(status!=FG_OK)return retire(runtime,status,err);
            submitted+=count;
            submitted_chunks++;
            outstanding_chunks++;
            if(final)pending_uniform_valid=false;
        }
        bool consumed_result=false;
        for(;;){
            fg_pipeline_result result={0};uint32_t sequence=0u;
            status=fg_pipeline_take_result(
                runtime->pipeline,&result,&sequence,err);
            if(status==FG_ERR_UNAVAILABLE){
                memset(err,0,sizeof(*err));
                break;
            }
            if(status!=FG_OK)return retire(runtime,status,err);
            consumed_result=true;
            if(!outstanding_chunks||
               result.completed_first_token!=first_token+completed||
               !result.completed_token_count||
               result.completed_token_count>token_count-completed||
               result.has_output!=(completed+result.completed_token_count==
                                   token_count)){
                fg_error_set(
                    err,FG_ERR_MISMATCH,
                    "pipeline prefill completion order or output selection changed");
                return retire(runtime,FG_ERR_MISMATCH,err);
            }
            for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
                stage_seconds[stage]+=result.stage_seconds[stage];
            completed+=result.completed_token_count;
            outstanding_chunks--;
            if(result.has_output){
                *terminal=result;
                for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
                    terminal->stage_seconds[stage]=(float)stage_seconds[stage];
            }
        }
        if(completed==token_count)break;
        if(submitted<token_count&&!backpressured&&
           fg_pipeline_available_slots(runtime->pipeline)&&
           fg_pipeline_available_inflight(runtime->pipeline))
            continue;
        if(!consumed_result||backpressured||
           !fg_pipeline_available_slots(runtime->pipeline)||
           !fg_pipeline_available_inflight(runtime->pipeline)){
            status=progress_once(runtime,err);
            if(status!=FG_OK)return status;
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(seconds)*seconds=timer_started?elapsed(&start,&end):0.0;
    return FG_OK;
}

fg_status fg_pipeline_runtime_decode(fg_pipeline_runtime *runtime,
                                     uint32_t token_id,uint32_t token_index,
                                     fg_pipeline_result *terminal,
                                     fg_error *err){
    fg_status status=ready(runtime,err);
    if(status!=FG_OK)return status;
    if(!runtime->active||!terminal||token_id>=FG_Q38_VOCAB_SIZE||
       token_index>=runtime->manifest->max_context){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline decode request");
        return FG_ERR_ARGUMENT;
    }
    status=runtime->prepare(runtime->context,FG_PIPELINE_EXECUTION_DECODE,
        &token_id,token_index,1u,runtime->positions[0],runtime->boundary[0],err);
    if(status!=FG_OK)return retire(runtime,status,err);
    float uniform=sampler_draw(runtime);
    for(;;){
        status=progress_until_admission(runtime,err);
        if(status!=FG_OK)return status;
        status=fg_pipeline_submit_with_sampler(runtime->pipeline,FG_PIPELINE_EXECUTION_DECODE,
            token_index,1u,true,runtime->positions[0],runtime->boundary[0],
            &runtime->sampler,uniform,NULL,err);
        if(!admission_backpressured(runtime,status))break;
        memset(err,0,sizeof(*err));
    }
    if(status!=FG_OK)return retire(runtime,status,err);
    status=progress_until_result(runtime,terminal,err);
    if(status!=FG_OK)return status;
    if(!terminal->has_output||terminal->completed_first_token!=token_index||
       terminal->completed_token_count!=1u||
       terminal->completed_frontier!=token_index+1u){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid terminal pipeline decode result");
        return retire(runtime,FG_ERR_MISMATCH,err);
    }
    return FG_OK;
}

fg_status fg_pipeline_runtime_finish(fg_pipeline_runtime *runtime,fg_error *err){
    fg_status status=ready(runtime,err);
    if(status!=FG_OK)return status;
    if(!runtime->active){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline request is not active");
        return FG_ERR_ARGUMENT;
    }
    status=fg_pipeline_request_drain(runtime->pipeline,err);
    if(status!=FG_OK)return retire(runtime,status,err);
    while(!fg_pipeline_is_drained(runtime->pipeline)){
        status=runtime->progress(runtime->context,runtime->pipeline,err);
        if(status!=FG_OK)return retire(runtime,status,err);
    }
    runtime->active=false;
    return FG_OK;
}

bool fg_pipeline_runtime_reopen_required(const fg_pipeline_runtime *runtime){
    return runtime&&runtime->retired;
}
