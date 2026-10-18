#include "fg_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

static _Thread_local fg_pipeline *running;

static fg_status pipeline_option(const char *text,uint32_t fallback,uint32_t min,
    uint32_t max,uint32_t *value,fg_error *err){
    *value=fallback;
    if(!text)return FG_OK;
    char *end=NULL;errno=0;
    unsigned long parsed=strtoul(text,&end,10);
    if(errno||!text[0]||*end||parsed<min||parsed>max){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid multi-session scheduler option '%s'",text);
        return FG_ERR_ARGUMENT;
    }
    *value=(uint32_t)parsed;return FG_OK;
}

fg_status fg_pipeline_options_load(fg_pipeline_options *options,fg_error *err){
    uint32_t priority=1u;
    fg_status status=pipeline_option(getenv("FG_MULTI_PREFILL_TOKENS"),128u,32u,256u,
        &options->prefill_tokens,err);
    if(status==FG_OK)status=pipeline_option(getenv("FG_MULTI_PREFILL_WINDOW"),1u,1u,8u,
        &options->prefill_frames,err);
    if(status==FG_OK)status=pipeline_option(getenv("FG_MULTI_PREFILL_ONLY_WINDOW"),4u,1u,4u,
        &options->prefill_only_frames,err);
    if(status==FG_OK)status=pipeline_option(getenv("FG_MULTI_PREFILL_SLICE_US"),15000u,
        0u,1000000u,&options->prefill_slice_us,err);
    if(status==FG_OK)status=pipeline_option(getenv("FG_MULTI_DECODE_PRIORITY"),1u,0u,1u,
        &priority,err);
    options->decode_priority=priority!=0u;return status;
}

void fg_pipeline_set_phase(fg_pipeline_phase phase){
    if(running)running->slots[running->current].phase=phase;
}

void fg_pipeline_set_streaming(bool streaming){
    if(running)running->slots[running->current].streaming=streaming;
}

void fg_pipeline_checkpoint(void){
    if(running){
        fg_pipeline *pipeline=running;
        swapcontext(&pipeline->slots[pipeline->current].context,&pipeline->caller);
    }
}

fg_pipeline_budget fg_pipeline_prefill_budget(uint32_t tokens,uint32_t frames){
    fg_pipeline_budget budget={tokens,frames};
    if(!running)return budget;
    for(uint32_t i=0;i<FG_PIPELINE_SESSIONS;i++){
        if(i==running->current||!running->slots[i].active)continue;
        if(budget.tokens>running->options.prefill_tokens)
            budget.tokens=running->options.prefill_tokens;
        uint32_t window=running->slots[running->current].phase==FG_PIPELINE_PREFILL&&
            running->slots[i].phase==FG_PIPELINE_PREFILL?
            running->options.prefill_only_frames:running->options.prefill_frames;
        if(budget.frames>window)budget.frames=window;
        if(running->slots[i].phase==FG_PIPELINE_DECODE&&running->options.prefill_slice_us){
            uint64_t limit=running->prefill_ns_per_token?
                (uint64_t)running->options.prefill_slice_us*1000u/running->prefill_ns_per_token:1u;
            if(!limit)limit=1u;
            if(limit<budget.tokens)budget.tokens=(uint32_t)limit;
        }
    }
    return budget;
}

void fg_pipeline_observe_prefill(uint32_t tokens,uint64_t service_ns){
    if(!running||!tokens||!service_ns)return;
    uint64_t cost=service_ns/tokens+(service_ns%tokens!=0u);
    /* Retain the worst observed service cost; small chunks have fixed overhead
     * that a full-microbatch per-token estimate otherwise underestimates. */
    if(cost>running->prefill_ns_per_token)running->prefill_ns_per_token=cost;
}

static uint64_t pipeline_ms(void){
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return (uint64_t)now.tv_sec*1000u+(uint64_t)now.tv_nsec/1000000u;
}

static void pipeline_entry(void){
    fg_pipeline *pipeline=running;
    fg_pipeline_slot *slot=&pipeline->slots[pipeline->current];
    slot->job(slot->argument);
    slot->active=false;
}

static fg_status pipeline_receive(void *context,uint32_t wanted_peer,
    fg_fabric_class cls,int32_t timeout,uint32_t *peer,fg_frame_header *header,
    void *payload,uint32_t capacity,uint32_t *bytes,fg_error *err){
    fg_pipeline *pipeline=context;
    if(pipeline->failed){if(err)*err=pipeline->failure;return pipeline->failure.code;}
    fg_pipeline_slot *slot=&pipeline->slots[pipeline->current];
    slot->wanted_peer=wanted_peer;slot->cls=cls;slot->peer=peer;
    slot->header=header;slot->payload=payload;slot->capacity=capacity;
    slot->bytes=bytes;slot->error=err;slot->waiting=true;
    slot->deadline=timeout<0?0u:pipeline_ms()+(uint32_t)timeout;
    swapcontext(&slot->context,&pipeline->caller);
    return slot->result;
}

bool fg_pipeline_failed(void){return running&&running->failed;}

void fg_pipeline_abort(fg_pipeline *pipeline,const fg_error *error){
    if(pipeline->failed)return;
    pipeline->failed=true;
    pipeline->failure=*error;
    if(pipeline->failure.code==FG_OK)
        fg_error_set(&pipeline->failure,FG_ERR_UNAVAILABLE,"ring recovery in progress");
    for(uint32_t i=0;i<FG_PIPELINE_SESSIONS;i++){
        fg_pipeline_slot *slot=&pipeline->slots[i];
        if(!slot->active||!slot->waiting)continue;
        if(slot->error)*slot->error=pipeline->failure;
        slot->result=pipeline->failure.code;
        slot->waiting=false;
    }
}
fg_status fg_pipeline_init(fg_pipeline *pipeline,fg_fabric *fabric,fg_error *err){
    memset(pipeline,0,sizeof(*pipeline));
    fg_status status=fg_pipeline_options_load(&pipeline->options,err);
    if(status!=FG_OK)return status;
    pipeline->fabric=fabric;
    for(uint32_t i=0;i<FG_PIPELINE_SESSIONS;i++){
        pipeline->slots[i].stack=malloc(FG_PIPELINE_STACK_BYTES);
        if(!pipeline->slots[i].stack){
            fg_pipeline_destroy(pipeline);
            fg_error_set(err,FG_ERR_OOM,"allocate pipeline stack");
            return FG_ERR_OOM;
        }
    }
    fg_fabric_set_receiver(fabric,pipeline_receive,pipeline);
    return FG_OK;
}

void fg_pipeline_destroy(fg_pipeline *pipeline){
    fg_fabric_set_receiver(pipeline->fabric,NULL,NULL);
    for(uint32_t i=0;i<FG_PIPELINE_SESSIONS;i++)free(pipeline->slots[i].stack);
    memset(pipeline,0,sizeof(*pipeline));
}

fg_status fg_pipeline_start(fg_pipeline *pipeline,uint32_t index,
    fg_pipeline_job job,void *argument,fg_error *err){
    if(index>=FG_PIPELINE_SESSIONS||pipeline->slots[index].active||!job){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline slot is busy or invalid");
        return FG_ERR_ARGUMENT;
    }
    fg_pipeline_slot *slot=&pipeline->slots[index];
    if(getcontext(&slot->context)!=0){
        fg_error_set(err,FG_ERR_IO,"create pipeline context");
        return FG_ERR_IO;
    }
    slot->context.uc_stack.ss_sp=slot->stack;
    slot->context.uc_stack.ss_size=FG_PIPELINE_STACK_BYTES;
    slot->context.uc_stack.ss_flags=0;
    slot->context.uc_link=&pipeline->caller;
    slot->job=job;slot->argument=argument;slot->active=true;slot->waiting=false;
    slot->streaming=false;slot->phase=FG_PIPELINE_PREFILL;
    makecontext(&slot->context,pipeline_entry,0);
    return FG_OK;
}

static bool pipeline_ready(fg_pipeline *pipeline,uint32_t index){
    if(index>=FG_PIPELINE_SESSIONS)return false;
    fg_pipeline_slot *slot=&pipeline->slots[index];
    if(!slot->active)return false;
    if(slot->waiting){
        uint32_t peer=0;
        fg_status status=fg_fabric_poll_slot(pipeline->fabric,index,slot->wanted_peer,
            slot->cls,&peer,slot->header,slot->payload,slot->capacity,
            slot->bytes,slot->error);
        if(status==FG_ERR_LIMIT){
            if(!slot->deadline||pipeline_ms()<slot->deadline)return false;
            fg_error_set(slot->error,FG_ERR_LIMIT,"pipeline receive timed out");
        }
        if(slot->peer)*slot->peer=peer;
        slot->result=status;slot->waiting=false;
    }
    return true;
}

uint32_t fg_pipeline_pick(fg_pipeline *pipeline){
    uint32_t levels=pipeline->options.decode_priority?3u:1u;
    for(uint32_t level=0;level<levels;level++){
        for(uint32_t offset=0;offset<FG_PIPELINE_SESSIONS;offset++){
            uint32_t i=(pipeline->next+offset)%FG_PIPELINE_SESSIONS;
            fg_pipeline_slot *slot=&pipeline->slots[i];
            uint32_t priority=slot->phase==FG_PIPELINE_DECODE?
                (slot->streaming?0u:1u):2u;
            if(levels>1u&&priority!=level)continue;
            if(pipeline_ready(pipeline,i))return i;
        }
    }
    return FG_PIPELINE_SESSIONS;
}

bool fg_pipeline_turn(fg_pipeline *pipeline,uint32_t index){
    if(!pipeline_ready(pipeline,index))return false;
    fg_pipeline_slot *slot=&pipeline->slots[index];
    running=pipeline;pipeline->current=index;
    fg_fabric_set_slot(pipeline->fabric,index);
    swapcontext(&pipeline->caller,&slot->context);
    running=NULL;
    pipeline->next=(index+1u)%FG_PIPELINE_SESSIONS;
    return true;
}
