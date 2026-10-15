#include "fg_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static _Thread_local fg_pipeline *running;

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
    fg_pipeline_slot *slot=&pipeline->slots[pipeline->current];
    slot->wanted_peer=wanted_peer;slot->cls=cls;slot->peer=peer;
    slot->header=header;slot->payload=payload;slot->capacity=capacity;
    slot->bytes=bytes;slot->error=err;slot->waiting=true;
    slot->deadline=timeout<0?0u:pipeline_ms()+(uint32_t)timeout;
    swapcontext(&slot->context,&pipeline->caller);
    return slot->result;
}

fg_status fg_pipeline_init(fg_pipeline *pipeline,fg_fabric *fabric,fg_error *err){
    memset(pipeline,0,sizeof(*pipeline));
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
    makecontext(&slot->context,pipeline_entry,0);
    return FG_OK;
}

bool fg_pipeline_turn(fg_pipeline *pipeline,uint32_t index){
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
    running=pipeline;pipeline->current=index;
    fg_fabric_set_slot(pipeline->fabric,index);
    swapcontext(&pipeline->caller,&slot->context);
    return true;
}
