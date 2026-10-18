#include "fg_pipeline.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static fg_fabric_receiver receiver;
static void *receiver_context;
static fg_pipeline pipeline;
static uint64_t iterations, polls, checksum;
static bool budget_mode;

void fg_error_set(fg_error *error,fg_status code,const char *format,...){
    if(!error)return;
    error->code=code;
    va_list args;va_start(args,format);
    vsnprintf(error->message,sizeof(error->message),format,args);
    va_end(args);
}
void fg_fabric_set_receiver(fg_fabric *fabric,fg_fabric_receiver callback,void *context){
    (void)fabric;receiver=callback;receiver_context=context;
}
void fg_fabric_set_slot(fg_fabric *fabric,uint32_t slot){(void)fabric;(void)slot;}
fg_status fg_fabric_poll_slot(fg_fabric *fabric,uint32_t slot,uint32_t wanted,
    fg_fabric_class cls,uint32_t *peer,fg_frame_header *header,void *payload,
    uint32_t capacity,uint32_t *bytes,fg_error *error){
    (void)fabric;(void)slot;(void)cls;(void)header;(void)payload;
    (void)capacity;(void)error;
    *peer=wanted;*bytes=0u;polls++;return FG_OK;
}
static uint64_t now_ns(void){
    struct timespec now;clock_gettime(CLOCK_MONOTONIC,&now);
    return (uint64_t)now.tv_sec*1000000000u+(uint64_t)now.tv_nsec;
}
static void job(void *unused){
    (void)unused;
    fg_pipeline_set_phase(budget_mode?FG_PIPELINE_PREFILL:FG_PIPELINE_DECODE);
    if(budget_mode){
        pipeline.slots[1].active=true;
        pipeline.slots[1].phase=FG_PIPELINE_PREFILL;
        for(uint64_t i=0;i<iterations;i++){
            fg_pipeline_budget budget=fg_pipeline_prefill_budget(128u,8u);
            checksum+=budget.tokens+budget.frames;
        }
        pipeline.slots[1].active=false;
    }else{
        for(uint64_t i=0;i<iterations;i++){
            fg_error error={0};fg_frame_header header={0};uint32_t peer,bytes;
            assert(receiver(receiver_context,1u,FG_FABRIC_CONTROL,-1,&peer,
                &header,NULL,0u,&bytes,&error)==FG_OK);
            checksum+=peer+bytes;
        }
    }
}
int main(int argc,char **argv){
    assert(argc==3);
    budget_mode=!strcmp(argv[1],"budget");
    iterations=strtoull(argv[2],NULL,10);
    fg_error error={0};
    assert(fg_pipeline_init(&pipeline,NULL,&error)==FG_OK);
    assert(fg_pipeline_start(&pipeline,0u,job,NULL,&error)==FG_OK);
    uint64_t started=now_ns();
    while(pipeline.slots[0].active){
        uint32_t slot=fg_pipeline_pick(&pipeline);
        assert(slot==0u);
        assert(fg_pipeline_turn(&pipeline,slot));
    }
    uint64_t elapsed=now_ns()-started;
    assert(budget_mode?checksum>iterations:polls==iterations);
    printf("{\"mode\":\"%s\",\"iterations\":%llu,\"elapsed_ns\":%llu,"
        "\"ns_per_op\":%.6f,\"polls\":%llu,\"checksum\":%llu}\n",
        argv[1],(unsigned long long)iterations,(unsigned long long)elapsed,
        (double)elapsed/(double)iterations,(unsigned long long)polls,
        (unsigned long long)checksum);
    fg_pipeline_destroy(&pipeline);
}
