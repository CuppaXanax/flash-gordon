#include <sys/stat.h>
#include "../src/runtime.c"

#define REQUIRE(expression) do { \
    if(!(expression)){ \
        fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#expression,error.message); \
        result=1;goto cleanup; \
    } \
} while(0)

typedef struct scheduler_job {
    fg_pipeline_phase phase;
    bool streaming;
    uint32_t steps,completed;
    uint32_t service_tokens;
    uint64_t service_ns;
    fg_pipeline_budget budgets[5];
} scheduler_job;

static void scheduler_run(void *argument){
    scheduler_job *job=argument;
    fg_pipeline_set_phase(job->phase);
    fg_pipeline_set_streaming(job->streaming);
    while(job->completed<job->steps){
        if(job->service_ns)fg_pipeline_observe_prefill(
            job->service_tokens?job->service_tokens:128u,job->service_ns);
        job->budgets[job->completed++]=fg_pipeline_prefill_budget(512u,FG_PREFILL_FRAMES);
        fg_pipeline_checkpoint();
    }
}

static int test_scheduler(void){
    int result=0;fg_error error={0};
    fg_pipeline pipeline={0};bool initialized=false;
    fg_pipeline_options options;
    unsetenv("FG_MULTI_PREFILL_TOKENS");unsetenv("FG_MULTI_PREFILL_WINDOW");
    unsetenv("FG_MULTI_DECODE_PRIORITY");
    unsetenv("FG_MULTI_PREFILL_SLICE_US");
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_OK);
    REQUIRE(options.prefill_tokens==128u&&options.prefill_frames==1u&&options.decode_priority);
    REQUIRE(options.prefill_slice_us==15000u);
    setenv("FG_MULTI_PREFILL_TOKENS","256",1);
    setenv("FG_MULTI_PREFILL_WINDOW","2",1);setenv("FG_MULTI_DECODE_PRIORITY","0",1);
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_OK);
    REQUIRE(options.prefill_tokens==256u&&options.prefill_frames==2u&&!options.decode_priority);
    const char *invalid[]={"","31","257","garbage","128x","-1","999999999999999999999"};
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++){
        setenv("FG_MULTI_PREFILL_TOKENS",invalid[i],1);
        REQUIRE(fg_pipeline_options_load(&options,&error)==FG_ERR_ARGUMENT);
    }
    unsetenv("FG_MULTI_PREFILL_TOKENS");
    setenv("FG_MULTI_PREFILL_WINDOW","9",1);
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_ERR_ARGUMENT);
    setenv("FG_MULTI_PREFILL_WINDOW","0",1);
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_ERR_ARGUMENT);
    unsetenv("FG_MULTI_PREFILL_WINDOW");setenv("FG_MULTI_DECODE_PRIORITY","2",1);
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_ERR_ARGUMENT);
    unsetenv("FG_MULTI_DECODE_PRIORITY");
    setenv("FG_MULTI_PREFILL_SLICE_US","1000001",1);
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_ERR_ARGUMENT);
    setenv("FG_MULTI_PREFILL_SLICE_US","0",1);
    REQUIRE(fg_pipeline_options_load(&options,&error)==FG_OK&&options.prefill_slice_us==0u);
    unsetenv("FG_MULTI_PREFILL_SLICE_US");
    REQUIRE(fg_pipeline_init(&pipeline,NULL,&error)==FG_OK);initialized=true;
    scheduler_job prefill={.phase=FG_PIPELINE_PREFILL,.steps=5u};
    scheduler_job decode={.phase=FG_PIPELINE_DECODE,.streaming=true,.steps=2u};
    REQUIRE(fg_pipeline_start(&pipeline,0u,scheduler_run,&prefill,&error)==FG_OK);
    REQUIRE(fg_pipeline_pick(&pipeline)==0u&&fg_pipeline_turn(&pipeline,0u));
    REQUIRE(prefill.budgets[0].tokens==512u&&prefill.budgets[0].frames==FG_PREFILL_FRAMES);
    REQUIRE(fg_pipeline_start(&pipeline,1u,scheduler_run,&decode,&error)==FG_OK);
    REQUIRE(fg_pipeline_turn(&pipeline,0u));
    REQUIRE(prefill.budgets[1].tokens==128u&&prefill.budgets[1].frames==1u);
    REQUIRE(fg_pipeline_turn(&pipeline,1u));
    REQUIRE(pipeline.slots[1].streaming&&pipeline.slots[1].phase==FG_PIPELINE_DECODE);
    REQUIRE(fg_pipeline_turn(&pipeline,0u));
    REQUIRE(prefill.budgets[2].tokens==1u); /* No service sample: fail conservative. */
    prefill.service_ns=320000000u;
    REQUIRE(fg_pipeline_turn(&pipeline,0u));
    REQUIRE(prefill.budgets[3].tokens==6u&&prefill.budgets[3].frames==1u);
    pipeline.next=0u;
    REQUIRE(fg_pipeline_pick(&pipeline)==1u);
    pipeline.options.decode_priority=false;
    REQUIRE(fg_pipeline_pick(&pipeline)==0u);
    pipeline.options.decode_priority=true;
    REQUIRE(fg_pipeline_turn(&pipeline,fg_pipeline_pick(&pipeline)));
    REQUIRE(decode.completed==2u&&prefill.completed==4u);
    REQUIRE(fg_pipeline_turn(&pipeline,fg_pipeline_pick(&pipeline)));
    REQUIRE(!pipeline.slots[1].active);
    prefill.service_tokens=6u;prefill.service_ns=60000000u;
    REQUIRE(fg_pipeline_turn(&pipeline,fg_pipeline_pick(&pipeline)));
    REQUIRE(prefill.budgets[4].tokens==512u&&prefill.budgets[4].frames==FG_PREFILL_FRAMES);
    REQUIRE(pipeline.prefill_ns_per_token==10000000u);
    REQUIRE(fg_pipeline_turn(&pipeline,0u));
    REQUIRE(fg_pipeline_pick(&pipeline)==FG_PIPELINE_SESSIONS);
    scheduler_job a={.phase=FG_PIPELINE_DECODE,.streaming=true,.steps=2u},b=a;
    REQUIRE(fg_pipeline_start(&pipeline,0u,scheduler_run,&a,&error)==FG_OK);
    REQUIRE(fg_pipeline_start(&pipeline,1u,scheduler_run,&b,&error)==FG_OK);
    REQUIRE(fg_pipeline_turn(&pipeline,0u)&&fg_pipeline_turn(&pipeline,1u));
    pipeline.options.prefill_tokens=256u;pipeline.options.prefill_frames=2u;
    pipeline.options.prefill_slice_us=0u;
    REQUIRE(fg_pipeline_pick(&pipeline)==0u&&fg_pipeline_turn(&pipeline,0u));
    REQUIRE(fg_pipeline_pick(&pipeline)==1u&&fg_pipeline_turn(&pipeline,1u));
    REQUIRE(a.completed==2u&&b.completed==2u);
    REQUIRE(a.budgets[1].tokens==256u&&a.budgets[1].frames==2u);
    pipeline.next=1u;pipeline.slots[1].streaming=false;
    REQUIRE(fg_pipeline_pick(&pipeline)==0u); /* Streamed decode beats nonstream decode. */
    pipeline.slots[0].phase=FG_PIPELINE_PREFILL;
    REQUIRE(fg_pipeline_pick(&pipeline)==1u); /* Nonstream decode still beats prefill. */
    REQUIRE(fg_pipeline_turn(&pipeline,0u)&&fg_pipeline_turn(&pipeline,1u));
    fg_pipeline_budget outside=fg_pipeline_prefill_budget(64u,FG_PREFILL_FRAMES);
    REQUIRE(outside.tokens==64u&&outside.frames==FG_PREFILL_FRAMES);
    fg_pipeline_observe_prefill(1u,UINT64_MAX);
    REQUIRE(pipeline.prefill_ns_per_token==10000000u);
    ring_slot slots[FG_PREFILL_FRAMES]={0};
    uint32_t frontier=0u,first=17u;
    const uint16_t counts[]={128u,32u,1u,6u,256u,19u};
    for(uint32_t i=0,offset=0;i<6u;i++){
        slots[i]=(ring_slot){.active=true,.first_token=first+offset,.token_count=counts[i]};
        REQUIRE(ring_slot_find(slots,first+offset,counts[i])==&slots[i]);
        REQUIRE(!ring_slot_find(slots,first+offset,counts[i]+1u));
        offset+=counts[i];
    }
    REQUIRE(!ring_slot_finish(&slots[1],first,&frontier)&&frontier==0u);
    for(uint32_t i=0;i<6u;i++){
        REQUIRE(ring_slot_finish(&slots[i],first,&frontier));
        REQUIRE(!ring_slot_find(slots,slots[i].first_token,counts[i]));
        if(i==1u)REQUIRE(frontier==160u); /* Interrupted prefix, not chunks * quantum. */
    }
    REQUIRE(frontier==442u&&!ring_slot_finish(&slots[5],first,&frontier));
cleanup:
    if(initialized)fg_pipeline_destroy(&pipeline);
    unsetenv("FG_MULTI_PREFILL_TOKENS");unsetenv("FG_MULTI_PREFILL_WINDOW");
    unsetenv("FG_MULTI_DECODE_PRIORITY");
    unsetenv("FG_MULTI_PREFILL_SLICE_US");
    if(!result)puts("Adaptive prefill, streamed decode priority, fair turns and variable chunks: PASS");
    return result;
}

int main(void){
    if(test_scheduler())return 1;
    int result=0;
    fg_error error={0};
    fg_manifest *manifest=malloc(sizeof(*manifest));
    qsa_owner_runtime sessions[2]={0};
    bool mutex_ready[2]={false,false};
    fg_qsa_state *owner_handle=NULL;
    char directory[128];
    snprintf(directory,sizeof(directory),".build/multisess-state-%u",(unsigned)getpid());
    if(mkdir(".build",0700)!=0&&errno!=EEXIST)return 1;
    if(mkdir(directory,0700)!=0){free(manifest);return 1;}
    REQUIRE(manifest!=NULL);
    fg_manifest_init(manifest);
    manifest->protocol_version=FG_PROTOCOL_VERSION;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        manifest->layer_owner[layer]=2u;
    manifest->layer_owner[3u]=1u;
    fg_session_identity identity={0};
    fg_owner_session_control controls[2];
    memset(controls,0,sizeof(controls));
    for(uint32_t slot=0;slot<2u;slot++){
        sessions[slot].enabled=true;
        sessions[slot].layer_count=1u;sessions[slot].layers[0]=3u;
        fg_qsa_owner_guard_init(&sessions[slot].guard,1u);
        REQUIRE(pthread_mutex_init(&sessions[slot].writer_mutex,NULL)==0);
        mutex_ready[slot]=true;
        controls[slot]=(fg_owner_session_control){
            .version=FG_OWNER_SESSION_CONTROL_VERSION,.operation=FG_OWNER_SESSION_BEGIN,
            .rank=1u,.state_slot=(uint8_t)slot,.session_nonce=10u+slot,
            .logical_context_tokens=32768u,.gpu_index_tokens=32768u,
            .qsa_page_cache_bytes=UINT64_C(16)<<20u};
        memcpy(controls[slot].state_format_sha256,
            manifest->session.rank_state_format_sha256[1u],32u);
        fg_status status=qsa_owner_open_session(&sessions[slot],manifest,directory,
            &identity,controls[slot].session_nonce,&controls[slot],false,&error);
        if(status==FG_ERR_UNAVAILABLE){result=77;goto cleanup;}
        REQUIRE(status==FG_OK);
    }
    REQUIRE(strstr(sessions[1].state_path,"-s1.state")!=NULL);
    uint8_t other[FG_QSA_PAGE_RECORD_BYTES],changed[FG_QSA_PAGE_RECORD_BYTES];
    uint8_t actual[FG_QSA_PAGE_RECORD_BYTES];
    memset(other,0x5a,sizeof(other));memset(changed,0xa5,sizeof(changed));
    REQUIRE(fg_qsa_state_write_block(sessions[1].state,0u,0u,other,4u,&error)==FG_OK);
    REQUIRE(fg_qsa_state_open(&owner_handle,sessions[0].state_path,sessions[0].layers,
        1u,32768u,false,&error)==FG_OK);
    struct stat before,after;
    REQUIRE(stat(sessions[0].state_path,&before)==0);
    controls[0].session_nonce++;
    REQUIRE(qsa_owner_open_session(&sessions[0],manifest,directory,&identity,
        controls[0].session_nonce,&controls[0],true,&error)==FG_OK);
    REQUIRE(stat(sessions[0].state_path,&after)==0&&before.st_ino==after.st_ino);
    REQUIRE(fg_qsa_state_write_block(owner_handle,0u,0u,changed,4u,&error)==FG_OK);
    uint32_t committed=0;
    REQUIRE(fg_qsa_state_read_block(sessions[0].state,0u,0u,actual,&committed,&error)==FG_OK);
    REQUIRE(committed==4u&&!memcmp(actual,changed,sizeof(actual)));
    REQUIRE(fg_qsa_state_read_block(sessions[1].state,0u,0u,actual,&committed,&error)==FG_OK);
    REQUIRE(committed==4u&&!memcmp(actual,other,sizeof(actual)));
    REQUIRE(sessions[1].guard.session_nonce==11u);
    REQUIRE(qsa_owner_open_session(&sessions[0],manifest,directory,&identity,
        controls[0].session_nonce,&controls[0],true,&error)==FG_ERR_MISMATCH);
cleanup:
    fg_qsa_state_close(owner_handle);
    for(uint32_t slot=0;slot<2u;slot++){
        if(mutex_ready[slot])pthread_mutex_destroy(&sessions[slot].writer_mutex);
        qsa_owner_runtime_destroy(&sessions[slot]);
    }
    rmdir(directory);free(manifest);
    if(!result)puts("Independent QSA namespaces, retained inode and reset isolation: PASS");
    return result;
}
