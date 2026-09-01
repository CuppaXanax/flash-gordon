#include "fg_pipeline.h"
#include "fg_pipeline_runtime.h"
#include "fg_topology.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_MESSAGE_CAPACITY 256u
#define TEST_EXECUTION_CAPACITY 32u

static int failures;
#define CHECK(expression) do { \
    if(!(expression)){ \
        fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#expression); \
        failures++; \
    } \
} while(0)

typedef struct test_message {
    uint32_t source;
    uint32_t destination;
    fg_frame_header header;
    uint8_t *payload;
    uint32_t bytes;
} test_message;

typedef struct test_network {
    test_message messages[TEST_MESSAGE_CAPACITY];
    uint32_t count;
    uint32_t activation_before_credit_receives;
    bool fail_send[FG_RANK_COUNT];
    bool fail_receive[FG_RANK_COUNT];
    bool lower_source_priority;
} test_network;

typedef struct test_endpoint {
    test_network *network;
    uint32_t rank;
} test_endpoint;

typedef struct execution_record {
    uint32_t sequence;
    uint8_t slot;
    fg_pipeline_execution_kind kind;
    bool request_output;
} execution_record;

typedef struct execution_state {
    execution_record records[FG_PIPELINE_STAGE_COUNT][TEST_EXECUTION_CAPACITY];
    uint32_t count[FG_PIPELINE_STAGE_COUNT];
    uint32_t fail_stage;
    uint32_t fail_sequence;
    uint32_t corrupt_stage;
    uint32_t corrupt_sequence;
    uint32_t omit_result_stage;
    uint32_t omit_result_sequence;
} execution_state;

static void network_clear(test_network *network){
    for(uint32_t i=0;i<network->count;i++)free(network->messages[i].payload);
    memset(network,0,sizeof(*network));
}

static fg_status test_send(void *opaque,uint32_t peer,fg_message_type type,
                           uint64_t request_id,uint32_t sequence,const void *payload,
                           uint32_t bytes,fg_error *err){
    test_endpoint *endpoint=opaque;
    test_network *network=endpoint->network;
    if(network->fail_send[endpoint->rank]){
        fg_error_set(err,FG_ERR_IO,"synthetic peer send failure");
        return FG_ERR_IO;
    }
    if(peer>=FG_RANK_COUNT||network->count>=TEST_MESSAGE_CAPACITY){
        fg_error_set(err,FG_ERR_LIMIT,"synthetic network queue is full");
        return FG_ERR_LIMIT;
    }
    uint8_t *copy=bytes?malloc(bytes):NULL;
    if(bytes&&!copy){
        fg_error_set(err,FG_ERR_OOM,"allocate synthetic pipeline frame");
        return FG_ERR_OOM;
    }
    if(bytes)memcpy(copy,payload,bytes);
    test_message *message=&network->messages[network->count];
    fg_status status=fg_frame_encode_version(&message->header,
        FG_PIPELINE_PROTOCOL_VERSION,type,request_id,sequence,0u,copy,bytes,err);
    if(status!=FG_OK){free(copy);return status;}
    message->source=endpoint->rank;
    message->destination=peer;
    message->payload=copy;
    message->bytes=bytes;
    network->count++;
    return FG_OK;
}

static fg_status test_receive(void *opaque,uint32_t *peer,fg_frame_header *header,
                              void *payload,uint32_t capacity,uint32_t *bytes,
                              fg_error *err){
    test_endpoint *endpoint=opaque;
    test_network *network=endpoint->network;
    if(network->fail_receive[endpoint->rank]){
        fg_error_set(err,FG_ERR_IO,"synthetic peer receive failure");
        return FG_ERR_IO;
    }
    uint32_t index=network->count;
    for(uint32_t i=0;i<network->count;i++){
        if(network->messages[i].destination!=endpoint->rank)continue;
        if(index==network->count||
           (network->lower_source_priority&&
            network->messages[i].source<network->messages[index].source))
            index=i;
        if(!network->lower_source_priority)break;
    }
    if(index==network->count){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"synthetic pipeline queue is empty");
        return FG_ERR_UNAVAILABLE;
    }
    test_message message=network->messages[index];
    if(network->lower_source_priority&&
       fg_frame_type(&message.header)==FG_MSG_PIPELINE_ACTIVATION){
        for(uint32_t i=0;i<network->count;i++)
            if(network->messages[i].destination==endpoint->rank&&
               fg_frame_type(&network->messages[i].header)==
                   FG_MSG_PIPELINE_CREDIT){
                network->activation_before_credit_receives++;
                break;
            }
    }
    if(message.bytes>capacity){
        fg_error_set(err,FG_ERR_LIMIT,"synthetic receive buffer is too small");
        return FG_ERR_LIMIT;
    }
    *peer=message.source;
    *header=message.header;
    *bytes=message.bytes;
    if(message.bytes)memcpy(payload,message.payload,message.bytes);
    free(message.payload);
    for(uint32_t i=index+1u;i<network->count;i++)
        network->messages[i-1u]=network->messages[i];
    network->count--;
    return FG_OK;
}

static fg_status test_receive_peer(void *opaque,uint32_t peer,
                                   fg_frame_header *header,void *payload,
                                   uint32_t capacity,uint32_t *bytes,
                                   fg_error *err){
    test_endpoint *endpoint=opaque;
    test_network *network=endpoint->network;
    if(network->fail_receive[endpoint->rank]){
        fg_error_set(err,FG_ERR_IO,"synthetic peer receive failure");
        return FG_ERR_IO;
    }
    uint32_t index=network->count;
    for(uint32_t i=0;i<network->count;i++)
        if(network->messages[i].destination==endpoint->rank&&
           network->messages[i].source==peer){index=i;break;}
    if(index==network->count){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "synthetic pipeline peer queue is empty");
        return FG_ERR_UNAVAILABLE;
    }
    test_message message=network->messages[index];
    if(message.bytes>capacity){
        fg_error_set(err,FG_ERR_LIMIT,"synthetic receive buffer is too small");
        return FG_ERR_LIMIT;
    }
    *header=message.header;
    *bytes=message.bytes;
    if(message.bytes)memcpy(payload,message.payload,message.bytes);
    free(message.payload);
    for(uint32_t i=index+1u;i<network->count;i++)
        network->messages[i-1u]=network->messages[i];
    network->count--;
    return FG_OK;
}

static bool network_has_message(const test_network *network,uint32_t rank){
    for(uint32_t i=0;i<network->count;i++)
        if(network->messages[i].destination==rank)return true;
    return false;
}

static uint32_t network_message_count_type(const test_network *network,
                                           fg_message_type type){
    uint32_t count=0u;
    for(uint32_t i=0;i<network->count;i++)
        if(fg_frame_type(&network->messages[i].header)==type)count++;
    return count;
}

static bool network_has_message_type(const test_network *network,
                                     uint32_t destination,
                                     fg_message_type type){
    for(uint32_t i=0;i<network->count;i++)
        if(network->messages[i].destination==destination&&
           fg_frame_type(&network->messages[i].header)==type)
            return true;
    return false;
}

static bool network_promote_message(test_network *network,uint32_t destination,
                                    fg_message_type type){
    uint32_t first=network->count,found=network->count;
    for(uint32_t i=0;i<network->count;i++){
        if(network->messages[i].destination!=destination)continue;
        if(first==network->count)first=i;
        if(fg_frame_type(&network->messages[i].header)==type){
            found=i;
            break;
        }
    }
    if(found==network->count)return false;
    test_message message=network->messages[found];
    for(uint32_t i=found;i>first;i--)
        network->messages[i]=network->messages[i-1u];
    network->messages[first]=message;
    return true;
}

static fg_message_type network_first_message_type(
    const test_network *network,uint32_t destination){
    for(uint32_t i=0;i<network->count;i++)
        if(network->messages[i].destination==destination)
            return fg_frame_type(&network->messages[i].header);
    return 0;
}

static fg_pipeline_transport test_transport(test_endpoint *endpoint){
    return (fg_pipeline_transport){.context=endpoint,.send=test_send,
        .receive=test_receive,.receive_peer=test_receive_peer};
}

static fg_status execute_stage(void *opaque,uint32_t stage,uint64_t request_id,
                               uint32_t sequence,fg_pipeline_activation *activation,
                               float *boundary,fg_pipeline_result *terminal_result,
                               fg_error *err){
    execution_state *state=opaque;
    if(!request_id||stage>=FG_PIPELINE_STAGE_COUNT||
       state->count[stage]>=TEST_EXECUTION_CAPACITY){
        fg_error_set(err,FG_ERR_LIMIT,"invalid synthetic execution record");
        return FG_ERR_LIMIT;
    }
    uint32_t at=state->count[stage]++;
    state->records[stage][at]=(execution_record){.sequence=sequence,
        .slot=activation->slot,.kind=activation->execution_kind,
        .request_output=activation->request_output};
    if(stage==state->fail_stage&&sequence==state->fail_sequence){
        fg_error_set(err,FG_ERR_IO,"synthetic stage callback failure");
        return FG_ERR_IO;
    }
    uint64_t values=(uint64_t)activation->token_count*FG_PIPELINE_BOUNDARY_WIDTH;
    for(uint64_t i=0;i<values;i++)boundary[i]+=(float)(stage+1u);
    if(stage==state->corrupt_stage&&sequence==state->corrupt_sequence)
        boundary[0]=NAN;
    if(terminal_result&&activation->request_output&&
       (stage!=state->omit_result_stage||sequence!=state->omit_result_sequence)){
        terminal_result->final_token=1000u+sequence;
        terminal_result->final_logit=boundary[0];
    }
    return FG_OK;
}

static bool build_pipeline_manifest(fg_manifest *manifest){
    fg_error error={0};
    fg_manifest_init(manifest);
    manifest->protocol_version=FG_PIPELINE_PROTOCOL_VERSION;
    manifest->prefill_microbatch=FG_PIPELINE_DEFAULT_MICROBATCH;
    manifest->max_context=manifest->native_context;
    fg_topology_build_pipeline(manifest);
    return fg_topology_validate(manifest,&error)==FG_OK;
}

static bool create_chain(fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT],
                         test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT],
                         test_network *network,fg_manifest *manifest,
                         execution_state *execution,uint64_t request_id,
                         uint32_t first_sequence){
    fg_error error={0};
    memset(pipelines,0,sizeof(*pipelines)*FG_PIPELINE_STAGE_COUNT);
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        endpoints[stage]=(test_endpoint){.network=network,
            .rank=manifest->stage_ranks[stage]};
        fg_pipeline_config config={.manifest=manifest,.rank=endpoints[stage].rank,
            .transport=test_transport(&endpoints[stage]),.execute=execute_stage,
            .execute_context=execution};
        if(fg_pipeline_create(&pipelines[stage],&config,&error)!=FG_OK)return false;
        if(fg_pipeline_stage(pipelines[stage])!=stage)return false;
        if(fg_pipeline_begin(
              pipelines[stage],request_id,first_sequence,&error)!=FG_OK)return false;
    }
    return true;
}

static void destroy_chain(fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]){
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
        fg_pipeline_destroy(pipelines[stage]);
}

static bool pump_network(test_network *network,
                         fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]){
    fg_error error={0};
    for(uint32_t round=0;round<TEST_MESSAGE_CAPACITY;round++){
        bool progress=false;
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
            uint32_t rank=fg_pipeline_stage(pipelines[stage]);
            if(network_has_message(network,rank)){
                if(fg_pipeline_step(pipelines[stage],&error)!=FG_OK)return false;
                progress=true;
            }

        }
        if(!progress)return true;
    }
    return false;
}

typedef struct runtime_mock {
    test_network *network;
    fg_pipeline **pipelines;
    uint32_t prefill_prepares;
    uint32_t decode_prepares;
    uint32_t progress_calls;
    uint32_t activations_before_first_progress;
    uint32_t prepares_before_first_progress;
    uint32_t stage0_credit_steps;
    uint32_t max_inflight_used;
    execution_state *execution;
    bool delay_stage0_credits;
    bool result_before_credit;
    bool third_prepare_after_credit;
    bool overlap_mode;
    bool production_order_mode;
    bool capacity_backpressured;
    bool capacity_limit_healthy;
    bool eight_chunks_crossed_all_stages;
    bool first_result_processed;
    bool seventeenth_prepare_after_result;
} runtime_mock;

static fg_status runtime_prepare(
    void *opaque,fg_pipeline_execution_kind kind,const uint32_t *token_ids,
    uint32_t first_token,uint16_t token_count,uint32_t *positions,float *boundary,
    fg_error *error){
    runtime_mock *mock=opaque;
    if(!token_ids||!token_count){
        fg_error_set(error,FG_ERR_ARGUMENT,"invalid runtime prepare input");
        return FG_ERR_ARGUMENT;
    }
    if(kind==FG_PIPELINE_EXECUTION_PREFILL){
        mock->prefill_prepares++;
        if(mock->delay_stage0_credits&&mock->prefill_prepares==3u)
            mock->third_prepare_after_credit=mock->stage0_credit_steps>0u&&
                fg_pipeline_available_slots(mock->pipelines[0])>0u;
        if((mock->overlap_mode||mock->production_order_mode)&&
           mock->prefill_prepares==17u)
            mock->seventeenth_prepare_after_result=
                mock->first_result_processed;
    }
    else if(kind==FG_PIPELINE_EXECUTION_DECODE)mock->decode_prepares++;
    for(uint32_t token=0;token<token_count;token++){
        for(uint32_t axis=0;axis<FG_PIPELINE_POSITION_AXES;axis++)
            positions[(uint64_t)token*FG_PIPELINE_POSITION_AXES+axis]=
                first_token+token;
        for(uint32_t value=0;value<FG_PIPELINE_BOUNDARY_WIDTH;value++)
            boundary[(uint64_t)token*FG_PIPELINE_BOUNDARY_WIDTH+value]=
                (float)token_ids[token];
    }
    return FG_OK;
}

static fg_status runtime_progress(void *opaque,fg_pipeline *pipeline,
                                  fg_error *error){
    (void)pipeline;
    runtime_mock *mock=opaque;
    if(mock->progress_calls++==0u){
        mock->activations_before_first_progress=
            network_message_count_type(mock->network,
                                       FG_MSG_PIPELINE_ACTIVATION);
        mock->prepares_before_first_progress=mock->prefill_prepares;
    }
    if(mock->overlap_mode){
        uint32_t capacity=fg_pipeline_max_inflight(mock->pipelines[0]);
        uint32_t available=fg_pipeline_available_inflight(mock->pipelines[0]);
        uint32_t used=capacity-available;
        if(used>mock->max_inflight_used)mock->max_inflight_used=used;
        if(!available&&mock->prefill_prepares==capacity)
            mock->capacity_backpressured=true;
        bool crossed=true;
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
            crossed=crossed&&mock->execution->count[stage]>=8u;
        if(crossed)mock->eight_chunks_crossed_all_stages=true;
        uint32_t rank0=fg_pipeline_stage(mock->pipelines[0]);
        if(mock->capacity_backpressured&&
           fg_pipeline_available_slots(mock->pipelines[0])&&
           !mock->capacity_limit_healthy){
            fg_error probe_error={0};
            fg_status probe=fg_pipeline_submit(mock->pipelines[0],
                FG_PIPELINE_EXECUTION_PREFILL,
                100u+capacity*FG_PIPELINE_DEFAULT_MICROBATCH,
                FG_PIPELINE_DEFAULT_MICROBATCH,false,NULL,NULL,NULL,
                &probe_error);
            fg_error status_error={0};
            mock->capacity_limit_healthy=probe==FG_ERR_LIMIT&&
                fg_pipeline_status(mock->pipelines[0],&status_error)==FG_OK;
        }
        bool release_results=
            (mock->capacity_limit_healthy&&crossed)||
            mock->prefill_prepares>capacity;
        if(release_results&&network_promote_message(
               mock->network,rank0,FG_MSG_PIPELINE_RESULT)){
            mock->first_result_processed=true;
            return fg_pipeline_step(mock->pipelines[0],error);
        }
        for(uint32_t offset=0;offset<FG_PIPELINE_STAGE_COUNT-1u;offset++){
            uint32_t stage=FG_PIPELINE_STAGE_COUNT-1u-offset;
            uint32_t rank=fg_pipeline_stage(mock->pipelines[stage]);
            if(network_promote_message(
                   mock->network,rank,FG_MSG_PIPELINE_CREDIT))
                return fg_pipeline_step(mock->pipelines[stage],error);
            if(network_promote_message(
                   mock->network,rank,FG_MSG_PIPELINE_ACTIVATION))
                return fg_pipeline_step(mock->pipelines[stage],error);
        }
        if(network_promote_message(
               mock->network,rank0,FG_MSG_PIPELINE_CREDIT))
            return fg_pipeline_step(mock->pipelines[0],error);
    }
    if(mock->production_order_mode){
        uint32_t capacity=fg_pipeline_max_inflight(mock->pipelines[0]);
        uint32_t available=fg_pipeline_available_inflight(mock->pipelines[0]);
        uint32_t used=capacity-available;
        if(used>mock->max_inflight_used)mock->max_inflight_used=used;
        if(!available&&mock->prefill_prepares==capacity)
            mock->capacity_backpressured=true;
        bool terminal_fill=
            mock->execution->count[FG_PIPELINE_STAGE_COUNT-1u]>=capacity;
        uint32_t rank0=fg_pipeline_stage(mock->pipelines[0]);
        if(mock->capacity_backpressured&&terminal_fill&&
           fg_pipeline_available_slots(mock->pipelines[0])&&
           !mock->capacity_limit_healthy){
            fg_error probe_error={0};
            fg_status probe=fg_pipeline_submit(mock->pipelines[0],
                FG_PIPELINE_EXECUTION_PREFILL,
                100u+capacity*FG_PIPELINE_DEFAULT_MICROBATCH,
                FG_PIPELINE_DEFAULT_MICROBATCH,false,NULL,NULL,NULL,
                &probe_error);
            fg_error status_error={0};
            mock->capacity_limit_healthy=probe==FG_ERR_LIMIT&&
                fg_pipeline_status(mock->pipelines[0],&status_error)==FG_OK;
        }
        uint32_t rank1=fg_pipeline_stage(mock->pipelines[1]);
        bool stage0_credit=network_has_message_type(
            mock->network,rank0,FG_MSG_PIPELINE_CREDIT);
        bool stage1_credit=network_has_message_type(
            mock->network,rank1,FG_MSG_PIPELINE_CREDIT);
        if(!mock->network->activation_before_credit_receives&&
           stage0_credit&&stage1_credit)
            return fg_pipeline_step(mock->pipelines[0],error);
        for(uint32_t stage=1u;stage<FG_PIPELINE_STAGE_COUNT;stage++){
            uint32_t rank=fg_pipeline_stage(mock->pipelines[stage]);
            if(!mock->network->activation_before_credit_receives&&stage==1u&&
               stage0_credit&&stage1_credit&&
               !network_has_message_type(
                   mock->network,rank,FG_MSG_PIPELINE_ACTIVATION))
                continue;
            if(network_has_message(mock->network,rank))
                return fg_pipeline_step(mock->pipelines[stage],error);
        }
        if(stage0_credit)
            return fg_pipeline_step(mock->pipelines[0],error);
        if(((mock->capacity_limit_healthy&&terminal_fill)||
            mock->prefill_prepares>capacity)&&
           network_has_message_type(
               mock->network,rank0,FG_MSG_PIPELINE_RESULT)){
            mock->first_result_processed=true;
            return fg_pipeline_step(mock->pipelines[0],error);
        }
    }
    if(mock->delay_stage0_credits){
        uint32_t rank0=fg_pipeline_stage(mock->pipelines[0]);
        if(network_promote_message(mock->network,rank0,
                                   FG_MSG_PIPELINE_RESULT)){
            if(!fg_pipeline_available_slots(mock->pipelines[0]))
                mock->result_before_credit=true;
            return fg_pipeline_step(mock->pipelines[0],error);
        }
        for(uint32_t stage=1u;stage<FG_PIPELINE_STAGE_COUNT;stage++){
            uint32_t rank=fg_pipeline_stage(mock->pipelines[stage]);
            if(network_has_message(mock->network,rank))
                return fg_pipeline_step(mock->pipelines[stage],error);
        }
        if(network_has_message(mock->network,rank0)){
            if(network_first_message_type(mock->network,rank0)==
               FG_MSG_PIPELINE_CREDIT)
                mock->stage0_credit_steps++;
            return fg_pipeline_step(mock->pipelines[0],error);
        }
    }
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        uint32_t rank=fg_pipeline_stage(mock->pipelines[stage]);
        if(network_has_message(mock->network,rank))
            return fg_pipeline_step(mock->pipelines[stage],error);
    }
    fg_error_set(error,FG_ERR_UNAVAILABLE,"synthetic pipeline made no progress");
    return FG_ERR_UNAVAILABLE;
}

static void test_pipeline_runtime_orchestration(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    test_network network={0};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=UINT32_MAX,
        .fail_sequence=UINT32_MAX,.corrupt_stage=UINT32_MAX,
        .corrupt_sequence=UINT32_MAX,.omit_result_stage=UINT32_MAX,
        .omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]={0};
    fg_error error={0};
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        endpoints[stage]=(test_endpoint){.network=&network,
            .rank=manifest.stage_ranks[stage]};
        fg_pipeline_config pipeline_config={.manifest=&manifest,
            .rank=endpoints[stage].rank,.transport=test_transport(&endpoints[stage]),
            .execute=execute_stage,.execute_context=&execution};
        CHECK(fg_pipeline_create(&pipelines[stage],&pipeline_config,&error)==FG_OK);
    }
    runtime_mock mock={.network=&network,.pipelines=pipelines};
    fg_pipeline_runtime *runtime=NULL;
    fg_pipeline_runtime_config config={.manifest=&manifest,
        .pipeline=pipelines[0],.prepare=runtime_prepare,
        .progress=runtime_progress,.context=&mock};
    CHECK(fg_pipeline_runtime_create(&runtime,&config,&error)==FG_OK);
    const uint64_t request=UINT64_C(0x8877665544332211);
    for(uint32_t stage=1;stage<FG_PIPELINE_STAGE_COUNT;stage++)
        CHECK(fg_pipeline_begin(pipelines[stage],request,20u,&error)==FG_OK);
    CHECK(fg_pipeline_runtime_begin(runtime,request,20u,&error)==FG_OK);
    uint32_t *tokens=malloc(300u*sizeof(*tokens));
    CHECK(tokens!=NULL);
    for(uint32_t i=0;i<300u;i++)tokens[i]=100u+i;
    fg_pipeline_result terminal={0};double seconds=0.0;
    if(tokens)CHECK(fg_pipeline_runtime_prefill(runtime,tokens,0u,300u,
                                                &terminal,&seconds,&error)==FG_OK);
    CHECK(mock.prefill_prepares==3u);
    CHECK(mock.activations_before_first_progress==manifest.slot_count);
    CHECK(mock.prepares_before_first_progress==manifest.slot_count);
    CHECK(terminal.has_output&&terminal.completed_first_token==256u&&
          terminal.completed_token_count==44u&&terminal.final_token==1022u);
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        CHECK(execution.count[stage]==3u);
        CHECK(!execution.records[stage][0].request_output);
        CHECK(!execution.records[stage][1].request_output);
        CHECK(execution.records[stage][2].request_output);
        CHECK(execution.records[stage][0].sequence==20u);
        CHECK(execution.records[stage][1].sequence==21u);
        CHECK(execution.records[stage][2].sequence==22u);
    }
    CHECK(fg_pipeline_runtime_decode(runtime,777u,300u,&terminal,&error)==FG_OK);
    CHECK(mock.decode_prepares==1u);
    CHECK(terminal.has_output&&terminal.completed_frontier==301u);
    CHECK(fg_pipeline_runtime_finish(runtime,&error)==FG_OK);
    CHECK(!fg_pipeline_runtime_reopen_required(runtime));
    free(tokens);
    fg_pipeline_runtime_destroy(runtime);
    destroy_chain(pipelines);
    network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX};
    test_endpoint endpoint={.network=&network,.rank=0u};
    fg_pipeline_config pipeline_config={.manifest=&manifest,.rank=0u,
        .transport=test_transport(&endpoint),.execute=execute_stage,
        .execute_context=&execution};
    pipelines[0]=NULL;
    CHECK(fg_pipeline_create(&pipelines[0],&pipeline_config,&error)==FG_OK);
    fg_pipeline *single[FG_PIPELINE_STAGE_COUNT]={0};
    single[0]=pipelines[0];
    mock=(runtime_mock){.network=&network,.pipelines=single};
    config=(fg_pipeline_runtime_config){.manifest=&manifest,
        .pipeline=pipelines[0],.prepare=runtime_prepare,
        .progress=runtime_progress,.context=&mock};
    runtime=NULL;
    CHECK(fg_pipeline_runtime_create(&runtime,&config,&error)==FG_OK);
    CHECK(fg_pipeline_runtime_begin(runtime,request+1u,30u,&error)==FG_OK);
    network.fail_send[0u]=true;
    CHECK(fg_pipeline_runtime_decode(runtime,1u,0u,&terminal,&error)==FG_ERR_IO);
    CHECK(fg_pipeline_runtime_reopen_required(runtime));
    CHECK(fg_pipeline_runtime_begin(runtime,request+2u,40u,&error)==
          FG_ERR_UNAVAILABLE);
    fg_pipeline_runtime_destroy(runtime);
    fg_pipeline_destroy(pipelines[0]);
    network_clear(&network);
}

static void test_pipeline_runtime_full_overlap(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    test_network network={0};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=UINT32_MAX,
        .fail_sequence=UINT32_MAX,.corrupt_stage=UINT32_MAX,
        .corrupt_sequence=UINT32_MAX,.omit_result_stage=UINT32_MAX,
        .omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]={0};
    fg_error error={0};
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        endpoints[stage]=(test_endpoint){.network=&network,
            .rank=manifest.stage_ranks[stage]};
        fg_pipeline_config pipeline_config={.manifest=&manifest,
            .rank=endpoints[stage].rank,
            .transport=test_transport(&endpoints[stage]),
            .execute=execute_stage,.execute_context=&execution};
        CHECK(fg_pipeline_create(&pipelines[stage],&pipeline_config,
                                 &error)==FG_OK);
    }
    CHECK(fg_pipeline_max_inflight(pipelines[0])==
          FG_PIPELINE_MAX_INFLIGHT_CHUNKS);
    runtime_mock mock={.network=&network,.pipelines=pipelines,
        .execution=&execution,.overlap_mode=true};
    fg_pipeline_runtime *runtime=NULL;
    fg_pipeline_runtime_config config={.manifest=&manifest,
        .pipeline=pipelines[0],.prepare=runtime_prepare,
        .progress=runtime_progress,.context=&mock};
    CHECK(fg_pipeline_runtime_create(&runtime,&config,&error)==FG_OK);
    const uint64_t request=UINT64_C(0x1111222233334444);
    for(uint32_t stage=1u;stage<FG_PIPELINE_STAGE_COUNT;stage++)
        CHECK(fg_pipeline_begin(pipelines[stage],request,100u,&error)==FG_OK);
    CHECK(fg_pipeline_runtime_begin(runtime,request,100u,&error)==FG_OK);
    const uint32_t token_count=17u*FG_PIPELINE_DEFAULT_MICROBATCH;
    uint32_t *tokens=malloc((size_t)token_count*sizeof(*tokens));
    CHECK(tokens!=NULL);
    for(uint32_t i=0;i<token_count;i++)tokens[i]=300u+i;
    fg_pipeline_result terminal={0};
    if(tokens)CHECK(fg_pipeline_runtime_prefill(
        runtime,tokens,0u,token_count,&terminal,NULL,&error)==FG_OK);
    CHECK(mock.max_inflight_used==FG_PIPELINE_MAX_INFLIGHT_CHUNKS);
    CHECK(mock.capacity_backpressured);
    CHECK(mock.capacity_limit_healthy);
    CHECK(mock.eight_chunks_crossed_all_stages);
    CHECK(mock.seventeenth_prepare_after_result);
    CHECK(mock.prefill_prepares==17u);
    CHECK(terminal.has_output&&
          terminal.completed_first_token==
              16u*FG_PIPELINE_DEFAULT_MICROBATCH&&
          terminal.completed_token_count==FG_PIPELINE_DEFAULT_MICROBATCH);
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        CHECK(execution.count[stage]==17u);
        for(uint32_t chunk=0;chunk<16u;chunk++)
            CHECK(!execution.records[stage][chunk].request_output);
        CHECK(execution.records[stage][16u].request_output);
    }
    CHECK(!fg_pipeline_runtime_reopen_required(runtime));
    free(tokens);
    fg_pipeline_runtime_destroy(runtime);
    destroy_chain(pipelines);
    network_clear(&network);
}

static void test_pipeline_runtime_production_receive_order(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    test_network network={.lower_source_priority=true};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=UINT32_MAX,
        .fail_sequence=UINT32_MAX,.corrupt_stage=UINT32_MAX,
        .corrupt_sequence=UINT32_MAX,.omit_result_stage=UINT32_MAX,
        .omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]={0};
    fg_error error={0};
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        endpoints[stage]=(test_endpoint){.network=&network,
            .rank=manifest.stage_ranks[stage]};
        fg_pipeline_config pipeline_config={.manifest=&manifest,
            .rank=endpoints[stage].rank,
            .transport=test_transport(&endpoints[stage]),
            .execute=execute_stage,.execute_context=&execution};
        CHECK(fg_pipeline_create(&pipelines[stage],&pipeline_config,
                                 &error)==FG_OK);
    }
    runtime_mock mock={.network=&network,.pipelines=pipelines,
        .execution=&execution,.production_order_mode=true};
    fg_pipeline_runtime *runtime=NULL;
    fg_pipeline_runtime_config config={.manifest=&manifest,
        .pipeline=pipelines[0],.prepare=runtime_prepare,
        .progress=runtime_progress,.context=&mock};
    CHECK(fg_pipeline_runtime_create(&runtime,&config,&error)==FG_OK);
    const uint64_t request=UINT64_C(0x2222333344445555);
    for(uint32_t stage=1u;stage<FG_PIPELINE_STAGE_COUNT;stage++)
        CHECK(fg_pipeline_begin(pipelines[stage],request,100u,&error)==FG_OK);
    CHECK(fg_pipeline_runtime_begin(runtime,request,100u,&error)==FG_OK);
    const uint32_t token_count=17u*FG_PIPELINE_DEFAULT_MICROBATCH;
    uint32_t *tokens=malloc((size_t)token_count*sizeof(*tokens));
    CHECK(tokens!=NULL);
    for(uint32_t i=0;i<token_count;i++)tokens[i]=500u+i;
    fg_pipeline_result terminal={0};
    if(tokens)CHECK(fg_pipeline_runtime_prefill(
        runtime,tokens,0u,token_count,&terminal,NULL,&error)==FG_OK);
    CHECK(network.activation_before_credit_receives>0u);
    CHECK(mock.max_inflight_used==FG_PIPELINE_MAX_INFLIGHT_CHUNKS);
    CHECK(mock.capacity_backpressured);
    CHECK(mock.capacity_limit_healthy);
    CHECK(mock.seventeenth_prepare_after_result);
    CHECK(mock.prefill_prepares==17u);
    CHECK(terminal.has_output);
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        CHECK(execution.count[stage]==17u);
        for(uint32_t chunk=0;chunk<execution.count[stage];chunk++)
            CHECK(execution.records[stage][chunk].sequence==100u+chunk);
        fg_error status_error={0};
        CHECK(fg_pipeline_status(pipelines[stage],&status_error)==FG_OK);
    }
    free(tokens);
    fg_pipeline_runtime_destroy(runtime);
    destroy_chain(pipelines);
    network_clear(&network);
}

static void test_pipeline_runtime_result_before_credit(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    test_network network={0};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=UINT32_MAX,
        .fail_sequence=UINT32_MAX,.corrupt_stage=UINT32_MAX,
        .corrupt_sequence=UINT32_MAX,.omit_result_stage=UINT32_MAX,
        .omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]={0};
    fg_error error={0};
    const uint64_t request=UINT64_C(0x9988776655443322);
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        endpoints[stage]=(test_endpoint){.network=&network,
            .rank=manifest.stage_ranks[stage]};
        fg_pipeline_config pipeline_config={.manifest=&manifest,
            .rank=endpoints[stage].rank,
            .transport=test_transport(&endpoints[stage]),
            .execute=execute_stage,.execute_context=&execution};
        CHECK(fg_pipeline_create(&pipelines[stage],&pipeline_config,
                                 &error)==FG_OK);
    }
    runtime_mock mock={.network=&network,.pipelines=pipelines,
                       .delay_stage0_credits=true};
    fg_pipeline_runtime *runtime=NULL;
    fg_pipeline_runtime_config config={.manifest=&manifest,
        .pipeline=pipelines[0],.prepare=runtime_prepare,
        .progress=runtime_progress,.context=&mock};
    CHECK(fg_pipeline_runtime_create(&runtime,&config,&error)==FG_OK);
    for(uint32_t stage=1u;stage<FG_PIPELINE_STAGE_COUNT;stage++)
        CHECK(fg_pipeline_begin(pipelines[stage],request,50u,&error)==FG_OK);
    CHECK(fg_pipeline_runtime_begin(runtime,request,50u,&error)==FG_OK);
    uint32_t *tokens=malloc(300u*sizeof(*tokens));
    CHECK(tokens!=NULL);
    for(uint32_t i=0;i<300u;i++)tokens[i]=200u+i;
    fg_pipeline_result terminal={0};
    if(tokens)CHECK(fg_pipeline_runtime_prefill(
        runtime,tokens,0u,300u,&terminal,NULL,&error)==FG_OK);
    CHECK(mock.result_before_credit);
    CHECK(mock.stage0_credit_steps>0u);
    CHECK(mock.third_prepare_after_credit);
    CHECK(mock.prefill_prepares==3u);
    CHECK(terminal.has_output&&terminal.completed_first_token==256u&&
          terminal.completed_token_count==44u);
    CHECK(!fg_pipeline_runtime_reopen_required(runtime));
    free(tokens);
    fg_pipeline_runtime_destroy(runtime);
    destroy_chain(pipelines);
    network_clear(&network);
}

static void test_pipeline_session_begin_sequence(void){
    fg_error error={0};
    fg_frame_header header;
    CHECK(fg_frame_encode_version(&header,FG_PIPELINE_PROTOCOL_VERSION,
          FG_MSG_SESSION_BEGIN,77u,0u,FG_PIPELINE_SESSION_RESET_FLAG,
          NULL,0u,&error)==FG_OK);
    CHECK(fg_pipeline_session_begin_validate(&header,0u,&error)==FG_OK);
    CHECK(fg_frame_encode_version(&header,FG_PIPELINE_PROTOCOL_VERSION,
          FG_MSG_SESSION_BEGIN,77u,1u,FG_PIPELINE_SESSION_RESET_FLAG,
          NULL,0u,&error)==FG_OK);
    CHECK(fg_pipeline_session_begin_validate(&header,0u,&error)==
          FG_ERR_MISMATCH);
}

static bool pump_abort_chain(test_network *network,
                             fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT]){
    fg_error error={0};
    for(uint32_t round=0;round<TEST_MESSAGE_CAPACITY;round++){
        bool all_aborted=true,progress=false;
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
            all_aborted=all_aborted&&fg_pipeline_abort_complete(pipelines[stage]);
        if(all_aborted)return true;
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
            if(fg_pipeline_abort_complete(pipelines[stage]))continue;
            uint32_t rank=fg_pipeline_stage(pipelines[stage]);
            if(!network_has_message(network,rank))continue;
            fg_status status=fg_pipeline_step(pipelines[stage],&error);
            if(status!=FG_OK&&!fg_pipeline_abort_complete(pipelines[stage]))
                return false;
            progress=true;
        }
        if(!progress)return false;
    }
    return false;
}

static void test_credit_waits_for_local_release(void){
    fg_manifest manifest;
    CHECK(build_pipeline_manifest(&manifest));
    test_network network={0};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=UINT32_MAX,
        .fail_sequence=UINT32_MAX,.corrupt_stage=UINT32_MAX,
        .corrupt_sequence=UINT32_MAX,.omit_result_stage=UINT32_MAX,
        .omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT];
    const uint64_t request=UINT64_C(0x33445566778899aa);
    CHECK(create_chain(pipelines,endpoints,&network,&manifest,&execution,
                       request,10u));
    if(!pipelines[0]){network_clear(&network);return;}
    uint32_t positions[FG_PIPELINE_POSITION_AXES]={0u,0u,0u};
    float *boundary=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(*boundary));
    CHECK(boundary!=NULL);
    if(boundary){
        fg_error error={0};
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_PREFILL,
              0u,1u,false,positions,boundary,NULL,&error)==FG_OK);
        positions[0]=positions[1]=positions[2]=1u;
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_PREFILL,
              1u,1u,false,positions,boundary,NULL,&error)==FG_OK);
        CHECK(fg_pipeline_step(pipelines[1],&error)==FG_OK);
        CHECK(fg_pipeline_step(pipelines[1],&error)==FG_OK);
        CHECK(fg_pipeline_step(pipelines[0],&error)==FG_OK);
        positions[0]=positions[1]=positions[2]=2u;
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_PREFILL,
              2u,1u,false,positions,boundary,NULL,&error)==FG_OK);
        CHECK(fg_pipeline_step(pipelines[0],&error)==FG_OK);
        positions[0]=positions[1]=positions[2]=3u;
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_PREFILL,
              3u,1u,true,positions,boundary,NULL,&error)==FG_OK);
        CHECK(fg_pipeline_step(pipelines[1],&error)==FG_OK);
        CHECK(fg_pipeline_step(pipelines[1],&error)==FG_OK);
        CHECK(!network_has_message_type(
              &network,manifest.stage_ranks[0],FG_MSG_PIPELINE_CREDIT));
        CHECK(fg_pipeline_available_slots(pipelines[0])==0u);
        CHECK(execution.count[1]==4u&&execution.count[2]==0u);
        CHECK(pump_network(&network,pipelines));
        CHECK(execution.count[FG_PIPELINE_STAGE_COUNT-1u]==4u);
        CHECK(fg_pipeline_available_slots(pipelines[0])==
              manifest.slot_count);
    }
    free(boundary);
    destroy_chain(pipelines);
    network_clear(&network);
}

static void test_two_slot_chain(void){
    fg_manifest manifest;
    CHECK(build_pipeline_manifest(&manifest));
    test_network network={0};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT];
    uint64_t request_id=UINT64_C(0x1122334455667788);
    CHECK(create_chain(pipelines,endpoints,&network,&manifest,&execution,
                       request_id,10u));
    if(!pipelines[0]){network_clear(&network);return;}
    CHECK(fg_pipeline_slot_count(pipelines[0])==manifest.slot_count);
    CHECK(fg_pipeline_available_slots(pipelines[0])==manifest.slot_count);
    CHECK(fg_pipeline_host_bytes(pipelines[0])>
          (uint64_t)manifest.slot_count*manifest.prefill_microbatch*
              FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float));

    uint32_t positions0[2u*FG_PIPELINE_POSITION_AXES]={
        0u,0u,0u,1u,1u,1u
    };
    uint32_t positions1[FG_PIPELINE_POSITION_AXES]={2u,2u,2u};
    float *prefill=calloc(2u*FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
    float *decode=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
    CHECK(prefill&&decode);
    if(prefill&&decode){
        fg_error error={0};uint32_t sequence0=0u,sequence1=0u;
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_PREFILL,
              0u,2u,false,positions0,prefill,&sequence0,&error)==FG_OK);
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_DECODE,
              2u,1u,true,positions1,decode,&sequence1,&error)==FG_OK);
        CHECK(sequence0==10u&&sequence1==11u);
        CHECK(fg_pipeline_available_slots(pipelines[0])==0u);
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_DECODE,
              3u,1u,true,positions1,decode,NULL,&error)==FG_ERR_LIMIT);
        CHECK(pump_network(&network,pipelines));
        CHECK(fg_pipeline_published_frontier(pipelines[0])==3u);
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
            CHECK(execution.count[stage]==2u);
            CHECK(execution.records[stage][0].sequence==10u);
            CHECK(execution.records[stage][1].sequence==11u);
            CHECK(execution.records[stage][0].kind==FG_PIPELINE_EXECUTION_PREFILL);
            CHECK(execution.records[stage][1].kind==FG_PIPELINE_EXECUTION_DECODE);
        }
        fg_pipeline_result result={0};uint32_t result_sequence=0u;
        CHECK(fg_pipeline_take_result(
              pipelines[0],&result,&result_sequence,&error)==FG_OK);
        CHECK(result_sequence==10u&&result.completed_frontier==2u&&
              !result.has_output&&result.final_token==FG_Q38_VOCAB_SIZE&&
              result.final_logit==0.0f);
        CHECK(fg_pipeline_take_result(
              pipelines[0],&result,&result_sequence,&error)==FG_OK);
        CHECK(result_sequence==11u&&result.completed_frontier==3u&&
              result.has_output&&result.final_token==1011u);
        CHECK(fg_pipeline_available_slots(pipelines[0])==manifest.slot_count);

        uint32_t positions2[FG_PIPELINE_POSITION_AXES]={3u,3u,3u};
        uint32_t sequence2=0u;
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_PREFILL,
              3u,1u,true,positions2,decode,&sequence2,&error)==FG_OK);
        CHECK(sequence2==12u);
        CHECK(execution.records[0][2].slot==0u);
        CHECK(fg_pipeline_request_drain(pipelines[0],&error)==FG_OK);
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_DECODE,
              4u,1u,true,positions2,decode,NULL,&error)==FG_ERR_ARGUMENT);
        CHECK(pump_network(&network,pipelines));
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
            CHECK(fg_pipeline_is_drained(pipelines[stage]));
        CHECK(fg_pipeline_admission_frontier(pipelines[0])==4u);
        CHECK(fg_pipeline_published_frontier(pipelines[0])==4u);
        CHECK(fg_pipeline_begin(
              pipelines[0],request_id+1u,20u,&error)==FG_ERR_LIMIT);
        CHECK(fg_pipeline_take_result(
              pipelines[0],&result,&result_sequence,&error)==FG_OK);
        CHECK(result_sequence==12u&&result.completed_frontier==4u&&
              result.has_output&&result.final_token==1012u);
        CHECK(fg_pipeline_begin(pipelines[0],request_id+1u,20u,&error)==FG_OK);
        CHECK(fg_pipeline_admission_frontier(pipelines[0])==0u);
        CHECK(fg_pipeline_published_frontier(pipelines[0])==0u);
    }
    free(decode);free(prefill);
    destroy_chain(pipelines);
    network_clear(&network);
}

static void test_full_chain_abort(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    test_network network={0};
    test_endpoint endpoints[FG_PIPELINE_STAGE_COUNT];
    execution_state execution={.fail_stage=3u,.fail_sequence=10u,
        .corrupt_stage=UINT32_MAX,.corrupt_sequence=UINT32_MAX,
        .omit_result_stage=UINT32_MAX,.omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipelines[FG_PIPELINE_STAGE_COUNT];
    const uint64_t request_id=UINT64_C(0xfeed0000feed0000);
    CHECK(create_chain(pipelines,endpoints,&network,&manifest,&execution,
                       request_id,10u));
    if(!pipelines[0]){network_clear(&network);return;}
    uint32_t positions[FG_PIPELINE_POSITION_AXES]={0u,0u,0u};
    uint32_t positions1[FG_PIPELINE_POSITION_AXES]={1u,1u,1u};
    float *boundary=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
    CHECK(boundary!=NULL);
    if(boundary){
        fg_error error={0};
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_DECODE,
              0u,1u,true,positions,boundary,NULL,&error)==FG_OK);
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_DECODE,
              1u,1u,true,positions1,boundary,NULL,&error)==FG_OK);
        CHECK(pump_abort_chain(&network,pipelines));
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
            CHECK(fg_pipeline_abort_complete(pipelines[stage]));
            CHECK(fg_pipeline_status(pipelines[stage],&error)==FG_ERR_IO);
            CHECK(fg_pipeline_published_frontier(pipelines[stage])==0u);
        }
        CHECK(execution.count[0u]==2u);
        CHECK(execution.count[3u]==1u);
        for(uint32_t stage=4u;stage<FG_PIPELINE_STAGE_COUNT;stage++)
            CHECK(execution.count[stage]==0u);
        CHECK(fg_pipeline_admission_frontier(pipelines[0])==2u);
        CHECK(fg_pipeline_begin(
              pipelines[0],request_id+1u,20u,&error)==FG_ERR_IO);
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
            CHECK(fg_pipeline_discard_aborted(pipelines[stage],&error)==FG_OK);
        for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++)
            CHECK(fg_pipeline_begin(pipelines[stage],request_id+1u,20u,&error)==
                  FG_ERR_IO);
        CHECK(network.count>0u);
    }
    free(boundary);
    destroy_chain(pipelines);

    test_network replacement={0};
    execution_state replacement_execution={.fail_stage=UINT32_MAX,
        .fail_sequence=UINT32_MAX,.corrupt_stage=UINT32_MAX,
        .corrupt_sequence=UINT32_MAX,.omit_result_stage=UINT32_MAX,
        .omit_result_sequence=UINT32_MAX};
    CHECK(create_chain(pipelines,endpoints,&replacement,&manifest,
                       &replacement_execution,request_id+1u,20u));
    boundary=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
    CHECK(boundary!=NULL);
    if(boundary&&pipelines[0]){
        fg_error error={0};fg_pipeline_result result={0};uint32_t sequence=0u;
        CHECK(fg_pipeline_submit(pipelines[0],FG_PIPELINE_EXECUTION_DECODE,
              0u,1u,true,positions,boundary,NULL,&error)==FG_OK);
        CHECK(pump_network(&replacement,pipelines));
        CHECK(fg_pipeline_take_result(
              pipelines[0],&result,&sequence,&error)==FG_OK);
        CHECK(sequence==20u&&result.completed_frontier==1u);
    }
    free(boundary);
    destroy_chain(pipelines);
    network_clear(&replacement);
    network_clear(&network);
}

static bool inject_activation(test_endpoint *source,uint64_t request_id,
                              uint32_t sequence,uint8_t slot,fg_error *error){
    uint32_t positions[FG_PIPELINE_POSITION_AXES]={0u,0u,0u};
    float *boundary=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
    uint8_t *wire=malloc(FG_PIPELINE_ACTIVATION_HEADER_BYTES+
        FG_PIPELINE_POSITION_AXES*4u+
        FG_PIPELINE_BOUNDARY_WIDTH*FG_PIPELINE_BOUNDARY_FP32_BYTES);
    if(!boundary||!wire){free(wire);free(boundary);return false;}
    fg_pipeline_activation activation={.execution_kind=FG_PIPELINE_EXECUTION_DECODE,
        .slot=slot,.source_stage=0u,.destination_stage=1u,.first_token=0u,
        .token_count=1u,.request_output=true,.positions=positions,.boundary=boundary};
    uint32_t bytes=0u;
    fg_status status=fg_pipeline_activation_encode(wire,
        FG_PIPELINE_ACTIVATION_HEADER_BYTES+FG_PIPELINE_POSITION_AXES*4u+
            FG_PIPELINE_BOUNDARY_WIDTH*FG_PIPELINE_BOUNDARY_FP32_BYTES,
        &bytes,&activation,error);
    if(status==FG_OK)status=test_send(source,1u,FG_MSG_PIPELINE_ACTIVATION,
                                     request_id,sequence,wire,bytes,error);
    free(wire);free(boundary);
    return status==FG_OK;
}

static fg_pipeline *create_single_stage_one(fg_manifest *manifest,test_network *network,
                                            test_endpoint *endpoint,
                                            execution_state *execution,
                                            uint64_t request_id){
    *endpoint=(test_endpoint){.network=network,.rank=1u};
    fg_pipeline_config config={.manifest=manifest,.rank=1u,
        .transport=test_transport(endpoint),.execute=execute_stage,
        .execute_context=execution};
    fg_pipeline *pipeline=NULL;fg_error error={0};
    if(fg_pipeline_create(&pipeline,&config,&error)!=FG_OK)return NULL;
    if(fg_pipeline_begin(pipeline,request_id,10u,&error)!=FG_OK){
        fg_pipeline_destroy(pipeline);return NULL;
    }
    return pipeline;
}

static fg_pipeline *create_single_stage_zero(fg_manifest *manifest,test_network *network,
                                             test_endpoint *endpoint,
                                             execution_state *execution,
                                             uint64_t request_id){
    *endpoint=(test_endpoint){.network=network,.rank=0u};
    fg_pipeline_config config={.manifest=manifest,.rank=0u,
        .transport=test_transport(endpoint),.execute=execute_stage,
        .execute_context=execution};
    fg_pipeline *pipeline=NULL;fg_error error={0};
    if(fg_pipeline_create(&pipeline,&config,&error)!=FG_OK)return NULL;
    if(fg_pipeline_begin(pipeline,request_id,10u,&error)!=FG_OK){
        fg_pipeline_destroy(pipeline);return NULL;
    }
    return pipeline;
}

static bool inject_result(test_endpoint *source,uint64_t request_id,uint32_t sequence,
                          uint32_t first_token,uint16_t token_count,bool has_output,
                          fg_error *error){
    fg_pipeline_result result={.completed_first_token=first_token,
        .completed_token_count=token_count,
        .completed_frontier=first_token+token_count,
        .has_output=has_output,
        .final_token=has_output?17u:FG_Q38_VOCAB_SIZE,
        .final_logit=has_output?1.0f:0.0f};
    uint8_t payload[FG_PIPELINE_RESULT_BYTES];
    if(fg_pipeline_result_encode(payload,&result,error)!=FG_OK)return false;
    return test_send(source,0u,FG_MSG_PIPELINE_RESULT,request_id,sequence,
                     payload,sizeof(payload),error)==FG_OK;
}

static bool inject_drain(test_endpoint *source,uint64_t request_id,uint32_t sequence,
                         fg_error *error){
    fg_pipeline_drain drain={.source_stage=0u,.destination_stage=1u};
    uint8_t payload[FG_PIPELINE_DRAIN_BYTES];
    if(fg_pipeline_drain_encode(payload,&drain,error)!=FG_OK)return false;
    return test_send(source,1u,FG_MSG_PIPELINE_DRAIN,request_id,sequence,
                     payload,sizeof(payload),error)==FG_OK;
}

static void test_activation_validation_and_frontiers(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    const uint64_t request_id=UINT64_C(0xaabbccddeeff0011);
    fg_error error={0};
    uint32_t positions[2u*FG_PIPELINE_POSITION_AXES]={
        5u,5u,5u,6u,6u,6u
    };
    float *boundary=calloc(2u*FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
    CHECK(boundary!=NULL);
    if(!boundary)return;
    fg_pipeline_activation activation={.execution_kind=FG_PIPELINE_EXECUTION_DECODE,
        .slot=0u,.source_stage=0u,.destination_stage=1u,.first_token=5u,
        .token_count=1u,.request_output=true,.positions=positions,.boundary=boundary};
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_OK);
    activation.request_output=false;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    activation.request_output=true;
    activation.execution_kind=(fg_pipeline_execution_kind)99u;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    activation.execution_kind=FG_PIPELINE_EXECUTION_DECODE;activation.token_count=2u;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    activation.token_count=1u;activation.first_token=FG_NATIVE_CONTEXT;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    activation.first_token=5u;positions[0]=FG_NATIVE_CONTEXT;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    positions[0]=5u;
    for(uint32_t axis=0;axis<FG_PIPELINE_POSITION_AXES;axis++){
        positions[axis]=6u;
        CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
        positions[axis]=5u;
    }
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_OK);
    boundary[0]=NAN;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    boundary[0]=0.0f;activation.slot=FG_PIPELINE_DEFAULT_SLOT_COUNT;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);
    activation.slot=0u;activation.destination_stage=2u;
    CHECK(fg_pipeline_activation_validate(&activation,&error)==FG_ERR_FORMAT);

    test_network network={0};test_endpoint endpoint={0};
    execution_state execution={.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX,
        .corrupt_stage=UINT32_MAX,.corrupt_sequence=UINT32_MAX};
    fg_pipeline *pipeline=create_single_stage_zero(
        &manifest,&network,&endpoint,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_DECODE,
              5u,1u,false,positions,boundary,NULL,&error)==FG_ERR_FORMAT);
        CHECK(execution.count[0]==0u);
        CHECK(network_message_count_type(
              &network,FG_MSG_PIPELINE_ACTIVATION)==0u);
        CHECK(network_message_count_type(&network,FG_MSG_PIPELINE_ABORT)==1u);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_FORMAT);
    }
    fg_pipeline_destroy(pipeline);network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX,
        .corrupt_stage=0u,.corrupt_sequence=10u};
    pipeline=create_single_stage_zero(
        &manifest,&network,&endpoint,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_DECODE,
              5u,1u,true,positions,boundary,NULL,&error)==FG_ERR_FORMAT);
        CHECK(execution.count[0]==1u);
        CHECK(network_message_count_type(
              &network,FG_MSG_PIPELINE_ACTIVATION)==0u);
        CHECK(network_message_count_type(&network,FG_MSG_PIPELINE_ABORT)==1u);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_FORMAT);
    }
    fg_pipeline_destroy(pipeline);network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX,
        .corrupt_stage=UINT32_MAX,.corrupt_sequence=UINT32_MAX};
    pipeline=create_single_stage_zero(
        &manifest,&network,&endpoint,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        uint32_t first_positions[2u*FG_PIPELINE_POSITION_AXES]={
            5u,5u,5u,6u,6u,6u
        };
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_PREFILL,
              5u,2u,false,first_positions,boundary,NULL,&error)==FG_OK);
        CHECK(fg_pipeline_admission_frontier(pipeline)==7u);
        uint32_t gap_positions[FG_PIPELINE_POSITION_AXES]={8u,8u,8u};
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_DECODE,
              8u,1u,true,gap_positions,boundary,NULL,&error)==FG_ERR_MISMATCH);
        uint32_t overlap_positions[FG_PIPELINE_POSITION_AXES]={6u,6u,6u};
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_DECODE,
              6u,1u,true,overlap_positions,boundary,NULL,&error)==FG_ERR_MISMATCH);
        uint32_t next_positions[FG_PIPELINE_POSITION_AXES]={7u,7u,7u};
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_DECODE,
              7u,1u,true,next_positions,boundary,NULL,&error)==FG_OK);
        CHECK(execution.count[0]==2u);
        CHECK(fg_pipeline_admission_frontier(pipeline)==8u);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_OK);
    }
    fg_pipeline_destroy(pipeline);network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX,
        .corrupt_stage=UINT32_MAX,.corrupt_sequence=UINT32_MAX};
    pipeline=create_single_stage_zero(
        &manifest,&network,&endpoint,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        CHECK(fg_pipeline_submit(pipeline,FG_PIPELINE_EXECUTION_DECODE,
              5u,1u,true,positions,boundary,NULL,&error)==FG_OK);
        test_endpoint terminal={.network=&network,.rank=7u};
        CHECK(inject_result(&terminal,request_id,10u,5u,1u,false,&error));
        CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_MISMATCH);
        CHECK(fg_pipeline_admission_frontier(pipeline)==6u);
        CHECK(fg_pipeline_published_frontier(pipeline)==0u);
    }
    fg_pipeline_destroy(pipeline);network_clear(&network);
    free(boundary);
}

static void test_sequence_poison(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    const uint64_t request_id=UINT64_C(0x8877665544332211);
    fg_error error={0};
    for(uint32_t scenario=0;scenario<3u;scenario++){
        test_network network={0};
        test_endpoint receiver={0},source={.network=&network,.rank=0u};
        execution_state execution={.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX};
        fg_pipeline *pipeline=create_single_stage_one(
            &manifest,&network,&receiver,&execution,request_id);
        CHECK(pipeline!=NULL);
        if(pipeline){
            if(scenario==0u){
                CHECK(inject_activation(&source,request_id,11u,0u,&error));
                CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_MISMATCH);
            }else if(scenario==1u){
                CHECK(inject_activation(&source,request_id,10u,0u,&error));
                CHECK(fg_pipeline_step(pipeline,&error)==FG_OK);
                CHECK(inject_activation(&source,request_id,10u,1u,&error));
                CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_MISMATCH);
            }else{
                CHECK(inject_activation(&source,request_id+1u,10u,0u,&error));
                CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_MISMATCH);
            }

            CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_MISMATCH);
            CHECK(fg_pipeline_published_frontier(pipeline)==0u);
        }
        fg_pipeline_destroy(pipeline);
        network_clear(&network);
    }
}

static void test_post_drain_activation_poison(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    const uint64_t request_id=UINT64_C(0x0dd00dd00dd00dd0);
    fg_error error={0};
    test_network network={0};
    test_endpoint receiver={0},source={.network=&network,.rank=0u};
    execution_state execution={.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX,
        .corrupt_stage=UINT32_MAX,.corrupt_sequence=UINT32_MAX,
        .omit_result_stage=UINT32_MAX,.omit_result_sequence=UINT32_MAX};
    fg_pipeline *pipeline=create_single_stage_one(
        &manifest,&network,&receiver,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        CHECK(inject_drain(&source,request_id,10u,&error));
        CHECK(fg_pipeline_step(pipeline,&error)==FG_OK);
        CHECK(inject_activation(&source,request_id,10u,0u,&error));
        CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_MISMATCH);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_MISMATCH);
        CHECK(execution.count[1u]==0u);
    }
    fg_pipeline_destroy(pipeline);
    network_clear(&network);
}

static void test_callback_and_peer_failure(void){
    fg_manifest manifest;CHECK(build_pipeline_manifest(&manifest));
    const uint64_t request_id=UINT64_C(0x1234000012340000);
    fg_error error={0};
    test_network network={0};
    test_endpoint receiver={0},source={.network=&network,.rank=0u};
    execution_state execution={.fail_stage=1u,.fail_sequence=10u};
    fg_pipeline *pipeline=create_single_stage_one(
        &manifest,&network,&receiver,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        CHECK(inject_activation(&source,request_id,10u,0u,&error));
        CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_IO);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_IO);
        CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_IO);
        CHECK(fg_pipeline_published_frontier(pipeline)==0u);
    }
    fg_pipeline_destroy(pipeline);
    network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX,
        .corrupt_stage=UINT32_MAX,.corrupt_sequence=UINT32_MAX,
        .omit_result_stage=7u,.omit_result_sequence=10u};
    test_endpoint terminal_receiver={.network=&network,.rank=7u};
    fg_pipeline_config terminal_config={.manifest=&manifest,.rank=7u,
        .transport=test_transport(&terminal_receiver),.execute=execute_stage,
        .execute_context=&execution};
    pipeline=NULL;
    CHECK(fg_pipeline_create(&pipeline,&terminal_config,&error)==FG_OK);
    if(pipeline){
        CHECK(fg_pipeline_begin(pipeline,request_id,10u,&error)==FG_OK);
        uint32_t positions[FG_PIPELINE_POSITION_AXES]={0u,0u,0u};
        float *boundary=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
        uint8_t *wire=malloc(FG_PIPELINE_ACTIVATION_HEADER_BYTES+
            FG_PIPELINE_POSITION_AXES*4u+
            FG_PIPELINE_BOUNDARY_WIDTH*FG_PIPELINE_BOUNDARY_FP32_BYTES);
        CHECK(boundary&&wire);
        if(boundary&&wire){
            fg_pipeline_activation activation={
                .execution_kind=FG_PIPELINE_EXECUTION_DECODE,.slot=0u,
                .source_stage=6u,.destination_stage=7u,.first_token=0u,
                .token_count=1u,.request_output=true,
                .positions=positions,.boundary=boundary
            };
            uint32_t bytes=0u;
            CHECK(fg_pipeline_activation_encode(wire,
                  FG_PIPELINE_ACTIVATION_HEADER_BYTES+
                      FG_PIPELINE_POSITION_AXES*4u+
                      FG_PIPELINE_BOUNDARY_WIDTH*FG_PIPELINE_BOUNDARY_FP32_BYTES,
                  &bytes,&activation,&error)==FG_OK);
            test_endpoint terminal_source={.network=&network,.rank=6u};
            CHECK(test_send(&terminal_source,7u,FG_MSG_PIPELINE_ACTIVATION,
                  request_id,10u,wire,bytes,&error)==FG_OK);
            CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_FORMAT);
            CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_FORMAT);
            CHECK(execution.count[7u]==1u);
            CHECK(!network_has_message(&network,0u));
        }
        free(wire);free(boundary);
    }
    fg_pipeline_destroy(pipeline);
    network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX};
    pipeline=create_single_stage_one(&manifest,&network,&receiver,&execution,request_id);
    CHECK(pipeline!=NULL);
    if(pipeline){
        network.fail_receive[1u]=true;
        CHECK(fg_pipeline_step(pipeline,&error)==FG_ERR_IO);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_IO);
    }
    fg_pipeline_destroy(pipeline);
    network_clear(&network);

    execution=(execution_state){.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX};
    test_endpoint sender={.network=&network,.rank=0u};
    fg_pipeline_config config={.manifest=&manifest,.rank=0u,
        .transport=test_transport(&sender),.execute=execute_stage,
        .execute_context=&execution};
    pipeline=NULL;
    CHECK(fg_pipeline_create(&pipeline,&config,&error)==FG_OK);
    if(pipeline){
        CHECK(fg_pipeline_begin(pipeline,request_id,10u,&error)==FG_OK);
        network.fail_send[0u]=true;
        uint32_t positions[FG_PIPELINE_POSITION_AXES]={0u,0u,0u};
        float *boundary=calloc(FG_PIPELINE_BOUNDARY_WIDTH,sizeof(float));
        CHECK(boundary!=NULL);
        if(boundary)CHECK(fg_pipeline_submit(
            pipeline,FG_PIPELINE_EXECUTION_DECODE,0u,1u,true,positions,boundary,
            NULL,&error)==FG_ERR_IO);
        CHECK(fg_pipeline_status(pipeline,&error)==FG_ERR_IO);
        CHECK(!fg_pipeline_abort_complete(pipeline));
        CHECK(fg_pipeline_discard_aborted(pipeline,&error)==FG_OK);
        CHECK(fg_pipeline_begin(pipeline,request_id+1u,20u,&error)==FG_ERR_IO);
        free(boundary);
    }
    fg_pipeline_destroy(pipeline);
    network_clear(&network);
}

static void test_protocol6_unaffected(void){
    fg_manifest manifest;fg_manifest_init(&manifest);
    test_network network={0};test_endpoint endpoint={.network=&network,.rank=0u};
    execution_state execution={.fail_stage=UINT32_MAX,.fail_sequence=UINT32_MAX};
    fg_pipeline_config config={.manifest=&manifest,.rank=0u,
        .transport=test_transport(&endpoint),.execute=execute_stage,
        .execute_context=&execution};
    fg_pipeline *pipeline=NULL;fg_error error={0};
    CHECK(fg_pipeline_create(&pipeline,&config,&error)==FG_ERR_MISMATCH);
    CHECK(pipeline==NULL);
    fg_frame_header header;
    CHECK(fg_frame_encode_version(&header,FG_PROTOCOL_VERSION,FG_MSG_CONTROL,
          1u,1u,0u,NULL,0u,&error)==FG_OK);
    CHECK(fg_frame_validate_version(
          &header,FG_PROTOCOL_VERSION,NULL,NULL,&error)==FG_OK);
    network_clear(&network);
}

int main(void){
    test_pipeline_runtime_orchestration();
    test_pipeline_runtime_full_overlap();
    test_pipeline_runtime_production_receive_order();
    test_credit_waits_for_local_release();
    test_pipeline_runtime_result_before_credit();
    test_pipeline_session_begin_sequence();
    test_two_slot_chain();
    test_full_chain_abort();
    test_activation_validation_and_frontiers();
    test_sequence_poison();
    test_post_drain_activation_poison();
    test_callback_and_peer_failure();
    test_protocol6_unaffected();
    if(failures){
        fprintf(stderr,"%d pipeline scheduler test(s) failed\n",failures);
        return 1;
    }
    puts("Flash Gordon bounded pipeline scheduler: PASS");
    return 0;
}
