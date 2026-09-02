#include "fg_fabric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define fg_prefill_layer_work_encode(output,capacity,bytes,work,error) \
    fg_prefill_layer_work_encode(output,capacity,bytes,fg_fabric_protocol_version(fabric),work,error)
#define fg_prefill_layer_work_decode(work,positions,position_capacity,hyper,hyper_capacity,ngram,ngram_capacity,payload,bytes,error) \
    fg_prefill_layer_work_decode(work,fg_fabric_protocol_version(fabric),positions,position_capacity,hyper,hyper_capacity,ngram,ngram_capacity,payload,bytes,error)
#define fg_layer_work_encode(output,capacity,bytes,work,error) \
    fg_layer_work_encode(output,capacity,bytes,fg_fabric_protocol_version(fabric),work,error)
#define fg_layer_work_decode(work,payload,bytes,error) \
    fg_layer_work_decode(work,fg_fabric_protocol_version(fabric),payload,bytes,error)

static int protocol_failures;
#define PROTOCOL_CHECK(expression) do { \
    if(!(expression)){ \
        fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#expression); \
        protocol_failures++; \
    } \
} while(0)

static void pipeline_protocol_test(void){
    enum{TOKENS=2};
    fg_error error={0};
    uint32_t positions[TOKENS*FG_PIPELINE_POSITION_AXES]={
        40u,40u,40u,41u,41u,41u
    };
    uint32_t decoded_positions[TOKENS*FG_PIPELINE_POSITION_AXES]={0};
    uint64_t boundary_values=(uint64_t)TOKENS*FG_PIPELINE_BOUNDARY_WIDTH;
    float *boundary=malloc((size_t)boundary_values*sizeof(*boundary));
    float *decoded_boundary=malloc((size_t)boundary_values*sizeof(*decoded_boundary));
    uint8_t *wire=malloc(FG_PIPELINE_ACTIVATION_MAX_BYTES);
    PROTOCOL_CHECK(boundary&&decoded_boundary&&wire);
    if(!boundary||!decoded_boundary||!wire){
        free(wire);free(decoded_boundary);free(boundary);return;
    }
    for(uint64_t i=0;i<boundary_values;i++)boundary[i]=(float)i*0.0001f;

    fg_pipeline_activation activation={
        .execution_kind=FG_PIPELINE_EXECUTION_PREFILL,
        .slot=1u,.source_stage=3u,.destination_stage=4u,
        .first_token=40u,.token_count=TOKENS,.request_output=false,
        .sampler={1.0f,0.95f,20u,0u},.uniform=0.37f,
        .positions=positions,.boundary=boundary
    },decoded={0};
    uint32_t bytes=0;
    PROTOCOL_CHECK(fg_protocol_version_supported(FG_PIPELINE_PROTOCOL_VERSION));
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_OK);
    PROTOCOL_CHECK(bytes==FG_PIPELINE_ACTIVATION_HEADER_BYTES+
        TOKENS*FG_PIPELINE_POSITION_AXES*4u+boundary_values*4u);
    PROTOCOL_CHECK(memcmp(
        wire+FG_PIPELINE_ACTIVATION_HEADER_BYTES+
            TOKENS*FG_PIPELINE_POSITION_AXES*4u,
        boundary,(size_t)boundary_values*4u)==0);
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes,&error)==FG_OK);
    PROTOCOL_CHECK(decoded.execution_kind==activation.execution_kind&&
        decoded.slot==activation.slot&&decoded.source_stage==activation.source_stage&&
        decoded.destination_stage==activation.destination_stage&&
        decoded.first_token==activation.first_token&&decoded.token_count==TOKENS&&
        !decoded.request_output&&decoded.sampler.temperature==activation.sampler.temperature&&
        decoded.sampler.top_p==activation.sampler.top_p&&decoded.sampler.top_k==activation.sampler.top_k&&
        decoded.uniform==activation.uniform);
    PROTOCOL_CHECK(memcmp(decoded_positions,positions,sizeof(positions))==0);
    PROTOCOL_CHECK(memcmp(decoded_boundary,boundary,(size_t)boundary_values*4u)==0);
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes-1u,&error)==FG_ERR_FORMAT);
    wire[10]=2u;
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes,&error)==FG_ERR_FORMAT);
    wire[10]=0u;
    wire[11]=1u;
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes,&error)==FG_ERR_FORMAT);
    wire[11]=0u;
    float finite=boundary[0],nonfinite=NAN;
    uint32_t tensor_offset=FG_PIPELINE_ACTIVATION_HEADER_BYTES+
        TOKENS*FG_PIPELINE_POSITION_AXES*4u;
    memcpy(wire+tensor_offset,&nonfinite,sizeof(nonfinite));
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes,&error)==FG_ERR_FORMAT);
    memcpy(wire+tensor_offset,&finite,sizeof(finite));
    activation.slot=FG_PIPELINE_DEFAULT_SLOT_COUNT;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_ERR_FORMAT);
    activation.slot=1u;activation.destination_stage=5u;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_ERR_FORMAT);
    activation.source_stage=7u;activation.destination_stage=8u;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_ERR_FORMAT);
    activation.source_stage=3u;activation.destination_stage=4u;
    activation.execution_kind=FG_PIPELINE_EXECUTION_DECODE;
    activation.token_count=1u;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_ERR_FORMAT);
    activation.request_output=true;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes,&error)==FG_OK);
    PROTOCOL_CHECK(decoded.request_output);
    wire[10]=0u;
    PROTOCOL_CHECK(fg_pipeline_activation_decode(
        &decoded,decoded_positions,TOKENS*FG_PIPELINE_POSITION_AXES,
        decoded_boundary,boundary_values,wire,bytes,&error)==FG_ERR_FORMAT);
    activation.execution_kind=FG_PIPELINE_EXECUTION_PREFILL;
    activation.token_count=TOKENS;activation.request_output=false;
    positions[5]=FG_NATIVE_CONTEXT;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_ERR_FORMAT);
    positions[5]=41u;boundary[1]=INFINITY;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_ERR_FORMAT);
    boundary[1]=0.0001f;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_HEADER_BYTES,&bytes,&activation,&error)==FG_ERR_LIMIT);

    fg_pipeline_credit credit={.source_stage=4u,.destination_stage=3u,.slot=1u};
    fg_pipeline_credit decoded_credit={0};
    uint8_t credit_wire[FG_PIPELINE_CREDIT_BYTES];
    PROTOCOL_CHECK(fg_pipeline_credit_encode(credit_wire,&credit,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_credit_decode(
        &decoded_credit,credit_wire,sizeof(credit_wire),&error)==FG_OK);
    PROTOCOL_CHECK(decoded_credit.source_stage==4u&&
        decoded_credit.destination_stage==3u&&decoded_credit.slot==1u);
    credit_wire[3]=1u;
    PROTOCOL_CHECK(fg_pipeline_credit_decode(
        &decoded_credit,credit_wire,sizeof(credit_wire),&error)==FG_ERR_FORMAT);
    credit_wire[3]=0u;credit.destination_stage=2u;
    PROTOCOL_CHECK(fg_pipeline_credit_encode(credit_wire,&credit,&error)==FG_ERR_FORMAT);
    credit.source_stage=0u;credit.destination_stage=UINT8_MAX;
    PROTOCOL_CHECK(fg_pipeline_credit_encode(credit_wire,&credit,&error)==FG_ERR_FORMAT);
    credit.source_stage=FG_PIPELINE_STAGE_COUNT;credit.destination_stage=7u;
    PROTOCOL_CHECK(fg_pipeline_credit_encode(credit_wire,&credit,&error)==FG_ERR_FORMAT);

    fg_pipeline_result result={.completed_first_token=40u,.completed_token_count=TOKENS,
        .completed_frontier=42u,.has_output=true,
        .final_token=1234u,.final_logit=7.5f};
    fg_pipeline_result decoded_result={0};
    uint8_t result_wire[FG_PIPELINE_RESULT_BYTES];
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_result_decode(
        &decoded_result,result_wire,sizeof(result_wire),&error)==FG_OK);
    PROTOCOL_CHECK(decoded_result.completed_first_token==40u&&
        decoded_result.completed_token_count==TOKENS&&
        decoded_result.completed_frontier==42u&&decoded_result.has_output&&
        decoded_result.final_token==1234u&&
        decoded_result.final_logit==7.5f);
    result_wire[6]=2u;
    PROTOCOL_CHECK(fg_pipeline_result_decode(
        &decoded_result,result_wire,sizeof(result_wire),&error)==FG_ERR_FORMAT);
    result_wire[6]=1u;result_wire[7]=1u;
    PROTOCOL_CHECK(fg_pipeline_result_decode(
        &decoded_result,result_wire,sizeof(result_wire),&error)==FG_ERR_FORMAT);
    result_wire[7]=0u;result.completed_frontier=43u;
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_ERR_FORMAT);
    result.completed_frontier=42u;result.final_logit=NAN;
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_ERR_FORMAT);
    result.has_output=false;result.final_token=FG_Q38_VOCAB_SIZE;result.final_logit=0.0f;
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_result_decode(
        &decoded_result,result_wire,sizeof(result_wire),&error)==FG_OK);
    PROTOCOL_CHECK(!decoded_result.has_output&&
        decoded_result.final_token==FG_Q38_VOCAB_SIZE&&decoded_result.final_logit==0.0f);
    memset(result_wire+12u,0,4u);
    PROTOCOL_CHECK(fg_pipeline_result_decode(
        &decoded_result,result_wire,sizeof(result_wire),&error)==FG_ERR_FORMAT);
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_OK);
    result_wire[16]=0x80u;
    PROTOCOL_CHECK(fg_pipeline_result_decode(
        &decoded_result,result_wire,sizeof(result_wire),&error)==FG_ERR_FORMAT);
    result.final_token=0u;
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_ERR_FORMAT);
    result.final_token=FG_Q38_VOCAB_SIZE;result.final_logit=-0.0f;
    PROTOCOL_CHECK(fg_pipeline_result_encode(result_wire,&result,&error)==FG_ERR_FORMAT);

    fg_pipeline_drain drain={.source_stage=3u,.destination_stage=4u},decoded_drain={0};
    fg_pipeline_drained drained={.source_stage=4u,.destination_stage=3u},
        decoded_drained={0};
    uint8_t drain_wire[FG_PIPELINE_DRAIN_BYTES];
    PROTOCOL_CHECK(fg_pipeline_drain_encode(drain_wire,&drain,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_drain_decode(
        &decoded_drain,drain_wire,sizeof(drain_wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_drained_encode(drain_wire,&drained,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_drained_decode(
        &decoded_drained,drain_wire,sizeof(drain_wire),&error)==FG_OK);
    drain_wire[2]=1u;
    PROTOCOL_CHECK(fg_pipeline_drained_decode(
        &decoded_drained,drain_wire,sizeof(drain_wire),&error)==FG_ERR_FORMAT);
    drain.source_stage=7u;drain.destination_stage=8u;
    PROTOCOL_CHECK(fg_pipeline_drain_encode(drain_wire,&drain,&error)==FG_ERR_FORMAT);
    drained.source_stage=0u;drained.destination_stage=UINT8_MAX;
    PROTOCOL_CHECK(fg_pipeline_drained_encode(
        drain_wire,&drained,&error)==FG_ERR_FORMAT);
    drained.source_stage=FG_PIPELINE_STAGE_COUNT;drained.destination_stage=7u;
    PROTOCOL_CHECK(fg_pipeline_drained_encode(
        drain_wire,&drained,&error)==FG_ERR_FORMAT);
    fg_pipeline_abort abort={.origin_stage=3u,.status=FG_ERR_IO,
        .failing_sequence=91u},decoded_abort={0};
    uint8_t abort_wire[FG_PIPELINE_ABORT_BYTES];
    PROTOCOL_CHECK(fg_pipeline_abort_encode(abort_wire,&abort,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_abort_decode(
        &decoded_abort,abort_wire,sizeof(abort_wire),&error)==FG_OK);
    PROTOCOL_CHECK(decoded_abort.origin_stage==3u&&
        decoded_abort.status==FG_ERR_IO&&decoded_abort.failing_sequence==91u);
    fg_frame_header abort_frame;
    PROTOCOL_CHECK(fg_frame_encode_version(
        &abort_frame,FG_PIPELINE_PROTOCOL_VERSION,FG_MSG_PIPELINE_ABORT,
        UINT64_C(0x55aa),91u,0u,abort_wire,sizeof(abort_wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_frame_validate_sequence(
        &abort_frame,FG_MSG_PIPELINE_ABORT,UINT64_C(0x55aa),91u,&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_encode_version(
        &abort_frame,FG_PROTOCOL_VERSION,FG_MSG_PIPELINE_ABORT,
        UINT64_C(0x55aa),91u,0u,abort_wire,sizeof(abort_wire),&error)==
        FG_ERR_ARGUMENT);
    abort_wire[1]=1u;
    PROTOCOL_CHECK(fg_pipeline_abort_decode(
        &decoded_abort,abort_wire,sizeof(abort_wire),&error)==FG_ERR_FORMAT);
    abort_wire[1]=0u;abort.status=FG_OK;
    PROTOCOL_CHECK(fg_pipeline_abort_encode(abort_wire,&abort,&error)==FG_ERR_FORMAT);

    activation.execution_kind=FG_PIPELINE_EXECUTION_PREFILL;
    PROTOCOL_CHECK(fg_pipeline_activation_encode(
        wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,&activation,&error)==FG_OK);
    fg_frame_header frame;
    uint64_t request=UINT64_C(0x1020304050607080);
    PROTOCOL_CHECK(fg_frame_encode_version(
        &frame,FG_PIPELINE_PROTOCOL_VERSION,FG_MSG_PIPELINE_ACTIVATION,
        request,77u,0u,wire,bytes,&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_validate_version(
        &frame,FG_PIPELINE_PROTOCOL_VERSION,wire,NULL,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_frame_validate_sequence(
        &frame,FG_MSG_PIPELINE_ACTIVATION,request,77u,&error)==FG_OK);
    PROTOCOL_CHECK(fg_pipeline_frame_validate_sequence(
        &frame,FG_MSG_PIPELINE_ACTIVATION,request,76u,&error)==FG_ERR_MISMATCH);
    PROTOCOL_CHECK(fg_pipeline_frame_validate_sequence(
        &frame,FG_MSG_PIPELINE_ACTIVATION,request+1u,77u,&error)==FG_ERR_MISMATCH);
    PROTOCOL_CHECK(fg_frame_encode_version(
        &frame,FG_PROTOCOL_VERSION,FG_MSG_PIPELINE_ACTIVATION,
        request,77u,0u,wire,bytes,&error)==FG_ERR_ARGUMENT);

    fg_layer_work *legacy=calloc(1,sizeof(*legacy));
    PROTOCOL_CHECK(legacy!=NULL);
    if(legacy){
        legacy->layer=2u;legacy->source_rank=1u;legacy->destination_rank=2u;
        legacy->position_mode=FG_POSITION_TEXT;
        legacy->token_index=UINT32_C(0x01020304);
        legacy->position[0]=5u;legacy->position[1]=6u;legacy->position[2]=7u;
        PROTOCOL_CHECK((fg_layer_work_encode)(
            wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,
            FG_PROTOCOL_MIN_VERSION,legacy,&error)==FG_OK);
        static const uint8_t v5_header[FG_LAYER_WORK_LEGACY_HEADER_BYTES]={
            2u,1u,2u,0u,1u,2u,3u,4u,0u,0u,0u,5u,0u,0u,0u,6u,0u,0u,0u,7u
        };
        PROTOCOL_CHECK(bytes==FG_LAYER_WORK_LEGACY_HEADER_BYTES+FG_HYPER_WIDTH*4u);
        PROTOCOL_CHECK(memcmp(wire,v5_header,sizeof(v5_header))==0);
        PROTOCOL_CHECK((fg_layer_work_encode)(
            wire,FG_PIPELINE_ACTIVATION_MAX_BYTES,&bytes,
            FG_PROTOCOL_VERSION,legacy,&error)==FG_OK);
        static const uint8_t v6_header[FG_LAYER_WORK_TEXT_HEADER_BYTES]={
            2u,1u,2u,0u,1u,2u,3u,4u,0u,3u,0u,0u,
            0u,0u,0u,5u,0u,0u,0u,6u,0u,0u,0u,7u
        };
        PROTOCOL_CHECK(bytes==FG_LAYER_WORK_BASE_BYTES);
        PROTOCOL_CHECK(memcmp(wire,v6_header,sizeof(v6_header))==0);
        free(legacy);
    }
    free(wire);free(decoded_boundary);free(boundary);
}

static fg_status batch_send_roundtrip(fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    uint16_t protocol=fg_fabric_protocol_version(fabric);
    uint16_t other=protocol==FG_PROTOCOL_MIN_VERSION?FG_PROTOCOL_VERSION:FG_PROTOCOL_MIN_VERSION;
    fg_frame_header probe;
    fg_status probe_status=fg_frame_encode_version(&probe,protocol,FG_MSG_CONTROL,request,
                                                   199u,0u,NULL,0u,error);
    if(probe_status==FG_OK)
        probe_status=fg_fabric_validate_frame(fabric,&probe,NULL,NULL,error);
    if(probe_status!=FG_OK)return probe_status;
    probe_status=fg_frame_encode_version(&probe,other,FG_MSG_CONTROL,request,199u,0u,
                                         NULL,0u,error);
    if(probe_status!=FG_OK)return probe_status;
    if(fg_fabric_validate_frame(fabric,&probe,NULL,NULL,error)!=FG_ERR_MISMATCH){
        fg_error_set(error,FG_ERR_MISMATCH,"fabric accepted a cross-version frame");
        return FG_ERR_MISMATCH;
    }
    memset(error,0,sizeof(*error));
    enum{PAYLOAD_BYTES=16};uint8_t payloads[FG_RANK_COUNT-1u][PAYLOAD_BYTES];for(uint32_t peer=1u;peer<FG_RANK_COUNT;peer++)for(uint32_t i=0;i<PAYLOAD_BYTES;i++)payloads[peer-1u][i]=(uint8_t)(peer*29u+i*7u);if(rank==0u){fg_fabric_send_item items[FG_RANK_COUNT-1u];for(uint32_t peer=1u;peer<FG_RANK_COUNT;peer++)items[peer-1u]=(fg_fabric_send_item){.peer=peer,.cls=FG_FABRIC_BULK,.type=FG_MSG_CONTROL,.request_id=request,.sequence=200u+peer,.payload=payloads[peer-1u],.bytes=PAYLOAD_BYTES};return fg_fabric_send_batch(fabric,items,FG_RANK_COUNT-1u,error);}fg_frame_header header;uint8_t payload[PAYLOAD_BYTES];uint32_t bytes=0;fg_status status=fg_fabric_recv(fabric,0u,FG_FABRIC_BULK,&header,payload,sizeof(payload),&bytes,error);if(status==FG_OK&&(bytes!=PAYLOAD_BYTES||fg_frame_type(&header)!=FG_MSG_CONTROL||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=200u+rank||memcmp(payload,payloads[rank-1u],PAYLOAD_BYTES)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"batch send payload mismatch on rank %u",rank);status=FG_ERR_MISMATCH;}return status;
}

static fg_status prefill_bulk_roundtrip(fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    enum{TOKENS=128,PAIRS=2};const uint32_t work_capacity=FG_PREFILL_WORK_HEADER_BYTES+TOKENS*FG_Q8K_ACTIVATION_BYTES+PAIRS*FG_PREFILL_PAIR_BYTES,result_capacity=FG_PREFILL_RESULT_HEADER_BYTES+PAIRS*FG_PREFILL_RESULT_PAIR_BYTES;
    if(rank>1u)return FG_OK;
    uint8_t *work_wire=malloc(work_capacity),*result_wire=malloc(result_capacity),*activations=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES),*decoded_activations=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES);fg_prefill_pair pairs[PAIRS]={{0,7,0,0.75f},{127,9,4,0.25f}},decoded_pairs[PAIRS];fg_prefill_result_pair result_pairs[PAIRS]={{0,0},{127,4}},decoded_result_pairs[PAIRS];float *outputs=malloc((size_t)PAIRS*FG_HIDDEN_SIZE*4u),*decoded_outputs=malloc((size_t)PAIRS*FG_HIDDEN_SIZE*4u);fg_status status=work_wire&&result_wire&&activations&&decoded_activations&&outputs&&decoded_outputs?FG_OK:FG_ERR_OOM;if(status!=FG_OK)fg_error_set(error,status,"allocate fabric prefill roundtrip buffers");
    for(uint32_t i=0;status==FG_OK&&i<TOKENS*FG_Q8K_ACTIVATION_BYTES;i++)activations[i]=(uint8_t)(i*13u+5u);
    for(uint32_t i=0;status==FG_OK&&i<PAIRS*FG_HIDDEN_SIZE;i++)outputs[i]=(float)i*0.0001f;
    if(status==FG_OK&&rank==0u){fg_prefill_work work={.layer=0,.source_rank=0,.destination_rank=1,.first_position=8192,.token_count=TOKENS,.pair_count=PAIRS,.activations_q8k=activations,.pairs=pairs};uint32_t bytes=0;status=fg_prefill_work_encode(work_wire,work_capacity,&bytes,&work,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_BULK,FG_MSG_PREFILL_WORK,request,99u,0,work_wire,bytes,error);fg_frame_header header;if(status==FG_OK)status=fg_fabric_recv(fabric,1u,FG_FABRIC_BULK,&header,result_wire,result_capacity,&bytes,error);fg_prefill_result decoded={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_RESULT||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=99u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid prefill result frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_result_decode(&decoded,decoded_result_pairs,PAIRS,decoded_outputs,(uint64_t)PAIRS*FG_HIDDEN_SIZE,result_wire,bytes,error);if(status==FG_OK&&(decoded.first_position!=8192u||decoded.token_count!=TOKENS||decoded.pair_count!=PAIRS||decoded_result_pairs[1].token_slot!=127u||decoded_outputs[PAIRS*FG_HIDDEN_SIZE-1u]!=outputs[PAIRS*FG_HIDDEN_SIZE-1u])){fg_error_set(error,FG_ERR_MISMATCH,"prefill result payload mismatch");status=FG_ERR_MISMATCH;}}
    if(status==FG_OK&&rank==1u){fg_frame_header header;uint32_t bytes=0;status=fg_fabric_recv(fabric,0u,FG_FABRIC_BULK,&header,work_wire,work_capacity,&bytes,error);fg_prefill_work decoded={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=99u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid prefill work frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_work_decode(&decoded,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES,decoded_pairs,PAIRS,work_wire,bytes,error);if(status==FG_OK&&(decoded.first_position!=8192u||decoded.token_count!=TOKENS||decoded.pair_count!=PAIRS||memcmp(activations,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"prefill work payload mismatch");status=FG_ERR_MISMATCH;}fg_prefill_result result={.layer=0,.source_rank=1,.destination_rank=0,.first_position=8192,.token_count=TOKENS,.pair_count=PAIRS,.pairs=result_pairs,.outputs=outputs};if(status==FG_OK)status=fg_prefill_result_encode(result_wire,result_capacity,&bytes,&result,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_PREFILL_RESULT,request,99u,0,result_wire,bytes,error);}
    free(decoded_outputs);free(outputs);free(decoded_activations);free(activations);free(result_wire);free(work_wire);return status;
}

static fg_status delayed_bulk_ngram_control_roundtrip(
    fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    if(rank>1u)return FG_OK;
    enum{TOKEN_INDEX=333u};
    fg_status status=FG_OK;
    uint8_t credit_wire[FG_PIPELINE_CREDIT_BYTES];
    uint8_t *result_wire=malloc(FG_NGRAM_RESULT_MAX_BYTES);
    if(!result_wire){
        fg_error_set(error,FG_ERR_OOM,"allocate resident n-gram result wire");
        return FG_ERR_OOM;
    }
    if(rank==1u){
        fg_pipeline_credit credit={.source_stage=1u,.destination_stage=0u,
                                   .slot=0u};
        status=fg_pipeline_credit_encode(credit_wire,&credit,error);
        if(status==FG_OK)status=fg_fabric_send(
            fabric,0u,FG_FABRIC_BULK,FG_MSG_PIPELINE_CREDIT,request,
            TOKEN_INDEX,0u,credit_wire,sizeof(credit_wire),error);
        fg_ngram_result result={.source_rank=1u,.destination_rank=0u,
            .item_count=1u,.token_index=TOKEN_INDEX};
        result.heads[0]=0u;
        uint32_t bytes=0u;
        if(status==FG_OK)status=fg_ngram_result_encode(
            result_wire,FG_NGRAM_RESULT_MAX_BYTES,&bytes,&result,error);
        if(status==FG_OK)status=fg_fabric_send(
            fabric,0u,FG_FABRIC_CONTROL,FG_MSG_NGRAM_RESULT,request,
            TOKEN_INDEX,0u,result_wire,bytes,error);
    }else{
        uint32_t peer=0u,bytes=0u;
        fg_frame_header header;
        status=fg_fabric_recv_any(fabric,FG_FABRIC_CONTROL,&peer,&header,
                                  result_wire,FG_NGRAM_RESULT_MAX_BYTES,
                                  &bytes,error);
        fg_ngram_result result={0};
        if(status==FG_OK&&(peer!=1u||
           fg_frame_type(&header)!=FG_MSG_NGRAM_RESULT||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "invalid CONTROL resident n-gram result");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_ngram_result_decode(
            &result,result_wire,bytes,error);
        if(status==FG_OK&&(result.source_rank!=1u||
           result.destination_rank!=0u||result.item_count!=1u||
           result.token_index!=TOKEN_INDEX||result.heads[0]!=0u)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "misrouted CONTROL resident n-gram result");
            status=FG_ERR_MISMATCH;
        }
        fg_fabric_class ready_class=FG_FABRIC_CONTROL;
        if(status==FG_OK)status=fg_fabric_wait_ready(
            fabric,1u<<FG_FABRIC_BULK,&peer,&ready_class,error);
        if(status==FG_OK&&(peer!=1u||ready_class!=FG_FABRIC_BULK)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "delayed pipeline credit left BULK channel");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_fabric_recv(
            fabric,1u,FG_FABRIC_BULK,&header,credit_wire,
            sizeof(credit_wire),&bytes,error);
        fg_pipeline_credit credit={0};
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PIPELINE_CREDIT||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "invalid delayed pipeline credit");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_pipeline_credit_decode(
            &credit,credit_wire,bytes,error);
        if(status==FG_OK&&(credit.source_stage!=1u||
           credit.destination_stage!=0u||credit.slot!=0u)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "delayed pipeline credit payload changed");
            status=FG_ERR_MISMATCH;
        }
    }
    free(result_wire);
    return status;
}

static fg_status prefill_layer_chain_roundtrip(fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    enum{TOKENS=8};uint32_t work_capacity=FG_PREFILL_LAYER_HEADER_BYTES+TOKENS*3u*4u+TOKENS*FG_HYPER_WIDTH*4u+TOKENS*FG_NGRAM_EMBED_VALUES*4u,result_capacity=FG_PREFILL_LAYER_HEADER_BYTES+TOKENS*FG_HYPER_WIDTH*4u;uint8_t *wire=malloc(work_capacity),*result_wire=malloc(result_capacity);uint32_t positions[TOKENS*3u],decoded_positions[TOKENS*3u];float *hyper=malloc((size_t)TOKENS*FG_HYPER_WIDTH*4u),*decoded_hyper=malloc((size_t)TOKENS*FG_HYPER_WIDTH*4u),*ngram=malloc((size_t)TOKENS*FG_NGRAM_EMBED_VALUES*4u),*decoded_ngram=malloc((size_t)TOKENS*FG_NGRAM_EMBED_VALUES*4u);fg_status status=wire&&result_wire&&hyper&&decoded_hyper&&ngram&&decoded_ngram?FG_OK:FG_ERR_OOM;if(status!=FG_OK)fg_error_set(error,status,"allocate direct prefill layer chain buffers");for(uint32_t i=0;status==FG_OK&&i<TOKENS*3u;i++)positions[i]=12000u+i;for(uint32_t i=0;status==FG_OK&&i<TOKENS*FG_HYPER_WIDTH;i++)hyper[i]=(float)i*0.00001f;for(uint32_t i=0;status==FG_OK&&i<TOKENS*FG_NGRAM_EMBED_VALUES;i++)ngram[i]=-(float)i*0.00002f;
    if(status==FG_OK&&rank==0u){fg_prefill_layer_work work={.layer=1u,.source_rank=0u,.destination_rank=1u,.flags=FG_LAYER_WORK_HAS_NGRAM,.first_token=12000u,.token_count=TOKENS,.positions=positions,.hyper=hyper,.ngram_embeddings=ngram};uint32_t bytes=0;status=fg_prefill_layer_work_encode(wire,work_capacity,&bytes,&work,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_BULK,FG_MSG_PREFILL_LAYER_WORK,request,101u,0,wire,bytes,error);fg_frame_header header;if(status==FG_OK)status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,result_wire,result_capacity,&bytes,error);fg_prefill_layer_result result={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_LAYER_RESULT||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=108u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct prefill layer frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_layer_result_decode(&result,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,result_wire,bytes,error);if(status==FG_OK&&(result.layer!=7u||result.source_rank!=7u||result.destination_rank!=0u||result.first_token!=12000u||result.token_count!=TOKENS||memcmp(result.hyper,hyper,(size_t)TOKENS*FG_HYPER_WIDTH*4u)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct prefill layer payload");status=FG_ERR_MISMATCH;}}
    if(status==FG_OK&&rank>0u){uint32_t peer=rank-1u,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,peer,FG_FABRIC_BULK,&header,wire,work_capacity,&bytes,error);fg_prefill_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=100u+rank)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct prefill layer hop at rank %u",rank);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_layer_work_decode(&work,decoded_positions,TOKENS*3u,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,decoded_ngram,(uint64_t)TOKENS*FG_NGRAM_EMBED_VALUES,wire,bytes,error);if(status==FG_OK&&(work.layer!=rank||work.source_rank!=peer||work.destination_rank!=rank||work.first_token!=12000u||work.token_count!=TOKENS||memcmp(decoded_positions,positions,sizeof(positions))!=0||memcmp(decoded_hyper,hyper,(size_t)TOKENS*FG_HYPER_WIDTH*4u)!=0||(rank==1u&&memcmp(decoded_ngram,ngram,(size_t)TOKENS*FG_NGRAM_EMBED_VALUES*4u)!=0))){fg_error_set(error,FG_ERR_MISMATCH,"direct prefill layer payload mismatch at rank %u",rank);status=FG_ERR_MISMATCH;}if(status==FG_OK&&rank<7u){fg_prefill_layer_work next={.layer=(uint8_t)(rank+1u),.source_rank=(uint8_t)rank,.destination_rank=(uint8_t)(rank+1u),.first_token=12000u,.token_count=TOKENS,.positions=decoded_positions,.hyper=decoded_hyper};status=fg_prefill_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,rank+1u,FG_FABRIC_BULK,FG_MSG_PREFILL_LAYER_WORK,request,101u+rank,0,wire,bytes,error);}else if(status==FG_OK){fg_prefill_layer_result result={.layer=7u,.source_rank=7u,.destination_rank=0u,.first_token=12000u,.token_count=TOKENS,.hyper=decoded_hyper};status=fg_prefill_layer_result_encode(result_wire,result_capacity,&bytes,&result,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_PREFILL_LAYER_RESULT,request,108u,0,result_wire,bytes,error);}}
    free(decoded_ngram);free(ngram);free(decoded_hyper);free(hyper);free(result_wire);free(wire);return status;
}

static fg_status decode_layer_chain_roundtrip(fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    enum{TOKEN=77,CYCLES=FG_LAYER_COUNT/FG_RANK_COUNT};uint32_t work_capacity=FG_LAYER_WORK_MAX_BYTES,result_capacity=FG_LAYER_RESULT_BYTES;uint8_t *wire=malloc(work_capacity),*result_wire=malloc(result_capacity);uint32_t position[3];float *hyper=malloc((size_t)FG_HYPER_WIDTH*4u),*ngram=malloc((size_t)FG_NGRAM_EMBED_VALUES*4u);fg_status status=wire&&result_wire&&hyper&&ngram?FG_OK:FG_ERR_OOM;if(status!=FG_OK)fg_error_set(error,status,"allocate direct decode layer chain buffers");for(uint32_t i=0;status==FG_OK&&i<3u;i++)position[i]=TOKEN;for(uint32_t i=0;status==FG_OK&&i<FG_HYPER_WIDTH;i++)hyper[i]=(float)(i+1u)*0.00001f;for(uint32_t i=0;status==FG_OK&&i<FG_NGRAM_EMBED_VALUES;i++)ngram[i]=-(float)(i+1u)*0.00002f;
    for(uint32_t cycle=0;status==FG_OK&&cycle<CYCLES;cycle++){
        uint32_t base=cycle*FG_RANK_COUNT;
        if(rank==0u){
            if(cycle==0u){fg_layer_work first={.layer=1u,.source_rank=0u,.destination_rank=1u,.flags=FG_LAYER_WORK_HAS_NGRAM,.token_index=TOKEN};memcpy(first.position,position,sizeof(position));memcpy(first.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u);memcpy(first.ngram_embedding,ngram,(size_t)FG_NGRAM_EMBED_VALUES*4u);uint32_t bytes=0;status=fg_layer_work_encode(wire,work_capacity,&bytes,&first,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+1u,0,wire,bytes,error);}
            else{uint32_t boundary=base,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,7u,FG_FABRIC_CONTROL,&header,wire,work_capacity,&bytes,error);fg_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+boundary)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct decode boundary frame at layer %u",boundary);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_work_decode(&work,wire,bytes,error);if(status==FG_OK&&(work.layer!=boundary||work.source_rank!=7u||work.destination_rank!=0u||work.token_index!=TOKEN||memcmp(work.position,position,sizeof(position))!=0||memcmp(work.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0||work.flags!=0u)){fg_error_set(error,FG_ERR_MISMATCH,"direct decode boundary payload mismatch at layer %u",boundary);status=FG_ERR_MISMATCH;}if(status==FG_OK){fg_layer_work next={.layer=(uint8_t)(boundary+1u),.source_rank=0u,.destination_rank=1u,.token_index=TOKEN};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+boundary+1u,0,wire,bytes,error);}}
        }else{uint32_t layer=base+rank,peer=rank-1u,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,peer,FG_FABRIC_CONTROL,&header,wire,work_capacity,&bytes,error);fg_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+layer)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct decode layer hop at rank %u",rank);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_work_decode(&work,wire,bytes,error);if(status==FG_OK&&(work.layer!=layer||work.source_rank!=peer||work.destination_rank!=rank||work.token_index!=TOKEN||memcmp(work.position,position,sizeof(position))!=0||memcmp(work.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0||work.flags!=(layer==1u?FG_LAYER_WORK_HAS_NGRAM:0u)||(layer==1u&&memcmp(work.ngram_embedding,ngram,(size_t)FG_NGRAM_EMBED_VALUES*4u)!=0))){fg_error_set(error,FG_ERR_MISMATCH,"direct decode layer payload mismatch at rank %u layer %u",rank,layer);status=FG_ERR_MISMATCH;}if(status==FG_OK){if(rank<7u){fg_layer_work next={.layer=(uint8_t)(layer+1u),.source_rank=(uint8_t)rank,.destination_rank=(uint8_t)(rank+1u),.token_index=TOKEN};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,rank+1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+layer+1u,0,wire,bytes,error);}else if(cycle+1u<CYCLES){fg_layer_work next={.layer=(uint8_t)(layer+1u),.source_rank=7u,.destination_rank=0u,.token_index=TOKEN};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+layer+1u,0,wire,bytes,error);}else{fg_layer_result result={.layer=(uint8_t)layer,.source_rank=7u,.destination_rank=0u,.token_index=TOKEN};memcpy(result.hyper,work.hyper,sizeof(result.hyper));status=fg_layer_result_encode(result_wire,&result,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_LAYER_RESULT,request,TOKEN*FG_LAYER_COUNT+layer,0,result_wire,result_capacity,error);}}}
    }
    if(status==FG_OK&&rank==0u){uint32_t bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,result_wire,result_capacity,&bytes,error);fg_layer_result result={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_RESULT||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct decode layer frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_result_decode(&result,result_wire,bytes,error);if(status==FG_OK&&(result.layer!=FG_LAYER_COUNT-1u||result.source_rank!=7u||result.destination_rank!=0u||result.token_index!=TOKEN||memcmp(result.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct decode layer payload");status=FG_ERR_MISMATCH;}}
    free(ngram);free(hyper);free(result_wire);free(wire);return status;
}

static int child_main(const fg_manifest *manifest,uint32_t rank){fg_error error={0};fg_fabric *fabric=NULL;fg_status status=fg_fabric_open(&fabric,manifest,rank,&error);if(status==FG_ERR_UNAVAILABLE)return 77;if(status!=FG_OK){fprintf(stderr,"rank %u fabric open: %s\n",rank,error.message);return 1;}uint64_t request=UINT64_C(0x1122334455667788);for(uint32_t peer=0;status==FG_OK&&peer<FG_RANK_COUNT;peer++)if(peer!=rank)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_READY,request,rank,0,NULL,0,&error);bool seen[FG_RANK_COUNT]={0};for(uint32_t received=0;status==FG_OK&&received<FG_RANK_COUNT-1u;received++){uint32_t peer=0,bytes=0;fg_frame_header header;status=fg_fabric_recv_any(fabric,FG_FABRIC_CONTROL,&peer,&header,NULL,0,&bytes,&error);if(status==FG_OK&&(peer==rank||seen[peer]||bytes||fg_frame_type(&header)!=FG_MSG_READY||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=peer)){fprintf(stderr,"rank %u invalid READY from %u\n",rank,peer);status=FG_ERR_MISMATCH;}seen[peer]=true;}if(status==FG_OK)status=batch_send_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=prefill_bulk_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=delayed_bulk_ngram_control_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=prefill_layer_chain_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=decode_layer_chain_roundtrip(fabric,rank,request,&error);fg_fabric_close(fabric);if(status!=FG_OK){fprintf(stderr,"rank %u fabric exchange: %s\n",rank,error.message);return 1;}return 0;}

int main(void){pipeline_protocol_test();if(protocol_failures)return 1;fg_manifest manifest;fg_manifest_init(&manifest);manifest.protocol_version=FG_PIPELINE_PROTOCOL_VERSION;uint32_t base=24000u+(uint32_t)(getpid()%5000u)*2u;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)snprintf(manifest.ranks[rank].endpoint,sizeof(manifest.ranks[rank].endpoint),"127.0.0.1:%u",base+rank*2u);for(uint32_t i=0;i<32u;i++)manifest.manifest_sha256[i]=(uint8_t)(i*7u+3u);pid_t children[FG_RANK_COUNT];for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){children[rank]=fork();if(children[rank]<0){perror("fork");return 1;}if(children[rank]==0)_exit(child_main(&manifest,rank));}uint32_t passed=0,skipped=0;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){int status;if(waitpid(children[rank],&status,0)<0){perror("waitpid");return 1;}if(WIFEXITED(status)&&WEXITSTATUS(status)==0)passed++;else if(WIFEXITED(status)&&WEXITSTATUS(status)==77)skipped++;else{fprintf(stderr,"fabric rank %u exited abnormally\n",rank);return 1;}}if(skipped==FG_RANK_COUNT){fprintf(stderr,"SKIP eight-process fabric: io_uring unavailable\n");return 77;}if(passed!=FG_RANK_COUNT||skipped){fprintf(stderr,"inconsistent fabric qualification: %u pass %u skip\n",passed,skipped);return 1;}puts("Flash Gordon protocol 7 pipeline wire and eight-process dual-channel mesh: PASS");return 0;}
