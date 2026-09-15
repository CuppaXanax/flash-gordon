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
    enum{TOKENS=128,PAIRS=2};const uint32_t work_capacity=FG_PREFILL_WORK_HEADER_BYTES+TOKENS*FG_Q8K_ACTIVATION_BYTES+PAIRS*FG_PREFILL_PAIR_BYTES,result_capacity=FG_PREFILL_RESULT_HEADER_BYTES+PAIRS*FG_PREFILL_RESULT_PAIR_BYTES+TOKENS*FG_HIDDEN_SIZE*4u;
    if(rank>1u)return FG_OK;
    uint8_t *work_wire=malloc(work_capacity),*result_wire=malloc(result_capacity),*activations=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES),*decoded_activations=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES);fg_prefill_pair pairs[PAIRS]={{0,7,0,0.75f},{127,9,4,0.25f}},decoded_pairs[PAIRS];fg_prefill_result_pair result_pairs[PAIRS]={{0,0},{127,4}},decoded_result_pairs[PAIRS];float *outputs=malloc((size_t)TOKENS*FG_HIDDEN_SIZE*4u),*decoded_outputs=malloc((size_t)TOKENS*FG_HIDDEN_SIZE*4u);fg_status status=work_wire&&result_wire&&activations&&decoded_activations&&outputs&&decoded_outputs?FG_OK:FG_ERR_OOM;if(status!=FG_OK)fg_error_set(error,status,"allocate fabric prefill roundtrip buffers");
    for(uint32_t i=0;status==FG_OK&&i<TOKENS*FG_Q8K_ACTIVATION_BYTES;i++)activations[i]=(uint8_t)(i*13u+5u);
    for(uint32_t i=0;status==FG_OK&&i<TOKENS*FG_HIDDEN_SIZE;i++)outputs[i]=(i/FG_HIDDEN_SIZE==0u||i/FG_HIDDEN_SIZE==127u)?(float)i*0.0001f:0.0f;
    if(status==FG_OK&&rank==0u){fg_prefill_work work={.layer=0,.source_rank=0,.destination_rank=1,.first_position=8192,.token_count=TOKENS,.pair_count=PAIRS,.activations_q8k=activations,.pairs=pairs};uint32_t bytes=0;status=fg_prefill_work_encode(work_wire,work_capacity,&bytes,&work,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_BULK,FG_MSG_PREFILL_WORK,request,99u,0,work_wire,bytes,error);fg_frame_header header;if(status==FG_OK)status=fg_fabric_recv(fabric,1u,FG_FABRIC_BULK,&header,result_wire,result_capacity,&bytes,error);fg_prefill_result decoded={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_RESULT||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=99u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid prefill result frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_result_decode(&decoded,decoded_result_pairs,PAIRS,decoded_outputs,(uint64_t)TOKENS*FG_HIDDEN_SIZE,result_wire,bytes,error);if(status==FG_OK&&(decoded.first_position!=8192u||decoded.token_count!=TOKENS||decoded.pair_count!=PAIRS||decoded_result_pairs[1].token_slot!=127u||decoded_outputs[TOKENS*FG_HIDDEN_SIZE-1u]!=outputs[TOKENS*FG_HIDDEN_SIZE-1u])){fg_error_set(error,FG_ERR_MISMATCH,"prefill result payload mismatch");status=FG_ERR_MISMATCH;}}
    if(status==FG_OK&&rank==1u){fg_frame_header header;uint32_t bytes=0;status=fg_fabric_recv(fabric,0u,FG_FABRIC_BULK,&header,work_wire,work_capacity,&bytes,error);fg_prefill_work decoded={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_PREFILL_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=99u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid prefill work frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_prefill_work_decode(&decoded,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES,decoded_pairs,PAIRS,work_wire,bytes,error);if(status==FG_OK&&(decoded.first_position!=8192u||decoded.token_count!=TOKENS||decoded.pair_count!=PAIRS||memcmp(activations,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"prefill work payload mismatch");status=FG_ERR_MISMATCH;}fg_prefill_result result={.layer=0,.source_rank=1,.destination_rank=0,.first_position=8192,.token_count=TOKENS,.pair_count=PAIRS,.pairs=result_pairs,.outputs=outputs,.contributor_mask=1u<<1u};if(status==FG_OK)status=fg_prefill_result_encode(result_wire,result_capacity,&bytes,&result,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_PREFILL_RESULT,request,99u,0,result_wire,bytes,error);}
    free(decoded_outputs);free(outputs);free(decoded_activations);free(activations);free(result_wire);free(work_wire);return status;
}

static fg_status delayed_bulk_ngram_control_roundtrip(
    fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    if(rank>1u)return FG_OK;
    enum{TOKEN_INDEX=333u};
    fg_status status=FG_OK;
    uint8_t credit_wire[1]={42u};
    uint8_t *result_wire=malloc(FG_NGRAM_RESULT_MAX_BYTES);
    if(!result_wire){
        fg_error_set(error,FG_ERR_OOM,"allocate resident n-gram result wire");
        return FG_ERR_OOM;
    }
    if(rank==1u){
        if(status==FG_OK)status=fg_fabric_send(
            fabric,0u,FG_FABRIC_BULK,FG_MSG_CONTROL,request,
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
                         "delayed bulk control left BULK channel");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_fabric_recv(
            fabric,1u,FG_FABRIC_BULK,&header,credit_wire,
            sizeof(credit_wire),&bytes,error);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_CONTROL||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "invalid delayed bulk control");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK&&(bytes!=1u||credit_wire[0]!=42u)){
            fg_error_set(error,FG_ERR_MISMATCH,
                         "delayed bulk control payload changed");
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
            if(cycle==0u){fg_layer_work first={.layer=1u,.source_rank=0u,.destination_rank=1u,.flags=FG_LAYER_WORK_HAS_NGRAM|FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY,.token_index=TOKEN};memcpy(first.position,position,sizeof(position));memcpy(first.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u);memcpy(first.ngram_embedding,ngram,(size_t)FG_NGRAM_EMBED_VALUES*4u);uint32_t bytes=0;status=fg_layer_work_encode(wire,work_capacity,&bytes,&first,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+1u,0,wire,bytes,error);}
            else{uint32_t boundary=base,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,7u,FG_FABRIC_CONTROL,&header,wire,work_capacity,&bytes,error);fg_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+boundary)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct decode boundary frame at layer %u",boundary);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_work_decode(&work,wire,bytes,error);if(status==FG_OK&&(work.layer!=boundary||work.source_rank!=7u||work.destination_rank!=0u||work.token_index!=TOKEN||memcmp(work.position,position,sizeof(position))!=0||memcmp(work.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0||work.flags!=FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)){fg_error_set(error,FG_ERR_MISMATCH,"direct decode boundary payload mismatch at layer %u",boundary);status=FG_ERR_MISMATCH;}if(status==FG_OK){fg_layer_work next={.layer=(uint8_t)(boundary+1u),.source_rank=0u,.destination_rank=1u,.token_index=TOKEN,.flags=(uint8_t)(work.flags&FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+boundary+1u,0,wire,bytes,error);}}
        }else{uint32_t layer=base+rank,peer=rank-1u,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,peer,FG_FABRIC_CONTROL,&header,wire,work_capacity,&bytes,error);fg_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+layer)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct decode layer hop at rank %u",rank);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_work_decode(&work,wire,bytes,error);if(status==FG_OK&&(work.layer!=layer||work.source_rank!=peer||work.destination_rank!=rank||work.token_index!=TOKEN||memcmp(work.position,position,sizeof(position))!=0||memcmp(work.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0||work.flags!=((layer==1u?FG_LAYER_WORK_HAS_NGRAM:0u)|FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)||(layer==1u&&memcmp(work.ngram_embedding,ngram,(size_t)FG_NGRAM_EMBED_VALUES*4u)!=0))){fg_error_set(error,FG_ERR_MISMATCH,"direct decode layer payload mismatch at rank %u layer %u",rank,layer);status=FG_ERR_MISMATCH;}if(status==FG_OK){if(rank<7u){fg_layer_work next={.layer=(uint8_t)(layer+1u),.source_rank=(uint8_t)rank,.destination_rank=(uint8_t)(rank+1u),.token_index=TOKEN,.flags=(uint8_t)(work.flags&FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,rank+1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+layer+1u,0,wire,bytes,error);}else if(cycle+1u<CYCLES){fg_layer_work next={.layer=(uint8_t)(layer+1u),.source_rank=7u,.destination_rank=0u,.token_index=TOKEN,.flags=(uint8_t)(work.flags&FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY)};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+layer+1u,0,wire,bytes,error);}else{fg_layer_result result={.layer=(uint8_t)layer,.source_rank=7u,.destination_rank=0u,.token_index=TOKEN};memcpy(result.hyper,work.hyper,sizeof(result.hyper));uint32_t result_bytes=0u;status=fg_layer_result_encode(result_wire,&result,&result_bytes,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_LAYER_RESULT,request,TOKEN*FG_LAYER_COUNT+layer,0,result_wire,result_bytes,error);}}}
    }
    if(status==FG_OK&&rank==0u){uint32_t bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,result_wire,result_capacity,&bytes,error);fg_layer_result result={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_RESULT||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct decode layer frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_result_decode(&result,result_wire,bytes,error);if(status==FG_OK&&(result.layer!=FG_LAYER_COUNT-1u||result.source_rank!=7u||result.destination_rank!=0u||result.token_index!=TOKEN||memcmp(result.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct decode layer payload");status=FG_ERR_MISMATCH;}}
    free(ngram);free(hyper);free(result_wire);free(wire);return status;
}

static void protocol_output_handoff_selfcheck(void){
    fg_error error={0};
    fg_error missing={0};
    PROTOCOL_CHECK(fg_output_split_require_slice(true,4u,1u,&missing)==FG_OK);
    PROTOCOL_CHECK(fg_output_split_require_slice(false,4u,1u,&missing)!=FG_OK);
    PROTOCOL_CHECK(strstr(missing.message,"slice executor")!=NULL);
    PROTOCOL_CHECK(strstr(missing.message,"rank 1")!=NULL);
    PROTOCOL_CHECK(fg_output_split_require_slice(false,2u,0u,&missing)!=FG_OK);
    fg_output_config config={.source_rank=0u,.destination_rank=4u,.token_index=17u,
        .uniform=0.25f};
    uint8_t wire[FG_OUTPUT_CONFIG_BYTES];
    PROTOCOL_CHECK(fg_output_config_encode(wire,&config,&error)==FG_OK);
    fg_output_config decoded={0};
    PROTOCOL_CHECK(fg_output_config_decode(&decoded,wire,sizeof(wire),&error)==FG_OK);
    PROTOCOL_CHECK(decoded.source_rank==0u&&decoded.destination_rank==4u&&
        decoded.token_index==17u&&decoded.uniform==0.25f&&decoded.flags==0u);
    fg_output_config split=config;split.flags=FG_OUTPUT_CONFIG_FLAG_SPLIT;
    PROTOCOL_CHECK(fg_output_config_encode(wire,&split,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_config_decode(&decoded,wire,sizeof(wire),&error)==FG_OK&&
        decoded.flags==FG_OUTPUT_CONFIG_FLAG_SPLIT);
    fg_output_config split4=config;
    split4.flags=FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4;
    PROTOCOL_CHECK(fg_output_config_encode(wire,&split4,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_config_decode(&decoded,wire,sizeof(wire),&error)==FG_OK&&
        decoded.flags==(FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4));
    uint8_t bad[FG_OUTPUT_CONFIG_BYTES];
    memcpy(bad,wire,sizeof(bad));bad[2]=FG_OUTPUT_CONFIG_FLAG_SPLIT_4;
    PROTOCOL_CHECK(fg_output_config_decode(&decoded,bad,sizeof(bad),&error)!=FG_OK);
    memcpy(bad,wire,sizeof(bad));bad[2]=0x80u;
    PROTOCOL_CHECK(fg_output_config_decode(&decoded,bad,sizeof(bad),&error)!=FG_OK);
    memcpy(bad,wire,sizeof(bad));bad[3]=1u;
    PROTOCOL_CHECK(fg_output_config_decode(&decoded,bad,sizeof(bad),&error)!=FG_OK);
    fg_output_config wrong=config;wrong.destination_rank=5u;
    PROTOCOL_CHECK(fg_output_config_encode(wire,&wrong,&error)!=FG_OK);
    PROTOCOL_CHECK(fg_output_config_encode(wire,&config,&error)==FG_OK);
    fg_output_partial partial={.token_index=17u,.value=-1.25f,.id=12345u};
    uint8_t partial_wire[FG_OUTPUT_PARTIAL_BYTES];
    PROTOCOL_CHECK(fg_output_partial_encode(partial_wire,&partial,&error)==FG_OK);
    fg_output_partial decoded_partial={0};
    PROTOCOL_CHECK(fg_output_partial_decode(&decoded_partial,partial_wire,
        sizeof(partial_wire),&error)==FG_OK&&decoded_partial.token_index==17u&&
        decoded_partial.value==-1.25f&&decoded_partial.id==12345u);
    PROTOCOL_CHECK(fg_output_partial_decode(&decoded_partial,partial_wire,
        sizeof(partial_wire)-1u,&error)!=FG_OK);
    fg_output_partial bad_partial=partial;bad_partial.id=FG_Q38_VOCAB_SIZE;
    PROTOCOL_CHECK(fg_output_partial_encode(partial_wire,&bad_partial,&error)!=FG_OK);
    fg_output_slice_hidden head_slice={.source_rank=7u,.destination_rank=1u,
        .token_index=17u};
    for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)head_slice.hidden[i]=(float)(i%23u)*0.125f-1.0f;
    uint8_t head_wire[FG_OUTPUT_SLICE_HIDDEN_BYTES];uint32_t head_bytes=0u;
    PROTOCOL_CHECK(fg_output_slice_hidden_encode(head_wire,&head_slice,&head_bytes,&error)==FG_OK);
    fg_output_slice_hidden decoded_head={0};
    PROTOCOL_CHECK(fg_output_slice_hidden_decode(&decoded_head,head_wire,
        sizeof(head_wire),&error)==FG_OK&&decoded_head.source_rank==7u&&
        decoded_head.destination_rank==1u&&decoded_head.token_index==17u&&
        memcmp(decoded_head.hidden,head_slice.hidden,sizeof(head_slice.hidden))==0);
    PROTOCOL_CHECK(fg_output_slice_hidden_decode(&decoded_head,head_wire,
        sizeof(head_wire)-1u,&error)!=FG_OK);
    uint8_t head_bad[FG_OUTPUT_SLICE_HIDDEN_BYTES];
    memcpy(head_bad,head_wire,sizeof(head_bad));head_bad[2]=1u;
    PROTOCOL_CHECK(fg_output_slice_hidden_decode(&decoded_head,head_bad,
        sizeof(head_bad),&error)!=FG_OK);
    memcpy(head_bad,head_wire,sizeof(head_bad));head_bad[3]=1u;
    PROTOCOL_CHECK(fg_output_slice_hidden_decode(&decoded_head,head_bad,
        sizeof(head_bad),&error)!=FG_OK);
    fg_output_slice_hidden bad_head=head_slice;bad_head.destination_rank=7u;
    PROTOCOL_CHECK(fg_output_slice_hidden_encode(head_wire,&bad_head,&head_bytes,&error)!=FG_OK);
    bad_head.destination_rank=1u;bad_head.hidden[9]=NAN;
    PROTOCOL_CHECK(fg_output_slice_hidden_encode(head_wire,&bad_head,&head_bytes,&error)!=FG_OK);
    PROTOCOL_CHECK(fg_output_slice_hidden_encode(head_wire,&head_slice,&head_bytes,&error)==FG_OK);
    fg_layer_result slice_result={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
        .destination_rank=4u,.token_index=17u};
    slice_result.hyper[0]=1.5f;slice_result.hyper[FG_HYPER_WIDTH-1u]=-2.5f;
    uint8_t slice_wire[FG_DECODE_LAYER_RESULT_BYTES];uint32_t slice_bytes=0u;
    PROTOCOL_CHECK(fg_decode_layer_result_encode(slice_wire,&slice_result,&slice_bytes,&error)==FG_OK&&slice_bytes==FG_DECODE_LAYER_RESULT_BYTES);
    fg_layer_result routed={0};
    PROTOCOL_CHECK(fg_decode_layer_result_decode(&routed,slice_wire,
        sizeof(slice_wire),&error)==FG_OK);
    PROTOCOL_CHECK(routed.destination_rank==4u);
    PROTOCOL_CHECK(routed.destination_rank!=0u);
    PROTOCOL_CHECK(fg_output_slice_encode(slice_wire,&slice_result,&slice_bytes,&error)==FG_OK);
    PROTOCOL_CHECK(fg_decode_layer_result_decode(&routed,slice_wire,
        sizeof(slice_wire),&error)==FG_OK);
    PROTOCOL_CHECK(routed.destination_rank==0u&&routed.source_rank==7u&&
        routed.layer==FG_LAYER_COUNT-1u&&routed.token_index==17u&&
        routed.hyper[0]==1.5f&&routed.hyper[FG_HYPER_WIDTH-1u]==-2.5f);
    fg_layer_work flagged={.layer=12u,.source_rank=1u,.destination_rank=2u,
        .token_index=17u,.position_mode=FG_POSITION_TEXT,
        .flags=FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY};
    uint8_t layer_wire[FG_LAYER_WORK_MAX_BYTES];uint32_t layer_bytes=0;
    PROTOCOL_CHECK(fg_decode_layer_work_encode(layer_wire,sizeof(layer_wire),&layer_bytes,
        FG_PROTOCOL_VERSION,&flagged,&error)==FG_OK);
    fg_layer_work decoded_flag={0};
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&decoded_flag,FG_PROTOCOL_VERSION,layer_wire,
        layer_bytes,&error)==FG_OK&&
        decoded_flag.flags==FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY);
    layer_wire[3]=0x80u;
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&decoded_flag,FG_PROTOCOL_VERSION,layer_wire,
        layer_bytes,&error)!=FG_OK);
    layer_wire[3]=0x01u;
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&decoded_flag,FG_PROTOCOL_VERSION,layer_wire,
        layer_bytes,&error)!=FG_OK);
    fg_layer_work both_flags=flagged;both_flags.layer=1u;
    both_flags.flags=FG_LAYER_WORK_HAS_NGRAM|FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY;
    PROTOCOL_CHECK(fg_decode_layer_work_encode(layer_wire,sizeof(layer_wire),&layer_bytes,
        FG_PROTOCOL_VERSION,&both_flags,&error)==FG_OK);
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&decoded_flag,FG_PROTOCOL_VERSION,layer_wire,
        layer_bytes,&error)==FG_OK&&decoded_flag.flags==
        (FG_LAYER_WORK_HAS_NGRAM|FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY));
    fg_layer_work bf16_src={.layer=6u,.source_rank=1u,.destination_rank=0u,
        .token_index=9u,.position_mode=FG_POSITION_TEXT};
    for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)bf16_src.hyper[i]=sinf((float)i)*8.0f+0.5f;
    PROTOCOL_CHECK(setenv("FG_FABRIC_HOP_BF16","1",1)==0);
    PROTOCOL_CHECK(fg_decode_layer_work_encode(layer_wire,sizeof(layer_wire),&layer_bytes,
        FG_PROTOCOL_VERSION,&bf16_src,&error)==FG_OK&&
        layer_bytes==FG_LAYER_WORK_TEXT_HEADER_BYTES+FG_HYPER_WIDTH*2u);
    fg_layer_work bf16_decoded={0};
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&bf16_decoded,FG_PROTOCOL_VERSION,layer_wire,
        layer_bytes,&error)==FG_OK&&
        bf16_decoded.flags==FG_LAYER_WORK_FLAG_BF16_HYPER);
    for(uint32_t i=0;i<FG_HYPER_WIDTH;i++){
        double delta=fabs((double)bf16_decoded.hyper[i]-(double)bf16_src.hyper[i]);
        double bound=0.00390625*fabs((double)bf16_src.hyper[i])+1e-9;
        PROTOCOL_CHECK(delta<=bound);
    }
    uint8_t result_bf16[FG_LAYER_RESULT_BYTES];uint32_t result_bf16_bytes=0u;
    PROTOCOL_CHECK(fg_decode_layer_result_encode(result_bf16,&slice_result,
        &result_bf16_bytes,&error)==FG_OK&&result_bf16_bytes==FG_LAYER_RESULT_BF16_BYTES);
    fg_layer_result result_bf16_decoded={0};
    PROTOCOL_CHECK(fg_decode_layer_result_decode(&result_bf16_decoded,result_bf16,
        result_bf16_bytes,&error)==FG_OK&&
        result_bf16_decoded.hyper[0]==slice_result.hyper[0]&&
        result_bf16_decoded.hyper[FG_HYPER_WIDTH-1u]==slice_result.hyper[FG_HYPER_WIDTH-1u]);
    uint32_t head_bf16_bytes=0u;
    PROTOCOL_CHECK(fg_output_slice_hidden_encode(head_wire,&head_slice,&head_bf16_bytes,
        &error)==FG_OK&&head_bf16_bytes==FG_OUTPUT_SLICE_HIDDEN_BF16_BYTES&&head_wire[2]==1u);
    fg_output_slice_hidden head_bf16_decoded={0};
    PROTOCOL_CHECK(fg_output_slice_hidden_decode(&head_bf16_decoded,head_wire,
        head_bf16_bytes,&error)==FG_OK&&
        memcmp(head_bf16_decoded.hidden,head_slice.hidden,sizeof(head_slice.hidden))==0);
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&bf16_decoded,FG_PROTOCOL_VERSION,layer_wire,
        FG_LAYER_WORK_TEXT_HEADER_BYTES+FG_HYPER_WIDTH*4u,&error)!=FG_OK);
    layer_wire[3]=(uint8_t)(layer_wire[3]&~FG_LAYER_WORK_FLAG_BF16_HYPER);
    PROTOCOL_CHECK(fg_decode_layer_work_decode(&bf16_decoded,FG_PROTOCOL_VERSION,layer_wire,
        FG_LAYER_WORK_TEXT_HEADER_BYTES+FG_HYPER_WIDTH*2u,&error)!=FG_OK);
    PROTOCOL_CHECK(unsetenv("FG_FABRIC_HOP_BF16")==0);
    PROTOCOL_CHECK(fg_decode_layer_work_encode(layer_wire,sizeof(layer_wire),&layer_bytes,
        FG_PROTOCOL_VERSION,&bf16_src,&error)==FG_OK&&
        layer_bytes==FG_LAYER_WORK_TEXT_HEADER_BYTES+FG_HYPER_WIDTH*4u);
    /* the split handoff messages must pass frame validation on protocol 6 */
    fg_frame_header frame;uint32_t frame_bytes=0;
    PROTOCOL_CHECK(fg_output_config_encode(wire,&config,&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_encode(&frame,FG_MSG_OUTPUT_CONFIG,1u,2u,0u,wire,
        sizeof(wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_validate(&frame,wire,&frame_bytes,&error)==FG_OK&&
        frame_bytes==sizeof(wire)&&fg_frame_type(&frame)==FG_MSG_OUTPUT_CONFIG);
    PROTOCOL_CHECK(fg_frame_encode(&frame,FG_MSG_OUTPUT_HIDDEN,1u,2u,0u,wire,
        sizeof(wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_validate(&frame,wire,&frame_bytes,&error)==FG_OK&&
        fg_frame_type(&frame)==FG_MSG_OUTPUT_HIDDEN);
    PROTOCOL_CHECK(fg_frame_encode(&frame,FG_MSG_OUTPUT_SLICE,1u,2u,0u,wire,
        sizeof(wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_validate(&frame,wire,&frame_bytes,&error)==FG_OK&&
        fg_frame_type(&frame)==FG_MSG_OUTPUT_SLICE);
    PROTOCOL_CHECK(fg_frame_encode(&frame,FG_MSG_OUTPUT_PARTIAL,1u,2u,0u,partial_wire,
        sizeof(partial_wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_validate(&frame,partial_wire,&frame_bytes,&error)==FG_OK&&
        fg_frame_type(&frame)==FG_MSG_OUTPUT_PARTIAL);
    PROTOCOL_CHECK(fg_frame_encode(&frame,FG_MSG_OUTPUT_SLICE_HIDDEN,1u,2u,0u,head_wire,
        sizeof(head_wire),&error)==FG_OK);
    PROTOCOL_CHECK(fg_frame_validate(&frame,head_wire,&frame_bytes,&error)==FG_OK&&
        frame_bytes==sizeof(head_wire)&&
        fg_frame_type(&frame)==FG_MSG_OUTPUT_SLICE_HIDDEN);

    fg_layer_result hidden={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
        .destination_rank=4u,.token_index=17u};
    hidden.hyper[0]=1.5f;hidden.hyper[FG_HYPER_WIDTH-1u]=-2.5f;
    fg_output_handoff state;
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(!fg_output_handoff_ready(&state));
    /* config-first order */
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&config,&error)==FG_OK);
    PROTOCOL_CHECK(!fg_output_handoff_ready(&state));
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&hidden,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_ready(&state));
    fg_output_config taken_config;fg_layer_result taken_hidden;
    fg_output_handoff_take(&state,&taken_config,&taken_hidden);
    PROTOCOL_CHECK(taken_config.token_index==17u&&taken_hidden.hyper[0]==1.5f&&
        taken_hidden.hyper[FG_HYPER_WIDTH-1u]==-2.5f);
    PROTOCOL_CHECK(!fg_output_handoff_ready(&state));
    /* hidden-first order */
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&hidden,&error)==FG_OK);
    PROTOCOL_CHECK(!fg_output_handoff_ready(&state));
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&config,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_ready(&state));
    /* stale halves are dropped and a newer token replaces the pending pair */
    fg_output_handoff_reset(&state);
    fg_output_config older={.source_rank=0u,.destination_rank=4u,.token_index=16u,
        .uniform=0.0f};
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&config,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&older,&error)==FG_OK);
    PROTOCOL_CHECK(state.config.token_index==17u);
    fg_layer_result stale=hidden;stale.token_index=16u;
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&stale,&error)==FG_OK);
    PROTOCOL_CHECK(!state.have_hidden);
    fg_layer_result newer=hidden;newer.token_index=18u;newer.hyper[0]=7.0f;
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&newer,&error)==FG_OK);
    PROTOCOL_CHECK(!state.have_config&&state.have_hidden&&state.hidden.hyper[0]==7.0f);
    fg_output_config next={.source_rank=0u,.destination_rank=4u,.token_index=18u,
        .uniform=0.0f};
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&next,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_ready(&state));
    /* duplicates refresh in place */
    newer.hyper[0]=8.0f;
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&newer,&error)==FG_OK);
    PROTOCOL_CHECK(state.hidden.hyper[0]==8.0f);
    /* the split partial binds to the pending config token and is cleared with it */
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,0u,0.5f,3u,&error)!=FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&config,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,16u,0u,0.5f,3u,&error)!=FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,0u,0.5f,3u,&error)==FG_OK);
    PROTOCOL_CHECK(state.remote_count==1u&&state.remote_value[0]==0.5f&&
        state.remote_id[0]==3u&&state.remote_rank[0]==0u);
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&hidden,&error)==FG_OK);
    PROTOCOL_CHECK(!state.have_local&&state.remote_count==1u);
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&next,&error)==FG_OK);
    PROTOCOL_CHECK(state.remote_count==0u&&!state.have_local);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,18u,0u,0.75f,4u,&error)==FG_OK);
    fg_output_handoff_take(&state,NULL,NULL);
    PROTOCOL_CHECK(state.remote_count==0u&&!state.have_local);
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(!state.have_config&&!state.have_hidden);
    fg_output_handoff_reset(&state);
    fg_output_config quad=config;
    quad.flags=FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4;
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&quad,&error)==FG_OK);
    PROTOCOL_CHECK(!fg_output_handoff_ready(&state));
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&hidden,&error)==FG_OK);
    PROTOCOL_CHECK(!fg_output_handoff_ready(&state)&&fg_output_handoff_sample_ready(&state));
    fg_layer_result head={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
        .destination_rank=4u,.token_index=17u};
    head.hyper[0]=1.25f;head.hyper[FG_HIDDEN_SIZE-1u]=-0.5f;
    PROTOCOL_CHECK(fg_output_handoff_hidden_slice(&state,&head,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_ready(&state)&&state.have_hidden);
    PROTOCOL_CHECK(fg_output_handoff_sample_ready(&state));
    fg_layer_result stale_head=head;stale_head.token_index=16u;
    PROTOCOL_CHECK(fg_output_handoff_hidden_slice(&state,&stale_head,&error)==FG_OK);
    PROTOCOL_CHECK(state.hidden_slice.token_index==17u);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,0u,0.5f,100u,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,1u,0.25f,200u,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,2u,0.75f,300u,&error)==FG_OK);
    PROTOCOL_CHECK(state.remote_count==3u&&state.remote_rank[2]==2u&&
        state.remote_value[2]==0.75f&&state.remote_id[1]==200u);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,3u,1.0f,400u,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,17u,1.0f,500u,&error)!=FG_OK);
    fg_output_handoff_take(&state,NULL,NULL);
    PROTOCOL_CHECK(state.remote_count==0u&&!state.have_hidden_slice);
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(fg_output_split_wait_remaining_ms(0u,1000u,2000u)==-1);
    PROTOCOL_CHECK(fg_output_split_wait_remaining_ms(1000u,1500u,2000u)==1500);
    PROTOCOL_CHECK(fg_output_split_wait_remaining_ms(1000u,3000u,2000u)==0);
    PROTOCOL_CHECK(fg_output_split_wait_remaining_ms(1000u,1000u,2000u)==2000);
    fg_error timeout={0};
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&quad,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_hidden_slice(&state,&head,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,0u,0.5f,100u,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_split_timeout_error(&state,4u,1234u,&timeout)!=FG_OK);
    PROTOCOL_CHECK(strstr(timeout.message,"1234")!=NULL);
    PROTOCOL_CHECK(strstr(timeout.message,"ranks 1,2")!=NULL);
    PROTOCOL_CHECK(fg_output_split_timeout_error(NULL,4u,1u,&timeout)!=FG_OK);
    fg_output_handoff_reset(&state);
}

static fg_status output_handoff_roundtrip(fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    enum{TOKEN_INDEX=77u,LOCAL_ID=100u,REMOTE_ID=200u,RESULT_TOKEN=REMOTE_ID};
    const float LOCAL_VALUE=1.5f,REMOTE_VALUE=2.5f;
    uint8_t wire[FG_OUTPUT_WORK_BYTES];
    if(rank!=0u&&rank!=4u&&rank!=7u)return FG_OK;
    if(rank==0u){
        fg_output_config config={.source_rank=0u,.destination_rank=4u,
            .flags=FG_OUTPUT_CONFIG_FLAG_SPLIT,.token_index=TOKEN_INDEX,.uniform=0.5f};
        fg_status status=fg_output_config_encode(wire,&config,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_CONFIG,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,
            0,wire,FG_OUTPUT_CONFIG_BYTES,error);
        fg_frame_header header;uint32_t recv_bytes=0;
        fg_layer_result slice;
        if(status==FG_OK)status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&recv_bytes,error);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid direct output slice frame");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_decode_layer_result_decode(&slice,wire,recv_bytes,error);
        if(status==FG_OK&&(slice.destination_rank!=0u||slice.source_rank!=7u||
           slice.layer!=FG_LAYER_COUNT-1u||slice.token_index!=TOKEN_INDEX)){
            fg_error_set(error,FG_ERR_MISMATCH,"misrouted direct output slice");
            status=FG_ERR_MISMATCH;
        }
        fg_output_partial partial={.token_index=TOKEN_INDEX,.value=REMOTE_VALUE,
            .id=REMOTE_ID};
        uint8_t partial_wire[FG_OUTPUT_PARTIAL_BYTES];
        if(status==FG_OK)status=fg_output_partial_encode(partial_wire,&partial,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_PARTIAL,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,
            0,partial_wire,FG_OUTPUT_PARTIAL_BYTES,error);
        if(status==FG_OK)status=fg_fabric_recv(fabric,4u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&recv_bytes,error);
        fg_output_result result;
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_RESULT||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid direct output result frame");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_output_result_decode(&result,wire,recv_bytes,error);
        if(status==FG_OK&&(result.source_rank!=4u||result.destination_rank!=0u||
           result.token_index!=TOKEN_INDEX||result.token!=RESULT_TOKEN)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid direct output result payload");
            status=FG_ERR_MISMATCH;
        }
        return status;
    }
    if(rank==7u){
        fg_layer_result hidden={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
            .destination_rank=4u,.token_index=TOKEN_INDEX};
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)hidden.hyper[i]=(float)(i%17u)*0.25f;
        hidden.hyper[0]=LOCAL_VALUE;
        uint32_t hidden_bytes=0u;
        fg_status status=fg_decode_layer_result_encode(wire,&hidden,&hidden_bytes,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_BULK,
            FG_MSG_OUTPUT_HIDDEN,request,
            TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0,wire,
            hidden_bytes,error);
        uint32_t slice_bytes=0u;
        if(status==FG_OK)status=fg_output_slice_encode(wire,&hidden,&slice_bytes,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
            FG_MSG_OUTPUT_SLICE,request,
            TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0,wire,
            slice_bytes,error);
        return status;
    }
    /* rank 4: the hidden arrives first, then the config and the remote partial */
    fg_output_handoff state;
    fg_output_handoff_reset(&state);
    fg_frame_header header;uint32_t bytes=0;
    fg_status status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,sizeof(wire),
        &bytes,error);
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_HIDDEN||
       fg_frame_request_id(&header)!=request||
       fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(error,FG_ERR_MISMATCH,"invalid direct output hidden frame");
        status=FG_ERR_MISMATCH;
    }
    fg_layer_result hidden;
    if(status==FG_OK)status=fg_decode_layer_result_decode(&hidden,wire,bytes,error);
    if(status==FG_OK)status=fg_output_handoff_hidden(&state,&hidden,error);
    if(status==FG_OK)status=fg_fabric_recv(fabric,0u,FG_FABRIC_CONTROL,&header,wire,
        sizeof(wire),&bytes,error);
    fg_output_config config;
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_CONFIG||
       fg_frame_request_id(&header)!=request||
       fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT)){
        fg_error_set(error,FG_ERR_MISMATCH,"invalid direct output config frame");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_config_decode(&config,wire,bytes,error);
    if(status==FG_OK&&(config.flags&FG_OUTPUT_CONFIG_FLAG_SPLIT)==0u){
        fg_error_set(error,FG_ERR_MISMATCH,"direct output config lost the split flag");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_handoff_config(&state,&config,error);
    if(status==FG_OK)status=fg_fabric_recv(fabric,0u,FG_FABRIC_CONTROL,&header,wire,
        sizeof(wire),&bytes,error);
    fg_output_partial partial;
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_PARTIAL||
       fg_frame_request_id(&header)!=request||
       fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT)){
        fg_error_set(error,FG_ERR_MISMATCH,"invalid direct output partial frame");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_partial_decode(&partial,wire,bytes,error);
    if(status==FG_OK)status=fg_output_handoff_partial(&state,partial.token_index,
        0u,partial.value,partial.id,error);
    if(status!=FG_OK)return status;
    if(!fg_output_handoff_ready(&state)){
        fg_error_set(error,FG_ERR_MISMATCH,"output handoff pair did not match");
        return FG_ERR_MISMATCH;
    }
    uint32_t token=state.remote_value[0]>hidden.hyper[0]?state.remote_id[0]:LOCAL_ID;
    float logit=state.remote_value[0]>hidden.hyper[0]?state.remote_value[0]:hidden.hyper[0];
    fg_output_result result={.source_rank=4u,.destination_rank=0u,
        .token_index=TOKEN_INDEX,.token=token,.logit=logit};
    status=fg_output_result_encode(wire,&result,error);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,
        request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,FG_OUTPUT_RESULT_BYTES,
        error);
    return status;
}

static fg_status output_split4_token_roundtrip(fg_fabric *fabric,uint32_t rank,
                                               uint64_t request,uint32_t TOKEN_INDEX,
                                               fg_error *error);

static fg_status output_split4_roundtrip(fg_fabric *fabric,uint32_t rank,uint64_t request,fg_error *error){
    for(uint32_t token=0u;token<4u;token++){
        fg_status status=output_split4_token_roundtrip(fabric,rank,request,token,error);
        if(status!=FG_OK)return status;
    }
    return FG_OK;
}

static fg_status output_split4_token_roundtrip(fg_fabric *fabric,uint32_t rank,
                                               uint64_t request,uint32_t TOKEN_INDEX,
                                               fg_error *error){
    enum{COMBINED_ID=186240u+1u};
    uint8_t wire[FG_OUTPUT_WORK_BYTES];
    fg_frame_header header;uint32_t bytes=0;
    if(rank==0u){
        fg_output_config config={.source_rank=0u,.destination_rank=4u,
            .flags=FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4,
            .token_index=TOKEN_INDEX,.uniform=0.5f};
        fg_status status=fg_output_config_encode(wire,&config,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_CONFIG,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,
            FG_OUTPUT_CONFIG_BYTES,error);
        if(status==FG_OK)status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&bytes,error);
        fg_output_slice_hidden head;
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE_HIDDEN||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way head frame at rank 0");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_output_slice_hidden_decode(&head,wire,bytes,error);
        if(status==FG_OK&&(head.destination_rank!=0u||head.source_rank!=7u||
           head.token_index!=TOKEN_INDEX)){
            fg_error_set(error,FG_ERR_MISMATCH,"misrouted 4-way head at rank 0");
            status=FG_ERR_MISMATCH;
        }
        fg_output_partial partial={.token_index=TOKEN_INDEX,.value=0.6f,.id=62080u+1u};
        uint8_t partial_wire[FG_OUTPUT_PARTIAL_BYTES];
        if(status==FG_OK)status=fg_output_partial_encode(partial_wire,&partial,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_PARTIAL,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,
            partial_wire,FG_OUTPUT_PARTIAL_BYTES,error);
        if(status==FG_OK)status=fg_fabric_recv(fabric,4u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&bytes,error);
        fg_output_result result;
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_RESULT||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way result frame at rank 0");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_output_result_decode(&result,wire,bytes,error);
        if(status==FG_OK&&(result.source_rank!=4u||result.destination_rank!=0u||
           result.token_index!=TOKEN_INDEX||result.token!=COMBINED_ID)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way combined result");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK&&TOKEN_INDEX+1u<4u)
            status=fg_fabric_send(fabric,7u,FG_FABRIC_CONTROL,FG_MSG_CONTROL,request,
                                  TOKEN_INDEX,0,NULL,0,error);
        return status;
    }
    if(rank==7u){
        static const uint32_t destinations[4]={4u,0u,1u,2u};
        fg_status status=FG_OK;
        if(TOKEN_INDEX>0u){
            status=fg_fabric_recv(fabric,0u,FG_FABRIC_CONTROL,&header,wire,sizeof(wire),
                                  &bytes,error);
            if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_CONTROL||
               fg_frame_request_id(&header)!=request)){
                fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way token ack at rank 7");
                status=FG_ERR_MISMATCH;
            }
        }
        for(uint32_t i=0u;status==FG_OK&&i<4u;i++){
            fg_output_slice_hidden head={.source_rank=7u,
                .destination_rank=(uint8_t)destinations[i],.token_index=TOKEN_INDEX};
            for(uint32_t value=0u;value<FG_HIDDEN_SIZE;value++)
                head.hidden[value]=(float)(value%19u)*0.25f;
            uint32_t head_bytes=0u;
            status=fg_output_slice_hidden_encode(wire,&head,&head_bytes,error);
            if(status==FG_OK)status=fg_fabric_send(fabric,destinations[i],FG_FABRIC_BULK,
                FG_MSG_OUTPUT_SLICE_HIDDEN,request,
                TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0,wire,
                head_bytes,error);
        }
        return status;
    }
    if(rank==1u||rank==2u){
        fg_status status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&bytes,error);
        fg_output_slice_hidden head;
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE_HIDDEN||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way head frame at rank %u",rank);
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_output_slice_hidden_decode(&head,wire,bytes,error);
        if(status==FG_OK&&(head.destination_rank!=rank||head.source_rank!=7u)){
            fg_error_set(error,FG_ERR_MISMATCH,"misrouted 4-way head at rank %u",rank);
            status=FG_ERR_MISMATCH;
        }
        fg_output_partial partial={.token_index=TOKEN_INDEX,
            .value=rank==1u?0.7f:0.8f,.id=(rank==1u?124160u:186240u)+1u};
        uint8_t partial_wire[FG_OUTPUT_PARTIAL_BYTES];
        if(status==FG_OK)status=fg_output_partial_encode(partial_wire,&partial,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_PARTIAL,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,
            partial_wire,FG_OUTPUT_PARTIAL_BYTES,error);
        return status;
    }
    if(rank!=4u)return FG_OK;
    fg_output_handoff state;
    fg_output_handoff_reset(&state);
    fg_status status=fg_fabric_recv(fabric,0u,FG_FABRIC_CONTROL,&header,wire,
        sizeof(wire),&bytes,error);
    fg_output_config config;
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_CONFIG||
       fg_frame_request_id(&header)!=request)){
        fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way config frame");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_config_decode(&config,wire,bytes,error);
    if(status==FG_OK&&config.flags!=(FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4)){
        fg_error_set(error,FG_ERR_MISMATCH,"4-way config flags changed");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_handoff_config(&state,&config,error);
    if(status==FG_OK)status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
        sizeof(wire),&bytes,error);
    fg_output_slice_hidden head;
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE_HIDDEN||
       fg_frame_request_id(&header)!=request||
       fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){
        fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way head frame at rank 4");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_output_slice_hidden_decode(&head,wire,bytes,error);
    if(status==FG_OK&&head.destination_rank!=4u){
        fg_error_set(error,FG_ERR_MISMATCH,"misrouted 4-way head at rank 4");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK){
        fg_layer_result stored={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
            .destination_rank=4u,.token_index=TOKEN_INDEX};
        memcpy(stored.hyper,head.hidden,sizeof(head.hidden));
        status=fg_output_handoff_hidden_slice(&state,&stored,error);
    }
    if(status==FG_OK&&!fg_output_handoff_ready(&state)){
        fg_error_set(error,FG_ERR_MISMATCH,"4-way handoff is not ready");
        status=FG_ERR_MISMATCH;
    }
    for(uint32_t received=0u;status==FG_OK&&received<3u;received++){
        uint32_t peer=0u;
        status=fg_fabric_recv_any(fabric,FG_FABRIC_CONTROL,&peer,&header,wire,
            sizeof(wire),&bytes,error);
        fg_output_partial partial;
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_PARTIAL||
           fg_frame_request_id(&header)!=request||
           fg_frame_sequence(&header)!=TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid 4-way partial frame");
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_output_partial_decode(&partial,wire,bytes,error);
        if(status==FG_OK)status=fg_output_handoff_partial(&state,partial.token_index,
            (uint8_t)peer,partial.value,partial.id,error);
    }
    if(status==FG_OK&&state.remote_count!=3u){
        fg_error_set(error,FG_ERR_MISMATCH,"4-way partial count changed");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK){
        state.local_value=0.5f;state.local_id=100u;state.have_local=true;
        uint32_t combined_id=state.local_id;
        float combined_value=state.local_value;
        for(uint32_t i=0u;i<state.remote_count;i++){
            uint32_t way=0u;
            if(!fg_output_split_way_for_rank(4u,state.remote_rank[i],&way)){
                fg_error_set(error,FG_ERR_MISMATCH,"4-way partial owner %u",state.remote_rank[i]);
                status=FG_ERR_MISMATCH;
                break;
            }
            uint32_t first=0u,rows=0u;
            fg_output_split_span(4u,way,&first,&rows);
            if(state.remote_id[i]<first||state.remote_id[i]>=first+rows){
                fg_error_set(error,FG_ERR_MISMATCH,"4-way partial id %u outside its slice",
                             state.remote_id[i]);
                status=FG_ERR_MISMATCH;
                break;
            }
            if(state.remote_value[i]>combined_value){
                combined_value=state.remote_value[i];combined_id=state.remote_id[i];
            }
        }
        fg_output_result result={.source_rank=4u,.destination_rank=0u,
            .token_index=TOKEN_INDEX,.token=combined_id,.logit=combined_value};
        uint8_t result_wire[FG_OUTPUT_RESULT_BYTES];
        if(status==FG_OK&&combined_id!=COMBINED_ID){
            fg_error_set(error,FG_ERR_MISMATCH,"4-way combined id %u",combined_id);
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_output_result_encode(result_wire,&result,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
            FG_MSG_OUTPUT_RESULT,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,
            result_wire,FG_OUTPUT_RESULT_BYTES,error);
    }
    return status;
}

static fg_status output_split4_timeout_roundtrip(fg_fabric *fabric,uint32_t rank,
                                                 uint64_t request,fg_error *error){
    enum{TOKEN_INDEX=99u,WAIT_MS=250};
    uint8_t wire[FG_OUTPUT_WORK_BYTES];
    fg_frame_header header;uint32_t bytes=0;
    if(rank==0u){
        fg_output_config config={.source_rank=0u,.destination_rank=4u,
            .flags=FG_OUTPUT_CONFIG_FLAG_SPLIT|FG_OUTPUT_CONFIG_FLAG_SPLIT_4,
            .token_index=TOKEN_INDEX,.uniform=0.5f};
        fg_status status=fg_output_config_encode(wire,&config,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_CONFIG,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,
            FG_OUTPUT_CONFIG_BYTES,error);
        if(status==FG_OK)status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&bytes,error);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE_HIDDEN||
           fg_frame_request_id(&header)!=request)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid timeout head frame at rank 0");
            status=FG_ERR_MISMATCH;
        }
        fg_output_partial partial={.token_index=TOKEN_INDEX,.value=0.6f,.id=62080u+1u};
        uint8_t partial_wire[FG_OUTPUT_PARTIAL_BYTES];
        if(status==FG_OK)status=fg_output_partial_encode(partial_wire,&partial,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
            FG_MSG_OUTPUT_PARTIAL,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,
            partial_wire,FG_OUTPUT_PARTIAL_BYTES,error);
        if(status==FG_OK){
            uint32_t peer=0u;
            status=fg_fabric_recv_any_timeout(fabric,FG_FABRIC_BULK,WAIT_MS*3,&peer,&header,
                wire,sizeof(wire),&bytes,error);
            if(status==FG_ERR_LIMIT||status==FG_ERR_IO){
                fprintf(stderr,"timeout probe rank 0: bounded result wait ended without a result\n");
                status=FG_OK;
            }else if(status==FG_OK)status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK){
            status=fg_fabric_recv(fabric,4u,FG_FABRIC_CONTROL,&header,wire,sizeof(wire),
                &bytes,error);
            if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_CONTROL||
               fg_frame_request_id(&header)!=request))status=FG_ERR_MISMATCH;
        }
        return status;
    }
    if(rank==7u){
        static const uint32_t destinations[4]={4u,0u,1u,2u};
        fg_status status=FG_OK;
        for(uint32_t i=0u;status==FG_OK&&i<4u;i++){
            fg_output_slice_hidden head={.source_rank=7u,
                .destination_rank=(uint8_t)destinations[i],.token_index=TOKEN_INDEX};
            for(uint32_t value=0u;value<FG_HIDDEN_SIZE;value++)
                head.hidden[value]=(float)(value%19u)*0.25f;
            uint32_t head_bytes=0u;
            status=fg_output_slice_hidden_encode(wire,&head,&head_bytes,error);
            if(status==FG_OK)status=fg_fabric_send(fabric,destinations[i],FG_FABRIC_BULK,
                FG_MSG_OUTPUT_SLICE_HIDDEN,request,
                TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0,wire,
                head_bytes,error);
        }
        if(status==FG_OK){
            status=fg_fabric_recv(fabric,4u,FG_FABRIC_CONTROL,&header,wire,sizeof(wire),
                &bytes,error);
            if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_CONTROL||
               fg_frame_request_id(&header)!=request))status=FG_ERR_MISMATCH;
        }
        return status;
    }
    if(rank==1u||rank==2u){
        fg_status status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
            sizeof(wire),&bytes,error);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE_HIDDEN||
           fg_frame_request_id(&header)!=request)){
            fg_error_set(error,FG_ERR_MISMATCH,"invalid timeout head frame at rank %u",rank);
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK&&rank==2u){
            fg_output_partial partial={.token_index=TOKEN_INDEX,.value=0.8f,.id=186240u+1u};
            uint8_t partial_wire[FG_OUTPUT_PARTIAL_BYTES];
            status=fg_output_partial_encode(partial_wire,&partial,error);
            if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_CONTROL,
                FG_MSG_OUTPUT_PARTIAL,request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,
                partial_wire,FG_OUTPUT_PARTIAL_BYTES,error);
        }
        if(status==FG_OK){
            status=fg_fabric_recv(fabric,4u,FG_FABRIC_CONTROL,&header,wire,sizeof(wire),
                &bytes,error);
            if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_CONTROL||
               fg_frame_request_id(&header)!=request))status=FG_ERR_MISMATCH;
        }
        return status;
    }
    if(rank!=4u){
        fg_status status=fg_fabric_recv(fabric,4u,FG_FABRIC_CONTROL,&header,wire,
            sizeof(wire),&bytes,error);
        if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_CONTROL||
           fg_frame_request_id(&header)!=request))status=FG_ERR_MISMATCH;
        return status;
    }
    fg_output_handoff state;
    fg_output_handoff_reset(&state);
    fg_status status=fg_fabric_recv(fabric,0u,FG_FABRIC_CONTROL,&header,wire,
        sizeof(wire),&bytes,error);
    fg_output_config config;
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_CONFIG||
       fg_frame_request_id(&header)!=request))status=FG_ERR_MISMATCH;
    if(status==FG_OK)status=fg_output_config_decode(&config,wire,bytes,error);
    if(status==FG_OK)status=fg_output_handoff_config(&state,&config,error);
    if(status==FG_OK)status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,wire,
        sizeof(wire),&bytes,error);
    fg_output_slice_hidden head;
    if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_OUTPUT_SLICE_HIDDEN||
       fg_frame_request_id(&header)!=request))status=FG_ERR_MISMATCH;
    if(status==FG_OK)status=fg_output_slice_hidden_decode(&head,wire,bytes,error);
    if(status==FG_OK){
        fg_layer_result stored={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
            .destination_rank=4u,.token_index=TOKEN_INDEX};
        memcpy(stored.hyper,head.hidden,sizeof(head.hidden));
        status=fg_output_handoff_hidden_slice(&state,&stored,error);
    }
    if(status!=FG_OK)return status;
    state.local_value=0.5f;state.local_id=100u;state.have_local=true;
    uint32_t received=0u;
    while(received<2u){
        uint32_t peer=0u;
        status=fg_fabric_recv_any_timeout(fabric,FG_FABRIC_CONTROL,WAIT_MS,&peer,&header,
            wire,sizeof(wire),&bytes,error);
        if(status==FG_ERR_LIMIT)return FG_ERR_MISMATCH;
        if(status!=FG_OK)return status;
        fg_output_partial partial;
        status=fg_output_partial_decode(&partial,wire,bytes,error);
        if(status!=FG_OK)return status;
        status=fg_output_handoff_partial(&state,partial.token_index,(uint8_t)peer,
            partial.value,partial.id,error);
        if(status!=FG_OK)return status;
        received++;
    }
    {
        uint32_t peer=0u;
        status=fg_fabric_recv_any_timeout(fabric,FG_FABRIC_CONTROL,WAIT_MS,&peer,&header,
            wire,sizeof(wire),&bytes,error);
        if(status!=FG_ERR_LIMIT){
            fg_error_set(error,FG_ERR_MISMATCH,
                "timeout probe: the missing partial did not time out");
            return FG_ERR_MISMATCH;
        }
        fg_error timeout={0};
        fg_status check=fg_output_split_timeout_error(&state,4u,WAIT_MS,&timeout);
        if(check!=FG_ERR_LIMIT||!strstr(timeout.message,"ranks 1")){
            fg_error_set(error,FG_ERR_MISMATCH,
                "timeout probe: expected rank 1 in '%s'",timeout.message);
            return FG_ERR_MISMATCH;
        }
        fprintf(stderr,"timeout probe rank 4: %s\n",timeout.message);
    }
    fg_status release=FG_OK;
    for(uint32_t peer=0u;release==FG_OK&&peer<FG_RANK_COUNT;peer++)
        if(peer!=rank)
            release=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_CONTROL,request,
                                   0u,0u,NULL,0u,error);
    return release;
}

static int child_main(const fg_manifest *manifest,uint32_t rank){fg_error error={0};fg_fabric *fabric=NULL;fg_status status=fg_fabric_open(&fabric,manifest,rank,&error);if(status==FG_ERR_UNAVAILABLE)return 77;if(status!=FG_OK){fprintf(stderr,"rank %u fabric open: %s\n",rank,error.message);return 1;}uint64_t request=UINT64_C(0x1122334455667788);for(uint32_t peer=0;status==FG_OK&&peer<FG_RANK_COUNT;peer++)if(peer!=rank)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_READY,request,rank,0,NULL,0,&error);bool seen[FG_RANK_COUNT]={0};for(uint32_t received=0;status==FG_OK&&received<FG_RANK_COUNT-1u;received++){uint32_t peer=0,bytes=0;fg_frame_header header;status=fg_fabric_recv_any(fabric,FG_FABRIC_CONTROL,&peer,&header,NULL,0,&bytes,&error);if(status==FG_OK&&(peer==rank||seen[peer]||bytes||fg_frame_type(&header)!=FG_MSG_READY||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=peer)){fprintf(stderr,"rank %u invalid READY from %u\n",rank,peer);status=FG_ERR_MISMATCH;}seen[peer]=true;}if(status==FG_OK)status=batch_send_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=prefill_bulk_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=delayed_bulk_ngram_control_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=prefill_layer_chain_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=decode_layer_chain_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=output_handoff_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=output_split4_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=output_split4_timeout_roundtrip(fabric,rank,request,&error);fg_fabric_close(fabric);if(status!=FG_OK){fprintf(stderr,"rank %u fabric exchange: %s\n",rank,error.message);return 1;}return 0;}

int main(void){protocol_output_handoff_selfcheck();if(protocol_failures)return 1;fg_manifest manifest;fg_manifest_init(&manifest);manifest.protocol_version=FG_PROTOCOL_VERSION;uint32_t base=24000u+(uint32_t)(getpid()%5000u)*2u;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)snprintf(manifest.ranks[rank].endpoint,sizeof(manifest.ranks[rank].endpoint),"127.0.0.1:%u",base+rank*2u);for(uint32_t i=0;i<32u;i++)manifest.manifest_sha256[i]=(uint8_t)(i*7u+3u);pid_t children[FG_RANK_COUNT];for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){children[rank]=fork();if(children[rank]<0){perror("fork");return 1;}if(children[rank]==0)_exit(child_main(&manifest,rank));}uint32_t passed=0,skipped=0;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){int status;if(waitpid(children[rank],&status,0)<0){perror("waitpid");return 1;}if(WIFEXITED(status)&&WEXITSTATUS(status)==0)passed++;else if(WIFEXITED(status)&&WEXITSTATUS(status)==77)skipped++;else{fprintf(stderr,"fabric rank %u exited abnormally\n",rank);return 1;}}if(skipped==FG_RANK_COUNT){fprintf(stderr,"SKIP eight-process fabric: io_uring unavailable\n");return 77;}if(passed!=FG_RANK_COUNT||skipped){fprintf(stderr,"inconsistent fabric qualification: %u pass %u skip\n",passed,skipped);return 1;}puts("Flash Gordon protocol 6 and eight-process dual-channel mesh: PASS");return 0;}
