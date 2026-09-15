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
            if(cycle==0u){fg_layer_work first={.layer=1u,.source_rank=0u,.destination_rank=1u,.flags=FG_LAYER_WORK_HAS_NGRAM,.token_index=TOKEN};memcpy(first.position,position,sizeof(position));memcpy(first.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u);memcpy(first.ngram_embedding,ngram,(size_t)FG_NGRAM_EMBED_VALUES*4u);uint32_t bytes=0;status=fg_layer_work_encode(wire,work_capacity,&bytes,&first,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+1u,0,wire,bytes,error);}
            else{uint32_t boundary=base,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,7u,FG_FABRIC_CONTROL,&header,wire,work_capacity,&bytes,error);fg_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+boundary)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct decode boundary frame at layer %u",boundary);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_work_decode(&work,wire,bytes,error);if(status==FG_OK&&(work.layer!=boundary||work.source_rank!=7u||work.destination_rank!=0u||work.token_index!=TOKEN||memcmp(work.position,position,sizeof(position))!=0||memcmp(work.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0||work.flags!=0u)){fg_error_set(error,FG_ERR_MISMATCH,"direct decode boundary payload mismatch at layer %u",boundary);status=FG_ERR_MISMATCH;}if(status==FG_OK){fg_layer_work next={.layer=(uint8_t)(boundary+1u),.source_rank=0u,.destination_rank=1u,.token_index=TOKEN};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+boundary+1u,0,wire,bytes,error);}}
        }else{uint32_t layer=base+rank,peer=rank-1u,bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,peer,FG_FABRIC_CONTROL,&header,wire,work_capacity,&bytes,error);fg_layer_work work={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_WORK||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+layer)){fg_error_set(error,FG_ERR_MISMATCH,"invalid direct decode layer hop at rank %u",rank);status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_work_decode(&work,wire,bytes,error);if(status==FG_OK&&(work.layer!=layer||work.source_rank!=peer||work.destination_rank!=rank||work.token_index!=TOKEN||memcmp(work.position,position,sizeof(position))!=0||memcmp(work.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0||work.flags!=(layer==1u?FG_LAYER_WORK_HAS_NGRAM:0u)||(layer==1u&&memcmp(work.ngram_embedding,ngram,(size_t)FG_NGRAM_EMBED_VALUES*4u)!=0))){fg_error_set(error,FG_ERR_MISMATCH,"direct decode layer payload mismatch at rank %u layer %u",rank,layer);status=FG_ERR_MISMATCH;}if(status==FG_OK){if(rank<7u){fg_layer_work next={.layer=(uint8_t)(layer+1u),.source_rank=(uint8_t)rank,.destination_rank=(uint8_t)(rank+1u),.token_index=TOKEN};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,rank+1u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+layer+1u,0,wire,bytes,error);}else if(cycle+1u<CYCLES){fg_layer_work next={.layer=(uint8_t)(layer+1u),.source_rank=7u,.destination_rank=0u,.token_index=TOKEN};memcpy(next.position,work.position,sizeof(next.position));memcpy(next.hyper,work.hyper,sizeof(next.hyper));status=fg_layer_work_encode(wire,work_capacity,&bytes,&next,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_CONTROL,FG_MSG_LAYER_WORK,request,TOKEN*FG_LAYER_COUNT+layer+1u,0,wire,bytes,error);}else{fg_layer_result result={.layer=(uint8_t)layer,.source_rank=7u,.destination_rank=0u,.token_index=TOKEN};memcpy(result.hyper,work.hyper,sizeof(result.hyper));status=fg_layer_result_encode(result_wire,&result,error);if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_LAYER_RESULT,request,TOKEN*FG_LAYER_COUNT+layer,0,result_wire,result_capacity,error);}}}
    }
    if(status==FG_OK&&rank==0u){uint32_t bytes=0;fg_frame_header header;status=fg_fabric_recv(fabric,7u,FG_FABRIC_BULK,&header,result_wire,result_capacity,&bytes,error);fg_layer_result result={0};if(status==FG_OK&&(fg_frame_type(&header)!=FG_MSG_LAYER_RESULT||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=TOKEN*FG_LAYER_COUNT+FG_LAYER_COUNT-1u)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct decode layer frame");status=FG_ERR_MISMATCH;}if(status==FG_OK)status=fg_layer_result_decode(&result,result_wire,bytes,error);if(status==FG_OK&&(result.layer!=FG_LAYER_COUNT-1u||result.source_rank!=7u||result.destination_rank!=0u||result.token_index!=TOKEN||memcmp(result.hyper,hyper,(size_t)FG_HYPER_WIDTH*4u)!=0)){fg_error_set(error,FG_ERR_MISMATCH,"invalid final direct decode layer payload");status=FG_ERR_MISMATCH;}}
    free(ngram);free(hyper);free(result_wire);free(wire);return status;
}

static void protocol_output_handoff_selfcheck(void){
    fg_error error={0};
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
    uint8_t bad[FG_OUTPUT_CONFIG_BYTES];
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
    /* the split slice must re-route the final layer result to rank 0: the
     * hidden payload alone names the output owner and must never be accepted
     * by the coordinator's slice handler */
    fg_layer_result slice_result={.layer=FG_LAYER_COUNT-1u,.source_rank=7u,
        .destination_rank=4u,.token_index=17u};
    slice_result.hyper[0]=1.5f;slice_result.hyper[FG_HYPER_WIDTH-1u]=-2.5f;
    uint8_t slice_wire[FG_DECODE_LAYER_RESULT_BYTES];
    PROTOCOL_CHECK(fg_decode_layer_result_encode(slice_wire,&slice_result,&error)==FG_OK);
    fg_layer_result routed={0};
    PROTOCOL_CHECK(fg_decode_layer_result_decode(&routed,slice_wire,
        sizeof(slice_wire),&error)==FG_OK);
    PROTOCOL_CHECK(routed.destination_rank==4u);
    PROTOCOL_CHECK(routed.destination_rank!=0u);
    PROTOCOL_CHECK(fg_output_slice_encode(slice_wire,&slice_result,&error)==FG_OK);
    PROTOCOL_CHECK(fg_decode_layer_result_decode(&routed,slice_wire,
        sizeof(slice_wire),&error)==FG_OK);
    PROTOCOL_CHECK(routed.destination_rank==0u&&routed.source_rank==7u&&
        routed.layer==FG_LAYER_COUNT-1u&&routed.token_index==17u&&
        routed.hyper[0]==1.5f&&routed.hyper[FG_HYPER_WIDTH-1u]==-2.5f);
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
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,0.5f,3u,&error)!=FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&config,&error)==FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,16u,0.5f,3u,&error)!=FG_OK);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,17u,0.5f,3u,&error)==FG_OK);
    PROTOCOL_CHECK(state.have_remote&&state.remote_value==0.5f&&state.remote_id==3u);
    PROTOCOL_CHECK(fg_output_handoff_hidden(&state,&hidden,&error)==FG_OK);
    PROTOCOL_CHECK(!state.have_local&&state.have_remote);
    PROTOCOL_CHECK(fg_output_handoff_config(&state,&next,&error)==FG_OK);
    PROTOCOL_CHECK(!state.have_remote&&!state.have_local);
    PROTOCOL_CHECK(fg_output_handoff_partial(&state,18u,0.75f,4u,&error)==FG_OK);
    fg_output_handoff_take(&state,NULL,NULL);
    PROTOCOL_CHECK(!state.have_remote&&!state.have_local);
    fg_output_handoff_reset(&state);
    PROTOCOL_CHECK(!state.have_config&&!state.have_hidden);
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
        fg_status status=fg_decode_layer_result_encode(wire,&hidden,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,4u,FG_FABRIC_BULK,
            FG_MSG_OUTPUT_HIDDEN,request,
            TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0,wire,
            FG_DECODE_LAYER_RESULT_BYTES,error);
        if(status==FG_OK)status=fg_output_slice_encode(wire,&hidden,error);
        if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,
            FG_MSG_OUTPUT_SLICE,request,
            TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT-1u,0,wire,
            FG_DECODE_LAYER_RESULT_BYTES,error);
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
        partial.value,partial.id,error);
    if(status!=FG_OK)return status;
    if(!fg_output_handoff_ready(&state)){
        fg_error_set(error,FG_ERR_MISMATCH,"output handoff pair did not match");
        return FG_ERR_MISMATCH;
    }
    uint32_t token=state.remote_value>hidden.hyper[0]?state.remote_id:LOCAL_ID;
    float logit=state.remote_value>hidden.hyper[0]?state.remote_value:hidden.hyper[0];
    fg_output_result result={.source_rank=4u,.destination_rank=0u,
        .token_index=TOKEN_INDEX,.token=token,.logit=logit};
    status=fg_output_result_encode(wire,&result,error);
    if(status==FG_OK)status=fg_fabric_send(fabric,0u,FG_FABRIC_BULK,FG_MSG_OUTPUT_RESULT,
        request,TOKEN_INDEX*FG_LAYER_COUNT+FG_LAYER_COUNT,0,wire,FG_OUTPUT_RESULT_BYTES,
        error);
    return status;
}

static int child_main(const fg_manifest *manifest,uint32_t rank){fg_error error={0};fg_fabric *fabric=NULL;fg_status status=fg_fabric_open(&fabric,manifest,rank,&error);if(status==FG_ERR_UNAVAILABLE)return 77;if(status!=FG_OK){fprintf(stderr,"rank %u fabric open: %s\n",rank,error.message);return 1;}uint64_t request=UINT64_C(0x1122334455667788);for(uint32_t peer=0;status==FG_OK&&peer<FG_RANK_COUNT;peer++)if(peer!=rank)status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_READY,request,rank,0,NULL,0,&error);bool seen[FG_RANK_COUNT]={0};for(uint32_t received=0;status==FG_OK&&received<FG_RANK_COUNT-1u;received++){uint32_t peer=0,bytes=0;fg_frame_header header;status=fg_fabric_recv_any(fabric,FG_FABRIC_CONTROL,&peer,&header,NULL,0,&bytes,&error);if(status==FG_OK&&(peer==rank||seen[peer]||bytes||fg_frame_type(&header)!=FG_MSG_READY||fg_frame_request_id(&header)!=request||fg_frame_sequence(&header)!=peer)){fprintf(stderr,"rank %u invalid READY from %u\n",rank,peer);status=FG_ERR_MISMATCH;}seen[peer]=true;}if(status==FG_OK)status=batch_send_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=prefill_bulk_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=delayed_bulk_ngram_control_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=prefill_layer_chain_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=decode_layer_chain_roundtrip(fabric,rank,request,&error);if(status==FG_OK)status=output_handoff_roundtrip(fabric,rank,request,&error);fg_fabric_close(fabric);if(status!=FG_OK){fprintf(stderr,"rank %u fabric exchange: %s\n",rank,error.message);return 1;}return 0;}

int main(void){protocol_output_handoff_selfcheck();if(protocol_failures)return 1;fg_manifest manifest;fg_manifest_init(&manifest);manifest.protocol_version=FG_PROTOCOL_VERSION;uint32_t base=24000u+(uint32_t)(getpid()%5000u)*2u;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)snprintf(manifest.ranks[rank].endpoint,sizeof(manifest.ranks[rank].endpoint),"127.0.0.1:%u",base+rank*2u);for(uint32_t i=0;i<32u;i++)manifest.manifest_sha256[i]=(uint8_t)(i*7u+3u);pid_t children[FG_RANK_COUNT];for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){children[rank]=fork();if(children[rank]<0){perror("fork");return 1;}if(children[rank]==0)_exit(child_main(&manifest,rank));}uint32_t passed=0,skipped=0;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){int status;if(waitpid(children[rank],&status,0)<0){perror("waitpid");return 1;}if(WIFEXITED(status)&&WEXITSTATUS(status)==0)passed++;else if(WIFEXITED(status)&&WEXITSTATUS(status)==77)skipped++;else{fprintf(stderr,"fabric rank %u exited abnormally\n",rank);return 1;}}if(skipped==FG_RANK_COUNT){fprintf(stderr,"SKIP eight-process fabric: io_uring unavailable\n");return 77;}if(passed!=FG_RANK_COUNT||skipped){fprintf(stderr,"inconsistent fabric qualification: %u pass %u skip\n",passed,skipped);return 1;}puts("Flash Gordon protocol 6 and eight-process dual-channel mesh: PASS");return 0;}
