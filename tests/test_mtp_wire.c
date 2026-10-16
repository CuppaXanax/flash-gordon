#include "fg_mtp_wire.h"
#include "fg_fabric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

static void target_codec(void){
    fg_error err={0};uint32_t bytes=0u;
    uint32_t values=2u*FG_HYPER_WIDTH;
    uint32_t capacity=FG_MTP_TARGET_HEADER_BYTES+(values+2u)*4u;
    uint8_t *wire=malloc(capacity);
    float *input=calloc(values,sizeof(float)),*output=calloc(values,sizeof(float));
    CHECK(wire&&input&&output);if(!wire||!input||!output)exit(1);
    input[0]=-0.0f;input[1]=1.0e-40f;input[values-1u]=17.5f;
    uint32_t next[2]={71u,UINT32_MAX},tokens[2]={0};
    fg_mtp_target_work work={.kind=FG_MTP_TARGET_PREFILL,.source_rank=7u,
        .destination_rank=6u,.first_token=12u,.count=2u,
        .hyper=input,.next_tokens=next},got;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,false,&err)==FG_OK);
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,2u,wire,bytes,false,&err)==FG_OK);
    CHECK(!memcmp(input,output,(size_t)values*4u)&&!memcmp(next,tokens,sizeof(next)));
    CHECK(got.kind==work.kind&&got.count==2u&&got.first_token==12u&&
          got.source_rank==7u&&got.destination_rank==6u);
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,2u,wire,bytes-1u,false,&err)!=FG_OK);
    CHECK(fg_mtp_target_decode(&got,output,values-1u,tokens,2u,wire,bytes,false,&err)!=FG_OK);
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,1u,wire,bytes,false,&err)!=FG_OK);
    wire[13]=1u;
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,2u,wire,bytes,false,&err)!=FG_OK);
    wire[13]=0u;wire[2]=6u;
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,2u,wire,bytes,false,&err)!=FG_OK);
    wire[2]=7u;wire[16]=0x7fu;wire[17]=0xc0u;
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,2u,wire,bytes,false,&err)!=FG_OK);
    input[1]=NAN;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,false,&err)!=FG_OK);
    input[1]=1.0f;next[0]=UINT32_MAX;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,false,&err)!=FG_OK);
    next[0]=71u;work.kind=FG_MTP_TARGET_DECODE;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,false,&err)!=FG_OK);
    work.count=1u;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,false,&err)==FG_OK);
    CHECK(fg_mtp_target_decode(&got,output,values,NULL,0u,wire,bytes,false,&err)==FG_OK);
    work.kind=FG_MTP_TARGET_VERIFY;work.count=FG_MTP_MAX_VERIFY+1u;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,true,&err)!=FG_OK);
    work.count=2u;work.source_rank=6u;work.destination_rank=7u;
    CHECK(fg_mtp_target_encode(wire,capacity,&bytes,&work,true,&err)==FG_OK&&bytes==16u);
    CHECK(fg_mtp_target_decode(&got,NULL,0u,NULL,0u,wire,bytes,true,&err)==FG_OK);
    CHECK(fg_mtp_target_decode(&got,output,values,tokens,2u,wire,bytes,false,&err)!=FG_OK);
    free(output);free(input);free(wire);
}

int main(void){
    target_codec();
    fg_error err={0};uint32_t bytes=0u;
    uint32_t capacity=FG_MTP_HEAD_WORK_MAX_BYTES;
    uint8_t *wire=malloc(capacity);
    float *input=malloc((size_t)FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE*sizeof(float));
    float *output=malloc((size_t)FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE*sizeof(float));
    CHECK(wire&&input&&output);
    if(!wire||!input||!output)return 1;
    for(uint32_t i=0;i<FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE;i++)
        input[i]=(float)((int)(i%127u)-63)*0.03125f;
    char path[FG_MTP_OVERLAY_PATH_MAX+1u],embedding_path[FG_MTP_OVERLAY_PATH_MAX+1u];
    uint32_t ids[4]={1u,2u,3u,4u},decoded_ids[4];
    fg_mtp_native_work decoded;
    fg_mtp_native_work work={.operation=FG_MTP_NATIVE_CONFIG,
        .overlay="weights/native-q8.gguf",.embedding_file="weights/embedding.fgme",
        .count=22u,.argument=8192u,.limit=64u};
    work.count=(uint32_t)strlen(work.overlay);
    CHECK(fg_mtp_native_work_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&work,&err)==FG_OK);
    CHECK(fg_mtp_native_work_decode(&decoded,decoded_ids,4u,
        path,embedding_path,wire,bytes,&err)==FG_OK);
    CHECK(!strcmp(decoded.overlay,work.overlay)&&!strcmp(decoded.embedding_file,work.embedding_file)&&
          decoded.argument==8192u&&decoded.limit==64u);
    work.owner_rank_plus_one=7u;
    CHECK(fg_mtp_native_work_encode(wire,capacity,&bytes,&work,&err)==FG_OK&&wire[2]==7u);
    CHECK(fg_mtp_native_work_decode(&decoded,decoded_ids,4u,path,embedding_path,
        wire,bytes,&err)==FG_OK&&decoded.owner_rank_plus_one==7u);
    work.owner_rank_plus_one=FG_RANK_COUNT+1u;
    CHECK(fg_mtp_native_work_encode(wire,capacity,&bytes,&work,&err)!=FG_OK);
    wire[FG_MTP_NATIVE_HEADER_BYTES]=0u;
    CHECK(fg_mtp_native_work_decode(&decoded,decoded_ids,4u,
        path,embedding_path,wire,bytes,&err)!=FG_OK);
    const fg_mtp_native_work cases[]={
        {.operation=FG_MTP_NATIVE_PREFIX,.first_token=10u,.count=1u,.argument=5u,
         .limit=20u,.tokens=ids},
        {.operation=FG_MTP_NATIVE_DRAFT,.first_token=20u,.count=1u,.argument=2u,
         .tokens=ids},
        {.operation=FG_MTP_NATIVE_FINISH,.first_token=20u,.count=4u,.tokens=ids},
        {.operation=FG_MTP_NATIVE_FINISH,.first_token=20u,.count=1u},
        {.operation=FG_MTP_NATIVE_ADVANCE,.first_token=20u,.count=1u,.tokens=ids}
    };
    for(uint32_t c=0;c<sizeof(cases)/sizeof(*cases);c++){
        work=cases[c];
        CHECK(fg_mtp_native_work_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&work,&err)==FG_OK);
        CHECK(fg_mtp_native_work_decode(&decoded,decoded_ids,4u,
            path,embedding_path,wire,bytes,&err)==FG_OK);
        CHECK(decoded.operation==work.operation&&decoded.first_token==work.first_token&&
              decoded.count==work.count&&decoded.argument==work.argument&&decoded.limit==work.limit);
        uint32_t rows=work.operation==FG_MTP_NATIVE_FINISH?work.count-1u:work.count;
        CHECK(!memcmp(ids,decoded_ids,(size_t)rows*sizeof(uint32_t)));
        CHECK(fg_mtp_native_work_decode(&decoded,decoded_ids,4u,
            path,embedding_path,wire,bytes-1u,&err)!=FG_OK);
        wire[20u]=1u;
        CHECK(fg_mtp_native_work_decode(&decoded,decoded_ids,4u,
            path,embedding_path,wire,bytes,&err)!=FG_OK);
    }
    work=cases[1];ids[0]=FG_Q38_VOCAB_SIZE;
    CHECK(fg_mtp_native_work_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&work,&err)!=FG_OK);
    ids[0]=1u;work.first_token=0u;
    CHECK(fg_mtp_native_work_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&work,&err)!=FG_OK);
    work=cases[1];work.first_token=UINT32_MAX;
    CHECK(fg_mtp_native_work_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&work,&err)!=FG_OK);
    for(uint32_t n=1u;n<=FG_MTP_MAX_VERIFY;n++){
        fg_mtp_head_work head={.kind=FG_MTP_HEAD_VERIFY,.source_rank=7u,
            .destination_rank=1u,.first_token=100u,.count=n,.hidden=input},got;
        CHECK(fg_mtp_head_work_encode(wire,capacity,&bytes,&head,&err)==FG_OK);
        CHECK(fg_mtp_head_work_decode(&got,output,FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE,
            wire,bytes,&err)==FG_OK);
        CHECK(got.count==n&&!memcmp(input,output,(size_t)n*FG_HIDDEN_SIZE*sizeof(float)));
        CHECK(fg_mtp_head_work_decode(&got,output,n*FG_HIDDEN_SIZE-1u,wire,bytes,&err)!=FG_OK);
        fg_mtp_head_partial partial={.kind=FG_MTP_HEAD_DRAFT,.source_rank=1u,
            .destination_rank=0u,.first_token=200u,.count=n},parsed;
        for(uint32_t i=0;i<n;i++){partial.ids[i]=1000u+i;partial.values[i]=(float)i-9.0f;}
        CHECK(fg_mtp_head_partial_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&partial,&err)==FG_OK);
        CHECK(fg_mtp_head_partial_decode(&parsed,wire,bytes,&err)==FG_OK);
        CHECK(parsed.count==n&&!memcmp(parsed.ids,partial.ids,n*sizeof(uint32_t))&&
              !memcmp(parsed.values,partial.values,n*sizeof(float)));
        wire[12u]=1u;
        CHECK(fg_mtp_head_partial_decode(&parsed,wire,bytes,&err)!=FG_OK);
        partial.ids[0]=FG_Q38_VOCAB_SIZE;
        CHECK(fg_mtp_head_partial_encode(wire,FG_MTP_NATIVE_WORK_MAX_BYTES,&bytes,&partial,&err)!=FG_OK);
        fg_mtp_verify_result result={.source_rank=7u,.first_token=123u,.count=n},read;
        CHECK(fg_mtp_verify_result_encode(wire,&result,&err)==FG_OK);
        CHECK(fg_mtp_verify_result_decode(&read,wire,FG_MTP_VERIFY_RESULT_BYTES,&err)==FG_OK);
        CHECK(read.first_token==123u&&read.count==n&&read.source_rank==7u);
        wire[2u]=1u;
        CHECK(fg_mtp_verify_result_decode(&read,wire,FG_MTP_VERIFY_RESULT_BYTES,&err)!=FG_OK);
    }
    fg_mtp_native_ack ack={.operation=FG_MTP_NATIVE_FINISH,.first_token=70u,
        .count=3u,.frontier=72u},read_ack;
    CHECK(fg_mtp_native_ack_encode(wire,&ack,&err)==FG_OK);
    CHECK(fg_mtp_native_ack_decode(&read_ack,wire,FG_MTP_NATIVE_ACK_BYTES,&err)==FG_OK);
    CHECK(read_ack.frontier==72u&&read_ack.count==3u);
    ack.status=FG_ERR_LIMIT;
    CHECK(fg_mtp_native_ack_encode(wire,&ack,&err)!=FG_OK);
    ack=(fg_mtp_native_ack){.operation=FG_MTP_NATIVE_PREPARE,.status=FG_ERR_LIMIT};
    CHECK(fg_mtp_native_ack_encode(wire,&ack,&err)==FG_OK);
    CHECK(fg_mtp_native_ack_decode(&read_ack,wire,FG_MTP_NATIVE_ACK_BYTES,&err)==FG_OK);
    CHECK(read_ack.status==FG_ERR_LIMIT);
    wire[2]=255u;
    CHECK(fg_mtp_native_ack_decode(&read_ack,wire,FG_MTP_NATIVE_ACK_BYTES,&err)!=FG_OK);
    for(uint32_t sessions=0u;sessions<=3u;sessions++){
        fg_mtp_native_work prepare={.operation=FG_MTP_NATIVE_PREPARE,.argument=sessions};
        fg_status status=fg_mtp_native_work_encode(wire,capacity,&bytes,&prepare,&err);
        CHECK((status==FG_OK)==(sessions==1u||sessions==2u));
    }
    for(uint32_t type=FG_MSG_MTP_VERIFY_WORK;type<=FG_MSG_MTP_TARGET_ACK;type++){
        fg_frame_header header;
        CHECK(fg_frame_encode_version(&header,6u,(fg_message_type)type,22u,1u,
            1u<<24u,wire,FG_MTP_NATIVE_ACK_BYTES,&err)==FG_OK);
        CHECK(fg_frame_validate(&header,wire,NULL,&err)==FG_OK);
        CHECK(fg_frame_session_slot(&header)==1u);
        CHECK(fg_frame_encode_version(&header,5u,(fg_message_type)type,22u,1u,
            0u,wire,FG_MTP_NATIVE_ACK_BYTES,&err)!=FG_OK);
    }
    memset(wire,0,FG_PREFILL_LAYER_HEADER_BYTES);wire[9]=4u;
    bytes=FG_PREFILL_LAYER_HEADER_BYTES;ids[3]=UINT32_MAX;
    CHECK(fg_mtp_prefill_tokens_encode(wire,capacity,&bytes,ids,4u,&err)==FG_OK);
    uint32_t body_bytes=0u;
    CHECK(fg_mtp_prefill_tokens_decode(decoded_ids,4u,&body_bytes,wire,bytes,&err)==FG_OK);
    CHECK(body_bytes==FG_PREFILL_LAYER_HEADER_BYTES&&!memcmp(ids,decoded_ids,sizeof(ids)));
    CHECK(fg_mtp_prefill_tokens_decode(decoded_ids,3u,&body_bytes,wire,bytes,&err)!=FG_OK);
    ids[0]=UINT32_MAX;bytes=FG_PREFILL_LAYER_HEADER_BYTES;
    CHECK(fg_mtp_prefill_tokens_encode(wire,capacity,&bytes,ids,4u,&err)!=FG_OK);
    free(output);free(input);free(wire);
    if(!failures)puts("MTP typed wire, finite payloads, bounds and session-slot frames: PASS");
    return failures?1:0;
}
