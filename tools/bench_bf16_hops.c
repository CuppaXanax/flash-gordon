#include "fg_fabric.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;
}
static int cmp(const void *a,const void *b){
    double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
int main(int argc,char **argv){
    if(argc!=2&&argc!=3)return 2;
    const char *codec=argc==3?argv[2]:"bf16";
    if(strcmp(codec,"bf16")&&strcmp(codec,"fp16")&&strcmp(codec,"fp16-128"))return 2;
    uint32_t rank=(uint32_t)strtoul(argv[1],NULL,10);
    if(rank>=FG_RANK_COUNT)return 2;
    fg_manifest manifest={0};manifest.protocol_version=FG_PROTOCOL_VERSION;
    for(uint32_t i=0;i<FG_RANK_COUNT;i++)
        snprintf(manifest.ranks[i].endpoint,sizeof(manifest.ranks[i].endpoint),
                 "192.168.42.%u:%u",(i&1)?53:52,19800+i*2);
    fg_error error={0};fg_fabric *fabric=NULL;
    fg_status status=fg_fabric_open(&fabric,&manifest,rank,&error);
    enum{SAMPLES=500,WARM=50};
    uint8_t wire[FG_LAYER_WORK_MAX_BYTES],returned[FG_LAYER_WORK_MAX_BYTES];
    fg_layer_work work={.layer=1,.source_rank=0,.destination_rank=1,
        .token_index=9,.position={9,9,9,0}},decoded,expected;
    for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)work.hyper[i]=(float)((int)(i%71)-35)/7.0f;
    for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)
        work.ngram_embedding[i]=(float)((int)(i%97)-48)/11.0f;
    uint32_t received=0;fg_frame_header header;
    if(rank>1){
        if(status==FG_OK)status=fg_fabric_recv(fabric,0,FG_FABRIC_CONTROL,&header,
                                              returned,sizeof(returned),&received,&error);
    }else for(unsigned round=0;status==FG_OK&&round<5;round++)
      for(unsigned shape=0;status==FG_OK&&shape<2;shape++)
      for(unsigned leg=0;status==FG_OK&&leg<3;leg++){
        bool compressed=((round&1u)?leg!=1u:leg==1u);
        bool bf16=compressed&&!strcmp(codec,"bf16");
        work.flags=(shape^(round&1u))?FG_LAYER_WORK_HAS_NGRAM:0u;
        setenv("FG_BF16_HOPS",bf16?"1":"0",1);
        setenv("FG_FP16_HOPS",compressed&&!strcmp(codec,"fp16")?"1":
            compressed&&!strcmp(codec,"fp16-128")?"128":"0",1);
        unsetenv("FG_FP16_HOP_MASK");
        uint32_t bytes=0;double samples[SAMPLES],sum=0;
        status=fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&work,&error);
        if(status==FG_OK)status=fg_decode_layer_work_decode(&expected,6,wire,bytes,&error);
        for(unsigned i=0;status==FG_OK&&i<SAMPLES+WARM;i++){
            double start=now_us();
            if(rank==0){
                status=fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&work,&error);
                if(status==FG_OK)status=fg_fabric_send(fabric,1,FG_FABRIC_BULK,
                    FG_MSG_DECODE_LAYER_WORK,7,i,0,wire,bytes,&error);
            }
            if(status==FG_OK)status=fg_fabric_recv(fabric,rank?0:1,FG_FABRIC_BULK,
                &header,returned,sizeof(returned),&received,&error);
            if(status==FG_OK)status=fg_decode_layer_work_decode(&decoded,6,
                returned,received,&error);
            if(status==FG_OK&&(received!=bytes||fg_frame_sequence(&header)!=i||
                fg_frame_request_id(&header)!=7||memcmp(&decoded,&expected,sizeof(decoded)))){
                fg_error_set(&error,FG_ERR_MISMATCH,"bf16 hop payload mismatch");status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK&&rank==1){
                status=fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&decoded,&error);
                if(status==FG_OK)status=fg_fabric_send(fabric,0,FG_FABRIC_BULK,
                    FG_MSG_DECODE_LAYER_WORK,7,i,0,wire,bytes,&error);
            }
            if(rank==0&&i>=WARM){samples[i-WARM]=now_us()-start;sum+=samples[i-WARM];}
        }
        if(status==FG_OK&&rank==0){
            qsort(samples,SAMPLES,sizeof(*samples),cmp);
            printf("{\"round\":%u,\"leg\":%u,\"ngram\":%s,\"bf16\":%s,"
                "\"compressed\":%s,\"codec\":\"%s\","
                "\"payload_bytes\":%u,\"frame_bytes\":%zu,\"samples\":%u,"
                "\"rtt_mean_us\":%.6f,\"rtt_p50_us\":%.6f,\"rtt_p99_us\":%.6f,"
                "\"hop_rtt_over_2_p50_us\":%.6f,\"codec_validated\":true}\n",
                round,leg,work.flags?"true":"false",bf16?"true":"false",
                compressed?"true":"false",compressed?codec:"f32",
                bytes,bytes+sizeof(fg_frame_header),SAMPLES,sum/SAMPLES,
                samples[SAMPLES/2],samples[SAMPLES*99/100],samples[SAMPLES/2]/2);
            fflush(stdout);
        }
    }
    if(rank==0)for(uint32_t peer=2;status==FG_OK&&peer<FG_RANK_COUNT;peer++)
        status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_CONTROL,7,0,0,NULL,0,&error);
    if(status!=FG_OK)fprintf(stderr,"rank %u: %s\n",rank,error.message);
    fg_fabric_close(fabric);return status==FG_OK?0:1;
}
