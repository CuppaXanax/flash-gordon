#include "fg_fabric.h"
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
    if(argc!=2)return 2;
    uint32_t rank=(uint32_t)strtoul(argv[1],NULL,10);
    fg_manifest manifest={0};manifest.protocol_version=FG_PROTOCOL_VERSION;
    for(uint32_t i=0;i<FG_RANK_COUNT;i++)
        snprintf(manifest.ranks[i].endpoint,sizeof(manifest.ranks[i].endpoint),
                 "192.168.42.%u:%u",(i&1)?53:52,19600+i*2);
    fg_error error={0};fg_fabric *fabric=NULL;
    fg_status status=fg_fabric_open(&fabric,&manifest,rank,&error);
    enum{SAMPLES=1000,WARM=100,MAX_BYTES=51224};
    uint8_t payload[MAX_BYTES],returned[MAX_BYTES];
    for(unsigned i=0;i<MAX_BYTES;i++)payload[i]=(uint8_t)(i*29u+7u);
    uint32_t sizes[]={64,20480,40960,40984,51224};
    uint32_t received=0;fg_frame_header header;
    if(rank>1){
        if(status==FG_OK)status=fg_fabric_recv(fabric,0,FG_FABRIC_CONTROL,&header,
                                              returned,sizeof(returned),&received,&error);
    }else for(unsigned size=0;status==FG_OK&&size<sizeof(sizes)/sizeof(*sizes);size++){
        double samples[SAMPLES];uint32_t bytes=sizes[size];
        for(unsigned i=0;status==FG_OK&&i<SAMPLES+WARM;i++){
            double start=now_us();
            if(rank==0)status=fg_fabric_send(fabric,1,FG_FABRIC_BULK,FG_MSG_DECODE_LAYER_WORK,
                                            7,i,0,payload,bytes,&error);
            if(status==FG_OK)status=fg_fabric_recv(fabric,rank?0:1,FG_FABRIC_BULK,&header,
                                                   returned,sizeof(returned),&received,&error);
            if(status==FG_OK&&(received!=bytes||fg_frame_sequence(&header)!=i||
                fg_frame_request_id(&header)!=7||memcmp(payload,returned,bytes))){
                fg_error_set(&error,FG_ERR_MISMATCH,"fabric latency payload mismatch");status=FG_ERR_MISMATCH;
            }
            if(status==FG_OK&&rank==1)status=fg_fabric_send(fabric,0,FG_FABRIC_BULK,
                FG_MSG_DECODE_LAYER_WORK,7,i,0,returned,bytes,&error);
            if(rank==0&&i>=WARM)samples[i-WARM]=now_us()-start;
        }
        if(status==FG_OK&&rank==0){
            qsort(samples,SAMPLES,sizeof(*samples),cmp);
            printf("{\"payload_bytes\":%u,\"frame_bytes\":%zu,\"samples\":%u,\"rtt_p50_us\":%.3f,\"rtt_p99_us\":%.3f,\"hop_rtt_over_2_p50_us\":%.3f,\"hop_rtt_over_2_p99_us\":%.3f,\"exact\":true}\n",
                bytes,bytes+sizeof(fg_frame_header),SAMPLES,samples[SAMPLES/2],
                samples[SAMPLES*99/100],samples[SAMPLES/2]/2,samples[SAMPLES*99/100]/2);
            fflush(stdout);
        }
    }
    if(rank==0)for(uint32_t peer=2;status==FG_OK&&peer<FG_RANK_COUNT;peer++)
        status=fg_fabric_send(fabric,peer,FG_FABRIC_CONTROL,FG_MSG_CONTROL,7,0,0,NULL,0,&error);
    if(status!=FG_OK)fprintf(stderr,"rank %u: %s\n",rank,error.message);
    fg_fabric_close(fabric);return status==FG_OK?0:1;
}
