#include "fg_protocol.h"
#include "fg_q38_schema.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static double now_us(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1e6+(double)t.tv_nsec/1e3;
}
static int compare(const void *a,const void *b){
    double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
static int transfer(int fd,void *data,size_t bytes,int sending){
    unsigned char *p=data;
    while(bytes){
        ssize_t n=sending?send(fd,p,bytes,MSG_NOSIGNAL):recv(fd,p,bytes,0);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return -1;
        p+=n;bytes-=(size_t)n;
    }
    return 0;
}
int main(int argc,char **argv){
    if(argc!=4)return 2;
    int server=!strcmp(argv[1],"server"),fd=-1,listener=-1,one=1;
    struct sockaddr_in address={.sin_family=AF_INET,.sin_port=htons((uint16_t)atoi(argv[3]))};
    if(inet_pton(AF_INET,argv[2],&address.sin_addr)!=1)return 2;
    if(server){
        listener=socket(AF_INET,SOCK_STREAM,0);
        if(listener<0)return 1;
        setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
        if(bind(listener,(void *)&address,sizeof(address))||listen(listener,1))return 1;
        fd=accept(listener,NULL,NULL);close(listener);
    }else{
        for(unsigned attempt=0;attempt<100;attempt++){
            fd=socket(AF_INET,SOCK_STREAM,0);
            if(fd>=0&&!connect(fd,(void *)&address,sizeof(address)))break;
            if(fd>=0)close(fd);
            fd=-1;struct timespec pause={.tv_nsec=100000000};nanosleep(&pause,NULL);
        }
    }
    if(fd<0)return 1;
    setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
    enum{CAPACITY=300000,SAMPLES=100,WARM=10};
    uint8_t *wire=malloc(CAPACITY),*received=malloc(CAPACITY);
    float *hyper=malloc(4u*FG_HYPER_WIDTH*sizeof(float));
    float *ngram=malloc(4u*FG_NGRAM_EMBED_VALUES*sizeof(float));
    float *decoded_hyper=malloc(4u*FG_HYPER_WIDTH*sizeof(float));
    float *decoded_ngram=malloc(4u*FG_NGRAM_EMBED_VALUES*sizeof(float));
    if(!wire||!received||!hyper||!ngram||!decoded_hyper||!decoded_ngram)return 1;
    uint32_t positions[12],decoded_positions[12];
    for(unsigned i=0;i<4u*FG_HYPER_WIDTH;i++)hyper[i]=(float)((int)(i%71)-35)/7.0f;
    for(unsigned i=0;i<4u*FG_NGRAM_EMBED_VALUES;i++)ngram[i]=(float)((int)(i%97)-48)/11.0f;
    for(unsigned i=0;i<12;i++)positions[i]=9u+i/3u;
    fg_error error={0};fg_status status=FG_OK;unsigned sequence=0;
    for(unsigned round=0;status==FG_OK&&round<5;round++)
    for(unsigned n=1;status==FG_OK&&n<=4;n++)
    for(unsigned initial=0;status==FG_OK&&initial<2;initial++){
        fg_prefill_layer_work work={.layer=initial?0u:6u,.source_rank=0,
            .destination_rank=1,.flags=initial?FG_LAYER_WORK_HAS_NGRAM:0,
            .first_token=9,.token_count=(uint16_t)n,.positions=positions,
            .hyper=hyper,.ngram_embeddings=initial?ngram:NULL};
        uint32_t bytes=0;double samples[SAMPLES],sum=0;
        for(unsigned sample=0;status==FG_OK&&sample<SAMPLES+WARM;sample++,sequence++){
            double start=now_us();fg_frame_header frame;
            status=fg_prefill_layer_work_encode(wire,CAPACITY,&bytes,6,&work,&error);
            if(status==FG_OK)status=fg_frame_encode_version(&frame,6,FG_MSG_MTP_VERIFY_WORK,
                7,sequence,0,wire,bytes,&error);
            if(!server&&status==FG_OK&&(transfer(fd,&frame,sizeof(frame),1)||transfer(fd,wire,bytes,1)))status=FG_ERR_IO;
            fg_frame_header reply;
            if(status==FG_OK&&(transfer(fd,&reply,sizeof(reply),0)||transfer(fd,received,bytes,0)))status=FG_ERR_IO;
            uint32_t validated=0;
            if(status==FG_OK)status=fg_frame_validate_version(&reply,6,received,&validated,&error);
            fg_prefill_layer_work decoded;
            if(status==FG_OK)status=fg_prefill_layer_work_decode(&decoded,6,decoded_positions,12,
                decoded_hyper,4u*FG_HYPER_WIDTH,decoded_ngram,4u*FG_NGRAM_EMBED_VALUES,
                received,validated,&error);
            if(status==FG_OK&&(validated!=bytes||fg_frame_sequence(&reply)!=sequence||
                fg_frame_request_id(&reply)!=7||memcmp(received,wire,bytes)))status=FG_ERR_MISMATCH;
            if(server&&status==FG_OK&&(transfer(fd,&frame,sizeof(frame),1)||transfer(fd,wire,bytes,1)))status=FG_ERR_IO;
            if(!server&&sample>=WARM){samples[sample-WARM]=now_us()-start;sum+=samples[sample-WARM];}
        }
        if(!server&&status==FG_OK){
            qsort(samples,SAMPLES,sizeof(double),compare);
            printf("{\"round\":%u,\"n\":%u,\"initial\":%s,\"payload_bytes\":%u,"
                "\"frame_bytes\":%zu,\"samples\":%u,\"rtt_mean_us\":%.6f,"
                "\"rtt_p50_us\":%.6f,\"rtt_p99_us\":%.6f,\"validated\":true}\n",
                round,n,initial?"true":"false",bytes,bytes+sizeof(fg_frame_header),
                SAMPLES,sum/SAMPLES,samples[SAMPLES/2],samples[SAMPLES*99/100]);
            fflush(stdout);
        }
    }
    if(status!=FG_OK)fprintf(stderr,"transport: status=%d %s\n",status,error.message);
    close(fd);free(wire);free(received);free(hyper);free(ngram);free(decoded_hyper);free(decoded_ngram);
    return status==FG_OK?0:1;
}
