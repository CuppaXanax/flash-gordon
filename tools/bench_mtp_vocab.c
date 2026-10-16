#include "fg_mtp_head.h"
#include "fg_q38_schema.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static fg_error error;
static void check(fg_status status){
    if(status!=FG_OK){fprintf(stderr,"draft vocab bench: %s\n",error.message);exit(1);}
}
static void need(bool ok){
    if(!ok){fprintf(stderr,"draft vocab bench: file/reference failure\n");exit(1);}
}
static void le32(uint8_t *p,uint32_t n){
    for(uint32_t i=0;i<4u;i++)p[i]=(uint8_t)(n>>(8u*i));
}
static double now(void){
    struct timespec t;need(!clock_gettime(CLOCK_MONOTONIC,&t));
    return t.tv_sec*1000.+t.tv_nsec*1e-6;
}
static int compare(const void *a,const void *b){
    double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: bench-mtp-vocab MANIFEST FULL_RAW_HEAD_Q8\n");return 2;}
    fg_manifest *manifest=malloc(sizeof(*manifest));need(manifest!=NULL);
    check(fg_manifest_read(argv[1],manifest,&error));
    const fg_tensor_record *record=NULL;
    for(uint32_t i=0;i<manifest->tensor_count;i++)
        if(!strcmp(manifest->tensors[i].name,"output.weight"))record=&manifest->tensors[i];
    need(record!=NULL);
    printf("{\"kind\":\"draft_vocab_parent\",\"cooked_sha256\":\"");
    for(uint32_t i=0;i<32u;i++)printf("%02x",record->sha256[i]);
    printf("\"}\n");fflush(stdout);
    fg_vk_context *vk=NULL;check(fg_vk_open(&vk,&error));
    fg_vk_tensor *hidden=NULL;check(fg_vk_tensor_create(vk,FG_HIDDEN_SIZE*4u,&hidden,&error));
    float input[8][FG_HIDDEN_SIZE];
    for(uint32_t j=0;j<8u;j++)for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)
        input[j][i]=sinf(i*.013f+j*.27f)*.13f;
    const uint32_t sizes[]={FG_Q38_VOCAB_SIZE,40525u,106299u};
    double full_ms=0.;uint64_t full_bytes=0u;
    for(uint32_t shape=0;shape<3u;shape++){
        uint32_t rows=sizes[shape],*ids=malloc((size_t)rows*4u);need(ids!=NULL);
        for(uint32_t i=0;i<rows;i++)ids[i]=i;
        char path[128];snprintf(path,sizeof(path),".build/vocab-bench-%ld.fgdv",(long)getpid());
        if(shape){
            FILE *src=fopen(argv[2],"rb"),*dst=fopen(path,"wbx");need(src&&dst);
            uint8_t header[44]={0};memcpy(header,"FGDVOC1\0",8u);le32(header+8u,rows);
            memcpy(header+12u,record->sha256,32u);need(fwrite(header,1,sizeof(header),dst)==sizeof(header));
            for(uint32_t i=0;i<rows;i++){uint8_t id[4];le32(id,i);need(fwrite(id,1,4u,dst)==4u);}
            uint8_t buffer[65536];size_t got;
            while((got=fread(buffer,1,sizeof(buffer),src)))need(fwrite(buffer,1,got,dst)==got);
            need(!ferror(src)&&!fclose(src)&&!fclose(dst));
        }
        fg_vk_memory_stats before,after;fg_vk_get_memory_stats(vk,&before);
        fg_mtp_vocab *vocab=NULL;
        check(fg_mtp_vocab_open_draft(&vocab,vk,shape?path:argv[2],record,&error));
        fg_vk_get_memory_stats(vk,&after);
        uint64_t bytes=after.requested_live_bytes-before.requested_live_bytes;
        uint32_t tokens[8];float scores[8];double timing[100];
        for(uint32_t i=0;i<108u;i++){
            check(fg_vk_tensor_write(hidden,0,input[i%8u],sizeof(input[0]),&error));
            double start=now();
            check(fg_mtp_vocab_argmax(vocab,hidden,&tokens[i%8u],&scores[i%8u],&error));
            if(i>=8u)timing[i-8u]=now()-start;
        }
        qsort(timing,100u,sizeof(double),compare);
        double mean=0.;for(uint32_t i=0;i<100u;i++)mean+=timing[i]/100.;
        fg_mtp_vocab_close(vocab);vocab=NULL;
        check(fg_mtp_vocab_open(&vocab,vk,argv[2],shape?ids:NULL,rows,&error));
        for(uint32_t i=0;i<8u;i++){
            uint32_t token;float score;
            check(fg_vk_tensor_write(hidden,0,input[i],sizeof(input[0]),&error));
            check(fg_mtp_vocab_argmax(vocab,hidden,&token,&score,&error));
            need(token==tokens[i]&&score==scores[i]);
        }
        fg_mtp_vocab_close(vocab);
        if(!shape){full_ms=mean;full_bytes=bytes;}
        printf("{\"kind\":\"draft_vocab_bench\",\"rows\":%u,\"reported_cus\":%u,"
            "\"mean_ms\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"resident_bytes\":%llu,"
            "\"saved_bytes\":%llu,\"three_call_saving_proxy_ms\":%.6f,"
            "\"independent_loader_ids_scores_exact\":true,"
            "\"subset_kind\":\"contiguous-low-IDs smoke test, not curated or acceptance-qualified\"}\n",
            rows,fg_vk_active_compute_units(vk),mean,(timing[49]+timing[50])*.5,timing[94],
            (unsigned long long)bytes,(unsigned long long)(full_bytes-bytes),3.*(full_ms-mean));
        fflush(stdout);if(shape)need(!unlink(path));free(ids);
    }
    fg_vk_tensor_destroy(hidden);fg_vk_close(vk);free(manifest);return 0;
}
