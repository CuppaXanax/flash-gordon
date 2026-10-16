#include "fg_ngram.h"
#include "fg_q38_schema.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_us(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;
}
static int cmp(const void *a,const void *b){
    double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
int main(int argc,char **argv){
    int warm_only=argc==3&&strcmp(argv[2],"--warm-only")==0;
    if(argc!=2&&!warm_only){fprintf(stderr,"bench-ngram-latency table.iq4nl [--warm-only]\n");return 2;}
    enum{SAMPLES=1000,WARM=32,VALUES=FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH};
    const uint64_t table_bytes=UINT64_C(320001536)*FG_NGRAM_ROW_BYTES;
    fg_error error={0};fg_vk_context *vk=NULL;
    if(warm_only){
        int fixture=open(argv[1],O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
        if(fixture<0)return 1;
        int ok=ftruncate(fixture,(off_t)fg_align_up_u64(table_bytes,FG_ALIGNMENT))==0;
        int32_t history[]={100,101,102};
        uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];
        if(ok)ok=fg_q38_ngram_lookup(history,3,rows,addresses,&error)==FG_OK;
        for(unsigned head=0;ok&&head<FG_NGRAM_HEAD_COUNT;head++){
            uint8_t packed[FG_NGRAM_ROW_BYTES];
            for(unsigned block=0;block<FG_NGRAM_EMBED_WIDTH/32;block++){
                packed[block*18]=0;packed[block*18+1]=0x38;
                for(unsigned j=0;j<16;j++)packed[block*18+2+j]=(uint8_t)(head*17+j*13+block);
            }
            ok=pwrite(fixture,packed,sizeof(packed),(off_t)addresses[head])==(ssize_t)sizeof(packed);
        }
        if(ok)ok=fsync(fixture)==0;
        close(fixture);
        if(!ok){unlink(argv[1]);return 1;}
    }
    if(fg_vk_open(&vk,&error)!=FG_OK){fprintf(stderr,"%s\n",error.message);return 1;}
    int fd=open(argv[1],O_RDONLY|O_CLOEXEC);
    float *reference=malloc((SAMPLES+WARM)*(size_t)VALUES*sizeof(float));
    int32_t (*histories)[3]=malloc((SAMPLES+WARM)*sizeof(*histories));
    if(fd<0||!reference||!histories)return 1;
    int ok=1;
    const unsigned reuse_levels[]={0,50,90,100};
    unsigned rounds=warm_only?3:1;
    for(unsigned round=0;ok&&round<rounds;round++)
    for(unsigned profile=warm_only?3:0;ok&&profile<sizeof(reuse_levels)/sizeof(*reuse_levels);profile++){
        uint32_t state=17;
        for(unsigned i=0;i<SAMPLES+WARM;i++){
            state=state*1664525u+1013904223u;
            int reuse=(state%100)<reuse_levels[profile];
            for(unsigned j=0;j<3;j++){
                state=state*1664525u+1013904223u;
                histories[i][j]=reuse?(int32_t)(100u+j):(int32_t)(state%240000u);
            }
        }
        for(unsigned mode=0;ok&&mode<(warm_only?3u:2u);mode++){
            int candidate=mode==1;
            if(setenv("FG_NGRAM_LOCAL_CACHE",candidate?"1":"0",1))return 1;
            fg_ngram_store *store=NULL;
            ok=fg_ngram_store_open(&store,vk,argv[1],table_bytes,1,&error)==FG_OK;
            double samples[SAMPLES];float got[VALUES];
            for(unsigned i=0;ok&&i<SAMPLES+WARM;i++){
                uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];
                ok=fg_q38_ngram_lookup(histories[i],3,rows,addresses,&error)==FG_OK;
                for(unsigned head=0;ok&&head<FG_NGRAM_HEAD_COUNT;head++){
                    uint64_t block=addresses[head]&~UINT64_C(4095);
                    uint64_t length=block+8192>table_bytes?4096:8192;
                    ok=posix_fadvise(fd,(off_t)block,(off_t)length,POSIX_FADV_DONTNEED)==0;
                }
                fg_vk_tensor *embedding=NULL;double start=now_us();
                if(ok)ok=fg_ngram_store_lookup(store,histories[i],3,&embedding,&error)==FG_OK;
                if(ok)ok=fg_vk_tensor_read(embedding,0,got,sizeof(got),&error)==FG_OK;
                double elapsed=now_us()-start;
                if(i>=WARM)samples[i-WARM]=elapsed;
                if(ok&&mode==0)memcpy(reference+(size_t)i*VALUES,got,sizeof(got));
                if(ok&&mode!=0)ok=memcmp(reference+(size_t)i*VALUES,got,sizeof(got))==0;
            }
            if(ok){
                qsort(samples,SAMPLES,sizeof(*samples),cmp);
                printf("{\"path\":\"fg_ngram_store_lookup\",\"buffered\":%s,\"token_reuse_percent\":%u,\"samples\":%u,\"round\":%u,\"phase\":%u,\"p50_us\":%.3f,\"p99_us\":%.3f,\"exact_embedding_bytes\":true,\"forced_os_cache_misses\":true,\"synthetic\":true,\"warm_only_sparse_fixture\":%s}\n",
                    candidate?"true":"false",reuse_levels[profile],SAMPLES,round,mode,
                    samples[SAMPLES/2],samples[SAMPLES*99/100],warm_only?"true":"false");
                fflush(stdout);
            }
            fg_ngram_store_close(store);
        }
    }
    if(!ok)fprintf(stderr,"ngram latency/parity failure: %s\n",error.message);
    close(fd);free(histories);free(reference);fg_vk_close(vk);
    if(warm_only)unlink(argv[1]);
    return ok?0:1;
}
