#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { HEADS=16, SAMPLES=1000 };
static const uint64_t rows=320001536, row_bytes=90;
static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec/1e3;
}
static uint64_t rng=1;
static uint64_t random64(void) {
    rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng;
}
static void fail(const char *text) {perror(text);exit(1);}
static int cmp(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y);
}
typedef struct pool {
    pthread_mutex_t mutex;
    pthread_cond_t ready,done;
    unsigned generation,pending;
    int fd,stop;
    uint64_t offsets[HEADS];
    unsigned char *buffers[HEADS];
    int errors[HEADS];
} pool;
typedef struct worker {pool *p; unsigned index;} worker;
static void *run(void *opaque) {
    worker *w=opaque;pool *p=w->p;unsigned seen=0,i=w->index;
    pthread_mutex_lock(&p->mutex);
    for(;;) {
        while(!p->stop&&seen==p->generation)pthread_cond_wait(&p->ready,&p->mutex);
        if(p->stop)break;
        seen=p->generation;uint64_t off=p->offsets[i];int fd=p->fd;
        pthread_mutex_unlock(&p->mutex);
        ssize_t n; do {n=pread(fd,p->buffers[i],8192,(off_t)off);}while(n<0&&errno==EINTR);
        pthread_mutex_lock(&p->mutex);
        p->errors[i]=n!=8192;
        if(!--p->pending)pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->mutex);return NULL;
}
static void batch(pool *p,int parallel) {
    if(parallel) {
        pthread_mutex_lock(&p->mutex);p->generation++;p->pending=HEADS;
        pthread_cond_broadcast(&p->ready);
        while(p->pending)pthread_cond_wait(&p->done,&p->mutex);
        pthread_mutex_unlock(&p->mutex);
    } else for(unsigned i=0;i<HEADS;i++)
        p->errors[i]=pread(p->fd,p->buffers[i],8192,(off_t)p->offsets[i])!=8192;
    for(unsigned i=0;i<HEADS;i++)if(p->errors[i])fail("pread");
}
int main(int argc,char **argv) {
    if(argc<2||argc>3)return 2;
    uint64_t bytes=(rows*row_bytes+4095)&~UINT64_C(4095);
    int fd=open(argv[1],O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
    if(fd<0)fail("create synthetic table");
    unsigned char *data=malloc(1u<<20);if(!data)fail("malloc");
    for(unsigned i=0;i<(1u<<20);i++)data[i]=(unsigned char)(i*29u+(i>>8u)*7u+13u);
    for(uint64_t offset=0;offset<bytes;) {
        size_t n=bytes-offset>(1u<<20)?(1u<<20):(size_t)(bytes-offset);
        ssize_t got=write(fd,data,n);if(got<=0)fail("materialize");offset+=(uint64_t)got;
    }
    if(fsync(fd))fail("fsync");
    free(data);
    printf("{\"materialized_bytes\":%llu,\"synthetic\":true}\n",(unsigned long long)bytes);fflush(stdout);
    pool p={.mutex=PTHREAD_MUTEX_INITIALIZER,.ready=PTHREAD_COND_INITIALIZER,
            .done=PTHREAD_COND_INITIALIZER};
    worker workers[HEADS];pthread_t threads[HEADS];
    for(unsigned i=0;i<HEADS;i++) {
        if(posix_memalign((void **)&p.buffers[i],4096,8192))fail("aligned buffer");
        workers[i]=(worker){&p,i};
        if(pthread_create(&threads[i],NULL,run,&workers[i]))fail("pthread_create");
    }
    uint64_t hot[HEADS];
    for(unsigned i=0;i<HEADS;i++)hot[i]=((random64()%rows)*row_bytes)&~UINT64_C(4095);
    for(int direct=0;direct<2;direct++)for(int parallel=0;parallel<2;parallel++)
    for(unsigned hit=0;hit<=100;hit+=25) {
        int readfd=open(argv[1],O_RDONLY|O_CLOEXEC|(direct?O_DIRECT:0));
        if(readfd<0)fail("open test");
        p.fd=readfd;double samples[SAMPLES];
        for(unsigned iteration=0;iteration<SAMPLES+32;iteration++) {
            for(unsigned i=0;i<HEADS;i++) {
                int cached=(random64()%100)<hit;
                p.offsets[i]=cached?hot[i]:((random64()%rows)*row_bytes)&~UINT64_C(4095);
                if(p.offsets[i]+8192>bytes)p.offsets[i]=bytes-8192;
                if(!cached) {
                    int error=posix_fadvise(fd,(off_t)p.offsets[i],8192,POSIX_FADV_DONTNEED);
                    if(error){errno=error;fail("fadvise");}
                }
            }
            double start=now_us();batch(&p,parallel);double elapsed=now_us()-start;
            if(iteration>=32)samples[iteration-32]=elapsed;
        }
        qsort(samples,SAMPLES,sizeof(*samples),cmp);
        printf("{\"direct\":%s,\"qd\":%d,\"requested_hot_percent\":%u,\"samples\":%d,\"p50_us\":%.3f,\"p99_us\":%.3f}\n",
            direct?"true":"false",parallel?16:1,hit,SAMPLES,samples[SAMPLES/2],samples[(SAMPLES*99)/100]);
        fflush(stdout);close(readfd);
    }
    pthread_mutex_lock(&p.mutex);p.stop=1;pthread_cond_broadcast(&p.ready);pthread_mutex_unlock(&p.mutex);
    for(unsigned i=0;i<HEADS;i++){pthread_join(threads[i],NULL);free(p.buffers[i]);}
    close(fd);if(argc==2&&unlink(argv[1]))fail("unlink synthetic table");
    return 0;
}
