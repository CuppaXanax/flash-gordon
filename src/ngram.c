#include "fg_ngram.h"
#include "fg_uring.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CACHE_BLOCKS (FG_NGRAM_CACHE_BYTES/FG_NGRAM_BLOCK_BYTES)
#define CACHE_WAYS 4u
#define CACHE_SETS (CACHE_BLOCKS/CACHE_WAYS)
typedef struct cache_entry{uint64_t offset;uint64_t stamp;bool valid;}cache_entry;
struct fg_ngram_cache{uint8_t *data;cache_entry *entry;uint64_t stamp;};
struct fg_ngram_store{int fd;fg_uring *ring;uint32_t slot;uint64_t table_bytes;uint8_t *io_buffer;fg_ngram_cache *cache;fg_vk_context *vk;fg_vk_tensor *packed,*embedding,*embedding_view;uint32_t last_read_count;uint64_t last_read_bytes;double last_io_ms;};
struct fg_ngram_resident{uint8_t *data;uint64_t row_begin,row_count,bytes;};
static int u64_cmp(const void *a,const void *b){uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;return x<y?-1:x>y;}
static double ngram_ts(void){struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return (double)value.tv_sec*1e3+(double)value.tv_nsec*1e-6;}
static bool ngram_trace_enabled(void){const char *enabled=getenv("FG_FRAME_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
static bool ngram_locality_trace_enabled(void){const char *enabled=getenv("FG_NGRAM_LOCALITY_TRACE");return enabled&&*enabled&&strcmp(enabled,"0")!=0;}
fg_status fg_ngram_plan_reads(const uint64_t *addresses,uint32_t count,uint64_t table_bytes,fg_ngram_read *reads,uint32_t cap,uint32_t *out_count,fg_error *err){
    if((count&&!addresses)||!reads||!out_count||!table_bytes){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram read planner arguments");return FG_ERR_ARGUMENT;}if(count>FG_NGRAM_PREFILL_MAX_BLOCKS){fg_error_set(err,FG_ERR_LIMIT,"n-gram read planner input exceeds bounded prefill capacity");return FG_ERR_LIMIT;}
    uint64_t padded_bytes=fg_align_up_u64(table_bytes,FG_NGRAM_BLOCK_BYTES),*blocks=malloc((size_t)count*sizeof(*blocks));if(count&&!blocks){fg_error_set(err,FG_ERR_OOM,"allocate n-gram block planner");return FG_ERR_OOM;}
    for(uint32_t i=0;i<count;i++){if(addresses[i]>=table_bytes){free(blocks);fg_error_set(err,FG_ERR_FORMAT,"n-gram address is outside table");return FG_ERR_FORMAT;}blocks[i]=addresses[i]&~(uint64_t)(FG_NGRAM_BLOCK_BYTES-1u);}qsort(blocks,count,sizeof(*blocks),u64_cmp);
    uint32_t n=0;for(uint32_t i=0;i<count;){uint64_t off=blocks[i];while(i<count&&blocks[i]==off)i++;uint32_t bytes=FG_NGRAM_BLOCK_BYTES;if(i<count&&blocks[i]==off+FG_NGRAM_BLOCK_BYTES){bytes=FG_NGRAM_MAX_READ_BYTES;uint64_t second=blocks[i];while(i<count&&blocks[i]==second)i++;}if(off+bytes>padded_bytes){free(blocks);fg_error_set(err,FG_ERR_FORMAT,"n-gram direct-read plan exceeds padded table");return FG_ERR_FORMAT;}if(n>=cap){free(blocks);fg_error_set(err,FG_ERR_LIMIT,"n-gram read plan exceeds capacity");return FG_ERR_LIMIT;}reads[n++]=(fg_ngram_read){.offset=off,.bytes=bytes};}
    free(blocks);*out_count=n;return FG_OK;
}
fg_status fg_ngram_cache_create(fg_ngram_cache **out,fg_error *err){if(!out){fg_error_set(err,FG_ERR_ARGUMENT,"n-gram cache output is null");return FG_ERR_ARGUMENT;}*out=NULL;fg_ngram_cache *c=calloc(1,sizeof(*c));if(!c){fg_error_set(err,FG_ERR_OOM,"allocate n-gram cache");return FG_ERR_OOM;}if(posix_memalign((void **)&c->data,FG_ALIGNMENT,FG_NGRAM_CACHE_BYTES)!=0){free(c);fg_error_set(err,FG_ERR_OOM,"allocate 64 MiB aligned n-gram data cache");return FG_ERR_OOM;}c->entry=calloc(CACHE_BLOCKS,sizeof(*c->entry));if(!c->entry){free(c->data);free(c);fg_error_set(err,FG_ERR_OOM,"allocate n-gram cache metadata");return FG_ERR_OOM;}*out=c;return FG_OK;}
void fg_ngram_cache_destroy(fg_ngram_cache *c){if(!c)return;free(c->entry);free(c->data);free(c);}
static uint32_t cache_set(uint64_t off){uint64_t x=off/FG_NGRAM_BLOCK_BYTES;x^=x>>33;x*=UINT64_C(0xff51afd7ed558ccd);x^=x>>33;return (uint32_t)x&(CACHE_SETS-1u);}
bool fg_ngram_cache_get(fg_ngram_cache *c,uint64_t off,const void **data){if(!c||!data||!fg_is_aligned_u64(off,FG_NGRAM_BLOCK_BYTES))return false;uint32_t base=cache_set(off)*CACHE_WAYS;for(uint32_t w=0;w<CACHE_WAYS;w++){uint32_t i=base+w;if(c->entry[i].valid&&c->entry[i].offset==off){c->entry[i].stamp=++c->stamp;*data=c->data+(size_t)i*FG_NGRAM_BLOCK_BYTES;return true;}}return false;}
fg_status fg_ngram_cache_put(fg_ngram_cache *c,uint64_t off,const void *block,fg_error *err){if(!c||!block||!fg_is_aligned_u64(off,FG_NGRAM_BLOCK_BYTES)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid aligned n-gram cache block");return FG_ERR_ARGUMENT;}uint32_t base=cache_set(off)*CACHE_WAYS,victim=base;for(uint32_t w=0;w<CACHE_WAYS;w++){uint32_t i=base+w;if(c->entry[i].valid&&c->entry[i].offset==off){victim=i;goto store;}if(!c->entry[i].valid){victim=i;goto store;}if(c->entry[i].stamp<c->entry[victim].stamp)victim=i;}store:memcpy(c->data+(size_t)victim*FG_NGRAM_BLOCK_BYTES,block,FG_NGRAM_BLOCK_BYTES);c->entry[victim]=(cache_entry){.offset=off,.stamp=++c->stamp,.valid=true};return FG_OK;}

static uint64_t splitmix64(uint64_t value){
    value+=UINT64_C(0x9e3779b97f4a7c15);value=(value^(value>>30u))*UINT64_C(0xbf58476d1ce4e5b9);
    value=(value^(value>>27u))*UINT64_C(0x94d049bb133111eb);return value^(value>>31u);
}

static uint64_t signed_remainder_u64(uint64_t bits,uint64_t divisor){
    if(bits<=INT64_MAX)return bits%divisor;
    uint64_t magnitude=(~bits)+1u,rem=magnitude%divisor;
    return rem?divisor-rem:0;
}

static const uint32_t q38_ngram_vocab[FG_NGRAM_HEAD_COUNT]={20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171};
static const uint64_t q38_ngram_offset[FG_NGRAM_HEAD_COUNT]={0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275};

fg_status fg_q38_ngram_head_range(uint32_t head_begin,uint32_t head_count,uint64_t *row_begin,uint64_t *row_count,fg_error *err){if(!row_begin||!row_count||head_begin>=FG_NGRAM_HEAD_COUNT||!head_count||head_count>FG_NGRAM_HEAD_COUNT-head_begin){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Qwen3.8 n-gram head range");return FG_ERR_ARGUMENT;}uint32_t head_end=head_begin+head_count;uint64_t begin=q38_ngram_offset[head_begin],end=head_end<FG_NGRAM_HEAD_COUNT?q38_ngram_offset[head_end]:q38_ngram_offset[FG_NGRAM_HEAD_COUNT-1u]+q38_ngram_vocab[FG_NGRAM_HEAD_COUNT-1u];*row_begin=begin;*row_count=end-begin;return FG_OK;}

fg_status fg_q38_ngram_rank_range(uint32_t rank,uint64_t *row_begin,uint64_t *row_count,fg_error *err){static const uint64_t begin[FG_RANK_COUNT]={0u,0u,46666896u,93333792u,140000688u,180000846u,226667743u,273334639u},count[FG_RANK_COUNT]={0u,46666896u,46666896u,46666896u,40000158u,46666897u,46666896u,46666897u};if(!row_begin||!row_count||rank==0u||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident n-gram rank");return FG_ERR_ARGUMENT;}*row_begin=begin[rank];*row_count=count[rank];return FG_OK;}

void fg_ngram_resident_close(fg_ngram_resident *resident){if(!resident)return;if(resident->data){munlock(resident->data,(size_t)resident->bytes);free(resident->data);}free(resident);}

fg_status fg_ngram_resident_open(fg_ngram_resident **out,const char *path,uint64_t row_begin,uint64_t row_count,fg_error *err){if(!out||!path||!row_count||row_count>SIZE_MAX/FG_NGRAM_ROW_BYTES){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident n-gram shard");return FG_ERR_ARGUMENT;}*out=NULL;uint64_t bytes=row_count*FG_NGRAM_ROW_BYTES;int fd=open(path,O_RDONLY|O_CLOEXEC);if(fd<0){fg_error_set(err,FG_ERR_IO,"open resident n-gram shard: %s",strerror(errno));return FG_ERR_IO;}struct stat stat_value;if(fstat(fd,&stat_value)!=0||(uint64_t)stat_value.st_size!=bytes){close(fd);fg_error_set(err,FG_ERR_MISMATCH,"resident n-gram shard size mismatch");return FG_ERR_MISMATCH;}fg_ngram_resident *resident=calloc(1,sizeof(*resident));if(!resident){close(fd);fg_error_set(err,FG_ERR_OOM,"allocate resident n-gram metadata");return FG_ERR_OOM;}resident->row_begin=row_begin;resident->row_count=row_count;resident->bytes=bytes;if(posix_memalign((void **)&resident->data,FG_ALIGNMENT,(size_t)bytes)!=0){close(fd);fg_ngram_resident_close(resident);fg_error_set(err,FG_ERR_OOM,"allocate resident n-gram shard");return FG_ERR_OOM;}if(mlock(resident->data,(size_t)bytes)!=0){close(fd);fg_ngram_resident_close(resident);fg_error_set(err,FG_ERR_UNAVAILABLE,"lock resident n-gram shard: %s",strerror(errno));return FG_ERR_UNAVAILABLE;}uint64_t offset=0;while(offset<bytes){size_t request=(size_t)((bytes-offset)>(8u*1024u*1024u)?8u*1024u*1024u:bytes-offset);ssize_t got=pread(fd,resident->data+offset,request,(off_t)offset);if(got<0&&errno==EINTR)continue;if(got<=0){close(fd);fg_ngram_resident_close(resident);fg_error_set(err,FG_ERR_IO,"load resident n-gram shard: %s",got<0?strerror(errno):"short read");return FG_ERR_IO;}offset+=(uint64_t)got;}close(fd);*out=resident;return FG_OK;}

fg_status fg_ngram_resident_read(const fg_ngram_resident *resident,const uint64_t *rows,uint32_t row_count,uint8_t *packed,uint64_t packed_capacity,fg_error *err){if(!resident||!rows||!row_count||!packed||packed_capacity<(uint64_t)row_count*FG_NGRAM_ROW_BYTES){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident n-gram read");return FG_ERR_ARGUMENT;}for(uint32_t i=0;i<row_count;i++){if(rows[i]<resident->row_begin||rows[i]-resident->row_begin>=resident->row_count){fg_error_set(err,FG_ERR_MISMATCH,"n-gram row %llu is outside resident shard",(unsigned long long)rows[i]);return FG_ERR_MISMATCH;}memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,resident->data+(rows[i]-resident->row_begin)*FG_NGRAM_ROW_BYTES,FG_NGRAM_ROW_BYTES);}return FG_OK;}

static void q38_ngram_fill(const int32_t *tokens,size_t end,size_t segment_start,
                           uint64_t rows[FG_NGRAM_HEAD_COUNT],uint64_t addresses[FG_NGRAM_HEAD_COUNT]){
    uint64_t max_long=INT64_MAX,multiplier_max=max_long/248320u,half_bound=multiplier_max/2u,multiplier[3];
    for(uint32_t i=0;i<3;i++){uint64_t seed=UINT64_C(1234)+UINT64_C(0x9e3779b97f4a7c15)*(i+1u);multiplier[i]=2u*(splitmix64(seed)%half_bound)+1u;}
    uint64_t shifted[3]={(uint64_t)(int64_t)tokens[end-1u],FG_Q38_EOS_TOKEN,FG_Q38_EOS_TOKEN};
    for(uint32_t shift=1;shift<3;shift++)if(end>shift&&end-1u-shift>=segment_start)shifted[shift]=(uint64_t)(int64_t)tokens[end-1u-shift];
    for(uint32_t ngram=2;ngram<=3;ngram++){uint64_t mixed=shifted[0]*multiplier[0];for(uint32_t pos=1;pos<ngram;pos++)mixed^=shifted[pos]*multiplier[pos];uint32_t begin=(ngram-2u)*8u;for(uint32_t h=begin;h<begin+8u;h++){rows[h]=signed_remainder_u64(mixed,q38_ngram_vocab[h])+q38_ngram_offset[h];addresses[h]=rows[h]*FG_NGRAM_ROW_BYTES;}}
}

fg_status fg_q38_ngram_lookup(const int32_t *tokens,size_t count,uint64_t rows[FG_NGRAM_HEAD_COUNT],
                              uint64_t addresses[FG_NGRAM_HEAD_COUNT],fg_error *err){
    if(!tokens||count==0||!rows||!addresses){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Qwen3.8 n-gram lookup arguments");return FG_ERR_ARGUMENT;}
    for(size_t i=0;i<count;i++)if(tokens[i]<0||tokens[i]>=248320){fg_error_set(err,FG_ERR_FORMAT,"token %zu is outside Qwen3.8 vocabulary",i);return FG_ERR_FORMAT;}
    size_t segment_start=0;for(size_t i=0;i+1u<count;i++)if((uint32_t)tokens[i]==FG_Q38_EOS_TOKEN)segment_start=i+1u;
    q38_ngram_fill(tokens,count,segment_start,rows,addresses);
    return FG_OK;
}

fg_status fg_ngram_store_open(fg_ngram_store **out,fg_vk_context *vk,const char *path,uint64_t table_bytes,fg_error *err){
    if(!out||!vk||!path||!table_bytes){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram store arguments");return FG_ERR_ARGUMENT;}*out=NULL;fg_ngram_store *s=calloc(1,sizeof(*s));if(!s){fg_error_set(err,FG_ERR_OOM,"allocate n-gram store");return FG_ERR_OOM;}s->fd=-1;s->vk=vk;s->table_bytes=table_bytes;s->fd=open(path,O_RDONLY|O_DIRECT|O_CLOEXEC);if(s->fd<0){fg_error_set(err,FG_ERR_IO,"open n-gram table: %s",strerror(errno));fg_ngram_store_close(s);return FG_ERR_IO;}struct stat st;if(fstat(s->fd,&st)!=0||(uint64_t)st.st_size!=fg_align_up_u64(table_bytes,FG_ALIGNMENT)){fg_error_set(err,FG_ERR_MISMATCH,"n-gram table size does not match sealed tensor");fg_ngram_store_close(s);return FG_ERR_MISMATCH;}
    fg_status status=fg_uring_create(&s->ring,FG_RING_STORAGE,64u,err);if(status==FG_OK)status=fg_uring_register_file(s->ring,s->fd,&s->slot,err);if(status==FG_OK&&posix_memalign((void **)&s->io_buffer,FG_ALIGNMENT,(size_t)FG_NGRAM_PREFILL_IO_BYTES)!=0){fg_error_set(err,FG_ERR_OOM,"allocate bounded n-gram prefill direct-read buffers");status=FG_ERR_OOM;}if(status==FG_OK)status=fg_uring_register_buffer(s->ring,s->io_buffer,FG_NGRAM_PREFILL_IO_BYTES,err);if(status==FG_OK)status=fg_ngram_cache_create(&s->cache,err);if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)FG_NGRAM_PREFILL_MAX_ROWS*FG_NGRAM_ROW_BYTES,&s->packed,err);if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)FG_NGRAM_PREFILL_MAX_ROWS*FG_NGRAM_EMBED_WIDTH*4u,&s->embedding,err);if(status!=FG_OK){fg_ngram_store_close(s);return status;}*out=s;return FG_OK;
}

void fg_ngram_store_close(fg_ngram_store *s){if(!s)return;fg_vk_tensor_destroy(s->embedding_view);fg_vk_tensor_destroy(s->embedding);fg_vk_tensor_destroy(s->packed);fg_ngram_cache_destroy(s->cache);fg_uring_destroy(s->ring);if(s->fd>=0)close(s->fd);free(s->io_buffer);free(s);}

static bool cache_has(fg_ngram_cache *cache,uint64_t offset){const void *ignored=NULL;return fg_ngram_cache_get(cache,offset,&ignored);}

static fg_status load_missing_blocks(fg_ngram_store *s,const uint64_t *addresses,uint32_t address_count,fg_error *err){
    if(!s||!addresses||!address_count||address_count>FG_NGRAM_PREFILL_MAX_ROWS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid bounded n-gram block load");return FG_ERR_ARGUMENT;}
    s->last_read_count=0;s->last_read_bytes=0;s->last_io_ms=0.0;
    uint64_t probes[FG_NGRAM_PREFILL_MAX_BLOCKS];uint32_t probe_count=0;
    for(uint32_t i=0;i<address_count;i++){if(addresses[i]>UINT64_MAX-(FG_NGRAM_ROW_BYTES-1u)){fg_error_set(err,FG_ERR_FORMAT,"n-gram row address overflows");return FG_ERR_FORMAT;}probes[probe_count++]=addresses[i];probes[probe_count++]=addresses[i]+FG_NGRAM_ROW_BYTES-1u;}
    fg_ngram_read plan[FG_NGRAM_PREFILL_MAX_BLOCKS];uint32_t plan_count=0;fg_status status=fg_ngram_plan_reads(probes,probe_count,s->table_bytes,plan,FG_NGRAM_PREFILL_MAX_BLOCKS,&plan_count,err);if(status!=FG_OK)return status;
    fg_uring_read reads[FG_NGRAM_PREFILL_MAX_BLOCKS];uint32_t read_count=0;uint64_t cursor=0;
    for(uint32_t i=0;i<plan_count;i++){
        uint64_t first=plan[i].offset,second=first+FG_NGRAM_BLOCK_BYTES;bool first_cached=cache_has(s->cache,first),second_cached=plan[i].bytes==FG_NGRAM_MAX_READ_BYTES&&cache_has(s->cache,second);
        if(first_cached&&(plan[i].bytes==FG_NGRAM_BLOCK_BYTES||second_cached))continue;
        uint64_t offset=first;uint32_t bytes=plan[i].bytes;
        if(plan[i].bytes==FG_NGRAM_MAX_READ_BYTES){if(first_cached){offset=second;bytes=FG_NGRAM_BLOCK_BYTES;}else if(second_cached){offset=first;bytes=FG_NGRAM_BLOCK_BYTES;}}
        if(read_count>=FG_NGRAM_PREFILL_MAX_BLOCKS||cursor+(uint64_t)bytes>FG_NGRAM_PREFILL_IO_BYTES){fg_error_set(err,FG_ERR_LIMIT,"n-gram prefill I/O arena is exhausted");return FG_ERR_LIMIT;}
        reads[read_count++]=(fg_uring_read){.buffer=s->io_buffer+cursor,.bytes=bytes,.offset=offset};cursor+=bytes;
    }
    s->last_read_count=read_count;s->last_read_bytes=cursor;double io_start=ngram_ts();
    if(read_count)status=fg_uring_pread_batch(s->ring,s->slot,reads,read_count,err);
    s->last_io_ms=ngram_ts()-io_start;
    for(uint32_t i=0;status==FG_OK&&i<read_count;i++){status=fg_ngram_cache_put(s->cache,reads[i].offset,reads[i].buffer,err);if(status==FG_OK&&reads[i].bytes==FG_NGRAM_MAX_READ_BYTES)status=fg_ngram_cache_put(s->cache,reads[i].offset+FG_NGRAM_BLOCK_BYTES,(const uint8_t *)reads[i].buffer+FG_NGRAM_BLOCK_BYTES,err);}
    return status;
}

static fg_status pack_rows(fg_ngram_store *s,const uint64_t *addresses,uint32_t row_count,fg_error *err){
    uint8_t *packed=fg_vk_tensor_map(s->packed);if(!packed){fg_error_set(err,FG_ERR_OOM,"map n-gram packed prefill tensor");return FG_ERR_OOM;}
    for(uint32_t i=0;i<row_count;i++){uint64_t block=addresses[i]&~(uint64_t)(FG_NGRAM_BLOCK_BYTES-1u);uint32_t within=(uint32_t)(addresses[i]-block),first=FG_NGRAM_ROW_BYTES;if(first>FG_NGRAM_BLOCK_BYTES-within)first=FG_NGRAM_BLOCK_BYTES-within;const void *data=NULL;if(!fg_ngram_cache_get(s->cache,block,&data)){fg_error_set(err,FG_ERR_MISMATCH,"n-gram cache missed a planned block");return FG_ERR_MISMATCH;}memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,(const uint8_t *)data+within,first);if(first<FG_NGRAM_ROW_BYTES){if(!fg_ngram_cache_get(s->cache,block+FG_NGRAM_BLOCK_BYTES,&data)){fg_error_set(err,FG_ERR_MISMATCH,"n-gram cache missed a row continuation");return FG_ERR_MISMATCH;}memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES+first,data,FG_NGRAM_ROW_BYTES-first);}}
    return FG_OK;
}

fg_status fg_ngram_store_lookup_prefill(fg_ngram_store *s,const int32_t *tokens,size_t history_count,uint32_t first_token,uint32_t token_count,fg_vk_tensor **embedding,fg_error *err){
    if(!s||!tokens||!history_count||token_count==0u||token_count>FG_NGRAM_PREFILL_MAX_TOKENS||!embedding||first_token>history_count||(size_t)token_count>history_count-(size_t)first_token){fg_error_set(err,FG_ERR_ARGUMENT,"invalid bounded n-gram prefill lookup range");return FG_ERR_ARGUMENT;}
    double trace_start=ngram_ts();uint64_t addresses[FG_NGRAM_PREFILL_MAX_ROWS],rows[FG_NGRAM_PREFILL_MAX_ROWS];size_t end=(size_t)first_token+token_count;fg_status status=FG_OK;
    for(size_t i=0;i<end;i++)if(tokens[i]<0||tokens[i]>=248320){fg_error_set(err,FG_ERR_FORMAT,"token %zu is outside Qwen3.8 vocabulary",i);return FG_ERR_FORMAT;}
    size_t segment_start=0;for(size_t i=0;i<(size_t)first_token;i++)if((uint32_t)tokens[i]==FG_Q38_EOS_TOKEN)segment_start=i+1u;for(uint32_t token=0;token<token_count;token++){size_t pos=(size_t)first_token+token;q38_ngram_fill(tokens,pos+1u,segment_start,rows+(uint64_t)token*FG_NGRAM_HEAD_COUNT,addresses+(uint64_t)token*FG_NGRAM_HEAD_COUNT);if((uint32_t)tokens[pos]==FG_Q38_EOS_TOKEN)segment_start=pos+1u;}
    if(ngram_locality_trace_enabled())for(uint32_t token=0;token<token_count;token++){char line[768];int used=snprintf(line,sizeof(line),"NGRAM_LOCALITY position=%u batch_first=%u batch_tokens=%u",first_token+token,first_token,token_count);for(uint32_t head=0;used>0&&(size_t)used<sizeof(line)&&head<FG_NGRAM_HEAD_COUNT;head++){int written=snprintf(line+(size_t)used,sizeof(line)-(size_t)used," h%u=%llu",head,(unsigned long long)addresses[(uint64_t)token*FG_NGRAM_HEAD_COUNT+head]);if(written<0||(size_t)written>=sizeof(line)-(size_t)used){used=-1;break;}used+=written;}if(used>0&&(size_t)used+1u<sizeof(line)){line[used++]='\n';line[used]=0;fputs(line,stderr);}}
    uint32_t row_count=token_count*FG_NGRAM_HEAD_COUNT;double hash_end=ngram_ts();if(status==FG_OK)status=load_missing_blocks(s,addresses,row_count,err);double load_end=ngram_ts();if(status==FG_OK)status=pack_rows(s,addresses,row_count,err);double pack_end=ngram_ts();if(status==FG_OK)status=fg_vk_dequantize_iq4_nl(s->vk,s->embedding,s->packed,row_count,FG_NGRAM_EMBED_WIDTH,err);double dequant_end=ngram_ts();if(status==FG_OK){fg_vk_tensor_destroy(s->embedding_view);s->embedding_view=NULL;uint64_t bytes=(uint64_t)token_count*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH*4u;if(bytes==fg_vk_tensor_bytes(s->embedding))*embedding=s->embedding;else{status=fg_vk_tensor_view(s->embedding,0,bytes,&s->embedding_view,err);if(status==FG_OK)*embedding=s->embedding_view;}}if(ngram_trace_enabled())fprintf(stderr,"NGRAM_TRACE first=%u tokens=%u rows=%u reads=%u bytes=%llu hash_ms=%.3f load_ms=%.3f io_ms=%.3f pack_ms=%.3f dequant_ms=%.3f total_ms=%.3f\n",first_token,token_count,row_count,s->last_read_count,(unsigned long long)s->last_read_bytes,hash_end-trace_start,load_end-hash_end,s->last_io_ms,pack_end-load_end,dequant_end-pack_end,ngram_ts()-trace_start);return status;
}

fg_status fg_ngram_store_lookup(fg_ngram_store *s,const int32_t *tokens,size_t count,fg_vk_tensor **embedding,fg_error *err){
    if(!count||count-1u>UINT32_MAX){fg_error_set(err,FG_ERR_LIMIT,"n-gram lookup history exceeds supported context index");return FG_ERR_LIMIT;}return fg_ngram_store_lookup_prefill(s,tokens,count,(uint32_t)(count-1u),1u,embedding,err);
}

fg_status fg_ngram_store_decode_packed(fg_ngram_store *s,const uint8_t *packed,uint32_t row_count,fg_vk_tensor **embedding,fg_error *err){if(!s||!packed||!row_count||row_count>FG_NGRAM_PREFILL_MAX_ROWS||!embedding){fg_error_set(err,FG_ERR_ARGUMENT,"invalid packed n-gram decode");return FG_ERR_ARGUMENT;}uint64_t packed_bytes=(uint64_t)row_count*FG_NGRAM_ROW_BYTES;fg_status status=fg_vk_tensor_write(s->packed,0,packed,packed_bytes,err);if(status==FG_OK)status=fg_vk_dequantize_iq4_nl(s->vk,s->embedding,s->packed,row_count,FG_NGRAM_EMBED_WIDTH,err);if(status==FG_OK){fg_vk_tensor_destroy(s->embedding_view);s->embedding_view=NULL;uint64_t bytes=(uint64_t)row_count*FG_NGRAM_EMBED_WIDTH*4u;if(bytes==fg_vk_tensor_bytes(s->embedding))*embedding=s->embedding;else{status=fg_vk_tensor_view(s->embedding,0,bytes,&s->embedding_view,err);if(status==FG_OK)*embedding=s->embedding_view;}}return status;}
