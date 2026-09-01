#include "fg_ngram.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"
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
struct fg_ngram_store{int fd;fg_uring *ring;uint32_t slot,max_tokens,max_rows,max_blocks;uint64_t table_bytes,io_bytes;uint8_t *io_buffer;fg_ngram_cache *cache;fg_vk_context *vk;fg_vk_tensor *packed,*embedding,*embedding_view;uint32_t last_read_count;uint64_t last_read_bytes;double last_io_ms;};
struct fg_ngram_resident{uint8_t *data;uint64_t row_begin,row_count,bytes;};
#define PIPELINE_CACHE_WAYS 4u
#define PIPELINE_CACHE_MAX_PAGES (FG_NGRAM_HEAD_COUNT*2u)
#define PIPELINE_CACHE_STAGING_BYTES \
    ((uint64_t)PIPELINE_CACHE_MAX_PAGES*FG_NGRAM_BLOCK_BYTES)
typedef struct pipeline_cache_entry{
    uint64_t offset;
    uint8_t valid;
    uint8_t referenced;
    uint8_t hand;
    uint8_t reserved[5];
}pipeline_cache_entry;
_Static_assert(sizeof(pipeline_cache_entry)==16u,
               "pipeline cache entry layout must remain fixed");
struct fg_ngram_pipeline_cache{
    int fd;
    bool direct_io;
    fg_uring *ring;
    uint32_t slot;
    uint8_t *arena;
    uint64_t arena_bytes;
    uint8_t *staging;
    pipeline_cache_entry *entries;
    uint8_t *data;
    uint32_t page_capacity;
    uint32_t set_count;
    uint64_t row_begin,row_count,file_bytes;
    fg_ngram_pipeline_cache_stats stats;
};
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
uint64_t fg_ngram_store_vk_bytes(const fg_ngram_store *s){
    if(!s)return 0;
    return (s->packed?fg_vk_tensor_bytes(s->packed):0u)+
           (s->embedding?fg_vk_tensor_bytes(s->embedding):0u);
}
fg_status fg_ngram_cache_create(fg_ngram_cache **out,fg_error *err){if(!out){fg_error_set(err,FG_ERR_ARGUMENT,"n-gram cache output is null");return FG_ERR_ARGUMENT;}*out=NULL;fg_ngram_cache *c=calloc(1,sizeof(*c));if(!c){fg_error_set(err,FG_ERR_OOM,"allocate n-gram cache");return FG_ERR_OOM;}if(posix_memalign((void **)&c->data,FG_ALIGNMENT,FG_NGRAM_CACHE_BYTES)!=0){free(c);fg_error_set(err,FG_ERR_OOM,"allocate 8 MiB aligned n-gram data cache");return FG_ERR_OOM;}c->entry=calloc(CACHE_BLOCKS,sizeof(*c->entry));if(!c->entry){free(c->data);free(c);fg_error_set(err,FG_ERR_OOM,"allocate n-gram cache metadata");return FG_ERR_OOM;}*out=c;return FG_OK;}
void fg_ngram_cache_destroy(fg_ngram_cache *c){if(!c)return;free(c->entry);free(c->data);free(c);}
uint64_t fg_ngram_cache_memory_bytes(void){
    return FG_NGRAM_CACHE_BYTES+sizeof(fg_ngram_cache)+
           (uint64_t)CACHE_BLOCKS*sizeof(cache_entry);
}
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

static fg_status resident_open_impl(fg_ngram_resident **out,const char *path,
                                    uint64_t row_begin,uint64_t row_count,
                                    const uint8_t expected_sha256[32],
                                    fg_error *err){
    if(!out||!path||!row_count||row_count>SIZE_MAX/FG_NGRAM_ROW_BYTES){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident n-gram shard");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    uint64_t bytes=row_count*FG_NGRAM_ROW_BYTES;
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    if(fd<0){
        fg_error_set(err,FG_ERR_IO,"open resident n-gram shard %s: %s",
                     path,strerror(errno));
        return FG_ERR_IO;
    }
    struct stat stat_value;
    if(fstat(fd,&stat_value)!=0||!S_ISREG(stat_value.st_mode)||
       stat_value.st_size<0||(uint64_t)stat_value.st_size!=bytes){
        close(fd);
        fg_error_set(err,FG_ERR_MISMATCH,
                     "resident n-gram shard %s size is not exactly %llu bytes",
                     path,(unsigned long long)bytes);
        return FG_ERR_MISMATCH;
    }
    fg_ngram_resident *resident=calloc(1,sizeof(*resident));
    if(!resident){
        close(fd);fg_error_set(err,FG_ERR_OOM,"allocate resident n-gram metadata");
        return FG_ERR_OOM;
    }
    resident->row_begin=row_begin;resident->row_count=row_count;resident->bytes=bytes;
    if(posix_memalign((void **)&resident->data,FG_ALIGNMENT,(size_t)bytes)!=0){
        close(fd);fg_ngram_resident_close(resident);
        fg_error_set(err,FG_ERR_OOM,"allocate resident n-gram shard");
        return FG_ERR_OOM;
    }
    if(mlock(resident->data,(size_t)bytes)!=0){
        close(fd);fg_ngram_resident_close(resident);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"lock resident n-gram shard: %s",
                     strerror(errno));
        return FG_ERR_UNAVAILABLE;
    }
    fg_sha256 hash;
    if(expected_sha256)fg_sha256_init(&hash);
    uint64_t offset=0u;
    while(offset<bytes){
        size_t request=(size_t)((bytes-offset)>(8u*1024u*1024u)?
            8u*1024u*1024u:bytes-offset);
        ssize_t got=pread(fd,resident->data+offset,request,(off_t)offset);
        if(got<0&&errno==EINTR)continue;
        if(got<=0){
            close(fd);fg_ngram_resident_close(resident);
            fg_error_set(err,FG_ERR_IO,"load resident n-gram shard: %s",
                         got<0?strerror(errno):"short read");
            return FG_ERR_IO;
        }
        if(expected_sha256)fg_sha256_update(&hash,resident->data+offset,(size_t)got);
        offset+=(uint64_t)got;
    }
    close(fd);
    if(expected_sha256){
        uint8_t digest[32];fg_sha256_final(&hash,digest);
        if(memcmp(digest,expected_sha256,sizeof(digest))){
            fg_ngram_resident_close(resident);
            fg_error_set(err,FG_ERR_MISMATCH,
                         "resident n-gram shard %s SHA-256 mismatch",path);
            return FG_ERR_MISMATCH;
        }
    }
    *out=resident;
    return FG_OK;
}

fg_status fg_ngram_resident_open(fg_ngram_resident **out,const char *path,
                                 uint64_t row_begin,uint64_t row_count,
                                 fg_error *err){
    return resident_open_impl(out,path,row_begin,row_count,NULL,err);
}

fg_status fg_ngram_resident_open_sealed(fg_ngram_resident **out,const char *path,
                                        uint64_t row_begin,uint64_t row_count,
                                        const uint8_t sha256[32],fg_error *err){
    if(!sha256){
        fg_error_set(err,FG_ERR_ARGUMENT,"resident n-gram SHA-256 is null");
        return FG_ERR_ARGUMENT;
    }
    return resident_open_impl(out,path,row_begin,row_count,sha256,err);
}

fg_status fg_ngram_resident_open_manifest(fg_ngram_resident **out,
                                          const fg_manifest *manifest,
                                          const char *pack_dir,uint32_t rank,
                                          fg_error *err){
    if(!out||!manifest||!pack_dir||!*pack_dir||rank==0u||rank>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid manifest resident n-gram open arguments");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    fg_status status=fg_q38_validate_ngram_shards(manifest,err);
    if(status!=FG_OK)return status;
    const fg_ngram_shard_record *record=fg_q38_find_ngram_shard(manifest,rank);
    if(!record){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "manifest has no unique resident n-gram shard for rank %u",rank);
        return FG_ERR_MISMATCH;
    }
    char path[1200];
    if(snprintf(path,sizeof(path),"%s/" FG_NGRAM_SHARD_ARTIFACT_FORMAT,
                pack_dir,rank)>=(int)sizeof(path)){
        fg_error_set(err,FG_ERR_LIMIT,"resident n-gram shard path is too long");
        return FG_ERR_LIMIT;
    }
    return fg_ngram_resident_open_sealed(out,path,record->row_begin,
                                         record->row_count,record->sha256,err);
}

fg_status fg_ngram_resident_read(const fg_ngram_resident *resident,const uint64_t *rows,uint32_t row_count,uint8_t *packed,uint64_t packed_capacity,fg_error *err){if(!resident||!rows||!row_count||!packed||packed_capacity<(uint64_t)row_count*FG_NGRAM_ROW_BYTES){fg_error_set(err,FG_ERR_ARGUMENT,"invalid resident n-gram read");return FG_ERR_ARGUMENT;}for(uint32_t i=0;i<row_count;i++){if(rows[i]<resident->row_begin||rows[i]-resident->row_begin>=resident->row_count){fg_error_set(err,FG_ERR_MISMATCH,"n-gram row %llu is outside resident shard",(unsigned long long)rows[i]);return FG_ERR_MISMATCH;}memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,resident->data+(rows[i]-resident->row_begin)*FG_NGRAM_ROW_BYTES,FG_NGRAM_ROW_BYTES);}return FG_OK;}

static bool pipeline_cache_layout(uint64_t arena_bytes,uint32_t *page_capacity,
                                  uint64_t *data_offset){
    if(!page_capacity||!data_offset||
       arena_bytes<PIPELINE_CACHE_STAGING_BYTES+
                   PIPELINE_CACHE_WAYS*(sizeof(pipeline_cache_entry)+
                                        FG_NGRAM_BLOCK_BYTES)||
       arena_bytes>SIZE_MAX||arena_bytes%FG_NGRAM_BLOCK_BYTES)
        return false;
    uint64_t pages=(arena_bytes-PIPELINE_CACHE_STAGING_BYTES)/
        (FG_NGRAM_BLOCK_BYTES+sizeof(pipeline_cache_entry));
    pages-=pages%PIPELINE_CACHE_WAYS;
    while(pages>=PIPELINE_CACHE_WAYS){
        uint64_t entries=pages*sizeof(pipeline_cache_entry);
        uint64_t offset=fg_align_up_u64(
            PIPELINE_CACHE_STAGING_BYTES+entries,FG_NGRAM_BLOCK_BYTES);
        if(offset<=arena_bytes&&pages<=UINT32_MAX&&
           pages*FG_NGRAM_BLOCK_BYTES<=arena_bytes-offset){
            *page_capacity=(uint32_t)pages;
            *data_offset=offset;
            return true;
        }
        pages-=PIPELINE_CACHE_WAYS;
    }
    return false;
}

static fg_status pipeline_cache_verify(fg_ngram_pipeline_cache *cache,
                                       const uint8_t expected_sha256[32],
                                       fg_error *err){
    fg_sha256 hash;fg_sha256_init(&hash);
    uint64_t chunk=cache->arena_bytes;
    if(chunk>UINT64_C(8)*1024u*1024u)chunk=UINT64_C(8)*1024u*1024u;
    chunk-=chunk%FG_NGRAM_BLOCK_BYTES;
    if(!chunk){
        fg_error_set(err,FG_ERR_LIMIT,"pipeline n-gram hash buffer is too small");
        return FG_ERR_LIMIT;
    }
    uint64_t offset=0u;
    while(offset<cache->file_bytes){
        uint64_t remaining=cache->file_bytes-offset;
        uint64_t request=remaining>chunk?chunk:remaining;
        if(cache->direct_io)request=fg_align_up_u64(request,FG_NGRAM_BLOCK_BYTES);
        ssize_t got=pread(cache->fd,cache->arena,(size_t)request,(off_t)offset);
        if(got<0&&errno==EINTR)continue;
        uint64_t expected=remaining<request?remaining:request;
        if(got<0||(uint64_t)got!=expected){
            fg_error_set(err,FG_ERR_IO,
                         "verify pipeline n-gram shard: %s",
                         got<0?strerror(errno):"short read");
            return FG_ERR_IO;
        }
        fg_sha256_update(&hash,cache->arena,(size_t)got);
        offset+=(uint64_t)got;
    }
    uint8_t digest[32];fg_sha256_final(&hash,digest);
    if(memcmp(digest,expected_sha256,sizeof(digest))){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline n-gram shard SHA-256 mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

static fg_status pipeline_cache_open_impl(
    fg_ngram_pipeline_cache **out,const char *path,uint64_t row_begin,
    uint64_t row_count,const uint8_t sha256[32],uint64_t cache_bytes,
    bool direct_io,fg_error *err){
    if(!out||!path||!*path||!row_count||!sha256||
       row_count>UINT64_MAX/FG_NGRAM_ROW_BYTES){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid pipeline n-gram cache open arguments");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    uint32_t page_capacity=0u;uint64_t data_offset=0u;
    if(!pipeline_cache_layout(cache_bytes,&page_capacity,&data_offset)){
        fg_error_set(err,FG_ERR_LIMIT,"invalid pipeline n-gram cache allocation");
        return FG_ERR_LIMIT;
    }
    fg_ngram_pipeline_cache *cache=calloc(1,sizeof(*cache));
    if(!cache){
        fg_error_set(err,FG_ERR_OOM,"allocate pipeline n-gram cache metadata");
        return FG_ERR_OOM;
    }
    cache->fd=-1;cache->direct_io=direct_io;
    cache->row_begin=row_begin;cache->row_count=row_count;
    cache->file_bytes=row_count*FG_NGRAM_ROW_BYTES;
    cache->arena_bytes=cache_bytes;cache->page_capacity=page_capacity;
    cache->set_count=page_capacity/PIPELINE_CACHE_WAYS;
    int flags=O_RDONLY|O_CLOEXEC|(direct_io?O_DIRECT:0);
    cache->fd=open(path,flags);
    if(cache->fd<0){
        fg_error_set(err,FG_ERR_IO,"open pipeline n-gram shard %s: %s",
                     path,strerror(errno));
        fg_ngram_pipeline_cache_close(cache);
        return FG_ERR_IO;
    }
    struct stat info;
    if(fstat(cache->fd,&info)!=0||!S_ISREG(info.st_mode)||info.st_size<0||
       (uint64_t)info.st_size!=cache->file_bytes){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline n-gram shard %s size is not exactly %llu bytes",
                     path,(unsigned long long)cache->file_bytes);
        fg_ngram_pipeline_cache_close(cache);
        return FG_ERR_MISMATCH;
    }
    if(posix_memalign((void **)&cache->arena,FG_NGRAM_BLOCK_BYTES,
                      (size_t)cache_bytes)!=0){
        fg_error_set(err,FG_ERR_OOM,
                     "allocate fixed pipeline n-gram cache");
        fg_ngram_pipeline_cache_close(cache);
        return FG_ERR_OOM;
    }
    memset(cache->arena,0,(size_t)cache_bytes);
    fg_status status=pipeline_cache_verify(cache,sha256,err);
    cache->staging=cache->arena;
    cache->entries=(pipeline_cache_entry *)(cache->arena+
        PIPELINE_CACHE_STAGING_BYTES);
    cache->data=cache->arena+data_offset;
    memset(cache->arena,0,(size_t)data_offset);
    if(status==FG_OK&&direct_io)
        status=fg_uring_create(&cache->ring,FG_RING_STORAGE,64u,err);
    if(status==FG_OK&&direct_io)
        status=fg_uring_register_file(cache->ring,cache->fd,&cache->slot,err);
    if(status==FG_OK&&direct_io)
        status=fg_uring_register_buffer(cache->ring,cache->staging,
                                        PIPELINE_CACHE_STAGING_BYTES,err);
    if(status!=FG_OK){
        fg_ngram_pipeline_cache_close(cache);
        return status;
    }
    *out=cache;
    return FG_OK;
}

fg_status fg_ngram_pipeline_cache_open_manifest(
    fg_ngram_pipeline_cache **out,const fg_manifest *manifest,const char *pack_dir,
    uint32_t rank,fg_error *err){
    if(!out||!manifest||!pack_dir||!*pack_dir||
       rank==0u||rank>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid manifest pipeline n-gram cache open arguments");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;
    fg_status status=fg_q38_validate_ngram_shards(manifest,err);
    if(status!=FG_OK)return status;
    const fg_ngram_shard_record *record=fg_q38_find_ngram_shard(manifest,rank);
    if(!record){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "manifest has no unique pipeline n-gram shard for rank %u",
                     rank);
        return FG_ERR_MISMATCH;
    }
    char path[1200];
    if(snprintf(path,sizeof(path),"%s/" FG_NGRAM_SHARD_ARTIFACT_FORMAT,
                pack_dir,rank)>=(int)sizeof(path)){
        fg_error_set(err,FG_ERR_LIMIT,"pipeline n-gram shard path is too long");
        return FG_ERR_LIMIT;
    }
    return pipeline_cache_open_impl(out,path,record->row_begin,record->row_count,
                                    record->sha256,
                                    FG_PIPELINE_NGRAM_CACHE_BYTES,true,err);
}

fg_status fg_ngram_pipeline_cache_open_test(
    fg_ngram_pipeline_cache **out,const char *path,uint64_t row_begin,
    uint64_t row_count,const uint8_t sha256[32],uint64_t cache_bytes,
    fg_error *err){
    return pipeline_cache_open_impl(out,path,row_begin,row_count,sha256,
                                    cache_bytes,false,err);
}

void fg_ngram_pipeline_cache_close(fg_ngram_pipeline_cache *cache){
    if(!cache)return;
    fg_uring_destroy(cache->ring);
    if(cache->fd>=0)close(cache->fd);
    free(cache->arena);
    free(cache);
}

static uint32_t pipeline_cache_set(const fg_ngram_pipeline_cache *cache,
                                   uint64_t offset){
    return (uint32_t)(splitmix64(offset/FG_NGRAM_BLOCK_BYTES)%
                      cache->set_count);
}

static const uint8_t *pipeline_cache_get(fg_ngram_pipeline_cache *cache,
                                         uint64_t offset){
    uint32_t base=pipeline_cache_set(cache,offset)*PIPELINE_CACHE_WAYS;
    for(uint32_t way=0u;way<PIPELINE_CACHE_WAYS;way++){
        pipeline_cache_entry *entry=&cache->entries[base+way];
        if(entry->valid&&entry->offset==offset){
            entry->referenced=1u;
            return cache->data+(uint64_t)(base+way)*FG_NGRAM_BLOCK_BYTES;
        }
    }
    return NULL;
}

static void pipeline_cache_put(fg_ngram_pipeline_cache *cache,uint64_t offset,
                               const uint8_t data[FG_NGRAM_BLOCK_BYTES]){
    uint32_t base=pipeline_cache_set(cache,offset)*PIPELINE_CACHE_WAYS;
    uint32_t hand=cache->entries[base].hand%PIPELINE_CACHE_WAYS;
    uint32_t victim=base+hand;
    for(;;){
        pipeline_cache_entry *entry=&cache->entries[base+hand];
        if(!entry->valid||!entry->referenced){
            victim=base+hand;
            break;
        }
        entry->referenced=0u;
        hand=(hand+1u)%PIPELINE_CACHE_WAYS;
    }
    if(cache->entries[victim].valid&&cache->entries[victim].offset!=offset)
        cache->stats.evictions++;
    memcpy(cache->data+(uint64_t)victim*FG_NGRAM_BLOCK_BYTES,
           data,FG_NGRAM_BLOCK_BYTES);
    uint8_t next=(uint8_t)((hand+1u)%PIPELINE_CACHE_WAYS);
    cache->entries[victim]=(pipeline_cache_entry){
        .offset=offset,.valid=1u,.referenced=1u
    };
    cache->entries[base].hand=next;
}

typedef struct pipeline_request_page{
    uint64_t offset;
    const uint8_t *data;
    bool hit;
}pipeline_request_page;

static uint32_t pipeline_request_page_index(
    const pipeline_request_page *pages,uint32_t page_count,uint64_t offset){
    for(uint32_t i=0u;i<page_count;i++)if(pages[i].offset==offset)return i;
    return UINT32_MAX;
}

static fg_status pipeline_cache_read_batch(
    fg_ngram_pipeline_cache *cache,const fg_uring_read *reads,uint32_t count,
    int32_t results[PIPELINE_CACHE_MAX_PAGES],fg_error *err){
    if(cache->direct_io)
        return fg_uring_pread_batch_results(
            cache->ring,cache->slot,reads,count,results,
            PIPELINE_CACHE_MAX_PAGES,err);
    for(uint32_t i=0u;i<count;i++){
        ssize_t got;
        do got=pread(cache->fd,reads[i].buffer,reads[i].bytes,
                     (off_t)reads[i].offset);
        while(got<0&&errno==EINTR);
        results[i]=got<0?-errno:
            (got>INT32_MAX?INT32_MIN:(int32_t)got);
    }
    return FG_OK;
}

fg_status fg_ngram_pipeline_cache_read(
    fg_ngram_pipeline_cache *cache,const uint64_t *rows,uint32_t row_count,
    uint8_t *packed,uint64_t packed_capacity,fg_error *err){
    if(!cache||!rows||!row_count||row_count>FG_NGRAM_HEAD_COUNT||!packed||
       packed_capacity<(uint64_t)row_count*FG_NGRAM_ROW_BYTES){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline n-gram cache read");
        return FG_ERR_ARGUMENT;
    }
    pipeline_request_page pages[PIPELINE_CACHE_MAX_PAGES];
    uint32_t page_count=0u;
    for(uint32_t i=0u;i<row_count;i++){
        if(rows[i]<cache->row_begin||
           rows[i]-cache->row_begin>=cache->row_count){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "n-gram row %llu is outside pipeline shard",
                         (unsigned long long)rows[i]);
            return FG_ERR_MISMATCH;
        }
        uint64_t address=(rows[i]-cache->row_begin)*FG_NGRAM_ROW_BYTES;
        uint64_t first=address&~(uint64_t)(FG_NGRAM_BLOCK_BYTES-1u);
        uint64_t last=(address+FG_NGRAM_ROW_BYTES-1u)&
            ~(uint64_t)(FG_NGRAM_BLOCK_BYTES-1u);
        pages[page_count++]=(pipeline_request_page){.offset=first};
        if(last!=first)pages[page_count++]=(pipeline_request_page){.offset=last};
    }
    for(uint32_t i=1u;i<page_count;i++){
        pipeline_request_page value=pages[i];uint32_t j=i;
        while(j&&pages[j-1u].offset>value.offset){pages[j]=pages[j-1u];j--;}
        pages[j]=value;
    }
    uint32_t unique=0u;
    for(uint32_t i=0u;i<page_count;i++)
        if(!unique||pages[i].offset!=pages[unique-1u].offset)
            pages[unique++]=pages[i];
    page_count=unique;
    uint32_t hits=0u,misses=0u;
    for(uint32_t i=0u;i<page_count;i++){
        pages[i].data=pipeline_cache_get(cache,pages[i].offset);
        pages[i].hit=pages[i].data!=NULL;
        if(pages[i].hit)hits++;else misses++;
    }
    fg_uring_read reads[PIPELINE_CACHE_MAX_PAGES];
    int32_t results[PIPELINE_CACHE_MAX_PAGES];
    uint32_t read_count=0u,staging_pages=0u;
    for(uint32_t i=0u;i<page_count;){
        if(pages[i].hit){i++;continue;}
        uint32_t first=i,page_bytes=1u;
        if(i+1u<page_count&&!pages[i+1u].hit&&
           pages[i+1u].offset==pages[i].offset+FG_NGRAM_BLOCK_BYTES)
            page_bytes=2u;
        uint8_t *buffer=cache->staging+
            (uint64_t)staging_pages*FG_NGRAM_BLOCK_BYTES;
        memset(buffer,0,(size_t)page_bytes*FG_NGRAM_BLOCK_BYTES);
        reads[read_count++]=(fg_uring_read){
            .buffer=buffer,.bytes=page_bytes*FG_NGRAM_BLOCK_BYTES,
            .offset=pages[first].offset
        };
        pages[first].data=buffer;
        if(page_bytes==2u)pages[first+1u].data=buffer+FG_NGRAM_BLOCK_BYTES;
        staging_pages+=page_bytes;i+=page_bytes;
    }
    fg_status status=FG_OK;
    if(read_count)status=pipeline_cache_read_batch(
        cache,reads,read_count,results,err);
    for(uint32_t i=0u;status==FG_OK&&i<read_count;i++){
        uint64_t remaining=cache->file_bytes-reads[i].offset;
        uint32_t expected=remaining<reads[i].bytes?(uint32_t)remaining:
            reads[i].bytes;
        if(results[i]<0){
            fg_error_set(err,FG_ERR_IO,
                         "pipeline n-gram direct read %u: %s",
                         i,strerror(-results[i]));
            status=FG_ERR_IO;
        }else if((uint32_t)results[i]!=expected){
            fg_error_set(err,FG_ERR_IO,
                         "pipeline n-gram short read %u: got %d of %u bytes",
                         i,results[i],expected);
            status=FG_ERR_IO;
        }
    }
    for(uint32_t i=0u;status==FG_OK&&i<row_count;i++){
        uint64_t address=(rows[i]-cache->row_begin)*FG_NGRAM_ROW_BYTES;
        uint64_t first_page=address&~(uint64_t)(FG_NGRAM_BLOCK_BYTES-1u);
        uint32_t within=(uint32_t)(address-first_page);
        uint32_t first_bytes=FG_NGRAM_ROW_BYTES;
        if(first_bytes>FG_NGRAM_BLOCK_BYTES-within)
            first_bytes=FG_NGRAM_BLOCK_BYTES-within;
        uint32_t page=pipeline_request_page_index(
            pages,page_count,first_page);
        if(page==UINT32_MAX||!pages[page].data){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "pipeline n-gram request page was not loaded");
            status=FG_ERR_MISMATCH;
            break;
        }
        memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,
               pages[page].data+within,first_bytes);
        if(first_bytes<FG_NGRAM_ROW_BYTES){
            page=pipeline_request_page_index(
                pages,page_count,first_page+FG_NGRAM_BLOCK_BYTES);
            if(page==UINT32_MAX||!pages[page].data){
                fg_error_set(err,FG_ERR_MISMATCH,
                             "pipeline n-gram cross-page row was not loaded");
                status=FG_ERR_MISMATCH;
                break;
            }
            memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES+first_bytes,
                   pages[page].data,FG_NGRAM_ROW_BYTES-first_bytes);
        }
    }
    if(status==FG_OK)
        for(uint32_t i=0u;i<page_count;i++)
            if(!pages[i].hit)
                pipeline_cache_put(cache,pages[i].offset,pages[i].data);
    if(status==FG_OK){
        cache->stats.requests++;
        cache->stats.page_hits+=hits;
        cache->stats.page_misses+=misses;
        cache->stats.pages_read+=staging_pages;
        cache->stats.read_operations+=read_count;
        cache->stats.last_page_hits=hits;
        cache->stats.last_page_misses=misses;
        cache->stats.last_pages_read=staging_pages;
        cache->stats.last_read_operations=read_count;
    }
    return status;
}

void fg_ngram_pipeline_cache_get_stats(
    const fg_ngram_pipeline_cache *cache,fg_ngram_pipeline_cache_stats *stats){
    if(stats)*stats=cache?cache->stats:(fg_ngram_pipeline_cache_stats){0};
}

uint64_t fg_ngram_pipeline_cache_host_bytes(const fg_ngram_pipeline_cache *cache){
    return cache?cache->arena_bytes:0u;
}

uint32_t fg_ngram_pipeline_cache_page_capacity(const fg_ngram_pipeline_cache *cache){
    return cache?cache->page_capacity:0u;
}

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

fg_status fg_ngram_store_open(fg_ngram_store **out,fg_vk_context *vk,const char *path,
                               uint64_t table_bytes,uint32_t max_tokens,fg_error *err){
    if(!out||!vk||!path||!table_bytes||!max_tokens||
       max_tokens>FG_NGRAM_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram store arguments");return FG_ERR_ARGUMENT;}*out=NULL;fg_ngram_store *s=calloc(1,sizeof(*s));if(!s){fg_error_set(err,FG_ERR_OOM,"allocate n-gram store");return FG_ERR_OOM;}s->fd=-1;s->vk=vk;s->table_bytes=table_bytes;s->max_tokens=max_tokens;s->max_rows=max_tokens*FG_NGRAM_HEAD_COUNT;s->max_blocks=s->max_rows*2u;s->io_bytes=FG_NGRAM_PREFILL_IO_BYTES;s->fd=open(path,O_RDONLY|O_DIRECT|O_CLOEXEC);if(s->fd<0){fg_error_set(err,FG_ERR_IO,"open n-gram table: %s",strerror(errno));fg_ngram_store_close(s);return FG_ERR_IO;}struct stat st;if(fstat(s->fd,&st)!=0||(uint64_t)st.st_size!=fg_align_up_u64(table_bytes,FG_ALIGNMENT)){fg_error_set(err,FG_ERR_MISMATCH,"n-gram table size does not match sealed tensor");fg_ngram_store_close(s);return FG_ERR_MISMATCH;}
    fg_status status=fg_uring_create(&s->ring,FG_RING_STORAGE,64u,err);if(status==FG_OK)status=fg_uring_register_file(s->ring,s->fd,&s->slot,err);if(status==FG_OK&&posix_memalign((void **)&s->io_buffer,FG_ALIGNMENT,(size_t)s->io_bytes)!=0){fg_error_set(err,FG_ERR_OOM,"allocate bounded n-gram direct-read buffers");status=FG_ERR_OOM;}if(status==FG_OK)status=fg_uring_register_buffer(s->ring,s->io_buffer,s->io_bytes,err);if(status!=FG_OK){fg_ngram_store_close(s);return status;}*out=s;return FG_OK;
}

void fg_ngram_store_close(fg_ngram_store *s){if(!s)return;fg_vk_tensor_destroy(s->embedding_view);fg_vk_tensor_destroy(s->embedding);fg_vk_tensor_destroy(s->packed);fg_ngram_cache_destroy(s->cache);fg_uring_destroy(s->ring);if(s->fd>=0)close(s->fd);free(s->io_buffer);free(s);}

uint64_t fg_ngram_store_host_bytes(const fg_ngram_store *s){
    if(!s)return 0;
    return s->io_bytes+(s->cache?fg_ngram_cache_memory_bytes():0u)+
           fg_uring_host_bytes(s->ring);
}
uint64_t fg_ngram_store_io_host_bytes(const fg_ngram_store *s){
    return s?s->io_bytes:0u;
}
uint64_t fg_ngram_store_cache_host_bytes(const fg_ngram_store *s){
    return s&&s->cache?fg_ngram_cache_memory_bytes():0u;
}

static bool cache_has(fg_ngram_cache *cache,uint64_t offset){const void *ignored=NULL;return fg_ngram_cache_get(cache,offset,&ignored);}

static fg_status ensure_tensors(fg_ngram_store *s,fg_error *err){
    if(!s)return FG_ERR_ARGUMENT;
    fg_status status=FG_OK;
    if(!s->packed)status=fg_vk_tensor_create(s->vk,
        (uint64_t)s->max_rows*FG_NGRAM_ROW_BYTES,&s->packed,err);
    if(status==FG_OK&&!s->embedding)status=fg_vk_tensor_create(s->vk,
        (uint64_t)s->max_rows*FG_NGRAM_EMBED_WIDTH*4u,&s->embedding,err);
    return status;
}

static fg_status ensure_cache(fg_ngram_store *s,fg_error *err){
    if(!s||s->cache)return FG_OK;
    return fg_ngram_cache_create(&s->cache,err);
}

static fg_status load_missing_blocks(fg_ngram_store *s,const uint64_t *addresses,uint32_t address_count,fg_error *err){
    if(!s||!addresses||!address_count||address_count>s->max_rows){fg_error_set(err,FG_ERR_ARGUMENT,"invalid bounded n-gram block load");return FG_ERR_ARGUMENT;}
    fg_status status=ensure_cache(s,err);
    if(status!=FG_OK)return status;
    s->last_read_count=0;s->last_read_bytes=0;s->last_io_ms=0.0;
    uint64_t probes[FG_NGRAM_PREFILL_MAX_BLOCKS];uint32_t probe_count=0;
    for(uint32_t i=0;i<address_count;i++){if(addresses[i]>UINT64_MAX-(FG_NGRAM_ROW_BYTES-1u)){fg_error_set(err,FG_ERR_FORMAT,"n-gram row address overflows");return FG_ERR_FORMAT;}probes[probe_count++]=addresses[i];probes[probe_count++]=addresses[i]+FG_NGRAM_ROW_BYTES-1u;}
    fg_ngram_read plan[FG_NGRAM_PREFILL_MAX_BLOCKS];uint32_t plan_count=0;status=fg_ngram_plan_reads(probes,probe_count,s->table_bytes,plan,FG_NGRAM_PREFILL_MAX_BLOCKS,&plan_count,err);if(status!=FG_OK)return status;
    s->last_read_count=0;s->last_read_bytes=0;s->last_io_ms=0.0;
    for(uint32_t base=0;status==FG_OK&&base<plan_count;){
        fg_uring_read batch[FG_NGRAM_IO_SLOTS];
        uint32_t batch_count=0;
        while(base<plan_count&&batch_count<FG_NGRAM_IO_SLOTS){
            uint64_t first=plan[base].offset,second=first+FG_NGRAM_BLOCK_BYTES;
            bool first_cached=cache_has(s->cache,first);
            bool second_cached=plan[base].bytes==FG_NGRAM_MAX_READ_BYTES&&
                               cache_has(s->cache,second);
            if(first_cached&&(plan[base].bytes==FG_NGRAM_BLOCK_BYTES||second_cached)){
                base++;
                continue;
            }
            uint64_t offset=first;
            uint32_t bytes=plan[base].bytes;
            if(bytes==FG_NGRAM_MAX_READ_BYTES){
                if(first_cached){offset=second;bytes=FG_NGRAM_BLOCK_BYTES;}
                else if(second_cached){bytes=FG_NGRAM_BLOCK_BYTES;}
            }
            uint32_t slot=batch_count++;
            batch[slot]=(fg_uring_read){
                .buffer=s->io_buffer+(uint64_t)slot*FG_NGRAM_MAX_READ_BYTES,
                .bytes=bytes,.offset=offset};
            base++;
        }
        if(!batch_count)continue;
        s->last_read_count+=batch_count;
        for(uint32_t i=0;i<batch_count;i++)s->last_read_bytes+=batch[i].bytes;
        double io_start=ngram_ts();
        status=fg_uring_pread_batch(s->ring,s->slot,batch,batch_count,err);
        s->last_io_ms+=ngram_ts()-io_start;
        for(uint32_t i=0;status==FG_OK&&i<batch_count;i++){
            status=fg_ngram_cache_put(s->cache,batch[i].offset,batch[i].buffer,err);
            if(status==FG_OK&&batch[i].bytes==FG_NGRAM_MAX_READ_BYTES)
                status=fg_ngram_cache_put(s->cache,batch[i].offset+FG_NGRAM_BLOCK_BYTES,
                    (const uint8_t *)batch[i].buffer+FG_NGRAM_BLOCK_BYTES,err);
        }
    }
    return status;
}

static fg_status get_or_reload_block(fg_ngram_store *s,uint64_t block,const void **data,fg_error *err){
    fg_status status=ensure_cache(s,err);
    if(status!=FG_OK)return status;
    if(fg_ngram_cache_get(s->cache,block,data))return FG_OK;
    double io_start=ngram_ts();
    status=fg_uring_pread(s->ring,s->slot,s->io_buffer,FG_NGRAM_BLOCK_BYTES,block,err);
    s->last_io_ms+=ngram_ts()-io_start;
    if(status==FG_OK){s->last_read_count++;s->last_read_bytes+=FG_NGRAM_BLOCK_BYTES;status=fg_ngram_cache_put(s->cache,block,s->io_buffer,err);}
    if(status==FG_OK&&!fg_ngram_cache_get(s->cache,block,data)){fg_error_set(err,FG_ERR_MISMATCH,"n-gram cache rejected a reloaded block");status=FG_ERR_MISMATCH;}
    return status;
}

static fg_status pack_rows(fg_ngram_store *s,const uint64_t *addresses,uint32_t row_count,fg_error *err){
    fg_status status=ensure_tensors(s,err);
    if(status!=FG_OK)return status;
    uint8_t *packed=fg_vk_tensor_map(s->packed);if(!packed){fg_error_set(err,FG_ERR_OOM,"map n-gram packed prefill tensor");return FG_ERR_OOM;}
    for(uint32_t i=0;i<row_count;i++){uint64_t block=addresses[i]&~(uint64_t)(FG_NGRAM_BLOCK_BYTES-1u);uint32_t within=(uint32_t)(addresses[i]-block),first=FG_NGRAM_ROW_BYTES;if(first>FG_NGRAM_BLOCK_BYTES-within)first=FG_NGRAM_BLOCK_BYTES-within;const void *data=NULL;status=get_or_reload_block(s,block,&data,err);if(status!=FG_OK)return status;memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,(const uint8_t *)data+within,first);if(first<FG_NGRAM_ROW_BYTES){status=get_or_reload_block(s,block+FG_NGRAM_BLOCK_BYTES,&data,err);if(status!=FG_OK)return status;memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES+first,data,FG_NGRAM_ROW_BYTES-first);}}
    return FG_OK;
}

fg_status fg_ngram_store_lookup_prefill(fg_ngram_store *s,const int32_t *tokens,size_t history_count,uint32_t first_token,uint32_t token_count,fg_vk_tensor **embedding,fg_error *err){
    if(!s||!tokens||!history_count||token_count==0u||token_count>s->max_tokens||!embedding||first_token>history_count||(size_t)token_count>history_count-(size_t)first_token){fg_error_set(err,FG_ERR_ARGUMENT,"invalid bounded n-gram prefill lookup range");return FG_ERR_ARGUMENT;}
    double trace_start=ngram_ts();uint64_t addresses[FG_NGRAM_PREFILL_MAX_ROWS],rows[FG_NGRAM_PREFILL_MAX_ROWS];size_t end=(size_t)first_token+token_count;fg_status status=FG_OK;
    for(size_t i=0;i<end;i++)if(tokens[i]<0||tokens[i]>=248320){fg_error_set(err,FG_ERR_FORMAT,"token %zu is outside Qwen3.8 vocabulary",i);return FG_ERR_FORMAT;}
    size_t segment_start=0;for(size_t i=0;i<(size_t)first_token;i++)if((uint32_t)tokens[i]==FG_Q38_EOS_TOKEN)segment_start=i+1u;for(uint32_t token=0;token<token_count;token++){size_t pos=(size_t)first_token+token;q38_ngram_fill(tokens,pos+1u,segment_start,rows+(uint64_t)token*FG_NGRAM_HEAD_COUNT,addresses+(uint64_t)token*FG_NGRAM_HEAD_COUNT);if((uint32_t)tokens[pos]==FG_Q38_EOS_TOKEN)segment_start=pos+1u;}
    if(ngram_locality_trace_enabled())for(uint32_t token=0;token<token_count;token++){char line[768];int used=snprintf(line,sizeof(line),"NGRAM_LOCALITY position=%u batch_first=%u batch_tokens=%u",first_token+token,first_token,token_count);for(uint32_t head=0;used>0&&(size_t)used<sizeof(line)&&head<FG_NGRAM_HEAD_COUNT;head++){int written=snprintf(line+(size_t)used,sizeof(line)-(size_t)used," h%u=%llu",head,(unsigned long long)addresses[(uint64_t)token*FG_NGRAM_HEAD_COUNT+head]);if(written<0||(size_t)written>=sizeof(line)-(size_t)used){used=-1;break;}used+=written;}if(used>0&&(size_t)used+1u<sizeof(line)){line[used++]='\n';line[used]=0;fputs(line,stderr);}}
    uint32_t row_count=token_count*FG_NGRAM_HEAD_COUNT;double hash_end=ngram_ts();if(status==FG_OK)status=load_missing_blocks(s,addresses,row_count,err);double load_end=ngram_ts();if(status==FG_OK)status=pack_rows(s,addresses,row_count,err);double pack_end=ngram_ts();if(status==FG_OK)status=fg_vk_dequantize_iq4_nl(s->vk,s->embedding,s->packed,row_count,FG_NGRAM_EMBED_WIDTH,err);double dequant_end=ngram_ts();if(status==FG_OK){fg_vk_tensor_destroy(s->embedding_view);s->embedding_view=NULL;uint64_t bytes=(uint64_t)token_count*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH*4u;if(bytes==fg_vk_tensor_bytes(s->embedding))*embedding=s->embedding;else{status=fg_vk_tensor_view(s->embedding,0,bytes,&s->embedding_view,err);if(status==FG_OK)*embedding=s->embedding_view;}}if(ngram_trace_enabled())fprintf(stderr,"NGRAM_TRACE first=%u tokens=%u rows=%u reads=%u bytes=%llu hash_ms=%.3f load_ms=%.3f io_ms=%.3f pack_ms=%.3f dequant_ms=%.3f total_ms=%.3f\n",first_token,token_count,row_count,s->last_read_count,(unsigned long long)s->last_read_bytes,hash_end-trace_start,load_end-hash_end,s->last_io_ms,pack_end-load_end,dequant_end-pack_end,ngram_ts()-trace_start);return status;
}

fg_status fg_ngram_store_lookup(fg_ngram_store *s,const int32_t *tokens,size_t count,fg_vk_tensor **embedding,fg_error *err){
    if(!count||count-1u>UINT32_MAX){fg_error_set(err,FG_ERR_LIMIT,"n-gram lookup history exceeds supported context index");return FG_ERR_LIMIT;}return fg_ngram_store_lookup_prefill(s,tokens,count,(uint32_t)(count-1u),1u,embedding,err);
}

fg_status fg_ngram_store_decode_packed(fg_ngram_store *s,const uint8_t *packed,uint32_t row_count,fg_vk_tensor **embedding,fg_error *err){if(!s||!packed||!row_count||row_count>s->max_rows||!embedding){fg_error_set(err,FG_ERR_ARGUMENT,"invalid packed n-gram decode");return FG_ERR_ARGUMENT;}fg_status status=ensure_tensors(s,err);if(status!=FG_OK)return status;uint64_t packed_bytes=(uint64_t)row_count*FG_NGRAM_ROW_BYTES;status=fg_vk_tensor_write(s->packed,0,packed,packed_bytes,err);if(status==FG_OK)status=fg_vk_dequantize_iq4_nl(s->vk,s->embedding,s->packed,row_count,FG_NGRAM_EMBED_WIDTH,err);if(status==FG_OK){fg_vk_tensor_destroy(s->embedding_view);s->embedding_view=NULL;uint64_t bytes=(uint64_t)row_count*FG_NGRAM_EMBED_WIDTH*4u;if(bytes==fg_vk_tensor_bytes(s->embedding))*embedding=s->embedding;else{status=fg_vk_tensor_view(s->embedding,0,bytes,&s->embedding_view,err);if(status==FG_OK)*embedding=s->embedding_view;}}return status;}

fg_status fg_ngram_store_verify_packed(fg_ngram_store *s,const uint64_t *addresses,uint32_t row_count,const uint8_t *packed,uint32_t *mismatch_row,fg_error *err){if(!s||!addresses||!row_count||row_count>s->max_rows||!packed){fg_error_set(err,FG_ERR_ARGUMENT,"invalid packed n-gram verification");return FG_ERR_ARGUMENT;}fg_status status=load_missing_blocks(s,addresses,row_count,err);if(status==FG_OK)status=pack_rows(s,addresses,row_count,err);if(status!=FG_OK)return status;const uint8_t *local=fg_vk_tensor_map(s->packed);for(uint32_t row=0;row<row_count;row++)if(memcmp(local+(uint64_t)row*FG_NGRAM_ROW_BYTES,packed+(uint64_t)row*FG_NGRAM_ROW_BYTES,FG_NGRAM_ROW_BYTES)!=0){if(mismatch_row)*mismatch_row=row;fg_error_set(err,FG_ERR_MISMATCH,"resident n-gram packed row %u differs from sealed table",row);return FG_ERR_MISMATCH;}if(mismatch_row)*mismatch_row=UINT32_MAX;return FG_OK;}
