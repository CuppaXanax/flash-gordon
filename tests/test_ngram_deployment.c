#include "fg_ngram.h"
#include "fg_pack.h"
#include "fg_q38_schema.h"
#include "fg_runtime.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;
static bool track_allocations;
static uint64_t tracked_allocations;

void *__real_malloc(size_t bytes);
void *__real_calloc(size_t count,size_t bytes);
void *__real_realloc(void *pointer,size_t bytes);
void *__wrap_malloc(size_t bytes){
    if(track_allocations)tracked_allocations++;
    return __real_malloc(bytes);
}
void *__wrap_calloc(size_t count,size_t bytes){
    if(track_allocations)tracked_allocations++;
    return __real_calloc(count,bytes);
}
void *__wrap_realloc(void *pointer,size_t bytes){
    if(track_allocations)tracked_allocations++;
    return __real_realloc(pointer,bytes);
}

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#expression); \
        failures++; \
    } \
} while (0)

static bool write_rows(const char *path,uint8_t seed,uint32_t rows){
    FILE *file=fopen(path,"wb");
    if(!file)return false;
    for(uint32_t row=0;row<rows;row++)
        for(uint32_t byte=0;byte<FG_NGRAM_ROW_BYTES;byte++)
            if(fputc((int)(seed+row*17u+byte*29u),file)==EOF){
                fclose(file);return false;
            }
    return fclose(file)==0;
}

static uint8_t expected_row_byte(uint8_t seed,uint64_t row,uint32_t byte){
    return (uint8_t)(seed+row*17u+byte*29u);
}

static bool packed_row_matches(const uint8_t *packed,uint8_t seed,uint64_t row){
    for(uint32_t byte=0u;byte<FG_NGRAM_ROW_BYTES;byte++)
        if(packed[byte]!=expected_row_byte(seed,row,byte))return false;
    return true;
}

static uint64_t test_splitmix64(uint64_t value){
    value+=UINT64_C(0x9e3779b97f4a7c15);
    value=(value^(value>>30u))*UINT64_C(0xbf58476d1ce4e5b9);
    value=(value^(value>>27u))*UINT64_C(0x94d049bb133111eb);
    return value^(value>>31u);
}

static uint64_t row_for_page(uint64_t page){
    uint64_t begin=(page*FG_NGRAM_BLOCK_BYTES+FG_NGRAM_ROW_BYTES-1u)/
        FG_NGRAM_ROW_BYTES;
    for(uint64_t row=begin;row<begin+FG_NGRAM_BLOCK_BYTES;row++){
        uint64_t offset=row*FG_NGRAM_ROW_BYTES;
        if(offset/FG_NGRAM_BLOCK_BYTES==page&&
           offset%FG_NGRAM_BLOCK_BYTES<=FG_NGRAM_BLOCK_BYTES-FG_NGRAM_ROW_BYTES)
            return row;
    }
    return UINT64_MAX;
}

static void test_pipeline_cache(const char *path,uint8_t seed,uint32_t rows,
                                const uint8_t hash[32],fg_error *error){
    const uint64_t cache_bytes=UINT64_C(256)*1024u;
    fg_ngram_pipeline_cache *cache=NULL;
    CHECK(fg_ngram_pipeline_cache_open_test(
        &cache,path,1000u,rows,hash,cache_bytes,error)==FG_OK);
    CHECK(cache!=NULL);
    if(!cache)return;
    CHECK(fg_ngram_pipeline_cache_host_bytes(cache)==cache_bytes);
    CHECK(fg_ngram_pipeline_cache_page_capacity(cache)>=4u);

    uint64_t cross=UINT64_MAX;
    for(uint64_t row=0u;row<rows;row++)
        if((row*FG_NGRAM_ROW_BYTES)%FG_NGRAM_BLOCK_BYTES>
           FG_NGRAM_BLOCK_BYTES-FG_NGRAM_ROW_BYTES){
            cross=row;break;
        }
    CHECK(cross!=UINT64_MAX);
    uint64_t request[FG_NGRAM_HEAD_COUNT];
    uint8_t packed[FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES];
    for(uint32_t i=0u;i<FG_NGRAM_HEAD_COUNT;i++)request[i]=1000u+cross;
    tracked_allocations=0u;track_allocations=true;
    CHECK(fg_ngram_pipeline_cache_read(
        cache,request,FG_NGRAM_HEAD_COUNT,packed,sizeof(packed),error)==FG_OK);
    track_allocations=false;
    CHECK(tracked_allocations==0u);
    CHECK(packed_row_matches(packed,seed,cross));
    CHECK(packed_row_matches(
        packed+(FG_NGRAM_HEAD_COUNT-1u)*FG_NGRAM_ROW_BYTES,seed,cross));
    fg_ngram_pipeline_cache_stats stats;
    fg_ngram_pipeline_cache_get_stats(cache,&stats);
    CHECK(stats.last_page_hits==0u);
    CHECK(stats.last_page_misses==2u);
    CHECK(stats.last_pages_read==2u);
    CHECK(stats.last_read_operations==1u);

    memset(packed,0,sizeof(packed));
    CHECK(fg_ngram_pipeline_cache_read(
        cache,request,FG_NGRAM_HEAD_COUNT,packed,sizeof(packed),error)==FG_OK);
    fg_ngram_pipeline_cache_get_stats(cache,&stats);
    CHECK(stats.last_page_hits==2u);
    CHECK(stats.last_page_misses==0u);
    CHECK(stats.last_pages_read==0u);
    CHECK(stats.last_read_operations==0u);
    CHECK(packed_row_matches(packed,seed,cross));
    fg_ngram_pipeline_cache_close(cache);

    cache=NULL;
    CHECK(fg_ngram_pipeline_cache_open_test(
        &cache,path,1000u,rows,hash,cache_bytes,error)==FG_OK);
    if(!cache)return;
    uint32_t sets=fg_ngram_pipeline_cache_page_capacity(cache)/4u;
    uint64_t selected[5];uint32_t selected_count=0u;
    for(uint64_t page=0u;page*FG_NGRAM_BLOCK_BYTES<
                            (uint64_t)rows*FG_NGRAM_ROW_BYTES&&
                            selected_count<5u;page++){
        uint64_t row=row_for_page(page);
        if(row<rows&&test_splitmix64(page)%sets==0u)
            selected[selected_count++]=row;
    }
    CHECK(selected_count==5u);
    uint8_t one[FG_NGRAM_ROW_BYTES];
    if(selected_count==5u){
        for(uint32_t i=0u;i<4u;i++){
            uint64_t one_row=1000u+selected[i];
            CHECK(fg_ngram_pipeline_cache_read(
                cache,&one_row,1u,one,sizeof(one),error)==FG_OK);
        }
        uint64_t first=1000u+selected[0];
        CHECK(fg_ngram_pipeline_cache_read(
            cache,&first,1u,one,sizeof(one),error)==FG_OK);
        uint64_t fifth=1000u+selected[4];
        CHECK(fg_ngram_pipeline_cache_read(
            cache,&fifth,1u,one,sizeof(one),error)==FG_OK);
        fg_ngram_pipeline_cache_get_stats(cache,&stats);
        uint64_t evictions=stats.evictions;
        CHECK(fg_ngram_pipeline_cache_read(
            cache,&first,1u,one,sizeof(one),error)==FG_OK);
        fg_ngram_pipeline_cache_get_stats(cache,&stats);
        CHECK(stats.last_page_hits==0u);
        CHECK(stats.last_page_misses==1u);
        CHECK(stats.evictions==evictions+1u);
        CHECK(packed_row_matches(one,seed,selected[0]));
    }
    fg_ngram_pipeline_cache_close(cache);
}

static bool write_empty_gguf(const char *path){
    FILE *file=fopen(path,"wb");
    if(!file)return false;
    uint32_t magic=0x46554747u,version=3u;
    uint64_t tensors=0u,metadata=0u;
    bool ok=fwrite(&magic,1,sizeof(magic),file)==sizeof(magic)&&
        fwrite(&version,1,sizeof(version),file)==sizeof(version)&&
        fwrite(&tensors,1,sizeof(tensors),file)==sizeof(tensors)&&
        fwrite(&metadata,1,sizeof(metadata),file)==sizeof(metadata);
    return fclose(file)==0&&ok;
}

static void fill_pipeline_ngram_metadata(fg_manifest *manifest){
    fg_error error={0};
    fg_manifest_init(manifest);
    CHECK(fg_runtime_profile_apply(
        manifest,FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,&error)==FG_OK);
    fg_tensor_record *full=&manifest->tensors[manifest->tensor_count++];
    snprintf(full->name,sizeof(full->name),"per_layer_token_embd.weight");
    full->bytes=UINT64_C(320001536)*FG_NGRAM_ROW_BYTES;
    full->ggml_type=20u;full->dims=2u;
    full->shape[0]=FG_NGRAM_EMBED_WIDTH;full->shape[1]=UINT64_C(320001536);
    full->rank=UINT16_MAX;full->layer=UINT16_MAX;full->expert=UINT16_MAX;
    full->kind=FG_TENSOR_NGRAM;full->sha256[0]=1u;
    manifest->flags|=FG_MANIFEST_HAS_NGRAM;
    for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){
        fg_ngram_shard_record *record=
            &manifest->ngram_shards[manifest->ngram_shard_count++];
        record->logical_rank=rank;
        CHECK(fg_q38_ngram_rank_range(rank,&record->row_begin,
                                      &record->row_count,&error)==FG_OK);
        record->bytes=record->row_count*FG_NGRAM_ROW_BYTES;
        record->sha256[0]=(uint8_t)rank;
        manifest->host_resident_bytes[rank]=FG_PIPELINE_NGRAM_CACHE_BYTES;
    }
}

int main(void){
    char first[128],second[128],large[128],missing_dir[128],source[128],pack_dir[128];
    snprintf(first,sizeof(first),"test-ngram-sealed-a-%ld.iq4nl",(long)getpid());
    snprintf(second,sizeof(second),"test-ngram-sealed-b-%ld.iq4nl",(long)getpid());
    snprintf(large,sizeof(large),"test-ngram-cache-%ld.iq4nl",(long)getpid());
    snprintf(missing_dir,sizeof(missing_dir),"test-ngram-missing-%ld",(long)getpid());
    snprintf(source,sizeof(source),"test-ngram-pack-source-%ld.gguf",(long)getpid());
    snprintf(pack_dir,sizeof(pack_dir),"test-ngram-pack-%ld",(long)getpid());
    unlink(first);unlink(second);unlink(large);unlink(source);rmdir(missing_dir);rmdir(pack_dir);
    CHECK(write_rows(first,3u,4u));
    CHECK(write_rows(second,91u,4u));
    uint8_t first_hash[32],second_hash[32];
    fg_error error={0};
    CHECK(fg_sha256_file(first,first_hash,&error)==FG_OK);
    CHECK(fg_sha256_file(second,second_hash,&error)==FG_OK);
    fg_ngram_resident *resident=NULL;
    CHECK(fg_ngram_resident_open_sealed(
        &resident,first,100u,4u,first_hash,&error)==FG_OK);
    uint64_t row=102u;uint8_t packed[FG_NGRAM_ROW_BYTES];
    CHECK(fg_ngram_resident_read(
        resident,&row,1u,packed,sizeof(packed),&error)==FG_OK);
    CHECK(packed_row_matches(packed,3u,2u));
    fg_ngram_resident_close(resident);resident=NULL;
    fg_ngram_pipeline_cache *pipeline_cache=NULL;
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,first,100u,4u,first_hash,
        UINT64_C(256)*1024u,&error)==FG_OK);
    fg_ngram_pipeline_cache_close(pipeline_cache);pipeline_cache=NULL;
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,first,100u,4u,second_hash,
        UINT64_C(256)*1024u,&error)==FG_ERR_MISMATCH);
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,first,100u,4u,first_hash,
        FG_NGRAM_BLOCK_BYTES,&error)==FG_ERR_LIMIT);
    CHECK(fg_ngram_resident_open_sealed(
        &resident,"test-ngram-does-not-exist.iq4nl",100u,4u,
        first_hash,&error)==FG_ERR_IO);
    CHECK(fg_ngram_resident_open_sealed(
        &resident,first,100u,4u,second_hash,&error)==FG_ERR_MISMATCH);
    CHECK(write_rows(second,91u,3u));
    CHECK(fg_ngram_resident_open_sealed(
        &resident,second,100u,4u,second_hash,&error)==FG_ERR_MISMATCH);
    CHECK(write_rows(second,91u,4u));
    FILE *corrupt=fopen(first,"r+b");CHECK(corrupt!=NULL);
    if(corrupt){CHECK(fputc(0xff,corrupt)!=EOF);CHECK(fclose(corrupt)==0);}
    CHECK(fg_ngram_resident_open_sealed(
        &resident,first,100u,4u,first_hash,&error)==FG_ERR_MISMATCH);
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,first,100u,4u,first_hash,
        UINT64_C(256)*1024u,&error)==FG_ERR_MISMATCH);
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,second,100u,4u,second_hash,
        UINT64_C(256)*1024u,&error)==FG_OK);
    fg_ngram_pipeline_cache_close(pipeline_cache);pipeline_cache=NULL;
    CHECK(truncate(second,3u*FG_NGRAM_ROW_BYTES)==0);
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,second,100u,4u,second_hash,
        UINT64_C(256)*1024u,&error)==FG_ERR_MISMATCH);

    CHECK(write_rows(large,37u,5000u));
    uint8_t large_hash[32];
    CHECK(fg_sha256_file(large,large_hash,&error)==FG_OK);
    test_pipeline_cache(large,37u,5000u,large_hash,&error);
    CHECK(fg_ngram_pipeline_cache_open_test(
        &pipeline_cache,large,1000u,5000u,large_hash,
        UINT64_C(256)*1024u,&error)==FG_OK);
    if(pipeline_cache){
        CHECK(truncate(large,0u)==0);
        uint64_t last=5999u;uint8_t last_row[FG_NGRAM_ROW_BYTES];
        CHECK(fg_ngram_pipeline_cache_read(
            pipeline_cache,&last,1u,last_row,sizeof(last_row),&error)==FG_ERR_IO);
        fg_ngram_pipeline_cache_close(pipeline_cache);pipeline_cache=NULL;
    }

    fg_manifest manifest;
    fill_pipeline_ngram_metadata(&manifest);
    CHECK(fg_q38_validate_ngram_shards(&manifest,&error)==FG_OK);
    const fg_ngram_shard_record *rank6=fg_q38_find_ngram_shard(&manifest,6u);
    CHECK(rank6&&rank6->row_begin==UINT64_C(226667743)&&
          rank6->row_count==UINT64_C(46666896));
    uint64_t resident_bytes=0u;
    manifest.ranks[6].persistent_bytes=11u;
    manifest.ranks[6].transient_bytes=13u;
    manifest.ranks[6].kv_bytes=17u;
    manifest.ranks[6].scratch_bytes=19u;
    manifest.ranks[6].driver_reserve_bytes=23u;
    CHECK(fg_q38_rank_residency_bytes(
        &manifest,6u,&resident_bytes,&error)==FG_OK&&
        resident_bytes==manifest.host_resident_bytes[6]+83u);
    uint64_t saved=manifest.ngram_shards[5].row_begin;
    manifest.ngram_shards[5].row_begin=manifest.ngram_shards[4].row_begin;
    CHECK(fg_q38_validate_ngram_shards(&manifest,&error)==FG_ERR_MISMATCH);
    manifest.ngram_shards[5].row_begin=saved;
    CHECK(mkdir(missing_dir,0700)==0);
    CHECK(fg_ngram_resident_open_manifest(
        &resident,&manifest,missing_dir,6u,&error)==FG_ERR_IO);

    CHECK(write_empty_gguf(source));
    CHECK(mkdir(pack_dir,0700)==0);
    char blocked[256],temporary[256];
    snprintf(blocked,sizeof(blocked),"%s/rank-03.fgw",pack_dir);
    FILE *existing=fopen(blocked,"wb");CHECK(existing!=NULL);
    if(existing)CHECK(fclose(existing)==0);
    const char *sources[]={source};
    fg_pack_options options={.output_dir=pack_dir,.source_paths=sources,
        .source_count=1u,.runtime_profile=FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,
        .skip_model_validation=true};
    CHECK(fg_pack_run(&options,&error)==FG_ERR_IO);
    struct stat info;CHECK(stat(blocked,&info)==0);
    snprintf(temporary,sizeof(temporary),"%s/rank-00.fgw.tmp.%ld",
             pack_dir,(long)getpid());
    CHECK(stat(temporary,&info)!=0);
    snprintf(temporary,sizeof(temporary),"%s/rank-00.fgw",pack_dir);
    CHECK(stat(temporary,&info)!=0);

    unlink(blocked);unlink(source);unlink(first);unlink(second);unlink(large);
    rmdir(pack_dir);rmdir(missing_dir);
    if(failures){
        fprintf(stderr,"%d n-gram deployment test(s) failed\n",failures);
        return 1;
    }
    puts("Flash Gordon sealed n-gram deployment: PASS");
    return 0;
}
