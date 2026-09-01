#include "fg_pack.h"
#include "fg_ngram.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"
#include "fg_runtime.h"
#include "fg_sha256.h"
#include "fg_topology.h"
#include "fg_tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct pack_output{
    FILE *file;
    char path[1024];
    char final_path[1024];
    uint64_t offset;
    bool committed;
}pack_output;
static fg_status pad_to(pack_output *out,uint64_t target,fg_error *err);

typedef struct canonical_shard {
    uint64_t bytes;
    const char *sha256_hex;
} canonical_shard;

static const canonical_shard q38_ud_q4_k_xl[4]={
    {UINT64_C(10946624),"4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"},
    {UINT64_C(49859583136),"3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"},
    {UINT64_C(49376141504),"56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"},
    {UINT64_C(12087983520),"753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"}
};

static uint8_t hex_nibble(char value){
    if(value>='0'&&value<='9')return (uint8_t)(value-'0');
    if(value>='a'&&value<='f')return (uint8_t)(value-'a'+10);
    return UINT8_MAX;
}

static bool canonical_digest(uint32_t shard,uint8_t digest[32]){
    if(shard>=4u)return false;
    const char *hex=q38_ud_q4_k_xl[shard].sha256_hex;
    for(uint32_t i=0;i<32u;i++){uint8_t hi=hex_nibble(hex[i*2u]),lo=hex_nibble(hex[i*2u+1u]);if(hi==UINT8_MAX||lo==UINT8_MAX)return false;digest[i]=(uint8_t)((hi<<4u)|lo);}return hex[64]==0;
}

static fg_status hash_pack_sources(const fg_pack_options *options,uint8_t composite[32],fg_error *err){
    if(!options->skip_model_validation&&options->source_count!=4u){fg_error_set(err,FG_ERR_MISMATCH,"canonical UD-Q4_K_XL pack requires exactly four shards");return FG_ERR_MISMATCH;}
    fg_sha256 source_hash;fg_sha256_init(&source_hash);
    for(uint32_t shard=0;shard<options->source_count;shard++){
        struct stat info;if(stat(options->source_paths[shard],&info)!=0||info.st_size<0){fg_error_set(err,FG_ERR_IO,"stat source %s: %s",options->source_paths[shard],strerror(errno));return FG_ERR_IO;}
        uint8_t digest[32];
        if(!options->skip_model_validation){
            if((uint64_t)info.st_size!=q38_ud_q4_k_xl[shard].bytes||!canonical_digest(shard,digest)){fg_error_set(err,FG_ERR_MISMATCH,"canonical UD-Q4_K_XL shard %u has the wrong size",shard+1u);return FG_ERR_MISMATCH;}
            if(!options->dry_run){uint8_t expected[32];fg_status status=fg_sha256_file(options->source_paths[shard],digest,err);if(status!=FG_OK)return status;if(!canonical_digest(shard,expected)||memcmp(digest,expected,32)!=0){fg_error_set(err,FG_ERR_MISMATCH,"canonical UD-Q4_K_XL shard %u SHA-256 mismatch",shard+1u);return FG_ERR_MISMATCH;}}
        }else if(options->dry_run){
            uint64_t bytes=(uint64_t)info.st_size;fg_sha256_update(&source_hash,&bytes,sizeof(bytes));fg_sha256_update(&source_hash,options->source_paths[shard],strlen(options->source_paths[shard]));continue;
        }else{fg_status status=fg_sha256_file(options->source_paths[shard],digest,err);if(status!=FG_OK)return status;}
        fg_sha256_update(&source_hash,digest,32);
    }
    fg_sha256_final(&source_hash,composite);
    if(options->dry_run&&!options->skip_model_validation)fprintf(stderr,"pack dry-run: canonical shard sizes/schema verified; payload SHA-256 verification is deferred to the full pack\n");
    return FG_OK;
}

static fg_status mkdir_one(const char *p,fg_error *err){if(mkdir(p,0755)==0||errno==EEXIST)return FG_OK;fg_error_set(err,FG_ERR_IO,"mkdir %s: %s",p,strerror(errno));return FG_ERR_IO;}

static fg_status open_pack_output(pack_output *out,const char *final_path,
                                  bool require_absent,fg_error *err){
    struct stat existing;
    if(require_absent&&(stat(final_path,&existing)==0||errno!=ENOENT)){
        fg_error_set(err,FG_ERR_IO,"pack output already exists: %s",final_path);
        return FG_ERR_IO;
    }
    if(snprintf(out->final_path,sizeof(out->final_path),"%s",final_path)>=
       (int)sizeof(out->final_path)||
       snprintf(out->path,sizeof(out->path),"%s.tmp.%ld",final_path,(long)getpid())>=
       (int)sizeof(out->path)){
        fg_error_set(err,FG_ERR_LIMIT,"pack output path is too long");
        return FG_ERR_LIMIT;
    }
    out->file=fopen(out->path,"wbx");
    if(!out->file){
        fg_error_set(err,FG_ERR_IO,"create %s: %s",out->path,strerror(errno));
        return FG_ERR_IO;
    }
    return FG_OK;
}

static fg_status open_outputs(const fg_pack_options *o,
                              pack_output rank[FG_RANK_COUNT],
                              pack_output *ngram,pack_output *embedding,
                              pack_output shards[FG_NGRAM_SHARD_COUNT],
                              fg_error *err){
    if(o->dry_run)return FG_OK;
    if(mkdir_one(o->output_dir,err)!=FG_OK)return err->code;
    bool pipeline=o->runtime_profile==FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K;
    char path[1024];
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){
        if(snprintf(path,sizeof(path),"%s/rank-%02u.fgw",o->output_dir,r)>=
           (int)sizeof(path)){
            fg_error_set(err,FG_ERR_LIMIT,"rank output path is too long");
            return FG_ERR_LIMIT;
        }
        fg_status status=open_pack_output(&rank[r],path,pipeline,err);
        if(status!=FG_OK)return status;
    }
    if(snprintf(path,sizeof(path),"%s/ngram.iq4nl",o->output_dir)>=(int)sizeof(path)){
        fg_error_set(err,FG_ERR_LIMIT,"n-gram output path is too long");
        return FG_ERR_LIMIT;
    }
    fg_status status=open_pack_output(ngram,path,pipeline,err);
    if(status!=FG_OK)return status;
    if(o->runtime_profile==FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K){
        for(uint32_t rank_index=1u;rank_index<FG_RANK_COUNT;rank_index++){
            if(snprintf(path,sizeof(path),"%s/" FG_NGRAM_SHARD_ARTIFACT_FORMAT,
                        o->output_dir,rank_index)>=(int)sizeof(path)){
                fg_error_set(err,FG_ERR_LIMIT,
                             "resident n-gram shard output path is too long");
                return FG_ERR_LIMIT;
            }
            status=open_pack_output(&shards[rank_index-1u],path,true,err);
            if(status!=FG_OK)return status;
        }
        if(snprintf(path,sizeof(path),"%s/%s",o->output_dir,
                    FG_TOKEN_EMBEDDING_ARTIFACT)>=(int)sizeof(path)){
            fg_error_set(err,FG_ERR_LIMIT,
                         "token embedding output path is too long");
            return FG_ERR_LIMIT;
        }
        status=open_pack_output(embedding,path,true,err);
        if(status!=FG_OK)return status;
    }
    return FG_OK;
}
static fg_status finalize_output(pack_output *out,fg_error *err){fg_status rc=pad_to(out,fg_align_up_u64(out->offset,FG_ALIGNMENT),err);if(!out->file)return rc;if(rc==FG_OK&&(fflush(out->file)!=0||fsync(fileno(out->file))!=0)){fg_error_set(err,FG_ERR_IO,"flush %s: %s",out->path,strerror(errno));rc=FG_ERR_IO;}if(fclose(out->file)!=0&&rc==FG_OK){fg_error_set(err,FG_ERR_IO,"close %s: %s",out->path,strerror(errno));rc=FG_ERR_IO;}out->file=NULL;return rc;}
static fg_status finalize_exact_output(pack_output *out,fg_error *err){if(!out->file)return FG_OK;fg_status rc=FG_OK;if(fflush(out->file)!=0||fsync(fileno(out->file))!=0){fg_error_set(err,FG_ERR_IO,"flush %s: %s",out->path,strerror(errno));rc=FG_ERR_IO;}if(fclose(out->file)!=0&&rc==FG_OK){fg_error_set(err,FG_ERR_IO,"close %s: %s",out->path,strerror(errno));rc=FG_ERR_IO;}out->file=NULL;return rc;}
static fg_status close_outputs(pack_output rank[FG_RANK_COUNT],pack_output *ngram,
                               pack_output *embedding,
                               pack_output shards[FG_NGRAM_SHARD_COUNT],
                               fg_error *err){
    fg_status rc=FG_OK;
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){
        fg_status one=finalize_output(&rank[r],err);if(rc==FG_OK)rc=one;
    }
    fg_status one=finalize_output(ngram,err);if(rc==FG_OK)rc=one;
    one=finalize_exact_output(embedding,err);if(rc==FG_OK)rc=one;
    for(uint32_t i=0;i<FG_NGRAM_SHARD_COUNT;i++){
        one=finalize_exact_output(&shards[i],err);if(rc==FG_OK)rc=one;
    }
    return rc;
}

static void discard_output(pack_output *out){
    if(!out)return;
    if(out->file){fclose(out->file);out->file=NULL;}
    if(out->path[0])unlink(out->path);
    if(out->committed&&out->final_path[0])unlink(out->final_path);
}

static void discard_outputs(pack_output rank[FG_RANK_COUNT],pack_output *ngram,
                            pack_output *embedding,
                            pack_output shards[FG_NGRAM_SHARD_COUNT]){
    for(uint32_t r=0;r<FG_RANK_COUNT;r++)discard_output(&rank[r]);
    discard_output(ngram);discard_output(embedding);
    for(uint32_t i=0;i<FG_NGRAM_SHARD_COUNT;i++)discard_output(&shards[i]);
}

static fg_status commit_output(pack_output *out,fg_error *err){
    if(!out->path[0])return FG_OK;
    if(rename(out->path,out->final_path)!=0){
        fg_error_set(err,FG_ERR_IO,"rename %s to %s: %s",out->path,
                     out->final_path,strerror(errno));
        return FG_ERR_IO;
    }
    out->committed=true;
    return FG_OK;
}

static fg_status commit_outputs(pack_output rank[FG_RANK_COUNT],
                                pack_output *ngram,pack_output *embedding,
                                pack_output shards[FG_NGRAM_SHARD_COUNT],
                                fg_error *err){
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){
        fg_status status=commit_output(&rank[r],err);if(status!=FG_OK)return status;
    }
    fg_status status=commit_output(ngram,err);if(status!=FG_OK)return status;
    status=commit_output(embedding,err);if(status!=FG_OK)return status;
    for(uint32_t i=0;i<FG_NGRAM_SHARD_COUNT;i++){
        status=commit_output(&shards[i],err);if(status!=FG_OK)return status;
    }
    return FG_OK;
}
static fg_status pad_to(pack_output *out,uint64_t target,fg_error *err){static const uint8_t zero[FG_ALIGNMENT]={0};while(out->offset<target){size_t n=(size_t)(target-out->offset);if(n>sizeof(zero))n=sizeof(zero);if(out->file&&fwrite(zero,1,n,out->file)!=n){fg_error_set(err,FG_ERR_IO,"pad %s: %s",out->path,strerror(errno));return FG_ERR_IO;}out->offset+=n;}return FG_OK;}
static fg_status copy_range(FILE *src,uint64_t offset,uint64_t bytes,pack_output *out,fg_sha256 *hash,fg_error *err){if(!out->file){uint8_t descriptor[16];memcpy(descriptor,&offset,8);memcpy(descriptor+8,&bytes,8);fg_sha256_update(hash,descriptor,sizeof(descriptor));out->offset+=bytes;return FG_OK;}uint8_t *buf=malloc(1u<<20);if(!buf){fg_error_set(err,FG_ERR_OOM,"allocate pack copy buffer");return FG_ERR_OOM;}if(fseeko(src,(off_t)offset,SEEK_SET)!=0){free(buf);fg_error_set(err,FG_ERR_IO,"seek source: %s",strerror(errno));return FG_ERR_IO;}while(bytes){size_t n=bytes>(1u<<20)?(1u<<20):(size_t)bytes;if(fread(buf,1,n,src)!=n||fwrite(buf,1,n,out->file)!=n){free(buf);fg_error_set(err,FG_ERR_IO,"copy tensor: %s",ferror(src)?"unexpected end of source":strerror(errno));return FG_ERR_IO;}fg_sha256_update(hash,buf,n);out->offset+=n;bytes-=n;}free(buf);return FG_OK;}

static void hash_range_descriptor(fg_sha256 *hash,uint64_t offset,uint64_t bytes){
    uint8_t descriptor[16];
    for(uint32_t i=0;i<8u;i++){
        descriptor[i]=(uint8_t)(offset>>(i*8u));
        descriptor[8u+i]=(uint8_t)(bytes>>(i*8u));
    }
    fg_sha256_update(hash,descriptor,sizeof(descriptor));
}
static fg_status load_profile(const char *path,double profile[FG_LAYER_COUNT][FG_EXPERT_COUNT],fg_error *err){FILE *f=fopen(path,"r");if(!f){fg_error_set(err,FG_ERR_IO,"open router profile %s: %s",path,strerror(errno));return FG_ERR_IO;}unsigned l,e;double v;while(fscanf(f,"%u %u %lf",&l,&e,&v)==3){if(l>=FG_LAYER_COUNT||e>=FG_EXPERT_COUNT||v<0){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"invalid router profile row");return FG_ERR_FORMAT;}profile[l][e]=v;}if(!feof(f)){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"malformed router profile");return FG_ERR_FORMAT;}fclose(f);return FG_OK;}
static fg_status record_segment(fg_manifest *m,const fg_gguf_tensor *source,const char *name,uint64_t start,uint64_t bytes,uint32_t rank,uint32_t layer,uint32_t expert,fg_tensor_kind kind,fg_tensor_layout layout,uint64_t local_experts,fg_sha256 *hash,fg_error *err){fg_tensor_record r={0};snprintf(r.name,sizeof(r.name),"%s",name);r.offset=start;r.bytes=bytes;r.ggml_type=source->type;r.dims=source->dims;memcpy(r.shape,source->shape,sizeof(r.shape));if(local_experts)r.shape[r.dims-1]=local_experts;r.rank=(uint16_t)rank;r.layer=(uint16_t)(layer<FG_LAYER_COUNT?layer:UINT16_MAX);r.expert=(uint16_t)(expert<FG_EXPERT_COUNT?expert:UINT16_MAX);r.kind=(uint8_t)kind;r.layout=(uint8_t)layout;fg_sha256_final(hash,r.sha256);fg_status rc=fg_manifest_add_tensor(m,&r,err);if(rc==FG_OK){if(kind==FG_TENSOR_COMMON||kind==FG_TENSOR_ROUTED_EXPERT||kind==FG_TENSOR_HOST_CACHE)m->flags|=FG_MANIFEST_HAS_TEXT;else if(kind==FG_TENSOR_NGRAM)m->flags|=FG_MANIFEST_HAS_NGRAM;else if(kind==FG_TENSOR_VISION)m->flags|=FG_MANIFEST_HAS_VISION;else if(kind==FG_TENSOR_MTP)m->flags|=FG_MANIFEST_HAS_MTP;else if(kind==FG_TENSOR_TOKENIZER)m->flags|=FG_MANIFEST_HAS_TOKENIZER;}if(rc==FG_OK&&rank<FG_RANK_COUNT&&kind!=FG_TENSOR_HOST_CACHE){m->ranks[rank].tensor_count++;m->ranks[rank].persistent_bytes+=fg_align_up_u64(bytes,FG_ALIGNMENT);}return rc;}

static bool use_cooked_q8_0(const fg_gguf_tensor *tensor){if(tensor->type!=8u||tensor->dims!=2u||tensor->shape[0]>UINT32_MAX||tensor->shape[1]>UINT32_MAX||tensor->shape[0]%FG_QK8_0||tensor->shape[1]<2560u||strcmp(tensor->name,"token_embd.weight")==0)return false;uint64_t bytes=fg_q8_0_cooked_matrix_bytes((uint32_t)tensor->shape[0],(uint32_t)tensor->shape[1]);return bytes&&bytes<=UINT32_MAX;}

static fg_status process_cooked_q8_0(FILE *source,const fg_gguf_tensor *tensor,FILE *destination,fg_sha256 *hash,fg_error *err){
    uint64_t source_row=(tensor->shape[0]/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES,tile_bytes=fg_q8_0_cooked_tile_bytes((uint32_t)tensor->shape[0]);
    if(!source_row||!tile_bytes||tensor->shape[0]>UINT32_MAX||tensor->shape[1]>UINT32_MAX||tensor->shape[1]>UINT64_MAX/source_row||tensor->bytes!=source_row*tensor->shape[1]){fg_error_set(err,FG_ERR_FORMAT,"invalid Q8_0 matrix %s for cooking",tensor->name);return FG_ERR_FORMAT;}
    uint8_t *packed=malloc((size_t)source_row*FG_Q8_0_COOK_ROWS),*cooked=malloc((size_t)tile_bytes);if(!packed||!cooked){free(cooked);free(packed);fg_error_set(err,FG_ERR_OOM,"allocate cooked Q8_0 conversion buffers");return FG_ERR_OOM;}
    if(fseeko(source,(off_t)tensor->offset,SEEK_SET)!=0){free(cooked);free(packed);fg_error_set(err,FG_ERR_IO,"seek Q8_0 tensor %s: %s",tensor->name,strerror(errno));return FG_ERR_IO;}
    fg_status status=FG_OK;for(uint64_t first=0;first<tensor->shape[1];first+=FG_Q8_0_COOK_ROWS){uint32_t rows=(uint32_t)(tensor->shape[1]-first);if(rows>FG_Q8_0_COOK_ROWS)rows=FG_Q8_0_COOK_ROWS;size_t packed_bytes=(size_t)source_row*rows;if(fread(packed,1,packed_bytes,source)!=packed_bytes||!fg_cook_q8_0_rows(packed,cooked,tile_bytes,(uint32_t)tensor->shape[0],rows)||(destination&&fwrite(cooked,1,(size_t)tile_bytes,destination)!=(size_t)tile_bytes)){fg_error_set(err,FG_ERR_IO,"cook Q8_0 tensor %s: %s",tensor->name,ferror(source)?"unexpected end of source":strerror(errno));status=FG_ERR_IO;break;}fg_sha256_update(hash,cooked,(size_t)tile_bytes);}
    free(cooked);free(packed);return status;
}

static fg_tensor_layout expert_layout(const fg_gguf_tensor *tensor){if(tensor->dims!=3u||tensor->shape[0]>UINT32_MAX||tensor->shape[1]>UINT32_MAX)return FG_TENSOR_LAYOUT_GGML;uint32_t input=(uint32_t)tensor->shape[0],output=(uint32_t)tensor->shape[1];if((tensor->type==12u||tensor->type==13u)&&fg_k_quant_cooked_matrix_bytes(input,output,tensor->type))return FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED;if(tensor->type==7u&&fg_q5_1_cooked_matrix_bytes(input,output))return FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED;return FG_TENSOR_LAYOUT_GGML;}
static bool cook_expert_data(const fg_gguf_tensor *tensor,fg_tensor_layout layout,const void *packed,void *cooked,uint64_t bytes){if(layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED)return fg_cook_k_quant_rows(packed,cooked,bytes,(uint32_t)tensor->shape[0],(uint32_t)tensor->shape[1],tensor->type);if(layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED)return fg_cook_q5_1_rows(packed,cooked,bytes,(uint32_t)tensor->shape[0],(uint32_t)tensor->shape[1]);return false;}

static fg_status process_cooked_expert(FILE *source,const fg_gguf_tensor *tensor,uint64_t offset,uint64_t bytes,fg_tensor_layout layout,pack_output *output,fg_sha256 *hash,fg_error *err){if(!output->file){uint64_t descriptor[3]={offset,bytes,layout};fg_sha256_update(hash,descriptor,sizeof(descriptor));output->offset+=bytes;return FG_OK;}uint8_t *packed=malloc((size_t)bytes),*cooked=malloc((size_t)bytes);if(!packed||!cooked){free(cooked);free(packed);fg_error_set(err,FG_ERR_OOM,"allocate cooked expert buffers");return FG_ERR_OOM;}if(fseeko(source,(off_t)offset,SEEK_SET)!=0||fread(packed,1,(size_t)bytes,source)!=(size_t)bytes){free(cooked);free(packed);fg_error_set(err,FG_ERR_IO,"read expert tensor %s: %s",tensor->name,ferror(source)?"unexpected end of source":strerror(errno));return FG_ERR_IO;}bool converted=layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED?fg_cook_k_quant_rows(packed,cooked,bytes,(uint32_t)tensor->shape[0],(uint32_t)tensor->shape[1],tensor->type):layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED?fg_cook_q5_1_rows(packed,cooked,bytes,(uint32_t)tensor->shape[0],(uint32_t)tensor->shape[1]):false;if(!converted||fwrite(cooked,1,(size_t)bytes,output->file)!=(size_t)bytes){free(cooked);free(packed);fg_error_set(err,FG_ERR_IO,"cook expert tensor %s: %s",tensor->name,converted?strerror(errno):"invalid layout");return FG_ERR_IO;}fg_sha256_update(hash,cooked,(size_t)bytes);output->offset+=bytes;free(cooked);free(packed);return FG_OK;}

static uint32_t common_owner(const fg_manifest *manifest,const fg_gguf_tensor *tensor,int layer){
    if(manifest->execution_mode==FG_EXECUTION_PIPELINE){
        if(layer>=0)return manifest->layer_owner[layer];
        if(strcmp(tensor->name,"output.weight")==0||
           strncmp(tensor->name,"output_hc_",10u)==0)
            return manifest->stage_ranks[manifest->stage_count-1u];
        return manifest->stage_ranks[0];
    }
    if(layer>=0)return (uint32_t)layer%FG_RANK_COUNT;
    if(strcmp(tensor->name,"token_embd.weight")==0)return 0u;
    if(strcmp(tensor->name,"output.weight")==0||
       strncmp(tensor->name,"output_hc_",10u)==0)return 4u;
    return 0u;
}

static fg_status copy_ngram_segment(FILE *source,uint64_t source_offset,
                                    uint64_t bytes,pack_output *full,
                                    fg_sha256 *full_hash,pack_output *shard,
                                    fg_sha256 *shard_hash,fg_error *err){
    uint8_t *buffer=malloc(1u<<20u);
    if(!buffer){
        fg_error_set(err,FG_ERR_OOM,"allocate n-gram shard copy buffer");
        return FG_ERR_OOM;
    }
    if(fseeko(source,(off_t)source_offset,SEEK_SET)!=0){
        free(buffer);fg_error_set(err,FG_ERR_IO,"seek n-gram tensor: %s",
                                  strerror(errno));
        return FG_ERR_IO;
    }
    uint64_t remaining=bytes;
    while(remaining){
        size_t request=remaining>(1u<<20u)?(1u<<20u):(size_t)remaining;
        if(fread(buffer,1,request,source)!=request||
           (full->file&&fwrite(buffer,1,request,full->file)!=request)||
           (shard->file&&fwrite(buffer,1,request,shard->file)!=request)){
            free(buffer);fg_error_set(err,FG_ERR_IO,
                "copy n-gram shard: %s",ferror(source)?"unexpected end of source":
                strerror(errno));
            return FG_ERR_IO;
        }
        fg_sha256_update(full_hash,buffer,request);
        fg_sha256_update(shard_hash,buffer,request);
        full->offset+=request;shard->offset+=request;remaining-=request;
    }
    free(buffer);
    return FG_OK;
}

static fg_status pack_pipeline_ngram(const fg_gguf_tensor *tensor,FILE *source,
                                     fg_manifest *manifest,pack_output *full,
                                     pack_output shards[FG_NGRAM_SHARD_COUNT],
                                     fg_error *err){
    const uint64_t total_rows=UINT64_C(320001536);
    if(tensor->type!=20u||tensor->dims!=2u||
       tensor->shape[0]!=FG_NGRAM_EMBED_WIDTH||
       tensor->shape[1]!=total_rows||tensor->bytes!=total_rows*FG_NGRAM_ROW_BYTES||
       full->offset){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline n-gram tensor does not have canonical row geometry");
        return FG_ERR_MISMATCH;
    }
    fg_sha256 full_hash;fg_sha256_init(&full_hash);
    if(!full->file){
        hash_range_descriptor(&full_hash,tensor->offset,tensor->bytes);
        full->offset=tensor->bytes;
    }
    manifest->ngram_shard_count=0u;
    for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){
        uint64_t row_begin=0u,row_count=0u;
        fg_status status=fg_q38_ngram_rank_range(rank,&row_begin,&row_count,err);
        if(status!=FG_OK)return status;
        uint64_t bytes=row_count*FG_NGRAM_ROW_BYTES;
        uint64_t source_offset=tensor->offset+row_begin*FG_NGRAM_ROW_BYTES;
        pack_output *shard=&shards[rank-1u];
        fg_sha256 shard_hash;fg_sha256_init(&shard_hash);
        if(full->file){
            status=copy_ngram_segment(source,source_offset,bytes,full,&full_hash,
                                      shard,&shard_hash,err);
            if(status!=FG_OK)return status;
        }else{
            hash_range_descriptor(&shard_hash,source_offset,bytes);
            shard->offset=bytes;
        }
        fg_ngram_shard_record *record=
            &manifest->ngram_shards[manifest->ngram_shard_count++];
        record->logical_rank=rank;
        record->row_begin=row_begin;
        record->row_count=row_count;
        record->bytes=bytes;
        fg_sha256_final(&shard_hash,record->sha256);
        manifest->host_resident_bytes[rank]=FG_PIPELINE_NGRAM_CACHE_BYTES;
        if(!full->file){
            char digest[65];fg_sha256_hex(record->sha256,digest);
            printf(FG_NGRAM_SHARD_ARTIFACT_FORMAT
                   " rows=[%llu,%llu) source_offset=%llu bytes=%llu "
                   "descriptor_sha256=%s\n",rank,
                   (unsigned long long)row_begin,
                   (unsigned long long)(row_begin+row_count),
                   (unsigned long long)source_offset,
                   (unsigned long long)bytes,digest);
        }
    }
    return record_segment(manifest,tensor,tensor->name,0u,tensor->bytes,
                          UINT16_MAX,UINT32_MAX,UINT32_MAX,FG_TENSOR_NGRAM,
                          FG_TENSOR_LAYOUT_GGML,0u,&full_hash,err);
}

static fg_status pack_common(const fg_gguf_tensor *t,FILE *src,fg_manifest *m,
                             pack_output rank[FG_RANK_COUNT],pack_output *ngram,
                             pack_output *embedding,
                             pack_output shards[FG_NGRAM_SHARD_COUNT],
                             fg_error *err){
    fg_tensor_kind kind=fg_gguf_tensor_kind(t->name);
    if(kind==FG_TENSOR_NGRAM&&m->execution_mode==FG_EXECUTION_PIPELINE)
        return pack_pipeline_ngram(t,src,m,ngram,shards,err);
    int layer=fg_gguf_tensor_layer(t->name);uint32_t owner=common_owner(m,t,layer);
    bool external_embedding=m->execution_mode==FG_EXECUTION_PIPELINE&&
        !strcmp(t->name,"token_embd.weight");
    if(external_embedding)kind=FG_TENSOR_HOST_CACHE;
    pack_output *out=kind==FG_TENSOR_NGRAM?ngram:
        external_embedding?embedding:&rank[owner];
    uint64_t start=fg_align_up_u64(out->offset,FG_ALIGNMENT);
    fg_status rc=pad_to(out,start,err);if(rc!=FG_OK)return rc;
    fg_sha256 hash;fg_sha256_init(&hash);
    fg_tensor_layout layout=external_embedding?FG_TENSOR_LAYOUT_HOST_Q8_0:
        use_cooked_q8_0(t)?FG_TENSOR_LAYOUT_Q8_0_COOKED:FG_TENSOR_LAYOUT_GGML;
    uint64_t bytes=t->bytes;
    if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED){
        bytes=fg_q8_0_cooked_matrix_bytes((uint32_t)t->shape[0],
                                         (uint32_t)t->shape[1]);
        if(!bytes){fg_error_set(err,FG_ERR_LIMIT,
            "cooked Q8_0 tensor %s is too large",t->name);return FG_ERR_LIMIT;}
        if(out->file)rc=process_cooked_q8_0(src,t,out->file,&hash,err);
        else{uint64_t descriptor[3]={t->offset,t->bytes,bytes};
            fg_sha256_update(&hash,descriptor,sizeof(descriptor));}
        if(rc==FG_OK)out->offset+=bytes;
    }else rc=copy_range(src,t->offset,t->bytes,out,&hash,err);
    if(rc!=FG_OK)return rc;
    rc=record_segment(m,t,t->name,start,bytes,
        kind==FG_TENSOR_NGRAM?UINT16_MAX:owner,
        layer<0?UINT32_MAX:(uint32_t)layer,UINT32_MAX,kind,layout,0,&hash,err);
    if(rc==FG_OK&&external_embedding){
        if(m->host_resident_bytes[owner]){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "pipeline stage 0 host embedding is duplicated");
            return FG_ERR_MISMATCH;
        }
        m->host_resident_bytes[owner]=bytes;
    }
    return rc;
}

static fg_status pack_expert_tensor(const fg_gguf *g,const fg_gguf_tensor *t,FILE *src,fg_manifest *m,pack_output rank[FG_RANK_COUNT],fg_error *err){
    (void)g;int layer=fg_gguf_tensor_layer(t->name);if(layer<0||t->shape[t->dims-1]!=FG_EXPERT_COUNT||t->bytes%FG_EXPERT_COUNT){fg_error_set(err,FG_ERR_FORMAT,"routed tensor %s is not a 512-expert layer tensor",t->name);return FG_ERR_FORMAT;}uint64_t expert_bytes=t->bytes/FG_EXPERT_COUNT;fg_tensor_layout layout=expert_layout(t);
    if(m->execution_mode==FG_EXECUTION_PIPELINE){
        uint32_t owner=m->layer_owner[layer];pack_output *out=&rank[owner];
        uint64_t start=fg_align_up_u64(out->offset,FG_ALIGNMENT);
        fg_status rc=pad_to(out,start,err);if(rc!=FG_OK)return rc;
        fg_sha256 hash;fg_sha256_init(&hash);
        if(layout==FG_TENSOR_LAYOUT_GGML)
            rc=copy_range(src,t->offset,t->bytes,out,&hash,err);
        else for(uint32_t expert=0;expert<FG_EXPERT_COUNT&&rc==FG_OK;expert++)
            rc=process_cooked_expert(src,t,t->offset+(uint64_t)expert*expert_bytes,
                                     expert_bytes,layout,out,&hash,err);
        if(rc!=FG_OK)return rc;
        return record_segment(m,t,t->name,start,t->bytes,owner,(uint32_t)layer,
                              UINT32_MAX,FG_TENSOR_ROUTED_EXPERT,layout,
                              FG_EXPERT_COUNT,&hash,err);
    }
    for(uint32_t gi=0;gi<FG_GROUP_SIZE;gi++){uint32_t r=m->layer_groups[layer][gi];pack_output *out=&rank[r];uint64_t start=fg_align_up_u64(out->offset,FG_ALIGNMENT);fg_status rc=pad_to(out,start,err);if(rc!=FG_OK)return rc;fg_sha256 hash;fg_sha256_init(&hash);uint32_t copied=0;
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)if(m->expert_rank[layer][e]==r){uint64_t offset=t->offset+(uint64_t)e*expert_bytes;rc=layout==FG_TENSOR_LAYOUT_GGML?copy_range(src,offset,expert_bytes,out,&hash,err):process_cooked_expert(src,t,offset,expert_bytes,layout,out,&hash,err);if(rc!=FG_OK)return rc;copied++;}
        if(copied!=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_FORMAT,"layer %d rank %u selected %u experts",layer,r,copied);return FG_ERR_FORMAT;}char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"%.80s.rank%u",t->name,r);rc=record_segment(m,t,name,start,expert_bytes*copied,r,(uint32_t)layer,UINT32_MAX,FG_TENSOR_ROUTED_EXPERT,layout,FG_EXPERTS_PER_RANK,&hash,err);if(rc!=FG_OK)return rc;
    }return FG_OK;
}

static fg_status validate_pack_memory(const fg_manifest *manifest,bool print,
                                      fg_error *err){
    fg_status result=FG_OK;
    uint32_t failed_rank=UINT32_MAX;
    uint64_t failed_total=0u;
    int64_t worst_persistent_margin=INT64_MAX,worst_residency_margin=INT64_MAX;
    uint32_t worst_persistent_rank=0u,worst_residency_rank=0u;
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        const fg_rank_record *record=&manifest->ranks[rank];
        uint64_t total=0u;
        fg_status status=fg_q38_rank_residency_bytes(
            manifest,rank,&total,err);
        if(status!=FG_OK)return status;
        if(print){
            uint64_t host_embedding=0u,host_ngram=0u;
            const fg_tensor_record *embedding=fg_q38_find_tensor(
                manifest,"token_embd.weight",rank);
            if(embedding&&embedding->kind==FG_TENSOR_HOST_CACHE)
                host_embedding=embedding->bytes;
            const fg_ngram_shard_record *ngram=
                fg_q38_find_ngram_shard(manifest,rank);
            if(ngram)host_ngram=manifest->host_resident_bytes[rank];
            uint64_t host_total=manifest->host_resident_bytes[rank];
            uint64_t vulkan_total=total-manifest->host_resident_bytes[rank];
            long long persistent_delta=(long long)record->persistent_bytes-
                (long long)manifest->persistent_cap_bytes;
            long long residency_delta=(long long)total-
                (long long)manifest->residency_cap_bytes;
            long long persistent_margin=-persistent_delta;
            long long residency_margin=-residency_delta;
            if(persistent_margin<worst_persistent_margin){
                worst_persistent_margin=persistent_margin;
                worst_persistent_rank=rank;
            }
            if(residency_margin<worst_residency_margin){
                worst_residency_margin=residency_margin;
                worst_residency_rank=rank;
            }
            printf("rank %u ledger gpu-persistent=%llu vulkan-transient=%llu "
                   "vulkan-kv=%llu vulkan-scratch=%llu vulkan-driver=%llu "
                   "vulkan-total=%llu host-embedding=%llu host-ngram-cache=%llu "
                   "host-total=%llu conservative-total=%llu persistent-cap=%llu "
                   "persistent-over-under=%+lld residency-cap=%llu "
                   "residency-over-under=%+lld persistent-remaining=%+lld "
                   "residency-remaining=%+lld\n",rank,
                   (unsigned long long)record->persistent_bytes,
                   (unsigned long long)record->transient_bytes,
                   (unsigned long long)record->kv_bytes,
                   (unsigned long long)record->scratch_bytes,
                   (unsigned long long)record->driver_reserve_bytes,
                   (unsigned long long)vulkan_total,
                   (unsigned long long)host_embedding,
                   (unsigned long long)host_ngram,
                   (unsigned long long)host_total,
                   (unsigned long long)total,
                   (unsigned long long)manifest->persistent_cap_bytes,
                   persistent_delta,
                   (unsigned long long)manifest->residency_cap_bytes,
                   residency_delta,persistent_margin,residency_margin);
        }
        if(print)
            printf("pipeline worst margins persistent-rank=%u persistent-remaining=%+lld "
                   "residency-rank=%u residency-remaining=%+lld\n",
                   worst_persistent_rank,(long long)worst_persistent_margin,
                   worst_residency_rank,(long long)worst_residency_margin);
        if(failed_rank==UINT32_MAX&&
           (record->persistent_bytes>manifest->persistent_cap_bytes||
            total>manifest->residency_cap_bytes)){
            failed_rank=rank;failed_total=total;result=FG_ERR_LIMIT;
        }
    }
    if(result!=FG_OK){
        const fg_rank_record *record=&manifest->ranks[failed_rank];
        long long persistent_delta=(long long)record->persistent_bytes-
            (long long)manifest->persistent_cap_bytes;
        long long residency_delta=(long long)failed_total-
            (long long)manifest->residency_cap_bytes;
        fg_error_set(err,FG_ERR_LIMIT,
                     "rank %u memory cap: gpu-persistent=%llu host-resident=%llu "
                     "transient=%llu kv=%llu scratch=%llu driver=%llu total=%llu "
                     "persistent-cap=%llu persistent-over-under=%+lld "
                     "residency-cap=%llu residency-over-under=%+lld",
                     failed_rank,(unsigned long long)record->persistent_bytes,
                     (unsigned long long)manifest->host_resident_bytes[failed_rank],
                     (unsigned long long)record->transient_bytes,
                     (unsigned long long)record->kv_bytes,
                     (unsigned long long)record->scratch_bytes,
                     (unsigned long long)record->driver_reserve_bytes,
                     (unsigned long long)failed_total,
                     (unsigned long long)manifest->persistent_cap_bytes,
                     persistent_delta,
                     (unsigned long long)manifest->residency_cap_bytes,
                     residency_delta);
    }
    return result;
}

fg_status fg_pack_run(const fg_pack_options *o,fg_error *err){
    if(!o||!o->output_dir||!o->source_paths||!o->source_count){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "pack requires --output and at least one --source");
        return FG_ERR_ARGUMENT;
    }
    if(o->router_profile_path&&o->expert_map_path){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "pack accepts only one of --router-profile and --expert-map");
        return FG_ERR_ARGUMENT;
    }
    if(o->runtime_profile!=FG_RUNTIME_PROFILE_NONE&&
       !fg_runtime_profile_definition_get(o->runtime_profile)){
        fg_error_set(err,FG_ERR_ARGUMENT,"unsupported runtime profile %u",
                     o->runtime_profile);
        return FG_ERR_ARGUMENT;
    }
    if(o->runtime_profile==FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K&&
       (o->router_profile_path||o->expert_map_path)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "pipeline packing does not accept router profiles or expert maps");
        return FG_ERR_UNAVAILABLE;
    }
    if(o->runtime_profile==FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K&&
       !o->dry_run&&!o->skip_model_validation){
        fg_pack_options preflight=*o;
        preflight.dry_run=true;
        fg_status preflight_status=fg_pack_run(&preflight,err);
        if(preflight_status!=FG_OK)return preflight_status;
    }
    fg_gguf g;
    fg_status rc=fg_gguf_open(o->source_paths,o->source_count,&g,err);
    if(rc!=FG_OK)return rc;
    if(!o->skip_model_validation){
        rc=fg_q38_validate_gguf(&g,err);
        if(rc!=FG_OK){fg_gguf_close(&g);return rc;}
    }
    fg_manifest *m=malloc(sizeof(*m));
    if(!m){
        fg_gguf_close(&g);fg_error_set(err,FG_ERR_OOM,"allocate manifest");
        return FG_ERR_OOM;
    }
    pack_output rank[FG_RANK_COUNT]={0},ngram={0},embedding={0};
    pack_output shards[FG_NGRAM_SHARD_COUNT]={0};
    bool tokenizer_attempted=false;
    fg_manifest_init(m);
    if(o->runtime_profile!=FG_RUNTIME_PROFILE_NONE){
        rc=fg_runtime_profile_apply(m,o->runtime_profile,err);
        if(rc!=FG_OK)goto done;
    }
    if(o->router_profile_path){
        double (*profile)[FG_EXPERT_COUNT]=calloc(FG_LAYER_COUNT,sizeof(*profile));
        if(!profile){rc=FG_ERR_OOM;fg_error_set(err,rc,"allocate router profile");goto done;}
        rc=load_profile(o->router_profile_path,profile,err);
        if(rc==FG_OK)rc=fg_topology_assign_profile(
            m,(const double (*)[FG_EXPERT_COUNT])profile,err);
        free(profile);if(rc!=FG_OK)goto done;
    }
    if(o->expert_map_path){
        rc=fg_topology_assign_map_file(m,o->expert_map_path,err);
        if(rc!=FG_OK)goto done;
    }
    rc=hash_pack_sources(o,m->source_sha256,err);
    if(rc!=FG_OK)goto done;
    rc=open_outputs(o,rank,&ngram,&embedding,shards,err);
    if(rc!=FG_OK)goto done;
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){
        snprintf(m->ranks[r].endpoint,sizeof(m->ranks[r].endpoint),
                 "192.168.42.%u:19100",42u+r);
        m->ranks[r].driver_reserve_bytes=FG_PACK_DRIVER_RESERVE_BYTES;
        m->ranks[r].scratch_bytes=fg_q38_runtime_scratch_bytes_for_manifest(
            m,r,m->prefill_microbatch,m->prefill_window,m->max_context);
    }
    for(uint64_t i=0;i<g.tensor_count;i++){
        const fg_gguf_tensor *tensor=&g.tensors[i];
        FILE *source=fopen(g.paths[tensor->shard],"rb");
        if(!source){
            fg_error_set(err,FG_ERR_IO,"reopen %s: %s",
                         g.paths[tensor->shard],strerror(errno));
            rc=FG_ERR_IO;break;
        }
        if(fg_gguf_tensor_kind(tensor->name)==FG_TENSOR_ROUTED_EXPERT)
            rc=pack_expert_tensor(&g,tensor,source,m,rank,err);
        else rc=pack_common(tensor,source,m,rank,&ngram,&embedding,shards,err);
        fclose(source);
        if(rc!=FG_OK)break;
    }
    {
        fg_error close_error={0};
        fg_status close_rc=close_outputs(
            rank,&ngram,&embedding,shards,rc==FG_OK?err:&close_error);
        if(rc==FG_OK)rc=close_rc;
    }
    if(rc!=FG_OK)goto done;
    for(uint32_t r=0;r<FG_RANK_COUNT;r++)
        m->ranks[r].persistent_bytes=rank[r].offset;
    if(!o->skip_model_validation&&!o->dry_run){
        char tokenizer_path[1200];struct stat existing;
        if(snprintf(tokenizer_path,sizeof(tokenizer_path),
                    "%s/tokenizer/tokenizer.fgt",o->output_dir)>=
           (int)sizeof(tokenizer_path)){
            fg_error_set(err,FG_ERR_LIMIT,"tokenizer output path is too long");
            rc=FG_ERR_LIMIT;goto done;
        }
        if(m->execution_mode==FG_EXECUTION_PIPELINE&&
           (stat(tokenizer_path,&existing)==0||errno!=ENOENT)){
            fg_error_set(err,FG_ERR_IO,"pack output already exists: %s",
                         tokenizer_path);
            rc=FG_ERR_IO;goto done;
        }
        tokenizer_attempted=true;
        rc=fg_tokenizer_pack_gguf(o->source_paths[0],o->output_dir,m,err);
        if(rc!=FG_OK)goto done;
    }
    for(uint32_t r=0;r<FG_RANK_COUNT;r++)
        m->ranks[r].transient_bytes=FG_PACK_RANK_TRANSIENT_BYTES;
    fg_q38_account_session_state(m);
    rc=fg_manifest_validate_tensor_storage(m,err);
    if(rc!=FG_OK)goto done;
    if(!o->skip_model_validation){
        rc=fg_q38_validate_packed_manifest(m,err);
        if(rc==FG_OK&&m->execution_mode==FG_EXECUTION_PIPELINE)
            rc=fg_q38_validate_ngram_shards(m,err);
        if(rc!=FG_OK)goto done;
    }
    rc=validate_pack_memory(m,o->dry_run,err);
    if(rc!=FG_OK)goto done;
    uint8_t zero[32]={0};memcpy(m->manifest_sha256,zero,32);
    if(!o->dry_run){
        rc=commit_outputs(rank,&ngram,&embedding,shards,err);
        if(rc!=FG_OK)goto done;
        char path[1024];
        if(snprintf(path,sizeof(path),"%s/manifest.fgm",o->output_dir)>=
           (int)sizeof(path)){
            fg_error_set(err,FG_ERR_LIMIT,"manifest output path is too long");
            rc=FG_ERR_LIMIT;goto done;
        }
        struct stat existing;
        if(m->execution_mode==FG_EXECUTION_PIPELINE&&
           (stat(path,&existing)==0||errno!=ENOENT)){
            fg_error_set(err,FG_ERR_IO,"pack output already exists: %s",path);
            rc=FG_ERR_IO;goto done;
        }
        rc=fg_manifest_write(path,m,err);
        if(rc==FG_OK)fg_manifest_print(m);
    }else fg_manifest_print(m);
done:
    if(rc!=FG_OK){
        discard_outputs(rank,&ngram,&embedding,shards);
        if(tokenizer_attempted&&m->execution_mode==FG_EXECUTION_PIPELINE){
            char path[1200],directory[1024];
            if(snprintf(path,sizeof(path),"%s/tokenizer/tokenizer.fgt",o->output_dir)<
               (int)sizeof(path))unlink(path);
            if(snprintf(directory,sizeof(directory),"%s/tokenizer",o->output_dir)<
               (int)sizeof(directory))rmdir(directory);
        }
    }
    free(m);fg_gguf_close(&g);return rc;
}

/* ---------- pack verification ---------- */

static fg_status hash_gguf_range(FILE *src,uint64_t offset,uint64_t bytes,uint8_t digest[32],fg_error *err){
    fg_sha256 ctx;fg_sha256_init(&ctx);uint8_t *buf=malloc(1u<<20);if(!buf){fg_error_set(err,FG_ERR_OOM,"allocate verify buffer");return FG_ERR_OOM;}
    if(fseeko(src,(off_t)offset,SEEK_SET)!=0){free(buf);fg_error_set(err,FG_ERR_IO,"seek source for verify: %s",strerror(errno));return FG_ERR_IO;}
    uint64_t remaining=bytes;while(remaining){size_t n=remaining>(1u<<20)?(1u<<20):(size_t)remaining;if(fread(buf,1,n,src)!=n){free(buf);fg_error_set(err,FG_ERR_IO,"read source for verify: %s",strerror(errno));return FG_ERR_IO;}fg_sha256_update(&ctx,buf,n);remaining-=n;}
    free(buf);fg_sha256_final(&ctx,digest);return FG_OK;
}

static fg_status hash_artifact_payload(const char *path,uint64_t bytes,
                                       bool exact_size,uint8_t digest[32],
                                       fg_error *err){
    FILE *file=fopen(path,"rb");
    if(!file){
        fg_error_set(err,FG_ERR_IO,"open artifact %s: %s",path,strerror(errno));
        return FG_ERR_IO;
    }
    struct stat info;
    uint64_t expected=exact_size?bytes:fg_align_up_u64(bytes,FG_ALIGNMENT);
    if(fstat(fileno(file),&info)!=0||!S_ISREG(info.st_mode)||info.st_size<0||
       (uint64_t)info.st_size!=expected){
        fclose(file);fg_error_set(err,FG_ERR_MISMATCH,
            "artifact %s size is not exactly %llu bytes",path,
            (unsigned long long)expected);
        return FG_ERR_MISMATCH;
    }
    fg_status status=hash_gguf_range(file,0u,bytes,digest,err);
    if(fclose(file)!=0&&status==FG_OK){
        fg_error_set(err,FG_ERR_IO,"close artifact %s: %s",path,strerror(errno));
        status=FG_ERR_IO;
    }
    return status;
}

static const fg_gguf_tensor *find_gguf_tensor(const fg_gguf *g,const char *name){
    for(uint64_t i=0;i<g->tensor_count;i++)if(strcmp(g->tensors[i].name,name)==0)return &g->tensors[i];
    return NULL;
}

fg_status fg_pack_verify(const fg_verify_options *o,fg_error *err){
    if(!o||!o->manifest_path||!o->pack_dir||!o->source_paths||!o->source_count){fg_error_set(err,FG_ERR_ARGUMENT,"verify requires --manifest, --pack-dir, and --source");return FG_ERR_ARGUMENT;}

    fg_manifest *m=malloc(sizeof(*m));if(!m){fg_error_set(err,FG_ERR_OOM,"allocate manifest");return FG_ERR_OOM;}
    fg_status rc=fg_manifest_read(o->manifest_path,m,err);if(rc!=FG_OK){free(m);return rc;}
    rc=fg_manifest_validate_tensor_storage(m,err);if(rc!=FG_OK){free(m);return rc;}
    if(m->execution_mode==FG_EXECUTION_PIPELINE&&
       ((m->flags&FG_MANIFEST_HAS_NGRAM)||m->ngram_shard_count)){
        rc=fg_q38_validate_ngram_shards(m,err);
        if(rc!=FG_OK){free(m);return rc;}
    }
    fg_gguf g;rc=fg_gguf_open(o->source_paths,o->source_count,&g,err);if(rc!=FG_OK){free(m);return rc;}

    printf("Verifying %u manifest tensors against %llu GGUF tensors\n",m->tensor_count,(unsigned long long)g.tensor_count);

    /* Phase 1: dump tensor inventory */
    uint32_t common_count=0,expert_count=0,ngram_count=0,token_count=0,host_count=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        switch(t->kind){
            case FG_TENSOR_COMMON:common_count++;break;
            case FG_TENSOR_ROUTED_EXPERT:expert_count++;break;
            case FG_TENSOR_NGRAM:ngram_count++;break;
            case FG_TENSOR_TOKENIZER:token_count++;break;
            case FG_TENSOR_HOST_CACHE:host_count++;break;
            default:break;
        }
    }
    printf("  common=%u expert=%u ngram=%u tokenizer=%u host-cache=%u\n",common_count,expert_count,ngram_count,token_count,host_count);

    /* Phase 2: verify common tensors */
    uint32_t pass=0,fail=0,skip=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        if(t->kind!=FG_TENSOR_COMMON)continue;
        const fg_gguf_tensor *gt=find_gguf_tensor(&g,t->name);
        if(!gt){printf("  SKIP common %.80s (not in GGUF)\n",t->name);skip++;continue;}
        uint64_t expected_bytes=gt->bytes;if(t->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED){expected_bytes=gt->shape[0]<=UINT32_MAX&&gt->shape[1]<=UINT32_MAX?fg_q8_0_cooked_matrix_bytes((uint32_t)gt->shape[0],(uint32_t)gt->shape[1]):0u;if(!expected_bytes){printf("  FAIL common %.80s cooked dimensions overflow\n",t->name);fail++;continue;}}
        if(expected_bytes!=t->bytes){printf("  FAIL common %.80s size mismatch: manifest=%llu expected=%llu\n",t->name,(unsigned long long)t->bytes,(unsigned long long)expected_bytes);fail++;continue;}
        FILE *src=fopen(g.paths[gt->shard],"rb");if(!src){printf("  FAIL common %.80s cannot open shard %u\n",t->name,gt->shard);fail++;continue;}
        uint8_t digest[32];if(t->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED){fg_sha256 hash;fg_sha256_init(&hash);rc=process_cooked_q8_0(src,gt,NULL,&hash,err);if(rc==FG_OK)fg_sha256_final(&hash,digest);}else rc=hash_gguf_range(src,gt->offset,gt->bytes,digest,err);fclose(src);
        if(rc!=FG_OK){printf("  FAIL common %.80s hash error: %s\n",t->name,err->message);fail++;continue;}
        if(memcmp(digest,t->sha256,32)==0){pass++;}else{
            char got[65],exp[65];fg_sha256_hex(digest,got);fg_sha256_hex(t->sha256,exp);
            printf("  FAIL common %.80s SHA-256 mismatch got=%.16s... expected=%.16s...\n",t->name,got,exp);fail++;
        }
    }
    printf("Common tensors: %u pass, %u fail, %u skip\n",pass,fail,skip);

    uint32_t hpass=0,hfail=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        if(t->kind!=FG_TENSOR_HOST_CACHE)continue;
        const fg_gguf_tensor *gt=find_gguf_tensor(&g,t->name);
        char path[1024];struct stat info;uint8_t digest[32];
        if(!gt||t->layout!=FG_TENSOR_LAYOUT_HOST_Q8_0||
           t->bytes!=gt->bytes||
           snprintf(path,sizeof(path),"%s/%s",o->pack_dir,
                    FG_TOKEN_EMBEDDING_ARTIFACT)>=(int)sizeof(path)||
           stat(path,&info)!=0||info.st_size<0||
           (uint64_t)info.st_size!=t->bytes){
            printf("  FAIL host cache %.80s metadata or artifact mismatch\n",t->name);
            hfail++;continue;
        }
        rc=fg_sha256_file(path,digest,err);
        if(rc==FG_OK&&!memcmp(digest,t->sha256,sizeof(digest)))hpass++;
        else{printf("  FAIL host cache %.80s SHA-256 mismatch\n",t->name);hfail++;}
    }
    printf("Host-cache tensors: %u pass, %u fail\n",hpass,hfail);

    uint32_t npass=0,nfail=0;
    const fg_tensor_record *full_ngram=NULL;
    for(uint32_t i=0;i<m->tensor_count;i++)if(m->tensors[i].kind==FG_TENSOR_NGRAM){
        if(full_ngram){nfail++;break;}
        full_ngram=&m->tensors[i];
    }
    if(!full_ngram){
        if(m->flags&FG_MANIFEST_HAS_NGRAM)nfail++;
    }else{
        char path[1200];uint8_t digest[32];
        if(snprintf(path,sizeof(path),"%s/ngram.iq4nl",o->pack_dir)>=(int)sizeof(path)||
           hash_artifact_payload(path,full_ngram->bytes,false,digest,err)!=FG_OK||
           memcmp(digest,full_ngram->sha256,sizeof(digest))){
            printf("  FAIL full n-gram artifact metadata, size, or SHA-256\n");nfail++;
        }else npass++;
    }
    if(m->execution_mode==FG_EXECUTION_PIPELINE&&full_ngram){
        for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){
            const fg_ngram_shard_record *record=fg_q38_find_ngram_shard(m,rank);
            char path[1200];uint8_t digest[32];
            if(!record||snprintf(path,sizeof(path),
               "%s/" FG_NGRAM_SHARD_ARTIFACT_FORMAT,o->pack_dir,rank)>=(int)sizeof(path)||
               hash_artifact_payload(path,record?record->bytes:0u,true,digest,err)!=FG_OK||
               memcmp(digest,record->sha256,sizeof(digest))){
                printf("  FAIL resident n-gram shard rank %u\n",rank);nfail++;
            }else npass++;
        }
    }
    printf("N-gram artifacts: %u pass, %u fail\n",npass,nfail);

    /* Phase 3: verify expert tensors */
    uint32_t epass=0,efail=0,eskip=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        if(t->kind!=FG_TENSOR_ROUTED_EXPERT)continue;

        char gguf_name[FG_TENSOR_NAME_MAX];
        uint32_t rank=t->rank;
        uint32_t local_experts=FG_EXPERTS_PER_RANK;
        if(m->execution_mode==FG_EXECUTION_PIPELINE){
            snprintf(gguf_name,sizeof(gguf_name),"%s",t->name);
            local_experts=FG_EXPERT_COUNT;
        }else{
            const char *rank_suffix=strstr(t->name,".rank");
            if(!rank_suffix){printf("  FAIL expert %.80s: cannot parse rank suffix\n",t->name);efail++;continue;}
            size_t base_len=(size_t)(rank_suffix-t->name);
            if(base_len>=sizeof(gguf_name)){efail++;continue;}
            memcpy(gguf_name,t->name,base_len);gguf_name[base_len]=0;
            rank=(uint32_t)strtoul(rank_suffix+5,NULL,10);
        }
        int layer=fg_gguf_tensor_layer(gguf_name);
        if(layer<0||rank>=FG_RANK_COUNT){printf("  FAIL expert %.80s: bad layer/rank\n",t->name);efail++;continue;}
        if(m->execution_mode==FG_EXECUTION_PIPELINE&&rank!=m->layer_owner[layer]){
            printf("  FAIL expert %.80s: not on layer owner\n",t->name);efail++;continue;
        }

        const fg_gguf_tensor *gt=find_gguf_tensor(&g,gguf_name);
        if(!gt){printf("  FAIL expert %.80s: GGUF tensor %s not found\n",t->name,gguf_name);efail++;continue;}
        if(gt->bytes%FG_EXPERT_COUNT){printf("  FAIL expert %.80s: bytes not divisible by expert count\n",t->name);efail++;continue;}
        uint64_t expert_bytes=gt->bytes/FG_EXPERT_COUNT;
        if(t->bytes!=expert_bytes*local_experts){printf("  FAIL expert %.80s: size mismatch manifest=%llu expected=%llu\n",t->name,(unsigned long long)t->bytes,(unsigned long long)(expert_bytes*local_experts));efail++;continue;}

        /* Replay the pack hashing: iterate global experts in ascending order, hash those belonging to this rank */
        FILE *src=fopen(g.paths[gt->shard],"rb");if(!src){printf("  FAIL expert %.80s: cannot open shard\n",t->name);efail++;continue;}
        fg_sha256 ctx;fg_sha256_init(&ctx);uint8_t *buf=malloc((size_t)expert_bytes),*cooked=t->layout==FG_TENSOR_LAYOUT_GGML?NULL:malloc((size_t)expert_bytes);
        if(!buf||(t->layout!=FG_TENSOR_LAYOUT_GGML&&!cooked)){free(cooked);free(buf);fclose(src);printf("  FAIL expert %.80s: OOM\n",t->name);efail++;continue;}
        uint32_t copied=0;bool ok=true;
        for(uint32_t e=0;e<FG_EXPERT_COUNT&&ok;e++){
            if(m->expert_rank[layer][e]!=rank)continue;
            if(fseeko(src,(off_t)(gt->offset+e*expert_bytes),SEEK_SET)!=0){ok=false;break;}
            if(fread(buf,1,(size_t)expert_bytes,src)!=(size_t)expert_bytes){ok=false;break;}
            const uint8_t *payload=buf;if(t->layout!=FG_TENSOR_LAYOUT_GGML){if(!cook_expert_data(gt,(fg_tensor_layout)t->layout,buf,cooked,expert_bytes)){ok=false;break;}payload=cooked;}
            fg_sha256_update(&ctx,payload,(size_t)expert_bytes);
            copied++;
        }
        free(cooked);free(buf);fclose(src);
        if(!ok||copied!=local_experts){printf("  FAIL expert %.80s: read error or count mismatch (copied=%u)\n",t->name,copied);efail++;continue;}
        uint8_t digest[32];fg_sha256_final(&ctx,digest);
        if(memcmp(digest,t->sha256,32)==0){epass++;}else{
            char got[65],exp[65];fg_sha256_hex(digest,got);fg_sha256_hex(t->sha256,exp);
            printf("  FAIL expert %.80s SHA-256 mismatch got=%.16s... expected=%.16s...\n",t->name,got,exp);
            /* Print first few local-to-global expert mappings for debugging */
            uint32_t local=0;
            for(uint32_t e=0;e<FG_EXPERT_COUNT;e++){
                if(m->expert_rank[layer][e]!=rank)continue;
                printf("    local[%u] = global expert %u\n",local,e);
                local++;if(local>=4)break;
            }
            efail++;
        }
    }
    printf("Expert tensors: %u pass, %u fail, %u skip\n",epass,efail,eskip);

    /* Phase 4: verify .fgw data matches GGUF via direct byte comparison (spot check) */
    printf("\nPhase 4: Direct byte comparison (first expert of layer 0 on each rank)\n");
    uint32_t rank_count=m->execution_mode==FG_EXECUTION_PIPELINE?1u:FG_GROUP_SIZE;
    for(uint32_t gi=0;gi<rank_count;gi++){
        uint32_t rank=m->execution_mode==FG_EXECUTION_PIPELINE?
            m->layer_owner[0]:m->layer_groups[0][gi];
        char tname[FG_TENSOR_NAME_MAX];
        if(m->execution_mode==FG_EXECUTION_PIPELINE)
            snprintf(tname,sizeof(tname),"blk.0.ffn_gate_exps.weight");
        else snprintf(tname,sizeof(tname),"blk.0.ffn_gate_exps.weight.rank%u",rank);
        const fg_tensor_record *t=NULL;
        for(uint32_t i=0;i<m->tensor_count;i++)if(strcmp(m->tensors[i].name,tname)==0){t=&m->tensors[i];break;}
        if(!t){printf("  SKIP rank %u: tensor %s not in manifest\n",rank,tname);continue;}

        const fg_gguf_tensor *gt=find_gguf_tensor(&g,"blk.0.ffn_gate_exps.weight");
        if(!gt)continue;
        uint64_t expert_bytes=gt->bytes/FG_EXPERT_COUNT;

        /* Find first global expert assigned to this rank */
        uint32_t first_global=UINT32_MAX;
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)if(m->expert_rank[0][e]==rank){first_global=e;break;}
        if(first_global==UINT32_MAX)continue;

        /* Read and transform the first expert exactly as the packer did. */
        FILE *src=fopen(g.paths[gt->shard],"rb");if(!src)continue;
        uint8_t gguf_head[64]={0},*packed=malloc((size_t)expert_bytes),*cooked=t->layout==FG_TENSOR_LAYOUT_GGML?NULL:malloc((size_t)expert_bytes);bool expected_ok=packed&&fseeko(src,(off_t)(gt->offset+first_global*expert_bytes),SEEK_SET)==0&&fread(packed,1,(size_t)expert_bytes,src)==(size_t)expert_bytes;if(expected_ok&&t->layout!=FG_TENSOR_LAYOUT_GGML)expected_ok=cooked&&cook_expert_data(gt,(fg_tensor_layout)t->layout,packed,cooked,expert_bytes);if(expected_ok)memcpy(gguf_head,t->layout==FG_TENSOR_LAYOUT_GGML?packed:cooked,sizeof(gguf_head));free(cooked);free(packed);fclose(src);if(!expected_ok){printf("  SKIP rank %u: cannot prepare expected expert bytes\n",rank);continue;}

        /* Read first 64 bytes from .fgw */
        char fgw_path[1024];snprintf(fgw_path,sizeof(fgw_path),"%s/rank-%02u.fgw",o->pack_dir,rank);
        FILE *fgw=fopen(fgw_path,"rb");if(!fgw){printf("  SKIP rank %u: cannot open %s\n",rank,fgw_path);continue;}
        uint8_t fgw_head[64]={0};
        if(fseeko(fgw,(off_t)t->offset,SEEK_SET)!=0||fread(fgw_head,1,64,fgw)!=64u){fclose(fgw);printf("  SKIP rank %u: cannot read %s\n",rank,fgw_path);continue;}
        fclose(fgw);

        bool match=memcmp(gguf_head,fgw_head,64)==0;
        printf("  rank %u (global expert %u): %s  [gguf: %02x%02x%02x%02x fgw: %02x%02x%02x%02x]\n",
            rank,first_global,match?"PASS":"FAIL",
            gguf_head[0],gguf_head[1],gguf_head[2],gguf_head[3],
            fgw_head[0],fgw_head[1],fgw_head[2],fgw_head[3]);
    }

    uint32_t total_pass=pass+epass+hpass+npass,total_fail=fail+efail+hfail+nfail;
    printf("\n=== VERIFICATION %s: %u pass, %u fail ===\n",total_fail?"FAILED":"PASSED",total_pass,total_fail);
    fg_gguf_close(&g);free(m);
    return total_fail?FG_ERR_MISMATCH:FG_OK;
}
