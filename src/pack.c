#include "fg_pack.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"
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

typedef struct pack_output{FILE *file;char path[1024];uint64_t offset;}pack_output;
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
static fg_status open_outputs(const fg_pack_options *o,pack_output rank[FG_RANK_COUNT],pack_output *ngram,fg_error *err){if(o->dry_run)return FG_OK;if(mkdir_one(o->output_dir,err)!=FG_OK)return err->code;for(uint32_t r=0;r<FG_RANK_COUNT;r++){snprintf(rank[r].path,sizeof(rank[r].path),"%s/rank-%02u.fgw",o->output_dir,r);rank[r].file=fopen(rank[r].path,"wb");if(!rank[r].file){fg_error_set(err,FG_ERR_IO,"create %s: %s",rank[r].path,strerror(errno));return FG_ERR_IO;}}snprintf(ngram->path,sizeof(ngram->path),"%s/ngram.iq4nl",o->output_dir);ngram->file=fopen(ngram->path,"wb");if(!ngram->file){fg_error_set(err,FG_ERR_IO,"create %s: %s",ngram->path,strerror(errno));return FG_ERR_IO;}return FG_OK;}
static fg_status finalize_output(pack_output *out,fg_error *err){if(!out->file)return FG_OK;fg_status rc=pad_to(out,fg_align_up_u64(out->offset,FG_ALIGNMENT),err);if(rc==FG_OK&&(fflush(out->file)!=0||fsync(fileno(out->file))!=0)){fg_error_set(err,FG_ERR_IO,"flush %s: %s",out->path,strerror(errno));rc=FG_ERR_IO;}if(fclose(out->file)!=0&&rc==FG_OK){fg_error_set(err,FG_ERR_IO,"close %s: %s",out->path,strerror(errno));rc=FG_ERR_IO;}out->file=NULL;return rc;}
static fg_status close_outputs(pack_output rank[FG_RANK_COUNT],pack_output *ngram,fg_error *err){fg_status rc=FG_OK;for(uint32_t r=0;r<FG_RANK_COUNT;r++){fg_status one=finalize_output(&rank[r],err);if(rc==FG_OK)rc=one;}fg_status one=finalize_output(ngram,err);if(rc==FG_OK)rc=one;return rc;}
static fg_status pad_to(pack_output *out,uint64_t target,fg_error *err){static const uint8_t zero[FG_ALIGNMENT]={0};while(out->offset<target){size_t n=(size_t)(target-out->offset);if(n>sizeof(zero))n=sizeof(zero);if(out->file&&fwrite(zero,1,n,out->file)!=n){fg_error_set(err,FG_ERR_IO,"pad %s: %s",out->path,strerror(errno));return FG_ERR_IO;}out->offset+=n;}return FG_OK;}
static fg_status copy_range(FILE *src,uint64_t offset,uint64_t bytes,pack_output *out,fg_sha256 *hash,fg_error *err){if(!out->file){uint8_t descriptor[16];memcpy(descriptor,&offset,8);memcpy(descriptor+8,&bytes,8);fg_sha256_update(hash,descriptor,sizeof(descriptor));out->offset+=bytes;return FG_OK;}uint8_t *buf=malloc(1u<<20);if(!buf){fg_error_set(err,FG_ERR_OOM,"allocate pack copy buffer");return FG_ERR_OOM;}if(fseeko(src,(off_t)offset,SEEK_SET)!=0){free(buf);fg_error_set(err,FG_ERR_IO,"seek source: %s",strerror(errno));return FG_ERR_IO;}while(bytes){size_t n=bytes>(1u<<20)?(1u<<20):(size_t)bytes;if(fread(buf,1,n,src)!=n||fwrite(buf,1,n,out->file)!=n){free(buf);fg_error_set(err,FG_ERR_IO,"copy tensor: %s",ferror(src)?"unexpected end of source":strerror(errno));return FG_ERR_IO;}fg_sha256_update(hash,buf,n);out->offset+=n;bytes-=n;}free(buf);return FG_OK;}
static fg_status load_profile(const char *path,double profile[FG_LAYER_COUNT][FG_EXPERT_COUNT],fg_error *err){FILE *f=fopen(path,"r");if(!f){fg_error_set(err,FG_ERR_IO,"open router profile %s: %s",path,strerror(errno));return FG_ERR_IO;}unsigned l,e;double v;while(fscanf(f,"%u %u %lf",&l,&e,&v)==3){if(l>=FG_LAYER_COUNT||e>=FG_EXPERT_COUNT||v<0){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"invalid router profile row");return FG_ERR_FORMAT;}profile[l][e]=v;}if(!feof(f)){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"malformed router profile");return FG_ERR_FORMAT;}fclose(f);return FG_OK;}
static fg_status record_segment(fg_manifest *m,const fg_gguf_tensor *source,const char *name,uint64_t start,uint64_t bytes,uint32_t rank,uint32_t layer,uint32_t expert,fg_tensor_kind kind,fg_tensor_layout layout,uint64_t local_experts,fg_sha256 *hash,fg_error *err){fg_tensor_record r={0};snprintf(r.name,sizeof(r.name),"%s",name);r.offset=start;r.bytes=bytes;r.ggml_type=source->type;r.dims=source->dims;memcpy(r.shape,source->shape,sizeof(r.shape));if(local_experts)r.shape[r.dims-1]=local_experts;r.rank=(uint16_t)rank;r.layer=(uint16_t)(layer<FG_LAYER_COUNT?layer:UINT16_MAX);r.expert=(uint16_t)(expert<FG_EXPERT_COUNT?expert:UINT16_MAX);r.kind=(uint8_t)kind;r.layout=(uint8_t)layout;fg_sha256_final(hash,r.sha256);fg_status rc=fg_manifest_add_tensor(m,&r,err);if(rc==FG_OK){if(kind==FG_TENSOR_COMMON||kind==FG_TENSOR_ROUTED_EXPERT)m->flags|=FG_MANIFEST_HAS_TEXT;else if(kind==FG_TENSOR_NGRAM)m->flags|=FG_MANIFEST_HAS_NGRAM;else if(kind==FG_TENSOR_VISION)m->flags|=FG_MANIFEST_HAS_VISION;else if(kind==FG_TENSOR_MTP)m->flags|=FG_MANIFEST_HAS_MTP;else if(kind==FG_TENSOR_TOKENIZER)m->flags|=FG_MANIFEST_HAS_TOKENIZER;}if(rc==FG_OK&&rank<FG_RANK_COUNT){m->ranks[rank].tensor_count++;m->ranks[rank].persistent_bytes+=fg_align_up_u64(bytes,FG_ALIGNMENT);}return rc;}

static bool use_cooked_q8_0(const fg_gguf_tensor *tensor){if(tensor->type!=8u||tensor->dims!=2u||tensor->shape[0]>UINT32_MAX||tensor->shape[1]>UINT32_MAX||tensor->shape[0]%FG_QK8_0||tensor->shape[1]<2560u||strcmp(tensor->name,"token_embd.weight")==0)return false;uint64_t bytes=fg_q8_0_cooked_matrix_bytes((uint32_t)tensor->shape[0],(uint32_t)tensor->shape[1]);return bytes&&bytes<=UINT32_MAX;}

static fg_status process_cooked_q8_0(FILE *source,const fg_gguf_tensor *tensor,FILE *destination,fg_sha256 *hash,fg_error *err){
    uint64_t source_row=(tensor->shape[0]/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES,tile_bytes=fg_q8_0_cooked_tile_bytes((uint32_t)tensor->shape[0]);
    if(!source_row||!tile_bytes||tensor->shape[0]>UINT32_MAX||tensor->shape[1]>UINT32_MAX||tensor->shape[1]>UINT64_MAX/source_row||tensor->bytes!=source_row*tensor->shape[1]){fg_error_set(err,FG_ERR_FORMAT,"invalid Q8_0 matrix %s for cooking",tensor->name);return FG_ERR_FORMAT;}
    uint8_t *packed=malloc((size_t)source_row*FG_Q8_0_COOK_ROWS),*cooked=malloc((size_t)tile_bytes);if(!packed||!cooked){free(cooked);free(packed);fg_error_set(err,FG_ERR_OOM,"allocate cooked Q8_0 conversion buffers");return FG_ERR_OOM;}
    if(fseeko(source,(off_t)tensor->offset,SEEK_SET)!=0){free(cooked);free(packed);fg_error_set(err,FG_ERR_IO,"seek Q8_0 tensor %s: %s",tensor->name,strerror(errno));return FG_ERR_IO;}
    fg_status status=FG_OK;for(uint64_t first=0;first<tensor->shape[1];first+=FG_Q8_0_COOK_ROWS){uint32_t rows=(uint32_t)(tensor->shape[1]-first);if(rows>FG_Q8_0_COOK_ROWS)rows=FG_Q8_0_COOK_ROWS;size_t packed_bytes=(size_t)source_row*rows;if(fread(packed,1,packed_bytes,source)!=packed_bytes||!fg_cook_q8_0_rows(packed,cooked,tile_bytes,(uint32_t)tensor->shape[0],rows)||(destination&&fwrite(cooked,1,(size_t)tile_bytes,destination)!=(size_t)tile_bytes)){fg_error_set(err,FG_ERR_IO,"cook Q8_0 tensor %s: %s",tensor->name,ferror(source)?"unexpected end of source":strerror(errno));status=FG_ERR_IO;break;}fg_sha256_update(hash,cooked,(size_t)tile_bytes);}
    free(cooked);free(packed);return status;
}

static uint32_t common_owner(const fg_gguf_tensor *tensor,int layer){if(layer>=0)return (uint32_t)layer%FG_RANK_COUNT;if(strcmp(tensor->name,"token_embd.weight")==0)return 0u;if(strcmp(tensor->name,"output.weight")==0||strncmp(tensor->name,"output_hc_",10u)==0)return 4u;return 0u;}
static fg_status pack_common(const fg_gguf_tensor *t,FILE *src,fg_manifest *m,pack_output rank[FG_RANK_COUNT],pack_output *ngram,fg_error *err){fg_tensor_kind kind=fg_gguf_tensor_kind(t->name);int layer=fg_gguf_tensor_layer(t->name);uint32_t owner=common_owner(t,layer);pack_output *out=kind==FG_TENSOR_NGRAM?ngram:&rank[owner];uint64_t start=fg_align_up_u64(out->offset,FG_ALIGNMENT);fg_status rc=pad_to(out,start,err);if(rc!=FG_OK)return rc;fg_sha256 hash;fg_sha256_init(&hash);fg_tensor_layout layout=use_cooked_q8_0(t)?FG_TENSOR_LAYOUT_Q8_0_COOKED:FG_TENSOR_LAYOUT_GGML;uint64_t bytes=t->bytes;if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED){bytes=fg_q8_0_cooked_matrix_bytes((uint32_t)t->shape[0],(uint32_t)t->shape[1]);if(!bytes){fg_error_set(err,FG_ERR_LIMIT,"cooked Q8_0 tensor %s is too large",t->name);return FG_ERR_LIMIT;}if(out->file)rc=process_cooked_q8_0(src,t,out->file,&hash,err);else{uint64_t descriptor[3]={t->offset,t->bytes,bytes};fg_sha256_update(&hash,descriptor,sizeof(descriptor));}if(rc==FG_OK)out->offset+=bytes;}else rc=copy_range(src,t->offset,t->bytes,out,&hash,err);if(rc!=FG_OK)return rc;return record_segment(m,t,t->name,start,bytes,kind==FG_TENSOR_NGRAM?UINT16_MAX:owner,layer<0?UINT32_MAX:(uint32_t)layer,UINT32_MAX,kind,layout,0,&hash,err);}

static fg_status pack_expert_tensor(const fg_gguf *g,const fg_gguf_tensor *t,FILE *src,fg_manifest *m,pack_output rank[FG_RANK_COUNT],fg_error *err){
    (void)g;int layer=fg_gguf_tensor_layer(t->name);if(layer<0||t->shape[t->dims-1]!=FG_EXPERT_COUNT||t->bytes%FG_EXPERT_COUNT){fg_error_set(err,FG_ERR_FORMAT,"routed tensor %s is not a 512-expert layer tensor",t->name);return FG_ERR_FORMAT;}uint64_t expert_bytes=t->bytes/FG_EXPERT_COUNT;
    for(uint32_t gi=0;gi<FG_GROUP_SIZE;gi++){uint32_t r=m->layer_groups[layer][gi];pack_output *out=&rank[r];uint64_t start=fg_align_up_u64(out->offset,FG_ALIGNMENT);fg_status rc=pad_to(out,start,err);if(rc!=FG_OK)return rc;fg_sha256 hash;fg_sha256_init(&hash);uint32_t copied=0;
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)if(m->expert_rank[layer][e]==r){rc=copy_range(src,t->offset+(uint64_t)e*expert_bytes,expert_bytes,out,&hash,err);if(rc!=FG_OK)return rc;copied++;}
        if(copied!=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_FORMAT,"layer %d rank %u selected %u experts",layer,r,copied);return FG_ERR_FORMAT;}char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"%.80s.rank%u",t->name,r);rc=record_segment(m,t,name,start,expert_bytes*copied,r,(uint32_t)layer,UINT32_MAX,FG_TENSOR_ROUTED_EXPERT,FG_TENSOR_LAYOUT_GGML,FG_EXPERTS_PER_RANK,&hash,err);if(rc!=FG_OK)return rc;
    }return FG_OK;
}

fg_status fg_pack_run(const fg_pack_options *o,fg_error *err){
    if(!o||!o->output_dir||!o->source_paths||!o->source_count){fg_error_set(err,FG_ERR_ARGUMENT,"pack requires --output and at least one --source");return FG_ERR_ARGUMENT;}if(o->router_profile_path&&o->expert_map_path){fg_error_set(err,FG_ERR_ARGUMENT,"pack accepts only one of --router-profile and --expert-map");return FG_ERR_ARGUMENT;}fg_gguf g;fg_status rc=fg_gguf_open(o->source_paths,o->source_count,&g,err);if(rc!=FG_OK)return rc;if(!o->skip_model_validation){rc=fg_q38_validate_gguf(&g,err);if(rc!=FG_OK){fg_gguf_close(&g);return rc;}}fg_manifest *m=malloc(sizeof(*m));if(!m){fg_gguf_close(&g);fg_error_set(err,FG_ERR_OOM,"allocate manifest");return FG_ERR_OOM;}fg_manifest_init(m);
    if(o->router_profile_path){double (*profile)[FG_EXPERT_COUNT]=calloc(FG_LAYER_COUNT,sizeof(*profile));if(!profile){rc=FG_ERR_OOM;fg_error_set(err,rc,"allocate router profile");goto done;}rc=load_profile(o->router_profile_path,profile,err);if(rc==FG_OK)rc=fg_topology_assign_profile(m,(const double (*)[FG_EXPERT_COUNT])profile,err);free(profile);if(rc!=FG_OK)goto done;}
    if(o->expert_map_path){rc=fg_topology_assign_map_file(m,o->expert_map_path,err);if(rc!=FG_OK)goto done;}
    rc=hash_pack_sources(o,m->source_sha256,err);if(rc!=FG_OK)goto done;
    pack_output rank[FG_RANK_COUNT]={0},ngram={0};rc=open_outputs(o,rank,&ngram,err);if(rc!=FG_OK){(void)close_outputs(rank,&ngram,err);goto done;}for(uint32_t r=0;r<FG_RANK_COUNT;r++){snprintf(m->ranks[r].endpoint,sizeof(m->ranks[r].endpoint),"192.168.42.%u:19100",42u+r);m->ranks[r].driver_reserve_bytes=512ull<<20;m->ranks[r].scratch_bytes=fg_q38_runtime_scratch_bytes(r,m->prefill_microbatch,m->prefill_window,m->max_context);}
    for(uint64_t i=0;i<g.tensor_count;i++){const fg_gguf_tensor *t=&g.tensors[i];FILE *src=fopen(g.paths[t->shard],"rb");if(!src){fg_error_set(err,FG_ERR_IO,"reopen %s: %s",g.paths[t->shard],strerror(errno));rc=FG_ERR_IO;break;}if(fg_gguf_tensor_kind(t->name)==FG_TENSOR_ROUTED_EXPERT)rc=pack_expert_tensor(&g,t,src,m,rank,err);else rc=pack_common(t,src,m,rank,&ngram,err);fclose(src);if(rc!=FG_OK)break;}
    {fg_status close_rc=close_outputs(rank,&ngram,err);if(rc==FG_OK)rc=close_rc;}if(rc!=FG_OK)goto done;if(!o->skip_model_validation&&!o->dry_run){rc=fg_tokenizer_pack_gguf(o->source_paths[0],o->output_dir,m,err);if(rc!=FG_OK)goto done;}for(uint32_t r=0;r<FG_RANK_COUNT;r++)m->ranks[r].transient_bytes=512ull<<20;fg_q38_account_session_state(m);
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){uint64_t resident=m->ranks[r].persistent_bytes+m->ranks[r].transient_bytes+m->ranks[r].kv_bytes+m->ranks[r].scratch_bytes+m->ranks[r].driver_reserve_bytes;if(m->ranks[r].persistent_bytes>m->persistent_cap_bytes||resident>m->residency_cap_bytes){fg_error_set(err,FG_ERR_LIMIT,"memory cap failed on rank %u",r);rc=FG_ERR_LIMIT;break;}}if(rc!=FG_OK)goto done;
    if(!o->skip_model_validation){rc=fg_q38_validate_packed_manifest(m,err);if(rc!=FG_OK)goto done;}
    uint8_t zero[32]={0};memcpy(m->manifest_sha256,zero,32);if(!o->dry_run){char path[1024];snprintf(path,sizeof(path),"%s/manifest.fgm",o->output_dir);rc=fg_manifest_write(path,m,err);if(rc==FG_OK)fg_manifest_print(m);}else fg_manifest_print(m);
done:free(m);fg_gguf_close(&g);return rc;
}

/* ---------- pack verification ---------- */

static fg_status hash_gguf_range(FILE *src,uint64_t offset,uint64_t bytes,uint8_t digest[32],fg_error *err){
    fg_sha256 ctx;fg_sha256_init(&ctx);uint8_t *buf=malloc(1u<<20);if(!buf){fg_error_set(err,FG_ERR_OOM,"allocate verify buffer");return FG_ERR_OOM;}
    if(fseeko(src,(off_t)offset,SEEK_SET)!=0){free(buf);fg_error_set(err,FG_ERR_IO,"seek source for verify: %s",strerror(errno));return FG_ERR_IO;}
    uint64_t remaining=bytes;while(remaining){size_t n=remaining>(1u<<20)?(1u<<20):(size_t)remaining;if(fread(buf,1,n,src)!=n){free(buf);fg_error_set(err,FG_ERR_IO,"read source for verify: %s",strerror(errno));return FG_ERR_IO;}fg_sha256_update(&ctx,buf,n);remaining-=n;}
    free(buf);fg_sha256_final(&ctx,digest);return FG_OK;
}

static const fg_gguf_tensor *find_gguf_tensor(const fg_gguf *g,const char *name){
    for(uint64_t i=0;i<g->tensor_count;i++)if(strcmp(g->tensors[i].name,name)==0)return &g->tensors[i];
    return NULL;
}

fg_status fg_pack_verify(const fg_verify_options *o,fg_error *err){
    if(!o||!o->manifest_path||!o->pack_dir||!o->source_paths||!o->source_count){fg_error_set(err,FG_ERR_ARGUMENT,"verify requires --manifest, --pack-dir, and --source");return FG_ERR_ARGUMENT;}

    fg_manifest *m=malloc(sizeof(*m));if(!m){fg_error_set(err,FG_ERR_OOM,"allocate manifest");return FG_ERR_OOM;}
    fg_status rc=fg_manifest_read(o->manifest_path,m,err);if(rc!=FG_OK){free(m);return rc;}
    fg_gguf g;rc=fg_gguf_open(o->source_paths,o->source_count,&g,err);if(rc!=FG_OK){free(m);return rc;}

    printf("Verifying %u manifest tensors against %llu GGUF tensors\n",m->tensor_count,(unsigned long long)g.tensor_count);

    /* Phase 1: dump tensor inventory */
    uint32_t common_count=0,expert_count=0,ngram_count=0,token_count=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        switch(t->kind){
            case FG_TENSOR_COMMON:common_count++;break;
            case FG_TENSOR_ROUTED_EXPERT:expert_count++;break;
            case FG_TENSOR_NGRAM:ngram_count++;break;
            case FG_TENSOR_TOKENIZER:token_count++;break;
            default:break;
        }
    }
    printf("  common=%u expert=%u ngram=%u tokenizer=%u\n",common_count,expert_count,ngram_count,token_count);

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

    /* Phase 3: verify expert tensors */
    uint32_t epass=0,efail=0,eskip=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        if(t->kind!=FG_TENSOR_ROUTED_EXPERT)continue;

        /* Parse "blk.L.ffn_xyz.weight.rankR" */
        char gguf_name[FG_TENSOR_NAME_MAX];
        const char *rank_suffix=strstr(t->name,".rank");
        if(!rank_suffix){printf("  FAIL expert %.80s: cannot parse rank suffix\n",t->name);efail++;continue;}
        size_t base_len=(size_t)(rank_suffix-t->name);
        if(base_len>=sizeof(gguf_name)){efail++;continue;}
        memcpy(gguf_name,t->name,base_len);gguf_name[base_len]=0;
        uint32_t rank=(uint32_t)strtoul(rank_suffix+5,NULL,10);
        int layer=fg_gguf_tensor_layer(gguf_name);
        if(layer<0||rank>=FG_RANK_COUNT){printf("  FAIL expert %.80s: bad layer/rank\n",t->name);efail++;continue;}

        const fg_gguf_tensor *gt=find_gguf_tensor(&g,gguf_name);
        if(!gt){printf("  FAIL expert %.80s: GGUF tensor %s not found\n",t->name,gguf_name);efail++;continue;}
        if(gt->bytes%FG_EXPERT_COUNT){printf("  FAIL expert %.80s: bytes not divisible by expert count\n",t->name);efail++;continue;}
        uint64_t expert_bytes=gt->bytes/FG_EXPERT_COUNT;
        if(t->bytes!=expert_bytes*FG_EXPERTS_PER_RANK){printf("  FAIL expert %.80s: size mismatch manifest=%llu expected=%llu\n",t->name,(unsigned long long)t->bytes,(unsigned long long)(expert_bytes*FG_EXPERTS_PER_RANK));efail++;continue;}

        /* Replay the pack hashing: iterate global experts in ascending order, hash those belonging to this rank */
        FILE *src=fopen(g.paths[gt->shard],"rb");if(!src){printf("  FAIL expert %.80s: cannot open shard\n",t->name);efail++;continue;}
        fg_sha256 ctx;fg_sha256_init(&ctx);uint8_t *buf=malloc((size_t)expert_bytes);
        if(!buf){fclose(src);printf("  FAIL expert %.80s: OOM\n",t->name);efail++;continue;}
        uint32_t copied=0;bool ok=true;
        for(uint32_t e=0;e<FG_EXPERT_COUNT&&ok;e++){
            if(m->expert_rank[layer][e]!=rank)continue;
            if(fseeko(src,(off_t)(gt->offset+e*expert_bytes),SEEK_SET)!=0){ok=false;break;}
            if(fread(buf,1,(size_t)expert_bytes,src)!=(size_t)expert_bytes){ok=false;break;}
            fg_sha256_update(&ctx,buf,(size_t)expert_bytes);
            copied++;
        }
        free(buf);fclose(src);
        if(!ok||copied!=FG_EXPERTS_PER_RANK){printf("  FAIL expert %.80s: read error or count mismatch (copied=%u)\n",t->name,copied);efail++;continue;}
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
    for(uint32_t gi=0;gi<FG_GROUP_SIZE;gi++){
        uint32_t rank=m->layer_groups[0][gi];
        char tname[FG_TENSOR_NAME_MAX];
        snprintf(tname,sizeof(tname),"blk.0.ffn_gate_exps.weight.rank%u",rank);
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

        /* Read first 64 bytes from GGUF source */
        FILE *src=fopen(g.paths[gt->shard],"rb");if(!src)continue;
        uint8_t gguf_head[64]={0};
        if(fseeko(src,(off_t)(gt->offset+first_global*expert_bytes),SEEK_SET)!=0||fread(gguf_head,1,64,src)!=64u){fclose(src);printf("  SKIP rank %u: cannot read GGUF expert bytes\n",rank);continue;}
        fclose(src);

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

    uint32_t total_pass=pass+epass,total_fail=fail+efail;
    printf("\n=== VERIFICATION %s: %u pass, %u fail ===\n",total_fail?"FAILED":"PASSED",total_pass,total_fail);
    fg_gguf_close(&g);free(m);
    return total_fail?FG_ERR_MISMATCH:FG_OK;
}
