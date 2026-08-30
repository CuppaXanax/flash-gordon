#include "fg_manifest.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"
#include "fg_topology.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void manifest_digest(const fg_manifest *m,uint8_t out[32]){fg_manifest *copy=malloc(sizeof(*copy));if(!copy){memset(out,0,32);return;}memcpy(copy,m,sizeof(*copy));memset(copy->manifest_sha256,0,32);fg_sha256 c;fg_sha256_init(&c);fg_sha256_update(&c,copy,sizeof(*copy));fg_sha256_final(&c,out);free(copy);}

void fg_manifest_init(fg_manifest *m){
    memset(m,0,sizeof(*m));m->magic=FG_MANIFEST_MAGIC;m->format_version=FG_MANIFEST_FORMAT_VERSION;m->protocol_version=FG_PROTOCOL_VERSION;m->header_bytes=(uint32_t)sizeof(*m);
    m->rank_count=FG_RANK_COUNT;m->layer_count=FG_LAYER_COUNT;m->expert_count=FG_EXPERT_COUNT;m->hidden_size=FG_HIDDEN_SIZE;m->top_k=FG_TOP_K;m->required_cu=FG_REQUIRED_CU;
    m->native_context=FG_NATIVE_CONTEXT;m->max_context=FG_MAX_CONTEXT;m->prefill_microbatch=FG_DEFAULT_MICROBATCH;m->prefill_window=FG_DEFAULT_WINDOW;
    m->persistent_cap_bytes=FG_PERSISTENT_CAP_BYTES;m->residency_cap_bytes=FG_RESIDENCY_CAP_BYTES;fg_topology_build(m);(void)fg_topology_assign_round_robin(m,NULL);for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)m->ranks[rank].scratch_bytes=fg_q38_runtime_scratch_bytes(rank,m->prefill_microbatch,m->prefill_window,m->max_context);
}

fg_status fg_manifest_validate(const fg_manifest *m,fg_error *err){
    if(!m||m->magic!=FG_MANIFEST_MAGIC||m->format_version!=FG_MANIFEST_FORMAT_VERSION||m->header_bytes!=sizeof(*m)){fg_error_set(err,FG_ERR_FORMAT,"unsupported manifest header");return FG_ERR_FORMAT;}
    if(m->protocol_version!=FG_PROTOCOL_VERSION||m->rank_count!=FG_RANK_COUNT||m->layer_count!=FG_LAYER_COUNT||m->expert_count!=FG_EXPERT_COUNT||m->hidden_size!=FG_HIDDEN_SIZE||m->top_k!=FG_TOP_K){fg_error_set(err,FG_ERR_MISMATCH,"manifest model or protocol constants do not match this binary");return FG_ERR_MISMATCH;}
    if(m->required_cu!=24&&m->required_cu!=40){fg_error_set(err,FG_ERR_FORMAT,"required CU count must be 24 or 40");return FG_ERR_FORMAT;}
    if(m->prefill_microbatch!=128&&m->prefill_microbatch!=256&&m->prefill_microbatch!=512){fg_error_set(err,FG_ERR_FORMAT,"prefill microbatch is not qualified");return FG_ERR_FORMAT;}
    if(m->prefill_window<1||m->prefill_window>4||m->tensor_count>FG_MAX_TENSORS){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill window or tensor count");return FG_ERR_FORMAT;}
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        if(t->dims==0||t->dims>4){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has invalid dimension count",i);return FG_ERR_FORMAT;}
        for(uint32_t d=0;d<t->dims;d++)if(t->shape[d]==0){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an empty dimension",i);return FG_ERR_FORMAT;}
        if(t->layout>FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has unknown storage layout %u",i,t->layout);return FG_ERR_FORMAT;}
        if(t->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED){if(t->ggml_type!=8u||t->dims!=2u||t->shape[0]>UINT32_MAX||t->shape[1]>UINT32_MAX||t->shape[0]%FG_QK8_0){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an invalid cooked Q8_0 layout",i);return FG_ERR_FORMAT;}uint64_t blocks=t->shape[0]/FG_QK8_0,quant_offset=fg_align_up_u64(FG_Q8_0_COOK_ROWS*blocks*sizeof(uint16_t),FG_Q8_0_COOK_ALIGNMENT),tile_bytes=fg_align_up_u64(quant_offset+FG_Q8_0_COOK_ROWS*blocks*FG_QK8_0,FG_Q8_0_COOK_ALIGNMENT),tiles=(t->shape[1]+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS;if(tile_bytes>UINT32_MAX||tiles>UINT64_MAX/tile_bytes||t->bytes!=tile_bytes*tiles||t->bytes>UINT32_MAX){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an invalid cooked Q8_0 layout",i);return FG_ERR_FORMAT;}}
        if(t->layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED){uint64_t matrix=t->shape[0]<=UINT32_MAX&&t->shape[1]<=UINT32_MAX?fg_k_quant_cooked_matrix_bytes((uint32_t)t->shape[0],(uint32_t)t->shape[1],t->ggml_type):0u;if(t->kind!=FG_TENSOR_ROUTED_EXPERT||t->dims!=3u||t->shape[2]!=FG_EXPERTS_PER_RANK||!matrix||matrix>UINT64_MAX/t->shape[2]||t->bytes!=matrix*t->shape[2]){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an invalid cooked K-quant expert layout",i);return FG_ERR_FORMAT;}}
        if(t->layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED){uint64_t matrix=t->shape[0]<=UINT32_MAX&&t->shape[1]<=UINT32_MAX?fg_q5_1_cooked_matrix_bytes((uint32_t)t->shape[0],(uint32_t)t->shape[1]):0u;if(t->kind!=FG_TENSOR_ROUTED_EXPERT||t->ggml_type!=7u||t->dims!=3u||t->shape[2]!=FG_EXPERTS_PER_RANK||!matrix||matrix>UINT64_MAX/t->shape[2]||t->bytes!=matrix*t->shape[2]){fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an invalid cooked Q5_1 expert layout",i);return FG_ERR_FORMAT;}}
    }
    for(uint32_t r=0;r<FG_RANK_COUNT;r++){
        const fg_rank_record *rank=&m->ranks[r];uint64_t resident=rank->persistent_bytes+rank->transient_bytes+rank->kv_bytes+rank->scratch_bytes+rank->driver_reserve_bytes;
        uint64_t required_scratch=fg_q38_runtime_scratch_bytes(r,m->prefill_microbatch,m->prefill_window,m->max_context);if(required_scratch==UINT64_MAX||rank->scratch_bytes<required_scratch){fg_error_set(err,FG_ERR_LIMIT,"rank %u scratch ledger is %llu bytes, requires at least %llu for prefill %ux%u",r,(unsigned long long)rank->scratch_bytes,(unsigned long long)required_scratch,m->prefill_microbatch,m->prefill_window);return FG_ERR_LIMIT;}
        if(rank->persistent_bytes>m->persistent_cap_bytes||resident>m->residency_cap_bytes){fg_error_set(err,FG_ERR_LIMIT,"rank %u memory ledger exceeds cap: persistent %.3f GiB, residency %.3f GiB",r,(double)rank->persistent_bytes/(1ull<<30),(double)resident/(1ull<<30));return FG_ERR_LIMIT;}
    }
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        if(m->layer_owner[l]!=l%FG_RANK_COUNT){fg_error_set(err,FG_ERR_FORMAT,"layer %u owner mismatch",l);return FG_ERR_FORMAT;}
        uint16_t counts[FG_RANK_COUNT]={0};
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++){uint16_t r=m->expert_rank[l][e];if(r>=FG_RANK_COUNT||!fg_topology_rank_in_layer(m,l,r)){fg_error_set(err,FG_ERR_FORMAT,"layer %u expert %u assigned outside group",l,e);return FG_ERR_FORMAT;}counts[r]++;}
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++){uint32_t r=m->layer_groups[l][g];if(counts[r]!=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_FORMAT,"layer %u rank %u owns %u experts, expected %u",l,r,counts[r],FG_EXPERTS_PER_RANK);return FG_ERR_FORMAT;}}
    }
    uint8_t digest[32];manifest_digest(m,digest);if(memcmp(digest,m->manifest_sha256,32)!=0){fg_error_set(err,FG_ERR_MISMATCH,"manifest SHA-256 mismatch");return FG_ERR_MISMATCH;}return FG_OK;
}
fg_status fg_manifest_validate_deployment(const fg_manifest *m,fg_error *err){fg_status rc=fg_manifest_validate(m,err);if(rc!=FG_OK)return rc;uint32_t missing=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED&~m->flags;if(missing){fg_error_set(err,FG_ERR_MISMATCH,"text deployment pack is incomplete (missing component flags 0x%02x)",missing);return FG_ERR_MISMATCH;}return FG_OK;}

fg_status fg_manifest_write(const char *path,fg_manifest *m,fg_error *err){
    uint8_t digest[32];manifest_digest(m,digest);memcpy(m->manifest_sha256,digest,32);
    char tmp[1024];if(snprintf(tmp,sizeof(tmp),"%s.tmp.%ld",path,(long)getpid())>=(int)sizeof(tmp)){fg_error_set(err,FG_ERR_ARGUMENT,"manifest path too long");return FG_ERR_ARGUMENT;}
    FILE *f=fopen(tmp,"wb");if(!f){fg_error_set(err,FG_ERR_IO,"create %s: %s",tmp,strerror(errno));return FG_ERR_IO;}
    bool ok=fwrite(m,1,sizeof(*m),f)==sizeof(*m)&&fflush(f)==0&&fsync(fileno(f))==0&&fclose(f)==0;
    if(!ok){fg_error_set(err,FG_ERR_IO,"write %s: %s",tmp,strerror(errno));fclose(f);unlink(tmp);return FG_ERR_IO;}
    if(rename(tmp,path)!=0){fg_error_set(err,FG_ERR_IO,"rename %s to %s: %s",tmp,path,strerror(errno));unlink(tmp);return FG_ERR_IO;}return FG_OK;
}
fg_status fg_manifest_read(const char *path,fg_manifest *m,fg_error *err){FILE *f=fopen(path,"rb");if(!f){fg_error_set(err,FG_ERR_IO,"open %s: %s",path,strerror(errno));return FG_ERR_IO;}size_t n=fread(m,1,sizeof(*m),f);int extra=fgetc(f);fclose(f);if(n!=sizeof(*m)||extra!=EOF){fg_error_set(err,FG_ERR_FORMAT,"manifest %s has wrong size",path);return FG_ERR_FORMAT;}return fg_manifest_validate(m,err);}
fg_status fg_manifest_add_tensor(fg_manifest *m,const fg_tensor_record *r,fg_error *err){if(m->tensor_count>=FG_MAX_TENSORS){fg_error_set(err,FG_ERR_LIMIT,"manifest tensor limit exceeded");return FG_ERR_LIMIT;}m->tensors[m->tensor_count++]=*r;return FG_OK;}
void fg_manifest_print(const fg_manifest *m){printf("Flash Gordon manifest v%u protocol=%u CU=%u tensors=%u prefill=%ux%u context=%u/%u\n",m->format_version,m->protocol_version,m->required_cu,m->tensor_count,m->prefill_microbatch,m->prefill_window,m->native_context,m->max_context);uint32_t cooked_q8=0,cooked_k=0,cooked_q5=0;uint64_t cooked_bytes=0;for(uint32_t i=0;i<m->tensor_count;i++){uint32_t layout=m->tensors[i].layout;if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED)cooked_q8++;else if(layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED)cooked_k++;else if(layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED)cooked_q5++;if(layout!=FG_TENSOR_LAYOUT_GGML)cooked_bytes+=m->tensors[i].bytes;}printf("layouts ggml=%u cooked-q8=%u cooked-k=%u cooked-q5_1=%u cooked-bytes=%.3f GiB\n",m->tensor_count-cooked_q8-cooked_k-cooked_q5,cooked_q8,cooked_k,cooked_q5,(double)cooked_bytes/(1ull<<30));for(uint32_t r=0;r<FG_RANK_COUNT;r++){const fg_rank_record *x=&m->ranks[r];printf("rank %u %-21s persistent=%6.3f GiB residency=%6.3f GiB state-file=%6.3f GiB tensors=%u\n",r,x->endpoint,(double)x->persistent_bytes/(1ull<<30),(double)(x->persistent_bytes+x->transient_bytes+x->kv_bytes+x->scratch_bytes+x->driver_reserve_bytes)/(1ull<<30),(double)x->state_file_bytes/(1ull<<30),x->tensor_count);}}
