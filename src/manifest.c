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

static bool digest_is_zero(const uint8_t digest[32]){
    uint8_t value=0;
    for(uint32_t i=0;i<32u;i++)value|=digest[i];
    return value==0;
}

static void hash_u16(fg_sha256 *hash,uint16_t value){
    uint8_t wire[2]={(uint8_t)(value>>8u),(uint8_t)value};
    fg_sha256_update(hash,wire,sizeof(wire));
}

static void hash_u32(fg_sha256 *hash,uint32_t value){
    uint8_t wire[4]={(uint8_t)(value>>24u),(uint8_t)(value>>16u),
                     (uint8_t)(value>>8u),(uint8_t)value};
    fg_sha256_update(hash,wire,sizeof(wire));
}

static void hash_u64(fg_sha256 *hash,uint64_t value){
    hash_u32(hash,(uint32_t)(value>>32u));
    hash_u32(hash,(uint32_t)value);
}

static uint32_t component_flag(uint32_t component){
    static const uint32_t flags[FG_COMPONENT_COUNT]={
        FG_MANIFEST_HAS_TEXT,FG_MANIFEST_HAS_NGRAM,FG_MANIFEST_HAS_TOKENIZER,
        FG_MANIFEST_HAS_VISION,FG_MANIFEST_HAS_MTP
    };
    return component<FG_COMPONENT_COUNT?flags[component]:0u;
}

static bool tensor_in_component(const fg_tensor_record *tensor,uint32_t component){
    switch(component){
        case FG_COMPONENT_TEXT:
            return tensor->kind==FG_TENSOR_COMMON||tensor->kind==FG_TENSOR_ROUTED_EXPERT;
        case FG_COMPONENT_NGRAM:return tensor->kind==FG_TENSOR_NGRAM;
        case FG_COMPONENT_TOKENIZER:return tensor->kind==FG_TENSOR_TOKENIZER;
        case FG_COMPONENT_VISION:return tensor->kind==FG_TENSOR_VISION;
        case FG_COMPONENT_MTP:return tensor->kind==FG_TENSOR_MTP;
        default:return false;
    }
}

static void hash_tensor_record(fg_sha256 *hash,const fg_tensor_record *tensor){
    size_t name_bytes=strnlen(tensor->name,sizeof(tensor->name));
    hash_u32(hash,(uint32_t)name_bytes);
    fg_sha256_update(hash,tensor->name,name_bytes);
    hash_u64(hash,tensor->offset);hash_u64(hash,tensor->bytes);
    hash_u32(hash,tensor->ggml_type);hash_u32(hash,tensor->dims);
    for(uint32_t axis=0;axis<4u;axis++)hash_u64(hash,tensor->shape[axis]);
    hash_u16(hash,tensor->rank);hash_u16(hash,tensor->layer);hash_u16(hash,tensor->expert);
    hash_u32(hash,tensor->kind);hash_u32(hash,tensor->layout);
    fg_sha256_update(hash,tensor->sha256,sizeof(tensor->sha256));
}

static void component_digest(const fg_manifest *manifest,uint32_t component,uint8_t out[32]){
    if(!(manifest->flags&component_flag(component))){memset(out,0,32);return;}
    static const char domain[]="flash-gordon-component-v1";
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,domain,sizeof(domain)-1u);
    hash_u32(&hash,component);uint32_t count=0;
    for(uint32_t i=0;i<manifest->tensor_count;i++)if(tensor_in_component(&manifest->tensors[i],component))count++;
    hash_u32(&hash,count);
    for(uint32_t i=0;i<manifest->tensor_count;i++)if(tensor_in_component(&manifest->tensors[i],component)){
        hash_u32(&hash,i);hash_tensor_record(&hash,&manifest->tensors[i]);
    }
    fg_sha256_final(&hash,out);
}

static void quantization_digest(const fg_manifest *manifest,uint8_t out[32]){
    static const char domain[]="flash-gordon-quantization-v1";
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,domain,sizeof(domain)-1u);
    hash_u32(&hash,manifest->tensor_count);
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        const fg_tensor_record *tensor=&manifest->tensors[i];
        hash_u32(&hash,i);hash_u32(&hash,tensor->kind);hash_u32(&hash,tensor->ggml_type);
        hash_u32(&hash,tensor->layout);hash_u32(&hash,tensor->dims);
        for(uint32_t axis=0;axis<4u;axis++)hash_u64(&hash,tensor->shape[axis]);
    }
    fg_sha256_final(&hash,out);
}

static void policy_digest(const char *domain,fg_position_mode mode,uint8_t out[32]){
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,domain,strlen(domain));
    hash_u32(&hash,(uint32_t)mode);fg_sha256_final(&hash,out);
}

static void rank_state_digest(const fg_manifest *manifest,const fg_manifest_contract *contract,
                              uint32_t rank,uint8_t out[32]){
    static const char domain[]="flash-gordon-rank-state-format-v1";
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,domain,sizeof(domain)-1u);
    hash_u32(&hash,rank);hash_u32(&hash,manifest->max_context);
    hash_u32(&hash,contract->position_mode);
    hash_u32(&hash,contract->logical_context_tokens);
    hash_u32(&hash,contract->gpu_index_tokens);
    hash_u32(&hash,contract->qsa_hot_record_tokens);
    hash_u64(&hash,contract->host_page_cache_bytes);
    fg_sha256_update(&hash,contract->state_format_sha256,32u);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)if(manifest->layer_owner[layer]==rank){
        hash_u32(&hash,layer);
    }
    fg_sha256_final(&hash,out);
}

static void build_contract(const fg_manifest *manifest,fg_manifest_contract *contract){
    uint32_t boot_context=manifest->native_context<FG_MANIFEST_DEFAULT_CONTEXT_TOKENS?
        manifest->native_context:FG_MANIFEST_DEFAULT_CONTEXT_TOKENS;
    *contract=manifest->session;
    contract->version=FG_MANIFEST_CONTRACT_VERSION;
    contract->minimum_protocol_version=
        manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION?
        FG_PROTOCOL_MIN_VERSION:FG_PROTOCOL_VERSION;
    contract->flags=0u;
    if(!contract->logical_context_tokens)contract->logical_context_tokens=boot_context;
    if(!contract->gpu_index_tokens)contract->gpu_index_tokens=boot_context;
    if(!contract->qsa_hot_record_tokens)contract->qsa_hot_record_tokens=boot_context;
    policy_digest("flash-gordon-rope-policy-v1",(fg_position_mode)contract->position_mode,
                  contract->rope_policy_sha256);
    quantization_digest(manifest,contract->quantization_sha256);
    policy_digest("flash-gordon-session-state-format-v1",
                  (fg_position_mode)contract->position_mode,contract->state_format_sha256);
    for(uint32_t component=0;component<FG_COMPONENT_COUNT;component++)
        component_digest(manifest,component,contract->component_sha256[component]);
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        rank_state_digest(manifest,contract,rank,contract->rank_state_format_sha256[rank]);
    memset(contract->reserved,0,sizeof(contract->reserved));
}

static size_t manifest_bytes(const fg_manifest *manifest){
    return manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION?
        FG_MANIFEST_V4_BYTES:sizeof(*manifest);
}

static void manifest_digest(const fg_manifest *manifest,uint8_t out[32]){
    size_t bytes=manifest_bytes(manifest);
    fg_manifest *copy=malloc(sizeof(*copy));
    if(!copy){memset(out,0,32);return;}
    memcpy(copy,manifest,sizeof(*copy));memset(copy->manifest_sha256,0,32);
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,copy,bytes);
    fg_sha256_final(&hash,out);free(copy);
}

void fg_manifest_init(fg_manifest *manifest){
    memset(manifest,0,sizeof(*manifest));
    manifest->magic=FG_MANIFEST_MAGIC;
    manifest->format_version=FG_MANIFEST_FORMAT_VERSION;
    manifest->protocol_version=FG_PROTOCOL_VERSION;
    manifest->header_bytes=(uint32_t)sizeof(*manifest);
    manifest->rank_count=FG_RANK_COUNT;manifest->layer_count=FG_LAYER_COUNT;
    manifest->expert_count=FG_EXPERT_COUNT;manifest->hidden_size=FG_HIDDEN_SIZE;
    manifest->top_k=FG_TOP_K;manifest->required_cu=FG_REQUIRED_CU;
    manifest->native_context=FG_NATIVE_CONTEXT;manifest->max_context=FG_MAX_CONTEXT;
    manifest->prefill_microbatch=FG_DEFAULT_MICROBATCH;
    manifest->prefill_window=FG_DEFAULT_WINDOW;
    manifest->persistent_cap_bytes=FG_PERSISTENT_CAP_BYTES;
    manifest->residency_cap_bytes=FG_RESIDENCY_CAP_BYTES;
    manifest->session.position_mode=FG_POSITION_TEXT;
    fg_topology_build(manifest);(void)fg_topology_assign_round_robin(manifest,NULL);
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        manifest->ranks[rank].scratch_bytes=fg_q38_runtime_scratch_bytes(
            rank,manifest->prefill_microbatch,manifest->prefill_window,manifest->max_context);
    build_contract(manifest,&manifest->session);
}

static fg_status validate_contract(const fg_manifest *manifest,fg_error *err){
    const fg_manifest_contract *contract=&manifest->session;
    if(contract->version!=FG_MANIFEST_CONTRACT_VERSION||
       contract->minimum_protocol_version!=FG_PROTOCOL_VERSION||
       contract->position_mode>FG_POSITION_FOUR_AXIS||contract->flags){
        fg_error_set(err,FG_ERR_MISMATCH,"unsupported manifest session contract");
        return FG_ERR_MISMATCH;
    }
    if(!contract->logical_context_tokens||
       contract->logical_context_tokens>manifest->native_context||
       !contract->gpu_index_tokens||
       contract->gpu_index_tokens>contract->logical_context_tokens||
       !contract->qsa_hot_record_tokens||
       contract->qsa_hot_record_tokens>contract->logical_context_tokens){
        fg_error_set(err,FG_ERR_FORMAT,"invalid manifest session memory budgets");
        return FG_ERR_FORMAT;
    }
    uint8_t reserved=0;for(size_t i=0;i<sizeof(contract->reserved);i++)reserved|=contract->reserved[i];
    if(reserved){fg_error_set(err,FG_ERR_FORMAT,"manifest session reserved bytes are non-zero");return FG_ERR_FORMAT;}
    fg_manifest_contract expected=*contract;build_contract(manifest,&expected);
    if(memcmp(contract->rope_policy_sha256,expected.rope_policy_sha256,32u)||
       memcmp(contract->quantization_sha256,expected.quantization_sha256,32u)||
       memcmp(contract->state_format_sha256,expected.state_format_sha256,32u)||
       memcmp(contract->component_sha256,expected.component_sha256,
              sizeof(contract->component_sha256))||
       memcmp(contract->rank_state_format_sha256,expected.rank_state_format_sha256,
              sizeof(contract->rank_state_format_sha256))){
        fg_error_set(err,FG_ERR_MISMATCH,"manifest session or component fingerprint mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_manifest_validate(const fg_manifest *manifest,fg_error *err){
    if(!manifest||manifest->magic!=FG_MANIFEST_MAGIC){
        fg_error_set(err,FG_ERR_FORMAT,"unsupported manifest header");return FG_ERR_FORMAT;
    }
    bool legacy=manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION;
    bool current=manifest->format_version==FG_MANIFEST_FORMAT_VERSION;
    uint32_t expected_header=legacy?(uint32_t)FG_MANIFEST_V4_BYTES:(uint32_t)sizeof(*manifest);
    uint32_t expected_protocol=legacy?FG_PROTOCOL_MIN_VERSION:FG_PROTOCOL_VERSION;
    if((!legacy&&!current)||manifest->header_bytes!=expected_header||
       manifest->protocol_version!=expected_protocol){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "unsupported manifest v%u protocol %u header %u",
                     manifest->format_version,manifest->protocol_version,manifest->header_bytes);
        return FG_ERR_MISMATCH;
    }
    if(manifest->rank_count!=FG_RANK_COUNT||manifest->layer_count!=FG_LAYER_COUNT||
       manifest->expert_count!=FG_EXPERT_COUNT||manifest->hidden_size!=FG_HIDDEN_SIZE||
       manifest->top_k!=FG_TOP_K){
        fg_error_set(err,FG_ERR_MISMATCH,"manifest model constants do not match this binary");
        return FG_ERR_MISMATCH;
    }
    if(!manifest->native_context||manifest->native_context>manifest->max_context||
       manifest->max_context>FG_MAX_CONTEXT){
        fg_error_set(err,FG_ERR_FORMAT,"manifest context limits are invalid");return FG_ERR_FORMAT;
    }
    if(manifest->required_cu!=24&&manifest->required_cu!=40){
        fg_error_set(err,FG_ERR_FORMAT,"required CU count must be 24 or 40");return FG_ERR_FORMAT;
    }
    if(manifest->prefill_microbatch!=128&&manifest->prefill_microbatch!=256&&
       manifest->prefill_microbatch!=512){
        fg_error_set(err,FG_ERR_FORMAT,"prefill microbatch is not qualified");return FG_ERR_FORMAT;
    }
    if(manifest->prefill_window<1||manifest->prefill_window>4||
       manifest->tensor_count>FG_MAX_TENSORS){
        fg_error_set(err,FG_ERR_FORMAT,"invalid prefill window or tensor count");return FG_ERR_FORMAT;
    }
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        const fg_tensor_record *tensor=&manifest->tensors[i];
        if(tensor->dims==0||tensor->dims>4){
            fg_error_set(err,FG_ERR_FORMAT,"tensor %u has invalid dimension count",i);
            return FG_ERR_FORMAT;
        }
        for(uint32_t dimension=0;dimension<tensor->dims;dimension++)if(!tensor->shape[dimension]){
            fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an empty dimension",i);
            return FG_ERR_FORMAT;
        }
        if(tensor->layout>FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED){
            fg_error_set(err,FG_ERR_FORMAT,"tensor %u has unknown storage layout %u",
                         i,tensor->layout);return FG_ERR_FORMAT;
        }
        if(tensor->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED){
            if(tensor->ggml_type!=8u||tensor->dims!=2u||tensor->shape[0]>UINT32_MAX||
               tensor->shape[1]>UINT32_MAX||tensor->shape[0]%FG_QK8_0){
                fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an invalid cooked Q8_0 layout",i);
                return FG_ERR_FORMAT;
            }
            uint64_t blocks=tensor->shape[0]/FG_QK8_0;
            uint64_t quant_offset=fg_align_up_u64(FG_Q8_0_COOK_ROWS*blocks*sizeof(uint16_t),
                                                  FG_Q8_0_COOK_ALIGNMENT);
            uint64_t tile_bytes=fg_align_up_u64(
                quant_offset+FG_Q8_0_COOK_ROWS*blocks*FG_QK8_0,FG_Q8_0_COOK_ALIGNMENT);
            uint64_t tiles=(tensor->shape[1]+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS;
            if(tile_bytes>UINT32_MAX||tiles>UINT64_MAX/tile_bytes||
               tensor->bytes!=tile_bytes*tiles||tensor->bytes>UINT32_MAX){
                fg_error_set(err,FG_ERR_FORMAT,"tensor %u has an invalid cooked Q8_0 layout",i);
                return FG_ERR_FORMAT;
            }
        }
        if(tensor->layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED){
            uint64_t matrix=tensor->shape[0]<=UINT32_MAX&&tensor->shape[1]<=UINT32_MAX?
                fg_k_quant_cooked_matrix_bytes((uint32_t)tensor->shape[0],
                                               (uint32_t)tensor->shape[1],
                                               tensor->ggml_type):0u;
            if(tensor->kind!=FG_TENSOR_ROUTED_EXPERT||tensor->dims!=3u||
               tensor->shape[2]!=FG_EXPERTS_PER_RANK||!matrix||
               matrix>UINT64_MAX/tensor->shape[2]||
               tensor->bytes!=matrix*tensor->shape[2]){
                fg_error_set(err,FG_ERR_FORMAT,
                             "tensor %u has an invalid cooked K-quant expert layout",i);
                return FG_ERR_FORMAT;
            }
        }
        if(tensor->layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED){
            uint64_t matrix=tensor->shape[0]<=UINT32_MAX&&tensor->shape[1]<=UINT32_MAX?
                fg_q5_1_cooked_matrix_bytes((uint32_t)tensor->shape[0],
                                            (uint32_t)tensor->shape[1]):0u;
            if(tensor->kind!=FG_TENSOR_ROUTED_EXPERT||tensor->ggml_type!=7u||
               tensor->dims!=3u||tensor->shape[2]!=FG_EXPERTS_PER_RANK||!matrix||
               matrix>UINT64_MAX/tensor->shape[2]||
               tensor->bytes!=matrix*tensor->shape[2]){
                fg_error_set(err,FG_ERR_FORMAT,
                             "tensor %u has an invalid cooked Q5_1 expert layout",i);
                return FG_ERR_FORMAT;
            }
        }
    }
    for(uint32_t rank_index=0;rank_index<FG_RANK_COUNT;rank_index++){
        const fg_rank_record *rank=&manifest->ranks[rank_index];
        uint64_t resident=rank->persistent_bytes+rank->transient_bytes+rank->kv_bytes+
                          rank->scratch_bytes+rank->driver_reserve_bytes;
        uint64_t required_scratch=fg_q38_runtime_scratch_bytes(
            rank_index,manifest->prefill_microbatch,manifest->prefill_window,
            manifest->max_context);
        if(required_scratch==UINT64_MAX||rank->scratch_bytes<required_scratch){
            fg_error_set(err,FG_ERR_LIMIT,
                         "rank %u scratch ledger is %llu bytes, requires at least %llu for prefill %ux%u",
                         rank_index,(unsigned long long)rank->scratch_bytes,
                         (unsigned long long)required_scratch,manifest->prefill_microbatch,
                         manifest->prefill_window);return FG_ERR_LIMIT;
        }
        if(rank->persistent_bytes>manifest->persistent_cap_bytes||
           resident>manifest->residency_cap_bytes){
            fg_error_set(err,FG_ERR_LIMIT,
                         "rank %u memory ledger exceeds cap: persistent %.3f GiB, residency %.3f GiB",
                         rank_index,(double)rank->persistent_bytes/(1ull<<30),
                         (double)resident/(1ull<<30));return FG_ERR_LIMIT;
        }
    }
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        if(manifest->layer_owner[layer]!=layer%FG_RANK_COUNT){
            fg_error_set(err,FG_ERR_FORMAT,"layer %u owner mismatch",layer);return FG_ERR_FORMAT;
        }
        uint16_t counts[FG_RANK_COUNT]={0};
        for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++){
            uint16_t rank=manifest->expert_rank[layer][expert];
            if(rank>=FG_RANK_COUNT||!fg_topology_rank_in_layer(manifest,layer,rank)){
                fg_error_set(err,FG_ERR_FORMAT,"layer %u expert %u assigned outside group",
                             layer,expert);return FG_ERR_FORMAT;
            }
            counts[rank]++;
        }
        for(uint32_t group=0;group<FG_GROUP_SIZE;group++){
            uint32_t rank=manifest->layer_groups[layer][group];
            if(counts[rank]!=FG_EXPERTS_PER_RANK){
                fg_error_set(err,FG_ERR_FORMAT,"layer %u rank %u owns %u experts, expected %u",
                             layer,rank,counts[rank],FG_EXPERTS_PER_RANK);
                return FG_ERR_FORMAT;
            }
        }
    }
    if(current){
        fg_status status=validate_contract(manifest,err);
        if(status!=FG_OK)return status;
    }
    uint8_t digest[32];manifest_digest(manifest,digest);
    if(memcmp(digest,manifest->manifest_sha256,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"manifest SHA-256 mismatch");return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_manifest_validate_deployment(const fg_manifest *manifest,fg_error *err){
    fg_status status=fg_manifest_validate(manifest,err);if(status!=FG_OK)return status;
    if(manifest->format_version==FG_MANIFEST_FORMAT_VERSION&&
       manifest->session.position_mode==FG_POSITION_FOUR_AXIS){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "four-axis position execution is not enabled in this runtime");
        return FG_ERR_UNAVAILABLE;
    }
    uint32_t boot_context=manifest->native_context<FG_MANIFEST_DEFAULT_CONTEXT_TOKENS?
        manifest->native_context:FG_MANIFEST_DEFAULT_CONTEXT_TOKENS;
    if(manifest->format_version==FG_MANIFEST_FORMAT_VERSION&&
       (manifest->session.logical_context_tokens!=boot_context||
        manifest->session.gpu_index_tokens!=boot_context||
        manifest->session.qsa_hot_record_tokens!=boot_context||
        manifest->session.host_page_cache_bytes)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "tiered QSA is not enabled for deployment");
        return FG_ERR_UNAVAILABLE;
    }
    uint32_t missing=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED&~manifest->flags;
    if(missing){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "text deployment pack is incomplete (missing component flags 0x%02x)",missing);
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_manifest_validate_compatibility(const fg_manifest *manifest,
                                             uint32_t required_protocol_version,
                                             fg_position_mode position_mode,
                                             fg_error *err){
    if(!manifest||required_protocol_version<FG_PROTOCOL_MIN_VERSION||
       required_protocol_version>FG_PROTOCOL_VERSION||
       position_mode>FG_POSITION_FOUR_AXIS){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid manifest compatibility request");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_manifest_validate(manifest,err);
    if(status!=FG_OK)return status;
    uint32_t minimum_protocol_version=
        manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION?
        FG_PROTOCOL_MIN_VERSION:manifest->session.minimum_protocol_version;
    if(required_protocol_version<minimum_protocol_version||
       required_protocol_version>manifest->protocol_version){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "protocol %u is outside manifest compatibility range [%u,%u]",
                     required_protocol_version,minimum_protocol_version,
                     manifest->protocol_version);
        return FG_ERR_MISMATCH;
    }
    fg_position_mode actual=manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION?
        FG_POSITION_TEXT:(fg_position_mode)manifest->session.position_mode;
    if(actual!=position_mode){
        fg_error_set(err,FG_ERR_MISMATCH,"manifest position mode %u does not match required mode %u",
                     actual,position_mode);return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_manifest_write(const char *path,fg_manifest *manifest,fg_error *err){
    if(!path||!manifest){fg_error_set(err,FG_ERR_ARGUMENT,"invalid manifest write");return FG_ERR_ARGUMENT;}
    if(manifest->format_version==FG_MANIFEST_FORMAT_VERSION){
        manifest->protocol_version=FG_PROTOCOL_VERSION;
        manifest->header_bytes=(uint32_t)sizeof(*manifest);
        build_contract(manifest,&manifest->session);
    }else if(manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION){
        manifest->protocol_version=FG_PROTOCOL_MIN_VERSION;
        manifest->header_bytes=(uint32_t)FG_MANIFEST_V4_BYTES;
        memset(&manifest->session,0,sizeof(manifest->session));
        manifest->session.position_mode=FG_POSITION_TEXT;
        build_contract(manifest,&manifest->session);
    }else{
        fg_error_set(err,FG_ERR_MISMATCH,"cannot write unsupported manifest version %u",
                     manifest->format_version);return FG_ERR_MISMATCH;
    }
    uint8_t digest[32];manifest_digest(manifest,digest);
    if(digest_is_zero(digest)){fg_error_set(err,FG_ERR_OOM,"hash manifest");return FG_ERR_OOM;}
    memcpy(manifest->manifest_sha256,digest,32u);
    size_t bytes=manifest_bytes(manifest);
    char temporary[1024];
    if(snprintf(temporary,sizeof(temporary),"%s.tmp.%ld",path,(long)getpid())>=
       (int)sizeof(temporary)){
        fg_error_set(err,FG_ERR_ARGUMENT,"manifest path too long");return FG_ERR_ARGUMENT;
    }
    FILE *file=fopen(temporary,"wb");
    if(!file){fg_error_set(err,FG_ERR_IO,"create %s: %s",temporary,strerror(errno));return FG_ERR_IO;}
    bool ok=fwrite(manifest,1,bytes,file)==bytes&&fflush(file)==0&&fsync(fileno(file))==0;
    int close_status=fclose(file);
    if(!ok||close_status){
        fg_error_set(err,FG_ERR_IO,"write %s: %s",temporary,strerror(errno));
        unlink(temporary);return FG_ERR_IO;
    }
    if(rename(temporary,path)!=0){
        fg_error_set(err,FG_ERR_IO,"rename %s to %s: %s",temporary,path,strerror(errno));
        unlink(temporary);return FG_ERR_IO;
    }
    return FG_OK;
}

fg_status fg_manifest_read(const char *path,fg_manifest *manifest,fg_error *err){
    if(!path||!manifest){fg_error_set(err,FG_ERR_ARGUMENT,"invalid manifest read");return FG_ERR_ARGUMENT;}
    FILE *file=fopen(path,"rb");
    if(!file){fg_error_set(err,FG_ERR_IO,"open %s: %s",path,strerror(errno));return FG_ERR_IO;}
    fg_manifest probe;memset(&probe,0,sizeof(probe));
    size_t probe_bytes=offsetof(fg_manifest,rank_count);
    if(fread(&probe,1,probe_bytes,file)!=probe_bytes){
        fclose(file);fg_error_set(err,FG_ERR_FORMAT,"manifest %s has a truncated header",path);
        return FG_ERR_FORMAT;
    }
    size_t expected;
    if(probe.format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION)expected=FG_MANIFEST_V4_BYTES;
    else if(probe.format_version==FG_MANIFEST_FORMAT_VERSION)expected=sizeof(*manifest);
    else{
        fclose(file);fg_error_set(err,FG_ERR_MISMATCH,"manifest %s has unsupported version %u",
                                  path,probe.format_version);return FG_ERR_MISMATCH;
    }
    if(fseeko(file,0,SEEK_END)!=0||ftello(file)!=(off_t)expected||fseeko(file,0,SEEK_SET)!=0){
        fclose(file);fg_error_set(err,FG_ERR_FORMAT,"manifest %s has wrong size",path);
        return FG_ERR_FORMAT;
    }
    memset(manifest,0,sizeof(*manifest));
    size_t read_bytes=fread(manifest,1,expected,file);int extra=fgetc(file);fclose(file);
    if(read_bytes!=expected||extra!=EOF){
        fg_error_set(err,FG_ERR_FORMAT,"manifest %s has wrong size",path);return FG_ERR_FORMAT;
    }
    fg_status status=fg_manifest_validate(manifest,err);
    if(status==FG_OK&&manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION)
        build_contract(manifest,&manifest->session);
    return status;
}

fg_status fg_manifest_add_tensor(fg_manifest *manifest,const fg_tensor_record *record,
                                 fg_error *err){
    if(!manifest||!record){fg_error_set(err,FG_ERR_ARGUMENT,"invalid manifest tensor");return FG_ERR_ARGUMENT;}
    if(manifest->tensor_count>=FG_MAX_TENSORS){
        fg_error_set(err,FG_ERR_LIMIT,"manifest tensor limit exceeded");return FG_ERR_LIMIT;
    }
    manifest->tensors[manifest->tensor_count++]=*record;return FG_OK;
}

void fg_manifest_print(const fg_manifest *manifest){
    printf("Flash Gordon manifest v%u protocol=%u CU=%u tensors=%u prefill=%ux%u context=%u/%u position=%s\n",
           manifest->format_version,manifest->protocol_version,manifest->required_cu,
           manifest->tensor_count,manifest->prefill_microbatch,manifest->prefill_window,
           manifest->native_context,manifest->max_context,
           manifest->session.position_mode==FG_POSITION_FOUR_AXIS?"four-axis":"text");
    uint32_t cooked_q8=0,cooked_k=0,cooked_q5=0;uint64_t cooked_bytes=0;
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        uint32_t layout=manifest->tensors[i].layout;
        if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED)cooked_q8++;
        else if(layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED)cooked_k++;
        else if(layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED)cooked_q5++;
        if(layout!=FG_TENSOR_LAYOUT_GGML)cooked_bytes+=manifest->tensors[i].bytes;
    }
    printf("layouts ggml=%u cooked-q8=%u cooked-k=%u cooked-q5_1=%u cooked-bytes=%.3f GiB\n",
           manifest->tensor_count-cooked_q8-cooked_k-cooked_q5,cooked_q8,cooked_k,cooked_q5,
           (double)cooked_bytes/(1ull<<30));
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        const fg_rank_record *record=&manifest->ranks[rank];
        printf("rank %u %-21s persistent=%6.3f GiB residency=%6.3f GiB state-file=%6.3f GiB tensors=%u\n",
               rank,record->endpoint,(double)record->persistent_bytes/(1ull<<30),
               (double)(record->persistent_bytes+record->transient_bytes+record->kv_bytes+
                        record->scratch_bytes+record->driver_reserve_bytes)/(1ull<<30),
               (double)record->state_file_bytes/(1ull<<30),record->tensor_count);
    }
}
