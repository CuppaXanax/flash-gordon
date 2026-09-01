#ifndef FLASH_GORDON_PIPELINE_MANIFEST_FIXTURE_H
#define FLASH_GORDON_PIPELINE_MANIFEST_FIXTURE_H

#include "fg_embedding.h"
#include "fg_ngram.h"

typedef struct pipeline_fixture_tensor {
    const char *suffix;
    uint32_t type;
    uint32_t dims;
    uint64_t shape[3];
    fg_tensor_kind kind;
} pipeline_fixture_tensor;

static bool pipeline_fixture_add_tensor(
    fg_manifest *manifest,const char *name,const pipeline_fixture_tensor *spec,
    uint32_t rank,uint32_t layer,uint64_t rank_offsets[FG_RANK_COUNT],
    uint64_t *external_offset,const uint8_t digest[32],fg_error *error){
    fg_tensor_record record={0};
    snprintf(record.name,sizeof(record.name),"%s",name);
    record.ggml_type=spec->type;
    record.dims=spec->dims;
    memcpy(record.shape,spec->shape,sizeof(spec->shape));
    record.rank=(uint16_t)rank;
    record.layer=(uint16_t)layer;
    record.expert=UINT16_MAX;
    record.kind=(uint8_t)spec->kind;
    record.layout=spec->kind==FG_TENSOR_HOST_CACHE?
        FG_TENSOR_LAYOUT_HOST_Q8_0:FG_TENSOR_LAYOUT_GGML;
    if(fg_tensor_record_expected_bytes(&record,&record.bytes,error)!=FG_OK)
        return false;
    if(spec->kind==FG_TENSOR_HOST_CACHE)record.offset=0u;
    else if(rank<FG_RANK_COUNT){
        record.offset=fg_align_up_u64(rank_offsets[rank],FG_ALIGNMENT);
        rank_offsets[rank]=record.offset+record.bytes;
        manifest->ranks[rank].tensor_count++;
    }else{
        record.offset=fg_align_up_u64(*external_offset,FG_ALIGNMENT);
        *external_offset=record.offset+record.bytes;
    }
    memcpy(record.sha256,digest,32u);
    return fg_manifest_add_tensor(manifest,&record,error)==FG_OK;
}

static bool build_pipeline_deployment_fixture(
    fg_manifest *manifest,const uint8_t digest[32],fg_error *error){
    static const pipeline_fixture_tensor global[]={
        {"token_embd.weight",8u,2u,{2560u,248320u},FG_TENSOR_HOST_CACHE},
        {"output.weight",8u,2u,{2560u,248320u},FG_TENSOR_COMMON},
        {"output_hc_down.weight",8u,2u,{10240u,320u},FG_TENSOR_COMMON},
        {"output_hc_norm.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"output_hc_up.weight",8u,2u,{320u,10240u},FG_TENSOR_COMMON},
        {"per_layer_token_embd.weight",20u,2u,{160u,320001536u},
         FG_TENSOR_NGRAM}
    };
    static const pipeline_fixture_tensor common[]={
        {"ffn_down_exps.weight",7u,3u,{640u,2560u,512u},
         FG_TENSOR_ROUTED_EXPERT},
        {"ffn_down_shexp.weight",8u,2u,{640u,2560u},FG_TENSOR_COMMON},
        {"ffn_gate_exps.weight",12u,3u,{2560u,640u,512u},
         FG_TENSOR_ROUTED_EXPERT},
        {"ffn_gate_inp.weight",0u,2u,{2560u,512u},FG_TENSOR_COMMON},
        {"ffn_gate_inp_shexp.weight",0u,1u,{2560u},FG_TENSOR_COMMON},
        {"ffn_gate_shexp.weight",8u,2u,{2560u,640u},FG_TENSOR_COMMON},
        {"ffn_up_exps.weight",12u,3u,{2560u,640u,512u},
         FG_TENSOR_ROUTED_EXPERT},
        {"ffn_up_shexp.weight",8u,2u,{2560u,640u},FG_TENSOR_COMMON},
        {"hc_attn_down.weight",8u,2u,{10240u,320u},FG_TENSOR_COMMON},
        {"hc_attn_inject.weight",0u,2u,{10240u,4u},FG_TENSOR_COMMON},
        {"hc_attn_norm.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"hc_attn_up.weight",8u,2u,{320u,10240u},FG_TENSOR_COMMON},
        {"hc_ffn_down.weight",8u,2u,{10240u,320u},FG_TENSOR_COMMON},
        {"hc_ffn_inject.weight",0u,2u,{10240u,4u},FG_TENSOR_COMMON},
        {"hc_ffn_norm.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"hc_ffn_up.weight",8u,2u,{320u,10240u},FG_TENSOR_COMMON}
    };
    static const pipeline_fixture_tensor gdn[]={
        {"attn_gate.weight",8u,2u,{2560u,6144u},FG_TENSOR_COMMON},
        {"attn_qkv.weight",8u,2u,{2560u,10240u},FG_TENSOR_COMMON},
        {"ssm_a",0u,1u,{48u},FG_TENSOR_COMMON},
        {"ssm_alpha.weight",0u,2u,{2560u,48u},FG_TENSOR_COMMON},
        {"ssm_beta.weight",0u,2u,{2560u,48u},FG_TENSOR_COMMON},
        {"ssm_conv1d.weight",0u,2u,{4u,10240u},FG_TENSOR_COMMON},
        {"ssm_dt.bias",0u,1u,{48u},FG_TENSOR_COMMON},
        {"ssm_norm.weight",0u,1u,{128u},FG_TENSOR_COMMON},
        {"ssm_out.weight",8u,2u,{6144u,2560u},FG_TENSOR_COMMON}
    };
    static const pipeline_fixture_tensor attention[]={
        {"attn_k.weight",8u,2u,{2560u,512u},FG_TENSOR_COMMON},
        {"attn_k_norm.weight",0u,1u,{256u},FG_TENSOR_COMMON},
        {"attn_output.weight",8u,2u,{6144u,2560u},FG_TENSOR_COMMON},
        {"attn_q.weight",8u,2u,{2560u,12288u},FG_TENSOR_COMMON},
        {"attn_q_norm.weight",0u,1u,{256u},FG_TENSOR_COMMON},
        {"attn_v.weight",8u,2u,{2560u,512u},FG_TENSOR_COMMON},
        {"indexer.k_norm.weight",0u,1u,{128u},FG_TENSOR_COMMON},
        {"indexer.k_proj.weight",30u,2u,{2560u,128u},FG_TENSOR_COMMON},
        {"indexer.q_norm.weight",0u,1u,{128u},FG_TENSOR_COMMON},
        {"indexer.q_proj.weight",30u,2u,{2560u,512u},FG_TENSOR_COMMON}
    };
    static const pipeline_fixture_tensor ple[]={
        {"ple_conv1d.weight",0u,2u,{4u,10240u},FG_TENSOR_COMMON},
        {"ple_key.weight",8u,2u,{2560u,10240u},FG_TENSOR_COMMON},
        {"ple_norm_conv.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"ple_norm_key.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"ple_norm_query.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"ple_value.weight",8u,2u,{2560u,2560u},FG_TENSOR_COMMON}
    };
    fg_manifest_init(manifest);
    if(fg_runtime_profile_apply(
           manifest,FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,error)!=FG_OK)
        return false;
    uint64_t rank_offsets[FG_RANK_COUNT]={0},external_offset=0u;
    for(uint32_t i=0;i<sizeof(global)/sizeof(global[0]);i++){
        uint32_t rank=i==5u?UINT16_MAX:
            i>=1u&&i<=4u?
                manifest->stage_ranks[manifest->stage_count-1u]:
                manifest->stage_ranks[0];
        if(!pipeline_fixture_add_tensor(
               manifest,global[i].suffix,&global[i],rank,UINT16_MAX,
               rank_offsets,&external_offset,digest,error))
            return false;
    }
    char name[FG_TENSOR_NAME_MAX];
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        for(uint32_t i=0;i<sizeof(common)/sizeof(common[0]);i++){
            snprintf(name,sizeof(name),"blk.%u.%s",layer,common[i].suffix);
            if(!pipeline_fixture_add_tensor(
                   manifest,name,&common[i],manifest->layer_owner[layer],layer,
                   rank_offsets,&external_offset,digest,error))
                return false;
        }
        const pipeline_fixture_tensor *set=(layer&3u)==3u?attention:gdn;
        size_t count=(layer&3u)==3u?
            sizeof(attention)/sizeof(attention[0]):
            sizeof(gdn)/sizeof(gdn[0]);
        for(size_t i=0;i<count;i++){
            snprintf(name,sizeof(name),"blk.%u.%s",layer,set[i].suffix);
            if(!pipeline_fixture_add_tensor(
                   manifest,name,&set[i],manifest->layer_owner[layer],layer,
                   rank_offsets,&external_offset,digest,error))
                return false;
        }
    }
    for(uint32_t i=0;i<sizeof(ple)/sizeof(ple[0]);i++){
        snprintf(name,sizeof(name),"blk.1.%s",ple[i].suffix);
        if(!pipeline_fixture_add_tensor(
               manifest,name,&ple[i],manifest->layer_owner[1u],1u,
               rank_offsets,&external_offset,digest,error))
            return false;
    }
    fg_tensor_record tokenizer={0};
    snprintf(tokenizer.name,sizeof(tokenizer.name),"tokenizer/tokenizer.fgt");
    tokenizer.offset=fg_align_up_u64(external_offset,FG_ALIGNMENT);
    tokenizer.bytes=1u;
    tokenizer.dims=1u;
    tokenizer.shape[0]=1u;
    tokenizer.rank=UINT16_MAX;
    tokenizer.layer=UINT16_MAX;
    tokenizer.expert=UINT16_MAX;
    tokenizer.kind=FG_TENSOR_TOKENIZER;
    tokenizer.layout=FG_TENSOR_LAYOUT_GGML;
    memcpy(tokenizer.sha256,digest,32u);
    if(fg_manifest_add_tensor(manifest,&tokenizer,error)!=FG_OK)return false;
    manifest->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
    manifest->host_resident_bytes[manifest->stage_ranks[0]]=
        FG_EMBEDDING_ARTIFACT_BYTES;
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        manifest->ranks[rank].persistent_bytes=
            fg_align_up_u64(rank_offsets[rank],FG_ALIGNMENT);
        manifest->ranks[rank].transient_bytes=
            FG_PACK_RANK_TRANSIENT_BYTES;
        manifest->ranks[rank].driver_reserve_bytes=
            FG_PACK_DRIVER_RESERVE_BYTES;
    }
    fg_q38_account_session_state(manifest);
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        manifest->ranks[rank].scratch_bytes=
            fg_q38_runtime_scratch_bytes_for_manifest(
                manifest,rank,manifest->prefill_microbatch,
                manifest->prefill_window,manifest->max_context);
    for(uint32_t rank=1u;rank<FG_RANK_COUNT;rank++){
        fg_ngram_shard_record *record=
            &manifest->ngram_shards[manifest->ngram_shard_count++];
        record->logical_rank=rank;
        if(fg_q38_ngram_rank_range(rank,&record->row_begin,
                                   &record->row_count,error)!=FG_OK)
            return false;
        record->bytes=record->row_count*FG_NGRAM_ROW_BYTES;
        memcpy(record->sha256,digest,sizeof(record->sha256));
        manifest->host_resident_bytes[rank]=FG_PIPELINE_NGRAM_CACHE_BYTES;
    }
    return manifest->tensor_count==1225u;
}

#endif
