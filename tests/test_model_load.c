#include "fg_manifest.h"
#include "fg_model.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"
#include "fg_runtime.h"
#include "fg_sha256.h"
#include "fg_topology.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int test_cooked_expert_load(void){
    enum{INPUT=256,OUTPUT=8};const uint64_t matrix_bytes=fg_k_quant_cooked_matrix_bytes(INPUT,OUTPUT,12u),tensor_bytes=matrix_bytes*FG_EXPERTS_PER_RANK,file_bytes=fg_align_up_u64(tensor_bytes,FG_ALIGNMENT);char directory[128],path[160];snprintf(directory,sizeof(directory),"/tmp/fg-model-cook-%ld",(long)getpid());snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);if(!matrix_bytes||mkdir(directory,0700)!=0)return 1;uint8_t *source=aligned_alloc(FG_ALIGNMENT,(size_t)file_bytes),*expected=malloc((size_t)matrix_bytes),*candidate=malloc((size_t)matrix_bytes);if(!source||!expected||!candidate){free(candidate);free(expected);free(source);rmdir(directory);return 1;}for(uint64_t i=0;i<tensor_bytes;i++)source[i]=(uint8_t)(i*37u+11u);memset(source+tensor_bytes,0,(size_t)(file_bytes-tensor_bytes));FILE *stream=fopen(path,"wb");if(!stream||fwrite(source,1,(size_t)file_bytes,stream)!=(size_t)file_bytes||fclose(stream)!=0){if(stream)fclose(stream);free(candidate);free(expected);free(source);unlink(path);rmdir(directory);return 1;}fg_manifest *manifest=malloc(sizeof(*manifest));if(!manifest){free(candidate);free(expected);free(source);unlink(path);rmdir(directory);return 1;}fg_manifest_init(manifest);fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"blk.0.ffn_gate_exps.weight.rank0");record.bytes=tensor_bytes;record.ggml_type=12u;record.dims=3u;record.shape[0]=INPUT;record.shape[1]=OUTPUT;record.shape[2]=FG_EXPERTS_PER_RANK;record.rank=0;record.layer=0;record.expert=UINT16_MAX;record.kind=FG_TENSOR_ROUTED_EXPERT;record.layout=FG_TENSOR_LAYOUT_GGML;fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,source,(size_t)tensor_bytes);fg_sha256_final(&hash,record.sha256);fg_error error={0};int ok=fg_manifest_add_tensor(manifest,&record,&error)==FG_OK&&fg_cook_k_quant_rows(source,expected,matrix_bytes,INPUT,OUTPUT,12u);if(setenv("FG_COOK_EXPERTS_ON_LOAD","1",1)!=0)ok=0;fg_model *model=NULL;fg_status status=ok?fg_model_open(&model,manifest,directory,0,&error):FG_ERR_FORMAT;unsetenv("FG_COOK_EXPERTS_ON_LOAD");if(status==FG_ERR_UNAVAILABLE){fprintf(stderr,"SKIP cooked expert model load: %s\n",error.message);free(manifest);free(candidate);free(expected);free(source);unlink(path);rmdir(directory);return 77;}fg_vk_tensor *tensor=status==FG_OK?fg_model_tensor(model,record.name):NULL;ok=ok&&status==FG_OK&&tensor&&fg_vk_tensor_get_format(tensor)==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&fg_vk_tensor_read(tensor,0,candidate,matrix_bytes,&error)==FG_OK&&memcmp(candidate,expected,(size_t)matrix_bytes)==0;fg_model_close(model);free(manifest);free(candidate);free(expected);free(source);unlink(path);rmdir(directory);if(!ok)fprintf(stderr,"cooked expert model load failed: %s\n",error.message);return ok?0:1;
}

typedef struct stage_tensor_spec {
    const char *suffix;
    uint32_t type;
    uint32_t dims;
    uint64_t shape[3];
    fg_tensor_kind kind;
} stage_tensor_spec;

static bool add_stage_tensor(fg_manifest *manifest,const char *name,
                             const stage_tensor_spec *spec,uint32_t rank,
                             uint32_t layer,uint64_t rank_offsets[FG_RANK_COUNT],
                             uint64_t *external_offset,const uint8_t digest[32],
                             fg_error *error){
    fg_tensor_record record={0};
    snprintf(record.name,sizeof(record.name),"%s",name);
    record.ggml_type=spec->type;record.dims=spec->dims;
    memcpy(record.shape,spec->shape,sizeof(spec->shape));record.rank=(uint16_t)rank;
    record.layer=(uint16_t)layer;record.expert=UINT16_MAX;record.kind=(uint8_t)spec->kind;
    record.layout=spec->kind==FG_TENSOR_HOST_CACHE?
        FG_TENSOR_LAYOUT_HOST_Q8_0:FG_TENSOR_LAYOUT_GGML;
    if(fg_tensor_record_expected_bytes(&record,&record.bytes,error)!=FG_OK)return false;
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

static bool build_stage_manifest(fg_manifest *manifest,const uint8_t digest[32],
                                 fg_error *error){
    static const stage_tensor_spec global[]={
        {"token_embd.weight",8u,2u,{2560u,248320u},FG_TENSOR_HOST_CACHE},
        {"output.weight",8u,2u,{2560u,248320u},FG_TENSOR_COMMON},
        {"output_hc_down.weight",8u,2u,{10240u,320u},FG_TENSOR_COMMON},
        {"output_hc_norm.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"output_hc_up.weight",8u,2u,{320u,10240u},FG_TENSOR_COMMON},
        {"per_layer_token_embd.weight",20u,2u,{160u,320001536u},FG_TENSOR_NGRAM}
    };
    static const stage_tensor_spec common[]={
        {"ffn_down_exps.weight",7u,3u,{640u,2560u,512u},FG_TENSOR_ROUTED_EXPERT},
        {"ffn_down_shexp.weight",8u,2u,{640u,2560u},FG_TENSOR_COMMON},
        {"ffn_gate_exps.weight",12u,3u,{2560u,640u,512u},FG_TENSOR_ROUTED_EXPERT},
        {"ffn_gate_inp.weight",0u,2u,{2560u,512u},FG_TENSOR_COMMON},
        {"ffn_gate_inp_shexp.weight",0u,1u,{2560u},FG_TENSOR_COMMON},
        {"ffn_gate_shexp.weight",8u,2u,{2560u,640u},FG_TENSOR_COMMON},
        {"ffn_up_exps.weight",12u,3u,{2560u,640u,512u},FG_TENSOR_ROUTED_EXPERT},
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
    static const stage_tensor_spec gdn[]={
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
    static const stage_tensor_spec attention[]={
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
    static const stage_tensor_spec ple[]={
        {"ple_conv1d.weight",0u,2u,{4u,10240u},FG_TENSOR_COMMON},
        {"ple_key.weight",8u,2u,{2560u,10240u},FG_TENSOR_COMMON},
        {"ple_norm_conv.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"ple_norm_key.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"ple_norm_query.weight",0u,1u,{10240u},FG_TENSOR_COMMON},
        {"ple_value.weight",8u,2u,{2560u,2560u},FG_TENSOR_COMMON}
    };
    fg_manifest_init(manifest);
    if(fg_runtime_profile_apply(
           manifest,FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,error)!=FG_OK)return false;
    uint64_t rank_offsets[FG_RANK_COUNT]={0},external_offset=0u;
    for(uint32_t i=0;i<sizeof(global)/sizeof(global[0]);i++){
        uint32_t rank=i==5u?UINT16_MAX:
            i>=1u&&i<=4u?manifest->stage_ranks[manifest->stage_count-1u]:
            manifest->stage_ranks[0];
        if(!add_stage_tensor(manifest,global[i].suffix,&global[i],rank,UINT16_MAX,
                             rank_offsets,&external_offset,digest,error))return false;
    }
    char name[FG_TENSOR_NAME_MAX];
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        for(uint32_t i=0;i<sizeof(common)/sizeof(common[0]);i++){
            snprintf(name,sizeof(name),"blk.%u.%s",layer,common[i].suffix);
            if(!add_stage_tensor(manifest,name,&common[i],manifest->layer_owner[layer],
                                 layer,rank_offsets,&external_offset,digest,error))return false;
        }
        const stage_tensor_spec *set=(layer&3u)==3u?attention:gdn;
        size_t count=(layer&3u)==3u?sizeof(attention)/sizeof(attention[0]):
            sizeof(gdn)/sizeof(gdn[0]);
        for(size_t i=0;i<count;i++){
            snprintf(name,sizeof(name),"blk.%u.%s",layer,set[i].suffix);
            if(!add_stage_tensor(manifest,name,&set[i],manifest->layer_owner[layer],
                                 layer,rank_offsets,&external_offset,digest,error))return false;
        }
    }
    for(uint32_t i=0;i<sizeof(ple)/sizeof(ple[0]);i++){
        snprintf(name,sizeof(name),"blk.1.%s",ple[i].suffix);
        if(!add_stage_tensor(manifest,name,&ple[i],manifest->layer_owner[1u],1u,
                             rank_offsets,&external_offset,digest,error))return false;
    }
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        manifest->ranks[rank].persistent_bytes=
            fg_align_up_u64(rank_offsets[rank],FG_ALIGNMENT);
    fg_q38_account_session_state(manifest);
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        manifest->ranks[rank].scratch_bytes=fg_q38_runtime_scratch_bytes_for_manifest(
            manifest,rank,manifest->prefill_microbatch,manifest->prefill_window,
            manifest->max_context);
    return manifest->tensor_count==1224u;
}

static int test_pipeline_stage_load(void){
    char directory[128],path[160];
    snprintf(directory,sizeof(directory),"test-model-stage-%ld",(long)getpid());
    snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);
    if(mkdir(directory,0700)!=0)return 1;
    fg_manifest *manifest=malloc(sizeof(*manifest));
    if(!manifest){rmdir(directory);return 1;}
    uint8_t digest[32]={0};fg_error error={0};
    int ok=build_stage_manifest(manifest,digest,&error)&&
        fg_q38_validate_packed_manifest(manifest,&error)==FG_OK;
    fg_manifest *invalid=malloc(sizeof(*invalid));
    if(!invalid)ok=0;
    fg_model *model=NULL;
    if(ok){
        *invalid=*manifest;
        invalid->tensor_count--;
        memmove(&invalid->tensors[0],&invalid->tensors[1],
                (size_t)invalid->tensor_count*sizeof(invalid->tensors[0]));
        ok=fg_model_open_stage(&model,invalid,directory,0u,&error)==FG_ERR_MISMATCH&&
            model==NULL;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[7].offset=invalid->tensors[6].offset;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_MISMATCH;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[6].bytes--;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_MISMATCH;
    }
    if(ok){
        *invalid=*manifest;
        invalid->ranks[0].persistent_bytes-=FG_ALIGNMENT;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_MISMATCH;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[0].rank=UINT16_MAX;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_FORMAT;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[6].rank=UINT16_MAX;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_FORMAT;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[6].rank=FG_RANK_COUNT;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_FORMAT;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[5].rank=0u;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_FORMAT;
    }
    if(ok){
        *invalid=*manifest;
        invalid->tensors[1].rank=0u;
        ok=fg_q38_validate_packed_manifest(invalid,&error)==FG_ERR_MISMATCH;
    }
    if(ok){
        memset(invalid,0,sizeof(*invalid));invalid->tensor_count=1u;
        fg_tensor_record *tokenizer=&invalid->tensors[0];
        snprintf(tokenizer->name,sizeof(tokenizer->name),"tokenizer/tokenizer.fgt");
        tokenizer->bytes=1u;tokenizer->dims=1u;tokenizer->shape[0]=1u;
        tokenizer->rank=UINT16_MAX;tokenizer->layer=UINT16_MAX;
        tokenizer->expert=UINT16_MAX;tokenizer->kind=FG_TENSOR_TOKENIZER;
        ok=fg_manifest_validate_tensor_storage(invalid,&error)==FG_OK;
        tokenizer->rank=0u;
        if(ok)ok=fg_manifest_validate_tensor_storage(invalid,&error)==FG_ERR_FORMAT;
    }
    FILE *file=NULL;
    if(ok){
        file=fopen(path,"wb");
        if(!file||ftruncate(fileno(file),
            (off_t)(manifest->ranks[0].persistent_bytes-FG_ALIGNMENT))!=0||
           fclose(file)!=0)ok=0;
        else ok=fg_model_open_stage(&model,manifest,directory,0u,&error)==
                FG_ERR_MISMATCH&&model==NULL;
    }
    if(ok){
        file=fopen(path,"r+b");
        if(!file||ftruncate(fileno(file),
            (off_t)manifest->ranks[0].persistent_bytes)!=0||fclose(file)!=0)ok=0;
    }
    if(ok){
        *invalid=*manifest;invalid->ranks[0].scratch_bytes--;
        ok=fg_model_open_stage(&model,invalid,directory,0u,&error)==FG_ERR_LIMIT&&
            model==NULL;
    }
    if(ok){
        *invalid=*manifest;invalid->ranks[0].kv_bytes--;
        ok=fg_model_open_stage(&model,invalid,directory,0u,&error)==FG_ERR_LIMIT&&
            model==NULL;
    }
    if(ok){
        *invalid=*manifest;invalid->ranks[0].state_file_bytes--;
        ok=fg_model_open_stage(&model,invalid,directory,0u,&error)==FG_ERR_LIMIT&&
            model==NULL;
    }
    if(ok){
        *invalid=*manifest;invalid->slot_count=1u;fg_topology_seal(invalid);
        ok=fg_model_open_stage(&model,invalid,directory,0u,&error)==FG_ERR_LIMIT&&
            model==NULL;
    }
    if(ok){
        ok=fg_model_open_stage(&model,manifest,directory,0u,&error)==FG_ERR_IO&&
            model==NULL;
    }
    free(invalid);
    fg_model_close(model);free(manifest);unlink(path);rmdir(directory);
    if(!ok)fprintf(stderr,"pipeline stage model load failed: %s\n",error.message);
    return ok?0:1;
}

int main(void){int stage=test_pipeline_stage_load();if(stage)return stage;
    int cooked=test_cooked_expert_load();if(cooked)return cooked;
    char directory[128],path[160];snprintf(directory,sizeof(directory),"/tmp/fg-model-load-%ld",(long)getpid());snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);if(mkdir(directory,0700)!=0){perror("mkdir");return 1;}uint8_t *bytes=aligned_alloc(FG_ALIGNMENT,FG_ALIGNMENT);if(!bytes)return 1;for(uint32_t i=0;i<FG_ALIGNMENT;i++)bytes[i]=(uint8_t)(i*29u+7u);FILE *stream=fopen(path,"wb");if(!stream||fwrite(bytes,1,FG_ALIGNMENT,stream)!=FG_ALIGNMENT||fwrite(bytes,1,FG_ALIGNMENT,stream)!=FG_ALIGNMENT||fclose(stream)!=0){perror("write rank artifact");free(bytes);return 1;}
    fg_manifest *manifest=malloc(sizeof(*manifest));if(!manifest){free(bytes);return 1;}fg_manifest_init(manifest);fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"probe.weight");record.bytes=FG_ALIGNMENT;record.ggml_type=8u;record.dims=2;record.shape[0]=32u;record.shape[1]=32u;record.rank=0;record.layer=UINT16_MAX;record.expert=UINT16_MAX;record.kind=FG_TENSOR_COMMON;record.layout=FG_TENSOR_LAYOUT_Q8_0_COOKED;fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,bytes,FG_ALIGNMENT);fg_sha256_final(&hash,record.sha256);fg_error error={0};if(fg_manifest_add_tensor(manifest,&record,&error)!=FG_OK){fprintf(stderr,"manifest: %s\n",error.message);free(manifest);free(bytes);return 1;}fg_tensor_record generic=record;snprintf(generic.name,sizeof(generic.name),"generic.weight");generic.offset=FG_ALIGNMENT;generic.layout=FG_TENSOR_LAYOUT_GGML;if(fg_manifest_add_tensor(manifest,&generic,&error)!=FG_OK){fprintf(stderr,"manifest: %s\n",error.message);free(manifest);free(bytes);return 1;}
    fg_model *model=NULL;fg_status status=fg_model_open(&model,manifest,directory,0,&error);if(status==FG_ERR_UNAVAILABLE){fprintf(stderr,"SKIP direct Vulkan/io_uring model load: %s\n",error.message);unlink(path);rmdir(directory);free(manifest);free(bytes);return 77;}if(status!=FG_OK){fprintf(stderr,"model load: %s\n",error.message);unlink(path);rmdir(directory);free(manifest);free(bytes);return 1;}fg_vk_tensor *tensor=fg_model_tensor(model,"probe.weight"),*generic_tensor=fg_model_tensor(model,"generic.weight");const fg_tensor_record *found=fg_model_tensor_record(model,"probe.weight");uint8_t check[64];int ok=tensor&&generic_tensor&&found==&manifest->tensors[0]&&fg_vk_tensor_get_format(tensor)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED&&fg_vk_tensor_get_format(generic_tensor)==FG_VK_TENSOR_FORMAT_DEFAULT&&!fg_model_tensor(model,"probe.weigh")&&!fg_model_tensor(model,"missing.weight")&&!fg_model_tensor_record(model,"missing.weight")&&fg_vk_tensor_read(tensor,0,check,sizeof(check),&error)==FG_OK&&memcmp(check,bytes,sizeof(check))==0&&fg_model_weight_bytes(model)==2u*FG_ALIGNMENT;fg_model_close(model);unlink(path);rmdir(directory);free(manifest);free(bytes);if(!ok){fprintf(stderr,"direct model load parity failed: %s\n",error.message);return 1;}puts("Flash Gordon fixed-buffer O_DIRECT to Vulkan arena: PASS");return 0;
}
