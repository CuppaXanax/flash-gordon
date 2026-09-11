#include "fg_manifest.h"
#include "fg_model.h"
#include "fg_ngram.h"
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
    enum{INPUT=256,OUTPUT=8};const uint64_t matrix_bytes=fg_k_quant_cooked_matrix_bytes(INPUT,OUTPUT,12u),tensor_bytes=matrix_bytes*FG_EXPERTS_PER_RANK,file_bytes=fg_align_up_u64(tensor_bytes,FG_ALIGNMENT);char directory[128],path[160];snprintf(directory,sizeof(directory),"/tmp/fg-model-cook-%ld",(long)getpid());snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);if(!matrix_bytes||mkdir(directory,0700)!=0)return 1;uint8_t *source=aligned_alloc(FG_ALIGNMENT,(size_t)file_bytes),*expected=malloc((size_t)matrix_bytes),*candidate=malloc((size_t)matrix_bytes);if(!source||!expected||!candidate){free(candidate);free(expected);free(source);rmdir(directory);return 1;}for(uint64_t i=0;i<tensor_bytes;i++)source[i]=(uint8_t)(i*37u+11u);memset(source+tensor_bytes,0,(size_t)(file_bytes-tensor_bytes));FILE *stream=fopen(path,"wb");if(!stream||fwrite(source,1,(size_t)file_bytes,stream)!=(size_t)file_bytes||fclose(stream)!=0){if(stream)fclose(stream);free(candidate);free(expected);free(source);unlink(path);rmdir(directory);return 1;}fg_manifest *manifest=malloc(sizeof(*manifest));if(!manifest){free(candidate);free(expected);free(source);unlink(path);rmdir(directory);return 1;}fg_manifest_init(manifest);fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"blk.0.ffn_gate_exps.weight.rank0");record.bytes=tensor_bytes;record.ggml_type=12u;record.dims=3u;record.shape[0]=INPUT;record.shape[1]=OUTPUT;record.shape[2]=FG_EXPERTS_PER_RANK;record.rank=0;record.layer=0;record.expert=UINT16_MAX;record.kind=FG_TENSOR_ROUTED_EXPERT;record.layout=FG_TENSOR_LAYOUT_GGML;fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,source,(size_t)tensor_bytes);fg_sha256_final(&hash,record.sha256);fg_error error={0};int ok=fg_manifest_add_tensor(manifest,&record,&error)==FG_OK&&fg_cook_k_quant_rows(source,expected,matrix_bytes,INPUT,OUTPUT,12u);fg_model *model=NULL;fg_status status=ok?fg_model_open(&model,manifest,directory,0,&error):FG_ERR_FORMAT;if(status==FG_ERR_UNAVAILABLE){fprintf(stderr,"SKIP cooked expert model load: %s\n",error.message);free(manifest);free(candidate);free(expected);free(source);unlink(path);rmdir(directory);return 77;}fg_vk_tensor *tensor=status==FG_OK?fg_model_tensor(model,record.name):NULL;ok=ok&&status==FG_OK&&tensor&&fg_vk_tensor_get_format(tensor)==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&fg_vk_tensor_read(tensor,0,candidate,matrix_bytes,&error)==FG_OK&&memcmp(candidate,expected,(size_t)matrix_bytes)==0;fg_model_close(model);free(manifest);free(candidate);free(expected);free(source);unlink(path);rmdir(directory);if(!ok)fprintf(stderr,"cooked expert model load failed: %s\n",error.message);return ok?0:1;
}

static int test_common_projection_load(void){
    enum{INPUT=256,OUTPUT=64,COUNT=3};
    const char *names[]={"blk.0.attn_qkv.weight","blk.3.attn_q.weight","token_embd.weight"};
    uint64_t bytes=fg_q8_0_cooked_matrix_bytes(INPUT,OUTPUT);
    uint64_t stride=fg_align_up_u64(bytes,FG_ALIGNMENT);
    uint8_t *source=calloc(COUNT,(size_t)stride),*expected=malloc((size_t)bytes);
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_model *model=NULL;
    fg_error error={0};
    if(!source||!expected||!manifest){free(manifest);free(expected);free(source);return 1;}
    fg_manifest_init(manifest);
    char directory[128],path[160];
    snprintf(directory,sizeof(directory),"/tmp/fg-common-cook-%ld",(long)getpid());
    snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);
    int ok=mkdir(directory,0700)==0;
    for(uint32_t i=0;ok&&i<COUNT;i++){
        uint8_t *raw=source+i*stride;
        for(uint64_t b=0;b<bytes;b++)raw[b]=(uint8_t)(b*29u+i*17u);
        fg_tensor_record r={0};
        snprintf(r.name,sizeof(r.name),"%s",names[i]);
        r.offset=i*stride;r.bytes=bytes;r.ggml_type=8u;r.dims=2u;
        r.shape[0]=INPUT;r.shape[1]=OUTPUT;r.kind=FG_TENSOR_COMMON;
        r.layer=i==0u?0u:i==1u?3u:UINT16_MAX;r.expert=UINT16_MAX;
        fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,raw,(size_t)bytes);
        fg_sha256_final(&hash,r.sha256);
        ok=fg_manifest_add_tensor(manifest,&r,&error)==FG_OK;
    }
    if(ok){
        FILE *file=fopen(path,"wb");
        if(!file)ok=0;
        else{ok=fwrite(source,(size_t)stride,COUNT,file)==COUNT;if(fclose(file))ok=0;}
    }
    fg_status status=ok?fg_model_open(&model,manifest,directory,0u,&error):FG_ERR_IO;
    ok=ok&&status==FG_OK;
    for(uint32_t i=0;ok&&i<COUNT;i++){
        fg_vk_tensor *tensor=fg_model_tensor(model,names[i]);
        ok=tensor&&fg_vk_tensor_bytes(tensor)==bytes;
        if(i<2u){
            ok=ok&&fg_vk_tensor_get_format(tensor)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED&&
                fg_cook_q8_0_rows(source+i*stride,expected,bytes,INPUT,OUTPUT)&&
                !memcmp(fg_vk_tensor_map(tensor),expected,(size_t)bytes);
        }else{
            ok=ok&&fg_vk_tensor_get_format(tensor)==FG_VK_TENSOR_FORMAT_DEFAULT&&
                !memcmp(fg_vk_tensor_map(tensor),source+i*stride,(size_t)bytes);
        }
    }
    ok=ok&&fg_model_weight_bytes(model)==COUNT*stride;
    fg_model_close(model);unlink(path);rmdir(directory);
    free(manifest);free(expected);free(source);
    if(!ok)fprintf(stderr,"common projection cooking: %s\n",error.message);
    return status==FG_ERR_UNAVAILABLE?77:ok?0:1;
}

int main(void){
    int cooked=test_cooked_expert_load();if(cooked)return cooked;
    cooked=test_common_projection_load();if(cooked)return cooked;
    char directory[128],path[160];snprintf(directory,sizeof(directory),"/tmp/fg-model-load-%ld",(long)getpid());snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);if(mkdir(directory,0700)!=0){perror("mkdir");return 1;}uint8_t *bytes=aligned_alloc(FG_ALIGNMENT,FG_ALIGNMENT);if(!bytes)return 1;for(uint32_t i=0;i<FG_ALIGNMENT;i++)bytes[i]=(uint8_t)(i*29u+7u);FILE *stream=fopen(path,"wb");if(!stream||fwrite(bytes,1,FG_ALIGNMENT,stream)!=FG_ALIGNMENT||fwrite(bytes,1,FG_ALIGNMENT,stream)!=FG_ALIGNMENT||fclose(stream)!=0){perror("write rank artifact");free(bytes);return 1;}
    fg_manifest *manifest=malloc(sizeof(*manifest));if(!manifest){free(bytes);return 1;}fg_manifest_init(manifest);fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"probe.weight");record.bytes=FG_ALIGNMENT;record.ggml_type=8u;record.dims=2;record.shape[0]=32u;record.shape[1]=32u;record.rank=0;record.layer=UINT16_MAX;record.expert=UINT16_MAX;record.kind=FG_TENSOR_COMMON;record.layout=FG_TENSOR_LAYOUT_Q8_0_COOKED;fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,bytes,FG_ALIGNMENT);fg_sha256_final(&hash,record.sha256);fg_error error={0};if(fg_manifest_add_tensor(manifest,&record,&error)!=FG_OK){fprintf(stderr,"manifest: %s\n",error.message);free(manifest);free(bytes);return 1;}fg_tensor_record generic=record;snprintf(generic.name,sizeof(generic.name),"generic.weight");generic.offset=FG_ALIGNMENT;generic.layout=FG_TENSOR_LAYOUT_GGML;if(fg_manifest_add_tensor(manifest,&generic,&error)!=FG_OK){fprintf(stderr,"manifest: %s\n",error.message);free(manifest);free(bytes);return 1;}
    fg_model *model=NULL;fg_status status=fg_model_open(&model,manifest,directory,0,&error);if(status==FG_ERR_UNAVAILABLE){fprintf(stderr,"SKIP direct Vulkan/io_uring model load: %s\n",error.message);unlink(path);rmdir(directory);free(manifest);free(bytes);return 77;}if(status!=FG_OK){fprintf(stderr,"model load: %s\n",error.message);unlink(path);rmdir(directory);free(manifest);free(bytes);return 1;}fg_vk_tensor *tensor=fg_model_tensor(model,"probe.weight"),*generic_tensor=fg_model_tensor(model,"generic.weight");const fg_tensor_record *found=fg_model_tensor_record(model,"probe.weight");uint8_t check[64];int ok=tensor&&generic_tensor&&found==&manifest->tensors[0]&&fg_vk_tensor_get_format(tensor)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED&&fg_vk_tensor_get_format(generic_tensor)==FG_VK_TENSOR_FORMAT_DEFAULT&&!fg_model_tensor(model,"probe.weigh")&&!fg_model_tensor(model,"missing.weight")&&!fg_model_tensor_record(model,"missing.weight")&&fg_vk_tensor_read(tensor,0,check,sizeof(check),&error)==FG_OK&&memcmp(check,bytes,sizeof(check))==0&&fg_model_weight_bytes(model)==2u*FG_ALIGNMENT;fg_model_close(model);unlink(path);rmdir(directory);free(manifest);free(bytes);if(!ok){fprintf(stderr,"direct model load parity failed: %s\n",error.message);return 1;}puts("Flash Gordon fixed-buffer O_DIRECT to Vulkan arena: PASS");return 0;
}
