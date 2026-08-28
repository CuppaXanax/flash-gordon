#include "fg_model.h"
#include "fg_loader.h"

#include <stdlib.h>
#include <string.h>

struct fg_model {
    fg_vk_context *vk;
    fg_vk_tensor *arena;
    fg_vk_tensor **tensor;
    const fg_manifest *manifest;
    uint64_t weight_bytes;
    uint32_t rank;
};

static uint64_t rank_high_water(const fg_manifest *manifest,uint32_t rank){
    uint64_t high=0;
    for(uint32_t i=0;i<manifest->tensor_count;i++)if(manifest->tensors[i].rank==rank){uint64_t end=fg_align_up_u64(manifest->tensors[i].offset+manifest->tensors[i].bytes,FG_ALIGNMENT);if(end>high)high=end;}
    return high;
}

fg_status fg_model_open(fg_model **out,const fg_manifest *manifest,const char *pack_dir,uint32_t rank,fg_error *err){
    if(!out||!manifest||!pack_dir||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"invalid model open arguments");return FG_ERR_ARGUMENT;}*out=NULL;uint64_t bytes=rank_high_water(manifest,rank);if(bytes==0||bytes>manifest->persistent_cap_bytes){fg_error_set(err,FG_ERR_LIMIT,"rank %u weight arena is invalid or exceeds persistent cap",rank);return FG_ERR_LIMIT;}
    fg_model *model=calloc(1,sizeof(*model));if(!model){fg_error_set(err,FG_ERR_OOM,"allocate model binding");return FG_ERR_OOM;}model->manifest=manifest;model->rank=rank;model->weight_bytes=bytes;model->tensor=calloc(manifest->tensor_count,sizeof(*model->tensor));if(!model->tensor){fg_model_close(model);fg_error_set(err,FG_ERR_OOM,"allocate model tensor index");return FG_ERR_OOM;}
    fg_status status=fg_vk_open(&model->vk,err);if(status==FG_OK)status=fg_vk_tensor_create(model->vk,bytes,&model->arena,err);void *mapped=status==FG_OK?fg_vk_tensor_map(model->arena):NULL;if(status==FG_OK&&(!mapped||!fg_is_aligned_u64((uintptr_t)mapped,FG_ALIGNMENT))){fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan weight arena is not 4 KiB aligned for O_DIRECT");status=FG_ERR_UNAVAILABLE;}
    if(status==FG_OK)status=fg_load_rank_weights(manifest,pack_dir,rank,mapped,bytes,err);
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)if(manifest->tensors[i].rank==rank)status=fg_vk_tensor_view(model->arena,manifest->tensors[i].offset,manifest->tensors[i].bytes,&model->tensor[i],err);
    if(status!=FG_OK){fg_model_close(model);return status;}*out=model;return FG_OK;
}

void fg_model_close(fg_model *model){if(!model)return;if(model->tensor){for(uint32_t i=0;i<model->manifest->tensor_count;i++)fg_vk_tensor_destroy(model->tensor[i]);}free(model->tensor);fg_vk_tensor_destroy(model->arena);fg_vk_close(model->vk);free(model);}
fg_vk_context *fg_model_vk(fg_model *model){return model?model->vk:NULL;}
fg_vk_tensor *fg_model_tensor(fg_model *model,const char *name){if(!model||!name)return NULL;for(uint32_t i=0;i<model->manifest->tensor_count;i++)if(model->tensor[i]&&strcmp(model->manifest->tensors[i].name,name)==0)return model->tensor[i];return NULL;}
const fg_tensor_record *fg_model_tensor_record(const fg_model *model,const char *name){if(!model||!name)return NULL;for(uint32_t i=0;i<model->manifest->tensor_count;i++)if(model->tensor[i]&&strcmp(model->manifest->tensors[i].name,name)==0)return &model->manifest->tensors[i];return NULL;}
const fg_manifest *fg_model_manifest(const fg_model *model){return model?model->manifest:NULL;}
uint64_t fg_model_weight_bytes(const fg_model *model){return model?model->weight_bytes:0;}
uint32_t fg_model_rank(const fg_model *model){return model?model->rank:UINT32_MAX;}
