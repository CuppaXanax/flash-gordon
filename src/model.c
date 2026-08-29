#include "fg_model.h"
#include "fg_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

/* Expert-parallel model loading: loads ALL shared weights from ALL rank files,
   plus this rank's expert weights, into a single combined arena.  Every rank
   can then process all 48 layers locally — only MoE dispatch goes to the network. */
fg_status fg_model_open_replicated(fg_model **out,const fg_manifest *manifest,const char *pack_dir,uint32_t rank,fg_error *err){
    if(!out||!manifest||!pack_dir||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"invalid replicated model open arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    /* Phase 1: compute combined arena layout.  Walk all tensors and assign new
       offsets in the combined arena, keeping shared tensors from every rank and
       expert tensors only from this rank. */
    uint64_t cursor=0;
    uint64_t *remap=calloc(manifest->tensor_count,sizeof(*remap));
    bool *included=calloc(manifest->tensor_count,sizeof(*included));
    if(!remap||!included){free(included);free(remap);fg_error_set(err,FG_ERR_OOM,"allocate replicated remap table");return FG_ERR_OOM;}
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        const fg_tensor_record *t=&manifest->tensors[i];
        bool is_shared=(t->kind==FG_TENSOR_COMMON);
        bool is_my_expert=(t->kind==FG_TENSOR_ROUTED_EXPERT&&t->rank==rank);
        if(is_shared||is_my_expert){
            remap[i]=fg_align_up_u64(cursor,FG_ALIGNMENT);
            cursor=remap[i]+fg_align_up_u64(t->bytes,FG_ALIGNMENT);
            included[i]=true;
        }
    }
    if(!cursor){free(included);free(remap);fg_error_set(err,FG_ERR_MISMATCH,"replicated layout produced an empty arena");return FG_ERR_MISMATCH;}
    /* Phase 2: allocate model + arena */
    fg_model *model=calloc(1,sizeof(*model));if(!model){free(included);free(remap);fg_error_set(err,FG_ERR_OOM,"allocate replicated model");return FG_ERR_OOM;}
    model->manifest=manifest;model->rank=rank;model->weight_bytes=cursor;
    model->tensor=calloc(manifest->tensor_count,sizeof(*model->tensor));
    if(!model->tensor){fg_model_close(model);free(included);free(remap);fg_error_set(err,FG_ERR_OOM,"allocate replicated tensor index");return FG_ERR_OOM;}
    fg_status status=fg_vk_open(&model->vk,err);
    if(status==FG_OK)status=fg_vk_tensor_create(model->vk,cursor,&model->arena,err);
    void *mapped=status==FG_OK?fg_vk_tensor_map(model->arena):NULL;
    if(status==FG_OK&&(!mapped||!fg_is_aligned_u64((uintptr_t)mapped,FG_ALIGNMENT))){fg_error_set(err,FG_ERR_UNAVAILABLE,"replicated arena is not 4 KiB aligned");status=FG_ERR_UNAVAILABLE;}
    /* Phase 3: load weights — read each tensor directly from its source rank file
       using a small bounce buffer instead of loading entire rank files. */
    const uint32_t chunk=8u*1024u*1024u;
    void *bounce=status==FG_OK?aligned_alloc(FG_ALIGNMENT,chunk):NULL;
    if(status==FG_OK&&!bounce){fg_error_set(err,FG_ERR_OOM,"allocate replicated bounce chunk");status=FG_ERR_OOM;}
    for(uint32_t src=0;status==FG_OK&&src<FG_RANK_COUNT;src++){
        bool need_src=false;
        for(uint32_t i=0;!need_src&&i<manifest->tensor_count;i++)
            if(included[i]&&manifest->tensors[i].rank==src)need_src=true;
        if(!need_src)continue;
        char path[1024];if(snprintf(path,sizeof(path),"%s/rank-%02u.fgw",pack_dir,src)>=(int)sizeof(path)){fg_error_set(err,FG_ERR_LIMIT,"rank file path overflow");status=FG_ERR_LIMIT;break;}
        int fd=open(path,O_RDONLY|O_CLOEXEC);if(fd<0){fg_error_set(err,FG_ERR_IO,"open rank %u weights: %s",src,strerror(errno));status=FG_ERR_IO;break;}
        for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++){
            if(!included[i]||manifest->tensors[i].rank!=src)continue;
            const fg_tensor_record *t=&manifest->tensors[i];
            /* Copy tensor from rank file to arena via small bounce */
            for(uint64_t off=0;status==FG_OK&&off<t->bytes;off+=chunk){
                uint32_t n=(uint32_t)((t->bytes-off)>chunk?chunk:t->bytes-off);
                ssize_t got=pread(fd,bounce,n,(off_t)(t->offset+off));
                if(got!=(ssize_t)n){fg_error_set(err,FG_ERR_IO,"pread rank %u tensor %.48s: %s",src,t->name,got<0?strerror(errno):"short read");status=FG_ERR_IO;}
                else memcpy((uint8_t *)mapped+remap[i]+off,bounce,n);
            }
        }
        close(fd);
    }
    free(bounce);
    /* Phase 4: create tensor views at remapped offsets */
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++){
        if(!included[i])continue;
        status=fg_vk_tensor_view(model->arena,remap[i],manifest->tensors[i].bytes,&model->tensor[i],err);
    }
    free(included);free(remap);
    if(status!=FG_OK){fg_model_close(model);return status;}
    fprintf(stderr,"[rank %u] replicated model: %.3f GiB (%u tensors from %u ranks)\n",rank,(double)cursor/(1024.0*1024.0*1024.0),manifest->tensor_count,FG_RANK_COUNT);
    *out=model;return FG_OK;
}
fg_vk_context *fg_model_vk(fg_model *model){return model?model->vk:NULL;}
fg_vk_tensor *fg_model_tensor(fg_model *model,const char *name){if(!model||!name)return NULL;for(uint32_t i=0;i<model->manifest->tensor_count;i++)if(model->tensor[i]&&strcmp(model->manifest->tensors[i].name,name)==0)return model->tensor[i];return NULL;}
const fg_tensor_record *fg_model_tensor_record(const fg_model *model,const char *name){if(!model||!name)return NULL;for(uint32_t i=0;i<model->manifest->tensor_count;i++)if(model->tensor[i]&&strcmp(model->manifest->tensors[i].name,name)==0)return &model->manifest->tensors[i];return NULL;}
const fg_manifest *fg_model_manifest(const fg_model *model){return model?model->manifest:NULL;}
uint64_t fg_model_weight_bytes(const fg_model *model){return model?model->weight_bytes:0;}
uint32_t fg_model_rank(const fg_model *model){return model?model->rank:UINT32_MAX;}
