#include "fg_model.h"
#include "fg_loader.h"
#include "fg_quant.h"

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
    uint32_t *tensor_lookup;
    uint32_t tensor_lookup_capacity;
    const fg_manifest *manifest;
    uint64_t weight_bytes;
    uint32_t rank;
};

static uint64_t rank_high_water(const fg_manifest *manifest,uint32_t rank){
    uint64_t high=0;
    for(uint32_t i=0;i<manifest->tensor_count;i++)if(manifest->tensors[i].rank==rank){uint64_t end=fg_align_up_u64(manifest->tensors[i].offset+manifest->tensors[i].bytes,FG_ALIGNMENT);if(end>high)high=end;}
    return high;
}

static uint64_t tensor_name_hash(const char *name){uint64_t hash=UINT64_C(1469598103934665603);for(const uint8_t *p=(const uint8_t *)name;*p;p++){hash^=*p;hash*=UINT64_C(1099511628211);}return hash;}
static bool cook_experts_on_load(void){const char *value=getenv("FG_COOK_EXPERTS_ON_LOAD");return value&&*value&&strcmp(value,"0")!=0;}
static bool hc_down_tensor(const fg_tensor_record *record){return record->kind==FG_TENSOR_COMMON&&record->ggml_type==8u&&record->dims==2u&&record->shape[0]==10240u&&record->shape[1]==320u&&(strstr(record->name,".hc_attn_down.weight")||strstr(record->name,".hc_ffn_down.weight"));}
static bool narrow_common_q8(const fg_tensor_record *record){return record->kind==FG_TENSOR_COMMON&&record->ggml_type==8u&&record->dims==2u&&record->shape[0]==2560u&&(record->shape[1]==640u||record->shape[1]==512u);}
static fg_tensor_layout runtime_layout(const fg_tensor_record *record,bool cook){fg_tensor_layout layout=(fg_tensor_layout)record->layout;if(!cook||layout!=FG_TENSOR_LAYOUT_GGML||record->shape[0]>UINT32_MAX||record->shape[1]>UINT32_MAX)return layout;uint32_t input=(uint32_t)record->shape[0],output=(uint32_t)record->shape[1];if((hc_down_tensor(record)||narrow_common_q8(record))&&fg_q8_0_cooked_matrix_bytes(input,output)==record->bytes)return FG_TENSOR_LAYOUT_Q8_0_COOKED;if(record->kind!=FG_TENSOR_ROUTED_EXPERT||record->dims!=3u)return layout;if((record->ggml_type==12u||record->ggml_type==13u)&&fg_k_quant_cooked_matrix_bytes(input,output,record->ggml_type))return FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED;if(record->ggml_type==7u&&fg_q5_1_cooked_matrix_bytes(input,output))return FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED;return layout;}
static fg_vk_tensor_format tensor_format(fg_tensor_layout layout){if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED)return FG_VK_TENSOR_FORMAT_Q8_0_COOKED;if(layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED)return FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED;if(layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED)return FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED;return FG_VK_TENSOR_FORMAT_DEFAULT;}
static fg_status cook_expert_tensor(const fg_tensor_record *record,void *data,fg_tensor_layout layout,fg_error *err){uint64_t experts=record->shape[2];if(!experts||record->bytes%experts){fg_error_set(err,FG_ERR_FORMAT,"expert tensor %.96s has invalid local shape",record->name);return FG_ERR_FORMAT;}uint64_t matrix_bytes=record->bytes/experts;if(matrix_bytes>SIZE_MAX){fg_error_set(err,FG_ERR_LIMIT,"expert tensor %.96s matrix is too large",record->name);return FG_ERR_LIMIT;}uint8_t *temporary=malloc((size_t)matrix_bytes);if(!temporary){fg_error_set(err,FG_ERR_OOM,"allocate expert cooking buffer");return FG_ERR_OOM;}for(uint64_t expert=0;expert<experts;expert++){uint8_t *matrix=(uint8_t *)data+expert*matrix_bytes;bool converted=layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED?fg_cook_k_quant_rows(matrix,temporary,matrix_bytes,(uint32_t)record->shape[0],(uint32_t)record->shape[1],record->ggml_type):layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED?fg_cook_q5_1_rows(matrix,temporary,matrix_bytes,(uint32_t)record->shape[0],(uint32_t)record->shape[1]):false;if(!converted){free(temporary);fg_error_set(err,FG_ERR_FORMAT,"cannot cook expert tensor %.96s",record->name);return FG_ERR_FORMAT;}memcpy(matrix,temporary,(size_t)matrix_bytes);}free(temporary);return FG_OK;}
static fg_status cook_runtime_tensor(const fg_tensor_record *record,void *data,fg_tensor_layout layout,fg_error *err){if(layout!=FG_TENSOR_LAYOUT_Q8_0_COOKED)return cook_expert_tensor(record,data,layout,err);if(record->bytes>SIZE_MAX){fg_error_set(err,FG_ERR_LIMIT,"cooked tensor %.96s is too large",record->name);return FG_ERR_LIMIT;}uint8_t *temporary=malloc((size_t)record->bytes);if(!temporary){fg_error_set(err,FG_ERR_OOM,"allocate Q8 cooking buffer");return FG_ERR_OOM;}bool converted=fg_cook_q8_0_rows(data,temporary,record->bytes,(uint32_t)record->shape[0],(uint32_t)record->shape[1]);if(converted)memcpy(data,temporary,(size_t)record->bytes);free(temporary);if(!converted){fg_error_set(err,FG_ERR_FORMAT,"cannot cook Q8 tensor %.96s",record->name);return FG_ERR_FORMAT;}return FG_OK;}
static fg_status cook_rank_experts(const fg_manifest *manifest,uint32_t rank,void *arena,uint64_t arena_bytes,bool cook,fg_error *err){if(!cook)return FG_OK;uint32_t tensors=0;uint64_t bytes=0;for(uint32_t i=0;i<manifest->tensor_count;i++){const fg_tensor_record *record=&manifest->tensors[i];fg_tensor_layout layout=runtime_layout(record,true);if(record->rank!=rank||layout==(fg_tensor_layout)record->layout)continue;if(record->offset>arena_bytes||record->bytes>arena_bytes-record->offset){fg_error_set(err,FG_ERR_FORMAT,"tensor %.96s lies outside rank arena",record->name);return FG_ERR_FORMAT;}fg_status status=cook_runtime_tensor(record,(uint8_t *)arena+record->offset,layout,err);if(status!=FG_OK)return status;tensors++;bytes+=record->bytes;}fprintf(stderr,"[rank %u] cooked %u runtime tensors (%.3f GiB) in the verified Vulkan arena\n",rank,tensors,(double)bytes/(1ull<<30));return FG_OK;}

static fg_status build_tensor_lookup(fg_model *model,fg_error *err){
    uint32_t capacity=1u;while(capacity<model->manifest->tensor_count*2u)capacity<<=1u;model->tensor_lookup=calloc(capacity,sizeof(*model->tensor_lookup));if(!model->tensor_lookup){fg_error_set(err,FG_ERR_OOM,"allocate model tensor lookup");return FG_ERR_OOM;}model->tensor_lookup_capacity=capacity;
    for(uint32_t i=0;i<model->manifest->tensor_count;i++)if(model->tensor[i]){const char *name=model->manifest->tensors[i].name;uint32_t slot=(uint32_t)tensor_name_hash(name)&(capacity-1u);while(model->tensor_lookup[slot]){uint32_t existing=model->tensor_lookup[slot]-1u;if(strcmp(model->manifest->tensors[existing].name,name)==0)break;slot=(slot+1u)&(capacity-1u);}if(!model->tensor_lookup[slot])model->tensor_lookup[slot]=i+1u;}
    return FG_OK;
}

static uint32_t find_tensor(const fg_model *model,const char *name){if(!model||!name||!model->tensor_lookup_capacity)return UINT32_MAX;uint32_t slot=(uint32_t)tensor_name_hash(name)&(model->tensor_lookup_capacity-1u);for(uint32_t probes=0;probes<model->tensor_lookup_capacity;probes++){uint32_t entry=model->tensor_lookup[slot];if(!entry)return UINT32_MAX;uint32_t index=entry-1u;if(strcmp(model->manifest->tensors[index].name,name)==0)return index;slot=(slot+1u)&(model->tensor_lookup_capacity-1u);}return UINT32_MAX;}

fg_status fg_model_open(fg_model **out,const fg_manifest *manifest,const char *pack_dir,uint32_t rank,fg_error *err){
    if(!out||!manifest||!pack_dir||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"invalid model open arguments");return FG_ERR_ARGUMENT;}*out=NULL;uint64_t bytes=rank_high_water(manifest,rank);if(bytes==0||bytes>manifest->persistent_cap_bytes){fg_error_set(err,FG_ERR_LIMIT,"rank %u weight arena is invalid or exceeds persistent cap",rank);return FG_ERR_LIMIT;}
    fg_model *model=calloc(1,sizeof(*model));if(!model){fg_error_set(err,FG_ERR_OOM,"allocate model binding");return FG_ERR_OOM;}model->manifest=manifest;model->rank=rank;model->weight_bytes=bytes;model->tensor=calloc(manifest->tensor_count,sizeof(*model->tensor));if(!model->tensor){fg_model_close(model);fg_error_set(err,FG_ERR_OOM,"allocate model tensor index");return FG_ERR_OOM;}
    fg_status status=fg_vk_open(&model->vk,err);if(status==FG_OK)status=fg_vk_tensor_create(model->vk,bytes,&model->arena,err);void *mapped=status==FG_OK?fg_vk_tensor_map(model->arena):NULL;if(status==FG_OK&&(!mapped||!fg_is_aligned_u64((uintptr_t)mapped,FG_ALIGNMENT))){fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan weight arena is not 4 KiB aligned for O_DIRECT");status=FG_ERR_UNAVAILABLE;}
    bool cook=cook_experts_on_load();if(status==FG_OK)status=fg_load_rank_weights(manifest,pack_dir,rank,mapped,bytes,err);if(status==FG_OK)status=cook_rank_experts(manifest,rank,mapped,bytes,cook,err);
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)if(manifest->tensors[i].rank==rank){status=fg_vk_tensor_view(model->arena,manifest->tensors[i].offset,manifest->tensors[i].bytes,&model->tensor[i],err);if(status==FG_OK)fg_vk_tensor_set_format(model->tensor[i],tensor_format(runtime_layout(&manifest->tensors[i],cook)));}
    if(status==FG_OK)status=build_tensor_lookup(model,err);
    if(status!=FG_OK){fg_model_close(model);return status;}*out=model;return FG_OK;
}

void fg_model_close(fg_model *model){if(!model)return;if(model->tensor){for(uint32_t i=0;i<model->manifest->tensor_count;i++)fg_vk_tensor_destroy(model->tensor[i]);}free(model->tensor_lookup);free(model->tensor);fg_vk_tensor_destroy(model->arena);fg_vk_close(model->vk);free(model);}

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
    bool cook=cook_experts_on_load();
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++){
        if(!included[i])continue;
        const fg_tensor_record *record=&manifest->tensors[i];fg_tensor_layout layout=runtime_layout(record,cook);
        if(layout!=(fg_tensor_layout)record->layout)status=cook_runtime_tensor(record,(uint8_t *)mapped+remap[i],layout,err);
    }
    /* Phase 4: create tensor views at remapped offsets */
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++){
        if(!included[i])continue;
        status=fg_vk_tensor_view(model->arena,remap[i],manifest->tensors[i].bytes,&model->tensor[i],err);
        if(status==FG_OK)fg_vk_tensor_set_format(model->tensor[i],tensor_format(runtime_layout(&manifest->tensors[i],cook)));
    }
    if(status==FG_OK)status=build_tensor_lookup(model,err);
    free(included);free(remap);
    if(status!=FG_OK){fg_model_close(model);return status;}
    fprintf(stderr,"[rank %u] replicated model: %.3f GiB (%u tensors from %u ranks)\n",rank,(double)cursor/(1024.0*1024.0*1024.0),manifest->tensor_count,FG_RANK_COUNT);
    *out=model;return FG_OK;
}
fg_vk_context *fg_model_vk(fg_model *model){return model?model->vk:NULL;}
fg_vk_tensor *fg_model_tensor(fg_model *model,const char *name){uint32_t index=find_tensor(model,name);return index==UINT32_MAX?NULL:model->tensor[index];}
const fg_tensor_record *fg_model_tensor_record(const fg_model *model,const char *name){uint32_t index=find_tensor(model,name);return index==UINT32_MAX?NULL:&model->manifest->tensors[index];}
const fg_manifest *fg_model_manifest(const fg_model *model){return model?model->manifest:NULL;}
uint64_t fg_model_weight_bytes(const fg_model *model){return model?model->weight_bytes:0;}
uint32_t fg_model_rank(const fg_model *model){return model?model->rank:UINT32_MAX;}
