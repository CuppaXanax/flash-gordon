#include "fg_stage.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(expression) do { \
    if(!(expression)){ \
        fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#expression); \
        failures++; \
    } \
} while(0)

struct fg_vk_context {
    uint32_t batch_depth;
    uint64_t submissions;
};
struct fg_vk_tensor {
    uint8_t *data;
    uint64_t bytes,written;
    fg_vk_tensor_format format;
};
struct fg_vk_expert_graph {
    fg_vk_tensor *tiles,*gates,*reduced;
    uint32_t weight_experts;
};
struct fg_model {
    fg_manifest *manifest;
    fg_vk_context vk;
    fg_vk_tensor weights[3];
    fg_tensor_record records[3];
    uint32_t rank;
    bool bad_tensor_name;
};
struct fg_owner_executor {
    fg_model *model;
    fg_vk_tensor *input,*ping[2];
    uint32_t next;
};
struct fg_output_executor { fg_model *model; };

static uint32_t graph_weight_experts;
static uint64_t graph_tiles_bytes,graph_reduced_bytes;
static uint32_t owner_calls,owner_resets,qsa_opens,ngram_calls;
static uint32_t output_creates,output_destroys,output_calls,output_history_resets;
static uint32_t fail_layer=UINT32_MAX;
static uint32_t abort_calls;
static uint32_t decode_schedule_calls,fast_moe_calls,grouped_moe_calls;
static uint32_t fast_down_calls,fast_reduce_calls,shared_combine_calls;
static uint32_t q5_down_calls,q8_down_calls,last_down_stride,last_down_used;
static fg_status abort_status=FG_OK,owner_reset_status=FG_OK;
static bool profile_active,fail_moe;
static bool decode_dispatch_ok,prefill_dispatch_ok;

static void manifest_pipeline(fg_manifest *manifest){
    memset(manifest,0,sizeof(*manifest));
    manifest->format_version=FG_MANIFEST_FORMAT_VERSION;
    manifest->protocol_version=FG_PIPELINE_PROTOCOL_VERSION;
    manifest->execution_mode=FG_EXECUTION_PIPELINE;
    manifest->stage_count=FG_PIPELINE_STAGE_COUNT;
    manifest->prefill_microbatch=2u;
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        manifest->stage_ranks[stage]=(uint8_t)stage;
        manifest->layer_offsets[stage]=stage*FG_PIPELINE_DEFAULT_LAYERS_PER_STAGE;
        for(uint32_t layer=manifest->layer_offsets[stage];
            layer<(stage+1u)*FG_PIPELINE_DEFAULT_LAYERS_PER_STAGE;layer++){
            manifest->layer_owner[layer]=(uint8_t)stage;
            for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
                manifest->expert_rank[layer][expert]=(uint16_t)stage;
        }
    }
    manifest->layer_offsets[FG_PIPELINE_STAGE_COUNT]=FG_LAYER_COUNT;
}

static void manifest_ep(fg_manifest *manifest,uint32_t rank){
    memset(manifest,0,sizeof(*manifest));
    manifest->format_version=FG_MANIFEST_FORMAT_VERSION;
    manifest->protocol_version=FG_PROTOCOL_VERSION;
    manifest->execution_mode=FG_EXECUTION_EXPERT_PARALLEL;
    manifest->prefill_microbatch=2u;
    manifest->layer_owner[0]=0u;
    for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
        manifest->expert_rank[0][expert]=expert<FG_EXPERTS_PER_RANK?
            (uint16_t)rank:(uint16_t)((rank+1u)%FG_RANK_COUNT);
}

static void model_init(fg_model *model,fg_manifest *manifest,uint32_t rank){
    memset(model,0,sizeof(*model));
    model->manifest=manifest;model->rank=rank;
    for(uint32_t family=0;family<3u;family++){
        uint32_t count=manifest->execution_mode==FG_EXECUTION_PIPELINE?
            FG_EXPERT_COUNT:FG_EXPERTS_PER_RANK;
        model->weights[family].bytes=64u;
        model->weights[family].data=calloc(1,64u);
        model->weights[family].format=family==2u?
            FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED:
            FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED;
        model->records[family].bytes=(uint64_t)count*64u;
        model->records[family].ggml_type=family==2u?7u:12u;
        model->records[family].dims=3u;
        model->records[family].shape[2]=count;
    }
}

static void model_destroy(fg_model *model){
    for(uint32_t i=0;i<3u;i++)free(model->weights[i].data);
}

static int tensor_family(const char *name){
    if(strstr(name,"ffn_gate_exps.weight"))return 0;
    if(strstr(name,"ffn_up_exps.weight"))return 1;
    if(strstr(name,"ffn_down_exps.weight"))return 2;
    return -1;
}

static bool tensor_available(fg_model *model,const char *name){
    unsigned layer=UINT32_MAX,rank=UINT32_MAX;
    bool pipeline=model->manifest->execution_mode==FG_EXECUTION_PIPELINE;
    int parsed=pipeline?
        sscanf(name,"blk.%u.%*[^.].weight",&layer):
        sscanf(name,"blk.%u.%*[^.].weight.rank%u",&layer,&rank);
    if(parsed!=(pipeline?1:2)||(pipeline&&strstr(name,".rank"))||
       (!pipeline&&rank!=model->rank)){
        model->bad_tensor_name=true;
        return false;
    }
    if(layer>=FG_LAYER_COUNT)return false;
    if(pipeline)return model->manifest->layer_owner[layer]==model->rank;
    return layer==0u&&fg_expert_local_count(
        model->manifest,layer,model->rank)==FG_EXPERTS_PER_RANK;
}

fg_vk_context *fg_model_vk(fg_model *model){return &model->vk;}
const fg_manifest *fg_model_manifest(const fg_model *model){return model->manifest;}
uint32_t fg_model_rank(const fg_model *model){return model->rank;}
fg_vk_tensor *fg_model_tensor(fg_model *model,const char *name){
    int family=tensor_family(name);
    return family>=0&&tensor_available(model,name)?&model->weights[family]:NULL;
}
const fg_tensor_record *fg_model_tensor_record(const fg_model *model,
                                                const char *name){
    int family=tensor_family(name);
    return family>=0&&tensor_available((fg_model *)model,name)?
        &model->records[family]:NULL;
}

fg_status fg_vk_tensor_create(fg_vk_context *context,uint64_t bytes,
                              fg_vk_tensor **out,fg_error *err){
    (void)context;
    fg_vk_tensor *tensor=calloc(1,sizeof(*tensor));
    if(!tensor){fg_error_set(err,FG_ERR_OOM,"fake tensor");return FG_ERR_OOM;}
    tensor->data=calloc(1,(size_t)bytes);
    if(!tensor->data){free(tensor);fg_error_set(err,FG_ERR_OOM,"fake tensor data");return FG_ERR_OOM;}
    tensor->bytes=bytes;*out=tensor;return FG_OK;
}
void fg_vk_tensor_destroy(fg_vk_tensor *tensor){
    if(!tensor)return;
    free(tensor->data);free(tensor);
}
uint64_t fg_vk_tensor_bytes(const fg_vk_tensor *tensor){
    return tensor?tensor->bytes:0u;
}
void *fg_vk_tensor_map(fg_vk_tensor *tensor){return tensor?tensor->data:NULL;}
const void *fg_vk_tensor_const_map(const fg_vk_tensor *tensor){
    return tensor?tensor->data:NULL;
}
fg_vk_tensor_format fg_vk_tensor_get_format(const fg_vk_tensor *tensor){
    return tensor?tensor->format:FG_VK_TENSOR_FORMAT_DEFAULT;
}
void fg_vk_tensor_set_format(fg_vk_tensor *tensor,fg_vk_tensor_format format){
    if(tensor)tensor->format=format;
}
fg_status fg_vk_tensor_write(fg_vk_tensor *tensor,uint64_t offset,
                             const void *data,uint64_t bytes,fg_error *err){
    if(!tensor||!data||offset>tensor->bytes||bytes>tensor->bytes-offset){
        fg_error_set(err,FG_ERR_LIMIT,"fake tensor write");return FG_ERR_LIMIT;
    }
    memcpy(tensor->data+offset,data,(size_t)bytes);
    tensor->written=offset+bytes;return FG_OK;
}
fg_status fg_vk_tensor_read(const fg_vk_tensor *tensor,uint64_t offset,
                            void *data,uint64_t bytes,fg_error *err){
    if(!tensor||!data||offset>tensor->bytes||bytes>tensor->bytes-offset){
        fg_error_set(err,FG_ERR_LIMIT,"fake tensor read");return FG_ERR_LIMIT;
    }
    memcpy(data,tensor->data+offset,(size_t)bytes);return FG_OK;
}

bool fg_vk_profile_active(const fg_vk_context *context){(void)context;return profile_active;}
fg_status fg_vk_profile_begin(fg_vk_context *context,fg_error *err){
    (void)context;(void)err;return FG_OK;
}
fg_status fg_vk_profile_end(fg_vk_context *context,fg_vk_profile *profile,
                            fg_error *err){
    (void)context;(void)err;memset(profile,0,sizeof(*profile));return FG_OK;
}
fg_status fg_vk_profile_set_scope(fg_vk_context *context,const char *scope,
                                  fg_error *err){
    (void)context;(void)scope;(void)err;return FG_OK;
}
fg_status fg_vk_begin(fg_vk_context *context,fg_error *err){
    (void)err;context->batch_depth++;return FG_OK;
}
fg_status fg_vk_end(fg_vk_context *context,fg_error *err){
    if(!context->batch_depth){
        fg_error_set(err,FG_ERR_ARGUMENT,"synthetic batch is not active");
        return FG_ERR_ARGUMENT;
    }
    context->batch_depth--;
    if(!context->batch_depth)context->submissions++;
    return FG_OK;
}
bool fg_vk_batch_active(const fg_vk_context *context){
    return context->batch_depth>0u;
}
fg_status fg_vk_abort(fg_vk_context *context,fg_error *err){
    abort_calls++;context->batch_depth=0u;
    if(abort_status!=FG_OK){
        fg_error_set(err,abort_status,"synthetic Vulkan abort failure");
        return abort_status;
    }
    return FG_OK;
}

fg_status fg_vk_expert_graph_create(
    fg_vk_context *context,fg_vk_expert_graph **out,fg_vk_tensor *activation,
    fg_vk_tensor *tiles,fg_vk_tensor *gates,fg_vk_tensor *gate,fg_vk_tensor *up,
    fg_vk_tensor *mid,fg_vk_tensor *down,fg_vk_tensor *reduced,
    const fg_vk_tensor *gate_weights,const fg_vk_tensor *up_weights,
    const fg_vk_tensor *down_weights,uint32_t gate_type,uint32_t up_type,
    uint32_t down_type,uint32_t hidden_width,uint32_t mid_width,
    uint32_t gate_stride,uint32_t up_stride,uint32_t down_stride,
    uint32_t weight_experts,uint32_t slots,fg_error *err){
    (void)context;(void)activation;(void)gate;(void)up;(void)mid;(void)down;
    (void)gate_weights;(void)up_weights;(void)down_weights;(void)gate_type;
    (void)up_type;(void)down_type;(void)hidden_width;(void)mid_width;
    (void)slots;(void)err;
    CHECK(gate_stride==64u&&up_stride==64u&&down_stride==64u);
    fg_vk_expert_graph *graph=calloc(1,sizeof(*graph));
    if(!graph)return FG_ERR_OOM;
    graph->tiles=tiles;graph->gates=gates;graph->reduced=reduced;
    graph->weight_experts=weight_experts;graph_weight_experts=weight_experts;
    graph_tiles_bytes=tiles->bytes;graph_reduced_bytes=reduced->bytes;
    *out=graph;return FG_OK;
}
void fg_vk_expert_graph_destroy(fg_vk_expert_graph *graph){free(graph);}
fg_status fg_vk_expert_graph_execute(fg_vk_expert_graph *graph,fg_error *err){
    (void)err;
    uint32_t selected=(uint32_t)(graph->gates->written/sizeof(float));
    const uint32_t *tiles=(const uint32_t *)graph->tiles->data;
    const float *gates=(const float *)graph->gates->data;
    float value=0.0f;
    for(uint32_t i=0;i<selected;i++)value+=gates[i]*(float)(tiles[i*9u]+1u);
    float *output=(float *)graph->reduced->data;
    for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)output[i]=value;
    return FG_OK;
}

fg_status fg_vk_moe_kquant(fg_vk_context *context,fg_vk_tensor *output,
    const fg_vk_tensor *weights,const fg_vk_tensor *activation,
    const fg_vk_tensor *tiles,uint32_t ggml_type,uint32_t output_width,
    uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,
    uint32_t routed_pairs,bool packed_weights,uint32_t tile_count,fg_error *err){
    (void)context;(void)output;(void)weights;(void)activation;(void)tiles;
    (void)ggml_type;(void)output_width;(void)input_width;(void)expert_stride;
    (void)used_experts;(void)routed_pairs;(void)packed_weights;(void)tile_count;
    fast_moe_calls++;
    if(fail_moe){
        fg_error_set(err,FG_ERR_IO,"synthetic stage-local expert failure");
        return FG_ERR_IO;
    }
    return FG_OK;
}
fg_status fg_vk_moe_kquant_cooked_grouped(fg_vk_context *context,
    fg_vk_tensor *output,const fg_vk_tensor *weights,
    const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,
    uint32_t output_width,uint32_t input_width,uint32_t expert_stride,
    uint32_t weight_experts,uint32_t tokens,fg_error *err){
    (void)context;(void)output;(void)weights;(void)activation;(void)tiles;
    (void)type;(void)output_width;(void)input_width;(void)expert_stride;
    (void)weight_experts;(void)tokens;(void)err;grouped_moe_calls++;return FG_OK;
}
fg_status fg_vk_moe_q5_1_down_cooked_grouped(fg_vk_context *context,
    fg_vk_tensor *output,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,
    uint32_t expert_stride,uint32_t weight_experts,uint32_t tokens,
    fg_error *err){
    (void)context;(void)output;(void)weights;(void)tiles;(void)input;
    (void)output_width;(void)input_width;(void)expert_stride;
    (void)weight_experts;(void)tokens;(void)err;return FG_OK;
}
fg_status fg_vk_moe_q8_0_down_grouped(fg_vk_context *context,
    fg_vk_tensor *output,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,
    uint32_t expert_stride,uint32_t weight_experts,uint32_t tokens,
    fg_error *err){
    return fg_vk_moe_q5_1_down_cooked_grouped(context,output,weights,tiles,
        input,output_width,input_width,expert_stride,weight_experts,tokens,err);
}
fg_status fg_vk_swiglu(fg_vk_context *context,fg_vk_tensor *output,
                       const fg_vk_tensor *gate,const fg_vk_tensor *up,
                       uint32_t values,fg_error *err){
    (void)context;(void)output;(void)gate;(void)up;(void)values;(void)err;
    return FG_OK;
}

static fg_status fake_down(fg_vk_tensor *output,const fg_vk_tensor *tiles,
                           uint32_t tile_count){
    const uint32_t *schedule=(const uint32_t *)tiles->data;
    float *values=(float *)output->data;
    for(uint32_t tile=0;tile<tile_count;tile++){
        uint32_t local=schedule[tile*9u],pair=schedule[tile*9u+1u];
        if(local==UINT32_MAX||pair==UINT32_MAX)continue;
        float value=(float)(local+1u)+(float)pair*0.01f;
        for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)
            values[(uint64_t)pair*FG_HIDDEN_SIZE+element]=value;
    }
    return FG_OK;
}
fg_status fg_vk_moe_q5_1_down_cooked_pairs(fg_vk_context *context,
    fg_vk_tensor *output,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,
    uint32_t expert_stride,uint32_t routed_pairs,bool packed_weights,
    uint32_t tile_count,fg_error *err){
    (void)context;(void)weights;(void)input;(void)output_width;(void)input_width;
    (void)packed_weights;(void)err;
    fast_down_calls++;q5_down_calls++;
    last_down_stride=expert_stride;last_down_used=routed_pairs;
    return fake_down(output,tiles,tile_count);
}
fg_status fg_vk_moe_q5_1_down(fg_vk_context *context,fg_vk_tensor *output,
    const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,
    uint32_t expert_stride,uint32_t used_experts,bool packed_weights,
    uint32_t tile_count,fg_error *err){
    return fg_vk_moe_q5_1_down_cooked_pairs(context,output,weights,tiles,input,
        output_width,input_width,expert_stride,used_experts,packed_weights,
        tile_count,err);
}
fg_status fg_vk_moe_q8_0_down(fg_vk_context *context,fg_vk_tensor *output,
    const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,
    uint32_t expert_stride,uint32_t used_experts,bool packed_weights,
    uint32_t tile_count,fg_error *err){
    (void)context;(void)weights;(void)input;(void)output_width;(void)input_width;
    (void)packed_weights;(void)err;
    fast_down_calls++;q8_down_calls++;
    last_down_stride=expert_stride;last_down_used=used_experts;
    return fake_down(output,tiles,tile_count);
}
fg_status fg_vk_moe_reduce(fg_vk_context *context,fg_vk_tensor *output,
    const fg_vk_tensor *down,const fg_vk_tensor *gates,
    const fg_vk_tensor *tiles,uint32_t output_width,uint32_t selected_count,
    uint32_t slot_count,fg_error *err){
    (void)context;(void)output;(void)down;(void)gates;(void)tiles;
    (void)output_width;(void)selected_count;(void)slot_count;(void)err;
    fast_reduce_calls++;
    return FG_OK;
}
fg_status fg_vk_decode_tile_schedule(fg_vk_context *context,
    fg_vk_tensor *tiles,const fg_vk_tensor *selected,fg_error *err){
    (void)context;(void)err;decode_schedule_calls++;
    uint32_t *schedule=(uint32_t *)tiles->data;
    const uint32_t *experts=(const uint32_t *)selected->data;
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){
        for(uint32_t word=0;word<9u;word++)
            schedule[slot*9u+word]=UINT32_MAX;
        schedule[slot*9u]=experts[slot];
        schedule[slot*9u+1u]=slot;
    }
    return FG_OK;
}
fg_status fg_vk_moe_prefill_reduce(fg_vk_context *context,
    fg_vk_tensor *output,const fg_vk_tensor *expert_output,
    const fg_vk_tensor *gates,const fg_vk_tensor *shared_output,
    const fg_vk_tensor *shared_logit,uint32_t width,uint32_t tokens,
    fg_error *err){
    (void)context;(void)expert_output;(void)gates;(void)shared_output;
    (void)shared_logit;(void)width;(void)tokens;(void)err;
    shared_combine_calls++;
    memset(output->data,0,(size_t)output->bytes);
    return FG_OK;
}

fg_status fg_owner_executor_create(fg_owner_executor **out,fg_model *model,
                                   fg_error *err){
    fg_owner_executor *owner=calloc(1,sizeof(*owner));
    if(!owner)return FG_ERR_OOM;
    owner->model=model;
    uint64_t bytes=(uint64_t)model->manifest->prefill_microbatch*
        FG_PIPELINE_BOUNDARY_WIDTH*sizeof(float);
    fg_status status=fg_vk_tensor_create(&model->vk,bytes,&owner->input,err);
    for(uint32_t i=0;status==FG_OK&&i<2u;i++)
        status=fg_vk_tensor_create(&model->vk,bytes,&owner->ping[i],err);
    if(status!=FG_OK){fg_owner_executor_destroy(owner);return status;}
    *out=owner;return FG_OK;
}
void fg_owner_executor_destroy(fg_owner_executor *owner){
    if(!owner)return;
    fg_vk_tensor_destroy(owner->ping[1]);fg_vk_tensor_destroy(owner->ping[0]);
    fg_vk_tensor_destroy(owner->input);free(owner);
}
bool fg_owner_owns_layer(const fg_owner_executor *owner,uint32_t layer){
    return owner&&owner->model->manifest->layer_owner[layer]==owner->model->rank;
}
fg_vk_tensor *fg_owner_prefill_input(fg_owner_executor *owner){
    return owner?owner->input:NULL;
}
fg_status fg_owner_qsa_open(fg_owner_executor *owner,const char *path,
                            bool create,fg_error *err){
    (void)owner;(void)create;
    if(!path){fg_error_set(err,FG_ERR_ARGUMENT,"fake qsa path");return FG_ERR_ARGUMENT;}
    qsa_opens++;return FG_OK;
}
fg_status fg_owner_qsa_open_resident(fg_owner_executor *owner,fg_error *err){
    (void)err;
    if(!owner)return FG_ERR_ARGUMENT;
    qsa_opens++;return FG_OK;
}
fg_status fg_owner_reset_state(fg_owner_executor *owner,fg_error *err){
    (void)owner;owner_resets++;
    if(owner_reset_status!=FG_OK){
        fg_error_set(err,owner_reset_status,"synthetic owner reset failure");
        return owner_reset_status;
    }
    return FG_OK;
}

static fg_vk_tensor *owner_output(fg_owner_executor *owner,
                                  const fg_vk_tensor *input,uint32_t tokens,
                                  uint32_t layer){
    fg_vk_tensor *output=owner->ping[owner->next++&1u];
    const float *source=(const float *)input->data;
    float *destination=(float *)output->data;
    for(uint64_t i=0;i<(uint64_t)tokens*FG_PIPELINE_BOUNDARY_WIDTH;i++)
        destination[i]=source[i]+1.0f;
    owner_calls++;
    (void)layer;
    return output;
}

fg_status fg_owner_decode_layer(fg_owner_executor *owner,uint32_t layer,
    uint32_t token_index,const uint32_t position[3],
    const fg_vk_tensor *input,const fg_vk_tensor *ngram,
    fg_owner_expert_dispatch_fn dispatch,void *context,fg_vk_tensor **output,
    fg_error *err){
    (void)position;
    if(layer==fail_layer){fg_error_set(err,FG_ERR_IO,"synthetic layer failure");return FG_ERR_IO;}
    CHECK((layer==1u)==(ngram!=NULL));
    uint16_t ids[FG_TOP_K];float gates[FG_TOP_K];uint8_t activation[FG_Q8K_ACTIVATION_BYTES]={0};
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){ids[slot]=(uint16_t)slot;gates[slot]=0.1f;}
    fg_expert_result results[FG_GROUP_SIZE]={0};uint32_t count=0u;
    fg_status status=dispatch(context,layer,token_index,ids,gates,activation,
                              results,&count,err);
    if(status==FG_OK){
        decode_dispatch_ok=count==1u&&results[0].selected_count==1u&&
            results[0].routing_slots[0]==0xFFu&&
            fabsf(results[0].outputs[0][0]-5.5f)<1e-5f;
        *output=owner_output(owner,input,1u,layer);
    }
    return status;
}

fg_status fg_owner_prefill_layer(fg_owner_executor *owner,uint32_t layer,
    uint32_t first_token,const uint32_t *positions,uint16_t token_count,
    const fg_vk_tensor *input,const fg_vk_tensor *ngram,
    fg_owner_prefill_dispatch_fn dispatch,void *context,
    fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,void *qsa_context,
    fg_vk_tensor **output,fg_error *err){
    (void)positions;(void)qsa_dispatch;(void)qsa_context;
    if(layer==fail_layer){fg_error_set(err,FG_ERR_IO,"synthetic layer failure");return FG_ERR_IO;}
    CHECK((layer==1u)==(ngram!=NULL));
    uint16_t ids[FG_PREFILL_MAX_PAIRS];float gates[FG_PREFILL_MAX_PAIRS];
    uint8_t activations[2u*FG_Q8K_ACTIVATION_BYTES]={0};
    for(uint32_t token=0;token<token_count;token++)
        for(uint32_t slot=0;slot<FG_TOP_K;slot++){
            uint32_t at=token*FG_TOP_K+slot;
            ids[at]=(uint16_t)slot;gates[at]=0.1f;
        }
    fg_prefill_result results[FG_GROUP_SIZE]={0};uint32_t count=0u;
    fg_status status=dispatch(context,layer,first_token,token_count,ids,gates,
                              activations,results,&count,err);
    if(status==FG_OK){
        bool ok=count==1u&&results[0].pair_count==token_count*FG_TOP_K;
        for(uint32_t pair=0;ok&&pair<results[0].pair_count;pair++){
            uint32_t token=results[0].pairs[pair].token_slot;
            uint32_t slot=results[0].pairs[pair].routing_slot;
            float expected=(float)(slot+1u)+(float)(token*FG_TOP_K+slot)*0.01f;
            ok=fabsf(results[0].outputs[(uint64_t)pair*FG_HIDDEN_SIZE]-
                     expected)<1e-5f;
        }
        prefill_dispatch_ok=ok;
        *output=owner_output(owner,input,token_count,layer);
    }
    return status;
}

fg_status fg_owner_prefill_layer_pipeline(fg_owner_executor *owner,
    fg_expert_executor *expert,uint32_t layer,uint32_t first_token,
    const uint32_t *positions,uint16_t token_count,
    const fg_vk_tensor *input,const fg_vk_tensor *ngram,
    fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,void *qsa_context,
    fg_vk_tensor **output,fg_error *err){
    (void)expert;(void)first_token;(void)positions;(void)qsa_dispatch;
    (void)qsa_context;
    if(layer==fail_layer){
        fg_error_set(err,FG_ERR_IO,"synthetic layer failure");
        return FG_ERR_IO;
    }
    CHECK(token_count>1u);
    CHECK((layer==1u)==(ngram!=NULL));
    fg_status status=fg_vk_begin(&owner->model->vk,err);
    if(status==FG_OK&&fail_moe){
        fg_error_set(err,FG_ERR_IO,"synthetic stage-local expert failure");
        status=FG_ERR_IO;
    }
    if(status==FG_OK){
        prefill_dispatch_ok=true;
        *output=owner_output(owner,input,token_count,layer);
        status=fg_vk_end(&owner->model->vk,err);
    }
    if(status!=FG_OK&&fg_vk_batch_active(&owner->model->vk)){
        fg_error ignored={0};
        fg_vk_abort(&owner->model->vk,&ignored);
    }
    return status;
}

fg_status fg_owner_decode_layer_pipeline(fg_owner_executor *owner,
    fg_expert_executor *expert,uint32_t layer,uint32_t token_index,
    const uint32_t position[3],const fg_vk_tensor *input,
    const fg_vk_tensor *ngram,fg_vk_tensor **output,fg_error *err){
    (void)expert;(void)token_index;(void)position;
    CHECK(fg_vk_batch_active(&owner->model->vk));
    CHECK((layer==1u)==(ngram!=NULL));
    if(layer==fail_layer||fail_moe){
        fg_error_set(err,FG_ERR_IO,"synthetic pipeline decode failure");
        return FG_ERR_IO;
    }
    fg_status status=fg_vk_begin(&owner->model->vk,err);
    if(status==FG_OK){
        decode_dispatch_ok=true;
        *output=owner_output(owner,input,1u,layer);
        status=fg_vk_end(&owner->model->vk,err);
    }
    return status;
}

fg_status fg_output_executor_create(fg_output_executor **out,fg_model *model,
                                    fg_error *err){
    (void)err;
    if(model->rank!=fg_output_owner_rank(model->manifest))return FG_ERR_MISMATCH;
    fg_output_executor *output=calloc(1,sizeof(*output));
    if(!output)return FG_ERR_OOM;
    output->model=model;output_creates++;*out=output;return FG_OK;
}
void fg_output_executor_destroy(fg_output_executor *output){
    if(output)output_destroys++;
    free(output);
}
fg_status fg_output_history_reset(fg_output_executor *output,
                                  const uint32_t *history,uint32_t count,
                                  fg_error *err){
    (void)output;(void)history;(void)count;(void)err;
    output_history_resets++;
    return FG_OK;
}
fg_status fg_output_greedy(fg_output_executor *output,const fg_vk_tensor *hyper,
                           uint32_t *token,float *logit,fg_error *err){
    CHECK(!fg_vk_batch_active(&output->model->vk));
    fg_status status=fg_vk_begin(&output->model->vk,err);
    if(status!=FG_OK)return status;
    output_calls++;
    *token=1234u;*logit=((const float *)hyper->data)[0];
    return fg_vk_end(&output->model->vk,err);
}

fg_status fg_output_sample(fg_output_executor *output,const fg_vk_tensor *hyper,
                           const fg_sampler_config *config,
                           float uniform,uint32_t *token,float *logit,fg_error *err){
    (void)config;(void)uniform;
    return fg_output_greedy(output,hyper,token,logit,err);
}

static fg_status prepared_ngram_decode(void *context,uint64_t request_id,
    uint32_t sequence,uint32_t token_index,fg_vk_tensor **embedding,
    fg_error *err){
    (void)request_id;(void)sequence;(void)token_index;(void)err;
    ngram_calls++;*embedding=context;return FG_OK;
}
static fg_status prepared_ngram_prefill(void *context,uint64_t request_id,
    uint32_t sequence,uint32_t first_token,uint16_t token_count,
    fg_vk_tensor **embedding,fg_error *err){
    (void)request_id;(void)sequence;(void)first_token;(void)token_count;(void)err;
    ngram_calls++;*embedding=context;return FG_OK;
}

static fg_stage_executor *create_stage(fg_model *model,fg_vk_tensor *ngram){
    fg_stage_config config={.model=model,.qsa_state_path="stage.qsa",
        .create_qsa_state=true,.ngram_decode=prepared_ngram_decode,
        .ngram_prefill=prepared_ngram_prefill,.ngram_context=ngram};
    fg_stage_executor *stage=NULL;fg_error error={0};
    CHECK(fg_stage_executor_create(&stage,&config,&error)==FG_OK);
    return stage;
}

static void test_mapping_and_names(void){
    fg_manifest manifest;fg_model model;fg_error error={0};
    manifest_ep(&manifest,3u);
    CHECK(fg_expert_uses_rank_suffix(&manifest));
    CHECK(fg_expert_local_count(&manifest,0u,3u)==FG_EXPERTS_PER_RANK);
    CHECK(fg_expert_local_index(&manifest,0u,3u,127u)==127u);
    CHECK(fg_expert_local_index(&manifest,0u,3u,128u)==UINT32_MAX);
    model_init(&model,&manifest,3u);
    fg_expert_executor *expert=NULL;
    CHECK(fg_expert_executor_create(&expert,&model,&error)==FG_OK);
    CHECK(!model.bad_tensor_name);
    CHECK(graph_weight_experts==FG_EXPERTS_PER_RANK);
    CHECK(graph_tiles_bytes==(uint64_t)manifest.prefill_microbatch*FG_TOP_K*
          FG_Q38_DECODE_TILE_WORDS*4u);
    CHECK(graph_reduced_bytes==FG_HIDDEN_SIZE*4u);
    fg_expert_executor_destroy(expert);model_destroy(&model);

    manifest_pipeline(&manifest);model_init(&model,&manifest,5u);
    CHECK(!fg_expert_uses_rank_suffix(&manifest));
    CHECK(fg_expert_local_count(&manifest,30u,5u)==FG_EXPERT_COUNT);
    CHECK(fg_expert_local_index(&manifest,30u,5u,511u)==511u);
    expert=NULL;graph_weight_experts=0u;
    CHECK(fg_expert_executor_create(&expert,&model,&error)==FG_OK);
    CHECK(!model.bad_tensor_name);
    CHECK(graph_weight_experts==FG_EXPERT_COUNT);
    CHECK(graph_tiles_bytes==(uint64_t)manifest.prefill_microbatch*FG_TOP_K*
          FG_Q38_DECODE_TILE_WORDS*4u);
    CHECK(graph_reduced_bytes==(uint64_t)FG_TOP_K*FG_HIDDEN_SIZE*4u);
    fg_vk_tensor *activation=NULL,*selected=NULL,*gates=NULL;
    fg_vk_tensor *shared=NULL,*shared_logit=NULL,*pipeline_output=NULL;
    CHECK(fg_vk_tensor_create(&model.vk,FG_Q8K_ACTIVATION_BYTES,
                              &activation,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,FG_TOP_K*sizeof(uint32_t),
                              &selected,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,FG_TOP_K*sizeof(float),
                              &gates,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,FG_HIDDEN_SIZE*sizeof(float),
                              &shared,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,sizeof(float),&shared_logit,
                              &error)==FG_OK);
    for(uint32_t slot=0;slot<FG_TOP_K;slot++)
        ((uint32_t *)selected->data)[slot]=slot;
    CHECK(fg_expert_decode_pipeline(expert,30u,activation,selected,gates,
                                    shared,shared_logit,
                                    &pipeline_output,&error)==FG_ERR_MISMATCH);
    uint64_t submissions=model.vk.submissions;
    CHECK(fg_vk_begin(&model.vk,&error)==FG_OK);
    uint32_t schedules_before=decode_schedule_calls;
    uint32_t fast_before=fast_moe_calls,grouped_before=grouped_moe_calls;
    uint32_t down_before=fast_down_calls,reduce_before=fast_reduce_calls;
    uint32_t combine_before=shared_combine_calls;
    CHECK(fg_expert_decode_pipeline(expert,30u,activation,selected,gates,
                                    shared,shared_logit,
                                    &pipeline_output,&error)==FG_OK);
    CHECK(pipeline_output&&fg_vk_batch_active(&model.vk)&&
          model.vk.submissions==submissions);
    CHECK(decode_schedule_calls==schedules_before+1u&&
          fast_moe_calls==fast_before+2u&&
          grouped_moe_calls==grouped_before&&
          fast_down_calls==down_before+1u&&
          fast_reduce_calls==reduce_before+1u&&
          shared_combine_calls==combine_before+1u);
    CHECK(fg_vk_end(&model.vk,&error)==FG_OK&&
          model.vk.submissions==submissions+1u);
    fg_vk_tensor_destroy(shared_logit);fg_vk_tensor_destroy(shared);
    fg_vk_tensor_destroy(gates);fg_vk_tensor_destroy(selected);
    fg_vk_tensor_destroy(activation);
    fg_expert_executor_destroy(expert);model_destroy(&model);
}

static void test_manifest_validation(void){
    fg_manifest manifest;fg_error error={0};uint32_t stage=0,begin=0,end=0;
    manifest_pipeline(&manifest);
    CHECK(fg_stage_validate_manifest(&manifest,4u,&stage,&begin,&end,&error)==FG_OK);
    CHECK(stage==4u&&begin==24u&&end==30u);
    manifest.layer_owner[25u]=5u;
    CHECK(fg_stage_validate_manifest(&manifest,4u,NULL,NULL,NULL,&error)==
          FG_ERR_FORMAT);
    manifest_pipeline(&manifest);
    manifest.expert_rank[24u][511u]=3u;
    CHECK(fg_stage_validate_manifest(&manifest,4u,NULL,NULL,NULL,&error)==
          FG_ERR_FORMAT);
}

static void test_decode_prefill_and_reset(void){
    fg_manifest manifest;manifest_pipeline(&manifest);
    fg_model model;model_init(&model,&manifest,2u);
    fg_vk_tensor *ngram=NULL;fg_error error={0};
    CHECK(fg_vk_tensor_create(&model.vk,
          2u*FG_NGRAM_EMBED_VALUES*sizeof(float),&ngram,&error)==FG_OK);
    fg_stage_executor *stage=create_stage(&model,ngram);
    float boundary[2u*FG_PIPELINE_BOUNDARY_WIDTH]={0};
    uint32_t positions[2u*FG_PIPELINE_POSITION_AXES]={0,0,0,1,1,1};
    fg_pipeline_activation activation={.execution_kind=FG_PIPELINE_EXECUTION_DECODE,
        .token_count=1u,.first_token=7u,.request_output=true,
        .sampler={.temperature=0.0f,.top_p=1.0f,.top_k=1u,
                  .repetition_penalty=1.0f},
        .positions=positions,.boundary=boundary};
    decode_dispatch_ok=false;
    uint64_t submissions=model.vk.submissions;
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,10u,&activation,boundary,NULL,
                                    &error)==FG_OK);
    CHECK(decode_dispatch_ok&&boundary[0]==6.0f&&owner_calls==6u&&
          model.vk.submissions==submissions+1u);

    memset(boundary,0,sizeof(boundary));boundary[FG_PIPELINE_BOUNDARY_WIDTH]=10.0f;
    activation.execution_kind=FG_PIPELINE_EXECUTION_PREFILL;
    activation.token_count=2u;activation.first_token=8u;
    activation.request_output=false;
    prefill_dispatch_ok=false;
    uint32_t outputs_before=output_calls;
    uint64_t prefill_submissions=model.vk.submissions;
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,11u,&activation,boundary,NULL,
                                    &error)==FG_OK);
    CHECK(prefill_dispatch_ok&&boundary[0]==6.0f&&
          boundary[FG_PIPELINE_BOUNDARY_WIDTH]==16.0f);
    CHECK(output_calls==outputs_before&&
          model.vk.submissions==prefill_submissions+1u&&
          !fg_vk_batch_active(&model.vk));

    uint32_t prefill_aborts=abort_calls;
    prefill_submissions=model.vk.submissions;
    fail_moe=true;
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,12u,&activation,boundary,NULL,
                                    &error)==FG_ERR_IO);
    CHECK(!fg_vk_batch_active(&model.vk)&&abort_calls==prefill_aborts+1u&&
          model.vk.submissions==prefill_submissions);
    fail_moe=false;
    CHECK(fg_stage_executor_reset(stage,&error)==FG_OK);
    fail_layer=14u;uint32_t before=owner_calls;
    activation.execution_kind=FG_PIPELINE_EXECUTION_DECODE;
    activation.token_count=1u;activation.request_output=true;
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,12u,&activation,boundary,NULL,
                                    &error)==FG_ERR_IO);
    fail_layer=UINT32_MAX;
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,13u,&activation,boundary,NULL,
                                    &error)==FG_ERR_IO);
    CHECK(owner_calls==before+2u);
    CHECK(fg_vk_begin(&model.vk,&error)==FG_OK&&fg_vk_batch_active(&model.vk));
    uint32_t resets_before=owner_resets;
    abort_status=FG_ERR_IO;
    CHECK(fg_stage_executor_reset(stage,&error)==FG_ERR_IO);
    CHECK(!fg_vk_batch_active(&model.vk)&&owner_resets==resets_before);
    abort_status=FG_OK;owner_reset_status=FG_ERR_MISMATCH;
    CHECK(fg_stage_executor_reset(stage,&error)==FG_ERR_MISMATCH);
    CHECK(owner_resets==resets_before+1u);
    owner_reset_status=FG_OK;
    CHECK(fg_stage_executor_reset(stage,&error)==FG_OK&&
          owner_resets==resets_before+2u);
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,14u,&activation,boundary,NULL,
                                    &error)==FG_OK);
    fg_stage_executor_close(stage);
    fg_vk_tensor_destroy(ngram);model_destroy(&model);
}

static void test_stage_local_batch_failure(void){
    fg_manifest manifest;manifest_pipeline(&manifest);
    fg_model model;model_init(&model,&manifest,2u);
    fg_vk_tensor *ngram=NULL;fg_error error={0};
    CHECK(fg_vk_tensor_create(&model.vk,
          2u*FG_NGRAM_EMBED_VALUES*sizeof(float),&ngram,&error)==FG_OK);
    fg_stage_executor *stage=create_stage(&model,ngram);
    float boundary[FG_PIPELINE_BOUNDARY_WIDTH]={0};
    uint32_t position[FG_PIPELINE_POSITION_AXES]={0};
    fg_pipeline_activation activation={
        .execution_kind=FG_PIPELINE_EXECUTION_DECODE,.token_count=1u,
        .first_token=1u,.request_output=true,
        .sampler={.temperature=0.0f,.top_p=1.0f,.top_k=1u,
                  .repetition_penalty=1.0f},.positions=position,
        .boundary=boundary
    };
    profile_active=true;fail_moe=true;
    uint32_t aborts_before=abort_calls;
    uint64_t submissions=model.vk.submissions;
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,20u,&activation,boundary,NULL,
                                    &error)==FG_ERR_IO);
    CHECK(!fg_vk_batch_active(&model.vk)&&abort_calls==aborts_before+1u&&
          model.vk.submissions==submissions);
    profile_active=false;fail_moe=false;
    CHECK(fg_stage_executor_reset(stage,&error)==FG_OK);
    CHECK(fg_stage_pipeline_execute(stage,2u,1u,21u,&activation,boundary,NULL,
                                    &error)==FG_OK);
    CHECK(model.vk.submissions==submissions+1u);
    fg_stage_executor_close(stage);
    fg_vk_tensor_destroy(ngram);model_destroy(&model);
}

static void test_stage0_ngram_and_terminal(void){
    fg_manifest manifest;manifest_pipeline(&manifest);fg_error error={0};
    fg_vk_tensor *ngram=NULL;
    fg_model stage0_model;model_init(&stage0_model,&manifest,0u);
    CHECK(fg_vk_tensor_create(&stage0_model.vk,
          2u*FG_NGRAM_EMBED_VALUES*sizeof(float),&ngram,&error)==FG_OK);
    fg_stage_executor *stage0=create_stage(&stage0_model,ngram);
    float boundary[FG_PIPELINE_BOUNDARY_WIDTH]={0};
    uint32_t position[FG_PIPELINE_POSITION_AXES]={0};
    fg_pipeline_activation activation={.execution_kind=FG_PIPELINE_EXECUTION_DECODE,
        .token_count=1u,.request_output=true,
        .sampler={.temperature=0.0f,.top_p=1.0f,.top_k=1u,
                  .repetition_penalty=1.0f},.positions=position,
        .boundary=boundary};
    uint32_t prior_ngram=ngram_calls;
    uint64_t stage0_submissions=stage0_model.vk.submissions;
    CHECK(fg_stage_pipeline_execute(stage0,0u,1u,1u,&activation,boundary,NULL,
                                    &error)==FG_OK);
    CHECK(ngram_calls==prior_ngram+1u&&boundary[0]==6.0f&&
          stage0_model.vk.submissions==stage0_submissions+1u);
    fg_stage_executor_close(stage0);model_destroy(&stage0_model);

    fg_model stage7_model;model_init(&stage7_model,&manifest,7u);
    fg_stage_executor *stage7=create_stage(&stage7_model,ngram);
    float terminal_boundary[2u*FG_PIPELINE_BOUNDARY_WIDTH]={0};
    terminal_boundary[FG_PIPELINE_BOUNDARY_WIDTH]=10.0f;
    uint32_t positions[2u*FG_PIPELINE_POSITION_AXES]={0};
    fg_pipeline_result result={.completed_first_token=20u,
        .completed_token_count=2u,.completed_frontier=22u,
        .final_token=FG_Q38_VOCAB_SIZE,.final_logit=0.0f};
    activation=(fg_pipeline_activation){
        .execution_kind=FG_PIPELINE_EXECUTION_PREFILL,.token_count=2u,
        .first_token=20u,.request_output=false,
        .sampler={.temperature=0.0f,.top_p=1.0f,.top_k=1u,
                  .repetition_penalty=1.0f},.positions=positions,
        .boundary=terminal_boundary};
    uint32_t calls_before=output_calls;
    CHECK(fg_stage_pipeline_execute(stage7,7u,1u,2u,&activation,
                                   terminal_boundary,&result,&error)==FG_OK);
    CHECK(terminal_boundary[0]==6.0f&&
          terminal_boundary[FG_PIPELINE_BOUNDARY_WIDTH]==16.0f);
    CHECK(result.completed_first_token==20u&&result.completed_token_count==2u&&
          result.completed_frontier==22u&&!result.has_output&&
          result.final_token==FG_Q38_VOCAB_SIZE&&
          result.final_logit==0.0f&&output_calls==calls_before);

    memset(terminal_boundary,0,sizeof(terminal_boundary));
    terminal_boundary[FG_PIPELINE_BOUNDARY_WIDTH]=10.0f;
    activation.request_output=true;
    uint64_t prefill_submissions=stage7_model.vk.submissions;
    CHECK(fg_stage_pipeline_execute(stage7,7u,1u,3u,&activation,
                                   terminal_boundary,&result,&error)==FG_OK);
    CHECK(result.has_output&&result.final_token==1234u&&
          result.final_logit==16.0f&&output_calls==calls_before+1u&&
          stage7_model.vk.submissions==prefill_submissions+2u&&
          !fg_vk_batch_active(&stage7_model.vk));

    activation.execution_kind=FG_PIPELINE_EXECUTION_DECODE;
    activation.token_count=1u;activation.request_output=false;
    activation.first_token=22u;activation.positions=position;
    CHECK(fg_stage_pipeline_execute(stage7,7u,1u,4u,&activation,
                                   terminal_boundary,&result,&error)==
          FG_ERR_MISMATCH);
    CHECK(output_calls==calls_before+1u);
    uint32_t history_resets_stage7=output_history_resets;
    CHECK(fg_stage_executor_reset(stage7,&error)==FG_OK&&
          output_history_resets==history_resets_stage7+1u);
    activation.request_output=true;
    result.has_output=false;result.final_token=FG_Q38_VOCAB_SIZE;
    result.final_logit=0.0f;
    uint64_t decode_submissions=stage7_model.vk.submissions;
    CHECK(fg_stage_pipeline_execute(stage7,7u,1u,5u,&activation,
                                   terminal_boundary,&result,&error)==FG_OK);
    CHECK(result.has_output&&result.final_token==1234u&&
          output_calls==calls_before+2u&&
          stage7_model.vk.submissions==decode_submissions+2u);
    CHECK(fg_output_owner_rank(&manifest)==7u&&output_creates==1u);
    fg_stage_executor_close(stage7);
    CHECK(output_destroys==1u);
    fg_vk_tensor_destroy(ngram);model_destroy(&stage7_model);

    fg_manifest ep;manifest_ep(&ep,3u);
    CHECK(fg_output_owner_rank(&ep)==4u);
}

static void test_pipeline_decode_geometry(void){
    fg_q38_pipeline_owner_transient_layout layout={0};
    CHECK(FG_Q38_DECODE_TILE_WORDS==9u);
    CHECK(FG_Q38_PIPELINE_DECODE_EXTRA_BYTES==UINT64_C(92200));
    CHECK(fg_q38_pipeline_owner_transient_layout_get(128u,&layout));
    CHECK(layout.total_bytes==UINT64_C(16166912));
    CHECK(layout.bytes[FG_Q38_PIPELINE_TRANSIENT_SELECTED]==
          128u*FG_TOP_K*4u);
    CHECK(layout.bytes[FG_Q38_PIPELINE_TRANSIENT_GATES]==
          128u*FG_TOP_K*4u);
    CHECK(layout.bytes[FG_Q38_PIPELINE_TRANSIENT_PREFILL_TILES]==
          128u*FG_TOP_K*FG_Q38_PREFILL_TILE_WORDS*4u);
}

static fg_status run_pipeline_decode_format_case(fg_model *model,
    fg_expert_executor *expert,fg_vk_tensor *activation,fg_vk_tensor *selected,
    fg_vk_tensor *gates,fg_vk_tensor *shared,fg_vk_tensor *shared_logit,
    fg_vk_tensor **output,fg_error *error){
    CHECK(fg_vk_begin(&model->vk,error)==FG_OK);
    fg_status status=fg_expert_decode_pipeline(expert,30u,activation,selected,
        gates,shared,shared_logit,output,error);
    if(status==FG_OK){
        CHECK(fg_vk_batch_active(&model->vk));
        CHECK(fg_vk_end(&model->vk,error)==FG_OK);
    }else{
        CHECK(!fg_vk_batch_active(&model->vk));
    }
    return status;
}

static void test_pipeline_decode_down_formats(void){
    fg_manifest manifest;manifest_pipeline(&manifest);
    fg_model model;model_init(&model,&manifest,5u);
    fg_error error={0};fg_expert_executor *expert=NULL;
    CHECK(fg_expert_executor_create(&expert,&model,&error)==FG_OK);
    fg_vk_tensor *activation=NULL,*selected=NULL,*gates=NULL,*shared=NULL;
    fg_vk_tensor *shared_logit=NULL,*output=NULL;
    CHECK(fg_vk_tensor_create(&model.vk,FG_Q8K_ACTIVATION_BYTES,
                              &activation,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,FG_TOP_K*4u,&selected,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,FG_TOP_K*4u,&gates,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,FG_HIDDEN_SIZE*4u,
                              &shared,&error)==FG_OK);
    CHECK(fg_vk_tensor_create(&model.vk,4u,&shared_logit,&error)==FG_OK);
    for(uint32_t slot=0;slot<FG_TOP_K;slot++)
        ((uint32_t *)selected->data)[slot]=slot;

    uint32_t q5_before=q5_down_calls,q8_before=q8_down_calls;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_OK);
    CHECK(q5_down_calls==q5_before+1u&&q8_down_calls==q8_before&&
          last_down_stride==64u&&last_down_used==FG_TOP_K);

    model.records[2].ggml_type=8u;
    model.weights[2].format=FG_VK_TENSOR_FORMAT_DEFAULT;
    q5_before=q5_down_calls;q8_before=q8_down_calls;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_OK);
    CHECK(q5_down_calls==q5_before&&q8_down_calls==q8_before+1u&&
          last_down_stride==64u&&last_down_used==FG_TOP_K);

    model.records[2].ggml_type=7u;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_ERR_FORMAT);
    model.records[2].ggml_type=8u;
    model.weights[2].format=FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_ERR_FORMAT);
    model.records[2].ggml_type=9u;
    model.weights[2].format=FG_VK_TENSOR_FORMAT_DEFAULT;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_ERR_FORMAT);

    model.records[2].ggml_type=8u;
    model.weights[0].format=FG_VK_TENSOR_FORMAT_DEFAULT;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_ERR_FORMAT);
    model.weights[0].format=FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED;
    model.records[0].ggml_type=8u;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_ERR_FORMAT);
    model.records[0].ggml_type=12u;
    model.weights[1].format=FG_VK_TENSOR_FORMAT_DEFAULT;
    CHECK(run_pipeline_decode_format_case(&model,expert,activation,selected,
          gates,shared,shared_logit,&output,&error)==FG_ERR_FORMAT);

    fg_vk_tensor_destroy(shared_logit);fg_vk_tensor_destroy(shared);
    fg_vk_tensor_destroy(gates);fg_vk_tensor_destroy(selected);
    fg_vk_tensor_destroy(activation);fg_expert_executor_destroy(expert);
    model_destroy(&model);
}

static bool test_selected(const char *name){
    const char *filter=getenv("DS4_REMOTE_TEST_FILTER");
    return !filter||!*filter||strcmp(filter,name)==0;
}

int main(void){
    const char *filter=getenv("DS4_REMOTE_TEST_FILTER");
    uint32_t selected=0u;
    if(test_selected("pipeline_decode_mapping")){
        selected++;test_mapping_and_names();test_manifest_validation();
    }
    if(test_selected("pipeline_decode_geometry")){
        selected++;test_pipeline_decode_geometry();
    }
    if(test_selected("pipeline_decode_down_formats")){
        selected++;test_pipeline_decode_down_formats();
    }
    if(test_selected("pipeline_decode_batch")){
        selected++;test_decode_prefill_and_reset();
    }
    if(test_selected("pipeline_decode_failure_cleanup")){
        selected++;test_stage_local_batch_failure();
    }
    if(test_selected("pipeline_decode_terminal_submissions")){
        selected++;test_stage0_ngram_and_terminal();
    }
    if(!selected){
        fprintf(stderr,"no stage test matched DS4_REMOTE_TEST_FILTER\n");
        return 2;
    }
    if(!filter||!*filter)CHECK(qsa_opens==4u);
    if(failures){
        fprintf(stderr,"%d stage tests failed\n",failures);
        return 1;
    }
    printf("stage tests passed\n");
    return 0;
}
