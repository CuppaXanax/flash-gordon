#include "fg_expert.h"
#include "fg_vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FG_EXPERT_TILES_WORDS (FG_TOP_K * FG_Q38_DECODE_TILE_WORDS)
#define FG_EXPERT_MID_VALUES (FG_TOP_K * 640u)
#define FG_EXPERT_OUTPUT_VALUES (FG_TOP_K * FG_HIDDEN_SIZE)
_Static_assert(FG_Q38_DECODE_TILE_WORDS==9u,
               "decode schedules use nine words per pair");

struct fg_expert_executor {
    fg_model *model;
    fg_vk_tensor *activation,*tiles,*gates,*selected;
    fg_vk_tensor *gate,*up,*mid,*down,*reduced,*result_readback;
    fg_vk_expert_graph *decode_graph[FG_LAYER_COUNT];
    uint32_t max_tokens,max_pairs;
    uint16_t *locals;
    uint32_t *schedule;
};

static fg_status create_decode_graphs(fg_expert_executor *executor,fg_error *err);

static fg_status create_scratch(fg_expert_executor *executor,fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);executor->max_tokens=manifest->prefill_microbatch;executor->max_pairs=executor->max_tokens*FG_TOP_K;
    if(!executor->max_tokens||executor->max_tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds expert executor limit");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(executor->model);fg_status status=fg_vk_tensor_create(vk,(uint64_t)executor->max_tokens*FG_Q8K_ACTIVATION_BYTES,&executor->activation,err);
    uint64_t tile_bytes=(uint64_t)executor->max_pairs*
        FG_Q38_PREFILL_TILE_WORDS*4u;
    if(status==FG_OK)status=fg_vk_tensor_create(vk,tile_bytes,
                                                &executor->tiles,err);
    if(status==FG_OK&&fg_vk_tensor_bytes(executor->tiles)!=tile_bytes){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "expert schedule allocation violates grouped prefill geometry");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*4u,&executor->gates,err);
    /* Top-10 selection scratch: one row per token so the batch-2 union can
     * select for the pair in one dispatch. */
    if(status==FG_OK)status=fg_vk_tensor_create(vk,
        (uint64_t)executor->max_tokens*FG_TOP_K*4u,&executor->selected,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->gate,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->up,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->mid,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*FG_HIDDEN_SIZE*4u,&executor->down,err);

    /* The routed reduction is one hidden row per token; the batch-2 union
     * writes two, so the arena is sized for the sealed microbatch. */
    if(status==FG_OK)status=fg_vk_tensor_create(vk,
        (uint64_t)executor->max_tokens*FG_HIDDEN_SIZE*4u,
        &executor->reduced,err);
    if(status==FG_OK)status=fg_vk_tensor_create_cached(vk,
        (uint64_t)executor->max_tokens*FG_HIDDEN_SIZE*4u,
        &executor->result_readback,err);
    if(status==FG_OK){
        executor->locals=malloc((size_t)executor->max_pairs*sizeof(*executor->locals));
        executor->schedule=malloc((size_t)executor->max_pairs*
                                  FG_Q38_PREFILL_TILE_WORDS*
                                  sizeof(*executor->schedule));
        if(!executor->locals||!executor->schedule){
            fg_error_set(err,FG_ERR_OOM,"allocate expert schedules");
            status=FG_ERR_OOM;
        }
    }

    return status;
}

fg_status fg_expert_executor_create(fg_expert_executor **out,fg_model *model,fg_error *err){if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert executor arguments");return FG_ERR_ARGUMENT;}*out=NULL;fg_expert_executor *executor=calloc(1,sizeof(*executor));if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate expert executor");return FG_ERR_OOM;}executor->model=model;fg_status status=create_scratch(executor,err);if(status==FG_OK)status=create_decode_graphs(executor,err);if(status!=FG_OK){fg_expert_executor_destroy(executor);return status;}*out=executor;return FG_OK;}
void fg_expert_executor_destroy(fg_expert_executor *executor){if(!executor)return;for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)fg_vk_expert_graph_destroy(executor->decode_graph[layer]);free(executor->schedule);free(executor->locals);fg_vk_tensor_destroy(executor->result_readback);fg_vk_tensor_destroy(executor->reduced);fg_vk_tensor_destroy(executor->down);fg_vk_tensor_destroy(executor->mid);fg_vk_tensor_destroy(executor->up);fg_vk_tensor_destroy(executor->gate);fg_vk_tensor_destroy(executor->selected);fg_vk_tensor_destroy(executor->gates);fg_vk_tensor_destroy(executor->tiles);fg_vk_tensor_destroy(executor->activation);free(executor);}

uint32_t fg_expert_local_count(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank){
    if(!manifest||layer>=FG_LAYER_COUNT||rank>=FG_RANK_COUNT)return 0u;

    uint32_t count=0u;
    for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
        if(manifest->expert_rank[layer][expert]==rank)count++;
    return count;
}

uint32_t fg_expert_local_index(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank,uint32_t global){
    if(!manifest||layer>=FG_LAYER_COUNT||rank>=FG_RANK_COUNT||
       global>=FG_EXPERT_COUNT||
       manifest->expert_rank[layer][global]!=rank)return UINT32_MAX;

    uint32_t local=0u;
    for(uint32_t expert=0;expert<global;expert++)
        if(manifest->expert_rank[layer][expert]==rank)local++;
    return local;
}

static bool tensor_name(char output[FG_TENSOR_NAME_MAX],
                        uint32_t layer,
                        const char *family,uint32_t rank){
    int length=snprintf(output,FG_TENSOR_NAME_MAX,"blk.%u.%s.weight.rank%u",
                        layer,family,rank);
    return length>=0&&(uint32_t)length<FG_TENSOR_NAME_MAX;
}

static fg_status expert_binding(fg_expert_executor *executor,uint32_t layer,
                                const char *family,fg_vk_tensor **weight,
                                const fg_tensor_record **record,
                                uint32_t *local_count,uint32_t *stride,
                                fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t rank=fg_model_rank(executor->model);
    char name[FG_TENSOR_NAME_MAX];
    if(!tensor_name(name,layer,family,rank)){
        fg_error_set(err,FG_ERR_LIMIT,"expert tensor name overflow");
        return FG_ERR_LIMIT;
    }
    *weight=fg_model_tensor(executor->model,name);
    *record=fg_model_tensor_record(executor->model,name);
    *local_count=fg_expert_local_count(manifest,layer,rank);
    uint32_t expected=*local_count;
    if(!*weight||!*record||*local_count!=expected||
       (*record)->dims!=3u||(*record)->shape[2]!=*local_count||
       (*record)->bytes%*local_count||
       (*record)->bytes/(*local_count)>UINT32_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "rank is missing or has malformed layer %u expert tensor %s",
                     layer,name);
        return FG_ERR_MISMATCH;
    }
    *stride=(uint32_t)((*record)->bytes/(*local_count));
    return FG_OK;
}

static fg_status project_down(fg_vk_context *vk,fg_vk_tensor *output,
                              const fg_vk_tensor *weights,
                              const fg_vk_tensor *tiles,
                              const fg_vk_tensor *input,
                              const fg_tensor_record *record,uint32_t stride,
                              uint32_t routed_pairs,uint32_t tile_count,
                              fg_error *err){
    if(fg_vk_tensor_get_format(weights)==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED)
        return fg_vk_moe_q5_1_down_cooked_pairs(vk,output,weights,tiles,input,
            FG_HIDDEN_SIZE,640u,stride,routed_pairs,false,tile_count,err);
    if(fg_vk_tensor_get_format(weights)==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED)
        return fg_vk_moe_q8_0_down_cooked_pairs(vk,output,weights,tiles,input,
            FG_HIDDEN_SIZE,640u,stride,routed_pairs,false,tile_count,err);
    if(record->ggml_type==7u)
        return fg_vk_moe_q5_1_down(vk,output,weights,tiles,input,FG_HIDDEN_SIZE,
            640u,stride,FG_TOP_K,false,tile_count,err);
    if(record->ggml_type==8u)
        return fg_vk_moe_q8_0_down(vk,output,weights,tiles,input,FG_HIDDEN_SIZE,
            640u,stride,FG_TOP_K,false,tile_count,err);
    fg_error_set(err,FG_ERR_FORMAT,"unsupported expert down quant %u",
                 record->ggml_type);
    return FG_ERR_FORMAT;
}

static fg_status create_decode_graphs(fg_expert_executor *executor,fg_error *err){
    uint32_t rank=fg_model_rank(executor->model);
    fg_vk_context *vk=fg_model_vk(executor->model);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        char gate_name[FG_TENSOR_NAME_MAX],up_name[FG_TENSOR_NAME_MAX];
        char down_name[FG_TENSOR_NAME_MAX];
        if(!tensor_name(gate_name,layer,"ffn_gate_exps",rank)||
           !tensor_name(up_name,layer,"ffn_up_exps",rank)||
           !tensor_name(down_name,layer,"ffn_down_exps",rank)){
            fg_error_set(err,FG_ERR_LIMIT,"expert graph tensor name overflow");
            return FG_ERR_LIMIT;
        }
        bool any=fg_model_tensor(executor->model,gate_name)||
            fg_model_tensor_record(executor->model,gate_name)||
            fg_model_tensor(executor->model,up_name)||
            fg_model_tensor_record(executor->model,up_name)||
            fg_model_tensor(executor->model,down_name)||
            fg_model_tensor_record(executor->model,down_name);
        if(!any)continue;
        fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
        const fg_tensor_record *gate_record=NULL,*up_record=NULL,*down_record=NULL;
        uint32_t gate_count=0u,up_count=0u,down_count=0u;
        uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
        fg_status status=expert_binding(executor,layer,"ffn_gate_exps",
            &gate_weight,&gate_record,&gate_count,&gate_stride,err);
        if(status==FG_OK)status=expert_binding(executor,layer,"ffn_up_exps",
            &up_weight,&up_record,&up_count,&up_stride,err);
        if(status==FG_OK)status=expert_binding(executor,layer,"ffn_down_exps",
            &down_weight,&down_record,&down_count,&down_stride,err);
        if(status==FG_OK&&(gate_count!=up_count||gate_count!=down_count)){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "layer %u expert tensor counts disagree",layer);
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_vk_expert_graph_create(vk,
            &executor->decode_graph[layer],executor->activation,executor->tiles,
            executor->gates,executor->gate,executor->up,executor->mid,
            executor->down,executor->reduced,gate_weight,up_weight,down_weight,
            gate_record->ggml_type,up_record->ggml_type,down_record->ggml_type,
            FG_HIDDEN_SIZE,640u,gate_stride,up_stride,down_stride,gate_count,
            FG_TOP_K,err);
        if(status!=FG_OK)return status;
    }
    return FG_OK;
}

static uint32_t prefill_pair_id(const fg_prefill_pair *pair){
    return pair->token_slot*FG_TOP_K+pair->routing_slot;
}


/* Stage the routed work and dispatch the fixed expert graph on the async
 * fence.  The decode block fast path enqueues the shared expert batch behind
 * this submission on the same queue, so the graph executes back to back with
 * it and the result readback moves off the critical path into collect. */
fg_status fg_expert_decode_submit(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err){
    if(!executor||!work||!result){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert decode arguments");return FG_ERR_ARGUMENT;}
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t rank=fg_model_rank(executor->model);
    uint32_t local_count=fg_expert_local_count(manifest,work->layer,rank);
    if(work->layer>=FG_LAYER_COUNT||work->destination_rank!=rank||
       (work->source_rank!=0u&&
        work->source_rank!=manifest->layer_owner[work->layer])||
       !work->selected_count||work->selected_count>FG_TOP_K||!local_count){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "expert work does not match rank, owner, or layer");
        return FG_ERR_MISMATCH;
    }
    for(uint32_t i=0;i<FG_EXPERT_TILES_WORDS;i++)
        executor->schedule[i]=UINT32_MAX;
    bool slots[FG_TOP_K]={0},experts[FG_EXPERT_COUNT]={0};
    for(uint32_t i=0;i<work->selected_count;i++){
        uint32_t global=work->expert_ids[i],slot=work->routing_slots[i];
        uint32_t local=fg_expert_local_index(manifest,work->layer,rank,global);
        if(global>=FG_EXPERT_COUNT||experts[global]||local>=local_count||
           slot>=FG_TOP_K||slots[slot]){
            fg_error_set(err,FG_ERR_MISMATCH,"invalid expert work route %u",i);
            return FG_ERR_MISMATCH;
        }
        executor->schedule[i*FG_Q38_DECODE_TILE_WORDS]=local;
        executor->schedule[i*FG_Q38_DECODE_TILE_WORDS+1u]=slot;
        slots[slot]=true;experts[global]=true;
    }
    fg_vk_context *vk=fg_model_vk(executor->model);
    fg_status status=fg_vk_tensor_write(executor->activation,0,
        work->activation_q8k,FG_Q8K_ACTIVATION_BYTES,err);
    if(status==FG_OK)status=fg_vk_tensor_write(executor->tiles,0,
        executor->schedule,FG_EXPERT_TILES_WORDS*sizeof(uint32_t),err);
    if(status==FG_OK)status=fg_vk_tensor_write(executor->gates,0,work->gates,
        (uint64_t)work->selected_count*4u,err);
    if(status==FG_OK&&!fg_vk_profile_active(vk)){
        if(!executor->decode_graph[work->layer]){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "rank has no fixed expert graph for layer %u",
                         work->layer);
            status=FG_ERR_MISMATCH;
        }else status=fg_vk_expert_graph_submit(
            executor->decode_graph[work->layer],err);
    }else if(status==FG_OK){
        fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
        const fg_tensor_record *gate_record=NULL,*up_record=NULL,*down_record=NULL;
        uint32_t gate_count=0u,up_count=0u,down_count=0u;
        uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
        status=expert_binding(executor,work->layer,"ffn_gate_exps",
            &gate_weight,&gate_record,&gate_count,&gate_stride,err);
        if(status==FG_OK)status=expert_binding(executor,work->layer,
            "ffn_up_exps",&up_weight,&up_record,&up_count,&up_stride,err);
        if(status==FG_OK)status=expert_binding(executor,work->layer,
            "ffn_down_exps",&down_weight,&down_record,&down_count,
            &down_stride,err);
        if(status==FG_OK&&(gate_count!=local_count||up_count!=local_count||
                           down_count!=local_count)){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "layer %u expert tensor counts disagree",work->layer);
            status=FG_ERR_MISMATCH;
        }
        if(status==FG_OK)status=fg_vk_begin(vk,err);
        uint32_t tiles=work->selected_count;
        bool fused=fg_vk_decode_experts_fusable(gate_weight,up_weight,down_weight,
            down_record->ggml_type);
        if(fused){
            if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_gate_up",err);
            if(status==FG_OK)status=fg_vk_moe_decode_gate_up(vk,executor->mid,
                gate_weight,up_weight,executor->activation,executor->tiles,640u,
                FG_HIDDEN_SIZE,gate_stride,up_stride,gate_record->ggml_type,
                up_record->ggml_type,FG_TOP_K,err);
            if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_down_reduce",err);
            if(status==FG_OK)status=fg_vk_moe_decode_down_reduce(vk,
                executor->reduced,down_weight,executor->tiles,executor->mid,
                executor->gates,FG_HIDDEN_SIZE,640u,down_stride,FG_TOP_K,
                down_record->ggml_type,err);
        }else{
            if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_gate",err);
            if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->gate,gate_weight,
                executor->activation,executor->tiles,gate_record->ggml_type,640u,
                FG_HIDDEN_SIZE,gate_stride,FG_TOP_K,FG_TOP_K,false,tiles,err);
            if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_up",err);
            if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->up,up_weight,
                executor->activation,executor->tiles,up_record->ggml_type,640u,
                FG_HIDDEN_SIZE,up_stride,FG_TOP_K,FG_TOP_K,false,tiles,err);
            if(status==FG_OK)status=fg_vk_profile_set_scope(
                vk,"expert_activation",err);
            if(status==FG_OK)status=fg_vk_swiglu(vk,executor->mid,
                executor->gate,executor->up,FG_EXPERT_MID_VALUES,err);
            if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_down",err);
            if(status==FG_OK)status=project_down(vk,executor->down,down_weight,
                executor->tiles,executor->mid,down_record,down_stride,FG_TOP_K,
                tiles,err);
            if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_reduce",err);
            if(status==FG_OK)status=fg_vk_moe_reduce(vk,executor->reduced,
                executor->down,executor->gates,executor->tiles,FG_HIDDEN_SIZE,
                work->selected_count,FG_TOP_K,err);
        }
        if(status==FG_OK){
            fg_status end_status=fg_vk_end(vk,err);
            if(end_status!=FG_OK)status=end_status;
        }
        if(status!=FG_OK&&fg_vk_batch_active(vk)){
            fg_error ignored={0};
            fg_vk_abort(vk,&ignored);
        }
    }
    if(status!=FG_OK)return status;
    memset(result,0,sizeof(*result));result->layer=work->layer;result->source_rank=(uint8_t)rank;result->destination_rank=work->source_rank;result->selected_count=1u;result->routing_slots[0]=0xFFu;result->position=work->position;return FG_OK;
}

/* The routed-reduction arena (one hidden row per token). */
fg_vk_tensor *fg_expert_reduced(fg_expert_executor *executor){
    return executor?executor->reduced:NULL;
}

fg_status fg_expert_decode_finish(fg_expert_executor *executor,fg_expert_result *result,fg_error *err){
    if(!executor||!result){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert decode finish");return FG_ERR_ARGUMENT;}
    fg_status status=fg_vk_expert_graph_wait(fg_model_vk(executor->model),err);
    if(status==FG_OK)
        status=fg_vk_tensor_read(executor->reduced,0,result->outputs[0],
                                 FG_HIDDEN_SIZE*4u,err);
    return status;
}

fg_status fg_expert_decode(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err){
    fg_status status=fg_expert_decode_submit(executor,work,result,err);
    if(status==FG_OK)status=fg_expert_decode_finish(executor,result,err);
    return status;
}

/* Chained decode fast path: the router top-10 selection, the tile schedule,
 * and the fused expert pair are recorded into the caller's active command
 * buffer, so a block owner never submits or fences on a per-layer boundary and
 * never round-trips the routed selection through the host.  The result is the
 * gate-weighted rank-local expert sum in the executor's reduced tensor. */
static fg_status chain_expert_layout(fg_expert_executor *executor,uint32_t layer,
    fg_vk_tensor **gate_weight,fg_vk_tensor **up_weight,fg_vk_tensor **down_weight,
    uint32_t *gate_stride,uint32_t *up_stride,uint32_t *down_stride,
    uint32_t *gate_type,uint32_t *up_type,uint32_t *down_type,fg_error *err){
    const fg_tensor_record *gate_record=NULL,*up_record=NULL,*down_record=NULL;
    uint32_t gate_count=0u,up_count=0u,down_count=0u;
    fg_status status=expert_binding(executor,layer,"ffn_gate_exps",
        gate_weight,&gate_record,&gate_count,gate_stride,err);
    if(status==FG_OK)status=expert_binding(executor,layer,"ffn_up_exps",
        up_weight,&up_record,&up_count,up_stride,err);
    if(status==FG_OK)status=expert_binding(executor,layer,"ffn_down_exps",
        down_weight,&down_record,&down_count,down_stride,err);
    if(status!=FG_OK)return status;
    if(gate_count!=FG_EXPERT_COUNT||up_count!=FG_EXPERT_COUNT||
       down_count!=FG_EXPERT_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "chained decode layer %u is not a single-owner expert slab",layer);
        return FG_ERR_MISMATCH;
    }
    *gate_type=gate_record->ggml_type;*up_type=up_record->ggml_type;
    *down_type=down_record->ggml_type;
    if(!fg_vk_decode_experts_fusable(*gate_weight,*up_weight,*down_weight,*down_type)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "chained decode layer %u has no fusable expert pair",layer);
        return FG_ERR_UNAVAILABLE;
    }
    return FG_OK;
}

fg_status fg_expert_decode_chain_ready(fg_expert_executor *executor,uint32_t layer,
    fg_error *err){
    if(!executor||layer>=FG_LAYER_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid chained expert layer");return FG_ERR_ARGUMENT;
    }
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t rank=fg_model_rank(executor->model);
    if(fg_expert_local_count(manifest,layer,rank)!=FG_EXPERT_COUNT){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "chained decode requires a single-owner layer %u",layer);
        return FG_ERR_UNAVAILABLE;
    }
    fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
    uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
    uint32_t gate_type=0u,up_type=0u,down_type=0u;
    return chain_expert_layout(executor,layer,&gate_weight,&up_weight,&down_weight,
        &gate_stride,&up_stride,&down_stride,&gate_type,&up_type,&down_type,err);
}

fg_status fg_expert_decode_chain(fg_expert_executor *executor,uint32_t layer,
    const fg_vk_tensor *activation,const fg_vk_tensor *router_logits,
    fg_vk_tensor **reduced,fg_error *err){
    if(!executor||!activation||!router_logits||!reduced){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid chained expert arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
    uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
    uint32_t gate_type=0u,up_type=0u,down_type=0u;
    fg_status status=chain_expert_layout(executor,layer,&gate_weight,&up_weight,
        &down_weight,&gate_stride,&up_stride,&down_stride,&gate_type,&up_type,
        &down_type,err);
    if(status!=FG_OK)return status;
    fg_vk_context *vk=fg_model_vk(executor->model);
    status=fg_vk_router_top10(vk,executor->selected,executor->gates,router_logits,
        FG_EXPERT_COUNT,1u,err);
    if(status==FG_OK)status=fg_vk_decode_tile_schedule(vk,executor->tiles,
        executor->selected,err);
    if(status==FG_OK)status=fg_vk_moe_decode_gate_up(vk,executor->mid,gate_weight,
        up_weight,activation,executor->tiles,640u,FG_HIDDEN_SIZE,gate_stride,
        up_stride,gate_type,up_type,FG_TOP_K,err);
    if(status==FG_OK)status=fg_vk_moe_decode_down_reduce(vk,executor->reduced,
        down_weight,executor->tiles,executor->mid,executor->gates,FG_HIDDEN_SIZE,
        640u,down_stride,FG_TOP_K,down_type,err);
    if(status==FG_OK)*reduced=executor->reduced;
    return status;
}

/* Single-token chained decode that writes into a caller-provided tensor (a
 * row of the batch arena), used by the test-only serial-expert A/B. */
fg_status fg_expert_decode_chain_into(fg_expert_executor *executor,uint32_t layer,
    const fg_vk_tensor *activation,const fg_vk_tensor *router_logits,
    fg_vk_tensor *reduced,fg_error *err){
    if(!executor||!activation||!router_logits||!reduced){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid chained expert arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
    uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
    uint32_t gate_type=0u,up_type=0u,down_type=0u;
    fg_status status=chain_expert_layout(executor,layer,&gate_weight,&up_weight,
        &down_weight,&gate_stride,&up_stride,&down_stride,&gate_type,&up_type,
        &down_type,err);
    if(status!=FG_OK)return status;
    fg_vk_context *vk=fg_model_vk(executor->model);
    status=fg_vk_router_top10(vk,executor->selected,executor->gates,router_logits,
        FG_EXPERT_COUNT,1u,err);
    if(status==FG_OK)status=fg_vk_decode_tile_schedule(vk,executor->tiles,
        executor->selected,err);
    if(status==FG_OK)status=fg_vk_moe_decode_gate_up(vk,executor->mid,gate_weight,
        up_weight,activation,executor->tiles,640u,FG_HIDDEN_SIZE,gate_stride,
        up_stride,gate_type,up_type,FG_TOP_K,err);
    if(status==FG_OK)status=fg_vk_moe_decode_down_reduce(vk,reduced,down_weight,
        executor->tiles,executor->mid,executor->gates,FG_HIDDEN_SIZE,640u,
        down_stride,FG_TOP_K,down_type,err);
    return status;
}

/* Batch-2 chained decode: GPU routing for two tokens, one token-tagged union
 * schedule, one fused gate/up dispatch and one down/reduce dispatch.  The
 * result is the two gate-weighted rank-local expert sums, one hidden row per
 * token, in the executor's reduced tensor. */
fg_status fg_expert_decode_chain_b2(fg_expert_executor *executor,uint32_t layer,
    const fg_vk_tensor *activation,const fg_vk_tensor *router_logits,
    fg_vk_tensor **reduced,fg_error *err){
    if(!executor||!activation||!router_logits||!reduced){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid chained batch-2 expert arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
    uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
    uint32_t gate_type=0u,up_type=0u,down_type=0u;
    fg_status status=chain_expert_layout(executor,layer,&gate_weight,&up_weight,
        &down_weight,&gate_stride,&up_stride,&down_stride,&gate_type,&up_type,
        &down_type,err);
    if(status!=FG_OK)return status;
    fg_vk_context *vk=fg_model_vk(executor->model);
    uint32_t slots=2u*FG_TOP_K;
    status=fg_vk_router_top10(vk,executor->selected,executor->gates,router_logits,
        FG_EXPERT_COUNT,2u,err);
    if(status==FG_OK)status=fg_vk_decode_tile_schedule_b2(vk,executor->tiles,
        executor->selected,err);
    if(status==FG_OK)status=fg_vk_moe_decode_gate_up_b2(vk,executor->mid,gate_weight,
        up_weight,activation,executor->tiles,640u,FG_HIDDEN_SIZE,gate_stride,
        up_stride,gate_type,up_type,slots,err);
    if(status==FG_OK)status=fg_vk_moe_decode_down_reduce_b2(vk,executor->reduced,
        down_weight,executor->tiles,executor->mid,executor->gates,FG_HIDDEN_SIZE,
        640u,down_stride,slots,down_type,err);
    if(status==FG_OK)*reduced=executor->reduced;
    return status;
}

fg_status fg_expert_prefill_enqueue(fg_expert_executor *executor,const fg_prefill_work *work,fg_prefill_result *result,fg_prefill_result_pair *pair_storage,uint32_t pair_capacity,float *output_storage,uint64_t output_capacity_values,fg_error *err){
    if(!executor||!work||!work->pairs||!work->activations_q8k||!result||!pair_storage||!output_storage){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert prefill arguments");return FG_ERR_ARGUMENT;}
    const fg_manifest *manifest=fg_model_manifest(executor->model);uint32_t rank=fg_model_rank(executor->model);
    uint32_t local_count=fg_expert_local_count(manifest,work->layer,rank);
    if(work->layer>=FG_LAYER_COUNT||work->destination_rank!=rank||
       (work->source_rank!=0u&&work->source_rank!=manifest->layer_owner[work->layer])||
       !work->token_count||work->token_count>executor->max_tokens||
       !work->pair_count||work->pair_count>executor->max_pairs||
       pair_capacity<work->pair_count||
       output_capacity_values<(uint64_t)work->token_count*FG_HIDDEN_SIZE||
       !local_count){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "expert prefill work does not match rank, owner, or configured microbatch");
        return FG_ERR_MISMATCH;
    }

    uint32_t counts[FG_EXPERT_COUNT]={0};
    uint16_t local_ids[FG_EXPERT_COUNT];uint16_t next_local=0u;
    for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
        local_ids[expert]=manifest->expert_rank[work->layer][expert]==rank?
            next_local++:UINT16_MAX;
    bool seen[FG_PREFILL_MAX_TOKENS][FG_TOP_K]={{false}};
    fg_status status=FG_OK;
    for(uint32_t i=0;i<work->pair_count;i++){
        const fg_prefill_pair *pair=&work->pairs[i];
        uint32_t local=pair->expert_id<FG_EXPERT_COUNT?
            local_ids[pair->expert_id]:UINT16_MAX;
        if(pair->token_slot>=work->token_count||
           pair->routing_slot>=FG_TOP_K||pair->expert_id>=FG_EXPERT_COUNT||
           seen[pair->token_slot][pair->routing_slot]||local>=local_count||
           !isfinite(pair->gate)){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "invalid expert prefill route %u",i);
            status=FG_ERR_MISMATCH;break;
        }
        executor->locals[i]=(uint16_t)local;
        counts[local]++;
        seen[pair->token_slot][pair->routing_slot]=true;
    }
    fg_vk_tensor *gate_weight=NULL,*up_weight=NULL,*down_weight=NULL;
    const fg_tensor_record *gate_record=NULL,*up_record=NULL,*down_record=NULL;
    uint32_t gate_count=0u,up_count=0u,down_count=0u;
    uint32_t gate_stride=0u,up_stride=0u,down_stride=0u;
    if(status==FG_OK)status=expert_binding(executor,work->layer,
        "ffn_gate_exps",&gate_weight,&gate_record,&gate_count,&gate_stride,err);
    if(status==FG_OK)status=expert_binding(executor,work->layer,
        "ffn_up_exps",&up_weight,&up_record,&up_count,&up_stride,err);
    if(status==FG_OK)status=expert_binding(executor,work->layer,
        "ffn_down_exps",&down_weight,&down_record,&down_count,&down_stride,err);
    if(status==FG_OK&&(gate_count!=local_count||up_count!=local_count||
                       down_count!=local_count)){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "layer %u expert tensor counts disagree",work->layer);
        status=FG_ERR_MISMATCH;
    }
    bool grouped=status==FG_OK&&
        fg_vk_tensor_get_format(gate_weight)==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&
        fg_vk_tensor_get_format(up_weight)==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&
        (fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED||
         fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED||
         (down_record->ggml_type==8u&&
          fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_DEFAULT));
    bool single_pair=!grouped&&status==FG_OK&&
        (fg_vk_tensor_get_format(gate_weight)==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||
         fg_vk_tensor_get_format(up_weight)==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||
         fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED);
    uint32_t width=grouped?FG_VK_PREFILL_PAIR_TILE:single_pair?1u:8u;
    uint32_t words=grouped?FG_Q38_PREFILL_TILE_WORDS:FG_Q38_DECODE_TILE_WORDS;
    uint32_t starts[FG_EXPERT_COUNT]={0},used[FG_EXPERT_COUNT]={0},tile_count=0u;
    if(status==FG_OK){
        for(uint32_t local=0;local<local_count;local++){
            starts[local]=tile_count;
            tile_count+=(counts[local]+width-1u)/width;
        }
        for(uint32_t i=0;i<tile_count*words;i++)executor->schedule[i]=UINT32_MAX;
        for(uint32_t pair=0;pair<work->pair_count;pair++){
            uint32_t local=executor->locals[pair],index=used[local]++;
            uint32_t tile=starts[local]+index/width;
            executor->schedule[tile*words]=local;
            executor->schedule[tile*words+1u+index%width]=prefill_pair_id(&work->pairs[pair]);
        }
    }
    fg_vk_context *vk=fg_model_vk(executor->model);uint32_t dense_pairs=(uint32_t)work->token_count*FG_TOP_K;
    if(status==FG_OK)status=fg_vk_tensor_write(executor->activation,0,work->activations_q8k,(uint64_t)work->token_count*FG_Q8K_ACTIVATION_BYTES,err);
    if(status==FG_OK)status=fg_vk_tensor_write(
        executor->tiles,0,executor->schedule,
        (uint64_t)tile_count*words*4u,err);
    if(status==FG_OK){
        float *dense_gates=fg_vk_tensor_map(executor->gates);
        memset(dense_gates,0,(size_t)dense_pairs*4u);
        for(uint32_t i=0;i<work->pair_count;i++)dense_gates[prefill_pair_id(&work->pairs[i])]=work->pairs[i].gate;
    }
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(grouped){
        if(status==FG_OK)status=fg_vk_moe_kquant_cooked_grouped(vk,
            executor->gate,gate_weight,executor->activation,executor->tiles,
            gate_record->ggml_type,640u,FG_HIDDEN_SIZE,gate_stride,
            local_count,work->token_count,tile_count,err);
        if(status==FG_OK)status=fg_vk_moe_kquant_cooked_grouped(vk,
            executor->up,up_weight,executor->activation,executor->tiles,
            up_record->ggml_type,640u,FG_HIDDEN_SIZE,up_stride,
            local_count,work->token_count,tile_count,err);
        if(status==FG_OK)status=fg_vk_swiglu(vk,executor->mid,
            executor->gate,executor->up,dense_pairs*640u,err);
        if(status==FG_OK&&down_record->ggml_type==7u)
            status=fg_vk_moe_q5_1_down_cooked_grouped(vk,executor->down,
                down_weight,executor->tiles,executor->mid,FG_HIDDEN_SIZE,
                640u,down_stride,local_count,work->token_count,tile_count,err);
        else if(status==FG_OK&&
                fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED)
            status=fg_vk_moe_q8_0_down_cooked_grouped(vk,executor->down,
                down_weight,executor->tiles,executor->mid,FG_HIDDEN_SIZE,
                640u,down_stride,local_count,work->token_count,tile_count,err);
        else if(status==FG_OK)
            status=fg_vk_moe_q8_0_down_grouped(vk,executor->down,down_weight,
                executor->tiles,executor->mid,FG_HIDDEN_SIZE,640u,down_stride,
                local_count,work->token_count,tile_count,err);
    }else{
        if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->gate,gate_weight,
            executor->activation,executor->tiles,gate_record->ggml_type,640u,
            FG_HIDDEN_SIZE,gate_stride,FG_TOP_K,dense_pairs,false,tile_count,err);
        if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->up,up_weight,
            executor->activation,executor->tiles,up_record->ggml_type,640u,
            FG_HIDDEN_SIZE,up_stride,FG_TOP_K,dense_pairs,false,tile_count,err);
        if(status==FG_OK)status=fg_vk_swiglu(vk,executor->mid,
            executor->gate,executor->up,dense_pairs*640u,err);
        if(status==FG_OK)status=project_down(vk,executor->down,down_weight,
            executor->tiles,executor->mid,down_record,down_stride,dense_pairs,tile_count,err);
    }
    /* Gate projection storage is dead after SwiGLU and has room for token sums. */
    if(status==FG_OK)status=fg_vk_moe_prefill_shard_reduce(vk,executor->result_readback,
        executor->down,executor->gates,work->token_count,err);
    if(status!=FG_OK&&fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    return status;
}

fg_status fg_expert_prefill_finish(fg_expert_executor *executor,const fg_prefill_work *work,
    fg_prefill_result *result,fg_prefill_result_pair *pair_storage,float *output_storage,
    fg_error *err){
    if(!executor||!work||!result||!pair_storage||!output_storage){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert prefill finish arguments");return FG_ERR_ARGUMENT;}
    fg_vk_context *vk=fg_model_vk(executor->model);uint32_t rank=fg_model_rank(executor->model);
    fg_status status=fg_vk_end(vk,err);
    if(status!=FG_OK&&fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    if(status==FG_OK){
        memcpy(output_storage,fg_vk_tensor_const_map(executor->result_readback),
               (uint64_t)work->token_count*FG_HIDDEN_SIZE*4u);
        memset(result,0,sizeof(*result));result->layer=work->layer;
        result->source_rank=(uint8_t)rank;result->destination_rank=work->source_rank;
        result->contributor_mask=(uint8_t)(1u<<rank);
        result->first_position=work->first_position;result->token_count=work->token_count;
        result->pair_count=work->pair_count;result->pairs=pair_storage;
        result->outputs=output_storage;
        for(uint32_t i=0u;i<work->pair_count;i++){
            pair_storage[i].token_slot=work->pairs[i].token_slot;
            pair_storage[i].routing_slot=work->pairs[i].routing_slot;
        }
    }
    return status;
}

fg_status fg_expert_prefill(fg_expert_executor *executor,const fg_prefill_work *work,fg_prefill_result *result,fg_prefill_result_pair *pair_storage,uint32_t pair_capacity,float *output_storage,uint64_t output_capacity_values,fg_error *err){
    fg_status status=fg_expert_prefill_enqueue(executor,work,result,pair_storage,pair_capacity,output_storage,output_capacity_values,err);
    if(status==FG_OK)status=fg_expert_prefill_finish(executor,work,result,pair_storage,output_storage,err);
    return status;
}
