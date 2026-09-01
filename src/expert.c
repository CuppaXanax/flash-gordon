#include "fg_expert.h"
#include "fg_vk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FG_EXPERT_TILES_WORDS (FG_TOP_K * FG_Q38_DECODE_TILE_WORDS)
#define FG_EXPERT_MID_VALUES (FG_TOP_K * 640u)
#define FG_EXPERT_OUTPUT_VALUES (FG_TOP_K * FG_HIDDEN_SIZE)
_Static_assert(FG_Q38_DECODE_TILE_WORDS==9u,
               "decode and EP expert schedules use nine words per pair");

struct fg_expert_executor {
    fg_model *model;
    fg_vk_tensor *activation,*tiles,*gates,*unit_gates;
    fg_vk_tensor *gate,*up,*mid,*down,*reduced;
    fg_vk_expert_graph *decode_graph[FG_LAYER_COUNT];
    uint32_t max_tokens,max_pairs;
    uint16_t *locals;
    uint32_t *schedule;
};

static fg_status create_decode_graphs(fg_expert_executor *executor,fg_error *err);

static fg_status create_scratch(fg_expert_executor *executor,fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);executor->max_tokens=manifest->prefill_microbatch;executor->max_pairs=executor->max_tokens*FG_TOP_K;
    if(!executor->max_tokens||executor->max_tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds expert executor limit");return FG_ERR_MISMATCH;}
    bool pipeline=manifest->execution_mode==FG_EXECUTION_PIPELINE;
    fg_vk_context *vk=fg_model_vk(executor->model);fg_status status=fg_vk_tensor_create(vk,(uint64_t)executor->max_tokens*FG_Q8K_ACTIVATION_BYTES,&executor->activation,err);
    uint64_t tile_bytes=(uint64_t)executor->max_pairs*
        FG_Q38_DECODE_TILE_WORDS*4u;
    if(status==FG_OK)status=fg_vk_tensor_create(vk,tile_bytes,
                                                &executor->tiles,err);
    if(status==FG_OK&&fg_vk_tensor_bytes(executor->tiles)!=tile_bytes){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "expert decode tile allocation violates nine-word geometry");
        status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK)status=fg_vk_tensor_create(vk,FG_TOP_K*4u,&executor->gates,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->gate,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->up,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->mid,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*FG_HIDDEN_SIZE*4u,&executor->down,err);
    if(status==FG_OK&&pipeline)status=fg_vk_tensor_create(
        vk,FG_TOP_K*4u,&executor->unit_gates,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,
        (uint64_t)(pipeline?FG_TOP_K:1u)*FG_HIDDEN_SIZE*4u,
        &executor->reduced,err);
    if(status==FG_OK){
        executor->locals=malloc((size_t)executor->max_pairs*sizeof(*executor->locals));
        executor->schedule=malloc((size_t)executor->max_pairs*
                                  FG_Q38_DECODE_TILE_WORDS*
                                  sizeof(*executor->schedule));
        if(!executor->locals||!executor->schedule){
            fg_error_set(err,FG_ERR_OOM,"allocate expert schedules");
            status=FG_ERR_OOM;
        }
    }
    if(status==FG_OK&&pipeline){
        float unit[FG_TOP_K]={1.0f};
        status=fg_vk_tensor_write(executor->unit_gates,0,unit,sizeof(unit),err);
        if(status==FG_OK)memset(fg_vk_tensor_map(executor->reduced),0,
                                (size_t)fg_vk_tensor_bytes(executor->reduced));
    }
    return status;
}

fg_status fg_expert_executor_create(fg_expert_executor **out,fg_model *model,fg_error *err){if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert executor arguments");return FG_ERR_ARGUMENT;}*out=NULL;fg_expert_executor *executor=calloc(1,sizeof(*executor));if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate expert executor");return FG_ERR_OOM;}executor->model=model;fg_status status=create_scratch(executor,err);if(status==FG_OK)status=create_decode_graphs(executor,err);if(status!=FG_OK){fg_expert_executor_destroy(executor);return status;}*out=executor;return FG_OK;}
void fg_expert_executor_destroy(fg_expert_executor *executor){if(!executor)return;for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)fg_vk_expert_graph_destroy(executor->decode_graph[layer]);free(executor->schedule);free(executor->locals);fg_vk_tensor_destroy(executor->reduced);fg_vk_tensor_destroy(executor->down);fg_vk_tensor_destroy(executor->mid);fg_vk_tensor_destroy(executor->up);fg_vk_tensor_destroy(executor->gate);fg_vk_tensor_destroy(executor->unit_gates);fg_vk_tensor_destroy(executor->gates);fg_vk_tensor_destroy(executor->tiles);fg_vk_tensor_destroy(executor->activation);free(executor);}

bool fg_expert_uses_rank_suffix(const fg_manifest *manifest){
    return !manifest||manifest->execution_mode!=FG_EXECUTION_PIPELINE;
}

uint32_t fg_expert_local_count(const fg_manifest *manifest,uint32_t layer,
                               uint32_t rank){
    if(!manifest||layer>=FG_LAYER_COUNT||rank>=FG_RANK_COUNT)return 0u;
    if(manifest->execution_mode==FG_EXECUTION_PIPELINE)
        return manifest->layer_owner[layer]==rank?FG_EXPERT_COUNT:0u;
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
    if(manifest->execution_mode==FG_EXECUTION_PIPELINE)return global;
    uint32_t local=0u;
    for(uint32_t expert=0;expert<global;expert++)
        if(manifest->expert_rank[layer][expert]==rank)local++;
    return local;
}

static bool tensor_name(char output[FG_TENSOR_NAME_MAX],
                        const fg_manifest *manifest,uint32_t layer,
                        const char *family,uint32_t rank){
    int length=fg_expert_uses_rank_suffix(manifest)?
        snprintf(output,FG_TENSOR_NAME_MAX,"blk.%u.%s.weight.rank%u",
                 layer,family,rank):
        snprintf(output,FG_TENSOR_NAME_MAX,"blk.%u.%s.weight",layer,family);
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
    if(!tensor_name(name,manifest,layer,family,rank)){
        fg_error_set(err,FG_ERR_LIMIT,"expert tensor name overflow");
        return FG_ERR_LIMIT;
    }
    *weight=fg_model_tensor(executor->model,name);
    *record=fg_model_tensor_record(executor->model,name);
    *local_count=fg_expert_local_count(manifest,layer,rank);
    uint32_t expected=manifest->execution_mode==FG_EXECUTION_PIPELINE?
        FG_EXPERT_COUNT:FG_EXPERTS_PER_RANK;
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

static bool pipeline_k_quant_cooked(const fg_vk_tensor *weight,
                                    const fg_tensor_record *record){
    return weight&&record&&
        fg_vk_tensor_get_format(weight)==
            FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&
        (record->ggml_type==12u||record->ggml_type==13u);
}

static bool pipeline_down_supported(const fg_vk_tensor *weight,
                                    const fg_tensor_record *record){
    if(!weight||!record)return false;
    fg_vk_tensor_format format=fg_vk_tensor_get_format(weight);
    if(record->ggml_type==7u)
        return format==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED;
    if(record->ggml_type==8u)
        return format==FG_VK_TENSOR_FORMAT_DEFAULT;
    return false;
}

static fg_status create_decode_graphs(fg_expert_executor *executor,fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t rank=fg_model_rank(executor->model);
    fg_vk_context *vk=fg_model_vk(executor->model);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        char gate_name[FG_TENSOR_NAME_MAX],up_name[FG_TENSOR_NAME_MAX];
        char down_name[FG_TENSOR_NAME_MAX];
        if(!tensor_name(gate_name,manifest,layer,"ffn_gate_exps",rank)||
           !tensor_name(up_name,manifest,layer,"ffn_up_exps",rank)||
           !tensor_name(down_name,manifest,layer,"ffn_down_exps",rank)){
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

static void compact_prefill_outputs(const fg_prefill_work *work,float *storage){
    bool done[FG_PREFILL_MAX_PAIRS]={0};
    bool referenced[FG_PREFILL_MAX_PAIRS]={0};
    uint32_t chain[FG_PREFILL_MAX_PAIRS];
    float temporary[FG_HIDDEN_SIZE],swap[FG_HIDDEN_SIZE];
    for(uint32_t i=0u;i<work->pair_count;i++){
        uint32_t source=prefill_pair_id(&work->pairs[i]);
        if(source<work->pair_count)referenced[source]=true;
    }
    for(uint32_t pass=0u;pass<2u;pass++)for(uint32_t start=0u;
        start<work->pair_count;start++){
        if(done[start]||(pass==0u&&referenced[start]))continue;
        uint32_t length=0u,current=start;bool cycle=false;
        for(;;){
            chain[length++]=current;
            uint32_t source=prefill_pair_id(&work->pairs[current]);
            if(source>=work->pair_count)break;
            bool repeated=false;
            for(uint32_t i=0u;i<length;i++)
                if(chain[i]==source){repeated=true;break;}
            if(repeated){cycle=true;break;}
            current=source;
        }
        uint32_t source=cycle?chain[0]:
            prefill_pair_id(&work->pairs[chain[length-1u]]);
        memcpy(temporary,storage+(uint64_t)source*FG_HIDDEN_SIZE,
               FG_HIDDEN_SIZE*sizeof(float));
        for(uint32_t i=length;i>0u;i--){
            float *destination=storage+
                (uint64_t)chain[i-1u]*FG_HIDDEN_SIZE;
            memcpy(swap,destination,FG_HIDDEN_SIZE*sizeof(float));
            memcpy(destination,temporary,FG_HIDDEN_SIZE*sizeof(float));
            memcpy(temporary,swap,FG_HIDDEN_SIZE*sizeof(float));
            done[chain[i-1u]]=true;
        }
    }
}

fg_status fg_expert_decode(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err){
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
        }else status=fg_vk_expert_graph_execute(
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
        if(status==FG_OK)status=fg_vk_swiglu(vk,executor->mid,executor->gate,
            executor->up,FG_EXPERT_MID_VALUES,err);
        if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_down",err);
        if(status==FG_OK)status=project_down(vk,executor->down,down_weight,
            executor->tiles,executor->mid,down_record,down_stride,FG_TOP_K,
            tiles,err);
        if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"expert_reduce",err);
        if(status==FG_OK)status=fg_vk_moe_reduce(vk,executor->reduced,
            executor->down,executor->gates,executor->tiles,FG_HIDDEN_SIZE,
            work->selected_count,FG_TOP_K,err);
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
    memset(result,0,sizeof(*result));result->layer=work->layer;result->source_rank=(uint8_t)rank;result->destination_rank=work->source_rank;result->selected_count=1u;result->routing_slots[0]=0xFFu;result->position=work->position;memcpy(result->outputs[0],fg_vk_tensor_map(executor->reduced),FG_HIDDEN_SIZE*4u);return FG_OK;
}

fg_status fg_expert_prefill(fg_expert_executor *executor,const fg_prefill_work *work,fg_prefill_result *result,fg_prefill_result_pair *pair_storage,uint32_t pair_capacity,float *output_storage,uint64_t output_capacity_values,fg_error *err){
    if(!executor||!work||!result||!pair_storage||!output_storage){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert prefill arguments");return FG_ERR_ARGUMENT;}
    const fg_manifest *manifest=fg_model_manifest(executor->model);uint32_t rank=fg_model_rank(executor->model);
    uint32_t local_count=fg_expert_local_count(manifest,work->layer,rank);
    if(work->layer>=FG_LAYER_COUNT||work->destination_rank!=rank||
       (work->source_rank!=0u&&work->source_rank!=manifest->layer_owner[work->layer])||
       !work->token_count||work->token_count>executor->max_tokens||
       !work->pair_count||work->pair_count>executor->max_pairs||
       pair_capacity<work->pair_count||
       output_capacity_values<(uint64_t)work->pair_count*FG_HIDDEN_SIZE||
       !local_count){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "expert prefill work does not match rank, owner, or configured microbatch");
        return FG_ERR_MISMATCH;
    }

    uint32_t counts[FG_EXPERT_COUNT]={0};
    bool seen[FG_PREFILL_MAX_TOKENS][FG_TOP_K]={{false}};
    fg_status status=FG_OK;
    for(uint32_t i=0;i<work->pair_count;i++){
        const fg_prefill_pair *pair=&work->pairs[i];
        uint32_t local=fg_expert_local_index(manifest,work->layer,rank,
                                             pair->expert_id);
        if(pair->token_slot>=work->token_count||
           pair->routing_slot>=FG_TOP_K||pair->expert_id>=FG_EXPERT_COUNT||
           seen[pair->token_slot][pair->routing_slot]||local>=local_count){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "invalid expert prefill route %u",i);
            status=FG_ERR_MISMATCH;break;
        }
        executor->locals[i]=(uint16_t)local;
        counts[local]++;
        seen[pair->token_slot][pair->routing_slot]=true;
    }
    uint32_t tile_count=0u;
    if(status==FG_OK){
        for(uint32_t local=0;local<local_count;local++)
            tile_count+=(counts[local]+7u)/8u;
        for(uint32_t i=0;i<tile_count*FG_Q38_DECODE_TILE_WORDS;i++)
            executor->schedule[i]=UINT32_MAX;
        uint32_t tile=0u;
        for(uint32_t local=0;local<local_count;local++)if(counts[local]){
            uint32_t slot=0u;
            executor->schedule[tile*FG_Q38_DECODE_TILE_WORDS]=local;
            for(uint32_t pair=0;pair<work->pair_count;pair++)
                if(executor->locals[pair]==local){
                    if(slot==8u){
                        tile++;slot=0u;
                        executor->schedule[tile*FG_Q38_DECODE_TILE_WORDS]=local;
                    }
                    executor->schedule[tile*FG_Q38_DECODE_TILE_WORDS+1u+slot]=
                        work->pairs[pair].token_slot*FG_TOP_K+
                        work->pairs[pair].routing_slot;
                    slot++;
                }
            tile++;
        }
        if(tile!=tile_count){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "expert-major prefill schedule count mismatch");
            status=FG_ERR_MISMATCH;
        }
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
    bool cooked=status==FG_OK&&
        fg_vk_tensor_get_format(gate_weight)==
            FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&
        fg_vk_tensor_get_format(up_weight)==
            FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED&&
        fg_vk_tensor_get_format(down_weight)==
            FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED;
    if(cooked){
        for(uint32_t i=0;i<work->pair_count*FG_Q38_DECODE_TILE_WORDS;i++)
            executor->schedule[i]=UINT32_MAX;
        for(uint32_t i=0;i<work->pair_count;i++){
            executor->schedule[i*FG_Q38_DECODE_TILE_WORDS]=executor->locals[i];
            executor->schedule[i*FG_Q38_DECODE_TILE_WORDS+1u]=
                work->pairs[i].token_slot*FG_TOP_K+
                work->pairs[i].routing_slot;
        }
        tile_count=work->pair_count;
    }
    fg_vk_context *vk=fg_model_vk(executor->model);uint32_t dense_pairs=(uint32_t)work->token_count*FG_TOP_K;
    if(status==FG_OK)status=fg_vk_tensor_write(executor->activation,0,work->activations_q8k,(uint64_t)work->token_count*FG_Q8K_ACTIVATION_BYTES,err);
    if(status==FG_OK)status=fg_vk_tensor_write(
        executor->tiles,0,executor->schedule,
        (uint64_t)tile_count*FG_Q38_DECODE_TILE_WORDS*4u,err);
    if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->gate,gate_weight,executor->activation,executor->tiles,gate_record->ggml_type,640u,FG_HIDDEN_SIZE,gate_stride,FG_TOP_K,dense_pairs,false,tile_count,err);
    if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->up,up_weight,executor->activation,executor->tiles,up_record->ggml_type,640u,FG_HIDDEN_SIZE,up_stride,FG_TOP_K,dense_pairs,false,tile_count,err);
    if(status==FG_OK){status=fg_vk_swiglu(vk,executor->mid,executor->gate,executor->up,(uint32_t)((uint64_t)dense_pairs*640u),err);}
    if(status==FG_OK)status=project_down(vk,executor->down,down_weight,executor->tiles,executor->mid,down_record,down_stride,dense_pairs,tile_count,err);
    if(status==FG_OK){
        const float *host=fg_vk_tensor_map(executor->down);
        /* Coordinator aliases down to result storage; compact after submission. */
        if(host==(const float *)output_storage)
            compact_prefill_outputs(work,output_storage);
        else for(uint32_t i=0u;i<work->pair_count;i++){
            uint32_t pair_id=prefill_pair_id(&work->pairs[i]);
            memcpy(output_storage+(uint64_t)i*FG_HIDDEN_SIZE,
                   host+(uint64_t)pair_id*FG_HIDDEN_SIZE,
                   FG_HIDDEN_SIZE*sizeof(float));
        }
        memset(result,0,sizeof(*result));result->layer=work->layer;
        result->source_rank=(uint8_t)rank;result->destination_rank=work->source_rank;
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

static fg_status expert_pipeline_grouped(fg_expert_executor *executor,
                                     uint32_t layer,uint16_t token_count,
                                     const fg_vk_tensor *activation_q8k,
                                     const fg_vk_tensor *tiles,
                                     fg_vk_tensor **output,fg_error *err){
    if(!executor||!activation_q8k||!tiles||!output||!token_count||
       token_count>executor->max_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid pipeline grouped expert execution");
        return FG_ERR_ARGUMENT;
    }
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t rank=fg_model_rank(executor->model);
    if(manifest->execution_mode!=FG_EXECUTION_PIPELINE||
       layer>=FG_LAYER_COUNT||manifest->layer_owner[layer]!=rank){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline grouped expert execution is not stage-local");
        return FG_ERR_MISMATCH;
    }
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
    if(status==FG_OK&&(gate_count!=FG_EXPERT_COUNT||
                       up_count!=FG_EXPERT_COUNT||
                       down_count!=FG_EXPERT_COUNT||
                       fg_vk_tensor_get_format(gate_weight)!=
                           FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||
                       fg_vk_tensor_get_format(up_weight)!=
                           FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED)){
        fg_error_set(err,FG_ERR_FORMAT,
                     "pipeline grouped experts require cooked stage-local gate/up weights");
        status=FG_ERR_FORMAT;
    }
    fg_vk_context *vk=fg_model_vk(executor->model);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_moe_kquant_cooked_grouped(vk,
        executor->gate,gate_weight,activation_q8k,tiles,gate_record->ggml_type,
        640u,FG_HIDDEN_SIZE,gate_stride,gate_count,token_count,err);
    if(status==FG_OK)status=fg_vk_moe_kquant_cooked_grouped(vk,
        executor->up,up_weight,activation_q8k,tiles,up_record->ggml_type,
        640u,FG_HIDDEN_SIZE,up_stride,up_count,token_count,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,executor->mid,executor->gate,
        executor->up,(uint32_t)token_count*FG_TOP_K*640u,err);
    if(status==FG_OK&&fg_vk_tensor_get_format(down_weight)==
                         FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED)
        status=fg_vk_moe_q5_1_down_cooked_grouped(vk,executor->down,
            down_weight,tiles,executor->mid,FG_HIDDEN_SIZE,640u,down_stride,
            down_count,token_count,err);
    else if(status==FG_OK&&down_record->ggml_type==8u)
        status=fg_vk_moe_q8_0_down_grouped(vk,executor->down,down_weight,tiles,
            executor->mid,FG_HIDDEN_SIZE,640u,down_stride,down_count,
            token_count,err);
    else if(status==FG_OK){
        fg_error_set(err,FG_ERR_FORMAT,
                     "pipeline grouped experts require cooked Q5_1 or Q8_0 down weights");
        status=FG_ERR_FORMAT;
    }
    if(status==FG_OK)status=fg_vk_end(vk,err);
    else if(fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    if(status==FG_OK)*output=executor->down;
    return status;
}

fg_status fg_expert_prefill_pipeline(fg_expert_executor *executor,uint32_t layer,
                                     uint16_t token_count,
                                     const fg_vk_tensor *activation_q8k,
                                     const fg_vk_tensor *tiles,
                                     fg_vk_tensor **output,fg_error *err){
    if(token_count<2u){
        fg_error_set(err,FG_ERR_ARGUMENT,"pipeline expert prefill requires at least two tokens");
        return FG_ERR_ARGUMENT;
    }
    return expert_pipeline_grouped(executor,layer,token_count,activation_q8k,
                                   tiles,output,err);
}

fg_status fg_expert_decode_pipeline(fg_expert_executor *executor,uint32_t layer,
                                    const fg_vk_tensor *activation_q8k,
                                    const fg_vk_tensor *selected,
                                    const fg_vk_tensor *gates,
                                    const fg_vk_tensor *shared_output,
                                    const fg_vk_tensor *shared_logit,
                                    fg_vk_tensor **output,fg_error *err){
    if(!executor||!activation_q8k||!selected||!gates||!shared_output||
       !shared_logit||!output||
       !fg_vk_batch_active(fg_model_vk(executor->model))){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline expert decode requires an active stage batch");
        return FG_ERR_MISMATCH;
    }
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t rank=fg_model_rank(executor->model);
    if(manifest->execution_mode!=FG_EXECUTION_PIPELINE||
       layer>=FG_LAYER_COUNT||manifest->layer_owner[layer]!=rank){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline expert decode is not stage-local");
        return FG_ERR_MISMATCH;
    }
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
    if(status==FG_OK&&(gate_count!=FG_EXPERT_COUNT||
                       up_count!=FG_EXPERT_COUNT||
                       down_count!=FG_EXPERT_COUNT||
                       !pipeline_k_quant_cooked(gate_weight,gate_record)||
                       !pipeline_k_quant_cooked(up_weight,up_record)||
                       !pipeline_down_supported(down_weight,down_record))){
        fg_error_set(err,FG_ERR_FORMAT,
                     "pipeline decode requires cooked K-quant gate/up and "
                     "cooked Q5_1 or sealed Q8_0 down expert weights");
        status=FG_ERR_FORMAT;
    }
    fg_vk_context *vk=fg_model_vk(executor->model);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))
        status=fg_vk_profile_set_scope(vk,"expert_decode_schedule",err);
    if(status==FG_OK)status=fg_vk_decode_tile_schedule(
        vk,executor->tiles,selected,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))
        status=fg_vk_profile_set_scope(vk,"expert_decode_gate",err);
    if(status==FG_OK)status=fg_vk_moe_kquant(
        vk,executor->gate,gate_weight,activation_q8k,executor->tiles,
        gate_record->ggml_type,640u,FG_HIDDEN_SIZE,gate_stride,FG_TOP_K,
        FG_TOP_K,false,FG_TOP_K,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))
        status=fg_vk_profile_set_scope(vk,"expert_decode_up",err);
    if(status==FG_OK)status=fg_vk_moe_kquant(
        vk,executor->up,up_weight,activation_q8k,executor->tiles,
        up_record->ggml_type,640u,FG_HIDDEN_SIZE,up_stride,FG_TOP_K,
        FG_TOP_K,false,FG_TOP_K,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))
        status=fg_vk_profile_set_scope(vk,"expert_decode_activation",err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,executor->mid,executor->gate,
        executor->up,FG_EXPERT_MID_VALUES,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))
        status=fg_vk_profile_set_scope(vk,"expert_decode_down",err);
    if(status==FG_OK)status=project_down(
        vk,executor->down,down_weight,executor->tiles,executor->mid,
        down_record,down_stride,FG_TOP_K,FG_TOP_K,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))
        status=fg_vk_profile_set_scope(vk,"expert_decode_reduce",err);
    if(status==FG_OK)status=fg_vk_moe_reduce(
        vk,executor->reduced,executor->down,gates,executor->tiles,
        FG_HIDDEN_SIZE,FG_TOP_K,FG_TOP_K,err);
    if(status==FG_OK)status=fg_vk_moe_prefill_reduce(
        vk,executor->down,executor->reduced,executor->unit_gates,
        shared_output,shared_logit,FG_HIDDEN_SIZE,1u,err);
    if(status==FG_OK)status=fg_vk_end(vk,err);
    else if(fg_vk_batch_active(vk)){
        fg_error ignored={0};
        fg_vk_abort(vk,&ignored);
    }
    if(status==FG_OK)*output=executor->down;
    return status;
}
