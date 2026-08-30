#include "fg_expert.h"
#include "fg_vk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FG_EXPERT_TILES_WORDS (FG_TOP_K * 9u)
#define FG_EXPERT_MID_VALUES (FG_TOP_K * 640u)
#define FG_EXPERT_OUTPUT_VALUES (FG_TOP_K * FG_HIDDEN_SIZE)

struct fg_expert_executor {
    fg_model *model;
    fg_vk_tensor *activation,*tiles,*gates,*gate,*up,*mid,*down,*reduced;
    uint32_t max_tokens,max_pairs;
};

static fg_status create_scratch(fg_expert_executor *executor,fg_error *err){
    const fg_manifest *manifest=fg_model_manifest(executor->model);executor->max_tokens=manifest->prefill_microbatch;executor->max_pairs=executor->max_tokens*FG_TOP_K;
    if(!executor->max_tokens||executor->max_tokens>FG_PREFILL_MAX_TOKENS){fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds expert executor limit");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(executor->model);fg_status status=fg_vk_tensor_create(vk,(uint64_t)executor->max_tokens*FG_Q8K_ACTIVATION_BYTES,&executor->activation,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*9u*4u,&executor->tiles,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,FG_TOP_K*4u,&executor->gates,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->gate,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->up,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*640u*4u,&executor->mid,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)executor->max_pairs*FG_HIDDEN_SIZE*4u,&executor->down,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,FG_HIDDEN_SIZE*4u,&executor->reduced,err);
    return status;
}

fg_status fg_expert_executor_create(fg_expert_executor **out,fg_model *model,fg_error *err){if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert executor arguments");return FG_ERR_ARGUMENT;}*out=NULL;fg_expert_executor *executor=calloc(1,sizeof(*executor));if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate expert executor");return FG_ERR_OOM;}executor->model=model;fg_status status=create_scratch(executor,err);if(status!=FG_OK){fg_expert_executor_destroy(executor);return status;}*out=executor;return FG_OK;}
void fg_expert_executor_destroy(fg_expert_executor *executor){if(!executor)return;fg_vk_tensor_destroy(executor->reduced);fg_vk_tensor_destroy(executor->down);fg_vk_tensor_destroy(executor->mid);fg_vk_tensor_destroy(executor->up);fg_vk_tensor_destroy(executor->gate);fg_vk_tensor_destroy(executor->gates);fg_vk_tensor_destroy(executor->tiles);fg_vk_tensor_destroy(executor->activation);free(executor);}

static uint32_t local_expert(const fg_manifest *manifest,uint32_t layer,uint32_t rank,uint32_t global){uint32_t local=0;for(uint32_t expert=0;expert<global;expert++)if(manifest->expert_rank[layer][expert]==rank)local++;return local;}
static bool tensor_name(char output[FG_TENSOR_NAME_MAX],uint32_t layer,const char *family,uint32_t rank){int length=snprintf(output,FG_TENSOR_NAME_MAX,"blk.%u.%s.weight.rank%u",layer,family,rank);return length>=0&&(uint32_t)length<FG_TENSOR_NAME_MAX;}

fg_status fg_expert_decode(fg_expert_executor *executor,const fg_decode_work *work,fg_expert_result *result,fg_error *err){
    if(!executor||!work||!result){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert decode arguments");return FG_ERR_ARGUMENT;}const fg_manifest *manifest=fg_model_manifest(executor->model);uint32_t rank=fg_model_rank(executor->model);if(work->layer>=FG_LAYER_COUNT||work->destination_rank!=rank||(work->source_rank!=0u&&work->source_rank!=manifest->layer_owner[work->layer])||work->selected_count==0||work->selected_count>FG_TOP_K){fg_error_set(err,FG_ERR_MISMATCH,"expert work does not match rank, owner, or layer");return FG_ERR_MISMATCH;}
    uint32_t schedule[FG_EXPERT_TILES_WORDS];for(uint32_t i=0;i<FG_EXPERT_TILES_WORDS;i++)schedule[i]=UINT32_MAX;bool slots[FG_TOP_K]={0},experts[FG_EXPERT_COUNT]={0};for(uint32_t i=0;i<work->selected_count;i++){uint32_t global=work->expert_ids[i],slot=work->routing_slots[i];if(global>=FG_EXPERT_COUNT||experts[global]||manifest->expert_rank[work->layer][global]!=rank||slot>=FG_TOP_K||slots[slot]){fg_error_set(err,FG_ERR_MISMATCH,"invalid expert work route %u",i);return FG_ERR_MISMATCH;}uint32_t local=local_expert(manifest,work->layer,rank,global);if(local>=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_MISMATCH,"local expert mapping overflow");return FG_ERR_MISMATCH;}schedule[i*9u]=local;schedule[i*9u+1u]=slot;slots[slot]=true;experts[global]=true;}
    char gate_name[FG_TENSOR_NAME_MAX],up_name[FG_TENSOR_NAME_MAX],down_name[FG_TENSOR_NAME_MAX];if(!tensor_name(gate_name,work->layer,"ffn_gate_exps",rank)||!tensor_name(up_name,work->layer,"ffn_up_exps",rank)||!tensor_name(down_name,work->layer,"ffn_down_exps",rank)){fg_error_set(err,FG_ERR_LIMIT,"expert tensor name overflow");return FG_ERR_LIMIT;}fg_vk_tensor *gate_weight=fg_model_tensor(executor->model,gate_name),*up_weight=fg_model_tensor(executor->model,up_name),*down_weight=fg_model_tensor(executor->model,down_name);const fg_tensor_record *gate_record=fg_model_tensor_record(executor->model,gate_name),*up_record=fg_model_tensor_record(executor->model,up_name),*down_record=fg_model_tensor_record(executor->model,down_name);if(!gate_weight||!up_weight||!down_weight||!gate_record||!up_record||!down_record){fg_error_set(err,FG_ERR_MISMATCH,"rank is missing layer %u expert tensors",work->layer);return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(executor->model);fg_status status=fg_vk_tensor_write(executor->activation,0,work->activation_q8k,FG_Q8K_ACTIVATION_BYTES,err);if(status==FG_OK)status=fg_vk_tensor_write(executor->tiles,0,schedule,sizeof(schedule),err);if(status==FG_OK)status=fg_vk_tensor_write(executor->gates,0,work->gates,(uint64_t)work->selected_count*4u,err);uint32_t tiles=work->selected_count;
    /* Fused GPU batch: gate/up/SwiGLU + down + reduction in one submission. */
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"routed_expert",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_moe_gate_up_swiglu(vk,executor->mid,gate_weight,up_weight,executor->activation,executor->tiles,gate_record->ggml_type,up_record->ggml_type,640u,FG_HIDDEN_SIZE,(uint32_t)(gate_record->bytes/FG_EXPERTS_PER_RANK),(uint32_t)(up_record->bytes/FG_EXPERTS_PER_RANK),FG_TOP_K,FG_TOP_K,false,tiles,err);
    uint32_t down_stride=(uint32_t)(down_record->bytes/FG_EXPERTS_PER_RANK);
    if(status==FG_OK&&down_record->ggml_type==7u)status=fg_vk_moe_q5_1_down(vk,executor->down,down_weight,executor->tiles,executor->mid,FG_HIDDEN_SIZE,640u,down_stride,FG_TOP_K,false,tiles,err);else if(status==FG_OK&&down_record->ggml_type==8u)status=fg_vk_moe_q8_0_down(vk,executor->down,down_weight,executor->tiles,executor->mid,FG_HIDDEN_SIZE,640u,down_stride,FG_TOP_K,false,tiles,err);else if(status==FG_OK){fg_error_set(err,FG_ERR_FORMAT,"unsupported expert down quant %u",down_record->ggml_type);status=FG_ERR_FORMAT;}
    if(status==FG_OK)status=fg_vk_moe_reduce(vk,executor->reduced,executor->down,executor->gates,executor->tiles,FG_HIDDEN_SIZE,work->selected_count,FG_TOP_K,err);
    if(status==FG_OK){fg_status es=fg_vk_end(vk,err);if(es!=FG_OK)status=es;}else if(fg_vk_batch_active(vk))fg_vk_end(vk,err);
    if(status!=FG_OK)return status;
    memset(result,0,sizeof(*result));result->layer=work->layer;result->source_rank=(uint8_t)rank;result->destination_rank=work->source_rank;result->selected_count=1u;result->routing_slots[0]=0xFFu;result->position=work->position;memcpy(result->outputs[0],fg_vk_tensor_map(executor->reduced),FG_HIDDEN_SIZE*4u);return FG_OK;
}

fg_status fg_expert_prefill(fg_expert_executor *executor,const fg_prefill_work *work,fg_prefill_result *result,fg_prefill_result_pair *pair_storage,uint32_t pair_capacity,float *output_storage,uint64_t output_capacity_values,fg_error *err){
    if(!executor||!work||!result||!pair_storage||!output_storage){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert prefill arguments");return FG_ERR_ARGUMENT;}
    const fg_manifest *manifest=fg_model_manifest(executor->model);uint32_t rank=fg_model_rank(executor->model);
    if(work->layer>=FG_LAYER_COUNT||work->destination_rank!=rank||(work->source_rank!=0u&&work->source_rank!=manifest->layer_owner[work->layer])||!work->token_count||work->token_count>executor->max_tokens||!work->pair_count||work->pair_count>executor->max_pairs||pair_capacity<work->pair_count||output_capacity_values<(uint64_t)work->pair_count*FG_HIDDEN_SIZE){fg_error_set(err,FG_ERR_MISMATCH,"expert prefill work does not match rank, owner, or configured microbatch");return FG_ERR_MISMATCH;}
    uint16_t *locals=malloc((size_t)work->pair_count*sizeof(*locals));uint32_t *schedule=malloc((size_t)work->pair_count*9u*sizeof(*schedule));if(!locals||!schedule){free(schedule);free(locals);fg_error_set(err,FG_ERR_OOM,"allocate expert-major prefill schedule");return FG_ERR_OOM;}
    uint32_t counts[FG_EXPERTS_PER_RANK]={0};bool seen[FG_PREFILL_MAX_TOKENS][FG_TOP_K]={{false}};fg_status status=FG_OK;
    for(uint32_t i=0;i<work->pair_count;i++){const fg_prefill_pair *pair=&work->pairs[i];if(pair->token_slot>=work->token_count||pair->routing_slot>=FG_TOP_K||pair->expert_id>=FG_EXPERT_COUNT||seen[pair->token_slot][pair->routing_slot]||manifest->expert_rank[work->layer][pair->expert_id]!=rank){fg_error_set(err,FG_ERR_MISMATCH,"invalid expert prefill route %u",i);status=FG_ERR_MISMATCH;break;}uint32_t local=local_expert(manifest,work->layer,rank,pair->expert_id);if(local>=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_MISMATCH,"prefill local expert mapping overflow");status=FG_ERR_MISMATCH;break;}locals[i]=(uint16_t)local;counts[local]++;seen[pair->token_slot][pair->routing_slot]=true;}
    uint32_t tile_count=0;if(status==FG_OK){for(uint32_t local=0;local<FG_EXPERTS_PER_RANK;local++)tile_count+=(counts[local]+7u)/8u;for(uint32_t i=0;i<tile_count*9u;i++)schedule[i]=UINT32_MAX;uint32_t tile=0;for(uint32_t local=0;local<FG_EXPERTS_PER_RANK;local++)if(counts[local]){uint32_t slot=0;schedule[tile*9u]=local;for(uint32_t pair=0;pair<work->pair_count;pair++)if(locals[pair]==local){if(slot==8u){tile++;slot=0;schedule[tile*9u]=local;}schedule[tile*9u+1u+slot]=work->pairs[pair].token_slot*FG_TOP_K+work->pairs[pair].routing_slot;slot++;}tile++;}if(tile!=tile_count){fg_error_set(err,FG_ERR_MISMATCH,"expert-major prefill schedule count mismatch");status=FG_ERR_MISMATCH;}}
    char gate_name[FG_TENSOR_NAME_MAX],up_name[FG_TENSOR_NAME_MAX],down_name[FG_TENSOR_NAME_MAX];if(status==FG_OK&&(!tensor_name(gate_name,work->layer,"ffn_gate_exps",rank)||!tensor_name(up_name,work->layer,"ffn_up_exps",rank)||!tensor_name(down_name,work->layer,"ffn_down_exps",rank))){fg_error_set(err,FG_ERR_LIMIT,"expert tensor name overflow");status=FG_ERR_LIMIT;}
    fg_vk_tensor *gate_weight=status==FG_OK?fg_model_tensor(executor->model,gate_name):NULL,*up_weight=status==FG_OK?fg_model_tensor(executor->model,up_name):NULL,*down_weight=status==FG_OK?fg_model_tensor(executor->model,down_name):NULL;const fg_tensor_record *gate_record=status==FG_OK?fg_model_tensor_record(executor->model,gate_name):NULL,*up_record=status==FG_OK?fg_model_tensor_record(executor->model,up_name):NULL,*down_record=status==FG_OK?fg_model_tensor_record(executor->model,down_name):NULL;
    if(status==FG_OK&&(!gate_weight||!up_weight||!down_weight||!gate_record||!up_record||!down_record||gate_record->bytes%FG_EXPERTS_PER_RANK||up_record->bytes%FG_EXPERTS_PER_RANK||down_record->bytes%FG_EXPERTS_PER_RANK||gate_record->bytes/FG_EXPERTS_PER_RANK>UINT32_MAX||up_record->bytes/FG_EXPERTS_PER_RANK>UINT32_MAX||down_record->bytes/FG_EXPERTS_PER_RANK>UINT32_MAX)){fg_error_set(err,FG_ERR_MISMATCH,"rank is missing or has malformed layer %u expert tensors",work->layer);status=FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(executor->model);uint32_t dense_pairs=(uint32_t)work->token_count*FG_TOP_K;
    if(status==FG_OK)status=fg_vk_tensor_write(executor->activation,0,work->activations_q8k,(uint64_t)work->token_count*FG_Q8K_ACTIVATION_BYTES,err);
    if(status==FG_OK)status=fg_vk_tensor_write(executor->tiles,0,schedule,(uint64_t)tile_count*9u*4u,err);
    if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->gate,gate_weight,executor->activation,executor->tiles,gate_record->ggml_type,640u,FG_HIDDEN_SIZE,(uint32_t)(gate_record->bytes/FG_EXPERTS_PER_RANK),FG_TOP_K,dense_pairs,false,tile_count,err);
    if(status==FG_OK)status=fg_vk_moe_kquant(vk,executor->up,up_weight,executor->activation,executor->tiles,up_record->ggml_type,640u,FG_HIDDEN_SIZE,(uint32_t)(up_record->bytes/FG_EXPERTS_PER_RANK),FG_TOP_K,dense_pairs,false,tile_count,err);
    if(status==FG_OK){status=fg_vk_swiglu(vk,executor->mid,executor->gate,executor->up,(uint32_t)((uint64_t)dense_pairs*640u),err);}
    uint32_t down_stride=status==FG_OK?(uint32_t)(down_record->bytes/FG_EXPERTS_PER_RANK):0u;
    if(status==FG_OK&&down_record->ggml_type==7u)status=fg_vk_moe_q5_1_down(vk,executor->down,down_weight,executor->tiles,executor->mid,FG_HIDDEN_SIZE,640u,down_stride,FG_TOP_K,false,tile_count,err);else if(status==FG_OK&&down_record->ggml_type==8u)status=fg_vk_moe_q8_0_down(vk,executor->down,down_weight,executor->tiles,executor->mid,FG_HIDDEN_SIZE,640u,down_stride,FG_TOP_K,false,tile_count,err);else if(status==FG_OK){fg_error_set(err,FG_ERR_FORMAT,"unsupported expert down quant %u",down_record->ggml_type);status=FG_ERR_FORMAT;}
    if(status==FG_OK){const float *host=fg_vk_tensor_map(executor->down);memset(result,0,sizeof(*result));result->layer=work->layer;result->source_rank=(uint8_t)rank;result->destination_rank=work->source_rank;result->first_position=work->first_position;result->token_count=work->token_count;result->pair_count=work->pair_count;result->pairs=pair_storage;result->outputs=output_storage;for(uint32_t i=0;i<work->pair_count;i++){uint32_t pair_id=work->pairs[i].token_slot*FG_TOP_K+work->pairs[i].routing_slot;pair_storage[i].token_slot=work->pairs[i].token_slot;pair_storage[i].routing_slot=work->pairs[i].routing_slot;memcpy(output_storage+(uint64_t)i*FG_HIDDEN_SIZE,host+(uint64_t)pair_id*FG_HIDDEN_SIZE,FG_HIDDEN_SIZE*4u);}}
    free(schedule);free(locals);return status;
}
