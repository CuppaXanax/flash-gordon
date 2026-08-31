#include "fg_owner.h"
#include "fg_q38_math.h"
#include "fg_qsa.h"
#include "fg_topology.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ts_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec*1e3+(double)t.tv_nsec*1e-6;}
static uint64_t wall_ns(void){struct timespec value;clock_gettime(CLOCK_REALTIME,&value);return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;}
static bool gdn_diag_enabled(void){const char *value=getenv("FG_GDN_DIAG");return value&&*value&&strcmp(value,"0")!=0;}

#define FG_HC_INJECT_PIECES 24u
#define FG_HC_DOWN_SPLITS 8u

struct fg_owner_executor {
    fg_model *model;
    uint32_t max_tokens;
    bool replicated;
    fg_vk_tensor *hyper_norm,*low,*hc_down_partials,*low_active,*up_logits,*inject_partials,*mixed,*injection,*hyper_output,*hyper_output_b;
    fg_vk_tensor *router_logits,*activation_q8k,*shared_gate,*shared_up,*shared_mid,*shared_output,*shared_scalar,*reduced;
    fg_vk_tensor *gdn_qkv,*gdn_conv_output,*gdn_z,*gdn_alpha,*gdn_beta,*gdn_core,*gdn_output;
    fg_vk_tensor *ple_key,*ple_value,*ple_key_norm,*ple_query_norm,*ple_gated,*ple_gated_norm,*ple_output,*ple_added,*ple_state;
    struct {fg_vk_tensor *conv_state,*recurrent_state;} gdn_state[FG_LAYER_COUNT];
    fg_vk_tensor *attention_family_scratch;
    struct {bool active;uint32_t layer,token;const fg_vk_tensor *hyper,*block,*injection;fg_vk_tensor *output;} pending_write;
    fg_qsa_session *qsa;
};

static fg_status scratch(fg_vk_context *vk,uint64_t values,fg_vk_tensor **out,fg_error *err){return fg_vk_tensor_create(vk,values*4u,out,err);}
static fg_status family_view(fg_owner_executor *executor,uint64_t *offset,uint64_t bytes,
                             fg_vk_tensor **out,fg_error *err){
    fg_vk_tensor *arena=executor?executor->attention_family_scratch:NULL;
    if(!arena||!offset||!out||!bytes||*offset>fg_vk_tensor_bytes(arena)||
       bytes>fg_vk_tensor_bytes(arena)-*offset){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid shared attention scratch view");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_tensor_view(arena,*offset,bytes,out,err);
    if(status==FG_OK)*offset+=bytes;
    return status;
}
static fg_status create_attention_family_views(fg_owner_executor *executor,fg_error *err){
    uint32_t tokens=executor->max_tokens;uint64_t offset=0;fg_status status=FG_OK;
    const uint64_t gdn[]={
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u,
        (uint64_t)tokens*6144u*4u,(uint64_t)tokens*48u*4u,
        (uint64_t)tokens*48u*4u,(uint64_t)tokens*6144u*4u,
        (uint64_t)tokens*2560u*4u
    };
    fg_vk_tensor **gdn_fields[]={
        &executor->gdn_qkv,&executor->gdn_conv_output,&executor->gdn_z,
        &executor->gdn_alpha,&executor->gdn_beta,&executor->gdn_core,
        &executor->gdn_output
    };
    for(uint32_t i=0;status==FG_OK&&i<sizeof(gdn)/sizeof(gdn[0]);i++)
        status=family_view(executor,&offset,gdn[i],gdn_fields[i],err);
    offset=0;
    const uint64_t ple[]={
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*2560u*4u,
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u,
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u,
        (uint64_t)tokens*10240u*4u,(uint64_t)tokens*10240u*4u
    };
    fg_vk_tensor **ple_fields[]={
        &executor->ple_key,&executor->ple_value,&executor->ple_key_norm,
        &executor->ple_query_norm,&executor->ple_gated,&executor->ple_gated_norm,
        &executor->ple_output,&executor->ple_added
    };
    for(uint32_t i=0;status==FG_OK&&i<sizeof(ple)/sizeof(ple[0]);i++)
        status=family_view(executor,&offset,ple[i],ple_fields[i],err);
    if(status==FG_OK&&offset!=fg_qsa_ple_scratch_bytes(tokens)){
        fg_error_set(err,FG_ERR_MISMATCH,"shared PLE scratch geometry mismatch");
        status=FG_ERR_MISMATCH;
    }
    return status;
}

static fg_status create_transient_views(fg_owner_executor *executor,fg_error *err){
    fg_vk_tensor *arena=executor->attention_family_scratch;
    const fg_manifest *manifest=fg_model_manifest(executor->model);
    uint32_t tokens=executor->max_tokens;
    uint64_t offset=0;
    fg_status status=FG_OK;
#define TRANSIENT(field,bytes) do{ \
        status=fg_vk_tensor_view(arena,offset,(bytes),&executor->field,err); \
        if(status==FG_OK)offset=fg_align_up_u64(offset+(bytes),FG_ALIGNMENT); \
    }while(0)
    TRANSIENT(hyper_norm,(uint64_t)tokens*10240u*4u);
    TRANSIENT(up_logits,(uint64_t)tokens*10240u*4u);
    TRANSIENT(low,(uint64_t)tokens*320u*4u);
    TRANSIENT(low_active,(uint64_t)tokens*320u*4u);
    TRANSIENT(inject_partials,(uint64_t)tokens*FG_HC_INJECT_PIECES*4u*4u);
    TRANSIENT(hc_down_partials,(uint64_t)FG_HC_DOWN_SPLITS*320u*4u);
    TRANSIENT(router_logits,(uint64_t)tokens*FG_EXPERT_COUNT*4u);
    TRANSIENT(shared_gate,(uint64_t)tokens*640u*4u);
    TRANSIENT(shared_up,(uint64_t)tokens*640u*4u);
    TRANSIENT(shared_mid,(uint64_t)tokens*640u*4u);
    TRANSIENT(reduced,(uint64_t)tokens*FG_HIDDEN_SIZE*4u);
    if(status==FG_OK){
        offset=fg_align_up_u64(fg_qsa_attention_scratch_bytes(tokens),FG_ALIGNMENT)+
               fg_qsa_selection_scratch_bytes(manifest->native_context,tokens);
        TRANSIENT(mixed,(uint64_t)tokens*2560u*4u);
        /* GR injection has one gate value for every group and token. */
        TRANSIENT(injection,(uint64_t)tokens*FG_GROUP_SIZE*4u);
        TRANSIENT(shared_output,(uint64_t)tokens*FG_HIDDEN_SIZE*4u);
        TRANSIENT(shared_scalar,(uint64_t)tokens*4u);
    }
#undef TRANSIENT
    return status;
}

fg_status fg_owner_executor_create(fg_owner_executor **out,fg_model *model,fg_error *err){
    if(!out||!model){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner executor arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    fg_owner_executor *executor=calloc(1,sizeof(*executor));if(!executor){fg_error_set(err,FG_ERR_OOM,"allocate owner executor");return FG_ERR_OOM;}executor->model=model;fg_vk_context *vk=fg_model_vk(model);
    const fg_manifest *manifest=fg_model_manifest(model);executor->max_tokens=manifest->prefill_microbatch;if(!executor->max_tokens||executor->max_tokens>FG_PREFILL_MAX_TOKENS){fg_owner_executor_destroy(executor);fg_error_set(err,FG_ERR_MISMATCH,"manifest prefill microbatch exceeds owner executor limit");return FG_ERR_MISMATCH;}uint64_t tokens=executor->max_tokens;
    fg_status status=fg_vk_tensor_create(vk,(uint64_t)10240u*tokens*4u,&executor->hyper_output,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,(uint64_t)10240u*tokens*4u,
                                                  &executor->hyper_output_b,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,tokens*FG_Q8K_ACTIVATION_BYTES,
                                                  &executor->activation_q8k,err);
    if(status==FG_OK)status=fg_vk_tensor_create(
        vk,fg_qsa_attention_family_scratch_bytes(executor->max_tokens),
        &executor->attention_family_scratch,err);
    if(status==FG_OK)status=create_attention_family_views(executor,err);
    if(status==FG_OK)status=create_transient_views(executor,err);
    for(uint32_t layer=0;status==FG_OK&&layer<FG_LAYER_COUNT;layer++){if((layer&3u)!=3u){status=scratch(vk,10240u*4u,&executor->gdn_state[layer].conv_state,err);if(status==FG_OK)status=scratch(vk,48u*128u*128u,&executor->gdn_state[layer].recurrent_state,err);if(status==FG_OK){memset(fg_vk_tensor_map(executor->gdn_state[layer].conv_state),0,10240u*4u*4u);memset(fg_vk_tensor_map(executor->gdn_state[layer].recurrent_state),0,48u*128u*128u*4u);}}}
    if(status==FG_OK)status=scratch(vk,10240u*9u,&executor->ple_state,err);
    if(status==FG_OK)memset(fg_vk_tensor_map(executor->ple_state),0,10240u*9u*4u);
    if(status!=FG_OK){fg_owner_executor_destroy(executor);return status;}executor->replicated=true;*out=executor;return FG_OK;
}
fg_vk_tensor *fg_owner_prefill_input(fg_owner_executor *executor){
    /*
     * GR read writes up_logits, so the embedding input must stay in the
     * ping-pong buffer rather than aliasing that transient output.
     */
    return executor?executor->hyper_output_b:NULL;
}
uint64_t fg_owner_qsa_host_bytes(const fg_owner_executor *executor){
    return executor?fg_qsa_session_host_bytes(executor->qsa):0;
}
void fg_owner_executor_destroy(fg_owner_executor *e){if(!e)return;fg_qsa_session_close(e->qsa);for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){fg_vk_tensor_destroy(e->gdn_state[layer].recurrent_state);fg_vk_tensor_destroy(e->gdn_state[layer].conv_state);}fg_vk_tensor_destroy(e->ple_state);fg_vk_tensor_destroy(e->ple_added);fg_vk_tensor_destroy(e->ple_output);fg_vk_tensor_destroy(e->ple_gated_norm);fg_vk_tensor_destroy(e->ple_gated);fg_vk_tensor_destroy(e->ple_query_norm);fg_vk_tensor_destroy(e->ple_key_norm);fg_vk_tensor_destroy(e->ple_value);fg_vk_tensor_destroy(e->ple_key);fg_vk_tensor_destroy(e->gdn_output);fg_vk_tensor_destroy(e->gdn_core);fg_vk_tensor_destroy(e->gdn_beta);fg_vk_tensor_destroy(e->gdn_alpha);fg_vk_tensor_destroy(e->gdn_z);fg_vk_tensor_destroy(e->gdn_conv_output);fg_vk_tensor_destroy(e->gdn_qkv);fg_vk_tensor_destroy(e->attention_family_scratch);fg_vk_tensor_destroy(e->reduced);fg_vk_tensor_destroy(e->shared_scalar);fg_vk_tensor_destroy(e->shared_output);fg_vk_tensor_destroy(e->shared_mid);fg_vk_tensor_destroy(e->shared_up);fg_vk_tensor_destroy(e->shared_gate);fg_vk_tensor_destroy(e->activation_q8k);fg_vk_tensor_destroy(e->router_logits);fg_vk_tensor_destroy(e->hyper_output_b);fg_vk_tensor_destroy(e->hyper_output);fg_vk_tensor_destroy(e->injection);fg_vk_tensor_destroy(e->mixed);fg_vk_tensor_destroy(e->inject_partials);fg_vk_tensor_destroy(e->up_logits);fg_vk_tensor_destroy(e->low_active);fg_vk_tensor_destroy(e->hc_down_partials);fg_vk_tensor_destroy(e->low);fg_vk_tensor_destroy(e->hyper_norm);free(e);}
fg_status fg_owner_reset_state(fg_owner_executor *e,fg_error *err){if(!e){fg_error_set(err,FG_ERR_ARGUMENT,"owner state reset is null");return FG_ERR_ARGUMENT;}for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){if(e->gdn_state[layer].conv_state)memset(fg_vk_tensor_map(e->gdn_state[layer].conv_state),0,(size_t)fg_vk_tensor_bytes(e->gdn_state[layer].conv_state));if(e->gdn_state[layer].recurrent_state)memset(fg_vk_tensor_map(e->gdn_state[layer].recurrent_state),0,(size_t)fg_vk_tensor_bytes(e->gdn_state[layer].recurrent_state));}if(e->ple_state)memset(fg_vk_tensor_map(e->ple_state),0,(size_t)fg_vk_tensor_bytes(e->ple_state));memset(&e->pending_write,0,sizeof(e->pending_write));return e->qsa?fg_qsa_session_reset(e->qsa,err):FG_OK;}
fg_status fg_owner_qsa_checkpoint(fg_owner_executor *executor,fg_error *err){if(!executor||!executor->qsa){fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA checkpoint is unavailable");return FG_ERR_ARGUMENT;}return fg_qsa_session_checkpoint(executor->qsa,err);}

static fg_vk_tensor *weight(fg_owner_executor *executor,uint32_t layer,const char *suffix,fg_error *err){char name[FG_TENSOR_NAME_MAX];int length=snprintf(name,sizeof(name),"blk.%u.%s",layer,suffix);if(length<0||(uint32_t)length>=sizeof(name)){fg_error_set(err,FG_ERR_LIMIT,"owner tensor name overflow");return NULL;}fg_vk_tensor *tensor=fg_model_tensor(executor->model,name);if(!tensor)fg_error_set(err,FG_ERR_MISMATCH,"owner rank is missing %s",name);return tensor;}
static bool owns_layer(const fg_owner_executor *executor,uint32_t layer){if(executor->replicated)return layer<FG_LAYER_COUNT;const fg_manifest *manifest=fg_model_manifest(executor->model);return layer<FG_LAYER_COUNT&&manifest->layer_owner[layer]==fg_model_rank(executor->model);}

fg_status fg_owner_gr_read_batch(fg_owner_executor *e,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,uint32_t token_count,fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err){
    if(!e||!hyper_input||!mixed||!residual||!injection||!token_count||token_count>e->max_tokens||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"gated residual batch is not on the layer owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}const char *prefix=ffn?"hc_ffn":"hc_attn";char suffix[48];fg_vk_tensor *norm_weight,*down_weight,*up_weight,*inject_weight;snprintf(suffix,sizeof(suffix),"%s_norm.weight",prefix);norm_weight=weight(e,layer,suffix,err);snprintf(suffix,sizeof(suffix),"%s_down.weight",prefix);down_weight=weight(e,layer,suffix,err);snprintf(suffix,sizeof(suffix),"%s_up.weight",prefix);up_weight=weight(e,layer,suffix,err);snprintf(suffix,sizeof(suffix),"%s_inject.weight",prefix);inject_weight=weight(e,layer,suffix,err);if(!norm_weight||!down_weight||!up_weight||!inject_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->hyper_norm,hyper_input,norm_weight,FG_HIDDEN_SIZE,4u,token_count,1e-6f,err);if(status==FG_OK&&token_count==1u&&fg_vk_tensor_get_format(down_weight)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED)status=fg_vk_dense_q8_0_cooked_split(vk,e->low,e->hc_down_partials,down_weight,e->hyper_norm,10240u,320u,1u,FG_HC_DOWN_SPLITS,1.0f,err);else if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->low,down_weight,e->hyper_norm,10240u,320u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_hc_inject_partial(vk,e->inject_partials,e->hyper_norm,inject_weight,FG_HIDDEN_SIZE,4u,token_count,FG_HC_INJECT_PIECES,err);if(status==FG_OK)status=fg_vk_silu_scaled(vk,e->low_active,e->low,token_count*320u,0.25f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->up_logits,up_weight,e->low_active,320u,10240u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_gr_mix_partial(vk,e->mixed,e->injection,e->hyper_norm,e->up_logits,e->inject_partials,FG_HIDDEN_SIZE,4u,token_count,FG_HC_INJECT_PIECES,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status==FG_OK){*mixed=e->mixed;*residual=hyper_input;*injection=e->injection;}return status;
}

fg_status fg_owner_gr_read(fg_owner_executor *e,uint32_t layer,bool ffn,const fg_vk_tensor *hyper_input,fg_vk_tensor **mixed,const fg_vk_tensor **residual,fg_vk_tensor **injection,fg_error *err){return fg_owner_gr_read_batch(e,layer,ffn,hyper_input,1u,mixed,residual,injection,err);}

fg_status fg_owner_moe_prepare_batch(fg_owner_executor *e,uint32_t layer,const fg_vk_tensor *hidden,uint16_t token_count,uint16_t *expert_ids,float *gates,const uint8_t **activation,fg_error *err){
    if(!e||!hidden||!token_count||token_count>e->max_tokens||!expert_ids||!gates||!activation||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"MoE prepare batch is not on the layer owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}fg_vk_tensor *router=weight(e,layer,"ffn_gate_inp.weight",err),*shared_gate_weight=weight(e,layer,"ffn_gate_inp_shexp.weight",err),*gate_weight=weight(e,layer,"ffn_gate_shexp.weight",err),*up_weight=weight(e,layer,"ffn_up_shexp.weight",err),*down_weight=weight(e,layer,"ffn_down_shexp.weight",err);if(!router||!shared_gate_weight||!gate_weight||!up_weight||!down_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router,hidden,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,token_count,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status!=FG_OK)return status;const float *router_values=fg_vk_tensor_map(e->router_logits);for(uint32_t token=0;status==FG_OK&&token<token_count;token++){uint32_t ids[FG_TOP_K];status=fg_q38_router_topk(router_values+(uint64_t)token*FG_EXPERT_COUNT,FG_EXPERT_COUNT,FG_TOP_K,ids,gates+(uint64_t)token*FG_TOP_K,err);for(uint32_t slot=0;status==FG_OK&&slot<FG_TOP_K;slot++)expert_ids[(uint64_t)token*FG_TOP_K+slot]=(uint16_t)ids[slot];}if(status==FG_OK)status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,hidden,FG_HIDDEN_SIZE,token_count,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate_weight,hidden,FG_HIDDEN_SIZE,640u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up_weight,hidden,FG_HIDDEN_SIZE,640u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,e->shared_up,(uint32_t)token_count*640u,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_output,down_weight,e->shared_mid,640u,FG_HIDDEN_SIZE,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_dense_f32(vk,e->shared_scalar,shared_gate_weight,hidden,FG_HIDDEN_SIZE,1u,token_count,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status==FG_OK)*activation=fg_vk_tensor_map(e->activation_q8k);return status;
}

fg_status fg_owner_moe_prepare(fg_owner_executor *e,uint32_t layer,const fg_vk_tensor *hidden,uint16_t expert_ids[FG_TOP_K],float gates[FG_TOP_K],const uint8_t **activation,fg_error *err){return fg_owner_moe_prepare_batch(e,layer,hidden,1u,expert_ids,gates,activation,err);}

fg_status fg_owner_moe_reduce(fg_owner_executor *e,uint32_t layer,uint32_t position,const uint16_t expert_ids[FG_TOP_K],const float gates[FG_TOP_K],const fg_expert_result *results,uint32_t result_count,fg_vk_tensor **output,fg_error *err){
    if(!e||!expert_ids||!gates||!results||!result_count||result_count>FG_GROUP_SIZE||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid MoE reduction arguments");return FG_ERR_ARGUMENT;}
    if(!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"MoE reduction is not on the layer owner");return FG_ERR_MISMATCH;}
    const fg_manifest *manifest=fg_model_manifest(e->model);uint32_t owner=fg_model_rank(e->model);const float *slot_output[FG_TOP_K]={0};bool seen_expert[FG_EXPERT_COUNT]={0};
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){if(expert_ids[slot]>=FG_EXPERT_COUNT||seen_expert[expert_ids[slot]]||!isfinite(gates[slot])){fg_error_set(err,FG_ERR_FORMAT,"invalid canonical route slot %u",slot);return FG_ERR_FORMAT;}seen_expert[expert_ids[slot]]=true;}
    fg_status route_status=fg_expert_results_validate_route(manifest,layer,position,owner,expert_ids,results,result_count,err);if(route_status!=FG_OK)return route_status;
    /* Check for pre-reduced results (routing_slot 0xFF = worker already applied gates) */
    bool has_prereduced=false;
    const float *prereduced[FG_RANK_COUNT]={0};
    for(uint32_t r=0;r<result_count;r++)for(uint32_t i=0;i<results[r].selected_count;i++){if(results[r].routing_slots[i]==0xFFu){uint32_t source=results[r].source_rank;if(prereduced[source]){fg_error_set(err,FG_ERR_MISMATCH,"duplicate pre-reduced expert result from rank %u",source);return FG_ERR_MISMATCH;}has_prereduced=true;prereduced[source]=results[r].outputs[i];}else{slot_output[results[r].routing_slots[i]]=results[r].outputs[i];}}
    if(!has_prereduced){for(uint32_t slot=0;slot<FG_TOP_K;slot++){if(!slot_output[slot]){fg_error_set(err,FG_ERR_MISMATCH,"missing expert result slot %u",slot);return FG_ERR_MISMATCH;}}}
    float shared_scale=1.0f/(1.0f+expf(-*(const float *)fg_vk_tensor_map(e->shared_scalar)));
    /* Cache-friendly reduction: copy GPU-mapped (write-combining) data to stack,
       reduce in L1, then write result back.  The WC mapping makes scattered reads
       ~20x slower than cached DRAM; this copy+reduce pattern eliminates that. */
    float shared_local[FG_HIDDEN_SIZE],result_local[FG_HIDDEN_SIZE];
    memcpy(shared_local,fg_vk_tensor_map(e->shared_output),FG_HIDDEN_SIZE*sizeof(float));
    for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]=shared_scale*shared_local[element];
    if(has_prereduced){for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(prereduced[rank])for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]+=prereduced[rank][element];
    /* Add any non-pre-reduced slot outputs (e.g., local experts on coordinator) */
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){if(slot_output[slot]){float g=gates[slot];const float *out=slot_output[slot];for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]=fmaf(g,out[element],result_local[element]);}}
    }else{for(uint32_t slot=0;slot<FG_TOP_K;slot++){float g=gates[slot];const float *out=slot_output[slot];for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)result_local[element]=fmaf(g,out[element],result_local[element]);}}
    memcpy(fg_vk_tensor_map(e->reduced),result_local,FG_HIDDEN_SIZE*sizeof(float));
    *output=e->reduced;return FG_OK;
}

fg_status fg_owner_moe_reduce_batch(fg_owner_executor *e,uint32_t layer,uint32_t first_position,uint16_t token_count,const uint16_t *expert_ids,const float *gates,const fg_prefill_result *results,uint32_t result_count,fg_vk_tensor **output,fg_error *err){
    if(!e||!token_count||token_count>e->max_tokens||!expert_ids||!gates||!results||!result_count||result_count>FG_GROUP_SIZE||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner MoE batch reduction arguments");return FG_ERR_ARGUMENT;}fg_status status=fg_prefill_results_validate_route(fg_model_manifest(e->model),layer,first_position,fg_model_rank(e->model),token_count,expert_ids,results,result_count,err);if(status!=FG_OK)return status;
    const float **slot_outputs=calloc((size_t)token_count*FG_TOP_K,sizeof(*slot_outputs));if(!slot_outputs){fg_error_set(err,FG_ERR_OOM,"allocate canonical prefill reduction slots");return FG_ERR_OOM;}for(uint32_t result_index=0;result_index<result_count;result_index++)for(uint32_t pair=0;pair<results[result_index].pair_count;pair++){uint32_t token=results[result_index].pairs[pair].token_slot,slot=results[result_index].pairs[pair].routing_slot;slot_outputs[(uint64_t)token*FG_TOP_K+slot]=results[result_index].outputs+(uint64_t)pair*FG_HIDDEN_SIZE;}
    const float *shared=fg_vk_tensor_map(e->shared_output),*shared_scalar=fg_vk_tensor_map(e->shared_scalar);float *reduced=fg_vk_tensor_map(e->reduced);for(uint32_t token=0;token<token_count;token++){float shared_scale=1.0f/(1.0f+expf(-shared_scalar[token]));for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++){float value=shared_scale*shared[(uint64_t)token*FG_HIDDEN_SIZE+element];for(uint32_t slot=0;slot<FG_TOP_K;slot++){const float *expert=slot_outputs[(uint64_t)token*FG_TOP_K+slot];float gate=gates[(uint64_t)token*FG_TOP_K+slot];if(!expert||!isfinite(gate)){free(slot_outputs);fg_error_set(err,FG_ERR_MISMATCH,"missing or invalid canonical prefill reduction slot");return FG_ERR_MISMATCH;}value=fmaf(gate,expert[element],value);}reduced[(uint64_t)token*FG_HIDDEN_SIZE+element]=value;}}
    free(slot_outputs);*output=e->reduced;return FG_OK;
}

fg_status fg_owner_gr_write_batch(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,uint32_t token_count,fg_vk_tensor **output,fg_error *err){if(!executor||!hyper_input||!block_output||!injection||!token_count||token_count>executor->max_tokens||!output){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual batch write arguments");return FG_ERR_ARGUMENT;}fg_vk_tensor *dst=hyper_input!=executor->hyper_output?executor->hyper_output:executor->hyper_output_b;fg_status status=fg_vk_gr_write(fg_model_vk(executor->model),dst,hyper_input,block_output,injection,FG_HIDDEN_SIZE,4u,token_count,err);if(status==FG_OK)*output=dst;return status;}

fg_status fg_owner_gr_write(fg_owner_executor *executor,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,fg_vk_tensor **output,fg_error *err){return fg_owner_gr_write_batch(executor,hyper_input,block_output,injection,1u,output,err);}

static fg_status defer_gr_write(fg_owner_executor *executor,uint32_t layer,uint32_t token,const fg_vk_tensor *hyper_input,const fg_vk_tensor *block_output,const fg_vk_tensor *injection,fg_vk_tensor **output,fg_error *err){if(!executor||!hyper_input||!block_output||!injection||!output||executor->pending_write.active){fg_error_set(err,FG_ERR_MISMATCH,"invalid or duplicate deferred residual write");return FG_ERR_MISMATCH;}fg_vk_tensor *destination=hyper_input!=executor->hyper_output?executor->hyper_output:executor->hyper_output_b;executor->pending_write.active=true;executor->pending_write.layer=layer;executor->pending_write.token=token;executor->pending_write.hyper=hyper_input;executor->pending_write.block=block_output;executor->pending_write.injection=injection;executor->pending_write.output=destination;*output=destination;return FG_OK;}

static fg_status flush_gr_write(fg_owner_executor *executor,fg_error *err){if(!executor||!executor->pending_write.active){fg_error_set(err,FG_ERR_MISMATCH,"deferred residual write is unavailable");return FG_ERR_MISMATCH;}fg_status status=fg_vk_gr_write(fg_model_vk(executor->model),executor->pending_write.output,executor->pending_write.hyper,executor->pending_write.block,executor->pending_write.injection,FG_HIDDEN_SIZE,4u,1u,err);if(status==FG_OK)memset(&executor->pending_write,0,sizeof(executor->pending_write));return status;}

fg_status fg_owner_gdn_decode(fg_owner_executor *executor,uint32_t layer,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    if(!executor||!hidden||!output||!owns_layer(executor,layer)||(layer&3u)==3u||!executor->gdn_state[layer].conv_state||!executor->gdn_state[layer].recurrent_state){fg_error_set(err,FG_ERR_MISMATCH,"GDN decode is not on an owned linear-attention layer");return FG_ERR_MISMATCH;}
    fg_vk_tensor *qkv_weight=weight(executor,layer,"attn_qkv.weight",err),*z_weight=weight(executor,layer,"attn_gate.weight",err),*alpha_weight=weight(executor,layer,"ssm_alpha.weight",err),*beta_weight=weight(executor,layer,"ssm_beta.weight",err),*conv_weight=weight(executor,layer,"ssm_conv1d.weight",err),*a_decay=weight(executor,layer,"ssm_a",err),*dt_bias=weight(executor,layer,"ssm_dt.bias",err),*norm_weight=weight(executor,layer,"ssm_norm.weight",err),*out_weight=weight(executor,layer,"ssm_out.weight",err);if(!qkv_weight||!z_weight||!alpha_weight||!beta_weight||!conv_weight||!a_decay||!dt_bias||!norm_weight||!out_weight)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(executor->model);
    static atomic_int gdn_diag_done=0;
    if(gdn_diag_enabled()&&!gdn_diag_done){gdn_diag_done=1;const float *a_vals=fg_vk_tensor_map(a_decay),*dt_vals=fg_vk_tensor_map(dt_bias);fprintf(stderr,"GDN_DIAG layer=%u ssm_a[0..7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f dt_bias[0..3]=%.4f,%.4f,%.4f,%.4f\n",layer,a_vals[0],a_vals[1],a_vals[2],a_vals[3],a_vals[4],a_vals[5],a_vals[6],a_vals[7],dt_vals[0],dt_vals[1],dt_vals[2],dt_vals[3]);}
    fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"gdn_projection",err):FG_OK;
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_gdn_project_decode(vk,executor->gdn_qkv,executor->gdn_z,executor->gdn_alpha,executor->gdn_beta,qkv_weight,z_weight,alpha_weight,beta_weight,hidden,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gdn_recurrent",err);
    if(status==FG_OK)status=fg_vk_gdn_conv_decode(vk,executor->gdn_conv_output,executor->gdn_state[layer].conv_state,executor->gdn_qkv,conv_weight,10240u,err);
    if(status==FG_OK)status=fg_vk_gdn_recurrent_algebraic(vk,executor->gdn_core,executor->gdn_state[layer].recurrent_state,executor->gdn_conv_output,executor->gdn_z,executor->gdn_alpha,executor->gdn_beta,a_decay,dt_bias,norm_weight,48u,16u,128u,1e-6f,err);
    if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}
    if(status==FG_OK&&layer==0u&&gdn_diag_enabled()){const float *alpha_vals=fg_vk_tensor_map(executor->gdn_alpha),*a_vals2=fg_vk_tensor_map(a_decay),*dt_vals2=fg_vk_tensor_map(dt_bias);float sp0=alpha_vals[0]+dt_vals2[0];sp0=sp0>0.0f?sp0+logf(1.0f+expf(-sp0)):logf(1.0f+expf(sp0));float example_decay=expf(a_vals2[0]*sp0);fprintf(stderr,"GDN_DECAY layer=0 alpha[0]=%.4f softplus=%.4f a[0]=%.4f decay[0]=%.6f\n",alpha_vals[0],sp0,a_vals2[0],example_decay);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gdn_output",err);
    if(status==FG_OK){status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_output,out_weight,executor->gdn_core,6144u,2560u,1u,1.0f,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}}if(status==FG_OK){*output=executor->gdn_output;}return status;
}

fg_status fg_owner_gdn_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t token_count,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    if(!executor||!hidden||!output||!token_count||token_count>executor->max_tokens||!owns_layer(executor,layer)||(layer&3u)==3u||!executor->gdn_state[layer].conv_state||!executor->gdn_state[layer].recurrent_state){fg_error_set(err,FG_ERR_MISMATCH,"GDN prefill is not on an owned linear-attention layer or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}fg_vk_tensor *qkv_weight=weight(executor,layer,"attn_qkv.weight",err),*z_weight=weight(executor,layer,"attn_gate.weight",err),*alpha_weight=weight(executor,layer,"ssm_alpha.weight",err),*beta_weight=weight(executor,layer,"ssm_beta.weight",err),*conv_weight=weight(executor,layer,"ssm_conv1d.weight",err),*a_decay=weight(executor,layer,"ssm_a",err),*dt_bias=weight(executor,layer,"ssm_dt.bias",err),*norm_weight=weight(executor,layer,"ssm_norm.weight",err),*out_weight=weight(executor,layer,"ssm_out.weight",err);if(!qkv_weight||!z_weight||!alpha_weight||!beta_weight||!conv_weight||!a_decay||!dt_bias||!norm_weight||!out_weight)return FG_ERR_MISMATCH;
    fg_vk_context *vk=fg_model_vk(executor->model);fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_qkv,qkv_weight,hidden,2560u,10240u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_z,z_weight,hidden,2560u,6144u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_dense_f32(vk,executor->gdn_alpha,alpha_weight,hidden,2560u,48u,token_count,err);if(status==FG_OK)status=fg_vk_dense_f32(vk,executor->gdn_beta,beta_weight,hidden,2560u,48u,token_count,err);if(status==FG_OK)status=fg_vk_gdn_conv_prefill(vk,executor->gdn_conv_output,executor->gdn_state[layer].conv_state,executor->gdn_qkv,conv_weight,10240u,token_count,err);if(status==FG_OK)status=fg_vk_gdn_recurrent_prefill(vk,executor->gdn_core,executor->gdn_state[layer].recurrent_state,executor->gdn_conv_output,executor->gdn_z,executor->gdn_alpha,executor->gdn_beta,a_decay,dt_bias,norm_weight,48u,16u,128u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,executor->gdn_output,out_weight,executor->gdn_core,6144u,2560u,token_count,1.0f,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status==FG_OK)*output=executor->gdn_output;return status;
}

fg_status fg_owner_ple_decode(fg_owner_executor *e,const fg_vk_tensor *hyper,const fg_vk_tensor *embedding,fg_vk_tensor **output,fg_error *err){
    if(!e||!hyper||!embedding||!output||!owns_layer(e,1u)||!e->ple_state){fg_error_set(err,FG_ERR_MISMATCH,"PLE decode is not on the layer-1 owner");return FG_ERR_MISMATCH;}fg_vk_tensor *key_weight=weight(e,1u,"ple_key.weight",err),*value_weight=weight(e,1u,"ple_value.weight",err),*key_norm=weight(e,1u,"ple_norm_key.weight",err),*query_norm=weight(e,1u,"ple_norm_query.weight",err),*conv_norm=weight(e,1u,"ple_norm_conv.weight",err),*conv_weight=weight(e,1u,"ple_conv1d.weight",err);if(!key_weight||!value_weight||!key_norm||!query_norm||!conv_norm||!conv_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_key,key_weight,embedding,2560u,10240u,1u,1.0f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_value,value_weight,embedding,2560u,2560u,1u,1.0f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_key_norm,e->ple_key,key_norm,2560u,4u,1u,1e-6f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_query_norm,hyper,query_norm,2560u,4u,1u,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_gate(vk,e->ple_gated,e->ple_key_norm,e->ple_query_norm,e->ple_value,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_gated_norm,e->ple_gated,conv_norm,2560u,4u,1u,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_conv_decode(vk,e->ple_output,e->ple_state,e->ple_gated,e->ple_gated_norm,conv_weight,err);if(status==FG_OK)status=fg_vk_add_f32(vk,e->ple_added,hyper,e->ple_output,10240u,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status==FG_OK)*output=e->ple_added;return status;
}

fg_status fg_owner_ple_prefill(fg_owner_executor *e,const fg_vk_tensor *hyper,const fg_vk_tensor *embedding,uint32_t token_count,fg_vk_tensor **output,fg_error *err){
    if(!e||!hyper||!embedding||!output||!token_count||token_count>e->max_tokens||!owns_layer(e,1u)||!e->ple_state){fg_error_set(err,FG_ERR_MISMATCH,"PLE prefill is not on the layer-1 owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}fg_vk_tensor *key_weight=weight(e,1u,"ple_key.weight",err),*value_weight=weight(e,1u,"ple_value.weight",err),*key_norm=weight(e,1u,"ple_norm_key.weight",err),*query_norm=weight(e,1u,"ple_norm_query.weight",err),*conv_norm=weight(e,1u,"ple_norm_conv.weight",err),*conv_weight=weight(e,1u,"ple_conv1d.weight",err);if(!key_weight||!value_weight||!key_norm||!query_norm||!conv_norm||!conv_weight)return FG_ERR_MISMATCH;fg_vk_context *vk=fg_model_vk(e->model);fg_status status=fg_vk_begin(vk,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_key,key_weight,embedding,2560u,10240u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->ple_value,value_weight,embedding,2560u,2560u,token_count,1.0f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_key_norm,e->ple_key,key_norm,2560u,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_query_norm,hyper,query_norm,2560u,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_gate_prefill(vk,e->ple_gated,e->ple_key_norm,e->ple_query_norm,e->ple_value,token_count,err);if(status==FG_OK)status=fg_vk_group_rms_norm(vk,e->ple_gated_norm,e->ple_gated,conv_norm,2560u,4u,token_count,1e-6f,err);if(status==FG_OK)status=fg_vk_ple_conv_prefill(vk,e->ple_output,e->ple_state,e->ple_gated,e->ple_gated_norm,conv_weight,token_count,err);if(status==FG_OK)status=fg_vk_add_f32(vk,e->ple_added,hyper,e->ple_output,token_count*10240u,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}if(status==FG_OK)*output=e->ple_added;return status;
}

fg_status fg_owner_qsa_open(fg_owner_executor *executor,const char *state_path,bool create,fg_error *err){
    if(!executor||!state_path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner QSA open arguments");return FG_ERR_ARGUMENT;}
    if(executor->qsa){fg_error_set(err,FG_ERR_MISMATCH,"owner QSA session is already open");return FG_ERR_MISMATCH;}
    return fg_qsa_session_open(&executor->qsa,executor->model,state_path,create,err);
}
fg_status fg_owner_qsa_open_decode(fg_owner_executor *executor,const char *state_path,uint32_t resident_tokens,uint32_t batch_size,fg_error *err){
    if(!executor||!state_path){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner QSA decode open arguments");return FG_ERR_ARGUMENT;}
    if(executor->qsa){fg_error_set(err,FG_ERR_MISMATCH,"owner QSA session is already open");return FG_ERR_MISMATCH;}
    return fg_qsa_session_open_decode(&executor->qsa,executor->model,state_path,resident_tokens,batch_size,err);
}
fg_status fg_owner_qsa_open_mirror(fg_owner_executor *executor,uint32_t logical_context,
                                   uint32_t hot_tokens,uint32_t cache_pages,uint32_t batch_size,
                                   fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                   fg_error *err){
    if(!executor){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner QSA mirror open");return FG_ERR_ARGUMENT;}
    if(executor->qsa){fg_error_set(err,FG_ERR_MISMATCH,"owner QSA session is already open");return FG_ERR_MISMATCH;}
    return fg_qsa_session_open_mirror_with_scratch(
        &executor->qsa,executor->model,logical_context,hot_tokens,cache_pages,batch_size,
        executor->attention_family_scratch,fetch_pages,fetch_opaque,err);
}
void fg_owner_qsa_set_tokens(fg_owner_executor *executor,uint32_t tokens){if(executor&&executor->qsa)fg_qsa_session_set_tokens(executor->qsa,tokens);}

fg_status fg_owner_qsa_decode(fg_owner_executor *executor,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    if(!executor||!executor->qsa||!owns_layer(executor,layer)||(layer&3u)!=3u){fg_error_set(err,FG_ERR_MISMATCH,"QSA decode is not on an initialized QSA layer owner");return FG_ERR_MISMATCH;}
    return fg_qsa_session_decode(executor->qsa,layer,token,position,hidden,output,err);
}

fg_status fg_owner_qsa_prefill(fg_owner_executor *executor,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint32_t token_count,const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err){
    if(!executor||!executor->qsa||!owns_layer(executor,layer)||(layer&3u)!=3u||!token_count||token_count>executor->max_tokens){fg_error_set(err,FG_ERR_MISMATCH,"QSA prefill is not on an initialized QSA layer owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}
    return fg_qsa_session_prefill(executor->qsa,layer,first_token,positions,token_count,hidden,output,err);
}
fg_status fg_owner_qsa_page_records(const fg_owner_executor *executor,uint32_t layer,
                                    uint32_t block,const uint8_t **records,fg_error *err){
    if(!executor||!executor->qsa){
        fg_error_set(err,FG_ERR_ARGUMENT,"owner QSA page lookup is unavailable");
        return FG_ERR_ARGUMENT;
    }
    return fg_qsa_session_page_records(executor->qsa,layer,block,records,err);
}
void fg_owner_qsa_page_published(fg_owner_executor *executor,uint32_t layer,uint32_t block){
    if(executor)fg_qsa_session_page_published(executor->qsa,layer,block);
}

static float tensor_l2(const fg_vk_tensor *t,uint32_t n){const float *p=fg_vk_tensor_map((fg_vk_tensor *)t);if(!p)return -1.0f;double s=0.0;for(uint32_t i=0;i<n;i++)s+=(double)p[i]*p[i];return (float)sqrt(s/n);}

fg_status fg_owner_decode_layer(fg_owner_executor *e,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,fg_owner_expert_dispatch_fn dispatch,void *dispatch_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!position||!hyper_input||!dispatch||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"text layer decode is not on its owner");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embedding!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 PLE embedding presence mismatch");return FG_ERR_MISMATCH;}double t0=ts_ms();fg_vk_context *vk=fg_model_vk(e->model);const fg_vk_tensor *layer_input=hyper_input;if(layer==1u){fg_vk_tensor *ple_input=NULL;fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"ple",err):FG_OK;if(status==FG_OK)status=fg_owner_ple_decode(e,hyper_input,ngram_embedding,&ple_input,err);if(status!=FG_OK)return status;layer_input=ple_input;}double t_ple=ts_ms();
    int diag=token<30u; /* first ~11 decode tokens */
    /* ---- FUSED BATCH 1: gr_read(attn) + attention + gr_write + gr_read(FFN) + router ---- */
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;
    fg_status status=fg_vk_profile_active(vk)?fg_vk_profile_set_scope(vk,"gr_attn_read",err):FG_OK;if(status==FG_OK)status=fg_vk_begin(vk,err);if(status==FG_OK){status=fg_owner_gr_read(e,layer,false,layer_input,&mixed,&residual,&injection,err);}double t_gr1=ts_ms();
    if(status==FG_OK){status=(layer&3u)==3u?fg_owner_qsa_decode(e,layer,token,position,mixed,&block,err):fg_owner_gdn_decode(e,layer,mixed,&block,err);}double t_attn=ts_ms();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_write",err);
    if(status==FG_OK){status=fg_owner_gr_write(e,residual,block,injection,&after_attention,err);}double t_grw1=ts_ms();
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_read",err);
    if(status==FG_OK){status=fg_owner_gr_read(e,layer,true,after_attention,&mixed,&residual,&injection,err);}double t_gr2=ts_ms();
    /* router matmul — stays in the same fused batch */
    fg_vk_tensor *router_w=status==FG_OK?weight(e,layer,"ffn_gate_inp.weight",err):NULL;
    if(status==FG_OK&&!router_w)status=FG_ERR_MISMATCH;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router",err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router_w,mixed,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,1u,err);
    if(status==FG_OK){fg_status es=fg_vk_end(vk,err);if(es!=FG_OK)status=es;} /* SYNC 1: read router logits for CPU top-K */
    else if(fg_vk_batch_active(vk))fg_vk_end(vk,err); /* clean up batch on error path */
    double t_router=ts_ms();
    /* CPU top-K routing */
    uint16_t expert_ids[FG_TOP_K];float gates[FG_TOP_K];const uint8_t *activation=NULL;
    if(status==FG_OK){float router_local[FG_EXPERT_COUNT];memcpy(router_local,fg_vk_tensor_map(e->router_logits),sizeof(router_local));uint32_t ids[FG_TOP_K];status=fg_q38_router_topk(router_local,FG_EXPERT_COUNT,FG_TOP_K,ids,gates,err);for(uint32_t s=0;status==FG_OK&&s<FG_TOP_K;s++)expert_ids[s]=(uint16_t)ids[s];}
    /* ---- FUSED BATCH 2: quantize + shared expert ---- */
    fg_vk_tensor *shared_gate_w=NULL,*gate_w=NULL,*up_w=NULL,*down_w=NULL;
    if(status==FG_OK){shared_gate_w=weight(e,layer,"ffn_gate_inp_shexp.weight",err);gate_w=weight(e,layer,"ffn_gate_shexp.weight",err);up_w=weight(e,layer,"ffn_up_shexp.weight",err);down_w=weight(e,layer,"ffn_down_shexp.weight",err);if(!shared_gate_w||!gate_w||!up_w||!down_w)status=FG_ERR_MISMATCH;}
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router_quantization",err);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,mixed,FG_HIDDEN_SIZE,1u,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"shared_expert",err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,e->shared_up,640u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_output,down_w,e->shared_mid,640u,FG_HIDDEN_SIZE,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->shared_scalar,shared_gate_w,mixed,FG_HIDDEN_SIZE,1u,1u,err);
    if(status==FG_OK){fg_status es=fg_vk_end(vk,err);if(es!=FG_OK)status=es;} /* SYNC 2: read activation + shared output */
    else if(fg_vk_batch_active(vk))fg_vk_end(vk,err);
    if(status==FG_OK)activation=fg_vk_tensor_map(e->activation_q8k);
    double t_mprep=ts_ms();
    /* expert dispatch + reduce + final gr_write (unchanged) */
    fg_expert_result results[FG_GROUP_SIZE];uint32_t result_count=0;if(status==FG_OK)status=dispatch(dispatch_context,layer,token,expert_ids,gates,activation,results,&result_count,err);double t_exp=ts_ms();if(status==FG_OK)status=fg_owner_moe_reduce(e,layer,token,expert_ids,gates,results,result_count,&block,err);double t_red=ts_ms();if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);if(status==FG_OK)status=fg_owner_gr_write(e,residual,block,injection,output,err);double t_end=ts_ms();
    float attn_block_l2=diag&&status==FG_OK?tensor_l2(block,FG_HIDDEN_SIZE):0.0f;
    if(diag&&status==FG_OK){fprintf(stderr,"layer[%u] t=%u in=%.4f attn_blk=%.4f attn=%.4f moe=%.4f out=%.4f exp=%u,%u,%u",layer,token,tensor_l2(layer_input,FG_HYPER_WIDTH),attn_block_l2,tensor_l2(after_attention,FG_HYPER_WIDTH),tensor_l2(block,FG_HIDDEN_SIZE),tensor_l2(*output,FG_HYPER_WIDTH),expert_ids[0],expert_ids[1],expert_ids[2]);if(layer==0u&&(layer&3u)!=3u)fprintf(stderr," gdn_state=%.6f",tensor_l2(e->gdn_state[layer].recurrent_state,48u*128u*128u));fprintf(stderr,"\n");}
    if(token>=26u&&token<32u){fprintf(stderr,"TIMING layer[%u] t=%u total=%.1f ple=%.1f gr_read=%.1f attn=%.1f gr_write=%.1f gr_read2=%.1f router=%.1f moe_prep=%.1f expert=%.1f moe_red=%.1f gr_write2=%.1f\n",layer,token,t_end-t0,t_ple-t0,t_gr1-t_ple,t_attn-t_gr1,t_grw1-t_attn,t_gr2-t_grw1,t_router-t_gr2,t_mprep-t_router,t_exp-t_mprep,t_red-t_exp,t_end-t_red);}
    return status;
}

fg_status fg_owner_decode_layer_async(fg_owner_executor *e,uint32_t layer,uint32_t token,const uint32_t position[3],const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embedding,fg_owner_expert_fire_fn fire,fg_owner_expert_collect_fn collect,void *dispatch_context,fg_owner_qsa_decode_dispatch_fn qsa_dispatch,void *qsa_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!position||!hyper_input||!fire||!collect||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"async decode layer precondition");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embedding!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 PLE embedding presence mismatch");return FG_ERR_MISMATCH;}if((layer==0u&&e->pending_write.active)||(layer>0u&&(!e->pending_write.active||e->pending_write.layer+1u!=layer||e->pending_write.token!=token||e->pending_write.output!=hyper_input))){fg_error_set(err,FG_ERR_MISMATCH,"deferred residual write does not match successor layer");return FG_ERR_MISMATCH;}double t0=ts_ms();fg_vk_context *vk=fg_model_vk(e->model);bool ep_trace=fg_vk_profile_active(vk);uint64_t trace_start=ep_trace?wall_ns():0;fg_vk_counters counters_before={0};if(ep_trace)fg_vk_get_counters(vk,&counters_before);fg_status status=FG_OK;const fg_vk_tensor *layer_input=hyper_input;if(layer==1u){fg_vk_tensor *ple_input=NULL;if(fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);if(status==FG_OK)status=fg_vk_begin(vk,err);if(status==FG_OK)status=flush_gr_write(e,err);if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"ple",err);if(status==FG_OK)status=fg_owner_ple_decode(e,hyper_input,ngram_embedding,&ple_input,err);if(status==FG_OK){fg_status end_status=fg_vk_end(vk,err);if(end_status!=FG_OK)status=end_status;}else if(fg_vk_batch_active(vk))fg_vk_end(vk,err);if(status!=FG_OK)return status;layer_input=ple_input;}
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;
    /* Quantization joins Batch 1 so routed work can fire before shared expert compute. */
    if(status==FG_OK&&layer>1u&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&layer>1u)status=flush_gr_write(e,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_read",err);
    if(status==FG_OK){status=fg_owner_gr_read(e,layer,false,layer_input,&mixed,&residual,&injection,err);}
    bool remote_qsa=(layer&3u)==3u&&qsa_dispatch;
    if(status==FG_OK&&remote_qsa){status=fg_vk_end(vk,err);if(status==FG_OK)status=qsa_dispatch(qsa_context,layer,token,position,mixed,&block,err);if(status==FG_OK)status=fg_vk_begin(vk,err);}
    else if(status==FG_OK){status=(layer&3u)==3u?fg_owner_qsa_decode(e,layer,token,position,mixed,&block,err):fg_owner_gdn_decode(e,layer,mixed,&block,err);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_attn_write",err);
    if(status==FG_OK){status=fg_owner_gr_write(e,residual,block,injection,&after_attention,err);}
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_read",err);
    if(status==FG_OK){status=fg_owner_gr_read(e,layer,true,after_attention,&mixed,&residual,&injection,err);}
    fg_vk_tensor *router_w=status==FG_OK?weight(e,layer,"ffn_gate_inp.weight",err):NULL;
    if(status==FG_OK&&!router_w)status=FG_ERR_MISMATCH;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router",err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->router_logits,router_w,mixed,FG_HIDDEN_SIZE,FG_EXPERT_COUNT,1u,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"router_quantization",err);
    if(status==FG_OK)status=fg_vk_quantize_q8_k(vk,e->activation_q8k,mixed,FG_HIDDEN_SIZE,1u,err);
    if(status==FG_OK){fg_status es=fg_vk_end(vk,err);if(es!=FG_OK)status=es;} /* SYNC 1: router + activation */
    else if(fg_vk_batch_active(vk))fg_vk_end(vk,err);
    double t_sync1=ts_ms();uint64_t trace_sync1=ep_trace?wall_ns():0;
    uint16_t expert_ids[FG_TOP_K];float gates[FG_TOP_K];
    if(status==FG_OK){float router_local[FG_EXPERT_COUNT];memcpy(router_local,fg_vk_tensor_map(e->router_logits),sizeof(router_local));uint32_t ids[FG_TOP_K];status=fg_q38_router_topk(router_local,FG_EXPERT_COUNT,FG_TOP_K,ids,gates,err);for(uint32_t s=0;status==FG_OK&&s<FG_TOP_K;s++)expert_ids[s]=(uint16_t)ids[s];}
    const uint8_t *activation=status==FG_OK?fg_vk_tensor_map(e->activation_q8k):NULL;
    fg_vk_tensor *shared_gate_w=NULL,*gate_w=NULL,*up_w=NULL,*down_w=NULL;
    if(status==FG_OK){shared_gate_w=weight(e,layer,"ffn_gate_inp_shexp.weight",err);gate_w=weight(e,layer,"ffn_gate_shexp.weight",err);up_w=weight(e,layer,"ffn_up_shexp.weight",err);down_w=weight(e,layer,"ffn_down_shexp.weight",err);if(!shared_gate_w||!gate_w||!up_w||!down_w)status=FG_ERR_MISMATCH;}
    bool fire_called=false;
    if(status==FG_OK){
        fire_called=true;
        status=fire(dispatch_context,layer,token,expert_ids,gates,activation,err);
    }
    double t_fire=ts_ms();uint64_t trace_fire=ep_trace?wall_ns():0;
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"shared_expert",err);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_gate,gate_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_up,up_w,mixed,FG_HIDDEN_SIZE,640u,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_swiglu(vk,e->shared_mid,e->shared_gate,e->shared_up,640u,err);
    if(status==FG_OK)status=fg_vk_dense_q8_0_f32(vk,e->shared_output,down_w,e->shared_mid,640u,FG_HIDDEN_SIZE,1u,1.0f,err);
    if(status==FG_OK)status=fg_vk_dense_f32(vk,e->shared_scalar,shared_gate_w,mixed,FG_HIDDEN_SIZE,1u,1u,err);
    if(status==FG_OK){fg_status es=fg_vk_end(vk,err);if(es!=FG_OK)status=es;}
    else if(fg_vk_batch_active(vk))fg_vk_end(vk,err);
    double t_sync2=ts_ms();uint64_t trace_sync2=ep_trace?wall_ns():0;
    fg_expert_result results[FG_GROUP_SIZE];uint32_t result_count=0;
    if(fire_called){
        fg_error collect_error={0};
        fg_status collect_status=collect(dispatch_context,layer,token,results,&result_count,
                                         status==FG_OK?err:&collect_error);
        if(status==FG_OK)status=collect_status;
    }
    double t_collect=ts_ms();uint64_t trace_collect=ep_trace?wall_ns():0;
    if(status==FG_OK)status=fg_owner_moe_reduce(e,layer,token,expert_ids,gates,results,result_count,&block,err);
    if(status==FG_OK&&fg_vk_profile_active(vk))status=fg_vk_profile_set_scope(vk,"gr_ffn_write",err);
    if(status==FG_OK)status=layer+1u<FG_LAYER_COUNT?defer_gr_write(e,layer,token,residual,block,injection,output,err):fg_owner_gr_write(e,residual,block,injection,output,err);
    double t_end=ts_ms();uint64_t trace_end=ep_trace?wall_ns():0;
    if(ep_trace){fg_vk_counters counters_after={0};fg_vk_get_counters(vk,&counters_after);fprintf(stderr,"EP_LAYER_TRACE token=%u layer=%u status=%d total_ms=%.3f sync1_ms=%.3f fire_ms=%.3f shared_ms=%.3f collect_ms=%.3f finish_ms=%.3f submissions=%llu dispatches=%llu start_ns=%llu router_ready_ns=%llu fire_end_ns=%llu shared_end_ns=%llu collect_end_ns=%llu finish_end_ns=%llu\n",token,layer,(int)status,t_end-t0,t_sync1-t0,t_fire-t_sync1,t_sync2-t_fire,t_collect-t_sync2,t_end-t_collect,(unsigned long long)(counters_after.submissions-counters_before.submissions),(unsigned long long)(counters_after.dispatches-counters_before.dispatches),(unsigned long long)trace_start,(unsigned long long)trace_sync1,(unsigned long long)trace_fire,(unsigned long long)trace_sync2,(unsigned long long)trace_collect,(unsigned long long)trace_end);}
    if(token>=26u&&token<32u){fprintf(stderr,"OVERLAP_TIMING layer[%u] t=%u total=%.1f sync1=%.1f fire=%.1f shared=%.1f collect=%.1f reduce+grw=%.1f\n",layer,token,t_end-t0,t_sync1-t0,t_fire-t_sync1,t_sync2-t_fire,t_collect-t_sync2,t_end-t_collect);}
    return status;
}

fg_status fg_owner_prefill_layer(fg_owner_executor *e,uint32_t layer,uint32_t first_token,const uint32_t *positions,uint16_t token_count,const fg_vk_tensor *hyper_input,const fg_vk_tensor *ngram_embeddings,fg_owner_prefill_dispatch_fn dispatch,void *dispatch_context,fg_owner_qsa_prefill_dispatch_fn qsa_dispatch,void *qsa_context,fg_vk_tensor **output,fg_error *err){
    if(!e||!positions||!token_count||token_count>e->max_tokens||!hyper_input||!dispatch||!output||!owns_layer(e,layer)){fg_error_set(err,FG_ERR_MISMATCH,"text layer prefill is not on its owner or exceeds the sealed microbatch");return FG_ERR_MISMATCH;}if((layer==1u)!=(ngram_embeddings!=NULL)){fg_error_set(err,FG_ERR_MISMATCH,"layer-1 batched PLE embedding presence mismatch");return FG_ERR_MISMATCH;}
    fg_vk_context *vk=fg_model_vk(e->model);bool profiling=fg_vk_profile_active(vk);double t0=profiling?ts_ms():0.0;const fg_vk_tensor *layer_input=hyper_input;
    if(layer==1u){fg_vk_tensor *ple_input=NULL;fg_status status=profiling?fg_vk_profile_set_scope(vk,"ple_prefill",err):FG_OK;if(status==FG_OK)status=fg_owner_ple_prefill(e,hyper_input,ngram_embeddings,token_count,&ple_input,err);if(status!=FG_OK)return status;layer_input=ple_input;}
    double t_ple=profiling?ts_ms():0.0;
    fg_vk_tensor *mixed=NULL,*injection=NULL,*block=NULL,*after_attention=NULL;const fg_vk_tensor *residual=NULL;fg_status status=profiling?fg_vk_profile_set_scope(vk,"gr_attn_read_prefill",err):FG_OK;
    if(status==FG_OK)status=fg_owner_gr_read_batch(e,layer,false,layer_input,token_count,&mixed,&residual,&injection,err);
    double t_attn_read=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,(layer&3u)==3u?"qsa_prefill":"gdn_prefill",err);
    if(status==FG_OK&&(layer&3u)==3u&&qsa_dispatch)
        status=qsa_dispatch(qsa_context,layer,first_token,positions,token_count,mixed,&block,err);
    else if(status==FG_OK&&(layer&3u)==3u){
        bool qsa_batch=false;status=fg_vk_begin(vk,err);qsa_batch=status==FG_OK;
        if(status==FG_OK)status=fg_owner_qsa_prefill(e,layer,first_token,positions,
                                                     token_count,mixed,&block,err);
        if(status==FG_OK){status=fg_vk_end(vk,err);qsa_batch=false;}
        if(qsa_batch){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    }else if(status==FG_OK)
        status=fg_owner_gdn_prefill(e,layer,token_count,mixed,&block,err);
    double t_attention=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_attn_write_prefill",err);
    if(status==FG_OK)status=fg_owner_gr_write_batch(e,residual,block,injection,token_count,&after_attention,err);
    double t_attn_write=profiling?ts_ms():0.0;
    if(status!=FG_OK)return status;
    if(profiling)status=fg_vk_profile_set_scope(vk,"gr_ffn_read_prefill",err);
    if(status==FG_OK)status=fg_owner_gr_read_batch(e,layer,true,after_attention,token_count,&mixed,&residual,&injection,err);
    double t_ffn_read=profiling?ts_ms():0.0;
    uint16_t expert_ids[FG_PREFILL_MAX_PAIRS];float gates[FG_PREFILL_MAX_PAIRS];const uint8_t *activations=NULL;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"router_prefill",err);
    if(status==FG_OK)status=fg_owner_moe_prepare_batch(e,layer,mixed,token_count,expert_ids,gates,&activations,err);
    double t_router=profiling?ts_ms():0.0;
    fg_prefill_result results[FG_GROUP_SIZE]={0};uint32_t result_count=0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"expert_dispatch_prefill",err);
    if(status==FG_OK)status=dispatch(dispatch_context,layer,first_token,token_count,expert_ids,gates,activations,results,&result_count,err);
    double t_dispatch=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"expert_reduce_prefill",err);
    if(status==FG_OK)status=fg_owner_moe_reduce_batch(e,layer,first_token,token_count,expert_ids,gates,results,result_count,&block,err);
    double t_reduce=profiling?ts_ms():0.0;
    if(status==FG_OK&&profiling)status=fg_vk_profile_set_scope(vk,"gr_ffn_write_prefill",err);
    if(status==FG_OK)status=fg_owner_gr_write_batch(e,residual,block,injection,token_count,output,err);
    if(profiling){double t_end=ts_ms();fprintf(stderr,"PREFILL_PROFILE_STAGE first=%u tokens=%u layer=%u ple_ms=%.3f attn_read_ms=%.3f attention_ms=%.3f attn_write_ms=%.3f ffn_read_ms=%.3f router_ms=%.3f dispatch_ms=%.3f reduce_ms=%.3f final_write_ms=%.3f total_ms=%.3f\n",first_token,token_count,layer,t_ple-t0,t_attn_read-t_ple,t_attention-t_attn_read,t_attn_write-t_attention,t_ffn_read-t_attn_write,t_router-t_ffn_read,t_dispatch-t_router,t_reduce-t_dispatch,t_end-t_reduce,t_end-t0);}
    return status;
}
