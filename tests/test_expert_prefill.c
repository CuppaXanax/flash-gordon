#include "fg_expert.h"
#include "fg_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* In-memory model fixture; executor, schedules, shaders and quantization are real. */
struct fg_model {
    fg_manifest manifest;
    fg_vk_context *vk;
    fg_vk_tensor *weights[3];
    fg_tensor_record records[3];
};
fg_vk_context *fg_model_vk(fg_model *m){return m->vk;}
const fg_manifest *fg_model_manifest(const fg_model *m){return &m->manifest;}
uint32_t fg_model_rank(const fg_model *m){(void)m;return 0u;}
const fg_tensor_record *fg_model_tensor_record(const fg_model *m,const char *name){
    for(uint32_t i=0;i<3u;i++)if(!strcmp(name,m->records[i].name))return &m->records[i];
    return NULL;
}
fg_vk_tensor *fg_model_tensor(fg_model *m,const char *name){
    const fg_tensor_record *r=fg_model_tensor_record(m,name);
    return r?m->weights[r-m->records]:NULL;
}

static fg_status make_weights(fg_model *m,uint32_t family,fg_error *err){
    const char *names[]={"gate","up","down"};
    uint32_t input=family==2u?640u:FG_HIDDEN_SIZE;
    uint32_t output=family==2u?FG_HIDDEN_SIZE:640u;
    uint32_t type=family==2u?7u:family==1u?13u:12u;
    uint64_t stride=family==2u?fg_q5_1_cooked_matrix_bytes(input,output):
        fg_k_quant_cooked_matrix_bytes(input,output,type);
    fg_tensor_record *r=&m->records[family];
    snprintf(r->name,sizeof(r->name),"blk.0.ffn_%s_exps.weight.rank0",names[family]);
    r->dims=3u;r->shape[0]=input;r->shape[1]=output;
    r->shape[2]=FG_EXPERTS_PER_RANK;r->bytes=stride*FG_EXPERTS_PER_RANK;
    r->ggml_type=type;r->kind=FG_TENSOR_ROUTED_EXPERT;
    fg_status status=fg_vk_tensor_create(m->vk,r->bytes,&m->weights[family],err);
    if(status!=FG_OK)return status;
    fg_vk_tensor_set_format(m->weights[family],family==2u?
        FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED:FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED);
    uint8_t *raw=malloc(stride),*cooked=malloc(stride);
    if(!raw||!cooked){free(cooked);free(raw);return FG_ERR_OOM;}
    const uint32_t locals[]={0u,1u,127u};
    for(uint32_t e=0;status==FG_OK&&e<3u;e++){
        for(uint64_t i=0;i<stride;i++)raw[i]=(uint8_t)(i*13u+e*37u+family*11u);
        uint32_t block=family==2u?24u:type==12u?144u:176u;
        for(uint64_t i=0;i<stride;i+=block){
            /* Small finite scales keep all three projections well conditioned. */
            raw[i]=0u;raw[i+1u]=0x14u;
            raw[i+2u]=0u;raw[i+3u]=family==2u?0u:0x10u;
        }
        bool ok=family==2u?fg_cook_q5_1_rows(raw,cooked,stride,input,output):
            fg_cook_k_quant_rows(raw,cooked,stride,input,output,type);
        if(!ok)status=FG_ERR_FORMAT;
        if(status==FG_OK)status=fg_vk_tensor_write(m->weights[family],
            stride*locals[e],cooked,stride,err);
    }
    free(cooked);free(raw);return status;
}

int main(void){
    enum{TOKENS=17,PAIRS=24};
    fg_model *model=calloc(1,sizeof(*model));
    if(!model)return 1;
    fg_error err={0};
    fg_status status=fg_vk_open(&model->vk,&err);
    if(status==FG_ERR_UNAVAILABLE){free(model);return 77;}
    fg_manifest_init(&model->manifest);
    model->manifest.prefill_microbatch=TOKENS;
    /* Non-default placement: global IDs 3, 7, 511 map to local 0, 1, 127. */
    const uint8_t owners[]={1u,3u,5u,0u};
    for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)model->manifest.expert_rank[0][e]=owners[e%4u];
    for(uint32_t f=0;status==FG_OK&&f<3u;f++)status=make_weights(model,f,&err);
    fg_expert_executor *executor=NULL;
    if(status==FG_OK)status=fg_expert_executor_create(&executor,model,&err);
    float *input=malloc(TOKENS*FG_HIDDEN_SIZE*sizeof(float));
    uint8_t *activation=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES);
    float *output=malloc(PAIRS*FG_HIDDEN_SIZE*sizeof(float));
    if(!input||!activation||!output)status=FG_ERR_OOM;
    if(status==FG_OK){
        for(uint32_t i=0;i<TOKENS*FG_HIDDEN_SIZE;i++)input[i]=sinf((float)i*0.017f);
        fg_quantize_q8_k(input,activation,TOKENS*FG_HIDDEN_SIZE);
    }
    fg_prefill_pair pairs[PAIRS];
    for(uint32_t i=0;i<PAIRS;i++){
        uint32_t token=i<TOKENS?i:i-TOKENS;
        pairs[PAIRS-1u-i]=(fg_prefill_pair){.token_slot=(uint16_t)token,
            .routing_slot=(uint8_t)((token*7u+(i<TOKENS?0u:1u))%FG_TOP_K),
            .expert_id=(uint16_t)(i<TOKENS?3u:i==PAIRS-1u?7u:511u),.gate=0.03f*(float)(i%7u+1u)};
    }
    fg_prefill_work work={.layer=0u,.source_rank=0u,.destination_rank=0u,
        .first_position=51u,.token_count=TOKENS,.pair_count=PAIRS,
        .activations_q8k=activation,.pairs=pairs};
    fg_prefill_result result={0};fg_prefill_result_pair result_pairs[PAIRS];
    fg_vk_counters before={0},after={0};
    fg_vk_get_counters(model->vk,&before);
    if(status==FG_OK)status=fg_expert_prefill(executor,&work,&result,
        result_pairs,PAIRS,output,PAIRS*FG_HIDDEN_SIZE,&err);
    fg_vk_get_counters(model->vk,&after);
    if(status==FG_OK&&(after.submissions-before.submissions!=1u||
       after.dispatches-before.dispatches!=5u)){
        fg_error_set(&err,FG_ERR_MISMATCH,"prefill did not use one five-dispatch submission");
        status=FG_ERR_MISMATCH;
    }
    float reference[TOKENS*FG_HIDDEN_SIZE]={0};
    for(uint32_t i=0;status==FG_OK&&i<PAIRS;i++){
        fg_decode_work one={.layer=0u,.source_rank=0u,.destination_rank=0u,
            .position=51u+pairs[i].token_slot,.selected_count=1u};
        one.expert_ids[0]=pairs[i].expert_id;one.routing_slots[0]=pairs[i].routing_slot;
        one.gates[0]=1.0f;
        memcpy(one.activation_q8k,activation+pairs[i].token_slot*FG_Q8K_ACTIVATION_BYTES,
               FG_Q8K_ACTIVATION_BYTES);
        fg_expert_result decoded={0};
        status=fg_expert_decode(executor,&one,&decoded,&err);
        if(status==FG_OK&&(result_pairs[i].token_slot!=pairs[i].token_slot||
           result_pairs[i].routing_slot!=pairs[i].routing_slot))status=FG_ERR_MISMATCH;
        for(uint32_t j=0;status==FG_OK&&j<FG_HIDDEN_SIZE;j++){
            uint32_t index=pairs[i].token_slot*FG_HIDDEN_SIZE+j;
            reference[index]=fmaf(pairs[i].gate,decoded.outputs[0][j],reference[index]);
        }
    }
    for(uint32_t i=0;status==FG_OK&&i<TOKENS*FG_HIDDEN_SIZE;i++){
        if(!isfinite(output[i])||fabsf(output[i]-reference[i])>3e-4f*fmaxf(1.0f,fabsf(reference[i])))status=FG_ERR_MISMATCH;
    }
    if(status==FG_OK){
        fg_prefill_pair saved=pairs[1];pairs[1]=pairs[0];
        if(fg_expert_prefill(executor,&work,&result,result_pairs,PAIRS,output,
            PAIRS*FG_HIDDEN_SIZE,&err)!=FG_ERR_MISMATCH)status=FG_ERR_MISMATCH;
        pairs[1]=saved;
        /* A rejected route and interleaved decode must not damage the next batch. */
        if(status==FG_OK)status=fg_expert_prefill(executor,&work,&result,
            result_pairs,PAIRS,output,PAIRS*FG_HIDDEN_SIZE,&err);
    }
    if(status==FG_OK){
        /* The single-token tail uses the same grouped implementation. */
        fg_prefill_pair tail[2];float expected[FG_HIDDEN_SIZE];uint32_t count=0u;
        memcpy(expected,output,sizeof(expected));
        for(uint32_t i=0;i<PAIRS;i++)if(pairs[i].token_slot==0u){
            tail[count]=pairs[i];

            count++;
        }
        work.token_count=1u;work.pair_count=(uint16_t)count;work.pairs=tail;
        status=fg_expert_prefill(executor,&work,&result,result_pairs,PAIRS,
            output,PAIRS*FG_HIDDEN_SIZE,&err);
        if(status==FG_OK&&memcmp(output,expected,sizeof(expected)))
            status=FG_ERR_MISMATCH;
    }
    free(output);free(activation);free(input);fg_expert_executor_destroy(executor);
    for(uint32_t f=0;f<3u;f++)fg_vk_tensor_destroy(model->weights[f]);
    fg_vk_close(model->vk);free(model);
    if(status!=FG_OK){fprintf(stderr,"expert prefill: %s\n",err.message);return 1;}
    puts("EP shard-local grouped prefill versus decode: PASS");return 0;
}
