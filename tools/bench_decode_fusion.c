#include "../src/model.c"
#include "fg_expert.h"
#include "fg_owner.h"
#include "fg_qsa_state.h"
#include "fg_mtp.h"
#include <math.h>
#include <time.h>

typedef struct bench_spec {
    const char *name;
    uint32_t k,m,type;
} bench_spec;
static uint32_t bench_seed=1u;

static void check(fg_status status,fg_error *error){
    if(status!=FG_OK){fprintf(stderr,"fusion bench: %s\n",error->message);exit(1);}
}
static double now_us(void){
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e6+ts.tv_nsec*1e-3;
}
static uint32_t random_word(uint32_t *state){
    *state^=*state<<13;*state^=*state>>17;*state^=*state<<5;return *state;
}
static void synthetic_raw(void *data,uint64_t bytes,uint32_t type,bool norm,uint32_t seed){
    uint8_t *p=data;uint32_t state=seed+1u;
    if(type==0u){
        float *f=data;
        for(uint64_t i=0;i<bytes/4u;i++)
            f[i]=norm?1.0f:((int32_t)(random_word(&state)%2001u)-1000)*0.00002f;
        return;
    }
    if(type==30u){
        uint16_t *b=data;
        for(uint64_t i=0;i<bytes/2u;i++)b[i]=(uint16_t)(0x3b00u+(random_word(&state)&255u));
        return;
    }
    uint32_t block=type==12u?144u:type==20u?18u:type==7u?24u:34u;
    for(uint64_t off=0;off<bytes;off+=block){
        for(uint32_t j=0;j<block;j++)p[off+j]=(uint8_t)random_word(&state);
        uint16_t d=fg_f32_to_f16(type==12u?0.0001f:0.0005f);
        memcpy(p+off,&d,2);
        if(type==12u||type==7u){
            uint16_t min=fg_f32_to_f16(type==7u?-0.008f:0.00005f);
            memcpy(p+off+2u,&min,2);
        }
    }
}

static void add_weight(fg_model *model,fg_manifest *manifest,uint32_t layer,
                       bench_spec spec,bool q4,fg_error *error){
    bool expert=strstr(spec.name,"_exps")!=NULL;
    uint32_t type=spec.type,k=spec.k,m=spec.m;
    if(q4&&type==8u&&!expert)type=k%256u==0u?12u:20u;
    uint64_t raw_bytes=0;
    const uint64_t shape[FG_GGUF_MAX_DIMS]={k,m,1u,1u};
    if(!fg_gguf_tensor_bytes(type,2u,shape,&raw_bytes)){
        fprintf(stderr,"invalid synthetic shape %s\n",spec.name);exit(1);
    }
    uint8_t layout=FG_TENSOR_LAYOUT_GGML;
    uint64_t cooked_bytes=raw_bytes;
    fg_vk_tensor_format format=FG_VK_TENSOR_FORMAT_DEFAULT;
    if(expert){
        if(type==12u){
            layout=FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED;
            format=FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED;
            cooked_bytes=fg_k_quant_cooked_matrix_bytes(k,m,type);
        }else{
            layout=FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED;
            format=FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED;
            cooked_bytes=fg_q5_1_cooked_matrix_bytes(k,m);
        }
    }else if(type==8u){
        layout=FG_TENSOR_LAYOUT_Q8_0_COOKED;format=FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
        cooked_bytes=fg_q8_0_cooked_matrix_bytes(k,m);
    }else if(type==12u||type==20u){
        bool compact=fg_dense_prefer_compact(k,m,type);
        layout=compact?FG_TENSOR_LAYOUT_DENSE_COMPACT:FG_TENSOR_LAYOUT_DENSE_SOA;
        format=fg_vk_dense_saved_format(type,layout);
        cooked_bytes=compact?fg_dense_compact_bytes(k,m,type):fg_dense_soa_bytes(k,m,type);
    }
    uint32_t count=expert?FG_EXPERT_COUNT:1u,index=manifest->tensor_count;
    fg_tensor_record record={0};
    if(expert)snprintf(record.name,sizeof(record.name),"blk.%u.%s.rank1",layer,spec.name);
    else snprintf(record.name,sizeof(record.name),"blk.%u.%s",layer,spec.name);
    record.layer=(uint16_t)layer;record.rank=1u;record.expert=UINT16_MAX;
    record.kind=expert?FG_TENSOR_ROUTED_EXPERT:FG_TENSOR_COMMON;
    record.ggml_type=type;record.layout=layout;record.dims=expert?3u:2u;
    record.shape[0]=k;record.shape[1]=m;record.shape[2]=count;
    record.bytes=cooked_bytes*count;record.offset=fg_align_up_u64(model->weight_bytes,FG_ALIGNMENT);
    check(fg_manifest_add_tensor(manifest,&record,error),error);
    check(fg_vk_tensor_create(model->vk,record.bytes,&model->tensor[index],error),error);
    fg_vk_tensor_set_format(model->tensor[index],format);
    uint8_t *raw=malloc(raw_bytes),*cooked=malloc(cooked_bytes);
    if(!raw||!cooked){fprintf(stderr,"weight host allocation failed\n");exit(1);}
    bool norm=strstr(spec.name,"norm")!=NULL;
    synthetic_raw(raw,raw_bytes,type,norm,layer*101u+index+bench_seed*997u);
    if(strcmp(spec.name,"ssm_a")==0)
        for(uint32_t i=0;i<48u;i++)((float *)raw)[i]=-1.0f;
    bool ok=true;
    if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED)ok=fg_cook_q8_0_rows(raw,cooked,cooked_bytes,k,m);
    else if(layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED)ok=fg_cook_k_quant_rows(raw,cooked,cooked_bytes,k,m,type);
    else if(layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED)ok=fg_cook_q5_1_rows(raw,cooked,cooked_bytes,k,m);
    else if(layout==FG_TENSOR_LAYOUT_DENSE_COMPACT)ok=fg_cook_dense_compact(raw,cooked,cooked_bytes,k,m,type);
    else if(layout==FG_TENSOR_LAYOUT_DENSE_SOA)ok=fg_cook_dense_soa(raw,cooked,cooked_bytes,k,m,type);
    else memcpy(cooked,raw,raw_bytes);
    if(!ok){fprintf(stderr,"cooking %s failed\n",record.name);exit(1);}
    uint8_t *mapped=fg_vk_tensor_map(model->tensor[index]);
    for(uint32_t e=0;e<count;e++)memcpy(mapped+(uint64_t)e*cooked_bytes,cooked,cooked_bytes);
    free(cooked);free(raw);model->weight_bytes=record.offset+record.bytes;
}

static fg_status inline_expert(void *opaque,uint32_t layer,const fg_vk_tensor *activation,
                                const fg_vk_tensor *logits,fg_vk_tensor **output,fg_error *error){
    return fg_expert_decode_chain(opaque,layer,activation,logits,output,error);
}

typedef struct bench_prefill {
    fg_expert_executor *expert;
    fg_prefill_pair pairs[FG_MTP_MAX_VERIFY*FG_TOP_K];
    fg_prefill_result_pair result_pairs[FG_MTP_MAX_VERIFY*FG_TOP_K];
    float outputs[FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE];
} bench_prefill;

static fg_status batch_expert(void *opaque,uint32_t layer,uint32_t first,
    uint16_t count,const uint16_t *ids,const float *gates,const uint8_t *activation,
    fg_owner_prefill_shared_fn shared,void *shared_context,
    fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *error){
    bench_prefill *context=opaque;
    for(uint32_t t=0;t<count;t++)for(uint32_t k=0;k<FG_TOP_K;k++)
        context->pairs[t*FG_TOP_K+k]=(fg_prefill_pair){
            .token_slot=(uint16_t)t,.routing_slot=(uint16_t)k,
            .expert_id=ids[t*FG_TOP_K+k],.gate=gates[t*FG_TOP_K+k]};
    fg_prefill_work work={.layer=(uint8_t)layer,.source_rank=1u,.destination_rank=1u,
        .first_position=first,.token_count=count,.pair_count=(uint16_t)(count*FG_TOP_K),
        .pairs=context->pairs,.activations_q8k=(uint8_t *)activation};
    fg_status status=shared(shared_context,error);
    if(status==FG_OK)status=fg_expert_prefill(context->expert,&work,&results[0],
        context->result_pairs,FG_MTP_MAX_VERIFY*FG_TOP_K,context->outputs,
        FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE,error);
    if(status==FG_OK)*result_count=1u;
    return status;
}

static bool same_state(const fg_owner_session_checkpoint *a,
                       const fg_owner_session_checkpoint *b){
    if(a->gdn_count!=b->gdn_count||a->ple_values!=b->ple_values||
       memcmp(a->qsa_tokens,b->qsa_tokens,sizeof(a->qsa_tokens)))return false;
    for(uint32_t i=0;i<a->gdn_count;i++)
        if(a->gdn[i].values!=b->gdn[i].values||
           memcmp(a->gdn[i].data,b->gdn[i].data,a->gdn[i].values*4u))return false;
    return !a->ple_values||!memcmp(a->ple_data,b->ple_data,a->ple_values*4u);
}

static void reset_static(fg_vk_context *vk,fg_error *error){
    for(uint32_t slot=0;slot<FG_VK_STATIC_SLOTS;slot++)
        check(fg_vk_static_reset(vk,slot,error),error);
}

static void serial_prefix(fg_owner_executor *owner,fg_expert_executor *expert,
    fg_vk_context *vk,uint32_t first,uint32_t count,const float *hidden,
    float *last,fg_error *error){
    reset_static(vk,error);
    fg_vk_tensor *input=fg_owner_session_input(owner,0u),*output=NULL;
    for(uint32_t t=0;t<count;t++){
        uint32_t position[3]={first+t,first+t,first+t};
        check(fg_vk_tensor_write(input,0,hidden+(uint64_t)t*FG_HYPER_WIDTH,
            FG_HYPER_WIDTH*4u,error),error);
        check(fg_owner_decode_block_chained(owner,6u,11u,first+t,position,input,NULL,
            inline_expert,expert,&output,error),error);
    }
    check(fg_vk_tensor_read(output,0,last,FG_HYPER_WIDTH*4u,error),error);
}

static void test_rollback(fg_owner_executor *owner,fg_expert_executor *expert,
    fg_vk_context *vk,bench_prefill *prefill,uint32_t first,fg_error *error){
    fg_owner_session_checkpoint initial={0};
    check(fg_owner_session_snapshot(owner,0u,&initial,error),error);
    float hidden[FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH];
    for(uint32_t i=0;i<FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH;i++)
        hidden[i]=sinf(i*.17f)*.13f;
    for(uint32_t keep=1u;keep<=FG_MTP_MAX_VERIFY;keep++){
        check(fg_owner_session_rollback(owner,&initial,error),error);
        float reference[FG_HYPER_WIDTH],replayed[FG_HYPER_WIDTH];
        serial_prefix(owner,expert,vk,first,keep,hidden,reference,error);
        fg_owner_session_checkpoint expected={0},actual={0},device={0};
        check(fg_owner_session_snapshot(owner,0u,&expected,error),error);
        check(fg_owner_session_rollback(owner,&initial,error),error);
        check(fg_owner_session_device_snapshot(owner,0u,&device,error),error);
        fg_vk_tensor *input=fg_owner_prefill_input(owner),*output=NULL;
        check(fg_vk_tensor_write(input,0,hidden,sizeof(hidden),error),error);
        uint32_t positions[FG_MTP_MAX_VERIFY*3u];
        for(uint32_t t=0;t<FG_MTP_MAX_VERIFY;t++)
            for(uint32_t axis=0;axis<3u;axis++)positions[t*3u+axis]=first+t+100u;
        check(fg_owner_verify_block(owner,6u,11u,first,positions,FG_MTP_MAX_VERIFY,
            input,NULL,batch_expert,prefill,&output,error),error);
        check(fg_owner_session_rollback(owner,&device,error),error);
        serial_prefix(owner,expert,vk,first,keep,hidden,replayed,error);
        check(fg_owner_session_snapshot(owner,0u,&actual,error),error);
        bool exact=same_state(&expected,&actual)&&
            memcmp(reference,replayed,sizeof(reference))==0;
        printf("{\"kind\":\"rollback\",\"kept_inputs\":%u,"
               "\"gdn_and_output_bit_exact\":%s,\"changed_mrope_suffix\":true}\n",
               keep,exact?"true":"false");
        if(!exact){fprintf(stderr,"rollback parity failed at prefix %u\n",keep);exit(1);}
        fg_owner_session_snapshot_release(&expected);
        fg_owner_session_snapshot_release(&actual);
        fg_owner_session_snapshot_release(&device);
    }
    check(fg_owner_session_rollback(owner,&initial,error),error);
    fg_owner_session_snapshot_release(&initial);
}

int main(int argc,char **argv){
    bool q4=false,intrusive=false,rollback_test=false;
    uint32_t warmup=256u,iterations=64u,windows=9u,batch=1u;
    const char *dump=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--q4")==0)q4=true;
        else if(strcmp(argv[i],"--profile")==0)intrusive=true;
        else if(strcmp(argv[i],"--no-clocks")==0){}
        else if(strcmp(argv[i],"--batch")==0&&i+1<argc)batch=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--rollback-test")==0)rollback_test=true;
        else if(strcmp(argv[i],"--dump")==0&&i+1<argc)dump=argv[++i];
        else if(strcmp(argv[i],"--warmup")==0&&i+1<argc)warmup=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--iterations")==0&&i+1<argc)iterations=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--windows")==0&&i+1<argc)windows=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--seed")==0&&i+1<argc)bench_seed=(uint32_t)atoi(argv[++i]);
        else {fprintf(stderr,"unknown/incomplete argument %s\n",argv[i]);return 1;}
    }
    if(!batch||batch>FG_MTP_MAX_VERIFY||!warmup||!iterations||!windows||
       (uint64_t)batch*(warmup+(uint64_t)iterations*windows)>8192u)return 1;
    fg_error error={0};fg_manifest *manifest=malloc(sizeof(*manifest));
    fg_model *model=calloc(1,sizeof(*model));
    if(!manifest||!model)return 1;
    fg_manifest_init(manifest);manifest->prefill_microbatch=128u;
    manifest->max_context=manifest->native_context=8192u;
    manifest->session.logical_context_tokens=8192u;
    model->manifest=manifest;model->rank=1u;
    model->tensor=calloc(256u,sizeof(*model->tensor));
    if(!model->tensor)return 1;
    check(fg_vk_open(&model->vk,&error),&error);
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        manifest->layer_owner[l]=(l>=6u&&l<=11u)?1u:2u;
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++)
            manifest->layer_groups[l][g]=manifest->layer_owner[l];
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)manifest->expert_rank[l][e]=manifest->layer_owner[l];
    }
    const bench_spec common[]={
        {"ffn_down_exps.weight",640,2560,7},
        {"ffn_gate_exps.weight",2560,640,12},{"ffn_up_exps.weight",2560,640,12},
        {"ffn_down_shexp.weight",640,2560,8},{"ffn_gate_shexp.weight",2560,640,8},
        {"ffn_up_shexp.weight",2560,640,8},{"ffn_gate_inp.weight",2560,512,0},
        {"ffn_gate_inp_shexp.weight",2560,1,0},
        {"hc_attn_down.weight",10240,320,8},{"hc_attn_up.weight",320,10240,8},
        {"hc_attn_norm.weight",10240,1,0},{"hc_attn_inject.weight",10240,4,0},
        {"hc_ffn_down.weight",10240,320,8},{"hc_ffn_up.weight",320,10240,8},
        {"hc_ffn_norm.weight",10240,1,0},{"hc_ffn_inject.weight",10240,4,0}
    };
    const bench_spec gdn[]={
        {"attn_gate.weight",2560,6144,8},{"attn_qkv.weight",2560,10240,8},
        {"ssm_a",48,1,0},{"ssm_alpha.weight",2560,48,0},
        {"ssm_beta.weight",2560,48,0},{"ssm_conv1d.weight",4,10240,0},
        {"ssm_dt.bias",48,1,0},{"ssm_norm.weight",128,1,0},
        {"ssm_out.weight",6144,2560,8}
    };
    const bench_spec qsa[]={
        {"attn_k.weight",2560,512,8},{"attn_k_norm.weight",256,1,0},
        {"attn_output.weight",6144,2560,8},{"attn_q.weight",2560,12288,8},
        {"attn_q_norm.weight",256,1,0},{"attn_v.weight",2560,512,8},
        {"indexer.k_norm.weight",128,1,0},{"indexer.k_proj.weight",2560,128,30},
        {"indexer.q_norm.weight",128,1,0},{"indexer.q_proj.weight",2560,512,30}
    };
    for(uint32_t l=6;l<=11;l++){
        for(size_t j=0;j<sizeof(common)/sizeof(*common);j++)add_weight(model,manifest,l,common[j],q4,&error);
        const bench_spec *spec=(l&3u)==3u?qsa:gdn;
        size_t count=(l&3u)==3u?sizeof(qsa)/sizeof(*qsa):sizeof(gdn)/sizeof(*gdn);
        for(size_t j=0;j<count;j++)add_weight(model,manifest,l,spec[j],q4,&error);
    }
    check(build_tensor_lookup(model,&error),&error);
    fg_owner_executor *owner=NULL;fg_expert_executor *expert=NULL;
    check(fg_owner_executor_create_worker(&owner,model,&error),&error);
    check(fg_expert_executor_create(&expert,model,&error),&error);
    bench_prefill prefill={.expert=expert};
    const uint8_t layers[]={7u,11u};fg_qsa_state *state=NULL;
    const char *state_path=".build/mtp-qsa.fgqs";
    unlink(state_path);
    check(fg_qsa_state_open(&state,state_path,layers,2u,8192u,true,&error),&error);
    fg_qsa_state_close(state);
    check(fg_owner_qsa_open_state(owner,state_path,8192u,0u,4096u,128u,&error),&error);
    fg_vk_tensor *input=batch==1u?fg_owner_session_input(owner,0u):
        fg_owner_prefill_input(owner),*output=NULL;
    float hidden[FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH];
    printf("{\"kind\":\"config\",\"weights\":\"synthetic-%s\",\"first_layer\":6,"
           "\"last_layer\":11,\"resident_bytes\":%llu,\"warmup\":%u,\"iterations\":%u,"
           "\"windows\":%u,\"seed\":%u,\"batch\":%u,\"intrusive\":%s,\"device\":\"%s\"}\n",
           q4?"q4-compact":"q8-cooked",(unsigned long long)model->weight_bytes,
           warmup,iterations,windows,bench_seed,batch,
           intrusive?"true":"false",
           fg_vk_device_name(model->vk));
    uint64_t hash=UINT64_C(1469598103934665603);
    FILE *raw=dump?fopen(dump,"wb"):NULL;
    if(dump&&!raw)return 1;
    for(uint32_t t=0;t<warmup+iterations*windows;t++){
        uint32_t position[FG_MTP_MAX_VERIFY*3u];fg_vk_profile profile={0};
        for(uint32_t b=0;b<batch;b++){
            for(uint32_t axis=0;axis<3u;axis++)position[b*3u+axis]=t*batch+b;
            for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
                hidden[b*FG_HYPER_WIDTH+i]=sinf(i*0.13f+(t*batch+b)*0.019f+bench_seed*0.11f)*0.2f;
        }
        check(fg_vk_tensor_write(input,0,hidden,(uint64_t)batch*FG_HYPER_WIDTH*4u,&error),&error);
        if(intrusive)check(fg_vk_profile_begin(model->vk,&error),&error);
        double begin=now_us();
        if(batch==1u)
            check(fg_owner_decode_block_chained(owner,6u,11u,t,position,input,NULL,
                inline_expert,expert,&output,&error),&error);
        else check(fg_owner_verify_block(owner,6u,11u,t*batch,position,(uint16_t)batch,
                input,NULL,batch_expert,&prefill,&output,&error),&error);
        double elapsed=now_us()-begin;
        if(intrusive)check(fg_vk_profile_end(model->vk,&profile,&error),&error);
        if(t>=warmup){
            printf("{\"kind\":\"sample\",\"window\":%u,\"token\":%u,\"wall_us\":%.3f,",
                (t-warmup)/iterations,t,elapsed);
            printf("\"static_slots\":%u}\n",(unsigned)fg_vk_static_recorded(model->vk,0u)+
                (unsigned)fg_vk_static_recorded(model->vk,1u));
            if(intrusive&&t==warmup){
                for(uint32_t k=0;k<profile.kernel_count;k++){
                    fg_vk_profile_kernel *p=&profile.kernels[k];
                    printf("{\"kind\":\"kernel\",\"scope\":\"%s\",\"name\":\"%s\","
                           "\"count\":%llu,\"gpu_us\":%.3f}\n",p->scope,p->name,
                           (unsigned long long)p->invocations,p->gpu_ms*1000.0);
                }
                printf("{\"kind\":\"profile\",\"gpu_us\":%.3f,\"gaps_us\":%.3f,"
                       "\"submissions\":%llu,\"dispatches\":%llu}\n",profile.gpu_ms*1000,
                       profile.gap_ms*1000,(unsigned long long)profile.submissions,
                       (unsigned long long)profile.dispatches);
            }
            float values[FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH];
            uint64_t value_bytes=(uint64_t)batch*FG_HYPER_WIDTH*4u;
            check(fg_vk_tensor_read(output,0,values,value_bytes,&error),&error);
            for(uint32_t i=0;i<batch*FG_HYPER_WIDTH;i++){
                if(!isfinite(values[i])){fprintf(stderr,"nonfinite token %u value %u\n",t,i);return 1;}
                uint32_t bits;memcpy(&bits,&values[i],4u);hash^=bits;hash*=UINT64_C(1099511628211);
            }
            if(raw&&fwrite(values,1,value_bytes,raw)!=value_bytes)return 1;
        }
    }
    if(raw&&fclose(raw))return 1;
    if(rollback_test)test_rollback(owner,expert,model->vk,&prefill,
        batch*(warmup+iterations*windows),&error);
    fg_owner_session_checkpoint snapshot={0};
    check(fg_owner_session_snapshot(owner,0u,&snapshot,&error),&error);
    uint64_t state_hash=UINT64_C(1469598103934665603);
    for(uint32_t s=0;s<snapshot.gdn_count;s++)
        for(uint64_t i=0;i<snapshot.gdn[s].values;i++){
            uint32_t bits;memcpy(&bits,snapshot.gdn[s].data+i,4u);
            state_hash^=bits;state_hash*=UINT64_C(1099511628211);
        }
    printf("{\"kind\":\"parity\",\"output_hash\":\"%016llx\",\"gdn_state_hash\":\"%016llx\"}\n",
        (unsigned long long)hash,(unsigned long long)state_hash);
    fg_owner_session_snapshot_release(&snapshot);
    fg_expert_executor_destroy(expert);fg_owner_executor_destroy(owner);
    fg_model_close(model);free(manifest);unlink(state_path);
    return 0;
}
