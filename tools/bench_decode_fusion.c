#include "../src/model.c"
#include "fg_expert.h"
#include "fg_owner.h"
#include "fg_qsa_state.h"
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

static void print_dispatches(fg_vk_context *vk,uint32_t layer,fg_error *error){
    fg_vk_layer_sample *sample=malloc(sizeof(*sample));
    if(!sample)exit(1);
    check(fg_vk_layer_clock_read(vk,layer,sample,error),error);
    for(uint32_t d=0;d<sample->dispatch_count;d++){
        fg_vk_layer_dispatch *r=&sample->dispatches[d];
        printf("{\"kind\":\"dispatch\",\"layer\":%u,\"index\":%u,\"name\":\"%s\","
               "\"barrier\":%s,\"grid\":[%u,%u,%u],\"push\":[",
               layer,d,r->name,r->barrier?"true":"false",r->grid[0],r->grid[1],r->grid[2]);
        for(uint32_t p=0;p<r->push_words;p++)printf("%s%u",p?",":"",r->push[p]);
        printf("],\"binding_bytes\":[");
        for(uint32_t b=0;b<r->bindings;b++)printf("%s%llu",b?",":"",(unsigned long long)r->bytes[b]);
        puts("]}");
    }
    free(sample);
}

int main(int argc,char **argv){
    bool q4=false,intrusive=false,clocks=true;
    uint32_t warmup=256u,iterations=64u,windows=9u;
    const char *dump=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--q4")==0)q4=true;
        else if(strcmp(argv[i],"--profile")==0)intrusive=true;
        else if(strcmp(argv[i],"--no-clocks")==0)clocks=false;
        else if(strcmp(argv[i],"--dump")==0&&i+1<argc)dump=argv[++i];
        else if(strcmp(argv[i],"--warmup")==0&&i+1<argc)warmup=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--iterations")==0&&i+1<argc)iterations=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--windows")==0&&i+1<argc)windows=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--seed")==0&&i+1<argc)bench_seed=(uint32_t)atoi(argv[++i]);
        else {fprintf(stderr,"unknown/incomplete argument %s\n",argv[i]);return 1;}
    }
    if(!warmup||!iterations||!windows||(uint64_t)warmup+(uint64_t)iterations*windows>8192u)return 1;
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
    printf("{\"kind\":\"roofline\",\"mask\":\"%s\",\"dense_variant\":\"%s\","
           "\"reported_cus\":%u,\"subgroup_size\":%u}\n",
           getenv("FG_ROOFLINE")?getenv("FG_ROOFLINE"):"0",
           getenv("FG_ROOF_DENSE")?getenv("FG_ROOF_DENSE"):"7",
           fg_vk_active_compute_units(model->vk),fg_vk_subgroup_size(model->vk));
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        manifest->layer_owner[l]=(l>=6u&&l<=11u)?1u:2u;
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
    const uint8_t layers[]={7u,11u};fg_qsa_state *state=NULL;
    const char *state_path=".build/fusion-qsa.fgqs";
    unlink(state_path);
    check(fg_qsa_state_open(&state,state_path,layers,2u,8192u,true,&error),&error);
    fg_qsa_state_close(state);
    check(fg_owner_qsa_open_state(owner,state_path,8192u,0u,4096u,128u,&error),&error);
    fg_vk_tensor *input=fg_owner_session_input(owner,0u),*output=NULL;
    float hidden[FG_HYPER_WIDTH];
    for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)hidden[i]=sinf((float)i*0.13f)*0.2f;
    check(fg_vk_tensor_write(input,0,hidden,sizeof(hidden),&error),&error);
    if(clocks&&!intrusive)check(fg_vk_layer_clock_enable(model->vk,&error),&error);
    printf("{\"kind\":\"config\",\"weights\":\"synthetic-%s\",\"first_layer\":6,"
           "\"last_layer\":11,\"resident_bytes\":%llu,\"warmup\":%u,\"iterations\":%u,"
           "\"windows\":%u,\"seed\":%u,\"fusion_mask\":%u,\"clocks\":%s,\"intrusive\":%s,\"device\":\"%s\"}\n",
           q4?"q4-compact":"q8-cooked",(unsigned long long)model->weight_bytes,
           warmup,iterations,windows,bench_seed,fg_vk_decode_fusion_mask(),
           clocks?"true":"false",intrusive?"true":"false",
           fg_vk_device_name(model->vk));
    uint64_t hash=UINT64_C(1469598103934665603);
    FILE *raw=dump?fopen(dump,"wb"):NULL;
    if(dump&&!raw)return 1;
    for(uint32_t t=0;t<warmup+iterations*windows;t++){
        uint32_t position[3]={t,t,t};fg_vk_profile profile={0};
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
            hidden[i]=sinf(i*0.13f+t*0.019f+bench_seed*0.11f)*0.2f;
        check(fg_vk_tensor_write(input,0,hidden,sizeof(hidden),&error),&error);
        if(intrusive)check(fg_vk_profile_begin(model->vk,&error),&error);
        double begin=now_us();
        check(fg_owner_decode_block_chained(owner,6u,11u,t,position,input,NULL,
            inline_expert,expert,&output,&error),&error);
        double elapsed=now_us()-begin;
        if(intrusive)check(fg_vk_profile_end(model->vk,&profile,&error),&error);
        if(t==warmup&&clocks&&!intrusive)
            for(uint32_t l=6u;l<=11u;l++)print_dispatches(model->vk,l,&error);
        if(t>=warmup){
            printf("{\"kind\":\"sample\",\"window\":%u,\"token\":%u,\"wall_us\":%.3f,\"layers_us\":[",
                (t-warmup)/iterations,t,elapsed);
            for(uint32_t l=6u;l<=11u;l++){
                fg_vk_layer_sample sample={0};
                if(clocks&&!intrusive)check(fg_vk_layer_clock_read(model->vk,l,&sample,&error),&error);
                printf("%s%.3f",l>6u?",":"",sample.gpu_us);
            }
            printf("],\"static_slots\":%u}\n",(unsigned)fg_vk_static_recorded(model->vk,0u)+
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
            float values[FG_HYPER_WIDTH];
            check(fg_vk_tensor_read(output,0,values,sizeof(values),&error),&error);
            for(uint32_t i=0;i<FG_HYPER_WIDTH;i++){
                if(!isfinite(values[i])){fprintf(stderr,"nonfinite token %u value %u\n",t,i);return 1;}
                uint32_t bits;memcpy(&bits,&values[i],4u);hash^=bits;hash*=UINT64_C(1099511628211);
            }
            if(raw&&fwrite(values,1,sizeof(values),raw)!=sizeof(values))return 1;
        }
    }
    if(raw&&fclose(raw))return 1;
    fg_owner_session_checkpoint snapshot={0};
    check(fg_owner_session_snapshot(owner,0u,&snapshot,&error),&error);
    FILE *state_raw=NULL;
    if(dump){
        char path[4096];
        int length=snprintf(path,sizeof(path),"%s.gdn",dump);
        if(length<0||(size_t)length>=sizeof(path)||(state_raw=fopen(path,"wb"))==NULL)return 1;
    }
    uint64_t state_hash=UINT64_C(1469598103934665603);
    for(uint32_t s=0;s<snapshot.gdn_count;s++){
        if(state_raw&&fwrite(snapshot.gdn[s].data,sizeof(float),snapshot.gdn[s].values,
                           state_raw)!=snapshot.gdn[s].values)return 1;
        for(uint64_t i=0;i<snapshot.gdn[s].values;i++){
            uint32_t bits;memcpy(&bits,snapshot.gdn[s].data+i,4u);
            state_hash^=bits;state_hash*=UINT64_C(1099511628211);
        }
    }
    if(state_raw&&fclose(state_raw))return 1;
    printf("{\"kind\":\"parity\",\"output_hash\":\"%016llx\",\"gdn_state_hash\":\"%016llx\"}\n",
        (unsigned long long)hash,(unsigned long long)state_hash);
    fg_owner_session_snapshot_release(&snapshot);
    fg_expert_executor_destroy(expert);fg_owner_executor_destroy(owner);
    fg_model_close(model);free(manifest);unlink(state_path);
    return 0;
}
