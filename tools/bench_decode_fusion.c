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
static uint32_t bench_dense_type=UINT32_MAX;

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
    uint32_t block=type==12u?144u:type==13u?176u:type==20u?18u:type==7u?24u:34u;
    if(bytes%block){fprintf(stderr,"invalid synthetic block size for type %u\n",type);exit(1);}
    for(uint64_t off=0;off<bytes;off+=block){
        for(uint32_t j=0;j<block;j++)p[off+j]=(uint8_t)random_word(&state);
        uint16_t d=fg_f32_to_f16((type==12u||type==13u)?0.0001f:0.0005f);
        memcpy(p+off,&d,2);
        if(type==12u||type==13u||type==7u){
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
    const char *hc=getenv("FG_ROOF_HC");
    uint32_t hc_variant=hc?(uint32_t)strtoul(hc,NULL,0)&3u:0u;
    bool expand=type==20u&&hc_variant>=1u&&hc_variant<=2u&&
        strstr(spec.name,"hc_")&&strstr(spec.name,"_up.weight");
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
    }else if(type==8u||expand){
        layout=FG_TENSOR_LAYOUT_Q8_0_COOKED;format=FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
        cooked_bytes=fg_q8_0_cooked_matrix_bytes(k,m);
    }else if(type==12u||type==20u){
        bool compact=fg_dense_prefer_compact(k,m,type);
        const char *variant=getenv("FG_ROOF_DENSE"),*mask=getenv("FG_ROOFLINE");
        if(type==12u&&m>=2560u&&variant&&strtoul(variant,NULL,0)==21u&&mask&&
           ((uint32_t)strtoul(mask,NULL,0)&1u)!=0u)compact=false;
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
    record.ggml_type=expand?8u:type;record.layout=layout;record.dims=expert?3u:2u;
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
    if(expand)ok=fg_expand_iq4_q8_cooked(raw,cooked,cooked_bytes,k,m,false);
    else if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED)ok=fg_cook_q8_0_rows(raw,cooked,cooked_bytes,k,m);
    else if(layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED)ok=fg_cook_k_quant_rows(raw,cooked,cooked_bytes,k,m,type);
    else if(layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED)ok=fg_cook_q5_1_rows(raw,cooked,cooked_bytes,k,m);
    else if(layout==FG_TENSOR_LAYOUT_DENSE_COMPACT)ok=fg_cook_dense_compact(raw,cooked,cooked_bytes,k,m,type);
    else if(layout==FG_TENSOR_LAYOUT_DENSE_SOA)ok=fg_cook_dense_soa(raw,cooked,cooked_bytes,k,m,type);
    else memcpy(cooked,raw,raw_bytes);
    if(!ok){fprintf(stderr,"cooking %s failed\n",record.name);exit(1);}
    uint8_t *mapped=fg_vk_tensor_map(model->tensor[index]);
    for(uint32_t e=0;e<count;e++){
        uint8_t *slab=mapped+(uint64_t)e*cooked_bytes;
        memcpy(slab,cooked,cooked_bytes);
        if(expert){
            uint16_t d=0u;memcpy(&d,slab,2u);
            d=fg_f32_to_f16(fg_f16_to_f32(d)*(1.0f+e/512.0f));
            memcpy(slab,&d,2u);
        }
    }
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

static fg_status small_expert(void *opaque,uint32_t layer,const fg_vk_tensor *activation,
    const fg_vk_tensor *logits,uint32_t count,fg_vk_tensor **output,fg_error *error){
    return fg_expert_decode_small_n(opaque,layer,activation,logits,count,output,error);
}
static fg_status fail_small_expert(void *opaque,uint32_t layer,const fg_vk_tensor *activation,
    const fg_vk_tensor *logits,uint32_t count,fg_vk_tensor **output,fg_error *error){
    if(layer==7u){fg_error_set(error,FG_ERR_UNAVAILABLE,"injected expert failure");return FG_ERR_UNAVAILABLE;}
    return small_expert(opaque,layer,activation,logits,count,output,error);
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
    uint32_t first,uint32_t count,const float *hidden,
    float *last,fg_error *error){
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
    bench_prefill *prefill,uint32_t first,bool small,fg_error *error){
    fg_owner_session_checkpoint initial={0};
    check(fg_owner_session_snapshot(owner,0u,&initial,error),error);
    float hidden[FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH];
    for(uint32_t i=0;i<FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH;i++)
        hidden[i]=sinf(i*.17f)*.13f;
    for(uint32_t keep=1u;keep<=FG_MTP_MAX_VERIFY;keep++){
        check(fg_owner_session_rollback(owner,&initial,error),error);
        float reference[FG_HYPER_WIDTH],replayed[FG_HYPER_WIDTH];
        serial_prefix(owner,expert,first,keep,hidden,reference,error);
        fg_owner_session_checkpoint expected={0},actual={0},device={0};
        check(fg_owner_session_snapshot(owner,0u,&expected,error),error);
        check(fg_owner_session_rollback(owner,&initial,error),error);
        check(fg_owner_session_device_snapshot(owner,0u,&device,error),error);
        fg_vk_tensor *input=fg_owner_prefill_input(owner),*output=NULL;
        check(fg_vk_tensor_write(input,0,hidden,sizeof(hidden),error),error);
        uint32_t positions[FG_MTP_MAX_VERIFY*3u];
        for(uint32_t t=0;t<FG_MTP_MAX_VERIFY;t++)
            for(uint32_t axis=0;axis<3u;axis++)positions[t*3u+axis]=first+t+100u;
        if(small)check(fg_owner_decode_small_n(owner,6u,11u,first,positions,
            FG_MTP_MAX_VERIFY,input,NULL,small_expert,expert,&output,error),error);
        else check(fg_owner_verify_block(owner,6u,11u,first,positions,FG_MTP_MAX_VERIFY,
            input,NULL,batch_expert,prefill,&output,error),error);
        check(fg_owner_session_rollback(owner,&device,error),error);
        serial_prefix(owner,expert,first,keep,hidden,replayed,error);
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

static bool close_values(const float *a,const float *b,uint64_t count,float *maximum){
    bool ok=true;
    for(uint64_t i=0;i<count;i++){
        float delta=fabsf(a[i]-b[i]);
        if(delta>*maximum)*maximum=delta;
        if(!isfinite(b[i])||delta>2e-5f+1e-3f*fabsf(a[i]))ok=false;
    }
    return ok;
}

static fg_status test_dense_dispatch(fg_vk_context *vk,fg_vk_tensor *out,
    const fg_vk_tensor *weight,const fg_vk_tensor *input,uint32_t in,
    uint32_t rows,uint32_t n,uint32_t type,fg_error *error){
    if(type==0u||type==31u)return fg_vk_dense_f32(vk,out,weight,input,in,rows,n,error);
    if(type==30u)return fg_vk_dense_bf16_f32(vk,out,weight,input,in,rows,n,error);
    return fg_vk_dense_q8_0_f32(vk,out,weight,input,in,rows,n,1.0f,error);
}

static void test_dense_small(fg_vk_context *vk,bool measure,fg_error *error){
    const uint32_t types[]={12u,13u,20u,8u,0u,30u,31u};
    for(uint32_t j=0;j<7u;j++)for(uint32_t geometry=0u;geometry<3u;geometry++)
    for(uint32_t compact=0u;compact<((types[j]==12u||types[j]==13u||types[j]==20u)?2u:1u);compact++){
        if(bench_dense_type!=UINT32_MAX&&types[j]!=bench_dense_type)continue;
        uint32_t in=geometry==1u?10240u:geometry==2u?2560u:512u,
            rows=geometry==2u?6144u:16u;
        uint32_t type=types[j];
        uint64_t raw_bytes=0u,shape[FG_GGUF_MAX_DIMS]={in,rows,1u,1u};
        uint32_t source_type=type==31u?30u:type;
        if(!fg_gguf_tensor_bytes(source_type,2u,shape,&raw_bytes))exit(1);
        uint64_t bytes=type==0u||type>=30u?raw_bytes:
            type==8u?fg_q8_0_cooked_matrix_bytes(in,rows):
            compact?fg_dense_compact_bytes(in,rows,type):fg_dense_soa_bytes(in,rows,type);
        uint8_t *raw=malloc(raw_bytes),*cooked=malloc(bytes);
        if(!raw||!cooked)exit(1);
        synthetic_raw(raw,raw_bytes,source_type,false,71u+j);
        bool plain=type==0u||type>=30u;
        if(plain)memcpy(cooked,raw,bytes);
        bool ok=plain?true:type==8u?fg_cook_q8_0_rows(raw,cooked,bytes,in,rows):
            compact?fg_cook_dense_compact(raw,cooked,bytes,in,rows,type):
                fg_cook_dense_soa(raw,cooked,bytes,in,rows,type);
        if(!ok)exit(1);
        fg_vk_tensor *w=NULL,*x=NULL,*y=NULL,*serial=NULL,*one=NULL;
        check(fg_vk_tensor_create(vk,bytes,&w,error),error);
        check(fg_vk_tensor_create(vk,4u*in*4u,&x,error),error);
        check(fg_vk_tensor_create(vk,4u*rows*4u,&y,error),error);
        check(fg_vk_tensor_create(vk,in*4u,&one,error),error);
        check(fg_vk_tensor_create(vk,rows*4u,&serial,error),error);
        fg_vk_tensor_set_format(w,type==31u?FG_VK_TENSOR_FORMAT_BF16_ROUTER:
            plain?FG_VK_TENSOR_FORMAT_DEFAULT:type==8u?FG_VK_TENSOR_FORMAT_Q8_0_COOKED:
            fg_vk_dense_saved_format(type,compact?FG_TENSOR_LAYOUT_DENSE_COMPACT:
                FG_TENSOR_LAYOUT_DENSE_SOA));
        check(fg_vk_tensor_write(w,0,cooked,bytes,error),error);
        float *inputs=malloc(4u*in*4u),*reference=malloc(4u*rows*4u),
            *actual=malloc(4u*rows*4u);
        if(!inputs||!reference||!actual)exit(1);
        for(uint32_t i=0u;i<4u*in;i++)inputs[i]=sinf(i*.031f)*.3f;
        check(fg_vk_tensor_write(x,0,inputs,4u*in*4u,error),error);
        for(uint32_t t=0u;t<4u;t++){
            check(fg_vk_tensor_write(one,0,inputs+t*in,in*4u,error),error);
            check(test_dense_dispatch(vk,serial,w,one,in,rows,1u,type,error),error);
            check(fg_vk_tensor_read(serial,0,reference+t*rows,rows*4u,error),error);
        }
        for(uint32_t n=2u;n<=4u;n++){
            fg_vk_small_n_mode(vk,true);
            check(test_dense_dispatch(vk,y,w,x,in,rows,n,type,error),error);
            fg_vk_small_n_mode(vk,false);
            check(fg_vk_tensor_read(y,0,actual,n*rows*4u,error),error);
            float maximum=0.0f;
            bool bit_exact=!memcmp(reference,actual,n*rows*4u);
            ok=close_values(reference,actual,n*rows,&maximum)&&bit_exact;
            printf("{\"kind\":\"dense_small_test\",\"type\":%u,\"in\":%u,"
                "\"rows\":%u,\"compact\":%u,\"n\":%u,"
                "\"max_abs\":%.9g,\"bit_exact\":%s,\"tolerance_ok\":%s}\n",
                type,in,rows,compact,n,maximum,bit_exact?"true":"false",ok?"true":"false");
            if(!ok)exit(1);
        }
        if(measure&&geometry==2u&&(compact||plain||type==8u)){
            double baseline=0.0;
            for(uint32_t n=1u;n<=4u;n++){
                check(fg_vk_static_reset(vk,3u,error),error);
                check(fg_vk_static_begin(vk,3u,error),error);
                fg_vk_small_n_mode(vk,n>1u);
                for(uint32_t r=0u;r<32u;r++)
                    check(test_dense_dispatch(vk,y,w,x,in,rows,n,type,error),error);
                fg_vk_small_n_mode(vk,false);
                check(fg_vk_static_end(vk,3u,error),error);
                for(uint32_t r=0u;r<32u;r++){
                    check(fg_vk_static_submit(vk,3u,error),error);
                    check(fg_vk_static_drain(vk,error),error);
                }
                double begin=now_us();
                for(uint32_t r=0u;r<16u;r++){
                    check(fg_vk_static_submit(vk,3u,error),error);
                    check(fg_vk_static_drain(vk,error),error);
                }
                double us=(now_us()-begin)/(16.0*32.0);
                if(n==1u)baseline=us;
                printf("{\"kind\":\"dense_cost\",\"type\":%u,\"in\":%u,\"rows\":%u,"
                    "\"n\":%u,\"us\":%.3f,\"ratio\":%.4f}\n",type,in,rows,n,us,us/baseline);
            }
            check(fg_vk_static_reset(vk,3u,error),error);
        }
        fg_vk_tensor_destroy(serial);fg_vk_tensor_destroy(one);fg_vk_tensor_destroy(y);
        fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);free(cooked);free(raw);
        free(inputs);free(reference);free(actual);
    }
}

static void test_small_schedule(fg_vk_context *vk,fg_error *error){
    fg_vk_tensor *selected=NULL,*tiles=NULL;
    check(fg_vk_tensor_create(vk,40u*4u,&selected,error),error);
    check(fg_vk_tensor_create(vk,40u*5u*4u,&tiles,error),error);
    for(uint32_t n=2u;n<=4u;n++)for(uint32_t pattern=0u;pattern<4u;pattern++){
        uint32_t ids[40],packed[200],seen[40]={0},pairs=n*FG_TOP_K;
        for(uint32_t p=0u;p<pairs;p++)ids[p]=pattern==0u?p%10u:
            pattern==1u?p:pattern==2u?p%3u:(p*17u+p/3u)%23u;
        check(fg_vk_tensor_write(selected,0,ids,pairs*4u,error),error);
        check(fg_vk_small_n_schedule(vk,tiles,selected,n,error),error);
        check(fg_vk_tensor_read(tiles,0,packed,pairs*5u*4u,error),error);
        bool ok=true;
        for(uint32_t p=0u;p<pairs;p++)if(packed[p*5u]!=UINT32_MAX){
            for(uint32_t j=0u;j<4u;j++){
                uint32_t pair=packed[p*5u+1u+j];
                if(pair==UINT32_MAX)continue;
                if(pair>=pairs||ids[pair]!=packed[p*5u])ok=false;
                else seen[pair]++;
            }
        }
        for(uint32_t p=0u;p<pairs;p++)if(seen[p]!=1u)ok=false;
        printf("{\"kind\":\"small_union_test\",\"n\":%u,\"pattern\":%u,\"exact\":%s}\n",
            n,pattern,ok?"true":"false");
        if(!ok)exit(1);
    }
    fg_vk_tensor_destroy(tiles);fg_vk_tensor_destroy(selected);
}

static void test_small_block(fg_owner_executor *owner,fg_expert_executor *expert,
    fg_vk_context *vk,uint32_t first,fg_error *error){
    test_small_schedule(vk,error);
    fg_owner_session_checkpoint initial={0};
    check(fg_owner_session_snapshot(owner,0u,&initial,error),error);
    float hidden[4u*FG_HYPER_WIDTH],reference[4u*FG_HYPER_WIDTH],actual[4u*FG_HYPER_WIDTH];
    uint32_t positions[12];
    for(uint32_t t=0u;t<4u;t++){
        for(uint32_t axis=0u;axis<3u;axis++)positions[t*3u+axis]=first+t;
        for(uint32_t i=0u;i<FG_HYPER_WIDTH;i++)hidden[t*FG_HYPER_WIDTH+i]=sinf(i*.17f+t*.2f)*.13f;
    }
    fg_vk_tensor *failure_input=fg_owner_prefill_input(owner),*failure_output=NULL;
    check(fg_vk_tensor_write(failure_input,0,hidden,2u*FG_HYPER_WIDTH*4u,error),error);
    fg_status failure=fg_owner_decode_small_n(owner,6u,11u,first,positions,2u,
        failure_input,NULL,fail_small_expert,expert,&failure_output,error);
    bool clean=failure==FG_ERR_UNAVAILABLE&&!fg_vk_small_n_active(vk)&&!fg_vk_batch_active(vk);
    printf("{\"kind\":\"small_error_cleanup_test\",\"clean\":%s}\n",clean?"true":"false");
    if(!clean)exit(1);
    memset(error,0,sizeof(*error));
    for(uint32_t n=2u;n<=4u;n++){
        check(fg_owner_session_rollback(owner,&initial,error),error);
        reset_static(vk,error);
        fg_vk_tensor *input=fg_owner_session_input(owner,0u),*output=NULL;
        for(uint32_t t=0u;t<n;t++){
            check(fg_vk_tensor_write(input,0,hidden+t*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,error),error);
            check(fg_owner_decode_block_chained(owner,6u,11u,first+t,positions+t*3u,
                input,NULL,inline_expert,expert,&output,error),error);
            check(fg_vk_tensor_read(output,0,reference+t*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,error),error);
        }
        fg_owner_session_checkpoint expected={0},result={0};
        check(fg_owner_session_snapshot(owner,0u,&expected,error),error);
        check(fg_owner_session_rollback(owner,&initial,error),error);
        input=fg_owner_prefill_input(owner);
        check(fg_vk_tensor_write(input,0,hidden,n*FG_HYPER_WIDTH*4u,error),error);
        check(fg_owner_decode_small_n(owner,6u,11u,first,positions,n,input,NULL,
            small_expert,expert,&output,error),error);
        check(fg_vk_tensor_read(output,0,actual,n*FG_HYPER_WIDTH*4u,error),error);
        check(fg_owner_session_snapshot(owner,0u,&result,error),error);
        float max_output=0.0f,max_state=0.0f;
        bool ok=close_values(reference,actual,n*FG_HYPER_WIDTH,&max_output);
        bool state_exact=same_state(&expected,&result);
        bool output_exact=!memcmp(reference,actual,n*FG_HYPER_WIDTH*4u);
        ok=ok&&state_exact&&output_exact;
        if(expected.gdn_count!=result.gdn_count)ok=false;
        for(uint32_t i=0u;i<expected.gdn_count&&i<result.gdn_count;i++){
            if(expected.gdn[i].values!=result.gdn[i].values)ok=false;
            else if(!close_values(expected.gdn[i].data,result.gdn[i].data,
                expected.gdn[i].values,&max_state))ok=false;
        }
        printf("{\"kind\":\"small_block_test\",\"n\":%u,\"max_abs_output\":%.9g,"
            "\"max_abs_state\":%.9g,\"state_bit_exact\":%s,\"output_bit_exact\":%s,"
            "\"tolerance_ok\":%s}\n",
            n,max_output,max_state,state_exact?"true":"false",output_exact?"true":"false",
            ok?"true":"false");
        if(!ok)exit(1);
        if(n==4u){
            check(fg_owner_session_rollback(owner,&initial,error),error);
            for(uint32_t i=3u*FG_HYPER_WIDTH;i<4u*FG_HYPER_WIDTH;i++)hidden[i]+=.7f;
            check(fg_vk_tensor_write(input,0,hidden,sizeof(hidden),error),error);
            check(fg_owner_decode_small_n(owner,6u,11u,first,positions,n,input,NULL,
                small_expert,expert,&output,error),error);
            check(fg_vk_tensor_read(output,0,reference,sizeof(reference),error),error);
            bool causal=!memcmp(actual,reference,3u*FG_HYPER_WIDTH*4u);
            printf("{\"kind\":\"small_causal_test\",\"prefix_bit_exact\":%s}\n",causal?"true":"false");
            if(!causal)exit(1);
        }
        fg_owner_session_snapshot_release(&result);fg_owner_session_snapshot_release(&expected);
    }
    check(fg_owner_session_rollback(owner,&initial,error),error);
    fg_owner_session_snapshot_release(&initial);
}

static void test_small_sessions(fg_owner_executor *owner,fg_expert_executor *expert,
    fg_vk_context *vk,fg_error *error){
    fg_owner_session_checkpoint initial[2]={{0},{0}};
    for(uint32_t slot=0;slot<2u;slot++)
        check(fg_owner_session_snapshot(owner,slot,&initial[slot],error),error);
    float hidden[2][4u*FG_HYPER_WIDTH],gold[2][4u*FG_HYPER_WIDTH],actual[4u*FG_HYPER_WIDTH];
    uint32_t positions[12];
    for(uint32_t cell=0;cell<4u;cell++){
        for(uint32_t axis=0;axis<3u;axis++)positions[cell*3u+axis]=cell;
        for(uint32_t slot=0;slot<2u;slot++)for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
            hidden[slot][cell*FG_HYPER_WIDTH+i]=sinf(i*.13f+cell*.019f+slot*.41f)*.2f;
    }
    fg_vk_tensor *input=fg_owner_prefill_input(owner),*output=NULL;
    for(uint32_t n=2u;n<=4u;n++){
        uint32_t sizes[2]={n,n==2u?4u:2u};
        for(uint32_t slot=0;slot<2u;slot++){
            check(fg_owner_set_active_session(owner,slot,error),error);
            check(fg_vk_tensor_write(input,0,hidden[slot],sizes[slot]*FG_HYPER_WIDTH*4u,error),error);
            check(fg_owner_decode_small_n(owner,6u,11u,0u,positions,sizes[slot],input,NULL,
                small_expert,expert,&output,error),error);
            check(fg_vk_tensor_read(output,0,gold[slot],sizes[slot]*FG_HYPER_WIDTH*4u,error),error);
            if(!fg_vk_static_recorded(vk,0u)){
                fprintf(stderr,"slot 1 invalidated slot 0 small-N replay\n");exit(1);
            }
        }
        check(fg_owner_session_rollback(owner,&initial[0],error),error);
        check(fg_owner_set_active_session(owner,0u,error),error);
        fg_vk_tensor *single=fg_owner_session_input(owner,0u);
        check(fg_vk_tensor_write(single,0,hidden[0],FG_HYPER_WIDTH*4u,error),error);
        check(fg_owner_decode_block_chained(owner,6u,11u,0u,positions,single,NULL,
            inline_expert,expert,&output,error),error);
        if(!fg_vk_static_recorded(vk,FG_VK_OWNER_STATIC_SLOTS/FG_OWNER_SESSION_MAX)){
            fprintf(stderr,"slot 0 N=1 invalidated slot 1 small-N replay\n");exit(1);
        }
        for(uint32_t slot=0;slot<2u;slot++){
            check(fg_owner_session_rollback(owner,&initial[slot],error),error);
            check(fg_owner_set_active_session(owner,slot,error),error);
            check(fg_vk_tensor_write(input,0,hidden[slot],sizes[slot]*FG_HYPER_WIDTH*4u,error),error);
            check(fg_owner_decode_small_n(owner,6u,11u,0u,positions,sizes[slot],input,NULL,
                small_expert,expert,&output,error),error);
            check(fg_vk_tensor_read(output,0,actual,sizes[slot]*FG_HYPER_WIDTH*4u,error),error);
            if(memcmp(actual,gold[slot],sizes[slot]*FG_HYPER_WIDTH*4u)){
                fprintf(stderr,"small-N session replay differs slot=%u n=%u\n",slot,sizes[slot]);exit(1);
            }
        }
        for(uint32_t slot=0;slot<2u;slot++)
            check(fg_owner_session_rollback(owner,&initial[slot],error),error);
    }
    for(uint32_t slot=0;slot<2u;slot++)fg_owner_session_snapshot_release(&initial[slot]);
    check(fg_owner_set_active_session(owner,0u,error),error);
    printf("{\"kind\":\"small_n_sessions\",\"slots\":2,\"shapes\":3,"
        "\"bit_exact\":true,\"other_slot_replay_retained\":true,\"n1_transition_isolated\":true}\n");
}

int main(int argc,char **argv){
    bool q4=false,intrusive=false,clocks=true,rollback_test=false,small_n=false,small_test=false,dense_bench=false,dense_only=false,session_test=false;
    uint32_t warmup=256u,iterations=64u,windows=9u,batch=1u;
    const char *dump=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--q4")==0)q4=true;
        else if(strcmp(argv[i],"--profile")==0)intrusive=true;
        else if(strcmp(argv[i],"--no-clocks")==0)clocks=false;
        else if(strcmp(argv[i],"--batch")==0&&i+1<argc)batch=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--rollback-test")==0)rollback_test=true;
        else if(strcmp(argv[i],"--small-n")==0)small_n=true;
        else if(strcmp(argv[i],"--session-test")==0)session_test=true;
        else if(strcmp(argv[i],"--small-test")==0)small_test=true;
        else if(strcmp(argv[i],"--dense-bench")==0)dense_bench=true;
        else if(strcmp(argv[i],"--dense-only")==0){dense_only=true;dense_bench=true;}
        else if(strcmp(argv[i],"--dense-type")==0&&i+1<argc){
            char *end=NULL;const char *value=argv[++i];
            unsigned long type=strtoul(value,&end,10);
            if(!*value||*end||(type!=0u&&type!=8u&&type!=12u&&type!=13u&&
               type!=20u&&type!=30u&&type!=31u)){
                fprintf(stderr,"unsupported dense type %s\n",value);return 1;
            }
            bench_dense_type=(uint32_t)type;
        }
        else if(strcmp(argv[i],"--dump")==0&&i+1<argc)dump=argv[++i];
        else if(strcmp(argv[i],"--warmup")==0&&i+1<argc)warmup=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--iterations")==0&&i+1<argc)iterations=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--windows")==0&&i+1<argc)windows=(uint32_t)atoi(argv[++i]);
        else if(strcmp(argv[i],"--seed")==0&&i+1<argc)bench_seed=(uint32_t)atoi(argv[++i]);
        else {fprintf(stderr,"unknown/incomplete argument %s\n",argv[i]);return 1;}
    }
    if(!batch||batch>FG_MTP_MAX_VERIFY||!warmup||!iterations||!windows||
       (uint64_t)batch*(warmup+(uint64_t)iterations*windows)>8192u)return 1;
    if(batch>1u)clocks=false;
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
           "\"qsa_variant\":\"%s\",\"expert_variant\":\"%s\",\"hc_variant\":\"%s\",\"recurrent_variant\":\"%s\",\"gdn_variant\":\"%s\","
           "\"reported_cus\":%u,\"subgroup_size\":%u}\n",
           getenv("FG_ROOFLINE")?getenv("FG_ROOFLINE"):"0",
           getenv("FG_ROOF_DENSE")?getenv("FG_ROOF_DENSE"):"7",
           getenv("FG_ROOF_QSA")?getenv("FG_ROOF_QSA"):"2",
           getenv("FG_ROOF_EXPERT")?getenv("FG_ROOF_EXPERT"):"0",
           getenv("FG_ROOF_HC")?getenv("FG_ROOF_HC"):"0",
           getenv("FG_ROOF_RECURRENT")?getenv("FG_ROOF_RECURRENT"):"0",
           getenv("FG_ROOF_GDN")?getenv("FG_ROOF_GDN"):"8",
           fg_vk_active_compute_units(model->vk),fg_vk_subgroup_size(model->vk));
    if(dense_only||small_test||dense_bench)
        test_dense_small(model->vk,dense_only||dense_bench,&error);
    if(dense_only){
        fg_model_close(model);free(manifest);
        return 0;
    }
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
    check(fg_owner_executor_create_worker_slots(&owner,model,session_test?2u:1u,&error),&error);
    check(fg_expert_executor_create(&expert,model,&error),&error);
    bench_prefill prefill={.expert=expert};
    const uint8_t layers[]={7u,11u};fg_qsa_state *state=NULL;
    const char *state_path=".build/mtp-qsa.fgqs";
    unlink(state_path);
    check(fg_qsa_state_open(&state,state_path,layers,2u,8192u,true,&error),&error);
    fg_qsa_state_close(state);
    check(fg_owner_qsa_open_state(owner,state_path,8192u,0u,4096u,128u,&error),&error);
    const char *second_state_path=".build/mtp-qsa-s1.fgqs";
    if(session_test){
        unlink(second_state_path);
        check(fg_qsa_state_open(&state,second_state_path,layers,2u,8192u,true,&error),&error);
        fg_qsa_state_close(state);state=NULL;
        check(fg_owner_qsa_open_state_slot(owner,1u,second_state_path,8192u,0u,4096u,128u,&error),&error);
        test_small_sessions(owner,expert,model->vk,&error);
    }
    fg_vk_tensor *input=batch==1u?fg_owner_session_input(owner,0u):
        fg_owner_prefill_input(owner),*output=NULL;
    float hidden[FG_MTP_MAX_VERIFY*FG_HYPER_WIDTH];
    if(clocks&&!intrusive)check(fg_vk_layer_clock_enable(model->vk,&error),&error);
    printf("{\"kind\":\"config\",\"weights\":\"synthetic-%s\",\"first_layer\":6,"
           "\"last_layer\":11,\"resident_bytes\":%llu,\"warmup\":%u,\"iterations\":%u,"
           "\"windows\":%u,\"seed\":%u,\"batch\":%u,\"small_n\":%s,\"fusion_mask\":%u,\"clocks\":%s,\"intrusive\":%s,\"device\":\"%s\"}\n",
           q4?"q4-compact":"q8-cooked",(unsigned long long)model->weight_bytes,
           warmup,iterations,windows,bench_seed,batch,
           small_n?"true":"false",fg_vk_decode_fusion_mask(),clocks?"true":"false",intrusive?"true":"false",
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
        else if(small_n)check(fg_owner_decode_small_n(owner,6u,11u,t*batch,position,
                batch,input,NULL,small_expert,expert,&output,&error),&error);
        else check(fg_owner_verify_block(owner,6u,11u,t*batch,position,(uint16_t)batch,
                input,NULL,batch_expert,&prefill,&output,&error),&error);
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
    if(small_n){
        uint32_t selected[4u*FG_TOP_K],unique=0u;
        check(fg_expert_selected_readback(expert,batch,selected,&error),&error);
        for(uint32_t p=0u;p<batch*FG_TOP_K;p++){
            bool first=true;
            for(uint32_t q=0u;q<p;q++)if(selected[p]==selected[q])first=false;
            if(first)unique++;
        }
        printf("{\"kind\":\"expert_union\",\"layer\":11,\"selected_pairs\":%u,"
            "\"unique_experts\":%u}\n",batch*FG_TOP_K,unique);
    }
    if(small_test)test_small_block(owner,expert,model->vk,
        batch*(warmup+iterations*windows),&error);
    if(rollback_test)test_rollback(owner,expert,&prefill,
        batch*(warmup+iterations*windows),small_n,&error);
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
    if(session_test)unlink(second_state_path);
    return 0;
}
