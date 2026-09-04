#include "fg_bc250_roofline.h"
#include "fg_q38_schema.h"
#include "fg_quant.h"
#include "fg_vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct bench_result {double gpu_ms;double wall_ms;} bench_result;

static double elapsed_ms(struct timespec begin,struct timespec end){
    return ((double)(end.tv_sec-begin.tv_sec)*1e3)+
           ((double)(end.tv_nsec-begin.tv_nsec)*1e-6);
}

static int emit(const char *device,const char *benchmark,const char *name,
                const char *shape,const char *access,uint32_t batch,uint32_t tokens,
                uint32_t iterations,uint64_t read_bytes,uint64_t write_bytes,
                uint64_t operations,bench_result result){
    fg_bc250_roofline_record record={
        .benchmark=benchmark,.name=name,.shape=shape,.access_pattern=access,
        .batch=batch,.tokens=tokens,.iterations=iterations,
        .bytes_read=read_bytes,.bytes_written=write_bytes,.operations=operations,
        .gpu_ms=result.gpu_ms,.wall_ms=result.wall_ms,.derived_gbps=NAN,
        .derived_flops=NAN,.device=device};
    if(result.gpu_ms>0.0&&read_bytes!=UINT64_MAX&&write_bytes!=UINT64_MAX){
        record.derived_gbps=(double)(read_bytes+write_bytes)/
                            (result.gpu_ms*1e6);
    }
    if(result.gpu_ms>0.0&&operations!=UINT64_MAX&&
       strcmp(benchmark,"q8_dense_shapes")==0)
        record.derived_flops=(double)operations/(result.gpu_ms*1e-3);
    char line[4096];int bytes=fg_bc250_roofline_record_format(line,sizeof(line),&record);
    return bytes>0&&fwrite(line,1u,(size_t)bytes,stdout)==(size_t)bytes;
}

static int run_linear(fg_vk_context *vk,uint64_t source_bytes,bool copy,
                      uint32_t iterations,bench_result *result,fg_error *err){
    uint32_t words=(uint32_t)(source_bytes/4u),groups=(words+255u)/256u;
    uint64_t output_bytes=(uint64_t)(copy?words:groups)*4u;
    fg_vk_tensor *source=NULL,*destination=NULL;
    fg_status status=fg_vk_tensor_create(vk,source_bytes,&source,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,output_bytes,&destination,err);
    if(status==FG_OK){
        uint32_t *values=fg_vk_tensor_map(source);
        if(!values){status=FG_ERR_UNAVAILABLE;fg_error_set(err,status,"map roofline source");}
        else for(uint32_t i=0;i<words;i++)values[i]=0x9e3779b9u+i;
    }
    for(uint32_t i=0;status==FG_OK&&i<3u;i++)
        status=fg_vk_bench_linear(vk,destination,source,words,copy,err);
    fg_vk_profile profile={0};struct timespec begin,end;
    if(status==FG_OK)status=fg_vk_profile_begin(vk,err);
    if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"bc250_roofline",err);
    clock_gettime(CLOCK_MONOTONIC,&begin);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    for(uint32_t i=0;status==FG_OK&&i<iterations;i++)
        status=fg_vk_bench_linear(vk,destination,source,words,copy,err);
    if(status==FG_OK)status=fg_vk_end(vk,err);
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(status==FG_OK)status=fg_vk_profile_end(vk,&profile,err);
    if(status==FG_OK){result->gpu_ms=profile.kernel_ms/(double)iterations;
        result->wall_ms=elapsed_ms(begin,end)/(double)iterations;}
    fg_vk_tensor_destroy(destination);fg_vk_tensor_destroy(source);
    return status==FG_OK;
}

static int run_minimal_batch(fg_vk_context *vk,fg_vk_tensor *source,
                             fg_vk_tensor *destination,uint32_t batch,
                             uint32_t iterations,bench_result *result,
                             fg_error *err){
    fg_vk_profile profile={0};struct timespec begin,end;fg_status status=FG_OK;
    for(uint32_t i=0;status==FG_OK&&i<3u;i++){
        status=fg_vk_begin(vk,err);
        for(uint32_t j=0;status==FG_OK&&j<batch;j++)
            status=fg_vk_bench_linear(vk,destination,source,1u,false,err);
        if(status==FG_OK)status=fg_vk_end(vk,err);
    }
    if(status==FG_OK)status=fg_vk_profile_begin(vk,err);
    if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"bc250_roofline",err);
    clock_gettime(CLOCK_MONOTONIC,&begin);
    for(uint32_t i=0;status==FG_OK&&i<iterations;i++){
        status=fg_vk_begin(vk,err);
        for(uint32_t j=0;status==FG_OK&&j<batch;j++)
            status=fg_vk_bench_linear(vk,destination,source,1u,false,err);
        if(status==FG_OK)status=fg_vk_end(vk,err);
    }
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(status==FG_OK)status=fg_vk_profile_end(vk,&profile,err);
    if(status==FG_OK){result->gpu_ms=profile.kernel_ms/(double)iterations;
        result->wall_ms=elapsed_ms(begin,end)/(double)iterations;}
    return status==FG_OK;
}

static int run_empty(fg_vk_context *vk,uint32_t iterations,bench_result *result,fg_error *err){
    fg_vk_profile profile={0};struct timespec begin,end;fg_status status=FG_OK;
    status=fg_vk_profile_begin(vk,err);
    if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"bc250_roofline",err);
    clock_gettime(CLOCK_MONOTONIC,&begin);
    for(uint32_t i=0;status==FG_OK&&i<iterations;i++){
        status=fg_vk_begin(vk,err);
        if(status==FG_OK)status=fg_vk_end(vk,err);
    }
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(status==FG_OK)status=fg_vk_profile_end(vk,&profile,err);
    if(status==FG_OK){result->gpu_ms=profile.gpu_ms/(double)iterations;
        result->wall_ms=elapsed_ms(begin,end)/(double)iterations;}
    return status==FG_OK;
}

static int run_dense(fg_vk_context *vk,uint32_t input_width,uint32_t output_width,
                     uint32_t tokens,bool cooked,uint32_t iterations,
                     bench_result *result,fg_error *err){
    uint64_t weight_bytes=cooked?fg_q8_0_cooked_matrix_bytes(input_width,output_width):0u;
    if(!cooked&&!fg_bc250_q8_0_weight_bytes(input_width,output_width,&weight_bytes)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8 roofline shape");return 0;
    }
    uint64_t input_bytes=(uint64_t)input_width*tokens*4u;
    uint64_t output_bytes=(uint64_t)output_width*tokens*4u;
    fg_vk_tensor *weights=NULL,*input=NULL,*output=NULL;fg_status status=FG_OK;
    status=fg_vk_tensor_create(vk,weight_bytes,&weights,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,input_bytes,&input,err);
    if(status==FG_OK)status=fg_vk_tensor_create(vk,output_bytes,&output,err);
    if(status==FG_OK){memset(fg_vk_tensor_map(weights),0x42,(size_t)weight_bytes);
        memset(fg_vk_tensor_map(input),0x3c,(size_t)input_bytes);
        if(cooked)fg_vk_tensor_set_format(weights,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);}
    for(uint32_t i=0;status==FG_OK&&i<3u;i++){
        status=cooked&&tokens>1u?
            fg_vk_dense_q8_0_cooked_prefill(vk,output,weights,input,input_width,
                                             output_width,tokens,1.0f,err):
            fg_vk_dense_q8_0_f32(vk,output,weights,input,input_width,output_width,
                                  tokens,1.0f,err);
    }
    fg_vk_profile profile={0};struct timespec begin,end;
    if(status==FG_OK)status=fg_vk_profile_begin(vk,err);
    if(status==FG_OK)status=fg_vk_profile_set_scope(vk,"bc250_roofline",err);
    clock_gettime(CLOCK_MONOTONIC,&begin);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    for(uint32_t i=0;status==FG_OK&&i<iterations;i++){
        status=cooked&&tokens>1u?
            fg_vk_dense_q8_0_cooked_prefill(vk,output,weights,input,input_width,
                                             output_width,tokens,1.0f,err):
            fg_vk_dense_q8_0_f32(vk,output,weights,input,input_width,output_width,
                                  tokens,1.0f,err);
    }
    if(status==FG_OK)status=fg_vk_end(vk,err);
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(status==FG_OK)status=fg_vk_profile_end(vk,&profile,err);
    if(status==FG_OK){result->gpu_ms=profile.kernel_ms/(double)iterations;
        result->wall_ms=elapsed_ms(begin,end)/(double)iterations;}
    fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(weights);
    return status==FG_OK;
}

static int run_linear_cases(fg_vk_context *vk,const char *device,fg_error *err){
    const uint64_t sizes[]={UINT64_C(64)<<10,UINT64_C(1)<<20,UINT64_C(16)<<20,
                            UINT64_C(256)<<20,UINT64_C(1)<<30};
    for(uint32_t i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++)for(uint32_t mode=0;mode<2u;mode++){
        bench_result result={0};uint32_t iterations=sizes[i]>=((uint64_t)256<<20)?3u:10u;
        if(!run_linear(vk,sizes[i],mode!=0u,iterations,&result,err))return 0;
        char name[64];snprintf(name,sizeof(name),"%s_%lluMiB",mode?"copy":"sequential_read",
                                (unsigned long long)(sizes[i]>>20));
        uint64_t read_bytes=0,write_bytes=0,operations=0;
        if(!fg_bc250_linear_accounting(sizes[i],mode!=0u,&read_bytes,&write_bytes,&operations))return 0;
        if(!emit(device,"linear_traffic",name,mode?"read_write":"read","coalesced",
                 1u,1u,iterations,read_bytes,write_bytes,operations,result))return 0;
    }
    return 1;
}

static int run_dense_cases(fg_vk_context *vk,const char *device,fg_error *err){
    const struct{uint32_t input,output;const char *name;} shapes[]={
        {FG_HIDDEN_SIZE,FG_Q38_HYPER_RANK,"hc_down"},
        {FG_Q38_HYPER_RANK,FG_Q38_HYPER_WIDTH,"hc_up"},
        {FG_HIDDEN_SIZE,FG_Q38_EXPERT_WIDTH,"shared_gate"},
        {FG_Q38_EXPERT_WIDTH,FG_HIDDEN_SIZE,"shared_down"},
        {FG_HIDDEN_SIZE,FG_Q38_ATTN_QUERY_WIDTH,"qsa_query"},
        {FG_HIDDEN_SIZE,FG_Q38_GDN_CONV_WIDTH,"gdn_projection"},
        {FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,"output_vocab"}
    };
    const uint32_t batches[]={1u,8u,16u,32u,64u,128u};
    for(uint32_t s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++)for(uint32_t cooked=0;cooked<2u;cooked++)
        for(uint32_t b=0;b<sizeof(batches)/sizeof(batches[0]);b++){
            uint32_t tokens=batches[b],iterations=(shapes[s].output==FG_Q38_VOCAB_SIZE?1u:3u);
            bench_result result={0};
            if(!run_dense(vk,shapes[s].input,shapes[s].output,tokens,cooked!=0u,
                          iterations,&result,err))return 0;
            uint64_t read_bytes=0,write_bytes=0,operations=0;
            if(!fg_bc250_q8_0_logical_traffic(shapes[s].input,shapes[s].output,
                                              tokens,cooked!=0u,&read_bytes,
                                              &write_bytes,&operations))return 0;
            char name[96];snprintf(name,sizeof(name),"q8_%s_%s",cooked?"cooked":"dense",shapes[s].name);
            if(!emit(device,"q8_dense_shapes",name,shapes[s].name,
                     cooked?"q8_0_cooked_logical":"q8_0_logical",tokens,tokens,iterations,
                     read_bytes,write_bytes,operations,result))return 0;
        }
    return 1;
}

int main(void){
    fg_error error={0};fg_vk_context *vk=NULL;
    fg_status status=fg_vk_open(&vk,&error);
    if(status!=FG_OK){fprintf(stderr,"bc250 roofline unavailable: %s\n",error.message);return 77;}
    const char *device=fg_vk_device_name(vk);
    int ok=1;bench_result result={0};
    if(ok&&!run_empty(vk,20u,&result,&error))ok=0;
    if(ok&&!emit(device,"command_overhead","empty_submission","empty","none",1u,0u,
                 20u,UINT64_MAX,UINT64_MAX,UINT64_MAX,result))ok=0;
    if(ok){
        fg_vk_tensor *source=NULL,*destination=NULL;
        status=fg_vk_tensor_create(vk,4u,&source,&error);
        if(status==FG_OK)status=fg_vk_tensor_create(vk,4u,&destination,&error);
        if(status==FG_OK){memset(fg_vk_tensor_map(source),0x5a,4u);memset(fg_vk_tensor_map(destination),0,4u);}
        for(uint32_t batch=1u;status==FG_OK&&batch<=64u;batch*=2u){
            if(run_minimal_batch(vk,source,destination,batch,10u,&result,&error)){
                char name[64];snprintf(name,sizeof(name),"minimal_dispatch_batch_%u",batch);
                if(!emit(device,"command_overhead",name,"one_word","coalesced",batch,1u,10u,
                         (uint64_t)batch*4u,(uint64_t)batch*4u,batch,result))status=FG_ERR_IO;
            }else{status=FG_ERR_IO;break;}
        }
        fg_vk_tensor_destroy(destination);fg_vk_tensor_destroy(source);
        if(status!=FG_OK)ok=0;
    }
    if(ok)ok=run_linear_cases(vk,device,&error);
    if(ok)ok=run_dense_cases(vk,device,&error);
    if(!ok)fprintf(stderr,"bc250 roofline failed: %s\n",error.message);
    fg_vk_close(vk);return ok?0:1;
}
