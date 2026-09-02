#include "fg_q38_math.h"
#include "fg_q38_schema.h"
#include "fg_ngram.h"
#include "fg_quant.h"
#include "fg_qsa.h"
#include "fg_vk.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static fg_vk_context *context;
static fg_error error;

static fg_vk_tensor *tensor(const void *data,uint64_t bytes){fg_vk_tensor *result=NULL;if(fg_vk_tensor_create(context,bytes,&result,&error)!=FG_OK)return NULL;if(data&&fg_vk_tensor_write(result,0,data,bytes,&error)!=FG_OK){fg_vk_tensor_destroy(result);return NULL;}return result;}

static void make_q5_1_row(uint8_t *row,uint32_t values,uint32_t phase){
    for(uint32_t block=0;block<values/32u;block++){
        uint8_t *p=row+(uint64_t)block*24u;p[0]=0;p[1]=0x3c;p[2]=0;p[3]=0;uint32_t high=0;
        for(uint32_t i=0;i<32u;i++){uint32_t q=(i+block+phase)&31u;if(q&16u)high|=1u<<i;uint32_t j=i&15u;if(i<16u)p[8u+j]=(uint8_t)((p[8u+j]&0xf0u)|(q&15u));else p[8u+j]=(uint8_t)((p[8u+j]&0x0fu)|((q&15u)<<4u));}
        memcpy(p+4u,&high,4u);
    }
}

static int test_q8_dense(void){
    enum{INPUT=32,OUTPUT=8};float source[INPUT*OUTPUT],x[INPUT],got[OUTPUT];uint8_t quantized[FG_Q8_0_BLOCK_BYTES*OUTPUT];
    for(uint32_t i=0;i<INPUT;i++)x[i]=cosf((float)i*0.17f);
    for(uint32_t r=0;r<OUTPUT;r++){for(uint32_t i=0;i<INPUT;i++)source[r*INPUT+i]=sinf((float)(r*INPUT+i)*0.013f);fg_quantize_q8_0(source+r*INPUT,quantized+r*FG_Q8_0_BLOCK_BYTES,INPUT);}
    fg_vk_tensor *w=tensor(quantized,sizeof(quantized)),*input=tensor(x,sizeof(x)),*output=tensor(NULL,sizeof(got));int ok=w&&input&&output&&fg_vk_dense_q8_0_f32(context,output,w,input,INPUT,OUTPUT,1,1.0f,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,sizeof(got),&error)==FG_OK;
    for(uint32_t r=0;ok&&r<OUTPUT;r++){const uint8_t *row=quantized+r*FG_Q8_0_BLOCK_BYTES;uint16_t half=(uint16_t)row[0]|(uint16_t)((uint16_t)row[1]<<8u);float ref=0.0f,d=fg_f16_to_f32(half);for(uint32_t i=0;i<INPUT;i++)ref=fmaf(d*(float)(int8_t)row[2u+i],x[i],ref);if(fabsf(got[r]-ref)>1e-4f*fmaxf(1.0f,fabsf(ref)))ok=0;}
    fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(w);return ok;
}

static int test_q8_dense_subgroup(void){
    enum{INPUT=256,OUTPUT=17,TOKENS=2};float *source=malloc((size_t)INPUT*OUTPUT*4u),input[TOKENS*INPUT],got[TOKENS*OUTPUT];uint32_t row_bytes=(INPUT/32u)*FG_Q8_0_BLOCK_BYTES;uint8_t *quantized=malloc((size_t)OUTPUT*row_bytes);if(!source||!quantized){free(quantized);free(source);return 0;}for(uint32_t token=0;token<TOKENS;token++)for(uint32_t i=0;i<INPUT;i++)input[token*INPUT+i]=cosf((float)(token+1u)*(float)(i+3u)*0.013f);for(uint32_t row=0;row<OUTPUT;row++){for(uint32_t i=0;i<INPUT;i++)source[row*INPUT+i]=sinf((float)(row*INPUT+i)*0.009f);fg_quantize_q8_0(source+row*INPUT,quantized+(uint64_t)row*row_bytes,INPUT);}fg_vk_tensor *w=tensor(quantized,(uint64_t)OUTPUT*row_bytes),*x=tensor(input,sizeof(input)),*y=tensor(NULL,sizeof(got));int ok=w&&x&&y&&fg_vk_dense_q8_0_subgroup(context,y,w,x,INPUT,OUTPUT,TOKENS,0.75f,&error)==FG_OK&&fg_vk_tensor_read(y,0,got,sizeof(got),&error)==FG_OK;for(uint32_t token=0;ok&&token<TOKENS;token++)for(uint32_t row=0;row<OUTPUT;row++){float ref=0.0f;const uint8_t *blocks=quantized+(uint64_t)row*row_bytes;for(uint32_t block=0;block<INPUT/32u;block++){const uint8_t *p=blocks+block*FG_Q8_0_BLOCK_BYTES;float delta=fg_f16_to_f32((uint16_t)p[0]|(uint16_t)((uint16_t)p[1]<<8u));for(uint32_t i=0;i<32u;i++)ref=fmaf(delta*(float)(int8_t)p[2u+i],input[token*INPUT+block*32u+i],ref);}ref*=0.75f;if(fabsf(got[token*OUTPUT+row]-ref)>2e-4f*fmaxf(1.0f,fabsf(ref))){fprintf(stderr,"subgroup Q8 token %u row %u GPU=%g CPU=%g\n",token,row,got[token*OUTPUT+row],ref);ok=0;break;}}fg_vk_tensor_destroy(y);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);free(quantized);free(source);return ok;
}

static int test_q8_dense_cooked(void){
    enum{INPUT=320,OUTPUT=17,TOKENS=2};uint32_t source_row=(INPUT/32u)*FG_Q8_0_BLOCK_BYTES;uint64_t source_bytes=(uint64_t)OUTPUT*source_row,cooked_bytes=fg_q8_0_cooked_matrix_bytes(INPUT,OUTPUT);float *source=malloc((size_t)INPUT*OUTPUT*4u),input[TOKENS*INPUT],got[TOKENS*OUTPUT];uint8_t *quantized=malloc((size_t)source_bytes),*cooked=malloc((size_t)cooked_bytes);if(!source||!quantized||!cooked){free(cooked);free(quantized);free(source);return 0;}for(uint32_t token=0;token<TOKENS;token++)for(uint32_t i=0;i<INPUT;i++)input[token*INPUT+i]=cosf((float)(token+1u)*(float)(i+5u)*0.011f);for(uint32_t row=0;row<OUTPUT;row++){for(uint32_t i=0;i<INPUT;i++)source[row*INPUT+i]=sinf((float)(row*INPUT+i)*0.007f);fg_quantize_q8_0(source+row*INPUT,quantized+(uint64_t)row*source_row,INPUT);}int ok=fg_cook_q8_0_rows(quantized,cooked,cooked_bytes,INPUT,OUTPUT);fg_vk_tensor *w=ok?tensor(cooked,cooked_bytes):NULL,*x=ok?tensor(input,sizeof(input)):NULL,*y=ok?tensor(NULL,sizeof(got)):NULL;if(w)fg_vk_tensor_set_format(w,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);ok=ok&&w&&x&&y&&fg_vk_tensor_get_format(w)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED&&fg_vk_dense_q8_0_f32(context,y,w,x,INPUT,OUTPUT,TOKENS,0.75f,&error)==FG_OK&&fg_vk_tensor_read(y,0,got,sizeof(got),&error)==FG_OK&&fg_vk_dense_q8_0_subgroup(context,y,w,x,INPUT,OUTPUT,TOKENS,0.75f,&error)==FG_ERR_ARGUMENT&&fg_vk_embedding_q8_0(context,y,w,0u,INPUT,OUTPUT,1u,&error)==FG_ERR_ARGUMENT;for(uint32_t token=0;ok&&token<TOKENS;token++)for(uint32_t row=0;row<OUTPUT;row++){float ref=0.0f;const uint8_t *blocks=quantized+(uint64_t)row*source_row;for(uint32_t block=0;block<INPUT/32u;block++){const uint8_t *p=blocks+block*FG_Q8_0_BLOCK_BYTES;float delta=fg_f16_to_f32((uint16_t)p[0]|(uint16_t)((uint16_t)p[1]<<8u));for(uint32_t i=0;i<32u;i++)ref=fmaf(delta*(float)(int8_t)p[2u+i],input[token*INPUT+block*32u+i],ref);}ref*=0.75f;if(fabsf(got[token*OUTPUT+row]-ref)>2e-4f*fmaxf(1.0f,fabsf(ref))){fprintf(stderr,"cooked Q8 token %u row %u GPU=%g CPU=%g\n",token,row,got[token*OUTPUT+row],ref);ok=0;break;}}fg_vk_tensor_destroy(y);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);free(cooked);free(quantized);free(source);return ok;
}

static double elapsed_ms(struct timespec begin,struct timespec end){return(double)(end.tv_sec-begin.tv_sec)*1000.0+(double)(end.tv_nsec-begin.tv_nsec)*1e-6;}

static int bench_q8_subgroup_shape(uint32_t input_width,uint32_t output_width,uint32_t iterations,const char *name){
    uint32_t blocks=input_width/32u,row_bytes=blocks*FG_Q8_0_BLOCK_BYTES;uint64_t weight_bytes=(uint64_t)output_width*row_bytes,logical_bytes=weight_bytes+(uint64_t)(input_width+output_width)*4u;fg_vk_tensor *weights=tensor(NULL,weight_bytes),*input=tensor(NULL,(uint64_t)input_width*4u),*generic=tensor(NULL,(uint64_t)output_width*4u),*subgroup=tensor(NULL,(uint64_t)output_width*4u);float *generic_values=malloc((size_t)output_width*4u),*subgroup_values=malloc((size_t)output_width*4u);int ok=weights&&input&&generic&&subgroup&&generic_values&&subgroup_values;if(!ok)goto done;uint8_t *packed=fg_vk_tensor_map(weights);memset(packed,1,(size_t)weight_bytes);for(uint32_t row=0;row<output_width;row++)for(uint32_t block=0;block<blocks;block++){uint16_t delta=fg_f32_to_f16(0.001f+(float)((row+block)%7u)*0.0001f);memcpy(packed+(uint64_t)row*row_bytes+(uint64_t)block*FG_Q8_0_BLOCK_BYTES,&delta,sizeof(delta));}float *input_values=fg_vk_tensor_map(input);for(uint32_t i=0;i<input_width;i++)input_values[i]=sinf((float)(i+3u)*0.013f)+0.1f*cosf((float)i*0.007f);ok=fg_vk_dense_q8_0_f32(context,generic,weights,input,input_width,output_width,1u,1.0f,&error)==FG_OK&&fg_vk_dense_q8_0_subgroup(context,subgroup,weights,input,input_width,output_width,1u,1.0f,&error)==FG_OK&&fg_vk_tensor_read(generic,0,generic_values,(uint64_t)output_width*4u,&error)==FG_OK&&fg_vk_tensor_read(subgroup,0,subgroup_values,(uint64_t)output_width*4u,&error)==FG_OK;double max_abs=0.0,max_rel=0.0;for(uint32_t i=0;ok&&i<output_width;i++){double difference=fabs((double)generic_values[i]-subgroup_values[i]),relative=difference/fmax(1.0,fabs((double)generic_values[i]));if(difference>max_abs)max_abs=difference;if(relative>max_rel)max_rel=relative;if(relative>2e-4)ok=0;}fg_vk_profile generic_profile={0},subgroup_profile={0};struct timespec begin,end;double generic_wall=0.0,subgroup_wall=0.0;if(ok)ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"bench_generic",&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&begin);if(ok)ok=fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_dense_q8_0_f32(context,generic,weights,input,input_width,output_width,1u,1.0f,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);generic_wall=elapsed_ms(begin,end)/(double)iterations;if(ok)ok=fg_vk_profile_end(context,&generic_profile,&error)==FG_OK;if(ok)ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"bench_subgroup",&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&begin);if(ok)ok=fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_dense_q8_0_subgroup(context,subgroup,weights,input,input_width,output_width,1u,1.0f,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);subgroup_wall=elapsed_ms(begin,end)/(double)iterations;if(ok)ok=fg_vk_profile_end(context,&subgroup_profile,&error)==FG_OK;if(ok){double generic_gpu=generic_profile.kernel_ms/(double)iterations,subgroup_gpu=subgroup_profile.kernel_ms/(double)iterations;fprintf(stderr,"Q8_SUBGROUP_AB shape=%s input=%u output=%u logical_mb=%.3f iterations=%u generic_gpu_us=%.3f generic_wall_us=%.3f generic_gbps=%.3f subgroup_gpu_us=%.3f subgroup_wall_us=%.3f subgroup_gbps=%.3f speedup=%.4f max_abs=%.8g max_rel=%.8g parity=PASS\n",name,input_width,output_width,(double)logical_bytes/1e6,iterations,generic_gpu*1000.0,generic_wall*1000.0,(double)logical_bytes/(generic_gpu*1e6),subgroup_gpu*1000.0,subgroup_wall*1000.0,(double)logical_bytes/(subgroup_gpu*1e6),generic_gpu/subgroup_gpu,max_abs,max_rel);}
done:fg_vk_tensor_destroy(subgroup);fg_vk_tensor_destroy(generic);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(weights);free(subgroup_values);free(generic_values);return ok;
}

static int test_q8_subgroup_benchmark(void){if(!getenv("FG_BENCH_Q8_SUBGROUP"))return 1;struct{uint32_t input,output,iterations;const char *name;}shapes[]={{2560u,FG_Q38_VOCAB_SIZE,3u,"output_vocab"},{10240u,320u,12u,"hc_down"},{320u,10240u,12u,"hc_up"},{2560u,12288u,12u,"qsa_qg"},{2560u,10240u,12u,"gdn_qkv_ple"},{2560u,6144u,12u,"gdn_z"},{2560u,640u,20u,"shared_gate"}};int ok=1;for(uint32_t i=0;ok&&i<sizeof(shapes)/sizeof(shapes[0]);i++)ok=bench_q8_subgroup_shape(shapes[i].input,shapes[i].output,shapes[i].iterations,shapes[i].name);return ok;}

static int bench_q8_cooked_shape(uint32_t input_width,uint32_t output_width,uint32_t iterations,const char *name){
    uint32_t blocks=input_width/32u,source_row=blocks*FG_Q8_0_BLOCK_BYTES;uint64_t source_bytes=(uint64_t)output_width*source_row,cooked_bytes=fg_q8_0_cooked_matrix_bytes(input_width,output_width),logical_bytes=source_bytes+(uint64_t)(input_width+output_width)*4u;fg_vk_tensor *weights=tensor(NULL,source_bytes),*cooked=tensor(NULL,cooked_bytes),*input=tensor(NULL,(uint64_t)input_width*4u),*generic=tensor(NULL,(uint64_t)output_width*4u),*candidate=tensor(NULL,(uint64_t)output_width*4u);float *generic_values=malloc((size_t)output_width*4u),*candidate_values=malloc((size_t)output_width*4u);int ok=weights&&cooked&&input&&generic&&candidate&&generic_values&&candidate_values;if(!ok)goto done;uint8_t *packed=fg_vk_tensor_map(weights);memset(packed,1,(size_t)source_bytes);for(uint32_t row=0;row<output_width;row++)for(uint32_t block=0;block<blocks;block++){uint16_t delta=fg_f32_to_f16(0.001f+(float)((row+block)%7u)*0.0001f);memcpy(packed+(uint64_t)row*source_row+(uint64_t)block*FG_Q8_0_BLOCK_BYTES,&delta,sizeof(delta));}ok=fg_cook_q8_0_rows(packed,fg_vk_tensor_map(cooked),cooked_bytes,input_width,output_width);float *input_values=fg_vk_tensor_map(input);for(uint32_t i=0;ok&&i<input_width;i++)input_values[i]=sinf((float)(i+3u)*0.013f)+0.1f*cosf((float)i*0.007f);if(ok)ok=fg_vk_dense_q8_0_f32(context,generic,weights,input,input_width,output_width,1u,1.0f,&error)==FG_OK&&fg_vk_dense_q8_0_cooked(context,candidate,cooked,input,input_width,output_width,1u,1.0f,&error)==FG_OK&&fg_vk_tensor_read(generic,0,generic_values,(uint64_t)output_width*4u,&error)==FG_OK&&fg_vk_tensor_read(candidate,0,candidate_values,(uint64_t)output_width*4u,&error)==FG_OK;double max_abs=0.0,max_rel=0.0;for(uint32_t i=0;ok&&i<output_width;i++){double difference=fabs((double)generic_values[i]-candidate_values[i]),relative=difference/fmax(1.0,fabs((double)generic_values[i]));if(difference>max_abs)max_abs=difference;if(relative>max_rel)max_rel=relative;if(relative>2e-4)ok=0;}fg_vk_profile generic_profile={0},cooked_profile={0};struct timespec begin,end;double generic_wall=0.0,cooked_wall=0.0;if(ok)ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"bench_generic",&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&begin);if(ok)ok=fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_dense_q8_0_f32(context,generic,weights,input,input_width,output_width,1u,1.0f,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);generic_wall=elapsed_ms(begin,end)/(double)iterations;if(ok)ok=fg_vk_profile_end(context,&generic_profile,&error)==FG_OK;if(ok)ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"bench_cooked",&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&begin);if(ok)ok=fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_dense_q8_0_cooked(context,candidate,cooked,input,input_width,output_width,1u,1.0f,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);cooked_wall=elapsed_ms(begin,end)/(double)iterations;if(ok)ok=fg_vk_profile_end(context,&cooked_profile,&error)==FG_OK;if(ok){double generic_gpu=generic_profile.kernel_ms/(double)iterations,cooked_gpu=cooked_profile.kernel_ms/(double)iterations;fprintf(stderr,"Q8_COOKED_AB shape=%s input=%u output=%u logical_mb=%.3f cooked_mb=%.3f iterations=%u generic_gpu_us=%.3f generic_wall_us=%.3f generic_gbps=%.3f cooked_gpu_us=%.3f cooked_wall_us=%.3f cooked_logical_gbps=%.3f speedup=%.4f max_abs=%.8g max_rel=%.8g parity=PASS\n",name,input_width,output_width,(double)logical_bytes/1e6,(double)cooked_bytes/1e6,iterations,generic_gpu*1000.0,generic_wall*1000.0,(double)logical_bytes/(generic_gpu*1e6),cooked_gpu*1000.0,cooked_wall*1000.0,(double)logical_bytes/(cooked_gpu*1e6),generic_gpu/cooked_gpu,max_abs,max_rel);}
done:fg_vk_tensor_destroy(candidate);fg_vk_tensor_destroy(generic);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(cooked);fg_vk_tensor_destroy(weights);free(candidate_values);free(generic_values);return ok;
}

static int test_q8_cooked_benchmark(void){if(!getenv("FG_BENCH_Q8_COOKED"))return 1;struct{uint32_t input,output,iterations;const char *name;}shapes[]={{2560u,FG_Q38_VOCAB_SIZE,3u,"output_vocab"},{10240u,320u,12u,"hc_down"},{320u,10240u,12u,"hc_up"},{2560u,12288u,12u,"qsa_qg"},{2560u,10240u,12u,"gdn_qkv_ple"},{2560u,6144u,12u,"gdn_z"},{2560u,2560u,16u,"ple_value"},{6144u,2560u,16u,"attention_out"},{640u,2560u,20u,"shared_down"},{2560u,640u,20u,"shared_gate"},{2560u,512u,20u,"qsa_kv"}};int ok=1;for(uint32_t i=0;ok&&i<sizeof(shapes)/sizeof(shapes[0]);i++)ok=bench_q8_cooked_shape(shapes[i].input,shapes[i].output,shapes[i].iterations,shapes[i].name);return ok;}

static int test_q8_embedding(void){
    enum{ROWS=3,WIDTH=2560,COPIES=4};
    const uint64_t row_bytes=(uint64_t)(WIDTH/32u)*FG_Q8_0_BLOCK_BYTES;
    float *source=malloc((size_t)ROWS*WIDTH*sizeof(*source));
    float *expected=malloc(WIDTH*sizeof(*expected));
    float *got=malloc((size_t)COPIES*WIDTH*sizeof(*got));
    uint8_t *weights=malloc((size_t)ROWS*row_bytes);
    if(!source||!expected||!got||!weights){free(weights);free(got);free(expected);free(source);return 0;}
    for(uint32_t row=0;row<ROWS;row++)for(uint32_t i=0;i<WIDTH;i++)source[row*WIDTH+i]=sinf((float)(row*WIDTH+i)*0.0037f)*(float)(row+1u)+cosf((float)i*0.011f)*0.125f;
    for(uint32_t row=0;row<ROWS;row++)fg_quantize_q8_0(source+row*WIDTH,weights+(uint64_t)row*row_bytes,WIDTH);
    fg_dequantize_q8_0(weights+row_bytes,expected,WIDTH);
    uint32_t token_ids[ROWS]={2u,0u,1u};float *batch=malloc((size_t)ROWS*COPIES*WIDTH*4u);fg_vk_tensor *w=tensor(weights,(uint64_t)ROWS*row_bytes),*out=tensor(NULL,(uint64_t)COPIES*WIDTH*sizeof(*got)),*ids=tensor(token_ids,sizeof(token_ids)),*batch_out=tensor(NULL,(uint64_t)ROWS*COPIES*WIDTH*4u);
    int ok=batch&&w&&out&&ids&&batch_out&&fg_vk_embedding_q8_0(context,out,w,1u,WIDTH,ROWS,COPIES,&error)==FG_OK&&fg_vk_tensor_read(out,0,got,(uint64_t)COPIES*WIDTH*sizeof(*got),&error)==FG_OK&&fg_vk_embedding_q8_0_batch(context,batch_out,w,ids,ROWS,WIDTH,ROWS,COPIES,&error)==FG_OK&&fg_vk_tensor_read(batch_out,0,batch,(uint64_t)ROWS*COPIES*WIDTH*4u,&error)==FG_OK;
    for(uint32_t copy=0;ok&&copy<COPIES;copy++)for(uint32_t i=0;i<WIDTH;i++)if(got[copy*WIDTH+i]!=expected[i]){fprintf(stderr,"Q8 embedding copy %u element %u GPU=%g CPU=%g\n",copy,i,got[copy*WIDTH+i],expected[i]);ok=0;break;}
    for(uint32_t token=0;ok&&token<ROWS;token++){fg_dequantize_q8_0(weights+(uint64_t)token_ids[token]*row_bytes,expected,WIDTH);for(uint32_t copy=0;ok&&copy<COPIES;copy++)for(uint32_t i=0;i<WIDTH;i++)if(batch[((uint64_t)token*COPIES+copy)*WIDTH+i]!=expected[i]){fprintf(stderr,"batched Q8 embedding token %u copy %u element %u mismatch\n",token,copy,i);ok=0;break;}}
    fg_vk_tensor_destroy(batch_out);fg_vk_tensor_destroy(ids);fg_vk_tensor_destroy(out);fg_vk_tensor_destroy(w);free(batch);free(weights);free(got);free(expected);free(source);return ok;
}

static int test_hc_finalize(void){
    enum{HIDDEN=2560,GROUPS=4,HYPER=10240};
    float *normalized=malloc(HYPER*sizeof(*normalized)),*up=malloc(HYPER*sizeof(*up));
    float *expected=malloc(HIDDEN*sizeof(*expected)),*got=malloc(HIDDEN*sizeof(*got));
    if(!normalized||!up||!expected||!got){free(got);free(expected);free(up);free(normalized);return 0;}
    for(uint32_t group=0;group<GROUPS;group++)for(uint32_t i=0;i<HIDDEN;i++){uint32_t index=group*HIDDEN+i;normalized[index]=sinf((float)index*0.0019f);up[index]=cosf((float)index*0.0023f)*3.0f;}
    for(uint32_t i=0;i<HIDDEN;i++){expected[i]=0.0f;for(uint32_t group=0;group<GROUPS;group++){uint32_t index=group*HIDDEN+i;expected[i]+=normalized[index]/(1.0f+expf(-up[index]));}expected[i]/=(float)GROUPS;}
    fg_vk_tensor *n=tensor(normalized,HYPER*sizeof(*normalized)),*u=tensor(up,HYPER*sizeof(*up)),*o=tensor(NULL,HIDDEN*sizeof(*got));
    int ok=n&&u&&o&&fg_vk_hc_finalize(context,o,n,u,HIDDEN,GROUPS,1u,&error)==FG_OK&&fg_vk_tensor_read(o,0,got,HIDDEN*sizeof(*got),&error)==FG_OK;
    for(uint32_t i=0;ok&&i<HIDDEN;i++)if(fabsf(got[i]-expected[i])>2e-6f){fprintf(stderr,"final hyper-connection %u GPU=%g CPU=%g\n",i,got[i],expected[i]);ok=0;}
    fg_vk_tensor_destroy(o);fg_vk_tensor_destroy(u);fg_vk_tensor_destroy(n);free(got);free(expected);free(up);free(normalized);return ok;
}

static int test_q8_k_quant(void){float input[512];uint8_t cpu[2*FG_Q8_K_BLOCK_BYTES],gpu[sizeof(cpu)];for(uint32_t i=0;i<512u;i++)input[i]=sinf((float)i*0.031f)*3.0f+cosf((float)i*0.007f);fg_quantize_q8_k(input,cpu,512);fg_vk_tensor *x=tensor(input,sizeof(input)),*q=tensor(NULL,sizeof(gpu));int ok=x&&q&&fg_vk_quantize_q8_k(context,q,x,256,2,&error)==FG_OK&&fg_vk_tensor_read(q,0,gpu,sizeof(gpu),&error)==FG_OK;if(ok&&memcmp(cpu,gpu,sizeof(cpu))!=0){fprintf(stderr,"Q8_K GPU bytes differ from CPU oracle\n");ok=0;}fg_vk_tensor_destroy(q);fg_vk_tensor_destroy(x);return ok;}
static int test_q4_0_bytes(void){float input[32],got[32];uint8_t quantized[18];for(uint32_t i=0;i<16u;i++)input[i]=input[i+16u]=(float)((int32_t)i-8);fg_quantize_q4_0(input,quantized,32);int ok=quantized[0]==0x00u&&quantized[1]==0x3cu;for(uint32_t i=0;ok&&i<16u;i++)if(quantized[2u+i]!=(uint8_t)(i|(i<<4u)))ok=0;fg_dequantize_q4_0(quantized,got,32);for(uint32_t i=0;ok&&i<32u;i++)if(got[i]!=input[i])ok=0;return ok;}
static uint16_t to_bf16(float value){uint32_t bits;memcpy(&bits,&value,4u);return(uint16_t)(bits>>16u);}
static float from_bf16(uint16_t value){uint32_t bits=(uint32_t)value<<16u;float result;memcpy(&result,&bits,4u);return result;}
static int test_qsa_quant_and_bf16(void){
    enum{WIDTH=512,OUTPUT=7};float input[WIDTH],deq_cpu[WIDTH],deq_gpu[WIDTH],expected[OUTPUT],got[OUTPUT];uint8_t q8_cpu[(WIDTH/32)*34],q8_gpu[sizeof(q8_cpu)],q4_cpu[(WIDTH/32)*18],q4_gpu[sizeof(q4_cpu)];for(uint32_t i=0;i<WIDTH;i++)input[i]=sinf((float)i*0.023f)*4.0f+cosf((float)i*0.071f)*0.25f;fg_quantize_q8_0(input,q8_cpu,WIDTH);fg_quantize_q4_0(input,q4_cpu,WIDTH);fg_vk_tensor *x=tensor(input,sizeof(input)),*q8=tensor(NULL,sizeof(q8_gpu)),*q4=tensor(NULL,sizeof(q4_gpu));int ok=x&&q8&&q4&&fg_vk_quantize_q8_0(context,q8,x,WIDTH,1,&error)==FG_OK&&fg_vk_quantize_q4_0(context,q4,x,WIDTH,1,&error)==FG_OK&&fg_vk_tensor_read(q8,0,q8_gpu,sizeof(q8_gpu),&error)==FG_OK&&fg_vk_tensor_read(q4,0,q4_gpu,sizeof(q4_gpu),&error)==FG_OK;
    /* Compare dequantized values rather than raw bytes — real GPU float
       rounding (e.g. RADV GFX1013) may produce a different scale factor
       while the dequantized result stays within one quantization step. */
    fg_dequantize_q8_0(q8_cpu,deq_cpu,WIDTH);fg_dequantize_q8_0(q8_gpu,deq_gpu,WIDTH);for(uint32_t i=0;ok&&i<WIDTH;i++)if(fabsf(deq_cpu[i]-deq_gpu[i])>0.05f*fmaxf(1.0f,fabsf(input[i]))){fprintf(stderr,"Q8_0 GPU dequant %u CPU=%g GPU=%g input=%g\n",i,deq_cpu[i],deq_gpu[i],input[i]);ok=0;}
    fg_dequantize_q4_0(q4_cpu,deq_cpu,WIDTH);fg_dequantize_q4_0(q4_gpu,deq_gpu,WIDTH);for(uint32_t i=0;ok&&i<WIDTH;i++)if(fabsf(deq_cpu[i]-deq_gpu[i])>0.25f*fmaxf(1.0f,fabsf(input[i]))){fprintf(stderr,"Q4_0 GPU dequant %u CPU=%g GPU=%g input=%g\n",i,deq_cpu[i],deq_gpu[i],input[i]);ok=0;}
    for(uint32_t i=0;ok&&i<WIDTH;i++)if(!isfinite(deq_gpu[i]))ok=0;
    uint16_t weights[WIDTH*OUTPUT];for(uint32_t row=0;row<OUTPUT;row++){expected[row]=0.0f;for(uint32_t i=0;i<WIDTH;i++){weights[row*WIDTH+i]=to_bf16(cosf((float)(row*WIDTH+i)*0.0031f));expected[row]=fmaf(from_bf16(weights[row*WIDTH+i]),input[i],expected[row]);}}fg_vk_tensor *w=tensor(weights,sizeof(weights)),*o=tensor(NULL,sizeof(got));ok=ok&&w&&o&&fg_vk_dense_bf16_f32(context,o,w,x,WIDTH,OUTPUT,1,&error)==FG_OK&&fg_vk_tensor_read(o,0,got,sizeof(got),&error)==FG_OK;for(uint32_t row=0;ok&&row<OUTPUT;row++)if(fabsf(got[row]-expected[row])>3e-4f*fmaxf(1.0f,fabsf(expected[row]))){fprintf(stderr,"BF16 dense row %u GPU=%g CPU=%g\n",row,got[row],expected[row]);ok=0;}fg_vk_tensor_destroy(o);fg_vk_tensor_destroy(w);fg_vk_tensor_destroy(q4);fg_vk_tensor_destroy(q8);fg_vk_tensor_destroy(x);return ok;
}

typedef struct score_id {float score;uint32_t id;} score_id;
static int score_id_compare(const void *a,const void *b);
static int test_qsa_record_commit(void){
    enum{SLOTS=2,TOKENS=3,KEY=512,INDEX=128};float key[KEY],value[KEY],index_key[INDEX];uint32_t position[3]={31u,37u,41u};uint8_t expected[FG_Q38_QSA_TOKEN_RECORD_BYTES],got[sizeof(expected)],history[FG_Q38_QSA_INDEX_KEY_BYTES];for(uint32_t i=0;i<KEY;i++){key[i]=sinf((float)i*0.013f);value[i]=cosf((float)i*0.017f)*2.0f;}for(uint32_t i=0;i<INDEX;i++)index_key[i]=sinf((float)i*0.029f)-0.25f;memset(expected,0,sizeof(expected));
    fg_vk_tensor *gkey=tensor(key,sizeof(key)),*gvalue=tensor(value,sizeof(value)),*gindex=tensor(index_key,sizeof(index_key)),*gposition=tensor(position,sizeof(position)),*qkey=tensor(NULL,FG_Q38_QSA_KEY_BYTES),*qvalue=tensor(NULL,FG_Q38_QSA_VALUE_BYTES),*qindex=tensor(NULL,FG_Q38_QSA_INDEX_KEY_BYTES),*records=tensor(NULL,(uint64_t)SLOTS*TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES),*index_history=tensor(NULL,(uint64_t)SLOTS*TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES);int ok=gkey&&gvalue&&gindex&&gposition&&qkey&&qvalue&&qindex&&records&&index_history;if(ok){memset(fg_vk_tensor_map(records),0,(size_t)fg_vk_tensor_bytes(records));memset(fg_vk_tensor_map(index_history),0,(size_t)fg_vk_tensor_bytes(index_history));ok=fg_vk_begin(context,&error)==FG_OK&&fg_vk_quantize_q8_0(context,qkey,gkey,KEY,1u,&error)==FG_OK&&fg_vk_quantize_q8_0(context,qvalue,gvalue,KEY,1u,&error)==FG_OK&&fg_vk_quantize_q8_0(context,qindex,gindex,INDEX,1u,&error)==FG_OK&&fg_vk_qsa_record_commit(context,records,index_history,qkey,qvalue,qindex,gposition,1u,2u,TOKENS,&error)==FG_OK;}if(ok){const uint8_t *pending=(const uint8_t *)fg_vk_tensor_map(records)+(1u*TOKENS+2u)*FG_Q38_QSA_TOKEN_RECORD_BYTES;for(uint32_t i=0;i<FG_Q38_QSA_TOKEN_RECORD_BYTES;i++)if(pending[i]!=0u){ok=0;break;}}if(ok)ok=fg_vk_end(context,&error)==FG_OK;if(ok)ok=fg_vk_tensor_read(qkey,0,expected,FG_Q38_QSA_KEY_BYTES,&error)==FG_OK&&fg_vk_tensor_read(qvalue,0,expected+FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_VALUE_BYTES,&error)==FG_OK&&fg_vk_tensor_read(qindex,0,expected+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,FG_Q38_QSA_INDEX_KEY_BYTES,&error)==FG_OK;if(ok){memcpy(expected+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES,position,sizeof(position));ok=fg_vk_tensor_read(records,(uint64_t)(1u*TOKENS+2u)*FG_Q38_QSA_TOKEN_RECORD_BYTES,got,sizeof(got),&error)==FG_OK&&fg_vk_tensor_read(index_history,(uint64_t)(1u*TOKENS+2u)*FG_Q38_QSA_INDEX_KEY_BYTES,history,sizeof(history),&error)==FG_OK&&memcmp(got,expected,sizeof(got))==0&&memcmp(history,expected+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,sizeof(history))==0;}else if(fg_vk_batch_active(context))fg_vk_end(context,&error);for(uint32_t slot=0;ok&&slot<SLOTS;slot++)for(uint32_t token_id=0;token_id<TOKENS;token_id++)if(slot!=1u||token_id!=2u){const uint8_t *record=(const uint8_t *)fg_vk_tensor_map(records)+(slot*TOKENS+token_id)*FG_Q38_QSA_TOKEN_RECORD_BYTES;for(uint32_t i=0;i<FG_Q38_QSA_TOKEN_RECORD_BYTES;i++)if(record[i]!=0u){ok=0;break;}}fg_vk_tensor_destroy(index_history);fg_vk_tensor_destroy(records);fg_vk_tensor_destroy(qindex);fg_vk_tensor_destroy(qvalue);fg_vk_tensor_destroy(qkey);fg_vk_tensor_destroy(gposition);fg_vk_tensor_destroy(gindex);fg_vk_tensor_destroy(gvalue);fg_vk_tensor_destroy(gkey);return ok;
}

static int test_qsa_tiered_record_commit(void){
    enum{SLOTS=2,HOT=4,INDEX=12};uint8_t key[FG_Q38_QSA_KEY_BYTES],
        value[FG_Q38_QSA_VALUE_BYTES],index_key[FG_Q38_QSA_INDEX_KEY_BYTES];
    uint32_t position[3]={101u,103u,107u};
    for(uint32_t i=0;i<sizeof(key);i++)key[i]=(uint8_t)(i*3u+1u);
    for(uint32_t i=0;i<sizeof(value);i++)value[i]=(uint8_t)(i*5u+2u);
    for(uint32_t i=0;i<sizeof(index_key);i++)index_key[i]=(uint8_t)(i*7u+4u);
    fg_vk_tensor *gkey=tensor(key,sizeof(key)),*gvalue=tensor(value,sizeof(value)),
        *gindex=tensor(index_key,sizeof(index_key)),*gposition=tensor(position,sizeof(position)),
        *records=tensor(NULL,(uint64_t)SLOTS*HOT*FG_Q38_QSA_TOKEN_RECORD_BYTES),
        *history=tensor(NULL,(uint64_t)SLOTS*INDEX*FG_Q38_QSA_INDEX_KEY_BYTES);
    int ok=gkey&&gvalue&&gindex&&gposition&&records&&history;
    if(ok){memset(fg_vk_tensor_map(records),0,(size_t)fg_vk_tensor_bytes(records));
        memset(fg_vk_tensor_map(history),0,(size_t)fg_vk_tensor_bytes(history));
        ok=fg_vk_begin(context,&error)==FG_OK&&
           fg_vk_qsa_record_commit_tiered(context,records,history,gkey,gvalue,gindex,
               gposition,1u,9u,INDEX,1u,HOT,&error)==FG_OK;}
    const uint8_t *pending=ok?(const uint8_t *)fg_vk_tensor_map(records)+
        (1u*HOT+1u)*FG_Q38_QSA_TOKEN_RECORD_BYTES:NULL;
    for(uint32_t i=0;ok&&i<FG_Q38_QSA_TOKEN_RECORD_BYTES;i++)if(pending[i])ok=0;
    if(ok)ok=fg_vk_end(context,&error)==FG_OK;
    uint8_t expected[FG_Q38_QSA_TOKEN_RECORD_BYTES],got[sizeof(expected)],got_index[sizeof(index_key)];
    memcpy(expected,key,sizeof(key));memcpy(expected+sizeof(key),value,sizeof(value));
    memcpy(expected+sizeof(key)+sizeof(value),index_key,sizeof(index_key));
    memcpy(expected+sizeof(key)+sizeof(value)+sizeof(index_key),position,sizeof(position));
    if(ok)ok=fg_vk_tensor_read(records,(uint64_t)(1u*HOT+1u)*
        FG_Q38_QSA_TOKEN_RECORD_BYTES,got,sizeof(got),&error)==FG_OK&&
        fg_vk_tensor_read(history,(uint64_t)(1u*INDEX+9u)*FG_Q38_QSA_INDEX_KEY_BYTES,
                          got_index,sizeof(got_index),&error)==FG_OK&&
        memcmp(got,expected,sizeof(got))==0&&memcmp(got_index,index_key,sizeof(got_index))==0;
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(history);fg_vk_tensor_destroy(records);fg_vk_tensor_destroy(gposition);
    fg_vk_tensor_destroy(gindex);fg_vk_tensor_destroy(gvalue);fg_vk_tensor_destroy(gkey);
    return ok;
}

static int test_qsa_segmented_record_commit(void){
    enum{SLOTS=2,HOT=4,INDEX=4};
    uint8_t key[FG_Q38_QSA_KEY_BYTES],value[FG_Q38_QSA_VALUE_BYTES],
        index_key[FG_Q38_QSA_INDEX_KEY_BYTES];
    uint32_t position[3]={211u,223u,227u};
    for(uint32_t i=0;i<sizeof(key);i++)key[i]=(uint8_t)(i*3u+1u);
    for(uint32_t i=0;i<sizeof(value);i++)value[i]=(uint8_t)(i*5u+2u);
    for(uint32_t i=0;i<sizeof(index_key);i++)index_key[i]=(uint8_t)(i*7u+4u);
    fg_vk_tensor *gkey=tensor(key,sizeof(key)),*gvalue=tensor(value,sizeof(value)),
        *gindex=tensor(index_key,sizeof(index_key)),*gposition=tensor(position,sizeof(position)),
        *contiguous_records=tensor(NULL,(uint64_t)SLOTS*HOT*
                                   FG_Q38_QSA_TOKEN_RECORD_BYTES),
        *segmented_records=tensor(NULL,(uint64_t)SLOTS*HOT*
                                  FG_Q38_QSA_TOKEN_RECORD_BYTES),
        *contiguous_history=tensor(NULL,(uint64_t)SLOTS*12u*
                                   FG_Q38_QSA_INDEX_KEY_BYTES),
        *segmented_history=tensor(NULL,(uint64_t)INDEX*
                                  FG_Q38_QSA_INDEX_KEY_BYTES);
    int ok=gkey&&gvalue&&gindex&&gposition&&contiguous_records&&
        segmented_records&&contiguous_history&&segmented_history;
    if(ok){
        memset(fg_vk_tensor_map(contiguous_records),0,
               (size_t)fg_vk_tensor_bytes(contiguous_records));
        memset(fg_vk_tensor_map(segmented_records),0,
               (size_t)fg_vk_tensor_bytes(segmented_records));
        memset(fg_vk_tensor_map(contiguous_history),0,
               (size_t)fg_vk_tensor_bytes(contiguous_history));
        memset(fg_vk_tensor_map(segmented_history),0,
               (size_t)fg_vk_tensor_bytes(segmented_history));
        ok=fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_qsa_record_commit_tiered(context,contiguous_records,
                contiguous_history,gkey,gvalue,gindex,gposition,1u,9u,12u,
                1u,HOT,&error)==FG_OK&&
            fg_vk_qsa_record_commit_segmented(context,segmented_records,
                segmented_history,gkey,gvalue,gindex,gposition,1u,9u,1u,
                INDEX,1u,HOT,&error)==FG_OK;
    }
    if(ok)ok=fg_vk_end(context,&error)==FG_OK;
    uint8_t contiguous_record[FG_Q38_QSA_TOKEN_RECORD_BYTES],
        segmented_record[FG_Q38_QSA_TOKEN_RECORD_BYTES],
        contiguous_index[FG_Q38_QSA_INDEX_KEY_BYTES],
        segmented_index[FG_Q38_QSA_INDEX_KEY_BYTES];
    if(ok)ok=fg_vk_tensor_read(contiguous_records,
        (uint64_t)(HOT+1u)*FG_Q38_QSA_TOKEN_RECORD_BYTES,contiguous_record,
        sizeof(contiguous_record),&error)==FG_OK&&
        fg_vk_tensor_read(segmented_records,
        (uint64_t)(HOT+1u)*FG_Q38_QSA_TOKEN_RECORD_BYTES,segmented_record,
        sizeof(segmented_record),&error)==FG_OK&&
        fg_vk_tensor_read(contiguous_history,
        (uint64_t)(12u+9u)*FG_Q38_QSA_INDEX_KEY_BYTES,contiguous_index,
        sizeof(contiguous_index),&error)==FG_OK&&
        fg_vk_tensor_read(segmented_history,
        (uint64_t)1u*FG_Q38_QSA_INDEX_KEY_BYTES,segmented_index,
        sizeof(segmented_index),&error)==FG_OK&&
        memcmp(contiguous_record,segmented_record,sizeof(contiguous_record))==0&&
        memcmp(contiguous_index,segmented_index,sizeof(contiguous_index))==0;
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(segmented_history);fg_vk_tensor_destroy(contiguous_history);
    fg_vk_tensor_destroy(segmented_records);fg_vk_tensor_destroy(contiguous_records);
    fg_vk_tensor_destroy(gposition);fg_vk_tensor_destroy(gindex);
    fg_vk_tensor_destroy(gvalue);fg_vk_tensor_destroy(gkey);
    return ok;
}

static void qsa_constant_record(uint8_t *record,float value){
    float key[FG_Q38_ATTN_KV_WIDTH]={0};
    float values[FG_Q38_ATTN_KV_WIDTH];
    float index_key[FG_Q38_INDEX_WIDTH]={0};
    for(uint32_t i=0;i<FG_Q38_ATTN_KV_WIDTH;i++)values[i]=value;
    fg_quantize_q8_0(key,record,FG_Q38_ATTN_KV_WIDTH);
    fg_quantize_q8_0(values,record+FG_Q38_QSA_KEY_BYTES,
                     FG_Q38_ATTN_KV_WIDTH);
    fg_quantize_q8_0(index_key,record+FG_Q38_QSA_KEY_BYTES+
                     FG_Q38_QSA_VALUE_BYTES,FG_Q38_INDEX_WIDTH);
    memset(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+
           FG_Q38_QSA_INDEX_KEY_BYTES,0,FG_Q38_QSA_POSITION_BYTES);
}

static int test_qsa_resident_commit_exact(void){
    enum{TOKENS=5,GUARD=64};
    const uint32_t segment_capacity=FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY;
    const uint32_t capacity=segment_capacity+1u,first=segment_capacity-4u;
    uint8_t keys[TOKENS*FG_Q38_QSA_KEY_BYTES];
    uint8_t values[TOKENS*FG_Q38_QSA_VALUE_BYTES];
    uint8_t index_keys[TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES];
    uint32_t positions[TOKENS*3u];
    for(uint32_t token=0;token<TOKENS;token++){
        uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES];
        qsa_constant_record(record,(float)(token+1u));
        memcpy(keys+(uint64_t)token*FG_Q38_QSA_KEY_BYTES,record,
               FG_Q38_QSA_KEY_BYTES);
        memcpy(values+(uint64_t)token*FG_Q38_QSA_VALUE_BYTES,
               record+FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_VALUE_BYTES);
        for(uint32_t i=0;i<FG_Q38_QSA_INDEX_KEY_BYTES;i++)
            index_keys[(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES+i]=
                (uint8_t)(token*31u+i*5u);
        positions[token*3u]=first+token;
        positions[token*3u+1u]=first+token+7u;
        positions[token*3u+2u]=first+token+11u;
    }
    uint64_t record_0_bytes=(uint64_t)segment_capacity*
        FG_Q38_QSA_TOKEN_RECORD_BYTES;
    uint64_t record_1_bytes=FG_Q38_QSA_TOKEN_RECORD_BYTES;
    uint64_t index_0_bytes=(uint64_t)segment_capacity*
        FG_Q38_QSA_INDEX_KEY_BYTES;
    uint64_t index_1_bytes=FG_Q38_QSA_INDEX_KEY_BYTES;
    fg_vk_tensor *record_parent_0=tensor(NULL,record_0_bytes+2u*GUARD);
    fg_vk_tensor *record_parent_1=tensor(NULL,record_1_bytes+2u*GUARD);
    fg_vk_tensor *index_parent_0=tensor(NULL,index_0_bytes+2u*GUARD);
    fg_vk_tensor *index_parent_1=tensor(NULL,index_1_bytes+2u*GUARD);
    fg_vk_tensor *record_0=NULL,*record_1=NULL,*index_0=NULL,*index_1=NULL;
    fg_vk_tensor *gkeys=tensor(keys,sizeof(keys)),*gvalues=tensor(values,sizeof(values));
    fg_vk_tensor *gindex=tensor(index_keys,sizeof(index_keys));
    fg_vk_tensor *gpositions=tensor(positions,sizeof(positions));
    int ok=record_parent_0&&record_parent_1&&index_parent_0&&index_parent_1&&
        gkeys&&gvalues&&gindex&&gpositions;
    if(ok){
        memset(fg_vk_tensor_map(record_parent_0),0xa5,
               (size_t)fg_vk_tensor_bytes(record_parent_0));
        memset(fg_vk_tensor_map(record_parent_1),0xa5,
               (size_t)fg_vk_tensor_bytes(record_parent_1));
        memset(fg_vk_tensor_map(index_parent_0),0xa5,
               (size_t)fg_vk_tensor_bytes(index_parent_0));
        memset(fg_vk_tensor_map(index_parent_1),0xa5,
               (size_t)fg_vk_tensor_bytes(index_parent_1));
        ok=fg_vk_tensor_view(record_parent_0,GUARD,record_0_bytes,&record_0,&error)==FG_OK&&
            fg_vk_tensor_view(record_parent_1,GUARD,record_1_bytes,&record_1,&error)==FG_OK&&
            fg_vk_tensor_view(index_parent_0,GUARD,index_0_bytes,&index_0,&error)==FG_OK&&
            fg_vk_tensor_view(index_parent_1,GUARD,index_1_bytes,&index_1,&error)==FG_OK;
    }
    if(ok)ok=fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_qsa_resident_record_commit(context,record_0,record_1,index_0,index_1,
            gkeys,gvalues,gindex,gpositions,first,TOKENS,capacity,
            segment_capacity,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK;
    fg_vk_tensor *parents[]={record_parent_0,record_parent_1,index_parent_0,index_parent_1};
    uint64_t payloads[]={record_0_bytes,record_1_bytes,index_0_bytes,index_1_bytes};
    for(uint32_t p=0;ok&&p<4u;p++){
        const uint8_t *bytes=fg_vk_tensor_const_map(parents[p]);
        for(uint32_t i=0;i<GUARD;i++)
            if(bytes[i]!=0xa5||bytes[GUARD+payloads[p]+i]!=0xa5){ok=0;break;}
    }
    for(uint32_t token=0;ok&&token<TOKENS;token++){
        uint32_t absolute=first+token;
        fg_vk_tensor *record_segment=absolute<segment_capacity?record_0:record_1;
        fg_vk_tensor *index_segment=absolute<segment_capacity?index_0:index_1;
        uint32_t local=absolute%segment_capacity;
        const uint8_t *record=(const uint8_t *)fg_vk_tensor_const_map(record_segment)+
            (uint64_t)local*FG_Q38_QSA_TOKEN_RECORD_BYTES;
        const uint8_t *index=(const uint8_t *)fg_vk_tensor_const_map(index_segment)+
            (uint64_t)local*FG_Q38_QSA_INDEX_KEY_BYTES;
        ok=memcmp(record,keys+(uint64_t)token*FG_Q38_QSA_KEY_BYTES,
                  FG_Q38_QSA_KEY_BYTES)==0&&
            memcmp(record+FG_Q38_QSA_KEY_BYTES,
                   values+(uint64_t)token*FG_Q38_QSA_VALUE_BYTES,
                   FG_Q38_QSA_VALUE_BYTES)==0&&
            memcmp(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,
                   index_keys+(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES,
                   FG_Q38_QSA_INDEX_KEY_BYTES)==0&&
            memcmp(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+
                   FG_Q38_QSA_INDEX_KEY_BYTES,positions+token*3u,
                   FG_Q38_QSA_POSITION_BYTES)==0&&
            memcmp(index,index_keys+(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES,
                   FG_Q38_QSA_INDEX_KEY_BYTES)==0;
    }
    uint32_t selected[2u*FG_QSA_TOPK_CANDIDATES];
    float attention_query[2u*6144u]={0},attention_gate[2u*6144u],
        attention_output[2u*6144u];
    for(uint32_t i=0;i<2u*FG_QSA_TOPK_CANDIDATES;i++)selected[i]=UINT32_MAX;
    selected[0]=segment_capacity/FG_Q38_QSA_COMPRESS_RATIO-1u;
    selected[FG_QSA_TOPK_CANDIDATES]=selected[0];
    for(uint32_t i=0;i<2u*6144u;i++)attention_gate[i]=20.0f;
    fg_vk_tensor *gselected=ok?tensor(selected,sizeof(selected)):NULL;
    fg_vk_tensor *gaq=ok?tensor(attention_query,sizeof(attention_query)):NULL;
    fg_vk_tensor *gag=ok?tensor(attention_gate,sizeof(attention_gate)):NULL;
    fg_vk_tensor *gao=ok?tensor(NULL,sizeof(attention_output)):NULL;
    if(ok)ok=gselected&&gaq&&gag&&gao&&fg_vk_qsa_resident_attention(
        context,gao,record_0,record_1,gselected,gaq,gag,segment_capacity-1u,2u,
        capacity,segment_capacity,FG_QSA_TOPK_CANDIDATES,&error)==FG_OK&&
        fg_vk_tensor_read(gao,0,attention_output,sizeof(attention_output),
                          &error)==FG_OK;
    for(uint32_t query_slot=0;ok&&query_slot<2u;query_slot++)
        for(uint32_t i=0;i<6144u;i++){
            float expected=(query_slot?3.0f:2.5f)/(1.0f+expf(-20.0f));
            if(fabsf(attention_output[(uint64_t)query_slot*6144u+i]-expected)>
               2e-3f){ok=0;break;}
        }
    fg_vk_tensor_destroy(gao);fg_vk_tensor_destroy(gag);
    fg_vk_tensor_destroy(gaq);fg_vk_tensor_destroy(gselected);
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(gpositions);fg_vk_tensor_destroy(gindex);
    fg_vk_tensor_destroy(gvalues);fg_vk_tensor_destroy(gkeys);
    fg_vk_tensor_destroy(index_1);fg_vk_tensor_destroy(index_0);
    fg_vk_tensor_destroy(record_1);fg_vk_tensor_destroy(record_0);
    fg_vk_tensor_destroy(index_parent_1);fg_vk_tensor_destroy(index_parent_0);
    fg_vk_tensor_destroy(record_parent_1);fg_vk_tensor_destroy(record_parent_0);
    return ok;
}

static int test_qsa_resident_early_exhaustive(void){
    enum{BLOCKS=512,TOKENS=BLOCKS*4};
    uint8_t *keys=malloc((size_t)TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES);
    uint32_t *positions=calloc((size_t)TOKENS*3u,sizeof(*positions));
    float query[FG_Q38_INDEX_QUERY_WIDTH]={0},norm[FG_Q38_INDEX_WIDTH];
    uint32_t ids[FG_QSA_TOPK_CANDIDATES];
    for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)norm[i]=NAN;
    if(keys)memset(keys,0xff,(size_t)TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES);
    fg_vk_tensor *gkeys=keys?tensor(keys,(uint64_t)TOKENS*
                                   FG_Q38_QSA_INDEX_KEY_BYTES):NULL;
    fg_vk_tensor *gpositions=positions?tensor(positions,(uint64_t)TOKENS*3u*4u):NULL;
    fg_vk_tensor *gquery=tensor(query,sizeof(query)),*gnorm=tensor(norm,sizeof(norm));
    fg_vk_tensor *scores[2]={tensor(NULL,sizeof(ids)),tensor(NULL,sizeof(ids))};
    fg_vk_tensor *gpu_ids[2]={tensor(NULL,sizeof(ids)),tensor(NULL,sizeof(ids))};
    uint32_t side=UINT32_MAX;
    int ok=gkeys&&gpositions&&gquery&&gnorm&&scores[0]&&scores[1]&&
        gpu_ids[0]&&gpu_ids[1]&&fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_qsa_resident_select(context,scores[0],gpu_ids[0],scores[1],
            gpu_ids[1],gquery,gkeys,NULL,gnorm,gpositions,TOKENS-1u,1u,TOKENS,
            FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,&side,&error)==FG_OK&&
        fg_vk_end(context,&error)==FG_OK&&side==0u&&
        fg_vk_tensor_read(gpu_ids[side],0,ids,sizeof(ids),&error)==FG_OK;
    for(uint32_t i=0;ok&&i<FG_QSA_TOPK_CANDIDATES;i++)if(ids[i]!=i)ok=0;
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(gpu_ids[i]);fg_vk_tensor_destroy(scores[i]);}
    fg_vk_tensor_destroy(gnorm);fg_vk_tensor_destroy(gquery);
    fg_vk_tensor_destroy(gpositions);fg_vk_tensor_destroy(gkeys);
    free(positions);free(keys);return ok;
}

static float qsa_cpu_block_score(const float query[FG_Q38_INDEX_QUERY_WIDTH],
                                 const uint8_t *quantized,
                                 const float norm[FG_Q38_INDEX_WIDTH],
                                 const uint32_t position[3]){
    float key[FG_Q38_INDEX_WIDTH]={0},token[FG_Q38_INDEX_WIDTH];
    for(uint32_t inside=0;inside<FG_Q38_QSA_COMPRESS_RATIO;inside++){
        fg_dequantize_q8_0(quantized+(uint64_t)inside*
                           FG_Q38_QSA_INDEX_KEY_BYTES,token,FG_Q38_INDEX_WIDTH);
        for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)key[i]+=token[i]*0.25f;
    }
    fg_error local={0};
    if(fg_q38_rms_mrope(key,1u,FG_Q38_INDEX_WIDTH,norm,position,&local)!=FG_OK)
        return -INFINITY;
    float score=0.0f;
    for(uint32_t head=0;head<FG_Q38_INDEX_HEADS;head++){
        float dot=0.0f;
        for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)
            dot=fmaf(query[head*FG_Q38_INDEX_WIDTH+i],key[i],dot);
        score+=fmaxf(dot,0.0f);
    }
    return score/sqrtf((float)FG_Q38_INDEX_WIDTH);
}

static int test_qsa_resident_hierarchical_topk(void){
    static const uint32_t block_counts[]={513u,4097u,32769u,65536u};
    const uint32_t max_blocks=block_counts[3],tokens=max_blocks*4u;
    uint8_t *keys=malloc((uint64_t)tokens*FG_Q38_QSA_INDEX_KEY_BYTES);
    uint32_t *positions=calloc((uint64_t)tokens*3u,sizeof(*positions));
    score_id *all_scores=malloc((size_t)max_blocks*sizeof(*all_scores));
    score_id *reference=malloc((size_t)max_blocks*sizeof(*reference));
    float projected[FG_Q38_INDEX_QUERY_WIDTH]={0},query[FG_Q38_INDEX_QUERY_WIDTH];
    float qnorm[FG_Q38_INDEX_WIDTH],knorm[FG_Q38_INDEX_WIDTH],raw[FG_Q38_INDEX_WIDTH];
    for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)qnorm[i]=knorm[i]=1.0f;
    for(uint32_t head=0;head<FG_Q38_INDEX_HEADS;head++)
        projected[head*FG_Q38_INDEX_WIDTH]=1.0f;
    memcpy(query,projected,sizeof(query));uint32_t zero_position[3]={0};
    int ok=keys&&positions&&all_scores&&reference&&
        fg_q38_rms_mrope(query,FG_Q38_INDEX_HEADS,FG_Q38_INDEX_WIDTH,qnorm,
                         zero_position,&error)==FG_OK;
    for(uint32_t block=0;ok&&block<max_blocks;block++){
        memset(raw,0,sizeof(raw));
        if(block<1024u){
            raw[0]=1.0f;
            raw[1]=(float)(block/2u)*0.001f;
        }else raw[0]=-1.0f;
        for(uint32_t inside=0;inside<4u;inside++)
            fg_quantize_q8_0(raw,keys+((uint64_t)block*4u+inside)*
                             FG_Q38_QSA_INDEX_KEY_BYTES,FG_Q38_INDEX_WIDTH);
        all_scores[block]=(score_id){
            qsa_cpu_block_score(query,keys+(uint64_t)block*4u*
                                FG_Q38_QSA_INDEX_KEY_BYTES,knorm,zero_position),
            block
        };
    }
    uint64_t first_keys=(uint64_t)FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY*
        FG_Q38_QSA_INDEX_KEY_BYTES;
    fg_vk_tensor *gkeys_0=ok?tensor(keys,first_keys):NULL;
    fg_vk_tensor *gkeys_1=ok?tensor(keys+first_keys,
        (uint64_t)(tokens-FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY)*
        FG_Q38_QSA_INDEX_KEY_BYTES):NULL;
    fg_vk_tensor *gpositions=ok?tensor(positions,(uint64_t)tokens*3u*4u):NULL;
    fg_vk_tensor *gquery=ok?tensor(query,sizeof(query)):NULL;
    fg_vk_tensor *gnorm=ok?tensor(knorm,sizeof(knorm)):NULL;
    uint64_t candidates=(uint64_t)fg_qsa_resident_candidate_groups(tokens)*
        FG_QSA_TOPK_CANDIDATES;
    fg_vk_tensor *scores[2]={ok?tensor(NULL,candidates*4u):NULL,
                             ok?tensor(NULL,candidates*4u):NULL};
    fg_vk_tensor *ids[2]={ok?tensor(NULL,candidates*4u):NULL,
                          ok?tensor(NULL,candidates*4u):NULL};
    ok=ok&&gkeys_0&&gkeys_1&&gpositions&&gquery&&gnorm&&scores[0]&&scores[1]&&
        ids[0]&&ids[1];
    uint32_t got[FG_QSA_TOPK_CANDIDATES];
    for(uint32_t test=0;ok&&test<sizeof(block_counts)/sizeof(block_counts[0]);test++){
        uint32_t blocks=block_counts[test],side=UINT32_MAX;
        memcpy(reference,all_scores,(size_t)blocks*sizeof(*reference));
        qsort(reference,blocks,sizeof(*reference),score_id_compare);
        ok=fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_qsa_resident_select(context,scores[0],ids[0],scores[1],ids[1],
                gquery,gkeys_0,gkeys_1,gnorm,gpositions,blocks*4u-1u,1u,tokens,
                FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,&side,&error)==FG_OK&&
            fg_vk_end(context,&error)==FG_OK&&
            fg_vk_tensor_read(ids[side],0,got,sizeof(got),&error)==FG_OK;
        for(uint32_t i=0;ok&&i<FG_QSA_TOPK_CANDIDATES;i++)
            if(got[i]!=reference[i].id){
                fprintf(stderr,"resident top-k %u blocks rank %u GPU=%u CPU=%u\n",
                        blocks,i,got[i],reference[i].id);
                ok=0;
            }
    }
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(ids[i]);fg_vk_tensor_destroy(scores[i]);}
    fg_vk_tensor_destroy(gnorm);fg_vk_tensor_destroy(gquery);
    fg_vk_tensor_destroy(gpositions);fg_vk_tensor_destroy(gkeys_1);
    fg_vk_tensor_destroy(gkeys_0);free(reference);free(all_scores);
    free(positions);free(keys);return ok;
}

static int test_qsa_resident_batch_attention(void){
    enum{TOKENS=9,VALUES=TOKENS*6144};
    uint8_t records[TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES];
    uint8_t index_keys[TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES];
    float query[VALUES],gate[VALUES],batch[VALUES],repeated[VALUES];
    uint32_t positions[TOKENS*3u];
    for(uint32_t token=0;token<TOKENS;token++){
        qsa_constant_record(records+(uint64_t)token*FG_Q38_QSA_TOKEN_RECORD_BYTES,
                            (float)(token+1u));
        memcpy(index_keys+(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES,
               records+(uint64_t)token*FG_Q38_QSA_TOKEN_RECORD_BYTES+
               FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,
               FG_Q38_QSA_INDEX_KEY_BYTES);
        positions[token*3u]=positions[token*3u+1u]=positions[token*3u+2u]=token;
    }
    memset(query,0,sizeof(query));
    for(uint32_t i=0;i<VALUES;i++)gate[i]=20.0f;
    fg_vk_tensor *grecords=tensor(records,sizeof(records));
    fg_vk_tensor *gindex=tensor(index_keys,sizeof(index_keys));
    fg_vk_tensor *gpositions=tensor(positions,sizeof(positions));
    fg_vk_tensor *gquery=tensor(query,sizeof(query)),*ggate=tensor(gate,sizeof(gate));
    fg_vk_tensor *gbatch=tensor(NULL,sizeof(batch)),*grepeated=tensor(NULL,sizeof(repeated));
    fg_vk_tensor *scores[2]={tensor(NULL,(uint64_t)TOKENS*512u*4u),
                             tensor(NULL,(uint64_t)TOKENS*512u*4u)};
    fg_vk_tensor *ids[2]={tensor(NULL,(uint64_t)TOKENS*512u*4u),
                          tensor(NULL,(uint64_t)TOKENS*512u*4u)};
    float norm[FG_Q38_INDEX_WIDTH]={0};fg_vk_tensor *gnorm=tensor(norm,sizeof(norm));
    uint32_t side=UINT32_MAX;
    int ok=grecords&&gindex&&gpositions&&gquery&&ggate&&gbatch&&grepeated&&
        scores[0]&&scores[1]&&ids[0]&&ids[1]&&gnorm&&
        fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_qsa_resident_select(context,scores[0],ids[0],scores[1],ids[1],
            gquery,gindex,NULL,gnorm,gpositions,0u,TOKENS,TOKENS,
            FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,&side,&error)==FG_OK&&
        fg_vk_qsa_resident_attention(context,gbatch,grecords,NULL,ids[side],
            gquery,ggate,0u,TOKENS,TOKENS,FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,
            FG_QSA_TOPK_CANDIDATES,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK;
    for(uint32_t token=0;ok&&token<TOKENS;token++){
        fg_vk_tensor *record_view=NULL,*query_view=NULL,*gate_view=NULL,*output_view=NULL;
        ok=fg_vk_tensor_view(grecords,0,(uint64_t)(token+1u)*
                             FG_Q38_QSA_TOKEN_RECORD_BYTES,&record_view,&error)==FG_OK&&
            fg_vk_tensor_view(gquery,(uint64_t)token*6144u*4u,6144u*4u,
                              &query_view,&error)==FG_OK&&
            fg_vk_tensor_view(ggate,(uint64_t)token*6144u*4u,6144u*4u,
                              &gate_view,&error)==FG_OK&&
            fg_vk_tensor_view(grepeated,(uint64_t)token*6144u*4u,6144u*4u,
                              &output_view,&error)==FG_OK&&
            fg_vk_qsa_attention(context,output_view,record_view,query_view,gate_view,
                                token+1u,&error)==FG_OK;
        fg_vk_tensor_destroy(output_view);fg_vk_tensor_destroy(gate_view);
        fg_vk_tensor_destroy(query_view);fg_vk_tensor_destroy(record_view);
    }
    if(ok)ok=fg_vk_tensor_read(gbatch,0,batch,sizeof(batch),&error)==FG_OK&&
        fg_vk_tensor_read(grepeated,0,repeated,sizeof(repeated),&error)==FG_OK;
    for(uint32_t token=0;ok&&token<TOKENS;token++)
        for(uint32_t i=0;i<6144u;i++){
            uint64_t at=(uint64_t)token*6144u+i;
            float expected=(float)(token+2u)*0.5f/(1.0f+expf(-20.0f));
            if(fabsf(batch[at]-repeated[at])>1e-5f||
               fabsf(batch[at]-expected)>2e-3f){ok=0;break;}
        }
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(gnorm);
    for(uint32_t i=0;i<2u;i++){fg_vk_tensor_destroy(ids[i]);fg_vk_tensor_destroy(scores[i]);}
    fg_vk_tensor_destroy(grepeated);fg_vk_tensor_destroy(gbatch);
    fg_vk_tensor_destroy(ggate);fg_vk_tensor_destroy(gquery);
    fg_vk_tensor_destroy(gpositions);fg_vk_tensor_destroy(gindex);
    fg_vk_tensor_destroy(grecords);return ok;
}

static int test_qsa_resident_t1_compat(void){
    uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES];
    qsa_constant_record(record,3.25f);
    uint32_t selected[FG_QSA_TOPK_CANDIDATES];
    float query[6144]={0},gate[6144],resident[6144],decode[6144];
    for(uint32_t i=0;i<FG_QSA_TOPK_CANDIDATES;i++)selected[i]=UINT32_MAX;
    for(uint32_t i=0;i<6144u;i++)gate[i]=(float)((int32_t)(i%7u)-3)*0.25f;
    fg_vk_tensor *grecord=tensor(record,sizeof(record));
    fg_vk_tensor *gselected=tensor(selected,sizeof(selected));
    fg_vk_tensor *gquery=tensor(query,sizeof(query)),*ggate=tensor(gate,sizeof(gate));
    fg_vk_tensor *gresident=tensor(NULL,sizeof(resident)),*gdecode=tensor(NULL,sizeof(decode));
    int ok=grecord&&gselected&&gquery&&ggate&&gresident&&gdecode&&
        fg_vk_qsa_resident_attention(context,gresident,grecord,NULL,gselected,
            gquery,ggate,0u,1u,1u,FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY,
            FG_QSA_TOPK_CANDIDATES,&error)==FG_OK&&
        fg_vk_qsa_attention(context,gdecode,grecord,gquery,ggate,1u,&error)==FG_OK&&
        fg_vk_tensor_read(gresident,0,resident,sizeof(resident),&error)==FG_OK&&
        fg_vk_tensor_read(gdecode,0,decode,sizeof(decode),&error)==FG_OK;
    for(uint32_t i=0;ok&&i<6144u;i++)
        if(fabsf(resident[i]-decode[i])>1e-6f)ok=0;
    fg_vk_tensor_destroy(gdecode);fg_vk_tensor_destroy(gresident);
    fg_vk_tensor_destroy(ggate);fg_vk_tensor_destroy(gquery);
    fg_vk_tensor_destroy(gselected);fg_vk_tensor_destroy(grecord);
    return ok;
}

static int test_qsa_record_gather(void){
    enum{CAPACITY=10,BLOCKS=2,TAIL=2,SELECTED=BLOCKS*4+TAIL};uint8_t *records=malloc((size_t)CAPACITY*FG_Q38_QSA_TOKEN_RECORD_BYTES),*got=malloc((size_t)SELECTED*FG_Q38_QSA_TOKEN_RECORD_BYTES);uint32_t ids[BLOCKS]={1u,0u};if(!records||!got){free(got);free(records);return 0;}for(uint32_t token=0;token<CAPACITY;token++)memset(records+(uint64_t)token*FG_Q38_QSA_TOKEN_RECORD_BYTES,(int)(token+1u),FG_Q38_QSA_TOKEN_RECORD_BYTES);fg_vk_tensor *source=tensor(records,(uint64_t)CAPACITY*FG_Q38_QSA_TOKEN_RECORD_BYTES),*block_ids=tensor(ids,sizeof(ids)),*output=tensor(NULL,(uint64_t)SELECTED*FG_Q38_QSA_TOKEN_RECORD_BYTES);int ok=source&&block_ids&&output&&fg_vk_qsa_record_gather(context,output,source,block_ids,0u,CAPACITY,BLOCKS,8u,TAIL,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,(uint64_t)SELECTED*FG_Q38_QSA_TOKEN_RECORD_BYTES,&error)==FG_OK;const uint32_t expected[SELECTED]={4u,5u,6u,7u,0u,1u,2u,3u,8u,9u};for(uint32_t i=0;ok&&i<SELECTED;i++)for(uint32_t byte=0;byte<FG_Q38_QSA_TOKEN_RECORD_BYTES;byte++)if(got[(uint64_t)i*FG_Q38_QSA_TOKEN_RECORD_BYTES+byte]!=(uint8_t)(expected[i]+1u)){ok=0;break;}fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(block_ids);fg_vk_tensor_destroy(source);free(got);free(records);return ok;
}

static int score_id_compare(const void *a,const void *b){const score_id *x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;return x->id<y->id?-1:x->id>y->id;}
static int test_qsa_indexer(void){
    enum{TOKENS=13,BLOCKS=3};float projected[512],query[512],qnorm[128],knorm[128],raw[TOKENS*128],dequant[TOKENS*128];uint8_t quant[TOKENS*136];uint32_t positions[TOKENS*3];for(uint32_t i=0;i<512u;i++)projected[i]=sinf((float)(i+3u)*0.019f)+0.2f*cosf((float)i*0.007f);memcpy(query,projected,sizeof(query));for(uint32_t i=0;i<128u;i++){qnorm[i]=0.03f*sinf((float)i*0.051f);knorm[i]=0.04f*cosf((float)i*0.037f);}for(uint32_t token_id=0;token_id<TOKENS;token_id++){for(uint32_t axis=0;axis<3u;axis++)positions[axis*TOKENS+token_id]=token_id;for(uint32_t i=0;i<128u;i++)raw[token_id*128u+i]=sinf((float)(token_id+1u)*(float)(i+1u)*0.0017f)+0.15f*cosf((float)(token_id+2u)*(float)(i+3u)*0.0009f);fg_quantize_q8_0(raw+token_id*128u,quant+token_id*136u,128u);fg_dequantize_q8_0(quant+token_id*136u,dequant+token_id*128u,128u);}uint32_t current[3]={TOKENS-1,TOKENS-1,TOKENS-1};int ok=fg_q38_rms_mrope(query,4u,128u,qnorm,current,&error)==FG_OK;uint32_t expected_tokens[TOKENS],expected_count=0;ok=ok&&fg_q38_qsa_index_select_reference(projected,dequant,TOKENS,qnorm,knorm,expected_tokens,TOKENS,&expected_count,&error)==FG_OK;
    uint32_t interleaved_positions[TOKENS*3];for(uint32_t token_id=0;token_id<TOKENS;token_id++)for(uint32_t axis=0;axis<3u;axis++)interleaved_positions[token_id*3u+axis]=positions[axis*TOKENS+token_id];
    fg_vk_tensor *gq=tensor(query,sizeof(query)),*gk=tensor(quant,sizeof(quant)),*gn=tensor(knorm,sizeof(knorm)),*gp=tensor(interleaved_positions,sizeof(interleaved_positions)),*gs=tensor(NULL,BLOCKS*4u),*gi=tensor(NULL,BLOCKS*4u),*gos=tensor(NULL,BLOCKS*4u),*goi=tensor(NULL,BLOCKS*4u);uint32_t produced=0,ids[BLOCKS];ok=ok&&gq&&gk&&gn&&gp&&gs&&gi&&gos&&goi&&fg_vk_qsa_index_score(context,gs,gi,gq,gk,gn,gp,TOKENS,&error)==FG_OK&&fg_vk_topk_reduce(context,gos,goi,gs,gi,BLOCKS,&produced,&error)==FG_OK&&produced==BLOCKS&&fg_vk_tensor_read(goi,0,ids,sizeof(ids),&error)==FG_OK;for(uint32_t i=0;ok&&i<BLOCKS;i++)if(ids[i]!=expected_tokens[i*4u]/4u)ok=0;fg_vk_tensor_destroy(goi);fg_vk_tensor_destroy(gos);fg_vk_tensor_destroy(gi);fg_vk_tensor_destroy(gs);fg_vk_tensor_destroy(gp);fg_vk_tensor_destroy(gn);fg_vk_tensor_destroy(gk);fg_vk_tensor_destroy(gq);
    enum{COUNT=5000};score_id *reference=malloc(COUNT*sizeof(*reference));float *scores=malloc(COUNT*4u);uint32_t *input_ids=malloc(COUNT*4u),top_ids[512];if(!reference||!scores||!input_ids)ok=0;for(uint32_t i=0;ok&&i<COUNT;i++){scores[i]=sinf((float)i*0.013f)+(float)i*1e-7f;input_ids[i]=i;reference[i]=(score_id){scores[i],i};}if(ok)qsort(reference,COUNT,sizeof(*reference),score_id_compare);fg_vk_tensor *a_s=ok?tensor(scores,COUNT*4u):NULL,*a_i=ok?tensor(input_ids,COUNT*4u):NULL,*b_s=ok?tensor(NULL,1024u*4u):NULL,*b_i=ok?tensor(NULL,1024u*4u):NULL,*c_s=ok?tensor(NULL,512u*4u):NULL,*c_i=ok?tensor(NULL,512u*4u):NULL;uint32_t mid=0,final=0;ok=ok&&a_s&&a_i&&b_s&&b_i&&c_s&&c_i&&fg_vk_topk_reduce(context,b_s,b_i,a_s,a_i,COUNT,&mid,&error)==FG_OK&&mid==1024u&&fg_vk_topk_reduce(context,c_s,c_i,b_s,b_i,mid,&final,&error)==FG_OK&&final==512u&&fg_vk_tensor_read(c_i,0,top_ids,sizeof(top_ids),&error)==FG_OK;for(uint32_t i=0;ok&&i<512u;i++)if(top_ids[i]!=reference[i].id){fprintf(stderr,"top-k mismatch %u GPU=%u CPU=%u\n",i,top_ids[i],reference[i].id);ok=0;}fg_vk_tensor_destroy(c_i);fg_vk_tensor_destroy(c_s);fg_vk_tensor_destroy(b_i);fg_vk_tensor_destroy(b_s);fg_vk_tensor_destroy(a_i);fg_vk_tensor_destroy(a_s);free(input_ids);free(scores);free(reference);return ok;
}

static int test_qsa_segmented_index_score(void){
    enum{TOKENS=8,SEGMENT_TOKENS=4,BLOCKS=2};
    float query[FG_Q38_INDEX_QUERY_WIDTH],raw_keys[TOKENS*FG_Q38_INDEX_WIDTH],
        norm[FG_Q38_INDEX_WIDTH];
    uint8_t keys[TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES];
    uint32_t positions[TOKENS*3u];
    for(uint32_t i=0;i<FG_Q38_INDEX_QUERY_WIDTH;i++)
        query[i]=sinf((float)(i+7u)*0.013f);
    for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)
        norm[i]=cosf((float)(i+3u)*0.019f);
    for(uint32_t token=0;token<TOKENS;token++){
        positions[token*3u]=token+11u;
        positions[token*3u+1u]=token*2u+5u;
        positions[token*3u+2u]=token*3u+9u;
        for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)
            raw_keys[token*FG_Q38_INDEX_WIDTH+i]=
                cosf((float)(token+1u)*(float)(i+5u)*0.007f);
        fg_quantize_q8_0(raw_keys+token*FG_Q38_INDEX_WIDTH,
                         keys+(uint64_t)token*FG_Q38_QSA_INDEX_KEY_BYTES,
                         FG_Q38_INDEX_WIDTH);
    }
    fg_vk_tensor *gq=tensor(query,sizeof(query)),*gkeys=tensor(keys,sizeof(keys)),
        *gnorm=tensor(norm,sizeof(norm)),*gpositions=tensor(positions,sizeof(positions)),
        *reference_scores=tensor(NULL,BLOCKS*4u),
        *reference_ids=tensor(NULL,BLOCKS*4u),*segmented_scores=tensor(NULL,BLOCKS*4u),
        *segmented_ids=tensor(NULL,BLOCKS*4u);
    int ok=gq&&gkeys&&gnorm&&gpositions&&reference_scores&&reference_ids&&
        segmented_scores&&segmented_ids;
    if(ok)ok=fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_qsa_index_score(context,reference_scores,reference_ids,gq,gkeys,
                              gnorm,gpositions,TOKENS,&error)==FG_OK&&
        fg_vk_end(context,&error)==FG_OK;
    for(uint32_t segment=0;ok&&segment<2u;segment++){
        fg_vk_tensor *key_view=NULL,*position_view=NULL,*score_view=NULL,
            *id_view=NULL;
        ok=fg_vk_tensor_view(gkeys,(uint64_t)segment*SEGMENT_TOKENS*
                             FG_Q38_QSA_INDEX_KEY_BYTES,
                             SEGMENT_TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES,
                             &key_view,&error)==FG_OK&&
            fg_vk_tensor_view(gpositions,(uint64_t)segment*SEGMENT_TOKENS*12u,
                              SEGMENT_TOKENS*12u,&position_view,&error)==FG_OK&&
            fg_vk_tensor_view(segmented_scores,(uint64_t)segment*4u,4u,
                              &score_view,&error)==FG_OK&&
            fg_vk_tensor_view(segmented_ids,(uint64_t)segment*4u,4u,
                              &id_view,&error)==FG_OK;
        if(ok)ok=fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_qsa_index_score_segment(context,score_view,id_view,gq,key_view,
                gnorm,position_view,SEGMENT_TOKENS,segment,&error)==FG_OK&&
            fg_vk_end(context,&error)==FG_OK;
        fg_vk_tensor_destroy(id_view);fg_vk_tensor_destroy(score_view);
        fg_vk_tensor_destroy(position_view);fg_vk_tensor_destroy(key_view);
    }
    float expected_scores[BLOCKS],got_scores[BLOCKS];
    uint32_t expected_ids[BLOCKS],got_ids[BLOCKS];
    if(ok)ok=fg_vk_tensor_read(reference_scores,0,expected_scores,
                               sizeof(expected_scores),&error)==FG_OK&&
        fg_vk_tensor_read(reference_ids,0,expected_ids,sizeof(expected_ids),
                          &error)==FG_OK&&
        fg_vk_tensor_read(segmented_scores,0,got_scores,sizeof(got_scores),
                          &error)==FG_OK&&
        fg_vk_tensor_read(segmented_ids,0,got_ids,sizeof(got_ids),&error)==FG_OK;
    for(uint32_t i=0;ok&&i<BLOCKS;i++)
        if(expected_scores[i]!=got_scores[i]||expected_ids[i]!=got_ids[i])ok=0;
    fg_vk_tensor_destroy(segmented_ids);fg_vk_tensor_destroy(segmented_scores);
    fg_vk_tensor_destroy(reference_ids);fg_vk_tensor_destroy(reference_scores);
    fg_vk_tensor_destroy(gpositions);fg_vk_tensor_destroy(gnorm);
    fg_vk_tensor_destroy(gkeys);fg_vk_tensor_destroy(gq);
    return ok;
}

static uint64_t vk_test_monotonic_ns(void){
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC,&value);
    return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;
}

static int test_qsa_prefill_chunk_liveness(void){
    enum{MAX_TOKENS=416,Q=6144};
    const uint32_t totals[]={127u,128u,129u,256u,416u};
    const size_t key_bytes=(size_t)MAX_TOKENS*FG_Q38_QSA_KEY_BYTES;
    const size_t value_bytes=(size_t)MAX_TOKENS*FG_Q38_QSA_VALUE_BYTES;
    const size_t index_bytes=(size_t)MAX_TOKENS*FG_Q38_QSA_INDEX_KEY_BYTES;
    const size_t position_bytes=(size_t)MAX_TOKENS*FG_Q38_QSA_POSITION_BYTES;
    const size_t record_bytes=(size_t)MAX_TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES;
    uint8_t *keys=malloc(key_bytes),*values=malloc(value_bytes),
        *index_keys=malloc(index_bytes);
    uint32_t *positions=malloc(position_bytes);
    float *query=malloc((size_t)Q*4u),*gate=malloc((size_t)Q*4u);
    if(!keys||!values||!index_keys||!positions||!query||!gate){
        free(gate);free(query);free(positions);free(index_keys);free(values);free(keys);
        return 0;
    }
    for(uint32_t token=0u;token<MAX_TOKENS;token++){
        for(uint32_t i=0u;i<FG_Q38_QSA_KEY_BYTES;i++)
            keys[(size_t)token*FG_Q38_QSA_KEY_BYTES+i]=(uint8_t)(token*3u+i*7u);
        for(uint32_t i=0u;i<FG_Q38_QSA_VALUE_BYTES;i++)
            values[(size_t)token*FG_Q38_QSA_VALUE_BYTES+i]=(uint8_t)(token*5u+i*11u);
        for(uint32_t i=0u;i<FG_Q38_QSA_INDEX_KEY_BYTES;i++)
            index_keys[(size_t)token*FG_Q38_QSA_INDEX_KEY_BYTES+i]=(uint8_t)(token*13u+i*17u);
        positions[(size_t)token*3u]=token;
        positions[(size_t)token*3u+1u]=token;
        positions[(size_t)token*3u+2u]=token;
    }
    for(uint32_t i=0u;i<Q;i++){
        query[i]=sinf((float)(i+3u)*0.0017f);
        gate[i]=cosf((float)(i+5u)*0.0023f);
    }
    fg_vk_tensor *key=tensor(keys,key_bytes),*value=tensor(values,value_bytes),
        *index=tensor(index_keys,index_bytes),*position=tensor(positions,position_bytes),
        *serial_records=tensor(NULL,record_bytes),*chunk_records=tensor(NULL,record_bytes),
        *serial_index=tensor(NULL,index_bytes),*chunk_index=tensor(NULL,index_bytes),
        *query_tensor=tensor(query,(size_t)Q*4u),*gate_tensor=tensor(gate,(size_t)Q*4u),
        *serial_output=tensor(NULL,(size_t)Q*4u),*chunk_output=tensor(NULL,(size_t)Q*4u);
    fg_vk_tensor *key_view=NULL,*value_view=NULL,*index_view=NULL,*position_view=NULL;
    int ok=key&&value&&index&&position&&serial_records&&chunk_records&&serial_index&&
        chunk_index&&query_tensor&&gate_tensor&&serial_output&&chunk_output;
    if(ok)ok=fg_vk_tensor_view(key,0,FG_Q38_QSA_KEY_BYTES,&key_view,&error)==FG_OK&&
        fg_vk_tensor_view(value,0,FG_Q38_QSA_VALUE_BYTES,&value_view,&error)==FG_OK&&
        fg_vk_tensor_view(index,0,FG_Q38_QSA_INDEX_KEY_BYTES,&index_view,&error)==FG_OK&&
        fg_vk_tensor_view(position,0,FG_Q38_QSA_POSITION_BYTES,&position_view,&error)==FG_OK;
    uint64_t deadline=vk_test_monotonic_ns()+UINT64_C(30000000000);
    for(uint32_t shape=0u;ok&&shape<sizeof(totals)/sizeof(totals[0]);shape++){
        uint32_t total=totals[shape];
        memset(fg_vk_tensor_map(serial_records),0,(size_t)total*
               FG_Q38_QSA_TOKEN_RECORD_BYTES);
        memset(fg_vk_tensor_map(chunk_records),0,(size_t)total*
               FG_Q38_QSA_TOKEN_RECORD_BYTES);
        memset(fg_vk_tensor_map(serial_index),0,(size_t)total*
               FG_Q38_QSA_INDEX_KEY_BYTES);
        memset(fg_vk_tensor_map(chunk_index),0,(size_t)total*
               FG_Q38_QSA_INDEX_KEY_BYTES);
        ok=fg_vk_begin(context,&error)==FG_OK;
        for(uint32_t token=0u;ok&&token<total;token++){
            ok=fg_vk_tensor_view_rebind(key_view,key,(uint64_t)token*
                                        FG_Q38_QSA_KEY_BYTES,
                                        FG_Q38_QSA_KEY_BYTES,&error)==FG_OK&&
               fg_vk_tensor_view_rebind(value_view,value,(uint64_t)token*
                                        FG_Q38_QSA_VALUE_BYTES,
                                        FG_Q38_QSA_VALUE_BYTES,&error)==FG_OK&&
               fg_vk_tensor_view_rebind(index_view,index,(uint64_t)token*
                                        FG_Q38_QSA_INDEX_KEY_BYTES,
                                        FG_Q38_QSA_INDEX_KEY_BYTES,&error)==FG_OK&&
               fg_vk_tensor_view_rebind(position_view,position,(uint64_t)token*
                                        FG_Q38_QSA_POSITION_BYTES,
                                        FG_Q38_QSA_POSITION_BYTES,&error)==FG_OK&&
               fg_vk_qsa_record_commit(context,serial_records,serial_index,key_view,
                                       value_view,index_view,position_view,0u,token,
                                       total,&error)==FG_OK;
        }
        if(ok)ok=fg_vk_end(context,&error)==FG_OK;
        else if(fg_vk_batch_active(context)){
            fg_error ignored={0};
            fg_vk_abort(context,&ignored);
        }
        for(uint32_t first=0u;ok&&first<total;first+=128u){
            uint32_t count=total-first;
            if(count>128u)count=128u;
            ok=fg_vk_begin(context,&error)==FG_OK;
            for(uint32_t token=first;ok&&token<first+count;token++){
                ok=fg_vk_tensor_view_rebind(key_view,key,(uint64_t)token*
                                            FG_Q38_QSA_KEY_BYTES,
                                            FG_Q38_QSA_KEY_BYTES,&error)==FG_OK&&
                   fg_vk_tensor_view_rebind(value_view,value,(uint64_t)token*
                                            FG_Q38_QSA_VALUE_BYTES,
                                            FG_Q38_QSA_VALUE_BYTES,&error)==FG_OK&&
                   fg_vk_tensor_view_rebind(index_view,index,(uint64_t)token*
                                            FG_Q38_QSA_INDEX_KEY_BYTES,
                                            FG_Q38_QSA_INDEX_KEY_BYTES,&error)==FG_OK&&
                   fg_vk_tensor_view_rebind(position_view,position,(uint64_t)token*
                                            FG_Q38_QSA_POSITION_BYTES,
                                            FG_Q38_QSA_POSITION_BYTES,&error)==FG_OK&&
                   fg_vk_qsa_record_commit(context,chunk_records,chunk_index,key_view,
                                           value_view,index_view,position_view,0u,token,
                                           total,&error)==FG_OK;
            }
            if(ok)ok=fg_vk_end(context,&error)==FG_OK;
            else if(fg_vk_batch_active(context)){
                fg_error ignored={0};
                fg_vk_abort(context,&ignored);
            }
        }
        if(ok)ok=memcmp(fg_vk_tensor_const_map(serial_records),
                       fg_vk_tensor_const_map(chunk_records),(size_t)total*
                       FG_Q38_QSA_TOKEN_RECORD_BYTES)==0&&
               memcmp(fg_vk_tensor_const_map(serial_index),
                      fg_vk_tensor_const_map(chunk_index),(size_t)total*
                      FG_Q38_QSA_INDEX_KEY_BYTES)==0;
        if(ok)ok=fg_vk_qsa_attention(context,serial_output,serial_records,
                                     query_tensor,gate_tensor,total,&error)==FG_OK&&
               fg_vk_qsa_attention(context,chunk_output,chunk_records,
                                   query_tensor,gate_tensor,total,&error)==FG_OK&&
               memcmp(fg_vk_tensor_const_map(serial_output),
                      fg_vk_tensor_const_map(chunk_output),(size_t)Q*4u)==0;
        if(ok)ok=vk_test_monotonic_ns()<=deadline;
    }
    fg_vk_tensor_destroy(position_view);fg_vk_tensor_destroy(index_view);
    fg_vk_tensor_destroy(value_view);fg_vk_tensor_destroy(key_view);
    fg_vk_tensor_destroy(chunk_output);fg_vk_tensor_destroy(serial_output);
    fg_vk_tensor_destroy(gate_tensor);fg_vk_tensor_destroy(query_tensor);
    fg_vk_tensor_destroy(chunk_index);fg_vk_tensor_destroy(serial_index);
    fg_vk_tensor_destroy(chunk_records);fg_vk_tensor_destroy(serial_records);
    fg_vk_tensor_destroy(position);fg_vk_tensor_destroy(index);
    fg_vk_tensor_destroy(value);fg_vk_tensor_destroy(key);
    free(gate);free(query);free(positions);free(index_keys);free(values);free(keys);
    return ok;
}

static int test_output_topk(void){
    enum{COUNT=FG_Q38_VOCAB_SIZE,FIRST=(COUNT+4095)/4096*512};float *scores=malloc((size_t)COUNT*4u),top_scores[4];uint32_t *ids=malloc((size_t)COUNT*4u),top_ids[4];if(!scores||!ids){free(ids);free(scores);return 0;}for(uint32_t i=0;i<COUNT;i++){scores[i]=-(float)i;ids[i]=i;}scores[12345]=100.0f;scores[23456]=99.0f;scores[34567]=98.0f;scores[45678]=NAN;fg_vk_tensor *input_scores=tensor(scores,(uint64_t)COUNT*4u),*input_ids=tensor(ids,(uint64_t)COUNT*4u),*scratch_scores[2]={tensor(NULL,(uint64_t)FIRST*4u),tensor(NULL,(uint64_t)FIRST*4u)},*scratch_ids[2]={tensor(NULL,(uint64_t)FIRST*4u),tensor(NULL,(uint64_t)FIRST*4u)};uint32_t count=COUNT,slot=0;int ok=input_scores&&input_ids&&scratch_scores[0]&&scratch_scores[1]&&scratch_ids[0]&&scratch_ids[1]&&fg_vk_begin(context,&error)==FG_OK;const fg_vk_tensor *source_scores=input_scores,*source_ids=input_ids;while(ok&&count>512u){uint32_t next=0;ok=fg_vk_topk_reduce(context,scratch_scores[slot],scratch_ids[slot],source_scores,source_ids,count,&next,&error)==FG_OK;source_scores=scratch_scores[slot];source_ids=scratch_ids[slot];count=next;slot^=1u;}if(ok)ok=fg_vk_end(context,&error)==FG_OK&&count==512u&&fg_vk_tensor_read(source_scores,0,top_scores,sizeof(top_scores),&error)==FG_OK&&fg_vk_tensor_read(source_ids,0,top_ids,sizeof(top_ids),&error)==FG_OK&&top_ids[0]==45678u&&!isfinite(top_scores[0])&&top_ids[1]==12345u&&top_ids[2]==23456u&&top_ids[3]==34567u&&top_scores[1]==100.0f&&top_scores[2]==99.0f&&top_scores[3]==98.0f;else if(fg_vk_batch_active(context))fg_vk_end(context,&error);fg_vk_tensor_destroy(scratch_ids[1]);fg_vk_tensor_destroy(scratch_ids[0]);fg_vk_tensor_destroy(scratch_scores[1]);fg_vk_tensor_destroy(scratch_scores[0]);fg_vk_tensor_destroy(input_ids);fg_vk_tensor_destroy(input_scores);free(ids);free(scores);return ok;
}

static int test_generation_topk_selector(void){
    enum{COUNT=19,K=4};
    float scores[COUNT]={1.0f,9.0f,7.0f,9.0f,3.0f,8.0f,2.0f,7.0f,6.0f,5.0f,
                        4.0f,0.0f,-1.0f,-2.0f,-3.0f,-4.0f,-5.0f,-6.0f,NAN};
    uint32_t ids[COUNT],got_ids[K];float got_scores[K];
    for(uint32_t i=0;i<COUNT;i++)ids[i]=i;
    fg_vk_tensor *input_scores=tensor(scores,sizeof(scores)),
        *input_ids=tensor(ids,sizeof(ids)),*output_scores=tensor(NULL,sizeof(got_scores)),
        *output_ids=tensor(NULL,sizeof(got_ids));
    uint32_t produced=0;int ok=input_scores&&input_ids&&output_scores&&output_ids&&
        fg_vk_topk_select(context,output_scores,output_ids,input_scores,input_ids,
                          COUNT,0,&produced,&error)==FG_ERR_ARGUMENT&&
        fg_vk_topk_select(context,output_scores,output_ids,input_scores,input_ids,
                          COUNT,65,&produced,&error)==FG_ERR_ARGUMENT&&
        fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_topk_select(context,output_scores,output_ids,input_scores,input_ids,
                          COUNT,K,&produced,&error)==FG_OK&&produced==K&&
        fg_vk_end(context,&error)==FG_OK&&
        fg_vk_tensor_read(output_scores,0,got_scores,sizeof(got_scores),&error)==FG_OK&&
        fg_vk_tensor_read(output_ids,0,got_ids,sizeof(got_ids),&error)==FG_OK;
    const uint32_t expected_ids[K]={1u,3u,5u,2u};
    bool seen[K]={0};
    for(uint32_t i=0;ok&&i<K;i++){
        uint32_t match=K;
        for(uint32_t j=0u;j<K;j++)if(got_ids[i]==expected_ids[j])match=j;
        if(match==K||seen[match]||got_scores[i]!=scores[got_ids[i]])ok=0;
        else seen[match]=true;
    }
    uint32_t repeat_ids[K];float repeat_scores[K];
    if(ok)ok=fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_topk_select(context,output_scores,output_ids,input_scores,input_ids,
                          COUNT,K,&produced,&error)==FG_OK&&produced==K&&
        fg_vk_end(context,&error)==FG_OK&&
        fg_vk_tensor_read(output_scores,0,repeat_scores,sizeof(repeat_scores),&error)==FG_OK&&
        fg_vk_tensor_read(output_ids,0,repeat_ids,sizeof(repeat_ids),&error)==FG_OK&&
        memcmp(got_ids,repeat_ids,sizeof(got_ids))==0&&
        memcmp(got_scores,repeat_scores,sizeof(got_scores))==0;
    uint32_t single_id=UINT32_MAX;float single_score=0.0f;
    if(ok)ok=fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_topk_select(context,output_scores,output_ids,input_scores,input_ids,
                          COUNT,1u,&produced,&error)==FG_OK&&produced==1u&&
        fg_vk_end(context,&error)==FG_OK&&
        fg_vk_tensor_read(output_scores,0,&single_score,sizeof(single_score),&error)==FG_OK&&
        fg_vk_tensor_read(output_ids,0,&single_id,sizeof(single_id),&error)==FG_OK&&
        /* IDs 1 and 3 tie at 9.0; the deterministic winner is the smaller ID. */
        single_id==1u&&single_score==9.0f;
    float shortage[COUNT];
    for(uint32_t i=0u;i<COUNT;i++)shortage[i]=NAN;
    shortage[1]=9.0f;shortage[3]=8.0f;
    fg_vk_tensor *short_input=tensor(shortage,sizeof(shortage));
    uint32_t shortage_ids[K];float shortage_scores[K];
    if(ok)ok=short_input&&fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_topk_select(context,output_scores,output_ids,short_input,input_ids,
                          COUNT,K,&produced,&error)==FG_OK&&produced==K&&
        fg_vk_end(context,&error)==FG_OK&&
        fg_vk_tensor_read(output_scores,0,shortage_scores,sizeof(shortage_scores),&error)==FG_OK&&
        fg_vk_tensor_read(output_ids,0,shortage_ids,sizeof(shortage_ids),&error)==FG_OK&&
        shortage_ids[0]!=UINT32_MAX&&shortage_ids[1]!=UINT32_MAX&&
        shortage_ids[2]==UINT32_MAX&&!isfinite(shortage_scores[2]);
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(short_input);
    fg_vk_tensor_destroy(output_ids);fg_vk_tensor_destroy(output_scores);
    fg_vk_tensor_destroy(input_ids);fg_vk_tensor_destroy(input_scores);
    return ok;
}

static int test_output_argmax(void){
    enum{COUNT=FG_Q38_VOCAB_SIZE,GROUPS=(COUNT+4095u)/4096u};float *scores=malloc((size_t)COUNT*4u),got_score=0.0f;uint32_t *ids=malloc((size_t)COUNT*4u),got_id=UINT32_MAX;if(!scores||!ids){free(ids);free(scores);return 0;}for(uint32_t i=0;i<COUNT;i++){scores[i]=-(float)i;ids[i]=i;}scores[12345]=100.0f;scores[23456]=100.0f;fg_vk_tensor *input_scores=tensor(scores,(uint64_t)COUNT*4u),*input_ids=tensor(ids,(uint64_t)COUNT*4u),*scratch_scores[2]={tensor(NULL,GROUPS*4u),tensor(NULL,GROUPS*4u)},*scratch_ids[2]={tensor(NULL,GROUPS*4u),tensor(NULL,GROUPS*4u)};uint32_t count=COUNT,slot=0;int ok=input_scores&&input_ids&&scratch_scores[0]&&scratch_scores[1]&&scratch_ids[0]&&scratch_ids[1]&&fg_vk_begin(context,&error)==FG_OK;const fg_vk_tensor *source_scores=input_scores,*source_ids=input_ids;while(ok&&count>1u){uint32_t next=0;ok=fg_vk_argmax_reduce(context,scratch_scores[slot],scratch_ids[slot],source_scores,source_ids,count,&next,&error)==FG_OK;source_scores=scratch_scores[slot];source_ids=scratch_ids[slot];count=next;slot^=1u;}if(ok)ok=fg_vk_end(context,&error)==FG_OK&&count==1u&&fg_vk_tensor_read(source_scores,0,&got_score,sizeof(got_score),&error)==FG_OK&&fg_vk_tensor_read(source_ids,0,&got_id,sizeof(got_id),&error)==FG_OK&&got_id==12345u&&got_score==100.0f;else if(fg_vk_batch_active(context))fg_vk_end(context,&error);if(ok){scores[45678]=NAN;ok=fg_vk_tensor_write(input_scores,0,scores,(uint64_t)COUNT*4u,&error)==FG_OK;count=COUNT;slot=0;source_scores=input_scores;source_ids=input_ids;if(ok)ok=fg_vk_begin(context,&error)==FG_OK;while(ok&&count>1u){uint32_t next=0;ok=fg_vk_argmax_reduce(context,scratch_scores[slot],scratch_ids[slot],source_scores,source_ids,count,&next,&error)==FG_OK;source_scores=scratch_scores[slot];source_ids=scratch_ids[slot];count=next;slot^=1u;}if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_tensor_read(source_scores,0,&got_score,sizeof(got_score),&error)==FG_OK&&fg_vk_tensor_read(source_ids,0,&got_id,sizeof(got_id),&error)==FG_OK&&got_id==45678u&&!isfinite(got_score);else if(fg_vk_batch_active(context))fg_vk_end(context,&error);}fg_vk_tensor_destroy(scratch_ids[1]);fg_vk_tensor_destroy(scratch_ids[0]);fg_vk_tensor_destroy(scratch_scores[1]);fg_vk_tensor_destroy(scratch_scores[0]);fg_vk_tensor_destroy(input_ids);fg_vk_tensor_destroy(input_scores);free(ids);free(scores);return ok;
}

static int test_qsa_prefill_prepare(void){
    enum{TOKENS=3,QG=12288,Q=6144,K=512,IQ=512};float *raw_qg=malloc((size_t)TOKENS*QG*4u),*raw_k=malloc((size_t)TOKENS*K*4u),*raw_iq=malloc((size_t)TOKENS*IQ*4u),*batch_q=malloc((size_t)TOKENS*Q*4u),*batch_gate=malloc((size_t)TOKENS*Q*4u),*batch_k=malloc((size_t)TOKENS*K*4u),*batch_iq=malloc((size_t)TOKENS*IQ*4u);float qnorm[256],knorm[256],inorm[128];uint32_t positions[TOKENS*3u];if(!raw_qg||!raw_k||!raw_iq||!batch_q||!batch_gate||!batch_k||!batch_iq){free(batch_iq);free(batch_k);free(batch_gate);free(batch_q);free(raw_iq);free(raw_k);free(raw_qg);return 0;}for(uint32_t token=0;token<TOKENS;token++){positions[token*3u]=17u+token;positions[token*3u+1u]=5u+token*2u;positions[token*3u+2u]=9u+token*3u;for(uint32_t i=0;i<QG;i++)raw_qg[(uint64_t)token*QG+i]=sinf((float)(token+1u)*(float)(i+3u)*0.0013f)+0.07f*cosf((float)i*0.0047f);for(uint32_t i=0;i<K;i++)raw_k[(uint64_t)token*K+i]=cosf((float)(token+2u)*(float)(i+1u)*0.0021f);for(uint32_t i=0;i<IQ;i++)raw_iq[(uint64_t)token*IQ+i]=sinf((float)(token+3u)*(float)(i+2u)*0.0019f);}for(uint32_t i=0;i<256u;i++){qnorm[i]=0.02f*sinf((float)i*0.017f);knorm[i]=0.03f*cosf((float)i*0.013f);}for(uint32_t i=0;i<128u;i++)inorm[i]=0.025f*sinf((float)i*0.023f);
    fg_vk_tensor *rq=tensor(raw_qg,(uint64_t)TOKENS*QG*4u),*rk=tensor(raw_k,(uint64_t)TOKENS*K*4u),*ri=tensor(raw_iq,(uint64_t)TOKENS*IQ*4u),*qn=tensor(qnorm,sizeof(qnorm)),*kn=tensor(knorm,sizeof(knorm)),*in=tensor(inorm,sizeof(inorm)),*pos=tensor(positions,sizeof(positions)),*bq=tensor(NULL,(uint64_t)TOKENS*Q*4u),*bg=tensor(NULL,(uint64_t)TOKENS*Q*4u),*bk=tensor(NULL,(uint64_t)TOKENS*K*4u),*bi=tensor(NULL,(uint64_t)TOKENS*IQ*4u);int ok=rq&&rk&&ri&&qn&&kn&&in&&pos&&bq&&bg&&bk&&bi&&fg_vk_qsa_prepare_prefill(context,bq,bg,bk,rq,rk,qn,kn,pos,TOKENS,&error)==FG_OK&&fg_vk_qsa_index_prepare_prefill(context,bi,ri,in,pos,TOKENS,&error)==FG_OK&&fg_vk_tensor_read(bq,0,batch_q,(uint64_t)TOKENS*Q*4u,&error)==FG_OK&&fg_vk_tensor_read(bg,0,batch_gate,(uint64_t)TOKENS*Q*4u,&error)==FG_OK&&fg_vk_tensor_read(bk,0,batch_k,(uint64_t)TOKENS*K*4u,&error)==FG_OK&&fg_vk_tensor_read(bi,0,batch_iq,(uint64_t)TOKENS*IQ*4u,&error)==FG_OK;
    for(uint32_t token=0;ok&&token<TOKENS;token++){fg_vk_tensor *rqv=NULL,*rkv=NULL,*riv=NULL,*pv=NULL,*sq=tensor(NULL,Q*4u),*sg=tensor(NULL,Q*4u),*sk=tensor(NULL,K*4u),*si=tensor(NULL,IQ*4u);float *single_q=malloc(Q*4u),*single_gate=malloc(Q*4u);float single_k[K],single_iq[IQ];ok=sq&&sg&&sk&&si&&single_q&&single_gate&&fg_vk_tensor_view(rq,(uint64_t)token*QG*4u,QG*4u,&rqv,&error)==FG_OK&&fg_vk_tensor_view(rk,(uint64_t)token*K*4u,K*4u,&rkv,&error)==FG_OK&&fg_vk_tensor_view(ri,(uint64_t)token*IQ*4u,IQ*4u,&riv,&error)==FG_OK&&fg_vk_tensor_view(pos,(uint64_t)token*3u*4u,3u*4u,&pv,&error)==FG_OK&&fg_vk_qsa_prepare(context,sq,sg,sk,rqv,rkv,qn,kn,pv,&error)==FG_OK&&fg_vk_qsa_index_prepare(context,si,riv,in,pv,&error)==FG_OK&&fg_vk_tensor_read(sq,0,single_q,Q*4u,&error)==FG_OK&&fg_vk_tensor_read(sg,0,single_gate,Q*4u,&error)==FG_OK&&fg_vk_tensor_read(sk,0,single_k,K*4u,&error)==FG_OK&&fg_vk_tensor_read(si,0,single_iq,IQ*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<Q;i++)if(fabsf(batch_q[(uint64_t)token*Q+i]-single_q[i])>4e-5f||batch_gate[(uint64_t)token*Q+i]!=single_gate[i])ok=0;for(uint32_t i=0;ok&&i<K;i++)if(fabsf(batch_k[(uint64_t)token*K+i]-single_k[i])>4e-5f||fabsf(batch_iq[(uint64_t)token*IQ+i]-single_iq[i])>4e-5f)ok=0;free(single_gate);free(single_q);fg_vk_tensor_destroy(si);fg_vk_tensor_destroy(sk);fg_vk_tensor_destroy(sg);fg_vk_tensor_destroy(sq);fg_vk_tensor_destroy(pv);fg_vk_tensor_destroy(riv);fg_vk_tensor_destroy(rkv);fg_vk_tensor_destroy(rqv);}
    fg_vk_tensor_destroy(bi);fg_vk_tensor_destroy(bk);fg_vk_tensor_destroy(bg);fg_vk_tensor_destroy(bq);fg_vk_tensor_destroy(pos);fg_vk_tensor_destroy(in);fg_vk_tensor_destroy(kn);fg_vk_tensor_destroy(qn);fg_vk_tensor_destroy(ri);fg_vk_tensor_destroy(rk);fg_vk_tensor_destroy(rq);free(batch_iq);free(batch_k);free(batch_gate);free(batch_q);free(raw_iq);free(raw_k);free(raw_qg);return ok;
}

static int test_qsa_attention(void){
    enum{QG=12288,Q=6144,K=512,SELECTED=17};float *raw_qg=malloc(QG*4u),raw_k[K],qnorm[256],knorm[256],expected_q[Q],expected_gate[Q],expected_k[K],got_q[Q],got_gate[Q],got_k[K],raw_iq[512],inorm[128],expected_iq[512],got_iq[512];if(!raw_qg)return 0;for(uint32_t i=0;i<QG;i++)raw_qg[i]=sinf((float)i*0.0041f)+0.1f*cosf((float)i*0.013f);for(uint32_t i=0;i<K;i++)raw_k[i]=cosf((float)i*0.009f)-0.05f*sinf((float)i*0.021f);for(uint32_t i=0;i<256u;i++){qnorm[i]=0.03f*sinf((float)i*0.017f);knorm[i]=0.02f*cosf((float)i*0.023f);}uint32_t position[3]={19,7,11};for(uint32_t head=0;head<24u;head++){for(uint32_t i=0;i<256u;i++){expected_q[head*256u+i]=raw_qg[head*512u+i];expected_gate[head*256u+i]=raw_qg[head*512u+256u+i];}if(fg_q38_rms_mrope(expected_q+head*256u,1u,256u,qnorm,position,&error)!=FG_OK){free(raw_qg);return 0;}}memcpy(expected_k,raw_k,sizeof(raw_k));for(uint32_t head=0;head<2u;head++)if(fg_q38_rms_mrope(expected_k+head*256u,1u,256u,knorm,position,&error)!=FG_OK){free(raw_qg);return 0;}for(uint32_t i=0;i<512u;i++)raw_iq[i]=sinf((float)i*0.012f);for(uint32_t i=0;i<128u;i++)inorm[i]=0.04f*cosf((float)i*0.031f);memcpy(expected_iq,raw_iq,sizeof(raw_iq));if(fg_q38_rms_mrope(expected_iq,4u,128u,inorm,position,&error)!=FG_OK){free(raw_qg);return 0;}
    fg_vk_tensor *grq=tensor(raw_qg,QG*4u),*grk=tensor(raw_k,sizeof(raw_k)),*gqn=tensor(qnorm,sizeof(qnorm)),*gkn=tensor(knorm,sizeof(knorm)),*gpos=tensor(position,sizeof(position)),*gq=tensor(NULL,Q*4u),*gg=tensor(NULL,Q*4u),*gk=tensor(NULL,K*4u),*griq=tensor(raw_iq,sizeof(raw_iq)),*gin=tensor(inorm,sizeof(inorm)),*giq=tensor(NULL,sizeof(got_iq));int ok=grq&&grk&&gqn&&gkn&&gpos&&gq&&gg&&gk&&griq&&gin&&giq&&fg_vk_qsa_prepare(context,gq,gg,gk,grq,grk,gqn,gkn,gpos,&error)==FG_OK&&fg_vk_qsa_index_prepare(context,giq,griq,gin,gpos,&error)==FG_OK&&fg_vk_tensor_read(gq,0,got_q,sizeof(got_q),&error)==FG_OK&&fg_vk_tensor_read(gg,0,got_gate,sizeof(got_gate),&error)==FG_OK&&fg_vk_tensor_read(gk,0,got_k,sizeof(got_k),&error)==FG_OK&&fg_vk_tensor_read(giq,0,got_iq,sizeof(got_iq),&error)==FG_OK;for(uint32_t i=0;ok&&i<Q;i++)if(fabsf(got_q[i]-expected_q[i])>4e-5f||got_gate[i]!=expected_gate[i])ok=0;for(uint32_t i=0;ok&&i<K;i++)if(fabsf(got_k[i]-expected_k[i])>4e-5f||fabsf(got_iq[i]-expected_iq[i])>4e-5f)ok=0;
    uint8_t *records=calloc(SELECTED,FG_Q38_QSA_TOKEN_RECORD_BYTES);float *decoded_keys=malloc(SELECTED*K*4u),*decoded_values=malloc(SELECTED*K*4u),*cpu=malloc(Q*4u),*gpu=malloc(Q*4u);if(!records||!decoded_keys||!decoded_values||!cpu||!gpu)ok=0;for(uint32_t token_id=0;ok&&token_id<SELECTED;token_id++){float key[K],value[K];uint8_t *record=records+(uint64_t)token_id*FG_Q38_QSA_TOKEN_RECORD_BYTES;for(uint32_t i=0;i<K;i++){key[i]=sinf((float)(token_id+1u)*(float)(i+2u)*0.0023f);value[i]=cosf((float)(token_id+3u)*(float)(i+1u)*0.0017f)*2.0f;}fg_quantize_q8_0(key,record,K);fg_quantize_q8_0(value,record+FG_Q38_QSA_KEY_BYTES,K);fg_dequantize_q8_0(record,decoded_keys+token_id*K,K);fg_dequantize_q8_0(record+FG_Q38_QSA_KEY_BYTES,decoded_values+token_id*K,K);}
    for(uint32_t head=0;ok&&head<24u;head++){uint32_t kv=head/12u;float scores[SELECTED],maximum=-INFINITY,denom=0.0f;for(uint32_t token_id=0;token_id<SELECTED;token_id++){float dot=0.0f;for(uint32_t i=0;i<256u;i++)dot=fmaf(expected_q[head*256u+i],decoded_keys[token_id*K+kv*256u+i],dot);scores[token_id]=dot*0.0625f;maximum=fmaxf(maximum,scores[token_id]);}for(uint32_t token_id=0;token_id<SELECTED;token_id++)denom+=expf(scores[token_id]-maximum);for(uint32_t i=0;i<256u;i++){float sum=0.0f;for(uint32_t token_id=0;token_id<SELECTED;token_id++)sum=fmaf(expf(scores[token_id]-maximum),decoded_values[token_id*K+kv*256u+i],sum);cpu[head*256u+i]=sum/denom/(1.0f+expf(-expected_gate[head*256u+i]));}}
    fg_vk_tensor *grecord=ok?tensor(records,SELECTED*FG_Q38_QSA_TOKEN_RECORD_BYTES):NULL,*gout=ok?tensor(NULL,Q*4u):NULL;ok=ok&&grecord&&gout&&fg_vk_qsa_attention(context,gout,grecord,gq,gg,SELECTED,&error)==FG_OK&&fg_vk_tensor_read(gout,0,gpu,Q*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<Q;i++)if(fabsf(gpu[i]-cpu[i])>4e-4f*fmaxf(1.0f,fabsf(cpu[i]))){fprintf(stderr,"QSA attention %u GPU=%g CPU=%g\n",i,gpu[i],cpu[i]);ok=0;}fg_vk_tensor_destroy(gout);fg_vk_tensor_destroy(grecord);free(gpu);free(cpu);free(decoded_values);free(decoded_keys);free(records);fg_vk_tensor_destroy(giq);fg_vk_tensor_destroy(gin);fg_vk_tensor_destroy(griq);fg_vk_tensor_destroy(gk);fg_vk_tensor_destroy(gg);fg_vk_tensor_destroy(gq);fg_vk_tensor_destroy(gpos);fg_vk_tensor_destroy(gkn);fg_vk_tensor_destroy(gqn);fg_vk_tensor_destroy(grk);fg_vk_tensor_destroy(grq);free(raw_qg);return ok;
}

static int test_qsa_attention_single(void){float query[6144]={0},gate[6144]={0},key[512]={0},value[512],decoded[512],got[6144];uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES]={0};for(uint32_t i=0;i<512u;i++)value[i]=cosf((float)i*0.017f)*2.0f;fg_quantize_q8_0(key,record,512u);fg_quantize_q8_0(value,record+FG_Q38_QSA_KEY_BYTES,512u);fg_dequantize_q8_0(record+FG_Q38_QSA_KEY_BYTES,decoded,512u);fg_vk_tensor *r=tensor(record,sizeof(record)),*q=tensor(query,sizeof(query)),*g=tensor(gate,sizeof(gate)),*o=tensor(NULL,sizeof(got));int ok=r&&q&&g&&o&&fg_vk_qsa_attention(context,o,r,q,g,1u,&error)==FG_OK&&fg_vk_tensor_read(o,0,got,sizeof(got),&error)==FG_OK;for(uint32_t head=0;ok&&head<24u;head++){uint32_t kv=head/12u;for(uint32_t i=0;i<256u;i++)if(fabsf(got[head*256u+i]-decoded[kv*256u+i]*0.5f)>2e-6f){fprintf(stderr,"QSA single %u GPU=%g CPU=%g\n",head*256u+i,got[head*256u+i],decoded[kv*256u+i]*0.5f);ok=0;break;}}fg_vk_tensor_destroy(o);fg_vk_tensor_destroy(g);fg_vk_tensor_destroy(q);fg_vk_tensor_destroy(r);return ok;}
static int test_swiglu(void){enum{VALUES=1280};float gate[VALUES],up[VALUES],got[VALUES];for(uint32_t i=0;i<VALUES;i++){gate[i]=sinf((float)i*0.03f)*4.0f;up[i]=cosf((float)i*0.017f)*2.0f;}fg_vk_tensor *g=tensor(gate,sizeof(gate)),*u=tensor(up,sizeof(up)),*o=tensor(NULL,sizeof(got));int ok=g&&u&&o&&fg_vk_swiglu(context,o,g,u,VALUES,&error)==FG_OK&&fg_vk_tensor_read(o,0,got,sizeof(got),&error)==FG_OK;for(uint32_t i=0;ok&&i<VALUES;i++){float ref=gate[i]/(1.0f+expf(-gate[i]))*up[i];if(fabsf(got[i]-ref)>2e-6f)ok=0;}fg_vk_tensor_destroy(o);fg_vk_tensor_destroy(u);fg_vk_tensor_destroy(g);return ok;}
static int test_dense_f32_and_silu(void){enum{INPUT=96,OUTPUT=17,TOKENS=2};float weights[INPUT*OUTPUT],input[INPUT*TOKENS],dense[OUTPUT*TOKENS],silu[OUTPUT*TOKENS];for(uint32_t i=0;i<INPUT*OUTPUT;i++)weights[i]=sinf((float)i*0.021f);for(uint32_t i=0;i<INPUT*TOKENS;i++)input[i]=cosf((float)i*0.037f);fg_vk_tensor *w=tensor(weights,sizeof(weights)),*x=tensor(input,sizeof(input)),*d=tensor(NULL,sizeof(dense)),*s=tensor(NULL,sizeof(silu));int ok=w&&x&&d&&s&&fg_vk_dense_f32(context,d,w,x,INPUT,OUTPUT,TOKENS,&error)==FG_OK&&fg_vk_silu_scaled(context,s,d,OUTPUT*TOKENS,0.25f,&error)==FG_OK&&fg_vk_tensor_read(d,0,dense,sizeof(dense),&error)==FG_OK&&fg_vk_tensor_read(s,0,silu,sizeof(silu),&error)==FG_OK;for(uint32_t token=0;ok&&token<TOKENS;token++)for(uint32_t row=0;ok&&row<OUTPUT;row++){float ref=0.0f;for(uint32_t i=0;i<INPUT;i++)ref=fmaf(weights[row*INPUT+i],input[token*INPUT+i],ref);float activated=(ref*0.25f)/(1.0f+expf(-ref*0.25f));if(fabsf(dense[token*OUTPUT+row]-ref)>2e-4f*fmaxf(1.0f,fabsf(ref))||fabsf(silu[token*OUTPUT+row]-activated)>2e-5f)ok=0;}fg_vk_tensor_destroy(s);fg_vk_tensor_destroy(d);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);return ok;}

static int test_group_norm(void){
    enum{WIDTH=2560,GROUPS=4,TOTAL=WIDTH*GROUPS};float *x=malloc(TOTAL*4u),*w=malloc(TOTAL*4u),*got=malloc(TOTAL*4u),*ref=malloc(TOTAL*4u);if(!x||!w||!got||!ref){free(ref);free(got);free(w);free(x);return 0;}
    for(uint32_t i=0;i<TOTAL;i++){x[i]=sinf((float)i*0.007f)+0.1f;w[i]=cosf((float)i*0.003f)*0.05f;}fg_q38_group_rms_norm(ref,x,w,GROUPS,WIDTH,1e-6f);
    fg_vk_tensor *gx=tensor(x,TOTAL*4u),*gw=tensor(w,TOTAL*4u),*gy=tensor(NULL,TOTAL*4u);int ok=gx&&gw&&gy&&fg_vk_group_rms_norm(context,gy,gx,gw,WIDTH,GROUPS,1,1e-6f,&error)==FG_OK&&fg_vk_tensor_read(gy,0,got,TOTAL*4u,&error)==FG_OK;
    for(uint32_t i=0;ok&&i<TOTAL;i++)if(fabsf(got[i]-ref[i])>2e-5f)ok=0;
    fg_vk_tensor_destroy(gy);fg_vk_tensor_destroy(gw);fg_vk_tensor_destroy(gx);free(ref);free(got);free(w);free(x);return ok;
}

static int test_gr_mix(void){
    enum{HIDDEN=2560,GROUPS=4,HYPER=HIDDEN*GROUPS};float *norm=malloc(HYPER*4u),*up=malloc(HYPER*4u),*mixed=malloc(HIDDEN*4u),*written=malloc(HYPER*4u);float inject_logits[GROUPS]={-2,-1,1,2},injection[GROUPS];if(!norm||!up||!mixed||!written){free(written);free(mixed);free(up);free(norm);return 0;}for(uint32_t i=0;i<HYPER;i++){norm[i]=sinf((float)i*0.01f);up[i]=cosf((float)i*0.02f);}
    fg_vk_tensor *gn=tensor(norm,HYPER*4u),*gu=tensor(up,HYPER*4u),*gi=tensor(inject_logits,sizeof(inject_logits)),*gm=tensor(NULL,HIDDEN*4u),*go=tensor(NULL,sizeof(injection)),*gw=tensor(NULL,HYPER*4u);int ok=gn&&gu&&gi&&gm&&go&&gw&&fg_vk_gr_mix(context,gm,go,gn,gu,gi,HIDDEN,GROUPS,1,&error)==FG_OK&&fg_vk_gr_write(context,gw,gn,gm,go,HIDDEN,GROUPS,1,&error)==FG_OK&&fg_vk_tensor_read(gm,0,mixed,HIDDEN*4u,&error)==FG_OK&&fg_vk_tensor_read(go,0,injection,sizeof(injection),&error)==FG_OK&&fg_vk_tensor_read(gw,0,written,HYPER*4u,&error)==FG_OK;
    for(uint32_t i=0;ok&&i<HIDDEN;i++){float ref=0.0f;for(uint32_t g=0;g<GROUPS;g++){uint32_t j=g*HIDDEN+i;ref+=norm[j]/(1.0f+expf(-up[j]));}ref/=GROUPS;if(fabsf(mixed[i]-ref)>2e-6f)ok=0;}for(uint32_t g=0;ok&&g<GROUPS;g++){float ref=2.0f/(1.0f+expf(-inject_logits[g]/GROUPS));if(fabsf(injection[g]-ref)>1e-6f)ok=0;}
    for(uint32_t g=0;ok&&g<GROUPS;g++)for(uint32_t i=0;ok&&i<HIDDEN;i++)if(fabsf(written[g*HIDDEN+i]-(norm[g*HIDDEN+i]+mixed[i]*injection[g]))>2e-6f)ok=0;
    fg_vk_tensor_destroy(gw);fg_vk_tensor_destroy(go);fg_vk_tensor_destroy(gm);fg_vk_tensor_destroy(gi);fg_vk_tensor_destroy(gu);fg_vk_tensor_destroy(gn);free(written);free(mixed);free(up);free(norm);return ok;
}

static int test_hc_inject_partial(void){
    enum{HIDDEN=2560,GROUPS=4,HYPER=HIDDEN*GROUPS,PIECES=24,ITERATIONS=24};float *normalized=malloc(HYPER*4u),*up=malloc(HYPER*4u),*weights=malloc((size_t)GROUPS*HYPER*4u),mixed_expected[HIDDEN],mixed_got[HIDDEN],inject_logits[GROUPS],injection_expected[GROUPS],injection_got[GROUPS];if(!normalized||!up||!weights){free(weights);free(up);free(normalized);return 0;}
    for(uint32_t i=0;i<HYPER;i++){normalized[i]=sinf((float)i*0.0031f);up[i]=cosf((float)i*0.0027f);}for(uint32_t i=0;i<GROUPS*HYPER;i++)weights[i]=sinf((float)i*0.00017f)*0.125f;for(uint32_t group=0;group<GROUPS;group++){float dot=0.0f;for(uint32_t i=0;i<HYPER;i++)dot=fmaf(weights[(uint64_t)group*HYPER+i],normalized[i],dot);inject_logits[group]=dot;injection_expected[group]=2.0f/(1.0f+expf(-dot/(float)GROUPS));}for(uint32_t i=0;i<HIDDEN;i++){float value=0.0f;for(uint32_t group=0;group<GROUPS;group++){uint32_t index=group*HIDDEN+i;value=fmaf(1.0f/(1.0f+expf(-up[index])),normalized[index],value);}mixed_expected[i]=value/(float)GROUPS;}
    fg_vk_tensor *gn=tensor(normalized,HYPER*4u),*gu=tensor(up,HYPER*4u),*gw=tensor(weights,(uint64_t)GROUPS*HYPER*4u),*gp=tensor(NULL,PIECES*GROUPS*4u),*gm=tensor(NULL,HIDDEN*4u),*gi=tensor(NULL,GROUPS*4u),*gl=tensor(inject_logits,sizeof(inject_logits)),*gom=tensor(NULL,HIDDEN*4u),*goi=tensor(NULL,GROUPS*4u);int ok=gn&&gu&&gw&&gp&&gm&&gi&&gl&&gom&&goi&&fg_vk_begin(context,&error)==FG_OK&&fg_vk_hc_inject_partial(context,gp,gn,gw,HIDDEN,GROUPS,1u,PIECES,&error)==FG_OK&&fg_vk_gr_mix_partial(context,gm,gi,gn,gu,gp,HIDDEN,GROUPS,1u,PIECES,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&fg_vk_tensor_read(gm,0,mixed_got,sizeof(mixed_got),&error)==FG_OK&&fg_vk_tensor_read(gi,0,injection_got,sizeof(injection_got),&error)==FG_OK;for(uint32_t i=0;ok&&i<HIDDEN;i++)if(fabsf(mixed_got[i]-mixed_expected[i])>2e-6f){fprintf(stderr,"HC partial mixed %u GPU=%g CPU=%g diff=%g\n",i,mixed_got[i],mixed_expected[i],mixed_got[i]-mixed_expected[i]);ok=0;}for(uint32_t group=0;ok&&group<GROUPS;group++)if(fabsf(injection_got[group]-injection_expected[group])>2e-5f*fmaxf(1.0f,fabsf(injection_expected[group]))){fprintf(stderr,"HC partial injection %u GPU=%g CPU=%g diff=%g\n",group,injection_got[group],injection_expected[group],injection_got[group]-injection_expected[group]);ok=0;}
    if(ok&&getenv("FG_BENCH_HC_INJECT")){fg_vk_profile old_profile={0},new_profile={0};struct timespec start,end;clock_gettime(CLOCK_MONOTONIC,&start);ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<ITERATIONS;i++)ok=fg_vk_dense_f32(context,gl,gw,gn,HYPER,GROUPS,1u,&error)==FG_OK&&fg_vk_gr_mix(context,gom,goi,gn,gu,gl,HIDDEN,GROUPS,1u,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&old_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double old_wall=((double)(end.tv_sec-start.tv_sec)*1e3+(double)(end.tv_nsec-start.tv_nsec)*1e-6)/(double)ITERATIONS;clock_gettime(CLOCK_MONOTONIC,&start);ok=ok&&fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<ITERATIONS;i++)ok=fg_vk_hc_inject_partial(context,gp,gn,gw,HIDDEN,GROUPS,1u,PIECES,&error)==FG_OK&&fg_vk_gr_mix_partial(context,gm,gi,gn,gu,gp,HIDDEN,GROUPS,1u,PIECES,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&new_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double new_wall=((double)(end.tv_sec-start.tv_sec)*1e3+(double)(end.tv_nsec-start.tv_nsec)*1e-6)/(double)ITERATIONS;if(ok)fprintf(stderr,"HC_INJECT_BENCH iterations=%u old_gpu_us=%.3f new_gpu_us=%.3f old_wall_us=%.3f new_wall_us=%.3f\n",ITERATIONS,old_profile.gpu_ms*1000.0/(double)ITERATIONS,new_profile.gpu_ms*1000.0/(double)ITERATIONS,old_wall*1000.0,new_wall*1000.0);}
    fg_vk_tensor_destroy(goi);fg_vk_tensor_destroy(gom);fg_vk_tensor_destroy(gl);fg_vk_tensor_destroy(gi);fg_vk_tensor_destroy(gm);fg_vk_tensor_destroy(gp);fg_vk_tensor_destroy(gw);fg_vk_tensor_destroy(gu);fg_vk_tensor_destroy(gn);free(weights);free(up);free(normalized);return ok;
}

static int test_gr_batch(void){
    enum{TOKENS=3,HIDDEN=32,GROUPS=4,HYPER=HIDDEN*GROUPS};float normalized[TOKENS*HYPER],up[TOKENS*HYPER],inject_logits[TOKENS*GROUPS],mixed[TOKENS*HIDDEN],injection[TOKENS*GROUPS],written[TOKENS*HYPER];for(uint32_t token=0;token<TOKENS;token++){for(uint32_t group=0;group<GROUPS;group++)inject_logits[token*GROUPS+group]=(float)((int32_t)token-(int32_t)group)*0.7f;for(uint32_t i=0;i<HYPER;i++){normalized[token*HYPER+i]=sinf((float)(token*HYPER+i)*0.019f);up[token*HYPER+i]=cosf((float)(token*HYPER+i)*0.023f);}}
    fg_vk_tensor *n=tensor(normalized,sizeof(normalized)),*u=tensor(up,sizeof(up)),*il=tensor(inject_logits,sizeof(inject_logits)),*m=tensor(NULL,sizeof(mixed)),*in=tensor(NULL,sizeof(injection)),*w=tensor(NULL,sizeof(written));int ok=n&&u&&il&&m&&in&&w&&fg_vk_gr_mix(context,m,in,n,u,il,HIDDEN,GROUPS,TOKENS,&error)==FG_OK&&fg_vk_gr_write(context,w,n,m,in,HIDDEN,GROUPS,TOKENS,&error)==FG_OK&&fg_vk_tensor_read(m,0,mixed,sizeof(mixed),&error)==FG_OK&&fg_vk_tensor_read(in,0,injection,sizeof(injection),&error)==FG_OK&&fg_vk_tensor_read(w,0,written,sizeof(written),&error)==FG_OK;
    for(uint32_t token=0;ok&&token<TOKENS;token++){for(uint32_t i=0;i<HIDDEN;i++){float expected=0.0f;for(uint32_t group=0;group<GROUPS;group++){uint32_t index=token*HYPER+group*HIDDEN+i;expected+=normalized[index]/(1.0f+expf(-up[index]));}expected/=GROUPS;if(fabsf(mixed[token*HIDDEN+i]-expected)>2e-6f)ok=0;}for(uint32_t group=0;ok&&group<GROUPS;group++){float scale=2.0f/(1.0f+expf(-inject_logits[token*GROUPS+group]/GROUPS));if(fabsf(injection[token*GROUPS+group]-scale)>1e-6f)ok=0;for(uint32_t i=0;ok&&i<HIDDEN;i++){uint32_t index=token*HYPER+group*HIDDEN+i;float expected=normalized[index]+mixed[token*HIDDEN+i]*scale;if(fabsf(written[index]-expected)>2e-6f)ok=0;}}}
    fg_vk_tensor_destroy(w);fg_vk_tensor_destroy(in);fg_vk_tensor_destroy(m);fg_vk_tensor_destroy(il);fg_vk_tensor_destroy(u);fg_vk_tensor_destroy(n);return ok;
}

static int test_gr_partial_boundaries(void){
    enum{HIDDEN=32,GROUPS=4,HYPER=HIDDEN*GROUPS,PIECES=24};
    const uint32_t counts[]={1u,127u,128u};
    float weights[GROUPS*HYPER];
    for(uint32_t i=0;i<GROUPS*HYPER;i++)
        weights[i]=sinf((float)(i+5u)*0.017f)*0.125f;
    fg_vk_tensor *weight_tensor=tensor(weights,sizeof(weights));
    int ok=weight_tensor!=NULL;
    for(uint32_t shape=0;ok&&shape<sizeof(counts)/sizeof(counts[0]);shape++){
        uint32_t tokens=counts[shape];
        uint64_t hyper_values=(uint64_t)tokens*HYPER;
        float *normalized=malloc((size_t)hyper_values*4u);
        float *up=malloc((size_t)hyper_values*4u);
        float *mixed=malloc((size_t)tokens*HIDDEN*4u);
        float *injection=malloc((size_t)tokens*GROUPS*4u);
        float *written=malloc((size_t)hyper_values*4u);
        if(!normalized||!up||!mixed||!injection||!written){
            free(written);free(injection);free(mixed);free(up);free(normalized);
            ok=0;break;
        }
        for(uint64_t i=0;i<hyper_values;i++){
            normalized[i]=sinf((float)(i+tokens)*0.011f);
            up[i]=cosf((float)(i+3u)*0.013f);
        }
        fg_vk_tensor *n=tensor(normalized,hyper_values*4u);
        fg_vk_tensor *u=tensor(up,hyper_values*4u);
        fg_vk_tensor *p=tensor(NULL,(uint64_t)tokens*PIECES*GROUPS*4u);
        fg_vk_tensor *m=tensor(NULL,(uint64_t)tokens*HIDDEN*4u);
        fg_vk_tensor *i=tensor(NULL,(uint64_t)tokens*GROUPS*4u);
        fg_vk_tensor *w=tensor(NULL,hyper_values*4u);
        ok=n&&u&&p&&m&&i&&w&&fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_hc_inject_partial(context,p,n,weight_tensor,HIDDEN,GROUPS,
                                    tokens,PIECES,&error)==FG_OK&&
            fg_vk_gr_mix_partial(context,m,i,n,u,p,HIDDEN,GROUPS,tokens,
                                 PIECES,&error)==FG_OK&&
            fg_vk_end(context,&error)==FG_OK&&
            fg_vk_tensor_read(m,0,mixed,(uint64_t)tokens*HIDDEN*4u,&error)==FG_OK&&
            fg_vk_tensor_read(i,0,injection,(uint64_t)tokens*GROUPS*4u,&error)==FG_OK&&
            fg_vk_gr_write(context,w,n,m,i,HIDDEN,GROUPS,tokens,&error)==FG_OK&&
            fg_vk_tensor_read(w,0,written,hyper_values*4u,&error)==FG_OK;
        if(!ok&&fg_vk_batch_active(context)){
            fg_error ignored={0};fg_vk_abort(context,&ignored);
        }
        for(uint32_t token=0;ok&&token<tokens;token++){
            for(uint32_t element=0;element<HIDDEN;element++){
                float mixed_expected=0.0f;
                for(uint32_t group=0;group<GROUPS;group++){
                    uint32_t index=token*HYPER+group*HIDDEN+element;
                    mixed_expected+=normalized[index]/
                        (1.0f+expf(-up[index]));
                }
                mixed_expected/=(float)GROUPS;
                if(fabsf(mixed[token*HIDDEN+element]-mixed_expected)>2e-5f){
                    ok=0;break;
                }
            }
            for(uint32_t group=0;ok&&group<GROUPS;group++){
                float dot=0.0f;
                for(uint32_t index=0;index<HYPER;index++)
                    dot=fmaf(weights[group*HYPER+index],
                             normalized[token*HYPER+index],dot);
                float expected=2.0f/(1.0f+expf(-dot/(float)GROUPS));
                if(fabsf(injection[token*GROUPS+group]-expected)>2e-5f){
                    ok=0;break;
                }
                for(uint32_t element=0;ok&&element<HIDDEN;element++){
                    uint32_t index=token*HYPER+group*HIDDEN+element;
                    float value=normalized[index]+
                        mixed[token*HIDDEN+element]*expected;
                    if(fabsf(written[index]-value)>2e-5f)ok=0;
                }
            }
        }
        fg_vk_tensor_destroy(w);fg_vk_tensor_destroy(i);fg_vk_tensor_destroy(m);
        fg_vk_tensor_destroy(p);fg_vk_tensor_destroy(u);fg_vk_tensor_destroy(n);
        free(written);free(injection);free(mixed);free(up);free(normalized);
    }
    fg_vk_tensor_destroy(weight_tensor);
    return ok;
}

static int test_q5_1_down(void){
    enum{INPUT=640,OUTPUT=8};uint8_t weights[OUTPUT*(INPUT/32)*24];float x[INPUT],got[OUTPUT];uint32_t tiles[9];for(uint32_t i=0;i<9;i++)tiles[i]=UINT32_MAX;tiles[0]=0;tiles[1]=0;for(uint32_t r=0;r<OUTPUT;r++)make_q5_1_row(weights+(uint64_t)r*(INPUT/32)*24,INPUT,r);for(uint32_t i=0;i<INPUT;i++)x[i]=sinf((float)i*0.03125f);
    fg_vk_tensor *w=tensor(weights,sizeof(weights)),*schedule=tensor(tiles,sizeof(tiles)),*input=tensor(x,sizeof(x)),*output=tensor(NULL,sizeof(got));int ok=w&&schedule&&input&&output&&fg_vk_moe_q5_1_down(context,output,w,schedule,input,OUTPUT,INPUT,sizeof(weights),1,false,1,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,sizeof(got),&error)==FG_OK;
    for(uint32_t r=0;ok&&r<OUTPUT;r++){float ref=fg_q5_1_dot_f32(weights+(uint64_t)r*(INPUT/32)*24,x,INPUT);if(fabsf(got[r]-ref)>1e-4f*fmaxf(1.0f,fabsf(ref)))ok=0;}fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(schedule);fg_vk_tensor_destroy(w);return ok;
}

static int run_q5_1_cooked(uint32_t output_width,uint32_t selected,uint32_t iterations){
    const uint32_t input_width=640u,slots=FG_TOP_K,blocks=input_width/FG_QK8_0,row_bytes=blocks*FG_Q5_1_BLOCK_BYTES;uint64_t expert_bytes=(uint64_t)output_width*row_bytes,weight_bytes=(uint64_t)selected*expert_bytes,output_values=(uint64_t)slots*output_width,input_values=(uint64_t)slots*input_width;uint8_t *weights=malloc((size_t)weight_bytes),*cooked=malloc((size_t)weight_bytes);float *input=malloc((size_t)input_values*4u),*generic_values=malloc((size_t)output_values*4u),*cooked_values=malloc((size_t)output_values*4u);uint32_t *tiles=malloc((size_t)selected*9u*4u);int ok=weights&&cooked&&input&&generic_values&&cooked_values&&tiles;if(!ok)goto done;
    for(uint32_t expert=0;expert<selected;expert++){for(uint32_t row=0;row<output_width;row++)make_q5_1_row(weights+(uint64_t)expert*expert_bytes+(uint64_t)row*row_bytes,input_width,expert*37u+row);ok=fg_cook_q5_1_rows(weights+(uint64_t)expert*expert_bytes,cooked+(uint64_t)expert*expert_bytes,expert_bytes,input_width,output_width);if(!ok)goto done;}for(uint64_t i=0;i<input_values;i++)input[i]=sinf((float)(i+3u)*0.013f)+0.1f*cosf((float)i*0.007f);for(uint32_t i=0;i<selected*9u;i++)tiles[i]=UINT32_MAX;for(uint32_t expert=0;expert<selected;expert++){tiles[expert*9u]=expert;tiles[expert*9u+1u]=expert;}
    fg_vk_tensor *generic_weights=tensor(weights,weight_bytes),*cooked_weights=tensor(cooked,weight_bytes),*schedule=tensor(tiles,(uint64_t)selected*9u*4u),*inputs=tensor(input,input_values*4u),*generic=tensor(NULL,output_values*4u),*candidate=tensor(NULL,output_values*4u);ok=generic_weights&&cooked_weights&&schedule&&inputs&&generic&&candidate&&fg_vk_moe_q5_1_down(context,generic,generic_weights,schedule,inputs,output_width,input_width,(uint32_t)expert_bytes,selected,true,selected,&error)==FG_OK&&fg_vk_moe_q5_1_down_cooked(context,candidate,cooked_weights,schedule,inputs,output_width,input_width,(uint32_t)expert_bytes,true,selected,&error)==FG_OK&&fg_vk_tensor_read(generic,0,generic_values,output_values*4u,&error)==FG_OK&&fg_vk_tensor_read(candidate,0,cooked_values,output_values*4u,&error)==FG_OK;double max_abs=0.0,max_rel=0.0;for(uint64_t i=0;ok&&i<(uint64_t)selected*output_width;i++){double difference=fabs((double)generic_values[i]-cooked_values[i]),relative=difference/fmax(1.0,fabs((double)generic_values[i]));if(difference>max_abs)max_abs=difference;if(relative>max_rel)max_rel=relative;if(relative>2e-4)ok=0;}
    if(ok&&iterations){fg_vk_profile generic_profile={0},cooked_profile={0};struct timespec begin,end;clock_gettime(CLOCK_MONOTONIC,&begin);ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"q5_1_generic",&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_moe_q5_1_down(context,generic,generic_weights,schedule,inputs,output_width,input_width,(uint32_t)expert_bytes,selected,true,selected,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&generic_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double generic_wall=elapsed_ms(begin,end)/(double)iterations;clock_gettime(CLOCK_MONOTONIC,&begin);ok=ok&&fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"q5_1_cooked",&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_moe_q5_1_down_cooked(context,candidate,cooked_weights,schedule,inputs,output_width,input_width,(uint32_t)expert_bytes,true,selected,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&cooked_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double cooked_wall=elapsed_ms(begin,end)/(double)iterations;if(ok){double generic_gpu=generic_profile.kernel_ms/(double)iterations,cooked_gpu=cooked_profile.kernel_ms/(double)iterations;fprintf(stderr,"Q5_1_COOKED_AB selected=%u input=%u output=%u bytes=%llu iterations=%u generic_gpu_us=%.3f cooked_gpu_us=%.3f generic_wall_us=%.3f cooked_wall_us=%.3f speedup=%.4f max_abs=%.9g max_rel=%.9g parity=PASS\n",selected,input_width,output_width,(unsigned long long)weight_bytes,iterations,generic_gpu*1000.0,cooked_gpu*1000.0,generic_wall*1000.0,cooked_wall*1000.0,generic_gpu/cooked_gpu,max_abs,max_rel);}}
    fg_vk_tensor_destroy(candidate);fg_vk_tensor_destroy(generic);fg_vk_tensor_destroy(inputs);fg_vk_tensor_destroy(schedule);fg_vk_tensor_destroy(cooked_weights);fg_vk_tensor_destroy(generic_weights);
done:free(tiles);free(cooked_values);free(generic_values);free(input);free(cooked);free(weights);return ok;
}

static int test_q5_1_down_cooked(void){return run_q5_1_cooked(16u,2u,0u);}
static int test_q5_1_down_cooked_benchmark(void){return getenv("FG_BENCH_Q5_1_COOKED")?run_q5_1_cooked(FG_HIDDEN_SIZE,6u,20u):1;}

static int test_q8_0_down(void){
    enum{INPUT=640,OUTPUT=8};float *source=malloc((size_t)INPUT*OUTPUT*4u),x[INPUT],got[OUTPUT];uint8_t *weights=malloc((size_t)OUTPUT*(INPUT/32u)*FG_Q8_0_BLOCK_BYTES);uint32_t tiles[9];for(uint32_t i=0;i<9;i++)tiles[i]=UINT32_MAX;tiles[0]=0;tiles[1]=0;if(!source||!weights){free(weights);free(source);return 0;}for(uint32_t i=0;i<INPUT;i++)x[i]=cosf((float)i*0.017f);for(uint32_t row=0;row<OUTPUT;row++){for(uint32_t i=0;i<INPUT;i++)source[row*INPUT+i]=sinf((float)(row*INPUT+i)*0.009f);fg_quantize_q8_0(source+row*INPUT,weights+(uint64_t)row*(INPUT/32u)*FG_Q8_0_BLOCK_BYTES,INPUT);}
    uint32_t expert_stride=OUTPUT*(INPUT/32u)*FG_Q8_0_BLOCK_BYTES;fg_vk_tensor *w=tensor(weights,expert_stride),*schedule=tensor(tiles,sizeof(tiles)),*input=tensor(x,sizeof(x)),*output=tensor(NULL,sizeof(got));int ok=w&&schedule&&input&&output&&fg_vk_moe_q8_0_down(context,output,w,schedule,input,OUTPUT,INPUT,expert_stride,1,false,1,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,sizeof(got),&error)==FG_OK;
    for(uint32_t row=0;ok&&row<OUTPUT;row++){float ref=0.0f;const uint8_t *blocks=weights+(uint64_t)row*(INPUT/32u)*FG_Q8_0_BLOCK_BYTES;for(uint32_t block=0;block<INPUT/32u;block++){const uint8_t *p=blocks+block*FG_Q8_0_BLOCK_BYTES;float d=fg_f16_to_f32((uint16_t)p[0]|(uint16_t)((uint16_t)p[1]<<8u));for(uint32_t i=0;i<32u;i++)ref=fmaf(d*(float)(int8_t)p[2u+i],x[block*32u+i],ref);}if(fabsf(got[row]-ref)>2e-4f*fmaxf(1.0f,fabsf(ref)))ok=0;}
    fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(schedule);fg_vk_tensor_destroy(w);free(weights);free(source);return ok;
}

static int test_pipeline_decode_down_formats(void){
    return test_q5_1_down_cooked()&&test_q8_0_down();
}

static int test_moe_reduce(void){
    enum{WIDTH=513,SLOTS=4,SELECTED=3};float down[SLOTS*WIDTH],gates[SELECTED]={0.25f,-0.75f,1.5f},got[WIDTH],expected[WIDTH];uint32_t tiles[SELECTED*9u];for(uint32_t i=0;i<SELECTED*9u;i++)tiles[i]=UINT32_MAX;tiles[1]=3u;tiles[10]=0u;tiles[19]=2u;for(uint32_t slot=0;slot<SLOTS;slot++)for(uint32_t i=0;i<WIDTH;i++)down[slot*WIDTH+i]=sinf((float)(slot*WIDTH+i)*0.013f);for(uint32_t i=0;i<WIDTH;i++){expected[i]=0.0f;for(uint32_t selected=0;selected<SELECTED;selected++)expected[i]=fmaf(gates[selected],down[tiles[selected*9u+1u]*WIDTH+i],expected[i]);}
    fg_vk_tensor *gd=tensor(down,sizeof(down)),*gg=tensor(gates,sizeof(gates)),*gt=tensor(tiles,sizeof(tiles)),*go=tensor(NULL,sizeof(got));int ok=gd&&gg&&gt&&go&&fg_vk_moe_reduce(context,go,gd,gg,gt,WIDTH,SELECTED,SLOTS,&error)==FG_OK&&fg_vk_tensor_read(go,0,got,sizeof(got),&error)==FG_OK;for(uint32_t i=0;ok&&i<WIDTH;i++)if(fabsf(got[i]-expected[i])>2e-6f*fmaxf(1.0f,fabsf(expected[i]))){fprintf(stderr,"MoE reduce %u GPU=%g CPU=%g diff=%g\n",i,got[i],expected[i],got[i]-expected[i]);ok=0;}fg_vk_tensor_destroy(go);fg_vk_tensor_destroy(gt);fg_vk_tensor_destroy(gg);fg_vk_tensor_destroy(gd);return ok;
}

static void make_k_row(uint8_t *block,int q5,uint32_t phase){
    uint16_t d=fg_f32_to_f16(0.25f),dmin=fg_f32_to_f16(0.03125f);memcpy(block,&d,2);memcpy(block+2,&dmin,2);
    for(uint32_t group=0;group<8u;group++){uint32_t scale=group+1u,minimum=8u-group;if(group<4u){block[4u+group]=(uint8_t)scale;block[8u+group]=(uint8_t)minimum;}else block[8u+group]=(uint8_t)(scale|(minimum<<4u));}
    for(uint32_t group=0;group<8u;group++)for(uint32_t i=0;i<32u;i++){uint32_t value=(group*3u+i+phase)&(q5?31u:15u);uint32_t packed_offset=(q5?48u:16u)+(group>>1u)*32u+i;if(group&1u)block[packed_offset]|=(uint8_t)((value&15u)<<4u);else block[packed_offset]|=(uint8_t)(value&15u);if(q5&&(value&16u))block[16u+i]|=(uint8_t)(1u<<group);}
}

static int test_kquant(int type){
    enum{INPUT=256,OUTPUT=8};uint32_t block_bytes=type==12?144u:176u;uint8_t *weights=calloc(OUTPUT,block_bytes);float x[INPUT],got[OUTPUT];uint8_t q8[FG_Q8_K_BLOCK_BYTES];uint32_t tiles[9];for(uint32_t i=0;i<9;i++)tiles[i]=UINT32_MAX;tiles[0]=0;tiles[1]=0;if(!weights)return 0;for(uint32_t row=0;row<OUTPUT;row++)make_k_row(weights+row*block_bytes,type==13,row);for(uint32_t i=0;i<INPUT;i++)x[i]=sinf((float)i*0.019f);fg_quantize_q8_k(x,q8,INPUT);
    fg_vk_tensor *w=tensor(weights,(uint64_t)OUTPUT*block_bytes),*activation=tensor(q8,sizeof(q8)),*schedule=tensor(tiles,sizeof(tiles)),*output=tensor(NULL,sizeof(got));int ok=w&&activation&&schedule&&output&&fg_vk_moe_kquant(context,output,w,activation,schedule,(uint32_t)type,OUTPUT,INPUT,OUTPUT*block_bytes,1,1,false,1,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,sizeof(got),&error)==FG_OK;
    for(uint32_t row=0;ok&&row<OUTPUT;row++){float ref=type==12?fg_dot_q4_k_q8_k(weights+row*block_bytes,q8,INPUT):fg_dot_q5_k_q8_k(weights+row*block_bytes,q8,INPUT);if(fabsf(got[row]-ref)>2e-4f*fmaxf(1.0f,fabsf(ref))){fprintf(stderr,"K-quant type %d row %u GPU=%g CPU=%g\n",type,row,got[row],ref);ok=0;}}
    fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(schedule);fg_vk_tensor_destroy(activation);fg_vk_tensor_destroy(w);free(weights);return ok;
}

static int run_kquant_cooked(int type,uint32_t input_width,uint32_t output_width,uint32_t selected,uint32_t iterations){
    uint32_t block_bytes=type==12?144u:176u,blocks=input_width/FG_QK8_K,row_bytes=blocks*block_bytes,slots=FG_TOP_K;uint64_t expert_bytes=(uint64_t)output_width*row_bytes,weight_bytes=(uint64_t)selected*expert_bytes,output_values=(uint64_t)slots*output_width;uint8_t *weights=calloc(1,(size_t)weight_bytes),*cooked=malloc((size_t)weight_bytes),*activation=malloc((size_t)blocks*FG_Q8_K_BLOCK_BYTES);float *input=malloc((size_t)input_width*4u),*generic_values=malloc((size_t)output_values*4u),*cooked_values=malloc((size_t)output_values*4u);uint32_t *tiles=malloc((size_t)selected*9u*4u);int ok=weights&&cooked&&activation&&input&&generic_values&&cooked_values&&tiles;if(!ok)goto done;
    for(uint32_t expert=0;expert<selected;expert++){for(uint32_t row=0;row<output_width;row++)for(uint32_t block=0;block<blocks;block++)make_k_row(weights+(uint64_t)expert*expert_bytes+(uint64_t)row*row_bytes+(uint64_t)block*block_bytes,type==13,expert*37u+row*11u+block);ok=fg_cook_k_quant_rows(weights+(uint64_t)expert*expert_bytes,cooked+(uint64_t)expert*expert_bytes,expert_bytes,input_width,output_width,(uint32_t)type);if(!ok)goto done;}for(uint32_t i=0;i<input_width;i++)input[i]=sinf((float)(i+3u)*0.013f)+0.1f*cosf((float)i*0.007f);fg_quantize_q8_k(input,activation,input_width);for(uint32_t i=0;i<selected*9u;i++)tiles[i]=UINT32_MAX;for(uint32_t expert=0;expert<selected;expert++){tiles[expert*9u]=expert;tiles[expert*9u+1u]=expert;}
    fg_vk_tensor *generic_weights=tensor(weights,weight_bytes),*cooked_weights=tensor(cooked,weight_bytes),*activations=tensor(activation,(uint64_t)blocks*FG_Q8_K_BLOCK_BYTES),*schedule=tensor(tiles,(uint64_t)selected*9u*4u),*generic=tensor(NULL,output_values*4u),*candidate=tensor(NULL,output_values*4u);ok=generic_weights&&cooked_weights&&activations&&schedule&&generic&&candidate&&fg_vk_moe_kquant(context,generic,generic_weights,activations,schedule,(uint32_t)type,output_width,input_width,(uint32_t)expert_bytes,slots,slots,true,selected,&error)==FG_OK&&fg_vk_moe_kquant_cooked(context,candidate,cooked_weights,activations,schedule,(uint32_t)type,output_width,input_width,(uint32_t)expert_bytes,true,selected,&error)==FG_OK&&fg_vk_tensor_read(generic,0,generic_values,output_values*4u,&error)==FG_OK&&fg_vk_tensor_read(candidate,0,cooked_values,output_values*4u,&error)==FG_OK;double max_abs=0.0,max_rel=0.0;for(uint64_t i=0;ok&&i<(uint64_t)selected*output_width;i++){double difference=fabs((double)generic_values[i]-cooked_values[i]),relative=difference/fmax(1.0,fabs((double)generic_values[i]));if(difference>max_abs)max_abs=difference;if(relative>max_rel)max_rel=relative;if(relative>2e-4)ok=0;}
    if(ok&&iterations){fg_vk_profile generic_profile={0},cooked_profile={0};struct timespec begin,end;clock_gettime(CLOCK_MONOTONIC,&begin);ok=fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"kquant_generic",&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_moe_kquant(context,generic,generic_weights,activations,schedule,(uint32_t)type,output_width,input_width,(uint32_t)expert_bytes,slots,slots,true,selected,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&generic_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double generic_wall=elapsed_ms(begin,end)/(double)iterations;clock_gettime(CLOCK_MONOTONIC,&begin);ok=ok&&fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_set_scope(context,"kquant_cooked",&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<iterations;i++)ok=fg_vk_moe_kquant_cooked(context,candidate,cooked_weights,activations,schedule,(uint32_t)type,output_width,input_width,(uint32_t)expert_bytes,true,selected,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&cooked_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double cooked_wall=elapsed_ms(begin,end)/(double)iterations;if(ok){double generic_gpu=generic_profile.kernel_ms/(double)iterations,cooked_gpu=cooked_profile.kernel_ms/(double)iterations;fprintf(stderr,"KQUANT_COOKED_AB type=%d selected=%u input=%u output=%u bytes=%llu iterations=%u generic_gpu_us=%.3f cooked_gpu_us=%.3f generic_wall_us=%.3f cooked_wall_us=%.3f speedup=%.4f max_abs=%.9g max_rel=%.9g parity=PASS\n",type,selected,input_width,output_width,(unsigned long long)weight_bytes,iterations,generic_gpu*1000.0,cooked_gpu*1000.0,generic_wall*1000.0,cooked_wall*1000.0,generic_gpu/cooked_gpu,max_abs,max_rel);}}
    fg_vk_tensor_destroy(candidate);fg_vk_tensor_destroy(generic);fg_vk_tensor_destroy(schedule);fg_vk_tensor_destroy(activations);fg_vk_tensor_destroy(cooked_weights);fg_vk_tensor_destroy(generic_weights);
done:free(tiles);free(cooked_values);free(generic_values);free(input);free(activation);free(cooked);free(weights);return ok;
}

static int test_kquant_cooked(int type){return run_kquant_cooked(type,256u,8u,2u,0u);}
static int test_kquant_cooked_benchmark(void){if(!getenv("FG_BENCH_KQUANT_COOKED"))return 1;return run_kquant_cooked(12,FG_HIDDEN_SIZE,640u,6u,20u)&&run_kquant_cooked(13,FG_HIDDEN_SIZE,640u,6u,20u);}

static int test_kquant_expert_major_batch(int type){
    enum{INPUT=256,OUTPUT=8,EXPERTS=2,TOKENS=3,PAIRS=EXPERTS*TOKENS};uint32_t block_bytes=type==12?144u:176u,expert_stride=OUTPUT*block_bytes;
    uint8_t *weights=calloc(EXPERTS,expert_stride),activations[TOKENS*FG_Q8_K_BLOCK_BYTES];float input[TOKENS][INPUT],got[PAIRS*OUTPUT];uint32_t tiles[18];if(!weights)return 0;for(uint32_t i=0;i<18u;i++)tiles[i]=UINT32_MAX;tiles[0]=0;tiles[1]=0;tiles[2]=2;tiles[3]=4;tiles[9]=1;tiles[10]=1;tiles[11]=3;tiles[12]=5;
    for(uint32_t expert=0;expert<EXPERTS;expert++)for(uint32_t row=0;row<OUTPUT;row++)make_k_row(weights+(uint64_t)expert*expert_stride+row*block_bytes,type==13,expert*17u+row);
    for(uint32_t token=0;token<TOKENS;token++){for(uint32_t i=0;i<INPUT;i++)input[token][i]=sinf((float)(token+1u)*(float)(i+3u)*0.013f)+0.1f*cosf((float)(i+token)*0.031f);fg_quantize_q8_k(input[token],activations+token*FG_Q8_K_BLOCK_BYTES,INPUT);}
    fg_vk_tensor *w=tensor(weights,(uint64_t)EXPERTS*expert_stride),*a=tensor(activations,sizeof(activations)),*schedule=tensor(tiles,sizeof(tiles)),*output=tensor(NULL,sizeof(got));int ok=w&&a&&schedule&&output&&fg_vk_moe_kquant(context,output,w,a,schedule,(uint32_t)type,OUTPUT,INPUT,expert_stride,EXPERTS,PAIRS,false,EXPERTS,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,sizeof(got),&error)==FG_OK;
    for(uint32_t pair=0;ok&&pair<PAIRS;pair++){uint32_t token=pair/EXPERTS,expert=pair%EXPERTS;const uint8_t *q8=activations+token*FG_Q8_K_BLOCK_BYTES;for(uint32_t row=0;row<OUTPUT;row++){const uint8_t *weight=weights+(uint64_t)expert*expert_stride+row*block_bytes;float ref=type==12?fg_dot_q4_k_q8_k(weight,q8,INPUT):fg_dot_q5_k_q8_k(weight,q8,INPUT);if(fabsf(got[pair*OUTPUT+row]-ref)>2e-4f*fmaxf(1.0f,fabsf(ref))){fprintf(stderr,"expert-major K-quant type %d pair %u row %u GPU=%g CPU=%g\n",type,pair,row,got[pair*OUTPUT+row],ref);ok=0;break;}}}
    fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(schedule);fg_vk_tensor_destroy(a);fg_vk_tensor_destroy(w);free(weights);return ok;
}

static float q8_row_reference(const uint8_t *row,const float *input,
                              uint32_t width){
    float value=0.0f;
    for(uint32_t block=0;block<width/FG_QK8_0;block++){
        const uint8_t *packed=row+(uint64_t)block*FG_Q8_0_BLOCK_BYTES;
        float delta=fg_f16_to_f32((uint16_t)packed[0]|
                                  (uint16_t)((uint16_t)packed[1]<<8u));
        for(uint32_t i=0;i<FG_QK8_0;i++)
            value=fmaf(delta*(float)(int8_t)packed[2u+i],
                       input[block*FG_QK8_0+i],value);
    }
    return value;
}

static int test_q8_cooked_token_tiles(void){
    const uint32_t widths[]={2560u,10240u,640u};
    const uint32_t token_counts[]={2u,32u,64u,128u};
    enum{ROWS=65,GUARD_BYTES=4096};
    for(uint32_t shape=0;shape<3u;shape++){
        uint32_t width=widths[shape],blocks=width/FG_QK8_0;
        uint64_t row_bytes=(uint64_t)blocks*FG_Q8_0_BLOCK_BYTES;
        uint64_t raw_bytes=ROWS*row_bytes;
        uint64_t cooked_bytes=fg_q8_0_cooked_matrix_bytes(width,ROWS);
        uint8_t *raw=malloc((size_t)raw_bytes);
        uint8_t *cooked=malloc((size_t)cooked_bytes);
        if(!raw||!cooked){free(cooked);free(raw);return 0;}
        for(uint32_t row=0;row<ROWS;row++)
            for(uint32_t block=0;block<blocks;block++){
                uint8_t *value=raw+(uint64_t)row*row_bytes+
                    (uint64_t)block*FG_Q8_0_BLOCK_BYTES;
                uint16_t delta=fg_f32_to_f16(
                    0.001f+(float)((row+block)%11u)*0.0001f);
                memcpy(value,&delta,sizeof(delta));
                for(uint32_t i=0;i<FG_QK8_0;i++)
                    value[2u+i]=(uint8_t)(row*31u+block*17u+i*7u);
            }
        int ok=fg_cook_q8_0_rows(raw,cooked,cooked_bytes,width,ROWS);
        fg_vk_tensor *weight_storage=ok?
            tensor(NULL,cooked_bytes+2u*GUARD_BYTES):NULL;
        fg_vk_tensor *weights=NULL;
        if(weight_storage){
            uint8_t *storage=fg_vk_tensor_map(weight_storage);
            memset(storage,0xa5,GUARD_BYTES);
            memcpy(storage+GUARD_BYTES,cooked,(size_t)cooked_bytes);
            memset(storage+GUARD_BYTES+cooked_bytes,0xa5,GUARD_BYTES);
            ok=fg_vk_tensor_view(weight_storage,GUARD_BYTES,cooked_bytes,
                                 &weights,&error)==FG_OK;
        }else ok=0;
        if(weights)fg_vk_tensor_set_format(weights,
                                            FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
        for(uint32_t count=0;ok&&count<4u;count++){
            uint32_t tokens=token_counts[count];
            float *input=malloc((size_t)tokens*width*sizeof(*input));
            float *got=malloc((size_t)tokens*ROWS*sizeof(*got));
            if(!input||!got){free(got);free(input);ok=0;break;}
            for(uint32_t token=0;token<tokens;token++)
                for(uint32_t i=0;i<width;i++)
                    input[(uint64_t)token*width+i]=
                        sinf((float)(token+1u)*(float)(i+3u)*0.0013f)+
                        0.1f*cosf((float)(i+token)*0.007f);
            uint64_t output_bytes=(uint64_t)tokens*ROWS*4u;
            fg_vk_tensor *x=tensor(input,(uint64_t)tokens*width*4u);
            fg_vk_tensor *output_storage=tensor(
                NULL,output_bytes+2u*GUARD_BYTES);
            fg_vk_tensor *y=NULL;
            if(output_storage){
                memset(fg_vk_tensor_map(output_storage),0x5a,
                       (size_t)(output_bytes+2u*GUARD_BYTES));
                ok=fg_vk_tensor_view(output_storage,GUARD_BYTES,output_bytes,
                                     &y,&error)==FG_OK;
            }else ok=0;
            struct timespec begin,end;clock_gettime(CLOCK_MONOTONIC,&begin);
            ok=x&&y&&fg_vk_dense_q8_0_cooked_prefill(context,y,weights,x,
                width,ROWS,tokens,0.75f,&error)==FG_OK&&
                fg_vk_tensor_read(y,0,got,output_bytes,
                                  &error)==FG_OK;
            clock_gettime(CLOCK_MONOTONIC,&end);
            const uint8_t *output_guard=output_storage?
                fg_vk_tensor_const_map(output_storage):NULL;
            for(uint32_t i=0;ok&&i<GUARD_BYTES;i++)
                if(output_guard[i]!=0x5au||
                   output_guard[GUARD_BYTES+output_bytes+i]!=0x5au){
                    fprintf(stderr,
                        "Q8 tile output guard modified width=%u tokens=%u byte=%u\n",
                        width,tokens,i);
                    ok=0;
                }
            for(uint32_t token=0;ok&&token<tokens;token++)
                for(uint32_t row=0;row<ROWS;row++){
                    float reference=0.75f*q8_row_reference(
                        raw+(uint64_t)row*row_bytes,
                        input+(uint64_t)token*width,width);
                    if(fabsf(got[(uint64_t)token*ROWS+row]-reference)>
                       3e-4f*fmaxf(1.0f,fabsf(reference))){
                        fprintf(stderr,
                            "Q8 tile width=%u tokens=%u token=%u row=%u GPU=%g CPU=%g\n",
                            width,tokens,token,row,
                            got[(uint64_t)token*ROWS+row],reference);
                        ok=0;break;
                    }
                }
            if(getenv("FG_BENCH_PREFILL"))
                fprintf(stderr,
                    "PREFILL_Q8_TILE width=%u rows=%u tokens=%u wall_ms=%.3f\n",
                    width,ROWS,tokens,elapsed_ms(begin,end));
            fg_vk_tensor_destroy(y);fg_vk_tensor_destroy(output_storage);
            fg_vk_tensor_destroy(x);
            free(got);free(input);
        }
        const uint8_t *weight_guard=weight_storage?
            fg_vk_tensor_const_map(weight_storage):NULL;
        for(uint32_t i=0;ok&&i<GUARD_BYTES;i++)
            if(weight_guard[i]!=0xa5u||
               weight_guard[GUARD_BYTES+cooked_bytes+i]!=0xa5u){
                fprintf(stderr,
                    "Q8 tile weight guard modified width=%u byte=%u\n",
                    width,i);
                ok=0;
            }
        if(ok&&memcmp(weight_guard+GUARD_BYTES,cooked,(size_t)cooked_bytes)!=0){
            fprintf(stderr,"Q8 tile weights modified width=%u\n",width);
            ok=0;
        }
        fg_vk_tensor_destroy(weights);fg_vk_tensor_destroy(weight_storage);
        free(cooked);free(raw);
        if(!ok)return 0;
    }
    return 1;
}

static int verify_expert_tiles(const uint32_t *selected,const uint32_t *tiles,
                               uint32_t pairs){
    bool *seen=calloc(pairs,sizeof(*seen));
    if(!seen)return 0;
    int ok=1;
    for(uint32_t tile=0;tile<pairs&&ok;tile++){
        uint32_t expert=tiles[(uint64_t)tile*FG_VK_PREFILL_TILE_WORDS];
        if(expert==UINT32_MAX)continue;
        if(expert>=FG_EXPERT_COUNT){ok=0;break;}
        for(uint32_t slot=0;slot<FG_VK_PREFILL_PAIR_TILE;slot++){
            uint32_t pair=tiles[(uint64_t)tile*FG_VK_PREFILL_TILE_WORDS+1u+slot];
            if(pair==UINT32_MAX)continue;
            if(pair>=pairs||seen[pair]||selected[pair]!=expert){ok=0;break;}
            seen[pair]=true;
        }
    }
    for(uint32_t pair=0;pair<pairs;pair++)if(!seen[pair])ok=0;
    free(seen);return ok;
}

static int test_decode_tile_schedule(void){
    float logits[FG_EXPERT_COUNT];
    for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
        logits[expert]=expert<20u?7.0f:-20.0f-(float)expert*0.001f;
    uint32_t selected[FG_TOP_K],reference[FG_TOP_K];
    float gates[FG_TOP_K],reference_gates[FG_TOP_K];
    uint32_t tiles[FG_TOP_K*9u];
    fg_vk_tensor *gl=tensor(logits,sizeof(logits));
    fg_vk_tensor *gs=tensor(NULL,sizeof(selected));
    fg_vk_tensor *gg=tensor(NULL,sizeof(gates));
    fg_vk_tensor *gt=tensor(NULL,sizeof(tiles));
    fg_vk_counters before={0},after={0};
    fg_vk_get_counters(context,&before);
    int ok=gl&&gs&&gg&&gt&&fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_router_top10(context,gs,gg,gl,FG_EXPERT_COUNT,1u,&error)==FG_OK&&
        fg_vk_decode_tile_schedule(context,gt,gs,&error)==FG_OK&&
        fg_vk_end(context,&error)==FG_OK&&
        fg_vk_tensor_read(gs,0,selected,sizeof(selected),&error)==FG_OK&&
        fg_vk_tensor_read(gg,0,gates,sizeof(gates),&error)==FG_OK&&
        fg_vk_tensor_read(gt,0,tiles,sizeof(tiles),&error)==FG_OK&&
        fg_q38_router_topk(logits,FG_EXPERT_COUNT,FG_TOP_K,reference,
                           reference_gates,&error)==FG_OK;
    fg_vk_get_counters(context,&after);
    if(ok)ok=after.submissions==before.submissions+1u&&
             after.dispatches==before.dispatches+2u;
    for(uint32_t slot=0;ok&&slot<FG_TOP_K;slot++){
        ok=selected[slot]==reference[slot]&&selected[slot]==slot&&
           fabsf(gates[slot]-reference_gates[slot])<2e-6f&&
           tiles[slot*9u]==selected[slot]&&tiles[slot*9u+1u]==slot;
        for(uint32_t word=2u;ok&&word<9u;word++)
            ok=tiles[slot*9u+word]==UINT32_MAX;
    }
    if(fg_vk_batch_active(context)){
        fg_error ignored={0};fg_vk_abort(context,&ignored);
    }
    fg_vk_tensor_destroy(gt);fg_vk_tensor_destroy(gg);
    fg_vk_tensor_destroy(gs);fg_vk_tensor_destroy(gl);
    return ok;
}

static int test_router_and_expert_packing(void){
    const uint32_t token_counts[]={1u,2u,32u,64u,128u};
    for(uint32_t pattern=0;pattern<3u;pattern++)
        for(uint32_t count=0;count<5u;count++){
            uint32_t tokens=token_counts[count],pairs=tokens*FG_TOP_K;
            float *logits=malloc((size_t)tokens*FG_EXPERT_COUNT*4u);
            uint32_t *ids=malloc((size_t)pairs*4u);
            float *gates=malloc((size_t)pairs*4u);
            uint32_t *tiles=malloc((size_t)pairs*FG_VK_PREFILL_TILE_WORDS*4u);
            if(!logits||!ids||!gates||!tiles){
                free(tiles);free(gates);free(ids);free(logits);return 0;
            }
            for(uint32_t token=0;token<tokens;token++)
                for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++){
                    float value=0.0f;
                    if(pattern==0u)value=expert<FG_TOP_K?
                        20.0f-(float)expert:-20.0f;
                    else if(pattern==1u){
                        uint32_t center=(token*37u)%FG_EXPERT_COUNT;
                        uint32_t distance=(expert+FG_EXPERT_COUNT-center)%
                            FG_EXPERT_COUNT;
                        value=distance<FG_TOP_K?20.0f-(float)distance:
                            -20.0f-(float)distance*0.001f;
                    }
                    logits[(uint64_t)token*FG_EXPERT_COUNT+expert]=value;
                }
            fg_vk_tensor *gl=tensor(logits,(uint64_t)tokens*FG_EXPERT_COUNT*4u);
            fg_vk_tensor *gi=tensor(NULL,(uint64_t)pairs*4u);
            fg_vk_tensor *gg=tensor(NULL,(uint64_t)pairs*4u);
            fg_vk_tensor *gt=tensor(NULL,(uint64_t)pairs*
                                    FG_VK_PREFILL_TILE_WORDS*4u);
            int ok=gl&&gi&&gg&&gt&&fg_vk_begin(context,&error)==FG_OK&&
                fg_vk_router_top10(context,gi,gg,gl,FG_EXPERT_COUNT,tokens,
                                   &error)==FG_OK&&
                fg_vk_expert_major_pack(context,gt,gi,FG_EXPERT_COUNT,tokens,
                                        &error)==FG_OK&&
                fg_vk_end(context,&error)==FG_OK&&
                fg_vk_tensor_read(gi,0,ids,(uint64_t)pairs*4u,&error)==FG_OK&&
                fg_vk_tensor_read(gg,0,gates,(uint64_t)pairs*4u,&error)==FG_OK&&
                fg_vk_tensor_read(gt,0,tiles,(uint64_t)pairs*
                    FG_VK_PREFILL_TILE_WORDS*4u,&error)==FG_OK;
            for(uint32_t token=0;ok&&token<tokens;token++){
                uint32_t reference_ids[FG_TOP_K];float reference_gates[FG_TOP_K];
                ok=fg_q38_router_topk(logits+(uint64_t)token*FG_EXPERT_COUNT,
                    FG_EXPERT_COUNT,FG_TOP_K,reference_ids,reference_gates,
                    &error)==FG_OK;
                for(uint32_t slot=0;ok&&slot<FG_TOP_K;slot++){
                    uint32_t at=token*FG_TOP_K+slot;
                    if(ids[at]!=reference_ids[slot]||
                       fabsf(gates[at]-reference_gates[slot])>2e-6f){
                        fprintf(stderr,
                            "router pattern=%u tokens=%u token=%u slot=%u GPU=(%u,%g) CPU=(%u,%g)\n",
                            pattern,tokens,token,slot,ids[at],gates[at],
                            reference_ids[slot],reference_gates[slot]);
                        ok=0;
                    }
                }
            }
            if(ok)ok=verify_expert_tiles(ids,tiles,pairs);
            fg_vk_tensor_destroy(gt);fg_vk_tensor_destroy(gg);
            fg_vk_tensor_destroy(gi);fg_vk_tensor_destroy(gl);
            free(tiles);free(gates);free(ids);free(logits);
            if(!ok)return 0;
        }
    {
        enum{TOKENS=128,PAIRS=TOKENS*FG_TOP_K};
        uint32_t *selected=calloc(PAIRS,sizeof(*selected));
        uint32_t *tiles=malloc((size_t)PAIRS*FG_VK_PREFILL_TILE_WORDS*4u);
        fg_vk_tensor *gs=selected?tensor(selected,sizeof(*selected)*PAIRS):NULL;
        fg_vk_tensor *gt=gs?tensor(NULL,
            (uint64_t)PAIRS*FG_VK_PREFILL_TILE_WORDS*4u):NULL;
        int ok=gs&&gt&&tiles&&fg_vk_expert_major_pack(context,gt,gs,
            FG_EXPERT_COUNT,TOKENS,&error)==FG_OK&&
            fg_vk_tensor_read(gt,0,tiles,
                (uint64_t)PAIRS*FG_VK_PREFILL_TILE_WORDS*4u,&error)==FG_OK&&
            verify_expert_tiles(selected,tiles,PAIRS);
        fg_vk_tensor_destroy(gt);fg_vk_tensor_destroy(gs);
        free(tiles);free(selected);
        if(!ok)return 0;
    }
    return 1;
}

static int test_grouped_kquant_prefill(int type){
    const uint32_t token_counts[]={1u,2u,32u,64u,128u};
    enum{INPUT=2560,OUTPUT=8,EXPERTS=2};
    uint32_t block_bytes=type==12?144u:176u,blocks=INPUT/FG_QK8_K;
    uint32_t row_bytes=blocks*block_bytes,expert_stride=OUTPUT*row_bytes;
    uint8_t *raw=calloc(EXPERTS,expert_stride);
    uint8_t *cooked=malloc((size_t)EXPERTS*expert_stride);
    if(!raw||!cooked){free(cooked);free(raw);return 0;}
    int ok=1;
    for(uint32_t expert=0;expert<EXPERTS;expert++){
        for(uint32_t row=0;row<OUTPUT;row++)
            for(uint32_t block=0;block<blocks;block++)
                make_k_row(raw+(uint64_t)expert*expert_stride+
                    (uint64_t)row*row_bytes+(uint64_t)block*block_bytes,
                    type==13,expert*41u+row*13u+block);
        ok=ok&&fg_cook_k_quant_rows(raw+(uint64_t)expert*expert_stride,
            cooked+(uint64_t)expert*expert_stride,expert_stride,INPUT,OUTPUT,
            (uint32_t)type);
    }
    fg_vk_tensor *weights=ok?tensor(cooked,(uint64_t)EXPERTS*expert_stride):NULL;
    if(weights)fg_vk_tensor_set_format(weights,
        FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED);
    for(uint32_t count=0;ok&&count<5u;count++){
        uint32_t tokens=token_counts[count],pairs=tokens*FG_TOP_K;
        float *input=malloc((size_t)tokens*INPUT*4u);
        uint8_t *activation=malloc((size_t)tokens*blocks*FG_Q8_K_BLOCK_BYTES);
        uint32_t *selected=malloc((size_t)pairs*4u);
        float *got=malloc((size_t)pairs*OUTPUT*4u);
        if(!input||!activation||!selected||!got){
            free(got);free(selected);free(activation);free(input);ok=0;break;
        }
        for(uint32_t token=0;token<tokens;token++){
            for(uint32_t i=0;i<INPUT;i++)
                input[(uint64_t)token*INPUT+i]=
                    sinf((float)(token+1u)*(float)(i+5u)*0.0021f);
            fg_quantize_q8_k(input+(uint64_t)token*INPUT,
                activation+(uint64_t)token*blocks*FG_Q8_K_BLOCK_BYTES,INPUT);
            for(uint32_t slot=0;slot<FG_TOP_K;slot++)
                selected[token*FG_TOP_K+slot]=(token+slot)&1u;
        }
        fg_vk_tensor *ga=tensor(activation,
            (uint64_t)tokens*blocks*FG_Q8_K_BLOCK_BYTES);
        fg_vk_tensor *gs=tensor(selected,(uint64_t)pairs*4u);
        fg_vk_tensor *gt=tensor(NULL,(uint64_t)pairs*
            FG_VK_PREFILL_TILE_WORDS*4u);
        fg_vk_tensor *go=tensor(NULL,(uint64_t)pairs*OUTPUT*4u);
        ok=ga&&gs&&gt&&go&&fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_expert_major_pack(context,gt,gs,FG_EXPERT_COUNT,tokens,
                                    &error)==FG_OK&&
            fg_vk_moe_kquant_cooked_grouped(context,go,weights,ga,gt,
                (uint32_t)type,OUTPUT,INPUT,expert_stride,EXPERTS,tokens,
                &error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&
            fg_vk_tensor_read(go,0,got,(uint64_t)pairs*OUTPUT*4u,
                              &error)==FG_OK;
        for(uint32_t pair=0;ok&&pair<pairs;pair++){
            uint32_t expert=selected[pair],token=pair/FG_TOP_K;
            const uint8_t *q8=activation+
                (uint64_t)token*blocks*FG_Q8_K_BLOCK_BYTES;
            for(uint32_t row=0;row<OUTPUT;row++){
                const uint8_t *weight=raw+(uint64_t)expert*expert_stride+
                    (uint64_t)row*row_bytes;
                float reference=type==12?fg_dot_q4_k_q8_k(weight,q8,INPUT):
                    fg_dot_q5_k_q8_k(weight,q8,INPUT);
                if(fabsf(got[(uint64_t)pair*OUTPUT+row]-reference)>
                   2e-3f*fmaxf(1.0f,fabsf(reference))){
                    fprintf(stderr,
                        "grouped K type=%d tokens=%u pair=%u row=%u GPU=%g CPU=%g\n",
                        type,tokens,pair,row,
                        got[(uint64_t)pair*OUTPUT+row],reference);
                    ok=0;break;
                }
            }
        }
        fg_vk_tensor_destroy(go);fg_vk_tensor_destroy(gt);
        fg_vk_tensor_destroy(gs);fg_vk_tensor_destroy(ga);
        free(got);free(selected);free(activation);free(input);
    }
    fg_vk_tensor_destroy(weights);free(cooked);free(raw);return ok;
}

static int test_grouped_down_prefill(void){
    const uint32_t token_counts[]={1u,2u,32u,64u,128u};
    enum{INPUT=640,OUTPUT=16,EXPERTS=2};
    uint32_t q5_row=(INPUT/FG_QK8_0)*FG_Q5_1_BLOCK_BYTES;
    uint32_t q8_row=(INPUT/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES;
    uint32_t q5_stride=OUTPUT*q5_row,q8_stride=OUTPUT*q8_row;
    uint8_t *q5=malloc((size_t)EXPERTS*q5_stride);
    uint8_t *q5_cooked=malloc((size_t)EXPERTS*q5_stride);
    uint8_t *q8=malloc((size_t)EXPERTS*q8_stride);
    if(!q5||!q5_cooked||!q8){
        free(q8);free(q5_cooked);free(q5);return 0;
    }
    int ok=1;
    for(uint32_t expert=0;expert<EXPERTS;expert++){
        for(uint32_t row=0;row<OUTPUT;row++){
            make_q5_1_row(q5+(uint64_t)expert*q5_stride+
                (uint64_t)row*q5_row,INPUT,expert*31u+row);
            for(uint32_t block=0;block<INPUT/FG_QK8_0;block++){
                uint8_t *value=q8+(uint64_t)expert*q8_stride+
                    (uint64_t)row*q8_row+
                    (uint64_t)block*FG_Q8_0_BLOCK_BYTES;
                uint16_t delta=fg_f32_to_f16(
                    0.001f+(float)((expert+row+block)%9u)*0.0002f);
                memcpy(value,&delta,sizeof(delta));
                for(uint32_t i=0;i<FG_QK8_0;i++)
                    value[2u+i]=(uint8_t)(expert*43u+row*17u+block*7u+i);
            }
        }
        ok=ok&&fg_cook_q5_1_rows(q5+(uint64_t)expert*q5_stride,
            q5_cooked+(uint64_t)expert*q5_stride,q5_stride,INPUT,OUTPUT);
    }
    fg_vk_tensor *gq5=ok?tensor(q5_cooked,(uint64_t)EXPERTS*q5_stride):NULL;
    fg_vk_tensor *gq8=ok?tensor(q8,(uint64_t)EXPERTS*q8_stride):NULL;
    if(gq5)fg_vk_tensor_set_format(gq5,
                                    FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED);
    for(uint32_t count=0;ok&&count<5u;count++){
        uint32_t tokens=token_counts[count],pairs=tokens*FG_TOP_K;
        float *input=malloc((size_t)pairs*INPUT*4u);
        uint32_t *selected=malloc((size_t)pairs*4u);
        float *got_q5=malloc((size_t)pairs*OUTPUT*4u);
        float *got_q8=malloc((size_t)pairs*OUTPUT*4u);
        if(!input||!selected||!got_q5||!got_q8){
            free(got_q8);free(got_q5);free(selected);free(input);ok=0;break;
        }
        for(uint32_t pair=0;pair<pairs;pair++){
            selected[pair]=(pair/FG_TOP_K+pair%FG_TOP_K)&1u;
            for(uint32_t i=0;i<INPUT;i++)
                input[(uint64_t)pair*INPUT+i]=
                    cosf((float)(pair+1u)*(float)(i+3u)*0.0031f);
        }
        fg_vk_tensor *gi=tensor(input,(uint64_t)pairs*INPUT*4u);
        fg_vk_tensor *gs=tensor(selected,(uint64_t)pairs*4u);
        fg_vk_tensor *gt=tensor(NULL,(uint64_t)pairs*
            FG_VK_PREFILL_TILE_WORDS*4u);
        fg_vk_tensor *o5=tensor(NULL,(uint64_t)pairs*OUTPUT*4u);
        fg_vk_tensor *o8=tensor(NULL,(uint64_t)pairs*OUTPUT*4u);
        ok=gi&&gs&&gt&&o5&&o8&&fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_expert_major_pack(context,gt,gs,FG_EXPERT_COUNT,tokens,
                                    &error)==FG_OK&&
            fg_vk_moe_q5_1_down_cooked_grouped(context,o5,gq5,gt,gi,
                OUTPUT,INPUT,q5_stride,EXPERTS,tokens,&error)==FG_OK&&
            fg_vk_moe_q8_0_down_grouped(context,o8,gq8,gt,gi,OUTPUT,INPUT,
                q8_stride,EXPERTS,tokens,&error)==FG_OK&&
            fg_vk_end(context,&error)==FG_OK&&
            fg_vk_tensor_read(o5,0,got_q5,(uint64_t)pairs*OUTPUT*4u,
                              &error)==FG_OK&&
            fg_vk_tensor_read(o8,0,got_q8,(uint64_t)pairs*OUTPUT*4u,
                              &error)==FG_OK;
        for(uint32_t pair=0;ok&&pair<pairs;pair++)
            for(uint32_t row=0;row<OUTPUT;row++){
                uint32_t expert=selected[pair];
                float ref5=fg_q5_1_dot_f32(q5+(uint64_t)expert*q5_stride+
                    (uint64_t)row*q5_row,input+(uint64_t)pair*INPUT,INPUT);
                float ref8=q8_row_reference(q8+(uint64_t)expert*q8_stride+
                    (uint64_t)row*q8_row,input+(uint64_t)pair*INPUT,INPUT);
                if(fabsf(got_q5[(uint64_t)pair*OUTPUT+row]-ref5)>
                       1e-3f*fmaxf(1.0f,fabsf(ref5))||
                   fabsf(got_q8[(uint64_t)pair*OUTPUT+row]-ref8)>
                       1e-3f*fmaxf(1.0f,fabsf(ref8))){
                    fprintf(stderr,
                        "grouped down tokens=%u pair=%u row=%u Q5=(%g,%g) Q8=(%g,%g)\n",
                        tokens,pair,row,got_q5[(uint64_t)pair*OUTPUT+row],
                        ref5,got_q8[(uint64_t)pair*OUTPUT+row],ref8);
                    ok=0;break;
                }
            }
        fg_vk_tensor_destroy(o8);fg_vk_tensor_destroy(o5);
        fg_vk_tensor_destroy(gt);fg_vk_tensor_destroy(gs);
        fg_vk_tensor_destroy(gi);
        free(got_q8);free(got_q5);free(selected);free(input);
    }
    fg_vk_tensor_destroy(gq8);fg_vk_tensor_destroy(gq5);
    free(q8);free(q5_cooked);free(q5);return ok;
}

static int test_moe_prefill_scatter_reduce(void){
    const uint32_t token_counts[]={1u,2u,32u,64u,128u};
    enum{WIDTH=2560};
    for(uint32_t count=0;count<5u;count++){
        uint32_t tokens=token_counts[count],pairs=tokens*FG_TOP_K;
        float *expert=malloc((size_t)pairs*WIDTH*4u);
        float *gates=malloc((size_t)pairs*4u);
        float *shared=malloc((size_t)tokens*WIDTH*4u);
        float *logit=malloc((size_t)tokens*4u);
        float *got=malloc((size_t)tokens*WIDTH*4u);
        if(!expert||!gates||!shared||!logit||!got){
            free(got);free(logit);free(shared);free(gates);free(expert);
            return 0;
        }
        for(uint32_t token=0;token<tokens;token++){
            logit[token]=(float)((int32_t)(token%7u)-3)*0.2f;
            for(uint32_t slot=0;slot<FG_TOP_K;slot++)
                gates[token*FG_TOP_K+slot]=(float)(slot+1u)/55.0f;
            for(uint32_t i=0;i<WIDTH;i++){
                shared[(uint64_t)token*WIDTH+i]=
                    sinf((float)(token*WIDTH+i)*0.0007f);
                for(uint32_t slot=0;slot<FG_TOP_K;slot++)
                    expert[((uint64_t)token*FG_TOP_K+slot)*WIDTH+i]=
                        cosf((float)((token*FG_TOP_K+slot)*WIDTH+i)*0.0003f);
            }
        }
        fg_vk_tensor *ge=tensor(expert,(uint64_t)pairs*WIDTH*4u);
        fg_vk_tensor *gg=tensor(gates,(uint64_t)pairs*4u);
        fg_vk_tensor *gs=tensor(shared,(uint64_t)tokens*WIDTH*4u);
        fg_vk_tensor *gl=tensor(logit,(uint64_t)tokens*4u);
        fg_vk_tensor *go=tensor(NULL,(uint64_t)tokens*WIDTH*4u);
        int ok=ge&&gg&&gs&&gl&&go&&fg_vk_moe_prefill_reduce(context,go,ge,
            gg,gs,gl,WIDTH,tokens,&error)==FG_OK&&fg_vk_tensor_read(go,0,got,
            (uint64_t)tokens*WIDTH*4u,&error)==FG_OK;
        for(uint32_t token=0;ok&&token<tokens;token++)
            for(uint32_t i=0;i<WIDTH;i++){
                float reference=shared[(uint64_t)token*WIDTH+i]/
                    (1.0f+expf(-logit[token]));
                for(uint32_t slot=0;slot<FG_TOP_K;slot++)
                    reference=fmaf(gates[token*FG_TOP_K+slot],
                        expert[((uint64_t)token*FG_TOP_K+slot)*WIDTH+i],
                        reference);
                if(fabsf(got[(uint64_t)token*WIDTH+i]-reference)>
                   2e-6f*fmaxf(1.0f,fabsf(reference))){ok=0;break;}
            }
        fg_vk_tensor_destroy(go);fg_vk_tensor_destroy(gl);
        fg_vk_tensor_destroy(gs);fg_vk_tensor_destroy(gg);
        fg_vk_tensor_destroy(ge);
        free(got);free(logit);free(shared);free(gates);free(expert);
        if(!ok)return 0;
    }
    return 1;
}

static int test_expert_graph_replay(void){
    enum{HIDDEN=256,MID=32,SLOTS=4,EXPERTS=2,PASSES=2};uint32_t gate_block=144u,up_block=176u,down_row=(MID/32u)*FG_Q8_0_BLOCK_BYTES,gate_stride=MID*gate_block,up_stride=MID*up_block,down_stride=HIDDEN*down_row;uint8_t *gate_weights=calloc(EXPERTS,gate_stride),*up_weights=calloc(EXPERTS,up_stride),*down_weights=malloc((size_t)EXPERTS*down_stride);float *down_source=malloc((size_t)EXPERTS*HIDDEN*MID*4u);if(!gate_weights||!up_weights||!down_weights||!down_source){free(down_source);free(down_weights);free(up_weights);free(gate_weights);return 0;}
    for(uint32_t expert=0;expert<EXPERTS;expert++){for(uint32_t row=0;row<MID;row++){make_k_row(gate_weights+(uint64_t)expert*gate_stride+row*gate_block,0,expert*13u+row);make_k_row(up_weights+(uint64_t)expert*up_stride+row*up_block,1,expert*17u+row);}for(uint32_t row=0;row<HIDDEN;row++){float *source=down_source+((uint64_t)expert*HIDDEN+row)*MID;for(uint32_t i=0;i<MID;i++)source[i]=sinf((float)(expert*HIDDEN*MID+row*MID+i)*0.009f);fg_quantize_q8_0(source,down_weights+(uint64_t)expert*down_stride+row*down_row,MID);}}
    fg_vk_tensor *gw=tensor(gate_weights,(uint64_t)EXPERTS*gate_stride),*uw=tensor(up_weights,(uint64_t)EXPERTS*up_stride),*dw=tensor(down_weights,(uint64_t)EXPERTS*down_stride),*activation=tensor(NULL,FG_Q8_K_BLOCK_BYTES),*tiles=tensor(NULL,SLOTS*9u*4u),*gates=tensor(NULL,SLOTS*4u),*gate=tensor(NULL,SLOTS*MID*4u),*up=tensor(NULL,SLOTS*MID*4u),*mid=tensor(NULL,SLOTS*MID*4u),*down=tensor(NULL,SLOTS*HIDDEN*4u),*reduced=tensor(NULL,HIDDEN*4u);fg_vk_expert_graph *graph=NULL;int ok=gw&&uw&&dw&&activation&&tiles&&gates&&gate&&up&&mid&&down&&reduced&&fg_vk_expert_graph_create(context,&graph,activation,tiles,gates,gate,up,mid,down,reduced,gw,uw,dw,12u,13u,8u,HIDDEN,MID,gate_stride,up_stride,down_stride,EXPERTS,SLOTS,&error)==FG_OK;
    for(uint32_t pass=0;ok&&pass<PASSES;pass++){uint32_t selected_count=pass+1u;float input[HIDDEN],gate_values[SLOTS]={0},expected[HIDDEN],got[HIDDEN],poison_gate[SLOTS*MID],poison_up[SLOTS*MID],poison_mid[SLOTS*MID],poison_down[SLOTS*HIDDEN],poison_reduced[HIDDEN];uint8_t q8[FG_Q8_K_BLOCK_BYTES];uint32_t schedule[SLOTS*9u];for(uint32_t i=0;i<SLOTS*9u;i++)schedule[i]=UINT32_MAX;for(uint32_t i=0;i<HIDDEN;i++){input[i]=sinf((float)(pass+1u)*(float)(i+3u)*0.017f)+0.1f*cosf((float)(i+pass)*0.031f);poison_reduced[i]=1234.0f;for(uint32_t slot=0;slot<SLOTS;slot++)poison_down[slot*HIDDEN+i]=1234.0f;}for(uint32_t i=0;i<SLOTS*MID;i++){poison_gate[i]=1234.0f;poison_up[i]=1234.0f;poison_mid[i]=1234.0f;}fg_quantize_q8_k(input,q8,HIDDEN);for(uint32_t selected=0;selected<selected_count;selected++){schedule[selected*9u]=(selected+pass)%EXPERTS;schedule[selected*9u+1u]=(selected*2u+pass)%SLOTS;gate_values[selected]=(pass?0.65f:-0.35f)+(float)selected*0.4f;}ok=fg_vk_tensor_write(activation,0,q8,sizeof(q8),&error)==FG_OK&&fg_vk_tensor_write(tiles,0,schedule,sizeof(schedule),&error)==FG_OK&&fg_vk_tensor_write(gates,0,gate_values,sizeof(gate_values),&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK&&fg_vk_moe_kquant(context,gate,gw,activation,tiles,12u,MID,HIDDEN,gate_stride,SLOTS,SLOTS,false,selected_count,&error)==FG_OK&&fg_vk_moe_kquant(context,up,uw,activation,tiles,13u,MID,HIDDEN,up_stride,SLOTS,SLOTS,false,selected_count,&error)==FG_OK&&fg_vk_swiglu(context,mid,gate,up,SLOTS*MID,&error)==FG_OK&&fg_vk_moe_q8_0_down(context,down,dw,tiles,mid,HIDDEN,MID,down_stride,SLOTS,false,selected_count,&error)==FG_OK&&fg_vk_moe_reduce(context,reduced,down,gates,tiles,HIDDEN,selected_count,SLOTS,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&fg_vk_tensor_read(reduced,0,expected,sizeof(expected),&error)==FG_OK&&fg_vk_tensor_write(gate,0,poison_gate,sizeof(poison_gate),&error)==FG_OK&&fg_vk_tensor_write(up,0,poison_up,sizeof(poison_up),&error)==FG_OK&&fg_vk_tensor_write(mid,0,poison_mid,sizeof(poison_mid),&error)==FG_OK&&fg_vk_tensor_write(down,0,poison_down,sizeof(poison_down),&error)==FG_OK&&fg_vk_tensor_write(reduced,0,poison_reduced,sizeof(poison_reduced),&error)==FG_OK;fg_vk_counters before,after;fg_vk_get_counters(context,&before);if(ok)ok=fg_vk_expert_graph_execute(graph,&error)==FG_OK&&fg_vk_tensor_read(reduced,0,got,sizeof(got),&error)==FG_OK;fg_vk_get_counters(context,&after);if(ok&&(after.submissions!=before.submissions+1u||after.dispatches!=before.dispatches+5u)){fprintf(stderr,"expert graph counters submissions=%llu dispatches=%llu\n",(unsigned long long)(after.submissions-before.submissions),(unsigned long long)(after.dispatches-before.dispatches));ok=0;}for(uint32_t i=0;ok&&i<HIDDEN;i++)if(got[i]!=expected[i]){fprintf(stderr,"expert graph pass %u value %u replay=%g dynamic=%g diff=%g\n",pass,i,got[i],expected[i],got[i]-expected[i]);ok=0;}}
    fg_vk_expert_graph_destroy(graph);fg_vk_tensor_destroy(reduced);fg_vk_tensor_destroy(down);fg_vk_tensor_destroy(mid);fg_vk_tensor_destroy(up);fg_vk_tensor_destroy(gate);fg_vk_tensor_destroy(gates);fg_vk_tensor_destroy(tiles);fg_vk_tensor_destroy(activation);fg_vk_tensor_destroy(dw);fg_vk_tensor_destroy(uw);fg_vk_tensor_destroy(gw);free(down_source);free(down_weights);free(up_weights);free(gate_weights);return ok;
}

static int test_expert_graph_rejects_overlap(void){
    enum{HIDDEN=256,MID=32,SLOTS=4,EXPERTS=2};
    uint32_t gate_stride=MID*144u,up_stride=MID*176u;
    uint32_t down_stride=HIDDEN*(MID/32u)*FG_Q8_0_BLOCK_BYTES;
    uint64_t phase=(uint64_t)SLOTS*MID*4u;
    fg_vk_tensor *gw=tensor(NULL,(uint64_t)EXPERTS*gate_stride);
    fg_vk_tensor *uw=tensor(NULL,(uint64_t)EXPERTS*up_stride);
    fg_vk_tensor *dw=tensor(NULL,(uint64_t)EXPERTS*down_stride);
    fg_vk_tensor *activation=tensor(NULL,FG_Q8_K_BLOCK_BYTES);
    fg_vk_tensor *tiles=tensor(NULL,(uint64_t)SLOTS*9u*4u);
    fg_vk_tensor *gates=tensor(NULL,(uint64_t)SLOTS*4u);
    fg_vk_tensor *arena=tensor(NULL,phase*2u);
    fg_vk_tensor *gate=NULL,*up=NULL,*mid=NULL;
    fg_vk_tensor *down=tensor(NULL,(uint64_t)SLOTS*HIDDEN*4u);
    fg_vk_tensor *reduced=tensor(NULL,(uint64_t)HIDDEN*4u);
    fg_vk_expert_graph *graph=NULL;
    int ok=gw&&uw&&dw&&activation&&tiles&&gates&&arena&&down&&reduced;
    if(ok)ok=fg_vk_tensor_view(arena,0u,phase,&gate,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_view(arena,phase,phase,&up,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_view(arena,0u,phase,&mid,&error)==FG_OK;
    if(ok)ok=fg_vk_expert_graph_create(context,&graph,activation,tiles,gates,
        gate,up,mid,down,reduced,gw,uw,dw,12u,13u,8u,HIDDEN,MID,
        gate_stride,up_stride,down_stride,EXPERTS,SLOTS,&error)==FG_ERR_MISMATCH&&
        graph==NULL;
    fg_vk_tensor_destroy(reduced);fg_vk_tensor_destroy(down);
    fg_vk_tensor_destroy(mid);fg_vk_tensor_destroy(up);fg_vk_tensor_destroy(gate);
    fg_vk_tensor_destroy(arena);fg_vk_tensor_destroy(gates);fg_vk_tensor_destroy(tiles);
    fg_vk_tensor_destroy(activation);fg_vk_tensor_destroy(dw);fg_vk_tensor_destroy(uw);
    fg_vk_tensor_destroy(gw);
    return ok;
}

static float softplus_ref(float value){return fmaxf(value,0.0f)+log1pf(expf(-fabsf(value)));}
static int test_gdn_project_cooked(void){
    enum{INPUT=2560,QKV=10240,Z=6144,CONTROL=48,BLOCKS=INPUT/FG_QK8_0};uint32_t source_row=BLOCKS*FG_Q8_0_BLOCK_BYTES;uint64_t qkv_source_bytes=(uint64_t)QKV*source_row,z_source_bytes=(uint64_t)Z*source_row,qkv_cooked_bytes=fg_q8_0_cooked_matrix_bytes(INPUT,QKV),z_cooked_bytes=fg_q8_0_cooked_matrix_bytes(INPUT,Z);uint8_t *qkv_source=malloc((size_t)qkv_source_bytes),*z_source=malloc((size_t)z_source_bytes),*qkv_cooked=malloc((size_t)qkv_cooked_bytes),*z_cooked=malloc((size_t)z_cooked_bytes);float hidden[INPUT],*alpha_weight=malloc((size_t)INPUT*CONTROL*4u),*beta_weight=malloc((size_t)INPUT*CONTROL*4u),*qkv_reference=malloc((size_t)QKV*4u),*qkv_candidate=malloc((size_t)QKV*4u),*z_reference=malloc((size_t)Z*4u),*z_candidate=malloc((size_t)Z*4u),alpha_reference[CONTROL],alpha_candidate[CONTROL],beta_reference[CONTROL],beta_candidate[CONTROL];if(!qkv_source||!z_source||!qkv_cooked||!z_cooked||!alpha_weight||!beta_weight||!qkv_reference||!qkv_candidate||!z_reference||!z_candidate){free(z_candidate);free(z_reference);free(qkv_candidate);free(qkv_reference);free(beta_weight);free(alpha_weight);free(z_cooked);free(qkv_cooked);free(z_source);free(qkv_source);return 0;}for(uint32_t i=0;i<INPUT;i++)hidden[i]=sinf((float)(i+3u)*0.013f)+0.1f*cosf((float)i*0.007f);for(uint32_t i=0;i<INPUT*CONTROL;i++){alpha_weight[i]=sinf((float)i*0.0009f)*0.01f;beta_weight[i]=cosf((float)i*0.0011f)*0.01f;}uint8_t *sources[]={qkv_source,z_source};uint32_t rows[]={QKV,Z};for(uint32_t matrix=0;matrix<2u;matrix++)for(uint32_t row=0;row<rows[matrix];row++)for(uint32_t block=0;block<BLOCKS;block++){uint8_t *value=sources[matrix]+((uint64_t)row*BLOCKS+block)*FG_Q8_0_BLOCK_BYTES;uint16_t scale=fg_f32_to_f16(0.0005f+(float)((row+block+matrix)%13u)*0.00005f);memcpy(value,&scale,sizeof(scale));for(uint32_t i=0;i<FG_QK8_0;i++)value[2u+i]=(uint8_t)(row*17u+block*29u+i*11u+matrix*7u);}int ok=fg_cook_q8_0_rows(qkv_source,qkv_cooked,qkv_cooked_bytes,INPUT,QKV)&&fg_cook_q8_0_rows(z_source,z_cooked,z_cooked_bytes,INPUT,Z);fg_vk_tensor *gq_ref=ok?tensor(qkv_source,qkv_source_bytes):NULL,*gz_ref=ok?tensor(z_source,z_source_bytes):NULL,*gq_new=ok?tensor(qkv_cooked,qkv_cooked_bytes):NULL,*gz_new=ok?tensor(z_cooked,z_cooked_bytes):NULL,*gah=ok?tensor(alpha_weight,(uint64_t)INPUT*CONTROL*4u):NULL,*gbh=ok?tensor(beta_weight,(uint64_t)INPUT*CONTROL*4u):NULL,*gh=ok?tensor(hidden,sizeof(hidden)):NULL,*q_ref=ok?tensor(NULL,(uint64_t)QKV*4u):NULL,*z_ref=ok?tensor(NULL,(uint64_t)Z*4u):NULL,*a_ref=ok?tensor(NULL,sizeof(alpha_reference)):NULL,*b_ref=ok?tensor(NULL,sizeof(beta_reference)):NULL,*q_new=ok?tensor(NULL,(uint64_t)QKV*4u):NULL,*z_new=ok?tensor(NULL,(uint64_t)Z*4u):NULL,*a_new=ok?tensor(NULL,sizeof(alpha_candidate)):NULL,*b_new=ok?tensor(NULL,sizeof(beta_candidate)):NULL;if(gq_new)fg_vk_tensor_set_format(gq_new,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);if(gz_new)fg_vk_tensor_set_format(gz_new,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);ok=ok&&gq_ref&&gz_ref&&gq_new&&gz_new&&gah&&gbh&&gh&&q_ref&&z_ref&&a_ref&&b_ref&&q_new&&z_new&&a_new&&b_new&&fg_vk_begin(context,&error)==FG_OK&&fg_vk_gdn_project_decode(context,q_ref,z_ref,a_ref,b_ref,gq_ref,gz_ref,gah,gbh,gh,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK&&fg_vk_gdn_project_decode(context,q_new,z_new,a_new,b_new,gq_new,gz_new,gah,gbh,gh,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&fg_vk_tensor_read(q_ref,0,qkv_reference,(uint64_t)QKV*4u,&error)==FG_OK&&fg_vk_tensor_read(q_new,0,qkv_candidate,(uint64_t)QKV*4u,&error)==FG_OK&&fg_vk_tensor_read(z_ref,0,z_reference,(uint64_t)Z*4u,&error)==FG_OK&&fg_vk_tensor_read(z_new,0,z_candidate,(uint64_t)Z*4u,&error)==FG_OK&&fg_vk_tensor_read(a_ref,0,alpha_reference,sizeof(alpha_reference),&error)==FG_OK&&fg_vk_tensor_read(a_new,0,alpha_candidate,sizeof(alpha_candidate),&error)==FG_OK&&fg_vk_tensor_read(b_ref,0,beta_reference,sizeof(beta_reference),&error)==FG_OK&&fg_vk_tensor_read(b_new,0,beta_candidate,sizeof(beta_candidate),&error)==FG_OK;for(uint32_t i=0;ok&&i<QKV;i++)if(fabsf(qkv_candidate[i]-qkv_reference[i])>2e-4f*fmaxf(1.0f,fabsf(qkv_reference[i])))ok=0;for(uint32_t i=0;ok&&i<Z;i++)if(fabsf(z_candidate[i]-z_reference[i])>2e-4f*fmaxf(1.0f,fabsf(z_reference[i])))ok=0;for(uint32_t i=0;ok&&i<CONTROL;i++)if(alpha_candidate[i]!=alpha_reference[i]||beta_candidate[i]!=beta_reference[i])ok=0;fg_vk_tensor_destroy(b_new);fg_vk_tensor_destroy(a_new);fg_vk_tensor_destroy(z_new);fg_vk_tensor_destroy(q_new);fg_vk_tensor_destroy(b_ref);fg_vk_tensor_destroy(a_ref);fg_vk_tensor_destroy(z_ref);fg_vk_tensor_destroy(q_ref);fg_vk_tensor_destroy(gh);fg_vk_tensor_destroy(gbh);fg_vk_tensor_destroy(gah);fg_vk_tensor_destroy(gz_new);fg_vk_tensor_destroy(gq_new);fg_vk_tensor_destroy(gz_ref);fg_vk_tensor_destroy(gq_ref);free(z_candidate);free(z_reference);free(qkv_candidate);free(qkv_reference);free(beta_weight);free(alpha_weight);free(z_cooked);free(qkv_cooked);free(z_source);free(qkv_source);return ok;
}

static int test_gdn_decode(void){
    enum{VALUE_HEADS=48,KEY_HEADS=16,DIM=128,KEY_WIDTH=KEY_HEADS*DIM,VALUE_WIDTH=VALUE_HEADS*DIM,QKV_WIDTH=2*KEY_WIDTH+VALUE_WIDTH,STATE_VALUES=VALUE_HEADS*DIM*DIM};
    float *projection=malloc(QKV_WIDTH*4u),*conv_weight=malloc(QKV_WIDTH*4u*4u),*conv_state=malloc(QKV_WIDTH*4u*4u),*conv_expected=malloc(QKV_WIDTH*4u),*conv_got=malloc(QKV_WIDTH*4u);if(!projection||!conv_weight||!conv_state||!conv_expected||!conv_got)return 0;
    for(uint32_t i=0;i<QKV_WIDTH;i++){projection[i]=sinf((float)i*0.013f);for(uint32_t k=0;k<4u;k++){conv_weight[i*4u+k]=cosf((float)(i*4u+k)*0.017f)*0.2f;conv_state[i*4u+k]=sinf((float)(i*4u+k)*0.009f)*0.1f;}float value=conv_weight[i*4u]*conv_state[i*4u+1u]+conv_weight[i*4u+1u]*conv_state[i*4u+2u]+conv_weight[i*4u+2u]*conv_state[i*4u+3u]+conv_weight[i*4u+3u]*projection[i];conv_expected[i]=value/(1.0f+expf(-value));}
    fg_vk_tensor *gp=tensor(projection,QKV_WIDTH*4u),*gcw=tensor(conv_weight,QKV_WIDTH*16u),*gcs=tensor(conv_state,QKV_WIDTH*16u),*gco=tensor(NULL,QKV_WIDTH*4u);int ok=gp&&gcw&&gcs&&gco&&fg_vk_gdn_conv_decode(context,gco,gcs,gp,gcw,QKV_WIDTH,&error)==FG_OK&&fg_vk_tensor_read(gco,0,conv_got,QKV_WIDTH*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<QKV_WIDTH;i++)if(fabsf(conv_got[i]-conv_expected[i])>2e-6f){fprintf(stderr,"GDN conv %u GPU=%g CPU=%g\n",i,conv_got[i],conv_expected[i]);ok=0;}
    float *qkv=conv_got,*z=malloc(VALUE_WIDTH*4u),alpha[VALUE_HEADS],beta[VALUE_HEADS],a_decay[VALUE_HEADS],dt[VALUE_HEADS],norm[DIM],*state=malloc(STATE_VALUES*4u),*cpu_state=malloc(STATE_VALUES*4u),*got=malloc(VALUE_WIDTH*4u),*expected=malloc(VALUE_WIDTH*4u);if(!z||!state||!cpu_state||!got||!expected)ok=0;
    if(ok){for(uint32_t i=0;i<VALUE_WIDTH;i++)z[i]=cosf((float)i*0.011f);for(uint32_t h=0;h<VALUE_HEADS;h++){alpha[h]=sinf((float)h*0.1f);beta[h]=cosf((float)h*0.07f);a_decay[h]=-(0.02f+(float)(h+1u)*0.01f);dt[h]=-0.3f+(float)h*0.005f;}for(uint32_t i=0;i<DIM;i++)norm[i]=0.9f+0.2f*sinf((float)i*0.03f);for(uint32_t i=0;i<STATE_VALUES;i++)state[i]=sinf((float)i*0.0007f)*0.01f;memcpy(cpu_state,state,STATE_VALUES*4u);
        for(uint32_t h=0;h<VALUE_HEADS;h++){uint32_t kh=h%KEY_HEADS;float qnorm=0.0f,knorm=0.0f;for(uint32_t i=0;i<DIM;i++){float q=qkv[kh*DIM+i],k=qkv[KEY_WIDTH+kh*DIM+i];qnorm=fmaf(q,q,qnorm);knorm=fmaf(k,k,knorm);}float qs=1.0f/sqrtf(qnorm+1e-6f)/sqrtf((float)DIM),ks=1.0f/sqrtf(knorm+1e-6f),decay=expf(a_decay[h]*softplus_ref(alpha[h]+dt[h])),mix=1.0f/(1.0f+expf(-beta[h]));float delta[DIM];uint32_t base=h*DIM*DIM;for(uint32_t v=0;v<DIM;v++){float memory=0.0f;for(uint32_t k=0;k<DIM;k++)memory=fmaf(cpu_state[base+k*DIM+v]*decay,qkv[KEY_WIDTH+kh*DIM+k]*ks,memory);delta[v]=(qkv[2*KEY_WIDTH+h*DIM+v]-memory)*mix;}for(uint32_t k=0;k<DIM;k++)for(uint32_t v=0;v<DIM;v++)cpu_state[base+k*DIM+v]=cpu_state[base+k*DIM+v]*decay+qkv[KEY_WIDTH+kh*DIM+k]*ks*delta[v];float ss=0.0f;for(uint32_t v=0;v<DIM;v++){float value=0.0f;for(uint32_t k=0;k<DIM;k++)value=fmaf(cpu_state[base+k*DIM+v],qkv[kh*DIM+k]*qs,value);expected[h*DIM+v]=value;ss=fmaf(value,value,ss);}float scale=1.0f/sqrtf(ss/(float)DIM+1e-6f);for(uint32_t v=0;v<DIM;v++){float gate=z[h*DIM+v];expected[h*DIM+v]*=scale*norm[v]*(1.0f/(1.0f+expf(-gate)));}}
        fg_vk_tensor *gq=tensor(qkv,QKV_WIDTH*4u),*gz=tensor(z,VALUE_WIDTH*4u),*ga=tensor(alpha,sizeof(alpha)),*gb=tensor(beta,sizeof(beta)),*gal=tensor(a_decay,sizeof(a_decay)),*gdt=tensor(dt,sizeof(dt)),*gn=tensor(norm,sizeof(norm)),*gs=tensor(state,STATE_VALUES*4u),*go=tensor(NULL,VALUE_WIDTH*4u);ok=gq&&gz&&ga&&gb&&gal&&gdt&&gn&&gs&&go&&fg_vk_gdn_recurrent_decode(context,go,gs,gq,gz,ga,gb,gal,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,1e-6f,&error)==FG_OK&&fg_vk_tensor_read(go,0,got,VALUE_WIDTH*4u,&error)==FG_OK&&fg_vk_tensor_read(gs,0,state,STATE_VALUES*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<VALUE_WIDTH;i++)if(fabsf(got[i]-expected[i])>3e-4f*fmaxf(1.0f,fabsf(expected[i]))){fprintf(stderr,"GDN out %u GPU=%g CPU=%g\n",i,got[i],expected[i]);ok=0;}for(uint32_t i=0;ok&&i<STATE_VALUES;i+=97u)if(fabsf(state[i]-cpu_state[i])>3e-5f){fprintf(stderr,"GDN state %u GPU=%g CPU=%g\n",i,state[i],cpu_state[i]);ok=0;}fg_vk_tensor_destroy(go);fg_vk_tensor_destroy(gs);fg_vk_tensor_destroy(gn);fg_vk_tensor_destroy(gdt);fg_vk_tensor_destroy(gal);fg_vk_tensor_destroy(gb);fg_vk_tensor_destroy(ga);fg_vk_tensor_destroy(gz);fg_vk_tensor_destroy(gq);
    }
    free(expected);free(got);free(cpu_state);free(state);free(z);fg_vk_tensor_destroy(gco);fg_vk_tensor_destroy(gcs);fg_vk_tensor_destroy(gcw);fg_vk_tensor_destroy(gp);free(conv_got);free(conv_expected);free(conv_state);free(conv_weight);free(projection);return ok;
}

static int test_gdn_algebraic(void){
    enum{VALUE_HEADS=48,KEY_HEADS=16,DIM=128,KEY_WIDTH=KEY_HEADS*DIM,VALUE_WIDTH=VALUE_HEADS*DIM,QKV_WIDTH=2*KEY_WIDTH+VALUE_WIDTH,STATE_VALUES=VALUE_HEADS*DIM*DIM,ITERATIONS=24};float *qkv=malloc(QKV_WIDTH*4u),*z=malloc(VALUE_WIDTH*4u),alpha[VALUE_HEADS],beta[VALUE_HEADS],a_decay[VALUE_HEADS],dt[VALUE_HEADS],norm[DIM],*initial=malloc(STATE_VALUES*4u),*state_reference=malloc(STATE_VALUES*4u),*state_candidate=malloc(STATE_VALUES*4u),*reference=malloc(VALUE_WIDTH*4u),*candidate=malloc(VALUE_WIDTH*4u);if(!qkv||!z||!initial||!state_reference||!state_candidate||!reference||!candidate){free(candidate);free(reference);free(state_candidate);free(state_reference);free(initial);free(z);free(qkv);return 0;}for(uint32_t i=0;i<QKV_WIDTH;i++)qkv[i]=sinf((float)i*0.007f)+0.1f*cosf((float)i*0.019f);for(uint32_t i=0;i<VALUE_WIDTH;i++)z[i]=cosf((float)i*0.009f);for(uint32_t head=0;head<VALUE_HEADS;head++){alpha[head]=-0.2f+(float)head*0.01f;beta[head]=0.3f-(float)head*0.007f;a_decay[head]=-0.03f*(float)(head+1u);dt[head]=-0.1f+(float)head*0.002f;}for(uint32_t i=0;i<DIM;i++)norm[i]=0.8f+0.1f*sinf((float)i*0.03f);for(uint32_t i=0;i<STATE_VALUES;i++)initial[i]=sinf((float)i*0.0007f)*0.01f;
    fg_vk_tensor *gq=tensor(qkv,QKV_WIDTH*4u),*gz=tensor(z,VALUE_WIDTH*4u),*ga=tensor(alpha,sizeof(alpha)),*gb=tensor(beta,sizeof(beta)),*gal=tensor(a_decay,sizeof(a_decay)),*gdt=tensor(dt,sizeof(dt)),*gn=tensor(norm,sizeof(norm)),*gsr=tensor(initial,STATE_VALUES*4u),*gsc=tensor(initial,STATE_VALUES*4u),*gor=tensor(NULL,VALUE_WIDTH*4u),*goc=tensor(NULL,VALUE_WIDTH*4u);int ok=gq&&gz&&ga&&gb&&gal&&gdt&&gn&&gsr&&gsc&&gor&&goc&&fg_vk_gdn_recurrent_decode(context,gor,gsr,gq,gz,ga,gb,gal,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,1e-6f,&error)==FG_OK&&fg_vk_gdn_recurrent_algebraic(context,goc,gsc,gq,gz,ga,gb,gal,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,1e-6f,&error)==FG_OK&&fg_vk_tensor_read(gor,0,reference,VALUE_WIDTH*4u,&error)==FG_OK&&fg_vk_tensor_read(goc,0,candidate,VALUE_WIDTH*4u,&error)==FG_OK&&fg_vk_tensor_read(gsr,0,state_reference,STATE_VALUES*4u,&error)==FG_OK&&fg_vk_tensor_read(gsc,0,state_candidate,STATE_VALUES*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<VALUE_WIDTH;i++)if(fabsf(candidate[i]-reference[i])>3e-4f*fmaxf(1.0f,fabsf(reference[i]))){fprintf(stderr,"GDN algebraic out %u candidate=%g reference=%g diff=%g\n",i,candidate[i],reference[i],candidate[i]-reference[i]);ok=0;}for(uint32_t i=0;ok&&i<STATE_VALUES;i++)if(state_candidate[i]!=state_reference[i]){fprintf(stderr,"GDN algebraic state %u candidate=%g reference=%g\n",i,state_candidate[i],state_reference[i]);ok=0;}
    if(ok&&getenv("FG_BENCH_GDN_ALGEBRAIC")){fg_vk_profile old_profile={0},new_profile={0};struct timespec start,end;ok=fg_vk_tensor_write(gsr,0,initial,STATE_VALUES*4u,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&start);ok=ok&&fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<ITERATIONS;i++)ok=fg_vk_gdn_recurrent_decode(context,gor,gsr,gq,gz,ga,gb,gal,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,1e-6f,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&old_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double old_wall=((double)(end.tv_sec-start.tv_sec)*1e3+(double)(end.tv_nsec-start.tv_nsec)*1e-6)/(double)ITERATIONS;ok=ok&&fg_vk_tensor_write(gsc,0,initial,STATE_VALUES*4u,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&start);ok=ok&&fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;for(uint32_t i=0;ok&&i<ITERATIONS;i++)ok=fg_vk_gdn_recurrent_algebraic(context,goc,gsc,gq,gz,ga,gb,gal,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,1e-6f,&error)==FG_OK;if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_profile_end(context,&new_profile,&error)==FG_OK;clock_gettime(CLOCK_MONOTONIC,&end);double new_wall=((double)(end.tv_sec-start.tv_sec)*1e3+(double)(end.tv_nsec-start.tv_nsec)*1e-6)/(double)ITERATIONS;if(ok)fprintf(stderr,"GDN_ALGEBRAIC_BENCH iterations=%u old_gpu_us=%.3f new_gpu_us=%.3f old_wall_us=%.3f new_wall_us=%.3f\n",ITERATIONS,old_profile.gpu_ms*1000.0/(double)ITERATIONS,new_profile.gpu_ms*1000.0/(double)ITERATIONS,old_wall*1000.0,new_wall*1000.0);}
    fg_vk_tensor_destroy(goc);fg_vk_tensor_destroy(gor);fg_vk_tensor_destroy(gsc);fg_vk_tensor_destroy(gsr);fg_vk_tensor_destroy(gn);fg_vk_tensor_destroy(gdt);fg_vk_tensor_destroy(gal);fg_vk_tensor_destroy(gb);fg_vk_tensor_destroy(ga);fg_vk_tensor_destroy(gz);fg_vk_tensor_destroy(gq);free(candidate);free(reference);free(state_candidate);free(state_reference);free(initial);free(z);free(qkv);return ok;
}

static int test_gdn_prefill_scan(void){
    enum{TOKENS=3,CHANNELS=257,VALUE_HEADS=2,KEY_HEADS=1,DIM=128,KEY_WIDTH=KEY_HEADS*DIM,VALUE_WIDTH=VALUE_HEADS*DIM,QKV_WIDTH=2*KEY_WIDTH+VALUE_WIDTH,STATE_VALUES=VALUE_HEADS*DIM*DIM};
    float *projection=malloc(TOKENS*CHANNELS*4u),*conv_weight=malloc(CHANNELS*4u*4u),*conv_initial=malloc(CHANNELS*4u*4u),*conv_sequential=malloc(CHANNELS*4u*4u),*conv_batched=malloc(CHANNELS*4u*4u),*conv_output_sequential=malloc(TOKENS*CHANNELS*4u),*conv_output_batched=malloc(TOKENS*CHANNELS*4u);if(!projection||!conv_weight||!conv_initial||!conv_sequential||!conv_batched||!conv_output_sequential||!conv_output_batched)return 0;
    for(uint32_t i=0;i<TOKENS*CHANNELS;i++)projection[i]=sinf((float)i*0.017f);
    for(uint32_t i=0;i<CHANNELS*4u;i++){conv_weight[i]=cosf((float)i*0.013f)*0.25f;conv_initial[i]=sinf((float)i*0.011f)*0.1f;}
    fg_vk_tensor *gp=tensor(projection,TOKENS*CHANNELS*4u),*gcw=tensor(conv_weight,CHANNELS*16u),*gcs=tensor(conv_initial,CHANNELS*16u),*gcb=tensor(conv_initial,CHANNELS*16u),*gso=tensor(NULL,TOKENS*CHANNELS*4u),*gbo=tensor(NULL,TOKENS*CHANNELS*4u);int ok=gp&&gcw&&gcs&&gcb&&gso&&gbo;
    for(uint32_t token=0;ok&&token<TOKENS;token++){fg_vk_tensor *pv=NULL,*ov=NULL;ok=fg_vk_tensor_view(gp,(uint64_t)token*CHANNELS*4u,CHANNELS*4u,&pv,&error)==FG_OK&&fg_vk_tensor_view(gso,(uint64_t)token*CHANNELS*4u,CHANNELS*4u,&ov,&error)==FG_OK&&fg_vk_gdn_conv_decode(context,ov,gcs,pv,gcw,CHANNELS,&error)==FG_OK;fg_vk_tensor_destroy(ov);fg_vk_tensor_destroy(pv);}if(ok)ok=fg_vk_gdn_conv_prefill(context,gbo,gcb,gp,gcw,CHANNELS,TOKENS,&error)==FG_OK&&fg_vk_tensor_read(gso,0,conv_output_sequential,TOKENS*CHANNELS*4u,&error)==FG_OK&&fg_vk_tensor_read(gbo,0,conv_output_batched,TOKENS*CHANNELS*4u,&error)==FG_OK&&fg_vk_tensor_read(gcs,0,conv_sequential,CHANNELS*16u,&error)==FG_OK&&fg_vk_tensor_read(gcb,0,conv_batched,CHANNELS*16u,&error)==FG_OK;for(uint32_t i=0;ok&&i<TOKENS*CHANNELS;i++)if(conv_output_sequential[i]!=conv_output_batched[i])ok=0;for(uint32_t i=0;ok&&i<CHANNELS*4u;i++)if(conv_sequential[i]!=conv_batched[i])ok=0;
    float *qkv=malloc(TOKENS*QKV_WIDTH*4u),*z=malloc(TOKENS*VALUE_WIDTH*4u),alpha[TOKENS*VALUE_HEADS],beta[TOKENS*VALUE_HEADS],a_decay[VALUE_HEADS],dt[VALUE_HEADS],norm[DIM],*state_initial=malloc(STATE_VALUES*4u),*state_sequential=malloc(STATE_VALUES*4u),*state_batched=malloc(STATE_VALUES*4u),*output_sequential=malloc(TOKENS*VALUE_WIDTH*4u),*output_batched=malloc(TOKENS*VALUE_WIDTH*4u);if(!qkv||!z||!state_initial||!state_sequential||!state_batched||!output_sequential||!output_batched)ok=0;
    for(uint32_t i=0;ok&&i<TOKENS*QKV_WIDTH;i++)qkv[i]=sinf((float)i*0.007f)+0.1f*cosf((float)i*0.019f);
    for(uint32_t i=0;ok&&i<TOKENS*VALUE_WIDTH;i++)z[i]=cosf((float)i*0.009f);
    for(uint32_t token=0;ok&&token<TOKENS;token++)for(uint32_t head=0;head<VALUE_HEADS;head++){alpha[token*VALUE_HEADS+head]=-0.2f+(float)(token+head)*0.1f;beta[token*VALUE_HEADS+head]=0.3f-(float)(token+head)*0.07f;}
    for(uint32_t head=0;ok&&head<VALUE_HEADS;head++){a_decay[head]=-0.03f*(float)(head+1u);dt[head]=-0.1f+(float)head*0.02f;}
    for(uint32_t i=0;ok&&i<DIM;i++)norm[i]=0.8f+0.1f*sinf((float)i*0.03f);
    for(uint32_t i=0;ok&&i<STATE_VALUES;i++)state_initial[i]=sinf((float)i*0.0007f)*0.01f;
    fg_vk_tensor *gq=ok?tensor(qkv,TOKENS*QKV_WIDTH*4u):NULL,*gz=ok?tensor(z,TOKENS*VALUE_WIDTH*4u):NULL,*ga=ok?tensor(alpha,sizeof(alpha)):NULL,*gb=ok?tensor(beta,sizeof(beta)):NULL,*gad=ok?tensor(a_decay,sizeof(a_decay)):NULL,*gdt=ok?tensor(dt,sizeof(dt)):NULL,*gn=ok?tensor(norm,sizeof(norm)):NULL,*gss=ok?tensor(state_initial,STATE_VALUES*4u):NULL,*gsb=ok?tensor(state_initial,STATE_VALUES*4u):NULL,*gos=ok?tensor(NULL,TOKENS*VALUE_WIDTH*4u):NULL,*gob=ok?tensor(NULL,TOKENS*VALUE_WIDTH*4u):NULL;ok=ok&&gq&&gz&&ga&&gb&&gad&&gdt&&gn&&gss&&gsb&&gos&&gob;
    for(uint32_t token=0;ok&&token<TOKENS;token++){fg_vk_tensor *qv=NULL,*zv=NULL,*av=NULL,*bv=NULL,*ov=NULL;ok=fg_vk_tensor_view(gq,(uint64_t)token*QKV_WIDTH*4u,QKV_WIDTH*4u,&qv,&error)==FG_OK&&fg_vk_tensor_view(gz,(uint64_t)token*VALUE_WIDTH*4u,VALUE_WIDTH*4u,&zv,&error)==FG_OK&&fg_vk_tensor_view(ga,(uint64_t)token*VALUE_HEADS*4u,VALUE_HEADS*4u,&av,&error)==FG_OK&&fg_vk_tensor_view(gb,(uint64_t)token*VALUE_HEADS*4u,VALUE_HEADS*4u,&bv,&error)==FG_OK&&fg_vk_tensor_view(gos,(uint64_t)token*VALUE_WIDTH*4u,VALUE_WIDTH*4u,&ov,&error)==FG_OK&&fg_vk_gdn_recurrent_decode(context,ov,gss,qv,zv,av,bv,gad,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,1e-6f,&error)==FG_OK;fg_vk_tensor_destroy(ov);fg_vk_tensor_destroy(bv);fg_vk_tensor_destroy(av);fg_vk_tensor_destroy(zv);fg_vk_tensor_destroy(qv);}if(ok)ok=fg_vk_gdn_recurrent_prefill(context,gob,gsb,gq,gz,ga,gb,gad,gdt,gn,VALUE_HEADS,KEY_HEADS,DIM,TOKENS,1e-6f,&error)==FG_OK&&fg_vk_tensor_read(gos,0,output_sequential,TOKENS*VALUE_WIDTH*4u,&error)==FG_OK&&fg_vk_tensor_read(gob,0,output_batched,TOKENS*VALUE_WIDTH*4u,&error)==FG_OK&&fg_vk_tensor_read(gss,0,state_sequential,STATE_VALUES*4u,&error)==FG_OK&&fg_vk_tensor_read(gsb,0,state_batched,STATE_VALUES*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<TOKENS*VALUE_WIDTH;i++)if(fabsf(output_sequential[i]-output_batched[i])>2e-6f)ok=0;for(uint32_t i=0;ok&&i<STATE_VALUES;i++)if(fabsf(state_sequential[i]-state_batched[i])>2e-7f)ok=0;
    fg_vk_tensor_destroy(gob);fg_vk_tensor_destroy(gos);fg_vk_tensor_destroy(gsb);fg_vk_tensor_destroy(gss);fg_vk_tensor_destroy(gn);fg_vk_tensor_destroy(gdt);fg_vk_tensor_destroy(gad);fg_vk_tensor_destroy(gb);fg_vk_tensor_destroy(ga);fg_vk_tensor_destroy(gz);fg_vk_tensor_destroy(gq);free(output_batched);free(output_sequential);free(state_batched);free(state_sequential);free(state_initial);free(z);free(qkv);fg_vk_tensor_destroy(gbo);fg_vk_tensor_destroy(gso);fg_vk_tensor_destroy(gcb);fg_vk_tensor_destroy(gcs);fg_vk_tensor_destroy(gcw);fg_vk_tensor_destroy(gp);free(conv_output_batched);free(conv_output_sequential);free(conv_batched);free(conv_sequential);free(conv_initial);free(conv_weight);free(projection);return ok;
}

enum{
    GDN_VALUE_HEADS=48,
    GDN_KEY_HEADS=16,
    GDN_DIM=128,
    GDN_KEY_WIDTH=GDN_KEY_HEADS*GDN_DIM,
    GDN_VALUE_WIDTH=GDN_VALUE_HEADS*GDN_DIM,
    GDN_QKV_WIDTH=2*GDN_KEY_WIDTH+GDN_VALUE_WIDTH,
    GDN_STATE_VALUES=GDN_VALUE_HEADS*GDN_DIM*GDN_DIM,
    GDN_STATE_GUARD=4096
};

typedef struct gdn_prefill_data {
    float *qkv,*z,*alpha,*beta,*state;
    float a_decay[GDN_VALUE_HEADS];
    float dt_bias[GDN_VALUE_HEADS];
    float norm_weight[GDN_DIM];
} gdn_prefill_data;

static void gdn_prefill_data_free(gdn_prefill_data *data){
    if(!data)return;
    free(data->state);free(data->beta);free(data->alpha);free(data->z);
    free(data->qkv);memset(data,0,sizeof(*data));
}

static int gdn_prefill_data_init(gdn_prefill_data *data,uint32_t tokens,
                                 uint32_t mode){
    memset(data,0,sizeof(*data));
    data->qkv=malloc((size_t)tokens*GDN_QKV_WIDTH*4u);
    data->z=malloc((size_t)tokens*GDN_VALUE_WIDTH*4u);
    data->alpha=malloc((size_t)tokens*GDN_VALUE_HEADS*4u);
    data->beta=malloc((size_t)tokens*GDN_VALUE_HEADS*4u);
    data->state=malloc(GDN_STATE_VALUES*4u);
    if(!data->qkv||!data->z||!data->alpha||!data->beta||!data->state){
        gdn_prefill_data_free(data);return 0;
    }
    for(uint32_t token=0;token<tokens;token++){
        for(uint32_t i=0;i<GDN_QKV_WIDTH;i++){
            uint64_t index=(uint64_t)token*GDN_QKV_WIDTH+i;
            if(mode==0u)data->qkv[index]=0.0f;
            else if(mode==1u)data->qkv[index]=
                sinf((float)(index+3u)*0.007f)+
                0.1f*cosf((float)(index+11u)*0.019f);
            else if(i<2u*GDN_KEY_WIDTH)
                data->qkv[index]=(index&1u)?-1.0e10f:1.0e10f;
            else data->qkv[index]=(index&1u)?-1.0e4f:1.0e4f;
        }
        for(uint32_t i=0;i<GDN_VALUE_WIDTH;i++){
            uint64_t index=(uint64_t)token*GDN_VALUE_WIDTH+i;
            data->z[index]=mode==0u?0.0f:mode==1u?
                cosf((float)(index+5u)*0.009f):
                ((index&1u)?-30.0f:30.0f);
        }
        for(uint32_t head=0;head<GDN_VALUE_HEADS;head++){
            uint64_t index=(uint64_t)token*GDN_VALUE_HEADS+head;
            data->alpha[index]=mode==0u?0.0f:mode==1u?
                -0.2f+(float)((token+head)%17u)*0.03f:
                ((index&1u)?-40.0f:40.0f);
            data->beta[index]=mode==0u?0.0f:mode==1u?
                0.3f-(float)((token+head)%19u)*0.025f:
                ((index&1u)?-30.0f:30.0f);
        }
    }
    for(uint32_t head=0;head<GDN_VALUE_HEADS;head++){
        data->a_decay[head]=mode==0u?-0.01f:
            -(0.01f+(float)(head%11u)*0.004f);
        data->dt_bias[head]=mode==2u?
            ((head&1u)?-5.0f:5.0f):
            -0.1f+(float)head*0.002f;
    }
    for(uint32_t i=0;i<GDN_DIM;i++)
        data->norm_weight[i]=mode==2u?
            0.5f+(float)(i%17u)*0.0625f:
            0.8f+0.1f*sinf((float)i*0.03f);
    for(uint32_t i=0;i<GDN_STATE_VALUES;i++)
        data->state[i]=mode==0u?0.0f:mode==1u?
            sinf((float)i*0.0007f)*0.01f:
            ((i&1u)?-1.0e-5f:1.0e-5f);
    return 1;
}

static int gdn_guarded_state(const float *initial,uint8_t pattern,
                             fg_vk_tensor **arena,fg_vk_tensor **state){
    uint64_t state_bytes=(uint64_t)GDN_STATE_VALUES*4u;
    uint64_t arena_bytes=state_bytes+2u*GDN_STATE_GUARD;
    *arena=tensor(NULL,arena_bytes);*state=NULL;
    if(!*arena)return 0;
    memset(fg_vk_tensor_map(*arena),pattern,(size_t)arena_bytes);
    return fg_vk_tensor_view(*arena,GDN_STATE_GUARD,state_bytes,state,&error)==FG_OK&&
        fg_vk_tensor_write(*state,0,initial,state_bytes,&error)==FG_OK;
}

static int gdn_canary_ok(const fg_vk_tensor *arena,uint8_t pattern){
    const uint8_t *bytes=fg_vk_tensor_const_map(arena);
    uint64_t state_bytes=(uint64_t)GDN_STATE_VALUES*4u;
    if(!bytes)return 0;
    for(uint32_t i=0;i<GDN_STATE_GUARD;i++)
        if(bytes[i]!=pattern||
           bytes[GDN_STATE_GUARD+state_bytes+i]!=pattern)return 0;
    return 1;
}

static int gdn_values_close(const float *reference,const float *candidate,
                            uint64_t count,float absolute,float relative,
                            const char *label){
    for(uint64_t i=0;i<count;i++){
        float a=reference[i],b=candidate[i];
        float tolerance=absolute+relative*fmaxf(fabsf(a),fabsf(b));
        if(!isfinite(a)||!isfinite(b)||fabsf(a-b)>tolerance){
            fprintf(stderr,"%s %llu reference=%g candidate=%g tolerance=%g\n",
                    label,(unsigned long long)i,a,b,tolerance);
            return 0;
        }
    }
    return 1;
}

static int gdn_sequential_dispatch(uint32_t tokens,const gdn_prefill_data *data,
                                   fg_vk_tensor *state,fg_vk_tensor *output){
    fg_vk_tensor *qkv=tensor(data->qkv,(uint64_t)tokens*GDN_QKV_WIDTH*4u);
    fg_vk_tensor *z=tensor(data->z,(uint64_t)tokens*GDN_VALUE_WIDTH*4u);
    fg_vk_tensor *alpha=tensor(data->alpha,(uint64_t)tokens*GDN_VALUE_HEADS*4u);
    fg_vk_tensor *beta=tensor(data->beta,(uint64_t)tokens*GDN_VALUE_HEADS*4u);
    fg_vk_tensor *a=tensor(data->a_decay,sizeof(data->a_decay));
    fg_vk_tensor *dt=tensor(data->dt_bias,sizeof(data->dt_bias));
    fg_vk_tensor *norm=tensor(data->norm_weight,sizeof(data->norm_weight));
    int ok=qkv&&z&&alpha&&beta&&a&&dt&&norm&&
        fg_vk_begin(context,&error)==FG_OK;
    for(uint32_t token=0;ok&&token<tokens;token++){
        fg_vk_tensor *qv=NULL,*zv=NULL,*av=NULL,*bv=NULL,*ov=NULL;
        ok=fg_vk_tensor_view(qkv,(uint64_t)token*GDN_QKV_WIDTH*4u,
                GDN_QKV_WIDTH*4u,&qv,&error)==FG_OK&&
            fg_vk_tensor_view(z,(uint64_t)token*GDN_VALUE_WIDTH*4u,
                GDN_VALUE_WIDTH*4u,&zv,&error)==FG_OK&&
            fg_vk_tensor_view(alpha,(uint64_t)token*GDN_VALUE_HEADS*4u,
                GDN_VALUE_HEADS*4u,&av,&error)==FG_OK&&
            fg_vk_tensor_view(beta,(uint64_t)token*GDN_VALUE_HEADS*4u,
                GDN_VALUE_HEADS*4u,&bv,&error)==FG_OK&&
            fg_vk_tensor_view(output,(uint64_t)token*GDN_VALUE_WIDTH*4u,
                GDN_VALUE_WIDTH*4u,&ov,&error)==FG_OK&&
            fg_vk_gdn_recurrent_decode(context,ov,state,qv,zv,av,bv,a,dt,norm,
                GDN_VALUE_HEADS,GDN_KEY_HEADS,GDN_DIM,1e-6f,&error)==FG_OK;
        fg_vk_tensor_destroy(ov);fg_vk_tensor_destroy(bv);
        fg_vk_tensor_destroy(av);fg_vk_tensor_destroy(zv);
        fg_vk_tensor_destroy(qv);
    }
    if(ok)ok=fg_vk_end(context,&error)==FG_OK;
    else if(fg_vk_batch_active(context)){
        fg_error ignored={0};fg_vk_abort(context,&ignored);
    }
    fg_vk_tensor_destroy(norm);fg_vk_tensor_destroy(dt);fg_vk_tensor_destroy(a);
    fg_vk_tensor_destroy(beta);fg_vk_tensor_destroy(alpha);
    fg_vk_tensor_destroy(z);fg_vk_tensor_destroy(qkv);
    return ok;
}

static int gdn_pipeline_dispatch(uint32_t tokens,const gdn_prefill_data *data,
                                 fg_vk_tensor *state,fg_vk_tensor *output){
    fg_vk_tensor *qkv=tensor(data->qkv,(uint64_t)tokens*GDN_QKV_WIDTH*4u);
    fg_vk_tensor *z=tensor(data->z,(uint64_t)tokens*GDN_VALUE_WIDTH*4u);
    fg_vk_tensor *alpha=tensor(data->alpha,(uint64_t)tokens*GDN_VALUE_HEADS*4u);
    fg_vk_tensor *beta=tensor(data->beta,(uint64_t)tokens*GDN_VALUE_HEADS*4u);
    fg_vk_tensor *a=tensor(data->a_decay,sizeof(data->a_decay));
    fg_vk_tensor *dt=tensor(data->dt_bias,sizeof(data->dt_bias));
    fg_vk_tensor *norm=tensor(data->norm_weight,sizeof(data->norm_weight));
    fg_vk_counters before={0},after={0};
    fg_vk_get_counters(context,&before);
    int ok=qkv&&z&&alpha&&beta&&a&&dt&&norm&&
        fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_gdn_recurrent_prefill_pipeline(context,output,state,qkv,z,alpha,beta,
            a,dt,norm,tokens,1e-6f,&error)==FG_OK&&
        fg_vk_end(context,&error)==FG_OK;
    fg_vk_get_counters(context,&after);
    ok=ok&&after.dispatches-before.dispatches==
        FG_VK_GDN_PIPELINE_PREFILL_DISPATCHES;
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(norm);fg_vk_tensor_destroy(dt);fg_vk_tensor_destroy(a);
    fg_vk_tensor_destroy(beta);fg_vk_tensor_destroy(alpha);
    fg_vk_tensor_destroy(z);fg_vk_tensor_destroy(qkv);
    return ok;
}

static int gdn_pipeline_prefill_case(uint32_t tokens,uint32_t mode){
    gdn_prefill_data data;
    if(!gdn_prefill_data_init(&data,tokens,mode))return 0;
    uint64_t output_values=(uint64_t)tokens*GDN_VALUE_WIDTH;
    float *reference=malloc((size_t)output_values*4u);
    float *candidate=malloc((size_t)output_values*4u);
    float *state_reference=malloc(GDN_STATE_VALUES*4u);
    float *state_candidate=malloc(GDN_STATE_VALUES*4u);
    fg_vk_tensor *reference_arena=NULL,*reference_state=NULL;
    fg_vk_tensor *candidate_arena=NULL,*candidate_state=NULL;
    fg_vk_tensor *reference_output=tensor(NULL,output_values*4u);
    fg_vk_tensor *candidate_output=tensor(NULL,output_values*4u);
    int ok=reference&&candidate&&state_reference&&state_candidate&&
        reference_output&&candidate_output&&
        gdn_guarded_state(data.state,0xa5u,&reference_arena,&reference_state)&&
        gdn_guarded_state(data.state,0x5au,&candidate_arena,&candidate_state)&&
        gdn_sequential_dispatch(tokens,&data,reference_state,reference_output)&&
        gdn_pipeline_dispatch(tokens,&data,candidate_state,candidate_output)&&
        fg_vk_tensor_read(reference_output,0,reference,output_values*4u,&error)==FG_OK&&
        fg_vk_tensor_read(candidate_output,0,candidate,output_values*4u,&error)==FG_OK&&
        fg_vk_tensor_read(reference_state,0,state_reference,
                          GDN_STATE_VALUES*4u,&error)==FG_OK&&
        fg_vk_tensor_read(candidate_state,0,state_candidate,
                          GDN_STATE_VALUES*4u,&error)==FG_OK&&
        gdn_canary_ok(reference_arena,0xa5u)&&
        gdn_canary_ok(candidate_arena,0x5au)&&
        gdn_values_close(reference,candidate,output_values,1e-5f,5e-4f,
                         "GDN pipeline output")&&
        gdn_values_close(state_reference,state_candidate,GDN_STATE_VALUES,
                         1e-6f,1e-4f,"GDN pipeline state");
    fg_vk_tensor_destroy(candidate_output);fg_vk_tensor_destroy(reference_output);
    fg_vk_tensor_destroy(candidate_state);fg_vk_tensor_destroy(candidate_arena);
    fg_vk_tensor_destroy(reference_state);fg_vk_tensor_destroy(reference_arena);
    free(state_candidate);free(state_reference);free(candidate);free(reference);
    gdn_prefill_data_free(&data);
    return ok;
}

static int gdn_pipeline_prefill_parity_mode(uint32_t mode){
    static const uint32_t lengths[]={1u,2u,17u,128u};
    for(uint32_t i=0;i<sizeof(lengths)/sizeof(lengths[0]);i++)
        if(!gdn_pipeline_prefill_case(lengths[i],mode))return 0;
    return 1;
}

static int test_gdn_pipeline_prefill_parity_zero(void){
    return gdn_pipeline_prefill_parity_mode(0u);
}

static int test_gdn_pipeline_prefill_parity_random(void){
    return gdn_pipeline_prefill_parity_mode(1u);
}

static int test_gdn_pipeline_prefill_parity_extreme(void){
    return gdn_pipeline_prefill_parity_mode(2u);
}

static int test_gdn_pipeline_prefill_composition(void){
    enum{TOKENS=128,CHUNK=64};
    gdn_prefill_data data;
    if(!gdn_prefill_data_init(&data,TOKENS,1u))return 0;
    uint64_t qkv_bytes=(uint64_t)TOKENS*GDN_QKV_WIDTH*4u;
    uint64_t value_bytes=(uint64_t)TOKENS*GDN_VALUE_WIDTH*4u;
    uint64_t control_bytes=(uint64_t)TOKENS*GDN_VALUE_HEADS*4u;
    fg_vk_tensor *qkv_full=tensor(data.qkv,qkv_bytes);
    fg_vk_tensor *qkv_chunk=tensor(data.qkv,qkv_bytes);
    fg_vk_tensor *z=tensor(data.z,value_bytes);
    fg_vk_tensor *alpha=tensor(data.alpha,control_bytes);
    fg_vk_tensor *beta=tensor(data.beta,control_bytes);
    fg_vk_tensor *a=tensor(data.a_decay,sizeof(data.a_decay));
    fg_vk_tensor *dt=tensor(data.dt_bias,sizeof(data.dt_bias));
    fg_vk_tensor *norm=tensor(data.norm_weight,sizeof(data.norm_weight));
    fg_vk_tensor *state_full=tensor(data.state,GDN_STATE_VALUES*4u);
    fg_vk_tensor *state_chunk=tensor(data.state,GDN_STATE_VALUES*4u);
    fg_vk_tensor *output_full=tensor(NULL,value_bytes);
    fg_vk_tensor *output_chunk=tensor(NULL,value_bytes);
    int ok=qkv_full&&qkv_chunk&&z&&alpha&&beta&&a&&dt&&norm&&state_full&&
        state_chunk&&output_full&&output_chunk&&fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_gdn_recurrent_prefill_pipeline(context,output_full,state_full,qkv_full,
            z,alpha,beta,a,dt,norm,TOKENS,1e-6f,&error)==FG_OK&&
        fg_vk_end(context,&error)==FG_OK;
    for(uint32_t chunk=0;ok&&chunk<2u;chunk++){
        fg_vk_tensor *qv=NULL,*zv=NULL,*av=NULL,*bv=NULL,*ov=NULL;
        uint64_t q_offset=(uint64_t)chunk*CHUNK*GDN_QKV_WIDTH*4u;
        uint64_t v_offset=(uint64_t)chunk*CHUNK*GDN_VALUE_WIDTH*4u;
        uint64_t c_offset=(uint64_t)chunk*CHUNK*GDN_VALUE_HEADS*4u;
        ok=fg_vk_tensor_view(qkv_chunk,q_offset,
                (uint64_t)CHUNK*GDN_QKV_WIDTH*4u,&qv,&error)==FG_OK&&
            fg_vk_tensor_view(z,v_offset,(uint64_t)CHUNK*GDN_VALUE_WIDTH*4u,
                &zv,&error)==FG_OK&&
            fg_vk_tensor_view(alpha,c_offset,
                (uint64_t)CHUNK*GDN_VALUE_HEADS*4u,&av,&error)==FG_OK&&
            fg_vk_tensor_view(beta,c_offset,
                (uint64_t)CHUNK*GDN_VALUE_HEADS*4u,&bv,&error)==FG_OK&&
            fg_vk_tensor_view(output_chunk,v_offset,
                (uint64_t)CHUNK*GDN_VALUE_WIDTH*4u,&ov,&error)==FG_OK&&
            fg_vk_begin(context,&error)==FG_OK&&
            fg_vk_gdn_recurrent_prefill_pipeline(context,ov,state_chunk,qv,zv,av,bv,
                a,dt,norm,CHUNK,1e-6f,&error)==FG_OK&&
            fg_vk_end(context,&error)==FG_OK;
        fg_vk_tensor_destroy(ov);fg_vk_tensor_destroy(bv);
        fg_vk_tensor_destroy(av);fg_vk_tensor_destroy(zv);
        fg_vk_tensor_destroy(qv);
    }
    float *full=malloc((size_t)value_bytes),*chunked=malloc((size_t)value_bytes);
    float *full_state=malloc(GDN_STATE_VALUES*4u);
    float *chunked_state=malloc(GDN_STATE_VALUES*4u);
    ok=ok&&full&&chunked&&full_state&&chunked_state&&
        fg_vk_tensor_read(output_full,0,full,value_bytes,&error)==FG_OK&&
        fg_vk_tensor_read(output_chunk,0,chunked,value_bytes,&error)==FG_OK&&
        fg_vk_tensor_read(state_full,0,full_state,GDN_STATE_VALUES*4u,&error)==FG_OK&&
        fg_vk_tensor_read(state_chunk,0,chunked_state,GDN_STATE_VALUES*4u,&error)==FG_OK&&
        gdn_values_close(full,chunked,(uint64_t)TOKENS*GDN_VALUE_WIDTH,
                         1e-5f,5e-4f,"GDN chunk output")&&
        gdn_values_close(full_state,chunked_state,GDN_STATE_VALUES,
                         1e-6f,1e-4f,"GDN chunk state");
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    free(chunked_state);free(full_state);free(chunked);free(full);
    fg_vk_tensor_destroy(output_chunk);fg_vk_tensor_destroy(output_full);
    fg_vk_tensor_destroy(state_chunk);fg_vk_tensor_destroy(state_full);
    fg_vk_tensor_destroy(norm);fg_vk_tensor_destroy(dt);fg_vk_tensor_destroy(a);
    fg_vk_tensor_destroy(beta);fg_vk_tensor_destroy(alpha);
    fg_vk_tensor_destroy(z);fg_vk_tensor_destroy(qkv_chunk);
    fg_vk_tensor_destroy(qkv_full);gdn_prefill_data_free(&data);
    return ok;
}

static int test_gdn_pipeline_prefill_decode_compat(void){
    enum{PREFILL=17,TOKENS=18};
    gdn_prefill_data data;
    if(!gdn_prefill_data_init(&data,TOKENS,1u))return 0;
    fg_vk_tensor *reference_state=tensor(data.state,GDN_STATE_VALUES*4u);
    fg_vk_tensor *candidate_state=tensor(data.state,GDN_STATE_VALUES*4u);
    fg_vk_tensor *reference_output=tensor(NULL,(uint64_t)TOKENS*GDN_VALUE_WIDTH*4u);
    fg_vk_tensor *candidate_output=tensor(NULL,(uint64_t)TOKENS*GDN_VALUE_WIDTH*4u);
    int ok=reference_state&&candidate_state&&reference_output&&candidate_output&&
        gdn_sequential_dispatch(TOKENS,&data,reference_state,reference_output);
    fg_vk_tensor *qkv=tensor(data.qkv,(uint64_t)TOKENS*GDN_QKV_WIDTH*4u);
    fg_vk_tensor *z=tensor(data.z,(uint64_t)TOKENS*GDN_VALUE_WIDTH*4u);
    fg_vk_tensor *alpha=tensor(data.alpha,(uint64_t)TOKENS*GDN_VALUE_HEADS*4u);
    fg_vk_tensor *beta=tensor(data.beta,(uint64_t)TOKENS*GDN_VALUE_HEADS*4u);
    fg_vk_tensor *a=tensor(data.a_decay,sizeof(data.a_decay));
    fg_vk_tensor *dt=tensor(data.dt_bias,sizeof(data.dt_bias));
    fg_vk_tensor *norm=tensor(data.norm_weight,sizeof(data.norm_weight));
    fg_vk_tensor *q_prefill=NULL,*z_prefill=NULL,*alpha_prefill=NULL;
    fg_vk_tensor *beta_prefill=NULL,*out_prefill=NULL;
    ok=ok&&qkv&&z&&alpha&&beta&&a&&dt&&norm&&
        fg_vk_tensor_view(qkv,0,(uint64_t)PREFILL*GDN_QKV_WIDTH*4u,
                          &q_prefill,&error)==FG_OK&&
        fg_vk_tensor_view(z,0,(uint64_t)PREFILL*GDN_VALUE_WIDTH*4u,
                          &z_prefill,&error)==FG_OK&&
        fg_vk_tensor_view(alpha,0,(uint64_t)PREFILL*GDN_VALUE_HEADS*4u,
                          &alpha_prefill,&error)==FG_OK&&
        fg_vk_tensor_view(beta,0,(uint64_t)PREFILL*GDN_VALUE_HEADS*4u,
                          &beta_prefill,&error)==FG_OK&&
        fg_vk_tensor_view(candidate_output,0,
                          (uint64_t)PREFILL*GDN_VALUE_WIDTH*4u,
                          &out_prefill,&error)==FG_OK&&
        fg_vk_begin(context,&error)==FG_OK&&
        fg_vk_gdn_recurrent_prefill_pipeline(context,out_prefill,candidate_state,
            q_prefill,z_prefill,alpha_prefill,beta_prefill,a,dt,norm,PREFILL,
            1e-6f,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK;
    fg_vk_tensor *q_decode=NULL,*z_decode=NULL,*alpha_decode=NULL,*beta_decode=NULL;
    fg_vk_tensor *out_decode=NULL;
    ok=ok&&fg_vk_tensor_view(qkv,(uint64_t)PREFILL*GDN_QKV_WIDTH*4u,
            GDN_QKV_WIDTH*4u,&q_decode,&error)==FG_OK&&
        fg_vk_tensor_view(z,(uint64_t)PREFILL*GDN_VALUE_WIDTH*4u,
            GDN_VALUE_WIDTH*4u,&z_decode,&error)==FG_OK&&
        fg_vk_tensor_view(alpha,(uint64_t)PREFILL*GDN_VALUE_HEADS*4u,
            GDN_VALUE_HEADS*4u,&alpha_decode,&error)==FG_OK&&
        fg_vk_tensor_view(beta,(uint64_t)PREFILL*GDN_VALUE_HEADS*4u,
            GDN_VALUE_HEADS*4u,&beta_decode,&error)==FG_OK&&
        fg_vk_tensor_view(candidate_output,(uint64_t)PREFILL*GDN_VALUE_WIDTH*4u,
            GDN_VALUE_WIDTH*4u,&out_decode,&error)==FG_OK&&
        fg_vk_gdn_recurrent_decode(context,out_decode,candidate_state,q_decode,
            z_decode,alpha_decode,beta_decode,a,dt,norm,GDN_VALUE_HEADS,
            GDN_KEY_HEADS,GDN_DIM,1e-6f,&error)==FG_OK;
    float reference[GDN_VALUE_WIDTH],candidate[GDN_VALUE_WIDTH];
    float *reference_final=malloc(GDN_STATE_VALUES*4u);
    float *candidate_final=malloc(GDN_STATE_VALUES*4u);
    ok=ok&&reference_final&&candidate_final&&
        fg_vk_tensor_read(reference_output,
            (uint64_t)PREFILL*GDN_VALUE_WIDTH*4u,reference,sizeof(reference),
            &error)==FG_OK&&
        fg_vk_tensor_read(candidate_output,
            (uint64_t)PREFILL*GDN_VALUE_WIDTH*4u,candidate,sizeof(candidate),
            &error)==FG_OK&&
        fg_vk_tensor_read(reference_state,0,reference_final,
                          GDN_STATE_VALUES*4u,&error)==FG_OK&&
        fg_vk_tensor_read(candidate_state,0,candidate_final,
                          GDN_STATE_VALUES*4u,&error)==FG_OK&&
        gdn_values_close(reference,candidate,GDN_VALUE_WIDTH,1e-5f,5e-4f,
                         "GDN decode-after-prefill output")&&
        gdn_values_close(reference_final,candidate_final,GDN_STATE_VALUES,
                         1e-6f,1e-4f,"GDN decode-after-prefill state");
    free(candidate_final);free(reference_final);
    fg_vk_tensor_destroy(out_decode);fg_vk_tensor_destroy(beta_decode);
    fg_vk_tensor_destroy(alpha_decode);fg_vk_tensor_destroy(z_decode);
    fg_vk_tensor_destroy(q_decode);fg_vk_tensor_destroy(out_prefill);
    fg_vk_tensor_destroy(beta_prefill);fg_vk_tensor_destroy(alpha_prefill);
    fg_vk_tensor_destroy(z_prefill);fg_vk_tensor_destroy(q_prefill);
    fg_vk_tensor_destroy(norm);fg_vk_tensor_destroy(dt);fg_vk_tensor_destroy(a);
    fg_vk_tensor_destroy(beta);fg_vk_tensor_destroy(alpha);
    fg_vk_tensor_destroy(z);fg_vk_tensor_destroy(qkv);
    fg_vk_tensor_destroy(candidate_output);fg_vk_tensor_destroy(reference_output);
    fg_vk_tensor_destroy(candidate_state);fg_vk_tensor_destroy(reference_state);
    gdn_prefill_data_free(&data);
    return ok;
}

static int test_iq4_nl_dequant(void){
    enum{ROWS=16,WIDTH=160,ROW_BYTES=90};uint8_t packed[ROWS*ROW_BYTES];float expected[ROWS*WIDTH],got[ROWS*WIDTH];memset(packed,0,sizeof(packed));for(uint32_t row=0;row<ROWS;row++)for(uint32_t block=0;block<WIDTH/32u;block++){uint8_t *p=packed+row*ROW_BYTES+block*18u;uint16_t d=fg_f32_to_f16(0.00390625f*(float)(row+block+1u));memcpy(p,&d,2u);for(uint32_t i=0;i<16u;i++)p[2u+i]=(uint8_t)(((i+row)&15u)|(((15u-i+block)&15u)<<4u));}fg_dequantize_iq4_nl(packed,expected,ROWS*WIDTH);fg_vk_tensor *input=tensor(packed,sizeof(packed)),*output=tensor(NULL,sizeof(got));int ok=input&&output&&fg_vk_dequantize_iq4_nl(context,output,input,ROWS,WIDTH,&error)==FG_OK&&fg_vk_tensor_read(output,0,got,sizeof(got),&error)==FG_OK;for(uint32_t i=0;ok&&i<ROWS*WIDTH;i++)if(got[i]!=expected[i]){fprintf(stderr,"IQ4_NL %u GPU=%g CPU=%g\n",i,got[i],expected[i]);ok=0;}fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(input);return ok;
}

static int test_ngram_direct_lookup(void){
    const uint64_t table_bytes=UINT64_C(320001536)*FG_NGRAM_ROW_BYTES;char path[128];snprintf(path,sizeof(path),"/tmp/fg-ngram-%ld.iq4nl",(long)getpid());unlink(path);int fd=open(path,O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);if(fd<0)return 0;int ok=ftruncate(fd,(off_t)fg_align_up_u64(table_bytes,FG_ALIGNMENT))==0;const int32_t history[]={10,20,30};uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];uint8_t packed[FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES];float expected[FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH],got[FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH];if(ok)ok=fg_q38_ngram_lookup(history,3,rows,addresses,&error)==FG_OK;for(uint32_t row=0;ok&&row<FG_NGRAM_HEAD_COUNT;row++){uint8_t *p=packed+row*FG_NGRAM_ROW_BYTES;for(uint32_t block=0;block<FG_NGRAM_EMBED_WIDTH/32u;block++){uint8_t *b=p+block*18u;uint16_t d=fg_f32_to_f16((float)(row+1u)*0.001953125f);memcpy(b,&d,2u);for(uint32_t i=0;i<16u;i++)b[2u+i]=(uint8_t)(((i+row)&15u)|(((i+block+3u)&15u)<<4u));}ok=pwrite(fd,p,FG_NGRAM_ROW_BYTES,(off_t)addresses[row])==FG_NGRAM_ROW_BYTES;}close(fd);fg_dequantize_iq4_nl(packed,expected,FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH);fg_ngram_store *store=NULL;fg_vk_tensor *embedding=NULL;if(ok)ok=fg_ngram_store_open(&store,context,path,table_bytes,1u,&error)==FG_OK;if(ok)ok=fg_ngram_store_lookup(store,history,3,&embedding,&error)==FG_OK;if(ok)ok=fg_vk_tensor_read(embedding,0,got,sizeof(got),&error)==FG_OK;for(uint32_t i=0;ok&&i<FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH;i++)if(got[i]!=expected[i]){fprintf(stderr,"n-gram lookup %u GPU=%g CPU=%g\n",i,got[i],expected[i]);ok=0;}fg_ngram_store_close(store);unlink(path);return ok;
}

static int test_ngram_resident(void){
    enum{ROWS=7,REQUESTS=3};char path[128];snprintf(path,sizeof(path),"/tmp/fg-ngram-resident-%ld.iq4nl",(long)getpid());unlink(path);uint8_t source[ROWS*FG_NGRAM_ROW_BYTES],got[REQUESTS*FG_NGRAM_ROW_BYTES];for(uint32_t i=0;i<sizeof(source);i++)source[i]=(uint8_t)(i*29u+7u);int fd=open(path,O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0600);int ok=fd>=0&&write(fd,source,sizeof(source))==(ssize_t)sizeof(source);if(fd>=0)close(fd);fg_ngram_resident *resident=NULL;uint64_t rows[REQUESTS]={106u,100u,103u};if(ok)ok=fg_ngram_resident_open(&resident,path,100u,ROWS,&error)==FG_OK;if(ok)ok=fg_ngram_resident_read(resident,rows,REQUESTS,got,sizeof(got),&error)==FG_OK;for(uint32_t i=0;ok&&i<REQUESTS;i++)if(memcmp(got+(uint64_t)i*FG_NGRAM_ROW_BYTES,source+(rows[i]-100u)*FG_NGRAM_ROW_BYTES,FG_NGRAM_ROW_BYTES)!=0)ok=0;if(ok){uint64_t outside=99u;ok=fg_ngram_resident_read(resident,&outside,1u,got,sizeof(got),&error)==FG_ERR_MISMATCH;}uint64_t begin=0,count=0;if(ok)ok=fg_q38_ngram_head_range(2u,3u,&begin,&count,&error)==FG_OK&&begin==40000026u&&count==60000139u;uint64_t rank_begin=0,rank_count=0;if(ok)ok=fg_q38_ngram_rank_range(6u,&rank_begin,&rank_count,&error)==FG_OK&&rank_begin==226667743u&&rank_count==46666896u;fg_ngram_resident_close(resident);unlink(path);return ok;
}

static int test_ngram_prefill_lookup(void){
    const uint64_t table_bytes=UINT64_C(320001536)*FG_NGRAM_ROW_BYTES;char path[128];snprintf(path,sizeof(path),"/tmp/fg-ngram-prefill-%ld.iq4nl",(long)getpid());unlink(path);int fd=open(path,O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);if(fd<0)return 0;int ok=ftruncate(fd,(off_t)fg_align_up_u64(table_bytes,FG_ALIGNMENT))==0;const int32_t history[]={100,200,300,400,500};enum{FIRST=1,TOKENS=3};uint64_t addresses[TOKENS*FG_NGRAM_HEAD_COUNT],rows[TOKENS*FG_NGRAM_HEAD_COUNT];uint8_t packed[sizeof(addresses)/sizeof(addresses[0])*FG_NGRAM_ROW_BYTES];float expected[sizeof(addresses)/sizeof(addresses[0])*FG_NGRAM_EMBED_WIDTH],got[sizeof(expected)/sizeof(expected[0])];
    for(uint32_t token=0;ok&&token<TOKENS;token++)ok=fg_q38_ngram_lookup(history,FIRST+token+1u,rows+(uint64_t)token*FG_NGRAM_HEAD_COUNT,addresses+(uint64_t)token*FG_NGRAM_HEAD_COUNT,&error)==FG_OK;
    for(uint32_t i=0;ok&&i<TOKENS*FG_NGRAM_HEAD_COUNT;i++){uint8_t row[FG_NGRAM_ROW_BYTES];uint64_t row_id=addresses[i]/FG_NGRAM_ROW_BYTES;for(uint32_t block=0;block<FG_NGRAM_EMBED_WIDTH/32u;block++){uint8_t *b=row+block*18u;uint16_t d=fg_f32_to_f16(0.001f+(float)(row_id%17u)*0.0001f);memcpy(b,&d,2u);for(uint32_t x=0;x<16u;x++)b[2u+x]=(uint8_t)(((row_id+block+x)&15u)|(((row_id+block*3u+x)&15u)<<4u));}memcpy(packed+(uint64_t)i*FG_NGRAM_ROW_BYTES,row,sizeof(row));ok=pwrite(fd,row,FG_NGRAM_ROW_BYTES,(off_t)addresses[i])==FG_NGRAM_ROW_BYTES;}
    close(fd);fg_dequantize_iq4_nl(packed,expected,TOKENS*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH);fg_ngram_store *store=NULL;fg_vk_tensor *embedding=NULL;if(ok)ok=fg_ngram_store_open(&store,context,path,table_bytes,TOKENS,&error)==FG_OK;if(ok)ok=fg_ngram_store_lookup_prefill(store,history,sizeof(history)/sizeof(history[0]),FIRST,TOKENS,&embedding,&error)==FG_OK&&fg_vk_tensor_bytes(embedding)==(uint64_t)TOKENS*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH*4u;if(ok)ok=fg_vk_tensor_read(embedding,0,got,sizeof(got),&error)==FG_OK;for(uint32_t i=0;ok&&i<TOKENS*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH;i++)if(got[i]!=expected[i]){fprintf(stderr,"n-gram prefill lookup %u GPU=%g CPU=%g\n",i,got[i],expected[i]);ok=0;}
    if(ok){fg_vk_tensor *single=NULL;ok=fg_ngram_store_lookup(store,history,4u,&single,&error)==FG_OK;float one[FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH];if(ok)ok=fg_vk_tensor_read(single,0,one,sizeof(one),&error)==FG_OK;for(uint32_t i=0;ok&&i<FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH;i++)if(one[i]!=got[2u*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH+i])ok=0;}
    fg_ngram_store_close(store);unlink(path);return ok;
}

static int test_ple_decode(void){
    enum{HIDDEN=2560,GROUPS=4,HYPER=10240,HISTORY=9};float *key=malloc(HYPER*4u),*query=malloc(HYPER*4u),*value=malloc(HIDDEN*4u),*gated=malloc(HYPER*4u),*normalized=malloc(HYPER*4u),*weight=malloc(HYPER*4u*4u),*state=malloc(HYPER*HISTORY*4u),*state_expected=malloc(HYPER*HISTORY*4u),*ple_expected=malloc(HYPER*4u),*got=malloc(HYPER*4u),*left=malloc(HYPER*4u),*sum_expected=malloc(HYPER*4u);if(!key||!query||!value||!gated||!normalized||!weight||!state||!state_expected||!ple_expected||!got||!left||!sum_expected)return 0;for(uint32_t i=0;i<HYPER;i++){key[i]=sinf((float)i*0.0021f);query[i]=cosf((float)i*0.0017f);normalized[i]=sinf((float)i*0.0031f)*0.5f;left[i]=cosf((float)i*0.0043f);for(uint32_t k=0;k<4u;k++)weight[i*4u+k]=0.1f*cosf((float)(i*4u+k)*0.0013f);for(uint32_t h=0;h<HISTORY;h++)state[i*HISTORY+h]=0.03f*sinf((float)(i*HISTORY+h)*0.0007f);}for(uint32_t i=0;i<HIDDEN;i++)value[i]=cosf((float)i*0.0053f);memcpy(state_expected,state,HYPER*HISTORY*4u);for(uint32_t group=0;group<GROUPS;group++){float dot=0.0f;for(uint32_t i=0;i<HIDDEN;i++)dot=fmaf(key[group*HIDDEN+i],query[group*HIDDEN+i],dot);float gate=dot/sqrtf((float)HIDDEN);gate=copysignf(sqrtf(fmaxf(fabsf(gate),1e-6f)),gate);float scale=1.0f/(1.0f+expf(-gate));for(uint32_t i=0;i<HIDDEN;i++)gated[group*HIDDEN+i]=scale*value[i];}for(uint32_t i=0;i<HYPER;i++){uint32_t base=i*HISTORY;float conv=weight[i*4u]*state_expected[base]+weight[i*4u+1u]*state_expected[base+3u]+weight[i*4u+2u]*state_expected[base+6u]+weight[i*4u+3u]*normalized[i];for(uint32_t h=0;h<8u;h++)state_expected[base+h]=state_expected[base+h+1u];state_expected[base+8u]=normalized[i];ple_expected[i]=gated[i]+conv/(1.0f+expf(-conv));sum_expected[i]=left[i]+ple_expected[i];}
    fg_vk_tensor *gk=tensor(key,HYPER*4u),*gq=tensor(query,HYPER*4u),*gv=tensor(value,HIDDEN*4u),*gg=tensor(NULL,HYPER*4u),*gn=tensor(normalized,HYPER*4u),*gw=tensor(weight,HYPER*16u),*gs=tensor(state,HYPER*HISTORY*4u),*go=tensor(NULL,HYPER*4u),*gl=tensor(left,HYPER*4u),*ga=tensor(NULL,HYPER*4u);int ok=gk&&gq&&gv&&gg&&gn&&gw&&gs&&go&&gl&&ga&&fg_vk_ple_gate(context,gg,gk,gq,gv,&error)==FG_OK&&fg_vk_tensor_read(gg,0,got,HYPER*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<HYPER;i++)if(fabsf(got[i]-gated[i])>2e-5f){fprintf(stderr,"PLE gate %u GPU=%g CPU=%g\n",i,got[i],gated[i]);ok=0;}if(ok)ok=fg_vk_ple_conv_decode(context,go,gs,gg,gn,gw,&error)==FG_OK&&fg_vk_tensor_read(go,0,got,HYPER*4u,&error)==FG_OK&&fg_vk_tensor_read(gs,0,state,HYPER*HISTORY*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<HYPER;i++)if(fabsf(got[i]-ple_expected[i])>2e-5f){fprintf(stderr,"PLE conv %u GPU=%g CPU=%g\n",i,got[i],ple_expected[i]);ok=0;}for(uint32_t i=0;ok&&i<HYPER*HISTORY;i+=37u)if(state[i]!=state_expected[i]){fprintf(stderr,"PLE state %u GPU=%g CPU=%g\n",i,state[i],state_expected[i]);ok=0;}if(ok)ok=fg_vk_add_f32(context,ga,gl,go,HYPER,&error)==FG_OK&&fg_vk_tensor_read(ga,0,got,HYPER*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<HYPER;i++)if(fabsf(got[i]-sum_expected[i])>2e-5f){fprintf(stderr,"PLE add %u GPU=%g CPU=%g\n",i,got[i],sum_expected[i]);ok=0;}fg_vk_tensor_destroy(ga);fg_vk_tensor_destroy(gl);fg_vk_tensor_destroy(go);fg_vk_tensor_destroy(gs);fg_vk_tensor_destroy(gw);fg_vk_tensor_destroy(gn);fg_vk_tensor_destroy(gg);fg_vk_tensor_destroy(gv);fg_vk_tensor_destroy(gq);fg_vk_tensor_destroy(gk);free(sum_expected);free(left);free(got);free(ple_expected);free(state_expected);free(state);free(weight);free(normalized);free(gated);free(value);free(query);free(key);return ok;
}

static int test_ple_prefill_scan_tokens(uint32_t tokens){
    const uint32_t TOKENS=tokens;enum{HIDDEN=2560,HYPER=10240,HISTORY=9};float *key=malloc(TOKENS*HYPER*4u),*query=malloc(TOKENS*HYPER*4u),*value=malloc(TOKENS*HIDDEN*4u),*normalized=malloc(TOKENS*HYPER*4u),*weight=malloc(HYPER*4u*4u),*state_initial=malloc(HYPER*HISTORY*4u),*gate_sequential=malloc(TOKENS*HYPER*4u),*gate_batched=malloc(TOKENS*HYPER*4u),*conv_sequential=malloc(TOKENS*HYPER*4u),*conv_batched=malloc(TOKENS*HYPER*4u),*state_sequential=malloc(HYPER*HISTORY*4u),*state_batched=malloc(HYPER*HISTORY*4u);if(!key||!query||!value||!normalized||!weight||!state_initial||!gate_sequential||!gate_batched||!conv_sequential||!conv_batched||!state_sequential||!state_batched)return 0;
    for(uint32_t i=0;i<TOKENS*HYPER;i++){key[i]=sinf((float)i*0.003f);query[i]=cosf((float)i*0.004f);normalized[i]=sinf((float)i*0.005f)*0.5f;}for(uint32_t i=0;i<TOKENS*HIDDEN;i++)value[i]=cosf((float)i*0.007f);for(uint32_t i=0;i<HYPER*4u;i++)weight[i]=0.1f*cosf((float)i*0.0011f);for(uint32_t i=0;i<HYPER*HISTORY;i++)state_initial[i]=0.02f*sinf((float)i*0.0003f);
    fg_vk_tensor *gk=tensor(key,TOKENS*HYPER*4u),*gq=tensor(query,TOKENS*HYPER*4u),*gv=tensor(value,TOKENS*HIDDEN*4u),*gn=tensor(normalized,TOKENS*HYPER*4u),*gw=tensor(weight,HYPER*16u),*gss=tensor(state_initial,HYPER*HISTORY*4u),*gsb=tensor(state_initial,HYPER*HISTORY*4u),*ggs=tensor(NULL,TOKENS*HYPER*4u),*ggb=tensor(NULL,TOKENS*HYPER*4u),*gos=tensor(NULL,TOKENS*HYPER*4u),*gob=tensor(NULL,TOKENS*HYPER*4u);int ok=gk&&gq&&gv&&gn&&gw&&gss&&gsb&&ggs&&ggb&&gos&&gob;
    for(uint32_t token=0;ok&&token<TOKENS;token++){fg_vk_tensor *kv=NULL,*qv=NULL,*vv=NULL,*ov=NULL;ok=fg_vk_tensor_view(gk,(uint64_t)token*HYPER*4u,HYPER*4u,&kv,&error)==FG_OK&&fg_vk_tensor_view(gq,(uint64_t)token*HYPER*4u,HYPER*4u,&qv,&error)==FG_OK&&fg_vk_tensor_view(gv,(uint64_t)token*HIDDEN*4u,HIDDEN*4u,&vv,&error)==FG_OK&&fg_vk_tensor_view(ggs,(uint64_t)token*HYPER*4u,HYPER*4u,&ov,&error)==FG_OK&&fg_vk_ple_gate(context,ov,kv,qv,vv,&error)==FG_OK;fg_vk_tensor_destroy(ov);fg_vk_tensor_destroy(vv);fg_vk_tensor_destroy(qv);fg_vk_tensor_destroy(kv);}if(ok)ok=fg_vk_ple_gate_prefill(context,ggb,gk,gq,gv,TOKENS,&error)==FG_OK&&fg_vk_tensor_read(ggs,0,gate_sequential,TOKENS*HYPER*4u,&error)==FG_OK&&fg_vk_tensor_read(ggb,0,gate_batched,TOKENS*HYPER*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<TOKENS*HYPER;i++)if(gate_sequential[i]!=gate_batched[i])ok=0;
    for(uint32_t token=0;ok&&token<TOKENS;token++){fg_vk_tensor *gvw=NULL,*nv=NULL,*ov=NULL;ok=fg_vk_tensor_view(ggs,(uint64_t)token*HYPER*4u,HYPER*4u,&gvw,&error)==FG_OK&&fg_vk_tensor_view(gn,(uint64_t)token*HYPER*4u,HYPER*4u,&nv,&error)==FG_OK&&fg_vk_tensor_view(gos,(uint64_t)token*HYPER*4u,HYPER*4u,&ov,&error)==FG_OK&&fg_vk_ple_conv_decode(context,ov,gss,gvw,nv,gw,&error)==FG_OK;fg_vk_tensor_destroy(ov);fg_vk_tensor_destroy(nv);fg_vk_tensor_destroy(gvw);}if(ok)ok=fg_vk_ple_conv_prefill(context,gob,gsb,ggb,gn,gw,TOKENS,&error)==FG_OK&&fg_vk_tensor_read(gos,0,conv_sequential,TOKENS*HYPER*4u,&error)==FG_OK&&fg_vk_tensor_read(gob,0,conv_batched,TOKENS*HYPER*4u,&error)==FG_OK&&fg_vk_tensor_read(gss,0,state_sequential,HYPER*HISTORY*4u,&error)==FG_OK&&fg_vk_tensor_read(gsb,0,state_batched,HYPER*HISTORY*4u,&error)==FG_OK;for(uint32_t i=0;ok&&i<TOKENS*HYPER;i++)if(conv_sequential[i]!=conv_batched[i])ok=0;for(uint32_t i=0;ok&&i<HYPER*HISTORY;i++)if(state_sequential[i]!=state_batched[i])ok=0;
    fg_vk_tensor_destroy(gob);fg_vk_tensor_destroy(gos);fg_vk_tensor_destroy(ggb);fg_vk_tensor_destroy(ggs);fg_vk_tensor_destroy(gsb);fg_vk_tensor_destroy(gss);fg_vk_tensor_destroy(gw);fg_vk_tensor_destroy(gn);fg_vk_tensor_destroy(gv);fg_vk_tensor_destroy(gq);fg_vk_tensor_destroy(gk);free(state_batched);free(state_sequential);free(conv_batched);free(conv_sequential);free(gate_batched);free(gate_sequential);free(state_initial);free(weight);free(normalized);free(value);free(query);free(key);return ok;
}
static int test_ple_prefill_scan(void){return test_ple_prefill_scan_tokens(3u);}
static int test_ple_prefill_t1_compat(void){
    return test_ple_prefill_scan_tokens(1u);
}

static int test_tensor_view_rebind(void){
    float values[8]={0,1,2,3,4,5,6,7},other_values[8]={8,9,10,11,12,13,14,15},got[2]={0},replacement[2]={20,21};
    fg_vk_tensor *base=tensor(values,sizeof(values)),*other=tensor(other_values,sizeof(other_values)),*view=NULL;int ok=base&&other;
    if(ok){fg_vk_tensor_set_format(base,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);ok=fg_vk_tensor_view(base,sizeof(float),2u*sizeof(float),&view,&error)==FG_OK&&fg_vk_tensor_get_format(view)==FG_VK_TENSOR_FORMAT_Q8_0_COOKED&&fg_vk_tensor_read(view,0,got,sizeof(got),&error)==FG_OK&&got[0]==1.0f&&got[1]==2.0f;}
    if(ok)ok=fg_vk_tensor_write(view,0,replacement,sizeof(replacement),&error)==FG_OK&&((float *)fg_vk_tensor_map(base))[1]==20.0f&&((float *)fg_vk_tensor_map(base))[2]==21.0f;
    if(ok){fg_vk_tensor_set_format(base,FG_VK_TENSOR_FORMAT_DEFAULT);ok=fg_vk_tensor_view_rebind(view,base,4u*sizeof(float),2u*sizeof(float),&error)==FG_OK&&fg_vk_tensor_get_format(view)==FG_VK_TENSOR_FORMAT_DEFAULT&&fg_vk_tensor_read(view,0,got,sizeof(got),&error)==FG_OK&&got[0]==4.0f&&got[1]==5.0f;}
    void *binding=fg_vk_tensor_map(view);uint64_t binding_bytes=fg_vk_tensor_bytes(view);
    if(ok)ok=fg_vk_tensor_view_rebind(view,base,7u*sizeof(float),2u*sizeof(float),&error)==FG_ERR_ARGUMENT&&fg_vk_tensor_map(view)==binding&&fg_vk_tensor_bytes(view)==binding_bytes;
    if(ok)ok=fg_vk_tensor_view_rebind(view,other,0,2u*sizeof(float),&error)==FG_ERR_ARGUMENT&&fg_vk_tensor_map(view)==binding&&fg_vk_tensor_bytes(view)==binding_bytes;
    if(ok)ok=fg_vk_tensor_view_rebind(base,base,0,2u*sizeof(float),&error)==FG_ERR_ARGUMENT;
    float zeros[2]={0,0},output_initial[8]={-1,-1,-1,-1,-1,-1,-1,-1},output_got[8]={0};fg_vk_tensor *zero=tensor(zeros,sizeof(zeros)),*output_base=tensor(output_initial,sizeof(output_initial)),*output_view=NULL;
    if(ok)ok=zero&&output_base&&fg_vk_tensor_view(output_base,0,2u*sizeof(float),&output_view,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_view_rebind(view,base,0,2u*sizeof(float),&error)==FG_OK&&fg_vk_tensor_view_rebind(output_view,output_base,0,2u*sizeof(float),&error)==FG_OK&&fg_vk_add_f32(context,output_view,view,zero,2u,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_view_rebind(view,base,4u*sizeof(float),2u*sizeof(float),&error)==FG_OK&&fg_vk_tensor_view_rebind(output_view,output_base,4u*sizeof(float),2u*sizeof(float),&error)==FG_OK&&fg_vk_add_f32(context,output_view,view,zero,2u,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_read(output_base,0,output_got,sizeof(output_got),&error)==FG_OK&&output_got[0]==0.0f&&output_got[1]==20.0f&&output_got[4]==4.0f&&output_got[5]==5.0f;
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(output_view);fg_vk_tensor_destroy(output_base);fg_vk_tensor_destroy(zero);
    fg_vk_tensor_destroy(base);base=NULL;
    if(ok)ok=fg_vk_tensor_read(view,0,got,sizeof(got),&error)==FG_OK&&got[0]==4.0f&&got[1]==5.0f;
    fg_vk_tensor_destroy(view);fg_vk_tensor_destroy(other);return ok;
}

static int test_memory_telemetry_and_canary(void){
    enum{BYTES=8193};fg_vk_memory_stats before={0},during={0},after={0};
    fg_vk_get_memory_stats(context,&before);
    fg_vk_tensor *value=NULL;
    int ok=fg_vk_tensor_create(context,BYTES,&value,&error)==FG_OK&&value;
    uint8_t expected[BYTES];for(uint32_t i=0;i<BYTES;i++)expected[i]=(uint8_t)(i*37u);
    if(ok)memcpy(fg_vk_tensor_map(value),expected,BYTES);
    uint64_t touched=0;
    if(ok)ok=fg_vk_tensor_residency_canary(value,&touched,&error)==FG_OK&&
             touched==BYTES&&memcmp(fg_vk_tensor_const_map(value),expected,BYTES)==0&&
             fg_vk_tensor_allocation_bytes(value)>=BYTES;
    fg_vk_get_memory_stats(context,&during);
    if(ok)ok=during.requested_live_bytes==before.requested_live_bytes+BYTES&&
             during.allocated_live_bytes==before.allocated_live_bytes+
                                             fg_vk_tensor_allocation_bytes(value)&&
             during.live_allocations==before.live_allocations+1u&&
             during.allocation_count==before.allocation_count+1u;
    fg_vk_tensor_destroy(value);fg_vk_get_memory_stats(context,&after);
    return ok&&after.requested_live_bytes==before.requested_live_bytes&&
           after.allocated_live_bytes==before.allocated_live_bytes&&
           after.live_allocations==before.live_allocations&&
           after.allocation_count==before.allocation_count+1u;
}

static int test_batch_submission_parity(void){
    enum{VALUES=64,STEPS=13,QSA_PREFILL_MAX_DISPATCHES=2571};float input[VALUES],bias[VALUES],standalone_got[VALUES],batched_got[VALUES],sentinel[VALUES],abort_got[VALUES];
    for(uint32_t i=0;i<VALUES;i++){input[i]=(float)i*0.25f;bias[i]=(float)(i%7u)-3.0f;sentinel[i]=1000.0f+(float)i;}
    fg_vk_tensor *si=tensor(input,sizeof(input)),*sb=tensor(bias,sizeof(bias)),*sa=tensor(NULL,sizeof(input)),*sc=tensor(NULL,sizeof(input));
    fg_vk_tensor *bi=tensor(input,sizeof(input)),*bb=tensor(bias,sizeof(bias)),*ba=tensor(NULL,sizeof(input)),*bc=tensor(NULL,sizeof(input));
    fg_vk_tensor *abort_out=tensor(sentinel,sizeof(sentinel));int ok=si&&sb&&sa&&sc&&bi&&bb&&ba&&bc&&abort_out;
    fg_vk_counters before={0},after={0};const fg_vk_tensor *current=si;
    fg_vk_get_counters(context,&before);
    for(uint32_t step=0;ok&&step<STEPS;step++){fg_vk_tensor *target=(step&1u)?sc:sa;ok=fg_vk_add_f32(context,target,current,sb,VALUES,&error)==FG_OK;current=target;}
    if(ok)ok=fg_vk_tensor_read(current,0,standalone_got,sizeof(standalone_got),&error)==FG_OK;
    fg_vk_get_counters(context,&after);
    if(ok)ok=after.submissions-before.submissions==STEPS;
    current=bi;fg_vk_get_counters(context,&before);
    if(ok)ok=fg_vk_begin(context,&error)==FG_OK;
    for(uint32_t step=0;ok&&step<STEPS;step++){fg_vk_tensor *target=(step&1u)?bc:ba;ok=fg_vk_add_f32(context,target,current,bb,VALUES,&error)==FG_OK;current=target;}
    if(ok)ok=fg_vk_end(context,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_read(current,0,batched_got,sizeof(batched_got),&error)==FG_OK;
    fg_vk_get_counters(context,&after);
    if(ok)ok=after.submissions-before.submissions==1u&&memcmp(standalone_got,batched_got,sizeof(standalone_got))==0;
    fg_vk_get_counters(context,&before);if(ok)ok=fg_vk_begin(context,&error)==FG_OK&&fg_vk_add_f32(context,abort_out,bi,bb,VALUES,&error)==FG_OK&&fg_vk_abort(context,&error)==FG_OK&&!fg_vk_batch_active(context);fg_vk_get_counters(context,&after);
    if(ok)ok=after.submissions==before.submissions&&fg_vk_tensor_read(abort_out,0,abort_got,sizeof(abort_got),&error)==FG_OK&&memcmp(abort_got,sentinel,sizeof(sentinel))==0;
    float zero=0.0f,one=1.0f,large_got=0.0f;fg_vk_tensor *large_input=ok?tensor(&zero,sizeof(zero)):NULL,*large_bias=ok?tensor(&one,sizeof(one)):NULL,*large_a=ok?tensor(&zero,sizeof(zero)):NULL,*large_b=ok?tensor(&zero,sizeof(zero)):NULL;ok=ok&&large_input&&large_bias&&large_a&&large_b;current=large_input;fg_vk_get_counters(context,&before);
    if(ok)ok=fg_vk_begin(context,&error)==FG_OK;
    for(uint32_t step=0;ok&&step<QSA_PREFILL_MAX_DISPATCHES;step++){fg_vk_tensor *target=(step&1u)?large_b:large_a;ok=fg_vk_add_f32(context,target,current,large_bias,1u,&error)==FG_OK;current=target;}
    if(ok)ok=fg_vk_end(context,&error)==FG_OK&&fg_vk_tensor_read(current,0,&large_got,sizeof(large_got),&error)==FG_OK;
    fg_vk_get_counters(context,&after);
    if(ok)ok=after.submissions-before.submissions==1u&&large_got==(float)QSA_PREFILL_MAX_DISPATCHES;
    if(fg_vk_batch_active(context)){fg_error ignored={0};fg_vk_abort(context,&ignored);}
    fg_vk_tensor_destroy(large_b);fg_vk_tensor_destroy(large_a);fg_vk_tensor_destroy(large_bias);fg_vk_tensor_destroy(large_input);fg_vk_tensor_destroy(abort_out);fg_vk_tensor_destroy(bc);fg_vk_tensor_destroy(ba);fg_vk_tensor_destroy(bb);fg_vk_tensor_destroy(bi);fg_vk_tensor_destroy(sc);fg_vk_tensor_destroy(sa);fg_vk_tensor_destroy(sb);fg_vk_tensor_destroy(si);return ok;
}

static const fg_vk_profile_kernel *find_profile_kernel(const fg_vk_profile *profile,const char *scope,const char *name){for(uint32_t i=0;i<profile->kernel_count;i++)if(strcmp(profile->kernels[i].scope,scope)==0&&strcmp(profile->kernels[i].name,name)==0)return &profile->kernels[i];return NULL;}
static int test_gpu_profile(void){
    enum{VALUES=256};float left[VALUES],right[VALUES];for(uint32_t i=0;i<VALUES;i++){left[i]=sinf((float)i*0.031f);right[i]=cosf((float)i*0.017f);}fg_vk_tensor *l=tensor(left,sizeof(left)),*r=tensor(right,sizeof(right)),*sum=tensor(NULL,sizeof(left)),*out=tensor(NULL,sizeof(left));fg_vk_profile profile={0};fg_vk_counters before={0},after={0};fg_vk_get_counters(context,&before);int ok=l&&r&&sum&&out&&!fg_vk_batch_active(context)&&!fg_vk_profile_active(context)&&fg_vk_profile_begin(context,&error)==FG_OK&&fg_vk_profile_active(context)&&fg_vk_begin(context,&error)==FG_OK&&fg_vk_begin(context,&error)==FG_OK&&fg_vk_batch_active(context)&&fg_vk_profile_set_scope(context,"sum",&error)==FG_OK&&fg_vk_add_f32(context,sum,l,r,VALUES,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&fg_vk_batch_active(context)&&fg_vk_profile_set_scope(context,"activation",&error)==FG_OK&&fg_vk_silu_scaled(context,out,sum,VALUES,1.0f,&error)==FG_OK&&fg_vk_end(context,&error)==FG_OK&&!fg_vk_batch_active(context)&&fg_vk_profile_end(context,&profile,&error)==FG_OK&&!fg_vk_profile_active(context);fg_vk_get_counters(context,&after);const fg_vk_profile_kernel *add=ok?find_profile_kernel(&profile,"sum","fg_add_f32.spv"):NULL,*silu=ok?find_profile_kernel(&profile,"activation","fg_silu_scaled.spv"):NULL;ok=ok&&after.submissions-before.submissions==2u&&after.dispatches-before.dispatches==2u&&profile.submissions==1u&&profile.dispatches==2u&&profile.gpu_ms>0.0&&profile.kernel_ms>0.0&&profile.gpu_ms>=profile.kernel_ms&&add&&add->invocations==1u&&add->gpu_ms>0.0&&silu&&silu->invocations==1u&&silu->gpu_ms>0.0;fg_vk_tensor_destroy(out);fg_vk_tensor_destroy(sum);fg_vk_tensor_destroy(r);fg_vk_tensor_destroy(l);return ok;
}

static uint32_t selected_test_count;
static bool test_selected(const char *name){
    const char *filter=getenv("DS4_REMOTE_TEST_FILTER");
    return !filter||!*filter||strcmp(filter,name)==0;
}
static int run_test(const char *name,int (*fn)(void)){if(!test_selected(name))return 1;selected_test_count++;fprintf(stderr,"  [%s] ... ",name);fflush(stderr);int ok=fn();fprintf(stderr,"%s\n",ok?"ok":"FAIL");return ok;}
static int run_test_i(const char *name,int (*fn)(int),int arg){if(!test_selected(name))return 1;selected_test_count++;fprintf(stderr,"  [%s(%d)] ... ",name,arg);fflush(stderr);int ok=fn(arg);fprintf(stderr,"%s\n",ok?"ok":"FAIL");return ok;}
int main(void){if(fg_vk_open(&context,&error)!=FG_OK){fprintf(stderr,"Vulkan unavailable: %s\n",error.message);return 77;}fprintf(stderr,"Flash Gordon Vulkan device: %s\n",fg_vk_device_name(context));int ok=1;
ok=run_test("memory_telemetry_and_canary",test_memory_telemetry_and_canary)&&ok;
ok=run_test("tensor_view_rebind",test_tensor_view_rebind)&&ok;
ok=run_test("batch_submission_parity",test_batch_submission_parity)&&ok;
ok=run_test("gpu_profile",test_gpu_profile)&&ok;
ok=run_test("q8_dense",test_q8_dense)&&ok;
ok=run_test("q8_dense_subgroup",test_q8_dense_subgroup)&&ok;
ok=run_test("q8_dense_cooked",test_q8_dense_cooked)&&ok;
ok=run_test("q8_subgroup_benchmark",test_q8_subgroup_benchmark)&&ok;
ok=run_test("q8_cooked_benchmark",test_q8_cooked_benchmark)&&ok;
ok=run_test("q8_embedding",test_q8_embedding)&&ok;
ok=run_test("hc_finalize",test_hc_finalize)&&ok;
ok=run_test("q8_k_quant",test_q8_k_quant)&&ok;
ok=run_test("q4_0_bytes",test_q4_0_bytes)&&ok;
ok=run_test("iq4_nl_dequant",test_iq4_nl_dequant)&&ok;
ok=run_test("ngram_direct_lookup",test_ngram_direct_lookup)&&ok;
ok=run_test("ngram_resident",test_ngram_resident)&&ok;
ok=run_test("ngram_prefill_lookup",test_ngram_prefill_lookup)&&ok;
ok=run_test("ple_decode",test_ple_decode)&&ok;
ok=run_test("ple_prefill_scan",test_ple_prefill_scan)&&ok;
ok=run_test("ple_prefill_t1_compat",test_ple_prefill_t1_compat)&&ok;
ok=run_test("qsa_quant_and_bf16",test_qsa_quant_and_bf16)&&ok;
ok=run_test("qsa_record_commit",test_qsa_record_commit)&&ok;
ok=run_test("qsa_tiered_record_commit",test_qsa_tiered_record_commit)&&ok;
ok=run_test("qsa_segmented_record_commit",test_qsa_segmented_record_commit)&&ok;
ok=run_test("qsa_resident_commit_exact",test_qsa_resident_commit_exact)&&ok;
ok=run_test("qsa_resident_early_exhaustive",test_qsa_resident_early_exhaustive)&&ok;
ok=run_test("qsa_resident_hierarchical_topk",test_qsa_resident_hierarchical_topk)&&ok;
ok=run_test("qsa_resident_causal_batch_attention",test_qsa_resident_batch_attention)&&ok;
ok=run_test("qsa_resident_t1_compat",test_qsa_resident_t1_compat)&&ok;
ok=run_test("qsa_record_gather",test_qsa_record_gather)&&ok;
ok=run_test("swiglu",test_swiglu)&&ok;
ok=run_test("dense_f32_and_silu",test_dense_f32_and_silu)&&ok;
ok=run_test("group_norm",test_group_norm)&&ok;
ok=run_test("gr_mix",test_gr_mix)&&ok;
ok=run_test("hc_inject_partial",test_hc_inject_partial)&&ok;
ok=run_test("gr_batch",test_gr_batch)&&ok;
ok=run_test("gr_partial_boundaries",test_gr_partial_boundaries)&&ok;
ok=run_test("q5_1_down",test_q5_1_down)&&ok;
ok=run_test("q5_1_down_cooked",test_q5_1_down_cooked)&&ok;
ok=run_test("q5_1_down_cooked_benchmark",test_q5_1_down_cooked_benchmark)&&ok;
ok=run_test("q8_0_down",test_q8_0_down)&&ok;
ok=run_test("pipeline_decode_down_formats",
            test_pipeline_decode_down_formats)&&ok;
ok=run_test("moe_reduce",test_moe_reduce)&&ok;
ok=run_test_i("kquant",test_kquant,12)&&ok;
ok=run_test_i("kquant",test_kquant,13)&&ok;
ok=run_test_i("kquant_cooked",test_kquant_cooked,12)&&ok;
ok=run_test_i("kquant_cooked",test_kquant_cooked,13)&&ok;
ok=run_test("kquant_cooked_benchmark",test_kquant_cooked_benchmark)&&ok;
ok=run_test_i("kquant_expert_major_batch",test_kquant_expert_major_batch,12)&&ok;
ok=run_test_i("kquant_expert_major_batch",test_kquant_expert_major_batch,13)&&ok;
ok=run_test("q8_cooked_token_tiles",test_q8_cooked_token_tiles)&&ok;
ok=run_test("decode_tile_schedule",test_decode_tile_schedule)&&ok;
ok=run_test("router_and_expert_packing",test_router_and_expert_packing)&&ok;
ok=run_test_i("grouped_kquant_prefill",test_grouped_kquant_prefill,12)&&ok;
ok=run_test_i("grouped_kquant_prefill",test_grouped_kquant_prefill,13)&&ok;
ok=run_test("grouped_down_prefill",test_grouped_down_prefill)&&ok;
ok=run_test("moe_prefill_scatter_reduce",test_moe_prefill_scatter_reduce)&&ok;
ok=run_test("expert_graph_replay",test_expert_graph_replay)&&ok;
ok=run_test("expert_graph_rejects_overlap",test_expert_graph_rejects_overlap)&&ok;
ok=run_test("gdn_project_cooked",test_gdn_project_cooked)&&ok;
ok=run_test("gdn_decode",test_gdn_decode)&&ok;
ok=run_test("gdn_algebraic",test_gdn_algebraic)&&ok;
ok=run_test("gdn_prefill_scan",test_gdn_prefill_scan)&&ok;
ok=run_test("gdn_pipeline_prefill_parity_zero",test_gdn_pipeline_prefill_parity_zero)&&ok;
ok=run_test("gdn_pipeline_prefill_parity_random",test_gdn_pipeline_prefill_parity_random)&&ok;
ok=run_test("gdn_pipeline_prefill_parity_extreme",test_gdn_pipeline_prefill_parity_extreme)&&ok;
ok=run_test("gdn_pipeline_prefill_composition",test_gdn_pipeline_prefill_composition)&&ok;
ok=run_test("gdn_pipeline_prefill_decode_compat",test_gdn_pipeline_prefill_decode_compat)&&ok;
ok=run_test("qsa_indexer",test_qsa_indexer)&&ok;
ok=run_test("qsa_segmented_index_score",test_qsa_segmented_index_score)&&ok;
ok=run_test("qsa_prefill_chunk_liveness",test_qsa_prefill_chunk_liveness)&&ok;
ok=run_test("output_topk",test_output_topk)&&ok;
ok=run_test("generation_topk_selector",test_generation_topk_selector)&&ok;
ok=run_test("output_argmax",test_output_argmax)&&ok;
ok=run_test("qsa_prefill_prepare",test_qsa_prefill_prepare)&&ok;
ok=run_test("qsa_attention_single",test_qsa_attention_single)&&ok;
ok=run_test("qsa_attention",test_qsa_attention)&&ok;
fg_vk_close(context);if(!selected_test_count){fprintf(stderr,"no Vulkan test matched DS4_REMOTE_TEST_FILTER\n");return 2;}if(!ok){fprintf(stderr,"native Vulkan oracle failed: %s\n",error.message);return 1;}puts("Flash Gordon native Vulkan production-dimension oracles: PASS");return 0;}
