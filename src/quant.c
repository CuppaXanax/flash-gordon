#include "fg_quant.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

typedef struct fg_block_q8_0 {uint16_t d;int8_t qs[FG_QK8_0];} fg_block_q8_0;
typedef struct fg_block_q8_k {float d;float dmin;int8_t qs[FG_QK8_K];int16_t bsums[FG_QK8_K/16u];} fg_block_q8_k;

float fg_f16_to_f32(uint16_t h){
    uint32_t s=(uint32_t)(h&0x8000u)<<16u,e=(h>>10u)&31u,m=h&1023u,bits;
    if(e==0){if(m==0)bits=s;else{e=113u;while((m&1024u)==0){m<<=1u;e--;}bits=s|(e<<23u)|((m&1023u)<<13u);}}
    else if(e==31u)bits=s|0x7f800000u|(m<<13u);else bits=s|((e+112u)<<23u)|(m<<13u);
    float value;memcpy(&value,&bits,sizeof(value));return value;
}

uint16_t fg_f32_to_f16(float value){
    uint32_t bits;memcpy(&bits,&value,sizeof(bits));uint32_t sign=(bits>>16u)&0x8000u,exp=(bits>>23u)&255u,mant=bits&0x7fffffu;
    if(exp==255u)return (uint16_t)(sign|0x7c00u|(mant?0x0200u:0u));
    int32_t e=(int32_t)exp-127+15;
    if(e>=31)return (uint16_t)(sign|0x7c00u);
    if(e<=0){if(e<-10)return (uint16_t)sign;mant|=0x800000u;uint32_t shift=(uint32_t)(14-e),half=mant>>shift,round=(mant>>(shift-1u))&1u,sticky=mant&((UINT32_C(1)<<(shift-1u))-1u);half+=round&&(sticky||(half&1u));return (uint16_t)(sign|half);}
    uint32_t half=(uint32_t)e<<10u|mant>>13u,round=(mant>>12u)&1u,sticky=mant&0xfffu;half+=round&&(sticky||(half&1u));return (uint16_t)(sign|half);
}

void fg_quantize_q8_0(const float *input,void *output,uint64_t values){
    fg_block_q8_0 *blocks=output;
    for(uint64_t b=0;b<values/FG_QK8_0;b++){
        float maximum=0.0f;for(uint32_t i=0;i<FG_QK8_0;i++)maximum=fmaxf(maximum,fabsf(input[b*FG_QK8_0+i]));
        float d=maximum/127.0f,inv=d?1.0f/d:0.0f;blocks[b].d=fg_f32_to_f16(d);
        for(uint32_t i=0;i<FG_QK8_0;i++)blocks[b].qs[i]=(int8_t)roundf(input[b*FG_QK8_0+i]*inv);
    }
}

void fg_quantize_q4_0(const float *input,void *output,uint64_t values){
    uint8_t *blocks=output;
    for(uint64_t b=0;b<values/FG_QK4_0;b++){
        const float *x=input+b*FG_QK4_0;float amax=0.0f,maximum=0.0f;
        for(uint32_t i=0;i<FG_QK4_0;i++)if(amax<fabsf(x[i])){amax=fabsf(x[i]);maximum=x[i];}
        float d=maximum/-8.0f,inv=d?1.0f/d:0.0f;uint16_t half=fg_f32_to_f16(d);uint8_t *block=blocks+b*18u;block[0]=(uint8_t)half;block[1]=(uint8_t)(half>>8u);
        for(uint32_t i=0;i<FG_QK4_0/2u;i++){uint8_t lo=(uint8_t)fminf(15.0f,(float)(int8_t)(x[i]*inv+8.5f));uint8_t hi=(uint8_t)fminf(15.0f,(float)(int8_t)(x[i+16u]*inv+8.5f));block[2u+i]=(uint8_t)(lo|(hi<<4u));}
    }
}

void fg_dequantize_q8_0(const void *input,float *output,uint64_t values){
    const fg_block_q8_0 *blocks=input;for(uint64_t b=0;b<values/FG_QK8_0;b++){float d=fg_f16_to_f32(blocks[b].d);for(uint32_t i=0;i<FG_QK8_0;i++)output[b*FG_QK8_0+i]=d*(float)blocks[b].qs[i];}
}

void fg_dequantize_q4_0(const void *input,float *output,uint64_t values){
    const uint8_t *blocks=input;for(uint64_t b=0;b<values/FG_QK4_0;b++){const uint8_t *block=blocks+b*18u;float d=fg_f16_to_f32((uint16_t)block[0]|((uint16_t)block[1]<<8u));for(uint32_t i=0;i<16u;i++){output[b*32u+i]=d*(float)((int32_t)(block[2u+i]&15u)-8);output[b*32u+16u+i]=d*(float)((int32_t)(block[2u+i]>>4u)-8);}}
}

void fg_dequantize_iq4_nl(const void *input,float *output,uint64_t values){
    static const int8_t grid[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const uint8_t *blocks=input;for(uint64_t b=0;b<values/32u;b++){const uint8_t *block=blocks+b*18u;float d=fg_f16_to_f32((uint16_t)block[0]|((uint16_t)block[1]<<8u));for(uint32_t i=0;i<16u;i++){uint8_t packed=block[2u+i];output[b*32u+i]=d*(float)grid[packed&15u];output[b*32u+16u+i]=d*(float)grid[packed>>4u];}}
}

void fg_quantize_q8_k(const float *input,void *output,uint64_t values){
    fg_block_q8_k *blocks=output;
    for(uint64_t b=0;b<values/FG_QK8_K;b++){
        float maximum=0.0f;for(uint32_t i=0;i<FG_QK8_K;i++)maximum=fmaxf(maximum,fabsf(input[b*FG_QK8_K+i]));
        float d=maximum/127.0f,inv=d?1.0f/d:0.0f;blocks[b].d=d;blocks[b].dmin=0.0f;
        for(uint32_t i=0;i<FG_QK8_K;i++)blocks[b].qs[i]=(int8_t)nearbyintf(input[b*FG_QK8_K+i]*inv);
        for(uint32_t g=0;g<FG_QK8_K/16u;g++){int32_t sum=0;for(uint32_t i=0;i<16u;i++)sum+=blocks[b].qs[g*16u+i];blocks[b].bsums[g]=(int16_t)sum;}
    }
}

uint64_t fg_q8_0_cooked_tile_bytes(uint32_t input_width){if(!input_width||input_width%FG_QK8_0)return 0;uint64_t blocks=input_width/FG_QK8_0,mask=FG_Q8_0_COOK_ALIGNMENT-1u,quant_offset=(FG_Q8_0_COOK_ROWS*blocks*sizeof(uint16_t)+mask)&~mask;return(quant_offset+FG_Q8_0_COOK_ROWS*blocks*FG_QK8_0+mask)&~mask;}

uint64_t fg_q8_0_cooked_matrix_bytes(uint32_t input_width,uint32_t rows){uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(input_width),tiles=((uint64_t)rows+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS;if(!tile_bytes||!rows||tiles>UINT64_MAX/tile_bytes)return 0;return tiles*tile_bytes;}

bool fg_cook_q8_0_rows(const void *input,void *output,uint64_t output_bytes,uint32_t input_width,uint32_t rows){uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(input_width),required=fg_q8_0_cooked_matrix_bytes(input_width,rows);if(!input||!output||!tile_bytes||!required||output_bytes<required)return false;uint32_t blocks=input_width/FG_QK8_0;uint64_t mask=FG_Q8_0_COOK_ALIGNMENT-1u,quant_offset=(FG_Q8_0_COOK_ROWS*(uint64_t)blocks*sizeof(uint16_t)+mask)&~mask;const uint8_t *source=input;uint8_t *destination=output;memset(destination,0,(size_t)required);for(uint32_t row=0;row<rows;row++)for(uint32_t block=0;block<blocks;block++){const uint8_t *src=source+((uint64_t)row*blocks+block)*FG_Q8_0_BLOCK_BYTES;uint8_t *dst=destination+(uint64_t)(row/FG_Q8_0_COOK_ROWS)*tile_bytes;uint32_t tile_row=row%FG_Q8_0_COOK_ROWS;memcpy(dst+((uint64_t)block*FG_Q8_0_COOK_ROWS+tile_row)*sizeof(uint16_t),src,sizeof(uint16_t));memcpy(dst+quant_offset+((uint64_t)tile_row*blocks+block)*FG_QK8_0,src+sizeof(uint16_t),FG_QK8_0);}return true;}

float fg_dot_q8_0(const void *lhs,const void *rhs,uint64_t values){
    const fg_block_q8_0 *a=lhs,*b=rhs;float result=0.0f;
    for(uint64_t block=0;block<values/FG_QK8_0;block++){int32_t sum=0;for(uint32_t i=0;i<FG_QK8_0;i++)sum+=(int32_t)a[block].qs[i]*b[block].qs[i];result+=fg_f16_to_f32(a[block].d)*fg_f16_to_f32(b[block].d)*(float)sum;}
    return result;
}

static void k_scale_min(const uint8_t *block,uint32_t group,uint32_t *scale,uint32_t *minimum){
    uint32_t s0=block[4u+group];
    if(group<4u){*scale=s0&63u;*minimum=block[8u+group]&63u;}
    else{uint32_t upper=block[8u+group],lower=block[group];*scale=(upper&15u)|((lower>>6u)<<4u);*minimum=(upper>>4u)|((s0>>6u)<<4u);}
}

static uint32_t q4_k_value(const uint8_t *block,uint32_t index){uint32_t group=index>>5u,i=index&31u;uint8_t packed=block[16u+(group>>1u)*32u+i];return (group&1u)?packed>>4u:packed&15u;}
static uint32_t q5_k_value(const uint8_t *block,uint32_t index){uint32_t group=index>>5u,i=index&31u;uint8_t packed=block[48u+(group>>1u)*32u+i];uint32_t low=(group&1u)?packed>>4u:packed&15u,high=(block[16u+i]>>group)&1u;return low|(high<<4u);}

static float dot_k_q8(const uint8_t *weights,const fg_block_q8_k *activation,uint64_t values,bool q5){
    const uint32_t block_bytes=q5?176u:144u;float result=0.0f;
    for(uint64_t block_index=0;block_index<values/FG_QK8_K;block_index++){
        const uint8_t *block=weights+block_index*block_bytes;float d=fg_f16_to_f32((uint16_t)block[0]|(uint16_t)((uint16_t)block[1]<<8u));float dmin=fg_f16_to_f32((uint16_t)block[2]|(uint16_t)((uint16_t)block[3]<<8u));float xd=activation[block_index].d;
        for(uint32_t group=0;group<8u;group++){uint32_t scale,minimum;k_scale_min(block,group,&scale,&minimum);for(uint32_t i=0;i<32u;i++){uint32_t index=group*32u+i;uint32_t quant=q5?q5_k_value(block,index):q4_k_value(block,index);float weight=d*(float)(scale*quant)-dmin*(float)minimum;result=fmaf(weight,xd*(float)activation[block_index].qs[index],result);}}
    }
    return result;
}
float fg_dot_q4_k_q8_k(const void *weights,const void *activation,uint64_t values){return dot_k_q8(weights,activation,values,false);}
float fg_dot_q5_k_q8_k(const void *weights,const void *activation,uint64_t values){return dot_k_q8(weights,activation,values,true);}
