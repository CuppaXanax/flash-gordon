#include "fg_q38_math.h"
#include "fg_q38_schema.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float fp16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16u;
    uint32_t exp=(h>>10u)&31u;
    uint32_t frac=h&1023u;
    uint32_t bits;
    if(exp==0){
        if(frac==0)bits=sign;
        else{uint32_t shift=0;while((frac&0x400u)==0){frac<<=1u;shift++;}frac&=0x3ffu;bits=sign|((127u-15u-shift)<<23u)|(frac<<13u);}
    }else if(exp==31u)bits=sign|0x7f800000u|(frac<<13u);
    else bits=sign|((exp+112u)<<23u)|(frac<<13u);
    union{uint32_t u;float f;}v={bits};return v.f;
}

float fg_q5_1_dot_f32(const uint8_t *row,const float *x,uint32_t n){
    if(!row||!x||n==0||(n%32u)!=0)return NAN;
    float total=0.0f;
    for(uint32_t b=0;b<n/32u;b++){
        const uint8_t *p=row+(uint64_t)b*24u;
        float d=fp16_to_f32((uint16_t)p[0]|((uint16_t)p[1]<<8u));
        float m=fp16_to_f32((uint16_t)p[2]|((uint16_t)p[3]<<8u));
        for(uint32_t i=0;i<32u;i++){
            uint32_t hi=(p[4u+(i>>3u)]>>(i&7u))&1u;
            uint8_t packed=p[8u+(i&15u)];
            uint32_t lo=i<16u?(packed&15u):(packed>>4u);
            total=fmaf(fmaf(d,(float)(lo|(hi<<4u)),m),x[b*32u+i],total);
        }
    }
    return total;
}

void fg_q38_group_rms_norm(float *out,const float *x,const float *weight,uint32_t groups,uint32_t width,float eps){
    if(!out||!x||groups==0||width==0)return;
    for(uint32_t g=0;g<groups;g++){
        const float *src=x+(uint64_t)g*width;float *dst=out+(uint64_t)g*width;double sum=0.0;
        for(uint32_t i=0;i<width;i++)sum+=(double)src[i]*(double)src[i];
        float scale=1.0f/sqrtf((float)(sum/(double)width)+eps);
        for(uint32_t i=0;i<width;i++){uint64_t j=(uint64_t)g*width+i;dst[i]=src[i]*scale*(weight?weight[j]:1.0f);}
    }
}

fg_status fg_q38_router_topk(const float *logits,uint32_t count,uint32_t k,uint32_t *ids,float *weights,fg_error *err){
    if(!logits||!ids||!weights||count==0||k==0||k>count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid router top-k arguments");return FG_ERR_ARGUMENT;}
    float maxv=-INFINITY;for(uint32_t i=0;i<count;i++)if(logits[i]>maxv)maxv=logits[i];
    double denom=0.0;for(uint32_t i=0;i<count;i++)denom+=exp((double)logits[i]-maxv);
    double selected_sum=0.0;
    for(uint32_t slot=0;slot<k;slot++){
        uint32_t best=UINT32_MAX;float bestv=-INFINITY;
        for(uint32_t i=0;i<count;i++){
            bool used=false;for(uint32_t j=0;j<slot;j++)if(ids[j]==i){used=true;break;}
            if(!used&&(logits[i]>bestv||(logits[i]==bestv&&i<best))){best=i;bestv=logits[i];}
        }
        ids[slot]=best;weights[slot]=(float)(exp((double)bestv-maxv)/denom);selected_sum+=weights[slot];
    }
    if(!(selected_sum>0.0)){fg_error_set(err,FG_ERR_FORMAT,"router probabilities are not finite");return FG_ERR_FORMAT;}
    for(uint32_t slot=0;slot<k;slot++)weights[slot]=(float)(weights[slot]/selected_sum);
    return FG_OK;
}

fg_status fg_q38_rms_mrope(float *vector,uint32_t heads,uint32_t width,const float *weight,const uint32_t position[3],fg_error *err){
    if(!vector||!weight||!position||!heads||width<FG_Q38_ROTARY_WIDTH){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Qwen RMS/MRoPE arguments");return FG_ERR_ARGUMENT;}
    for(uint32_t head=0;head<heads;head++){
        float *x=vector+(uint64_t)head*width;double ss=0.0;for(uint32_t i=0;i<width;i++)ss+=(double)x[i]*x[i];float scale=1.0f/sqrtf((float)(ss/width)+1e-6f);for(uint32_t i=0;i<width;i++)x[i]*=scale*weight[i];
        float first[FG_Q38_ROTARY_WIDTH/2u],second[FG_Q38_ROTARY_WIDTH/2u];memcpy(first,x,sizeof(first));memcpy(second,x+FG_Q38_ROTARY_WIDTH/2u,sizeof(second));
        for(uint32_t i=0;i<FG_Q38_ROTARY_WIDTH/2u;i++){
            uint32_t axis=0;if(i%3u==1u&&i<33u)axis=1u;else if(i%3u==2u&&i<30u)axis=2u;
            float angle=(float)position[axis]/powf(FG_Q38_ROPE_THETA,(float)(2u*i)/(float)FG_Q38_ROTARY_WIDTH),c=cosf(angle),s=sinf(angle);
            x[i]=first[i]*c-second[i]*s;x[i+FG_Q38_ROTARY_WIDTH/2u]=second[i]*c+first[i]*s;
        }
    }
    return FG_OK;
}

typedef struct fg_qsa_score {float value;uint32_t block;} fg_qsa_score;
static int score_compare(const void *a,const void *b){const fg_qsa_score *x=a,*y=b;if(x->value>y->value)return -1;if(x->value<y->value)return 1;return x->block<y->block?-1:x->block>y->block;}

fg_status fg_q38_qsa_index_select_reference(const float query[FG_Q38_INDEX_QUERY_WIDTH],const float *raw_keys,uint32_t tokens,const float q_norm[FG_Q38_INDEX_WIDTH],const float k_norm[FG_Q38_INDEX_WIDTH],uint32_t *selected,uint32_t capacity,uint32_t *selected_count,fg_error *err){
    if(!query||!raw_keys||!tokens||!q_norm||!k_norm||!selected||!selected_count){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA indexer reference arguments");return FG_ERR_ARGUMENT;}uint32_t blocks=tokens/FG_Q38_QSA_COMPRESS_RATIO,tail=tokens%FG_Q38_QSA_COMPRESS_RATIO,chosen=blocks<FG_Q38_INDEX_BUDGET/FG_Q38_QSA_COMPRESS_RATIO?blocks:FG_Q38_INDEX_BUDGET/FG_Q38_QSA_COMPRESS_RATIO,required=chosen*FG_Q38_QSA_COMPRESS_RATIO+tail;if(capacity<required){fg_error_set(err,FG_ERR_LIMIT,"QSA selected-token buffer is too small");return FG_ERR_LIMIT;}
    float q[FG_Q38_INDEX_QUERY_WIDTH];memcpy(q,query,sizeof(q));uint32_t current_position[3]={tokens-1u,tokens-1u,tokens-1u};fg_status status=fg_q38_rms_mrope(q,FG_Q38_INDEX_HEADS,FG_Q38_INDEX_WIDTH,q_norm,current_position,err);if(status!=FG_OK)return status;
    fg_qsa_score *scores=blocks?malloc((size_t)blocks*sizeof(*scores)):NULL;if(blocks&&!scores){fg_error_set(err,FG_ERR_OOM,"allocate QSA reference block scores");return FG_ERR_OOM;}
    for(uint32_t block=0;block<blocks;block++){
        float key[FG_Q38_INDEX_WIDTH]={0};for(uint32_t token=0;token<FG_Q38_QSA_COMPRESS_RATIO;token++)for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)key[i]+=raw_keys[((uint64_t)block*FG_Q38_QSA_COMPRESS_RATIO+token)*FG_Q38_INDEX_WIDTH+i]*0.25f;uint32_t start=block*FG_Q38_QSA_COMPRESS_RATIO,position[3]={start,start,start};status=fg_q38_rms_mrope(key,1u,FG_Q38_INDEX_WIDTH,k_norm,position,err);if(status!=FG_OK){free(scores);return status;}float score=0.0f;for(uint32_t head=0;head<FG_Q38_INDEX_HEADS;head++){float dot=0.0f;for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)dot=fmaf(q[head*FG_Q38_INDEX_WIDTH+i],key[i],dot);score+=fmaxf(dot,0.0f);}scores[block]=(fg_qsa_score){score/sqrtf((float)FG_Q38_INDEX_WIDTH),block};
    }
    qsort(scores,blocks,sizeof(*scores),score_compare);uint32_t out=0;for(uint32_t i=0;i<chosen;i++)for(uint32_t token=0;token<FG_Q38_QSA_COMPRESS_RATIO;token++)selected[out++]=scores[i].block*FG_Q38_QSA_COMPRESS_RATIO+token;for(uint32_t token=tokens-tail;token<tokens;token++)selected[out++]=token;free(scores);*selected_count=out;return FG_OK;
}
