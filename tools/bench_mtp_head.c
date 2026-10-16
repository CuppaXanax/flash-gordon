#include "fg_mtp_head.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void binding_test(fg_vk_context *vk,fg_error *err){
    fg_manifest *m=malloc(sizeof(*m));
    if(!m){fprintf(stderr,"manifest OOM\n");exit(1);}
    fg_manifest_init(m);m->tensor_count=1;
    fg_tensor_record *r=&m->tensors[0];
    snprintf(r->name,sizeof(r->name),"borrowed.weight");
    r->bytes=16;r->ggml_type=0;r->dims=1;r->shape[0]=4;
    fg_vk_tensor *w=NULL;
    if(fg_vk_tensor_create(vk,16,&w,err)!=FG_OK){fprintf(stderr,"%s\n",err->message);exit(1);}
    float input[4]={1,2,3,4},output[4]={0};
    if(fg_vk_tensor_write(w,0,input,sizeof(input),err)!=FG_OK)exit(1);
    fg_model *view=NULL;fg_vk_tensor *weights[]={w};
    if(fg_model_bind_view(&view,vk,m,0,weights,err)!=FG_OK)exit(1);
    fg_vk_tensor_destroy(w);
    if(fg_vk_tensor_read(fg_model_tensor(view,r->name),0,output,sizeof(output),err)!=FG_OK||
       memcmp(input,output,sizeof(input))||fg_model_weight_bytes(view)!=16)exit(1);
    fg_model_close(view);
    if(fg_vk_tensor_create(vk,16,&w,err)!=FG_OK)exit(1);
    fg_vk_tensor_destroy(w);
    m->layer_owner[FG_LAYER_COUNT-1u]=2;
    fg_mtp_head *h=NULL;
    if(fg_mtp_head_open(&h,vk,m,1,"does-not-exist",16,err)!=FG_ERR_ARGUMENT||h){
        fprintf(stderr,"MTP opened on the wrong final-layer owner\n");exit(1);
    }
    free(m);
    printf("{\"kind\":\"borrowed_model_test\",\"retained_tensor\":true,\"context_alive\":true,\"wrong_owner_rejected\":true}\n");
}

static void check(fg_status status,fg_error *err){
    if(status!=FG_OK){fprintf(stderr,"MTP head: %s\n",err->message);exit(1);}
}
static double now_us(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3;
}
static fg_vk_tensor *tensor(fg_vk_context *vk,uint64_t values,fg_error *err){
    fg_vk_tensor *t=NULL;check(fg_vk_tensor_create(vk,values*4u,&t,err),err);return t;
}
static fg_status dump_stage(void *context,const char *stage,const fg_vk_tensor *values,
    uint32_t count,fg_error *err){
    FILE *f=context;uint32_t length=(uint32_t)strlen(stage);
    float *data=malloc((size_t)count*4u);
    if(!data){fg_error_set(err,FG_ERR_OOM,"native trace");return FG_ERR_OOM;}
    fg_status status=fg_vk_tensor_read(values,0,data,(uint64_t)count*4u,err);
    if(status==FG_OK&&(fwrite(&length,4,1,f)!=1||fwrite(&count,4,1,f)!=1||
       fwrite(stage,1,length,f)!=length||fwrite(data,4,count,f)!=count)){
        fg_error_set(err,FG_ERR_IO,"write native trace");status=FG_ERR_IO;
    }
    free(data);return status;
}
static float signal(uint64_t i,uint32_t seed){
    return sinf((float)(i+1u)*(0.0031f+seed*0.00017f))*0.25f;
}
static bool parse_count(const char *text,uint32_t *value,uint32_t maximum){
    if(*text<'0'||*text>'9')return false;
    char *end=NULL;errno=0;unsigned long parsed=strtoul(text,&end,10);
    if(errno||*end||parsed>maximum)return false;
    *value=(uint32_t)parsed;return true;
}

static void attention_test(fg_vk_context *vk,fg_error *err){
    fg_vk_tensor *q=tensor(vk,6144,err),*gate=tensor(vk,6144,err);
    fg_vk_tensor *key=tensor(vk,512,err),*value=tensor(vk,512,err);
    fg_vk_tensor *keys=tensor(vk,16u*512u,err),*values=tensor(vk,16u*512u,err);
    fg_vk_tensor *out=tensor(vk,6144,err);
    fg_vk_tensor *position=tensor(vk,1,err);
    float queries[6144],gates[6144],ks[16][512],vs[16][512],actual[6144];
    for(uint32_t i=0;i<6144u;i++){queries[i]=signal(i,3);gates[i]=signal(i,7);}
    check(fg_vk_tensor_write(q,0,queries,sizeof(queries),err),err);
    check(fg_vk_tensor_write(gate,0,gates,sizeof(gates),err),err);
    double maximum=0;
    for(uint32_t cell=0;cell<16u;cell++){
        for(uint32_t j=0;j<512u;j++){ks[cell][j]=signal(j,cell+11);vs[cell][j]=signal(j,cell+31);}
        check(fg_vk_tensor_write(key,0,ks[cell],sizeof(ks[cell]),err),err);
        check(fg_vk_tensor_write(value,0,vs[cell],sizeof(vs[cell]),err),err);
        check(fg_vk_mtp_attention(vk,out,keys,values,q,gate,key,value,cell,err),err);
        check(fg_vk_tensor_read(out,0,actual,sizeof(actual),err),err);
        float direct[6144];memcpy(direct,actual,sizeof(direct));
        check(fg_vk_tensor_write(position,0,&cell,4u,err),err);
        check(fg_vk_mtp_attention_indirect(vk,out,keys,values,q,gate,key,value,position,16u,err),err);
        check(fg_vk_tensor_read(out,0,actual,sizeof(actual),err),err);
        if(memcmp(direct,actual,sizeof(direct))){fprintf(stderr,"indirect draft attention mismatch\n");exit(1);}
        for(uint32_t head=0;head<24u;head++){
            double scores[16],sum=0,max=-INFINITY;uint32_t kv=head/12u;
            for(uint32_t t=0;t<=cell;t++){
                double dot=0;
                for(uint32_t j=0;j<256u;j++)dot+=(double)queries[head*256u+j]*ks[t][kv*256u+j];
                scores[t]=dot/16.0;if(scores[t]>max)max=scores[t];
            }
            for(uint32_t t=0;t<=cell;t++){scores[t]=exp(scores[t]-max);sum+=scores[t];}
            for(uint32_t j=0;j<256u;j++){
                double weighted=0;
                for(uint32_t t=0;t<=cell;t++)weighted+=scores[t]*vs[t][kv*256u+j];
                double expected=(weighted/sum)/(1.0+exp(-(double)gates[head*256u+j]));
                double delta=fabs(actual[head*256u+j]-expected);
                if(!isfinite(actual[head*256u+j])||delta>1e-5){
                    fprintf(stderr,"dense draft attention mismatch cell=%u head=%u dim=%u error=%g\n",cell,head,j,delta);exit(1);
                }
                if(delta>maximum)maximum=delta;
            }
        }
    }
    float suffix[6144],prefix[6144];
    check(fg_vk_tensor_read(out,0,suffix,sizeof(suffix),err),err);
    check(fg_vk_tensor_write(key,0,ks[3],sizeof(ks[3]),err),err);
    check(fg_vk_tensor_write(value,0,vs[3],sizeof(vs[3]),err),err);
    check(fg_vk_mtp_attention(vk,out,keys,values,q,gate,key,value,3u,err),err);
    check(fg_vk_tensor_read(out,0,prefix,sizeof(prefix),err),err);
    if(!memcmp(prefix,suffix,sizeof(prefix))){fprintf(stderr,"draft attention ignored frontier\n");exit(1);}
    /* Rebuild a short prefix over hostile future values: they must not enter
     * the causal result after a frontier truncation. */
    memset((char *)fg_vk_tensor_map(keys)+4u*512u*4u,0x7f,12u*512u*4u);
    memset((char *)fg_vk_tensor_map(values)+4u*512u*4u,0x7f,12u*512u*4u);
    check(fg_vk_mtp_attention(vk,out,keys,values,q,gate,key,value,3u,err),err);
    check(fg_vk_tensor_read(out,0,suffix,sizeof(suffix),err),err);
    if(memcmp(prefix,suffix,sizeof(prefix))){fprintf(stderr,"draft attention read future KV\n");exit(1);}
    uint32_t cell=3u;check(fg_vk_tensor_write(position,0,&cell,4u,err),err);
    check(fg_vk_mtp_attention_indirect(vk,out,keys,values,q,gate,key,value,position,16u,err),err);
    check(fg_vk_tensor_read(out,0,suffix,sizeof(suffix),err),err);
    if(memcmp(prefix,suffix,sizeof(prefix))){fprintf(stderr,"indirect draft attention read future KV\n");exit(1);}
    fg_vk_tensor *all[]={q,gate,key,value,keys,values,out,position};
    for(uint32_t i=0;i<sizeof(all)/sizeof(all[0]);i++)fg_vk_tensor_destroy(all[i]);
    printf("{\"kind\":\"dense_draft_attention_test\",\"rows\":16,\"max_abs\":%.9g,\"future_mask_exact\":true}\n",maximum);
}

static void vocab_test(fg_vk_context *vk,fg_error *err){
    const uint32_t ids[7]={71,65,80,50,9,40,3};
    float weights[7u*64u],input[64],reference[7],best=0;
    for(uint32_t i=0;i<7u*64u;i++)weights[i]=-1.0f;
    for(uint32_t i=0;i<64u;i++)input[i]=1.0f;
    uint8_t raw[7u*2u*34u];fg_quantize_q8_0(weights,raw,7u*64u);
    uint64_t bytes=fg_q8_0_cooked_matrix_bytes(64u,7u);fg_vk_tensor *w=NULL;
    check(fg_vk_tensor_create(vk,bytes,&w,err),err);
    if(!fg_cook_q8_0_rows(raw,fg_vk_tensor_map(w),bytes,64u,7u))exit(1);
    fg_vk_tensor_set_format(w,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
    fg_vk_tensor *x=tensor(vk,64u,err),*map=tensor(vk,7u,err),*logits=tensor(vk,7u,err);
    fg_vk_tensor *s=tensor(vk,2u,err),*indices=tensor(vk,2u,err);
    fg_vk_tensor *result=tensor(vk,1u,err),*result_id=tensor(vk,1u,err);
    check(fg_vk_tensor_write(x,0,input,sizeof(input),err),err);
    check(fg_vk_tensor_write(map,0,ids,sizeof(ids),err),err);
    check(fg_vk_dense_q8_0_f32(vk,logits,w,x,64u,7u,1u,1.0f,err),err);
    check(fg_vk_tensor_read(logits,0,reference,sizeof(reference),err),err);
    check(fg_vk_begin(vk,err),err);
    check(fg_vk_mtp_vocab_q8(vk,s,indices,w,x,map,64u,7u,err),err);
    uint32_t count=0,id=UINT32_MAX;
    check(fg_vk_argmax_reduce(vk,result,result_id,s,indices,1u,&count,err),err);
    check(fg_vk_end(vk,err),err);
    check(fg_vk_tensor_read(result,0,&best,4u,err),err);
    check(fg_vk_tensor_read(result_id,0,&id,4u,err),err);
    if(count!=1u||id!=3u||fabsf(best-reference[6])>1e-4f){
        fprintf(stderr,"fused vocabulary tail/tie mismatch id=%u score=%g expected=%g\n",id,best,reference[6]);exit(1);
    }
    input[0]=NAN;check(fg_vk_tensor_write(x,0,input,sizeof(input),err),err);
    check(fg_vk_begin(vk,err),err);
    check(fg_vk_mtp_vocab_q8(vk,s,indices,w,x,map,64u,7u,err),err);
    check(fg_vk_argmax_reduce(vk,result,result_id,s,indices,1u,&count,err),err);
    check(fg_vk_end(vk,err),err);
    check(fg_vk_tensor_read(result,0,&best,4u,err),err);
    if(isfinite(best)){fprintf(stderr,"fused vocabulary concealed a nonfinite logit\n");exit(1);}
    input[0]=1.0f;check(fg_vk_tensor_write(x,0,input,4u,err),err);
    fg_vk_tensor *copy_source=tensor(vk,64u,err);
    for(uint32_t slot=FG_VK_MTP_STATIC_SLOT;slot<FG_VK_STATIC_SLOTS;slot++){
        input[0]=1.0f;check(fg_vk_tensor_write(x,0,input,4u,err),err);
        check(fg_vk_static_claim(vk,slot,err),err);
        if(fg_vk_static_claim(vk,slot,err)!=FG_ERR_MISMATCH)exit(1);
        check(fg_vk_static_begin(vk,slot,err),err);
        check(fg_vk_add_f32(vk,result,x,x,1u,err),err);
        check(fg_vk_static_end(vk,slot,err),err);
        check(fg_vk_static_submit(vk,slot,err),err);
        check(fg_vk_static_wait(vk,slot,err),err);
        check(fg_vk_tensor_read(result,0,&best,4u,err),err);
        if(best!=2.0f){fprintf(stderr,"private MTP replay slot failed\n");exit(1);}
        for(uint32_t iteration=0;iteration<2u;iteration++){
            input[0]=3.0f+iteration;
            check(fg_vk_tensor_write(copy_source,0,input,sizeof(input),err),err);
            fg_vk_tensor *dest[]={x};const fg_vk_tensor *source[]={copy_source};
            check(fg_vk_static_copy_run(vk,slot,dest,source,1u,err),err);
            check(fg_vk_tensor_read(result,0,&best,4u,err),err);
            if(best!=2.0f*input[0]){fprintf(stderr,"copied static graph input was stale\n");exit(1);}
        }
        input[0]=1.0f;check(fg_vk_tensor_write(x,0,input,sizeof(input),err),err);
        check(fg_vk_static_begin(vk,0u,err),err);
        check(fg_vk_add_f32(vk,copy_source,x,x,64u,err),err);
        check(fg_vk_static_end(vk,0u,err),err);
        check(fg_vk_begin(vk,err),err);
        check(fg_vk_add_f32(vk,result,x,x,1u,err),err);
        check(fg_vk_flush(vk,err),err);
        check(fg_vk_static_submit(vk,0u,err),err);
        fg_vk_tensor *dest[]={x};const fg_vk_tensor *source[]={copy_source};
        check(fg_vk_static_copy_run(vk,slot,dest,source,1u,err),err);
        check(fg_vk_tensor_read(result,0,&best,4u,err),err);
        if(best!=4.0f){fprintf(stderr,"copied graph ran before deferred owner input\n");exit(1);}
        check(fg_vk_static_reset(vk,0u,err),err);
        check(fg_vk_static_release(vk,slot,err),err);
    }
    fg_vk_tensor *all[]={w,x,map,logits,s,indices,result,result_id,copy_source};
    for(uint32_t i=0;i<sizeof(all)/sizeof(all[0]);i++)fg_vk_tensor_destroy(all[i]);
    printf("{\"kind\":\"fused_draft_vocab_test\",\"negative_tail_and_global_ties\":true,\"nonfinite_propagation\":true,\"private_slot_exclusion\":true}\n");
}

static void vocab_random_test(fg_vk_context *vk,uint32_t rows,uint32_t width,fg_error *err){
    const uint32_t partials=(rows+7u)/8u;
    float *weights=malloc((size_t)rows*width*4u),*reference=malloc(rows*4u);
    float *actual=malloc(partials*4u),*input=malloc(width*4u);
    uint32_t *map=malloc(rows*4u),*ids=malloc(partials*4u);
    uint8_t *raw=malloc((size_t)rows*width/32u*34u);
    if(!weights||!reference||!actual||!input||!map||!ids||!raw)exit(1);
    for(uint32_t i=0;i<rows*width;i++)weights[i]=signal(i,17u);
    for(uint32_t i=0;i<width;i++)input[i]=signal(i,29u);
    for(uint32_t i=0;i<rows;i++)map[i]=rows-1u-i;
    fg_quantize_q8_0(weights,raw,(uint64_t)rows*width);
    uint64_t bytes=fg_q8_0_cooked_matrix_bytes(width,rows);fg_vk_tensor *w=NULL;
    check(fg_vk_tensor_create(vk,bytes,&w,err),err);
    if(!fg_cook_q8_0_rows(raw,fg_vk_tensor_map(w),bytes,width,rows))exit(1);
    fg_vk_tensor_set_format(w,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
    fg_dequantize_q8_0(raw,weights,(uint64_t)rows*width);
    uint32_t expected=0;
    for(uint32_t row=0;row<rows;row++){
        float sum=0;for(uint32_t i=0;i<width;i++)sum=fmaf(weights[(uint64_t)row*width+i],input[i],sum);
        reference[row]=sum;
        if(sum>reference[expected]||(sum==reference[expected]&&map[row]<map[expected]))expected=row;
    }
    fg_vk_tensor *x=tensor(vk,width,err),*tokens=tensor(vk,rows,err);
    fg_vk_tensor *scores[]={tensor(vk,partials,err),tensor(vk,partials,err)};
    fg_vk_tensor *indices[]={tensor(vk,partials,err),tensor(vk,partials,err)};
    check(fg_vk_tensor_write(x,0,input,width*4u,err),err);
    check(fg_vk_tensor_write(tokens,0,map,rows*4u,err),err);
    check(fg_vk_mtp_vocab_q8(vk,scores[0],indices[0],w,x,tokens,width,rows,err),err);
    check(fg_vk_tensor_read(scores[0],0,actual,partials*4u,err),err);
    check(fg_vk_tensor_read(indices[0],0,ids,partials*4u,err),err);
    double maximum=0;
    for(uint32_t group=0;group<partials;group++){
        uint32_t best=group*8u;
        for(uint32_t row=best;row<rows&&row<group*8u+8u;row++)
            if(reference[row]>reference[best]||(reference[row]==reference[best]&&map[row]<map[best]))best=row;
        double delta=fabs(actual[group]-reference[best])/fmax(1.0,fabs(reference[best]));
        if(ids[group]!=map[best]||!isfinite(actual[group])||delta>1e-4){
            fprintf(stderr,"fused vocabulary scalar oracle failed group=%u error=%g\n",group,delta);exit(1);
        }
        if(delta>maximum)maximum=delta;
    }
    uint32_t count=partials,bank=0;
    check(fg_vk_begin(vk,err),err);
    while(count>1u){
        uint32_t next=0;
        check(fg_vk_argmax_reduce(vk,scores[bank^1u],indices[bank^1u],scores[bank],indices[bank],count,&next,err),err);
        count=next;bank^=1u;
    }
    check(fg_vk_end(vk,err),err);
    uint32_t token=UINT32_MAX;check(fg_vk_tensor_read(indices[bank],0,&token,4u,err),err);
    if(token!=map[expected]){fprintf(stderr,"multi-stage vocabulary argmax failed\n");exit(1);}
    fg_vk_tensor *all[]={w,x,tokens,scores[0],scores[1],indices[0],indices[1]};
    for(uint32_t i=0;i<sizeof(all)/sizeof(all[0]);i++)fg_vk_tensor_destroy(all[i]);
    free(weights);free(reference);free(actual);free(input);free(map);free(ids);free(raw);
    printf("{\"kind\":\"fused_draft_vocab_scalar_oracle\",\"rows\":%u,\"width\":%u,\"max_scaled\":%.9g,\"reduction_stages\":%u}\n",
        rows,width,maximum,partials>4096u?2u:1u);
}

static void split_attention_test(fg_vk_context *vk,fg_error *err){
    const uint32_t capacity=8193u;
    fg_vk_tensor *q=tensor(vk,6144u,err),*gate=tensor(vk,6144u,err);
    fg_vk_tensor *key=tensor(vk,512u,err),*value=tensor(vk,512u,err);
    fg_vk_tensor *keys=tensor(vk,capacity*512u,err),*values=tensor(vk,capacity*512u,err);
    fg_vk_tensor *partials=tensor(vk,24u*64u*258u,err),*position=tensor(vk,1u,err);
    fg_vk_tensor *reference=tensor(vk,6144u,err),*actual=tensor(vk,6144u,err);
    float data[6144],expected[6144],got[6144];
    fg_vk_tensor *inputs[]={q,gate,key,value};
    for(uint32_t t=0;t<4u;t++){
        uint32_t count=t<2u?6144u:512u;
        for(uint32_t i=0;i<count;i++)data[i]=signal(i,t+11u);
        check(fg_vk_tensor_write(inputs[t],0,data,count*4u,err),err);
    }
    float *cache=malloc((size_t)capacity*512u*4u);if(!cache)exit(1);
    for(uint32_t i=0;i<capacity*512u;i++)cache[i]=signal(i,23u);
    check(fg_vk_tensor_write(keys,0,cache,(uint64_t)capacity*512u*4u,err),err);
    for(uint32_t i=0;i<capacity*512u;i++)cache[i]=signal(i,29u);
    check(fg_vk_tensor_write(values,0,cache,(uint64_t)capacity*512u*4u,err),err);
    free(cache);double maximum=0;
    uint32_t cells[]={0u,15u,127u,8191u};
    for(uint32_t t=0;t<4u;t++){
        uint32_t cell=cells[t];check(fg_vk_tensor_write(position,0,&cell,4u,err),err);
        check(fg_vk_mtp_attention(vk,reference,keys,values,q,gate,key,value,cell,err),err);
        check(fg_vk_mtp_attention_split(vk,actual,partials,keys,values,q,gate,key,value,position,capacity,8u,err),err);
        check(fg_vk_tensor_read(reference,0,expected,sizeof(expected),err),err);
        check(fg_vk_tensor_read(actual,0,got,sizeof(got),err),err);
        for(uint32_t i=0;i<6144u;i++){
            double delta=fabs(got[i]-expected[i])/fmax(1.0,fabs(expected[i]));
            if(!isfinite(got[i])||delta>1e-5){fprintf(stderr,"split native attention mismatch cell=%u error=%g\n",cell,delta);exit(1);}
            if(delta>maximum)maximum=delta;
        }
    }
    fg_vk_tensor *all[]={q,gate,key,value,keys,values,partials,position,reference,actual};
    for(uint32_t i=0;i<10u;i++)fg_vk_tensor_destroy(all[i]);
    printf("{\"kind\":\"native_split_attention\",\"max_context\":8192,\"max_scaled\":%.9g}\n",maximum);
}
static void hc_norm_test(fg_vk_context *vk,fg_error *err){
    fg_vk_tensor *x=tensor(vk,10240u,err),*w=tensor(vk,10240u,err);
    fg_vk_tensor *reference=tensor(vk,10240u,err),*actual=tensor(vk,10240u,err);
    float values[10240],expected[10240],got[10240];
    for(uint32_t i=0;i<10240u;i++)values[i]=signal(i,17u);
    check(fg_vk_tensor_write(x,0,values,sizeof(values),err),err);
    for(uint32_t i=0;i<10240u;i++)values[i]=1.0f+(i%17u)*0.001f;
    check(fg_vk_tensor_write(w,0,values,sizeof(values),err),err);
    check(fg_vk_group_rms_norm(vk,reference,x,w,10240u,1u,1u,1e-6f,err),err);
    check(fg_vk_mtp_hc_norm(vk,actual,x,w,err),err);
    check(fg_vk_tensor_read(reference,0,expected,sizeof(expected),err),err);
    check(fg_vk_tensor_read(actual,0,got,sizeof(got),err),err);
    if(memcmp(expected,got,sizeof(got))){fprintf(stderr,"native HC normalization changed arithmetic\n");exit(1);}
    fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);
    fg_vk_tensor_destroy(reference);fg_vk_tensor_destroy(actual);
    printf("{\"kind\":\"native_hc_norm\",\"n1_bit_exact\":true}\n");
}
static void fc_stream_test(fg_vk_context *vk,fg_error *err){
    uint32_t row_bytes=2560u/32u*34u;
    uint8_t *raw=malloc((size_t)row_bytes*2560u);
    if(!raw)exit(1);
    for(uint32_t row=0;row<2560u;row++)for(uint32_t b=0;b<80u;b++){
        uint8_t *block=raw+(uint64_t)row*row_bytes+b*34u;
        uint16_t d=fg_f32_to_f16(0.001f+(row+b)%7u*0.0001f);memcpy(block,&d,2u);
        for(uint32_t j=0;j<32u;j++)block[2u+j]=(uint8_t)((int32_t)((row+b*3u+j*11u)%127u)-63);
    }
    uint64_t bytes=fg_q8_0_cooked_matrix_bytes(2560u,2560u);fg_vk_tensor *w=NULL;
    check(fg_vk_tensor_create(vk,bytes,&w,err),err);
    if(!fg_cook_q8_0_rows(raw,fg_vk_tensor_map(w),bytes,2560u,2560u))exit(1);
    free(raw);fg_vk_tensor_set_format(w,FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
    fg_vk_tensor *input=tensor(vk,10240u,err),*reference=tensor(vk,10240u,err);
    fg_vk_tensor *actual=tensor(vk,10240u,err),*x[4]={0},*y[4]={0};
    float data[10240],expected[10240],got[10240];
    for(uint32_t i=0;i<10240u;i++)data[i]=signal(i%2560u,11u+i/2560u*7u);
    check(fg_vk_tensor_write(input,0,data,sizeof(data),err),err);
    check(fg_vk_begin(vk,err),err);
    for(uint32_t stream=0;stream<4u;stream++){
        check(fg_vk_tensor_view(input,(uint64_t)stream*2560u*4u,2560u*4u,&x[stream],err),err);
        check(fg_vk_tensor_view(reference,(uint64_t)stream*2560u*4u,2560u*4u,&y[stream],err),err);
        check(fg_vk_dense_q8_0_f32(vk,y[stream],w,x[stream],2560u,2560u,1u,1.0f,err),err);
    }
    check(fg_vk_end(vk,err),err);
    check(fg_vk_mtp_fc_hidden(vk,actual,w,input,err),err);
    check(fg_vk_tensor_read(reference,0,expected,sizeof(expected),err),err);
    check(fg_vk_tensor_read(actual,0,got,sizeof(got),err),err);
    if(memcmp(expected,got,sizeof(expected))){fprintf(stderr,"interleaved native FC changed N=1 arithmetic\n");exit(1);}
    for(uint32_t i=0;i<4u;i++){fg_vk_tensor_destroy(x[i]);fg_vk_tensor_destroy(y[i]);}
    fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(reference);fg_vk_tensor_destroy(actual);fg_vk_tensor_destroy(w);
    printf("{\"kind\":\"native_fc_four_streams\",\"n1_bit_exact\":true}\n");
}

static fg_vk_tensor *head_weight(fg_vk_context *vk,const char *path,fg_error *err){
    if(!path)return NULL;
    uint64_t bytes=(uint64_t)FG_Q38_VOCAB_SIZE*FG_HIDDEN_SIZE/32u*34u;
    FILE *f=fopen(path,"rb");if(!f){perror(path);exit(1);}
    fg_vk_tensor *w=NULL;check(fg_vk_tensor_create(vk,bytes,&w,err),err);
    if(fread(fg_vk_tensor_map(w),1,(size_t)bytes,f)!=(size_t)bytes||fgetc(f)!=EOF){
        fprintf(stderr,"invalid shared-head Q8 span\n");exit(1);
    }
    fclose(f);return w;
}

int main(int argc,char **argv){
    const char *overlay=NULL,*head_path=NULL,*embedding_path=NULL,*dump_path=NULL;uint32_t iterations=32,warmup=16;
    bool self_test=false,replay=false,profile=false,head_gpu=false,compare_reference=false;
    uint32_t head_ways=1,head_shard=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--self-test"))self_test=true;
        else if(!strcmp(argv[i],"--replay"))replay=true;
        else if(!strcmp(argv[i],"--profile"))profile=true;
        else if(!strcmp(argv[i],"--head-gpu"))head_gpu=true;
        else if(!strcmp(argv[i],"--compare-reference"))compare_reference=true;
        else if(i+1<argc&&!strcmp(argv[i],"--head-ways")){
            if(!parse_count(argv[++i],&head_ways,4u))return 2;
        }else if(i+1<argc&&!strcmp(argv[i],"--head-shard")){
            if(!parse_count(argv[++i],&head_shard,3u))return 2;
        }
        else if(i+1<argc&&!strcmp(argv[i],"--overlay"))overlay=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--head-q8"))head_path=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--embedding-f32"))embedding_path=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--dump-first"))dump_path=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--iterations")){
            if(!parse_count(argv[++i],&iterations,1024u))return 2;
        }else if(i+1<argc&&!strcmp(argv[i],"--warmup")){
            if(!parse_count(argv[++i],&warmup,16384u))return 2;
        }
        else {fprintf(stderr,"invalid argument %s\n",argv[i]);return 2;}
    }
    if(!iterations||(!warmup&&dump_path)||(!self_test&&!overlay)||
       (compare_reference&&!replay)||(head_gpu&&!head_path)||(head_ways!=1u&&head_ways!=4u)||head_shard>=head_ways||
       ((!head_gpu||!head_path)&&(head_ways!=1u||head_shard))){
        fprintf(stderr,"usage: bench-mtp-head --self-test | --overlay FILE [--head-q8 FILE]\n");return 2;
    }
    fg_error err={0};fg_vk_context *vk=NULL;check(fg_vk_open(&vk,&err),&err);
    if(self_test){
        binding_test(vk,&err);attention_test(vk,&err);vocab_test(vk,&err);
        vocab_random_test(vk,65553u,64u,&err);vocab_random_test(vk,113u,2560u,&err);
        split_attention_test(vk,&err);hc_norm_test(vk,&err);fc_stream_test(vk,&err);
    }
    if(overlay){
        fg_manifest *manifest=malloc(sizeof(*manifest));
        if(!manifest){fprintf(stderr,"manifest OOM\n");return 1;}
        fg_manifest_init(manifest);
        uint32_t rank=manifest->layer_owner[FG_LAYER_COUNT-1u];
        fg_mtp_head *h=NULL;check(fg_mtp_head_open(&h,vk,manifest,rank,overlay,warmup+iterations,&err),&err);
        fg_mtp_head *reference_head=NULL;
        if(compare_reference)check(fg_mtp_head_open(&reference_head,vk,manifest,rank,overlay,warmup+iterations,&err),&err);
        double parity_max=0;
        fg_vk_tensor *hyper=tensor(vk,10240,&err),*embedding=tensor(vk,2560,&err);
        fg_vk_tensor *vocab=head_weight(vk,head_gpu&&!dump_path?NULL:head_path,&err);
        fg_vk_tensor *logits=vocab?tensor(vk,FG_Q38_VOCAB_SIZE,&err):NULL;
        fg_mtp_vocab *gpu_vocab=NULL;
        uint32_t head_rows=FG_Q38_VOCAB_SIZE/head_ways;
        if(head_gpu&&head_path){
            uint32_t *map=malloc(head_rows*4u);if(!map)return 1;
            for(uint32_t j=0;j<head_rows;j++)map[j]=head_shard*head_rows+j;
            check(fg_mtp_vocab_open(&gpu_vocab,vk,head_path,map,head_rows,&err),&err);
            free(map);
        }
        float hc[10240],emb[2560],hidden[2560],*scores=vocab?malloc(FG_Q38_VOCAB_SIZE*4u):NULL;
        if(vocab&&!scores){fprintf(stderr,"logit OOM\n");return 1;}
        float actual_embedding[2560];
        if(embedding_path){
            FILE *f=fopen(embedding_path,"rb");if(!f){perror(embedding_path);return 1;}
            if(fread(actual_embedding,1,sizeof(actual_embedding),f)!=sizeof(actual_embedding)||fgetc(f)!=EOF){
                fprintf(stderr,"invalid embedding span\n");return 1;
            }
            fclose(f);
        }
        double core=0,projection=0;uint32_t last_token=UINT32_MAX;double checksum=0;
        uint64_t core_submits=0,recorded_dispatches=0;
        FILE *trace_file=NULL;
        if(dump_path&&!replay){
            char trace_path[1200];
            if(snprintf(trace_path,sizeof(trace_path),"%s.trace",dump_path)>=(int)sizeof(trace_path))return 1;
            trace_file=fopen(trace_path,"wbx");if(!trace_file){perror(trace_path);return 1;}
            fg_mtp_head_set_trace(h,dump_stage,trace_file);
        }
        for(uint32_t cell=0;cell<warmup+iterations;cell++){
            for(uint32_t j=0;j<10240u;j++)hc[j]=signal(j,cell+5u);
            for(uint32_t j=0;j<2560u;j++)emb[j]=embedding_path?actual_embedding[j]:signal(j,cell+13u);
            check(fg_vk_tensor_write(hyper,0,hc,sizeof(hc),&err),&err);
            check(fg_vk_tensor_write(embedding,0,emb,sizeof(emb),&err),&err);
            fg_vk_counters before={0},after={0};fg_vk_get_counters(vk,&before);
            fg_vk_tensor *next=NULL,*contracted=NULL;double start=now_us();
            bool sample_profile=profile&&cell==warmup;
            if(sample_profile)check(fg_vk_profile_begin(vk,&err),&err);
            check((replay?fg_mtp_head_step_replay:fg_mtp_head_step)(
                h,hyper,embedding,cell,&next,&contracted,&err),&err);
            if(!cell&&trace_file){
                fg_mtp_head_set_trace(h,NULL,NULL);
                if(fclose(trace_file))return 1;
                trace_file=NULL;
            }
            double done=now_us();
            fg_vk_get_counters(vk,&after);
            if(sample_profile){
                fg_vk_profile p={0};check(fg_vk_profile_end(vk,&p,&err),&err);
                fprintf(stderr,"CORE_PROFILE replay=%d gpu_us=%.3f kernel_us=%.3f wait_us=%.3f record_us=%.3f submits=%llu dispatches=%llu total_submits=%llu\n",
                    replay,p.gpu_ms*1000,p.kernel_ms*1000,p.wait_ms*1000,p.record_ms*1000,
                    (unsigned long long)p.submissions,(unsigned long long)p.dispatches,
                    (unsigned long long)(after.submissions-before.submissions));
                for(uint32_t j=0;j<p.kernel_count;j++)
                    fprintf(stderr,"KERNEL %s calls=%llu gpu_us=%.3f\n",p.kernels[j].name,
                        (unsigned long long)p.kernels[j].invocations,p.kernels[j].gpu_ms*1000);
            }
            if(gpu_vocab){
                float score=0;check(fg_mtp_vocab_argmax(gpu_vocab,contracted,&last_token,&score,&err),&err);
            }else if(vocab){
                check(fg_vk_dense_q8_0_f32(vk,logits,vocab,contracted,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,1u,1.0f,&err),&err);
                check(fg_vk_tensor_read(logits,0,scores,FG_Q38_VOCAB_SIZE*4u,&err),&err);
                float best=-INFINITY;last_token=UINT32_MAX;
                for(uint32_t j=0;j<FG_Q38_VOCAB_SIZE;j++){
                    if(!isfinite(scores[j])){fprintf(stderr,"nonfinite MTP logit\n");return 1;}
                    if(scores[j]>best){best=scores[j];last_token=j;}
                }
            }
            double stop=now_us();
            if(reference_head){
                fg_vk_tensor *ref_hc=NULL,*ref_hidden=NULL;
                check(fg_mtp_head_step(reference_head,hyper,embedding,cell,&ref_hc,&ref_hidden,&err),&err);
                const fg_vk_tensor *actual[]={next,contracted},*expected[]={ref_hc,ref_hidden};
                uint32_t counts[]={10240u,2560u};
                float a[10240],b[10240];
                for(uint32_t j=0;j<2u;j++){
                    check(fg_vk_tensor_read(actual[j],0,a,(uint64_t)counts[j]*4u,&err),&err);
                    check(fg_vk_tensor_read(expected[j],0,b,(uint64_t)counts[j]*4u,&err),&err);
                    for(uint32_t i=0;i<counts[j];i++){
                        double delta=fabs((double)a[i]-b[i])/fmax(1.0,fabs(b[i]));
                        if(!isfinite(a[i])||!isfinite(b[i])||delta>1e-4){
                            fprintf(stderr,"native replay parity cell=%u tensor=%u index=%u error=%g\n",cell,j,i,delta);return 1;
                        }
                        if(delta>parity_max)parity_max=delta;
                    }
                }
            }
            check(fg_vk_tensor_read(contracted,0,hidden,sizeof(hidden),&err),&err);
            if(!cell&&dump_path){
                if(gpu_vocab&&vocab){
                    check(fg_vk_dense_q8_0_f32(vk,logits,vocab,contracted,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,1u,1.0f,&err),&err);
                    check(fg_vk_tensor_read(logits,0,scores,FG_Q38_VOCAB_SIZE*4u,&err),&err);
                    uint32_t expected=head_shard*head_rows;
                    for(uint32_t j=expected;j<(head_shard+1u)*head_rows;j++)
                        if(scores[j]>scores[expected])expected=j;
                    if(expected!=last_token){fprintf(stderr,"GPU shard argmax mismatch expected=%u actual=%u\n",expected,last_token);return 1;}
                }
                float final_hc[10240];check(fg_vk_tensor_read(next,0,final_hc,sizeof(final_hc),&err),&err);
                FILE *f=fopen(dump_path,"wbx");
                bool ok=f&&fwrite(hc,1,sizeof(hc),f)==sizeof(hc)&&
                    fwrite(emb,1,sizeof(emb),f)==sizeof(emb)&&
                    fwrite(final_hc,1,sizeof(final_hc),f)==sizeof(final_hc)&&
                    fwrite(hidden,1,sizeof(hidden),f)==sizeof(hidden);
                if(vocab&&ok)ok=fwrite(scores,4,FG_Q38_VOCAB_SIZE,f)==FG_Q38_VOCAB_SIZE;
                if(f&&fclose(f))ok=false;
                if(!ok){fprintf(stderr,"write native oracle dump\n");return 1;}
            }
            for(uint32_t j=0;j<2560u;j++){
                if(!isfinite(hidden[j])){fprintf(stderr,"nonfinite MTP hidden\n");return 1;}
                checksum+=hidden[j];
            }
            if(cell>=warmup){
                core+=done-start;projection+=stop-done;
                core_submits+=after.submissions-before.submissions;
                recorded_dispatches+=after.dispatches-before.dispatches;
            }
        }
        /* The last cell must rebuild identically after truncation, including
         * its current KV row. This is draft-cache repair, not target rollback. */
        uint32_t last=warmup+iterations-1u;
        fg_vk_tensor *next=NULL,*contracted=NULL;
        if(fg_mtp_head_truncate(h,last+2u,&err)!=FG_ERR_ARGUMENT||
           fg_mtp_head_frontier(h)!=last+1u||
           (replay?fg_mtp_head_step_replay:fg_mtp_head_step)(
               h,hyper,embedding,last+1u,&next,&contracted,&err)!=FG_ERR_ARGUMENT||
           fg_mtp_head_frontier(h)!=last+1u){
            fprintf(stderr,"native draft frontier/capacity error changed state\n");return 1;
        }
        check(fg_mtp_head_truncate(h,last,&err),&err);
        check((replay?fg_mtp_head_step_replay:fg_mtp_head_step)(
            h,hyper,embedding,last,&next,&contracted,&err),&err);
        float rebuilt[2560];check(fg_vk_tensor_read(contracted,0,rebuilt,sizeof(rebuilt),&err),&err);
        if(memcmp(hidden,rebuilt,sizeof(hidden))){fprintf(stderr,"native draft KV repair mismatch\n");return 1;}
        printf("{\"kind\":\"native_mtp_cost\",\"synthetic_hc\":true,\"acceptance_measured\":false,"
               "\"replay\":%s,\"owner_rank\":%u,\"weight_bytes\":%llu,\"warmup\":%u,\"iterations\":%u,"
               "\"mean_core_us\":%.6f,\"mean_vocab_projection_us\":%.6f,\"vocab_rows\":%u,\"vocab_gpu\":%s,"
               "\"shared_head_present\":%s,\"real_embedding\":%s,\"draft_token\":%u,\"repair_bit_exact\":true,"
               "\"reference_compared\":%s,\"reference_max_scaled\":%.9g,"
               "\"mean_core_submits\":%.3f,\"mean_recorded_dispatches\":%.3f,\"checksum\":%.9g}\n",
               replay?"true":"false",rank,(unsigned long long)fg_mtp_head_weight_bytes(h),warmup,iterations,
               core/iterations,projection/iterations,head_rows,gpu_vocab?"true":"false",
               (vocab||gpu_vocab)?"true":"false",embedding_path?"true":"false",last_token,
               compare_reference?"true":"false",parity_max,(double)core_submits/iterations,
               (double)recorded_dispatches/iterations,checksum);
        free(scores);fg_mtp_vocab_close(gpu_vocab);fg_vk_tensor_destroy(logits);fg_vk_tensor_destroy(vocab);
        fg_vk_tensor_destroy(embedding);fg_vk_tensor_destroy(hyper);
        fg_mtp_head_close(reference_head);fg_mtp_head_close(h);free(manifest);
        /* A borrowed model must not close the caller's Vulkan context. */
        fg_vk_tensor *alive=tensor(vk,4,&err);fg_vk_tensor_destroy(alive);
    }
    fg_vk_close(vk);return 0;
}
