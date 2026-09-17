#include "fg_tower.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct resample_plan {
    uint32_t kernel;
    uint32_t *first;
    float *weights;
} resample_plan;

static float gelu_tanh(float x){
    const float c=0.7978845608028654f;
    return 0.5f*x*(1.0f+tanhf(c*(x+0.044715f*x*x*x)));
}

static bool tower_geometry_valid(const fg_tower_geometry *g){
    return g&&g->image_width&&g->image_height&&
        g->image_width%FG_TOWER_PATCH_SIZE==0&&g->image_height%FG_TOWER_PATCH_SIZE==0&&
        g->grid_width==g->image_width/FG_TOWER_PATCH_SIZE&&
        g->grid_height==g->image_height/FG_TOWER_PATCH_SIZE&&
        g->grid_width&&g->grid_height&&
        g->grid_width%FG_TOWER_MERGE==0&&g->grid_height%FG_TOWER_MERGE==0&&
        g->tokens==g->grid_width*g->grid_height&&
        g->merged_tokens==g->tokens/FG_TOWER_MERGE_FACTOR;
}

fg_status fg_tower_image_geometry(uint32_t width,uint32_t height,
                                  fg_tower_geometry *geometry,fg_error *err){
    if(!geometry||!width||!height){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower image geometry request");
        return FG_ERR_ARGUMENT;
    }
    if(width%FG_TOWER_ALIGN||height%FG_TOWER_ALIGN){
        fg_error_set(err,FG_ERR_FORMAT,
                     "tower image %ux%u is not aligned to %u pixels",width,height,
                     FG_TOWER_ALIGN);
        return FG_ERR_FORMAT;
    }
    geometry->image_width=width;
    geometry->image_height=height;
    geometry->grid_width=width/FG_TOWER_PATCH_SIZE;
    geometry->grid_height=height/FG_TOWER_PATCH_SIZE;
    geometry->tokens=geometry->grid_width*geometry->grid_height;
    geometry->merged_tokens=geometry->tokens/FG_TOWER_MERGE_FACTOR;
    if(!tower_geometry_valid(geometry)){
        fg_error_set(err,FG_ERR_FORMAT,"tower image geometry is invalid");
        return FG_ERR_FORMAT;
    }
    return FG_OK;
}

fg_status fg_tower_smart_resize(uint32_t width,uint32_t height,uint32_t align,
                                uint64_t min_pixels,uint64_t max_pixels,
                                uint32_t *out_width,uint32_t *out_height,fg_error *err){
    if(!width||!height||!align||!out_width||!out_height){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower smart_resize request");
        return FG_ERR_ARGUMENT;
    }
    const double w=(double)width,h=(double)height,f=(double)align;
    double w_bar=round(w/f)*f,h_bar=round(h/f)*f;
    if(w_bar<f)w_bar=f;
    if(h_bar<f)h_bar=f;
    if(max_pixels&&h_bar*w_bar>(double)max_pixels){
        const double beta=sqrt((w*h)/(double)max_pixels);
        h_bar=floor(h/beta/f)*f;
        w_bar=floor(w/beta/f)*f;
        if(h_bar<f)h_bar=f;
        if(w_bar<f)w_bar=f;
    }else if(min_pixels&&h_bar*w_bar<(double)min_pixels){
        const double beta=sqrt((double)min_pixels/(w*h));
        h_bar=ceil(h*beta/f)*f;
        w_bar=ceil(w*beta/f)*f;
    }
    if(w_bar>(double)UINT32_MAX||h_bar>(double)UINT32_MAX){
        fg_error_set(err,FG_ERR_LIMIT,"tower smart_resize overflows");
        return FG_ERR_LIMIT;
    }
    *out_width=(uint32_t)w_bar;
    *out_height=(uint32_t)h_bar;
    return FG_OK;
}

fg_status fg_tower_normalize_image(const uint8_t *rgb,uint32_t width,uint32_t height,
                                   float *out,fg_error *err){
    if(!rgb||!out||!width||!height){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower normalize request");
        return FG_ERR_ARGUMENT;
    }
    const uint64_t count=(uint64_t)width*height*3u;
    for(uint64_t i=0;i<count;i++)out[i]=(float)rgb[i]/127.5f-1.0f;
    return FG_OK;
}

static double resample_filter(double x){
    const double a=-0.5;
    if(x<0.0)x=-x;
    if(x<1.0)return ((a+2.0)*x-(a+3.0))*x*x+1.0;
    if(x<2.0)return (((x-5.0)*x+8.0)*x-4.0)*a;
    return 0.0;
}

static fg_status resample_plan_build(uint32_t in_size,uint32_t out_size,double support,
                                     resample_plan *plan,fg_error *err){
    double filterscale=(double)in_size/(double)out_size;
    if(filterscale<1.0)filterscale=1.0;
    const double radius=support*filterscale;
    const uint32_t kernel=(uint32_t)ceil(radius)*2u+1u;
    plan->kernel=kernel;
    plan->first=calloc(out_size,sizeof(*plan->first));
    plan->weights=calloc((size_t)out_size*kernel,sizeof(*plan->weights));
    if(!plan->first||!plan->weights){
        free(plan->first);
        free(plan->weights);
        fg_error_set(err,FG_ERR_OOM,"allocate tower resample plan");
        return FG_ERR_OOM;
    }
    const double scale=(double)in_size/(double)out_size;
    for(uint32_t xx=0;xx<out_size;xx++){
        const double center=((double)xx+0.5)*scale;
        const double step=1.0/filterscale;
        int first=(int)(center-radius+0.5);
        if(first<0)first=0;
        int last=(int)(center+radius+0.5);
        if(last>(int)in_size)last=(int)in_size;
        plan->first[xx]=(uint32_t)first;
        double total=0.0;
        for(int x=first;x<last;x++){
            const double w=resample_filter(((double)x-center+0.5)*step);
            plan->weights[(size_t)xx*kernel+(uint32_t)(x-first)]=(float)w;
            total+=w;
        }
        if(total!=0.0)
            for(int x=first;x<last;x++)
                plan->weights[(size_t)xx*kernel+(uint32_t)(x-first)]/=(float)total;
    }
    return FG_OK;
}

static void resample_plan_free(resample_plan *plan){
    free(plan->first);
    free(plan->weights);
    plan->first=NULL;
    plan->weights=NULL;
}

static void resample_line(const float *src,const resample_plan *plan,
                          uint32_t out_size,float *dst){
    for(uint32_t xx=0;xx<out_size;xx++){
        const float *row=src+plan->first[xx];
        const float *w=plan->weights+(size_t)xx*plan->kernel;
        float acc=0.0f;
        for(uint32_t k=0;k<plan->kernel;k++)acc+=row[k]*w[k];
        dst[xx]=acc;
    }
}

fg_status fg_tower_resize_bicubic(const float *source,uint32_t source_width,
                                  uint32_t source_height,float *destination,
                                  uint32_t width,uint32_t height,fg_error *err){
    if(!source||!destination||!source_width||!source_height||!width||!height){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower resize request");
        return FG_ERR_ARGUMENT;
    }
    if(source_width==width&&source_height==height){
        memcpy(destination,source,(size_t)source_width*source_height*3u*sizeof(float));
        return FG_OK;
    }
    resample_plan horizontal={0},vertical={0};
    fg_status status=resample_plan_build(source_width,width,2.0,&horizontal,err);
    if(status!=FG_OK)return status;
    status=resample_plan_build(source_height,height,2.0,&vertical,err);
    if(status!=FG_OK){
        resample_plan_free(&horizontal);
        return status;
    }
    float *mid=malloc((size_t)width*source_height*3u*sizeof(float));
    if(!mid){
        resample_plan_free(&horizontal);
        resample_plan_free(&vertical);
        fg_error_set(err,FG_ERR_OOM,"allocate tower resize buffer");
        return FG_ERR_OOM;
    }
    for(uint32_t c=0;c<3u;c++)
        for(uint32_t y=0;y<source_height;y++)
            resample_line(source+((size_t)c*source_height+y)*source_width,
                          &horizontal,width,mid+((size_t)c*source_height+y)*width);
    float *column=malloc((size_t)(source_height>height?source_height:height)*sizeof(float));
    float *sampled=malloc((size_t)height*sizeof(float));
    if(!column||!sampled){
        free(sampled);
        free(column);
        free(mid);
        resample_plan_free(&horizontal);
        resample_plan_free(&vertical);
        fg_error_set(err,FG_ERR_OOM,"allocate tower resize column");
        return FG_ERR_OOM;
    }
    for(uint32_t c=0;c<3u;c++)
        for(uint32_t x=0;x<width;x++){
            for(uint32_t y=0;y<source_height;y++)
                column[y]=mid[((size_t)c*source_height+y)*width+x];
            resample_line(column,&vertical,height,sampled);
            for(uint32_t y=0;y<height;y++)
                destination[((size_t)c*height+y)*width+x]=sampled[y];
        }
    free(sampled);
    free(column);
    free(mid);
    resample_plan_free(&horizontal);
    resample_plan_free(&vertical);
    return FG_OK;
}

fg_status fg_tower_patchify(const float *image,uint32_t width,uint32_t height,
                            float *tokens,const fg_tower_geometry *geometry,fg_error *err){
    if(!image||!tokens||!tower_geometry_valid(geometry)||
       geometry->image_width!=width||geometry->image_height!=height){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower patchify request");
        return FG_ERR_ARGUMENT;
    }
    const uint32_t half_width=geometry->grid_width/FG_TOWER_MERGE;
    for(uint32_t by=0;by<geometry->grid_height/FG_TOWER_MERGE;by++){
        for(uint32_t bx=0;bx<half_width;bx++){
            for(uint32_t dy=0;dy<FG_TOWER_MERGE;dy++){
                for(uint32_t dx=0;dx<FG_TOWER_MERGE;dx++){
                    const uint32_t token=((by*half_width+bx)*FG_TOWER_MERGE_FACTOR)+
                        dy*FG_TOWER_MERGE+dx;
                    float *dst=tokens+(size_t)token*FG_TOWER_TOKEN_VALUES;
                    for(uint32_t c=0;c<3u;c++){
                        for(uint32_t py=0;py<FG_TOWER_PATCH_SIZE;py++){
                            const uint32_t y=(by*FG_TOWER_MERGE+dy)*FG_TOWER_PATCH_SIZE+py;
                            const float *src=image+((size_t)c*height+y)*width+
                                (bx*FG_TOWER_MERGE+dx)*FG_TOWER_PATCH_SIZE;
                            float *patch=dst+c*FG_TOWER_PATCH_SIZE*FG_TOWER_PATCH_SIZE+
                                py*FG_TOWER_PATCH_SIZE;
                            memcpy(patch,src,FG_TOWER_PATCH_SIZE*sizeof(float));
                        }
                    }
                    memcpy(dst+FG_TOWER_PATCH_VALUES,dst,
                           FG_TOWER_PATCH_VALUES*sizeof(float));
                }
            }
        }
    }
    return FG_OK;
}

static void tower_token_coords(const fg_tower_geometry *g,uint32_t token,
                               uint32_t *x,uint32_t *y){
    const uint32_t half=g->grid_width/FG_TOWER_MERGE;
    const uint32_t group=token/FG_TOWER_MERGE_FACTOR;
    const uint32_t inner=token%FG_TOWER_MERGE_FACTOR;
    const uint32_t dx=inner%FG_TOWER_MERGE,dy=inner/FG_TOWER_MERGE;
    *x=(group%half)*FG_TOWER_MERGE+dx;
    *y=(group/half)*FG_TOWER_MERGE+dy;
}

fg_status fg_tower_vision_positions(const fg_tower_geometry *geometry,int32_t *positions,
                                    fg_error *err){
    if(!positions||!tower_geometry_valid(geometry)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower position request");
        return FG_ERR_ARGUMENT;
    }
    for(uint32_t token=0;token<geometry->tokens;token++){
        uint32_t x=0,y=0;
        tower_token_coords(geometry,token,&x,&y);
        positions[token]=(int32_t)y;
        positions[geometry->tokens+token]=(int32_t)x;
        positions[2u*geometry->tokens+token]=(int32_t)y;
        positions[3u*geometry->tokens+token]=(int32_t)x;
    }
    return FG_OK;
}

fg_status fg_tower_position_embeddings(const float *position_weight,
                                       const fg_tower_geometry *geometry,float *tokens,
                                       fg_error *err){
    if(!position_weight||!tokens||!tower_geometry_valid(geometry)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower position embedding request");
        return FG_ERR_ARGUMENT;
    }
    const uint32_t grid=FG_TOWER_POS_GRID;
    for(uint32_t token=0;token<geometry->tokens;token++){
        uint32_t x=0,y=0;
        tower_token_coords(geometry,token,&x,&y);
        const float fx=geometry->grid_width>1?
            (float)x*(float)(grid-1u)/(float)(geometry->grid_width-1u):0.0f;
        const float fy=geometry->grid_height>1?
            (float)y*(float)(grid-1u)/(float)(geometry->grid_height-1u):0.0f;
        const uint32_t x0=(uint32_t)floorf(fx);
        const uint32_t y0=(uint32_t)floorf(fy);
        const uint32_t x1=x0+1u<grid?x0+1u:grid-1u;
        const uint32_t y1=y0+1u<grid?y0+1u:grid-1u;
        const float dx=fx-(float)x0,dy=fy-(float)y0;
        const float *p00=position_weight+((size_t)y0*grid+x0)*FG_TOWER_HIDDEN;
        const float *p01=position_weight+((size_t)y0*grid+x1)*FG_TOWER_HIDDEN;
        const float *p10=position_weight+((size_t)y1*grid+x0)*FG_TOWER_HIDDEN;
        const float *p11=position_weight+((size_t)y1*grid+x1)*FG_TOWER_HIDDEN;
        float *dst=tokens+(size_t)token*FG_TOWER_HIDDEN;
        for(uint32_t c=0;c<FG_TOWER_HIDDEN;c++){
            const float top=p00[c]*(1.0f-dx)+p01[c]*dx;
            const float bottom=p10[c]*(1.0f-dx)+p11[c]*dx;
            dst[c]=top*(1.0f-dy)+bottom*dy;
        }
    }
    return FG_OK;
}

static void matmul_f32(const float *weights,const float *input,uint32_t rows,
                       uint32_t outputs,uint32_t width,float *out){
    for(uint32_t n=0;n<rows;n++){
        const float *x=input+(size_t)n*width;
        float *y=out+(size_t)n*outputs;
        for(uint32_t m=0;m<outputs;m++){
            const float *w=weights+(size_t)m*width;
            float acc=0.0f;
            for(uint32_t k=0;k<width;k++)acc+=w[k]*x[k];
            y[m]=acc;
        }
    }
}

static void bias_add_f32(float *values,const float *bias,uint32_t rows,uint32_t width){
    for(uint32_t n=0;n<rows;n++){
        float *y=values+(size_t)n*width;
        for(uint32_t m=0;m<width;m++)y[m]+=bias[m];
    }
}

static void gelu_f32(float *values,uint64_t count){
    for(uint64_t i=0;i<count;i++)values[i]=gelu_tanh(values[i]);
}

static void layernorm_f32(const float *input,const float *weight,const float *bias,
                          uint32_t rows,uint32_t width,float epsilon,float *out){
    for(uint32_t n=0;n<rows;n++){
        const float *x=input+(size_t)n*width;
        float *y=out+(size_t)n*width;
        double mean=0.0;
        for(uint32_t i=0;i<width;i++)mean+=x[i];
        mean/=(double)width;
        double variance=0.0;
        for(uint32_t i=0;i<width;i++){
            const double d=(double)x[i]-mean;
            variance+=d*d;
        }
        variance/=(double)width;
        const float scale=(float)(1.0/sqrt(variance+(double)epsilon));
        for(uint32_t i=0;i<width;i++)
            y[i]=((float)((double)x[i]-mean)*scale)*weight[i]+bias[i];
    }
}

static void add_inplace_f32(float *target,const float *source,uint64_t count){
    for(uint64_t i=0;i<count;i++)target[i]+=source[i];
}

static void rope_vision_f32(float *qkv,uint32_t tokens,uint32_t section,
                            uint32_t grid_width){
    const uint32_t pairs=FG_TOWER_ROPE_PAIRS;
    const uint32_t section_pairs=FG_TOWER_ROPE_SECTION;
    for(uint32_t token=0;token<tokens;token++){
        const uint32_t half=grid_width/FG_TOWER_MERGE;
        const uint32_t group=token/FG_TOWER_MERGE_FACTOR;
        const uint32_t inner=token%FG_TOWER_MERGE_FACTOR;
        const uint32_t dx=inner%FG_TOWER_MERGE,dy=inner/FG_TOWER_MERGE;
        const float px=(float)((group%half)*FG_TOWER_MERGE+dx);
        const float py=(float)((group/half)*FG_TOWER_MERGE+dy);
        float *base=qkv+(size_t)token*FG_TOWER_QKV_WIDTH+section;
        for(uint32_t head=0;head<FG_TOWER_HEADS;head++){
            float *q=base+(size_t)head*FG_TOWER_HEAD_DIM;
            for(uint32_t pair=0;pair<pairs;pair++){
                const float position=pair<section_pairs?py:px;
                const uint32_t frequency=pair%section_pairs;
                const float theta=position*powf(FG_TOWER_ROPE_BASE,
                    -2.0f*(float)frequency/(float)FG_TOWER_ROPE_PAIRS);
                const float cosine=cosf(theta),sine=sinf(theta);
                const float x0=q[pair],x1=q[pair+pairs];
                q[pair]=x0*cosine-x1*sine;
                q[pair+pairs]=x0*sine+x1*cosine;
            }
        }
    }
}

static void attention_f32(const float *qkv,uint32_t tokens,float *out,
                          float *scores){
    const float scale=1.0f/sqrtf((float)FG_TOWER_HEAD_DIM);
    for(uint32_t token=0;token<tokens;token++){
        const float *qbase=qkv+(size_t)token*FG_TOWER_QKV_WIDTH;
        const float *kbase=qkv+FG_TOWER_HIDDEN;
        for(uint32_t head=0;head<FG_TOWER_HEADS;head++){
            const float *q=qbase+(size_t)head*FG_TOWER_HEAD_DIM;
            for(uint32_t j=0;j<tokens;j++){
                const float *k=kbase+(size_t)j*FG_TOWER_QKV_WIDTH+
                    (size_t)head*FG_TOWER_HEAD_DIM;
                float acc=0.0f;
                for(uint32_t d=0;d<FG_TOWER_HEAD_DIM;d++)acc+=q[d]*k[d];
                scores[j]=acc*scale;
            }
            float maximum=scores[0];
            for(uint32_t j=1;j<tokens;j++)if(scores[j]>maximum)maximum=scores[j];
            float total=0.0f;
            for(uint32_t j=0;j<tokens;j++){
                scores[j]=expf(scores[j]-maximum);
                total+=scores[j];
            }
            const float inverse=1.0f/total;
            float *y=out+(size_t)token*FG_TOWER_HIDDEN+(size_t)head*FG_TOWER_HEAD_DIM;
            for(uint32_t d=0;d<FG_TOWER_HEAD_DIM;d++)y[d]=0.0f;
            for(uint32_t j=0;j<tokens;j++){
                const float probability=scores[j]*inverse;
                const float *v=qkv+(size_t)j*FG_TOWER_QKV_WIDTH+
                    (size_t)2u*FG_TOWER_HIDDEN+(size_t)head*FG_TOWER_HEAD_DIM;
                for(uint32_t d=0;d<FG_TOWER_HEAD_DIM;d++)y[d]+=probability*v[d];
            }
        }
    }
}

fg_status fg_tower_embed_cpu(const fg_tower_weights *weights,const float *tokens,
                             const fg_tower_geometry *geometry,float *hidden,fg_error *err){
    if(!weights||!tokens||!hidden||!tower_geometry_valid(geometry)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower embedding request");
        return FG_ERR_ARGUMENT;
    }
    const uint32_t count=geometry->tokens;
    matmul_f32(weights->patch_weight,tokens,count,FG_TOWER_HIDDEN,
               FG_TOWER_TOKEN_VALUES,hidden);
    bias_add_f32(hidden,weights->patch_bias,count,FG_TOWER_HIDDEN);
    float *position=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    if(!position){
        fg_error_set(err,FG_ERR_OOM,"allocate tower position buffer");
        return FG_ERR_OOM;
    }
    fg_status status=fg_tower_position_embeddings(weights->position_weight,geometry,position,err);
    if(status==FG_OK)add_inplace_f32(hidden,position,(uint64_t)count*FG_TOWER_HIDDEN);
    free(position);
    return status;
}

fg_status fg_tower_block_cpu(const fg_tower_block_weights *weights,const float *input,
                             uint32_t tokens,uint32_t grid_width,float *output,fg_error *err){
    if(!weights||!input||!output||!tokens||!grid_width||grid_width%FG_TOWER_MERGE){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower block request");
        return FG_ERR_ARGUMENT;
    }
    float *x=malloc((size_t)tokens*FG_TOWER_HIDDEN*sizeof(float));
    float *t=malloc((size_t)tokens*FG_TOWER_HIDDEN*sizeof(float));
    float *qkv=malloc((size_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float));
    float *up=malloc((size_t)tokens*FG_TOWER_MLP*sizeof(float));
    float *attn=malloc((size_t)tokens*FG_TOWER_HIDDEN*sizeof(float));
    float *block=malloc((size_t)tokens*FG_TOWER_HIDDEN*sizeof(float));
    float *scores=malloc((size_t)tokens*sizeof(float));
    if(!x||!t||!qkv||!up||!attn||!block||!scores){
        free(scores);free(block);free(attn);free(up);free(qkv);free(t);free(x);
        fg_error_set(err,FG_ERR_OOM,"allocate tower block buffers");
        return FG_ERR_OOM;
    }
    memcpy(x,input,(size_t)tokens*FG_TOWER_HIDDEN*sizeof(float));
    layernorm_f32(x,weights->ln1_weight,weights->ln1_bias,tokens,FG_TOWER_HIDDEN,
                  FG_TOWER_LN_EPS,t);
    matmul_f32(weights->qkv_weight,t,tokens,FG_TOWER_QKV_WIDTH,FG_TOWER_HIDDEN,qkv);
    bias_add_f32(qkv,weights->qkv_bias,tokens,FG_TOWER_QKV_WIDTH);
    rope_vision_f32(qkv,tokens,0u,grid_width);
    rope_vision_f32(qkv,tokens,FG_TOWER_HIDDEN,grid_width);
    attention_f32(qkv,tokens,attn,scores);
    matmul_f32(weights->attn_out_weight,attn,tokens,FG_TOWER_HIDDEN,FG_TOWER_HIDDEN,block);
    bias_add_f32(block,weights->attn_out_bias,tokens,FG_TOWER_HIDDEN);
    add_inplace_f32(x,block,(uint64_t)tokens*FG_TOWER_HIDDEN);
    layernorm_f32(x,weights->ln2_weight,weights->ln2_bias,tokens,FG_TOWER_HIDDEN,
                  FG_TOWER_LN_EPS,t);
    matmul_f32(weights->ffn_up_weight,t,tokens,FG_TOWER_MLP,FG_TOWER_HIDDEN,up);
    bias_add_f32(up,weights->ffn_up_bias,tokens,FG_TOWER_MLP);
    gelu_f32(up,(uint64_t)tokens*FG_TOWER_MLP);
    matmul_f32(weights->ffn_down_weight,up,tokens,FG_TOWER_HIDDEN,FG_TOWER_MLP,block);
    bias_add_f32(block,weights->ffn_down_bias,tokens,FG_TOWER_HIDDEN);
    add_inplace_f32(x,block,(uint64_t)tokens*FG_TOWER_HIDDEN);
    memcpy(output,x,(size_t)tokens*FG_TOWER_HIDDEN*sizeof(float));
    free(scores);free(block);free(attn);free(up);free(qkv);free(t);free(x);
    return FG_OK;
}

fg_status fg_tower_merger_cpu(const fg_tower_weights *weights,const float *input,
                              uint32_t merged_tokens,float *embeddings,fg_error *err){
    if(!weights||!input||!embeddings||!merged_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower merger request");
        return FG_ERR_ARGUMENT;
    }
    float *block=malloc((size_t)merged_tokens*FG_TOWER_MERGED_WIDTH*sizeof(float));
    if(!block){
        fg_error_set(err,FG_ERR_OOM,"allocate tower merger buffer");
        return FG_ERR_OOM;
    }
    matmul_f32(weights->merger_fc1_weight,input,merged_tokens,FG_TOWER_MERGED_WIDTH,
               FG_TOWER_MERGED_WIDTH,block);
    bias_add_f32(block,weights->merger_fc1_bias,merged_tokens,FG_TOWER_MERGED_WIDTH);
    gelu_f32(block,(uint64_t)merged_tokens*FG_TOWER_MERGED_WIDTH);
    matmul_f32(weights->merger_fc2_weight,block,merged_tokens,FG_TOWER_OUT_HIDDEN,
               FG_TOWER_MERGED_WIDTH,embeddings);
    bias_add_f32(embeddings,weights->merger_fc2_bias,merged_tokens,FG_TOWER_OUT_HIDDEN);
    free(block);
    return FG_OK;
}

fg_status fg_tower_forward_cpu(const fg_tower_weights *weights,const float *tokens,
                               const fg_tower_geometry *geometry,float *embeddings,
                               fg_error *err){
    if(!weights||!tokens||!embeddings||!tower_geometry_valid(geometry)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower CPU forward request");
        return FG_ERR_ARGUMENT;
    }
    const uint32_t count=geometry->tokens;
    float *x=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    float *t=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    if(!x||!t){
        free(t);free(x);
        fg_error_set(err,FG_ERR_OOM,"allocate tower CPU forward buffers");
        return FG_ERR_OOM;
    }
    fg_status status=fg_tower_embed_cpu(weights,tokens,geometry,x,err);
    for(uint32_t layer=0;layer<FG_TOWER_LAYERS&&status==FG_OK;layer++){
        status=fg_tower_block_cpu(&weights->blocks[layer],x,count,geometry->grid_width,t,err);
        if(status==FG_OK)memcpy(x,t,(size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    }
    if(status==FG_OK){
        layernorm_f32(x,weights->post_ln_weight,weights->post_ln_bias,count,
                      FG_TOWER_HIDDEN,FG_TOWER_LN_EPS,t);
        status=fg_tower_merger_cpu(weights,t,geometry->merged_tokens,embeddings,err);
    }
    free(t);free(x);
    return status;
}
