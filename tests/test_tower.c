#include "fg_manifest.h"
#include "fg_quant.h"
#include "fg_tower.h"
#include "fg_tower_vk.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

static uint32_t rng_state=0x12345678u;

static float rng_next(void){
    rng_state=rng_state*1664525u+1013904223u;
    return (float)((rng_state>>8u)&0xFFFFFFu)/8388608.0f-1.0f;
}

static void fill_random(float *values,size_t count,float scale){
    for(size_t i=0;i<count;i++)values[i]=rng_next()*scale;
}

static double cosine_similarity(const float *left,const float *right,size_t count){
    double dot=0.0,left_norm=0.0,right_norm=0.0;
    for(size_t i=0;i<count;i++){
        dot+=(double)left[i]*right[i];
        left_norm+=(double)left[i]*left[i];
        right_norm+=(double)right[i]*right[i];
    }
    if(left_norm==0.0||right_norm==0.0)return 0.0;
    return dot/sqrt(left_norm*right_norm);
}

static double vector_norm(const float *values,size_t count){
    double norm=0.0;
    for(size_t i=0;i<count;i++)norm+=(double)values[i]*values[i];
    return sqrt(norm);
}

static double relative_error(const float *left,const float *right,size_t count){
    double error=0.0,norm=0.0;
    for(size_t i=0;i<count;i++){
        const double d=(double)left[i]-right[i];
        error+=d*d;
        norm+=(double)right[i]*right[i];
    }
    return norm>0.0?sqrt(error/norm):sqrt(error);
}

static bool all_finite(const float *values,size_t count){
    for(size_t i=0;i<count;i++)if(!isfinite(values[i]))return false;
    return true;
}

static void test_smart_resize(void){
    fg_error err={0};
    uint32_t width=0,height=0;
    CHECK(fg_tower_smart_resize(512u,512u,FG_TOWER_ALIGN,FG_TOWER_IMAGE_MIN_PIXELS,
                                FG_TOWER_IMAGE_MAX_PIXELS,&width,&height,&err)==FG_OK);
    CHECK(width==512u&&height==512u);
    CHECK(fg_tower_smart_resize(1000u,800u,FG_TOWER_ALIGN,0u,FG_TOWER_IMAGE_MAX_PIXELS,
                                &width,&height,&err)==FG_OK);
    CHECK(width==992u&&height==800u);
    CHECK(fg_tower_smart_resize(100u,100u,FG_TOWER_ALIGN,FG_TOWER_IMAGE_MIN_PIXELS,
                                FG_TOWER_IMAGE_MAX_PIXELS,&width,&height,&err)==FG_OK);
    CHECK(width==256u&&height==256u);
    CHECK(fg_tower_smart_resize(5000u,5000u,FG_TOWER_ALIGN,0u,FG_TOWER_IMAGE_MAX_PIXELS,
                                &width,&height,&err)==FG_OK);
    CHECK(width%FG_TOWER_ALIGN==0u&&height%FG_TOWER_ALIGN==0u);
    CHECK((uint64_t)width*height<=FG_TOWER_IMAGE_MAX_PIXELS);
    CHECK(fg_tower_smart_resize(8192u,64u,FG_TOWER_ALIGN,0u,FG_TOWER_IMAGE_MAX_PIXELS,
                                &width,&height,&err)==FG_OK);
    CHECK((uint64_t)width*height<=FG_TOWER_IMAGE_MAX_PIXELS);
}

static void test_geometry_and_patchify(void){
    fg_error err={0};
    fg_tower_geometry geometry={0};
    CHECK(fg_tower_image_geometry(128u,128u,&geometry,&err)==FG_OK);
    CHECK(geometry.grid_width==8u&&geometry.grid_height==8u);
    CHECK(geometry.tokens==64u&&geometry.merged_tokens==16u);
    CHECK(fg_tower_image_geometry(100u,100u,&geometry,&err)==FG_ERR_FORMAT);
    float *image=malloc(3u*128u*128u*sizeof(float));
    float *tokens=malloc(64u*FG_TOWER_TOKEN_VALUES*sizeof(float));
    CHECK(image&&tokens);
    if(!image||!tokens){
        free(tokens);free(image);
        return;
    }
    for(uint32_t i=0;i<3u*128u*128u;i++)image[i]=(float)(i%251)*0.001f-0.5f;
    CHECK(fg_tower_image_geometry(128u,128u,&geometry,&err)==FG_OK);
    CHECK(fg_tower_patchify(image,128u,128u,tokens,&geometry,&err)==FG_OK);
    for(uint32_t token=0;token<64u;token++){
        const uint32_t group=token/4u,inner=token%4u;
        const uint32_t dx=inner%2u,dy=inner/2u;
        const uint32_t bx=group%4u,by=group/4u;
        const uint32_t x=bx*2u+dx,y=by*2u+dy;
        const float *patch=tokens+(size_t)token*FG_TOWER_TOKEN_VALUES;
        for(uint32_t c=0;c<3u;c++)
            for(uint32_t py=0;py<16u;py++)
                for(uint32_t px=0;px<16u;px++){
                    const float expected=image[((size_t)c*128u+y*16u+py)*128u+x*16u+px];
                    CHECK(patch[(c*16u+py)*16u+px]==expected);
                    CHECK(patch[FG_TOWER_PATCH_VALUES+(c*16u+py)*16u+px]==expected);
                }
    }
    free(tokens);
    free(image);
}

static double reference_filter(double x){
    const double a=-0.5;
    if(x<0.0)x=-x;
    if(x<1.0)return ((a+2.0)*x-(a+3.0))*x*x+1.0;
    if(x<2.0)return (((x-5.0)*x+8.0)*x-4.0)*a;
    return 0.0;
}

static void reference_resample_line(const double *source,uint32_t in_size,uint32_t out_size,
                                    double *destination){
    double filterscale=(double)in_size/(double)out_size;
    if(filterscale<1.0)filterscale=1.0;
    const double radius=2.0*filterscale;
    const double scale=(double)in_size/(double)out_size;
    for(uint32_t xx=0;xx<out_size;xx++){
        const double center=((double)xx+0.5)*scale;
        int first=(int)(center-radius+0.5);
        if(first<0)first=0;
        int last=(int)(center+radius+0.5);
        if(last>(int)in_size)last=(int)in_size;
        double total=0.0;
        for(int x=first;x<last;x++)total+=reference_filter(((double)x-center+0.5)/filterscale);
        double accumulator=0.0;
        for(int x=first;x<last;x++)
            accumulator+=source[x]*reference_filter(((double)x-center+0.5)/filterscale)/total;
        destination[xx]=accumulator;
    }
}

static void test_resize_reference(void){
    fg_error err={0};
    const uint32_t source_width=37u,source_height=29u;
    const uint32_t width=64u,height=48u;
    float *source=malloc(source_width*source_height*3u*sizeof(float));
    float *destination=malloc(width*height*3u*sizeof(float));
    double *column=malloc((source_height>height?source_height:height)*sizeof(double));
    double *sampled=malloc((width>height?width:height)*sizeof(double));
    double *expected=malloc(width*height*3u*sizeof(double));
    CHECK(source&&destination&&column&&sampled&&expected);
    if(!source||!destination||!column||!sampled||!expected){
        free(expected);free(sampled);free(column);free(destination);free(source);
        return;
    }
    for(uint32_t i=0;i<source_width*source_height*3u;i++)source[i]=rng_next();
    CHECK(fg_tower_resize_bicubic(source,source_width,source_height,destination,width,height,
                                  &err)==FG_OK);
    for(uint32_t c=0;c<3u;c++){
        for(uint32_t y=0;y<source_height;y++){
            for(uint32_t x=0;x<source_width;x++)column[x]=source[(c*source_height+y)*source_width+x];
            reference_resample_line(column,source_width,width,sampled);
            for(uint32_t x=0;x<width;x++)
                expected[(c*height+y)*width+x]=sampled[x];
        }
        for(uint32_t x=0;x<width;x++){
            for(uint32_t y=0;y<source_height;y++)column[y]=expected[(c*height+y)*width+x];
            reference_resample_line(column,source_height,height,sampled);
            for(uint32_t y=0;y<height;y++)expected[(c*height+y)*width+x]=sampled[y];
        }
    }
    double max_difference=0.0;
    for(size_t i=0;i<(size_t)width*height*3u;i++){
        const double difference=fabs((double)destination[i]-expected[i]);
        if(difference>max_difference)max_difference=difference;
    }
    printf("resize reference: max_abs_diff=%.3e\n",max_difference);
    CHECK(max_difference<1e-4);
    free(expected);free(sampled);free(column);free(destination);free(source);
}

static void test_positions(void){
    fg_error err={0};
    fg_tower_geometry geometry={0};
    CHECK(fg_tower_image_geometry(128u,128u,&geometry,&err)==FG_OK);
    int32_t positions[4u*64u];
    CHECK(fg_tower_vision_positions(&geometry,positions,&err)==FG_OK);
    for(uint32_t token=0;token<geometry.tokens;token++){
        const uint32_t group=token/4u,inner=token%4u;
        const uint32_t dx=inner%2u,dy=inner/2u;
        const uint32_t x=(group%4u)*2u+dx,y=(group/4u)*2u+dy;
        CHECK(positions[token]==(int32_t)y);
        CHECK(positions[64u+token]==(int32_t)x);
        CHECK(positions[128u+token]==(int32_t)y);
        CHECK(positions[192u+token]==(int32_t)x);
    }
    const uint32_t grid=FG_TOWER_POS_GRID;
    float *position_weight=malloc((size_t)grid*grid*FG_TOWER_HIDDEN*sizeof(float));
    float *tokens_out=malloc(64u*FG_TOWER_HIDDEN*sizeof(float));
    CHECK(position_weight&&tokens_out);
    if(position_weight&&tokens_out){
        for(size_t i=0;i<(size_t)grid*grid*FG_TOWER_HIDDEN;i++)position_weight[i]=rng_next();
        CHECK(fg_tower_position_embeddings(position_weight,&geometry,tokens_out,&err)==FG_OK);
        for(uint32_t token=0;token<64u;token++){
            const uint32_t group=token/4u,inner=token%4u;
            const uint32_t dx=inner%2u,dy=inner/2u;
            const uint32_t x=(group%4u)*2u+dx,y=(group/4u)*2u+dy;
            const double fx=(double)x*47.0/7.0,fy=(double)y*47.0/7.0;
            const uint32_t x0=(uint32_t)fx,y0=(uint32_t)fy;
            const uint32_t x1=x0+1u<grid?x0+1u:grid-1u;
            const uint32_t y1=y0+1u<grid?y0+1u:grid-1u;
            const double tx=fx-x0,ty=fy-y0;
            for(uint32_t c=0;c<FG_TOWER_HIDDEN;c+=97u){
                const double p00=position_weight[((size_t)y0*grid+x0)*FG_TOWER_HIDDEN+c];
                const double p01=position_weight[((size_t)y0*grid+x1)*FG_TOWER_HIDDEN+c];
                const double p10=position_weight[((size_t)y1*grid+x0)*FG_TOWER_HIDDEN+c];
                const double p11=position_weight[((size_t)y1*grid+x1)*FG_TOWER_HIDDEN+c];
                const double top=p00*(1.0-tx)+p01*tx;
                const double bottom=p10*(1.0-tx)+p11*tx;
                const double expected=top*(1.0-ty)+bottom*ty;
                const double actual=tokens_out[(size_t)token*FG_TOWER_HIDDEN+c];
                CHECK(fabs(actual-expected)<1e-5);
            }
        }
        free(tokens_out);
        free(position_weight);
    }
}

static void randomize_weights(fg_tower_weights *weights,float scale){
    weights->patch_weight=malloc((size_t)FG_TOWER_HIDDEN*FG_TOWER_TOKEN_VALUES*sizeof(float));
    weights->patch_bias=malloc(FG_TOWER_HIDDEN*sizeof(float));
    weights->position_weight=malloc((size_t)FG_TOWER_POS_GRID*FG_TOWER_POS_GRID*FG_TOWER_HIDDEN*
                                    sizeof(float));
    weights->post_ln_weight=malloc(FG_TOWER_HIDDEN*sizeof(float));
    weights->post_ln_bias=malloc(FG_TOWER_HIDDEN*sizeof(float));
    weights->merger_fc1_weight=malloc((size_t)FG_TOWER_MERGED_WIDTH*FG_TOWER_MERGED_WIDTH*
                                      sizeof(float));
    weights->merger_fc1_bias=malloc(FG_TOWER_MERGED_WIDTH*sizeof(float));
    weights->merger_fc2_weight=malloc((size_t)FG_TOWER_OUT_HIDDEN*FG_TOWER_MERGED_WIDTH*
                                      sizeof(float));
    weights->merger_fc2_bias=malloc(FG_TOWER_OUT_HIDDEN*sizeof(float));
    CHECK(weights->patch_weight&&weights->patch_bias&&weights->position_weight&&
          weights->post_ln_weight&&weights->post_ln_bias&&weights->merger_fc1_weight&&
          weights->merger_fc1_bias&&weights->merger_fc2_weight&&weights->merger_fc2_bias);
    fill_random((float *)weights->patch_weight,
                (size_t)FG_TOWER_HIDDEN*FG_TOWER_TOKEN_VALUES,scale);
    fill_random((float *)weights->patch_bias,FG_TOWER_HIDDEN,0.05f);
    fill_random((float *)weights->position_weight,
                (size_t)FG_TOWER_POS_GRID*FG_TOWER_POS_GRID*FG_TOWER_HIDDEN,scale);
    fill_random((float *)weights->post_ln_weight,FG_TOWER_HIDDEN,0.1f);
    fill_random((float *)weights->post_ln_bias,FG_TOWER_HIDDEN,0.05f);
    fill_random((float *)weights->merger_fc1_weight,
                (size_t)FG_TOWER_MERGED_WIDTH*FG_TOWER_MERGED_WIDTH,scale);
    fill_random((float *)weights->merger_fc1_bias,FG_TOWER_MERGED_WIDTH,0.05f);
    fill_random((float *)weights->merger_fc2_weight,
                (size_t)FG_TOWER_OUT_HIDDEN*FG_TOWER_MERGED_WIDTH,scale);
    fill_random((float *)weights->merger_fc2_bias,FG_TOWER_OUT_HIDDEN,0.05f);
    fg_tower_block_weights *block=&weights->blocks[0];
    block->ln1_weight=malloc(FG_TOWER_HIDDEN*sizeof(float));
    block->ln1_bias=malloc(FG_TOWER_HIDDEN*sizeof(float));
    block->ln2_weight=malloc(FG_TOWER_HIDDEN*sizeof(float));
    block->ln2_bias=malloc(FG_TOWER_HIDDEN*sizeof(float));
    block->qkv_weight=malloc((size_t)FG_TOWER_QKV_WIDTH*FG_TOWER_HIDDEN*sizeof(float));
    block->qkv_bias=malloc(FG_TOWER_QKV_WIDTH*sizeof(float));
    block->attn_out_weight=malloc((size_t)FG_TOWER_HIDDEN*FG_TOWER_HIDDEN*sizeof(float));
    block->attn_out_bias=malloc(FG_TOWER_HIDDEN*sizeof(float));
    block->ffn_up_weight=malloc((size_t)FG_TOWER_MLP*FG_TOWER_HIDDEN*sizeof(float));
    block->ffn_up_bias=malloc(FG_TOWER_MLP*sizeof(float));
    block->ffn_down_weight=malloc((size_t)FG_TOWER_HIDDEN*FG_TOWER_MLP*sizeof(float));
    block->ffn_down_bias=malloc(FG_TOWER_HIDDEN*sizeof(float));
    CHECK(block->ln1_weight&&block->ln1_bias&&block->ln2_weight&&block->ln2_bias&&
          block->qkv_weight&&block->qkv_bias&&block->attn_out_weight&&block->attn_out_bias&&
          block->ffn_up_weight&&block->ffn_up_bias&&block->ffn_down_weight&&
          block->ffn_down_bias);
    fill_random((float *)block->ln1_weight,FG_TOWER_HIDDEN,0.1f);
    fill_random((float *)block->ln1_bias,FG_TOWER_HIDDEN,0.05f);
    fill_random((float *)block->ln2_weight,FG_TOWER_HIDDEN,0.1f);
    fill_random((float *)block->ln2_bias,FG_TOWER_HIDDEN,0.05f);
    fill_random((float *)block->qkv_weight,(size_t)FG_TOWER_QKV_WIDTH*FG_TOWER_HIDDEN,scale);
    fill_random((float *)block->qkv_bias,FG_TOWER_QKV_WIDTH,0.05f);
    fill_random((float *)block->attn_out_weight,
                (size_t)FG_TOWER_HIDDEN*FG_TOWER_HIDDEN,scale);
    fill_random((float *)block->attn_out_bias,FG_TOWER_HIDDEN,0.05f);
    fill_random((float *)block->ffn_up_weight,(size_t)FG_TOWER_MLP*FG_TOWER_HIDDEN,scale);
    fill_random((float *)block->ffn_up_bias,FG_TOWER_MLP,0.05f);
    fill_random((float *)block->ffn_down_weight,(size_t)FG_TOWER_HIDDEN*FG_TOWER_MLP,scale);
    fill_random((float *)block->ffn_down_bias,FG_TOWER_HIDDEN,0.05f);
    for(uint32_t layer=1;layer<FG_TOWER_LAYERS;layer++)weights->blocks[layer]=weights->blocks[0];
}

static void test_cpu_forward(void){
    fg_error err={0};
    fg_tower_geometry geometry={0};
    CHECK(fg_tower_image_geometry(64u,64u,&geometry,&err)==FG_OK);
    fg_tower_weights weights={0};
    randomize_weights(&weights,0.02f);
    const uint32_t count=geometry.tokens;
    float *tokens=malloc((size_t)count*FG_TOWER_TOKEN_VALUES*sizeof(float));
    float *embeddings=malloc((size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN*sizeof(float));
    float *repeat=malloc((size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN*sizeof(float));
    CHECK(tokens&&embeddings&&repeat);
    if(tokens&&embeddings&&repeat){
        fill_random(tokens,(size_t)count*FG_TOWER_TOKEN_VALUES,1.0f);
        CHECK(fg_tower_forward_cpu(&weights,tokens,&geometry,embeddings,&err)==FG_OK);
        CHECK(all_finite(embeddings,(size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN));
        CHECK(fg_tower_forward_cpu(&weights,tokens,&geometry,repeat,&err)==FG_OK);
        CHECK(memcmp(embeddings,repeat,(size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN*
                     sizeof(float))==0);
        double norm=0.0;
        for(size_t i=0;i<(size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN;i++)
            norm+=(double)embeddings[i]*embeddings[i];
        CHECK(norm>0.0);
        printf("cpu forward: tokens=%u merged=%u norm=%.4f\n",count,geometry.merged_tokens,
               sqrt(norm));
    }
    free(repeat);free(embeddings);free(tokens);
    free((void *)weights.patch_weight);free((void *)weights.patch_bias);
    free((void *)weights.position_weight);free((void *)weights.post_ln_weight);
    free((void *)weights.post_ln_bias);free((void *)weights.merger_fc1_weight);
    free((void *)weights.merger_fc1_bias);free((void *)weights.merger_fc2_weight);
    free((void *)weights.merger_fc2_bias);
    free((void *)weights.blocks[0].ln1_weight);free((void *)weights.blocks[0].ln1_bias);
    free((void *)weights.blocks[0].ln2_weight);free((void *)weights.blocks[0].ln2_bias);
    free((void *)weights.blocks[0].qkv_weight);free((void *)weights.blocks[0].qkv_bias);
    free((void *)weights.blocks[0].attn_out_weight);free((void *)weights.blocks[0].attn_out_bias);
    free((void *)weights.blocks[0].ffn_up_weight);free((void *)weights.blocks[0].ffn_up_bias);
    free((void *)weights.blocks[0].ffn_down_weight);free((void *)weights.blocks[0].ffn_down_bias);
}

static void test_vulkan_stages(void){
    fg_error err={0};
    fg_tower_vk *tower=NULL;
    if(fg_tower_vk_open(&tower,&err)!=FG_OK){
        printf("vulkan stages: SKIP (%s)\n",err.message);
        return;
    }
    printf("vulkan device: %s\n",fg_tower_vk_device_name(tower));
    fg_tower_geometry geometry={0};
    CHECK(fg_tower_image_geometry(128u,128u,&geometry,&err)==FG_OK);
    fg_tower_weights weights={0};
    randomize_weights(&weights,0.02f);
    const uint32_t count=geometry.tokens;
    float *tokens=malloc((size_t)count*FG_TOWER_TOKEN_VALUES*sizeof(float));
    float *cpu_hidden=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    float *gpu_hidden=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    float *cpu_block=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    float *gpu_block=malloc((size_t)count*FG_TOWER_HIDDEN*sizeof(float));
    float *cpu_merged=malloc((size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN*sizeof(float));
    float *gpu_merged=malloc((size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN*sizeof(float));
    float *merged_input=malloc((size_t)geometry.merged_tokens*FG_TOWER_MERGED_WIDTH*
                               sizeof(float));
    CHECK(tokens&&cpu_hidden&&gpu_hidden&&cpu_block&&gpu_block&&cpu_merged&&gpu_merged&&
          merged_input);
    if(tokens&&cpu_hidden&&gpu_hidden&&cpu_block&&gpu_block&&cpu_merged&&gpu_merged&&
       merged_input){
        fill_random(tokens,(size_t)count*FG_TOWER_TOKEN_VALUES,1.0f);
        fill_random(cpu_block,(size_t)count*FG_TOWER_HIDDEN,0.5f);
        memcpy(gpu_block,cpu_block,(size_t)count*FG_TOWER_HIDDEN*sizeof(float));
        fill_random(merged_input,(size_t)geometry.merged_tokens*FG_TOWER_MERGED_WIDTH,0.5f);
        fg_tower_vk_weights *device_weights=NULL;
        fg_status status=fg_tower_vk_weights_upload(tower,&weights,&device_weights,&err);
        CHECK(status==FG_OK);
        if(status==FG_OK){
            fg_tower_vk_stats stats={0};
            CHECK(fg_tower_embed_cpu(&weights,tokens,&geometry,cpu_hidden,&err)==FG_OK);
            CHECK(fg_tower_vk_run_patch(tower,device_weights,tokens,&geometry,gpu_hidden,&stats,
                                        &err)==FG_OK);
            const double embed_cosine=cosine_similarity(cpu_hidden,gpu_hidden,
                (size_t)count*FG_TOWER_HIDDEN);
            printf("patch/pos parity: cosine=%.9f relative=%.3e (%.3f ms)\n",embed_cosine,
                   relative_error(gpu_hidden,cpu_hidden,(size_t)count*FG_TOWER_HIDDEN),
                   stats.forward_ms);
            CHECK(embed_cosine>=0.999);
            CHECK(fg_tower_block_cpu(&weights.blocks[0],cpu_block,count,geometry.grid_width,
                                     cpu_block,&err)==FG_OK);
            CHECK(fg_tower_vk_run_block(tower,device_weights,0u,gpu_block,count,
                                        geometry.grid_width,gpu_block,&stats,&err)==FG_OK);
            const double block_cosine=cosine_similarity(cpu_block,gpu_block,
                (size_t)count*FG_TOWER_HIDDEN);
            printf("block parity: cosine=%.9f relative=%.3e (%.3f ms, %u dispatches)\n",
                   block_cosine,
                   relative_error(gpu_block,cpu_block,(size_t)count*FG_TOWER_HIDDEN),
                   stats.forward_ms,stats.dispatches);
            CHECK(block_cosine>=0.999);
            float *merger_input=malloc((size_t)geometry.merged_tokens*FG_TOWER_MERGED_WIDTH*
                                       sizeof(float));
            if(merger_input){
                memcpy(merger_input,merged_input,(size_t)geometry.merged_tokens*
                       FG_TOWER_MERGED_WIDTH*sizeof(float));
                CHECK(fg_tower_merger_cpu(&weights,merger_input,geometry.merged_tokens,
                                          cpu_merged,&err)==FG_OK);
                CHECK(fg_tower_vk_run_merger(tower,device_weights,merged_input,
                                             geometry.merged_tokens,gpu_merged,&stats,&err)==FG_OK);
                const double merger_cosine=cosine_similarity(cpu_merged,gpu_merged,
                    (size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN);
                printf("merger parity: cosine=%.9f relative=%.3e (%.3f ms)\n",merger_cosine,
                       relative_error(gpu_merged,cpu_merged,
                                      (size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN),
                       stats.forward_ms);
                CHECK(merger_cosine>=0.999);
                free(merger_input);
            }
            fg_tower_vk_weights_destroy(tower,device_weights);
        }
    }
    free(merged_input);free(gpu_merged);free(cpu_merged);free(gpu_block);free(cpu_block);
    free(gpu_hidden);free(cpu_hidden);free(tokens);
    fg_tower_vk_close(tower);
    free((void *)weights.patch_weight);free((void *)weights.patch_bias);
    free((void *)weights.position_weight);free((void *)weights.post_ln_weight);
    free((void *)weights.post_ln_bias);free((void *)weights.merger_fc1_weight);
    free((void *)weights.merger_fc1_bias);free((void *)weights.merger_fc2_weight);
    free((void *)weights.merger_fc2_bias);
    free((void *)weights.blocks[0].ln1_weight);free((void *)weights.blocks[0].ln1_bias);
    free((void *)weights.blocks[0].ln2_weight);free((void *)weights.blocks[0].ln2_bias);
    free((void *)weights.blocks[0].qkv_weight);free((void *)weights.blocks[0].qkv_bias);
    free((void *)weights.blocks[0].attn_out_weight);free((void *)weights.blocks[0].attn_out_bias);
    free((void *)weights.blocks[0].ffn_up_weight);free((void *)weights.blocks[0].ffn_up_bias);
    free((void *)weights.blocks[0].ffn_down_weight);free((void *)weights.blocks[0].ffn_down_bias);
}

/* ---------- real tower pack probe ---------- */

static uint8_t *read_file(const char *path,uint64_t *bytes,fg_error *err){
    FILE *file=fopen(path,"rb");
    if(!file){
        fg_error_set(err,FG_ERR_IO,"open %s: %s",path,strerror(errno));
        return NULL;
    }
    fseek(file,0,SEEK_END);
    long length=ftell(file);
    fseek(file,0,SEEK_SET);
    if(length<=0){
        fclose(file);
        fg_error_set(err,FG_ERR_FORMAT,"%s is empty",path);
        return NULL;
    }
    uint8_t *data=malloc((size_t)length);
    if(!data||fread(data,1,(size_t)length,file)!=(size_t)length){
        fclose(file);
        free(data);
        fg_error_set(err,FG_ERR_IO,"read %s",path);
        return NULL;
    }
    fclose(file);
    *bytes=(uint64_t)length;
    return data;
}

static float *dequantize_tensor(const uint8_t *data,uint64_t bytes,uint32_t ggml_type,
                                uint64_t values){
    float *out=malloc((size_t)values*sizeof(float));
    if(!out)return NULL;
    if(ggml_type==0u){
        if(bytes!=values*4u){free(out);return NULL;}
        memcpy(out,data,(size_t)bytes);
        return out;
    }
    if(ggml_type==1u){
        if(bytes!=values*2u){free(out);return NULL;}
        const uint16_t *source=(const uint16_t *)data;
        for(uint64_t i=0;i<values;i++)out[i]=fg_f16_to_f32(source[i]);
        return out;
    }
    if(ggml_type==8u){
        if(values%FG_QK8_0||bytes!=(values/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES){free(out);return NULL;}
        const uint8_t *block=data;
        for(uint64_t i=0;i<values;i+=FG_QK8_0){
            const float scale=fg_f16_to_f32((uint16_t)(block[0]|(block[1]<<8)));
            for(uint32_t j=0;j<FG_QK8_0;j++)
                out[i+j]=scale*(float)(int8_t)block[2u+j];
            block+=FG_Q8_0_BLOCK_BYTES;
        }
        return out;
    }
    free(out);
    return NULL;
}

static bool tensor_values(const char *name,uint32_t dims,const uint64_t shape[4],
                          uint64_t *values){
    uint64_t count=1;
    for(uint32_t d=0;d<dims;d++){
        if(!shape[d]||count>UINT64_MAX/shape[d])return false;
        count*=shape[d];
    }
    (void)name;
    *values=count;
    return true;
}

typedef struct probe_owned {
    void *pointers[512];
    uint32_t count;
} probe_owned;

static double probe_now_ms(void){
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return (double)now.tv_sec*1000.0+(double)now.tv_nsec/1.0e6;
}

static float *probe_take(probe_owned *owned,float *data){
    if(owned->count<512u)owned->pointers[owned->count++]=data;
    return data;
}

static fg_status probe_load_weights(const char *tower_dir,fg_tower_weights *weights,
                                    probe_owned *owned,fg_error *err){
    char manifest_path[1200],pack_path[1200];
    snprintf(manifest_path,sizeof(manifest_path),"%s/tower.fgm",tower_dir);
    snprintf(pack_path,sizeof(pack_path),"%s/tower.fgw",tower_dir);
    fg_manifest *manifest=malloc(sizeof(*manifest));
    if(!manifest){
        fg_error_set(err,FG_ERR_OOM,"allocate tower manifest");
        return FG_ERR_OOM;
    }
    fg_status status=fg_manifest_read(manifest_path,manifest,err);
    if(status!=FG_OK){free(manifest);return status;}
    if(!(manifest->flags&FG_MANIFEST_HAS_VISION)){
        free(manifest);
        fg_error_set(err,FG_ERR_FORMAT,"tower manifest has no vision component");
        return FG_ERR_FORMAT;
    }
    uint64_t pack_bytes=0;
    uint8_t *pack=read_file(pack_path,&pack_bytes,err);
    if(!pack){free(manifest);return err->code;}
    float *patch0=NULL,*patch1=NULL;
    uint32_t block_tensors=0;
    for(uint32_t i=0;i<manifest->tensor_count&&status==FG_OK;i++){
        const fg_tensor_record *record=&manifest->tensors[i];
        if(record->kind!=FG_TENSOR_VISION)continue;
        if(record->offset+record->bytes>pack_bytes){
            status=FG_ERR_FORMAT;
            fg_error_set(err,FG_ERR_FORMAT,"tower tensor %s exceeds payload",record->name);
            break;
        }
        uint64_t values=0;
        if(!tensor_values(record->name,record->dims,record->shape,&values)){
            status=FG_ERR_FORMAT;
            fg_error_set(err,FG_ERR_FORMAT,"tower tensor %s shape overflows",record->name);
            break;
        }
        float *data=dequantize_tensor(pack+record->offset,record->bytes,record->ggml_type,values);
        if(!data){
            status=FG_ERR_FORMAT;
            fg_error_set(err,FG_ERR_FORMAT,"tower tensor %s type %u cannot be dequantized",
                         record->name,record->ggml_type);
            break;
        }
        probe_take(owned,data);
        if(!strcmp(record->name,"v.patch_embd.weight"))patch0=data;
        else if(!strcmp(record->name,"v.patch_embd.weight.1"))patch1=data;
        else if(!strcmp(record->name,"v.patch_embd.bias"))weights->patch_bias=data;
        else if(!strcmp(record->name,"v.position_embd.weight"))weights->position_weight=data;
        else if(!strcmp(record->name,"v.post_ln.weight"))weights->post_ln_weight=data;
        else if(!strcmp(record->name,"v.post_ln.bias"))weights->post_ln_bias=data;
        else if(!strcmp(record->name,"mm.0.weight"))weights->merger_fc1_weight=data;
        else if(!strcmp(record->name,"mm.0.bias"))weights->merger_fc1_bias=data;
        else if(!strcmp(record->name,"mm.2.weight"))weights->merger_fc2_weight=data;
        else if(!strcmp(record->name,"mm.2.bias"))weights->merger_fc2_bias=data;
        else{
            uint32_t layer=0;
            char suffix[64];
            if(sscanf(record->name,"v.blk.%u.%63s",&layer,suffix)==2&&layer<FG_TOWER_LAYERS){
                fg_tower_block_weights *block=&weights->blocks[layer];
                if(!strcmp(suffix,"ln1.weight"))block->ln1_weight=data;
                else if(!strcmp(suffix,"ln1.bias"))block->ln1_bias=data;
                else if(!strcmp(suffix,"ln2.weight"))block->ln2_weight=data;
                else if(!strcmp(suffix,"ln2.bias"))block->ln2_bias=data;
                else if(!strcmp(suffix,"attn_qkv.weight"))block->qkv_weight=data;
                else if(!strcmp(suffix,"attn_qkv.bias"))block->qkv_bias=data;
                else if(!strcmp(suffix,"attn_out.weight"))block->attn_out_weight=data;
                else if(!strcmp(suffix,"attn_out.bias"))block->attn_out_bias=data;
                else if(!strcmp(suffix,"ffn_up.weight"))block->ffn_up_weight=data;
                else if(!strcmp(suffix,"ffn_up.bias"))block->ffn_up_bias=data;
                else if(!strcmp(suffix,"ffn_down.weight"))block->ffn_down_weight=data;
                else if(!strcmp(suffix,"ffn_down.bias"))block->ffn_down_bias=data;
                block_tensors++;
            }
        }
    }
    free(pack);
    free(manifest);
    if(status!=FG_OK)return status;
    if(!patch0||!patch1||!weights->patch_bias||!weights->position_weight||
       !weights->post_ln_weight||!weights->post_ln_bias||!weights->merger_fc1_weight||
       !weights->merger_fc1_bias||!weights->merger_fc2_weight||!weights->merger_fc2_bias||
       block_tensors!=FG_TOWER_LAYERS*12u){
        fg_error_set(err,FG_ERR_FORMAT,"tower pack is incomplete (%u block tensors)",block_tensors);
        return FG_ERR_FORMAT;
    }
    float *combined=malloc((size_t)FG_TOWER_HIDDEN*FG_TOWER_TOKEN_VALUES*sizeof(float));
    if(!combined){
        fg_error_set(err,FG_ERR_OOM,"allocate combined patch weight");
        return FG_ERR_OOM;
    }
    probe_take(owned,combined);
    for(uint32_t row=0;row<FG_TOWER_HIDDEN;row++){
        memcpy(combined+(size_t)row*FG_TOWER_TOKEN_VALUES,patch0+(size_t)row*FG_TOWER_PATCH_VALUES,
               FG_TOWER_PATCH_VALUES*sizeof(float));
        memcpy(combined+(size_t)row*FG_TOWER_TOKEN_VALUES+FG_TOWER_PATCH_VALUES,
               patch1+(size_t)row*FG_TOWER_PATCH_VALUES,FG_TOWER_PATCH_VALUES*sizeof(float));
    }
    weights->patch_weight=combined;
    return FG_OK;
}

static void probe_owned_free(probe_owned *owned){
    for(uint32_t i=0;i<owned->count;i++)free(owned->pointers[i]);
    owned->count=0;
}

static uint8_t *probe_load_ppm(const char *path,uint32_t *width,uint32_t *height,
                               fg_error *err){
    uint64_t bytes=0;
    uint8_t *data=read_file(path,&bytes,err);
    if(!data)return NULL;
    uint64_t offset=0;
    if(bytes<8u||data[0]!='P'||data[1]!='6'){
        free(data);
        fg_error_set(err,FG_ERR_FORMAT,"%s is not a P6 PPM image",path);
        return NULL;
    }
    offset=2u;
    uint32_t fields[3]={0,0,0};
    for(uint32_t field=0;field<3u;field++){
        while(offset<bytes&&(data[offset]==' '||data[offset]=='\n'||data[offset]=='\r'||
              data[offset]=='\t'))offset++;
        if(offset<bytes&&data[offset]=='#'){
            while(offset<bytes&&data[offset]!='\n')offset++;
            field--;
            continue;
        }
        if(offset>=bytes||data[offset]<'0'||data[offset]>'9'){
            free(data);
            fg_error_set(err,FG_ERR_FORMAT,"%s has a malformed PPM header",path);
            return NULL;
        }
        uint64_t value=0;
        while(offset<bytes&&data[offset]>='0'&&data[offset]<='9'){
            value=value*10u+(uint64_t)(data[offset]-'0');
            offset++;
        }
        if(value>UINT32_MAX){
            free(data);
            fg_error_set(err,FG_ERR_FORMAT,"%s PPM header overflows",path);
            return NULL;
        }
        fields[field]=(uint32_t)value;
    }
    if(offset>=bytes||fields[0]==0||fields[1]==0||fields[2]!=255u||
       offset+1u+(uint64_t)fields[0]*fields[1]*3u>bytes){
        free(data);
        fg_error_set(err,FG_ERR_FORMAT,"%s is not an 8-bit P6 PPM image",path);
        return NULL;
    }
    offset++;
    uint32_t w=fields[0],h=fields[1];
    uint8_t *planar=malloc((size_t)w*h*3u);
    if(!planar){
        free(data);
        fg_error_set(err,FG_ERR_OOM,"allocate PPM planar image");
        return NULL;
    }
    const uint8_t *pixels=data+offset;
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++)
            for(uint32_t c=0;c<3u;c++)
                planar[((size_t)c*h+y)*w+x]=pixels[((size_t)y*w+x)*3u+c];
    free(data);
    *width=w;
    *height=h;
    return planar;
}

static int probe_run(const char *tower_dir,const char *image_path,uint32_t repeats,bool cpu_ref,
                     bool layer_sweep,int layer_limit,bool smoke){
    fg_error err={0};
    fg_tower_weights weights={0};
    probe_owned owned={0};
    fg_status status=probe_load_weights(tower_dir,&weights,&owned,&err);
    if(status!=FG_OK){
        fprintf(stderr,"probe: %s\n",err.message);
        probe_owned_free(&owned);
        return 1;
    }
    uint32_t width=0,height=0;
    uint8_t *image=probe_load_ppm(image_path,&width,&height,&err);
    if(!image){
        fprintf(stderr,"probe: %s\n",err.message);
        probe_owned_free(&owned);
        return 1;
    }
    uint32_t resized_width=0,resized_height=0;
    status=fg_tower_smart_resize(width,height,FG_TOWER_ALIGN,FG_TOWER_IMAGE_MIN_PIXELS,
                                 FG_TOWER_IMAGE_MAX_PIXELS,&resized_width,&resized_height,&err);
    fg_tower_geometry geometry={0};
    if(status==FG_OK)status=fg_tower_image_geometry(resized_width,resized_height,&geometry,&err);
    float *normalized=NULL,*resized=NULL,*tokens=NULL;
    if(status==FG_OK){
        normalized=malloc((size_t)width*height*3u*sizeof(float));
        resized=malloc((size_t)resized_width*resized_height*3u*sizeof(float));
        tokens=malloc((size_t)geometry.tokens*FG_TOWER_TOKEN_VALUES*sizeof(float));
        if(!normalized||!resized||!tokens)status=FG_ERR_OOM;
    }
    if(status==FG_OK)status=fg_tower_normalize_image(image,width,height,normalized,&err);
    if(status==FG_OK)status=fg_tower_resize_bicubic(normalized,width,height,resized,
                                                    resized_width,resized_height,&err);
    if(status==FG_OK)status=fg_tower_patchify(resized,resized_width,resized_height,tokens,
                                              &geometry,&err);
    free(image);
    free(normalized);
    free(resized);
    if(status!=FG_OK){
        fprintf(stderr,"probe: %s\n",err.message);
        free(tokens);
        probe_owned_free(&owned);
        return 1;
    }
    printf("probe image %ux%u -> %ux%u grid=%ux%u tokens=%u merged=%u\n",width,height,
           resized_width,resized_height,geometry.grid_width,geometry.grid_height,
           geometry.tokens,geometry.merged_tokens);
    fg_tower_vk *tower=NULL;
    if(fg_tower_vk_open(&tower,&err)!=FG_OK){
        fprintf(stderr,"probe: %s\n",err.message);
        free(tokens);
        probe_owned_free(&owned);
        return 1;
    }
    printf("probe device: %s\n",fg_tower_vk_device_name(tower));
    const double upload_begin=probe_now_ms();
    fg_tower_vk_weights *device_weights=NULL;
    status=fg_tower_vk_weights_upload(tower,&weights,&device_weights,&err);
    const double upload_ms=probe_now_ms()-upload_begin;
    if(status!=FG_OK){
        fprintf(stderr,"probe upload: %s\n",err.message);
        fg_tower_vk_close(tower);
        free(tokens);
        probe_owned_free(&owned);
        return 1;
    }
    const size_t embedding_values=(size_t)geometry.merged_tokens*FG_TOWER_OUT_HIDDEN;
    float *first=malloc(embedding_values*sizeof(float));
    float *second=malloc(embedding_values*sizeof(float));
    if(!first||!second){
        fg_tower_vk_weights_destroy(tower,device_weights);
        fg_tower_vk_close(tower);
        free(tokens);
        probe_owned_free(&owned);
        return 1;
    }
    if(smoke){
        float values[256];
        for(uint32_t i=0;i<256u;i++)values[i]=(float)i*0.01f-1.0f;
        fg_status smoke_status=fg_tower_vk_debug_layernorm(tower,values,values,values,1u,256u,
                                                           first,&err);
        printf("smoke layernorm: %s finite=%s\n",smoke_status==FG_OK?"ok":"fail",
               smoke_status==FG_OK?(all_finite(first,256u)?"yes":"no"):"n/a");
        fg_tower_vk_weights_destroy(tower,device_weights);
        fg_tower_vk_close(tower);
        free(second);free(first);free(tokens);probe_owned_free(&owned);
        return smoke_status==FG_OK?0:1;
    }
    if(layer_limit>=0){
        const size_t hidden_values=(size_t)geometry.tokens*FG_TOWER_HIDDEN;
        float *hidden=malloc(hidden_values*sizeof(float));
        fg_status staged=FG_OK;
        fg_tower_vk_stats stage_stats={0};
        if(hidden){
            staged=fg_tower_vk_run_patch(tower,device_weights,tokens,&geometry,hidden,
                                         &stage_stats,&err);
            printf("staged patch: %s %.1f ms finite=%s\n",staged==FG_OK?"ok":"fail",
                   stage_stats.forward_ms,staged==FG_OK&&all_finite(hidden,hidden_values)
                   ?"yes":"no");
            for(int layer=0;layer<layer_limit&&staged==FG_OK;layer++){
                staged=fg_tower_vk_run_block(tower,device_weights,(uint32_t)layer,hidden,
                                             geometry.tokens,geometry.grid_width,hidden,
                                             &stage_stats,&err);
                printf("staged block %d: %s %.1f ms finite=%s\n",layer,
                       staged==FG_OK?"ok":"fail",stage_stats.forward_ms,
                       staged==FG_OK&&all_finite(hidden,hidden_values)?"yes":"no");
            }
            if(staged!=FG_OK)fprintf(stderr,"staged: %s\n",err.message);
            free(hidden);
        }
        fg_tower_vk_weights_destroy(tower,device_weights);
        fg_tower_vk_close(tower);
        free(second);free(first);free(tokens);probe_owned_free(&owned);
        return staged==FG_OK?0:1;
    }
    fg_tower_vk_stats stats={0};
    status=fg_tower_vk_run(tower,device_weights,tokens,&geometry,first,&stats,&err);
    if(status!=FG_OK){
        fprintf(stderr,"probe run: %s\n",err.message);
    }else{
        printf("probe run: %.1f ms upload=%.1f ms dispatches=%u finite=%s\n",stats.forward_ms,
               upload_ms,stats.dispatches,all_finite(first,embedding_values)?"yes":"no");
        double norm=0.0;
        for(size_t i=0;i<embedding_values;i++)norm+=(double)first[i]*first[i];
        printf("probe embedding norm=%.4f first=%.6f\n",sqrt(norm),
               first[0]);
        for(uint32_t repeat=0;repeat<repeats&&status==FG_OK;repeat++){
            fg_tower_vk_stats again={0};
            status=fg_tower_vk_run(tower,device_weights,tokens,&geometry,second,&again,&err);
            if(status!=FG_OK){
                fprintf(stderr,"probe repeat: %s\n",err.message);
                break;
            }
            const bool identical=memcmp(first,second,embedding_values*sizeof(float))==0;
            printf("probe repeat %u: %.1f ms cosine=%.9f identical=%s\n",repeat+1,
                   again.forward_ms,cosine_similarity(first,second,embedding_values),
                   identical?"yes":"no");
        }
        if(cpu_ref&&status==FG_OK){
            float *cpu_embeddings=malloc(embedding_values*sizeof(float));
            if(cpu_embeddings){
                const double cpu_begin=probe_now_ms();
                fg_status cpu_status=fg_tower_forward_cpu(&weights,tokens,&geometry,cpu_embeddings,
                                                          &err);
                const double cpu_ms=probe_now_ms()-cpu_begin;
                if(cpu_status==FG_OK)
                    printf("probe cpu reference: %.1f ms cosine=%.9f\n",cpu_ms,
                           cosine_similarity(first,cpu_embeddings,embedding_values));
                else
                    fprintf(stderr,"probe cpu reference: %s\n",err.message);
                free(cpu_embeddings);
            }
        }
        if(layer_sweep&&status==FG_OK){
            const size_t hidden_values=(size_t)geometry.tokens*FG_TOWER_HIDDEN;
            float *cpu_hidden=malloc(hidden_values*sizeof(float));
            float *gpu_hidden=malloc(hidden_values*sizeof(float));
            if(cpu_hidden&&gpu_hidden){
                fg_status sweep=fg_tower_embed_cpu(&weights,tokens,&geometry,cpu_hidden,&err);
                if(sweep==FG_OK)
                    sweep=fg_tower_vk_run_patch(tower,device_weights,tokens,&geometry,gpu_hidden,
                                                NULL,&err);
                if(sweep==FG_OK)
                    printf("sweep embed: cosine=%.9f\n",
                           cosine_similarity(cpu_hidden,gpu_hidden,hidden_values));
                for(uint32_t layer=0;layer<FG_TOWER_LAYERS&&sweep==FG_OK;layer++){
                    sweep=fg_tower_block_cpu(&weights.blocks[layer],cpu_hidden,geometry.tokens,
                                             geometry.grid_width,cpu_hidden,&err);
                    if(sweep==FG_OK)
                        sweep=fg_tower_vk_run_block(tower,device_weights,layer,gpu_hidden,
                                                    geometry.tokens,geometry.grid_width,
                                                    gpu_hidden,NULL,&err);
                    if(sweep!=FG_OK){
                        fprintf(stderr,"sweep layer %u: %s\n",layer,err.message);
                        break;
                    }
                    printf("sweep layer %2u: cosine=%.9f cpu_norm=%.4f gpu_norm=%.4f\n",layer,
                           cosine_similarity(cpu_hidden,gpu_hidden,hidden_values),
                           vector_norm(cpu_hidden,hidden_values),
                           vector_norm(gpu_hidden,hidden_values));
                }
            }
            free(gpu_hidden);
            free(cpu_hidden);
        }
    }
    fg_tower_vk_weights_destroy(tower,device_weights);
    fg_tower_vk_close(tower);
    free(second);
    free(first);
    free(tokens);
    probe_owned_free(&owned);
    return status==FG_OK?0:1;
}

int main(int argc,char **argv){
    const char *tower_dir=NULL,*image_path=NULL;
    uint32_t repeats=1u;
    bool cpu_ref=false,layer_sweep=false,smoke=false;
    int layer_limit=-1;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--tower-dir")&&i+1<argc)tower_dir=argv[++i];
        else if(!strcmp(argv[i],"--image")&&i+1<argc)image_path=argv[++i];
        else if(!strcmp(argv[i],"--repeat")&&i+1<argc)repeats=(uint32_t)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--cpu"))cpu_ref=true;
        else if(!strcmp(argv[i],"--layer-sweep"))layer_sweep=true;
        else if(!strcmp(argv[i],"--smoke"))smoke=true;
        else if(!strcmp(argv[i],"--layers")&&i+1<argc)layer_limit=atoi(argv[++i]);
        else{
            fprintf(stderr,"usage: test_tower [--tower-dir DIR --image FILE [--repeat N] "
                           "[--cpu] [--layer-sweep] [--smoke] [--layers N]]\n");
            return 2;
        }
    }
    if(tower_dir&&image_path)
        return probe_run(tower_dir,image_path,repeats,cpu_ref,layer_sweep,layer_limit,smoke);
    test_smart_resize();
    test_geometry_and_patchify();
    test_resize_reference();
    test_positions();
    test_cpu_forward();
    test_vulkan_stages();
    if(failures){fprintf(stderr,"%d tower test(s) failed\n",failures);return 1;}
    puts("tower tests: PASS");
    return 0;
}
