#include "fg_tower.h"
#include "fg_video.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

static uint32_t rng_state=0x9e3779b9u;

static float rng_next(void){
    rng_state=rng_state*1664525u+1013904223u;
    return (float)((rng_state>>8u)&0xFFFFFFu)/8388608.0f-1.0f;
}

static const uint8_t test_png[]={
    0x89u,0x50u,0x4eu,0x47u,0x0du,0x0au,0x1au,0x0au,0x00u,0x00u,0x00u,0x0du,0x49u,0x48u,
    0x44u,0x52u,0x00u,0x00u,0x00u,0x01u,0x00u,0x00u,0x00u,0x01u,0x08u,0x06u,0x00u,0x00u,
    0x00u,0x1fu,0x15u,0xc4u,0x89u,0x00u,0x00u,0x00u,0x0du,0x49u,0x44u,0x41u,0x54u,0x78u,
    0xdau,0x63u,0xfcu,0xcfu,0xc0u,0x50u,0x0fu,0x00u,0x04u,0x85u,0x01u,0x80u,0x84u,0xa9u,
    0x8cu,0x21u,0x00u,0x00u,0x00u,0x00u,0x49u,0x45u,0x4eu,0x44u,0xaeu,0x42u,0x60u,0x82u
};

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

static void reference_frame_patch(const float *frame,uint32_t width,uint32_t height,
                                  const fg_tower_geometry *geometry,uint32_t token,
                                  float *patch){
    const uint32_t half=geometry->grid_width/FG_TOWER_MERGE;
    const uint32_t group=token/FG_TOWER_MERGE_FACTOR;
    const uint32_t inner=token%FG_TOWER_MERGE_FACTOR;
    const uint32_t dx=inner%FG_TOWER_MERGE,dy=inner/FG_TOWER_MERGE;
    const uint32_t bx=group%half,by=group/half;
    for(uint32_t c=0;c<3u;c++)
        for(uint32_t py=0;py<FG_TOWER_PATCH_SIZE;py++){
            const uint32_t y=(by*FG_TOWER_MERGE+dy)*FG_TOWER_PATCH_SIZE+py;
            const float *source=frame+((size_t)c*height+y)*width+
                (bx*FG_TOWER_MERGE+dx)*FG_TOWER_PATCH_SIZE;
            memcpy(patch+c*FG_TOWER_PATCH_SIZE*FG_TOWER_PATCH_SIZE+py*FG_TOWER_PATCH_SIZE,
                   source,FG_TOWER_PATCH_SIZE*sizeof(float));
        }
}

static void test_options(void){
    fg_video_options options;
    fg_video_options_defaults(&options);
    CHECK(options.fps==FG_VIDEO_FPS_TARGET);
    CHECK(options.max_frames==FG_VIDEO_MAX_FRAMES);
    CHECK(options.max_pixels==FG_VIDEO_MAX_PIXELS);
    CHECK(options.max_tokens==FG_VIDEO_MAX_TOKENS);
    printf("video tower entry present: %s\n",fg_video_available()?"yes":"no");
    CHECK(!fg_video_available());
    CHECK(!fg_video_mp4_available(NULL));
}

static void test_select_frames(void){
    uint32_t indices[16];
    CHECK(fg_video_select_frames(30u,30.0,2.0,8u,indices)==2u);
    CHECK(indices[0]==0u&&indices[1]==15u);
    CHECK(fg_video_select_frames(24u,24.0,2.0,8u,indices)==2u);
    CHECK(indices[0]==0u&&indices[1]==12u);
    CHECK(fg_video_select_frames(5u,10.0,4.0,8u,indices)==2u);
    CHECK(indices[0]==0u&&indices[1]==3u);
    CHECK(fg_video_select_frames(8u,2.0,2.0,8u,indices)==8u);
    for(uint32_t i=0;i<8u;i++)CHECK(indices[i]==i);
    CHECK(fg_video_select_frames(9u,4.0,6.0,16u,indices)==9u);
    for(uint32_t i=0;i<9u;i++)CHECK(indices[i]==i);
    CHECK(fg_video_select_frames(10u,1.0,1.0,4u,indices)==4u);
    for(uint32_t i=0;i<4u;i++)CHECK(indices[i]==i);
    CHECK(fg_video_select_frames(1u,30.0,2.0,8u,indices)==1u);
    CHECK(indices[0]==0u);
    CHECK(fg_video_select_frames(0u,30.0,2.0,8u,indices)==0u);
    CHECK(fg_video_select_frames(16u,30.0,2.0,0u,indices)==0u);
}

static void test_token_count(void){
    CHECK(fg_video_token_count(0u,64u)==0u);
    CHECK(fg_video_token_count(1u,64u)==64u);
    CHECK(fg_video_token_count(2u,64u)==64u);
    CHECK(fg_video_token_count(3u,64u)==128u);
    CHECK(fg_video_token_count(4u,64u)==128u);
    CHECK(fg_video_token_count(7u,256u)==1024u);
    CHECK(fg_video_token_count(8u,1008u)==4032u);
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
    printf("video resize reference: max_abs_diff=%.3e\n",max_difference);
    CHECK(max_difference<1e-4);
    free(expected);free(sampled);free(column);free(destination);free(source);
}

static void test_superframes(void){
    fg_error err={0};
    const uint32_t width=64u,height=64u;
    const uint32_t frame_values=width*height*3u;
    float *frames=malloc(3u*frame_values*sizeof(float));
    CHECK(frames!=NULL);
    if(!frames)return;
    for(uint32_t frame=0;frame<3u;frame++)
        for(uint32_t i=0;i<frame_values;i++)
            frames[(size_t)frame*frame_values+i]=(float)frame*10.0f+(float)(i%97u)*0.01f;
    float *tokens=NULL;
    uint32_t pair_count=0;
    fg_tower_geometry geometry={0};
    CHECK(fg_video_superframes(frames,3u,width,height,&tokens,&pair_count,&geometry,&err)==FG_OK);
    CHECK(pair_count==2u);
    CHECK(geometry.tokens==16u);
    CHECK(geometry.merged_tokens==4u);
    if(tokens){
        const uint32_t first_frame[2]={0u,2u};
        const uint32_t second_frame[2]={1u,2u};
        float expected_first[FG_TOWER_PATCH_VALUES];
        float expected_second[FG_TOWER_PATCH_VALUES];
        for(uint32_t pair=0;pair<2u;pair++){
            for(uint32_t token=0;token<geometry.tokens;token++){
                reference_frame_patch(frames+(size_t)first_frame[pair]*frame_values,width,
                                      height,&geometry,token,expected_first);
                reference_frame_patch(frames+(size_t)second_frame[pair]*frame_values,width,
                                      height,&geometry,token,expected_second);
                const float *actual=tokens+
                    ((size_t)pair*geometry.tokens+token)*FG_TOWER_TOKEN_VALUES;
                for(uint32_t i=0;i<FG_TOWER_PATCH_VALUES;i++){
                    CHECK(actual[i]==expected_first[i]);
                    CHECK(actual[FG_TOWER_PATCH_VALUES+i]==expected_second[i]);
                }
            }
        }
        for(uint32_t pair=0;pair<2u;pair++)
            for(uint32_t token=0;token<geometry.tokens;token++)
                for(uint32_t i=0;i<FG_TOWER_TOKEN_VALUES;i++){
                    const float *actual=tokens+
                        ((size_t)pair*geometry.tokens+token)*FG_TOWER_TOKEN_VALUES;
                    CHECK(isfinite(actual[i]));
                }
    }
    free(tokens);
    free(frames);
}

static void test_pairing_matches_image(void){
    fg_error err={0};
    const uint32_t width=64u,height=64u;
    const uint32_t frame_values=width*height*3u;
    float *frame=malloc(frame_values*sizeof(float));
    float *image_tokens=malloc(16u*FG_TOWER_TOKEN_VALUES*sizeof(float));
    CHECK(frame&&image_tokens);
    if(frame&&image_tokens){
        for(uint32_t i=0;i<frame_values;i++)frame[i]=rng_next();
        fg_tower_geometry geometry={0};
        CHECK(fg_tower_image_geometry(width,height,&geometry,&err)==FG_OK);
        CHECK(fg_tower_patchify(frame,width,height,image_tokens,&geometry,&err)==FG_OK);
        float *duplicated=malloc(frame_values*2u*sizeof(float));
        CHECK(duplicated!=NULL);
        if(duplicated){
            memcpy(duplicated,frame,frame_values*sizeof(float));
            memcpy(duplicated+frame_values,frame,frame_values*sizeof(float));
            float *video_tokens=NULL;
            uint32_t pair_count=0;
            fg_tower_geometry video_geometry={0};
            CHECK(fg_video_superframes(duplicated,2u,width,height,&video_tokens,&pair_count,
                                       &video_geometry,&err)==FG_OK);
            CHECK(pair_count==1u);
            if(video_tokens){
                double max_difference=0.0;
                for(size_t i=0;i<(size_t)geometry.tokens*FG_TOWER_TOKEN_VALUES;i++){
                    const double difference=fabs((double)video_tokens[i]-
                                                 (double)image_tokens[i]);
                    if(difference>max_difference)max_difference=difference;
                }
                printf("video/image pairing identity: max_abs_diff=%.3e\n",max_difference);
                CHECK(max_difference==0.0);
            }
            free(video_tokens);
            free(duplicated);
        }
    }
    free(image_tokens);
    free(frame);
}

static void test_frames_preprocess(void){
    fg_error err={0};
    const uint8_t *frames[6]={test_png,test_png,test_png,test_png,test_png,test_png};
    const size_t lengths[6]={sizeof(test_png),sizeof(test_png),sizeof(test_png),
                             sizeof(test_png),sizeof(test_png),sizeof(test_png)};
    uint32_t indices[8];
    CHECK(fg_video_select_frames(6u,6.0,2.0,8u,indices)==2u);
    CHECK(indices[0]==0u&&indices[1]==3u);
    fg_video_options options;
    fg_video_options_defaults(&options);
    float *tokens=NULL;
    uint32_t pair_count=0;
    fg_tower_geometry geometry={0};
    CHECK(fg_video_frames_preprocess(frames,lengths,6u,indices,2u,&options,&tokens,&pair_count,
                                     &geometry,&err)==FG_OK);
    CHECK(pair_count==1u);
    CHECK(geometry.image_width==64u&&geometry.image_height==64u);
    CHECK(geometry.tokens==16u);
    CHECK(geometry.merged_tokens==4u);
    CHECK(fg_video_token_count(2u,geometry.merged_tokens)==4u);
    if(tokens)
        for(size_t i=0;i<(size_t)pair_count*geometry.tokens*FG_TOWER_TOKEN_VALUES;i++)
            CHECK(isfinite(tokens[i]));
    free(tokens);

    const uint32_t all=fg_video_select_frames(3u,0.0,2.0,4u,indices);
    CHECK(all==3u);
    CHECK(indices[0]==0u&&indices[1]==1u&&indices[2]==2u);
    tokens=NULL;
    pair_count=0;
    memset(&geometry,0,sizeof(geometry));
    CHECK(fg_video_frames_preprocess(frames,lengths,3u,indices,3u,&options,&tokens,&pair_count,
                                     &geometry,&err)==FG_OK);
    CHECK(pair_count==2u);
    CHECK(fg_video_token_count(3u,geometry.merged_tokens)==8u);
    if(tokens){
        const uint32_t pair_tokens=geometry.tokens;
        for(uint32_t token=0;token<pair_tokens;token++){
            const float *actual=tokens+((size_t)pair_tokens+token)*FG_TOWER_TOKEN_VALUES;
            for(uint32_t i=0;i<FG_TOWER_PATCH_VALUES;i++)
                CHECK(actual[i]==actual[FG_TOWER_PATCH_VALUES+i]);
        }
    }
    free(tokens);

    tokens=NULL;
    pair_count=0;
    memset(&geometry,0,sizeof(geometry));
    CHECK(fg_video_frames_preprocess(frames,lengths,6u,indices,0u,&options,&tokens,&pair_count,
                                     &geometry,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_video_frames_preprocess(NULL,lengths,6u,indices,2u,&options,&tokens,&pair_count,
                                     &geometry,&err)==FG_ERR_ARGUMENT);
}

static void test_positions(void){
    uint32_t positions[3u*18u];
    const uint32_t advance=fg_video_positions(positions,18u,3u,3u,2u,5u);
    CHECK(advance==5u+3u*2u);
    for(uint32_t group=0;group<2u;group++){
        for(uint32_t local=0;local<9u;local++){
            const uint32_t token=group*9u+local;
            const uint32_t base=5u+group*3u;
            CHECK(positions[token*3u]==base);
            CHECK(positions[token*3u+1u]==base+local/3u);
            CHECK(positions[token*3u+2u]==base+local%3u);
        }
    }
    const uint32_t image_advance=fg_video_positions(positions,9u,3u,3u,1u,7u);
    CHECK(image_advance==10u);
    CHECK(positions[8u*3u]==7u);
    CHECK(positions[8u*3u+1u]==9u);
    CHECK(positions[8u*3u+2u]==9u);
    uint32_t groups[3u*32u];
    CHECK(fg_video_positions(groups,32u,2u,2u,8u,3u)==3u+16u);
    for(uint32_t group=0;group<8u;group++)
        for(uint32_t local=0;local<4u;local++){
            const uint32_t token=group*4u+local;
            const uint32_t base=3u+group*2u;
            CHECK(groups[token*3u]==base);
            CHECK(groups[token*3u+1u]==base+local/2u);
            CHECK(groups[token*3u+2u]==base+local%2u);
        }
}

static void test_frames_forward_without_tower_entry(void){
    fg_error err={0};
    fg_video_clip clip={0};
    fg_tower_vk_stats stats={0};
    const uint8_t *frames[1]={test_png};
    const size_t lengths[1]={sizeof(test_png)};
    CHECK(fg_video_frames_forward("/nonexistent",frames,lengths,1u,2.0,NULL,&clip,&stats,&err)==
          FG_ERR_UNAVAILABLE);
    CHECK(strstr(err.message,"temporal token entry")!=NULL);
    CHECK(fg_video_frames_forward("/nonexistent",NULL,lengths,1u,2.0,NULL,&clip,&stats,&err)==
          FG_ERR_ARGUMENT);
    CHECK(fg_video_frames_forward("/nonexistent",frames,lengths,0u,2.0,NULL,&clip,&stats,&err)==
          FG_ERR_ARGUMENT);
    const uint8_t payload[4]={0,1,2,3};
    CHECK(fg_video_forward("/nonexistent",payload,sizeof(payload),NULL,&clip,&stats,&err)==
          FG_ERR_UNAVAILABLE);
    CHECK(fg_video_forward("/nonexistent",payload,0u,NULL,&clip,&stats,&err)==FG_ERR_ARGUMENT);
}

static void test_extract_ffmpeg(void){
    if(system("command -v ffmpeg >/dev/null 2>&1")!=0||
       system("command -v ffprobe >/dev/null 2>&1")!=0){
        printf("video extraction: SKIP (ffmpeg/ffprobe absent)\n");
        return;
    }
    const char *clip_path="fg-test-video.avi";
    char command[512];
    snprintf(command,sizeof(command),
             "ffmpeg -v error -y -f lavfi -i testsrc=size=96x80:rate=24 -t 2.5 "
             "-c:v mpeg4 -q:v 5 %s >/dev/null 2>&1",clip_path);
    if(system(command)!=0){
        printf("video extraction: SKIP (ffmpeg could not generate the fixture)\n");
        return;
    }
    fg_error err={0};
    fg_video_source source={0};
    CHECK(fg_video_probe(clip_path,NULL,&source,&err)==FG_OK);
    CHECK(source.width==96u&&source.height==80u);
    CHECK(fabs(source.fps-24.0)<0.5);
    CHECK(fabs(source.duration_seconds-2.5)<0.3);
    fg_video_options options;
    fg_video_options_defaults(&options);
    options.fps=2.0;
    options.max_frames=4u;
    float *frames=NULL;
    uint32_t frame_count=0,width=0,height=0;
    CHECK(fg_video_extract(clip_path,NULL,&options,&frames,&frame_count,&width,&height,&err)==
          FG_OK);
    CHECK(frame_count==4u);
    CHECK(width==96u&&height==96u);
    if(frames){
        float minimum=frames[0],maximum=frames[0];
        for(uint32_t frame=0;frame<frame_count;frame++)
            for(uint32_t i=0;i<width*height*3u;i++){
                const float value=frames[(size_t)frame*width*height*3u+i];
                CHECK(isfinite(value));
                if(value<minimum)minimum=value;
                if(value>maximum)maximum=value;
            }
        printf("video extraction range: min=%.4f max=%.4f\n",minimum,maximum);
        CHECK(minimum>=-1.25f&&maximum<=1.25f);
        double difference=0.0;
        for(uint32_t i=0;i<width*height*3u;i++)
            difference+=fabs((double)frames[i]-
                             (double)frames[(size_t)width*height*3u+i]);
        CHECK(difference>0.0);
        float *tokens=NULL;
        uint32_t pair_count=0;
        fg_tower_geometry geometry={0};
        CHECK(fg_video_superframes(frames,frame_count,width,height,&tokens,&pair_count,
                                   &geometry,&err)==FG_OK);
        CHECK(pair_count==2u);
        CHECK(fg_video_token_count(frame_count,geometry.merged_tokens)==18u);
        free(tokens);
    }
    free(frames);
    unlink(clip_path);
}

int main(void){
    printf("video preprocessing tests\n");
    test_options();
    test_select_frames();
    test_token_count();
    test_resize_reference();
    test_superframes();
    test_pairing_matches_image();
    test_frames_preprocess();
    test_positions();
    test_frames_forward_without_tower_entry();
    test_extract_ffmpeg();
    if(failures)fprintf(stderr,"%d video test(s) failed\n",failures);
    return failures?1:0;
}
