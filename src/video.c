#include "fg_video.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define VIDEO_MAX_SOURCE_PIXELS UINT64_C(2073600)

void fg_video_options_defaults(fg_video_options *options){
    if(!options)return;
    options->fps=FG_VIDEO_FPS_TARGET;
    options->max_frames=FG_VIDEO_MAX_FRAMES;
    options->max_pixels=FG_VIDEO_MAX_PIXELS;
    options->max_tokens=FG_VIDEO_MAX_TOKENS;
}

static bool video_path_is_safe(const char *path){
    return path&&*path&&!strchr(path,'\'')&&!strchr(path,'\n')&&!strchr(path,'\r');
}

static bool video_tool_path(const char *directory,const char *tool,char *output,size_t size){
    if(!tool||!output||!size)return false;
    if(directory&&*directory){
        char candidate[1200];
        if(snprintf(candidate,sizeof(candidate),"%s/tools/%s",directory,tool)<
           (int)sizeof(candidate)&&access(candidate,X_OK)==0&&video_path_is_safe(candidate)){
            snprintf(output,size,"%s",candidate);
            return true;
        }
        if(snprintf(candidate,sizeof(candidate),"%s/%s",directory,tool)<(int)sizeof(candidate)&&
           access(candidate,X_OK)==0&&video_path_is_safe(candidate)){
            snprintf(output,size,"%s",candidate);
            return true;
        }
    }
    const char *path=getenv("PATH");
    if(!path||!*path)return false;
    const size_t tool_length=strlen(tool);
    const char *cursor=path;
    while(*cursor){
        const char *end=strchr(cursor,':');
        const size_t length=end?(size_t)(end-cursor):strlen(cursor);
        if(length&&length+tool_length+2u<size){
            memcpy(output,cursor,length);
            output[length]='/';
            memcpy(output+length+1u,tool,tool_length+1u);
            if(access(output,X_OK)==0&&video_path_is_safe(output))return true;
        }
        if(!end)break;
        cursor=end+1;
    }
    return false;
}

static fg_status video_read_line(FILE *stream,char *line,size_t size){
    if(!fgets(line,(int)size,stream))return FG_ERR_IO;
    size_t length=strlen(line);
    while(length&&(line[length-1u]=='\n'||line[length-1u]=='\r'))line[--length]=0;
    return FG_OK;
}

static double video_parse_fraction(const char *text){
    unsigned numerator=0,denominator=1;
    if(!text||sscanf(text,"%u/%u",&numerator,&denominator)<2||!denominator)return 0.0;
    return (double)numerator/(double)denominator;
}

fg_status fg_video_probe(const char *path,const char *tool_dir,fg_video_source *source,
                         fg_error *err){
    if(!path||!source||!video_path_is_safe(path)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video probe request");
        return FG_ERR_ARGUMENT;
    }
    memset(source,0,sizeof(*source));
    char tool[1200];
    if(!video_tool_path(tool_dir,"ffprobe",tool,sizeof(tool))){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "MP4 video input requires a static ffprobe binary (tools directory or PATH)");
        return FG_ERR_UNAVAILABLE;
    }
    char command[2400];
    if(snprintf(command,sizeof(command),
                 "'%s' -v error -select_streams v:0 "
                 "-show_entries stream=width,height,avg_frame_rate,duration "
                 "-of default=noprint_wrappers=1:nokey=1 '%s'",
                 tool,path)>=(int)sizeof(command)){
        fg_error_set(err,FG_ERR_LIMIT,"video probe command exceeds buffer");
        return FG_ERR_LIMIT;
    }
    FILE *pipe=popen(command,"r");
    if(!pipe){
        fg_error_set(err,FG_ERR_IO,"spawn ffprobe: %s",strerror(errno));
        return FG_ERR_IO;
    }
    char line[128]={0};
    unsigned long width=0,height=0;
    char rate[128]={0},time[128]={0};
    fg_status status=video_read_line(pipe,line,sizeof(line));
    if(status==FG_OK){width=strtoul(line,NULL,10);status=video_read_line(pipe,line,sizeof(line));}
    if(status==FG_OK){height=strtoul(line,NULL,10);status=video_read_line(pipe,line,sizeof(line));}
    if(status==FG_OK){snprintf(rate,sizeof(rate),"%s",line);}
    if(status==FG_OK&&video_read_line(pipe,line,sizeof(line))==FG_OK)
        snprintf(time,sizeof(time),"%s",line);
    const int exit_code=pclose(pipe);
    const int code=exit_code==-1?-1:WEXITSTATUS(exit_code);
    if(status!=FG_OK||code!=0){
        fg_error_set(err,FG_ERR_FORMAT,"video probe failed for '%s'",path);
        return FG_ERR_FORMAT;
    }
    if(!width||!height||width>UINT32_MAX||height>UINT32_MAX){
        fg_error_set(err,FG_ERR_FORMAT,"video probe found no decodable video stream");
        return FG_ERR_FORMAT;
    }
    source->width=(uint32_t)width;
    source->height=(uint32_t)height;
    source->fps=video_parse_fraction(rate);
    source->duration_seconds=strtod(time,NULL);
    return FG_OK;
}

static int video_read_frame(FILE *pipe,uint8_t *buffer,size_t length){
    size_t filled=0;
    while(filled<length){
        const size_t count=fread(buffer+filled,1u,length-filled,pipe);
        if(!count)return filled?(-1):0;
        filled+=count;
    }
    return 1;
}

fg_status fg_video_extract(const char *path,const char *tool_dir,
                           const fg_video_options *options,float **frames,
                           uint32_t *frame_count,uint32_t *frame_width,uint32_t *frame_height,
                           fg_error *err){
    if(!path||!frames||!frame_count||!frame_width||!frame_height||!video_path_is_safe(path)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video extraction request");
        return FG_ERR_ARGUMENT;
    }
    fg_video_options defaults;
    fg_video_options_defaults(&defaults);
    const fg_video_options *resolved=options?options:&defaults;
    if(!(resolved->fps>0.0)||!resolved->max_frames||!resolved->max_pixels){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video extraction options");
        return FG_ERR_ARGUMENT;
    }
    *frames=NULL;
    *frame_count=0;
    fg_video_source source={0};
    fg_status status=fg_video_probe(path,tool_dir,&source,err);
    if(status!=FG_OK)return status;
    const uint64_t source_pixels=(uint64_t)source.width*source.height;
    if(source_pixels>VIDEO_MAX_SOURCE_PIXELS){
        fg_error_set(err,FG_ERR_LIMIT,"video frame %ux%u exceeds the source pixel limit",
                     source.width,source.height);
        return FG_ERR_LIMIT;
    }
    uint64_t max_pixels=resolved->max_pixels;
    if(max_pixels>FG_TOWER_VIDEO_MAX_PIXELS)max_pixels=FG_TOWER_VIDEO_MAX_PIXELS;
    uint32_t target_width=0,target_height=0;
    status=fg_tower_smart_resize(source.width,source.height,FG_TOWER_ALIGN,
                                 FG_TOWER_VIDEO_MIN_PIXELS,max_pixels,
                                 &target_width,&target_height,err);
    if(status!=FG_OK)return status;
    char tool[1200];
    if(!video_tool_path(tool_dir,"ffmpeg",tool,sizeof(tool))){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "MP4 video input requires a static ffmpeg binary (tools directory or PATH)");
        return FG_ERR_UNAVAILABLE;
    }
    uint32_t max_frames=resolved->max_frames;
    if(max_frames>FG_VIDEO_MAX_FRAMES)max_frames=FG_VIDEO_MAX_FRAMES;
    char command[2400];
    if(snprintf(command,sizeof(command),
                 "'%s' -v error -nostdin -noautorotate -i '%s' -vf fps=%.6g -frames:v %u "
                 "-f rawvideo -pix_fmt rgb24 -",
                 tool,path,resolved->fps,(unsigned)max_frames)>=(int)sizeof(command)){
        fg_error_set(err,FG_ERR_LIMIT,"video extraction command exceeds buffer");
        return FG_ERR_LIMIT;
    }
    FILE *pipe=popen(command,"r");
    if(!pipe){
        fg_error_set(err,FG_ERR_IO,"spawn ffmpeg: %s",strerror(errno));
        return FG_ERR_IO;
    }
    const uint64_t source_bytes=(uint64_t)source.width*source.height*3u;
    const uint64_t target_values=(uint64_t)target_width*target_height*3u;
    uint8_t *packed=malloc(source_bytes);
    uint8_t *planar=malloc(source_bytes);
    float *resample_source=malloc(source_bytes*sizeof(float));
    float *resample_target=malloc(target_values*sizeof(float));
    float *captured=malloc((uint64_t)max_frames*target_values*sizeof(float));
    bool failed=!packed||!planar||!resample_source||!resample_target||!captured;
    uint32_t captured_frames=0;
    while(!failed&&captured_frames<max_frames){
        const int got=video_read_frame(pipe,packed,source_bytes);
        if(got==0)break;
        if(got<0){
            fg_error_set(err,FG_ERR_FORMAT,"video extraction ended inside a frame");
            failed=true;
            break;
        }
        for(uint32_t y=0;y<source.height;y++)
            for(uint32_t x=0;x<source.width;x++)
                for(uint32_t c=0;c<3u;c++)
                    planar[((size_t)c*source.height+y)*source.width+x]=
                        packed[((size_t)y*source.width+x)*3u+c];
        fg_status step=fg_tower_normalize_image(planar,source.width,source.height,
                                                resample_source,err);
        if(step==FG_OK)step=fg_tower_resize_bicubic(resample_source,source.width,source.height,
                                                    resample_target,target_width,target_height,
                                                    err);
        if(step!=FG_OK){
            failed=true;
            break;
        }
        memcpy(captured+(size_t)captured_frames*target_values,resample_target,
               target_values*sizeof(float));
        captured_frames++;
    }
    const int exit_code=pclose(pipe);
    const int code=exit_code==-1?-1:WEXITSTATUS(exit_code);
    const bool stream_error=code!=0;
    if(!failed&&!captured_frames){
        fg_error_set(err,FG_ERR_FORMAT,"video extraction produced no frames (ffmpeg exit %d)",
                     code);
        failed=true;
    }
    if(!failed&&stream_error){
        fg_error_set(err,FG_ERR_IO,"video extraction failed (ffmpeg exit %d)",code);
        failed=true;
    }
    free(resample_target);
    free(resample_source);
    free(planar);
    free(packed);
    if(failed){
        free(captured);
        return err->code?err->code:FG_ERR_OOM;
    }
    *frames=captured;
    *frame_count=captured_frames;
    *frame_width=target_width;
    *frame_height=target_height;
    return FG_OK;
}

uint32_t fg_video_select_frames(uint32_t frame_count,double source_fps,double target_fps,
                                uint32_t max_frames,uint32_t *indices){
    if(!frame_count||!max_frames||!indices)return 0;
    if(!(target_fps>0.0))target_fps=FG_VIDEO_FPS_TARGET;
    if(!(source_fps>0.0)||source_fps<=target_fps){
        const uint32_t count=frame_count<max_frames?frame_count:max_frames;
        for(uint32_t i=0;i<count;i++)indices[i]=i;
        return count;
    }
    const double ratio=source_fps/target_fps;
    uint32_t selected=0;
    for(uint32_t step=0;selected<max_frames&&step<frame_count;step++){
        const long long index=llround((double)step*ratio);
        if(index>=(long long)frame_count)break;
        if(selected&&indices[selected-1u]>=(uint32_t)index)continue;
        indices[selected++]=(uint32_t)index;
    }
    return selected;
}

static fg_status video_frame_resize(const uint8_t *bytes,size_t length,uint32_t target_width,
                                    uint32_t target_height,float *destination,fg_error *err){
    uint8_t *rgb=NULL;
    uint32_t width=0,height=0;
    fg_status status=fg_tower_image_decode(bytes,(uint64_t)length,&rgb,&width,&height,err);
    if(status!=FG_OK)return status;
    const uint64_t source_values=(uint64_t)width*height*3u;
    float *normalized=malloc(source_values*sizeof(float));
    if(!normalized){
        free(rgb);
        fg_error_set(err,FG_ERR_OOM,"allocate video frame normalization");
        return FG_ERR_OOM;
    }
    status=fg_tower_normalize_image(rgb,width,height,normalized,err);
    free(rgb);
    if(status==FG_OK)status=fg_tower_resize_bicubic(normalized,width,height,destination,
                                                    target_width,target_height,err);
    free(normalized);
    return status;
}

fg_status fg_video_frames_preprocess(const uint8_t *const *frames,const size_t *lengths,
                                     uint32_t frame_count,const uint32_t *indices,
                                     uint32_t selected_count,const fg_video_options *options,
                                     float **tokens,uint32_t *pair_count,
                                     fg_tower_geometry *geometry,fg_error *err){
    if(!frames||!lengths||!frame_count||!indices||!selected_count||!tokens||!pair_count||
       !geometry){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video frame preprocessing request");
        return FG_ERR_ARGUMENT;
    }
    *tokens=NULL;
    *pair_count=0;
    if(selected_count>frame_count||selected_count>FG_VIDEO_MAX_FRAMES){
        fg_error_set(err,FG_ERR_LIMIT,"video frame selection count is out of range");
        return FG_ERR_LIMIT;
    }
    for(uint32_t i=0;i<selected_count;i++)
        if(indices[i]>=frame_count){
            fg_error_set(err,FG_ERR_LIMIT,"video frame selection index is out of range");
            return FG_ERR_LIMIT;
        }
    fg_video_options defaults;
    fg_video_options_defaults(&defaults);
    const fg_video_options *resolved=options?options:&defaults;
    uint64_t max_pixels=resolved->max_pixels?resolved->max_pixels:FG_VIDEO_MAX_PIXELS;
    if(max_pixels>FG_TOWER_VIDEO_MAX_PIXELS)max_pixels=FG_TOWER_VIDEO_MAX_PIXELS;
    uint8_t *probe_rgb=NULL;
    uint32_t source_width=0,source_height=0;
    fg_status status=fg_tower_image_decode(frames[indices[0]],(uint64_t)lengths[indices[0]],
                                           &probe_rgb,&source_width,&source_height,err);
    if(status!=FG_OK)return status;
    free(probe_rgb);
    uint32_t target_width=0,target_height=0;
    status=fg_tower_smart_resize(source_width,source_height,FG_TOWER_ALIGN,
                                 FG_TOWER_VIDEO_MIN_PIXELS,max_pixels,
                                 &target_width,&target_height,err);
    if(status!=FG_OK)return status;
    const uint64_t frame_values=(uint64_t)target_width*target_height*3u;
    if(frame_values>(uint64_t)SIZE_MAX/sizeof(float)/selected_count){
        fg_error_set(err,FG_ERR_LIMIT,"video frame buffer budget overflows");
        return FG_ERR_LIMIT;
    }
    float *resized=malloc((size_t)(frame_values*selected_count)*sizeof(float));
    if(!resized){
        fg_error_set(err,FG_ERR_OOM,"allocate video frames");
        return FG_ERR_OOM;
    }
    for(uint32_t i=0;status==FG_OK&&i<selected_count;i++)
        status=video_frame_resize(frames[indices[i]],lengths[indices[i]],target_width,
                                  target_height,resized+(size_t)i*frame_values,err);
    if(status==FG_OK)
        status=fg_video_superframes(resized,selected_count,target_width,target_height,tokens,
                                    pair_count,geometry,err);
    free(resized);
    return status;
}

uint64_t fg_video_token_count(uint32_t frame_count,uint32_t tokens_per_pair){
    if(!frame_count||!tokens_per_pair)return 0;
    return (((uint64_t)frame_count+1u)/2u)*(uint64_t)tokens_per_pair;
}

uint32_t fg_video_positions(uint32_t *positions,uint32_t token_count,uint32_t grid_width,
                            uint32_t grid_height,uint32_t temporal_groups,
                            uint32_t position){
    if(!positions||!token_count||!grid_width||!grid_height||!temporal_groups)return position;
    const uint32_t step=grid_width>grid_height?grid_width:grid_height;
    const uint32_t group_tokens=grid_width*grid_height;
    for(uint32_t token=0;token<token_count;token++){
        const uint32_t group=token/group_tokens;
        const uint32_t local=token%group_tokens;
        const uint32_t base=position+group*step;
        positions[(size_t)token*3u]=base;
        positions[(size_t)token*3u+1u]=base+local/grid_width;
        positions[(size_t)token*3u+2u]=base+local%grid_width;
    }
    return position+step*temporal_groups;
}

fg_status fg_video_superframes(const float *frames,uint32_t frame_count,
                               uint32_t width,uint32_t height,float **tokens,
                               uint32_t *pair_count,fg_tower_geometry *geometry,
                               fg_error *err){
    if(!frames||!frame_count||!tokens||!pair_count||!geometry){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video super-frame request");
        return FG_ERR_ARGUMENT;
    }
    *tokens=NULL;
    fg_status status=fg_tower_image_geometry(width,height,geometry,err);
    if(status!=FG_OK)return status;
    const uint32_t pairs=(frame_count+1u)/2u;
    const uint64_t pair_values=(uint64_t)geometry->tokens*FG_TOWER_TOKEN_VALUES;
    if(pair_values>(uint64_t)SIZE_MAX/sizeof(float)/pairs){
        fg_error_set(err,FG_ERR_LIMIT,"video super-frame token budget overflows");
        return FG_ERR_LIMIT;
    }
    float *output=malloc((size_t)(pair_values*pairs)*sizeof(float));
    float *scratch=malloc((size_t)pair_values*sizeof(float));
    if(!output||!scratch){
        free(scratch);
        free(output);
        fg_error_set(err,FG_ERR_OOM,"allocate video super-frame buffers");
        return FG_ERR_OOM;
    }
    const uint64_t frame_values=(uint64_t)width*height*3u;
    for(uint32_t pair=0;pair<pairs&&status==FG_OK;pair++){
        const uint32_t first=2u*pair;
        uint32_t second=first+1u;
        if(second>=frame_count)second=first;
        float *destination=output+(size_t)pair*pair_values;
        status=fg_tower_patchify(frames+(size_t)first*frame_values,width,height,destination,
                                 geometry,err);
        if(status==FG_OK)status=fg_tower_patchify(frames+(size_t)second*frame_values,width,
                                                  height,scratch,geometry,err);
        for(uint32_t token=0;status==FG_OK&&token<geometry->tokens;token++)
            memcpy(destination+(size_t)token*FG_TOWER_TOKEN_VALUES+FG_TOWER_PATCH_VALUES,
                   scratch+(size_t)token*FG_TOWER_TOKEN_VALUES+FG_TOWER_PATCH_VALUES,
                   FG_TOWER_PATCH_VALUES*sizeof(float));
    }
    free(scratch);
    if(status!=FG_OK){
        free(output);
        return status;
    }
    *tokens=output;
    *pair_count=pairs;
    return FG_OK;
}

static fg_status video_clip_tokens(const char *tower_dir,float *tokens,uint32_t token_count,
                                   uint32_t pair_count,const fg_tower_geometry *geometry,
                                   uint32_t frame_count,const fg_video_options *options,
                                   fg_video_clip *clip,fg_tower_vk_stats *stats,fg_error *err){
    const uint64_t expanded=fg_video_token_count(frame_count,geometry->merged_tokens);
    const uint32_t max_tokens=options->max_tokens?options->max_tokens:FG_VIDEO_MAX_TOKENS;
    if(expanded>max_tokens){
        fg_error_set(err,FG_ERR_LIMIT,"video expands to %llu tokens (limit %u)",
                     (unsigned long long)expanded,max_tokens);
        return FG_ERR_LIMIT;
    }
    const uint64_t values=(uint64_t)pair_count*geometry->merged_tokens*FG_TOWER_OUT_HIDDEN;
    if(values>(uint64_t)SIZE_MAX/sizeof(float)){
        fg_error_set(err,FG_ERR_LIMIT,"video embedding budget overflows");
        return FG_ERR_LIMIT;
    }
    float *embeddings=malloc((size_t)values*sizeof(float));
    if(!embeddings){
        fg_error_set(err,FG_ERR_OOM,"allocate video embeddings");
        return FG_ERR_OOM;
    }
    fg_status status=fg_tower_vision_forward_tokens(tower_dir,tokens,token_count,pair_count,
                                                    geometry,&embeddings,stats,err);
    if(status!=FG_OK){
        free(embeddings);
        return status;
    }
    clip->embeddings=embeddings;
    clip->token_count=(uint32_t)expanded;
    clip->tokens_per_pair=geometry->merged_tokens;
    clip->grid_width=geometry->grid_width/FG_TOWER_MERGE;
    clip->grid_height=geometry->grid_height/FG_TOWER_MERGE;
    clip->pair_count=pair_count;
    clip->frame_count=frame_count;
    return FG_OK;
}

fg_status fg_video_frames_forward(const char *tower_dir,const uint8_t *const *frames,
                                  const size_t *lengths,uint32_t frame_count,double source_fps,
                                  const fg_video_options *options,fg_video_clip *clip,
                                  fg_tower_vk_stats *stats,fg_error *err){
    if(!tower_dir||!frames||!lengths||!frame_count||!clip){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video frame forward request");
        return FG_ERR_ARGUMENT;
    }
    memset(clip,0,sizeof(*clip));
    if(frame_count>FG_VIDEO_MAX_INPUT_FRAMES){
        fg_error_set(err,FG_ERR_LIMIT,"video frame input exceeds %u frames",
                     FG_VIDEO_MAX_INPUT_FRAMES);
        return FG_ERR_LIMIT;
    }
    if(!fg_video_available()){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "video input requires the tower temporal token entry, which this "
                     "deployment does not provide");
        return FG_ERR_UNAVAILABLE;
    }
    fg_video_options defaults;
    fg_video_options_defaults(&defaults);
    const fg_video_options *resolved=options?options:&defaults;
    uint32_t max_frames=resolved->max_frames?resolved->max_frames:FG_VIDEO_MAX_FRAMES;
    if(max_frames>FG_VIDEO_MAX_FRAMES)max_frames=FG_VIDEO_MAX_FRAMES;
    const double target_fps=resolved->fps>0.0?resolved->fps:FG_VIDEO_FPS_TARGET;
    uint32_t *indices=malloc((size_t)max_frames*sizeof(*indices));
    if(!indices){
        fg_error_set(err,FG_ERR_OOM,"allocate video frame selection");
        return FG_ERR_OOM;
    }
    const uint32_t selected=fg_video_select_frames(frame_count,source_fps,target_fps,max_frames,
                                                   indices);
    if(!selected){
        free(indices);
        fg_error_set(err,FG_ERR_LIMIT,"video frame selection produced no frames");
        return FG_ERR_LIMIT;
    }
    uint64_t total_bytes=0;
    for(uint32_t i=0;i<selected;i++){
        total_bytes+=lengths[indices[i]];
        if(total_bytes>FG_VIDEO_MAX_BYTES){
            free(indices);
            fg_error_set(err,FG_ERR_LIMIT,"video frame payload exceeds %llu bytes",
                         (unsigned long long)FG_VIDEO_MAX_BYTES);
            return FG_ERR_LIMIT;
        }
    }
    float *tokens=NULL;
    uint32_t pair_count=0;
    fg_tower_geometry geometry={0};
    fg_status status=fg_video_frames_preprocess(frames,lengths,frame_count,indices,selected,
                                                resolved,&tokens,&pair_count,&geometry,err);
    free(indices);
    if(status!=FG_OK)return status;
    status=video_clip_tokens(tower_dir,tokens,geometry.tokens,pair_count,&geometry,selected,
                             resolved,clip,stats,err);
    free(tokens);
    return status;
}

fg_status fg_video_forward(const char *tower_dir,const uint8_t *bytes,size_t length,
                           const fg_video_options *options,fg_video_clip *clip,
                           fg_tower_vk_stats *stats,fg_error *err){
    if(!tower_dir||!bytes||!length||!clip){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid video forward request");
        return FG_ERR_ARGUMENT;
    }
    memset(clip,0,sizeof(*clip));
    if(!fg_video_available()){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "video input requires the tower temporal token entry, which this "
                     "deployment does not provide");
        return FG_ERR_UNAVAILABLE;
    }
    if(length>FG_VIDEO_MAX_BYTES){
        fg_error_set(err,FG_ERR_LIMIT,"video payload exceeds %llu bytes",
                     (unsigned long long)FG_VIDEO_MAX_BYTES);
        return FG_ERR_LIMIT;
    }
    if(!fg_video_mp4_available(tower_dir)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "MP4 video input requires static ffmpeg and ffprobe binaries "
                     "(tools directory or PATH)");
        return FG_ERR_UNAVAILABLE;
    }
    fg_video_options defaults;
    fg_video_options_defaults(&defaults);
    const fg_video_options *resolved=options?options:&defaults;
    char path[1200];
    if(snprintf(path,sizeof(path),"%s/.fg-video-XXXXXX",tower_dir)>=(int)sizeof(path)){
        fg_error_set(err,FG_ERR_LIMIT,"video scratch path exceeds buffer");
        return FG_ERR_LIMIT;
    }
    const int fd=mkstemp(path);
    if(fd<0){
        fg_error_set(err,FG_ERR_IO,"create video scratch file: %s",strerror(errno));
        return FG_ERR_IO;
    }
    size_t written=0;
    bool write_failed=false;
    while(written<length){
        const ssize_t count=write(fd,bytes+written,length-written);
        if(count<0){
            if(errno==EINTR)continue;
            write_failed=true;
            break;
        }
        written+=(size_t)count;
    }
    close(fd);
    fg_status status=FG_OK;
    float *frames=NULL,*tokens=NULL;
    uint32_t frame_count=0,frame_width=0,frame_height=0,pair_count=0;
    fg_tower_geometry geometry={0};
    if(write_failed){
        fg_error_set(err,FG_ERR_IO,"write video scratch file: %s",strerror(errno));
        status=FG_ERR_IO;
    }
    if(status==FG_OK)
        status=fg_video_extract(path,tower_dir,resolved,&frames,&frame_count,&frame_width,
                                &frame_height,err);
    if(status==FG_OK)
        status=fg_video_superframes(frames,frame_count,frame_width,frame_height,&tokens,
                                    &pair_count,&geometry,err);
    free(frames);
    if(status==FG_OK)
        status=video_clip_tokens(tower_dir,tokens,geometry.tokens,pair_count,&geometry,
                                 frame_count,resolved,clip,stats,err);
    free(tokens);
    unlink(path);
    return status;
}

bool fg_video_available(void){
    return fg_tower_vision_forward_tokens!=NULL;
}

bool fg_video_mp4_available(const char *tool_dir){
    if(!fg_video_available())return false;
    char tool[1200];
    return video_tool_path(tool_dir,"ffmpeg",tool,sizeof(tool))&&
           video_tool_path(tool_dir,"ffprobe",tool,sizeof(tool));
}
