#include "fg_bc250_qsa_curve.h"
#include "fg_bc250_roofline.h"
#include "fg_q38_schema.h"
#include "fg_vk.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct qsa_buffers {
    fg_vk_tensor *record[2],*index[2],*query_select,*query_attention,*gate;
    fg_vk_tensor *positions,*norm,*scores[2],*ids[2],*output;
} qsa_buffers;

static double elapsed_ms(struct timespec start,struct timespec end){
    return (double)(end.tv_sec-start.tv_sec)*1e3+
           (double)(end.tv_nsec-start.tv_nsec)*1e-6;
}

static int make_tensor(fg_vk_context *vk,uint64_t bytes,fg_vk_tensor **out,
                       fg_error *err){
    return fg_vk_tensor_create(vk,bytes,out,err)==FG_OK;
}

static int fill_tensor(fg_vk_tensor *tensor,uint8_t value){
    void *mapped=fg_vk_tensor_map(tensor);
    if(!mapped)return 0;
    memset(mapped,value,(size_t)fg_vk_tensor_bytes(tensor));return 1;
}

static int create_buffers(fg_vk_context *vk,uint32_t query_tokens,
                          uint32_t candidate_count,qsa_buffers *b,fg_error *err){
    uint64_t query_select=(uint64_t)query_tokens*512u*4u;
    uint64_t query_attention=(uint64_t)query_tokens*6144u*4u;
    uint64_t output=(uint64_t)query_tokens*6144u*4u;
    uint64_t selected=(uint64_t)query_tokens*candidate_count*4u;
    memset(b,0,sizeof(*b));
    for(uint32_t side=0u;side<2u;side++){
        if(!make_tensor(vk,UINT64_C(131072)*FG_Q38_QSA_TOKEN_RECORD_BYTES,
                        &b->record[side],err)||
           !make_tensor(vk,UINT64_C(131072)*FG_Q38_QSA_INDEX_KEY_BYTES,
                        &b->index[side],err)||
           !make_tensor(vk,selected,&b->scores[side],err)||
           !make_tensor(vk,selected,&b->ids[side],err)||
           !fill_tensor(b->record[side],0u)||!fill_tensor(b->index[side],0u)||
           !fill_tensor(b->scores[side],0xffu)||!fill_tensor(b->ids[side],0xffu))return 0;
    }
    if(!make_tensor(vk,query_select,&b->query_select,err)||
       !make_tensor(vk,query_attention,&b->query_attention,err)||
       !make_tensor(vk,query_attention,&b->gate,err)||
       !make_tensor(vk,UINT64_C(262144)*12u,&b->positions,err)||
       !make_tensor(vk,128u*4u,&b->norm,err))return 0;
    if(!make_tensor(vk,output,&b->output,err)||!fill_tensor(b->query_select,0u)||
       !fill_tensor(b->query_attention,0u)||!fill_tensor(b->gate,0u)||
       !fill_tensor(b->positions,0u)||!fill_tensor(b->norm,0u)||
       !fill_tensor(b->output,0u))return 0;
    for(uint32_t i=0u;i<128u;i++)((float *)fg_vk_tensor_map(b->norm))[i]=1.0f;
    for(uint32_t i=0u;i<query_tokens*512u;i++)
        ((float *)fg_vk_tensor_map(b->query_select))[i]=0.01f;
    for(uint32_t i=0u;i<query_tokens*6144u;i++){
        ((float *)fg_vk_tensor_map(b->query_attention))[i]=0.01f;
        ((float *)fg_vk_tensor_map(b->gate))[i]=0.0f;
    }
    return 1;
}

static void destroy_buffers(qsa_buffers *b){
    if(!b)return;
    fg_vk_tensor_destroy(b->output);fg_vk_tensor_destroy(b->norm);
    fg_vk_tensor_destroy(b->positions);fg_vk_tensor_destroy(b->gate);
    fg_vk_tensor_destroy(b->query_attention);fg_vk_tensor_destroy(b->query_select);
    for(uint32_t side=0u;side<2u;side++){
        fg_vk_tensor_destroy(b->ids[side]);fg_vk_tensor_destroy(b->scores[side]);
        fg_vk_tensor_destroy(b->index[side]);fg_vk_tensor_destroy(b->record[side]);
    }
    memset(b,0,sizeof(*b));
}

static double profile_kernel(const fg_vk_profile *profile,const char *name){
    double result=0.0;
    for(uint32_t i=0u;i<profile->kernel_count;i++)
        if(strcmp(profile->kernels[i].name,name)==0)result+=profile->kernels[i].gpu_ms;
    return result;
}

static fg_status dispatch_curve(fg_vk_context *vk,qsa_buffers *b,
                                const fg_bc250_qsa_geometry *geometry,
                                int select,int attention,int profile_enabled,
                                fg_vk_profile *profile,double *wall,
                                uint32_t *final_side,fg_error *err){
    fg_status status=FG_OK;struct timespec start={0},end={0};
    if(profile_enabled)status=fg_vk_profile_begin(vk,err);
    if(status==FG_OK&&profile_enabled)
        status=fg_vk_profile_set_scope(vk,"bc250_qsa_curve",err);
    clock_gettime(CLOCK_MONOTONIC,&start);
    if(status==FG_OK)status=fg_vk_begin(vk,err);
    if(status==FG_OK&&select)status=fg_vk_qsa_resident_select(
        vk,b->scores[0],b->ids[0],b->scores[1],b->ids[1],b->query_select,
        b->index[0],b->index[1],b->norm,b->positions,geometry->visible_tokens-
        geometry->query_tokens,geometry->query_tokens,geometry->capacity,
        geometry->segment_capacity,final_side,err);
    if(status==FG_OK&&attention)status=fg_vk_qsa_resident_attention(
        vk,b->output,b->record[0],b->record[1],b->ids[*final_side],
        b->query_attention,b->gate,geometry->visible_tokens-geometry->query_tokens,
        geometry->query_tokens,geometry->capacity,geometry->segment_capacity,
        geometry->selected_stride,err);
    if(status==FG_OK)status=fg_vk_end(vk,err);
    else if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(status==FG_OK&&profile_enabled){
        status=fg_vk_profile_end(vk,profile,err);
        if(status==FG_OK)*wall=elapsed_ms(start,end);
    }
    return status;
}

static int sanity_check(qsa_buffers *b,const fg_bc250_qsa_geometry *g,
                        uint32_t side){
    const uint32_t *ids=(const uint32_t *)fg_vk_tensor_const_map(b->ids[side]);
    const float *output=(const float *)fg_vk_tensor_const_map(b->output);
    if(!ids||!output)return 0;
    uint64_t selected=(uint64_t)g->query_tokens*g->selected_stride;
    for(uint64_t i=0u;i<selected;i++)
        if(ids[i]!=UINT32_MAX&&ids[i]>=g->complete_blocks)return 0;
    for(uint64_t i=0u;i<(uint64_t)g->query_tokens*6144u;i++)
        if(!isfinite(output[i]))return 0;
    return 1;
}

static int emit_record(const char *device,const char *component,
                       const fg_bc250_qsa_geometry *g,uint32_t iterations,
                       double gpu_ms,double wall_ms){
    fg_bc250_roofline_record record={
        .schema=FG_BC250_QSA_CURVE_SCHEMA,
        .benchmark="qsa_context_curve",.name="qwen3.8_resident_qsa",
        .shape="native_qsa",.access_pattern="synthetic_resident",
        .batch=g->query_tokens,.tokens=g->visible_tokens,.iterations=iterations,
        .bytes_read=UINT64_MAX,.bytes_written=UINT64_MAX,.operations=UINT64_MAX,
        .gpu_ms=gpu_ms,.wall_ms=wall_ms,.derived_gbps=NAN,.derived_flops=NAN,
        .device=device,.component=component,.visible_tokens=g->visible_tokens,
        .query_tokens=g->query_tokens,.complete_blocks=g->complete_blocks,
        .selector_groups=g->selector_groups,.candidate_count=g->candidate_count,
        .merge_passes=g->merge_passes,.selected_stride=g->selected_stride,
        .capacity=g->capacity,.segment_capacity=g->segment_capacity,
        .scratch_bytes=g->scratch_bytes};
    char line[4096];int length=fg_bc250_roofline_record_format(line,sizeof(line),&record);
    return length>0&&fwrite(line,1u,(size_t)length,stdout)==(size_t)length;
}

static int run_case(fg_vk_context *vk,const char *device,uint32_t visible,
                    uint32_t query_tokens,uint32_t iterations,fg_error *err){
    fg_bc250_qsa_geometry geometry={0};
    if(!fg_bc250_qsa_geometry_make(visible,query_tokens,FG_BC250_QSA_CAPACITY,
                                   FG_BC250_QSA_SEGMENT_CAPACITY,&geometry))return 0;
    qsa_buffers buffers={0};
    if(!create_buffers(vk,query_tokens,geometry.candidate_count,&buffers,err)){
        destroy_buffers(&buffers);return 0;
    }
    uint32_t side=0u;fg_status status=dispatch_curve(vk,&buffers,&geometry,1,1,0,
                                                       &(fg_vk_profile){0},
                                                       &(double){0.0},&side,err);
    if(status==FG_OK&&!sanity_check(&buffers,&geometry,side)){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA curve synthetic output sanity check failed");
        status=FG_ERR_MISMATCH;
    }
    fg_vk_profile profile={0};double wall=0.0;
    if(status==FG_OK)for(uint32_t i=0u;i<iterations;i++){
        status=dispatch_curve(vk,&buffers,&geometry,1,0,1,&profile,&wall,&side,err);
        if(status!=FG_OK)break;
        double scan=profile_kernel(&profile,"fg_qsa_resident_select.spv");
        double merge=profile_kernel(&profile,"fg_qsa_resident_topk_merge.spv");
        if(!emit_record(device,"selection_scan",&geometry,1u,scan,NAN)||
           !emit_record(device,"hierarchical_merge",&geometry,1u,merge,NAN)){
            status=FG_ERR_IO;break;
        }
    }
    if(status==FG_OK){
        for(uint32_t i=0u;i<iterations;i++){
            status=dispatch_curve(vk,&buffers,&geometry,0,1,1,&profile,&wall,&side,err);
            if(status!=FG_OK)break;
            if(!emit_record(device,"fixed_budget_attention",&geometry,1u,
                            profile_kernel(&profile,"fg_qsa_resident_attention.spv"),wall)){
                status=FG_ERR_IO;break;
            }
        }
    }
    if(status==FG_OK){
        for(uint32_t i=0u;i<iterations;i++){
            status=dispatch_curve(vk,&buffers,&geometry,1,1,1,&profile,&wall,&side,err);
            if(status!=FG_OK)break;
            if(!emit_record(device,"selector_plus_attention",&geometry,1u,
                            profile.kernel_ms,wall)){
                status=FG_ERR_IO;break;
            }
            if(!sanity_check(&buffers,&geometry,side)){
                fg_error_set(err,FG_ERR_MISMATCH,"QSA curve output became non-finite");
                status=FG_ERR_MISMATCH;break;
            }
        }
    }
    destroy_buffers(&buffers);
    return status==FG_OK;
}

int main(void){
    fg_error error={0};fg_vk_context *vk=NULL;
    fg_status status=fg_vk_open(&vk,&error);
    if(status!=FG_OK){fprintf(stderr,"BC250 QSA curve unavailable: %s\n",error.message);return 77;}
    const char *device=fg_vk_device_name(vk);int ok=1;
    const uint32_t decode_points[]={1u,511u,512u,2047u,2048u,2049u,2051u,2052u,
        4096u,8192u,16383u,16384u,16385u,16387u,16388u,32768u,65536u,131071u,131072u,
        131073u,196608u,261888u,262144u};
    for(uint32_t i=0u;ok&&i<sizeof(decode_points)/sizeof(decode_points[0]);i++)
        ok=run_case(vk,device,decode_points[i],1u,1u,&error);
    const uint32_t batch_points[]={4096u,65536u,131072u,262144u};
    const uint32_t batches[]={8u,32u,64u,128u};
    for(uint32_t p=0u;ok&&p<sizeof(batch_points)/sizeof(batch_points[0]);p++)
        for(uint32_t q=0u;ok&&q<sizeof(batches)/sizeof(batches[0]);q++)
            ok=run_case(vk,device,batch_points[p],batches[q],1u,&error);
    if(!ok)fprintf(stderr,"BC250 QSA curve failed: %s\n",error.message);
    fg_vk_close(vk);return ok?0:1;
}
