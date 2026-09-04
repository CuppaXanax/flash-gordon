#include "fg_bc250_qsa_curve.h"
#include "fg_bc250_roofline.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(value) do { if(!(value)){fprintf(stderr,"FAIL:%s:%d: %s\n",__FILE__,__LINE__,#value);failures++;} } while(0)

static void test_geometry_boundaries(void){
    const uint32_t points[]={1u,511u,512u,2047u,2048u,2049u,2051u,2052u,
        4096u,8192u,16383u,16384u,16385u,16387u,16388u,32768u,65536u,131071u,131072u,
        131073u,196608u,261888u,262144u};
    CHECK(fg_bc250_qsa_curve_boundaries_valid(points,sizeof(points)/sizeof(points[0]),
                                              FG_BC250_QSA_CAPACITY));
    for(size_t i=0;i<sizeof(points)/sizeof(points[0]);i++){
        fg_bc250_qsa_geometry g={0};
        CHECK(fg_bc250_qsa_geometry_make(points[i],1u,FG_BC250_QSA_CAPACITY,
                                         FG_BC250_QSA_SEGMENT_CAPACITY,&g));
        CHECK(g.visible_tokens==points[i]&&g.query_tokens==1u&&g.selected_stride==512u);
    }
    fg_bc250_qsa_geometry g={0};
    CHECK(fg_bc250_qsa_geometry_make(2048u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==512u&&g.selector_groups==1u&&g.merge_passes==0u);
    CHECK(fg_bc250_qsa_geometry_make(2049u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==512u&&g.selector_groups==1u);
    CHECK(fg_bc250_qsa_geometry_make(2052u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==513u&&g.selector_groups==1u&&
          g.candidate_count==512u);
    CHECK(fg_bc250_qsa_geometry_make(16384u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==4096u&&g.selector_groups==1u);
    CHECK(fg_bc250_qsa_geometry_make(16385u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==4096u&&g.selector_groups==1u);
    CHECK(fg_bc250_qsa_geometry_make(16388u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==4097u&&g.selector_groups==2u&&
          g.candidate_count==1024u&&g.merge_passes==1u);
    CHECK(fg_bc250_qsa_geometry_make(131072u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==32768u&&g.selector_groups==8u);
    CHECK(fg_bc250_qsa_geometry_make(131073u,1u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==32768u&&g.selector_groups==8u);
    CHECK(fg_bc250_qsa_geometry_make(262144u,128u,FG_BC250_QSA_CAPACITY,
                                     FG_BC250_QSA_SEGMENT_CAPACITY,&g)&&
          g.complete_blocks==65536u&&g.selector_groups==16u&&
          g.candidate_count==8192u&&g.merge_passes==2u&&
          g.scratch_bytes==UINT64_C(16777216));
}

static void test_ranges_and_overflow(void){
    fg_bc250_qsa_geometry g={0};
    CHECK(!fg_bc250_qsa_geometry_make(0u,1u,262144u,131072u,&g));
    CHECK(!fg_bc250_qsa_geometry_make(10u,0u,262144u,131072u,&g));
    CHECK(!fg_bc250_qsa_geometry_make(10u,11u,262144u,131072u,&g));
    CHECK(!fg_bc250_qsa_geometry_make(262145u,1u,262144u,131072u,&g));
    CHECK(!fg_bc250_qsa_geometry_make(10u,1u,10u,4u,&g));
    CHECK(!fg_bc250_qsa_curve_boundaries_valid(NULL,1u,262144u));
    CHECK(!fg_bc250_qsa_curve_boundaries_valid((uint32_t[]){2u,2u},2u,262144u));
    CHECK(!fg_bc250_qsa_curve_boundaries_valid((uint32_t[]){1u,262145u},2u,262144u));
}

static void test_json(void){
    char output[4096];
    fg_bc250_roofline_record record={
        .schema=FG_BC250_QSA_CURVE_SCHEMA,
        .benchmark="qsa_curve",.name="scan",.shape="native",
        .access_pattern="resident",.batch=1u,.tokens=1u,.iterations=3u,
        .bytes_read=UINT64_MAX,.bytes_written=UINT64_MAX,.operations=UINT64_MAX,
        .gpu_ms=0.0,.wall_ms=0.0,.derived_gbps=0.0,.derived_flops=0.0,
        .device="BC250\"x",.component="selector\\scan",
        .visible_tokens=2049u,.query_tokens=1u,.complete_blocks=512u,
        .selector_groups=1u,.candidate_count=512u,.merge_passes=0u,
        .selected_stride=512u,.capacity=262144u,.segment_capacity=131072u,
        .scratch_bytes=UINT64_MAX};
    int length=fg_bc250_roofline_record_format(output,sizeof(output),&record);
    CHECK(length>0&&strstr(output,"\"schema\":\"fg.bc250.qsa_curve.v1\"")!=NULL&&
          strstr(output,"\"bytes_read\":null")!=NULL&&
          strstr(output,"\"component\":\"selector\\\\scan\"")!=NULL&&
          strstr(output,"\"visible_tokens\":2049")!=NULL&&
          strstr(output,"\"scratch_bytes\":null")!=NULL);
    CHECK(fg_bc250_roofline_record_format(output,8u,&record)==0);
}

int main(void){
    test_geometry_boundaries();test_ranges_and_overflow();test_json();
    if(failures){fprintf(stderr,"%d QSA curve CPU test(s) failed\n",failures);return 1;}
    puts("BC250 QSA curve CPU tests: PASS");return 0;
}
