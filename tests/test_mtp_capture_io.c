#include "fg_mtp_capture.h"
#include "fg_protocol.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc,char **argv){
    assert(argc==2);
    double now_ms=1000000000.0;
    double deadline=fg_mtp_capture_deadline_ms(now_ms,2100u);
    assert(deadline-now_ms==2100000.0);
    assert(now_ms+2099999.0<deadline);
    assert(now_ms+2100000.0>=deadline);
    fg_error e={0};
    assert(fg_mtp_capture_configure(NULL,&e)==FG_OK);
    assert(!fg_mtp_capture_enabled());
    assert(fg_mtp_capture_routes(0,0,0,0,NULL,&e)==FG_OK);
    assert(fg_mtp_capture_verify_routes(0,0,0,0,false,NULL,&e)==FG_OK);
    assert(fg_mtp_capture_hyper(0,0,0,NULL,&e)==FG_OK);
    assert(!mkdir(argv[1],0700));
    assert(fg_mtp_capture_configure(argv[1],&e)==FG_OK);
    assert(fg_mtp_capture_enabled());
    float *values=malloc(2u*FG_HYPER_WIDTH*4u),*readback=malloc(3u*FG_HYPER_WIDTH*4u);
    assert(values&&readback);
    for(uint32_t i=0;i<2u*FG_HYPER_WIDTH;i++)values[i]=(float)i*0.01f;
    assert(fg_mtp_capture_hyper(42u,2u,1u,values,&e)==FG_OK);
    assert(fg_mtp_capture_hyper(42u,0u,2u,values,&e)==FG_OK);
    assert(fg_mtp_capture_hyper(42u,FG_NATIVE_CONTEXT,1u,values,&e)==FG_ERR_ARGUMENT);
    char path[1200];
    snprintf(path,sizeof(path),"%s/hc-42.f32",argv[1]);
    FILE *f=fopen(path,"rb");assert(f);
    assert(fread(readback,4u,3u*FG_HYPER_WIDTH,f)==3u*FG_HYPER_WIDTH);
    assert(fgetc(f)==EOF);assert(!fclose(f));
    assert(!memcmp(values,readback,2u*FG_HYPER_WIDTH*4u));
    assert(!memcmp(values,readback+2u*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u));
    assert(!unlink(path));
    snprintf(path,sizeof(path),"%s/hc-42.coverage",argv[1]);f=fopen(path,"rb");assert(f);
    for(uint32_t i=0;i<3u;i++)assert(fgetc(f)==1);
    assert(fgetc(f)==EOF);assert(!fclose(f));assert(!unlink(path));
    uint32_t ids[2u*FG_TOP_K],actual[3u*FG_TOP_K];
    for(uint32_t i=0;i<2u*FG_TOP_K;i++)ids[i]=i;
    assert(fg_mtp_capture_routes(42u,6u,2u,1u,ids,&e)==FG_OK);
    assert(fg_mtp_capture_routes(42u,6u,0u,2u,ids,&e)==FG_OK);
    assert(fg_mtp_capture_routes(42u,FG_LAYER_COUNT,0u,1u,ids,&e)==FG_ERR_ARGUMENT);
    ids[1]=ids[0];
    assert(fg_mtp_capture_routes(42u,6u,0u,1u,ids,&e)==FG_ERR_ARGUMENT);
    ids[1]=1u;
    snprintf(path,sizeof(path),"%s/routes-42-6.u32",argv[1]);f=fopen(path,"rb");assert(f);
    assert(fread(actual,4u,3u*FG_TOP_K,f)==3u*FG_TOP_K);
    assert(fgetc(f)==EOF);assert(!fclose(f));
    assert(!memcmp(actual,ids,sizeof(ids)));
    assert(!memcmp(actual+2u*FG_TOP_K,ids,FG_TOP_K*4u));assert(!unlink(path));
    snprintf(path,sizeof(path),"%s/routes-42-6.coverage",argv[1]);f=fopen(path,"rb");assert(f);
    for(uint32_t i=0;i<3u;i++)assert(fgetc(f)==1);
    assert(fgetc(f)==EOF);assert(!fclose(f));assert(!unlink(path));
    uint32_t verify[4u*FG_TOP_K];
    for(uint32_t i=0;i<4u*FG_TOP_K;i++)verify[i]=i;
    assert(fg_mtp_capture_verify_routes(42u,6u,2u,4u,false,verify,&e)==FG_OK);
    verify[FG_TOP_K]=100u;
    assert(fg_mtp_capture_verify_routes(42u,6u,2u,4u,true,verify,&e)==FG_OK);
    assert(fg_mtp_capture_verify_routes(42u,6u,2u,5u,false,verify,&e)==FG_ERR_ARGUMENT);
    verify[1]=verify[0];
    assert(fg_mtp_capture_verify_routes(42u,6u,2u,4u,false,verify,&e)==FG_ERR_ARGUMENT);
    snprintf(path,sizeof(path),"%s/verify-routes-42-6.jsonl",argv[1]);
    f=fopen(path,"rb");assert(f);
    char line[2048];
    assert(fgets(line,sizeof(line),f)&&strstr(line,"\"replay\":false")&&
        strstr(line,"\"first\":2,\"n\":4")&&strstr(line,"[10,11,12"));
    assert(fgets(line,sizeof(line),f)&&strstr(line,"\"replay\":true")&&strstr(line,"[100,11,12"));
    assert(!fgets(line,sizeof(line),f)&&!fclose(f));assert(!unlink(path));
    assert(fg_mtp_capture_configure(NULL,&e)==FG_OK);
    assert(!rmdir(argv[1]));free(values);free(readback);
    puts("capture I/O: disabled, out-of-order, exact offsets, coverage and bounds PASS");
    return 0;
}
