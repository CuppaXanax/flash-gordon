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
    fg_error e={0};
    assert(fg_mtp_capture_configure(NULL,&e)==FG_OK);
    assert(fg_mtp_capture_hyper(0,0,0,NULL,&e)==FG_OK);
    assert(!mkdir(argv[1],0700));
    assert(fg_mtp_capture_configure(argv[1],&e)==FG_OK);
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
    assert(fg_mtp_capture_configure(NULL,&e)==FG_OK);
    assert(!rmdir(argv[1]));free(values);free(readback);
    puts("capture I/O: disabled, out-of-order, exact offsets, coverage and bounds PASS");
    return 0;
}
