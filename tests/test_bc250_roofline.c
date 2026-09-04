#include "fg_bc250_roofline.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(condition) do{if(!(condition)){fprintf(stderr,"FAIL: %s:%d: %s\n",__FILE__,__LINE__,#condition);failures++;}}while(0)

static void test_accounting(void){
    uint64_t bytes=0,read_bytes=0,write_bytes=0,operations=0;
    CHECK(fg_bc250_q8_0_weight_bytes(2560u,320u,&bytes));
    CHECK(bytes==(uint64_t)(2560u/32u)*34u*320u);
    CHECK(!fg_bc250_q8_0_weight_bytes(2559u,320u,&bytes));
    CHECK(fg_bc250_linear_accounting(4096u,0,&read_bytes,&write_bytes,&operations));
    CHECK(read_bytes==4096u&&write_bytes==16u&&operations==1024u);
    CHECK(fg_bc250_linear_accounting(4096u,1,&read_bytes,&write_bytes,&operations));
    CHECK(read_bytes==4096u&&write_bytes==4096u&&operations==1024u);
    CHECK(!fg_bc250_linear_accounting(3u,0,&read_bytes,&write_bytes,&operations));
    uint64_t weight_bytes=(uint64_t)(2560u/32u)*34u*320u;
    CHECK(fg_bc250_q8_0_logical_traffic(2560u,320u,32u,0,
                                        &read_bytes,&write_bytes,&operations));
    CHECK(read_bytes==weight_bytes*32u+(uint64_t)2560u*32u*4u);
    CHECK(fg_bc250_q8_0_logical_traffic(2560u,320u,32u,1,
                                        &read_bytes,&write_bytes,&operations));
    CHECK(read_bytes==weight_bytes+(uint64_t)2560u*32u*4u);
    CHECK(write_bytes==(uint64_t)320u*32u*4u);
    CHECK(operations==UINT64_C(2)*2560u*320u*32u);
}

static void test_format(void){
    char output[2048];fg_bc250_roofline_record record={
        .benchmark="linear_traffic",.name="read\"case",.shape="64KiB",
        .access_pattern="coalesced",.batch=1u,.tokens=1u,.iterations=4u,
        .bytes_read=65536u,.bytes_written=1024u,.operations=16384u,
        .gpu_ms=2.5,.wall_ms=3.0,.derived_gbps=0.026624,
        .derived_flops=NAN,.device="AMD\\BC250"};
    int length=fg_bc250_roofline_record_format(output,sizeof(output),&record);
    CHECK(length>0);CHECK(strstr(output,"\"schema\":\"fg.bc250.roofline.v1\"")!=NULL);
    CHECK(strstr(output,"read\\\"case")!=NULL);CHECK(strstr(output,"AMD\\\\BC250")!=NULL);
    CHECK(strstr(output,"\"derived_flops\":null")!=NULL);CHECK(output[length-1]=='\n');
    CHECK(fg_bc250_roofline_record_format(output,8u,&record)==0);
}

int main(void){test_accounting();test_format();if(failures){fprintf(stderr,"%d roofline test(s) failed\n",failures);return 1;}puts("bc250 roofline CPU tests: PASS");return 0;}
