#include "fg_ledger.h"

#include <stdio.h>
#include <stdint.h>

static unsigned failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

int main(void){
    uint64_t bytes=0;
    CHECK(fg_ledger_pipeline_boundary_bytes(1u,&bytes));
    CHECK(bytes==(uint64_t)FG_Q38_HYPER_WIDTH*4u);
    CHECK(fg_ledger_pipeline_boundary_bytes(128u,&bytes));
    CHECK(bytes==(uint64_t)128u*FG_Q38_HYPER_WIDTH*4u);
    CHECK(fg_ledger_frame_bytes(bytes,&bytes));
    CHECK(bytes==(uint64_t)sizeof(fg_frame_header)+
          (uint64_t)128u*FG_Q38_HYPER_WIDTH*4u);
    CHECK(fg_ledger_pipeline_activation_payload_bytes(8u,&bytes));
    CHECK(bytes==FG_PIPELINE_ACTIVATION_HEADER_BYTES+8u*
          (FG_PIPELINE_POSITION_AXES*4u+FG_PIPELINE_BOUNDARY_WIDTH*4u));
    CHECK(fg_ledger_pipeline_result_payload_bytes(&bytes)&&
          bytes==FG_PIPELINE_RESULT_BYTES);
    CHECK(fg_ledger_decode_work_payload_bytes(&bytes)&&
          bytes==FG_DECODE_WORK_BYTES);
    CHECK(fg_ledger_expert_result_payload_bytes(1u,&bytes)&&
          bytes==FG_EXPERT_RESULT_SINGLE_BYTES);
    CHECK(fg_ledger_expert_result_payload_bytes(FG_TOP_K,&bytes)&&
          bytes==FG_EXPERT_RESULT_MAX_BYTES);
    CHECK(!fg_ledger_expert_result_payload_bytes(0u,&bytes));
    CHECK(!fg_ledger_expert_result_payload_bytes(FG_TOP_K+1u,&bytes));
    CHECK(fg_ledger_prefill_work_payload_bytes(8u,8u*FG_TOP_K,&bytes));
    CHECK(bytes==FG_PREFILL_WORK_HEADER_BYTES+8u*FG_Q8K_ACTIVATION_BYTES+
          8u*FG_TOP_K*FG_PREFILL_PAIR_BYTES);
    CHECK(fg_ledger_prefill_result_payload_bytes(8u*FG_TOP_K,&bytes));
    CHECK(bytes==FG_PREFILL_RESULT_HEADER_BYTES+8u*FG_TOP_K*
          FG_PREFILL_RESULT_PAIR_BYTES);
    CHECK(fg_ledger_pipeline_boundary_bytes(UINT32_MAX,&bytes));
    CHECK(bytes==(uint64_t)UINT32_MAX*FG_Q38_HYPER_WIDTH*4u);
    CHECK(!fg_ledger_pipeline_activation_payload_bytes(0u,&bytes));
    CHECK(!fg_ledger_add_u64(UINT64_MAX,1u,&bytes));
    CHECK(!fg_ledger_mul_u64(UINT64_MAX,2u,&bytes));
    if(failures){fprintf(stderr,"%u ledger checks failed\n",failures);return 1;}
    puts("fg ledger tests: ok");return 0;
}
