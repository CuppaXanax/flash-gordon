#ifndef FLASH_GORDON_LEDGER_H
#define FLASH_GORDON_LEDGER_H

/*
 * Wire-size accounting for the Qwen 3.8 pipeline.  These helpers deliberately
 * use the protocol constants instead of reproducing struct layouts, so an
 * analyzer cannot silently drift from the bytes sent by the runtime.
 */
#include "fg_protocol.h"

#include <stdint.h>
#include <stddef.h>

static inline bool fg_ledger_mul_u64(uint64_t a,uint64_t b,uint64_t *out){
    if(!out||(b&&a>UINT64_MAX/b))return false;
    *out=a*b;return true;
}

static inline bool fg_ledger_add_u64(uint64_t a,uint64_t b,uint64_t *out){
    if(!out||a>UINT64_MAX-b)return false;
    *out=a+b;return true;
}

static inline bool fg_ledger_pipeline_boundary_bytes(uint32_t tokens,
                                                     uint64_t *bytes){
    uint64_t values=0;
    return fg_ledger_mul_u64(tokens,(uint64_t)FG_PIPELINE_BOUNDARY_WIDTH,
                             &values)&&
        fg_ledger_mul_u64(values,FG_PIPELINE_BOUNDARY_FP32_BYTES,bytes);
}

static inline bool fg_ledger_frame_bytes(uint64_t payload,uint64_t *bytes){
    return payload<=UINT64_MAX-(uint64_t)sizeof(fg_frame_header)&&
        fg_ledger_add_u64((uint64_t)sizeof(fg_frame_header),payload,bytes);
}

static inline bool fg_ledger_pipeline_activation_payload_bytes(
    uint32_t tokens,uint64_t *bytes){
    uint64_t positions=0,boundary=0,total=0;
    if(!tokens||!fg_ledger_mul_u64(tokens,FG_PIPELINE_POSITION_AXES*4u,
                                   &positions)||
       !fg_ledger_pipeline_boundary_bytes(tokens,&boundary)||
       !fg_ledger_add_u64(FG_PIPELINE_ACTIVATION_HEADER_BYTES,positions,
                          &total)||
       !fg_ledger_add_u64(total,boundary,bytes))return false;
    return true;
}

static inline bool fg_ledger_pipeline_result_payload_bytes(uint64_t *bytes){
    if(!bytes)return false;
    *bytes=FG_PIPELINE_RESULT_BYTES;
    return true;
}

static inline bool fg_ledger_decode_work_payload_bytes(uint64_t *bytes){
    if(!bytes)return false;
    *bytes=FG_DECODE_WORK_BYTES;
    return true;
}

static inline bool fg_ledger_expert_result_payload_bytes(uint32_t selected,
                                                          uint64_t *bytes){
    uint64_t entries=0;
    if(!bytes||!selected||selected>FG_TOP_K||
       !fg_ledger_mul_u64(selected,FG_EXPERT_RESULT_ENTRY_BYTES,&entries)||
       !fg_ledger_add_u64(8u,entries,bytes))return false;
    return true;
}

static inline bool fg_ledger_prefill_work_payload_bytes(uint32_t tokens,
                                                         uint32_t pairs,
                                                         uint64_t *bytes){
    uint64_t activation=0,wire_pairs=0,total=0;
    if(!bytes||!tokens||tokens>FG_PREFILL_MAX_TOKENS||
       pairs>FG_PREFILL_MAX_PAIRS||
       !fg_ledger_mul_u64(tokens,FG_Q8K_ACTIVATION_BYTES,&activation)||
       !fg_ledger_mul_u64(pairs,FG_PREFILL_PAIR_BYTES,&wire_pairs)||
       !fg_ledger_add_u64(FG_PREFILL_WORK_HEADER_BYTES,activation,&total)||
       !fg_ledger_add_u64(total,wire_pairs,bytes))return false;
    return true;
}

static inline bool fg_ledger_prefill_result_payload_bytes(uint32_t pairs,
                                                           uint64_t *bytes){
    uint64_t wire_pairs=0;
    if(!bytes||pairs>FG_PREFILL_MAX_PAIRS||
       !fg_ledger_mul_u64(pairs,FG_PREFILL_RESULT_PAIR_BYTES,&wire_pairs)||
       !fg_ledger_add_u64(FG_PREFILL_RESULT_HEADER_BYTES,wire_pairs,bytes))
        return false;
    return true;
}

#endif
