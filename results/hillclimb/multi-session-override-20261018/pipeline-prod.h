#ifndef FG_PIPELINE_H
#define FG_PIPELINE_H

#include "fg_fabric.h"
#include <ucontext.h>

#define FG_PIPELINE_SESSIONS 2u
#define FG_PIPELINE_STACK_BYTES (2u << 20u)

typedef void (*fg_pipeline_job)(void *context);
typedef enum fg_pipeline_phase {
    FG_PIPELINE_PREFILL,
    FG_PIPELINE_DECODE
} fg_pipeline_phase;
typedef struct fg_pipeline_options {
    uint32_t prefill_tokens,prefill_frames,prefill_slice_us;
    bool decode_priority;
} fg_pipeline_options;
typedef struct fg_pipeline_budget {
    uint32_t tokens,frames;
} fg_pipeline_budget;
typedef struct fg_pipeline_slot {
    ucontext_t context;
    void *stack;
    fg_pipeline_job job;
    void *argument;
    bool active,waiting;
    bool streaming;
    fg_pipeline_phase phase;
    uint32_t wanted_peer,*peer,*bytes,capacity;
    fg_fabric_class cls;
    fg_frame_header *header;
    void *payload;
    fg_status result;
    fg_error *error;
    uint64_t deadline;
} fg_pipeline_slot;

typedef struct fg_pipeline {
    ucontext_t caller;
    fg_pipeline_slot slots[FG_PIPELINE_SESSIONS];
    fg_fabric *fabric;
    uint32_t current;
    uint32_t next;
    fg_pipeline_options options;
    uint64_t prefill_ns_per_token;
} fg_pipeline;

fg_status fg_pipeline_options_load(fg_pipeline_options *options,fg_error *err);
fg_status fg_pipeline_init(fg_pipeline *pipeline,fg_fabric *fabric,fg_error *err);
void fg_pipeline_destroy(fg_pipeline *pipeline);
fg_status fg_pipeline_start(fg_pipeline *pipeline,uint32_t slot,
                             fg_pipeline_job job,void *argument,fg_error *err);
/* A turn never resumes more than one job. The caller admits requests between turns. */
bool fg_pipeline_turn(fg_pipeline *pipeline,uint32_t slot);
/* Pick a ready job before switching GPU state; return SESSIONS when none is ready. */
uint32_t fg_pipeline_pick(fg_pipeline *pipeline);
void fg_pipeline_set_phase(fg_pipeline_phase phase);
void fg_pipeline_set_streaming(bool streaming);
void fg_pipeline_checkpoint(void);
fg_pipeline_budget fg_pipeline_prefill_budget(uint32_t tokens,uint32_t frames);
void fg_pipeline_observe_prefill(uint32_t tokens,uint64_t service_ns);

#endif
