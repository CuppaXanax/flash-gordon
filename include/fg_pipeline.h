#ifndef FG_PIPELINE_H
#define FG_PIPELINE_H

#include "fg_fabric.h"
#include <ucontext.h>

#define FG_PIPELINE_SESSIONS 2u
#define FG_PIPELINE_STACK_BYTES (2u << 20u)

typedef void (*fg_pipeline_job)(void *context);
typedef struct fg_pipeline_slot {
    ucontext_t context;
    void *stack;
    fg_pipeline_job job;
    void *argument;
    bool active,waiting;
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
} fg_pipeline;

fg_status fg_pipeline_init(fg_pipeline *pipeline,fg_fabric *fabric,fg_error *err);
void fg_pipeline_destroy(fg_pipeline *pipeline);
fg_status fg_pipeline_start(fg_pipeline *pipeline,uint32_t slot,
                             fg_pipeline_job job,void *argument,fg_error *err);
/* A turn never resumes more than one job. The caller admits requests between turns. */
bool fg_pipeline_turn(fg_pipeline *pipeline,uint32_t slot);

#endif
