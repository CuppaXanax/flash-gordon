#ifndef FLASH_GORDON_PIPELINE_H
#define FLASH_GORDON_PIPELINE_H

#include "fg.h"
#include <string.h>

typedef struct fg_vk_tensor fg_vk_tensor;

typedef struct fg_pipeline_slot {
    uint32_t token_index;
    uint32_t position[3];
    int32_t last_committed_layer; /* -1 = embedding done, 0..47 = layer completed */
    fg_vk_tensor *hyper;
    fg_vk_tensor *ngram;
    bool output_pending;
    uint32_t next_token;
    float logit;
} fg_pipeline_slot;

typedef struct fg_pipeline_ring {
    fg_pipeline_slot slots[FG_PIPELINE_DEPTH];
    uint32_t head;  /* next slot to fill  */
    uint32_t tail;  /* next slot to retire */
    uint32_t count; /* in-flight count     */
} fg_pipeline_ring;

static inline void fg_pipeline_ring_init(fg_pipeline_ring *ring){memset(ring,0,sizeof(*ring));}

static inline fg_pipeline_slot *fg_pipeline_ring_push(fg_pipeline_ring *ring){
    if(ring->count>=FG_PIPELINE_DEPTH)return NULL;
    fg_pipeline_slot *slot=&ring->slots[ring->head];
    memset(slot,0,sizeof(*slot));slot->last_committed_layer=-1;
    ring->head=(ring->head+1u)%FG_PIPELINE_DEPTH;ring->count++;
    return slot;
}

static inline fg_pipeline_slot *fg_pipeline_ring_pop(fg_pipeline_ring *ring){
    if(!ring->count)return NULL;
    fg_pipeline_slot *slot=&ring->slots[ring->tail];
    ring->tail=(ring->tail+1u)%FG_PIPELINE_DEPTH;ring->count--;
    return slot;
}

static inline fg_pipeline_slot *fg_pipeline_ring_lookup(fg_pipeline_ring *ring,uint32_t token_index){
    for(uint32_t i=0;i<ring->count;i++){
        uint32_t idx=(ring->tail+i)%FG_PIPELINE_DEPTH;
        if(ring->slots[idx].token_index==token_index)return &ring->slots[idx];
    }
    return NULL;
}

#endif
