#include "fg_bc250_qsa_curve.h"

#include <limits.h>
#include <string.h>

int fg_bc250_qsa_geometry_make(uint32_t visible_tokens,uint32_t query_tokens,
                               uint32_t capacity,uint32_t segment_capacity,
                               fg_bc250_qsa_geometry *geometry){
    if(!geometry||!visible_tokens||!query_tokens||query_tokens>visible_tokens||
       visible_tokens>capacity||!capacity||!segment_capacity||
       (uint64_t)segment_capacity*2u<capacity)return 0;
    uint32_t blocks=visible_tokens/FG_BC250_QSA_COMPRESS_RATIO;
    if(blocks>UINT32_MAX-(FG_BC250_QSA_GROUP_BLOCKS-1u))return 0;
    uint32_t groups=blocks?((blocks+FG_BC250_QSA_GROUP_BLOCKS-1u)/
                            FG_BC250_QSA_GROUP_BLOCKS):1u;
    if(groups>UINT32_MAX/FG_BC250_QSA_SELECTED_STRIDE)return 0;
    uint32_t candidates=groups*FG_BC250_QSA_SELECTED_STRIDE;
    uint32_t count=candidates,passes=0u;
    while(count>FG_BC250_QSA_SELECTED_STRIDE){
        if(count>UINT32_MAX-(FG_BC250_QSA_GROUP_BLOCKS-1u))return 0;
        uint32_t merge_groups=(count+FG_BC250_QSA_GROUP_BLOCKS-1u)/
                               FG_BC250_QSA_GROUP_BLOCKS;
        if(!merge_groups||merge_groups>UINT32_MAX/
           FG_BC250_QSA_SELECTED_STRIDE)return 0;
        count=merge_groups*FG_BC250_QSA_SELECTED_STRIDE;
        if(passes==UINT32_MAX)return 0;
        passes++;
    }
    uint64_t entries=(uint64_t)query_tokens*candidates;
    if(entries>UINT64_MAX/(4u*sizeof(uint32_t)))return 0;
    memset(geometry,0,sizeof(*geometry));
    geometry->visible_tokens=visible_tokens;
    geometry->query_tokens=query_tokens;
    geometry->complete_blocks=blocks;
    geometry->selector_groups=groups;
    geometry->candidate_count=candidates;
    geometry->merge_passes=passes;
    geometry->selected_stride=FG_BC250_QSA_SELECTED_STRIDE;
    geometry->capacity=capacity;
    geometry->segment_capacity=segment_capacity;
    geometry->scratch_bytes=entries*4u*sizeof(uint32_t);
    return 1;
}

int fg_bc250_qsa_curve_boundaries_valid(const uint32_t *points,size_t count,
                                        uint32_t capacity){
    if(!points||!count)return 0;
    for(size_t i=0;i<count;i++)
        if(points[i]>capacity||(i&&points[i]<=points[i-1u]))return 0;
    return 1;
}
