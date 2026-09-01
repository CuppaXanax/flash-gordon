#ifndef FLASH_GORDON_QSA_H
#define FLASH_GORDON_QSA_H

#include "fg_model.h"
#include "fg_protocol.h"

typedef struct fg_qsa_session fg_qsa_session;
typedef fg_status (*fg_qsa_page_fetch_fn)(void *opaque,uint32_t layer,
                                          const uint32_t *blocks,uint32_t block_count,
                                          uint8_t *records,fg_error *err);

/* Submits every open batch level before QSA reads mapped projection data. */
fg_status fg_qsa_submit_host_reads(fg_vk_context *vk,fg_error *err);

#define FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY 131072u
#define FG_QSA_RESIDENT_MAX_SEGMENTS 2u
#define FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY FG_QSA_RESIDENT_SEGMENT_TOKEN_CAPACITY
#define FG_QSA_INDEX_MAX_SEGMENTS FG_QSA_RESIDENT_MAX_SEGMENTS
#define FG_QSA_TOPK_BLOCK_GROUP 4096u
#define FG_QSA_TOPK_CANDIDATES 512u
#define FG_QSA_SELECTED_TOKENS (FG_Q38_INDEX_BUDGET+FG_Q38_QSA_COMPRESS_RATIO-1u)

uint32_t fg_qsa_index_segment_count(uint32_t logical_context);
uint32_t fg_qsa_index_segment_first(uint32_t logical_context,uint32_t segment);
uint32_t fg_qsa_index_segment_tokens(uint32_t logical_context,uint32_t segment);
uint64_t fg_qsa_index_segment_bytes(uint32_t logical_context,uint32_t segment);
uint64_t fg_qsa_record_segment_bytes(uint32_t logical_context,uint32_t segment);
uint64_t fg_qsa_resident_layer_bytes(uint32_t logical_context);
bool fg_qsa_index_token_location(uint32_t logical_context,uint32_t token,
                                 uint32_t *segment,uint32_t *offset);
static inline uint32_t fg_qsa_resident_candidate_groups(uint32_t logical_context){
    if(!logical_context)return 0u;
    uint32_t blocks=logical_context/FG_Q38_QSA_COMPRESS_RATIO;
    if(!blocks)return 1u;
    return blocks/FG_QSA_TOPK_BLOCK_GROUP+
        (blocks%FG_QSA_TOPK_BLOCK_GROUP!=0u);
}
static inline uint64_t fg_qsa_resident_candidate_entries(
    uint32_t logical_context,uint32_t batch_size){
    return (uint64_t)fg_qsa_resident_candidate_groups(logical_context)*
        FG_QSA_TOPK_CANDIDATES*batch_size;
}

static inline uint64_t fg_qsa_attention_scratch_bytes(uint32_t batch_size){
    if(!batch_size||batch_size>FG_PREFILL_MAX_TOKENS)return UINT64_MAX;
    uint64_t tokens=batch_size;
    uint64_t f32_values=12288u+512u+512u+6144u+6144u+512u+512u+
                        128u+512u+6144u+2560u;
    return tokens*f32_values*4u+
           tokens*(FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+
                   FG_Q38_QSA_INDEX_KEY_BYTES);
}

static inline uint64_t fg_gdn_pipeline_prefill_scratch_bytes(uint32_t batch_size){
    if(!batch_size||batch_size>FG_VK_GDN_PIPELINE_PREFILL_MAX_TOKENS)
        return UINT64_MAX;
    return (uint64_t)batch_size*(10240u+10240u+6144u+48u+48u+6144u+2560u)*4u;
}

static inline uint64_t fg_qsa_gdn_scratch_bytes(uint32_t batch_size){
    if(!batch_size||batch_size>FG_PREFILL_MAX_TOKENS)return UINT64_MAX;
    return (uint64_t)batch_size*(10240u+10240u+6144u+48u+48u+6144u+2560u)*4u;
}

static inline uint64_t fg_qsa_ple_scratch_bytes(uint32_t batch_size){
    if(!batch_size||batch_size>FG_PREFILL_MAX_TOKENS)return UINT64_MAX;
    return (uint64_t)batch_size*(10240u*7u+2560u)*4u;
}

static inline uint64_t fg_qsa_attention_family_scratch_bytes(uint32_t batch_size){
    uint64_t qsa=fg_qsa_attention_scratch_bytes(batch_size);
    uint64_t gdn=fg_qsa_gdn_scratch_bytes(batch_size);
    uint64_t ple=fg_qsa_ple_scratch_bytes(batch_size);
    if(qsa==UINT64_MAX||gdn==UINT64_MAX||ple==UINT64_MAX)return UINT64_MAX;
    return qsa>gdn?(qsa>ple?qsa:ple):(gdn>ple?gdn:ple);
}

static inline uint64_t fg_qsa_selection_scratch_bytes(uint32_t logical_context,
                                                      uint32_t batch_size){
    if(!logical_context||!batch_size||
       batch_size>FG_PREFILL_MAX_TOKENS)return UINT64_MAX;
    uint64_t blocks=((uint64_t)logical_context+FG_Q38_QSA_COMPRESS_RATIO-1u)/
                    FG_Q38_QSA_COMPRESS_RATIO;
    uint64_t score_bytes=blocks*sizeof(uint32_t);
    uint64_t base=fg_align_up_u64(fg_qsa_attention_scratch_bytes(batch_size),
                                  FG_ALIGNMENT);
    uint64_t offset=base;
    for(uint32_t i=0;i<4u;i++)offset=fg_align_up_u64(offset+score_bytes,FG_ALIGNMENT);
    offset=fg_align_up_u64(offset+
                           (uint64_t)FG_QSA_SELECTED_TOKENS*
                               FG_Q38_QSA_TOKEN_RECORD_BYTES,FG_ALIGNMENT);
    return offset-base;
}

static inline uint64_t fg_qsa_resident_selection_scratch_bytes(
    uint32_t logical_context,uint32_t batch_size){
    if(!logical_context||!batch_size||
       batch_size>FG_PREFILL_MAX_TOKENS)return UINT64_MAX;
    uint64_t entries=fg_qsa_resident_candidate_entries(logical_context,batch_size);
    if(!entries||entries>UINT64_MAX/(4u*sizeof(uint32_t)))return UINT64_MAX;
    uint64_t base=fg_align_up_u64(fg_qsa_attention_scratch_bytes(batch_size),
                                  FG_ALIGNMENT);
    uint64_t offset=base,bytes=entries*sizeof(uint32_t);
    for(uint32_t i=0;i<4u;i++)offset=fg_align_up_u64(offset+bytes,FG_ALIGNMENT);
    return offset-base;
}

fg_status fg_qsa_session_open(fg_qsa_session **out,fg_model *model,const char *state_path,
                              bool create,fg_error *err);
fg_status fg_qsa_session_open_decode(fg_qsa_session **out,fg_model *model,const char *state_path,
                                     uint32_t resident_tokens,uint32_t batch_size,fg_error *err);
fg_status fg_qsa_session_open_mirror(fg_qsa_session **out,fg_model *model,
                                     uint32_t logical_context,uint32_t hot_tokens,
                                     uint32_t cache_pages,uint32_t batch_size,
                                     fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,
                                     fg_error *err);
fg_status fg_qsa_session_open_mirror_with_scratch(
    fg_qsa_session **out,fg_model *model,uint32_t logical_context,uint32_t hot_tokens,
    uint32_t cache_pages,uint32_t batch_size,fg_vk_tensor *scratch,
    fg_qsa_page_fetch_fn fetch_pages,void *fetch_opaque,fg_error *err);
fg_status fg_qsa_session_open_resident(fg_qsa_session **out,fg_model *model,
                                       uint32_t batch_size,fg_vk_tensor *scratch,
                                       fg_error *err);
void fg_qsa_session_close(fg_qsa_session *session);
fg_status fg_qsa_session_reset(fg_qsa_session *session,fg_error *err);
fg_status fg_qsa_session_checkpoint(fg_qsa_session *session,fg_error *err);
uint32_t fg_qsa_session_tokens(const fg_qsa_session *session,uint32_t layer);
void fg_qsa_session_set_tokens(fg_qsa_session *session,uint32_t tokens);
uint64_t fg_qsa_session_host_bytes(const fg_qsa_session *session);
fg_status fg_qsa_session_decode(fg_qsa_session *session,uint32_t layer,uint32_t token_index,
                                const uint32_t position[3],const fg_vk_tensor *hidden,
                                fg_vk_tensor **output,fg_error *err);
/* Records resident T=1 QSA into an already-active pipeline stage batch. */
fg_status fg_qsa_session_decode_pipeline(fg_qsa_session *session,uint32_t layer,
                                        uint32_t token_index,
                                        const uint32_t position[3],
                                        const fg_vk_tensor *hidden,
                                        fg_vk_tensor **output,fg_error *err);
fg_status fg_qsa_session_prefill(fg_qsa_session *session,uint32_t layer,uint32_t first_token,
                                 const uint32_t *positions,uint32_t token_count,
                                 const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err);
fg_status fg_qsa_session_page_records(const fg_qsa_session *session,uint32_t layer,
                                      uint32_t block,const uint8_t **records,fg_error *err);
void fg_qsa_session_page_published(fg_qsa_session *session,uint32_t layer,uint32_t block);

#endif
