#include "fg_manifest.h"
#include "fg_model.h"
#include "fg_expert.h"
#include "fg_owner.h"
#include "fg_qsa.h"
#include "fg_qsa_cache.h"
#include "fg_qsa_state.h"
#include "fg_ngram.h"
#include "fg_quant.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *qsa_suffixes[]={
    "attn_k.weight","attn_k_norm.weight","attn_output.weight","attn_q.weight",
    "attn_q_norm.weight","attn_v.weight","indexer.k_norm.weight",
    "indexer.k_proj.weight","indexer.q_norm.weight","indexer.q_proj.weight"
};

static int write_rank(const char *path,const uint8_t *block,uint32_t blocks){
    FILE *file=fopen(path,"wb");if(!file)return 0;
    for(uint32_t i=0;i<blocks;i++)
        if(fwrite(block,1,FG_ALIGNMENT,file)!=FG_ALIGNMENT){fclose(file);return 0;}
    return fclose(file)==0;
}

static int add_common(fg_manifest *manifest,const char *name,uint32_t rank,uint64_t offset,
                      const uint8_t *block,fg_error *err){
    fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"%s",name);
    record.offset=offset;record.bytes=FG_ALIGNMENT;record.ggml_type=8u;record.dims=2u;
    record.shape[0]=32u;record.shape[1]=32u;record.rank=(uint8_t)rank;
    record.layer=strncmp(name,"blk.3.",6u)==0?3u:UINT16_MAX;
    record.expert=UINT16_MAX;record.kind=FG_TENSOR_COMMON;
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,block,FG_ALIGNMENT);
    fg_sha256_final(&hash,record.sha256);
    return fg_manifest_add_tensor(manifest,&record,err)==FG_OK;
}

static int add_owner_weight(FILE *file,fg_manifest *manifest,const char *name,
                            uint64_t *offset,const void *data,uint64_t bytes,
                            uint64_t input_width,uint64_t output_width,
                            uint8_t layout,fg_error *err){
    fg_tensor_record record={0};
    snprintf(record.name,sizeof(record.name),"%s",name);
    record.offset=*offset;record.bytes=bytes;record.ggml_type=8u;record.dims=2u;
    record.shape[0]=input_width;record.shape[1]=output_width;record.rank=0u;
    record.layer=0u;record.expert=UINT16_MAX;record.kind=FG_TENSOR_COMMON;
    record.layout=layout;
    fg_sha256 hash;fg_sha256_init(&hash);
    fg_sha256_update(&hash,data,(size_t)bytes);fg_sha256_final(&hash,record.sha256);
    if(fg_manifest_add_tensor(manifest,&record,err)!=FG_OK||
       fwrite(data,1,(size_t)bytes,file)!=(size_t)bytes)return 0;
    *offset+=fg_align_up_u64(bytes,FG_ALIGNMENT);
    return 1;
}

static int test_owner_gr_prefill_boundaries(void){
    const uint32_t max_tokens=128u,hidden=FG_HIDDEN_SIZE,groups=FG_GROUP_SIZE;
    const uint64_t norm_bytes=(uint64_t)hidden*groups*sizeof(float);
    const uint64_t down_bytes=fg_q8_0_cooked_matrix_bytes(10240u,320u);
    const uint64_t up_bytes=fg_q8_0_cooked_matrix_bytes(320u,10240u);
    const uint64_t inject_bytes=(uint64_t)hidden*groups*groups*sizeof(float);
    uint8_t *norm=calloc(1,(size_t)norm_bytes);
    uint8_t *down=calloc(1,(size_t)down_bytes);
    uint8_t *up=calloc(1,(size_t)up_bytes);
    uint8_t *inject=calloc(1,(size_t)inject_bytes);
    float *hyper=malloc((size_t)max_tokens*FG_HYPER_WIDTH*sizeof(float));
    float *block=calloc((size_t)max_tokens*hidden,sizeof(float));
    float *result_values=malloc((size_t)max_tokens*FG_HYPER_WIDTH*sizeof(float));
    if(!norm||!down||!up||!inject||!hyper||!block||!result_values){
        free(result_values);free(block);free(hyper);free(inject);free(up);free(down);
        free(norm);return 1;
    }
    for(uint32_t token=0;token<max_tokens;token++)
        for(uint32_t value=0;value<FG_HYPER_WIDTH;value++)
            hyper[(uint64_t)token*FG_HYPER_WIDTH+value]=
                0.001f*(float)(value%19u)+0.002f*(float)token;
    char directory[96],path[128];
    snprintf(directory,sizeof(directory),"test-owner-gr-prefill-%ld",(long)getpid());
    snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_error error={0};
    int ok=manifest&&mkdir(directory,0700)==0;
    FILE *file=ok?fopen(path,"wb"):NULL;
    if(ok&&!file)ok=0;
    uint64_t offset=0;
    if(ok)fg_manifest_init(manifest);
    if(ok)ok=add_owner_weight(file,manifest,"blk.0.hc_attn_norm.weight",
                               &offset,norm,norm_bytes,hidden,groups,
                               FG_TENSOR_LAYOUT_GGML,&error);
    if(ok)ok=add_owner_weight(file,manifest,"blk.0.hc_attn_down.weight",
                               &offset,down,down_bytes,10240u,320u,
                               FG_TENSOR_LAYOUT_Q8_0_COOKED,&error);
    if(ok)ok=add_owner_weight(file,manifest,"blk.0.hc_attn_up.weight",
                               &offset,up,up_bytes,320u,10240u,
                               FG_TENSOR_LAYOUT_Q8_0_COOKED,&error);
    if(ok)ok=add_owner_weight(file,manifest,"blk.0.hc_attn_inject.weight",
                               &offset,inject,inject_bytes,FG_HYPER_WIDTH,
                               groups,FG_TENSOR_LAYOUT_GGML,&error);
    if(file)ok=ok&&fclose(file)==0;
    fg_model *model=NULL;fg_owner_executor *owner=NULL;fg_vk_tensor *hidden_tensor=NULL;
    fg_vk_tensor *block_tensor=NULL;
    if(ok){
        manifest->prefill_microbatch=max_tokens;
        if(fg_model_open_coordinator(&model,manifest,directory,0u,&error)==
           FG_ERR_UNAVAILABLE)ok=77;
        else ok=model!=NULL;
    }
    if(ok==1)ok=fg_owner_executor_create(&owner,model,&error)==FG_OK;
    if(ok==1)ok=fg_vk_tensor_create(fg_model_vk(model),
                                    (uint64_t)max_tokens*FG_HYPER_WIDTH*4u,
                                    &hidden_tensor,&error)==FG_OK;
    if(ok==1)ok=fg_vk_tensor_create(fg_model_vk(model),
                                    (uint64_t)max_tokens*hidden*4u,
                                    &block_tensor,&error)==FG_OK;
    if(ok==1){
        ok=fg_vk_tensor_write(hidden_tensor,0,hyper,
                              (uint64_t)max_tokens*FG_HYPER_WIDTH*4u,
                              &error)==FG_OK;
        if(ok)memset(fg_vk_tensor_map(block_tensor),0,
                     (size_t)fg_vk_tensor_bytes(block_tensor));
    }
    fg_vk_tensor *prefill_input=fg_owner_prefill_input(owner);
    if(ok==1)ok=prefill_input!=NULL;
    const uint32_t counts[]={37u,1u,127u,128u};
    for(uint32_t shape=0;ok==1&&shape<sizeof(counts)/sizeof(counts[0]);shape++){
        uint32_t tokens=counts[shape];fg_vk_tensor *mixed=NULL;
        const fg_vk_tensor *residual=NULL;fg_vk_tensor *injection_output=NULL;
        fg_vk_tensor *output=NULL;
        ok=fg_owner_gr_read_batch(owner,0u,false,hidden_tensor,tokens,&mixed,
                                  &residual,&injection_output,&error)==FG_OK&&
           mixed&&residual==hidden_tensor&&injection_output&&
           fg_vk_tensor_bytes(injection_output)==
               (uint64_t)max_tokens*groups*4u;
        if(ok)ok=fg_owner_gr_write_batch(owner,residual,block_tensor,
                                         injection_output,tokens,&output,
                                         &error)==FG_OK&&output;
        if(ok)ok=fg_vk_tensor_read(output,0,result_values,
                                   (uint64_t)tokens*FG_HYPER_WIDTH*4u,
                                   &error)==FG_OK;
        for(uint64_t i=0;ok&&i<(uint64_t)tokens*FG_HYPER_WIDTH;i++)
            if(result_values[i]!=hyper[i])ok=0;
        if(ok){
            memset(fg_vk_tensor_map(block_tensor),0,
                   (size_t)fg_vk_tensor_bytes(block_tensor));
            ok=fg_vk_tensor_write(prefill_input,0,hyper,
                                  (uint64_t)max_tokens*FG_HYPER_WIDTH*4u,
                                  &error)==FG_OK;
        }
        mixed=NULL;residual=NULL;injection_output=NULL;output=NULL;
        if(ok)ok=fg_owner_gr_read_batch(owner,0u,false,prefill_input,tokens,
                                        &mixed,&residual,&injection_output,
                                        &error)==FG_OK&&mixed&&
               residual==prefill_input&&injection_output&&
               fg_vk_tensor_bytes(injection_output)==
                   (uint64_t)max_tokens*groups*4u;
        if(ok)ok=fg_owner_gr_write_batch(owner,residual,block_tensor,
                                         injection_output,tokens,&output,
                                         &error)==FG_OK&&output;
        if(ok)ok=fg_vk_tensor_read(output,0,result_values,
                                   (uint64_t)tokens*FG_HYPER_WIDTH*4u,
                                   &error)==FG_OK;
        for(uint64_t i=0;ok&&i<(uint64_t)tokens*FG_HYPER_WIDTH;i++)
            if(result_values[i]!=hyper[i]){
                fprintf(stderr,"prefill input alias mismatch at token=%u value=%u: "
                               "got=%g expected=%g\n",tokens,
                        (uint32_t)(i%FG_HYPER_WIDTH),result_values[i],hyper[i]);
                ok=0;
            }
    }
    fg_vk_tensor_destroy(block_tensor);fg_vk_tensor_destroy(hidden_tensor);
    fg_owner_executor_destroy(owner);fg_model_close(model);
    free(manifest);if(path[0])unlink(path);if(directory[0])rmdir(directory);
    free(result_values);free(block);free(hyper);free(inject);free(up);free(down);free(norm);
    int result_code=ok==77?77:ok?0:1;
    if(result_code==1)fprintf(stderr,"owner GR prefill boundary test failed: %s\n",
                         error.message);
    return result_code;
}

static fg_status unexpected_fetch(void *opaque,uint32_t layer,const uint32_t *blocks,
                                  uint32_t block_count,uint8_t *records,fg_error *err){
    (void)opaque;(void)layer;(void)blocks;(void)block_count;(void)records;
    fg_error_set(err,FG_ERR_MISMATCH,"unexpected cold fetch during allocation test");
    return FG_ERR_MISMATCH;
}

static int test_index_segment_geometry(void){
    uint32_t segment=UINT32_MAX,offset=UINT32_MAX;
    int ok=FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY==131072u&&
        fg_qsa_index_segment_count(131071u)==1u&&
        fg_qsa_index_segment_tokens(131071u,0u)==131071u&&
        fg_qsa_index_segment_count(131072u)==1u&&
        fg_qsa_index_segment_tokens(131072u,0u)==131072u&&
        fg_qsa_index_segment_count(262144u)==2u&&
        fg_qsa_index_segment_first(262144u,1u)==131072u&&
        fg_qsa_index_segment_tokens(262144u,0u)==131072u&&
        fg_qsa_index_segment_tokens(262144u,1u)==131072u&&
        fg_qsa_index_segment_tokens(262144u,2u)==0u&&
        fg_qsa_index_segment_bytes(262144u,0u)==
            (uint64_t)131072u*FG_Q38_QSA_INDEX_KEY_BYTES&&
        fg_qsa_index_segment_bytes(262144u,0u)==
            (uint64_t)17u*1024u*1024u;
    ok=ok&&fg_qsa_index_token_location(262144u,131071u,&segment,&offset)&&
        segment==0u&&offset==131071u;
    ok=ok&&fg_qsa_index_token_location(262144u,131072u,&segment,&offset)&&
        segment==1u&&offset==0u;
    ok=ok&&fg_qsa_index_token_location(262144u,262143u,&segment,&offset)&&
        segment==1u&&offset==131071u;
    ok=ok&&!fg_qsa_index_token_location(262144u,262144u,&segment,&offset);
    for(uint32_t token=131071u;ok&&token<=131072u;token++){
        ok=fg_qsa_index_token_location(262144u,token,&segment,&offset)&&
            fg_qsa_index_segment_first(262144u,segment)+offset==token;
    }
    uint64_t family=fg_qsa_attention_family_scratch_bytes(256u);
    uint64_t qsa=fg_qsa_attention_scratch_bytes(256u);
    uint64_t selection=fg_qsa_selection_scratch_bytes(262144u,256u);
    ok=ok&&selection>0u&&qsa+selection<family;
    ok=ok&&fg_qsa_attention_scratch_bytes(128u)==18572288u&&
        fg_qsa_gdn_scratch_bytes(128u)==18137088u&&
        fg_qsa_ple_scratch_bytes(128u)==38010880u&&
        fg_qsa_attention_family_scratch_bytes(128u)==38010880u;
    ok=ok&&FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES>=128u/FG_Q38_QSA_COMPRESS_RATIO&&
        FG_QSA_PAGE_APPEND_MAX_PAGES>=
            FG_QSA_OWNER_LAYER_COUNT*(128u/FG_Q38_QSA_COMPRESS_RATIO);
    uint32_t first_block=UINT32_MAX,block_count=UINT32_MAX;fg_error error={0};
    ok=ok&&fg_qsa_completed_page_range(0u,128u,&first_block,&block_count,&error)==
        FG_OK&&first_block==0u&&block_count==32u;
    return ok;
}

static int test_memory_ledger_arithmetic(void){
    const uint32_t tokens=FG_DEFAULT_MICROBATCH,logical=262144u;
    const uint64_t pairs=(uint64_t)tokens*FG_TOP_K;
    uint64_t owner_transient=(uint64_t)tokens*10240u*4u+
        (uint64_t)tokens*320u*4u+(uint64_t)8u*320u*4u+
        (uint64_t)tokens*320u*4u+(uint64_t)tokens*10240u*4u+
        (uint64_t)24u*4u*tokens*4u+(uint64_t)tokens*2560u*4u+
        (uint64_t)tokens*FG_GROUP_SIZE*4u+
        (uint64_t)tokens*FG_EXPERT_COUNT*4u+
        (uint64_t)tokens*640u*4u*3u+(uint64_t)tokens*FG_HIDDEN_SIZE*4u+
        (uint64_t)tokens*4u+(uint64_t)tokens*FG_HIDDEN_SIZE*4u;
    uint64_t qsa_aux=(uint64_t)((logical+3u)/4u)*4u*sizeof(uint32_t)+
        (uint64_t)FG_QSA_SELECTED_TOKENS*FG_Q38_QSA_TOKEN_RECORD_BYTES;
    uint64_t prefill_layer=(uint64_t)tokens*
        (FG_HYPER_WIDTH*4u+FG_NGRAM_EMBED_VALUES*4u);
    uint64_t ngram_vk=(uint64_t)tokens*FG_NGRAM_HEAD_COUNT*FG_NGRAM_ROW_BYTES+
        (uint64_t)tokens*FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH*4u;
    uint64_t vulkan_reclaim=owner_transient+qsa_aux+prefill_layer+
        FG_HYPER_WIDTH*sizeof(float)+ngram_vk;
    uint64_t current_positions=(uint64_t)tokens*3u*sizeof(uint32_t);
    uint64_t host_position_reclaim=(uint64_t)tokens*sizeof(uint32_t);
    uint32_t cache_pages=(uint32_t)((UINT64_C(16)<<20u)/
                                    FG_QSA_PAGE_RECORD_BYTES);
    uint64_t qsa_staging_cache=(uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*
        FG_QSA_PAGE_RECORD_BYTES+
        fg_qsa_page_cache_memory_bytes_for_pages(cache_pages);
    uint64_t prefill_wire=FG_PREFILL_RESULT_HEADER_BYTES+
        pairs*FG_PREFILL_RESULT_PAIR_BYTES;
    uint64_t prefill_work_wire=FG_PREFILL_WORK_HEADER_BYTES+
        (uint64_t)tokens*FG_Q8K_ACTIVATION_BYTES+
        (uint64_t)tokens*FG_TOP_K*FG_PREFILL_PAIR_BYTES;
    uint64_t transport=2u*(uint64_t)FG_QSA_PAGE_APPEND_MAX_PAGES*
        sizeof(fg_qsa_page)+2u*(uint64_t)FG_QSA_PAGE_FETCH_MAX_PAGES*
        sizeof(fg_qsa_page)+FG_QSA_PAGE_FETCH_MAX_BYTES+
        FG_QSA_PAGE_RESULT_MAX_BYTES+2u*FG_QSA_PAGE_APPEND_MAX_BYTES;
    uint64_t deferred=qsa_staging_cache+fg_ngram_cache_memory_bytes()+
        prefill_wire+prefill_work_wire+transport+ngram_vk;
    uint64_t record_cache=(uint64_t)cache_pages*FG_QSA_PAGE_RECORD_BYTES;
    return vulkan_reclaim==UINT64_C(51817084)&&
        current_positions==UINT64_C(3072)&&
        host_position_reclaim==UINT64_C(1024)&&
        FG_NGRAM_PREFILL_IO_BYTES==UINT64_C(524288)&&
        prefill_work_wire==UINT64_C(788496)&&
        record_cache<=(UINT64_C(16)<<20u)&&
        qsa_staging_cache<(uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*
            FG_QSA_PAGE_RECORD_BYTES+(UINT64_C(1)<<20u)&&
        deferred<(UINT64_C(52)<<20u);
}

int main(void){
    if(!test_index_segment_geometry()){
        fprintf(stderr,"QSA segmented index geometry: FAIL\n");
        return 1;
    }
    if(!test_memory_ledger_arithmetic()){
        fprintf(stderr,"QSA memory ledger arithmetic: FAIL\n");
        return 1;
    }
    int owner_prefill=test_owner_gr_prefill_boundaries();
    if(owner_prefill==1)return 1;
    if(owner_prefill==77)
        fprintf(stderr,"SKIP owner GR prefill boundary test: Vulkan unavailable\n");
    char directory[96],rank0_path[128],rank3_path[128];
    snprintf(directory,sizeof(directory),"test-qsa-model-load-%ld",(long)getpid());
    snprintf(rank0_path,sizeof(rank0_path),"%s/rank-00.fgw",directory);
    snprintf(rank3_path,sizeof(rank3_path),"%s/rank-03.fgw",directory);
    if(mkdir(directory,0700)!=0){perror("mkdir");return 1;}
    uint8_t *block=aligned_alloc(FG_ALIGNMENT,FG_ALIGNMENT);
    if(!block){rmdir(directory);return 1;}
    for(uint32_t i=0;i<FG_ALIGNMENT;i++)block[i]=(uint8_t)(i*19u+5u);
    uint32_t qsa_count=(uint32_t)(sizeof(qsa_suffixes)/sizeof(qsa_suffixes[0]));
    int ok=write_rank(rank0_path,block,1u)&&write_rank(rank3_path,block,qsa_count+1u);
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_error error={0};
    if(!manifest)ok=0;
    if(ok){
        fg_manifest_init(manifest);
        ok=add_common(manifest,"token_embd.weight",0u,0u,block,&error);
        for(uint32_t i=0;ok&&i<qsa_count;i++){
            char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"blk.3.%s",qsa_suffixes[i]);
            ok=add_common(manifest,name,3u,(uint64_t)i*FG_ALIGNMENT,block,&error);
        }
        if(ok)ok=add_common(manifest,"blk.3.hc_attn_norm.weight",3u,
                            (uint64_t)qsa_count*FG_ALIGNMENT,block,&error);
    }
    fg_model *coordinator=NULL,*owner=NULL;
    fg_expert_executor *expert_executor=NULL;
    fg_owner_executor *owner_executor=NULL;
    fg_status status=ok?fg_model_open_coordinator(&coordinator,manifest,directory,0u,&error):
                        FG_ERR_FORMAT;
    if(status==FG_ERR_UNAVAILABLE){
        fprintf(stderr,"SKIP QSA owner model placement: %s\n",error.message);
        free(manifest);free(block);unlink(rank3_path);unlink(rank0_path);rmdir(directory);
        return 77;
    }
    ok=ok&&status==FG_OK&&fg_model_tensor(coordinator,"token_embd.weight")&&
       fg_model_tensor(coordinator,"blk.3.hc_attn_norm.weight");
    for(uint32_t i=0;ok&&i<qsa_count;i++){
        char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"blk.3.%s",qsa_suffixes[i]);
        ok=fg_model_tensor(coordinator,name)!=NULL;
    }
    fg_vk_memory_stats memory_before={0},memory_after={0};
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_before);
    fg_qsa_session *mirror=NULL;
    if(ok)ok=fg_qsa_session_open_mirror(&mirror,coordinator,4u,4u,2u,1u,
                                         NULL,NULL,&error)==FG_OK;
    if(ok)ok=fg_qsa_session_tokens(mirror,3u)==0u&&
             fg_qsa_session_reset(mirror,&error)==FG_OK;
    fg_qsa_session_close(mirror);
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_after);
    if(ok)ok=memory_after.requested_live_bytes==memory_before.requested_live_bytes&&
             memory_after.allocated_live_bytes==memory_before.allocated_live_bytes&&
             memory_after.live_allocations==memory_before.live_allocations;
    fg_vk_tensor *shared_scratch=NULL;
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_before);
    if(ok)ok=fg_vk_tensor_create(fg_model_vk(coordinator),
                                 fg_qsa_attention_family_scratch_bytes(256u),
                                 &shared_scratch,&error)==FG_OK;
    fg_vk_memory_stats shared_open={0};
    if(ok)ok=fg_qsa_session_open_mirror_with_scratch(
        &mirror,coordinator,4u,4u,2u,1u,shared_scratch,NULL,NULL,&error)==FG_OK;
    if(ok){
        fg_vk_get_memory_stats(fg_model_vk(coordinator),&shared_open);
        ok=shared_open.requested_live_bytes==memory_before.requested_live_bytes+
            fg_qsa_attention_family_scratch_bytes(256u)+48u+6528u+
            2u*FG_QSA_PAGE_RECORD_BYTES;
    }
    fg_qsa_session_close(mirror);mirror=NULL;fg_vk_tensor_destroy(shared_scratch);
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_after);
    if(ok)ok=memory_after.requested_live_bytes==memory_before.requested_live_bytes&&
             memory_after.allocated_live_bytes==memory_before.allocated_live_bytes&&
             memory_after.live_allocations==memory_before.live_allocations;
    fg_vk_memory_stats executor_baseline={0};
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&executor_baseline);
    if(ok)ok=fg_expert_executor_create(&expert_executor,coordinator,&error)==FG_OK;
    if(ok)ok=fg_owner_executor_create(&owner_executor,coordinator,&error)==FG_OK;
    if(ok)ok=fg_owner_prefill_input(owner_executor)!=NULL;
    fg_owner_executor_destroy(owner_executor);owner_executor=NULL;
    fg_expert_executor_destroy(expert_executor);expert_executor=NULL;
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_after);
    if(ok)ok=memory_after.requested_live_bytes==executor_baseline.requested_live_bytes&&
             memory_after.allocated_live_bytes==executor_baseline.allocated_live_bytes&&
             memory_after.live_allocations==executor_baseline.live_allocations;
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_before);
    fg_vk_counters canary_before={0},canary_after={0};
    fg_vk_get_counters(fg_model_vk(coordinator),&canary_before);
    if(ok)ok=fg_qsa_session_open_mirror(&mirror,coordinator,262144u,8192u,2u,1u,
                                         unexpected_fetch,NULL,&error)==FG_OK;
    fg_vk_memory_stats memory_large={0};fg_vk_get_memory_stats(fg_model_vk(coordinator),
                                                               &memory_large);
    fg_vk_get_counters(fg_model_vk(coordinator),&canary_after);
    uint64_t expected_index=UINT64_C(262144)*12u*FG_Q38_QSA_INDEX_KEY_BYTES;
    if(ok)ok=memory_large.requested_live_bytes>=
             memory_before.requested_live_bytes+expected_index;
    if(ok)ok=memory_large.live_allocations>=
             memory_before.live_allocations+12u*FG_QSA_INDEX_MAX_SEGMENTS;
    if(ok)ok=canary_after.residency_canary_calls==
             canary_before.residency_canary_calls+12u*FG_QSA_INDEX_MAX_SEGMENTS;
    fg_qsa_session_close(mirror);mirror=NULL;
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_after);
    if(ok)ok=memory_after.requested_live_bytes==memory_before.requested_live_bytes&&
             memory_after.allocated_live_bytes==memory_before.allocated_live_bytes&&
             memory_after.live_allocations==memory_before.live_allocations;
    mirror=NULL;
    if(ok)ok=fg_qsa_session_open_mirror(&mirror,coordinator,4u,4u,UINT32_MAX,1u,
                                         NULL,NULL,&error)==FG_ERR_LIMIT&&!mirror;
    if(ok)ok=fg_model_open(&owner,manifest,directory,3u,&error)==FG_OK;
    ok=ok&&fg_model_tensor(owner,"blk.3.hc_attn_norm.weight");
    for(uint32_t i=0;ok&&i<qsa_count;i++){
        char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"blk.3.%s",qsa_suffixes[i]);
        ok=fg_model_tensor(owner,name)!=NULL;
    }
    char owner_state_path[128];snprintf(owner_state_path,sizeof(owner_state_path),
        "%s/owner-session.qsa",directory);unlink(owner_state_path);
    fg_qsa_session *file_session=NULL;struct stat owner_state_info={0};
    if(ok){
        status=fg_qsa_session_open(&file_session,owner,owner_state_path,true,&error);
        ok=status==FG_OK&&file_session&&
           manifest->session.logical_context_tokens<=
               FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY*FG_QSA_INDEX_MAX_SEGMENTS&&
           manifest->max_context>
               FG_QSA_INDEX_SEGMENT_TOKEN_CAPACITY*FG_QSA_INDEX_MAX_SEGMENTS&&
           stat(owner_state_path,&owner_state_info)==0&&
           (uint64_t)owner_state_info.st_size==
               fg_qsa_state_required_bytes(6u,manifest->session.logical_context_tokens)&&
           fg_qsa_session_tokens(file_session,3u)==0u;
    }
    fg_qsa_session_close(file_session);file_session=NULL;
    if(ok){
        status=fg_qsa_session_open(&file_session,owner,owner_state_path,false,&error);
        ok=status==FG_OK&&file_session&&fg_qsa_session_tokens(file_session,3u)==0u;
    }
    fg_qsa_session_close(file_session);file_session=NULL;unlink(owner_state_path);
    char failed_state_path[128];snprintf(failed_state_path,sizeof(failed_state_path),
        "%s/failed-session.qsa",directory);unlink(failed_state_path);
    fg_qsa_session *failed_session=NULL;fg_error create_error={0};
    if(ok){
        status=fg_qsa_session_open_decode(&failed_session,owner,failed_state_path,4u,
                                          UINT32_MAX,&create_error);
        ok=status!=FG_OK&&!failed_session&&access(failed_state_path,F_OK)!=0&&
           create_error.code==status&&create_error.message[0];
    }
    fg_qsa_session_close(failed_session);unlink(failed_state_path);
    fg_model_close(owner);fg_model_close(coordinator);free(manifest);free(block);
    unlink(rank3_path);unlink(rank0_path);rmdir(directory);
    if(!ok){fprintf(stderr,"QSA owner model placement failed: %s\n",error.message);return 1;}
    puts("QSA owner model placement: PASS");return 0;
}
