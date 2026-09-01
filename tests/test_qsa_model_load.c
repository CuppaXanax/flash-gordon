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
#include "fg_topology.h"

#include <math.h>
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
    unsigned layer=0u;
    record.layer=sscanf(name,"blk.%u.",&layer)==1?(uint16_t)layer:UINT16_MAX;
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

static int reset_and_probe(fg_owner_executor *owner,fg_vk_context *vk,
                           fg_vk_tensor *left,fg_vk_tensor *right,
                           fg_vk_tensor *sum,fg_error *error){
    const uint32_t values=32u;
    float expected[values];
    const float *a=fg_vk_tensor_map(left),*b=fg_vk_tensor_map(right);
    for(uint32_t i=0;i<values;i++)expected[i]=a[i]+b[i];
    fg_status status=fg_owner_reset_state(owner,error);
    if(status==FG_OK)status=fg_vk_begin(vk,error);
    if(status==FG_OK)status=fg_vk_add_f32(vk,sum,left,right,values,error);
    if(status==FG_OK)status=fg_vk_end(vk,error);
    if(status!=FG_OK||fg_vk_batch_active(vk))return 0;
    return memcmp(fg_vk_tensor_map(sum),expected,sizeof(expected))==0;
}

static int test_owner_prefill_failure_cleanup(void){
    static const char *weights[]={
        "blk.0.hc_attn_norm.weight","blk.0.hc_attn_down.weight",
        "blk.0.hc_attn_up.weight","blk.0.hc_attn_inject.weight",
        "blk.0.ffn_gate_inp.weight","blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_gate_shexp.weight","blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight","blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight","blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight","blk.0.ssm_conv1d.weight","blk.0.ssm_a",
        "blk.0.ssm_dt.bias","blk.0.ssm_norm.weight","blk.0.ssm_out.weight",
        "blk.1.ple_key.weight","blk.1.ple_value.weight",
        "blk.1.ple_norm_key.weight","blk.1.ple_norm_query.weight",
        "blk.1.ple_norm_conv.weight","blk.1.ple_conv1d.weight"
    };
    char directory[96],path[128];
    snprintf(directory,sizeof(directory),"test-owner-prefill-fail-%ld",(long)getpid());
    snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);
    uint8_t *block=aligned_alloc(FG_ALIGNMENT,FG_ALIGNMENT);
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_error error={0};
    int ok=block&&manifest&&mkdir(directory,0700)==0;
    if(ok)for(uint32_t i=0;i<FG_ALIGNMENT;i++)block[i]=(uint8_t)(i*13u+7u);
    if(ok)ok=write_rank(path,block,(uint32_t)(sizeof(weights)/sizeof(weights[0])));
    if(ok){
        fg_manifest_init(manifest);
        manifest->execution_mode=FG_EXECUTION_PIPELINE;
        manifest->protocol_version=FG_PIPELINE_PROTOCOL_VERSION;
        manifest->prefill_microbatch=32u;
        for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
            manifest->layer_owner[layer]=(uint8_t)(layer<2u?0u:1u);
        for(uint32_t i=0;ok&&i<sizeof(weights)/sizeof(weights[0]);i++)
            ok=add_common(manifest,weights[i],0u,(uint64_t)i*FG_ALIGNMENT,
                          block,&error);
    }
    fg_model *model=NULL;fg_owner_executor *owner=NULL;
    fg_vk_tensor *hyper=NULL,*hidden=NULL,*embedding=NULL;
    fg_vk_tensor *probe_left=NULL,*probe_right=NULL,*probe_sum=NULL;
    if(ok){
        fg_status status=fg_model_open_coordinator(&model,manifest,directory,0u,&error);
        if(status==FG_ERR_UNAVAILABLE)ok=77;
        else ok=status==FG_OK;
    }
    if(ok==1)ok=fg_owner_executor_create(&owner,model,&error)==FG_OK;
    fg_vk_context *vk=ok==1?fg_model_vk(model):NULL;
    if(ok==1)ok=fg_vk_tensor_create(vk,(uint64_t)FG_HYPER_WIDTH*4u,&hyper,
                                     &error)==FG_OK&&
                 fg_vk_tensor_create(vk,(uint64_t)FG_HIDDEN_SIZE*4u,&hidden,
                                     &error)==FG_OK&&
                 fg_vk_tensor_create(vk,(uint64_t)FG_HIDDEN_SIZE*4u,&embedding,
                                     &error)==FG_OK&&
                 fg_vk_tensor_create(vk,32u*4u,&probe_left,&error)==FG_OK&&
                 fg_vk_tensor_create(vk,32u*4u,&probe_right,&error)==FG_OK&&
                 fg_vk_tensor_create(vk,32u*4u,&probe_sum,&error)==FG_OK;
    if(ok==1){
        float *a=fg_vk_tensor_map(probe_left),*b=fg_vk_tensor_map(probe_right);
        for(uint32_t i=0;i<32u;i++){a[i]=(float)i*0.25f;b[i]=1.0f;}
    }
    fg_vk_tensor *output=NULL,*mixed=NULL,*injection=NULL;
    const fg_vk_tensor *residual=NULL;
    const uint8_t *activation=NULL;
    uint16_t expert_ids[FG_TOP_K];float gates[FG_TOP_K];
    if(ok==1){
        fg_status status=fg_owner_gr_read_batch(owner,0u,false,hyper,1u,&mixed,
            &residual,&injection,&error);
        ok=status!=FG_OK&&!fg_vk_batch_active(vk)&&
           reset_and_probe(owner,vk,probe_left,probe_right,probe_sum,&error);
    }
    if(ok==1){
        fg_status status=fg_owner_moe_prepare_batch(owner,0u,hidden,1u,
            expert_ids,gates,&activation,&error);
        ok=status!=FG_OK&&!fg_vk_batch_active(vk)&&
           reset_and_probe(owner,vk,probe_left,probe_right,probe_sum,&error);
    }
    if(ok==1){
        fg_status status=fg_owner_gdn_prefill(owner,0u,1u,hidden,&output,&error);
        ok=status!=FG_OK&&!fg_vk_batch_active(vk)&&
           reset_and_probe(owner,vk,probe_left,probe_right,probe_sum,&error);
    }
    if(ok==1){
        fg_status status=fg_owner_ple_prefill(owner,hyper,embedding,1u,&output,
                                              &error);
        ok=status!=FG_OK&&!fg_vk_batch_active(vk)&&
           reset_and_probe(owner,vk,probe_left,probe_right,probe_sum,&error);
    }
    if(vk&&fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    fg_vk_tensor_destroy(probe_sum);fg_vk_tensor_destroy(probe_right);
    fg_vk_tensor_destroy(probe_left);fg_vk_tensor_destroy(embedding);
    fg_vk_tensor_destroy(hidden);fg_vk_tensor_destroy(hyper);
    fg_owner_executor_destroy(owner);fg_model_close(model);
    free(manifest);free(block);unlink(path);rmdir(directory);
    if(ok==0)fprintf(stderr,"owner prefill failure cleanup failed: %s\n",
                     error.message);
    return ok==77?77:ok?0:1;
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
    uint64_t resident_selection=
        fg_qsa_resident_selection_scratch_bytes(262144u,128u);
    uint64_t legacy_selection=fg_qsa_selection_scratch_bytes(262144u,128u);
    ok=ok&&selection>0u&&qsa+selection<family;
    ok=ok&&FG_Q38_QSA_KEY_BYTES==544u&&FG_Q38_QSA_VALUE_BYTES==544u&&
        FG_Q38_QSA_INDEX_KEY_BYTES==136u&&FG_Q38_QSA_POSITION_BYTES==12u&&
        FG_Q38_QSA_TOKEN_RECORD_BYTES==1236u&&
        FG_Q38_QSA_COMPRESS_RATIO==4u&&FG_Q38_INDEX_BUDGET==2048u&&
        FG_QSA_TOPK_CANDIDATES==512u&&FG_QSA_TOPK_BLOCK_GROUP==4096u;
    ok=ok&&fg_qsa_record_segment_bytes(262144u,0u)==
        UINT64_C(131072)*FG_Q38_QSA_TOKEN_RECORD_BYTES&&
        fg_qsa_record_segment_bytes(262144u,1u)==
        UINT64_C(131072)*FG_Q38_QSA_TOKEN_RECORD_BYTES&&
        fg_qsa_resident_layer_bytes(262144u)==UINT64_C(359661568);
    ok=ok&&fg_qsa_resident_candidate_groups(513u*4u)==1u&&
        fg_qsa_resident_candidate_groups(4097u*4u)==2u&&
        fg_qsa_resident_candidate_groups(32769u*4u)==9u&&
        fg_qsa_resident_candidate_groups(65536u*4u)==16u&&
        fg_qsa_resident_candidate_entries(262144u,128u)==UINT64_C(1048576);
    ok=ok&&resident_selection==UINT64_C(16777216)&&
        resident_selection-legacy_selection==UINT64_C(13193216)&&
        fg_qsa_attention_scratch_bytes(128u)+resident_selection==
            UINT64_C(35349504)&&
        fg_qsa_attention_scratch_bytes(128u)+resident_selection<
            fg_qsa_attention_family_scratch_bytes(128u);
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
    if(!ok)fprintf(stderr,"geometry detail: record=%u layer=%llu groups=%u/%u/%u/%u "
                           "entries=%llu legacy=%llu resident=%llu total=%llu family=%llu\n",
        FG_Q38_QSA_TOKEN_RECORD_BYTES,
        (unsigned long long)fg_qsa_resident_layer_bytes(262144u),
        fg_qsa_resident_candidate_groups(513u*4u),
        fg_qsa_resident_candidate_groups(4097u*4u),
        fg_qsa_resident_candidate_groups(32769u*4u),
        fg_qsa_resident_candidate_groups(65536u*4u),
        (unsigned long long)fg_qsa_resident_candidate_entries(262144u,128u),
        (unsigned long long)legacy_selection,
        (unsigned long long)resident_selection,
        (unsigned long long)(fg_qsa_attention_scratch_bytes(128u)+resident_selection),
        (unsigned long long)fg_qsa_attention_family_scratch_bytes(128u));
    fg_manifest *pipeline=malloc(sizeof(*pipeline));
    if(pipeline){
        fg_manifest_init(pipeline);fg_topology_build_pipeline(pipeline);
        pipeline->max_context=FG_NATIVE_CONTEXT;
        pipeline->session.logical_context_tokens=FG_NATIVE_CONTEXT;
        pipeline->prefill_microbatch=128u;pipeline->prefill_window=2u;
        static const uint64_t expected[FG_RANK_COUNT]={
            335544320u,268435456u,268435456u,268435456u,
            268435456u,268435456u,268435456u,268435456u
        };
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
            ok=ok&&fg_q38_runtime_scratch_bytes_for_manifest(
                pipeline,rank,128u,2u,FG_NATIVE_CONTEXT)==expected[rank];
        free(pipeline);
    }else ok=0;
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

static int test_qsa_projection_submission(fg_vk_context *vk,fg_error *failure){
    enum{VALUES=128};
    float left[VALUES],right[VALUES],expected_f32[VALUES],got_f32[VALUES],
        cpu_dequantized[VALUES],gpu_dequantized[VALUES];
    uint8_t expected_q8[(VALUES/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES],
        cpu_q8[sizeof(expected_q8)],got_q8[sizeof(expected_q8)];
    for(uint32_t i=0;i<VALUES;i++){
        left[i]=0.25f+sinf((float)i*0.031f);
        right[i]=0.5f*cosf((float)i*0.017f);
        expected_f32[i]=left[i]+right[i];
    }
    fg_quantize_q8_0(expected_f32,cpu_q8,VALUES);
    fg_error error={0};
    fg_vk_tensor *lhs=NULL,*rhs=NULL,*reference_projection=NULL,
        *reference_quantized=NULL,*projection=NULL,*quantized=NULL;
    int ok=fg_vk_tensor_create(vk,sizeof(left),&lhs,&error)==FG_OK&&
        fg_vk_tensor_create(vk,sizeof(right),&rhs,&error)==FG_OK&&
        fg_vk_tensor_create(vk,sizeof(left),&reference_projection,&error)==FG_OK&&
        fg_vk_tensor_create(vk,sizeof(expected_q8),&reference_quantized,&error)==FG_OK&&
        fg_vk_tensor_create(vk,sizeof(left),&projection,&error)==FG_OK&&
        fg_vk_tensor_create(vk,sizeof(expected_q8),&quantized,&error)==FG_OK;
    if(ok){
        memcpy(fg_vk_tensor_map(lhs),left,sizeof(left));
        memcpy(fg_vk_tensor_map(rhs),right,sizeof(right));
        ok=fg_vk_add_f32(vk,reference_projection,lhs,rhs,VALUES,&error)==FG_OK&&
            fg_vk_quantize_q8_0(vk,reference_quantized,reference_projection,
                               VALUES,1u,&error)==FG_OK&&
            fg_vk_tensor_read(reference_quantized,0,expected_q8,
                             sizeof(expected_q8),&error)==FG_OK;
    }
    if(ok){
        memset(fg_vk_tensor_map(projection),0,sizeof(left));
        memset(fg_vk_tensor_map(quantized),0xA5,sizeof(expected_q8));
    }
    fg_vk_counters before={0},after={0};
    if(ok)fg_vk_get_counters(vk,&before);
    if(ok&&fg_vk_begin(vk,&error)!=FG_OK)ok=0;
    if(ok&&fg_vk_begin(vk,&error)!=FG_OK)ok=0;
    if(ok&&fg_vk_add_f32(vk,projection,lhs,rhs,VALUES,&error)!=FG_OK)ok=0;
    if(ok&&fg_vk_quantize_q8_0(vk,quantized,projection,VALUES,1u,&error)!=FG_OK)
        ok=0;
    if(ok){
        const uint8_t *pending=fg_vk_tensor_const_map(quantized);
        for(uint32_t i=0;i<sizeof(expected_q8);i++)
            if(pending[i]!=0xA5u){
                fg_error_set(&error,FG_ERR_MISMATCH,
                   "nested QSA projection batch executed before submission "
                   "at byte %u (got 0x%02x)",i,(unsigned)pending[i]);
                ok=0;break;
            }
    }
    if(ok&&fg_qsa_submit_host_reads(vk,&error)!=FG_OK)ok=0;
    if(ok&&fg_vk_batch_active(vk)){
        fg_error_set(&error,FG_ERR_MISMATCH,
                    "QSA host-read barrier left a nested Vulkan batch active");
        ok=0;
    }
    if(ok&&fg_vk_tensor_read(quantized,0,got_q8,sizeof(got_q8),&error)!=FG_OK)
        ok=0;
    if(ok&&fg_vk_tensor_read(projection,0,got_f32,sizeof(got_f32),&error)!=FG_OK)
        ok=0;
    if(ok){
        for(uint32_t i=0;i<VALUES;i++){
            if(!isfinite(left[i])||!isfinite(right[i])||
               !isfinite(expected_f32[i])||!isfinite(got_f32[i])){
                fg_error_set(&error,FG_ERR_MISMATCH,
                    "non-finite batched QSA projection at %u "
                    "(left %.9g, right %.9g, CPU %.9g, GPU %.9g)",
                    i,left[i],right[i],expected_f32[i],got_f32[i]);
                ok=0;break;
            }
            if(fabsf(got_f32[i]-expected_f32[i])>1e-6f){
                fg_error_set(&error,FG_ERR_MISMATCH,
                    "batched QSA projection value mismatch at %u "
                    "(GPU %.9g, CPU %.9g)",i,got_f32[i],expected_f32[i]);
                ok=0;break;
            }
        }
    }
    if(ok)for(uint32_t i=0;i<sizeof(expected_q8);i++)
        if(got_q8[i]!=expected_q8[i]){
            fg_error_set(&error,FG_ERR_MISMATCH,
                "batched QSA projection differs from standalone GPU reference "
                "at byte %u (batch 0x%02x, reference 0x%02x)",
                i,(unsigned)got_q8[i],(unsigned)expected_q8[i]);
            ok=0;break;
        }
    if(ok){
        fg_dequantize_q8_0(cpu_q8,cpu_dequantized,VALUES);
        fg_dequantize_q8_0(got_q8,gpu_dequantized,VALUES);
        for(uint32_t i=0;i<VALUES;i++){
            if(!isfinite(expected_f32[i])||
               !isfinite(cpu_dequantized[i])||
               !isfinite(gpu_dequantized[i])){
                fg_error_set(&error,FG_ERR_MISMATCH,
                    "non-finite batched QSA Q8_0 value at %u "
                    "(GPU %.9g, CPU %.9g, source %.9g)",
                    i,gpu_dequantized[i],cpu_dequantized[i],expected_f32[i]);
                ok=0;break;
            }
            if(fabsf(cpu_dequantized[i]-gpu_dequantized[i])>
               0.05f*fmaxf(1.0f,fabsf(expected_f32[i]))){
                fg_error_set(&error,FG_ERR_MISMATCH,
                    "batched QSA Q8_0 value mismatch at %u "
                    "(GPU %.9g, CPU %.9g, source %.9g)",
                    i,gpu_dequantized[i],cpu_dequantized[i],expected_f32[i]);
                ok=0;break;
            }
        }
    }
    if(ok){
        fg_vk_get_counters(vk,&after);
        if(after.submissions!=before.submissions+1u||
           after.dispatches!=before.dispatches+2u){
            fg_error_set(&error,FG_ERR_MISMATCH,
                "nested QSA projection submitted %llu times and dispatched "
                "%llu kernels (expected 1 submission, 2 dispatches)",
                (unsigned long long)(after.submissions-before.submissions),
                (unsigned long long)(after.dispatches-before.dispatches));
            ok=0;
        }
    }
    if(ok&&fg_qsa_submit_host_reads(vk,&error)!=FG_OK)ok=0;
    if(fg_vk_batch_active(vk)){fg_error ignored={0};fg_vk_abort(vk,&ignored);}
    fg_vk_tensor_destroy(quantized);fg_vk_tensor_destroy(projection);
    fg_vk_tensor_destroy(reference_quantized);
    fg_vk_tensor_destroy(reference_projection);
    fg_vk_tensor_destroy(rhs);fg_vk_tensor_destroy(lhs);
    if(!ok&&failure)*failure=error;
    if(!ok)fprintf(stderr,"QSA projection submission regression failed: %s\n",
                   error.message);
    return ok;
}

int main(void){
    const char *filter=getenv("DS4_REMOTE_TEST_FILTER");
    bool geometry_only=filter&&strcmp(filter,"qsa_resident_geometry")==0;
    if(!test_index_segment_geometry()){
        fprintf(stderr,"QSA segmented index geometry: FAIL\n");
        return 1;
    }
    if(!test_memory_ledger_arithmetic()){
        fprintf(stderr,"QSA memory ledger arithmetic: FAIL\n");
        return 1;
    }
    if(geometry_only){
        puts("QSA resident geometry and ledger arithmetic: PASS");
        return 0;
    }
    int owner_prefill=test_owner_gr_prefill_boundaries();
    if(owner_prefill==1)return 1;
    if(owner_prefill==77)
        fprintf(stderr,"SKIP owner GR prefill boundary test: Vulkan unavailable\n");
    int failure_cleanup=test_owner_prefill_failure_cleanup();
    if(failure_cleanup==1)return 1;
    if(failure_cleanup==77)
        fprintf(stderr,"SKIP owner prefill failure cleanup: Vulkan unavailable\n");
    char directory[96],rank0_path[128],rank3_path[128];
    snprintf(directory,sizeof(directory),"test-qsa-model-load-%ld",(long)getpid());
    snprintf(rank0_path,sizeof(rank0_path),"%s/rank-00.fgw",directory);
    snprintf(rank3_path,sizeof(rank3_path),"%s/rank-03.fgw",directory);
    if(mkdir(directory,0700)!=0){perror("mkdir");return 1;}
    uint8_t *block=aligned_alloc(FG_ALIGNMENT,FG_ALIGNMENT);
    if(!block){rmdir(directory);return 1;}
    for(uint32_t i=0;i<FG_ALIGNMENT;i++)block[i]=(uint8_t)(i*19u+5u);
    uint32_t qsa_count=(uint32_t)(sizeof(qsa_suffixes)/sizeof(qsa_suffixes[0]));
    const char *phase="fixture creation";
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
    if(ok)phase="coordinator model open";
    fg_status status=ok?fg_model_open_coordinator(&coordinator,manifest,directory,0u,&error):
                        FG_ERR_FORMAT;
    if(status==FG_ERR_UNAVAILABLE){
        fprintf(stderr,"SKIP QSA owner model placement: %s\n",error.message);
        free(manifest);free(block);unlink(rank3_path);unlink(rank0_path);rmdir(directory);
        return 77;
    }
    ok=ok&&status==FG_OK&&fg_model_tensor(coordinator,"token_embd.weight")&&
       fg_model_tensor(coordinator,"blk.3.hc_attn_norm.weight");
    if(ok)phase="projection submission";
    if(ok)ok=test_qsa_projection_submission(fg_model_vk(coordinator),&error);
    if(ok)phase="coordinator QSA tensor placement";
    for(uint32_t i=0;ok&&i<qsa_count;i++){
        char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"blk.3.%s",qsa_suffixes[i]);
        ok=fg_model_tensor(coordinator,name)!=NULL;
    }
    fg_vk_memory_stats memory_before={0},memory_after={0};
    fg_vk_get_memory_stats(fg_model_vk(coordinator),&memory_before);
    fg_qsa_session *mirror=NULL;
    if(ok)phase="small mirror lifecycle";
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
    if(ok)phase="shared-scratch mirror lifecycle";
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
    if(ok)phase="owner/expert executor lifecycle";
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
    if(ok)phase="segmented mirror residency";
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
    if(ok)phase="invalid mirror cache limit";
    if(ok)ok=fg_qsa_session_open_mirror(&mirror,coordinator,4u,4u,UINT32_MAX,1u,
                                         NULL,NULL,&error)==FG_ERR_LIMIT&&!mirror;
    if(ok)memset(&error,0,sizeof(error));
    if(ok)phase="owner model open";
    if(ok)ok=fg_model_open(&owner,manifest,directory,3u,&error)==FG_OK;
    if(ok)phase="owner QSA tensor placement";
    ok=ok&&fg_model_tensor(owner,"blk.3.hc_attn_norm.weight");
    for(uint32_t i=0;ok&&i<qsa_count;i++){
        char name[FG_TENSOR_NAME_MAX];snprintf(name,sizeof(name),"blk.3.%s",qsa_suffixes[i]);
        ok=fg_model_tensor(owner,name)!=NULL;
    }
    uint32_t saved_mode=manifest->execution_mode;
    uint32_t saved_logical=manifest->session.logical_context_tokens;
    uint32_t saved_microbatch=manifest->prefill_microbatch;
    uint16_t saved_qsa_owners[FG_LAYER_COUNT/4u];uint32_t saved_owner_count=0u;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        saved_qsa_owners[saved_owner_count++]=manifest->layer_owner[layer];
    manifest->execution_mode=FG_EXECUTION_PIPELINE;
    manifest->session.logical_context_tokens=4u;
    manifest->prefill_microbatch=1u;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        manifest->layer_owner[layer]=(uint16_t)(layer==3u?3u:0u);
    fg_vk_tensor *resident_scratch=NULL;fg_qsa_session *resident_session=NULL;
    fg_vk_memory_stats resident_before={0},resident_open={0},resident_after={0};
    fg_vk_counters resident_canary_before={0},resident_canary_after={0};
    if(ok)phase="resident QSA lifecycle";
    if(ok)ok=fg_vk_tensor_create(fg_model_vk(owner),
        fg_qsa_attention_family_scratch_bytes(1u),&resident_scratch,&error)==FG_OK;
    if(ok){
        fg_vk_get_memory_stats(fg_model_vk(owner),&resident_before);
        fg_vk_get_counters(fg_model_vk(owner),&resident_canary_before);
        ok=fg_qsa_session_open_resident(&resident_session,owner,1u,
                                        resident_scratch,&error)==FG_OK;
    }
    if(ok){
        fg_vk_get_memory_stats(fg_model_vk(owner),&resident_open);
        fg_vk_get_counters(fg_model_vk(owner),&resident_canary_after);
        uint64_t expected_delta=4u*(FG_Q38_QSA_TOKEN_RECORD_BYTES+
            FG_Q38_QSA_INDEX_KEY_BYTES+FG_Q38_QSA_POSITION_BYTES);
        uint64_t requested_delta=resident_open.requested_live_bytes-
            resident_before.requested_live_bytes;
        uint64_t canary_delta=resident_canary_after.residency_canary_calls-
            resident_canary_before.residency_canary_calls;
        if(fg_qsa_session_host_bytes(resident_session)!=0u||
           fg_qsa_session_tokens(resident_session,3u)!=0u||
           requested_delta!=expected_delta||canary_delta!=3u){
            fg_error_set(&error,FG_ERR_MISMATCH,
                "resident QSA open mismatch: host=%llu tokens=%u "
                "requested_delta=%llu/%llu canary_delta=%llu/3",
                (unsigned long long)fg_qsa_session_host_bytes(resident_session),
                fg_qsa_session_tokens(resident_session,3u),
                (unsigned long long)requested_delta,
                (unsigned long long)expected_delta,
                (unsigned long long)canary_delta);
            ok=0;
        }else if(fg_qsa_session_reset(resident_session,&error)!=FG_OK)ok=0;
    }
    fg_qsa_session_close(resident_session);resident_session=NULL;
    fg_vk_get_memory_stats(fg_model_vk(owner),&resident_after);
    if(ok&&(resident_after.requested_live_bytes!=resident_before.requested_live_bytes||
           resident_after.allocated_live_bytes!=resident_before.allocated_live_bytes||
           resident_after.live_allocations!=resident_before.live_allocations)){
        fg_error_set(&error,FG_ERR_MISMATCH,
            "resident QSA close memory mismatch: requested=%llu/%llu "
            "allocated=%llu/%llu live_allocations=%llu/%llu",
            (unsigned long long)resident_after.requested_live_bytes,
            (unsigned long long)resident_before.requested_live_bytes,
            (unsigned long long)resident_after.allocated_live_bytes,
            (unsigned long long)resident_before.allocated_live_bytes,
            (unsigned long long)resident_after.live_allocations,
            (unsigned long long)resident_before.live_allocations);
        ok=0;
    }
    fg_vk_tensor_destroy(resident_scratch);
    manifest->execution_mode=saved_mode;
    manifest->session.logical_context_tokens=saved_logical;
    manifest->prefill_microbatch=saved_microbatch;
    saved_owner_count=0u;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        manifest->layer_owner[layer]=saved_qsa_owners[saved_owner_count++];
    char owner_state_path[128];snprintf(owner_state_path,sizeof(owner_state_path),
        "%s/owner-session.qsa",directory);unlink(owner_state_path);
    fg_qsa_session *file_session=NULL;struct stat owner_state_info={0};
    if(ok)phase="file-backed QSA create";
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
    if(ok)phase="file-backed QSA reopen";
    if(ok){
        status=fg_qsa_session_open(&file_session,owner,owner_state_path,false,&error);
        ok=status==FG_OK&&file_session&&fg_qsa_session_tokens(file_session,3u)==0u;
    }
    fg_qsa_session_close(file_session);file_session=NULL;unlink(owner_state_path);
    char failed_state_path[128];snprintf(failed_state_path,sizeof(failed_state_path),
        "%s/failed-session.qsa",directory);unlink(failed_state_path);
    fg_qsa_session *failed_session=NULL;fg_error create_error={0};
    if(ok)phase="failed decode-open cleanup";
    if(ok){
        status=fg_qsa_session_open_decode(&failed_session,owner,failed_state_path,4u,
                                          UINT32_MAX,&create_error);
        ok=status!=FG_OK&&!failed_session&&access(failed_state_path,F_OK)!=0&&
           create_error.code==status&&create_error.message[0];
    }
    fg_qsa_session_close(failed_session);unlink(failed_state_path);
    fg_model_close(owner);fg_model_close(coordinator);free(manifest);free(block);
    unlink(rank3_path);unlink(rank0_path);rmdir(directory);
    if(!ok){
        if(!error.message[0])
            fg_error_set(&error,FG_ERR_MISMATCH,
                         "assertion failed without a lower-level Vulkan error");
        fprintf(stderr,"QSA owner model placement failed during %s: %s\n",
                phase,error.message);
        return 1;
    }
    puts("QSA owner model placement: PASS");return 0;
}
