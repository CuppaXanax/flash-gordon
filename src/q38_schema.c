#include "fg_q38_schema.h"
#include "fg_protocol.h"
#include "fg_qsa.h"
#include "fg_quant.h"
#include "fg_ngram.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(FG_Q38_PIPELINE_DECODE_EXTRA_BYTES==UINT64_C(92200),
               "pipeline decode scratch delta must match the admission ledger");

uint64_t fg_q38_pipeline_activation_slot_bytes(uint32_t microbatch){
    return (uint64_t)microbatch*
        ((uint64_t)FG_PIPELINE_BOUNDARY_WIDTH*FG_PIPELINE_BOUNDARY_FP32_BYTES+
         FG_PIPELINE_POSITION_AXES*sizeof(uint32_t));
}

fg_status fg_tensor_record_expected_bytes(const fg_tensor_record *record,uint64_t *bytes,
                                          fg_error *err){
    if(!record||!bytes){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tensor byte derivation");
        return FG_ERR_ARGUMENT;
    }
    *bytes=0u;
    if(record->kind==FG_TENSOR_TOKENIZER||record->kind==FG_TENSOR_VISION||
       record->kind==FG_TENSOR_MTP){
        if(!record->bytes){
            fg_error_set(err,FG_ERR_FORMAT,"opaque tensor %.80s is empty",record->name);
            return FG_ERR_FORMAT;
        }
        *bytes=record->bytes;
        return FG_OK;
    }
    uint64_t matrix=0u;
    switch(record->layout){
        case FG_TENSOR_LAYOUT_GGML:
        case FG_TENSOR_LAYOUT_HOST_Q8_0:
            if(!fg_gguf_tensor_bytes(record->ggml_type,record->dims,record->shape,
                                     bytes)){
                fg_error_set(err,FG_ERR_FORMAT,
                             "tensor %.80s has invalid GGML type/block geometry",
                             record->name);
                return FG_ERR_FORMAT;
            }
            return FG_OK;
        case FG_TENSOR_LAYOUT_Q8_0_COOKED:
            if(record->dims==2u&&record->shape[0]<=UINT32_MAX&&
               record->shape[1]<=UINT32_MAX)
                matrix=fg_q8_0_cooked_matrix_bytes(
                    (uint32_t)record->shape[0],(uint32_t)record->shape[1]);
            break;
        case FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED:
            if(record->dims==3u&&record->shape[0]<=UINT32_MAX&&
               record->shape[1]<=UINT32_MAX)
                matrix=fg_k_quant_cooked_matrix_bytes(
                    (uint32_t)record->shape[0],(uint32_t)record->shape[1],
                    record->ggml_type);
            break;
        case FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED:
            if(record->dims==3u&&record->shape[0]<=UINT32_MAX&&
               record->shape[1]<=UINT32_MAX)
                matrix=fg_q5_1_cooked_matrix_bytes(
                    (uint32_t)record->shape[0],(uint32_t)record->shape[1]);
            break;
        default:break;
    }
    if(!matrix){
        fg_error_set(err,FG_ERR_FORMAT,
                     "tensor %.80s has invalid cooked geometry",record->name);
        return FG_ERR_FORMAT;
    }
    if(record->dims==3u){
        if(!record->shape[2]||matrix>UINT64_MAX/record->shape[2]){
            fg_error_set(err,FG_ERR_LIMIT,
                         "tensor %.80s cooked byte count overflows",record->name);
            return FG_ERR_LIMIT;
        }
        matrix*=record->shape[2];
    }
    *bytes=matrix;
    return FG_OK;
}

typedef struct tensor_range {
    uint64_t begin,end;
    uint32_t index;
    uint16_t rank;
} tensor_range;

static int tensor_range_compare(const void *left,const void *right){
    const tensor_range *a=left,*b=right;
    if(a->rank!=b->rank)return a->rank<b->rank?-1:1;
    if(a->begin!=b->begin)return a->begin<b->begin?-1:1;
    return a->index<b->index?-1:a->index>b->index;
}

fg_status fg_manifest_validate_tensor_storage(const fg_manifest *manifest,fg_error *err){
    if(!manifest){
        fg_error_set(err,FG_ERR_ARGUMENT,"tensor storage manifest is null");
        return FG_ERR_ARGUMENT;
    }
    tensor_range *ranges=calloc(manifest->tensor_count,sizeof(*ranges));
    if(manifest->tensor_count&&!ranges){
        fg_error_set(err,FG_ERR_OOM,"allocate tensor range validation");
        return FG_ERR_OOM;
    }
    uint32_t range_count=0u,rank_counts[FG_RANK_COUNT]={0};
    uint64_t high_water[FG_RANK_COUNT]={0};
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        const fg_tensor_record *record=&manifest->tensors[i];uint64_t expected=0u;
        fg_status status=fg_tensor_record_expected_bytes(record,&expected,err);
        if(status!=FG_OK){free(ranges);return status;}
        if(record->bytes!=expected){
            free(ranges);fg_error_set(err,FG_ERR_MISMATCH,
                "tensor %.80s stores %llu bytes, expected %llu",record->name,
                (unsigned long long)record->bytes,(unsigned long long)expected);
            return FG_ERR_MISMATCH;
        }
        if(record->offset>UINT64_MAX-record->bytes){
            free(ranges);fg_error_set(err,FG_ERR_LIMIT,
                                     "tensor %.80s range overflows",record->name);
            return FG_ERR_LIMIT;
        }
        if(record->kind==FG_TENSOR_HOST_CACHE){
            if(record->rank>=FG_RANK_COUNT||record->offset!=0u){
                free(ranges);fg_error_set(err,FG_ERR_FORMAT,
                                         "host tensor %.80s has an invalid owner or offset",
                                         record->name);
                return FG_ERR_FORMAT;
            }
            continue;
        }
        bool external=record->kind==FG_TENSOR_NGRAM||
            record->kind==FG_TENSOR_TOKENIZER;
        if(record->rank>=FG_RANK_COUNT){
            if(record->rank!=UINT16_MAX||!external){
                free(ranges);fg_error_set(err,FG_ERR_FORMAT,
                                         "tensor %.80s has invalid rank %u",
                                         record->name,record->rank);
                return FG_ERR_FORMAT;
            }
            continue;
        }
        if(external){
            free(ranges);fg_error_set(err,FG_ERR_FORMAT,
                                     "external tensor %.80s has rank %u",
                                     record->name,record->rank);
            return FG_ERR_FORMAT;
        }
        if(!fg_is_aligned_u64(record->offset,FG_ALIGNMENT)){
            free(ranges);fg_error_set(err,FG_ERR_FORMAT,
                                     "rank tensor %.80s is not 4 KiB aligned",
                                     record->name);
            return FG_ERR_FORMAT;
        }
        uint64_t end=record->offset+record->bytes;
        ranges[range_count++]=(tensor_range){record->offset,end,i,record->rank};
        rank_counts[record->rank]++;
        uint64_t aligned=fg_align_up_u64(end,FG_ALIGNMENT);
        if(aligned<end){
            free(ranges);fg_error_set(err,FG_ERR_LIMIT,
                                     "tensor %.80s aligned range overflows",
                                     record->name);
            return FG_ERR_LIMIT;
        }
        if(aligned>high_water[record->rank])high_water[record->rank]=aligned;
    }
    qsort(ranges,range_count,sizeof(*ranges),tensor_range_compare);
    for(uint32_t i=1u;i<range_count;i++)if(ranges[i-1u].rank==ranges[i].rank&&
                                           ranges[i].begin<ranges[i-1u].end){
        const fg_tensor_record *left=&manifest->tensors[ranges[i-1u].index];
        const fg_tensor_record *right=&manifest->tensors[ranges[i].index];
        free(ranges);fg_error_set(err,FG_ERR_MISMATCH,
                                 "rank tensor ranges overlap: %.60s and %.60s",
                                 left->name,right->name);
        return FG_ERR_MISMATCH;
    }
    free(ranges);
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        if(manifest->ranks[rank].tensor_count!=rank_counts[rank]||
           manifest->ranks[rank].persistent_bytes!=high_water[rank]){
            fg_error_set(err,FG_ERR_MISMATCH,
                "rank %u inventory ledger is count=%u bytes=%llu, expected count=%u bytes=%llu",
                rank,manifest->ranks[rank].tensor_count,
                (unsigned long long)manifest->ranks[rank].persistent_bytes,
                rank_counts[rank],(unsigned long long)high_water[rank]);
            return FG_ERR_MISMATCH;
        }
    }
    return FG_OK;
}

static uint64_t runtime_scratch_bytes(uint32_t rank,uint32_t microbatch,uint32_t window,
                                      uint32_t max_context,bool ple,bool qsa,bool output,
                                      uint32_t activation_slots,bool resident_qsa,
                                      uint64_t pipeline_owner_transient,
                                      uint64_t pipeline_decode_extra){
    if(rank>=FG_RANK_COUNT||!microbatch||microbatch>512u||!window||window>4u||!max_context)return UINT64_MAX;
    uint64_t tokens=microbatch,pairs=tokens*FG_TOP_K,q8k=(FG_HIDDEN_SIZE/256u)*296u;
    uint64_t expert=tokens*q8k+
        pairs*FG_Q38_DECODE_TILE_WORDS*4u+
        pairs*FG_Q38_EXPERT_WIDTH*4u*3u+
        pairs*FG_HIDDEN_SIZE*4u;
    uint64_t owner_stateless=(FG_Q38_HYPER_WIDTH+FG_Q38_HYPER_RANK*2u+FG_Q38_HYPER_WIDTH+FG_Q38_HYPER_COUNT+FG_HIDDEN_SIZE+FG_Q38_HYPER_COUNT+FG_Q38_HYPER_WIDTH+FG_EXPERT_COUNT)*4u*tokens+tokens*q8k+(FG_Q38_EXPERT_WIDTH*3u+FG_HIDDEN_SIZE+1u+FG_HIDDEN_SIZE)*4u*tokens;
    uint64_t gdn=(FG_Q38_GDN_CONV_WIDTH*2u+FG_Q38_GDN_VALUE_WIDTH+FG_Q38_GDN_HEADS*2u+FG_Q38_GDN_VALUE_WIDTH+FG_HIDDEN_SIZE)*4u*tokens;
    uint64_t ple_dynamic=ple?(FG_Q38_HYPER_WIDTH+FG_HIDDEN_SIZE+FG_Q38_HYPER_WIDTH*6u)*4u*tokens:0u;
    uint64_t qsa_dynamic=qsa?((FG_Q38_ATTN_QUERY_WIDTH+FG_Q38_ATTN_KV_WIDTH*2u+FG_Q38_ATTN_QUERY_WIDTH+FG_Q38_ATTN_KV_WIDTH+FG_Q38_INDEX_QUERY_WIDTH+FG_Q38_INDEX_WIDTH+FG_Q38_INDEX_QUERY_WIDTH+FG_Q38_ATTN_QUERY_WIDTH/2u+FG_HIDDEN_SIZE)*4u+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES)*tokens:0u;
    uint64_t transport=(FG_Q38_HYPER_WIDTH+FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH)*4u*tokens+tokens*4u;
    uint64_t per_window=expert+owner_stateless+gdn+ple_dynamic+qsa_dynamic+transport;
    uint64_t fixed=(FG_Q38_HYPER_WIDTH+FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH)*4u;
    if(ple)fixed+=(uint64_t)FG_Q38_HYPER_WIDTH*9u*4u;
    if(qsa){
        if(resident_qsa)
            fixed+=fg_qsa_resident_selection_scratch_bytes(max_context,microbatch);
        else{
            uint64_t blocks=((uint64_t)max_context+3u)/4u;
            fixed+=blocks*4u*4u+
                (uint64_t)(FG_Q38_INDEX_BUDGET+
                           FG_Q38_QSA_COMPRESS_RATIO-1u)*
                    FG_Q38_QSA_TOKEN_RECORD_BYTES;
        }
        fixed+=4u*1024u*1024u;
    }
    if(output)fixed+=(uint64_t)(FG_Q38_HYPER_WIDTH*2u+FG_Q38_HYPER_RANK*2u+FG_HIDDEN_SIZE+FG_Q38_VOCAB_SIZE)*4u;
    fixed+=(uint64_t)activation_slots*
        fg_q38_pipeline_activation_slot_bytes(microbatch);
    fixed+=pipeline_owner_transient;
    fixed+=pipeline_decode_extra;
    return fg_align_up_u64(per_window*window+fixed,64ull<<20u);
}

uint64_t fg_q38_runtime_scratch_bytes(uint32_t rank,uint32_t microbatch,uint32_t window,uint32_t max_context){
    return runtime_scratch_bytes(rank,microbatch,window,max_context,rank==1u,
                                 rank==3u||rank==7u,rank==4u,0u,false,0u,0u);
}

uint64_t fg_q38_runtime_scratch_bytes_for_manifest(const fg_manifest *manifest,
                                                   uint32_t rank,uint32_t microbatch,
                                                   uint32_t window,uint32_t max_context){
    if(!manifest)return UINT64_MAX;
    if(manifest->execution_mode!=FG_EXECUTION_PIPELINE)
        return fg_q38_runtime_scratch_bytes(rank,microbatch,window,max_context);
    bool qsa=false;
    for(uint32_t layer=3u;layer<FG_LAYER_COUNT;layer+=4u)
        if(manifest->layer_owner[layer]==rank){qsa=true;break;}
    bool ple=manifest->layer_owner[1u]==rank;
    bool output=manifest->stage_count&&
        manifest->stage_ranks[manifest->stage_count-1u]==rank;
    uint64_t pipeline_transient=fg_q38_pipeline_owner_transient_bytes(microbatch);
    if(pipeline_transient==UINT64_MAX)return UINT64_MAX;
    uint64_t pipeline=runtime_scratch_bytes(
        rank,microbatch,window,max_context,ple,qsa,output,manifest->slot_count,true,
        pipeline_transient,FG_Q38_PIPELINE_DECODE_EXTRA_BYTES);
    uint64_t admission=fg_q38_runtime_scratch_bytes(
        rank,microbatch,window,max_context);
    return pipeline>admission?pipeline:admission;
}

typedef struct tensor_spec {
    const char *suffix;
    uint32_t type_a;
    uint32_t type_b;
    uint32_t dims;
    uint64_t shape[3];
} tensor_spec;

static const fg_gguf_tensor *find_gguf(const fg_gguf *g,const char *name){
    const fg_gguf_tensor *found=NULL;
    for(uint64_t i=0;i<g->tensor_count;i++)if(strcmp(g->tensors[i].name,name)==0){if(found)return NULL;found=&g->tensors[i];}
    return found;
}

static fg_status require_tensor(const fg_gguf *g,const char *name,const tensor_spec *s,fg_error *err){
    const fg_gguf_tensor *t=find_gguf(g,name);
    if(!t){fg_error_set(err,FG_ERR_MISMATCH,"Qwen3.8 tensor is missing or duplicated: %s",name);return FG_ERR_MISMATCH;}
    if(t->type!=s->type_a&&(s->type_b==UINT32_MAX||t->type!=s->type_b)){
        fg_error_set(err,FG_ERR_MISMATCH,"Qwen3.8 tensor %s has quant type %u, expected %u%s",name,t->type,s->type_a,s->type_b==UINT32_MAX?"":" or alternate");return FG_ERR_MISMATCH;
    }
    if(t->dims!=s->dims){fg_error_set(err,FG_ERR_MISMATCH,"Qwen3.8 tensor %s has %u dimensions, expected %u",name,t->dims,s->dims);return FG_ERR_MISMATCH;}
    for(uint32_t d=0;d<s->dims;d++)if(t->shape[d]!=s->shape[d]){fg_error_set(err,FG_ERR_MISMATCH,"Qwen3.8 tensor %s shape mismatch at dimension %u",name,d);return FG_ERR_MISMATCH;}
    return FG_OK;
}

static fg_status require_layer_set(const fg_gguf *g,uint32_t layer,const tensor_spec *specs,size_t count,fg_error *err){
    char name[FG_TENSOR_NAME_MAX];
    for(size_t i=0;i<count;i++){
        if(snprintf(name,sizeof(name),"blk.%u.%s",layer,specs[i].suffix)>=(int)sizeof(name)){fg_error_set(err,FG_ERR_LIMIT,"tensor name overflow");return FG_ERR_LIMIT;}
        fg_status rc=require_tensor(g,name,&specs[i],err);if(rc!=FG_OK)return rc;
    }
    return FG_OK;
}

fg_status fg_q38_validate_gguf(const fg_gguf *g,fg_error *err){
    static const tensor_spec global[]={
        {"token_embd.weight",8,UINT32_MAX,2,{2560,248320}},
        {"output.weight",8,UINT32_MAX,2,{2560,248320}},
        {"output_hc_down.weight",8,UINT32_MAX,2,{10240,320}},
        {"output_hc_norm.weight",0,UINT32_MAX,1,{10240}},
        {"output_hc_up.weight",8,UINT32_MAX,2,{320,10240}},
        {"per_layer_token_embd.weight",20,UINT32_MAX,2,{160,320001536}}
    };
    static const tensor_spec common[]={
        {"ffn_down_exps.weight",7,8,3,{640,2560,512}},
        {"ffn_down_shexp.weight",8,UINT32_MAX,2,{640,2560}},
        {"ffn_gate_exps.weight",12,13,3,{2560,640,512}},
        {"ffn_gate_inp.weight",0,UINT32_MAX,2,{2560,512}},
        {"ffn_gate_inp_shexp.weight",0,UINT32_MAX,1,{2560}},
        {"ffn_gate_shexp.weight",8,UINT32_MAX,2,{2560,640}},
        {"ffn_up_exps.weight",12,13,3,{2560,640,512}},
        {"ffn_up_shexp.weight",8,UINT32_MAX,2,{2560,640}},
        {"hc_attn_down.weight",8,UINT32_MAX,2,{10240,320}},
        {"hc_attn_inject.weight",0,UINT32_MAX,2,{10240,4}},
        {"hc_attn_norm.weight",0,UINT32_MAX,1,{10240}},
        {"hc_attn_up.weight",8,UINT32_MAX,2,{320,10240}},
        {"hc_ffn_down.weight",8,UINT32_MAX,2,{10240,320}},
        {"hc_ffn_inject.weight",0,UINT32_MAX,2,{10240,4}},
        {"hc_ffn_norm.weight",0,UINT32_MAX,1,{10240}},
        {"hc_ffn_up.weight",8,UINT32_MAX,2,{320,10240}}
    };
    static const tensor_spec gdn[]={
        {"attn_gate.weight",8,UINT32_MAX,2,{2560,6144}},
        {"attn_qkv.weight",8,UINT32_MAX,2,{2560,10240}},
        {"ssm_a",0,UINT32_MAX,1,{48}},
        {"ssm_alpha.weight",0,UINT32_MAX,2,{2560,48}},
        {"ssm_beta.weight",0,UINT32_MAX,2,{2560,48}},
        {"ssm_conv1d.weight",0,UINT32_MAX,2,{4,10240}},
        {"ssm_dt.bias",0,UINT32_MAX,1,{48}},
        {"ssm_norm.weight",0,UINT32_MAX,1,{128}},
        {"ssm_out.weight",8,UINT32_MAX,2,{6144,2560}}
    };
    static const tensor_spec attention[]={
        {"attn_k.weight",8,UINT32_MAX,2,{2560,512}},
        {"attn_k_norm.weight",0,UINT32_MAX,1,{256}},
        {"attn_output.weight",8,UINT32_MAX,2,{6144,2560}},
        {"attn_q.weight",8,UINT32_MAX,2,{2560,12288}},
        {"attn_q_norm.weight",0,UINT32_MAX,1,{256}},
        {"attn_v.weight",8,UINT32_MAX,2,{2560,512}},
        {"indexer.k_norm.weight",0,UINT32_MAX,1,{128}},
        {"indexer.k_proj.weight",30,UINT32_MAX,2,{2560,128}},
        {"indexer.q_norm.weight",0,UINT32_MAX,1,{128}},
        {"indexer.q_proj.weight",30,UINT32_MAX,2,{2560,512}}
    };
    static const tensor_spec ple[]={
        {"ple_conv1d.weight",0,UINT32_MAX,2,{4,10240}},
        {"ple_key.weight",8,UINT32_MAX,2,{2560,10240}},
        {"ple_norm_conv.weight",0,UINT32_MAX,1,{10240}},
        {"ple_norm_key.weight",0,UINT32_MAX,1,{10240}},
        {"ple_norm_query.weight",0,UINT32_MAX,1,{10240}},
        {"ple_value.weight",8,UINT32_MAX,2,{2560,2560}}
    };
    if(!g){fg_error_set(err,FG_ERR_ARGUMENT,"Qwen3.8 schema is null");return FG_ERR_ARGUMENT;}
    for(size_t i=0;i<sizeof(global)/sizeof(global[0]);i++){fg_status rc=require_tensor(g,global[i].suffix,&global[i],err);if(rc!=FG_OK)return rc;}
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        fg_status rc=require_layer_set(g,l,common,sizeof(common)/sizeof(common[0]),err);if(rc!=FG_OK)return rc;
        const tensor_spec *set=(l&3u)==3u?attention:gdn;size_t n=(l&3u)==3u?sizeof(attention)/sizeof(attention[0]):sizeof(gdn)/sizeof(gdn[0]);
        rc=require_layer_set(g,l,set,n,err);if(rc!=FG_OK)return rc;
    }
    return require_layer_set(g,1,ple,sizeof(ple)/sizeof(ple[0]),err);
}

const fg_tensor_record *fg_q38_find_tensor(const fg_manifest *m,const char *name,uint32_t rank){
    if(!m||!name)return NULL;
    for(uint32_t i=0;i<m->tensor_count;i++)if(strcmp(m->tensors[i].name,name)==0&&(rank==UINT32_MAX||m->tensors[i].rank==rank))return &m->tensors[i];
    return NULL;
}

const fg_ngram_shard_record *fg_q38_find_ngram_shard(const fg_manifest *manifest,
                                                     uint32_t rank){
    if(!manifest||rank==0u||rank>=FG_RANK_COUNT||
       manifest->ngram_shard_count>FG_NGRAM_SHARD_COUNT)return NULL;
    const fg_ngram_shard_record *found=NULL;
    for(uint32_t i=0;i<manifest->ngram_shard_count;i++){
        const fg_ngram_shard_record *record=&manifest->ngram_shards[i];
        if(record->logical_rank!=rank)continue;
        if(found)return NULL;
        found=record;
    }
    return found;
}

static bool digest_nonzero(const uint8_t digest[32]){
    uint8_t value=0u;
    for(uint32_t i=0;i<32u;i++)value|=digest[i];
    return value!=0u;
}

static bool canonical_ngram_rank_range(uint32_t rank,uint64_t *row_begin,
                                       uint64_t *row_count){
    static const uint64_t begin[FG_RANK_COUNT]={
        0u,0u,46666896u,93333792u,140000688u,180000846u,226667743u,273334639u
    };
    static const uint64_t count[FG_RANK_COUNT]={
        0u,46666896u,46666896u,46666896u,40000158u,46666897u,46666896u,46666897u
    };
    if(rank==0u||rank>=FG_RANK_COUNT)return false;
    *row_begin=begin[rank];*row_count=count[rank];return true;
}

fg_status fg_q38_validate_ngram_shards(const fg_manifest *manifest,fg_error *err){
    if(!manifest){
        fg_error_set(err,FG_ERR_ARGUMENT,"n-gram shard manifest is null");
        return FG_ERR_ARGUMENT;
    }
    if(manifest->format_version!=FG_MANIFEST_FORMAT_VERSION||
       manifest->execution_mode!=FG_EXECUTION_PIPELINE){
        if(manifest->ngram_shard_count){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "resident n-gram shards require a pipeline v6 manifest");
            return FG_ERR_MISMATCH;
        }
        return FG_OK;
    }
    if(manifest->ngram_shard_count!=FG_NGRAM_SHARD_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline manifest seals %u resident n-gram shards, expected %u",
                     manifest->ngram_shard_count,FG_NGRAM_SHARD_COUNT);
        return FG_ERR_MISMATCH;
    }
    if(manifest->deployment_reserved){
        fg_error_set(err,FG_ERR_FORMAT,
                     "pipeline resident n-gram metadata has invalid reserved bytes");
        return FG_ERR_FORMAT;
    }
    const fg_tensor_record *full=NULL;
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        if(manifest->tensors[i].kind!=FG_TENSOR_NGRAM)continue;
        if(full){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "pipeline manifest has multiple full n-gram tensors");
            return FG_ERR_MISMATCH;
        }
        full=&manifest->tensors[i];
    }
    const uint64_t total_rows=UINT64_C(320001536);
    if(!full||full->rank!=UINT16_MAX||full->offset!=0u||full->ggml_type!=20u||
       full->dims!=2u||full->shape[0]!=FG_NGRAM_EMBED_WIDTH||
       full->shape[1]!=total_rows||full->shape[2]||full->shape[3]||
       full->bytes!=total_rows*FG_NGRAM_ROW_BYTES){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline full n-gram tensor geometry is not canonical");
        return FG_ERR_MISMATCH;
    }
    uint64_t next_row=0u;
    bool seen[FG_RANK_COUNT]={0};
    for(uint32_t i=0;i<manifest->ngram_shard_count;i++){
        const fg_ngram_shard_record *record=&manifest->ngram_shards[i];
        uint32_t rank=record->logical_rank;
        uint64_t row_begin=0u,row_count=0u;
        if(rank==0u||rank>=FG_RANK_COUNT||seen[rank]||record->reserved||
           !canonical_ngram_rank_range(rank,&row_begin,&row_count)||
           record->row_begin!=row_begin||record->row_count!=row_count||
           row_count>UINT64_MAX/FG_NGRAM_ROW_BYTES||
           record->bytes!=row_count*FG_NGRAM_ROW_BYTES||
           record->row_begin!=next_row||!digest_nonzero(record->sha256)||
           manifest->host_resident_bytes[rank]!=
               FG_PIPELINE_NGRAM_CACHE_BYTES){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "pipeline sealed n-gram shard/cache metadata is invalid at index %u",
                         i);
            return FG_ERR_MISMATCH;
        }
        seen[rank]=true;
        next_row+=row_count;
    }
    if(next_row!=total_rows){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "pipeline resident n-gram shards cover %llu rows, expected %llu",
                     (unsigned long long)next_row,(unsigned long long)total_rows);
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_q38_rank_residency_bytes(const fg_manifest *manifest,uint32_t rank,
                                      uint64_t *bytes,fg_error *err){
    if(!manifest||!bytes||rank>=FG_RANK_COUNT){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid rank residency query");
        return FG_ERR_ARGUMENT;
    }
    const fg_rank_record *record=&manifest->ranks[rank];
    const uint64_t parts[]={
        record->persistent_bytes,manifest->host_resident_bytes[rank],
        record->transient_bytes,record->kv_bytes,record->scratch_bytes,
        record->driver_reserve_bytes
    };
    uint64_t total=0u;
    for(uint32_t i=0;i<sizeof(parts)/sizeof(parts[0]);i++){
        if(parts[i]>UINT64_MAX-total){
            fg_error_set(err,FG_ERR_LIMIT,"rank %u residency ledger overflows",rank);
            return FG_ERR_LIMIT;
        }
        total+=parts[i];
    }
    *bytes=total;
    return FG_OK;
}

void fg_q38_session_state_bytes_for_rank(const fg_manifest *manifest,uint32_t rank,
                                         uint64_t *kv_bytes,uint64_t *state_file_bytes){
    if(kv_bytes)*kv_bytes=0u;
    if(state_file_bytes)*state_file_bytes=0u;
    if(!manifest||rank>=FG_RANK_COUNT)return;
    uint64_t kv=0u;uint32_t qsa_layers=0u;
    const uint64_t qsa_state_blocks=
        (manifest->max_context+FG_Q38_QSA_COMPRESS_RATIO-1u)/
        FG_Q38_QSA_COMPRESS_RATIO;
    const uint64_t gdn_layer_bytes=(uint64_t)FG_Q38_GDN_CONV_WIDTH*4u*4u+(uint64_t)FG_Q38_GDN_HEADS*128u*128u*4u;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        if(manifest->layer_owner[layer]!=rank)continue;
        if((layer&3u)==3u){
            qsa_layers++;
            kv+=(uint64_t)manifest->max_context*
                (FG_Q38_QSA_INDEX_KEY_BYTES+
                 (manifest->execution_mode==FG_EXECUTION_PIPELINE?
                  FG_Q38_QSA_TOKEN_RECORD_BYTES:0u));
        }else kv+=gdn_layer_bytes;
    }
    uint64_t state=0u;
    if(qsa_layers){
        kv+=(uint64_t)manifest->max_context*FG_Q38_QSA_POSITION_BYTES;
        state=FG_Q38_QSA_STATE_PAGE_BYTES+
            (uint64_t)qsa_layers*qsa_state_blocks*FG_Q38_QSA_STATE_PAGE_BYTES;
    }
    if(kv_bytes)*kv_bytes=kv;
    if(state_file_bytes)*state_file_bytes=state;
}

void fg_q38_account_session_state(fg_manifest *manifest){
    if(!manifest)return;
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        fg_q38_session_state_bytes_for_rank(
            manifest,rank,&manifest->ranks[rank].kv_bytes,
            &manifest->ranks[rank].state_file_bytes);
}

fg_status fg_q38_validate_packed_manifest(const fg_manifest *m,fg_error *err){
    enum{
        SOURCE_TENSORS=1224,
        EXPERT_FAMILIES=3
    };
    static const char *expert_suffix[EXPERT_FAMILIES]={
        "ffn_down_exps.weight","ffn_gate_exps.weight","ffn_up_exps.weight"
    };
    if(!m){fg_error_set(err,FG_ERR_ARGUMENT,"packed Qwen3.8 manifest is null");return FG_ERR_ARGUMENT;}
    bool pipeline=m->execution_mode==FG_EXECUTION_PIPELINE;
    if(pipeline&&m->stage_count!=FG_PIPELINE_STAGE_COUNT){
        fg_error_set(err,FG_ERR_MISMATCH,"pipeline stage count is invalid");
        return FG_ERR_MISMATCH;
    }
    fg_status storage_status=fg_manifest_validate_tensor_storage(m,err);
    if(storage_status!=FG_OK)return storage_status;
    if(pipeline){
        fg_status shard_status=fg_q38_validate_ngram_shards(m,err);
        if(shard_status!=FG_OK)return shard_status;
    }
    uint32_t packed_model_tensors=pipeline?SOURCE_TENSORS:
        SOURCE_TENSORS+FG_LAYER_COUNT*EXPERT_FAMILIES*(FG_GROUP_SIZE-1u);
    uint32_t model_tensors=0,tokenizer_tensors=0,vision_tensors=0,mtp_tensors=0;
    for(uint32_t i=0;i<m->tensor_count;i++)switch(m->tensors[i].kind){
        case FG_TENSOR_COMMON:case FG_TENSOR_ROUTED_EXPERT:case FG_TENSOR_NGRAM:
        case FG_TENSOR_HOST_CACHE:model_tensors++;break;
        case FG_TENSOR_TOKENIZER:tokenizer_tensors++;break;
        case FG_TENSOR_VISION:vision_tensors++;break;
        case FG_TENSOR_MTP:mtp_tensors++;break;
        default:fg_error_set(err,FG_ERR_MISMATCH,"packed Qwen3.8 manifest contains unknown tensor kind %u",m->tensors[i].kind);return FG_ERR_MISMATCH;
    }
    if(model_tensors!=packed_model_tensors){
        fg_error_set(err,FG_ERR_MISMATCH,"packed Qwen3.8 model tensor count is %u, expected %u",model_tensors,packed_model_tensors);
        return FG_ERR_MISMATCH;
    }
    if(tokenizer_tensors>1u||((m->flags&FG_MANIFEST_HAS_TOKENIZER)!=0u)!=(tokenizer_tensors==1u)||
       ((m->flags&FG_MANIFEST_HAS_VISION)!=0u)!=(vision_tensors!=0u)||
       ((m->flags&FG_MANIFEST_HAS_MTP)!=0u)!=(mtp_tensors!=0u)){
        fg_error_set(err,FG_ERR_MISMATCH,"packed Qwen3.8 auxiliary tensor flags/counts disagree");return FG_ERR_MISMATCH;
    }
    fg_gguf synthetic={0};
    synthetic.tensors=calloc(SOURCE_TENSORS,sizeof(*synthetic.tensors));
    char (*names)[FG_TENSOR_NAME_MAX]=calloc(SOURCE_TENSORS,sizeof(*names));
    if(!synthetic.tensors||!names){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_OOM,"allocate packed schema reconstruction");return FG_ERR_OOM;}
    uint32_t reconstructed=0;
    for(uint32_t i=0;i<m->tensor_count;i++){
        const fg_tensor_record *r=&m->tensors[i];
        if(r->kind==FG_TENSOR_ROUTED_EXPERT||r->kind==FG_TENSOR_TOKENIZER||r->kind==FG_TENSOR_VISION||r->kind==FG_TENSOR_MTP)continue;
        if(r->layer<FG_LAYER_COUNT&&r->kind==FG_TENSOR_COMMON&&r->rank!=m->layer_owner[r->layer]){
            free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"owner tensor %s is on rank %u, expected %u",r->name,r->rank,m->layer_owner[r->layer]);return FG_ERR_MISMATCH;
        }
        uint32_t first_rank=pipeline?m->stage_ranks[0]:0u;
        uint32_t terminal_rank=pipeline?m->stage_ranks[m->stage_count-1u]:4u;
        if(strcmp(r->name,"token_embd.weight")==0){
            bool storage_ok=pipeline?
                r->kind==FG_TENSOR_HOST_CACHE&&
                    r->layout==FG_TENSOR_LAYOUT_HOST_Q8_0&&r->offset==0u:
                r->kind==FG_TENSOR_COMMON&&r->layout==FG_TENSOR_LAYOUT_GGML;
            if(r->rank!=first_rank||!storage_ok){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"token embedding storage or owner is invalid");return FG_ERR_MISMATCH;}
        }
        if((strcmp(r->name,"output.weight")==0||strncmp(r->name,"output_hc_",10u)==0)&&r->rank!=terminal_rank){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"output bundle tensor %s must be on rank %u",r->name,terminal_rank);return FG_ERR_MISMATCH;}
        if(reconstructed>=SOURCE_TENSORS){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_LIMIT,"too many reconstructed tensors");return FG_ERR_LIMIT;}
        snprintf(names[reconstructed],FG_TENSOR_NAME_MAX,"%s",r->name);
        synthetic.tensors[reconstructed].name=names[reconstructed];synthetic.tensors[reconstructed].dims=r->dims;
        memcpy(synthetic.tensors[reconstructed].shape,r->shape,sizeof(r->shape));synthetic.tensors[reconstructed].type=r->ggml_type;reconstructed++;
    }
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++)for(uint32_t family=0;family<EXPERT_FAMILIES;family++){
        char base[FG_TENSOR_NAME_MAX];snprintf(base,sizeof(base),"blk.%u.%s",l,expert_suffix[family]);
        const fg_tensor_record *representative=NULL;
        if(pipeline){
            uint32_t rank=m->layer_owner[l];
            const fg_tensor_record *r=fg_q38_find_tensor(m,base,rank);
            if(!r||r->kind!=FG_TENSOR_ROUTED_EXPERT||r->layer!=l||
               r->dims!=3||r->shape[2]!=FG_EXPERT_COUNT){
                free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"invalid stage-local expert tensor %s",base);return FG_ERR_MISMATCH;
            }
            representative=r;
        }else{
            for(uint32_t gi=0;gi<FG_GROUP_SIZE;gi++){
                uint32_t rank=m->layer_groups[l][gi];char packed[FG_TENSOR_NAME_MAX+16u];snprintf(packed,sizeof(packed),"%s.rank%u",base,rank);
                const fg_tensor_record *r=fg_q38_find_tensor(m,packed,rank);
                if(!r||r->kind!=FG_TENSOR_ROUTED_EXPERT||r->layer!=l||r->dims!=3||r->shape[2]!=FG_EXPERTS_PER_RANK){
                    free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"invalid local expert tensor %s",packed);return FG_ERR_MISMATCH;
                }
                if(representative&&(r->ggml_type!=representative->ggml_type||r->shape[0]!=representative->shape[0]||r->shape[1]!=representative->shape[1])){
                    free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"local expert shards disagree for %s",base);return FG_ERR_MISMATCH;
                }
                representative=r;
            }
        }
        if(reconstructed>=SOURCE_TENSORS){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_LIMIT,"too many reconstructed tensors");return FG_ERR_LIMIT;}
        snprintf(names[reconstructed],FG_TENSOR_NAME_MAX,"%s",base);synthetic.tensors[reconstructed].name=names[reconstructed];
        synthetic.tensors[reconstructed].dims=representative->dims;memcpy(synthetic.tensors[reconstructed].shape,representative->shape,sizeof(representative->shape));
        synthetic.tensors[reconstructed].shape[2]=FG_EXPERT_COUNT;synthetic.tensors[reconstructed].type=representative->ggml_type;reconstructed++;
    }
    synthetic.tensor_count=reconstructed;
    fg_status rc=reconstructed==SOURCE_TENSORS?fg_q38_validate_gguf(&synthetic,err):FG_ERR_MISMATCH;
    if(reconstructed!=SOURCE_TENSORS)fg_error_set(err,FG_ERR_MISMATCH,"reconstructed Qwen3.8 tensor count is %u, expected %u",reconstructed,SOURCE_TENSORS);
    free(names);free(synthetic.tensors);return rc;
}
