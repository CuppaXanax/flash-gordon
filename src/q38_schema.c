#include "fg_q38_schema.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t fg_q38_runtime_scratch_bytes(uint32_t rank,uint32_t microbatch,uint32_t window,uint32_t max_context){
    if(rank>=FG_RANK_COUNT||!microbatch||microbatch>512u||!window||window>4u||!max_context)return UINT64_MAX;
    uint64_t tokens=microbatch,pairs=tokens*FG_TOP_K,q8k=(FG_HIDDEN_SIZE/256u)*296u;
    uint64_t expert=tokens*q8k+pairs*9u*4u+pairs*FG_Q38_EXPERT_WIDTH*4u*3u+pairs*FG_HIDDEN_SIZE*4u;
    uint64_t owner_stateless=(FG_Q38_HYPER_WIDTH+FG_Q38_HYPER_RANK*2u+FG_Q38_HYPER_WIDTH+FG_Q38_HYPER_COUNT+FG_HIDDEN_SIZE+FG_Q38_HYPER_COUNT+FG_Q38_HYPER_WIDTH+FG_EXPERT_COUNT)*4u*tokens+tokens*q8k+(FG_Q38_EXPERT_WIDTH*3u+FG_HIDDEN_SIZE+1u+FG_HIDDEN_SIZE)*4u*tokens;
    uint64_t gdn=(FG_Q38_GDN_CONV_WIDTH*2u+FG_Q38_GDN_VALUE_WIDTH+FG_Q38_GDN_HEADS*2u+FG_Q38_GDN_VALUE_WIDTH+FG_HIDDEN_SIZE)*4u*tokens;
    uint64_t ple_dynamic=rank==1u?(FG_Q38_HYPER_WIDTH+FG_HIDDEN_SIZE+FG_Q38_HYPER_WIDTH*6u)*4u*tokens:0u;
    uint64_t qsa_dynamic=(rank==3u||rank==7u)?((FG_Q38_ATTN_QUERY_WIDTH+FG_Q38_ATTN_KV_WIDTH*2u+FG_Q38_ATTN_QUERY_WIDTH+FG_Q38_ATTN_KV_WIDTH+FG_Q38_INDEX_QUERY_WIDTH+FG_Q38_INDEX_WIDTH+FG_Q38_INDEX_QUERY_WIDTH+FG_Q38_ATTN_QUERY_WIDTH/2u+FG_HIDDEN_SIZE)*4u+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES)*tokens:0u;
    uint64_t transport=(FG_Q38_HYPER_WIDTH+FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH)*4u*tokens+tokens*4u;
    uint64_t per_window=expert+owner_stateless+gdn+ple_dynamic+qsa_dynamic+transport;
    uint64_t fixed=(FG_Q38_HYPER_WIDTH+FG_NGRAM_HEAD_COUNT*FG_NGRAM_EMBED_WIDTH)*4u;
    if(rank==1u)fixed+=(uint64_t)FG_Q38_HYPER_WIDTH*9u*4u;
    if(rank==3u||rank==7u){uint64_t blocks=((uint64_t)max_context+3u)/4u;fixed+=blocks*4u*4u+(uint64_t)(FG_Q38_INDEX_BUDGET+FG_Q38_QSA_COMPRESS_RATIO-1u)*FG_Q38_QSA_TOKEN_RECORD_BYTES+4u*1024u*1024u;}
    if(rank==4u)fixed+=(uint64_t)(FG_Q38_HYPER_WIDTH*2u+FG_Q38_HYPER_RANK*2u+FG_HIDDEN_SIZE+FG_Q38_VOCAB_SIZE)*4u;
    return fg_align_up_u64(per_window*window+fixed,64ull<<20u);
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

void fg_q38_account_session_state(fg_manifest *manifest){
    if(!manifest)return;
    bool has_qsa[FG_RANK_COUNT]={0};for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){manifest->ranks[rank].kv_bytes=0;manifest->ranks[rank].state_file_bytes=0;}
    const uint64_t qsa_index_bytes_per_token=(FG_Q38_INDEX_WIDTH/32u)*FG_Q38_Q8_0_BLOCK_BYTES;
    const uint64_t qsa_state_blocks=(manifest->max_context+FG_Q38_QSA_COMPRESS_RATIO-1u)/FG_Q38_QSA_COMPRESS_RATIO;
    const uint64_t qsa_state_file_bytes=qsa_state_blocks*FG_Q38_QSA_STATE_PAGE_BYTES;
    const uint64_t gdn_layer_bytes=(uint64_t)FG_Q38_GDN_CONV_WIDTH*4u*4u+(uint64_t)FG_Q38_GDN_HEADS*128u*128u*4u;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        uint32_t owner=manifest->layer_owner[layer];
        if((layer&3u)==3u){
            has_qsa[owner]=true;
            manifest->ranks[owner].kv_bytes+=qsa_index_bytes_per_token*manifest->max_context;
            manifest->ranks[owner].state_file_bytes+=qsa_state_file_bytes;
        }else manifest->ranks[owner].kv_bytes+=gdn_layer_bytes;
    }
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(has_qsa[rank]){
        manifest->ranks[rank].kv_bytes+=(uint64_t)manifest->max_context*3u*4u;
        manifest->ranks[rank].state_file_bytes+=FG_Q38_QSA_STATE_PAGE_BYTES;
    }
}

fg_status fg_q38_validate_packed_manifest(const fg_manifest *m,fg_error *err){
    enum{
        SOURCE_TENSORS=1224,
        EXPERT_FAMILIES=3,
        PACKED_MODEL_TENSORS=SOURCE_TENSORS+
            FG_LAYER_COUNT*EXPERT_FAMILIES*(FG_GROUP_SIZE-1)
    };
    static const char *expert_suffix[EXPERT_FAMILIES]={
        "ffn_down_exps.weight","ffn_gate_exps.weight","ffn_up_exps.weight"
    };
    if(!m){fg_error_set(err,FG_ERR_ARGUMENT,"packed Qwen3.8 manifest is null");return FG_ERR_ARGUMENT;}
    uint32_t model_tensors=0,tokenizer_tensors=0,vision_tensors=0,mtp_tensors=0;
    for(uint32_t i=0;i<m->tensor_count;i++)switch(m->tensors[i].kind){
        case FG_TENSOR_COMMON:case FG_TENSOR_ROUTED_EXPERT:case FG_TENSOR_NGRAM:model_tensors++;break;
        case FG_TENSOR_TOKENIZER:tokenizer_tensors++;break;
        case FG_TENSOR_VISION:vision_tensors++;break;
        case FG_TENSOR_MTP:mtp_tensors++;break;
        default:fg_error_set(err,FG_ERR_MISMATCH,"packed Qwen3.8 manifest contains unknown tensor kind %u",m->tensors[i].kind);return FG_ERR_MISMATCH;
    }
    if(model_tensors!=PACKED_MODEL_TENSORS){
        fg_error_set(err,FG_ERR_MISMATCH,"packed Qwen3.8 model tensor count is %u, expected %u",model_tensors,PACKED_MODEL_TENSORS);
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
        if(strcmp(r->name,"token_embd.weight")==0&&r->rank!=0u){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"token embedding must be on coordinator rank 0");return FG_ERR_MISMATCH;}
        if((strcmp(r->name,"output.weight")==0||strncmp(r->name,"output_hc_",10u)==0)&&r->rank!=4u){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_MISMATCH,"output bundle tensor %s must be on rank 4",r->name);return FG_ERR_MISMATCH;}
        if(reconstructed>=SOURCE_TENSORS){free(names);free(synthetic.tensors);fg_error_set(err,FG_ERR_LIMIT,"too many reconstructed tensors");return FG_ERR_LIMIT;}
        snprintf(names[reconstructed],FG_TENSOR_NAME_MAX,"%s",r->name);
        synthetic.tensors[reconstructed].name=names[reconstructed];synthetic.tensors[reconstructed].dims=r->dims;
        memcpy(synthetic.tensors[reconstructed].shape,r->shape,sizeof(r->shape));synthetic.tensors[reconstructed].type=r->ggml_type;reconstructed++;
    }
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++)for(uint32_t family=0;family<EXPERT_FAMILIES;family++){
        char base[FG_TENSOR_NAME_MAX];snprintf(base,sizeof(base),"blk.%u.%s",l,expert_suffix[family]);
        const fg_tensor_record *representative=NULL;
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
