#include "fg_quant.h"
#include "../src/qsa.c"

/* Real QSA orchestration and Vulkan kernels with a single-layer model fixture. */
struct fg_model {fg_vk_context *vk;fg_manifest manifest;fg_vk_tensor *norm;};
fg_vk_context *fg_model_vk(fg_model *m){return m->vk;}
const fg_manifest *fg_model_manifest(const fg_model *m){return &m->manifest;}
uint32_t fg_model_rank(const fg_model *m){(void)m;return 0u;}
fg_vk_tensor *fg_model_tensor(fg_model *m,const char *name){
    return !strcmp(name,"blk.3.indexer.k_norm.weight")?m->norm:NULL;
}

static fg_error error;
#define REQUIRE(expr) do{if(!(expr)){fprintf(stderr,"line %d: %s: %s\n",__LINE__,#expr,error.message);return 0;}}while(0)

static int setup(fg_qsa_session *s,fg_model *m,fg_vk_tensor **scratch){
    s->model=m;s->max_context=262144u;s->max_blocks=65536u;
    s->max_tokens=128u;s->layer_count=1u;s->layers[0]=3u;s->cache_pages=2u;
    REQUIRE(make_tensor(s,fg_qsa_attention_family_scratch_bytes(128u),scratch,&error)==FG_OK);
    REQUIRE(create_attention_views(s,*scratch,128u,&error)==FG_OK);
    REQUIRE(create_selection_views(s,*scratch,128u,&error)==FG_OK);
    REQUIRE(create_tile_views(s,128u,&error)==FG_OK);
    REQUIRE(make_tensor(s,(uint64_t)s->max_context*12u,&s->positions,&error)==FG_OK);
    REQUIRE(create_index_segments(s,&error)==FG_OK);
    REQUIRE(create_reusable_views(s,&error)==FG_OK);
    REQUIRE(fg_qsa_page_cache_create(&s->cache,2u,&error)==FG_OK);
    REQUIRE(make_tensor(s,2u*FG_QSA_PAGE_RECORD_BYTES,&s->cache_records,&error)==FG_OK);
    REQUIRE(make_tensor(s,128u*4u,&m->norm,&error)==FG_OK);
    float *norm=fg_vk_tensor_map(m->norm);
    for(uint32_t i=0;i<128u;i++)norm[i]=1.0f+0.1f*sinf((float)i);
    uint32_t *positions=fg_vk_tensor_map(s->positions);
    for(uint32_t t=0;t<s->max_context;t++)for(uint32_t a=0;a<3u;a++)positions[t*3u+a]=t+a;
    /* Segment 0 is eager; later segments stay lazy until first use. */
    REQUIRE(s->index_keys[0][0]!=NULL);
    for(uint32_t seg=1;seg<s->index_segment_count;seg++)
        REQUIRE(s->index_keys[0][seg]==NULL);
    float *query=fg_vk_tensor_map(s->index_query);
    for(uint32_t q=0;q<128u;q++)for(uint32_t i=0;i<512u;i++)
        query[q*512u+i]=cosf((float)(q+1u)*(float)(i+1u)*0.023f);
    return 1;
}

/* The batched selector must materialize a lazy segment the first time the
 * visible range crosses into it, and must not reallocate it afterwards. */
static int lazy_index_segment(fg_qsa_session *s){
    REQUIRE(s->index_segment_count>1u);
    REQUIRE(s->index_keys[0][1]==NULL);
    uint32_t selected[4][512],counts[4];
    REQUIRE(select_prefill_batch(s,0u,131075u,0u,4u,&selected[0][0],512u,
                                 counts,&error)==FG_OK);
    REQUIRE(s->index_keys[0][1]!=NULL);
    fg_vk_tensor *tensor=s->index_keys[0][1];
    REQUIRE(ensure_index_segment(s,0u,1u,&error)==FG_OK);
    REQUIRE(s->index_keys[0][1]==tensor);
    return 1;
}

static int fill_index_rows(fg_qsa_session *s){
    /* Repeated quantized rows are inexpensive to build; MRoPE makes block scores distinct. */
    uint8_t rows[64][136];float raw[128];
    for(uint32_t r=0;r<64u;r++){
        for(uint32_t i=0;i<128u;i++)raw[i]=sinf((float)(r+1u)*(float)(i+1u)*0.013f);
        fg_quantize_q8_0(raw,rows[r],128u);
    }
    for(uint32_t seg=0;seg<s->index_segment_count;seg++){
        REQUIRE(ensure_index_segment(s,0u,seg,&error)==FG_OK);
        uint8_t *keys=fg_vk_tensor_map(s->index_keys[0][seg]);
        for(uint32_t t=0;t<s->index_segment_tokens[seg];t++)memcpy(keys+(uint64_t)t*136u,rows[t%64u],136u);
    }
    return 1;
}

/* Regression for the fleet's "invalid causal QSA score tile" death: the
 * segment window counted one extra visible token, so a prefill chunk whose
 * endpoint completed a four-token block (first_visible+queries == 0 mod 4)
 * scored blocks_end one past blocks_total and the score dispatch rejected the
 * tile.  Sweep the boundary with a full microbatch and every partial endpoint,
 * over both fresh (first_token % 4 == 0) and continued (any residue) starts. */
static int selection_window_boundary(fg_qsa_session *s){
    static const uint32_t probes[]={128u,1u,2u,3u};
    uint32_t counts[128];
    for(uint32_t base=10240u;base<=13312u;base+=128u){
        for(uint32_t residue=0u;residue<4u;residue++){
            uint32_t first_visible=base+residue+1u;
            for(uint32_t p=0;p<4u;p++){
                uint32_t queries=probes[p];
                REQUIRE(select_prefill_batch(s,0u,first_visible,0u,queries,
                    s->select_ids,FG_QSA_MAX_SELECTED_BLOCKS,counts,&error)==FG_OK);
                for(uint32_t q=0;q<queries;q++){
                    uint32_t complete=(first_visible+q)/FG_Q38_QSA_COMPRESS_RATIO;
                    uint32_t expected=complete<FG_QSA_MAX_SELECTED_BLOCKS?
                        complete:FG_QSA_MAX_SELECTED_BLOCKS;
                    REQUIRE(counts[q]==expected);
                    for(uint32_t i=0;i<counts[q];i++)
                        REQUIRE(s->select_ids[(uint64_t)q*
                            FG_QSA_MAX_SELECTED_BLOCKS+i]<complete);
                }
            }
        }
    }
    return 1;
}

static int selection_parity(fg_qsa_session *s,uint32_t first){
    uint32_t selected[FG_QSA_PREFILL_QUERY_TILE][512],counts[4],reference[512],count=0;
    /* Nonzero query offset catches incorrect row strides in the score tile. */
    REQUIRE(select_prefill_tile(s,0u,first,3u,4u,selected,counts,&error)==FG_OK);
    for(uint32_t q=0;q<4u;q++){
        REQUIRE(counts[q]==512u);
        REQUIRE(fg_vk_tensor_view_rebind(s->index_query_view,s->index_query,
            (uint64_t)(q+3u)*512u*4u,512u*4u,&error)==FG_OK);
        REQUIRE(fg_vk_begin(fg_model_vk(s->model),&error)==FG_OK);
        REQUIRE(select_blocks(s,0u,s->index_query_view,first+q+4u,reference,&count,false,&error)==FG_OK);
        REQUIRE(count==counts[q]);
        REQUIRE(memcmp(reference,selected[q],count*4u)==0);
        for(uint32_t i=0;i<count;i++)REQUIRE(selected[q][i]<(first+q+4u)/4u);
    }
    return 1;
}

typedef struct fetch_fixture {uint8_t pages[6][FG_QSA_PAGE_RECORD_BYTES];uint32_t calls,pages_read;bool fail;} fetch_fixture;
static fg_status fetch_pages(void *opaque,uint32_t layer,const uint32_t *blocks,
    uint32_t count,uint8_t *records,fg_error *err){
    fetch_fixture *f=opaque;
    if(f->fail){fg_error_set(err,FG_ERR_IO,"fixture fetch failure");return FG_ERR_IO;}
    if(layer!=3u)return FG_ERR_MISMATCH;
    f->calls++;f->pages_read+=count;
    for(uint32_t i=0;i<count;i++){
        if(blocks[i]>=6u)return FG_ERR_LIMIT;
        memcpy(records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,f->pages[blocks[i]],FG_QSA_PAGE_RECORD_BYTES);
    }
    return FG_OK;
}

static int gather_eviction(fg_qsa_session *s,bool pin_all){
    fetch_fixture f={0};
    for(uint32_t b=0;b<6u;b++)for(uint32_t i=0;i<FG_QSA_PAGE_RECORD_BYTES;i++)
        f.pages[b][i]=(uint8_t)(b*31u+i*7u);
    s->fetch_pages=fetch_pages;s->fetch_opaque=&f;
    fg_qsa_page_cache_reset(s->cache);
    const uint32_t resident[]={0u,5u};
    for(uint32_t i=0;i<2u;i++){
        uint32_t slot;bool hit;
        REQUIRE(fg_qsa_page_cache_acquire(s->cache,3u,resident[i],&slot,&hit,&error)==FG_OK);
        REQUIRE(fg_vk_tensor_write(s->cache_records,(uint64_t)slot*FG_QSA_PAGE_RECORD_BYTES,
            f.pages[resident[i]],FG_QSA_PAGE_RECORD_BYTES,&error)==FG_OK);
        if(pin_all||i==1u)REQUIRE(fg_qsa_page_cache_pin(s->cache,3u,resident[i],&error)==FG_OK);
    }
    uint32_t selected[FG_QSA_PREFILL_QUERY_TILE][512]={{0u,1u,2u},{2u,0u,3u},{1u,3u,4u},{4u,2u,0u}};
    uint32_t counts[4]={3u,3u,3u,3u};
    REQUIRE(gather_prefill_tile(s,0u,21u,4u,selected,counts,&error)==FG_OK);
    REQUIRE(f.calls==1u&&f.pages_read==4u);
    for(uint32_t q=0;q<4u;q++){
        const uint8_t *got=fg_vk_tensor_map(s->tile_records[q]);
        for(uint32_t i=0;i<3u;i++)REQUIRE(!memcmp(got+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES,
            f.pages[selected[q][i]],FG_QSA_PAGE_RECORD_BYTES));
        REQUIRE(!memcmp(got+3u*FG_QSA_PAGE_RECORD_BYTES,f.pages[5],
            ((21u+q)%4u)*FG_Q38_QSA_TOKEN_RECORD_BYTES));
    }
    /* Unpublished future data may be in cache, but selection must reject it. */
    selected[0][0]=5u;
    REQUIRE(gather_prefill_tile(s,0u,21u,4u,selected,counts,&error)==FG_ERR_MISMATCH);
    REQUIRE(fg_vk_abort(fg_model_vk(s->model),&error)==FG_OK);
    selected[0][0]=1u;f.fail=true;
    REQUIRE(gather_prefill_tile(s,0u,21u,4u,selected,counts,&error)==FG_ERR_IO);
    REQUIRE(!fg_vk_batch_active(fg_model_vk(s->model)));
    REQUIRE(s->committed[0]==0u);
    s->fetch_opaque=NULL;
    return 1;
}

static int causal_attention(fg_qsa_session *s){
    fg_qsa_page_cache_reset(s->cache);
    s->committed[0]=1u;
    uint8_t records[6][FG_Q38_QSA_TOKEN_RECORD_BYTES];
    float key[512]={0},value[512],index[128]={0},decoded[512],sum=0.0f;
    float expected[6];
    for(uint32_t t=0;t<6u;t++){
        for(uint32_t i=0;i<512u;i++)value[i]=(float)(t+1u);
        uint32_t position[]={t,t,t};
        fg_qsa_encode_full_token_record(key,value,index,position,records[t]);
        fg_dequantize_q8_0(records[t]+FG_Q38_QSA_KEY_BYTES,decoded,512u);
        sum+=decoded[0];expected[t]=0.5f*sum/(float)(t+1u);
        if(!t)continue;
        REQUIRE(fg_vk_tensor_write(s->key_q8,(t-1u)*FG_Q38_QSA_KEY_BYTES,
            records[t],FG_Q38_QSA_KEY_BYTES,&error)==FG_OK);
        REQUIRE(fg_vk_tensor_write(s->value_q4,(t-1u)*FG_Q38_QSA_VALUE_BYTES,
            records[t]+FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_VALUE_BYTES,&error)==FG_OK);
        REQUIRE(fg_vk_tensor_write(s->index_key_q8,(t-1u)*FG_Q38_QSA_INDEX_KEY_BYTES,
            records[t]+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,
            FG_Q38_QSA_INDEX_KEY_BYTES,&error)==FG_OK);
    }
    uint32_t slot=0;bool hit=false;
    REQUIRE(fg_qsa_page_cache_acquire(s->cache,3u,0u,&slot,&hit,&error)==FG_OK);
    REQUIRE(fg_vk_tensor_write(s->cache_records,(uint64_t)slot*FG_QSA_PAGE_RECORD_BYTES,
        records[0],FG_Q38_QSA_TOKEN_RECORD_BYTES,&error)==FG_OK);
    memset(fg_vk_tensor_map(s->query),0,(size_t)fg_vk_tensor_bytes(s->query));
    memset(fg_vk_tensor_map(s->gate),0,(size_t)fg_vk_tensor_bytes(s->gate));
    /* All six records exist before query 1 runs. Uniform attention must still
     * average only its visible prefix, including the partial four-token page. */
    REQUIRE(attend_prefill_tiles(s,0u,1u,5u,&error)==FG_OK);
    const float *got=fg_vk_tensor_map(s->attention);
    for(uint32_t q=0;q<5u;q++)for(uint32_t i=0;i<6144u;i++)
        REQUIRE(fabsf(got[q*6144u+i]-expected[q+1u])<2e-6f);
    REQUIRE(s->committed[0]==1u); /* Outer output projection owns publication. */
    REQUIRE(commit_prefill_records(s,0u,8u,1u,&error)==FG_ERR_LIMIT);
    REQUIRE(fg_vk_abort(fg_model_vk(s->model),&error)==FG_OK);
    REQUIRE(s->committed[0]==1u);
    s->committed[0]=0u;
    return 1;
}

/* Nonzero queries and gates exercise the whole split/merge chain against a
 * CPU softmax oracle.  The uniform test above cannot catch a broken score
 * reduction because every score stays zero when the query is zero. */
static int causal_attention_weighted(fg_qsa_session *s){
    fg_qsa_page_cache_reset(s->cache);
    s->committed[0]=1u;
    uint8_t records[6][FG_Q38_QSA_TOKEN_RECORD_BYTES];
    float key[512],value[512],index[128]={0},query[5u*6144u],gate[5u*6144u];
    float decoded_key[6][2][256],decoded_value[6][2][256];
    for(uint32_t t=0;t<6u;t++){
        for(uint32_t i=0;i<512u;i++){
            key[i]=0.9f*sinf((float)(t+1u)*(float)(i+1u)*0.013f);
            value[i]=cosf((float)(t+1u)*(float)(i+1u)*0.007f)+0.25f*sinf((float)(i+1u));
        }
        uint32_t position[]={t,t,t};
        fg_qsa_encode_full_token_record(key,value,index,position,records[t]);
        for(uint32_t kv=0;kv<2u;kv++){
            fg_dequantize_q8_0(records[t]+kv*272u,decoded_key[t][kv],256u);
            fg_dequantize_q8_0(records[t]+FG_Q38_QSA_KEY_BYTES+kv*272u,decoded_value[t][kv],256u);
        }
        if(!t)continue;
        REQUIRE(fg_vk_tensor_write(s->key_q8,(t-1u)*FG_Q38_QSA_KEY_BYTES,
            records[t],FG_Q38_QSA_KEY_BYTES,&error)==FG_OK);
        REQUIRE(fg_vk_tensor_write(s->value_q4,(t-1u)*FG_Q38_QSA_VALUE_BYTES,
            records[t]+FG_Q38_QSA_KEY_BYTES,FG_Q38_QSA_VALUE_BYTES,&error)==FG_OK);
        REQUIRE(fg_vk_tensor_write(s->index_key_q8,(t-1u)*FG_Q38_QSA_INDEX_KEY_BYTES,
            records[t]+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,
            FG_Q38_QSA_INDEX_KEY_BYTES,&error)==FG_OK);
    }
    uint32_t slot=0;bool hit=false;
    REQUIRE(fg_qsa_page_cache_acquire(s->cache,3u,0u,&slot,&hit,&error)==FG_OK);
    REQUIRE(fg_vk_tensor_write(s->cache_records,(uint64_t)slot*FG_QSA_PAGE_RECORD_BYTES,
        records[0],FG_Q38_QSA_TOKEN_RECORD_BYTES,&error)==FG_OK);
    for(uint32_t q=0;q<5u;q++)for(uint32_t h=0;h<24u;h++)for(uint32_t d=0;d<256u;d++){
        query[q*6144u+h*256u+d]=0.5f*cosf((float)(q+1u)*(float)(h*256u+d+1u)*0.0011f);
        gate[q*6144u+h*256u+d]=1.2f*sinf((float)(q+1u)*(float)(h+1u)*(float)(d+1u)*0.003f);
    }
    memcpy(fg_vk_tensor_map(s->query),query,sizeof(query));
    memcpy(fg_vk_tensor_map(s->gate),gate,sizeof(gate));
    memset(fg_vk_tensor_map(s->attention),0,(size_t)fg_vk_tensor_bytes(s->attention));
    REQUIRE(attend_prefill_tiles(s,0u,1u,5u,&error)==FG_OK);
    const float *got=fg_vk_tensor_map(s->attention);
    for(uint32_t q=0;q<5u;q++){
        uint32_t visible=q+2u; /* query slot q sees tokens 0..q+1 */
        for(uint32_t h=0;h<24u;h++){
            uint32_t kv=h/12u;
            float scores[6],maximum=-3.402823466e38f,sum=0.0f;
            for(uint32_t t=0;t<visible;t++){
                float dot=0.0f;
                for(uint32_t d=0;d<256u;d++)
                    dot+=query[q*6144u+h*256u+d]*decoded_key[t][kv][d];
                scores[t]=dot*0.0625f;
                maximum=fmaxf(maximum,scores[t]);
            }
            for(uint32_t t=0;t<visible;t++){scores[t]=expf(scores[t]-maximum);sum+=scores[t];}
            for(uint32_t d=0;d<256u;d++){
                float weighted=0.0f;
                for(uint32_t t=0;t<visible;t++)
                    weighted+=scores[t]/sum*decoded_value[t][kv][d];
                float expected=weighted/(1.0f+expf(-gate[q*6144u+h*256u+d]));
                float observed=got[q*6144u+h*256u+d];
                if(fabsf(observed-expected)>2e-4f*fmaxf(1.0f,fabsf(expected))){
                    fprintf(stderr,"line %d: weighted attention q=%u head=%u dim=%u GPU=%g CPU=%g\n",
                        __LINE__,q,h,d,observed,expected);
                    return 0;
                }
            }
        }
    }
    s->committed[0]=0u;
    return 1;
}

/* Directly drive the split/merge kernels at record counts that force multiple
 * eight-record online-softmax tiles and unbalanced (sometimes empty) splits.
 * The merge oracle is an exact full-softmax, which holds for any split
 * partition. */
static int split_merge_selected(fg_qsa_session *s,uint32_t selected,uint32_t splits){
    fg_vk_context *vk=fg_model_vk(s->model);
    if(fg_vk_batch_active(vk))REQUIRE(fg_vk_abort(vk,&error)==FG_OK);
    uint8_t *arena=malloc((size_t)selected*FG_Q38_QSA_TOKEN_RECORD_BYTES);
    float key[512],value[512],index[128]={0},query[6144],gate[6144];
    float *decoded_key=malloc((size_t)2u*selected*256u*sizeof(float));
    float *decoded_value=malloc((size_t)2u*selected*256u*sizeof(float));
    float *scores=malloc((size_t)selected*sizeof(float));
    if(!arena||!decoded_key||!decoded_value||!scores){
        free(arena);free(decoded_key);free(decoded_value);free(scores);
        return 0;
    }
    for(uint32_t t=0;t<selected;t++){
        for(uint32_t i=0;i<512u;i++){
            key[i]=0.8f*sinf((float)(t+1u)*(float)(i+1u)*0.0023f);
            value[i]=cosf((float)(t+1u)*(float)(i+1u)*0.0031f);
        }
        uint32_t position[]={t,t,t};
        fg_qsa_encode_full_token_record(key,value,index,position,
            arena+(uint64_t)t*FG_Q38_QSA_TOKEN_RECORD_BYTES);
        for(uint32_t kv=0;kv<2u;kv++){
            fg_dequantize_q8_0(arena+(uint64_t)t*FG_Q38_QSA_TOKEN_RECORD_BYTES+kv*272u,
                decoded_key+(uint64_t)(kv*selected+t)*256u,256u);
            fg_dequantize_q8_0(arena+(uint64_t)t*FG_Q38_QSA_TOKEN_RECORD_BYTES+
                FG_Q38_QSA_KEY_BYTES+kv*272u,
                decoded_value+(uint64_t)(kv*selected+t)*256u,256u);
        }
    }
    for(uint32_t h=0;h<24u;h++)for(uint32_t d=0;d<256u;d++){
        query[h*256u+d]=0.6f*sinf((float)(h+1u)*(float)(d+1u)*0.0053f);
        gate[h*256u+d]=1.5f*cosf((float)(h+1u)*(float)(d+1u)*0.0017f);
    }
    fg_vk_tensor *records=NULL,*query_tensor=NULL,*gate_tensor=NULL;
    fg_vk_tensor *counts=NULL,*partials=NULL,*output=NULL;
    uint32_t count=selected;
    bool ok=make_tensor(s,(uint64_t)selected*FG_Q38_QSA_TOKEN_RECORD_BYTES,
            &records,&error)==FG_OK&&
        make_tensor(s,sizeof(query),&query_tensor,&error)==FG_OK&&
        make_tensor(s,sizeof(gate),&gate_tensor,&error)==FG_OK&&
        make_tensor(s,4u,&counts,&error)==FG_OK&&
        make_tensor(s,24u*splits*258u*4u,&partials,&error)==FG_OK&&
        make_tensor(s,sizeof(query),&output,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_write(records,0,arena,
        (uint64_t)selected*FG_Q38_QSA_TOKEN_RECORD_BYTES,&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_write(query_tensor,0,query,sizeof(query),&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_write(gate_tensor,0,gate,sizeof(gate),&error)==FG_OK;
    if(ok)ok=fg_vk_tensor_write(counts,0,&count,4u,&error)==FG_OK;
    if(ok)ok=fg_vk_begin(vk,&error)==FG_OK;
    if(ok)ok=fg_vk_qsa_attention_split_batch(vk,partials,records,query_tensor,counts,
        1u,selected,6144u,splits,&error)==FG_OK;
    if(ok)ok=fg_vk_qsa_attention_merge_batch(vk,output,partials,gate_tensor,
        1u,6144u,splits,&error)==FG_OK;
    if(ok)ok=fg_qsa_submit_host_reads(vk,&error)==FG_OK;
    if(ok){
        const float *got=fg_vk_tensor_map(output);
        for(uint32_t h=0;ok&&h<24u;h++){
            uint32_t kv=h/12u;
            float maximum=-3.402823466e38f,sum=0.0f;
            for(uint32_t t=0;t<selected;t++){
                const float *krow=decoded_key+(uint64_t)(kv*selected+t)*256u;
                float dot=0.0f;
                for(uint32_t d=0;d<256u;d++)dot+=query[h*256u+d]*krow[d];
                scores[t]=dot*0.0625f;
                maximum=fmaxf(maximum,scores[t]);
            }
            for(uint32_t t=0;t<selected;t++){scores[t]=expf(scores[t]-maximum);sum+=scores[t];}
            for(uint32_t d=0;d<256u;d++){
                float weighted=0.0f;
                for(uint32_t t=0;t<selected;t++)
                    weighted+=scores[t]/sum*
                        decoded_value[(uint64_t)(kv*selected+t)*256u+d];
                float expected=weighted/(1.0f+expf(-gate[h*256u+d]));
                float observed=got[h*256u+d];
                if(fabsf(observed-expected)>2e-4f*fmaxf(1.0f,fabsf(expected))){
                    fprintf(stderr,"line %d: split/merge selected=%u splits=%u head=%u "
                        "dim=%u GPU=%g CPU=%g\n",__LINE__,selected,splits,h,d,observed,expected);
                    ok=0;break;
                }
            }
        }
    }
    fg_vk_tensor_destroy(output);fg_vk_tensor_destroy(partials);fg_vk_tensor_destroy(counts);
    fg_vk_tensor_destroy(gate_tensor);fg_vk_tensor_destroy(query_tensor);
    fg_vk_tensor_destroy(records);
    free(arena);free(decoded_key);free(decoded_value);free(scores);
    return ok;
}

int main(void){
    fg_model *model=calloc(1,sizeof(*model));
    fg_qsa_session *session=calloc(1,sizeof(*session));fg_vk_tensor *scratch=NULL;
    if(!model||!session)return 1;
    fg_status status=fg_vk_open(&model->vk,&error);
    if(status==FG_ERR_UNAVAILABLE){free(session);free(model);return 77;}
    int ok=status==FG_OK&&setup(session,model,&scratch);
    if(ok)ok=lazy_index_segment(session)&&fill_index_rows(session);
    if(ok)ok=selection_window_boundary(session);
    if(ok)ok=selection_parity(session,131071u)&&selection_parity(session,2045u);
    if(ok)ok=causal_attention(session)&&causal_attention_weighted(session)&&gather_eviction(session,false)&&gather_eviction(session,true);
    if(ok)ok=split_merge_selected(session,5u,8u)&&split_merge_selected(session,100u,1u)&&
        split_merge_selected(session,2051u,8u);
    if(fg_vk_batch_active(model->vk))fg_vk_abort(model->vk,&error);
    fg_qsa_session_close(session);fg_vk_tensor_destroy(scratch);
    fg_vk_tensor_destroy(model->norm);fg_vk_close(model->vk);free(model);
    printf("QSA causal tiles, 128k segment boundary, cache eviction and deduplicated fetch: %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}
