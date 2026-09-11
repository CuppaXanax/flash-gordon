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
    /* Repeated quantized rows are inexpensive to build; MRoPE makes block scores distinct. */
    uint8_t rows[64][136];float raw[128];
    for(uint32_t r=0;r<64u;r++){
        for(uint32_t i=0;i<128u;i++)raw[i]=sinf((float)(r+1u)*(float)(i+1u)*0.013f);
        fg_quantize_q8_0(raw,rows[r],128u);
    }
    for(uint32_t seg=0;seg<s->index_segment_count;seg++){
        uint8_t *keys=fg_vk_tensor_map(s->index_keys[0][seg]);
        for(uint32_t t=0;t<s->index_segment_tokens[seg];t++)memcpy(keys+(uint64_t)t*136u,rows[t%64u],136u);
    }
    float *query=fg_vk_tensor_map(s->index_query);
    for(uint32_t q=0;q<128u;q++)for(uint32_t i=0;i<512u;i++)
        query[q*512u+i]=cosf((float)(q+1u)*(float)(i+1u)*0.023f);
    return 1;
}

static int selection_parity(fg_qsa_session *s,uint32_t first){
    uint32_t selected[4][512],counts[4],reference[512],count=0;
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
    uint32_t selected[4][512]={{0u,1u,2u},{2u,0u,3u},{1u,3u,4u},{4u,2u,0u}};
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

int main(void){
    fg_model *model=calloc(1,sizeof(*model));
    fg_qsa_session *session=calloc(1,sizeof(*session));fg_vk_tensor *scratch=NULL;
    if(!model||!session)return 1;
    fg_status status=fg_vk_open(&model->vk,&error);
    if(status==FG_ERR_UNAVAILABLE){free(session);free(model);return 77;}
    int ok=status==FG_OK&&setup(session,model,&scratch);
    if(ok)ok=selection_parity(session,131071u)&&selection_parity(session,2045u);
    if(ok)ok=causal_attention(session)&&gather_eviction(session,false)&&gather_eviction(session,true);
    if(fg_vk_batch_active(model->vk))fg_vk_abort(model->vk,&error);
    fg_qsa_session_close(session);fg_vk_tensor_destroy(scratch);
    fg_vk_tensor_destroy(model->norm);fg_vk_close(model->vk);free(model);
    printf("QSA causal tiles, 128k segment boundary, cache eviction and deduplicated fetch: %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}
