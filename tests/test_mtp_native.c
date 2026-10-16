#include "fg_mtp_native.h"
#include "fg_mtp_embedding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fg_vk_tensor {uint64_t bytes;float *values;};
struct fg_mtp_embedding {int unused;};
struct fg_mtp_head {
    uint32_t slot,frontier,first;
    fg_vk_tensor *next,*hidden;
};
typedef struct warm_record {uint32_t slot,cell;float hyper,embedding;} warm_record;
static warm_record warmed[128];
static uint32_t warm_count,opens,forks,admission_checks;
static bool admission_rejected;
static unsigned failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

fg_status fg_mtp_embedding_open(fg_mtp_embedding **out,const fg_manifest *m,
    const char *path,fg_error *err){
    (void)m;(void)path;(void)err;*out=calloc(1,sizeof(**out));return *out?FG_OK:FG_ERR_OOM;
}
fg_status fg_mtp_embedding_row(fg_mtp_embedding *e,uint32_t token,float *values,fg_error *err){
    (void)e;(void)err;memset(values,0,FG_HIDDEN_SIZE*sizeof(float));values[0]=(float)token;
    return token<FG_Q38_VOCAB_SIZE?FG_OK:FG_ERR_ARGUMENT;
}
void fg_mtp_embedding_close(fg_mtp_embedding *e){free(e);}

fg_status fg_vk_tensor_create(fg_vk_context *vk,uint64_t bytes,fg_vk_tensor **out,fg_error *err){
    (void)vk;(void)err;*out=calloc(1,sizeof(**out));if(!*out)return FG_ERR_OOM;
    (*out)->bytes=bytes;(*out)->values=calloc(1,(size_t)bytes);
    return (*out)->values?FG_OK:FG_ERR_OOM;
}
void fg_vk_tensor_destroy(fg_vk_tensor *t){if(t){free(t->values);free(t);}}
fg_status fg_vk_tensor_write(fg_vk_tensor *t,uint64_t at,const void *p,uint64_t bytes,fg_error *err){
    (void)err;if(!t||at>t->bytes||bytes>t->bytes-at)return FG_ERR_LIMIT;
    memcpy((uint8_t *)t->values+at,p,(size_t)bytes);return FG_OK;
}
fg_status fg_vk_tensor_read(const fg_vk_tensor *t,uint64_t at,void *p,uint64_t bytes,fg_error *err){
    (void)err;if(!t||at>t->bytes||bytes>t->bytes-at)return FG_ERR_LIMIT;
    memcpy(p,(uint8_t *)t->values+at,(size_t)bytes);return FG_OK;
}
static fg_status mock_head(fg_mtp_head **out,uint32_t slot,fg_error *err){
    *out=calloc(1,sizeof(**out));if(!*out)return FG_ERR_OOM;(*out)->slot=slot;
    fg_status status=fg_vk_tensor_create(NULL,FG_HYPER_WIDTH*4u,&(*out)->next,err);
    if(status==FG_OK)status=fg_vk_tensor_create(NULL,FG_HIDDEN_SIZE*4u,&(*out)->hidden,err);
    return status;
}
fg_status fg_mtp_head_open_windowed(fg_mtp_head **out,fg_vk_context *vk,
    const fg_manifest *m,uint32_t rank,const char *path,uint32_t context,
    uint32_t window,fg_error *err){
    (void)vk;(void)m;(void)rank;(void)path;(void)context;(void)window;
    opens++;return mock_head(out,0u,err);
}
fg_status fg_mtp_head_fork(fg_mtp_head **out,fg_mtp_head *source,uint32_t slot,fg_error *err){
    CHECK(source!=NULL);forks++;return mock_head(out,slot,err);
}
fg_status fg_mtp_head_check_residency(const fg_mtp_head *head,fg_error *err){
    (void)err;CHECK(head!=NULL);admission_checks++;
    return admission_rejected?FG_ERR_LIMIT:FG_OK;
}
void fg_mtp_head_close(fg_mtp_head *h){
    if(h){fg_vk_tensor_destroy(h->next);fg_vk_tensor_destroy(h->hidden);free(h);}
}
uint32_t fg_mtp_head_frontier(const fg_mtp_head *h){return h->frontier;}
fg_status fg_mtp_head_reset(fg_mtp_head *h,uint32_t cell,fg_error *err){
    (void)err;h->frontier=cell;h->first=cell;return FG_OK;
}
fg_status fg_mtp_head_truncate(fg_mtp_head *h,uint32_t cell,fg_error *err){
    (void)err;if(cell<h->first||cell>h->frontier)return FG_ERR_ARGUMENT;
    h->frontier=cell;return FG_OK;
}
fg_status fg_mtp_head_warm_replay(fg_mtp_head *h,const fg_vk_tensor *hc,
    const fg_vk_tensor *embedding,uint32_t cell,fg_error *err){
    (void)err;if(cell!=h->frontier)return FG_ERR_MISMATCH;
    CHECK(warm_count<128u);
    warmed[warm_count++]=(warm_record){h->slot,cell,hc->values[0],embedding->values[0]};
    h->frontier++;return FG_OK;
}
fg_status fg_mtp_head_step_replay(fg_mtp_head *h,const fg_vk_tensor *hc,
    const fg_vk_tensor *embedding,uint32_t cell,fg_vk_tensor **next,
    fg_vk_tensor **hidden,fg_error *err){
    (void)embedding;(void)err;if(cell!=h->frontier)return FG_ERR_MISMATCH;
    float input=hc->values[0];
    h->next->values[0]=input+10000.0f;
    h->hidden->values[0]=input;h->frontier++;
    *next=h->next;*hidden=h->hidden;return FG_OK;
}
static void hyper(float *p,uint32_t first,uint32_t count,uint32_t slot){
    memset(p,0,(size_t)count*FG_HYPER_WIDTH*sizeof(float));
    for(uint32_t i=0;i<count;i++)p[(uint64_t)i*FG_HYPER_WIDTH]=(float)(slot*1000u+first+i);
}
static void embeddings(uint32_t *p,uint32_t first,uint32_t count,uint32_t slot){
    for(uint32_t i=0;i<count;i++)p[i]=slot*1000u+first+i;
}
static fg_status run(fg_mtp_native *n,uint32_t slot,fg_mtp_native_operation op,
    uint32_t first,uint32_t count,uint32_t argument,uint32_t limit,
    const uint32_t *embedding,float *hidden,fg_mtp_native_ack *ack,fg_error *err){
    fg_mtp_native_work w={.operation=op,.first_token=first,.count=count,
        .argument=argument,.limit=limit,.tokens=embedding};
    return fg_mtp_native_run(n,slot,&w,hidden,ack,err);
}
int main(void){
    fg_error err={0};fg_manifest *m=calloc(1,sizeof(*m));CHECK(m);
    if(!m)return 1;
    m->layer_owner[FG_LAYER_COUNT-1u]=7u;
    fg_mtp_native *n=NULL,*other=NULL;fg_vk_context *vk=(fg_vk_context *)(uintptr_t)1u;
    CHECK(fg_mtp_native_create(&n,vk,m,7u,&err)==FG_OK);
    CHECK(fg_mtp_native_create(&other,vk,m,1u,&err)==FG_OK);
    CHECK(!fg_mtp_native_enabled(n)&&opens==0u);
    float hc[6u*FG_HYPER_WIDTH],hidden[FG_HIDDEN_SIZE];
    uint32_t embed[6],next_tokens[6]={1u,2u,3u,4u,5u,UINT32_MAX};
    fg_mtp_native_ack ack;
    fg_mtp_native_work cfg={.operation=FG_MTP_NATIVE_CONFIG,.overlay="native.gguf",
        .embedding_file="embedding.fgme",.count=11u,.argument=8192u,.limit=3u};
    CHECK(fg_mtp_native_run(other,0u,&cfg,hidden,&ack,&err)==FG_OK&&opens==0u);
    CHECK(fg_mtp_native_run(n,0u,&cfg,hidden,&ack,&err)==FG_OK&&opens==1u);
    CHECK(run(n,0u,FG_MTP_NATIVE_PREPARE,0u,0u,1u,0u,NULL,NULL,&ack,&err)==FG_OK);
    CHECK(admission_checks==1u&&forks==0u);
    CHECK(run(n,0u,FG_MTP_NATIVE_PREPARE,0u,0u,2u,0u,NULL,NULL,&ack,&err)==FG_OK);
    CHECK(admission_checks==2u&&forks==1u);
    admission_rejected=true;
    CHECK(run(n,0u,FG_MTP_NATIVE_PREPARE,0u,0u,2u,0u,NULL,NULL,&ack,&err)==FG_ERR_LIMIT);
    admission_rejected=false;
    CHECK(run(n,0u,FG_MTP_NATIVE_PREPARE,0u,0u,2u,0u,NULL,NULL,&ack,&err)==FG_OK);
    CHECK(forks==1u&&opens==1u);
    fg_mtp_native_work malformed={.operation=FG_MTP_NATIVE_DRAFT,.count=1u};
    CHECK(fg_mtp_native_run(n,0u,&malformed,hidden,&ack,&err)!=FG_OK);
    malformed=(fg_mtp_native_work){.operation=FG_MTP_NATIVE_CONFIG};
    CHECK(fg_mtp_native_run(n,0u,&malformed,NULL,&ack,&err)!=FG_OK);
    CHECK(run(n,0u,FG_MTP_NATIVE_PREFIX,0u,0u,2u,6u,NULL,hidden,&ack,&err)==FG_OK);
    hyper(hc,0u,6u,0u);
    CHECK(fg_mtp_native_observe_prefill(n,0u,0u,3u,hc,next_tokens,&err)==FG_OK);
    CHECK(fg_mtp_native_observe_prefill(n,0u,3u,3u,hc+3u*FG_HYPER_WIDTH,next_tokens+3u,&err)==FG_OK);
    CHECK(warmed[0].cell==2u&&warmed[0].hyper==2.0f&&warmed[0].embedding==3.0f);
    CHECK(warm_count==3u);
    embeddings(embed,6u,3u,0u);
    for(uint32_t j=0;j<3u;j++){
        CHECK(run(n,0u,FG_MTP_NATIVE_DRAFT,6u+j,1u,j,0u,
            embed+j,hidden,&ack,&err)==FG_OK);
        CHECK(hidden[0]==5.0f+10000.0f*(float)j);
    }
    /* Interleave another session while slot zero has an outstanding draft. */
    CHECK(run(n,1u,FG_MTP_NATIVE_PREFIX,0u,0u,3u,5u,NULL,hidden,&ack,&err)==FG_OK&&forks==1u);
    hyper(hc,0u,5u,1u);
    embeddings(next_tokens,1u,4u,1u);next_tokens[4]=UINT32_MAX;
    CHECK(fg_mtp_native_observe_prefill(n,1u,0u,5u,hc,next_tokens,&err)==FG_OK);
    CHECK(warmed[3].slot==1u&&warmed[3].hyper==1003.0f&&warmed[3].embedding==1004.0f);
    hyper(hc,6u,4u,0u);
    CHECK(fg_mtp_native_observe_verify(n,0u,6u,4u,hc,&err)==FG_OK);
    /* The retained replay may replace the speculative batch's HC prefix. */
    hc[0]=66.0f;
    CHECK(fg_mtp_native_observe_verify(n,0u,6u,2u,hc,&err)==FG_OK);
    embeddings(embed,7u,1u,0u);
    CHECK(run(n,0u,FG_MTP_NATIVE_FINISH,6u,2u,0u,0u,embed,hidden,&ack,&err)==FG_OK);
    CHECK(ack.frontier==7u&&warmed[4].cell==6u&&warmed[4].hyper==66.0f&&
          warmed[4].embedding==7.0f);
    embeddings(embed,8u,1u,0u);
    CHECK(run(n,0u,FG_MTP_NATIVE_DRAFT,8u,1u,0u,0u,embed,hidden,&ack,&err)==FG_OK);
    CHECK(hidden[0]==7.0f);
    hyper(hc,8u,1u,0u);
    CHECK(fg_mtp_native_observe_verify(n,0u,8u,1u,hc,&err)==FG_OK);
    CHECK(run(n,0u,FG_MTP_NATIVE_FINISH,8u,1u,0u,0u,NULL,hidden,&ack,&err)==FG_OK);
    CHECK(ack.frontier==8u);
    hyper(hc,9u,1u,0u);embeddings(embed,9u,1u,0u);
    CHECK(fg_mtp_native_observe_decode(n,0u,9u,hc,&err)==FG_OK);
    CHECK(run(n,0u,FG_MTP_NATIVE_ADVANCE,9u,1u,0u,0u,embed,hidden,&ack,&err)==FG_OK);
    CHECK(ack.frontier==9u&&warmed[5].cell==8u&&warmed[5].hyper==8.0f);
    embeddings(embed,10u,1u,0u);
    CHECK(run(n,0u,FG_MTP_NATIVE_PREFIX,10u,1u,7u,12u,embed,hidden,&ack,&err)==FG_OK);
    CHECK(ack.frontier==10u&&warmed[6].cell==9u&&warmed[6].hyper==9.0f);
    hyper(hc,10u,2u,0u);
    next_tokens[0]=11u;next_tokens[1]=UINT32_MAX;
    CHECK(fg_mtp_native_observe_prefill(n,0u,10u,2u,hc,next_tokens,&err)==FG_OK);
    CHECK(warmed[7].cell==10u&&warmed[7].hyper==10.0f&&warmed[7].embedding==11.0f);
    CHECK(fg_mtp_native_observe_prefill(n,0u,10u,2u,hc,next_tokens,&err)!=FG_OK);
    CHECK(fg_mtp_native_reset(n,0u,&err)==FG_OK);
    CHECK(run(n,0u,FG_MTP_NATIVE_DRAFT,12u,1u,0u,0u,embed,hidden,&ack,&err)!=FG_OK);
    embeddings(embed,1005u,1u,0u);
    CHECK(run(n,1u,FG_MTP_NATIVE_DRAFT,5u,1u,0u,0u,embed,hidden,&ack,&err)==FG_OK);
    CHECK(hidden[0]==1004.0f);
    fg_mtp_native_close(other);fg_mtp_native_close(n);free(m);
    if(!failures)puts("MTP causal warmfill, authoritative repair, continuation and independent sessions: PASS");
    return failures?1:0;
}
