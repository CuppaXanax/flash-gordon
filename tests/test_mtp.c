#include "fg_mtp.h"
#include "fg_protocol.h"
#include "fg_qsa_state.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

typedef struct synthetic {
    uint32_t gdn,ple,kv,tokens;
    struct {uint32_t gdn,ple,kv,tokens;} saved;
    uint32_t fail;
} synthetic;

static uint32_t decode(synthetic *s,uint32_t token){
    s->gdn=s->gdn*1664525u+token+1013904223u;
    s->ple=(s->ple<<3u)^token;
    s->kv=token;s->tokens++;
    return (s->gdn^(s->ple>>4u))%23u;
}
static fg_status checkpoint(void *opaque,fg_error *err){
    synthetic *s=opaque;(void)err;
    s->saved.gdn=s->gdn;s->saved.ple=s->ple;
    s->saved.kv=s->kv;s->saved.tokens=s->tokens;
    return FG_OK;
}
static fg_status restore(void *opaque,fg_error *err){
    synthetic *s=opaque;(void)err;
    s->gdn=s->saved.gdn;s->ple=s->saved.ple;
    s->kv=s->saved.kv;s->tokens=s->saved.tokens;
    return FG_OK;
}
static fg_status verify(void *opaque,const uint32_t *inputs,uint32_t count,
                        uint32_t *targets,fg_error *err){
    synthetic *s=opaque;(void)err;
    for(uint32_t i=0;i<count;i++)targets[i]=decode(s,inputs[i]);
    return s->fail==1u?FG_ERR_IO:FG_OK;
}
static fg_status replay(void *opaque,const uint32_t *inputs,uint32_t count,fg_error *err){
    synthetic *s=opaque;(void)err;
    for(uint32_t i=0;i<count;i++)(void)decode(s,inputs[i]);
    return s->fail==2u?FG_ERR_IO:FG_OK;
}
static fg_status commit(void *opaque,fg_error *err){
    synthetic *s=opaque;(void)err;
    return s->fail==3u?FG_ERR_IO:FG_OK;
}

static void cycles(void){
    fg_error err={0};
    for(uint32_t k=1u;k<=FG_MTP_MAX_DRAFT;k++)
    for(uint32_t rejection=0;rejection<=k;rejection++)
    for(uint32_t limit=1u;limit<=k+1u;limit++){
        synthetic base={.gdn=7u,.ple=3u},trial=base,oracle=base;
        uint32_t pending=5u,drafts[FG_MTP_MAX_DRAFT],t=pending;
        for(uint32_t i=0;i<k;i++){t=decode(&oracle,t);drafts[i]=t;}
        if(rejection<k)drafts[rejection]=(drafts[rejection]+1u)%23u;
        fg_mtp_ops ops={checkpoint,verify,restore,replay,commit,&trial};
        fg_mtp_result result={0};
        CHECK(fg_mtp_cycle_greedy(&ops,pending,drafts,k,limit,UINT32_MAX,&result,&err)==FG_OK);
        CHECK(result.count<=limit&&result.count>=1u);
        t=pending;
        for(uint32_t i=0;i<result.count;i++){
            t=decode(&base,t);CHECK(result.tokens[i]==t);
        }
        CHECK(trial.gdn==base.gdn&&trial.ple==base.ple&&trial.kv==base.kv&&trial.tokens==base.tokens);
        CHECK(result.consumed==result.count);
        CHECK(result.accepted==(rejection<limit?rejection:limit));
    }
    for(uint32_t fail=1u;fail<=3u;fail++){
        synthetic trial={.gdn=7u,.ple=3u,.fail=fail};
        uint32_t drafts[]={24u,25u,26u};fg_mtp_result result={0};
        fg_mtp_ops ops={checkpoint,verify,restore,replay,commit,&trial};
        CHECK(fg_mtp_cycle_greedy(&ops,5u,drafts,3u,4u,UINT32_MAX,&result,&err)==FG_ERR_IO);
        CHECK(trial.gdn==7u&&trial.ple==3u&&trial.tokens==0u&&result.count==0u);
    }
    synthetic trial={.gdn=7u,.ple=3u},oracle=trial;
    uint32_t eos=decode(&oracle,5u),drafts[]={eos,1u,2u};
    fg_mtp_ops ops={checkpoint,verify,restore,replay,commit,&trial};fg_mtp_result result={0};
    CHECK(fg_mtp_cycle_greedy(&ops,5u,drafts,3u,4u,eos,&result,&err)==FG_OK);
    CHECK(result.count==1u&&result.tokens[0]==eos&&trial.tokens==1u);
}

static void sampling(void){
    fg_error err={0};bool accepted=false;uint32_t token=99u;
    const float p[]={.25f,.75f},q[]={.75f,.25f};
    CHECK(fg_mtp_rejection_sample(p,q,2u,0u,.2f,.5f,&accepted,&token,&err)==FG_OK&&accepted&&token==0u);
    CHECK(fg_mtp_rejection_sample(p,q,2u,0u,.8f,.5f,&accepted,&token,&err)==FG_OK&&!accepted&&token==1u);
    CHECK(fg_mtp_rejection_sample(p,q,2u,1u,.99f,.5f,&accepted,&token,&err)==FG_OK&&accepted&&token==1u);
    CHECK(fg_mtp_rejection_sample(p,p,2u,0u,.99f,.5f,&accepted,&token,&err)==FG_OK&&accepted);
    const float disjoint_p[]={0.0f,1.0f},disjoint_q[]={1.0f,0.0f};
    CHECK(fg_mtp_rejection_sample(disjoint_p,disjoint_q,2u,0u,0.0f,0.0f,&accepted,&token,&err)==FG_OK&&!accepted&&token==1u);
    CHECK(fg_mtp_rejection_sample(p,q,2u,0u,1.0f,.5f,&accepted,&token,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_mtp_rejection_sample(p,disjoint_q,2u,1u,.5f,.5f,&accepted,&token,&err)==FG_ERR_ARGUMENT);
    /* Integrate independent uniform draws on an exact rational grid. */
    uint32_t histogram[2]={0};
    for(uint32_t d=0;d<4u;d++)for(uint32_t a=0;a<12u;a++){
        uint32_t proposal=d<3u?0u:1u;
        CHECK(fg_mtp_rejection_sample(p,q,2u,proposal,((float)a+.5f)/12.0f,
              .5f,&accepted,&token,&err)==FG_OK);
        histogram[token]++;
    }
    CHECK(histogram[0]==12u&&histogram[1]==36u);
}

static void state_truncation(void){
    char path[128];
    snprintf(path,sizeof(path),".mtp-state-test-%ld.fgqs",(long)getpid());
    const uint8_t layers[]={3u};fg_qsa_state *state=NULL;fg_error err={0};
    fg_status status=fg_qsa_state_open(&state,path,layers,1u,16u,true,&err);
    CHECK(status==FG_OK);
    if(status!=FG_OK)return;
    uint8_t first[FG_QSA_PAGE_RECORD_BYTES],second[FG_QSA_PAGE_RECORD_BYTES];
    for(uint32_t i=0;i<sizeof(first);i++){first[i]=(uint8_t)(i*17u);second[i]=(uint8_t)(i*13u);}
    CHECK(fg_qsa_state_write_block(state,0u,0u,first,4u,&err)==FG_OK);
    CHECK(fg_qsa_state_write_block(state,0u,1u,second,4u,&err)==FG_OK);
    CHECK(fg_qsa_state_truncate(state,0u,3u,&err)==FG_OK);
    CHECK(fg_qsa_state_layer_tokens(state,0u)==3u);
    fg_qsa_state_close(state);state=NULL;
    CHECK(fg_qsa_state_open(&state,path,layers,1u,16u,false,&err)==FG_OK);
    if(state){
        CHECK(fg_qsa_state_layer_tokens(state,0u)==3u);
        uint8_t got[FG_QSA_PAGE_RECORD_BYTES];uint32_t committed=0u;
        CHECK(fg_qsa_state_read_block(state,0u,0u,got,&committed,&err)==FG_OK);
        CHECK(committed==3u&&!memcmp(first,got,3u*FG_Q38_QSA_TOKEN_RECORD_BYTES));
        CHECK(fg_qsa_state_write_block(state,0u,0u,first,4u,&err)==FG_OK);
        CHECK(fg_qsa_state_truncate(state,0u,0u,&err)==FG_OK);
        CHECK(fg_qsa_state_truncate(state,0u,1u,&err)==FG_ERR_ARGUMENT);
        fg_qsa_state_close(state);
    }
    CHECK(unlink(path)==0);
}

int main(void){
    cycles();sampling();state_truncation();
    if(!failures)puts("MTP greedy prefix rollback and rejection sampling: PASS");
    return failures?1:0;
}
