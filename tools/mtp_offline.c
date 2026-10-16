#include "fg_chat.h"
#include "fg_expert.h"
#include "fg_ngram.h"
#include "fg_output.h"
#include "fg_owner.h"
#include "fg_qsa_state.h"
#include "fg_sha256.h"
#include "fg_tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum{MAX_CASES=16,MAX_TOKENS=256,MAX_ROUNDS=128};
typedef struct round_plan{uint32_t first,n,keep;} round_plan;
typedef struct sample{
    char name[128],prompt[4096];
    uint32_t prompt_tokens,divergence,expected_on,generated_count,generated[128];
    uint32_t round_count;
    round_plan rounds[MAX_ROUNDS];
} sample;
typedef struct prefill_context{
    fg_expert_executor *expert;
    uint32_t rank;
    fg_prefill_pair pairs[128u*FG_TOP_K];
    fg_prefill_result_pair results[128u*FG_TOP_K];
    float output[128u*FG_HIDDEN_SIZE];
} prefill_context;
static fg_error error;
static void need(bool yes,const char *why){
    if(!yes){fprintf(stderr,"MTP offline: %s\n",why);exit(1);}
}
static void check(fg_status status){if(status!=FG_OK)need(false,error.message);}
static void read_data(FILE *f,void *data,size_t bytes){need(fread(data,1,bytes,f)==bytes,"read");}
static void write_data(FILE *f,const void *data,size_t bytes){need(fwrite(data,1,bytes,f)==bytes,"write");}
static uint32_t u32(FILE *f){uint32_t value;read_data(f,&value,4);return value;}
static double now_ms(void){
    struct timespec t;need(!clock_gettime(CLOCK_MONOTONIC,&t),"clock");
    return (double)t.tv_sec*1000.+(double)t.tv_nsec*1e-6;
}
static FILE *file(const char *prefix,const char *suffix,const char *mode){
    char path[4096];need(snprintf(path,sizeof(path),"%s.%s",prefix,suffix)<(int)sizeof(path),"path");
    FILE *f=fopen(path,mode);need(f!=NULL,path);return f;
}
static uint32_t load_plan(const char *path,sample *cases){
    FILE *f=fopen(path,"rb");need(f!=NULL,"plan");char magic[8];read_data(f,magic,8);
    need(!memcmp(magic,"FGMDIAG1",8),"plan magic");
    uint32_t count=u32(f);need(count&&count<=MAX_CASES,"case count");
    for(uint32_t c=0;c<count;c++){
        sample *s=&cases[c];uint32_t length=u32(f);need(length&&length<sizeof(s->name),"name");
        read_data(f,s->name,length);s->name[length]=0;
        length=u32(f);need(length&&length<sizeof(s->prompt),"prompt");
        read_data(f,s->prompt,length);s->prompt[length]=0;
        s->prompt_tokens=u32(f);s->divergence=u32(f);s->expected_on=u32(f);
        s->generated_count=u32(f);need(s->prompt_tokens&&s->prompt_tokens<=128u&&s->generated_count<=128u&&
            s->divergence&&s->divergence<s->generated_count&&
            s->prompt_tokens+s->generated_count<=MAX_TOKENS,"generation bounds");
        read_data(f,s->generated,s->generated_count*4u);
        s->round_count=u32(f);need(s->round_count&&s->round_count<=MAX_ROUNDS,"round count");
        uint32_t frontier=s->prompt_tokens;
        for(uint32_t r=0;r<s->round_count;r++){
            round_plan *p=&s->rounds[r];p->first=u32(f);p->n=u32(f);p->keep=u32(f);
            need(p->first==frontier&&p->n>=2u&&p->n<=4u&&p->keep&&p->keep<=p->n&&
                p->first+p->n<=s->prompt_tokens+s->generated_count,"round bounds");
            frontier+=p->keep;
        }
        round_plan *last=&s->rounds[s->round_count-1u];
        need(last->first<=s->prompt_tokens+s->divergence-1u&&
            last->first+last->keep>s->prompt_tokens+s->divergence-1u,"divergence not in final round");
    }
    need(fgetc(f)==EOF,"plan trailing bytes");fclose(f);return count;
}
static fg_manifest *manifest(const char *path){
    fg_manifest *m=malloc(sizeof(*m));need(m!=NULL,"manifest allocation");
    check(fg_manifest_read(path,m,&error));return m;
}
static fg_status expert(void *opaque,uint32_t layer,const fg_vk_tensor *activation,
    const fg_vk_tensor *router,fg_vk_tensor **output,fg_error *err){
    return fg_expert_decode_chain(opaque,layer,activation,router,output,err);
}
static fg_status small_expert(void *opaque,uint32_t layer,const fg_vk_tensor *activation,
    const fg_vk_tensor *router,uint32_t count,fg_vk_tensor **output,fg_error *err){
    return fg_expert_decode_small_n(opaque,layer,activation,router,count,output,err);
}
static fg_status unexpected_fetch(void *opaque,uint32_t layer,const uint32_t *blocks,
    uint32_t count,uint8_t *records,fg_error *err){
    (void)opaque;(void)layer;(void)blocks;(void)count;(void)records;
    fg_error_set(err,FG_ERR_UNAVAILABLE,"owned-only offline QSA attempted a remote fetch");
    return FG_ERR_UNAVAILABLE;
}
static fg_status batch_expert(void *opaque,uint32_t layer,uint32_t first,uint16_t count,
    const uint16_t *ids,const float *gates,const uint8_t *activation,
    fg_owner_prefill_shared_fn shared,void *shared_context,fg_prefill_result results[FG_GROUP_SIZE],
    uint32_t *result_count,fg_error *err){
    prefill_context *c=opaque;
    for(uint32_t t=0;t<count;t++)for(uint32_t k=0;k<FG_TOP_K;k++)
        c->pairs[t*FG_TOP_K+k]=(fg_prefill_pair){.token_slot=(uint16_t)t,
            .routing_slot=(uint16_t)k,.expert_id=ids[t*FG_TOP_K+k],.gate=gates[t*FG_TOP_K+k]};
    fg_prefill_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)c->rank,
        .destination_rank=(uint8_t)c->rank,.first_position=first,.token_count=count,
        .pair_count=(uint16_t)(count*FG_TOP_K),.pairs=c->pairs,.activations_q8k=(uint8_t *)activation};
    fg_status status=shared(shared_context,err);
    if(status==FG_OK)status=fg_expert_prefill(c->expert,&work,&results[0],c->results,
        128u*FG_TOP_K,c->output,128u*FG_HIDDEN_SIZE,err);
    if(status==FG_OK)*result_count=1u;
    return status;
}
static uint32_t prepare_tokens(fg_tokenizer *t,const sample *s,uint32_t tokens[MAX_TOKENS]){
    need(!strncmp(s->prompt,"/no_think ",10),"frozen prompt directive");
    fg_chat_message message={.role="user",.content=s->prompt+10};
    fg_chat_render_options options={.think_mode=FG_CHAT_THINK_OFF};
    char *rendered=NULL;check(fg_chat_render(&message,1u,&options,&rendered,&error));
    fg_tokens prompt={0};check(fg_tokenizer_encode(t,rendered,true,&prompt,&error));free(rendered);
    need(prompt.count==s->prompt_tokens,"historical prompt token count differs");
    memcpy(tokens,prompt.data,prompt.count*4u);
    memcpy(tokens+prompt.count,s->generated,s->generated_count*4u);
    uint32_t count=(uint32_t)prompt.count+s->generated_count;fg_tokens_free(&prompt);return count;
}
static void prepare(const char *mp,const char *pack,const char *plan,const char *out){
    sample *cases=calloc(MAX_CASES,sizeof(*cases));need(cases!=NULL,"cases");
    uint32_t count=load_plan(plan,cases);fg_manifest *m=manifest(mp);
    fg_tokenizer *tok=NULL;check(fg_tokenizer_open(&tok,pack,m,&error));
    fg_model *model=NULL;check(fg_model_open(&model,m,pack,0u,&error));
    fg_vk_context *vk=fg_model_vk(model);fg_vk_tensor *input=NULL;
    check(fg_vk_tensor_create(vk,FG_HYPER_WIDTH*4u,&input,&error));
    char path[4096];snprintf(path,sizeof(path),"%s/ngram.iq4nl",pack);
    uint64_t table=0;for(uint32_t i=0;i<m->tensor_count;i++)
        if(m->tensors[i].kind==FG_TENSOR_NGRAM)table=m->tensors[i].bytes;
    fg_ngram_store *ngram=NULL;check(fg_ngram_store_open(&ngram,vk,path,table,1u,&error));
    FILE *outputs[2]={file(out,"off","wbx"),file(out,"on","wbx")},*ng=file(out,"ng","wbx");
    float *hyper=malloc(MAX_TOKENS*FG_HYPER_WIDTH*4u),*emb=malloc(MAX_TOKENS*FG_NGRAM_EMBED_VALUES*4u);
    need(hyper&&emb,"prepare buffers");
    for(uint32_t c=0;c<count;c++){
        sample *s=&cases[c];uint32_t tokens[MAX_TOKENS];int32_t history[MAX_TOKENS];
        uint32_t nt=prepare_tokens(tok,s,tokens);
        for(uint32_t p=0;p<nt;p++){
            history[p]=(int32_t)tokens[p];
            check(fg_vk_embedding_q8_0(vk,input,fg_model_tensor(model,"token_embd.weight"),tokens[p],
                FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,&error));
            check(fg_vk_tensor_read(input,0,hyper+(size_t)p*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,&error));
            fg_vk_tensor *row=NULL;check(fg_ngram_store_lookup(ngram,history,p+1u,&row,&error));
            check(fg_vk_tensor_read(row,0,emb+(size_t)p*FG_NGRAM_EMBED_VALUES,FG_NGRAM_EMBED_VALUES*4u,&error));
        }
        write_data(ng,emb,(size_t)nt*FG_NGRAM_EMBED_VALUES*4u);
        write_data(outputs[0],hyper,(size_t)(s->prompt_tokens+s->divergence)*FG_HYPER_WIDTH*4u);
        write_data(outputs[1],hyper,(size_t)s->prompt_tokens*FG_HYPER_WIDTH*4u);
        for(uint32_t r=0;r<s->round_count;r++){
            round_plan *p=&s->rounds[r];
            write_data(outputs[1],hyper+(size_t)p->first*FG_HYPER_WIDTH,(size_t)p->n*FG_HYPER_WIDTH*4u);
            if(r+1u<s->round_count&&p->keep<p->n)
                write_data(outputs[1],hyper+(size_t)p->first*FG_HYPER_WIDTH,(size_t)p->keep*FG_HYPER_WIDTH*4u);
        }
        printf("{\"kind\":\"prepared\",\"case\":\"%s\",\"tokens\":%u}\n",s->name,nt);fflush(stdout);
    }
    need(!fclose(outputs[0])&&!fclose(outputs[1])&&!fclose(ng),"prepare close");
    free(hyper);free(emb);fg_ngram_store_close(ngram);fg_vk_tensor_destroy(input);
    fg_model_close(model);fg_tokenizer_close(tok);free(m);free(cases);
}
static fg_vk_tensor *block(fg_owner_executor *owner,fg_expert_executor *ex,prefill_context *prefill,
    fg_vk_context *vk,uint32_t first,uint32_t last,uint32_t position,uint32_t n,const char *kind,
    FILE *in,FILE *out,float *ngram,fg_vk_tensor *ng_tensor){
    double start=now_ms();bool prompt=!strcmp(kind,"prompt");
    float hyper[128u*FG_HYPER_WIDTH];uint32_t positions[128u*3u];
    read_data(in,hyper,(size_t)n*FG_HYPER_WIDTH*4u);
    if(!strcmp(kind,"causal_verify"))
        for(uint32_t i=FG_HYPER_WIDTH;i<n*FG_HYPER_WIDTH;i++)
            hyper[i]=.25f*(float)((int)(i%17u)-8);
    for(uint32_t p=0;p<n;p++)for(uint32_t a=0;a<3u;a++)positions[p*3u+a]=position+p;
    fg_vk_tensor *input=n==1u?fg_owner_session_input(owner,fg_owner_active_session(owner)):
        fg_owner_prefill_input(owner),*output=NULL;
    check(fg_vk_tensor_write(input,0,hyper,(size_t)n*FG_HYPER_WIDTH*4u,&error));
    if(ng_tensor)check(fg_vk_tensor_write(ng_tensor,0,ngram+(size_t)position*FG_NGRAM_EMBED_VALUES,
        (size_t)n*FG_NGRAM_EMBED_VALUES*4u,&error));
    double written=now_ms();
    if(prompt){
        const fg_vk_tensor *current=input;
        for(uint32_t l=first;l<=last;l++){
            check(fg_owner_prefill_layer(owner,l,position,positions,(uint16_t)n,current,
                l==1u?ng_tensor:NULL,batch_expert,prefill,NULL,NULL,&output,&error));
            current=output;
        }
    }else if(n==1u)check(fg_owner_decode_block_chained(owner,first,last,position,positions,
        input,ng_tensor,expert,ex,&output,&error));
    else check(fg_owner_decode_small_n(owner,first,last,position,positions,n,input,ng_tensor,
        small_expert,ex,&output,&error));
    (void)vk;
    double computed=now_ms();
    check(fg_vk_tensor_read(output,0,hyper,(size_t)n*FG_HYPER_WIDTH*4u,&error));
    write_data(out,hyper,(size_t)n*FG_HYPER_WIDTH*4u);
    double ended=now_ms();
    printf("{\"kind\":\"block_ledger\",\"operation\":\"%s\",\"first_layer\":%u,"
        "\"position\":%u,\"n\":%u,\"input_ms\":%.6f,\"compute_ms\":%.6f,"
        "\"read_ms\":%.6f,\"total_ms\":%.6f}\n",kind,first,position,n,written-start,
        computed-written,ended-computed,ended-start);
    return output;
}
static bool state_equal(const fg_owner_session_checkpoint *a,const fg_owner_session_checkpoint *b){
    if(a->gdn_count!=b->gdn_count||a->ple_values!=b->ple_values||
       a->qsa_valid!=b->qsa_valid||memcmp(a->qsa_tokens,b->qsa_tokens,sizeof(a->qsa_tokens)))return false;
    for(uint32_t i=0;i<a->gdn_count;i++)
        if(a->gdn[i].values!=b->gdn[i].values||
           memcmp(a->gdn[i].data,b->gdn[i].data,(size_t)a->gdn[i].values*4u))return false;
    return !a->ple_values||!memcmp(a->ple_data,b->ple_data,(size_t)a->ple_values*4u);
}
static void rollback_oracle(fg_owner_executor *owner,fg_expert_executor *ex,
    prefill_context *prefill,fg_vk_context *vk,uint32_t first,uint32_t last,
    const round_plan *p,FILE *in,FILE *out,float *ngram,fg_vk_tensor *ng_tensor,
    const fg_owner_session_checkpoint *checkpoint,fg_vk_tensor *verified_output,const char *name){
    float expected[4u*FG_HYPER_WIDTH],actual[4u*FG_HYPER_WIDTH];
    uint64_t bytes=(uint64_t)p->n*FG_HYPER_WIDTH*4u;
    fg_owner_session_checkpoint verified={0},reconstructed={0},repeated={0};
    check(fg_vk_tensor_read(verified_output,0,expected,bytes,&error));
    check(fg_owner_session_snapshot(owner,1u,&verified,&error));
    check(fg_owner_mtp_retain(owner,checkpoint,p->first,p->n,p->n,&error));
    check(fg_owner_session_snapshot(owner,1u,&reconstructed,&error));
    need(state_equal(&verified,&reconstructed),"trained full-journal state reconstruction");
    check(fg_vk_tensor_read(verified_output,0,actual,bytes,&error));
    need(!memcmp(expected,actual,(size_t)bytes),"trained journal altered verified HC");
    double causal=0,relative=0;
    for(uint32_t pass=0;pass<3u;pass++){
        check(fg_owner_session_rollback(owner,checkpoint,&error));
        need(!fseeko(in,-(off_t)bytes,SEEK_CUR)&&!fseeko(out,-(off_t)bytes,SEEK_CUR),"oracle rewind");
        fg_vk_tensor *result=block(owner,ex,prefill,vk,first,last,p->first,p->n,
            pass==1u?"causal_verify":"rollback_verify",in,out,ngram,ng_tensor);
        check(fg_vk_tensor_read(result,0,actual,bytes,&error));
        if(pass!=1u)need(!memcmp(expected,actual,(size_t)bytes),"trained rollback changed verified HC");
        if(pass==0u){
            check(fg_owner_session_snapshot(owner,1u,&repeated,&error));
            need(state_equal(&verified,&repeated),"trained rollback changed verified state");
        }else if(pass==1u)for(uint32_t i=0;i<FG_HYPER_WIDTH;i++){
            need(isfinite(actual[i]),"nonfinite causal prefix");
            double d=fabs((double)expected[i]-actual[i]);
            if(d>causal)causal=d;
            double r=d/fmax(1.,fmax(fabs(expected[i]),fabs(actual[i])));
            if(r>relative)relative=r;
        }
    }
    printf("{\"kind\":\"trained_rollback_oracle\",\"case\":\"%s\",\"first_layer\":%u,"
        "\"full_journal_exact\":true,\"rollback_hc_exact\":true,\"rollback_state_exact\":true,"
        "\"future_changed_prefix_max_abs\":%.9g,\"future_changed_prefix_max_relative\":%.9g}\n",
        name,first,causal,relative);
    fg_owner_session_snapshot_release(&verified);
    fg_owner_session_snapshot_release(&reconstructed);
    fg_owner_session_snapshot_release(&repeated);
}
static void stage(const char *mp,const char *pack,const char *plan,uint32_t first,
    uint32_t last,const char *in,const char *out,const char *ng_path,bool oracle){
    sample *cases=calloc(MAX_CASES,sizeof(*cases));need(cases!=NULL,"cases");
    uint32_t count=load_plan(plan,cases);fg_manifest *m=manifest(mp);
    need(first<=last&&last<FG_LAYER_COUNT,"stage bounds");
    uint32_t rank=m->layer_owner[first];
    for(uint32_t l=first;l<=last;l++)need(m->layer_owner[l]==rank,"stage crosses rank");
    m->prefill_microbatch=128u;
    fg_model *model=NULL;check(fg_model_open(&model,m,pack,rank,&error));
    fg_vk_context *vk=fg_model_vk(model);fg_owner_executor *owner=NULL;
    check(fg_owner_executor_create_worker_slots(&owner,model,2u,&error));
    fg_expert_executor *ex=NULL;check(fg_expert_executor_create(&ex,model,&error));
    for(uint32_t l=first;l<=last;l++)check(fg_expert_decode_chain_ready(ex,l,&error));
    check(fg_owner_mtp_prepare(owner,2u,&error));
    prefill_context *prefill=calloc(1,sizeof(*prefill));need(prefill!=NULL,"prefill");
    prefill->expert=ex;prefill->rank=rank;
    uint8_t layers[FG_LAYER_COUNT];uint32_t nl=0;char states[2][4096];
    for(uint32_t l=first;l<=last;l++)if((l&3u)==3u)layers[nl++]=(uint8_t)l;
    for(uint32_t s=0;s<2u;s++){
        snprintf(states[s],sizeof(states[s]),"%s/diagnose-qsa-%u-%u",pack,first,s);
        unlink(states[s]);
        fg_qsa_state *state=NULL;
        check(fg_qsa_state_open(&state,states[s],layers,nl,MAX_TOKENS,true,&error));
        fg_qsa_state_close(state);
        check(fg_owner_qsa_open_state_mirror_slot(owner,s,states[s],MAX_TOKENS,0u,
            (MAX_TOKENS/FG_Q38_QSA_COMPRESS_RATIO)*nl,128u,true,unexpected_fetch,NULL,&error));
    }
    fg_vk_tensor *ng_tensor=NULL;
    if(first==0u)check(fg_vk_tensor_create(vk,128u*FG_NGRAM_EMBED_VALUES*4u,&ng_tensor,&error));
    FILE *inputs[2]={file(in,"off","rb"),file(in,"on","rb")};
    FILE *outputs[2]={file(out,"off","wbx"),file(out,"on","wbx")};
    FILE *ng=first==0u?fopen(ng_path,"rb"):NULL;
    if(!first)need(ng!=NULL,"ngram frames");
    float *embedding=malloc(MAX_TOKENS*FG_NGRAM_EMBED_VALUES*4u);need(embedding!=NULL,"ngram");
    for(uint32_t c=0;c<count;c++){
        sample *s=&cases[c];
        fg_owner_session_checkpoint reference={0};
        if(ng)read_data(ng,embedding,(size_t)(s->prompt_tokens+s->generated_count)*FG_NGRAM_EMBED_VALUES*4u);
        for(uint32_t variant=0;variant<2u;variant++){
            check(fg_owner_reset_session(owner,variant,&error));
            check(fg_owner_set_active_session(owner,variant,&error));
            block(owner,ex,prefill,vk,first,last,0u,s->prompt_tokens,"prompt",inputs[variant],
                outputs[variant],embedding,ng_tensor);
            if(!variant){
                for(uint32_t p=0;p<s->divergence;p++)
                    block(owner,ex,prefill,vk,first,last,s->prompt_tokens+p,1u,"off",
                        inputs[variant],outputs[variant],embedding,ng_tensor);
                check(fg_owner_session_snapshot(owner,variant,&reference,&error));
            }else for(uint32_t r=0;r<s->round_count;r++){
                round_plan *p=&s->rounds[r];fg_owner_session_checkpoint checkpoint={0};
                check(fg_owner_session_device_snapshot(owner,variant,&checkpoint,&error));
                fg_vk_tensor *verified_output=block(owner,ex,prefill,vk,first,last,p->first,p->n,"verify",inputs[variant],
                    outputs[variant],embedding,ng_tensor);
                if(oracle&&r+1u==s->round_count)
                    rollback_oracle(owner,ex,prefill,vk,first,last,p,inputs[variant],
                        outputs[variant],embedding,ng_tensor,&checkpoint,verified_output,s->name);
                if(r+1u<s->round_count&&p->keep<p->n){
                    check(fg_owner_session_rollback(owner,&checkpoint,&error));
                    block(owner,ex,prefill,vk,first,last,p->first,p->keep,"target_replay",inputs[variant],
                        outputs[variant],embedding,ng_tensor);
                }
                if(r+1u==s->round_count){
                    uint32_t keep=s->prompt_tokens+s->divergence-p->first;
                    check(fg_owner_mtp_retain(owner,&checkpoint,p->first,p->n,keep,&error));
                    fg_owner_session_checkpoint candidate={0};
                    check(fg_owner_session_snapshot(owner,variant,&candidate,&error));
                    need(reference.gdn_count==candidate.gdn_count&&
                        reference.ple_values==candidate.ple_values,"state shapes");
                    double absolute=0,relative=0;
                    for(uint32_t g=0;g<=reference.gdn_count;g++){
                        const float *a=g==reference.gdn_count?reference.ple_data:reference.gdn[g].data;
                        const float *b=g==candidate.gdn_count?candidate.ple_data:candidate.gdn[g].data;
                        uint64_t values=g==reference.gdn_count?reference.ple_values:reference.gdn[g].values;
                        need(g==reference.gdn_count||values==candidate.gdn[g].values,"state values");
                        for(uint64_t i=0;i<values;i++){
                            need(isfinite(a[i])&&isfinite(b[i]),"nonfinite recurrent state");
                            double delta=fabs((double)a[i]-b[i]);
                            double normalized=delta/fmax(1.0,fmax(fabs(a[i]),fabs(b[i])));
                            if(delta>absolute)absolute=delta;
                            if(normalized>relative)relative=normalized;
                        }
                    }
                    bool frontiers=!memcmp(reference.qsa_tokens,candidate.qsa_tokens,sizeof(reference.qsa_tokens));
                    printf("{\"kind\":\"state_comparison\",\"case\":\"%s\",\"first_layer\":%u,"
                        "\"frontier\":%u,\"max_abs_recurrent_state\":%.9g,"
                        "\"max_relative_recurrent_state\":%.9g,\"qsa_frontiers_equal\":%s}\n",
                        s->name,first,s->prompt_tokens+s->divergence,absolute,relative,
                        frontiers?"true":"false");fflush(stdout);
                    fg_owner_session_snapshot_release(&candidate);
                }
                fg_owner_session_snapshot_release(&checkpoint);
            }
        }
        fg_owner_session_snapshot_release(&reference);
        printf("{\"kind\":\"stage\",\"first\":%u,\"last\":%u,\"case\":\"%s\"}\n",
            first,last,s->name);fflush(stdout);
    }
    for(uint32_t s=0;s<2u;s++){
        need(fgetc(inputs[s])==EOF&&!fclose(inputs[s])&&!fclose(outputs[s]),"stage stream");
    }
    if(ng){need(fgetc(ng)==EOF,"ngram trailing frames");fclose(ng);}
    free(embedding);free(prefill);fg_vk_tensor_destroy(ng_tensor);
    fg_expert_executor_destroy(ex);fg_owner_executor_destroy(owner);fg_model_close(model);
    for(uint32_t s=0;s<2u;s++)unlink(states[s]);
    free(m);free(cases);
}
static void top2(const float *scores,uint32_t ids[2]){
    need(isfinite(scores[0])&&isfinite(scores[1]),"nonfinite logits");
    ids[0]=0u;ids[1]=1u;
    if(scores[ids[1]]>scores[ids[0]]){uint32_t t=ids[0];ids[0]=ids[1];ids[1]=t;}
    for(uint32_t i=2u;i<FG_Q38_VOCAB_SIZE;i++){
        need(isfinite(scores[i]),"nonfinite logits");
        if(scores[i]>scores[ids[0]]){ids[1]=ids[0];ids[0]=i;}
        else if(scores[i]>scores[ids[1]])ids[1]=i;
    }
}
static uint32_t score(fg_output_executor *head,fg_vk_tensor *input,const float *hyper,float *logits){
    check(fg_vk_tensor_write(input,0,hyper,FG_HYPER_WIDTH*4u,&error));
    fg_vk_tensor *output=NULL;check(fg_output_logits(head,input,&output,&error));
    check(fg_vk_tensor_read(output,0,logits,FG_Q38_VOCAB_SIZE*4u,&error));
    uint32_t ids[2];top2(logits,ids);return ids[0];
}
static void head(const char *mp,const char *pack,const char *plan,const char *in,const char *out){
    sample *cases=calloc(MAX_CASES,sizeof(*cases));need(cases!=NULL,"cases");
    uint32_t count=load_plan(plan,cases);fg_manifest *m=manifest(mp);
    fg_model *model=NULL;check(fg_model_open(&model,m,pack,fg_output_owner_rank(m),&error));
    fg_output_executor *head=NULL;check(fg_output_executor_create(&head,model,&error));
    fg_vk_tensor *input=NULL;check(fg_vk_tensor_create(fg_model_vk(model),FG_HYPER_WIDTH*4u,&input,&error));
    FILE *inputs[2]={file(in,"off","rb"),file(in,"on","rb")};
    FILE *raw=file(out,"logits","wbx"),*hc=file(out,"hc","wbx");
    float *logits[2]={malloc(FG_Q38_VOCAB_SIZE*4u),malloc(FG_Q38_VOCAB_SIZE*4u)};
    float frames[128u*FG_HYPER_WIDTH],selected[2][FG_HYPER_WIDTH];
    need(logits[0]&&logits[1],"logits allocation");
    for(uint32_t c=0;c<count;c++){
        sample *s=&cases[c];uint32_t checked=0,mismatched=0;
        for(uint32_t v=0;v<2u;v++){
            read_data(inputs[v],frames,(size_t)s->prompt_tokens*FG_HYPER_WIDTH*4u);
            uint32_t token=score(head,input,frames+(size_t)(s->prompt_tokens-1u)*FG_HYPER_WIDTH,logits[v]);
            checked++;mismatched+=token!=s->generated[0];
            if(!v){
                for(uint32_t p=0;p<s->divergence;p++){
                    read_data(inputs[v],selected[v],FG_HYPER_WIDTH*4u);
                    token=score(head,input,selected[v],logits[v]);
                    if(p+1u<s->divergence){checked++;mismatched+=token!=s->generated[p+1u];}
                }
            }else for(uint32_t r=0;r<s->round_count;r++){
                round_plan *p=&s->rounds[r];
                read_data(inputs[v],frames,(size_t)p->n*FG_HYPER_WIDTH*4u);
                for(uint32_t i=0;i<p->keep;i++){
                    uint32_t index=p->first-s->prompt_tokens+i+1u;
                    if(index>s->divergence)break;
                    token=score(head,input,frames+(size_t)i*FG_HYPER_WIDTH,logits[v]);
                    if(index<s->divergence){checked++;mismatched+=token!=s->generated[index];}
                    else memcpy(selected[v],frames+(size_t)i*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u);
                }
                if(r+1u<s->round_count&&p->keep<p->n)
                    read_data(inputs[v],frames,(size_t)p->keep*FG_HYPER_WIDTH*4u);
            }
        }
        uint32_t ids[2][2];double maximum=0;
        for(uint32_t v=0;v<2u;v++){
            top2(logits[v],ids[v]);write_data(raw,logits[v],FG_Q38_VOCAB_SIZE*4u);
            write_data(hc,selected[v],FG_HYPER_WIDTH*4u);
        }
        for(uint32_t i=0;i<FG_Q38_VOCAB_SIZE;i++)
            if(fabs((double)logits[0][i]-logits[1][i])>maximum)
                maximum=fabs((double)logits[0][i]-logits[1][i]);
        printf("{\"case\":\"%s\",\"token_index\":%u,\"off_top1\":%u,\"on_top1\":%u,"
            "\"off_top2\":%u,\"on_top2\":%u,\"off_margin\":%.9g,\"on_margin\":%.9g,"
            "\"off_top1_logit\":%.9g,\"on_top1_logit\":%.9g,"
            "\"max_logit_delta\":%.9g,\"prefix_predictions_checked\":%u,"
            "\"prefix_predictions_mismatched\":%u,\"historical_choices_reproduced\":%s}\n",
            s->name,s->divergence,ids[0][0],ids[1][0],ids[0][1],ids[1][1],
            logits[0][ids[0][0]]-logits[0][ids[0][1]],
            logits[1][ids[1][0]]-logits[1][ids[1][1]],
            logits[0][ids[0][0]],logits[1][ids[1][0]],maximum,checked,mismatched,
            !mismatched&&ids[0][0]==s->generated[s->divergence]&&ids[1][0]==s->expected_on?"true":"false");
        fflush(stdout);
    }
    for(uint32_t v=0;v<2u;v++){need(fgetc(inputs[v])==EOF,"head trailing frames");
        fclose(inputs[v]);free(logits[v]);}
    need(!fclose(raw)&&!fclose(hc),"head output close");
    fg_vk_tensor_destroy(input);fg_output_executor_destroy(head);fg_model_close(model);
    free(m);free(cases);
}
int main(int argc,char **argv){
    if(argc==3&&!strcmp(argv[1],"plan")){
        sample *cases=calloc(MAX_CASES,sizeof(*cases));need(cases!=NULL,"cases");
        uint32_t count=load_plan(argv[2],cases);
        printf("{\"cases\":%u,\"plan_valid\":true}\n",count);free(cases);
    }else if(argc==6&&!strcmp(argv[1],"prepare"))prepare(argv[2],argv[3],argv[4],argv[5]);
    else if(argc==10&&(!strcmp(argv[1],"stage")||!strcmp(argv[1],"oracle")))stage(argv[2],argv[3],argv[4],
        (uint32_t)strtoul(argv[5],NULL,10),(uint32_t)strtoul(argv[6],NULL,10),argv[7],argv[8],argv[9],
        !strcmp(argv[1],"oracle"));
    else if(argc==7&&!strcmp(argv[1],"head"))head(argv[2],argv[3],argv[4],argv[5],argv[6]);
    else{fprintf(stderr,"usage: mtp-offline prepare M PACK PLAN OUT | stage|oracle M PACK PLAN FIRST LAST IN OUT NG | head M PACK PLAN IN OUT\n");return 2;}
    return 0;
}
