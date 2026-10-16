/* Included after the resident runtime helpers: diagnostic only, never serving. */
typedef struct mtpq_case {
    char id[48],category[8],path[1024];
    uint64_t session;
    uint32_t prompt,generated,token[259];
} mtpq_case;

typedef struct mtpq_clock { double end; } mtpq_clock;
static bool mtpq_expired(void *opaque){
    return dispatch_ts()>=((mtpq_clock *)opaque)->end;
}
static fg_status mtpq_fail(fg_error *err,const char *message){
    fg_error_set(err,FG_ERR_MISMATCH,"MTP qualification: %s",message);
    return FG_ERR_MISMATCH;
}
static char *mtpq_text(const char *path){
    FILE *f=fopen(path,"rb");
    if(!f)return NULL;
    if(fseeko(f,0,SEEK_END)){fclose(f);return NULL;}
    off_t size=ftello(f);
    if(size<1||size>65536||fseeko(f,0,SEEK_SET)){fclose(f);return NULL;}
    char *s=malloc((size_t)size+1u);
    if(!s){fclose(f);return NULL;}
    if(fread(s,1,(size_t)size,f)!=(size_t)size){free(s);fclose(f);return NULL;}
    s[size]=0;fclose(f);return s;
}
static fg_status mtpq_token(void *opaque,uint32_t token,const char *text,
                            size_t bytes,fg_error *err){
    (void)text;(void)bytes;
    mtpq_case *c=opaque;
    if(c->generated>=259u)return mtpq_fail(err,"too many baseline tokens");
    c->token[c->generated++]=token;
    return FG_OK;
}
static fg_status mtpq_shared(fg_runtime *r,const char *out,fg_error *err){
    const char *names[]={"token_embd.weight","output.weight"};
    const char *files[]={"embedding.q8","head.q8"};
    const uint32_t row_bytes=2560u/32u*34u;
    uint64_t cooked_bytes=fg_q8_0_cooked_tile_bytes(2560u);
    uint8_t cooked[65536],raw[16u*2560u/32u*34u];
    if(cooked_bytes>sizeof(cooked))return mtpq_fail(err,"shared Q8 tile size");
    for(uint32_t n=0;n<2u;n++){
        const fg_tensor_record *t=NULL;
        for(uint32_t i=0;i<r->manifest->tensor_count;i++)
            if(!strcmp(r->manifest->tensors[i].name,names[n]))t=&r->manifest->tensors[i];
        if(!t||t->ggml_type!=8u||t->shape[0]!=2560u||
           t->shape[1]!=FG_Q38_VOCAB_SIZE||
           (t->layout!=FG_TENSOR_LAYOUT_Q8_0_COOKED&&t->layout!=0u))
            return mtpq_fail(err,"production shared weight is not supported Q8");
        char source[1200],destination[1200];
        snprintf(source,sizeof(source),"%s/rank-%02u.fgw",r->directory,t->rank);
        snprintf(destination,sizeof(destination),"%s/%s",out,files[n]);
        FILE *in=fopen(source,"rb"),*f=fopen(destination,"wb");
        if(!in||!f){if(in)fclose(in);if(f)fclose(f);return mtpq_fail(err,"shared export open");}
        bool ok=fseeko(in,(off_t)t->offset,SEEK_SET)==0;
        for(uint32_t row=0;ok&&row<FG_Q38_VOCAB_SIZE;row+=16u){
            size_t bytes=t->layout==0u?sizeof(raw):(size_t)cooked_bytes;
            ok=fread(cooked,1,bytes,in)==bytes;
            if(ok&&t->layout==0u)memcpy(raw,cooked,sizeof(raw));
            else if(ok)ok=fg_uncook_q8_0_rows(cooked,raw,sizeof(raw),2560u,16u);
            if(ok)ok=fwrite(raw,row_bytes,16u,f)==16u;
        }
        if(fclose(in))ok=false;
        if(fclose(f))ok=false;
        if(!ok)return mtpq_fail(err,"shared export read/write");
    }
    return FG_OK;
}
typedef struct mtpq_verify {
    fg_runtime *runtime;
    int32_t *history;
    uint32_t base;
    mtpq_clock *clock;
} mtpq_verify;
static fg_status mtpq_checkpoint(void *opaque,fg_error *err){
    return coordinator_owner_transaction(((mtpq_verify *)opaque)->runtime?
        &((mtpq_verify *)opaque)->runtime->coordinator:NULL,FG_OWNER_SESSION_PREPARE,0u,err);
}
static fg_status mtpq_forward(void *opaque,const uint32_t *inputs,uint32_t count,
                               uint32_t *target,fg_error *err){
    mtpq_verify *v=opaque;
    for(uint32_t j=0;j<count;j++){
        if(mtpq_expired(v->clock))return mtpq_fail(err,"lease work deadline");
        uint32_t next;float logit;
        v->history[v->base+j]=(int32_t)inputs[j];
        fg_status s=coordinator_decode_token(&v->runtime->coordinator,v->history,
            v->base+j+1u,v->base+j,v->base+j,&next,&logit,err);
        if(s!=FG_OK)return s;
        if(target)target[j]=next;
    }
    return FG_OK;
}
static fg_status mtpq_restore(void *opaque,fg_error *err){
    mtpq_verify *v=opaque;
    fg_status s=coordinator_owner_transaction(&v->runtime->coordinator,
        FG_OWNER_SESSION_RESTORE,0u,err);
    /* Keep a recoverable original checkpoint if retained-prefix replay fails. */
    if(s==FG_OK)s=coordinator_owner_transaction(&v->runtime->coordinator,
        FG_OWNER_SESSION_PREPARE,0u,err);
    return s;
}
static fg_status mtpq_replay(void *opaque,const uint32_t *inputs,uint32_t count,fg_error *err){
    return mtpq_forward(opaque,inputs,count,NULL,err);
}
static fg_status mtpq_commit(void *opaque,fg_error *err){
    return coordinator_owner_transaction(&((mtpq_verify *)opaque)->runtime->coordinator,
        FG_OWNER_SESSION_COMMIT,0u,err);
}
static fg_status mtpq_greedy(fg_runtime *r,const mtpq_case *c,const char *dir,
                            uint32_t k,mtpq_clock *clock,FILE *receipts,fg_error *err){
    char *text=mtpq_text(c->path),path[1200];
    fg_tokens prompt={0};
    fg_status s=text?fg_tokenizer_encode(r->coordinator.tokenizer,text,true,&prompt,err):
        mtpq_fail(err,"greedy prompt read");
    free(text);
    uint32_t proposals[256u*3u];
    snprintf(path,sizeof(path),"%s/%s.drafts.u32",dir,c->id);
    FILE *f=s==FG_OK?fopen(path,"rb"):NULL;
    if(s==FG_OK&&(!f||fread(proposals,4u,256u*3u,f)!=256u*3u||fgetc(f)!=EOF))
        s=mtpq_fail(err,"proposal dimensions");
    if(f)fclose(f);
    int32_t *history=s==FG_OK?malloc((prompt.count+264u)*4u):NULL;
    if(s==FG_OK&&!history)s=mtpq_fail(err,"greedy history allocation");
    if(s==FG_OK)s=fg_runtime_reset(r,err);
    uint32_t pending=0u;float logit=0;
    if(s==FG_OK)s=depthb_prefill_sample(r,&prompt,history,&pending,&logit,err);
    uint32_t emitted=1u,cycles=0u,rollbacks=0u,accepted=0u;
    if(s==FG_OK&&(prompt.count!=c->prompt||pending!=c->token[0]))
        s=mtpq_fail(err,"prefill greedy mismatch");
    while(s==FG_OK&&emitted<256u){
        mtpq_verify v={r,history,c->prompt+emitted-1u,clock};
        uint32_t draft[3];memcpy(draft,proposals+(emitted-1u)*3u,k*4u);
        /* Independent of acceptance statistics: force real rejected-state work. */
        if(cycles==0u)draft[0]=(c->token[emitted]+1u)%FG_Q38_VOCAB_SIZE;
        fg_mtp_ops ops={mtpq_checkpoint,mtpq_forward,mtpq_restore,mtpq_replay,mtpq_commit,&v,NULL};
        fg_mtp_result result;
        s=fg_mtp_cycle_greedy(&ops,pending,draft,k,256u-emitted,UINT32_MAX,&result,err);
        if(s!=FG_OK)break;
        for(uint32_t j=0;j<result.count;j++)
            if(result.tokens[j]!=c->token[emitted+j]){
                s=mtpq_fail(err,"full-ring MTP-on/off token mismatch");break;
            }
        if(s!=FG_OK)break;
        if(result.consumed<k+1u)rollbacks++;
        accepted+=result.accepted;cycles++;emitted+=result.count;
        pending=result.tokens[result.count-1u];
    }
    fprintf(receipts,"{\"case\":\"%s\",\"k\":%u,\"tokens\":%u,\"cycles\":%u,"
        "\"rollbacks\":%u,\"accepted\":%u,\"equal\":%s,\"verifier\":\"causal-N1-reference\"}\n",
        c->id,k,emitted,cycles,rollbacks,accepted,s==FG_OK?"true":"false");
    fflush(receipts);
    fg_tokens_free(&prompt);free(history);
    return s;
}

fg_status fg_mtp_qualify_main(const char *manifest,const char *cases,
    const char *directory,uint32_t budget_seconds,fg_error *err){
    mtpq_clock clock={fg_mtp_capture_deadline_ms(dispatch_ts(),budget_seconds)};
    mtpq_case eligible[40]={0};uint32_t found=0u,counts[2]={0};
    FILE *list=fopen(cases,"r");
    if(!list)return mtpq_fail(err,"case list open");
    fg_runtime_options options;fg_runtime_options_init(&options);
    options.no_prefix_continuation=true;
    options.logical_context_tokens=FG_NATIVE_CONTEXT;options.gpu_index_tokens=FG_NATIVE_CONTEXT;
    options.qsa_hot_tokens=0u;options.qsa_page_cache_bytes=UINT64_C(700)<<20u;
    options.specified=FG_RUNTIME_OPTION_LOGICAL_CONTEXT|FG_RUNTIME_OPTION_GPU_INDEX|
        FG_RUNTIME_OPTION_QSA_HOT|FG_RUNTIME_OPTION_PAGE_CACHE|FG_RUNTIME_OPTION_PREFIX_CONT;
    fg_runtime *r=NULL;
    fg_status s=fg_mtp_capture_configure(directory,err);
    if(s==FG_OK)s=fg_runtime_open_with_options(&r,manifest,&options,err);
    if(s==FG_OK&&r->manifest->layer_owner[FG_LAYER_COUNT-1u]==0u)
        s=mtpq_fail(err,"capture currently requires a remote final owner");
    if(s==FG_OK)s=mtpq_shared(r,directory,err);
    char path[1200];
    snprintf(path,sizeof(path),"%s/cases.tsv",directory);
    FILE *index=s==FG_OK?fopen(path,"w"):NULL;
    if(s==FG_OK&&!index)s=mtpq_fail(err,"index open");
    char line[1200];
    while(s==FG_OK&&found<40u&&fgets(line,sizeof(line),list)){
        mtpq_case c={0};
        if(sscanf(line,"%47s %7s %1023s",c.id,c.category,c.path)!=3){
            s=mtpq_fail(err,"case list format");break;
        }
        if(strspn(c.id,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=strlen(c.id)){
            s=mtpq_fail(err,"unsafe case id");break;
        }
        uint32_t category=!strcmp(c.category,"chat")?0u:!strcmp(c.category,"code")?1u:2u;
        if(category>1u){s=mtpq_fail(err,"unknown category");break;}
        if(counts[category]>=20u)continue;
        char *text=mtpq_text(c.path);
        if(!text){s=mtpq_fail(err,"prompt file read");break;}
        fg_generation_stats stats={0};
        s=fg_runtime_reset(r,err);
        if(s==FG_OK)s=fg_runtime_generate(r,text,259u,mtpq_token,&c,mtpq_expired,
            &clock,&stats,err);
        free(text);
        if(s!=FG_OK)break;
        if(mtpq_expired(&clock)){s=mtpq_fail(err,"capture deadline");break;}
        if(c.generated<259u)continue; /* EOS is not ignored or padded. */
        c.prompt=stats.prompt_tokens;c.session=r->coordinator.session_id;
        snprintf(path,sizeof(path),"%s/%s.tokens.u32",directory,c.id);
        FILE *tokens=fopen(path,"wb");
        bool ok=tokens&&fwrite(r->history,4u,c.prompt+c.generated,tokens)==c.prompt+c.generated;
        if(tokens&&fclose(tokens))ok=false;
        if(!ok){s=mtpq_fail(err,"baseline token write");break;}
        fprintf(index,"%s\t%s\t%llu\t%u\t%u\t%u\n",c.id,c.category,
            (unsigned long long)c.session,c.prompt,c.generated,
            r->manifest->layer_owner[FG_LAYER_COUNT-1u]);
        fflush(index);eligible[found++]=c;counts[category]++;
    }
    fclose(list);if(index)fclose(index);
    if(s==FG_OK&&(counts[0]!=20u||counts[1]!=20u))
        s=mtpq_fail(err,"fewer than twenty eligible chat/code cases");
    if(s==FG_OK){
        snprintf(path,sizeof(path),"%s/baseline.ready",directory);
        FILE *ready=fopen(path,"w");
        if(!ready)s=mtpq_fail(err,"baseline readiness write");
        else {fputs("40 cases; 259 generated each\n",ready);fclose(ready);}
    }
    snprintf(path,sizeof(path),"%s/proposals.ready",directory);
    while(s==FG_OK&&access(path,F_OK)!=0){
        if(mtpq_expired(&clock)){s=mtpq_fail(err,"proposal wait deadline");break;}
        sleep(1);
    }
    snprintf(path,sizeof(path),"%s/greedy.jsonl",directory);
    FILE *receipts=s==FG_OK?fopen(path,"w"):NULL;
    if(s==FG_OK&&!receipts)s=mtpq_fail(err,"greedy receipts open");
    uint32_t gates[2]={0};
    for(uint32_t i=0;s==FG_OK&&i<found;i++){
        uint32_t cat=!strcmp(eligible[i].category,"chat")?0u:1u;
        if(gates[cat]>=4u)continue;
        for(uint32_t k=1;s==FG_OK&&k<=3u;k++)
            s=mtpq_greedy(r,&eligible[i],directory,k,&clock,receipts,err);
        gates[cat]++;
    }
    if(receipts)fclose(receipts);
    snprintf(path,sizeof(path),"%s/qualification.%s",directory,s==FG_OK?"passed":"failed");
    FILE *done=fopen(path,"w");
    if(done){fprintf(done,"%s\n",s==FG_OK?"24 paired 256-token greedy checks passed":err->message);fclose(done);}
    fg_runtime_close(r);
    fg_mtp_capture_configure(NULL,NULL);
    return s;
}
