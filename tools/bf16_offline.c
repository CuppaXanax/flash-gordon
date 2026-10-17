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
#include <unistd.h>

enum{MAX_CASES=1024,MAX_TOKENS=512};
typedef struct sample {uint32_t count,tokens[MAX_TOKENS];} sample;
static fg_error error;
static void check(fg_status status){
    if(status!=FG_OK){fprintf(stderr,"offline: %s\n",error.message);exit(1);}
}
static void need(bool condition,const char *what){
    if(!condition){fprintf(stderr,"offline: %s\n",what);exit(1);}
}
static FILE *file(const char *directory,const char *name,const char *mode){
    char path[2048];need(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path),"path");
    FILE *f=fopen(path,mode);need(f!=NULL,path);return f;
}
static void write_data(FILE *f,const void *p,size_t bytes){need(fwrite(p,1,bytes,f)==bytes,"write");}
static void read_data(FILE *f,void *p,size_t bytes){need(fread(p,1,bytes,f)==bytes,"read");}
static uint32_t load_cases(const char *pack,sample *cases){
    FILE *f=file(pack,"cases.bin","rb");char magic[8];uint32_t count;
    read_data(f,magic,8);need(!memcmp(magic,"FGBCASE1",8),"cases magic");
    read_data(f,&count,4);need(count&&count<=MAX_CASES,"case count");
    for(uint32_t i=0;i<count;i++){
        read_data(f,&cases[i].count,4);
        need(cases[i].count&&cases[i].count<=MAX_TOKENS,"token count");
        read_data(f,cases[i].tokens,cases[i].count*4u);
    }
    need(fgetc(f)==EOF,"cases trailing data");fclose(f);return count;
}
static fg_manifest *manifest(const char *path){
    fg_manifest *m=malloc(sizeof(*m));need(m!=NULL,"manifest allocation");
    check(fg_manifest_read(path,m,&error));return m;
}
static fg_status expert(void *opaque,uint32_t layer,const fg_vk_tensor *activation,
                         const fg_vk_tensor *logits,fg_vk_tensor **out,fg_error *err){
    return fg_expert_decode_chain(opaque,layer,activation,logits,out,err);
}
typedef struct prefill_experts {
    fg_expert_executor *executor;
    uint32_t rank;
    fg_prefill_pair pairs[FG_PREFILL_MAX_PAIRS];
    fg_prefill_result_pair result_pairs[FG_PREFILL_MAX_PAIRS];
    float *outputs;
} prefill_experts;
static fg_status prefill_dispatch(void *opaque,uint32_t layer,uint32_t first,
    uint16_t count,const uint16_t *ids,const float *gates,const uint8_t *activations,
    fg_owner_prefill_shared_fn shared,void *shared_context,
    fg_prefill_result results[FG_GROUP_SIZE],uint32_t *result_count,fg_error *err){
    prefill_experts *e=opaque;
    for(uint32_t i=0;i<(uint32_t)count*FG_TOP_K;i++)
        e->pairs[i]=(fg_prefill_pair){.token_slot=i/FG_TOP_K,.expert_id=ids[i],
            .routing_slot=i%FG_TOP_K,.gate=gates[i]};
    fg_prefill_work work={.layer=(uint8_t)layer,.source_rank=(uint8_t)e->rank,
        .destination_rank=(uint8_t)e->rank,.first_position=first,
        .token_count=count,.pair_count=count*FG_TOP_K,
        .activations_q8k=(uint8_t *)activations,.pairs=e->pairs};
    fg_status status=shared(shared_context,err);
    if(status==FG_OK)status=fg_expert_prefill(e->executor,&work,&results[0],
        e->result_pairs,FG_PREFILL_MAX_PAIRS,e->outputs,
        (uint64_t)FG_PREFILL_MAX_TOKENS*FG_HIDDEN_SIZE,err);
    if(status==FG_OK)*result_count=1;
    return status;
}
static FILE *frames(const char *prefix,const char *variant,const char *mode){
    char path[2048];need(snprintf(path,sizeof(path),"%s.%s",prefix,variant)<(int)sizeof(path),"frame path");
    FILE *f=fopen(path,mode);need(f!=NULL,path);return f;
}
static fg_tokenizer *tokenizer(const char *header,const char *pack,fg_manifest *m){
    fg_manifest *draft=calloc(1,sizeof(*draft));need(draft!=NULL,"tokenizer manifest");
    check(fg_tokenizer_pack_gguf(header,pack,draft,&error));free(draft);
    fg_tokenizer *tok=NULL;check(fg_tokenizer_open(&tok,pack,m,&error));return tok;
}
static void prepare(const char *mp,const char *pack,const char *header,const char *corpus){
    fg_manifest *m=manifest(mp);
    fg_tokenizer *tok=tokenizer(header,pack,m);
    fg_model *model=NULL;check(fg_model_open(&model,m,pack,0,&error));
    fg_vk_context *vk=fg_model_vk(model);fg_vk_tensor *input=NULL;
    check(fg_vk_tensor_create(vk,FG_HYPER_WIDTH*4u,&input,&error));
    fg_ngram_store *ngram=NULL;uint64_t table=0;
    for(uint32_t i=0;i<m->tensor_count;i++)if(m->tensors[i].kind==FG_TENSOR_NGRAM)table=m->tensors[i].bytes;
    char path[2048];snprintf(path,sizeof(path),"%s/ngram.iq4nl",pack);
    check(fg_ngram_store_open(&ngram,vk,path,table,1u,&error));
    FILE *prompts=fopen(corpus,"rb");need(prompts!=NULL,"open corpus");char magic[8];
    read_data(prompts,magic,8);need(!memcmp(magic,"FGEVAL01",8),"corpus magic");
    FILE *cases=file(pack,"cases.bin","wbx"),*ng=file(pack,"ngram.f32","wbx");
    FILE *teacher=file(pack,"input.teacher","wbx"),*candidate=file(pack,"input.candidate","wbx");
    uint32_t count=0;write_data(cases,"FGBCASE1",8);write_data(cases,&count,4);
    float hyper[FG_HYPER_WIDTH],embedding[FG_NGRAM_EMBED_VALUES];
    uint64_t total=0;
    for(;;){
        uint32_t bytes;size_t n=fread(&bytes,1,4,prompts);
        if(!n&&feof(prompts))break;
        need(n==4&&bytes&&bytes<(1u<<20u)&&count<MAX_CASES,"prompt size");
        char *text=malloc((size_t)bytes+1u);need(text!=NULL,"prompt allocation");
        read_data(prompts,text,bytes);text[bytes]=0;
        fg_tokens tokens={0};check(fg_tokenizer_encode(tok,text,true,&tokens,&error));free(text);
        need(tokens.count>=4&&tokens.count<=MAX_TOKENS,"prompt token limit");
        uint32_t nt=(uint32_t)tokens.count;write_data(cases,&nt,4);
        write_data(cases,tokens.data,nt*4u);
        int32_t history[MAX_TOKENS];
        for(uint32_t p=0;p<nt;p++){
            history[p]=(int32_t)tokens.data[p];
            check(fg_vk_embedding_q8_0(vk,input,fg_model_tensor(model,"token_embd.weight"),
                tokens.data[p],FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,FG_Q38_HYPER_COUNT,&error));
            check(fg_vk_tensor_read(input,0,hyper,sizeof(hyper),&error));
            fg_vk_tensor *row=NULL;check(fg_ngram_store_lookup(ngram,history,p+1u,&row,&error));
            check(fg_vk_tensor_read(row,0,embedding,sizeof(embedding),&error));
            write_data(teacher,hyper,sizeof(hyper));write_data(candidate,hyper,sizeof(hyper));
            write_data(ng,embedding,sizeof(embedding));
        }
        printf("{\"kind\":\"prepared\",\"case\":%u,\"tokens\":%u}\n",count,nt);fflush(stdout);
        count++;total+=nt;fg_tokens_free(&tokens);
    }
    need(count>0,"empty corpus");need(fseek(cases,8,SEEK_SET)==0,"case seek");write_data(cases,&count,4);
    need(fclose(cases)==0&&fclose(teacher)==0&&fclose(candidate)==0&&fclose(ng)==0,"close inputs");
    fclose(prompts);fg_ngram_store_close(ngram);fg_vk_tensor_destroy(input);
    fg_model_close(model);fg_tokenizer_close(tok);free(m);
    printf("{\"kind\":\"prepared_complete\",\"cases\":%u,\"tokens\":%llu}\n",
        count,(unsigned long long)total);
}
static void stage(const char *mp,const char *pack,uint32_t first,uint32_t last,
                    const char *in,const char *out){
    const char *codec=getenv("HOP_OFFLINE_CODEC");if(!codec)codec="bf16";
    need(!strcmp(codec,"bf16")||!strcmp(codec,"fp16")||!strcmp(codec,"fp16-128")||
        !strcmp(codec,"f32"),"unknown offline codec");
    const char *mask_text=getenv("HOP_OFFLINE_MASK");char *end=NULL;
    unsigned long mask=mask_text?strtoul(mask_text,&end,0):255u;
    need(!mask_text||(*mask_text&&!*end&&mask<256u),"offline destination mask");
    fg_manifest *m=manifest(mp);need(first<=last&&last<FG_LAYER_COUNT,"stage bounds");
    uint32_t rank=m->layer_owner[first];
    for(uint32_t l=first;l<=last;l++)need(m->layer_owner[l]==rank,"stage crosses rank");
    sample *cases=calloc(MAX_CASES,sizeof(*cases));need(cases!=NULL,"cases allocation");
    uint32_t count=load_cases(pack,cases);
    const char *path=getenv("HOP_OFFLINE_PATH");
    bool prefill=path&&!strcmp(path,"prefill");
    need(!path||prefill||!strcmp(path,"decode"),"offline arithmetic path");
    need(!prefill||!strcmp(codec,"f32"),"prefill control must use f32 boundaries");
    /* Decode schedules use nine words/pair versus five for prefill scratch. */
    if(!prefill)m->prefill_microbatch=2u;
    fg_model *model=NULL;check(fg_model_open(&model,m,pack,rank,&error));
    fg_vk_context *vk=fg_model_vk(model);fg_owner_executor *owner=NULL;
    check(fg_owner_executor_create_worker_slots(&owner,model,2u,&error));
    fg_expert_executor *ex=NULL;check(fg_expert_executor_create(&ex,model,&error));
    if(!prefill)for(uint32_t l=first;l<=last;l++)
        check(fg_expert_decode_chain_ready(ex,l,&error));
    prefill_experts *pe=prefill?calloc(1,sizeof(*pe)):NULL;
    if(prefill){
        need(pe!=NULL,"prefill dispatch allocation");pe->executor=ex;pe->rank=rank;
        pe->outputs=malloc((size_t)FG_PREFILL_MAX_TOKENS*FG_HIDDEN_SIZE*4u);
        need(pe->outputs!=NULL,"prefill expert output allocation");
    }
    char states[2][2048];
    uint8_t layers[FG_LAYER_COUNT];uint32_t layer_count=0;
    for(uint32_t l=3u;l<FG_LAYER_COUNT;l+=4u)
        if(m->layer_owner[l]==rank)layers[layer_count++]=(uint8_t)l;
    need(layer_count>0,"owned QSA layers");
    /* Rank 0 is a coordinator mirror: keep the entire evaluation window hot. */
    uint32_t cache_pages=(MAX_TOKENS/FG_Q38_QSA_COMPRESS_RATIO)*layer_count;
    for(uint32_t s=0;s<2;s++){
        snprintf(states[s],sizeof(states[s]),"%s/qsa-%u-%u.state",pack,first,s);
        fg_qsa_state *state=NULL;
        check(fg_qsa_state_open(&state,states[s],layers,layer_count,MAX_TOKENS,true,&error));
        fg_qsa_state_close(state);
        check(fg_owner_qsa_open_state_mirror_slot(owner,s,states[s],MAX_TOKENS,MAX_TOKENS,
            cache_pages,prefill?m->prefill_microbatch:1u,true,NULL,NULL,&error));
    }
    fg_vk_tensor *ngram=NULL;
    if(!first)check(fg_vk_tensor_create(vk,
        (uint64_t)FG_NGRAM_EMBED_VALUES*4u*(prefill?m->prefill_microbatch:1u),&ngram,&error));
    FILE *inputs[2]={frames(in,"teacher","rb"),frames(in,"candidate","rb")};
    FILE *outputs[2]={frames(out,"teacher","wbx"),frames(out,"candidate","wbx")};
    const char *saved_teacher=getenv("HOP_OFFLINE_TEACHER");
    FILE *saved=saved_teacher?fopen(saved_teacher,"rb"):NULL;
    need(!saved_teacher||saved,"saved f32 teacher output");
    FILE *ng=!first?file(pack,"ngram.f32","rb"):NULL;
    float *h[2],*n=malloc((size_t)MAX_TOKENS*FG_NGRAM_EMBED_VALUES*4u);
    for(uint32_t s=0;s<2;s++)h[s]=malloc((size_t)MAX_TOKENS*FG_HYPER_WIDTH*4u);
    need(n&&h[0]&&h[1],"frame allocation");
    uint8_t wire[FG_LAYER_WORK_MAX_BYTES];fg_layer_work w,decoded;
    double prefix_error=0,square_error[2]={0},square_reference[2]={0};uint64_t rounded=0;
    for(uint32_t c=0;c<count;c++){
        uint32_t nt=cases[c].count;
        if(ng)read_data(ng,n,(size_t)nt*FG_NGRAM_EMBED_VALUES*4u);
        for(uint32_t s=0;s<2;s++){
            read_data(inputs[s],h[s],(size_t)nt*FG_HYPER_WIDTH*4u);
            if(!s&&saved){
                read_data(saved,h[s],(size_t)nt*FG_HYPER_WIDTH*4u);
                write_data(outputs[s],h[s],(size_t)nt*FG_HYPER_WIDTH*4u);continue;
            }
            check(fg_owner_reset_session(owner,s,&error));
            check(fg_owner_set_active_session(owner,s,&error));
            if(prefill){
                fg_vk_tensor *input=fg_owner_prefill_input(owner);
                for(uint32_t p=0;p<nt;){
                    uint32_t batch=nt-p;
                    if(batch>m->prefill_microbatch)batch=m->prefill_microbatch;
                    uint32_t positions[FG_PREFILL_MAX_TOKENS*3u];
                    for(uint32_t i=0;i<batch;i++)for(uint32_t axis=0;axis<3;axis++)
                        positions[3u*i+axis]=p+i;
                    check(fg_vk_tensor_write(input,0,h[s]+(size_t)p*FG_HYPER_WIDTH,
                        (uint64_t)batch*FG_HYPER_WIDTH*4u,&error));
                    if(ngram)check(fg_vk_tensor_write(ngram,0,n+(size_t)p*FG_NGRAM_EMBED_VALUES,
                        (uint64_t)batch*FG_NGRAM_EMBED_VALUES*4u,&error));
                    fg_vk_tensor *output=input;
                    for(uint32_t l=first;l<=last;l++)
                        check(fg_owner_prefill_layer(owner,l,p,positions,(uint16_t)batch,
                            output,l==1u?ngram:NULL,prefill_dispatch,pe,NULL,NULL,&output,&error));
                    check(fg_vk_tensor_read(output,0,h[s]+(size_t)p*FG_HYPER_WIDTH,
                        (uint64_t)batch*FG_HYPER_WIDTH*4u,&error));
                    p+=batch;
                }
                write_data(outputs[s],h[s],(size_t)nt*FG_HYPER_WIDTH*4u);
                continue;
            }
            fg_vk_tensor *input=fg_owner_session_input(owner,s);
            for(uint32_t p=0;p<nt;p++){
                bool compressed=s&&p>=nt-4u&&(mask&(1ul<<rank))&&strcmp(codec,"f32");
                setenv("FG_BF16_HOPS",compressed&&!strcmp(codec,"bf16")?"1":"0",1);
                setenv("FG_FP16_HOPS",compressed&&!strcmp(codec,"fp16")?"1":
                    compressed&&!strcmp(codec,"fp16-128")?"128":"0",1);
                unsetenv("FG_FP16_HOP_MASK");
                memset(&w,0,sizeof(w));w.layer=(uint8_t)first;
                w.source_rank=first?m->layer_owner[first-1u]:0u;w.destination_rank=(uint8_t)rank;
                w.token_index=p;w.flags=!first?FG_LAYER_WORK_HAS_NGRAM:0u;
                for(uint32_t axis=0;axis<3;axis++)w.position[axis]=p;
                memcpy(w.hyper,h[s]+(size_t)p*FG_HYPER_WIDTH,sizeof(w.hyper));
                if(!first)memcpy(w.ngram_embedding,n+(size_t)p*FG_NGRAM_EMBED_VALUES,sizeof(w.ngram_embedding));
                uint32_t bytes=0;check(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&w,&error));
                check(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error));rounded+=compressed;
                if(compressed)for(uint32_t k=0;k<(!first?2u:1u);k++){
                    const float *a=k?w.ngram_embedding:w.hyper,*b=k?decoded.ngram_embedding:decoded.hyper;
                    for(uint32_t j=0;j<(k?FG_NGRAM_EMBED_VALUES:FG_HYPER_WIDTH);j++){
                        double delta=(double)a[j]-b[j];square_error[k]+=delta*delta;
                        square_reference[k]+=(double)a[j]*a[j];
                    }
                }
                check(fg_vk_tensor_write(input,0,decoded.hyper,sizeof(decoded.hyper),&error));
                if(ngram)check(fg_vk_tensor_write(ngram,0,decoded.ngram_embedding,sizeof(decoded.ngram_embedding),&error));
                fg_vk_tensor *output=NULL;
                check(fg_owner_decode_block_chained(owner,first,last,p,w.position,input,ngram,
                    expert,ex,&output,&error));
                check(fg_vk_tensor_read(output,0,h[s]+(size_t)p*FG_HYPER_WIDTH,sizeof(w.hyper),&error));
            }
            write_data(outputs[s],h[s],(size_t)nt*FG_HYPER_WIDTH*4u);
        }
        for(uint32_t p=0;p<nt-4u;p++)for(uint32_t i=0;i<FG_HYPER_WIDTH;i++){
            size_t j=(size_t)p*FG_HYPER_WIDTH+i;
            double e=fabs((double)h[0][j]-h[1][j])/fmax(1.0,fabs(h[0][j]));
            if(e>prefix_error)prefix_error=e;
        }
        printf("{\"kind\":\"stage_case\",\"first\":%u,\"last\":%u,\"case\":%u,\"tokens\":%u}\n",
            first,last,c,nt);fflush(stdout);
    }
    need(prefix_error<=1e-4,"f32 common-prefix oracle");
    for(uint32_t s=0;s<2;s++){
        need(fgetc(inputs[s])==EOF,"input trailing frames");fclose(inputs[s]);
        need(fclose(outputs[s])==0,"close stage output");free(h[s]);
    }
    if(ng)fclose(ng);
    if(saved){need(fgetc(saved)==EOF,"saved teacher trailing frames");fclose(saved);}
    free(n);free(cases);
    if(pe){free(pe->outputs);free(pe);}
    fg_vk_tensor_destroy(ngram);fg_expert_executor_destroy(ex);
    fg_owner_executor_destroy(owner);fg_model_close(model);
    for(uint32_t s=0;s<2;s++)unlink(states[s]);
    printf("{\"kind\":\"stage_complete\",\"first\":%u,\"last\":%u,\"rank\":%u,"
        "\"rounded_boundaries\":%llu,\"prefix_max_relative\":%.12g,"
        "\"codec\":\"%s\",\"hop_mask\":%lu,\"arithmetic_path\":\"%s\",\"microbatch\":%u,"
        "\"wire_hyper_relative_l2\":%.12g,"
        "\"wire_ngram_relative_l2\":%.12g}\n",
        first,last,rank,(unsigned long long)rounded,prefix_error,codec,mask,
        prefill?"prefill":"decode",m->prefill_microbatch,
        square_reference[0]?sqrt(square_error[0]/square_reference[0]):0,
        square_reference[1]?sqrt(square_error[1]/square_reference[1]):0);free(m);
}
static void head(const char *mp,const char *pack,const char *in){
    fg_manifest *m=manifest(mp);sample *cases=calloc(MAX_CASES,sizeof(*cases));
    need(cases!=NULL,"cases allocation");uint32_t count=load_cases(pack,cases);
    fg_model *model=NULL;check(fg_model_open(&model,m,pack,4u,&error));
    fg_vk_context *vk=fg_model_vk(model);fg_output_executor *output=NULL;
    check(fg_output_executor_create(&output,model,&error));
    fg_vk_tensor *input=NULL;check(fg_vk_tensor_create(vk,FG_HYPER_WIDTH*4u,&input,&error));
    FILE *inputs[2]={frames(in,"teacher","rb"),frames(in,"candidate","rb")};
    float *logits[2]={malloc(FG_Q38_VOCAB_SIZE*4u),malloc(FG_Q38_VOCAB_SIZE*4u)};
    need(logits[0]&&logits[1],"logits allocation");float hyper[FG_HYPER_WIDTH];
    for(uint32_t c=0;c<count;c++)for(uint32_t p=0;p<cases[c].count;p++){
        for(uint32_t s=0;s<2;s++){
            read_data(inputs[s],hyper,sizeof(hyper));
            if(p+4u<cases[c].count)continue;
            check(fg_vk_tensor_write(input,0,hyper,sizeof(hyper),&error));
            fg_vk_tensor *scores=NULL;check(fg_output_logits(output,input,&scores,&error));
            check(fg_vk_tensor_read(scores,0,logits[s],FG_Q38_VOCAB_SIZE*4u,&error));
        }
        if(p+4u<cases[c].count)continue;
        uint32_t top[2]={0,0};double mx[2]={-INFINITY,-INFINITY},z[2]={0,0};
        for(uint32_t s=0;s<2;s++){
            for(uint32_t j=0;j<FG_Q38_VOCAB_SIZE;j++){
                need(isfinite(logits[s][j]),"nonfinite logits");
                if(logits[s][j]>mx[s]){mx[s]=logits[s][j];top[s]=j;}
            }
            for(uint32_t j=0;j<FG_Q38_VOCAB_SIZE;j++)z[s]+=exp(logits[s][j]-mx[s]);
            z[s]=log(z[s])+mx[s];
        }
        double kl=0;
        for(uint32_t j=0;j<FG_Q38_VOCAB_SIZE;j++)
            kl+=exp(logits[0][j]-z[0])*((logits[0][j]-z[0])-(logits[1][j]-z[1]));
        uint8_t digest[2][32];char hex[2][65];
        for(uint32_t s=0;s<2;s++){
            fg_sha256 hash;fg_sha256_init(&hash);
            fg_sha256_update(&hash,logits[s],FG_Q38_VOCAB_SIZE*4u);fg_sha256_final(&hash,digest[s]);
            for(uint32_t j=0;j<32;j++)snprintf(hex[s]+j*2u,3,"%02x",digest[s][j]);
        }
        const char *logits_directory=getenv("HOP_OFFLINE_LOGITS");
        if(logits_directory&&p+1u==cases[c].count){
            char name[64];snprintf(name,sizeof(name),"%03u.fgl",c);
            FILE *f=file(logits_directory,name,"wbx");uint32_t vocab=FG_Q38_VOCAB_SIZE;
            write_data(f,"FGLOG01\n",8);write_data(f,&vocab,4);
            write_data(f,logits[1],FG_Q38_VOCAB_SIZE*4u);need(fclose(f)==0,"close saved logits");
        }
        printf("{\"kind\":\"quality\",\"case\":%u,\"position\":%u,\"final\":%s,"
            "\"vocab\":%u,\"kl\":%.15g,\"teacher_top1\":%u,\"bf16_top1\":%u,"
            "\"teacher_logits_sha256\":\"%s\",\"bf16_logits_sha256\":\"%s\"}\n",
            c,p,p+1u==cases[c].count?"true":"false",FG_Q38_VOCAB_SIZE,
            fmax(0,kl),top[0],top[1],hex[0],hex[1]);fflush(stdout);
    }
    for(uint32_t s=0;s<2;s++){need(fgetc(inputs[s])==EOF,"head trailing input");
        fclose(inputs[s]);free(logits[s]);}
    fg_vk_tensor_destroy(input);fg_output_executor_destroy(output);fg_model_close(model);
    free(cases);free(m);
}
int main(int argc,char **argv){
    if(argc==6&&!strcmp(argv[1],"prepare"))prepare(argv[2],argv[3],argv[4],argv[5]);
    else if(argc==8&&!strcmp(argv[1],"stage"))stage(argv[2],argv[3],
        (uint32_t)strtoul(argv[4],NULL,10),(uint32_t)strtoul(argv[5],NULL,10),argv[6],argv[7]);
    else if(argc==5&&!strcmp(argv[1],"head"))head(argv[2],argv[3],argv[4]);
    else if(argc==5&&!strcmp(argv[1],"tokenizer")){
        fg_manifest *m=manifest(argv[2]);fg_tokenizer *tok=tokenizer(argv[4],argv[3],m);
        check(fg_tokenizer_validate_qwen38(tok,&error));fg_tokenizer_close(tok);free(m);
        puts("Sealed tokenizer SHA and Qwen tokenizer validation PASS");
    }
    else if(argc==3&&!strcmp(argv[1],"plan")){
        fg_manifest *m=manifest(argv[2]);
        for(uint32_t first=0;first<FG_LAYER_COUNT;){
            uint32_t last=first;
            while(last+1u<FG_LAYER_COUNT&&m->layer_owner[last+1u]==m->layer_owner[first])last++;
            printf("%u %u %u\n",first,last,m->layer_owner[first]);first=last+1u;
        }
        free(m);
    }else{fprintf(stderr,"usage: bf16-offline prepare M P HEADER CORPUS | stage M P FIRST LAST IN OUT | head M P IN | tokenizer M P HEADER | plan M\n");return 2;}
    return 0;
}
