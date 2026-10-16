#include "fg_mtp_head.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void binding_test(fg_vk_context *vk,fg_error *err){
    fg_manifest *m=malloc(sizeof(*m));
    if(!m){fprintf(stderr,"manifest OOM\n");exit(1);}
    fg_manifest_init(m);m->tensor_count=1;
    fg_tensor_record *r=&m->tensors[0];
    snprintf(r->name,sizeof(r->name),"borrowed.weight");
    r->bytes=16;r->ggml_type=0;r->dims=1;r->shape[0]=4;
    fg_vk_tensor *w=NULL;
    if(fg_vk_tensor_create(vk,16,&w,err)!=FG_OK){fprintf(stderr,"%s\n",err->message);exit(1);}
    float input[4]={1,2,3,4},output[4]={0};
    if(fg_vk_tensor_write(w,0,input,sizeof(input),err)!=FG_OK)exit(1);
    fg_model *view=NULL;fg_vk_tensor *weights[]={w};
    if(fg_model_bind_view(&view,vk,m,0,weights,err)!=FG_OK)exit(1);
    fg_vk_tensor_destroy(w);
    if(fg_vk_tensor_read(fg_model_tensor(view,r->name),0,output,sizeof(output),err)!=FG_OK||
       memcmp(input,output,sizeof(input))||fg_model_weight_bytes(view)!=16)exit(1);
    fg_model_close(view);
    if(fg_vk_tensor_create(vk,16,&w,err)!=FG_OK)exit(1);
    fg_vk_tensor_destroy(w);
    m->layer_owner[FG_LAYER_COUNT-1u]=2;
    fg_mtp_head *h=NULL;
    if(fg_mtp_head_open(&h,vk,m,1,"does-not-exist",16,err)!=FG_ERR_ARGUMENT||h){
        fprintf(stderr,"MTP opened on the wrong final-layer owner\n");exit(1);
    }
    free(m);
    printf("{\"kind\":\"borrowed_model_test\",\"retained_tensor\":true,\"context_alive\":true,\"wrong_owner_rejected\":true}\n");
}

static void check(fg_status status,fg_error *err){
    if(status!=FG_OK){fprintf(stderr,"MTP head: %s\n",err->message);exit(1);}
}
static double now_us(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3;
}
static fg_vk_tensor *tensor(fg_vk_context *vk,uint64_t values,fg_error *err){
    fg_vk_tensor *t=NULL;check(fg_vk_tensor_create(vk,values*4u,&t,err),err);return t;
}
static fg_status dump_stage(void *context,const char *stage,const fg_vk_tensor *values,
    uint32_t count,fg_error *err){
    FILE *f=context;uint32_t length=(uint32_t)strlen(stage);
    float *data=malloc((size_t)count*4u);
    if(!data){fg_error_set(err,FG_ERR_OOM,"native trace");return FG_ERR_OOM;}
    fg_status status=fg_vk_tensor_read(values,0,data,(uint64_t)count*4u,err);
    if(status==FG_OK&&(fwrite(&length,4,1,f)!=1||fwrite(&count,4,1,f)!=1||
       fwrite(stage,1,length,f)!=length||fwrite(data,4,count,f)!=count)){
        fg_error_set(err,FG_ERR_IO,"write native trace");status=FG_ERR_IO;
    }
    free(data);return status;
}
static float signal(uint64_t i,uint32_t seed){
    return sinf((float)(i+1u)*(0.0031f+seed*0.00017f))*0.25f;
}
static bool parse_count(const char *text,uint32_t *value){
    if(*text<'0'||*text>'9')return false;
    char *end=NULL;errno=0;unsigned long parsed=strtoul(text,&end,10);
    if(errno||*end||parsed>1024u)return false;
    *value=(uint32_t)parsed;return true;
}

static void attention_test(fg_vk_context *vk,fg_error *err){
    fg_vk_tensor *q=tensor(vk,6144,err),*gate=tensor(vk,6144,err);
    fg_vk_tensor *key=tensor(vk,512,err),*value=tensor(vk,512,err);
    fg_vk_tensor *keys=tensor(vk,16u*512u,err),*values=tensor(vk,16u*512u,err);
    fg_vk_tensor *out=tensor(vk,6144,err);
    float queries[6144],gates[6144],ks[16][512],vs[16][512],actual[6144];
    for(uint32_t i=0;i<6144u;i++){queries[i]=signal(i,3);gates[i]=signal(i,7);}
    check(fg_vk_tensor_write(q,0,queries,sizeof(queries),err),err);
    check(fg_vk_tensor_write(gate,0,gates,sizeof(gates),err),err);
    double maximum=0;
    for(uint32_t cell=0;cell<16u;cell++){
        for(uint32_t j=0;j<512u;j++){ks[cell][j]=signal(j,cell+11);vs[cell][j]=signal(j,cell+31);}
        check(fg_vk_tensor_write(key,0,ks[cell],sizeof(ks[cell]),err),err);
        check(fg_vk_tensor_write(value,0,vs[cell],sizeof(vs[cell]),err),err);
        check(fg_vk_mtp_attention(vk,out,keys,values,q,gate,key,value,cell,err),err);
        check(fg_vk_tensor_read(out,0,actual,sizeof(actual),err),err);
        for(uint32_t head=0;head<24u;head++){
            double scores[16],sum=0,max=-INFINITY;uint32_t kv=head/12u;
            for(uint32_t t=0;t<=cell;t++){
                double dot=0;
                for(uint32_t j=0;j<256u;j++)dot+=(double)queries[head*256u+j]*ks[t][kv*256u+j];
                scores[t]=dot/16.0;if(scores[t]>max)max=scores[t];
            }
            for(uint32_t t=0;t<=cell;t++){scores[t]=exp(scores[t]-max);sum+=scores[t];}
            for(uint32_t j=0;j<256u;j++){
                double weighted=0;
                for(uint32_t t=0;t<=cell;t++)weighted+=scores[t]*vs[t][kv*256u+j];
                double expected=(weighted/sum)/(1.0+exp(-(double)gates[head*256u+j]));
                double delta=fabs(actual[head*256u+j]-expected);
                if(!isfinite(actual[head*256u+j])||delta>1e-5){
                    fprintf(stderr,"dense draft attention mismatch cell=%u head=%u dim=%u error=%g\n",cell,head,j,delta);exit(1);
                }
                if(delta>maximum)maximum=delta;
            }
        }
    }
    float suffix[6144],prefix[6144];
    check(fg_vk_tensor_read(out,0,suffix,sizeof(suffix),err),err);
    check(fg_vk_tensor_write(key,0,ks[3],sizeof(ks[3]),err),err);
    check(fg_vk_tensor_write(value,0,vs[3],sizeof(vs[3]),err),err);
    check(fg_vk_mtp_attention(vk,out,keys,values,q,gate,key,value,3u,err),err);
    check(fg_vk_tensor_read(out,0,prefix,sizeof(prefix),err),err);
    if(!memcmp(prefix,suffix,sizeof(prefix))){fprintf(stderr,"draft attention ignored frontier\n");exit(1);}
    /* Rebuild a short prefix over hostile future values: they must not enter
     * the causal result after a frontier truncation. */
    memset((char *)fg_vk_tensor_map(keys)+4u*512u*4u,0x7f,12u*512u*4u);
    memset((char *)fg_vk_tensor_map(values)+4u*512u*4u,0x7f,12u*512u*4u);
    check(fg_vk_mtp_attention(vk,out,keys,values,q,gate,key,value,3u,err),err);
    check(fg_vk_tensor_read(out,0,suffix,sizeof(suffix),err),err);
    if(memcmp(prefix,suffix,sizeof(prefix))){fprintf(stderr,"draft attention read future KV\n");exit(1);}
    fg_vk_tensor *all[]={q,gate,key,value,keys,values,out};
    for(uint32_t i=0;i<sizeof(all)/sizeof(all[0]);i++)fg_vk_tensor_destroy(all[i]);
    printf("{\"kind\":\"dense_draft_attention_test\",\"rows\":16,\"max_abs\":%.9g,\"future_mask_exact\":true}\n",maximum);
}

static fg_vk_tensor *head_weight(fg_vk_context *vk,const char *path,fg_error *err){
    if(!path)return NULL;
    uint64_t bytes=(uint64_t)FG_Q38_VOCAB_SIZE*FG_HIDDEN_SIZE/32u*34u;
    FILE *f=fopen(path,"rb");if(!f){perror(path);exit(1);}
    fg_vk_tensor *w=NULL;check(fg_vk_tensor_create(vk,bytes,&w,err),err);
    if(fread(fg_vk_tensor_map(w),1,(size_t)bytes,f)!=(size_t)bytes||fgetc(f)!=EOF){
        fprintf(stderr,"invalid shared-head Q8 span\n");exit(1);
    }
    fclose(f);return w;
}

int main(int argc,char **argv){
    const char *overlay=NULL,*head_path=NULL,*embedding_path=NULL,*dump_path=NULL;uint32_t iterations=32,warmup=16;
    bool self_test=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--self-test"))self_test=true;
        else if(i+1<argc&&!strcmp(argv[i],"--overlay"))overlay=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--head-q8"))head_path=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--embedding-f32"))embedding_path=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--dump-first"))dump_path=argv[++i];
        else if(i+1<argc&&!strcmp(argv[i],"--iterations")){
            if(!parse_count(argv[++i],&iterations))return 2;
        }else if(i+1<argc&&!strcmp(argv[i],"--warmup")){
            if(!parse_count(argv[++i],&warmup))return 2;
        }
        else {fprintf(stderr,"invalid argument %s\n",argv[i]);return 2;}
    }
    if(!iterations||(!warmup&&dump_path)||(!self_test&&!overlay)){
        fprintf(stderr,"usage: bench-mtp-head --self-test | --overlay FILE [--head-q8 FILE]\n");return 2;
    }
    fg_error err={0};fg_vk_context *vk=NULL;check(fg_vk_open(&vk,&err),&err);
    if(self_test){binding_test(vk,&err);attention_test(vk,&err);}
    if(overlay){
        fg_manifest *manifest=malloc(sizeof(*manifest));
        if(!manifest){fprintf(stderr,"manifest OOM\n");return 1;}
        fg_manifest_init(manifest);
        uint32_t rank=manifest->layer_owner[FG_LAYER_COUNT-1u];
        fg_mtp_head *h=NULL;check(fg_mtp_head_open(&h,vk,manifest,rank,overlay,warmup+iterations,&err),&err);
        fg_vk_tensor *hyper=tensor(vk,10240,&err),*embedding=tensor(vk,2560,&err);
        fg_vk_tensor *vocab=head_weight(vk,head_path,&err),*logits=vocab?tensor(vk,FG_Q38_VOCAB_SIZE,&err):NULL;
        float hc[10240],emb[2560],hidden[2560],*scores=vocab?malloc(FG_Q38_VOCAB_SIZE*4u):NULL;
        if(vocab&&!scores){fprintf(stderr,"logit OOM\n");return 1;}
        float actual_embedding[2560];
        if(embedding_path){
            FILE *f=fopen(embedding_path,"rb");if(!f){perror(embedding_path);return 1;}
            if(fread(actual_embedding,1,sizeof(actual_embedding),f)!=sizeof(actual_embedding)||fgetc(f)!=EOF){
                fprintf(stderr,"invalid embedding span\n");return 1;
            }
            fclose(f);
        }
        double core=0,projection=0;uint32_t last_token=UINT32_MAX;double checksum=0;
        FILE *trace_file=NULL;
        if(dump_path){
            char trace_path[1200];
            if(snprintf(trace_path,sizeof(trace_path),"%s.trace",dump_path)>=(int)sizeof(trace_path))return 1;
            trace_file=fopen(trace_path,"wbx");if(!trace_file){perror(trace_path);return 1;}
            fg_mtp_head_set_trace(h,dump_stage,trace_file);
        }
        for(uint32_t cell=0;cell<warmup+iterations;cell++){
            for(uint32_t j=0;j<10240u;j++)hc[j]=signal(j,cell+5u);
            for(uint32_t j=0;j<2560u;j++)emb[j]=embedding_path?actual_embedding[j]:signal(j,cell+13u);
            check(fg_vk_tensor_write(hyper,0,hc,sizeof(hc),&err),&err);
            check(fg_vk_tensor_write(embedding,0,emb,sizeof(emb),&err),&err);
            fg_vk_tensor *next=NULL,*contracted=NULL;double start=now_us();
            check(fg_mtp_head_step(h,hyper,embedding,cell,&next,&contracted,&err),&err);
            if(!cell&&trace_file){
                fg_mtp_head_set_trace(h,NULL,NULL);
                if(fclose(trace_file))return 1;
                trace_file=NULL;
            }
            double done=now_us();
            if(vocab){
                check(fg_vk_dense_q8_0_f32(vk,logits,vocab,contracted,FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE,1u,1.0f,&err),&err);
                check(fg_vk_tensor_read(logits,0,scores,FG_Q38_VOCAB_SIZE*4u,&err),&err);
                float best=-INFINITY;last_token=UINT32_MAX;
                for(uint32_t j=0;j<FG_Q38_VOCAB_SIZE;j++){
                    if(!isfinite(scores[j])){fprintf(stderr,"nonfinite MTP logit\n");return 1;}
                    if(scores[j]>best){best=scores[j];last_token=j;}
                }
            }
            double stop=now_us();
            check(fg_vk_tensor_read(contracted,0,hidden,sizeof(hidden),&err),&err);
            if(!cell&&dump_path){
                float final_hc[10240];check(fg_vk_tensor_read(next,0,final_hc,sizeof(final_hc),&err),&err);
                FILE *f=fopen(dump_path,"wbx");
                bool ok=f&&fwrite(hc,1,sizeof(hc),f)==sizeof(hc)&&
                    fwrite(emb,1,sizeof(emb),f)==sizeof(emb)&&
                    fwrite(final_hc,1,sizeof(final_hc),f)==sizeof(final_hc)&&
                    fwrite(hidden,1,sizeof(hidden),f)==sizeof(hidden);
                if(vocab&&ok)ok=fwrite(scores,4,FG_Q38_VOCAB_SIZE,f)==FG_Q38_VOCAB_SIZE;
                if(f&&fclose(f))ok=false;
                if(!ok){fprintf(stderr,"write native oracle dump\n");return 1;}
            }
            for(uint32_t j=0;j<2560u;j++){
                if(!isfinite(hidden[j])){fprintf(stderr,"nonfinite MTP hidden\n");return 1;}
                checksum+=hidden[j];
            }
            if(cell>=warmup){core+=done-start;projection+=stop-done;}
        }
        /* The last cell must rebuild identically after truncation, including
         * its current KV row. This is draft-cache repair, not target rollback. */
        uint32_t last=warmup+iterations-1u;
        fg_vk_tensor *next=NULL,*contracted=NULL;
        if(fg_mtp_head_truncate(h,last+2u,&err)!=FG_ERR_ARGUMENT||
           fg_mtp_head_frontier(h)!=last+1u||
           fg_mtp_head_step(h,hyper,embedding,last+1u,&next,&contracted,&err)!=FG_ERR_ARGUMENT||
           fg_mtp_head_frontier(h)!=last+1u){
            fprintf(stderr,"native draft frontier/capacity error changed state\n");return 1;
        }
        check(fg_mtp_head_truncate(h,last,&err),&err);
        check(fg_mtp_head_step(h,hyper,embedding,last,&next,&contracted,&err),&err);
        float replay[2560];check(fg_vk_tensor_read(contracted,0,replay,sizeof(replay),&err),&err);
        if(memcmp(hidden,replay,sizeof(hidden))){fprintf(stderr,"native draft KV repair mismatch\n");return 1;}
        printf("{\"kind\":\"native_mtp_cost\",\"synthetic_hc\":true,\"acceptance_measured\":false,"
               "\"owner_rank\":%u,\"weight_bytes\":%llu,\"warmup\":%u,\"iterations\":%u,"
               "\"mean_core_us\":%.6f,\"mean_full_vocab_projection_us\":%.6f,"
               "\"shared_head_present\":%s,\"real_embedding\":%s,\"draft_token\":%u,\"repair_bit_exact\":true,\"checksum\":%.9g}\n",
               rank,(unsigned long long)fg_mtp_head_weight_bytes(h),warmup,iterations,
               core/iterations,projection/iterations,vocab?"true":"false",embedding_path?"true":"false",last_token,checksum);
        free(scores);fg_vk_tensor_destroy(logits);fg_vk_tensor_destroy(vocab);
        fg_vk_tensor_destroy(embedding);fg_vk_tensor_destroy(hyper);
        fg_mtp_head_close(h);free(manifest);
        /* A borrowed model must not close the caller's Vulkan context. */
        fg_vk_tensor *alive=tensor(vk,4,&err);fg_vk_tensor_destroy(alive);
    }
    fg_vk_close(vk);return 0;
}
