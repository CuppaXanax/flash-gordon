#include "fg_mtp_native.h"
#include "fg_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static fg_error error;
static void check(fg_status status){
    if(status!=FG_OK){fprintf(stderr,"native replay: %s\n",error.message);exit(1);}
}
static void need(bool ok,const char *what){
    if(!ok){fprintf(stderr,"native replay: %s\n",what);exit(1);}
}
static FILE *file(const char *dir,const char *name){
    char path[1600];need(snprintf(path,sizeof(path),"%s/%s",dir,name)<(int)sizeof(path),"path");
    FILE *f=fopen(path,"rb");need(f!=NULL,path);return f;
}
static void embed(FILE *f,uint32_t token,float out[FG_HIDDEN_SIZE]){
    uint8_t raw[FG_HIDDEN_SIZE/32u*34u];
    need(!fseeko(f,(off_t)((uint64_t)token*sizeof(raw)),SEEK_SET)&&
        fread(raw,1,sizeof(raw),f)==sizeof(raw),"embedding row");
    fg_dequantize_q8_0(raw,out,FG_HIDDEN_SIZE);
}
int main(int argc,char **argv){
    if(argc!=5&&argc!=6){
        fprintf(stderr,"usage: mtp-native-replay CAPTURE MANIFEST OVERLAY EMBEDDING [MAX_CASES]\n");
        return 2;
    }
    uint32_t maximum=argc==6?(uint32_t)strtoul(argv[5],NULL,10):40u;
    fg_manifest *m=malloc(sizeof(*m));need(m!=NULL,"manifest");
    check(fg_manifest_read(argv[2],m,&error));
    fg_vk_context *vk=NULL,*reference_vk=NULL;
    check(fg_vk_open(&vk,&error));check(fg_vk_open(&reference_vk,&error));
    fg_mtp_native *native=NULL;
    check(fg_mtp_native_create(&native,vk,m,m->layer_owner[FG_LAYER_COUNT-1u],&error));
    fg_mtp_native_work config={.operation=FG_MTP_NATIVE_CONFIG,.overlay=argv[3],
        .embedding_file=argv[4],.count=(uint32_t)strlen(argv[3]),
        .argument=8192u,.limit=8192u};
    fg_mtp_native_ack ack;
    check(fg_mtp_native_run(native,0,&config,NULL,&ack,&error));
    fg_mtp_head *reference=NULL;
    check(fg_mtp_head_open(&reference,reference_vk,m,m->layer_owner[FG_LAYER_COUNT-1u],
        argv[3],8192u,&error));
    char path[1600];snprintf(path,sizeof(path),"%s/head.q8",argv[1]);
    fg_mtp_vocab *vocab=NULL;
    check(fg_mtp_vocab_open(&vocab,reference_vk,path,NULL,FG_Q38_VOCAB_SIZE,&error));
    fg_vk_tensor *input=NULL,*embedding=NULL;
    check(fg_vk_tensor_create(reference_vk,FG_HYPER_WIDTH*4u,&input,&error));
    check(fg_vk_tensor_create(reference_vk,FG_HIDDEN_SIZE*4u,&embedding,&error));
    FILE *index=file(argv[1],"cases.tsv"),*embeddings=file(argv[1],"embedding.q8");
    char id[48],category[8];unsigned long long session;
    uint32_t prompt,generated,rank,cases=0,exact=0,retained[5]={0};
    while(cases<maximum&&fscanf(index,"%47s %7s %llu %u %u %u",id,category,&session,
        &prompt,&generated,&rank)==6){
        need(rank==m->layer_owner[FG_LAYER_COUNT-1u]&&generated==259u&&prompt,"case");
        uint32_t rows=prompt+generated,*tokens=malloc((size_t)rows*4u);
        float *hc=malloc((size_t)rows*FG_HYPER_WIDTH*4u);
        uint32_t *next_tokens=malloc((size_t)prompt*4u);
        need(tokens&&hc&&next_tokens,"case allocation");
        snprintf(path,sizeof(path),"%s.tokens.u32",id);FILE *f=file(argv[1],path);
        need(fread(tokens,4u,rows,f)==rows&&fgetc(f)==EOF,"tokens");fclose(f);
        snprintf(path,sizeof(path),"hc-%llu.f32",session);f=file(argv[1],path);
        need(fread(hc,FG_HYPER_WIDTH*4u,rows,f)==rows&&fgetc(f)==EOF,"HC");fclose(f);
        for(uint32_t i=0;i<prompt;i++)next_tokens[i]=tokens[i+1u];
        for(uint32_t mode=0;mode<2u;mode++){
            uint32_t slot=cases%2u;
            check(fg_mtp_native_reset(native,slot,&error));
            check(fg_mtp_head_reset(reference,0,&error));
            fg_mtp_native_work prefix={.operation=FG_MTP_NATIVE_PREFIX,.limit=prompt};
            check(fg_mtp_native_run(native,slot,&prefix,NULL,&ack,&error));
            check(fg_mtp_native_observe_prefill(native,slot,0,prompt,hc,next_tokens,&error));
            float emb[FG_HIDDEN_SIZE];
            for(uint32_t c=0;c+1u<prompt;c++){
                embed(embeddings,tokens[c+1u],emb);
                check(fg_vk_tensor_write(input,0,hc+(size_t)c*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,&error));
                check(fg_vk_tensor_write(embedding,0,emb,sizeof(emb),&error));
                fg_vk_tensor *unused_hyper=NULL,*unused_hidden=NULL;
                check(fg_mtp_head_step_replay(reference,input,embedding,c,
                    &unused_hyper,&unused_hidden,&error));
            }
            uint32_t position=0,cycle=0;
            while(position<256u){
                uint32_t first=prompt+position,cell=first-1u,pending=tokens[first];
                fg_vk_tensor *hyper=input,*contracted=NULL;
                check(fg_vk_tensor_write(input,0,hc+(size_t)cell*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,&error));
                for(uint32_t j=0;j<3u;j++){
                    float actual[FG_HIDDEN_SIZE],expected[FG_HIDDEN_SIZE],score;
                    fg_mtp_native_work draft={.operation=FG_MTP_NATIVE_DRAFT,
                        .first_token=first+j,.count=1u,.argument=j,.tokens=&pending};
                    check(fg_mtp_native_run(native,slot,&draft,actual,&ack,&error));
                    embed(embeddings,pending,emb);
                    check(fg_vk_tensor_write(embedding,0,emb,sizeof(emb),&error));
                    check(fg_mtp_head_step_replay(reference,hyper,embedding,cell+j,&hyper,&contracted,&error));
                    check(fg_vk_tensor_read(contracted,0,expected,sizeof(expected),&error));
                    if(memcmp(actual,expected,sizeof(actual))){
                        float max_error=0;
                        for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)max_error=fmaxf(max_error,fabsf(actual[i]-expected[i]));
                        fprintf(stderr,"HC/token/position mismatch case=%s mode=%u anchor=%u draft=%u error=%g\n",
                            id,mode,position,j,max_error);return 1;
                    }
                    exact++;
                    check(fg_mtp_vocab_argmax(vocab,contracted,&pending,&score,&error));
                }
                uint32_t keep=mode?1u+cycle++%4u:1u;
                if(keep>256u-position)keep=256u-position;
                check(fg_mtp_native_observe_verify(native,slot,first,4u,
                    hc+(size_t)first*FG_HYPER_WIDTH,&error));
                fg_mtp_native_work finish={.operation=FG_MTP_NATIVE_FINISH,.first_token=first,
                    .count=keep,.tokens=tokens+first+1u};
                check(fg_mtp_native_run(native,slot,&finish,NULL,&ack,&error));
                check(fg_mtp_head_truncate(reference,first,&error));
                for(uint32_t i=0;i+1u<keep;i++){
                    embed(embeddings,tokens[first+i+1u],emb);
                    check(fg_vk_tensor_write(input,0,hc+(size_t)(first+i)*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,&error));
                    check(fg_vk_tensor_write(embedding,0,emb,sizeof(emb),&error));
                    fg_vk_tensor *unused_hyper=NULL,*unused_hidden=NULL;
                    check(fg_mtp_head_step_replay(reference,input,embedding,first+i,
                        &unused_hyper,&unused_hidden,&error));
                }
                retained[keep]++;position+=keep;
            }
        }
        printf("{\"case\":\"%s\",\"category\":\"%s\",\"exact_steps_total\":%u}\n",id,category,exact);
        fflush(stdout);free(tokens);free(hc);free(next_tokens);cases++;
    }
    need(cases==maximum,"case count");
    printf("{\"cases\":%u,\"bit_exact_contracted_steps\":%u,\"retain_1_2_3_4\":[%u,%u,%u,%u],"
        "\"scope\":\"trained native wrapper vs direct authoritative-HC replay; not whole-target logits\"}\n",
        cases,exact,retained[1],retained[2],retained[3],retained[4]);
    fclose(index);fclose(embeddings);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(embedding);
    fg_mtp_vocab_close(vocab);fg_mtp_head_close(reference);fg_mtp_native_close(native);
    fg_vk_close(reference_vk);fg_vk_close(vk);free(m);return 0;
}
