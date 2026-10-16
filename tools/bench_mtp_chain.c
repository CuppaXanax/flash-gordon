#include "fg_mtp_native.h"
#include "fg_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static fg_error error;
static void check(fg_status status){
    if(status!=FG_OK){fprintf(stderr,"local chain: %s\n",error.message);exit(1);}
}
static void need(bool ok,const char *message){
    if(!ok){fprintf(stderr,"local chain: %s\n",message);exit(1);}
}
static double now(void){
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1000.0+(double)t.tv_nsec*1e-6;
}
static FILE *open_case(const char *root,const char *name){
    char path[1600];
    need(snprintf(path,sizeof(path),"%s/%s",root,name)<(int)sizeof(path),"path length");
    FILE *f=fopen(path,"rb");need(f!=NULL,path);return f;
}
static void embedding_row(FILE *f,uint32_t token,float *out){
    uint8_t raw[FG_HIDDEN_SIZE/32u*34u];
    need(token<FG_Q38_VOCAB_SIZE&&!fseeko(f,(off_t)((uint64_t)token*sizeof(raw)),SEEK_SET)&&
        fread(raw,1,sizeof(raw),f)==sizeof(raw),"reference embedding");
    fg_dequantize_q8_0(raw,out,FG_HIDDEN_SIZE);
}
static uint32_t read_le32(FILE *file){
    uint8_t p[4];need(fread(p,1,4u,file)==4u,"subset integer");
    return (uint32_t)p[0]|(uint32_t)p[1]<<8u|
        (uint32_t)p[2]<<16u|(uint32_t)p[3]<<24u;
}
int main(int argc,char **argv){
    if(argc!=5&&argc!=6){
        fprintf(stderr,"usage: bench-mtp-chain CAPTURE MANIFEST OVERLAY EMBEDDING [DRAFT_HEAD_Q8_OR_FGDV]\n");
        return 2;
    }
    fg_manifest *manifest=malloc(sizeof(*manifest));need(manifest!=NULL,"manifest");
    check(fg_manifest_read(argv[2],manifest,&error));
    fg_vk_context *vk=NULL;check(fg_vk_open(&vk,&error));
    fg_mtp_native *native=NULL;
    check(fg_mtp_native_create(&native,vk,manifest,
        manifest->layer_owner[FG_LAYER_COUNT-1u],&error));
    fg_mtp_native_work config={.operation=FG_MTP_NATIVE_CONFIG,
        .overlay=argv[3],.embedding_file=argv[4],.count=(uint32_t)strlen(argv[3]),
        .argument=8192u,.limit=8192u};
    fg_mtp_native_ack ack;
    check(fg_mtp_native_run(native,0u,&config,NULL,&ack,&error));
    char name[1600];need(snprintf(name,sizeof(name),"%s/head.q8",argv[1])<(int)sizeof(name),"head path");
    const char *draft_head=argc==6?argv[5]:name;
    config=(fg_mtp_native_work){.operation=FG_MTP_NATIVE_LOCAL_HEAD,
        .overlay=draft_head,.count=(uint32_t)strlen(draft_head)};
    check(fg_mtp_native_run(native,0u,&config,NULL,&ack,&error));
    uint32_t head_rows=FG_Q38_VOCAB_SIZE,*ids=NULL;
    if(argc==6){
        FILE *subset=fopen(draft_head,"rb");need(subset!=NULL,"draft head");
        uint8_t magic[8],parent[32];
        need(fread(magic,1,8u,subset)==8u,"draft head header");
        if(!memcmp(magic,"FGDVOC1\0",8u)){
            head_rows=read_le32(subset);need(head_rows&&head_rows<=FG_Q38_VOCAB_SIZE,"subset rows");
            need(fread(parent,1,32u,subset)==32u,"subset parent");
            ids=malloc((size_t)head_rows*4u);need(ids!=NULL,"subset IDs");
            for(uint32_t i=0;i<head_rows;i++)ids[i]=read_le32(subset);
        }
        fclose(subset);
    }
    fg_vk_context *reference_vk=NULL;check(fg_vk_open(&reference_vk,&error));
    fg_mtp_head *reference=NULL;
    check(fg_mtp_head_open(&reference,reference_vk,manifest,
        manifest->layer_owner[FG_LAYER_COUNT-1u],argv[3],8192u,&error));
    fg_mtp_vocab *reference_vocab=NULL;
    check(fg_mtp_vocab_open(&reference_vocab,reference_vk,name,ids,head_rows,&error));
    free(ids);
    fg_vk_tensor *input=NULL,*embedding=NULL;
    check(fg_vk_tensor_create(reference_vk,FG_HYPER_WIDTH*4u,&input,&error));
    check(fg_vk_tensor_create(reference_vk,FG_HIDDEN_SIZE*4u,&embedding,&error));
    FILE *embeddings=open_case(argv[1],"embedding.q8");
    FILE *index=open_case(argv[1],"cases.tsv");
    char id[48],category[8];unsigned long long session;
    uint32_t prompt,generated,rank,cases=0u,checked=0u,frozen_equal=0u;
    while(fscanf(index,"%47s %7s %llu %u %u %u",id,category,&session,
        &prompt,&generated,&rank)==6){
        need(rank==manifest->layer_owner[FG_LAYER_COUNT-1u]&&generated==259u&&prompt,"case geometry");
        uint32_t rows=prompt+generated,slot=cases%2u;
        uint32_t *tokens=malloc((size_t)rows*4u);
        float *hc=malloc((size_t)rows*FG_HYPER_WIDTH*4u);
        uint32_t proposals[256u*3u];
        need(tokens&&hc,"case allocation");
        snprintf(name,sizeof(name),"%s.tokens.u32",id);FILE *f=open_case(argv[1],name);
        need(fread(tokens,4u,rows,f)==rows&&fgetc(f)==EOF,"tokens");fclose(f);
        snprintf(name,sizeof(name),"hc-%llu.f32",session);f=open_case(argv[1],name);
        need(fread(hc,FG_HYPER_WIDTH*4u,rows,f)==rows&&fgetc(f)==EOF,"hidden states");fclose(f);
        snprintf(name,sizeof(name),"%s.drafts.u32",id);f=open_case(argv[1],name);
        need(fread(proposals,4u,256u*3u,f)==256u*3u&&fgetc(f)==EOF,"frozen proposals");fclose(f);
        for(uint32_t mode=0;mode<(argc==6?3u:2u);mode++){
        check(fg_mtp_native_reset(native,slot,&error));
        fg_mtp_native_work prefix={.operation=FG_MTP_NATIVE_PREFIX,.limit=prompt};
        check(fg_mtp_native_run(native,slot,&prefix,NULL,&ack,&error));
        check(fg_mtp_native_observe_prefill(native,slot,0u,prompt,hc,tokens+1u,&error));
        check(fg_mtp_head_reset(reference,0u,&error));
        float emb[FG_HIDDEN_SIZE];fg_vk_tensor *unused=NULL,*contracted=NULL;
        for(uint32_t i=0;i+1u<prompt;i++){
            embedding_row(embeddings,tokens[i+1u],emb);
            check(fg_vk_tensor_write(input,0u,hc+(size_t)i*FG_HYPER_WIDTH,FG_HYPER_WIDTH*4u,&error));
            check(fg_vk_tensor_write(embedding,0u,emb,sizeof(emb),&error));
            check(fg_mtp_head_step_replay(reference,input,embedding,i,&unused,&contracted,&error));
        }
        uint32_t position=0u,round=0u,limit=mode==1u?64u:256u;
        while(position<limit){
            uint32_t first=prompt+position,keep=mode?1u+round%4u:1u;
            if(keep>limit-position)keep=limit-position;
            fg_mtp_chain_result result;double start=now();
            check(fg_mtp_native_chain(native,slot,first,tokens[first],3u,&result,&error));
            double draft_ms=now()-start;
            uint32_t pending=tokens[first],matched=0u,prefix_match=1u;
            double reference_vocab_ms=0;
            fg_vk_tensor *hyper=input;
            check(fg_vk_tensor_write(input,0u,hc+(size_t)(first-1u)*FG_HYPER_WIDTH,
                FG_HYPER_WIDTH*4u,&error));
            for(uint32_t j=0;j<3u;j++){
                embedding_row(embeddings,pending,emb);
                check(fg_vk_tensor_write(embedding,0u,emb,sizeof(emb),&error));
                check(fg_mtp_head_step_replay(reference,hyper,embedding,first-1u+j,
                    &hyper,&contracted,&error));
                float score;
                start=now();
                check(fg_mtp_vocab_argmax(reference_vocab,contracted,&pending,&score,&error));
                reference_vocab_ms+=now()-start;
                need(pending==result.tokens[j],"local chain differs from direct current reference");
                frozen_equal+=pending==proposals[position*3u+j];
                prefix_match&=pending==tokens[first+1u+j];matched+=prefix_match;
            }
            if(mode==2u){
                /* Counterfactual accepted-frontier replay against the N1 teacher. */
                keep=matched+1u;
                if(keep>limit-position)keep=limit-position;
            }
            checked+=3u;
            check(fg_mtp_native_observe_verify(native,slot,first,4u,
                hc+(size_t)first*FG_HYPER_WIDTH,&error));
            fg_mtp_native_work finish={.operation=FG_MTP_NATIVE_FINISH,.first_token=first,
                .count=keep,.tokens=tokens+first+1u};
            start=now();
            check(fg_mtp_native_run(native,slot,&finish,NULL,&ack,&error));
            double repair_ms=now()-start;
            check(fg_mtp_head_truncate(reference,first,&error));
            for(uint32_t i=0;i+1u<keep;i++){
                embedding_row(embeddings,tokens[first+i+1u],emb);
                check(fg_vk_tensor_write(input,0u,hc+(size_t)(first+i)*FG_HYPER_WIDTH,
                    FG_HYPER_WIDTH*4u,&error));
                check(fg_vk_tensor_write(embedding,0u,emb,sizeof(emb),&error));
                check(fg_mtp_head_step_replay(reference,input,embedding,first+i,
                    &unused,&contracted,&error));
            }
            printf("{\"case\":\"%s\",\"category\":\"%s\",\"mode\":%u,\"head_rows\":%u,\"position\":%u,\"k\":3,"
                "\"round\":%u,\"keep\":%u,\"accepted\":%u,\"draft_ms\":%.9f,"
                "\"native_repair_ms\":%.9f,\"reference_vocab_ms\":%.9f,\"reference_equal\":true}\n",
                id,category,mode,head_rows,position,round,keep,matched,draft_ms,repair_ms,reference_vocab_ms);
            position+=keep;round++;
        }
        }
        fflush(stdout);free(tokens);free(hc);cases++;
    }
    need(cases==40u,"all forty authentic cases");
    fg_vk_memory_stats memory;fg_vk_get_memory_stats(vk,&memory);
    printf("{\"cases\":%u,\"draft_tokens_reference_equal\":%u,\"frozen_draft_ids_equal\":%u,"
        "\"requested_live_bytes\":%llu}\n",cases,checked,frozen_equal,
        (unsigned long long)memory.requested_live_bytes);
    fclose(index);fclose(embeddings);fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(embedding);
    fg_mtp_vocab_close(reference_vocab);fg_mtp_head_close(reference);fg_vk_close(reference_vk);
    fg_mtp_native_close(native);fg_vk_close(vk);free(manifest);
    return 0;
}
