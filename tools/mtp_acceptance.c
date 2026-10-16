#include "fg_mtp_head.h"
#include "fg_quant.h"
#include "fg_protocol.h"
#include "fg_q38_schema.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void check(fg_status s,fg_error *e){
    if(s!=FG_OK){fprintf(stderr,"acceptance: %s\n",e->message);exit(1);}
}
static void require(bool ok,const char *message){
    if(!ok){fprintf(stderr,"acceptance: %s\n",message);exit(1);}
}
static void embedding(FILE *f,uint32_t token,float values[2560]){
    uint8_t row[2560u/32u*34u];
    require(token<FG_Q38_VOCAB_SIZE&&
        !fseeko(f,(off_t)((uint64_t)token*sizeof(row)),SEEK_SET)&&
        fread(row,1,sizeof(row),f)==sizeof(row),"production embedding row");
    fg_dequantize_q8_0(row,values,2560u);
}
int main(int argc,char **argv){
    if(argc!=4){
        fprintf(stderr,"usage: mtp-acceptance CAPTURE_DIR MANIFEST OVERLAY\n");return 2;
    }
    const char *dir=argv[1];
    char path[1200];
    fg_error err={0};fg_vk_context *vk=NULL;
    fg_manifest *manifest=malloc(sizeof(*manifest));
    require(manifest!=NULL,"manifest allocation");
    check(fg_manifest_read(argv[2],manifest,&err),&err);
    uint32_t rank=manifest->layer_owner[FG_LAYER_COUNT-1u];
    check(fg_vk_open(&vk,&err),&err);
    fg_mtp_head *head=NULL;
    check(fg_mtp_head_open(&head,vk,manifest,rank,argv[3],8192u,&err),&err);
    snprintf(path,sizeof(path),"%s/head.q8",dir);
    fg_mtp_vocab *vocab=NULL;
    check(fg_mtp_vocab_open(&vocab,vk,path,NULL,FG_Q38_VOCAB_SIZE,&err),&err);
    fg_vk_tensor *input=NULL,*embed=NULL;
    check(fg_vk_tensor_create(vk,FG_HYPER_WIDTH*4u,&input,&err),&err);
    check(fg_vk_tensor_create(vk,2560u*4u,&embed,&err),&err);
    snprintf(path,sizeof(path),"%s/embedding.q8",dir);FILE *ef=fopen(path,"rb");
    snprintf(path,sizeof(path),"%s/cases.tsv",dir);FILE *index=fopen(path,"r");
    snprintf(path,sizeof(path),"%s/acceptance.jsonl",dir);FILE *out=fopen(path,"w");
    require(ef&&index&&out,"input/output open");
    char id[48],category[8];unsigned long long session;
    uint32_t prompt,generated,captured_rank,cases=0u;
    while(fscanf(index,"%47s %7s %llu %u %u %u",id,category,&session,
                 &prompt,&generated,&captured_rank)==6){
        require(strspn(id,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")==strlen(id),
                "unsafe case id");
        require(captured_rank==rank&&generated==259u&&prompt&&prompt+259u+3u<=8192u,
                "capture identity/dimensions");
        uint32_t *tokens=malloc((prompt+generated)*4u),drafts[256u*3u];
        require(tokens!=NULL,"tokens allocation");
        snprintf(path,sizeof(path),"%s/%s.tokens.u32",dir,id);
        FILE *f=fopen(path,"rb");
        require(f&&fread(tokens,4u,prompt+generated,f)==prompt+generated&&fgetc(f)==EOF,
                "token dimensions");
        fclose(f);
        snprintf(path,sizeof(path),"%s/hc-%llu.coverage",dir,session);
        f=fopen(path,"rb");require(f!=NULL,"capture coverage file");
        for(uint32_t i=0;i<prompt+generated;i++)
            require(fgetc(f)==1,"hole in authoritative capture");
        require(fgetc(f)==EOF,"capture coverage length");fclose(f);
        snprintf(path,sizeof(path),"%s/hc-%llu.f32",dir,session);
        FILE *hc=fopen(path,"rb");
        struct stat st;
        require(hc&&!fstat(fileno(hc),&st)&&
            (uint64_t)st.st_size==(uint64_t)(prompt+generated)*FG_HYPER_WIDTH*4u,
            "complete authoritative HC required");
        check(fg_mtp_head_truncate(head,0u,&err),&err);
        for(uint32_t c=0u;c<prompt-1u+256u;c++){
            float hidden[FG_HYPER_WIDTH],emb[2560];
            require(fread(hidden,4u,FG_HYPER_WIDTH,hc)==FG_HYPER_WIDTH,"authoritative HC read");
            embedding(ef,tokens[c+1u],emb);
            check(fg_vk_tensor_write(input,0,hidden,sizeof(hidden),&err),&err);
            check(fg_vk_tensor_write(embed,0,emb,sizeof(emb),&err),&err);
            fg_vk_tensor *next=NULL,*contracted=NULL;
            check(fg_mtp_head_step_replay(head,input,embed,c,&next,&contracted,&err),&err);
            if(c<prompt-1u)continue;
            uint32_t position=c-(prompt-1u),prefix=1u,match[3],accepted[3];
            for(uint32_t j=0u;j<3u;j++){
                float score;uint32_t token;
                check(fg_mtp_vocab_argmax(vocab,contracted,&token,&score,&err),&err);
                require(isfinite(score),"nonfinite draft score");
                drafts[position*3u+j]=token;
                match[j]=token==tokens[c+2u+j];prefix&=match[j];accepted[j]=prefix;
                if(j+1u<3u){
                    embedding(ef,token,emb);
                    check(fg_vk_tensor_write(embed,0,emb,sizeof(emb),&err),&err);
                    check(fg_mtp_head_step_replay(head,next,embed,c+j+1u,
                        &next,&contracted,&err),&err);
                }
            }
            check(fg_mtp_head_truncate(head,c+1u,&err),&err);
            fprintf(out,"{\"case\":\"%s\",\"category\":\"%s\",\"position\":%u,"
                "\"draft\":[%u,%u,%u],\"target\":[%u,%u,%u],"
                "\"match\":[%u,%u,%u],\"prefix\":[%u,%u,%u]}\n",
                id,category,position,drafts[position*3u],drafts[position*3u+1u],
                drafts[position*3u+2u],tokens[c+2u],tokens[c+3u],tokens[c+4u],
                match[0],match[1],match[2],accepted[0],accepted[1],accepted[2]);
        }
        fclose(hc);free(tokens);
        snprintf(path,sizeof(path),"%s/%s.drafts.u32",dir,id);f=fopen(path,"wb");
        require(f&&fwrite(drafts,4u,256u*3u,f)==256u*3u,"proposal write");
        require(!fclose(f),"proposal close");
        fflush(out);cases++;
    }
    require(cases==40u,"exactly forty eligible captures required");
    require(!fclose(out)&&!fclose(index)&&!fclose(ef),"output close");
    fg_vk_tensor_destroy(input);fg_vk_tensor_destroy(embed);
    fg_mtp_vocab_close(vocab);fg_mtp_head_close(head);fg_vk_close(vk);free(manifest);
    return 0;
}
