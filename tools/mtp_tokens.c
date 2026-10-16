#include "fg_tokenizer.h"
#include "fg_ngram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static fg_error error;
static void check(fg_status s){
    if(s!=FG_OK){fprintf(stderr,"mtp tokens: %s\n",error.message);exit(1);}
}
static void need(bool ok){
    if(!ok){fprintf(stderr,"invalid diagnostic token input\n");exit(1);}
}
int main(int argc,char **argv){
    if(argc<3||argc>4||(argc==4&&strcmp(argv[3],"--vocab"))){
        fprintf(stderr,"usage: mtp-tokens MANIFEST PACK [--vocab] < INPUT\n");return 2;
    }
    fg_manifest *m=malloc(sizeof(*m));need(m!=NULL);
    check(fg_manifest_read(argv[1],m,&error));
    fg_tokenizer *t=NULL;check(fg_tokenizer_open(&t,argv[2],m,&error));
    if(argc==4){
        for(uint32_t id=0;id<fg_tokenizer_vocab_size(t);id++){
            char text[4096];size_t bytes=0;
            check(fg_tokenizer_decode_token(t,id,text,sizeof(text),&bytes,&error));
            uint32_t length=(uint32_t)bytes;
            need(fwrite(&id,4,1,stdout)==1&&fwrite(&length,4,1,stdout)==1&&
                 fwrite(text,1,bytes,stdout)==bytes);
        }
        fg_tokenizer_close(t);free(m);return 0;
    }
    uint32_t name_bytes;
    for(;;){
        size_t got=fread(&name_bytes,1,4,stdin);
        if(!got)break;
        need(got==4);
        need(name_bytes&&name_bytes<128);char name[128];uint32_t bytes;
        need(fread(name,1,name_bytes,stdin)==name_bytes);name[name_bytes]=0;
        need(strspn(name,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")==name_bytes);
        need(fread(&bytes,1,4,stdin)==4&&bytes&&(bytes<(1u<<20u)));
        char *text=malloc((size_t)bytes+1u);need(text!=NULL);
        need(fread(text,1,bytes,stdin)==bytes);text[bytes]=0;
        fg_tokens tokens={0};check(fg_tokenizer_encode(t,text,false,&tokens,&error));free(text);
        printf("{\"name\":\"%s\",\"eos\":%u,\"tokens\":[",name,fg_tokenizer_eos(t));
        for(size_t i=0;i<tokens.count;i++)printf("%s%u",i?",":"",tokens.data[i]);
        printf("],\"ngram_rows\":[");
        for(size_t i=0;i<tokens.count;i++){
            uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];
            int32_t *history=malloc((i+1u)*4u);
            need(history!=NULL);
            for(size_t j=0;j<=i;j++)history[j]=(int32_t)tokens.data[j];
            check(fg_q38_ngram_lookup(history,i+1u,rows,addresses,&error));free(history);
            printf("%s[",i?",":"");
            for(uint32_t h=0;h<FG_NGRAM_HEAD_COUNT;h++)
                printf("%s%llu",h?",":"",(unsigned long long)rows[h]);
            printf("]");
        }
        printf("]}\n");fflush(stdout);fg_tokens_free(&tokens);
    }
    need(feof(stdin));fg_tokenizer_close(t);free(m);return 0;
}
