#include "fg_protocol.h"
#include "fg_manifest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void need(int ok,const char *what){if(!ok){fprintf(stderr,"%s\n",what);exit(1);}}
static void read_exact(FILE *f,void *out,size_t n){need(fread(out,1,n,f)==n,"short input");}
int main(int argc,char **argv){
    if(argc!=6){fprintf(stderr,"usage: fp16-wire-oracle MANIFEST PACK FIRST INPUT_PREFIX MODE\n");return 2;}
    fg_manifest *m=malloc(sizeof(*m));fg_error error={0};need(m!=NULL,"manifest allocation");
    need(fg_manifest_read(argv[1],m,&error)==FG_OK,error.message);
    uint32_t first=(uint32_t)strtoul(argv[3],NULL,10);
    need(first<FG_LAYER_COUNT&&(!strcmp(argv[5],"1")||!strcmp(argv[5],"128")),"first/mode");
    char path[2048];snprintf(path,sizeof(path),"%s/cases.bin",argv[2]);
    FILE *cases=fopen(path,"rb");need(cases!=NULL,"cases open");
    snprintf(path,sizeof(path),"%s.candidate",argv[4]);
    FILE *input=fopen(path,"rb");need(input!=NULL,"input open");
    FILE *ngram=NULL;
    if(!first){snprintf(path,sizeof(path),"%s/ngram.f32",argv[2]);ngram=fopen(path,"rb");need(ngram!=NULL,"ngram open");}
    char magic[8];uint32_t count;read_exact(cases,magic,8);read_exact(cases,&count,4);
    need(!memcmp(magic,"FGBCASE1",8)&&count==80,"cases identity");
    fg_layer_work work={.layer=(uint8_t)first,.source_rank=first?m->layer_owner[first-1u]:0u,
        .destination_rank=m->layer_owner[first],.flags=first?0:FG_LAYER_WORK_HAS_NGRAM},a,b;
    uint8_t fast[FG_LAYER_WORK_MAX_BYTES],portable[FG_LAYER_WORK_MAX_BYTES];uint64_t rows=0;
    setenv("FG_BF16_HOPS","0",1);setenv("FG_FP16_HOPS",argv[5],1);unsetenv("FG_FP16_HOP_MASK");
    for(uint32_t c=0;c<count;c++){
        uint32_t nt;read_exact(cases,&nt,4);need(nt>=4&&nt<=512,"token count");
        need(fseek(cases,(long)nt*4u,SEEK_CUR)==0,"token skip");
        for(uint32_t p=0;p<nt;p++){
            read_exact(input,work.hyper,sizeof(work.hyper));
            if(ngram)read_exact(ngram,work.ngram_embedding,sizeof(work.ngram_embedding));
            if(p+4u<nt)continue;
            work.token_index=p;for(uint32_t axis=0;axis<3;axis++)work.position[axis]=p;
            uint32_t na=0,nb=0;setenv("FG_FP16_SCALAR","0",1);
            need(fg_decode_layer_work_encode(fast,sizeof(fast),&na,6,&work,&error)==FG_OK,error.message);
            need(fg_decode_layer_work_decode(&a,6,fast,na,&error)==FG_OK,error.message);
            setenv("FG_FP16_SCALAR","1",1);
            need(fg_decode_layer_work_encode(portable,sizeof(portable),&nb,6,&work,&error)==FG_OK,error.message);
            need(fg_decode_layer_work_decode(&b,6,portable,nb,&error)==FG_OK,error.message);
            need(na==nb&&!memcmp(fast,portable,na)&&!memcmp(&a,&b,sizeof(a)),"hardware/portable wire mismatch");
            rows++;
        }
    }
    need(rows==320&&fgetc(cases)==EOF&&fgetc(input)==EOF,"frame count/trailing bytes");
    fclose(cases);fclose(input);if(ngram){need(fgetc(ngram)==EOF,"ngram trailing");fclose(ngram);}
    printf("{\"kind\":\"fp16_wire_oracle\",\"first\":%u,\"destination\":%u,"
        "\"mode\":%s,\"rows\":%llu,\"wire_and_decode_bit_exact\":true}\n",
        first,m->layer_owner[first],argv[5],(unsigned long long)rows);
    free(m);return 0;
}
