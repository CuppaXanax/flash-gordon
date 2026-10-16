#include "fg_mtp_embedding.h"
#include "fg_quant.h"
#include "fg_sha256.h"
#include "fg_vk.h"
#include "fg_protocol.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static unsigned failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

int main(int argc,char **argv){
    if(argc!=2&&(argc!=3||strcmp(argv[2],"--gpu")))return 2;
    CHECK(mkdir(argv[1],0755)==0||errno==EEXIST);
    char source[1200],output[1200];
    snprintf(source,sizeof(source),"%s/rank-00.fgw",argv[1]);
    snprintf(output,sizeof(output),"%s/embedding.fgme",argv[1]);
    fg_manifest *m=calloc(1,sizeof(*m));CHECK(m);
    if(!m)return 1;
    m->tensor_count=1u;fg_tensor_record *t=&m->tensors[0];
    snprintf(t->name,sizeof(t->name),"token_embd.weight");
    t->ggml_type=8u;t->dims=2u;t->shape[0]=FG_HIDDEN_SIZE;t->shape[1]=16u;t->offset=128u;
    float input[16u*FG_HIDDEN_SIZE],expected[FG_HIDDEN_SIZE],actual[FG_HIDDEN_SIZE];
    uint8_t raw[16u*FG_HIDDEN_SIZE/32u*34u],cooked[65536],padding[128]={0};
    for(uint32_t i=0;i<16u*FG_HIDDEN_SIZE;i++)input[i]=(float)((int)(i%251u)-125)*0.003f;
    fg_quantize_q8_0(input,raw,16u*FG_HIDDEN_SIZE);
    fg_error err={0};
    fg_vk_context *vk=NULL;fg_vk_tensor *weights=NULL,*ids=NULL,*gpu_output=NULL;
    if(argc==3){
        uint32_t tokens[16];for(uint32_t i=0;i<16u;i++)tokens[i]=i;
        CHECK(fg_vk_open(&vk,&err)==FG_OK);
        if(!vk){fprintf(stderr,"%s\n",err.message);free(m);return 1;}
        CHECK(fg_vk_tensor_create(vk,sizeof(raw),&weights,&err)==FG_OK);
        CHECK(fg_vk_tensor_create(vk,sizeof(tokens),&ids,&err)==FG_OK);
        CHECK(fg_vk_tensor_create(vk,16u*FG_HYPER_WIDTH*4u,&gpu_output,&err)==FG_OK);
        CHECK(fg_vk_tensor_write(weights,0u,raw,sizeof(raw),&err)==FG_OK);
        CHECK(fg_vk_tensor_write(ids,0u,tokens,sizeof(tokens),&err)==FG_OK);
        CHECK(fg_vk_embedding_q8_0_batch(vk,gpu_output,weights,ids,16u,
            FG_HIDDEN_SIZE,16u,FG_HYPER_WIDTH/FG_HIDDEN_SIZE,&err)==FG_OK);
    }
    for(uint32_t layout=0;layout<=1u;layout++){
        t->layout=(uint8_t)layout;
        t->bytes=layout?fg_q8_0_cooked_tile_bytes(FG_HIDDEN_SIZE):sizeof(raw);
        const uint8_t *packed=raw;
        if(layout){
            CHECK(t->bytes<=sizeof(cooked));
            CHECK(fg_cook_q8_0_rows(raw,cooked,sizeof(cooked),FG_HIDDEN_SIZE,16u));
            packed=cooked;
        }
        fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,packed,(size_t)t->bytes);
        fg_sha256_final(&hash,t->sha256);
        FILE *f=fopen(source,"wb");CHECK(f);
        if(!f){free(m);return 1;}
        CHECK(fwrite(padding,1,sizeof(padding),f)==sizeof(padding));
        CHECK(fwrite(packed,1,(size_t)t->bytes,f)==t->bytes);CHECK(fclose(f)==0);
        CHECK(fg_mtp_embedding_export(m,argv[1],output,&err)==FG_OK);
        CHECK(fg_mtp_embedding_export(m,argv[1],output,&err)!=FG_OK);
        fg_mtp_embedding *e=NULL;
        CHECK(fg_mtp_embedding_open(&e,m,output,&err)==FG_OK);
        for(uint32_t row=0;row<16u;row++){
            fg_dequantize_q8_0(raw+(uint64_t)row*FG_HIDDEN_SIZE/32u*34u,expected,FG_HIDDEN_SIZE);
            CHECK(fg_mtp_embedding_row(e,row,actual,&err)==FG_OK);
            CHECK(!memcmp(expected,actual,sizeof(actual)));
            if(vk){
                float gpu[FG_HYPER_WIDTH];
                CHECK(fg_vk_tensor_read(gpu_output,(uint64_t)row*FG_HYPER_WIDTH*4u,
                    gpu,sizeof(gpu),&err)==FG_OK);
                for(uint32_t copy=0;copy<FG_HYPER_WIDTH/FG_HIDDEN_SIZE;copy++)
                    CHECK(!memcmp(actual,gpu+(uint64_t)copy*FG_HIDDEN_SIZE,sizeof(actual)));
            }
        }
        CHECK(fg_mtp_embedding_row(e,16u,actual,&err)!=FG_OK);
        fg_mtp_embedding_close(e);e=NULL;
        t->sha256[0]^=1u;
        CHECK(fg_mtp_embedding_open(&e,m,output,&err)!=FG_OK&&e==NULL);
        t->sha256[0]^=1u;
        f=fopen(output,"r+b");CHECK(f);
        if(!f){free(m);return 1;}
        CHECK(fseek(f,128,SEEK_SET)==0);int byte=fgetc(f);
        CHECK(byte!=EOF&&fseek(f,128,SEEK_SET)==0&&fputc(byte^1,f)!=EOF);
        CHECK(fclose(f)==0);
        CHECK(fg_mtp_embedding_open(&e,m,output,&err)!=FG_OK&&e==NULL);
        f=fopen(output,"r+b");CHECK(f);
        if(!f){free(m);return 1;}
        uint8_t changed[sizeof(raw)],digest[32];
        CHECK(fseek(f,128,SEEK_SET)==0&&fread(changed,1,sizeof(changed),f)==sizeof(changed));
        fg_sha256_init(&hash);fg_sha256_update(&hash,changed,sizeof(changed));
        fg_sha256_final(&hash,digest);
        CHECK(fseek(f,56,SEEK_SET)==0&&fwrite(digest,1,sizeof(digest),f)==sizeof(digest));
        CHECK(fclose(f)==0);
        CHECK(fg_mtp_embedding_open(&e,m,output,&err)!=FG_OK&&e==NULL);
        CHECK(unlink(output)==0);CHECK(unlink(source)==0);
    }
    CHECK(rmdir(argv[1])==0);free(m);
    fg_vk_tensor_destroy(gpu_output);fg_vk_tensor_destroy(ids);
    fg_vk_tensor_destroy(weights);fg_vk_close(vk);
    if(!failures)puts("MTP exact raw/cooked Q8 sidecar, source binding and corruption rejection: PASS");
    return failures?1:0;
}
