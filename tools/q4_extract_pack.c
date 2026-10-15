#include "fg_pack.h"
#include "fg_quant.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int put(FILE *f,const void *p,size_t n){return fwrite(p,1u,n,f)==n;}
static int u32(FILE *f,uint32_t n){return put(f,&n,sizeof(n));}
static int u64(FILE *f,uint64_t n){return put(f,&n,sizeof(n));}
static uint64_t align32(uint64_t n){return (n+31u)&~UINT64_C(31);}
static int selected(const fg_tensor_record *t){
    return t->kind==FG_TENSOR_COMMON&&t->ggml_type==8u&&t->dims==2u&&
        strcmp(t->name,"output.weight")&&strcmp(t->name,"token_embd.weight");
}

int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: q4-extract-pack MANIFEST PACK NEW.gguf\n");return 2;}
    const uint32_t endian=1u;
    if(*(const uint8_t *)&endian!=1u)return 2;
    fg_manifest *m=malloc(sizeof(*m));fg_error err={0};
    FILE *out=NULL,*source[FG_RANK_COUNT]={0};int rc=1;
    if(!m||fg_manifest_read(argv[1],m,&err)!=FG_OK)goto done;
    uint64_t count=0u,header=24u;
    for(uint32_t i=0;i<m->tensor_count;i++)if(selected(&m->tensors[i])){
        const fg_tensor_record *t=&m->tensors[i];
        if(t->shape[0]>UINT32_MAX||t->shape[1]>UINT32_MAX||t->shape[0]%32u)goto done;
        count++;header+=8u+strlen(t->name)+4u+16u+4u+8u;
    }
    if(!count||(out=fopen(argv[3],"wbx"))==NULL)goto done;
    if(!u32(out,UINT32_C(0x46554747))||!u32(out,3u)||!u64(out,count)||!u64(out,0u))goto done;
    uint64_t cursor=0u;
    for(uint32_t i=0;i<m->tensor_count;i++)if(selected(&m->tensors[i])){
        const fg_tensor_record *t=&m->tensors[i];size_t n=strlen(t->name);
        if(!u64(out,n)||!put(out,t->name,n)||!u32(out,2u)||
           !u64(out,t->shape[0])||!u64(out,t->shape[1])||!u32(out,8u)||!u64(out,cursor))goto done;
        cursor=align32(cursor+t->shape[0]/32u*34u*t->shape[1]);
    }
    uint64_t data_start=align32(header);cursor=0u;
    for(uint32_t i=0;i<m->tensor_count;i++)if(selected(&m->tensors[i])){
        const fg_tensor_record *t=&m->tensors[i];uint32_t rank=t->rank;
        if(rank>=FG_RANK_COUNT)goto done;
        if(!source[rank]){
            char path[2048];snprintf(path,sizeof(path),"%s/rank-%02u.fgw",argv[2],rank);
            if((source[rank]=fopen(path,"rb"))==NULL)goto done;
        }
        uint64_t bytes=t->shape[0]/32u*34u*t->shape[1];
        if(t->bytes>SIZE_MAX||bytes>SIZE_MAX)goto done;
        void *saved=malloc((size_t)t->bytes),*raw=malloc((size_t)bytes);
        if(!saved||!raw){free(saved);free(raw);goto done;}
        int ok=fseeko(source[rank],(off_t)t->offset,SEEK_SET)==0&&
            fread(saved,1u,(size_t)t->bytes,source[rank])==t->bytes;
        uint8_t hash[32];fg_sha256 digest;
        fg_sha256_init(&digest);fg_sha256_update(&digest,saved,ok?(size_t)t->bytes:0u);fg_sha256_final(&digest,hash);
        ok=ok&&!memcmp(hash,t->sha256,32u);
        if(ok&&t->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED)
            ok=t->bytes==fg_q8_0_cooked_matrix_bytes((uint32_t)t->shape[0],(uint32_t)t->shape[1])&&
                fg_uncook_q8_0_rows(saved,raw,bytes,(uint32_t)t->shape[0],(uint32_t)t->shape[1]);
        else if(ok&&t->layout==FG_TENSOR_LAYOUT_GGML&&t->bytes==bytes)memcpy(raw,saved,(size_t)bytes);
        else ok=0;
        if(ok)ok=fseeko(out,(off_t)(data_start+cursor),SEEK_SET)==0&&put(out,raw,(size_t)bytes);
        free(raw);free(saved);
        if(!ok){fprintf(stderr,"extract failed: %s\n",t->name);goto done;}
        cursor=align32(cursor+bytes);
        printf("EXTRACT name=%s source_layout=%u bytes=%llu\n",t->name,t->layout,(unsigned long long)bytes);
    }
    if(fflush(out)||fsync(fileno(out)))goto done;
    printf("EXTRACT_PASS tensors=%llu\n",(unsigned long long)count);rc=0;
done:
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)if(source[rank])fclose(source[rank]);
    if(out){fclose(out);if(rc)unlink(argv[3]);}
    free(m);return rc;
}
