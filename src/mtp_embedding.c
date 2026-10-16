#include "fg_mtp_embedding.h"
#include "fg_quant.h"
#include "fg_sha256.h"
#include "fg_q38_schema.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define EMBEDDING_HEADER_BYTES 128u
#define EMBEDDING_ROW_BYTES (FG_HIDDEN_SIZE/32u*34u)

struct fg_mtp_embedding {int fd;uint32_t rows;};

static void put32(uint8_t *p,uint32_t x){
    p[0]=(uint8_t)(x>>24u);p[1]=(uint8_t)(x>>16u);
    p[2]=(uint8_t)(x>>8u);p[3]=(uint8_t)x;
}
static uint32_t get32(const uint8_t *p){
    return (uint32_t)p[0]<<24u|(uint32_t)p[1]<<16u|(uint32_t)p[2]<<8u|p[3];
}
static const fg_tensor_record *embedding_record(const fg_manifest *m){
    for(uint32_t i=0;m&&i<m->tensor_count;i++){
        const fg_tensor_record *t=&m->tensors[i];
        if(!strcmp(t->name,"token_embd.weight")&&t->ggml_type==8u&&t->dims==2u&&
           t->rank<FG_RANK_COUNT&&
           t->shape[0]==FG_HIDDEN_SIZE&&t->shape[1]&&t->shape[1]<=FG_Q38_VOCAB_SIZE&&
           (t->layout==0u||t->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED))return t;
    }
    return NULL;
}
static fg_status fail(fg_error *err,const char *what){
    fg_error_set(err,FG_ERR_MISMATCH,"MTP production embedding %s",what);return FG_ERR_MISMATCH;
}
static bool read_at(int fd,void *p,size_t bytes,uint64_t offset){
    uint8_t *out=p;
    while(bytes){
        ssize_t n=pread(fd,out,bytes,(off_t)offset);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return false;
        out+=n;offset+=(uint64_t)n;bytes-=(size_t)n;
    }
    return true;
}
static bool write_all(int fd,const void *p,size_t bytes){
    const uint8_t *in=p;
    while(bytes){
        ssize_t n=write(fd,in,bytes);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return false;
        in+=n;bytes-=(size_t)n;
    }
    return true;
}
fg_status fg_mtp_embedding_export(const fg_manifest *m,const char *directory,
    const char *output,fg_error *err){
    const fg_tensor_record *t=embedding_record(m);
    if(!t||!directory||!output||t->shape[1]%16u)return fail(err,"export arguments or Q8 layout");
    char source[1200];
    if(snprintf(source,sizeof(source),"%s/rank-%02u.fgw",directory,t->rank)>=(int)sizeof(source))
        return fail(err,"source path length");
    int in=open(source,O_RDONLY|O_CLOEXEC);
    if(in<0)return fail(err,"source open");
    int out=open(output,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(out<0){close(in);return fail(err,"output exists or cannot be created");}
    uint8_t header[EMBEDDING_HEADER_BYTES]={0};
    uint8_t raw[16u*EMBEDDING_ROW_BYTES],cooked[65536];
    size_t tile=t->layout==0u?sizeof(raw):(size_t)fg_q8_0_cooked_tile_bytes(FG_HIDDEN_SIZE);
    uint64_t source_bytes=(uint64_t)tile*(t->shape[1]/16u);
    bool ok=tile<=sizeof(cooked)&&source_bytes==t->bytes&&write_all(out,header,sizeof(header));
    fg_sha256 packed_hash,raw_hash;fg_sha256_init(&packed_hash);fg_sha256_init(&raw_hash);
    for(uint32_t row=0;ok&&row<t->shape[1];row+=16u){
        ok=read_at(in,cooked,tile,t->offset+(uint64_t)(row/16u)*tile);
        if(!ok)break;
        fg_sha256_update(&packed_hash,cooked,tile);
        if(t->layout==0u)memcpy(raw,cooked,sizeof(raw));
        else ok=fg_uncook_q8_0_rows(cooked,raw,sizeof(raw),FG_HIDDEN_SIZE,16u);
        if(ok){fg_sha256_update(&raw_hash,raw,sizeof(raw));ok=write_all(out,raw,sizeof(raw));}
    }
    uint8_t source_sha[32],raw_sha[32];
    fg_sha256_final(&packed_hash,source_sha);fg_sha256_final(&raw_hash,raw_sha);
    ok=ok&&!memcmp(source_sha,t->sha256,32u);
    memcpy(header,"FGMEQ801",8u);put32(header+8u,1u);put32(header+12u,FG_HIDDEN_SIZE);
    put32(header+16u,(uint32_t)t->shape[1]);put32(header+20u,EMBEDDING_ROW_BYTES);
    memcpy(header+24u,t->sha256,32u);memcpy(header+56u,raw_sha,32u);
    put32(header+88u,t->layout);
    if(ok)ok=lseek(out,0,SEEK_SET)==0&&write_all(out,header,sizeof(header))&&fsync(out)==0;
    if(close(in))ok=false;
    if(close(out))ok=false;
    if(!ok){unlink(output);return fail(err,"export bytes or sealed source SHA256");}
    return FG_OK;
}
fg_status fg_mtp_embedding_open(fg_mtp_embedding **out,const fg_manifest *m,
    const char *path,fg_error *err){
    if(!out||!path)return fail(err,"open arguments");
    *out=NULL;const fg_tensor_record *t=embedding_record(m);
    if(!t)return fail(err,"sealed Q8 tensor is missing");
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    if(fd<0)return fail(err,"sidecar open");
    struct stat st;uint8_t header[EMBEDDING_HEADER_BYTES];
    uint64_t bytes=t->shape[1]*EMBEDDING_ROW_BYTES;
    size_t tile=t->layout==0u?16u*EMBEDDING_ROW_BYTES:
        (size_t)fg_q8_0_cooked_tile_bytes(FG_HIDDEN_SIZE);
    bool ok=t->shape[1]%16u==0u&&t->bytes==(t->shape[1]/16u)*tile&&
        fstat(fd,&st)==0&&st.st_size>=0&&
        (uint64_t)st.st_size==EMBEDDING_HEADER_BYTES+bytes&&read_at(fd,header,sizeof(header),0u);
    if(ok)ok=!memcmp(header,"FGMEQ801",8u)&&get32(header+8u)==1u&&
        get32(header+12u)==FG_HIDDEN_SIZE&&get32(header+16u)==t->shape[1]&&
        get32(header+20u)==EMBEDDING_ROW_BYTES&&!memcmp(header+24u,t->sha256,32u)&&
        get32(header+88u)==t->layout;
    for(uint32_t i=92u;ok&&i<EMBEDDING_HEADER_BYTES;i++)if(header[i])ok=false;
    fg_sha256 hash,packed_hash;fg_sha256_init(&hash);fg_sha256_init(&packed_hash);
    uint8_t buffer[16u*EMBEDDING_ROW_BYTES],cooked[65536];
    for(uint64_t offset=0u;ok&&offset<bytes;offset+=sizeof(buffer)){
        size_t count=sizeof(buffer);
        ok=read_at(fd,buffer,count,EMBEDDING_HEADER_BYTES+offset);
        if(ok){
            fg_sha256_update(&hash,buffer,count);
            if(t->layout==0u)fg_sha256_update(&packed_hash,buffer,count);
            else{
                ok=tile<=sizeof(cooked)&&fg_cook_q8_0_rows(buffer,cooked,
                    sizeof(cooked),FG_HIDDEN_SIZE,16u);
                if(ok)fg_sha256_update(&packed_hash,cooked,tile);
            }
        }
    }
    uint8_t digest[32],packed_digest[32];
    fg_sha256_final(&hash,digest);fg_sha256_final(&packed_hash,packed_digest);
    if(ok)ok=!memcmp(digest,header+56u,32u)&&!memcmp(packed_digest,t->sha256,32u);
    /* The integrity scan must not reserve a second resident embedding table. */
    (void)posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);
    if(!ok){close(fd);return fail(err,"sidecar binding, length or payload SHA256");}
    fg_mtp_embedding *e=calloc(1,sizeof(*e));
    if(!e){close(fd);fg_error_set(err,FG_ERR_OOM,"allocate MTP embedding reader");return FG_ERR_OOM;}
    e->fd=fd;e->rows=(uint32_t)t->shape[1];*out=e;return FG_OK;
}
fg_status fg_mtp_embedding_row(fg_mtp_embedding *e,uint32_t token,float *values,fg_error *err){
    uint8_t raw[EMBEDDING_ROW_BYTES];
    if(!e||!values||token>=e->rows||
       !read_at(e->fd,raw,sizeof(raw),EMBEDDING_HEADER_BYTES+(uint64_t)token*sizeof(raw)))
        return fail(err,"row lookup");
    fg_dequantize_q8_0(raw,values,FG_HIDDEN_SIZE);return FG_OK;
}
void fg_mtp_embedding_close(fg_mtp_embedding *e){if(e){close(e->fd);free(e);}}
