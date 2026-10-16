#include "fg_mtp_head.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static fg_error error;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"MTP_VOCAB_FAIL line=%d %s: %s\n",__LINE__,#x,error.message);return 1;}}while(0)
static void le32(uint8_t *p,uint32_t n){
    for(uint32_t i=0;i<4u;i++)p[i]=(uint8_t)(n>>(8u*i));
}
int main(void){
    char path[128];snprintf(path,sizeof(path),".build/mtp-vocab-%ld.fgdv",(long)getpid());
    int fd=open(path,O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC,0600);CHECK(fd>=0);
    uint32_t ids[]={1u,17u,500u};
    uint8_t header[56]={0};memcpy(header,"FGDVOC1\0",8u);le32(header+8u,3u);
    for(uint32_t i=0;i<3u;i++)le32(header+44u+4u*i,ids[i]);
    const uint32_t row_bytes=FG_HIDDEN_SIZE/32u*34u;
    const uint64_t source_bytes=(uint64_t)row_bytes*FG_Q38_VOCAB_SIZE;
    CHECK(!ftruncate(fd,(off_t)(sizeof(header)+source_bytes)));
    uint8_t raw[16u*FG_HIDDEN_SIZE/32u*34u],cooked[sizeof(raw)];
    for(uint32_t i=0;i<3u;i++){
        memset(raw,0,row_bytes);
        uint16_t scale=fg_f32_to_f16(i==1u?.03125f:i==0u?.015625f:.0078125f);
        for(uint32_t b=0;b<FG_HIDDEN_SIZE/32u;b++){
            memcpy(raw+b*34u,&scale,2u);memset(raw+b*34u+2u,1,32u);
        }
        CHECK(pwrite(fd,raw,row_bytes,(off_t)(sizeof(header)+(uint64_t)ids[i]*row_bytes))==row_bytes);
    }
    fg_tensor_record record={.ggml_type=8u,.layout=FG_TENSOR_LAYOUT_Q8_0_COOKED};
    record.shape[0]=FG_HIDDEN_SIZE;record.shape[1]=FG_Q38_VOCAB_SIZE;
    record.bytes=fg_q8_0_cooked_matrix_bytes(FG_HIDDEN_SIZE,FG_Q38_VOCAB_SIZE);
    fg_sha256 hash;fg_sha256_init(&hash);
    for(uint32_t row=0;row<FG_Q38_VOCAB_SIZE;row+=16u){
        CHECK(pread(fd,raw,sizeof(raw),(off_t)(sizeof(header)+(uint64_t)row*row_bytes))==(ssize_t)sizeof(raw));
        CHECK(fg_cook_q8_0_rows(raw,cooked,sizeof(cooked),FG_HIDDEN_SIZE,16u));
        fg_sha256_update(&hash,cooked,sizeof(cooked));
    }
    fg_sha256_final(&hash,record.sha256);memcpy(header+12u,record.sha256,32u);
    CHECK(pwrite(fd,header,sizeof(header),0)==(ssize_t)sizeof(header));
    fg_vk_context *vk=NULL;CHECK(fg_vk_open(&vk,&error)==FG_OK);
    fg_vk_memory_stats before,after;fg_vk_get_memory_stats(vk,&before);
    fg_mtp_vocab *vocab=NULL;
    CHECK(fg_mtp_vocab_open_draft(&vocab,vk,path,&record,&error)==FG_OK);
    fg_vk_get_memory_stats(vk,&after);
    CHECK(after.requested_live_bytes-before.requested_live_bytes<1024u*1024u);
    fg_vk_tensor *hidden=NULL;CHECK(fg_vk_tensor_create(vk,FG_HIDDEN_SIZE*4u,&hidden,&error)==FG_OK);
    float input[FG_HIDDEN_SIZE];for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++)input[i]=1.0f;
    CHECK(fg_vk_tensor_write(hidden,0u,input,sizeof(input),&error)==FG_OK);
    for(uint32_t replay=0;replay<3u;replay++){
        uint32_t token=0;float score=0;
        CHECK(fg_mtp_vocab_argmax(vocab,hidden,&token,&score,&error)==FG_OK);
        CHECK(token==17u&&fabsf(score-80.0f)<1e-5f);
    }
    fg_mtp_vocab_close(vocab);vocab=NULL;
    header[12u]^=1u;CHECK(pwrite(fd,header,sizeof(header),0)==(ssize_t)sizeof(header));
    CHECK(fg_mtp_vocab_open_draft(&vocab,vk,path,&record,&error)==FG_ERR_FORMAT&&!vocab);
    header[12u]^=1u;le32(header+48u,1u);
    CHECK(pwrite(fd,header,sizeof(header),0)==(ssize_t)sizeof(header));
    CHECK(fg_mtp_vocab_open_draft(&vocab,vk,path,&record,&error)==FG_ERR_FORMAT&&!vocab);
    le32(header+48u,17u);CHECK(pwrite(fd,header,sizeof(header),0)==(ssize_t)sizeof(header));
    uint8_t altered=1u;
    CHECK(pwrite(fd,&altered,1u,(off_t)(sizeof(header)+600u*row_bytes))==1);
    CHECK(fg_mtp_vocab_open_draft(&vocab,vk,path,&record,&error)==FG_ERR_MISMATCH&&!vocab);
    altered=0u;CHECK(pwrite(fd,&altered,1u,(off_t)(sizeof(header)+600u*row_bytes))==1);
    CHECK(!ftruncate(fd,(off_t)(sizeof(header)+source_bytes-1u)));
    CHECK(fg_mtp_vocab_open_draft(&vocab,vk,path,&record,&error)==FG_ERR_FORMAT&&!vocab);
    fg_vk_tensor_destroy(hidden);fg_vk_close(vk);close(fd);CHECK(!unlink(path));
    printf("{\"kind\":\"draft_vocab_test\",\"rows\":3,\"mapped_argmax\":17,"
        "\"replay\":true,\"wrong_parent_rejected\":true,\"duplicate_ids_rejected\":true,"
        "\"unselected_source_corruption_rejected\":true,\"truncation_rejected\":true,"
        "\"resident_bytes\":%llu}\n",
        (unsigned long long)(after.requested_live_bytes-before.requested_live_bytes));
    return 0;
}
