#include "fg_manifest.h"
#include "fg_quant.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc,char **argv){
    if(argc!=5){
        fprintf(stderr,"usage: mtp-real-cook MANIFEST TENSOR RAW COOKED\n");
        return 2;
    }
    fg_manifest *manifest=malloc(sizeof(*manifest));
    fg_error error={0};
    if(!manifest)return 1;
    fg_status status=fg_manifest_read(argv[1],manifest,&error);
    const fg_tensor_record *record=NULL;
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)
        if(strcmp(manifest->tensors[i].name,argv[2])==0)record=&manifest->tensors[i];
    if(status!=FG_OK||!record||record->kind!=FG_TENSOR_ROUTED_EXPERT||
       record->ggml_type!=8u||record->layout!=FG_TENSOR_LAYOUT_Q8_0_EXPERT_COOKED||
       record->dims!=3u||record->shape[0]!=640u||record->shape[1]!=2560u||
       record->shape[2]!=FG_EXPERT_COUNT||record->bytes!=UINT64_C(891289600)){
        fprintf(stderr,"invalid real Q8 expert slab: %s\n",error.message);
        free(manifest);return 1;
    }
    uint64_t matrix=record->bytes/record->shape[2];
    uint8_t *raw=malloc((size_t)matrix),*cooked=malloc((size_t)matrix);
    FILE *input=fopen(argv[3],"rb");
    FILE *output=raw&&cooked&&input?fopen(argv[4],"wbx"):NULL;
    bool created=output!=NULL,ok=created;
    fg_sha256 hash;fg_sha256_init(&hash);
    for(uint32_t expert=0;ok&&expert<FG_EXPERT_COUNT;expert++){
        ok=fread(raw,1,(size_t)matrix,input)==(size_t)matrix&&
            fg_cook_q8_0_rows(raw,cooked,matrix,640u,2560u)&&
            fwrite(cooked,1,(size_t)matrix,output)==(size_t)matrix;
        if(ok)fg_sha256_update(&hash,cooked,(size_t)matrix);
    }
    if(ok)ok=fgetc(input)==EOF&&!ferror(input);
    if(input)fclose(input);
    if(output&&fclose(output)!=0)ok=false;
    uint8_t digest[32];fg_sha256_final(&hash,digest);
    ok=ok&&memcmp(digest,record->sha256,sizeof(digest))==0;
    if(ok){
        printf("{\"tensor\":\"%s\",\"bytes\":%llu,\"cooked_sha256\":\"",
               record->name,(unsigned long long)record->bytes);
        for(uint32_t i=0;i<32u;i++)printf("%02x",digest[i]);
        printf("\",\"manifest_tensor_sha_exact\":true}\n");
    }else{
        fprintf(stderr,"real-pack cooking/hash verification failed\n");
        if(created)unlink(argv[4]);
    }
    free(cooked);free(raw);free(manifest);
    return ok?0:1;
}
