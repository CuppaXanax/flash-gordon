#include "fg_manifest.h"
#include "fg_gguf.h"
#include "fg_quant.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool cook(const fg_tensor_record *t,const void *raw,void *out,
                 uint64_t bytes){
    uint32_t columns=(uint32_t)t->shape[0],rows=(uint32_t)t->shape[1];
    switch(t->layout){
    case FG_TENSOR_LAYOUT_GGML:memcpy(out,raw,(size_t)bytes);return true;
    case FG_TENSOR_LAYOUT_Q8_0_COOKED:
    case FG_TENSOR_LAYOUT_Q8_0_EXPERT_COOKED:
        return fg_cook_q8_0_rows(raw,out,bytes,columns,rows);
    case FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED:
        return fg_cook_k_quant_rows(raw,out,bytes,columns,rows,t->ggml_type);
    case FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED:
        return fg_cook_q5_1_rows(raw,out,bytes,columns,rows);
    case FG_TENSOR_LAYOUT_DENSE_SOA:
        return fg_cook_dense_soa(raw,out,bytes,columns,rows,t->ggml_type);
    case FG_TENSOR_LAYOUT_DENSE_COMPACT:
        return fg_cook_dense_compact(raw,out,bytes,columns,rows,t->ggml_type);
    default:return false;
    }
}

int main(int argc,char **argv){
    if(argc!=5){
        fprintf(stderr,"usage: mtp-real-place MANIFEST TENSOR RAW RANK_FILE\n");
        return 2;
    }
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_error error={0};
    if(!manifest)return 1;
    fg_status status=fg_manifest_read(argv[1],manifest,&error);
    const fg_tensor_record *record=NULL;
    for(uint32_t i=0;status==FG_OK&&i<manifest->tensor_count;i++)
        if(strcmp(manifest->tensors[i].name,argv[2])==0)record=&manifest->tensors[i];
    if(status!=FG_OK||!record||record->kind==FG_TENSOR_HOST_CACHE||
       !record->bytes||record->dims<1u||record->dims>3u){
        fprintf(stderr,"invalid tensor: %s\n",error.message);
        free(manifest);return 1;
    }
    uint64_t high=0;
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        const fg_tensor_record *t=&manifest->tensors[i];
        if(t->rank==record->rank&&t->kind!=FG_TENSOR_HOST_CACHE&&
           t->offset+t->bytes>high)high=t->offset+t->bytes;
    }
    high=(high+FG_ALIGNMENT-1u)&~(uint64_t)(FG_ALIGNMENT-1u);
    uint64_t count=record->dims==3u?record->shape[2]:1u;
    if(!count||record->bytes%count){free(manifest);return 1;}
    uint64_t matrix=record->bytes/count;
    uint64_t raw_bytes=0u;
    if(!fg_gguf_tensor_bytes(record->ggml_type,record->dims,record->shape,&raw_bytes)||
       raw_bytes%count||(record->layout==FG_TENSOR_LAYOUT_GGML&&raw_bytes!=record->bytes)){
        free(manifest);return 1;
    }
    uint64_t raw_matrix=raw_bytes/count;
    uint8_t *raw=malloc((size_t)raw_matrix),*cooked=malloc((size_t)matrix);
    FILE *input=fopen(argv[3],"rb"),*output=NULL;
    if(raw&&cooked&&input){
        output=fopen(argv[4],"r+b");
        if(!output)output=fopen(argv[4],"wbx");
    }
    struct stat st;
    bool ok=output&&fstat(fileno(output),&st)==0&&
        ((uint64_t)st.st_size==high||
         (!st.st_size&&ftruncate(fileno(output),(off_t)high)==0))&&
        fseeko(output,(off_t)record->offset,SEEK_SET)==0;
    fg_sha256 hash;fg_sha256_init(&hash);
    for(uint64_t i=0;ok&&i<count;i++){
        ok=fread(raw,1,(size_t)raw_matrix,input)==(size_t)raw_matrix&&
           cook(record,raw,cooked,matrix)&&
           fwrite(cooked,1,(size_t)matrix,output)==(size_t)matrix;
        if(ok)fg_sha256_update(&hash,cooked,(size_t)matrix);
    }
    if(ok)ok=fgetc(input)==EOF&&!ferror(input);
    if(input)fclose(input);
    if(output&&fclose(output)!=0)ok=false;
    uint8_t digest[32];fg_sha256_final(&hash,digest);
    ok=ok&&memcmp(digest,record->sha256,32u)==0;
    if(ok)printf("{\"tensor\":\"%s\",\"bytes\":%llu,"
        "\"manifest_tensor_sha_exact\":true}\n",
        record->name,(unsigned long long)record->bytes);
    else fprintf(stderr,"placement/hash verification failed for %s\n",record->name);
    free(cooked);free(raw);free(manifest);
    return ok?0:1;
}
