#include "fg_manifest.h"
#include "fg_model.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void){
    char directory[128],path[160];snprintf(directory,sizeof(directory),"/tmp/fg-model-load-%ld",(long)getpid());snprintf(path,sizeof(path),"%s/rank-00.fgw",directory);if(mkdir(directory,0700)!=0){perror("mkdir");return 1;}uint8_t *bytes=aligned_alloc(FG_ALIGNMENT,FG_ALIGNMENT);if(!bytes)return 1;for(uint32_t i=0;i<FG_ALIGNMENT;i++)bytes[i]=(uint8_t)(i*29u+7u);FILE *stream=fopen(path,"wb");if(!stream||fwrite(bytes,1,FG_ALIGNMENT,stream)!=FG_ALIGNMENT||fclose(stream)!=0){perror("write rank artifact");free(bytes);return 1;}
    fg_manifest *manifest=malloc(sizeof(*manifest));if(!manifest){free(bytes);return 1;}fg_manifest_init(manifest);fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"probe.weight");record.bytes=FG_ALIGNMENT;record.dims=1;record.shape[0]=FG_ALIGNMENT;record.rank=0;record.layer=UINT16_MAX;record.expert=UINT16_MAX;record.kind=FG_TENSOR_COMMON;fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,bytes,FG_ALIGNMENT);fg_sha256_final(&hash,record.sha256);fg_error error={0};if(fg_manifest_add_tensor(manifest,&record,&error)!=FG_OK){fprintf(stderr,"manifest: %s\n",error.message);free(manifest);free(bytes);return 1;}
    fg_model *model=NULL;fg_status status=fg_model_open(&model,manifest,directory,0,&error);if(status==FG_ERR_UNAVAILABLE){fprintf(stderr,"SKIP direct Vulkan/io_uring model load: %s\n",error.message);unlink(path);rmdir(directory);free(manifest);free(bytes);return 77;}if(status!=FG_OK){fprintf(stderr,"model load: %s\n",error.message);unlink(path);rmdir(directory);free(manifest);free(bytes);return 1;}fg_vk_tensor *tensor=fg_model_tensor(model,"probe.weight");uint8_t check[64];int ok=tensor&&fg_vk_tensor_read(tensor,0,check,sizeof(check),&error)==FG_OK&&memcmp(check,bytes,sizeof(check))==0&&fg_model_weight_bytes(model)==FG_ALIGNMENT;fg_model_close(model);unlink(path);rmdir(directory);free(manifest);free(bytes);if(!ok){fprintf(stderr,"direct model load parity failed: %s\n",error.message);return 1;}puts("Flash Gordon fixed-buffer O_DIRECT to Vulkan arena: PASS");return 0;
}
