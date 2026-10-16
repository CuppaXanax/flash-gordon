#include "fg_mtp_embedding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: tools/mtp-embedding MANIFEST OUTPUT.fgme\n");return 2;}
    fg_manifest *m=malloc(sizeof(*m));fg_error err={0};
    if(!m)return 1;
    fg_status status=fg_manifest_read(argv[1],m,&err);
    char directory[1024];size_t length=strlen(argv[1]);
    if(length>=sizeof(directory)){free(m);return 2;}
    memcpy(directory,argv[1],length+1u);
    char *slash=strrchr(directory,'/');
    if(slash)*slash=0;else snprintf(directory,sizeof(directory),".");
    if(status==FG_OK)status=fg_manifest_validate_deployment(m,&err);
    if(status==FG_OK)status=fg_mtp_embedding_export(m,directory,argv[2],&err);
    if(status!=FG_OK)fprintf(stderr,"%s\n",err.message);
    else printf("MTP production-Q8 embedding sidecar: %s\n",argv[2]);
    free(m);return status==FG_OK?0:1;
}
