#include "fg_gguf.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define GGUF_MAGIC UINT32_C(0x46554747)

static bool read_exact(FILE *f,void *p,size_t n){return fread(p,1,n,f)==n;}
static bool read_u32(FILE *f,uint32_t *v){return read_exact(f,v,4);}
static bool read_u64(FILE *f,uint64_t *v){return read_exact(f,v,8);}
static char *read_string(FILE *f){uint64_t n;if(!read_u64(f,&n)||n>(1u<<24))return NULL;char *s=malloc((size_t)n+1);if(!s)return NULL;if(!read_exact(f,s,(size_t)n)){free(s);return NULL;}s[n]=0;return s;}

static bool skip_bytes(FILE *f,uint64_t n){while(n){long step=n>0x3fffffffu?0x3fffffffu:(long)n;if(fseek(f,step,SEEK_CUR)!=0)return false;n-=(uint64_t)step;}return true;}
static bool skip_value(FILE *f,uint32_t type,uint32_t *alignment){
    static const uint8_t scalar[13]={1,1,2,2,4,4,4,1,0,0,8,8,8};
    if(type<13&&scalar[type])return skip_bytes(f,scalar[type]);
    if(type==8){char *s=read_string(f);if(!s)return false;free(s);return true;}
    if(type==9){uint32_t elem;uint64_t count;if(!read_u32(f,&elem)||!read_u64(f,&count)||count>(UINT64_MAX/16))return false;for(uint64_t i=0;i<count;i++)if(!skip_value(f,elem,alignment))return false;return true;}
    (void)alignment;return false;
}
static bool read_metadata_value(FILE *f,uint32_t type,const char *key,uint32_t *alignment){
    if(strcmp(key,"general.alignment")==0&&type==4){uint32_t v;if(!read_u32(f,&v))return false;if(v>=32&&v<=65536&&(v&(v-1u))==0)*alignment=v;return true;}
    return skip_value(f,type,alignment);
}

static bool type_layout(uint32_t type,uint32_t *block,uint32_t *bytes){
    switch(type){
        case 0:*block=1;*bytes=4;return true;case 1:*block=1;*bytes=2;return true;
        case 2:*block=32;*bytes=18;return true;case 3:*block=32;*bytes=20;return true;
        case 6:*block=32;*bytes=22;return true;case 7:*block=32;*bytes=24;return true;
        case 8:*block=32;*bytes=34;return true;case 9:*block=32;*bytes=40;return true;
        case 10:*block=256;*bytes=84;return true;case 11:*block=256;*bytes=110;return true;
        case 12:*block=256;*bytes=144;return true;case 13:*block=256;*bytes=176;return true;
        case 14:*block=256;*bytes=210;return true;case 15:*block=256;*bytes=292;return true;
        case 16:*block=256;*bytes=66;return true;case 17:*block=256;*bytes=74;return true;
        case 18:*block=256;*bytes=98;return true;case 19:*block=256;*bytes=50;return true;
        case 20:*block=32;*bytes=18;return true;case 21:*block=256;*bytes=110;return true;
        case 22:*block=256;*bytes=82;return true;case 23:*block=256;*bytes=136;return true;
        case 24:*block=1;*bytes=1;return true;case 25:*block=1;*bytes=2;return true;
        case 26:*block=1;*bytes=4;return true;case 27:*block=1;*bytes=8;return true;
        case 28:*block=1;*bytes=8;return true;case 29:*block=256;*bytes=56;return true;
        case 30:*block=1;*bytes=2;return true;case 34:*block=256;*bytes=54;return true;
        case 35:*block=256;*bytes=66;return true;case 36:*block=256;*bytes=108;return true;
        default:return false;
    }
}

static fg_status parse_shard(const char *path,uint32_t shard,fg_gguf *g,fg_error *err){
    FILE *f=fopen(path,"rb");if(!f){fg_error_set(err,FG_ERR_IO,"open GGUF %s: %s",path,strerror(errno));return FG_ERR_IO;}uint32_t magic,version;uint64_t nt,nkv;
    if(!read_u32(f,&magic)||!read_u32(f,&version)||!read_u64(f,&nt)||!read_u64(f,&nkv)||magic!=GGUF_MAGIC||version<2||version>3||nt>FG_MAX_TENSORS*32ull||nkv>(1u<<20)){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"invalid GGUF header in %s",path);return FG_ERR_FORMAT;}
    uint32_t alignment=32;
    for(uint64_t i=0;i<nkv;i++){char *key=read_string(f);uint32_t type;if(!key||!read_u32(f,&type)||!read_metadata_value(f,type,key,&alignment)){free(key);fclose(f);fg_error_set(err,FG_ERR_FORMAT,"invalid GGUF metadata in %s",path);return FG_ERR_FORMAT;}free(key);}
    uint64_t base=g->tensor_count,new_count=base+nt;fg_gguf_tensor *next=realloc(g->tensors,(size_t)new_count*sizeof(*next));if(!next){fclose(f);fg_error_set(err,FG_ERR_OOM,"allocate GGUF tensor directory");return FG_ERR_OOM;}g->tensors=next;memset(g->tensors+base,0,(size_t)nt*sizeof(*next));
    for(uint64_t i=0;i<nt;i++){
        fg_gguf_tensor *t=&g->tensors[base+i];t->name=read_string(f);if(!t->name||!read_u32(f,&t->dims)||t->dims<1||t->dims>FG_GGUF_MAX_DIMS){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"invalid tensor descriptor in %s",path);return FG_ERR_FORMAT;}
        uint64_t elements=1;for(uint32_t d=0;d<t->dims;d++){if(!read_u64(f,&t->shape[d])||t->shape[d]==0||elements>UINT64_MAX/t->shape[d]){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"invalid tensor shape %s",t->name);return FG_ERR_FORMAT;}elements*=t->shape[d];}
        uint64_t relative;uint32_t block,block_bytes;if(!read_u32(f,&t->type)||!read_u64(f,&relative)||!type_layout(t->type,&block,&block_bytes)||elements%block){fclose(f);fg_error_set(err,FG_ERR_FORMAT,"unsupported tensor type/shape for %s (type %u)",t->name,t->type);return FG_ERR_FORMAT;}t->bytes=(elements/block)*block_bytes;t->offset=relative;t->shard=shard;
    }
    long pos=ftell(f);if(pos<0){fclose(f);fg_error_set(err,FG_ERR_IO,"tell GGUF %s",path);return FG_ERR_IO;}uint64_t data=fg_align_up_u64((uint64_t)pos,alignment);for(uint64_t i=0;i<nt;i++)g->tensors[base+i].offset+=data;g->tensor_count=new_count;g->alignment=alignment;fclose(f);return FG_OK;
}

fg_status fg_gguf_open(const char *const *paths,uint32_t count,fg_gguf *g,fg_error *err){if(!paths||!count||!g){fg_error_set(err,FG_ERR_ARGUMENT,"GGUF source list is empty");return FG_ERR_ARGUMENT;}memset(g,0,sizeof(*g));g->paths=calloc(count,sizeof(char*));if(!g->paths){fg_error_set(err,FG_ERR_OOM,"allocate source list");return FG_ERR_OOM;}g->shard_count=count;for(uint32_t i=0;i<count;i++){g->paths[i]=strdup(paths[i]);if(!g->paths[i]){fg_gguf_close(g);fg_error_set(err,FG_ERR_OOM,"copy source path");return FG_ERR_OOM;}fg_status rc=parse_shard(paths[i],i,g,err);if(rc!=FG_OK){fg_gguf_close(g);return rc;}}return FG_OK;}
void fg_gguf_close(fg_gguf *g){if(!g)return;for(uint64_t i=0;i<g->tensor_count;i++)free(g->tensors[i].name);for(uint32_t i=0;i<g->shard_count;i++)free(g->paths[i]);free(g->tensors);free(g->paths);memset(g,0,sizeof(*g));}
int fg_gguf_tensor_layer(const char *name){const char *p=strstr(name,"blk.");if(!p)return -1;char *end;long v=strtol(p+4,&end,10);return end==p+4||v<0||v>=FG_LAYER_COUNT?-1:(int)v;}
int fg_gguf_tensor_expert(const char *name){const char *p=strstr(name,"expert.");if(!p)return -1;char *end;long v=strtol(p+7,&end,10);return end==p+7||v<0||v>=FG_EXPERT_COUNT?-1:(int)v;}
fg_tensor_kind fg_gguf_tensor_kind(const char *name){if(strstr(name,"per_layer_token_embd"))return FG_TENSOR_NGRAM;if(strstr(name,"_exps")||strstr(name,"experts"))return FG_TENSOR_ROUTED_EXPERT;if(strstr(name,"vision")||strstr(name,"visual"))return FG_TENSOR_VISION;if(strstr(name,"mtp")||strstr(name,"draft"))return FG_TENSOR_MTP;return FG_TENSOR_COMMON;}
void fg_gguf_print_schema(const fg_gguf *g){if(!g)return;for(uint64_t i=0;i<g->tensor_count;i++){const fg_gguf_tensor *t=&g->tensors[i];printf("%s\ttype=%u\tshape=",t->name,t->type);for(uint32_t d=0;d<t->dims;d++)printf("%s%" PRIu64,d?"x":"",t->shape[d]);printf("\tbytes=%" PRIu64 "\tshard=%u\toffset=%" PRIu64 "\n",t->bytes,t->shard,t->offset);}}
