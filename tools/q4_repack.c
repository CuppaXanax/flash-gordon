#include "fg_pack.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int copy_tensor(FILE *src,uint64_t off,uint64_t bytes,FILE *dst,
                       uint8_t digest[32]){
    uint8_t *buffer=malloc(8u<<20u);
    if(!buffer||fseeko(src,(off_t)off,SEEK_SET)!=0){free(buffer);return 0;}
    fg_sha256 hash;fg_sha256_init(&hash);
    while(bytes){
        size_t n=bytes>(8u<<20u)?8u<<20u:(size_t)bytes;
        if(fread(buffer,1u,n,src)!=n||(dst&&fwrite(buffer,1u,n,dst)!=n)){
            free(buffer);return 0;
        }
        fg_sha256_update(&hash,buffer,n);bytes-=n;
    }
    fg_sha256_final(&hash,digest);free(buffer);return 1;
}

static int router_lossless(FILE *old,const fg_tensor_record *t,FILE *dense,
                           const fg_gguf_tensor *replacement){
    uint32_t f32[4096];uint16_t bf16[4096];uint64_t values=t->bytes/4u;
    if(fseeko(old,(off_t)t->offset,SEEK_SET)!=0||
       fseeko(dense,(off_t)replacement->offset,SEEK_SET)!=0)return 0;
    while(values){
        size_t n=values>4096u?4096u:(size_t)values;
        if(fread(f32,4u,n,old)!=n||fread(bf16,2u,n,dense)!=n)return 0;
        for(size_t i=0;i<n;i++)
            if((f32[i]&65535u)||(f32[i]>>16u)!=bf16[i])return 0;
        values-=n;
    }
    return 1;
}

int main(int argc,char **argv){
    if(argc<5||argc>7){
        fprintf(stderr,"usage: q4-repack OLD.fgm OLD_PACK DENSE.gguf NEW_DIR [RANK] [--keep-q8-head]\n");
        return 2;
    }
    int only=-1;
    bool keep_q8_head=false;
    for(int i=5;i<argc;i++){
        if(strcmp(argv[i],"--keep-q8-head")==0){keep_q8_head=true;continue;}
        char *end=NULL;long rank=strtol(argv[i],&end,10);
        if(!end||end==argv[i]||*end||rank<0||rank>=FG_RANK_COUNT||only>=0)return 2;
        only=(int)rank;
    }
    fg_error err={0};fg_manifest *m=malloc(sizeof(*m));fg_gguf g={0};
    const char *sources[]={argv[3]};int rc=1;FILE *dense=NULL;
    if(!m||fg_manifest_read(argv[1],m,&err)!=FG_OK||
       fg_q38_validate_packed_manifest(m,&err)!=FG_OK||
       fg_gguf_open(sources,1u,&g,&err)!=FG_OK)goto done;
    char *old_dir=realpath(argv[2],NULL);
    if(!old_dir)goto done;
    if(mkdir(argv[4],0755)!=0){free(old_dir);goto done;}
    dense=fopen(argv[3],"rb");
    if(!dense){free(old_dir);goto done;}
    uint64_t cursor[FG_RANK_COUNT]={0};
    uint32_t replaced=0u;
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        FILE *old=NULL,*out=NULL;char path[2048];
        bool full=only<0||(uint32_t)only==rank;
        bool emit=full;
        for(uint32_t i=0;!emit&&i<m->tensor_count;i++)
            if(m->tensors[i].rank==rank&&m->tensors[i].kind==FG_TENSOR_COMMON)emit=true;
        if(emit){
            snprintf(path,sizeof(path),"%s/rank-%02u.fgw",old_dir,rank);
            old=fopen(path,"rb");if(!old){free(old_dir);goto done;}
            snprintf(path,sizeof(path),"%s/rank-%02u.fgw",argv[4],rank);
            out=fopen(path,"wbx");if(!out){fclose(old);free(old_dir);goto done;}
        }
        for(uint32_t i=0;i<m->tensor_count;i++){
            fg_tensor_record *t=&m->tensors[i];
            if(t->rank!=rank||t->kind==FG_TENSOR_HOST_CACHE)continue;
            bool copy_record=full||t->kind==FG_TENSOR_COMMON;
            const fg_gguf_tensor *replacement=NULL;
            bool router=t->kind==FG_TENSOR_COMMON&&t->ggml_type==0u&&
                strstr(t->name,".ffn_gate_inp.weight")!=NULL;
            if(router){
                for(uint64_t j=0;j<g.tensor_count;j++)
                    if(strcmp(t->name,g.tensors[j].name)==0&&g.tensors[j].type==30u)
                        replacement=&g.tensors[j];
                if(replacement&&(replacement->dims!=2u||
                   memcmp(replacement->shape,t->shape,2u*sizeof(uint64_t)))){
                    fprintf(stderr,"invalid BF16 router shape: %s\n",t->name);
                    if(out)fclose(out);
                    if(old)fclose(old);
                    free(old_dir);goto done;
                }
                if(replacement&&copy_record&&!router_lossless(old,t,dense,replacement)){
                    fprintf(stderr,"router is not lossless vs current pack: %s\n",t->name);
                    fclose(out);fclose(old);free(old_dir);goto done;
                }
            }
            if(t->kind==FG_TENSOR_COMMON&&t->dims==2u&&t->ggml_type==8u&&
               strcmp(t->name,"token_embd.weight")!=0&&
               !(keep_q8_head&&strcmp(t->name,"output.weight")==0)){
                for(uint64_t j=0;j<g.tensor_count;j++)
                    if(strcmp(t->name,g.tensors[j].name)==0)replacement=&g.tensors[j];
                if(!replacement||replacement->dims!=2u||
                   memcmp(replacement->shape,t->shape,2u*sizeof(uint64_t))||
                   !(replacement->type==12u||replacement->type==13u||
                     replacement->type==14u||replacement->type==20u)){
                    fprintf(stderr,"missing/invalid replacement: %s\n",t->name);
                    if(out)fclose(out);
                    if(old)fclose(old);
                    free(old_dir);goto done;
                }
            }
            uint64_t offset=fg_align_up_u64(cursor[rank],FG_ALIGNMENT);
            if(out&&copy_record&&fseeko(out,(off_t)offset,SEEK_SET)!=0){
                fclose(out);fclose(old);free(old_dir);goto done;
            }
            uint8_t hash[32];
            if(replacement){
                if(!copy_tensor(dense,replacement->offset,replacement->bytes,
                                copy_record?out:NULL,hash)){
                    if(out)fclose(out);
                    if(old)fclose(old);
                    free(old_dir);goto done;
                }
                t->bytes=replacement->bytes;t->ggml_type=replacement->type;
                t->layout=FG_TENSOR_LAYOUT_GGML;memcpy(t->sha256,hash,32u);
                replaced++;
            }else if(out&&copy_record){
                if(!copy_tensor(old,t->offset,t->bytes,out,hash)||
                   memcmp(hash,t->sha256,32u)){
                    fprintf(stderr,"source hash mismatch: %s\n",t->name);
                    fclose(out);fclose(old);free(old_dir);goto done;
                }
            }
            t->offset=offset;cursor[rank]=fg_align_up_u64(offset+t->bytes,FG_ALIGNMENT);
        }
        if(out){
            int failed=fflush(out)!=0;
            if(!failed&&ftruncate(fileno(out),(off_t)cursor[rank])!=0)failed=1;
            if(!failed&&fsync(fileno(out))!=0)failed=1;
            if(fclose(out)!=0)failed=1;
            fclose(old);
            if(failed){free(old_dir);goto done;}
        }
        m->ranks[rank].persistent_bytes=cursor[rank];
    }
    /* Keep external tokenizer/ngram/host-cache artifacts byte-identical. */
    DIR *dir=opendir(old_dir);
    if(!dir){free(old_dir);goto done;}
    struct dirent *entry;
    while((entry=readdir(dir))){
        if(entry->d_name[0]=='.'||!strncmp(entry->d_name,"rank-",5u)||
           strstr(entry->d_name,".fgm"))continue;
        char source[2048],destination[2048];
        snprintf(source,sizeof(source),"%s/%s",old_dir,entry->d_name);
        snprintf(destination,sizeof(destination),"%s/%s",argv[4],entry->d_name);
        if(symlink(source,destination)!=0){
            closedir(dir);free(old_dir);goto done;
        }
    }
    closedir(dir);free(old_dir);
    uint8_t digest[32];fg_sha256 combined;fg_sha256_init(&combined);
    if(fg_sha256_file(argv[3],digest,&err)!=FG_OK)goto done;
    fg_sha256_update(&combined,m->source_sha256,32u);fg_sha256_update(&combined,digest,32u);
    fg_sha256_final(&combined,m->source_sha256);
    if(fg_q38_validate_packed_manifest(m,&err)!=FG_OK)goto done;
    char path[2048];snprintf(path,sizeof(path),"%s/manifest.fgm",argv[4]);
    if(fg_manifest_write(path,m,&err)!=FG_OK)goto done;
    printf("replaced=%u rank=%d manifest=%s\n",replaced,only,path);rc=0;
done:
    if(dense)fclose(dense);
    fg_gguf_close(&g);free(m);
    if(rc)fprintf(stderr,"repack failed: %s (%s); incomplete output is not deployable\n",
                  err.message,strerror(errno));
    return rc;
}
