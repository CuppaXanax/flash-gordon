#include "fg_pack.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"
#include "fg_quant.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/fs.h>
#include <sys/ioctl.h>

static bool use_reflink;
static uint64_t reflink_bytes;

static bool owner_record(const fg_manifest *m,const fg_tensor_record *t,int owner){
    if(owner<0||t->rank==(uint32_t)owner)return true;
    bool helper=owner==0||owner==1||owner==2||
        (uint32_t)owner==m->layer_owner[FG_LAYER_COUNT-1u];
    /* Workers load owned layers; head/HC helpers additionally need rank 4 globals. */
    return helper&&t->rank==4u&&t->kind==FG_TENSOR_COMMON&&t->layer>=FG_LAYER_COUNT;
}

static int copy_tensor(FILE *src,uint64_t off,uint64_t bytes,FILE *dst,
                       uint8_t digest[32]){
    uint8_t *buffer=malloc(8u<<20u);
    if(!buffer||fseeko(src,(off_t)off,SEEK_SET)!=0){free(buffer);return 0;}
    bool cloned=false;
    if(dst&&use_reflink&&fflush(dst)==0){
        off_t offset=ftello(dst);
        struct file_clone_range range={.src_fd=fileno(src),.src_offset=off,
            .src_length=bytes,.dest_offset=(uint64_t)offset};
        if(offset>=0&&ioctl(fileno(dst),FICLONERANGE,&range)==0){
            cloned=true;reflink_bytes+=bytes;
            if(fseeko(dst,offset+(off_t)bytes,SEEK_SET)!=0){free(buffer);return 0;}
        }
    }
    fg_sha256 hash;fg_sha256_init(&hash);
    while(bytes){
        size_t n=bytes>(8u<<20u)?8u<<20u:(size_t)bytes;
        if(fread(buffer,1u,n,src)!=n||(dst&&!cloned&&fwrite(buffer,1u,n,dst)!=n)){
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

static int cook_tensor(FILE *source,const fg_gguf_tensor *t,FILE *out,bool compact,
                       uint64_t bytes,uint8_t hash[32]){
    if(t->bytes>SIZE_MAX||bytes>SIZE_MAX)return 0;
    void *raw=malloc((size_t)t->bytes),*cooked=malloc((size_t)bytes);
    int ok=raw&&cooked&&fseeko(source,(off_t)t->offset,SEEK_SET)==0&&
        fread(raw,1u,(size_t)t->bytes,source)==t->bytes&&
        (compact?fg_cook_dense_compact(raw,cooked,bytes,(uint32_t)t->shape[0],(uint32_t)t->shape[1],t->type):
         fg_cook_dense_soa(raw,cooked,bytes,(uint32_t)t->shape[0],(uint32_t)t->shape[1],t->type));
    if(ok){
        fg_sha256 digest;fg_sha256_init(&digest);fg_sha256_update(&digest,cooked,(size_t)bytes);
        fg_sha256_final(&digest,hash);
        ok=!out||fwrite(cooked,1u,(size_t)bytes,out)==bytes;
    }
    free(cooked);free(raw);return ok;
}

int main(int argc,char **argv){
    if(argc<5){
        fprintf(stderr,"usage: q4-repack OLD.fgm OLD_PACK DENSE.gguf NEW_DIR [RANK] [--keep-q8-head] [--cook-dense] [--reflink] [--plan]\n");
        return 2;
    }
    int only=-1;
    bool keep_q8_head=false;
    bool cook_dense=false,plan=false;
    for(int i=5;i<argc;i++){
        if(strcmp(argv[i],"--keep-q8-head")==0){keep_q8_head=true;continue;}
        if(strcmp(argv[i],"--cook-dense")==0){cook_dense=true;continue;}
        if(strcmp(argv[i],"--plan")==0){plan=true;continue;}
        if(strcmp(argv[i],"--reflink")==0){use_reflink=true;continue;}
        char *end=NULL;long rank=strtol(argv[i],&end,10);
        if(!end||end==argv[i]||*end||rank<0||rank>=FG_RANK_COUNT||only>=0)return 2;
        only=(int)rank;
    }
    fg_error err={0};fg_manifest *m=malloc(sizeof(*m));fg_gguf g={0};
    const char *sources[]={argv[3]};int rc=1;FILE *dense=NULL;
    if(!m||fg_manifest_read(argv[1],m,&err)!=FG_OK||
       fg_q38_validate_packed_manifest(m,&err)!=FG_OK||
       fg_gguf_open(sources,1u,&g,&err)!=FG_OK)goto done;
    char *old_dir=plan?strdup(argv[2]):realpath(argv[2],NULL);
    if(!old_dir)goto done;
    if(!plan&&mkdir(argv[4],0755)!=0){free(old_dir);goto done;}
    dense=fopen(argv[3],"rb");
    if(!dense){free(old_dir);goto done;}
    uint64_t cursor[FG_RANK_COUNT]={0};
    uint64_t common_bytes[FG_RANK_COUNT]={0},expert_bytes[FG_RANK_COUNT]={0};
    uint32_t replaced=0u;
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        FILE *old=NULL,*out=NULL;char path[2048];
        bool full=only<0||(uint32_t)only==rank;
        bool emit=full;
        for(uint32_t i=0;!emit&&i<m->tensor_count;i++)
            if(m->tensors[i].rank==rank&&owner_record(m,&m->tensors[i],only))emit=true;
        if(emit&&!plan){
            snprintf(path,sizeof(path),"%s/rank-%02u.fgw",old_dir,rank);
            old=fopen(path,"rb");if(!old){free(old_dir);goto done;}
            snprintf(path,sizeof(path),"%s/rank-%02u.fgw",argv[4],rank);
            out=fopen(path,"wbx");if(!out){fclose(old);free(old_dir);goto done;}
        }
        for(uint32_t i=0;i<m->tensor_count;i++){
            fg_tensor_record *t=&m->tensors[i];
            if(t->rank!=rank||t->kind==FG_TENSOR_HOST_CACHE)continue;
            bool copy_record=owner_record(m,t,only);
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
                if(replacement&&copy_record&&!plan&&!router_lossless(old,t,dense,replacement)){
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
                bool eligible=cook_dense&&(replacement->type==12u||replacement->type==13u||replacement->type==20u);
                bool compact=eligible&&fg_dense_prefer_compact((uint32_t)t->shape[0],(uint32_t)t->shape[1],replacement->type);
                uint64_t bytes=eligible?(compact?fg_dense_compact_bytes((uint32_t)t->shape[0],(uint32_t)t->shape[1],replacement->type):
                    fg_dense_soa_bytes((uint32_t)t->shape[0],(uint32_t)t->shape[1],replacement->type)):replacement->bytes;
                if(!bytes||(!plan&&!(eligible?cook_tensor(dense,replacement,copy_record?out:NULL,compact,bytes,hash):
                   copy_tensor(dense,replacement->offset,replacement->bytes,copy_record?out:NULL,hash)))){
                    if(out)fclose(out);
                    if(old)fclose(old);
                    free(old_dir);goto done;
                }
                t->bytes=bytes;t->ggml_type=replacement->type;
                t->layout=eligible?(compact?FG_TENSOR_LAYOUT_DENSE_COMPACT:FG_TENSOR_LAYOUT_DENSE_SOA):FG_TENSOR_LAYOUT_GGML;
                if(!plan)memcpy(t->sha256,hash,32u);
                replaced++;
            }else if(out&&copy_record){
                if(!copy_tensor(old,t->offset,t->bytes,out,hash)||
                   memcmp(hash,t->sha256,32u)){
                    fprintf(stderr,"source hash mismatch: %s\n",t->name);
                    fclose(out);fclose(old);free(old_dir);goto done;
                }
            }
            t->offset=offset;cursor[rank]=fg_align_up_u64(offset+t->bytes,FG_ALIGNMENT);
            if(t->kind==FG_TENSOR_COMMON)common_bytes[rank]+=fg_align_up_u64(t->bytes,FG_ALIGNMENT);
            if(t->kind==FG_TENSOR_ROUTED_EXPERT)expert_bytes[rank]+=fg_align_up_u64(t->bytes,FG_ALIGNMENT);
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
    if(plan){
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            uint32_t layers=0u;
            for(uint32_t l=0;l<FG_LAYER_COUNT;l++)layers+=m->layer_owner[l]==rank;
            uint64_t resident=0u;
            if(fg_q38_rank_residency_bytes(m,rank,&resident,&err)!=FG_OK){free(old_dir);goto done;}
            printf("PLAN_RANK rank=%u layers=%u resident_weight_bytes=%llu common_bytes=%llu expert_bytes=%llu endpoint=%s residency_budget_bytes=%llu\n",
                rank,layers,(unsigned long long)cursor[rank],(unsigned long long)common_bytes[rank],
                (unsigned long long)expert_bytes[rank],m->ranks[rank].endpoint,(unsigned long long)resident);
        }
        uint64_t foreign_global=0u,mtp=0u;
        for(uint32_t i=0;i<m->tensor_count;i++){
            const fg_tensor_record *t=&m->tensors[i];
            if(t->kind==FG_TENSOR_COMMON&&t->layer>=FG_LAYER_COUNT&&t->rank!=0u)
                foreign_global+=fg_align_up_u64(t->bytes,FG_ALIGNMENT);
            if(t->kind==FG_TENSOR_MTP&&t->rank==0u)mtp+=fg_align_up_u64(t->bytes,FG_ALIGNMENT);
        }
        printf("PLAN_COORDINATOR_RING rank=0 resident_weight_bytes=%llu foreign_global_bytes=%llu\n",
            (unsigned long long)(common_bytes[0]+expert_bytes[0]+foreign_global+mtp),
            (unsigned long long)foreign_global);
        printf("PLAN_ONLY: no output created; source hashes/router proof and runtime scratch/KV/driver budgets not verified.\n");
        free(old_dir);rc=0;goto done;
    }
    /* Keep external tokenizer/ngram/host-cache artifacts byte-identical. */
    DIR *dir=opendir(old_dir);
    if(!dir){free(old_dir);goto done;}
    struct dirent *entry;
    while((entry=readdir(dir))){
        if(entry->d_name[0]=='.'||!strncmp(entry->d_name,"rank-",5u)||
           strstr(entry->d_name,".fgm")||strstr(entry->d_name,".state"))continue;
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
    printf("replaced=%u rank=%d manifest=%s reflink_bytes=%llu\n",replaced,only,path,
           (unsigned long long)reflink_bytes);rc=0;
done:
    if(dense)fclose(dense);
    fg_gguf_close(&g);free(m);
    if(rc)fprintf(stderr,"repack failed: %s (%s); incomplete output is not deployable\n",
                  err.message,strerror(errno));
    return rc;
}
