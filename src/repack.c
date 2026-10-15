#include "fg_pack.h"
#include "fg_ngram.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"
#include "fg_topology.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static fg_status repack_path(char out[1200],const char *dir,const char *name,fg_error *err){
    if(snprintf(out,1200,"%s/%s",dir,name)>=1200){
        fg_error_set(err,FG_ERR_LIMIT,"repack path is too long");return FG_ERR_LIMIT;
    }
    return FG_OK;
}

static fg_status copy_extent(FILE *source,uint64_t offset,FILE *dest,uint64_t bytes,
                             const uint8_t expected[32],uint8_t digest[32],fg_error *err){
    uint8_t buffer[65536];fg_sha256 hash;fg_sha256_init(&hash);
    if(fseeko(source,(off_t)offset,SEEK_SET)!=0)goto io_error;
    while(bytes){
        size_t chunk=bytes>sizeof(buffer)?sizeof(buffer):(size_t)bytes;
        if(fread(buffer,1,chunk,source)!=chunk)goto io_error;
        if(dest&&fwrite(buffer,1,chunk,dest)!=chunk)goto io_error;
        fg_sha256_update(&hash,buffer,chunk);bytes-=chunk;
    }
    fg_sha256_final(&hash,digest);
    if(expected&&memcmp(expected,digest,32)){
        fg_error_set(err,FG_ERR_MISMATCH,"repack source extent SHA-256 mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
io_error:
    fg_error_set(err,FG_ERR_IO,"repack extent read/write failed: %s",strerror(errno));
    return FG_ERR_IO;
}

static fg_status pad_file(FILE *file,uint64_t *offset,fg_error *err){
    uint8_t zero[FG_ALIGNMENT]={0};
    uint64_t end=fg_align_up_u64(*offset,FG_ALIGNMENT),bytes=end-*offset;
    if(bytes&&fwrite(zero,1,(size_t)bytes,file)!=bytes){
        fg_error_set(err,FG_ERR_IO,"repack alignment write failed");return FG_ERR_IO;
    }
    *offset=end;return FG_OK;
}

static fg_status close_file(FILE **file,fg_error *err){
    if(!*file)return FG_OK;
    bool ok=fflush(*file)==0&&fsync(fileno(*file))==0;
    if(fclose(*file)!=0)ok=false;
    *file=NULL;
    if(!ok){fg_error_set(err,FG_ERR_IO,"flush repack output");return FG_ERR_IO;}
    return FG_OK;
}

/* An owner-local expert record is expert-major, including cooked matrices.
 * Moving its entire byte span changes placement, not quantization or order. */
fg_status fg_repack_run(const fg_repack_options *o,fg_error *err){
    if(!o||!o->input_dir||!o->output_dir){
        fg_error_set(err,FG_ERR_ARGUMENT,"repack requires input and output directories");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_rank_count_set(o->rank_count,err);
    if(status!=FG_OK)return status;
    if(o->endpoint_prefix&&o->endpoint_first>256u-o->rank_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"endpoint suffixes exceed 255");return FG_ERR_ARGUMENT;
    }
    fg_manifest *source=malloc(sizeof(*source)),*m=malloc(sizeof(*m));
    if(!source||!m){free(source);free(m);fg_error_set(err,FG_ERR_OOM,"allocate repack metadata");return FG_ERR_OOM;}
    char path[1200],name[128];bool created=false,tokenizer_created=false;
    FILE *inputs[FG_MAX_RANK_COUNT]={0},*outputs[FG_MAX_RANK_COUNT]={0};
    FILE *full=NULL,*artifact=NULL;
    uint64_t offsets[FG_MAX_RANK_COUNT]={0};
    status=repack_path(path,o->input_dir,"manifest.fgm",err);
    if(status==FG_OK)status=fg_manifest_read(path,source,err);
    if(status!=FG_OK)goto done;
    if(!o->skip_model_validation){
        status=fg_q38_validate_packed_manifest(source,err);
        if(status!=FG_OK)goto done;
    }
    for(uint32_t i=0;i<source->tensor_count;i++){
        const fg_tensor_record *r=&source->tensors[i];
        if(r->kind==FG_TENSOR_ROUTED_EXPERT&&
           (r->dims!=3u||r->shape[2]!=FG_EXPERT_COUNT||
            fg_topology_layer_rank_experts(source,r->layer,r->rank)!=FG_EXPERT_COUNT)){
            fg_error_set(err,FG_ERR_MISMATCH,"repack requires owner-local expert records: %s",r->name);
            status=FG_ERR_MISMATCH;goto done;
        }
        if(r->kind==FG_TENSOR_HOST_CACHE||r->kind==FG_TENSOR_VISION){
            fg_error_set(err,FG_ERR_MISMATCH,"unsupported repack artifact: %s",r->name);
            status=FG_ERR_MISMATCH;goto done;
        }
    }
    status=fg_rank_count_set(o->rank_count,err);
    if(status!=FG_OK)goto done;
    fg_manifest_init(m);
    m->flags=source->flags;m->required_cu=source->required_cu;
    m->native_context=source->native_context;m->max_context=source->max_context;
    m->prefill_microbatch=source->prefill_microbatch;m->prefill_window=source->prefill_window;
    m->persistent_cap_bytes=source->persistent_cap_bytes;
    m->residency_cap_bytes=source->residency_cap_bytes;
    memcpy(m->source_sha256,source->source_sha256,32);
    memcpy(m->quant_profile_sha256,source->quant_profile_sha256,32);
    m->session=source->session;
    memset(m->session.rank_state_format_sha256,0,sizeof(m->session.rank_state_format_sha256));
    if(o->rank_count==source->rank_count){
        memcpy(m->layer_owner,source->layer_owner,sizeof(m->layer_owner));
        memcpy(m->layer_groups,source->layer_groups,sizeof(m->layer_groups));
        memcpy(m->expert_rank,source->expert_rank,sizeof(m->expert_rank));
    }
    if(mkdir(o->output_dir,0755)!=0){
        fg_error_set(err,FG_ERR_IO,"repack output must be a new directory: %s",strerror(errno));
        status=FG_ERR_IO;goto done;
    }
    created=true;
    for(uint32_t rank=0;rank<m->rank_count;rank++){
        snprintf(name,sizeof(name),"rank-%02u.fgw",rank);
        status=repack_path(path,o->output_dir,name,err);
        if(status!=FG_OK)goto done;
        outputs[rank]=fopen(path,"wbx");
        if(!outputs[rank]){status=FG_ERR_IO;fg_error_set(err,status,"create %s: %s",path,strerror(errno));goto done;}
        if(o->endpoint_prefix){
            if(snprintf(m->ranks[rank].endpoint,FG_MAX_ENDPOINT,"%s%u:19100",
                        o->endpoint_prefix,o->endpoint_first+rank)>=(int)FG_MAX_ENDPOINT){
                status=FG_ERR_LIMIT;fg_error_set(err,status,"endpoint is too long");goto done;
            }
        }else if(rank<source->rank_count)
            memcpy(m->ranks[rank].endpoint,source->ranks[rank].endpoint,FG_MAX_ENDPOINT);
        else snprintf(m->ranks[rank].endpoint,FG_MAX_ENDPOINT,"192.0.2.%u:19100",42u+rank);
        m->ranks[rank].transient_bytes=FG_PACK_RANK_TRANSIENT_BYTES;
        m->ranks[rank].driver_reserve_bytes=FG_PACK_DRIVER_RESERVE_BYTES;
        m->ranks[rank].scratch_bytes=fg_q38_runtime_scratch_bytes_for_manifest(
            m,rank,m->prefill_microbatch,m->prefill_window,m->max_context);
    }
    const fg_tensor_record *ngram=NULL;
    for(uint32_t i=0;i<source->tensor_count;i++){
        const fg_tensor_record *r=&source->tensors[i];
        fg_tensor_record record=*r;
        if(r->kind==FG_TENSOR_NGRAM){
            if(ngram){status=FG_ERR_FORMAT;fg_error_set(err,status,"duplicate n-gram tensor");goto done;}
            ngram=r;
            status=fg_manifest_add_tensor(m,r,err);
            if(status!=FG_OK)goto done;
            continue;
        }
        if(r->kind==FG_TENSOR_TOKENIZER){
            if(strcmp(r->name,"tokenizer/tokenizer.fgt")){
                status=FG_ERR_FORMAT;fg_error_set(err,status,"unexpected tokenizer artifact");goto done;
            }
            status=repack_path(path,o->output_dir,"tokenizer",err);
            if(status!=FG_OK)goto done;
            if(mkdir(path,0755)!=0){status=FG_ERR_IO;fg_error_set(err,status,"create tokenizer directory");goto done;}
            tokenizer_created=true;
            status=repack_path(path,o->input_dir,r->name,err);
            if(status!=FG_OK)goto done;
            FILE *input=fopen(path,"rb");
            status=repack_path(path,o->output_dir,r->name,err);
            if(status!=FG_OK){if(input)fclose(input);goto done;}
            artifact=fopen(path,"wbx");
            if(!input||!artifact){
                if(input)fclose(input);
                status=FG_ERR_IO;fg_error_set(err,status,"open tokenizer artifact");goto done;
            }
            status=copy_extent(input,r->offset,artifact,r->bytes,r->sha256,record.sha256,err);
            fclose(input);
            if(status==FG_OK)status=close_file(&artifact,err);
            if(status==FG_OK)status=fg_manifest_add_tensor(m,&record,err);
            if(status!=FG_OK)goto done;
            continue;
        }
        if(r->rank>=source->rank_count){
            status=FG_ERR_FORMAT;fg_error_set(err,status,"invalid source tensor rank");goto done;
        }
        uint32_t rank=r->layer<FG_LAYER_COUNT?m->layer_owner[r->layer]:0u;
        if(!strcmp(r->name,"output.weight")||!strncmp(r->name,"output_hc_",10u)||r->kind==FG_TENSOR_MTP)
            rank=FG_OUTPUT_RANK;
        if(m->rank_count==source->rank_count)rank=r->rank;
        if(r->kind==FG_TENSOR_ROUTED_EXPERT){
            const char *suffix=strrchr(r->name,'.');
            if(!suffix||strncmp(suffix,".rank",5u)){
                status=FG_ERR_FORMAT;fg_error_set(err,status,"expert record lacks rank suffix");goto done;
            }
            int base=(int)(suffix-r->name);
            if(snprintf(record.name,sizeof(record.name),"%.*s.rank%u",base,r->name,rank)>=(int)sizeof(record.name)){
                status=FG_ERR_LIMIT;fg_error_set(err,status,"expert name is too long");goto done;
            }
        }
        if(!inputs[r->rank]){
            snprintf(name,sizeof(name),"rank-%02u.fgw",(unsigned)r->rank);
            status=repack_path(path,o->input_dir,name,err);
            if(status!=FG_OK)goto done;
            inputs[r->rank]=fopen(path,"rb");
            if(!inputs[r->rank]){status=FG_ERR_IO;fg_error_set(err,status,"open %s: %s",path,strerror(errno));goto done;}
        }
        status=pad_file(outputs[rank],&offsets[rank],err);
        if(status!=FG_OK)goto done;
        record.offset=offsets[rank];record.rank=(uint16_t)rank;
        status=copy_extent(inputs[r->rank],r->offset,outputs[rank],r->bytes,
                           r->sha256,record.sha256,err);
        if(status!=FG_OK)goto done;
        offsets[rank]+=r->bytes;m->ranks[rank].tensor_count++;
        status=fg_manifest_add_tensor(m,&record,err);
        if(status!=FG_OK)goto done;
    }
    for(uint32_t rank=0;rank<m->rank_count;rank++){
        status=pad_file(outputs[rank],&offsets[rank],err);
        if(status==FG_OK)status=close_file(&outputs[rank],err);
        if(status!=FG_OK)goto done;
        m->ranks[rank].persistent_bytes=offsets[rank];
    }
    if(ngram){
        status=repack_path(path,o->output_dir,"ngram.iq4nl",err);
        if(status!=FG_OK)goto done;
        full=fopen(path,"wbx");
        if(!full){status=FG_ERR_IO;fg_error_set(err,status,"create full n-gram");goto done;}
        status=repack_path(path,o->input_dir,"ngram.iq4nl",err);
        if(status!=FG_OK)goto done;
        FILE *input=fopen(path,"rb");
        uint8_t digest[32];
        if(input){
            status=copy_extent(input,ngram->offset,full,ngram->bytes,ngram->sha256,digest,err);
            fclose(input);
        }else{
            uint64_t next=0u;
            for(uint32_t shard=0;status==FG_OK&&shard<source->ngram_shard_count;shard++){
                const fg_ngram_shard_record *s=&source->ngram_shards[shard];
                if(s->row_begin!=next){status=FG_ERR_FORMAT;fg_error_set(err,status,"source n-gram shard gap");break;}
                snprintf(name,sizeof(name),FG_NGRAM_SHARD_ARTIFACT_FORMAT,s->logical_rank);
                status=repack_path(path,o->input_dir,name,err);
                if(status!=FG_OK)break;
                input=fopen(path,"rb");
                if(!input){status=FG_ERR_IO;fg_error_set(err,status,"open %s",path);break;}
                status=copy_extent(input,0u,full,s->bytes,s->sha256,digest,err);
                fclose(input);next+=s->row_count;
            }
            if(status==FG_OK&&next*FG_NGRAM_ROW_BYTES!=ngram->bytes){
                status=FG_ERR_MISMATCH;fg_error_set(err,status,"incomplete source n-gram shards");
            }
        }
        if(status==FG_OK)status=close_file(&full,err);
        if(status!=FG_OK)goto done;
        status=repack_path(path,o->output_dir,"ngram.iq4nl",err);
        if(status==FG_OK)status=fg_sha256_file(path,digest,err);
        if(status==FG_OK&&memcmp(digest,ngram->sha256,32)){
            status=FG_ERR_MISMATCH;fg_error_set(err,status,"reconstructed full n-gram SHA-256 mismatch");
        }
        if(status!=FG_OK)goto done;
        full=fopen(path,"rb");
        if(!full){status=FG_ERR_IO;fg_error_set(err,status,"reopen n-gram");goto done;}
        for(uint32_t rank=1u;rank<m->rank_count;rank++){
            fg_ngram_shard_record *s=&m->ngram_shards[rank-1u];
            s->logical_rank=rank;
            status=fg_q38_ngram_rank_range(rank,&s->row_begin,&s->row_count,err);
            if(status!=FG_OK)goto done;
            s->bytes=s->row_count*FG_NGRAM_ROW_BYTES;
            snprintf(name,sizeof(name),FG_NGRAM_SHARD_ARTIFACT_FORMAT,rank);
            status=repack_path(path,o->output_dir,name,err);
            if(status!=FG_OK)goto done;
            artifact=fopen(path,"wbx");
            if(!artifact){status=FG_ERR_IO;fg_error_set(err,status,"create n-gram shard");goto done;}
            status=copy_extent(full,s->row_begin*FG_NGRAM_ROW_BYTES,artifact,s->bytes,NULL,s->sha256,err);
            if(status==FG_OK)status=close_file(&artifact,err);
            if(status!=FG_OK)goto done;
            m->host_resident_bytes[rank]=s->bytes;m->ngram_shard_count++;
        }
        fclose(full);full=NULL;
    }
    fg_q38_account_session_state(m);fg_topology_seal(m);
    if(!o->skip_model_validation){
        status=fg_q38_validate_packed_manifest(m,err);
        if(status==FG_OK)status=fg_q38_validate_ngram_shards(m,err);
        if(status!=FG_OK)goto done;
    }
    status=repack_path(path,o->output_dir,"manifest.fgm",err);
    if(status==FG_OK)status=fg_manifest_write(path,m,err);
    if(status==FG_OK)status=fg_manifest_read(path,m,err);
    if(status==FG_OK&&!o->skip_model_validation)
        status=fg_manifest_validate_deployment(m,err);
    if(status==FG_OK){
        status=repack_path(path,o->output_dir,"memory.json",err);
        if(status!=FG_OK)goto done;
        artifact=fopen(path,"wbx");
        if(!artifact){status=FG_ERR_IO;fg_error_set(err,status,"create memory ledger");goto done;}
        fprintf(artifact,"{\"rank_count\":%u,\"planning_budget_bytes\":8000000000,"
            "\"note\":\"pageable_estimate includes 128 MiB hot ngram; unpinned file cache is reclaimable, not bounded here\","
            "\"ranks\":[\n",m->rank_count);
        for(uint32_t rank=0;rank<m->rank_count;rank++){
            const fg_rank_record *r=&m->ranks[rank];
            uint64_t total=0u;
            status=fg_q38_rank_residency_bytes(m,rank,&total,err);
            if(status!=FG_OK)goto done;
            if(total>m->residency_cap_bytes){
                status=FG_ERR_LIMIT;
                fg_error_set(err,status,"rank %u conservative residency exceeds sealed cap",rank);
                goto done;
            }
            uint64_t host=m->host_resident_bytes[rank];
            uint64_t hot=host<FG_NGRAM_PAGEABLE_HOT_BYTES?host:FG_NGRAM_PAGEABLE_HOT_BYTES;
            fprintf(artifact,"%s{\"rank\":%u,\"persistent\":%llu,\"ngram_mapped\":%llu,"
                "\"transient\":%llu,\"kv\":%llu,\"scratch\":%llu,\"driver\":%llu,"
                "\"state_file\":%llu,\"conservative_total\":%llu,\"pageable_estimate\":%llu}",
                rank?",\n":"",rank,(unsigned long long)r->persistent_bytes,
                (unsigned long long)host,(unsigned long long)r->transient_bytes,
                (unsigned long long)r->kv_bytes,(unsigned long long)r->scratch_bytes,
                (unsigned long long)r->driver_reserve_bytes,(unsigned long long)r->state_file_bytes,
                (unsigned long long)total,(unsigned long long)(total-host+hot));
        }
        fprintf(artifact,"\n]}\n");
        if(ferror(artifact)){status=FG_ERR_IO;fg_error_set(err,status,"write memory ledger");goto done;}
        status=close_file(&artifact,err);
        if(status==FG_OK)fg_manifest_print(m);
    }
done:
    for(uint32_t rank=0;rank<FG_MAX_RANK_COUNT;rank++){
        if(inputs[rank])fclose(inputs[rank]);
        if(outputs[rank])fclose(outputs[rank]);
    }
    if(full)fclose(full);
    if(artifact)fclose(artifact);
    if(status!=FG_OK&&created){
        for(uint32_t rank=0;rank<o->rank_count;rank++){
            snprintf(name,sizeof(name),"rank-%02u.fgw",rank);
            if(repack_path(path,o->output_dir,name,NULL)==FG_OK)unlink(path);
            snprintf(name,sizeof(name),FG_NGRAM_SHARD_ARTIFACT_FORMAT,rank);
            if(repack_path(path,o->output_dir,name,NULL)==FG_OK)unlink(path);
        }
        const char *names[]={"manifest.fgm","ngram.iq4nl","tokenizer/tokenizer.fgt","memory.json"};
        for(uint32_t i=0;i<4u;i++)
            if(repack_path(path,o->output_dir,names[i],NULL)==FG_OK)unlink(path);
        if(tokenizer_created&&repack_path(path,o->output_dir,"tokenizer",NULL)==FG_OK)rmdir(path);
        rmdir(o->output_dir);
    }
    free(m);free(source);return status;
}
