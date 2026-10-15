#include "fg_manifest.h"
#include "fg_ledger.h"
#include "fg_ngram.h"
#include "fg_output.h"
#include "fg_pack.h"
#include "fg_q38_schema.h"
#include "fg_runtime.h"
#include "fg_quant.h"
#include "fg_sha256.h"
#include "fg_topology.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(expr) do{if(!(expr)){fprintf(stderr,"%s:%d: %s (%s)\n",__FILE__,__LINE__,#expr,err.message);exit(1);}}while(0)

/* This generator also compiles against main to supply independent ABI goldens. */
static void golden(const char *directory){
    fg_error err={0};fg_manifest *m=malloc(sizeof(*m));CHECK(m);
    char path[512];
    const uint32_t versions[]={4u,5u,6u};
    for(uint32_t i=0;i<3u;i++){
        fg_manifest_init(m);m->format_version=versions[i];
        snprintf(path,sizeof(path),"%s/v%u.fgm",directory,versions[i]);
        CHECK(fg_manifest_write(path,m,&err)==FG_OK);
    }
    free(m);
}

#ifdef FG_MAX_RANK_COUNT
static void check_identity(void){
    static const char *expected[]={
        "752a42edfee852bc4edbc22572e18779cb509215dd780a019f8eed78f95ca503",
        "8ed0b4d30257f84e751e8fcd445a479b716002a6483327862ef1d6bc1a25550b",
        "962fe26946e08659961425b667b8d0d2d4c1eac5e1292025f832bdfb62789998"
    };
    fg_error err={0};char directory[128],path[256],hex[65];
    CHECK(fg_rank_count_set(8u,&err)==FG_OK);
    snprintf(directory,sizeof(directory),"tests/n-rank-identity-%ld",(long)getpid());
    CHECK(mkdir(directory,0755)==0);golden(directory);
    for(uint32_t i=0;i<3u;i++){
        uint8_t digest[32];
        snprintf(path,sizeof(path),"%s/v%u.fgm",directory,4u+i);
        CHECK(fg_sha256_file(path,digest,&err)==FG_OK);
        for(uint32_t b=0;b<32u;b++)snprintf(hex+b*2u,3u,"%02x",digest[b]);
        CHECK(!strcmp(hex,expected[i]));unlink(path);
    }
    rmdir(directory);
}

static void memory_geometry(void){
    fg_error err={0};CHECK(fg_rank_count_set(12u,&err)==FG_OK);
    fg_manifest *m=malloc(sizeof(*m));CHECK(m);fg_manifest_init(m);
    CHECK(fg_runtime_profile_apply(m,FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128,&err)==FG_OK);
    for(uint32_t rank=0;rank<12u;rank++)
        printf("rank=%u scratch=%llu kv=%llu state_file=%llu\n",rank,
            (unsigned long long)m->ranks[rank].scratch_bytes,
            (unsigned long long)m->ranks[rank].kv_bytes,
            (unsigned long long)m->ranks[rank].state_file_bytes);
    free(m);
}

static void check_goldens(const char *directory){
    fg_error err={0};fg_manifest *m=malloc(sizeof(*m));CHECK(m);
    char path[512],out[512];
    for(uint32_t version=4u;version<=6u;version++){
        snprintf(path,sizeof(path),"%s/v%u.fgm",directory,version);
        CHECK(fg_manifest_read(path,m,&err)==FG_OK);
        CHECK(m->rank_count==8u&&m->format_version==version);
        snprintf(out,sizeof(out),"%s/new-v%u.fgm",directory,version);
        CHECK(fg_manifest_write(out,m,&err)==FG_OK);
        uint8_t a[32],b[32];
        CHECK(fg_sha256_file(path,a,&err)==FG_OK);
        CHECK(fg_sha256_file(out,b,&err)==FG_OK);
        CHECK(!memcmp(a,b,32));unlink(out);
    }
    free(m);
}

static void check_layout(uint32_t ranks){
    fg_error err={0};CHECK(fg_rank_count_set(ranks,&err)==FG_OK);
    fg_manifest *m=malloc(sizeof(*m));CHECK(m);fg_manifest_init(m);
    CHECK(m->rank_count==ranks);
    CHECK(fg_topology_validate(m,&err)==FG_OK);
    fg_runtime_options options={0};char ledger[FG_LEDGER_LINE_MAX];
    fg_ledger_inputs inputs={.manifest=m,.options=&options,.rank=0u,
        .prefill_frames=8u,.ring_prefill=true,.ring_decode=true};
    CHECK(fg_ledger_format(ledger,sizeof(ledger),&inputs,&err)==FG_OK);
    if(ranks==12u)CHECK(strstr(ledger,"wire_hops=13 ")&&strstr(ledger,"ranks=12 "));
    if(ranks==8u){
        for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
            CHECK(m->layer_owner[layer]==layer%8u);
        CHECK(fg_output_owner_rank(m)==4u);
    }else{
        for(uint32_t rank=0;rank<ranks;rank++){
            uint32_t layers=0u,qsa=0u;
            for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
                if(m->layer_owner[layer]==rank){
                    layers++;qsa+=(layer&3u)==3u;
                    CHECK(layer/(FG_LAYER_COUNT/ranks)==rank);
                    CHECK(fg_topology_layer_rank_experts(m,layer,rank)==FG_EXPERT_COUNT);
                }
            CHECK(layers==FG_LAYER_COUNT/ranks);
            if(ranks==12u)CHECK(qsa==1u);
            CHECK(m->ranks[rank].scratch_bytes==fg_q38_runtime_scratch_bytes_for_manifest(
                m,rank,m->prefill_microbatch,m->prefill_window,m->max_context));
        }
        CHECK(fg_output_owner_rank(m)==ranks-1u);
    }
    fg_tensor_record full={.kind=FG_TENSOR_NGRAM,.ggml_type=20u,
        .rank=UINT16_MAX,.layer=UINT16_MAX,.expert=UINT16_MAX,.dims=2u,
        .shape={160u,320001536u},.bytes=UINT64_C(320001536)*90u};
    snprintf(full.name,sizeof(full.name),"per_layer_token_embd.weight");
    CHECK(fg_manifest_add_tensor(m,&full,&err)==FG_OK);
    uint64_t next=0u;
    for(uint32_t rank=1u;rank<ranks;rank++){
        fg_ngram_shard_record *s=&m->ngram_shards[rank-1u];
        s->logical_rank=rank;
        CHECK(fg_q38_ngram_rank_range(rank,&s->row_begin,&s->row_count,&err)==FG_OK);
        CHECK(s->row_begin==next);next+=s->row_count;
        s->bytes=s->row_count*FG_NGRAM_ROW_BYTES;s->sha256[0]=1u;
        m->host_resident_bytes[rank]=s->bytes;m->ngram_shard_count++;
    }
    CHECK(next==UINT64_C(320001536));
    CHECK(fg_q38_validate_ngram_shards(m,&err)==FG_OK);
    char path[128];snprintf(path,sizeof(path),"tests/n-rank-layout-%ld.fgm",(long)getpid());
    CHECK(fg_manifest_write(path,m,&err)==FG_OK);
    CHECK(fg_manifest_read(path,m,&err)==FG_OK);
    CHECK(m->rank_count==ranks&&m->ngram_shard_count==ranks-1u);
    CHECK(fg_q38_validate_ngram_shards(m,&err)==FG_OK);unlink(path);
    if(ranks==8u)CHECK(m->ngram_shards[3].row_count==40000158u);
    if(ranks==12u){
        m->ngram_shards[10].row_begin++;
        CHECK(fg_q38_validate_ngram_shards(m,&err)==FG_ERR_MISMATCH);
    }
    free(m);
}

static void check_high_rank_protocol(void){
    fg_error err={0};CHECK(fg_rank_count_set(12u,&err)==FG_OK);
    float values[FG_HIDDEN_SIZE]={1.0f},decoded[FG_HIDDEN_SIZE];
    fg_prefill_result_pair pair={.token_slot=0u,.routing_slot=0u},out_pair;
    fg_prefill_result r={.layer=47u,.source_rank=11u,.destination_rank=0u,
        .contributor_mask=1u<<11u,.token_count=1u,.pair_count=1u,
        .pairs=&pair,.outputs=values},out={0};
    uint8_t wire[FG_PREFILL_RESULT_HEADER_BYTES+4u+FG_HIDDEN_SIZE*4u];uint32_t bytes;
    CHECK(fg_prefill_result_encode(wire,sizeof(wire),&bytes,&r,&err)==FG_OK);
    CHECK(wire[3]==4u&&wire[12]==0u&&wire[13]==8u);
    CHECK(fg_prefill_result_decode(&out,&out_pair,1u,decoded,FG_HIDDEN_SIZE,wire,bytes,&err)==FG_OK);
    CHECK(out.contributor_mask==(1u<<11u)&&out.source_rank==11u);
    CHECK(!memcmp(values,decoded,sizeof(values)));
    fg_output_result result={.source_rank=11u,.destination_rank=0u,.token=17u,.logit=1.0f};
    uint8_t result_wire[FG_OUTPUT_RESULT_BYTES];
    CHECK(fg_output_result_encode(result_wire,&result,&err)==FG_OK);
    CHECK(fg_rank_count_set(8u,&err)==FG_OK);
    CHECK(fg_prefill_result_decode(&out,&out_pair,1u,decoded,FG_HIDDEN_SIZE,wire,bytes,&err)!=FG_OK);
    r.source_rank=7u;r.contributor_mask=1u<<7u;
    CHECK(fg_prefill_result_encode(wire,sizeof(wire),&bytes,&r,&err)==FG_OK);
    CHECK(wire[3]==3u&&wire[12]==128u&&!wire[13]);
}

static void check_repack(void){
    fg_error err={0};CHECK(fg_rank_count_set(8u,&err)==FG_OK);
    char directory[128],output[128],path[256];
    snprintf(directory,sizeof(directory),"tests/n-rank-source-%ld",(long)getpid());
    snprintf(output,sizeof(output),"tests/n-rank-dest-%ld",(long)getpid());
    CHECK(mkdir(directory,0755)==0);
    fg_manifest *m=malloc(sizeof(*m));CHECK(m);fg_manifest_init(m);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        m->layer_owner[layer]=(uint8_t)(layer/6u);
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++)m->layer_groups[layer][g]=m->layer_owner[layer];
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)m->expert_rank[layer][e]=m->layer_owner[layer];
    }
    float experts[FG_EXPERT_COUNT];for(uint32_t i=0;i<FG_EXPERT_COUNT;i++)experts[i]=(float)i;
    fg_tensor_record r={.kind=FG_TENSOR_ROUTED_EXPERT,.layer=47u,.rank=7u,
        .expert=UINT16_MAX,.ggml_type=0u,.dims=3u,.shape={1u,1u,FG_EXPERT_COUNT},
        .bytes=sizeof(experts)};
    snprintf(r.name,sizeof(r.name),"blk.47.ffn_gate_exps.weight.rank7");
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,experts,sizeof(experts));
    fg_sha256_final(&hash,r.sha256);
    CHECK(fg_manifest_add_tensor(m,&r,&err)==FG_OK);
    snprintf(path,sizeof(path),"%s/rank-07.fgw",directory);
    FILE *file=fopen(path,"wb");CHECK(file);
    CHECK(fwrite(experts,1,sizeof(experts),file)==sizeof(experts));
    uint8_t padding[FG_ALIGNMENT-sizeof(experts)]={0};
    CHECK(fwrite(padding,1,sizeof(padding),file)==sizeof(padding));
    fg_tensor_record cooked=r;
    snprintf(cooked.name,sizeof(cooked.name),"blk.47.ffn_up_exps.weight.rank7");
    cooked.ggml_type=8u;cooked.layout=FG_TENSOR_LAYOUT_Q8_0_EXPERT_COOKED;
    cooked.shape[0]=32u;cooked.shape[1]=32u;cooked.offset=FG_ALIGNMENT;
    cooked.bytes=fg_q8_0_cooked_matrix_bytes(32u,32u)*FG_EXPERT_COUNT;
    uint8_t *cooked_data=malloc((size_t)cooked.bytes);CHECK(cooked_data);
    for(uint64_t i=0;i<cooked.bytes;i++)cooked_data[i]=(uint8_t)(i*19u+i/31u);
    fg_sha256_init(&hash);fg_sha256_update(&hash,cooked_data,(size_t)cooked.bytes);
    fg_sha256_final(&hash,cooked.sha256);
    CHECK(fwrite(cooked_data,1,(size_t)cooked.bytes,file)==cooked.bytes);
    uint64_t end=fg_align_up_u64(cooked.offset+cooked.bytes,FG_ALIGNMENT);
    for(uint64_t i=cooked.offset+cooked.bytes;i<end;i++)CHECK(fputc(0,file)!=EOF);
    CHECK(fclose(file)==0);free(cooked_data);
    CHECK(fg_manifest_add_tensor(m,&cooked,&err)==FG_OK);
    m->ranks[7].persistent_bytes=end;m->ranks[7].tensor_count=2u;
    snprintf(path,sizeof(path),"%s/manifest.fgm",directory);
    CHECK(fg_manifest_write(path,m,&err)==FG_OK);
    fg_repack_options options={.input_dir=directory,.output_dir=output,
        .rank_count=12u,.endpoint_prefix="192.168.42.",.endpoint_first=42u,.skip_model_validation=true};
    CHECK(fg_repack_run(&options,&err)==FG_OK);
    snprintf(path,sizeof(path),"%s/manifest.fgm",output);
    CHECK(fg_rank_count_set(8u,&err)==FG_OK);
    CHECK(fg_manifest_read(path,m,&err)==FG_OK);
    CHECK(FG_RANK_COUNT==12u&&m->format_version==7u);
    CHECK(m->tensors[0].rank==11u&&m->tensors[0].shape[2]==512u);
    CHECK(!strcmp(m->tensors[0].name,"blk.47.ffn_gate_exps.weight.rank11"));
    CHECK(!strcmp(m->ranks[11].endpoint,"192.168.42.53:19100"));
    snprintf(path,sizeof(path),"%s/rank-11.fgw",output);file=fopen(path,"rb");CHECK(file);
    float copied[FG_EXPERT_COUNT];
    CHECK(fread(copied,1,sizeof(copied),file)==sizeof(copied));fclose(file);
    CHECK(!memcmp(experts,copied,sizeof(experts)));
    CHECK(m->tensor_count==2u&&m->tensors[1].layout==cooked.layout);
    CHECK(!memcmp(m->tensors[1].sha256,cooked.sha256,32u));
    CHECK(fg_repack_run(&options,&err)==FG_ERR_IO);
    unlink(path);
    char bad[128];snprintf(bad,sizeof(bad),"tests/n-rank-bad-%ld",(long)getpid());
    snprintf(path,sizeof(path),"%s/rank-07.fgw",directory);
    file=fopen(path,"r+b");CHECK(file);CHECK(fputc(1,file)!=EOF);fclose(file);
    options.output_dir=bad;
    CHECK(fg_repack_run(&options,&err)==FG_ERR_MISMATCH);
    CHECK(access(bad,F_OK)!=0);
    snprintf(path,sizeof(path),"%s/rank-07.fgw",directory);unlink(path);
    snprintf(path,sizeof(path),"%s/manifest.fgm",directory);unlink(path);rmdir(directory);
    snprintf(path,sizeof(path),"%s/manifest.fgm",output);unlink(path);
    snprintf(path,sizeof(path),"%s/memory.json",output);unlink(path);
    for(uint32_t rank=0;rank<12u;rank++){
        snprintf(path,sizeof(path),"%s/rank-%02u.fgw",output,rank);unlink(path);
    }
    rmdir(output);free(m);
}
#endif

int main(int argc,char **argv){
    if(argc==3&&!strcmp(argv[1],"--golden")){golden(argv[2]);return 0;}
#ifdef FG_MAX_RANK_COUNT
    if(argc==3&&!strcmp(argv[1],"--check-golden")){check_goldens(argv[2]);return 0;}
    if(argc==2&&!strcmp(argv[1],"--memory-geometry")){memory_geometry();return 0;}
    check_identity();
    check_layout(8u);check_layout(12u);check_layout(4u);check_layout(6u);check_layout(16u);
    check_high_rank_protocol();check_repack();
    fg_error err={0};CHECK(fg_rank_count_set(0u,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_rank_count_set(7u,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_rank_count_set(17u,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_rank_count_set(8u,&err)==FG_OK);
    puts("N-rank topology, protocol, manifest and byte-preserving repack: PASS");
#endif
    return 0;
}
