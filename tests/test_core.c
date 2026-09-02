#include "fg_manifest.h"
#include "fg_embedding.h"
#include "fg_loader.h"
#include "fg_ngram.h"
#include "fg_pack.h"
#include "fg_protocol.h"
#include "fg_quant.h"
#include "fg_q38_math.h"
#include "fg_q38_schema.h"
#include "fg_qsa.h"
#include "fg_qsa_locality.h"
#include "fg_qsa_replica.h"
#include "fg_qsa_cache.h"
#include "fg_qsa_state.h"
#include "fg_sha256.h"
#include "fg_topology.h"

#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

static void test_sha(void){fg_sha256 c;uint8_t d[32];char hex[65];fg_sha256_init(&c);fg_sha256_update(&c,"abc",3);fg_sha256_final(&c,d);fg_sha256_hex(d,hex);CHECK(strcmp(hex,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")==0);}
static void test_topology(void){fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(!m)return;fg_manifest_init(m);for(uint32_t l=0;l<FG_LAYER_COUNT;l++){CHECK(m->layer_owner[l]==l%8);uint16_t count[8]={0};for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)count[m->expert_rank[l][e]]++;for(uint32_t g=0;g<4;g++)CHECK(count[m->layer_groups[l][g]]==128);}fg_error err={0};CHECK(fg_topology_validate(m,&err)==FG_OK);m->layer_offsets[3]=256u;fg_topology_seal(m);CHECK(fg_topology_validate(m,&err)==FG_ERR_FORMAT);m->layer_offsets[3]=0u;fg_topology_seal(m);fg_q38_account_session_state(m);uint64_t gdn_layer=(uint64_t)FG_Q38_GDN_CONV_WIDTH*16u+(uint64_t)FG_Q38_GDN_HEADS*128u*128u*4u;CHECK(m->ranks[0].kv_bytes==6u*gdn_layer);CHECK(m->ranks[0].state_file_bytes==0);CHECK(m->ranks[3].kv_bytes==(6ull*136u+12u)*FG_MAX_CONTEXT);CHECK(m->ranks[3].state_file_bytes==FG_Q38_QSA_STATE_PAGE_BYTES+6ull*(FG_MAX_CONTEXT/4u)*FG_Q38_QSA_STATE_PAGE_BYTES);CHECK(m->ranks[7].kv_bytes==m->ranks[3].kv_bytes&&m->ranks[7].state_file_bytes==m->ranks[3].state_file_bytes);free(m);}
static void test_pipeline_topology(void){
    fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(!m)return;
    fg_manifest_init(m);fg_topology_build_pipeline(m);fg_error err={0};
    CHECK(m->execution_mode==FG_EXECUTION_PIPELINE);
    CHECK(m->stage_count==FG_PIPELINE_STAGE_COUNT&&
          m->slot_count==FG_PIPELINE_DEFAULT_SLOT_COUNT);
    for(uint32_t stage=0;stage<FG_PIPELINE_STAGE_COUNT;stage++){
        CHECK(m->stage_ranks[stage]==stage);
        CHECK(m->layer_offsets[stage]==stage*FG_PIPELINE_DEFAULT_LAYERS_PER_STAGE);
        for(uint32_t layer=m->layer_offsets[stage];layer<m->layer_offsets[stage+1u];layer++){
            CHECK(m->layer_owner[layer]==stage);
            for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
                CHECK(m->expert_rank[layer][expert]==stage);
        }
    }
    CHECK(m->layer_offsets[FG_PIPELINE_STAGE_COUNT]==FG_LAYER_COUNT);
    CHECK(fg_topology_validate(m,&err)==FG_OK);
    CHECK(2u*fg_q38_pipeline_activation_slot_bytes(128u)==
          2ull*128ull*(10240ull*4ull+3ull*4ull));
    m->max_context=FG_NATIVE_CONTEXT;m->prefill_microbatch=128u;
    m->prefill_window=2u;fg_q38_account_session_state(m);
    uint64_t gdn=(uint64_t)FG_Q38_GDN_CONV_WIDTH*16u+
        (uint64_t)FG_Q38_GDN_HEADS*128u*128u*4u;
    uint64_t qsa=(uint64_t)m->max_context*
        (FG_Q38_QSA_TOKEN_RECORD_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES);
    uint64_t positions=(uint64_t)m->max_context*FG_Q38_QSA_POSITION_BYTES;
    CHECK(m->ranks[0].kv_bytes==5u*gdn+qsa+positions);
    CHECK(m->ranks[1].kv_bytes==4u*gdn+2u*qsa+positions);
    static const uint64_t scratch[FG_RANK_COUNT]={
        335544320u,268435456u,268435456u,268435456u,
        268435456u,268435456u,268435456u,268435456u
    };
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        CHECK(fg_q38_runtime_scratch_bytes_for_manifest(
                  m,rank,128u,2u,FG_NATIVE_CONTEXT)==scratch[rank]);
        uint64_t resident=FG_PERSISTENT_CAP_BYTES+(1024ull<<20u)+
            m->ranks[rank].kv_bytes+scratch[rank];
        CHECK(resident<=FG_RESIDENCY_CAP_BYTES);
    }
    m->expert_rank[17][511]=0u;fg_topology_seal(m);
    CHECK(fg_topology_validate(m,&err)==FG_ERR_FORMAT);free(m);
}
static void test_profile(void){fg_manifest *m=malloc(sizeof(*m));double (*p)[FG_EXPERT_COUNT]=calloc(FG_LAYER_COUNT,sizeof(*p));CHECK(m&&p);if(!m||!p){free(m);free(p);return;}fg_manifest_init(m);for(uint32_t l=0;l<FG_LAYER_COUNT;l++)for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)p[l][e]=(double)(FG_EXPERT_COUNT-e);fg_error err={0};CHECK(fg_topology_assign_profile(m,(const double (*)[FG_EXPERT_COUNT])p,&err)==FG_OK);for(uint32_t l=0;l<FG_LAYER_COUNT;l++){uint16_t count[8]={0};for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)count[m->expert_rank[l][e]]++;for(uint32_t g=0;g<4;g++)CHECK(count[m->layer_groups[l][g]]==128);}fg_topology_build_pipeline(m);uint8_t fingerprint[32];memcpy(fingerprint,m->topology_sha256,sizeof(fingerprint));CHECK(fg_topology_assign_profile(m,(const double (*)[FG_EXPERT_COUNT])p,&err)==FG_ERR_UNAVAILABLE);CHECK(memcmp(fingerprint,m->topology_sha256,sizeof(fingerprint))==0);CHECK(m->expert_rank[17][511]==m->layer_owner[17]);free(p);free(m);}
static void test_expert_map(void){fg_manifest *m=malloc(sizeof(*m));uint16_t (*map)[FG_EXPERT_COUNT]=malloc(sizeof(*map)*FG_LAYER_COUNT);CHECK(m&&map);if(!m||!map){free(map);free(m);return;}fg_manifest_init(m);for(uint32_t l=0;l<FG_LAYER_COUNT;l++)for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)map[l][e]=m->layer_groups[l][(e+1u)%FG_GROUP_SIZE];fg_error err={0};CHECK(fg_topology_assign_map(m,(const uint16_t (*)[FG_EXPERT_COUNT])map,&err)==FG_OK);CHECK(m->expert_rank[0][0]==m->layer_groups[0][1]);uint16_t saved=map[0][0];map[0][0]=2u;CHECK(fg_topology_assign_map(m,(const uint16_t (*)[FG_EXPERT_COUNT])map,&err)==FG_ERR_FORMAT);CHECK(m->expert_rank[0][0]==saved);map[0][0]=map[0][1];CHECK(fg_topology_assign_map(m,(const uint16_t (*)[FG_EXPERT_COUNT])map,&err)==FG_ERR_FORMAT);CHECK(m->expert_rank[0][0]==saved);free(map);free(m);}
static void test_expert_map_file(void){fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(!m)return;fg_manifest_init(m);char path[128];snprintf(path,sizeof(path),"/tmp/fg-expert-map-%ld.txt",(long)getpid());FILE *file=fopen(path,"w");CHECK(file!=NULL);if(file){static const uint32_t delta[FG_GROUP_SIZE]={0,1,3,5};fputs("# Flash Gordon route placement v1\n",file);for(uint32_t l=0;l<FG_LAYER_COUNT;l++){fprintf(file,"layer=%u ranks=",l);for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)fprintf(file,"%s%u",e?",":"",(l+delta[(e+1u)%FG_GROUP_SIZE])%FG_RANK_COUNT);fputc('\n',file);}fclose(file);fg_error err={0};CHECK(fg_topology_assign_map_file(m,path,&err)==FG_OK);CHECK(m->expert_rank[0][0]==m->layer_groups[0][1]);file=fopen(path,"w");CHECK(file!=NULL);if(file){fputs("layer=0 ranks=0\n",file);fclose(file);CHECK(fg_topology_assign_map_file(m,path,&err)==FG_ERR_FORMAT);CHECK(m->expert_rank[0][0]==m->layer_groups[0][1]);}}unlink(path);free(m);}
static void test_sealed_expert_map(void){fg_manifest *manifest=malloc(sizeof(*manifest)),*sealed=malloc(sizeof(*sealed));CHECK(manifest&&sealed);if(!manifest||!sealed){free(sealed);free(manifest);return;}fg_manifest_init(manifest);char map_path[128],manifest_path[128];snprintf(map_path,sizeof(map_path),"/tmp/fg-sealed-map-%ld.txt",(long)getpid());snprintf(manifest_path,sizeof(manifest_path),"/tmp/fg-sealed-map-%ld.fgm",(long)getpid());FILE *file=fopen(map_path,"w");CHECK(file!=NULL);if(file){for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){fprintf(file,"layer=%u ranks=",layer);for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)fprintf(file,"%s%u",expert?",":"",manifest->layer_groups[layer][(expert+1u)%FG_GROUP_SIZE]);fputc('\n',file);}fclose(file);fg_error err={0};CHECK(fg_topology_assign_map_file(manifest,map_path,&err)==FG_OK);CHECK(fg_manifest_write(manifest_path,manifest,&err)==FG_OK);unlink(map_path);CHECK(fg_manifest_read(manifest_path,sealed,&err)==FG_OK);CHECK(sealed->expert_rank[0][0]==sealed->layer_groups[0][1]);}unlink(map_path);unlink(manifest_path);free(sealed);free(manifest);}
static void test_deployment_profile(void){fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(!m)return;fg_manifest_init(m);m->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;char path[128];snprintf(path,sizeof(path),"/tmp/fg-profile-%ld.fgm",(long)getpid());fg_error err={0};CHECK(fg_manifest_write(path,m,&err)==FG_OK);fg_manifest *sealed=malloc(sizeof(*sealed));CHECK(sealed!=NULL);if(sealed){CHECK(fg_manifest_read(path,sealed,&err)==FG_OK);CHECK(fg_manifest_validate_deployment(sealed,&err)==FG_OK);sealed->flags&=~FG_MANIFEST_HAS_NGRAM;CHECK(fg_manifest_write(path,sealed,&err)==FG_OK);CHECK(fg_manifest_read(path,m,&err)==FG_OK);CHECK(fg_manifest_validate_deployment(m,&err)==FG_ERR_MISMATCH);free(sealed);}unlink(path);free(m);}
static void test_native_262k_profile_geometry(void){fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(!m)return;fg_manifest_init(m);m->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;m->max_context=m->native_context;m->prefill_microbatch=128u;m->session.logical_context_tokens=m->native_context;m->session.gpu_index_tokens=m->native_context;m->session.qsa_hot_record_tokens=8192u;m->session.host_page_cache_bytes=FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES;for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)m->ranks[rank].scratch_bytes=fg_q38_runtime_scratch_bytes(rank,m->prefill_microbatch,m->prefill_window,m->max_context);char path[128];snprintf(path,sizeof(path),"/tmp/fg-native-262k-%ld.fgm",(long)getpid());fg_error err={0};CHECK(fg_manifest_write(path,m,&err)==FG_OK);fg_manifest *sealed=malloc(sizeof(*sealed));CHECK(sealed!=NULL);if(sealed){CHECK(fg_manifest_read(path,sealed,&err)==FG_OK);CHECK(fg_manifest_validate_deployment(sealed,&err)==FG_OK);CHECK(sealed->prefill_microbatch==128u&&sealed->max_context==262144u&&sealed->session.logical_context_tokens==262144u&&sealed->session.gpu_index_tokens==262144u&&sealed->session.qsa_hot_record_tokens==8192u&&sealed->session.host_page_cache_bytes==FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES);free(sealed);}unlink(path);free(m);}
static void test_protocol(void){const char p[]="expert payload";fg_frame_header h;fg_error err={0};CHECK(fg_crc32c(NULL,0)==0u);CHECK(fg_crc32c("123456789",9)==UINT32_C(0xe3069283));CHECK(fg_frame_encode(&h,FG_MSG_EXPERT_RESULT,0x123456789abcdef0ull,7,0,p,sizeof(p),&err)==FG_OK);uint32_t n=0;CHECK(fg_frame_validate(&h,p,&n,&err)==FG_OK);CHECK(n==sizeof(p));char broken[sizeof(p)];memcpy(broken,p,sizeof(p));broken[0]^=1;CHECK(fg_frame_validate(&h,broken,NULL,&err)==FG_ERR_MISMATCH);int32_t tok[]={1,2,3};CHECK(fg_token_hash_update(0,tok,3)==fg_token_hash_update(fg_token_hash_update(0,tok,1),tok+1,2));}
static void test_layer_protocol(void){fg_layer_work *work=calloc(1,sizeof(*work)),*decoded=calloc(1,sizeof(*decoded));uint8_t *wire=malloc(FG_LAYER_WORK_MAX_BYTES);CHECK(work&&decoded&&wire);if(!work||!decoded||!wire){free(wire);free(decoded);free(work);return;}work->layer=1;work->source_rank=0;work->destination_rank=1;work->flags=FG_LAYER_WORK_HAS_NGRAM;work->token_index=123;work->position[0]=17;work->position[1]=23;work->position[2]=31;for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)work->hyper[i]=(float)i*0.001f;for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)work->ngram_embedding[i]=-(float)i*0.002f;uint32_t bytes=0;fg_error err={0};CHECK(fg_layer_work_encode(wire,FG_LAYER_WORK_MAX_BYTES,&bytes,FG_PROTOCOL_VERSION,work,&err)==FG_OK);CHECK(bytes==FG_LAYER_WORK_TEXT_MAX_BYTES);CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_VERSION,wire,bytes,&err)==FG_OK);CHECK(memcmp(work,decoded,sizeof(*work))==0);wire[3]=0;CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_VERSION,wire,bytes,&err)==FG_ERR_FORMAT);fg_layer_result *result=calloc(1,sizeof(*result)),*result_decoded=calloc(1,sizeof(*result_decoded));uint8_t *result_wire=malloc(FG_LAYER_RESULT_BYTES);CHECK(result&&result_decoded&&result_wire);if(result&&result_decoded&&result_wire){result->layer=7;result->source_rank=7;result->destination_rank=0;result->token_index=123;for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)result->hyper[i]=(float)i*0.003f;CHECK(fg_layer_result_encode(result_wire,result,&err)==FG_OK);CHECK(fg_layer_result_decode(result_decoded,result_wire,FG_LAYER_RESULT_BYTES,&err)==FG_OK);CHECK(memcmp(result,result_decoded,sizeof(*result))==0);result_wire[3]=1;CHECK(fg_layer_result_decode(result_decoded,result_wire,FG_LAYER_RESULT_BYTES,&err)==FG_ERR_FORMAT);}free(result_wire);free(result_decoded);free(result);free(wire);free(decoded);free(work);}

static void test_qsa_block_protocol(void){
    enum{TOKENS=2};fg_error err={0};float *hidden=malloc((size_t)TOKENS*FG_HIDDEN_SIZE*4u),*decoded=malloc((size_t)TOKENS*FG_HIDDEN_SIZE*4u);uint32_t positions[TOKENS*3u]={17u,17u,17u,18u,18u,18u},decoded_positions[TOKENS*3u];uint8_t *work_wire=malloc(FG_QSA_BLOCK_PREFILL_WORK_MAX_BYTES),*result_wire=malloc(FG_QSA_BLOCK_PREFILL_RESULT_MAX_BYTES);CHECK(hidden&&decoded&&work_wire&&result_wire);if(hidden&&decoded&&work_wire&&result_wire){for(uint32_t i=0;i<TOKENS*FG_HIDDEN_SIZE;i++)hidden[i]=sinf((float)i*0.001f);CHECK(FG_QSA_BLOCK_WORK_TEXT_BYTES==10264u);CHECK(FG_QSA_BLOCK_RESULT_BYTES==10248u);CHECK(12u*(FG_QSA_BLOCK_WORK_TEXT_BYTES+FG_QSA_BLOCK_RESULT_BYTES)==246144u);
        fg_qsa_block_work work={.layer=3u,.source_rank=0u,.destination_rank=3u,.position_mode=FG_POSITION_TEXT,.token_index=17u,.position={17u,17u,17u,0u},.hidden=hidden},decoded_work={0};uint32_t bytes=0;CHECK(fg_qsa_block_work_encode(work_wire,FG_QSA_BLOCK_WORK_MAX_BYTES,&bytes,FG_PROTOCOL_VERSION,&work,&err)==FG_OK);CHECK(bytes==FG_QSA_BLOCK_WORK_TEXT_BYTES);CHECK(fg_qsa_block_work_decode(&decoded_work,FG_PROTOCOL_VERSION,decoded,FG_HIDDEN_SIZE,work_wire,bytes,&err)==FG_OK);CHECK(decoded_work.token_index==17u&&memcmp(decoded,hidden,FG_HIDDEN_SIZE*4u)==0);CHECK(fg_qsa_block_work_decode(&decoded_work,FG_PROTOCOL_VERSION,decoded,FG_HIDDEN_SIZE,work_wire,bytes-1u,&err)==FG_ERR_FORMAT);work_wire[3]=1u;CHECK(fg_qsa_block_work_decode(&decoded_work,FG_PROTOCOL_VERSION,decoded,FG_HIDDEN_SIZE,work_wire,bytes,&err)==FG_ERR_FORMAT);work_wire[3]=0u;work_wire[FG_QSA_BLOCK_WORK_TEXT_HEADER_BYTES]=0x7fu;work_wire[FG_QSA_BLOCK_WORK_TEXT_HEADER_BYTES+1u]=0x80u;work_wire[FG_QSA_BLOCK_WORK_TEXT_HEADER_BYTES+2u]=0u;work_wire[FG_QSA_BLOCK_WORK_TEXT_HEADER_BYTES+3u]=0u;CHECK(fg_qsa_block_work_decode(&decoded_work,FG_PROTOCOL_VERSION,decoded,FG_HIDDEN_SIZE,work_wire,bytes,&err)==FG_ERR_FORMAT);CHECK(fg_qsa_block_work_encode(work_wire,FG_QSA_BLOCK_WORK_MAX_BYTES,&bytes,FG_PROTOCOL_MIN_VERSION,&work,&err)==FG_OK);CHECK(bytes==FG_QSA_BLOCK_WORK_LEGACY_HEADER_BYTES+FG_HIDDEN_SIZE*4u);CHECK(fg_qsa_block_work_decode(&decoded_work,FG_PROTOCOL_MIN_VERSION,decoded,FG_HIDDEN_SIZE,work_wire,bytes,&err)==FG_OK);
        fg_qsa_block_result result={.layer=3u,.source_rank=3u,.destination_rank=0u,.token_index=17u,.hidden=hidden},decoded_result={0};CHECK(fg_qsa_block_result_encode(result_wire,&result,&err)==FG_OK);CHECK(fg_qsa_block_result_decode(&decoded_result,decoded,FG_HIDDEN_SIZE,result_wire,FG_QSA_BLOCK_RESULT_BYTES,&err)==FG_OK);CHECK(memcmp(decoded,hidden,FG_HIDDEN_SIZE*4u)==0);
        fg_qsa_block_prefill_work prefill={.layer=7u,.source_rank=0u,.destination_rank=7u,.position_mode=FG_POSITION_TEXT,.first_token=17u,.token_count=TOKENS,.positions=positions,.hidden=hidden},decoded_prefill={0};CHECK(fg_qsa_block_prefill_work_encode(work_wire,FG_QSA_BLOCK_PREFILL_WORK_MAX_BYTES,&bytes,FG_PROTOCOL_VERSION,&prefill,&err)==FG_OK);CHECK(bytes==FG_QSA_BLOCK_PREFILL_HEADER_BYTES+TOKENS*3u*4u+TOKENS*FG_HIDDEN_SIZE*4u);CHECK(fg_qsa_block_prefill_work_decode(&decoded_prefill,FG_PROTOCOL_VERSION,decoded_positions,TOKENS*3u,decoded,(uint64_t)TOKENS*FG_HIDDEN_SIZE,work_wire,bytes,&err)==FG_OK);CHECK(memcmp(decoded_positions,positions,sizeof(positions))==0&&memcmp(decoded,hidden,(size_t)TOKENS*FG_HIDDEN_SIZE*4u)==0);
        fg_qsa_block_prefill_result prefill_result={.layer=7u,.source_rank=7u,.destination_rank=0u,.first_token=17u,.token_count=TOKENS,.hidden=hidden},decoded_prefill_result={0};CHECK(fg_qsa_block_prefill_result_encode(result_wire,FG_QSA_BLOCK_PREFILL_RESULT_MAX_BYTES,&bytes,&prefill_result,&err)==FG_OK);CHECK(bytes==FG_QSA_BLOCK_PREFILL_HEADER_BYTES+TOKENS*FG_HIDDEN_SIZE*4u);CHECK(fg_qsa_block_prefill_result_decode(&decoded_prefill_result,decoded,(uint64_t)TOKENS*FG_HIDDEN_SIZE,result_wire,bytes,&err)==FG_OK);CHECK(memcmp(decoded,hidden,(size_t)TOKENS*FG_HIDDEN_SIZE*4u)==0);
        fg_frame_header frame;CHECK(fg_frame_encode_version(&frame,FG_PROTOCOL_MIN_VERSION,FG_MSG_QSA_BLOCK_WORK,1u,1u,0u,work_wire,FG_QSA_BLOCK_WORK_LEGACY_HEADER_BYTES+FG_HIDDEN_SIZE*4u,&err)==FG_OK);CHECK(fg_frame_encode_version(&frame,FG_PROTOCOL_VERSION,FG_MSG_QSA_BLOCK_WORK,1u,1u,0u,work_wire,FG_QSA_BLOCK_WORK_LEGACY_HEADER_BYTES+FG_HIDDEN_SIZE*4u,&err)==FG_OK);CHECK(fg_frame_encode_version(&frame,FG_PROTOCOL_MIN_VERSION,FG_MSG_SESSION_PREPARE,1u,1u,0u,work_wire,1u,&err)==FG_ERR_ARGUMENT);
    }free(result_wire);free(work_wire);free(decoded);free(hidden);
}

static void test_qsa_page_protocol(void){
    enum{PAGES=2};fg_error err={0};
    uint32_t first_block=99u,block_count=99u;
    CHECK(fg_qsa_completed_page_range(0u,1u,&first_block,&block_count,&err)==FG_OK&&
          first_block==0u&&block_count==0u);
    CHECK(fg_qsa_completed_page_range(3u,1u,&first_block,&block_count,&err)==FG_OK&&
          first_block==0u&&block_count==1u);
    CHECK(fg_qsa_completed_page_range(4u,4u,&first_block,&block_count,&err)==FG_OK&&
          first_block==1u&&block_count==1u);
    CHECK(fg_qsa_completed_page_range(2u,6u,&first_block,&block_count,&err)==FG_OK&&
          first_block==0u&&block_count==2u);
    CHECK(fg_qsa_completed_page_range(FG_MAX_CONTEXT-1u,2u,&first_block,&block_count,
                                      &err)==FG_ERR_LIMIT);
    uint8_t *records=malloc((size_t)PAGES*FG_QSA_PAGE_RECORD_BYTES);
    uint8_t *wire=malloc(FG_QSA_PAGE_RESULT_MAX_BYTES);
    fg_qsa_page pages[PAGES],decoded_pages[PAGES];
    CHECK(records&&wire);
    if(!records||!wire){free(wire);free(records);return;}
    for(uint32_t i=0;i<PAGES*FG_QSA_PAGE_RECORD_BYTES;i++)records[i]=(uint8_t)(i*17u+3u);
    pages[0]=(fg_qsa_page){.layer=3u,.block=7u,.records=records};
    pages[1]=(fg_qsa_page){.layer=11u,.block=8u,
        .records=records+FG_QSA_PAGE_RECORD_BYTES};
    fg_qsa_page_batch batch={.source_rank=0u,.destination_rank=3u,.batch_id=19u,
        .page_count=PAGES,.pages=pages},decoded={0};uint32_t bytes=0;
    CHECK(FG_QSA_PAGE_RECORD_BYTES==4944u);
    CHECK(FG_QSA_PAGE_BATCH_HEADER_BYTES+
          FG_QSA_OWNER_LAYER_COUNT*FG_QSA_PAGE_ENTRY_BYTES==29724u);
    CHECK(2u*(29724u+(uint32_t)sizeof(fg_frame_header))/4u==14878u);
    CHECK(fg_qsa_page_append_encode(wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&bytes,
                                    &batch,&err)==FG_OK);
    CHECK(bytes==FG_QSA_PAGE_BATCH_HEADER_BYTES+PAGES*FG_QSA_PAGE_ENTRY_BYTES);
    CHECK(fg_qsa_page_append_decode(&decoded,decoded_pages,PAGES,wire,bytes,&err)==FG_OK);
    CHECK(decoded.batch_id==batch.batch_id&&decoded.page_count==PAGES&&
          decoded.pages[1].layer==11u&&decoded.pages[1].block==8u&&
          memcmp(decoded.pages[1].records,pages[1].records,FG_QSA_PAGE_RECORD_BYTES)==0);
    wire[10]=1u;
    CHECK(fg_qsa_page_append_decode(&decoded,decoded_pages,PAGES,wire,bytes,&err)==FG_ERR_FORMAT);
    wire[10]=0u;pages[1].layer=3u;pages[1].block=7u;
    CHECK(fg_qsa_page_append_encode(wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&bytes,
                                    &batch,&err)==FG_ERR_FORMAT);
    pages[1].layer=11u;pages[1].block=8u;pages[0].records=NULL;pages[1].records=NULL;
    CHECK(fg_qsa_page_fetch_encode(wire,FG_QSA_PAGE_FETCH_MAX_BYTES,&bytes,
                                   &batch,&err)==FG_OK);
    CHECK(bytes==FG_QSA_PAGE_BATCH_HEADER_BYTES+PAGES*FG_QSA_PAGE_ENTRY_HEADER_BYTES);
    CHECK(fg_qsa_page_fetch_decode(&decoded,decoded_pages,PAGES,wire,bytes,&err)==FG_OK);
    CHECK(decoded.pages[0].records==NULL&&decoded.pages[1].block==8u);
    pages[0].records=records;pages[1].records=records+FG_QSA_PAGE_RECORD_BYTES;
    CHECK(fg_qsa_page_result_encode(wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&bytes,
                                    &batch,&err)==FG_OK);
    CHECK(fg_qsa_page_result_decode(&decoded,decoded_pages,PAGES,wire,bytes,&err)==FG_OK);
    fg_frame_header page_frame;
    CHECK(fg_frame_encode_version(&page_frame,FG_PROTOCOL_VERSION,FG_MSG_QSA_PAGE_RESULT,
                                  1u,19u,0u,wire,bytes,&err)==FG_OK);
    CHECK(fg_frame_validate_version(&page_frame,FG_PROTOCOL_VERSION,wire,NULL,&err)==FG_OK);
    wire[FG_QSA_PAGE_BATCH_HEADER_BYTES+FG_QSA_PAGE_ENTRY_HEADER_BYTES]^=1u;
    CHECK(fg_frame_validate_version(&page_frame,FG_PROTOCOL_VERSION,wire,NULL,
                                    &err)==FG_ERR_MISMATCH);
    wire[FG_QSA_PAGE_BATCH_HEADER_BYTES+FG_QSA_PAGE_ENTRY_HEADER_BYTES]^=1u;
    batch.page_count=FG_QSA_PAGE_APPEND_MAX_PAGES+1u;
    CHECK(fg_qsa_page_append_encode(wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&bytes,
                                    &batch,&err)==FG_ERR_FORMAT);
    fg_qsa_page *oversized=calloc(FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES+1u,
                                  sizeof(*oversized));
    CHECK(oversized!=NULL);
    if(oversized){
        for(uint32_t i=0;i<=FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES;i++)
            oversized[i]=(fg_qsa_page){.layer=3u,.block=i,.records=records};
        batch.page_count=FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES+1u;
        batch.pages=oversized;
        CHECK(fg_qsa_page_append_encode(wire,FG_QSA_PAGE_RESULT_MAX_BYTES,&bytes,
                                       &batch,&err)==FG_ERR_FORMAT);
        free(oversized);
    }
    batch.page_count=PAGES;
    batch.pages=pages;
    fg_qsa_page_barrier barrier={.source_rank=0u,.destination_rank=3u,.batch_id=20u},
        decoded_barrier={0};uint8_t barrier_wire[FG_QSA_PAGE_BARRIER_BYTES];
    CHECK(fg_qsa_page_barrier_encode(barrier_wire,&barrier,&err)==FG_OK);
    CHECK(fg_qsa_page_barrier_decode(&decoded_barrier,barrier_wire,sizeof(barrier_wire),
                                     &err)==FG_OK);
    CHECK(decoded_barrier.batch_id==20u);
    fg_frame_header frame;
    CHECK(fg_frame_encode_version(&frame,FG_PROTOCOL_VERSION,FG_MSG_QSA_PAGE_APPEND,
                                  1u,19u,0u,wire,bytes,&err)==FG_OK);
    CHECK(fg_frame_encode_version(&frame,FG_PROTOCOL_MIN_VERSION,FG_MSG_QSA_PAGE_APPEND,
                                  1u,19u,0u,wire,bytes,&err)==FG_ERR_ARGUMENT);
    free(wire);free(records);
}

static uint64_t test_monotonic_ns(void){
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC,&value);
    return (uint64_t)value.tv_sec*UINT64_C(1000000000)+(uint64_t)value.tv_nsec;
}

static void test_prefill_chunk_frontiers(void){
    const uint32_t totals[]={127u,128u,129u,256u,416u};
    const uint32_t chunk_capacity=128u;
    const uint32_t qsa_layers=FG_LAYER_COUNT/4u;
    const uint32_t record_bytes=FG_Q38_QSA_TOKEN_RECORD_BYTES;
    for(uint32_t shape=0u;shape<sizeof(totals)/sizeof(totals[0]);shape++){
        uint32_t total=totals[shape];
        size_t state_bytes=(size_t)total*qsa_layers*record_bytes;
        uint8_t *reference=malloc(state_bytes),*chunked=calloc(1,state_bytes);
        uint32_t *reference_output=malloc((size_t)total*sizeof(*reference_output));
        uint32_t *chunked_output=calloc(total,sizeof(*chunked_output));
        CHECK(reference&&chunked&&reference_output&&chunked_output);
        if(!reference||!chunked||!reference_output||!chunked_output){
            free(chunked_output);free(reference_output);free(chunked);free(reference);
            continue;
        }
        for(uint32_t token=0u;token<total;token++){
            reference_output[token]=UINT32_C(0x9e3779b9)*(token+1u);
            for(uint32_t layer=0u;layer<qsa_layers;layer++)
                memset(reference+((size_t)layer*total+token)*record_bytes,
                       (int)(token+layer*17u),record_bytes);
        }
        uint32_t frontier=0u,append_batches=0u,expected_batches=0u;
        for(uint32_t first=0u;first<total;first+=chunk_capacity){
            uint32_t count=total-first;
            if(count>chunk_capacity)count=chunk_capacity;
            if((first+count)/FG_Q38_QSA_COMPRESS_RATIO>
               first/FG_Q38_QSA_COMPRESS_RATIO)expected_batches++;
        }
        uint64_t deadline=test_monotonic_ns()+UINT64_C(5000000000);
        while(frontier<total){
            uint32_t count=total-frontier;
            if(count>chunk_capacity)count=chunk_capacity;
            uint32_t first_block=UINT32_MAX,block_count=UINT32_MAX;
            fg_error error={0};
            CHECK(fg_qsa_completed_page_range(frontier,count,&first_block,
                                              &block_count,&error)==FG_OK);
            CHECK(first_block==frontier/FG_Q38_QSA_COMPRESS_RATIO&&
                  block_count==(frontier+count)/FG_Q38_QSA_COMPRESS_RATIO-
                               frontier/FG_Q38_QSA_COMPRESS_RATIO);
            for(uint32_t block=0u;block<block_count;block++)
                for(uint32_t layer=0u;layer<qsa_layers;layer++)
                    memcpy(chunked+((size_t)layer*total+
                                    (first_block+block)*FG_Q38_QSA_COMPRESS_RATIO)*
                               record_bytes,
                           reference+((size_t)layer*total+
                                      (first_block+block)*FG_Q38_QSA_COMPRESS_RATIO)*
                               record_bytes,
                           (size_t)FG_Q38_QSA_COMPRESS_RATIO*record_bytes);
            memcpy(chunked_output+(size_t)frontier,reference_output+frontier,
                   (size_t)count*sizeof(*chunked_output));
            if(block_count)append_batches++;
            frontier+=count;
            CHECK(test_monotonic_ns()<=deadline);
        }
        uint32_t complete=(total/FG_Q38_QSA_COMPRESS_RATIO)*
                          FG_Q38_QSA_COMPRESS_RATIO;
        CHECK(frontier==total&&append_batches==expected_batches);
        for(uint32_t layer=0u;layer<qsa_layers;layer++)
            CHECK(memcmp(chunked+(size_t)layer*total*record_bytes,
                         reference+(size_t)layer*total*record_bytes,
                         (size_t)complete*record_bytes)==0);
        CHECK(memcmp(chunked_output,reference_output,
                     (size_t)total*sizeof(*reference_output))==0);
        for(uint32_t layer=0u;layer<qsa_layers;layer++)
            for(uint32_t token=complete;token<total;token++){
                const uint8_t *tail=chunked+
                    ((size_t)layer*total+token)*record_bytes;
                for(uint32_t byte=0u;byte<record_bytes;byte++)
                    CHECK(tail[byte]==0u);
            }
        free(chunked_output);free(reference_output);free(chunked);free(reference);
    }
}

static void test_qsa_locality_metrics(void){
    uint32_t budget=1u;
    fg_qsa_locality *trace=fg_qsa_locality_create(FG_QSA_LOCALITY_SUMMARY,16u,8u,
                                                   &budget,1u,1234u);
    CHECK(trace!=NULL);
    if(!trace)return;
    uint32_t first[]={0u,1u,1u},second[]={0u,2u},third[]={1u,0u,3u};
    fg_qsa_locality_record_selection(trace,3u,8u,first,3u);
    fg_qsa_locality_record_selection(trace,3u,12u,second,2u);
    fg_qsa_locality_record_cache(trace,3u,false);
    fg_qsa_locality_record_selection(trace,3u,16u,third,3u);
    fg_qsa_locality_record_cache(trace,3u,true);
    fg_qsa_locality_stats stats={0};fg_qsa_locality_get_stats(trace,&stats);
    CHECK(stats.selections==3u&&stats.selected_refs==8u&&stats.unique_pages==7u);
    CHECK(stats.dedup_pages==1u&&stats.hot_tail_hits==4u&&stats.cold_refs==3u);
    CHECK(stats.previous_overlap==2u&&stats.cache_hits==1u&&stats.cache_misses==1u);
    CHECK(stats.first_cold_refs==2u&&stats.reused_cold_refs==1u);
    CHECK(stats.reuse_distance_sum==1u&&stats.reuse_distance_max==1u);
    CHECK(stats.budget_count==1u&&stats.budget_mib[0]==1u);
    CHECK(stats.budget_hits[0]==1u&&stats.budget_misses[0]==2u);
    CHECK(stats.selection_digest!=0u);
    fg_qsa_locality_reset(trace,NULL);fg_qsa_locality_get_stats(trace,&stats);
    CHECK(stats.selections==0u&&stats.cache_hits==0u&&stats.budget_count==1u);
    fg_qsa_locality_destroy(trace,NULL);
}

static void test_output_protocol(void){fg_output_work *work=calloc(1,sizeof(*work)),*decoded=calloc(1,sizeof(*decoded));uint8_t *wire=malloc(FG_OUTPUT_WORK_BYTES);CHECK(work&&decoded&&wire);if(work&&decoded&&wire){work->source_rank=0;work->destination_rank=4;work->token_index=91;work->sampler.temperature=1.0f;work->sampler.top_p=.95f;work->sampler.top_k=20u;work->sampler.presence_penalty=1.5f;work->sampler.frequency_penalty=0.25f;work->sampler.repetition_penalty=1.0f;work->sampler.min_p=0.0f;work->uniform=.37f;for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)work->hyper[i]=sinf((float)i*0.001f);fg_error err={0};CHECK(fg_output_work_encode(wire,work,&err)==FG_OK);CHECK(fg_output_work_decode(decoded,wire,FG_OUTPUT_WORK_BYTES,&err)==FG_OK);CHECK(decoded->sampler.temperature==work->sampler.temperature&&decoded->sampler.top_p==work->sampler.top_p&&decoded->sampler.top_k==work->sampler.top_k&&decoded->sampler.presence_penalty==work->sampler.presence_penalty&&decoded->sampler.frequency_penalty==work->sampler.frequency_penalty&&decoded->sampler.repetition_penalty==work->sampler.repetition_penalty&&decoded->sampler.min_p==work->sampler.min_p&&decoded->uniform==work->uniform);CHECK(memcmp(work->hyper,decoded->hyper,sizeof(work->hyper))==0);wire[2]=1;CHECK(fg_output_work_decode(decoded,wire,FG_OUTPUT_WORK_BYTES,&err)==FG_ERR_FORMAT);fg_output_result result={.source_rank=4,.destination_rank=0,.token_index=91,.token=42,.logit=12.5f},result_decoded;uint8_t result_wire[FG_OUTPUT_RESULT_BYTES];CHECK(fg_output_result_encode(result_wire,&result,&err)==FG_OK);CHECK(fg_output_result_decode(&result_decoded,result_wire,sizeof(result_wire),&err)==FG_OK);CHECK(result.source_rank==result_decoded.source_rank&&result.destination_rank==result_decoded.destination_rank&&result.token_index==result_decoded.token_index&&result.token==result_decoded.token&&result.logit==result_decoded.logit);result_wire[8]=0xff;CHECK(fg_output_result_decode(&result_decoded,result_wire,sizeof(result_wire),&err)==FG_ERR_FORMAT);}free(wire);free(decoded);free(work);}
static void test_ngram_protocol(void){fg_error err={0};fg_ngram_work work={.source_rank=0,.destination_rank=2,.item_count=3,.token_index=77,.heads={2u,4u,3u},.rows={40000031u,80000111u,60000071u}},decoded_work;uint8_t work_wire[FG_NGRAM_WORK_MAX_BYTES];uint32_t work_bytes=0;CHECK(fg_ngram_work_encode(work_wire,sizeof(work_wire),&work_bytes,&work,&err)==FG_OK);CHECK(work_bytes==35u);CHECK(fg_ngram_work_decode(&decoded_work,work_wire,work_bytes,&err)==FG_OK);CHECK(memcmp(&work,&decoded_work,sizeof(work))==0);CHECK(fg_ngram_work_decode(&decoded_work,work_wire,work_bytes-1u,&err)==FG_ERR_FORMAT);fg_ngram_result result={.source_rank=7,.destination_rank=0,.item_count=2,.token_index=77,.heads={15u,14u}},decoded_result;for(uint32_t i=0;i<2u*FG_NGRAM_WIRE_ROW_BYTES;i++)result.packed[i]=(uint8_t)(i*31u+9u);uint8_t result_wire[FG_NGRAM_RESULT_MAX_BYTES];uint32_t result_bytes=0;CHECK(fg_ngram_result_encode(result_wire,sizeof(result_wire),&result_bytes,&result,&err)==FG_OK);CHECK(result_bytes==8u+2u*(1u+FG_NGRAM_WIRE_ROW_BYTES));CHECK(fg_ngram_result_decode(&decoded_result,result_wire,result_bytes,&err)==FG_OK);CHECK(decoded_result.source_rank==result.source_rank&&decoded_result.destination_rank==result.destination_rank&&decoded_result.item_count==result.item_count&&decoded_result.token_index==result.token_index&&memcmp(decoded_result.heads,result.heads,2u)==0&&memcmp(decoded_result.packed,result.packed,2u*FG_NGRAM_WIRE_ROW_BYTES)==0);result_wire[3]=4u;CHECK(fg_ngram_result_decode(&decoded_result,result_wire,result_bytes,&err)==FG_ERR_FORMAT);}
static void test_ngram(void){uint64_t a[]={9000,2,4097,4100,16000,8191};fg_ngram_read r[6];uint32_t n=0;fg_error err={0};CHECK(fg_ngram_plan_reads(a,6,32768,r,6,&n,&err)==FG_OK);CHECK(n==2);CHECK(r[0].offset==0&&r[0].bytes==8192);CHECK(r[1].offset==8192&&r[1].bytes==8192);fg_ngram_cache *c=NULL;CHECK(fg_ngram_cache_create(&c,&err)==FG_OK);uint8_t *block=malloc(FG_NGRAM_BLOCK_BYTES);CHECK(block!=NULL);if(c&&block){memset(block,0xa5,FG_NGRAM_BLOCK_BYTES);CHECK(fg_ngram_cache_put(c,4096,block,&err)==FG_OK);const void *got=NULL;CHECK(fg_ngram_cache_get(c,4096,&got));CHECK(((const uint8_t *)got)[123]==0xa5);}free(block);fg_ngram_cache_destroy(c);
    const int32_t history[]={10,20,30};
    const uint64_t golden[]={9878115,26555603,54895210,62571545,80580723,119917398,128922427,147596134,168936175,195223391,219226064,233524685,246670267,279816194,297531600,306108296};
    uint64_t rows[FG_NGRAM_HEAD_COUNT],addresses[FG_NGRAM_HEAD_COUNT];
    CHECK(fg_q38_ngram_lookup(history,3,rows,addresses,&err)==FG_OK);
    for(uint32_t i=0;i<FG_NGRAM_HEAD_COUNT;i++){CHECK(rows[i]==golden[i]);CHECK(addresses[i]==golden[i]*FG_NGRAM_ROW_BYTES);}
    const int32_t reset_history[]={FG_Q38_EOS_TOKEN,5};
    const uint64_t reset_golden[]={15389869,39778609,55713969,62213332,88817728,118483999,133731511,155458159,179763390,197956758,205378969,220499474,242466248,265658744,293662119,315720898};
    CHECK(fg_q38_ngram_lookup(reset_history,2,rows,addresses,&err)==FG_OK);
    for(uint32_t i=0;i<FG_NGRAM_HEAD_COUNT;i++)CHECK(rows[i]==reset_golden[i]);
}
static void test_ngram_planner_batch_capacity(void){
    uint64_t addresses[]={12288,8192,4096,8192,12289,0,4097,8193};fg_ngram_read reads[6];uint32_t count=0;fg_error err={0};
    CHECK(fg_ngram_plan_reads(addresses,8,32768,reads,6,&count,&err)==FG_OK);CHECK(count==2);CHECK(reads[0].offset==0&&reads[0].bytes==8192);CHECK(reads[1].offset==8192&&reads[1].bytes==8192);
    CHECK(fg_ngram_plan_reads(addresses,8,32768,reads,1,&count,&err)==FG_ERR_LIMIT);uint64_t edge[]={4096};CHECK(fg_ngram_plan_reads(edge,1,4096,reads,1,&count,&err)==FG_ERR_FORMAT);
}
static void test_pipeline_owner_transient_geometry(void){
    fg_q38_pipeline_owner_transient_layout layout={0};
    CHECK(!fg_q38_pipeline_owner_transient_layout_get(0u,&layout));
    CHECK(!fg_q38_pipeline_owner_transient_layout_get(513u,&layout));
    CHECK(fg_q38_pipeline_owner_transient_bytes(0u)==UINT64_MAX);
    CHECK(fg_q38_pipeline_owner_transient_layout_get(128u,&layout));
    CHECK(layout.total_bytes==UINT64_C(16166912));
    CHECK(FG_Q38_DECODE_TILE_WORDS==9u);
    CHECK(FG_Q38_PIPELINE_DECODE_EXTRA_BYTES==UINT64_C(92200));
    CHECK(layout.bytes[FG_Q38_PIPELINE_TRANSIENT_SELECTED]==
          128u*FG_TOP_K*4u);
    CHECK(layout.bytes[FG_Q38_PIPELINE_TRANSIENT_GATES]==
          128u*FG_TOP_K*4u);
    CHECK(layout.bytes[FG_Q38_PIPELINE_TRANSIENT_PREFILL_TILES]==
          128u*FG_TOP_K*FG_Q38_PREFILL_TILE_WORDS*4u);
    CHECK(FG_Q38_PREFILL_TILE_WORDS==FG_VK_PREFILL_TILE_WORDS);
    for(uint32_t i=0;i<FG_Q38_PIPELINE_TRANSIENT_COUNT;i++){
        CHECK(layout.offsets[i]%FG_ALIGNMENT==0u);
        CHECK(layout.bytes[i]>0u&&
              layout.offsets[i]+layout.bytes[i]<=layout.total_bytes);
        for(uint32_t j=i+1u;j<FG_Q38_PIPELINE_TRANSIENT_COUNT;j++)
            CHECK(layout.offsets[i]+layout.bytes[i]<=layout.offsets[j]||
                  layout.offsets[j]+layout.bytes[j]<=layout.offsets[i]);
    }

    enum{TOKENS=2};
    CHECK(fg_q38_pipeline_owner_transient_layout_get(TOKENS,&layout));
    uint64_t ple_bytes=fg_qsa_ple_scratch_bytes(TOKENS);
    uint64_t residual_offset=(uint64_t)TOKENS*
        (6u*FG_Q38_HYPER_WIDTH+FG_HIDDEN_SIZE)*4u;
    uint64_t residual_bytes=(uint64_t)TOKENS*FG_Q38_HYPER_WIDTH*4u;
    uint8_t *ple=malloc((size_t)ple_bytes);
    uint8_t *transient=malloc((size_t)layout.total_bytes);
    CHECK(ple&&transient);
    if(ple&&transient){
        uintptr_t ple_begin=(uintptr_t)ple,ple_end=ple_begin+ple_bytes;
        uintptr_t transient_begin=(uintptr_t)transient;
        uintptr_t transient_end=transient_begin+layout.total_bytes;
        CHECK(ple_end<=transient_begin||transient_end<=ple_begin);
        memset(ple,0,(size_t)ple_bytes);
        memset(ple+residual_offset,0x5a,(size_t)residual_bytes);
        /* Simulate every MoE transient being overwritten before layer-1 GR write. */
        memset(transient,0xa5,(size_t)layout.total_bytes);
        bool intact=true;
        for(uint64_t i=0;i<residual_bytes;i++)
            if(ple[residual_offset+i]!=0x5a){intact=false;break;}
        CHECK(intact);
    }
    free(transient);free(ple);
}

static void test_qsa_scratch_geometry(void){
    CHECK(fg_qsa_attention_scratch_bytes(0u)==UINT64_MAX);
    CHECK(fg_qsa_attention_scratch_bytes(FG_PREFILL_MAX_TOKENS+1u)==UINT64_MAX);
    CHECK(fg_qsa_attention_scratch_bytes(256u)==37144576u);
    CHECK(fg_qsa_gdn_scratch_bytes(256u)==36274176u);
    CHECK(fg_gdn_pipeline_prefill_scratch_bytes(128u)==18137088u);
    CHECK(fg_gdn_pipeline_prefill_scratch_bytes(129u)==UINT64_MAX);
    CHECK(FG_VK_GDN_PIPELINE_PREFILL_MAX_TOKENS==128u);
    CHECK(FG_VK_GDN_PIPELINE_PREFILL_DISPATCHES==3u);
    CHECK(fg_qsa_ple_scratch_bytes(256u)==76021760u);
    CHECK(fg_qsa_attention_family_scratch_bytes(256u)==76021760u);
    CHECK(fg_qsa_attention_family_scratch_bytes(128u)==38010880u);
}
static void test_qsa_state(void){
    char path[128];snprintf(path,sizeof(path),"/tmp/fg-qsa-state-%ld.bin",(long)getpid());unlink(path);uint8_t layers[]={3};fg_error err={0};fg_qsa_state *state=NULL;fg_status status=fg_qsa_state_open(&state,path,layers,1,8,true,&err);if(status==FG_ERR_UNAVAILABLE){unlink(path);return;}CHECK(status==FG_OK);if(status!=FG_OK){unlink(path);return;}
    uint8_t records[FG_Q38_QSA_COMPRESS_RATIO][FG_Q38_QSA_TOKEN_RECORD_BYTES],got[FG_Q38_QSA_COMPRESS_RATIO][FG_Q38_QSA_TOKEN_RECORD_BYTES];for(uint32_t i=0;i<sizeof(records);i++)((uint8_t *)records)[i]=(uint8_t)(i*29u+7u);float key[FG_Q38_ATTN_KV_WIDTH],value[FG_Q38_ATTN_KV_WIDTH],index_key[FG_Q38_INDEX_WIDTH],decoded_key[FG_Q38_ATTN_KV_WIDTH],decoded_value[FG_Q38_ATTN_KV_WIDTH],decoded_index[FG_Q38_INDEX_WIDTH];uint32_t position[3]={17,23,31},decoded_position[3]={0};for(uint32_t i=0;i<FG_Q38_ATTN_KV_WIDTH;i++){key[i]=sinf((float)i*0.031f)*2.0f;value[i]=cosf((float)i*0.019f)*3.0f;}for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)index_key[i]=sinf((float)i*0.013f);fg_qsa_encode_full_token_record(key,value,index_key,position,records[0]);fg_qsa_decode_token_record(records[0],decoded_key,decoded_value);fg_qsa_decode_token_metadata(records[0],decoded_index,decoded_position);double key_error=0.0,value_error=0.0,index_error=0.0;for(uint32_t i=0;i<FG_Q38_ATTN_KV_WIDTH;i++){key_error+=fabs((double)key[i]-decoded_key[i]);value_error+=fabs((double)value[i]-decoded_value[i]);}for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)index_error+=fabs((double)index_key[i]-decoded_index[i]);CHECK(key_error/FG_Q38_ATTN_KV_WIDTH<0.02&&value_error/FG_Q38_ATTN_KV_WIDTH<0.2&&index_error/FG_Q38_INDEX_WIDTH<0.02);CHECK(memcmp(position,decoded_position,sizeof(position))==0);CHECK(fg_qsa_state_write_block(state,0,0,&records[0][0],3,&err)==FG_OK);CHECK(fg_qsa_state_layer_tokens(state,0)==3);uint32_t committed=0;CHECK(fg_qsa_state_read_block(state,0,0,&got[0][0],&committed,&err)==FG_OK);CHECK(committed==3&&memcmp(records,got,3u*FG_Q38_QSA_TOKEN_RECORD_BYTES)==0);for(uint32_t i=3u*FG_Q38_QSA_TOKEN_RECORD_BYTES;i<sizeof(got);i++)CHECK(((uint8_t *)got)[i]==0);fg_qsa_state_close(state);state=NULL;
    CHECK(fg_qsa_state_open(&state,path,layers,1,8,false,&err)==FG_OK);if(state){CHECK(fg_qsa_state_layer_tokens(state,0)==3);CHECK(fg_qsa_state_read_block(state,0,0,&got[0][0],&committed,&err)==FG_OK);CHECK(fg_qsa_state_reset(state,&err)==FG_OK);fg_qsa_state_close(state);state=NULL;}CHECK(fg_qsa_state_open(&state,path,layers,1,8,false,&err)==FG_OK);if(state){CHECK(fg_qsa_state_layer_tokens(state,0)==0);fg_qsa_state_close(state);state=NULL;}CHECK(truncate(path,4096)==0);CHECK(fg_qsa_state_open(&state,path,layers,1,8,false,&err)==FG_ERR_MISMATCH);unlink(path);
}
static void test_qsa_state_failed_create_cleanup(void){
    char path[128];snprintf(path,sizeof(path),"test-qsa-create-failure-%ld.bin",(long)getpid());
    unlink(path);pid_t child=fork();CHECK(child>=0);if(child==0){
        struct rlimit limit;if(getrlimit(RLIMIT_NOFILE,&limit)!=0)_exit(10);
        if(limit.rlim_cur>128u)limit.rlim_cur=128u;
        if(setrlimit(RLIMIT_NOFILE,&limit)!=0)_exit(11);
        int reserve[256];uint32_t opened=0;while(opened<256u){
            int fd=open("/dev/null",O_RDONLY);if(fd<0)break;reserve[opened++]=fd;
        }
        if(!opened)_exit(12);
        close(reserve[--opened]);
        uint8_t layers[]={3u};fg_qsa_state *state=NULL;fg_error err={0};
        fg_status status=fg_qsa_state_open(&state,path,layers,1u,8u,true,&err);
        int clean=status!=FG_OK&&!state&&access(path,F_OK)!=0&&
            err.code==status&&strstr(err.message,"io_uring_setup")!=NULL;
        fg_qsa_state_close(state);unlink(path);_exit(clean?0:13);
    }
    if(child>0){int status=0;CHECK(waitpid(child,&status,0)==child);CHECK(WIFEXITED(status)&&WEXITSTATUS(status)==0);}
    unlink(path);
}
static void test_qsa_state_batch(void){char path[128];snprintf(path,sizeof(path),"/tmp/fg-qsa-batch-%ld.bin",(long)getpid());unlink(path);uint8_t layers[]={3};fg_error err={0};fg_qsa_state *state=NULL;fg_status status=fg_qsa_state_open(&state,path,layers,1,8,true,&err);if(status==FG_ERR_UNAVAILABLE){unlink(path);return;}CHECK(status==FG_OK);if(status!=FG_OK){unlink(path);return;}uint8_t first[4u*FG_Q38_QSA_TOKEN_RECORD_BYTES],second[4u*FG_Q38_QSA_TOKEN_RECORD_BYTES],got[8u*FG_Q38_QSA_TOKEN_RECORD_BYTES];for(uint32_t i=0;i<sizeof(first);i++){first[i]=(uint8_t)(i*7u+3u);second[i]=(uint8_t)(i*11u+5u);}CHECK(fg_qsa_state_write_block(state,0,0,first,4,&err)==FG_OK);CHECK(fg_qsa_state_write_block(state,0,1,second,2,&err)==FG_OK);uint32_t blocks[]={1,0},committed[2]={0};CHECK(fg_qsa_state_read_blocks(state,0,blocks,2,got,committed,&err)==FG_OK);CHECK(committed[0]==2&&committed[1]==4);CHECK(memcmp(got,second,2u*FG_Q38_QSA_TOKEN_RECORD_BYTES)==0);CHECK(memcmp(got+4u*FG_Q38_QSA_TOKEN_RECORD_BYTES,first,sizeof(first))==0);fg_qsa_state_close(state);unlink(path);}

static void test_qsa_state_write_batch(void){
    char path[128];snprintf(path,sizeof(path),"test-qsa-write-batch-%ld.bin",(long)getpid());
    unlink(path);uint8_t layers[]={3};fg_error err={0};fg_qsa_state *state=NULL;
    fg_status status=fg_qsa_state_open(&state,path,layers,1u,16u,true,&err);
    if(status==FG_ERR_UNAVAILABLE){unlink(path);return;}
    CHECK(status==FG_OK);if(status!=FG_OK){unlink(path);return;}
    uint32_t blocks[]={0u,1u,2u},read_blocks[]={2u,0u,1u},committed[3]={0};
    uint8_t records[3u*FG_QSA_PAGE_RECORD_BYTES],got[3u*FG_QSA_PAGE_RECORD_BYTES];
    for(uint32_t i=0;i<sizeof(records);i++)records[i]=(uint8_t)(i*13u+9u);
    CHECK(fg_qsa_state_write_blocks(state,0u,blocks,3u,records,&err)==FG_OK);
    CHECK(fg_qsa_state_layer_tokens(state,0u)==12u);
    CHECK(fg_qsa_state_read_blocks(state,0u,read_blocks,3u,got,committed,&err)==FG_OK);
    CHECK(committed[0]==4u&&committed[1]==4u&&committed[2]==4u);
    CHECK(memcmp(got,records+2u*FG_QSA_PAGE_RECORD_BYTES,FG_QSA_PAGE_RECORD_BYTES)==0);
    CHECK(memcmp(got+FG_QSA_PAGE_RECORD_BYTES,records,FG_QSA_PAGE_RECORD_BYTES)==0);
    CHECK(memcmp(got+2u*FG_QSA_PAGE_RECORD_BYTES,records+FG_QSA_PAGE_RECORD_BYTES,
                 FG_QSA_PAGE_RECORD_BYTES)==0);
    fg_qsa_state_close(state);unlink(path);
}

typedef struct replica_probe {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    uint32_t calls;
    bool entered,release,fail;
} replica_probe;

static fg_status replica_probe_send(void *opaque,uint32_t owner,uint64_t session_id,
                                    uint32_t batch_id,const void *payload,uint32_t bytes,
                                    fg_error *err){
    replica_probe *probe=opaque;
    CHECK((owner==3u||owner==7u)&&session_id==99u&&batch_id<2u&&payload&&bytes==1u);
    pthread_mutex_lock(&probe->mutex);probe->calls++;probe->entered=true;
    pthread_cond_broadcast(&probe->ready);
    while(!probe->release)pthread_cond_wait(&probe->ready,&probe->mutex);
    bool fail=probe->fail;pthread_mutex_unlock(&probe->mutex);
    if(fail){fg_error_set(err,FG_ERR_IO,"injected replica send failure");return FG_ERR_IO;}
    return FG_OK;
}

static void test_qsa_replica_queue(void){
    replica_probe probe={0};CHECK(pthread_mutex_init(&probe.mutex,NULL)==0);
    CHECK(pthread_cond_init(&probe.ready,NULL)==0);fg_error err={0};
    fg_qsa_replica *replica=NULL;
    CHECK(fg_qsa_replica_create(&replica,replica_probe_send,&probe,&err)==FG_OK);
    uint8_t *buffers[2]={0};
    CHECK(fg_qsa_replica_reserve(replica,2u,buffers,&err)==FG_OK);
    buffers[0][0]=1u;buffers[1][0]=2u;
    fg_qsa_replica_item items[2]={
        {.owner=3u,.batch_id=0u,.bytes=1u,.session_id=99u},
        {.owner=7u,.batch_id=1u,.bytes=1u,.session_id=99u}
    };
    CHECK(fg_qsa_replica_commit(replica,items,2u,&err)==FG_OK);
    pthread_mutex_lock(&probe.mutex);
    while(!probe.entered)pthread_cond_wait(&probe.ready,&probe.mutex);
    pthread_mutex_unlock(&probe.mutex);
    CHECK(fg_qsa_replica_reserve(replica,1u,buffers,&err)==FG_ERR_LIMIT);
    pthread_mutex_lock(&probe.mutex);probe.release=true;
    pthread_cond_broadcast(&probe.ready);pthread_mutex_unlock(&probe.mutex);
    CHECK(fg_qsa_replica_drain(replica,&err)==FG_OK);CHECK(probe.calls==2u);
    fg_qsa_replica_destroy(replica);replica=NULL;
    pthread_cond_destroy(&probe.ready);pthread_mutex_destroy(&probe.mutex);
    probe=(replica_probe){0};CHECK(pthread_mutex_init(&probe.mutex,NULL)==0);
    CHECK(pthread_cond_init(&probe.ready,NULL)==0);probe.release=true;probe.fail=true;
    CHECK(fg_qsa_replica_create(&replica,replica_probe_send,&probe,&err)==FG_OK);
    CHECK(fg_qsa_replica_reserve(replica,1u,buffers,&err)==FG_OK);
    buffers[0][0]=3u;items[0]=(fg_qsa_replica_item){
        .owner=3u,.batch_id=0u,.bytes=1u,.session_id=99u};
    CHECK(fg_qsa_replica_commit(replica,items,1u,&err)==FG_OK);
    CHECK(fg_qsa_replica_drain(replica,&err)==FG_ERR_IO);
    CHECK(fg_qsa_replica_status(replica,&err)==FG_ERR_IO);
    fg_qsa_replica_destroy(replica);
    pthread_cond_destroy(&probe.ready);pthread_mutex_destroy(&probe.mutex);
    probe=(replica_probe){0};CHECK(pthread_mutex_init(&probe.mutex,NULL)==0);
    CHECK(pthread_cond_init(&probe.ready,NULL)==0);probe.fail=true;
    CHECK(fg_qsa_replica_create(&replica,replica_probe_send,&probe,&err)==FG_OK);
    CHECK(fg_qsa_replica_reserve(replica,1u,buffers,&err)==FG_OK);
    buffers[0][0]=4u;items[0]=(fg_qsa_replica_item){
        .owner=3u,.batch_id=0u,.bytes=1u,.session_id=99u};
    CHECK(fg_qsa_replica_commit(replica,items,1u,&err)==FG_OK);
    pthread_mutex_lock(&probe.mutex);
    while(!probe.entered)pthread_cond_wait(&probe.ready,&probe.mutex);
    pthread_mutex_unlock(&probe.mutex);
    CHECK(fg_qsa_replica_reserve(replica,1u,buffers,&err)==FG_OK);
    buffers[0][0]=5u;items[0]=(fg_qsa_replica_item){
        .owner=7u,.batch_id=1u,.bytes=1u,.session_id=99u};
    pthread_mutex_lock(&probe.mutex);probe.release=true;
    pthread_cond_broadcast(&probe.ready);pthread_mutex_unlock(&probe.mutex);
    while(fg_qsa_replica_status(replica,&err)==FG_OK)usleep(1000u);
    CHECK(fg_qsa_replica_commit(replica,items,1u,&err)==FG_ERR_IO);
    CHECK(fg_qsa_replica_drain(replica,&err)==FG_ERR_IO);CHECK(probe.calls==1u);
    fg_qsa_replica_destroy(replica);
    pthread_cond_destroy(&probe.ready);pthread_mutex_destroy(&probe.mutex);
}

static void test_lazy_qsa_clear_barrier(void){
    uint64_t session_id=1u;
    bool state_ready=false;
    fg_error error={0};
    fg_qsa_replica *replica=NULL;
    CHECK(session_id&&fg_qsa_replica_drain_if_present(replica,&error)==FG_OK);
    state_ready=true;
    CHECK(state_ready&&session_id==1u&&error.code==FG_OK&&error.message[0]==0);
}

static void test_prefill_storage_geometry(void){
    _Static_assert(sizeof(fg_prefill_pair)==FG_PREFILL_PAIR_BYTES,
                   "prefill pair storage must match wire geometry");
    _Static_assert(sizeof(fg_prefill_result_pair)==4u,
                   "prefill result pair storage must remain four bytes");
    _Static_assert(sizeof(fg_prefill_pair)%_Alignof(fg_prefill_result_pair)==0u,
                   "combined prefill storage must align result pairs");
    enum{PAIRS=FG_TOP_K*4u};
    size_t pair_bytes=(size_t)PAIRS*sizeof(fg_prefill_pair);
    size_t result_bytes=(size_t)PAIRS*sizeof(fg_prefill_result_pair);
    uint8_t *arena=malloc(pair_bytes+result_bytes);
    CHECK(arena!=NULL);
    if(!arena)return;
    fg_prefill_pair *pairs=(fg_prefill_pair *)arena;
    fg_prefill_result_pair *result_pairs=
        (fg_prefill_result_pair *)(arena+pair_bytes);
    CHECK((uint8_t *)result_pairs>=arena+pair_bytes);
    CHECK((uint8_t *)result_pairs+result_bytes<=
          arena+pair_bytes+result_bytes);
    pairs[0]=(fg_prefill_pair){.token_slot=1u,.expert_id=2u,
        .routing_slot=3u,.gate=0.5f};
    result_pairs[0]=(fg_prefill_result_pair){.token_slot=1u,.routing_slot=3u};
    CHECK(pairs[0].expert_id==2u&&pairs[0].gate==0.5f&&
          result_pairs[0].token_slot==1u&&result_pairs[0].routing_slot==3u);
    free(arena);
}

static void test_qsa_page_cache(void){
    fg_error err={0};fg_qsa_page_cache *cache=NULL;
    CHECK(fg_qsa_page_cache_create(&cache,2u,&err)==FG_OK);if(!cache)return;
    uint32_t slot1=0,slot2=0,slot3=0,lookup=0;bool hit=false;
    CHECK(fg_qsa_page_cache_acquire(cache,3u,1u,&slot1,&hit,&err)==FG_OK&&!hit);
    CHECK(fg_qsa_page_cache_acquire(cache,3u,2u,&slot2,&hit,&err)==FG_OK&&!hit);
    CHECK(slot1!=slot2&&fg_qsa_page_cache_lookup(cache,3u,1u,&lookup)&&lookup==slot1);
    CHECK(fg_qsa_page_cache_acquire(cache,3u,3u,&slot3,&hit,&err)==FG_OK&&!hit);
    CHECK(!fg_qsa_page_cache_lookup(cache,3u,2u,&lookup));
    CHECK(fg_qsa_page_cache_lookup(cache,3u,1u,&lookup)&&lookup==slot1);
    CHECK(fg_qsa_page_cache_lookup(cache,3u,3u,&lookup)&&lookup==slot3);
    uint32_t missing[]={0u,1u},fetch[4]={0},fetch_count=0;
    CHECK(fg_qsa_page_cache_plan_fetch(cache,3u,missing,2u,6u,24u,fetch,4u,
                                       &fetch_count,&err)==FG_OK);
    CHECK(fetch_count==3u&&fetch[0]==0u&&fetch[1]==1u&&fetch[2]==2u);
    missing[0]=2u;
    CHECK(fg_qsa_page_cache_plan_fetch(cache,3u,missing,1u,6u,12u,fetch,4u,
                                       &fetch_count,&err)==FG_OK);
    CHECK(fetch_count==1u&&fetch[0]==2u);
    fg_qsa_page_cache_reset(cache);
    CHECK(!fg_qsa_page_cache_lookup(cache,3u,1u,&lookup));
    CHECK(fg_qsa_page_cache_acquire(cache,7u,4u,&slot1,&hit,&err)==FG_OK&&!hit);
    CHECK(fg_qsa_page_cache_lookup(cache,7u,4u,&lookup)&&lookup==slot1);
    fg_qsa_page_cache_reset(cache);
    CHECK(fg_qsa_page_cache_acquire(cache,3u,1u,&slot1,&hit,&err)==FG_OK);
    CHECK(fg_qsa_page_cache_acquire(cache,3u,2u,&slot2,&hit,&err)==FG_OK);
    CHECK(fg_qsa_page_cache_pin(cache,3u,1u,&err)==FG_OK);
    CHECK(fg_qsa_page_cache_acquire(cache,3u,3u,&slot3,&hit,&err)==FG_OK);
    CHECK(fg_qsa_page_cache_lookup(cache,3u,1u,&lookup));
    CHECK(!fg_qsa_page_cache_lookup(cache,3u,2u,&lookup));
    fg_qsa_page_cache_unpin(cache,3u,1u);
    fg_qsa_page_cache_destroy(cache);
    cache=NULL;CHECK(fg_qsa_page_cache_create(&cache,UINT32_MAX,&err)==FG_ERR_LIMIT&&!cache);
}
static void put_u32(FILE *f,uint32_t v){CHECK(fwrite(&v,1,4,f)==4);}static void put_u64(FILE *f,uint64_t v){CHECK(fwrite(&v,1,8,f)==8);}static void put_string(FILE *f,const char *s){uint64_t n=strlen(s);put_u64(f,n);CHECK(fwrite(s,1,(size_t)n,f)==n);}
static void test_pack(void){char source[128],dir[128],manifest_path[160];snprintf(source,sizeof(source),"/tmp/fg-test-%ld.gguf",(long)getpid());snprintf(dir,sizeof(dir),"/tmp/fg-test-%ld-pack",(long)getpid());snprintf(manifest_path,sizeof(manifest_path),"%s/manifest.fgm",dir);FILE *f=fopen(source,"wb");CHECK(f!=NULL);if(!f)return;put_u32(f,0x46554747u);put_u32(f,3);put_u64(f,2);put_u64(f,0);put_string(f,"token_embd.weight");put_u32(f,1);put_u64(f,1);put_u32(f,0);put_u64(f,0);put_string(f,"blk.0.ffn_gate_exps.weight");put_u32(f,2);put_u64(f,1);put_u64(f,512);put_u32(f,0);put_u64(f,32);while((ftell(f)&31)!=0)fputc(0,f);float one=1.0f;CHECK(fwrite(&one,1,4,f)==4);for(unsigned i=4;i<32;i++)fputc(0,f);for(unsigned i=0;i<512;i++){float v=(float)i;CHECK(fwrite(&v,1,4,f)==4);}fclose(f);const char *sources[]={source};fg_pack_options o={.output_dir=dir,.source_paths=sources,.source_count=1,.skip_model_validation=true};fg_error err={0};fg_status pack_rc=fg_pack_run(&o,&err);if(pack_rc!=FG_OK)fprintf(stderr,"pack error: %s\n",err.message);CHECK(pack_rc==FG_OK);fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(m){CHECK(fg_manifest_read(manifest_path,m,&err)==FG_OK);for(uint32_t r=0;r<FG_RANK_COUNT;r++)CHECK(m->ranks[r].scratch_bytes==fg_q38_runtime_scratch_bytes(r,m->prefill_microbatch,m->prefill_window,m->max_context));CHECK(m->tensor_count==5);CHECK(m->tensors[0].dims==1&&m->tensors[0].shape[0]==1&&m->tensors[0].kind==FG_TENSOR_COMMON&&m->tensors[0].layout==FG_TENSOR_LAYOUT_GGML);char external_path[160];struct stat external_info;snprintf(external_path,sizeof(external_path),"%s/%s",dir,FG_TOKEN_EMBEDDING_ARTIFACT);CHECK(stat(external_path,&external_info)!=0&&errno==ENOENT);for(uint32_t i=1;i<5;i++){uint32_t rank=m->layer_groups[0][i-1u];char expected[FG_TENSOR_NAME_MAX];snprintf(expected,sizeof(expected),"blk.0.ffn_gate_exps.weight.rank%u",rank);CHECK(m->tensors[i].dims==2&&m->tensors[i].shape[1]==128);CHECK(!strcmp(m->tensors[i].name,expected)&&m->tensors[i].rank==rank&&m->tensors[i].bytes==512u);}for(uint32_t r=0;r<4;r++)CHECK(m->ranks[m->layer_groups[0][r]].tensor_count>=1);char rank_path[160];snprintf(rank_path,sizeof(rank_path),"%s/rank-00.fgw",dir);struct stat st;CHECK(stat(rank_path,&st)==0);void *arena=NULL;CHECK(posix_memalign(&arena,FG_ALIGNMENT,(size_t)st.st_size)==0);FILE *rf=fopen(rank_path,"rb");CHECK(rf!=NULL);if(arena&&rf){CHECK(fread(arena,1,(size_t)st.st_size,rf)==(size_t)st.st_size);fclose(rf);CHECK(fg_verify_rank_arena(m,0,arena,(uint64_t)st.st_size,&err)==FG_OK);((uint8_t *)arena)[m->tensors[0].offset]^=1;CHECK(fg_verify_rank_arena(m,0,arena,(uint64_t)st.st_size,&err)==FG_ERR_MISMATCH);}free(arena);uint64_t sealed_scratch=m->ranks[1].scratch_bytes;m->ranks[1].scratch_bytes=sealed_scratch-1u;CHECK(fg_manifest_write(manifest_path,m,&err)==FG_OK);fg_manifest *understated=malloc(sizeof(*understated));CHECK(understated!=NULL);if(understated){CHECK(fg_manifest_read(manifest_path,understated,&err)==FG_ERR_LIMIT);free(understated);}m->ranks[1].scratch_bytes=sealed_scratch;CHECK(fg_manifest_write(manifest_path,m,&err)==FG_OK);free(m);}f=fopen(manifest_path,"r+b");CHECK(f!=NULL);if(f){CHECK(fseek(f,(long)offsetof(fg_manifest,source_sha256),SEEK_SET)==0);int c=fgetc(f);CHECK(c!=EOF);CHECK(fseek(f,-1,SEEK_CUR)==0);fputc(c^1,f);fclose(f);m=malloc(sizeof(*m));if(m){CHECK(fg_manifest_read(manifest_path,m,&err)==FG_ERR_MISMATCH);free(m);}}unlink(manifest_path);char p[160];for(unsigned r=0;r<8;r++){snprintf(p,sizeof(p),"%s/rank-%02u.fgw",dir,r);unlink(p);}snprintf(p,sizeof(p),"%s/ngram.iq4nl",dir);unlink(p);rmdir(dir);unlink(source);}

static const fg_tensor_record *manifest_record(const fg_manifest *manifest,const char *name){
    for(uint32_t i=0;i<manifest->tensor_count;i++)
        if(!strcmp(manifest->tensors[i].name,name))return &manifest->tensors[i];
    return NULL;
}

static void test_pipeline_pack(void){
    char source[128],directory[128],manifest_path[160],path[160];
    long pid=(long)getpid();
    snprintf(source,sizeof(source),"test-pipeline-pack-%ld.gguf",pid);
    snprintf(directory,sizeof(directory),"test-pipeline-pack-%ld",pid);
    snprintf(manifest_path,sizeof(manifest_path),"%s/manifest.fgm",directory);
    FILE *file=fopen(source,"wb");CHECK(file!=NULL);if(!file)return;
    put_u32(file,0x46554747u);put_u32(file,3u);put_u64(file,4u);put_u64(file,0u);
    put_string(file,"token_embd.weight");put_u32(file,2u);put_u64(file,32u);put_u64(file,1u);put_u32(file,8u);put_u64(file,0u);
    put_string(file,"output.weight");put_u32(file,1u);put_u64(file,1u);put_u32(file,0u);put_u64(file,64u);
    put_string(file,"blk.6.probe.weight");put_u32(file,1u);put_u64(file,1u);put_u32(file,0u);put_u64(file,96u);
    put_string(file,"blk.6.ffn_gate_exps.weight");put_u32(file,2u);put_u64(file,1u);put_u64(file,FG_EXPERT_COUNT);put_u32(file,0u);put_u64(file,128u);
    while((ftell(file)&31)!=0)fputc(0,file);
    long data_start=ftell(file);for(uint32_t i=0;i<FG_Q38_Q8_0_BLOCK_BYTES;i++)fputc((int)(i*7u+3u),file);
    while((uint64_t)(ftell(file)-data_start)<64u)fputc(0,file);
    float value=2.0f;CHECK(fwrite(&value,1,4u,file)==4u);for(uint32_t i=4u;i<32u;i++)fputc(0,file);
    value=3.0f;CHECK(fwrite(&value,1,4u,file)==4u);for(uint32_t i=4u;i<32u;i++)fputc(0,file);
    for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++){value=(float)expert;CHECK(fwrite(&value,1,4u,file)==4u);}
    fclose(file);
    const char *sources[]={source};fg_error error={0};
    fg_pack_options options={.output_dir=directory,.source_paths=sources,.source_count=1u,
        .runtime_profile=FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,
        .skip_model_validation=true};
    CHECK(fg_pack_run(&options,&error)==FG_OK);
    fg_manifest *manifest=malloc(sizeof(*manifest));CHECK(manifest!=NULL);
    if(manifest){
        fg_status read_status=fg_manifest_read(manifest_path,manifest,&error);
        if(read_status!=FG_OK)fprintf(stderr,"pipeline manifest read: %s\n",error.message);
        CHECK(read_status==FG_OK);
        CHECK(manifest->execution_mode==FG_EXECUTION_PIPELINE&&manifest->tensor_count==4u);
        const fg_tensor_record *embedding=manifest_record(manifest,"token_embd.weight");
        const fg_tensor_record *output=manifest_record(manifest,"output.weight");
        const fg_tensor_record *common=manifest_record(manifest,"blk.6.probe.weight");
        const fg_tensor_record *expert=manifest_record(manifest,"blk.6.ffn_gate_exps.weight");
        CHECK(embedding&&embedding->rank==manifest->stage_ranks[0]&&
              embedding->kind==FG_TENSOR_HOST_CACHE&&
              embedding->layout==FG_TENSOR_LAYOUT_HOST_Q8_0&&
              embedding->bytes==FG_Q38_Q8_0_BLOCK_BYTES);
        CHECK(embedding&&manifest->host_resident_bytes[manifest->stage_ranks[0]]==
              embedding->bytes);
        CHECK(manifest->ranks[manifest->stage_ranks[0]].tensor_count==0u&&
              manifest->ranks[manifest->stage_ranks[0]].persistent_bytes==0u);
        CHECK(output&&output->rank==manifest->stage_ranks[manifest->stage_count-1u]);
        CHECK(common&&common->rank==manifest->layer_owner[6]);
        CHECK(expert&&expert->rank==manifest->layer_owner[6]&&
              expert->bytes==(uint64_t)FG_EXPERT_COUNT*4u&&
              expert->shape[1]==FG_EXPERT_COUNT&&!strstr(expert->name,".rank"));
        CHECK(manifest->ranks[manifest->layer_owner[6]].tensor_count==2u);
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            CHECK(manifest->ranks[rank].scratch_bytes==
                  fg_q38_runtime_scratch_bytes_for_manifest(
                      manifest,rank,manifest->prefill_microbatch,
                      manifest->prefill_window,manifest->max_context));
            snprintf(path,sizeof(path),"%s/rank-%02u.fgw",directory,rank);
            struct stat info;CHECK(stat(path,&info)==0);
            CHECK((uint64_t)info.st_size==manifest->ranks[rank].persistent_bytes);
        }
        fg_verify_options verify={.manifest_path=manifest_path,.pack_dir=directory,
            .source_paths=sources,.source_count=1u};
        fg_status verify_status=fg_pack_verify(&verify,&error);
        if(verify_status!=FG_OK)fprintf(stderr,"pipeline pack verify: %s\n",error.message);
        CHECK(verify_status==FG_OK);
        snprintf(path,sizeof(path),"%s/%s",directory,FG_TOKEN_EMBEDDING_ARTIFACT);
        struct stat embedding_info;CHECK(stat(path,&embedding_info)==0&&
            (uint64_t)embedding_info.st_size==embedding->bytes);
        uint8_t embedding_digest[32];CHECK(fg_sha256_file(path,embedding_digest,&error)==FG_OK&&
            !memcmp(embedding_digest,embedding->sha256,sizeof(embedding_digest)));
        CHECK(unlink(path)==0);
        CHECK(fg_pack_verify(&verify,&error)==FG_ERR_MISMATCH);
        FILE *external=fopen(path,"wb");CHECK(external!=NULL);
        if(external){for(uint32_t i=0;i<FG_Q38_Q8_0_BLOCK_BYTES;i++)fputc((int)(i*7u+3u),external);CHECK(fclose(external)==0);}
        external=fopen(path,"r+b");CHECK(external!=NULL);
        if(external){CHECK(fputc(0xff,external)!=EOF);CHECK(fclose(external)==0);}
        CHECK(fg_pack_verify(&verify,&error)==FG_ERR_MISMATCH);
        free(manifest);
    }
    options.router_profile_path="unused-router-profile";
    CHECK(fg_pack_run(&options,&error)==FG_ERR_UNAVAILABLE);
    options.router_profile_path=NULL;options.expert_map_path="unused-expert-map";
    CHECK(fg_pack_run(&options,&error)==FG_ERR_UNAVAILABLE);
    unlink(manifest_path);
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){snprintf(path,sizeof(path),"%s/rank-%02u.fgw",directory,rank);unlink(path);}
    for(uint32_t rank=1;rank<FG_RANK_COUNT;rank++){snprintf(path,sizeof(path),"%s/" FG_NGRAM_SHARD_ARTIFACT_FORMAT,directory,rank);unlink(path);}
    snprintf(path,sizeof(path),"%s/ngram.iq4nl",directory);unlink(path);
    snprintf(path,sizeof(path),"%s/%s",directory,FG_TOKEN_EMBEDDING_ARTIFACT);unlink(path);
    rmdir(directory);unlink(source);
}

static void test_pipeline_spill_admission(void){
    char source[128];snprintf(source,sizeof(source),
                              "test-pipeline-spill-%ld.gguf",(long)getpid());
    FILE *file=fopen(source,"wb");CHECK(file!=NULL);if(!file)return;
    const uint64_t token_bytes=(uint64_t)FG_Q38_VOCAB_SIZE*
        (FG_HIDDEN_SIZE/FG_QK8_0)*FG_Q38_Q8_0_BLOCK_BYTES;
    const uint64_t rank_bytes=(FG_PERSISTENT_CAP_BYTES-2u*FG_ALIGNMENT)&~3ull;
    const uint64_t common_offset=fg_align_up_u64(token_bytes,32u);
    put_u32(file,0x46554747u);put_u32(file,3u);put_u64(file,2u);put_u64(file,0u);
    put_string(file,"token_embd.weight");put_u32(file,2u);
    put_u64(file,FG_HIDDEN_SIZE);put_u64(file,FG_Q38_VOCAB_SIZE);
    put_u32(file,8u);put_u64(file,0u);
    put_string(file,"blk.0.probe.weight");put_u32(file,1u);
    put_u64(file,rank_bytes/4u);put_u32(file,0u);put_u64(file,common_offset);
    while((ftell(file)&31)!=0)fputc(0,file);
    uint64_t data_start=(uint64_t)ftell(file);
    CHECK(ftruncate(fileno(file),(off_t)(data_start+common_offset+rank_bytes))==0);
    CHECK(fclose(file)==0);
    CHECK(fg_align_up_u64(token_bytes,FG_ALIGNMENT)+
          fg_align_up_u64(rank_bytes,FG_ALIGNMENT)>FG_PERSISTENT_CAP_BYTES);
    const char *sources[]={source};fg_pack_options options={
        .output_dir="unused-pipeline-spill-output",.source_paths=sources,
        .source_count=1u,.runtime_profile=FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,
        .dry_run=true,.skip_model_validation=true
    };fg_error error={0};
    CHECK(fg_pack_run(&options,&error)==FG_OK);
    unlink(source);
}

static void test_q38_math(void){
    uint8_t q5[24]={0};float x[32];
    q5[0]=0x00;q5[1]=0x3c;q5[2]=0x00;q5[3]=0x38;
    for(uint32_t i=0;i<16;i++)q5[8+i]=(uint8_t)(i|((15u-i)<<4u));
    q5[4]=0x01;
    float expected=0.0f;for(uint32_t i=0;i<32;i++){x[i]=(float)(i+1);uint32_t lo=i<16?i:31u-i;uint32_t hi=i==0?16u:0u;expected=fmaf((float)(lo+hi)+0.5f,x[i],expected);}
    CHECK(fabsf(fg_q5_1_dot_f32(q5,x,32)-expected)<1e-3f);
    float norm_in[]={3,4,0,0,1,1,1,1},norm_w[]={1,1,1,1,1,1,1,1},norm_out[8];
    fg_q38_group_rms_norm(norm_out,norm_in,norm_w,2,4,0.0f);
    CHECK(fabsf(norm_out[0]-1.2f)<1e-6f&&fabsf(norm_out[1]-1.6f)<1e-6f);
    CHECK(fabsf(norm_out[4]-1.0f)<1e-6f&&fabsf(norm_out[7]-1.0f)<1e-6f);
    float logits[]={1,3,3,0};uint32_t ids[2];float gates[2];fg_error err={0};
    CHECK(fg_q38_router_topk(logits,4,2,ids,gates,&err)==FG_OK);
    CHECK(ids[0]==1&&ids[1]==2&&fabsf(gates[0]-0.5f)<1e-6f&&fabsf(gates[1]-0.5f)<1e-6f);
    float rope[128],rope_weight[128];for(uint32_t i=0;i<128u;i++){rope[i]=sinf((float)i*0.071f)+0.03f*cosf((float)i*0.013f);rope_weight[i]=1.0f+0.1f*cosf((float)i*0.027f);}uint32_t media_position[3]={17,23,31};CHECK(fg_q38_rms_mrope(rope,1,128,rope_weight,media_position,&err)==FG_OK);const uint32_t probe_index[]={0,1,2,31,32,33,34,63,64,127};const float probe_value[]={1.1080422f,-1.02584225f,1.04547368f,1.23273355f,-0.364600268f,0.407246308f,0.0768160129f,-1.29797908f,-1.31442412f,0.492792885f};for(uint32_t i=0;i<10u;i++)CHECK(fabsf(rope[probe_index[i]]-probe_value[i])<3e-5f);
    enum{INDEX_TOKENS=13};float query[FG_Q38_INDEX_QUERY_WIDTH],raw[INDEX_TOKENS*FG_Q38_INDEX_WIDTH],qnorm[FG_Q38_INDEX_WIDTH],knorm[FG_Q38_INDEX_WIDTH];for(uint32_t i=0;i<FG_Q38_INDEX_QUERY_WIDTH;i++)query[i]=sinf((float)(i+3u)*0.019f)+0.2f*cosf((float)i*0.007f);for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++){qnorm[i]=0.03f*sinf((float)i*0.051f);knorm[i]=0.04f*cosf((float)i*0.037f);}for(uint32_t token=0;token<INDEX_TOKENS;token++)for(uint32_t i=0;i<FG_Q38_INDEX_WIDTH;i++)raw[token*FG_Q38_INDEX_WIDTH+i]=sinf((float)(token+1u)*(float)(i+1u)*0.0017f)+0.15f*cosf((float)(token+2u)*(float)(i+3u)*0.0009f);uint32_t selected[INDEX_TOKENS],selected_count=0;const uint32_t selected_golden[]={8,9,10,11,4,5,6,7,0,1,2,3,12};CHECK(fg_q38_qsa_index_select_reference(query,raw,INDEX_TOKENS,qnorm,knorm,selected,INDEX_TOKENS,&selected_count,&err)==FG_OK);CHECK(selected_count==INDEX_TOKENS&&memcmp(selected,selected_golden,sizeof(selected_golden))==0);
}

static void test_cooked_q8(void){enum{WIDTH=320,ROWS=2,BLOCKS=WIDTH/FG_QK8_0,TILE_BYTES=FG_Q8_0_COOK_ROWS*BLOCKS*FG_Q8_0_BLOCK_BYTES,QUANT_OFFSET=FG_Q8_0_COOK_ROWS*BLOCKS*2u};uint8_t packed[ROWS*BLOCKS*FG_Q8_0_BLOCK_BYTES],cooked[TILE_BYTES];memset(packed,0,sizeof(packed));for(uint32_t row=0;row<ROWS;row++)for(uint32_t block=0;block<BLOCKS;block++){uint8_t *source=packed+((uint64_t)row*BLOCKS+block)*FG_Q8_0_BLOCK_BYTES;uint16_t scale=fg_f32_to_f16(0.25f+(float)(row*BLOCKS+block)*0.03125f);memcpy(source,&scale,sizeof(scale));for(uint32_t i=0;i<FG_QK8_0;i++)source[2u+i]=(uint8_t)(row*73u+block*31u+i*7u);}CHECK(fg_q8_0_cooked_tile_bytes(WIDTH)==TILE_BYTES);CHECK(fg_q8_0_cooked_matrix_bytes(WIDTH,ROWS)==TILE_BYTES);CHECK(fg_cook_q8_0_rows(packed,cooked,sizeof(cooked),WIDTH,ROWS));for(uint32_t row=0;row<ROWS;row++)for(uint32_t block=0;block<BLOCKS;block++){const uint8_t *source=packed+((uint64_t)row*BLOCKS+block)*FG_Q8_0_BLOCK_BYTES;CHECK(memcmp(cooked+((uint64_t)block*FG_Q8_0_COOK_ROWS+row)*2u,source,2u)==0);CHECK(memcmp(cooked+QUANT_OFFSET+((uint64_t)row*BLOCKS+block)*FG_QK8_0,source+2u,FG_QK8_0)==0);}for(uint32_t block=0;block<BLOCKS;block++)for(uint32_t row=ROWS;row<FG_Q8_0_COOK_ROWS;row++)CHECK(cooked[((uint64_t)block*FG_Q8_0_COOK_ROWS+row)*2u]==0u);CHECK(!fg_cook_q8_0_rows(packed,cooked,sizeof(cooked)-1u,WIDTH,ROWS));}

static void test_pack_cooked_q8(void){
    enum{WIDTH=320,ROWS=2560,BLOCKS=WIDTH/FG_QK8_0,SOURCE_ROW=BLOCKS*FG_Q8_0_BLOCK_BYTES,COOKED_TILE=FG_Q8_0_COOK_ROWS*SOURCE_ROW};
    char source[128],directory[128],manifest_path[160],rank_path[160],path[160];
    snprintf(source,sizeof(source),"/tmp/fg-cooked-%ld.gguf",(long)getpid());
    snprintf(directory,sizeof(directory),"/tmp/fg-cooked-%ld-pack",(long)getpid());
    snprintf(manifest_path,sizeof(manifest_path),"%s/manifest.fgm",directory);
    snprintf(rank_path,sizeof(rank_path),"%s/rank-00.fgw",directory);
    FILE *file=fopen(source,"wb");CHECK(file!=NULL);if(!file)return;
    put_u32(file,0x46554747u);put_u32(file,3u);put_u64(file,1u);put_u64(file,0u);
    put_string(file,"probe_q8.weight");put_u32(file,2u);put_u64(file,WIDTH);put_u64(file,ROWS);put_u32(file,8u);put_u64(file,0u);
    while((ftell(file)&31)!=0)fputc(0,file);
    uint8_t packed[SOURCE_ROW],first_packed[FG_Q8_0_COOK_ROWS*SOURCE_ROW];
    for(uint32_t row=0;row<ROWS;row++){
        for(uint32_t block=0;block<BLOCKS;block++){uint8_t *value=packed+block*FG_Q8_0_BLOCK_BYTES;uint16_t scale=fg_f32_to_f16(0.001f*(float)(1u+(row+block)%31u));memcpy(value,&scale,sizeof(scale));for(uint32_t i=0;i<FG_QK8_0;i++)value[2u+i]=(uint8_t)(row*19u+block*23u+i*29u);}
        if(row<FG_Q8_0_COOK_ROWS)memcpy(first_packed+(uint64_t)row*SOURCE_ROW,packed,sizeof(packed));
        CHECK(fwrite(packed,1,sizeof(packed),file)==sizeof(packed));
    }
    fclose(file);
    const char *sources[]={source};fg_pack_options options={.output_dir=directory,.source_paths=sources,.source_count=1u,.skip_model_validation=true};fg_error error={0};
    CHECK(fg_pack_run(&options,&error)==FG_OK);
    fg_manifest *manifest=malloc(sizeof(*manifest));CHECK(manifest!=NULL);
    if(manifest){
        CHECK(fg_manifest_read(manifest_path,manifest,&error)==FG_OK);
        const fg_tensor_record *record=NULL;for(uint32_t i=0;i<manifest->tensor_count;i++)if(strcmp(manifest->tensors[i].name,"probe_q8.weight")==0)record=&manifest->tensors[i];
        CHECK(record!=NULL);
        if(record){
            CHECK(record->layout==FG_TENSOR_LAYOUT_Q8_0_COOKED&&record->ggml_type==8u&&record->bytes==(uint64_t)ROWS*SOURCE_ROW);
            uint8_t expected[COOKED_TILE],actual[COOKED_TILE];CHECK(fg_cook_q8_0_rows(first_packed,expected,sizeof(expected),WIDTH,FG_Q8_0_COOK_ROWS));
            FILE *rank=fopen(rank_path,"rb");CHECK(rank!=NULL);if(rank){CHECK(fseeko(rank,(off_t)record->offset,SEEK_SET)==0);CHECK(fread(actual,1,sizeof(actual),rank)==sizeof(actual));CHECK(memcmp(actual,expected,sizeof(actual))==0);fclose(rank);}
        }
        fg_verify_options verify={.manifest_path=manifest_path,.pack_dir=directory,.source_paths=sources,.source_count=1u};CHECK(fg_pack_verify(&verify,&error)==FG_OK);
        free(manifest);
    }
    unlink(manifest_path);for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){snprintf(path,sizeof(path),"%s/rank-%02u.fgw",directory,rank);unlink(path);}snprintf(path,sizeof(path),"%s/ngram.iq4nl",directory);unlink(path);rmdir(directory);unlink(source);
}

static void test_pack_cooked_experts(void){
    enum{GATE_INPUT=256,GATE_OUTPUT=8,GATE_BLOCK=144,DOWN_INPUT=32,DOWN_OUTPUT=16,DOWN_BLOCK=24};
    const uint64_t gate_expert=(uint64_t)GATE_OUTPUT*GATE_BLOCK,gate_bytes=gate_expert*FG_EXPERT_COUNT,down_expert=(uint64_t)DOWN_OUTPUT*DOWN_BLOCK,down_bytes=down_expert*FG_EXPERT_COUNT;
    char source[128],directory[128],manifest_path[160],path[160];snprintf(source,sizeof(source),"/tmp/fg-cooked-expert-%ld.gguf",(long)getpid());snprintf(directory,sizeof(directory),"/tmp/fg-cooked-expert-%ld-pack",(long)getpid());snprintf(manifest_path,sizeof(manifest_path),"%s/manifest.fgm",directory);FILE *file=fopen(source,"wb");CHECK(file!=NULL);if(!file)return;
    put_u32(file,0x46554747u);put_u32(file,3u);put_u64(file,2u);put_u64(file,0u);put_string(file,"blk.0.ffn_gate_exps.weight");put_u32(file,3u);put_u64(file,GATE_INPUT);put_u64(file,GATE_OUTPUT);put_u64(file,FG_EXPERT_COUNT);put_u32(file,12u);put_u64(file,0u);put_string(file,"blk.0.ffn_down_exps.weight");put_u32(file,3u);put_u64(file,DOWN_INPUT);put_u64(file,DOWN_OUTPUT);put_u64(file,FG_EXPERT_COUNT);put_u32(file,7u);put_u64(file,gate_bytes);while((ftell(file)&31)!=0)fputc(0,file);for(uint64_t i=0;i<gate_bytes;i++)fputc((int)(i*29u+7u),file);for(uint64_t i=0;i<down_bytes;i++)fputc((int)(i*31u+11u),file);fclose(file);
    const char *sources[]={source};fg_pack_options options={.output_dir=directory,.source_paths=sources,.source_count=1u,.skip_model_validation=true};fg_error error={0};CHECK(fg_pack_run(&options,&error)==FG_OK);fg_manifest *manifest=malloc(sizeof(*manifest));CHECK(manifest!=NULL);if(manifest){CHECK(fg_manifest_read(manifest_path,manifest,&error)==FG_OK);uint32_t gate_count=0,down_count=0;for(uint32_t i=0;i<manifest->tensor_count;i++){const fg_tensor_record *record=&manifest->tensors[i];if(strstr(record->name,"ffn_gate_exps")){gate_count++;CHECK(record->layout==FG_TENSOR_LAYOUT_K_QUANT_EXPERT_COOKED&&record->bytes==gate_expert*FG_EXPERTS_PER_RANK);}if(strstr(record->name,"ffn_down_exps")){down_count++;CHECK(record->layout==FG_TENSOR_LAYOUT_Q5_1_EXPERT_COOKED&&record->bytes==down_expert*FG_EXPERTS_PER_RANK);}}CHECK(gate_count==FG_GROUP_SIZE&&down_count==FG_GROUP_SIZE);fg_verify_options verify={.manifest_path=manifest_path,.pack_dir=directory,.source_paths=sources,.source_count=1u};CHECK(fg_pack_verify(&verify,&error)==FG_OK);free(manifest);}
    unlink(manifest_path);for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){snprintf(path,sizeof(path),"%s/rank-%02u.fgw",directory,rank);unlink(path);}snprintf(path,sizeof(path),"%s/ngram.iq4nl",directory);unlink(path);rmdir(directory);unlink(source);
}

static void test_decode_protocol(void){
    fg_decode_work in={.layer=7,.source_rank=7,.destination_rank=0,.selected_count=2,.position=123456};
    in.expert_ids[0]=17;in.expert_ids[1]=511;in.routing_slots[0]=2;in.routing_slots[1]=9;in.gates[0]=0.25f;in.gates[1]=0.75f;
    for(uint32_t i=0;i<FG_Q8K_ACTIVATION_BYTES;i++)in.activation_q8k[i]=(uint8_t)(i*17u);
    uint8_t wire[FG_DECODE_WORK_BYTES];fg_decode_work out;fg_error err={0};
    CHECK(fg_decode_work_encode(wire,&in,&err)==FG_OK);
    CHECK(fg_decode_work_decode(&out,wire,sizeof(wire),&err)==FG_OK);
    CHECK(out.position==in.position&&out.expert_ids[1]==511&&out.routing_slots[0]==2&&out.gates[1]==0.75f);
    CHECK(memcmp(out.activation_q8k,in.activation_q8k,sizeof(in.activation_q8k))==0);
    CHECK(fg_decode_work_decode(&out,wire,sizeof(wire)-1u,&err)==FG_ERR_FORMAT);
    fg_manifest *m=malloc(sizeof(*m));CHECK(m!=NULL);if(!m)return;fg_manifest_init(m);
    uint16_t ids[FG_TOP_K];float gates[FG_TOP_K];for(uint32_t i=0;i<FG_TOP_K;i++){ids[i]=(uint16_t)i;gates[i]=0.1f;}
    fg_expert_route routes[FG_GROUP_SIZE];uint32_t count=0;
    CHECK(fg_partition_route(m,0,ids,gates,routes,&count,&err)==FG_OK);CHECK(count==4);
    uint32_t total=0;for(uint32_t r=0;r<count;r++){total+=routes[r].selected_count;for(uint32_t j=0;j<routes[r].selected_count;j++)CHECK(routes[r].local_expert_ids[j]<FG_EXPERTS_PER_RANK);}
    CHECK(total==FG_TOP_K);
    fg_expert_result *routed_results=calloc(count,sizeof(*routed_results));CHECK(routed_results!=NULL);
    if(routed_results){
        for(uint32_t r=0;r<count;r++){routed_results[r].layer=0;routed_results[r].source_rank=routes[r].destination_rank;routed_results[r].destination_rank=m->layer_owner[0];routed_results[r].selected_count=routes[r].selected_count;routed_results[r].position=55;memcpy(routed_results[r].routing_slots,routes[r].routing_slots,routes[r].selected_count);}
        CHECK(fg_expert_results_validate_route(m,0,55,m->layer_owner[0],ids,routed_results,count,&err)==FG_OK);
        routed_results[0].position=54;CHECK(fg_expert_results_validate_route(m,0,55,m->layer_owner[0],ids,routed_results,count,&err)==FG_ERR_MISMATCH);routed_results[0].position=55;
        uint8_t source=routed_results[1].source_rank;routed_results[1].source_rank=routed_results[0].source_rank;CHECK(fg_expert_results_validate_route(m,0,55,m->layer_owner[0],ids,routed_results,count,&err)==FG_ERR_MISMATCH);routed_results[1].source_rank=source;
        routed_results[0].destination_rank=(uint8_t)((m->layer_owner[0]+1u)%FG_RANK_COUNT);CHECK(fg_expert_results_validate_route(m,0,55,m->layer_owner[0],ids,routed_results,count,&err)==FG_ERR_MISMATCH);
        free(routed_results);
    }
    free(m);
    fg_expert_result *result=calloc(1,sizeof(*result)),*decoded=malloc(sizeof(*decoded));uint8_t *result_wire=malloc(FG_EXPERT_RESULT_MAX_BYTES);CHECK(result&&decoded&&result_wire);if(result&&decoded&&result_wire){result->layer=4;result->source_rank=1;result->destination_rank=4;result->selected_count=2;result->position=987;result->routing_slots[0]=1;result->routing_slots[1]=8;for(uint32_t i=0;i<FG_HIDDEN_SIZE;i++){result->outputs[0][i]=(float)i*0.001f;result->outputs[1][i]=-(float)i*0.002f;}uint32_t result_bytes=0;CHECK(fg_expert_result_encode(result_wire,FG_EXPERT_RESULT_MAX_BYTES,&result_bytes,result,&err)==FG_OK);CHECK(result_bytes==8u+2u*FG_EXPERT_RESULT_ENTRY_BYTES);CHECK(fg_expert_result_decode(decoded,result_wire,result_bytes,&err)==FG_OK);CHECK(decoded->routing_slots[1]==8&&decoded->position==987&&decoded->outputs[1][2559]==result->outputs[1][2559]);result_wire[9]=1;CHECK(fg_expert_result_decode(decoded,result_wire,result_bytes,&err)==FG_ERR_FORMAT);}free(result_wire);free(decoded);free(result);
}

static void test_prefill_protocol(void){
    enum{TOKENS=3,PAIRS=5};fg_error err={0};
    uint8_t *activations=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES),*decoded_activations=malloc(TOKENS*FG_Q8K_ACTIVATION_BYTES),*wire=malloc(FG_PREFILL_WORK_MAX_BYTES);
    fg_prefill_pair pairs[PAIRS]={{0,11,0,0.1f},{0,22,1,0.2f},{1,33,0,0.3f},{2,44,0,0.4f},{2,55,7,0.5f}},decoded_pairs[PAIRS];
    CHECK(activations&&decoded_activations&&wire);if(!activations||!decoded_activations||!wire){free(wire);free(decoded_activations);free(activations);return;}
    for(uint32_t i=0;i<TOKENS*FG_Q8K_ACTIVATION_BYTES;i++)activations[i]=(uint8_t)(i*29u+7u);
    fg_prefill_work work={.layer=9,.source_rank=1,.destination_rank=3,.first_position=4096,.token_count=TOKENS,.pair_count=PAIRS,.activations_q8k=activations,.pairs=pairs},decoded={0};uint32_t bytes=0;
    CHECK(fg_prefill_work_encode(wire,FG_PREFILL_WORK_MAX_BYTES,&bytes,&work,&err)==FG_OK);
    CHECK(bytes==FG_PREFILL_WORK_HEADER_BYTES+TOKENS*FG_Q8K_ACTIVATION_BYTES+PAIRS*FG_PREFILL_PAIR_BYTES);
    CHECK(fg_prefill_work_decode(&decoded,decoded_activations,sizeof(uint8_t)*TOKENS*FG_Q8K_ACTIVATION_BYTES,decoded_pairs,PAIRS,wire,bytes,&err)==FG_OK);
    CHECK(decoded.layer==work.layer&&decoded.source_rank==work.source_rank&&decoded.destination_rank==work.destination_rank&&decoded.first_position==work.first_position&&decoded.token_count==TOKENS&&decoded.pair_count==PAIRS);
    CHECK(memcmp(activations,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES)==0&&decoded_pairs[4].expert_id==55&&decoded_pairs[4].routing_slot==7&&decoded_pairs[4].gate==0.5f);
    uint32_t pair_offset=FG_PREFILL_WORK_HEADER_BYTES+TOKENS*FG_Q8K_ACTIVATION_BYTES;wire[pair_offset+5u]=1u;CHECK(fg_prefill_work_decode(&decoded,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES,decoded_pairs,PAIRS,wire,bytes,&err)==FG_ERR_FORMAT);wire[pair_offset+5u]=0u;
    CHECK(fg_prefill_work_decode(&decoded,decoded_activations,TOKENS*FG_Q8K_ACTIVATION_BYTES,decoded_pairs,PAIRS,wire,bytes-1u,&err)==FG_ERR_FORMAT);
    fg_prefill_result_pair result_pairs[PAIRS]={{0,0},{0,1},{1,0},{2,0},{2,7}},decoded_result_pairs[PAIRS];float *outputs=malloc((size_t)PAIRS*FG_HIDDEN_SIZE*4u),*decoded_outputs=malloc((size_t)PAIRS*FG_HIDDEN_SIZE*4u);uint8_t *result_wire=malloc(FG_PREFILL_RESULT_MAX_BYTES);CHECK(outputs&&decoded_outputs&&result_wire);
    if(outputs&&decoded_outputs&&result_wire){for(uint32_t i=0;i<PAIRS*FG_HIDDEN_SIZE;i++)outputs[i]=sinf((float)i*0.0007f);fg_prefill_result result={.layer=9,.source_rank=3,.destination_rank=1,.first_position=4096,.token_count=TOKENS,.pair_count=PAIRS,.pairs=result_pairs,.outputs=outputs},decoded_result={0};uint32_t result_bytes=0;CHECK(fg_prefill_result_encode(result_wire,FG_PREFILL_RESULT_MAX_BYTES,&result_bytes,&result,&err)==FG_OK);CHECK(result_bytes==FG_PREFILL_RESULT_HEADER_BYTES+PAIRS*FG_PREFILL_RESULT_PAIR_BYTES);CHECK(fg_prefill_result_decode(&decoded_result,decoded_result_pairs,PAIRS,decoded_outputs,(uint64_t)PAIRS*FG_HIDDEN_SIZE,result_wire,result_bytes,&err)==FG_OK);CHECK(decoded_result.layer==9&&decoded_result.first_position==4096&&decoded_result_pairs[4].token_slot==2&&decoded_result_pairs[4].routing_slot==7&&decoded_outputs[PAIRS*FG_HIDDEN_SIZE-1u]==outputs[PAIRS*FG_HIDDEN_SIZE-1u]);result_wire[15]=1u;CHECK(fg_prefill_result_decode(&decoded_result,decoded_result_pairs,PAIRS,decoded_outputs,(uint64_t)PAIRS*FG_HIDDEN_SIZE,result_wire,result_bytes,&err)==FG_ERR_FORMAT);}
    uint32_t positions[TOKENS*3u],decoded_positions[TOKENS*3u];float *layer_hyper=malloc((size_t)TOKENS*FG_HYPER_WIDTH*4u),*decoded_hyper=malloc((size_t)TOKENS*FG_HYPER_WIDTH*4u),*layer_ngram=malloc((size_t)TOKENS*FG_NGRAM_EMBED_VALUES*4u),*decoded_ngram=malloc((size_t)TOKENS*FG_NGRAM_EMBED_VALUES*4u);uint8_t *layer_wire=malloc(FG_PREFILL_LAYER_WORK_MAX_BYTES),*layer_result_wire=malloc(FG_PREFILL_LAYER_RESULT_MAX_BYTES);CHECK(layer_hyper&&decoded_hyper&&layer_ngram&&decoded_ngram&&layer_wire&&layer_result_wire);if(layer_hyper&&decoded_hyper&&layer_ngram&&decoded_ngram&&layer_wire&&layer_result_wire){for(uint32_t i=0;i<TOKENS*3u;i++)positions[i]=700u+i;for(uint32_t i=0;i<TOKENS*FG_HYPER_WIDTH;i++)layer_hyper[i]=sinf((float)i*0.0003f);for(uint32_t i=0;i<TOKENS*FG_NGRAM_EMBED_VALUES;i++)layer_ngram[i]=cosf((float)i*0.0009f);fg_prefill_layer_work layer_work={.layer=1,.source_rank=0,.destination_rank=1,.flags=FG_LAYER_WORK_HAS_NGRAM,.first_token=700,.token_count=TOKENS,.positions=positions,.hyper=layer_hyper,.ngram_embeddings=layer_ngram},decoded_layer={0};uint32_t layer_bytes=0;CHECK(fg_prefill_layer_work_encode(layer_wire,FG_PREFILL_LAYER_WORK_MAX_BYTES,&layer_bytes,FG_PROTOCOL_VERSION,&layer_work,&err)==FG_OK);CHECK(layer_bytes==FG_PREFILL_LAYER_HEADER_BYTES+TOKENS*3u*4u+TOKENS*FG_HYPER_WIDTH*4u+TOKENS*FG_NGRAM_EMBED_VALUES*4u);CHECK(fg_prefill_layer_work_decode(&decoded_layer,FG_PROTOCOL_VERSION,decoded_positions,TOKENS*3u,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,decoded_ngram,(uint64_t)TOKENS*FG_NGRAM_EMBED_VALUES,layer_wire,layer_bytes,&err)==FG_OK);CHECK(decoded_layer.layer==1u&&decoded_layer.first_token==700u&&decoded_layer.token_count==TOKENS&&memcmp(positions,decoded_positions,sizeof(positions))==0&&memcmp(layer_hyper,decoded_hyper,(size_t)TOKENS*FG_HYPER_WIDTH*4u)==0&&memcmp(layer_ngram,decoded_ngram,(size_t)TOKENS*FG_NGRAM_EMBED_VALUES*4u)==0);layer_wire[12]=1u;CHECK(fg_prefill_layer_work_decode(&decoded_layer,FG_PROTOCOL_VERSION,decoded_positions,TOKENS*3u,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,decoded_ngram,(uint64_t)TOKENS*FG_NGRAM_EMBED_VALUES,layer_wire,layer_bytes,&err)==FG_ERR_FORMAT);layer_wire[12]=0u;CHECK(fg_prefill_layer_work_decode(&decoded_layer,FG_PROTOCOL_VERSION,decoded_positions,TOKENS*3u,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,decoded_ngram,(uint64_t)TOKENS*FG_NGRAM_EMBED_VALUES,layer_wire,layer_bytes-1u,&err)==FG_ERR_FORMAT);fg_prefill_layer_result layer_result={.layer=1,.source_rank=1,.destination_rank=0,.first_token=700,.token_count=TOKENS,.hyper=layer_hyper},decoded_layer_result={0};uint32_t layer_result_bytes=0;CHECK(fg_prefill_layer_result_encode(layer_result_wire,FG_PREFILL_LAYER_RESULT_MAX_BYTES,&layer_result_bytes,&layer_result,&err)==FG_OK);CHECK(fg_prefill_layer_result_decode(&decoded_layer_result,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,layer_result_wire,layer_result_bytes,&err)==FG_OK);CHECK(decoded_layer_result.first_token==700u&&decoded_layer_result.token_count==TOKENS&&memcmp(layer_hyper,decoded_hyper,(size_t)TOKENS*FG_HYPER_WIDTH*4u)==0);layer_result_wire[10]=1u;CHECK(fg_prefill_layer_result_decode(&decoded_layer_result,decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,layer_result_wire,layer_result_bytes,&err)==FG_ERR_FORMAT);}free(layer_result_wire);free(layer_wire);free(decoded_ngram);free(layer_ngram);free(decoded_hyper);free(layer_hyper);
    fg_manifest *manifest=malloc(sizeof(*manifest));fg_prefill_pair route_storage[TOKENS*FG_TOP_K];fg_prefill_route routes[FG_GROUP_SIZE];uint16_t route_ids[TOKENS*FG_TOP_K];float route_gates[TOKENS*FG_TOP_K];uint32_t route_count=0;CHECK(manifest!=NULL);if(manifest){fg_manifest_init(manifest);for(uint32_t token=0;token<TOKENS;token++)for(uint32_t slot=0;slot<FG_TOP_K;slot++){route_ids[token*FG_TOP_K+slot]=(uint16_t)(token*FG_TOP_K+slot);route_gates[token*FG_TOP_K+slot]=0.01f*(float)(slot+1u);}CHECK(fg_partition_prefill_routes(manifest,0,TOKENS,route_ids,route_gates,routes,&route_count,route_storage,TOKENS*FG_TOP_K,&err)==FG_OK);CHECK(route_count==FG_GROUP_SIZE);fg_prefill_result routed[FG_GROUP_SIZE];fg_prefill_result_pair routed_pairs[TOKENS*FG_TOP_K];float *routed_outputs=calloc((size_t)TOKENS*FG_TOP_K*FG_HIDDEN_SIZE,sizeof(*routed_outputs));CHECK(routed_outputs!=NULL);if(routed_outputs){uint32_t pair_offset=0;for(uint32_t route=0;route<route_count;route++){routed[route]=(fg_prefill_result){.layer=0,.source_rank=routes[route].destination_rank,.destination_rank=manifest->layer_owner[0],.first_position=700,.token_count=TOKENS,.pair_count=routes[route].pair_count,.pairs=routed_pairs+pair_offset,.outputs=routed_outputs+(uint64_t)pair_offset*FG_HIDDEN_SIZE};for(uint32_t pair=0;pair<routes[route].pair_count;pair++){routed_pairs[pair_offset+pair].token_slot=routes[route].pairs[pair].token_slot;routed_pairs[pair_offset+pair].routing_slot=routes[route].pairs[pair].routing_slot;}pair_offset+=routes[route].pair_count;}CHECK(pair_offset==TOKENS*FG_TOP_K);CHECK(fg_prefill_results_validate_route(manifest,0,700,manifest->layer_owner[0],TOKENS,route_ids,routed,route_count,&err)==FG_OK);uint8_t source=routed[0].source_rank;routed[0].source_rank=routed[1].source_rank;CHECK(fg_prefill_results_validate_route(manifest,0,700,manifest->layer_owner[0],TOKENS,route_ids,routed,route_count,&err)==FG_ERR_MISMATCH);routed[0].source_rank=source;}free(routed_outputs);free(manifest);}
    free(result_wire);free(decoded_outputs);free(outputs);free(wire);free(decoded_activations);free(activations);
}

static void test_output_history_protocol(void){
    uint32_t tokens[]={7u,7u,42u},storage[8]={0},bytes=0u;
    uint8_t wire[64]={0};fg_error err={0};
    fg_output_history source={.tokens=tokens,.count=3u},decoded={0};
    CHECK(fg_output_history_encode(wire,sizeof(wire),&bytes,&source,&err)==FG_OK);
    CHECK(bytes==FG_OUTPUT_HISTORY_HEADER_BYTES+12u);
    fg_frame_header frame={0};uint32_t frame_bytes=0u;
    CHECK(fg_frame_encode_version(&frame,FG_PIPELINE_PROTOCOL_VERSION,
        FG_MSG_OUTPUT_HISTORY,17u,0u,0u,wire,bytes,&err)==FG_OK);
    CHECK(fg_frame_validate_version(&frame,FG_PIPELINE_PROTOCOL_VERSION,
        wire,&frame_bytes,&err)==FG_OK&&frame_bytes==bytes);
    CHECK(fg_frame_encode_version(&frame,FG_PIPELINE_PROTOCOL_VERSION,
        FG_MSG_OUTPUT_HISTORY_ACK,17u,0u,0u,NULL,0u,&err)==FG_OK);
    CHECK(fg_frame_validate_version(&frame,FG_PIPELINE_PROTOCOL_VERSION,
        NULL,&frame_bytes,&err)==FG_OK&&frame_bytes==0u);
    CHECK(fg_frame_encode_version(&frame,FG_PROTOCOL_VERSION,
        FG_MSG_OUTPUT_HISTORY,17u,0u,0u,wire,bytes,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_output_history_decode(&decoded,storage,8u,wire,bytes,&err)==FG_OK);
    CHECK(decoded.count==3u&&storage[0]==7u&&storage[1]==7u&&storage[2]==42u);
    fg_output_history empty={.tokens=NULL,.count=0u},empty_decoded={0};
    uint32_t empty_bytes=0u;
    CHECK(fg_output_history_encode(wire,sizeof(wire),&empty_bytes,&empty,&err)==FG_OK);
    CHECK(empty_bytes==FG_OUTPUT_HISTORY_HEADER_BYTES);
    CHECK(fg_output_history_decode(&empty_decoded,NULL,0u,wire,empty_bytes,&err)==FG_OK&&empty_decoded.count==0u);
    wire[4]=1u;
    CHECK(fg_output_history_decode(&decoded,storage,8u,wire,empty_bytes,&err)==FG_ERR_FORMAT);
    wire[4]=0u;wire[3]=1u;
    CHECK(fg_output_history_decode(&decoded,storage,8u,wire,empty_bytes,&err)==FG_ERR_FORMAT);
    CHECK(fg_output_history_decode(&decoded,storage,8u,wire,empty_bytes-1u,&err)==FG_ERR_ARGUMENT);
    CHECK(fg_output_history_encode(wire,sizeof(wire),&bytes,&source,&err)==FG_OK);
    wire[FG_OUTPUT_HISTORY_HEADER_BYTES]=0xffu;wire[FG_OUTPUT_HISTORY_HEADER_BYTES+1u]=0xffu;
    wire[FG_OUTPUT_HISTORY_HEADER_BYTES+2u]=0xffu;wire[FG_OUTPUT_HISTORY_HEADER_BYTES+3u]=0xffu;
    CHECK(fg_output_history_decode(&decoded,storage,8u,wire,bytes,&err)==FG_ERR_FORMAT);
}

int main(void){test_sha();test_topology();test_pipeline_topology();test_profile();test_expert_map();test_expert_map_file();test_sealed_expert_map();test_deployment_profile();test_native_262k_profile_geometry();test_protocol();test_layer_protocol();test_qsa_block_protocol();test_qsa_page_protocol();test_prefill_chunk_frontiers();test_qsa_locality_metrics();test_output_protocol();test_output_history_protocol();test_ngram_protocol();test_ngram();test_ngram_planner_batch_capacity();test_pipeline_owner_transient_geometry();test_qsa_scratch_geometry();test_qsa_state();test_qsa_state_failed_create_cleanup();test_qsa_state_batch();test_qsa_state_write_batch();test_qsa_replica_queue();test_lazy_qsa_clear_barrier();test_prefill_storage_geometry();test_qsa_page_cache();test_q38_math();test_cooked_q8();test_pack_cooked_q8();test_pack_cooked_experts();test_decode_protocol();test_prefill_protocol();test_pack();test_pipeline_pack();test_pipeline_spill_admission();if(failures){fprintf(stderr,"%d test(s) failed\n",failures);return 1;}puts("core tests: PASS");return 0;}
