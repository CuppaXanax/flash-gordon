#include "fg_manifest.h"
#include "fg_protocol.h"
#include "fg_session.h"
#include "fg_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
#define CHECK(expression) do{if(!(expression)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#expression);failures++;}}while(0)

static void fill_digest(uint8_t digest[32],uint8_t seed){
    for(uint32_t i=0;i<32u;i++)digest[i]=(uint8_t)(seed+i*17u);
}

static fg_manifest *write_read_manifest(const char *path,uint8_t source_seed){
    fg_manifest *manifest=malloc(sizeof(*manifest)),*sealed=malloc(sizeof(*sealed));
    CHECK(manifest&&sealed);if(!manifest||!sealed){free(sealed);free(manifest);return NULL;}
    fg_manifest_init(manifest);manifest->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
    fill_digest(manifest->source_sha256,source_seed);
    fg_error error={0};CHECK(fg_manifest_write(path,manifest,&error)==FG_OK);
    CHECK(fg_manifest_read(path,sealed,&error)==FG_OK);free(manifest);
    if(error.code!=FG_OK){free(sealed);return NULL;}return sealed;
}

static void write_legacy_manifest(const char *path,fg_manifest *manifest){
    manifest->format_version=FG_MANIFEST_LEGACY_FORMAT_VERSION;
    manifest->protocol_version=FG_PROTOCOL_MIN_VERSION;
    manifest->header_bytes=(uint32_t)FG_MANIFEST_V4_BYTES;
    memset(manifest->manifest_sha256,0,32u);
    fg_sha256 hash;fg_sha256_init(&hash);
    fg_sha256_update(&hash,manifest,FG_MANIFEST_V4_BYTES);
    fg_sha256_final(&hash,manifest->manifest_sha256);
    FILE *file=fopen(path,"wb");CHECK(file!=NULL);
    if(file){CHECK(fwrite(manifest,1,FG_MANIFEST_V4_BYTES,file)==FG_MANIFEST_V4_BYTES);CHECK(fclose(file)==0);}
}

static void test_manifest_evolution(void){
    char current_path[96],legacy_path[96];
    snprintf(current_path,sizeof(current_path),"test-session-current-%ld.fgm",(long)getpid());
    snprintf(legacy_path,sizeof(legacy_path),"test-session-legacy-%ld.fgm",(long)getpid());
    unlink(current_path);unlink(legacy_path);
    fg_manifest *current=write_read_manifest(current_path,11u);fg_error error={0};
    if(current){
        CHECK(current->format_version==FG_MANIFEST_FORMAT_VERSION);
        CHECK(current->protocol_version==FG_PROTOCOL_VERSION);
        CHECK(current->session.version==FG_MANIFEST_CONTRACT_VERSION);
        CHECK(current->session.gpu_index_tokens==8192u);
        CHECK(current->session.qsa_hot_record_tokens==8192u);
        CHECK(current->session.host_page_cache_bytes==0u);
        CHECK(current->session.position_mode==FG_POSITION_TEXT);
        CHECK(memcmp(current->session.component_sha256[FG_COMPONENT_TEXT],
                     current->session.component_sha256[FG_COMPONENT_TOKENIZER],32u)!=0);
        CHECK(memcmp(current->session.rank_state_format_sha256[0],
                     current->session.rank_state_format_sha256[3],32u)!=0);
        CHECK(fg_manifest_validate_compatibility(current,FG_PROTOCOL_VERSION,
                                                 FG_POSITION_TEXT,&error)==FG_OK);
        CHECK(fg_manifest_validate_compatibility(current,FG_PROTOCOL_VERSION,
                                                 FG_POSITION_FOUR_AXIS,&error)==FG_ERR_MISMATCH);
        current->session.reserved[0]=1u;
        CHECK(fg_manifest_validate(current,&error)==FG_ERR_FORMAT);
        current->session.reserved[0]=0u;
    }
    fg_manifest *legacy=malloc(sizeof(*legacy)),*decoded=malloc(sizeof(*decoded));
    CHECK(legacy&&decoded);
    if(legacy&&decoded){
        fg_manifest_init(legacy);legacy->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
        fill_digest(legacy->source_sha256,23u);write_legacy_manifest(legacy_path,legacy);
        CHECK(fg_manifest_read(legacy_path,decoded,&error)==FG_OK);
        CHECK(decoded->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION);
        CHECK(decoded->protocol_version==FG_PROTOCOL_MIN_VERSION);
        CHECK(decoded->session.version==FG_MANIFEST_CONTRACT_VERSION);
        CHECK(decoded->session.position_mode==FG_POSITION_TEXT);
        CHECK(fg_manifest_validate_compatibility(decoded,FG_PROTOCOL_MIN_VERSION,
                                                 FG_POSITION_TEXT,&error)==FG_OK);
        CHECK(fg_manifest_validate_compatibility(decoded,FG_PROTOCOL_VERSION,
                                                 FG_POSITION_TEXT,&error)==FG_ERR_MISMATCH);
    }
    free(decoded);free(legacy);free(current);unlink(current_path);unlink(legacy_path);
}

static void test_identity_and_frontier(void){
    char first_path[96],second_path[96];
    snprintf(first_path,sizeof(first_path),"test-session-identity-a-%ld.fgm",(long)getpid());
    snprintf(second_path,sizeof(second_path),"test-session-identity-b-%ld.fgm",(long)getpid());
    fg_manifest *first=write_read_manifest(first_path,31u);
    fg_manifest *second=write_read_manifest(second_path,47u);
    fg_error error={0};fg_session_identity identity={0},decoded_identity={0},other={0};
    if(first&&second){
        CHECK(fg_session_identity_from_manifest(first,&identity,&error)==FG_OK);
        CHECK(fg_session_identity_from_manifest(second,&other,&error)==FG_OK);
        uint8_t wire[FG_SESSION_IDENTITY_WIRE_BYTES],again[FG_SESSION_IDENTITY_WIRE_BYTES];
        CHECK(fg_session_identity_encode(wire,&identity,&error)==FG_OK);
        CHECK(fg_session_identity_encode(again,&identity,&error)==FG_OK);
        CHECK(memcmp(wire,again,sizeof(wire))==0);
        CHECK(fg_session_identity_decode(&decoded_identity,wire,sizeof(wire),&error)==FG_OK);
        CHECK(fg_session_identity_validate_compatible(&identity,&decoded_identity,&error)==FG_OK);
        CHECK(fg_session_identity_validate_compatible(&identity,&other,&error)==FG_ERR_MISMATCH);
        wire[32]^=1u;
        CHECK(fg_session_identity_decode(&decoded_identity,wire,sizeof(wire),&error)==FG_ERR_MISMATCH);

        int32_t tokens[]={1,17,248319,99};
        fg_session_frontier frontier={.version=FG_SESSION_FRONTIER_VERSION,
            .position_mode=FG_POSITION_TEXT,.generation=7u,.committed_tokens=4u,
            .token_count=4u,.next_token_valid=true,.next_token=99u,.next_logit=1.25f,
            .position={4u,4u,4u,0u},.tokens=tokens};
        memcpy(frontier.identity_sha256,identity.identity_sha256,32u);
        fill_digest(frontier.rendered_transcript_sha256,61u);
        fill_digest(frontier.next_token_state_sha256,73u);
        for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
            frontier.qsa_lengths[layer]=(layer&3u)==3u?4u:0u;
            frontier.gdn_lengths[layer]=(layer&3u)==0u?4u:0u;
            frontier.ple_lengths[layer]=(layer&3u)==2u?4u:0u;
        }
        uint32_t capacity=FG_SESSION_FRONTIER_HEADER_BYTES+sizeof(tokens)+
                          FG_SESSION_FRONTIER_DIGEST_BYTES;
        uint8_t *frontier_wire=malloc(capacity),*frontier_again=malloc(capacity);
        int32_t decoded_tokens[4]={0};fg_session_frontier decoded={0};uint32_t bytes=0,again_bytes=0;
        CHECK(frontier_wire&&frontier_again);
        if(frontier_wire&&frontier_again){
            CHECK(fg_session_frontier_encode(frontier_wire,capacity,&bytes,&frontier,&error)==FG_OK);
            CHECK(bytes==capacity);
            CHECK(fg_session_frontier_decode(&decoded,decoded_tokens,4u,frontier_wire,bytes,
                                             &error)==FG_OK);
            CHECK(decoded.generation==7u&&decoded.committed_tokens==4u&&
                  decoded.next_token==99u&&decoded.next_logit==1.25f);
            CHECK(memcmp(tokens,decoded_tokens,sizeof(tokens))==0);
            CHECK(fg_session_frontier_validate_compatible(&identity,&decoded,FG_POSITION_TEXT,
                                                          &error)==FG_OK);
            CHECK(fg_session_frontier_validate_compatible(&identity,&decoded,
                                                          FG_POSITION_FOUR_AXIS,
                                                          &error)==FG_ERR_MISMATCH);
            CHECK(fg_session_frontier_encode(frontier_again,capacity,&again_bytes,&decoded,
                                             &error)==FG_OK);
            CHECK(again_bytes==bytes&&memcmp(frontier_wire,frontier_again,bytes)==0);
            frontier_wire[FG_SESSION_FRONTIER_HEADER_BYTES]^=1u;
            CHECK(fg_session_frontier_decode(&decoded,decoded_tokens,4u,frontier_wire,bytes,
                                             &error)==FG_ERR_MISMATCH);
            frontier.committed_tokens=5u;
            CHECK(fg_session_frontier_encode(frontier_wire,capacity,&bytes,&frontier,
                                             &error)==FG_ERR_MISMATCH);
        }
        free(frontier_again);free(frontier_wire);
    }
    free(second);free(first);unlink(first_path);unlink(second_path);
}

static void test_owner_controls(void){
    fg_error error={0};uint8_t wire[FG_OWNER_SESSION_CONTROL_BYTES];
    for(uint32_t operation=FG_OWNER_SESSION_BEGIN;operation<=FG_OWNER_SESSION_RESTORED;
        operation++){
        fg_owner_session_control control={.version=FG_OWNER_SESSION_CONTROL_VERSION,
            .operation=(uint8_t)operation,.rank=3u,.position_mode=FG_POSITION_TEXT,
            .session_nonce=UINT64_C(0x1122334455667788)};
        fill_digest(control.identity_sha256,3u);fill_digest(control.state_format_sha256,5u);
        if(operation>=FG_OWNER_SESSION_PREPARE){
            control.generation=9u;control.committed_tokens=41u;
            fill_digest(control.frontier_sha256,7u);
        }
        if(operation==FG_OWNER_SESSION_PREPARED||operation==FG_OWNER_SESSION_RESTORED)
            fill_digest(control.state_sha256,9u);
        CHECK(fg_owner_session_control_encode(wire,&control,&error)==FG_OK);
        fg_owner_session_control decoded={0};
        CHECK(fg_owner_session_control_decode(&decoded,wire,sizeof(wire),&error)==FG_OK);
        CHECK(decoded.operation==operation&&decoded.rank==3u&&decoded.generation==control.generation);
    }
    wire[6]=1u;fg_owner_session_control decoded={0};
    CHECK(fg_owner_session_control_decode(&decoded,wire,sizeof(wire),&error)==FG_ERR_FORMAT);
}

static void put_u32_be(uint8_t *output,uint32_t value){
    output[0]=(uint8_t)(value>>24u);output[1]=(uint8_t)(value>>16u);
    output[2]=(uint8_t)(value>>8u);output[3]=(uint8_t)value;
}

static void test_protocol_evolution(void){
    fg_error error={0};fg_frame_header header;const uint8_t payload[]={1u,2u,3u};
    CHECK(fg_frame_encode_version(&header,FG_PROTOCOL_MIN_VERSION,FG_MSG_CONTROL,9u,1u,0u,
                                  payload,sizeof(payload),&error)==FG_OK);
    CHECK(fg_frame_version(&header)==FG_PROTOCOL_MIN_VERSION);
    CHECK(fg_frame_validate(&header,payload,NULL,&error)==FG_OK);
    CHECK(fg_frame_encode_version(&header,FG_PROTOCOL_MIN_VERSION,FG_MSG_SESSION_PREPARE,
                                  9u,1u,0u,payload,sizeof(payload),&error)==FG_ERR_ARGUMENT);
    CHECK(fg_frame_encode_version(&header,FG_PROTOCOL_VERSION,FG_MSG_SESSION_PREPARE,
                                  9u,1u,0u,payload,sizeof(payload),&error)==FG_OK);
    CHECK(fg_frame_validate(&header,payload,NULL,&error)==FG_OK);

    fg_layer_work *work=calloc(1,sizeof(*work)),*decoded=calloc(1,sizeof(*decoded));
    uint8_t *wire=malloc(FG_LAYER_WORK_MAX_BYTES);
    CHECK(work&&decoded&&wire);
    if(work&&decoded&&wire){
        work->layer=2u;work->source_rank=0u;work->destination_rank=2u;
        work->position_mode=FG_POSITION_FOUR_AXIS;work->token_index=81u;
        work->position[0]=5u;work->position[1]=7u;work->position[2]=11u;work->position[3]=13u;
        uint32_t bytes=0;
        CHECK(fg_layer_work_encode(wire,FG_LAYER_WORK_MAX_BYTES,&bytes,work,&error)==FG_OK);
        CHECK(bytes==FG_LAYER_WORK_FOUR_AXIS_BASE_BYTES);
        CHECK(fg_layer_work_decode(decoded,wire,bytes,&error)==FG_OK);
        CHECK(decoded->position_mode==FG_POSITION_FOUR_AXIS&&decoded->position[3]==13u);

        uint32_t legacy_bytes=FG_LAYER_WORK_LEGACY_HEADER_BYTES+FG_HYPER_WIDTH*4u;
        memset(wire,0,legacy_bytes);wire[0]=2u;wire[1]=0u;wire[2]=2u;
        put_u32_be(wire+4u,82u);put_u32_be(wire+8u,17u);
        put_u32_be(wire+12u,19u);put_u32_be(wire+16u,23u);
        CHECK(fg_layer_work_decode(decoded,wire,legacy_bytes,&error)==FG_OK);
        CHECK(decoded->position_mode==FG_POSITION_TEXT&&decoded->position[0]==17u&&
              decoded->position[2]==23u&&decoded->position[3]==0u);
    }
    free(wire);free(decoded);free(work);

    enum{TOKENS=2};uint32_t positions[TOKENS*4u],decoded_positions[TOKENS*4u];
    float *hyper=calloc((size_t)TOKENS*FG_HYPER_WIDTH,sizeof(*hyper));
    float *decoded_hyper=calloc((size_t)TOKENS*FG_HYPER_WIDTH,sizeof(*decoded_hyper));
    uint8_t *prefill_wire=malloc(FG_PREFILL_LAYER_WORK_MAX_BYTES);
    CHECK(hyper&&decoded_hyper&&prefill_wire);
    if(hyper&&decoded_hyper&&prefill_wire){
        for(uint32_t i=0;i<TOKENS*4u;i++)positions[i]=100u+i;
        fg_prefill_layer_work work_batch={.layer=2u,.source_rank=0u,.destination_rank=2u,
            .position_mode=FG_POSITION_FOUR_AXIS,.first_token=100u,.token_count=TOKENS,
            .positions=positions,.hyper=hyper},decoded_batch={0};
        uint32_t bytes=0;
        CHECK(fg_prefill_layer_work_encode(prefill_wire,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &bytes,&work_batch,&error)==FG_OK);
        CHECK(bytes==FG_PREFILL_LAYER_HEADER_BYTES+TOKENS*4u*4u+
                     TOKENS*FG_HYPER_WIDTH*4u);
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_OK);
        CHECK(decoded_batch.position_mode==FG_POSITION_FOUR_AXIS&&
              decoded_positions[TOKENS*4u-1u]==positions[TOKENS*4u-1u]);
        prefill_wire[11]=3u;
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_ERR_FORMAT);
    }
    free(prefill_wire);free(decoded_hyper);free(hyper);
}

int main(void){
    test_manifest_evolution();test_identity_and_frontier();test_owner_controls();
    test_protocol_evolution();
    if(failures){fprintf(stderr,"%d session contract test(s) failed\n",failures);return 1;}
    puts("session contracts: PASS");return 0;
}
