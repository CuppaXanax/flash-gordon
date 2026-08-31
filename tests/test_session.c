#include "fg_manifest.h"
#include "fg_protocol.h"
#include "fg_qsa_owner.h"
#include "fg_runtime.h"
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
        CHECK(current->session.minimum_protocol_version==FG_PROTOCOL_VERSION);
        CHECK(current->session.logical_context_tokens==8192u);
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
        CHECK(fg_manifest_validate_compatibility(current,FG_PROTOCOL_MIN_VERSION,
                                                 FG_POSITION_TEXT,&error)==FG_ERR_MISMATCH);
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
        CHECK(decoded->session.minimum_protocol_version==FG_PROTOCOL_MIN_VERSION);
        CHECK(decoded->session.logical_context_tokens==8192u);
        CHECK(decoded->session.position_mode==FG_POSITION_TEXT);
        CHECK(fg_manifest_validate_compatibility(decoded,FG_PROTOCOL_MIN_VERSION,
                                                 FG_POSITION_TEXT,&error)==FG_OK);
        CHECK(fg_manifest_validate_compatibility(decoded,FG_PROTOCOL_VERSION,
                                                 FG_POSITION_TEXT,&error)==FG_ERR_MISMATCH);
    }
    free(decoded);free(legacy);free(current);unlink(current_path);unlink(legacy_path);
}

static void test_manifest_upgrade(void){
    char legacy_path[96],current_path[96],profile_path[96];
    snprintf(legacy_path,sizeof(legacy_path),"test-upgrade-legacy-%ld.fgm",(long)getpid());
    snprintf(current_path,sizeof(current_path),"test-upgrade-current-%ld.fgm",(long)getpid());
    snprintf(profile_path,sizeof(profile_path),"test-upgrade-profile-%ld.fgm",(long)getpid());
    unlink(legacy_path);unlink(current_path);unlink(profile_path);
    fg_manifest *legacy=malloc(sizeof(*legacy)),*upgraded=malloc(sizeof(*upgraded)),
        *profiled=malloc(sizeof(*profiled));
    CHECK(legacy&&upgraded&&profiled);fg_error error={0};
    if(legacy&&upgraded&&profiled){
        fg_manifest_init(legacy);
        legacy->format_version=FG_MANIFEST_LEGACY_FORMAT_VERSION;
        legacy->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
        legacy->required_cu=40u;
        legacy->persistent_cap_bytes+=123456u;
        legacy->residency_cap_bytes+=654321u;
        fill_digest(legacy->source_sha256,43u);
        fill_digest(legacy->quant_profile_sha256,59u);
        legacy->tensor_count=2u;
        snprintf(legacy->tensors[0].name,sizeof(legacy->tensors[0].name),"token_embd.weight");
        legacy->tensors[0].offset=4096u;
        legacy->tensors[0].bytes=8192u;
        legacy->tensors[0].ggml_type=8u;
        legacy->tensors[0].dims=2u;
        legacy->tensors[0].shape[0]=2560u;
        legacy->tensors[0].shape[1]=4u;
        legacy->tensors[0].rank=0u;
        legacy->tensors[0].layer=UINT16_MAX;
        legacy->tensors[0].expert=UINT16_MAX;
        legacy->tensors[0].kind=FG_TENSOR_COMMON;
        fill_digest(legacy->tensors[0].sha256,67u);
        snprintf(legacy->tensors[1].name,sizeof(legacy->tensors[1].name),
                 "blk.0.ffn_gate_exps.weight.rank0");
        legacy->tensors[1].offset=16384u;
        legacy->tensors[1].bytes=32768u;
        legacy->tensors[1].ggml_type=12u;
        legacy->tensors[1].dims=3u;
        legacy->tensors[1].shape[0]=2560u;
        legacy->tensors[1].shape[1]=2560u;
        legacy->tensors[1].shape[2]=128u;
        legacy->tensors[1].rank=0u;
        legacy->tensors[1].layer=0u;
        legacy->tensors[1].expert=UINT16_MAX;
        legacy->tensors[1].kind=FG_TENSOR_ROUTED_EXPERT;
        fill_digest(legacy->tensors[1].sha256,71u);
        legacy->ranks[3].state_file_bytes=123456u;
        legacy->ranks[3].persistent_bytes=234567u;
        legacy->ranks[3].transient_bytes=345678u;
        legacy->ranks[3].kv_bytes=456789u;
        legacy->ranks[3].driver_reserve_bytes=567890u;
        legacy->ranks[3].tensor_begin=7u;
        legacy->ranks[3].tensor_count=9u;
        CHECK(fg_manifest_write(legacy_path,legacy,&error)==FG_OK);
        fg_session_identity before={0},after={0};
        CHECK(fg_session_identity_from_manifest(legacy,&before,&error)==FG_OK);
        CHECK(fg_manifest_upgrade(legacy_path,legacy_path,&error)==FG_ERR_ARGUMENT);
        CHECK(fg_manifest_upgrade(legacy_path,current_path,&error)==FG_OK);
        CHECK(fg_manifest_read(current_path,upgraded,&error)==FG_OK);
        CHECK(upgraded->format_version==FG_MANIFEST_FORMAT_VERSION);
        CHECK(upgraded->protocol_version==FG_PROTOCOL_VERSION);
        CHECK(upgraded->session.minimum_protocol_version==FG_PROTOCOL_VERSION);
        CHECK(fg_manifest_validate_deployment(upgraded,&error)==FG_OK);
        CHECK(fg_manifest_validate_compatibility(upgraded,FG_PROTOCOL_VERSION,
                                                  FG_POSITION_TEXT,&error)==FG_OK);
        CHECK(fg_manifest_validate_compatibility(upgraded,FG_PROTOCOL_MIN_VERSION,
                                                  FG_POSITION_TEXT,&error)==FG_ERR_MISMATCH);
        CHECK(upgraded->required_cu==legacy->required_cu);
        CHECK(upgraded->persistent_cap_bytes==legacy->persistent_cap_bytes);
        CHECK(upgraded->residency_cap_bytes==legacy->residency_cap_bytes);
        CHECK(upgraded->prefill_microbatch==FG_DEFAULT_MICROBATCH);
        CHECK(upgraded->max_context==FG_MAX_CONTEXT);
        CHECK(upgraded->session.logical_context_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS);
        CHECK(upgraded->session.gpu_index_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS);
        CHECK(upgraded->session.qsa_hot_record_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS);
        CHECK(upgraded->session.host_page_cache_bytes==0u);
        fg_runtime_options resolved={0};
        CHECK(fg_runtime_options_resolve(&resolved,upgraded,NULL,&error)==FG_OK);
        CHECK(resolved.logical_context_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS&&
              resolved.gpu_index_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS&&
              resolved.qsa_hot_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS&&
              resolved.qsa_page_cache_bytes==0u&&
              resolved.prefill_microbatch==FG_DEFAULT_MICROBATCH);
        CHECK(upgraded->tensor_count==legacy->tensor_count);
        CHECK(upgraded->ranks[3].state_file_bytes==123456u);
        CHECK(memcmp(upgraded->ranks,legacy->ranks,sizeof(legacy->ranks))==0);
        CHECK(memcmp(upgraded->layer_owner,legacy->layer_owner,sizeof(legacy->layer_owner))==0);
        CHECK(memcmp(upgraded->layer_groups,legacy->layer_groups,sizeof(legacy->layer_groups))==0);
        CHECK(memcmp(upgraded->expert_rank,legacy->expert_rank,sizeof(legacy->expert_rank))==0);
        CHECK(memcmp(upgraded->tensors,legacy->tensors,sizeof(legacy->tensors))==0);
        CHECK(memcmp(upgraded->source_sha256,legacy->source_sha256,32u)==0);
        CHECK(memcmp(upgraded->quant_profile_sha256,legacy->quant_profile_sha256,32u)==0);
        CHECK(memcmp(upgraded->manifest_sha256,legacy->manifest_sha256,32u)!=0);
        CHECK(fg_session_identity_from_manifest(upgraded,&after,&error)==FG_OK);
        CHECK(memcmp(before.model_sha256,after.model_sha256,32u)==0);
        CHECK(memcmp(before.tokenizer_sha256,after.tokenizer_sha256,32u)==0);
        CHECK(memcmp(before.quantization_sha256,after.quantization_sha256,32u)==0);
        CHECK(memcmp(before.rope_policy_sha256,after.rope_policy_sha256,32u)==0);
        CHECK(memcmp(before.state_format_sha256,after.state_format_sha256,32u)==0);

        CHECK(fg_manifest_upgrade_with_profile(
                  legacy_path,profile_path,
                  FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128,&error)==FG_OK);
        CHECK(fg_manifest_read(profile_path,profiled,&error)==FG_OK);
        CHECK(profiled->format_version==FG_MANIFEST_FORMAT_VERSION);
        CHECK(profiled->protocol_version==FG_PROTOCOL_VERSION);
        CHECK(profiled->prefill_microbatch==128u);
        CHECK(profiled->prefill_window==FG_DEFAULT_WINDOW);
        CHECK(profiled->max_context==FG_NATIVE_CONTEXT);
        CHECK(profiled->session.logical_context_tokens==FG_NATIVE_CONTEXT);
        CHECK(profiled->session.gpu_index_tokens==FG_NATIVE_CONTEXT);
        CHECK(profiled->session.qsa_hot_record_tokens==0u);
        CHECK(profiled->session.host_page_cache_bytes==
              FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES);
        CHECK(fg_manifest_validate_deployment(profiled,&error)==FG_OK);
        CHECK(fg_manifest_validate_compatibility(profiled,FG_PROTOCOL_VERSION,
                                                  FG_POSITION_TEXT,&error)==FG_OK);
        CHECK(fg_runtime_options_resolve(&resolved,profiled,NULL,&error)==FG_OK);
        CHECK(resolved.logical_context_tokens==FG_NATIVE_CONTEXT&&
              resolved.gpu_index_tokens==FG_NATIVE_CONTEXT&&
              resolved.qsa_hot_tokens==0u&&
              resolved.qsa_page_cache_bytes==
                  FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES&&
              resolved.prefill_microbatch==128u);
        CHECK(memcmp(profiled->tensors,legacy->tensors,sizeof(legacy->tensors))==0);
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            CHECK(strcmp(profiled->ranks[rank].endpoint,legacy->ranks[rank].endpoint)==0);
            CHECK(profiled->ranks[rank].persistent_bytes==
                  legacy->ranks[rank].persistent_bytes);
            CHECK(profiled->ranks[rank].transient_bytes==
                  legacy->ranks[rank].transient_bytes);
            CHECK(profiled->ranks[rank].driver_reserve_bytes==
                  legacy->ranks[rank].driver_reserve_bytes);
            CHECK(profiled->ranks[rank].tensor_begin==legacy->ranks[rank].tensor_begin);
            CHECK(profiled->ranks[rank].tensor_count==legacy->ranks[rank].tensor_count);
        }
        fg_manifest expected_profile=*legacy;
        fg_error profile_error={0};
        CHECK(fg_runtime_profile_apply(
                  &expected_profile,
                  FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128,
                  &profile_error)==FG_OK);
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
            CHECK(profiled->ranks[rank].scratch_bytes==
                  expected_profile.ranks[rank].scratch_bytes);
            CHECK(profiled->ranks[rank].kv_bytes==expected_profile.ranks[rank].kv_bytes);
            CHECK(profiled->ranks[rank].state_file_bytes==
                  expected_profile.ranks[rank].state_file_bytes);
        }
        CHECK(memcmp(profiled->session.rank_state_format_sha256,
                     upgraded->session.rank_state_format_sha256,
                     sizeof(profiled->session.rank_state_format_sha256))!=0);
        CHECK(memcmp(profiled->manifest_sha256,upgraded->manifest_sha256,32u)!=0);
        uint32_t parsed_profile=FG_RUNTIME_PROFILE_NONE;
        CHECK(fg_runtime_profile_parse(
                  FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128_NAME,
                  &parsed_profile,&error)==FG_OK&&
              parsed_profile==FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128);
        CHECK(fg_runtime_profile_parse("unknown-profile",&parsed_profile,&error)==
              FG_ERR_ARGUMENT);
        fg_manifest incompatible=*profiled;
        incompatible.session.position_mode=FG_POSITION_FOUR_AXIS;
        CHECK(fg_runtime_profile_apply(
                  &incompatible,FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128,
                  &error)==FG_ERR_MISMATCH);
    }
    free(profiled);free(upgraded);free(legacy);
    unlink(profile_path);unlink(current_path);unlink(legacy_path);
}

static void test_deployment_admission(void){
    char path[96];snprintf(path,sizeof(path),"test-session-four-axis-%ld.fgm",(long)getpid());
    unlink(path);fg_manifest *manifest=malloc(sizeof(*manifest)),*sealed=malloc(sizeof(*sealed));
    CHECK(manifest&&sealed);fg_error error={0};
    if(manifest&&sealed){
        fg_manifest_init(manifest);manifest->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
        manifest->session.position_mode=FG_POSITION_FOUR_AXIS;
        fill_digest(manifest->source_sha256,29u);
        CHECK(fg_manifest_write(path,manifest,&error)==FG_OK);
        CHECK(fg_manifest_read(path,sealed,&error)==FG_OK);
        CHECK(sealed->session.position_mode==FG_POSITION_FOUR_AXIS);
        CHECK(fg_manifest_validate(sealed,&error)==FG_OK);
        CHECK(fg_manifest_validate_deployment(sealed,&error)==FG_ERR_UNAVAILABLE);
    }
    free(sealed);free(manifest);unlink(path);
}

static void test_legacy_identity_roundtrip(void){
    char path[96];snprintf(path,sizeof(path),"test-session-v4-identity-%ld.fgm",(long)getpid());
    unlink(path);fg_manifest *manifest=malloc(sizeof(*manifest)),*sealed=malloc(sizeof(*sealed));
    CHECK(manifest&&sealed);fg_error error={0};
    if(manifest&&sealed){
        fg_manifest_init(manifest);manifest->format_version=FG_MANIFEST_LEGACY_FORMAT_VERSION;
        manifest->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
        fill_digest(manifest->source_sha256,31u);
        CHECK(fg_manifest_write(path,manifest,&error)==FG_OK);
        CHECK(manifest->session.minimum_protocol_version==FG_PROTOCOL_MIN_VERSION);
        fg_session_identity before={0},after={0};
        uint8_t before_wire[FG_SESSION_IDENTITY_WIRE_BYTES];
        uint8_t after_wire[FG_SESSION_IDENTITY_WIRE_BYTES];
        CHECK(fg_session_identity_from_manifest(manifest,&before,&error)==FG_OK);
        CHECK(fg_session_identity_encode(before_wire,&before,&error)==FG_OK);
        CHECK(fg_manifest_read(path,sealed,&error)==FG_OK);
        CHECK(memcmp(&manifest->session,&sealed->session,sizeof(manifest->session))==0);
        CHECK(fg_session_identity_from_manifest(sealed,&after,&error)==FG_OK);
        CHECK(fg_session_identity_encode(after_wire,&after,&error)==FG_OK);
        CHECK(memcmp(before_wire,after_wire,sizeof(before_wire))==0);
    }
    free(sealed);free(manifest);unlink(path);
}

static fg_manifest *write_read_budget_manifest(const char *path,uint32_t logical,
                                               uint32_t gpu,uint32_t hot,uint64_t page){
    fg_manifest *manifest=malloc(sizeof(*manifest)),*sealed=malloc(sizeof(*sealed));
    CHECK(manifest&&sealed);if(!manifest||!sealed){free(sealed);free(manifest);return NULL;}
    fg_manifest_init(manifest);manifest->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
    manifest->session.logical_context_tokens=logical;
    manifest->session.gpu_index_tokens=gpu;
    manifest->session.qsa_hot_record_tokens=hot;
    manifest->session.host_page_cache_bytes=page;
    fill_digest(manifest->source_sha256,37u);
    fg_error error={0};CHECK(fg_manifest_write(path,manifest,&error)==FG_OK);
    CHECK(fg_manifest_read(path,sealed,&error)==FG_OK);free(manifest);
    if(error.code!=FG_OK){free(sealed);return NULL;}return sealed;
}

static void test_runtime_option_contract(void){
    char current_path[96],large_path[96],tiered_path[96],legacy_path[96];
    long pid=(long)getpid();
    snprintf(current_path,sizeof(current_path),"test-session-budget-current-%ld.fgm",pid);
    snprintf(large_path,sizeof(large_path),"test-session-budget-large-%ld.fgm",pid);
    snprintf(tiered_path,sizeof(tiered_path),"test-session-budget-tiered-%ld.fgm",pid);
    snprintf(legacy_path,sizeof(legacy_path),"test-session-budget-legacy-%ld.fgm",pid);
    unlink(current_path);unlink(large_path);unlink(tiered_path);unlink(legacy_path);
    fg_manifest *current=write_read_budget_manifest(current_path,8192u,8192u,8192u,0u);
    fg_manifest *large=write_read_budget_manifest(large_path,16384u,16384u,8192u,0u);
    fg_manifest *tiered=write_read_budget_manifest(tiered_path,32768u,32768u,0u,
                                                   UINT64_C(64)<<20u);
    fg_manifest *legacy=malloc(sizeof(*legacy)),*decoded=malloc(sizeof(*decoded));
    CHECK(legacy&&decoded);fg_error error={0};fg_runtime_options resolved={0},requested={0};
    if(current){
        CHECK(fg_runtime_options_resolve(&resolved,current,NULL,&error)==FG_OK);
        CHECK(resolved.logical_context_tokens==8192u&&resolved.gpu_index_tokens==8192u&&
              resolved.qsa_hot_tokens==8192u&&resolved.qsa_page_cache_bytes==0u);
        uint32_t qsa_capacity=0u;
        CHECK(fg_runtime_eval_capacity(&qsa_capacity,&resolved,4096u,4096u,
                                       &error)==FG_OK);
        CHECK(qsa_capacity==8192u);
        CHECK(fg_runtime_eval_capacity(&qsa_capacity,&resolved,8192u,0u,
                                       &error)==FG_OK);
        CHECK(qsa_capacity==8192u);
        CHECK(fg_runtime_eval_capacity(&qsa_capacity,&resolved,8192u,1u,
                                       &error)==FG_ERR_LIMIT);
        CHECK(qsa_capacity==0u);
        CHECK(fg_runtime_eval_capacity(&qsa_capacity,&resolved,8193u,0u,
                                       &error)==FG_ERR_LIMIT);
        fg_runtime_options_init(&requested);
        requested.logical_context_tokens=8192u;requested.gpu_index_tokens=8192u;
        requested.qsa_hot_tokens=8192u;requested.qsa_page_cache_bytes=0u;
        requested.specified=FG_RUNTIME_OPTION_LOGICAL_CONTEXT|FG_RUNTIME_OPTION_GPU_INDEX|
            FG_RUNTIME_OPTION_QSA_HOT|FG_RUNTIME_OPTION_PAGE_CACHE;
        CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_OK);
        requested.qsa_page_cache_bytes=1u;
        CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_ERR_MISMATCH);
        fg_runtime_options_init(&requested);requested.logical_context_tokens=4096u;
        CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_ERR_ARGUMENT);
        static const uint32_t stages[]={32768u,65536u,131072u,262144u};
        for(uint32_t i=0;i<sizeof(stages)/sizeof(stages[0]);i++){
            fg_runtime_options_init(&requested);
            requested.logical_context_tokens=stages[i];
            requested.gpu_index_tokens=stages[i];
            requested.qsa_hot_tokens=0u;
            requested.qsa_page_cache_bytes=UINT64_C(64)<<20u;
            requested.specified=FG_RUNTIME_OPTION_LOGICAL_CONTEXT|FG_RUNTIME_OPTION_GPU_INDEX|
                FG_RUNTIME_OPTION_QSA_HOT|FG_RUNTIME_OPTION_PAGE_CACHE;
            CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_OK);
            CHECK(resolved.logical_context_tokens==stages[i]&&
                  resolved.gpu_index_tokens==stages[i]&&resolved.qsa_hot_tokens==0u);
        }
        requested.gpu_index_tokens=131072u;
        CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_ERR_MISMATCH);
        requested.gpu_index_tokens=262144u;requested.qsa_hot_tokens=4096u;
        CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_ERR_MISMATCH);
        requested.qsa_hot_tokens=0u;requested.qsa_page_cache_bytes=UINT64_C(8)<<20u;
        CHECK(fg_runtime_options_resolve(&resolved,current,&requested,&error)==FG_ERR_ARGUMENT);
    }
    if(large){
        CHECK(fg_manifest_validate_deployment(large,&error)==FG_ERR_UNAVAILABLE);
        CHECK(fg_runtime_options_resolve(&resolved,large,NULL,&error)==FG_ERR_UNAVAILABLE);
        CHECK(resolved.logical_context_tokens==16384u&&resolved.gpu_index_tokens==16384u&&
              resolved.qsa_hot_tokens==8192u);
    }
    if(tiered){
        CHECK(fg_manifest_validate_deployment(tiered,&error)==FG_OK);
        CHECK(fg_runtime_options_resolve(&resolved,tiered,NULL,&error)==FG_OK);
        CHECK(resolved.qsa_hot_tokens==0u&&
              resolved.qsa_page_cache_bytes==(UINT64_C(64)<<20u));
        fg_runtime_options_init(&requested);
        requested.specified=FG_RUNTIME_OPTION_PAGE_CACHE;
        CHECK(fg_runtime_options_resolve(&resolved,tiered,&requested,&error)==FG_ERR_ARGUMENT);
    }
    if(legacy&&decoded){
        fg_manifest_init(legacy);legacy->flags=FG_MANIFEST_COMPONENTS_TEXT_REQUIRED;
        fill_digest(legacy->source_sha256,41u);write_legacy_manifest(legacy_path,legacy);
        CHECK(fg_manifest_read(legacy_path,decoded,&error)==FG_OK);
        CHECK(fg_runtime_options_resolve(&resolved,decoded,NULL,&error)==FG_OK);
        CHECK(resolved.logical_context_tokens==8192u&&resolved.gpu_index_tokens==8192u&&
              resolved.qsa_hot_tokens==8192u&&resolved.qsa_page_cache_bytes==0u);
        uint32_t qsa_capacity=0u;
        CHECK(fg_runtime_eval_capacity(&qsa_capacity,&resolved,8192u,0u,
                                       &error)==FG_OK);
        CHECK(qsa_capacity==8192u);
        CHECK(fg_runtime_eval_capacity(&qsa_capacity,&resolved,8192u,1u,
                                       &error)==FG_ERR_LIMIT);
        fg_runtime_options_init(&requested);
        requested.logical_context_tokens=4096u;requested.gpu_index_tokens=4096u;
        requested.qsa_hot_tokens=4096u;
        CHECK(fg_runtime_options_resolve(&resolved,decoded,&requested,&error)==FG_ERR_ARGUMENT);
    }
    free(decoded);free(legacy);free(tiered);free(large);free(current);
    unlink(legacy_path);unlink(tiered_path);unlink(large_path);unlink(current_path);
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
            .session_nonce=UINT64_C(0x1122334455667788),
            .logical_context_tokens=8192u,.gpu_index_tokens=8192u,
            .qsa_hot_tokens=8192u};
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
        CHECK(decoded.operation==operation&&decoded.rank==3u&&
              decoded.generation==control.generation&&
              decoded.logical_context_tokens==8192u);
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
    CHECK(fg_frame_validate_version(&header,FG_PROTOCOL_MIN_VERSION,payload,NULL,
                                    &error)==FG_OK);
    CHECK(fg_frame_validate_version(&header,FG_PROTOCOL_VERSION,payload,NULL,
                                    &error)==FG_ERR_MISMATCH);
    CHECK(fg_frame_encode_version(&header,FG_PROTOCOL_MIN_VERSION,FG_MSG_SESSION_PREPARE,
                                  9u,1u,0u,payload,sizeof(payload),&error)==FG_ERR_ARGUMENT);
    CHECK(fg_frame_encode_version(&header,FG_PROTOCOL_VERSION,FG_MSG_SESSION_PREPARE,
                                  9u,1u,0u,payload,sizeof(payload),&error)==FG_OK);
    CHECK(fg_frame_validate(&header,payload,NULL,&error)==FG_OK);
    CHECK(fg_frame_validate_version(&header,FG_PROTOCOL_VERSION,payload,NULL,
                                    &error)==FG_OK);
    CHECK(fg_frame_validate_version(&header,FG_PROTOCOL_MIN_VERSION,payload,NULL,
                                    &error)==FG_ERR_MISMATCH);

    fg_layer_work *work=calloc(1,sizeof(*work)),*decoded=calloc(1,sizeof(*decoded));
    uint8_t *wire=malloc(FG_LAYER_WORK_MAX_BYTES),*again=malloc(FG_LAYER_WORK_MAX_BYTES);
    CHECK(work&&decoded&&wire&&again);
    if(work&&decoded&&wire&&again){
        work->layer=2u;work->source_rank=0u;work->destination_rank=2u;
        work->position_mode=FG_POSITION_FOUR_AXIS;work->token_index=81u;
        work->position[0]=5u;work->position[1]=7u;work->position[2]=11u;work->position[3]=13u;
        uint32_t bytes=0,again_bytes=0;
        CHECK(fg_layer_work_encode(wire,FG_LAYER_WORK_MAX_BYTES,&bytes,
                                   FG_PROTOCOL_MIN_VERSION,work,&error)==FG_ERR_MISMATCH);

        work->position_mode=FG_POSITION_TEXT;work->position[3]=0u;
        CHECK(fg_layer_work_encode(wire,FG_LAYER_WORK_MAX_BYTES,&bytes,
                                   FG_PROTOCOL_MIN_VERSION,work,&error)==FG_OK);
        CHECK(bytes==FG_LAYER_WORK_LEGACY_HEADER_BYTES+FG_HYPER_WIDTH*4u);
        uint8_t legacy_header[FG_LAYER_WORK_LEGACY_HEADER_BYTES]={0};
        legacy_header[0]=2u;legacy_header[2]=2u;
        put_u32_be(legacy_header+4u,81u);put_u32_be(legacy_header+8u,5u);
        put_u32_be(legacy_header+12u,7u);put_u32_be(legacy_header+16u,11u);
        CHECK(memcmp(wire,legacy_header,sizeof(legacy_header))==0);
        CHECK(fg_layer_work_encode(again,FG_LAYER_WORK_MAX_BYTES,&again_bytes,
                                   FG_PROTOCOL_MIN_VERSION,work,&error)==FG_OK);
        CHECK(again_bytes==bytes&&memcmp(wire,again,bytes)==0);
        CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_MIN_VERSION,wire,bytes,&error)==FG_OK);
        CHECK(decoded->position_mode==FG_POSITION_TEXT&&decoded->position[2]==11u);
        CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_VERSION,wire,bytes,
                                   &error)==FG_ERR_FORMAT);

        CHECK(fg_layer_work_encode(wire,FG_LAYER_WORK_MAX_BYTES,&bytes,FG_PROTOCOL_VERSION,
                                   work,&error)==FG_OK);
        CHECK(bytes==FG_LAYER_WORK_BASE_BYTES);
        CHECK(wire[8]==FG_POSITION_TEXT&&wire[9]==3u&&!wire[10]&&!wire[11]);
        CHECK(fg_layer_work_encode(again,FG_LAYER_WORK_MAX_BYTES,&again_bytes,
                                   FG_PROTOCOL_VERSION,work,&error)==FG_OK);
        CHECK(again_bytes==bytes&&memcmp(wire,again,bytes)==0);
        CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_VERSION,wire,bytes,&error)==FG_OK);
        CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_MIN_VERSION,wire,bytes,
                                   &error)==FG_ERR_FORMAT);

        work->position_mode=FG_POSITION_FOUR_AXIS;work->position[3]=13u;
        CHECK(fg_layer_work_encode(wire,FG_LAYER_WORK_MAX_BYTES,&bytes,FG_PROTOCOL_VERSION,
                                   work,&error)==FG_OK);
        CHECK(bytes==FG_LAYER_WORK_FOUR_AXIS_BASE_BYTES);
        CHECK(fg_layer_work_decode(decoded,FG_PROTOCOL_VERSION,wire,bytes,&error)==FG_OK);
        CHECK(decoded->position_mode==FG_POSITION_FOUR_AXIS&&decoded->position[3]==13u);
    }
    free(again);free(wire);free(decoded);free(work);

    enum{TOKENS=2};uint32_t positions[TOKENS*4u],decoded_positions[TOKENS*4u];
    float *hyper=calloc((size_t)TOKENS*FG_HYPER_WIDTH,sizeof(*hyper));
    float *decoded_hyper=calloc((size_t)TOKENS*FG_HYPER_WIDTH,sizeof(*decoded_hyper));
    uint8_t *prefill_wire=malloc(FG_PREFILL_LAYER_WORK_MAX_BYTES);
    uint8_t *prefill_again=malloc(FG_PREFILL_LAYER_WORK_MAX_BYTES);
    CHECK(hyper&&decoded_hyper&&prefill_wire&&prefill_again);
    if(hyper&&decoded_hyper&&prefill_wire&&prefill_again){
        for(uint32_t i=0;i<TOKENS*4u;i++)positions[i]=100u+i;
        fg_prefill_layer_work work_batch={.layer=2u,.source_rank=0u,.destination_rank=2u,
            .position_mode=FG_POSITION_FOUR_AXIS,.first_token=100u,.token_count=TOKENS,
            .positions=positions,.hyper=hyper},decoded_batch={0};
        uint32_t bytes=0,again_bytes=0;
        CHECK(fg_prefill_layer_work_encode(prefill_wire,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &bytes,FG_PROTOCOL_MIN_VERSION,&work_batch,
                                           &error)==FG_ERR_MISMATCH);

        work_batch.position_mode=FG_POSITION_TEXT;
        CHECK(fg_prefill_layer_work_encode(prefill_wire,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &bytes,FG_PROTOCOL_MIN_VERSION,&work_batch,
                                           &error)==FG_OK);
        CHECK(bytes==FG_PREFILL_LAYER_HEADER_BYTES+TOKENS*3u*4u+
                     TOKENS*FG_HYPER_WIDTH*4u);
        uint8_t legacy_header[FG_PREFILL_LAYER_HEADER_BYTES]={0};
        legacy_header[0]=2u;legacy_header[2]=2u;legacy_header[9]=TOKENS;
        put_u32_be(legacy_header+4u,100u);
        CHECK(memcmp(prefill_wire,legacy_header,sizeof(legacy_header))==0);
        CHECK(fg_prefill_layer_work_encode(prefill_again,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &again_bytes,FG_PROTOCOL_MIN_VERSION,&work_batch,
                                           &error)==FG_OK);
        CHECK(again_bytes==bytes&&memcmp(prefill_wire,prefill_again,bytes)==0);
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,FG_PROTOCOL_MIN_VERSION,
                                           decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_OK);
        CHECK(decoded_batch.position_mode==FG_POSITION_TEXT&&decoded_positions[5]==105u);
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,FG_PROTOCOL_VERSION,
                                           decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_ERR_FORMAT);

        CHECK(fg_prefill_layer_work_encode(prefill_wire,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &bytes,FG_PROTOCOL_VERSION,&work_batch,
                                           &error)==FG_OK);
        CHECK(prefill_wire[10]==FG_POSITION_TEXT&&prefill_wire[11]==3u);
        CHECK(fg_prefill_layer_work_encode(prefill_again,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &again_bytes,FG_PROTOCOL_VERSION,&work_batch,
                                           &error)==FG_OK);
        CHECK(again_bytes==bytes&&memcmp(prefill_wire,prefill_again,bytes)==0);
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,FG_PROTOCOL_MIN_VERSION,
                                           decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_ERR_FORMAT);

        work_batch.position_mode=FG_POSITION_FOUR_AXIS;
        CHECK(fg_prefill_layer_work_encode(prefill_wire,FG_PREFILL_LAYER_WORK_MAX_BYTES,
                                           &bytes,FG_PROTOCOL_VERSION,&work_batch,
                                           &error)==FG_OK);
        CHECK(bytes==FG_PREFILL_LAYER_HEADER_BYTES+TOKENS*4u*4u+
                     TOKENS*FG_HYPER_WIDTH*4u);
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,FG_PROTOCOL_VERSION,
                                           decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_OK);
        CHECK(decoded_batch.position_mode==FG_POSITION_FOUR_AXIS&&
              decoded_positions[TOKENS*4u-1u]==positions[TOKENS*4u-1u]);
        prefill_wire[11]=3u;
        CHECK(fg_prefill_layer_work_decode(&decoded_batch,FG_PROTOCOL_VERSION,
                                           decoded_positions,TOKENS*4u,
                                           decoded_hyper,(uint64_t)TOKENS*FG_HYPER_WIDTH,
                                           NULL,0u,prefill_wire,bytes,&error)==FG_ERR_FORMAT);
    }
    free(prefill_again);free(prefill_wire);free(decoded_hyper);free(hyper);
}

static void test_qsa_owner_contract(void){
    char path[96];snprintf(path,sizeof(path),"test-qsa-owner-%ld.fgm",(long)getpid());
    unlink(path);fg_manifest *manifest=write_read_manifest(path,83u);fg_error error={0};
    if(!manifest){unlink(path);return;}
    fg_session_identity identity={0};
    CHECK(fg_session_identity_from_manifest(manifest,&identity,&error)==FG_OK);

    fg_qsa_owner_guard rank3;fg_qsa_owner_guard_init(&rank3,3u);
    fg_owner_session_control control={.version=FG_OWNER_SESSION_CONTROL_VERSION,
        .operation=FG_OWNER_SESSION_BEGIN,.rank=3u,.position_mode=FG_POSITION_TEXT,
        .session_nonce=101u,.logical_context_tokens=8192u,.gpu_index_tokens=8192u,
        .qsa_hot_tokens=8192u};
    memcpy(control.identity_sha256,identity.identity_sha256,32u);
    memcpy(control.state_format_sha256,manifest->session.rank_state_format_sha256[3],32u);
    CHECK(fg_qsa_owner_guard_begin(&rank3,manifest,&identity,101u,&control,&error)==FG_OK);
    CHECK(rank3.active&&rank3.generation==1u);

    float *decode_hidden=malloc(FG_HIDDEN_SIZE*4u),*decoded_hidden=malloc(FG_HIDDEN_SIZE*4u);
    fg_qsa_block_work decode={.layer=3u,.source_rank=0u,.destination_rank=3u,
        .position_mode=FG_POSITION_TEXT,.token_index=0u,.position={0u,0u,0u,0u},
        .hidden=decode_hidden};
    for(uint32_t i=0;decode_hidden&&i<FG_HIDDEN_SIZE;i++)decode_hidden[i]=(float)((int32_t)(i%31u)-15)*0.125f;
    uint8_t *decode_wire=malloc(FG_QSA_BLOCK_WORK_MAX_BYTES);
    fg_qsa_block_work decoded_work={0};uint32_t decode_bytes=0;
    CHECK(decode_hidden&&decoded_hidden&&decode_wire);
    if(decode_hidden&&decoded_hidden&&decode_wire){
        CHECK(fg_qsa_block_work_encode(decode_wire,FG_QSA_BLOCK_WORK_MAX_BYTES,&decode_bytes,
                                       manifest->protocol_version,&decode,&error)==FG_OK);
        CHECK(decode_bytes==FG_QSA_BLOCK_WORK_TEXT_BYTES);
        CHECK(fg_qsa_block_work_decode(&decoded_work,manifest->protocol_version,decoded_hidden,
                                       FG_HIDDEN_SIZE,decode_wire,decode_bytes,&error)==FG_OK);
        CHECK(memcmp(decoded_hidden,decode_hidden,FG_HIDDEN_SIZE*4u)==0);
    }
    free(decode_wire);
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,3u,&decode,&error)==FG_OK);
    CHECK(fg_qsa_owner_guard_commit(&rank3,3u,0u,1u,&error)==FG_OK);
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,3u,&decode,
                                             &error)==FG_ERR_MISMATCH);
    decode.token_index=2u;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,
        2u*FG_LAYER_COUNT+3u,&decode,&error)==FG_ERR_MISMATCH);
    decode.token_index=1u;decode.destination_rank=7u;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,
        FG_LAYER_COUNT+3u,&decode,&error)==FG_ERR_MISMATCH);
    decode.destination_rank=3u;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,
        FG_LAYER_COUNT+3u,&decode,&error)==FG_ERR_MISMATCH);
    decode.position[0]=decode.position[1]=decode.position[2]=1u;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,
        FG_LAYER_COUNT+3u,&decode,&error)==FG_OK);
    decode.position[1]=0u;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,
        FG_LAYER_COUNT+3u,&decode,&error)==FG_ERR_MISMATCH);
    decode.position[1]=1u;decode.position_mode=FG_POSITION_FOUR_AXIS;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,
        FG_LAYER_COUNT+3u,&decode,&error)==FG_ERR_MISMATCH);
    decode.position_mode=FG_POSITION_TEXT;

    control.session_nonce=99u;
    CHECK(fg_qsa_owner_guard_begin(&rank3,manifest,&identity,99u,&control,
                                   &error)==FG_ERR_MISMATCH);
    control.session_nonce=202u;
    CHECK(fg_qsa_owner_guard_begin(&rank3,manifest,&identity,202u,&control,&error)==FG_OK);
    decode.token_index=0u;decode.position[0]=decode.position[1]=decode.position[2]=0u;
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,101u,3u,&decode,
                                             &error)==FG_ERR_MISMATCH);
    CHECK(fg_qsa_owner_guard_validate_decode(&rank3,manifest,202u,3u,&decode,&error)==FG_OK);

    fg_qsa_block_result decode_result={.layer=3u,.source_rank=3u,.destination_rank=0u,
        .token_index=0u,.hidden=decode_hidden};
    uint8_t result_wire[FG_QSA_BLOCK_RESULT_BYTES];fg_qsa_block_result decoded_result={0};
    CHECK(fg_qsa_block_result_encode(result_wire,&decode_result,&error)==FG_OK);
    CHECK(fg_qsa_block_result_decode(&decoded_result,decoded_hidden,FG_HIDDEN_SIZE,
                                     result_wire,sizeof(result_wire),&error)==FG_OK);
    CHECK(memcmp(decoded_hidden,decode_hidden,FG_HIDDEN_SIZE*4u)==0);
    CHECK(fg_qsa_owner_validate_decode_result(manifest,3u,3u,&decode_result,&error)==FG_OK);
    decode_result.hidden=NULL;
    CHECK(fg_qsa_owner_validate_decode_result(manifest,3u,3u,&decode_result,
                                              &error)==FG_ERR_MISMATCH);
    free(decoded_hidden);free(decode_hidden);

    fg_qsa_owner_guard rank7;fg_qsa_owner_guard_init(&rank7,7u);
    control.rank=7u;control.session_nonce=303u;
    memcpy(control.state_format_sha256,manifest->session.rank_state_format_sha256[7],32u);
    CHECK(fg_qsa_owner_guard_begin(&rank7,manifest,&identity,303u,&control,&error)==FG_OK);
    uint32_t positions[6]={0u,0u,0u,1u,1u,1u};
    float *hidden=calloc(2u*FG_HIDDEN_SIZE,sizeof(*hidden));
    CHECK(hidden!=NULL);
    if(hidden){
        fg_qsa_block_prefill_work prefill={.layer=7u,.source_rank=0u,.destination_rank=7u,
            .position_mode=FG_POSITION_TEXT,.first_token=0u,.token_count=2u,
            .positions=positions,.hidden=hidden};
        CHECK(fg_qsa_owner_guard_validate_prefill(&rank7,manifest,303u,7u,&prefill,
                                                  &error)==FG_OK);
        positions[4]=0u;
        CHECK(fg_qsa_owner_guard_validate_prefill(&rank7,manifest,303u,7u,&prefill,
                                                  &error)==FG_ERR_MISMATCH);
        positions[4]=1u;prefill.position_mode=FG_POSITION_FOUR_AXIS;
        CHECK(fg_qsa_owner_guard_validate_prefill(&rank7,manifest,303u,7u,&prefill,
                                                  &error)==FG_ERR_MISMATCH);
        prefill.position_mode=FG_POSITION_TEXT;prefill.positions=NULL;
        CHECK(fg_qsa_owner_guard_validate_prefill(&rank7,manifest,303u,7u,&prefill,
                                                  &error)==FG_ERR_FORMAT);
        prefill.positions=positions;
        CHECK(fg_qsa_owner_guard_commit(&rank7,7u,0u,2u,&error)==FG_OK);
        CHECK(fg_qsa_owner_guard_validate_prefill(&rank7,manifest,303u,7u,&prefill,
                                                  &error)==FG_ERR_MISMATCH);
        fg_qsa_block_prefill_result result={.layer=7u,.source_rank=7u,.destination_rank=0u,
            .first_token=0u,.token_count=2u,.hidden=hidden};
        CHECK(fg_qsa_owner_validate_prefill_result(manifest,7u,7u,2u,&result,
                                                   &error)==FG_OK);
        CHECK(fg_qsa_owner_validate_prefill_result(manifest,7u,7u,1u,&result,
                                                   &error)==FG_ERR_MISMATCH);
        result.hidden=NULL;
        CHECK(fg_qsa_owner_validate_prefill_result(manifest,7u,7u,2u,&result,
                                                   &error)==FG_ERR_MISMATCH);
        free(hidden);
    }
    free(manifest);unlink(path);
}

static void test_qsa_page_owner_contract(void){
    char path[96];snprintf(path,sizeof(path),"test-qsa-page-owner-%ld.fgm",(long)getpid());
    unlink(path);fg_manifest *manifest=write_read_manifest(path,91u);fg_error error={0};
    if(!manifest){unlink(path);return;}
    fg_session_identity identity={0};
    CHECK(fg_session_identity_from_manifest(manifest,&identity,&error)==FG_OK);
    fg_owner_session_control control={.version=FG_OWNER_SESSION_CONTROL_VERSION,
        .operation=FG_OWNER_SESSION_BEGIN,.rank=3u,.position_mode=FG_POSITION_TEXT,
        .session_nonce=401u,.logical_context_tokens=8192u,.gpu_index_tokens=8192u,
        .qsa_hot_tokens=8192u};
    memcpy(control.identity_sha256,identity.identity_sha256,32u);
    memcpy(control.state_format_sha256,manifest->session.rank_state_format_sha256[3],32u);
    fg_qsa_owner_guard guard;fg_qsa_owner_guard_init(&guard,3u);
    CHECK(fg_qsa_owner_guard_begin(&guard,manifest,&identity,401u,&control,&error)==FG_OK);
    uint8_t records[2u*FG_QSA_PAGE_RECORD_BYTES];memset(records,0x5a,sizeof(records));
    fg_qsa_page pages[2]={
        {.layer=3u,.block=0u,.records=records},
        {.layer=11u,.block=0u,.records=records+FG_QSA_PAGE_RECORD_BYTES}
    };
    fg_qsa_page_batch append={.source_rank=0u,.destination_rank=3u,.batch_id=0u,
        .page_count=2u,.pages=pages};
    CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,401u,&append,&error)==FG_OK);
    CHECK(guard.next_token[3]==4u&&guard.next_token[11]==4u&&guard.next_append_batch==1u);
    CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,401u,&append,
                                            &error)==FG_ERR_MISMATCH);
    append.batch_id=1u;pages[0].block=2u;pages[1].block=2u;
    CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,401u,&append,
                                            &error)==FG_ERR_MISMATCH);
    pages[0].block=1u;pages[1].block=1u;
    CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,400u,&append,
                                            &error)==FG_ERR_MISMATCH);
    CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,401u,&append,&error)==FG_OK);
    CHECK(guard.next_token[3]==8u&&guard.next_token[11]==8u&&guard.next_append_batch==2u);
    fg_qsa_page fetch_page={.layer=3u,.block=0u,.records=NULL};
    fg_qsa_page_batch fetch={.source_rank=0u,.destination_rank=3u,.batch_id=0u,
        .page_count=1u,.pages=&fetch_page};
    CHECK(fg_qsa_owner_guard_accept_fetch(&guard,manifest,401u,&fetch,&error)==FG_OK);
    CHECK(fg_qsa_owner_guard_accept_fetch(&guard,manifest,401u,&fetch,
                                           &error)==FG_ERR_MISMATCH);
    fetch.batch_id=1u;fetch_page.block=2u;
    CHECK(fg_qsa_owner_guard_accept_fetch(&guard,manifest,401u,&fetch,
                                           &error)==FG_ERR_MISMATCH);
    fetch_page.layer=7u;fetch_page.block=0u;
    CHECK(fg_qsa_owner_guard_accept_fetch(&guard,manifest,401u,&fetch,
                                           &error)==FG_ERR_MISMATCH);
    fetch_page.layer=3u;fetch_page.block=1u;
    CHECK(fg_qsa_owner_guard_accept_fetch(&guard,manifest,401u,&fetch,&error)==FG_OK);
    fg_qsa_page_barrier barrier={.source_rank=0u,.destination_rank=3u,.batch_id=2u};
    CHECK(fg_qsa_owner_guard_accept_barrier(&guard,401u,&barrier,&error)==FG_OK);
    barrier.batch_id=1u;
    CHECK(fg_qsa_owner_guard_accept_barrier(&guard,401u,&barrier,
                                             &error)==FG_ERR_MISMATCH);
    control.session_nonce=402u;
    CHECK(fg_qsa_owner_guard_begin(&guard,manifest,&identity,402u,&control,&error)==FG_OK);
    CHECK(guard.next_token[3]==0u&&guard.next_append_batch==0u&&guard.next_fetch_batch==0u);
    fetch.batch_id=0u;fetch_page.block=0u;
    CHECK(fg_qsa_owner_guard_accept_fetch(&guard,manifest,401u,&fetch,
                                           &error)==FG_ERR_MISMATCH);
    pages[0].block=0u;pages[1].block=0u;append.batch_id=0u;
    CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,402u,&append,&error)==FG_OK);
    fg_qsa_page *oversized=calloc(FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES+1u,
                                  sizeof(*oversized));
    uint8_t *oversized_records=calloc(FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES+1u,
                                      FG_QSA_PAGE_RECORD_BYTES);
    CHECK(oversized&&oversized_records);
    if(oversized&&oversized_records){
        for(uint32_t i=0;i<=FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES;i++)
            oversized[i]=(fg_qsa_page){.layer=3u,.block=1u+i,
                .records=oversized_records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES};
        append.batch_id=1u;append.page_count=FG_QSA_PAGE_APPEND_LAYER_MAX_PAGES+1u;
        append.pages=oversized;
        CHECK(fg_qsa_owner_guard_accept_append(&guard,manifest,402u,&append,
                                               &error)==FG_ERR_MISMATCH);
        CHECK(guard.next_token[3]==4u&&guard.next_append_batch==1u);
    }
    free(oversized_records);free(oversized);
    free(manifest);unlink(path);
}

int main(void){
    test_manifest_evolution();test_manifest_upgrade();test_deployment_admission();
    test_legacy_identity_roundtrip();
    test_runtime_option_contract();
    test_identity_and_frontier();test_owner_controls();
    test_protocol_evolution();test_qsa_owner_contract();test_qsa_page_owner_contract();
    if(failures){fprintf(stderr,"%d session contract test(s) failed\n",failures);return 1;}
    puts("session contracts: PASS");return 0;
}
