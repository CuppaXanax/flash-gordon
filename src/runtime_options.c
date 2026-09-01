#include "fg_runtime.h"
#include "fg_q38_schema.h"
#include "fg_topology.h"

#include <string.h>

static const fg_runtime_profile_definition runtime_profiles[]={
    {
        FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128,
        FG_RUNTIME_PROFILE_NATIVE_262K_MICROBATCH_128_NAME,
        FG_NATIVE_CONTEXT,
        FG_NATIVE_CONTEXT,
        0u,
        FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES,
        128u,
        FG_DEFAULT_WINDOW,
        FG_NATIVE_CONTEXT,
        FG_POSITION_TEXT
    },
    {
        FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K,
        FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K_NAME,
        FG_NATIVE_CONTEXT,
        FG_NATIVE_CONTEXT,
        0u,
        FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES,
        FG_PIPELINE_DEFAULT_MICROBATCH,
        FG_DEFAULT_WINDOW,
        FG_NATIVE_CONTEXT,
        FG_POSITION_TEXT
    }
};

const fg_runtime_profile_definition *fg_runtime_profile_definition_get(uint32_t profile){
    for(uint32_t i=0;i<sizeof(runtime_profiles)/sizeof(runtime_profiles[0]);i++)
        if(runtime_profiles[i].id==profile)return &runtime_profiles[i];
    return NULL;
}

fg_status fg_runtime_profile_parse(const char *name,uint32_t *profile,fg_error *err){
    if(!name||!profile){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime profile name");
        return FG_ERR_ARGUMENT;
    }
    for(uint32_t i=0;i<sizeof(runtime_profiles)/sizeof(runtime_profiles[0]);i++)
        if(!strcmp(name,runtime_profiles[i].name)){
            *profile=runtime_profiles[i].id;
            return FG_OK;
        }
    fg_error_set(err,FG_ERR_ARGUMENT,"unsupported runtime profile: %s",name);
    return FG_ERR_ARGUMENT;
}

static uint32_t requested_mask(const fg_runtime_options *requested){
    if(!requested)return 0u;
    uint32_t mask=requested->specified;
    if(requested->logical_context_tokens)mask|=FG_RUNTIME_OPTION_LOGICAL_CONTEXT;
    if(requested->gpu_index_tokens)mask|=FG_RUNTIME_OPTION_GPU_INDEX;
    if(requested->qsa_hot_tokens)mask|=FG_RUNTIME_OPTION_QSA_HOT;
    if(requested->qsa_page_cache_bytes)mask|=FG_RUNTIME_OPTION_PAGE_CACHE;
    if(requested->prefill_microbatch)mask|=FG_RUNTIME_OPTION_PREFILL_MICROBATCH;
    if(requested->prefill_window)mask|=FG_RUNTIME_OPTION_PREFILL_WINDOW;
    return mask;
}

void fg_runtime_options_init(fg_runtime_options *options){
    if(options)memset(options,0,sizeof(*options));
}

fg_status fg_runtime_profile_apply(fg_manifest *manifest,uint32_t profile,fg_error *err){
    if(!manifest){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime profile manifest");
        return FG_ERR_ARGUMENT;
    }
    const fg_runtime_profile_definition *definition=
        fg_runtime_profile_definition_get(profile);
    if(!definition){
        fg_error_set(err,FG_ERR_ARGUMENT,"unsupported runtime profile %u",profile);
        return FG_ERR_ARGUMENT;
    }
    if(manifest->format_version!=FG_MANIFEST_LEGACY_FORMAT_VERSION&&
       manifest->format_version!=FG_MANIFEST_SESSION_FORMAT_VERSION&&
       manifest->format_version!=FG_MANIFEST_FORMAT_VERSION){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime profile %s is incompatible with manifest format v%u",
                     definition->name,manifest->format_version);
        return FG_ERR_MISMATCH;
    }
    if(manifest->magic!=FG_MANIFEST_MAGIC||
       manifest->native_context!=definition->logical_context_tokens){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime profile %s is incompatible with manifest native context",
                     definition->name);
        return FG_ERR_MISMATCH;
    }
    bool pipeline=profile==FG_RUNTIME_PROFILE_PIPELINE_8STAGE_262K;
    if(pipeline){
        if(manifest->format_version!=FG_MANIFEST_FORMAT_VERSION){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "runtime profile %s requires a manifest v6 source pack",
                         definition->name);
            return FG_ERR_MISMATCH;
        }
        bool packed=manifest->tensor_count!=0u;
        for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
            packed=packed||manifest->ranks[rank].tensor_count!=0u||
                manifest->ranks[rank].persistent_bytes!=0u;
        if(packed){
            fg_error_set(err,FG_ERR_UNAVAILABLE,
                         "runtime profile %s requires repacking from source",
                         definition->name);
            return FG_ERR_UNAVAILABLE;
        }
    }else if(manifest->format_version==FG_MANIFEST_FORMAT_VERSION&&
             manifest->execution_mode!=FG_EXECUTION_EXPERT_PARALLEL){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime profile %s requires expert-parallel execution",
                     definition->name);
        return FG_ERR_MISMATCH;
    }
    if(manifest->session.position_mode!=definition->position_mode){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime profile %s requires text position mode",
                     definition->name);
        return FG_ERR_MISMATCH;
    }
    if(manifest->prefill_window!=definition->prefill_window){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime profile %s requires prefill window %u",
                     definition->name,definition->prefill_window);
        return FG_ERR_MISMATCH;
    }
    if(manifest->prefill_microbatch!=FG_DEFAULT_MICROBATCH&&
       manifest->prefill_microbatch!=definition->prefill_microbatch){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime profile %s cannot replace prefill microbatch %u",
                     definition->name,manifest->prefill_microbatch);
        return FG_ERR_MISMATCH;
    }
    if(manifest->format_version!=FG_MANIFEST_LEGACY_FORMAT_VERSION){
        uint32_t boot=manifest->native_context<FG_RUNTIME_BOOT_CONTEXT_TOKENS?
            manifest->native_context:FG_RUNTIME_BOOT_CONTEXT_TOKENS;
        bool default_contract=manifest->session.logical_context_tokens==boot&&
            manifest->session.gpu_index_tokens==boot&&
            manifest->session.qsa_hot_record_tokens==boot&&
            manifest->session.host_page_cache_bytes==0u;
        bool same_profile=manifest->session.logical_context_tokens==
                definition->logical_context_tokens&&
            manifest->session.gpu_index_tokens==definition->gpu_index_tokens&&
            manifest->session.qsa_hot_record_tokens==definition->qsa_hot_tokens&&
            manifest->session.host_page_cache_bytes==definition->qsa_page_cache_bytes;
        if(!default_contract&&!same_profile){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "runtime profile %s is incompatible with existing session budgets",
                         definition->name);
            return FG_ERR_MISMATCH;
        }
    }
    fg_runtime_options options={
        .logical_context_tokens=definition->logical_context_tokens,
        .gpu_index_tokens=definition->gpu_index_tokens,
        .qsa_hot_tokens=definition->qsa_hot_tokens,
        .qsa_page_cache_bytes=definition->qsa_page_cache_bytes,
        .prefill_microbatch=definition->prefill_microbatch,
        .prefill_window=definition->prefill_window
    };
    fg_status status=fg_runtime_profile_validate(&options,manifest->native_context,err);
    if(status!=FG_OK)return status;
    manifest->max_context=definition->max_context;
    manifest->prefill_microbatch=definition->prefill_microbatch;
    manifest->prefill_window=definition->prefill_window;
    manifest->session.logical_context_tokens=definition->logical_context_tokens;
    manifest->session.gpu_index_tokens=definition->gpu_index_tokens;
    manifest->session.qsa_hot_record_tokens=definition->qsa_hot_tokens;
    manifest->session.host_page_cache_bytes=definition->qsa_page_cache_bytes;
    if(pipeline){
        fg_topology_build_pipeline(manifest);
        manifest->protocol_version=FG_PIPELINE_PROTOCOL_VERSION;
    }else if(manifest->format_version==FG_MANIFEST_FORMAT_VERSION){
        fg_topology_set_expert_parallel_metadata(manifest);
        manifest->protocol_version=FG_PROTOCOL_VERSION;
    }
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)
        manifest->ranks[rank].scratch_bytes=fg_q38_runtime_scratch_bytes(
            rank,manifest->prefill_microbatch,manifest->prefill_window,
            manifest->max_context);
    fg_q38_account_session_state(manifest);
    return FG_OK;
}

static bool staged_context(uint32_t tokens){
    return tokens==32768u||tokens==65536u||tokens==131072u||tokens==262144u;
}

fg_status fg_runtime_profile_validate(const fg_runtime_options *options,
                                      uint32_t native_context,fg_error *err){
    if(!options||!native_context){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime profile");
        return FG_ERR_ARGUMENT;
    }
    if(options->logical_context_tokens>native_context){
        fg_error_set(err,FG_ERR_LIMIT,"logical context %u exceeds native context %u",
                     options->logical_context_tokens,native_context);
        return FG_ERR_LIMIT;
    }
    if(options->logical_context_tokens==FG_RUNTIME_BOOT_CONTEXT_TOKENS){
        if(options->gpu_index_tokens!=FG_RUNTIME_BOOT_CONTEXT_TOKENS||
           options->qsa_hot_tokens!=FG_RUNTIME_BOOT_CONTEXT_TOKENS||
           options->qsa_page_cache_bytes){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "default QSA profile must be 8192/8192/8192/0");
            return FG_ERR_MISMATCH;
        }
        return FG_OK;
    }
    if(!staged_context(options->logical_context_tokens)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "logical context %u is not a staged 32K, 64K, 128K, or 262K profile",
                     options->logical_context_tokens);
        return FG_ERR_UNAVAILABLE;
    }
    if(options->gpu_index_tokens!=options->logical_context_tokens){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "native QSA requires GPU index tokens %u to equal logical context %u",
                     options->gpu_index_tokens,options->logical_context_tokens);
        return FG_ERR_MISMATCH;
    }
    if(options->qsa_hot_tokens&&
       options->qsa_hot_tokens!=FG_RUNTIME_QSA_HOT_TOKENS){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "tiered QSA hot-token compatibility hint must be zero or 8192");
        return FG_ERR_MISMATCH;
    }
    if(options->qsa_page_cache_bytes<FG_RUNTIME_QSA_CACHE_MIN_BYTES||
       options->qsa_page_cache_bytes>FG_RUNTIME_QSA_CACHE_MAX_BYTES||
       (options->qsa_page_cache_bytes&((UINT64_C(1)<<20u)-1u))){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "staged QSA page cache must be a whole MiB from 16 to 512 MiB");
        return FG_ERR_ARGUMENT;
    }
    return FG_OK;
}

fg_status fg_runtime_options_resolve(fg_runtime_options *resolved,
                                     const fg_manifest *manifest,
                                     const fg_runtime_options *requested,
                                     fg_error *err){
    if(!resolved||!manifest){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid runtime option resolution arguments");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_manifest_validate(manifest,err);
    if(status!=FG_OK)return status;
    fg_runtime_options input={0};if(requested)input=*requested;
    uint32_t mask=requested_mask(requested);
    uint32_t boot_context=manifest->native_context<FG_RUNTIME_BOOT_CONTEXT_TOKENS?
        manifest->native_context:FG_RUNTIME_BOOT_CONTEXT_TOKENS;
    bool legacy=manifest->format_version==FG_MANIFEST_LEGACY_FORMAT_VERSION;
    fg_runtime_options defaults={
        .logical_context_tokens=legacy?boot_context:manifest->session.logical_context_tokens,
        .gpu_index_tokens=legacy?boot_context:manifest->session.gpu_index_tokens,
        .qsa_hot_tokens=legacy?boot_context:manifest->session.qsa_hot_record_tokens,
        .qsa_page_cache_bytes=legacy?0u:manifest->session.host_page_cache_bytes,
        .prefill_microbatch=manifest->prefill_microbatch,
        .prefill_window=manifest->prefill_window
    };
    uint32_t budget_mask=FG_RUNTIME_OPTION_LOGICAL_CONTEXT|FG_RUNTIME_OPTION_GPU_INDEX|
        FG_RUNTIME_OPTION_QSA_HOT|FG_RUNTIME_OPTION_PAGE_CACHE;
    if((mask&budget_mask)&&((mask&budget_mask)!=budget_mask)){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "staged QSA requires explicit context, GPU index, hot, and page-cache budgets");
        return FG_ERR_ARGUMENT;
    }
    *resolved=defaults;
    if(mask&budget_mask){
        resolved->logical_context_tokens=input.logical_context_tokens;
        resolved->gpu_index_tokens=input.gpu_index_tokens;
        resolved->qsa_hot_tokens=input.qsa_hot_tokens;
        resolved->qsa_page_cache_bytes=input.qsa_page_cache_bytes;
    }
    if((mask&FG_RUNTIME_OPTION_PREFILL_MICROBATCH)&&
       input.prefill_microbatch!=defaults.prefill_microbatch){
        fg_error_set(err,FG_ERR_MISMATCH,"runtime prefill microbatch does not match manifest");
        return FG_ERR_MISMATCH;
    }
    if((mask&FG_RUNTIME_OPTION_PREFILL_WINDOW)&&
       input.prefill_window!=defaults.prefill_window){
        fg_error_set(err,FG_ERR_MISMATCH,"runtime prefill window does not match manifest");
        return FG_ERR_MISMATCH;
    }
    resolved->experimental_flags=input.experimental_flags;
    resolved->specified=mask;
    if(resolved->prefill_microbatch!=manifest->prefill_microbatch||
       resolved->prefill_window!=manifest->prefill_window){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "runtime prefill %ux%u does not match sealed manifest %ux%u",
                     resolved->prefill_microbatch,resolved->prefill_window,
                     manifest->prefill_microbatch,manifest->prefill_window);
        return FG_ERR_MISMATCH;
    }
    if(resolved->experimental_flags){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "experimental context, MTP, and vision are not enabled in this runtime");
        return FG_ERR_UNAVAILABLE;
    }
    if(legacy&&(resolved->logical_context_tokens!=boot_context||
       resolved->gpu_index_tokens!=boot_context||resolved->qsa_hot_tokens!=boot_context||
       resolved->qsa_page_cache_bytes)){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"staged QSA requires protocol version 6");
        return FG_ERR_UNAVAILABLE;
    }
    return fg_runtime_profile_validate(resolved,manifest->native_context,err);
}

fg_status fg_runtime_eval_capacity(uint32_t *qsa_capacity,
                                   const fg_runtime_options *options,
                                   size_t prompt_tokens,uint32_t generation_tokens,
                                   fg_error *err){
    if(!qsa_capacity||!options||!options->logical_context_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid eval capacity arguments");
        return FG_ERR_ARGUMENT;
    }
    *qsa_capacity=0u;
    if(!prompt_tokens||prompt_tokens>SIZE_MAX-(size_t)generation_tokens){
        fg_error_set(err,FG_ERR_LIMIT,"eval prompt plus generation exceeds context capacity");
        return FG_ERR_LIMIT;
    }
    size_t total=prompt_tokens+(size_t)generation_tokens;
    if(total>options->logical_context_tokens){
        fg_error_set(err,FG_ERR_LIMIT,
                     "eval prompt plus generation would use %zu of %u context tokens",
                     total,options->logical_context_tokens);
        return FG_ERR_LIMIT;
    }
    *qsa_capacity=options->logical_context_tokens;
    return FG_OK;
}
