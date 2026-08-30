#include "fg_runtime.h"

#include <string.h>

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

static fg_status require_sealed_u32(uint32_t mask,uint32_t flag,uint32_t requested,
                                    uint32_t sealed,const char *name,fg_error *err){
    if((mask&flag)&&requested!=sealed){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "requested %s %u does not match sealed manifest default %u",
                     name,requested,sealed);
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

static fg_status require_sealed_u64(uint32_t mask,uint32_t flag,uint64_t requested,
                                    uint64_t sealed,const char *name,fg_error *err){
    if((mask&flag)&&requested!=sealed){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "requested %s %llu does not match sealed manifest default %llu",
                     name,(unsigned long long)requested,(unsigned long long)sealed);
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

void fg_runtime_options_init(fg_runtime_options *options){
    if(options)memset(options,0,sizeof(*options));
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
    if(!legacy){
        status=require_sealed_u32(mask,FG_RUNTIME_OPTION_LOGICAL_CONTEXT,
                                  input.logical_context_tokens,
                                  defaults.logical_context_tokens,"logical context",err);
        if(status==FG_OK)status=require_sealed_u32(mask,FG_RUNTIME_OPTION_GPU_INDEX,
                                                   input.gpu_index_tokens,
                                                   defaults.gpu_index_tokens,
                                                   "GPU index tokens",err);
        if(status==FG_OK)status=require_sealed_u32(mask,FG_RUNTIME_OPTION_QSA_HOT,
                                                   input.qsa_hot_tokens,
                                                   defaults.qsa_hot_tokens,
                                                   "QSA hot records",err);
        if(status==FG_OK)status=require_sealed_u64(mask,FG_RUNTIME_OPTION_PAGE_CACHE,
                                                   input.qsa_page_cache_bytes,
                                                   defaults.qsa_page_cache_bytes,
                                                   "QSA page cache bytes",err);
        if(status==FG_OK)status=require_sealed_u32(mask,FG_RUNTIME_OPTION_PREFILL_MICROBATCH,
                                                   input.prefill_microbatch,
                                                   defaults.prefill_microbatch,
                                                   "prefill microbatch",err);
        if(status==FG_OK)status=require_sealed_u32(mask,FG_RUNTIME_OPTION_PREFILL_WINDOW,
                                                   input.prefill_window,
                                                   defaults.prefill_window,
                                                   "prefill window",err);
        if(status!=FG_OK)return status;
        *resolved=defaults;
    }else{
        *resolved=input;
        if(!resolved->logical_context_tokens)resolved->logical_context_tokens=defaults.logical_context_tokens;
        if(!resolved->gpu_index_tokens)resolved->gpu_index_tokens=defaults.gpu_index_tokens;
        if(!resolved->qsa_hot_tokens)
            resolved->qsa_hot_tokens=resolved->logical_context_tokens;
        if(!resolved->prefill_microbatch)resolved->prefill_microbatch=defaults.prefill_microbatch;
        if(!resolved->prefill_window)resolved->prefill_window=defaults.prefill_window;
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
    if(resolved->logical_context_tokens>manifest->native_context){
        fg_error_set(err,FG_ERR_LIMIT,"logical context %u exceeds native context %u",
                     resolved->logical_context_tokens,manifest->native_context);
        return FG_ERR_LIMIT;
    }
    if(resolved->qsa_hot_tokens>resolved->logical_context_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"QSA hot tokens %u exceed logical context %u",
                     resolved->qsa_hot_tokens,resolved->logical_context_tokens);
        return FG_ERR_ARGUMENT;
    }
    if(resolved->gpu_index_tokens>resolved->logical_context_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"GPU index tokens %u exceed logical context %u",
                     resolved->gpu_index_tokens,resolved->logical_context_tokens);
        return FG_ERR_ARGUMENT;
    }
    if(resolved->logical_context_tokens!=boot_context||
       resolved->gpu_index_tokens!=boot_context||
       resolved->qsa_hot_tokens!=boot_context||
       resolved->qsa_page_cache_bytes){
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "tiered QSA is not enabled; qualified logical/index/hot context is %u with page cache 0",
                     boot_context);
        return FG_ERR_UNAVAILABLE;
    }
    return FG_OK;
}
