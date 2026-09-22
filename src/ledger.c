#include "fg_ledger.h"
#include "fg_output.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef struct ledger_buffer {
    char *data;
    size_t capacity;
    size_t used;
} ledger_buffer;

static bool ledger_append(ledger_buffer *buffer,const char *format,...){
    if(!buffer||!buffer->data||buffer->used>=buffer->capacity)return false;
    va_list args;
    va_start(args,format);
    int written=vsnprintf(buffer->data+buffer->used,
                          buffer->capacity-buffer->used,format,args);
    va_end(args);
    if(written<0||(size_t)written>=buffer->capacity-buffer->used)return false;
    buffer->used+=(size_t)written;
    return true;
}

static bool ledger_append_values(ledger_buffer *buffer,const uint64_t *values,
                                 uint32_t count){
    for(uint32_t i=0;i<count;i++)
        if(!ledger_append(buffer,"%s%llu",i?",":"",
                          (unsigned long long)values[i]))return false;
    return true;
}

static bool ledger_append_named_list(ledger_buffer *buffer,const char *key,
                                     const uint64_t *values,uint32_t count){
    return ledger_append(buffer," %s=",key)&&
        ledger_append_values(buffer,values,count);
}

static bool ledger_append_blocks(ledger_buffer *buffer,const fg_manifest *manifest){
    uint32_t layer=0u;
    bool first=true;
    while(layer<FG_LAYER_COUNT){
        uint32_t owner=manifest->layer_owner[layer],end=layer+1u;
        while(end<FG_LAYER_COUNT&&manifest->layer_owner[end]==owner)end++;
        if(!ledger_append(buffer,"%s%u:%u",first?"":",",owner,end-layer))return false;
        first=false;
        layer=end;
    }
    return true;
}

static bool ledger_single_owner(const fg_manifest *manifest){
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        uint32_t owner=manifest->layer_owner[layer];
        for(uint32_t group=0;group<FG_GROUP_SIZE;group++)
            if(manifest->layer_groups[layer][group]!=owner)return false;
        for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
            if(manifest->expert_rank[layer][expert]!=owner)return false;
    }
    return true;
}

static uint32_t ledger_wire_hops(const fg_manifest *manifest){
    uint32_t hops=manifest->layer_owner[0u]!=0u?1u:0u;
    for(uint32_t layer=1u;layer<FG_LAYER_COUNT;layer++)
        if(manifest->layer_owner[layer]!=manifest->layer_owner[layer-1u])hops++;
    if(manifest->layer_owner[FG_LAYER_COUNT-1u]!=fg_output_owner_rank(manifest))hops++;
    return hops;
}

static void ledger_rank_bytes(const fg_manifest *manifest,uint64_t weights[FG_RANK_COUNT],
                              uint64_t experts[FG_RANK_COUNT]){
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        weights[rank]=0u;
        experts[rank]=0u;
    }
    for(uint32_t i=0;i<manifest->tensor_count;i++){
        const fg_tensor_record *record=&manifest->tensors[i];
        if(record->rank>=FG_RANK_COUNT||record->kind==FG_TENSOR_HOST_CACHE)continue;
        uint64_t bytes=fg_align_up_u64(record->bytes,FG_ALIGNMENT);
        weights[record->rank]=UINT64_MAX-weights[record->rank]<bytes?
            UINT64_MAX:weights[record->rank]+bytes;
        if(record->kind==FG_TENSOR_ROUTED_EXPERT)
            experts[record->rank]=UINT64_MAX-experts[record->rank]<bytes?
                UINT64_MAX:experts[record->rank]+bytes;
    }
}

static void ledger_hex_prefix(const uint8_t digest[32],char out[17]){
    static const char digits[]="0123456789abcdef";
    for(uint32_t i=0;i<8u;i++){
        out[i*2u]=digits[digest[i]>>4u];
        out[i*2u+1u]=digits[digest[i]&15u];
    }
    out[16]='\0';
}

fg_status fg_ledger_format(char *buffer,size_t capacity,
                           const fg_ledger_inputs *inputs,fg_error *err){
    if(!buffer||!capacity||!inputs||!inputs->manifest||!inputs->options){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid ledger arguments");
        return FG_ERR_ARGUMENT;
    }
    const fg_manifest *manifest=inputs->manifest;
    const fg_runtime_options *options=inputs->options;
    if(manifest->rank_count!=FG_RANK_COUNT||manifest->layer_count!=FG_LAYER_COUNT||
       manifest->expert_count!=FG_EXPERT_COUNT||manifest->hidden_size!=FG_HIDDEN_SIZE||
       manifest->top_k!=FG_TOP_K||manifest->tensor_count>FG_MAX_TENSORS){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "ledger manifest geometry is not the sealed fleet geometry");
        return FG_ERR_MISMATCH;
    }
    uint64_t weights[FG_RANK_COUNT],experts[FG_RANK_COUNT],dense[FG_RANK_COUNT];
    ledger_rank_bytes(manifest,weights,experts);
    uint64_t total=0u;
    for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++){
        dense[rank]=weights[rank]-experts[rank];
        total=UINT64_MAX-total<weights[rank]?UINT64_MAX:total+weights[rank];
    }
    char manifest_sha[17],topology_sha[17];
    ledger_hex_prefix(manifest->manifest_sha256,manifest_sha);
    ledger_hex_prefix(manifest->topology_sha256,topology_sha);
    buffer[0]='\0';
    ledger_buffer out={buffer,capacity,0u};
    bool ok=ledger_append(&out,
        "%s rank=%u format=%u protocol=%u execution=%s layer_mode=%s "
        "ranks=%u layers=%u experts=%u topk=%u hidden=%u blocks=",
        FG_LEDGER_PREFIX,inputs->rank,manifest->format_version,manifest->protocol_version,
        manifest->execution_mode==FG_EXECUTION_EXPERT_PARALLEL?"ep":"unknown",
        ledger_single_owner(manifest)?"single":"grouped",
        manifest->rank_count,manifest->layer_count,manifest->expert_count,
        manifest->top_k,manifest->hidden_size);
    if(ok)ok=ledger_append_blocks(&out,manifest);
    if(ok)ok=ledger_append(&out,
        " wire_hops=%u prefill_frames=%u batch=%u window=%u logical=%u gpu_index=%u "
        "qsa_hot=%u qsa_cache_pages=%u qsa_cache_bytes=%llu ring_prefill=%u ring_decode=%u "
        "sealed_context=%u sealed_gpu_index=%u sealed_hot=%u sealed_cache_bytes=%llu",
        ledger_wire_hops(manifest),inputs->prefill_frames,manifest->prefill_microbatch,
        manifest->prefill_window,options->logical_context_tokens,options->gpu_index_tokens,
        options->qsa_hot_tokens,inputs->qsa_cache_pages,
        (unsigned long long)inputs->qsa_cache_bytes,
        inputs->ring_prefill?1u:0u,inputs->ring_decode?1u:0u,
        manifest->session.logical_context_tokens,manifest->session.gpu_index_tokens,
        manifest->session.qsa_hot_record_tokens,
        (unsigned long long)manifest->session.host_page_cache_bytes);
    if(ok)ok=ledger_append_named_list(&out,"weights",weights,FG_RANK_COUNT);
    if(ok)ok=ledger_append_named_list(&out,"dense",dense,FG_RANK_COUNT);
    if(ok)ok=ledger_append_named_list(&out,"expert",experts,FG_RANK_COUNT);
    if(ok)ok=ledger_append(&out," weights_total=%llu manifest_sha=%s topology_sha=%s",
                           (unsigned long long)total,manifest_sha,topology_sha);
    if(!ok){
        buffer[0]='\0';
        fg_error_set(err,FG_ERR_LIMIT,"ledger line exceeds %zu bytes",capacity);
        return FG_ERR_LIMIT;
    }
    return FG_OK;
}
