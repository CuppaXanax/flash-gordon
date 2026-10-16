#include "fg_vk.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"

#define FG_VK_QSA_SELECTED_BLOCKS (FG_Q38_INDEX_BUDGET/FG_Q38_QSA_COMPRESS_RATIO)
#define FG_VK_QSA_SELECTED_TOKENS (FG_Q38_INDEX_BUDGET+FG_Q38_QSA_COMPRESS_RATIO-1u)

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__)||defined(__i386__)
#include <immintrin.h>
#endif

typedef struct fg_vk_allocation {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    uint64_t bytes,allocated_bytes;
    uint32_t references;
} fg_vk_allocation;

struct fg_vk_tensor {fg_vk_context *context;fg_vk_allocation *allocation;uint64_t offset,bytes;fg_vk_tensor_format format;bool is_view;};
typedef struct fg_vk_kernel {const char *file;uint32_t bindings,push_bytes;VkDescriptorSetLayout set_layout;VkPipelineLayout layout;VkPipeline pipeline;uint32_t specialization_count,specialization_values[2];} fg_vk_kernel;
typedef struct fg_vk_profile_dispatch {const char *scope,*name;uint32_t begin_query,end_query;} fg_vk_profile_dispatch;

#define FG_VK_BATCH_MAX_SETS 4096u
#define FG_VK_GROUP_BUFFERS 16u
#define FG_VK_GROUP_COMMANDS (FG_VK_GROUP_BUFFERS+FG_VK_STATIC_SLOTS)
/* Static runs own a separate descriptor pool: the dynamic epoch counter may
 * sweep the whole batch pool (a single recording span can exceed the old half
 * of it), so sharing one pool let a dynamic tail overwrite static bindings
 * and corrupt a later replay. */
#define FG_VK_STATIC_SET_BASE (FG_VK_BATCH_MAX_SETS/2u)
#define FG_VK_STATIC_SET_STRIDE (FG_VK_BATCH_MAX_SETS/8u)
#define FG_VK_STATIC_SET_COUNT (FG_VK_STATIC_SLOTS*FG_VK_STATIC_SET_STRIDE)
#define FG_VK_PROFILE_MAX_DISPATCHES 4096u
#define FG_VK_PROFILE_QUERY_COUNT (2u+2u*FG_VK_PROFILE_MAX_DISPATCHES)

/* Shadow of the most recent bindings written into each pooled descriptor set,
 * so a repeated dispatch can skip vkUpdateDescriptorSets entirely. */
typedef struct fg_vk_descriptor_cache {
    uint32_t bindings;
    VkBuffer buffer[16];
    uint64_t offset[16],range[16];
} fg_vk_descriptor_cache;

struct fg_vk_context {
    fg_vk_kernel mtp_attention,mtp_attention_indirect,mtp_attention_split,mtp_cache_kv;
    bool static_claimed[FG_VK_STATIC_SLOTS];
    VkInstance instance;VkPhysicalDevice physical;VkDevice device;VkQueue queue;uint32_t queue_family;
    VkCommandPool command_pool;VkCommandBuffer command;VkFence fence;VkFence expert_fence;bool expert_fence_pending;VkDescriptorPool descriptor_pool;VkDescriptorPool static_descriptor_pool;VkDescriptorSetLayout descriptor_set_layout;VkPipelineCache pipeline_cache;
    VkCommandPool copy_pool;VkCommandBuffer copy_command;VkFence copy_fence;
    VkCommandBuffer command_alt;VkFence fence_alt;VkFence pending_fences[2];uint32_t pending_fence_count;bool pipeline_ready;double record_start_s;
    VkSemaphore flush_semaphores[2];VkSemaphore pipelined_semaphore;bool pipelined_semaphore_valid;uint32_t flush_semaphore_slot;
    VkCommandBuffer static_command[FG_VK_STATIC_SLOTS];VkFence static_fence[FG_VK_STATIC_SLOTS];
    bool static_recorded[FG_VK_STATIC_SLOTS],static_pending[FG_VK_STATIC_SLOTS],static_ready[FG_VK_STATIC_SLOTS];
    VkCommandBuffer held_commands[FG_VK_STATIC_SLOTS];uint32_t held_slots[FG_VK_STATIC_SLOTS],held_count;VkFence static_submit_fence[FG_VK_STATIC_SLOTS];VkFence held_fence;bool held_fence_pending;
    VkCommandBuffer command_swap;bool static_active;uint32_t static_slot_active;uint32_t static_saved_set_count;
    uint32_t batch_set_base;
    bool group_active,group_ready,group_published;
    VkCommandBuffer group_buffers[FG_VK_GROUP_BUFFERS],group_barrier,group_saved;
    VkCommandBuffer group_commands[FG_VK_GROUP_COMMANDS];
    uint32_t group_count,group_index;
    VkPhysicalDeviceMemoryProperties memory;char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    bool subgroup_topk;bool integer_dot_product;bool subgroup_expert_q8;
    uint32_t batch_depth;uint32_t batch_set_count;VkDescriptorSet descriptor_sets[FG_VK_BATCH_MAX_SETS];bool batch_has_dispatch;bool barrier_elide_next;
    fg_vk_descriptor_cache descriptor_cache[FG_VK_BATCH_MAX_SETS];bool descriptor_cache_valid[FG_VK_BATCH_MAX_SETS];
    VkDescriptorSet static_descriptor_sets[FG_VK_STATIC_SET_COUNT];fg_vk_descriptor_cache static_descriptor_cache[FG_VK_STATIC_SET_COUNT];bool static_descriptor_cache_valid[FG_VK_STATIC_SET_COUNT];
    VkQueryPool profile_query_pool;float timestamp_period;uint32_t timestamp_valid_bits;
    VkQueryPool layer_query_pool;
    fg_vk_layer_sample *layer_samples;
    uint32_t layer_clock_current;
    bool layer_clock_recording;
    uint32_t active_compute_units;
    bool profile_active,profile_overflow;const char *profile_scope;uint32_t profile_query_count,profile_dispatch_count;
    fg_vk_counters counters;
    fg_vk_memory_stats memory_stats;
    fg_vk_profile_dispatch profile_dispatches[FG_VK_PROFILE_MAX_DISPATCHES];fg_vk_profile profile;
    fg_vk_kernel quant_q8k,quant_q8,quant_q4,dequant_iq4nl,embedding,embedding_batch,swiglu,silu_scaled,dense,dense_f32,dense_bf16,rms,gr,hc_inject_partial,gr_partial,hc_finalize,gr_write,ple_gate,ple_gate_prefill,ple_conv,ple_conv_prefill,ple_conv_add,add,apply_penalties,gdn_conv,gdn_conv_prefill,gdn_recurrent,gdn_recurrent_algebraic,gdn_recurrent_prefill,gdn_prefill_qk_norm,gdn_prefill_recurrence,gdn_prefill_output,qsa_prepare,qsa_prepare_prefill,qsa_index_prepare,qsa_index_prepare_prefill,qsa_record_commit,qsa_record_gather,qsa_score,qsa_attention,qsa_attention_split,qsa_decode_attention_split,qsa_attention_merge,qsa_attention_split_batch,qsa_attention_merge_batch,qsa_record_gather_batch,qsa_select_resolve,qsa_resident_commit,qsa_resident_select,qsa_resident_merge,qsa_resident_attention,topk_v2,topk_select,topk_select_fallback,moe_q5_1,moe_q5_1_cooked,moe_q8_0,moe_q8_0_cooked,moe_reduce,kquant,kquant_cooked;
    fg_vk_kernel argmax,dense_subgroup,dense_cooked,dense_cooked_r8,mtp_vocab,mtp_fc,mtp_norm,dense_quant[37];
    bool small_n;
    fg_vk_kernel small_dense[75],small_gate_up[8],small_down,small_down_reduce,small_shared,small_union,small_gdn;
    fg_vk_kernel small_plain[9];
    /* Decomposition benchmark kernels */
    fg_vk_kernel bench_stream,bench_dequant,bench_dot_nored;
    fg_vk_kernel dense_cooked_split,dense_cooked_split_reduce,dense_cooked_split_reduce_silu;
    fg_vk_kernel bench_stream_wide,bench_stream_vec;
    fg_vk_kernel bench_cooked_layout;
    fg_vk_kernel dense_cooked_tile,router_top10,expert_major_pack;
    fg_vk_kernel decode_tile_schedule;
    fg_vk_kernel kquant_cooked_grouped,kquant_cooked_grouped_int,q5_1_cooked_grouped,q8_0_grouped,q8_0_cooked_grouped;
    fg_vk_kernel moe_decode_gate_up,moe_decode_down_reduce,moe_decode_down_q8;
    fg_vk_kernel moe_decode_shared_add;
    fg_vk_kernel moe_prefill_reduce,moe_prefill_shard_reduce;
    fg_vk_kernel router_top10_tiles,router_top10_fast,moe_shared_gr_write;
    fg_vk_kernel shared_swiglu_q4,shared_swiglu_q8;
    fg_vk_kernel roof[88];
    bool roof_supported,roof_wave32;
    fg_vk_tensor *roof_partials,*roof_gdn,*roof_expert_partials;
};

static bool roof_enabled(const fg_vk_context *c,uint32_t bit){
    const char *value=getenv("FG_ROOFLINE");
    return c->roof_supported&&value&&((uint32_t)strtoul(value,NULL,0)&bit)!=0u;
}
static fg_vk_kernel *roof_kernel(fg_vk_context *c,uint32_t slot,const char *file,
                               uint32_t bindings,uint32_t push_bytes){
    if(!c->roof[slot].file)c->roof[slot]=(fg_vk_kernel){
        .file=file,.bindings=bindings,.push_bytes=push_bytes};
    return &c->roof[slot];
}

struct fg_vk_expert_graph {
    fg_vk_context *context;
    VkDescriptorPool descriptor_pool;
    VkCommandBuffer command;
    uint32_t dispatches;
};

static fg_status vk_error(fg_error *err,const char *operation,VkResult result){fg_error_set(err,FG_ERR_UNAVAILABLE,"%s: Vulkan result %d",operation,(int)result);return FG_ERR_UNAVAILABLE;}
void fg_vk_small_n_mode(fg_vk_context *c,bool enabled){if(c)c->small_n=enabled;}
bool fg_vk_small_n_active(const fg_vk_context *c){return c&&c->small_n;}
static double vk_wall_seconds(void){struct timespec value;clock_gettime(CLOCK_MONOTONIC,&value);return (double)value.tv_sec+(double)value.tv_nsec/1000000000.0;}
static void vk_profile_begin_record(fg_vk_context *c){if(c->profile_active)c->record_start_s=vk_wall_seconds();}
static void vk_profile_end_record(fg_vk_context *c){if(c->profile_active)c->profile.record_ms+=(vk_wall_seconds()-c->record_start_s)*1000.0;}
static void vk_profile_add_wait(fg_vk_context *c,double seconds){if(c->profile_active)c->profile.wait_ms+=seconds*1000.0;}
static void vk_fence_track(fg_vk_context *c,VkFence fence){if(c->pending_fence_count<2u)c->pending_fences[c->pending_fence_count++]=fence;}
static fg_status vk_fence_wait(fg_vk_context *c,VkFence fence,fg_error *err){uint32_t index=0u;while(index<c->pending_fence_count&&c->pending_fences[index]!=fence)index++;if(index==c->pending_fence_count)return FG_OK;double wait_begin=vk_wall_seconds();VkResult vr=vkWaitForFences(c->device,1u,&fence,VK_TRUE,UINT64_MAX);vk_profile_add_wait(c,vk_wall_seconds()-wait_begin);if(vr!=VK_SUCCESS)return vk_error(err,"wait for queued command",vr);c->pending_fences[index]=c->pending_fences[--c->pending_fence_count];return FG_OK;}
static fg_status vk_pipeline_drain(fg_vk_context *c,fg_error *err){while(c->pending_fence_count){fg_status status=vk_fence_wait(c,c->pending_fences[0],err);if(status!=FG_OK)return status;}return FG_OK;}
/* Chain an asynchronous flush submission to the next one with a binary
 * semaphore so the later command buffer cannot read a tensor while the earlier
 * one is still writing it; the host still records ahead of the GPU. */
static void vk_submit_attach(fg_vk_context *c,bool signal_flush,VkSemaphore *wait,VkSemaphore *signal,VkPipelineStageFlags *wait_stage,VkSubmitInfo *submit){*wait=c->pipelined_semaphore_valid?c->pipelined_semaphore:VK_NULL_HANDLE;*signal=VK_NULL_HANDLE;if(*wait!=VK_NULL_HANDLE){submit->waitSemaphoreCount=1u;submit->pWaitSemaphores=wait;submit->pWaitDstStageMask=wait_stage;}if(signal_flush){*signal=c->flush_semaphores[c->flush_semaphore_slot];c->flush_semaphore_slot^=1u;submit->signalSemaphoreCount=1u;submit->pSignalSemaphores=signal;c->pipelined_semaphore=*signal;c->pipelined_semaphore_valid=true;}else c->pipelined_semaphore_valid=false;}
static bool tensor_range(const fg_vk_tensor *tensor,uint64_t offset,uint64_t bytes){return tensor&&offset<=tensor->bytes&&bytes<=tensor->bytes-offset;}
static bool tensor_on_context(const fg_vk_context *context,const fg_vk_tensor *tensor){return tensor&&tensor->context==context;}

static fg_status vk_flush_held(fg_vk_context *c,fg_error *err);
static void profile_command_begin(fg_vk_context *c){
    if(!c->profile_active)return;
    c->profile_query_count=1u;c->profile_dispatch_count=0u;c->profile_overflow=false;
    vkCmdResetQueryPool(c->command,c->profile_query_pool,0,FG_VK_PROFILE_QUERY_COUNT);
    vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,c->profile_query_pool,0u);
}

static bool profile_dispatch_begin(fg_vk_context *c,const fg_vk_kernel *kernel){
    if(!c->profile_active)return false;
    if(c->profile_dispatch_count>=FG_VK_PROFILE_MAX_DISPATCHES||c->profile_query_count+2u>=FG_VK_PROFILE_QUERY_COUNT){c->profile_overflow=true;return false;}
    fg_vk_profile_dispatch *record=&c->profile_dispatches[c->profile_dispatch_count++];record->scope=c->profile_scope;record->name=kernel->file;record->begin_query=c->profile_query_count++;
    vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,c->profile_query_pool,record->begin_query);return true;
}

static void profile_dispatch_end(fg_vk_context *c,bool profiled){
    if(!profiled)return;
    fg_vk_profile_dispatch *record=&c->profile_dispatches[c->profile_dispatch_count-1u];record->end_query=c->profile_query_count++;
    vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,c->profile_query_pool,record->end_query);
}

static void profile_command_end(fg_vk_context *c){if(c->profile_active)vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,c->profile_query_pool,c->profile_query_count++);}

static uint64_t profile_tick_delta(const fg_vk_context *c,uint64_t begin,uint64_t end){if(c->timestamp_valid_bits>=64u)return end-begin;uint64_t mask=(UINT64_C(1)<<c->timestamp_valid_bits)-1u;return(end-begin)&mask;}

fg_status fg_vk_layer_clock_enable(fg_vk_context *c,fg_error *err){
    if(!c||c->batch_depth||c->profile_active||!c->timestamp_valid_bits){
        fg_error_set(err,FG_ERR_ARGUMENT,"layer clocks require an idle timestamp-capable context");
        return FG_ERR_ARGUMENT;
    }
    for(uint32_t slot=0;slot<FG_VK_STATIC_SLOTS;slot++)if(c->static_recorded[slot]){
        fg_error_set(err,FG_ERR_MISMATCH,"enable layer clocks before static recording");
        return FG_ERR_MISMATCH;
    }
    if(c->layer_query_pool)return FG_OK;
    c->layer_samples=calloc(FG_LAYER_COUNT,sizeof(*c->layer_samples));
    if(!c->layer_samples){fg_error_set(err,FG_ERR_OOM,"allocate layer dispatch ledger");return FG_ERR_OOM;}
    VkQueryPoolCreateInfo info={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2u*FG_LAYER_COUNT};
    VkResult vr=vkCreateQueryPool(c->device,&info,NULL,&c->layer_query_pool);
    if(vr!=VK_SUCCESS){free(c->layer_samples);c->layer_samples=NULL;return vk_error(err,"create layer clock pool",vr);}
    return FG_OK;
}

fg_status fg_vk_layer_clock_mark(fg_vk_context *c,uint32_t layer,bool end,fg_error *err){
    if(!c||!c->layer_query_pool)return FG_OK;
    if(layer>=FG_LAYER_COUNT||!c->batch_depth||
       (end&&(!c->layer_clock_recording||c->layer_clock_current!=layer))){
        fg_error_set(err,FG_ERR_MISMATCH,"layer clock boundary outside recording");
        return FG_ERR_MISMATCH;
    }
    if(!end){
        c->layer_samples[layer].dispatch_count=0;
        c->layer_clock_current=layer;c->layer_clock_recording=true;
        vkCmdResetQueryPool(c->command,c->layer_query_pool,layer*2u,2u);
    }
    vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        c->layer_query_pool,layer*2u+(end?1u:0u));
    if(end)c->layer_clock_recording=false;
    return FG_OK;
}

fg_status fg_vk_layer_clock_read(fg_vk_context *c,uint32_t layer,
                                  fg_vk_layer_sample *sample,fg_error *err){
    if(!c||!sample||!c->layer_query_pool||layer>=FG_LAYER_COUNT||
       !c->layer_samples[layer].dispatch_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"layer clocks are unavailable");return FG_ERR_ARGUMENT;
    }
    uint64_t ticks[2];
    VkResult vr=vkGetQueryPoolResults(c->device,c->layer_query_pool,layer*2u,2u,
        sizeof(ticks),ticks,sizeof(*ticks),VK_QUERY_RESULT_64_BIT);
    if(vr!=VK_SUCCESS)return vk_error(err,"read drained layer clocks",vr);
    *sample=c->layer_samples[layer];
    sample->gpu_us=(double)profile_tick_delta(c,ticks[0],ticks[1])*c->timestamp_period/1000.0;
    return FG_OK;
}

static fg_status profile_resolve(fg_vk_context *c,fg_error *err){
    if(!c->profile_active)return FG_OK;
    uint64_t ticks[FG_VK_PROFILE_QUERY_COUNT];VkResult vr=vkGetQueryPoolResults(c->device,c->profile_query_pool,0,c->profile_query_count,(size_t)c->profile_query_count*sizeof(*ticks),ticks,sizeof(*ticks),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);if(vr!=VK_SUCCESS)return vk_error(err,"read timestamp queries",vr);
    if(c->profile_overflow){fg_error_set(err,FG_ERR_LIMIT,"Vulkan profile exceeded %u dispatches in one submission",FG_VK_PROFILE_MAX_DISPATCHES);return FG_ERR_LIMIT;}
    double scale=(double)c->timestamp_period/1000000.0,total=(double)profile_tick_delta(c,ticks[0],ticks[c->profile_query_count-1u])*scale;c->profile.submissions++;c->profile.gpu_ms+=total;
    {uint64_t previous_end=ticks[0];for(uint32_t i=0;i<c->profile_dispatch_count;i++){const fg_vk_profile_dispatch *record=&c->profile_dispatches[i];double gap=(double)profile_tick_delta(c,previous_end,ticks[record->begin_query])*scale;c->profile.gap_ms+=gap;c->profile.gap_count++;if(gap>c->profile.gap_max_ms){c->profile.gap_max_ms=gap;c->profile.gap_max_kernel=record->name;}previous_end=ticks[record->end_query];}double tail=(double)profile_tick_delta(c,previous_end,ticks[c->profile_query_count-1u])*scale;c->profile.gap_ms+=tail;c->profile.gap_count++;if(tail>c->profile.gap_max_ms){c->profile.gap_max_ms=tail;c->profile.gap_max_kernel="<submission-tail>";}}
    for(uint32_t i=0;i<c->profile_dispatch_count;i++){const fg_vk_profile_dispatch *record=&c->profile_dispatches[i];double ms=(double)profile_tick_delta(c,ticks[record->begin_query],ticks[record->end_query])*scale;fg_vk_profile_kernel *kernel=NULL;for(uint32_t j=0;j<c->profile.kernel_count;j++)if(strcmp(c->profile.kernels[j].scope,record->scope)==0&&strcmp(c->profile.kernels[j].name,record->name)==0){kernel=&c->profile.kernels[j];break;}if(!kernel){if(c->profile.kernel_count>=FG_VK_PROFILE_MAX_KERNELS){fg_error_set(err,FG_ERR_LIMIT,"Vulkan profile exceeded %u scoped kernel types",FG_VK_PROFILE_MAX_KERNELS);return FG_ERR_LIMIT;}kernel=&c->profile.kernels[c->profile.kernel_count++];kernel->scope=record->scope;kernel->name=record->name;}kernel->invocations++;kernel->gpu_ms+=ms;c->profile.dispatches++;c->profile.kernel_ms+=ms;}
    return FG_OK;
}

static fg_status read_shader(const char *file,uint32_t **words,size_t *bytes,fg_error *err){
    const char *directory=getenv("FG_SHADER_DIR");char path[1024];
    if(directory&&*directory)snprintf(path,sizeof(path),"%s/%s",directory,file);else snprintf(path,sizeof(path),"vulkan/%s",file);
    FILE *stream=fopen(path,"rb");if(!stream){fg_error_set(err,FG_ERR_IO,"open Vulkan shader %s",path);return FG_ERR_IO;}
    if(fseek(stream,0,SEEK_END)!=0){fclose(stream);fg_error_set(err,FG_ERR_IO,"seek Vulkan shader %s",path);return FG_ERR_IO;}
    long length=ftell(stream);if(length<=0||(length&3)!=0||fseek(stream,0,SEEK_SET)!=0){fclose(stream);fg_error_set(err,FG_ERR_FORMAT,"invalid SPIR-V shader %s",path);return FG_ERR_FORMAT;}
    uint32_t *data=malloc((size_t)length);if(!data){fclose(stream);fg_error_set(err,FG_ERR_OOM,"allocate SPIR-V shader");return FG_ERR_OOM;}
    if(fread(data,1,(size_t)length,stream)!=(size_t)length){free(data);fclose(stream);fg_error_set(err,FG_ERR_IO,"read Vulkan shader %s",path);return FG_ERR_IO;}
    fclose(stream);*words=data;*bytes=(size_t)length;return FG_OK;
}

static void destroy_kernel(fg_vk_context *context,fg_vk_kernel *kernel){
    if(kernel==&context->router_top10){
        destroy_kernel(context,&context->router_top10_tiles);
        destroy_kernel(context,&context->router_top10_fast);
    }
    if(kernel==&context->moe_decode_shared_add)destroy_kernel(context,&context->moe_shared_gr_write);
    if(kernel==&context->swiglu){
        destroy_kernel(context,&context->shared_swiglu_q4);
        destroy_kernel(context,&context->shared_swiglu_q8);
    }
    if(kernel==&context->qsa_attention){
        destroy_kernel(context,&context->mtp_attention);
        destroy_kernel(context,&context->mtp_attention_indirect);
        destroy_kernel(context,&context->mtp_attention_split);
        destroy_kernel(context,&context->mtp_cache_kv);
    }
    if(kernel==&context->dense){
        for(uint32_t i=0;i<75u;i++)destroy_kernel(context,&context->small_dense[i]);
        for(uint32_t i=0;i<9u;i++)destroy_kernel(context,&context->small_plain[i]);
        for(uint32_t i=0;i<8u;i++)destroy_kernel(context,&context->small_gate_up[i]);
        destroy_kernel(context,&context->small_down);
        destroy_kernel(context,&context->small_down_reduce);
        destroy_kernel(context,&context->small_shared);
        destroy_kernel(context,&context->small_union);
        destroy_kernel(context,&context->small_gdn);
    }
    if(kernel==&context->dense)
        for(uint32_t i=0;i<37u;i++)destroy_kernel(context,&context->dense_quant[i]);
    if(kernel==&context->dense)
        for(uint32_t i=0;i<sizeof(context->roof)/sizeof(context->roof[0]);i++)destroy_kernel(context,&context->roof[i]);
    if(kernel==&context->moe_decode_down_reduce)destroy_kernel(context,&context->moe_decode_down_q8);
    if(kernel==&context->moe_q5_1)destroy_kernel(context,&context->moe_q5_1_cooked);
    if(kernel==&context->kquant)destroy_kernel(context,&context->kquant_cooked);
    if(kernel==&context->dense_cooked){
        destroy_kernel(context,&context->dense_cooked_r8);
        destroy_kernel(context,&context->mtp_vocab);
        destroy_kernel(context,&context->mtp_fc);
        destroy_kernel(context,&context->mtp_norm);
    }
    if(kernel->pipeline)vkDestroyPipeline(context->device,kernel->pipeline,NULL);
    if(kernel->layout)vkDestroyPipelineLayout(context->device,kernel->layout,NULL);
    kernel->pipeline=VK_NULL_HANDLE;kernel->layout=VK_NULL_HANDLE;kernel->set_layout=VK_NULL_HANDLE;
}

static fg_status create_kernel(fg_vk_context *context,fg_vk_kernel *kernel,fg_error *err){
    if(kernel->pipeline)return FG_OK;
    uint32_t *code=NULL;size_t code_bytes=0;fg_status status=read_shader(kernel->file,&code,&code_bytes,err);if(status!=FG_OK)return status;
    VkShaderModule module=VK_NULL_HANDLE;VkShaderModuleCreateInfo sm={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=code_bytes,.pCode=code};VkResult vr=vkCreateShaderModule(context->device,&sm,NULL,&module);free(code);if(vr!=VK_SUCCESS)return vk_error(err,"create shader module",vr);
    if(kernel->bindings>16u){vkDestroyShaderModule(context->device,module,NULL);fg_error_set(err,FG_ERR_LIMIT,"Vulkan kernel has too many bindings");return FG_ERR_LIMIT;}kernel->set_layout=context->descriptor_set_layout;
    VkPushConstantRange push={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=kernel->push_bytes};VkPipelineLayoutCreateInfo pl={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&kernel->set_layout,.pushConstantRangeCount=kernel->push_bytes?1u:0u,.pPushConstantRanges=kernel->push_bytes?&push:NULL};
    vr=vkCreatePipelineLayout(context->device,&pl,NULL,&kernel->layout);if(vr!=VK_SUCCESS){vkDestroyShaderModule(context->device,module,NULL);destroy_kernel(context,kernel);return vk_error(err,"create pipeline layout",vr);}
    VkSpecializationMapEntry entries[]={
        {.constantID=0,.offset=0,.size=sizeof(uint32_t)},
        {.constantID=1,.offset=sizeof(uint32_t),.size=sizeof(uint32_t)}};
    VkSpecializationInfo specialization={.mapEntryCount=kernel->specialization_count,
        .pMapEntries=entries,.dataSize=kernel->specialization_count*sizeof(uint32_t),
        .pData=kernel->specialization_values};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT required_wave={
        .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .requiredSubgroupSize=32u};
    VkPipelineShaderStageCreateInfo stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext=context->roof_wave32&&strstr(kernel->file,"_w32.spv")?&required_wave:NULL,
        .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=module,.pName="main",
        .pSpecializationInfo=kernel->specialization_count?&specialization:NULL};
    VkComputePipelineCreateInfo cp={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=stage,.layout=kernel->layout};
    const char *radv_debug=getenv("RADV_DEBUG");
    if(context->profile_active||(radv_debug&&strstr(radv_debug,"shader")))fprintf(stderr,"VK_PIPELINE file=%s specialization=[%u,%u]\n",
        kernel->file,kernel->specialization_values[0],kernel->specialization_values[1]);
    vr=vkCreateComputePipelines(context->device,context->pipeline_cache,1,&cp,NULL,&kernel->pipeline);vkDestroyShaderModule(context->device,module,NULL);if(vr!=VK_SUCCESS){destroy_kernel(context,kernel);return vk_error(err,"create compute pipeline",vr);}return FG_OK;
}

static bool device_has_extension(VkPhysicalDevice physical,const char *extension_name){
    uint32_t count=0;
    if(vkEnumerateDeviceExtensionProperties(physical,NULL,&count,NULL)!=VK_SUCCESS)return false;
    VkExtensionProperties *extensions=malloc((size_t)count*sizeof(*extensions));
    if(!extensions)return false;
    bool found=false;
    if(vkEnumerateDeviceExtensionProperties(physical,NULL,&count,extensions)==VK_SUCCESS)
        for(uint32_t i=0;i<count;i++)if(strcmp(extensions[i].extensionName,extension_name)==0){found=true;break;}
    free(extensions);
    return found;
}

fg_status fg_vk_open(fg_vk_context **out,fg_error *err){
    if(!out){fg_error_set(err,FG_ERR_ARGUMENT,"Vulkan context output is null");return FG_ERR_ARGUMENT;}*out=NULL;fg_vk_context *c=calloc(1,sizeof(*c));if(!c){fg_error_set(err,FG_ERR_OOM,"allocate Vulkan context");return FG_ERR_OOM;}
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="flash-gordon",.applicationVersion=1,.pEngineName="flash-gordon",.engineVersion=1,.apiVersion=VK_API_VERSION_1_1};VkInstanceCreateInfo ic={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};VkResult vr=vkCreateInstance(&ic,NULL,&c->instance);if(vr!=VK_SUCCESS){free(c);return vk_error(err,"create Vulkan instance",vr);}
    uint32_t count=0;vr=vkEnumeratePhysicalDevices(c->instance,&count,NULL);if(vr!=VK_SUCCESS||count==0){fg_vk_close(c);return vk_error(err,"enumerate Vulkan devices",vr);}
    VkPhysicalDevice *devices=malloc((size_t)count*sizeof(*devices));if(!devices){fg_vk_close(c);fg_error_set(err,FG_ERR_OOM,"enumerate Vulkan devices");return FG_ERR_OOM;}vr=vkEnumeratePhysicalDevices(c->instance,&count,devices);if(vr!=VK_SUCCESS){free(devices);fg_vk_close(c);return vk_error(err,"enumerate Vulkan devices",vr);}
    int best=-1;uint32_t family=0;for(uint32_t d=0;d<count;d++){uint32_t families=0;vkGetPhysicalDeviceQueueFamilyProperties(devices[d],&families,NULL);VkQueueFamilyProperties *properties=malloc((size_t)families*sizeof(*properties));if(!properties)continue;vkGetPhysicalDeviceQueueFamilyProperties(devices[d],&families,properties);VkPhysicalDeviceProperties device_properties;vkGetPhysicalDeviceProperties(devices[d],&device_properties);int score=device_properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU?3:device_properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?2:1;for(uint32_t f=0;f<families;f++)if((properties[f].queueFlags&VK_QUEUE_COMPUTE_BIT)&&score>best){best=score;c->physical=devices[d];family=f;c->timestamp_valid_bits=properties[f].timestampValidBits;}free(properties);}free(devices);
    if(best<0){fg_vk_close(c);fg_error_set(err,FG_ERR_UNAVAILABLE,"no Vulkan compute queue");return FG_ERR_UNAVAILABLE;}c->queue_family=family;VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(c->physical,&properties);memcpy(c->device_name,properties.deviceName,sizeof(c->device_name));c->timestamp_period=properties.limits.timestampPeriod;
#ifdef VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME
    if(device_has_extension(c->physical,VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME)){
        VkPhysicalDeviceShaderCoreProperties2AMD shader_core2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD};
        VkPhysicalDeviceProperties2 shader_core_properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&shader_core2};
        vkGetPhysicalDeviceProperties2(c->physical,&shader_core_properties);
        c->active_compute_units=shader_core2.activeComputeUnitCount;
    }
#endif
#ifdef VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME
    if(!c->active_compute_units&&
       device_has_extension(c->physical,VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME)){
        VkPhysicalDeviceShaderCorePropertiesAMD shader_core={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD};
        VkPhysicalDeviceProperties2 shader_core_properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&shader_core};
        vkGetPhysicalDeviceProperties2(c->physical,&shader_core_properties);
        uint64_t units=(uint64_t)shader_core.shaderEngineCount*
            shader_core.shaderArraysPerEngineCount*shader_core.computeUnitsPerShaderArray;
        c->active_compute_units=units<=UINT32_MAX?(uint32_t)units:0u;
    }
#endif
    if(c->active_compute_units>256u)c->active_compute_units=0u;
    vkGetPhysicalDeviceMemoryProperties(c->physical,&c->memory);VkPhysicalDeviceSubgroupProperties subgroup={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};VkPhysicalDeviceProperties2 properties2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&subgroup};vkGetPhysicalDeviceProperties2(c->physical,&properties2);c->subgroup_topk=properties.limits.maxComputeWorkGroupInvocations>=1024u&&properties.limits.maxComputeWorkGroupSize[0]>=1024u&&subgroup.subgroupSize>=32u&&subgroup.subgroupSize<=64u&&(subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0u&&(subgroup.supportedOperations&(VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))==(VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    c->roof_supported=subgroup.subgroupSize==64u&&
        (subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0u&&
        (subgroup.supportedOperations&(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT|
        VK_SUBGROUP_FEATURE_SHUFFLE_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT))==
        (VK_SUBGROUP_FEATURE_ARITHMETIC_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT|
        VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT);
    const char *expert_q8_wide=getenv("FG_EXPERT_Q8_WIDE");
    c->subgroup_expert_q8=expert_q8_wide&&strcmp(expert_q8_wide,"1")==0&&subgroup.subgroupSize>=32u&&
        (subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0u&&
        (subgroup.supportedOperations&VK_SUBGROUP_FEATURE_SHUFFLE_BIT)!=0u;
    bool integer_dot_extension=device_has_extension(c->physical,"VK_KHR_shader_integer_dot_product");
    VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR};
    if(integer_dot_extension){VkPhysicalDeviceProperties2 integer_properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&integer_dot_properties};vkGetPhysicalDeviceProperties2(c->physical,&integer_properties);}
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR integer_dot_features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
    if(integer_dot_extension){VkPhysicalDeviceFeatures2 available_features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&integer_dot_features};vkGetPhysicalDeviceFeatures2(c->physical,&available_features);}
    c->integer_dot_product=integer_dot_extension&&integer_dot_properties.integerDotProduct4x8BitPackedSignedAccelerated&&integer_dot_features.shaderIntegerDotProduct;
    float priority=1.0f;VkDeviceQueueCreateInfo qc={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT wave_features={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT wave_properties={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
    if(device_has_extension(c->physical,VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)){
        VkPhysicalDeviceFeatures2 query={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&wave_features};
        VkPhysicalDeviceProperties2 props={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&wave_properties};
        vkGetPhysicalDeviceFeatures2(c->physical,&query);vkGetPhysicalDeviceProperties2(c->physical,&props);
        c->roof_wave32=wave_features.subgroupSizeControl&&wave_properties.minSubgroupSize<=32u&&
            wave_properties.maxSubgroupSize>=32u&&
            (wave_properties.requiredSubgroupSizeStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0u;
    }
    const char *device_extensions[2];uint32_t device_extension_count=0;
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT enabled_wave={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
        .subgroupSizeControl=c->roof_wave32?VK_TRUE:VK_FALSE};
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR enabled_integer_dot={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
        .pNext=c->roof_wave32?&enabled_wave:NULL,.shaderIntegerDotProduct=c->integer_dot_product?VK_TRUE:VK_FALSE};
    VkPhysicalDeviceFeatures2 enabled_features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext=c->integer_dot_product?(void *)&enabled_integer_dot:c->roof_wave32?(void *)&enabled_wave:NULL};
    if(c->integer_dot_product)device_extensions[device_extension_count++]="VK_KHR_shader_integer_dot_product";
    if(c->roof_wave32)device_extensions[device_extension_count++]=VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
    VkDeviceCreateInfo dc={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=enabled_features.pNext?&enabled_features:NULL,.queueCreateInfoCount=1,.pQueueCreateInfos=&qc,.enabledExtensionCount=device_extension_count,.ppEnabledExtensionNames=device_extension_count?device_extensions:NULL};vr=vkCreateDevice(c->physical,&dc,NULL,&c->device);if(vr!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create Vulkan device",vr);}vkGetDeviceQueue(c->device,family,0,&c->queue);
    VkCommandPoolCreateInfo pc={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=family};if((vr=vkCreateCommandPool(c->device,&pc,NULL,&c->command_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create command pool",vr);}VkCommandBufferAllocateInfo ca={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=2};VkCommandBuffer commands[2]={0};if((vr=vkAllocateCommandBuffers(c->device,&ca,commands))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"allocate command buffer",vr);}c->command=commands[0];c->command_alt=commands[1];
    VkFenceCreateInfo fc={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};if((vr=vkCreateFence(c->device,&fc,NULL,&c->fence))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create fence",vr);}if((vr=vkCreateFence(c->device,&fc,NULL,&c->expert_fence))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create expert fence",vr);}if((vr=vkCreateFence(c->device,&fc,NULL,&c->fence_alt))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create alternate fence",vr);}if((vr=vkCreateFence(c->device,&fc,NULL,&c->held_fence))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create held-batch fence",vr);}VkSemaphoreCreateInfo sc={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};for(uint32_t i=0;i<2u;i++)if((vr=vkCreateSemaphore(c->device,&sc,NULL,&c->flush_semaphores[i]))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create pipeline semaphore",vr);}c->pipeline_ready=true;VkDescriptorSetLayoutBinding layout_bindings[16];for(uint32_t i=0;i<16u;i++)layout_bindings[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1u,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};VkDescriptorSetLayoutCreateInfo layout_create={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=16u,.pBindings=layout_bindings};if((vr=vkCreateDescriptorSetLayout(c->device,&layout_create,NULL,&c->descriptor_set_layout))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create shared descriptor layout",vr);}VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=FG_VK_BATCH_MAX_SETS*16u};VkDescriptorPoolCreateInfo dp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=FG_VK_BATCH_MAX_SETS,.poolSizeCount=1,.pPoolSizes=&ps};if((vr=vkCreateDescriptorPool(c->device,&dp,NULL,&c->descriptor_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create descriptor pool",vr);}
    enum { layout_count=FG_VK_STATIC_SET_COUNT>FG_VK_BATCH_MAX_SETS?
        FG_VK_STATIC_SET_COUNT:FG_VK_BATCH_MAX_SETS };
    VkDescriptorSetLayout set_layouts[layout_count];
    for(uint32_t i=0;i<layout_count;i++)set_layouts[i]=c->descriptor_set_layout;
    VkDescriptorSetAllocateInfo set_allocate={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=c->descriptor_pool,.descriptorSetCount=FG_VK_BATCH_MAX_SETS,.pSetLayouts=set_layouts};if((vr=vkAllocateDescriptorSets(c->device,&set_allocate,c->descriptor_sets))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"preallocate descriptor sets",vr);}VkDescriptorPoolSize static_ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=FG_VK_STATIC_SET_COUNT*16u};VkDescriptorPoolCreateInfo static_dp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=FG_VK_STATIC_SET_COUNT,.poolSizeCount=1,.pPoolSizes=&static_ps};if((vr=vkCreateDescriptorPool(c->device,&static_dp,NULL,&c->static_descriptor_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create static descriptor pool",vr);}VkDescriptorSetAllocateInfo static_allocate={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=c->static_descriptor_pool,.descriptorSetCount=FG_VK_STATIC_SET_COUNT,.pSetLayouts=set_layouts};if((vr=vkAllocateDescriptorSets(c->device,&static_allocate,c->static_descriptor_sets))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"preallocate static descriptor sets",vr);}VkPipelineCacheCreateInfo pci={.sType=VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};if((vr=vkCreatePipelineCache(c->device,&pci,NULL,&c->pipeline_cache))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create pipeline cache",vr);}
    c->quant_q8k=(fg_vk_kernel){.file="fg_quantize_q8_k.spv",.bindings=2,.push_bytes=12};c->quant_q8=(fg_vk_kernel){.file="fg_quantize_q8_0.spv",.bindings=2,.push_bytes=12};c->quant_q4=(fg_vk_kernel){.file="fg_quantize_q4_0.spv",.bindings=2,.push_bytes=12};
    c->dequant_iq4nl=(fg_vk_kernel){.file="fg_dequantize_iq4_nl.spv",.bindings=2,.push_bytes=8};c->embedding=(fg_vk_kernel){.file="fg_embedding_q8_0.spv",.bindings=2,.push_bytes=16};c->embedding_batch=(fg_vk_kernel){.file="fg_embedding_q8_0_batch.spv",.bindings=3,.push_bytes=16};c->swiglu=(fg_vk_kernel){.file="fg_swiglu.spv",.bindings=3,.push_bytes=4};c->silu_scaled=(fg_vk_kernel){.file="fg_silu_scaled.spv",.bindings=2,.push_bytes=8};c->dense=(fg_vk_kernel){.file="fg_dense_q8_0_f32.spv",.bindings=3,.push_bytes=20};c->dense_f32=(fg_vk_kernel){.file="fg_dense_f32.spv",.bindings=3,.push_bytes=12};c->dense_bf16=(fg_vk_kernel){.file="fg_dense_bf16_f32.spv",.bindings=3,.push_bytes=12};
    c->rms=(fg_vk_kernel){.file="fg_group_rms_norm.spv",.bindings=3,.push_bytes=16};c->gr=(fg_vk_kernel){.file="fg_gr_mix.spv",.bindings=5,.push_bytes=12};c->hc_inject_partial=(fg_vk_kernel){.file="fg_hc_inject_partial.spv",.bindings=3,.push_bytes=16};c->gr_partial=(fg_vk_kernel){.file="fg_gr_mix_partial.spv",.bindings=5,.push_bytes=16};c->hc_finalize=(fg_vk_kernel){.file="fg_hc_finalize.spv",.bindings=3,.push_bytes=12};c->gr_write=(fg_vk_kernel){.file="fg_gr_write.spv",.bindings=4,.push_bytes=12};c->ple_gate=(fg_vk_kernel){.file="fg_ple_gate.spv",.bindings=4,.push_bytes=0};c->ple_gate_prefill=(fg_vk_kernel){.file="fg_ple_gate_prefill.spv",.bindings=4,.push_bytes=4};c->ple_conv=(fg_vk_kernel){.file="fg_ple_conv_decode.spv",.bindings=5,.push_bytes=0};c->ple_conv_prefill=(fg_vk_kernel){.file="fg_ple_conv_prefill.spv",.bindings=5,.push_bytes=4};c->ple_conv_add=(fg_vk_kernel){.file="fg_ple_conv_decode_add.spv",.bindings=6,.push_bytes=0};c->add=(fg_vk_kernel){.file="fg_add_f32.spv",.bindings=3,.push_bytes=4};c->gdn_conv=(fg_vk_kernel){.file="fg_gdn_conv_decode.spv",.bindings=4,.push_bytes=4};c->gdn_conv_prefill=(fg_vk_kernel){.file="fg_gdn_conv_prefill.spv",.bindings=4,.push_bytes=8};c->gdn_recurrent=(fg_vk_kernel){.file="fg_gdn_recurrent_decode.spv",.bindings=9,.push_bytes=16};c->gdn_recurrent_algebraic=(fg_vk_kernel){.file="fg_gdn_recurrent_algebraic.spv",.bindings=9,.push_bytes=16};c->gdn_recurrent_prefill=(fg_vk_kernel){.file="fg_gdn_recurrent_prefill.spv",.bindings=9,.push_bytes=20};c->gdn_prefill_qk_norm=(fg_vk_kernel){.file="fg_gdn_prefill_qk_norm.spv",.bindings=1,.push_bytes=8};c->gdn_prefill_recurrence=(fg_vk_kernel){.file="fg_gdn_prefill_recurrence.spv",.bindings=7,.push_bytes=4};c->gdn_prefill_output=(fg_vk_kernel){.file="fg_gdn_prefill_output.spv",.bindings=3,.push_bytes=8};
    c->argmax=(fg_vk_kernel){.file="fg_argmax_reduce.spv",.bindings=4,.push_bytes=4};
    c->dense_subgroup=(fg_vk_kernel){.file="fg_dense_q8_0_subgroup.spv",.bindings=3,.push_bytes=20};
    c->dense_cooked=(fg_vk_kernel){.file="fg_dense_q8_0_cooked.spv",.bindings=3,.push_bytes=20};
    c->mtp_vocab=(fg_vk_kernel){.file="fg_mtp_vocab_q8.spv",.bindings=5,.push_bytes=20};
    c->mtp_fc=(fg_vk_kernel){.file="fg_mtp_fc_hidden.spv",.bindings=3,.push_bytes=20};
    c->mtp_norm=(fg_vk_kernel){.file="fg_mtp_hc_norm.spv",.bindings=3,.push_bytes=4};
    c->dense_cooked_r8=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_r8.spv",.bindings=3,.push_bytes=20};
    c->dense_cooked_split=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_split.spv",.bindings=3,.push_bytes=24};
    c->dense_cooked_split_reduce=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_split_reduce.spv",.bindings=2,.push_bytes=8};
    c->dense_cooked_split_reduce_silu=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_split_reduce_silu.spv",.bindings=3,.push_bytes=12};
    c->moe_q5_1=(fg_vk_kernel){.file="fg_moe_q5_1_down.spv",.bindings=4,.push_bytes=28};c->moe_q5_1_cooked=(fg_vk_kernel){.file="fg_moe_q5_1_down_cooked.spv",.bindings=4,.push_bytes=28};c->moe_q8_0=(fg_vk_kernel){.file="fg_moe_q8_0_down.spv",.bindings=4,.push_bytes=28};c->moe_q8_0_cooked=(fg_vk_kernel){.file="fg_moe_q8_0_down_cooked.spv",.bindings=4,.push_bytes=28};c->moe_reduce=(fg_vk_kernel){.file="fg_moe_reduce.spv",.bindings=4,.push_bytes=12};c->kquant=(fg_vk_kernel){.file="fg_moe_kquant.spv",.bindings=4,.push_bytes=36};c->kquant_cooked=(fg_vk_kernel){.file="fg_moe_kquant_cooked.spv",.bindings=4,.push_bytes=32};
    c->moe_decode_gate_up=(fg_vk_kernel){.file="fg_moe_decode_gate_up.spv",.bindings=5,.push_bytes=32};
    c->moe_decode_down_reduce=(fg_vk_kernel){.file="fg_moe_decode_down_reduce.spv",.bindings=5,.push_bytes=28};c->moe_decode_down_q8=(fg_vk_kernel){.file="fg_moe_decode_down_q8.spv",.bindings=5,.push_bytes=28};c->moe_decode_shared_add=(fg_vk_kernel){.file="fg_moe_shared_add.spv",.bindings=4,.push_bytes=4};
    c->bench_stream=(fg_vk_kernel){.file="fg_bench_stream.spv",.bindings=3,.push_bytes=20};c->bench_dequant=(fg_vk_kernel){.file="fg_bench_dequant.spv",.bindings=3,.push_bytes=20};c->bench_dot_nored=(fg_vk_kernel){.file="fg_bench_dot_nored.spv",.bindings=3,.push_bytes=20};
    c->bench_stream_wide=(fg_vk_kernel){.file="fg_bench_stream_wide.spv",.bindings=3,.push_bytes=20};c->bench_stream_vec=(fg_vk_kernel){.file="fg_bench_stream_vec.spv",.bindings=3,.push_bytes=20};
    c->bench_cooked_layout=(fg_vk_kernel){.file="fg_bench_cooked_layout.spv",.bindings=3,.push_bytes=32};
    c->dense_cooked_tile=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_tile.spv",.bindings=3,.push_bytes=20};
    c->router_top10=(fg_vk_kernel){.file="fg_router_top10.spv",.bindings=3,.push_bytes=8};
    c->expert_major_pack=(fg_vk_kernel){.file="fg_expert_major_pack.spv",.bindings=2,.push_bytes=12};
    c->decode_tile_schedule=(fg_vk_kernel){.file="fg_decode_tile_schedule.spv",.bindings=2,.push_bytes=0};
    c->kquant_cooked_grouped=(fg_vk_kernel){.file="fg_moe_kquant_cooked_grouped.spv",.bindings=4,.push_bytes=24};c->kquant_cooked_grouped_int=(fg_vk_kernel){.file="fg_moe_kquant_cooked_grouped_int.spv",.bindings=4,.push_bytes=24};
    c->q5_1_cooked_grouped=(fg_vk_kernel){.file="fg_moe_q5_1_down_cooked_grouped.spv",.bindings=4,.push_bytes=24};
    c->q8_0_grouped=(fg_vk_kernel){.file="fg_moe_q8_0_down_grouped.spv",.bindings=4,.push_bytes=20};
    c->q8_0_cooked_grouped=(fg_vk_kernel){.file="fg_moe_q8_0_down_cooked_grouped.spv",.bindings=4,.push_bytes=24};
    c->moe_prefill_shard_reduce=(fg_vk_kernel){.file="fg_moe_prefill_shard_reduce.spv",.bindings=3,.push_bytes=8};
    c->moe_prefill_reduce=(fg_vk_kernel){.file="fg_moe_prefill_reduce.spv",.bindings=5,.push_bytes=8};
    c->apply_penalties=(fg_vk_kernel){.file="fg_apply_penalties.spv",.bindings=2,.push_bytes=16};
    c->qsa_prepare=(fg_vk_kernel){.file="fg_qsa_prepare.spv",.bindings=8,.push_bytes=0};c->qsa_prepare_prefill=(fg_vk_kernel){.file="fg_qsa_prepare_prefill.spv",.bindings=8,.push_bytes=4};c->qsa_index_prepare=(fg_vk_kernel){.file="fg_qsa_index_prepare.spv",.bindings=4,.push_bytes=0};c->qsa_index_prepare_prefill=(fg_vk_kernel){.file="fg_qsa_index_prepare_prefill.spv",.bindings=4,.push_bytes=4};c->qsa_record_commit=(fg_vk_kernel){.file="fg_qsa_record_commit.spv",.bindings=6,.push_bytes=24};c->qsa_record_gather=(fg_vk_kernel){.file="fg_qsa_record_gather.spv",.bindings=3,.push_bytes=20};c->qsa_score=(fg_vk_kernel){.file="fg_qsa_index_score.spv",.bindings=6,.push_bytes=28};c->qsa_attention=(fg_vk_kernel){.file="fg_qsa_attention.spv",.bindings=4,.push_bytes=4};c->qsa_attention_split=(fg_vk_kernel){.file="fg_qsa_attention_split.spv",.bindings=3,.push_bytes=12};c->qsa_decode_attention_split=(fg_vk_kernel){.file="fg_qsa_decode_attention_split.spv",.bindings=3,.push_bytes=8};c->qsa_attention_merge=(fg_vk_kernel){.file="fg_qsa_attention_merge.spv",.bindings=3,.push_bytes=4};c->qsa_attention_split_batch=(fg_vk_kernel){.file="fg_qsa_attention_split_batch.spv",.bindings=4,.push_bytes=16};c->qsa_attention_merge_batch=(fg_vk_kernel){.file="fg_qsa_attention_merge_batch.spv",.bindings=3,.push_bytes=12};c->qsa_record_gather_batch=(fg_vk_kernel){.file="fg_qsa_record_gather_batch.spv",.bindings=3,.push_bytes=20};c->qsa_select_resolve=(fg_vk_kernel){.file="fg_qsa_select_resolve.spv",.bindings=4,.push_bytes=12};c->qsa_resident_commit=(fg_vk_kernel){.file="fg_qsa_resident_record_commit.spv",.bindings=8,.push_bytes=16};c->qsa_resident_select=(fg_vk_kernel){.file="fg_qsa_resident_select.spv",.bindings=7,.push_bytes=20};c->qsa_resident_merge=(fg_vk_kernel){.file="fg_qsa_resident_topk_merge.spv",.bindings=4,.push_bytes=16};c->qsa_resident_attention=(fg_vk_kernel){.file="fg_qsa_resident_attention.spv",.bindings=6,.push_bytes=20};
    c->topk_v2=(fg_vk_kernel){.file="fg_topk_reduce_v2.spv",.bindings=4,.push_bytes=8};
    c->mtp_attention=(fg_vk_kernel){.file="fg_mtp_attention.spv",.bindings=7,.push_bytes=4};
    c->mtp_attention_indirect=(fg_vk_kernel){.file="fg_mtp_attention_indirect.spv",.bindings=8,.push_bytes=4};
    c->mtp_attention_split=(fg_vk_kernel){.file="fg_mtp_attention_split.spv",.bindings=8,.push_bytes=12};
    c->mtp_cache_kv=(fg_vk_kernel){.file="fg_mtp_cache_kv.spv",.bindings=6,.push_bytes=4};
    c->topk_select=(fg_vk_kernel){.file="fg_topk_select.spv",.bindings=4,.push_bytes=8};
    c->topk_select_fallback=(fg_vk_kernel){.file="fg_topk_select_fallback.spv",.bindings=4,.push_bytes=8};
    *out=c;return FG_OK;
}

void fg_vk_close(fg_vk_context *c){
    if(!c)return;
    if(c->device)vkDeviceWaitIdle(c->device);
    fg_vk_tensor_destroy(c->roof_partials);
    fg_vk_tensor_destroy(c->roof_gdn);
    fg_vk_tensor_destroy(c->roof_expert_partials);
    if(c->layer_query_pool)vkDestroyQueryPool(c->device,c->layer_query_pool,NULL);
    free(c->layer_samples);
    destroy_kernel(c,&c->moe_decode_shared_add);destroy_kernel(c,&c->moe_decode_down_reduce);destroy_kernel(c,&c->moe_decode_gate_up);destroy_kernel(c,&c->moe_prefill_shard_reduce);destroy_kernel(c,&c->moe_prefill_reduce);destroy_kernel(c,&c->q8_0_grouped);destroy_kernel(c,&c->q8_0_cooked_grouped);destroy_kernel(c,&c->q5_1_cooked_grouped);destroy_kernel(c,&c->kquant_cooked_grouped_int);destroy_kernel(c,&c->kquant_cooked_grouped);destroy_kernel(c,&c->decode_tile_schedule);destroy_kernel(c,&c->expert_major_pack);destroy_kernel(c,&c->router_top10);destroy_kernel(c,&c->dense_cooked_tile);destroy_kernel(c,&c->bench_cooked_layout);destroy_kernel(c,&c->bench_stream_vec);destroy_kernel(c,&c->bench_stream_wide);destroy_kernel(c,&c->bench_dot_nored);destroy_kernel(c,&c->bench_dequant);destroy_kernel(c,&c->bench_stream);destroy_kernel(c,&c->kquant);destroy_kernel(c,&c->moe_reduce);destroy_kernel(c,&c->moe_q8_0);destroy_kernel(c,&c->moe_q8_0_cooked);destroy_kernel(c,&c->moe_q5_1);destroy_kernel(c,&c->topk_v2);destroy_kernel(c,&c->topk_select);destroy_kernel(c,&c->topk_select_fallback);destroy_kernel(c,&c->apply_penalties);destroy_kernel(c,&c->qsa_resident_attention);destroy_kernel(c,&c->qsa_resident_merge);destroy_kernel(c,&c->qsa_resident_select);destroy_kernel(c,&c->qsa_resident_commit);destroy_kernel(c,&c->qsa_attention_split);destroy_kernel(c,&c->qsa_decode_attention_split);destroy_kernel(c,&c->qsa_attention_merge);destroy_kernel(c,&c->qsa_attention_split_batch);destroy_kernel(c,&c->qsa_attention_merge_batch);destroy_kernel(c,&c->qsa_record_gather_batch);destroy_kernel(c,&c->qsa_select_resolve);destroy_kernel(c,&c->qsa_attention);destroy_kernel(c,&c->qsa_score);destroy_kernel(c,&c->qsa_record_gather);destroy_kernel(c,&c->qsa_record_commit);destroy_kernel(c,&c->qsa_index_prepare_prefill);destroy_kernel(c,&c->qsa_index_prepare);destroy_kernel(c,&c->qsa_prepare_prefill);destroy_kernel(c,&c->qsa_prepare);destroy_kernel(c,&c->gdn_prefill_output);destroy_kernel(c,&c->gdn_prefill_recurrence);destroy_kernel(c,&c->gdn_prefill_qk_norm);destroy_kernel(c,&c->gdn_recurrent_prefill);destroy_kernel(c,&c->gdn_recurrent_algebraic);destroy_kernel(c,&c->gdn_recurrent);destroy_kernel(c,&c->gdn_conv_prefill);destroy_kernel(c,&c->gdn_conv);destroy_kernel(c,&c->add);destroy_kernel(c,&c->ple_conv_prefill);destroy_kernel(c,&c->ple_conv);destroy_kernel(c,&c->ple_conv_add);destroy_kernel(c,&c->ple_gate_prefill);destroy_kernel(c,&c->ple_gate);destroy_kernel(c,&c->gr_write);destroy_kernel(c,&c->hc_finalize);destroy_kernel(c,&c->gr_partial);destroy_kernel(c,&c->hc_inject_partial);destroy_kernel(c,&c->gr);destroy_kernel(c,&c->rms);destroy_kernel(c,&c->dense_cooked);destroy_kernel(c,&c->dense_subgroup);destroy_kernel(c,&c->dense_bf16);destroy_kernel(c,&c->dense_f32);destroy_kernel(c,&c->dense);destroy_kernel(c,&c->silu_scaled);destroy_kernel(c,&c->swiglu);destroy_kernel(c,&c->embedding_batch);destroy_kernel(c,&c->embedding);destroy_kernel(c,&c->dequant_iq4nl);destroy_kernel(c,&c->quant_q4);destroy_kernel(c,&c->quant_q8);destroy_kernel(c,&c->quant_q8k);if(c->profile_query_pool)vkDestroyQueryPool(c->device,c->profile_query_pool,NULL);if(c->pipeline_cache)vkDestroyPipelineCache(c->device,c->pipeline_cache,NULL);if(c->descriptor_pool)vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);if(c->static_descriptor_pool)vkDestroyDescriptorPool(c->device,c->static_descriptor_pool,NULL);if(c->descriptor_set_layout)vkDestroyDescriptorSetLayout(c->device,c->descriptor_set_layout,NULL);if(c->expert_fence)vkDestroyFence(c->device,c->expert_fence,NULL);if(c->fence)vkDestroyFence(c->device,c->fence,NULL);if(c->copy_fence)vkDestroyFence(c->device,c->copy_fence,NULL);if(c->command_pool)vkDestroyCommandPool(c->device,c->command_pool,NULL);if(c->copy_pool)vkDestroyCommandPool(c->device,c->copy_pool,NULL);if(c->device)vkDestroyDevice(c->device,NULL);if(c->instance)vkDestroyInstance(c->instance,NULL);free(c);
}
const char *fg_vk_device_name(const fg_vk_context *c){return c?c->device_name:"";}
bool fg_vk_integer_dot_product_enabled(const fg_vk_context *c){return c&&c->integer_dot_product;}
uint32_t fg_vk_active_compute_units(const fg_vk_context *c){return c?c->active_compute_units:0u;}
uint32_t fg_vk_hc_inject_pieces(const fg_vk_context *c){
    uint32_t maximum=FG_HIDDEN_SIZE*FG_GROUP_SIZE/256u;
    uint32_t units=c?c->active_compute_units:0u;
    if(!units||units>maximum)units=maximum;
    return units;
}

fg_status fg_vk_profile_begin(fg_vk_context *c,fg_error *err){if(!c||c->batch_depth||c->profile_active||c->group_active){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile begin");return FG_ERR_ARGUMENT;}fg_status drain=vk_pipeline_drain(c,err);if(drain!=FG_OK)return drain;if(!c->timestamp_valid_bits||c->timestamp_period<=0.0f){fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan compute queue has no timestamp support");return FG_ERR_UNAVAILABLE;}if(!c->profile_query_pool){VkQueryPoolCreateInfo create={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=FG_VK_PROFILE_QUERY_COUNT};VkResult vr=vkCreateQueryPool(c->device,&create,NULL,&c->profile_query_pool);if(vr!=VK_SUCCESS)return vk_error(err,"create timestamp query pool",vr);vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin timestamp query prime",vr);vkCmdResetQueryPool(c->command,c->profile_query_pool,0,FG_VK_PROFILE_QUERY_COUNT);if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS)return vk_error(err,"end timestamp query prime",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit timestamp query prime",vr);c->counters.submissions++;if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait timestamp query prime",vr);}memset(&c->profile,0,sizeof(c->profile));c->profile_scope="unscoped";c->profile_active=true;return FG_OK;}
fg_status fg_vk_profile_set_scope(fg_vk_context *c,const char *scope,fg_error *err){if(!c||!c->profile_active||!scope||!*scope){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile scope");return FG_ERR_ARGUMENT;}c->profile_scope=scope;return FG_OK;}
fg_status fg_vk_profile_end(fg_vk_context *c,fg_vk_profile *profile,fg_error *err){if(!c||!profile||c->batch_depth||!c->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile end");return FG_ERR_ARGUMENT;}*profile=c->profile;c->profile_active=false;return FG_OK;}
bool fg_vk_profile_active(const fg_vk_context *c){return c&&c->profile_active;}
void fg_vk_get_counters(const fg_vk_context *c,fg_vk_counters *counters){if(!counters)return;if(c)*counters=c->counters;else memset(counters,0,sizeof(*counters));}
void fg_vk_get_memory_stats(const fg_vk_context *c,fg_vk_memory_stats *stats){if(!stats)return;if(c)*stats=c->memory_stats;else memset(stats,0,sizeof(*stats));}

fg_status fg_vk_tensor_create(fg_vk_context *c,uint64_t bytes,fg_vk_tensor **out,fg_error *err){
    if(!c||!out||bytes==0){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor allocation");return FG_ERR_ARGUMENT;}*out=NULL;fg_vk_allocation *a=calloc(1,sizeof(*a));fg_vk_tensor *t=calloc(1,sizeof(*t));if(!a||!t){free(a);free(t);fg_error_set(err,FG_ERR_OOM,"allocate Vulkan tensor metadata");return FG_ERR_OOM;}
    VkBufferCreateInfo bc={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};VkResult vr=vkCreateBuffer(c->device,&bc,NULL,&a->buffer);if(vr!=VK_SUCCESS){free(a);free(t);return vk_error(err,"create Vulkan tensor buffer",vr);}VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(c->device,a->buffer,&requirements);uint32_t type=UINT32_MAX;VkMemoryPropertyFlags wanted=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;for(uint32_t i=0;i<c->memory.memoryTypeCount;i++)if((requirements.memoryTypeBits&(1u<<i))&&(c->memory.memoryTypes[i].propertyFlags&wanted)==wanted){if(type==UINT32_MAX||c->memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)type=i;}if(type==UINT32_MAX){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan device has no coherent host-visible memory");return FG_ERR_UNAVAILABLE;}VkMemoryAllocateInfo ma={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=requirements.size,.memoryTypeIndex=type};if((vr=vkAllocateMemory(c->device,&ma,NULL,&a->memory))!=VK_SUCCESS){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"allocate Vulkan tensor memory",vr);}if((vr=vkBindBufferMemory(c->device,a->buffer,a->memory,0))!=VK_SUCCESS||(vr=vkMapMemory(c->device,a->memory,0,VK_WHOLE_SIZE,0,&a->mapped))!=VK_SUCCESS){if(a->mapped)vkUnmapMemory(c->device,a->memory);vkFreeMemory(c->device,a->memory,NULL);vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"map Vulkan tensor memory",vr);}a->bytes=bytes;a->allocated_bytes=requirements.size;a->references=1;t->context=c;t->allocation=a;t->bytes=bytes;c->memory_stats.requested_live_bytes+=bytes;c->memory_stats.allocated_live_bytes+=requirements.size;c->memory_stats.allocation_count++;c->memory_stats.live_allocations++;if(c->memory_stats.requested_live_bytes>c->memory_stats.requested_peak_bytes)c->memory_stats.requested_peak_bytes=c->memory_stats.requested_live_bytes;if(c->memory_stats.allocated_live_bytes>c->memory_stats.allocated_peak_bytes)c->memory_stats.allocated_peak_bytes=c->memory_stats.allocated_live_bytes;*out=t;return FG_OK;
}

fg_status fg_vk_tensor_create_cached(fg_vk_context *c,uint64_t bytes,fg_vk_tensor **out,fg_error *err){
    if(!c||!out||bytes==0){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor allocation");return FG_ERR_ARGUMENT;}*out=NULL;fg_vk_allocation *a=calloc(1,sizeof(*a));fg_vk_tensor *t=calloc(1,sizeof(*t));if(!a||!t){free(a);free(t);fg_error_set(err,FG_ERR_OOM,"allocate Vulkan tensor metadata");return FG_ERR_OOM;}
    VkBufferCreateInfo bc={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkResult vr=vkCreateBuffer(c->device,&bc,NULL,&a->buffer);if(vr!=VK_SUCCESS){free(a);free(t);return vk_error(err,"create Vulkan tensor buffer",vr);}
    VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(c->device,a->buffer,&requirements);
    uint32_t type=UINT32_MAX;VkMemoryPropertyFlags wanted=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for(uint32_t i=0;i<c->memory.memoryTypeCount;i++)if((requirements.memoryTypeBits&(1u<<i))&&(c->memory.memoryTypes[i].propertyFlags&wanted)==wanted&&(c->memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT)){type=i;break;}
    if(type==UINT32_MAX)for(uint32_t i=0;i<c->memory.memoryTypeCount;i++)if((requirements.memoryTypeBits&(1u<<i))&&(c->memory.memoryTypes[i].propertyFlags&wanted)==wanted){if(type==UINT32_MAX||c->memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)type=i;}
    if(type==UINT32_MAX){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan device has no coherent host-visible memory");return FG_ERR_UNAVAILABLE;}
    VkMemoryAllocateInfo ma={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=requirements.size,.memoryTypeIndex=type};
    if((vr=vkAllocateMemory(c->device,&ma,NULL,&a->memory))!=VK_SUCCESS){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"allocate Vulkan tensor memory",vr);}
    if((vr=vkBindBufferMemory(c->device,a->buffer,a->memory,0))!=VK_SUCCESS||(vr=vkMapMemory(c->device,a->memory,0,VK_WHOLE_SIZE,0,&a->mapped))!=VK_SUCCESS){if(a->mapped)vkUnmapMemory(c->device,a->memory);vkFreeMemory(c->device,a->memory,NULL);vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"map Vulkan tensor memory",vr);}
    a->bytes=bytes;a->allocated_bytes=requirements.size;a->references=1;t->context=c;t->allocation=a;t->bytes=bytes;
    c->memory_stats.requested_live_bytes+=bytes;c->memory_stats.allocated_live_bytes+=requirements.size;c->memory_stats.allocation_count++;c->memory_stats.live_allocations++;
    if(c->memory_stats.requested_live_bytes>c->memory_stats.requested_peak_bytes)c->memory_stats.requested_peak_bytes=c->memory_stats.requested_live_bytes;
    if(c->memory_stats.allocated_live_bytes>c->memory_stats.allocated_peak_bytes)c->memory_stats.allocated_peak_bytes=c->memory_stats.allocated_live_bytes;
    *out=t;return FG_OK;
}

fg_status fg_vk_tensor_view(fg_vk_tensor *base,uint64_t offset,uint64_t bytes,fg_vk_tensor **out,fg_error *err){if(!out||!tensor_range(base,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor view");return FG_ERR_ARGUMENT;}fg_vk_tensor *view=calloc(1,sizeof(*view));if(!view){fg_error_set(err,FG_ERR_OOM,"allocate Vulkan tensor view");return FG_ERR_OOM;}base->allocation->references++;view->context=base->context;view->allocation=base->allocation;view->offset=base->offset+offset;view->bytes=bytes;view->format=base->format;view->is_view=true;*out=view;return FG_OK;}
fg_status fg_vk_tensor_view_rebind(fg_vk_tensor *view,fg_vk_tensor *base,uint64_t offset,uint64_t bytes,fg_error *err){
    if(!view||!view->is_view||!tensor_range(base,offset,bytes)||view->context!=base->context||view->allocation!=base->allocation){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor view rebind");return FG_ERR_ARGUMENT;
    }
    view->offset=base->offset+offset;view->bytes=bytes;view->format=base->format;return FG_OK;
}
void fg_vk_tensor_destroy(fg_vk_tensor *t){if(!t)return;fg_vk_allocation *a=t->allocation;if(--a->references==0){t->context->memory_stats.requested_live_bytes-=a->bytes;t->context->memory_stats.allocated_live_bytes-=a->allocated_bytes;t->context->memory_stats.live_allocations--;vkUnmapMemory(t->context->device,a->memory);vkFreeMemory(t->context->device,a->memory,NULL);vkDestroyBuffer(t->context->device,a->buffer,NULL);free(a);}free(t);}
uint64_t fg_vk_tensor_bytes(const fg_vk_tensor *t){return t?t->bytes:0;}uint64_t fg_vk_tensor_allocation_bytes(const fg_vk_tensor *t){return t?t->allocation->allocated_bytes:0;}void *fg_vk_tensor_map(fg_vk_tensor *t){return t?(uint8_t *)t->allocation->mapped+t->offset:NULL;}const void *fg_vk_tensor_const_map(const fg_vk_tensor *t){return t?(const uint8_t *)t->allocation->mapped+t->offset:NULL;}void fg_vk_tensor_set_format(fg_vk_tensor *t,fg_vk_tensor_format format){if(t)t->format=format;}fg_vk_tensor_format fg_vk_tensor_get_format(const fg_vk_tensor *t){return t?t->format:FG_VK_TENSOR_FORMAT_DEFAULT;}
fg_status fg_vk_tensor_residency_canary(fg_vk_tensor *t,uint64_t *touched_bytes,fg_error *err){
    if(touched_bytes)*touched_bytes=0;
    if(!t||t->is_view||t->context->batch_depth){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan residency canary tensor");
        return FG_ERR_ARGUMENT;
    }
    t->context->counters.residency_canary_calls++;
    const uint64_t stride=4096u,count=(t->bytes+stride-1u)/stride;
    if(count>SIZE_MAX){fg_error_set(err,FG_ERR_LIMIT,"Vulkan canary exceeds address space");return FG_ERR_LIMIT;}
    uint8_t *saved=malloc((size_t)count);
    if(!saved){fg_error_set(err,FG_ERR_OOM,"allocate Vulkan residency canary");return FG_ERR_OOM;}
    volatile uint8_t *mapped=t->allocation->mapped;
    for(uint64_t i=0;i<count;i++){uint64_t offset=i*stride;saved[i]=mapped[offset];mapped[offset]=(uint8_t)(saved[i]^0xa5u);}
    fg_status status=FG_OK;
    for(uint64_t i=0;i<count;i++){uint64_t offset=i*stride;if(mapped[offset]!=(uint8_t)(saved[i]^0xa5u)){fg_error_set(err,FG_ERR_MISMATCH,"Vulkan residency canary verification failed at page %llu",(unsigned long long)i);status=FG_ERR_MISMATCH;break;}}
    for(uint64_t i=0;i<count;i++)mapped[i*stride]=saved[i];
    free(saved);if(status==FG_OK&&touched_bytes)*touched_bytes=t->bytes;return status;
}
fg_status fg_vk_tensor_write(fg_vk_tensor *t,uint64_t offset,const void *data,uint64_t bytes,fg_error *err){if(!data||!tensor_range(t,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor write");return FG_ERR_ARGUMENT;}memcpy((uint8_t *)t->allocation->mapped+t->offset+offset,data,(size_t)bytes);return FG_OK;}
/* Host-visible decode buffers are mapped write-combining, where ordinary
 * cache-line loads stall on every line (~100 MB/s measured for a 40 KiB
 * hyper state).  MOVNTDQA streams the lines instead and restores most of the
 * DRAM read bandwidth; the instruction is a no-op hint on cached memory. */
#if defined(__x86_64__)||defined(__i386__)
__attribute__((target("sse4.1")))
static void vk_stream_read_sse41(void *destination,const void *source,size_t bytes){
    uint8_t *out=destination;const uint8_t *in=source;size_t i=0;
    for(;i+64u<=bytes;i+=64u){
        __m128i a=_mm_stream_load_si128((__m128i *)(uintptr_t)(in+i));
        __m128i b=_mm_stream_load_si128((__m128i *)(uintptr_t)(in+i+16u));
        __m128i c=_mm_stream_load_si128((__m128i *)(uintptr_t)(in+i+32u));
        __m128i d=_mm_stream_load_si128((__m128i *)(uintptr_t)(in+i+48u));
        _mm_storeu_si128((__m128i *)(out+i),a);
        _mm_storeu_si128((__m128i *)(out+i+16u),b);
        _mm_storeu_si128((__m128i *)(out+i+32u),c);
        _mm_storeu_si128((__m128i *)(out+i+48u),d);
    }
    for(;i+16u<=bytes;i+=16u)
        _mm_storeu_si128((__m128i *)(out+i),
                         _mm_stream_load_si128((__m128i *)(uintptr_t)(in+i)));
    if(i<bytes)memcpy(out+i,in+i,bytes-i);
}
static void vk_read_mapped(void *destination,const void *source,size_t bytes){
    if(__builtin_cpu_supports("sse4.1")){vk_stream_read_sse41(destination,source,bytes);return;}
    memcpy(destination,source,bytes);
}
#else
static void vk_read_mapped(void *destination,const void *source,size_t bytes){memcpy(destination,source,bytes);}
#endif
fg_status fg_vk_tensor_read(const fg_vk_tensor *t,uint64_t offset,void *data,uint64_t bytes,fg_error *err){if(!data||!tensor_range(t,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor read");return FG_ERR_ARGUMENT;}vk_read_mapped(data,(const uint8_t *)t->allocation->mapped+t->offset+offset,(size_t)bytes);return FG_OK;}

/* One-shot device-side copy used by the depth-B owner session checkpoints.  The
 * state tensors live in device-local coherent memory whose host reads run at
 * uncached bandwidth; copying them with the GPU keeps a per-step snapshot at
 * transfer speed instead of a 100+ ms host readback. */
static fg_status record_tensor_copies(fg_vk_context *c,fg_vk_tensor *const *dst,
                             const fg_vk_tensor *const *src,uint32_t count,fg_error *err){
    if(c&&c->group_active){
        fg_error_set(err,FG_ERR_ARGUMENT,"tensor copy cannot bypass a submit group");
        return FG_ERR_ARGUMENT;
    }
    if(!c||c->batch_depth||count>FG_LAYER_COUNT*2u+2u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor copy");
        return FG_ERR_ARGUMENT;
    }
    if(!count)return FG_OK;
    if(!dst||!src){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor copy");
        return FG_ERR_ARGUMENT;
    }
    for(uint32_t i=0;i<count;i++)
        if(!tensor_on_context(c,dst[i])||!tensor_on_context(c,src[i])||
           dst[i]->bytes!=src[i]->bytes||dst[i]->bytes==0u){
            fg_error_set(err,FG_ERR_ARGUMENT,"Vulkan tensor copy pair %u is not size-matched",i);
            return FG_ERR_ARGUMENT;
        }
    fg_status status=fg_vk_static_drain(c,err);
    if(status!=FG_OK)return status;
    VkResult vr;
    if(!c->copy_pool){
        VkCommandPoolCreateInfo pool={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT|VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex=c->queue_family};
        if((vr=vkCreateCommandPool(c->device,&pool,NULL,&c->copy_pool))!=VK_SUCCESS)
            return vk_error(err,"create tensor copy command pool",vr);
        VkCommandBufferAllocateInfo allocate={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool=c->copy_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
        if((vr=vkAllocateCommandBuffers(c->device,&allocate,&c->copy_command))!=VK_SUCCESS)
            return vk_error(err,"allocate tensor copy command buffer",vr);
        VkFenceCreateInfo fence={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if((vr=vkCreateFence(c->device,&fence,NULL,&c->copy_fence))!=VK_SUCCESS)
            return vk_error(err,"create tensor copy fence",vr);
    }
    vkResetCommandBuffer(c->copy_command,0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if((vr=vkBeginCommandBuffer(c->copy_command,&begin))!=VK_SUCCESS)
        return vk_error(err,"begin tensor copy command",vr);
    VkMemoryBarrier pre={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|
                       VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(c->copy_command,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT|VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&pre,0,NULL,0,NULL);
    for(uint32_t i=0;i<count;i++){
        VkBufferCopy region={.srcOffset=src[i]->offset,.dstOffset=dst[i]->offset,
            .size=src[i]->bytes};
        vkCmdCopyBuffer(c->copy_command,src[i]->allocation->buffer,
                        dst[i]->allocation->buffer,1,&region);
    }
    VkMemoryBarrier post={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->copy_command,VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&post,0,NULL,0,NULL);
    if((vr=vkEndCommandBuffer(c->copy_command))!=VK_SUCCESS)
        return vk_error(err,"end tensor copy command",vr);
    return FG_OK;
}
fg_status fg_vk_copy_tensors(fg_vk_context *c,fg_vk_tensor *const *dst,
    const fg_vk_tensor *const *src,uint32_t count,fg_error *err){
    fg_status status=record_tensor_copies(c,dst,src,count,err);
    if(status!=FG_OK||!count)return status;
    VkResult vr;
    vkResetFences(c->device,1,&c->copy_fence);
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,
        .pCommandBuffers=&c->copy_command};
    if((vr=vkQueueSubmit(c->queue,1,&submit,c->copy_fence))!=VK_SUCCESS)
        return vk_error(err,"submit tensor copy command",vr);
    c->counters.submissions++;
    if((vr=vkWaitForFences(c->device,1,&c->copy_fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)
        return vk_error(err,"wait for tensor copy",vr);
    return FG_OK;
}

static fg_status dispatch_impl(fg_vk_context *c,fg_vk_kernel *kernel,const fg_vk_tensor *const *tensors,const void *push,uint32_t gx,uint32_t gy,uint32_t gz,bool batch_barrier,fg_error *err){
    if(c->group_active&&!c->batch_depth){
        fg_error_set(err,FG_ERR_ARGUMENT,"submit-group dispatch requires an active batch");
        return FG_ERR_ARGUMENT;
    }
    if(c->layer_clock_recording){
        fg_vk_layer_sample *sample=&c->layer_samples[c->layer_clock_current];
        if(sample->dispatch_count>=FG_VK_LAYER_DISPATCHES||kernel->push_bytes>128u){
            fg_error_set(err,FG_ERR_LIMIT,"layer dispatch ledger overflow");return FG_ERR_LIMIT;
        }
        fg_vk_layer_dispatch *record=&sample->dispatches[sample->dispatch_count++];
        *record=(fg_vk_layer_dispatch){.name=kernel->file,.grid={gx,gy,gz},
            .push_words=kernel->push_bytes/4u,.bindings=kernel->bindings,
            .barrier=batch_barrier&&c->batch_has_dispatch&&!c->barrier_elide_next};
        if(push)memcpy(record->push,push,kernel->push_bytes);
        for(uint32_t b=0;b<kernel->bindings;b++)record->bytes[b]=tensors[b]->bytes;
    }
    fg_status status=create_kernel(c,kernel,err);if(status!=FG_OK)return status;double op_begin=0.0;if(c->profile_active)op_begin=vk_wall_seconds();
    /* Static bases tag a separate pool; slots 4..7 exceed the dynamic pool's range. */
    if(c->batch_depth&&c->batch_set_base==0u&&c->batch_set_count>=FG_VK_BATCH_MAX_SETS){fg_error_set(err,FG_ERR_LIMIT,"Vulkan batch exceeded %u descriptor sets",FG_VK_BATCH_MAX_SETS);return FG_ERR_LIMIT;}
    uint32_t set_index=c->batch_depth?c->batch_set_base+c->batch_set_count:0u;VkDescriptorSet set;fg_vk_descriptor_cache *cache;bool *cache_valid;if(c->batch_depth&&c->batch_set_base!=0u){uint32_t slot=(c->batch_set_base-FG_VK_STATIC_SET_BASE)/FG_VK_STATIC_SET_STRIDE;if(slot>=FG_VK_STATIC_SLOTS||c->batch_set_count>=FG_VK_STATIC_SET_STRIDE){fg_error_set(err,FG_ERR_LIMIT,"static recording crossed its descriptor window");return FG_ERR_LIMIT;}uint32_t static_index=slot*FG_VK_STATIC_SET_STRIDE+c->batch_set_count;set=c->static_descriptor_sets[static_index];cache=&c->static_descriptor_cache[static_index];cache_valid=&c->static_descriptor_cache_valid[static_index];}else{set=c->descriptor_sets[set_index];cache=&c->descriptor_cache[set_index];cache_valid=&c->descriptor_cache_valid[set_index];if(set_index>=FG_VK_STATIC_SET_BASE){const char *trace=getenv("FG_SET_TRACE");static bool crossed=false;if(trace&&*trace&&strcmp(trace,"0")!=0&&!crossed){crossed=true;fprintf(stderr,"FG_SET_CROSS dynamic descriptor epoch reached set %u\n",set_index);}}}VkResult vr;bool cached=*cache_valid&&cache->bindings==kernel->bindings;for(uint32_t i=0;cached&&i<kernel->bindings;i++)cached=cache->buffer[i]==tensors[i]->allocation->buffer&&cache->offset[i]==tensors[i]->offset&&cache->range[i]==tensors[i]->bytes;if(!cached){VkDescriptorBufferInfo info[16];VkWriteDescriptorSet write[16];for(uint32_t i=0;i<kernel->bindings;i++){info[i]=(VkDescriptorBufferInfo){.buffer=tensors[i]->allocation->buffer,.offset=tensors[i]->offset,.range=tensors[i]->bytes};write[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info[i]};}vkUpdateDescriptorSets(c->device,kernel->bindings,write,0,NULL);cache->bindings=kernel->bindings;for(uint32_t i=0;i<kernel->bindings;i++){cache->buffer[i]=tensors[i]->allocation->buffer;cache->offset[i]=tensors[i]->offset;cache->range[i]=tensors[i]->bytes;}*cache_valid=true;}
    bool elide=c->barrier_elide_next;c->barrier_elide_next=false;
    if(c->batch_depth){if(batch_barrier&&c->batch_has_dispatch&&!elide){VkMemoryBarrier bar={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};if(c->profile_active){double barrier_begin=vk_wall_seconds();vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&bar,0,NULL,0,NULL);c->profile.barrier_ms+=(vk_wall_seconds()-barrier_begin)*1000.0;}else vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&bar,0,NULL,0,NULL);}vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);bool profiled=profile_dispatch_begin(c,kernel);vkCmdDispatch(c->command,gx,gy,gz);c->counters.dispatches++;c->batch_has_dispatch=true;profile_dispatch_end(c,profiled);c->batch_set_count++;if(c->profile_active)c->profile.op_ms+=(vk_wall_seconds()-op_begin)*1000.0;return FG_OK;}
    status=vk_pipeline_drain(c,err);if(status!=FG_OK)return status;
    vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin compute command",vr);profile_command_begin(c);VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);bool profiled=profile_dispatch_begin(c,kernel);vkCmdDispatch(c->command,gx,gy,gz);c->counters.dispatches++;profile_dispatch_end(c,profiled);VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);profile_command_end(c);if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS)return vk_error(err,"end compute command",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};VkSemaphore sem_wait,sem_signal;VkPipelineStageFlags sem_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;vk_submit_attach(c,false,&sem_wait,&sem_signal,&sem_stage,&submit);double submit_begin=c->profile_active?vk_wall_seconds():0.0;if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit compute command",vr);if(c->profile_active)c->profile.submit_ms+=(vk_wall_seconds()-submit_begin)*1000.0;c->counters.submissions++;double wait_begin=vk_wall_seconds();if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait for compute command",vr);vk_profile_add_wait(c,vk_wall_seconds()-wait_begin);if(c->profile_active)c->profile.op_ms+=(vk_wall_seconds()-op_begin)*1000.0;status=profile_resolve(c,err);return status;
}

static bool dense_cooked_rows8_shape(uint32_t input_width,uint32_t output_width,uint32_t tokens){return tokens==1u&&((input_width==320u&&output_width==10240u)||(input_width==2560u&&(output_width==10240u||output_width==640u||output_width==2560u||output_width==12288u||output_width==512u||output_width==FG_Q38_VOCAB_SIZE))||(input_width==640u&&output_width==2560u)||(input_width==6144u&&output_width==2560u));}
static fg_status dispatch(fg_vk_context *c,fg_vk_kernel *kernel,const fg_vk_tensor *const *tensors,const void *push,uint32_t gx,uint32_t gy,uint32_t gz,fg_error *err){
    if(kernel==&c->dense_cooked&&push){
        const uint32_t *parameters=push;
        uint32_t output_width=parameters[0],tokens=parameters[1],input_width=parameters[2]*FG_QK8_0;
        if(dense_cooked_rows8_shape(input_width,output_width,tokens)){
            kernel=&c->dense_cooked_r8;gx=(output_width+7u)/8u;
        }
    }
    if(kernel==&c->qsa_decode_attention_split&&roof_enabled(c,2u)){
        const char *value=getenv("FG_ROOF_QSA");
        uint32_t v=value?(uint32_t)strtoul(value,NULL,0):2u;
        if(v==1u){kernel=roof_kernel(c,5u,"fg_roof_qsa.spv",3u,8u);gx=24u;}
        else if(v==2u){kernel=roof_kernel(c,10u,"fg_roof_qsa_h3.spv",3u,8u);gx=8u;}
        else if(v==3u){kernel=roof_kernel(c,11u,"fg_roof_qsa_h6.spv",3u,8u);gx=4u;}
        else if(v==4u){kernel=roof_kernel(c,12u,"fg_roof_qsa_h1.spv",3u,8u);gx=24u;}
    }else if(kernel==&c->gdn_recurrent_algebraic&&roof_enabled(c,4u)){
        const char *value=getenv("FG_ROOF_RECURRENT");
        uint32_t recurrent_variant=value?(uint32_t)strtoul(value,NULL,0):0u;
        if(recurrent_variant==3u&&push&&((const uint32_t *)push)[0]==48u)
            return dispatch_impl(c,roof_kernel(c,87u,"fg_roof_recurrent_single.spv",9u,16u),
                tensors,push,48u,1u,1u,true,err);
        if((recurrent_variant==1u||recurrent_variant==2u)&&push&&((const uint32_t *)push)[0]==48u){
            fg_status status=FG_OK;
            if(!c->roof_gdn)status=fg_vk_tensor_create(c,48u*128u*4u,&c->roof_gdn,err);
            if(status!=FG_OK)return status;
            const fg_vk_tensor *first[9];
            for(uint32_t i=0;i<8u;i++)first[i]=tensors[i];
            first[8]=c->roof_gdn;
            status=dispatch_impl(c,roof_kernel(c,recurrent_variant==1u?53u:73u,
                recurrent_variant==1u?"fg_roof_recurrent_split.spv":"fg_roof_recurrent_vec.spv",9u,16u),
                first,push,48u,4u,1u,true,err);
            const fg_vk_tensor *last[]={c->roof_gdn,tensors[1],tensors[6],tensors[8]};
            return status==FG_OK?dispatch_impl(c,roof_kernel(c,54u,"fg_roof_recurrent_norm.spv",4u,16u),
                last,push,48u,1u,1u,true,err):status;
        }
        kernel=roof_kernel(c,6u,"fg_roof_recurrent.spv",9u,16u);
    }else if(kernel==&c->router_top10&&roof_enabled(c,8u)){
        kernel=roof_kernel(c,7u,"fg_roof_router.spv",3u,8u);
    }else if(kernel==&c->hc_inject_partial&&push&&((const uint32_t *)push)[1]==4u&&roof_enabled(c,16u)){
        kernel=roof_kernel(c,8u,"fg_roof_hc_inject.spv",3u,16u);
    }else if(kernel==&c->gr_partial&&push&&((const uint32_t *)push)[1]==4u&&roof_enabled(c,16u)){
        kernel=roof_kernel(c,13u,"fg_roof_hc_mix.spv",5u,16u);
    }else if(kernel==&c->moe_decode_gate_up&&roof_enabled(c,32u)){
        kernel=roof_kernel(c,9u,"fg_roof_expert.spv",5u,32u);
        const char *value=getenv("FG_ROOF_EXPERT");
        uint32_t v=value?(uint32_t)strtoul(value,NULL,0)&7u:0u;
        if(v>=1u&&v<=5u&&(v<3u||c->roof_wave32)){
            static const char *files[]={"fg_roof_expert_p2.spv","fg_roof_expert_p4.spv",
                "fg_roof_expert_w32.spv","fg_roof_expert_p2_w32.spv","fg_roof_expert_p4_w32.spv"};
            kernel=roof_kernel(c,24u+v,files[v-1u],5u,32u);
            if(v>=3u)gx*=2u;
        }else if(v==6u||v==7u){
            kernel=roof_kernel(c,41u+v-6u,v==6u?"fg_roof_expert_w128.spv":"fg_roof_expert_w256.spv",5u,32u);
            gx=(gx+(v==6u?1u:3u))/(v==6u?2u:4u);
        }
        uint32_t full=value?(uint32_t)strtoul(value,NULL,0):0u;
        if(gy==10u&&push&&((const uint32_t *)push)[1]==10u&&
           ((const uint32_t *)push)[6]==12u&&((const uint32_t *)push)[7]==12u&&(full&1984u)!=0u){
            if((full&1024u)!=0u&&c->roof_wave32){
                kernel=roof_kernel(c,58u,"fg_roof_expert_cache_w32.spv",5u,32u);
            }else if((full&512u)!=0u){
                kernel=roof_kernel(c,56u,"fg_roof_expert_slots2_p4.spv",5u,32u);gy=5u;
            }else if((full&256u)!=0u){
                kernel=roof_kernel(c,55u,"fg_roof_expert_cache.spv",5u,32u);
            }else if((full&192u)==192u){
                kernel=roof_kernel(c,57u,"fg_roof_expert_slots5.spv",5u,32u);gy=2u;
            }else if((full&128u)!=0u){
                kernel=roof_kernel(c,47u,"fg_roof_expert_slots10.spv",5u,32u);gy=1u;
            }else if((full&64u)!=0u){
                kernel=roof_kernel(c,46u,"fg_roof_expert_slots2.spv",5u,32u);gy=5u;
            }
            gx=(((const uint32_t *)push)[0]+7u)/8u;
        }
    }else if(kernel==&c->moe_decode_down_reduce&&roof_enabled(c,32u)){
        const char *value=getenv("FG_ROOF_EXPERT");
        uint32_t v=value?(uint32_t)strtoul(value,NULL,0):0u;
        if(push&&((const uint32_t *)push)[0]==2560u&&((const uint32_t *)push)[5]==10u&&
           ((const uint32_t *)push)[6]==7u&&(v&6144u)!=0u){
            fg_status status=FG_OK;
            if(!c->roof_expert_partials)status=fg_vk_tensor_create(c,10u*2560u*4u,&c->roof_expert_partials,err);
            if(status!=FG_OK)return status;
            const fg_vk_tensor *first[]={tensors[0],tensors[1],tensors[2],tensors[3],c->roof_expert_partials};
            bool wide=(v&4096u)!=0u;
            status=dispatch_impl(c,roof_kernel(c,wide?60u:59u,
                wide?"fg_roof_down_split_w64.spv":"fg_roof_down_split.spv",5u,28u),
                first,push,wide?80u:160u,10u,1u,true,err);
            const fg_vk_tensor *last[]={c->roof_expert_partials,tensors[4]};
            struct{uint32_t width,slots;}merge={2560u,10u};
            return status==FG_OK?dispatch_impl(c,roof_kernel(c,61u,"fg_roof_down_merge.spv",2u,8u),
                last,&merge,20u,1u,1u,true,err):status;
        }
        if(push&&((const uint32_t *)push)[6]==7u&&((const uint32_t *)push)[2]==20u&&
           ((const uint32_t *)push)[5]==10u&&(v&122880u)!=0u){
            bool four=(v&(16384u|65536u))!=0u,wave32=c->roof_wave32&&(v&98304u)!=0u;
            uint32_t slot=four?(wave32?78u:76u):(wave32?77u:75u);
            const char *file=four?(wave32?"fg_roof_down_slots4_w32.spv":"fg_roof_down_slots4.spv"):
                (wave32?"fg_roof_down_slots2_w32.spv":"fg_roof_down_slots2.spv");
            kernel=roof_kernel(c,slot,file,5u,28u);
        }else if(push&&((const uint32_t *)push)[6]==7u&&(v&48u)!=0u){
            if((v&48u)==16u){kernel=roof_kernel(c,43u,"fg_roof_down_w64.spv",5u,28u);gx=(gx+1u)/2u;}
            else if((v&48u)==32u){kernel=roof_kernel(c,44u,"fg_roof_down_rows1.spv",5u,28u);gx=(gx+3u)/4u;}
            else{kernel=roof_kernel(c,45u,"fg_roof_down_rows2.spv",5u,28u);gx=(gx+1u)/2u;}
        }else if(c->roof_wave32&&(v&8u)!=0u)
            kernel=roof_kernel(c,30u,"fg_roof_down_w32.spv",5u,28u);
    }else if((kernel==&c->dense_cooked||kernel==&c->dense_cooked_r8)&&push&&((const uint32_t *)push)[0]==10240u&&
             ((const uint32_t *)push)[1]==1u&&(((const uint32_t *)push)[2]==10u||
             ((const uint32_t *)push)[2]==20u)&&c->roof_supported){
        const char *value=getenv("FG_ROOF_HC");
        if(value&&((uint32_t)strtoul(value,NULL,0)&3u)==2u){
            kernel=roof_kernel(c,50u,"fg_roof_hc_q8_rows.spv",3u,20u);gx=160u;
        }
    }else if(kernel==&c->dense_f32&&push&&((const uint32_t *)push)[0]==2560u&&
             ((const uint32_t *)push)[1]==48u&&roof_enabled(c,128u)){
        kernel=roof_kernel(c,14u,"fg_roof_control.spv",3u,12u);
    }
    return dispatch_impl(c,kernel,tensors,push,gx,gy,gz,true,err);
}

uint32_t fg_vk_decode_fusion_mask(void){
    const char *value=getenv("FG_DECODE_FUSION");
    if(!value||!*value)return 0u;
    char *end=NULL;unsigned long mask=strtoul(value,&end,10);
    return end&&!*end&&mask<=7u?(uint32_t)mask:0u;
}

bool fg_vk_shared_swiglu_supported(const fg_vk_tensor *gate,const fg_vk_tensor *up){
    return gate&&up&&gate->format==up->format&&
        (gate->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
         gate->format==FG_VK_TENSOR_FORMAT_DENSE_SOA_Q4_K);
}

fg_status fg_vk_shared_swiglu(fg_vk_context *c,fg_vk_tensor *output,
    const fg_vk_tensor *gate,const fg_vk_tensor *up,const fg_vk_tensor *input,fg_error *err){
    if(!c||!fg_vk_shared_swiglu_supported(gate,up)||!tensor_on_context(c,gate)||
       !tensor_on_context(c,up)||!tensor_on_context(c,input)||!tensor_on_context(c,output)||
       !tensor_range(input,0,2560u*4u)||!tensor_range(output,0,640u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid shared SwiGLU fusion tensors");return FG_ERR_ARGUMENT;
    }
    bool q8=gate->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
    uint64_t bytes=q8?fg_q8_0_cooked_matrix_bytes(2560u,640u):fg_dense_soa_bytes(2560u,640u,12u);
    if(!tensor_range(gate,0,bytes)||!tensor_range(up,0,bytes)){
        fg_error_set(err,FG_ERR_ARGUMENT,"short shared SwiGLU fusion weights");return FG_ERR_ARGUMENT;
    }
    fg_vk_kernel *kernel=q8?&c->shared_swiglu_q8:&c->shared_swiglu_q4;
    if(!kernel->file)*kernel=(fg_vk_kernel){.file=q8?"fg_shared_swiglu_q8.spv":"fg_shared_swiglu_q4.spv",
        .bindings=4u,.push_bytes=q8?20u:24u};
    struct{uint32_t rows,tokens,blocks,tile_bytes;float scale;}q8_push={
        640u,1u,80u,(uint32_t)fg_q8_0_cooked_tile_bytes(2560u),1.0f};
    struct{uint32_t input,rows,tokens,type,row_bytes;float scale;}q4_push={
        2560u,640u,1u,12u,0u,1.0f};
    const fg_vk_tensor *t[]={gate,input,output,up};
    return dispatch(c,kernel,t,q8?(void *)&q8_push:(void *)&q4_push,q8?80u:160u,1u,1u,err);
}

fg_status fg_vk_router_top10_tiles(fg_vk_context *c,fg_vk_tensor *selected,
    fg_vk_tensor *gates,fg_vk_tensor *tiles,const fg_vk_tensor *logits,fg_error *err){
    if(!c||!tensor_on_context(c,selected)||!tensor_on_context(c,gates)||
       !tensor_on_context(c,tiles)||!tensor_on_context(c,logits)||
       !tensor_range(selected,0,40u)||!tensor_range(gates,0,40u)||
       !tensor_range(tiles,0,360u)||!tensor_range(logits,0,512u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused router tensors");return FG_ERR_ARGUMENT;
    }
    fg_vk_kernel *kernel=c->subgroup_topk?&c->router_top10_fast:&c->router_top10_tiles;
    if(!kernel->file)*kernel=(fg_vk_kernel){
        .file=c->subgroup_topk?"fg_router_top10_subgroup_tiles.spv":"fg_router_top10_tiles.spv",
        .bindings=4u,.push_bytes=8u};
    const fg_vk_tensor *t[]={logits,selected,gates,tiles};
    uint32_t push[]={512u,1u};
    return dispatch(c,kernel,t,push,1u,1u,1u,err);
}

fg_status fg_vk_moe_shared_gr_write(fg_vk_context *c,fg_vk_tensor *output,
    const fg_vk_tensor *hyper,const fg_vk_tensor *reduced,const fg_vk_tensor *shared,
    const fg_vk_tensor *scalar,const fg_vk_tensor *injection,fg_error *err){
    const fg_vk_tensor *t[]={hyper,reduced,shared,scalar,injection,output};
    const uint64_t sizes[]={40960u,10240u,10240u,4u,16u,40960u};
    for(uint32_t i=0;i<6u;i++)if(!tensor_on_context(c,t[i])||!tensor_range(t[i],0,sizes[i])){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused shared residual tensors");return FG_ERR_ARGUMENT;
    }
    if(!c->moe_shared_gr_write.file)c->moe_shared_gr_write=(fg_vk_kernel){
        .file="fg_moe_shared_gr_write.spv",.bindings=6u,.push_bytes=8u};
    uint32_t push[]={2560u,4u};
    return dispatch(c,&c->moe_shared_gr_write,t,push,10u,1u,1u,err);
}

static fg_status submit_group_send(fg_vk_context *c,VkFence fence,bool signal_flush,
    fg_error *err){
    VkCommandBuffer commands[FG_VK_GROUP_COMMANDS*2u];uint32_t count=0;
    for(uint32_t i=0;i<c->group_count;i++){
        if(i)commands[count++]=c->group_barrier;
        commands[count++]=c->group_commands[i];
    }
    VkResult vr=vkResetFences(c->device,1u,&fence);
    if(vr!=VK_SUCCESS)return vk_error(err,"reset submit-group fence",vr);
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount=count,.pCommandBuffers=commands};
    VkSemaphore wait,signal;VkPipelineStageFlags stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    vk_submit_attach(c,signal_flush,&wait,&signal,&stage,&submit);
    vr=vkQueueSubmit(c->queue,1u,&submit,fence);
    if(vr!=VK_SUCCESS)return vk_error(err,"submit command group",vr);
    c->counters.submissions++;
    vk_fence_track(c,fence);c->group_count=0;
    return FG_OK;
}

static fg_status submit_group_flush(fg_vk_context *c,fg_error *err){
    if(!c->group_count&&!c->group_published)return FG_OK;
    fg_status status=c->group_count?submit_group_send(c,c->fence,false,err):FG_OK;
    if(status==FG_OK)status=vk_pipeline_drain(c,err);
    if(status!=FG_OK)return status;
    for(uint32_t slot=0;slot<FG_VK_STATIC_SLOTS;slot++)
        if(c->static_pending[slot]&&!c->static_submit_fence[slot])
            c->static_pending[slot]=false;
    c->group_count=0;c->group_index=0;c->batch_set_count=0;c->group_published=false;
    c->command=c->group_buffers[0];
    return FG_OK;
}

static fg_status submit_group_append(fg_vk_context *c,VkCommandBuffer command,
    fg_error *err){
    if(c->group_count==FG_VK_GROUP_COMMANDS){
        fg_status status=submit_group_flush(c,err);
        if(status!=FG_OK)return status;
    }
    c->group_commands[c->group_count++]=command;
    return FG_OK;
}

fg_status fg_vk_submit_group_begin(fg_vk_context *c,fg_error *err){
    if(!c||c->group_active||c->batch_depth||c->profile_active||c->static_active){
        fg_error_set(err,FG_ERR_ARGUMENT,"submit group requires an idle unprofiled context");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_static_drain(c,err);
    if(status!=FG_OK)return status;
    if(!c->group_ready){
        VkCommandBufferAllocateInfo allocate={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount=FG_VK_GROUP_BUFFERS};
        VkResult vr=vkAllocateCommandBuffers(c->device,&allocate,c->group_buffers);
        if(vr!=VK_SUCCESS)return vk_error(err,"allocate submit-group commands",vr);
        allocate.commandBufferCount=1u;
        vr=vkAllocateCommandBuffers(c->device,&allocate,&c->group_barrier);
        if(vr!=VK_SUCCESS)return vk_error(err,"allocate submit-group barrier",vr);
        VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT};
        vr=vkBeginCommandBuffer(c->group_barrier,&begin);
        if(vr!=VK_SUCCESS)return vk_error(err,"begin submit-group barrier",vr);
        /* Submission order alone is not a memory dependency between primaries.
         * Preserve the visibility previously provided by the semaphore chain. */
        VkMemoryBarrier memory={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(c->group_barrier,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&memory,0,NULL,0,NULL);
        vr=vkEndCommandBuffer(c->group_barrier);
        if(vr!=VK_SUCCESS)return vk_error(err,"end submit-group barrier",vr);
        c->group_ready=true;
    }
    c->group_saved=c->command;c->command=c->group_buffers[0];
    c->group_count=0;c->group_index=0;c->batch_set_count=0;c->group_active=true;
    return FG_OK;
}

bool fg_vk_submit_group_active(const fg_vk_context *c){return c&&c->group_active;}

fg_status fg_vk_submit_group_publish(fg_vk_context *c,fg_error *err){
    if(!c||!c->group_active||c->batch_depth||c->group_published){
        fg_error_set(err,FG_ERR_ARGUMENT,"submit group cannot publish this recording");
        return FG_ERR_ARGUMENT;
    }
    if(!c->group_count)return FG_OK;
    fg_status status=submit_group_send(c,c->fence_alt,true,err);
    if(status==FG_OK)c->group_published=true;
    return status;
}

fg_status fg_vk_submit_group_end(fg_vk_context *c,fg_error *err){
    if(!c||!c->group_active||c->static_active){
        fg_error_set(err,FG_ERR_ARGUMENT,"submit group is not ready to end");
        return FG_ERR_ARGUMENT;
    }
    fg_status status=FG_OK;
    while(status==FG_OK&&c->batch_depth)status=fg_vk_end(c,err);
    if(status==FG_OK)status=submit_group_flush(c,err);
    if(status==FG_OK){c->command=c->group_saved;c->group_active=false;}
    return status;
}

void fg_vk_submit_group_abort(fg_vk_context *c){
    if(!c||!c->group_active)return;
    vkDeviceWaitIdle(c->device);
    if(c->batch_depth){fg_error ignored={0};fg_vk_abort(c,&ignored);}
    for(uint32_t slot=0;slot<FG_VK_STATIC_SLOTS;slot++)
        if(c->static_pending[slot]&&!c->static_submit_fence[slot])
            c->static_pending[slot]=false;
    c->command=c->group_saved;c->group_active=false;c->group_count=0;
    c->group_index=0;c->batch_set_count=0;c->pending_fence_count=0;c->group_published=false;
}

fg_status fg_vk_tensor_write_ordered(fg_vk_tensor *t,uint64_t offset,
    const void *data,uint64_t bytes,fg_error *err){
    if(!t||!t->context->group_active)return fg_vk_tensor_write(t,offset,data,bytes,err);
    fg_vk_context *c=t->context;
    if(!c->batch_depth&&!c->group_count&&!c->group_published)
        return fg_vk_tensor_write(t,offset,data,bytes,err);
    if(!data||!tensor_range(t,offset,bytes)||!c->batch_depth||
       (((t->offset+offset)|bytes)&3u)!=0u||bytes>65536u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid ordered Vulkan write");
        return FG_ERR_ARGUMENT;
    }
    if(!bytes)return FG_OK;
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&before,0,NULL,0,NULL);
    vkCmdUpdateBuffer(c->command,t->allocation->buffer,t->offset+offset,bytes,data);
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|
            VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_HOST_BIT,
        0,1,&after,0,NULL,0,NULL);
    return FG_OK;
}

fg_status fg_vk_begin(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    if(c->batch_depth){c->batch_depth++;return FG_OK;}
    fg_status pending=vk_fence_wait(c,c->fence,err);if(pending!=FG_OK)return pending;
    if(c->group_active&&c->batch_set_count>=FG_VK_STATIC_SET_BASE){
        pending=submit_group_flush(c,err);if(pending!=FG_OK)return pending;
    }
    vk_profile_begin_record(c);
    VkResult vr;if(!c->group_active)vkResetFences(c->device,1,&c->fence);
    vkResetCommandBuffer(c->command,0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin batch command",vr);
    profile_command_begin(c);
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);
    /* The descriptor set epoch must not restart here: a previous flush
     * submission may still be queued, and it reads its bindings at execution
     * time. The epoch only resets in fg_vk_end after every pending submission
     * has been drained. */
    if(c->batch_set_count>=FG_VK_BATCH_MAX_SETS/2u){
        fg_status drain=vk_pipeline_drain(c,err);
        if(drain!=FG_OK)return drain;
        c->batch_set_count=0;
    }
    c->batch_set_base=0;
    c->batch_depth=1;c->batch_has_dispatch=false;return FG_OK;
}

fg_status fg_vk_end(fg_vk_context *c,fg_error *err){
    if(!c||!c->batch_depth){fg_error_set(err,FG_ERR_ARGUMENT,"not in batch");return FG_ERR_ARGUMENT;}
    if(c->batch_depth>1){c->batch_depth--;return FG_OK;}
    if(c->batch_set_base){fg_error_set(err,FG_ERR_MISMATCH,"static run ended by the dynamic batch path");return FG_ERR_MISMATCH;}
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    profile_command_end(c);
    VkResult vr;if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS){fg_status status=vk_error(err,"end batch command",vr);fg_error ignored={0};fg_vk_abort(c,&ignored);return status;}
    vk_profile_end_record(c);
    if(c->group_active){
        c->batch_depth=0;c->batch_has_dispatch=false;
        fg_status status=submit_group_append(c,c->command,err);
        return status==FG_OK?submit_group_flush(c,err):status;
    }
    VkCommandBuffer commands[FG_VK_STATIC_SLOTS+1u];uint32_t command_count=0u;
    for(uint32_t i=0u;i<c->held_count;i++)commands[command_count++]=c->held_commands[i];
    commands[command_count++]=c->command;
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=command_count,.pCommandBuffers=commands};
    VkSemaphore sem_wait,sem_signal;VkPipelineStageFlags sem_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;vk_submit_attach(c,false,&sem_wait,&sem_signal,&sem_stage,&submit);
    double submit_begin=c->profile_active?vk_wall_seconds():0.0;if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS){fg_status status=vk_error(err,"submit batch command",vr);fg_error ignored={0};fg_vk_abort(c,&ignored);return status;}if(c->profile_active)c->profile.submit_ms+=(vk_wall_seconds()-submit_begin)*1000.0;c->counters.submissions++;
    for(uint32_t i=0u;i<c->held_count;i++)c->static_submit_fence[c->held_slots[i]]=c->fence;
    c->held_count=0u;
    vk_fence_track(c,c->fence);
    fg_status wait_status=vk_fence_wait(c,c->fence,err);
    if(wait_status!=FG_OK){vkDeviceWaitIdle(c->device);c->batch_depth=0;c->batch_set_count=0;c->batch_has_dispatch=false;return wait_status;}
    /* Descriptor set indices restart at zero for the next batch, and a queued
     * submission reads its bindings at execution time: drain every pending
     * submission before those sets may be rewritten. */
    fg_status drain_status=vk_pipeline_drain(c,err);
    if(drain_status!=FG_OK){vkDeviceWaitIdle(c->device);c->pending_fence_count=0;c->batch_depth=0;c->batch_set_count=0;c->batch_has_dispatch=false;return drain_status;}
    fg_status status=profile_resolve(c,err);
    c->batch_depth=0;c->batch_set_count=0;c->batch_has_dispatch=false;return status;
}
fg_status fg_vk_flush(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    if(!c->batch_depth){fg_error_set(err,FG_ERR_ARGUMENT,"not in batch");return FG_ERR_ARGUMENT;}
    if(c->batch_depth>1){c->batch_depth--;return FG_OK;}
    if(c->batch_set_base){fg_error_set(err,FG_ERR_MISMATCH,"static run flushed by the dynamic batch path");return FG_ERR_MISMATCH;}
    if(!c->pipeline_ready||c->profile_active)return fg_vk_end(c,err);
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    VkResult vr;if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS){fg_status status=vk_error(err,"end flush command",vr);fg_error ignored={0};fg_vk_abort(c,&ignored);return status;}
    if(c->group_active){
        c->batch_depth=0;c->batch_has_dispatch=false;
        fg_status status=submit_group_append(c,c->command,err);
        if(status!=FG_OK)return status;
        if(++c->group_index==FG_VK_GROUP_BUFFERS)return submit_group_flush(c,err);
        c->command=c->group_buffers[c->group_index];
        return FG_OK;
    }
    VkCommandBuffer commands[FG_VK_STATIC_SLOTS+1u];uint32_t command_count=0u;
    for(uint32_t i=0u;i<c->held_count;i++)commands[command_count++]=c->held_commands[i];
    commands[command_count++]=c->command;
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=command_count,.pCommandBuffers=commands};
    VkSemaphore sem_wait,sem_signal;VkPipelineStageFlags sem_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;vk_submit_attach(c,true,&sem_wait,&sem_signal,&sem_stage,&submit);
    double submit_begin=c->profile_active?vk_wall_seconds():0.0;if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS){fg_status status=vk_error(err,"submit flush command",vr);fg_error ignored={0};fg_vk_abort(c,&ignored);return status;}if(c->profile_active)c->profile.submit_ms+=(vk_wall_seconds()-submit_begin)*1000.0;c->counters.submissions++;
    for(uint32_t i=0u;i<c->held_count;i++)c->static_submit_fence[c->held_slots[i]]=c->fence;
    c->held_count=0u;
    vk_fence_track(c,c->fence);
    VkCommandBuffer command=c->command;c->command=c->command_alt;c->command_alt=command;
    VkFence fence=c->fence;c->fence=c->fence_alt;c->fence_alt=fence;
    /* Descriptor sets stay monotonic across flushes so a pending submission
     * never shares a set with the next recording. */
    c->batch_depth=0;c->batch_has_dispatch=false;return FG_OK;
}
fg_status fg_vk_abort(fg_vk_context *c,fg_error *err){
    if(!c||!c->batch_depth){fg_error_set(err,FG_ERR_ARGUMENT,"not in batch");return FG_ERR_ARGUMENT;}
    if(c->held_count){fg_error ignored={0};fg_status held_status=vk_flush_held(c,&ignored);(void)held_status;}
    if(c->static_active){c->command=c->command_swap;c->static_active=false;c->batch_set_count=c->static_saved_set_count;c->batch_set_base=0;}
    VkResult vr=vkResetCommandBuffer(c->command,0);
    c->batch_depth=0;c->batch_has_dispatch=false;
    return vr==VK_SUCCESS?FG_OK:vk_error(err,"abort batch command",vr);
}
bool fg_vk_batch_active(const fg_vk_context *c){return c&&c->batch_depth>0;}
void fg_vk_next_dispatch_independent(fg_vk_context *c){if(c)c->barrier_elide_next=true;}
void fg_vk_host_write_visible(fg_vk_context *c){
    if(!c||!c->batch_depth)return;
    VkMemoryBarrier host={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&host,0,NULL,0,NULL);
}
void fg_vk_device_write_visible(fg_vk_context *c){
    if(!c||!c->batch_depth)return;
    VkMemoryBarrier dependency={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&dependency,0,NULL,0,NULL);
}
bool fg_vk_static_recorded(const fg_vk_context *c,uint32_t slot){
    return c&&slot<FG_VK_STATIC_SLOTS&&c->static_recorded[slot];
}
static fg_status static_slot_ready(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(c->static_ready[slot])return FG_OK;
    VkCommandBufferAllocateInfo allocate={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkResult vr=vkAllocateCommandBuffers(c->device,&allocate,&c->static_command[slot]);
    if(vr!=VK_SUCCESS){c->static_command[slot]=VK_NULL_HANDLE;return vk_error(err,"allocate static command buffer",vr);}
    VkFenceCreateInfo fence={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if((vr=vkCreateFence(c->device,&fence,NULL,&c->static_fence[slot]))!=VK_SUCCESS)
        return vk_error(err,"create static fence",vr);
    c->static_ready[slot]=true;
    return FG_OK;
}
/* Record a token-invariant run into a private command buffer.  The buffer is
 * submitted with the same semaphore chain as the dynamic batches, so it can be
 * replayed for later tokens without any host recording on their critical path. */
fg_status fg_vk_static_claim(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot<FG_VK_OWNER_STATIC_SLOTS||slot>=FG_VK_STATIC_SLOTS){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid private replay slot");return FG_ERR_ARGUMENT;
    }
    if(c->static_claimed[slot]){
        fg_error_set(err,FG_ERR_MISMATCH,"private replay slot is already owned");return FG_ERR_MISMATCH;
    }
    c->static_claimed[slot]=true;return FG_OK;
}
fg_status fg_vk_static_release(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot<FG_VK_OWNER_STATIC_SLOTS||slot>=FG_VK_STATIC_SLOTS||!c->static_claimed[slot]){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid private replay release");return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_vk_static_wait(c,slot,err);
    if(status==FG_OK)status=fg_vk_static_reset(c,slot,err);
    if(status==FG_OK)c->static_claimed[slot]=false;
    return status;
}
fg_status fg_vk_static_begin(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot>=FG_VK_STATIC_SLOTS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid static batch slot");return FG_ERR_ARGUMENT;}
    if(c->batch_depth||c->static_active){fg_error_set(err,FG_ERR_ARGUMENT,"static batch requires an idle recording");return FG_ERR_ARGUMENT;}
    if(c->static_recorded[slot]){fg_error_set(err,FG_ERR_MISMATCH,"static batch slot is already recorded");return FG_ERR_MISMATCH;}
    if(c->static_pending[slot]){fg_error_set(err,FG_ERR_MISMATCH,"static batch slot is still in flight");return FG_ERR_MISMATCH;}
    fg_status status=static_slot_ready(c,slot,err);if(status!=FG_OK)return status;
    VkResult vr=vkResetFences(c->device,1,&c->static_fence[slot]);
    if(vr!=VK_SUCCESS)return vk_error(err,"reset static fence",vr);
    vkResetCommandBuffer(c->static_command[slot],0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if((vr=vkBeginCommandBuffer(c->static_command[slot],&begin))!=VK_SUCCESS)
        return vk_error(err,"begin static batch",vr);
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->static_command[slot],VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);
    c->command_swap=c->command;c->command=c->static_command[slot];
    c->static_active=true;c->static_slot_active=slot;
    c->static_saved_set_count=c->batch_set_count;
    c->batch_set_base=FG_VK_STATIC_SET_BASE+slot*FG_VK_STATIC_SET_STRIDE;
    c->batch_set_count=0;c->batch_depth=1;c->batch_has_dispatch=false;
    return FG_OK;
}
fg_status fg_vk_static_end(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot>=FG_VK_STATIC_SLOTS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid static batch slot");return FG_ERR_ARGUMENT;}
    if(!c->static_active||c->static_slot_active!=slot){fg_error_set(err,FG_ERR_MISMATCH,"static batch slot is not recording");return FG_ERR_MISMATCH;}
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    VkResult vr=vkEndCommandBuffer(c->command);
    c->command=c->command_swap;c->static_active=false;
    c->batch_depth=0;c->batch_has_dispatch=false;
    if(vr!=VK_SUCCESS)return vk_error(err,"end static batch",vr);
    c->batch_set_count=c->static_saved_set_count;
    c->batch_set_base=0;
    c->static_recorded[slot]=true;
    return FG_OK;
}
fg_status fg_vk_static_reset(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot>=FG_VK_STATIC_SLOTS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid static batch slot");return FG_ERR_ARGUMENT;}
    if(c->static_active){fg_error_set(err,FG_ERR_MISMATCH,"static batch reset during recording");return FG_ERR_MISMATCH;}
    if(c->static_pending[slot]){fg_error_set(err,FG_ERR_MISMATCH,"static batch slot is still in flight");return FG_ERR_MISMATCH;}
    c->static_recorded[slot]=false;
    return FG_OK;
}
static fg_status vk_flush_held(fg_vk_context *c,fg_error *err){
    if(!c->held_count)return FG_OK;
    VkResult vr;
    if(c->held_fence_pending){
        vr=vkWaitForFences(c->device,1,&c->held_fence,VK_TRUE,UINT64_MAX);
        if(vr!=VK_SUCCESS)return vk_error(err,"wait held static batches",vr);
        c->held_fence_pending=false;
    }
    vr=vkResetFences(c->device,1,&c->held_fence);
    if(vr!=VK_SUCCESS)return vk_error(err,"reset held-batch fence",vr);
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=c->held_count,.pCommandBuffers=c->held_commands};
    VkSemaphore sem_wait,sem_signal;VkPipelineStageFlags sem_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    vk_submit_attach(c,true,&sem_wait,&sem_signal,&sem_stage,&submit);
    if((vr=vkQueueSubmit(c->queue,1,&submit,c->held_fence))!=VK_SUCCESS)
        return vk_error(err,"submit held static batches",vr);
    c->counters.submissions++;
    for(uint32_t i=0u;i<c->held_count;i++)c->static_submit_fence[c->held_slots[i]]=c->held_fence;
    c->held_fence_pending=true;c->held_count=0u;
    return FG_OK;
}
fg_status fg_vk_static_submit(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot>=FG_VK_STATIC_SLOTS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid static batch slot");return FG_ERR_ARGUMENT;}
    if(!c->static_recorded[slot]){fg_error_set(err,FG_ERR_MISMATCH,"static batch slot is not recorded");return FG_ERR_MISMATCH;}
    if(c->static_pending[slot]){fg_error_set(err,FG_ERR_MISMATCH,"static batch slot is already pending");return FG_ERR_MISMATCH;}
    if(c->batch_depth){fg_error_set(err,FG_ERR_MISMATCH,"static batch cannot submit inside a dynamic batch");return FG_ERR_MISMATCH;}
    if(c->group_active){
        fg_status status=submit_group_append(c,c->static_command[slot],err);
        if(status==FG_OK){c->static_pending[slot]=true;c->static_submit_fence[slot]=VK_NULL_HANDLE;}
        return status;
    }
    VkResult vr=vkResetFences(c->device,1,&c->static_fence[slot]);
    if(vr!=VK_SUCCESS)return vk_error(err,"reset static fence",vr);
    c->static_pending[slot]=true;
    if(c->pending_fence_count>0u){
        c->held_commands[c->held_count]=c->static_command[slot];
        c->held_slots[c->held_count]=slot;c->held_count++;
        c->static_submit_fence[slot]=VK_NULL_HANDLE;
        return FG_OK;
    }
    VkCommandBuffer command=c->static_command[slot];
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&command};
    VkSemaphore sem_wait,sem_signal;VkPipelineStageFlags sem_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    vk_submit_attach(c,true,&sem_wait,&sem_signal,&sem_stage,&submit);
    if((vr=vkQueueSubmit(c->queue,1,&submit,c->static_fence[slot]))!=VK_SUCCESS)
        return vk_error(err,"submit static batch",vr);
    c->counters.submissions++;
    c->static_submit_fence[slot]=c->static_fence[slot];
    return FG_OK;
}
fg_status fg_vk_static_copy_run(fg_vk_context *c,uint32_t slot,
    fg_vk_tensor *const *dst,const fg_vk_tensor *const *src,uint32_t count,fg_error *err){
    if(!c||slot>=FG_VK_STATIC_SLOTS||!count||c->batch_depth||
       !c->static_recorded[slot]||c->static_pending[slot]){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid copied static execution");return FG_ERR_ARGUMENT;
    }
    fg_status status=record_tensor_copies(c,dst,src,count,err);
    if(status!=FG_OK)return status;
    VkResult vr=vkResetFences(c->device,1,&c->static_fence[slot]);
    if(vr!=VK_SUCCESS)return vk_error(err,"reset copied static fence",vr);
    VkCommandBuffer commands[]={c->copy_command,c->static_command[slot]};
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=2u,.pCommandBuffers=commands};
    VkSemaphore wait,signal;VkPipelineStageFlags stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    vk_submit_attach(c,true,&wait,&signal,&stage,&submit);
    if((vr=vkQueueSubmit(c->queue,1,&submit,c->static_fence[slot]))!=VK_SUCCESS)
        return vk_error(err,"submit copied static graph",vr);
    c->counters.submissions++;c->static_pending[slot]=true;
    c->static_submit_fence[slot]=c->static_fence[slot];
    return fg_vk_static_wait(c,slot,err);
}
fg_status fg_vk_static_wait(fg_vk_context *c,uint32_t slot,fg_error *err){
    if(!c||slot>=FG_VK_STATIC_SLOTS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid static batch slot");return FG_ERR_ARGUMENT;}
    if(!c->static_pending[slot])return FG_OK;
    if(c->group_active){
        if(c->batch_depth){fg_error_set(err,FG_ERR_MISMATCH,"static wait inside submit-group recording");return FG_ERR_MISMATCH;}
        return submit_group_flush(c,err);
    }
    fg_status status=vk_flush_held(c,err);
    if(status!=FG_OK)return status;
    VkResult vr=vkWaitForFences(c->device,1,&c->static_submit_fence[slot],VK_TRUE,UINT64_MAX);
    if(vr!=VK_SUCCESS)return vk_error(err,"wait static batch",vr);
    c->static_pending[slot]=false;
    return FG_OK;
}
fg_status fg_vk_static_drain(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    for(uint32_t slot=0u;slot<FG_VK_STATIC_SLOTS;slot++){
        fg_status status=fg_vk_static_wait(c,slot,err);
        if(status!=FG_OK)return status;
    }
    return vk_pipeline_drain(c,err);
}

static fg_status expert_graph_dispatch(fg_vk_expert_graph *graph,fg_vk_kernel *kernel,
                                       const fg_vk_tensor *const *tensors,const void *push,
                                       uint32_t gx,uint32_t gy,uint32_t gz,bool barrier,
                                       fg_error *err){
    fg_vk_context *c=graph->context;fg_status status=create_kernel(c,kernel,err);if(status!=FG_OK)return status;
    VkDescriptorSetAllocateInfo allocate={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=graph->descriptor_pool,.descriptorSetCount=1,.pSetLayouts=&kernel->set_layout};VkDescriptorSet set;VkResult vr=vkAllocateDescriptorSets(c->device,&allocate,&set);if(vr!=VK_SUCCESS)return vk_error(err,"allocate expert graph descriptor set",vr);
    VkDescriptorBufferInfo info[16];VkWriteDescriptorSet write[16];for(uint32_t i=0;i<kernel->bindings;i++){info[i]=(VkDescriptorBufferInfo){.buffer=tensors[i]->allocation->buffer,.offset=tensors[i]->offset,.range=tensors[i]->bytes};write[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info[i]};}vkUpdateDescriptorSets(c->device,kernel->bindings,write,0,NULL);
    if(barrier){VkMemoryBarrier memory={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(graph->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&memory,0,NULL,0,NULL);}
    vkCmdBindPipeline(graph->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(graph->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(graph->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);vkCmdDispatch(graph->command,gx,gy,gz);graph->dispatches++;return FG_OK;
}

static bool tensor_ranges_overlap(const fg_vk_tensor *a,const fg_vk_tensor *b){
    if(!a||!b||a->allocation!=b->allocation)return false;
    return a->offset<b->offset+b->bytes&&b->offset<a->offset+a->bytes;
}

static fg_status validate_expert_graph_scratch(fg_vk_tensor *const scratch[8],
                                                fg_error *err){
    for(uint32_t i=0u;i<8u;i++)for(uint32_t j=i+1u;j<8u;j++)
        if(tensor_ranges_overlap(scratch[i],scratch[j])){
            fg_error_set(err,FG_ERR_MISMATCH,
                         "expert graph scratch ranges %u and %u overlap",i,j);
            return FG_ERR_MISMATCH;
        }
    return FG_OK;
}

void fg_vk_expert_graph_destroy(fg_vk_expert_graph *graph){if(!graph)return;fg_vk_context *c=graph->context;if(c&&graph->command)vkFreeCommandBuffers(c->device,c->command_pool,1,&graph->command);if(c&&graph->descriptor_pool)vkDestroyDescriptorPool(c->device,graph->descriptor_pool,NULL);free(graph);}

fg_status fg_vk_expert_graph_create(fg_vk_context *c,fg_vk_expert_graph **out,fg_vk_tensor *activation,fg_vk_tensor *tiles,fg_vk_tensor *gates,fg_vk_tensor *gate,fg_vk_tensor *up,fg_vk_tensor *mid,fg_vk_tensor *down,fg_vk_tensor *reduced,const fg_vk_tensor *gate_weights,const fg_vk_tensor *up_weights,const fg_vk_tensor *down_weights,uint32_t gate_type,uint32_t up_type,uint32_t down_type,uint32_t hidden_width,uint32_t mid_width,uint32_t gate_expert_stride,uint32_t up_expert_stride,uint32_t down_expert_stride,uint32_t weight_experts,uint32_t slots,fg_error *err){
    if(out)*out=NULL;
    uint32_t gate_block=gate_type==12u?144u:gate_type==13u?176u:0u,up_block=up_type==12u?144u:up_type==13u?176u:0u,down_block=down_type==7u?24u:down_type==8u?34u:0u;uint64_t gate_row=(uint64_t)(hidden_width/256u)*gate_block,up_row=(uint64_t)(hidden_width/256u)*up_block,down_row=(uint64_t)(mid_width/32u)*down_block,q8_bytes=(uint64_t)(hidden_width/256u)*296u;
    bool gate_k=gate_weights&&gate_weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED,up_k=up_weights&&up_weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED;
    bool gate_q8=gate_weights&&gate_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED,up_q8=up_weights&&up_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED;
    bool down_q51=down_weights&&down_weights->format==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED,down_q8=down_weights&&down_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED;
    bool gate_cooked=gate_k||gate_q8,up_cooked=up_k||up_q8;
    uint64_t gate_matrix=gate_q8?fg_q8_0_cooked_matrix_bytes(hidden_width,mid_width):gate_k?fg_k_quant_cooked_matrix_bytes(hidden_width,mid_width,gate_type):gate_row*mid_width,up_matrix=up_q8?fg_q8_0_cooked_matrix_bytes(hidden_width,mid_width):up_k?fg_k_quant_cooked_matrix_bytes(hidden_width,mid_width,up_type):up_row*mid_width,down_matrix=down_q51?fg_q5_1_cooked_matrix_bytes(mid_width,hidden_width):down_q8?fg_q8_0_cooked_matrix_bytes(mid_width,hidden_width):down_row*hidden_width;
    if(!out||!c||c->batch_depth||!gate_matrix||!up_matrix||!down_matrix||!hidden_width||hidden_width%256u||!mid_width||mid_width%32u||!weight_experts||!slots||gate_row>UINT32_MAX||up_row>UINT32_MAX||down_row>UINT32_MAX||gate_matrix>UINT32_MAX||up_matrix>UINT32_MAX||down_matrix>UINT32_MAX||(gate_type==8u&&!gate_q8)||(up_type==8u&&!up_q8)||gate_expert_stride<gate_matrix||up_expert_stride<up_matrix||down_expert_stride<down_matrix||!tensor_on_context(c,activation)||!tensor_on_context(c,tiles)||!tensor_on_context(c,gates)||!tensor_on_context(c,gate)||!tensor_on_context(c,up)||!tensor_on_context(c,mid)||!tensor_on_context(c,down)||!tensor_on_context(c,reduced)||!tensor_on_context(c,gate_weights)||!tensor_on_context(c,up_weights)||!tensor_on_context(c,down_weights)||!tensor_range(activation,0,q8_bytes)||!tensor_range(tiles,0,(uint64_t)slots*9u*4u)||!tensor_range(gates,0,(uint64_t)slots*4u)||!tensor_range(gate,0,(uint64_t)slots*mid_width*4u)||!tensor_range(up,0,(uint64_t)slots*mid_width*4u)||!tensor_range(mid,0,(uint64_t)slots*mid_width*4u)||!tensor_range(down,0,(uint64_t)slots*hidden_width*4u)||!tensor_range(reduced,0,(uint64_t)hidden_width*4u)||!tensor_range(gate_weights,0,(uint64_t)weight_experts*gate_expert_stride)||!tensor_range(up_weights,0,(uint64_t)weight_experts*up_expert_stride)||!tensor_range(down_weights,0,(uint64_t)weight_experts*down_expert_stride)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fixed expert graph");return FG_ERR_ARGUMENT;}
    fg_vk_tensor *scratch[]={activation,tiles,gates,gate,up,mid,down,reduced};
    fg_status scratch_status=validate_expert_graph_scratch(scratch,err);
    if(scratch_status!=FG_OK)return scratch_status;
    fg_vk_expert_graph *graph=calloc(1,sizeof(*graph));if(!graph){fg_error_set(err,FG_ERR_OOM,"allocate fixed expert graph");return FG_ERR_OOM;}graph->context=c;
    VkDescriptorPoolSize pool_size={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=80u};VkDescriptorPoolCreateInfo pool_create={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=5u,.poolSizeCount=1u,.pPoolSizes=&pool_size};VkResult vr=vkCreateDescriptorPool(c->device,&pool_create,NULL,&graph->descriptor_pool);if(vr!=VK_SUCCESS){fg_vk_expert_graph_destroy(graph);return vk_error(err,"create expert graph descriptor pool",vr);}
    VkCommandBufferAllocateInfo command_allocate={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1u};if((vr=vkAllocateCommandBuffers(c->device,&command_allocate,&graph->command))!=VK_SUCCESS){fg_vk_expert_graph_destroy(graph);return vk_error(err,"allocate expert graph command",vr);}VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};if((vr=vkBeginCommandBuffer(graph->command,&begin))!=VK_SUCCESS){fg_vk_expert_graph_destroy(graph);return vk_error(err,"begin expert graph command",vr);}
    VkMemoryBarrier host={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(graph->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&host,0,NULL,0,NULL);
    struct{uint32_t out_dim,blocks,row_bytes,expert_stride,type,n_used,routed_pairs,packed_weights,reserved;}gate_push={mid_width,hidden_width/256u,(uint32_t)gate_row,gate_expert_stride,gate_type,slots,slots,0u,0u},up_push={mid_width,hidden_width/256u,(uint32_t)up_row,up_expert_stride,up_type,slots,slots,0u,0u};struct{uint32_t out_dim,blocks,tile_bytes,expert_stride,type,n_used,routed_pairs,packed_weights;}gate_cooked_push={mid_width,hidden_width/FG_QK8_K,(uint32_t)(gate_q8?fg_q8_0_cooked_tile_bytes(hidden_width):fg_k_quant_cooked_tile_bytes(hidden_width,gate_type)),gate_expert_stride,gate_type,slots,slots,0u},up_cooked_push={mid_width,hidden_width/FG_QK8_K,(uint32_t)(up_q8?fg_q8_0_cooked_tile_bytes(hidden_width):fg_k_quant_cooked_tile_bytes(hidden_width,up_type)),up_expert_stride,up_type,slots,slots,0u};struct{uint32_t values;}swiglu_push={slots*mid_width};struct{uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;}down_push={hidden_width,mid_width,(uint32_t)down_row,down_expert_stride,slots,0u,0u};struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,routed_pairs,packed_weights;}down_cooked_push={hidden_width,mid_width,mid_width/FG_QK8_0,(uint32_t)(down_q8?fg_q8_0_cooked_tile_bytes(mid_width):fg_q5_1_cooked_tile_bytes(mid_width)),down_expert_stride,slots,0u};struct{uint32_t width,selected_count,slot_count;}reduce_push={hidden_width,slots,slots};
    bool fused=fg_vk_decode_experts_fusable(gate_weights,up_weights,down_weights,down_type);
    fg_status status=FG_OK;
    if(fused){
        struct{uint32_t out_dim,blocks,gate_tile_bytes,up_tile_bytes,gate_stride,up_stride,gate_type,up_type;}gate_up_push={mid_width,hidden_width/FG_QK8_K,(uint32_t)(gate_q8?fg_q8_0_cooked_tile_bytes(hidden_width):fg_k_quant_cooked_tile_bytes(hidden_width,gate_type)),(uint32_t)(up_q8?fg_q8_0_cooked_tile_bytes(hidden_width):fg_k_quant_cooked_tile_bytes(hidden_width,up_type)),gate_expert_stride,up_expert_stride,gate_type,up_type};
        struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,slots,down_type;}down_reduce_push={hidden_width,mid_width,mid_width/FG_QK8_0,down_q51?(uint32_t)fg_q5_1_cooked_tile_bytes(mid_width):down_q8?(uint32_t)fg_q8_0_cooked_tile_bytes(mid_width):0u,down_expert_stride,slots,down_type};
        const fg_vk_tensor *gate_up_bindings[]={gate_weights,up_weights,activation,tiles,mid},*down_reduce_bindings[]={down_weights,tiles,mid,gates,reduced};
        status=expert_graph_dispatch(graph,&c->moe_decode_gate_up,gate_up_bindings,&gate_up_push,(mid_width+7u)/8u,slots,1u,false,err);
        bool wide_q8=down_q8&&c->subgroup_expert_q8;
        uint32_t down_groups=wide_q8?(hidden_width+1u)/2u:down_q8?(hidden_width+7u)/8u:(hidden_width+15u)/16u;
        if(status==FG_OK)status=expert_graph_dispatch(graph,wide_q8?&c->moe_decode_down_q8:&c->moe_decode_down_reduce,down_reduce_bindings,&down_reduce_push,down_groups,1u,1u,true,err);
    }else{
        const fg_vk_tensor *gate_bindings[]={gate_weights,activation,tiles,gate},*up_bindings[]={up_weights,activation,tiles,up},*swiglu_bindings[]={gate,up,mid},*down_bindings[]={down_weights,tiles,mid,down},*reduce_bindings[]={down,gates,tiles,reduced};fg_vk_kernel *gate_kernel=gate_cooked?&c->kquant_cooked:&c->kquant,*up_kernel=up_cooked?&c->kquant_cooked:&c->kquant,*down_kernel=down_q51?&c->moe_q5_1_cooked:down_q8?&c->moe_q8_0_cooked:down_type==7u?&c->moe_q5_1:&c->moe_q8_0;const void *gate_parameters=gate_cooked?(const void *)&gate_cooked_push:(const void *)&gate_push,*up_parameters=up_cooked?(const void *)&up_cooked_push:(const void *)&up_push,*down_parameters=(down_q51||down_q8)?(const void *)&down_cooked_push:(const void *)&down_push;status=expert_graph_dispatch(graph,gate_kernel,gate_bindings,gate_parameters,(mid_width+7u)/8u,slots,1u,false,err);if(status==FG_OK)status=expert_graph_dispatch(graph,up_kernel,up_bindings,up_parameters,(mid_width+7u)/8u,slots,1u,false,err);if(status==FG_OK)status=expert_graph_dispatch(graph,&c->swiglu,swiglu_bindings,&swiglu_push,(slots*mid_width+255u)/256u,1u,1u,true,err);if(status==FG_OK)status=expert_graph_dispatch(graph,down_kernel,down_bindings,down_parameters,(hidden_width+7u)/8u,slots,1u,true,err);if(status==FG_OK)status=expert_graph_dispatch(graph,&c->moe_reduce,reduce_bindings,&reduce_push,(hidden_width+255u)/256u,1u,1u,true,err);
    }
    if(status==FG_OK){VkMemoryBarrier finish={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(graph->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&finish,0,NULL,0,NULL);if((vr=vkEndCommandBuffer(graph->command))!=VK_SUCCESS)status=vk_error(err,"end expert graph command",vr);}if(status!=FG_OK){fg_vk_expert_graph_destroy(graph);return status;}*out=graph;return FG_OK;
}

fg_status fg_vk_expert_graph_execute(fg_vk_expert_graph *graph,fg_error *err){if(!graph||!graph->context||graph->context->batch_depth||graph->context->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"fixed expert graph unavailable while batching or profiling");return FG_ERR_ARGUMENT;}fg_vk_context *c=graph->context;fg_status drain=vk_pipeline_drain(c,err);if(drain!=FG_OK)return drain;VkResult vr=vkResetFences(c->device,1,&c->fence);if(vr!=VK_SUCCESS)return vk_error(err,"reset expert graph fence",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1u,.pCommandBuffers=&graph->command};if((vr=vkQueueSubmit(c->queue,1u,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit expert graph",vr);c->counters.submissions++;c->counters.dispatches+=graph->dispatches;if((vr=vkWaitForFences(c->device,1u,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait for expert graph",vr);return FG_OK;}

/* Fire-and-forget variant used by the decode block fast path: the expert
 * graph is submitted on its own fence so the caller can enqueue the shared
 * expert batch behind it and collect the result at the next sync. */
fg_status fg_vk_expert_graph_submit(fg_vk_expert_graph *graph,fg_error *err){if(!graph||!graph->context||graph->context->batch_depth||graph->context->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"fixed expert graph unavailable while batching or profiling");return FG_ERR_ARGUMENT;}fg_vk_context *c=graph->context;if(c->expert_fence_pending){fg_error_set(err,FG_ERR_MISMATCH,"expert graph submitted before collecting the previous one");return FG_ERR_MISMATCH;}VkResult vr=vkResetFences(c->device,1,&c->expert_fence);if(vr!=VK_SUCCESS)return vk_error(err,"reset async expert graph fence",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1u,.pCommandBuffers=&graph->command};VkSemaphore sem_wait,sem_signal;VkPipelineStageFlags sem_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;vk_submit_attach(c,false,&sem_wait,&sem_signal,&sem_stage,&submit);if((vr=vkQueueSubmit(c->queue,1u,&submit,c->expert_fence))!=VK_SUCCESS)return vk_error(err,"submit async expert graph",vr);c->counters.submissions++;c->counters.dispatches+=graph->dispatches;c->expert_fence_pending=true;return FG_OK;}
fg_status fg_vk_expert_graph_wait(fg_vk_context *c,fg_error *err){if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"expert graph wait has no context");return FG_ERR_ARGUMENT;}if(!c->expert_fence_pending)return FG_OK;VkResult vr=vkWaitForFences(c->device,1u,&c->expert_fence,VK_TRUE,UINT64_MAX);if(vr!=VK_SUCCESS)return vk_error(err,"wait for async expert graph",vr);c->expert_fence_pending=false;return FG_OK;}

fg_status fg_vk_quantize_q8_k(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){if(!c||!width||width%256u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,(uint64_t)(width/256u)*tokens*296u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_K quantization dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t width,blocks,tokens;}push={width,width/256u,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q8k,bindings,&push,width/256u,tokens,1,err);}
fg_status fg_vk_quantize_q8_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){uint32_t blocks=width/32u;uint64_t bytes=(uint64_t)blocks*34u*tokens;if(!c||!width||width%64u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 quantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,blocks,tokens;}push={width,blocks,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q8,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_quantize_q4_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){uint32_t blocks=width/32u;uint64_t bytes=(uint64_t)blocks*18u*tokens;if(!c||!width||width%64u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q4_0 quantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,blocks,tokens;}push={width,blocks,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q4,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_dequantize_iq4_nl(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t rows,uint32_t width,fg_error *err){uint64_t input_bytes=(uint64_t)rows*(width/32u)*18u;if(!c||!rows||!width||width%32u||!tensor_range(input,0,input_bytes)||!tensor_range(out,0,(uint64_t)rows*width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid IQ4_NL dequantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t rows,width;}push={rows,width};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->dequant_iq4nl,bindings,&push,(rows*width+255u)/256u,1,1,err);}
fg_status fg_vk_swiglu(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *gate,const fg_vk_tensor *up,uint32_t values,fg_error *err){if(!c||!values||!tensor_range(gate,0,(uint64_t)values*4u)||!tensor_range(up,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid SwiGLU dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gate,up,out};return dispatch(c,&c->swiglu,bindings,&values,(values+255u)/256u,1,1,err);}

fg_vk_tensor_format fg_vk_dense_format(uint32_t type){
    switch(type){
        case 12u:return FG_VK_TENSOR_FORMAT_Q4_K;
        case 13u:return FG_VK_TENSOR_FORMAT_Q5_K;
        case 14u:return FG_VK_TENSOR_FORMAT_Q6_K;
        case 20u:return FG_VK_TENSOR_FORMAT_IQ4_NL;
        case 30u:return FG_VK_TENSOR_FORMAT_BF16_ROUTER;
        default:return FG_VK_TENSOR_FORMAT_DEFAULT;
    }
}

fg_vk_tensor_format fg_vk_dense_saved_format(uint32_t type,uint32_t layout){
    if(layout==FG_TENSOR_LAYOUT_DENSE_SOA||layout==FG_TENSOR_LAYOUT_DENSE_COMPACT){
        if(type!=12u&&type!=13u&&type!=20u)return FG_VK_TENSOR_FORMAT_DEFAULT;
        uint32_t base=layout==FG_TENSOR_LAYOUT_DENSE_COMPACT?
            FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q4_K:FG_VK_TENSOR_FORMAT_DENSE_SOA_Q4_K;
        return (fg_vk_tensor_format)(base+(type==12u?0u:type==13u?1u:2u));
    }
    if(layout==FG_TENSOR_LAYOUT_Q8_0_COOKED&&type==8u)
        return FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
    return fg_vk_dense_format(type);
}

uint32_t fg_vk_subgroup_size(const fg_vk_context *c){
    if(!c)return 0u;
    VkPhysicalDeviceSubgroupProperties subgroup={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&subgroup};
    vkGetPhysicalDeviceProperties2(c->physical,&properties);
    return subgroup.subgroupSize;
}

fg_status fg_vk_dense_quant(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,
                            const fg_vk_tensor *x,uint32_t in,uint32_t rows,
                            uint32_t tokens,float scale,fg_error *err){
    uint32_t type=0u,block=256u,bytes=0u;
    bool compact=w&&w->format>=FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q4_K&&
        w->format<=FG_VK_TENSOR_FORMAT_DENSE_COMPACT_IQ4_NL;
    bool soa=w&&((w->format>=FG_VK_TENSOR_FORMAT_DENSE_SOA_Q4_K&&
        w->format<=FG_VK_TENSOR_FORMAT_DENSE_SOA_IQ4_NL)||compact);
    bool cooked=w&&(w->format==FG_VK_TENSOR_FORMAT_DENSE_Q4_K_COOKED||
                    w->format==FG_VK_TENSOR_FORMAT_DENSE_Q5_K_COOKED);
    if(w)switch(w->format){
        case FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q4_K:
        case FG_VK_TENSOR_FORMAT_DENSE_SOA_Q4_K:
        case FG_VK_TENSOR_FORMAT_DENSE_Q4_K_COOKED:
        case FG_VK_TENSOR_FORMAT_Q4_K:type=12u;bytes=144u;break;
        case FG_VK_TENSOR_FORMAT_DENSE_SOA_Q5_K:
        case FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q5_K:
        case FG_VK_TENSOR_FORMAT_DENSE_Q5_K_COOKED:
        case FG_VK_TENSOR_FORMAT_Q5_K:type=13u;bytes=176u;break;
        case FG_VK_TENSOR_FORMAT_Q6_K:type=14u;bytes=210u;break;
        case FG_VK_TENSOR_FORMAT_DENSE_SOA_IQ4_NL:
        case FG_VK_TENSOR_FORMAT_DENSE_COMPACT_IQ4_NL:
        case FG_VK_TENSOR_FORMAT_IQ4_NL:type=20u;block=32u;bytes=18u;break;
        default:break;
    }
    uint64_t row_bytes=(uint64_t)(in/block)*bytes;
    if(!c||!type||!in||in%block||!rows||!tokens||(cooked&&rows%FG_K_QUANT_COOK_ROWS)||row_bytes>UINT32_MAX||
       !tensor_range(w,0,compact?fg_dense_compact_bytes(in,rows,type):soa?fg_dense_soa_bytes(in,rows,type):row_bytes*rows)||
       !tensor_range(x,0,(uint64_t)in*tokens*4u)||
       !tensor_range(out,0,(uint64_t)rows*tokens*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid quantized dense dispatch");
        return FG_ERR_ARGUMENT;
    }
    if(c->small_n&&soa&&tokens>=2u&&tokens<=4u){
        static const char *small_files[]={
            "fg_small_q4_n2.spv","fg_small_q4_n3.spv","fg_small_q4_n4.spv",
            "fg_small_q5_n2.spv","fg_small_q5_n3.spv","fg_small_q5_n4.spv",
            "fg_small_iq4_n2.spv","fg_small_iq4_n3.spv","fg_small_iq4_n4.spv"};
        static const char *wide_files[]={
            "fg_small_q4_r8_n2.spv","fg_small_q4_r8_n3.spv","fg_small_q4_r8_n4.spv",
            "fg_small_q5_r8_n2.spv","fg_small_q5_r8_n3.spv","fg_small_q5_r8_n4.spv"};
        uint32_t i=(type==12u?0u:type==13u?3u:6u)+tokens-2u;
        uint32_t geometry=type==12u&&rows<=1024u&&in>=8192u?2u:
            type==12u&&in==2560u&&rows==6144u?1u:
            type!=20u&&rows>1024u?3u:0u;
        bool wide=type!=20u&&rows>1024u;
        fg_vk_kernel *kernel=&c->small_dense[(compact?0u:36u)+i*4u+geometry];
        if(!kernel->file)*kernel=(fg_vk_kernel){.file=wide?wide_files[i]:small_files[i],.bindings=3,
            .push_bytes=24,.specialization_count=2,
            .specialization_values={(geometry==3u?64u:64u<<geometry)*2u,compact?1u:0u}};
        struct{uint32_t in,rows,tokens,type,row_bytes;float scale;}
            p={in,rows,tokens,type,(uint32_t)row_bytes,scale};
        const fg_vk_tensor *b[]={w,x,out};
        uint32_t small_rows=wide?8u:4u;
        return dispatch(c,kernel,b,&p,(rows+small_rows-1u)/small_rows,1u,1u,err);
    }
    struct{uint32_t input_width,output_width,tokens,type,row_bytes;float scale;}
        push={in,rows,tokens,type|(cooked?256u:0u),(uint32_t)row_bytes,scale};
    const fg_vk_tensor *bindings[]={w,x,out};
    uint32_t index=soa?(type==12u?5u:type==13u?6u:7u):type==20u?3u:type-12u;
    static const char *files[]={"fg_dense_q4_k.spv","fg_dense_q5_k.spv",
        "fg_dense_q6_k.spv","fg_dense_iq4_nl.spv",NULL,"fg_dense_soa.spv",
        "fg_dense_soa_q5.spv","fg_dense_soa_iq4.spv","fg_dense_soa_r4.spv",
        "fg_dense_soa_q5_r4.spv","fg_dense_soa_long.spv","fg_dense_soa_mid.spv",
        "fg_dense_soa_tile.spv","fg_dense_soa_q5_tile.spv","fg_dense_soa_iq4_tile.spv",
        "fg_dense_soa_tile8.spv","fg_dense_soa_q5_tile8.spv","fg_dense_soa_iq4_tile8.spv",
        "fg_dense_soa_tile16.spv","fg_dense_soa_q5_tile16.spv","fg_dense_soa_iq4_tile16.spv",
        "fg_dense_compact.spv","fg_dense_compact_q5.spv","fg_dense_compact_iq4.spv",
        "fg_dense_compact_r4.spv","fg_dense_compact_q5_r4.spv","fg_dense_compact_long.spv",
        "fg_dense_compact_mid.spv","fg_dense_compact_tile.spv","fg_dense_compact_q5_tile.spv",
        "fg_dense_compact_iq4_tile.spv","fg_dense_compact_tile8.spv","fg_dense_compact_q5_tile8.spv",
        "fg_dense_compact_iq4_tile8.spv","fg_dense_compact_tile16.spv",
        "fg_dense_compact_q5_tile16.spv","fg_dense_compact_iq4_tile16.spv"};
    uint32_t row_tile=soa?8u:4u;
    uint32_t token_tile=tokens<=8u?8u:tokens<=16u?16u:32u;
    if(soa){
        if(type==20u)row_tile=4u;
        else if(rows<=1024u){
            row_tile=4u;index=type==12u?(in>=8192u?10u:8u):9u;
        }else if(type==12u&&in==2560u&&rows==6144u)index=11u;
        if(tokens>1u)index=(type==12u?12u:type==13u?13u:14u)+
            (token_tile==8u?3u:token_tile==16u?6u:0u);
    }
    const char *file=files[index];
    if(compact){
        index+=16u;file=files[index];
        if(type==20u&&tokens==1u){
            row_tile=4u;
            const char *variant=getenv("FG_BENCH_IQ_VARIANT");
            if(variant){
                if(strcmp(variant,"r4w32")==0){file="fg_dense_compact_iq4_r4w32.spv";row_tile=4u;}
                else if(strcmp(variant,"r8w32")==0){file="fg_dense_compact_iq4_r8w32.spv";row_tile=8u;}
                else if(strcmp(variant,"r4w128")==0){file="fg_dense_compact_iq4_r4w128.spv";row_tile=4u;}
                else if(strcmp(variant,"r8w128")==0){file="fg_dense_compact_iq4_r8w128.spv";row_tile=8u;}
            }
        }
    }
    if(soa&&!compact&&type==12u&&tokens==1u&&rows>=2560u&&roof_enabled(c,1u)){
        const char *roof_variant=getenv("FG_ROOF_DENSE");
        if(roof_variant&&strtoul(roof_variant,NULL,0)==21u)
            return dispatch(c,roof_kernel(c,39u,"fg_roof_dense_wide_l2.spv",3u,24u),
                bindings,&push,(rows+7u)/8u,1u,1u,err);
    }
    if(compact&&(type==12u||type==20u)&&tokens==1u&&roof_enabled(c,1u)){
        const char *variant=getenv("FG_ROOF_DENSE");
        uint32_t v=variant?(uint32_t)strtoul(variant,NULL,0):7u;
        if(v>=25u&&v<=28u){
            if(type==12u&&rows>=2560u&&(v<=26u||c->roof_wave32)){
                static const char *files[]={"fg_roof_dense_rows_p2.spv","fg_roof_dense_rows_p4.spv",
                    "fg_roof_dense_rows_p2_w32.spv","fg_roof_dense_rows_p4_w32.spv"};
                return dispatch(c,roof_kernel(c,62u+v-25u,files[v-25u],3u,24u),bindings,&push,
                    (rows+(v<=26u?15u:7u))/(v<=26u?16u:8u),1u,1u,err);
            }
            v=19u;
        }
        if(v==23u||v==24u){
            if(type==20u&&(v==23u||c->roof_wave32))
                return dispatch(c,roof_kernel(c,v==23u?51u:52u,
                    v==23u?"fg_roof_dense_iq4_l2.spv":"fg_roof_dense_iq4_l1_w32.spv",3u,24u),
                    bindings,&push,(rows+7u)/8u,1u,1u,err);
            v=19u;
        }
        if(v==22u){
            if(type==12u)
                return dispatch(c,roof_kernel(c,40u,"fg_roof_dense_l2_w128.spv",3u,24u),
                    bindings,&push,(rows+7u)/8u,1u,1u,err);
            v=6u;
        }
        if(v==19u)v=type==12u&&rows>=2560u?
            (in==2560u&&(rows==6144u||rows==12288u)?18u:15u):6u;
        if(v==20u){
            if(type==20u&&c->roof_wave32)
                return dispatch(c,roof_kernel(c,38u,"fg_roof_dense_iq4_w32.spv",3u,24u),
                    bindings,&push,(rows+7u)/8u,1u,1u,err);
            v=6u;
        }
        if(v>=15u&&v<=18u){
            if(type==12u&&(v<=16u||c->roof_wave32)){
                static const char *files[]={"fg_roof_dense_l2.spv","fg_roof_dense_l1.spv",
                    "fg_roof_dense_l2_w32.spv","fg_roof_dense_l1_w32.spv"};
                return dispatch(c,roof_kernel(c,34u+v-15u,files[v-15u],3u,24u),
                    bindings,&push,(rows+7u)/8u,1u,1u,err);
            }
            v=6u;
        }
        if(v==13u||v==14u){
            if(type==12u&&in>=8192u&&rows<=1024u){
                fg_status status=FG_OK;
                if(!c->roof_partials)status=fg_vk_tensor_create(c,8u*1024u*4u,&c->roof_partials,err);
                if(status!=FG_OK)return status;
                uint32_t splits=v==13u?4u:8u;
                const fg_vk_tensor *partial_bindings[]={w,x,c->roof_partials};
                status=dispatch(c,roof_kernel(c,31u+(v-13u),
                    v==13u?"fg_roof_stream_q4_split4.spv":"fg_roof_stream_q4_split8.spv",3u,24u),
                    partial_bindings,&push,(rows+7u)/8u,splits,1u,err);
                const fg_vk_tensor *merge_bindings[]={c->roof_partials,out};
                struct{uint32_t m,splits;}merge={rows,splits};
                return status==FG_OK?dispatch(c,roof_kernel(c,33u,"fg_roof_split_merge.spv",2u,8u),
                    merge_bindings,&merge,(rows+255u)/256u,1u,1u,err):status;
            }
            v=6u;
        }
        if(v>=8u&&v<=12u&&(v<11u||c->roof_wave32)){
            static const char *files[][2]={
                {"fg_roof_stream_q4_p1.spv","fg_roof_stream_iq4_p1.spv"},
                {"fg_roof_stream_q4_p2.spv","fg_roof_stream_iq4_p2.spv"},
                {"fg_roof_stream_q4_p4.spv","fg_roof_stream_iq4_p4.spv"},
                {"fg_roof_stream_q4_p2_w32.spv","fg_roof_stream_iq4_p2_w32.spv"},
                {"fg_roof_stream_q4_p4_w32.spv","fg_roof_stream_iq4_p4_w32.spv"}};
            return dispatch(c,roof_kernel(c,15u+(v-8u)*2u+(type==20u),files[v-8u][type==20u],3u,24u),
                bindings,&push,(rows+7u)/8u,1u,1u,err);
        }
        if(type!=12u||rows<2560u)v=0u;
        if(v==6u)v=in==2560u?5u:4u;
        if(v==7u)v=in==2560u&&rows==6144u?0u:5u;
        if(v>=1u&&v<=5u){
            static const char *roof_files[]={"fg_roof_dense_r4w64.spv",
                "fg_roof_dense_r4w128.spv","fg_roof_dense_r8w256.spv",
                "fg_roof_dense_r8w128.spv","fg_roof_dense_r8w64.spv"};
            row_tile=v<=2u?4u:8u;
            return dispatch(c,roof_kernel(c,v-1u,roof_files[v-1u],3u,24u),
                bindings,&push,(rows+row_tile-1u)/row_tile,1u,1u,err);
        }
    }
    if(!c->dense_quant[index].file)c->dense_quant[index]=(fg_vk_kernel){
        .file=file,.bindings=3,.push_bytes=24};
    return dispatch(c,&c->dense_quant[index],bindings,&push,
        soa&&tokens>1u?(rows+63u)/64u:(rows+row_tile-1u)/row_tile,
        soa&&tokens>1u?(tokens+token_tile-1u)/token_tile:tokens,1u,err);
}

fg_status fg_vk_dense_q8_0_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){if(w&&w->format>=FG_VK_TENSOR_FORMAT_Q4_K)return fg_vk_dense_quant(c,out,w,x,in,rows,tokens,scale,err);if(w&&w->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED)return fg_vk_dense_q8_0_cooked(c,out,w,x,in,rows,tokens,scale,err);uint64_t row_bytes=(uint64_t)(in/32u)*34u;if(!c||in==0||in%32u||!tensor_range(w,0,row_bytes*rows)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,n_tok,blocks,row_bytes;float scale;} push={rows,tokens,in/32u,(uint32_t)row_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense,bindings,&push,rows,tokens,1,err);}
fg_status fg_vk_dense_q8_0_subgroup(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){uint64_t row_bytes=(uint64_t)(in/32u)*34u;if(!c||!w||w->format!=FG_VK_TENSOR_FORMAT_DEFAULT||in==0||in%32u||!tensor_range(w,0,row_bytes*rows)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid subgroup Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,row_bytes;float scale;}push={rows,tokens,in/32u,(uint32_t)row_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense_subgroup,bindings,&push,rows,tokens,1,err);}
fg_status fg_vk_dense_q8_0_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){if(tokens>1u)return fg_vk_dense_q8_0_cooked_prefill(c,out,w,x,in,rows,tokens,scale,err);uint32_t blocks=in/FG_QK8_0;uint64_t quant_offset=fg_align_up_u64(FG_Q8_0_COOK_ROWS*(uint64_t)blocks*sizeof(uint16_t),FG_Q8_0_COOK_ALIGNMENT),tile_bytes=fg_align_up_u64(quant_offset+FG_Q8_0_COOK_ROWS*(uint64_t)blocks*FG_QK8_0,FG_Q8_0_COOK_ALIGNMENT),tiles=((uint64_t)rows+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS;if(!c||!in||in%FG_QK8_0||tile_bytes>UINT32_MAX||tiles>UINT64_MAX/tile_bytes||!tensor_range(w,0,tile_bytes*tiles)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;}push={rows,tokens,blocks,(uint32_t)tile_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense_cooked,bindings,&push,(rows+3u)/4u,tokens,1,err);}
fg_status fg_vk_mtp_vocab_q8(fg_vk_context *c,fg_vk_tensor *scores,fg_vk_tensor *ids,
    const fg_vk_tensor *w,const fg_vk_tensor *x,const fg_vk_tensor *token_ids,
    uint32_t in,uint32_t rows,fg_error *err){
    uint64_t tile=fg_q8_0_cooked_tile_bytes(in),bytes=fg_q8_0_cooked_matrix_bytes(in,rows);
    uint32_t groups=(rows+7u)/8u;
    if(!c||!in||in%32u||!rows||tile>UINT32_MAX||!bytes||
       !tensor_on_context(c,w)||!tensor_on_context(c,x)||!tensor_on_context(c,token_ids)||
       !tensor_on_context(c,scores)||!tensor_on_context(c,ids)||
       fg_vk_tensor_get_format(w)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
       !tensor_range(w,0,bytes)||!tensor_range(x,0,(uint64_t)in*4u)||
       !tensor_range(token_ids,0,(uint64_t)rows*4u)||
       !tensor_range(scores,0,(uint64_t)groups*4u)||!tensor_range(ids,0,(uint64_t)groups*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP Q8 vocabulary dispatch");return FG_ERR_ARGUMENT;
    }
    struct{uint32_t rows,tokens,blocks,tile;float scale;} push={rows,1u,in/32u,(uint32_t)tile,1.0f};
    const fg_vk_tensor *bindings[]={w,x,scores,token_ids,ids};
    return dispatch(c,&c->mtp_vocab,bindings,&push,groups,1u,1u,err);
}
fg_status fg_vk_mtp_fc_hidden(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *w,const fg_vk_tensor *x,fg_error *err){
    uint64_t bytes=fg_q8_0_cooked_matrix_bytes(2560u,2560u);
    if(!c||!tensor_on_context(c,w)||!tensor_on_context(c,x)||!tensor_on_context(c,out)||
       fg_vk_tensor_get_format(w)!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||
       !tensor_range(w,0,bytes)||!tensor_range(x,0,10240u*4u)||!tensor_range(out,0,10240u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid native four-stream FC");return FG_ERR_ARGUMENT;
    }
    struct{uint32_t rows,tokens,blocks,tile;float scale;}push={
        2560u,4u,80u,(uint32_t)fg_q8_0_cooked_tile_bytes(2560u),1.0f};
    const fg_vk_tensor *bindings[]={w,x,out};
    return dispatch(c,&c->mtp_fc,bindings,&push,2560u/8u*4u,1u,1u,err);
}
fg_status fg_vk_dense_q8_0_cooked_prefill(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,
    uint32_t tokens,float scale,fg_error *err){
    uint32_t blocks=in/FG_QK8_0;
    uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(in),
        matrix_bytes=fg_q8_0_cooked_matrix_bytes(in,rows);
    if(!c||!w||!blocks||!rows||tokens<2u||tile_bytes>UINT32_MAX||
       w->format!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||!tensor_range(w,0,matrix_bytes)||
       !tensor_range(x,0,(uint64_t)in*tokens*4u)||
       !tensor_range(out,0,(uint64_t)rows*tokens*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid token-tiled cooked Q8_0 prefill dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,tokens,blocks,tile_bytes;float scale;}
        push={rows,tokens,blocks,(uint32_t)tile_bytes,scale};
    const fg_vk_tensor *bindings[]={w,x,out};
    if(c->small_n&&tokens<=4u){
        static const char *files[]={"fg_small_q8_n2.spv","fg_small_q8_n3.spv","fg_small_q8_n4.spv"};
        fg_vk_kernel *k=&c->small_dense[72u+tokens-2u];
        if(!k->file)*k=(fg_vk_kernel){.file=files[tokens-2u],.bindings=3,.push_bytes=20,
            .specialization_count=1,.specialization_values={128u,0u}};
        return dispatch(c,k,bindings,&push,(rows+3u)/4u,1u,1u,err);
    }
    return dispatch(c,&c->dense_cooked_tile,bindings,&push,(rows+63u)/64u,
        (tokens+31u)/32u,1u,err);
}
fg_status fg_vk_dense_q8_0_cooked_split(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *partials,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,uint32_t splits,float scale,fg_error *err){uint32_t blocks=in/FG_QK8_0;uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(in),matrix_bytes=fg_q8_0_cooked_matrix_bytes(in,rows),values=(uint64_t)rows*tokens;if(!c||!blocks||!rows||!tokens||splits<2u||splits>16u||tile_bytes>UINT32_MAX||values>UINT32_MAX||!tensor_range(w,0,matrix_bytes)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(partials,0,values*splits*4u)||!tensor_range(out,0,values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid split cooked Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,tile_bytes,splits;float scale;}split_push={rows,tokens,blocks,(uint32_t)tile_bytes,splits,scale};const fg_vk_tensor *split_bindings[]={w,x,partials};fg_status status=dispatch(c,&c->dense_cooked_split,split_bindings,&split_push,(rows+3u)/4u,tokens*splits,1,err);struct{uint32_t values,splits;}reduce_push={(uint32_t)values,splits};const fg_vk_tensor *reduce_bindings[]={partials,out};if(status==FG_OK)status=dispatch(c,&c->dense_cooked_split_reduce,reduce_bindings,&reduce_push,((uint32_t)values+255u)/256u,1,1,err);return status;}
fg_status fg_vk_dense_q8_0_cooked_split_silu(fg_vk_context *c,fg_vk_tensor *low,fg_vk_tensor *active,fg_vk_tensor *partials,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,uint32_t splits,float scale,float input_scale,fg_error *err){uint32_t blocks=in/FG_QK8_0;uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(in),matrix_bytes=fg_q8_0_cooked_matrix_bytes(in,rows),values=(uint64_t)rows*tokens;if(!c||!blocks||!rows||!tokens||splits<2u||splits>16u||tile_bytes>UINT32_MAX||values>UINT32_MAX||!tensor_range(w,0,matrix_bytes)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(partials,0,values*splits*4u)||!tensor_range(low,0,values*4u)||!tensor_range(active,0,values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid split cooked Q8_0 dense scaled SiLU dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,tile_bytes,splits;float scale;}split_push={rows,tokens,blocks,(uint32_t)tile_bytes,splits,scale};const fg_vk_tensor *split_bindings[]={w,x,partials};fg_status status=dispatch(c,&c->dense_cooked_split,split_bindings,&split_push,(rows+3u)/4u,tokens*splits,1,err);struct{uint32_t values,splits;float input_scale;}reduce_push={(uint32_t)values,splits,input_scale};const fg_vk_tensor *reduce_bindings[]={partials,low,active};if(status==FG_OK)status=dispatch(c,&c->dense_cooked_split_reduce_silu,reduce_bindings,&reduce_push,((uint32_t)values+255u)/256u,1,1,err);return status;}
fg_status fg_vk_embedding_q8_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,uint32_t token,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){
    uint64_t row_bytes=(uint64_t)(width/32u)*34u;
    if(!c||!weights||weights->format!=FG_VK_TENSOR_FORMAT_DEFAULT||!width||width%32u||!rows||token>=rows||!copies||!tensor_range(weights,0,row_bytes*rows)||!tensor_range(out,0,(uint64_t)width*copies*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 embedding dispatch");return FG_ERR_ARGUMENT;}
    const uint8_t *source=fg_vk_tensor_const_map(weights);float *destination=fg_vk_tensor_map(out);
    if(source&&destination){
        const uint8_t *row=source+(uint64_t)token*row_bytes;uint32_t blocks=width/32u;
        for(uint32_t block=0;block<blocks;block++){
            const uint8_t *packed=row+(uint64_t)block*34u;
            float scale=fg_f16_to_f32((uint16_t)packed[0]|((uint16_t)packed[1]<<8u));
            for(uint32_t i=0;i<32u;i++){
                float value=scale*(float)(int8_t)packed[2u+i];
                for(uint32_t copy=0;copy<copies;copy++)destination[(uint64_t)copy*width+block*32u+i]=value;
            }
        }
        return FG_OK;
    }
    struct{uint32_t token,width,rows,copies;}push={token,width,rows,copies};const fg_vk_tensor *bindings[]={weights,out};return dispatch(c,&c->embedding,bindings,&push,(width*copies+255u)/256u,1,1,err);
}
fg_status fg_vk_embedding_q8_0_batch(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *token_ids,uint32_t token_count,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){uint64_t row_bytes=(uint64_t)(width/32u)*34u,total=(uint64_t)token_count*width*copies;if(!c||!weights||weights->format!=FG_VK_TENSOR_FORMAT_DEFAULT||!token_count||!width||width%32u||!rows||!copies||!tensor_range(weights,0,row_bytes*rows)||!tensor_range(token_ids,0,(uint64_t)token_count*4u)||!tensor_range(out,0,total*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched Q8_0 embedding dispatch");return FG_ERR_ARGUMENT;}const uint32_t *ids=fg_vk_tensor_map((fg_vk_tensor *)token_ids);for(uint32_t i=0;i<token_count;i++)if(ids[i]>=rows){fg_error_set(err,FG_ERR_FORMAT,"embedding token %u is outside vocabulary",i);return FG_ERR_FORMAT;}struct{uint32_t token_count,width,rows,copies;}push={token_count,width,rows,copies};const fg_vk_tensor *bindings[]={weights,token_ids,out};return dispatch(c,&c->embedding_batch,bindings,&push,(uint32_t)((total+255u)/256u),1,1,err);}
fg_status fg_vk_mtp_hc_norm(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *x,const fg_vk_tensor *w,fg_error *err){
    if(!c||!tensor_on_context(c,out)||!tensor_on_context(c,x)||!tensor_on_context(c,w)||
       !tensor_range(out,0,10240u*4u)||!tensor_range(x,0,10240u*4u)||!tensor_range(w,0,10240u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid native HC normalization");return FG_ERR_ARGUMENT;
    }
    float epsilon=1e-6f;
    const fg_vk_tensor *bindings[]={x,w,out};
    return dispatch(c,&c->mtp_norm,bindings,&epsilon,1u,1u,1u,err);
}
fg_status fg_vk_mtp_attention_split(fg_vk_context *c,fg_vk_tensor *out,
    fg_vk_tensor *partials,fg_vk_tensor *keys,fg_vk_tensor *values,
    const fg_vk_tensor *q,const fg_vk_tensor *gate,const fg_vk_tensor *key,
    const fg_vk_tensor *value,const fg_vk_tensor *position,uint32_t capacity,
    uint32_t splits,fg_error *err){
    return fg_vk_mtp_attention_window(c,out,partials,keys,values,q,gate,key,value,
        position,capacity,splits,0u,err);
}
fg_status fg_vk_mtp_attention_window(fg_vk_context *c,fg_vk_tensor *out,
    fg_vk_tensor *partials,fg_vk_tensor *keys,fg_vk_tensor *values,
    const fg_vk_tensor *q,const fg_vk_tensor *gate,const fg_vk_tensor *key,
    const fg_vk_tensor *value,const fg_vk_tensor *position,uint32_t capacity,
    uint32_t splits,uint32_t window,fg_error *err){
    const fg_vk_tensor *tensors[]={q,gate,key,value,keys,values,partials,position,out};
    if(!c||!capacity||capacity>UINT32_MAX/128u||!splits||splits>64u||
       (window&&(window>capacity||capacity-window<4u))){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid split draft attention");return FG_ERR_ARGUMENT;
    }
    if(fg_vk_subgroup_size(c)!=64u){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"split MTP attention requires wave64");return FG_ERR_UNAVAILABLE;
    }
    uint64_t sizes[]={6144u*4u,6144u*4u,512u*4u,512u*4u,
        (uint64_t)capacity*512u*4u,(uint64_t)capacity*512u*4u,
        (uint64_t)24u*splits*258u*4u,window?16u:4u,6144u*4u};
    for(uint32_t i=0;i<9u;i++)if(!tensor_on_context(c,tensors[i])||!tensor_range(tensors[i],0,sizes[i])){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid split draft attention storage");return FG_ERR_ARGUMENT;
    }
    struct{uint32_t capacity,splits,window;}push={capacity,splits,window};
    fg_status status=dispatch(c,&c->mtp_attention_split,tensors,&push,24u,splits,1u,err);
    const fg_vk_tensor *merge[]={partials,gate,out};
    if(status==FG_OK)status=dispatch(c,&c->qsa_attention_merge,merge,&splits,24u,1u,1u,err);
    return status;
}
fg_status fg_vk_mtp_cache_kv(fg_vk_context *c,fg_vk_tensor *keys,
    fg_vk_tensor *values,const fg_vk_tensor *key,const fg_vk_tensor *value,
    const fg_vk_tensor *norm,const fg_vk_tensor *position,uint32_t capacity,
    fg_error *err){
    const fg_vk_tensor *tensors[]={key,value,norm,position,keys,values};
    uint64_t sizes[]={512u*4u,512u*4u,256u*4u,12u,
        (uint64_t)capacity*512u*4u,(uint64_t)capacity*512u*4u};
    if(!c||!capacity||capacity>UINT32_MAX/512u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid native MTP KV capacity");return FG_ERR_ARGUMENT;
    }
    for(uint32_t i=0;i<6u;i++)
        if(!tensor_on_context(c,tensors[i])||!tensor_range(tensors[i],0,sizes[i])){
            fg_error_set(err,FG_ERR_ARGUMENT,"invalid native MTP KV storage");return FG_ERR_ARGUMENT;
        }
    return dispatch(c,&c->mtp_cache_kv,tensors,&capacity,2u,1u,1u,err);
}
fg_status fg_vk_mtp_attention_indirect(fg_vk_context *c,fg_vk_tensor *out,
    fg_vk_tensor *keys,fg_vk_tensor *values,const fg_vk_tensor *query,
    const fg_vk_tensor *gate,const fg_vk_tensor *key,const fg_vk_tensor *value,
    const fg_vk_tensor *position,uint32_t capacity,fg_error *err){
    uint64_t bytes=(uint64_t)capacity*512u*4u;
    if(!c||!capacity||!tensor_range(out,0,6144u*4u)||!tensor_range(query,0,6144u*4u)||
       !tensor_range(gate,0,6144u*4u)||!tensor_range(key,0,512u*4u)||
       !tensor_range(value,0,512u*4u)||!tensor_range(keys,0,bytes)||
       !tensor_range(values,0,bytes)||!tensor_range(position,0,4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid indirect MTP attention");return FG_ERR_ARGUMENT;
    }
    if(fg_vk_subgroup_size(c)!=64u){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"dense MTP attention requires wave64");return FG_ERR_UNAVAILABLE;
    }
    const fg_vk_tensor *bindings[]={query,gate,key,value,keys,values,out,position};
    return dispatch(c,&c->mtp_attention_indirect,bindings,&capacity,24u,1u,1u,err);
}
fg_status fg_vk_mtp_attention(fg_vk_context *c,fg_vk_tensor *out,
    fg_vk_tensor *keys,fg_vk_tensor *values,const fg_vk_tensor *query,
    const fg_vk_tensor *gate,const fg_vk_tensor *key,const fg_vk_tensor *value,
    uint32_t cell,fg_error *err){
    uint64_t cache_bytes=((uint64_t)cell+1u)*512u*4u;
    if(!c||!tensor_range(out,0,6144u*4u)||!tensor_range(query,0,6144u*4u)||
       !tensor_range(gate,0,6144u*4u)||!tensor_range(key,0,512u*4u)||
       !tensor_range(value,0,512u*4u)||!tensor_range(keys,0,cache_bytes)||
       !tensor_range(values,0,cache_bytes)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid dense MTP attention");return FG_ERR_ARGUMENT;
    }
    if(fg_vk_subgroup_size(c)!=64u){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"dense MTP attention requires wave64");return FG_ERR_UNAVAILABLE;
    }
    const fg_vk_tensor *bindings[]={query,gate,key,value,keys,values,out};
    return dispatch(c,&c->mtp_attention,bindings,&cell,24u,1u,1u,err);
}

static fg_status small_plain_dispatch(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *weights,const fg_vk_tensor *input,uint32_t in,
    uint32_t rows,uint32_t tokens,uint32_t kind,fg_error *err){
    static const char *files[]={
        "fg_small_f32_n2.spv","fg_small_f32_n3.spv","fg_small_f32_n4.spv",
        "fg_small_router_bf16_n2.spv","fg_small_router_bf16_n3.spv","fg_small_router_bf16_n4.spv",
        "fg_small_index_bf16_n2.spv","fg_small_index_bf16_n3.spv","fg_small_index_bf16_n4.spv"};
    uint32_t i=kind*3u+tokens-2u;
    fg_vk_kernel *kernel=&c->small_plain[i];
    if(!kernel->file)*kernel=(fg_vk_kernel){.file=files[i],.bindings=3,.push_bytes=12};
    uint32_t push[]={in,rows,tokens};
    const fg_vk_tensor *bindings[]={weights,input,out};
    return dispatch(c,kernel,bindings,push,rows,1u,1u,err);
}
fg_status fg_vk_dense_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *input,uint32_t input_width,uint32_t output_width,uint32_t tokens,fg_error *err){
    bool bf16=weights&&weights->format==FG_VK_TENSOR_FORMAT_BF16_ROUTER;
    if(!c||!input_width||(bf16&&input_width%2u)||!output_width||!tokens||
       !tensor_range(weights,0,(uint64_t)input_width*output_width*(bf16?2u:4u))||
       !tensor_range(input,0,(uint64_t)input_width*tokens*4u)||
       !tensor_range(out,0,(uint64_t)output_width*tokens*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid F32/BF16 router dispatch");
        return FG_ERR_ARGUMENT;
    }
    if(c->small_n&&tokens>=2u&&tokens<=4u)
        return small_plain_dispatch(c,out,weights,input,input_width,output_width,
            tokens,bf16?1u:0u,err);
    struct{uint32_t input_width,output_width,tokens;}push={input_width,output_width,tokens};
    const fg_vk_tensor *bindings[]={weights,input,out};
    if(bf16&&!c->dense_quant[4].file)c->dense_quant[4]=(fg_vk_kernel){
        .file="fg_dense_router_bf16.spv",.bindings=3,.push_bytes=12};
    return dispatch(c,bf16?&c->dense_quant[4]:&c->dense_f32,bindings,&push,output_width,tokens,1,err);
}
fg_status fg_vk_dense_bf16_f32(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *weights,const fg_vk_tensor *input,uint32_t input_width,
    uint32_t output_width,uint32_t tokens,fg_error *err){
    if(!c||!input_width||input_width%2u||!output_width||!tokens||
       !tensor_range(weights,0,(uint64_t)input_width*output_width*2u)||
       !tensor_range(input,0,(uint64_t)input_width*tokens*4u)||
       !tensor_range(out,0,(uint64_t)output_width*tokens*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid BF16 dense dispatch");
        return FG_ERR_ARGUMENT;
    }
    if(c->small_n&&tokens>=2u&&tokens<=4u)
        return small_plain_dispatch(c,out,weights,input,input_width,output_width,tokens,2u,err);
    struct{uint32_t input_width,output_width,tokens;}push={input_width,output_width,tokens};
    const fg_vk_tensor *bindings[]={weights,input,out};
    return dispatch(c,&c->dense_bf16,bindings,&push,output_width,tokens,1u,err);
}
fg_status fg_vk_silu_scaled(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t values,float input_scale,fg_error *err){if(!c||!values||!tensor_range(input,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid scaled SiLU dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t values;float scale;}push={values,input_scale};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->silu_scaled,bindings,&push,(values+255u)/256u,1,1,err);}
fg_status fg_vk_group_rms_norm(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *x,const fg_vk_tensor *w,uint32_t width,uint32_t groups,uint32_t tokens,float epsilon,fg_error *err){uint64_t values=(uint64_t)width*groups*tokens;if(!c||!width||!groups||!tokens||!tensor_range(x,0,values*4u)||!tensor_range(w,0,(uint64_t)width*groups*4u)||!tensor_range(out,0,values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped RMS dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t width,groups,tokens;float epsilon;} push={width,groups,tokens,epsilon};const fg_vk_tensor *bindings[]={x,w,out};return dispatch(c,&c->rms,bindings,&push,groups,tokens,1,err);}
fg_status fg_vk_gr_mix(fg_vk_context *c,fg_vk_tensor *mixed,fg_vk_tensor *injection,const fg_vk_tensor *norm,const fg_vk_tensor *up,const fg_vk_tensor *inject,uint32_t hidden,uint32_t groups,uint32_t tokens,fg_error *err){uint64_t hyper=(uint64_t)hidden*groups*tokens;if(!c||!hidden||!groups||!tokens||!tensor_range(norm,0,hyper*4u)||!tensor_range(up,0,hyper*4u)||!tensor_range(inject,0,(uint64_t)groups*tokens*4u)||!tensor_range(mixed,0,(uint64_t)hidden*tokens*4u)||!tensor_range(injection,0,(uint64_t)groups*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t hidden,groups,tokens;} push={hidden,groups,tokens};const fg_vk_tensor *bindings[]={norm,up,inject,mixed,injection};return dispatch(c,&c->gr,bindings,&push,tokens,1,1,err);}
bool fg_vk_hc_up_mix_supported(const fg_vk_context *c,const fg_vk_tensor *w){
    return c&&w&&roof_enabled(c,256u)&&(w->format==FG_VK_TENSOR_FORMAT_DENSE_COMPACT_IQ4_NL||
        w->format==FG_VK_TENSOR_FORMAT_DENSE_SOA_IQ4_NL||w->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
}
bool fg_vk_hc_down_inject_supported(const fg_vk_context *c,const fg_vk_tensor *w){
    return c&&w&&roof_enabled(c,512u)&&w->format==FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q4_K;
}
bool fg_vk_gdn_gate_controls_supported(const fg_vk_context *c,const fg_vk_tensor *w){
    return c&&w&&roof_enabled(c,1024u)&&c->roof_wave32&&
        w->format==FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q4_K;
}
fg_status fg_vk_gdn_gate_controls(fg_vk_context *c,fg_vk_tensor *gate,
    fg_vk_tensor *alpha,fg_vk_tensor *beta,const fg_vk_tensor *w,
    const fg_vk_tensor *alpha_w,const fg_vk_tensor *beta_w,const fg_vk_tensor *hidden,fg_error *err){
    if(!fg_vk_gdn_gate_controls_supported(c,w)||
        !tensor_range(w,0,fg_dense_compact_bytes(2560u,6144u,12u))||
        !tensor_range(hidden,0,2560u*4u)||!tensor_range(gate,0,6144u*4u)||
        !tensor_range(alpha_w,0,48u*2560u*4u)||!tensor_range(beta_w,0,48u*2560u*4u)||
        !tensor_range(alpha,0,48u*4u)||!tensor_range(beta,0,48u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused GDN gate/control dispatch");return FG_ERR_ARGUMENT;
    }
    struct{uint32_t input,output,tokens,type,row_bytes;float scale;}push={2560u,6144u,1u,12u,0u,1.0f};
    const fg_vk_tensor *bindings[]={w,hidden,gate,alpha_w,beta_w,alpha,beta};
    const char *value=getenv("FG_ROOF_GDN");
    uint32_t rows=value?(uint32_t)strtoul(value,NULL,0):8u,slot=74u;
    const char *file="fg_roof_gdn_gate_controls_w32.spv";
    if(rows==1u){slot=79u;file="fg_roof_gdn_gate_controls1_w32.spv";}
    else if(rows==2u){slot=80u;file="fg_roof_gdn_gate_controls2_w32.spv";}
    else if(rows==4u){slot=81u;file="fg_roof_gdn_gate_controls4_w32.spv";}
    else rows=8u;
    return dispatch(c,roof_kernel(c,slot,file,7u,24u),
        bindings,&push,6144u/8u+96u/rows,1u,1u,err);
}
fg_status fg_vk_hc_down_inject(fg_vk_context *c,fg_vk_tensor *low,fg_vk_tensor *active,
    fg_vk_tensor *partials,const fg_vk_tensor *w,const fg_vk_tensor *input,
    const fg_vk_tensor *inject_w,uint32_t pieces,fg_error *err){
    if(!fg_vk_hc_down_inject_supported(c,w)||!pieces||pieces>40u||
       !tensor_range(w,0,fg_dense_compact_bytes(10240u,320u,12u))||
       !tensor_range(input,0,10240u*4u)||!tensor_range(inject_w,0,10240u*16u)||
       !tensor_range(low,0,320u*4u)||!tensor_range(active,0,320u*4u)||!tensor_range(partials,0,(uint64_t)pieces*16u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused HC down/injection dispatch");return FG_ERR_ARGUMENT;
    }
    const char *value=getenv("FG_ROOF_HC");
    uint32_t variant=value?(uint32_t)strtoul(value,NULL,0):0u;
    struct{uint32_t in,out,tokens,type,row_bytes;float scale;uint32_t pieces;}push={10240u,320u,1u,12u,0u,1.0f,pieces};
    if((variant&32u)!=0u){
        uint32_t single_slot=82u;
        const char *single="fg_roof_hc_down_single.spv";
        if((variant&128u)!=0u){single_slot=86u;single="fg_roof_hc_down_single_w512.spv";}
        else if((variant&64u)!=0u){single_slot=85u;single="fg_roof_hc_down_single_w256.spv";}
        else if((variant&16u)!=0u&&c->roof_wave32){single_slot=84u;single="fg_roof_hc_down_single_w32.spv";}
        else if((variant&8u)!=0u){single_slot=83u;single="fg_roof_hc_down_single_w64.spv";}
        const fg_vk_tensor *bindings[]={w,input,low,inject_w,partials,active};
        return dispatch(c,roof_kernel(c,single_slot,single,6u,28u),bindings,&push,40u,1u,1u,err);
    }
    fg_status status=FG_OK;
    if(!c->roof_partials)status=fg_vk_tensor_create(c,8u*1024u*4u,&c->roof_partials,err);
    if(status!=FG_OK)return status;
    uint32_t splits=(variant&4u)!=0u?8u:4u,slot=splits==4u?66u:68u;
    const char *file=splits==4u?"fg_roof_hc_down4.spv":"fg_roof_hc_down8.spv";
    if((variant&16u)!=0u&&c->roof_wave32){
        slot=splits==4u?71u:72u;file=splits==4u?"fg_roof_hc_down4_w32.spv":"fg_roof_hc_down8_w32.spv";
    }else if((variant&8u)!=0u){
        slot=splits==4u?69u:70u;file=splits==4u?"fg_roof_hc_down4_w64.spv":"fg_roof_hc_down8_w64.spv";
    }
    const fg_vk_tensor *first[]={w,input,c->roof_partials,inject_w,partials};
    status=dispatch(c,roof_kernel(c,slot,file,5u,28u),first,&push,40u,splits,1u,err);
    const fg_vk_tensor *last[]={c->roof_partials,low,active};
    struct{uint32_t width,splits;}merge={320u,splits};
    return status==FG_OK?dispatch(c,roof_kernel(c,67u,"fg_roof_hc_down_merge.spv",3u,8u),last,&merge,3u,1u,1u,err):status;
}
fg_status fg_vk_hc_up_mix(fg_vk_context *c,fg_vk_tensor *mixed,fg_vk_tensor *injection,
    const fg_vk_tensor *w,const fg_vk_tensor *input,const fg_vk_tensor *normalized,
    const fg_vk_tensor *partials,uint32_t in,uint32_t hidden,uint32_t pieces,fg_error *err){
    bool q8=w&&w->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
    uint64_t bytes=hidden>UINT32_MAX/4u?0u:q8?fg_q8_0_cooked_matrix_bytes(in,hidden*4u):
        fg_dense_compact_bytes(in,hidden*4u,20u);
    if(!fg_vk_hc_up_mix_supported(c,w)||(in!=320u&&in!=640u)||!hidden||hidden%8u||
       !bytes||!pieces||pieces>64u||!tensor_range(w,0,bytes)||!tensor_range(input,0,(uint64_t)in*4u)||
       !tensor_range(normalized,0,(uint64_t)hidden*16u)||!tensor_range(partials,0,(uint64_t)pieces*16u)||
       !tensor_range(mixed,0,(uint64_t)hidden*4u)||!tensor_range(injection,0,16u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused HC up/mix dispatch");return FG_ERR_ARGUMENT;
    }
    struct{uint32_t in,hidden,pieces;}push={in,hidden,pieces};
    const fg_vk_tensor *bindings[]={w,input,normalized,partials,mixed,injection};
    return dispatch(c,roof_kernel(c,q8?49u:48u,q8?"fg_roof_hc_up_mix_q8.spv":"fg_roof_hc_up_mix_iq4.spv",6u,12u),
        bindings,&push,hidden/8u,1u,1u,err);
}
fg_status fg_vk_hc_inject_partial(fg_vk_context *c,fg_vk_tensor *partials,const fg_vk_tensor *normalized,const fg_vk_tensor *weights,uint32_t hidden,uint32_t groups,uint32_t tokens,uint32_t pieces,fg_error *err){uint64_t width=(uint64_t)hidden*groups;if(!c||!hidden||!groups||groups>4u||!tokens||!pieces||pieces>64u||!tensor_range(normalized,0,width*tokens*4u)||!tensor_range(weights,0,width*groups*4u)||!tensor_range(partials,0,(uint64_t)tokens*pieces*groups*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid HC injection partial dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t hidden,groups,tokens,pieces;}push={hidden,groups,tokens,pieces};const fg_vk_tensor *bindings[]={normalized,weights,partials};return dispatch(c,&c->hc_inject_partial,bindings,&push,pieces,tokens,1,err);}
fg_status fg_vk_gr_mix_partial(fg_vk_context *c,fg_vk_tensor *mixed,fg_vk_tensor *injection,const fg_vk_tensor *normalized,const fg_vk_tensor *up,const fg_vk_tensor *partials,uint32_t hidden,uint32_t groups,uint32_t tokens,uint32_t pieces,fg_error *err){uint64_t hyper=(uint64_t)hidden*groups*tokens;if(!c||!hidden||!groups||groups>4u||!tokens||!pieces||pieces>64u||!tensor_range(normalized,0,hyper*4u)||!tensor_range(up,0,hyper*4u)||!tensor_range(partials,0,(uint64_t)tokens*pieces*groups*4u)||!tensor_range(mixed,0,(uint64_t)hidden*tokens*4u)||!tensor_range(injection,0,(uint64_t)groups*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual partial dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t hidden,groups,tokens,pieces;}push={hidden,groups,tokens,pieces};const fg_vk_tensor *bindings[]={normalized,up,partials,mixed,injection};return dispatch(c,&c->gr_partial,bindings,&push,tokens*((hidden+255u)/256u),1,1,err);}
fg_status fg_vk_hc_finalize(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *normalized,const fg_vk_tensor *up,uint32_t hidden,uint32_t groups,uint32_t tokens,fg_error *err){uint64_t hyper=(uint64_t)hidden*groups*tokens;if(!c||!hidden||!groups||!tokens||!tensor_range(normalized,0,hyper*4u)||!tensor_range(up,0,hyper*4u)||!tensor_range(output,0,(uint64_t)hidden*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid final hyper-connection dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t hidden,groups,tokens;}push={hidden,groups,tokens};const fg_vk_tensor *bindings[]={normalized,up,output};return dispatch(c,&c->hc_finalize,bindings,&push,(hidden*tokens+255u)/256u,1,1,err);}
fg_status fg_vk_gr_write(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *hyper,const fg_vk_tensor *block,const fg_vk_tensor *injection,uint32_t hidden,uint32_t groups,uint32_t tokens,fg_error *err){uint64_t values=(uint64_t)hidden*groups*tokens;if(!c||!hidden||!groups||!tokens||!tensor_range(hyper,0,values*4u)||!tensor_range(block,0,(uint64_t)hidden*tokens*4u)||!tensor_range(injection,0,(uint64_t)groups*tokens*4u)||!tensor_range(out,0,values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual write dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t hidden,groups,tokens;}push={hidden,groups,tokens};const fg_vk_tensor *bindings[]={hyper,block,injection,out};return dispatch(c,&c->gr_write,bindings,&push,(uint32_t)((values+255u)/256u),1,1,err);}
fg_status fg_vk_ple_gate(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *key,const fg_vk_tensor *query,const fg_vk_tensor *value,fg_error *err){if(!c||!tensor_range(key,0,10240u*4u)||!tensor_range(query,0,10240u*4u)||!tensor_range(value,0,2560u*4u)||!tensor_range(out,0,10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE gate dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={key,query,value,out};return dispatch(c,&c->ple_gate,bindings,NULL,4u,1,1,err);}
fg_status fg_vk_ple_gate_prefill(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *key,const fg_vk_tensor *query,const fg_vk_tensor *value,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(key,0,(uint64_t)tokens*10240u*4u)||!tensor_range(query,0,(uint64_t)tokens*10240u*4u)||!tensor_range(value,0,(uint64_t)tokens*2560u*4u)||!tensor_range(out,0,(uint64_t)tokens*10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE gate prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={key,query,value,out};return dispatch(c,&c->ple_gate_prefill,bindings,&tokens,tokens*4u,1,1,err);}

/* -------- GPU-timestamped dense Q8_0 kernel benchmark -------- */

static fg_status bench_record_dispatch_no_barrier(fg_vk_context *c,fg_vk_kernel *kernel,
    const fg_vk_tensor *w,const fg_vk_tensor *x,fg_vk_tensor *y,
    const void *push,uint32_t gx,uint32_t gy,fg_error *err)
{
    fg_status status=create_kernel(c,kernel,err);if(status!=FG_OK)return status;
    if(c->batch_set_count>=FG_VK_BATCH_MAX_SETS){fg_error_set(err,FG_ERR_LIMIT,"benchmark descriptor capacity exceeded");return FG_ERR_LIMIT;}
    VkDescriptorSet set=c->descriptor_sets[c->batch_set_count++];
    VkDescriptorBufferInfo info[3];VkWriteDescriptorSet write[3];
    const fg_vk_tensor *tensors[]={w,x,y};
    for(uint32_t i=0;i<3;i++){
        info[i]=(VkDescriptorBufferInfo){.buffer=tensors[i]->allocation->buffer,.offset=tensors[i]->offset,.range=tensors[i]->bytes};
        write[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info[i]};
    }
    vkUpdateDescriptorSets(c->device,3,write,0,NULL);
    /* NO barrier — caller manages dependencies via independent output tensors */
    vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);
    vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);
    vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);
    vkCmdDispatch(c->command,gx,gy,1);
    return FG_OK;
}

fg_status fg_vk_bench_dense_q8(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    fg_status drain=vk_pipeline_drain(c,err);if(drain!=FG_OK)return drain;
    /* Get timestamp period */
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(c->physical,&props);
    float ts_period=props.limits.timestampPeriod; /* nanoseconds per tick */
    uint32_t ts_valid=0;
    {uint32_t fc=0;vkGetPhysicalDeviceQueueFamilyProperties(c->physical,&fc,NULL);
     VkQueueFamilyProperties *fp=malloc(fc*sizeof(*fp));if(fp){vkGetPhysicalDeviceQueueFamilyProperties(c->physical,&fc,fp);ts_valid=fp[c->queue_family].timestampValidBits;free(fp);}}
    fprintf(stderr,"timestamp: period=%.2f ns, validBits=%u, device=%s\n",ts_period,ts_valid,c->device_name);
    if(!ts_valid){fg_error_set(err,FG_ERR_UNAVAILABLE,"GPU does not support timestamp queries");return FG_ERR_UNAVAILABLE;}

    /* Create a large descriptor pool and query pool for benchmarking */
    VkDescriptorPoolSize bps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=2048};
    VkDescriptorPoolCreateInfo bdp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,.maxSets=512,.poolSizeCount=1,.pPoolSizes=&bps};
    VkDescriptorPool saved_pool=c->descriptor_pool;
    VkResult vr=vkCreateDescriptorPool(c->device,&bdp,NULL,&c->descriptor_pool);
    if(vr!=VK_SUCCESS){c->descriptor_pool=saved_pool;return vk_error(err,"create bench descriptor pool",vr);}

    #define BENCH_N 10
    #define TS_PER_SHAPE (2+BENCH_N*2+BENCH_N*2) /* warmup boundary + A pairs + B pairs */
    struct {uint32_t in,out;const char *name;} shapes[]={
        {10240,320,"hc_down 10240→320"},{320,10240,"hc_up 320→10240"},
        {2560,640,"shexp_gate 2560→640"},{640,2560,"shexp_down 640→2560"},
        {2560,512,"qsa_attn_q 2560→512"},{2560,10240,"ple_key 2560→10240"},
    };
    uint32_t n_shapes=sizeof(shapes)/sizeof(shapes[0]);
    uint32_t max_ts=n_shapes*TS_PER_SHAPE+16;

    VkQueryPoolCreateInfo qpc={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=max_ts};
    VkQueryPool qpool=VK_NULL_HANDLE;
    vr=vkCreateQueryPool(c->device,&qpc,NULL,&qpool);
    if(vr!=VK_SUCCESS){vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);c->descriptor_pool=saved_pool;return vk_error(err,"create timestamp query pool",vr);}

    fprintf(stderr,"\n=== GPU-timestamped Q8_0 dense matvec benchmark ===\n");
    fprintf(stderr,"%-20s %7s │ %8s %8s %8s │ %8s %8s %8s │ %8s\n",
        "shape","wt MB","A:nobar","A:GB/s","A:rfl%","B:bar","B:GB/s","B:rfl%","C:stalone");
    fprintf(stderr,"─────────────────────────────────────────────────────────────────────────────────\n");

    fg_status status=FG_OK;
    for(uint32_t s=0;s<n_shapes&&status==FG_OK;s++){
        uint32_t in_dim=shapes[s].in,out_dim=shapes[s].out;
        uint32_t blocks=in_dim/32u;uint64_t row_bytes=(uint64_t)blocks*34u;
        uint64_t weight_bytes=row_bytes*out_dim;
        uint64_t total_data=weight_bytes+(uint64_t)in_dim*4u+(uint64_t)out_dim*4u;
        struct{uint32_t out_dim,n_tok,blocks,row_bytes;float scale;}push={out_dim,1,blocks,(uint32_t)row_bytes,1.0f};

        /* Create tensors: 1 weight, 1 input, BENCH_N independent outputs */
        fg_vk_tensor *w=NULL,*x=NULL,*y[BENCH_N]={0};
        status=fg_vk_tensor_create(c,weight_bytes,&w,err);
        if(status==FG_OK)status=fg_vk_tensor_create(c,(uint64_t)in_dim*4u,&x,err);
        for(uint32_t i=0;status==FG_OK&&i<BENCH_N;i++)
            status=fg_vk_tensor_create(c,(uint64_t)out_dim*4u,&y[i],err);
        if(status!=FG_OK){fprintf(stderr,"tensor alloc failed: %s\n",err->message);for(uint32_t i=0;i<BENCH_N;i++)fg_vk_tensor_destroy(y[i]);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);break;}
        memset(fg_vk_tensor_map(w),0x42,weight_bytes);
        float *xp=fg_vk_tensor_map(x);for(uint32_t i=0;i<in_dim;i++)xp[i]=1.0f/(float)(i+1);

        /* Warmup: 20 standalone dispatches */
        for(uint32_t i=0;status==FG_OK&&i<20;i++)
            status=fg_vk_dense_q8_0_f32(c,y[0],w,x,in_dim,out_dim,1u,1.0f,err);
        if(status!=FG_OK){fprintf(stderr,"warmup failed at shape %s: %s\n",shapes[s].name,err->message);}
        else{fprintf(stderr,"  warmup OK for %s\n",shapes[s].name);}

        /* --- MODE A: no-barrier dispatches with GPU timestamps --- */
        uint32_t ts_base_a=s*TS_PER_SHAPE;
        if(status==FG_OK){
            fprintf(stderr,"  Mode A: recording %u dispatches...\n",BENCH_N);
            vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);
            VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vr=vkBeginCommandBuffer(c->command,&begin);if(vr!=VK_SUCCESS){fprintf(stderr,"  beginCB failed: %d\n",(int)vr);status=FG_ERR_IO;break;}
            vkCmdResetQueryPool(c->command,qpool,ts_base_a,BENCH_N*2+2);
            /* Host→device barrier once at start */
            VkMemoryBarrier hd={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&hd,0,NULL,0,NULL);
            c->batch_depth=1;c->batch_set_count=0;c->batch_set_base=0;
            vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base_a); /* start marker */
            for(uint32_t i=0;status==FG_OK&&i<BENCH_N;i++){
                vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base_a+1+i*2);
                status=bench_record_dispatch_no_barrier(c,&c->dense,w,x,y[i],&push,out_dim,1,err);
                if(status!=FG_OK){fprintf(stderr,"  dispatch %u failed: %s\n",i,err->message);}
                vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base_a+2+i*2);
            }
            fprintf(stderr,"  Mode A: recorded, ending CB...\n");
            VkMemoryBarrier dh={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&dh,0,NULL,0,NULL);
            vr=vkEndCommandBuffer(c->command);if(vr!=VK_SUCCESS){fprintf(stderr,"  endCB failed: %d\n",(int)vr);status=FG_ERR_IO;}
            if(status==FG_OK){
                fprintf(stderr,"  Mode A: submitting...\n");
                VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};
                vr=vkQueueSubmit(c->queue,1,&submit,c->fence);
                if(vr!=VK_SUCCESS){fprintf(stderr,"Mode A submit failed: %d\n",(int)vr);status=FG_ERR_IO;}
            }
            if(status==FG_OK){fprintf(stderr,"  Mode A: waiting fence...\n");vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,5000000000ULL);if(vr!=VK_SUCCESS){fprintf(stderr,"Mode A fence: %d\n",(int)vr);status=FG_ERR_IO;}}
            c->batch_depth=0;c->batch_set_count=0;
        }

        /* --- MODE B: with-barrier dispatches with GPU timestamps --- */
        uint32_t ts_base_b=ts_base_a+BENCH_N*2+2;
        if(status==FG_OK){
            vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);
            VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(c->command,&begin);
            vkCmdResetQueryPool(c->command,qpool,ts_base_b,BENCH_N*2+2);
            VkMemoryBarrier hd={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&hd,0,NULL,0,NULL);
            c->batch_depth=1;c->batch_set_count=0;c->batch_set_base=0;
            vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base_b);
            for(uint32_t i=0;status==FG_OK&&i<BENCH_N;i++){
                vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base_b+1+i*2);
                /* Use dispatch() which inserts compute→compute barrier */
                const fg_vk_tensor *bindings[]={w,x,y[i%BENCH_N]};
                status=dispatch(c,&c->dense,bindings,&push,out_dim,1,1,err);
                vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base_b+2+i*2);
            }
            VkMemoryBarrier dh={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&dh,0,NULL,0,NULL);
            vkEndCommandBuffer(c->command);
            VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};
            vr=vkQueueSubmit(c->queue,1,&submit,c->fence);
            if(vr!=VK_SUCCESS){fprintf(stderr,"Mode B submit failed: %d\n",(int)vr);status=FG_ERR_IO;}
            if(status==FG_OK){vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,5000000000ULL);if(vr!=VK_SUCCESS){fprintf(stderr,"Mode B fence: %d\n",(int)vr);status=FG_ERR_IO;}}
            c->batch_depth=0;c->batch_set_count=0;
        }

        /* Read timestamps and compute results */
        uint64_t ts_a[BENCH_N*2+2],ts_b[BENCH_N*2+2];
        if(status==FG_OK){
            fprintf(stderr,"  reading Mode A timestamps...\n");
            vr=vkGetQueryPoolResults(c->device,qpool,ts_base_a,BENCH_N*2+2,sizeof(ts_a),ts_a,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
            if(vr!=VK_SUCCESS&&vr!=VK_NOT_READY){fprintf(stderr,"  Mode A query read: %d\n",(int)vr);status=FG_ERR_IO;}
        }
        if(status==FG_OK){
            fprintf(stderr,"  reading Mode B timestamps...\n");
            vr=vkGetQueryPoolResults(c->device,qpool,ts_base_b,BENCH_N*2+2,sizeof(ts_b),ts_b,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
            if(vr!=VK_SUCCESS&&vr!=VK_NOT_READY){fprintf(stderr,"  Mode B query read: %d\n",(int)vr);status=FG_ERR_IO;}
        }

        /* Mode C: standalone wall-clock (200 iterations) */
        double standalone_us=0;
        if(status==FG_OK){
            struct timespec t0,t1;uint32_t c_iters=200;
            clock_gettime(CLOCK_MONOTONIC,&t0);
            for(uint32_t i=0;status==FG_OK&&i<c_iters;i++)
                status=fg_vk_dense_q8_0_f32(c,y[0],w,x,in_dim,out_dim,1u,1.0f,err);
            clock_gettime(CLOCK_MONOTONIC,&t1);
            standalone_us=((double)(t1.tv_sec-t0.tv_sec)+(double)(t1.tv_nsec-t0.tv_nsec)*1e-9)*1e6/(double)c_iters;
        }

        if(status==FG_OK){
            /* Compute per-dispatch GPU time for Mode A (no barriers) */
            double a_total_ns=0;
            for(uint32_t i=0;i<BENCH_N;i++){
                uint64_t dt=ts_a[1+i*2+1]-ts_a[1+i*2];
                a_total_ns+=(double)dt*(double)ts_period;
            }
            double a_us=a_total_ns/1e3/(double)BENCH_N;
            double a_gbps=(double)total_data/(a_us*1e-6)/1e9;

            /* Compute per-dispatch GPU time for Mode B (with barriers) */
            double b_total_ns=0;
            for(uint32_t i=0;i<BENCH_N;i++){
                uint64_t dt=ts_b[1+i*2+1]-ts_b[1+i*2];
                b_total_ns+=(double)dt*(double)ts_period;
            }
            double b_us=b_total_ns/1e3/(double)BENCH_N;
            double b_gbps=(double)total_data/(b_us*1e-6)/1e9;

            fprintf(stderr,"%-20s %6.2f │ %7.1f %7.1f %7.1f%% │ %7.1f %7.1f %7.1f%% │ %7.1f\n",
                shapes[s].name,(double)weight_bytes/1e6,
                a_us,a_gbps,a_gbps/357.0*100.0,
                b_us,b_gbps,b_gbps/357.0*100.0,
                standalone_us);
        }

        for(uint32_t i=0;i<BENCH_N;i++)fg_vk_tensor_destroy(y[i]);
        fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);
    }
    #undef BENCH_N
    #undef TS_PER_SHAPE

    vkDestroyQueryPool(c->device,qpool,NULL);
    vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);
    c->descriptor_pool=saved_pool;
    return status;
}
fg_status fg_vk_gdn_project_decode(fg_vk_context *c,fg_vk_tensor *qkv,fg_vk_tensor *z,fg_vk_tensor *alpha,fg_vk_tensor *beta,const fg_vk_tensor *qkv_weight,const fg_vk_tensor *z_weight,const fg_vk_tensor *alpha_weight,const fg_vk_tensor *beta_weight,const fg_vk_tensor *hidden,fg_error *err){const uint32_t input_width=2560u,qkv_width=10240u,z_width=6144u,control_width=48u,blocks=input_width/FG_QK8_0;bool qkv_cooked=qkv_weight&&qkv_weight->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED,z_cooked=z_weight&&z_weight->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED;uint64_t generic_row=(uint64_t)blocks*FG_Q8_0_BLOCK_BYTES,cooked_quant_offset=fg_align_up_u64(FG_Q8_0_COOK_ROWS*(uint64_t)blocks*sizeof(uint16_t),FG_Q8_0_COOK_ALIGNMENT),cooked_tile=fg_align_up_u64(cooked_quant_offset+FG_Q8_0_COOK_ROWS*(uint64_t)blocks*FG_QK8_0,FG_Q8_0_COOK_ALIGNMENT),qkv_bytes=qkv_cooked?cooked_tile*((qkv_width+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS):generic_row*qkv_width,z_bytes=z_cooked?cooked_tile*((z_width+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS):generic_row*z_width,qkv_stride=qkv_cooked?cooked_tile:generic_row,z_stride=z_cooked?cooked_tile:generic_row;if(!c||!qkv_weight||!z_weight||(qkv_weight->format!=FG_VK_TENSOR_FORMAT_DEFAULT&&!qkv_cooked)||(z_weight->format!=FG_VK_TENSOR_FORMAT_DEFAULT&&!z_cooked)||qkv_stride>UINT32_MAX||z_stride>UINT32_MAX||!tensor_range(hidden,0,(uint64_t)input_width*4u)||!tensor_range(qkv_weight,0,qkv_bytes)||!tensor_range(z_weight,0,z_bytes)||!tensor_range(alpha_weight,0,(uint64_t)input_width*control_width*4u)||!tensor_range(beta_weight,0,(uint64_t)input_width*control_width*4u)||!tensor_range(qkv,0,(uint64_t)qkv_width*4u)||!tensor_range(z,0,(uint64_t)z_width*4u)||!tensor_range(alpha,0,(uint64_t)control_width*4u)||!tensor_range(beta,0,(uint64_t)control_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN sibling projection graph");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,layout_stride;float scale;}qkv_push={qkv_width,1u,blocks,(uint32_t)qkv_stride,1.0f},z_push={z_width,1u,blocks,(uint32_t)z_stride,1.0f};struct{uint32_t input_width,output_width,tokens;}control_push={input_width,control_width,1u};const fg_vk_tensor *qkv_bindings[]={qkv_weight,hidden,qkv},*z_bindings[]={z_weight,hidden,z},*alpha_bindings[]={alpha_weight,hidden,alpha},*beta_bindings[]={beta_weight,hidden,beta};fg_vk_kernel *qkv_kernel=qkv_cooked?&c->dense_cooked:&c->dense,*z_kernel=z_cooked?&c->dense_cooked:&c->dense;uint32_t qkv_groups=qkv_cooked?(qkv_width+3u)/4u:qkv_width,z_groups=z_cooked?(z_width+3u)/4u:z_width;fg_status status=dispatch(c,qkv_kernel,qkv_bindings,&qkv_push,qkv_groups,1,1,err);if(status==FG_OK)status=dispatch_impl(c,z_kernel,z_bindings,&z_push,z_groups,1,1,false,err);if(status==FG_OK)status=dispatch_impl(c,&c->dense_f32,alpha_bindings,&control_push,control_width,1,1,false,err);if(status==FG_OK)status=dispatch_impl(c,&c->dense_f32,beta_bindings,&control_push,control_width,1,1,false,err);return status;}
fg_status fg_vk_gdn_recurrent_algebraic(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,float epsilon,fg_error *err){uint64_t key_width=(uint64_t)key_heads*head_dim,value_width=(uint64_t)value_heads*head_dim,state_values=(uint64_t)value_heads*head_dim*head_dim;if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tensor_range(qkv,0,(2u*key_width+value_width)*4u)||!tensor_range(z,0,value_width*4u)||!tensor_range(alpha,0,(uint64_t)value_heads*4u)||!tensor_range(beta,0,(uint64_t)value_heads*4u)||!tensor_range(a_log,0,(uint64_t)value_heads*4u)||!tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||!tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||!tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid algebraic GDN recurrent dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t value_heads,key_heads,head_dim;float epsilon;}push={value_heads,key_heads,head_dim,epsilon};const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};return dispatch(c,&c->gdn_recurrent_algebraic,bindings,&push,value_heads,1,1,err);}
fg_status fg_vk_ple_conv_decode(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *gated,const fg_vk_tensor *normalized,const fg_vk_tensor *weight,fg_error *err){if(!c||!tensor_range(gated,0,10240u*4u)||!tensor_range(normalized,0,10240u*4u)||!tensor_range(weight,0,10240u*4u*4u)||!tensor_range(state,0,10240u*9u*4u)||!tensor_range(out,0,10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE convolution dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gated,normalized,weight,state,out};return dispatch(c,&c->ple_conv,bindings,NULL,40u,1,1,err);}
fg_status fg_vk_ple_conv_prefill(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *gated,const fg_vk_tensor *normalized,const fg_vk_tensor *weight,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(gated,0,(uint64_t)tokens*10240u*4u)||!tensor_range(normalized,0,(uint64_t)tokens*10240u*4u)||!tensor_range(weight,0,10240u*4u*4u)||!tensor_range(state,0,10240u*9u*4u)||!tensor_range(out,0,(uint64_t)tokens*10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE convolution prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gated,normalized,weight,state,out};return dispatch(c,&c->ple_conv_prefill,bindings,&tokens,40u,1,1,err);}
fg_status fg_vk_ple_conv_decode_add(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *gated,const fg_vk_tensor *normalized,const fg_vk_tensor *weight,const fg_vk_tensor *hyper,fg_error *err){if(!c||!tensor_range(gated,0,10240u*4u)||!tensor_range(normalized,0,10240u*4u)||!tensor_range(weight,0,10240u*4u*4u)||!tensor_range(state,0,10240u*9u*4u)||!tensor_range(out,0,10240u*4u)||!tensor_range(hyper,0,10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE convolution add dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gated,normalized,weight,state,out,hyper};return dispatch(c,&c->ple_conv_add,bindings,NULL,40u,1,1,err);}
fg_status fg_vk_add_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *left,const fg_vk_tensor *right,uint32_t values,fg_error *err){if(!c||!values||!tensor_range(left,0,(uint64_t)values*4u)||!tensor_range(right,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid F32 add dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={left,right,out};return dispatch(c,&c->add,bindings,&values,(values+255u)/256u,1,1,err);}

fg_status fg_vk_apply_penalties(fg_vk_context *c,fg_vk_tensor *logits,
                                const fg_vk_tensor *counts,uint32_t values,
                                float presence_penalty,float frequency_penalty,
                                float repetition_penalty,fg_error *err){
    if(!c||!values||!tensor_on_context(c,logits)||!tensor_on_context(c,counts)||
       !tensor_range(logits,0,(uint64_t)values*4u)||
       !tensor_range(counts,0,(uint64_t)values*4u)||
       !isfinite(presence_penalty)||!isfinite(frequency_penalty)||
       !isfinite(repetition_penalty)||repetition_penalty<=0.0f){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid vocabulary penalty dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct { uint32_t values; float presence,frequency,repetition; } push={
        values,presence_penalty,frequency_penalty,repetition_penalty};
    const fg_vk_tensor *bindings[]={logits,counts};
    return dispatch(c,&c->apply_penalties,bindings,&push,(values+255u)/256u,1u,1u,err);
}
fg_status fg_vk_gdn_conv_decode(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *projection,const fg_vk_tensor *weight,uint32_t channels,fg_error *err){if(!c||!channels||!tensor_range(projection,0,(uint64_t)channels*4u)||!tensor_range(weight,0,(uint64_t)channels*4u*4u)||!tensor_range(state,0,(uint64_t)channels*4u*4u)||!tensor_range(out,0,(uint64_t)channels*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN convolution dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={projection,weight,state,out};return dispatch(c,&c->gdn_conv,bindings,&channels,(channels+255u)/256u,1,1,err);}
fg_status fg_vk_gdn_conv_prefill(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *projection,const fg_vk_tensor *weight,uint32_t channels,uint32_t tokens,fg_error *err){if(!c||!channels||!tokens||!tensor_range(projection,0,(uint64_t)channels*tokens*4u)||!tensor_range(weight,0,(uint64_t)channels*4u*4u)||!tensor_range(state,0,(uint64_t)channels*4u*4u)||!tensor_range(out,0,(uint64_t)channels*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN convolution prefill dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t channels,tokens;}push={channels,tokens};const fg_vk_tensor *bindings[]={projection,weight,state,out};return dispatch(c,&c->gdn_conv_prefill,bindings,&push,(channels+255u)/256u,1,1,err);}
fg_status fg_vk_gdn_recurrent_decode(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,float epsilon,fg_error *err){uint64_t key_width=(uint64_t)key_heads*head_dim,value_width=(uint64_t)value_heads*head_dim,state_values=(uint64_t)value_heads*head_dim*head_dim;if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tensor_range(qkv,0,(2u*key_width+value_width)*4u)||!tensor_range(z,0,value_width*4u)||!tensor_range(alpha,0,(uint64_t)value_heads*4u)||!tensor_range(beta,0,(uint64_t)value_heads*4u)||!tensor_range(a_log,0,(uint64_t)value_heads*4u)||!tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||!tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||!tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN recurrent dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t value_heads,key_heads,head_dim;float epsilon;}push={value_heads,key_heads,head_dim,epsilon};const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};return dispatch(c,&c->gdn_recurrent,bindings,&push,value_heads,1,1,err);}
fg_status fg_vk_gdn_recurrent_prefill(fg_vk_context *c,fg_vk_tensor *out,
    fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,
    const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,
    const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,
    uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,uint32_t tokens,
    float epsilon,fg_error *err){
    uint64_t key_width=(uint64_t)key_heads*head_dim,
        value_width=(uint64_t)value_heads*head_dim,qkv_width=2u*key_width+value_width,
        state_values=(uint64_t)value_heads*head_dim*head_dim;
    if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tokens||
       !tensor_range(qkv,0,qkv_width*tokens*4u)||!tensor_range(z,0,value_width*tokens*4u)||
       !tensor_range(alpha,0,(uint64_t)value_heads*tokens*4u)||
       !tensor_range(beta,0,(uint64_t)value_heads*tokens*4u)||
       !tensor_range(a_log,0,(uint64_t)value_heads*4u)||
       !tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||
       !tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||
       !tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*tokens*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN recurrent prefill dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t value_heads,key_heads,head_dim,tokens;float epsilon;}
        push={value_heads,key_heads,head_dim,tokens,epsilon};
    const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};
    fg_vk_kernel *kernel=&c->gdn_recurrent_prefill;
    if(c->small_n&&tokens<=4u){
        kernel=&c->small_gdn;
        if(!kernel->file)*kernel=(fg_vk_kernel){.file="fg_gdn_small.spv",
            .bindings=9,.push_bytes=20};
    }
    return dispatch(c,kernel,bindings,&push,value_heads,1u,1u,err);
}
fg_status fg_vk_gdn_recurrent_prefill_chunked(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_decay,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t tokens,float epsilon,fg_error *err){
    const uint64_t qkv_values=(uint64_t)tokens*10240u;
    const uint64_t value_values=(uint64_t)tokens*6144u;
    const uint64_t control_values=(uint64_t)tokens*48u;
    const uint64_t state_values=48u*128u*128u;
    if(!c||!tokens||tokens>FG_VK_GDN_CHUNKED_PREFILL_MAX_TOKENS||
       !tensor_range(qkv,0,qkv_values*4u)||
       !tensor_range(z,0,value_values*4u)||
       !tensor_range(alpha,0,control_values*4u)||
       !tensor_range(beta,0,control_values*4u)||
       !tensor_range(a_decay,0,48u*4u)||
       !tensor_range(dt_bias,0,48u*4u)||
       !tensor_range(norm_weight,0,128u*4u)||
       !tensor_range(state,0,state_values*4u)||
       !tensor_range(out,0,value_values*4u)||
       tensor_ranges_overlap(qkv,z)||tensor_ranges_overlap(qkv,alpha)||
       tensor_ranges_overlap(qkv,beta)||tensor_ranges_overlap(qkv,a_decay)||
       tensor_ranges_overlap(qkv,dt_bias)||
       tensor_ranges_overlap(qkv,norm_weight)||
       tensor_ranges_overlap(qkv,state)||tensor_ranges_overlap(qkv,out)||
       tensor_ranges_overlap(out,z)||tensor_ranges_overlap(out,alpha)||
       tensor_ranges_overlap(out,beta)||tensor_ranges_overlap(out,a_decay)||
       tensor_ranges_overlap(out,dt_bias)||
       tensor_ranges_overlap(out,norm_weight)||
       tensor_ranges_overlap(out,state)){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid chunked GDN recurrent prefill dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t tokens;float epsilon;}norm_push={tokens,epsilon};
    const fg_vk_tensor *norm_bindings[]={qkv};
    /* dispatch() inserts shader-write to shader-read/write barriers between stages. */
    fg_status status=dispatch(c,&c->gdn_prefill_qk_norm,norm_bindings,
                              &norm_push,16u,tokens,1u,err);
    struct{uint32_t tokens;}recurrence_push={tokens};
    const fg_vk_tensor *recurrence_bindings[]={
        qkv,alpha,beta,a_decay,dt_bias,state,out
    };
    if(status==FG_OK)status=dispatch(c,&c->gdn_prefill_recurrence,
        recurrence_bindings,&recurrence_push,48u,1u,128u,err);
    const fg_vk_tensor *output_bindings[]={out,z,norm_weight};
    if(status==FG_OK)status=dispatch(c,&c->gdn_prefill_output,output_bindings,
        &norm_push,48u,tokens,1u,err);
    return status;
}
fg_status fg_vk_qsa_prepare(fg_vk_context *c,fg_vk_tensor *query,fg_vk_tensor *gate,fg_vk_tensor *key,const fg_vk_tensor *raw_qg,const fg_vk_tensor *raw_key,const fg_vk_tensor *qnorm,const fg_vk_tensor *knorm,const fg_vk_tensor *position,fg_error *err){if(!c||!tensor_range(raw_qg,0,12288u*4u)||!tensor_range(raw_key,0,512u*4u)||!tensor_range(qnorm,0,256u*4u)||!tensor_range(knorm,0,256u*4u)||!tensor_range(position,0,3u*4u)||!tensor_range(query,0,6144u*4u)||!tensor_range(gate,0,6144u*4u)||!tensor_range(key,0,512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prepare dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_qg,raw_key,qnorm,knorm,position,query,gate,key};return dispatch(c,&c->qsa_prepare,bindings,NULL,24u,1,1,err);}
fg_status fg_vk_qsa_prepare_prefill(fg_vk_context *c,fg_vk_tensor *query,fg_vk_tensor *gate,fg_vk_tensor *key,const fg_vk_tensor *raw_qg,const fg_vk_tensor *raw_key,const fg_vk_tensor *qnorm,const fg_vk_tensor *knorm,const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(raw_qg,0,(uint64_t)tokens*12288u*4u)||!tensor_range(raw_key,0,(uint64_t)tokens*512u*4u)||!tensor_range(qnorm,0,256u*4u)||!tensor_range(knorm,0,256u*4u)||!tensor_range(positions,0,(uint64_t)tokens*3u*4u)||!tensor_range(query,0,(uint64_t)tokens*6144u*4u)||!tensor_range(gate,0,(uint64_t)tokens*6144u*4u)||!tensor_range(key,0,(uint64_t)tokens*512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prepare prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_qg,raw_key,qnorm,knorm,positions,query,gate,key};return dispatch(c,&c->qsa_prepare_prefill,bindings,&tokens,tokens*24u,1,1,err);}
fg_status fg_vk_qsa_index_prepare(fg_vk_context *c,fg_vk_tensor *query,const fg_vk_tensor *raw_query,const fg_vk_tensor *norm,const fg_vk_tensor *position,fg_error *err){if(!c||!tensor_range(raw_query,0,512u*4u)||!tensor_range(norm,0,128u*4u)||!tensor_range(position,0,3u*4u)||!tensor_range(query,0,512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA index prepare dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_query,norm,position,query};return dispatch(c,&c->qsa_index_prepare,bindings,NULL,4u,1,1,err);}
fg_status fg_vk_qsa_index_prepare_prefill(fg_vk_context *c,fg_vk_tensor *query,const fg_vk_tensor *raw_query,const fg_vk_tensor *norm,const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(raw_query,0,(uint64_t)tokens*512u*4u)||!tensor_range(norm,0,128u*4u)||!tensor_range(positions,0,(uint64_t)tokens*3u*4u)||!tensor_range(query,0,(uint64_t)tokens*512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA index prepare prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_query,norm,positions,query};return dispatch(c,&c->qsa_index_prepare_prefill,bindings,&tokens,tokens*4u,1,1,err);}
static fg_status qsa_record_commit_dispatch(
    fg_vk_context *c,fg_vk_tensor *records,fg_vk_tensor *index_history,
    const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,
    const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,
    uint32_t record_layer_slot,uint32_t index_layer_slot,uint32_t index_token,
    uint32_t index_capacity,uint32_t hot_slot,
    uint32_t hot_capacity,fg_error *err){
    uint64_t record_end=((uint64_t)record_layer_slot*hot_capacity+hot_slot+1u)*
        FG_Q38_QSA_TOKEN_RECORD_BYTES;
    uint64_t index_end=((uint64_t)index_layer_slot*index_capacity+index_token+1u)*
        FG_Q38_QSA_INDEX_KEY_BYTES;
    if(!c||!index_capacity||!hot_capacity||index_token>=index_capacity||
       hot_slot>=hot_capacity||!tensor_range(key_q8,0,FG_Q38_QSA_KEY_BYTES)||
       !tensor_range(value_q8,0,FG_Q38_QSA_VALUE_BYTES)||
       !tensor_range(index_key_q8,0,FG_Q38_QSA_INDEX_KEY_BYTES)||
       !tensor_range(position,0,FG_Q38_QSA_POSITION_BYTES)||
       !tensor_range(records,0,record_end)||
       !tensor_range(index_history,0,index_end)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tiered QSA record commit");
        return FG_ERR_ARGUMENT;
    }
    struct{
        uint32_t record_layer_slot,index_layer_slot,index_capacity,index_token;
        uint32_t hot_slot,hot_capacity;
    } push={record_layer_slot,index_layer_slot,index_capacity,index_token,hot_slot,
            hot_capacity};
    const fg_vk_tensor *bindings[]={
        key_q8,value_q8,index_key_q8,position,records,index_history
    };
    return dispatch(c,&c->qsa_record_commit,bindings,&push,
                    (FG_Q38_QSA_TOKEN_RECORD_BYTES/4u+255u)/256u,1,1,err);
}

fg_status fg_vk_qsa_record_commit_tiered(
    fg_vk_context *c,fg_vk_tensor *records,fg_vk_tensor *index_history,
    const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,
    const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,
    uint32_t layer_slot,uint32_t token,uint32_t index_capacity,
    uint32_t hot_slot,uint32_t hot_capacity,fg_error *err){
    return qsa_record_commit_dispatch(c,records,index_history,key_q8,value_q8,
        index_key_q8,position,layer_slot,layer_slot,token,index_capacity,
        hot_slot,hot_capacity,err);
}

fg_status fg_vk_qsa_record_commit_segmented(
    fg_vk_context *c,fg_vk_tensor *records,fg_vk_tensor *index_segment,
    const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,
    const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,
    uint32_t layer_slot,uint32_t token,uint32_t index_token,
    uint32_t index_capacity,uint32_t hot_slot,uint32_t hot_capacity,
    fg_error *err){
    (void)token;
    return qsa_record_commit_dispatch(c,records,index_segment,key_q8,value_q8,
        index_key_q8,position,layer_slot,0u,index_token,index_capacity,
        hot_slot,hot_capacity,err);
}

fg_status fg_vk_qsa_record_commit(
    fg_vk_context *c,fg_vk_tensor *records,fg_vk_tensor *index_history,
    const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,
    const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,
    uint32_t layer_slot,uint32_t token,uint32_t capacity,fg_error *err){
    return fg_vk_qsa_record_commit_tiered(c,records,index_history,key_q8,value_q8,
        index_key_q8,position,layer_slot,token,capacity,token,capacity,err);
}
fg_status fg_vk_qsa_record_gather(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *records,const fg_vk_tensor *block_ids,uint32_t layer_slot,uint32_t capacity,uint32_t block_count,uint32_t tail_start,uint32_t tail_count,fg_error *err){uint32_t selected=block_count*FG_Q38_QSA_COMPRESS_RATIO+tail_count;uint64_t record_end=(uint64_t)(layer_slot+1u)*capacity*FG_Q38_QSA_TOKEN_RECORD_BYTES;if(!c||!capacity||(!block_count&&!tail_count)||block_count>FG_Q38_INDEX_BUDGET/FG_Q38_QSA_COMPRESS_RATIO||tail_count>=FG_Q38_QSA_COMPRESS_RATIO||tail_start>capacity||tail_count>capacity-tail_start||selected>FG_Q38_INDEX_BUDGET+FG_Q38_QSA_COMPRESS_RATIO-1u||!tensor_range(records,0,record_end)||(block_count&&!tensor_range(block_ids,0,(uint64_t)block_count*4u))||!tensor_range(out,0,(uint64_t)selected*FG_Q38_QSA_TOKEN_RECORD_BYTES)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA record gather");return FG_ERR_ARGUMENT;}struct{uint32_t layer_slot,capacity,block_count,tail_start,tail_count;}push={layer_slot,capacity,block_count,tail_start,tail_count};const fg_vk_tensor *bindings[]={records,block_ids,out};uint64_t words=(uint64_t)selected*(FG_Q38_QSA_TOKEN_RECORD_BYTES/4u);return dispatch(c,&c->qsa_record_gather,bindings,&push,(uint32_t)((words+255u)/256u),1,1,err);}
fg_status fg_vk_qsa_index_score_segment(
    fg_vk_context *c,fg_vk_tensor *scores,fg_vk_tensor *ids,
    const fg_vk_tensor *query,const fg_vk_tensor *keys,const fg_vk_tensor *norm,
    const fg_vk_tensor *positions,uint32_t tokens,uint32_t block_base,
    fg_error *err){
    uint32_t blocks=tokens/4u;
    if(!c||!blocks||block_base>UINT32_MAX-blocks||
       !tensor_range(query,0,512u*4u)||
       !tensor_range(keys,0,(uint64_t)tokens*FG_Q38_QSA_INDEX_KEY_BYTES)||
       !tensor_range(norm,0,128u*4u)||
       !tensor_range(positions,0,(uint64_t)tokens*3u*4u)||
       !tensor_range(scores,0,(uint64_t)blocks*4u)||
       !tensor_range(ids,0,(uint64_t)blocks*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid segmented QSA index score dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t tokens,blocks,block_base,score_stride,output_base,first_visible,query_count;}
        push={tokens,blocks,block_base,blocks,0u,block_base*4u+tokens,1u};
    const fg_vk_tensor *bindings[]={query,keys,norm,positions,scores,ids};
    return dispatch(c,&c->qsa_score,bindings,&push,blocks,1,1,err);
}

fg_status fg_vk_qsa_index_score_batch(
    fg_vk_context *c,fg_vk_tensor *scores,fg_vk_tensor *ids,
    const fg_vk_tensor *queries,const fg_vk_tensor *keys,const fg_vk_tensor *norm,
    const fg_vk_tensor *positions,uint32_t tokens,uint32_t block_base,
    uint32_t score_stride,uint32_t first_visible,uint32_t query_count,fg_error *err){
    uint32_t blocks=tokens/4u;
    uint64_t entries=(uint64_t)query_count*score_stride;
    if(!c||!blocks||!query_count||!first_visible||
       first_visible>UINT32_MAX-query_count||block_base>score_stride||
       blocks>score_stride-block_base||
       !tensor_range(queries,0,(uint64_t)query_count*512u*4u)||
       !tensor_range(keys,0,(uint64_t)tokens*FG_Q38_QSA_INDEX_KEY_BYTES)||
       !tensor_range(norm,0,128u*4u)||
       !tensor_range(positions,0,(uint64_t)tokens*FG_Q38_QSA_POSITION_BYTES)||
       !tensor_range(scores,0,entries*4u)||!tensor_range(ids,0,entries*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid causal QSA score tile");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t tokens,blocks,block_base,score_stride,output_base,first_visible,query_count;}
        push={tokens,blocks,block_base,score_stride,block_base,first_visible,query_count};
    const fg_vk_tensor *bindings[]={queries,keys,norm,positions,scores,ids};
    /* One workgroup per block: the kernel scores every query of the dispatch
     * against a key transformed once, so the grid has no query dimension. */
    return dispatch(c,&c->qsa_score,bindings,&push,blocks,1u,1u,err);
}

fg_status fg_vk_qsa_index_score(
    fg_vk_context *c,fg_vk_tensor *scores,fg_vk_tensor *ids,
    const fg_vk_tensor *query,const fg_vk_tensor *keys,const fg_vk_tensor *norm,
    const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){
    return fg_vk_qsa_index_score_segment(c,scores,ids,query,keys,norm,positions,
                                         tokens,0u,err);
}
fg_status fg_vk_qsa_attention(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *records,const fg_vk_tensor *query,const fg_vk_tensor *gate,uint32_t selected_count,fg_error *err){if(!c||!selected_count||selected_count>2051u||!tensor_range(records,0,(uint64_t)selected_count*FG_Q38_QSA_TOKEN_RECORD_BYTES)||!tensor_range(query,0,6144u*4u)||!tensor_range(gate,0,6144u*4u)||!tensor_range(output,0,6144u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA attention dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={records,query,gate,output};return dispatch(c,&c->qsa_attention,bindings,&selected_count,24u,1,1,err);}
fg_status fg_vk_qsa_attention_split(fg_vk_context *c,fg_vk_tensor *partials,const fg_vk_tensor *records,const fg_vk_tensor *query,uint32_t selected_count,uint32_t splits,fg_error *err){if(!c||!selected_count||selected_count>2051u||!splits||splits>8u||!partials||!tensor_range(records,0,(uint64_t)selected_count*FG_Q38_QSA_TOKEN_RECORD_BYTES)||!tensor_range(query,0,6144u*4u)||!tensor_range(partials,0,(uint64_t)splits*24u*258u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid split QSA attention dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t selected_count,splits,split_index;}push={selected_count,splits,0u};const fg_vk_tensor *bindings[]={records,query,partials};return dispatch(c,&c->qsa_attention_split,bindings,&push,24u,splits,1u,err);}
fg_status fg_vk_qsa_decode_attention_split(fg_vk_context *c,fg_vk_tensor *partials,const fg_vk_tensor *records,const fg_vk_tensor *query,uint32_t selected_count,uint32_t splits,fg_error *err){
    if(!c||!selected_count||selected_count>2051u||!splits||splits>8u||!partials||
       !tensor_range(records,0,(uint64_t)selected_count*FG_Q38_QSA_TOKEN_RECORD_BYTES)||
       !tensor_range(query,0,6144u*4u)||
       !tensor_range(partials,0,(uint64_t)splits*24u*258u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid decode split QSA attention dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t selected_count,splits;}push={selected_count,splits};
    const fg_vk_tensor *bindings[]={records,query,partials};
    return dispatch(c,&c->qsa_decode_attention_split,bindings,&push,2u,splits,1u,err);
}
fg_status fg_vk_qsa_attention_merge(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *partials,const fg_vk_tensor *gate,uint32_t splits,fg_error *err){if(!c||!output||!splits||splits>8u||!tensor_range(gate,0,6144u*4u)||!tensor_range(output,0,6144u*4u)||!tensor_range(partials,0,(uint64_t)splits*24u*258u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid merge QSA attention dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t splits;}push={splits};const fg_vk_tensor *bindings[]={partials,gate,output};return dispatch(c,&c->qsa_attention_merge,bindings,&push,24u,1u,1u,err);}
fg_status fg_vk_qsa_attention_split_batch(fg_vk_context *c,fg_vk_tensor *partials,const fg_vk_tensor *records,const fg_vk_tensor *query,const fg_vk_tensor *counts,uint32_t query_count,uint32_t record_stride,uint32_t query_stride,uint32_t splits,fg_error *err){
    if(!c||!query_count||query_count>512u||!splits||splits>8u||
       !record_stride||record_stride>FG_VK_QSA_SELECTED_TOKENS||
       query_stride<24u*FG_Q38_ATTN_HEAD_WIDTH||
       !tensor_range(partials,0,(uint64_t)query_count*24u*splits*258u*4u)||
       !tensor_range(records,0,(uint64_t)query_count*record_stride*FG_Q38_QSA_TOKEN_RECORD_BYTES)||
       !tensor_range(query,0,(uint64_t)query_count*query_stride*4u)||
       !tensor_range(counts,0,(uint64_t)query_count*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched split QSA attention dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t splits,query_count,record_stride,query_stride;}push={splits,query_count,record_stride,query_stride};
    const fg_vk_tensor *bindings[]={records,query,partials,counts};
    /* One workgroup per (query, KV head, split): its subgroups cover the twelve
     * query heads of the KV head and share one decoded record tile. */
    return dispatch(c,&c->qsa_attention_split_batch,bindings,&push,query_count,2u,splits,err);
}
fg_status fg_vk_qsa_attention_merge_batch(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *partials,const fg_vk_tensor *gate,uint32_t query_count,uint32_t query_stride,uint32_t splits,fg_error *err){
    if(!c||!query_count||query_count>512u||!splits||splits>8u||
       !query_stride||
       !tensor_range(partials,0,(uint64_t)query_count*24u*splits*258u*4u)||
       !tensor_range(gate,0,(uint64_t)query_count*query_stride*4u)||
       !tensor_range(output,0,(uint64_t)query_count*query_stride*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched merge QSA attention dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t splits,query_count,query_stride;}push={splits,query_count,query_stride};
    const fg_vk_tensor *bindings[]={partials,gate,output};
    return dispatch(c,&c->qsa_attention_merge_batch,bindings,&push,24u,query_count,1u,err);
}
fg_status fg_vk_qsa_record_gather_batch(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *records,const fg_vk_tensor *slots,uint32_t query_count,uint32_t slot_stride,uint32_t record_stride,uint32_t capacity,fg_error *err){
    const uint32_t record_words=FG_Q38_QSA_TOKEN_RECORD_BYTES/sizeof(uint32_t);
    uint64_t words=(uint64_t)record_stride*record_words;
    if(!c||!query_count||query_count>512u||!slot_stride||
       slot_stride>FG_VK_QSA_SELECTED_BLOCKS||!record_stride||
       record_stride>FG_VK_QSA_SELECTED_TOKENS||!capacity||capacity%FG_Q38_QSA_COMPRESS_RATIO||
       !tensor_range(records,0,(uint64_t)capacity*FG_Q38_QSA_TOKEN_RECORD_BYTES)||
       !tensor_range(slots,0,(uint64_t)query_count*slot_stride*4u)||
       !tensor_range(output,0,(uint64_t)query_count*record_stride*FG_Q38_QSA_TOKEN_RECORD_BYTES)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched QSA record gather dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t capacity,query_count,slot_stride,record_stride,record_words;}push={capacity,query_count,slot_stride,record_stride,record_words};
    const fg_vk_tensor *bindings[]={records,slots,output};
    return dispatch(c,&c->qsa_record_gather_batch,bindings,&push,(uint32_t)((words+255u)/256u),query_count,1u,err);
}
fg_status fg_vk_qsa_select_resolve(fg_vk_context *c,fg_vk_tensor *resolved,
    fg_vk_tensor *flags,const fg_vk_tensor *selected_ids,
    const fg_vk_tensor *slot_table,uint32_t count,uint32_t block_limit,
    uint32_t layer_base,fg_error *err){
    uint32_t groups=(count+255u)/256u;
    if(!c||!count||count>2u*256u||
       !tensor_range(selected_ids,0,(uint64_t)count*4u)||
       !tensor_range(resolved,0,(uint64_t)count*4u)||
       !tensor_range(flags,0,(uint64_t)groups*4u)||
       !tensor_range(slot_table,(uint64_t)layer_base*4u,(uint64_t)block_limit*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA selection resolve dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t count,block_limit,layer_base;}push={count,block_limit,layer_base};
    const fg_vk_tensor *bindings[]={selected_ids,slot_table,resolved,flags};
    return dispatch(c,&c->qsa_select_resolve,bindings,&push,groups,1,1,err);
}
fg_status fg_vk_qsa_select_merge(fg_vk_context *c,fg_vk_tensor *scores_0,fg_vk_tensor *ids_0,fg_vk_tensor *scores_1,fg_vk_tensor *ids_1,fg_vk_tensor *result_ids,uint32_t count,uint32_t input_stride,uint32_t query_count,fg_error *err){
    if(!c||count<1u||count>input_stride||!query_count||query_count>512u||
       !tensor_range(scores_0,0,(uint64_t)query_count*input_stride*4u)||
       !tensor_range(ids_0,0,(uint64_t)query_count*input_stride*4u)||
       !tensor_range(scores_1,0,(uint64_t)query_count*input_stride*4u)||
       !tensor_range(ids_1,0,(uint64_t)query_count*input_stride*4u)||
       !tensor_range(result_ids,0,(uint64_t)query_count*FG_VK_QSA_SELECTED_BLOCKS*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched QSA top-k merge dispatch");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_tensor *scores[2]={scores_0,scores_1},*ids[2]={ids_0,ids_1};
    uint32_t side=0u,stride=input_stride;
    while(count>FG_VK_QSA_SELECTED_BLOCKS){
        uint32_t groups=(count+4095u)/4096u,produced=groups*512u;
        bool last=count<=4096u;
        const fg_vk_tensor *bindings[]={scores[side],ids[side],scores[side^1u],last?result_ids:ids[side^1u]};
        struct{uint32_t count,input_stride,output_stride,query_tokens;}push={count,stride,last?512u:produced,query_count};
        fg_status status=dispatch(c,&c->qsa_resident_merge,bindings,&push,groups,query_count,1u,err);
        if(status!=FG_OK)return status;
        side^=1u;count=produced;stride=produced;
    }
    return FG_OK;
}

static bool qsa_resident_segment_ranges(
    uint32_t capacity,uint32_t segment_capacity,const fg_vk_tensor *segment_0,
    const fg_vk_tensor *segment_1,uint64_t bytes_per_token){
    if(!capacity||!segment_capacity||capacity>segment_capacity*2u)return false;
    uint32_t first=capacity<segment_capacity?capacity:segment_capacity;
    uint32_t second=capacity-first;
    return tensor_range(segment_0,0,(uint64_t)first*bytes_per_token)&&
        (!second||tensor_range(segment_1,0,(uint64_t)second*bytes_per_token));
}

fg_status fg_vk_qsa_resident_record_commit(
    fg_vk_context *c,fg_vk_tensor *record_0,fg_vk_tensor *record_1,
    fg_vk_tensor *index_0,fg_vk_tensor *index_1,const fg_vk_tensor *keys,
    const fg_vk_tensor *values,const fg_vk_tensor *index_keys,
    const fg_vk_tensor *positions,uint32_t first_token,uint32_t token_count,
    uint32_t capacity,uint32_t segment_capacity,fg_error *err){
    if(!c||!token_count||first_token>capacity||
       token_count>capacity-first_token||
       !tensor_range(keys,0,(uint64_t)token_count*FG_Q38_QSA_KEY_BYTES)||
       !tensor_range(values,0,(uint64_t)token_count*FG_Q38_QSA_VALUE_BYTES)||
       !tensor_range(index_keys,0,(uint64_t)token_count*
                     FG_Q38_QSA_INDEX_KEY_BYTES)||
       !tensor_range(positions,0,(uint64_t)token_count*
                     FG_Q38_QSA_POSITION_BYTES)||
       !qsa_resident_segment_ranges(capacity,segment_capacity,record_0,record_1,
                                    FG_Q38_QSA_TOKEN_RECORD_BYTES)||
       !qsa_resident_segment_ranges(capacity,segment_capacity,index_0,index_1,
                                    FG_Q38_QSA_INDEX_KEY_BYTES)){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid resident QSA record commit dispatch");
        return FG_ERR_ARGUMENT;
    }
    if(!record_1)record_1=record_0;
    if(!index_1)index_1=index_0;
    struct{uint32_t first_token,token_count,capacity,segment_capacity;}push={
        first_token,token_count,capacity,segment_capacity
    };
    const fg_vk_tensor *bindings[]={
        keys,values,index_keys,positions,record_0,record_1,index_0,index_1
    };
    uint64_t words=(uint64_t)token_count*
        (FG_Q38_QSA_TOKEN_RECORD_BYTES/sizeof(uint32_t));
    return dispatch(c,&c->qsa_resident_commit,bindings,&push,
                    (uint32_t)((words+255u)/256u),1u,1u,err);
}

fg_status fg_vk_qsa_resident_select(
    fg_vk_context *c,fg_vk_tensor *scores_0,fg_vk_tensor *ids_0,
    fg_vk_tensor *scores_1,fg_vk_tensor *ids_1,const fg_vk_tensor *query,
    const fg_vk_tensor *index_0,const fg_vk_tensor *index_1,
    const fg_vk_tensor *key_norm,const fg_vk_tensor *positions,
    uint32_t first_token,uint32_t query_tokens,uint32_t capacity,
    uint32_t segment_capacity,uint32_t *final_side,fg_error *err){
    uint32_t visible=first_token+query_tokens;
    uint32_t complete_blocks=visible/FG_Q38_QSA_COMPRESS_RATIO;
    uint32_t groups=(complete_blocks+4095u)/4096u;
    if(!groups)groups=1u;
    uint32_t candidates=groups*512u;
    uint64_t entries=(uint64_t)query_tokens*candidates;
    if(!c||!query_tokens||!final_side||first_token>capacity||
       query_tokens>capacity-first_token||groups>16u||
       !tensor_range(query,0,(uint64_t)query_tokens*512u*4u)||
       !tensor_range(key_norm,0,128u*4u)||
       !tensor_range(positions,0,(uint64_t)capacity*
                     FG_Q38_QSA_POSITION_BYTES)||
       !qsa_resident_segment_ranges(capacity,segment_capacity,index_0,index_1,
                                    FG_Q38_QSA_INDEX_KEY_BYTES)||
       !tensor_range(scores_0,0,entries*4u)||
       !tensor_range(ids_0,0,entries*4u)||
       !tensor_range(scores_1,0,entries*4u)||
       !tensor_range(ids_1,0,entries*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid resident QSA hierarchical selection dispatch");
        return FG_ERR_ARGUMENT;
    }
    if(!index_1)index_1=index_0;
    struct{
        uint32_t first_token,query_tokens,capacity,segment_capacity,group_count;
    }initial={first_token,query_tokens,capacity,segment_capacity,groups};
    const fg_vk_tensor *initial_bindings[]={
        query,index_0,index_1,key_norm,positions,scores_0,ids_0
    };
    fg_status status=dispatch(c,&c->qsa_resident_select,initial_bindings,
                              &initial,groups,query_tokens,1u,err);
    fg_vk_tensor *scores[]={scores_0,scores_1};
    fg_vk_tensor *ids[]={ids_0,ids_1};
    uint32_t side=0u,count=candidates,stride=candidates;
    while(status==FG_OK&&count>512u){
        uint32_t output_groups=(count+4095u)/4096u;
        uint32_t output_count=output_groups*512u;
        struct{
            uint32_t count,input_stride,output_stride,query_tokens;
        }merge={count,stride,output_count,query_tokens};
        const fg_vk_tensor *merge_bindings[]={
            scores[side],ids[side],scores[side^1u],ids[side^1u]
        };
        status=dispatch(c,&c->qsa_resident_merge,merge_bindings,&merge,
                        output_groups,query_tokens,1u,err);
        side^=1u;count=output_count;stride=output_count;
    }
    if(status==FG_OK)*final_side=side;
    return status;
}

fg_status fg_vk_qsa_resident_attention(
    fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *record_0,
    const fg_vk_tensor *record_1,const fg_vk_tensor *selected,
    const fg_vk_tensor *query,const fg_vk_tensor *gate,uint32_t first_token,
    uint32_t query_tokens,uint32_t capacity,uint32_t segment_capacity,
    uint32_t selected_stride,fg_error *err){
    if(!c||!query_tokens||selected_stride!=512u||first_token>capacity||
       query_tokens>capacity-first_token||
       !qsa_resident_segment_ranges(capacity,segment_capacity,record_0,record_1,
                                    FG_Q38_QSA_TOKEN_RECORD_BYTES)||
       !tensor_range(selected,0,(uint64_t)query_tokens*selected_stride*4u)||
       !tensor_range(query,0,(uint64_t)query_tokens*6144u*4u)||
       !tensor_range(gate,0,(uint64_t)query_tokens*6144u*4u)||
       !tensor_range(output,0,(uint64_t)query_tokens*6144u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "invalid resident QSA direct attention dispatch");
        return FG_ERR_ARGUMENT;
    }
    if(!record_1)record_1=record_0;
    struct{
        uint32_t first_token,query_tokens,capacity,segment_capacity;
        uint32_t selected_stride;
    }push={first_token,query_tokens,capacity,segment_capacity,selected_stride};
    const fg_vk_tensor *bindings[]={
        record_0,record_1,selected,query,gate,output
    };
    return dispatch(c,&c->qsa_resident_attention,bindings,&push,
                    24u,query_tokens,1u,err);
}

fg_status fg_vk_topk_reduce(fg_vk_context *c,fg_vk_tensor *out_scores,fg_vk_tensor *out_ids,const fg_vk_tensor *in_scores,const fg_vk_tensor *in_ids,uint32_t count,uint32_t *output_count,fg_error *err){uint32_t groups=(count+4095u)/4096u,produced=groups*512u;if(groups==1u&&count<512u)produced=count;if(!c||!count||!output_count||!tensor_range(in_scores,0,(uint64_t)count*4u)||!tensor_range(in_ids,0,(uint64_t)count*4u)||!tensor_range(out_scores,0,(uint64_t)produced*4u)||!tensor_range(out_ids,0,(uint64_t)produced*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid top-k reduction dispatch");return FG_ERR_ARGUMENT;}uint32_t width=2u;while(width<count&&width<4096u)width<<=1u;struct{uint32_t count,width;}push={count,width};const fg_vk_tensor *bindings[]={in_scores,in_ids,out_scores,out_ids};fg_status status=dispatch(c,&c->topk_v2,bindings,&push,groups,1,1,err);if(status==FG_OK)*output_count=produced;return status;}

fg_status fg_vk_topk_select(fg_vk_context *c,fg_vk_tensor *out_scores,fg_vk_tensor *out_ids,
                            const fg_vk_tensor *in_scores,const fg_vk_tensor *in_ids,
                            uint32_t count,uint32_t k,uint32_t *output_count,fg_error *err){
    uint32_t groups=(count+1023u)/1024u;
    uint64_t produced=(uint64_t)groups*k;
    if(!c||!count||k<1u||k>64u||!output_count||produced>UINT32_MAX||
       !tensor_range(in_scores,0,(uint64_t)count*4u)||
       !tensor_range(in_ids,0,(uint64_t)count*4u)||
       !tensor_range(out_scores,0,produced*4u)||
       !tensor_range(out_ids,0,produced*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid generation top-k selection dispatch");
        return FG_ERR_ARGUMENT;
    }
    const fg_vk_tensor *bindings[]={in_scores,in_ids,out_scores,out_ids};
    struct { uint32_t count,k; } push={count,k};
    fg_vk_kernel *kernel=c->subgroup_topk?&c->topk_select:&c->topk_select_fallback;
    fg_status status=dispatch(c,kernel,bindings,&push,groups,1u,1u,err);
    if(status==FG_OK)*output_count=(uint32_t)produced;
    return status;
}
fg_status fg_vk_router_top10(fg_vk_context *c,fg_vk_tensor *selected,fg_vk_tensor *gates,const fg_vk_tensor *logits,uint32_t experts,uint32_t tokens,fg_error *err){uint64_t pairs=(uint64_t)tokens*FG_TOP_K;if(!c||experts!=FG_EXPERT_COUNT||!tokens||!tensor_range(logits,0,(uint64_t)tokens*experts*4u)||!tensor_range(selected,0,pairs*4u)||!tensor_range(gates,0,pairs*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU top-10 router dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t experts,tokens;}push={experts,tokens};const fg_vk_tensor *bindings[]={logits,selected,gates};return dispatch(c,&c->router_top10,bindings,&push,tokens,1u,1u,err);}
fg_status fg_vk_small_n_experts(fg_vk_context *c,fg_vk_tensor *mid,fg_vk_tensor *down_parts,
    fg_vk_tensor *reduced,const fg_vk_tensor *gate,const fg_vk_tensor *up,
    const fg_vk_tensor *down,const fg_vk_tensor *activation,
    const fg_vk_tensor *tiles,const fg_vk_tensor *selected,const fg_vk_tensor *gates,
    uint32_t gs,uint32_t us,uint32_t ds,uint32_t gt,uint32_t ut,uint32_t dt,
    uint32_t tokens,fg_error *err){
    if(!c||tokens<2u||tokens>4u||
       (gt!=12u&&gt!=13u)||(ut!=12u&&ut!=13u)||
       fg_vk_tensor_get_format(down)!=FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED||
       dt!=7u||!tensor_range(gate,0,(uint64_t)gs*FG_EXPERT_COUNT)||
       !tensor_range(up,0,(uint64_t)us*FG_EXPERT_COUNT)||
       !tensor_range(down,0,(uint64_t)ds*FG_EXPERT_COUNT)||
       !tensor_range(activation,0,(uint64_t)tokens*(FG_HIDDEN_SIZE/FG_QK8_K)*FG_Q8_K_BLOCK_BYTES)||
       !tensor_range(tiles,0,(uint64_t)tokens*FG_TOP_K*5u*4u)||
       !tensor_range(selected,0,(uint64_t)tokens*FG_TOP_K*4u)||
       !tensor_range(gates,0,(uint64_t)tokens*FG_TOP_K*4u)||
       !tensor_range(mid,0,(uint64_t)tokens*FG_TOP_K*640u*4u)||
       !tensor_range(down_parts,0,(uint64_t)tokens*FG_TOP_K*FG_HIDDEN_SIZE*4u*4u)||
       !tensor_range(reduced,0,(uint64_t)tokens*FG_HIDDEN_SIZE*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"small-N expert path requires owned cooked K/Q5_1 tensors");
        return FG_ERR_ARGUMENT;
    }
    fg_vk_kernel *gate_up=&c->small_gate_up[((gt==13u?2u:0u)+(ut==13u?1u:0u))*2u+(tokens>2u)];
    if(!gate_up->file)*gate_up=(fg_vk_kernel){
        .file=tokens==2u?"fg_small_gate_up.spv":"fg_small_gate_up_four.spv",.bindings=5,.push_bytes=32,
        .specialization_count=2,.specialization_values={gt==13u,ut==13u}};
    if(!c->small_down.file)c->small_down=(fg_vk_kernel){
        .file="fg_small_down.spv",.bindings=4,.push_bytes=16};
    if(!c->small_down_reduce.file)c->small_down_reduce=(fg_vk_kernel){
        .file="fg_small_down_reduce.spv",.bindings=3,.push_bytes=4};
    struct{uint32_t rows,blocks,gtile,utile,gs,us,gt,ut;}gp={
        640u,10u,(uint32_t)fg_k_quant_cooked_tile_bytes(2560u,gt),
        (uint32_t)fg_k_quant_cooked_tile_bytes(2560u,ut),gs,us,gt,ut};
    const fg_vk_tensor *gb[]={gate,up,activation,tiles,mid};
    fg_status status=dispatch(c,gate_up,gb,&gp,80u,tokens*FG_TOP_K,1u,err);
    struct{uint32_t rows,in,tile,stride;}dp={
        FG_HIDDEN_SIZE,640u,(uint32_t)fg_q5_1_cooked_tile_bytes(640u),ds};
    const fg_vk_tensor *db[]={down,tiles,mid,down_parts};
    if(status==FG_OK)status=dispatch(c,&c->small_down,db,&dp,FG_HIDDEN_SIZE/16u,tokens*FG_TOP_K,1u,err);
    const fg_vk_tensor *rb[]={down_parts,gates,reduced};
    uint32_t width=FG_HIDDEN_SIZE;
    if(status==FG_OK)status=dispatch(c,&c->small_down_reduce,rb,&width,FG_HIDDEN_SIZE/32u,tokens,1u,err);
    return status;
}
fg_status fg_vk_small_n_schedule(fg_vk_context *c,fg_vk_tensor *tiles,
    const fg_vk_tensor *selected,uint32_t tokens,fg_error *err){
    uint32_t pairs=tokens*FG_TOP_K;
    if(!c||tokens<2u||tokens>4u||!tensor_range(tiles,0,(uint64_t)pairs*5u*4u)||
       !tensor_range(selected,0,(uint64_t)pairs*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid small-N expert schedule");
        return FG_ERR_ARGUMENT;
    }
    if(!c->small_union.file)c->small_union=(fg_vk_kernel){
        .file="fg_small_union.spv",.bindings=2,.push_bytes=4};
    const fg_vk_tensor *bindings[]={selected,tiles};
    return dispatch(c,&c->small_union,bindings,&pairs,1u,1u,1u,err);
}
fg_status fg_vk_small_n_shared(fg_vk_context *c,fg_vk_tensor *output,
    const fg_vk_tensor *experts,const fg_vk_tensor *shared,
    const fg_vk_tensor *scalar,uint32_t tokens,fg_error *err){
    uint64_t bytes=(uint64_t)tokens*FG_HIDDEN_SIZE*4u;
    if(!c||tokens<2u||tokens>4u||!tensor_range(output,0,bytes)||
       !tensor_range(experts,0,bytes)||!tensor_range(shared,0,bytes)||
       !tensor_range(scalar,0,(uint64_t)tokens*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid small-N shared reduction");return FG_ERR_ARGUMENT;
    }
    if(!c->small_shared.file)c->small_shared=(fg_vk_kernel){
        .file="fg_small_shared.spv",.bindings=4,.push_bytes=8};
    uint32_t push[]={FG_HIDDEN_SIZE,tokens};
    const fg_vk_tensor *b[]={experts,shared,scalar,output};
    return dispatch(c,&c->small_shared,b,push,(FG_HIDDEN_SIZE+255u)/256u,tokens,1u,err);
}
fg_status fg_vk_expert_major_pack(fg_vk_context *c,fg_vk_tensor *tiles,const fg_vk_tensor *selected,uint32_t experts,uint32_t tokens,fg_error *err){uint64_t pairs=(uint64_t)tokens*FG_TOP_K,tile_bytes=pairs*FG_VK_PREFILL_TILE_WORDS*4u;if(!c||experts!=FG_EXPERT_COUNT||!tokens||pairs>UINT32_MAX||!tensor_range(selected,0,pairs*4u)||!tensor_range(tiles,0,tile_bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU expert-major packing dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t pairs,experts,tile_capacity;}push={(uint32_t)pairs,experts,(uint32_t)pairs};const fg_vk_tensor *bindings[]={selected,tiles};return dispatch(c,&c->expert_major_pack,bindings,&push,1u,1u,1u,err);}
fg_status fg_vk_decode_tile_schedule(fg_vk_context *c,fg_vk_tensor *tiles,const fg_vk_tensor *selected,fg_error *err){if(!c||!tensor_range(selected,0,FG_TOP_K*4u)||!tensor_range(tiles,0,FG_TOP_K*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU decode tile schedule dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={selected,tiles};return dispatch(c,&c->decode_tile_schedule,bindings,NULL,1u,1u,1u,err);}
fg_status fg_vk_argmax_reduce(fg_vk_context *c,fg_vk_tensor *out_scores,fg_vk_tensor *out_ids,const fg_vk_tensor *in_scores,const fg_vk_tensor *in_ids,uint32_t count,uint32_t *output_count,fg_error *err){uint32_t groups=(count+4095u)/4096u;if(!c||!count||!output_count||!tensor_range(in_scores,0,(uint64_t)count*4u)||!tensor_range(in_ids,0,(uint64_t)count*4u)||!tensor_range(out_scores,0,(uint64_t)groups*4u)||!tensor_range(out_ids,0,(uint64_t)groups*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid argmax reduction dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={in_scores,in_ids,out_scores,out_ids};fg_status status=dispatch(c,&c->argmax,bindings,&count,groups,1,1,err);if(status==FG_OK)*output_count=groups;return status;}
fg_status fg_vk_moe_q5_1_down(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,bool packed,uint32_t tile_count,fg_error *err){if(weights&&weights->format==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED)return fg_vk_moe_q5_1_down_cooked_pairs(c,out,weights,tiles,input,output_width,input_width,expert_stride,used_experts,packed,tile_count,err);uint64_t row_bytes=(uint64_t)(input_width/32u)*24u;if(!c||!output_width||!input_width||input_width%32u||!tile_count||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q5_1 expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;} push={output_width,input_width,(uint32_t)row_bytes,expert_stride,used_experts,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q5_1,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_q5_1_down_cooked_pairs(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint64_t tile_bytes=fg_q5_1_cooked_tile_bytes(input_width),matrix_bytes=fg_q5_1_cooked_matrix_bytes(input_width,output_width);uint64_t weight_bytes=(uint64_t)expert_stride*(packed?tile_count:FG_EXPERTS_PER_RANK);if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!routed_pairs||!tile_count||expert_stride<matrix_bytes||!tensor_range(weights,0,weight_bytes)||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(input,0,(uint64_t)routed_pairs*input_width*4u)||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked Q5_1 expert dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,routed_pairs,packed_weights;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,routed_pairs,packed?1u:0u};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q5_1_cooked,bindings,&push,output_width/8u,tile_count,1,err);}
fg_status fg_vk_moe_q5_1_down_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,bool packed,uint32_t tile_count,fg_error *err){return fg_vk_moe_q5_1_down_cooked_pairs(c,out,weights,tiles,input,output_width,input_width,expert_stride,FG_TOP_K,packed,tile_count,err);}
fg_status fg_vk_moe_q8_0_down(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,bool packed,uint32_t tile_count,fg_error *err){if(weights&&weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED)return fg_vk_moe_q8_0_down_cooked_pairs(c,out,weights,tiles,input,output_width,input_width,expert_stride,used_experts,packed,tile_count,err);uint64_t row_bytes=(uint64_t)(input_width/32u)*34u;if(!c||!output_width||!input_width||input_width%32u||!tile_count||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;} push={output_width,input_width,(uint32_t)row_bytes,expert_stride,used_experts,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q8_0,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_q8_0_down_cooked_pairs(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(input_width),matrix_bytes=fg_q8_0_cooked_matrix_bytes(input_width,output_width);uint64_t weight_bytes=(uint64_t)expert_stride*(packed?tile_count:FG_EXPERTS_PER_RANK);if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!output_width||!routed_pairs||!tile_count||expert_stride<matrix_bytes||!tensor_range(weights,0,weight_bytes)||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(input,0,(uint64_t)routed_pairs*input_width*4u)||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked Q8_0 expert dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,routed_pairs,packed_weights;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,routed_pairs,packed?1u:0u};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q8_0_cooked,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_q8_0_down_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,bool packed,uint32_t tile_count,fg_error *err){return fg_vk_moe_q8_0_down_cooked_pairs(c,out,weights,tiles,input,output_width,input_width,expert_stride,FG_TOP_K,packed,tile_count,err);}
fg_status fg_vk_moe_reduce(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *down,const fg_vk_tensor *gates,const fg_vk_tensor *tiles,uint32_t output_width,uint32_t selected_count,uint32_t slot_count,fg_error *err){if(!c||!output_width||!selected_count||selected_count>slot_count||!tensor_range(down,0,(uint64_t)slot_count*output_width*4u)||!tensor_range(gates,0,(uint64_t)selected_count*4u)||!tensor_range(tiles,0,(uint64_t)selected_count*9u*4u)||!tensor_range(out,0,(uint64_t)output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid MoE reduction dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,selected_count,slot_count;}push={output_width,selected_count,slot_count};const fg_vk_tensor *bindings[]={down,gates,tiles,out};return dispatch(c,&c->moe_reduce,bindings,&push,(output_width+255u)/256u,1,1,err);}
fg_status fg_vk_moe_kquant(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){if(weights&&weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED)return fg_vk_moe_kquant_cooked_pairs(c,out,weights,activation,tiles,type,output_width,input_width,expert_stride,used_experts,routed_pairs,packed,tile_count,err);if(weights&&weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED)return fg_vk_moe_kquant_cooked_pairs(c,out,weights,activation,tiles,8u,output_width,input_width,expert_stride,used_experts,routed_pairs,packed,tile_count,err);uint32_t block_bytes=type==12u?144u:type==13u?176u:0u;uint64_t row_bytes=(uint64_t)(input_width/256u)*block_bytes;if(!c||!block_bytes||!output_width||!input_width||input_width%256u||!used_experts||!routed_pairs||!tile_count||row_bytes>UINT32_MAX||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(activation,0,(uint64_t)((input_width/256u)*296u)*(routed_pairs/used_experts+(routed_pairs%used_experts?1u:0u)))||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q4_K/Q5_K expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,blocks,row_bytes,expert_stride,type,n_used,routed_pairs,packed_weights,reserved;} push={output_width,input_width/256u,(uint32_t)row_bytes,expert_stride,type,used_experts,routed_pairs,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,activation,tiles,out};return dispatch(c,&c->kquant,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_kquant_cooked_pairs(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint64_t tile_bytes=type==8u?fg_q8_0_cooked_tile_bytes(input_width):fg_k_quant_cooked_tile_bytes(input_width,type),matrix_bytes=type==8u?fg_q8_0_cooked_matrix_bytes(input_width,output_width):fg_k_quant_cooked_matrix_bytes(input_width,output_width,type),weight_bytes=(uint64_t)expert_stride*(packed?tile_count:FG_EXPERTS_PER_RANK),activation_bytes=(uint64_t)(input_width/FG_QK8_K)*FG_Q8_K_BLOCK_BYTES*((routed_pairs+used_experts-1u)/used_experts);if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!used_experts||!routed_pairs||!tile_count||expert_stride<matrix_bytes||!tensor_range(weights,0,weight_bytes)||!tensor_range(activation,0,activation_bytes)||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked expert dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,blocks,tile_bytes,expert_stride,type,n_used,routed_pairs,packed_weights;}push={output_width,input_width/FG_QK8_K,(uint32_t)tile_bytes,expert_stride,type,used_experts,routed_pairs,packed?1u:0u};const fg_vk_tensor *bindings[]={weights,activation,tiles,out};return dispatch(c,&c->kquant_cooked,bindings,&push,output_width/8u,tile_count,1,err);}
fg_status fg_vk_moe_kquant_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,bool packed,uint32_t tile_count,fg_error *err){return fg_vk_moe_kquant_cooked_pairs(c,out,weights,activation,tiles,type,output_width,input_width,expert_stride,FG_TOP_K,FG_TOP_K,packed,tile_count,err);}
/* Purpose-built batch-1 expert pair: gate+up+SwiGLU in one dispatch, then the
   down projection with the gate reduction folded in.  The gate/up projections
   accept the cooked K-quant and cooked Q8_0 layouts; the down projection also
   accepts cooked Q5_1 and raw row-major Q8_0, so Q8_0 expert tensors use the
   two-dispatch path after the size-neutral cooked repack.  Tensors that cannot
   take the fused pair fall back to the generic projection dispatches. */
bool fg_vk_decode_experts_fusable(const fg_vk_tensor *gate_weights,
    const fg_vk_tensor *up_weights,const fg_vk_tensor *down_weights,
    uint32_t down_type){
    bool down_ok=down_weights&&
        (down_weights->format==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED||
         down_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED||
         (down_weights->format==FG_VK_TENSOR_FORMAT_DEFAULT&&down_type==8u));
    bool gate_ok=gate_weights&&
        (gate_weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||
         gate_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED);
    bool up_ok=up_weights&&
        (up_weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||
         up_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED);
    return gate_ok&&up_ok&&down_ok;
}

fg_status fg_vk_moe_decode_gate_up(fg_vk_context *c,fg_vk_tensor *mid,
    const fg_vk_tensor *gate_weights,const fg_vk_tensor *up_weights,
    const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t output_width,
    uint32_t input_width,uint32_t gate_stride,uint32_t up_stride,uint32_t gate_type,
    uint32_t up_type,uint32_t slots,fg_error *err){
    uint64_t gate_tile=gate_type==8u?fg_q8_0_cooked_tile_bytes(input_width):fg_k_quant_cooked_tile_bytes(input_width,gate_type);
    uint64_t up_tile=up_type==8u?fg_q8_0_cooked_tile_bytes(input_width):fg_k_quant_cooked_tile_bytes(input_width,up_type);
    uint64_t gate_matrix=gate_type==8u?fg_q8_0_cooked_matrix_bytes(input_width,output_width):fg_k_quant_cooked_matrix_bytes(input_width,output_width,gate_type);
    uint64_t up_matrix=up_type==8u?fg_q8_0_cooked_matrix_bytes(input_width,output_width):fg_k_quant_cooked_matrix_bytes(input_width,output_width,up_type);
    if(!c||!output_width||!input_width||input_width%FG_QK8_K||!slots||
       !gate_tile||!up_tile||gate_tile>UINT32_MAX||up_tile>UINT32_MAX||
       gate_stride<gate_matrix||up_stride<up_matrix||
       !tensor_on_context(c,mid)||!tensor_on_context(c,gate_weights)||
       !tensor_on_context(c,up_weights)||!tensor_on_context(c,activation)||
       !tensor_on_context(c,tiles)||
       !tensor_range(mid,0,(uint64_t)slots*output_width*4u)||
       !tensor_range(gate_weights,0,gate_stride)||
       !tensor_range(up_weights,0,up_stride)||
       !tensor_range(activation,0,(uint64_t)(input_width/FG_QK8_K)*FG_Q8_K_BLOCK_BYTES)||
       !tensor_range(tiles,0,(uint64_t)slots*9u*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused gate/up decode dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,blocks,gate_tile_bytes,up_tile_bytes,gate_stride,up_stride,gate_type,up_type;}push={output_width,input_width/FG_QK8_K,(uint32_t)gate_tile,(uint32_t)up_tile,gate_stride,up_stride,gate_type,up_type};
    const fg_vk_tensor *bindings[]={gate_weights,up_weights,activation,tiles,mid};
    return dispatch(c,&c->moe_decode_gate_up,bindings,&push,(output_width+7u)/8u,slots,1u,err);
}

fg_status fg_vk_moe_decode_down_reduce(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *down_weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,
    const fg_vk_tensor *gates,uint32_t output_width,uint32_t input_width,
    uint32_t expert_stride,uint32_t slots,uint32_t down_type,fg_error *err){
    uint64_t tile_bytes=0u,matrix_bytes=0u;
    bool q8_cooked=down_type==8u&&down_weights&&
        down_weights->format==FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED;
    if(down_type==7u){
        tile_bytes=fg_q5_1_cooked_tile_bytes(input_width);
        matrix_bytes=fg_q5_1_cooked_matrix_bytes(input_width,output_width);
    }else if(down_type==8u){
        if(q8_cooked){
            tile_bytes=fg_q8_0_cooked_tile_bytes(input_width);
            matrix_bytes=fg_q8_0_cooked_matrix_bytes(input_width,output_width);
        }else matrix_bytes=(uint64_t)(input_width/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES*output_width;
    }
    if(!c||!output_width||!input_width||input_width%FG_QK8_0||!slots||
       !matrix_bytes||tile_bytes>UINT32_MAX||expert_stride<matrix_bytes||
       !tensor_on_context(c,out)||!tensor_on_context(c,down_weights)||
       !tensor_on_context(c,tiles)||!tensor_on_context(c,input)||
       !tensor_on_context(c,gates)||
       !tensor_range(out,0,(uint64_t)output_width*4u)||
       !tensor_range(down_weights,0,expert_stride)||
       !tensor_range(tiles,0,(uint64_t)slots*9u*4u)||
       !tensor_range(input,0,(uint64_t)slots*input_width*4u)||
       !tensor_range(gates,0,(uint64_t)slots*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused down/reduce decode dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,slots,down_type;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,slots,down_type};
    const fg_vk_tensor *bindings[]={down_weights,tiles,input,gates,out};
    bool cooked=q8_cooked&&c->subgroup_expert_q8;
    uint32_t groups=cooked?(output_width+1u)/2u:q8_cooked?(output_width+7u)/8u:(output_width+15u)/16u;
    return dispatch(c,cooked?&c->moe_decode_down_q8:&c->moe_decode_down_reduce,bindings,&push,groups,1u,1u,err);
}

/* Decode common-path finish: fold the shared-expert contribution (scaled by
 * sigmoid of its router scalar) into the routed reduction so the host never
 * reads either tensor back. */
fg_status fg_vk_moe_decode_shared_add(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *reduced,const fg_vk_tensor *shared,const fg_vk_tensor *scalar,
    uint32_t width,fg_error *err){
    if(!c||!width||!tensor_range(reduced,0,(uint64_t)width*4u)||
       !tensor_range(shared,0,(uint64_t)width*4u)||!tensor_range(scalar,0,4u)||
       !tensor_range(out,0,(uint64_t)width*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid shared-expert fold dispatch");
        return FG_ERR_ARGUMENT;
    }
    const fg_vk_tensor *bindings[]={reduced,shared,scalar,out};
    return dispatch(c,&c->moe_decode_shared_add,bindings,&width,(width+255u)/256u,1u,1u,err);
}

fg_status fg_vk_moe_kquant_cooked_grouped(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,
    uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,
    uint32_t weight_experts,uint32_t tokens,uint32_t tile_count,fg_error *err){
    uint64_t matrix_bytes=fg_k_quant_cooked_matrix_bytes(input_width,output_width,type),tile_bytes=fg_k_quant_cooked_tile_bytes(input_width,type),pairs=(uint64_t)tokens*FG_TOP_K,activation_bytes=(uint64_t)tokens*(input_width/FG_QK8_K)*FG_Q8_K_BLOCK_BYTES;
    if(!c||
       !matrix_bytes||tile_bytes>UINT32_MAX||
       !tokens||
       !tile_count||
       tile_count>pairs||
       !weight_experts||expert_stride<matrix_bytes||
       !weights||weights->format!=FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||
       !tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||
       !tensor_range(activation,0,activation_bytes)||
       !tensor_range(tiles,0,(uint64_t)tile_count*FG_VK_PREFILL_TILE_WORDS*4u)||
       !tensor_range(out,0,pairs*output_width*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped cooked K-quant dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,blocks,tile_bytes,expert_stride,type,tokens;}push={output_width,input_width/FG_QK8_K,(uint32_t)tile_bytes,expert_stride,type,tokens};
    const fg_vk_tensor *bindings[]={weights,activation,tiles,out};
    /* GFX1013 (BC-250) has no V_DOT4: RADV reports
       integerDotProduct4x8BitPackedSignedAccelerated=false (ac_gpu_info.c
       excludes CHIP_GFX1013) and the LLVM gfx1013 ISA feature set omits
       Dot1Insts, so c->integer_dot_product stays false and the float kernel
       runs.  Only pick the dp4a kernel where the property is actually
       advertised. */
    fg_vk_kernel *kernel=c->integer_dot_product?&c->kquant_cooked_grouped_int:&c->kquant_cooked_grouped;
    /* The int fallback still walks the legacy eight-row workgroup; the float
       kernel uses the wider 32-row tile. */
    uint32_t rows=kernel==&c->kquant_cooked_grouped?FG_VK_PREFILL_ROW_TILE:FG_K_QUANT_COOK_ROWS;
    fg_status status=dispatch(c,kernel,bindings,&push,(output_width+rows-1u)/rows,tile_count,1u,err);
    if(status!=FG_OK&&kernel==&c->kquant_cooked_grouped_int){
        c->integer_dot_product=false;
        status=dispatch(c,&c->kquant_cooked_grouped,bindings,&push,(output_width+FG_VK_PREFILL_ROW_TILE-1u)/FG_VK_PREFILL_ROW_TILE,tile_count,1u,err);
    }
    return status;
}
fg_status fg_vk_moe_q5_1_down_cooked_grouped(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,
    uint32_t weight_experts,uint32_t tokens,uint32_t tile_count,fg_error *err){
    uint64_t matrix_bytes=fg_q5_1_cooked_matrix_bytes(input_width,output_width),tile_bytes=fg_q5_1_cooked_tile_bytes(input_width),pairs=(uint64_t)tokens*FG_TOP_K;
    if(!c||
       !matrix_bytes||tile_bytes>UINT32_MAX||
       !tokens||
       !tile_count||
       tile_count>pairs||
       !weight_experts||expert_stride<matrix_bytes||
       !weights||weights->format!=FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED||
       !tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||
       !tensor_range(tiles,0,(uint64_t)tile_count*FG_VK_PREFILL_TILE_WORDS*4u)||
       !tensor_range(input,0,pairs*input_width*4u)||
       !tensor_range(out,0,pairs*output_width*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped cooked Q5_1 down dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,tokens;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,tokens};
    const fg_vk_tensor *bindings[]={weights,tiles,input,out};
    return dispatch(c,&c->q5_1_cooked_grouped,bindings,&push,(output_width+FG_VK_PREFILL_ROW_TILE-1u)/FG_VK_PREFILL_ROW_TILE,tile_count,1u,err);
}
fg_status fg_vk_moe_q8_0_down_grouped(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,
    uint32_t weight_experts,uint32_t tokens,uint32_t tile_count,fg_error *err){
    uint64_t row_bytes=(uint64_t)(input_width/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES,matrix_bytes=row_bytes*output_width,pairs=(uint64_t)tokens*FG_TOP_K;
    if(!c||
       !input_width||input_width%FG_QK8_0||
       !output_width||
       !tokens||
       !tile_count||
       tile_count>pairs||
       !weight_experts||expert_stride<matrix_bytes||
       !weights||weights->format!=FG_VK_TENSOR_FORMAT_DEFAULT||row_bytes>UINT32_MAX||
       !tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||
       !tensor_range(tiles,0,(uint64_t)tile_count*FG_VK_PREFILL_TILE_WORDS*4u)||
       !tensor_range(input,0,pairs*input_width*4u)||
       !tensor_range(out,0,pairs*output_width*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped Q8_0 down dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,in_dim,row_bytes,expert_stride,tokens;}push={output_width,input_width,(uint32_t)row_bytes,expert_stride,tokens};
    const fg_vk_tensor *bindings[]={weights,tiles,input,out};
    return dispatch(c,&c->q8_0_grouped,bindings,&push,(output_width+FG_VK_PREFILL_ROW_TILE-1u)/FG_VK_PREFILL_ROW_TILE,tile_count,1u,err);
}
fg_status fg_vk_moe_q8_0_down_cooked_grouped(fg_vk_context *c,fg_vk_tensor *out,
    const fg_vk_tensor *weights,const fg_vk_tensor *tiles,
    const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,
    uint32_t weight_experts,uint32_t tokens,uint32_t tile_count,fg_error *err){
    uint64_t matrix_bytes=fg_q8_0_cooked_matrix_bytes(input_width,output_width),tile_bytes=fg_q8_0_cooked_tile_bytes(input_width),pairs=(uint64_t)tokens*FG_TOP_K;
    if(!c||
       !matrix_bytes||tile_bytes>UINT32_MAX||
       !tokens||
       !tile_count||
       tile_count>pairs||
       !weight_experts||expert_stride<matrix_bytes||
       !weights||weights->format!=FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED||
       !tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||
       !tensor_range(tiles,0,(uint64_t)tile_count*FG_VK_PREFILL_TILE_WORDS*4u)||
       !tensor_range(input,0,pairs*input_width*4u)||
       !tensor_range(out,0,pairs*output_width*4u)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped cooked Q8_0 down dispatch");
        return FG_ERR_ARGUMENT;
    }
    struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,tokens;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,tokens};
    const fg_vk_tensor *bindings[]={weights,tiles,input,out};
    return dispatch(c,&c->q8_0_cooked_grouped,bindings,&push,(output_width+FG_VK_PREFILL_ROW_TILE-1u)/FG_VK_PREFILL_ROW_TILE,tile_count,1u,err);
}
fg_status fg_vk_moe_prefill_reduce(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *expert_output,const fg_vk_tensor *gates,const fg_vk_tensor *shared_output,const fg_vk_tensor *shared_logit,uint32_t width,uint32_t tokens,fg_error *err){uint64_t pairs=(uint64_t)tokens*FG_TOP_K;if(!c||!width||!tokens||!tensor_range(expert_output,0,pairs*width*4u)||!tensor_range(gates,0,pairs*4u)||!tensor_range(shared_output,0,(uint64_t)tokens*width*4u)||!tensor_range(shared_logit,0,(uint64_t)tokens*4u)||!tensor_range(out,0,(uint64_t)tokens*width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU prefill gated reduction");return FG_ERR_ARGUMENT;}struct{uint32_t width,tokens;}push={width,tokens};const fg_vk_tensor *bindings[]={expert_output,gates,shared_output,shared_logit,out};return dispatch(c,&c->moe_prefill_reduce,bindings,&push,(width+255u)/256u,tokens,1u,err);}

/* -------- Decomposition benchmark: stream / dequant / dot-no-reduce / full -------- */

static double bench_run_timestamped(fg_vk_context *c,fg_vk_kernel *kernel,
    const fg_vk_tensor *w,const fg_vk_tensor *x,fg_vk_tensor **ys,uint32_t n,
    const void *push,uint32_t gx,uint32_t gy,
    VkQueryPool qpool,uint32_t ts_base,float ts_period,double *first_us,fg_error *err)
{
    fg_status drain=vk_pipeline_drain(c,err);if(drain!=FG_OK)return -1.0;
    vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkResult vr=vkBeginCommandBuffer(c->command,&begin);if(vr!=VK_SUCCESS)return -1.0;
    vkCmdResetQueryPool(c->command,qpool,ts_base,n*2+2);
    VkMemoryBarrier hd={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&hd,0,NULL,0,NULL);
    c->batch_depth=1;c->batch_set_count=0;c->batch_set_base=0;
    vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base);
    for(uint32_t i=0;i<n;i++){
        vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base+1+i*2);
        fg_status s=bench_record_dispatch_no_barrier(c,kernel,w,x,ys[i],push,gx,gy,err);
        if(s!=FG_OK){c->batch_depth=0;return -1.0;}
        vkCmdWriteTimestamp(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,qpool,ts_base+2+i*2);
    }
    VkMemoryBarrier dh={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&dh,0,NULL,0,NULL);
    vkEndCommandBuffer(c->command);
    VkSubmitInfo sub={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};
    vr=vkQueueSubmit(c->queue,1,&sub,c->fence);if(vr!=VK_SUCCESS){c->batch_depth=0;return -1.0;}
    vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,5000000000ULL);
    c->batch_depth=0;c->batch_set_count=0;
    if(vr!=VK_SUCCESS)return -1.0;
    uint64_t *ts=malloc((n*2+2)*sizeof(uint64_t));if(!ts)return -1.0;
    vr=vkGetQueryPoolResults(c->device,qpool,ts_base,n*2+2,(n*2+2)*sizeof(uint64_t),ts,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
    if(vr!=VK_SUCCESS&&vr!=VK_NOT_READY){free(ts);return -1.0;}
    double total_ns=0;for(uint32_t i=0;i<n;i++){total_ns+=(double)(ts[1+i*2+1]-ts[1+i*2])*(double)ts_period;}
    if(first_us)*first_us=(double)(ts[2]-ts[1])*(double)ts_period/1e3;
    free(ts);return total_ns/1e3/(double)n; /* average μs per dispatch */
}

fg_status fg_vk_bench_decompose(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(c->physical,&props);
    float ts_period=props.limits.timestampPeriod;
    VkDescriptorPoolSize bps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=2048};
    VkDescriptorPoolCreateInfo bdp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,.maxSets=512,.poolSizeCount=1,.pPoolSizes=&bps};
    VkDescriptorPool saved=c->descriptor_pool;
    VkResult vr=vkCreateDescriptorPool(c->device,&bdp,NULL,&c->descriptor_pool);
    if(vr!=VK_SUCCESS){c->descriptor_pool=saved;return vk_error(err,"decompose pool",vr);}
    VkQueryPoolCreateInfo qpc={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=512};
    VkQueryPool qpool=VK_NULL_HANDLE;
    vr=vkCreateQueryPool(c->device,&qpc,NULL,&qpool);
    if(vr!=VK_SUCCESS){vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);c->descriptor_pool=saved;return vk_error(err,"decompose qpool",vr);}

    #define DEC_N 10
    struct {uint32_t in,out;const char *name;} shapes[]={
        {10240,320,"hc_down 10240→320"},{320,10240,"hc_up 320→10240"},
        {2560,640,"shexp_gate 2560→640"},{640,2560,"shexp_down 640→2560"},
        {2560,512,"qsa_attn_q 2560→512"},{2560,10240,"ple_key 2560→10240"},
    };
    fprintf(stderr,"\n=== Q8_0 kernel decomposition (GPU timestamps, %u dispatches) ===\n",DEC_N);
    fprintf(stderr,"%-20s %7s │ %8s %8s │ %8s %8s │ %8s │ %8s %8s │ %8s\n",
        "shape","wt MB","stream","GB/s","dequant","GB/s","dot_nr","full","GB/s","theor");
    fprintf(stderr,"─────────────────────────────────────────────────────────────────────────────────────────\n");

    fg_status status=FG_OK;
    fg_vk_kernel *kernels[]={&c->bench_stream,&c->bench_dequant,&c->bench_dot_nored,&c->dense};

    for(uint32_t s=0;s<sizeof(shapes)/sizeof(shapes[0])&&status==FG_OK;s++){
        uint32_t in_dim=shapes[s].in,out_dim=shapes[s].out;
        uint32_t blocks=in_dim/32u;uint64_t row_bytes=(uint64_t)blocks*34u;
        uint64_t weight_bytes=row_bytes*out_dim;
        uint64_t total_data=weight_bytes+(uint64_t)in_dim*4u+(uint64_t)out_dim*4u;
        struct{uint32_t od,nt,bl,rb;float sc;}push={out_dim,1,blocks,(uint32_t)row_bytes,1.0f};

        fg_vk_tensor *w=NULL,*x=NULL;
        /* dot_nored writes 64 floats per row, so output needs out_dim*64*4 bytes */
        uint64_t big_out=(uint64_t)out_dim*64u*4u;
        fg_vk_tensor *ys[DEC_N]={0};
        status=fg_vk_tensor_create(c,weight_bytes,&w,err);
        if(status==FG_OK)status=fg_vk_tensor_create(c,(uint64_t)in_dim*4u,&x,err);
        for(uint32_t i=0;status==FG_OK&&i<DEC_N;i++)
            status=fg_vk_tensor_create(c,big_out,&ys[i],err);
        if(status!=FG_OK)break;
        memset(fg_vk_tensor_map(w),0x42,weight_bytes);
        float *xp=fg_vk_tensor_map(x);for(uint32_t i=0;i<in_dim;i++)xp[i]=1.0f/(float)(i+1);

        /* Warmup all kernels */
        for(uint32_t k=0;k<4&&status==FG_OK;k++){
            for(uint32_t i=0;status==FG_OK&&i<5;i++){
                const fg_vk_tensor *bindings[]={w,x,ys[0]};
                status=dispatch(c,kernels[k],bindings,&push,out_dim,1,1,err);
            }
        }

        double times[4]={0};
        uint32_t ts_off=0;
        for(uint32_t k=0;k<4&&status==FG_OK;k++){
            times[k]=bench_run_timestamped(c,kernels[k],w,x,ys,DEC_N,&push,out_dim,1,qpool,ts_off,ts_period,NULL,err);
            ts_off+=DEC_N*2+2;
            if(times[k]<0){status=FG_ERR_IO;break;}
        }

        if(status==FG_OK){
            double theor=(double)total_data/(357.0*1e3);
            fprintf(stderr,"%-20s %6.2f │ %7.1f %7.1f │ %7.1f %7.1f │ %7.1f │ %7.1f %7.1f │ %7.1f\n",
                shapes[s].name,(double)weight_bytes/1e6,
                times[0],(double)weight_bytes/(times[0]*1e-6)/1e9,
                times[1],(double)weight_bytes/(times[1]*1e-6)/1e9,
                times[2],
                times[3],(double)total_data/(times[3]*1e-6)/1e9,
                theor);
        }
        for(uint32_t i=0;i<DEC_N;i++)fg_vk_tensor_destroy(ys[i]);
        fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);
    }
    #undef DEC_N
    vkDestroyQueryPool(c->device,qpool,NULL);
    vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);
    c->descriptor_pool=saved;
    return status;
}

/* -------- Layout vs access experiment: A (current) / B (wide) / C (vec4) -------- */

fg_status fg_vk_bench_stream_abc(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(c->physical,&props);
    float ts_period=props.limits.timestampPeriod;
    VkDescriptorPoolSize bps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=2048};
    VkDescriptorPoolCreateInfo bdp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,.maxSets=512,.poolSizeCount=1,.pPoolSizes=&bps};
    VkDescriptorPool saved=c->descriptor_pool;
    VkResult vr=vkCreateDescriptorPool(c->device,&bdp,NULL,&c->descriptor_pool);
    if(vr!=VK_SUCCESS){c->descriptor_pool=saved;return vk_error(err,"stream_abc pool",vr);}
    VkQueryPoolCreateInfo qpc={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=512};
    VkQueryPool qpool=VK_NULL_HANDLE;
    vr=vkCreateQueryPool(c->device,&qpc,NULL,&qpool);
    if(vr!=VK_SUCCESS){vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);c->descriptor_pool=saved;return vk_error(err,"stream_abc qpool",vr);}

    #define ABC_N 10
    struct {uint32_t in,out;const char *name;} shapes[]={
        {10240,320,"hc_down 10240→320"},{320,10240,"hc_up 320→10240"},
        {2560,640,"shexp_gate 2560→640"},{640,2560,"shexp_down 640→2560"},
        {2560,512,"qsa_attn_q 2560→512"},{2560,10240,"ple_key 2560→10240"},
    };
    fprintf(stderr,"\n=== Layout vs Access experiment (GPU timestamps, %u dispatches) ===\n",ABC_N);
    fprintf(stderr,"%-20s %7s │ %8s %8s │ %8s %8s │ %8s %8s │ %8s\n",
        "shape","wt MB","A:cur","A:GB/s","B:wide","B:GB/s","C:vec4","C:GB/s","theor");
    fprintf(stderr,"────────────────────────────────────────────────────────────────────────────────\n");

    fg_vk_kernel *kernels[]={&c->bench_stream,&c->bench_stream_wide,&c->bench_stream_vec};

    fg_status status=FG_OK;
    for(uint32_t s=0;s<sizeof(shapes)/sizeof(shapes[0])&&status==FG_OK;s++){
        uint32_t in_dim=shapes[s].in,out_dim=shapes[s].out;
        uint32_t blocks=in_dim/32u;uint64_t row_bytes=(uint64_t)blocks*34u;
        uint64_t weight_bytes=row_bytes*out_dim;
        struct{uint32_t od,nt,bl,rb;float sc;}push={out_dim,1,blocks,(uint32_t)row_bytes,1.0f};

        /* Ensure weight buffer is 16-byte aligned for vec4 access.
           Pad weight_bytes up to multiple of 16. */
        uint64_t padded_weight=(weight_bytes+15u)&~15ULL;
        fg_vk_tensor *w=NULL,*x=NULL;
        fg_vk_tensor *ys[ABC_N]={0};
        status=fg_vk_tensor_create(c,padded_weight,&w,err);
        if(status==FG_OK)status=fg_vk_tensor_create(c,(uint64_t)in_dim*4u,&x,err);
        for(uint32_t i=0;status==FG_OK&&i<ABC_N;i++)
            status=fg_vk_tensor_create(c,(uint64_t)out_dim*4u,&ys[i],err);
        if(status!=FG_OK)break;
        memset(fg_vk_tensor_map(w),0x42,padded_weight);

        /* Warmup all 3 kernels */
        for(uint32_t k=0;k<3&&status==FG_OK;k++){
            for(uint32_t i=0;status==FG_OK&&i<5;i++){
                const fg_vk_tensor *bindings[]={w,x,ys[0]};
                status=dispatch(c,kernels[k],bindings,&push,out_dim,1,1,err);
            }
        }

        double times[3]={0};
        uint32_t ts_off=0;
        for(uint32_t k=0;k<3&&status==FG_OK;k++){
            times[k]=bench_run_timestamped(c,kernels[k],w,x,ys,ABC_N,&push,out_dim,1,qpool,ts_off,ts_period,NULL,err);
            ts_off+=ABC_N*2+2;
            if(times[k]<0){fprintf(stderr,"  kernel %u failed for %s\n",k,shapes[s].name);status=FG_ERR_IO;break;}
        }

        if(status==FG_OK){
            double theor=(double)weight_bytes/(357.0*1e3);
            fprintf(stderr,"%-20s %6.2f │ %7.1f %7.1f │ %7.1f %7.1f │ %7.1f %7.1f │ %7.1f\n",
                shapes[s].name,(double)weight_bytes/1e6,
                times[0],(double)weight_bytes/(times[0]*1e-6)/1e9,
                times[1],(double)weight_bytes/(times[1]*1e-6)/1e9,
                times[2],(double)weight_bytes/(times[2]*1e-6)/1e9,
                theor);
        }
        for(uint32_t i=0;i<ABC_N;i++)fg_vk_tensor_destroy(ys[i]);
        fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);
    }
    #undef ABC_N
    vkDestroyQueryPool(c->device,qpool,NULL);
    vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);
    c->descriptor_pool=saved;
    return status;
}

static uint64_t cooked_layout_quant_offset(uint32_t blocks){return((uint64_t)16u*blocks*2u+63u)&~UINT64_C(63);}

static bool cooked_layout_repack(const uint8_t *source,uint8_t *destination,uint32_t blocks,uint32_t rows,uint32_t row_stride,uint32_t block_stride,uint32_t group_stride){
    uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(blocks*FG_QK8_0),required=fg_q8_0_cooked_matrix_bytes(blocks*FG_QK8_0,rows),quant_offset=cooked_layout_quant_offset(blocks);
    if(!tile_bytes||!required||!quant_offset)return false;
    memset(destination,0,(size_t)required);
    for(uint64_t tile=0;tile*tile_bytes<required;tile++){
        const uint8_t *src=source+tile*tile_bytes;uint8_t *dst=destination+tile*tile_bytes;
        memcpy(dst,src,(size_t)quant_offset);
        for(uint32_t row=0;row<FG_Q8_0_COOK_ROWS;row++)for(uint32_t block=0;block<blocks;block++)
            memcpy(dst+quant_offset+(uint64_t)(row&7u)*row_stride+(uint64_t)block*block_stride+(uint64_t)(row>>3u)*group_stride,
                   src+quant_offset+((uint64_t)row*blocks+block)*FG_QK8_0,FG_QK8_0);
    }
    return true;
}

static void cooked_layout_strides(uint32_t which,uint32_t blocks,uint32_t *row_stride,uint32_t *block_stride,uint32_t *group_stride){
    if(which==0u){*row_stride=FG_QK8_0;*block_stride=8u*FG_QK8_0;*group_stride=blocks*8u*FG_QK8_0;}
    else if(which==1u){*row_stride=FG_QK8_0;*block_stride=16u*FG_QK8_0;*group_stride=8u*FG_QK8_0;}
    else{*row_stride=blocks*FG_QK8_0;*block_stride=FG_QK8_0;*group_stride=blocks*8u*FG_QK8_0;}
}

static bool bench_cooked_layout_parity_one(fg_vk_context *c,uint32_t input_width,uint32_t output_width,fg_error *err){
    uint32_t blocks=input_width/FG_QK8_0,source_row=blocks*FG_Q8_0_BLOCK_BYTES;
    uint64_t source_bytes=(uint64_t)output_width*source_row,cooked_bytes=fg_q8_0_cooked_matrix_bytes(input_width,output_width),tile_bytes=fg_q8_0_cooked_tile_bytes(input_width);
    float *source=malloc((size_t)input_width*output_width*4u),*input=malloc((uint64_t)input_width*4u),*reference=malloc((uint64_t)output_width*4u),*candidate=malloc((uint64_t)output_width*4u);
    uint8_t *quantized=malloc((size_t)source_bytes),*cooked=malloc((size_t)cooked_bytes),*grouped=malloc((size_t)cooked_bytes),*interleaved=malloc((size_t)cooked_bytes);
    bool ok=source&&input&&reference&&candidate&&quantized&&cooked&&grouped&&interleaved;
    if(ok){
        for(uint32_t row=0;row<output_width;row++){
            for(uint32_t i=0;i<input_width;i++)source[(uint64_t)row*input_width+i]=sinf((float)((uint64_t)row*input_width+i)*0.0031f)+0.25f*cosf((float)i*0.011f);
            fg_quantize_q8_0(source+(uint64_t)row*input_width,quantized+(uint64_t)row*source_row,input_width);
        }
        for(uint32_t i=0;i<input_width;i++)input[i]=cosf((float)(i+5u)*0.017f)-0.3f*sinf((float)i*0.0037f);
        ok=fg_cook_q8_0_rows(quantized,cooked,cooked_bytes,input_width,output_width)&&
            cooked_layout_repack(cooked,grouped,blocks,output_width,FG_QK8_0,8u*FG_QK8_0,blocks*8u*FG_QK8_0)&&
            cooked_layout_repack(cooked,interleaved,blocks,output_width,FG_QK8_0,16u*FG_QK8_0,8u*FG_QK8_0);
    }
    fg_vk_tensor *wA=NULL,*wB=NULL,*wC=NULL,*x=NULL,*yA=NULL,*yB=NULL,*yC=NULL,*yD=NULL;
    if(ok)ok=fg_vk_tensor_create(c,cooked_bytes,&wA,err)==FG_OK&&fg_vk_tensor_create(c,cooked_bytes,&wB,err)==FG_OK&&fg_vk_tensor_create(c,cooked_bytes,&wC,err)==FG_OK&&
        fg_vk_tensor_create(c,(uint64_t)input_width*4u,&x,err)==FG_OK&&fg_vk_tensor_create(c,(uint64_t)output_width*4u,&yA,err)==FG_OK&&
        fg_vk_tensor_create(c,(uint64_t)output_width*4u,&yB,err)==FG_OK&&fg_vk_tensor_create(c,(uint64_t)output_width*4u,&yC,err)==FG_OK&&fg_vk_tensor_create(c,(uint64_t)output_width*4u,&yD,err)==FG_OK;
    if(ok){
        memcpy(fg_vk_tensor_map(wA),cooked,(size_t)cooked_bytes);memcpy(fg_vk_tensor_map(wB),grouped,(size_t)cooked_bytes);memcpy(fg_vk_tensor_map(wC),interleaved,(size_t)cooked_bytes);memcpy(fg_vk_tensor_map(x),input,(size_t)input_width*4u);
        struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;}push20={output_width,1u,blocks,(uint32_t)tile_bytes,1.0f};
        struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;uint32_t row_stride,block_stride,group_stride;}push_prod={output_width,1u,blocks,(uint32_t)tile_bytes,1.0f,blocks*FG_QK8_0,FG_QK8_0,blocks*8u*FG_QK8_0};
        struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;uint32_t row_stride,block_stride,group_stride;}push_g8={output_width,1u,blocks,(uint32_t)tile_bytes,1.0f,FG_QK8_0,8u*FG_QK8_0,blocks*8u*FG_QK8_0};
        struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;uint32_t row_stride,block_stride,group_stride;}push_il={output_width,1u,blocks,(uint32_t)tile_bytes,1.0f,FG_QK8_0,16u*FG_QK8_0,8u*FG_QK8_0};
        const fg_vk_tensor *bindings_a[]={wA,x,yA},*bindings_b[]={wA,x,yB},*bindings_c[]={wB,x,yC},*bindings_d[]={wC,x,yD};
        ok=dispatch(c,&c->dense_cooked_r8,bindings_a,&push20,(output_width+7u)/8u,1u,1u,err)==FG_OK&&
           dispatch(c,&c->bench_cooked_layout,bindings_b,&push_prod,(output_width+7u)/8u,1u,1u,err)==FG_OK&&
           dispatch(c,&c->bench_cooked_layout,bindings_c,&push_g8,(output_width+7u)/8u,1u,1u,err)==FG_OK&&
           dispatch(c,&c->bench_cooked_layout,bindings_d,&push_il,(output_width+7u)/8u,1u,1u,err)==FG_OK;
        ok=ok&&fg_vk_tensor_read(yA,0,reference,(uint64_t)output_width*4u,err)==FG_OK&&
           fg_vk_tensor_read(yB,0,candidate,(uint64_t)output_width*4u,err)==FG_OK&&memcmp(reference,candidate,(size_t)output_width*4u)==0;
        ok=ok&&fg_vk_tensor_read(yC,0,candidate,(uint64_t)output_width*4u,err)==FG_OK&&memcmp(reference,candidate,(size_t)output_width*4u)==0;
        ok=ok&&fg_vk_tensor_read(yD,0,candidate,(uint64_t)output_width*4u,err)==FG_OK&&memcmp(reference,candidate,(size_t)output_width*4u)==0;
    }
    fprintf(stderr,"  layout parity in=%u out=%u blocks=%u: %s\n",input_width,output_width,blocks,ok?"PASS":"FAIL");
    fg_vk_tensor_destroy(yD);fg_vk_tensor_destroy(yC);fg_vk_tensor_destroy(yB);fg_vk_tensor_destroy(yA);fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(wC);fg_vk_tensor_destroy(wB);fg_vk_tensor_destroy(wA);
    free(interleaved);free(grouped);free(cooked);free(quantized);free(candidate);free(reference);free(input);free(source);
    return ok;
}

static bool bench_cooked_layout_parity(fg_vk_context *c,fg_error *err){
    return bench_cooked_layout_parity_one(c,512u,32u,err)&&bench_cooked_layout_parity_one(c,320u,48u,err);
}

fg_status fg_vk_bench_cooked_layout(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(c->physical,&props);
    float ts_period=props.limits.timestampPeriod;
    VkDescriptorPoolSize bps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=2048};
    VkDescriptorPoolCreateInfo bdp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,.maxSets=512,.poolSizeCount=1,.pPoolSizes=&bps};
    VkDescriptorPool saved=c->descriptor_pool;
    VkResult vr=vkCreateDescriptorPool(c->device,&bdp,NULL,&c->descriptor_pool);
    if(vr!=VK_SUCCESS){c->descriptor_pool=saved;return vk_error(err,"cooked layout pool",vr);}
    VkQueryPoolCreateInfo qpc={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=512};
    VkQueryPool qpool=VK_NULL_HANDLE;
    vr=vkCreateQueryPool(c->device,&qpc,NULL,&qpool);
    if(vr!=VK_SUCCESS){vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);c->descriptor_pool=saved;return vk_error(err,"cooked layout qpool",vr);}

    #define LC_N 10
    struct{uint32_t in,out;const char *name;}shapes[]={
        {2560u,640u,"gate_2560"},{2560u,2560u,"ple_value"},{2560u,10240u,"gdn_qkv"},
        {6144u,640u,"probe6144_640"},{6144u,2560u,"gdn_output"},{6144u,10240u,"probe6144_10k"},
        {640u,2560u,"shared_down"},{320u,10240u,"gr_up"}
    };
    const char *names[4]={"r8","g8_256","il_512","prod"};
    fprintf(stderr,"\n=== cooked layout bench (GPU timestamps, %u dispatches per set) ===\n",LC_N);
    bool parity=bench_cooked_layout_parity(c,err);
    fprintf(stderr,"layout parity: %s\n",parity?"PASS":"FAIL");
    fg_status status=FG_OK;
    for(uint32_t s=0;s<sizeof(shapes)/sizeof(shapes[0])&&status==FG_OK;s++){
        uint32_t in_dim=shapes[s].in,out_dim=shapes[s].out,blocks=in_dim/FG_QK8_0,gx=(out_dim+7u)/8u;
        uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(in_dim),matrix_bytes=fg_q8_0_cooked_matrix_bytes(in_dim,out_dim),weight_bytes=matrix_bytes+2u*tile_bytes+4096u;
        fg_vk_tensor *w=NULL,*x=NULL,*ys[LC_N]={0};
        status=fg_vk_tensor_create(c,weight_bytes,&w,err);
        if(status==FG_OK)status=fg_vk_tensor_create(c,(uint64_t)in_dim*4u,&x,err);
        for(uint32_t i=0;status==FG_OK&&i<LC_N;i++)status=fg_vk_tensor_create(c,(uint64_t)out_dim*4u,&ys[i],err);
        if(status!=FG_OK)break;
        memset(fg_vk_tensor_map(w),0x42,(size_t)weight_bytes);
        float *xp=fg_vk_tensor_map(x);for(uint32_t i=0;i<in_dim;i++)xp[i]=1.0f/(float)(i+1);
        struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;}push20={out_dim,1u,blocks,(uint32_t)tile_bytes,1.0f};
        struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;uint32_t row_stride,block_stride,group_stride;}push={out_dim,1u,blocks,(uint32_t)tile_bytes,1.0f,0u,0u,0u};
        for(uint32_t v=0;v<4u&&status==FG_OK;v++){
            cooked_layout_strides(v-1u,blocks,&push.row_stride,&push.block_stride,&push.group_stride);
            for(uint32_t i=0;i<3u&&status==FG_OK;i++){
                const fg_vk_tensor *bindings[]={w,x,ys[0]};
                fg_vk_kernel *kernel=(v==0)?&c->dense_cooked_r8:&c->bench_cooked_layout;
                status=dispatch(c,kernel,bindings,(v==0)?(const void *)&push20:(const void *)&push,gx,1u,1u,err);
            }
        }
        if(status!=FG_OK)break;
        fprintf(stderr,"  %s in=%u out=%u blocks=%u grid=%ux1 wt=%.2f MB\n",shapes[s].name,in_dim,out_dim,blocks,gx,(double)matrix_bytes/1e6);
        double times[2][4]={{0}},firsts[2][4]={{0}};
        for(uint32_t p=0;p<2u&&status==FG_OK;p++){
            for(uint32_t v=0;v<4u&&status==FG_OK;v++){
                cooked_layout_strides(v-1u,blocks,&push.row_stride,&push.block_stride,&push.group_stride);
                fg_vk_kernel *kernel=(v==0)?&c->dense_cooked_r8:&c->bench_cooked_layout;
                const void *constants=(v==0)?(const void *)&push20:(const void *)&push;
                times[p][v]=bench_run_timestamped(c,kernel,w,x,ys,LC_N,constants,gx,1u,qpool,(p*4u+v)*(LC_N*2u+2u),ts_period,&firsts[p][v],err);
                if(times[p][v]<0){status=FG_ERR_IO;break;}
                fprintf(stderr,"   p%u %-6s %8.1f us %7.1f GB/s  first %8.1f us\n",p+1u,names[v],
                    times[p][v],(double)matrix_bytes/(times[p][v]*1e-6)/1e9,firsts[p][v]);
            }
        }
        if(status==FG_OK){
            fprintf(stderr,"   prod/config:");
            for(uint32_t l=1u;l<3u;l++)fprintf(stderr," %s %.3fx/%.3fx",names[l],times[0][3]/times[0][l],times[1][3]/times[1][l]);
            fprintf(stderr,"\n");
        }
        for(uint32_t i=0;i<LC_N;i++)fg_vk_tensor_destroy(ys[i]);
        fg_vk_tensor_destroy(x);fg_vk_tensor_destroy(w);
    }
    #undef LC_N
    vkDestroyQueryPool(c->device,qpool,NULL);
    vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);
    c->descriptor_pool=saved;
    return status;
}

fg_status fg_vk_moe_prefill_shard_reduce(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *experts,const fg_vk_tensor *gates,uint32_t tokens,fg_error *err){
    if(!c||!tokens||!tensor_range(experts,0,(uint64_t)tokens*FG_TOP_K*FG_HIDDEN_SIZE*4u)||!tensor_range(gates,0,(uint64_t)tokens*FG_TOP_K*4u)||!tensor_range(out,0,(uint64_t)tokens*FG_HIDDEN_SIZE*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid shard reduction buffers");return FG_ERR_ARGUMENT;}
    struct{uint32_t width,tokens;}push={FG_HIDDEN_SIZE,tokens};
    const fg_vk_tensor *bindings[]={experts,gates,out};
    return dispatch(c,&c->moe_prefill_shard_reduce,bindings,&push,(FG_HIDDEN_SIZE+255u)/256u,tokens,1u,err);
}
