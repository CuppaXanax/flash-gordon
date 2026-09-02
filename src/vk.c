#include "fg_vk.h"
#include "fg_quant.h"
#include "fg_q38_schema.h"

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct fg_vk_allocation {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    uint64_t bytes,allocated_bytes;
    uint32_t references;
} fg_vk_allocation;

struct fg_vk_tensor {fg_vk_context *context;fg_vk_allocation *allocation;uint64_t offset,bytes;fg_vk_tensor_format format;bool is_view;};
typedef struct fg_vk_kernel {const char *file;uint32_t bindings,push_bytes;VkDescriptorSetLayout set_layout;VkPipelineLayout layout;VkPipeline pipeline;} fg_vk_kernel;
typedef struct fg_vk_profile_dispatch {const char *scope,*name;uint32_t begin_query,end_query;} fg_vk_profile_dispatch;

#define FG_VK_BATCH_MAX_SETS 4096u
#define FG_VK_PROFILE_MAX_DISPATCHES 4096u
#define FG_VK_PROFILE_QUERY_COUNT (2u+2u*FG_VK_PROFILE_MAX_DISPATCHES)

struct fg_vk_context {
    VkInstance instance;VkPhysicalDevice physical;VkDevice device;VkQueue queue;uint32_t queue_family;
    VkCommandPool command_pool;VkCommandBuffer command;VkFence fence;VkDescriptorPool descriptor_pool;VkDescriptorSetLayout descriptor_set_layout;VkPipelineCache pipeline_cache;
    VkPhysicalDeviceMemoryProperties memory;char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    bool subgroup_topk;
    uint32_t batch_depth;uint32_t batch_set_count;VkDescriptorSet descriptor_sets[FG_VK_BATCH_MAX_SETS];bool batch_has_dispatch;
    VkQueryPool profile_query_pool;float timestamp_period;uint32_t timestamp_valid_bits;
    bool profile_active,profile_overflow;const char *profile_scope;uint32_t profile_query_count,profile_dispatch_count;
    fg_vk_counters counters;
    fg_vk_memory_stats memory_stats;
    fg_vk_profile_dispatch profile_dispatches[FG_VK_PROFILE_MAX_DISPATCHES];fg_vk_profile profile;
    fg_vk_kernel quant_q8k,quant_q8,quant_q4,dequant_iq4nl,embedding,embedding_batch,swiglu,silu_scaled,dense,dense_f32,dense_bf16,rms,gr,hc_inject_partial,gr_partial,hc_finalize,gr_write,ple_gate,ple_gate_prefill,ple_conv,ple_conv_prefill,add,apply_penalties,gdn_conv,gdn_conv_prefill,gdn_recurrent,gdn_recurrent_algebraic,gdn_recurrent_prefill,gdn_prefill_qk_norm,gdn_prefill_recurrence,gdn_prefill_output,qsa_prepare,qsa_prepare_prefill,qsa_index_prepare,qsa_index_prepare_prefill,qsa_record_commit,qsa_record_gather,qsa_score,qsa_attention,qsa_resident_commit,qsa_resident_select,qsa_resident_merge,qsa_resident_attention,topk,topk_select,topk_select_fallback,moe_q5_1,moe_q5_1_cooked,moe_q8_0,moe_reduce,kquant,kquant_cooked;
    fg_vk_kernel argmax,dense_subgroup,dense_cooked,dense_cooked_r8;
    /* Decomposition benchmark kernels */
    fg_vk_kernel bench_stream,bench_dequant,bench_dot_nored;
    fg_vk_kernel dense_cooked_split,dense_cooked_split_reduce;
    fg_vk_kernel bench_stream_wide,bench_stream_vec;
    fg_vk_kernel dense_cooked_tile,router_top10,expert_major_pack;
    fg_vk_kernel decode_tile_schedule;
    fg_vk_kernel kquant_cooked_grouped,q5_1_cooked_grouped,q8_0_grouped;
    fg_vk_kernel moe_prefill_reduce;
};

struct fg_vk_expert_graph {
    fg_vk_context *context;
    VkDescriptorPool descriptor_pool;
    VkCommandBuffer command;
    uint32_t dispatches;
};

static fg_status vk_error(fg_error *err,const char *operation,VkResult result){fg_error_set(err,FG_ERR_UNAVAILABLE,"%s: Vulkan result %d",operation,(int)result);return FG_ERR_UNAVAILABLE;}
static bool tensor_range(const fg_vk_tensor *tensor,uint64_t offset,uint64_t bytes){return tensor&&offset<=tensor->bytes&&bytes<=tensor->bytes-offset;}
static bool tensor_on_context(const fg_vk_context *context,const fg_vk_tensor *tensor){return tensor&&tensor->context==context;}

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

static fg_status profile_resolve(fg_vk_context *c,fg_error *err){
    if(!c->profile_active)return FG_OK;
    uint64_t ticks[FG_VK_PROFILE_QUERY_COUNT];VkResult vr=vkGetQueryPoolResults(c->device,c->profile_query_pool,0,c->profile_query_count,(size_t)c->profile_query_count*sizeof(*ticks),ticks,sizeof(*ticks),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);if(vr!=VK_SUCCESS)return vk_error(err,"read timestamp queries",vr);
    if(c->profile_overflow){fg_error_set(err,FG_ERR_LIMIT,"Vulkan profile exceeded %u dispatches in one submission",FG_VK_PROFILE_MAX_DISPATCHES);return FG_ERR_LIMIT;}
    double scale=(double)c->timestamp_period/1000000.0,total=(double)profile_tick_delta(c,ticks[0],ticks[c->profile_query_count-1u])*scale;c->profile.submissions++;c->profile.gpu_ms+=total;
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
    if(kernel==&context->topk) { destroy_kernel(context,&context->topk_select); destroy_kernel(context,&context->topk_select_fallback); }
    if(kernel==&context->moe_q5_1)destroy_kernel(context,&context->moe_q5_1_cooked);
    if(kernel==&context->kquant)destroy_kernel(context,&context->kquant_cooked);
    if(kernel==&context->dense_cooked)destroy_kernel(context,&context->dense_cooked_r8);
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
    VkPipelineShaderStageCreateInfo stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=module,.pName="main"};VkComputePipelineCreateInfo cp={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=stage,.layout=kernel->layout};
    vr=vkCreateComputePipelines(context->device,context->pipeline_cache,1,&cp,NULL,&kernel->pipeline);vkDestroyShaderModule(context->device,module,NULL);if(vr!=VK_SUCCESS){destroy_kernel(context,kernel);return vk_error(err,"create compute pipeline",vr);}return FG_OK;
}

fg_status fg_vk_open(fg_vk_context **out,fg_error *err){
    if(!out){fg_error_set(err,FG_ERR_ARGUMENT,"Vulkan context output is null");return FG_ERR_ARGUMENT;}*out=NULL;fg_vk_context *c=calloc(1,sizeof(*c));if(!c){fg_error_set(err,FG_ERR_OOM,"allocate Vulkan context");return FG_ERR_OOM;}
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="flash-gordon",.applicationVersion=1,.pEngineName="flash-gordon",.engineVersion=1,.apiVersion=VK_API_VERSION_1_1};VkInstanceCreateInfo ic={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};VkResult vr=vkCreateInstance(&ic,NULL,&c->instance);if(vr!=VK_SUCCESS){free(c);return vk_error(err,"create Vulkan instance",vr);}
    uint32_t count=0;vr=vkEnumeratePhysicalDevices(c->instance,&count,NULL);if(vr!=VK_SUCCESS||count==0){fg_vk_close(c);return vk_error(err,"enumerate Vulkan devices",vr);}
    VkPhysicalDevice *devices=malloc((size_t)count*sizeof(*devices));if(!devices){fg_vk_close(c);fg_error_set(err,FG_ERR_OOM,"enumerate Vulkan devices");return FG_ERR_OOM;}vr=vkEnumeratePhysicalDevices(c->instance,&count,devices);if(vr!=VK_SUCCESS){free(devices);fg_vk_close(c);return vk_error(err,"enumerate Vulkan devices",vr);}
    int best=-1;uint32_t family=0;for(uint32_t d=0;d<count;d++){uint32_t families=0;vkGetPhysicalDeviceQueueFamilyProperties(devices[d],&families,NULL);VkQueueFamilyProperties *properties=malloc((size_t)families*sizeof(*properties));if(!properties)continue;vkGetPhysicalDeviceQueueFamilyProperties(devices[d],&families,properties);VkPhysicalDeviceProperties device_properties;vkGetPhysicalDeviceProperties(devices[d],&device_properties);int score=device_properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU?3:device_properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?2:1;for(uint32_t f=0;f<families;f++)if((properties[f].queueFlags&VK_QUEUE_COMPUTE_BIT)&&score>best){best=score;c->physical=devices[d];family=f;c->timestamp_valid_bits=properties[f].timestampValidBits;}free(properties);}free(devices);
    if(best<0){fg_vk_close(c);fg_error_set(err,FG_ERR_UNAVAILABLE,"no Vulkan compute queue");return FG_ERR_UNAVAILABLE;}c->queue_family=family;VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(c->physical,&properties);memcpy(c->device_name,properties.deviceName,sizeof(c->device_name));c->timestamp_period=properties.limits.timestampPeriod;vkGetPhysicalDeviceMemoryProperties(c->physical,&c->memory);VkPhysicalDeviceSubgroupProperties subgroup={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};VkPhysicalDeviceProperties2 properties2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&subgroup};vkGetPhysicalDeviceProperties2(c->physical,&properties2);c->subgroup_topk=properties.limits.maxComputeWorkGroupInvocations>=1024u&&properties.limits.maxComputeWorkGroupSize[0]>=1024u&&subgroup.subgroupSize>=32u&&subgroup.subgroupSize<=64u&&(subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)!=0u&&(subgroup.supportedOperations&(VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))==(VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    float priority=1.0f;VkDeviceQueueCreateInfo qc={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};VkDeviceCreateInfo dc={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qc};vr=vkCreateDevice(c->physical,&dc,NULL,&c->device);if(vr!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create Vulkan device",vr);}vkGetDeviceQueue(c->device,family,0,&c->queue);
    VkCommandPoolCreateInfo pc={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=family};if((vr=vkCreateCommandPool(c->device,&pc,NULL,&c->command_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create command pool",vr);}VkCommandBufferAllocateInfo ca={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};if((vr=vkAllocateCommandBuffers(c->device,&ca,&c->command))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"allocate command buffer",vr);}
    VkFenceCreateInfo fc={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};if((vr=vkCreateFence(c->device,&fc,NULL,&c->fence))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create fence",vr);}VkDescriptorSetLayoutBinding layout_bindings[16];for(uint32_t i=0;i<16u;i++)layout_bindings[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1u,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};VkDescriptorSetLayoutCreateInfo layout_create={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=16u,.pBindings=layout_bindings};if((vr=vkCreateDescriptorSetLayout(c->device,&layout_create,NULL,&c->descriptor_set_layout))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create shared descriptor layout",vr);}VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=FG_VK_BATCH_MAX_SETS*16u};VkDescriptorPoolCreateInfo dp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=FG_VK_BATCH_MAX_SETS,.poolSizeCount=1,.pPoolSizes=&ps};if((vr=vkCreateDescriptorPool(c->device,&dp,NULL,&c->descriptor_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create descriptor pool",vr);}VkDescriptorSetLayout set_layouts[FG_VK_BATCH_MAX_SETS];for(uint32_t i=0;i<FG_VK_BATCH_MAX_SETS;i++)set_layouts[i]=c->descriptor_set_layout;VkDescriptorSetAllocateInfo set_allocate={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=c->descriptor_pool,.descriptorSetCount=FG_VK_BATCH_MAX_SETS,.pSetLayouts=set_layouts};if((vr=vkAllocateDescriptorSets(c->device,&set_allocate,c->descriptor_sets))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"preallocate descriptor sets",vr);}VkPipelineCacheCreateInfo pci={.sType=VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};if((vr=vkCreatePipelineCache(c->device,&pci,NULL,&c->pipeline_cache))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create pipeline cache",vr);}
    c->quant_q8k=(fg_vk_kernel){.file="fg_quantize_q8_k.spv",.bindings=2,.push_bytes=12};c->quant_q8=(fg_vk_kernel){.file="fg_quantize_q8_0.spv",.bindings=2,.push_bytes=12};c->quant_q4=(fg_vk_kernel){.file="fg_quantize_q4_0.spv",.bindings=2,.push_bytes=12};
    c->dequant_iq4nl=(fg_vk_kernel){.file="fg_dequantize_iq4_nl.spv",.bindings=2,.push_bytes=8};c->embedding=(fg_vk_kernel){.file="fg_embedding_q8_0.spv",.bindings=2,.push_bytes=16};c->embedding_batch=(fg_vk_kernel){.file="fg_embedding_q8_0_batch.spv",.bindings=3,.push_bytes=16};c->swiglu=(fg_vk_kernel){.file="fg_swiglu.spv",.bindings=3,.push_bytes=4};c->silu_scaled=(fg_vk_kernel){.file="fg_silu_scaled.spv",.bindings=2,.push_bytes=8};c->dense=(fg_vk_kernel){.file="fg_dense_q8_0_f32.spv",.bindings=3,.push_bytes=20};c->dense_f32=(fg_vk_kernel){.file="fg_dense_f32.spv",.bindings=3,.push_bytes=12};c->dense_bf16=(fg_vk_kernel){.file="fg_dense_bf16_f32.spv",.bindings=3,.push_bytes=12};
    c->rms=(fg_vk_kernel){.file="fg_group_rms_norm.spv",.bindings=3,.push_bytes=16};c->gr=(fg_vk_kernel){.file="fg_gr_mix.spv",.bindings=5,.push_bytes=12};c->hc_inject_partial=(fg_vk_kernel){.file="fg_hc_inject_partial.spv",.bindings=3,.push_bytes=16};c->gr_partial=(fg_vk_kernel){.file="fg_gr_mix_partial.spv",.bindings=5,.push_bytes=16};c->hc_finalize=(fg_vk_kernel){.file="fg_hc_finalize.spv",.bindings=3,.push_bytes=12};c->gr_write=(fg_vk_kernel){.file="fg_gr_write.spv",.bindings=4,.push_bytes=12};c->ple_gate=(fg_vk_kernel){.file="fg_ple_gate.spv",.bindings=4,.push_bytes=0};c->ple_gate_prefill=(fg_vk_kernel){.file="fg_ple_gate_prefill.spv",.bindings=4,.push_bytes=4};c->ple_conv=(fg_vk_kernel){.file="fg_ple_conv_decode.spv",.bindings=5,.push_bytes=0};c->ple_conv_prefill=(fg_vk_kernel){.file="fg_ple_conv_prefill.spv",.bindings=5,.push_bytes=4};c->add=(fg_vk_kernel){.file="fg_add_f32.spv",.bindings=3,.push_bytes=4};c->gdn_conv=(fg_vk_kernel){.file="fg_gdn_conv_decode.spv",.bindings=4,.push_bytes=4};c->gdn_conv_prefill=(fg_vk_kernel){.file="fg_gdn_conv_prefill.spv",.bindings=4,.push_bytes=8};c->gdn_recurrent=(fg_vk_kernel){.file="fg_gdn_recurrent_decode.spv",.bindings=9,.push_bytes=16};c->gdn_recurrent_algebraic=(fg_vk_kernel){.file="fg_gdn_recurrent_algebraic.spv",.bindings=9,.push_bytes=16};c->gdn_recurrent_prefill=(fg_vk_kernel){.file="fg_gdn_recurrent_prefill.spv",.bindings=9,.push_bytes=20};c->gdn_prefill_qk_norm=(fg_vk_kernel){.file="fg_gdn_prefill_qk_norm.spv",.bindings=1,.push_bytes=8};c->gdn_prefill_recurrence=(fg_vk_kernel){.file="fg_gdn_prefill_recurrence.spv",.bindings=7,.push_bytes=4};c->gdn_prefill_output=(fg_vk_kernel){.file="fg_gdn_prefill_output.spv",.bindings=3,.push_bytes=8};
    c->argmax=(fg_vk_kernel){.file="fg_argmax_reduce.spv",.bindings=4,.push_bytes=4};
    c->dense_subgroup=(fg_vk_kernel){.file="fg_dense_q8_0_subgroup.spv",.bindings=3,.push_bytes=20};
    c->dense_cooked=(fg_vk_kernel){.file="fg_dense_q8_0_cooked.spv",.bindings=3,.push_bytes=20};
    c->dense_cooked_r8=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_r8.spv",.bindings=3,.push_bytes=20};
    c->dense_cooked_split=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_split.spv",.bindings=3,.push_bytes=24};
    c->dense_cooked_split_reduce=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_split_reduce.spv",.bindings=2,.push_bytes=8};
    c->moe_q5_1=(fg_vk_kernel){.file="fg_moe_q5_1_down.spv",.bindings=4,.push_bytes=28};c->moe_q5_1_cooked=(fg_vk_kernel){.file="fg_moe_q5_1_down_cooked.spv",.bindings=4,.push_bytes=28};c->moe_q8_0=(fg_vk_kernel){.file="fg_moe_q8_0_down.spv",.bindings=4,.push_bytes=28};c->moe_reduce=(fg_vk_kernel){.file="fg_moe_reduce.spv",.bindings=4,.push_bytes=12};c->kquant=(fg_vk_kernel){.file="fg_moe_kquant.spv",.bindings=4,.push_bytes=36};c->kquant_cooked=(fg_vk_kernel){.file="fg_moe_kquant_cooked.spv",.bindings=4,.push_bytes=32};
    c->bench_stream=(fg_vk_kernel){.file="fg_bench_stream.spv",.bindings=3,.push_bytes=20};c->bench_dequant=(fg_vk_kernel){.file="fg_bench_dequant.spv",.bindings=3,.push_bytes=20};c->bench_dot_nored=(fg_vk_kernel){.file="fg_bench_dot_nored.spv",.bindings=3,.push_bytes=20};
    c->bench_stream_wide=(fg_vk_kernel){.file="fg_bench_stream_wide.spv",.bindings=3,.push_bytes=20};c->bench_stream_vec=(fg_vk_kernel){.file="fg_bench_stream_vec.spv",.bindings=3,.push_bytes=20};
    c->dense_cooked_tile=(fg_vk_kernel){.file="fg_dense_q8_0_cooked_tile.spv",.bindings=3,.push_bytes=20};
    c->router_top10=(fg_vk_kernel){.file="fg_router_top10.spv",.bindings=3,.push_bytes=8};
    c->expert_major_pack=(fg_vk_kernel){.file="fg_expert_major_pack.spv",.bindings=2,.push_bytes=12};
    c->decode_tile_schedule=(fg_vk_kernel){.file="fg_decode_tile_schedule.spv",.bindings=2,.push_bytes=0};
    c->kquant_cooked_grouped=(fg_vk_kernel){.file="fg_moe_kquant_cooked_grouped.spv",.bindings=4,.push_bytes=24};
    c->q5_1_cooked_grouped=(fg_vk_kernel){.file="fg_moe_q5_1_down_cooked_grouped.spv",.bindings=4,.push_bytes=24};
    c->q8_0_grouped=(fg_vk_kernel){.file="fg_moe_q8_0_down_grouped.spv",.bindings=4,.push_bytes=20};
    c->moe_prefill_reduce=(fg_vk_kernel){.file="fg_moe_prefill_reduce.spv",.bindings=5,.push_bytes=8};
    c->apply_penalties=(fg_vk_kernel){.file="fg_apply_penalties.spv",.bindings=2,.push_bytes=16};
    c->qsa_prepare=(fg_vk_kernel){.file="fg_qsa_prepare.spv",.bindings=8,.push_bytes=0};c->qsa_prepare_prefill=(fg_vk_kernel){.file="fg_qsa_prepare_prefill.spv",.bindings=8,.push_bytes=4};c->qsa_index_prepare=(fg_vk_kernel){.file="fg_qsa_index_prepare.spv",.bindings=4,.push_bytes=0};c->qsa_index_prepare_prefill=(fg_vk_kernel){.file="fg_qsa_index_prepare_prefill.spv",.bindings=4,.push_bytes=4};c->qsa_record_commit=(fg_vk_kernel){.file="fg_qsa_record_commit.spv",.bindings=6,.push_bytes=24};c->qsa_record_gather=(fg_vk_kernel){.file="fg_qsa_record_gather.spv",.bindings=3,.push_bytes=20};c->qsa_score=(fg_vk_kernel){.file="fg_qsa_index_score.spv",.bindings=6,.push_bytes=12};c->qsa_attention=(fg_vk_kernel){.file="fg_qsa_attention.spv",.bindings=4,.push_bytes=4};c->qsa_resident_commit=(fg_vk_kernel){.file="fg_qsa_resident_record_commit.spv",.bindings=8,.push_bytes=16};c->qsa_resident_select=(fg_vk_kernel){.file="fg_qsa_resident_select.spv",.bindings=7,.push_bytes=20};c->qsa_resident_merge=(fg_vk_kernel){.file="fg_qsa_resident_topk_merge.spv",.bindings=4,.push_bytes=16};c->qsa_resident_attention=(fg_vk_kernel){.file="fg_qsa_resident_attention.spv",.bindings=6,.push_bytes=20};c->topk=(fg_vk_kernel){.file="fg_topk_reduce.spv",.bindings=4,.push_bytes=4};
    c->topk_select=(fg_vk_kernel){.file="fg_topk_select.spv",.bindings=4,.push_bytes=8};
    c->topk_select_fallback=(fg_vk_kernel){.file="fg_topk_select_fallback.spv",.bindings=4,.push_bytes=8};
    *out=c;return FG_OK;
}

void fg_vk_close(fg_vk_context *c){if(!c)return;if(c->device)vkDeviceWaitIdle(c->device);destroy_kernel(c,&c->moe_prefill_reduce);destroy_kernel(c,&c->q8_0_grouped);destroy_kernel(c,&c->q5_1_cooked_grouped);destroy_kernel(c,&c->kquant_cooked_grouped);destroy_kernel(c,&c->decode_tile_schedule);destroy_kernel(c,&c->expert_major_pack);destroy_kernel(c,&c->router_top10);destroy_kernel(c,&c->dense_cooked_tile);destroy_kernel(c,&c->bench_stream_vec);destroy_kernel(c,&c->bench_stream_wide);destroy_kernel(c,&c->bench_dot_nored);destroy_kernel(c,&c->bench_dequant);destroy_kernel(c,&c->bench_stream);destroy_kernel(c,&c->kquant);destroy_kernel(c,&c->moe_reduce);destroy_kernel(c,&c->moe_q8_0);destroy_kernel(c,&c->moe_q5_1);destroy_kernel(c,&c->topk);destroy_kernel(c,&c->apply_penalties);destroy_kernel(c,&c->qsa_resident_attention);destroy_kernel(c,&c->qsa_resident_merge);destroy_kernel(c,&c->qsa_resident_select);destroy_kernel(c,&c->qsa_resident_commit);destroy_kernel(c,&c->qsa_attention);destroy_kernel(c,&c->qsa_score);destroy_kernel(c,&c->qsa_record_gather);destroy_kernel(c,&c->qsa_record_commit);destroy_kernel(c,&c->qsa_index_prepare_prefill);destroy_kernel(c,&c->qsa_index_prepare);destroy_kernel(c,&c->qsa_prepare_prefill);destroy_kernel(c,&c->qsa_prepare);destroy_kernel(c,&c->gdn_prefill_output);destroy_kernel(c,&c->gdn_prefill_recurrence);destroy_kernel(c,&c->gdn_prefill_qk_norm);destroy_kernel(c,&c->gdn_recurrent_prefill);destroy_kernel(c,&c->gdn_recurrent_algebraic);destroy_kernel(c,&c->gdn_recurrent);destroy_kernel(c,&c->gdn_conv_prefill);destroy_kernel(c,&c->gdn_conv);destroy_kernel(c,&c->add);destroy_kernel(c,&c->ple_conv_prefill);destroy_kernel(c,&c->ple_conv);destroy_kernel(c,&c->ple_gate_prefill);destroy_kernel(c,&c->ple_gate);destroy_kernel(c,&c->gr_write);destroy_kernel(c,&c->hc_finalize);destroy_kernel(c,&c->gr_partial);destroy_kernel(c,&c->hc_inject_partial);destroy_kernel(c,&c->gr);destroy_kernel(c,&c->rms);destroy_kernel(c,&c->dense_cooked);destroy_kernel(c,&c->dense_subgroup);destroy_kernel(c,&c->dense_bf16);destroy_kernel(c,&c->dense_f32);destroy_kernel(c,&c->dense);destroy_kernel(c,&c->silu_scaled);destroy_kernel(c,&c->swiglu);destroy_kernel(c,&c->embedding_batch);destroy_kernel(c,&c->embedding);destroy_kernel(c,&c->dequant_iq4nl);destroy_kernel(c,&c->quant_q4);destroy_kernel(c,&c->quant_q8);destroy_kernel(c,&c->quant_q8k);if(c->profile_query_pool)vkDestroyQueryPool(c->device,c->profile_query_pool,NULL);if(c->pipeline_cache)vkDestroyPipelineCache(c->device,c->pipeline_cache,NULL);if(c->descriptor_pool)vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);if(c->descriptor_set_layout)vkDestroyDescriptorSetLayout(c->device,c->descriptor_set_layout,NULL);if(c->fence)vkDestroyFence(c->device,c->fence,NULL);if(c->command_pool)vkDestroyCommandPool(c->device,c->command_pool,NULL);if(c->device)vkDestroyDevice(c->device,NULL);if(c->instance)vkDestroyInstance(c->instance,NULL);free(c);}
const char *fg_vk_device_name(const fg_vk_context *c){return c?c->device_name:"";}

fg_status fg_vk_profile_begin(fg_vk_context *c,fg_error *err){if(!c||c->batch_depth||c->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile begin");return FG_ERR_ARGUMENT;}if(!c->timestamp_valid_bits||c->timestamp_period<=0.0f){fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan compute queue has no timestamp support");return FG_ERR_UNAVAILABLE;}if(!c->profile_query_pool){VkQueryPoolCreateInfo create={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=FG_VK_PROFILE_QUERY_COUNT};VkResult vr=vkCreateQueryPool(c->device,&create,NULL,&c->profile_query_pool);if(vr!=VK_SUCCESS)return vk_error(err,"create timestamp query pool",vr);vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin timestamp query prime",vr);vkCmdResetQueryPool(c->command,c->profile_query_pool,0,FG_VK_PROFILE_QUERY_COUNT);if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS)return vk_error(err,"end timestamp query prime",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit timestamp query prime",vr);c->counters.submissions++;if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait timestamp query prime",vr);}memset(&c->profile,0,sizeof(c->profile));c->profile_scope="unscoped";c->profile_active=true;return FG_OK;}
fg_status fg_vk_profile_set_scope(fg_vk_context *c,const char *scope,fg_error *err){if(!c||!c->profile_active||!scope||!*scope){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile scope");return FG_ERR_ARGUMENT;}c->profile_scope=scope;return FG_OK;}
fg_status fg_vk_profile_end(fg_vk_context *c,fg_vk_profile *profile,fg_error *err){if(!c||!profile||c->batch_depth||!c->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile end");return FG_ERR_ARGUMENT;}*profile=c->profile;c->profile_active=false;return FG_OK;}
bool fg_vk_profile_active(const fg_vk_context *c){return c&&c->profile_active;}
void fg_vk_get_counters(const fg_vk_context *c,fg_vk_counters *counters){if(!counters)return;if(c)*counters=c->counters;else memset(counters,0,sizeof(*counters));}
void fg_vk_get_memory_stats(const fg_vk_context *c,fg_vk_memory_stats *stats){if(!stats)return;if(c)*stats=c->memory_stats;else memset(stats,0,sizeof(*stats));}

fg_status fg_vk_tensor_create(fg_vk_context *c,uint64_t bytes,fg_vk_tensor **out,fg_error *err){
    if(!c||!out||bytes==0){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor allocation");return FG_ERR_ARGUMENT;}*out=NULL;fg_vk_allocation *a=calloc(1,sizeof(*a));fg_vk_tensor *t=calloc(1,sizeof(*t));if(!a||!t){free(a);free(t);fg_error_set(err,FG_ERR_OOM,"allocate Vulkan tensor metadata");return FG_ERR_OOM;}
    VkBufferCreateInfo bc={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};VkResult vr=vkCreateBuffer(c->device,&bc,NULL,&a->buffer);if(vr!=VK_SUCCESS){free(a);free(t);return vk_error(err,"create Vulkan tensor buffer",vr);}VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(c->device,a->buffer,&requirements);uint32_t type=UINT32_MAX;VkMemoryPropertyFlags wanted=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;for(uint32_t i=0;i<c->memory.memoryTypeCount;i++)if((requirements.memoryTypeBits&(1u<<i))&&(c->memory.memoryTypes[i].propertyFlags&wanted)==wanted){if(type==UINT32_MAX||c->memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)type=i;}if(type==UINT32_MAX){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan device has no coherent host-visible memory");return FG_ERR_UNAVAILABLE;}VkMemoryAllocateInfo ma={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=requirements.size,.memoryTypeIndex=type};if((vr=vkAllocateMemory(c->device,&ma,NULL,&a->memory))!=VK_SUCCESS){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"allocate Vulkan tensor memory",vr);}if((vr=vkBindBufferMemory(c->device,a->buffer,a->memory,0))!=VK_SUCCESS||(vr=vkMapMemory(c->device,a->memory,0,VK_WHOLE_SIZE,0,&a->mapped))!=VK_SUCCESS){if(a->mapped)vkUnmapMemory(c->device,a->memory);vkFreeMemory(c->device,a->memory,NULL);vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"map Vulkan tensor memory",vr);}a->bytes=bytes;a->allocated_bytes=requirements.size;a->references=1;t->context=c;t->allocation=a;t->bytes=bytes;c->memory_stats.requested_live_bytes+=bytes;c->memory_stats.allocated_live_bytes+=requirements.size;c->memory_stats.allocation_count++;c->memory_stats.live_allocations++;if(c->memory_stats.requested_live_bytes>c->memory_stats.requested_peak_bytes)c->memory_stats.requested_peak_bytes=c->memory_stats.requested_live_bytes;if(c->memory_stats.allocated_live_bytes>c->memory_stats.allocated_peak_bytes)c->memory_stats.allocated_peak_bytes=c->memory_stats.allocated_live_bytes;*out=t;return FG_OK;
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
fg_status fg_vk_tensor_read(const fg_vk_tensor *t,uint64_t offset,void *data,uint64_t bytes,fg_error *err){if(!data||!tensor_range(t,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor read");return FG_ERR_ARGUMENT;}memcpy(data,(const uint8_t *)t->allocation->mapped+t->offset+offset,(size_t)bytes);return FG_OK;}

static fg_status dispatch_impl(fg_vk_context *c,fg_vk_kernel *kernel,const fg_vk_tensor *const *tensors,const void *push,uint32_t gx,uint32_t gy,uint32_t gz,bool batch_barrier,fg_error *err){
    fg_status status=create_kernel(c,kernel,err);if(status!=FG_OK)return status;if(c->batch_depth&&c->batch_set_count>=FG_VK_BATCH_MAX_SETS){fg_error_set(err,FG_ERR_LIMIT,"Vulkan batch exceeded %u descriptor sets",FG_VK_BATCH_MAX_SETS);return FG_ERR_LIMIT;}VkDescriptorSet set=c->descriptor_sets[c->batch_depth?c->batch_set_count:0u];VkResult vr;VkDescriptorBufferInfo info[16];VkWriteDescriptorSet write[16];for(uint32_t i=0;i<kernel->bindings;i++){info[i]=(VkDescriptorBufferInfo){.buffer=tensors[i]->allocation->buffer,.offset=tensors[i]->offset,.range=tensors[i]->bytes};write[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info[i]};}vkUpdateDescriptorSets(c->device,kernel->bindings,write,0,NULL);
    if(c->batch_depth){if(batch_barrier&&c->batch_has_dispatch){VkMemoryBarrier bar={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&bar,0,NULL,0,NULL);}vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);bool profiled=profile_dispatch_begin(c,kernel);vkCmdDispatch(c->command,gx,gy,gz);c->counters.dispatches++;c->batch_has_dispatch=true;profile_dispatch_end(c,profiled);c->batch_set_count++;return FG_OK;}
    vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin compute command",vr);profile_command_begin(c);VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);bool profiled=profile_dispatch_begin(c,kernel);vkCmdDispatch(c->command,gx,gy,gz);c->counters.dispatches++;profile_dispatch_end(c,profiled);VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);profile_command_end(c);if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS)return vk_error(err,"end compute command",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit compute command",vr);c->counters.submissions++;if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait for compute command",vr);    status=profile_resolve(c,err);return status;
}

static bool dense_cooked_rows8_shape(uint32_t input_width,uint32_t output_width,uint32_t tokens){return tokens==1u&&((input_width==320u&&output_width==10240u)||(input_width==2560u&&(output_width==10240u||output_width==640u||output_width==2560u||output_width==12288u||output_width==512u))||(input_width==640u&&output_width==2560u)||(input_width==6144u&&output_width==2560u));}
static fg_status dispatch(fg_vk_context *c,fg_vk_kernel *kernel,const fg_vk_tensor *const *tensors,const void *push,uint32_t gx,uint32_t gy,uint32_t gz,fg_error *err){if(kernel==&c->dense_cooked&&push){const uint32_t *parameters=push;uint32_t output_width=parameters[0],tokens=parameters[1],input_width=parameters[2]*FG_QK8_0;if(dense_cooked_rows8_shape(input_width,output_width,tokens)){kernel=&c->dense_cooked_r8;gx=(output_width+7u)/8u;}}return dispatch_impl(c,kernel,tensors,push,gx,gy,gz,true,err);}

fg_status fg_vk_begin(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    if(c->batch_depth){c->batch_depth++;return FG_OK;}
    VkResult vr;vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin batch command",vr);
    profile_command_begin(c);
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);
    c->batch_depth=1;c->batch_set_count=0;c->batch_has_dispatch=false;return FG_OK;
}

fg_status fg_vk_end(fg_vk_context *c,fg_error *err){
    if(!c||!c->batch_depth){fg_error_set(err,FG_ERR_ARGUMENT,"not in batch");return FG_ERR_ARGUMENT;}
    if(c->batch_depth>1){c->batch_depth--;return FG_OK;}
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    profile_command_end(c);
    VkResult vr;if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS){fg_status status=vk_error(err,"end batch command",vr);fg_error ignored={0};fg_vk_abort(c,&ignored);return status;}
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};
    if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS){fg_status status=vk_error(err,"submit batch command",vr);fg_error ignored={0};fg_vk_abort(c,&ignored);return status;}c->counters.submissions++;
    if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS){fg_status status=vk_error(err,"wait for batch command",vr);vkDeviceWaitIdle(c->device);c->batch_depth=0;c->batch_set_count=0;c->batch_has_dispatch=false;return status;}
    fg_status status=profile_resolve(c,err);
    c->batch_depth=0;c->batch_set_count=0;c->batch_has_dispatch=false;return status;
}
fg_status fg_vk_abort(fg_vk_context *c,fg_error *err){
    if(!c||!c->batch_depth){fg_error_set(err,FG_ERR_ARGUMENT,"not in batch");return FG_ERR_ARGUMENT;}
    VkResult vr=vkResetCommandBuffer(c->command,0);
    c->batch_depth=0;c->batch_set_count=0;c->batch_has_dispatch=false;
    return vr==VK_SUCCESS?FG_OK:vk_error(err,"abort batch command",vr);
}
bool fg_vk_batch_active(const fg_vk_context *c){return c&&c->batch_depth>0;}

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
    bool gate_cooked=gate_weights&&gate_weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED,up_cooked=up_weights&&up_weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED,down_cooked=down_weights&&down_weights->format==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED;
    if(!out||!c||c->batch_depth||!gate_block||!up_block||!down_block||!hidden_width||hidden_width%256u||!mid_width||mid_width%32u||!weight_experts||!slots||gate_row>UINT32_MAX||up_row>UINT32_MAX||down_row>UINT32_MAX||gate_expert_stride<gate_row*mid_width||up_expert_stride<up_row*mid_width||down_expert_stride<down_row*hidden_width||!tensor_on_context(c,activation)||!tensor_on_context(c,tiles)||!tensor_on_context(c,gates)||!tensor_on_context(c,gate)||!tensor_on_context(c,up)||!tensor_on_context(c,mid)||!tensor_on_context(c,down)||!tensor_on_context(c,reduced)||!tensor_on_context(c,gate_weights)||!tensor_on_context(c,up_weights)||!tensor_on_context(c,down_weights)||!tensor_range(activation,0,q8_bytes)||!tensor_range(tiles,0,(uint64_t)slots*9u*4u)||!tensor_range(gates,0,(uint64_t)slots*4u)||!tensor_range(gate,0,(uint64_t)slots*mid_width*4u)||!tensor_range(up,0,(uint64_t)slots*mid_width*4u)||!tensor_range(mid,0,(uint64_t)slots*mid_width*4u)||!tensor_range(down,0,(uint64_t)slots*hidden_width*4u)||!tensor_range(reduced,0,(uint64_t)hidden_width*4u)||!tensor_range(gate_weights,0,(uint64_t)weight_experts*gate_expert_stride)||!tensor_range(up_weights,0,(uint64_t)weight_experts*up_expert_stride)||!tensor_range(down_weights,0,(uint64_t)weight_experts*down_expert_stride)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fixed expert graph");return FG_ERR_ARGUMENT;}
    fg_vk_tensor *scratch[]={activation,tiles,gates,gate,up,mid,down,reduced};
    fg_status scratch_status=validate_expert_graph_scratch(scratch,err);
    if(scratch_status!=FG_OK)return scratch_status;
    fg_vk_expert_graph *graph=calloc(1,sizeof(*graph));if(!graph){fg_error_set(err,FG_ERR_OOM,"allocate fixed expert graph");return FG_ERR_OOM;}graph->context=c;
    VkDescriptorPoolSize pool_size={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=80u};VkDescriptorPoolCreateInfo pool_create={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=5u,.poolSizeCount=1u,.pPoolSizes=&pool_size};VkResult vr=vkCreateDescriptorPool(c->device,&pool_create,NULL,&graph->descriptor_pool);if(vr!=VK_SUCCESS){fg_vk_expert_graph_destroy(graph);return vk_error(err,"create expert graph descriptor pool",vr);}
    VkCommandBufferAllocateInfo command_allocate={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1u};if((vr=vkAllocateCommandBuffers(c->device,&command_allocate,&graph->command))!=VK_SUCCESS){fg_vk_expert_graph_destroy(graph);return vk_error(err,"allocate expert graph command",vr);}VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};if((vr=vkBeginCommandBuffer(graph->command,&begin))!=VK_SUCCESS){fg_vk_expert_graph_destroy(graph);return vk_error(err,"begin expert graph command",vr);}
    VkMemoryBarrier host={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(graph->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&host,0,NULL,0,NULL);
    struct{uint32_t out_dim,blocks,row_bytes,expert_stride,type,n_used,routed_pairs,packed_weights,reserved;}gate_push={mid_width,hidden_width/256u,(uint32_t)gate_row,gate_expert_stride,gate_type,slots,slots,0u,0u},up_push={mid_width,hidden_width/256u,(uint32_t)up_row,up_expert_stride,up_type,slots,slots,0u,0u};struct{uint32_t out_dim,blocks,tile_bytes,expert_stride,type,n_used,routed_pairs,packed_weights;}gate_cooked_push={mid_width,hidden_width/FG_QK8_K,(uint32_t)fg_k_quant_cooked_tile_bytes(hidden_width,gate_type),gate_expert_stride,gate_type,slots,slots,0u},up_cooked_push={mid_width,hidden_width/FG_QK8_K,(uint32_t)fg_k_quant_cooked_tile_bytes(hidden_width,up_type),up_expert_stride,up_type,slots,slots,0u};struct{uint32_t values;}swiglu_push={slots*mid_width};struct{uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;}down_push={hidden_width,mid_width,(uint32_t)down_row,down_expert_stride,slots,0u,0u};struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,routed_pairs,packed_weights;}down_cooked_push={hidden_width,mid_width,mid_width/FG_QK8_0,(uint32_t)fg_q5_1_cooked_tile_bytes(mid_width),down_expert_stride,slots,0u};struct{uint32_t width,selected_count,slot_count;}reduce_push={hidden_width,slots,slots};
    const fg_vk_tensor *gate_bindings[]={gate_weights,activation,tiles,gate},*up_bindings[]={up_weights,activation,tiles,up},*swiglu_bindings[]={gate,up,mid},*down_bindings[]={down_weights,tiles,mid,down},*reduce_bindings[]={down,gates,tiles,reduced};fg_vk_kernel *gate_kernel=gate_cooked?&c->kquant_cooked:&c->kquant,*up_kernel=up_cooked?&c->kquant_cooked:&c->kquant,*down_kernel=down_cooked?&c->moe_q5_1_cooked:down_type==7u?&c->moe_q5_1:&c->moe_q8_0;const void *gate_parameters=gate_cooked?(const void *)&gate_cooked_push:(const void *)&gate_push,*up_parameters=up_cooked?(const void *)&up_cooked_push:(const void *)&up_push,*down_parameters=down_cooked?(const void *)&down_cooked_push:(const void *)&down_push;fg_status status=expert_graph_dispatch(graph,gate_kernel,gate_bindings,gate_parameters,(mid_width+7u)/8u,slots,1u,false,err);if(status==FG_OK)status=expert_graph_dispatch(graph,up_kernel,up_bindings,up_parameters,(mid_width+7u)/8u,slots,1u,false,err);if(status==FG_OK)status=expert_graph_dispatch(graph,&c->swiglu,swiglu_bindings,&swiglu_push,(slots*mid_width+255u)/256u,1u,1u,true,err);if(status==FG_OK)status=expert_graph_dispatch(graph,down_kernel,down_bindings,down_parameters,(hidden_width+7u)/8u,slots,1u,true,err);if(status==FG_OK)status=expert_graph_dispatch(graph,&c->moe_reduce,reduce_bindings,&reduce_push,(hidden_width+255u)/256u,1u,1u,true,err);
    if(status==FG_OK){VkMemoryBarrier finish={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(graph->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&finish,0,NULL,0,NULL);if((vr=vkEndCommandBuffer(graph->command))!=VK_SUCCESS)status=vk_error(err,"end expert graph command",vr);}if(status!=FG_OK){fg_vk_expert_graph_destroy(graph);return status;}*out=graph;return FG_OK;
}

fg_status fg_vk_expert_graph_execute(fg_vk_expert_graph *graph,fg_error *err){if(!graph||!graph->context||graph->context->batch_depth||graph->context->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"fixed expert graph unavailable while batching or profiling");return FG_ERR_ARGUMENT;}fg_vk_context *c=graph->context;VkResult vr=vkResetFences(c->device,1,&c->fence);if(vr!=VK_SUCCESS)return vk_error(err,"reset expert graph fence",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1u,.pCommandBuffers=&graph->command};if((vr=vkQueueSubmit(c->queue,1u,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit expert graph",vr);c->counters.submissions++;c->counters.dispatches+=graph->dispatches;if((vr=vkWaitForFences(c->device,1u,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait for expert graph",vr);return FG_OK;}

fg_status fg_vk_quantize_q8_k(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){if(!c||!width||width%256u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,(uint64_t)(width/256u)*tokens*296u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_K quantization dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t width,blocks,tokens;}push={width,width/256u,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q8k,bindings,&push,width/256u,tokens,1,err);}
fg_status fg_vk_quantize_q8_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){uint32_t blocks=width/32u;uint64_t bytes=(uint64_t)blocks*34u*tokens;if(!c||!width||width%64u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 quantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,blocks,tokens;}push={width,blocks,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q8,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_quantize_q4_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){uint32_t blocks=width/32u;uint64_t bytes=(uint64_t)blocks*18u*tokens;if(!c||!width||width%64u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q4_0 quantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,blocks,tokens;}push={width,blocks,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q4,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_dequantize_iq4_nl(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t rows,uint32_t width,fg_error *err){uint64_t input_bytes=(uint64_t)rows*(width/32u)*18u;if(!c||!rows||!width||width%32u||!tensor_range(input,0,input_bytes)||!tensor_range(out,0,(uint64_t)rows*width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid IQ4_NL dequantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t rows,width;}push={rows,width};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->dequant_iq4nl,bindings,&push,(rows*width+255u)/256u,1,1,err);}
fg_status fg_vk_swiglu(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *gate,const fg_vk_tensor *up,uint32_t values,fg_error *err){if(!c||!values||!tensor_range(gate,0,(uint64_t)values*4u)||!tensor_range(up,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid SwiGLU dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gate,up,out};return dispatch(c,&c->swiglu,bindings,&values,(values+255u)/256u,1,1,err);}

fg_status fg_vk_dense_q8_0_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){if(w&&w->format==FG_VK_TENSOR_FORMAT_Q8_0_COOKED)return fg_vk_dense_q8_0_cooked(c,out,w,x,in,rows,tokens,scale,err);uint64_t row_bytes=(uint64_t)(in/32u)*34u;if(!c||in==0||in%32u||!tensor_range(w,0,row_bytes*rows)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,n_tok,blocks,row_bytes;float scale;} push={rows,tokens,in/32u,(uint32_t)row_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense,bindings,&push,rows,tokens,1,err);}
fg_status fg_vk_dense_q8_0_subgroup(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){uint64_t row_bytes=(uint64_t)(in/32u)*34u;if(!c||!w||w->format!=FG_VK_TENSOR_FORMAT_DEFAULT||in==0||in%32u||!tensor_range(w,0,row_bytes*rows)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid subgroup Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,row_bytes;float scale;}push={rows,tokens,in/32u,(uint32_t)row_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense_subgroup,bindings,&push,rows,tokens,1,err);}
fg_status fg_vk_dense_q8_0_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){uint32_t blocks=in/FG_QK8_0;uint64_t quant_offset=fg_align_up_u64(FG_Q8_0_COOK_ROWS*(uint64_t)blocks*sizeof(uint16_t),FG_Q8_0_COOK_ALIGNMENT),tile_bytes=fg_align_up_u64(quant_offset+FG_Q8_0_COOK_ROWS*(uint64_t)blocks*FG_QK8_0,FG_Q8_0_COOK_ALIGNMENT),tiles=((uint64_t)rows+FG_Q8_0_COOK_ROWS-1u)/FG_Q8_0_COOK_ROWS;if(!c||!in||in%FG_QK8_0||tile_bytes>UINT32_MAX||tiles>UINT64_MAX/tile_bytes||!tensor_range(w,0,tile_bytes*tiles)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,tile_bytes;float scale;}push={rows,tokens,blocks,(uint32_t)tile_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense_cooked,bindings,&push,(rows+3u)/4u,tokens,1,err);}
fg_status fg_vk_dense_q8_0_cooked_prefill(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){uint32_t blocks=in/FG_QK8_0;uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(in),matrix_bytes=fg_q8_0_cooked_matrix_bytes(in,rows);if(!c||!w||!blocks||!rows||tokens<2u||tile_bytes>UINT32_MAX||w->format!=FG_VK_TENSOR_FORMAT_Q8_0_COOKED||!tensor_range(w,0,matrix_bytes)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid token-tiled cooked Q8_0 prefill dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,tokens,blocks,tile_bytes;float scale;}push={rows,tokens,blocks,(uint32_t)tile_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense_cooked_tile,bindings,&push,(rows+63u)/64u,(tokens+31u)/32u,1u,err);}
fg_status fg_vk_dense_q8_0_cooked_split(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *partials,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,uint32_t splits,float scale,fg_error *err){uint32_t blocks=in/FG_QK8_0;uint64_t tile_bytes=fg_q8_0_cooked_tile_bytes(in),matrix_bytes=fg_q8_0_cooked_matrix_bytes(in,rows),values=(uint64_t)rows*tokens;if(!c||!blocks||!rows||!tokens||splits<2u||splits>16u||tile_bytes>UINT32_MAX||values>UINT32_MAX||!tensor_range(w,0,matrix_bytes)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(partials,0,values*splits*4u)||!tensor_range(out,0,values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid split cooked Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,tile_bytes,splits;float scale;}split_push={rows,tokens,blocks,(uint32_t)tile_bytes,splits,scale};const fg_vk_tensor *split_bindings[]={w,x,partials};fg_status status=dispatch(c,&c->dense_cooked_split,split_bindings,&split_push,(rows+3u)/4u,tokens*splits,1,err);struct{uint32_t values,splits;}reduce_push={(uint32_t)values,splits};const fg_vk_tensor *reduce_bindings[]={partials,out};if(status==FG_OK)status=dispatch(c,&c->dense_cooked_split_reduce,reduce_bindings,&reduce_push,((uint32_t)values+255u)/256u,1,1,err);return status;}
fg_status fg_vk_embedding_q8_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,uint32_t token,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){uint64_t row_bytes=(uint64_t)(width/32u)*34u;if(!c||!weights||weights->format!=FG_VK_TENSOR_FORMAT_DEFAULT||!width||width%32u||!rows||token>=rows||!copies||!tensor_range(weights,0,row_bytes*rows)||!tensor_range(out,0,(uint64_t)width*copies*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 embedding dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t token,width,rows,copies;}push={token,width,rows,copies};const fg_vk_tensor *bindings[]={weights,out};return dispatch(c,&c->embedding,bindings,&push,(width*copies+255u)/256u,1,1,err);}
fg_status fg_vk_embedding_q8_0_batch(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *token_ids,uint32_t token_count,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){uint64_t row_bytes=(uint64_t)(width/32u)*34u,total=(uint64_t)token_count*width*copies;if(!c||!weights||weights->format!=FG_VK_TENSOR_FORMAT_DEFAULT||!token_count||!width||width%32u||!rows||!copies||!tensor_range(weights,0,row_bytes*rows)||!tensor_range(token_ids,0,(uint64_t)token_count*4u)||!tensor_range(out,0,total*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched Q8_0 embedding dispatch");return FG_ERR_ARGUMENT;}const uint32_t *ids=fg_vk_tensor_map((fg_vk_tensor *)token_ids);for(uint32_t i=0;i<token_count;i++)if(ids[i]>=rows){fg_error_set(err,FG_ERR_FORMAT,"embedding token %u is outside vocabulary",i);return FG_ERR_FORMAT;}struct{uint32_t token_count,width,rows,copies;}push={token_count,width,rows,copies};const fg_vk_tensor *bindings[]={weights,token_ids,out};return dispatch(c,&c->embedding_batch,bindings,&push,(uint32_t)((total+255u)/256u),1,1,err);}
fg_status fg_vk_dense_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *input,uint32_t input_width,uint32_t output_width,uint32_t tokens,fg_error *err){if(!c||!input_width||!output_width||!tokens||!tensor_range(weights,0,(uint64_t)input_width*output_width*4u)||!tensor_range(input,0,(uint64_t)input_width*tokens*4u)||!tensor_range(out,0,(uint64_t)output_width*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid F32 dense dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t input_width,output_width,tokens;}push={input_width,output_width,tokens};const fg_vk_tensor *bindings[]={weights,input,out};return dispatch(c,&c->dense_f32,bindings,&push,output_width,tokens,1,err);}
fg_status fg_vk_dense_bf16_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *input,uint32_t input_width,uint32_t output_width,uint32_t tokens,fg_error *err){if(!c||!input_width||input_width%2u||!output_width||!tokens||!tensor_range(weights,0,(uint64_t)input_width*output_width*2u)||!tensor_range(input,0,(uint64_t)input_width*tokens*4u)||!tensor_range(out,0,(uint64_t)output_width*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid BF16 dense dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t input_width,output_width,tokens;}push={input_width,output_width,tokens};const fg_vk_tensor *bindings[]={weights,input,out};return dispatch(c,&c->dense_bf16,bindings,&push,output_width,tokens,1,err);}
fg_status fg_vk_silu_scaled(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t values,float input_scale,fg_error *err){if(!c||!values||!tensor_range(input,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid scaled SiLU dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t values;float scale;}push={values,input_scale};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->silu_scaled,bindings,&push,(values+255u)/256u,1,1,err);}
fg_status fg_vk_group_rms_norm(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *x,const fg_vk_tensor *w,uint32_t width,uint32_t groups,uint32_t tokens,float epsilon,fg_error *err){uint64_t values=(uint64_t)width*groups*tokens;if(!c||!width||!groups||!tokens||!tensor_range(x,0,values*4u)||!tensor_range(w,0,(uint64_t)width*groups*4u)||!tensor_range(out,0,values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped RMS dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t width,groups,tokens;float epsilon;} push={width,groups,tokens,epsilon};const fg_vk_tensor *bindings[]={x,w,out};return dispatch(c,&c->rms,bindings,&push,groups,tokens,1,err);}
fg_status fg_vk_gr_mix(fg_vk_context *c,fg_vk_tensor *mixed,fg_vk_tensor *injection,const fg_vk_tensor *norm,const fg_vk_tensor *up,const fg_vk_tensor *inject,uint32_t hidden,uint32_t groups,uint32_t tokens,fg_error *err){uint64_t hyper=(uint64_t)hidden*groups*tokens;if(!c||!hidden||!groups||!tokens||!tensor_range(norm,0,hyper*4u)||!tensor_range(up,0,hyper*4u)||!tensor_range(inject,0,(uint64_t)groups*tokens*4u)||!tensor_range(mixed,0,(uint64_t)hidden*tokens*4u)||!tensor_range(injection,0,(uint64_t)groups*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t hidden,groups,tokens;} push={hidden,groups,tokens};const fg_vk_tensor *bindings[]={norm,up,inject,mixed,injection};return dispatch(c,&c->gr,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_hc_inject_partial(fg_vk_context *c,fg_vk_tensor *partials,const fg_vk_tensor *normalized,const fg_vk_tensor *weights,uint32_t hidden,uint32_t groups,uint32_t tokens,uint32_t pieces,fg_error *err){uint64_t width=(uint64_t)hidden*groups;if(!c||!hidden||!groups||groups>4u||!tokens||!pieces||pieces>64u||!tensor_range(normalized,0,width*tokens*4u)||!tensor_range(weights,0,width*groups*4u)||!tensor_range(partials,0,(uint64_t)tokens*pieces*groups*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid HC injection partial dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t hidden,groups,tokens,pieces;}push={hidden,groups,tokens,pieces};const fg_vk_tensor *bindings[]={normalized,weights,partials};return dispatch_impl(c,&c->hc_inject_partial,bindings,&push,pieces,tokens,1,false,err);}
fg_status fg_vk_gr_mix_partial(fg_vk_context *c,fg_vk_tensor *mixed,fg_vk_tensor *injection,const fg_vk_tensor *normalized,const fg_vk_tensor *up,const fg_vk_tensor *partials,uint32_t hidden,uint32_t groups,uint32_t tokens,uint32_t pieces,fg_error *err){uint64_t hyper=(uint64_t)hidden*groups*tokens;if(!c||!hidden||!groups||groups>4u||!tokens||!pieces||pieces>64u||!tensor_range(normalized,0,hyper*4u)||!tensor_range(up,0,hyper*4u)||!tensor_range(partials,0,(uint64_t)tokens*pieces*groups*4u)||!tensor_range(mixed,0,(uint64_t)hidden*tokens*4u)||!tensor_range(injection,0,(uint64_t)groups*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid gated residual partial dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t hidden,groups,tokens,pieces;}push={hidden,groups,tokens,pieces};const fg_vk_tensor *bindings[]={normalized,up,partials,mixed,injection};return dispatch(c,&c->gr_partial,bindings,&push,tokens,1,1,err);}
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
            c->batch_depth=1;c->batch_set_count=0;
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
            c->batch_depth=1;c->batch_set_count=0;
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
fg_status fg_vk_gdn_recurrent_prefill(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,uint32_t tokens,float epsilon,fg_error *err){uint64_t key_width=(uint64_t)key_heads*head_dim,value_width=(uint64_t)value_heads*head_dim,qkv_width=2u*key_width+value_width,state_values=(uint64_t)value_heads*head_dim*head_dim;if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tokens||!tensor_range(qkv,0,qkv_width*tokens*4u)||!tensor_range(z,0,value_width*tokens*4u)||!tensor_range(alpha,0,(uint64_t)value_heads*tokens*4u)||!tensor_range(beta,0,(uint64_t)value_heads*tokens*4u)||!tensor_range(a_log,0,(uint64_t)value_heads*4u)||!tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||!tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||!tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN recurrent prefill dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t value_heads,key_heads,head_dim,tokens;float epsilon;}push={value_heads,key_heads,head_dim,tokens,epsilon};const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};return dispatch(c,&c->gdn_recurrent_prefill,bindings,&push,value_heads,1,1,err);}
fg_status fg_vk_gdn_recurrent_prefill_pipeline(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_decay,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t tokens,float epsilon,fg_error *err){
    const uint64_t qkv_values=(uint64_t)tokens*10240u;
    const uint64_t value_values=(uint64_t)tokens*6144u;
    const uint64_t control_values=(uint64_t)tokens*48u;
    const uint64_t state_values=48u*128u*128u;
    if(!c||!tokens||tokens>FG_VK_GDN_PIPELINE_PREFILL_MAX_TOKENS||
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
                     "invalid pipeline GDN recurrent prefill dispatch");
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
    struct{uint32_t tokens,blocks,block_base;} push={tokens,blocks,block_base};
    const fg_vk_tensor *bindings[]={query,keys,norm,positions,scores,ids};
    return dispatch(c,&c->qsa_score,bindings,&push,blocks,1,1,err);
}

fg_status fg_vk_qsa_index_score(
    fg_vk_context *c,fg_vk_tensor *scores,fg_vk_tensor *ids,
    const fg_vk_tensor *query,const fg_vk_tensor *keys,const fg_vk_tensor *norm,
    const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){
    return fg_vk_qsa_index_score_segment(c,scores,ids,query,keys,norm,positions,
                                         tokens,0u,err);
}
fg_status fg_vk_qsa_attention(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *records,const fg_vk_tensor *query,const fg_vk_tensor *gate,uint32_t selected_count,fg_error *err){if(!c||!selected_count||selected_count>2051u||!tensor_range(records,0,(uint64_t)selected_count*FG_Q38_QSA_TOKEN_RECORD_BYTES)||!tensor_range(query,0,6144u*4u)||!tensor_range(gate,0,6144u*4u)||!tensor_range(output,0,6144u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA attention dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={records,query,gate,output};return dispatch(c,&c->qsa_attention,bindings,&selected_count,24u,1,1,err);}

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

fg_status fg_vk_topk_reduce(fg_vk_context *c,fg_vk_tensor *out_scores,fg_vk_tensor *out_ids,const fg_vk_tensor *in_scores,const fg_vk_tensor *in_ids,uint32_t count,uint32_t *output_count,fg_error *err){uint32_t groups=(count+4095u)/4096u,produced=groups*512u;if(groups==1u&&count<512u)produced=count;if(!c||!count||!output_count||!tensor_range(in_scores,0,(uint64_t)count*4u)||!tensor_range(in_ids,0,(uint64_t)count*4u)||!tensor_range(out_scores,0,(uint64_t)produced*4u)||!tensor_range(out_ids,0,(uint64_t)produced*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid top-k reduction dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={in_scores,in_ids,out_scores,out_ids};fg_status status=dispatch(c,&c->topk,bindings,&count,groups,1,1,err);if(status==FG_OK)*output_count=produced;return status;}

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
fg_status fg_vk_expert_major_pack(fg_vk_context *c,fg_vk_tensor *tiles,const fg_vk_tensor *selected,uint32_t experts,uint32_t tokens,fg_error *err){uint64_t pairs=(uint64_t)tokens*FG_TOP_K,tile_bytes=pairs*FG_VK_PREFILL_TILE_WORDS*4u;if(!c||experts!=FG_EXPERT_COUNT||!tokens||pairs>UINT32_MAX||!tensor_range(selected,0,pairs*4u)||!tensor_range(tiles,0,tile_bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU expert-major packing dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t pairs,experts,tile_capacity;}push={(uint32_t)pairs,experts,(uint32_t)pairs};const fg_vk_tensor *bindings[]={selected,tiles};return dispatch(c,&c->expert_major_pack,bindings,&push,1u,1u,1u,err);}
fg_status fg_vk_decode_tile_schedule(fg_vk_context *c,fg_vk_tensor *tiles,const fg_vk_tensor *selected,fg_error *err){if(!c||!tensor_range(selected,0,FG_TOP_K*4u)||!tensor_range(tiles,0,FG_TOP_K*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU decode tile schedule dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={selected,tiles};return dispatch(c,&c->decode_tile_schedule,bindings,NULL,1u,1u,1u,err);}
fg_status fg_vk_argmax_reduce(fg_vk_context *c,fg_vk_tensor *out_scores,fg_vk_tensor *out_ids,const fg_vk_tensor *in_scores,const fg_vk_tensor *in_ids,uint32_t count,uint32_t *output_count,fg_error *err){uint32_t groups=(count+4095u)/4096u;if(!c||!count||!output_count||!tensor_range(in_scores,0,(uint64_t)count*4u)||!tensor_range(in_ids,0,(uint64_t)count*4u)||!tensor_range(out_scores,0,(uint64_t)groups*4u)||!tensor_range(out_ids,0,(uint64_t)groups*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid argmax reduction dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={in_scores,in_ids,out_scores,out_ids};fg_status status=dispatch(c,&c->argmax,bindings,&count,groups,1,1,err);if(status==FG_OK)*output_count=groups;return status;}
fg_status fg_vk_moe_q5_1_down(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,bool packed,uint32_t tile_count,fg_error *err){if(weights&&weights->format==FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED)return fg_vk_moe_q5_1_down_cooked_pairs(c,out,weights,tiles,input,output_width,input_width,expert_stride,used_experts,packed,tile_count,err);uint64_t row_bytes=(uint64_t)(input_width/32u)*24u;if(!c||!output_width||!input_width||input_width%32u||!tile_count||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q5_1 expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;} push={output_width,input_width,(uint32_t)row_bytes,expert_stride,used_experts,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q5_1,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_q5_1_down_cooked_pairs(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint64_t tile_bytes=fg_q5_1_cooked_tile_bytes(input_width),matrix_bytes=fg_q5_1_cooked_matrix_bytes(input_width,output_width);uint64_t weight_bytes=(uint64_t)expert_stride*(packed?tile_count:FG_EXPERTS_PER_RANK);if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!routed_pairs||!tile_count||expert_stride<matrix_bytes||!tensor_range(weights,0,weight_bytes)||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(input,0,(uint64_t)routed_pairs*input_width*4u)||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked Q5_1 expert dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,routed_pairs,packed_weights;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,routed_pairs,packed?1u:0u};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q5_1_cooked,bindings,&push,output_width/8u,tile_count,1,err);}
fg_status fg_vk_moe_q5_1_down_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,bool packed,uint32_t tile_count,fg_error *err){return fg_vk_moe_q5_1_down_cooked_pairs(c,out,weights,tiles,input,output_width,input_width,expert_stride,FG_TOP_K,packed,tile_count,err);}
fg_status fg_vk_moe_q8_0_down(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,bool packed,uint32_t tile_count,fg_error *err){uint64_t row_bytes=(uint64_t)(input_width/32u)*34u;if(!c||!output_width||!input_width||input_width%32u||!tile_count||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;} push={output_width,input_width,(uint32_t)row_bytes,expert_stride,used_experts,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q8_0,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_reduce(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *down,const fg_vk_tensor *gates,const fg_vk_tensor *tiles,uint32_t output_width,uint32_t selected_count,uint32_t slot_count,fg_error *err){if(!c||!output_width||!selected_count||selected_count>slot_count||!tensor_range(down,0,(uint64_t)slot_count*output_width*4u)||!tensor_range(gates,0,(uint64_t)selected_count*4u)||!tensor_range(tiles,0,(uint64_t)selected_count*9u*4u)||!tensor_range(out,0,(uint64_t)output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid MoE reduction dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,selected_count,slot_count;}push={output_width,selected_count,slot_count};const fg_vk_tensor *bindings[]={down,gates,tiles,out};return dispatch(c,&c->moe_reduce,bindings,&push,(output_width+255u)/256u,1,1,err);}
fg_status fg_vk_moe_kquant(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){if(weights&&weights->format==FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED)return fg_vk_moe_kquant_cooked_pairs(c,out,weights,activation,tiles,type,output_width,input_width,expert_stride,used_experts,routed_pairs,packed,tile_count,err);uint32_t block_bytes=type==12u?144u:type==13u?176u:0u;uint64_t row_bytes=(uint64_t)(input_width/256u)*block_bytes;if(!c||!block_bytes||!output_width||!input_width||input_width%256u||!used_experts||!routed_pairs||!tile_count||row_bytes>UINT32_MAX||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(activation,0,(uint64_t)((input_width/256u)*296u)*(routed_pairs/used_experts+(routed_pairs%used_experts?1u:0u)))||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q4_K/Q5_K expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,blocks,row_bytes,expert_stride,type,n_used,routed_pairs,packed_weights,reserved;} push={output_width,input_width/256u,(uint32_t)row_bytes,expert_stride,type,used_experts,routed_pairs,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,activation,tiles,out};return dispatch(c,&c->kquant,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_kquant_cooked_pairs(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint64_t tile_bytes=fg_k_quant_cooked_tile_bytes(input_width,type),matrix_bytes=fg_k_quant_cooked_matrix_bytes(input_width,output_width,type),weight_bytes=(uint64_t)expert_stride*(packed?tile_count:FG_EXPERTS_PER_RANK),activation_bytes=(uint64_t)(input_width/FG_QK8_K)*FG_Q8_K_BLOCK_BYTES*((routed_pairs+used_experts-1u)/used_experts);if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!used_experts||!routed_pairs||!tile_count||expert_stride<matrix_bytes||!tensor_range(weights,0,weight_bytes)||!tensor_range(activation,0,activation_bytes)||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid cooked Q4_K/Q5_K expert dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,blocks,tile_bytes,expert_stride,type,n_used,routed_pairs,packed_weights;}push={output_width,input_width/FG_QK8_K,(uint32_t)tile_bytes,expert_stride,type,used_experts,routed_pairs,packed?1u:0u};const fg_vk_tensor *bindings[]={weights,activation,tiles,out};return dispatch(c,&c->kquant_cooked,bindings,&push,output_width/8u,tile_count,1,err);}
fg_status fg_vk_moe_kquant_cooked(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,bool packed,uint32_t tile_count,fg_error *err){return fg_vk_moe_kquant_cooked_pairs(c,out,weights,activation,tiles,type,output_width,input_width,expert_stride,FG_TOP_K,FG_TOP_K,packed,tile_count,err);}
fg_status fg_vk_moe_kquant_cooked_grouped(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t weight_experts,uint32_t tokens,fg_error *err){uint64_t matrix_bytes=fg_k_quant_cooked_matrix_bytes(input_width,output_width,type),tile_bytes=fg_k_quant_cooked_tile_bytes(input_width,type),pairs=(uint64_t)tokens*FG_TOP_K,activation_bytes=(uint64_t)tokens*(input_width/FG_QK8_K)*FG_Q8_K_BLOCK_BYTES;if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!tokens||!weight_experts||expert_stride<matrix_bytes||!weights||weights->format!=FG_VK_TENSOR_FORMAT_K_QUANT_EXPERT_COOKED||!tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||!tensor_range(activation,0,activation_bytes)||!tensor_range(tiles,0,pairs*FG_VK_PREFILL_TILE_WORDS*4u)||!tensor_range(out,0,pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped cooked K-quant dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,blocks,tile_bytes,expert_stride,type,tokens;}push={output_width,input_width/FG_QK8_K,(uint32_t)tile_bytes,expert_stride,type,tokens};const fg_vk_tensor *bindings[]={weights,activation,tiles,out};return dispatch(c,&c->kquant_cooked_grouped,bindings,&push,(output_width+FG_K_QUANT_COOK_ROWS-1u)/FG_K_QUANT_COOK_ROWS,(uint32_t)pairs,1u,err);}
fg_status fg_vk_moe_q5_1_down_cooked_grouped(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t weight_experts,uint32_t tokens,fg_error *err){uint64_t matrix_bytes=fg_q5_1_cooked_matrix_bytes(input_width,output_width),tile_bytes=fg_q5_1_cooked_tile_bytes(input_width),pairs=(uint64_t)tokens*FG_TOP_K;if(!c||!matrix_bytes||tile_bytes>UINT32_MAX||!tokens||!weight_experts||expert_stride<matrix_bytes||!weights||weights->format!=FG_VK_TENSOR_FORMAT_Q5_1_EXPERT_COOKED||!tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||!tensor_range(tiles,0,pairs*FG_VK_PREFILL_TILE_WORDS*4u)||!tensor_range(input,0,pairs*input_width*4u)||!tensor_range(out,0,pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped cooked Q5_1 down dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,in_dim,blocks,tile_bytes,expert_stride,tokens;}push={output_width,input_width,input_width/FG_QK8_0,(uint32_t)tile_bytes,expert_stride,tokens};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->q5_1_cooked_grouped,bindings,&push,(output_width+7u)/8u,(uint32_t)pairs,1u,err);}
fg_status fg_vk_moe_q8_0_down_grouped(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t weight_experts,uint32_t tokens,fg_error *err){uint64_t row_bytes=(uint64_t)(input_width/FG_QK8_0)*FG_Q8_0_BLOCK_BYTES,matrix_bytes=row_bytes*output_width,pairs=(uint64_t)tokens*FG_TOP_K;if(!c||!input_width||input_width%FG_QK8_0||!output_width||!tokens||!weight_experts||expert_stride<matrix_bytes||!weights||weights->format!=FG_VK_TENSOR_FORMAT_DEFAULT||row_bytes>UINT32_MAX||!tensor_range(weights,0,(uint64_t)weight_experts*expert_stride)||!tensor_range(tiles,0,pairs*FG_VK_PREFILL_TILE_WORDS*4u)||!tensor_range(input,0,pairs*input_width*4u)||!tensor_range(out,0,pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid grouped Q8_0 down dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,in_dim,row_bytes,expert_stride,tokens;}push={output_width,input_width,(uint32_t)row_bytes,expert_stride,tokens};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->q8_0_grouped,bindings,&push,(output_width+7u)/8u,(uint32_t)pairs,1u,err);}
fg_status fg_vk_moe_prefill_reduce(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *expert_output,const fg_vk_tensor *gates,const fg_vk_tensor *shared_output,const fg_vk_tensor *shared_logit,uint32_t width,uint32_t tokens,fg_error *err){uint64_t pairs=(uint64_t)tokens*FG_TOP_K;if(!c||!width||!tokens||!tensor_range(expert_output,0,pairs*width*4u)||!tensor_range(gates,0,pairs*4u)||!tensor_range(shared_output,0,(uint64_t)tokens*width*4u)||!tensor_range(shared_logit,0,(uint64_t)tokens*4u)||!tensor_range(out,0,(uint64_t)tokens*width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GPU prefill gated reduction");return FG_ERR_ARGUMENT;}struct{uint32_t width,tokens;}push={width,tokens};const fg_vk_tensor *bindings[]={expert_output,gates,shared_output,shared_logit,out};return dispatch(c,&c->moe_prefill_reduce,bindings,&push,(width+255u)/256u,tokens,1u,err);}

/* -------- Decomposition benchmark: stream / dequant / dot-no-reduce / full -------- */

static double bench_run_timestamped(fg_vk_context *c,fg_vk_kernel *kernel,
    const fg_vk_tensor *w,const fg_vk_tensor *x,fg_vk_tensor **ys,uint32_t n,
    const void *push,uint32_t gx,uint32_t gy,
    VkQueryPool qpool,uint32_t ts_base,float ts_period,fg_error *err)
{
    vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkResult vr=vkBeginCommandBuffer(c->command,&begin);if(vr!=VK_SUCCESS)return -1.0;
    vkCmdResetQueryPool(c->command,qpool,ts_base,n*2+2);
    VkMemoryBarrier hd={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&hd,0,NULL,0,NULL);
    c->batch_depth=1;c->batch_set_count=0;
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
            times[k]=bench_run_timestamped(c,kernels[k],w,x,ys,DEC_N,&push,out_dim,1,qpool,ts_off,ts_period,err);
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
            times[k]=bench_run_timestamped(c,kernels[k],w,x,ys,ABC_N,&push,out_dim,1,qpool,ts_off,ts_period,err);
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
