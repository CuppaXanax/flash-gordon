#include "fg_vk.h"
#include "fg_q38_schema.h"

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct fg_vk_allocation {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    uint64_t bytes;
    uint32_t references;
} fg_vk_allocation;

struct fg_vk_tensor {fg_vk_context *context;fg_vk_allocation *allocation;uint64_t offset,bytes;};
typedef struct fg_vk_kernel {const char *file;uint32_t bindings,push_bytes;VkDescriptorSetLayout set_layout;VkPipelineLayout layout;VkPipeline pipeline;} fg_vk_kernel;
typedef struct fg_vk_profile_dispatch {const char *scope,*name;uint32_t begin_query,end_query;} fg_vk_profile_dispatch;

#define FG_VK_PROFILE_MAX_DISPATCHES 256u
#define FG_VK_PROFILE_QUERY_COUNT (2u+2u*FG_VK_PROFILE_MAX_DISPATCHES)

struct fg_vk_context {
    VkInstance instance;VkPhysicalDevice physical;VkDevice device;VkQueue queue;uint32_t queue_family;
    VkCommandPool command_pool;VkCommandBuffer command;VkFence fence;VkDescriptorPool descriptor_pool;VkPipelineCache pipeline_cache;
    VkPhysicalDeviceMemoryProperties memory;char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t batch_depth;uint32_t batch_set_count;VkDescriptorSet batch_sets[256];
    VkQueryPool profile_query_pool;float timestamp_period;uint32_t timestamp_valid_bits;
    bool profile_active,profile_overflow;const char *profile_scope;uint32_t profile_query_count,profile_dispatch_count;
    fg_vk_counters counters;
    fg_vk_profile_dispatch profile_dispatches[FG_VK_PROFILE_MAX_DISPATCHES];fg_vk_profile profile;
    fg_vk_kernel quant_q8k,quant_q8,quant_q4,dequant_iq4nl,embedding,embedding_batch,swiglu,silu_scaled,dense,dense_f32,dense_bf16,rms,gr,hc_inject_partial,gr_partial,hc_finalize,gr_write,ple_gate,ple_gate_prefill,ple_conv,ple_conv_prefill,add,gdn_conv,gdn_conv_prefill,gdn_recurrent,gdn_recurrent_algebraic,gdn_recurrent_prefill,qsa_prepare,qsa_prepare_prefill,qsa_index_prepare,qsa_index_prepare_prefill,qsa_record_commit,qsa_record_gather,qsa_score,qsa_attention,topk,moe_q5_1,moe_q8_0,moe_reduce,moe_gate_up_swiglu,kquant;
    /* Decomposition benchmark kernels */
    fg_vk_kernel bench_stream,bench_dequant,bench_dot_nored;
    fg_vk_kernel bench_stream_wide,bench_stream_vec;
};

static fg_status vk_error(fg_error *err,const char *operation,VkResult result){fg_error_set(err,FG_ERR_UNAVAILABLE,"%s: Vulkan result %d",operation,(int)result);return FG_ERR_UNAVAILABLE;}
static bool tensor_range(const fg_vk_tensor *tensor,uint64_t offset,uint64_t bytes){return tensor&&offset<=tensor->bytes&&bytes<=tensor->bytes-offset;}

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
    if(kernel->pipeline)vkDestroyPipeline(context->device,kernel->pipeline,NULL);
    if(kernel->layout)vkDestroyPipelineLayout(context->device,kernel->layout,NULL);
    if(kernel->set_layout)vkDestroyDescriptorSetLayout(context->device,kernel->set_layout,NULL);
    kernel->pipeline=VK_NULL_HANDLE;kernel->layout=VK_NULL_HANDLE;kernel->set_layout=VK_NULL_HANDLE;
}

static fg_status create_kernel(fg_vk_context *context,fg_vk_kernel *kernel,fg_error *err){
    if(kernel->pipeline)return FG_OK;
    uint32_t *code=NULL;size_t code_bytes=0;fg_status status=read_shader(kernel->file,&code,&code_bytes,err);if(status!=FG_OK)return status;
    VkShaderModule module=VK_NULL_HANDLE;VkShaderModuleCreateInfo sm={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=code_bytes,.pCode=code};VkResult vr=vkCreateShaderModule(context->device,&sm,NULL,&module);free(code);if(vr!=VK_SUCCESS)return vk_error(err,"create shader module",vr);
    VkDescriptorSetLayoutBinding binding[16];if(kernel->bindings>16u){vkDestroyShaderModule(context->device,module,NULL);fg_error_set(err,FG_ERR_LIMIT,"Vulkan kernel has too many bindings");return FG_ERR_LIMIT;}for(uint32_t i=0;i<kernel->bindings;i++)binding[i]=(VkDescriptorSetLayoutBinding){.binding=i,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo ds={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=kernel->bindings,.pBindings=binding};
    vr=vkCreateDescriptorSetLayout(context->device,&ds,NULL,&kernel->set_layout);if(vr!=VK_SUCCESS){vkDestroyShaderModule(context->device,module,NULL);return vk_error(err,"create descriptor layout",vr);}
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
    if(best<0){fg_vk_close(c);fg_error_set(err,FG_ERR_UNAVAILABLE,"no Vulkan compute queue");return FG_ERR_UNAVAILABLE;}c->queue_family=family;VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(c->physical,&properties);memcpy(c->device_name,properties.deviceName,sizeof(c->device_name));c->timestamp_period=properties.limits.timestampPeriod;vkGetPhysicalDeviceMemoryProperties(c->physical,&c->memory);
    float priority=1.0f;VkDeviceQueueCreateInfo qc={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};VkDeviceCreateInfo dc={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qc};vr=vkCreateDevice(c->physical,&dc,NULL,&c->device);if(vr!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create Vulkan device",vr);}vkGetDeviceQueue(c->device,family,0,&c->queue);
    VkCommandPoolCreateInfo pc={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=family};if((vr=vkCreateCommandPool(c->device,&pc,NULL,&c->command_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create command pool",vr);}VkCommandBufferAllocateInfo ca={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};if((vr=vkAllocateCommandBuffers(c->device,&ca,&c->command))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"allocate command buffer",vr);}
    VkFenceCreateInfo fc={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};if((vr=vkCreateFence(c->device,&fc,NULL,&c->fence))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create fence",vr);}VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=512};VkDescriptorPoolCreateInfo dp={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,.maxSets=64,.poolSizeCount=1,.pPoolSizes=&ps};if((vr=vkCreateDescriptorPool(c->device,&dp,NULL,&c->descriptor_pool))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create descriptor pool",vr);}VkPipelineCacheCreateInfo pci={.sType=VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};if((vr=vkCreatePipelineCache(c->device,&pci,NULL,&c->pipeline_cache))!=VK_SUCCESS){fg_vk_close(c);return vk_error(err,"create pipeline cache",vr);}
    c->quant_q8k=(fg_vk_kernel){.file="fg_quantize_q8_k.spv",.bindings=2,.push_bytes=12};c->quant_q8=(fg_vk_kernel){.file="fg_quantize_q8_0.spv",.bindings=2,.push_bytes=12};c->quant_q4=(fg_vk_kernel){.file="fg_quantize_q4_0.spv",.bindings=2,.push_bytes=12};
    c->dequant_iq4nl=(fg_vk_kernel){.file="fg_dequantize_iq4_nl.spv",.bindings=2,.push_bytes=8};c->embedding=(fg_vk_kernel){.file="fg_embedding_q8_0.spv",.bindings=2,.push_bytes=16};c->embedding_batch=(fg_vk_kernel){.file="fg_embedding_q8_0_batch.spv",.bindings=3,.push_bytes=16};c->swiglu=(fg_vk_kernel){.file="fg_swiglu.spv",.bindings=3,.push_bytes=4};c->silu_scaled=(fg_vk_kernel){.file="fg_silu_scaled.spv",.bindings=2,.push_bytes=8};c->dense=(fg_vk_kernel){.file="fg_dense_q8_0_f32.spv",.bindings=3,.push_bytes=20};c->dense_f32=(fg_vk_kernel){.file="fg_dense_f32.spv",.bindings=3,.push_bytes=12};c->dense_bf16=(fg_vk_kernel){.file="fg_dense_bf16_f32.spv",.bindings=3,.push_bytes=12};
    c->rms=(fg_vk_kernel){.file="fg_group_rms_norm.spv",.bindings=3,.push_bytes=16};c->gr=(fg_vk_kernel){.file="fg_gr_mix.spv",.bindings=5,.push_bytes=12};c->hc_inject_partial=(fg_vk_kernel){.file="fg_hc_inject_partial.spv",.bindings=3,.push_bytes=16};c->gr_partial=(fg_vk_kernel){.file="fg_gr_mix_partial.spv",.bindings=5,.push_bytes=16};c->hc_finalize=(fg_vk_kernel){.file="fg_hc_finalize.spv",.bindings=3,.push_bytes=12};c->gr_write=(fg_vk_kernel){.file="fg_gr_write.spv",.bindings=4,.push_bytes=12};c->ple_gate=(fg_vk_kernel){.file="fg_ple_gate.spv",.bindings=4,.push_bytes=0};c->ple_gate_prefill=(fg_vk_kernel){.file="fg_ple_gate_prefill.spv",.bindings=4,.push_bytes=4};c->ple_conv=(fg_vk_kernel){.file="fg_ple_conv_decode.spv",.bindings=5,.push_bytes=0};c->ple_conv_prefill=(fg_vk_kernel){.file="fg_ple_conv_prefill.spv",.bindings=5,.push_bytes=4};c->add=(fg_vk_kernel){.file="fg_add_f32.spv",.bindings=3,.push_bytes=4};c->gdn_conv=(fg_vk_kernel){.file="fg_gdn_conv_decode.spv",.bindings=4,.push_bytes=4};c->gdn_conv_prefill=(fg_vk_kernel){.file="fg_gdn_conv_prefill.spv",.bindings=4,.push_bytes=8};c->gdn_recurrent=(fg_vk_kernel){.file="fg_gdn_recurrent_decode.spv",.bindings=9,.push_bytes=16};c->gdn_recurrent_algebraic=(fg_vk_kernel){.file="fg_gdn_recurrent_algebraic.spv",.bindings=9,.push_bytes=16};c->gdn_recurrent_prefill=(fg_vk_kernel){.file="fg_gdn_recurrent_prefill.spv",.bindings=9,.push_bytes=20};
    c->qsa_prepare=(fg_vk_kernel){.file="fg_qsa_prepare.spv",.bindings=8,.push_bytes=0};c->qsa_prepare_prefill=(fg_vk_kernel){.file="fg_qsa_prepare_prefill.spv",.bindings=8,.push_bytes=4};c->qsa_index_prepare=(fg_vk_kernel){.file="fg_qsa_index_prepare.spv",.bindings=4,.push_bytes=0};c->qsa_index_prepare_prefill=(fg_vk_kernel){.file="fg_qsa_index_prepare_prefill.spv",.bindings=4,.push_bytes=4};c->qsa_record_commit=(fg_vk_kernel){.file="fg_qsa_record_commit.spv",.bindings=6,.push_bytes=12};c->qsa_record_gather=(fg_vk_kernel){.file="fg_qsa_record_gather.spv",.bindings=3,.push_bytes=20};c->qsa_score=(fg_vk_kernel){.file="fg_qsa_index_score.spv",.bindings=6,.push_bytes=8};c->qsa_attention=(fg_vk_kernel){.file="fg_qsa_attention.spv",.bindings=4,.push_bytes=4};c->topk=(fg_vk_kernel){.file="fg_topk_reduce.spv",.bindings=4,.push_bytes=4};
    c->moe_q5_1=(fg_vk_kernel){.file="fg_moe_q5_1_down.spv",.bindings=4,.push_bytes=28};c->moe_q8_0=(fg_vk_kernel){.file="fg_moe_q8_0_down.spv",.bindings=4,.push_bytes=28};c->moe_reduce=(fg_vk_kernel){.file="fg_moe_reduce.spv",.bindings=4,.push_bytes=12};c->moe_gate_up_swiglu=(fg_vk_kernel){.file="fg_moe_gate_up_swiglu.spv",.bindings=5,.push_bytes=48};c->kquant=(fg_vk_kernel){.file="fg_moe_kquant.spv",.bindings=4,.push_bytes=36};
    c->bench_stream=(fg_vk_kernel){.file="fg_bench_stream.spv",.bindings=3,.push_bytes=20};c->bench_dequant=(fg_vk_kernel){.file="fg_bench_dequant.spv",.bindings=3,.push_bytes=20};c->bench_dot_nored=(fg_vk_kernel){.file="fg_bench_dot_nored.spv",.bindings=3,.push_bytes=20};
    c->bench_stream_wide=(fg_vk_kernel){.file="fg_bench_stream_wide.spv",.bindings=3,.push_bytes=20};c->bench_stream_vec=(fg_vk_kernel){.file="fg_bench_stream_vec.spv",.bindings=3,.push_bytes=20};
    *out=c;return FG_OK;
}

void fg_vk_close(fg_vk_context *c){if(!c)return;if(c->device)vkDeviceWaitIdle(c->device);destroy_kernel(c,&c->bench_stream_vec);destroy_kernel(c,&c->bench_stream_wide);destroy_kernel(c,&c->bench_dot_nored);destroy_kernel(c,&c->bench_dequant);destroy_kernel(c,&c->bench_stream);destroy_kernel(c,&c->kquant);destroy_kernel(c,&c->moe_gate_up_swiglu);destroy_kernel(c,&c->moe_reduce);destroy_kernel(c,&c->moe_q8_0);destroy_kernel(c,&c->moe_q5_1);destroy_kernel(c,&c->topk);destroy_kernel(c,&c->qsa_attention);destroy_kernel(c,&c->qsa_score);destroy_kernel(c,&c->qsa_record_gather);destroy_kernel(c,&c->qsa_record_commit);destroy_kernel(c,&c->qsa_index_prepare_prefill);destroy_kernel(c,&c->qsa_index_prepare);destroy_kernel(c,&c->qsa_prepare_prefill);destroy_kernel(c,&c->qsa_prepare);destroy_kernel(c,&c->gdn_recurrent_prefill);destroy_kernel(c,&c->gdn_recurrent_algebraic);destroy_kernel(c,&c->gdn_recurrent);destroy_kernel(c,&c->gdn_conv_prefill);destroy_kernel(c,&c->gdn_conv);destroy_kernel(c,&c->add);destroy_kernel(c,&c->ple_conv_prefill);destroy_kernel(c,&c->ple_conv);destroy_kernel(c,&c->ple_gate_prefill);destroy_kernel(c,&c->ple_gate);destroy_kernel(c,&c->gr_write);destroy_kernel(c,&c->hc_finalize);destroy_kernel(c,&c->gr_partial);destroy_kernel(c,&c->hc_inject_partial);destroy_kernel(c,&c->gr);destroy_kernel(c,&c->rms);destroy_kernel(c,&c->dense_bf16);destroy_kernel(c,&c->dense_f32);destroy_kernel(c,&c->dense);destroy_kernel(c,&c->silu_scaled);destroy_kernel(c,&c->swiglu);destroy_kernel(c,&c->embedding_batch);destroy_kernel(c,&c->embedding);destroy_kernel(c,&c->dequant_iq4nl);destroy_kernel(c,&c->quant_q4);destroy_kernel(c,&c->quant_q8);destroy_kernel(c,&c->quant_q8k);if(c->profile_query_pool)vkDestroyQueryPool(c->device,c->profile_query_pool,NULL);if(c->pipeline_cache)vkDestroyPipelineCache(c->device,c->pipeline_cache,NULL);if(c->descriptor_pool)vkDestroyDescriptorPool(c->device,c->descriptor_pool,NULL);if(c->fence)vkDestroyFence(c->device,c->fence,NULL);if(c->command_pool)vkDestroyCommandPool(c->device,c->command_pool,NULL);if(c->device)vkDestroyDevice(c->device,NULL);if(c->instance)vkDestroyInstance(c->instance,NULL);free(c);}
const char *fg_vk_device_name(const fg_vk_context *c){return c?c->device_name:"";}

fg_status fg_vk_profile_begin(fg_vk_context *c,fg_error *err){if(!c||c->batch_depth||c->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile begin");return FG_ERR_ARGUMENT;}if(!c->timestamp_valid_bits||c->timestamp_period<=0.0f){fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan compute queue has no timestamp support");return FG_ERR_UNAVAILABLE;}if(!c->profile_query_pool){VkQueryPoolCreateInfo create={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=FG_VK_PROFILE_QUERY_COUNT};VkResult vr=vkCreateQueryPool(c->device,&create,NULL,&c->profile_query_pool);if(vr!=VK_SUCCESS)return vk_error(err,"create timestamp query pool",vr);vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin timestamp query prime",vr);vkCmdResetQueryPool(c->command,c->profile_query_pool,0,FG_VK_PROFILE_QUERY_COUNT);if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS)return vk_error(err,"end timestamp query prime",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit timestamp query prime",vr);c->counters.submissions++;if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait timestamp query prime",vr);}memset(&c->profile,0,sizeof(c->profile));c->profile_scope="unscoped";c->profile_active=true;return FG_OK;}
fg_status fg_vk_profile_set_scope(fg_vk_context *c,const char *scope,fg_error *err){if(!c||!c->profile_active||!scope||!*scope){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile scope");return FG_ERR_ARGUMENT;}c->profile_scope=scope;return FG_OK;}
fg_status fg_vk_profile_end(fg_vk_context *c,fg_vk_profile *profile,fg_error *err){if(!c||!profile||c->batch_depth||!c->profile_active){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan profile end");return FG_ERR_ARGUMENT;}*profile=c->profile;c->profile_active=false;return FG_OK;}
bool fg_vk_profile_active(const fg_vk_context *c){return c&&c->profile_active;}
void fg_vk_get_counters(const fg_vk_context *c,fg_vk_counters *counters){if(!counters)return;if(c)*counters=c->counters;else memset(counters,0,sizeof(*counters));}

fg_status fg_vk_tensor_create(fg_vk_context *c,uint64_t bytes,fg_vk_tensor **out,fg_error *err){
    if(!c||!out||bytes==0){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor allocation");return FG_ERR_ARGUMENT;}*out=NULL;fg_vk_allocation *a=calloc(1,sizeof(*a));fg_vk_tensor *t=calloc(1,sizeof(*t));if(!a||!t){free(a);free(t);fg_error_set(err,FG_ERR_OOM,"allocate Vulkan tensor metadata");return FG_ERR_OOM;}
    VkBufferCreateInfo bc={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};VkResult vr=vkCreateBuffer(c->device,&bc,NULL,&a->buffer);if(vr!=VK_SUCCESS){free(a);free(t);return vk_error(err,"create Vulkan tensor buffer",vr);}VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(c->device,a->buffer,&requirements);uint32_t type=UINT32_MAX;VkMemoryPropertyFlags wanted=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;for(uint32_t i=0;i<c->memory.memoryTypeCount;i++)if((requirements.memoryTypeBits&(1u<<i))&&(c->memory.memoryTypes[i].propertyFlags&wanted)==wanted){if(type==UINT32_MAX||c->memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)type=i;}if(type==UINT32_MAX){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);fg_error_set(err,FG_ERR_UNAVAILABLE,"Vulkan device has no coherent host-visible memory");return FG_ERR_UNAVAILABLE;}VkMemoryAllocateInfo ma={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=requirements.size,.memoryTypeIndex=type};if((vr=vkAllocateMemory(c->device,&ma,NULL,&a->memory))!=VK_SUCCESS){vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"allocate Vulkan tensor memory",vr);}if((vr=vkBindBufferMemory(c->device,a->buffer,a->memory,0))!=VK_SUCCESS||(vr=vkMapMemory(c->device,a->memory,0,VK_WHOLE_SIZE,0,&a->mapped))!=VK_SUCCESS){if(a->mapped)vkUnmapMemory(c->device,a->memory);vkFreeMemory(c->device,a->memory,NULL);vkDestroyBuffer(c->device,a->buffer,NULL);free(a);free(t);return vk_error(err,"map Vulkan tensor memory",vr);}a->bytes=bytes;a->references=1;t->context=c;t->allocation=a;t->bytes=bytes;*out=t;return FG_OK;
}

fg_status fg_vk_tensor_view(fg_vk_tensor *base,uint64_t offset,uint64_t bytes,fg_vk_tensor **out,fg_error *err){if(!out||!tensor_range(base,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor view");return FG_ERR_ARGUMENT;}fg_vk_tensor *view=calloc(1,sizeof(*view));if(!view){fg_error_set(err,FG_ERR_OOM,"allocate Vulkan tensor view");return FG_ERR_OOM;}base->allocation->references++;view->context=base->context;view->allocation=base->allocation;view->offset=base->offset+offset;view->bytes=bytes;*out=view;return FG_OK;}
void fg_vk_tensor_destroy(fg_vk_tensor *t){if(!t)return;fg_vk_allocation *a=t->allocation;if(--a->references==0){vkUnmapMemory(t->context->device,a->memory);vkFreeMemory(t->context->device,a->memory,NULL);vkDestroyBuffer(t->context->device,a->buffer,NULL);free(a);}free(t);}
uint64_t fg_vk_tensor_bytes(const fg_vk_tensor *t){return t?t->bytes:0;}void *fg_vk_tensor_map(fg_vk_tensor *t){return t?(uint8_t *)t->allocation->mapped+t->offset:NULL;}
fg_status fg_vk_tensor_write(fg_vk_tensor *t,uint64_t offset,const void *data,uint64_t bytes,fg_error *err){if(!data||!tensor_range(t,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor write");return FG_ERR_ARGUMENT;}memcpy((uint8_t *)t->allocation->mapped+t->offset+offset,data,(size_t)bytes);return FG_OK;}
fg_status fg_vk_tensor_read(const fg_vk_tensor *t,uint64_t offset,void *data,uint64_t bytes,fg_error *err){if(!data||!tensor_range(t,offset,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Vulkan tensor read");return FG_ERR_ARGUMENT;}memcpy(data,(const uint8_t *)t->allocation->mapped+t->offset+offset,(size_t)bytes);return FG_OK;}

static fg_status dispatch_impl(fg_vk_context *c,fg_vk_kernel *kernel,const fg_vk_tensor *const *tensors,const void *push,uint32_t gx,uint32_t gy,uint32_t gz,bool batch_barrier,fg_error *err){
    fg_status status=create_kernel(c,kernel,err);if(status!=FG_OK)return status;VkDescriptorSetAllocateInfo da={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=c->descriptor_pool,.descriptorSetCount=1,.pSetLayouts=&kernel->set_layout};VkDescriptorSet set;VkResult vr=vkAllocateDescriptorSets(c->device,&da,&set);if(vr!=VK_SUCCESS)return vk_error(err,"allocate descriptor set",vr);VkDescriptorBufferInfo info[16];VkWriteDescriptorSet write[16];for(uint32_t i=0;i<kernel->bindings;i++){info[i]=(VkDescriptorBufferInfo){.buffer=tensors[i]->allocation->buffer,.offset=tensors[i]->offset,.range=tensors[i]->bytes};write[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info[i]};}vkUpdateDescriptorSets(c->device,kernel->bindings,write,0,NULL);
    if(c->batch_depth){if(batch_barrier){VkMemoryBarrier bar={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&bar,0,NULL,0,NULL);}vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);bool profiled=profile_dispatch_begin(c,kernel);vkCmdDispatch(c->command,gx,gy,gz);c->counters.dispatches++;profile_dispatch_end(c,profiled);if(c->batch_set_count<256)c->batch_sets[c->batch_set_count++]=set;return FG_OK;}
    vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin compute command",vr);profile_command_begin(c);VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);vkCmdBindPipeline(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);vkCmdBindDescriptorSets(c->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,&set,0,NULL);if(kernel->push_bytes)vkCmdPushConstants(c->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,kernel->push_bytes,push);bool profiled=profile_dispatch_begin(c,kernel);vkCmdDispatch(c->command,gx,gy,gz);c->counters.dispatches++;profile_dispatch_end(c,profiled);VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);profile_command_end(c);if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS)return vk_error(err,"end compute command",vr);VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS)return vk_error(err,"submit compute command",vr);c->counters.submissions++;if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS)return vk_error(err,"wait for compute command",vr);status=profile_resolve(c,err);vkFreeDescriptorSets(c->device,c->descriptor_pool,1,&set);return status;
}

static fg_status dispatch(fg_vk_context *c,fg_vk_kernel *kernel,const fg_vk_tensor *const *tensors,const void *push,uint32_t gx,uint32_t gy,uint32_t gz,fg_error *err){return dispatch_impl(c,kernel,tensors,push,gx,gy,gz,true,err);}

fg_status fg_vk_begin(fg_vk_context *c,fg_error *err){
    if(!c){fg_error_set(err,FG_ERR_ARGUMENT,"null context");return FG_ERR_ARGUMENT;}
    if(c->batch_depth){c->batch_depth++;return FG_OK;}
    VkResult vr;vkResetFences(c->device,1,&c->fence);vkResetCommandBuffer(c->command,0);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if((vr=vkBeginCommandBuffer(c->command,&begin))!=VK_SUCCESS)return vk_error(err,"begin batch command",vr);
    profile_command_begin(c);
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);
    c->batch_depth=1;c->batch_set_count=0;return FG_OK;
}

fg_status fg_vk_end(fg_vk_context *c,fg_error *err){
    if(!c||!c->batch_depth){fg_error_set(err,FG_ERR_ARGUMENT,"not in batch");return FG_ERR_ARGUMENT;}
    if(c->batch_depth>1){c->batch_depth--;return FG_OK;}
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(c->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    profile_command_end(c);
    VkResult vr;if((vr=vkEndCommandBuffer(c->command))!=VK_SUCCESS){c->batch_depth=0;return vk_error(err,"end batch command",vr);}
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&c->command};
    if((vr=vkQueueSubmit(c->queue,1,&submit,c->fence))!=VK_SUCCESS){c->batch_depth=0;return vk_error(err,"submit batch command",vr);}c->counters.submissions++;
    if((vr=vkWaitForFences(c->device,1,&c->fence,VK_TRUE,UINT64_MAX))!=VK_SUCCESS){c->batch_depth=0;return vk_error(err,"wait for batch command",vr);}
    fg_status status=profile_resolve(c,err);
    for(uint32_t i=0;i<c->batch_set_count;i++)vkFreeDescriptorSets(c->device,c->descriptor_pool,1,&c->batch_sets[i]);
    c->batch_depth=0;c->batch_set_count=0;return status;
}
bool fg_vk_batch_active(const fg_vk_context *c){return c&&c->batch_depth>0;}

fg_status fg_vk_quantize_q8_k(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){if(!c||!width||width%256u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,(uint64_t)(width/256u)*tokens*296u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_K quantization dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t width,blocks,tokens;}push={width,width/256u,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q8k,bindings,&push,width/256u,tokens,1,err);}
fg_status fg_vk_quantize_q8_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){uint32_t blocks=width/32u;uint64_t bytes=(uint64_t)blocks*34u*tokens;if(!c||!width||width%64u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 quantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,blocks,tokens;}push={width,blocks,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q8,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_quantize_q4_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t width,uint32_t tokens,fg_error *err){uint32_t blocks=width/32u;uint64_t bytes=(uint64_t)blocks*18u*tokens;if(!c||!width||width%64u||!tokens||!tensor_range(input,0,(uint64_t)width*tokens*4u)||!tensor_range(out,0,bytes)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q4_0 quantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,blocks,tokens;}push={width,blocks,tokens};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->quant_q4,bindings,&push,tokens,1,1,err);}
fg_status fg_vk_dequantize_iq4_nl(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *input,uint32_t rows,uint32_t width,fg_error *err){uint64_t input_bytes=(uint64_t)rows*(width/32u)*18u;if(!c||!rows||!width||width%32u||!tensor_range(input,0,input_bytes)||!tensor_range(out,0,(uint64_t)rows*width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid IQ4_NL dequantization dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t rows,width;}push={rows,width};const fg_vk_tensor *bindings[]={input,out};return dispatch(c,&c->dequant_iq4nl,bindings,&push,(rows*width+255u)/256u,1,1,err);}
fg_status fg_vk_swiglu(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *gate,const fg_vk_tensor *up,uint32_t values,fg_error *err){if(!c||!values||!tensor_range(gate,0,(uint64_t)values*4u)||!tensor_range(up,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid SwiGLU dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gate,up,out};return dispatch(c,&c->swiglu,bindings,&values,(values+255u)/256u,1,1,err);}

fg_status fg_vk_dense_q8_0_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *w,const fg_vk_tensor *x,uint32_t in,uint32_t rows,uint32_t tokens,float scale,fg_error *err){uint64_t row_bytes=(uint64_t)(in/32u)*34u;if(!c||in==0||in%32u||!tensor_range(w,0,row_bytes*rows)||!tensor_range(x,0,(uint64_t)in*tokens*4u)||!tensor_range(out,0,(uint64_t)rows*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 dense dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,n_tok,blocks,row_bytes;float scale;} push={rows,tokens,in/32u,(uint32_t)row_bytes,scale};const fg_vk_tensor *bindings[]={w,x,out};return dispatch(c,&c->dense,bindings,&push,rows,tokens,1,err);}
fg_status fg_vk_embedding_q8_0(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,uint32_t token,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){uint64_t row_bytes=(uint64_t)(width/32u)*34u;if(!c||!width||width%32u||!rows||token>=rows||!copies||!tensor_range(weights,0,row_bytes*rows)||!tensor_range(out,0,(uint64_t)width*copies*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 embedding dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t token,width,rows,copies;}push={token,width,rows,copies};const fg_vk_tensor *bindings[]={weights,out};return dispatch(c,&c->embedding,bindings,&push,(width*copies+255u)/256u,1,1,err);}
fg_status fg_vk_embedding_q8_0_batch(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *token_ids,uint32_t token_count,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err){uint64_t row_bytes=(uint64_t)(width/32u)*34u,total=(uint64_t)token_count*width*copies;if(!c||!token_count||!width||width%32u||!rows||!copies||!tensor_range(weights,0,row_bytes*rows)||!tensor_range(token_ids,0,(uint64_t)token_count*4u)||!tensor_range(out,0,total*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid batched Q8_0 embedding dispatch");return FG_ERR_ARGUMENT;}const uint32_t *ids=fg_vk_tensor_map((fg_vk_tensor *)token_ids);for(uint32_t i=0;i<token_count;i++)if(ids[i]>=rows){fg_error_set(err,FG_ERR_FORMAT,"embedding token %u is outside vocabulary",i);return FG_ERR_FORMAT;}struct{uint32_t token_count,width,rows,copies;}push={token_count,width,rows,copies};const fg_vk_tensor *bindings[]={weights,token_ids,out};return dispatch(c,&c->embedding_batch,bindings,&push,(uint32_t)((total+255u)/256u),1,1,err);}
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
    VkDescriptorSetAllocateInfo da={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=c->descriptor_pool,.descriptorSetCount=1,.pSetLayouts=&kernel->set_layout};
    VkDescriptorSet set;VkResult vr=vkAllocateDescriptorSets(c->device,&da,&set);
    if(vr!=VK_SUCCESS)return vk_error(err,"bench alloc descriptor",vr);
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
    if(c->batch_set_count<256)c->batch_sets[c->batch_set_count++]=set;
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
            for(uint32_t i=0;i<c->batch_set_count;i++)vkFreeDescriptorSets(c->device,c->descriptor_pool,1,&c->batch_sets[i]);
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
            for(uint32_t i=0;i<c->batch_set_count;i++)vkFreeDescriptorSets(c->device,c->descriptor_pool,1,&c->batch_sets[i]);
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
fg_status fg_vk_gdn_project_decode(fg_vk_context *c,fg_vk_tensor *qkv,fg_vk_tensor *z,fg_vk_tensor *alpha,fg_vk_tensor *beta,const fg_vk_tensor *qkv_weight,const fg_vk_tensor *z_weight,const fg_vk_tensor *alpha_weight,const fg_vk_tensor *beta_weight,const fg_vk_tensor *hidden,fg_error *err){const uint32_t input_width=2560u,qkv_width=10240u,z_width=6144u,control_width=48u;uint64_t q8_row_bytes=(uint64_t)(input_width/32u)*34u;if(!c||!tensor_range(hidden,0,(uint64_t)input_width*4u)||!tensor_range(qkv_weight,0,q8_row_bytes*qkv_width)||!tensor_range(z_weight,0,q8_row_bytes*z_width)||!tensor_range(alpha_weight,0,(uint64_t)input_width*control_width*4u)||!tensor_range(beta_weight,0,(uint64_t)input_width*control_width*4u)||!tensor_range(qkv,0,(uint64_t)qkv_width*4u)||!tensor_range(z,0,(uint64_t)z_width*4u)||!tensor_range(alpha,0,(uint64_t)control_width*4u)||!tensor_range(beta,0,(uint64_t)control_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN sibling projection graph");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,n_tok,blocks,row_bytes;float scale;}qkv_push={qkv_width,1u,input_width/32u,(uint32_t)q8_row_bytes,1.0f},z_push={z_width,1u,input_width/32u,(uint32_t)q8_row_bytes,1.0f};struct{uint32_t input_width,output_width,tokens;}control_push={input_width,control_width,1u};const fg_vk_tensor *qkv_bindings[]={qkv_weight,hidden,qkv},*z_bindings[]={z_weight,hidden,z},*alpha_bindings[]={alpha_weight,hidden,alpha},*beta_bindings[]={beta_weight,hidden,beta};fg_status status=dispatch(c,&c->dense,qkv_bindings,&qkv_push,qkv_width,1,1,err);if(status==FG_OK)status=dispatch_impl(c,&c->dense,z_bindings,&z_push,z_width,1,1,false,err);if(status==FG_OK)status=dispatch_impl(c,&c->dense_f32,alpha_bindings,&control_push,control_width,1,1,false,err);if(status==FG_OK)status=dispatch_impl(c,&c->dense_f32,beta_bindings,&control_push,control_width,1,1,false,err);return status;}
fg_status fg_vk_gdn_recurrent_algebraic(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,float epsilon,fg_error *err){uint64_t key_width=(uint64_t)key_heads*head_dim,value_width=(uint64_t)value_heads*head_dim,state_values=(uint64_t)value_heads*head_dim*head_dim;if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tensor_range(qkv,0,(2u*key_width+value_width)*4u)||!tensor_range(z,0,value_width*4u)||!tensor_range(alpha,0,(uint64_t)value_heads*4u)||!tensor_range(beta,0,(uint64_t)value_heads*4u)||!tensor_range(a_log,0,(uint64_t)value_heads*4u)||!tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||!tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||!tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid algebraic GDN recurrent dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t value_heads,key_heads,head_dim;float epsilon;}push={value_heads,key_heads,head_dim,epsilon};const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};return dispatch(c,&c->gdn_recurrent_algebraic,bindings,&push,value_heads,1,1,err);}
fg_status fg_vk_ple_conv_decode(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *gated,const fg_vk_tensor *normalized,const fg_vk_tensor *weight,fg_error *err){if(!c||!tensor_range(gated,0,10240u*4u)||!tensor_range(normalized,0,10240u*4u)||!tensor_range(weight,0,10240u*4u*4u)||!tensor_range(state,0,10240u*9u*4u)||!tensor_range(out,0,10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE convolution dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gated,normalized,weight,state,out};return dispatch(c,&c->ple_conv,bindings,NULL,40u,1,1,err);}
fg_status fg_vk_ple_conv_prefill(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *gated,const fg_vk_tensor *normalized,const fg_vk_tensor *weight,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(gated,0,(uint64_t)tokens*10240u*4u)||!tensor_range(normalized,0,(uint64_t)tokens*10240u*4u)||!tensor_range(weight,0,10240u*4u*4u)||!tensor_range(state,0,10240u*9u*4u)||!tensor_range(out,0,(uint64_t)tokens*10240u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid PLE convolution prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={gated,normalized,weight,state,out};return dispatch(c,&c->ple_conv_prefill,bindings,&tokens,40u,1,1,err);}
fg_status fg_vk_add_f32(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *left,const fg_vk_tensor *right,uint32_t values,fg_error *err){if(!c||!values||!tensor_range(left,0,(uint64_t)values*4u)||!tensor_range(right,0,(uint64_t)values*4u)||!tensor_range(out,0,(uint64_t)values*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid F32 add dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={left,right,out};return dispatch(c,&c->add,bindings,&values,(values+255u)/256u,1,1,err);}
fg_status fg_vk_gdn_conv_decode(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *projection,const fg_vk_tensor *weight,uint32_t channels,fg_error *err){if(!c||!channels||!tensor_range(projection,0,(uint64_t)channels*4u)||!tensor_range(weight,0,(uint64_t)channels*4u*4u)||!tensor_range(state,0,(uint64_t)channels*4u*4u)||!tensor_range(out,0,(uint64_t)channels*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN convolution dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={projection,weight,state,out};return dispatch(c,&c->gdn_conv,bindings,&channels,(channels+255u)/256u,1,1,err);}
fg_status fg_vk_gdn_conv_prefill(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *projection,const fg_vk_tensor *weight,uint32_t channels,uint32_t tokens,fg_error *err){if(!c||!channels||!tokens||!tensor_range(projection,0,(uint64_t)channels*tokens*4u)||!tensor_range(weight,0,(uint64_t)channels*4u*4u)||!tensor_range(state,0,(uint64_t)channels*4u*4u)||!tensor_range(out,0,(uint64_t)channels*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN convolution prefill dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t channels,tokens;}push={channels,tokens};const fg_vk_tensor *bindings[]={projection,weight,state,out};return dispatch(c,&c->gdn_conv_prefill,bindings,&push,(channels+255u)/256u,1,1,err);}
fg_status fg_vk_gdn_recurrent_decode(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,float epsilon,fg_error *err){uint64_t key_width=(uint64_t)key_heads*head_dim,value_width=(uint64_t)value_heads*head_dim,state_values=(uint64_t)value_heads*head_dim*head_dim;if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tensor_range(qkv,0,(2u*key_width+value_width)*4u)||!tensor_range(z,0,value_width*4u)||!tensor_range(alpha,0,(uint64_t)value_heads*4u)||!tensor_range(beta,0,(uint64_t)value_heads*4u)||!tensor_range(a_log,0,(uint64_t)value_heads*4u)||!tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||!tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||!tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN recurrent dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t value_heads,key_heads,head_dim;float epsilon;}push={value_heads,key_heads,head_dim,epsilon};const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};return dispatch(c,&c->gdn_recurrent,bindings,&push,value_heads,1,1,err);}
fg_status fg_vk_gdn_recurrent_prefill(fg_vk_context *c,fg_vk_tensor *out,fg_vk_tensor *state,const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,uint32_t tokens,float epsilon,fg_error *err){uint64_t key_width=(uint64_t)key_heads*head_dim,value_width=(uint64_t)value_heads*head_dim,qkv_width=2u*key_width+value_width,state_values=(uint64_t)value_heads*head_dim*head_dim;if(!c||head_dim!=128u||!value_heads||!key_heads||value_heads%key_heads||!tokens||!tensor_range(qkv,0,qkv_width*tokens*4u)||!tensor_range(z,0,value_width*tokens*4u)||!tensor_range(alpha,0,(uint64_t)value_heads*tokens*4u)||!tensor_range(beta,0,(uint64_t)value_heads*tokens*4u)||!tensor_range(a_log,0,(uint64_t)value_heads*4u)||!tensor_range(dt_bias,0,(uint64_t)value_heads*4u)||!tensor_range(norm_weight,0,(uint64_t)head_dim*4u)||!tensor_range(state,0,state_values*4u)||!tensor_range(out,0,value_width*tokens*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid GDN recurrent prefill dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t value_heads,key_heads,head_dim,tokens;float epsilon;}push={value_heads,key_heads,head_dim,tokens,epsilon};const fg_vk_tensor *bindings[]={qkv,z,alpha,beta,a_log,dt_bias,norm_weight,state,out};return dispatch(c,&c->gdn_recurrent_prefill,bindings,&push,value_heads,1,1,err);}
fg_status fg_vk_qsa_prepare(fg_vk_context *c,fg_vk_tensor *query,fg_vk_tensor *gate,fg_vk_tensor *key,const fg_vk_tensor *raw_qg,const fg_vk_tensor *raw_key,const fg_vk_tensor *qnorm,const fg_vk_tensor *knorm,const fg_vk_tensor *position,fg_error *err){if(!c||!tensor_range(raw_qg,0,12288u*4u)||!tensor_range(raw_key,0,512u*4u)||!tensor_range(qnorm,0,256u*4u)||!tensor_range(knorm,0,256u*4u)||!tensor_range(position,0,3u*4u)||!tensor_range(query,0,6144u*4u)||!tensor_range(gate,0,6144u*4u)||!tensor_range(key,0,512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prepare dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_qg,raw_key,qnorm,knorm,position,query,gate,key};return dispatch(c,&c->qsa_prepare,bindings,NULL,24u,1,1,err);}
fg_status fg_vk_qsa_prepare_prefill(fg_vk_context *c,fg_vk_tensor *query,fg_vk_tensor *gate,fg_vk_tensor *key,const fg_vk_tensor *raw_qg,const fg_vk_tensor *raw_key,const fg_vk_tensor *qnorm,const fg_vk_tensor *knorm,const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(raw_qg,0,(uint64_t)tokens*12288u*4u)||!tensor_range(raw_key,0,(uint64_t)tokens*512u*4u)||!tensor_range(qnorm,0,256u*4u)||!tensor_range(knorm,0,256u*4u)||!tensor_range(positions,0,(uint64_t)tokens*3u*4u)||!tensor_range(query,0,(uint64_t)tokens*6144u*4u)||!tensor_range(gate,0,(uint64_t)tokens*6144u*4u)||!tensor_range(key,0,(uint64_t)tokens*512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA prepare prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_qg,raw_key,qnorm,knorm,positions,query,gate,key};return dispatch(c,&c->qsa_prepare_prefill,bindings,&tokens,tokens*24u,1,1,err);}
fg_status fg_vk_qsa_index_prepare(fg_vk_context *c,fg_vk_tensor *query,const fg_vk_tensor *raw_query,const fg_vk_tensor *norm,const fg_vk_tensor *position,fg_error *err){if(!c||!tensor_range(raw_query,0,512u*4u)||!tensor_range(norm,0,128u*4u)||!tensor_range(position,0,3u*4u)||!tensor_range(query,0,512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA index prepare dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_query,norm,position,query};return dispatch(c,&c->qsa_index_prepare,bindings,NULL,4u,1,1,err);}
fg_status fg_vk_qsa_index_prepare_prefill(fg_vk_context *c,fg_vk_tensor *query,const fg_vk_tensor *raw_query,const fg_vk_tensor *norm,const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){if(!c||!tokens||!tensor_range(raw_query,0,(uint64_t)tokens*512u*4u)||!tensor_range(norm,0,128u*4u)||!tensor_range(positions,0,(uint64_t)tokens*3u*4u)||!tensor_range(query,0,(uint64_t)tokens*512u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA index prepare prefill dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={raw_query,norm,positions,query};return dispatch(c,&c->qsa_index_prepare_prefill,bindings,&tokens,tokens*4u,1,1,err);}
fg_status fg_vk_qsa_record_commit(fg_vk_context *c,fg_vk_tensor *records,fg_vk_tensor *index_history,const fg_vk_tensor *key_q8,const fg_vk_tensor *value_q8,const fg_vk_tensor *index_key_q8,const fg_vk_tensor *position,uint32_t layer_slot,uint32_t token,uint32_t capacity,fg_error *err){uint64_t record_end=((uint64_t)layer_slot*capacity+token+1u)*FG_Q38_QSA_TOKEN_RECORD_BYTES,index_end=((uint64_t)layer_slot*capacity+token+1u)*FG_Q38_QSA_INDEX_KEY_BYTES;if(!c||!capacity||token>=capacity||!tensor_range(key_q8,0,FG_Q38_QSA_KEY_BYTES)||!tensor_range(value_q8,0,FG_Q38_QSA_VALUE_BYTES)||!tensor_range(index_key_q8,0,FG_Q38_QSA_INDEX_KEY_BYTES)||!tensor_range(position,0,FG_Q38_QSA_POSITION_BYTES)||!tensor_range(records,0,record_end)||!tensor_range(index_history,0,index_end)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA record commit");return FG_ERR_ARGUMENT;}struct{uint32_t layer_slot,token,capacity;}push={layer_slot,token,capacity};const fg_vk_tensor *bindings[]={key_q8,value_q8,index_key_q8,position,records,index_history};return dispatch(c,&c->qsa_record_commit,bindings,&push,(FG_Q38_QSA_TOKEN_RECORD_BYTES/4u+255u)/256u,1,1,err);}
fg_status fg_vk_qsa_record_gather(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *records,const fg_vk_tensor *block_ids,uint32_t layer_slot,uint32_t capacity,uint32_t block_count,uint32_t tail_start,uint32_t tail_count,fg_error *err){uint32_t selected=block_count*FG_Q38_QSA_COMPRESS_RATIO+tail_count;uint64_t record_end=(uint64_t)(layer_slot+1u)*capacity*FG_Q38_QSA_TOKEN_RECORD_BYTES;if(!c||!capacity||!block_count||block_count>FG_Q38_INDEX_BUDGET/FG_Q38_QSA_COMPRESS_RATIO||tail_count>=FG_Q38_QSA_COMPRESS_RATIO||tail_start>capacity||tail_count>capacity-tail_start||selected>FG_Q38_INDEX_BUDGET+FG_Q38_QSA_COMPRESS_RATIO-1u||!tensor_range(records,0,record_end)||!tensor_range(block_ids,0,(uint64_t)block_count*4u)||!tensor_range(out,0,(uint64_t)selected*FG_Q38_QSA_TOKEN_RECORD_BYTES)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA record gather");return FG_ERR_ARGUMENT;}struct{uint32_t layer_slot,capacity,block_count,tail_start,tail_count;}push={layer_slot,capacity,block_count,tail_start,tail_count};const fg_vk_tensor *bindings[]={records,block_ids,out};uint64_t words=(uint64_t)selected*(FG_Q38_QSA_TOKEN_RECORD_BYTES/4u);return dispatch(c,&c->qsa_record_gather,bindings,&push,(uint32_t)((words+255u)/256u),1,1,err);}
fg_status fg_vk_qsa_index_score(fg_vk_context *c,fg_vk_tensor *scores,fg_vk_tensor *ids,const fg_vk_tensor *query,const fg_vk_tensor *keys,const fg_vk_tensor *norm,const fg_vk_tensor *positions,uint32_t tokens,fg_error *err){uint32_t blocks=tokens/4u;if(!c||!blocks||!tensor_range(query,0,512u*4u)||!tensor_range(keys,0,(uint64_t)tokens*136u)||!tensor_range(norm,0,128u*4u)||!tensor_range(positions,0,(uint64_t)tokens*3u*4u)||!tensor_range(scores,0,(uint64_t)blocks*4u)||!tensor_range(ids,0,(uint64_t)blocks*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA index score dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t tokens,blocks;}push={tokens,blocks};const fg_vk_tensor *bindings[]={query,keys,norm,positions,scores,ids};return dispatch(c,&c->qsa_score,bindings,&push,blocks,1,1,err);}
fg_status fg_vk_qsa_attention(fg_vk_context *c,fg_vk_tensor *output,const fg_vk_tensor *records,const fg_vk_tensor *query,const fg_vk_tensor *gate,uint32_t selected_count,fg_error *err){if(!c||!selected_count||selected_count>2051u||!tensor_range(records,0,(uint64_t)selected_count*FG_Q38_QSA_TOKEN_RECORD_BYTES)||!tensor_range(query,0,6144u*4u)||!tensor_range(gate,0,6144u*4u)||!tensor_range(output,0,6144u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA attention dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={records,query,gate,output};return dispatch(c,&c->qsa_attention,bindings,&selected_count,24u,1,1,err);}
fg_status fg_vk_topk_reduce(fg_vk_context *c,fg_vk_tensor *out_scores,fg_vk_tensor *out_ids,const fg_vk_tensor *in_scores,const fg_vk_tensor *in_ids,uint32_t count,uint32_t *output_count,fg_error *err){uint32_t groups=(count+4095u)/4096u,produced=groups*512u;if(groups==1u&&count<512u)produced=count;if(!c||!count||!output_count||!tensor_range(in_scores,0,(uint64_t)count*4u)||!tensor_range(in_ids,0,(uint64_t)count*4u)||!tensor_range(out_scores,0,(uint64_t)produced*4u)||!tensor_range(out_ids,0,(uint64_t)produced*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid top-k reduction dispatch");return FG_ERR_ARGUMENT;}const fg_vk_tensor *bindings[]={in_scores,in_ids,out_scores,out_ids};fg_status status=dispatch(c,&c->topk,bindings,&count,groups,1,1,err);if(status==FG_OK)*output_count=produced;return status;}
fg_status fg_vk_moe_q5_1_down(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,bool packed,uint32_t tile_count,fg_error *err){uint64_t row_bytes=(uint64_t)(input_width/32u)*24u;if(!c||!output_width||!input_width||input_width%32u||!tile_count||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q5_1 expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;} push={output_width,input_width,(uint32_t)row_bytes,expert_stride,used_experts,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q5_1,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_q8_0_down(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,bool packed,uint32_t tile_count,fg_error *err){uint64_t row_bytes=(uint64_t)(input_width/32u)*34u;if(!c||!output_width||!input_width||input_width%32u||!tile_count||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q8_0 expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,in_dim,row_bytes,expert_stride,n_used,packed_weights,reserved;} push={output_width,input_width,(uint32_t)row_bytes,expert_stride,used_experts,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,tiles,input,out};return dispatch(c,&c->moe_q8_0,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_reduce(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *down,const fg_vk_tensor *gates,const fg_vk_tensor *tiles,uint32_t output_width,uint32_t selected_count,uint32_t slot_count,fg_error *err){if(!c||!output_width||!selected_count||selected_count>slot_count||!tensor_range(down,0,(uint64_t)slot_count*output_width*4u)||!tensor_range(gates,0,(uint64_t)selected_count*4u)||!tensor_range(tiles,0,(uint64_t)selected_count*9u*4u)||!tensor_range(out,0,(uint64_t)output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid MoE reduction dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t width,selected_count,slot_count;}push={output_width,selected_count,slot_count};const fg_vk_tensor *bindings[]={down,gates,tiles,out};return dispatch(c,&c->moe_reduce,bindings,&push,(output_width+255u)/256u,1,1,err);}
fg_status fg_vk_moe_gate_up_swiglu(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *gate_weights,const fg_vk_tensor *up_weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t gate_type,uint32_t up_type,uint32_t output_width,uint32_t input_width,uint32_t gate_expert_stride,uint32_t up_expert_stride,uint32_t used_experts,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint32_t gate_block_bytes=gate_type==12u?144u:gate_type==13u?176u:0u,up_block_bytes=up_type==12u?144u:up_type==13u?176u:0u;uint64_t gate_row_bytes=(uint64_t)(input_width/256u)*gate_block_bytes,up_row_bytes=(uint64_t)(input_width/256u)*up_block_bytes;if(!c||!gate_block_bytes||!up_block_bytes||!output_width||!input_width||input_width%256u||!used_experts||!routed_pairs||!tile_count||gate_row_bytes>UINT32_MAX||up_row_bytes>UINT32_MAX||gate_expert_stride<gate_row_bytes*output_width||up_expert_stride<up_row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(activation,0,(uint64_t)((input_width/256u)*296u)*(routed_pairs/used_experts+(routed_pairs%used_experts?1u:0u)))||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid fused gate/up SwiGLU dispatch");return FG_ERR_ARGUMENT;}struct{uint32_t out_dim,blocks,gate_row_bytes,gate_expert_stride,gate_type,up_row_bytes,up_expert_stride,up_type,n_used,routed_pairs,packed_weights,reserved;}push={output_width,input_width/256u,(uint32_t)gate_row_bytes,gate_expert_stride,gate_type,(uint32_t)up_row_bytes,up_expert_stride,up_type,used_experts,routed_pairs,packed?1u:0u,0u};const fg_vk_tensor *bindings[]={gate_weights,up_weights,activation,tiles,out};return dispatch(c,&c->moe_gate_up_swiglu,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}
fg_status fg_vk_moe_kquant(fg_vk_context *c,fg_vk_tensor *out,const fg_vk_tensor *weights,const fg_vk_tensor *activation,const fg_vk_tensor *tiles,uint32_t type,uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,uint32_t routed_pairs,bool packed,uint32_t tile_count,fg_error *err){uint32_t block_bytes=type==12u?144u:type==13u?176u:0u;uint64_t row_bytes=(uint64_t)(input_width/256u)*block_bytes;if(!c||!block_bytes||!output_width||!input_width||input_width%256u||!used_experts||!routed_pairs||!tile_count||row_bytes>UINT32_MAX||expert_stride<row_bytes*output_width||!tensor_range(tiles,0,(uint64_t)tile_count*9u*4u)||!tensor_range(activation,0,(uint64_t)((input_width/256u)*296u)*(routed_pairs/used_experts+(routed_pairs%used_experts?1u:0u)))||!tensor_range(out,0,(uint64_t)routed_pairs*output_width*4u)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid Q4_K/Q5_K expert dispatch");return FG_ERR_ARGUMENT;}struct {uint32_t out_dim,blocks,row_bytes,expert_stride,type,n_used,routed_pairs,packed_weights,reserved;} push={output_width,input_width/256u,(uint32_t)row_bytes,expert_stride,type,used_experts,routed_pairs,packed?1u:0u,0};const fg_vk_tensor *bindings[]={weights,activation,tiles,out};return dispatch(c,&c->kquant,bindings,&push,(output_width+7u)/8u,tile_count,1,err);}

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
    for(uint32_t i=0;i<c->batch_set_count;i++)vkFreeDescriptorSets(c->device,c->descriptor_pool,1,&c->batch_sets[i]);
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
