#include "fg_tower_vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

typedef struct tower_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t bytes;
} tower_buffer;

typedef struct tower_kernel {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout set_layout;
} tower_kernel;

typedef struct tower_block_buffers {
    tower_buffer ln1_weight;
    tower_buffer ln1_bias;
    tower_buffer ln2_weight;
    tower_buffer ln2_bias;
    tower_buffer qkv_weight;
    tower_buffer qkv_bias;
    tower_buffer attn_out_weight;
    tower_buffer attn_out_bias;
    tower_buffer ffn_up_weight;
    tower_buffer ffn_up_bias;
    tower_buffer ffn_down_weight;
    tower_buffer ffn_down_bias;
} tower_block_buffers;

struct fg_tower_vk {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command;
    VkFence fence;
    VkDescriptorPool descriptor_pool;
    tower_kernel matmul;
    tower_kernel bias;
    tower_kernel gelu;
    tower_kernel add;
    tower_kernel layernorm;
    tower_kernel rope;
    tower_kernel attention;
    tower_kernel position;
    char device_name[256];
};

struct fg_tower_vk_weights {
    tower_buffer patch_weight;
    tower_buffer patch_bias;
    tower_buffer position_weight;
    tower_block_buffers blocks[FG_TOWER_LAYERS];
    tower_buffer post_ln_weight;
    tower_buffer post_ln_bias;
    tower_buffer merger_fc1_weight;
    tower_buffer merger_fc1_bias;
    tower_buffer merger_fc2_weight;
    tower_buffer merger_fc2_bias;
};

typedef struct push_matmul {
    uint32_t tokens;
    uint32_t outputs;
    uint32_t width;
} push_matmul;

typedef struct push_bias {
    uint32_t count;
    uint32_t width;
} push_bias;

typedef struct push_gelu {
    uint32_t count;
} push_gelu;

typedef struct push_layernorm {
    uint32_t rows;
    uint32_t width;
    float epsilon;
} push_layernorm;

typedef struct push_rope {
    uint32_t tokens;
    uint32_t grid_width;
    uint32_t section;
} push_rope;

typedef struct push_attention {
    uint32_t tokens;
    float scale;
} push_attention;

typedef struct push_position {
    uint32_t tokens;
    uint32_t grid_width;
    uint32_t grid_height;
} push_position;

static double tower_now_ms(void){
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return (double)now.tv_sec*1000.0+(double)now.tv_nsec/1.0e6;
}

static fg_status tower_vk_error(fg_error *err,const char *what,VkResult result){
    fg_error_set(err,FG_ERR_IO,"tower Vulkan %s failed (VkResult %d)",what,(int)result);
    return FG_ERR_IO;
}

static fg_status tower_buffer_create(fg_tower_vk *tower,uint64_t bytes,tower_buffer *buffer,
                                     fg_error *err){
    memset(buffer,0,sizeof(*buffer));
    buffer->bytes=bytes?bytes:4u;
    VkBufferCreateInfo create={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size=buffer->bytes,
                              .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              .sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkResult result=vkCreateBuffer(tower->device,&create,NULL,&buffer->buffer);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"create buffer",result);
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(tower->device,buffer->buffer,&requirements);
    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(tower->physical,&properties);
    uint32_t memory_type=UINT32_MAX;
    for(uint32_t i=0;i<properties.memoryTypeCount;i++)
        if((requirements.memoryTypeBits&(1u<<i))&&
           (properties.memoryTypes[i].propertyFlags&
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
            memory_type=i;
            break;
        }
    if(memory_type==UINT32_MAX){
        vkDestroyBuffer(tower->device,buffer->buffer,NULL);
        buffer->buffer=VK_NULL_HANDLE;
        fg_error_set(err,FG_ERR_UNAVAILABLE,"tower Vulkan has no host coherent memory type");
        return FG_ERR_UNAVAILABLE;
    }
    VkMemoryAllocateInfo allocation={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize=requirements.size,
                                     .memoryTypeIndex=memory_type};
    result=vkAllocateMemory(tower->device,&allocation,NULL,&buffer->memory);
    if(result!=VK_SUCCESS){
        vkDestroyBuffer(tower->device,buffer->buffer,NULL);
        buffer->buffer=VK_NULL_HANDLE;
        return tower_vk_error(err,"allocate buffer memory",result);
    }
    result=vkBindBufferMemory(tower->device,buffer->buffer,buffer->memory,0);
    if(result!=VK_SUCCESS){
        vkFreeMemory(tower->device,buffer->memory,NULL);
        vkDestroyBuffer(tower->device,buffer->buffer,NULL);
        buffer->buffer=VK_NULL_HANDLE;
        buffer->memory=VK_NULL_HANDLE;
        return tower_vk_error(err,"bind buffer memory",result);
    }
    return FG_OK;
}

static void tower_buffer_destroy(fg_tower_vk *tower,tower_buffer *buffer){
    if(buffer->buffer)vkDestroyBuffer(tower->device,buffer->buffer,NULL);
    if(buffer->memory)vkFreeMemory(tower->device,buffer->memory,NULL);
    memset(buffer,0,sizeof(*buffer));
}

static fg_status tower_buffer_write(fg_tower_vk *tower,tower_buffer *buffer,const void *data,
                                    uint64_t bytes,fg_error *err){
    if(bytes>buffer->bytes){
        fg_error_set(err,FG_ERR_ARGUMENT,"tower buffer write exceeds allocation");
        return FG_ERR_ARGUMENT;
    }
    void *mapped=NULL;
    VkResult result=vkMapMemory(tower->device,buffer->memory,0,buffer->bytes,0,&mapped);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"map buffer",result);
    memcpy(mapped,data,(size_t)bytes);
    vkUnmapMemory(tower->device,buffer->memory);
    return FG_OK;
}

static fg_status tower_buffer_read(fg_tower_vk *tower,const tower_buffer *buffer,void *data,
                                   uint64_t bytes,fg_error *err){
    if(bytes>buffer->bytes){
        fg_error_set(err,FG_ERR_ARGUMENT,"tower buffer read exceeds allocation");
        return FG_ERR_ARGUMENT;
    }
    void *mapped=NULL;
    VkResult result=vkMapMemory(tower->device,buffer->memory,0,buffer->bytes,0,&mapped);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"map buffer for read",result);
    memcpy(data,mapped,(size_t)bytes);
    vkUnmapMemory(tower->device,buffer->memory);
    return FG_OK;
}

static fg_status tower_kernel_build(fg_tower_vk *tower,const char *file,uint32_t bindings,
                                    tower_kernel *kernel,fg_error *err){
    const char *directory=getenv("FG_SHADER_DIR");
    char path[1024];
    if(directory&&*directory)snprintf(path,sizeof(path),"%s/%s",directory,file);
    else snprintf(path,sizeof(path),"vulkan/%s",file);
    FILE *stream=fopen(path,"rb");
    if(!stream){
        fg_error_set(err,FG_ERR_IO,"open tower shader %s failed",path);
        return FG_ERR_IO;
    }
    fseek(stream,0,SEEK_END);
    long length=ftell(stream);
    if(length<=0||(length&3)!=0){
        fclose(stream);
        fg_error_set(err,FG_ERR_FORMAT,"invalid tower SPIR-V %s",path);
        return FG_ERR_FORMAT;
    }
    uint32_t *code=malloc((size_t)length);
    if(!code){
        fclose(stream);
        fg_error_set(err,FG_ERR_OOM,"allocate tower SPIR-V");
        return FG_ERR_OOM;
    }
    fseek(stream,0,SEEK_SET);
    if(fread(code,1,(size_t)length,stream)!=(size_t)length){
        free(code);
        fclose(stream);
        fg_error_set(err,FG_ERR_IO,"read tower SPIR-V %s",path);
        return FG_ERR_IO;
    }
    fclose(stream);
    VkDescriptorSetLayoutBinding entries[4];
    for(uint32_t i=0;i<bindings;i++)
        entries[i]=(VkDescriptorSetLayoutBinding){.binding=i,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,
            .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo set_info={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                              .bindingCount=bindings,
                                              .pBindings=entries};
    VkResult result=vkCreateDescriptorSetLayout(tower->device,&set_info,NULL,&kernel->set_layout);
    if(result!=VK_SUCCESS){
        free(code);
        return tower_vk_error(err,"create descriptor set layout",result);
    }
    VkPushConstantRange range={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=64u};
    VkPipelineLayoutCreateInfo layout_info={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount=1,
                                            .pSetLayouts=&kernel->set_layout,
                                            .pushConstantRangeCount=1,
                                            .pPushConstantRanges=&range};
    result=vkCreatePipelineLayout(tower->device,&layout_info,NULL,&kernel->layout);
    if(result!=VK_SUCCESS){
        free(code);
        return tower_vk_error(err,"create pipeline layout",result);
    }
    VkShaderModuleCreateInfo module_info={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                          .codeSize=(size_t)length,
                                          .pCode=code};
    VkShaderModule module=VK_NULL_HANDLE;
    result=vkCreateShaderModule(tower->device,&module_info,NULL,&module);
    free(code);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"create shader module",result);
    VkComputePipelineCreateInfo pipeline_info={
        .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage=VK_SHADER_STAGE_COMPUTE_BIT,
                .module=module,
                .pName="main"},
        .layout=kernel->layout};
    result=vkCreateComputePipelines(tower->device,VK_NULL_HANDLE,1,&pipeline_info,NULL,
                                    &kernel->pipeline);
    vkDestroyShaderModule(tower->device,module,NULL);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"create compute pipeline",result);
    return FG_OK;
}

fg_status fg_tower_vk_open(fg_tower_vk **out,fg_error *err){
    if(!out){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower Vulkan open");
        return FG_ERR_ARGUMENT;
    }
    fg_tower_vk *tower=calloc(1,sizeof(*tower));
    if(!tower){
        fg_error_set(err,FG_ERR_OOM,"allocate tower Vulkan context");
        return FG_ERR_OOM;
    }
    VkApplicationInfo application={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                   .pApplicationName="flash-gordon-tower",
                                   .apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo instance_info={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                        .pApplicationInfo=&application};
    VkResult result=vkCreateInstance(&instance_info,NULL,&tower->instance);
    if(result!=VK_SUCCESS){
        free(tower);
        return tower_vk_error(err,"create instance",result);
    }
    uint32_t physical_count=0;
    if(vkEnumeratePhysicalDevices(tower->instance,&physical_count,NULL)!=VK_SUCCESS||
       !physical_count){
        fg_tower_vk_close(tower);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"tower Vulkan found no physical device");
        return FG_ERR_UNAVAILABLE;
    }
    VkPhysicalDevice *physicals=calloc(physical_count,sizeof(*physicals));
    if(!physicals){
        fg_tower_vk_close(tower);
        fg_error_set(err,FG_ERR_OOM,"allocate physical device list");
        return FG_ERR_OOM;
    }
    vkEnumeratePhysicalDevices(tower->instance,&physical_count,physicals);
    bool found=false;
    for(uint32_t d=0;d<physical_count&&!found;d++){
        uint32_t family_count=0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[d],&family_count,NULL);
        VkQueueFamilyProperties *families=calloc(family_count,sizeof(*families));
        if(!families)continue;
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[d],&family_count,families);
        for(uint32_t f=0;f<family_count;f++)if(families[f].queueFlags&VK_QUEUE_COMPUTE_BIT){
            tower->physical=physicals[d];
            tower->queue_family=f;
            found=true;
            break;
        }
        free(families);
    }
    free(physicals);
    if(!found){
        fg_tower_vk_close(tower);
        fg_error_set(err,FG_ERR_UNAVAILABLE,"tower Vulkan found no compute queue");
        return FG_ERR_UNAVAILABLE;
    }
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(tower->physical,&properties);
    snprintf(tower->device_name,sizeof(tower->device_name),"%s",properties.deviceName);
    const float priority=1.0f;
    VkDeviceQueueCreateInfo queue_info={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .queueFamilyIndex=tower->queue_family,
                                        .queueCount=1,
                                        .pQueuePriorities=&priority};
    VkDeviceCreateInfo device_info={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                    .queueCreateInfoCount=1,
                                    .pQueueCreateInfos=&queue_info};
    result=vkCreateDevice(tower->physical,&device_info,NULL,&tower->device);
    if(result!=VK_SUCCESS){
        fg_tower_vk_close(tower);
        return tower_vk_error(err,"create device",result);
    }
    vkGetDeviceQueue(tower->device,tower->queue_family,0,&tower->queue);
    VkCommandPoolCreateInfo pool_info={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                       .queueFamilyIndex=tower->queue_family};
    result=vkCreateCommandPool(tower->device,&pool_info,NULL,&tower->command_pool);
    if(result!=VK_SUCCESS){
        fg_tower_vk_close(tower);
        return tower_vk_error(err,"create command pool",result);
    }
    VkCommandBufferAllocateInfo command_info={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                              .commandPool=tower->command_pool,
                                              .level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                              .commandBufferCount=1};
    result=vkAllocateCommandBuffers(tower->device,&command_info,&tower->command);
    if(result!=VK_SUCCESS){
        fg_tower_vk_close(tower);
        return tower_vk_error(err,"allocate command buffer",result);
    }
    VkFenceCreateInfo fence_info={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    result=vkCreateFence(tower->device,&fence_info,NULL,&tower->fence);
    if(result!=VK_SUCCESS){
        fg_tower_vk_close(tower);
        return tower_vk_error(err,"create fence",result);
    }
    VkDescriptorPoolSize pool_size={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .descriptorCount=65536u};
    VkDescriptorPoolCreateInfo descriptor_info={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                .maxSets=16384u,
                                                .poolSizeCount=1,
                                                .pPoolSizes=&pool_size};
    result=vkCreateDescriptorPool(tower->device,&descriptor_info,NULL,&tower->descriptor_pool);
    if(result!=VK_SUCCESS){
        fg_tower_vk_close(tower);
        return tower_vk_error(err,"create descriptor pool",result);
    }
    struct kernel_request { const char *file; uint32_t bindings; tower_kernel *kernel; };
    const struct kernel_request requests[]={
        {"fg_tower_matmul.spv",3u,&tower->matmul},
        {"fg_tower_bias.spv",2u,&tower->bias},
        {"fg_tower_gelu.spv",1u,&tower->gelu},
        {"fg_tower_add.spv",3u,&tower->add},
        {"fg_tower_layernorm.spv",4u,&tower->layernorm},
        {"fg_tower_rope_vision.spv",1u,&tower->rope},
        {"fg_tower_attention.spv",2u,&tower->attention},
        {"fg_tower_pos_embd.spv",2u,&tower->position}
    };
    for(uint32_t i=0;i<sizeof(requests)/sizeof(requests[0]);i++){
        fg_status status=tower_kernel_build(tower,requests[i].file,requests[i].bindings,
                                            requests[i].kernel,err);
        if(status!=FG_OK){
            fg_tower_vk_close(tower);
            return status;
        }
    }
    *out=tower;
    return FG_OK;
}

void fg_tower_vk_close(fg_tower_vk *tower){
    if(!tower)return;
    const tower_kernel *kernels[]={&tower->matmul,&tower->bias,&tower->gelu,&tower->add,
                                   &tower->layernorm,&tower->rope,&tower->attention,
                                   &tower->position};
    if(tower->device){
        for(uint32_t i=0;i<sizeof(kernels)/sizeof(kernels[0]);i++){
            if(kernels[i]->pipeline)vkDestroyPipeline(tower->device,kernels[i]->pipeline,NULL);
            if(kernels[i]->layout)vkDestroyPipelineLayout(tower->device,kernels[i]->layout,NULL);
            if(kernels[i]->set_layout)
                vkDestroyDescriptorSetLayout(tower->device,kernels[i]->set_layout,NULL);
        }
        if(tower->descriptor_pool)
            vkDestroyDescriptorPool(tower->device,tower->descriptor_pool,NULL);
        if(tower->fence)vkDestroyFence(tower->device,tower->fence,NULL);
        if(tower->command_pool)vkDestroyCommandPool(tower->device,tower->command_pool,NULL);
        vkDestroyDevice(tower->device,NULL);
    }
    if(tower->instance)vkDestroyInstance(tower->instance,NULL);
    free(tower);
}

const char *fg_tower_vk_device_name(const fg_tower_vk *tower){
    return tower?tower->device_name:"";
}

static fg_status tower_dispatch(fg_tower_vk *tower,tower_kernel *kernel,
                                tower_buffer *const *buffers,uint32_t count,const void *push,
                                uint32_t push_bytes,uint32_t gx,uint32_t gy,uint32_t gz,
                                fg_error *err){
    VkDescriptorSetAllocateInfo allocate={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool=tower->descriptor_pool,
                                          .descriptorSetCount=1,
                                          .pSetLayouts=&kernel->set_layout};
    VkDescriptorSet set=VK_NULL_HANDLE;
    VkResult result=vkAllocateDescriptorSets(tower->device,&allocate,&set);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"allocate descriptor set",result);
    VkDescriptorBufferInfo infos[4];
    VkWriteDescriptorSet writes[4];
    for(uint32_t i=0;i<count;i++){
        infos[i]=(VkDescriptorBufferInfo){.buffer=buffers[i]->buffer,.offset=0,
                                          .range=buffers[i]->bytes};
        writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                         .dstSet=set,.dstBinding=i,.descriptorCount=1,
                                         .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .pBufferInfo=&infos[i]};
    }
    vkUpdateDescriptorSets(tower->device,count,writes,0,NULL);
    vkCmdBindPipeline(tower->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->pipeline);
    vkCmdBindDescriptorSets(tower->command,VK_PIPELINE_BIND_POINT_COMPUTE,kernel->layout,0,1,
                            &set,0,NULL);
    vkCmdPushConstants(tower->command,kernel->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,push_bytes,
                       push);
    vkCmdDispatch(tower->command,gx,gy,gz);
    VkMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                             .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                             .dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(tower->command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,NULL,0,NULL);
    return FG_OK;
}

static fg_status tower_upload_tensor(fg_tower_vk *tower,tower_buffer *buffer,const float *data,
                                     uint64_t values,fg_error *err){
    fg_status status=tower_buffer_create(tower,values*sizeof(float),buffer,err);
    if(status!=FG_OK)return status;
    return tower_buffer_write(tower,buffer,data,values*sizeof(float),err);
}

fg_status fg_tower_vk_weights_upload(fg_tower_vk *tower,const fg_tower_weights *weights,
                                     fg_tower_vk_weights **out,fg_error *err){
    if(!tower||!weights||!out){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower weight upload");
        return FG_ERR_ARGUMENT;
    }
    fg_tower_vk_weights *device=calloc(1,sizeof(*device));
    if(!device){
        fg_error_set(err,FG_ERR_OOM,"allocate tower device weights");
        return FG_ERR_OOM;
    }
    fg_status status=tower_upload_tensor(tower,&device->patch_weight,weights->patch_weight,
                                         (uint64_t)FG_TOWER_HIDDEN*FG_TOWER_TOKEN_VALUES,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->patch_bias,weights->patch_bias,
                                                FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->position_weight,
                                                weights->position_weight,
                                                (uint64_t)FG_TOWER_POS_GRID*FG_TOWER_POS_GRID*
                                                FG_TOWER_HIDDEN,err);
    for(uint32_t layer=0;layer<FG_TOWER_LAYERS&&status==FG_OK;layer++){
        const fg_tower_block_weights *source=&weights->blocks[layer];
        tower_block_buffers *target=&device->blocks[layer];
        status=tower_upload_tensor(tower,&target->ln1_weight,source->ln1_weight,
                                   FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ln1_bias,source->ln1_bias,
                                                    FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ln2_weight,source->ln2_weight,
                                                    FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ln2_bias,source->ln2_bias,
                                                    FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->qkv_weight,source->qkv_weight,
                                                    (uint64_t)FG_TOWER_QKV_WIDTH*FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->qkv_bias,source->qkv_bias,
                                                    FG_TOWER_QKV_WIDTH,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->attn_out_weight,
                                                    source->attn_out_weight,
                                                    (uint64_t)FG_TOWER_HIDDEN*FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->attn_out_bias,
                                                    source->attn_out_bias,FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ffn_up_weight,
                                                    source->ffn_up_weight,
                                                    (uint64_t)FG_TOWER_MLP*FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ffn_up_bias,
                                                    source->ffn_up_bias,FG_TOWER_MLP,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ffn_down_weight,
                                                    source->ffn_down_weight,
                                                    (uint64_t)FG_TOWER_HIDDEN*FG_TOWER_MLP,err);
        if(status==FG_OK)status=tower_upload_tensor(tower,&target->ffn_down_bias,
                                                    source->ffn_down_bias,FG_TOWER_HIDDEN,err);
    }
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->post_ln_weight,
                                                weights->post_ln_weight,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->post_ln_bias,
                                                weights->post_ln_bias,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->merger_fc1_weight,
                                                weights->merger_fc1_weight,
                                                (uint64_t)FG_TOWER_MERGED_WIDTH*
                                                FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->merger_fc1_bias,
                                                weights->merger_fc1_bias,
                                                FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->merger_fc2_weight,
                                                weights->merger_fc2_weight,
                                                (uint64_t)FG_TOWER_OUT_HIDDEN*
                                                FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_upload_tensor(tower,&device->merger_fc2_bias,
                                                weights->merger_fc2_bias,
                                                FG_TOWER_OUT_HIDDEN,err);
    if(status!=FG_OK){
        fg_tower_vk_weights_destroy(tower,device);
        return status;
    }
    *out=device;
    return FG_OK;
}

void fg_tower_vk_weights_destroy(fg_tower_vk *tower,fg_tower_vk_weights *weights){
    if(!tower||!weights)return;
    tower_buffer_destroy(tower,&weights->patch_weight);
    tower_buffer_destroy(tower,&weights->patch_bias);
    tower_buffer_destroy(tower,&weights->position_weight);
    for(uint32_t layer=0;layer<FG_TOWER_LAYERS;layer++){
        tower_block_buffers *block=&weights->blocks[layer];
        tower_buffer_destroy(tower,&block->ln1_weight);
        tower_buffer_destroy(tower,&block->ln1_bias);
        tower_buffer_destroy(tower,&block->ln2_weight);
        tower_buffer_destroy(tower,&block->ln2_bias);
        tower_buffer_destroy(tower,&block->qkv_weight);
        tower_buffer_destroy(tower,&block->qkv_bias);
        tower_buffer_destroy(tower,&block->attn_out_weight);
        tower_buffer_destroy(tower,&block->attn_out_bias);
        tower_buffer_destroy(tower,&block->ffn_up_weight);
        tower_buffer_destroy(tower,&block->ffn_up_bias);
        tower_buffer_destroy(tower,&block->ffn_down_weight);
        tower_buffer_destroy(tower,&block->ffn_down_bias);
    }
    tower_buffer_destroy(tower,&weights->post_ln_weight);
    tower_buffer_destroy(tower,&weights->post_ln_bias);
    tower_buffer_destroy(tower,&weights->merger_fc1_weight);
    tower_buffer_destroy(tower,&weights->merger_fc1_bias);
    tower_buffer_destroy(tower,&weights->merger_fc2_weight);
    tower_buffer_destroy(tower,&weights->merger_fc2_bias);
    free(weights);
}

static fg_status tower_dispatch_matmul(fg_tower_vk *tower,const tower_buffer *weights,
                                       const tower_buffer *input,tower_buffer *output,
                                       uint32_t tokens,uint32_t outputs,uint32_t width,
                                       fg_error *err){
    tower_buffer *buffers[3]={(tower_buffer *)weights,(tower_buffer *)input,output};
    push_matmul push={tokens,outputs,width};
    return tower_dispatch(tower,&tower->matmul,buffers,3u,&push,sizeof(push),
                          (outputs+63u)/64u,(tokens+63u)/64u,1u,err);
}

static fg_status tower_dispatch_bias(fg_tower_vk *tower,tower_buffer *values,
                                     const tower_buffer *bias,uint32_t count,uint32_t width,
                                     fg_error *err){
    tower_buffer *buffers[2]={values,(tower_buffer *)bias};
    push_bias push={count,width};
    return tower_dispatch(tower,&tower->bias,buffers,2u,&push,sizeof(push),
                          (count+255u)/256u,1u,1u,err);
}

static fg_status tower_dispatch_gelu(fg_tower_vk *tower,tower_buffer *values,uint32_t count,
                                     fg_error *err){
    tower_buffer *buffers[1]={values};
    push_gelu push={count};
    return tower_dispatch(tower,&tower->gelu,buffers,1u,&push,sizeof(push),
                          (count+255u)/256u,1u,1u,err);
}

static fg_status tower_dispatch_add(fg_tower_vk *tower,tower_buffer *left,
                                    const tower_buffer *right,tower_buffer *output,
                                    uint32_t count,fg_error *err){
    tower_buffer *buffers[3]={left,(tower_buffer *)right,output};
    push_gelu push={count};
    return tower_dispatch(tower,&tower->add,buffers,3u,&push,sizeof(push),
                          (count+255u)/256u,1u,1u,err);
}

static fg_status tower_dispatch_layernorm(fg_tower_vk *tower,const tower_buffer *input,
                                          const tower_buffer *weight,const tower_buffer *bias,
                                          tower_buffer *output,uint32_t rows,uint32_t width,
                                          fg_error *err){
    tower_buffer *buffers[4]={(tower_buffer *)input,(tower_buffer *)weight,
                              (tower_buffer *)bias,output};
    push_layernorm push={rows,width,FG_TOWER_LN_EPS};
    return tower_dispatch(tower,&tower->layernorm,buffers,4u,&push,sizeof(push),rows,1u,1u,err);
}

static fg_status tower_dispatch_rope(fg_tower_vk *tower,tower_buffer *qkv,uint32_t tokens,
                                     uint32_t grid_width,uint32_t section,fg_error *err){
    tower_buffer *buffers[1]={qkv};
    push_rope push={tokens,grid_width,section};
    const uint32_t threads=tokens*FG_TOWER_HEADS*FG_TOWER_ROPE_PAIRS;
    return tower_dispatch(tower,&tower->rope,buffers,1u,&push,sizeof(push),
                          (threads+63u)/64u,1u,1u,err);
}

static fg_status tower_dispatch_attention(fg_tower_vk *tower,const tower_buffer *qkv,
                                          tower_buffer *output,uint32_t tokens,fg_error *err){
    tower_buffer *buffers[2]={(tower_buffer *)qkv,output};
    push_attention push={tokens,1.0f/sqrtf((float)FG_TOWER_HEAD_DIM)};
    return tower_dispatch(tower,&tower->attention,buffers,2u,&push,sizeof(push),
                          tokens*FG_TOWER_HEADS,1u,1u,err);
}

static fg_status tower_dispatch_position(fg_tower_vk *tower,const tower_buffer *position,
                                         tower_buffer *output,const fg_tower_geometry *geometry,
                                         fg_error *err){
    tower_buffer *buffers[2]={(tower_buffer *)position,output};
    push_position push={geometry->tokens,geometry->grid_width,geometry->grid_height};
    const uint32_t count=geometry->tokens*FG_TOWER_HIDDEN;
    return tower_dispatch(tower,&tower->position,buffers,2u,&push,sizeof(push),
                          (count+255u)/256u,1u,1u,err);
}

static fg_status tower_record_patch(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                    const tower_buffer *token_buffer,tower_buffer *x,
                                    tower_buffer *t,const fg_tower_geometry *geometry,
                                    fg_error *err){
    const uint32_t n=geometry->tokens;
    fg_status status=tower_dispatch_matmul(tower,&weights->patch_weight,token_buffer,x,n,
                                           FG_TOWER_HIDDEN,FG_TOWER_TOKEN_VALUES,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,x,&weights->patch_bias,
                                                n*FG_TOWER_HIDDEN,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_position(tower,&weights->position_weight,t,geometry,
                                                    err);
    if(status==FG_OK)status=tower_dispatch_add(tower,x,t,x,n*FG_TOWER_HIDDEN,err);
    return status;
}

static fg_status tower_record_block(fg_tower_vk *tower,const tower_block_buffers *weights,
                                    const tower_buffer *x,tower_buffer *t,tower_buffer *qkv,
                                    tower_buffer *up,tower_buffer *attn,tower_buffer *block,
                                    uint32_t tokens,uint32_t grid_width,fg_error *err){
    fg_status status=tower_dispatch_layernorm(tower,x,&weights->ln1_weight,&weights->ln1_bias,t,
                                              tokens,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_matmul(tower,&weights->qkv_weight,t,qkv,tokens,
                                                  FG_TOWER_QKV_WIDTH,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,qkv,&weights->qkv_bias,
                                                tokens*FG_TOWER_QKV_WIDTH,FG_TOWER_QKV_WIDTH,err);
    if(status==FG_OK)status=tower_dispatch_rope(tower,qkv,tokens,grid_width,0u,err);
    if(status==FG_OK)status=tower_dispatch_rope(tower,qkv,tokens,grid_width,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_attention(tower,qkv,attn,tokens,err);
    if(status==FG_OK)status=tower_dispatch_matmul(tower,&weights->attn_out_weight,attn,block,
                                                  tokens,FG_TOWER_HIDDEN,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,block,&weights->attn_out_bias,
                                                tokens*FG_TOWER_HIDDEN,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_add(tower,(tower_buffer *)x,block,
                                               (tower_buffer *)x,tokens*FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_layernorm(tower,x,&weights->ln2_weight,
                                                    &weights->ln2_bias,t,tokens,FG_TOWER_HIDDEN,
                                                    err);
    if(status==FG_OK)status=tower_dispatch_matmul(tower,&weights->ffn_up_weight,t,up,tokens,
                                                  FG_TOWER_MLP,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,up,&weights->ffn_up_bias,
                                                tokens*FG_TOWER_MLP,FG_TOWER_MLP,err);
    if(status==FG_OK)status=tower_dispatch_gelu(tower,up,tokens*FG_TOWER_MLP,err);
    if(status==FG_OK)status=tower_dispatch_matmul(tower,&weights->ffn_down_weight,up,block,
                                                  tokens,FG_TOWER_HIDDEN,FG_TOWER_MLP,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,block,&weights->ffn_down_bias,
                                                tokens*FG_TOWER_HIDDEN,FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_dispatch_add(tower,(tower_buffer *)x,block,
                                               (tower_buffer *)x,tokens*FG_TOWER_HIDDEN,err);
    return status;
}

static fg_status tower_record_merger(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                     const tower_buffer *t,tower_buffer *block,tower_buffer *out,
                                     uint32_t merged,fg_error *err){
    fg_status status=tower_dispatch_matmul(tower,&weights->merger_fc1_weight,t,block,merged,
                                           FG_TOWER_MERGED_WIDTH,FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,block,&weights->merger_fc1_bias,
                                                merged*FG_TOWER_MERGED_WIDTH,
                                                FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_dispatch_gelu(tower,block,merged*FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_dispatch_matmul(tower,&weights->merger_fc2_weight,block,out,
                                                  merged,FG_TOWER_OUT_HIDDEN,
                                                  FG_TOWER_MERGED_WIDTH,err);
    if(status==FG_OK)status=tower_dispatch_bias(tower,out,&weights->merger_fc2_bias,
                                                merged*FG_TOWER_OUT_HIDDEN,
                                                FG_TOWER_OUT_HIDDEN,err);
    return status;
}

static fg_status tower_begin(fg_tower_vk *tower,fg_error *err){
    VkResult result=vkResetFences(tower->device,1,&tower->fence);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"reset fence",result);
    result=vkResetCommandBuffer(tower->command,0);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"reset command buffer",result);
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    result=vkBeginCommandBuffer(tower->command,&begin);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"begin command buffer",result);
    return FG_OK;
}

static fg_status tower_submit(fg_tower_vk *tower,fg_error *err){
    VkResult result=vkEndCommandBuffer(tower->command);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"end command buffer",result);
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1u,
                         .pCommandBuffers=&tower->command};
    result=vkQueueSubmit(tower->queue,1,&submit,tower->fence);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"submit",result);
    result=vkWaitForFences(tower->device,1,&tower->fence,VK_TRUE,UINT64_MAX);
    if(result!=VK_SUCCESS)return tower_vk_error(err,"wait",result);
    return FG_OK;
}

fg_status fg_tower_vk_run_patch(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                const float *tokens,const fg_tower_geometry *geometry,
                                float *hidden,fg_tower_vk_stats *stats,fg_error *err){
    if(!tower||!weights||!tokens||!hidden||!geometry||!geometry->tokens||
       geometry->tokens>4096u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower patch run");
        return FG_ERR_ARGUMENT;
    }
    const uint32_t n=geometry->tokens;
    const double begin=tower_now_ms();
    tower_buffer token_buffer={0},x={0},t={0};
    fg_status status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_TOKEN_VALUES*sizeof(float),
                                         &token_buffer,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&token_buffer,tokens,
                                               (uint64_t)n*FG_TOWER_TOKEN_VALUES*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),
                                                &x,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),
                                                &t,err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_record_patch(tower,weights,&token_buffer,&x,&t,geometry,err);
    if(status==FG_OK)status=tower_submit(tower,err);
    if(status==FG_OK)status=tower_buffer_read(tower,&x,hidden,
        (uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),err);
    if(stats){
        stats->dispatches=4u;
        stats->forward_ms=tower_now_ms()-begin;
        stats->upload_ms=0.0;
    }
    tower_buffer_destroy(tower,&t);
    tower_buffer_destroy(tower,&x);
    tower_buffer_destroy(tower,&token_buffer);
    return status;
}

fg_status fg_tower_vk_run_block(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                uint32_t layer,const float *input,uint32_t tokens,
                                uint32_t grid_width,float *output,fg_tower_vk_stats *stats,
                                fg_error *err){
    if(!tower||!weights||!input||!output||!tokens||layer>=FG_TOWER_LAYERS||
       !grid_width||grid_width%FG_TOWER_MERGE||tokens>4096u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower block run");
        return FG_ERR_ARGUMENT;
    }
    const double begin=tower_now_ms();
    tower_buffer x={0},t={0},qkv={0},up={0},attn={0},block={0};
    fg_status status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),
                                         &x,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&x,input,
                                               (uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),
                                                &t,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float),
                                                &qkv,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_MLP*sizeof(float),
                                                &up,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),
                                                &attn,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),
                                                &block,err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_record_block(tower,&weights->blocks[layer],&x,&t,&qkv,&up,
                                               &attn,&block,tokens,grid_width,err);
    if(status==FG_OK)status=tower_submit(tower,err);
    if(status==FG_OK)status=tower_buffer_read(tower,&x,output,
        (uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),err);
    if(stats){
        stats->dispatches=16u;
        stats->forward_ms=tower_now_ms()-begin;
        stats->upload_ms=0.0;
    }
    tower_buffer_destroy(tower,&block);
    tower_buffer_destroy(tower,&attn);
    tower_buffer_destroy(tower,&up);
    tower_buffer_destroy(tower,&qkv);
    tower_buffer_destroy(tower,&t);
    tower_buffer_destroy(tower,&x);
    return status;
}

fg_status fg_tower_vk_run_merger(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                                 const float *input,uint32_t merged_tokens,float *embeddings,
                                 fg_tower_vk_stats *stats,fg_error *err){
    if(!tower||!weights||!input||!embeddings||!merged_tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower merger run");
        return FG_ERR_ARGUMENT;
    }
    const double begin=tower_now_ms();
    tower_buffer t={0},block={0},out={0};
    fg_status status=tower_buffer_create(tower,
        (uint64_t)merged_tokens*FG_TOWER_MERGED_WIDTH*sizeof(float),&t,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&t,input,
        (uint64_t)merged_tokens*FG_TOWER_MERGED_WIDTH*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,
        (uint64_t)merged_tokens*FG_TOWER_MERGED_WIDTH*sizeof(float),&block,err);
    if(status==FG_OK)status=tower_buffer_create(tower,
        (uint64_t)merged_tokens*FG_TOWER_OUT_HIDDEN*sizeof(float),&out,err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_record_merger(tower,weights,&t,&block,&out,merged_tokens,err);
    if(status==FG_OK)status=tower_submit(tower,err);
    if(status==FG_OK)status=tower_buffer_read(tower,&out,embeddings,
        (uint64_t)merged_tokens*FG_TOWER_OUT_HIDDEN*sizeof(float),err);
    if(stats){
        stats->dispatches=6u;
        stats->forward_ms=tower_now_ms()-begin;
        stats->upload_ms=0.0;
    }
    tower_buffer_destroy(tower,&out);
    tower_buffer_destroy(tower,&block);
    tower_buffer_destroy(tower,&t);
    return status;
}

fg_status fg_tower_vk_run(fg_tower_vk *tower,const fg_tower_vk_weights *weights,
                          const float *tokens,const fg_tower_geometry *geometry,
                          float *embeddings,fg_tower_vk_stats *stats,fg_error *err){
    if(!tower||!weights||!tokens||!embeddings||!geometry||!geometry->tokens||
       geometry->tokens>4096u){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower Vulkan run");
        return FG_ERR_ARGUMENT;
    }
    const uint32_t n=geometry->tokens;
    const uint32_t merged=geometry->merged_tokens;
    const double begin=tower_now_ms();
    tower_buffer token_buffer={0},x={0},t={0},qkv={0},up={0},attn={0},block={0},out={0};
    fg_status status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_TOKEN_VALUES*sizeof(float),
                                         &token_buffer,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&token_buffer,tokens,
                                               (uint64_t)n*FG_TOWER_TOKEN_VALUES*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),
                                                &x,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),
                                                &t,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_QKV_WIDTH*sizeof(float),
                                                &qkv,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_MLP*sizeof(float),
                                                &up,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),
                                                &attn,err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)n*FG_TOWER_HIDDEN*sizeof(float),
                                                &block,err);
    if(status==FG_OK)status=tower_buffer_create(tower,
        (uint64_t)merged*FG_TOWER_OUT_HIDDEN*sizeof(float),&out,err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_record_patch(tower,weights,&token_buffer,&x,&t,geometry,err);
    if(status==FG_OK)status=tower_submit(tower,err);
    for(uint32_t layer=0;layer<FG_TOWER_LAYERS&&status==FG_OK;layer++){
        status=tower_begin(tower,err);
        if(status==FG_OK)
            status=tower_record_block(tower,&weights->blocks[layer],&x,&t,&qkv,&up,&attn,&block,
                                      n,geometry->grid_width,err);
        if(status==FG_OK)status=tower_submit(tower,err);
    }
    if(status==FG_OK){
        status=tower_begin(tower,err);
        if(status==FG_OK)status=tower_dispatch_layernorm(tower,&x,&weights->post_ln_weight,
                                                         &weights->post_ln_bias,&t,n,
                                                         FG_TOWER_HIDDEN,err);
        if(status==FG_OK)status=tower_record_merger(tower,weights,&t,&block,&out,merged,err);
        if(status==FG_OK)status=tower_submit(tower,err);
    }
    if(status==FG_OK)status=tower_buffer_read(tower,&out,embeddings,
        (uint64_t)merged*FG_TOWER_OUT_HIDDEN*sizeof(float),err);
    if(stats){
        stats->dispatches=4u+FG_TOWER_LAYERS*16u+6u;
        stats->forward_ms=tower_now_ms()-begin;
        stats->upload_ms=0.0;
    }
    tower_buffer_destroy(tower,&out);
    tower_buffer_destroy(tower,&block);
    tower_buffer_destroy(tower,&attn);
    tower_buffer_destroy(tower,&up);
    tower_buffer_destroy(tower,&qkv);
    tower_buffer_destroy(tower,&t);
    tower_buffer_destroy(tower,&x);
    tower_buffer_destroy(tower,&token_buffer);
    return status;
}

fg_status fg_tower_vk_debug_layernorm(fg_tower_vk *tower,const float *input,const float *weight,
                                      const float *bias,uint32_t rows,uint32_t width,
                                      float *output,fg_error *err){
    if(!tower||!input||!weight||!bias||!output||!rows||!width){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower layernorm debug request");
        return FG_ERR_ARGUMENT;
    }
    tower_buffer in={0},weights={0},biases={0},out={0};
    fg_status status=tower_buffer_create(tower,(uint64_t)rows*width*sizeof(float),&in,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&in,input,
                                               (uint64_t)rows*width*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)width*sizeof(float),&weights,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&weights,weight,
                                               (uint64_t)width*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)width*sizeof(float),&biases,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&biases,bias,
                                               (uint64_t)width*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,(uint64_t)rows*width*sizeof(float),&out,err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_dispatch_layernorm(tower,&in,&weights,&biases,&out,rows,width,
                                                     err);
    if(status==FG_OK)status=tower_submit(tower,err);
    if(status==FG_OK)status=tower_buffer_read(tower,&out,output,
                                              (uint64_t)rows*width*sizeof(float),err);
    tower_buffer_destroy(tower,&out);
    tower_buffer_destroy(tower,&biases);
    tower_buffer_destroy(tower,&weights);
    tower_buffer_destroy(tower,&in);
    return status;
}

fg_status fg_tower_vk_debug_rope(fg_tower_vk *tower,float *qkv,uint32_t tokens,
                                 uint32_t grid_width,fg_error *err){
    if(!tower||!qkv||!tokens||!grid_width){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower rope debug request");
        return FG_ERR_ARGUMENT;
    }
    tower_buffer buffer={0};
    fg_status status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float),
                                         &buffer,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&buffer,qkv,
        (uint64_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float),err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_dispatch_rope(tower,&buffer,tokens,grid_width,0u,err);
    if(status==FG_OK)status=tower_dispatch_rope(tower,&buffer,tokens,grid_width,
                                                FG_TOWER_HIDDEN,err);
    if(status==FG_OK)status=tower_submit(tower,err);
    if(status==FG_OK)status=tower_buffer_read(tower,&buffer,qkv,
        (uint64_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float),err);
    tower_buffer_destroy(tower,&buffer);
    return status;
}

fg_status fg_tower_vk_debug_attention(fg_tower_vk *tower,const float *qkv,uint32_t tokens,
                                      float *output,fg_error *err){
    if(!tower||!qkv||!output||!tokens){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid tower attention debug request");
        return FG_ERR_ARGUMENT;
    }
    tower_buffer in={0},out={0};
    fg_status status=tower_buffer_create(tower,(uint64_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float),
                                         &in,err);
    if(status==FG_OK)status=tower_buffer_write(tower,&in,qkv,
        (uint64_t)tokens*FG_TOWER_QKV_WIDTH*sizeof(float),err);
    if(status==FG_OK)status=tower_buffer_create(tower,
        (uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),&out,err);
    if(status==FG_OK)status=tower_begin(tower,err);
    if(status==FG_OK)status=tower_dispatch_attention(tower,&in,&out,tokens,err);
    if(status==FG_OK)status=tower_submit(tower,err);
    if(status==FG_OK)status=tower_buffer_read(tower,&out,output,
        (uint64_t)tokens*FG_TOWER_HIDDEN*sizeof(float),err);
    tower_buffer_destroy(tower,&out);
    tower_buffer_destroy(tower,&in);
    return status;
}
