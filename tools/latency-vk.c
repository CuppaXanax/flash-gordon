#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VK(call) do {VkResult result=(call);if(result!=VK_SUCCESS){fprintf(stderr,"%s: %d\n",#call,result);exit(1);}}while(0)
static double now_us(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;
}
static int cmp(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
int main(int argc,char **argv) {
    if(argc!=7){fprintf(stderr,"latency-vk shader bytes groups dispatches barrier timestamps\n");return 2;}
    uint32_t bytes=(uint32_t)strtoul(argv[2],NULL,0),groups=(uint32_t)strtoul(argv[3],NULL,0);
    uint32_t count=(uint32_t)strtoul(argv[4],NULL,0);
    int barriers=atoi(argv[5]),timestamps=atoi(argv[6]);
    if(bytes<256||bytes>16777216||groups>65535||count>1024)return 2;
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="hc-lat",.apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    VkInstance instance;VK(vkCreateInstance(&ici,NULL,&instance));
    uint32_t device_count=0;VK(vkEnumeratePhysicalDevices(instance,&device_count,NULL));
    VkPhysicalDevice *physical=calloc(device_count,sizeof(*physical));
    if(!physical||!device_count)return 1;
    VK(vkEnumeratePhysicalDevices(instance,&device_count,physical));
    VkPhysicalDevice gpu=VK_NULL_HANDLE;VkPhysicalDeviceProperties props;
    for(uint32_t i=0;i<device_count;i++) {
        vkGetPhysicalDeviceProperties(physical[i],&props);
        if(props.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU||
           props.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){gpu=physical[i];break;}
    }
    free(physical);if(!gpu)return 1;vkGetPhysicalDeviceProperties(gpu,&props);
    uint32_t families=0;vkGetPhysicalDeviceQueueFamilyProperties(gpu,&families,NULL);
    VkQueueFamilyProperties *family=calloc(families,sizeof(*family));
    if(!family)return 1;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu,&families,family);
    uint32_t index=0;while(index<families&&!(family[index].queueFlags&VK_QUEUE_COMPUTE_BIT))index++;
    if(index==families||!family[index].timestampValidBits)return 1;
    uint32_t valid_bits=family[index].timestampValidBits;free(family);
    float priority=1;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=index,.queueCount=1,.pQueuePriorities=&priority};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    VkDevice device;VK(vkCreateDevice(gpu,&dci,NULL,&device));
    VkQueue queue;vkGetDeviceQueue(device,index,0,&queue);
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=16777216,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VkBuffer buffer;VK(vkCreateBuffer(device,&bci,NULL,&buffer));
    VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(device,buffer,&requirements);
    VkPhysicalDeviceMemoryProperties memory_props;vkGetPhysicalDeviceMemoryProperties(gpu,&memory_props);
    uint32_t memory_index=0,flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while(memory_index<memory_props.memoryTypeCount&&
        (!(requirements.memoryTypeBits&(1u<<memory_index))||
         (memory_props.memoryTypes[memory_index].propertyFlags&flags)!=flags))memory_index++;
    if(memory_index==memory_props.memoryTypeCount)return 1;
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=requirements.size,.memoryTypeIndex=memory_index};
    VkDeviceMemory memory;VK(vkAllocateMemory(device,&mai,NULL,&memory));VK(vkBindBufferMemory(device,buffer,memory,0));
    uint32_t *mapped;VK(vkMapMemory(device,memory,0,VK_WHOLE_SIZE,0,(void **)&mapped));
    for(uint32_t i=0;i<16777216/4;i++)mapped[i]=i+1;
    /* Randomize the full cycle so adjacent dependent loads are not adjacent lines. */
    if(strstr(argv[1],"chase")) {
        uint32_t lines=bytes/64,*order=malloc(lines*sizeof(*order)),seed=17;
        if(!order)return 1;
        for(uint32_t i=0;i<lines;i++)order[i]=i;
        for(uint32_t i=lines-1;i>0;i--) {
            seed=seed*1664525u+1013904223u;
            uint32_t j=seed%(i+1),value=order[i];order[i]=order[j];order[j]=value;
        }
        for(uint32_t i=0;i<lines;i++)mapped[order[i]*16]=order[(i+1)%lines]*16;
        free(order);
    }
    VkDescriptorSetLayoutBinding binding={.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo lci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&binding};
    VkDescriptorSetLayout layout;VK(vkCreateDescriptorSetLayout(device,&lci,NULL,&layout));
    VkPushConstantRange push_range={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=16};
    VkPipelineLayoutCreateInfo pci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&layout,.pushConstantRangeCount=1,.pPushConstantRanges=&push_range};
    VkPipelineLayout pipeline_layout;VK(vkCreatePipelineLayout(device,&pci,NULL,&pipeline_layout));
    FILE *file=fopen(argv[1],"rb");if(!file)return 1;
    if(fseek(file,0,SEEK_END))return 1;
    long length=ftell(file);if(length<=0||length%4)return 1;
    rewind(file);uint32_t *code=malloc((size_t)length);if(!code)return 1;
    if(fread(code,1,(size_t)length,file)!=(size_t)length)return 1;
    fclose(file);
    VkShaderModuleCreateInfo sci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=(size_t)length,.pCode=code};
    VkShaderModule shader;VK(vkCreateShaderModule(device,&sci,NULL,&shader));free(code);
    VkComputePipelineCreateInfo compute={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=shader,.pName="main"},.layout=pipeline_layout};
    VkPipeline pipeline;VK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&compute,NULL,&pipeline));
    VkDescriptorPoolSize pool_size={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&pool_size};
    VkDescriptorPool descriptor_pool;VK(vkCreateDescriptorPool(device,&dpci,NULL,&descriptor_pool));
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=descriptor_pool,.descriptorSetCount=1,.pSetLayouts=&layout};
    VkDescriptorSet set;VK(vkAllocateDescriptorSets(device,&dsai,&set));
    VkDescriptorBufferInfo info={.buffer=buffer,.offset=0,.range=16777216};
    VkWriteDescriptorSet write={.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info};
    vkUpdateDescriptorSets(device,1,&write,0,NULL);
    VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=index};
    VkCommandPool command_pool;VK(vkCreateCommandPool(device,&cpci,NULL,&command_pool));
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=command_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer command;VK(vkAllocateCommandBuffers(device,&cbai,&command));
    VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,.queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2+2*count};
    VkQueryPool queries;VK(vkCreateQueryPool(device,&qpci,NULL,&queries));
    VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;VK(vkCreateFence(device,&fci,NULL,&fence));
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK(vkBeginCommandBuffer(command,&begin));
    vkCmdResetQueryPool(command,queries,0,qpci.queryCount);
    vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&set,0,NULL);
    uint32_t push[4]={bytes/4,groups*64,1024,0};
    vkCmdPushConstants(command,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,push);
    vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,0);
    VkMemoryBarrier barrier={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    for(uint32_t i=0;i<count;i++) {
        if(timestamps)vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,2+2*i);
        if(groups)vkCmdDispatch(command,groups,1,1);
        if(barriers)vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,NULL,0,NULL);
        if(timestamps)vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,queries,3+2*i);
    }
    vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,1);
    VK(vkEndCommandBuffer(command));
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&command};
    enum{REPS=201};double wall[REPS],gpu_us[REPS];
    uint64_t mask=valid_bits==64?UINT64_MAX:(UINT64_C(1)<<valid_bits)-1;
    for(int r=-32;r<REPS;r++) {
        VK(vkResetFences(device,1,&fence));
        double start=now_us();VK(vkQueueSubmit(queue,1,&submit,fence));
        VK(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX));double end=now_us();
        uint64_t ts[2];VK(vkGetQueryPoolResults(device,queries,0,2,sizeof(ts),ts,8,VK_QUERY_RESULT_64_BIT));
        if(r>=0){wall[r]=end-start;gpu_us[r]=((ts[1]-ts[0])&mask)*props.limits.timestampPeriod/1000.;}
    }
    qsort(wall,REPS,sizeof(*wall),cmp);qsort(gpu_us,REPS,sizeof(*gpu_us),cmp);
    uint32_t output=mapped[16777216/4-1];int valid=1;
    if((strstr(argv[1],"reduction")||strstr(argv[1],"subgroup"))&&count)
        for(uint32_t i=0;i<groups;i++)if(mapped[16777216/4-1-i]!=2016)valid=0;
    if(strstr(argv[1],"dependent")&&count&&groups==1)valid=mapped[0]==1+count*(REPS+32);
    if(strstr(argv[1],"read")&&count) {
        for(uint32_t g=0;g<groups;g++) {
            uint32_t expected=0;
            for(uint32_t lane=0;lane<64;lane++)
                for(uint32_t i=g*64+lane;i<bytes/4;i+=groups*64)expected+=i+1;
            if(mapped[16777216/4-1-g]!=expected)valid=0;
        }
    }
    if(strstr(argv[1],"chase")&&count) {
        uint32_t expected=0;for(uint32_t i=0;i<push[2];i++)expected=mapped[expected];
        valid=output==expected;
    }
    printf("{\"shader\":\"%s\",\"device\":\"%s\",\"timestamp_period_ns\":%.1f,\"bytes\":%u,\"groups\":%u,\"dispatches\":%u,\"barriers\":%s,\"inner_timestamps\":%s,\"gpu_p50_us\":%.3f,\"gpu_p99_us\":%.3f,\"gpu_min_us\":%.3f,\"wall_p50_us\":%.3f,\"wall_p99_us\":%.3f,\"output\":%u,\"validation_pass\":%s}\n",
        argv[1],props.deviceName,props.limits.timestampPeriod,bytes,groups,count,barriers?"true":"false",timestamps?"true":"false",
        gpu_us[REPS/2],gpu_us[REPS*99/100],gpu_us[0],wall[REPS/2],wall[REPS*99/100],output,valid?"true":"false");
    vkDestroyFence(device,fence,NULL);vkDestroyQueryPool(device,queries,NULL);
    vkDestroyCommandPool(device,command_pool,NULL);vkDestroyDescriptorPool(device,descriptor_pool,NULL);
    vkDestroyPipeline(device,pipeline,NULL);vkDestroyShaderModule(device,shader,NULL);
    vkDestroyPipelineLayout(device,pipeline_layout,NULL);vkDestroyDescriptorSetLayout(device,layout,NULL);
    vkUnmapMemory(device,memory);vkDestroyBuffer(device,buffer,NULL);vkFreeMemory(device,memory,NULL);
    vkDestroyDevice(device,NULL);vkDestroyInstance(instance,NULL);return valid?0:1;
}
