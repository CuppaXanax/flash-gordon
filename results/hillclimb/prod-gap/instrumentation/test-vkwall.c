#include <assert.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#ifdef FAKE
VkResult vkQueueSubmit(VkQueue q,uint32_t n,const VkSubmitInfo *s,VkFence f){
    assert((uintptr_t)q==1&&n==3&&s->sType==VK_STRUCTURE_TYPE_SUBMIT_INFO&&(uintptr_t)f==4);
    return VK_ERROR_DEVICE_LOST;
}
VkResult vkWaitForFences(VkDevice d,uint32_t n,const VkFence *f,VkBool32 a,uint64_t t){
    assert((uintptr_t)d==2&&n==1&&(uintptr_t)f[0]==7&&a==VK_TRUE&&t==123);
    return VK_TIMEOUT;
}
VkResult vkBeginCommandBuffer(VkCommandBuffer c,const VkCommandBufferBeginInfo *i){
    assert((uintptr_t)c==8&&i->sType==VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    return VK_SUCCESS;
}
VkResult vkEndCommandBuffer(VkCommandBuffer c){assert((uintptr_t)c==8);return VK_SUCCESS;}
VkResult vkResetCommandBuffer(VkCommandBuffer c,VkCommandBufferResetFlags f){
    assert((uintptr_t)c==8&&f==VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    return VK_SUCCESS;
}
#else
int main(void){
    VkSubmitInfo s={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkCommandBufferBeginInfo i={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkFence f=(VkFence)(uintptr_t)7;
    assert(vkQueueSubmit((VkQueue)(uintptr_t)1,3,&s,(VkFence)(uintptr_t)4)==VK_ERROR_DEVICE_LOST);
    assert(vkWaitForFences((VkDevice)(uintptr_t)2,1,&f,VK_TRUE,123)==VK_TIMEOUT);
    assert(vkBeginCommandBuffer((VkCommandBuffer)(uintptr_t)8,&i)==VK_SUCCESS);
    assert(vkEndCommandBuffer((VkCommandBuffer)(uintptr_t)8)==VK_SUCCESS);
    assert(vkResetCommandBuffer((VkCommandBuffer)(uintptr_t)8,VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)==VK_SUCCESS);
    return 0;
}
#endif
