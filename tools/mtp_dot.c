#include <vulkan/vulkan.h>
#include <stdio.h>

int main(void){
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo create={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    VkInstance instance=VK_NULL_HANDLE;
    if(vkCreateInstance(&create,NULL,&instance)!=VK_SUCCESS)return 1;
    uint32_t count=8u;VkPhysicalDevice devices[8];
    if(vkEnumeratePhysicalDevices(instance,&count,devices)!=VK_SUCCESS||!count){
        vkDestroyInstance(instance,NULL);
        return 1;
    }
    for(uint32_t i=0;i<count;i++){
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot={
            .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&dot};
        vkGetPhysicalDeviceFeatures2(devices[i],&features);
        VkPhysicalDeviceShaderIntegerDotProductProperties props={
            .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        VkPhysicalDeviceProperties2 properties={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,.pNext=&props};
        vkGetPhysicalDeviceProperties2(devices[i],&properties);
        printf("{\"device\":\"%s\",\"dot_available\":%u,\"signed_i8_accelerated\":%u,"
            "\"packed_i8_accelerated\":%u,\"mixed_i8_accelerated\":%u}\n",
            properties.properties.deviceName,dot.shaderIntegerDotProduct,
            props.integerDotProduct8BitSignedAccelerated,
            props.integerDotProduct4x8BitPackedSignedAccelerated,
            props.integerDotProduct4x8BitPackedMixedSignednessAccelerated);
    }
    vkDestroyInstance(instance,NULL);
    return 0;
}
