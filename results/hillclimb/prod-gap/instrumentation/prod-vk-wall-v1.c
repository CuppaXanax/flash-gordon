#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

static PFN_vkQueueSubmit real_submit;
static PFN_vkWaitForFences real_wait;
static PFN_vkBeginCommandBuffer real_begin;
static PFN_vkEndCommandBuffer real_end;
static PFN_vkResetCommandBuffer real_reset;
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void initialize(void) {
    real_submit = (PFN_vkQueueSubmit)dlsym(RTLD_NEXT, "vkQueueSubmit");
    real_wait = (PFN_vkWaitForFences)dlsym(RTLD_NEXT, "vkWaitForFences");
    real_begin = (PFN_vkBeginCommandBuffer)dlsym(RTLD_NEXT, "vkBeginCommandBuffer");
    real_end = (PFN_vkEndCommandBuffer)dlsym(RTLD_NEXT, "vkEndCommandBuffer");
    real_reset = (PFN_vkResetCommandBuffer)dlsym(RTLD_NEXT, "vkResetCommandBuffer");
    if (!real_submit || !real_wait || !real_begin || !real_end || !real_reset)
        _exit(127);
}

static uint64_t now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static void record(const char *op, uint64_t begin, uint64_t end,
                   uintptr_t object, VkResult result) {
    fprintf(stderr, "VKWALL op=%s begin_ns=%llu end_ns=%llu object=%llx result=%d\n",
            op, (unsigned long long)begin, (unsigned long long)end,
            (unsigned long long)object, result);
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t count,
        const VkSubmitInfo *submits, VkFence fence) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_submit(queue, count, submits, fence);
    record("submit", begin, now_ns(), (uintptr_t)queue, result);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t count,
        const VkFence *fences, VkBool32 all, uint64_t timeout) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_wait(device, count, fences, all, timeout);
    record("wait", begin, now_ns(), count ? (uintptr_t)fences[0] : 0, result);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer command,
        const VkCommandBufferBeginInfo *info) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_begin(command, info);
    record("begin", begin, now_ns(), (uintptr_t)command, result);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer command) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_end(command);
    record("end", begin, now_ns(), (uintptr_t)command, result);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer command,
        VkCommandBufferResetFlags flags) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_reset(command, flags);
    record("reset", begin, now_ns(), (uintptr_t)command, result);
    return result;
}
