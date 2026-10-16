#define _GNU_SOURCE
#include <dlfcn.h>
#include <dirent.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

static PFN_vkQueueSubmit real_submit;
static PFN_vkWaitForFences real_wait;
static PFN_vkBeginCommandBuffer real_begin;
static PFN_vkEndCommandBuffer real_end;
static PFN_vkResetCommandBuffer real_reset;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static char engine_path[128];

static uint64_t read_engine_path(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return UINT64_MAX;
    char buffer[8192];
    size_t bytes = fread(buffer, 1, sizeof(buffer)-1, file);
    fclose(file);
    buffer[bytes] = 0;
    char *key = strstr(buffer, "drm-engine-gfx:");
    return key ? strtoull(strchr(key, ':')+1, NULL, 10) : UINT64_MAX;
}

static void find_engine(void) {
    DIR *directory = opendir("/proc/self/fdinfo");
    if (!directory) return;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        char *end;
        unsigned long fd = strtoul(entry->d_name, &end, 10);
        if (!*entry->d_name || *end) continue;
        char path[128];
        snprintf(path, sizeof(path), "/proc/self/fdinfo/%lu", fd);
        if (read_engine_path(path) != UINT64_MAX) {
            memcpy(engine_path, path, strlen(path)+1);
            break;
        }
    }
    closedir(directory);
}

static uint64_t engine_busy(void) {
    if (!*engine_path) find_engine();
    return read_engine_path(engine_path);
}

static void initialize(void) {
    real_submit = (PFN_vkQueueSubmit)dlsym(RTLD_NEXT, "vkQueueSubmit");
    real_wait = (PFN_vkWaitForFences)dlsym(RTLD_NEXT, "vkWaitForFences");
    real_begin = (PFN_vkBeginCommandBuffer)dlsym(RTLD_NEXT, "vkBeginCommandBuffer");
    real_end = (PFN_vkEndCommandBuffer)dlsym(RTLD_NEXT, "vkEndCommandBuffer");
    real_reset = (PFN_vkResetCommandBuffer)dlsym(RTLD_NEXT, "vkResetCommandBuffer");
    if (!real_submit || !real_wait || !real_begin || !real_end || !real_reset)
        _exit(127);
    find_engine();
}

static uint64_t now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static void record(const char *op, uint64_t begin, uint64_t end,
                   uintptr_t object, VkResult result, uint64_t gpu_before,
                   uint64_t gpu_after) {
    fprintf(stderr, "VKWALL op=%s begin_ns=%llu end_ns=%llu object=%llx result=%d gpu_before_ns=%llu gpu_after_ns=%llu\n",
            op, (unsigned long long)begin, (unsigned long long)end,
            (unsigned long long)object, result, (unsigned long long)gpu_before,
            (unsigned long long)gpu_after);
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t count,
        const VkSubmitInfo *submits, VkFence fence) {
    pthread_once(&once, initialize);
    uint64_t gpu = engine_busy();
    uint64_t begin = now_ns();
    VkResult result = real_submit(queue, count, submits, fence);
    record("submit", begin, now_ns(), (uintptr_t)queue, result, gpu, 0);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t count,
        const VkFence *fences, VkBool32 all, uint64_t timeout) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_wait(device, count, fences, all, timeout);
    uint64_t end = now_ns();
    uint64_t gpu = engine_busy();
    record("wait", begin, end, count ? (uintptr_t)fences[0] : 0, result, 0, gpu);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer command,
        const VkCommandBufferBeginInfo *info) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_begin(command, info);
    record("begin", begin, now_ns(), (uintptr_t)command, result, 0, 0);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer command) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_end(command);
    record("end", begin, now_ns(), (uintptr_t)command, result, 0, 0);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer command,
        VkCommandBufferResetFlags flags) {
    pthread_once(&once, initialize);
    uint64_t begin = now_ns();
    VkResult result = real_reset(command, flags);
    record("reset", begin, now_ns(), (uintptr_t)command, result, 0, 0);
    return result;
}
