#ifndef FLASH_GORDON_GGUF_H
#define FLASH_GORDON_GGUF_H

#include "fg_manifest.h"

#define FG_GGUF_MAX_DIMS 4u

typedef struct fg_gguf_tensor {
    char *name;
    uint32_t dims;
    uint64_t shape[FG_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
    uint64_t bytes;
    uint32_t shard;
} fg_gguf_tensor;

typedef struct fg_gguf {
    char **paths;
    uint32_t shard_count;
    fg_gguf_tensor *tensors;
    uint64_t tensor_count;
    uint64_t data_offset;
    uint32_t alignment;
} fg_gguf;

fg_status fg_gguf_open(const char *const *paths, uint32_t path_count, fg_gguf *gguf, fg_error *err);
void fg_gguf_close(fg_gguf *gguf);
bool fg_gguf_type_layout(uint32_t type,uint32_t *block,uint32_t *block_bytes);
bool fg_gguf_tensor_bytes(uint32_t type,uint32_t dims,const uint64_t shape[FG_GGUF_MAX_DIMS],
                          uint64_t *bytes);
int fg_gguf_tensor_layer(const char *name);
int fg_gguf_tensor_expert(const char *name);
fg_tensor_kind fg_gguf_tensor_kind(const char *name);
void fg_gguf_print_schema(const fg_gguf *gguf);

#endif
