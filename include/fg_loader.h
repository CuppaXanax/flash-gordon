#ifndef FLASH_GORDON_LOADER_H
#define FLASH_GORDON_LOADER_H

#include "fg_manifest.h"

/* The arena must be the final host-visible Vulkan allocation, 4 KiB aligned. */
fg_status fg_load_rank_weights(const fg_manifest *manifest,const char *pack_dir,uint32_t rank,void *arena,uint64_t arena_bytes,fg_error *err);
fg_status fg_verify_rank_arena(const fg_manifest *manifest,uint32_t rank,const void *arena,uint64_t arena_bytes,fg_error *err);

#endif
