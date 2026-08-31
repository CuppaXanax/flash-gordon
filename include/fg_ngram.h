#ifndef FLASH_GORDON_NGRAM_H
#define FLASH_GORDON_NGRAM_H

#include "fg.h"
#include "fg_vk.h"

#define FG_NGRAM_CACHE_BYTES (8u * 1024u * 1024u)
#define FG_NGRAM_BLOCK_BYTES 4096u
#define FG_NGRAM_MAX_READ_BYTES 8192u
#define FG_NGRAM_ROW_BYTES 90u
#define FG_Q38_EOS_TOKEN 248044u
#define FG_NGRAM_PREFILL_MAX_TOKENS 512u
#define FG_NGRAM_PREFILL_MAX_ROWS (FG_NGRAM_PREFILL_MAX_TOKENS*FG_NGRAM_HEAD_COUNT)
#define FG_NGRAM_PREFILL_MAX_BLOCKS (FG_NGRAM_PREFILL_MAX_ROWS*2u)
#define FG_NGRAM_IO_SLOTS 64u
#define FG_NGRAM_PREFILL_IO_BYTES ((uint64_t)FG_NGRAM_IO_SLOTS*FG_NGRAM_MAX_READ_BYTES)

typedef struct fg_ngram_read {uint64_t offset;uint32_t bytes;} fg_ngram_read;
typedef struct fg_ngram_cache fg_ngram_cache;
typedef struct fg_ngram_store fg_ngram_store;
typedef struct fg_ngram_resident fg_ngram_resident;

/* Converts arbitrary tensor byte addresses to sorted, deduplicated 4-8 KiB
   O_DIRECT reads. Adjacent blocks are paired, never widened beyond 8 KiB. */
fg_status fg_ngram_plan_reads(const uint64_t *addresses,uint32_t address_count,uint64_t table_bytes,fg_ngram_read *reads,uint32_t read_capacity,uint32_t *read_count,fg_error *err);
fg_status fg_ngram_cache_create(fg_ngram_cache **out,fg_error *err);
void fg_ngram_cache_destroy(fg_ngram_cache *cache);
uint64_t fg_ngram_cache_memory_bytes(void);
bool fg_ngram_cache_get(fg_ngram_cache *cache,uint64_t block_offset,const void **data);
fg_status fg_ngram_cache_put(fg_ngram_cache *cache,uint64_t block_offset,const void *block,fg_error *err);
fg_status fg_q38_ngram_lookup(const int32_t *token_history,size_t token_count,
                              uint64_t row_ids[FG_NGRAM_HEAD_COUNT],
                              uint64_t byte_addresses[FG_NGRAM_HEAD_COUNT],fg_error *err);
fg_status fg_q38_ngram_head_range(uint32_t head_begin,uint32_t head_count,
                                  uint64_t *row_begin,uint64_t *row_count,fg_error *err);
fg_status fg_q38_ngram_rank_range(uint32_t rank,uint64_t *row_begin,
                                  uint64_t *row_count,fg_error *err);
fg_status fg_ngram_resident_open(fg_ngram_resident **out,const char *path,
                                 uint64_t row_begin,uint64_t row_count,fg_error *err);
void fg_ngram_resident_close(fg_ngram_resident *resident);
fg_status fg_ngram_resident_read(const fg_ngram_resident *resident,const uint64_t *rows,
                                 uint32_t row_count,uint8_t *packed,uint64_t packed_capacity,
                                 fg_error *err);
fg_status fg_ngram_store_open(fg_ngram_store **out,fg_vk_context *vk,const char *path,
                              uint64_t table_bytes,uint32_t max_tokens,fg_error *err);
void fg_ngram_store_close(fg_ngram_store *store);
uint64_t fg_ngram_store_host_bytes(const fg_ngram_store *store);
uint64_t fg_ngram_store_io_host_bytes(const fg_ngram_store *store);
uint64_t fg_ngram_store_cache_host_bytes(const fg_ngram_store *store);
uint64_t fg_ngram_store_vk_bytes(const fg_ngram_store *store);
fg_status fg_ngram_store_lookup(fg_ngram_store *store,const int32_t *token_history,
                                size_t token_count,fg_vk_tensor **embedding,fg_error *err);
fg_status fg_ngram_store_decode_packed(fg_ngram_store *store,const uint8_t *packed,
                                       uint32_t row_count,fg_vk_tensor **embedding,
                                       fg_error *err);
fg_status fg_ngram_store_verify_packed(fg_ngram_store *store,const uint64_t *addresses,
                                       uint32_t row_count,const uint8_t *packed,
                                       uint32_t *mismatch_row,fg_error *err);
/* Computes the n-gram rows for each sequential prompt position and returns one
   borrowed, exact-sized token-major [token_count, 2560] FP32 tensor view.  The
   input is bounded so all direct-I/O planning and Vulkan arenas are provisioned
   at startup; the view remains valid until the next lookup or store close. */
fg_status fg_ngram_store_lookup_prefill(fg_ngram_store *store,const int32_t *token_history,
                                         size_t history_count,uint32_t first_token,
                                         uint32_t token_count,fg_vk_tensor **embedding,
                                         fg_error *err);

#endif
