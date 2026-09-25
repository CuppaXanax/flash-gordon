#ifndef FLASH_GORDON_NGRAM_H
#define FLASH_GORDON_NGRAM_H

#include "fg_manifest.h"
#include "fg_vk.h"

#define FG_NGRAM_CACHE_BYTES (8u * 1024u * 1024u)
#define FG_NGRAM_BLOCK_BYTES 4096u
#define FG_NGRAM_MAX_READ_BYTES 8192u
#define FG_NGRAM_ROW_BYTES 90u
#define FG_Q38_EOS_TOKEN 248044u
#define FG_NGRAM_PREFILL_MAX_TOKENS 1024u
#define FG_NGRAM_PREFILL_MAX_ROWS (FG_NGRAM_PREFILL_MAX_TOKENS*FG_NGRAM_HEAD_COUNT)
#define FG_NGRAM_PREFILL_MAX_BLOCKS (FG_NGRAM_PREFILL_MAX_ROWS*2u)
#define FG_NGRAM_IO_SLOTS 64u
#define FG_NGRAM_PREFILL_IO_BYTES ((uint64_t)FG_NGRAM_IO_SLOTS*FG_NGRAM_MAX_READ_BYTES)
/* Asynchronous next-step read-ahead: one helper thread issues plain O_DIRECT
 * preads for the rows the next lookup will need into a private buffer while the
 * runtime keeps working; the main thread inserts the finished blocks into the
 * n-gram cache at the next lookup.  The cache stays single-threaded and the
 * mechanism is strictly best-effort: a missing or failed job just falls back to
 * the synchronous path. */
#define FG_NGRAM_PREFETCH_ROWS 64u
/* Pageable worker shard mode (FG_NGRAM_PAGEABLE=1): map the sealed shard and
 * serve rows from the kernel page cache with bounded WILLNEED read-ahead
 * instead of pinning the whole shard.  The first FG_NGRAM_PAGEABLE_HOT_BYTES
 * of the mapping stay mlocked as the pinned hot subset. */
#define FG_NGRAM_PAGEABLE_HOT_BYTES (128u * 1024u * 1024u)

typedef struct fg_ngram_read {uint64_t offset;uint32_t bytes;} fg_ngram_read;
typedef struct fg_ngram_cache fg_ngram_cache;
typedef struct fg_ngram_store fg_ngram_store;

/* Validate and hash a suffix plus its two-token dependency window. Earlier
 * history belongs to the runtime's already-validated prefix; no state is cached. */
fg_status fg_q38_ngram_lookup_range(const int32_t *history,size_t history_count,
    size_t first_token,uint32_t token_count,uint64_t *rows,uint64_t *addresses,
    fg_error *err);
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
fg_status fg_ngram_resident_open_sealed(fg_ngram_resident **out,const char *path,
                                        uint64_t row_begin,uint64_t row_count,
                                        const uint8_t sha256[32],fg_error *err);
fg_status fg_ngram_resident_open_manifest(fg_ngram_resident **out,
                                          const fg_manifest *manifest,
                                          const char *pack_dir,uint32_t rank,
                                          fg_error *err);
void fg_ngram_resident_close(fg_ngram_resident *resident);
bool fg_ngram_resident_pageable(const fg_ngram_resident *resident);
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
/* Addresses the next single-token lookup would touch after `next_token` is
   appended to a history of `history_count` tokens.  Only the two-token suffix
   window participates, so a short tail copy is hashed with the same planner the
   real lookup uses. */
fg_status fg_q38_ngram_next_addresses(const int32_t *history,size_t history_count,
                                      int32_t next_token,
                                      uint64_t addresses[FG_NGRAM_HEAD_COUNT],
                                      fg_error *err);
/* Best-effort asynchronous read-ahead of up to FG_NGRAM_PREFETCH_ROWS row
   addresses.  Starts (or reuses) the store's prefetch thread; a request while a
   job is still running is dropped.  Never fails a caller on operational
   grounds: a dropped or failed job simply leaves the blocks to the synchronous
   path. */
fg_status fg_ngram_store_prefetch(fg_ngram_store *store,const uint64_t *addresses,
                                  uint32_t address_count,fg_error *err);

#endif
