#ifndef FLASH_GORDON_QSA_CACHE_H
#define FLASH_GORDON_QSA_CACHE_H

#include "fg_protocol.h"

typedef struct fg_qsa_page_cache fg_qsa_page_cache;

fg_status fg_qsa_page_cache_create(fg_qsa_page_cache **out,uint32_t pages,fg_error *err);
void fg_qsa_page_cache_destroy(fg_qsa_page_cache *cache);
void fg_qsa_page_cache_reset(fg_qsa_page_cache *cache);
uint64_t fg_qsa_page_cache_memory_bytes(const fg_qsa_page_cache *cache);
uint64_t fg_qsa_page_cache_memory_bytes_for_pages(uint32_t pages);
bool fg_qsa_page_cache_lookup(fg_qsa_page_cache *cache,uint32_t layer,
                              uint32_t block,uint32_t *slot);
fg_status fg_qsa_page_cache_acquire(fg_qsa_page_cache *cache,uint32_t layer,
                                   uint32_t block,uint32_t *slot,bool *hit,
                                   fg_error *err);
fg_status fg_qsa_page_cache_pin(fg_qsa_page_cache *cache,uint32_t layer,
                                uint32_t block,fg_error *err);
void fg_qsa_page_cache_unpin(fg_qsa_page_cache *cache,uint32_t layer,uint32_t block);
fg_status fg_qsa_page_cache_plan_fetch(const fg_qsa_page_cache *cache,uint32_t layer,
                                       const uint32_t *missing,uint32_t missing_count,
                                       uint32_t complete_blocks,uint32_t retained_first,
                                       uint32_t *fetch,uint32_t fetch_capacity,
                                       uint32_t *fetch_count,fg_error *err);

#endif
