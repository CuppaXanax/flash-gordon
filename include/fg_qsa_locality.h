#ifndef FLASH_GORDON_QSA_LOCALITY_H
#define FLASH_GORDON_QSA_LOCALITY_H

#include "fg_qsa_state.h"

#define FG_QSA_LOCALITY_MAX_BUDGETS 8u

typedef enum fg_qsa_locality_mode {
    FG_QSA_LOCALITY_SUMMARY = 1,
    FG_QSA_LOCALITY_TOKEN = 2
} fg_qsa_locality_mode;

typedef struct fg_qsa_locality fg_qsa_locality;

typedef struct fg_qsa_locality_stats {
    uint64_t selections;
    uint64_t selected_refs;
    uint64_t unique_pages;
    uint64_t dedup_pages;
    uint64_t hot_tail_hits;
    uint64_t cold_refs;
    uint64_t previous_overlap;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t first_cold_refs;
    uint64_t reused_cold_refs;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_max;
    uint64_t selection_digest;
    uint32_t budget_count;
    uint32_t budget_mib[FG_QSA_LOCALITY_MAX_BUDGETS];
    uint64_t budget_hits[FG_QSA_LOCALITY_MAX_BUDGETS];
    uint64_t budget_misses[FG_QSA_LOCALITY_MAX_BUDGETS];
} fg_qsa_locality_stats;

fg_qsa_locality *fg_qsa_locality_create(fg_qsa_locality_mode mode,uint32_t max_blocks,
                                        uint32_t hot_tokens,const uint32_t *budget_mib,
                                        uint32_t budget_count,uint64_t digest_key);
fg_qsa_locality *fg_qsa_locality_create_from_env(uint32_t max_blocks,uint32_t hot_tokens);
uint32_t *fg_qsa_locality_selection_buffer(fg_qsa_locality *trace);
void fg_qsa_locality_record_selection(fg_qsa_locality *trace,uint32_t layer,uint32_t tokens,
                                      const uint32_t *page_ids,uint32_t page_count);
void fg_qsa_locality_record_cache(fg_qsa_locality *trace,uint32_t layer,bool hit);
void fg_qsa_locality_get_stats(const fg_qsa_locality *trace,fg_qsa_locality_stats *stats);
void fg_qsa_locality_reset(fg_qsa_locality *trace,const char *reason);
void fg_qsa_locality_destroy(fg_qsa_locality *trace,const char *reason);

#endif
