#ifndef FLASH_GORDON_PREFIX_H
#define FLASH_GORDON_PREFIX_H

#include "fg.h"

typedef enum fg_prefix_reset_reason {
    FG_PREFIX_RESET_NONE = 0,
    FG_PREFIX_RESET_COLD_START,
    FG_PREFIX_RESET_EXPLICIT,
    FG_PREFIX_RESET_REQUEST_SHORTER,
    FG_PREFIX_RESET_TOKEN_MISMATCH,
    FG_PREFIX_RESET_PUBLIC_MISMATCH,
    FG_PREFIX_RESET_FRONTIER_UNAVAILABLE,
    FG_PREFIX_RESET_FAILURE
} fg_prefix_reset_reason;

typedef struct fg_prefix_plan {
    bool hit;
    bool exact_frontier;
    size_t reused_tokens;
    size_t prefill_offset;
    size_t prefill_tokens;
    fg_prefix_reset_reason reset_reason;
} fg_prefix_plan;

fg_status fg_prefix_plan_tokens(const int32_t *history, size_t history_count,
                                bool next_token_valid, const uint32_t *transcript,
                                size_t transcript_count,
                                fg_prefix_reset_reason empty_reason,
                                fg_prefix_plan *plan, fg_error *err);
const char *fg_prefix_reset_reason_name(fg_prefix_reset_reason reason);

#endif
