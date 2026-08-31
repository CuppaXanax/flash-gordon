#include "fg_prefix.h"

#include <string.h>

const char *fg_prefix_reset_reason_name(fg_prefix_reset_reason reason) {
    switch (reason) {
        case FG_PREFIX_RESET_NONE: return "none";
        case FG_PREFIX_RESET_COLD_START: return "cold-start";
        case FG_PREFIX_RESET_EXPLICIT: return "explicit";
        case FG_PREFIX_RESET_REQUEST_SHORTER: return "request-shorter";
        case FG_PREFIX_RESET_TOKEN_MISMATCH: return "token-mismatch";
        case FG_PREFIX_RESET_FRONTIER_UNAVAILABLE: return "frontier-unavailable";
        case FG_PREFIX_RESET_FAILURE: return "failure";
        default: return "unknown";
    }
}

fg_status fg_prefix_plan_tokens(const int32_t *history, size_t history_count,
                                bool next_token_valid, const uint32_t *transcript,
                                size_t transcript_count,
                                fg_prefix_reset_reason empty_reason,
                                fg_prefix_plan *plan, fg_error *err) {
    if (!plan || (history_count && !history) || (transcript_count && !transcript)) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid live-prefix planner arguments");
        return FG_ERR_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    plan->prefill_tokens = transcript_count;
    if (!history_count) {
        plan->reset_reason =
            empty_reason == FG_PREFIX_RESET_NONE ? FG_PREFIX_RESET_COLD_START : empty_reason;
        return FG_OK;
    }
    if (transcript_count < history_count) {
        plan->reset_reason = FG_PREFIX_RESET_REQUEST_SHORTER;
        return FG_OK;
    }
    for (size_t i = 0; i < history_count; i++) {
        if (history[i] < 0 || (uint32_t)history[i] != transcript[i]) {
            plan->reset_reason = FG_PREFIX_RESET_TOKEN_MISMATCH;
            return FG_OK;
        }
    }
    if (transcript_count == history_count && !next_token_valid) {
        plan->reset_reason = FG_PREFIX_RESET_FRONTIER_UNAVAILABLE;
        return FG_OK;
    }
    plan->hit = true;
    plan->exact_frontier = transcript_count == history_count;
    plan->reused_tokens = history_count;
    plan->prefill_offset = history_count;
    plan->prefill_tokens = transcript_count - history_count;
    plan->reset_reason = FG_PREFIX_RESET_NONE;
    return FG_OK;
}
