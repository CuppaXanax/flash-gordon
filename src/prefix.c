#include "fg_prefix.h"

#include <stdlib.h>
#include <string.h>

const char *fg_prefix_reset_reason_name(fg_prefix_reset_reason reason) {
    switch (reason) {
        case FG_PREFIX_RESET_NONE: return "none";
        case FG_PREFIX_RESET_COLD_START: return "cold-start";
        case FG_PREFIX_RESET_EXPLICIT: return "explicit";
        case FG_PREFIX_RESET_REQUEST_SHORTER: return "request-shorter";
        case FG_PREFIX_RESET_TOKEN_MISMATCH: return "token-mismatch";
        case FG_PREFIX_RESET_PUBLIC_MISMATCH: return "public-history-mismatch";
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

fg_status fg_prefix_build_continuation_tokens(
    const int32_t *history, size_t history_count, uint32_t pending_token,
    const uint32_t *continuation, size_t continuation_count,
    uint32_t **tokens, size_t *token_count, fg_error *err) {
    if (tokens) *tokens = NULL;
    if (token_count) *token_count = 0;
    if (!tokens || !token_count || (history_count && !history) ||
        !continuation || !continuation_count || continuation[0] != pending_token) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "invalid authoritative continuation token boundary");
        return FG_ERR_ARGUMENT;
    }
    if (history_count > SIZE_MAX - continuation_count ||
        history_count + continuation_count > SIZE_MAX / sizeof(**tokens)) {
        fg_error_set(err, FG_ERR_LIMIT,
                     "authoritative continuation token stream exceeds address space");
        return FG_ERR_LIMIT;
    }
    size_t count = history_count + continuation_count;
    uint32_t *result = malloc(count * sizeof(*result));
    if (!result) {
        fg_error_set(err, FG_ERR_OOM,
                     "allocate authoritative continuation token stream");
        return FG_ERR_OOM;
    }
    for (size_t i = 0; i < history_count; i++) {
        if (history[i] < 0) {
            free(result);
            fg_error_set(err, FG_ERR_MISMATCH,
                         "resident token history contains an invalid token");
            return FG_ERR_MISMATCH;
        }
        result[i] = (uint32_t)history[i];
    }
    memcpy(result + history_count, continuation,
           continuation_count * sizeof(*continuation));
    *tokens = result;
    *token_count = count;
    return FG_OK;
}
