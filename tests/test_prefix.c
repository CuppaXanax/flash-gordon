#include "fg_prefix.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);         \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

static fg_prefix_plan plan(const int32_t *history, size_t history_count,
                           bool frontier_valid, const uint32_t *transcript,
                           size_t transcript_count, fg_prefix_reset_reason empty_reason) {
    fg_prefix_plan result = {0};
    fg_error err = {0};
    CHECK(fg_prefix_plan_tokens(history, history_count, frontier_valid, transcript,
                                transcript_count, empty_reason, &result, &err) == FG_OK);
    return result;
}

static void test_prefix_hit_uses_authoritative_full_tokens(void) {
    const int32_t history[] = {10, 20, 30};
    const uint32_t full_transcript[] = {10, 20, 30, 77, 88};
    const uint32_t separately_tokenized_suffix[] = {70, 7, 88};
    fg_prefix_plan result =
        plan(history, 3, true, full_transcript, 5, FG_PREFIX_RESET_NONE);
    CHECK(result.hit);
    CHECK(!result.exact_frontier);
    CHECK(result.reused_tokens == 3);
    CHECK(result.prefill_offset == 3);
    CHECK(result.prefill_tokens == 2);
    CHECK(full_transcript[result.prefill_offset] == 77);
    CHECK(full_transcript[result.prefill_offset] != separately_tokenized_suffix[0]);
}

static void test_exact_frontier_needs_no_prefill(void) {
    const int32_t history[] = {1, 2, 3};
    const uint32_t full_transcript[] = {1, 2, 3};
    fg_prefix_plan result =
        plan(history, 3, true, full_transcript, 3, FG_PREFIX_RESET_NONE);
    CHECK(result.hit);
    CHECK(result.exact_frontier);
    CHECK(result.reused_tokens == 3);
    CHECK(result.prefill_tokens == 0);
}

static void test_divergence_resets(void) {
    const int32_t history[] = {1, 2, 3};
    const uint32_t shorter[] = {1, 2};
    const uint32_t changed[] = {1, 9, 3, 4};
    fg_prefix_plan result =
        plan(history, 3, true, shorter, 2, FG_PREFIX_RESET_NONE);
    CHECK(!result.hit);
    CHECK(result.reset_reason == FG_PREFIX_RESET_REQUEST_SHORTER);
    CHECK(result.prefill_tokens == 2);
    result = plan(history, 3, true, changed, 4, FG_PREFIX_RESET_NONE);
    CHECK(!result.hit);
    CHECK(result.reset_reason == FG_PREFIX_RESET_TOKEN_MISMATCH);
    CHECK(result.reused_tokens == 0);
    CHECK(result.prefill_tokens == 4);
}

static void test_tool_loop_and_reasoning_are_token_identity(void) {
    const int32_t tool_history[] = {101, 102, 103, 104, 105, 106};
    const uint32_t tool_result_turn[] = {101, 102, 103, 104, 105, 106, 201, 202};
    const uint32_t changed_reasoning[] = {101, 102, 999, 104, 105, 106, 201, 202};
    fg_prefix_plan result =
        plan(tool_history, 6, true, tool_result_turn, 8, FG_PREFIX_RESET_NONE);
    CHECK(result.hit);
    CHECK(result.reused_tokens == 6);
    result = plan(tool_history, 6, true, changed_reasoning, 8, FG_PREFIX_RESET_NONE);
    CHECK(!result.hit);
    CHECK(result.reset_reason == FG_PREFIX_RESET_TOKEN_MISMATCH);
}

static void test_restart_reset_and_unavailable_frontier(void) {
    const uint32_t transcript[] = {7, 8};
    fg_prefix_plan result =
        plan(NULL, 0, false, transcript, 2, FG_PREFIX_RESET_COLD_START);
    CHECK(!result.hit);
    CHECK(result.reset_reason == FG_PREFIX_RESET_COLD_START);
    result = plan(NULL, 0, false, transcript, 2, FG_PREFIX_RESET_EXPLICIT);
    CHECK(result.reset_reason == FG_PREFIX_RESET_EXPLICIT);
    result = plan(NULL, 0, false, transcript, 2, FG_PREFIX_RESET_FAILURE);
    CHECK(result.reset_reason == FG_PREFIX_RESET_FAILURE);

    const int32_t history[] = {7, 8};
    result = plan(history, 2, false, transcript, 2, FG_PREFIX_RESET_NONE);
    CHECK(!result.hit);
    CHECK(result.reset_reason == FG_PREFIX_RESET_FRONTIER_UNAVAILABLE);
}

int main(void) {
    test_prefix_hit_uses_authoritative_full_tokens();
    test_exact_frontier_needs_no_prefill();
    test_divergence_resets();
    test_tool_loop_and_reasoning_are_token_identity();
    test_restart_reset_and_unavailable_frontier();
    CHECK(!strcmp(fg_prefix_reset_reason_name(FG_PREFIX_RESET_TOKEN_MISMATCH),
                  "token-mismatch"));
    if (failures) fprintf(stderr, "%d prefix test(s) failed\n", failures);
    return failures ? 1 : 0;
}
