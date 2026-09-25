#ifndef FLASH_GORDON_RUNTIME_H
#define FLASH_GORDON_RUNTIME_H

#include "fg_decode_batch.h"
#include "fg_manifest.h"
#include "fg_prefix.h"
#include "fg_sampler.h"

#include <stdlib.h>
#include <string.h>

/* Both ring halves must be active for the executor to spend its Vulkan budget
 * on the layers this rank actually executes.  The ring is the default fast
 * path; set FG_PREFILL_RING=0 or FG_DECODE_RING=0 to opt out. */
static inline bool fg_runtime_ring_enabled(void){
    const char *prefill=getenv("FG_PREFILL_RING");
    const char *decode=getenv("FG_DECODE_RING");
    return !(prefill&&*prefill&&strcmp(prefill,"0")==0)&&
           !(decode&&*decode&&strcmp(decode,"0")==0);
}

typedef struct fg_runtime fg_runtime;
/* M3 session objects: the token path state that used to live directly in
 * `fg_runtime` (history, frontier, position, sampler, rendered transcript) is
 * owned per session.  The engine binds a session for a request with
 * fg_runtime_session_begin and releases it with fg_runtime_session_end; only
 * one session is bound at a time today, so the generation path is unchanged.
 * A session whose ring/owner state was reset by another session fails closed
 * on resume (cold start) instead of resuming a stale frontier. */
typedef struct fg_runtime_session fg_runtime_session;
#define FG_RUNTIME_SESSION_MAX 8u
fg_runtime_session *fg_runtime_session_acquire(fg_runtime *runtime,uint64_t id);
fg_runtime_session *fg_runtime_session_find(fg_runtime *runtime,uint64_t id);
void fg_runtime_session_release(fg_runtime *runtime,fg_runtime_session *session);
uint64_t fg_runtime_session_id(const fg_runtime_session *session);
/* Bind/unbind the session's token-path state to the runtime.  Begin cold-starts
 * the session when its saved state cannot be resumed (fresh session, or the
 * shared ring state was reset since the session last ran). */
fg_status fg_runtime_session_begin(fg_runtime *runtime,fg_runtime_session *session,
                                   fg_error *err);
void fg_runtime_session_end(fg_runtime *runtime,fg_runtime_session *session);
#define FG_RUNTIME_BOOT_CONTEXT_TOKENS FG_MANIFEST_DEFAULT_CONTEXT_TOKENS
#define FG_RUNTIME_QSA_HOT_TOKENS 8192u
#define FG_RUNTIME_QSA_CACHE_MIN_BYTES (UINT64_C(16) << 20u)
/* The full 262K two-layer record set is ~618 MiB, so the staged window cap
 * must sit above it for the worker default and a full-context rank-0 request. */
#define FG_RUNTIME_QSA_CACHE_MAX_BYTES (UINT64_C(1024) << 20u)
#define FG_MTP_MAX_DRAFT_TOKENS 2u

enum {
    FG_RUNTIME_EXPERIMENTAL_CONTEXT = 1u << 0,
    FG_RUNTIME_EXPERIMENTAL_MTP = 1u << 1,
    FG_RUNTIME_EXPERIMENTAL_VISION = 1u << 2
};

enum {
    FG_RUNTIME_OPTION_LOGICAL_CONTEXT = 1u << 0,
    FG_RUNTIME_OPTION_GPU_INDEX = 1u << 1,
    FG_RUNTIME_OPTION_QSA_HOT = 1u << 2,
    FG_RUNTIME_OPTION_PAGE_CACHE = 1u << 3,
    FG_RUNTIME_OPTION_PREFILL_MICROBATCH = 1u << 4,
    FG_RUNTIME_OPTION_PREFILL_WINDOW = 1u << 5,
    FG_RUNTIME_OPTION_PREFIX_CONT = 1u << 6
};

typedef enum fg_mtp_capability {
    FG_MTP_CAPABILITY_UNSUPPORTED = 0,
    FG_MTP_CAPABILITY_WEIGHTS_SEALED = 1,
    FG_MTP_CAPABILITY_ENABLED = 2
} fg_mtp_capability;

typedef struct fg_runtime_options {
    uint32_t logical_context_tokens;
    uint32_t gpu_index_tokens;
    uint32_t qsa_hot_tokens;
    uint64_t qsa_page_cache_bytes;
    uint32_t prefill_microbatch;
    uint32_t prefill_window;
    uint32_t experimental_flags;
    uint32_t specified;
    bool no_prefix_continuation;
} fg_runtime_options;

typedef struct fg_runtime_profile_definition {
    uint32_t id;
    const char *name;
    uint32_t logical_context_tokens;
    uint32_t gpu_index_tokens;
    uint32_t qsa_hot_tokens;
    uint64_t qsa_page_cache_bytes;
    uint32_t prefill_microbatch;
    uint32_t prefill_window;
    uint32_t max_context;
    fg_position_mode position_mode;
} fg_runtime_profile_definition;

typedef struct fg_generation_stats {
    uint32_t prompt_tokens;
    uint32_t prefilled_tokens;
    uint32_t reused_tokens;
    uint32_t generated_tokens;
    uint32_t context_tokens;
    uint32_t image_tokens;
    uint32_t video_tokens;
    uint32_t video_frames;
    bool prefix_cache_hit;
    bool exact_frontier;
    fg_prefix_reset_reason reset_reason;
    fg_execution_mode execution_mode;
    double prefill_seconds;
    double decode_seconds;
    double tower_seconds;
} fg_generation_stats;
typedef fg_status (*fg_token_callback)(void *context,uint32_t token,const char *text,
                                      size_t bytes,fg_error *err);
typedef bool (*fg_interrupt_fn)(void *context);

/* M3.2 multiplex runner (text-only, penalty-free): drive a bound session one
 * phase at a time so an engine scheduler can share the ring between two
 * sessions.  The runner mirrors the production B=1 phase order (plan, prefill
 * pipeline + output head, decode loop) but returns between chunks/steps.  A
 * lone active session never uses these entry points; it stays on
 * fg_runtime_generate*. */
fg_status fg_runtime_session_runner_begin(fg_runtime *runtime,fg_runtime_session *session,
    const char *transcript,uint32_t max_tokens,const fg_sampler_config *sampler,
    fg_generation_stats *stats,fg_error *err);
/* Resume a session that stopped mid-decode (clean yield or client stop): the
 * rendered transcript and pending token come from the bound session state. */
fg_status fg_runtime_session_runner_resume_decode(fg_runtime *runtime,
    fg_runtime_session *session,uint32_t max_tokens,fg_generation_stats *stats,
    fg_error *err);
/* Prefill at most `token_budget` prompt tokens (one pipeline group); `done` is
 * set when the prompt is complete and the first sampled token is pending. */
fg_status fg_runtime_session_runner_prefill(fg_runtime *runtime,fg_runtime_session *session,
    uint32_t token_budget,bool *done,fg_error *err);
/* Emit every session's pending token, then run one depth-B ring step for the
 * batch table.  Sessions whose pending token is EOS or that reached
 * max_tokens are reported in `left`; they have left the table already.
 * `callbacks`/`contexts` are indexed by the caller's session array. */
fg_status fg_runtime_session_runner_batch(fg_runtime *runtime,
    fg_runtime_session **sessions,uint32_t count,fg_decode_batch_table *table,
    const fg_decode_batch_policy *policy,
    fg_token_callback callbacks[FG_DECODE_BATCH_MAX_SLOTS],
    void *contexts[FG_DECODE_BATCH_MAX_SLOTS],uint64_t now,bool left[],
    fg_error *err);
/* Commit the runner's rendered transcript/frontier/stats; the session stays
 * bound and must be ended by the caller. */
fg_status fg_runtime_session_runner_finish(fg_runtime *runtime,fg_runtime_session *session,
    fg_error *err);
uint32_t fg_runtime_session_runner_generated(const fg_runtime_session *session);
bool fg_runtime_session_runner_active(const fg_runtime_session *session);

enum {
    FG_RUNTIME_MEDIA_IMAGE = 1u,
    FG_RUNTIME_MEDIA_VIDEO = 2u,
    FG_RUNTIME_MEDIA_VIDEO_FRAMES = 3u
};

typedef struct fg_runtime_media {
    uint32_t kind;
    const uint8_t *bytes;
    size_t length;
    const uint8_t *const *frames;
    const size_t *frame_lengths;
    uint32_t frame_count;
    double fps;
    uint32_t max_frames;
} fg_runtime_media;

void fg_runtime_options_init(fg_runtime_options *options);
const fg_runtime_profile_definition *fg_runtime_profile_definition_get(uint32_t profile);
fg_status fg_runtime_profile_parse(const char *name,uint32_t *profile,fg_error *err);
fg_status fg_runtime_profile_validate(const fg_runtime_options *options,
                                      uint32_t native_context,fg_error *err);
fg_status fg_runtime_profile_apply(fg_manifest *manifest,uint32_t profile,fg_error *err);
fg_status fg_runtime_options_resolve(fg_runtime_options *resolved,
                                     const fg_manifest *manifest,
                                     const fg_runtime_options *requested,
                                     fg_error *err);
fg_status fg_runtime_eval_capacity(uint32_t *qsa_capacity,
                                   const fg_runtime_options *options,
                                   size_t prompt_tokens,uint32_t generation_tokens,
                                   fg_error *err);
fg_status fg_runtime_open(fg_runtime **out,const char *manifest_path,fg_error *err);
fg_status fg_runtime_open_with_options(fg_runtime **out,const char *manifest_path,
                                       const fg_runtime_options *options,fg_error *err);
void fg_runtime_close(fg_runtime *runtime);
fg_status fg_runtime_reset(fg_runtime *runtime,fg_error *err);
fg_status fg_runtime_reset_public_history(fg_runtime *runtime,fg_error *err);
fg_status fg_runtime_reset_failure(fg_runtime *runtime,fg_error *err);
fg_status fg_runtime_set_sampler(fg_runtime *runtime,const fg_sampler_config *config,fg_error *err);
fg_status fg_runtime_generate(fg_runtime *runtime,const char *rendered_transcript,
                              uint32_t max_tokens,
                              fg_token_callback callback,void *callback_context,
                              fg_interrupt_fn interrupted,void *interrupt_context,
                              fg_generation_stats *stats,fg_error *err);
fg_status fg_runtime_generate_continuation(
    fg_runtime *runtime,const char *public_transcript,
    const char *rendered_continuation,bool *prefix_miss,
    uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err);
fg_status fg_runtime_generate_vision(fg_runtime *runtime,const char *transcript,
                                     const fg_runtime_media *media,uint32_t media_count,
                                     uint32_t max_tokens,
                                     fg_token_callback callback,void *callback_context,
                                     fg_interrupt_fn interrupted,void *interrupt_context,
                                     fg_generation_stats *stats,fg_error *err);
/* Continue a stored session whose new suffix media are the tower inputs; the
 * prefix media stay in the reused owner state.  `media` holds only the new
 * suffix media, in transcript order. */
fg_status fg_runtime_generate_vision_continuation(
    fg_runtime *runtime,const char *public_transcript,const char *continuation,
    const fg_runtime_media *media,uint32_t media_count,
    bool *prefix_miss,uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err);
bool fg_runtime_vision_available(const fg_runtime *runtime);
bool fg_runtime_video_available(const fg_runtime *runtime);
bool fg_runtime_video_frames_available(const fg_runtime *runtime);
uint32_t fg_runtime_context_tokens(const fg_runtime *runtime);
uint32_t fg_runtime_context_limit(const fg_runtime *runtime);
const char *fg_runtime_ledger(const fg_runtime *runtime);
const char *fg_runtime_model_name(const fg_runtime *runtime);
fg_execution_mode fg_runtime_execution_mode(const fg_runtime *runtime);
const char *fg_execution_mode_name(fg_execution_mode mode);
fg_mtp_capability fg_runtime_mtp_capability(const fg_runtime *runtime);

fg_status fg_rank_main(const char *manifest_path, uint32_t rank, fg_error *err);
/* Test-only decode A/B switches (see depth-b-selftest --serial-batch /
 * --serial-expert); applied by the worker `rank` command for A/B windows. */
void fg_runtime_set_decode_ab(bool serial_batch,bool serial_expert);
/* Test-only depth-B parity gate: two canned conversations run at depth 1 and
 * interleaved through the B=2 ring, with exact token/logit/state comparison and
 * an injected abort-retry.  Not part of the serving path. */
fg_status fg_depthb_selftest_main(const char *manifest_path,uint32_t depth,
                                  uint32_t max_tokens,uint32_t long_tokens,
                                  uint32_t abort_step,bool serial_batch,bool serial_expert,
                                  const fg_runtime_options *requested,fg_error *err);
fg_status fg_serve_main(const char *manifest_path, fg_error *err);
fg_status fg_bench_main(const char *manifest_path, fg_error *err);
fg_status fg_eval_main(const char *manifest_path,const char *prompt,uint32_t generate,fg_error *err);

#endif
