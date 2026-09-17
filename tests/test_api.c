#include "fg_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../src/api.c"

static int failures;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);         \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

static char *run_chat_request(fg_runtime *runtime, api_public_session *session,
                              const char *body, fg_status *result);

struct fg_runtime {
    char *history;
    size_t history_length;
    size_t evaluated_length;
    fg_prefix_reset_reason empty_reason;
    const char *generated;
    bool fail_after_prefill;
    bool force_continuation_miss;
    bool require_clean_generation;
    uint32_t reset_count;
    uint32_t sampler_set_count;
    fg_sampler_config last_sampler;
};

fg_status fg_runtime_set_sampler(fg_runtime *runtime,const fg_sampler_config *config,
                                 fg_error *err){
    (void)err;
    if(runtime&&config){runtime->sampler_set_count++;runtime->last_sampler=*config;}
    return FG_OK;
}

const char *fg_execution_mode_name(fg_execution_mode mode){
    return mode==FG_EXECUTION_EXPERT_PARALLEL?"expert-parallel":"unsupported";
}

static fg_mtp_capability test_mtp_capability = FG_MTP_CAPABILITY_UNSUPPORTED;

fg_mtp_capability fg_runtime_mtp_capability(const fg_runtime *runtime){
    (void)runtime;
    return test_mtp_capability;
}

fg_status fg_runtime_open(fg_runtime **out, const char *manifest_path, fg_error *err) {
    (void)out;
    (void)manifest_path;
    fg_error_set(err, FG_ERR_UNAVAILABLE, "test runtime");
    return FG_ERR_UNAVAILABLE;
}

fg_status fg_runtime_open_with_options(fg_runtime **out, const char *manifest_path,
                                       const fg_runtime_options *options, fg_error *err) {
    (void)options;
    return fg_runtime_open(out, manifest_path, err);
}

void fg_runtime_close(fg_runtime *runtime) {
    if (!runtime) return;
    free(runtime->history);
    runtime->history = NULL;
    runtime->history_length = 0;
    runtime->evaluated_length = 0;
}

fg_status fg_runtime_reset(fg_runtime *runtime, fg_error *err) {
    (void)err;
    if (runtime) {
        free(runtime->history);
        runtime->history = NULL;
        runtime->history_length = 0;
        runtime->evaluated_length = 0;
        runtime->empty_reason = FG_PREFIX_RESET_EXPLICIT;
        runtime->reset_count++;
    }
    return FG_OK;
}

fg_status fg_runtime_reset_public_history(fg_runtime *runtime,fg_error *err) {
    fg_status status=fg_runtime_reset(runtime,err);
    if(status==FG_OK&&runtime)runtime->empty_reason=FG_PREFIX_RESET_PUBLIC_MISMATCH;
    return status;
}

fg_status fg_runtime_reset_failure(fg_runtime *runtime,fg_error *err) {
    fg_status status=fg_runtime_reset(runtime,err);
    if(status==FG_OK&&runtime)runtime->empty_reason=FG_PREFIX_RESET_FAILURE;
    return status;
}

fg_status fg_runtime_generate(fg_runtime *runtime, const char *rendered_transcript,
                              uint32_t max_tokens, fg_token_callback callback,
                              void *callback_context, fg_interrupt_fn interrupted,
                              void *interrupt_context, fg_generation_stats *stats,
                              fg_error *err) {
    (void)max_tokens;
    (void)interrupted;
    (void)interrupt_context;
    size_t rendered_length = strlen(rendered_transcript);
    bool hit = runtime && runtime->evaluated_length &&
               rendered_length >= runtime->evaluated_length &&
               !memcmp(rendered_transcript, runtime->history, runtime->evaluated_length);
    fg_prefix_reset_reason reason = FG_PREFIX_RESET_NONE;
    size_t reused = hit ? runtime->evaluated_length : 0u;
    if (!hit) {
        reason = runtime && runtime->history_length ? FG_PREFIX_RESET_TOKEN_MISMATCH :
                 runtime ? runtime->empty_reason : FG_PREFIX_RESET_COLD_START;
    }
    if (stats) {
        memset(stats, 0, sizeof(*stats));
        stats->prompt_tokens = (uint32_t)rendered_length;
        stats->prefilled_tokens = (uint32_t)(rendered_length - reused);
        stats->reused_tokens = (uint32_t)reused;
        stats->prefix_cache_hit = hit;
        stats->exact_frontier = hit && reused == rendered_length;
        stats->reset_reason = reason;
        stats->prefill_seconds = reused == rendered_length ? 0.0 : 1.0;
    }
    if (!runtime) {
        fg_error_set(err, FG_ERR_UNAVAILABLE, "test runtime");
        return FG_ERR_UNAVAILABLE;
    }
    if (runtime->require_clean_generation && runtime->history_length) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "fake runtime generation began with stale prefix metadata");
        return FG_ERR_MISMATCH;
    }
    runtime->require_clean_generation = false;
    char *history = strdup(rendered_transcript);
    if (!history) {
        fg_error_set(err, FG_ERR_OOM, "copy fake runtime history");
        return FG_ERR_OOM;
    }
    free(runtime->history);
    runtime->history = history;
    runtime->history_length = rendered_length;
    if (runtime->fail_after_prefill) {
        runtime->fail_after_prefill = false;
        free(runtime->history);
        runtime->history = NULL;
        runtime->history_length = 0;
        runtime->evaluated_length = 0;
        runtime->empty_reason = FG_PREFIX_RESET_FAILURE;
        fg_error_set(err, FG_ERR_MISMATCH, "injected generation failure");
        return FG_ERR_MISMATCH;
    }
    const char *generated = runtime->generated ? runtime->generated :
        "hidden\n</think>\n\nanswer";
    fg_status status = callback(callback_context, 42u, generated, strlen(generated), err);
    if (status != FG_OK) {
        free(runtime->history);
        runtime->history = NULL;
        runtime->history_length = 0;
        runtime->evaluated_length = 0;
        runtime->empty_reason = FG_PREFIX_RESET_FAILURE;
        return status;
    }
    size_t generated_length = strlen(generated);
    static const char boundary[]="<|im_end|>\n";
    size_t boundary_length=sizeof(boundary)-1u;
    char *committed = realloc(runtime->history,
                              rendered_length+generated_length+boundary_length+1u);
    if (!committed) {
        free(runtime->history);
        runtime->history = NULL;
        runtime->history_length = 0;
        runtime->evaluated_length = 0;
        runtime->empty_reason = FG_PREFIX_RESET_FAILURE;
        fg_error_set(err, FG_ERR_OOM, "commit fake runtime history");
        return FG_ERR_OOM;
    }
    memcpy(committed+rendered_length,generated,generated_length);
    memcpy(committed+rendered_length+generated_length,boundary,boundary_length+1u);
    runtime->history = committed;
    runtime->history_length=rendered_length+generated_length+boundary_length;
    runtime->evaluated_length=rendered_length+generated_length;
    runtime->empty_reason = FG_PREFIX_RESET_NONE;
    if (stats) {
        stats->generated_tokens = 1u;
        stats->context_tokens = (uint32_t)runtime->evaluated_length;
        stats->decode_seconds = 0.5;
    }
    return FG_OK;
}

fg_status fg_runtime_generate_continuation(
    fg_runtime *runtime,const char *public_transcript,
    const char *rendered_continuation,bool *prefix_miss,
    uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err) {
    (void)public_transcript;
    if(prefix_miss)*prefix_miss=false;
    if(runtime&&runtime->force_continuation_miss){
        runtime->force_continuation_miss=false;
        if(prefix_miss)*prefix_miss=true;
        fg_error_set(err,FG_ERR_UNAVAILABLE,
                     "injected authoritative continuation miss");
        return FG_ERR_UNAVAILABLE;
    }
    if(!runtime||!runtime->history){
        if(prefix_miss)*prefix_miss=true;
        fg_error_set(err,FG_ERR_UNAVAILABLE,"fake runtime has no continuation");
        return FG_ERR_UNAVAILABLE;
    }
    size_t continuation_length=strlen(rendered_continuation);
    char *combined=malloc(runtime->history_length+continuation_length+1u);
    if(!combined){
        fg_error_set(err,FG_ERR_OOM,"build fake runtime continuation");
        return FG_ERR_OOM;
    }
    memcpy(combined,runtime->history,runtime->history_length);
    memcpy(combined+runtime->history_length,rendered_continuation,continuation_length+1u);
    fg_status status=fg_runtime_generate(runtime,combined,max_tokens,callback,
                                         callback_context,interrupted,
                                         interrupt_context,stats,err);
    free(combined);
    return status;
}

static bool test_vision_available = false;
static bool test_video_available = false;
static bool test_video_frames_available = false;
static uint32_t test_vision_calls = 0;
static uint32_t test_vision_media = 0;
static uint32_t test_vision_images = 0;
static uint32_t test_vision_videos = 0;
static uint32_t test_vision_video_frames = 0;
static uint32_t test_vision_video_frame_count = 0;
static size_t test_vision_bytes = 0;

bool fg_runtime_vision_available(const fg_runtime *runtime) {
    return runtime != NULL && test_vision_available;
}

bool fg_runtime_video_available(const fg_runtime *runtime) {
    return runtime != NULL && test_video_available;
}

bool fg_runtime_video_frames_available(const fg_runtime *runtime) {
    return runtime != NULL && test_video_frames_available;
}

fg_status fg_runtime_generate_vision(fg_runtime *runtime,const char *transcript,
                                     const fg_runtime_media *media,uint32_t media_count,
                                     uint32_t max_tokens,fg_token_callback callback,
                                     void *callback_context,fg_interrupt_fn interrupted,
                                     void *interrupt_context,fg_generation_stats *stats,
                                     fg_error *err) {
    test_vision_calls++;
    test_vision_media = media_count;
    test_vision_images = 0;
    test_vision_videos = 0;
    test_vision_video_frames = 0;
    test_vision_video_frame_count = 0;
    test_vision_bytes = 0;
    for (uint32_t i = 0; i < media_count; i++) {
        if (media[i].kind == FG_RUNTIME_MEDIA_VIDEO) {
            test_vision_videos++;
            test_vision_bytes += media[i].length;
        } else if (media[i].kind == FG_RUNTIME_MEDIA_VIDEO_FRAMES) {
            test_vision_video_frames++;
            test_vision_video_frame_count += media[i].frame_count;
            for (uint32_t f = 0; f < media[i].frame_count; f++)
                test_vision_bytes += media[i].frame_lengths[f];
        } else {
            test_vision_images++;
            test_vision_bytes += media[i].length;
        }
    }
    fg_status status = fg_runtime_generate(runtime, transcript, max_tokens, callback,
                                           callback_context, interrupted, interrupt_context,
                                           stats, err);
    if (status == FG_OK && stats) {
        stats->image_tokens = 4u;
        stats->tower_seconds = 1.1;
    }
    return status;
}

uint32_t fg_runtime_context_tokens(const fg_runtime *runtime) {
    return runtime ? (uint32_t)runtime->evaluated_length : 0;
}

uint32_t fg_runtime_context_limit(const fg_runtime *runtime) {
    (void)runtime;
    return 8192u;
}

const char *fg_runtime_model_name(const fg_runtime *runtime) {
    (void)runtime;
    return "Qwen3.8-Flash-Next";
}

static char *read_socket_response(int fd) {
    api_buffer response = {0};
    char chunk[1024];
    for (;;) {
        ssize_t count = recv(fd, chunk, sizeof(chunk), 0);
        if (count < 0) {
            perror("recv");
            failures++;
            break;
        }
        if (!count) break;
        fg_error err = {0};
        if (buffer_append_n(&response, chunk, (size_t)count, &err) != FG_OK) {
            fprintf(stderr, "response allocation failed: %s\n", err.message);
            failures++;
            break;
        }
    }
    return response.data;
}

static void test_openai_tools_request(void) {
    const char *body =
        "{"
        "\"model\":\"Qwen3.8-Flash-Next\","
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"weather\", \"description\":\"Get weather\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}}}}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"weather\"}},"
        "\"messages\":["
        "{\"role\":\"user\",\"content\":\"Weather?\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{"
        "\"id\":\"call_old\",\"type\":\"function\",\"function\":{"
        "\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_old\",\"content\":\"20 C\"}"
        "]}";
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    api_chat_request request = {0};
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.tool_schema_count == 1);
    CHECK(request.tool_schemas[0] != NULL);
    CHECK(strcmp(request.tool_schemas[0],
                 "{\"name\":\"weather\",\"description\":\"Get weather\","
                 "\"parameters\":{\"type\":\"object\",\"properties\":{"
                 "\"city\":{\"type\":\"string\"}}}}") == 0);
    CHECK(request.tool_choice == FG_CHAT_TOOL_NAMED);
    CHECK(strcmp(request.tool_choice_name, "weather") == 0);
    CHECK(request.messages[1].tool_call_count == 1);
    CHECK(strcmp(request.messages[1].tool_calls[0].id, "call_old") == 0);
    CHECK(strcmp(request.messages[2].tool_call_id, "call_old") == 0);

    fg_chat_render_options options = {
        .tool_schemas = (const char *const *)request.tool_schemas,
        .tool_schema_count = request.tool_schema_count,
        .tool_choice = request.tool_choice,
        .tool_choice_name = request.tool_choice_name,
    };
    char *prompt = NULL;
    CHECK(fg_chat_render(request.messages, request.message_count, &options, &prompt, &err) ==
          FG_OK);
    CHECK(prompt && strstr(prompt, "<function=weather>"));
    CHECK(prompt && strstr(prompt, "<tool_response>\n20 C\n</tool_response>"));
    CHECK(prompt && strstr(prompt, "You must call only the function \"weather\""));
    free(prompt);
    api_chat_request_free(&request);
    json_free(root);
}

static void test_openai_structured_text_content(void) {
    const char *body =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"Hey Qwen, \"},"
        "{\"type\":\"input_text\",\"text\":\"can you see this repo?\"}]}]}";
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    api_chat_request request = {0};
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.message_count == 1u);
    CHECK(request.messages[0].content != NULL);
    CHECK(strcmp(request.messages[0].content,
                 "Hey Qwen, can you see this repo?") == 0);
    api_chat_request_free(&request);
    json_free(root);

    const char *image =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.invalid/x.png\"}}]}]}";
    memset(&err, 0, sizeof(err));
    root = parse_json_body(image, strlen(image), &err);
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "image_url") != NULL);
    api_chat_request_free(&request);
    json_free(root);
}

static const char test_png_base64[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAA"
    "AABJRU5ErkJggg==";

static void test_image_content_parts(void) {
    char body[4096];
    snprintf(body, sizeof(body),
             "{\"messages\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"text\",\"text\":\"what is this? \"},"
             "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,%s\"}}]}]}",
             test_png_base64);
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    CHECK(root != NULL);
    api_chat_request request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.media_count == 1u);
    CHECK(request.media[0].kind == FG_RUNTIME_MEDIA_IMAGE);
    CHECK(request.media[0].length > 50u);
    CHECK(request.media[0].data[0] == 0x89u && request.media[0].data[1] == 'P' &&
          request.media[0].data[2] == 'N' && request.media[0].data[3] == 'G');
    CHECK(request.messages[0].content &&
          !strncmp(request.messages[0].content, "what is this? <|vision_start|>", 30u));
    CHECK(request.messages[0].content &&
          strstr(request.messages[0].content, "<|image_pad|><|vision_end|>"));
    api_chat_request_free(&request);
    json_free(root);

    const char *plain =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"plain text\"}]}";
    root = parse_json_body(plain, strlen(plain), &err);
    CHECK(root != NULL);
    api_chat_request plain_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &plain_request, &err) == FG_OK);
    CHECK(plain_request.media_count == 0u);
    CHECK(!strcmp(plain_request.messages[0].content, "plain text"));
    CHECK(!strstr(plain_request.messages[0].content, "<|vision"));
    api_chat_request_free(&plain_request);
    json_free(root);

    const char *http_url =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/a.png\"}}]}]}";
    root = parse_json_body(http_url, strlen(http_url), &err);
    CHECK(root != NULL);
    api_chat_request http_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &http_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "http(s)") != NULL);
    api_chat_request_free(&http_request);
    json_free(root);

    const char *webp =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/webp;base64,AAAA\"}}]}]}";
    root = parse_json_body(webp, strlen(webp), &err);
    CHECK(root != NULL);
    api_chat_request webp_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &webp_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "media type") != NULL);
    api_chat_request_free(&webp_request);
    json_free(root);

    const char *broken =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,!!!!\"}}]}]}";
    root = parse_json_body(broken, strlen(broken), &err);
    CHECK(root != NULL);
    api_chat_request broken_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &broken_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "base64") != NULL);
    api_chat_request_free(&broken_request);
    json_free(root);
}

static void test_video_content_parts(void) {
    char body[4096];
    snprintf(body, sizeof(body),
             "{\"messages\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"text\",\"text\":\"what happens? \"},"
             "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,AAAA\"}}]}]}");
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    CHECK(root != NULL);
    api_chat_request request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.media_count == 1u);
    CHECK(request.media[0].kind == FG_RUNTIME_MEDIA_VIDEO);
    CHECK(request.media[0].length == 3u);
    CHECK(request.messages[0].content &&
          !strncmp(request.messages[0].content, "what happens? <|vision_start|>", 30u));
    CHECK(request.messages[0].content &&
          strstr(request.messages[0].content, "<|video_pad|><|vision_end|>"));
    api_chat_request_free(&request);
    json_free(root);

    const char *http_url =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"https://example.com/a.mp4\"}}]}]}";
    root = parse_json_body(http_url, strlen(http_url), &err);
    CHECK(root != NULL);
    api_chat_request http_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &http_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "video_url") != NULL);
    CHECK(strstr(err.message, "http(s)") != NULL);
    api_chat_request_free(&http_request);
    json_free(root);

    const char *bad_type =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/quicktime;base64,AAAA\"}}]}]}";
    root = parse_json_body(bad_type, strlen(bad_type), &err);
    CHECK(root != NULL);
    api_chat_request bad_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &bad_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "media type") != NULL);
    api_chat_request_free(&bad_request);
    json_free(root);

    const char *broken =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,!!!!\"}}]}]}";
    root = parse_json_body(broken, strlen(broken), &err);
    CHECK(root != NULL);
    api_chat_request broken_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &broken_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "base64") != NULL);
    api_chat_request_free(&broken_request);
    json_free(root);
}

static void test_video_http_flow(void) {
    const char *body =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,AAAA\"}},"
        "{\"type\":\"text\",\"text\":\"describe this\"}]}]}";
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    test_vision_available = true;
    test_video_available = false;
    test_vision_calls = 0;
    char *response = run_chat_request(&runtime, &session, body, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "400 Bad Request"));
    CHECK(response && strstr(response, "MP4 video input is not available"));
    CHECK(test_vision_calls == 0u);
    free(response);
    test_video_available = true;
    response = run_chat_request(&runtime, &session, body, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "200 OK"));
    CHECK(test_vision_calls == 1u);
    CHECK(test_vision_media == 1u);
    CHECK(test_vision_videos == 1u);
    CHECK(test_vision_images == 0u);
    CHECK(test_vision_bytes == 3u);
    CHECK(!session.valid);
    free(response);
    test_vision_available = false;
    test_video_available = false;
}

static void test_video_frames_content_parts(void) {
    char body[8192];
    snprintf(body, sizeof(body),
             "{\"messages\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"text\",\"text\":\"what happens? \"},"
             "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":["
             "\"data:image/png;base64,%s\",\"data:image/png;base64,%s\"],"
             "\"fps\":4.0,\"max_frames\":2}}]}]}",
             test_png_base64, test_png_base64);
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    CHECK(root != NULL);
    api_chat_request request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.media_count == 1u);
    CHECK(request.media[0].kind == FG_RUNTIME_MEDIA_VIDEO_FRAMES);
    CHECK(request.media[0].frame_count == 2u);
    CHECK(request.media[0].frame_lengths[0] > 50u);
    CHECK(request.media[0].frame_lengths[1] > 50u);
    CHECK(request.media[0].frames[0][0] == 0x89u && request.media[0].frames[0][1] == 'P');
    CHECK(request.media[0].fps == 4.0);
    CHECK(request.media[0].max_frames == 2u);
    CHECK(request.messages[0].content &&
          !strncmp(request.messages[0].content, "what happens? <|vision_start|>", 30u));
    CHECK(request.messages[0].content &&
          strstr(request.messages[0].content, "<|video_pad|><|vision_end|>"));
    api_chat_request_free(&request);
    json_free(root);

    const char *plain_frames =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[\"data:image/png;base64,AAAA\"]}}]}]}";
    root = parse_json_body(plain_frames, strlen(plain_frames), &err);
    CHECK(root != NULL);
    api_chat_request default_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &default_request, &err) == FG_OK);
    CHECK(default_request.media[0].fps == 2.0);
    CHECK(default_request.media[0].max_frames == 8u);
    api_chat_request_free(&default_request);
    json_free(root);

    const char *missing_object =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":[]}]}]}";
    root = parse_json_body(missing_object, strlen(missing_object), &err);
    CHECK(root != NULL);
    api_chat_request missing_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &missing_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "video_frames object") != NULL);
    api_chat_request_free(&missing_request);
    json_free(root);

    const char *empty_frames =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[]}}]}]}";
    root = parse_json_body(empty_frames, strlen(empty_frames), &err);
    CHECK(root != NULL);
    api_chat_request empty_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &empty_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "non-empty frames") != NULL);
    api_chat_request_free(&empty_request);
    json_free(root);

    const char *non_string =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[7]}}]}]}";
    root = parse_json_body(non_string, strlen(non_string), &err);
    CHECK(root != NULL);
    api_chat_request non_string_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &non_string_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "base64 image data URLs") != NULL);
    api_chat_request_free(&non_string_request);
    json_free(root);

    const char *http_frame =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[\"https://example.com/a.png\"]}}]}]}";
    root = parse_json_body(http_frame, strlen(http_frame), &err);
    CHECK(root != NULL);
    api_chat_request http_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &http_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "video_frames frame") != NULL);
    CHECK(strstr(err.message, "http(s)") != NULL);
    api_chat_request_free(&http_request);
    json_free(root);

    const char *bad_fps =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[\"data:image/png;base64,AAAA\"],"
        "\"fps\":0}}]}]}";
    root = parse_json_body(bad_fps, strlen(bad_fps), &err);
    CHECK(root != NULL);
    api_chat_request bad_fps_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &bad_fps_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "fps") != NULL);
    api_chat_request_free(&bad_fps_request);
    json_free(root);

    const char *bad_max =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[\"data:image/png;base64,AAAA\"],"
        "\"max_frames\":99}}]}]}";
    root = parse_json_body(bad_max, strlen(bad_max), &err);
    CHECK(root != NULL);
    api_chat_request bad_max_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &bad_max_request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "max_frames") != NULL);
    api_chat_request_free(&bad_max_request);
    json_free(root);

    const char *mixed =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[\"data:image/png;base64,AAAA\"]}},"
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]}]}";
    root = parse_json_body(mixed, strlen(mixed), &err);
    CHECK(root != NULL);
    api_chat_request mixed_request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &mixed_request, &err) == FG_OK);
    CHECK(mixed_request.media_count == 2u);
    CHECK(mixed_request.media[0].kind == FG_RUNTIME_MEDIA_VIDEO_FRAMES);
    CHECK(mixed_request.media[1].kind == FG_RUNTIME_MEDIA_IMAGE);
    CHECK(mixed_request.messages[0].content &&
          strstr(mixed_request.messages[0].content, "<|video_pad|>") != NULL);
    api_chat_request_free(&mixed_request);
    json_free(root);
}

static void test_video_frames_too_many(void) {
    char *body = malloc(65536);
    CHECK(body != NULL);
    if (!body) return;
    size_t offset = (size_t)snprintf(body, 65536,
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":[");
    for (size_t i = 0; i < 257u; i++) {
        offset += (size_t)snprintf(body + offset, 65536 - offset, "%s\"data:image/png;base64,AA\"",
                                   i ? "," : "");
    }
    snprintf(body + offset, 65536 - offset, "]}}]}]}");
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    CHECK(root != NULL);
    api_chat_request request = {0};
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_ERR_LIMIT);
    CHECK(strstr(err.message, "at most") != NULL);
    api_chat_request_free(&request);
    json_free(root);
    free(body);
}

static void test_video_frames_http_flow(void) {
    char body[8192];
    snprintf(body, sizeof(body),
             "{\"messages\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"video_frames\",\"video_frames\":{\"frames\":["
             "\"data:image/png;base64,%s\",\"data:image/png;base64,%s\"],\"fps\":2.0}},"
             "{\"type\":\"text\",\"text\":\"describe this\"}]}]}",
             test_png_base64, test_png_base64);
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    test_vision_available = true;
    test_video_frames_available = false;
    test_vision_calls = 0;
    char *response = run_chat_request(&runtime, &session, body, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "400 Bad Request"));
    CHECK(response && strstr(response, "video frame input is not available"));
    CHECK(test_vision_calls == 0u);
    free(response);
    test_video_frames_available = true;
    response = run_chat_request(&runtime, &session, body, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "200 OK"));
    CHECK(test_vision_calls == 1u);
    CHECK(test_vision_media == 1u);
    CHECK(test_vision_video_frames == 1u);
    CHECK(test_vision_video_frame_count == 2u);
    CHECK(test_vision_images == 0u);
    CHECK(test_vision_videos == 0u);
    CHECK(test_vision_bytes > 100u);
    CHECK(!session.valid);
    free(response);
    test_vision_available = false;
    test_video_frames_available = false;
}

static void test_unknown_tool_result_rejected(void) {
    const char *body =
        "{\"messages\":[{\"role\":\"tool\",\"tool_call_id\":\"missing\","
        "\"content\":\"nope\"}]}";
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    api_chat_request request = {0};
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "unknown prior tool call id") != NULL);
    api_chat_request_free(&request);
    json_free(root);
}

static void test_tool_choice_modes(void) {
    static const char *bodies[] = {
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "{\"tool_choice\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\","
        "\"parameters\":{}}}],\"tool_choice\":\"auto\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\","
        "\"parameters\":{}}}],\"tool_choice\":\"required\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
    };
    static const fg_chat_tool_choice expected[] = {
        FG_CHAT_TOOL_AUTO,
        FG_CHAT_TOOL_NONE,
        FG_CHAT_TOOL_AUTO,
        FG_CHAT_TOOL_REQUIRED,
    };
    for (size_t i = 0; i < sizeof(bodies) / sizeof(bodies[0]); i++) {
        fg_error err = {0};
        json_value *root = parse_json_body(bodies[i], strlen(bodies[i]), &err);
        api_chat_request request = {0};
        CHECK(root != NULL);
        CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
        CHECK(request.tool_choice == expected[i]);
        api_chat_request_free(&request);
        json_free(root);
    }

    const char *required_without_tools =
        "{\"tool_choice\":\"required\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    fg_error err = {0};
    json_value *root =
        parse_json_body(required_without_tools, strlen(required_without_tools), &err);
    api_chat_request request = {0};
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) ==
          FG_ERR_ARGUMENT);
    api_chat_request_free(&request);
    json_free(root);
}

static void test_history_reasoning_and_empty_calls(void) {
    const char *body =
        "{\"messages\":["
        "{\"role\":\"assistant\",\"content\":\"done\","
        "\"reasoning_content\":\"private\",\"tool_calls\":null},"
        "{\"role\":\"assistant\",\"content\":\"still done\",\"tool_calls\":[]},"
        "{\"role\":\"user\",\"content\":\"next\"}]}";
    fg_error err = {0};
    json_value *root = parse_json_body(body, strlen(body), &err);
    api_chat_request request = {0};
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.message_count == 3);
    CHECK(request.messages[0].reasoning != NULL);
    CHECK(strcmp(request.messages[0].reasoning, "private") == 0);
    CHECK(request.messages[0].tool_call_count == 0);
    CHECK(request.messages[1].tool_call_count == 0);
    char *prompt = NULL;
    CHECK(fg_chat_render(request.messages, request.message_count, NULL, &prompt, &err) == FG_OK);
    CHECK(prompt && strstr(prompt, "<think>\nprivate\n</think>\n\ndone"));
    free(prompt);
    api_chat_request_free(&request);
    json_free(root);
}

static void test_greedy_controls(void) {
    const char *accepted =
        "{\"temperature\":0,\"top_p\":1,\"n\":1,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    fg_error err = {0};
    json_value *root = parse_json_body(accepted, strlen(accepted), &err);
    api_chat_request request = {0};
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    api_chat_request_free(&request);
    json_free(root);

    const char *large_completion =
        "{\"max_tokens\":32768,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root = parse_json_body(large_completion, strlen(large_completion), &err);
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.max_tokens == 32768u);
    api_chat_request_free(&request);
    json_free(root);

    const char *sampled =
        "{\"temperature\":0.5,\"top_p\":0.8,\"top_k\":20,\"seed\":7,"
        "\"min_p\":0,\"presence_penalty\":1.5,\"frequency_penalty\":0.25,"
        "\"repetition_penalty\":1,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    memset(&err, 0, sizeof(err));
    root = parse_json_body(sampled, strlen(sampled), &err);
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) == FG_OK);
    CHECK(request.sampler.temperature==0.5f&&request.sampler.top_p==0.8f&&
          request.sampler.top_k==20u&&request.sampler.seed==7u&&
          request.sampler.min_p==0.0f&&request.sampler.presence_penalty==1.5f&&
          request.sampler.frequency_penalty==0.25f&&request.sampler.repetition_penalty==1.0f);
    api_chat_request_free(&request);
    json_free(root);

    const char *invalid = "{\"temperature\":-1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root = parse_json_body(invalid, strlen(invalid), &err);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);

    const char *bad_top_p = "{\"top_p\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root=parse_json_body(bad_top_p,strlen(bad_top_p),&err);
    CHECK(parse_chat_request(root,"Qwen3.8-Flash-Next",&request,&err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);
    const char *bad_top_k = "{\"top_k\":65,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root=parse_json_body(bad_top_k,strlen(bad_top_k),&err);
    CHECK(parse_chat_request(root,"Qwen3.8-Flash-Next",&request,&err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);

    const char *bad_presence = "{\"presence_penalty\":2.1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root=parse_json_body(bad_presence,strlen(bad_presence),&err);
    CHECK(parse_chat_request(root,"Qwen3.8-Flash-Next",&request,&err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);
    const char *bad_frequency = "{\"frequency_penalty\":-2.1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root=parse_json_body(bad_frequency,strlen(bad_frequency),&err);
    CHECK(parse_chat_request(root,"Qwen3.8-Flash-Next",&request,&err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);
    const char *bad_repetition = "{\"repetition_penalty\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root=parse_json_body(bad_repetition,strlen(bad_repetition),&err);
    CHECK(parse_chat_request(root,"Qwen3.8-Flash-Next",&request,&err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);
    const char *bad_min_p = "{\"min_p\":0.1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root=parse_json_body(bad_min_p,strlen(bad_min_p),&err);
    CHECK(parse_chat_request(root,"Qwen3.8-Flash-Next",&request,&err)==FG_ERR_ARGUMENT);
    api_chat_request_free(&request);json_free(root);

    const char *defaults = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    root = parse_json_body(defaults, strlen(defaults), &err);
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err)==FG_OK);
    CHECK(request.sampler.temperature==1.0f&&request.sampler.top_p==0.95f&&request.sampler.top_k==20u);
    api_chat_request_free(&request);json_free(root);

    fg_runtime runtime={0};api_public_session session={0};fg_status result=FG_OK;
    char *response=run_chat_request(&runtime,&session,
        "{\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,"
        "\"min_p\":0,\"presence_penalty\":1.5,\"frequency_penalty\":0,"
        "\"repetition_penalty\":1,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",&result);
    CHECK(result==FG_OK&&response!=NULL&&runtime.sampler_set_count==1u);
    CHECK(runtime.last_sampler.temperature==0.7f&&runtime.last_sampler.top_p==0.8f&&
          runtime.last_sampler.top_k==20u&&runtime.last_sampler.min_p==0.0f&&
          runtime.last_sampler.presence_penalty==1.5f&&
          runtime.last_sampler.frequency_penalty==0.0f&&
          runtime.last_sampler.repetition_penalty==1.0f);
    free(response);api_public_session_free(&session);fg_runtime_close(&runtime);
}

static void test_nonstream_tool_response(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    api_generation generation = {
        .fd = sockets[0],
        .id = "chatcmpl-test",
        .model = "Qwen3.8-Flash-Next",
        .created = 1,
    };
    fg_chat_tool_call call = {
        .name = "weather",
        .arguments_json = "{\"city\":\"Paris\"}",
    };
    fg_chat_generated generated = {
        .content = "",
        .tool_calls = &call,
        .tool_call_count = 1,
    };
    fg_generation_stats stats = {
        .prompt_tokens = 10,
        .prefilled_tokens = 10,
        .generated_tokens = 5,
        .context_tokens = 15,
        .reset_reason = FG_PREFIX_RESET_COLD_START,
        .execution_mode = FG_EXECUTION_EXPERT_PARALLEL,
        .prefill_seconds = 2.0,
        .decode_seconds = 0.5,
    };
    fg_error err = {0};
    CHECK(send_completion(&generation, &generated, &stats, "tool_calls", &err) == FG_OK);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\"content\":null"));
    CHECK(response && strstr(response, "\"finish_reason\":\"tool_calls\""));
    CHECK(response && strstr(response, "\"id\":\"call_chatcmpl-test_0\""));
    CHECK(response && strstr(response, "\"name\":\"weather\""));
    CHECK(response && strstr(response, "\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\""));
    CHECK(response && strstr(response, "X-Flash-Gordon-Prompt-Tokens: 10\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Execution-Mode: expert-parallel\r\n"));
    CHECK(response && !strstr(response, "X-Flash-Gordon-Stage-Timings:"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefilled-Tokens: 10\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reused-Tokens: 0\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: cold-start\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Context-Tokens: 15\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefill-TPS: 5.000000\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Decode-TPS: 10.000000\r\n"));
    CHECK(response && !strstr(response, "<tool_call>"));
    free(response);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_streamed_tool_response(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    char *schemas[] = {
        "{\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}",
        "{\"name\":\"clock\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}",
    };
    api_chat_request request = {
        .tool_schemas = schemas,
        .tool_schema_count = 2,
        .tool_choice = FG_CHAT_TOOL_AUTO,
    };
    api_generation generation = {
        .fd = sockets[0],
        .stream = true,
        .id = "chatcmpl-stream",
        .model = "Qwen3.8-Flash-Next",
        .created = 2,
        .request = &request,
    };
    fg_error err = {0};
    CHECK(send_stream_start(&generation, &err) == FG_OK);
    CHECK(api_token(&generation, 1, "hidden", 6, &err) == FG_OK);
    const char *prefix = "</think>\nAnswer before tool\n<tool_";
    CHECK(api_token(&generation, 2, prefix, strlen(prefix), &err) == FG_OK);
    const char *tail =
        "call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=clock>\n<parameter=zone>\nUTC\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(api_token(&generation, 3, tail, strlen(tail), &err) == FG_OK);
    fg_chat_generated generated = {0};
    CHECK(fg_chat_parse_generated(generation.content.data, true, &generated, &err) == FG_OK);
    CHECK(send_stream_end(&generation, &generated, "tool_calls", &err) == FG_OK);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\"content\":\"Answer before tool\""));
    CHECK(response && strstr(response, "\"reasoning_content\":\"hidden\""));
    CHECK(response && strstr(response, "\"tool_calls\":[{\"index\":0"));
    CHECK(response && strstr(response, "\"id\":\"call_chatcmpl-stream_0\""));
    CHECK(response && strstr(response, "\"tool_calls\":[{\"index\":1"));
    CHECK(response && strstr(response, "\"id\":\"call_chatcmpl-stream_1\""));
    CHECK(response && strstr(response, "\"name\":\"clock\""));
    CHECK(response && strstr(response, "\"finish_reason\":\"tool_calls\""));
    CHECK(response && strstr(response, "data: [DONE]"));
    CHECK(response && !strstr(response, "\"content\":\"hidden"));
    CHECK(response && !strstr(response, "<tool_call>"));
    free(response);
    fg_chat_generated_free(&generated);
    free(generation.content.data);
    free(generation.visible_pending.data);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_streamed_incomplete_tags_do_not_leak(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    api_chat_request request = {0};
    api_generation generation = {
        .fd = sockets[0],
        .stream = true,
        .id = "chatcmpl-malformed",
        .model = "Qwen3.8-Flash-Next",
        .created = 3,
        .request = &request,
    };
    fg_error err = {0};
    const char *text = "</think>\nvisible<function";
    CHECK(api_token(&generation, 1, text, strlen(text), &err) == FG_OK);
    CHECK(queue_visible_content(&generation, NULL, 0, true, &err) == FG_ERR_FORMAT);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\"content\":\"visible\""));
    CHECK(response && !strstr(response, "<function"));
    free(response);
    free(generation.content.data);
    free(generation.visible_pending.data);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_streamed_utf8_and_sentinel_filtering(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    api_chat_request request = {0};
    api_generation generation = {
        .fd = sockets[0],
        .stream = true,
        .id = "chatcmpl-utf8",
        .model = "Qwen3.8-Flash-Next",
        .created = 4,
        .request = &request,
    };
    fg_error err = {0};
    const char first[] = "</think>\n\xf0\x9f";
    const char second[] = "\x98\x80<|im_";
    const char third[] = "end|>ignored";
    CHECK(api_token(&generation, 1, first, sizeof(first) - 1u, &err) == FG_OK);
    CHECK(api_token(&generation, 2, second, sizeof(second) - 1u, &err) == FG_OK);
    CHECK(api_token(&generation, 3, third, sizeof(third) - 1u, &err) == FG_OK);
    CHECK(queue_visible_content(&generation, NULL, 0, true, &err) == FG_OK);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\xf0\x9f\x98\x80"));
    CHECK(response && !strstr(response, "\xef\xbf\xbd"));
    CHECK(response && !strstr(response, "<|im_end|>"));
    CHECK(response && !strstr(response, "ignored"));
    free(response);
    free(generation.content.data);
    free(generation.visible_pending.data);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_json_nul_and_member_limit(void) {
    const char *nul =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\\u0000b\"}]}";
    fg_error err = {0};
    json_value *root = parse_json_body(nul, strlen(nul), &err);
    CHECK(root == NULL);
    CHECK(err.code == FG_ERR_FORMAT);
    json_free(root);

    api_buffer object = {0};
    CHECK(buffer_append(&object, "{", &err) == FG_OK);
    for (unsigned i = 0; i <= FG_API_MAX_OBJECT_MEMBERS; i++) {
        char member[48];
        snprintf(member, sizeof(member), "%s\"k%u\":0", i ? "," : "", i);
        CHECK(buffer_append(&object, member, &err) == FG_OK);
    }
    CHECK(buffer_append(&object, "}", &err) == FG_OK);
    memset(&err, 0, sizeof(err));
    root = parse_json_body(object.data, object.length, &err);
    CHECK(root == NULL);
    CHECK(err.code == FG_ERR_LIMIT);
    json_free(root);
    free(object.data);
}

static void test_client_socket_timeouts(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    fg_error err = {0};
    CHECK(configure_client_socket(sockets[0], &err) == FG_OK);
    struct timeval receive_timeout = {0}, send_timeout = {0};
    socklen_t receive_bytes = sizeof(receive_timeout), send_bytes = sizeof(send_timeout);
    CHECK(getsockopt(sockets[0], SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                     &receive_bytes) == 0);
    CHECK(getsockopt(sockets[0], SOL_SOCKET, SO_SNDTIMEO, &send_timeout,
                     &send_bytes) == 0);
    CHECK(receive_timeout.tv_sec == FG_API_IO_TIMEOUT_SECONDS);
    CHECK(send_timeout.tv_sec == FG_API_IO_TIMEOUT_SECONDS);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_model_capabilities(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    fg_error err = {0};
    test_mtp_capability = FG_MTP_CAPABILITY_UNSUPPORTED;
    CHECK(handle_models(sockets[0], NULL, &err) == FG_OK);
    CHECK(shutdown(sockets[0], SHUT_WR) == 0);
    char *response = read_socket_response(sockets[1]);
    CHECK(response != NULL);
    if (response) {
        CHECK(strstr(response, "\"native_context\":8192") != NULL);
        CHECK(strstr(response, "\"experimental_context\":0") != NULL);
        CHECK(strstr(response, "\"tools\":true") != NULL);
        CHECK(strstr(response, "\"mtp\":false") != NULL);
        CHECK(strstr(response, "\"image\":false") != NULL);
        CHECK(strstr(response, "\"video\":false") != NULL);
    }
    free(response);
    close(sockets[0]);
    close(sockets[1]);

    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    test_mtp_capability = FG_MTP_CAPABILITY_ENABLED;
    CHECK(handle_models(sockets[0], NULL, &err) == FG_OK);
    CHECK(shutdown(sockets[0], SHUT_WR) == 0);
    response = read_socket_response(sockets[1]);
    CHECK(response != NULL);
    if (response) CHECK(strstr(response, "\"mtp\":true") != NULL);
    free(response);
    test_mtp_capability = FG_MTP_CAPABILITY_UNSUPPORTED;
    close(sockets[0]);
    close(sockets[1]);
}

static char *run_chat_request(fg_runtime *runtime, api_public_session *session,
                              const char *body, fg_status *result) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    http_request request = {
        .body = (char *)body,
        .body_length = strlen(body),
    };
    fg_error err = {0};
    fg_status status = handle_chat_completions(sockets[0], runtime, session,&request, &err);
    if (result) *result = status;
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    close(sockets[0]);
    close(sockets[1]);
    return response;
}

static json_value *parse_response_json(const char *response,fg_error *err) {
    const char *body=response?strstr(response,"\r\n\r\n"):NULL;
    if(!body){
        fg_error_set(err,FG_ERR_FORMAT,"test response has no HTTP body");
        return NULL;
    }
    body+=4u;
    return parse_json_body(body,strlen(body),err);
}

static void test_live_prefix_hit_divergence_and_reset(void) {
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session={0};
    fg_status status = FG_OK;
    char *response = run_chat_request(
        &runtime, &session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: cold-start\r\n"));
    CHECK(response && strstr(response,"\"reasoning_content\":\"hidden\\n\""));
    CHECK(response && !strstr(response,"<|im_end|>"));
    static const char text_boundary[]="answer<|im_end|>\n";
    CHECK(runtime.history_length>=sizeof(text_boundary)-1u);
    CHECK(runtime.history&&
          !memcmp(runtime.history+runtime.history_length-(sizeof(text_boundary)-1u),
                  text_boundary,sizeof(text_boundary)-1u));
    size_t prior_evaluated=runtime.evaluated_length;
    size_t prior_rendered=runtime.history_length;
    CHECK(prior_rendered-prior_evaluated==strlen("<|im_end|>\n"));
    fg_error err={0};
    json_value *root=parse_response_json(response,&err);
    json_value *choices=json_object_get(root,"choices");
    json_value *choice=choices&&choices->type==JSON_ARRAY&&choices->as.array.count?
        choices->as.array.items[0]:NULL;
    json_value *message=json_object_get(choice,"message");
    json_value *content=json_object_get(message,"content");
    CHECK(content&&content->type==JSON_STRING);
    api_buffer turn_two={0};
    CHECK(buffer_append(&turn_two,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"},"
        "{\"role\":\"assistant\",\"content\":",&err)==FG_OK);
    if(content&&content->type==JSON_STRING)
        CHECK(buffer_append_json_string(&turn_two,content->as.string,
                                        strlen(content->as.string),&err)==FG_OK);
    CHECK(buffer_append(&turn_two,
        "},{\"role\":\"user\",\"content\":\"next\"}]}",&err)==FG_OK);
    json_free(root);
    free(response);

    response = run_chat_request(&runtime,&session,turn_two.data,&status);
    free(turn_two.data);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: hit\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: none\r\n"));
    CHECK(response && !strstr(response, "X-Flash-Gordon-Reused-Tokens: 0\r\n"));
    char reused_header[96],unevaluated_header[96];
    snprintf(reused_header,sizeof(reused_header),
             "X-Flash-Gordon-Reused-Tokens: %zu\r\n",prior_evaluated);
    snprintf(unevaluated_header,sizeof(unevaluated_header),
             "X-Flash-Gordon-Reused-Tokens: %zu\r\n",prior_rendered);
    CHECK(response&&strstr(response,reused_header));
    CHECK(response&&!strstr(response,unevaluated_header));
    CHECK(runtime.history&&strstr(
        runtime.history,
        "answer<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"));
    CHECK(runtime.history&&!strstr(runtime.history,
                                  "answer<|im_end|>\n<|im_end|>"));
    free(response);

    response = run_chat_request(
        &runtime, &session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"different\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(response &&
          strstr(response, "X-Flash-Gordon-Reset-Reason: public-history-mismatch\r\n"));
    free(response);

    memset(&err,0,sizeof(err));
    CHECK(fg_runtime_reset(&runtime, &err) == FG_OK);
    api_public_session_free(&session);
    response = run_chat_request(
        &runtime, &session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"after clear\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: explicit\r\n"));
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_live_prefix_tool_loop(void) {
    fg_runtime runtime = {
        .empty_reason = FG_PREFIX_RESET_COLD_START,
        .generated =
            "hidden\n</think>\n\n"
            "<tool_call>\n<function=weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>",
    };
    api_public_session session={0};
    const char *tools =
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}}}}}],";
    api_buffer first = {0};
    fg_error err = {0};
    CHECK(buffer_append(&first, "{", &err) == FG_OK);
    CHECK(buffer_append(&first, tools, &err) == FG_OK);
    CHECK(buffer_append(&first,
        "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}]}", &err) == FG_OK);
    fg_status status = FG_OK;
    char *response = run_chat_request(&runtime, &session,first.data, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "\"finish_reason\":\"tool_calls\""));
    CHECK(response && strstr(response,"\"reasoning_content\":\"hidden\\n\""));
    CHECK(response && !strstr(response,"<|im_end|>"));
    CHECK(runtime.history&&strstr(runtime.history,
                                  "</function>\n</tool_call><|im_end|>\n"));
    CHECK(runtime.history&&!strstr(runtime.history,
                                  "</tool_call><|im_end|>\n<|im_end|>"));
    free(first.data);

    fg_error response_error={0};
    json_value *root=parse_response_json(response,&response_error);
    json_value *choices=json_object_get(root,"choices");
    json_value *choice=choices&&choices->type==JSON_ARRAY&&choices->as.array.count?
        choices->as.array.items[0]:NULL;
    json_value *message=json_object_get(choice,"message");
    json_value *content=json_object_get(message,"content");
    json_value *calls=json_object_get(message,"tool_calls");
    json_value *call=calls&&calls->type==JSON_ARRAY&&calls->as.array.count?
        calls->as.array.items[0]:NULL;
    json_value *call_id=json_object_get(call,"id");
    json_value *function=json_object_get(call,"function");
    json_value *name=json_object_get(function,"name");
    json_value *arguments=json_object_get(function,"arguments");
    CHECK(content&&content->type==JSON_NULL);
    CHECK(call_id&&call_id->type==JSON_STRING);
    CHECK(name&&name->type==JSON_STRING);
    CHECK(arguments&&arguments->type==JSON_STRING);

    runtime.generated = "done\n</think>\n\nIt is 20 C.";
    api_buffer second = {0};
    CHECK(buffer_append(&second, "{", &err) == FG_OK);
    CHECK(buffer_append(&second, tools, &err) == FG_OK);
    CHECK(buffer_append(&second,
        "\"messages\":["
        "{\"role\":\"user\",\"content\":\"weather?\"},"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":",&err)==FG_OK);
    if(call_id&&call_id->type==JSON_STRING)
        CHECK(buffer_append_json_string(&second,call_id->as.string,
                                        strlen(call_id->as.string),&err)==FG_OK);
    CHECK(buffer_append(&second,",\"type\":\"function\",\"function\":{\"name\":",&err)==FG_OK);
    if(name&&name->type==JSON_STRING)
        CHECK(buffer_append_json_string(&second,name->as.string,strlen(name->as.string),
                                        &err)==FG_OK);
    CHECK(buffer_append(&second,",\"arguments\":",&err)==FG_OK);
    if(arguments&&arguments->type==JSON_STRING)
        CHECK(buffer_append_json_string(&second,arguments->as.string,
                                        strlen(arguments->as.string),&err)==FG_OK);
    CHECK(buffer_append(&second,"}}]},{\"role\":\"tool\",\"tool_call_id\":",&err)==FG_OK);
    if(call_id&&call_id->type==JSON_STRING)
        CHECK(buffer_append_json_string(&second,call_id->as.string,
                                        strlen(call_id->as.string),&err)==FG_OK);
    CHECK(buffer_append(&second,",\"content\":\"20 C\"}]}",&err)==FG_OK);
    json_free(root);
    free(response);
    response = run_chat_request(&runtime, &session,second.data, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: hit\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: none\r\n"));
    CHECK(response && !strstr(response, "X-Flash-Gordon-Reused-Tokens: 0\r\n"));
    CHECK(runtime.history&&strstr(
        runtime.history,
        "</tool_call><|im_end|>\n<|im_start|>user\n"
        "<tool_response>\n20 C\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"));
    free(response);
    free(second.data);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_image_http_flow(void) {
    char body[4096];
    snprintf(body, sizeof(body),
             "{\"messages\":[{\"role\":\"user\",\"content\":["
             "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,%s\"}},"
             "{\"type\":\"text\",\"text\":\"describe this\"}]}]}",
             test_png_base64);
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    test_vision_available = false;
    test_vision_calls = 0;
    char *response = run_chat_request(&runtime, &session, body, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "400 Bad Request"));
    CHECK(response && strstr(response, "vision tower pack missing"));
    CHECK(test_vision_calls == 0u);
    free(response);
    test_vision_available = true;
    response = run_chat_request(&runtime, &session, body, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "200 OK"));
    CHECK(test_vision_calls == 1u);
    CHECK(test_vision_images == 1u);
    CHECK(test_vision_bytes > 50u);
    CHECK(!session.valid);
    free(response);
    test_vision_available = false;
}

static void test_divergent_tool_request_clears_prefix_metadata(void) {
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    char *response = run_chat_request(
        &runtime, &session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"ordinary\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(session.valid);
    free(response);

    uint32_t prior_resets = runtime.reset_count;
    runtime.generated = "private\n</think>\n\nBeta.";
    runtime.force_continuation_miss = true;
    runtime.require_clean_generation = true;
    response = run_chat_request(
        &runtime, &session,
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"ordinary\"},"
        "{\"role\":\"assistant\",\"content\":\"answer\"},"
        "{\"role\":\"user\",\"content\":\"second\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "\"content\":\"Beta.\""));
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: explicit\r\n"));
    CHECK(runtime.reset_count == prior_resets + 1u);
    CHECK(session.valid);
    free(response);

    prior_resets = runtime.reset_count;
    runtime.generated =
        "private\n</think>\n\n"
        "<tool_call>\n<function=weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "</function>\n</tool_call>";
    runtime.require_clean_generation = true;
    response = run_chat_request(
        &runtime, &session,
        "{"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}}}}}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "HTTP/1.1 200 OK\r\n"));
    CHECK(response && strstr(response, "\"finish_reason\":\"tool_calls\""));
    CHECK(response &&
          strstr(response, "X-Flash-Gordon-Reset-Reason: public-history-mismatch\r\n"));
    CHECK(runtime.reset_count == prior_resets + 1u);
    CHECK(session.valid);
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static const char *tool_update_weather_request =
    "\"tools\":[{\"type\":\"function\",\"function\":{"
    "\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"city\":{\"type\":\"string\"}}}}}],";

static const char *tool_update_weather_clock_request =
    "\"tools\":["
    "{\"type\":\"function\",\"function\":{"
    "\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"city\":{\"type\":\"string\"}}}}},"
    "{\"type\":\"function\",\"function\":{"
    "\"name\":\"clock\",\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"zone\":{\"type\":\"string\"}}}}}],";

static void test_tool_choice_change_continues_prefix(void) {
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    fg_error err = {0};
    api_buffer body = {0};
    CHECK(buffer_append(&body, "{", &err) == FG_OK);
    CHECK(buffer_append(&body, tool_update_weather_request, &err) == FG_OK);
    CHECK(buffer_append(&body,
                        "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}]}",
                        &err) == FG_OK);
    char *response = run_chat_request(&runtime, &session, body.data, &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(session.valid);
    free(body.data);
    free(response);

    size_t prior_evaluated = runtime.evaluated_length;
    uint32_t prior_resets = runtime.reset_count;
    runtime.generated =
        "hidden\n</think>\n\n"
        "<tool_call>\n<function=weather>\n"
        "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    api_buffer second = {0};
    CHECK(buffer_append(&second, "{", &err) == FG_OK);
    CHECK(buffer_append(&second, tool_update_weather_request, &err) == FG_OK);
    CHECK(buffer_append(&second, "\"tool_choice\":\"required\",", &err) == FG_OK);
    CHECK(buffer_append(&second,
                        "\"messages\":["
                        "{\"role\":\"user\",\"content\":\"weather?\"},"
                        "{\"role\":\"assistant\",\"content\":\"answer\"},"
                        "{\"role\":\"user\",\"content\":\"next\"}]}",
                        &err) == FG_OK);
    response = run_chat_request(&runtime, &session, second.data, &status);
    free(second.data);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "200 OK"));
    CHECK(response && strstr(response, "\"finish_reason\":\"tool_calls\""));
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: hit\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: none\r\n"));
    CHECK(runtime.reset_count == prior_resets);
    char reused_header[96];
    snprintf(reused_header, sizeof(reused_header),
             "X-Flash-Gordon-Reused-Tokens: %zu\r\n", prior_evaluated);
    CHECK(response && strstr(response, reused_header));
    CHECK(runtime.history &&
          strstr(runtime.history,
                 "answer<|im_end|>\n"
                 "<|im_start|>system\n"
                 "# Tools\n\nTool configuration updated for this turn."
                 "\n\nTool choice constraint: You must call one or more available functions. "
                 "Do not answer directly.<|im_end|>\n"
                 "<|im_start|>user\nnext<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n"));
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_tool_added_continues_prefix(void) {
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    fg_error err = {0};
    api_buffer body = {0};
    CHECK(buffer_append(&body, "{", &err) == FG_OK);
    CHECK(buffer_append(&body, tool_update_weather_request, &err) == FG_OK);
    CHECK(buffer_append(&body,
                        "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}]}",
                        &err) == FG_OK);
    char *response = run_chat_request(&runtime, &session, body.data, &status);
    CHECK(status == FG_OK);
    CHECK(session.valid);
    free(body.data);
    free(response);

    size_t prior_evaluated = runtime.evaluated_length;
    uint32_t prior_resets = runtime.reset_count;
    api_buffer second = {0};
    CHECK(buffer_append(&second, "{", &err) == FG_OK);
    CHECK(buffer_append(&second, tool_update_weather_clock_request, &err) == FG_OK);
    CHECK(buffer_append(&second,
                        "\"messages\":["
                        "{\"role\":\"user\",\"content\":\"weather?\"},"
                        "{\"role\":\"assistant\",\"content\":\"answer\"},"
                        "{\"role\":\"user\",\"content\":\"next\"}]}",
                        &err) == FG_OK);
    response = run_chat_request(&runtime, &session, second.data, &status);
    free(second.data);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: hit\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: none\r\n"));
    CHECK(runtime.reset_count == prior_resets);
    char reused_header[96];
    snprintf(reused_header, sizeof(reused_header),
             "X-Flash-Gordon-Reused-Tokens: %zu\r\n", prior_evaluated);
    CHECK(response && strstr(response, reused_header));
    CHECK(runtime.history &&
          strstr(runtime.history,
                 "answer<|im_end|>\n"
                 "<|im_start|>system\n"
                 "# Tools\n\nTool configuration updated for this turn."
                 "\n\nThe following functions are now available or have updated "
                 "definitions:\n\n<tools>\n"
                 "{\"name\":\"clock\",\"parameters\":{\"type\":\"object\",\"properties\":{"
                 "\"zone\":{\"type\":\"string\"}}}}\n"
                 "</tools><|im_end|>\n"
                 "<|im_start|>user\nnext<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n"));
    CHECK(runtime.history && !strstr(runtime.history, "no longer available"));
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_tool_removed_continues_prefix(void) {
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    fg_error err = {0};
    api_buffer body = {0};
    CHECK(buffer_append(&body, "{", &err) == FG_OK);
    CHECK(buffer_append(&body, tool_update_weather_clock_request, &err) == FG_OK);
    CHECK(buffer_append(&body,
                        "\"messages\":[{\"role\":\"user\",\"content\":\"time?\"}]}",
                        &err) == FG_OK);
    char *response = run_chat_request(&runtime, &session, body.data, &status);
    CHECK(status == FG_OK);
    CHECK(session.valid);
    free(body.data);
    free(response);

    uint32_t prior_resets = runtime.reset_count;
    api_buffer second = {0};
    CHECK(buffer_append(&second, "{", &err) == FG_OK);
    CHECK(buffer_append(&second, tool_update_weather_request, &err) == FG_OK);
    CHECK(buffer_append(&second,
                        "\"messages\":["
                        "{\"role\":\"user\",\"content\":\"time?\"},"
                        "{\"role\":\"assistant\",\"content\":\"answer\"},"
                        "{\"role\":\"user\",\"content\":\"next\"}]}",
                        &err) == FG_OK);
    response = run_chat_request(&runtime, &session, second.data, &status);
    free(second.data);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: hit\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: none\r\n"));
    CHECK(runtime.reset_count == prior_resets);
    CHECK(runtime.history &&
          strstr(runtime.history,
                 "answer<|im_end|>\n"
                 "<|im_start|>system\n"
                 "# Tools\n\nTool configuration updated for this turn."
                 "\n\nThe following functions are no longer available: clock.<|im_end|>\n"
                 "<|im_start|>user\nnext<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n"));
    CHECK(runtime.history && !strstr(runtime.history, "now available or have updated"));
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_tool_change_with_tampered_history_resets(void) {
    fg_runtime runtime = {.empty_reason = FG_PREFIX_RESET_COLD_START};
    api_public_session session = {0};
    fg_status status = FG_OK;
    fg_error err = {0};
    api_buffer body = {0};
    CHECK(buffer_append(&body, "{", &err) == FG_OK);
    CHECK(buffer_append(&body, tool_update_weather_request, &err) == FG_OK);
    CHECK(buffer_append(&body,
                        "\"messages\":[{\"role\":\"user\",\"content\":\"weather?\"}]}",
                        &err) == FG_OK);
    char *response = run_chat_request(&runtime, &session, body.data, &status);
    CHECK(status == FG_OK);
    CHECK(session.valid);
    free(body.data);
    free(response);

    uint32_t prior_resets = runtime.reset_count;
    response = run_chat_request(
        &runtime, &session,
        "{"
        "\"tools\":["
        "{\"type\":\"function\",\"function\":{"
        "\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"}}}}},"
        "{\"type\":\"function\",\"function\":{"
        "\"name\":\"clock\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"zone\":{\"type\":\"string\"}}}}}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"different\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(response &&
          strstr(response, "X-Flash-Gordon-Reset-Reason: public-history-mismatch\r\n"));
    CHECK(runtime.reset_count == prior_resets + 1u);
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_failed_generation_fails_closed(void) {
    fg_runtime runtime = {
        .empty_reason = FG_PREFIX_RESET_COLD_START,
    };
    api_public_session session={0};
    fg_status status = FG_OK;
    char *response = run_chat_request(
        &runtime, &session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"seed\"}]}",
        &status);
    CHECK(status == FG_OK);
    free(response);

    runtime.fail_after_prefill = true;
    response = run_chat_request(
        &runtime, &session,
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"seed\"},"
        "{\"role\":\"assistant\",\"content\":\"answer\"},"
        "{\"role\":\"user\",\"content\":\"fail\"}]}",
        &status);
    CHECK(status == FG_ERR_MISMATCH);
    CHECK(response && strstr(response, "injected generation failure"));
    CHECK(runtime.history == NULL);
    CHECK(runtime.history_length == 0);
    CHECK(runtime.empty_reason == FG_PREFIX_RESET_FAILURE);
    free(response);

    response = run_chat_request(
        &runtime, &session,
        "{\"messages\":["
        "{\"role\":\"user\",\"content\":\"seed\"},"
        "{\"role\":\"assistant\",\"content\":\"answer\"},"
        "{\"role\":\"user\",\"content\":\"retry\"}]}",
        &status);
    CHECK(status == FG_OK);
    CHECK(response && strstr(response, "X-Flash-Gordon-Prefix-Cache: miss\r\n"));
    CHECK(response && strstr(response, "X-Flash-Gordon-Reset-Reason: failure\r\n"));
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);
}

static void test_nonstream_reasoning_content(void) {
    fg_runtime runtime = {
        .generated = "private steps\n</think>\n\nVisible answer",
    };
    api_public_session session={0};
    fg_status status=FG_OK;
    char *response=run_chat_request(&runtime,&session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"Think then answer.\"}]}",
        &status);
    CHECK(status==FG_OK);
    CHECK(response && strstr(response, "\"reasoning_content\":\"private steps\\n\""));
    CHECK(response && strstr(response, "\"content\":\"Visible answer\""));
    free(response);
    api_public_session_free(&session);
    fg_runtime_close(&runtime);

    fg_runtime quiet = {
        .generated = "Four",
    };
    api_public_session quiet_session={0};
    response=run_chat_request(&quiet,&quiet_session,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"/no_think What is 2+2?\"}]}",
        &status);
    CHECK(status==FG_OK);
    CHECK(response && !strstr(response, "reasoning_content"));
    CHECK(response && strstr(response, "\"content\":\"Four\""));
    free(response);
    api_public_session_free(&quiet_session);
    fg_runtime_close(&quiet);
}

static void test_streamed_reasoning_content(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    api_chat_request request = {0};
    api_generation generation = {
        .fd = sockets[0],
        .stream = true,
        .id = "chatcmpl-reason-stream",
        .model = "Qwen3.8-Flash-Next",
        .created = 7,
        .request = &request,
    };
    fg_error err = {0};
    CHECK(send_stream_start(&generation, &err) == FG_OK);
    CHECK(api_token(&generation, 1, "step one", 8, &err) == FG_OK);
    CHECK(api_token(&generation, 2, " and two", 8, &err) == FG_OK);
    CHECK(api_token(&generation, 3, "</think>\n\nVisible",
                    sizeof("</think>\n\nVisible") - 1u, &err) == FG_OK);
    CHECK(api_token(&generation, 4, " answer", 7, &err) == FG_OK);
    fg_chat_generated generated = {0};
    CHECK(fg_chat_parse_generated(generation.content.data, true, &generated, &err) == FG_OK);
    CHECK(send_stream_end(&generation, &generated, "stop", &err) == FG_OK);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\"reasoning_content\":\"step one\""));
    CHECK(response && strstr(response, "\"reasoning_content\":\" and two\""));
    CHECK(response && strstr(response, "\"content\":\"Visible\""));
    CHECK(response && strstr(response, "\"content\":\" answer\""));
    const char *reasoning = response ? strstr(response, "\"reasoning_content\"") : NULL;
    const char *content = response ? strstr(response, "\"content\"") : NULL;
    CHECK(reasoning && content && reasoning < content);
    CHECK(response && !strstr(response, "</think>"));
    free(response);
    fg_chat_generated_free(&generated);
    free(generation.content.data);
    free(generation.visible_pending.data);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_streamed_no_think_has_no_reasoning(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    api_chat_request request = {0};
    api_generation generation = {
        .fd = sockets[0],
        .stream = true,
        .id = "chatcmpl-no-think",
        .model = "Qwen3.8-Flash-Next",
        .created = 8,
        .request = &request,
        .think_closed = true,
    };
    fg_error err = {0};
    CHECK(send_stream_start(&generation, &err) == FG_OK);
    CHECK(api_token(&generation, 1, "Four", 4, &err) == FG_OK);
    fg_chat_generated generated = {0};
    CHECK(fg_chat_parse_generated(generation.content.data, false, &generated, &err) == FG_OK);
    CHECK(send_stream_end(&generation, &generated, "stop", &err) == FG_OK);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\"content\":\"Four\""));
    CHECK(response && !strstr(response, "reasoning_content"));
    free(response);
    fg_chat_generated_free(&generated);
    free(generation.content.data);
    free(generation.visible_pending.data);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_streamed_unclosed_reasoning_flushed(void) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    api_chat_request request = {0};
    api_generation generation = {
        .fd = sockets[0],
        .stream = true,
        .id = "chatcmpl-unclosed",
        .model = "Qwen3.8-Flash-Next",
        .created = 9,
        .request = &request,
    };
    fg_error err = {0};
    CHECK(send_stream_start(&generation, &err) == FG_OK);
    CHECK(api_token(&generation, 1, "half a", 6, &err) == FG_OK);
    CHECK(api_token(&generation, 2, " thought", 8, &err) == FG_OK);
    fg_chat_generated generated = {0};
    CHECK(fg_chat_parse_generated(generation.content.data, true, &generated, &err) == FG_OK);
    CHECK(send_stream_end(&generation, &generated, "length", &err) == FG_OK);
    shutdown(sockets[0], SHUT_WR);
    char *response = read_socket_response(sockets[1]);
    CHECK(response && strstr(response, "\"reasoning_content\":\" thought\""));
    CHECK(response && !strstr(response, "\"content\":\"half a"));
    free(response);
    fg_chat_generated_free(&generated);
    free(generation.content.data);
    free(generation.visible_pending.data);
    close(sockets[0]);
    close(sockets[1]);
}

static void test_public_session_mismatch_reasons(void) {
    fg_chat_message stored_messages[] = {
        {.role = "user", .content = "hello"},
        {.role = "assistant", .content = "answer"},
    };
    const char *stored_schemas[] = {"{\"name\":\"weather\"}"};
    api_public_session session = {.valid = true};
    session.transcript.message_count = 2;
    session.transcript.messages = stored_messages;
    session.transcript.tool_schema_count = 1;
    session.transcript.tool_schemas = (char **)stored_schemas;
    session.transcript.tool_choice = FG_CHAT_TOOL_AUTO;

    fg_chat_message request_messages[] = {
        {.role = "user", .content = "hello"},
        {.role = "assistant", .content = "answer"},
    };
    api_chat_request request = {0};
    request.message_count = 2;
    request.messages = request_messages;
    request.tool_schema_count = 1;
    request.tool_schemas = (char **)stored_schemas;
    request.tool_choice = FG_CHAT_TOOL_AUTO;

    char reason[192] = {0};
    CHECK(api_public_session_prefix(&session, &request, reason, sizeof(reason)));
    CHECK(!reason[0]);

    request.tool_choice = FG_CHAT_TOOL_REQUIRED;
    CHECK(api_public_session_prefix(&session, &request, reason, sizeof(reason)));
    CHECK(!reason[0]);

    request.tool_choice = FG_CHAT_TOOL_AUTO;
    const char *extra_schemas[] = {"{\"name\":\"weather\"}", "{\"name\":\"forecast\"}"};
    request.tool_schema_count = 2;
    request.tool_schemas = (char **)extra_schemas;
    CHECK(api_public_session_prefix(&session, &request, reason, sizeof(reason)));
    CHECK(!reason[0]);

    const char *changed_schemas[] = {"{\"name\":\"weather\",\"extra\":true}"};
    request.tool_schema_count = 1;
    request.tool_schemas = (char **)changed_schemas;
    CHECK(api_public_session_prefix(&session, &request, reason, sizeof(reason)));
    CHECK(!reason[0]);

    request.tool_schemas = (char **)stored_schemas;
    request_messages[1].content = "different";
    CHECK(!api_public_session_prefix(&session, &request, reason, sizeof(reason)));
    CHECK(!strcmp(reason, "message[1].content"));

    request_messages[1].content = "answer";
    request.message_count = 1;
    CHECK(!api_public_session_prefix(&session, &request, reason, sizeof(reason)));
    CHECK(!strcmp(reason, "messages=2->1"));
}

int main(void) {
    test_openai_tools_request();
    test_openai_structured_text_content();
    test_image_content_parts();
    test_video_content_parts();
    test_video_frames_content_parts();
    test_video_frames_too_many();
    test_unknown_tool_result_rejected();
    test_tool_choice_modes();
    test_history_reasoning_and_empty_calls();
    test_greedy_controls();
    test_nonstream_tool_response();
    test_nonstream_reasoning_content();
    test_streamed_tool_response();
    test_streamed_reasoning_content();
    test_streamed_no_think_has_no_reasoning();
    test_streamed_unclosed_reasoning_flushed();
    test_streamed_incomplete_tags_do_not_leak();
    test_streamed_utf8_and_sentinel_filtering();
    test_json_nul_and_member_limit();
    test_client_socket_timeouts();
    test_model_capabilities();
    test_live_prefix_hit_divergence_and_reset();
    test_live_prefix_tool_loop();
    test_divergent_tool_request_clears_prefix_metadata();
    test_public_session_mismatch_reasons();
    test_tool_choice_change_continues_prefix();
    test_tool_added_continues_prefix();
    test_tool_removed_continues_prefix();
    test_tool_change_with_tampered_history_resets();
    test_image_http_flow();
    test_video_http_flow();
    test_video_frames_http_flow();
    test_failed_generation_fails_closed();
    if (failures) fprintf(stderr, "%d API test(s) failed\n", failures);
    return failures ? 1 : 0;
}
