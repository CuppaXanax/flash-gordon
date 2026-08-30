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
    (void)runtime;
}

fg_status fg_runtime_reset(fg_runtime *runtime, fg_error *err) {
    (void)runtime;
    (void)err;
    return FG_OK;
}

fg_status fg_runtime_generate(fg_runtime *runtime, const char *rendered_suffix,
                              uint32_t max_tokens, fg_token_callback callback,
                              void *callback_context, fg_interrupt_fn interrupted,
                              void *interrupt_context, fg_generation_stats *stats,
                              fg_error *err) {
    (void)runtime;
    (void)rendered_suffix;
    (void)max_tokens;
    (void)callback;
    (void)callback_context;
    (void)interrupted;
    (void)interrupt_context;
    (void)stats;
    (void)err;
    return FG_ERR_UNAVAILABLE;
}

uint32_t fg_runtime_context_tokens(const fg_runtime *runtime) {
    (void)runtime;
    return 0;
}

uint32_t fg_runtime_context_limit(const fg_runtime *runtime) {
    (void)runtime;
    return 0;
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

    const char *rejected =
        "{\"temperature\":0.5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    memset(&err, 0, sizeof(err));
    root = parse_json_body(rejected, strlen(rejected), &err);
    CHECK(root != NULL);
    CHECK(parse_chat_request(root, "Qwen3.8-Flash-Next", &request, &err) ==
          FG_ERR_ARGUMENT);
    CHECK(strstr(err.message, "greedy decoding only") != NULL);
    api_chat_request_free(&request);
    json_free(root);
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
        .generated_tokens = 5,
        .context_tokens = 15,
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
    CHECK(response && strstr(response, "\"tool_calls\":[{\"index\":0"));
    CHECK(response && strstr(response, "\"id\":\"call_chatcmpl-stream_0\""));
    CHECK(response && strstr(response, "\"tool_calls\":[{\"index\":1"));
    CHECK(response && strstr(response, "\"id\":\"call_chatcmpl-stream_1\""));
    CHECK(response && strstr(response, "\"name\":\"clock\""));
    CHECK(response && strstr(response, "\"finish_reason\":\"tool_calls\""));
    CHECK(response && strstr(response, "data: [DONE]"));
    CHECK(response && !strstr(response, "hidden"));
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

int main(void) {
    test_openai_tools_request();
    test_unknown_tool_result_rejected();
    test_tool_choice_modes();
    test_history_reasoning_and_empty_calls();
    test_greedy_controls();
    test_nonstream_tool_response();
    test_streamed_tool_response();
    test_streamed_incomplete_tags_do_not_leak();
    test_streamed_utf8_and_sentinel_filtering();
    test_json_nul_and_member_limit();
    test_client_socket_timeouts();
    if (failures) fprintf(stderr, "%d API test(s) failed\n", failures);
    return failures ? 1 : 0;
}
