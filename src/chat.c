#include "fg_chat.h"
#ifndef FG_CHAT_RENDER_ONLY
#include "fg_runtime.h"
#endif

#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct fg_text_buffer {
    char *data;
    size_t length;
    size_t capacity;
} fg_text_buffer;

#ifndef FG_CHAT_RENDER_ONLY
static volatile sig_atomic_t fg_chat_stop_requested;
#endif

static fg_status buffer_reserve(fg_text_buffer *buffer, size_t extra, fg_error *err) {
    if (extra > SIZE_MAX - buffer->length - 1u) {
        fg_error_set(err, FG_ERR_LIMIT, "chat text exceeds addressable memory");
        return FG_ERR_LIMIT;
    }
    size_t required = buffer->length + extra + 1u;
    if (required <= buffer->capacity) return FG_OK;
    size_t capacity = buffer->capacity ? buffer->capacity : 256u;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    char *data = realloc(buffer->data, capacity);
    if (!data) {
        fg_error_set(err, FG_ERR_OOM, "allocate chat text");
        return FG_ERR_OOM;
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return FG_OK;
}

static fg_status buffer_append_n(fg_text_buffer *buffer, const char *text, size_t length,
                                 fg_error *err) {
    fg_status status = buffer_reserve(buffer, length, err);
    if (status != FG_OK) return status;
    if (length) memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = 0;
    return FG_OK;
}

static fg_status buffer_append(fg_text_buffer *buffer, const char *text, fg_error *err) {
    return buffer_append_n(buffer, text ? text : "", text ? strlen(text) : 0u, err);
}

static void json_skip_space(const char **cursor) {
    while (isspace((unsigned char)**cursor)) (*cursor)++;
}

static bool json_scan_string(const char **cursor) {
    if (**cursor != '"') return false;
    (*cursor)++;
    while (**cursor && **cursor != '"') {
        if ((unsigned char)**cursor < 0x20u) return false;
        if (**cursor == '\\') {
            (*cursor)++;
            if (!**cursor) return false;
            if (**cursor == 'u') {
                (*cursor)++;
                for (unsigned i = 0; i < 4u; i++) {
                    if (!isxdigit((unsigned char)(*cursor)[i])) return false;
                }
                *cursor += 4;
                continue;
            }
            if (!strchr("\"\\/bfnrt", **cursor)) return false;
        }
        (*cursor)++;
    }
    if (**cursor != '"') return false;
    (*cursor)++;
    return true;
}

static bool json_scan_value(const char **cursor) {
    json_skip_space(cursor);
    if (**cursor == '"') return json_scan_string(cursor);
    if (**cursor == '{' || **cursor == '[') {
        char open = **cursor;
        char close = open == '{' ? '}' : ']';
        (*cursor)++;
        json_skip_space(cursor);
        if (**cursor == close) {
            (*cursor)++;
            return true;
        }
        for (;;) {
            if (open == '{') {
                if (!json_scan_string(cursor)) return false;
                json_skip_space(cursor);
                if (**cursor != ':') return false;
                (*cursor)++;
            }
            if (!json_scan_value(cursor)) return false;
            json_skip_space(cursor);
            if (**cursor == close) {
                (*cursor)++;
                return true;
            }
            if (**cursor != ',') return false;
            (*cursor)++;
            json_skip_space(cursor);
        }
    }
    const char *start = *cursor;
    while (**cursor && !isspace((unsigned char)**cursor) && !strchr(",]}", **cursor)) {
        (*cursor)++;
    }
    return *cursor != start;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static fg_status append_utf8(fg_text_buffer *buffer, uint32_t value, fg_error *err) {
    char bytes[4];
    size_t count;
    if (value <= 0x7fu) {
        bytes[0] = (char)value;
        count = 1u;
    } else if (value <= 0x7ffu) {
        bytes[0] = (char)(0xc0u | (value >> 6u));
        bytes[1] = (char)(0x80u | (value & 0x3fu));
        count = 2u;
    } else if (value <= 0xffffu) {
        bytes[0] = (char)(0xe0u | (value >> 12u));
        bytes[1] = (char)(0x80u | ((value >> 6u) & 0x3fu));
        bytes[2] = (char)(0x80u | (value & 0x3fu));
        count = 3u;
    } else {
        bytes[0] = (char)(0xf0u | (value >> 18u));
        bytes[1] = (char)(0x80u | ((value >> 12u) & 0x3fu));
        bytes[2] = (char)(0x80u | ((value >> 6u) & 0x3fu));
        bytes[3] = (char)(0x80u | (value & 0x3fu));
        count = 4u;
    }
    return buffer_append_n(buffer, bytes, count, err);
}

static fg_status append_decoded_json_string(fg_text_buffer *buffer, const char *start,
                                            const char *end, fg_error *err) {
    for (const char *p = start; p < end;) {
        unsigned char c = (unsigned char)*p++;
        if (c != '\\') {
            fg_status status = buffer_append_n(buffer, (const char *)&c, 1u, err);
            if (status != FG_OK) return status;
            continue;
        }
        if (p >= end) {
            fg_error_set(err, FG_ERR_FORMAT, "invalid tool argument string escape");
            return FG_ERR_FORMAT;
        }
        char escaped = *p++;
        char decoded;
        switch (escaped) {
            case '"': decoded = '"'; break;
            case '\\': decoded = '\\'; break;
            case '/': decoded = '/'; break;
            case 'b': decoded = '\b'; break;
            case 'f': decoded = '\f'; break;
            case 'n': decoded = '\n'; break;
            case 'r': decoded = '\r'; break;
            case 't': decoded = '\t'; break;
            case 'u': {
                if (end - p < 4) {
                    fg_error_set(err, FG_ERR_FORMAT, "short Unicode tool argument escape");
                    return FG_ERR_FORMAT;
                }
                uint32_t value = 0;
                for (unsigned i = 0; i < 4u; i++) {
                    int digit = hex_digit(p[i]);
                    if (digit < 0) {
                        fg_error_set(err, FG_ERR_FORMAT, "invalid Unicode tool argument escape");
                        return FG_ERR_FORMAT;
                    }
                    value = value * 16u + (uint32_t)digit;
                }
                p += 4;
                if (value >= 0xd800u && value <= 0xdbffu) {
                    if (end - p < 6 || p[0] != '\\' || p[1] != 'u') {
                        fg_error_set(err, FG_ERR_FORMAT,
                                     "unpaired Unicode tool argument surrogate");
                        return FG_ERR_FORMAT;
                    }
                    p += 2;
                    uint32_t low = 0;
                    for (unsigned i = 0; i < 4u; i++) {
                        int digit = hex_digit(p[i]);
                        if (digit < 0) {
                            fg_error_set(err, FG_ERR_FORMAT,
                                         "invalid Unicode tool argument surrogate");
                            return FG_ERR_FORMAT;
                        }
                        low = low * 16u + (uint32_t)digit;
                    }
                    p += 4;
                    if (low < 0xdc00u || low > 0xdfffu) {
                        fg_error_set(err, FG_ERR_FORMAT,
                                     "unpaired Unicode tool argument surrogate");
                        return FG_ERR_FORMAT;
                    }
                    value =
                        0x10000u + ((value - 0xd800u) << 10u) + (low - 0xdc00u);
                } else if (value >= 0xdc00u && value <= 0xdfffu) {
                    fg_error_set(err, FG_ERR_FORMAT,
                                 "unpaired Unicode tool argument surrogate");
                    return FG_ERR_FORMAT;
                }
                fg_status status = append_utf8(buffer, value, err);
                if (status != FG_OK) return status;
                continue;
            }
            default:
                fg_error_set(err, FG_ERR_FORMAT, "invalid tool argument string escape");
                return FG_ERR_FORMAT;
        }
        fg_status status = buffer_append_n(buffer, &decoded, 1u, err);
        if (status != FG_OK) return status;
    }
    return FG_OK;
}

static fg_status append_minified_json(fg_text_buffer *buffer, const char *start,
                                      const char *end, fg_error *err) {
    bool string = false;
    bool escaped = false;
    for (const char *p = start; p < end; p++) {
        char c = *p;
        if (!string && isspace((unsigned char)c)) continue;
        fg_status status = buffer_append_n(buffer, &c, 1u, err);
        if (status != FG_OK) return status;
        if (string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') string = false;
        } else if (c == '"') {
            string = true;
        }
    }
    return FG_OK;
}

typedef struct tool_argument {
    char *key;
    const char *value_start;
    const char *value_end;
    bool string;
    bool used;
} tool_argument;

static void tool_arguments_free(tool_argument *arguments, size_t count) {
    for (size_t i = 0; i < count; i++) free(arguments[i].key);
    free(arguments);
}

static bool text_has_chatml_control(const char *text) {
    return text && (strstr(text, "<|im_start|>") || strstr(text, "<|im_end|>"));
}

static bool valid_qwen_tag_value(const char *text, bool allow_empty) {
    if (!text) return false;
    if (!allow_empty && !text[0]) return false;
    return !strpbrk(text, "<>\r\n");
}

static fg_status append_tool_argument(fg_text_buffer *buffer, tool_argument *argument,
                                      fg_error *err) {
    if (!valid_qwen_tag_value(argument->key, true)) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "tool argument names cannot contain Qwen tag delimiters");
        return FG_ERR_ARGUMENT;
    }
    fg_status status = buffer_append(buffer, "<parameter=", err);
    if (status == FG_OK) status = buffer_append(buffer, argument->key, err);
    if (status == FG_OK) status = buffer_append(buffer, ">\n", err);
    if (status == FG_OK && argument->string) {
        fg_text_buffer decoded = {0};
        status = append_decoded_json_string(&decoded, argument->value_start + 1,
                                            argument->value_end - 1, err);
        if (status == FG_OK && text_has_chatml_control(decoded.data)) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "tool argument strings cannot contain reserved ChatML controls");
            status = FG_ERR_ARGUMENT;
        }
        if (status == FG_OK) status = buffer_append(buffer, decoded.data, err);
        free(decoded.data);
    } else if (status == FG_OK) {
        status = append_minified_json(buffer, argument->value_start, argument->value_end, err);
    }
    if (status == FG_OK) status = buffer_append(buffer, "\n</parameter>\n", err);
    if (status == FG_OK) argument->used = true;
    return status;
}

static bool json_object_member(const char *json, const char *name, const char **value_start,
                               const char **value_end) {
    const char *cursor = json;
    json_skip_space(&cursor);
    if (*cursor++ != '{') return false;
    json_skip_space(&cursor);
    while (*cursor && *cursor != '}') {
        const char *key_start = cursor;
        if (!json_scan_string(&cursor)) return false;
        const char *key_end = cursor;
        json_skip_space(&cursor);
        if (*cursor++ != ':') return false;
        json_skip_space(&cursor);
        const char *start = cursor;
        if (!json_scan_value(&cursor)) return false;
        const char *end = cursor;
        size_t key_length = (size_t)(key_end - key_start - 2);
        if (strlen(name) == key_length && !memcmp(key_start + 1, name, key_length)) {
            *value_start = start;
            *value_end = end;
            return true;
        }
        json_skip_space(&cursor);
        if (*cursor == ',') {
            cursor++;
            json_skip_space(&cursor);
        } else if (*cursor != '}') {
            return false;
        }
    }
    return false;
}

static const char *schema_properties_for_call(const fg_chat_render_options *options,
                                              const char *call_name) {
    if (!options || !call_name) return NULL;
    for (size_t i = 0; i < options->tool_schema_count; i++) {
        const char *schema = options->tool_schemas[i];
        const char *name_start = NULL, *name_end = NULL;
        if (!schema || !json_object_member(schema, "name", &name_start, &name_end) ||
            *name_start != '"')
            continue;
        fg_text_buffer decoded = {0};
        fg_error ignored = {0};
        if (append_decoded_json_string(&decoded, name_start + 1, name_end - 1, &ignored) !=
            FG_OK) {
            free(decoded.data);
            continue;
        }
        bool match = decoded.data && !strcmp(decoded.data, call_name);
        free(decoded.data);
        if (!match) continue;
        const char *parameters = NULL, *parameters_end = NULL;
        if (!json_object_member(schema, "parameters", &parameters, &parameters_end) &&
            !json_object_member(schema, "input_schema", &parameters, &parameters_end))
            return NULL;
        (void)parameters_end;
        const char *properties = NULL, *properties_end = NULL;
        if (!json_object_member(parameters, "properties", &properties, &properties_end))
            return NULL;
        (void)properties_end;
        return properties;
    }
    return NULL;
}

static fg_status append_schema_ordered_arguments(fg_text_buffer *buffer,
                                                 tool_argument *arguments, size_t count,
                                                 const char *properties, fg_error *err) {
    if (!properties) return FG_OK;
    const char *cursor = properties;
    json_skip_space(&cursor);
    if (*cursor++ != '{') return FG_OK;
    json_skip_space(&cursor);
    while (*cursor && *cursor != '}') {
        const char *key_start = cursor;
        if (!json_scan_string(&cursor)) return FG_OK;
        const char *key_end = cursor;
        fg_text_buffer decoded = {0};
        fg_status status =
            append_decoded_json_string(&decoded, key_start + 1, key_end - 1, err);
        if (status != FG_OK) {
            free(decoded.data);
            return status;
        }
        json_skip_space(&cursor);
        if (*cursor++ != ':') {
            free(decoded.data);
            return FG_OK;
        }
        json_skip_space(&cursor);
        if (!json_scan_value(&cursor)) {
            free(decoded.data);
            return FG_OK;
        }
        for (size_t i = 0; i < count; i++) {
            if (!arguments[i].used && arguments[i].key && decoded.data &&
                !strcmp(arguments[i].key, decoded.data)) {
                status = append_tool_argument(buffer, &arguments[i], err);
                break;
            }
        }
        free(decoded.data);
        if (status != FG_OK) return status;
        json_skip_space(&cursor);
        if (*cursor == ',') {
            cursor++;
            json_skip_space(&cursor);
        } else if (*cursor != '}') {
            return FG_OK;
        }
    }
    return FG_OK;
}

static bool raw_json_complete(const char *text);

static fg_status append_tool_arguments(fg_text_buffer *buffer, const char *json,
                                       const char *call_name,
                                       const fg_chat_render_options *options, fg_error *err) {
    const char *cursor = json ? json : "{}";
    tool_argument *arguments = NULL;
    size_t count = 0;
    json_skip_space(&cursor);
    if (*cursor != '{' || !raw_json_complete(cursor)) {
        fg_error_set(err, FG_ERR_FORMAT, "tool arguments must be a JSON object");
        return FG_ERR_FORMAT;
    }
    cursor++;
    json_skip_space(&cursor);
    while (*cursor && *cursor != '}') {
        const char *key_start = cursor;
        if (!json_scan_string(&cursor)) {
            fg_error_set(err, FG_ERR_FORMAT, "invalid tool argument name");
            tool_arguments_free(arguments, count);
            return FG_ERR_FORMAT;
        }
        const char *key_end = cursor;
        json_skip_space(&cursor);
        if (*cursor++ != ':') {
            fg_error_set(err, FG_ERR_FORMAT, "tool argument is missing ':'");
            tool_arguments_free(arguments, count);
            return FG_ERR_FORMAT;
        }
        json_skip_space(&cursor);
        const char *value_start = cursor;
        if (!json_scan_value(&cursor)) {
            fg_error_set(err, FG_ERR_FORMAT, "invalid tool argument value");
            tool_arguments_free(arguments, count);
            return FG_ERR_FORMAT;
        }
        const char *value_end = cursor;
        tool_argument *grown = realloc(arguments, (count + 1u) * sizeof(*grown));
        if (!grown) {
            tool_arguments_free(arguments, count);
            fg_error_set(err, FG_ERR_OOM, "allocate tool arguments");
            return FG_ERR_OOM;
        }
        arguments = grown;
        memset(&arguments[count], 0, sizeof(arguments[count]));
        fg_text_buffer key = {0};
        fg_status status =
            append_decoded_json_string(&key, key_start + 1, key_end - 1, err);
        if (status != FG_OK) {
            free(key.data);
            tool_arguments_free(arguments, count);
            return status;
        }
        if (!key.data) {
            key.data = strdup("");
            if (!key.data) {
                tool_arguments_free(arguments, count);
                fg_error_set(err, FG_ERR_OOM, "copy empty tool argument name");
                return FG_ERR_OOM;
            }
        }
        arguments[count++] = (tool_argument){
            .key = key.data,
            .value_start = value_start,
            .value_end = value_end,
            .string = *value_start == '"',
        };
        json_skip_space(&cursor);
        if (*cursor == ',') {
            cursor++;
            json_skip_space(&cursor);
        } else if (*cursor != '}') {
            fg_error_set(err, FG_ERR_FORMAT, "invalid tool arguments object separator");
            tool_arguments_free(arguments, count);
            return FG_ERR_FORMAT;
        }
    }
    if (*cursor != '}') {
        fg_error_set(err, FG_ERR_FORMAT, "unterminated tool arguments object");
        tool_arguments_free(arguments, count);
        return FG_ERR_FORMAT;
    }
    cursor++;
    json_skip_space(&cursor);
    if (*cursor) {
        fg_error_set(err, FG_ERR_FORMAT, "trailing text after tool arguments");
        tool_arguments_free(arguments, count);
        return FG_ERR_FORMAT;
    }
    const char *properties = schema_properties_for_call(options, call_name);
    fg_status status =
        append_schema_ordered_arguments(buffer, arguments, count, properties, err);
    for (size_t i = 0; status == FG_OK && i < count; i++)
        if (!arguments[i].used) status = append_tool_argument(buffer, &arguments[i], err);
    tool_arguments_free(arguments, count);
    return status;
}

static bool role_is_system(const char *role) {
    return role && (!strcmp(role, "system") || !strcmp(role, "developer"));
}

static bool has_trimmed_content(const char *text) {
    if (!text) return false;
    while (*text) {
        if (!isspace((unsigned char)*text)) return true;
        text++;
    }
    return false;
}

static fg_status append_tools_prompt(fg_text_buffer *buffer,
                                     const fg_chat_render_options *options,
                                     fg_error *err) {
    fg_status status = buffer_append(
        buffer, "# Tools\n\nYou have access to the following functions:\n\n<tools>\n", err);
    for (size_t i = 0; status == FG_OK && i < options->tool_schema_count; i++) {
        if (i) status = buffer_append(buffer, "\n", err);
        if (status == FG_OK) status = buffer_append(buffer, options->tool_schemas[i], err);
    }
    if (status != FG_OK) return status;
    status = buffer_append(
        buffer,
        "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- You can use the <think></think> block to plan your next tool call OR to synthesize "
        "data and formulate your final response to the user.\n"
        "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> "
        "block.\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> "
        "block must be nested within <tool_call></tool_call> XML tags.\n"
        "- If you choose to call a tool, output the <tool_call> block immediately after "
        "thinking, with no conversational text before it.\n"
        "- The <tool_call> and <function> tags must begin a new line with no indentation.\n"
        "- To call multiple functions, output a separate, completely closed "
        "<tool_call></tool_call> block for each function.\n"
        "- If you have all necessary data, provide your final answer directly to the user "
        "without any tool call.\n"
        "</IMPORTANT>",
        err);
    if (status != FG_OK) return status;
    if (options->tool_choice == FG_CHAT_TOOL_NONE)
        return buffer_append(
            buffer,
            "\n\nTool choice constraint: Do not call any function. Answer the user directly.",
            err);
    if (options->tool_choice == FG_CHAT_TOOL_REQUIRED)
        return buffer_append(
            buffer,
            "\n\nTool choice constraint: You must call one or more available functions. "
            "Do not answer directly.",
            err);
    if (options->tool_choice == FG_CHAT_TOOL_NAMED) {
        status = buffer_append(
            buffer, "\n\nTool choice constraint: You must call only the function \"", err);
        if (status == FG_OK) status = buffer_append(buffer, options->tool_choice_name, err);
        if (status == FG_OK)
            status = buffer_append(buffer, "\". Do not call any other function.", err);
    }
    return status;
}

static fg_status append_message(fg_text_buffer *buffer, const fg_chat_message *message,
                                const fg_chat_render_options *options, fg_error *err) {
    const char *role = message->role ? message->role : "";
    const char *content = message->content ? message->content : "";
    fg_status status;
    if (!strcmp(role, "user") || role_is_system(role)) {
        status = buffer_append(buffer, "<|im_start|>", err);
        if (status == FG_OK) status = buffer_append(buffer, role_is_system(role) ? "system\n" :
                                                                             "user\n",
                                                    err);
        if (status == FG_OK) status = buffer_append(buffer, content, err);
        if (status == FG_OK) status = buffer_append(buffer, "<|im_end|>\n", err);
        return status;
    }
    if (!strcmp(role, "assistant")) {
        status = buffer_append(buffer, "<|im_start|>assistant\n", err);
        if (status == FG_OK && message->reasoning && message->reasoning[0]) {
            status = buffer_append(buffer, "<think>\n", err);
            if (status == FG_OK) status = buffer_append(buffer, message->reasoning, err);
            if (status == FG_OK) status = buffer_append(buffer, "\n</think>\n\n", err);
        }
        if (status == FG_OK) status = buffer_append(buffer, content, err);
        for (size_t i = 0; status == FG_OK && i < message->tool_call_count; i++) {
            const fg_chat_tool_call *call = &message->tool_calls[i];
            if (i || has_trimmed_content(content))
                status = buffer_append(buffer, "\n\n", err);
            if (status == FG_OK) status = buffer_append(buffer, "<tool_call>\n<function=", err);
            if (status == FG_OK) status = buffer_append(buffer, call->name, err);
            if (status == FG_OK) status = buffer_append(buffer, ">\n", err);
            if (status == FG_OK)
                status =
                    append_tool_arguments(buffer, call->arguments_json, call->name, options, err);
            if (status == FG_OK)
                status = buffer_append(buffer, "</function>\n</tool_call>", err);
        }
        if (status == FG_OK) status = buffer_append(buffer, "<|im_end|>\n", err);
        return status;
    }
    status = buffer_append(buffer, "<|im_start|>user\n[", err);
    if (status == FG_OK) status = buffer_append(buffer, role, err);
    if (status == FG_OK) status = buffer_append(buffer, "]: ", err);
    if (status == FG_OK) status = buffer_append(buffer, content, err);
    if (status == FG_OK) status = buffer_append(buffer, "<|im_end|>\n", err);
    return status;
}

static fg_status chat_render(const fg_chat_message *messages, size_t message_count,
                             const fg_chat_render_options *options, bool continuation,
                             char **rendered, fg_error *err) {
    if (!rendered || (message_count && !messages) ||
        (options && options->tool_schema_count && !options->tool_schemas) ||
        (options && options->tool_choice == FG_CHAT_TOOL_NAMED &&
         (!options->tool_choice_name || !options->tool_choice_name[0])) ||
        (options && (options->tool_choice == FG_CHAT_TOOL_REQUIRED ||
                    options->tool_choice == FG_CHAT_TOOL_NAMED) &&
         !options->tool_schema_count)) {
        fg_error_set(err, FG_ERR_ARGUMENT, "chat renderer received invalid arguments");
        return FG_ERR_ARGUMENT;
    }
    for (size_t i = 0; i < message_count; i++) {
        const fg_chat_message *message = &messages[i];
        if ((message->tool_call_count && !message->tool_calls) ||
            text_has_chatml_control(message->content) ||
            text_has_chatml_control(message->reasoning)) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "message text cannot contain reserved ChatML control tokens");
            return FG_ERR_ARGUMENT;
        }
        for (size_t j = 0; j < message->tool_call_count; j++) {
            const fg_chat_tool_call *call = &message->tool_calls[j];
            if (!valid_qwen_tag_value(call->name, false) ||
                text_has_chatml_control(call->arguments_json)) {
                fg_error_set(err, FG_ERR_ARGUMENT,
                             "tool calls contain invalid Qwen or ChatML control text");
                return FG_ERR_ARGUMENT;
            }
        }
    }
    if (options) {
        if (options->tool_choice_name &&
            (strstr(options->tool_choice_name, "<|im_start|>") ||
             strstr(options->tool_choice_name, "<|im_end|>"))) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "tool choice name cannot contain reserved ChatML control tokens");
            return FG_ERR_ARGUMENT;
        }
        for (size_t i = 0; i < options->tool_schema_count; i++) {
            const char *schema = options->tool_schemas[i];
            if (schema && (strstr(schema, "<|im_start|>") || strstr(schema, "<|im_end|>"))) {
                fg_error_set(err, FG_ERR_ARGUMENT,
                             "tool schemas cannot contain reserved ChatML control tokens");
                return FG_ERR_ARGUMENT;
            }
        }
    }
    *rendered = NULL;
    fg_text_buffer buffer = {0};
    bool tools = options && options->tool_schema_count;
    size_t first = 0;
    fg_status status = FG_OK;
    if (tools && !continuation) {
        status = buffer_append(&buffer, "<|im_start|>system\n", err);
        if (status == FG_OK) status = append_tools_prompt(&buffer, options, err);
        if (message_count && role_is_system(messages[0].role)) {
            if (status == FG_OK && has_trimmed_content(messages[0].content)) {
                status = buffer_append(&buffer, "\n\n", err);
                if (status == FG_OK) status = buffer_append(&buffer, messages[0].content, err);
            }
            first = 1u;
        }
        if (status == FG_OK) status = buffer_append(&buffer, "<|im_end|>\n", err);
    }
    if (!continuation && !tools && message_count && role_is_system(messages[0].role) &&
        !has_trimmed_content(messages[0].content))
        first = 1u;
    for (size_t i = first; status == FG_OK && i < message_count;) {
        if (!strcmp(messages[i].role ? messages[i].role : "", "tool") ||
            !strcmp(messages[i].role ? messages[i].role : "", "function")) {
            status = buffer_append(&buffer, "<|im_start|>user", err);
            do {
                if (status == FG_OK)
                    status = buffer_append(&buffer, "\n<tool_response>\n", err);
                if (status == FG_OK)
                    status = buffer_append(&buffer,
                                           messages[i].content ? messages[i].content : "", err);
                if (status == FG_OK)
                    status = buffer_append(&buffer, "\n</tool_response>", err);
                i++;
            } while (status == FG_OK && i < message_count &&
                     (!strcmp(messages[i].role ? messages[i].role : "", "tool") ||
                      !strcmp(messages[i].role ? messages[i].role : "", "function")));
            if (status == FG_OK) status = buffer_append(&buffer, "<|im_end|>\n", err);
            continue;
        }
        status = append_message(&buffer, &messages[i], options, err);
        i++;
    }
    if (status == FG_OK) status = buffer_append(&buffer, "<|im_start|>assistant\n", err);
    if (status == FG_OK) {
        if (options && options->think_mode == FG_CHAT_THINK_OFF)
            status = buffer_append(&buffer, "<think>\n\n</think>\n\n", err);
        else
            status = buffer_append(&buffer, "<think>\n", err);
    }
    if (status != FG_OK) {
        free(buffer.data);
        return status;
    }
    *rendered = buffer.data;
    return FG_OK;
}

fg_status fg_chat_render(const fg_chat_message *messages, size_t message_count,
                         const fg_chat_render_options *options, char **rendered,
                         fg_error *err) {
    return chat_render(messages, message_count, options, false, rendered, err);
}

fg_status fg_chat_render_continuation(const fg_chat_message *messages, size_t message_count,
                                      const fg_chat_render_options *options, char **rendered,
                                      fg_error *err) {
    return chat_render(messages, message_count, options, true, rendered, err);
}

static fg_status append_json_escaped(fg_text_buffer *buffer, const char *text, size_t length,
                                     fg_error *err) {
    fg_status status = buffer_append(buffer, "\"", err);
    for (size_t i = 0; status == FG_OK && i < length; i++) {
        unsigned char c = (unsigned char)text[i];
        switch (c) {
            case '"': status = buffer_append(buffer, "\\\"", err); break;
            case '\\': status = buffer_append(buffer, "\\\\", err); break;
            case '\b': status = buffer_append(buffer, "\\b", err); break;
            case '\f': status = buffer_append(buffer, "\\f", err); break;
            case '\n': status = buffer_append(buffer, "\\n", err); break;
            case '\r': status = buffer_append(buffer, "\\r", err); break;
            case '\t': status = buffer_append(buffer, "\\t", err); break;
            default:
                if (c < 0x20u) {
                    char escaped[7];
                    snprintf(escaped, sizeof(escaped), "\\u%04x", c);
                    status = buffer_append(buffer, escaped, err);
                } else {
                    status = buffer_append_n(buffer, (const char *)&c, 1u, err);
                }
        }
    }
    if (status == FG_OK) status = buffer_append(buffer, "\"", err);
    return status;
}

static bool strict_json_number(const char **cursor) {
    const char *p = *cursor;
    if (*p == '-') p++;
    if (*p == '0') p++;
    else {
        if (!isdigit((unsigned char)*p)) return false;
        while (isdigit((unsigned char)*p)) p++;
    }
    if (*p == '.') {
        p++;
        if (!isdigit((unsigned char)*p)) return false;
        while (isdigit((unsigned char)*p)) p++;
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        if (!isdigit((unsigned char)*p)) return false;
        while (isdigit((unsigned char)*p)) p++;
    }
    *cursor = p;
    return true;
}

static bool strict_json_value(const char **cursor) {
    json_skip_space(cursor);
    if (**cursor == '"') return json_scan_string(cursor);
    if (**cursor == '{' || **cursor == '[') {
        char close = **cursor == '{' ? '}' : ']';
        bool object = **cursor == '{';
        (*cursor)++;
        json_skip_space(cursor);
        if (**cursor == close) {
            (*cursor)++;
            return true;
        }
        for (;;) {
            if (object) {
                if (!json_scan_string(cursor)) return false;
                json_skip_space(cursor);
                if (*(*cursor)++ != ':') return false;
            }
            if (!strict_json_value(cursor)) return false;
            json_skip_space(cursor);
            if (**cursor == close) {
                (*cursor)++;
                return true;
            }
            if (*(*cursor)++ != ',') return false;
            json_skip_space(cursor);
        }
    }
    if (!strncmp(*cursor, "true", 4)) {
        *cursor += 4;
        return true;
    }
    if (!strncmp(*cursor, "false", 5)) {
        *cursor += 5;
        return true;
    }
    if (!strncmp(*cursor, "null", 4)) {
        *cursor += 4;
        return true;
    }
    return strict_json_number(cursor);
}

static bool raw_json_complete(const char *text) {
    const char *cursor = text;
    if (!strict_json_value(&cursor)) return false;
    json_skip_space(&cursor);
    return *cursor == 0;
}

static char *copy_range(const char *start, const char *end) {
    size_t length = (size_t)(end - start);
    char *copy = malloc(length + 1u);
    if (!copy) return NULL;
    memcpy(copy, start, length);
    copy[length] = 0;
    return copy;
}

static void trim_one_newline(const char **start, const char **end) {
    if (*start < *end && **start == '\r') (*start)++;
    if (*start < *end && **start == '\n') (*start)++;
    if (*end > *start && (*end)[-1] == '\n') (*end)--;
    if (*end > *start && (*end)[-1] == '\r') (*end)--;
}

static fg_status generated_push_call(fg_chat_generated *generated, char *name, char *arguments,
                                     fg_error *err) {
    fg_chat_tool_call *calls =
        realloc(generated->tool_calls,
                (generated->tool_call_count + 1u) * sizeof(*generated->tool_calls));
    if (!calls) {
        free(name);
        free(arguments);
        fg_error_set(err, FG_ERR_OOM, "grow generated tool calls");
        return FG_ERR_OOM;
    }
    generated->tool_calls = calls;
    calls[generated->tool_call_count++] =
        (fg_chat_tool_call){.name = name, .arguments_json = arguments};
    return FG_OK;
}

static fg_status parse_native_tool_call(const char **cursor, fg_chat_generated *generated,
                                        fg_error *err) {
    static const char tool_start[] = "<tool_call>";
    static const char tool_end[] = "</tool_call>";
    static const char function_start[] = "<function=";
    static const char function_end[] = "</function>";
    static const char parameter_start[] = "<parameter=";
    static const char parameter_end[] = "</parameter>";
    const char *p = *cursor;
    if (strncmp(p, tool_start, sizeof(tool_start) - 1u)) return FG_ERR_FORMAT;
    p += sizeof(tool_start) - 1u;
    while (isspace((unsigned char)*p)) p++;
    if (strncmp(p, function_start, sizeof(function_start) - 1u)) goto malformed;
    const char *name_start = p + sizeof(function_start) - 1u;
    const char *tag_end = strchr(name_start, '>');
    if (!tag_end) goto malformed;
    const char *name_end = tag_end;
    while (name_end > name_start && isspace((unsigned char)name_end[-1])) name_end--;
    if (name_end == name_start) goto malformed;
    char *name = copy_range(name_start, name_end);
    if (!name) {
        fg_error_set(err, FG_ERR_OOM, "copy generated tool name");
        return FG_ERR_OOM;
    }
    if (!valid_qwen_tag_value(name, false)) {
        free(name);
        goto malformed;
    }
    p = tag_end + 1;
    fg_text_buffer arguments = {0};
    fg_status status = buffer_append(&arguments, "{", err);
    bool wrote = false;
    while (status == FG_OK) {
        while (isspace((unsigned char)*p)) p++;
        if (!strncmp(p, function_end, sizeof(function_end) - 1u)) break;
        if (strncmp(p, parameter_start, sizeof(parameter_start) - 1u)) {
            status = FG_ERR_FORMAT;
            break;
        }
        const char *key_start = p + sizeof(parameter_start) - 1u;
        tag_end = strchr(key_start, '>');
        if (!tag_end) {
            status = FG_ERR_FORMAT;
            break;
        }
        const char *key_end = tag_end;
        while (key_end > key_start && isspace((unsigned char)key_end[-1])) key_end--;
        char *key = copy_range(key_start, key_end);
        if (!key) {
            fg_error_set(err, FG_ERR_OOM, "copy generated tool argument name");
            status = FG_ERR_OOM;
            break;
        }
        if (!valid_qwen_tag_value(key, true)) {
            free(key);
            status = FG_ERR_FORMAT;
            break;
        }
        const char *value_start = tag_end + 1;
        const char *parameter_close = strstr(value_start, parameter_end);
        if (!parameter_close) {
            free(key);
            status = FG_ERR_FORMAT;
            break;
        }
        const char *value_end = parameter_close;
        trim_one_newline(&value_start, &value_end);
        char *raw_value = copy_range(value_start, value_end);
        if (!raw_value) {
            free(key);
            fg_error_set(err, FG_ERR_OOM, "copy generated tool argument");
            status = FG_ERR_OOM;
            break;
        }
        if (wrote) status = buffer_append(&arguments, ",", err);
        if (status == FG_OK)
            status = append_json_escaped(&arguments, key, strlen(key), err);
        free(key);
        if (status == FG_OK) status = buffer_append(&arguments, ":", err);
        if (status == FG_OK && raw_json_complete(raw_value))
            status = append_minified_json(&arguments, raw_value,
                                          raw_value + strlen(raw_value), err);
        else if (status == FG_OK)
            status = append_json_escaped(&arguments, raw_value, strlen(raw_value), err);
        free(raw_value);
        if (status != FG_OK) break;
        wrote = true;
        p = parameter_close + sizeof(parameter_end) - 1u;
    }
    if (status == FG_OK) status = buffer_append(&arguments, "}", err);
    if (status != FG_OK || strncmp(p, function_end, sizeof(function_end) - 1u)) {
        free(name);
        free(arguments.data);
        if (status == FG_OK || !err->message[0])
            fg_error_set(err, FG_ERR_FORMAT, "malformed generated Qwen tool call");
        return status == FG_OK ? FG_ERR_FORMAT : status;
    }
    p += sizeof(function_end) - 1u;
    while (isspace((unsigned char)*p)) p++;
    if (strncmp(p, tool_end, sizeof(tool_end) - 1u)) {
        free(name);
        free(arguments.data);
        goto malformed;
    }
    p += sizeof(tool_end) - 1u;
    status = generated_push_call(generated, name, arguments.data, err);
    if (status == FG_OK) *cursor = p;
    return status;

malformed:
    fg_error_set(err, FG_ERR_FORMAT, "malformed generated Qwen tool call");
    return FG_ERR_FORMAT;
}

static bool range_has_qwen_tool_syntax(const char *start, const char *end) {
    static const char *markers[] = {
        "<tool_call", "</tool_call", "<function=", "</function",
        "<parameter=", "</parameter",
    };
    for (size_t i = 0; i < sizeof(markers) / sizeof(markers[0]); i++) {
        const char *found = strstr(start, markers[i]);
        if (found && found < end) return true;
    }
    return false;
}

void fg_chat_generated_free(fg_chat_generated *generated) {
    if (!generated) return;
    free(generated->content);
    free(generated->reasoning);
    for (size_t i = 0; i < generated->tool_call_count; i++) {
        free((char *)generated->tool_calls[i].id);
        free((char *)generated->tool_calls[i].name);
        free((char *)generated->tool_calls[i].arguments_json);
    }
    free(generated->tool_calls);
    memset(generated, 0, sizeof(*generated));
}

fg_status fg_chat_parse_generated(const char *text, bool thinking,
                                  fg_chat_generated *generated, fg_error *err) {
    if (!text || !generated) {
        fg_error_set(err, FG_ERR_ARGUMENT, "invalid generated chat parser arguments");
        return FG_ERR_ARGUMENT;
    }
    memset(generated, 0, sizeof(*generated));
    static const char end_marker[] = "<|im_end|>";
    const char *sentinel = strstr(text, end_marker);
    const char *raw_end = sentinel ? sentinel : text + strlen(text);
    while (raw_end > text && isspace((unsigned char)raw_end[-1])) raw_end--;
    char *bounded = copy_range(text, raw_end);
    if (!bounded) {
        fg_error_set(err, FG_ERR_OOM, "copy generated chat text");
        return FG_ERR_OOM;
    }
    const char *visible = bounded;
    if (thinking) {
        const char *close = NULL;
        for (const char *scan = bounded; (scan = strstr(scan, "</think>")) != NULL; scan++)
            close = scan;
        if (close) {
            const char *reasoning_start =
                !strncmp(bounded, "<think>", 7u) ? bounded + 7u : bounded;
            generated->reasoning = copy_range(reasoning_start, close);
            visible = close + 8u;
        } else {
            const char *tool = strstr(bounded, "<tool_call>");
            if (!tool || !strstr(tool, "</tool_call>")) {
                if (range_has_qwen_tool_syntax(bounded, bounded + strlen(bounded))) {
                    free(bounded);
                    fg_chat_generated_free(generated);
                    fg_error_set(err, FG_ERR_FORMAT,
                                 "incomplete generated Qwen tool call");
                    return FG_ERR_FORMAT;
                }
                const char *reasoning_start =
                    !strncmp(bounded, "<think>", 7u) ? bounded + 7u : bounded;
                generated->reasoning = strdup(reasoning_start);
                generated->content = strdup("");
                free(bounded);
                if (!generated->reasoning || !generated->content) {
                    fg_chat_generated_free(generated);
                    fg_error_set(err, FG_ERR_OOM, "copy generated reasoning");
                    return FG_ERR_OOM;
                }
                return FG_OK;
            }
            const char *reasoning_start =
                !strncmp(bounded, "<think>", 7u) ? bounded + 7u : bounded;
            generated->reasoning = copy_range(reasoning_start, tool);
            visible = tool;
        }
        if (!generated->reasoning) {
            free(bounded);
            fg_error_set(err, FG_ERR_OOM, "copy generated reasoning");
            return FG_ERR_OOM;
        }
    }
    while (*visible == '\r' || *visible == '\n') visible++;
    const char *tool = strstr(visible, "<tool_call>");
    if (!tool) {
        if (range_has_qwen_tool_syntax(visible, visible + strlen(visible))) {
            free(bounded);
            fg_chat_generated_free(generated);
            fg_error_set(err, FG_ERR_FORMAT, "malformed generated Qwen tool syntax");
            return FG_ERR_FORMAT;
        }
        generated->content = strdup(visible);
    } else {
        if (range_has_qwen_tool_syntax(visible, tool)) {
            free(bounded);
            fg_chat_generated_free(generated);
            fg_error_set(err, FG_ERR_FORMAT, "malformed generated Qwen tool syntax");
            return FG_ERR_FORMAT;
        }
        const char *content_end = tool;
        while (content_end > visible && isspace((unsigned char)content_end[-1])) content_end--;
        generated->content = copy_range(visible, content_end);
        const char *cursor = tool;
        fg_status status = FG_OK;
        while (status == FG_OK) {
            status = parse_native_tool_call(&cursor, generated, err);
            if (status != FG_OK) break;
            while (isspace((unsigned char)*cursor)) cursor++;
            if (!*cursor) break;
            if (strncmp(cursor, "<tool_call>", 11u)) {
                fg_error_set(err, FG_ERR_FORMAT,
                             "generated text after a Qwen tool call is not supported");
                status = FG_ERR_FORMAT;
            }
        }
        if (status != FG_OK) {
            free(bounded);
            fg_chat_generated_free(generated);
            return status;
        }
    }
    free(bounded);
    if (!generated->content) {
        fg_chat_generated_free(generated);
        fg_error_set(err, FG_ERR_OOM, "copy generated assistant content");
        return FG_ERR_OOM;
    }
    return FG_OK;
}

#ifndef FG_CHAT_RENDER_ONLY
typedef struct fg_chat_transcript {
    fg_chat_message *messages;
    size_t count;
    size_t capacity;
} fg_chat_transcript;

static void transcript_clear(fg_chat_transcript *transcript) {
    for (size_t i = 0; i < transcript->count; i++) {
        free((char *)transcript->messages[i].role);
        free((char *)transcript->messages[i].content);
    }
    transcript->count = 0;
}

static void transcript_destroy(fg_chat_transcript *transcript) {
    transcript_clear(transcript);
    free(transcript->messages);
    memset(transcript, 0, sizeof(*transcript));
}

static fg_status transcript_push(fg_chat_transcript *transcript, const char *role,
                                 const char *content, fg_error *err) {
    if (transcript->count == transcript->capacity) {
        size_t capacity = transcript->capacity ? transcript->capacity * 2u : 8u;
        fg_chat_message *messages = realloc(transcript->messages, capacity * sizeof(*messages));
        if (!messages) {
            fg_error_set(err, FG_ERR_OOM, "grow chat transcript");
            return FG_ERR_OOM;
        }
        transcript->messages = messages;
        transcript->capacity = capacity;
    }
    char *role_copy = strdup(role);
    char *content_copy = strdup(content ? content : "");
    if (!role_copy || !content_copy) {
        free(role_copy);
        free(content_copy);
        fg_error_set(err, FG_ERR_OOM, "copy chat transcript message");
        return FG_ERR_OOM;
    }
    transcript->messages[transcript->count++] =
        (fg_chat_message){.role = role_copy, .content = content_copy};
    return FG_OK;
}

static void chat_signal_handler(int signal_number) {
    (void)signal_number;
    fg_chat_stop_requested = 1;
}

static bool chat_interrupted(void *context) {
    (void)context;
    return fg_chat_stop_requested != 0;
}

typedef struct chat_stream {
    fg_text_buffer captured;
    fg_text_buffer pending;
    bool stopped;
} chat_stream;

static fg_status chat_stream_write(chat_stream *stream,const char *text,size_t length,
                                   fg_error *err) {
    if (length && fwrite(text, 1u, length, stdout) != length) {
        fg_error_set(err, FG_ERR_IO, "write streamed chat token: %s", strerror(errno));
        return FG_ERR_IO;
    }
    fflush(stdout);
    return buffer_append_n(&stream->captured, text, length, err);
}

static fg_status chat_token(void *context, uint32_t token, const char *text, size_t length,
                            fg_error *err) {
    static const char marker[]="<|im_end|>";
    (void)token;
    chat_stream *stream = context;
    if(stream->stopped)return FG_OK;
    fg_status status=buffer_append_n(&stream->pending,text,length,err);
    if(status!=FG_OK)return status;
    char *found=strstr(stream->pending.data,marker);
    if(found){
        size_t bytes=(size_t)(found-stream->pending.data);
        if(bytes)status=chat_stream_write(stream,stream->pending.data,bytes,err);
        stream->pending.length=0;if(stream->pending.data)stream->pending.data[0]=0;
        stream->stopped=true;return status;
    }
    size_t keep=0,max_keep=sizeof(marker)-2u;
    if(max_keep>stream->pending.length)max_keep=stream->pending.length;
    for(size_t candidate=max_keep;candidate>0;candidate--){
        if(!memcmp(stream->pending.data+stream->pending.length-candidate,marker,candidate)){
            keep=candidate;break;
        }
    }
    size_t flush=stream->pending.length-keep;
    if(flush)status=chat_stream_write(stream,stream->pending.data,flush,err);
    if(status==FG_OK&&flush){
        memmove(stream->pending.data,stream->pending.data+flush,keep);
        stream->pending.length=keep;stream->pending.data[keep]=0;
    }
    return status;
}

static fg_status chat_generate_turn(
    fg_runtime *runtime,const char *full_prompt,const char *continuation,
    bool try_continuation,uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err) {
    if(!try_continuation)
        return fg_runtime_generate(runtime,full_prompt,max_tokens,callback,callback_context,
                                   interrupted,interrupt_context,stats,err);
    bool prefix_miss=false;
    fg_status status=fg_runtime_generate_continuation(
        runtime,full_prompt,continuation,&prefix_miss,max_tokens,callback,
        callback_context,interrupted,interrupt_context,stats,err);
    if(status!=FG_ERR_UNAVAILABLE||!prefix_miss)return status;
    memset(stats,0,sizeof(*stats));
    memset(err,0,sizeof(*err));
    status=fg_runtime_reset(runtime,err);
    if(status==FG_OK)
        status=fg_runtime_generate(runtime,full_prompt,max_tokens,callback,
                                   callback_context,interrupted,interrupt_context,
                                   stats,err);
    return status;
}

fg_status fg_chat_main_with_options(const char *manifest_path, uint32_t max_tokens,
                                    const fg_runtime_options *runtime_options,
                                    fg_error *err) {
    if (!manifest_path || !max_tokens) {
        fg_error_set(err, FG_ERR_ARGUMENT, "chat requires a manifest and positive max tokens");
        return FG_ERR_ARGUMENT;
    }
    fg_runtime *runtime = NULL;
    fg_status status = fg_runtime_open_with_options(&runtime, manifest_path,
                                                    runtime_options, err);
    if (status != FG_OK) return status;

    struct sigaction action = {0}, old_int = {0}, old_term = {0};
    action.sa_handler = chat_signal_handler;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, &old_int);
    sigaction(SIGTERM, &action, &old_term);

    fg_chat_transcript transcript = {0};
    bool continuation_ready=false;
    char *line = NULL;
    size_t line_capacity = 0;
    fprintf(stderr, "Flash Gordon chat (%s). /clear resets, /quit exits.\n",
            fg_runtime_model_name(runtime));
    while (status == FG_OK) {
        fg_chat_stop_requested = 0;
        fputs("> ", stdout);
        fflush(stdout);
        ssize_t length = getline(&line, &line_capacity, stdin);
        if (length < 0) break;
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r'))
            line[--length] = 0;
        if (!strcmp(line, "/quit")) break;
        if (!strcmp(line, "/clear")) {
            transcript_clear(&transcript);
            status = fg_runtime_reset(runtime, err);
            continuation_ready=false;
            if (status == FG_OK) fputs("session cleared\n", stdout);
            continue;
        }
        if (!line[0]) continue;
        size_t turn_start = transcript.count;
        status = transcript_push(&transcript, "user", line, err);
        char *prompt = NULL;
        if (status == FG_OK)
            status = fg_chat_render(transcript.messages, transcript.count, NULL, &prompt, err);
        char *continuation=NULL;
        if(status==FG_OK&&continuation_ready)
            status=fg_chat_render_continuation(
                transcript.messages+turn_start,transcript.count-turn_start,
                NULL,&continuation,err);
        chat_stream stream = {0};
        fg_generation_stats stats = {0};
        if (status == FG_OK)
            status = chat_generate_turn(
                runtime,prompt,continuation,continuation_ready,max_tokens,
                chat_token,&stream,chat_interrupted,NULL,&stats,err);
        if(status==FG_OK&&!stream.stopped&&stream.pending.length)
            status=chat_stream_write(&stream,stream.pending.data,stream.pending.length,err);
        free(prompt);
        free(continuation);
        fputc('\n', stdout);
        if (status == FG_OK) {
            fg_text_buffer replay = {0};
            status = buffer_append(&replay, "<think>\n", err);
            if (status == FG_OK && stream.captured.length)
                status = buffer_append_n(&replay, stream.captured.data, stream.captured.length, err);
            if (status == FG_OK)
                status = transcript_push(&transcript, "assistant", replay.data, err);
            free(replay.data);
            continuation_ready=status==FG_OK;
        }
        free(stream.captured.data);
        free(stream.pending.data);
        if (status == FG_OK) {
            double prefill_tps =
                stats.prefill_seconds > 0.0 ?
                    (double)stats.prefilled_tokens / stats.prefill_seconds : 0.0;
            double decode_tps =
                stats.decode_seconds > 0.0 ? (double)stats.generated_tokens / stats.decode_seconds :
                                             0.0;
            fprintf(stderr,
                    "[prefix %s, reused %u, reset %s; prefill %u/%u tokens, %.2f tok/s; "
                    "generation %u tokens, %.2f tok/s; context %u/%u]\n",
                    stats.prefix_cache_hit ? "hit" : "miss", stats.reused_tokens,
                    fg_prefix_reset_reason_name(stats.reset_reason), stats.prefilled_tokens,
                    stats.prompt_tokens, prefill_tps, stats.generated_tokens, decode_tps,
                    stats.context_tokens, fg_runtime_context_limit(runtime));
        }
        if (fg_chat_stop_requested && status == FG_OK)
            fputs("[generation stopped between tokens]\n", stderr);
        if (status == FG_ERR_ARGUMENT || status == FG_ERR_FORMAT || status == FG_ERR_LIMIT) {
            fprintf(stderr, "[turn rejected: %s]\n",
                    err->message[0] ? err->message : "invalid turn");
            while (transcript.count > turn_start) {
                transcript.count--;
                free((char *)transcript.messages[transcript.count].role);
                free((char *)transcript.messages[transcript.count].content);
                memset(&transcript.messages[transcript.count], 0,
                       sizeof(transcript.messages[transcript.count]));
            }
            fg_error reset_error = {0};
            status = fg_runtime_reset(runtime, &reset_error);
            continuation_ready=false;
            if (status != FG_OK) *err = reset_error;
        }
    }
    free(line);
    transcript_destroy(&transcript);
    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGTERM, &old_term, NULL);
    fg_runtime_close(runtime);
    return status;
}

fg_status fg_chat_main(const char *manifest_path, uint32_t max_tokens, fg_error *err) {
    return fg_chat_main_with_options(manifest_path, max_tokens, NULL, err);
}
#endif
