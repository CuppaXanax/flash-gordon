#include "fg_api.h"
#include "fg_chat.h"
#include "fg_runtime.h"

#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define FG_API_MAX_REQUEST_BYTES (1024u * 1024u)
#define FG_API_MAX_OBJECT_MEMBERS 256u
#define FG_API_IO_TIMEOUT_SECONDS 30
#define FG_API_DEFAULT_MAX_TOKENS 512u

typedef struct api_buffer {
    char *data;
    size_t length;
    size_t capacity;
} api_buffer;

typedef enum json_type {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type;

typedef struct json_value json_value;

typedef struct json_member {
    char *name;
    json_value *value;
} json_member;

struct json_value {
    json_type type;
    union {
        bool boolean;
        double number;
        char *string;
        struct {
            json_value **items;
            size_t count;
        } array;
        struct {
            json_member *members;
            size_t count;
        } object;
    } as;
};

typedef struct json_parser {
    const char *cursor;
    const char *end;
    unsigned depth;
    fg_error *err;
} json_parser;

typedef struct http_request {
    char method[8];
    char path[256];
    char *body;
    size_t body_length;
} http_request;

typedef struct api_chat_request {
    fg_chat_message *messages;
    size_t message_count;
    char **tool_schemas;
    size_t tool_schema_count;
    fg_chat_tool_choice tool_choice;
    char *tool_choice_name;
    uint32_t max_tokens;
    bool stream;
} api_chat_request;

typedef struct api_generation {
    int fd;
    bool stream;
    api_buffer content;
    const char *id;
    const char *model;
    time_t created;
    const api_chat_request *request;
    bool client_failed;
    bool think_closed;
    bool output_stopped;
    api_buffer visible_pending;
    char utf8_pending[4];
    size_t utf8_pending_length;
    size_t streamed_tool_calls;
} api_generation;

static volatile sig_atomic_t api_stop_requested;
static unsigned long long api_request_sequence;

static int utf8_unit(const unsigned char *text,size_t available,size_t *bytes);

static fg_status buffer_reserve(api_buffer *buffer, size_t extra, fg_error *err) {
    if (extra > SIZE_MAX - buffer->length - 1u) {
        fg_error_set(err, FG_ERR_LIMIT, "API response exceeds addressable memory");
        return FG_ERR_LIMIT;
    }
    size_t required = buffer->length + extra + 1u;
    if (required <= buffer->capacity) return FG_OK;
    size_t capacity = buffer->capacity ? buffer->capacity : 512u;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    char *data = realloc(buffer->data, capacity);
    if (!data) {
        fg_error_set(err, FG_ERR_OOM, "allocate API buffer");
        return FG_ERR_OOM;
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return FG_OK;
}

static fg_status buffer_append_n(api_buffer *buffer, const char *text, size_t length,
                                 fg_error *err) {
    fg_status status = buffer_reserve(buffer, length, err);
    if (status != FG_OK) return status;
    if (length) memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = 0;
    return FG_OK;
}

static fg_status buffer_append(api_buffer *buffer, const char *text, fg_error *err) {
    return buffer_append_n(buffer, text ? text : "", text ? strlen(text) : 0u, err);
}

static fg_status buffer_append_json_string(api_buffer *buffer, const char *text, size_t length,
                                           fg_error *err) {
    fg_status status = buffer_append(buffer, "\"", err);
    for (size_t i = 0; status == FG_OK && i < length;) {
        unsigned char c = (unsigned char)text[i];
        if(c>=0x80u){
            size_t unit=0;int valid=utf8_unit((const unsigned char *)text+i,length-i,&unit);
            if(valid>0){status=buffer_append_n(buffer,text+i,unit,err);i+=unit;continue;}
            status=buffer_append_n(buffer,"\xef\xbf\xbd",3u,err);
            i+=valid==0?length-i:1u;
            continue;
        }
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
        i++;
    }
    if (status == FG_OK) status = buffer_append(buffer, "\"", err);
    return status;
}

static fg_status send_all(int fd, const char *data, size_t length, fg_error *err) {
    while (length) {
        ssize_t sent = send(fd, data, length, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR && !api_stop_requested) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                fg_error_set(err, FG_ERR_IO, "HTTP response write timed out");
            else if (errno == EINTR)
                fg_error_set(err, FG_ERR_IO, "HTTP response interrupted by shutdown");
            else
                fg_error_set(err, FG_ERR_IO, "send HTTP response: %s", strerror(errno));
            return FG_ERR_IO;
        }
        if (!sent) {
            fg_error_set(err, FG_ERR_IO, "HTTP peer closed during response");
            return FG_ERR_IO;
        }
        data += (size_t)sent;
        length -= (size_t)sent;
    }
    return FG_OK;
}

static const char *http_reason(unsigned status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Content Too Large";
        case 415: return "Unsupported Media Type";
        default: return "Internal Server Error";
    }
}

static fg_status send_response_with_headers(int fd, unsigned status, const char *content_type,
                                            const char *extra_headers, const char *body,
                                            size_t body_length, fg_error *err) {
    char header[1024];
    int length = snprintf(header, sizeof(header),
                          "HTTP/1.1 %u %s\r\n"
                          "Content-Type: %s\r\n"
                          "Content-Length: %zu\r\n"
                          "Connection: close\r\n"
                          "Cache-Control: no-store\r\n"
                          "%s\r\n",
                          status, http_reason(status), content_type, body_length,
                          extra_headers ? extra_headers : "");
    if (length < 0 || (size_t)length >= sizeof(header)) {
        fg_error_set(err, FG_ERR_LIMIT, "HTTP response header overflow");
        return FG_ERR_LIMIT;
    }
    fg_status result = send_all(fd, header, (size_t)length, err);
    if (result == FG_OK) result = send_all(fd, body, body_length, err);
    return result;
}

static fg_status send_response(int fd, unsigned status, const char *content_type,
                               const char *body, size_t body_length, fg_error *err) {
    return send_response_with_headers(fd, status, content_type, NULL, body, body_length, err);
}

static fg_status send_error_response(int fd, unsigned status, const char *message,
                                     fg_error *err) {
    api_buffer body = {0};
    fg_status result = buffer_append(&body, "{\"error\":{\"message\":", err);
    if (result == FG_OK)
        result = buffer_append_json_string(&body, message, strlen(message), err);
    if (result == FG_OK)
        result = buffer_append(&body, ",\"type\":\"invalid_request_error\"}}", err);
    if (result == FG_OK)
        result = send_response(fd, status, "application/json", body.data, body.length, err);
    free(body.data);
    return result;
}

static void json_skip_space(json_parser *parser) {
    while (parser->cursor < parser->end &&
           isspace((unsigned char)*parser->cursor))
        parser->cursor++;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static fg_status json_append_utf8(api_buffer *buffer, uint32_t value, fg_error *err) {
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

static char *json_parse_string(json_parser *parser) {
    if (parser->cursor >= parser->end || *parser->cursor != '"') return NULL;
    parser->cursor++;
    api_buffer text = {0};
    while (parser->cursor < parser->end) {
        unsigned char c = (unsigned char)*parser->cursor++;
        if (c == '"') return text.data ? text.data : strdup("");
        if (c < 0x20u) break;
        if (c != '\\') {
            if (buffer_append_n(&text, (const char *)&c, 1u, parser->err) != FG_OK)
                goto fail;
            continue;
        }
        if (parser->cursor >= parser->end) break;
        char escaped = *parser->cursor++;
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
                if (parser->end - parser->cursor < 4) goto invalid;
                uint32_t value = 0;
                for (unsigned i = 0; i < 4u; i++) {
                    int digit = json_hex(parser->cursor[i]);
                    if (digit < 0) goto invalid;
                    value = value * 16u + (uint32_t)digit;
                }
                parser->cursor += 4;
                if (value >= 0xd800u && value <= 0xdbffu) {
                    if (parser->end - parser->cursor < 6 || parser->cursor[0] != '\\' ||
                        parser->cursor[1] != 'u')
                        goto invalid;
                    parser->cursor += 2;
                    uint32_t low = 0;
                    for (unsigned i = 0; i < 4u; i++) {
                        int digit = json_hex(parser->cursor[i]);
                        if (digit < 0) goto invalid;
                        low = low * 16u + (uint32_t)digit;
                    }
                    parser->cursor += 4;
                    if (low < 0xdc00u || low > 0xdfffu) goto invalid;
                    value = 0x10000u + ((value - 0xd800u) << 10u) + (low - 0xdc00u);
                } else if (value >= 0xdc00u && value <= 0xdfffu) {
                    goto invalid;
                }
                if (value == 0u) goto invalid;
                if (json_append_utf8(&text, value, parser->err) != FG_OK) goto fail;
                continue;
            }
            default: goto invalid;
        }
        if (buffer_append_n(&text, &decoded, 1u, parser->err) != FG_OK) goto fail;
    }
invalid:
    fg_error_set(parser->err, FG_ERR_FORMAT, "invalid JSON string");
fail:
    free(text.data);
    return NULL;
}

static void json_free(json_value *value) {
    if (!value) return;
    if (value->type == JSON_STRING) free(value->as.string);
    else if (value->type == JSON_ARRAY) {
        for (size_t i = 0; i < value->as.array.count; i++) json_free(value->as.array.items[i]);
        free(value->as.array.items);
    } else if (value->type == JSON_OBJECT) {
        for (size_t i = 0; i < value->as.object.count; i++) {
            free(value->as.object.members[i].name);
            json_free(value->as.object.members[i].value);
        }
        free(value->as.object.members);
    }
    free(value);
}

static json_value *json_parse_value(json_parser *parser);

static json_value *json_parse_number(json_parser *parser) {
    const char *start = parser->cursor;
    const char *cursor = start;
    if (cursor < parser->end && *cursor == '-') cursor++;
    if (cursor >= parser->end) return NULL;
    if (*cursor == '0') {
        cursor++;
        if (cursor < parser->end && isdigit((unsigned char)*cursor)) return NULL;
    } else {
        if (!isdigit((unsigned char)*cursor)) return NULL;
        while (cursor < parser->end && isdigit((unsigned char)*cursor)) cursor++;
    }
    if (cursor < parser->end && *cursor == '.') {
        cursor++;
        if (cursor >= parser->end || !isdigit((unsigned char)*cursor)) return NULL;
        while (cursor < parser->end && isdigit((unsigned char)*cursor)) cursor++;
    }
    if (cursor < parser->end && (*cursor == 'e' || *cursor == 'E')) {
        cursor++;
        if (cursor < parser->end && (*cursor == '+' || *cursor == '-')) cursor++;
        if (cursor >= parser->end || !isdigit((unsigned char)*cursor)) return NULL;
        while (cursor < parser->end && isdigit((unsigned char)*cursor)) cursor++;
    }
    errno = 0;
    char *number_end = NULL;
    double number = strtod(start, &number_end);
    if (number_end != cursor || errno == ERANGE) return NULL;
    json_value *value = calloc(1u, sizeof(*value));
    if (!value) {
        fg_error_set(parser->err, FG_ERR_OOM, "allocate JSON number");
        return NULL;
    }
    value->type = JSON_NUMBER;
    value->as.number = number;
    parser->cursor = cursor;
    return value;
}

static bool json_push_array(json_value *array, json_value *item, fg_error *err) {
    size_t count = array->as.array.count;
    json_value **items = realloc(array->as.array.items, (count + 1u) * sizeof(*items));
    if (!items) {
        fg_error_set(err, FG_ERR_OOM, "grow JSON array");
        return false;
    }
    array->as.array.items = items;
    items[count] = item;
    array->as.array.count++;
    return true;
}

static bool json_push_member(json_value *object, char *name, json_value *item,
                             fg_error *err) {
    size_t count = object->as.object.count;
    if (count >= FG_API_MAX_OBJECT_MEMBERS) {
        fg_error_set(err, FG_ERR_LIMIT, "JSON object exceeds %u members",
                     FG_API_MAX_OBJECT_MEMBERS);
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (!strcmp(object->as.object.members[i].name, name)) {
            fg_error_set(err, FG_ERR_FORMAT, "duplicate JSON object member '%s'", name);
            return false;
        }
    }
    json_member *members =
        realloc(object->as.object.members, (count + 1u) * sizeof(*members));
    if (!members) {
        fg_error_set(err, FG_ERR_OOM, "grow JSON object");
        return false;
    }
    object->as.object.members = members;
    members[count] = (json_member){.name = name, .value = item};
    object->as.object.count++;
    return true;
}

static json_value *json_parse_array(json_parser *parser) {
    json_value *array = calloc(1u, sizeof(*array));
    if (!array) {
        fg_error_set(parser->err, FG_ERR_OOM, "allocate JSON array");
        return NULL;
    }
    array->type = JSON_ARRAY;
    parser->cursor++;
    json_skip_space(parser);
    if (parser->cursor < parser->end && *parser->cursor == ']') {
        parser->cursor++;
        return array;
    }
    for (;;) {
        json_value *item = json_parse_value(parser);
        if (!item || !json_push_array(array, item, parser->err)) {
            json_free(item);
            json_free(array);
            return NULL;
        }
        json_skip_space(parser);
        if (parser->cursor >= parser->end) break;
        if (*parser->cursor == ']') {
            parser->cursor++;
            return array;
        }
        if (*parser->cursor++ != ',') break;
        json_skip_space(parser);
    }
    fg_error_set(parser->err, FG_ERR_FORMAT, "invalid JSON array");
    json_free(array);
    return NULL;
}

static json_value *json_parse_object(json_parser *parser) {
    json_value *object = calloc(1u, sizeof(*object));
    if (!object) {
        fg_error_set(parser->err, FG_ERR_OOM, "allocate JSON object");
        return NULL;
    }
    object->type = JSON_OBJECT;
    parser->cursor++;
    json_skip_space(parser);
    if (parser->cursor < parser->end && *parser->cursor == '}') {
        parser->cursor++;
        return object;
    }
    for (;;) {
        char *name = json_parse_string(parser);
        if (!name) break;
        json_skip_space(parser);
        if (parser->cursor >= parser->end || *parser->cursor++ != ':') {
            free(name);
            break;
        }
        json_skip_space(parser);
        json_value *item = json_parse_value(parser);
        if (!item || !json_push_member(object, name, item, parser->err)) {
            free(name);
            json_free(item);
            json_free(object);
            return NULL;
        }
        json_skip_space(parser);
        if (parser->cursor >= parser->end) break;
        if (*parser->cursor == '}') {
            parser->cursor++;
            return object;
        }
        if (*parser->cursor++ != ',') break;
        json_skip_space(parser);
    }
    if (parser->err->code == FG_OK)
        fg_error_set(parser->err, FG_ERR_FORMAT, "invalid JSON object");
    json_free(object);
    return NULL;
}

static json_value *json_parse_value(json_parser *parser) {
    if (++parser->depth > 64u) {
        fg_error_set(parser->err, FG_ERR_LIMIT, "JSON nesting exceeds 64 levels");
        parser->depth--;
        return NULL;
    }
    json_skip_space(parser);
    json_value *value = NULL;
    if (parser->cursor < parser->end && *parser->cursor == '{') {
        value = json_parse_object(parser);
    } else if (parser->cursor < parser->end && *parser->cursor == '[') {
        value = json_parse_array(parser);
    } else if (parser->cursor < parser->end && *parser->cursor == '"') {
        char *string = json_parse_string(parser);
        if (string) {
            value = calloc(1u, sizeof(*value));
            if (value) {
                value->type = JSON_STRING;
                value->as.string = string;
            } else {
                free(string);
                fg_error_set(parser->err, FG_ERR_OOM, "allocate JSON string");
            }
        }
    } else {
        const char *start = parser->cursor;
        if ((size_t)(parser->end - start) >= 4u && !memcmp(start, "null", 4u)) {
            parser->cursor += 4;
            value = calloc(1u, sizeof(*value));
            if (value) value->type = JSON_NULL;
        } else if ((size_t)(parser->end - start) >= 4u && !memcmp(start, "true", 4u)) {
            parser->cursor += 4;
            value = calloc(1u, sizeof(*value));
            if (value) {
                value->type = JSON_BOOL;
                value->as.boolean = true;
            }
        } else if ((size_t)(parser->end - start) >= 5u && !memcmp(start, "false", 5u)) {
            parser->cursor += 5;
            value = calloc(1u, sizeof(*value));
            if (value) value->type = JSON_BOOL;
        } else if (*start == '-' || isdigit((unsigned char)*start))
            value = json_parse_number(parser);
        if (!value && parser->err->code == FG_OK)
            fg_error_set(parser->err, FG_ERR_FORMAT, "invalid JSON value");
    }
    parser->depth--;
    return value;
}

static json_value *json_object_get(const json_value *object, const char *name) {
    if (!object || object->type != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < object->as.object.count; i++)
        if (!strcmp(object->as.object.members[i].name, name))
            return object->as.object.members[i].value;
    return NULL;
}

static bool json_object_has(const json_value *object, const char *name) {
    return json_object_get(object, name) != NULL;
}

static json_value *parse_json_body(const char *body, size_t length, fg_error *err) {
    json_parser parser = {.cursor = body, .end = body + length, .err = err};
    json_value *root = json_parse_value(&parser);
    json_skip_space(&parser);
    if (root && parser.cursor != parser.end) {
        fg_error_set(err, FG_ERR_FORMAT, "trailing text after JSON request");
        json_free(root);
        return NULL;
    }
    return root;
}

static fg_status json_serialize_value(api_buffer *buffer, const json_value *value,
                                      fg_error *err) {
    if (!value) {
        fg_error_set(err, FG_ERR_FORMAT, "cannot serialize missing JSON value");
        return FG_ERR_FORMAT;
    }
    fg_status status = FG_OK;
    switch (value->type) {
        case JSON_NULL:
            return buffer_append(buffer, "null", err);
        case JSON_BOOL:
            return buffer_append(buffer, value->as.boolean ? "true" : "false", err);
        case JSON_NUMBER: {
            char number[64];
            int length = snprintf(number, sizeof(number), "%.17g", value->as.number);
            if (length < 0 || (size_t)length >= sizeof(number)) {
                fg_error_set(err, FG_ERR_LIMIT, "JSON number serialization overflow");
                return FG_ERR_LIMIT;
            }
            return buffer_append_n(buffer, number, (size_t)length, err);
        }
        case JSON_STRING:
            return buffer_append_json_string(buffer, value->as.string,
                                             strlen(value->as.string), err);
        case JSON_ARRAY:
            status = buffer_append(buffer, "[", err);
            for (size_t i = 0; status == FG_OK && i < value->as.array.count; i++) {
                if (i) status = buffer_append(buffer, ",", err);
                if (status == FG_OK)
                    status = json_serialize_value(buffer, value->as.array.items[i], err);
            }
            if (status == FG_OK) status = buffer_append(buffer, "]", err);
            return status;
        case JSON_OBJECT:
            status = buffer_append(buffer, "{", err);
            for (size_t i = 0; status == FG_OK && i < value->as.object.count; i++) {
                if (i) status = buffer_append(buffer, ",", err);
                if (status == FG_OK)
                    status = buffer_append_json_string(
                        buffer, value->as.object.members[i].name,
                        strlen(value->as.object.members[i].name), err);
                if (status == FG_OK) status = buffer_append(buffer, ":", err);
                if (status == FG_OK)
                    status =
                        json_serialize_value(buffer, value->as.object.members[i].value, err);
            }
            if (status == FG_OK) status = buffer_append(buffer, "}", err);
            return status;
    }
    fg_error_set(err, FG_ERR_FORMAT, "unknown JSON value type");
    return FG_ERR_FORMAT;
}

static bool valid_function_name(const char *name) {
    if (!name || !name[0] || strlen(name) > 128u) return false;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++)
        if (!isalnum(*p) && *p != '_' && *p != '-' && *p != '.') return false;
    return true;
}

static bool request_has_tool_name(const api_chat_request *request, const char *name) {
    for (size_t i = 0; i < request->tool_schema_count; i++) {
        fg_error ignored = {0};
        json_value *schema =
            parse_json_body(request->tool_schemas[i], strlen(request->tool_schemas[i]),
                            &ignored);
        json_value *schema_name = json_object_get(schema, "name");
        bool match = schema_name && schema_name->type == JSON_STRING &&
                     !strcmp(schema_name->as.string, name);
        json_free(schema);
        if (match) return true;
    }
    return false;
}

static void api_chat_request_free(api_chat_request *request) {
    for (size_t i = 0; i < request->message_count; i++) {
        free((char *)request->messages[i].role);
        free((char *)request->messages[i].content);
        free((char *)request->messages[i].reasoning);
        free((char *)request->messages[i].tool_call_id);
        for (size_t j = 0; j < request->messages[i].tool_call_count; j++) {
            free((char *)request->messages[i].tool_calls[j].id);
            free((char *)request->messages[i].tool_calls[j].name);
            free((char *)request->messages[i].tool_calls[j].arguments_json);
        }
        free((void *)request->messages[i].tool_calls);
    }
    free(request->messages);
    for (size_t i = 0; i < request->tool_schema_count; i++) free(request->tool_schemas[i]);
    free(request->tool_schemas);
    free(request->tool_choice_name);
    memset(request, 0, sizeof(*request));
}

static bool number_is_integer(double value) {
    return value >= 0.0 && value <= (double)UINT32_MAX && value == (double)(uint32_t)value;
}

static fg_status reject_control(const json_value *root, const char *name, double allowed,
                                fg_error *err) {
    json_value *value = json_object_get(root, name);
    if (!value) return FG_OK;
    if (value->type != JSON_NUMBER || value->as.number != allowed) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "%s is unsupported; Flash Gordon currently serves greedy decoding only",
                     name);
        return FG_ERR_ARGUMENT;
    }
    return FG_OK;
}

static fg_status parse_tools(const json_value *root, api_chat_request *request,
                             fg_error *err) {
    json_value *tools = json_object_get(root, "tools");
    if (!tools || tools->type == JSON_NULL) return FG_OK;
    if (tools->type != JSON_ARRAY) {
        fg_error_set(err, FG_ERR_ARGUMENT, "tools must be an array");
        return FG_ERR_ARGUMENT;
    }
    if (!tools->as.array.count) return FG_OK;
    request->tool_schemas = calloc(tools->as.array.count, sizeof(*request->tool_schemas));
    if (!request->tool_schemas) {
        fg_error_set(err, FG_ERR_OOM, "allocate tool schemas");
        return FG_ERR_OOM;
    }
    for (size_t i = 0; i < tools->as.array.count; i++) {
        json_value *tool = tools->as.array.items[i];
        json_value *type = json_object_get(tool, "type");
        json_value *function = json_object_get(tool, "function");
        json_value *name = json_object_get(function, "name");
        json_value *description = json_object_get(function, "description");
        json_value *parameters = json_object_get(function, "parameters");
        if (!tool || tool->type != JSON_OBJECT || !type || type->type != JSON_STRING ||
            strcmp(type->as.string, "function") || !function ||
            function->type != JSON_OBJECT || !name || name->type != JSON_STRING ||
            !valid_function_name(name->as.string) || !parameters ||
            parameters->type != JSON_OBJECT ||
            (description && description->type != JSON_STRING)) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "each tool must be type function with a valid name and object "
                         "parameters");
            return FG_ERR_ARGUMENT;
        }
        if (request_has_tool_name(request, name->as.string)) {
            fg_error_set(err, FG_ERR_ARGUMENT, "duplicate tool function '%s'",
                         name->as.string);
            return FG_ERR_ARGUMENT;
        }
        api_buffer schema = {0};
        fg_status status = json_serialize_value(&schema, function, err);
        if (status != FG_OK) {
            free(schema.data);
            return status;
        }
        request->tool_schemas[request->tool_schema_count++] = schema.data;
    }
    return FG_OK;
}

static fg_status parse_tool_choice(const json_value *root, api_chat_request *request,
                                   fg_error *err) {
    json_value *choice = json_object_get(root, "tool_choice");
    request->tool_choice = FG_CHAT_TOOL_AUTO;
    if (!choice || choice->type == JSON_NULL) return FG_OK;
    if (choice->type == JSON_STRING) {
        if (!strcmp(choice->as.string, "auto"))
            request->tool_choice = FG_CHAT_TOOL_AUTO;
        else if (!strcmp(choice->as.string, "none"))
            request->tool_choice = FG_CHAT_TOOL_NONE;
        else if (!strcmp(choice->as.string, "required"))
            request->tool_choice = FG_CHAT_TOOL_REQUIRED;
        else {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "tool_choice must be auto, none, required, or a named function");
            return FG_ERR_ARGUMENT;
        }
    } else if (choice->type == JSON_OBJECT) {
        json_value *type = json_object_get(choice, "type");
        json_value *function = json_object_get(choice, "function");
        json_value *name = json_object_get(function, "name");
        if (!type || type->type != JSON_STRING || strcmp(type->as.string, "function") ||
            !function || function->type != JSON_OBJECT || !name ||
            name->type != JSON_STRING || !valid_function_name(name->as.string)) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "named tool_choice must contain type function and function.name");
            return FG_ERR_ARGUMENT;
        }
        request->tool_choice = FG_CHAT_TOOL_NAMED;
        request->tool_choice_name = strdup(name->as.string);
        if (!request->tool_choice_name) {
            fg_error_set(err, FG_ERR_OOM, "copy named tool choice");
            return FG_ERR_OOM;
        }
    } else {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "tool_choice must be a string or named function object");
        return FG_ERR_ARGUMENT;
    }
    if ((request->tool_choice == FG_CHAT_TOOL_REQUIRED ||
         request->tool_choice == FG_CHAT_TOOL_NAMED) &&
        !request->tool_schema_count) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "tool_choice requires at least one declared tool");
        return FG_ERR_ARGUMENT;
    }
    if (request->tool_choice == FG_CHAT_TOOL_NAMED &&
        !request_has_tool_name(request, request->tool_choice_name)) {
        fg_error_set(err, FG_ERR_ARGUMENT, "tool_choice names undeclared function '%s'",
                     request->tool_choice_name);
        return FG_ERR_ARGUMENT;
    }
    return FG_OK;
}

static fg_status parse_input_tool_calls(const json_value *value, fg_chat_message *message,
                                        fg_error *err) {
    if (!value || value->type == JSON_NULL) return FG_OK;
    if (value->type != JSON_ARRAY) {
        fg_error_set(err, FG_ERR_ARGUMENT, "assistant tool_calls must be an array or null");
        return FG_ERR_ARGUMENT;
    }
    if (!value->as.array.count) return FG_OK;
    fg_chat_tool_call *calls = calloc(value->as.array.count, sizeof(*calls));
    if (!calls) {
        fg_error_set(err, FG_ERR_OOM, "allocate assistant tool calls");
        return FG_ERR_OOM;
    }
    message->tool_calls = calls;
    for (size_t i = 0; i < value->as.array.count; i++) {
        json_value *call = value->as.array.items[i];
        json_value *id = json_object_get(call, "id");
        json_value *type = json_object_get(call, "type");
        json_value *function = json_object_get(call, "function");
        json_value *name = json_object_get(function, "name");
        json_value *arguments = json_object_get(function, "arguments");
        if (!call || call->type != JSON_OBJECT || !id || id->type != JSON_STRING ||
            !id->as.string[0] || !type || type->type != JSON_STRING ||
            strcmp(type->as.string, "function") || !function ||
            function->type != JSON_OBJECT || !name || name->type != JSON_STRING ||
            !valid_function_name(name->as.string) || !arguments ||
            arguments->type != JSON_STRING) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "assistant tool_calls require id, type function, name, and string "
                         "arguments");
            return FG_ERR_ARGUMENT;
        }
        fg_error json_error = {0};
        json_value *arguments_object =
            parse_json_body(arguments->as.string, strlen(arguments->as.string), &json_error);
        if (!arguments_object || arguments_object->type != JSON_OBJECT) {
            json_free(arguments_object);
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "assistant tool call arguments must encode a JSON object");
            return FG_ERR_ARGUMENT;
        }
        json_free(arguments_object);
        calls[i].id = strdup(id->as.string);
        calls[i].name = strdup(name->as.string);
        calls[i].arguments_json = strdup(arguments->as.string);
        message->tool_call_count++;
        if (!calls[i].id || !calls[i].name || !calls[i].arguments_json) {
            fg_error_set(err, FG_ERR_OOM, "copy assistant tool call");
            return FG_ERR_OOM;
        }
    }
    return FG_OK;
}

static bool previous_call_id(const api_chat_request *request, size_t before,
                             const char *id) {
    for (size_t i = 0; i < before; i++)
        for (size_t j = 0; j < request->messages[i].tool_call_count; j++)
            if (!strcmp(request->messages[i].tool_calls[j].id, id)) return true;
    return false;
}

static bool previous_result_id(const api_chat_request *request, size_t before,
                               const char *id) {
    for (size_t i = 0; i < before; i++)
        if (request->messages[i].tool_call_id &&
            !strcmp(request->messages[i].tool_call_id, id))
            return true;
    return false;
}

static fg_status validate_tool_associations(const api_chat_request *request,
                                            fg_error *err) {
    for (size_t i = 0; i < request->message_count; i++) {
        const fg_chat_message *message = &request->messages[i];
        for (size_t j = 0; j < message->tool_call_count; j++) {
            const char *id = message->tool_calls[j].id;
            if (previous_call_id(request, i, id)) {
                fg_error_set(err, FG_ERR_ARGUMENT, "duplicate tool call id '%s'", id);
                return FG_ERR_ARGUMENT;
            }
            for (size_t k = 0; k < j; k++) {
                if (!strcmp(message->tool_calls[k].id, id)) {
                    fg_error_set(err, FG_ERR_ARGUMENT, "duplicate tool call id '%s'", id);
                    return FG_ERR_ARGUMENT;
                }
            }
        }
        if (message->tool_call_id) {
            if (!previous_call_id(request, i, message->tool_call_id)) {
                fg_error_set(err, FG_ERR_ARGUMENT,
                             "tool result references unknown prior tool call id '%s'",
                             message->tool_call_id);
                return FG_ERR_ARGUMENT;
            }
            if (previous_result_id(request, i, message->tool_call_id)) {
                fg_error_set(err, FG_ERR_ARGUMENT,
                             "duplicate tool result for tool call id '%s'",
                             message->tool_call_id);
                return FG_ERR_ARGUMENT;
            }
        }
    }
    return FG_OK;
}

static fg_status parse_chat_request(const json_value *root, const char *runtime_model,
                                    api_chat_request *request, fg_error *err) {
    memset(request, 0, sizeof(*request));
    request->max_tokens = FG_API_DEFAULT_MAX_TOKENS;
    if (!root || root->type != JSON_OBJECT) {
        fg_error_set(err, FG_ERR_ARGUMENT, "request body must be a JSON object");
        return FG_ERR_ARGUMENT;
    }
    json_value *model = json_object_get(root, "model");
    if (model && (model->type != JSON_STRING || strcmp(model->as.string, runtime_model))) {
        fg_error_set(err, FG_ERR_ARGUMENT, "model must match the loaded model '%s'",
                     runtime_model);
        return FG_ERR_ARGUMENT;
    }
    fg_status status = parse_tools(root, request, err);
    if (status == FG_OK) status = parse_tool_choice(root, request, err);
    if (status != FG_OK) return status;
    static const char *unsupported[] = {
        "top_k", "min_p", "typical_p", "seed", "logit_bias",
    };
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
        if (json_object_has(root, unsupported[i])) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "%s is unsupported; Flash Gordon currently serves greedy decoding only",
                         unsupported[i]);
            return FG_ERR_ARGUMENT;
        }
    }
    status = reject_control(root, "temperature", 0.0, err);
    if (status == FG_OK) status = reject_control(root, "top_p", 1.0, err);
    if (status == FG_OK) status = reject_control(root, "presence_penalty", 0.0, err);
    if (status == FG_OK) status = reject_control(root, "frequency_penalty", 0.0, err);
    if (status == FG_OK) status = reject_control(root, "n", 1.0, err);
    if (status != FG_OK) return status;
    if (json_object_has(root, "stop") || json_object_has(root, "logprobs")) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "custom stop sequences and logprobs are not supported");
        return FG_ERR_ARGUMENT;
    }

    json_value *max_tokens = json_object_get(root, "max_tokens");
    json_value *max_completion = json_object_get(root, "max_completion_tokens");
    if (max_tokens && (max_tokens->type != JSON_NUMBER ||
                       !number_is_integer(max_tokens->as.number) ||
                       max_tokens->as.number < 1.0)) {
        fg_error_set(err, FG_ERR_ARGUMENT, "max_tokens must be a positive integer");
        return FG_ERR_ARGUMENT;
    }
    if (max_completion && (max_completion->type != JSON_NUMBER ||
                           !number_is_integer(max_completion->as.number) ||
                           max_completion->as.number < 1.0)) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "max_completion_tokens must be a positive integer");
        return FG_ERR_ARGUMENT;
    }
    if (max_tokens && max_completion &&
        max_tokens->as.number != max_completion->as.number) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "max_tokens and max_completion_tokens disagree");
        return FG_ERR_ARGUMENT;
    }
    if (max_completion) request->max_tokens = (uint32_t)max_completion->as.number;
    else if (max_tokens) request->max_tokens = (uint32_t)max_tokens->as.number;
    if (request->max_tokens > 4096u) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "max_tokens cannot exceed the runtime limit of 4096");
        return FG_ERR_ARGUMENT;
    }

    json_value *stream = json_object_get(root, "stream");
    if (stream) {
        if (stream->type != JSON_BOOL) {
            fg_error_set(err, FG_ERR_ARGUMENT, "stream must be a boolean");
            return FG_ERR_ARGUMENT;
        }
        request->stream = stream->as.boolean;
    }

    json_value *messages = json_object_get(root, "messages");
    if (!messages || messages->type != JSON_ARRAY || !messages->as.array.count) {
        fg_error_set(err, FG_ERR_ARGUMENT, "messages must be a non-empty array");
        return FG_ERR_ARGUMENT;
    }
    request->messages = calloc(messages->as.array.count, sizeof(*request->messages));
    if (!request->messages) {
        fg_error_set(err, FG_ERR_OOM, "allocate API chat messages");
        return FG_ERR_OOM;
    }
    for (size_t i = 0; i < messages->as.array.count; i++) {
        json_value *message = messages->as.array.items[i];
        json_value *role = json_object_get(message, "role");
        json_value *content = json_object_get(message, "content");
        json_value *reasoning = json_object_get(message, "reasoning_content");
        json_value *tool_calls = json_object_get(message, "tool_calls");
        json_value *tool_call_id = json_object_get(message, "tool_call_id");
        if (!message || message->type != JSON_OBJECT || !role ||
            role->type != JSON_STRING ||
            (content && content->type != JSON_STRING && content->type != JSON_NULL) ||
            (reasoning && reasoning->type != JSON_STRING &&
             reasoning->type != JSON_NULL)) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "each message requires string role and optional string or null content");
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        const char *accepted_role =
            (!strcmp(role->as.string, "system") || !strcmp(role->as.string, "developer") ||
             !strcmp(role->as.string, "user") || !strcmp(role->as.string, "assistant") ||
             !strcmp(role->as.string, "tool") || !strcmp(role->as.string, "function"))
                ? role->as.string
                : NULL;
        if (!accepted_role) {
            fg_error_set(err, FG_ERR_ARGUMENT, "unsupported message role '%s'",
                         role->as.string);
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        if ((!content || content->type == JSON_NULL) && strcmp(accepted_role, "assistant")) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "non-assistant messages require string content");
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        if (tool_calls && strcmp(accepted_role, "assistant")) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "tool_calls are only valid on assistant messages");
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        if (reasoning && strcmp(accepted_role, "assistant")) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "reasoning_content is only valid on assistant messages");
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        if (tool_call_id &&
            (strcmp(accepted_role, "tool") || tool_call_id->type != JSON_STRING ||
             !tool_call_id->as.string[0])) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "tool_call_id must be a non-empty string on a tool message");
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        if (!strcmp(accepted_role, "tool") && !tool_call_id) {
            fg_error_set(err, FG_ERR_ARGUMENT, "tool messages require tool_call_id");
            api_chat_request_free(request);
            return FG_ERR_ARGUMENT;
        }
        request->messages[i].role = strdup(accepted_role);
        request->messages[i].content =
            strdup(content && content->type == JSON_STRING ? content->as.string : "");
        request->message_count++;
        if (!request->messages[i].role || !request->messages[i].content) {
            fg_error_set(err, FG_ERR_OOM, "copy API chat message");
            api_chat_request_free(request);
            return FG_ERR_OOM;
        }
        if (reasoning && reasoning->type == JSON_STRING) {
            request->messages[i].reasoning = strdup(reasoning->as.string);
            if (!request->messages[i].reasoning) {
                fg_error_set(err, FG_ERR_OOM, "copy assistant reasoning");
                api_chat_request_free(request);
                return FG_ERR_OOM;
            }
        }
        if (tool_call_id) {
            request->messages[i].tool_call_id = strdup(tool_call_id->as.string);
            if (!request->messages[i].tool_call_id) {
                fg_error_set(err, FG_ERR_OOM, "copy tool result id");
                api_chat_request_free(request);
                return FG_ERR_OOM;
            }
        }
        status = parse_input_tool_calls(tool_calls, &request->messages[i], err);
        if (status != FG_OK) {
            api_chat_request_free(request);
            return status;
        }
    }
    return validate_tool_associations(request, err);
}

static char *find_header_end(char *data, size_t length) {
    if (length < 4u) return NULL;
    for (size_t i = 0; i + 3u < length; i++)
        if (!memcmp(data + i, "\r\n\r\n", 4u)) return data + i;
    return NULL;
}

static bool header_name_equal(const char *line, size_t name_length, const char *name) {
    size_t wanted = strlen(name);
    if (name_length != wanted) return false;
    for (size_t i = 0; i < wanted; i++)
        if (tolower((unsigned char)line[i]) != tolower((unsigned char)name[i])) return false;
    return true;
}

static fg_status read_http_request(int fd, http_request *request, unsigned *http_status,
                                   fg_error *err) {
    memset(request, 0, sizeof(*request));
    *http_status = 400u;
    api_buffer input = {0};
    char *header_end = NULL;
    while (!header_end) {
        if (input.length == FG_API_MAX_REQUEST_BYTES) {
            *http_status = 413u;
            fg_error_set(err, FG_ERR_LIMIT, "HTTP request exceeds 1 MiB");
            free(input.data);
            return FG_ERR_LIMIT;
        }
        size_t chunk = FG_API_MAX_REQUEST_BYTES - input.length;
        if (chunk > 8192u) chunk = 8192u;
        fg_status status = buffer_reserve(&input, chunk, err);
        if (status != FG_OK) {
            free(input.data);
            return status;
        }
        ssize_t received = recv(fd, input.data + input.length, chunk, 0);
        if (received < 0) {
            if (errno == EINTR && !api_stop_requested) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                *http_status = 408u;
                fg_error_set(err, FG_ERR_IO, "HTTP request read timed out");
            } else if (errno == EINTR) {
                fg_error_set(err, FG_ERR_IO, "HTTP request interrupted by shutdown");
            } else {
                fg_error_set(err, FG_ERR_IO, "receive HTTP request: %s", strerror(errno));
            }
            free(input.data);
            return FG_ERR_IO;
        }
        if (!received) {
            fg_error_set(err, FG_ERR_FORMAT, "client closed before HTTP headers completed");
            free(input.data);
            return FG_ERR_FORMAT;
        }
        input.length += (size_t)received;
        input.data[input.length] = 0;
        header_end = find_header_end(input.data, input.length);
    }
    size_t header_bytes = (size_t)(header_end - input.data) + 4u;
    char *line_end = strstr(input.data, "\r\n");
    if (!line_end) {
        fg_error_set(err, FG_ERR_FORMAT, "invalid HTTP request line");
        free(input.data);
        return FG_ERR_FORMAT;
    }
    *line_end = 0;
    char version[16];
    if (sscanf(input.data, "%7s %255s %15s", request->method, request->path, version) != 3 ||
        strcmp(version, "HTTP/1.1")) {
        fg_error_set(err, FG_ERR_FORMAT, "expected an HTTP/1.1 request line");
        free(input.data);
        return FG_ERR_FORMAT;
    }
    size_t content_length = 0;
    bool have_length = false;
    bool json_content = false;
    for (char *line = line_end + 2; line < header_end;) {
        char *next = strstr(line, "\r\n");
        if (!next || next > header_end) break;
        char *colon = memchr(line, ':', (size_t)(next - line));
        if (!colon) {
            fg_error_set(err, FG_ERR_FORMAT, "invalid HTTP header");
            free(input.data);
            return FG_ERR_FORMAT;
        }
        const char *value = colon + 1;
        while (value < next && isspace((unsigned char)*value)) value++;
        size_t name_length = (size_t)(colon - line);
        if (header_name_equal(line, name_length, "Content-Length")) {
            char *end = NULL;
            errno = 0;
            unsigned long long parsed = strtoull(value, &end, 10);
            while (end < next && isspace((unsigned char)*end)) end++;
            if (errno == ERANGE || end != next || parsed > FG_API_MAX_REQUEST_BYTES) {
                *http_status = parsed > FG_API_MAX_REQUEST_BYTES ? 413u : 400u;
                fg_error_set(err, FG_ERR_LIMIT, "invalid or excessive Content-Length");
                free(input.data);
                return FG_ERR_LIMIT;
            }
            content_length = (size_t)parsed;
            have_length = true;
        } else if (header_name_equal(line, name_length, "Transfer-Encoding")) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "chunked request bodies are not supported; send Content-Length");
            free(input.data);
            return FG_ERR_ARGUMENT;
        } else if (header_name_equal(line, name_length, "Content-Type")) {
            size_t value_length = (size_t)(next - value);
            json_content = value_length >= 16u &&
                           !strncasecmp(value, "application/json", 16u);
        }
        line = next + 2;
    }
    bool body_required = !strcmp(request->method, "POST");
    if (body_required && !have_length) {
        *http_status = 411u;
        fg_error_set(err, FG_ERR_ARGUMENT, "POST requires Content-Length");
        free(input.data);
        return FG_ERR_ARGUMENT;
    }
    if (body_required && !json_content) {
        *http_status = 415u;
        fg_error_set(err, FG_ERR_ARGUMENT, "POST requires application/json");
        free(input.data);
        return FG_ERR_ARGUMENT;
    }
    if (header_bytes + content_length > FG_API_MAX_REQUEST_BYTES) {
        *http_status = 413u;
        fg_error_set(err, FG_ERR_LIMIT, "HTTP request exceeds 1 MiB");
        free(input.data);
        return FG_ERR_LIMIT;
    }
    while (input.length < header_bytes + content_length) {
        size_t remaining = header_bytes + content_length - input.length;
        fg_status status = buffer_reserve(&input, remaining, err);
        if (status != FG_OK) {
            free(input.data);
            return status;
        }
        ssize_t received = recv(fd, input.data + input.length, remaining, 0);
        if (received < 0) {
            if (errno == EINTR && !api_stop_requested) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                *http_status = 408u;
                fg_error_set(err, FG_ERR_IO, "HTTP body read timed out");
            } else if (errno == EINTR) {
                fg_error_set(err, FG_ERR_IO, "HTTP body interrupted by shutdown");
            } else {
                fg_error_set(err, FG_ERR_IO, "receive HTTP body: %s", strerror(errno));
            }
            free(input.data);
            return FG_ERR_IO;
        }
        if (!received) {
            fg_error_set(err, FG_ERR_FORMAT, "client closed before HTTP body completed");
            free(input.data);
            return FG_ERR_FORMAT;
        }
        input.length += (size_t)received;
        input.data[input.length] = 0;
    }
    request->body = malloc(content_length + 1u);
    if (!request->body) {
        fg_error_set(err, FG_ERR_OOM, "allocate HTTP body");
        free(input.data);
        return FG_ERR_OOM;
    }
    memcpy(request->body, input.data + header_bytes, content_length);
    request->body[content_length] = 0;
    request->body_length = content_length;
    free(input.data);
    return FG_OK;
}

static void api_signal_handler(int signal_number) {
    (void)signal_number;
    api_stop_requested = 1;
}

static bool api_interrupted(void *context) {
    (void)context;
    return api_stop_requested != 0;
}

static fg_status send_sse_headers(int fd, fg_error *err) {
    static const char headers[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n\r\n";
    return send_all(fd, headers, sizeof(headers) - 1u, err);
}

static fg_status send_content_delta(api_generation *generation,const char *text,size_t length,
                                    fg_error *err) {
    if (!length) return FG_OK;
    api_buffer event = {0};
    fg_status status = buffer_append(&event, "data: {\"id\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, generation->id, strlen(generation->id), err);
    if (status == FG_OK)
        status = buffer_append(&event,
                               ",\"object\":\"chat.completion.chunk\",\"created\":", err);
    if (status == FG_OK) {
        char number[32];
        snprintf(number, sizeof(number), "%lld", (long long)generation->created);
        status = buffer_append(&event, number, err);
    }
    if (status == FG_OK) status = buffer_append(&event, ",\"model\":", err);
    if (status == FG_OK)
        status =
            buffer_append_json_string(&event, generation->model, strlen(generation->model), err);
    if (status == FG_OK)
        status = buffer_append(
            &event, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":", err);
    if (status == FG_OK) status = buffer_append_json_string(&event, text, length, err);
    if (status == FG_OK)
        status = buffer_append(&event, "},\"finish_reason\":null}]}\n\n", err);
    if (status == FG_OK) {
        status = send_all(generation->fd, event.data, event.length, err);
        if (status != FG_OK) generation->client_failed = true;
    }
    free(event.data);
    return status;
}

static void tool_call_id(const api_generation *generation, size_t index, char id[128]) {
    snprintf(id, 128u, "call_%s_%zu", generation->id, index);
}

static fg_status validate_generated_call(const api_chat_request *request,
                                         const fg_chat_tool_call *call, fg_error *err) {
    if (!request->tool_schema_count || !request_has_tool_name(request, call->name)) {
        fg_error_set(err, FG_ERR_FORMAT,
                     "model generated undeclared tool function '%s'", call->name);
        return FG_ERR_FORMAT;
    }
    if (request->tool_choice == FG_CHAT_TOOL_NONE) {
        fg_error_set(err, FG_ERR_FORMAT,
                     "model generated a tool call despite tool_choice none");
        return FG_ERR_FORMAT;
    }
    if (request->tool_choice == FG_CHAT_TOOL_NAMED &&
        strcmp(request->tool_choice_name, call->name)) {
        fg_error_set(err, FG_ERR_FORMAT,
                     "model called '%s' instead of required function '%s'", call->name,
                     request->tool_choice_name);
        return FG_ERR_FORMAT;
    }
    return FG_OK;
}

static fg_status validate_generated_tools(const api_chat_request *request,
                                          const fg_chat_generated *generated,
                                          fg_error *err) {
    if ((request->tool_choice == FG_CHAT_TOOL_REQUIRED ||
         request->tool_choice == FG_CHAT_TOOL_NAMED) &&
        !generated->tool_call_count) {
        fg_error_set(err, FG_ERR_FORMAT,
                     "model did not produce the tool call required by tool_choice");
        return FG_ERR_FORMAT;
    }
    for (size_t i = 0; i < generated->tool_call_count; i++) {
        fg_status status =
            validate_generated_call(request, &generated->tool_calls[i], err);
        if (status != FG_OK) return status;
    }
    return FG_OK;
}

static fg_status send_tool_call_delta(api_generation *generation,
                                      const fg_chat_tool_call *call, size_t index,
                                      fg_error *err) {
    char call_id[128];
    tool_call_id(generation, index, call_id);
    api_buffer event = {0};
    fg_status status = buffer_append(&event, "data: {\"id\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, generation->id, strlen(generation->id), err);
    if (status == FG_OK)
        status = buffer_append(&event,
                               ",\"object\":\"chat.completion.chunk\",\"created\":", err);
    if (status == FG_OK) {
        char number[32];
        snprintf(number, sizeof(number), "%lld", (long long)generation->created);
        status = buffer_append(&event, number, err);
    }
    if (status == FG_OK) status = buffer_append(&event, ",\"model\":", err);
    if (status == FG_OK)
        status =
            buffer_append_json_string(&event, generation->model, strlen(generation->model), err);
    if (status == FG_OK)
        status = buffer_append(
            &event,
            ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":", err);
    if (status == FG_OK) {
        char number[32];
        snprintf(number, sizeof(number), "%zu", index);
        status = buffer_append(&event, number, err);
    }
    if (status == FG_OK) status = buffer_append(&event, ",\"id\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, call_id, strlen(call_id), err);
    if (status == FG_OK)
        status = buffer_append(
            &event, ",\"type\":\"function\",\"function\":{\"name\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, call->name, strlen(call->name), err);
    if (status == FG_OK) status = buffer_append(&event, ",\"arguments\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, call->arguments_json,
                                           strlen(call->arguments_json), err);
    if (status == FG_OK)
        status = buffer_append(
            &event, "}}]},\"finish_reason\":null}]}\n\n", err);
    if (status == FG_OK) {
        status = send_all(generation->fd, event.data, event.length, err);
        if (status != FG_OK) generation->client_failed = true;
    }
    free(event.data);
    return status;
}

static int utf8_unit(const unsigned char *text,size_t available,size_t *bytes) {
    unsigned char first=text[0];uint32_t code=0;size_t need=0;
    if(first<0x80u){*bytes=1u;return 1;}
    if(first>=0xc2u&&first<=0xdfu){need=2u;code=first&0x1fu;}
    else if(first>=0xe0u&&first<=0xefu){need=3u;code=first&0x0fu;}
    else if(first>=0xf0u&&first<=0xf4u){need=4u;code=first&7u;}
    else {*bytes=1u;return -1;}
    if(available<need){*bytes=need;return 0;}
    for(size_t i=1;i<need;i++){if((text[i]&0xc0u)!=0x80u){*bytes=1u;return -1;}code=(code<<6u)|(text[i]&0x3fu);}
    if((need==3u&&code<0x800u)||(need==4u&&code<0x10000u)||
       (code>=0xd800u&&code<=0xdfffu)||code>0x10ffffu){*bytes=1u;return -1;}
    *bytes=need;return 1;
}

static fg_status send_utf8_delta(api_generation *generation,const char *text,size_t length,
                                 fg_error *err) {
    char combined[4100];size_t total=generation->utf8_pending_length+length;
    if(total>sizeof(combined)){fg_error_set(err,FG_ERR_LIMIT,"streamed token exceeds UTF-8 buffer");return FG_ERR_LIMIT;}
    memcpy(combined,generation->utf8_pending,generation->utf8_pending_length);
    memcpy(combined+generation->utf8_pending_length,text,length);
    generation->utf8_pending_length=0;
    size_t offset=0,run=0;fg_status status=FG_OK;
    while(status==FG_OK&&offset<total){
        size_t unit=0;int valid=utf8_unit((const unsigned char *)combined+offset,total-offset,&unit);
        if(valid>0){offset+=unit;continue;}
        if(offset>run)status=send_content_delta(generation,combined+run,offset-run,err);
        if(status!=FG_OK)break;
        if(valid==0){generation->utf8_pending_length=total-offset;memcpy(generation->utf8_pending,combined+offset,generation->utf8_pending_length);return FG_OK;}
        status=send_content_delta(generation,"\xef\xbf\xbd",3u,err);offset++;run=offset;
    }
    if(status==FG_OK&&offset>run)status=send_content_delta(generation,combined+run,offset-run,err);
    return status;
}

static void pending_consume(api_buffer *pending, size_t count) {
    if (count >= pending->length) {
        pending->length = 0;
        if (pending->data) pending->data[0] = 0;
        return;
    }
    memmove(pending->data, pending->data + count, pending->length - count);
    pending->length -= count;
    pending->data[pending->length] = 0;
}

static size_t marker_suffix(const char *text, size_t length, const char *marker) {
    size_t marker_length = strlen(marker);
    size_t maximum = marker_length - 1u;
    if (maximum > length) maximum = length;
    for (size_t candidate = maximum; candidate > 0; candidate--)
        if (!memcmp(text + length - candidate, marker, candidate)) return candidate;
    return 0;
}

static size_t tool_syntax_suffix(const char *text, size_t length) {
    static const char *markers[] = {
        "<tool_call>",  "</tool_call>", "<function=",  "</function>",
        "<parameter=", "</parameter>",
    };
    size_t keep = 0;
    for (size_t i = 0; i < sizeof(markers) / sizeof(markers[0]); i++) {
        size_t candidate = marker_suffix(text, length, markers[i]);
        if (candidate > keep) keep = candidate;
    }
    return keep;
}

static bool range_has_tool_syntax(const char *text, size_t length) {
    static const char *markers[] = {
        "<tool_call", "</tool_call", "<function",  "</function",
        "<parameter", "</parameter",
    };
    const char *end = text + length;
    for (size_t i = 0; i < sizeof(markers) / sizeof(markers[0]); i++) {
        const char *found = strstr(text, markers[i]);
        if (found && found < end) return true;
    }
    return false;
}

static fg_status queue_visible_content(api_generation *generation,const char *text,size_t length,
                                      bool final,fg_error *err) {
    static const char end_marker[]="<|im_end|>";
    static const char tool_start[]="<tool_call>";
    static const char tool_end[]="</tool_call>";
    if(generation->output_stopped)return FG_OK;
    fg_status status=FG_OK;
    if(length)status=buffer_append_n(&generation->visible_pending,text,length,err);
    if(status!=FG_OK)return status;
    while(status==FG_OK&&generation->visible_pending.length){
        char *tool=strstr(generation->visible_pending.data,tool_start);
        char *end=strstr(generation->visible_pending.data,end_marker);
        size_t boundary=generation->visible_pending.length;
        if(tool&&(size_t)(tool-generation->visible_pending.data)<boundary)
            boundary=(size_t)(tool-generation->visible_pending.data);
        if(end&&(size_t)(end-generation->visible_pending.data)<boundary)
            boundary=(size_t)(end-generation->visible_pending.data);
        size_t boundary_suffix =
            tool_syntax_suffix(generation->visible_pending.data,boundary);
        if(range_has_tool_syntax(generation->visible_pending.data,
                                 boundary-boundary_suffix)){
            fg_error_set(err,FG_ERR_FORMAT,"malformed generated Qwen tool syntax");
            return FG_ERR_FORMAT;
        }
        if(generation->streamed_tool_calls){
            size_t prefix=0;
            while(prefix<generation->visible_pending.length&&
                  isspace((unsigned char)generation->visible_pending.data[prefix]))
                prefix++;
            if(prefix==generation->visible_pending.length){
                if(final)pending_consume(&generation->visible_pending,prefix);
                break;
            }
            if(end==generation->visible_pending.data+prefix){
                generation->visible_pending.length=0;
                generation->visible_pending.data[0]=0;
                generation->output_stopped=true;
                break;
            }
            if(tool!=generation->visible_pending.data+prefix){
                size_t remaining=generation->visible_pending.length-prefix;
                size_t partial=tool_syntax_suffix(
                    generation->visible_pending.data+prefix,remaining);
                if(!final&&partial==remaining)break;
                fg_error_set(err,FG_ERR_FORMAT,
                             "generated text after a Qwen tool call is not supported");
                return FG_ERR_FORMAT;
            }
            if(prefix)pending_consume(&generation->visible_pending,prefix);
            tool=generation->visible_pending.data;
            end=strstr(generation->visible_pending.data,end_marker);
        }
        if(end&&(!tool||end<tool)){
            size_t bytes=(size_t)(end-generation->visible_pending.data);
            if(bytes)status=send_utf8_delta(generation,generation->visible_pending.data,bytes,err);
            generation->visible_pending.length=0;
            generation->visible_pending.data[0]=0;
            generation->output_stopped=true;
            break;
        }
        if(tool){
            size_t prefix=(size_t)(tool-generation->visible_pending.data);
            size_t visible_prefix=prefix;
            while(visible_prefix&&isspace(
                     (unsigned char)generation->visible_pending.data[visible_prefix-1u]))
                visible_prefix--;
            if(visible_prefix)
                status=send_utf8_delta(generation,generation->visible_pending.data,
                                      visible_prefix,err);
            if(status!=FG_OK)break;
            pending_consume(&generation->visible_pending,prefix);
            char *close=strstr(generation->visible_pending.data,tool_end);
            if(!close){
                if(final){
                   fg_error_set(err,FG_ERR_FORMAT,"unterminated generated Qwen tool call");
                   return FG_ERR_FORMAT;
                }
                break;
            }
            size_t block_length=(size_t)(close-generation->visible_pending.data)+
                               sizeof(tool_end)-1u;
            char *block=malloc(block_length+1u);
            if(!block){
                fg_error_set(err,FG_ERR_OOM,"copy streamed Qwen tool call");
                return FG_ERR_OOM;
            }
            memcpy(block,generation->visible_pending.data,block_length);
            block[block_length]=0;
            fg_chat_generated parsed={0};
            status=fg_chat_parse_generated(block,false,&parsed,err);
            free(block);
            if(status==FG_OK&&
               (parsed.tool_call_count!=1u||parsed.content[0])){
                fg_error_set(err,FG_ERR_FORMAT,"invalid streamed Qwen tool call block");
                status=FG_ERR_FORMAT;
            }
            if(status==FG_OK)
                status=validate_generated_call(generation->request,
                                               &parsed.tool_calls[0],err);
            if(status==FG_OK)
                status=send_tool_call_delta(generation,&parsed.tool_calls[0],
                                           generation->streamed_tool_calls,err);
            if(status==FG_OK)generation->streamed_tool_calls++;
            fg_chat_generated_free(&parsed);
            if(status!=FG_OK)break;
            pending_consume(&generation->visible_pending,block_length);
            continue;
        }
        size_t keep_tool=marker_suffix(generation->visible_pending.data,
                                      generation->visible_pending.length,tool_start);
        size_t keep_syntax=tool_syntax_suffix(generation->visible_pending.data,
                                             generation->visible_pending.length);
        if(keep_syntax>keep_tool)keep_tool=keep_syntax;
        size_t keep_end=marker_suffix(generation->visible_pending.data,
                                     generation->visible_pending.length,end_marker);
        size_t keep=keep_tool>keep_end?keep_tool:keep_end;
        size_t keep_start=generation->visible_pending.length-keep;
        while(keep_start&&isspace(
                  (unsigned char)generation->visible_pending.data[keep_start-1u])){
            keep_start--;
            keep++;
        }
        if(final&&(range_has_tool_syntax(generation->visible_pending.data,
                                         generation->visible_pending.length)||keep_tool)){
            fg_error_set(err,FG_ERR_FORMAT,"malformed generated Qwen tool call marker");
            return FG_ERR_FORMAT;
        }
        if(final)keep=0;
        size_t flush=generation->visible_pending.length-keep;
        if(flush)status=send_utf8_delta(generation,generation->visible_pending.data,flush,err);
        if(status==FG_OK&&flush)pending_consume(&generation->visible_pending,flush);
        break;
    }
    return status;
}

static fg_status api_token(void *context, uint32_t token, const char *text, size_t length,
                           fg_error *err) {
    (void)token;
    api_generation *generation = context;
    size_t previous=generation->content.length;
    fg_status status=buffer_append_n(&generation->content,text,length,err);
    if(status!=FG_OK||!generation->stream)return status;
    const char *emit=NULL;size_t emit_length=0;
    if(generation->think_closed){
        emit=generation->content.data+previous;emit_length=length;
    }else{
        const char *close=strstr(generation->content.data,"</think>");
        if(!close)return FG_OK;
        emit=close+strlen("</think>");
        while(*emit=='\r'||*emit=='\n')emit++;
        emit_length=generation->content.length-(size_t)(emit-generation->content.data);
        generation->think_closed=true;
    }
    if(!emit_length)return FG_OK;
    return queue_visible_content(generation,emit,emit_length,false,err);
}

static fg_status send_stream_start(const api_generation *generation, fg_error *err) {
    api_buffer event = {0};
    fg_status status = buffer_append(&event, "data: {\"id\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, generation->id, strlen(generation->id), err);
    if (status == FG_OK)
        status = buffer_append(&event,
                               ",\"object\":\"chat.completion.chunk\",\"created\":", err);
    if (status == FG_OK) {
        char number[32];
        snprintf(number, sizeof(number), "%lld", (long long)generation->created);
        status = buffer_append(&event, number, err);
    }
    if (status == FG_OK) status = buffer_append(&event, ",\"model\":", err);
    if (status == FG_OK)
        status =
            buffer_append_json_string(&event, generation->model, strlen(generation->model), err);
    if (status == FG_OK)
        status = buffer_append(
            &event,
            ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
            "\"finish_reason\":null}]}\n\n",
            err);
    if (status == FG_OK) status = send_all(generation->fd, event.data, event.length, err);
    free(event.data);
    return status;
}

static fg_status send_stream_end(api_generation *generation,
                                 const fg_chat_generated *generated,const char *reason,
                                 fg_error *err) {
    for(size_t i=generation->streamed_tool_calls;i<generated->tool_call_count;i++){
        fg_status tool=send_tool_call_delta(generation,&generated->tool_calls[i],i,err);
        if(tool!=FG_OK)return tool;
        generation->streamed_tool_calls++;
    }
    fg_status visible=queue_visible_content(generation,NULL,0,true,err);
    if(visible!=FG_OK)return visible;
    if(generation->utf8_pending_length){
        generation->utf8_pending_length=0;
        fg_status pending=send_content_delta(generation,"\xef\xbf\xbd",3u,err);
        if(pending!=FG_OK)return pending;
    }
    api_buffer event = {0};
    fg_status status = buffer_append(&event, "data: {\"id\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, generation->id, strlen(generation->id), err);
    if (status == FG_OK)
        status = buffer_append(&event,
                               ",\"object\":\"chat.completion.chunk\",\"created\":", err);
    if (status == FG_OK) {
        char number[32];
        snprintf(number, sizeof(number), "%lld", (long long)generation->created);
        status = buffer_append(&event, number, err);
    }
    if (status == FG_OK) status = buffer_append(&event, ",\"model\":", err);
    if (status == FG_OK)
        status =
            buffer_append_json_string(&event, generation->model, strlen(generation->model), err);
    if (status == FG_OK)
        status = buffer_append(
            &event, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&event, reason, strlen(reason), err);
    if (status == FG_OK) status = buffer_append(&event, "}]}\n\ndata: [DONE]\n\n", err);
    if (status == FG_OK) status = send_all(generation->fd, event.data, event.length, err);
    free(event.data);
    return status;
}

static fg_status send_stream_error(api_generation *generation,const char *message,
                                   fg_error *err) {
    api_buffer event={0};
    fg_status status=buffer_append(&event,"data: {\"error\":{\"message\":",err);
    if(status==FG_OK)status=buffer_append_json_string(&event,message,strlen(message),err);
    if(status==FG_OK)status=buffer_append(&event,",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n",err);
    if(status==FG_OK)status=send_all(generation->fd,event.data,event.length,err);
    free(event.data);return status;
}

static fg_status send_completion(const api_generation *generation,
                                 const fg_chat_generated *generated,
                                 const fg_generation_stats *stats, const char *reason,
                                 fg_error *err) {
    api_buffer body = {0};
    fg_status status = buffer_append(&body, "{\"id\":", err);
    if (status == FG_OK)
        status = buffer_append_json_string(&body, generation->id, strlen(generation->id), err);
    if (status == FG_OK)
        status = buffer_append(&body, ",\"object\":\"chat.completion\",\"created\":", err);
    if (status == FG_OK) {
        char number[64];
        snprintf(number, sizeof(number), "%lld", (long long)generation->created);
        status = buffer_append(&body, number, err);
    }
    if (status == FG_OK) status = buffer_append(&body, ",\"model\":", err);
    if (status == FG_OK)
        status =
            buffer_append_json_string(&body, generation->model, strlen(generation->model), err);
    if (status == FG_OK)
        status = buffer_append(
            &body, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":",
            err);
    if (status == FG_OK && generated->tool_call_count && !generated->content[0])
        status = buffer_append(&body, "null", err);
    else if (status == FG_OK)
        status = buffer_append_json_string(&body, generated->content,
                                           strlen(generated->content), err);
    if (status == FG_OK && generated->tool_call_count) {
        status = buffer_append(&body, ",\"tool_calls\":[", err);
        for (size_t i = 0; status == FG_OK && i < generated->tool_call_count; i++) {
            char call_id[128];
            tool_call_id(generation, i, call_id);
            if (i) status = buffer_append(&body, ",", err);
            if (status == FG_OK) status = buffer_append(&body, "{\"id\":", err);
            if (status == FG_OK)
                status = buffer_append_json_string(&body, call_id, strlen(call_id), err);
            if (status == FG_OK)
                status = buffer_append(
                    &body, ",\"type\":\"function\",\"function\":{\"name\":", err);
            if (status == FG_OK)
                status = buffer_append_json_string(
                    &body, generated->tool_calls[i].name,
                    strlen(generated->tool_calls[i].name), err);
            if (status == FG_OK) status = buffer_append(&body, ",\"arguments\":", err);
            if (status == FG_OK)
                status = buffer_append_json_string(
                    &body, generated->tool_calls[i].arguments_json,
                    strlen(generated->tool_calls[i].arguments_json), err);
            if (status == FG_OK) status = buffer_append(&body, "}}", err);
        }
        if (status == FG_OK) status = buffer_append(&body, "]", err);
    }
    if (status == FG_OK) status = buffer_append(&body, "},\"finish_reason\":", err);
    if (status == FG_OK) status = buffer_append_json_string(&body, reason, strlen(reason), err);
    if (status == FG_OK) status = buffer_append(&body, "}],\"usage\":{", err);
    if (status == FG_OK) {
        char usage[192];
        snprintf(usage, sizeof(usage),
                 "\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u}}",
                 stats->prompt_tokens, stats->generated_tokens,
                 stats->prompt_tokens + stats->generated_tokens);
        status = buffer_append(&body, usage, err);
    }
    if (status == FG_OK) {
        double prefill_tps =
            stats->prefill_seconds > 0.0 ? stats->prompt_tokens / stats->prefill_seconds : 0.0;
        double decode_tps =
            stats->decode_seconds > 0.0 ? stats->generated_tokens / stats->decode_seconds : 0.0;
        char metrics[512];
        int metrics_length = snprintf(
            metrics, sizeof(metrics),
            "X-Flash-Gordon-Prompt-Tokens: %u\r\n"
            "X-Flash-Gordon-Completion-Tokens: %u\r\n"
            "X-Flash-Gordon-Context-Tokens: %u\r\n"
            "X-Flash-Gordon-Prefill-Seconds: %.9f\r\n"
            "X-Flash-Gordon-Prefill-TPS: %.6f\r\n"
            "X-Flash-Gordon-Decode-Seconds: %.9f\r\n"
            "X-Flash-Gordon-Decode-TPS: %.6f\r\n",
            stats->prompt_tokens, stats->generated_tokens, stats->context_tokens,
            stats->prefill_seconds, prefill_tps, stats->decode_seconds, decode_tps);
        if (metrics_length < 0 || (size_t)metrics_length >= sizeof(metrics)) {
            fg_error_set(err, FG_ERR_LIMIT, "API metrics headers exceed buffer");
            status = FG_ERR_LIMIT;
        } else {
            status = send_response_with_headers(generation->fd, 200u, "application/json",
                                                metrics, body.data, body.length, err);
        }
    }
    free(body.data);
    return status;
}

static fg_status handle_models(int fd, fg_runtime *runtime, fg_error *err) {
    const char *model = fg_runtime_model_name(runtime);
    api_buffer body = {0};
    fg_status status = buffer_append(&body, "{\"object\":\"list\",\"data\":[{\"id\":", err);
    if (status == FG_OK) status = buffer_append_json_string(&body, model, strlen(model), err);
    if (status == FG_OK)
        status = buffer_append(&body,
                               ",\"object\":\"model\",\"created\":0,\"owned_by\":"
                               "\"flash-gordon\"}]}",
                               err);
    if (status == FG_OK)
        status = send_response(fd, 200u, "application/json", body.data, body.length, err);
    free(body.data);
    return status;
}

static fg_status handle_chat_completions(int fd, fg_runtime *runtime,
                                         const http_request *http, fg_error *err) {
    json_value *root = parse_json_body(http->body, http->body_length, err);
    if (!root) {
        char message[sizeof(err->message)];
        snprintf(message, sizeof(message), "%s", err->message);
        fg_error send_err = {0};
        send_error_response(fd, 400u, message, &send_err);
        return FG_OK;
    }
    api_chat_request request = {0};
    fg_status status =
        parse_chat_request(root, fg_runtime_model_name(runtime), &request, err);
    json_free(root);
    if (status != FG_OK) {
        char message[sizeof(err->message)];
        snprintf(message, sizeof(message), "%s", err->message);
        fg_error send_err = {0};
        send_error_response(fd, 400u, message, &send_err);
        api_chat_request_free(&request);
        return FG_OK;
    }

    char *rendered = NULL;
    fg_chat_render_options render_options = {
        .tool_schemas = (const char *const *)request.tool_schemas,
        .tool_schema_count = request.tool_schema_count,
        .tool_choice = request.tool_choice,
        .tool_choice_name = request.tool_choice_name,
    };
    status = fg_chat_render(request.messages, request.message_count, &render_options,
                            &rendered, err);
    if (status == FG_OK) status = fg_runtime_reset(runtime, err);
    char id[96];
    snprintf(id, sizeof(id), "chatcmpl-fg-%lld-%llu", (long long)time(NULL),
             ++api_request_sequence);
    api_generation generation = {
        .fd = fd,
        .stream = request.stream,
        .id = id,
        .model = fg_runtime_model_name(runtime),
        .created = time(NULL),
        .request = &request,
    };
    bool stream_started=false;
    if (status == FG_OK && request.stream) {
        status = send_sse_headers(fd, err);
        if (status == FG_OK){stream_started=true;status = send_stream_start(&generation, err);}
        if (status != FG_OK) generation.client_failed = true;
    }
    fg_generation_stats stats = {0};
    if (status == FG_OK)
        status = fg_runtime_generate(runtime, rendered, request.max_tokens, api_token,
                                     &generation, api_interrupted, NULL, &stats, err);
    fg_chat_generated generated={0};
    if(status==FG_OK)
        status=fg_chat_parse_generated(generation.content.data?generation.content.data:"",
                                       true,&generated,err);
    if(status==FG_OK)status=validate_generated_tools(&request,&generated,err);
    if(status==FG_OK){
        double prefill_tps=stats.prefill_seconds>0.0?
            (double)stats.prompt_tokens/stats.prefill_seconds:0.0;
        double decode_tps=stats.decode_seconds>0.0?
            (double)stats.generated_tokens/stats.decode_seconds:0.0;
        fprintf(stderr,
                "request %s: prefill %u tokens %.2f tok/s, generation %u tokens "
                "%.2f tok/s, context %u/%u\n",
                id,stats.prompt_tokens,prefill_tps,stats.generated_tokens,decode_tps,
                stats.context_tokens,fg_runtime_context_limit(runtime));
    }
    const char *finish_reason = generated.tool_call_count ? "tool_calls" :
        (stats.generated_tokens >= request.max_tokens ? "length" : "stop");
    if (status == FG_OK) {
        if (request.stream)
            status = send_stream_end(&generation, &generated, finish_reason, err);
        else
            status = send_completion(&generation, &generated, &stats, finish_reason, err);
    } else {
        char message[sizeof(err->message)];
        snprintf(message, sizeof(message), "%s", err->message);
        fg_error send_err = {0};
        unsigned response_status =
            status == FG_ERR_ARGUMENT || status == FG_ERR_FORMAT || status == FG_ERR_LIMIT ?
                400u :
                500u;
        if(stream_started)send_stream_error(&generation,message,&send_err);
        else send_error_response(fd, response_status, message, &send_err);
    }
    free(generation.content.data);
    free(generation.visible_pending.data);
    fg_chat_generated_free(&generated);
    free(rendered);
    api_chat_request_free(&request);
    if (generation.client_failed ||
        (status == FG_ERR_IO && stats.prompt_tokens + stats.generated_tokens > 0u))
        return FG_OK;
    if (status == FG_ERR_ARGUMENT || status == FG_ERR_FORMAT || status == FG_ERR_LIMIT) {
        fg_error reset_error = {0};
        if (fg_runtime_reset(runtime, &reset_error) != FG_OK) {
            *err = reset_error;
            return reset_error.code;
        }
        return FG_OK;
    }
    return status;
}

static fg_status open_listener(const char *host, uint16_t port, int *listener, fg_error *err) {
    char service[16];
    snprintf(service, sizeof(service), "%u", port);
    struct addrinfo hints = {
        .ai_family = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM,
        .ai_flags = AI_PASSIVE,
    };
    struct addrinfo *addresses = NULL;
    int rc = getaddrinfo(host, service, &hints, &addresses);
    if (rc) {
        fg_error_set(err, FG_ERR_IO, "resolve API host '%s': %s", host, gai_strerror(rc));
        return FG_ERR_IO;
    }
    int fd = -1;
    for (struct addrinfo *address = addresses; address; address = address->ai_next) {
        fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) continue;
        int enabled = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        if (!bind(fd, address->ai_addr, address->ai_addrlen) && !listen(fd, 16)) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    if (fd < 0) {
        fg_error_set(err, FG_ERR_IO, "bind API listener on %s:%u: %s", host, port,
                     strerror(errno));
        return FG_ERR_IO;
    }
    *listener = fd;
    return FG_OK;
}

static fg_status configure_client_socket(int fd, fg_error *err) {
    struct timeval timeout = {.tv_sec = FG_API_IO_TIMEOUT_SECONDS, .tv_usec = 0};
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        fg_error_set(err, FG_ERR_IO, "configure API client timeout: %s", strerror(errno));
        return FG_ERR_IO;
    }
    return FG_OK;
}

fg_status fg_api_main_with_options(const char *manifest_path, const char *host,
                                   uint16_t port,
                                   const fg_runtime_options *runtime_options,
                                   fg_error *err) {
    if (!manifest_path || !host || !host[0] || !port) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "api requires manifest, non-empty host, and non-zero port");
        return FG_ERR_ARGUMENT;
    }
    fg_runtime *runtime = NULL;
    fg_status status =
        fg_runtime_open_with_options(&runtime, manifest_path, runtime_options, err);
    if (status != FG_OK) return status;
    int listener = -1;
    status = open_listener(host, port, &listener, err);
    if (status != FG_OK) {
        fg_runtime_close(runtime);
        return status;
    }

    struct sigaction action = {0}, old_int = {0}, old_term = {0}, ignore_pipe = {0},
                     old_pipe = {0};
    api_stop_requested = 0;
    action.sa_handler = api_signal_handler;
    sigemptyset(&action.sa_mask);
    ignore_pipe.sa_handler = SIG_IGN;
    sigaction(SIGINT, &action, &old_int);
    sigaction(SIGTERM, &action, &old_term);
    sigaction(SIGPIPE, &ignore_pipe, &old_pipe);
    fprintf(stderr, "Flash Gordon API serving %s on http://%s:%u\n",
            fg_runtime_model_name(runtime), host, port);

    while (status == FG_OK && !api_stop_requested) {
        struct pollfd ready = {.fd = listener, .events = POLLIN};
        int polled = poll(&ready, 1u, 1000);
        if (polled < 0) {
            if (errno == EINTR) continue;
            fg_error_set(err, FG_ERR_IO, "poll API listener: %s", strerror(errno));
            status = FG_ERR_IO;
            break;
        }
        if (!polled) continue;
        if (!(ready.revents & POLLIN)) {
            fg_error_set(err, FG_ERR_IO, "API listener reported events 0x%x",
                         ready.revents);
            status = FG_ERR_IO;
            break;
        }
        int client = accept(listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR && api_stop_requested) break;
            if (errno == EINTR) continue;
            fg_error_set(err, FG_ERR_IO, "accept API connection: %s", strerror(errno));
            status = FG_ERR_IO;
            break;
        }
        fg_error request_error = {0};
        if (configure_client_socket(client, &request_error) != FG_OK) {
            close(client);
            if (!api_stop_requested) {
                *err = request_error;
                status = request_error.code;
            }
            continue;
        }
        http_request request = {0};
        unsigned http_status = 400u;
        fg_status read_status =
            read_http_request(client, &request, &http_status, &request_error);
        if (read_status != FG_OK) {
            if (!api_stop_requested) {
                fg_error send_error = {0};
                send_error_response(client, http_status,
                                    request_error.message[0] ? request_error.message :
                                                               "invalid HTTP request",
                                    &send_error);
            }
        } else if (!strcmp(request.method, "GET") && !strcmp(request.path, "/v1/models")) {
            fg_status response_status = handle_models(client, runtime, err);
            if (response_status != FG_ERR_IO) status = response_status;
        } else if (!strcmp(request.method, "POST") &&
                   !strcmp(request.path, "/v1/chat/completions")) {
            status = handle_chat_completions(client, runtime, &request, err);
        } else if (!strcmp(request.path, "/v1/models") ||
                   !strcmp(request.path, "/v1/chat/completions")) {
            fg_error send_error = {0};
            send_error_response(client, 405u, "method not allowed", &send_error);
        } else {
            fg_error send_error = {0};
            send_error_response(client, 404u, "not found", &send_error);
        }
        free(request.body);
        close(client);
    }
    close(listener);
    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGTERM, &old_term, NULL);
    sigaction(SIGPIPE, &old_pipe, NULL);
    fg_runtime_close(runtime);
    return status;
}

fg_status fg_api_main(const char *manifest_path, const char *host, uint16_t port,
                      fg_error *err) {
    return fg_api_main_with_options(manifest_path, host, port, NULL, err);
}
