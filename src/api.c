#include "fg_api.h"
#include "fg_chat.h"
#include "fg_runtime.h"
#include "fg_sha256.h"
#include "fg_video.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define FG_API_MAX_REQUEST_BYTES (32u * 1024u * 1024u)
#define FG_API_MAX_OBJECT_MEMBERS 256u
#define FG_API_IO_TIMEOUT_SECONDS 30
#define FG_API_DEFAULT_MAX_TOKENS 512u
#define FG_API_STREAM_KEEPALIVE_SECONDS 10.0
#define FG_API_CONTENT_SNIPPET 48u
/* Front-end bounds.  The front-end thread owns the listener and every client
 * socket; the engine only appends response bytes to a connection's outbound
 * buffer under out_mutex. */
#define FG_API_MAX_HEADER_BYTES (256u * 1024u)
#define FG_API_CONNECTION_OUTPUT_LIMIT (8u * 1024u * 1024u)
#define FG_API_CONNECTION_IDLE_SECONDS 120.0
#define FG_API_MAX_CONNECTIONS 64u
#define FG_API_ENGINE_QUEUE_CAPACITY 4u
#define FG_API_FRONTEND_POLL_MS 200
/* M3 live-session table: at most one engine request is served at a time today
 * (M2 bound 4), so 8 entries are ample; the LRU entry is evicted when full. */
#define FG_API_SESSION_MAX FG_RUNTIME_SESSION_MAX

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

typedef struct api_media {
    uint32_t kind;
    uint8_t *data;
    size_t length;
    uint8_t **frames;
    size_t *frame_lengths;
    size_t frame_count;
    double fps;
    uint32_t max_frames;
    /* SHA-256 over the decoded payload (kind, length and bytes; frames are
     * framed individually).  Two media items with the same marker text but
     * different bytes must never compare equal in the session prefix. */
    uint8_t digest[32];
} api_media;

typedef struct api_chat_request {
    fg_chat_message *messages;
    size_t message_count;
    char **tool_schemas;
    size_t tool_schema_count;
    fg_chat_tool_choice tool_choice;
    char *tool_choice_name;
    uint32_t max_tokens;
    bool stream;
    fg_sampler_config sampler;
    api_media *media;
    size_t media_count;
    size_t media_capacity;
    /* M3 public session identity: an opaque client-supplied continuation key.
     * The server echoes it as X-Flash-Gordon-Session; an unknown id starts a
     * new session. */
    char session_id[48];
    bool session_id_set;
} api_chat_request;

typedef struct api_media_identity {
    uint32_t kind;
    uint8_t digest[32];
} api_media_identity;

typedef struct api_public_session {
    api_chat_request transcript;
    /* Per-media identities of the committed transcript, in message order. */
    api_media_identity *media;
    size_t media_count;
    bool valid;
    /* Public identity (M3): echoed in X-Flash-Gordon-Session so a client can
     * continue this transcript across connections.  `numeric_id` is the engine
     * side key for the runtime session object. */
    char id[48];
    uint64_t numeric_id;
} api_public_session;

/* Incremental HTTP/1.1 request parser.  The front-end thread feeds it bytes
 * from a client socket; it consumes exactly one complete request at a time and
 * leaves any pipelined remainder in `input`. */
typedef struct api_http_parser {
    api_buffer input;
} api_http_parser;

typedef enum api_parse_state {
    API_PARSE_INCOMPLETE = 0,
    API_PARSE_COMPLETE,
    API_PARSE_FAILED
} api_parse_state;

typedef struct api_connection api_connection;
typedef struct api_frontend api_frontend;

/* Response sink.  In the serving process every write goes to a front-end
 * connection; the raw-fd mode exists for direct unit tests and keeps the
 * historical `Connection: close` framing. */
typedef struct api_sink {
    int fd;
    api_connection *connection;
} api_sink;

struct api_connection {
    int fd;
    api_frontend *frontend;
    api_http_parser parser;
    /* Front-end-thread state. */
    bool generating;      /* a chat request is in flight on this connection */
    bool peer_closed;     /* client half-closed; flush then close */
    bool keep_alive;      /* request allows a persistent connection */
    double last_activity; /* monotonic seconds of the last read or write */
    /* Engine/front-end shared state.  out_mutex guards `out` and
     * `client_failed`; the flags are atomic so the engine can publish response
     * state without stalling the front-end poll loop. */
    pthread_mutex_t out_mutex;
    api_buffer out;       /* response bytes pending write (chunk-framed when
                           * `chunked` is set) */
    _Atomic bool chunked;         /* response body uses HTTP chunked framing */
    _Atomic bool response_complete;
    _Atomic bool close_after_flush;
    bool client_failed;   /* engine could not enqueue more response bytes */
    /* Written by the front-end, read by the engine's interrupt hook. */
    _Atomic bool client_gone;
    api_connection *next;
};

typedef struct api_generation {
    api_sink sink;
    bool stream;
    bool keep_alive;
    api_buffer content;
    const char *id;
    const char *model;
    time_t created;
    const api_chat_request *request;
    bool client_failed;
    bool think_closed;
    bool output_stopped;
    bool visible_started;
    api_buffer visible_pending;
    char utf8_pending[4];
    size_t utf8_pending_length;
    char utf8_reasoning_pending[4];
    size_t utf8_reasoning_pending_length;
    size_t reasoning_emitted;
    size_t streamed_tool_calls;
    fg_runtime *runtime;
    const char *session_id;
} api_generation;

typedef struct api_engine_request {
    api_connection *connection;
    http_request http;
} api_engine_request;

/* Bounded FIFO transport queue (M2 admission policy).  A chat request is
 * admitted while the engine is busy and waits here; the bound counts every
 * admitted request (running + queued + not yet completed), so at most
 * FG_API_ENGINE_QUEUE_CAPACITY requests are in flight and the next one gets
 * `429` + Retry-After instead of a token slot. */
typedef struct api_engine_queue {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    api_engine_request entries[FG_API_ENGINE_QUEUE_CAPACITY];
    size_t head;
    size_t count;
    size_t outstanding;
    bool stopping;
} api_engine_queue;

typedef enum api_engine_admission {
    API_ENGINE_ADMIT_QUEUED,   /* request accepted; ownership moved to the queue */
    API_ENGINE_ADMIT_FULL,     /* bound reached: 429 + Retry-After */
    API_ENGINE_ADMIT_STOPPING  /* shutting down: 503 */
} api_engine_admission;

/* M3 session table (engine thread only).  An entry owns the public transcript
 * (`api_public_session`) and the runtime token-path object for one live chat
 * session.  Requests resolve by explicit `session_id`, by their connection (the
 * keep-alive default), or by strict prefix extension of the most recent
 * session; a full table evicts the least-recently-used idle entry. */
typedef struct api_session_entry {
    api_public_session session;
    fg_runtime_session *runtime_session;
    api_connection *owner;
    uint64_t last_used;
    bool busy;
    bool in_use;
} api_session_entry;

typedef struct api_session_table {
    api_session_entry entries[FG_API_SESSION_MAX];
    uint64_t sequence;
} api_session_table;

struct api_frontend {
    int listener;
    int wake_read;
    int wake_write;
    pthread_t thread;
    bool thread_started;
    _Atomic bool stopping;
    fg_runtime *runtime;
    api_engine_queue *engine;
    api_connection *connections;
    size_t connection_count;
};

static volatile sig_atomic_t api_stop_requested;
static unsigned long long api_request_sequence;
static int api_wake_fd = -1;

static int utf8_unit(const unsigned char *text,size_t available,size_t *bytes);
static void tool_call_id(const api_generation *generation,size_t index,char output[128]);
static fg_status api_sink_write(api_sink *sink,const char *data,size_t length,fg_error *err);
static fg_status api_connection_enqueue_raw(api_connection *conn,const char *data,
                                            size_t length,fg_error *err);
static void api_connection_complete_response(api_connection *conn,bool close_after);
static void api_connection_flush(api_connection *conn);
static bool api_connection_maybe_heartbeat(api_connection *conn,double now);
static bool api_connection_detect_client_gone(api_connection *conn);
static bool api_session_id_valid(const char *id);
static fg_status api_session_table_resolve(api_session_table *table,fg_runtime *runtime,
                                           api_connection *conn,
                                           const api_chat_request *request,
                                           api_session_entry **entry_out,fg_error *err);
static void api_session_table_destroy(api_session_table *table,fg_runtime *runtime);
static void api_frontend_wake(api_frontend *frontend);
static void api_frontend_accept(api_frontend *frontend);
static void api_frontend_read_available(api_frontend *frontend,api_connection *conn);
static bool api_frontend_dispatch(api_frontend *frontend,api_connection *conn,
                                  http_request *http,bool keep_alive);
static fg_status configure_client_socket(int fd,fg_error *err);

static double api_monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}

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
        case 429: return "Too Many Requests";
        case 503: return "Service Unavailable";
        default: return "Internal Server Error";
    }
}

static fg_status api_send_response_with_headers(api_sink *sink, unsigned status,
                                                const char *content_type,
                                                const char *extra_headers, const char *body,
                                                size_t body_length, bool keep_alive,
                                                fg_error *err) {
    char header[4096];
    int length = snprintf(header, sizeof(header),
                          "HTTP/1.1 %u %s\r\n"
                          "Content-Type: %s\r\n"
                          "Content-Length: %zu\r\n"
                          "Connection: %s\r\n"
                          "Cache-Control: no-store\r\n"
                          "%s\r\n",
                          status, http_reason(status), content_type, body_length,
                          keep_alive ? "keep-alive" : "close",
                          extra_headers ? extra_headers : "");
    if (length < 0 || (size_t)length >= sizeof(header)) {
        fg_error_set(err, FG_ERR_LIMIT, "HTTP response header overflow");
        return FG_ERR_LIMIT;
    }
    fg_status result = api_sink_write(sink, header, (size_t)length, err);
    if (result == FG_OK) result = api_sink_write(sink, body, body_length, err);
    return result;
}

static fg_status api_send_response(api_sink *sink, unsigned status, const char *content_type,
                                   const char *body, size_t body_length, bool keep_alive,
                                   fg_error *err) {
    return api_send_response_with_headers(sink, status, content_type, NULL, body,
                                          body_length, keep_alive, err);
}

static fg_status api_send_error_response(api_sink *sink, unsigned status, const char *message,
                                         bool keep_alive, fg_error *err) {
    api_buffer body = {0};
    fg_status result = buffer_append(&body, "{\"error\":{\"message\":", err);
    if (result == FG_OK)
        result = buffer_append_json_string(&body, message, strlen(message), err);
    if (result == FG_OK)
        result = buffer_append(&body, ",\"type\":\"invalid_request_error\"}}", err);
    if (result == FG_OK)
        result = api_send_response(sink, status, "application/json", body.data, body.length,
                                   keep_alive, err);
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

static fg_status message_content_text(const json_value *content,const char **output,
                                      fg_error *err){
    if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"message content output is null");return FG_ERR_ARGUMENT;}
    *output=NULL;
    if(!content||content->type==JSON_NULL){
        *output=strdup("");
        if(!*output){fg_error_set(err,FG_ERR_OOM,"copy empty message content");return FG_ERR_OOM;}
        return FG_OK;
    }
    if(content->type==JSON_STRING){
        *output=strdup(content->as.string);
        if(!*output){fg_error_set(err,FG_ERR_OOM,"copy message content");return FG_ERR_OOM;}
        return FG_OK;
    }
    if(content->type!=JSON_ARRAY){
        fg_error_set(err,FG_ERR_ARGUMENT,"message content must be a string, array, or null");
        return FG_ERR_ARGUMENT;
    }
    api_buffer text={0};fg_status status=FG_OK;
    for(size_t i=0;status==FG_OK&&i<content->as.array.count;i++){
        json_value *part=content->as.array.items[i];
        if(part&&part->type==JSON_STRING){
            status=buffer_append(&text,part->as.string,err);
            continue;
        }
        json_value *type=json_object_get(part,"type");
        json_value *value=json_object_get(part,"text");
        if(!part||part->type!=JSON_OBJECT||!type||type->type!=JSON_STRING||
           (strcmp(type->as.string,"text")&&strcmp(type->as.string,"input_text"))||
           !value||value->type!=JSON_STRING){
            if(type&&type->type==JSON_STRING)
                fg_error_set(err,FG_ERR_ARGUMENT,"message content part type '%s' is unsupported",type->as.string);
            else fg_error_set(err,FG_ERR_ARGUMENT,"message content array requires text parts");
            status=FG_ERR_ARGUMENT;
            break;
        }
        status=buffer_append(&text,value->as.string,err);
    }
    if(status==FG_OK&&!text.data){
        text.data=strdup("");
        if(!text.data){fg_error_set(err,FG_ERR_OOM,"copy empty message content array");status=FG_ERR_OOM;}
    }
    if(status!=FG_OK){free(text.data);return status;}
    *output=text.data;return FG_OK;
}

static int api_base64_value(unsigned char character){
    if(character>='A'&&character<='Z')return character-'A';
    if(character>='a'&&character<='z')return character-'a'+26;
    if(character>='0'&&character<='9')return character-'0'+52;
    if(character=='+')return 62;
    if(character=='/')return 63;
    return -1;
}

static fg_status api_base64_decode(const char *text,size_t length,uint8_t **output,
                                   size_t *output_length,fg_error *err){
    if(!text||!output||!output_length||length==0||length%4u){
        fg_error_set(err,FG_ERR_ARGUMENT,"image payload is not valid base64");
        return FG_ERR_ARGUMENT;
    }
    uint8_t *decoded=malloc(length/4u*3u);
    if(!decoded){
        fg_error_set(err,FG_ERR_OOM,"allocate decoded image payload");
        return FG_ERR_OOM;
    }
    size_t produced=0;
    for(size_t i=0;i<length;i+=4u){
        const int a=api_base64_value((unsigned char)text[i]);
        const int b=api_base64_value((unsigned char)text[i+1u]);
        const bool pad_c=text[i+2u]=='=';
        const bool pad_d=text[i+3u]=='=';
        const int c=pad_c?-2:api_base64_value((unsigned char)text[i+2u]);
        const int d=pad_d?-2:api_base64_value((unsigned char)text[i+3u]);
        if(a<0||b<0||c==-1||d==-1||(pad_c&&(!pad_d||i+4u!=length))||
           (!pad_c&&pad_d&&i+4u!=length)){
            free(decoded);
            fg_error_set(err,FG_ERR_ARGUMENT,"image payload is not valid base64");
            return FG_ERR_ARGUMENT;
        }
        decoded[produced++]=(uint8_t)((a<<2)|(b>>4));
        if(!pad_c)decoded[produced++]=(uint8_t)(((b&15)<<4)|(c>>2));
        if(!pad_d)decoded[produced++]=(uint8_t)(((c&3)<<6)|d);
    }
    *output=decoded;
    *output_length=produced;
    return FG_OK;
}

static api_media *api_request_media_slot(api_chat_request *request,fg_error *err){
    if(request->media_count==request->media_capacity){
        const size_t capacity=request->media_capacity?request->media_capacity*2u:2u;
        api_media *grown=realloc(request->media,capacity*sizeof(*grown));
        if(!grown){
            fg_error_set(err,FG_ERR_OOM,"grow API media list");
            return NULL;
        }
        request->media=grown;
        request->media_capacity=capacity;
    }
    api_media *slot=&request->media[request->media_count++];
    memset(slot,0,sizeof(*slot));
    return slot;
}

static void api_digest_u32(fg_sha256 *hash,uint32_t value){
    uint8_t bytes[4]={(uint8_t)value,(uint8_t)(value>>8u),(uint8_t)(value>>16u),
                      (uint8_t)(value>>24u)};
    fg_sha256_update(hash,bytes,sizeof(bytes));
}

static void api_digest_u64(fg_sha256 *hash,uint64_t value){
    uint8_t bytes[8];
    for(unsigned i=0;i<8u;i++)bytes[i]=(uint8_t)(value>>(8u*i));
    fg_sha256_update(hash,bytes,sizeof(bytes));
}

static void api_media_compute_digest(api_media *media){
    fg_sha256 hash;
    fg_sha256_init(&hash);
    api_digest_u32(&hash,media->kind);
    if(media->kind==FG_RUNTIME_MEDIA_VIDEO_FRAMES){
        api_digest_u64(&hash,(uint64_t)media->frame_count);
        api_digest_u64(&hash,(uint64_t)(media->fps*1000.0));
        api_digest_u32(&hash,media->max_frames);
        for(size_t frame=0;frame<media->frame_count;frame++){
            api_digest_u64(&hash,(uint64_t)media->frame_lengths[frame]);
            fg_sha256_update(&hash,media->frames[frame],media->frame_lengths[frame]);
        }
    }else{
        api_digest_u64(&hash,(uint64_t)media->length);
        fg_sha256_update(&hash,media->data,media->length);
    }
    fg_sha256_final(&hash,media->digest);
}

static fg_status api_request_add_media(api_chat_request *request,uint32_t kind,uint8_t *data,
                                       size_t length,fg_error *err){
    api_media *slot=api_request_media_slot(request,err);
    if(!slot){
        free(data);
        return FG_ERR_OOM;
    }
    slot->kind=kind;
    slot->data=data;
    slot->length=length;
    api_media_compute_digest(slot);
    return FG_OK;
}

static fg_status api_request_add_video_frames(api_chat_request *request,uint8_t **frames,
                                              size_t *frame_lengths,size_t frame_count,
                                              double fps,uint32_t max_frames,fg_error *err){
    api_media *slot=api_request_media_slot(request,err);
    if(!slot)return FG_ERR_OOM;
    slot->kind=FG_RUNTIME_MEDIA_VIDEO_FRAMES;
    slot->frames=frames;
    slot->frame_lengths=frame_lengths;
    slot->frame_count=frame_count;
    slot->fps=fps;
    slot->max_frames=max_frames;
    api_media_compute_digest(slot);
    return FG_OK;
}

static fg_status api_decode_media_url(const char *url,uint32_t kind,const char *name,
                                      uint8_t **data,size_t *length,fg_error *err){
    const bool video=kind==FG_RUNTIME_MEDIA_VIDEO;
    if(!url||strncmp(url,"data:",5u)!=0){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "%s must be a base64 data: URL; http(s) %s URLs are not supported",
                     name,name);
        return FG_ERR_ARGUMENT;
    }
    const char *comma=strchr(url,',');
    if(!comma){
        fg_error_set(err,FG_ERR_ARGUMENT,"%s data URL has no payload",name);
        return FG_ERR_ARGUMENT;
    }
    const char *header=url+5u;
    const size_t header_length=(size_t)(comma-header);
    const char *semi=memchr(header,';',header_length);
    const size_t mime_length=semi?(size_t)(semi-header):header_length;
    char mime[64];
    if(!mime_length||mime_length>=sizeof(mime)){
        fg_error_set(err,FG_ERR_ARGUMENT,"%s data URL has an invalid media type",name);
        return FG_ERR_ARGUMENT;
    }
    memcpy(mime,header,mime_length);
    mime[mime_length]=0;
    if(video){
        if(strcmp(mime,"video/mp4")&&strcmp(mime,"video/webm")&&strcmp(mime,"video/x-matroska")){
            fg_error_set(err,FG_ERR_ARGUMENT,
                         "unsupported video media type '%s' "
                         "(use video/mp4, video/webm or video/x-matroska)",mime);
            return FG_ERR_ARGUMENT;
        }
    }else if(strcmp(mime,"image/png")&&strcmp(mime,"image/jpeg")&&strcmp(mime,"image/jpg")){
        fg_error_set(err,FG_ERR_ARGUMENT,
                     "unsupported image media type '%s' (use image/png or image/jpeg)",mime);
        return FG_ERR_ARGUMENT;
    }
    if(!semi||!strstr(semi,"base64")){
        fg_error_set(err,FG_ERR_ARGUMENT,"%s data URL must be base64 encoded",name);
        return FG_ERR_ARGUMENT;
    }
    const char *payload=comma+1u;
    return api_base64_decode(payload,strlen(payload),data,length,err);
}

static bool number_is_integer(double value);

static fg_status message_content_parts(const json_value *content,const char **output,
                                       api_chat_request *request,fg_error *err){
    if(!content||content->type==JSON_NULL||content->type==JSON_STRING||
       content->type!=JSON_ARRAY){
        return message_content_text(content,output,err);
    }
    bool has_media=false;
    for(size_t i=0;i<content->as.array.count;i++){
        json_value *part=content->as.array.items[i];
        json_value *type=json_object_get(part,"type");
        if(type&&type->type==JSON_STRING&&
           (!strcmp(type->as.string,"image_url")||!strcmp(type->as.string,"video_url")||
            !strcmp(type->as.string,"video_frames"))){
            has_media=true;
            break;
        }
    }
    if(!has_media)return message_content_text(content,output,err);
    api_buffer text={0};
    fg_status status=FG_OK;
    for(size_t i=0;status==FG_OK&&i<content->as.array.count;i++){
        json_value *part=content->as.array.items[i];
        json_value *type=json_object_get(part,"type");
        if(part&&part->type==JSON_STRING){
            status=buffer_append(&text,part->as.string,err);
            continue;
        }
        if(!part||part->type!=JSON_OBJECT||!type||type->type!=JSON_STRING){
            fg_error_set(err,FG_ERR_ARGUMENT,"message content array requires typed parts");
            status=FG_ERR_ARGUMENT;
            break;
        }
        if(!strcmp(type->as.string,"text")||!strcmp(type->as.string,"input_text")){
            json_value *value=json_object_get(part,"text");
            if(!value||value->type!=JSON_STRING){
                fg_error_set(err,FG_ERR_ARGUMENT,"text content parts require a text string");
                status=FG_ERR_ARGUMENT;
                break;
            }
            status=buffer_append(&text,value->as.string,err);
            continue;
        }
        if(!strcmp(type->as.string,"video_frames")){
            json_value *frames_field=json_object_get(part,"video_frames");
            if(!frames_field||frames_field->type!=JSON_OBJECT){
                fg_error_set(err,FG_ERR_ARGUMENT,
                             "video_frames parts require a video_frames object");
                status=FG_ERR_ARGUMENT;
                break;
            }
            json_value *frames_json=json_object_get(frames_field,"frames");
            if(!frames_json||frames_json->type!=JSON_ARRAY||!frames_json->as.array.count){
                fg_error_set(err,FG_ERR_ARGUMENT,
                             "video_frames requires a non-empty frames array");
                status=FG_ERR_ARGUMENT;
                break;
            }
            const size_t count=frames_json->as.array.count;
            if(count>FG_VIDEO_MAX_INPUT_FRAMES){
                fg_error_set(err,FG_ERR_LIMIT,"video_frames accepts at most %u frames",
                             FG_VIDEO_MAX_INPUT_FRAMES);
                status=FG_ERR_LIMIT;
                break;
            }
            json_value *fps_json=json_object_get(frames_field,"fps");
            double fps=FG_VIDEO_FPS_TARGET;
            if(fps_json){
                if(fps_json->type!=JSON_NUMBER||!isfinite(fps_json->as.number)||
                   !(fps_json->as.number>0.0)||fps_json->as.number>1000.0){
                    fg_error_set(err,FG_ERR_ARGUMENT,
                                 "video_frames fps must be a positive number");
                    status=FG_ERR_ARGUMENT;
                    break;
                }
                fps=fps_json->as.number;
            }
            json_value *max_json=json_object_get(frames_field,"max_frames");
            uint32_t max_frames=FG_VIDEO_MAX_FRAMES;
            if(max_json){
                if(max_json->type!=JSON_NUMBER||!number_is_integer(max_json->as.number)||
                   max_json->as.number<1.0||max_json->as.number>FG_VIDEO_MAX_FRAMES){
                    fg_error_set(err,FG_ERR_ARGUMENT,
                                 "video_frames max_frames must be an integer from 1 through %u",
                                 FG_VIDEO_MAX_FRAMES);
                    status=FG_ERR_ARGUMENT;
                    break;
                }
                max_frames=(uint32_t)max_json->as.number;
            }
            uint8_t **frames=malloc(count*sizeof(*frames));
            size_t *frame_lengths=malloc(count*sizeof(*frame_lengths));
            if(!frames||!frame_lengths){
                free(frame_lengths);
                free(frames);
                fg_error_set(err,FG_ERR_OOM,"allocate video frame list");
                status=FG_ERR_OOM;
                break;
            }
            memset(frames,0,count*sizeof(*frames));
            memset(frame_lengths,0,count*sizeof(*frame_lengths));
            for(size_t frame=0;status==FG_OK&&frame<count;frame++){
                json_value *item=frames_json->as.array.items[frame];
                if(!item||item->type!=JSON_STRING){
                    fg_error_set(err,FG_ERR_ARGUMENT,
                                 "video_frames frames must be base64 image data URLs");
                    status=FG_ERR_ARGUMENT;
                    break;
                }
                status=api_decode_media_url(item->as.string,FG_RUNTIME_MEDIA_IMAGE,
                                            "video_frames frame",&frames[frame],
                                            &frame_lengths[frame],err);
            }
            if(status==FG_OK)
                status=api_request_add_video_frames(request,frames,frame_lengths,count,fps,
                                                    max_frames,err);
            if(status!=FG_OK){
                for(size_t frame=0;frame<count;frame++)free(frames[frame]);
                free(frames);
                free(frame_lengths);
                break;
            }
            status=buffer_append(&text,"<|vision_start|><|video_pad|><|vision_end|>",err);
            continue;
        }
        if(!strcmp(type->as.string,"image_url")||!strcmp(type->as.string,"video_url")){
            const bool video=!strcmp(type->as.string,"video_url");
            const char *field=video?"video_url":"image_url";
            json_value *media_url=json_object_get(part,field);
            const char *url=NULL;
            if(media_url&&media_url->type==JSON_STRING)url=media_url->as.string;
            else if(media_url&&media_url->type==JSON_OBJECT){
                json_value *nested=json_object_get(media_url,"url");
                if(nested&&nested->type==JSON_STRING)url=nested->as.string;
            }else if(media_url&&media_url->type==JSON_NULL)url=NULL;
            if(!url){
                fg_error_set(err,FG_ERR_ARGUMENT,"%s parts require %s.url",field,field);
                status=FG_ERR_ARGUMENT;
                break;
            }
            uint8_t *data=NULL;
            size_t length=0;
            status=api_decode_media_url(url,video?FG_RUNTIME_MEDIA_VIDEO:FG_RUNTIME_MEDIA_IMAGE,
                                        field,&data,&length,err);
            if(status==FG_OK)
                status=api_request_add_media(request,
                                             video?FG_RUNTIME_MEDIA_VIDEO:FG_RUNTIME_MEDIA_IMAGE,
                                             data,length,err);
            if(status==FG_OK)
                status=buffer_append(&text,video?
                                     "<|vision_start|><|video_pad|><|vision_end|>":
                                     "<|vision_start|><|image_pad|><|vision_end|>",err);
            if(status!=FG_OK)free(data);
            continue;
        }
        fg_error_set(err,FG_ERR_ARGUMENT,"message content part type '%s' is unsupported",
                     type->as.string);
        status=FG_ERR_ARGUMENT;
    }
    if(status==FG_OK&&!text.data){
        text.data=strdup("");
        if(!text.data){
            fg_error_set(err,FG_ERR_OOM,"copy empty message content array");
            status=FG_ERR_OOM;
        }
    }
    if(status!=FG_OK){free(text.data);return status;}
    *output=text.data;
    return FG_OK;
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
    for (size_t i = 0; i < request->media_count; i++) {
        for (size_t frame = 0; frame < request->media[i].frame_count; frame++)
            free(request->media[i].frames[frame]);
        free(request->media[i].frames);
        free(request->media[i].frame_lengths);
        free(request->media[i].data);
    }
    free(request->media);
    for (size_t i = 0; i < request->tool_schema_count; i++) free(request->tool_schemas[i]);
    free(request->tool_schemas);
    free(request->tool_choice_name);
    memset(request, 0, sizeof(*request));
}

static void api_public_session_free(api_public_session *session) {
    if (!session) return;
    api_chat_request_free(&session->transcript);
    free(session->media);
    session->media = NULL;
    session->media_count = 0;
    session->valid = false;
}

static bool api_text_equal(const char *left,const char *right) {
    return !strcmp(left ? left : "",right ? right : "");
}

static bool api_content_equal(const char *left,const char *right) {
    const char *a=left?left:"",*b=right?right:"";
    const char *end_a,*end_b;
    while(*a&&isspace((unsigned char)*a))a++;
    while(*b&&isspace((unsigned char)*b))b++;
    end_a=a+strlen(a);end_b=b+strlen(b);
    while(end_a>a&&isspace((unsigned char)end_a[-1]))end_a--;
    while(end_b>b&&isspace((unsigned char)end_b[-1]))end_b--;
    while(a<end_a&&b<end_b){
        char ca=*a,cb=*b;
        if(ca=='\r'){
            if(a+1<end_a&&a[1]=='\n')a++;
            ca='\n';
        }
        if(cb=='\r'){
            if(b+1<end_b&&b[1]=='\n')b++;
            cb='\n';
        }
        if(ca!=cb)return false;
        a++;b++;
    }
    return a==end_a&&b==end_b;
}

static size_t api_content_diff_offset(const char *left,const char *right) {
    size_t offset=0;
    while(left[offset]&&left[offset]==right[offset])offset++;
    return offset;
}

static void api_content_escape(const char *text,size_t limit,char *output,
                               size_t output_size) {
    static const char hex[]="0123456789abcdef";
    size_t written=0;
    if(!output_size)return;
    for(size_t i=0;text[i]&&i<limit;i++){
        unsigned char byte=(unsigned char)text[i];
        char encoded[4];size_t length;
        if(byte=='\\'||byte=='"'){encoded[0]='\\';encoded[1]=(char)byte;length=2u;}
        else if(byte=='\n'){encoded[0]='\\';encoded[1]='n';length=2u;}
        else if(byte=='\r'){encoded[0]='\\';encoded[1]='r';length=2u;}
        else if(byte=='\t'){encoded[0]='\\';encoded[1]='t';length=2u;}
        else if(byte>=0x20u&&byte<0x7fu){encoded[0]=(char)byte;length=1u;}
        else{
            encoded[0]='\\';encoded[1]='x';
            encoded[2]=hex[byte>>4u];encoded[3]=hex[byte&15u];
            length=4u;
        }
        if(written+length+1u>output_size)break;
        memcpy(output+written,encoded,length);
        written+=length;
    }
    output[written]=0;
}

static void api_mismatch(char *reason,size_t reason_size,const char *format,...) {
    if(!reason||!reason_size)return;
    va_list args;
    va_start(args,format);
    vsnprintf(reason,reason_size,format,args);
    va_end(args);
}

static bool api_message_equal(size_t index,const fg_chat_message *left,
                              const fg_chat_message *right,char *reason,
                              size_t reason_size) {
    if(!api_text_equal(left->role,right->role)){
        api_mismatch(reason,reason_size,"message[%zu].role",index);
        return false;
    }
    if(!api_content_equal(left->content,right->content)){
        const char *stored=left->content?left->content:"";
        const char *echoed=right->content?right->content:"";
        char stored_text[FG_API_CONTENT_SNIPPET*4u+1u];
        char echoed_text[FG_API_CONTENT_SNIPPET*4u+1u];
        size_t offset=api_content_diff_offset(stored,echoed);
        api_content_escape(stored+offset,FG_API_CONTENT_SNIPPET,stored_text,
                           sizeof(stored_text));
        api_content_escape(echoed+offset,FG_API_CONTENT_SNIPPET,echoed_text,
                           sizeof(echoed_text));
        api_mismatch(reason,reason_size,
                     "message[%zu].content@%zu stored=\"%s\" echoed=\"%s\"",
                     index,offset,stored_text,echoed_text);
        return false;
    }
    if(!api_text_equal(left->tool_call_id,right->tool_call_id)){
        api_mismatch(reason,reason_size,"message[%zu].tool_call_id",index);
        return false;
    }
    if(left->tool_call_count!=right->tool_call_count){
        api_mismatch(reason,reason_size,"message[%zu].tool_calls=%zu->%zu",index,
                     left->tool_call_count,right->tool_call_count);
        return false;
    }
    for(size_t i=0;i<left->tool_call_count;i++){
        const fg_chat_tool_call *a=&left->tool_calls[i],*b=&right->tool_calls[i];
        if(!api_text_equal(a->id,b->id)){
            api_mismatch(reason,reason_size,"message[%zu].tool_calls[%zu].id",index,i);
            return false;
        }
        if(!api_text_equal(a->name,b->name)){
            api_mismatch(reason,reason_size,"message[%zu].tool_calls[%zu].name",index,i);
            return false;
        }
        if(!api_text_equal(a->arguments_json,b->arguments_json)){
            api_mismatch(reason,reason_size,
                         "message[%zu].tool_calls[%zu].arguments_json",index,i);
            return false;
        }
    }
    return true;
}

static size_t api_vision_marker_count(const char *content){
    static const char marker[]="<|vision_start|>";
    size_t count=0;
    if(!content)return 0;
    for(const char *cursor=content;(cursor=strstr(cursor,marker))!=NULL;
        cursor+=sizeof(marker)-1u)
        count++;
    return count;
}

static bool api_public_session_prefix(const api_public_session *session,
                                      const api_chat_request *request,
                                      char *reason,size_t reason_size) {
    if(reason&&reason_size)reason[0]=0;
    if(!session||!session->valid)return false;
    if(request->message_count<session->transcript.message_count){
        api_mismatch(reason,reason_size,"messages=%zu->%zu",
                     session->transcript.message_count,request->message_count);
        return false;
    }
    size_t previous_system=fg_chat_leading_system_count(session->transcript.messages,
                                                        session->transcript.message_count);
    size_t current_system=fg_chat_leading_system_count(request->messages,
                                                       request->message_count);
    /* The leading system run is delta-eligible and is not compared here, so a
     * media marker inside it would desynchronize the per-message media index
     * from the request media list.  Refuse the continuation instead (the
     * request then cold-starts through the vision path). */
    for(size_t i=0;i<previous_system;i++)
        if(api_vision_marker_count(session->transcript.messages[i].content)){
            api_mismatch(reason,reason_size,"system media is not continuation-safe");
            return false;
        }
    for(size_t i=0;i<current_system;i++)
        if(api_vision_marker_count(request->messages[i].content)){
            api_mismatch(reason,reason_size,"system media is not continuation-safe");
            return false;
        }
    size_t media_index=0;
    for(size_t i=previous_system;i<session->transcript.message_count;i++){
        size_t current=current_system+(i-previous_system);
        if(!api_message_equal(current,&session->transcript.messages[i],
                              &request->messages[current],reason,reason_size))
            return false;
        /* The content compare only sees the placeholder markers, which are
         * identical for every image.  Every marker in the compared prefix must
         * also carry the stored payload identity, so different bytes at the
         * same position force a reset while a re-sent image continues. */
        size_t markers=api_vision_marker_count(session->transcript.messages[i].content);
        for(size_t marker=0;marker<markers;marker++,media_index++){
            if(media_index>=session->media_count||media_index>=request->media_count){
                api_mismatch(reason,reason_size,"message[%zu].media[%zu]=count",
                             current,marker);
                return false;
            }
            const api_media_identity *stored=&session->media[media_index];
            const api_media *echoed=&request->media[media_index];
            if(stored->kind!=echoed->kind||
               memcmp(stored->digest,echoed->digest,sizeof(stored->digest))){
                api_mismatch(reason,reason_size,
                             "message[%zu].media[%zu] kind=%u->%u digest",
                             current,marker,stored->kind,echoed->kind);
                return false;
            }
        }
    }
    if(media_index!=session->media_count){
        api_mismatch(reason,reason_size,"media=%zu->%zu",session->media_count,media_index);
        return false;
    }
    return true;
}

static fg_status api_copy_message(fg_chat_message *output,
                                  const fg_chat_message *input,fg_error *err) {
    output->role=strdup(input->role?input->role:"");
    output->content=strdup(input->content?input->content:"");
    if(input->reasoning)output->reasoning=strdup(input->reasoning);
    if(input->tool_call_id)output->tool_call_id=strdup(input->tool_call_id);
    if(!output->role||!output->content||
       (input->reasoning&&!output->reasoning)||
       (input->tool_call_id&&!output->tool_call_id)){
        fg_error_set(err,FG_ERR_OOM,"copy public API transcript message");
        return FG_ERR_OOM;
    }
    if(!input->tool_call_count)return FG_OK;
    fg_chat_tool_call *calls=calloc(input->tool_call_count,sizeof(*calls));
    if(!calls){fg_error_set(err,FG_ERR_OOM,"copy public API tool calls");return FG_ERR_OOM;}
    output->tool_calls=calls;output->tool_call_count=input->tool_call_count;
    for(size_t i=0;i<input->tool_call_count;i++){
        calls[i].id=strdup(input->tool_calls[i].id?input->tool_calls[i].id:"");
        calls[i].name=strdup(input->tool_calls[i].name?input->tool_calls[i].name:"");
        calls[i].arguments_json=strdup(input->tool_calls[i].arguments_json?
                                       input->tool_calls[i].arguments_json:"");
        if(!calls[i].id||!calls[i].name||!calls[i].arguments_json){
            fg_error_set(err,FG_ERR_OOM,"copy public API tool call");
            return FG_ERR_OOM;
        }
    }
    return FG_OK;
}

static fg_status api_public_session_build(api_public_session *output,
                                          const api_chat_request *request,
                                          const api_generation *generation,
                                          const fg_chat_generated *generated,
                                          const char *session_id,uint64_t numeric_id,
                                          fg_error *err) {
    memset(output,0,sizeof(*output));
    if(session_id)snprintf(output->id,sizeof(output->id),"%s",session_id);
    output->numeric_id=numeric_id;
    api_chat_request *copy=&output->transcript;
    copy->message_count=request->message_count+1u;
    copy->messages=calloc(copy->message_count,sizeof(*copy->messages));
    if(!copy->messages){fg_error_set(err,FG_ERR_OOM,"allocate public API transcript");return FG_ERR_OOM;}
    fg_status status=FG_OK;
    for(size_t i=0;status==FG_OK&&i<request->message_count;i++)
        status=api_copy_message(&copy->messages[i],&request->messages[i],err);
    fg_chat_message *assistant=&copy->messages[request->message_count];
    if(status==FG_OK){
        assistant->role=strdup("assistant");
        assistant->content=strdup(generated->content?generated->content:"");
        if(!assistant->role||!assistant->content){
            fg_error_set(err,FG_ERR_OOM,"copy public API assistant response");
            status=FG_ERR_OOM;
        }
    }
    if(status==FG_OK&&generated->tool_call_count){
        fg_chat_tool_call *calls=calloc(generated->tool_call_count,sizeof(*calls));
        if(!calls){fg_error_set(err,FG_ERR_OOM,"copy public API assistant tool calls");status=FG_ERR_OOM;}
        else{
            assistant->tool_calls=calls;
            assistant->tool_call_count=generated->tool_call_count;
            for(size_t i=0;status==FG_OK&&i<generated->tool_call_count;i++){
                char id[128];tool_call_id(generation,i,id);
                calls[i].id=strdup(id);
                calls[i].name=strdup(generated->tool_calls[i].name);
                calls[i].arguments_json=strdup(generated->tool_calls[i].arguments_json);
                if(!calls[i].id||!calls[i].name||!calls[i].arguments_json){
                    fg_error_set(err,FG_ERR_OOM,"copy public API assistant tool call");
                    status=FG_ERR_OOM;
                }
            }
        }
    }
    copy->tool_schema_count=request->tool_schema_count;
    if(status==FG_OK&&copy->tool_schema_count){
        copy->tool_schemas=calloc(copy->tool_schema_count,sizeof(*copy->tool_schemas));
        if(!copy->tool_schemas){fg_error_set(err,FG_ERR_OOM,"copy public API tool schemas");status=FG_ERR_OOM;}
        for(size_t i=0;status==FG_OK&&i<copy->tool_schema_count;i++){
            copy->tool_schemas[i]=strdup(request->tool_schemas[i]);
            if(!copy->tool_schemas[i]){fg_error_set(err,FG_ERR_OOM,"copy public API tool schema");status=FG_ERR_OOM;}
        }
    }
    copy->tool_choice=request->tool_choice;
    if(status==FG_OK&&request->tool_choice_name){
        copy->tool_choice_name=strdup(request->tool_choice_name);
        if(!copy->tool_choice_name){fg_error_set(err,FG_ERR_OOM,"copy public API tool choice");status=FG_ERR_OOM;}
    }
    if(status==FG_OK&&request->media_count){
        api_media_identity *media=calloc(request->media_count,sizeof(*media));
        if(!media){fg_error_set(err,FG_ERR_OOM,"copy public API media identity");status=FG_ERR_OOM;}
        else{
            output->media=media;
            output->media_count=request->media_count;
            for(size_t i=0;i<request->media_count;i++){
                media[i].kind=request->media[i].kind;
                memcpy(media[i].digest,request->media[i].digest,
                       sizeof(media[i].digest));
            }
        }
    }
    if(status!=FG_OK){api_public_session_free(output);return status;}
    output->valid=true;
    return FG_OK;
}

static bool number_is_integer(double value) {
    return value >= 0.0 && value <= (double)UINT32_MAX && value == (double)(uint32_t)value;
}

static fg_status parse_sampler_controls(const json_value *root,api_chat_request *request,
                                        fg_error *err){
    fg_sampler_config_defaults(&request->sampler);
    json_value *value=json_object_get(root,"temperature");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)||value->as.number<0.0||value->as.number>FLT_MAX){fg_error_set(err,FG_ERR_ARGUMENT,"temperature must be a finite non-negative number");return FG_ERR_ARGUMENT;}request->sampler.temperature=(float)value->as.number;}
    value=json_object_get(root,"top_p");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)||value->as.number<=0.0||value->as.number>1.0){fg_error_set(err,FG_ERR_ARGUMENT,"top_p must be greater than 0 and at most 1");return FG_ERR_ARGUMENT;}request->sampler.top_p=(float)value->as.number;}
    value=json_object_get(root,"top_k");
    if(value){if(value->type!=JSON_NUMBER||!number_is_integer(value->as.number)||value->as.number<1.0||value->as.number>64.0){fg_error_set(err,FG_ERR_ARGUMENT,"top_k must be an integer from 1 through 64");return FG_ERR_ARGUMENT;}request->sampler.top_k=(uint32_t)value->as.number;}
    value=json_object_get(root,"seed");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)||value->as.number<0.0||value->as.number>9007199254740991.0||value->as.number!=floor(value->as.number)){fg_error_set(err,FG_ERR_ARGUMENT,"seed must be a non-negative integer");return FG_ERR_ARGUMENT;}request->sampler.seed=(uint64_t)value->as.number;}
    value=json_object_get(root,"presence_penalty");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)){fg_error_set(err,FG_ERR_ARGUMENT,"presence_penalty must be finite");return FG_ERR_ARGUMENT;}request->sampler.presence_penalty=(float)value->as.number;}
    value=json_object_get(root,"frequency_penalty");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)){fg_error_set(err,FG_ERR_ARGUMENT,"frequency_penalty must be finite");return FG_ERR_ARGUMENT;}request->sampler.frequency_penalty=(float)value->as.number;}
    value=json_object_get(root,"repetition_penalty");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)){fg_error_set(err,FG_ERR_ARGUMENT,"repetition_penalty must be finite");return FG_ERR_ARGUMENT;}request->sampler.repetition_penalty=(float)value->as.number;}
    value=json_object_get(root,"min_p");
    if(value){if(value->type!=JSON_NUMBER||!isfinite(value->as.number)||value->as.number!=0.0){fg_error_set(err,FG_ERR_ARGUMENT,"min_p currently supports only 0");return FG_ERR_ARGUMENT;}request->sampler.min_p=0.0f;}
    return fg_sampler_config_validate(&request->sampler,err);
}

static fg_status reject_control(const json_value *root, const char *name, double allowed,
                                fg_error *err) {
    json_value *value = json_object_get(root, name);
    if (!value) return FG_OK;
    if (value->type != JSON_NUMBER || value->as.number != allowed) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "%s is unsupported",
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
    static const char *unsupported[] = {"typical_p", "logit_bias"};
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
        if (json_object_has(root, unsupported[i])) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "%s is unsupported",
                         unsupported[i]);
            return FG_ERR_ARGUMENT;
        }
    }
    status = parse_sampler_controls(root,request,err);
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
    json_value *stream = json_object_get(root, "stream");
    if (stream) {
        if (stream->type != JSON_BOOL) {
            fg_error_set(err, FG_ERR_ARGUMENT, "stream must be a boolean");
            return FG_ERR_ARGUMENT;
        }
        request->stream = stream->as.boolean;
    }

    /* M3 public session identity: an optional opaque continuation key. */
    json_value *session_id = json_object_get(root, "session_id");
    if (session_id) {
        if (session_id->type != JSON_STRING || !session_id->as.string[0] ||
            strlen(session_id->as.string) >= sizeof(request->session_id) ||
            !api_session_id_valid(session_id->as.string)) {
            fg_error_set(err, FG_ERR_ARGUMENT,
                         "session_id must be 1..47 characters of [A-Za-z0-9._:-]");
            return FG_ERR_ARGUMENT;
        }
        snprintf(request->session_id, sizeof(request->session_id), "%s",
                 session_id->as.string);
        request->session_id_set = true;
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
            (content && content->type != JSON_STRING && content->type != JSON_ARRAY &&
             content->type != JSON_NULL) ||
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
        fg_status content_status=message_content_parts(content,&request->messages[i].content,request,err);
        request->message_count++;
        if (!request->messages[i].role || content_status!=FG_OK) {
            if(content_status==FG_OK)
                fg_error_set(err, FG_ERR_OOM, "copy API chat message");
            api_chat_request_free(request);
            return content_status==FG_OK?FG_ERR_OOM:content_status;
        }
        if (!request->messages[i].content) {
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

static bool header_value_equal(const char *value, size_t length, const char *wanted) {
    while (length && isspace((unsigned char)value[length - 1u])) length--;
    size_t wanted_length = strlen(wanted);
    if (length != wanted_length) return false;
    for (size_t i = 0; i < length; i++)
        if (tolower((unsigned char)value[i]) != tolower((unsigned char)wanted[i])) return false;
    return true;
}

/* --- incremental HTTP parsing ------------------------------------------- */

static void api_http_parser_consume(api_http_parser *parser, size_t count) {
    if (count >= parser->input.length) {
        parser->input.length = 0;
        if (parser->input.data) parser->input.data[0] = 0;
        return;
    }
    memmove(parser->input.data, parser->input.data + count, parser->input.length - count);
    parser->input.length -= count;
    parser->input.data[parser->input.length] = 0;
}

/* Decode a chunked body already buffered in `data`.  Returns
 * API_PARSE_INCOMPLETE until the terminal chunk (and trailers) are present so
 * the front-end can retry after more bytes arrive. */
static api_parse_state api_chunked_decode(const char *data, size_t length, size_t cursor,
                                          api_buffer *decoded, size_t *end_cursor,
                                          unsigned *http_status, fg_error *err) {
    for (;;) {
        size_t line_end = SIZE_MAX;
        for (size_t i = cursor; i + 1u < length; i++)
            if (data[i] == '\r' && data[i + 1u] == '\n') {
                line_end = i;
                break;
            }
        if (line_end == SIZE_MAX) return API_PARSE_INCOMPLETE;
        size_t hex_begin = cursor, hex_end = cursor;
        while (hex_end < line_end && isxdigit((unsigned char)data[hex_end])) hex_end++;
        if (hex_end == hex_begin) {
            *http_status = 400u;
            fg_error_set(err, FG_ERR_FORMAT, "invalid chunk size");
            return API_PARSE_FAILED;
        }
        unsigned long long size = 0;
        for (size_t i = hex_begin; i < hex_end; i++) {
            int digit = json_hex(data[i]);
            size = size * 16u + (unsigned long long)digit;
            if (size > FG_API_MAX_REQUEST_BYTES) {
                *http_status = 413u;
                fg_error_set(err, FG_ERR_LIMIT, "chunked HTTP request exceeds 32 MiB");
                return API_PARSE_FAILED;
            }
        }
        cursor = line_end + 2u;
        if (!size) {
            for (;;) {
                if (cursor + 1u < length && data[cursor] == '\r' &&
                    data[cursor + 1u] == '\n') {
                    cursor += 2u;
                    *end_cursor = cursor;
                    return API_PARSE_COMPLETE;
                }
                bool trailer_end = false;
                for (size_t i = cursor; i + 3u < length; i++)
                    if (!memcmp(data + i, "\r\n\r\n", 4u)) {
                        cursor = i + 4u;
                        trailer_end = true;
                        break;
                    }
                if (trailer_end) {
                    *end_cursor = cursor;
                    return API_PARSE_COMPLETE;
                }
                return API_PARSE_INCOMPLETE;
            }
        }
        if (length < cursor + (size_t)size + 2u) return API_PARSE_INCOMPLETE;
        if (data[cursor + (size_t)size] != '\r' ||
            data[cursor + (size_t)size + 1u] != '\n') {
            *http_status = 400u;
            fg_error_set(err, FG_ERR_FORMAT, "malformed chunked HTTP body");
            return API_PARSE_FAILED;
        }
        fg_status appended = buffer_append_n(decoded, data + cursor, (size_t)size, err);
        if (appended != FG_OK) return API_PARSE_FAILED;
        cursor += (size_t)size + 2u;
    }
}

/* Try to consume exactly one complete request from the parser buffer.  On
 * API_PARSE_COMPLETE the caller must call api_http_parser_consume() with the
 * returned `consumed` count; pipelined bytes stay in the buffer. */
static api_parse_state api_http_parser_try(api_http_parser *parser, http_request *request,
                                           bool *keep_alive, size_t *consumed,
                                           unsigned *http_status, fg_error *err) {
    memset(request, 0, sizeof(*request));
    *keep_alive = true;
    *consumed = 0u;
    *http_status = 400u;
    char *data = parser->input.data;
    size_t length = parser->input.length;
    if (!data || length < 4u) return API_PARSE_INCOMPLETE;
    char *header_end = find_header_end(data, length);
    if (!header_end) {
        if (length >= FG_API_MAX_REQUEST_BYTES) {
            *http_status = 413u;
            fg_error_set(err, FG_ERR_LIMIT, "HTTP request exceeds 32 MiB");
            return API_PARSE_FAILED;
        }
        if (length >= FG_API_MAX_HEADER_BYTES) {
            *http_status = 413u;
            fg_error_set(err, FG_ERR_LIMIT, "HTTP request headers exceed 256 KiB");
            return API_PARSE_FAILED;
        }
        return API_PARSE_INCOMPLETE;
    }
    size_t header_bytes = (size_t)(header_end - data) + 4u;
    char *header = malloc(header_bytes + 1u);
    if (!header) {
        fg_error_set(err, FG_ERR_OOM, "allocate HTTP header copy");
        return API_PARSE_FAILED;
    }
    memcpy(header, data, header_bytes);
    header[header_bytes] = 0;
    char *line_end = strstr(header, "\r\n");
    if (!line_end) {
        free(header);
        fg_error_set(err, FG_ERR_FORMAT, "invalid HTTP request line");
        return API_PARSE_FAILED;
    }
    *line_end = 0;
    char version[16];
    if (sscanf(header, "%7s %255s %15s", request->method, request->path, version) != 3 ||
        strcmp(version, "HTTP/1.1")) {
        free(header);
        fg_error_set(err, FG_ERR_FORMAT, "expected an HTTP/1.1 request line");
        return API_PARSE_FAILED;
    }
    size_t content_length = 0;
    bool have_length = false;
    bool chunked = false;
    bool json_content = false;
    bool close_requested = false;
    char *header_limit = header + header_bytes - 4u;
    for (char *line = line_end + 2; line < header_limit;) {
        char *next = strstr(line, "\r\n");
        if (!next || next > header_limit) break;
        char *colon = memchr(line, ':', (size_t)(next - line));
        if (!colon) {
            free(header);
            fg_error_set(err, FG_ERR_FORMAT, "invalid HTTP header");
            return API_PARSE_FAILED;
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
                free(header);
                *http_status = parsed > FG_API_MAX_REQUEST_BYTES ? 413u : 400u;
                fg_error_set(err, FG_ERR_LIMIT, "invalid or excessive Content-Length");
                return API_PARSE_FAILED;
            }
            content_length = (size_t)parsed;
            have_length = true;
        } else if (header_name_equal(line, name_length, "Transfer-Encoding")) {
            size_t value_length = (size_t)(next - value);
            if (!header_value_equal(value, value_length, "chunked")) {
                free(header);
                *http_status = 400u;
                fg_error_set(err, FG_ERR_ARGUMENT,
                             "unsupported Transfer-Encoding; only chunked is accepted");
                return API_PARSE_FAILED;
            }
            chunked = true;
        } else if (header_name_equal(line, name_length, "Content-Type")) {
            size_t value_length = (size_t)(next - value);
            json_content = value_length >= 16u &&
                           !strncasecmp(value, "application/json", 16u);
        } else if (header_name_equal(line, name_length, "Connection")) {
            size_t value_length = (size_t)(next - value);
            if (header_value_equal(value, value_length, "close")) close_requested = true;
        }
        line = next + 2;
    }
    free(header);
    *keep_alive = !close_requested;
    bool body_required = !strcmp(request->method, "POST");
    if (chunked && have_length) {
        *http_status = 400u;
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "Content-Length and Transfer-Encoding are mutually exclusive");
        return API_PARSE_FAILED;
    }
    if (body_required && !have_length && !chunked) {
        *http_status = 411u;
        fg_error_set(err, FG_ERR_ARGUMENT, "POST requires Content-Length");
        return API_PARSE_FAILED;
    }
    if (body_required && !json_content) {
        *http_status = 415u;
        fg_error_set(err, FG_ERR_ARGUMENT, "POST requires application/json");
        return API_PARSE_FAILED;
    }
    if (chunked) {
        api_buffer decoded = {0};
        size_t end_cursor = 0u;
        api_parse_state state = api_chunked_decode(data, length, header_bytes, &decoded,
                                                   &end_cursor, http_status, err);
        if (state != API_PARSE_COMPLETE) {
            free(decoded.data);
            return state;
        }
        request->body = decoded.data ? decoded.data : strdup("");
        if (!request->body) {
            free(decoded.data);
            fg_error_set(err, FG_ERR_OOM, "allocate HTTP body");
            return API_PARSE_FAILED;
        }
        request->body_length = decoded.length;
        *consumed = end_cursor;
        return API_PARSE_COMPLETE;
    }
    if (header_bytes + content_length > FG_API_MAX_REQUEST_BYTES) {
        *http_status = 413u;
        fg_error_set(err, FG_ERR_LIMIT, "HTTP request exceeds 32 MiB");
        return API_PARSE_FAILED;
    }
    if (length < header_bytes + content_length) return API_PARSE_INCOMPLETE;
    request->body = malloc(content_length + 1u);
    if (!request->body) {
        fg_error_set(err, FG_ERR_OOM, "allocate HTTP body");
        return API_PARSE_FAILED;
    }
    memcpy(request->body, data + header_bytes, content_length);
    request->body[content_length] = 0;
    request->body_length = content_length;
    *consumed = header_bytes + content_length;
    return API_PARSE_COMPLETE;
}

/* Blocking compatibility wrapper used by the API unit tests only; the serving
 * process reads through api_http_parser_try() on the front-end poll loop. */
#ifdef FG_API_TEST_BUILD
static fg_status read_http_request(int fd, http_request *request, unsigned *http_status,
                                   fg_error *err) {
    api_http_parser parser = {0};
    memset(request, 0, sizeof(*request));
    *http_status = 400u;
    for (;;) {
        if (parser.input.length >= FG_API_MAX_REQUEST_BYTES) {
            *http_status = 413u;
            fg_error_set(err, FG_ERR_LIMIT, "HTTP request exceeds 32 MiB");
            free(parser.input.data);
            return FG_ERR_LIMIT;
        }
        size_t chunk = FG_API_MAX_REQUEST_BYTES - parser.input.length;
        if (chunk > 8192u) chunk = 8192u;
        fg_status status = buffer_reserve(&parser.input, chunk, err);
        if (status != FG_OK) {
            free(parser.input.data);
            return status;
        }
        ssize_t received;
        for (;;) {
            received = recv(fd, parser.input.data + parser.input.length, chunk, 0);
            if (received >= 0 || errno != EINTR || api_stop_requested) break;
        }
        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                *http_status = 408u;
                fg_error_set(err, FG_ERR_IO, "HTTP request read timed out");
            } else if (errno == EINTR) {
                fg_error_set(err, FG_ERR_IO, "HTTP request interrupted by shutdown");
            } else {
                fg_error_set(err, FG_ERR_IO, "receive HTTP request: %s", strerror(errno));
            }
            free(parser.input.data);
            return FG_ERR_IO;
        }
        if (!received) {
            fg_error_set(err, FG_ERR_FORMAT, "client closed before HTTP headers completed");
            free(parser.input.data);
            return FG_ERR_FORMAT;
        }
        parser.input.length += (size_t)received;
        parser.input.data[parser.input.length] = 0;
        bool keep_alive = true;
        size_t consumed = 0u;
        api_parse_state state = api_http_parser_try(&parser, request, &keep_alive, &consumed,
                                                    http_status, err);
        if (state == API_PARSE_COMPLETE) {
            free(parser.input.data);
            return FG_OK;
        }
        if (state == API_PARSE_FAILED) {
            fg_status failed = err->code ? err->code : FG_ERR_FORMAT;
            free(parser.input.data);
            free(request->body);
            request->body = NULL;
            request->body_length = 0u;
            return failed;
        }
    }
}
#endif /* FG_API_TEST_BUILD */

static void api_signal_handler(int signal_number) {
    (void)signal_number;
    api_stop_requested = 1;
    if (api_wake_fd >= 0) {
        char byte = 1;
        ssize_t ignored = write(api_wake_fd, &byte, 1u);
        (void)ignored;
    }
}

static bool api_client_gone(api_generation *generation) {
    if (!generation) return false;
    if (generation->sink.connection)
        return atomic_load(&generation->sink.connection->client_gone);
    int fd = generation->sink.fd;
    if (fd < 0) return false;
    struct pollfd probe = {.fd = fd, .events = POLLIN};
    int ready = poll(&probe, 1u, 0);
    if (ready <= 0) return false;
    if (probe.revents & (POLLERR | POLLHUP | POLLRDHUP)) return true;
    if (!(probe.revents & (POLLIN | POLLRDNORM))) return false;
    char byte = 0;
    ssize_t peeked = recv(fd, &byte, 1u, MSG_PEEK | MSG_DONTWAIT);
    return peeked == 0 ||
           (peeked < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
}

/* The engine's interrupt hook is a pure abort check: the front-end thread owns
 * socket I/O, keep-alive comments and listener servicing, and publishes client
 * disconnect through the connection's `client_gone` flag. */
static bool api_interrupted(void *context) {
    if (api_stop_requested) return true;
    api_generation *generation = context;
    if (generation && api_client_gone(generation)) {
        generation->client_failed = true;
        return true;
    }
    return false;
}

static fg_status api_send_sse_headers(api_sink *sink,const char *session_id,
                                      fg_error *err) {
    char session_header[80]={0};
    if(session_id&&session_id[0])
        snprintf(session_header,sizeof(session_header),
                 "X-Flash-Gordon-Session: %s\r\n",session_id);
    if (sink->connection) {
        /* Chunked framing lets a streaming response end without closing the
         * connection, so keep-alive works for SSE too. */
        char headers[512];
        int length=snprintf(headers,sizeof(headers),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Transfer-Encoding: chunked\r\n"
            "X-Accel-Buffering: no\r\n"
            "%s\r\n",session_header);
        if(length<0||(size_t)length>=sizeof(headers)){
            fg_error_set(err,FG_ERR_LIMIT,"SSE headers exceed buffer");
            return FG_ERR_LIMIT;
        }
        fg_status status = api_connection_enqueue_raw(sink->connection, headers,
                                                      (size_t)length, err);
        if (status == FG_OK) {
            api_connection *conn = sink->connection;
            pthread_mutex_lock(&conn->out_mutex);
            conn->chunked = true;
            pthread_mutex_unlock(&conn->out_mutex);
        }
        return status;
    }
    char headers[512];
    int length=snprintf(headers,sizeof(headers),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n"
        "%s\r\n",session_header);
    if(length<0||(size_t)length>=sizeof(headers)){
        fg_error_set(err,FG_ERR_LIMIT,"SSE headers exceed buffer");
        return FG_ERR_LIMIT;
    }
    return send_all(sink->fd, headers, (size_t)length, err);
}

static fg_status send_delta_field(api_generation *generation,const char *field,
                                  const char *text,size_t length,fg_error *err) {
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
            &event, ",\"choices\":[{\"index\":0,\"delta\":{\"", err);
    if (status == FG_OK) status = buffer_append(&event, field, err);
    if (status == FG_OK) status = buffer_append(&event, "\":", err);
    if (status == FG_OK) status = buffer_append_json_string(&event, text, length, err);
    if (status == FG_OK)
        status = buffer_append(&event, "},\"finish_reason\":null}]}\n\n", err);
    if (status == FG_OK) {
        status = api_sink_write(&generation->sink, event.data, event.length, err);
        if (status != FG_OK) generation->client_failed = true;
    }
    free(event.data);
    return status;
}

static fg_status send_content_delta(api_generation *generation,const char *text,size_t length,
                                    fg_error *err) {
    return send_delta_field(generation, "content", text, length, err);
}

static fg_status send_reasoning_delta(api_generation *generation,const char *text,
                                      size_t length,fg_error *err) {
    return send_delta_field(generation, "reasoning_content", text, length, err);
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
        status = api_sink_write(&generation->sink, event.data, event.length, err);
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

static fg_status send_utf8_delta_channel(api_generation *generation,const char *text,
                                         size_t length,bool reasoning,fg_error *err) {
    char *pending = reasoning ? generation->utf8_reasoning_pending : generation->utf8_pending;
    size_t *pending_length = reasoning ? &generation->utf8_reasoning_pending_length
                                       : &generation->utf8_pending_length;
    char combined[4100];size_t total=*pending_length+length;
    if(total>sizeof(combined)){fg_error_set(err,FG_ERR_LIMIT,"streamed token exceeds UTF-8 buffer");return FG_ERR_LIMIT;}
    memcpy(combined,pending,*pending_length);
    memcpy(combined+*pending_length,text,length);
    *pending_length=0;
    size_t offset=0,run=0;fg_status status=FG_OK;
    while(status==FG_OK&&offset<total){
        size_t unit=0;int valid=utf8_unit((const unsigned char *)combined+offset,total-offset,&unit);
        if(valid>0){offset+=unit;continue;}
        if(offset>run){
            status=reasoning?send_reasoning_delta(generation,combined+run,offset-run,err)
                            :send_content_delta(generation,combined+run,offset-run,err);
        }
        if(status!=FG_OK)break;
        if(valid==0){*pending_length=total-offset;memcpy(pending,combined+offset,*pending_length);return FG_OK;}
        status=reasoning?send_reasoning_delta(generation,"\xef\xbf\xbd",3u,err)
                        :send_content_delta(generation,"\xef\xbf\xbd",3u,err);
        offset++;run=offset;
    }
    if(status==FG_OK&&offset>run){
        status=reasoning?send_reasoning_delta(generation,combined+run,offset-run,err)
                        :send_content_delta(generation,combined+run,offset-run,err);
    }
    return status;
}

static fg_status send_utf8_delta(api_generation *generation,const char *text,size_t length,
                                 fg_error *err) {
    return send_utf8_delta_channel(generation, text, length, false, err);
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
    while(!generation->visible_started&&generation->visible_pending.length){
        char head=generation->visible_pending.data[0];
        if(head!='\r'&&head!='\n'){generation->visible_started=true;break;}
        pending_consume(&generation->visible_pending,1u);
    }
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
            while(bytes&&isspace(
                      (unsigned char)generation->visible_pending.data[bytes-1u]))
                bytes--;
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
        if(final)
            while(flush&&isspace(
                      (unsigned char)generation->visible_pending.data[flush-1u]))
                flush--;
        if(flush)status=send_utf8_delta(generation,generation->visible_pending.data,flush,err);
        if(status==FG_OK&&flush)pending_consume(&generation->visible_pending,flush);
        break;
    }
    return status;
}

static fg_status stream_reasoning(api_generation *generation, bool final, fg_error *err) {
    if (generation->think_closed || generation->output_stopped) return FG_OK;
    const char *base = generation->content.data;
    size_t total = generation->content.length;
    if (!base || !total) return FG_OK;
    if (generation->reasoning_emitted == 0u && total < 7u &&
        marker_suffix(base, total, "<think>"))
        return FG_OK;
    if (generation->reasoning_emitted < 7u && total >= 7u && !memcmp(base, "<think>", 7u))
        generation->reasoning_emitted = 7u;
    const char *close = strstr(base + generation->reasoning_emitted, "</think>");
    size_t limit = close ? (size_t)(close - base) : total;
    if (!close && !final) {
        size_t hold = marker_suffix(base + generation->reasoning_emitted,
                                    total - generation->reasoning_emitted, "</think>");
        limit -= hold;
    }
    fg_status status = FG_OK;
    if (limit > generation->reasoning_emitted) {
        status = send_utf8_delta_channel(generation, base + generation->reasoning_emitted,
                                         limit - generation->reasoning_emitted, true, err);
        generation->reasoning_emitted = limit;
    }
    if (status != FG_OK || !close) return status;
    generation->think_closed = true;
    const char *emit = close + sizeof("</think>") - 1u;
    while (*emit == '\r' || *emit == '\n') emit++;
    size_t emit_length = total - (size_t)(emit - base);
    if (!emit_length) return FG_OK;
    return queue_visible_content(generation, emit, emit_length, false, err);
}

static fg_status api_token(void *context, uint32_t token, const char *text, size_t length,
                           fg_error *err) {
    (void)token;
    api_generation *generation = context;
    size_t previous=generation->content.length;
    fg_status status=buffer_append_n(&generation->content,text,length,err);
    if(status!=FG_OK||!generation->stream)return status;
    if(!generation->think_closed)return stream_reasoning(generation,false,err);
    if(!length)return FG_OK;
    return queue_visible_content(generation,generation->content.data+previous,length,false,err);
}

static fg_status send_stream_start(api_generation *generation, fg_error *err) {
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
    if (status == FG_OK)
        status = api_sink_write(&generation->sink, event.data, event.length, err);
    free(event.data);
    return status;
}

static fg_status send_stream_end(api_generation *generation,
                                 const fg_chat_generated *generated,const char *reason,
                                 fg_error *err) {
    fg_status reasoning=stream_reasoning(generation,true,err);
    if(reasoning!=FG_OK)return reasoning;
    if(generation->utf8_reasoning_pending_length){
        generation->utf8_reasoning_pending_length=0;
        reasoning=send_reasoning_delta(generation,"\xef\xbf\xbd",3u,err);
        if(reasoning!=FG_OK)return reasoning;
    }
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
    if (status == FG_OK)
        status = api_sink_write(&generation->sink, event.data, event.length, err);
    free(event.data);
    return status;
}

static fg_status send_stream_error(api_generation *generation,const char *message,
                                   fg_error *err) {
    api_buffer event={0};
    fg_status status=buffer_append(&event,"data: {\"error\":{\"message\":",err);
    if(status==FG_OK)status=buffer_append_json_string(&event,message,strlen(message),err);
    if(status==FG_OK)status=buffer_append(&event,",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n",err);
    if(status==FG_OK)status=api_sink_write(&generation->sink,event.data,event.length,err);
    free(event.data);return status;
}

static fg_status send_completion(api_generation *generation,
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
            &body, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"", err);
    if (status == FG_OK && generated->reasoning && generated->reasoning[0]) {
        status = buffer_append(&body, ",\"reasoning_content\":", err);
        if (status == FG_OK)
            status = buffer_append_json_string(&body, generated->reasoning,
                                               strlen(generated->reasoning), err);
    }
    if (status == FG_OK) status = buffer_append(&body, ",\"content\":", err);
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
            stats->prefill_seconds > 0.0 ?
                stats->prefilled_tokens / stats->prefill_seconds : 0.0;
        double decode_tps =
            stats->decode_seconds > 0.0 ? stats->generated_tokens / stats->decode_seconds : 0.0;
        char session_header[80]={0};
        if(generation->session_id&&generation->session_id[0])
            snprintf(session_header,sizeof(session_header),
                     "X-Flash-Gordon-Session: %s\r\n",generation->session_id);
        char metrics[2304];
        int metrics_length = snprintf(
            metrics, sizeof(metrics),
            "%s"
            "X-Flash-Gordon-Execution-Mode: %s\r\n"
            "X-Flash-Gordon-Prompt-Tokens: %u\r\n"
            "X-Flash-Gordon-Prefilled-Tokens: %u\r\n"
            "X-Flash-Gordon-Reused-Tokens: %u\r\n"
            "X-Flash-Gordon-Prefix-Cache: %s\r\n"
            "X-Flash-Gordon-Exact-Frontier: %s\r\n"
            "X-Flash-Gordon-Reset-Reason: %s\r\n"
            "X-Flash-Gordon-Completion-Tokens: %u\r\n"
            "X-Flash-Gordon-Context-Tokens: %u\r\n"
            "X-Flash-Gordon-Prefill-Seconds: %.9f\r\n"
            "X-Flash-Gordon-Prefill-TPS: %.6f\r\n"
            "X-Flash-Gordon-Decode-Seconds: %.9f\r\n"
            "X-Flash-Gordon-Decode-TPS: %.6f\r\n"
            "X-Flash-Gordon-Ledger: %s\r\n",
            session_header,
            fg_execution_mode_name(stats->execution_mode),
            stats->prompt_tokens, stats->prefilled_tokens, stats->reused_tokens,
            stats->prefix_cache_hit ? "hit" : "miss",
            stats->exact_frontier ? "true" : "false",
            fg_prefix_reset_reason_name(stats->reset_reason),
            stats->generated_tokens, stats->context_tokens,
            stats->prefill_seconds, prefill_tps, stats->decode_seconds, decode_tps,
            fg_runtime_ledger(generation->runtime) ? fg_runtime_ledger(generation->runtime) : "");
        if(status==FG_OK&&
           (metrics_length < 0 || (size_t)metrics_length >= sizeof(metrics))) {
            fg_error_set(err, FG_ERR_LIMIT, "API metrics headers exceed buffer");
            status = FG_ERR_LIMIT;
        } else if(status==FG_OK) {
            status = api_send_response_with_headers(&generation->sink, 200u,
                                                    "application/json", metrics,
                                                    body.data, body.length,
                                                    generation->keep_alive, err);
        }
    }
    free(body.data);
    return status;
}

static fg_status handle_models(api_sink *sink, fg_runtime *runtime, bool keep_alive,
                               fg_error *err) {
    const char *model = fg_runtime_model_name(runtime);
    const char *mtp = fg_runtime_mtp_capability(runtime) == FG_MTP_CAPABILITY_ENABLED ?
        "true" : "false";
    const char *image = fg_runtime_vision_available(runtime) ? "true" : "false";
    const char *video = fg_runtime_video_available(runtime) ? "true" : "false";
    const char *video_frames = fg_runtime_video_frames_available(runtime) ? "true" : "false";
    api_buffer body = {0};
    fg_status status = buffer_append(&body, "{\"object\":\"list\",\"data\":[{\"id\":", err);
    if (status == FG_OK) status = buffer_append_json_string(&body, model, strlen(model), err);
    if (status == FG_OK) {
        char capabilities[256];
        int length = snprintf(capabilities, sizeof(capabilities),
                              ",\"object\":\"model\",\"created\":0,\"owned_by\":"
                              "\"flash-gordon\",\"capabilities\":{\"native_context\":%u,"
                              "\"experimental_context\":0,\"tools\":true,\"mtp\":%s,"
                              "\"image\":%s,\"video\":%s,\"video_frames\":%s}}]}",
                              fg_runtime_context_limit(runtime), mtp, image, video,
                              video_frames);
        if (length < 0 || (size_t)length >= sizeof(capabilities)) {
            fg_error_set(err, FG_ERR_LIMIT, "model capabilities exceed response buffer");
            status = FG_ERR_LIMIT;
        } else {
            status = buffer_append_n(&body, capabilities, (size_t)length, err);
        }
    }
    if (status == FG_OK)
        status = api_send_response(sink, 200u, "application/json", body.data, body.length,
                                   keep_alive, err);
    free(body.data);
    return status;
}

static fg_status handle_health(api_sink *sink, bool busy, bool keep_alive, fg_error *err) {
    static const char idle_body[] = "{\"status\":\"ok\",\"busy\":false}";
    static const char busy_body[] = "{\"status\":\"ok\",\"busy\":true}";
    const char *body = busy ? busy_body : idle_body;
    return api_send_response(sink, 200u, "application/json", body, strlen(body),
                             keep_alive, err);
}

static fg_status send_busy_response(api_sink *sink, bool keep_alive, fg_error *err) {
    static const char body[] =
        "{\"error\":{\"message\":\"Flash Gordon is shutting down\","
        "\"type\":\"server_busy\"}}";
    return api_send_response_with_headers(sink, 503u, "application/json",
                                          "Retry-After: 1\r\n", body, sizeof(body) - 1u,
                                          keep_alive, err);
}

static fg_status send_queue_full_response(api_sink *sink, bool keep_alive, fg_error *err) {
    static const char body[] =
        "{\"error\":{\"message\":\"Flash Gordon is at its request bound ("
        "one running plus a bounded FIFO queue); retry shortly\","
        "\"type\":\"queue_full\"}}";
    return api_send_response_with_headers(sink, 429u, "application/json",
                                          "Retry-After: 1\r\n", body, sizeof(body) - 1u,
                                          keep_alive, err);
}

/* --- front-end connections, engine queue and the HTTP thread ------------- */

static void api_frontend_wake(api_frontend *frontend) {
    if (!frontend || frontend->wake_write < 0) return;
    char byte = 1;
    ssize_t ignored = write(frontend->wake_write, &byte, 1u);
    (void)ignored;
}

/* Append response bytes under out_mutex.  Chunk framing is applied by the
 * front-end when the connection's response is chunked; the raw path is used
 * for response heads and the terminating zero chunk. */
static fg_status api_connection_append(api_connection *conn, const char *data,
                                       size_t length, bool chunked, fg_error *err) {
    if (!length) return FG_OK;
    pthread_mutex_lock(&conn->out_mutex);
    fg_status status = FG_OK;
    if (conn->client_failed) {
        fg_error_set(err, FG_ERR_IO, "HTTP client is not draining the response");
        status = FG_ERR_IO;
    } else if (conn->out.length + length > FG_API_CONNECTION_OUTPUT_LIMIT) {
        fg_error_set(err, FG_ERR_IO, "HTTP client is not draining the response");
        conn->client_failed = true;
        status = FG_ERR_IO;
    } else {
        if (chunked) {
            char header[32];
            int header_length = snprintf(header, sizeof(header), "%zx\r\n", length);
            if (header_length > 0)
                status = buffer_append_n(&conn->out, header, (size_t)header_length, err);
        }
        if (status == FG_OK) status = buffer_append_n(&conn->out, data, length, err);
        if (status == FG_OK && chunked)
            status = buffer_append_n(&conn->out, "\r\n", 2u, err);
        if (status != FG_OK) conn->client_failed = true;
    }
    if (status == FG_OK) conn->last_activity = api_monotonic_seconds();
    pthread_mutex_unlock(&conn->out_mutex);
    if (status == FG_OK) api_frontend_wake(conn->frontend);
    return status;
}

static fg_status api_connection_enqueue_raw(api_connection *conn, const char *data,
                                            size_t length, fg_error *err) {
    return api_connection_append(conn, data, length, false, err);
}

static fg_status api_sink_write(api_sink *sink, const char *data, size_t length,
                                fg_error *err) {
    if (!length) return FG_OK;
    if (sink->connection)
        return api_connection_append(sink->connection, data, length,
                                     atomic_load(&sink->connection->chunked), err);
    return send_all(sink->fd, data, length, err);
}

/* Engine-side response completion: emit the terminating chunk for a chunked
 * response, publish completion, and let the front-end flush/close/reuse. */
static void api_connection_complete_response(api_connection *conn, bool close_after) {
    if (!conn) return;
    pthread_mutex_lock(&conn->out_mutex);
    if (atomic_load(&conn->chunked) && !conn->client_failed &&
        !atomic_load(&conn->client_gone)) {
        fg_error ignored = {0};
        buffer_append_n(&conn->out, "0\r\n\r\n", 5u, &ignored);
    }
    if (close_after || conn->client_failed || atomic_load(&conn->client_gone))
        atomic_store(&conn->close_after_flush, true);
    pthread_mutex_unlock(&conn->out_mutex);
    atomic_store(&conn->response_complete, true);
    api_frontend_wake(conn->frontend);
}

/* Non-blocking flush of the outbound buffer.  A write failure drops the
 * remaining bytes and marks the client gone; the engine sees that through
 * api_interrupted() and aborts the generation. */
static void api_connection_flush(api_connection *conn) {
    pthread_mutex_lock(&conn->out_mutex);
    while (conn->out.length) {
        ssize_t sent = send(conn->fd, conn->out.data, conn->out.length, MSG_NOSIGNAL);
        if (sent > 0) {
            if ((size_t)sent >= conn->out.length) {
                conn->out.length = 0;
                if (conn->out.data) conn->out.data[0] = 0;
            } else {
                memmove(conn->out.data, conn->out.data + sent,
                        conn->out.length - (size_t)sent);
                conn->out.length -= (size_t)sent;
                conn->out.data[conn->out.length] = 0;
            }
            conn->last_activity = api_monotonic_seconds();
            continue;
        }
        if (sent < 0 && errno == EINTR) continue;
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        conn->out.length = 0;
        if (conn->out.data) conn->out.data[0] = 0;
        atomic_store(&conn->client_gone, true);
        atomic_store(&conn->close_after_flush, true);
        break;
    }
    pthread_mutex_unlock(&conn->out_mutex);
}

/* Front-end heartbeat: only streaming responses get `: keep-alive` comments,
 * and only when nothing else is queued.  This runs off the poll loop, so it
 * covers long prefills and the vision tower without touching the token path. */
static bool api_connection_maybe_heartbeat(api_connection *conn, double now) {
    if (!conn->generating || atomic_load(&conn->client_gone)) return false;
    pthread_mutex_lock(&conn->out_mutex);
    bool due = atomic_load(&conn->chunked) && !conn->client_failed &&
               !atomic_load(&conn->response_complete) && conn->out.length == 0 &&
               now - conn->last_activity >= FG_API_STREAM_KEEPALIVE_SECONDS;
    if (due) {
        static const char comment[] = ": keep-alive\n\n";
        fg_error ignored = {0};
        char header[32];
        int header_length = snprintf(header, sizeof(header), "%zx\r\n",
                                     sizeof(comment) - 1u);
        if (header_length > 0)
            buffer_append_n(&conn->out, header, (size_t)header_length, &ignored);
        buffer_append_n(&conn->out, comment, sizeof(comment) - 1u, &ignored);
        buffer_append_n(&conn->out, "\r\n", 2u, &ignored);
        conn->last_activity = now;
    }
    pthread_mutex_unlock(&conn->out_mutex);
    return due;
}

/* Client disconnect probe for a connection the engine is currently serving. */
static bool api_connection_detect_client_gone(api_connection *conn) {
    if (atomic_load(&conn->client_gone)) return true;
    struct pollfd probe = {.fd = conn->fd, .events = POLLIN | POLLRDHUP};
    int ready = poll(&probe, 1u, 0);
    if (ready <= 0) return false;
    if (probe.revents & (POLLERR | POLLHUP | POLLRDHUP)) {
        atomic_store(&conn->client_gone, true);
        return true;
    }
    if (probe.revents & (POLLIN | POLLRDNORM)) {
        char byte = 0;
        ssize_t peeked = recv(conn->fd, &byte, 1u, MSG_PEEK | MSG_DONTWAIT);
        if (peeked == 0 ||
            (peeked < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
            atomic_store(&conn->client_gone, true);
            return true;
        }
    }
    return false;
}

static void api_engine_queue_init(api_engine_queue *queue) {
    memset(queue, 0, sizeof(*queue));
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->cond, NULL);
}

static void api_engine_queue_destroy(api_engine_queue *queue) {
    pthread_mutex_lock(&queue->mutex);
    for (size_t i = 0; i < queue->count; i++) {
        size_t slot = (queue->head + i) % FG_API_ENGINE_QUEUE_CAPACITY;
        free(queue->entries[slot].http.body);
        queue->entries[slot].http.body = NULL;
    }
    queue->head = 0;
    queue->count = 0;
    queue->outstanding = 0;
    pthread_mutex_unlock(&queue->mutex);
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->cond);
}

/* M2 admission: a chat request is accepted while generations are running; the
 * engine runs admitted requests FIFO.  The bound is `outstanding` (running +
 * queued, not yet completed), not the physical ring occupancy, so the
 * documented capacity is the number of requests a burst can hold. */
static api_engine_admission api_engine_queue_try_push(api_engine_queue *queue,
                                                      api_connection *connection,
                                                      http_request *http) {
    api_engine_admission admission = API_ENGINE_ADMIT_FULL;
    pthread_mutex_lock(&queue->mutex);
    if (queue->stopping) {
        admission = API_ENGINE_ADMIT_STOPPING;
    } else if (queue->outstanding < FG_API_ENGINE_QUEUE_CAPACITY &&
               queue->count < FG_API_ENGINE_QUEUE_CAPACITY) {
        size_t slot = (queue->head + queue->count) % FG_API_ENGINE_QUEUE_CAPACITY;
        queue->entries[slot].connection = connection;
        queue->entries[slot].http = *http;
        queue->count++;
        queue->outstanding++;
        admission = API_ENGINE_ADMIT_QUEUED;
        pthread_cond_signal(&queue->cond);
    }
    pthread_mutex_unlock(&queue->mutex);
    return admission;
}

/* Cancel a queued request whose client disconnected before the engine picked
 * it up (front-end thread only; the connection's `generating` flag is
 * front-end state).  Frees the request body and the admission slot.  The
 * engine's pop-time `client_gone` check covers the race where the request is
 * popped between the disconnect and this sweep. */
static bool api_engine_queue_cancel(api_engine_queue *queue, api_connection *connection) {
    bool canceled = false;
    pthread_mutex_lock(&queue->mutex);
    size_t kept = 0;
    for (size_t i = 0; i < queue->count; i++) {
        size_t slot = (queue->head + i) % FG_API_ENGINE_QUEUE_CAPACITY;
        api_engine_request *entry = &queue->entries[slot];
        if (entry->connection == connection) {
            free(entry->http.body);
            entry->http.body = NULL;
            entry->connection = NULL;
            if (queue->outstanding) queue->outstanding--;
            canceled = true;
            continue;
        }
        if (kept != i) {
            size_t target = (queue->head + kept) % FG_API_ENGINE_QUEUE_CAPACITY;
            queue->entries[target] = *entry;
        }
        kept++;
    }
    queue->count = kept;
    pthread_mutex_unlock(&queue->mutex);
    return canceled;
}

static bool api_engine_queue_pop_wait(api_engine_queue *queue, api_engine_request *out,
                                      int timeout_ms) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += (long)timeout_ms * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += deadline.tv_nsec / 1000000000L;
        deadline.tv_nsec %= 1000000000L;
    }
    pthread_mutex_lock(&queue->mutex);
    while (!queue->count && !queue->stopping) {
        if (pthread_cond_timedwait(&queue->cond, &queue->mutex, &deadline) == ETIMEDOUT)
            break;
    }
    bool popped = false;
    if (queue->count) {
        *out = queue->entries[queue->head];
        queue->head = (queue->head + 1) % FG_API_ENGINE_QUEUE_CAPACITY;
        queue->count--;
        popped = true;
    }
    pthread_mutex_unlock(&queue->mutex);
    return popped;
}

static void api_engine_queue_complete(api_engine_queue *queue) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->outstanding) queue->outstanding--;
    pthread_mutex_unlock(&queue->mutex);
}

static bool api_engine_queue_busy(api_engine_queue *queue) {
    pthread_mutex_lock(&queue->mutex);
    bool busy = queue->outstanding != 0;
    pthread_mutex_unlock(&queue->mutex);
    return busy;
}

static void api_engine_queue_stop(api_engine_queue *queue) {
    pthread_mutex_lock(&queue->mutex);
    queue->stopping = true;
    pthread_cond_broadcast(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
}

static api_connection *api_frontend_connection_create(api_frontend *frontend, int fd) {
    api_connection *conn = calloc(1, sizeof(*conn));
    if (!conn) return NULL;
    conn->fd = fd;
    conn->frontend = frontend;
    pthread_mutex_init(&conn->out_mutex, NULL);
    atomic_init(&conn->client_gone, false);
    atomic_init(&conn->chunked, false);
    atomic_init(&conn->response_complete, false);
    atomic_init(&conn->close_after_flush, false);
    conn->last_activity = api_monotonic_seconds();
    conn->next = frontend->connections;
    frontend->connections = conn;
    frontend->connection_count++;
    return conn;
}

static void api_connection_free(api_connection *conn) {
    if (!conn) return;
    free(conn->parser.input.data);
    free(conn->out.data);
    pthread_mutex_destroy(&conn->out_mutex);
    free(conn);
}

static void api_frontend_accept(api_frontend *frontend) {
    for (;;) {
        int client = accept(frontend->listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (frontend->connection_count >= FG_API_MAX_CONNECTIONS) {
            close(client);
            continue;
        }
        fg_error err = {0};
        if (configure_client_socket(client, &err) != FG_OK) {
            close(client);
            continue;
        }
        if (!api_frontend_connection_create(frontend, client)) close(client);
    }
}

static void api_frontend_read_available(api_frontend *frontend, api_connection *conn) {
    for (;;) {
        fg_error err = {0};
        if (conn->parser.input.length >= FG_API_MAX_REQUEST_BYTES) break;
        size_t chunk = FG_API_MAX_REQUEST_BYTES - conn->parser.input.length;
        if (chunk > 8192u) chunk = 8192u;
        if (buffer_reserve(&conn->parser.input, chunk, &err) != FG_OK) {
            atomic_store(&conn->client_gone, true);
            atomic_store(&conn->close_after_flush, true);
            return;
        }
        ssize_t received = recv(conn->fd,
                                conn->parser.input.data + conn->parser.input.length,
                                chunk, 0);
        if (received > 0) {
            conn->parser.input.length += (size_t)received;
            conn->parser.input.data[conn->parser.input.length] = 0;
            conn->last_activity = api_monotonic_seconds();
            continue;
        }
        if (!received) {
            /* Peer closed its write side: serve any complete request already
             * buffered, then close after the response. */
            conn->peer_closed = true;
            conn->keep_alive = false;
            break;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        atomic_store(&conn->client_gone, true);
        atomic_store(&conn->close_after_flush, true);
        return;
    }
    for (;;) {
        if (conn->generating || atomic_load(&conn->client_gone) ||
            atomic_load(&conn->response_complete))
            break;
        http_request http = {0};
        bool keep_alive = true;
        size_t consumed = 0u;
        unsigned http_status = 400u;
        fg_error err = {0};
        api_parse_state state = api_http_parser_try(&conn->parser, &http, &keep_alive,
                                                    &consumed, &http_status, &err);
        if (state == API_PARSE_INCOMPLETE) break;
        if (state == API_PARSE_FAILED) {
            api_sink sink = {.fd = conn->fd, .connection = conn};
            fg_error send_err = {0};
            api_send_error_response(&sink, http_status,
                                    err.message[0] ? err.message : "invalid HTTP request",
                                    false, &send_err);
            free(http.body);
            api_connection_complete_response(conn, true);
            break;
        }
        api_http_parser_consume(&conn->parser, consumed);
        if (api_frontend_dispatch(frontend, conn, &http, keep_alive)) break;
    }
}

static bool api_frontend_dispatch(api_frontend *frontend, api_connection *conn,
                                  http_request *http, bool keep_alive) {
    api_sink sink = {.fd = conn->fd, .connection = conn};
    conn->keep_alive = keep_alive && !conn->peer_closed;
    fg_error err = {0};
    if (!strcmp(http->method, "GET") && !strcmp(http->path, "/v1/models")) {
        handle_models(&sink, frontend->runtime, conn->keep_alive, &err);
        api_connection_complete_response(conn, !conn->keep_alive);
    } else if (!strcmp(http->method, "GET") && !strcmp(http->path, "/health")) {
        handle_health(&sink, api_engine_queue_busy(frontend->engine), conn->keep_alive, &err);
        api_connection_complete_response(conn, !conn->keep_alive);
    } else if (!strcmp(http->method, "POST") &&
               !strcmp(http->path, "/v1/chat/completions")) {
        /* Ownership of the request body moves to the engine queue on success. */
        api_engine_admission admission =
            api_engine_queue_try_push(frontend->engine, conn, http);
        if (admission == API_ENGINE_ADMIT_QUEUED) {
            conn->generating = true;
            return true;
        }
        if (admission == API_ENGINE_ADMIT_FULL)
            send_queue_full_response(&sink, conn->keep_alive, &err);
        else
            send_busy_response(&sink, conn->keep_alive, &err);
        api_connection_complete_response(conn, !conn->keep_alive);
    } else if (!strcmp(http->path, "/v1/models") ||
               !strcmp(http->path, "/v1/chat/completions")) {
        api_send_error_response(&sink, 405u, "method not allowed", conn->keep_alive, &err);
        api_connection_complete_response(conn, !conn->keep_alive);
    } else {
        api_send_error_response(&sink, 404u, "not found", conn->keep_alive, &err);
        api_connection_complete_response(conn, !conn->keep_alive);
    }
    free(http->body);
    http->body = NULL;
    http->body_length = 0u;
    return false;
}

static bool api_connection_reusable(api_connection *conn) {
    return conn->keep_alive && !atomic_load(&conn->client_gone) &&
           !atomic_load(&conn->close_after_flush) && !conn->peer_closed;
}

/* Called from the front-end loop only.  Frees connections whose response has
 * flushed and that cannot serve another request; resets the rest for the next
 * request on the same connection. */
static void api_frontend_reap(api_frontend *frontend) {
    double now = api_monotonic_seconds();
    /* Cancel queued requests whose client disconnected before the engine
     * picked them up; the pop-time check covers the race.  Completion here is
     * safe because `generating` and `client_gone` are both front-end state. */
    for (api_connection *conn = frontend->connections; conn; conn = conn->next) {
        if (!conn->generating || atomic_load(&conn->response_complete)) continue;
        if (!atomic_load(&conn->client_gone)) continue;
        if (api_engine_queue_cancel(frontend->engine, conn))
            api_connection_complete_response(conn, true);
    }
    api_connection **cursor = &frontend->connections;
    while (*cursor) {
        api_connection *conn = *cursor;
        bool done = false;
        pthread_mutex_lock(&conn->out_mutex);
        bool flushed = atomic_load(&conn->response_complete) && conn->out.length == 0;
        pthread_mutex_unlock(&conn->out_mutex);
        if (conn->generating) {
            if (!flushed) {
                cursor = &conn->next;
                continue;
            }
            conn->generating = false; /* engine finished; front-end owns it again */
        }
        if (flushed) {
            if (api_connection_reusable(conn)) {
                pthread_mutex_lock(&conn->out_mutex);
                atomic_store(&conn->chunked, false);
                atomic_store(&conn->close_after_flush, false);
                conn->client_failed = false;
                pthread_mutex_unlock(&conn->out_mutex);
                atomic_store(&conn->response_complete, false);
                conn->peer_closed = false;
            } else {
                done = true;
            }
        } else if (atomic_load(&conn->client_gone)) {
            done = true;
        } else if (conn->peer_closed && conn->parser.input.length == 0) {
            done = true;
        } else if (now - conn->last_activity > FG_API_CONNECTION_IDLE_SECONDS) {
            done = true;
        }
        if (done) {
            *cursor = conn->next;
            frontend->connection_count--;
            close(conn->fd);
            api_connection_free(conn);
        } else {
            cursor = &conn->next;
        }
    }
}

static void *api_frontend_thread(void *context) {
    api_frontend *frontend = context;
    struct pollfd fds[2 + FG_API_MAX_CONNECTIONS];
    api_connection *owners[2 + FG_API_MAX_CONNECTIONS];
    while (!atomic_load(&frontend->stopping)) {
        size_t count = 0;
        fds[count] = (struct pollfd){.fd = frontend->wake_read, .events = POLLIN};
        owners[count++] = NULL;
        fds[count] = (struct pollfd){.fd = frontend->listener, .events = POLLIN};
        owners[count++] = NULL;
        for (api_connection *conn = frontend->connections; conn; conn = conn->next) {
            if (count >= sizeof(fds) / sizeof(fds[0])) break;
            short events = 0;
            if (conn->generating) {
                /* Only disconnect detection while the engine owns the
                 * response; pipelined bytes are read after it completes. */
                events |= POLLRDHUP;
            } else if (!atomic_load(&conn->response_complete)) {
                events |= POLLIN | POLLRDHUP;
            }
            pthread_mutex_lock(&conn->out_mutex);
            bool has_output = conn->out.length != 0;
            pthread_mutex_unlock(&conn->out_mutex);
            if (has_output && !atomic_load(&conn->client_gone)) events |= POLLOUT;
            fds[count] = (struct pollfd){.fd = conn->fd, .events = events};
            owners[count++] = conn;
        }
        int ready = poll(fds, (nfds_t)count, FG_API_FRONTEND_POLL_MS);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (fds[0].revents & POLLIN) {
            char discard[256];
            while (read(frontend->wake_read, discard, sizeof(discard)) > 0) {}
        }
        if (atomic_load(&frontend->stopping)) break;
        if (fds[1].revents & (POLLIN | POLLERR)) api_frontend_accept(frontend);
        for (size_t i = 2; i < count; i++) {
            api_connection *conn = owners[i];
            if (!conn) continue;
            short revents = fds[i].revents;
            if (!revents) continue;
            if (conn->generating) {
                if (revents & (POLLERR | POLLHUP | POLLRDHUP))
                    api_connection_detect_client_gone(conn);
            } else if (revents & (POLLIN | POLLERR | POLLHUP | POLLRDHUP)) {
                api_frontend_read_available(frontend, conn);
            }
            if (revents & POLLOUT) api_connection_flush(conn);
        }
        double now = api_monotonic_seconds();
        for (api_connection *conn = frontend->connections; conn; conn = conn->next)
            api_connection_maybe_heartbeat(conn, now);
        api_frontend_reap(frontend);
    }
    api_connection *conn = frontend->connections;
    while (conn) {
        api_connection *next = conn->next;
        close(conn->fd);
        api_connection_free(conn);
        conn = next;
    }
    frontend->connections = NULL;
    frontend->connection_count = 0;
    return NULL;
}

static fg_status api_frontend_start(api_frontend *frontend, int listener,
                                    fg_runtime *runtime, api_engine_queue *engine,
                                    fg_error *err) {
    memset(frontend, 0, sizeof(*frontend));
    frontend->listener = listener;
    frontend->wake_read = -1;
    frontend->wake_write = -1;
    frontend->runtime = runtime;
    frontend->engine = engine;
    int listener_flags = fcntl(listener, F_GETFL, 0);
    if (listener_flags < 0 ||
        fcntl(listener, F_SETFL, listener_flags | O_NONBLOCK) != 0) {
        fg_error_set(err, FG_ERR_IO, "configure API listener non-blocking mode: %s",
                     strerror(errno));
        return FG_ERR_IO;
    }
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) {
        fg_error_set(err, FG_ERR_IO, "create API front-end wake pipe: %s",
                     strerror(errno));
        return FG_ERR_IO;
    }
    frontend->wake_read = pipe_fds[0];
    frontend->wake_write = pipe_fds[1];
    /* Non-blocking both ends: the drain loop must never block on an empty
     * pipe, and a full pipe must never stall the engine. */
    int wake_read_flags = fcntl(frontend->wake_read, F_GETFL, 0);
    int wake_write_flags = fcntl(frontend->wake_write, F_GETFL, 0);
    if (wake_read_flags < 0 || wake_write_flags < 0 ||
        fcntl(frontend->wake_read, F_SETFL, wake_read_flags | O_NONBLOCK) != 0 ||
        fcntl(frontend->wake_write, F_SETFL, wake_write_flags | O_NONBLOCK) != 0) {
        fg_error_set(err, FG_ERR_IO, "configure API front-end wake pipe: %s",
                     strerror(errno));
        close(frontend->wake_read);
        close(frontend->wake_write);
        frontend->wake_read = frontend->wake_write = -1;
        return FG_ERR_IO;
    }
    api_wake_fd = frontend->wake_write;
    if (pthread_create(&frontend->thread, NULL, api_frontend_thread, frontend) != 0) {
        fg_error_set(err, FG_ERR_IO, "start API front-end thread: %s", strerror(errno));
        close(frontend->wake_read);
        close(frontend->wake_write);
        frontend->wake_read = frontend->wake_write = -1;
        api_wake_fd = -1;
        return FG_ERR_IO;
    }
    frontend->thread_started = true;
    return FG_OK;
}

static void api_frontend_stop(api_frontend *frontend) {
    if (!frontend) return;
    atomic_store(&frontend->stopping, true);
    api_frontend_wake(frontend);
    if (frontend->thread_started) {
        pthread_join(frontend->thread, NULL);
        frontend->thread_started = false;
    }
    api_wake_fd = -1;
    if (frontend->wake_read >= 0) close(frontend->wake_read);
    if (frontend->wake_write >= 0) close(frontend->wake_write);
    frontend->wake_read = frontend->wake_write = -1;
}

static bool api_session_id_valid(const char *id) {
    if (!id || !id[0]) return false;
    for (const unsigned char *cursor = (const unsigned char *)id; *cursor; cursor++) {
        unsigned char c = *cursor;
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == ':' || c == '-';
        if (!ok) return false;
    }
    return true;
}

static void api_session_id_generate(api_session_table *table,char output[48]) {
    uint64_t sequence = ++table->sequence;
    snprintf(output,48u,"fg-%08llx-%06llx",
             (unsigned long long)(uint64_t)time(NULL),
             (unsigned long long)(sequence & 0xffffffu));
}

static void api_session_table_release_entry(api_session_table *table,fg_runtime *runtime,
                                            api_session_entry *entry) {
    if (!table || !entry || !entry->in_use) return;
    api_public_session_free(&entry->session);
    fg_runtime_session_release(runtime,entry->runtime_session);
    memset(entry,0,sizeof(*entry));
}

/* Evict the least-recently-used idle session entry; `slots_only` restricts the
 * search to entries that actually hold an owner state slot.  Busy entries are
 * live turns and always survive. */
static api_session_entry *api_session_table_evict_idle(api_session_table *table,
                                                       fg_runtime *runtime,
                                                       bool slots_only) {
    api_session_entry *candidate = NULL;
    for (size_t i = 0; i < FG_API_SESSION_MAX; i++) {
        api_session_entry *entry = &table->entries[i];
        if (!entry->in_use || entry->busy) continue;
        if (slots_only && fg_runtime_session_slot(entry->runtime_session) == UINT32_MAX)
            continue;
        if (!candidate || entry->last_used < candidate->last_used) candidate = entry;
    }
    if (candidate) api_session_table_release_entry(table, runtime, candidate);
    return candidate;
}

static void api_session_table_drop(api_session_table *table,fg_runtime *runtime,
                                  api_session_entry *entry) {
    api_session_table_release_entry(table, runtime, entry);
}

static api_session_entry *api_session_table_new(api_session_table *table,fg_runtime *runtime,
                                                api_connection *conn,const char *id,
                                                fg_error *err) {
    /* Owner state slots are the admission currency: an idle session that still
     * holds one is evicted before a new session is created, so an aborted turn
     * frees its slot for the next request. */
    while (fg_runtime_owner_slots_free(runtime) == 0u)
        if (!api_session_table_evict_idle(table, runtime, true)) break;
    if (fg_runtime_owner_slots_free(runtime) == 0u) {
        fg_error_set(err,FG_ERR_LIMIT,"all owner state slots are busy");
        return NULL;
    }
    api_session_entry *entry = NULL;
    for (size_t i = 0; i < FG_API_SESSION_MAX; i++) {
        if (!table->entries[i].in_use) { entry = &table->entries[i]; break; }
    }
    if (!entry) {
        entry = api_session_table_evict_idle(table, runtime, false);
        if (!entry) {
            fg_error_set(err,FG_ERR_LIMIT,"no free API session slot");
            return NULL;
        }
    }
    entry->runtime_session = fg_runtime_session_acquire(runtime, 0);
    if (!entry->runtime_session) {
        fg_error_set(err,FG_ERR_LIMIT,"no free runtime session slot");
        return NULL;
    }
    if (id && id[0])
        snprintf(entry->session.id,sizeof(entry->session.id),"%s",id);
    else
        api_session_id_generate(table,entry->session.id);
    entry->session.numeric_id = fg_runtime_session_id(entry->runtime_session);
    entry->owner = conn;
    entry->in_use = true;
    entry->last_used = ++table->sequence;
    return entry;
}

static fg_status api_session_table_resolve(api_session_table *table,fg_runtime *runtime,
                                           api_connection *conn,
                                           const api_chat_request *request,
                                           api_session_entry **entry_out,fg_error *err) {
    *entry_out = NULL;
    if (!table || !runtime) {
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid session table arguments");
        return FG_ERR_ARGUMENT;
    }
    api_session_entry *entry = NULL;
    if (request->session_id_set) {
        for (size_t i = 0; i < FG_API_SESSION_MAX; i++) {
            api_session_entry *candidate = &table->entries[i];
            if (candidate->in_use && !strcmp(candidate->session.id, request->session_id)) {
                entry = candidate;
                break;
            }
        }
        if (!entry) entry = api_session_table_new(table,runtime,conn,request->session_id,err);
    } else {
        /* Default: the connection's session (keep-alive continuation). */
        for (size_t i = 0; i < FG_API_SESSION_MAX; i++) {
            api_session_entry *candidate = &table->entries[i];
            if (candidate->in_use && candidate->owner == conn) { entry = candidate; break; }
        }
        /* Preserve the pre-M3 cross-connection behaviour: a request that is a
         * strict prefix extension of the most recent session continues it. */
        if (!entry) {
            api_session_entry *recent = NULL;
            for (size_t i = 0; i < FG_API_SESSION_MAX; i++) {
                api_session_entry *candidate = &table->entries[i];
                if (!candidate->in_use) continue;
                if (!recent || candidate->last_used > recent->last_used) recent = candidate;
            }
            if (recent &&
                (!recent->session.valid ||
                 api_public_session_prefix(&recent->session,request,NULL,0u)))
                entry = recent;
        }
        if (!entry) entry = api_session_table_new(table,runtime,conn,NULL,err);
    }
    if (!entry) return err->code ? err->code : FG_ERR_LIMIT;
    entry->owner = conn;
    entry->last_used = ++table->sequence;
    *entry_out = entry;
    return FG_OK;
}

static void api_session_table_destroy(api_session_table *table,fg_runtime *runtime) {
    if (!table) return;
    for (size_t i = 0; i < FG_API_SESSION_MAX; i++) {
        api_session_entry *entry = &table->entries[i];
        if (!entry->in_use) continue;
        api_public_session_free(&entry->session);
        fg_runtime_session_release(runtime,entry->runtime_session);
        memset(entry,0,sizeof(*entry));
    }
}

/* M3.2 parked turn: one in-flight chat request whose model phase can be
 * suspended at a prefill chunk or decode step boundary and resumed without
 * finalising the HTTP response.  The synchronous legacy path builds the same
 * struct, so the request head (parse, session resolve, render, response open)
 * and tail (generated parse, public-session commit, response close, slot
 * release) exist exactly once. */
typedef struct api_turn {
    bool opened;              /* head complete: response may be open */
    bool started;             /* model phase started (runner or legacy) */
    bool left;                /* runner reported the turn finished */
    bool prefill_done;        /* whole prompt prefilled */
    bool stream_started;
    bool generation_attempted;
    bool public_continuation;
    bool eligible;            /* multiplex-eligible (text-only, greedy, cold) */
    bool owns_session;        /* resolved through the live session table */
    api_session_entry *entry;
    api_session_table *sessions;
    api_public_session *public_session;
    api_connection *connection;
    api_sink sink;
    http_request http;        /* body ownership stays with the queue entry */
    api_chat_request request;
    char *rendered;
    char *rendered_continuation;
    fg_chat_render_options render_options;
    api_generation generation;
    fg_generation_stats stats;
    char id[96];
} api_turn;

static void api_turn_init(api_turn *turn) {
    memset(turn, 0, sizeof(*turn));
}

static void api_turn_dispose(api_turn *turn) {
    api_chat_request_free(&turn->request);
    free(turn->rendered);
    free(turn->rendered_continuation);
    free(turn->generation.content.data);
    free(turn->generation.visible_pending.data);
    memset(turn, 0, sizeof(*turn));
}

/* Head: parse the request, resolve and bind its session, render the transcript,
 * open the response.  Returns FG_OK when the caller should proceed (opened) or
 * when an error response was already sent (`opened` stays false); a non-OK
 * return is an engine-visible failure after the client was answered. */
static fg_status api_turn_open(api_turn *turn, api_sink sink, fg_runtime *runtime,
                               api_public_session *public_session,
                               api_session_table *sessions, const http_request *http,
                               fg_error *err) {
    memset(err, 0, sizeof(*err));
    api_turn_init(turn);
    turn->sink = sink;
    turn->connection = sink.connection;
    turn->http = *http;
    turn->sessions = sessions;
    bool keep_alive = sink.connection ? sink.connection->keep_alive : false;
    json_value *root = parse_json_body(http->body, http->body_length, err);
    if (!root) {
        char message[sizeof(err->message)];
        snprintf(message, sizeof(message), "%s", err->message);
        fg_error send_err = {0};
        api_send_error_response(&turn->sink, 400u, message, keep_alive, &send_err);
        return FG_OK;
    }
    fg_status status =
        parse_chat_request(root, fg_runtime_model_name(runtime), &turn->request, err);
    json_free(root);
    if (status == FG_OK && turn->request.media_count) {
        bool needs_video = false;
        bool needs_frames = false;
        for (size_t i = 0; i < turn->request.media_count; i++) {
            if (turn->request.media[i].kind == FG_RUNTIME_MEDIA_VIDEO) needs_video = true;
            if (turn->request.media[i].kind == FG_RUNTIME_MEDIA_VIDEO_FRAMES) needs_frames = true;
        }
        const char *message = NULL;
        if (!fg_runtime_vision_available(runtime))
            message = "media input is not available on this deployment "
                      "(vision tower pack missing)";
        else if (needs_frames && !fg_runtime_video_frames_available(runtime))
            message = "video frame input is not available on this deployment "
                      "(tower temporal token entry missing)";
        else if (needs_video && !fg_runtime_video_available(runtime))
            message = "MP4 video input is not available on this deployment "
                      "(static ffmpeg/ffprobe or the tower temporal token entry is missing)";
        if (message) {
            fg_error send_err = {0};
            api_send_error_response(&turn->sink, 400u, message, keep_alive, &send_err);
            return FG_OK;
        }
    }
    if (status != FG_OK) {
        char message[sizeof(err->message)];
        snprintf(message, sizeof(message), "%s", err->message);
        fg_error send_err = {0};
        api_send_error_response(&turn->sink, 400u, message, keep_alive, &send_err);
        return FG_OK;
    }

    /* M3: bind the request to its live session.  Tests call the legacy path
     * with `sessions == NULL` and their own `public_session`. */
    if (sessions) {
        fg_status resolve = api_session_table_resolve(sessions, runtime, sink.connection,
                                                      &turn->request, &turn->entry, err);
        if (resolve != FG_OK) {
            char message[sizeof(err->message)];
            snprintf(message, sizeof(message), "%s", err->message);
            fg_error send_err = {0};
            api_send_error_response(&turn->sink, 503u, message, keep_alive, &send_err);
            return FG_OK;
        }
        public_session = &turn->entry->session;
        turn->owns_session = true;
        turn->entry->busy = true;
        fg_status begin = fg_runtime_session_begin(runtime, turn->entry->runtime_session, err);
        if (begin != FG_OK) {
            char message[sizeof(err->message)];
            snprintf(message, sizeof(message), "%s", err->message);
            fg_error send_err = {0};
            turn->entry->busy = false;
            api_send_error_response(&turn->sink, 500u, message, keep_alive, &send_err);
            return begin;
        }
        /* The head only needs the session's cold start / adoption to have run;
         * the runtime binding is taken per scheduler step, so a second parked
         * turn can be admitted without touching the first. */
        fg_runtime_session_end(runtime, turn->entry->runtime_session);
    }
    turn->public_session = public_session;
    turn->render_options = (fg_chat_render_options){
        .tool_schemas = (const char *const *)turn->request.tool_schemas,
        .tool_schema_count = turn->request.tool_schema_count,
        .tool_choice = turn->request.tool_choice,
        .tool_choice_name = turn->request.tool_choice_name,
    };
    /* Honor the /no_think directive: render an already-closed think block and
     * drop the directive from the user text, so short answers arrive without
     * spending the completion budget on stripped reasoning tokens. */
    for (size_t i = 0; i < turn->request.message_count; i++) {
        if (!turn->request.messages[i].content ||
            strcmp(turn->request.messages[i].role, "user") != 0) continue;
        char *content = (char *)turn->request.messages[i].content;
        size_t offset = 0;
        while (content[offset] == ' ' || content[offset] == '\n' ||
               content[offset] == '\t' || content[offset] == '\r') offset++;
        if (strncmp(content + offset, "/no_think", 9u) != 0) continue;
        turn->render_options.think_mode = FG_CHAT_THINK_OFF;
        size_t rest = offset + 9u;
        while (content[rest] == ' ' || content[rest] == '\n' ||
               content[rest] == '\t' || content[rest] == '\r') rest++;
        memmove(content, content + rest, strlen(content + rest) + 1u);
    }
    status = fg_chat_render(turn->request.messages, turn->request.message_count,
                            &turn->render_options, &turn->rendered, err);
    char session_mismatch[512]={0};
    turn->public_continuation=status==FG_OK&&
        api_public_session_prefix(public_session,&turn->request,session_mismatch,
                                  sizeof(session_mismatch));
    if(status==FG_OK&&public_session->valid&&!turn->public_continuation){
        if(session_mismatch[0])
            fprintf(stderr,"SESSION_MISMATCH %s\n",session_mismatch);
        api_public_session_free(public_session);
        status=turn->owns_session?
            fg_runtime_reset_public_history_session(
                runtime,turn->entry->runtime_session,err):
            fg_runtime_reset_public_history(runtime,err);
    }
    if(turn->public_continuation){
        size_t previous=public_session->transcript.message_count;
        size_t previous_system=fg_chat_leading_system_count(
            public_session->transcript.messages,previous);
        size_t current_system=fg_chat_leading_system_count(turn->request.messages,
                                                           turn->request.message_count);
        size_t skip=previous;
        if(current_system>=previous_system)skip+=current_system-previous_system;
        else skip-=previous_system-current_system;
        status=fg_chat_render_continuation(turn->request.messages+skip,
                                           turn->request.message_count-skip,
                                           &turn->render_options,&turn->rendered_continuation,err);
        if(status==FG_OK){
            fg_chat_render_options previous_options={
                .tool_schemas=(const char *const *)public_session->transcript.tool_schemas,
                .tool_schema_count=public_session->transcript.tool_schema_count,
                .tool_choice=public_session->transcript.tool_choice,
                .tool_choice_name=public_session->transcript.tool_choice_name,
            };
            char *tool_update=NULL;
            status=fg_chat_render_tool_update(&previous_options,&turn->render_options,
                                              &tool_update,err);
            if(status==FG_OK&&tool_update){
                size_t update_length=strlen(tool_update);
                size_t continuation_length=strlen(turn->rendered_continuation);
                char *combined=malloc(update_length+continuation_length+1u);
                if(!combined){
                    fg_error_set(err,FG_ERR_OOM,"allocate tool update continuation");
                    status=FG_ERR_OOM;
                }else{
                    memcpy(combined,tool_update,update_length);
                    memcpy(combined+update_length,turn->rendered_continuation,
                           continuation_length+1u);
                    free(turn->rendered_continuation);
                    turn->rendered_continuation=combined;
                }
            }
            free(tool_update);
        }
        if(status==FG_OK){
            char *system_update=NULL;
            status=fg_chat_render_system_update(public_session->transcript.messages,previous,
                                                turn->request.messages,turn->request.message_count,
                                                &system_update,err);
            if(status==FG_OK&&system_update){
                size_t update_length=strlen(system_update);
                size_t continuation_length=strlen(turn->rendered_continuation);
                char *combined=malloc(update_length+continuation_length+1u);
                if(!combined){
                    fg_error_set(err,FG_ERR_OOM,"allocate system update continuation");
                    status=FG_ERR_OOM;
                }else{
                    memcpy(combined,system_update,update_length);
                    memcpy(combined+update_length,turn->rendered_continuation,
                           continuation_length+1u);
                    free(turn->rendered_continuation);
                    turn->rendered_continuation=combined;
                }
            }
            free(system_update);
        }
    }
    snprintf(turn->id, sizeof(turn->id), "chatcmpl-fg-%lld-%llu", (long long)time(NULL),
             ++api_request_sequence);
    turn->generation = (api_generation){
        .sink = sink,
        .stream = turn->request.stream,
        .keep_alive = keep_alive,
        .id = turn->id,
        .model = fg_runtime_model_name(runtime),
        .created = time(NULL),
        .request = &turn->request,
        .think_closed = turn->render_options.think_mode == FG_CHAT_THINK_OFF,
        .runtime = runtime,
        .session_id = public_session ? public_session->id : NULL,
    };
    if (status == FG_OK && turn->request.stream) {
        status = api_send_sse_headers(&turn->sink, turn->generation.session_id, err);
        if (status == FG_OK){turn->stream_started=true;status = send_stream_start(&turn->generation, err);}
        if (status != FG_OK) turn->generation.client_failed = true;
    }
    if(status==FG_OK)status=fg_runtime_set_sampler(runtime,&turn->request.sampler,err);
    if(status!=FG_OK)return status;
    turn->opened=true;
    return FG_OK;
}

/* The pre-M3.2 synchronous model call: vision, continuation and plain
 * generate.  Used by the legacy engine path and by every request the
 * scheduler cannot multiplex (media, continuation, non-greedy sampler). */
static fg_status api_turn_run_legacy(api_turn *turn, fg_error *err) {
    api_generation *generation = &turn->generation;
    fg_runtime *runtime = generation->runtime;
    api_public_session *public_session = turn->public_session;
    fg_status status = FG_OK;
    turn->generation_attempted = true;
    fg_runtime_media *media=NULL;
    if(turn->request.media_count){
        media=calloc(turn->request.media_count,sizeof(*media));
        if(!media){
            fg_error_set(err,FG_ERR_OOM,"allocate vision request media");
            status=FG_ERR_OOM;
        }else{
            for(size_t item=0;item<turn->request.media_count;item++){
                media[item].kind=turn->request.media[item].kind;
                media[item].bytes=turn->request.media[item].data;
                media[item].length=turn->request.media[item].length;
                media[item].frames=(const uint8_t *const *)turn->request.media[item].frames;
                media[item].frame_lengths=turn->request.media[item].frame_lengths;
                media[item].frame_count=(uint32_t)turn->request.media[item].frame_count;
                media[item].fps=turn->request.media[item].fps;
                media[item].max_frames=turn->request.media[item].max_frames;
            }
        }
    }
    /* The media identities of the stored session are the first `prefix_media`
     * entries of the request media list (the prefix compare validated both
     * their markers and their payload digests).  Only the entries after that
     * are new suffix media and need tower work. */
    size_t prefix_media=turn->public_continuation?public_session->media_count:0u;
    if(status==FG_OK&&turn->public_continuation&&turn->request.media_count&&
       prefix_media<turn->request.media_count){
        bool prefix_miss=false;
        status=fg_runtime_generate_vision_continuation(
            runtime,turn->rendered,turn->rendered_continuation,media+prefix_media,
            (uint32_t)(turn->request.media_count-prefix_media),&prefix_miss,
            turn->request.max_tokens,api_token,generation,api_interrupted,generation,
            &turn->stats,err);
        if(status==FG_ERR_UNAVAILABLE&&prefix_miss){
            memset(&turn->stats,0,sizeof(turn->stats));
            memset(err,0,sizeof(*err));
            api_public_session_free(public_session);
            status=fg_runtime_reset(runtime,err);
            if(status==FG_OK)
                status=fg_runtime_generate_vision(runtime,turn->rendered,media,
                                                  (uint32_t)turn->request.media_count,
                                                  turn->request.max_tokens,api_token,
                                                  generation,api_interrupted,
                                                  generation,&turn->stats,err);
        }
    }else if(status==FG_OK&&turn->request.media_count&&!turn->public_continuation){
        status=fg_runtime_generate_vision(runtime,turn->rendered,media,
                                          (uint32_t)turn->request.media_count,
                                          turn->request.max_tokens,api_token,generation,
                                          api_interrupted,generation,&turn->stats,err);
    }else if(status==FG_OK&&turn->public_continuation){
        bool prefix_miss=false;
        status=fg_runtime_generate_continuation(
            runtime,turn->rendered,turn->rendered_continuation,&prefix_miss,
            turn->request.max_tokens,api_token,generation,api_interrupted,generation,
            &turn->stats,err);
        if(status==FG_ERR_UNAVAILABLE&&prefix_miss){
            memset(&turn->stats,0,sizeof(turn->stats));
            memset(err,0,sizeof(*err));
            api_public_session_free(public_session);
            status=fg_runtime_reset(runtime,err);
            if(status==FG_OK&&turn->request.media_count)
                /* All media are in the prefix; the cold fallback still has
                 * to run the tower for them. */
                status=fg_runtime_generate_vision(runtime,turn->rendered,media,
                                                  (uint32_t)turn->request.media_count,
                                                  turn->request.max_tokens,api_token,
                                                  generation,api_interrupted,
                                                  generation,&turn->stats,err);
            else if(status==FG_OK)
                status=fg_runtime_generate(runtime,turn->rendered,turn->request.max_tokens,
                                           api_token,generation,api_interrupted,generation,
                                           &turn->stats,err);
        }
    }else if(status==FG_OK){
        status=fg_runtime_generate(runtime,turn->rendered,turn->request.max_tokens,api_token,
                                   generation,api_interrupted,generation,&turn->stats,err);
    }
    free(media);
    return status;
}

/* Tail: parse the generated text, commit the public session, close the
 * response, release the runtime session.  Also the sole cleanup point for a
 * turn whose head already answered the client. */
static fg_status api_turn_close(api_turn *turn, fg_status status, fg_error *err) {
    if (!turn->opened) {
        api_turn_dispose(turn);
        return status;
    }
    api_generation *generation = &turn->generation;
    fg_runtime *runtime = generation->runtime;
    api_public_session *public_session = turn->public_session;
    fg_chat_generated generated={0};
    if(status==FG_OK)
        status=fg_chat_parse_generated(generation->content.data?generation->content.data:"",
                                       turn->render_options.think_mode != FG_CHAT_THINK_OFF,
                                       &generated,err);
    if(status==FG_OK)status=validate_generated_tools(&turn->request,&generated,err);
    api_public_session pending_session={0};
    if(status==FG_OK)
        status=api_public_session_build(&pending_session,&turn->request,generation,&generated,
                                        public_session?public_session->id:NULL,
                                        public_session?public_session->numeric_id:0u,err);
    if(status==FG_OK){
        double prefill_tps=turn->stats.prefill_seconds>0.0?
            (double)turn->stats.prefilled_tokens/turn->stats.prefill_seconds:0.0;
        double decode_tps=turn->stats.decode_seconds>0.0?
            (double)turn->stats.generated_tokens/turn->stats.decode_seconds:0.0;
        fprintf(stderr,
                "request %s: mode %s, prefix %s, reused %u, reset %s, "
                "prefill %u/%u tokens "
                "%.2f tok/s, generation %u tokens %.2f tok/s, context %u/%u "
                "media %u image-tokens %u video-tokens %u video-frames %u tower %.2f s\n",
                turn->id,fg_execution_mode_name(turn->stats.execution_mode),
                turn->stats.prefix_cache_hit?"hit":"miss",turn->stats.reused_tokens,
                fg_prefix_reset_reason_name(turn->stats.reset_reason),
                turn->stats.prefilled_tokens,turn->stats.prompt_tokens,prefill_tps,
                turn->stats.generated_tokens,decode_tps,turn->stats.context_tokens,
                fg_runtime_context_limit(runtime),(unsigned)turn->request.media_count,
                turn->stats.image_tokens,turn->stats.video_tokens,
                turn->stats.video_frames,turn->stats.tower_seconds);
    }
    const char *finish_reason = generated.tool_call_count ? "tool_calls" :
        (turn->stats.generated_tokens >= turn->request.max_tokens ? "length" : "stop");
    bool response_committed=false;
    if (status == FG_OK) {
        if (turn->request.stream)
            status = send_stream_end(generation, &generated, finish_reason, err);
        else
            status = send_completion(generation, &generated, &turn->stats, finish_reason, err);
        if(status==FG_OK){
            api_public_session_free(public_session);
            *public_session=pending_session;
            memset(&pending_session,0,sizeof(pending_session));
            response_committed=true;
        }
    } else {
        char message[sizeof(err->message)];
        snprintf(message, sizeof(message), "%s", err->message);
        fg_error send_err = {0};
        unsigned response_status =
            status == FG_ERR_ARGUMENT || status == FG_ERR_FORMAT || status == FG_ERR_LIMIT ?
                400u :
                500u;
        if(turn->stream_started)send_stream_error(generation,message,&send_err);
        else api_send_error_response(&turn->sink, response_status, message,
                                     generation->keep_alive, &send_err);
    }
    free(generation->content.data); generation->content.data=NULL;
    free(generation->visible_pending.data); generation->visible_pending.data=NULL;
    fg_chat_generated_free(&generated);
    free(turn->rendered); turn->rendered=NULL;
    free(turn->rendered_continuation); turn->rendered_continuation=NULL;
    size_t request_media_count=turn->request.media_count;
    api_chat_request_free(&turn->request);
    api_public_session_free(&pending_session);
    if(turn->generation_attempted&&!response_committed){
        fg_error reset_error={0};
        api_public_session_free(public_session);
        if(status!=FG_ERR_INTERRUPTED&&
           fg_runtime_reset_failure(runtime,&reset_error)!=FG_OK){
            *err=reset_error;
            status=reset_error.code;
        }
    }
    /* A media request that failed before the model produced a token is a
     * vision/tower failure (pack read, image decode, or a Vulkan allocation in
     * the tower device).  The request already received its 5xx with the precise
     * error; the tower session is self-healing and the runtime was reset above,
     * so keep serving instead of taking the API down for one oversized image. */
    if(status!=FG_OK&&request_media_count&&turn->stats.prompt_tokens==0u&&
       turn->stats.generated_tokens==0u)
        status=FG_OK;
    if(status==FG_ERR_INTERRUPTED)
        fprintf(stderr,"request %s: prefill interrupted after %u/%u tokens, "
                "frontier %u, %.1f s, client_failed %d\n",
                turn->id,turn->stats.prefilled_tokens,turn->stats.prompt_tokens,
                turn->stats.context_tokens,turn->stats.prefill_seconds,
                generation->client_failed?1:0);
    if (generation->client_failed || status == FG_ERR_INTERRUPTED ||
        (status == FG_ERR_IO && turn->stats.prompt_tokens + turn->stats.generated_tokens > 0u))
        status = FG_OK;
    else if (status == FG_ERR_ARGUMENT || status == FG_ERR_FORMAT || status == FG_ERR_LIMIT)
        status = FG_OK;
    if (turn->entry) {
        fg_runtime_session_end(runtime, turn->entry->runtime_session);
        turn->entry->busy = false;
        turn->entry->last_used = ++turn->sessions->sequence;
    }
    api_turn_dispose(turn);
    return status;
}

/* ---------------------------------------------------------------------------
 * M3.2 engine scheduler: at most two live chat turns share the ring.  A lone
 * eligible turn (or any non-eligible turn) with no other live session runs on
 * the production B=1 path; a prefill chunk-yields only when another session
 * has work; two decode-ready sessions share depth-B batch steps.  Requests
 * beyond the two owner slots stay in the M2 queue (bound 4).
 * ------------------------------------------------------------------------- */

typedef struct api_engine_scheduler {
    fg_runtime *runtime;
    api_engine_queue *queue;
    api_session_table sessions;
    api_turn *turns[FG_RUNTIME_SESSION_MAX];
    size_t turn_count;
    uint64_t now;
    fg_decode_batch_table table;
    fg_decode_batch_policy policy;
} api_engine_scheduler;

static void api_scheduler_init(api_engine_scheduler *scheduler, fg_runtime *runtime,
                               api_engine_queue *queue) {
    memset(scheduler, 0, sizeof(*scheduler));
    scheduler->runtime = runtime;
    scheduler->queue = queue;
    fg_decode_batch_policy_default(&scheduler->policy);
    fg_error ignored = {0};
    (void)fg_decode_batch_table_init(&scheduler->table, 2u, &ignored);
}

static bool api_scheduler_eligible_turn(const api_turn *turn, const api_turn *other) {
    if (!turn->owns_session || turn->public_continuation || turn->request.media_count)
        return false;
    if (fg_sampler_penalties_active(&turn->request.sampler) ||
        turn->request.sampler.temperature != 0.0f)
        return false;
    if (!turn->entry || !turn->entry->runtime_session) return false;
    if (other && memcmp(&turn->request.sampler, &other->request.sampler,
                        sizeof(turn->request.sampler)))
        return false;
    return true;
}

static bool api_scheduler_queue_waiting(api_engine_queue *queue) {
    pthread_mutex_lock(&queue->mutex);
    bool waiting = queue->count > 0;
    pthread_mutex_unlock(&queue->mutex);
    return waiting;
}

static fg_status api_scheduler_bind(api_engine_scheduler *scheduler, api_turn *turn,
                                    fg_error *err) {
    fg_runtime *runtime = scheduler->runtime;
    fg_runtime_session *session = turn->entry->runtime_session;
    fg_runtime_session *active = fg_runtime_active_session(runtime);
    if (active == session) return FG_OK;
    if (active) fg_runtime_session_end(runtime, active);
    return fg_runtime_session_begin(runtime, session, err);
}

static void api_scheduler_unbind(api_engine_scheduler *scheduler) {
    fg_runtime_session *active = fg_runtime_active_session(scheduler->runtime);
    if (active) fg_runtime_session_end(scheduler->runtime, active);
}

/* Run one prefill call: a single chunk while another turn has work, the whole
 * remaining prompt for a lone turn (the production prefill shape). */
static fg_status api_scheduler_prefill(api_engine_scheduler *scheduler, api_turn *turn,
                                       bool chunk, fg_error *err) {
    bool done = false;
    uint32_t budget = chunk ? fg_runtime_prefill_microbatch(scheduler->runtime) : 0u;
    fg_status status = api_scheduler_bind(scheduler, turn, err);
    if (status == FG_OK)
        status = fg_runtime_session_runner_prefill(scheduler->runtime,
            turn->entry->runtime_session, budget, &done, err);
    api_scheduler_unbind(scheduler);
    if (status != FG_OK) return status;
    if (done) {
        turn->prefill_done = true;
        fg_status enter = fg_decode_batch_sequence_enter(&scheduler->table,
            fg_runtime_session_id(turn->entry->runtime_session),
            fg_runtime_session_slot(turn->entry->runtime_session), err);
        if (enter == FG_OK)
            enter = fg_runtime_session_runner_sync_frontier(scheduler->runtime,
                turn->entry->runtime_session, &scheduler->table, err);
        if (enter != FG_OK) return enter;
    }
    return FG_OK;
}

static fg_status api_scheduler_decode_one(api_engine_scheduler *scheduler, api_turn *turn,
                                          bool *left, fg_error *err) {
    fg_runtime_session *session = turn->entry->runtime_session;
    fg_token_callback callbacks[FG_DECODE_BATCH_MAX_SLOTS] = {api_token};
    void *contexts[FG_DECODE_BATCH_MAX_SLOTS] = {&turn->generation};
    fg_runtime_session *single[1] = {session};
    return fg_runtime_session_runner_batch(scheduler->runtime, single, 1u,
        &scheduler->table, &scheduler->policy, callbacks, contexts, ++scheduler->now,
        left, err);
}

static fg_status api_scheduler_decode_batch(api_engine_scheduler *scheduler,
                                            api_turn **turns, uint32_t count,
                                            bool left[FG_DECODE_BATCH_MAX_SLOTS],
                                            fg_error *err) {
    fg_runtime_session *sessions[FG_DECODE_BATCH_MAX_SLOTS] = {NULL};
    fg_token_callback callbacks[FG_DECODE_BATCH_MAX_SLOTS] = {NULL};
    void *contexts[FG_DECODE_BATCH_MAX_SLOTS] = {NULL};
    for (uint32_t i = 0; i < count; i++) {
        sessions[i] = turns[i]->entry->runtime_session;
        callbacks[i] = api_token;
        contexts[i] = &turns[i]->generation;
    }
    return fg_runtime_session_runner_batch(scheduler->runtime, sessions, count,
        &scheduler->table, &scheduler->policy, callbacks, contexts, ++scheduler->now,
        left, err);
}

static fg_status api_scheduler_start(api_engine_scheduler *scheduler, api_turn *turn,
                                     fg_error *err) {
    if (!turn->eligible) {
        /* The legacy generate path owns the runtime token-path state, so bind
         * the session for the whole call (released by api_turn_close). */
        fg_status bind = api_scheduler_bind(scheduler, turn, err);
        if (bind != FG_OK) return bind;
        turn->started = true;   /* the legacy call runs and completes in pump */
        return FG_OK;
    }
    fg_status status = api_scheduler_bind(scheduler, turn, err);
    if (status == FG_OK)
        status = fg_runtime_session_runner_begin(scheduler->runtime,
            turn->entry->runtime_session, turn->rendered, turn->request.max_tokens,
            &turn->request.sampler, &turn->stats, err);
    api_scheduler_unbind(scheduler);
    if (status != FG_OK) return status;
    turn->started = true;
    return FG_OK;
}

static void api_scheduler_remove(api_engine_scheduler *scheduler, api_turn *turn) {
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        if (scheduler->turns[i] != turn) continue;
        for (size_t j = i + 1; j < scheduler->turn_count; j++)
            scheduler->turns[j - 1] = scheduler->turns[j];
        scheduler->turn_count--;
        return;
    }
}

/* Complete one turn: runner tail commit (when the turn decoded), response tail,
 * transport completion and queue slot release.  Returns api_turn_close's
 * softened status (non-OK is engine-fatal, exactly like the legacy loop). */
static fg_status api_scheduler_complete(api_engine_scheduler *scheduler, api_turn *turn,
                                        fg_status status, fg_error *err) {
    api_connection *connection = turn->connection;
    char *body = turn->http.body;
    bool close_after = connection ? !connection->keep_alive : true;
    bool client_gone = connection && atomic_load(&connection->client_gone);
    if (turn->started && turn->eligible && turn->entry) {
        fg_error finish_error = {0};
        /* Drop the batch-table sequence before the turn is closed: an aborted
         * or finished turn must not keep its owner state slot reserved. */
        uint64_t sequence_id = fg_runtime_session_id(turn->entry->runtime_session);
        if (fg_decode_batch_table_find(&scheduler->table, sequence_id) !=
            FG_DECODE_BATCH_INVALID_SLOT) {
            fg_error leave_error = {0};
            (void)fg_decode_batch_sequence_leave(&scheduler->table, sequence_id,
                                                 &leave_error);
        }
        if (turn->prefill_done) {
            fg_status finish = api_scheduler_bind(scheduler, turn, &finish_error);
            if (finish == FG_OK)
                finish = fg_runtime_session_runner_finish(scheduler->runtime,
                    turn->entry->runtime_session, &finish_error);
            api_scheduler_unbind(scheduler);
            if (finish != FG_OK && status == FG_OK) {
                status = finish;
                *err = finish_error;
            }
        } else {
            fg_runtime_session_runner_abort(scheduler->runtime,
                                            turn->entry->runtime_session);
        }
    }
    bool drop_session = turn->owns_session && turn->entry &&
        (status == FG_ERR_INTERRUPTED || client_gone);
    api_session_table *sessions = turn->sessions;
    api_session_entry *entry = turn->entry;
    status = api_turn_close(turn, status, err);
    if (drop_session) {
        /* An aborted turn's client is gone: drop the session so its owner
         * slot is reusable by the next request. */
        api_session_table_drop(sessions, scheduler->runtime, entry);
    }
    if (connection) api_connection_complete_response(connection, close_after);
    free(body);
    api_engine_queue_complete(scheduler->queue);
    api_scheduler_remove(scheduler, turn);
    free(turn);
    return status;
}

/* Abort a live turn because its client disconnected: commit partial decode
 * output like the production interrupt path, drop the session so its owner
 * slot is reusable, and complete the transport entry. */
static fg_status api_scheduler_abort(api_engine_scheduler *scheduler, api_turn *turn,
                                     fg_error *err) {
    turn->generation.client_failed = true;
    if (turn->started && turn->eligible && turn->prefill_done)
        return api_scheduler_complete(scheduler, turn, FG_OK, err);
    return api_scheduler_complete(scheduler, turn, FG_ERR_INTERRUPTED, err);
}

/* One scheduler action per call: start, abort, complete, then advance. */
static fg_status api_scheduler_pump(api_engine_scheduler *scheduler, fg_error *err) {
    bool any_started = false;
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (turn->started && !turn->left) any_started = true;
    }
    /* 1. Start the next turn when the ring allows it.  A non-eligible turn
     * runs strictly in admission order and never shares the ring. */
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (turn->started || turn->left) continue;
        if (!turn->eligible && (i > 0u || any_started)) continue;
        fg_status status = api_scheduler_start(scheduler, turn, err);
        if (status != FG_OK) return api_scheduler_complete(scheduler, turn, status, err);
        if (!turn->eligible) {
            fg_status run = api_turn_run_legacy(turn, err);
            return api_scheduler_complete(scheduler, turn, run, err);
        }
        return FG_OK;
    }
    /* 2. Abort a turn whose client disappeared (waiting turns included). */
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (turn->connection && atomic_load(&turn->connection->client_gone))
            return api_scheduler_abort(scheduler, turn, err);
    }
    /* 3. Complete a turn that the runner has finished. */
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (turn->started && turn->left)
            return api_scheduler_complete(scheduler, turn, FG_OK, err);
    }
    /* 4. Advance: batch when two decode-ready sessions exist, otherwise one
     * prefill chunk (only while another session has work) or one B=1 step. */
    api_turn *decodable[FG_DECODE_BATCH_MAX_SLOTS];
    uint32_t decodable_count = 0;
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (turn->started && turn->eligible && !turn->left && turn->prefill_done &&
            decodable_count < FG_DECODE_BATCH_MAX_SLOTS)
            decodable[decodable_count++] = turn;
    }
    if (decodable_count >= 2u) {
        bool left[FG_DECODE_BATCH_MAX_SLOTS] = {false};
        fg_status status = api_scheduler_decode_batch(scheduler, decodable,
                                                      decodable_count, left, err);
        if (status != FG_OK) return status;
        for (uint32_t i = 0; i < decodable_count; i++)
            if (left[i]) decodable[i]->left = true;
        return FG_OK;
    }
    bool other_work = api_scheduler_queue_waiting(scheduler->queue);
    uint32_t active = 0;
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (turn->started && turn->eligible && !turn->left) active++;
    }
    for (size_t i = 0; i < scheduler->turn_count; i++) {
        api_turn *turn = scheduler->turns[i];
        if (!turn->started || !turn->eligible || turn->left) continue;
        if (!turn->prefill_done) {
            fg_status status = api_scheduler_prefill(scheduler, turn,
                other_work || active > 1u, err);
            if (status != FG_OK) return status;
        } else {
            bool left = false;
            fg_status status = api_scheduler_decode_one(scheduler, turn, &left, err);
            if (status != FG_OK) return status;
            if (left) turn->left = true;
        }
    }
    return FG_OK;
}

/* Admit one popped queue entry into the scheduler. */
static fg_status api_scheduler_admit(api_engine_scheduler *scheduler,
                                     api_engine_request *engine_request, fg_error *err) {
    api_turn *turn = calloc(1u, sizeof(*turn));
    if (!turn) {
        fg_error_set(err, FG_ERR_OOM, "allocate parked turn");
        return FG_ERR_OOM;
    }
    api_sink sink = {
        .fd = engine_request->connection->fd,
        .connection = engine_request->connection,
    };
    fg_status status = api_turn_open(turn, sink, scheduler->runtime, NULL,
                                     &scheduler->sessions, &engine_request->http, err);
    engine_request->http.body = NULL;   /* ownership moved into the turn */
    if (status != FG_OK) {
        char *body = turn->http.body;
        (void)api_turn_close(turn, status, err);
        free(body);
        api_engine_queue_complete(scheduler->queue);
        free(turn);
        return status;
    }
    if (!turn->opened) {
        char *body = turn->http.body;
        (void)api_turn_close(turn, FG_OK, err);
        api_connection_complete_response(engine_request->connection,
                                         !engine_request->connection->keep_alive);
        free(body);
        api_engine_queue_complete(scheduler->queue);
        free(turn);
        return FG_OK;
    }
    const api_turn *other = scheduler->turn_count ? scheduler->turns[0] : NULL;
    turn->eligible = api_scheduler_eligible_turn(turn, other);
    scheduler->turns[scheduler->turn_count++] = turn;
    return FG_OK;
}

static void api_scheduler_shutdown(api_engine_scheduler *scheduler) {
    while (scheduler->turn_count) {
        api_turn *turn = scheduler->turns[0];
        fg_error ignored = {0};
        (void)api_scheduler_abort(scheduler, turn, &ignored);
    }
    api_session_table_destroy(&scheduler->sessions, scheduler->runtime);
}

/* Engine loop: the front-end thread owns the listener and every client socket;
 * this thread admits FIFO requests, keeps at most two turns live on the ring
 * and blocks only while it has nothing to advance. */
static fg_status api_scheduler_run(api_engine_scheduler *scheduler, fg_error *err) {
    const int timeout_ms = 200;
    while (!api_stop_requested) {
        while (scheduler->turn_count < 2u) {
            api_engine_request engine_request;
            if (!api_engine_queue_pop_wait(scheduler->queue, &engine_request, 0))
                break;
            if (atomic_load(&engine_request.connection->client_gone)) {
                free(engine_request.http.body);
                engine_request.http.body = NULL;
                api_connection_complete_response(engine_request.connection, true);
                api_engine_queue_complete(scheduler->queue);
                continue;
            }
            fg_status status = api_scheduler_admit(scheduler, &engine_request, err);
            if (status != FG_OK) return status;
        }
        if (scheduler->turn_count == 0) {
            api_engine_request engine_request;
            if (!api_engine_queue_pop_wait(scheduler->queue, &engine_request, timeout_ms))
                continue;
            if (atomic_load(&engine_request.connection->client_gone)) {
                free(engine_request.http.body);
                engine_request.http.body = NULL;
                api_connection_complete_response(engine_request.connection, true);
                api_engine_queue_complete(scheduler->queue);
                continue;
            }
            fg_status status = api_scheduler_admit(scheduler, &engine_request, err);
            if (status != FG_OK) return status;
        }
        fg_status status = api_scheduler_pump(scheduler, err);
        if (status != FG_OK) return status;
    }
    api_scheduler_shutdown(scheduler);
    return FG_OK;
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
    /* The front-end poll loop never blocks on one client; a slow reader or a
     * mid-body client must not stall probes on other connections. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        fg_error_set(err, FG_ERR_IO, "configure API client non-blocking mode: %s",
                     strerror(errno));
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

    /* The front-end thread owns the listener and every client connection; this
     * thread is the engine: it consumes complete chat requests one at a time
     * and never touches the listener. */
    api_engine_queue queue;
    api_engine_queue_init(&queue);
    api_frontend frontend;
    status = api_frontend_start(&frontend, listener, runtime, &queue, err);
    if (status != FG_OK) {
        api_engine_queue_destroy(&queue);
        close(listener);
        sigaction(SIGINT, &old_int, NULL);
        sigaction(SIGTERM, &old_term, NULL);
        sigaction(SIGPIPE, &old_pipe, NULL);
        fg_runtime_close(runtime);
        return status;
    }

    /* M3: live sessions survive across requests in the queue; the scheduler
     * resolves each request to its session (explicit id, connection, or prefix
     * extension), keeps at most two turns live on the ring and runs a lone
     * session on the production B=1 path. */
    api_engine_scheduler scheduler;
    api_scheduler_init(&scheduler, runtime, &queue);
    status = api_scheduler_run(&scheduler, err);

    api_engine_queue_stop(&queue);
    api_frontend_stop(&frontend);
    close(listener);
    api_engine_queue_destroy(&queue);
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
