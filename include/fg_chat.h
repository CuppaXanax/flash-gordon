#ifndef FLASH_GORDON_CHAT_H
#define FLASH_GORDON_CHAT_H

#include "fg.h"

typedef struct fg_chat_tool_call {
    const char *id;
    const char *name;
    const char *arguments_json;
} fg_chat_tool_call;

typedef struct fg_chat_message {
    const char *role;
    const char *content;
    const char *reasoning;
    const char *tool_call_id;
    const fg_chat_tool_call *tool_calls;
    size_t tool_call_count;
} fg_chat_message;

typedef enum fg_chat_think_mode {
    FG_CHAT_THINK_DEFAULT = 0,
    FG_CHAT_THINK_OFF = 1
} fg_chat_think_mode;

typedef enum fg_chat_tool_choice {
    FG_CHAT_TOOL_AUTO = 0,
    FG_CHAT_TOOL_NONE = 1,
    FG_CHAT_TOOL_REQUIRED = 2,
    FG_CHAT_TOOL_NAMED = 3
} fg_chat_tool_choice;

typedef struct fg_chat_render_options {
    const char *const *tool_schemas;
    size_t tool_schema_count;
    fg_chat_think_mode think_mode;
    fg_chat_tool_choice tool_choice;
    const char *tool_choice_name;
} fg_chat_render_options;

typedef struct fg_chat_generated {
    char *content;
    char *reasoning;
    fg_chat_tool_call *tool_calls;
    size_t tool_call_count;
} fg_chat_generated;

fg_status fg_chat_render(const fg_chat_message *messages, size_t message_count,
                         const fg_chat_render_options *options, char **rendered,
                         fg_error *err);
fg_status fg_chat_parse_generated(const char *text, bool thinking,
                                  fg_chat_generated *generated, fg_error *err);
void fg_chat_generated_free(fg_chat_generated *generated);
fg_status fg_chat_main(const char *manifest_path, uint32_t max_tokens, fg_error *err);

#endif
