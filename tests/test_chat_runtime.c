#include "fg_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fg_runtime {
    const char *generated;
    const char *expected_continuation;
    uint32_t full_calls;
    uint32_t continuation_calls;
    uint32_t reset_calls;
    bool force_continuation_miss;
};

const char *fg_execution_mode_name(fg_execution_mode mode){
    return mode==FG_EXECUTION_PIPELINE?"pipeline":"expert-parallel";
}

#include "../src/chat.c"

static int failures;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);         \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

typedef struct captured_text {
    char data[512];
    size_t length;
} captured_text;

static fg_status capture_token(void *context,uint32_t token,const char *text,
                               size_t length,fg_error *err){
    (void)token;
    captured_text *captured=context;
    if(length>sizeof(captured->data)-captured->length-1u){
        fg_error_set(err,FG_ERR_LIMIT,"captured test generation is too long");
        return FG_ERR_LIMIT;
    }
    memcpy(captured->data+captured->length,text,length);
    captured->length+=length;
    captured->data[captured->length]=0;
    return FG_OK;
}

fg_status fg_runtime_open(fg_runtime **out,const char *manifest_path,fg_error *err){
    (void)out;
    (void)manifest_path;
    fg_error_set(err,FG_ERR_UNAVAILABLE,"test runtime");
    return FG_ERR_UNAVAILABLE;
}

fg_status fg_runtime_open_with_options(fg_runtime **out,const char *manifest_path,
                                       const fg_runtime_options *options,fg_error *err){
    (void)options;
    return fg_runtime_open(out,manifest_path,err);
}

void fg_runtime_close(fg_runtime *runtime){
    (void)runtime;
}

fg_status fg_runtime_reset(fg_runtime *runtime,fg_error *err){
    (void)err;
    runtime->reset_calls++;
    return FG_OK;
}

fg_status fg_runtime_reset_public_history(fg_runtime *runtime,fg_error *err){
    return fg_runtime_reset(runtime,err);
}

fg_status fg_runtime_reset_failure(fg_runtime *runtime,fg_error *err){
    return fg_runtime_reset(runtime,err);
}

fg_status fg_runtime_generate(fg_runtime *runtime,const char *rendered_transcript,
                              uint32_t max_tokens,fg_token_callback callback,
                              void *callback_context,fg_interrupt_fn interrupted,
                              void *interrupt_context,fg_generation_stats *stats,
                              fg_error *err){
    (void)max_tokens;
    (void)interrupted;
    (void)interrupt_context;
    runtime->full_calls++;
    memset(stats,0,sizeof(*stats));
    stats->prompt_tokens=(uint32_t)strlen(rendered_transcript);
    stats->prefilled_tokens=stats->prompt_tokens;
    stats->reset_reason=runtime->reset_calls?
        FG_PREFIX_RESET_EXPLICIT:FG_PREFIX_RESET_COLD_START;
    const char *generated=runtime->generated?runtime->generated:"";
    return callback(callback_context,1u,generated,strlen(generated),err);
}

fg_status fg_runtime_generate_continuation(
    fg_runtime *runtime,const char *public_transcript,
    const char *rendered_continuation,bool *prefix_miss,uint32_t max_tokens,
    fg_token_callback callback,void *callback_context,
    fg_interrupt_fn interrupted,void *interrupt_context,
    fg_generation_stats *stats,fg_error *err){
    (void)max_tokens;
    (void)interrupted;
    (void)interrupt_context;
    CHECK(public_transcript&&public_transcript[0]);
    runtime->continuation_calls++;
    if(runtime->force_continuation_miss){
        runtime->force_continuation_miss=false;
        *prefix_miss=true;
        fg_error_set(err,FG_ERR_UNAVAILABLE,"injected missing CLI frontier");
        return FG_ERR_UNAVAILABLE;
    }
    CHECK(runtime->expected_continuation!=NULL);
    CHECK(!strcmp(rendered_continuation,runtime->expected_continuation));
    *prefix_miss=false;
    memset(stats,0,sizeof(*stats));
    stats->prompt_tokens=80u;
    stats->prefilled_tokens=12u;
    stats->reused_tokens=68u;
    stats->prefix_cache_hit=true;
    stats->reset_reason=FG_PREFIX_RESET_NONE;
    const char *generated=runtime->generated?runtime->generated:"";
    return callback(callback_context,2u,generated,strlen(generated),err);
}

uint32_t fg_runtime_context_tokens(const fg_runtime *runtime){
    (void)runtime;
    return 0u;
}

uint32_t fg_runtime_context_limit(const fg_runtime *runtime){
    (void)runtime;
    return 8192u;
}

const char *fg_runtime_model_name(const fg_runtime *runtime){
    (void)runtime;
    return "Qwen3.8-Flash-Next";
}

static void test_cli_uses_authoritative_multi_turn_continuation(void){
    fg_runtime runtime={
        .generated="private\n</think>\n\nAlpha."
    };
    fg_error err={0};
    const fg_chat_message first_messages[]={
        {.role="user",.content="one"}
    };
    char *first_prompt=NULL;
    CHECK(fg_chat_render(first_messages,1u,NULL,&first_prompt,&err)==FG_OK);
    captured_text first={0};
    fg_generation_stats stats={0};
    CHECK(chat_generate_turn(&runtime,first_prompt,NULL,false,32u,capture_token,&first,
                             NULL,NULL,&stats,&err)==FG_OK);
    CHECK(runtime.full_calls==1u);
    CHECK(runtime.continuation_calls==0u);
    CHECK(!strcmp(first.data,"private\n</think>\n\nAlpha."));

    const fg_chat_message second_messages[]={
        {.role="user",.content="one"},
        {.role="assistant",.content="<think>\nprivate\n</think>\n\nAlpha."},
        {.role="user",.content="two"}
    };
    char *second_prompt=NULL,*second_continuation=NULL;
    CHECK(fg_chat_render(second_messages,3u,NULL,&second_prompt,&err)==FG_OK);
    CHECK(fg_chat_render_continuation(second_messages+2u,1u,NULL,
                                      &second_continuation,&err)==FG_OK);
    CHECK(strstr(second_continuation,"Alpha.")==NULL);
    runtime.expected_continuation=second_continuation;
    runtime.generated="private two\n</think>\n\nBeta.";
    captured_text second={0};
    memset(&stats,0,sizeof(stats));
    CHECK(chat_generate_turn(&runtime,second_prompt,second_continuation,true,32u,
                             capture_token,&second,NULL,NULL,&stats,&err)==FG_OK);
    CHECK(runtime.full_calls==1u);
    CHECK(runtime.continuation_calls==1u);
    CHECK(runtime.reset_calls==0u);
    CHECK(stats.prefix_cache_hit);
    CHECK(stats.reused_tokens==68u);
    CHECK(!strcmp(second.data,"private two\n</think>\n\nBeta."));

    runtime.force_continuation_miss=true;
    runtime.generated="fallback";
    captured_text fallback={0};
    memset(&stats,0,sizeof(stats));
    CHECK(chat_generate_turn(&runtime,second_prompt,second_continuation,true,32u,
                             capture_token,&fallback,NULL,NULL,&stats,&err)==FG_OK);
    CHECK(runtime.continuation_calls==2u);
    CHECK(runtime.reset_calls==1u);
    CHECK(runtime.full_calls==2u);
    CHECK(!stats.prefix_cache_hit);
    CHECK(!strcmp(fallback.data,"fallback"));

    free(second_continuation);
    free(second_prompt);
    free(first_prompt);
}

int main(void){
    test_cli_uses_authoritative_multi_turn_continuation();
    if(failures)fprintf(stderr,"%d chat runtime test(s) failed\n",failures);
    return failures?1:0;
}
