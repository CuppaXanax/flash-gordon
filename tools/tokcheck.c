#include "fg_chat.h"
#include "fg_tokenizer.h"
#include "fg_manifest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shard.gguf> <workdir>\n", argv[0]); return 2; }
    fg_manifest *m = malloc(sizeof(*m));
    fg_manifest_init(m);
    fg_error err = {0};
    if (fg_tokenizer_pack_gguf(argv[1], argv[2], m, &err) != FG_OK) { fprintf(stderr, "pack: %s\n", err.message); return 1; }
    fg_tokenizer *tok = NULL;
    if (fg_tokenizer_open(&tok, argv[2], m, &err) != FG_OK) { fprintf(stderr, "open: %s\n", err.message); return 1; }
    fg_chat_message msgs[3] = {
        {.role="system", .content="You are Pi."},
        {.role="user", .content="heeeyyyy, what's hanging my friend! hello world~"},
    };
    char *rendered = NULL;
    if (fg_chat_render(msgs, 2, NULL, &rendered, &err) != FG_OK) { fprintf(stderr, "render: %s\n", err.message); return 1; }
    printf("RENDERED_BEGIN\n%s\nRENDERED_END\n", rendered);
    fg_tokens t = {0};
    if (fg_tokenizer_encode(tok, rendered, true, &t, &err) != FG_OK) { fprintf(stderr, "encode: %s\n", err.message); return 1; }
    printf("tokens=%zu\n", t.count);
    static const char *const probes[] = {"hello~<|im_end|>\n", "```<|im_end|>", "Hi<|im_end|>\n"};
    for (size_t probe = 0; probe < sizeof(probes) / sizeof(probes[0]); probe++) {
        fg_tokens p = {0};
        if (fg_tokenizer_encode(tok, probes[probe], true, &p, &err) != FG_OK) continue;
        printf("probe[%zu]=%s ->", probe, probes[probe]);
        for (size_t i = 0; i < p.count; i++) printf(" %u", p.data[i]);
        printf("\n");
        fg_tokens_free(&p);
    }
    for (size_t i = 0; i < t.count; i++) {
        const char *s = NULL; size_t n = 0;
        fg_tokenizer_token(tok, t.data[i], &s, &n, NULL, &err);
        char buf[80];
        size_t copy = n < sizeof(buf) - 1 ? n : sizeof(buf) - 1;
        memcpy(buf, s, copy); buf[copy] = 0;
        for (size_t k = 0; k < copy; k++) if (buf[k] == '\n') buf[k] = '~';
        printf("%zu: id=%u tok=[%s]\n", i, t.data[i], buf);
    }
    return 0;
}
