#include "fg_tokenizer.h"
#include "fg_manifest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <shard.gguf> <workdir>\n", argv[0]); return 2; }
    fg_manifest *m = malloc(sizeof(*m));
    fg_manifest_init(m);
    fg_error err = {0};
    if (fg_tokenizer_pack_gguf(argv[1], argv[2], m, &err) != FG_OK) { fprintf(stderr, "pack: %s\n", err.message); return 1; }
    fg_tokenizer *tok = NULL;
    if (fg_tokenizer_open(&tok, argv[2], m, &err) != FG_OK) { fprintf(stderr, "open: %s\n", err.message); return 1; }

    static const char para[] =
        "The user asks about the build system and the deploy pipeline for the cluster. "
        "We answer with measured numbers, short lines, and no unnecessary editing. "
        "<|im_end|>\n<|im_start|>assistant\n<think>\nCheck the logs first, then reply.\n</think>\n"
        "Done. <|im_end|>\n<|im_start|>user\n";

    for (int target = 4096; target <= 65536; target *= 2) {
        size_t para_len = sizeof(para) - 1u;
        size_t repeats = ((size_t)target * 4u) / para_len + 1u;
        size_t bytes = repeats * para_len;
        char *text = malloc(bytes + 1u);
        if (!text) return 1;
        for (size_t i = 0; i < repeats; i++) memcpy(text + i * para_len, para, para_len);
        text[bytes] = 0;

        fg_tokens t = {0};
        double start = now_ms();
        if (fg_tokenizer_encode(tok, text, true, &t, &err) != FG_OK) { fprintf(stderr, "encode: %s\n", err.message); return 1; }
        double elapsed = now_ms() - start;
        printf("tokens=%zu bytes=%zu encode_ms=%.1f ms_per_1k_tokens=%.1f\n",
               t.count, bytes, elapsed, elapsed / ((double)t.count / 1000.0));
        fg_tokens_free(&t);
        free(text);
    }
    return 0;
}
