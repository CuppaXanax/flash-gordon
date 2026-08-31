#include "fg_prefix.h"
#include "fg_sha256.h"
#include "fg_tokenizer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TOKENIZER_MAGIC UINT64_C(0x314e4b4f544746)

enum {
    TEST_EOS = 256u,
    TEST_BOS = 257u,
    TEST_IM_START = 258u,
    TEST_AL = 296u,
    TEST_ALP = 297u,
    TEST_ALPH = 298u,
    TEST_ALPHA = 300u
};

static int write_u32(FILE *file, uint32_t value) {
    return fwrite(&value, 1, 4u, file) == 4u;
}

static int write_u64(FILE *file, uint64_t value) {
    return fwrite(&value, 1, 8u, file) == 8u;
}

static uint32_t byte_codepoint(uint8_t byte) {
    if ((byte >= 33u && byte <= 126u) || (byte >= 161u && byte <= 172u) ||
        byte >= 174u)
        return byte;
    uint32_t ordinal = 0;
    for (uint32_t value = 0; value < 256u; value++) {
        if ((value >= 33u && value <= 126u) ||
            (value >= 161u && value <= 172u) || value >= 174u)
            continue;
        if (value == byte) return 256u + ordinal;
        ordinal++;
    }
    return byte;
}

static size_t encode_utf8(uint32_t code, char output[4]) {
    if (code <= 0x7fu) {
        output[0] = (char)code;
        return 1u;
    }
    if (code <= 0x7ffu) {
        output[0] = (char)(0xc0u | (code >> 6u));
        output[1] = (char)(0x80u | (code & 63u));
        return 2u;
    }
    output[0] = (char)(0xe0u | (code >> 12u));
    output[1] = (char)(0x80u | ((code >> 6u) & 63u));
    output[2] = (char)(0x80u | (code & 63u));
    return 3u;
}

static void cleanup_tokenizer_directory(const char *path, const char *directory) {
    unlink(path);
    rmdir(directory);
}

static size_t first_mismatch(const uint32_t *left, size_t left_count,
                             const uint32_t *right, size_t right_count) {
    size_t common = left_count < right_count ? left_count : right_count;
    size_t index = 0;
    while (index < common && left[index] == right[index]) index++;
    return index;
}

static int verify_authoritative_continuation(fg_tokenizer *tokenizer,
                                             uint32_t expected_alpha,
                                             fg_error *error) {
    static const char prompt[] =
        "<|im_start|>user\none<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n";
    static const char generated[] = "Alpha.";
    static const char suffix_text[] =
        "<|im_end|>\n<|im_start|>user\nBeta.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n";
    char private_rendered[sizeof(prompt) + sizeof(generated) + sizeof(suffix_text)];
    snprintf(private_rendered, sizeof(private_rendered), "%s%s%s",
             prompt, generated, suffix_text);

    fg_tokens prompt_tokens = {0};
    fg_tokens suffix_tokens = {0};
    fg_tokens retokenized = {0};
    int ok = fg_tokenizer_encode(tokenizer, prompt, true, &prompt_tokens, error) == FG_OK;
    int32_t *history = NULL;
    size_t history_count = prompt_tokens.count + sizeof(generated) - 1u;
    if (ok) {
        history = malloc(history_count * sizeof(*history));
        ok = history != NULL;
    }
    for (size_t i = 0; ok && i < prompt_tokens.count; i++)
        history[i] = (int32_t)prompt_tokens.data[i];
    for (size_t i = 0; ok && i < sizeof(generated) - 1u; i++) {
        uint32_t token = 0;
        ok = fg_tokenizer_lookup(tokenizer, generated + i, 1u, &token, error) == FG_OK;
        if (ok) history[prompt_tokens.count + i] = (int32_t)token;
    }
    if (ok)
        ok = fg_tokenizer_encode(tokenizer, suffix_text, true, &suffix_tokens, error) ==
                 FG_OK &&
             suffix_tokens.count && suffix_tokens.data[0] == fg_tokenizer_eos(tokenizer);

    uint32_t *constructed = NULL;
    size_t constructed_count = 0;
    if (ok)
        ok = fg_prefix_build_continuation_tokens(
                 history, history_count, fg_tokenizer_eos(tokenizer),
                 suffix_tokens.data, suffix_tokens.count,
                 &constructed, &constructed_count, error) == FG_OK;
    if (ok)
        ok = constructed_count == history_count + suffix_tokens.count &&
             !memcmp(constructed, history, history_count * sizeof(*constructed)) &&
             !memcmp(constructed + history_count, suffix_tokens.data,
                     suffix_tokens.count * sizeof(*constructed));
    if (ok)
        ok = fg_tokenizer_encode(tokenizer, private_rendered, true, &retokenized, error) ==
             FG_OK;
    size_t mismatch = ok ? first_mismatch(constructed, constructed_count,
                                          retokenized.data, retokenized.count) : 0u;
    if (ok)
        ok = mismatch == prompt_tokens.count &&
             constructed[mismatch] == (uint32_t)'A' &&
             (expected_alpha == UINT32_MAX || retokenized.data[mismatch] == expected_alpha);
    if (!ok && !error->message[0])
        fg_error_set(error, FG_ERR_MISMATCH,
                     "authoritative generated-token continuation test failed");

    free(constructed);
    free(history);
    fg_tokens_free(&retokenized);
    fg_tokens_free(&suffix_tokens);
    fg_tokens_free(&prompt_tokens);
    return ok;
}

static int test_real_qwen38_artifact(const char *source) {
    char directory[128], tokenizer_directory[160], path[200];
    snprintf(directory, sizeof(directory), "tests/.fg-q38-tokenizer-%ld", (long)getpid());
    snprintf(tokenizer_directory, sizeof(tokenizer_directory), "%s/tokenizer", directory);
    snprintf(path, sizeof(path), "%s/tokenizer.fgt", tokenizer_directory);
    if (mkdir(directory, 0700) != 0) {
        perror("create real-tokenizer test directory");
        return 1;
    }
    fg_manifest *manifest = malloc(sizeof(*manifest));
    fg_error error = {0};
    fg_tokenizer *tokenizer = NULL;
    int ok = manifest != NULL;
    if (ok) {
        fg_manifest_init(manifest);
        ok = fg_tokenizer_pack_gguf(source, directory, manifest, &error) == FG_OK;
    }
    if (ok) ok = fg_tokenizer_open(&tokenizer, directory, manifest, &error) == FG_OK;
    if (ok) ok = fg_tokenizer_validate_qwen38(tokenizer, &error) == FG_OK;
    if (ok) ok = verify_authoritative_continuation(tokenizer, UINT32_MAX, &error);
    fg_tokenizer_close(tokenizer);
    free(manifest);
    cleanup_tokenizer_directory(path, tokenizer_directory);
    rmdir(directory);
    if (!ok) {
        fprintf(stderr, "real Qwen3.8 tokenizer parity failed: %s\n", error.message);
        return 1;
    }
    puts("Flash Gordon real Unsloth Qwen3.8 tokenizer parity: PASS");
    return 0;
}

static int write_test_asset(const char *path, uint64_t *padded_bytes) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = write_u64(file, TOKENIZER_MAGIC) && write_u32(file, 1u) &&
             write_u32(file, 248320u) && write_u32(file, 4u) &&
             write_u32(file, TEST_BOS) && write_u32(file, TEST_EOS) &&
             write_u32(file, 0u);
    char text[32];
    for (uint32_t i = 0; ok && i < 248320u; i++) {
        size_t length = 0;
        uint32_t type = 1u;
        if (i < 256u) {
            length = encode_utf8(byte_codepoint((uint8_t)i), text);
        } else if (i == TEST_EOS) {
            memcpy(text, "<|im_end|>", 10u);
            length = 10u;
            type = 3u;
        } else if (i == TEST_IM_START) {
            memcpy(text, "<|im_start|>", 12u);
            length = 12u;
            type = 3u;
        } else if (i == TEST_AL) {
            memcpy(text, "Al", 2u);
            length = 2u;
        } else if (i == TEST_ALP) {
            memcpy(text, "Alp", 3u);
            length = 3u;
        } else if (i == TEST_ALPH) {
            memcpy(text, "Alph", 4u);
            length = 4u;
        } else if (i == TEST_ALPHA) {
            memcpy(text, "Alpha", 5u);
            length = 5u;
        } else {
            int result = snprintf(text, sizeof(text), "t%06u", i);
            length = result > 0 ? (size_t)result : 0u;
        }
        ok = length > 0 && write_u32(file, (uint32_t)length) &&
             fwrite(text, 1, length, file) == length && write_u32(file, type);
    }
    static const char *const merges[] = {"A l", "Al p", "Alp h", "Alph a"};
    for (size_t i = 0; ok && i < sizeof(merges) / sizeof(merges[0]); i++) {
        size_t length = strlen(merges[i]);
        ok = write_u32(file, (uint32_t)length) &&
             fwrite(merges[i], 1, length, file) == length;
    }
    long end = ftell(file);
    if (end < 0) ok = 0;
    uint64_t padded = ok ? fg_align_up_u64((uint64_t)end, FG_ALIGNMENT) : 0u;
    while (ok && (uint64_t)end < padded) {
        if (fputc(0, file) == EOF) ok = 0;
        end++;
    }
    if (fclose(file) != 0) ok = 0;
    *padded_bytes = padded;
    return ok;
}

static int test_sealed_tokenizer(void) {
    char directory[128], tokenizer_directory[160], path[200];
    snprintf(directory, sizeof(directory), "tests/.fg-tokenizer-%ld", (long)getpid());
    snprintf(tokenizer_directory, sizeof(tokenizer_directory), "%s/tokenizer", directory);
    snprintf(path, sizeof(path), "%s/tokenizer.fgt", tokenizer_directory);
    if (mkdir(directory, 0700) != 0 || mkdir(tokenizer_directory, 0700) != 0) return 1;

    uint64_t padded = 0;
    int ok = write_test_asset(path, &padded);
    fg_manifest manifest = {0};
    manifest.tensor_count = 1u;
    fg_tensor_record *record = &manifest.tensors[0];
    snprintf(record->name, sizeof(record->name), "tokenizer/tokenizer.fgt");
    record->bytes = padded;
    record->dims = 1u;
    record->shape[0] = padded;
    record->rank = UINT16_MAX;
    record->layer = UINT16_MAX;
    record->expert = UINT16_MAX;
    record->kind = FG_TENSOR_TOKENIZER;
    fg_error error = {0};
    if (ok) ok = fg_sha256_file(path, record->sha256, &error) == FG_OK;

    fg_tokenizer *tokenizer = NULL;
    fg_status status =
        ok ? fg_tokenizer_open(&tokenizer, directory, &manifest, &error) : FG_ERR_IO;
    if (status == FG_ERR_UNAVAILABLE) {
        fprintf(stderr, "SKIP direct tokenizer load: %s\n", error.message);
        cleanup_tokenizer_directory(path, tokenizer_directory);
        rmdir(directory);
        return 77;
    }
    if (status != FG_OK) {
        fprintf(stderr, "tokenizer open: %s\n", error.message);
        ok = 0;
    }

    uint32_t token = 0;
    const char *piece = NULL;
    size_t bytes = 0;
    uint32_t type = 0;
    if (ok)
        ok = fg_tokenizer_vocab_size(tokenizer) == 248320u &&
             fg_tokenizer_bos(tokenizer) == TEST_BOS &&
             fg_tokenizer_eos(tokenizer) == TEST_EOS &&
             !fg_tokenizer_add_bos(tokenizer) &&
             fg_tokenizer_lookup(tokenizer, "t123456", 7u, &token, &error) == FG_OK &&
             token == 123456u &&
             fg_tokenizer_token(tokenizer, token, &piece, &bytes, &type, &error) == FG_OK &&
             bytes == 7u && !memcmp(piece, "t123456", 7u) && type == 1u;

    fg_tokens tokens = {0};
    if (ok)
        ok = fg_tokenizer_encode(tokenizer, "hello", false, &tokens, &error) == FG_OK &&
             tokens.count == 5u && tokens.data[0] == (uint32_t)'h' &&
             tokens.data[1] == (uint32_t)'e' && tokens.data[2] == (uint32_t)'l' &&
             tokens.data[3] == (uint32_t)'l' && tokens.data[4] == (uint32_t)'o';
    char decoded[8];
    size_t decoded_bytes = 0;
    if (ok)
        ok = fg_tokenizer_decode_token(tokenizer, (uint32_t)'h', decoded,
                                       sizeof(decoded), &decoded_bytes, &error) == FG_OK &&
             decoded_bytes == 1u && decoded[0] == 'h';
    fg_tokens_free(&tokens);
    if (ok)
        ok = fg_tokenizer_encode(tokenizer, "<|im_end|>", true, &tokens, &error) == FG_OK &&
             tokens.count == 1u && tokens.data[0] == TEST_EOS;
    fg_tokens_free(&tokens);
    if (ok) ok = verify_authoritative_continuation(tokenizer, TEST_ALPHA, &error);
    fg_tokenizer_close(tokenizer);

    int fd = open(path, O_RDWR | O_CLOEXEC);
    uint8_t byte = 0;
    if (ok && fd >= 0 && pread(fd, &byte, 1u, 64) == 1) {
        byte ^= 1u;
        ok = pwrite(fd, &byte, 1u, 64) == 1;
    } else {
        ok = 0;
    }
    if (fd >= 0) close(fd);
    tokenizer = NULL;
    if (ok)
        ok = fg_tokenizer_open(&tokenizer, directory, &manifest, &error) ==
             FG_ERR_MISMATCH;
    fg_tokenizer_close(tokenizer);

    cleanup_tokenizer_directory(path, tokenizer_directory);
    rmdir(directory);
    if (!ok) {
        fprintf(stderr, "tokenizer direct-load/seal test failed: %s\n", error.message);
        return 1;
    }
    puts("Flash Gordon sealed direct-I/O tokenizer load: PASS");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2) return test_real_qwen38_artifact(argv[1]);
    if (argc != 1) {
        fprintf(stderr, "usage: %s [Qwen3.8 GGUF tokenizer shard]\n", argv[0]);
        return 2;
    }
    return test_sealed_tokenizer();
}
