#include "fg_embedding.h"
#include "fg_quant.h"
#include "fg_sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            failures++;                                                           \
        }                                                                         \
    } while (0)

static void build_manifest(fg_manifest *manifest, const uint8_t digest[32]) {
    memset(manifest, 0, sizeof(*manifest));
    manifest->magic = FG_MANIFEST_MAGIC;
    manifest->format_version = FG_MANIFEST_FORMAT_VERSION;
    manifest->protocol_version = FG_PIPELINE_PROTOCOL_VERSION;
    manifest->header_bytes = sizeof(*manifest);
    manifest->rank_count = FG_RANK_COUNT;
    manifest->layer_count = FG_LAYER_COUNT;
    manifest->expert_count = FG_EXPERT_COUNT;
    manifest->hidden_size = FG_HIDDEN_SIZE;
    manifest->top_k = FG_TOP_K;
    manifest->native_context = FG_NATIVE_CONTEXT;
    manifest->max_context = FG_NATIVE_CONTEXT;
    manifest->prefill_microbatch = FG_PIPELINE_DEFAULT_MICROBATCH;
    manifest->prefill_window = FG_DEFAULT_WINDOW;
    manifest->execution_mode = FG_EXECUTION_PIPELINE;
    manifest->stage_count = FG_PIPELINE_STAGE_COUNT;
    manifest->slot_count = FG_PIPELINE_DEFAULT_SLOT_COUNT;
    for (uint32_t stage = 0; stage < FG_PIPELINE_STAGE_COUNT; stage++)
        manifest->stage_ranks[stage] = (uint8_t)stage;
    manifest->session.version = FG_MANIFEST_CONTRACT_VERSION;
    manifest->session.position_mode = FG_POSITION_TEXT;
    manifest->session.logical_context_tokens = FG_NATIVE_CONTEXT;
    manifest->session.gpu_index_tokens = FG_NATIVE_CONTEXT;
    manifest->session.host_page_cache_bytes =
        FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES;
    manifest->tensor_count = 1u;
    fg_tensor_record *record = &manifest->tensors[0];
    snprintf(record->name, sizeof(record->name), "token_embd.weight");
    record->bytes = FG_EMBEDDING_ARTIFACT_BYTES;
    record->ggml_type = 8u;
    record->dims = 2u;
    record->shape[0] = FG_HIDDEN_SIZE;
    record->shape[1] = FG_EMBEDDING_VOCAB_SIZE;
    record->rank = 0u;
    record->layer = UINT16_MAX;
    record->expert = UINT16_MAX;
    record->kind = FG_TENSOR_HOST_CACHE;
    record->layout = FG_TENSOR_LAYOUT_HOST_Q8_0;
    memcpy(record->sha256, digest, 32u);
}

static void make_row(uint8_t row[FG_EMBEDDING_ROW_BYTES], uint32_t variant) {
    for (uint32_t block = 0; block < FG_HIDDEN_SIZE / FG_QK8_0; block++) {
        float scale = variant == 0u ? 0.5f :
            (block & 1u) ? 2.0f : 0.25f;
        uint16_t half = fg_f32_to_f16(scale);
        uint8_t *packed = row + (size_t)block * FG_Q8_0_BLOCK_BYTES;
        memcpy(packed, &half, sizeof(half));
        for (uint32_t i = 0; i < FG_QK8_0; i++) {
            int32_t quant = variant == 0u ?
                (int32_t)((block + i) % 31u) - 15 :
                (int32_t)((block * 3u + i * 5u) % 23u) - 11;
            packed[sizeof(half) + i] = (uint8_t)(int8_t)quant;
        }
    }
}

static float expected_value(uint32_t variant, uint32_t element) {
    uint32_t block = element / FG_QK8_0;
    uint32_t i = element % FG_QK8_0;
    float scale = variant == 0u ? 0.5f :
        (block & 1u) ? 2.0f : 0.25f;
    int32_t quant = variant == 0u ?
        (int32_t)((block + i) % 31u) - 15 :
        (int32_t)((block * 3u + i * 5u) % 23u) - 11;
    return scale * (float)quant;
}

static bool write_row(FILE *file, uint32_t token, const uint8_t *row) {
    uint64_t offset = (uint64_t)token * FG_EMBEDDING_ROW_BYTES;
    return fseeko(file, (off_t)offset, SEEK_SET) == 0 &&
           fwrite(row, 1, FG_EMBEDDING_ROW_BYTES, file) ==
               FG_EMBEDDING_ROW_BYTES;
}

static bool create_artifact(const char *path) {
    FILE *file = fopen(path, "wb");
    if (!file) return false;
    uint8_t row[FG_EMBEDDING_ROW_BYTES];
    make_row(row, 0u);
    bool ok = write_row(file, 0u, row);
    make_row(row, 1u);
    ok = ok && write_row(file, 7u, row);
    memset(row, 0, sizeof(row));
    uint16_t infinity = fg_f32_to_f16(INFINITY);
    memcpy(row, &infinity, sizeof(infinity));
    row[sizeof(infinity)] = 1u;
    ok = ok && write_row(file, 11u, row);
    ok = ok && ftruncate(fileno(file), (off_t)FG_EMBEDDING_ARTIFACT_BYTES) == 0;
    return fclose(file) == 0 && ok;
}

static bool overwrite_artifact_row(const char *path, uint32_t token,
                                   uint32_t variant) {
    FILE *file = fopen(path, "r+b");
    if (!file) return false;
    uint8_t row[FG_EMBEDDING_ROW_BYTES];
    make_row(row, variant);
    bool ok = write_row(file, token, row);
    return fclose(file) == 0 && ok;
}

static void expect_open_failure(const fg_manifest *manifest, const char *dir,
                                uint32_t rank, fg_status expected) {
    fg_embedding *embedding = (fg_embedding *)(uintptr_t)1u;
    fg_error error = {0};
    CHECK(fg_embedding_open(&embedding, manifest, dir, rank, &error) == expected);
    CHECK(embedding == NULL);
}

static void verify_token(const float *boundary, uint32_t variant) {
    for (uint32_t group = 0; group < FG_EMBEDDING_GROUP_COUNT; group++) {
        const float *values = boundary + (size_t)group * FG_HIDDEN_SIZE;
        CHECK(memcmp(boundary, values, FG_HIDDEN_SIZE * sizeof(*values)) == 0);
        for (uint32_t element = 0; element < FG_HIDDEN_SIZE; element++)
            CHECK(values[element] == expected_value(variant, element));
    }
}

int main(void) {
    char directory[128], path[192];
    snprintf(directory, sizeof(directory), "test-embedding-%ld", (long)getpid());
    snprintf(path, sizeof(path), "%s/%s", directory,
             FG_TOKEN_EMBEDDING_ARTIFACT);
    CHECK(mkdir(directory, 0700) == 0);

    fg_manifest *manifest = calloc(1, sizeof(*manifest));
    CHECK(manifest != NULL);
    if (!manifest) return 1;
    uint8_t digest[32];
    for (uint32_t i = 0; i < sizeof(digest); i++)
        digest[i] = (uint8_t)(i + 1u);
    build_manifest(manifest, digest);

    expect_open_failure(manifest, directory, 0u, FG_ERR_IO);
    FILE *truncated = fopen(path, "wb");
    CHECK(truncated != NULL);
    if (truncated) {
        uint8_t row[FG_EMBEDDING_ROW_BYTES] = {0};
        CHECK(fwrite(row, 1, sizeof(row), truncated) == sizeof(row));
        CHECK(fclose(truncated) == 0);
    }
    expect_open_failure(manifest, directory, 0u, FG_ERR_MISMATCH);
    CHECK(unlink(path) == 0);

    CHECK(create_artifact(path));
    fg_error error = {0};
    CHECK(fg_sha256_file(path, digest, &error) == FG_OK);
    build_manifest(manifest, digest);

    fg_manifest wrong = *manifest;
    wrong.execution_mode = FG_EXECUTION_EXPERT_PARALLEL;
    expect_open_failure(&wrong, directory, 0u, FG_ERR_MISMATCH);
    wrong = *manifest;
    wrong.tensors[0].kind = FG_TENSOR_COMMON;
    expect_open_failure(&wrong, directory, 0u, FG_ERR_MISMATCH);
    wrong = *manifest;
    wrong.tensors[0].layout = FG_TENSOR_LAYOUT_GGML;
    expect_open_failure(&wrong, directory, 0u, FG_ERR_MISMATCH);
    wrong = *manifest;
    wrong.tensors[0].rank = 1u;
    expect_open_failure(&wrong, directory, 0u, FG_ERR_MISMATCH);
    wrong = *manifest;
    wrong.tensors[0].shape[0] = FG_HIDDEN_SIZE - FG_QK8_0;
    expect_open_failure(&wrong, directory, 0u, FG_ERR_FORMAT);
    expect_open_failure(manifest, directory, 1u, FG_ERR_MISMATCH);

    fg_embedding *embedding = NULL;
    CHECK(fg_embedding_open(&embedding, manifest, directory, 0u, &error) == FG_OK);
    CHECK(embedding != NULL);
    const uint32_t tokens[] = {7u, 0u, 7u};
    size_t capacity = 3u * FG_EMBEDDING_BOUNDARY_WIDTH;
    float *boundary = malloc(capacity * sizeof(*boundary));
    CHECK(boundary != NULL);
    if (embedding && boundary) {
        CHECK(fg_embedding_gather(embedding, tokens, 3u, boundary, capacity,
                                  &error) == FG_OK);
        verify_token(boundary, 1u);
        verify_token(boundary + FG_EMBEDDING_BOUNDARY_WIDTH, 0u);
        verify_token(boundary + 2u * FG_EMBEDDING_BOUNDARY_WIDTH, 1u);

        const uint32_t repeat[] = {0u, 7u};
        CHECK(fg_embedding_gather(embedding, repeat, 2u, boundary, capacity,
                                  &error) == FG_OK);
        verify_token(boundary, 0u);
        verify_token(boundary + FG_EMBEDDING_BOUNDARY_WIDTH, 1u);

        CHECK(overwrite_artifact_row(path, 7u, 0u));
        CHECK(fg_embedding_gather(embedding, repeat, 2u, boundary, capacity,
                                  &error) == FG_OK);
        verify_token(boundary, 0u);
        verify_token(boundary + FG_EMBEDDING_BOUNDARY_WIDTH, 1u);

        CHECK(truncate(path, 0) == 0);
        CHECK(fg_embedding_gather(embedding, tokens, 3u, boundary, capacity,
                                  &error) == FG_OK);
        verify_token(boundary, 1u);
        verify_token(boundary + FG_EMBEDDING_BOUNDARY_WIDTH, 0u);
        verify_token(boundary + 2u * FG_EMBEDDING_BOUNDARY_WIDTH, 1u);

        uint32_t invalid = FG_EMBEDDING_VOCAB_SIZE;
        CHECK(fg_embedding_gather(embedding, &invalid, 1u, boundary, capacity,
                                  &error) == FG_ERR_FORMAT);
        CHECK(fg_embedding_gather(embedding, tokens, 3u, boundary, capacity - 1u,
                                  &error) == FG_ERR_LIMIT);
        CHECK(fg_embedding_gather(embedding, tokens, SIZE_MAX, boundary, capacity,
                                  &error) == FG_ERR_LIMIT);
        CHECK(fg_embedding_gather(embedding, tokens, 0u, boundary, capacity,
                                  &error) == FG_ERR_ARGUMENT);
        uint32_t nonfinite = 11u;
        CHECK(fg_embedding_gather(embedding, &nonfinite, 1u, boundary, capacity,
                                  &error) == FG_ERR_FORMAT);
    }
    fg_embedding_close(embedding);
    fg_embedding_close(NULL);

    CHECK(create_artifact(path));
    FILE *corrupt = fopen(path, "r+b");
    CHECK(corrupt != NULL);
    if (corrupt) {
        CHECK(fseeko(corrupt, (off_t)(100u * FG_EMBEDDING_ROW_BYTES), SEEK_SET) == 0);
        CHECK(fputc(1, corrupt) != EOF);
        CHECK(fclose(corrupt) == 0);
    }
    expect_open_failure(manifest, directory, 0u, FG_ERR_MISMATCH);

    free(boundary);
    free(manifest);
    CHECK(unlink(path) == 0);
    CHECK(rmdir(directory) == 0);
    if (failures) fprintf(stderr, "%d embedding test(s) failed\n", failures);
    else puts("pipeline host embedding tests: PASS");
    return failures ? 1 : 0;
}
