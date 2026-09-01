#include "fg_embedding.h"
#include "fg_quant.h"
#include "fg_sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct fg_embedding {
    const uint8_t *mapping;
    size_t bytes;
};

_Static_assert(FG_EMBEDDING_ROW_BYTES == 2720u,
               "Qwen3.8 token embedding rows must be 2720 bytes");

static bool pipeline_profile_matches(const fg_manifest *manifest) {
    return manifest->magic == FG_MANIFEST_MAGIC &&
           manifest->format_version == FG_MANIFEST_FORMAT_VERSION &&
           manifest->protocol_version == FG_PIPELINE_PROTOCOL_VERSION &&
           manifest->header_bytes == sizeof(*manifest) &&
           manifest->rank_count == FG_RANK_COUNT &&
           manifest->layer_count == FG_LAYER_COUNT &&
           manifest->expert_count == FG_EXPERT_COUNT &&
           manifest->hidden_size == FG_HIDDEN_SIZE &&
           manifest->top_k == FG_TOP_K &&
           manifest->native_context == FG_NATIVE_CONTEXT &&
           manifest->max_context == FG_NATIVE_CONTEXT &&
           manifest->prefill_microbatch == FG_PIPELINE_DEFAULT_MICROBATCH &&
           manifest->prefill_window == FG_DEFAULT_WINDOW &&
           manifest->execution_mode == FG_EXECUTION_PIPELINE &&
           manifest->stage_count == FG_PIPELINE_STAGE_COUNT &&
           manifest->slot_count == FG_PIPELINE_DEFAULT_SLOT_COUNT &&
           manifest->session.version == FG_MANIFEST_CONTRACT_VERSION &&
           manifest->session.position_mode == FG_POSITION_TEXT &&
           manifest->session.logical_context_tokens == FG_NATIVE_CONTEXT &&
           manifest->session.gpu_index_tokens == FG_NATIVE_CONTEXT &&
           manifest->session.qsa_hot_record_tokens == 0u &&
           manifest->session.host_page_cache_bytes ==
               FG_RUNTIME_PROFILE_NATIVE_262K_PAGE_CACHE_BYTES;
}

static const fg_tensor_record *embedding_record(const fg_manifest *manifest,
                                                fg_error *err) {
    const fg_tensor_record *found = NULL;
    if (manifest->tensor_count > FG_MAX_TENSORS) {
        fg_error_set(err, FG_ERR_FORMAT,
                     "pipeline embedding manifest tensor count is invalid");
        return NULL;
    }
    for (uint32_t i = 0; i < manifest->tensor_count; i++) {
        const fg_tensor_record *record = &manifest->tensors[i];
        if (strncmp(record->name, "token_embd.weight", sizeof(record->name)) != 0)
            continue;
        if (found) {
            fg_error_set(err, FG_ERR_MISMATCH,
                         "pipeline manifest has duplicate token embeddings");
            return NULL;
        }
        found = record;
    }
    if (!found)
        fg_error_set(err, FG_ERR_MISMATCH,
                     "pipeline manifest is missing token_embd.weight");
    return found;
}

static fg_status validate_record(const fg_manifest *manifest,
                                 const fg_tensor_record *record,
                                 uint32_t rank, fg_error *err) {
    if (rank >= FG_RANK_COUNT || manifest->stage_ranks[0] >= FG_RANK_COUNT ||
        rank != manifest->stage_ranks[0]) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "token embedding may only open on pipeline stage 0");
        return FG_ERR_MISMATCH;
    }
    if (record->kind != FG_TENSOR_HOST_CACHE ||
        record->layout != FG_TENSOR_LAYOUT_HOST_Q8_0 ||
        record->rank != rank) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "token embedding kind, layout, or owner is invalid");
        return FG_ERR_MISMATCH;
    }
    if (!fg_embedding_record_metadata_valid(record, rank)) {
        fg_error_set(err, FG_ERR_FORMAT,
                     "token embedding Q8_0 row geometry is invalid");
        return FG_ERR_FORMAT;
    }
    return FG_OK;
}

static fg_status close_embedding_fd(int *fd, const char *description,
                                   fg_status status, fg_error *err) {
    if (*fd < 0) return status;
    int closing = *fd;
    *fd = -1;
    if (close(closing) != 0 && status == FG_OK) {
        fg_error_set(err, FG_ERR_IO, "close %s: %s", description,
                     strerror(errno));
        return FG_ERR_IO;
    }
    return status;
}

static fg_status snapshot_embedding(int source_fd, int snapshot_fd,
                                   size_t expected_bytes,
                                   uint8_t digest[32], fg_error *err) {
    uint8_t buffer[1u << 20u];
    fg_sha256 hash;
    fg_sha256_init(&hash);
    size_t copied = 0u;
    while (copied < expected_bytes) {
        size_t request = expected_bytes - copied;
        if (request > sizeof(buffer)) request = sizeof(buffer);
        ssize_t got = read(source_fd, buffer, request);
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) {
            fg_error_set(err, FG_ERR_IO, "read token embedding: %s",
                         strerror(errno));
            return FG_ERR_IO;
        }
        if (!got) {
            fg_error_set(err, FG_ERR_MISMATCH,
                         "token embedding changed size while snapshotting");
            return FG_ERR_MISMATCH;
        }
        size_t written = 0u;
        while (written < (size_t)got) {
            ssize_t put = write(snapshot_fd, buffer + written,
                               (size_t)got - written);
            if (put < 0 && errno == EINTR) continue;
            if (put < 0) {
                fg_error_set(err, FG_ERR_IO,
                            "write token embedding snapshot: %s",
                            strerror(errno));
                return FG_ERR_IO;
            }
            if (!put) {
                fg_error_set(err, FG_ERR_IO,
                            "write token embedding snapshot returned zero");
                return FG_ERR_IO;
            }
            written += (size_t)put;
        }
        fg_sha256_update(&hash, buffer, (size_t)got);
        copied += (size_t)got;
    }
    for (;;) {
        uint8_t extra;
        ssize_t got = read(source_fd, &extra, sizeof(extra));
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) {
            fg_error_set(err, FG_ERR_IO, "read token embedding: %s",
                         strerror(errno));
            return FG_ERR_IO;
        }
        if (got) {
            fg_error_set(err, FG_ERR_MISMATCH,
                         "token embedding changed size while snapshotting");
            return FG_ERR_MISMATCH;
        }
        break;
    }
    fg_sha256_final(&hash, digest);
    return FG_OK;
}

fg_status fg_embedding_open(fg_embedding **out, const fg_manifest *manifest,
                            const char *pack_dir, uint32_t rank, fg_error *err) {
    if (!out || !manifest || !pack_dir || !*pack_dir) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "invalid pipeline embedding open arguments");
        return FG_ERR_ARGUMENT;
    }
    *out = NULL;
    if (!pipeline_profile_matches(manifest)) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "token embedding requires the pipeline-8stage-262k profile");
        return FG_ERR_MISMATCH;
    }
    const fg_tensor_record *record = embedding_record(manifest, err);
    if (!record) return err && err->code ? err->code : FG_ERR_MISMATCH;
    fg_status status = validate_record(manifest, record, rank, err);
    if (status != FG_OK) return status;

    char path[4096];
    int path_bytes = snprintf(path, sizeof(path), "%s/%s", pack_dir,
                              FG_TOKEN_EMBEDDING_ARTIFACT);
    if (path_bytes < 0 || (size_t)path_bytes >= sizeof(path)) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "token embedding artifact path is too long");
        return FG_ERR_ARGUMENT;
    }
    int source_fd;
    do {
        source_fd = open(path, O_RDONLY | O_CLOEXEC);
    } while (source_fd < 0 && errno == EINTR);
    if (source_fd < 0) {
        fg_error_set(err, FG_ERR_IO, "open token embedding: %s",
                     strerror(errno));
        return FG_ERR_IO;
    }
    struct stat info;
    while (fstat(source_fd, &info) != 0) {
        if (errno == EINTR) continue;
        fg_error_set(err, FG_ERR_IO, "stat token embedding: %s",
                     strerror(errno));
        return close_embedding_fd(&source_fd, "token embedding source",
                                 FG_ERR_IO, err);
    }
    if (!S_ISREG(info.st_mode) || info.st_size < 0 ||
        (uint64_t)info.st_size != record->bytes) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "token embedding artifact size is not exactly %llu bytes",
                     (unsigned long long)record->bytes);
        return close_embedding_fd(&source_fd, "token embedding source",
                                 FG_ERR_MISMATCH, err);
    }
    if (record->bytes > SIZE_MAX) {
        fg_error_set(err, FG_ERR_LIMIT,
                     "token embedding artifact cannot be mapped on this host");
        return close_embedding_fd(&source_fd, "token embedding source",
                                 FG_ERR_LIMIT, err);
    }
    off_t snapshot_bytes = (off_t)record->bytes;
    if (snapshot_bytes < 0 || (uint64_t)snapshot_bytes != record->bytes) {
        fg_error_set(err, FG_ERR_LIMIT,
                     "token embedding artifact cannot be snapshotted on this host");
        return close_embedding_fd(&source_fd, "token embedding source",
                                 FG_ERR_LIMIT, err);
    }

    int snapshot_fd;
    do {
        snapshot_fd = memfd_create("flash-gordon-token-embedding",
                                  MFD_CLOEXEC | MFD_ALLOW_SEALING);
    } while (snapshot_fd < 0 && errno == EINTR);
    if (snapshot_fd < 0) {
        fg_error_set(err, FG_ERR_IO, "create token embedding snapshot: %s",
                     strerror(errno));
        return close_embedding_fd(&source_fd, "token embedding source",
                                 FG_ERR_IO, err);
    }
    int truncated;
    do {
        truncated = ftruncate(snapshot_fd, snapshot_bytes);
    } while (truncated != 0 && errno == EINTR);
    if (truncated != 0) {
        fg_error_set(err, FG_ERR_IO, "size token embedding snapshot: %s",
                     strerror(errno));
        fg_status cleanup = close_embedding_fd(
            &source_fd, "token embedding source", FG_ERR_IO, err);
        return close_embedding_fd(&snapshot_fd, "token embedding snapshot",
                                 cleanup, err);
    }

    uint8_t digest[32];
    status = snapshot_embedding(source_fd, snapshot_fd, (size_t)record->bytes,
                               digest, err);
    if (status != FG_OK) {
        status = close_embedding_fd(&source_fd, "token embedding source",
                                   status, err);
        return close_embedding_fd(&snapshot_fd, "token embedding snapshot",
                                 status, err);
    }
    if (memcmp(digest, record->sha256, sizeof(digest)) != 0) {
        fg_error_set(err, FG_ERR_MISMATCH,
                     "token embedding artifact SHA-256 mismatch");
        status = close_embedding_fd(&source_fd, "token embedding source",
                                   FG_ERR_MISMATCH, err);
        return close_embedding_fd(&snapshot_fd, "token embedding snapshot",
                                 status, err);
    }

    const int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    int sealed;
    do {
        sealed = fcntl(snapshot_fd, F_ADD_SEALS, seals);
    } while (sealed < 0 && errno == EINTR);
    if (sealed < 0) {
        fg_error_set(err, FG_ERR_IO, "seal token embedding snapshot: %s",
                     strerror(errno));
        status = close_embedding_fd(&source_fd, "token embedding source",
                                   FG_ERR_IO, err);
        return close_embedding_fd(&snapshot_fd, "token embedding snapshot",
                                 status, err);
    }

    void *mapping;
    do {
        mapping = mmap(NULL, (size_t)record->bytes, PROT_READ, MAP_PRIVATE,
                       snapshot_fd, 0);
    } while (mapping == MAP_FAILED && errno == EINTR);
    if (mapping == MAP_FAILED) {
        fg_error_set(err, FG_ERR_IO, "mmap token embedding snapshot: %s",
                     strerror(errno));
        status = close_embedding_fd(&source_fd, "token embedding source",
                                   FG_ERR_IO, err);
        return close_embedding_fd(&snapshot_fd, "token embedding snapshot",
                                 status, err);
    }

    status = close_embedding_fd(&source_fd, "token embedding source",
                               FG_OK, err);
    status = close_embedding_fd(&snapshot_fd, "token embedding snapshot",
                               status, err);
    if (status != FG_OK) {
        munmap(mapping, (size_t)record->bytes);
        return status;
    }
    fg_embedding *embedding = calloc(1, sizeof(*embedding));
    if (!embedding) {
        munmap(mapping, (size_t)record->bytes);
        fg_error_set(err, FG_ERR_OOM,
                     "allocate pipeline embedding metadata");
        return FG_ERR_OOM;
    }
    embedding->mapping = mapping;
    embedding->bytes = (size_t)record->bytes;
    *out = embedding;
    return FG_OK;
}

void fg_embedding_close(fg_embedding *embedding) {
    if (!embedding) return;
    if (embedding->mapping)
        munmap((void *)embedding->mapping, embedding->bytes);
    free(embedding);
}

fg_status fg_embedding_gather(const fg_embedding *embedding,
                              const uint32_t *token_ids, size_t token_count,
                              float *boundary, size_t boundary_capacity,
                              fg_error *err) {
    if (!embedding || !token_ids || !token_count || !boundary) {
        fg_error_set(err, FG_ERR_ARGUMENT,
                     "invalid pipeline embedding gather arguments");
        return FG_ERR_ARGUMENT;
    }
    if (token_count >
        SIZE_MAX / sizeof(*boundary) / FG_EMBEDDING_BOUNDARY_WIDTH) {
        fg_error_set(err, FG_ERR_LIMIT,
                     "pipeline embedding boundary size overflows");
        return FG_ERR_LIMIT;
    }
    size_t required = token_count * FG_EMBEDDING_BOUNDARY_WIDTH;
    if (boundary_capacity < required) {
        fg_error_set(err, FG_ERR_LIMIT,
                     "pipeline embedding boundary capacity is too small");
        return FG_ERR_LIMIT;
    }
    for (size_t token = 0; token < token_count; token++) {
        uint32_t token_id = token_ids[token];
        if (token_id >= FG_EMBEDDING_VOCAB_SIZE) {
            fg_error_set(err, FG_ERR_FORMAT,
                         "token %zu is outside the Qwen3.8 vocabulary", token);
            return FG_ERR_FORMAT;
        }
        const uint8_t *row = embedding->mapping +
            (size_t)token_id * FG_EMBEDDING_ROW_BYTES;
        float *destination = boundary + token * FG_EMBEDDING_BOUNDARY_WIDTH;
        fg_dequantize_q8_0(row, destination, FG_HIDDEN_SIZE);
        for (uint32_t element = 0; element < FG_HIDDEN_SIZE; element++) {
            if (!isfinite(destination[element])) {
                fg_error_set(err, FG_ERR_FORMAT,
                             "non-finite token embedding at token %zu element %u",
                             token, element);
                return FG_ERR_FORMAT;
            }
        }
        for (uint32_t group = 1; group < FG_EMBEDDING_GROUP_COUNT; group++)
            memcpy(destination + (size_t)group * FG_HIDDEN_SIZE, destination,
                   FG_HIDDEN_SIZE * sizeof(*destination));
    }
    return FG_OK;
}
