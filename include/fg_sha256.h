#ifndef FLASH_GORDON_SHA256_H
#define FLASH_GORDON_SHA256_H

#include "fg.h"

typedef struct fg_sha256 {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t block[64];
    size_t block_bytes;
} fg_sha256;

void fg_sha256_init(fg_sha256 *ctx);
void fg_sha256_update(fg_sha256 *ctx, const void *data, size_t bytes);
void fg_sha256_final(fg_sha256 *ctx, uint8_t digest[32]);
fg_status fg_sha256_file(const char *path, uint8_t digest[32], fg_error *err);
void fg_sha256_hex(const uint8_t digest[32], char out[65]);

#endif
