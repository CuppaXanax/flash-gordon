#ifndef FLASH_GORDON_MTP_WIRE_H
#define FLASH_GORDON_MTP_WIRE_H

#include "fg_mtp.h"
#include "fg_protocol.h"

#define FG_MTP_WIRE_VERSION 1u
#define FG_MTP_NATIVE_HEADER_BYTES 24u
#define FG_MTP_NATIVE_ACK_BYTES 16u
#define FG_MTP_OVERLAY_PATH_MAX 1023u
#define FG_MTP_HEAD_HEADER_BYTES 16u
#define FG_MTP_HEAD_WORK_MAX_BYTES \
    (FG_MTP_HEAD_HEADER_BYTES+FG_MTP_MAX_VERIFY*FG_HIDDEN_SIZE*4u)
#define FG_MTP_HEAD_PARTIAL_MAX_BYTES \
    (FG_MTP_HEAD_HEADER_BYTES+FG_MTP_MAX_VERIFY*8u)
#define FG_MTP_NATIVE_WORK_MAX_BYTES \
    (FG_MTP_NATIVE_HEADER_BYTES+FG_PREFILL_MAX_TOKENS*4u+2u*FG_MTP_OVERLAY_PATH_MAX)
#define FG_MTP_VERIFY_REPLAY 1u
#define FG_MTP_VERIFY_RESULT_BYTES 16u

typedef struct fg_mtp_verify_result {
    uint8_t source_rank;
    uint32_t first_token,count;
} fg_mtp_verify_result;

typedef enum fg_mtp_native_operation {
    FG_MTP_NATIVE_CONFIG = 1,
    FG_MTP_NATIVE_PREFIX = 2,
    FG_MTP_NATIVE_DRAFT = 3,
    FG_MTP_NATIVE_FINISH = 4,
    FG_MTP_NATIVE_ADVANCE = 5,
    FG_MTP_NATIVE_PREPARE = 6
} fg_mtp_native_operation;

/* FINISH has count-1 token IDs; CONFIG has overlay and target-embedding paths.
 * The final owner reads exact production Q8 rows locally, without another
 * embedding tensor on the GPU or a coordinator activation round trip. */
typedef struct fg_mtp_native_work {
    fg_mtp_native_operation operation;
    uint32_t first_token,count,argument,limit;
    const uint32_t *tokens;
    const char *overlay,*embedding_file;
} fg_mtp_native_work;

typedef struct fg_mtp_native_ack {
    fg_mtp_native_operation operation;
    uint32_t first_token,count,frontier;
    fg_status status;
} fg_mtp_native_ack;

typedef enum fg_mtp_head_kind {
    FG_MTP_HEAD_DRAFT = 1,
    FG_MTP_HEAD_VERIFY = 2
} fg_mtp_head_kind;

typedef struct fg_mtp_head_work {
    fg_mtp_head_kind kind;
    uint8_t source_rank,destination_rank;
    uint32_t first_token,count;
    const float *hidden;
} fg_mtp_head_work;

typedef struct fg_mtp_head_partial {
    fg_mtp_head_kind kind;
    uint8_t source_rank,destination_rank;
    uint32_t first_token,count;
    float values[FG_MTP_MAX_VERIFY];
    uint32_t ids[FG_MTP_MAX_VERIFY];
} fg_mtp_head_partial;

fg_status fg_mtp_native_work_encode(uint8_t *output,uint32_t capacity,
    uint32_t *bytes,const fg_mtp_native_work *work,fg_error *err);
fg_status fg_mtp_native_work_decode(fg_mtp_native_work *work,uint32_t *tokens,
    uint32_t token_capacity,char overlay[FG_MTP_OVERLAY_PATH_MAX+1u],
    char embedding_file[FG_MTP_OVERLAY_PATH_MAX+1u],
    const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_mtp_native_ack_encode(uint8_t output[FG_MTP_NATIVE_ACK_BYTES],
    const fg_mtp_native_ack *ack,fg_error *err);
fg_status fg_mtp_native_ack_decode(fg_mtp_native_ack *ack,
    const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_mtp_head_work_encode(uint8_t *output,uint32_t capacity,
    uint32_t *bytes,const fg_mtp_head_work *work,fg_error *err);
fg_status fg_mtp_head_work_decode(fg_mtp_head_work *work,float *hidden,
    uint64_t hidden_values,const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_mtp_head_partial_encode(uint8_t *output,uint32_t capacity,
    uint32_t *bytes,const fg_mtp_head_partial *partial,fg_error *err);
fg_status fg_mtp_head_partial_decode(fg_mtp_head_partial *partial,
    const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_mtp_verify_result_encode(uint8_t output[FG_MTP_VERIFY_RESULT_BYTES],
    const fg_mtp_verify_result *result,fg_error *err);
fg_status fg_mtp_verify_result_decode(fg_mtp_verify_result *result,
    const uint8_t *payload,uint32_t bytes,fg_error *err);
fg_status fg_mtp_prefill_tokens_encode(uint8_t *wire,uint32_t capacity,
    uint32_t *bytes,const uint32_t *next_tokens,uint32_t count,fg_error *err);
fg_status fg_mtp_prefill_tokens_decode(uint32_t *next_tokens,uint32_t capacity,
    uint32_t *body_bytes,const uint8_t *wire,uint32_t bytes,fg_error *err);

#endif
