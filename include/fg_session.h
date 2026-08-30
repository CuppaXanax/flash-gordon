#ifndef FLASH_GORDON_SESSION_H
#define FLASH_GORDON_SESSION_H

#include "fg_manifest.h"

#define FG_SESSION_IDENTITY_VERSION 1u
#define FG_SESSION_IDENTITY_WIRE_BYTES 272u
#define FG_SESSION_FRONTIER_VERSION 1u
#define FG_SESSION_FRONTIER_HEADER_BYTES 1356u
#define FG_SESSION_FRONTIER_DIGEST_BYTES 32u

typedef struct fg_session_identity {
    uint32_t version;
    uint8_t model_sha256[32];
    uint8_t tokenizer_sha256[32];
    uint8_t quantization_sha256[32];
    uint8_t manifest_sha256[32];
    uint8_t rope_policy_sha256[32];
    uint8_t vision_projector_sha256[32];
    uint8_t state_format_sha256[32];
    uint8_t identity_sha256[32];
} fg_session_identity;

typedef struct fg_session_frontier {
    uint32_t version;
    fg_position_mode position_mode;
    uint64_t generation;
    uint64_t committed_tokens;
    uint64_t token_count;
    bool next_token_valid;
    uint32_t next_token;
    float next_logit;
    uint32_t position[4];
    uint8_t identity_sha256[32];
    uint8_t rendered_transcript_sha256[32];
    uint8_t next_token_state_sha256[32];
    uint8_t token_history_sha256[32];
    uint64_t qsa_lengths[FG_LAYER_COUNT];
    uint64_t gdn_lengths[FG_LAYER_COUNT];
    uint64_t ple_lengths[FG_LAYER_COUNT];
    int32_t *tokens;
    uint8_t frontier_sha256[32];
} fg_session_frontier;

fg_status fg_session_identity_from_manifest(const fg_manifest *manifest,
                                            fg_session_identity *identity,
                                            fg_error *err);
fg_status fg_session_identity_encode(uint8_t output[FG_SESSION_IDENTITY_WIRE_BYTES],
                                     const fg_session_identity *identity, fg_error *err);
fg_status fg_session_identity_decode(fg_session_identity *identity, const uint8_t *payload,
                                     uint32_t bytes, fg_error *err);
fg_status fg_session_identity_validate_compatible(const fg_session_identity *expected,
                                                  const fg_session_identity *actual,
                                                  fg_error *err);

fg_status fg_session_frontier_encode(uint8_t *output, uint32_t capacity, uint32_t *bytes,
                                     const fg_session_frontier *frontier, fg_error *err);
fg_status fg_session_frontier_decode(fg_session_frontier *frontier, int32_t *token_storage,
                                     uint64_t token_capacity, const uint8_t *payload,
                                     uint32_t bytes, fg_error *err);
fg_status fg_session_frontier_validate_compatible(const fg_session_identity *identity,
                                                  const fg_session_frontier *frontier,
                                                  fg_position_mode position_mode,
                                                  fg_error *err);

#endif
