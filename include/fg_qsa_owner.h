#ifndef FLASH_GORDON_QSA_OWNER_H
#define FLASH_GORDON_QSA_OWNER_H

#include "fg_protocol.h"

typedef struct fg_qsa_owner_guard {
    uint64_t session_nonce;
    uint64_t generation;
    uint32_t context_limit;
    uint32_t next_token[FG_LAYER_COUNT];
    uint32_t next_append_batch;
    uint32_t next_fetch_batch;
    uint8_t rank;
    fg_position_mode position_mode;
    bool active;
} fg_qsa_owner_guard;

void fg_qsa_owner_guard_init(fg_qsa_owner_guard *guard,uint32_t rank);
fg_status fg_qsa_owner_guard_begin(fg_qsa_owner_guard *guard,const fg_manifest *manifest,
                                   const fg_session_identity *identity,uint64_t request_id,
                                   const fg_owner_session_control *control,fg_error *err);
fg_status fg_qsa_owner_guard_validate_decode(const fg_qsa_owner_guard *guard,
                                             const fg_manifest *manifest,uint64_t request_id,
                                             uint32_t sequence,const fg_qsa_block_work *work,
                                             fg_error *err);
fg_status fg_qsa_owner_guard_validate_prefill(const fg_qsa_owner_guard *guard,
                                              const fg_manifest *manifest,uint64_t request_id,
                                              uint32_t sequence,
                                              const fg_qsa_block_prefill_work *work,
                                              fg_error *err);
fg_status fg_qsa_owner_guard_commit(fg_qsa_owner_guard *guard,uint32_t layer,
                                    uint32_t first_token,uint32_t token_count,
                                    fg_error *err);
fg_status fg_qsa_owner_validate_decode_result(const fg_manifest *manifest,uint32_t owner_rank,
                                              uint32_t sequence,
                                              const fg_qsa_block_result *result,fg_error *err);
fg_status fg_qsa_owner_validate_prefill_result(const fg_manifest *manifest,uint32_t owner_rank,
                                               uint32_t sequence,uint16_t token_count,
                                               const fg_qsa_block_prefill_result *result,
                                               fg_error *err);
fg_status fg_qsa_owner_guard_accept_append(fg_qsa_owner_guard *guard,
                                           const fg_manifest *manifest,uint64_t request_id,
                                           const fg_qsa_page_batch *batch,
                                           fg_error *err);
fg_status fg_qsa_owner_guard_accept_fetch(fg_qsa_owner_guard *guard,
                                          const fg_manifest *manifest,uint64_t request_id,
                                          const fg_qsa_page_batch *batch,
                                          fg_error *err);
fg_status fg_qsa_owner_guard_accept_barrier(const fg_qsa_owner_guard *guard,
                                            uint64_t request_id,
                                            const fg_qsa_page_barrier *barrier,
                                            fg_error *err);

#endif
