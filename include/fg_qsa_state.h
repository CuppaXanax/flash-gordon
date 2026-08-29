#ifndef FLASH_GORDON_QSA_STATE_H
#define FLASH_GORDON_QSA_STATE_H

#include "fg_q38_schema.h"

typedef struct fg_qsa_state fg_qsa_state;
#define FG_QSA_MAX_SELECTED_BLOCKS 512u

fg_status fg_qsa_state_open(fg_qsa_state **out,const char *path,const uint8_t *layers,
                            uint32_t layer_count,uint32_t max_context,bool create,
                            fg_error *err);
void fg_qsa_state_close(fg_qsa_state *state);
fg_status fg_qsa_state_write_block(fg_qsa_state *state,uint32_t layer_slot,uint32_t block,
                                   const uint8_t *records,
                                   uint32_t committed_tokens,fg_error *err);
fg_status fg_qsa_state_read_block(fg_qsa_state *state,uint32_t layer_slot,uint32_t block,
                                  uint8_t *records,
                                  uint32_t *committed_tokens,fg_error *err);
fg_status fg_qsa_state_read_blocks(fg_qsa_state *state,uint32_t layer_slot,const uint32_t *blocks,
                                   uint32_t block_count,uint8_t *records,uint32_t *committed_tokens,
                                   fg_error *err);
uint64_t fg_qsa_state_required_bytes(uint32_t layer_count,uint32_t max_context);
uint32_t fg_qsa_state_layer_tokens(const fg_qsa_state *state,uint32_t layer_slot);
void fg_qsa_state_set_layer_tokens(fg_qsa_state *state,uint32_t layer_slot,uint32_t tokens);
void fg_qsa_encode_token_record(const float key[FG_Q38_ATTN_KV_WIDTH],
                                const float value[FG_Q38_ATTN_KV_WIDTH],
                                uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES]);
void fg_qsa_encode_full_token_record(const float key[FG_Q38_ATTN_KV_WIDTH],
                                     const float value[FG_Q38_ATTN_KV_WIDTH],
                                     const float index_key[FG_Q38_INDEX_WIDTH],
                                     const uint32_t position[3],
                                     uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES]);
void fg_qsa_decode_token_record(const uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES],
                                float key[FG_Q38_ATTN_KV_WIDTH],
                                float value[FG_Q38_ATTN_KV_WIDTH]);
void fg_qsa_decode_token_metadata(const uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES],
                                  float index_key[FG_Q38_INDEX_WIDTH],
                                  uint32_t position[3]);

#endif
