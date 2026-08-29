#ifndef FLASH_GORDON_QSA_H
#define FLASH_GORDON_QSA_H

#include "fg_model.h"

typedef struct fg_qsa_session fg_qsa_session;

fg_status fg_qsa_session_open(fg_qsa_session **out,fg_model *model,const char *state_path,
                              bool create,fg_error *err);
fg_status fg_qsa_session_open_decode(fg_qsa_session **out,fg_model *model,const char *state_path,
                                     uint32_t resident_tokens,fg_error *err);
void fg_qsa_session_close(fg_qsa_session *session);
uint32_t fg_qsa_session_tokens(const fg_qsa_session *session,uint32_t layer);
void fg_qsa_session_set_tokens(fg_qsa_session *session,uint32_t tokens);
fg_status fg_qsa_session_decode(fg_qsa_session *session,uint32_t layer,uint32_t token_index,
                                const uint32_t position[3],const fg_vk_tensor *hidden,
                                fg_vk_tensor **output,fg_error *err);
fg_status fg_qsa_session_prefill(fg_qsa_session *session,uint32_t layer,uint32_t first_token,
                                 const uint32_t *positions,uint32_t token_count,
                                 const fg_vk_tensor *hidden,fg_vk_tensor **output,fg_error *err);

#endif
