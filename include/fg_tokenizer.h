#ifndef FLASH_GORDON_TOKENIZER_H
#define FLASH_GORDON_TOKENIZER_H

#include "fg_manifest.h"

typedef struct fg_tokenizer fg_tokenizer;
typedef struct fg_tokens {uint32_t *data;size_t count,capacity;} fg_tokens;

fg_status fg_tokenizer_pack_gguf(const char *source_path,const char *output_dir,
                                 fg_manifest *manifest,fg_error *err);
fg_status fg_tokenizer_open(fg_tokenizer **out,const char *pack_dir,const fg_manifest *manifest,
                            fg_error *err);
void fg_tokenizer_close(fg_tokenizer *tokenizer);
uint32_t fg_tokenizer_vocab_size(const fg_tokenizer *tokenizer);
uint32_t fg_tokenizer_bos(const fg_tokenizer *tokenizer);
uint32_t fg_tokenizer_eos(const fg_tokenizer *tokenizer);
bool fg_tokenizer_add_bos(const fg_tokenizer *tokenizer);
fg_status fg_tokenizer_lookup(const fg_tokenizer *tokenizer,const char *text,size_t bytes,
                              uint32_t *token,fg_error *err);
fg_status fg_tokenizer_token(const fg_tokenizer *tokenizer,uint32_t token,const char **text,
                             size_t *bytes,uint32_t *type,fg_error *err);
void fg_tokens_free(fg_tokens *tokens);
fg_status fg_tokenizer_encode(const fg_tokenizer *tokenizer,const char *text,bool allow_special,
                              fg_tokens *tokens,fg_error *err);
fg_status fg_tokenizer_decode_token(const fg_tokenizer *tokenizer,uint32_t token,char *output,
                                    size_t capacity,size_t *bytes,fg_error *err);
fg_status fg_tokenizer_validate_qwen38(const fg_tokenizer *tokenizer,fg_error *err);

#endif
