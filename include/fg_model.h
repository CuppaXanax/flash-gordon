#ifndef FLASH_GORDON_MODEL_H
#define FLASH_GORDON_MODEL_H

#include "fg_manifest.h"
#include "fg_vk.h"

typedef struct fg_model fg_model;

fg_status fg_model_open(fg_model **out,const fg_manifest *manifest,const char *pack_dir,uint32_t rank,fg_error *err);
fg_status fg_model_open_replicated(fg_model **out,const fg_manifest *manifest,const char *pack_dir,uint32_t rank,fg_error *err);
void fg_model_close(fg_model *model);
fg_vk_context *fg_model_vk(fg_model *model);
fg_vk_tensor *fg_model_tensor(fg_model *model,const char *name);
const fg_tensor_record *fg_model_tensor_record(const fg_model *model,const char *name);
const fg_manifest *fg_model_manifest(const fg_model *model);
uint64_t fg_model_weight_bytes(const fg_model *model);
uint32_t fg_model_rank(const fg_model *model);

#endif
