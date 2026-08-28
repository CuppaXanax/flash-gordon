#ifndef FLASH_GORDON_RUNTIME_H
#define FLASH_GORDON_RUNTIME_H

#include "fg_manifest.h"

fg_status fg_rank_main(const char *manifest_path, uint32_t rank, fg_error *err);
fg_status fg_serve_main(const char *manifest_path, fg_error *err);
fg_status fg_bench_main(const char *manifest_path, fg_error *err);
fg_status fg_eval_main(const char *manifest_path,const char *prompt,uint32_t generate,fg_error *err);

#endif
