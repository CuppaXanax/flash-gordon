#ifndef FLASH_GORDON_PACK_H
#define FLASH_GORDON_PACK_H

#include "fg_gguf.h"
#include "fg_manifest.h"

typedef struct fg_pack_options {
    const char *output_dir;
    const char **source_paths;
    uint32_t source_count;
    const char *router_profile_path;
    bool dry_run;
    bool skip_model_validation; /* Unit-test fixtures only; not exposed by the CLI. */
} fg_pack_options;

fg_status fg_pack_run(const fg_pack_options *options, fg_error *err);

typedef struct fg_verify_options {
    const char *manifest_path;
    const char *pack_dir;
    const char **source_paths;
    uint32_t source_count;
} fg_verify_options;

fg_status fg_pack_verify(const fg_verify_options *options, fg_error *err);

#endif
