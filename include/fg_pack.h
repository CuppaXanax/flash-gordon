#ifndef FLASH_GORDON_PACK_H
#define FLASH_GORDON_PACK_H

#include "fg_gguf.h"
#include "fg_manifest.h"

#define FG_TOWER_PACK_FILENAME "tower.fgw"
#define FG_TOWER_MANIFEST_FILENAME "tower.fgm"

typedef struct fg_pack_options {
    const char *output_dir;
    const char **source_paths;
    uint32_t source_count;
    const char *router_profile_path;
    const char *expert_map_path;
    uint32_t runtime_profile;
    bool dry_run;
    bool skip_model_validation; /* Unit-test fixtures only; not exposed by the CLI. */
} fg_pack_options;

fg_status fg_pack_run(const fg_pack_options *options, fg_error *err);

typedef struct fg_pack_tower_options {
    const char *output_dir;
    const char **source_paths;
    uint32_t source_count;
    bool dry_run;
} fg_pack_tower_options;

fg_status fg_pack_tower_run(const fg_pack_tower_options *options, fg_error *err);

typedef struct fg_verify_options {
    const char *manifest_path;
    const char *pack_dir;
    const char **source_paths;
    uint32_t source_count;
} fg_verify_options;

fg_status fg_pack_verify(const fg_verify_options *options, fg_error *err);

#endif
