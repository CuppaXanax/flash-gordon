#ifndef FLASH_GORDON_LEDGER_H
#define FLASH_GORDON_LEDGER_H

#include "fg_runtime.h"

/* Single-line, parseable startup ledger.  Every field is derived from the
 * sealed manifest and the resolved runtime options, never from prior docs. */
#define FG_LEDGER_LINE_MAX 2048u
#define FG_LEDGER_PREFIX "FG_LEDGER"

typedef struct fg_ledger_inputs {
    const fg_manifest *manifest;
    const fg_runtime_options *options;
    uint32_t rank;
    uint32_t prefill_frames;
    uint32_t qsa_cache_pages;
    uint64_t qsa_cache_bytes;
    bool ring_prefill;
    bool ring_decode;
} fg_ledger_inputs;

fg_status fg_ledger_format(char *buffer,size_t capacity,
                           const fg_ledger_inputs *inputs,fg_error *err);

#endif
