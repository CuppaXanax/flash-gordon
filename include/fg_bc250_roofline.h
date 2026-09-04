#ifndef FLASH_GORDON_BC250_ROOFLINE_H
#define FLASH_GORDON_BC250_ROOFLINE_H

#include <stddef.h>
#include <stdint.h>

#define FG_BC250_ROOFLINE_SCHEMA "fg.bc250.roofline.v1"

typedef struct fg_bc250_roofline_record {
    /* NULL selects FG_BC250_ROOFLINE_SCHEMA. */
    const char *schema;
    const char *benchmark;
    const char *name;
    const char *shape;
    const char *access_pattern;
    uint32_t batch;
    uint32_t tokens;
    uint32_t iterations;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t operations;
    double gpu_ms;
    double wall_ms;
    double derived_gbps;
    double derived_flops;
    const char *device;
    /* Optional QSA curve geometry.  NULL component keeps legacy records. */
    const char *component;
    uint32_t visible_tokens;
    uint32_t query_tokens;
    uint32_t complete_blocks;
    uint32_t selector_groups;
    uint32_t candidate_count;
    uint32_t merge_passes;
    uint32_t selected_stride;
    uint32_t capacity;
    uint32_t segment_capacity;
    uint64_t scratch_bytes;
} fg_bc250_roofline_record;

int fg_bc250_q8_0_weight_bytes(uint32_t input_width,uint32_t output_width,
                               uint64_t *bytes);
int fg_bc250_linear_accounting(uint64_t source_bytes,int copy,
                               uint64_t *bytes_read,uint64_t *bytes_written,
                               uint64_t *operations);
int fg_bc250_q8_0_logical_traffic(uint32_t input_width,uint32_t output_width,
                                 uint32_t tokens,int cooked,
                                 uint64_t *bytes_read,uint64_t *bytes_written,
                                 uint64_t *operations);
int fg_bc250_roofline_record_format(char *output,size_t capacity,
                                    const fg_bc250_roofline_record *record);

#endif
