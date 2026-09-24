/* Batch-2 decode MMV prototype measurement.
 *
 * For each production dense shape: build a cooked q8_0 weight, run the decode
 * r8 path for two tokens (two single-token dispatches, as the serial ring does
 * per slot) and the batch-2 kernel once; verify per-token bit identity and
 * report the GPU-time amortization.  Read-only with respect to the fleet: a
 * standalone Vulkan context, no pack access. */
#include "fg_quant.h"
#include "fg_vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct shape {
    uint32_t input;
    uint32_t output;
    const char *name;
} shape;

/* The r8-eligible production dense shapes (dense_cooked_rows8_shape). */
static const shape shapes[] = {
    {2560u, 10240u, "gdn_qkv"},     /* attn_qkv / attn_gate style */
    {2560u, 12288u, "qsa_qgate"},   /* raw_query_gate */
    {2560u, 512u, "qsa_kv"},        /* raw_key / raw_value */
    {6144u, 2560u, "qsa_output"},   /* attention -> hidden */
    {2560u, 2560u, "ssm_out"},      /* hidden -> hidden */
    {2560u, 640u, "shexp_up"},      /* shared expert gate/up */
    {640u, 2560u, "shexp_down"},    /* shared expert down */
    {320u, 10240u, "hc_up"},        /* GR read up projection */
};
enum { ITERATIONS = 24 };

static double elapsed_ms(struct timespec begin, struct timespec end) {
    return (double)(end.tv_sec - begin.tv_sec) * 1e3 +
           (double)(end.tv_nsec - begin.tv_nsec) * 1e-6;
}

static void fill_source(uint8_t *source, uint32_t input, uint32_t output) {
    uint32_t blocks = input / FG_QK8_0;
    for (uint32_t row = 0; row < output; row++)
        for (uint32_t block = 0; block < blocks; block++) {
            uint8_t *value = source + ((uint64_t)row * blocks + block) * FG_Q8_0_BLOCK_BYTES;
            uint16_t delta = fg_f32_to_f16(0.001f + (float)((row + block) % 7u) * 0.0001f);
            memcpy(value, &delta, sizeof(delta));
            for (uint32_t i = 0; i < FG_QK8_0; i++)
                value[2u + i] = (uint8_t)(row * 17u + block * 29u + i * 11u);
        }
}

int main(void) {
    fg_error error = {0};
    fg_vk_context *context = NULL;
    fg_status status = fg_vk_open(&context, &error);
    if (status != FG_OK) {
        fprintf(stderr, "Vulkan unavailable: %s\n", error.message);
        return 77;
    }
    double serial_bytes = 0.0, batched_serial_bytes = 0.0;
    int failures = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint32_t input = shapes[s].input, output = shapes[s].output;
        uint32_t blocks = input / FG_QK8_0, source_row = blocks * FG_Q8_0_BLOCK_BYTES;
        uint64_t source_bytes = (uint64_t)output * source_row;
        uint64_t cooked_bytes = fg_q8_0_cooked_matrix_bytes(input, output), quant_bytes = (uint64_t)output * blocks * 32u;
        uint8_t *source = malloc((size_t)source_bytes), *cooked = malloc((size_t)cooked_bytes);
        float *input_values = malloc((uint64_t)input * 2u * 4u);
        float *serial_values = malloc((uint64_t)output * 2u * 4u);
        float *batched_values = malloc((uint64_t)output * 2u * 4u);
        if (!source || !cooked || !input_values || !serial_values || !batched_values) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        fill_source(source, input, output);
        for (uint32_t i = 0; i < input * 2u; i++)
            input_values[i] = sinf((float)(i + 3u) * 0.013f) + 0.1f * cosf((float)i * 0.007f);
        if (!fg_cook_q8_0_rows(source, cooked, cooked_bytes, input, output)) {
            fprintf(stderr, "cook failed for %s\n", shapes[s].name);
            return 1;
        }
        fg_vk_tensor *weight = NULL, *cooked_weight = NULL, *activation = NULL,
                     *serial_out = NULL, *batched_out = NULL;
        status = fg_vk_tensor_create(context, source_bytes, &weight, &error);
        if (status == FG_OK) status = fg_vk_tensor_create(context, cooked_bytes, &cooked_weight, &error);
        if (status == FG_OK) status = fg_vk_tensor_create(context, (uint64_t)input * 2u * 4u, &activation, &error);
        if (status == FG_OK) status = fg_vk_tensor_create(context, (uint64_t)output * 2u * 4u, &serial_out, &error);
        if (status == FG_OK) status = fg_vk_tensor_create(context, (uint64_t)output * 2u * 4u, &batched_out, &error);
        if (status == FG_OK) status = fg_vk_tensor_write(weight, 0, source, source_bytes, &error);
        if (status == FG_OK) status = fg_vk_tensor_write(cooked_weight, 0, cooked, cooked_bytes, &error);
        if (status == FG_OK) fg_vk_tensor_set_format(cooked_weight, FG_VK_TENSOR_FORMAT_Q8_0_COOKED);
        if (status == FG_OK) status = fg_vk_tensor_write(activation, 0, input_values, (uint64_t)input * 2u * 4u, &error);
        fg_vk_tensor *token1 = NULL, *serial_token1 = NULL;
        if (status == FG_OK) status = fg_vk_tensor_view(activation, (uint64_t)input * 4u, (uint64_t)input * 4u, &token1, &error);
        if (status == FG_OK) status = fg_vk_tensor_view(serial_out, (uint64_t)output * 4u, (uint64_t)output * 4u, &serial_token1, &error);

        if (status == FG_OK)
            status = fg_vk_dense_q8_0_f32(context, serial_out, cooked_weight, activation,
                                          input, output, 1u, 1.0f, &error);
        if (status == FG_OK)
            status = fg_vk_dense_q8_0_f32(context, serial_token1, cooked_weight, token1,
                                          input, output, 1u, 1.0f, &error);
        if (status == FG_OK)
            status = fg_vk_dense_q8_0_b2(context, batched_out, cooked_weight, activation,
                                         input, output, 1.0f, &error);
        if (status == FG_OK) status = fg_vk_tensor_read(serial_out, 0, serial_values, (uint64_t)output * 2u * 4u, &error);
        if (status == FG_OK) status = fg_vk_tensor_read(batched_out, 0, batched_values, (uint64_t)output * 2u * 4u, &error);
        int exact = status == FG_OK &&
                    memcmp(serial_values, batched_values, (size_t)output * 2u * 4u) == 0;
        if (!exact) failures++;

        /* Serial: two single-token r8 dispatches.  Batched: one b2 dispatch. */
        fg_vk_profile serial_profile = {0}, batched_profile = {0};
        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        if (status == FG_OK) status = fg_vk_profile_begin(context, &error);
        if (status == FG_OK) status = fg_vk_begin(context, &error);
        for (uint32_t i = 0; status == FG_OK && i < ITERATIONS; i++) {
            status = fg_vk_dense_q8_0_f32(context, serial_out, cooked_weight, activation, input, output, 1u, 1.0f, &error);
            if (status == FG_OK) status = fg_vk_dense_q8_0_f32(context, serial_token1, cooked_weight, token1, input, output, 1u, 1.0f, &error);
        }
        if (status == FG_OK) status = fg_vk_end(context, &error);
        if (status == FG_OK) status = fg_vk_profile_end(context, &serial_profile, &error);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double serial_wall = elapsed_ms(begin, end) / ITERATIONS;

        clock_gettime(CLOCK_MONOTONIC, &begin);
        if (status == FG_OK) status = fg_vk_profile_begin(context, &error);
        if (status == FG_OK) status = fg_vk_begin(context, &error);
        for (uint32_t i = 0; status == FG_OK && i < ITERATIONS; i++)
            status = fg_vk_dense_q8_0_b2(context, batched_out, cooked_weight, activation, input, output, 1.0f, &error);
        if (status == FG_OK) status = fg_vk_end(context, &error);
        if (status == FG_OK) status = fg_vk_profile_end(context, &batched_profile, &error);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double batched_wall = elapsed_ms(begin, end) / ITERATIONS;

        double serial_us = serial_profile.kernel_ms * 1000.0 / ITERATIONS;
        double batched_us = batched_profile.kernel_ms * 1000.0 / ITERATIONS;
        double factor = batched_us > 0.0 ? serial_us / batched_us : 0.0;
        double bytes = (double)quant_bytes;
        serial_bytes += bytes;
        if (factor > 0.0) batched_serial_bytes += bytes / factor;
        fprintf(stderr, "BATCH2MMV shape=%s in=%u out=%u iters=%u serial_pair_gpu_us=%.3f "
                        "batched_gpu_us=%.3f factor=%.4f serial_pair_wall_us=%.3f batched_wall_us=%.3f "
                        "bits_exact=%d\n",
                shapes[s].name, input, output, ITERATIONS, serial_us, batched_us, factor,
                serial_wall * 1000.0, batched_wall * 1000.0, exact);
        fg_vk_tensor_destroy(serial_token1); fg_vk_tensor_destroy(token1);
        fg_vk_tensor_destroy(batched_out);
        fg_vk_tensor_destroy(serial_out); fg_vk_tensor_destroy(activation);
        fg_vk_tensor_destroy(cooked_weight); fg_vk_tensor_destroy(weight);
        free(batched_values); free(serial_values); free(input_values);
        free(cooked); free(source);
        if (status != FG_OK) break;
    }
    if (status == FG_OK)
        fprintf(stderr, "BATCH2MMV_SUMMARY shapes=%zu byte_weighted_factor=%.4f bits_failures=%d\n",
                sizeof(shapes) / sizeof(shapes[0]),
                batched_serial_bytes > 0.0 ? serial_bytes / batched_serial_bytes : 0.0, failures);
    else
        fprintf(stderr, "batch-2 MMV prototype failed: %s\n", error.message);
    fg_vk_close(context);
    return status == FG_OK && failures == 0 ? 0 : 1;
}
