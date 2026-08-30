#ifndef FLASH_GORDON_QUANT_H
#define FLASH_GORDON_QUANT_H

#include <stdint.h>
#include <stdbool.h>

#define FG_QK8_0 32u
#define FG_QK4_0 32u
#define FG_QK8_K 256u
#define FG_Q8_0_BLOCK_BYTES 34u
#define FG_Q8_K_BLOCK_BYTES 296u
#define FG_Q8_0_COOK_ALIGNMENT 64u
#define FG_Q8_0_COOK_ROWS 16u
#define FG_Q5_1_BLOCK_BYTES 24u
#define FG_Q5_1_COOK_ROWS 16u
#define FG_K_QUANT_COOK_ROWS 8u

float fg_f16_to_f32(uint16_t value);
uint16_t fg_f32_to_f16(float value);
void fg_quantize_q8_0(const float *input,void *output,uint64_t values);
void fg_quantize_q4_0(const float *input,void *output,uint64_t values);
void fg_quantize_q8_k(const float *input,void *output,uint64_t values);
uint64_t fg_q8_0_cooked_tile_bytes(uint32_t input_width);
uint64_t fg_q8_0_cooked_matrix_bytes(uint32_t input_width,uint32_t rows);
bool fg_cook_q8_0_rows(const void *input,void *output,uint64_t output_bytes,
					   uint32_t input_width,uint32_t rows);
uint64_t fg_q5_1_cooked_tile_bytes(uint32_t input_width);
uint64_t fg_q5_1_cooked_matrix_bytes(uint32_t input_width,uint32_t rows);
bool fg_cook_q5_1_rows(const void *input,void *output,uint64_t output_bytes,
                       uint32_t input_width,uint32_t rows);
uint64_t fg_k_quant_cooked_tile_bytes(uint32_t input_width,uint32_t ggml_type);
uint64_t fg_k_quant_cooked_matrix_bytes(uint32_t input_width,uint32_t rows,uint32_t ggml_type);
bool fg_cook_k_quant_rows(const void *input,void *output,uint64_t output_bytes,
						  uint32_t input_width,uint32_t rows,uint32_t ggml_type);
void fg_dequantize_q8_0(const void *input,float *output,uint64_t values);
void fg_dequantize_q4_0(const void *input,float *output,uint64_t values);
void fg_dequantize_iq4_nl(const void *input,float *output,uint64_t values);
float fg_dot_q8_0(const void *lhs,const void *rhs,uint64_t values);
float fg_dot_q4_k_q8_k(const void *weights,const void *activation,uint64_t values);
float fg_dot_q5_k_q8_k(const void *weights,const void *activation,uint64_t values);

#endif
