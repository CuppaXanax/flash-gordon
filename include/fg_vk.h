#ifndef FLASH_GORDON_VK_H
#define FLASH_GORDON_VK_H

#include "fg.h"

typedef struct fg_vk_context fg_vk_context;
typedef struct fg_vk_tensor fg_vk_tensor;
typedef struct fg_vk_expert_graph fg_vk_expert_graph;

#define FG_VK_PROFILE_MAX_KERNELS 64u

typedef struct fg_vk_profile_kernel {
    const char *scope;
    const char *name;
    uint64_t invocations;
    double gpu_ms;
} fg_vk_profile_kernel;

typedef struct fg_vk_profile {
    uint32_t kernel_count;
    uint64_t submissions;
    uint64_t dispatches;
    double gpu_ms;
    double kernel_ms;
    fg_vk_profile_kernel kernels[FG_VK_PROFILE_MAX_KERNELS];
} fg_vk_profile;

typedef struct fg_vk_counters {
    uint64_t submissions;
    uint64_t dispatches;
} fg_vk_counters;

fg_status fg_vk_open(fg_vk_context **out,fg_error *err);
void fg_vk_close(fg_vk_context *context);
const char *fg_vk_device_name(const fg_vk_context *context);

fg_status fg_vk_profile_begin(fg_vk_context *context,fg_error *err);
fg_status fg_vk_profile_set_scope(fg_vk_context *context,const char *scope,fg_error *err);
fg_status fg_vk_profile_end(fg_vk_context *context,fg_vk_profile *profile,fg_error *err);
bool fg_vk_profile_active(const fg_vk_context *context);
void fg_vk_get_counters(const fg_vk_context *context,fg_vk_counters *counters);

fg_status fg_vk_begin(fg_vk_context *context,fg_error *err);
fg_status fg_vk_end(fg_vk_context *context,fg_error *err);
bool fg_vk_batch_active(const fg_vk_context *context);

fg_status fg_vk_expert_graph_create(fg_vk_context *context,fg_vk_expert_graph **out,
                                     fg_vk_tensor *activation,fg_vk_tensor *tiles,
                                     fg_vk_tensor *gates,fg_vk_tensor *gate,fg_vk_tensor *up,
                                     fg_vk_tensor *mid,fg_vk_tensor *down,fg_vk_tensor *reduced,
                                     const fg_vk_tensor *gate_weights,const fg_vk_tensor *up_weights,
                                     const fg_vk_tensor *down_weights,uint32_t gate_type,
                                     uint32_t up_type,uint32_t down_type,uint32_t hidden_width,
                                     uint32_t mid_width,uint32_t gate_expert_stride,
                                     uint32_t up_expert_stride,uint32_t down_expert_stride,
                                     uint32_t weight_experts,uint32_t slots,fg_error *err);
void fg_vk_expert_graph_destroy(fg_vk_expert_graph *graph);
fg_status fg_vk_expert_graph_execute(fg_vk_expert_graph *graph,fg_error *err);

fg_status fg_vk_tensor_create(fg_vk_context *context,uint64_t bytes,fg_vk_tensor **out,fg_error *err);
fg_status fg_vk_tensor_view(fg_vk_tensor *base,uint64_t offset,uint64_t bytes,fg_vk_tensor **out,fg_error *err);
void fg_vk_tensor_destroy(fg_vk_tensor *tensor);
uint64_t fg_vk_tensor_bytes(const fg_vk_tensor *tensor);
void *fg_vk_tensor_map(fg_vk_tensor *tensor);
fg_status fg_vk_tensor_write(fg_vk_tensor *tensor,uint64_t offset,const void *data,uint64_t bytes,fg_error *err);
fg_status fg_vk_tensor_read(const fg_vk_tensor *tensor,uint64_t offset,void *data,uint64_t bytes,fg_error *err);
fg_status fg_vk_quantize_q8_k(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *input,
                              uint32_t input_width,uint32_t tokens,fg_error *err);
fg_status fg_vk_quantize_q8_0(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *input,
                              uint32_t input_width,uint32_t tokens,fg_error *err);
fg_status fg_vk_quantize_q4_0(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *input,
                              uint32_t input_width,uint32_t tokens,fg_error *err);
fg_status fg_vk_dequantize_iq4_nl(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *input,
                                  uint32_t rows,uint32_t width,fg_error *err);
fg_status fg_vk_swiglu(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *gate,
                       const fg_vk_tensor *up,uint32_t values,fg_error *err);

fg_status fg_vk_dense_q8_0_f32(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                               const fg_vk_tensor *input,uint32_t input_width,uint32_t output_width,
                               uint32_t tokens,float scale,fg_error *err);
fg_status fg_vk_embedding_q8_0(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                               uint32_t token,uint32_t width,uint32_t rows,uint32_t copies,fg_error *err);
fg_status fg_vk_embedding_q8_0_batch(fg_vk_context *context,fg_vk_tensor *output,
                                     const fg_vk_tensor *weights,const fg_vk_tensor *token_ids,
                                     uint32_t token_count,uint32_t width,uint32_t rows,
                                     uint32_t copies,fg_error *err);
fg_status fg_vk_dense_f32(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                          const fg_vk_tensor *input,uint32_t input_width,uint32_t output_width,
                          uint32_t tokens,fg_error *err);
fg_status fg_vk_dense_bf16_f32(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                               const fg_vk_tensor *input,uint32_t input_width,uint32_t output_width,
                               uint32_t tokens,fg_error *err);
fg_status fg_vk_silu_scaled(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *input,
                            uint32_t values,float input_scale,fg_error *err);
fg_status fg_vk_group_rms_norm(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *input,
                               const fg_vk_tensor *weight,uint32_t width,uint32_t groups,uint32_t tokens,
                               float epsilon,fg_error *err);
fg_status fg_vk_gr_mix(fg_vk_context *context,fg_vk_tensor *mixed,fg_vk_tensor *injection,
                       const fg_vk_tensor *normalized,const fg_vk_tensor *up_logits,
                       const fg_vk_tensor *injection_logits,uint32_t hidden,uint32_t groups,
                       uint32_t tokens,fg_error *err);
fg_status fg_vk_hc_inject_partial(fg_vk_context *context,fg_vk_tensor *partials,
                                  const fg_vk_tensor *normalized,const fg_vk_tensor *weights,
                                  uint32_t hidden,uint32_t groups,uint32_t tokens,
                                  uint32_t pieces,fg_error *err);
fg_status fg_vk_gr_mix_partial(fg_vk_context *context,fg_vk_tensor *mixed,fg_vk_tensor *injection,
                               const fg_vk_tensor *normalized,const fg_vk_tensor *up_logits,
                               const fg_vk_tensor *partials,uint32_t hidden,uint32_t groups,
                               uint32_t tokens,uint32_t pieces,fg_error *err);
fg_status fg_vk_hc_finalize(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *normalized,
                            const fg_vk_tensor *up,uint32_t hidden,uint32_t groups,uint32_t tokens,
                            fg_error *err);
fg_status fg_vk_gr_write(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *hyper_input,
                         const fg_vk_tensor *block_output,const fg_vk_tensor *injection,uint32_t hidden,
                         uint32_t groups,uint32_t tokens,fg_error *err);
fg_status fg_vk_ple_gate(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *key,
                         const fg_vk_tensor *query,const fg_vk_tensor *value,fg_error *err);
fg_status fg_vk_ple_gate_prefill(fg_vk_context *context,fg_vk_tensor *output,
                                 const fg_vk_tensor *key,const fg_vk_tensor *query,
                                 const fg_vk_tensor *value,uint32_t tokens,fg_error *err);
fg_status fg_vk_ple_conv_decode(fg_vk_context *context,fg_vk_tensor *output,fg_vk_tensor *state,
                                const fg_vk_tensor *gated,const fg_vk_tensor *normalized,
                                const fg_vk_tensor *weight,fg_error *err);
fg_status fg_vk_ple_conv_prefill(fg_vk_context *context,fg_vk_tensor *output,fg_vk_tensor *state,
                                 const fg_vk_tensor *gated,const fg_vk_tensor *normalized,
                                 const fg_vk_tensor *weight,uint32_t tokens,fg_error *err);
fg_status fg_vk_add_f32(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *left,
                        const fg_vk_tensor *right,uint32_t values,fg_error *err);
fg_status fg_vk_gdn_conv_decode(fg_vk_context *context,fg_vk_tensor *output,fg_vk_tensor *state,
                                const fg_vk_tensor *projection,const fg_vk_tensor *weight,
                                uint32_t channels,fg_error *err);
fg_status fg_vk_gdn_conv_prefill(fg_vk_context *context,fg_vk_tensor *output,fg_vk_tensor *state,
                                 const fg_vk_tensor *projection,const fg_vk_tensor *weight,
                                 uint32_t channels,uint32_t tokens,fg_error *err);
fg_status fg_vk_gdn_project_decode(fg_vk_context *context,fg_vk_tensor *qkv,
                                   fg_vk_tensor *z,fg_vk_tensor *alpha,fg_vk_tensor *beta,
                                   const fg_vk_tensor *qkv_weight,const fg_vk_tensor *z_weight,
                                   const fg_vk_tensor *alpha_weight,const fg_vk_tensor *beta_weight,
                                   const fg_vk_tensor *hidden,fg_error *err);
fg_status fg_vk_gdn_recurrent_decode(fg_vk_context *context,fg_vk_tensor *output,fg_vk_tensor *state,
                                     const fg_vk_tensor *qkv,const fg_vk_tensor *z,const fg_vk_tensor *alpha,
                                     const fg_vk_tensor *beta,const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,
                                     const fg_vk_tensor *norm_weight,uint32_t value_heads,uint32_t key_heads,
                                     uint32_t head_dim,float epsilon,fg_error *err);
fg_status fg_vk_gdn_recurrent_algebraic(fg_vk_context *context,fg_vk_tensor *output,fg_vk_tensor *state,
                                        const fg_vk_tensor *qkv,const fg_vk_tensor *z,
                                        const fg_vk_tensor *alpha,const fg_vk_tensor *beta,
                                        const fg_vk_tensor *a_log,const fg_vk_tensor *dt_bias,
                                        const fg_vk_tensor *norm_weight,uint32_t value_heads,
                                        uint32_t key_heads,uint32_t head_dim,float epsilon,fg_error *err);
fg_status fg_vk_gdn_recurrent_prefill(fg_vk_context *context,fg_vk_tensor *output,
                                      fg_vk_tensor *state,const fg_vk_tensor *qkv,
                                      const fg_vk_tensor *z,const fg_vk_tensor *alpha,
                                      const fg_vk_tensor *beta,const fg_vk_tensor *a_log,
                                      const fg_vk_tensor *dt_bias,const fg_vk_tensor *norm_weight,
                                      uint32_t value_heads,uint32_t key_heads,uint32_t head_dim,
                                      uint32_t tokens,float epsilon,fg_error *err);
fg_status fg_vk_qsa_index_score(fg_vk_context *context,fg_vk_tensor *scores,fg_vk_tensor *block_ids,
                                const fg_vk_tensor *query,const fg_vk_tensor *index_keys_q8,
                                const fg_vk_tensor *key_norm,const fg_vk_tensor *positions,
                                uint32_t tokens,fg_error *err);
fg_status fg_vk_qsa_prepare(fg_vk_context *context,fg_vk_tensor *query,fg_vk_tensor *gate,
                            fg_vk_tensor *key,const fg_vk_tensor *raw_query_gate,
                            const fg_vk_tensor *raw_key,const fg_vk_tensor *query_norm,
                            const fg_vk_tensor *key_norm,const fg_vk_tensor *position,
                            fg_error *err);
fg_status fg_vk_qsa_prepare_prefill(fg_vk_context *context,fg_vk_tensor *query,
                                    fg_vk_tensor *gate,fg_vk_tensor *key,
                                    const fg_vk_tensor *raw_query_gate,
                                    const fg_vk_tensor *raw_key,const fg_vk_tensor *query_norm,
                                    const fg_vk_tensor *key_norm,const fg_vk_tensor *positions,
                                    uint32_t tokens,fg_error *err);
fg_status fg_vk_qsa_index_prepare(fg_vk_context *context,fg_vk_tensor *query,const fg_vk_tensor *raw_query,
                                  const fg_vk_tensor *query_norm,const fg_vk_tensor *position,
                                  fg_error *err);
fg_status fg_vk_qsa_index_prepare_prefill(fg_vk_context *context,fg_vk_tensor *query,
                                          const fg_vk_tensor *raw_query,
                                          const fg_vk_tensor *query_norm,
                                          const fg_vk_tensor *positions,uint32_t tokens,
                                          fg_error *err);
fg_status fg_vk_qsa_record_commit(fg_vk_context *context,fg_vk_tensor *records,
                                  fg_vk_tensor *index_history,const fg_vk_tensor *key_q8,
                                  const fg_vk_tensor *value_q8,const fg_vk_tensor *index_key_q8,
                                  const fg_vk_tensor *position,uint32_t layer_slot,
                                  uint32_t token,uint32_t capacity,fg_error *err);
fg_status fg_vk_qsa_record_gather(fg_vk_context *context,fg_vk_tensor *output,
                                  const fg_vk_tensor *records,const fg_vk_tensor *block_ids,
                                  uint32_t layer_slot,uint32_t capacity,uint32_t block_count,
                                  uint32_t tail_start,uint32_t tail_count,fg_error *err);
fg_status fg_vk_qsa_attention(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *records,
                              const fg_vk_tensor *query,const fg_vk_tensor *gate,
                              uint32_t selected_count,fg_error *err);
fg_status fg_vk_topk_reduce(fg_vk_context *context,fg_vk_tensor *output_scores,fg_vk_tensor *output_ids,
                            const fg_vk_tensor *input_scores,const fg_vk_tensor *input_ids,
                            uint32_t count,uint32_t *output_count,fg_error *err);
fg_status fg_vk_argmax_reduce(fg_vk_context *context,fg_vk_tensor *output_scores,
                                                            fg_vk_tensor *output_ids,const fg_vk_tensor *input_scores,
                                                            const fg_vk_tensor *input_ids,uint32_t count,
                                                            uint32_t *output_count,fg_error *err);
fg_status fg_vk_moe_q5_1_down(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                              const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,
                              uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,
                              bool packed_weights,uint32_t tile_count,fg_error *err);
fg_status fg_vk_moe_q8_0_down(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                              const fg_vk_tensor *tiles,const fg_vk_tensor *input,uint32_t output_width,
                              uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,
                              bool packed_weights,uint32_t tile_count,fg_error *err);
fg_status fg_vk_moe_reduce(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *down,
                           const fg_vk_tensor *gates,const fg_vk_tensor *tiles,uint32_t output_width,
                           uint32_t selected_count,uint32_t slot_count,fg_error *err);
fg_status fg_vk_moe_kquant(fg_vk_context *context,fg_vk_tensor *output,const fg_vk_tensor *weights,
                           const fg_vk_tensor *activation_q8k,const fg_vk_tensor *tiles,uint32_t ggml_type,
                           uint32_t output_width,uint32_t input_width,uint32_t expert_stride,uint32_t used_experts,
                           uint32_t routed_pairs,bool packed_weights,uint32_t tile_count,fg_error *err);

/* GPU timestamp-profiled kernel benchmark. Reports per-shape:
    A. raw GPU kernel GB/s (no inter-dispatch barriers)
    B. kernel with current barrier policy
    C. complete standalone dispatch (full Vulkan overhead) */
fg_status fg_vk_bench_dense_q8(fg_vk_context *context,fg_error *err);
fg_status fg_vk_bench_decompose(fg_vk_context *context,fg_error *err);
fg_status fg_vk_bench_stream_abc(fg_vk_context *context,fg_error *err);

#endif
