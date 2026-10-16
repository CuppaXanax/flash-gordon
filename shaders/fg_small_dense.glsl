#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require
#ifndef SMALL_ROWS
#define SMALL_ROWS 4
#endif
#if SMALL_N == 2
#define SMALL_INTERLEAVED 1
#define SMALL_COLS 2
#else
#define SMALL_PAIR_COUNT 2
#undef SMALL_N
#define SMALL_N 2
#endif
#define WG 64u
#define ROWS SMALL_ROWS
#define COMPACT 1
#include "fg_dense_soa_impl.glsl"
