#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require
#ifndef SMALL_ROWS
#define SMALL_ROWS 4
#endif
#define WG 64u
#define ROWS SMALL_ROWS
#define COMPACT 1
#include "fg_dense_soa_impl.glsl"
