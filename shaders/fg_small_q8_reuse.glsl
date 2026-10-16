#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#include "fg_small_fma.glsl"
layout(local_size_x_id=0) in;
layout(set=0,binding=0) readonly buffer Weights{uint w[];};
layout(set=0,binding=1) readonly buffer Input{vec4 x[];};
layout(set=0,binding=2) writeonly buffer Output{float y[];};
layout(push_constant) uniform Push{
    uint out_dim,tokens,blocks,tile_bytes;
    float scale;
}pc;
#define Q8_COLUMN(T,A) { \
    vec4 input_value=x[T*pc.blocks*8u+block*8u+lane]; \
    A.x=small_fma4(lo,input_value,A.x); \
    A.y=small_fma4(hi,input_value,A.y); }
#define Q8_REDUCE(A) \
    for(uint offset=1u;offset<32u;offset*=2u)A+=subgroupShuffleXor(A,offset);
#define Q8_STORE(T,A) \
    if(row<pc.out_dim)y[T*pc.out_dim+row]=A.x*pc.scale; \
    if(row+1u<pc.out_dim)y[T*pc.out_dim+row+1u]=A.y*pc.scale;
void main(){
    uint lid=gl_LocalInvocationID.x,cohort=lid&31u,lane=cohort&7u;
    uint row=gl_WorkGroupID.x*(gl_WorkGroupSize.x/16u)+(lid/32u)*2u;
    if(row>=pc.out_dim)return;
    uint tile_base=(row>>4u)*pc.tile_bytes,tile_row=row&15u;
    uint quant_offset=(16u*pc.blocks*2u+63u)&~63u;
    vec2 acc0=vec2(0.0),acc1=vec2(0.0);
#if SMALL_N >= 3
    vec2 acc2=vec2(0.0);
#endif
#if SMALL_N >= 4
    vec2 acc3=vec2(0.0);
#endif
    for(uint block=cohort/8u;block<pc.blocks;block+=4u){
        vec2 delta=unpackHalf2x16(w[(tile_base>>2u)+block*8u+(tile_row>>1u)]);
        uint at=(tile_base+quant_offset+tile_row*pc.blocks*32u+block*32u)/4u+lane;
        vec4 lo=vec4(unpack8(int(w[at])))*delta.x;
        vec4 hi=vec4(unpack8(int(w[at+pc.blocks*8u])))*delta.y;
        Q8_COLUMN(0u,acc0) Q8_COLUMN(1u,acc1)
#if SMALL_N >= 3
        Q8_COLUMN(2u,acc2)
#endif
#if SMALL_N >= 4
        Q8_COLUMN(3u,acc3)
#endif
    }
    Q8_REDUCE(acc0) Q8_REDUCE(acc1)
#if SMALL_N >= 3
    Q8_REDUCE(acc2)
#endif
#if SMALL_N >= 4
    Q8_REDUCE(acc3)
#endif
    if(cohort==0u){
        Q8_STORE(0u,acc0) Q8_STORE(1u,acc1)
#if SMALL_N >= 3
        Q8_STORE(2u,acc2)
#endif
#if SMALL_N >= 4
        Q8_STORE(3u,acc3)
#endif
    }
}
