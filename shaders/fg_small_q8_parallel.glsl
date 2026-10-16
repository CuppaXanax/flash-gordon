#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=64*SMALL_N,local_size_x_id=0) in;
layout(set=0,binding=0) restrict readonly buffer Weights { uint w[]; };
layout(set=0,binding=1) restrict readonly buffer Input { vec4 x[]; };
layout(set=0,binding=2) restrict writeonly buffer Output { float y[]; };
layout(push_constant) uniform Push {
    uint out_dim,tokens,blocks,tile_bytes;
    float scale;
} pc;
shared vec4 sums[SMALL_N*SMALL_N*8];
float product(uint offset,vec4 input_value){
    uint packed=w[offset>>2u];
    vec4 quant=vec4(unpack8(int(packed)));
    return dot(quant,input_value);
}
void main(){
#if SMALL_N > 2
    uint lid=gl_LocalInvocationID.x,token=2u*(lid%2u),local_id=lid/2u;
#else
    uint lid=gl_LocalInvocationID.x,token=lid%SMALL_N,local_id=lid/SMALL_N;
#endif
    uint cohort=local_id>>3u,lane=local_id&7u;
    uint first=gl_WorkGroupID.x*4u,tile_row=first&15u;
    uint tile_base=(first>>4u)*pc.tile_bytes;
    uint quant_offset=(16u*pc.blocks*2u+63u)&~63u;
    vec4 accumulator=vec4(0.0);
#if SMALL_N > 2
    vec4 second=vec4(0.0);
#endif
    for(uint group=0u;group<(pc.blocks+7u)/8u;group++){
        uint block=group*8u+cohort,safe_block=min(block,pc.blocks-1u);
        bool live=block<pc.blocks;
        uint scale_word=(tile_base>>2u)+safe_block*8u+(tile_row>>1u);
        vec4 delta=live?vec4(unpackHalf2x16(w[scale_word]),
            unpackHalf2x16(w[scale_word+1u])):vec4(0.0);
        uint off=tile_base+quant_offset+tile_row*pc.blocks*32u+safe_block*32u+lane*4u;
        vec4 input_value=live?x[token*pc.blocks*8u+safe_block*8u+lane]:vec4(0.0);
#if SMALL_N > 2
        vec4 other=live?x[min(token+1u,pc.tokens-1u)*pc.blocks*8u+safe_block*8u+lane]:vec4(0.0);
        vec4 q0=vec4(unpack8(int(w[off>>2u])));
        vec4 q1=vec4(unpack8(int(w[(off+pc.blocks*32u)>>2u])));
        vec4 q2=vec4(unpack8(int(w[(off+pc.blocks*64u)>>2u])));
        vec4 q3=vec4(unpack8(int(w[(off+pc.blocks*96u)>>2u])));
        vec4 products=vec4(dot(q0,input_value),dot(q1,input_value),
            dot(q2,input_value),dot(q3,input_value));
        vec4 other_products=vec4(dot(q0,other),dot(q1,other),dot(q2,other),dot(q3,other));
        second=fma(delta,other_products,second);
#else
        vec4 products=vec4(product(off,input_value),product(off+pc.blocks*32u,input_value),
            product(off+pc.blocks*64u,input_value),product(off+pc.blocks*96u,input_value));
#endif
        accumulator=fma(delta,products,accumulator);
    }
#if SMALL_N > 2
    for(uint offset=2u;offset<gl_SubgroupSize;offset*=2u){
        accumulator+=subgroupShuffleXor(accumulator,offset);
        second+=subgroupShuffleXor(second,offset);
    }
    uint subgroups=64u/gl_SubgroupSize;
    uint pair=token/2u;
    if(gl_SubgroupInvocationID<2u){
        sums[gl_SubgroupID*2u+pair]=accumulator;
        sums[(gl_NumSubgroups+gl_SubgroupID)*2u+pair]=second;
    }
    barrier();
    if(local_id==0u)for(uint t=0u;t<2u;t++){
        vec4 total=vec4(0.0);
        for(uint s=0u;s<subgroups;s++){
            uint p=(t*gl_NumSubgroups+s*2u)*2u+pair;
            total+=sums[p]+sums[p+2u];
        }
        for(uint r=0u;r<4u;r++)if(first+r<pc.out_dim&&token+t<pc.tokens)
            y[(token+t)*pc.out_dim+first+r]=total[r]*pc.scale;
    }
#else
    for(uint offset=SMALL_N;offset<gl_SubgroupSize;offset*=2u)
        accumulator+=subgroupShuffleXor(accumulator,offset);
    uint subgroups=64u/gl_SubgroupSize;
    if(gl_SubgroupInvocationID<SMALL_N)sums[gl_SubgroupID*SMALL_N+token]=accumulator;
    barrier();
    if(local_id==0u){
        vec4 total=vec4(0.0);
        for(uint s=0u;s<subgroups;s++){
            uint p=s*SMALL_N*SMALL_N+token;
            vec4 v=sums[p]+sums[p+SMALL_N];
#if SMALL_N == 4
            v+=sums[p+2u*SMALL_N]+sums[p+3u*SMALL_N];
#endif
            total+=v;
        }
        for(uint r=0u;r<4u;r++)if(first+r<pc.out_dim)
            y[token*pc.out_dim+first+r]=total[r]*pc.scale;
    }
#endif
}
