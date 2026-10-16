#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=64) in;
#if defined(BF16_ROUTER) || defined(BF16_INDEX)
layout(set=0,binding=0) readonly buffer Weights { uint w[]; };
float weight(uint i){return uintBitsToFloat(((w[i>>1u]>>((i&1u)*16u))&65535u)<<16u);}
#else
layout(set=0,binding=0) readonly buffer Weights { float w[]; };
float weight(uint i){return w[i];}
#endif
layout(set=0,binding=1) readonly buffer Input { float x[]; };
layout(set=0,binding=2) writeonly buffer Output { float y[]; };
layout(push_constant) uniform Push { uint input_width,output_width,tokens; } pc;
shared vec4 partial[64];
float column_sum(vec4 acc){
#ifdef BF16_INDEX
    return acc.x;
#else
    return (acc.x+acc.y)+(acc.z+acc.w);
#endif
}
void store_columns(uint row,uint lane,vec4 values){
#ifdef BF16_INDEX
    partial[lane]=values;
    barrier();
    for(uint stride=32u;stride>0u;stride>>=1u){
        if(lane<stride)partial[lane]+=partial[lane+stride];
        barrier();
    }
#else
    vec4 sum=subgroupAdd(values);
    if(subgroupElect())partial[gl_SubgroupID]=sum;
    barrier();
    if(lane==0u){
        vec4 total=vec4(0.0);
        for(uint s=0u;s<gl_NumSubgroups;s++)total+=partial[s];
        partial[0]=total;
    }
#endif
    if(lane==0u){
        vec4 total=partial[0];
        y[row]=total.x;y[pc.output_width+row]=total.y;
#if SMALL_N >= 3
        y[2u*pc.output_width+row]=total.z;
#endif
#if SMALL_N >= 4
        y[3u*pc.output_width+row]=total.w;
#endif
    }
}
#define FG_PLAIN_FMA(T,A) { \
    uint tb=T*pc.input_width+i; \
    A.x=fma(a.x,x[tb],A.x);A.y=fma(a.y,x[tb+64u],A.y); \
    A.z=fma(a.z,x[tb+128u],A.z);A.w=fma(a.w,x[tb+192u],A.w); }
#define FG_PLAIN_TAIL(T,A) A.x=fma(a,x[T*pc.input_width+i],A.x);
void main(){
    uint lane=gl_LocalInvocationID.x,row=gl_WorkGroupID.x,rb=row*pc.input_width;
    vec4 acc0=vec4(0.0),acc1=vec4(0.0);
#if SMALL_N >= 3
    vec4 acc2=vec4(0.0);
#endif
#if SMALL_N >= 4
    vec4 acc3=vec4(0.0);
#endif
    uint i=lane;
#ifndef BF16_INDEX
    for(;i+192u<pc.input_width;i+=256u){
        vec4 a=vec4(weight(rb+i),weight(rb+i+64u),weight(rb+i+128u),weight(rb+i+192u));
        FG_PLAIN_FMA(0u,acc0) FG_PLAIN_FMA(1u,acc1)
#if SMALL_N >= 3
        FG_PLAIN_FMA(2u,acc2)
#endif
#if SMALL_N >= 4
        FG_PLAIN_FMA(3u,acc3)
#endif
    }
#endif
    for(;i<pc.input_width;i+=64u){
        float a=weight(rb+i);
        FG_PLAIN_TAIL(0u,acc0) FG_PLAIN_TAIL(1u,acc1)
#if SMALL_N >= 3
        FG_PLAIN_TAIL(2u,acc2)
#endif
#if SMALL_N >= 4
        FG_PLAIN_TAIL(3u,acc3)
#endif
    }
    vec4 values=vec4(column_sum(acc0),column_sum(acc1),0.0,0.0);
#if SMALL_N >= 3
    values.z=column_sum(acc2);
#endif
#if SMALL_N >= 4
    values.w=column_sum(acc3);
#endif
    store_columns(row,lane,values);
}
