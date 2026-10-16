#ifdef FG_SMALL_FMA
#include "fg_small_fma.glsl"
#endif
uint spread4(uint bits){return ((bits*0x00204081u)&0x01010101u)<<4u;}
vec4 decode(uint q,uint bits,vec2 scale){
    return fma(vec4(scale.x),vec4(unpack8(int(q|spread4(bits&15u)))),vec4(scale.y));
}
void apply_pair(uint pair,uint block,uint seg,
    vec4 w0,vec4 w1,vec4 w2,vec4 w3,
    vec4 p0,vec4 p1,vec4 p2,vec4 p3,inout vec2 accumulator){
    uint i=pair*(pc.in_dim/4u)+block*8u+seg*2u;
    vec4 i0=x[i],i4=x[i+4u],i1=x[i+1u],i5=x[i+5u];
#ifdef FG_SMALL_FMA
    accumulator.x=small_fma4(w3,i5,small_fma4(w2,i1,
        small_fma4(w1,i4,small_fma4(w0,i0,accumulator.x))));
    accumulator.y=small_fma4(p3,i5,small_fma4(p2,i1,
        small_fma4(p1,i4,small_fma4(p0,i0,accumulator.y))));
#else
    accumulator.x+=dot(w0,i0)+dot(w1,i4);
    accumulator.x+=dot(w2,i1)+dot(w3,i5);
    accumulator.y+=dot(p0,i0)+dot(p1,i4);
    accumulator.y+=dot(p2,i1)+dot(p3,i5);
#endif
}
