float small_fma4(vec4 a,vec4 b,float accumulator){
    accumulator=fma(a.x,b.x,accumulator);
    accumulator=fma(a.y,b.y,accumulator);
    accumulator=fma(a.z,b.z,accumulator);
    return fma(a.w,b.w,accumulator);
}
