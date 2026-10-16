#include "fg_small_fma.glsl"
#ifdef FG_SMALL_META_PREFETCH
uvec4 prefetched_dm0,prefetched_dm1,prefetched_codes,prefetched_quant;
void reuse_prefetch(uint base,uint groups,uint group){
    if((SMALL_PREFETCH&1u)!=0u){
        uint offset=(base+(group/8u)*8u)/4u;
        prefetched_dm0=W4(offset);prefetched_dm1=W4(offset+1u);
        prefetched_codes=W4((base+groups+group*4u)/4u);
    }
}
#endif
vec2 reuse_scale(uint base,uint groups,uint row,uint group){
    if(TYPE==20u){
        uint packed=W(base+group*4u+(row>>1u));
        vec2 d=unpackHalf2x16(packed);
        return vec2((row&1u)==0u?d.x:d.y,0.0);
    }
    if(COMPACT!=0u){
#ifdef FG_SMALL_META_PREFETCH
        if((SMALL_PREFETCH&1u)!=0u){
            uint packed=row<4u?prefetched_dm0[row]:prefetched_dm1[row-4u];
            vec2 dm=unpackHalf2x16(packed);
            uint codes=prefetched_codes[row>>1u]>>((row&1u)*16u);
            return vec2(dm.x*float(codes&255u),dm.y*float((codes>>8u)&255u));
        }
#endif
        vec2 dm=unpackHalf2x16(W(base+(group/8u)*8u+row));
        uint codes=W(base+groups+group*4u+(row>>1u))>>((row&1u)*16u);
        return vec2(dm.x*float(codes&255u),dm.y*float((codes>>8u)&255u));
    }
    return vec2(uintBitsToFloat(W(base+group*16u+row)),
        uintBitsToFloat(W(base+group*16u+row+8u)));
}
void reuse_weight(uint base,uint groups,uint row,uint group,uint lane,out vec4 lo,out vec4 hi){
    uint off=base+groups*(TYPE==20u?4u:COMPACT!=0u?5u:16u);
    if(TYPE==13u){
        off+=(row*groups+group)*5u+lane;
        uint p=W(off),next=W(off+1u),shift=lane*8u;
        uint a=(p>>shift)|(shift==0u?0u:next<<(32u-shift)),b=next>>shift;
        lo=vec4(bitfieldExtract(a,0,5),bitfieldExtract(a,5,5),
            bitfieldExtract(a,10,5),bitfieldExtract(a,15,5));
        hi=vec4(bitfieldExtract(a,20,5),bitfieldExtract(a,25,5),
            ((a>>30u)|(b<<2u))&31u,(b>>3u)&31u);
    }else{
        uint p;
#ifdef FG_SMALL_META_PREFETCH
        if((SMALL_PREFETCH&2u)!=0u)p=prefetched_quant[row&3u];
        else
#endif
        p=W(off+(row*groups+group)*4u+lane);
        lo=quants(p&0x0f0f0f0fu);hi=quants((p>>4u)&0x0f0f0f0fu);
    }
    vec2 scale=reuse_scale(base,groups,row,group);
    lo=fma(vec4(scale.x),lo,vec4(-scale.y));
    hi=fma(vec4(scale.x),hi,vec4(-scale.y));
}
