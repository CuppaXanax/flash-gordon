#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(local_size_x=64)in;
layout(set=0,binding=0)readonly buffer Weights{uint w[];};
layout(set=0,binding=0)readonly buffer Vectors{uvec4 w4[];};
layout(set=0,binding=1)readonly buffer Tiles{uint tiles[];};
layout(set=0,binding=2)readonly buffer Input{vec4 x[];};
layout(set=0,binding=3)readonly buffer Gates{float gates[];};
layout(set=0,binding=4)writeonly buffer Output{float y[];};
layout(push_constant)uniform Push{
    uint out_dim,in_dim,blocks,tile_bytes,expert_stride,slots,down_type;
}pc;
uint spread(uint b){return ((b*0x00204081u)&0x01010101u)<<4u;}
void main(){
    uint lid=gl_LocalInvocationID.x,lane=lid%FG_DOWN_LANES;
    uint row=gl_WorkGroupID.x*(64u/FG_DOWN_LANES)+lid/FG_DOWN_LANES;
    bool live=row<pc.out_dim;
    float acc=0.0;
    for(uint slot=0u;slot<pc.slots;slot++){
        uint expert=tiles[slot*9u],pair=tiles[slot*9u+1u];
        if(expert==0xffffffffu||pair==0xffffffffu)continue;
        float value=0.0;
        if(live){
            uint base=expert*pc.expert_stride+(row/16u)*pc.tile_bytes;
            for(uint first=lane;first<pc.blocks;first+=FG_DOWN_LANES*FG_DOWN_PREFETCH){
                uvec4 quant[FG_DOWN_PREFETCH];uint high[FG_DOWN_PREFETCH],meta[FG_DOWN_PREFETCH];
                [[unroll]]for(uint p=0u;p<FG_DOWN_PREFETCH;p++){
                    uint b=min(first+p*FG_DOWN_LANES,pc.blocks-1u);
                    uint at=base+b*384u+(row%16u)*4u;
                    meta[p]=w[at/4u];high[p]=w[at/4u+16u];
                    quant[p]=w4[(base+b*384u+128u+(row%16u)*16u)/16u];
                }
                [[unroll]]for(uint p=0u;p<FG_DOWN_PREFETCH;p++){
                    uint b=first+p*FG_DOWN_LANES;if(b>=pc.blocks)continue;
                    vec2 sm=unpackHalf2x16(meta[p]);
                    [[unroll]]for(uint j=0u;j<4u;j++){
                        uint lo=(quant[p][j]&0x0f0f0f0fu)|spread((high[p]>>(j*4u))&15u);
                        uint hi=((quant[p][j]>>4u)&0x0f0f0f0fu)|spread((high[p]>>(16u+j*4u))&15u);
                        vec4 a=fma(vec4(sm.x),vec4(unpack8(int(lo))),vec4(sm.y));
                        vec4 z=fma(vec4(sm.x),vec4(unpack8(int(hi))),vec4(sm.y));
                        value+=dot(a,x[pair*(pc.in_dim/4u)+b*8u+j])+
                               dot(z,x[pair*(pc.in_dim/4u)+b*8u+j+4u]);
                    }
                }
            }
        }
        acc=fma(gates[slot],value,acc);
    }
#if FG_DOWN_LANES == 2
    acc+=subgroupShuffleXor(acc,1u);
#endif
    if(live&&lane==0u)y[row]=acc;
}
