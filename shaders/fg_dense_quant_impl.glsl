layout(local_size_x=128) in;
layout(set=0,binding=0) readonly buffer Weights { uint w[]; };
layout(set=0,binding=1) readonly buffer Input { vec4 x4[]; };
layout(set=0,binding=2) writeonly buffer Output { float y[]; };
layout(push_constant) uniform Push {
    uint input_width, output_width, tokens, type, row_bytes;
    float scale;
} pc;
shared float sums[128];
int grid_value(uint q) {
    uint word=q<8u?(q<4u?0xbfad9881u:0xf6eaddcfu):(q<12u?0x26190d01u:0x71594535u);
    return int(word<<(24u-(q&3u)*8u))>>24;
}
uint u8(uint p) { return (w[p>>2u]>>((p&3u)*8u))&255u; }
uint u16(uint p) { return u8(p)|(u8(p+1u)<<8u); }
float f16(uint p) { return unpackHalf2x16(u16(p)).x; }
float weight(uint row, uint col) {
    uint p=row*pc.row_bytes;
    if(TYPE==20u) {
        p+=(col>>5u)*18u;
        uint j=col&31u, q=u8(p+2u+(j&15u));
        q=(q>>((j>>4u)*4u))&15u;
        return f16(p)*float(grid_value(q));
    }
    uint i=col&255u;
    p+=(col>>8u)*(TYPE==12u?144u:TYPE==13u?176u:210u);
    if(TYPE==14u) {
        uint h=i>>7u, j=i&127u, quadrant=j>>5u, l=j&31u;
        uint lo=u8(p+h*64u+(quadrant&1u)*32u+l);
        uint hi=u8(p+128u+h*32u+l);
        int q=int(((lo>>((quadrant>>1u)*4u))&15u)|(((hi>>(quadrant*2u))&3u)<<4u))-32;
        int s=int(u8(p+192u+(i>>4u))); if(s>=128) s-=256;
        return f16(p+208u)*float(s*q);
    }
    uint g=i>>5u, j=i&31u, sc, mn;
    if(g<4u) { sc=u8(p+4u+g)&63u; mn=u8(p+8u+g)&63u; }
    else {
        uint a=u8(p+8u+g);
        sc=(a&15u)|((u8(p+g)>>6u)<<4u);
        mn=(a>>4u)|((u8(p+4u+g)>>6u)<<4u);
    }
    uint q=u8(p+(TYPE==12u?16u:48u)+(g>>1u)*32u+j);
    q=(q>>((g&1u)*4u))&15u;
    if(TYPE==13u) q|=((u8(p+16u+j)>>g)&1u)<<4u;
    return f16(p)*float(sc*q)-f16(p+2u)*float(mn);
}
#if TYPE == 12 || TYPE == 13
void main() {
    uint lid=gl_LocalInvocationID.x,lane=lid&31u,cohort=lid>>5u;
    uint first=gl_WorkGroupID.x*4u,token=gl_WorkGroupID.y;
    vec4 acc=vec4(0.0);
    const uint bs=TYPE==12u?144u:176u;
    for(uint b=cohort;b<pc.input_width/256u;b+=4u) {
        uint g=(lane>>3u)*2u;
        uint col=token*(pc.input_width/4u)+b*64u+g*8u+(lane&7u);
        vec4 a=x4[col],bv=x4[col+8u];
        vec4 subtotal=vec4(0.0),minimum=vec4(0.0);
        [[unroll]] for(uint r=0u;r<4u;r++) {
            uint row=min(first+r,pc.output_width-1u),p=row*pc.row_bytes+b*bs;
            uint scale_base=p+4u,qs_base=p+(TYPE==12u?16u:48u),qh_base=p+16u;
            if((pc.type&256u)!=0u) {
                uint tile=(row/8u)*(pc.input_width/256u)*8u*bs+b*8u*bs;
                p=tile+(row&7u)*4u;
                scale_base=tile+32u+(row&7u)*12u;
                qs_base=tile+8u*(TYPE==12u?16u:48u)+(row&7u)*128u;
                qh_base=tile+128u+(row&7u)*32u;
            }
            uint sc0,mn0,sc1,mn1;
            if(g<4u) {
                sc0=u8(scale_base+g)&63u;mn0=u8(scale_base+4u+g)&63u;
                sc1=u8(scale_base+1u+g)&63u;mn1=u8(scale_base+5u+g)&63u;
            } else {
                uint a0=u8(scale_base+4u+g),a1=u8(scale_base+5u+g);
                sc0=(a0&15u)|((u8(scale_base+g-4u)>>6u)<<4u);
                mn0=(a0>>4u)|((u8(scale_base+g)>>6u)<<4u);
                sc1=(a1&15u)|((u8(scale_base+g-3u)>>6u)<<4u);
                mn1=(a1>>4u)|((u8(scale_base+g+1u)>>6u)<<4u);
            }
            uint qs=w[qs_base/4u+lane];
            uvec4 packed=(uvec4(qs)>>uvec4(0u,8u,16u,24u))&255u;
            uvec4 q0=packed&15u,q1=packed>>4u;
            if(TYPE==13u) {
                uint qh=w[qh_base/4u+(lane&7u)];
                uvec4 hi=(uvec4(qh)>>uvec4(0u,8u,16u,24u))&255u;
                q0|=((hi>>g)&1u)<<4u;q1|=((hi>>(g+1u))&1u)<<4u;
            }
            vec2 delta=unpackHalf2x16(w[p/4u]);
            subtotal[r]=delta.x*(float(sc0)*dot(vec4(q0),a)+float(sc1)*dot(vec4(q1),bv));
            minimum[r]=delta.y*(float(mn0)*dot(vec4(1.0),a)+float(mn1)*dot(vec4(1.0),bv));
        }
        acc+=subtotal-minimum;
    }
    vec4 reduced=vec4(subgroupAdd(acc.x),subgroupAdd(acc.y),
                      subgroupAdd(acc.z),subgroupAdd(acc.w));
    if(subgroupElect())for(uint r=0u;r<4u;r++)sums[gl_SubgroupID*4u+r]=reduced[r];
    barrier();
    if(lid<4u&&first+lid<pc.output_width) {
        float total=0.0;
        for(uint i=0u;i<gl_NumSubgroups;i++)total+=sums[i*4u+lid];
        y[token*pc.output_width+first+lid]=total*pc.scale;
    }
}
#elif TYPE == 20
void main() {
    uint lid=gl_LocalInvocationID.x,lane=lid&3u,cohort=lid>>2u;
    uint first=gl_WorkGroupID.x*4u,token=gl_WorkGroupID.y;
    vec4 acc=vec4(0.0);
    for(uint b=cohort;b<pc.input_width/32u;b+=32u) {
        uint col=token*(pc.input_width/4u)+b*8u+lane;
        vec4 a=x4[col],bv=x4[col+4u];
        [[unroll]] for(uint r=0u;r<4u;r++) {
            uint row=min(first+r,pc.output_width-1u),p=row*pc.row_bytes+b*18u;
            uint off=p+2u+lane*4u,shift=(off&3u)*8u;
            uint qs=w[off>>2u];
            if(shift!=0u)qs=(qs>>shift)|(w[(off>>2u)+1u]<<(32u-shift));
            uvec4 packed=(uvec4(qs)>>uvec4(0u,8u,16u,24u))&255u;
            uvec4 q0=packed&15u,q1=packed>>4u;
            vec4 lo=vec4(grid_value(q0.x),grid_value(q0.y),grid_value(q0.z),grid_value(q0.w));
            vec4 hi=vec4(grid_value(q1.x),grid_value(q1.y),grid_value(q1.z),grid_value(q1.w));
            acc[r]=fma(f16(p),dot(lo,a)+dot(hi,bv),acc[r]);
        }
    }
    vec4 reduced=vec4(subgroupAdd(acc.x),subgroupAdd(acc.y),
                      subgroupAdd(acc.z),subgroupAdd(acc.w));
    if(subgroupElect())for(uint r=0u;r<4u;r++)sums[gl_SubgroupID*4u+r]=reduced[r];
    barrier();
    if(lid<4u&&first+lid<pc.output_width) {
        float total=0.0;
        for(uint i=0u;i<gl_NumSubgroups;i++)total+=sums[i*4u+lid];
        y[token*pc.output_width+first+lid]=total*pc.scale;
    }
}
#else
void main() {
    uint lid=gl_LocalInvocationID.x,lane=lid&31u;
    uint row=gl_WorkGroupID.x*4u+(lid>>5u),token=gl_WorkGroupID.y;
    float sum=0.0;
    if(row<pc.output_width) {
        if(TYPE==14u) {
            for(uint b=0u;b<pc.input_width/256u;b++) {
                uint p=row*pc.row_bytes+b*210u;
                float d=f16(p+208u);
                for(uint h=0u;h<2u;h++) {
                    uint lo0=u8(p+h*64u+lane),lo1=u8(p+h*64u+lane+32u);
                    uint hi=u8(p+128u+h*32u+lane);
                    for(uint z=0u;z<4u;z++) {
                        uint lo=(z&1u)==0u?lo0:lo1;
                        int q=int(((lo>>((z>>1u)*4u))&15u)|(((hi>>(z*2u))&3u)<<4u))-32;
                        uint i=h*128u+z*32u+lane;
                        int sc=int(u8(p+192u+(i>>4u)));if(sc>=128)sc-=256;
                        uint col=b*256u+i;
                        sum=fma(d*float(sc*q),x4[token*(pc.input_width/4u)+(col>>2u)][col&3u],sum);
                    }
                }
            }
        } else {
            for(uint col=lane;col<pc.input_width;col+=32u)
                sum=fma(weight(row,col),x4[token*(pc.input_width/4u)+(col>>2u)][col&3u],sum);
        }
    }
    sums[lid]=sum;barrier();
    if(lane==0u&&row<pc.output_width) {
        float total=0.0;
        for(uint i=0u;i<32u;i++) total+=sums[lid+i];
        y[token*pc.output_width+row]=total*pc.scale;
    }
}
#endif
