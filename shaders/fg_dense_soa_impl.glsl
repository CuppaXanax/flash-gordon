#ifndef WG
#define WG 128u
#endif
#ifndef ROWS
#define ROWS 8u
#endif
layout(local_size_x=WG) in;
layout(set=0,binding=0) readonly buffer Weights { uint w[]; };
layout(set=0,binding=0) readonly buffer WideWeights { uvec4 w4[]; };
layout(set=0,binding=1) readonly buffer Input { vec4 x[]; };
layout(set=0,binding=2) writeonly buffer Output { float y[]; };
layout(push_constant) uniform Push {
    uint input_width,output_width,tokens,type,row_bytes;
    float scale;
} pc;
shared vec4 sums0[8],sums1[8];
#if TYPE == 20
shared float table[16];
#endif

int grid(uint q){
    uint v=q<8u?(q<4u?0xbfad9881u:0xf6eaddcfu):(q<12u?0x26190d01u:0x71594535u);
    return bitfieldExtract(int(v),int((q&3u)*8u),8);
}
vec4 quants(uint p){
    if(TYPE!=20u)return vec4(unpack8(int(p)));
#if TYPE == 20
    return vec4(table[p&255u],table[(p>>8u)&255u],table[(p>>16u)&255u],table[p>>24u]);
#else
    return vec4(0.0);
#endif
}
float product(uint p,vec4 a,vec4 b){
    return dot(quants(p&0x0f0f0f0fu),a)+dot(quants((p>>4u)&0x0f0f0f0fu),b);
}
float value(uint base,uint groups,uint row,uint g,uint lane,vec4 a,vec4 b){
    if(TYPE==13u){
        uint off=base+groups*16u+(row*groups+g)*5u+lane;
        uint p=w[off],next=w[off+1u],shift=lane*8u;
        uint lo=(p>>shift)|(shift==0u?0u:next<<(32u-shift));
        uint hi=next>>shift;
        vec4 q0=vec4(bitfieldExtract(lo,0,5),bitfieldExtract(lo,5,5),
            bitfieldExtract(lo,10,5),bitfieldExtract(lo,15,5));
        vec4 q1=vec4(bitfieldExtract(lo,20,5),bitfieldExtract(lo,25,5),
            ((lo>>30u)|(hi<<2u))&31u,(hi>>3u)&31u);
        return dot(q0,a)+dot(q1,b);
    }
    uint p=w[base+groups*(TYPE==20u?4u:16u)+(row*groups+g)*4u+lane];
    return product(p,a,b);
}
void main(){
    uint lid=gl_LocalInvocationID.x,lane=lid&3u,cohort=lid>>2u;
#if TYPE == 20
    if(lid<16u)table[lid]=float(grid(lid));
    barrier();
#endif
    uint first=gl_WorkGroupID.x*ROWS,token=gl_WorkGroupID.y;
    uint groups=pc.input_width/32u;
    uint base=(first/8u)*groups*(TYPE==13u?56u:TYPE==20u?36u:48u),row=first&7u;
    vec4 acc0=vec4(0.0),acc1=vec4(0.0);
    for(uint i=0u;i<(groups+WG/4u-1u)/(WG/4u);i++){
        uint group=i*(WG/4u)+cohort;
        bool live=group<groups;
        uint g=min(group,groups-1u);
        uint meta=base+g*16u;
        vec4 d0=uintBitsToFloat(w4[(meta+row)/4u]),d1=uintBitsToFloat(w4[meta/4u+1u]);
        vec4 m0=uintBitsToFloat(w4[(meta+row)/4u+2u]),m1=uintBitsToFloat(w4[meta/4u+3u]);
        if(TYPE==20u){
            uvec4 packed=w4[(base+g*4u)/4u];
            d0=row==0u?vec4(unpackHalf2x16(packed.x),unpackHalf2x16(packed.y)):
                vec4(unpackHalf2x16(packed.z),unpackHalf2x16(packed.w));
            d1=vec4(unpackHalf2x16(packed.z),unpackHalf2x16(packed.w));
            m0=vec4(0.0);m1=vec4(0.0);
        }
        vec4 a=live?x[token*(pc.input_width/4u)+g*8u+lane]:vec4(0.0);
        vec4 b=live?x[token*(pc.input_width/4u)+g*8u+lane+4u]:vec4(0.0);
        float sx=dot(a+b,vec4(1.0));
        vec4 p0=vec4(value(base,groups,row,g,lane,a,b),value(base,groups,row+1u,g,lane,a,b),
            value(base,groups,row+2u,g,lane,a,b),value(base,groups,row+3u,g,lane,a,b));
        acc0+=d0*p0-m0*sx;
        if(ROWS==8u){
            vec4 p1=vec4(value(base,groups,4u,g,lane,a,b),value(base,groups,5u,g,lane,a,b),
                value(base,groups,6u,g,lane,a,b),value(base,groups,7u,g,lane,a,b));
            acc1+=d1*p1-m1*sx;
        }
    }
    vec4 r0=vec4(subgroupAdd(acc0.x),subgroupAdd(acc0.y),subgroupAdd(acc0.z),subgroupAdd(acc0.w));
    vec4 r1=vec4(subgroupAdd(acc1.x),subgroupAdd(acc1.y),subgroupAdd(acc1.z),subgroupAdd(acc1.w));
    if(subgroupElect()){sums0[gl_SubgroupID]=r0;sums1[gl_SubgroupID]=r1;}
    barrier();
    if(lid==0u){
        vec4 a=vec4(0.0),b=vec4(0.0);
        for(uint s=0u;s<gl_NumSubgroups;s++){a+=sums0[s];b+=sums1[s];}
        for(uint r=0u;r<4u;r++){
            if(first+r<pc.output_width)y[token*pc.output_width+first+r]=a[r]*pc.scale;
            if(ROWS==8u&&first+r+4u<pc.output_width)y[token*pc.output_width+first+r+4u]=b[r]*pc.scale;
        }
    }
}
