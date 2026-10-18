#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=WG)in;
layout(set=0,binding=0)readonly buffer Weights{uint w[];};
layout(set=0,binding=0)readonly buffer Vectors{uvec4 w4[];};
layout(set=0,binding=1)readonly buffer Input{vec4 x[];};
layout(set=0,binding=2)writeonly buffer Output{float y[];};
layout(push_constant)uniform Push{uint k,m,tokens,type,row_bytes;float scale;}pc;
#if TYPE == 20
shared float lut[16];
int grid(uint q){
    uint v=q<8u?(q<4u?0xbfad9881u:0xf6eaddcfu):(q<12u?0x26190d01u:0x71594535u);
    return bitfieldExtract(int(v),int((q&3u)*8u),8);
}
#endif
vec4 decode(uint q){
#if TYPE == 20
    return vec4(lut[q&15u],lut[(q>>8u)&15u],lut[(q>>16u)&15u],lut[q>>24u]);
#else
    return vec4(unpack8(int(q)));
#endif
}
void main(){
    uint lid=gl_LocalInvocationID.x,row=lid&7u,part=(lid%WAVE)/8u;
    uint first=(gl_WorkGroupID.x*(WG/WAVE)+lid/WAVE)*8u;
    uint groups=pc.k/32u,base=(first/8u)*groups*(TYPE==20?36u:37u);
    uint token=gl_WorkGroupID.y,begin=0u,end=groups;
#ifdef FG_SPLITS
    token=0u;begin=groups*gl_WorkGroupID.y/FG_SPLITS;end=groups*(gl_WorkGroupID.y+1u)/FG_SPLITS;
#endif
#if TYPE == 20
    if(lid<16u)lut[lid]=float(grid(lid));
    barrier();
#endif
    float acc=0.0;
    for(uint first_group=begin+part;first_group<end;first_group+=(WAVE/8u)*PREFETCH){
        uvec4 packed[PREFETCH];
        vec2 dm[PREFETCH];
        uint codes[PREFETCH];
        [[unroll]]for(uint j=0u;j<PREFETCH;j++){
            uint g=min(first_group+j*(WAVE/8u),end-1u);
            packed[j]=w4[(base+groups*(TYPE==20?4u:5u))/4u+row*groups+g];
#if TYPE == 20
            vec2 h=unpackHalf2x16(w[base+g*4u+row/2u]);
            dm[j]=vec2((row&1u)==0u?h.x:h.y,0.0);codes[j]=1u;
#else
            dm[j]=unpackHalf2x16(w[base+(g/8u)*8u+row]);
            codes[j]=(w[base+groups+g*4u+row/2u]>>((row&1u)*16u))&65535u;
#endif
        }
        [[unroll]]for(uint j=0u;j<PREFETCH;j++){
            uint g=first_group+j*(WAVE/8u);
            if(g>=end)continue;
            float dot_sum=0.0,input_sum=0.0;
            [[unroll]]for(uint t=0u;t<4u;t++){
                vec4 a=x[token*(pc.k/4u)+g*8u+t];
                vec4 b=x[token*(pc.k/4u)+g*8u+t+4u];
                uint p=packed[j][t];
                dot_sum+=dot(decode(p&0x0f0f0f0fu),a)+dot(decode((p>>4u)&0x0f0f0f0fu),b);
                input_sum+=dot(a+b,vec4(1.0));
            }
            acc+=dm[j].x*float(codes[j]&255u)*dot_sum-dm[j].y*float(codes[j]>>8u)*input_sum;
        }
    }
    for(uint offset=WAVE/2u;offset>=8u;offset>>=1u)acc+=subgroupShuffleXor(acc,offset);
    if(part==0u&&first+row<pc.m)y[gl_WorkGroupID.y*pc.m+first+row]=acc*pc.scale;
}
