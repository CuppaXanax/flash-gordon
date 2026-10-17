#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=256)in;
layout(constant_id=0)const uint TOKENS=4u;
layout(set=0,binding=0)readonly buffer Weights{uint w[];};
layout(set=0,binding=1)readonly buffer Input{vec4 x[];};
layout(set=0,binding=2)readonly buffer Normed{float normed[];};
layout(set=0,binding=3)readonly buffer Partials{float partials[];};
layout(set=0,binding=4)writeonly buffer Mixed{float mixed[];};
layout(set=0,binding=5)writeonly buffer Injection{float injection[];};
layout(push_constant)uniform Push{uint input_width,hidden,pieces;}pc;
shared float lut[16],values[TOKENS*32u];
int grid(uint q){
    uint v=q<8u?(q<4u?0xbfad9881u:0xf6eaddcfu):(q<12u?0x26190d01u:0x71594535u);
    return bitfieldExtract(int(v),int((q&3u)*8u),8);
}
vec4 iq(uint p){return vec4(lut[p&15u],lut[(p>>8u)&15u],lut[(p>>16u)&15u],lut[p>>24u]);}
float sigmoid(float v){return 1.0/(1.0+exp(-v));}
void main(){
    uint lid=gl_LocalInvocationID.x;
#if TYPE == 20
    if(lid<16u)lut[lid]=float(grid(lid));
    barrier();
#endif
    uint lane=gl_SubgroupInvocationID&3u,cohort=gl_SubgroupInvocationID/4u;
    uint hc=gl_SubgroupID,first=gl_WorkGroupID.x*8u+hc*pc.hidden,groups=pc.input_width/32u;
    float acc[TOKENS][8];
    [[unroll]]for(uint t=0u;t<TOKENS;t++)
        [[unroll]]for(uint r=0u;r<8u;r++)acc[t][r]=0.0;
    for(uint g=cohort;g<groups;g+=16u){
        vec4 a[TOKENS],b[TOKENS];
        [[unroll]]for(uint t=0u;t<TOKENS;t++){
            a[t]=x[t*(pc.input_width/4u)+g*8u+lane];
            b[t]=x[t*(pc.input_width/4u)+g*8u+lane+4u];
        }
        [[unroll]]for(uint r=0u;r<8u;r++){
            uint row=first+r;
#if TYPE == 20
            uint base=(row/8u)*groups*36u;
            vec2 scale=unpackHalf2x16(w[base+g*4u+(row%8u)/2u]);
            float d=(row&1u)==0u?scale.x:scale.y;
            uint p=w[base+groups*4u+((row%8u)*groups+g)*4u+lane];
            vec4 q0=iq(p&0x0f0f0f0fu),q1=iq((p>>4u)&0x0f0f0f0fu);
#else
            uint quant_offset=(groups*32u+63u)&~63u;
            uint tile_bytes=(quant_offset+groups*512u+63u)&~63u;
            uint base=(row/16u)*tile_bytes;
            vec2 scale=unpackHalf2x16(w[base/4u+g*8u+(row%16u)/2u]);
            float d=(row&1u)==0u?scale.x:scale.y;
            uint at=(base+quant_offset+(row%16u)*groups*32u+g*32u)/4u+lane;
            vec4 q0=vec4(unpack8(int(w[at]))),q1=vec4(unpack8(int(w[at+4u])));
#endif
            [[unroll]]for(uint t=0u;t<TOKENS;t++)
                acc[t][r]=fma(d,dot(q0,a[t])+dot(q1,b[t]),acc[t][r]);
        }
    }
    [[unroll]]for(uint t=0u;t<TOKENS;t++)[[unroll]]for(uint r=0u;r<8u;r++){
        float v=subgroupAdd(acc[t][r]);
        if(subgroupElect())values[(t*4u+hc)*8u+r]=v;
    }
    barrier();
    if(lid<8u){
        uint index=gl_WorkGroupID.x*8u+lid;
        [[unroll]]for(uint t=0u;t<TOKENS;t++){
            float v=0.0;
            [[unroll]]for(uint g=0u;g<4u;g++)
                v=fma(sigmoid(values[(t*4u+g)*8u+lid]),normed[(t*4u+g)*pc.hidden+index],v);
            mixed[t*pc.hidden+index]=v*0.25;
        }
    }
    if(gl_WorkGroupID.x==0u&&lid<4u){
        [[unroll]]for(uint t=0u;t<TOKENS;t++){
            float v=0.0;
            for(uint p=0u;p<pc.pieces;p++)v+=partials[(t*pc.pieces+p)*4u+lid];
            injection[t*4u+lid]=2.0*sigmoid(v*0.25);
        }
    }
}
