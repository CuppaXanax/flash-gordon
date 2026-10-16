#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=FG_ROWS_WG)in;
layout(set=0,binding=0)readonly buffer Weights{uint w[];};
layout(set=0,binding=0)readonly buffer Vectors{uvec4 w4[];};
layout(set=0,binding=1)readonly buffer Input{vec4 x[];};
layout(set=0,binding=2)writeonly buffer Output{float y[];};
layout(push_constant)uniform Push{
    uint input_width,output_width,tokens,type,row_bytes;float scale;
}pc;
void main(){
    uint lid=gl_LocalInvocationID.x,lane=lid%4u;
    uint row=gl_WorkGroupID.x*(FG_ROWS_WG/4u)+lid/4u;
    uint groups=pc.input_width/32u,base=(row/8u)*groups*37u;
    float acc[FG_ROWS_PREFETCH];
    [[unroll]]for(uint p=0u;p<FG_ROWS_PREFETCH;p++)acc[p]=0.0;
    if(row<pc.output_width){
        for(uint first=lane;first<groups;first+=4u*FG_ROWS_PREFETCH){
            uvec4 quant[FG_ROWS_PREFETCH];uint meta[FG_ROWS_PREFETCH],codes[FG_ROWS_PREFETCH];
            [[unroll]]for(uint p=0u;p<FG_ROWS_PREFETCH;p++){
                uint g=min(first+p*4u,groups-1u);
                meta[p]=w[base+(g/8u)*8u+row%8u];
                codes[p]=w[base+groups+g*4u+(row%8u)/2u];
                quant[p]=w4[(base+groups*5u+((row%8u)*groups+g)*4u)/4u];
            }
            [[unroll]]for(uint p=0u;p<FG_ROWS_PREFETCH;p++){
                uint g=first+p*4u;if(g>=groups)continue;
                vec2 sm=unpackHalf2x16(meta[p]);
                uint shift=(row%2u)*16u;
                float d=sm.x*float((codes[p]>>shift)&255u),m=sm.y*float((codes[p]>>(shift+8u))&255u);
                float dot_value=0.0,input_sum=0.0;
                [[unroll]]for(uint j=0u;j<4u;j++){
                    vec4 a=x[g*8u+j],b=x[g*8u+j+4u];
                    uint q=quant[p][j];
                    dot_value+=dot(vec4(unpack8(int(q&0x0f0f0f0fu))),a)+
                               dot(vec4(unpack8(int((q>>4u)&0x0f0f0f0fu))),b);
                    input_sum+=dot(a+b,vec4(1.0));
                }
                acc[p]+=d*dot_value-m*input_sum;
            }
        }
    }
    float total=0.0;
    [[unroll]]for(uint p=0u;p<FG_ROWS_PREFETCH;p++)total+=acc[p];
    total+=subgroupShuffleXor(total,1u);
    total+=subgroupShuffleXor(total,2u);
    if(row<pc.output_width&&lane==0u)y[row]=total*pc.scale;
}
