#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_GOOGLE_include_directive : require
layout(local_size_x=32) in;
layout(set=0,binding=0) readonly buffer Weights{uvec2 w[];};
layout(set=0,binding=1) readonly buffer Tiles{uint tiles[];};
layout(set=0,binding=2) readonly buffer Input{vec4 x[];};
layout(set=0,binding=3) writeonly buffer Parts{float parts[];};
layout(push_constant) uniform Push{uint rows,in_dim,tile_bytes,stride;}pc;
#include "fg_small_down_decode.glsl"
void main(){
    uint lid=gl_LocalInvocationID.x;
    uint part=lid&3u,seg=part&1u,parity=part>>1u;
    uint row=gl_WorkGroupID.x*16u+(lid>>2u)*2u,base=gl_WorkGroupID.y*5u;
    uint expert=tiles[base];
    if(expert==0xffffffffu)return;
    uint first=tiles[base+1u],second=tiles[base+2u];
    uint third=tiles[base+3u],fourth=tiles[base+4u];
    if(first==0xffffffffu)return;
    uint row_base=expert*pc.stride+(row>>4u)*pc.tile_bytes,tile_row=row&15u;
    uint sm_base=(row_base+tile_row*4u)>>3u;
    uint qt_base=((row_base+128u+tile_row*16u)>>3u)+seg;
    uint bit_base=seg*8u;
    vec2 a=vec2(0.0),b=vec2(0.0),c=vec2(0.0),d=vec2(0.0);
    for(uint block=parity;block<pc.in_dim/32u;block+=2u){
        uvec2 sm=w[sm_base+block*48u],hw=w[sm_base+block*48u+8u];
        uvec2 q0=w[qt_base+block*48u],q1=w[qt_base+block*48u+2u];
        vec2 s0=unpackHalf2x16(sm.x),s1=unpackHalf2x16(sm.y);
        vec4 w0=decode(q0.x&0x0f0f0f0fu,hw.x>>bit_base,s0);
        vec4 w1=decode((q0.x>>4u)&0x0f0f0f0fu,hw.x>>(16u+bit_base),s0);
        vec4 w2=decode(q0.y&0x0f0f0f0fu,hw.x>>(bit_base+4u),s0);
        vec4 w3=decode((q0.y>>4u)&0x0f0f0f0fu,hw.x>>(20u+bit_base),s0);
        vec4 p0=decode(q1.x&0x0f0f0f0fu,hw.y>>bit_base,s1);
        vec4 p1=decode((q1.x>>4u)&0x0f0f0f0fu,hw.y>>(16u+bit_base),s1);
        vec4 p2=decode(q1.y&0x0f0f0f0fu,hw.y>>(bit_base+4u),s1);
        vec4 p3=decode((q1.y>>4u)&0x0f0f0f0fu,hw.y>>(20u+bit_base),s1);
        apply_pair(first,block,seg,w0,w1,w2,w3,p0,p1,p2,p3,a);
        if(second!=0xffffffffu)apply_pair(second,block,seg,w0,w1,w2,w3,p0,p1,p2,p3,b);
        if(third!=0xffffffffu)apply_pair(third,block,seg,w0,w1,w2,w3,p0,p1,p2,p3,c);
        if(fourth!=0xffffffffu)apply_pair(fourth,block,seg,w0,w1,w2,w3,p0,p1,p2,p3,d);
    }
    parts[(first*pc.rows+row)*4u+part]=a.x;
    parts[(first*pc.rows+row+1u)*4u+part]=a.y;
    if(second!=0xffffffffu){
        parts[(second*pc.rows+row)*4u+part]=b.x;
        parts[(second*pc.rows+row+1u)*4u+part]=b.y;
    }
    if(third!=0xffffffffu){
        parts[(third*pc.rows+row)*4u+part]=c.x;
        parts[(third*pc.rows+row+1u)*4u+part]=c.y;
    }
    if(fourth!=0xffffffffu){
        parts[(fourth*pc.rows+row)*4u+part]=d.x;
        parts[(fourth*pc.rows+row+1u)*4u+part]=d.y;
    }
}
