#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require
layout(constant_id=0) const uint gate_q5=0u;
layout(constant_id=1) const uint up_q5=0u;
layout(local_size_x=64*PAIR_LANES) in;
layout(set=0,binding=0) readonly buffer Gate{uint gw[];};
layout(set=0,binding=0) readonly buffer GateVector{uvec4 gv[];};
layout(set=0,binding=1) readonly buffer Up{uint uw[];};
layout(set=0,binding=1) readonly buffer UpVector{uvec4 uv[];};
layout(set=0,binding=2) readonly buffer Activation{uint q8[];};
layout(set=0,binding=3) readonly buffer Tiles{uint tiles[];};
layout(set=0,binding=4) writeonly buffer Mid{float mid[];};
layout(push_constant) uniform Push{
    uint rows,blocks,gtile,utile,gs,us,gt,ut;
}pc;
shared float gsum[64*PAIR_LANES],usum[64*PAIR_LANES];
vec2 scale_min(uint a,uint b,uint c,uint group){
    if(group<4u){
        uint s=group*8u;
        return vec2((a>>s)&63u,(b>>s)&63u);
    }
    uint s=(group-4u)*8u;
    return vec2(((c>>s)&15u)|(((a>>(s+6u))&3u)<<4u),
        ((c>>(s+4u))&15u)|(((b>>(s+6u))&3u)<<4u));
}
int sum32(uint p){
    int a=int(p&65535u),b=int(p>>16u);
    return a-(a>=32768?65536:0)+b-(b>=32768?65536:0);
}
void main(){
    uint pair_column=gl_LocalInvocationID.x%PAIR_LANES;
    uint lid=gl_LocalInvocationID.x/PAIR_LANES,lane=lid&7u,row_slot=lid>>3u;
    uint row=gl_WorkGroupID.x*8u+row_slot,base=gl_WorkGroupID.y*5u;
    uint expert=tiles[base];
    if(expert==0xffffffffu)return;
    // Top-10 IDs are distinct per token: a two-token union has at most two pairs.
    uint pairs[2];
    float ga[2],ua[2],gm[2],um[2];
    [[unroll]] for(uint p=0u;p<2u;p++){
        pairs[p]=tiles[base+1u+pair_column*2u+p];
        ga[p]=0.0;ua[p]=0.0;gm[p]=0.0;um[p]=0.0;
    }
    uint sub=lane&1u,group_a=(lane>>1u)*2u,group_b=group_a+1u;
    uint rt=row&7u;
    bool g5=gate_q5!=0u,u5=up_q5!=0u;
    for(uint block=0u;block<pc.blocks;block++){
        uint gb=expert*pc.gs+(row>>3u)*pc.gtile+block*(g5?1408u:1152u);
        uint ub=expert*pc.us+(row>>3u)*pc.utile+block*(u5?1408u:1152u);
        vec2 gd=unpackHalf2x16(gw[(gb+rt*4u)>>2u]);
        vec2 ud=unpackHalf2x16(uw[(ub+rt*4u)>>2u]);
        uint g0=gw[(gb+32u+rt*12u)>>2u],g1=gw[(gb+36u+rt*12u)>>2u],
             g2=gw[(gb+40u+rt*12u)>>2u];
        uint u0=uw[(ub+32u+rt*12u)>>2u],u1=uw[(ub+36u+rt*12u)>>2u],
             u2=uw[(ub+40u+rt*12u)>>2u];
        vec2 gma=scale_min(g0,g1,g2,group_a),gmb=scale_min(g0,g1,g2,group_b);
        vec2 uma=scale_min(u0,u1,u2,group_a),umb=scale_min(u0,u1,u2,group_b);
        uvec4 gq=gv[((gb+(g5?384u:128u)+rt*128u)>>4u)+lane];
        uvec4 uq=uv[((ub+(u5?384u:128u)+rt*128u)>>4u)+lane];
        uvec4 gh=uvec4(0u),uh=uvec4(0u);
        if(g5)gh=gv[((gb+128u+rt*32u)>>4u)+sub];
        if(u5)uh=uv[((ub+128u+rt*32u)>>4u)+sub];
        float gia[2],gib[2],uia[2],uib[2];
        [[unroll]] for(uint p=0u;p<2u;p++){
            gia[p]=0.0;gib[p]=0.0;uia[p]=0.0;uib[p]=0.0;
        }
        [[unroll]] for(uint w=0u;w<4u;w++){
            uint a=gq[w]&0x0f0f0f0fu,b=(gq[w]>>4u)&0x0f0f0f0fu;
            uint c=uq[w]&0x0f0f0f0fu,d=(uq[w]>>4u)&0x0f0f0f0fu;
            if(g5){a|=((gh[w]>>group_a)&0x01010101u)<<4u;b|=((gh[w]>>group_b)&0x01010101u)<<4u;}
            if(u5){c|=((uh[w]>>group_a)&0x01010101u)<<4u;d|=((uh[w]>>group_b)&0x01010101u)<<4u;}
            vec4 gl=vec4(unpack8(int(a))),gr=vec4(unpack8(int(b))),
                ul=vec4(unpack8(int(c))),ur=vec4(unpack8(int(d)));
            [[unroll]] for(uint p=0u;p<2u;p++){
                if(pairs[p]==0xffffffffu)continue;
                uint ab=((pairs[p]/10u)*pc.blocks+block)*74u;
                vec4 x=vec4(unpack8(int(q8[ab+2u+group_a*8u+sub*4u+w])));
                vec4 y=vec4(unpack8(int(q8[ab+2u+group_b*8u+sub*4u+w])));
                gia[p]+=dot(gl,x);gib[p]+=dot(gr,y);
                uia[p]+=dot(ul,x);uib[p]+=dot(ur,y);
            }
        }
        [[unroll]] for(uint p=0u;p<2u;p++){
            if(pairs[p]==0xffffffffu)continue;
            uint ab=((pairs[p]/10u)*pc.blocks+block)*74u;
            float delta=uintBitsToFloat(q8[ab]);
            ga[p]=fma(gd.x*delta,gma.x*gia[p]+gmb.x*gib[p],ga[p]);
            ua[p]=fma(ud.x*delta,uma.x*uia[p]+umb.x*uib[p],ua[p]);
            if(sub==0u){
                int a=sum32(q8[ab+66u+group_a]),b=sum32(q8[ab+66u+group_b]);
                gm[p]=fma(-gd.y*delta,float(int(gma.y)*a+int(gmb.y)*b),gm[p]);
                um[p]=fma(-ud.y*delta,float(int(uma.y)*a+int(umb.y)*b),um[p]);
            }
        }
    }
    [[unroll]] for(uint p=0u;p<2u;p++){
        gsum[gl_LocalInvocationID.x]=ga[p]+gm[p];
        usum[gl_LocalInvocationID.x]=ua[p]+um[p];
        barrier();
        if(lane==0u&&pairs[p]!=0xffffffffu){
            uint o=row_slot*8u*PAIR_LANES+pair_column;
            float g=((gsum[o]+gsum[o+4u*PAIR_LANES])+(gsum[o+2u*PAIR_LANES]+gsum[o+6u*PAIR_LANES]))+
                ((gsum[o+PAIR_LANES]+gsum[o+5u*PAIR_LANES])+(gsum[o+3u*PAIR_LANES]+gsum[o+7u*PAIR_LANES]));
            float u=((usum[o]+usum[o+4u*PAIR_LANES])+(usum[o+2u*PAIR_LANES]+usum[o+6u*PAIR_LANES]))+
                ((usum[o+PAIR_LANES]+usum[o+5u*PAIR_LANES])+(usum[o+3u*PAIR_LANES]+usum[o+7u*PAIR_LANES]));
            mid[pairs[p]*pc.rows+row]=(g/(1.0+exp(-g)))*u;
        }
        barrier();
    }
}
