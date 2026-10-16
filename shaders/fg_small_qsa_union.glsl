#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require
#ifdef FG_QSA_TILE16
layout(local_size_x=384) in;
#else
layout(local_size_x=768) in;
#endif
layout(constant_id=0) const uint N=4u;
layout(set=0,binding=0) readonly buffer Records{uint records[];};
layout(set=0,binding=1) readonly buffer Query{vec4 query4[];};
layout(set=0,binding=2) writeonly buffer Partials{float partials[];};
layout(set=0,binding=3) readonly buffer Descriptors{uvec4 descriptors[];};
layout(push_constant) uniform Push{uint splits,blocks;}pc;
shared vec4 keys[256],values[256];
#ifdef FG_QSA_TILE16
shared vec4 cached_query[384*N];
#endif
const float NEG=-3.402823466e38,L2E=1.4426950408889634;
float dot4(vec4 a,vec4 b){
    float s=a.x*b.x;s=fma(a.y,b.y,s);s=fma(a.z,b.z,s);
    return fma(a.w,b.w,s);
}
void decode(uint group,uint cache_slot,uint kv){
    uint record=group>>7u,within=group&127u,slot=within&63u;
    uint base=(cache_slot*4u+record)*1236u;
    uint scale_at=base+(within<64u?0u:544u)+kv*272u+(slot>>3u)*34u;
    uint packed_at=scale_at+2u+(slot&7u)*4u,shift=(packed_at&3u)*8u;
    uint lo=records[packed_at>>2u],hi=records[(packed_at>>2u)+1u];
    uint packed=(lo>>shift)|((hi<<((32u-shift)&31u))&(shift==0u?0u:0xffffffffu));
    float scale=unpackHalf2x16(records[scale_at>>2u]>>((scale_at&3u)*8u)).x;
    vec4 v=unpackSnorm4x8(packed)*(scale*127.0);
    if(within<64u)keys[record*64u+slot]=v;
    else values[record*64u+slot]=v;
}
void main(){
    uint kv=gl_WorkGroupID.y,split=gl_WorkGroupID.z;
    uint lane=gl_SubgroupInvocationID,head=kv*12u+gl_SubgroupID;
#ifdef FG_QSA_TILE16
    kv=gl_WorkGroupID.y/2u;
    head=kv*12u+(gl_WorkGroupID.y&1u)*6u+gl_SubgroupID;
#else
    vec4 q[N];
#endif
    vec4 acc[N];
    float maximum[N],denominator[N];
    [[unroll]] for(uint p=0u;p<N;p++){
#ifdef FG_QSA_TILE16
        cached_query[p*384u+gl_LocalInvocationIndex]=
            query4[p*1536u+head*64u+lane]*0.0625;
#else
        q[p]=query4[p*1536u+head*64u+lane]*0.0625;
#endif
        acc[p]=vec4(0.0);maximum[p]=NEG;denominator[p]=0.0;
    }
#ifdef FG_QSA_TILE16
    barrier();
#endif
    for(uint block=0u;block<pc.blocks;block++){
        uvec4 descriptor=descriptors[block];
        // Absolute-block partitioning keeps a query's reduction independent
        // of other queries' selected sets, including changed future suffixes.
        if(descriptor.z%pc.splits!=split)continue;
        barrier();
#ifdef FG_QSA_TILE16
        for(uint i=gl_LocalInvocationIndex;i<512u;i+=384u)
            decode(i,descriptor.x,kv);
#else
        if(gl_LocalInvocationIndex<512u)
            decode(gl_LocalInvocationIndex,descriptor.x,kv);
#endif
        barrier();
        [[unroll]] for(uint p=0u;p<N;p++){
            uint mask=(descriptor.y>>(p*4u))&15u;
            if(mask==0u)continue;
            float s0=NEG,s1=NEG,s2=NEG,s3=NEG;
#ifdef FG_QSA_TILE16
            uint quarter=lane&15u,record=lane>>4u;
            float score=0.0;
            [[unroll]] for(uint v=0u;v<4u;v++)
                score+=dot4(cached_query[p*384u+gl_SubgroupID*64u+quarter+v*16u],
                    keys[record*64u+quarter+v*16u]);
            score+=subgroupShuffleXor(score,1u);
            score+=subgroupShuffleXor(score,2u);
            score+=subgroupShuffleXor(score,4u);
            score+=subgroupShuffleXor(score,8u);
            if((mask&(1u<<record))==0u)score=NEG;
            s0=subgroupShuffle(score,0u);s1=subgroupShuffle(score,16u);
            s2=subgroupShuffle(score,32u);s3=subgroupShuffle(score,48u);
#else
            if((mask&1u)!=0u)s0=subgroupAdd(dot4(q[p],keys[lane]));
            if((mask&2u)!=0u)s1=subgroupAdd(dot4(q[p],keys[64u+lane]));
            if((mask&4u)!=0u)s2=subgroupAdd(dot4(q[p],keys[128u+lane]));
            if((mask&8u)!=0u)s3=subgroupAdd(dot4(q[p],keys[192u+lane]));
#endif
            float next=max(maximum[p],max(max(s0,s1),max(s2,s3)));
            float scale=maximum[p]<-3.0e38?0.0:exp2((maximum[p]-next)*L2E);
            float bias=-next*L2E;
            float w0=(mask&1u)!=0u?exp2(fma(s0,L2E,bias)):0.0;
            float w1=(mask&2u)!=0u?exp2(fma(s1,L2E,bias)):0.0;
            float w2=(mask&4u)!=0u?exp2(fma(s2,L2E,bias)):0.0;
            float w3=(mask&8u)!=0u?exp2(fma(s3,L2E,bias)):0.0;
            acc[p]*=scale;
            acc[p]=fma(vec4(w0),values[lane],acc[p]);
            acc[p]=fma(vec4(w1),values[64u+lane],acc[p]);
            acc[p]=fma(vec4(w2),values[128u+lane],acc[p]);
            acc[p]=fma(vec4(w3),values[192u+lane],acc[p]);
            denominator[p]=denominator[p]*scale+((w0+w1)+(w2+w3));
            maximum[p]=next;
        }
    }
    [[unroll]] for(uint p=0u;p<N;p++){
        uint base=((p*24u+head)*pc.splits+split)*258u;
        partials[base+lane*4u]=acc[p].x;
        partials[base+lane*4u+1u]=acc[p].y;
        partials[base+lane*4u+2u]=acc[p].z;
        partials[base+lane*4u+3u]=acc[p].w;
        if(lane==0u){partials[base+256u]=maximum[p];partials[base+257u]=denominator[p];}
    }
}
