#include "fg_small_reuse_decode.glsl"
#define REUSE_COLUMN(T,A) { \
    uint at=T*(pc.input_width/4u)+group*8u+lane; \
    vec4 a=x[at],b=x[at+4u]; \
    A.x=small_fma4(hi0,b,small_fma4(lo0,a,A.x)); \
    A.y=small_fma4(hi1,b,small_fma4(lo1,a,A.y)); }
#define REUSE_REDUCE(A) \
    for(uint offset=1u;offset<32u;offset*=2u)A+=subgroupShuffleXor(A,offset);
#define REUSE_STORE(T,A) \
    if(row<pc.output_width)y[T*pc.output_width+row]=A.x*pc.scale; \
    if(row+1u<pc.output_width)y[T*pc.output_width+row+1u]=A.y*pc.scale;
void main(){
#if TYPE == 20
    if(gl_LocalInvocationID.x<16u)table[gl_LocalInvocationID.x]=float(grid(gl_LocalInvocationID.x));
    barrier();
#endif
    uint lid=gl_LocalInvocationID.x,cohort=lid&31u,lane=cohort&3u;
    uint row=gl_WorkGroupID.x*(WG/16u)+(lid/32u)*2u;
    if(row>=pc.output_width)return;
    uint groups=pc.input_width/32u,tile_row=row&7u;
    uint base=(row/8u)*groups*(TYPE==13u?(COMPACT!=0u?45u:56u):
        TYPE==20u?36u:(COMPACT!=0u?37u:48u));
    vec2 acc0=vec2(0.0),acc1=vec2(0.0);
#if SMALL_N >= 3
    vec2 acc2=vec2(0.0);
#endif
#if SMALL_N >= 4
    vec2 acc3=vec2(0.0);
#endif
    for(uint group=cohort/4u;group<groups;group+=8u){
        vec4 lo0,hi0,lo1,hi1;
        reuse_weight(base,groups,tile_row,group,lane,lo0,hi0);
        reuse_weight(base,groups,tile_row+1u,group,lane,lo1,hi1);
        REUSE_COLUMN(0u,acc0) REUSE_COLUMN(1u,acc1)
#if SMALL_N >= 3
        REUSE_COLUMN(2u,acc2)
#endif
#if SMALL_N >= 4
        REUSE_COLUMN(3u,acc3)
#endif
    }
    REUSE_REDUCE(acc0) REUSE_REDUCE(acc1)
#if SMALL_N >= 3
    REUSE_REDUCE(acc2)
#endif
#if SMALL_N >= 4
    REUSE_REDUCE(acc3)
#endif
    if(cohort==0u){
        REUSE_STORE(0u,acc0) REUSE_STORE(1u,acc1)
#if SMALL_N >= 3
        REUSE_STORE(2u,acc2)
#endif
#if SMALL_N >= 4
        REUSE_STORE(3u,acc3)
#endif
    }
}
