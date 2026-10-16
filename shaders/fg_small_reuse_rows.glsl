#include "fg_small_reuse_decode.glsl"
layout(constant_id=2) const uint SMALL_SPLITS=1u;
#define ROW_INPUT(T,A,B) \
    vec4 A=x[T*(pc.input_width/4u)+group*8u+lane],B=x[T*(pc.input_width/4u)+group*8u+lane+4u];
#define ROW_ACC(A,C,X,Y) \
    A.C=small_fma4(qhi,Y,small_fma4(qlo,X,A.C));
#if SMALL_N >= 3
#define ROW_ACC2(A,C) ROW_ACC(A,C,x2,y2)
#else
#define ROW_ACC2(A,C)
#endif
#if SMALL_N >= 4
#define ROW_ACC3(A,C) ROW_ACC(A,C,x3,y3)
#else
#define ROW_ACC3(A,C)
#endif
#define REUSE_ROW(R,C,A0,A1,A2,A3) { \
    vec4 qlo,qhi;reuse_weight(base,groups,tile_row+R,group,lane,qlo,qhi); \
    ROW_ACC(A0,C,x0,y0) ROW_ACC(A1,C,x1,y1) ROW_ACC2(A2,C) ROW_ACC3(A3,C) }
#define ROW_REDUCE(A) for(uint offset=1u;offset<32u;offset*=2u)A+=subgroupShuffleXor(A,offset);
#define ROW_STORE(T,R,A) \
    for(uint i=0u;i<4u;i++)if(row+R+i<pc.output_width) \
        y[(T*pc.output_width+row+R+i)*SMALL_SPLITS+split]=A[i]*pc.scale;
void main(){
#if TYPE == 20
    if(gl_LocalInvocationID.x<16u)table[gl_LocalInvocationID.x]=float(grid(gl_LocalInvocationID.x));
    barrier();
#endif
    uint lid=gl_LocalInvocationID.x,cohort=lid&31u,lane=cohort&3u;
    uint split=SMALL_SPLITS==1u?0u:gl_WorkGroupID.y;
    uint row=gl_WorkGroupID.x*(WG/32u*FG_SMALL_REUSE_ROWS)+(lid/32u)*FG_SMALL_REUSE_ROWS;
    if(row>=pc.output_width)return;
    uint groups=pc.input_width/32u,tile_row=row&7u;
    uint base=(row/8u)*groups*(TYPE==13u?(COMPACT!=0u?45u:56u):
        TYPE==20u?36u:(COMPACT!=0u?37u:48u));
    vec4 a0=vec4(0.0),a1=vec4(0.0);
#if SMALL_N >= 3
    vec4 a2=vec4(0.0);
#endif
#if SMALL_N >= 4
    vec4 a3=vec4(0.0);
#endif
#if FG_SMALL_REUSE_ROWS == 8
    vec4 b0=vec4(0.0),b1=vec4(0.0);
#if SMALL_N >= 3
    vec4 b2=vec4(0.0);
#endif
#if SMALL_N >= 4
    vec4 b3=vec4(0.0);
#endif
#endif
    for(uint group=cohort/4u+split*8u;group<groups;group+=8u*SMALL_SPLITS){
        ROW_INPUT(0u,x0,y0) ROW_INPUT(1u,x1,y1)
#if SMALL_N >= 3
        ROW_INPUT(2u,x2,y2)
#endif
#if SMALL_N >= 4
        ROW_INPUT(3u,x3,y3)
#endif
        REUSE_ROW(0u,x,a0,a1,a2,a3) REUSE_ROW(1u,y,a0,a1,a2,a3)
        REUSE_ROW(2u,z,a0,a1,a2,a3) REUSE_ROW(3u,w,a0,a1,a2,a3)
#if FG_SMALL_REUSE_ROWS == 8
        REUSE_ROW(4u,x,b0,b1,b2,b3) REUSE_ROW(5u,y,b0,b1,b2,b3)
        REUSE_ROW(6u,z,b0,b1,b2,b3) REUSE_ROW(7u,w,b0,b1,b2,b3)
#endif
    }
    ROW_REDUCE(a0) ROW_REDUCE(a1)
#if SMALL_N >= 3
    ROW_REDUCE(a2)
#endif
#if SMALL_N >= 4
    ROW_REDUCE(a3)
#endif
#if FG_SMALL_REUSE_ROWS == 8
    ROW_REDUCE(b0) ROW_REDUCE(b1)
#if SMALL_N >= 3
    ROW_REDUCE(b2)
#endif
#if SMALL_N >= 4
    ROW_REDUCE(b3)
#endif
#endif
    if(cohort==0u){
        ROW_STORE(0u,0u,a0) ROW_STORE(1u,0u,a1)
#if SMALL_N >= 3
        ROW_STORE(2u,0u,a2)
#endif
#if SMALL_N >= 4
        ROW_STORE(3u,0u,a3)
#endif
#if FG_SMALL_REUSE_ROWS == 8
        ROW_STORE(0u,4u,b0) ROW_STORE(1u,4u,b1)
#if SMALL_N >= 3
        ROW_STORE(2u,4u,b2)
#endif
#if SMALL_N >= 4
        ROW_STORE(3u,4u,b3)
#endif
#endif
    }
}
