layout(local_size_x=256) in;
#ifndef TK
#define TK 32
#endif
layout(set=0,binding=0) readonly buffer Weights { uint w[]; };
layout(set=0,binding=1) readonly buffer Input { float x[]; };
layout(set=0,binding=1) readonly buffer InputVector { vec4 x4[]; };
layout(set=0,binding=2) writeonly buffer Output { float y[]; };
layout(push_constant) uniform Push {
    uint input_width,output_width,tokens,type,row_bytes;
    float scale;
} pc;
shared float weights[32][64];
shared float inputs[32][TK+1];
#if TYPE == 20
shared float grid_table[16];
#endif

float dequant(uint row,uint group,uint i){
    uint groups=pc.input_width/32u;
    uint tile=(row/8u)*groups*(TYPE==13?56u:TYPE==20?36u:48u),r=row&7u;
    uint q=0u;
    if(TYPE==13){
        uint bit=((i&15u)/4u)*40u+((i&3u)+(i/16u)*4u)*5u;
        uint off=tile+groups*16u+(r*groups+group)*5u+(bit/32u);
        uint shift=bit&31u;
        q=w[off]>>shift;
        if(shift>27u)q|=w[off+1u]<<(32u-shift);
        q&=31u;
    }else{
        uint off=tile+groups*(TYPE==20?4u:16u)+(r*groups+group)*4u+((i&15u)/4u);
        q=(w[off]>>((i&3u)*8u+(i/16u)*4u))&15u;
    }
    if(TYPE==20){
        vec2 d=unpackHalf2x16(w[tile+group*4u+r/2u]);
#if TYPE == 20
        return d[r&1u]*grid_table[q];
#endif
    }
    uint meta=tile+group*16u+r;
    return uintBitsToFloat(w[meta])*float(q)-uintBitsToFloat(w[meta+8u]);
}

void main(){
    uint tid=gl_LocalInvocationID.x,row0=gl_WorkGroupID.x*64u,token0=gl_WorkGroupID.y*TK;
#if TYPE == 20
    if(tid<16u){
        uint word=tid<8u?(tid<4u?0xbfad9881u:0xf6eaddcfu):(tid<12u?0x26190d01u:0x71594535u);
        grid_table[tid]=float(bitfieldExtract(int(word),int((tid&3u)*8u),8));
    }
    barrier();
#endif
    uint tg=tid%(TK/2u),rg=tid/(TK/2u);
    float sum[4][2];
    for(uint i=0u;i<4u;i++)for(uint j=0u;j<2u;j++)sum[i][j]=0.0;
    for(uint group=0u;group<pc.input_width/32u;group++){
        uint staged_row=tid>>2u,element=(tid&3u)*8u,row=row0+staged_row;
        vec4 w0=vec4(0.0),w1=vec4(0.0);
        if(row<pc.output_width){
            w0=vec4(dequant(row,group,element),dequant(row,group,element+1u),
                dequant(row,group,element+2u),dequant(row,group,element+3u));
            w1=vec4(dequant(row,group,element+4u),dequant(row,group,element+5u),
                dequant(row,group,element+6u),dequant(row,group,element+7u));
        }
        weights[element][staged_row]=w0.x;weights[element+1u][staged_row]=w0.y;
        weights[element+2u][staged_row]=w0.z;weights[element+3u][staged_row]=w0.w;
        weights[element+4u][staged_row]=w1.x;weights[element+5u][staged_row]=w1.y;
        weights[element+6u][staged_row]=w1.z;weights[element+7u][staged_row]=w1.w;
        uint st=tid>>3u,k0=(tid&7u)*4u,token=token0+st;
        if(st<TK){
            vec4 v=token<pc.tokens?x4[token*(pc.input_width/4u)+group*8u+k0/4u]:vec4(0.0);
            inputs[k0][st]=v.x;inputs[k0+1u][st]=v.y;
            inputs[k0+2u][st]=v.z;inputs[k0+3u][st]=v.w;
        }
        barrier();
        for(uint k=0u;k<32u;k++){
            float a=inputs[k][tg],b=inputs[k][tg+TK/2u];
            float v0=weights[k][rg];
            sum[0][0]=fma(v0,a,sum[0][0]);sum[0][1]=fma(v0,b,sum[0][1]);
#if TK >= 16
            float v1=weights[k][rg+512u/TK];
            sum[1][0]=fma(v1,a,sum[1][0]);sum[1][1]=fma(v1,b,sum[1][1]);
#endif
#if TK == 32
            float v2=weights[k][rg+32u],v3=weights[k][rg+48u];
            sum[2][0]=fma(v2,a,sum[2][0]);sum[2][1]=fma(v2,b,sum[2][1]);
            sum[3][0]=fma(v3,a,sum[3][0]);sum[3][1]=fma(v3,b,sum[3][1]);
#endif
        }
        barrier();
    }
    for(uint i=0u;i<TK/8u;i++)for(uint j=0u;j<2u;j++){
        uint row=row0+rg+i*(512u/TK),token=token0+tg+j*(TK/2u);
        if(row<pc.output_width&&token<pc.tokens)y[token*pc.output_width+row]=sum[i][j]*pc.scale;
    }
}
