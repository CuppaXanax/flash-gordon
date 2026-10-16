#define main q4_repack_cli
#include "../tools/q4_repack.c"
#undef main

#include "fg_vk.h"

static int lossless_iq4_expansion(void){
    int ok=1;
    for(uint32_t k=320u;k<=640u;k+=320u){
        uint32_t rows=17u;uint64_t values=(uint64_t)k*rows,raw_bytes=values/32u*18u;
        uint64_t cooked_bytes=fg_q8_0_cooked_matrix_bytes(k,rows),dense_bytes=fg_dense_compact_bytes(k,rows,20u);
        uint8_t *raw=malloc(raw_bytes),*dense=malloc(dense_bytes),*cooked=malloc(cooked_bytes),*q8=malloc(values/32u*34u);
        float *expected=malloc(values*4u),*got=malloc(values*4u);
        ok=ok&&raw&&dense&&cooked&&q8&&expected&&got;
        if(ok){
            for(uint64_t b=0;b<values/32u;b++){
                uint16_t d=fg_f32_to_f16(((int)(b%7u)-3)*0.03125f);
                memcpy(raw+b*18u,&d,2u);
                for(uint32_t j=0;j<16u;j++)raw[b*18u+2u+j]=(uint8_t)(b*37u+j*13u);
            }
            fg_dequantize_iq4_nl(raw,expected,values);
            ok=fg_cook_dense_compact(raw,dense,dense_bytes,k,rows,20u);
            for(uint32_t layout=0;ok&&layout<2u;layout++){
                ok=!fg_expand_iq4_q8_cooked(raw,cooked,cooked_bytes-1u,k,rows,false)&&
                    fg_expand_iq4_q8_cooked(layout?dense:raw,cooked,cooked_bytes,k,rows,layout!=0u)&&
                    fg_uncook_q8_0_rows(cooked,q8,values/32u*34u,k,rows);
                if(ok){fg_dequantize_q8_0(q8,got,values);ok=!memcmp(expected,got,values*4u);}
            }
        }
        free(got);free(expected);free(q8);free(cooked);free(dense);free(raw);
    }
    return ok;
}

int main(void){
    const uint32_t types[]={12u,13u,20u};
    int ok=lossless_iq4_expansion();
    ok=ok&&tensor_family("blk.0.attn_qkv.weight")==0u&&
        tensor_family("blk.0.ssm_out.weight")==1u&&
        tensor_family("blk.0.attn_gate.weight")==2u&&
        tensor_family("blk.3.attn_output.weight")==3u&&
        tensor_family("blk.0.ffn_gate_shexp.weight")==4u&&
        tensor_family("blk.0.ffn_up_shexp.weight")==5u&&
        tensor_family("blk.0.ffn_down_shexp.weight")==6u&&
        tensor_family("blk.0.hc_attn_down.weight")==7u&&
        tensor_family("output_hc_up.weight")==7u&&
        tensor_family("blk.1.ple_key.weight")==8u;
    for(uint32_t i=0u;i<3u;i++){
        ok=ok&&fg_vk_dense_saved_format(types[i],FG_TENSOR_LAYOUT_DENSE_SOA)==
            (fg_vk_tensor_format)(FG_VK_TENSOR_FORMAT_DENSE_SOA_Q4_K+i);
        ok=ok&&fg_vk_dense_saved_format(types[i],FG_TENSOR_LAYOUT_DENSE_COMPACT)==
            (fg_vk_tensor_format)(FG_VK_TENSOR_FORMAT_DENSE_COMPACT_Q4_K+i);
        ok=ok&&fg_vk_dense_saved_format(types[i],FG_TENSOR_LAYOUT_GGML)==
            fg_vk_dense_format(types[i]);
    }
    ok=ok&&fg_vk_dense_saved_format(8u,FG_TENSOR_LAYOUT_Q8_0_COOKED)==
        FG_VK_TENSOR_FORMAT_Q8_0_COOKED;
    ok=ok&&fg_vk_dense_saved_format(30u,FG_TENSOR_LAYOUT_GGML)==
        FG_VK_TENSOR_FORMAT_BF16_ROUTER;
    fg_manifest topology={0};topology.layer_owner[FG_LAYER_COUNT-1u]=7u;
    for(int owner=0;owner<8;owner++){
        fg_tensor_record global={.rank=4u,.kind=FG_TENSOR_COMMON,.layer=UINT16_MAX};
        ok=ok&&(owner_record(&topology,&global,owner)==
            (owner==0||owner==1||owner==2||owner==4||owner==7));
        global.layer=3u;
        ok=ok&&(owner_record(&topology,&global,owner)==(owner==4));
        global.rank=0u;global.layer=UINT16_MAX;
        ok=ok&&(owner_record(&topology,&global,owner)==(owner==0));
        ok=ok&&owner_record(&topology,&global,-1);
    }
    for(uint32_t i=0;ok&&i<4u;i++)for(uint32_t compact=0;ok&&compact<(i==3u?1u:2u);compact++){
        uint32_t type=i==3u?8u:types[i],block=(type==20u||type==8u)?32u:256u;
        uint32_t bs=type==8u?34u:type==12u?144u:type==13u?176u:18u;
        fg_gguf_tensor t={.type=type,.dims=2u,.shape={512u,17u}};
        t.bytes=(512u/block)*bs*17u;
        uint64_t bytes=type==8u?fg_q8_0_cooked_matrix_bytes(512u,17u):
            compact?fg_dense_compact_bytes(512u,17u,type):fg_dense_soa_bytes(512u,17u,type);
        uint8_t *raw=calloc(1u,(size_t)t.bytes),*expected=malloc((size_t)bytes),*got=malloc((size_t)bytes);
        FILE *source=fopen(".q4-repack-source.bin","wb+x"),*out=fopen(".q4-repack-cooked.bin","wb+x");
        ok=raw&&expected&&got&&source&&out;
        if(ok){
            for(uint64_t j=0;j<t.bytes;j++)raw[j]=(uint8_t)(j*37u+13u);
            for(uint64_t j=0;j<t.bytes;j+=bs){uint16_t d=fg_f32_to_f16(0.001f);memcpy(raw+j,&d,2u);if(type!=20u)memcpy(raw+j+2u,&d,2u);}
            uint8_t hash[32],actual[32];fg_error error={0};
            ok=fwrite(raw,1u,(size_t)t.bytes,source)==t.bytes&&fflush(source)==0&&
                cook_tensor(source,&t,out,compact!=0u,bytes,hash)&&fflush(out)==0&&
                fseeko(out,0,SEEK_SET)==0&&fread(got,1u,(size_t)bytes,out)==bytes&&
                (type==8u?fg_cook_q8_0_rows(raw,expected,bytes,512u,17u):
                 compact?fg_cook_dense_compact(raw,expected,bytes,512u,17u,type):
                    fg_cook_dense_soa(raw,expected,bytes,512u,17u,type))&&
                memcmp(got,expected,(size_t)bytes)==0&&
                fg_sha256_file(".q4-repack-cooked.bin",actual,&error)==FG_OK&&memcmp(hash,actual,32u)==0;
            if(ok&&type==8u){
                memset(expected,0,(size_t)bytes);
                ok=!fg_uncook_q8_0_rows(got,expected,t.bytes-1u,512u,17u)&&
                    fg_uncook_q8_0_rows(got,expected,t.bytes,512u,17u)&&
                    !memcmp(raw,expected,(size_t)t.bytes);
            }
        }
        if(out){fclose(out);unlink(".q4-repack-cooked.bin");}
        if(source){fclose(source);unlink(".q4-repack-source.bin");}
        free(got);free(expected);free(raw);
    }
    if(ok){
        uint8_t data[8192],got[8192],hash[32],expected[32];
        for(size_t i=0;i<sizeof(data);i++)data[i]=(uint8_t)(i*13u);
        FILE *source=fopen(".q4-repack-source.bin","wb+x"),*out=fopen(".q4-repack-cooked.bin","wb+x");
        ok=source&&out;
        fg_sha256 digest;fg_sha256_init(&digest);fg_sha256_update(&digest,data,sizeof(data));fg_sha256_final(&digest,expected);
        use_reflink=true;
        if(ok)ok=fwrite(data,1u,sizeof(data),source)==sizeof(data)&&fflush(source)==0&&
            copy_tensor(source,0u,sizeof(data),out,hash)&&fflush(out)==0&&fseeko(out,0,SEEK_SET)==0&&
            fread(got,1u,sizeof(got),out)==sizeof(got)&&!memcmp(data,got,sizeof(data))&&!memcmp(hash,expected,32u);
        if(out){fclose(out);unlink(".q4-repack-cooked.bin");}
        if(source){fclose(source);unlink(".q4-repack-source.bin");}
        printf("Q4 reflink/fallback copied_bytes=%llu\n",(unsigned long long)reflink_bytes);
    }
    puts(ok?"Q4 cooked repack bytes/hashes: PASS":"Q4 cooked repack bytes/hashes: FAIL");
    return ok?0:1;
}
