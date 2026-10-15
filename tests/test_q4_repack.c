#define main q4_repack_cli
#include "../tools/q4_repack.c"
#undef main

int main(void){
    const uint32_t types[]={12u,13u,20u};
    int ok=1;
    for(uint32_t i=0;ok&&i<3u;i++)for(uint32_t compact=0;ok&&compact<2u;compact++){
        uint32_t type=types[i],block=type==20u?32u:256u,bs=type==12u?144u:type==13u?176u:18u;
        fg_gguf_tensor t={.type=type,.dims=2u,.shape={512u,17u}};
        t.bytes=(512u/block)*bs*17u;
        uint64_t bytes=compact?fg_dense_compact_bytes(512u,17u,type):fg_dense_soa_bytes(512u,17u,type);
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
                (compact?fg_cook_dense_compact(raw,expected,bytes,512u,17u,type):
                    fg_cook_dense_soa(raw,expected,bytes,512u,17u,type))&&
                memcmp(got,expected,(size_t)bytes)==0&&
                fg_sha256_file(".q4-repack-cooked.bin",actual,&error)==FG_OK&&memcmp(hash,actual,32u)==0;
        }
        if(out){fclose(out);unlink(".q4-repack-cooked.bin");}
        if(source){fclose(source);unlink(".q4-repack-source.bin");}
        free(got);free(expected);free(raw);
    }
    puts(ok?"Q4 cooked repack bytes/hashes: PASS":"Q4 cooked repack bytes/hashes: FAIL");
    return ok?0:1;
}
