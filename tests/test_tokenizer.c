#include "fg_sha256.h"
#include "fg_tokenizer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TOKENIZER_MAGIC UINT64_C(0x314e4b4f544746)

static int write_u32(FILE *file,uint32_t value){return fwrite(&value,1,4u,file)==4u;}
static int write_u64(FILE *file,uint64_t value){return fwrite(&value,1,8u,file)==8u;}
static uint32_t byte_codepoint(uint8_t byte){if((byte>=33u&&byte<=126u)||(byte>=161u&&byte<=172u)||byte>=174u)return byte;uint32_t ordinal=0;for(uint32_t value=0;value<256u;value++){if((value>=33u&&value<=126u)||(value>=161u&&value<=172u)||value>=174u)continue;if(value==byte)return 256u+ordinal;ordinal++;}return byte;}
static size_t encode_utf8(uint32_t code,char output[4]){if(code<=0x7fu){output[0]=(char)code;return 1u;}if(code<=0x7ffu){output[0]=(char)(0xc0u|(code>>6u));output[1]=(char)(0x80u|(code&63u));return 2u;}output[0]=(char)(0xe0u|(code>>12u));output[1]=(char)(0x80u|((code>>6u)&63u));output[2]=(char)(0x80u|(code&63u));return 3u;}

static int test_real_qwen38_artifact(const char *source){
    char directory[128],tokenizer_directory[160],path[200];
    snprintf(directory,sizeof(directory),"/tmp/fg-q38-tokenizer-%ld",(long)getpid());
    snprintf(tokenizer_directory,sizeof(tokenizer_directory),"%s/tokenizer",directory);
    snprintf(path,sizeof(path),"%s/tokenizer.fgt",tokenizer_directory);
    if(mkdir(directory,0700)!=0){perror("create real-tokenizer test directory");return 1;}
    fg_manifest *manifest=malloc(sizeof(*manifest));
    fg_error error={0};
    fg_tokenizer *tokenizer=NULL;
    int ok=manifest!=NULL;
    if(ok){fg_manifest_init(manifest);ok=fg_tokenizer_pack_gguf(source,directory,manifest,&error)==FG_OK;}
    if(ok)ok=fg_tokenizer_open(&tokenizer,directory,manifest,&error)==FG_OK;
    if(ok)ok=fg_tokenizer_validate_qwen38(tokenizer,&error)==FG_OK;
    fg_tokenizer_close(tokenizer);
    free(manifest);
    unlink(path);rmdir(tokenizer_directory);rmdir(directory);
    if(!ok){fprintf(stderr,"real Qwen3.8 tokenizer parity failed: %s\n",error.message);return 1;}
    puts("Flash Gordon real Unsloth Qwen3.8 tokenizer parity: PASS");
    return 0;
}

int main(int argc,char **argv){
    if(argc==2)return test_real_qwen38_artifact(argv[1]);
    if(argc!=1){fprintf(stderr,"usage: %s [Qwen3.8 GGUF tokenizer shard]\n",argv[0]);return 2;}
    char directory[128],tokenizer_directory[160],path[200];snprintf(directory,sizeof(directory),"/tmp/fg-tokenizer-%ld",(long)getpid());snprintf(tokenizer_directory,sizeof(tokenizer_directory),"%s/tokenizer",directory);snprintf(path,sizeof(path),"%s/tokenizer.fgt",tokenizer_directory);mkdir(directory,0700);mkdir(tokenizer_directory,0700);
    FILE *file=fopen(path,"wb");if(!file)return 1;int ok=write_u64(file,TOKENIZER_MAGIC)&&write_u32(file,1u)&&write_u32(file,248320u)&&write_u32(file,1u)&&write_u32(file,257u)&&write_u32(file,258u)&&write_u32(file,1u);
    char text[32];for(uint32_t i=0;ok&&i<248320u;i++){size_t length;if(i<256u)length=encode_utf8(byte_codepoint((uint8_t)i),text);else if(i==256u){memcpy(text,"<|special|>",11u);length=11u;}else{int result=snprintf(text,sizeof(text),"t%06u",i);length=result>0?(size_t)result:0u;}ok=length>0&&write_u32(file,(uint32_t)length)&&fwrite(text,1,length,file)==length&&write_u32(file,i==256u?3u:1u);}
    const char merge[]="t000000 t000001";if(ok)ok=write_u32(file,sizeof(merge)-1u)&&fwrite(merge,1,sizeof(merge)-1u,file)==sizeof(merge)-1u;long end=ftell(file);if(end<0)ok=0;uint64_t padded=fg_align_up_u64((uint64_t)end,FG_ALIGNMENT);while(ok&&(uint64_t)end<padded){fputc(0,file);end++;}if(fclose(file)!=0)ok=0;
    fg_manifest manifest={0};manifest.tensor_count=1u;fg_tensor_record *record=&manifest.tensors[0];snprintf(record->name,sizeof(record->name),"tokenizer/tokenizer.fgt");record->bytes=padded;record->dims=1u;record->shape[0]=padded;record->rank=UINT16_MAX;record->layer=UINT16_MAX;record->expert=UINT16_MAX;record->kind=FG_TENSOR_TOKENIZER;fg_error error={0};if(ok)ok=fg_sha256_file(path,record->sha256,&error)==FG_OK;
    fg_tokenizer *tokenizer=NULL;fg_status status=ok?fg_tokenizer_open(&tokenizer,directory,&manifest,&error):FG_ERR_IO;if(status==FG_ERR_UNAVAILABLE){fprintf(stderr,"SKIP direct tokenizer load: %s\n",error.message);unlink(path);rmdir(tokenizer_directory);rmdir(directory);return 77;}if(status!=FG_OK){fprintf(stderr,"tokenizer open: %s\n",error.message);ok=0;}
    uint32_t token=0;const char *piece=NULL;size_t bytes=0;uint32_t type=0;if(ok)ok=fg_tokenizer_vocab_size(tokenizer)==248320u&&fg_tokenizer_bos(tokenizer)==257u&&fg_tokenizer_eos(tokenizer)==258u&&fg_tokenizer_add_bos(tokenizer)&&fg_tokenizer_lookup(tokenizer,"t123456",7u,&token,&error)==FG_OK&&token==123456u&&fg_tokenizer_token(tokenizer,token,&piece,&bytes,&type,&error)==FG_OK&&bytes==7u&&memcmp(piece,"t123456",7u)==0&&type==1u;
    fg_tokens tokens={0};if(ok)ok=fg_tokenizer_encode(tokenizer,"hello",false,&tokens,&error)==FG_OK&&tokens.count==6u&&tokens.data[0]==257u&&tokens.data[1]==(uint32_t)'h'&&tokens.data[2]==(uint32_t)'e'&&tokens.data[3]==(uint32_t)'l'&&tokens.data[4]==(uint32_t)'l'&&tokens.data[5]==(uint32_t)'o';char decoded[8];size_t decoded_bytes=0;if(ok)ok=fg_tokenizer_decode_token(tokenizer,(uint32_t)'h',decoded,sizeof(decoded),&decoded_bytes,&error)==FG_OK&&decoded_bytes==1u&&decoded[0]=='h';fg_tokens_free(&tokens);if(ok)ok=fg_tokenizer_encode(tokenizer,"<|special|>",true,&tokens,&error)==FG_OK&&tokens.count==2u&&tokens.data[0]==257u&&tokens.data[1]==256u;fg_tokens_free(&tokens);fg_tokenizer_close(tokenizer);
    int fd=open(path,O_RDWR|O_CLOEXEC);uint8_t byte;if(ok&&fd>=0&&pread(fd,&byte,1u,64)==1){byte^=1u;ok=pwrite(fd,&byte,1u,64)==1;}else ok=0;if(fd>=0)close(fd);tokenizer=NULL;if(ok)ok=fg_tokenizer_open(&tokenizer,directory,&manifest,&error)==FG_ERR_MISMATCH;fg_tokenizer_close(tokenizer);
    unlink(path);rmdir(tokenizer_directory);rmdir(directory);if(!ok){fprintf(stderr,"tokenizer direct-load/seal test failed: %s\n",error.message);return 1;}puts("Flash Gordon sealed direct-I/O tokenizer load: PASS");return 0;
}
