#include "fg_runtime.h"

#include <stdio.h>

typedef struct continuation_writer {FILE *file;uint32_t prompt,step;} continuation_writer;

static fg_status discard(void *context,uint32_t token,const char *text,size_t bytes,fg_error *err){
    (void)context;(void)token;(void)text;(void)bytes;(void)err;return FG_OK;
}

static fg_status continuation(void *context,uint32_t token,const char *text,size_t bytes,fg_error *err){
    continuation_writer *writer=context;
    if(bytes>UINT32_MAX)return FG_ERR_LIMIT;
    uint32_t fields[]={writer->prompt,writer->step++,token,(uint32_t)bytes};
    uint8_t header[16];
    for(uint32_t i=0;i<4u;i++)for(uint32_t j=0;j<4u;j++)header[i*4u+j]=(uint8_t)(fields[i]>>(j*8u));
    if(fwrite(header,1u,sizeof(header),writer->file)!=sizeof(header)||
       fwrite(text,1u,bytes,writer->file)!=bytes){
        fg_error_set(err,FG_ERR_IO,"write frozen teacher continuation");return FG_ERR_IO;
    }
    return FG_OK;
}

int main(int argc,char **argv){
    if(argc<5||strcmp(argv[1],"--manifest")||strcmp(argv[3],"--prompts")){
        fprintf(stderr,"usage: q4-eval-batch --manifest FILE --prompts PROMPTS.fgev [--continuations FILE --steps N]\n");return 2;
    }
    const char *continuation_path=NULL;uint32_t steps=1u;
    for(int i=5;i<argc;i+=2){
        if(i+1==argc)return 2;
        if(!strcmp(argv[i],"--continuations"))continuation_path=argv[i+1];
        else if(!strcmp(argv[i],"--steps")){
            char *end=NULL;unsigned long value=strtoul(argv[i+1],&end,10);
            if(!end||*end||value<1u||value>256u)return 2;
            steps=(uint32_t)value;
        }else return 2;
    }
    if(steps>1u&&!continuation_path)return 2;
    FILE *file=fopen(argv[4],"rb");
    char header[8];char *prompts[1024]={0};uint32_t count=0u;
    int valid=file&&fread(header,1u,8u,file)==8u&&!memcmp(header,"FGEVAL01",8u);
    while(valid){
        uint8_t size[4]={0};size_t n=fread(size,1u,4u,file);
        if(!n&&feof(file))break;
        uint32_t bytes=(uint32_t)size[0]|((uint32_t)size[1]<<8u)|
            ((uint32_t)size[2]<<16u)|((uint32_t)size[3]<<24u);
        if(n!=4u||!bytes||bytes>(1u<<20u)||count==1024u){valid=0;break;}
        char *prompt=malloc((size_t)bytes+1u);
        if(!prompt){valid=0;break;}
        if(fread(prompt,1u,bytes,file)!=bytes||memchr(prompt,0,bytes)){
            free(prompt);valid=0;break;
        }
        prompt[bytes]=0;prompts[count++]=prompt;
    }
    if(file)fclose(file);
    continuation_writer writer={0};
    if(valid&&continuation_path){
        writer.file=fopen(continuation_path,"wbx");
        valid=writer.file&&fwrite("FGCONT01",1u,8u,writer.file)==8u;
    }
    fg_error error={0};fg_runtime *runtime=NULL;
    fg_status status=valid&&count?fg_runtime_open(&runtime,argv[2],&error):FG_ERR_FORMAT;
    fg_sampler_config sampler;fg_sampler_config_greedy(&sampler);
    for(uint32_t i=0;status==FG_OK&&i<count;i++){
        status=fg_runtime_reset(runtime,&error);
        if(status==FG_OK)status=fg_runtime_set_sampler(runtime,&sampler,&error);
        fg_generation_stats stats={0};
        writer.prompt=i;writer.step=0u;
        /* Normal captures discard predictions; only corpus construction follows the teacher. */
        if(status==FG_OK)status=fg_runtime_generate(runtime,prompts[i],steps,
            writer.file?continuation:discard,writer.file?&writer:NULL,NULL,NULL,&stats,&error);
        if(status==FG_OK){printf("EVAL_CAPTURE index=%u prompt_tokens=%u generated=%u\n",
            i,stats.prompt_tokens,stats.generated_tokens);fflush(stdout);}
    }
    fg_runtime_close(runtime);
    if(writer.file&&fclose(writer.file)!=0&&status==FG_OK)status=FG_ERR_IO;
    for(uint32_t i=0;i<count;i++)free(prompts[i]);
    if(status!=FG_OK)fprintf(stderr,"batch evaluation failed: %s\n",error.message);
    return status==FG_OK?0:1;
}
