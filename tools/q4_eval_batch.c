#include "fg_runtime.h"

#include <stdio.h>

static fg_status discard(void *context,uint32_t token,const char *text,size_t bytes,fg_error *err){
    (void)context;(void)token;(void)text;(void)bytes;(void)err;return FG_OK;
}

int main(int argc,char **argv){
    if(argc!=5||strcmp(argv[1],"--manifest")||strcmp(argv[3],"--prompts")){
        fprintf(stderr,"usage: q4-eval-batch --manifest FILE --prompts PROMPTS.fgev\n");return 2;
    }
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
    fg_error error={0};fg_runtime *runtime=NULL;
    fg_status status=valid&&count?fg_runtime_open(&runtime,argv[2],&error):FG_ERR_FORMAT;
    fg_sampler_config sampler;fg_sampler_config_greedy(&sampler);
    for(uint32_t i=0;status==FG_OK&&i<count;i++){
        status=fg_runtime_reset(runtime,&error);
        if(status==FG_OK)status=fg_runtime_set_sampler(runtime,&sampler,&error);
        fg_generation_stats stats={0};
        /* One frontier only; reset discards its prediction before the next row. */
        if(status==FG_OK)status=fg_runtime_generate(runtime,prompts[i],1u,discard,
            NULL,NULL,NULL,&stats,&error);
        if(status==FG_OK){printf("EVAL_CAPTURE index=%u prompt_tokens=%u generated=%u\n",
            i,stats.prompt_tokens,stats.generated_tokens);fflush(stdout);}
    }
    fg_runtime_close(runtime);
    for(uint32_t i=0;i<count;i++)free(prompts[i]);
    if(status!=FG_OK)fprintf(stderr,"batch evaluation failed: %s\n",error.message);
    return status==FG_OK?0:1;
}
