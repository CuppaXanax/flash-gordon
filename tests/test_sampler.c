#include "fg_sampler.h"

#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)

int main(void){
    fg_sampler_config config;fg_sampler_config_defaults(&config);
    fg_error err={0};CHECK(fg_sampler_config_validate(&config,&err)==FG_OK);
    config.top_k=65u;CHECK(fg_sampler_config_validate(&config,&err)==FG_ERR_ARGUMENT);
    fg_sampler_config_greedy(&config);CHECK(config.temperature==0.0f);
    const float scores[]={5.0f,4.0f,3.0f};const uint32_t ids[]={9u,3u,1u};
    fg_sampler_state a,b;fg_sampler_state_init(&a,42u);fg_sampler_state_init(&b,42u);
    for(uint32_t i=0;i<4u;i++)CHECK(fg_sampler_uniform(&a)==fg_sampler_uniform(&b));
    config.temperature=1.0f;config.top_p=1.0f;config.top_k=2u;
    uint32_t x=0,y=0;float lx=0,ly=0;
    CHECK(fg_sampler_select(&config,scores,ids,3u,.3f,&x,&lx,&err)==FG_OK);
    CHECK(fg_sampler_select(&config,scores,ids,3u,.3f,&y,&ly,&err)==FG_OK);
    CHECK(x==y&&x!=1u);
    config.temperature=1.0f;config.top_p=1.0f;config.top_k=3u;
    CHECK(fg_sampler_select(&config,scores,ids,3u,.8f,&x,&lx,&err)==FG_OK&&x==3u);
    config.temperature=.1f;
    CHECK(fg_sampler_select(&config,scores,ids,3u,.8f,&y,&ly,&err)==FG_OK&&y==9u);
    config.top_k=3u;config.top_p=.5f;
    CHECK(fg_sampler_select(&config,scores,ids,3u,.99f,&x,&lx,&err)==FG_OK&&x==9u);
    config.top_p=1.0f;config.temperature=.01f;
    CHECK(fg_sampler_select(&config,scores,ids,3u,.99f,&x,&lx,&err)==FG_OK&&x==9u);
    const float tied[]={1.0f,1.0f};const uint32_t tied_ids[]={8u,2u};
    config.temperature=0.0f;config.top_k=1u;
    CHECK(fg_sampler_select(&config,tied,tied_ids,2u,0.5f,&x,&lx,&err)==FG_OK&&x==2u);
    return failures?1:0;
}
