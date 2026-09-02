#include "fg_sampler.h"

#include <math.h>
#include <string.h>

static const uint64_t FG_SAMPLER_DEFAULT_SEED = UINT64_C(0x9e3779b97f4a7c15);

void fg_sampler_config_greedy(fg_sampler_config *config){
    if(!config)return;
    config->temperature=0.0f;config->top_p=1.0f;config->top_k=1u;
    config->presence_penalty=0.0f;config->frequency_penalty=0.0f;
    config->repetition_penalty=1.0f;config->min_p=0.0f;
    config->seed=FG_SAMPLER_DEFAULT_SEED;
}

void fg_sampler_config_defaults(fg_sampler_config *config){
    if(!config)return;
    config->temperature=1.0f;config->top_p=0.95f;config->top_k=20u;
    config->presence_penalty=0.0f;config->frequency_penalty=0.0f;
    config->repetition_penalty=1.0f;config->min_p=0.0f;
    config->seed=FG_SAMPLER_DEFAULT_SEED;
}

fg_status fg_sampler_config_validate(const fg_sampler_config *config,fg_error *err){
    if(!config||!isfinite(config->temperature)||config->temperature<0.0f||
       !isfinite(config->top_p)||config->top_p<=0.0f||config->top_p>1.0f||
       !config->top_k||config->top_k>64u||
       !isfinite(config->presence_penalty)||config->presence_penalty < -2.0f||
       config->presence_penalty > 2.0f||
       !isfinite(config->frequency_penalty)||config->frequency_penalty < -2.0f||
       config->frequency_penalty > 2.0f||
       !isfinite(config->repetition_penalty)||config->repetition_penalty<=0.0f||
       !isfinite(config->min_p)||config->min_p!=0.0f){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid sampler controls");
        return FG_ERR_ARGUMENT;
    }
    return FG_OK;
}

bool fg_sampler_penalties_active(const fg_sampler_config *config){
    return config&&(
        config->presence_penalty!=0.0f||config->frequency_penalty!=0.0f||
        config->repetition_penalty!=1.0f);
}

void fg_sampler_apply_penalties(float *scores,const uint32_t *ids,
                                const uint32_t *counts,uint32_t count,
                                const fg_sampler_config *config){
    if(!scores||!ids||!counts||!config)return;
    for(uint32_t i=0;i<count;i++){
        uint32_t seen=counts[ids[i]];
        if(!seen)continue;
        if(config->repetition_penalty!=1.0f)
            scores[i]=scores[i]>0.0f?
                scores[i]/config->repetition_penalty:
                scores[i]*config->repetition_penalty;
        scores[i]-=config->frequency_penalty*(float)seen+
            config->presence_penalty;
    }
}

void fg_sampler_state_init(fg_sampler_state *state,uint64_t seed){
    if(!state)return;
    state->state=seed?seed:FG_SAMPLER_DEFAULT_SEED;
}

float fg_sampler_uniform(fg_sampler_state *state){
    /* xorshift64* is deterministic, local to one request, and never global. */
    uint64_t x=state->state;
    x^=x>>12u;x^=x<<25u;x^=x>>27u;state->state=x;
    uint64_t value=x*UINT64_C(2685821657736338717);
    return (float)((value>>40u)*(1.0/16777216.0));
}

static bool precedes(float a,uint32_t aid,float b,uint32_t bid){
    if(a!=b)return a>b;
    return aid<bid;
}

fg_status fg_sampler_select(const fg_sampler_config *config,
                            const float *scores,const uint32_t *ids,uint32_t count,
                            float uniform,uint32_t *token,float *logit,fg_error *err){
    if(fg_sampler_config_validate(config,err)!=FG_OK||!scores||!ids||
       !count||count>64u||!token||!logit){
        if(err&&!err->message[0])fg_error_set(err,FG_ERR_ARGUMENT,"invalid sampler candidates");
        return err&&err->code?err->code:FG_ERR_ARGUMENT;
    }
    uint32_t order[64],kept=0u;
    for(uint32_t i=0;i<count;i++){
        if(!isfinite(scores[i]))continue;
        uint32_t at=kept;
        while(at&&precedes(scores[i],ids[i],scores[order[at-1u]],ids[order[at-1u]])){
            order[at]=order[at-1u];at--;
        }
        order[at]=i;kept++;
    }
    if(!kept){fg_error_set(err,FG_ERR_FORMAT,"sampler received no finite candidates");return FG_ERR_FORMAT;}
    if(config->temperature==0.0f){*token=ids[order[0]];*logit=scores[order[0]];return FG_OK;}
    uint32_t limit=config->top_k<kept?config->top_k:kept;
    double max_raw=scores[order[0]],sum=0.0;
    for(uint32_t i=0;i<limit;i++)sum+=exp((double)scores[order[i]]-max_raw);
    if(!(sum>0.0)||!isfinite(sum)){fg_error_set(err,FG_ERR_FORMAT,"sampler raw softmax failed");return FG_ERR_FORMAT;}
    double cumulative=0.0;uint32_t cutoff=limit;
    for(uint32_t i=0;i<limit;i++){
        cumulative+=exp((double)scores[order[i]]-max_raw)/sum;
        if(cumulative>=config->top_p){cutoff=i+1u;break;}
    }
    double max_temp=(double)scores[order[0]]/(double)config->temperature;
    double temp_sum=0.0;
    for(uint32_t i=0;i<cutoff;i++)temp_sum+=exp((double)scores[order[i]]/(double)config->temperature-max_temp);
    if(!(temp_sum>0.0)||!isfinite(temp_sum)){fg_error_set(err,FG_ERR_FORMAT,"sampler temperature softmax failed");return FG_ERR_FORMAT;}
    if(!isfinite(uniform)||uniform<0.0f)uniform=0.0f;
    if(uniform>=1.0f)uniform=nextafterf(1.0f,0.0f);
    double draw=(double)uniform*temp_sum,chosen=0.0;
    uint32_t selected=cutoff-1u;
    for(uint32_t i=0;i<cutoff;i++){
        chosen+=exp((double)scores[order[i]]/(double)config->temperature-max_temp);
        if(draw<chosen){selected=i;break;}
    }
    *token=ids[order[selected]];*logit=scores[order[selected]];
    return FG_OK;
}
