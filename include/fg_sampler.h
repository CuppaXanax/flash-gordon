#ifndef FLASH_GORDON_SAMPLER_H
#define FLASH_GORDON_SAMPLER_H

#include "fg.h"

typedef struct fg_sampler_config {
    float temperature;
    float top_p;
    uint32_t top_k;
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty;
    float min_p;
    uint64_t seed;
} fg_sampler_config;

typedef struct fg_sampler_state {
    uint64_t state;
} fg_sampler_state;

void fg_sampler_config_greedy(fg_sampler_config *config);
void fg_sampler_config_defaults(fg_sampler_config *config);
fg_status fg_sampler_config_validate(const fg_sampler_config *config, fg_error *err);
void fg_sampler_state_init(fg_sampler_state *state, uint64_t seed);
float fg_sampler_uniform(fg_sampler_state *state);
fg_status fg_sampler_select(const fg_sampler_config *config,
                            const float *scores,const uint32_t *ids,uint32_t count,
                            float uniform,uint32_t *token,float *logit,fg_error *err);
bool fg_sampler_penalties_active(const fg_sampler_config *config);
void fg_sampler_apply_penalties(float *scores,const uint32_t *ids,
                                const uint32_t *counts,uint32_t count,
                                const fg_sampler_config *config);

#endif
