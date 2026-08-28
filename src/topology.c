#include "fg_topology.h"

#include <stdlib.h>
#include <string.h>

void fg_topology_build(fg_manifest *m) {
    static const uint8_t delta[FG_GROUP_SIZE]={0,1,3,5};
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        m->layer_owner[l]=(uint8_t)(l%FG_RANK_COUNT);
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++)m->layer_groups[l][g]=(uint8_t)((l+delta[g])%FG_RANK_COUNT);
    }
}
bool fg_topology_rank_in_layer(const fg_manifest *m,uint32_t layer,uint32_t rank){
    if(layer>=FG_LAYER_COUNT||rank>=FG_RANK_COUNT)return false;
    for(uint32_t i=0;i<FG_GROUP_SIZE;i++)if(m->layer_groups[layer][i]==rank)return true;
    return false;
}

fg_status fg_topology_assign_round_robin(fg_manifest *m,fg_error *err){
    (void)err;
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++)for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)m->expert_rank[l][e]=m->layer_groups[l][e%FG_GROUP_SIZE];
    return FG_OK;
}

typedef struct expert_score{double frequency;uint16_t expert;}expert_score;
static int score_cmp(const void *a,const void *b){const expert_score *x=a,*y=b;return x->frequency<y->frequency?1:x->frequency>y->frequency?-1:(int)x->expert-(int)y->expert;}

fg_status fg_topology_assign_profile(fg_manifest *m,const double frequency[FG_LAYER_COUNT][FG_EXPERT_COUNT],fg_error *err){
    if(!frequency){fg_error_set(err,FG_ERR_ARGUMENT,"router profile is null");return FG_ERR_ARGUMENT;}
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        expert_score scores[FG_EXPERT_COUNT];double load[FG_GROUP_SIZE]={0};uint16_t count[FG_GROUP_SIZE]={0};
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++){scores[e].frequency=frequency[l][e];scores[e].expert=(uint16_t)e;}
        qsort(scores,FG_EXPERT_COUNT,sizeof(scores[0]),score_cmp);
        for(uint32_t i=0;i<FG_EXPERT_COUNT;i++){
            uint32_t best=FG_GROUP_SIZE;
            for(uint32_t g=0;g<FG_GROUP_SIZE;g++)if(count[g]<FG_EXPERTS_PER_RANK&&(best==FG_GROUP_SIZE||load[g]<load[best]||(load[g]==load[best]&&g<best)))best=g;
            if(best==FG_GROUP_SIZE){fg_error_set(err,FG_ERR_FORMAT,"layer %u profile violated equal residency",l);return FG_ERR_FORMAT;}
            m->expert_rank[l][scores[i].expert]=m->layer_groups[l][best];count[best]++;load[best]+=scores[i].frequency;
        }
    }
    return FG_OK;
}
