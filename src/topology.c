#include "fg_topology.h"

#include <errno.h>
#include <stdio.h>
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

fg_status fg_topology_assign_map(fg_manifest *m,const uint16_t expert_rank[FG_LAYER_COUNT][FG_EXPERT_COUNT],fg_error *err){
    if(!m||!expert_rank){fg_error_set(err,FG_ERR_ARGUMENT,"expert map is null");return FG_ERR_ARGUMENT;}
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        uint16_t count[FG_RANK_COUNT]={0};
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++){
            uint32_t rank=expert_rank[l][e];
            if(rank>=FG_RANK_COUNT||!fg_topology_rank_in_layer(m,l,rank)){fg_error_set(err,FG_ERR_FORMAT,"layer %u expert %u assigned to rank %u outside its group",l,e,rank);return FG_ERR_FORMAT;}
            count[rank]++;
        }
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++){uint32_t rank=m->layer_groups[l][g];if(count[rank]!=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_FORMAT,"layer %u rank %u owns %u experts, expected %u",l,rank,count[rank],FG_EXPERTS_PER_RANK);return FG_ERR_FORMAT;}}
    }
    memcpy(m->expert_rank,expert_rank,sizeof(m->expert_rank));
    return FG_OK;
}

static bool map_space(char value){return value==' '||value=='\t'||value=='\r'||value=='\n';}

fg_status fg_topology_assign_map_file(fg_manifest *m,const char *path,fg_error *err){
    if(!m||!path){fg_error_set(err,FG_ERR_ARGUMENT,"expert map path is null");return FG_ERR_ARGUMENT;}FILE *file=fopen(path,"r");if(!file){fg_error_set(err,FG_ERR_IO,"open expert map %s: %s",path,strerror(errno));return FG_ERR_IO;}uint16_t (*expert_rank)[FG_EXPERT_COUNT]=calloc(FG_LAYER_COUNT,sizeof(*expert_rank));if(!expert_rank){fclose(file);fg_error_set(err,FG_ERR_OOM,"allocate expert map");return FG_ERR_OOM;}char *line=NULL;size_t capacity=0;ssize_t bytes;bool seen[FG_LAYER_COUNT]={0};uint32_t line_number=0;fg_status status=FG_OK;
    while((bytes=getline(&line,&capacity,file))>=0){(void)bytes;line_number++;char *cursor=line;while(map_space(*cursor))cursor++;if(!*cursor||*cursor=='#')continue;if(strncmp(cursor,"layer=",6u)!=0){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u is malformed",line_number);status=FG_ERR_FORMAT;break;}char *end=NULL;errno=0;unsigned long layer=strtoul(cursor+6u,&end,10);if(errno||end==cursor+6u||layer>=FG_LAYER_COUNT||strncmp(end," ranks=",7u)!=0||seen[layer]){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u has an invalid or duplicate layer",line_number);status=FG_ERR_FORMAT;break;}cursor=end+7u;for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++){errno=0;unsigned long rank=strtoul(cursor,&end,10);if(errno||end==cursor||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u has an invalid rank for expert %u",line_number,expert);status=FG_ERR_FORMAT;break;}expert_rank[layer][expert]=(uint16_t)rank;if(expert+1u<FG_EXPERT_COUNT){if(*end!=','){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u ends before expert %u",line_number,expert+1u);status=FG_ERR_FORMAT;break;}cursor=end+1u;}else{while(map_space(*end))end++;if(*end){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u has trailing data",line_number);status=FG_ERR_FORMAT;}}}if(status!=FG_OK)break;seen[layer]=true;}
    if(status==FG_OK&&ferror(file)){fg_error_set(err,FG_ERR_IO,"read expert map %s: %s",path,strerror(errno));status=FG_ERR_IO;}if(status==FG_OK)for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)if(!seen[layer]){fg_error_set(err,FG_ERR_FORMAT,"expert map is missing layer %u",layer);status=FG_ERR_FORMAT;break;}if(status==FG_OK)status=fg_topology_assign_map(m,(const uint16_t (*)[FG_EXPERT_COUNT])expert_rank,err);free(line);free(expert_rank);if(fclose(file)!=0&&status==FG_OK){fg_error_set(err,FG_ERR_IO,"close expert map %s: %s",path,strerror(errno));status=FG_ERR_IO;}return status;
}
