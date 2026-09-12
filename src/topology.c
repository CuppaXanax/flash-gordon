#include "fg_topology.h"
#include "fg_sha256.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t ep_delta[FG_GROUP_SIZE]={0,1,3,5};

static void topology_hash_u16(fg_sha256 *hash,uint16_t value){
    uint8_t wire[2]={(uint8_t)(value>>8u),(uint8_t)value};
    fg_sha256_update(hash,wire,sizeof(wire));
}

static void topology_hash_u32(fg_sha256 *hash,uint32_t value){
    uint8_t wire[4]={(uint8_t)(value>>24u),(uint8_t)(value>>16u),
                     (uint8_t)(value>>8u),(uint8_t)value};
    fg_sha256_update(hash,wire,sizeof(wire));
}

static void topology_fingerprint(const fg_manifest *m,uint8_t out[32]){
    static const char domain[]="flash-gordon-topology-v1";
    fg_sha256 hash;fg_sha256_init(&hash);
    fg_sha256_update(&hash,domain,sizeof(domain)-1u);
    topology_hash_u32(&hash,m->execution_mode);
    topology_hash_u32(&hash,m->stage_count);
    fg_sha256_update(&hash,m->stage_ranks,sizeof(m->stage_ranks));
    for(uint32_t i=0;i<FG_RESERVED_LAYER_OFFSETS;i++)
        topology_hash_u32(&hash,m->layer_offsets[i]);
    topology_hash_u32(&hash,m->slot_count);
    fg_sha256_update(&hash,m->layer_owner,sizeof(m->layer_owner));
    fg_sha256_update(&hash,m->layer_groups,sizeof(m->layer_groups));
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)
        for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
            topology_hash_u16(&hash,m->expert_rank[layer][expert]);
    fg_sha256_final(&hash,out);
}

void fg_topology_seal(fg_manifest *m){
    if(m)topology_fingerprint(m,m->topology_sha256);
}

void fg_topology_set_expert_parallel_metadata(fg_manifest *m){
    if(!m)return;
    m->execution_mode=FG_EXECUTION_EXPERT_PARALLEL;
    m->stage_count=0u;
    memset(m->stage_ranks,0,sizeof(m->stage_ranks));
    memset(m->layer_offsets,0,sizeof(m->layer_offsets));
    m->slot_count=0u;
    memset(m->topology_reserved,0,sizeof(m->topology_reserved));
    fg_topology_seal(m);
}

void fg_topology_build(fg_manifest *m) {
    if(!m)return;
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        m->layer_owner[l]=(uint8_t)(l%FG_RANK_COUNT);
        for(uint32_t g=0;g<FG_GROUP_SIZE;g++)
            m->layer_groups[l][g]=(uint8_t)((l+ep_delta[g])%FG_RANK_COUNT);
    }
    fg_topology_set_expert_parallel_metadata(m);
}

bool fg_topology_rank_in_layer(const fg_manifest *m,uint32_t layer,uint32_t rank){
    if(layer>=FG_LAYER_COUNT||rank>=FG_RANK_COUNT)return false;
    for(uint32_t i=0;i<FG_GROUP_SIZE;i++)if(m->layer_groups[layer][i]==rank)return true;
    return false;
}

uint32_t fg_topology_layer_rank_experts(const fg_manifest *m,uint32_t layer,uint32_t rank){
    if(!m||layer>=FG_LAYER_COUNT||rank>=FG_RANK_COUNT)return 0u;
    uint32_t count=0u;
    for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
        if(m->expert_rank[layer][expert]==rank)count++;
    return count;
}

static uint32_t layer_distinct_ranks(const fg_manifest *m,uint32_t layer,uint32_t out[FG_GROUP_SIZE]){
    uint32_t count=0u;
    for(uint32_t g=0;g<FG_GROUP_SIZE;g++){
        uint32_t rank=m->layer_groups[layer][g];bool seen=false;
        for(uint32_t i=0;i<count;i++)if(out[i]==rank)seen=true;
        if(!seen)out[count++]=rank;
    }
    return count;
}

fg_status fg_topology_assign_round_robin(fg_manifest *m,fg_error *err){
    (void)err;
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++)for(uint32_t e=0;e<FG_EXPERT_COUNT;e++)
        m->expert_rank[l][e]=m->layer_groups[l][e%FG_GROUP_SIZE];
    fg_topology_seal(m);
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
    fg_topology_seal(m);
    return FG_OK;
}

fg_status fg_topology_assign_map(fg_manifest *m,const uint16_t expert_rank[FG_LAYER_COUNT][FG_EXPERT_COUNT],fg_error *err){
    if(!m||!expert_rank){fg_error_set(err,FG_ERR_ARGUMENT,"expert map is null");return FG_ERR_ARGUMENT;}
    for(uint32_t l=0;l<FG_LAYER_COUNT;l++){
        uint16_t count[FG_RANK_COUNT]={0};uint32_t distinct[FG_GROUP_SIZE];uint32_t participants=0u;
        for(uint32_t e=0;e<FG_EXPERT_COUNT;e++){
            uint32_t rank=expert_rank[l][e];
            if(rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_FORMAT,"layer %u expert %u assigned to rank %u outside the fleet",l,e,rank);return FG_ERR_FORMAT;}
            if(!count[rank])distinct[participants++]=rank;
            count[rank]++;
        }
        if(participants==1u){
            uint32_t rank=distinct[0];
            if(count[rank]!=FG_EXPERT_COUNT){fg_error_set(err,FG_ERR_FORMAT,"layer %u rank %u owns %u experts, expected %u",l,rank,count[rank],FG_EXPERT_COUNT);return FG_ERR_FORMAT;}
            for(uint32_t g=0;g<FG_GROUP_SIZE;g++)m->layer_groups[l][g]=(uint8_t)rank;
        }else{
            if(participants!=FG_GROUP_SIZE){fg_error_set(err,FG_ERR_FORMAT,"layer %u spans %u ranks, expected %u or a single owner",l,participants,FG_GROUP_SIZE);return FG_ERR_FORMAT;}
            for(uint32_t i=0;i<participants;i++)
                if(!fg_topology_rank_in_layer(m,l,distinct[i])){fg_error_set(err,FG_ERR_FORMAT,"layer %u expert assigned to rank %u outside its group",l,distinct[i]);return FG_ERR_FORMAT;}
            for(uint32_t g=0;g<FG_GROUP_SIZE;g++){
                uint32_t rank=m->layer_groups[l][g];
                if(count[rank]!=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_FORMAT,"layer %u rank %u owns %u experts, expected %u",l,rank,count[rank],FG_EXPERTS_PER_RANK);return FG_ERR_FORMAT;}
            }
        }
    }
    memcpy(m->expert_rank,expert_rank,sizeof(m->expert_rank));
    fg_topology_seal(m);
    return FG_OK;
}

fg_status fg_topology_validate(const fg_manifest *m,fg_error *err){
    if(!m){fg_error_set(err,FG_ERR_ARGUMENT,"manifest topology is null");return FG_ERR_ARGUMENT;}
    if(m->execution_mode!=FG_EXECUTION_EXPERT_PARALLEL){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "retired pipeline manifest is unsupported; repack for expert-parallel");
        return FG_ERR_MISMATCH;
    }

    if(m->format_version>=FG_MANIFEST_FORMAT_VERSION&&
       (m->stage_count||m->slot_count)){
        fg_error_set(err,FG_ERR_FORMAT,"expert-parallel manifest has pipeline geometry");
        return FG_ERR_FORMAT;
    }
    if(m->format_version>=FG_MANIFEST_FORMAT_VERSION){
        for(uint32_t i=0;i<FG_RANK_COUNT;i++)if(m->stage_ranks[i]){
            fg_error_set(err,FG_ERR_FORMAT,
                         "expert-parallel stage rank metadata is non-zero");
            return FG_ERR_FORMAT;
        }
        for(uint32_t i=0;i<FG_RESERVED_LAYER_OFFSETS;i++)
            if(m->layer_offsets[i]){
                fg_error_set(err,FG_ERR_FORMAT,
                             "expert-parallel layer offset metadata is non-zero");
                return FG_ERR_FORMAT;
            }
    }
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++){
        if(m->layer_owner[layer]>=FG_RANK_COUNT){
            fg_error_set(err,FG_ERR_FORMAT,"layer %u owner is outside the fleet",layer);
            return FG_ERR_FORMAT;
        }
        uint32_t distinct[FG_GROUP_SIZE];uint32_t participants=layer_distinct_ranks(m,layer,distinct);
        uint16_t counts[FG_RANK_COUNT]={0};
        if(participants==1u){
            uint32_t rank=distinct[0];
            if(rank>=FG_RANK_COUNT){
                fg_error_set(err,FG_ERR_FORMAT,"layer %u group topology mismatch",layer);
                return FG_ERR_FORMAT;
            }
            for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++)
                if(m->expert_rank[layer][expert]!=rank){
                    fg_error_set(err,FG_ERR_FORMAT,"layer %u expert %u assigned outside its single owner",layer,expert);
                    return FG_ERR_FORMAT;
                }
            continue;
        }
        uint16_t groups=0u;
        for(uint32_t group=0;group<FG_GROUP_SIZE;group++){
            uint32_t rank=m->layer_groups[layer][group];
            uint32_t expected=(layer+ep_delta[group])%FG_RANK_COUNT;
            if(rank>=FG_RANK_COUNT||rank!=expected||(groups&(1u<<rank))){
                fg_error_set(err,FG_ERR_FORMAT,"layer %u group topology mismatch",layer);
                return FG_ERR_FORMAT;
            }
            groups|=(uint16_t)(1u<<rank);
        }
        for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++){
            uint32_t rank=m->expert_rank[layer][expert];
            if(rank>=FG_RANK_COUNT||!(groups&(1u<<rank))){
                fg_error_set(err,FG_ERR_FORMAT,"layer %u expert %u assigned outside group",layer,expert);
                return FG_ERR_FORMAT;
            }
            counts[rank]++;
        }
        for(uint32_t group=0;group<FG_GROUP_SIZE;group++){
            uint32_t rank=m->layer_groups[layer][group];
            if(counts[rank]!=FG_EXPERTS_PER_RANK){
                fg_error_set(err,FG_ERR_FORMAT,"layer %u rank %u owns %u experts, expected %u",layer,rank,counts[rank],FG_EXPERTS_PER_RANK);
                return FG_ERR_FORMAT;
            }
        }
    }

    if(m->format_version>=FG_MANIFEST_FORMAT_VERSION){
        uint8_t reserved=0u;
        for(size_t i=0;i<sizeof(m->topology_reserved);i++)reserved|=m->topology_reserved[i];
        if(reserved){fg_error_set(err,FG_ERR_FORMAT,"manifest topology reserved bytes are non-zero");return FG_ERR_FORMAT;}
        uint8_t digest[32];topology_fingerprint(m,digest);
        if(memcmp(digest,m->topology_sha256,sizeof(digest))){
            fg_error_set(err,FG_ERR_MISMATCH,"manifest topology fingerprint mismatch");
            return FG_ERR_MISMATCH;
        }
    }
    return FG_OK;
}

static bool map_space(char value){return value==' '||value=='\t'||value=='\r'||value=='\n';}

fg_status fg_topology_assign_map_file(fg_manifest *m,const char *path,fg_error *err){
    if(!m||!path){fg_error_set(err,FG_ERR_ARGUMENT,"expert map path is null");return FG_ERR_ARGUMENT;}FILE *file=fopen(path,"r");if(!file){fg_error_set(err,FG_ERR_IO,"open expert map %s: %s",path,strerror(errno));return FG_ERR_IO;}uint16_t (*expert_rank)[FG_EXPERT_COUNT]=calloc(FG_LAYER_COUNT,sizeof(*expert_rank));if(!expert_rank){fclose(file);fg_error_set(err,FG_ERR_OOM,"allocate expert map");return FG_ERR_OOM;}char *line=NULL;size_t capacity=0;ssize_t bytes;bool seen[FG_LAYER_COUNT]={0};uint32_t line_number=0;fg_status status=FG_OK;uint8_t owner_values[FG_LAYER_COUNT]={0};bool seen_owners=false;
    while((bytes=getline(&line,&capacity,file))>=0){(void)bytes;line_number++;char *cursor=line;while(map_space(*cursor))cursor++;if(!*cursor||*cursor=='#')continue;if(strncmp(cursor,"owners=",7u)==0){if(seen_owners){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u duplicates the owners line",line_number);status=FG_ERR_FORMAT;break;}cursor+=7u;char *owner_end=NULL;for(uint32_t owner_layer=0;owner_layer<FG_LAYER_COUNT;owner_layer++){errno=0;unsigned long owner_rank=strtoul(cursor,&owner_end,10);if(errno||owner_end==cursor||owner_rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_FORMAT,"owners line has an invalid rank for layer %u",owner_layer);status=FG_ERR_FORMAT;break;}owner_values[owner_layer]=(uint8_t)owner_rank;if(owner_layer+1u<FG_LAYER_COUNT){if(*owner_end!=','){fg_error_set(err,FG_ERR_FORMAT,"owners line ends before layer %u",owner_layer+1u);status=FG_ERR_FORMAT;break;}cursor=owner_end+1u;}else{while(map_space(*owner_end))owner_end++;if(*owner_end){fg_error_set(err,FG_ERR_FORMAT,"owners line has trailing data");status=FG_ERR_FORMAT;}}}if(status!=FG_OK)break;seen_owners=true;continue;}if(strncmp(cursor,"layer=",6u)!=0){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u is malformed",line_number);status=FG_ERR_FORMAT;break;}char *end=NULL;errno=0;unsigned long layer=strtoul(cursor+6u,&end,10);if(errno||end==cursor+6u||layer>=FG_LAYER_COUNT||strncmp(end," ranks=",7u)!=0||seen[layer]){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u has an invalid or duplicate layer",line_number);status=FG_ERR_FORMAT;break;}cursor=end+7u;for(uint32_t expert=0;expert<FG_EXPERT_COUNT;expert++){errno=0;unsigned long rank=strtoul(cursor,&end,10);if(errno||end==cursor||rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u has an invalid rank for expert %u",line_number,expert);status=FG_ERR_FORMAT;break;}expert_rank[layer][expert]=(uint16_t)rank;if(expert+1u<FG_EXPERT_COUNT){if(*end!=','){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u ends before expert %u",line_number,expert+1u);status=FG_ERR_FORMAT;break;}cursor=end+1u;}else{while(map_space(*end))end++;if(*end){fg_error_set(err,FG_ERR_FORMAT,"expert map line %u has trailing data",line_number);status=FG_ERR_FORMAT;}}}if(status!=FG_OK)break;seen[layer]=true;}
    if(status==FG_OK&&ferror(file)){fg_error_set(err,FG_ERR_IO,"read expert map %s: %s",path,strerror(errno));status=FG_ERR_IO;}if(status==FG_OK)for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)if(!seen[layer]){fg_error_set(err,FG_ERR_FORMAT,"expert map is missing layer %u",layer);status=FG_ERR_FORMAT;break;}if(status==FG_OK)status=fg_topology_assign_map(m,(const uint16_t (*)[FG_EXPERT_COUNT])expert_rank,err);if(status==FG_OK&&seen_owners){memcpy(m->layer_owner,owner_values,sizeof(owner_values));fg_topology_seal(m);}free(line);free(expert_rank);if(fclose(file)!=0&&status==FG_OK){fg_error_set(err,FG_ERR_IO,"close expert map %s: %s",path,strerror(errno));status=FG_ERR_IO;}return status;
}
