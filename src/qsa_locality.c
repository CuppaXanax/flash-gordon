#include "fg_qsa_locality.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define FG_QSA_LOCALITY_LAYER_COUNT 12u
#define FG_QSA_LOCALITY_PAGE_BYTES \
    (FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES)

typedef struct fg_qsa_locality_layer {
    uint64_t selections,selected_refs,unique_pages,dedup_pages;
    uint64_t hot_tail_hits,cold_refs,previous_overlap,cache_hits,cache_misses;
    uint32_t previous_count;
} fg_qsa_locality_layer;

struct fg_qsa_locality {
    fg_qsa_locality_mode mode;
    uint32_t max_blocks,hot_tokens,root,generation;
    uint64_t clock,digest_key;
    fg_qsa_locality_stats stats;
    fg_qsa_locality_layer layers[FG_QSA_LOCALITY_LAYER_COUNT];
    uint32_t previous[FG_QSA_LOCALITY_LAYER_COUNT][FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t selected[FG_QSA_MAX_SELECTED_BLOCKS];
    uint32_t *seen,*left,*right,*subtree;
    uint64_t *last_access;
};

static uint64_t mix64(uint64_t value){
    value^=value>>30u;value*=UINT64_C(0xbf58476d1ce4e5b9);
    value^=value>>27u;value*=UINT64_C(0x94d049bb133111eb);
    return value^(value>>31u);
}

static uint32_t node_size(const fg_qsa_locality *trace,uint32_t node){
    return node?trace->subtree[node]:0u;
}

static void update_size(fg_qsa_locality *trace,uint32_t node){
    trace->subtree[node]=1u+node_size(trace,trace->left[node])+
        node_size(trace,trace->right[node]);
}

static uint64_t node_key(const fg_qsa_locality *trace,uint32_t node){
    return trace->last_access[node];
}

static uint64_t node_priority(const fg_qsa_locality *trace,uint32_t node){
    return mix64(trace->digest_key^((uint64_t)node*UINT64_C(0x9e3779b97f4a7c15)));
}

static uint32_t merge_nodes(fg_qsa_locality *trace,uint32_t left,uint32_t right){
    if(!left)return right;
    if(!right)return left;
    if(node_priority(trace,left)>node_priority(trace,right)){
        trace->right[left]=merge_nodes(trace,trace->right[left],right);
        update_size(trace,left);return left;
    }
    trace->left[right]=merge_nodes(trace,left,trace->left[right]);
    update_size(trace,right);return right;
}

static void split_nodes(fg_qsa_locality *trace,uint32_t root,uint64_t key,
                        uint32_t *left,uint32_t *right){
    if(!root){*left=0;*right=0;return;}
    if(node_key(trace,root)<key){
        *left=root;split_nodes(trace,trace->right[root],key,&trace->right[root],right);
        update_size(trace,root);
    }else{
        *right=root;split_nodes(trace,trace->left[root],key,left,&trace->left[root]);
        update_size(trace,root);
    }
}

static uint32_t insert_node(fg_qsa_locality *trace,uint32_t root,uint32_t node){
    if(!root)return node;
    if(node_priority(trace,node)>node_priority(trace,root)){
        split_nodes(trace,root,node_key(trace,node),&trace->left[node],&trace->right[node]);
        update_size(trace,node);return node;
    }
    if(node_key(trace,node)<node_key(trace,root))
        trace->left[root]=insert_node(trace,trace->left[root],node);
    else trace->right[root]=insert_node(trace,trace->right[root],node);
    update_size(trace,root);return root;
}

static uint32_t erase_node(fg_qsa_locality *trace,uint32_t root,uint64_t key){
    if(!root)return 0;
    if(node_key(trace,root)==key)return merge_nodes(trace,trace->left[root],trace->right[root]);
    if(key<node_key(trace,root))trace->left[root]=erase_node(trace,trace->left[root],key);
    else trace->right[root]=erase_node(trace,trace->right[root],key);
    update_size(trace,root);return root;
}

static uint32_t newer_nodes(const fg_qsa_locality *trace,uint32_t root,uint64_t key){
    if(!root)return 0u;
    if(node_key(trace,root)>key)
        return 1u+node_size(trace,trace->right[root])+
            newer_nodes(trace,trace->left[root],key);
    return newer_nodes(trace,trace->right[root],key);
}

static uint32_t layer_slot(uint32_t layer){
    return layer<FG_LAYER_COUNT&&(layer&3u)==3u?layer/4u:UINT32_MAX;
}

static uint32_t parse_budgets(uint32_t output[FG_QSA_LOCALITY_MAX_BUDGETS]){
    static const uint32_t defaults[]={16u,32u,64u,128u,256u,512u,1024u};
    const char *value=getenv("FG_QSA_LOCALITY_MIB");
    if(!value||!*value){
        memcpy(output,defaults,sizeof(defaults));
        return (uint32_t)(sizeof(defaults)/sizeof(defaults[0]));
    }
    uint32_t count=0;
    while(*value&&count<FG_QSA_LOCALITY_MAX_BUDGETS){
        errno=0;char *end=NULL;unsigned long parsed=strtoul(value,&end,10);
        if(errno||end==value||!parsed||parsed>UINT32_MAX)return 0u;
        output[count++]=(uint32_t)parsed;
        if(!*end)return count;
        if(*end!=',')return 0u;
        value=end+1;
    }
    return 0u;
}

fg_qsa_locality *fg_qsa_locality_create(fg_qsa_locality_mode mode,uint32_t max_blocks,
                                        uint32_t hot_tokens,const uint32_t *budget_mib,
                                        uint32_t budget_count,uint64_t digest_key){
    if((mode!=FG_QSA_LOCALITY_SUMMARY&&mode!=FG_QSA_LOCALITY_TOKEN)||!max_blocks||
       !hot_tokens||hot_tokens%FG_Q38_QSA_COMPRESS_RATIO||!budget_mib||!budget_count||
       budget_count>FG_QSA_LOCALITY_MAX_BUDGETS)return NULL;
    uint64_t nodes=(uint64_t)FG_QSA_LOCALITY_LAYER_COUNT*max_blocks;
    if(nodes>=UINT32_MAX||nodes>SIZE_MAX/sizeof(uint64_t))return NULL;
    fg_qsa_locality *trace=calloc(1,sizeof(*trace));
    if(!trace)return NULL;
    trace->mode=mode;trace->max_blocks=max_blocks;trace->hot_tokens=hot_tokens;
    trace->digest_key=digest_key?digest_key:UINT64_C(0x6a09e667f3bcc909);
    trace->seen=calloc(max_blocks,sizeof(*trace->seen));
    trace->left=calloc((size_t)nodes+1u,sizeof(*trace->left));
    trace->right=calloc((size_t)nodes+1u,sizeof(*trace->right));
    trace->subtree=calloc((size_t)nodes+1u,sizeof(*trace->subtree));
    trace->last_access=calloc((size_t)nodes+1u,sizeof(*trace->last_access));
    if(!trace->seen||!trace->left||!trace->right||!trace->subtree||!trace->last_access){
        fg_qsa_locality_destroy(trace,NULL);return NULL;
    }
    trace->stats.budget_count=budget_count;
    memcpy(trace->stats.budget_mib,budget_mib,(size_t)budget_count*sizeof(*budget_mib));
    return trace;
}

fg_qsa_locality *fg_qsa_locality_create_from_env(uint32_t max_blocks,uint32_t hot_tokens){
    const char *value=getenv("FG_QSA_LOCALITY_TRACE");
    if(!value||!*value)return NULL;
    fg_qsa_locality_mode mode;
    if(!strcmp(value,"summary"))mode=FG_QSA_LOCALITY_SUMMARY;
    else if(!strcmp(value,"token"))mode=FG_QSA_LOCALITY_TOKEN;
    else{
        fprintf(stderr,"FG_QSA_LOCALITY_TRACE must be summary or token; locality trace disabled\n");
        return NULL;
    }
    uint32_t budgets[FG_QSA_LOCALITY_MAX_BUDGETS];
    uint32_t count=parse_budgets(budgets);
    if(!count){
        fprintf(stderr,"FG_QSA_LOCALITY_MIB is invalid; locality trace disabled\n");
        return NULL;
    }
    struct timespec now={0};clock_gettime(CLOCK_REALTIME,&now);
    uint64_t key=mix64((uint64_t)now.tv_sec^(uint64_t)now.tv_nsec<<32u^
                       (uint64_t)(uint32_t)getpid());
    fg_qsa_locality *trace=fg_qsa_locality_create(mode,max_blocks,hot_tokens,budgets,count,key);
    if(!trace)fprintf(stderr,"QSA locality trace allocation failed; trace disabled\n");
    return trace;
}

uint32_t *fg_qsa_locality_selection_buffer(fg_qsa_locality *trace){
    return trace?trace->selected:NULL;
}

static uint32_t previous_overlap(const fg_qsa_locality *trace,uint32_t slot,
                                 const uint32_t *pages,uint32_t count){
    uint32_t overlap=0;
    for(uint32_t i=0;i<count;i++)for(uint32_t j=0;j<trace->layers[slot].previous_count;j++)
        if(pages[i]==trace->previous[slot][j]){overlap++;break;}
    return overlap;
}

static uint32_t record_reuse(fg_qsa_locality *trace,uint32_t slot,uint32_t block,
                             uint64_t *distance_sum,uint64_t *distance_max,
                             uint32_t *first_refs,uint64_t *hit_mask){
    uint32_t node=slot*trace->max_blocks+block+1u;
    uint64_t previous=trace->last_access[node];
    uint32_t distance=previous?newer_nodes(trace,trace->root,previous):UINT32_MAX;
    if(previous){
        trace->root=erase_node(trace,trace->root,previous);
        trace->stats.reused_cold_refs++;*distance_sum+=distance;
        if(distance>*distance_max)*distance_max=distance;
    }else{
        trace->stats.first_cold_refs++;(*first_refs)++;
    }
    trace->last_access[node]=++trace->clock;
    trace->left[node]=0;trace->right[node]=0;trace->subtree[node]=1u;
    trace->root=insert_node(trace,trace->root,node);
    for(uint32_t budget=0;budget<trace->stats.budget_count;budget++){
        uint64_t capacity=(uint64_t)trace->stats.budget_mib[budget]*1024u*1024u/
            FG_QSA_LOCALITY_PAGE_BYTES;
        bool hit=previous&&distance<capacity;
        if(hit){trace->stats.budget_hits[budget]++;*hit_mask|=UINT64_C(1)<<budget;}
        else trace->stats.budget_misses[budget]++;
    }
    return distance;
}

void fg_qsa_locality_record_selection(fg_qsa_locality *trace,uint32_t layer,uint32_t tokens,
                                      const uint32_t *page_ids,uint32_t page_count){
    uint32_t slot=layer_slot(layer);
    if(!trace||slot==UINT32_MAX||(page_count&&!page_ids)||
       page_count>FG_QSA_MAX_SELECTED_BLOCKS)return;
    if(++trace->generation==0u){
        memset(trace->seen,0,(size_t)trace->max_blocks*sizeof(*trace->seen));
        trace->generation=1u;
    }
    uint32_t unique=0,dedup=0;
    for(uint32_t i=0;i<page_count;i++){
        uint32_t block=page_ids[i];
        if(block>=trace->max_blocks)continue;
        if(trace->seen[block]==trace->generation){dedup++;continue;}
        trace->seen[block]=trace->generation;trace->selected[unique++]=block;
    }
    uint32_t overlap=previous_overlap(trace,slot,trace->selected,unique);
    uint32_t retained_first=tokens>trace->hot_tokens?tokens-trace->hot_tokens:0u;
    uint32_t hot=0,cold=0,first_refs=0,reused=0;uint64_t distance_sum=0,distance_max=0;
    uint64_t hit_mask=0;
    uint64_t digest=mix64(trace->digest_key^((uint64_t)layer<<32u)^tokens);
    for(uint32_t i=0;i<unique;i++){
        uint32_t block=trace->selected[i];
        digest=mix64(digest^mix64(trace->digest_key+(uint64_t)block+
                                  (uint64_t)i*UINT64_C(0x9e3779b97f4a7c15)));
        if((uint64_t)block*FG_Q38_QSA_COMPRESS_RATIO>=retained_first)hot++;
        else{
            uint64_t before=trace->stats.reused_cold_refs;
            record_reuse(trace,slot,block,&distance_sum,&distance_max,&first_refs,&hit_mask);
            reused+=(uint32_t)(trace->stats.reused_cold_refs-before);cold++;
        }
    }
    fg_qsa_locality_layer *layer_stats=&trace->layers[slot];
    layer_stats->selections++;layer_stats->selected_refs+=page_count;
    layer_stats->unique_pages+=unique;layer_stats->dedup_pages+=dedup;
    layer_stats->hot_tail_hits+=hot;layer_stats->cold_refs+=cold;
    layer_stats->previous_overlap+=overlap;
    trace->stats.selections++;trace->stats.selected_refs+=page_count;
    trace->stats.unique_pages+=unique;trace->stats.dedup_pages+=dedup;
    trace->stats.hot_tail_hits+=hot;trace->stats.cold_refs+=cold;
    trace->stats.previous_overlap+=overlap;
    trace->stats.reuse_distance_sum+=distance_sum;
    if(distance_max>trace->stats.reuse_distance_max)
        trace->stats.reuse_distance_max=distance_max;
    trace->stats.selection_digest=mix64(trace->stats.selection_digest^digest);
    layer_stats->previous_count=unique;
    memcpy(trace->previous[slot],trace->selected,(size_t)unique*sizeof(*trace->selected));
    if(trace->mode==FG_QSA_LOCALITY_TOKEN){
        double reuse_mean=reused?(double)distance_sum/reused:0.0;
        fprintf(stderr,"QSA_LOCALITY_TOKEN token=%u layer=%u digest=%016llx "
                "selected=%u unique=%u dedup=%u hot_tail=%u cold=%u overlap=%u "
                "first=%u reuse_mean=%.2f reuse_max=%llu lru_hit_mask=%016llx\n",
                tokens?tokens-1u:0u,layer,(unsigned long long)digest,page_count,unique,dedup,
                hot,cold,overlap,first_refs,reuse_mean,(unsigned long long)distance_max,
                (unsigned long long)hit_mask);
    }
}

void fg_qsa_locality_record_cache(fg_qsa_locality *trace,uint32_t layer,bool hit){
    uint32_t slot=layer_slot(layer);
    if(!trace||slot==UINT32_MAX)return;
    if(hit){trace->stats.cache_hits++;trace->layers[slot].cache_hits++;}
    else{trace->stats.cache_misses++;trace->layers[slot].cache_misses++;}
}

void fg_qsa_locality_get_stats(const fg_qsa_locality *trace,fg_qsa_locality_stats *stats){
    if(!stats)return;
    memset(stats,0,sizeof(*stats));
    if(trace)*stats=trace->stats;
}

static void emit_summary(const fg_qsa_locality *trace,const char *reason){
    if(!trace||!trace->stats.selections)return;
    fprintf(stderr,"QSA_LOCALITY_SUMMARY reason=%s selections=%llu selected=%llu unique=%llu "
            "dedup=%llu hot_tail=%llu cold=%llu overlap=%llu cache_hits=%llu "
            "cache_misses=%llu first=%llu reused=%llu reuse_mean=%.2f reuse_max=%llu "
            "digest=%016llx\n",reason?reason:"unknown",
            (unsigned long long)trace->stats.selections,
            (unsigned long long)trace->stats.selected_refs,
            (unsigned long long)trace->stats.unique_pages,
            (unsigned long long)trace->stats.dedup_pages,
            (unsigned long long)trace->stats.hot_tail_hits,
            (unsigned long long)trace->stats.cold_refs,
            (unsigned long long)trace->stats.previous_overlap,
            (unsigned long long)trace->stats.cache_hits,
            (unsigned long long)trace->stats.cache_misses,
            (unsigned long long)trace->stats.first_cold_refs,
            (unsigned long long)trace->stats.reused_cold_refs,
            trace->stats.reused_cold_refs?
                (double)trace->stats.reuse_distance_sum/trace->stats.reused_cold_refs:0.0,
            (unsigned long long)trace->stats.reuse_distance_max,
            (unsigned long long)trace->stats.selection_digest);
    for(uint32_t slot=0;slot<FG_QSA_LOCALITY_LAYER_COUNT;slot++){
        const fg_qsa_locality_layer *layer=&trace->layers[slot];
        if(!layer->selections)continue;
        fprintf(stderr,"QSA_LOCALITY_LAYER layer=%u selections=%llu selected=%llu unique=%llu "
                "dedup=%llu hot_tail=%llu cold=%llu overlap=%llu cache_hits=%llu "
                "cache_misses=%llu\n",slot*4u+3u,(unsigned long long)layer->selections,
                (unsigned long long)layer->selected_refs,
                (unsigned long long)layer->unique_pages,
                (unsigned long long)layer->dedup_pages,
                (unsigned long long)layer->hot_tail_hits,
                (unsigned long long)layer->cold_refs,
                (unsigned long long)layer->previous_overlap,
                (unsigned long long)layer->cache_hits,
                (unsigned long long)layer->cache_misses);
    }
    for(uint32_t budget=0;budget<trace->stats.budget_count;budget++){
        uint64_t hits=trace->stats.budget_hits[budget];
        uint64_t misses=trace->stats.budget_misses[budget],total=hits+misses;
        fprintf(stderr,"QSA_LOCALITY_LRU budget_mib=%u capacity_pages=%llu hits=%llu "
                "misses=%llu hit_rate=%.6f\n",trace->stats.budget_mib[budget],
                (unsigned long long)((uint64_t)trace->stats.budget_mib[budget]*1024u*1024u/
                                     FG_QSA_LOCALITY_PAGE_BYTES),
                (unsigned long long)hits,(unsigned long long)misses,
                total?(double)hits/total:0.0);
    }
}

void fg_qsa_locality_reset(fg_qsa_locality *trace,const char *reason){
    if(!trace)return;
    emit_summary(trace,reason);
    uint32_t budget_count=trace->stats.budget_count;
    uint32_t budgets[FG_QSA_LOCALITY_MAX_BUDGETS];
    memcpy(budgets,trace->stats.budget_mib,(size_t)budget_count*sizeof(*budgets));
    memset(&trace->stats,0,sizeof(trace->stats));trace->stats.budget_count=budget_count;
    memcpy(trace->stats.budget_mib,budgets,(size_t)budget_count*sizeof(*budgets));
    memset(trace->layers,0,sizeof(trace->layers));
    memset(trace->seen,0,(size_t)trace->max_blocks*sizeof(*trace->seen));
    uint64_t nodes=(uint64_t)FG_QSA_LOCALITY_LAYER_COUNT*trace->max_blocks;
    memset(trace->left,0,((size_t)nodes+1u)*sizeof(*trace->left));
    memset(trace->right,0,((size_t)nodes+1u)*sizeof(*trace->right));
    memset(trace->subtree,0,((size_t)nodes+1u)*sizeof(*trace->subtree));
    memset(trace->last_access,0,((size_t)nodes+1u)*sizeof(*trace->last_access));
    trace->root=0;trace->generation=0;trace->clock=0;
}

void fg_qsa_locality_destroy(fg_qsa_locality *trace,const char *reason){
    if(!trace)return;
    emit_summary(trace,reason);
    free(trace->last_access);free(trace->subtree);free(trace->right);free(trace->left);
    free(trace->seen);free(trace);
}
