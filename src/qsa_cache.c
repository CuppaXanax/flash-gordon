#include "fg_qsa_cache.h"

#include <stdlib.h>
#include <string.h>

struct fg_qsa_page_cache {
    uint32_t pages,hash_capacity,used,lru_head,lru_tail;
    uint32_t *blocks,*hash_buckets,*hash_next,*lru_prev,*lru_next;
    uint8_t *layers,*valid,*pinned;
};

static uint32_t cache_bucket(const fg_qsa_page_cache *cache,uint32_t layer,uint32_t block){
    uint64_t value=((uint64_t)layer<<32u)|block;
    value^=value>>33u;value*=UINT64_C(0xff51afd7ed558ccd);value^=value>>33u;
    return (uint32_t)value&(cache->hash_capacity-1u);
}

static uint32_t cache_find(const fg_qsa_page_cache *cache,uint32_t layer,uint32_t block){
    if(!cache||!cache->pages)return UINT32_MAX;
    uint32_t link=cache->hash_buckets[cache_bucket(cache,layer,block)];
    while(link){
        uint32_t slot=link-1u;
        if(cache->valid[slot]&&cache->layers[slot]==layer&&cache->blocks[slot]==block)
            return slot;
        link=cache->hash_next[slot];
    }
    return UINT32_MAX;
}

static void cache_lru_front(fg_qsa_page_cache *cache,uint32_t slot){
    uint32_t previous=cache->lru_prev[slot],next=cache->lru_next[slot];
    if(previous!=UINT32_MAX)cache->lru_next[previous]=next;
    else if(cache->lru_head==slot)cache->lru_head=next;
    if(next!=UINT32_MAX)cache->lru_prev[next]=previous;
    else if(cache->lru_tail==slot)cache->lru_tail=previous;
    cache->lru_prev[slot]=UINT32_MAX;cache->lru_next[slot]=cache->lru_head;
    if(cache->lru_head!=UINT32_MAX)cache->lru_prev[cache->lru_head]=slot;
    else cache->lru_tail=slot;
    cache->lru_head=slot;
}

static void cache_remove_hash(fg_qsa_page_cache *cache,uint32_t slot){
    uint32_t bucket=cache_bucket(cache,cache->layers[slot],cache->blocks[slot]);
    uint32_t *link=&cache->hash_buckets[bucket];
    while(*link){
        uint32_t candidate=*link-1u;
        if(candidate==slot){*link=cache->hash_next[slot];return;}
        link=&cache->hash_next[candidate];
    }
}

fg_status fg_qsa_page_cache_create(fg_qsa_page_cache **out,uint32_t pages,fg_error *err){
    if(!out||!pages||pages>(UINT32_C(1)<<30u)){
        fg_error_set(err,pages>(UINT32_C(1)<<30u)?FG_ERR_LIMIT:FG_ERR_ARGUMENT,
                     "invalid QSA page cache capacity");
        return pages>(UINT32_C(1)<<30u)?FG_ERR_LIMIT:FG_ERR_ARGUMENT;
    }
    *out=NULL;fg_qsa_page_cache *cache=calloc(1,sizeof(*cache));
    if(!cache){fg_error_set(err,FG_ERR_OOM,"allocate QSA page cache");return FG_ERR_OOM;}
    cache->pages=pages;cache->hash_capacity=1u;
    while(cache->hash_capacity<pages*2u)cache->hash_capacity<<=1u;
    cache->blocks=calloc(pages,sizeof(*cache->blocks));
    cache->layers=calloc(pages,sizeof(*cache->layers));
    cache->valid=calloc(pages,sizeof(*cache->valid));
    cache->pinned=calloc(pages,sizeof(*cache->pinned));
    cache->hash_buckets=calloc(cache->hash_capacity,sizeof(*cache->hash_buckets));
    cache->hash_next=calloc(pages,sizeof(*cache->hash_next));
    cache->lru_prev=malloc((size_t)pages*sizeof(*cache->lru_prev));
    cache->lru_next=malloc((size_t)pages*sizeof(*cache->lru_next));
    if(!cache->blocks||!cache->layers||!cache->valid||!cache->pinned||!cache->hash_buckets||
       !cache->hash_next||!cache->lru_prev||!cache->lru_next){
        fg_qsa_page_cache_destroy(cache);
        fg_error_set(err,FG_ERR_OOM,"allocate QSA page cache storage");return FG_ERR_OOM;
    }
    cache->lru_head=cache->lru_tail=UINT32_MAX;*out=cache;return FG_OK;
}

void fg_qsa_page_cache_destroy(fg_qsa_page_cache *cache){
    if(!cache)return;
    free(cache->lru_next);free(cache->lru_prev);
    free(cache->hash_next);free(cache->hash_buckets);free(cache->pinned);
    free(cache->valid);free(cache->layers);
    free(cache->blocks);free(cache);
}

void fg_qsa_page_cache_reset(fg_qsa_page_cache *cache){
    if(!cache)return;
    memset(cache->valid,0,cache->pages);
    memset(cache->pinned,0,cache->pages);
    memset(cache->hash_buckets,0,(size_t)cache->hash_capacity*sizeof(*cache->hash_buckets));
    cache->used=0;cache->lru_head=cache->lru_tail=UINT32_MAX;
}

uint64_t fg_qsa_page_cache_memory_bytes(const fg_qsa_page_cache *cache){
    if(!cache)return 0;
    return sizeof(*cache)+(uint64_t)cache->pages*
        (sizeof(*cache->blocks)+sizeof(*cache->layers)+sizeof(*cache->valid)+
         sizeof(*cache->pinned)+
         sizeof(*cache->hash_next)+sizeof(*cache->lru_prev)+sizeof(*cache->lru_next))+
         (uint64_t)cache->hash_capacity*
             sizeof(*cache->hash_buckets);
}

uint64_t fg_qsa_page_cache_memory_bytes_for_pages(uint32_t pages){
    if(!pages)return 0;
    uint32_t hash_capacity=1u;
    while(hash_capacity<pages*2u)hash_capacity<<=1u;
    return sizeof(fg_qsa_page_cache)+(uint64_t)pages*
        (sizeof(uint32_t)+sizeof(uint8_t)+sizeof(uint8_t)+sizeof(uint8_t)+
         sizeof(uint32_t)+sizeof(uint32_t)+sizeof(uint32_t))+
         (uint64_t)hash_capacity*sizeof(uint32_t);
}

bool fg_qsa_page_cache_lookup(fg_qsa_page_cache *cache,uint32_t layer,
                              uint32_t block,uint32_t *slot){
    if(!slot)return false;
    uint32_t found=cache_find(cache,layer,block);
    if(found==UINT32_MAX)return false;
    cache_lru_front(cache,found);*slot=found;return true;
}

fg_status fg_qsa_page_cache_acquire(fg_qsa_page_cache *cache,uint32_t layer,
                                   uint32_t block,uint32_t *slot,bool *hit,
                                   fg_error *err){
    if(!cache||!slot||!hit){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA cache acquisition");
        return FG_ERR_ARGUMENT;
    }
    uint32_t found=cache_find(cache,layer,block);
    *hit=found!=UINT32_MAX;
    if(found==UINT32_MAX){
        if(cache->used<cache->pages){
            found=cache->used++;cache->lru_prev[found]=cache->lru_next[found]=UINT32_MAX;
        }else{
            found=cache->lru_tail;
            while(found!=UINT32_MAX&&cache->pinned[found])
                found=cache->lru_prev[found];
            if(found==UINT32_MAX){
                fg_error_set(err,FG_ERR_LIMIT,"QSA cache has no evictable slot");
                return FG_ERR_LIMIT;
            }
            cache_remove_hash(cache,found);
        }
        cache->valid[found]=1u;cache->layers[found]=(uint8_t)layer;
        cache->blocks[found]=block;
        uint32_t bucket=cache_bucket(cache,layer,block);
        cache->hash_next[found]=cache->hash_buckets[bucket];
        cache->hash_buckets[bucket]=found+1u;
    }
    cache_lru_front(cache,found);*slot=found;return FG_OK;
}

fg_status fg_qsa_page_cache_pin(fg_qsa_page_cache *cache,uint32_t layer,
                                uint32_t block,fg_error *err){
    uint32_t slot=cache_find(cache,layer,block);
    if(slot==UINT32_MAX){
        fg_error_set(err,FG_ERR_MISMATCH,"cannot pin a missing QSA cache page");
        return FG_ERR_MISMATCH;
    }
    cache->pinned[slot]=1u;return FG_OK;
}

void fg_qsa_page_cache_unpin(fg_qsa_page_cache *cache,uint32_t layer,uint32_t block){
    uint32_t slot=cache_find(cache,layer,block);
    if(slot!=UINT32_MAX)cache->pinned[slot]=0u;
}

fg_status fg_qsa_page_cache_plan_fetch(const fg_qsa_page_cache *cache,uint32_t layer,
                                       const uint32_t *missing,uint32_t missing_count,
                                       uint32_t complete_blocks,uint32_t retained_first,
                                       uint32_t *fetch,uint32_t fetch_capacity,
                                       uint32_t *fetch_count,fg_error *err){
    if(!missing||!missing_count||!fetch||!fetch_count||missing_count>fetch_capacity){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA cold-page fetch plan");
        return FG_ERR_ARGUMENT;
    }
    memcpy(fetch,missing,(size_t)missing_count*sizeof(*missing));uint32_t count=missing_count;
    for(uint32_t i=0;i<missing_count&&count<fetch_capacity;i++){
        uint32_t candidate=missing[i]+1u;
        if(candidate>=complete_blocks||
           (uint64_t)candidate*FG_Q38_QSA_COMPRESS_RATIO>=retained_first||
           cache_find(cache,layer,candidate)!=UINT32_MAX)continue;
        bool duplicate=false;
        for(uint32_t j=0;j<count;j++)if(fetch[j]==candidate){duplicate=true;break;}
        if(!duplicate)fetch[count++]=candidate;
    }
    *fetch_count=count;return FG_OK;
}
