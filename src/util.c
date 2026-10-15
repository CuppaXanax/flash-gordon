#include "fg.h"

#include <stdarg.h>
#include <stdio.h>

static uint32_t active_rank_count=FG_DEFAULT_RANK_COUNT;

uint32_t fg_rank_count(void){return active_rank_count;}

fg_status fg_rank_count_set(uint32_t count,fg_error *err){
    if(count<4u||count>FG_MAX_RANK_COUNT||FG_LAYER_COUNT%count){
        fg_error_set(err,FG_ERR_ARGUMENT,"rank count must divide 48 and be in [4,16]");
        return FG_ERR_ARGUMENT;
    }
    active_rank_count=count;
    return FG_OK;
}

void fg_error_set(fg_error *err,fg_status code,const char *fmt,...){if(!err)return;err->code=code;va_list ap;va_start(ap,fmt);vsnprintf(err->message,sizeof(err->message),fmt,ap);va_end(ap);}
uint64_t fg_align_up_u64(uint64_t v,uint64_t a){return a?((v+a-1u)&~(a-1u)):v;}
bool fg_is_aligned_u64(uint64_t v,uint64_t a){return a&&(v&(a-1u))==0;}
