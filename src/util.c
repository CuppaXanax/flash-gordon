#include "fg.h"

#include <stdarg.h>
#include <stdio.h>

void fg_error_set(fg_error *err,fg_status code,const char *fmt,...){if(!err)return;err->code=code;va_list ap;va_start(ap,fmt);vsnprintf(err->message,sizeof(err->message),fmt,ap);va_end(ap);}
uint64_t fg_align_up_u64(uint64_t v,uint64_t a){return a?((v+a-1u)&~(a-1u)):v;}
bool fg_is_aligned_u64(uint64_t v,uint64_t a){return a&&(v&(a-1u))==0;}
