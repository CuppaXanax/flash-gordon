#ifndef FG_MTP_RING_SSE_H
#define FG_MTP_RING_SSE_H
#include "fg_runtime.h"
typedef struct fixture_sse fixture_sse;
fg_status fixture_sse_open(fixture_sse **out,const char *stop,fg_error *err);
bool fixture_sse_stopped(fixture_sse *stream);
fg_status fixture_sse_token(void *opaque,uint32_t token,const char *text,size_t bytes,fg_error *err);
fg_status fixture_sse_finish(fixture_sse *stream,fg_error *err);
#endif
