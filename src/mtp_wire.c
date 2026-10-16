#include "fg_mtp_wire.h"

#include <math.h>
#include <string.h>

static void put32(uint8_t *p,uint32_t x){
    p[0]=(uint8_t)(x>>24u);p[1]=(uint8_t)(x>>16u);
    p[2]=(uint8_t)(x>>8u);p[3]=(uint8_t)x;
}
static uint32_t get32(const uint8_t *p){
    return (uint32_t)p[0]<<24u|(uint32_t)p[1]<<16u|
        (uint32_t)p[2]<<8u|p[3];
}
static void put_float(uint8_t *p,float x){
    uint32_t bits;memcpy(&bits,&x,4u);put32(p,bits);
}
static float get_float(const uint8_t *p){
    uint32_t bits=get32(p);float x;memcpy(&x,&bits,4u);return x;
}
static fg_status malformed(fg_error *err,const char *what){
    fg_error_set(err,FG_ERR_FORMAT,"invalid MTP %s",what);return FG_ERR_FORMAT;
}
static bool operation_valid(fg_mtp_native_operation op){
    return op>=FG_MTP_NATIVE_CONFIG&&op<=FG_MTP_NATIVE_PREPARE;
}
fg_status fg_mtp_prefill_tokens_encode(uint8_t *wire,uint32_t capacity,
    uint32_t *bytes,const uint32_t *next_tokens,uint32_t count,fg_error *err){
    if(!wire||!bytes||!next_tokens||!count||count>FG_PREFILL_MAX_TOKENS||
       *bytes<FG_PREFILL_LAYER_HEADER_BYTES||*bytes>capacity||
       count>(capacity-*bytes)/4u||
       ((uint32_t)wire[8]<<8u|wire[9])!=count)
        return malformed(err,"prefill token trailer");
    for(uint32_t i=0;i<count;i++){
        if(next_tokens[i]>=FG_Q38_VOCAB_SIZE&&
           !(next_tokens[i]==UINT32_MAX&&i+1u==count))
            return malformed(err,"prefill next-token ID");
        put32(wire+*bytes+i*4u,next_tokens[i]);
    }
    *bytes+=count*4u;return FG_OK;
}
fg_status fg_mtp_prefill_tokens_decode(uint32_t *next_tokens,uint32_t capacity,
    uint32_t *body_bytes,const uint8_t *wire,uint32_t bytes,fg_error *err){
    if(!next_tokens||!body_bytes||!wire||bytes<FG_PREFILL_LAYER_HEADER_BYTES)
        return malformed(err,"prefill token trailer");
    uint32_t count=(uint32_t)wire[8]<<8u|wire[9];
    if(!count||count>FG_PREFILL_MAX_TOKENS||count>capacity||
       bytes<FG_PREFILL_LAYER_HEADER_BYTES+count*4u)
        return malformed(err,"prefill token trailer size");
    *body_bytes=bytes-count*4u;
    for(uint32_t i=0;i<count;i++){
        next_tokens[i]=get32(wire+*body_bytes+i*4u);
        if(next_tokens[i]>=FG_Q38_VOCAB_SIZE&&
           !(next_tokens[i]==UINT32_MAX&&i+1u==count))
            return malformed(err,"prefill next-token ID");
    }
    return FG_OK;
}
static bool head_valid(fg_mtp_head_kind kind,uint32_t source,uint32_t destination,
                        uint32_t count,uint32_t first){
    return (kind==FG_MTP_HEAD_DRAFT||kind==FG_MTP_HEAD_VERIFY)&&
        source<FG_RANK_COUNT&&destination<FG_RANK_COUNT&&count&&
        count<=FG_MTP_MAX_VERIFY&&first<=FG_MAX_CONTEXT-count;
}
fg_status fg_mtp_verify_result_encode(uint8_t out[FG_MTP_VERIFY_RESULT_BYTES],
    const fg_mtp_verify_result *r,fg_error *err){
    if(!out||!r||r->source_rank>=FG_RANK_COUNT||!r->count||
       r->count>FG_MTP_MAX_VERIFY||r->first_token>FG_NATIVE_CONTEXT-r->count)
        return malformed(err,"verification completion");
    memset(out,0,FG_MTP_VERIFY_RESULT_BYTES);
    out[0]=FG_MTP_WIRE_VERSION;out[1]=r->source_rank;out[3]=FG_LAYER_COUNT-1u;
    put32(out+4u,r->first_token);put32(out+8u,r->count);return FG_OK;
}
fg_status fg_mtp_verify_result_decode(fg_mtp_verify_result *r,const uint8_t *p,
    uint32_t bytes,fg_error *err){
    if(!r||!p||bytes!=FG_MTP_VERIFY_RESULT_BYTES||p[0]!=FG_MTP_WIRE_VERSION||
       p[1]>=FG_RANK_COUNT||p[2]||p[3]!=FG_LAYER_COUNT-1u||get32(p+12u)||
       !get32(p+8u)||get32(p+8u)>FG_MTP_MAX_VERIFY||
       get32(p+4u)>FG_NATIVE_CONTEXT-get32(p+8u))
        return malformed(err,"verification completion");
    *r=(fg_mtp_verify_result){.source_rank=p[1],.first_token=get32(p+4u),
                              .count=get32(p+8u)};
    return FG_OK;
}
static fg_status native_shape(const fg_mtp_native_work *w,uint32_t *rows,
                               uint32_t *path_bytes,uint32_t *embedding_path,fg_error *err){
    *rows=0u;*path_bytes=0u;*embedding_path=0u;
    if(!w||!operation_valid(w->operation))return malformed(err,"native operation");
    switch(w->operation){
        case FG_MTP_NATIVE_CONFIG:
            if(w->first_token||!w->overlay||!w->overlay[0]||
               !w->embedding_file||!w->embedding_file[0]||
               !w->argument||w->argument>FG_NATIVE_CONTEXT||
               !w->limit||w->limit>w->argument)
                return malformed(err,"native configuration");
            *path_bytes=(uint32_t)strnlen(w->overlay,FG_MTP_OVERLAY_PATH_MAX+1u);
            *embedding_path=(uint32_t)strnlen(w->embedding_file,FG_MTP_OVERLAY_PATH_MAX+1u);
            if(!*path_bytes||*path_bytes>FG_MTP_OVERLAY_PATH_MAX||
               !*embedding_path||*embedding_path>FG_MTP_OVERLAY_PATH_MAX||
               w->count!=*path_bytes)return malformed(err,"overlay path");
            return FG_OK;
        case FG_MTP_NATIVE_PREFIX:
            if(w->count>1u||w->first_token>=w->limit||
               w->limit>FG_NATIVE_CONTEXT||w->argument>=w->limit||
               (w->count&&!w->first_token))return malformed(err,"prefix range");
            *rows=w->count;break;
        case FG_MTP_NATIVE_DRAFT:
            if(!w->first_token||w->first_token>=FG_NATIVE_CONTEXT||
               w->count!=1u||w->argument>=FG_MTP_MAX_DRAFT||w->limit)
                return malformed(err,"draft range");
            *rows=1u;break;
        case FG_MTP_NATIVE_FINISH:
            if(!w->first_token||!w->count||w->count>FG_MTP_MAX_VERIFY||
               w->first_token>FG_NATIVE_CONTEXT-w->count||
               w->argument||w->limit)return malformed(err,"repair range");
            *rows=w->count-1u;break;
        case FG_MTP_NATIVE_ADVANCE:
            if(!w->first_token||w->first_token>=FG_NATIVE_CONTEXT||
               w->count!=1u||w->argument||w->limit)
                return malformed(err,"advance range");
            *rows=1u;break;
        case FG_MTP_NATIVE_PREPARE:
            if(w->first_token||w->count||w->limit||
               !w->argument||w->argument>2u)
                return malformed(err,"session admission");
            break;
        default:return malformed(err,"native operation");
    }
    if(*rows&&!w->tokens)return malformed(err,"token storage");
    return FG_OK;
}

fg_status fg_mtp_native_work_encode(uint8_t *out,uint32_t capacity,uint32_t *bytes,
    const fg_mtp_native_work *w,fg_error *err){
    if(!out||!bytes)return malformed(err,"native output storage");
    uint32_t rows=0u,path=0u,embedding_path=0u;
    fg_status status=native_shape(w,&rows,&path,&embedding_path,err);
    if(status!=FG_OK)return status;
    uint32_t required=FG_MTP_NATIVE_HEADER_BYTES+path+embedding_path+rows*4u;
    if(required>capacity)return malformed(err,"native output capacity");
    memset(out,0,FG_MTP_NATIVE_HEADER_BYTES);
    out[0]=FG_MTP_WIRE_VERSION;out[1]=(uint8_t)w->operation;
    put32(out+4u,w->first_token);put32(out+8u,w->count);
    put32(out+12u,w->argument);put32(out+16u,w->limit);
    put32(out+20u,embedding_path);
    if(path)memcpy(out+FG_MTP_NATIVE_HEADER_BYTES,w->overlay,path);
    if(embedding_path)memcpy(out+FG_MTP_NATIVE_HEADER_BYTES+path,w->embedding_file,embedding_path);
    for(uint32_t i=0;i<rows;i++){
        if(w->tokens[i]>=FG_Q38_VOCAB_SIZE)return malformed(err,"embedding token ID");
        put32(out+FG_MTP_NATIVE_HEADER_BYTES+i*4u,w->tokens[i]);
    }
    *bytes=required;return FG_OK;
}

fg_status fg_mtp_native_work_decode(fg_mtp_native_work *w,uint32_t *tokens,
    uint32_t capacity,char overlay[FG_MTP_OVERLAY_PATH_MAX+1u],
    char embedding_file[FG_MTP_OVERLAY_PATH_MAX+1u],
    const uint8_t *p,uint32_t bytes,fg_error *err){
    if(!w||!p||bytes<FG_MTP_NATIVE_HEADER_BYTES||
       p[0]!=FG_MTP_WIRE_VERSION||p[2]||p[3])
        return malformed(err,"native header");
    *w=(fg_mtp_native_work){.operation=(fg_mtp_native_operation)p[1],
        .first_token=get32(p+4u),.count=get32(p+8u),.argument=get32(p+12u),
        .limit=get32(p+16u),.tokens=tokens};
    uint32_t second=get32(p+20u);
    if(w->operation==FG_MTP_NATIVE_CONFIG){
        if(!overlay||!embedding_file||!w->count||w->count>FG_MTP_OVERLAY_PATH_MAX||
           !second||second>FG_MTP_OVERLAY_PATH_MAX||
           bytes!=FG_MTP_NATIVE_HEADER_BYTES+w->count+second||
           memchr(p+FG_MTP_NATIVE_HEADER_BYTES,0,w->count+second))
            return malformed(err,"overlay path length");
        memcpy(overlay,p+FG_MTP_NATIVE_HEADER_BYTES,w->count);
        overlay[w->count]=0;w->overlay=overlay;
        memcpy(embedding_file,p+FG_MTP_NATIVE_HEADER_BYTES+w->count,second);
        embedding_file[second]=0;w->embedding_file=embedding_file;
    }else if(second)return malformed(err,"native reserved bytes");
    uint32_t rows=0u,path=0u,embedding_path=0u;
    fg_status status=native_shape(w,&rows,&path,&embedding_path,err);
    if(status!=FG_OK)return status;
    if(rows>capacity||
       FG_MTP_NATIVE_HEADER_BYTES+path+embedding_path+rows*4u!=bytes)
        return malformed(err,"native payload size");
    for(uint32_t i=0;i<rows;i++){
        tokens[i]=get32(p+FG_MTP_NATIVE_HEADER_BYTES+i*4u);
        if(tokens[i]>=FG_Q38_VOCAB_SIZE)return malformed(err,"embedding token ID");
    }
    return FG_OK;
}

static bool ack_status_valid(fg_mtp_native_operation operation,fg_status status){
    return status==FG_OK||(operation==FG_MTP_NATIVE_PREPARE&&
        (status==FG_ERR_LIMIT||status==FG_ERR_OOM));
}
fg_status fg_mtp_native_ack_encode(uint8_t out[FG_MTP_NATIVE_ACK_BYTES],
    const fg_mtp_native_ack *a,fg_error *err){
    if(!out||!a||!operation_valid(a->operation)||a->frontier>FG_NATIVE_CONTEXT||
       !ack_status_valid(a->operation,a->status))
        return malformed(err,"native acknowledgement");
    memset(out,0,FG_MTP_NATIVE_ACK_BYTES);
    out[0]=FG_MTP_WIRE_VERSION;out[1]=(uint8_t)a->operation;
    out[2]=(uint8_t)a->status;
    put32(out+4u,a->first_token);put32(out+8u,a->count);
    put32(out+12u,a->frontier);return FG_OK;
}
fg_status fg_mtp_native_ack_decode(fg_mtp_native_ack *a,const uint8_t *p,
    uint32_t bytes,fg_error *err){
    if(!a||!p||bytes!=FG_MTP_NATIVE_ACK_BYTES||
       p[0]!=FG_MTP_WIRE_VERSION||p[3]||
       !operation_valid((fg_mtp_native_operation)p[1])||
       !ack_status_valid((fg_mtp_native_operation)p[1],(fg_status)p[2])||
       get32(p+12u)>FG_NATIVE_CONTEXT)return malformed(err,"native acknowledgement");
    *a=(fg_mtp_native_ack){.operation=(fg_mtp_native_operation)p[1],
        .first_token=get32(p+4u),.count=get32(p+8u),.frontier=get32(p+12u),
        .status=(fg_status)p[2]};
    return FG_OK;
}

static void head_header(uint8_t *p,fg_mtp_head_kind kind,uint8_t source,
    uint8_t destination,uint32_t first,uint32_t count){
    memset(p,0,FG_MTP_HEAD_HEADER_BYTES);p[0]=FG_MTP_WIRE_VERSION;
    p[1]=(uint8_t)kind;p[2]=source;p[3]=destination;
    put32(p+4u,first);put32(p+8u,count);
}
static bool head_header_valid(const uint8_t *p,uint32_t bytes){
    return bytes>=FG_MTP_HEAD_HEADER_BYTES&&p[0]==FG_MTP_WIRE_VERSION&&
        !get32(p+12u)&&head_valid((fg_mtp_head_kind)p[1],p[2],p[3],
                                 get32(p+8u),get32(p+4u));
}
fg_status fg_mtp_head_work_encode(uint8_t *out,uint32_t capacity,uint32_t *bytes,
    const fg_mtp_head_work *w,fg_error *err){
    if(!out||!bytes||!w||!w->hidden||
       !head_valid(w->kind,w->source_rank,w->destination_rank,w->count,w->first_token))
        return malformed(err,"head work");
    uint32_t n=w->count*FG_HIDDEN_SIZE,required=FG_MTP_HEAD_HEADER_BYTES+n*4u;
    if(required>capacity)return malformed(err,"head output capacity");
    head_header(out,w->kind,w->source_rank,w->destination_rank,w->first_token,w->count);
    for(uint32_t i=0;i<n;i++){
        if(!isfinite(w->hidden[i]))return malformed(err,"non-finite contracted hidden");
        put_float(out+FG_MTP_HEAD_HEADER_BYTES+i*4u,w->hidden[i]);
    }
    *bytes=required;return FG_OK;
}
fg_status fg_mtp_head_work_decode(fg_mtp_head_work *w,float *hidden,uint64_t values,
    const uint8_t *p,uint32_t bytes,fg_error *err){
    if(!w||!hidden||!p||!head_header_valid(p,bytes))
        return malformed(err,"head work header");
    uint32_t count=get32(p+8u),n=count*FG_HIDDEN_SIZE;
    if(values<n||bytes!=FG_MTP_HEAD_HEADER_BYTES+n*4u)
        return malformed(err,"head work size");
    for(uint32_t i=0;i<n;i++){
        hidden[i]=get_float(p+FG_MTP_HEAD_HEADER_BYTES+i*4u);
        if(!isfinite(hidden[i]))return malformed(err,"non-finite contracted hidden");
    }
    *w=(fg_mtp_head_work){.kind=(fg_mtp_head_kind)p[1],.source_rank=p[2],
        .destination_rank=p[3],.first_token=get32(p+4u),.count=count,.hidden=hidden};
    return FG_OK;
}
fg_status fg_mtp_head_partial_encode(uint8_t *out,uint32_t capacity,uint32_t *bytes,
    const fg_mtp_head_partial *s,fg_error *err){
    if(!out||!bytes||!s||
       !head_valid(s->kind,s->source_rank,s->destination_rank,s->count,s->first_token))
        return malformed(err,"head partial");
    uint32_t required=FG_MTP_HEAD_HEADER_BYTES+s->count*8u;
    if(capacity<required)return malformed(err,"head partial capacity");
    head_header(out,s->kind,s->source_rank,s->destination_rank,s->first_token,s->count);
    for(uint32_t i=0;i<s->count;i++){
        if(!isfinite(s->values[i])||s->ids[i]>=FG_Q38_VOCAB_SIZE)
            return malformed(err,"head partial value");
        put_float(out+FG_MTP_HEAD_HEADER_BYTES+i*8u,s->values[i]);
        put32(out+FG_MTP_HEAD_HEADER_BYTES+i*8u+4u,s->ids[i]);
    }
    *bytes=required;return FG_OK;
}
fg_status fg_mtp_head_partial_decode(fg_mtp_head_partial *s,const uint8_t *p,
    uint32_t bytes,fg_error *err){
    if(!s||!p||!head_header_valid(p,bytes)||
       bytes!=FG_MTP_HEAD_HEADER_BYTES+get32(p+8u)*8u)
        return malformed(err,"head partial header");
    *s=(fg_mtp_head_partial){.kind=(fg_mtp_head_kind)p[1],.source_rank=p[2],
        .destination_rank=p[3],.first_token=get32(p+4u),.count=get32(p+8u)};
    for(uint32_t i=0;i<s->count;i++){
        s->values[i]=get_float(p+FG_MTP_HEAD_HEADER_BYTES+i*8u);
        s->ids[i]=get32(p+FG_MTP_HEAD_HEADER_BYTES+i*8u+4u);
        if(!isfinite(s->values[i])||s->ids[i]>=FG_Q38_VOCAB_SIZE)
            return malformed(err,"head partial value");
    }
    return FG_OK;
}
