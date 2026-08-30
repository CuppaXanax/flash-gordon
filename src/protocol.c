#include "fg_protocol.h"
#include "fg_q38_schema.h"
#include "fg_topology.h"

#include <arpa/inet.h>
#include <math.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#endif

static uint16_t bswap16(uint16_t x){return (uint16_t)((x>>8)|(x<<8));}
static uint64_t ntoh64_halves(uint32_t hi,uint32_t lo){return ((uint64_t)ntohl(hi)<<32)|ntohl(lo);}

bool fg_protocol_version_supported(uint16_t version){
    return version>=FG_PROTOCOL_MIN_VERSION&&version<=FG_PROTOCOL_VERSION;
}

static uint16_t max_message_type(uint16_t version){
    return version>=6u?FG_MSG_SESSION_RESTORED:FG_MSG_NGRAM_RESULT;
}

uint64_t fg_token_hash_update(uint64_t h,const int32_t *tokens,size_t count){if(h==0)h=UINT64_C(1469598103934665603);for(size_t i=0;i<count;i++){uint32_t x=(uint32_t)tokens[i];for(unsigned b=0;b<4;b++){h^=(uint8_t)(x>>(b*8));h*=UINT64_C(1099511628211);}}return h;}

static uint32_t crc32c_portable(const void *data,size_t bytes){const uint8_t *p=data;uint32_t crc=~0u;while(bytes--){crc^=*p++;for(unsigned k=0;k<8;k++)crc=(crc>>1)^(0x82f63b78u&-(int32_t)(crc&1u));}return ~crc;}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse4.2")))
static uint32_t crc32c_sse42(const void *data,size_t bytes){
    const uint8_t *p=data;uint64_t crc=UINT32_MAX;
    while(bytes>=8u){uint64_t word;memcpy(&word,p,sizeof(word));crc=_mm_crc32_u64(crc,word);p+=8u;bytes-=8u;}
    uint32_t tail=(uint32_t)crc;while(bytes--){tail=_mm_crc32_u8(tail,*p++);}return ~tail;
}
#endif

uint32_t fg_crc32c(const void *data,size_t bytes){
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if(__builtin_cpu_supports("sse4.2"))return crc32c_sse42(data,bytes);
#endif
    return crc32c_portable(data,bytes);
}

fg_status fg_frame_encode_version(fg_frame_header *h,uint16_t version,fg_message_type type,uint64_t request_id,uint32_t sequence,uint32_t flags,const void *payload,uint32_t bytes,fg_error *err){
    if(!h||!fg_protocol_version_supported(version)||(bytes&&!payload)||bytes>FG_MAX_FRAME_BYTES||
       type<FG_MSG_HELLO||type>max_message_type(version)){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid frame arguments for protocol %u",version);
        return FG_ERR_ARGUMENT;
    }
    memset(h,0,sizeof(*h));h->magic_be=htonl(FG_FRAME_MAGIC);h->version_be=bswap16(version);h->type_be=bswap16((uint16_t)type);h->bytes_be=htonl(bytes);h->request_hi_be=htonl((uint32_t)(request_id>>32));h->request_lo_be=htonl((uint32_t)request_id);h->sequence_be=htonl(sequence);h->flags_be=htonl(flags);h->crc32c_be=htonl(fg_crc32c(payload,bytes));return FG_OK;
}
fg_status fg_frame_encode(fg_frame_header *h,fg_message_type type,uint64_t request_id,uint32_t sequence,uint32_t flags,const void *payload,uint32_t bytes,fg_error *err){
    return fg_frame_encode_version(h,FG_PROTOCOL_VERSION,type,request_id,sequence,flags,payload,bytes,err);
}
fg_status fg_frame_validate(const fg_frame_header *h,const void *payload,uint32_t *payload_bytes,fg_error *err){
    if(!h){fg_error_set(err,FG_ERR_ARGUMENT,"frame header is null");return FG_ERR_ARGUMENT;}uint32_t bytes=ntohl(h->bytes_be);
    uint16_t version=bswap16(h->version_be);
    if(ntohl(h->magic_be)!=FG_FRAME_MAGIC||!fg_protocol_version_supported(version)){fg_error_set(err,FG_ERR_MISMATCH,"frame magic or protocol %u mismatch",version);return FG_ERR_MISMATCH;}
    uint16_t type=bswap16(h->type_be);if(type<FG_MSG_HELLO||type>max_message_type(version)||bytes>FG_MAX_FRAME_BYTES||(bytes&&!payload)){fg_error_set(err,FG_ERR_FORMAT,"invalid frame type or length for protocol %u",version);return FG_ERR_FORMAT;}
    if(fg_crc32c(payload,bytes)!=ntohl(h->crc32c_be)){fg_error_set(err,FG_ERR_MISMATCH,"frame CRC32C mismatch for request %llu",(unsigned long long)ntoh64_halves(h->request_hi_be,h->request_lo_be));return FG_ERR_MISMATCH;}if(payload_bytes)*payload_bytes=bytes;return FG_OK;
}
fg_status fg_frame_validate_version(const fg_frame_header *header,uint16_t protocol_version,
                                    const void *payload,uint32_t *payload_bytes,fg_error *err){
    if(!fg_protocol_version_supported(protocol_version)){
        fg_error_set(err,FG_ERR_ARGUMENT,"unsupported expected protocol %u",protocol_version);
        return FG_ERR_ARGUMENT;
    }
    if(!header){
        fg_error_set(err,FG_ERR_ARGUMENT,"frame header is null");
        return FG_ERR_ARGUMENT;
    }
    if(fg_frame_version(header)!=protocol_version){
        fg_error_set(err,FG_ERR_MISMATCH,
                     "frame protocol %u does not match negotiated protocol %u",
                     fg_frame_version(header),protocol_version);
        return FG_ERR_MISMATCH;
    }
    return fg_frame_validate(header,payload,payload_bytes,err);
}
uint16_t fg_frame_version(const fg_frame_header *header){return header?bswap16(header->version_be):0;}
fg_message_type fg_frame_type(const fg_frame_header *header){return header?(fg_message_type)bswap16(header->type_be):0;}
uint64_t fg_frame_request_id(const fg_frame_header *header){return header?ntoh64_halves(header->request_hi_be,header->request_lo_be):0;}
uint32_t fg_frame_sequence(const fg_frame_header *header){return header?ntohl(header->sequence_be):0;}

static void put_u16_be(uint8_t *p,uint16_t v){p[0]=(uint8_t)(v>>8u);p[1]=(uint8_t)v;}
static uint16_t get_u16_be(const uint8_t *p){return(uint16_t)(((uint16_t)p[0]<<8u)|p[1]);}
static void put_u32_be(uint8_t *p,uint32_t v){p[0]=(uint8_t)(v>>24u);p[1]=(uint8_t)(v>>16u);p[2]=(uint8_t)(v>>8u);p[3]=(uint8_t)v;}
static uint32_t get_u32_be(const uint8_t *p){return((uint32_t)p[0]<<24u)|((uint32_t)p[1]<<16u)|((uint32_t)p[2]<<8u)|p[3];}
static void put_u64_be(uint8_t *p,uint64_t value){put_u32_be(p,(uint32_t)(value>>32u));put_u32_be(p+4u,(uint32_t)value);}
static uint64_t get_u64_be(const uint8_t *p){return((uint64_t)get_u32_be(p)<<32u)|get_u32_be(p+4u);}
static void put_f32_be(uint8_t *p,float value){union{float f;uint32_t u;}bits={value};put_u32_be(p,bits.u);}
static float get_f32_be(const uint8_t *p){union{uint32_t u;float f;}bits={get_u32_be(p)};return bits.f;}

static fg_status validate_decode_work(const fg_decode_work *w,fg_error *err){
    if(!w||w->layer>=FG_LAYER_COUNT||w->source_rank>=FG_RANK_COUNT||w->destination_rank>=FG_RANK_COUNT||
       w->selected_count==0||w->selected_count>FG_TOP_K){fg_error_set(err,FG_ERR_FORMAT,"invalid decode work header");return FG_ERR_FORMAT;}
    bool slots[FG_TOP_K]={0},experts[FG_EXPERT_COUNT]={0};
    for(uint32_t i=0;i<w->selected_count;i++){
        if(w->expert_ids[i]>=FG_EXPERT_COUNT||experts[w->expert_ids[i]]||w->routing_slots[i]>=FG_TOP_K||slots[w->routing_slots[i]]||!isfinite(w->gates[i])){
            fg_error_set(err,FG_ERR_FORMAT,"invalid decode route entry %u",i);return FG_ERR_FORMAT;
        }
        slots[w->routing_slots[i]]=true;experts[w->expert_ids[i]]=true;
    }
    return FG_OK;
}

fg_status fg_decode_work_encode(uint8_t out[FG_DECODE_WORK_BYTES],const fg_decode_work *w,fg_error *err){
    if(!out){fg_error_set(err,FG_ERR_ARGUMENT,"decode work output is null");return FG_ERR_ARGUMENT;}
    fg_status rc=validate_decode_work(w,err);if(rc!=FG_OK)return rc;
    memset(out,0,FG_DECODE_WORK_BYTES);out[0]=w->layer;out[1]=w->source_rank;out[2]=w->destination_rank;out[3]=w->selected_count;put_u32_be(out+4,w->position);
    uint32_t ids=8u,slots=ids+FG_TOP_K*2u,gates=slots+FG_TOP_K,activation=gates+FG_TOP_K*4u;
    for(uint32_t i=0;i<FG_TOP_K;i++){put_u16_be(out+ids+i*2u,w->expert_ids[i]);out[slots+i]=w->routing_slots[i];union{float f;uint32_t u;}v={w->gates[i]};put_u32_be(out+gates+i*4u,v.u);}
    memcpy(out+activation,w->activation_q8k,FG_Q8K_ACTIVATION_BYTES);return FG_OK;
}

fg_status fg_decode_work_decode(fg_decode_work *w,const uint8_t *p,uint32_t bytes,fg_error *err){
    if(!w||!p){fg_error_set(err,FG_ERR_ARGUMENT,"decode work input is null");return FG_ERR_ARGUMENT;}
    if(bytes!=FG_DECODE_WORK_BYTES){fg_error_set(err,FG_ERR_FORMAT,"decode work payload is %u bytes, expected %u",bytes,FG_DECODE_WORK_BYTES);return FG_ERR_FORMAT;}
    memset(w,0,sizeof(*w));w->layer=p[0];w->source_rank=p[1];w->destination_rank=p[2];w->selected_count=p[3];w->position=get_u32_be(p+4);
    uint32_t ids=8u,slots=ids+FG_TOP_K*2u,gates=slots+FG_TOP_K,activation=gates+FG_TOP_K*4u;
    for(uint32_t i=0;i<FG_TOP_K;i++){w->expert_ids[i]=get_u16_be(p+ids+i*2u);w->routing_slots[i]=p[slots+i];union{uint32_t u;float f;}v={get_u32_be(p+gates+i*4u)};w->gates[i]=v.f;}
    memcpy(w->activation_q8k,p+activation,FG_Q8K_ACTIVATION_BYTES);return validate_decode_work(w,err);
}

static fg_status validate_expert_result(const fg_expert_result *result,fg_error *err){
    if(!result||result->layer>=FG_LAYER_COUNT||result->source_rank>=FG_RANK_COUNT||result->destination_rank>=FG_RANK_COUNT||result->selected_count==0||result->selected_count>FG_TOP_K){fg_error_set(err,FG_ERR_FORMAT,"invalid expert result header");return FG_ERR_FORMAT;}
    bool seen[FG_TOP_K]={0};for(uint32_t i=0;i<result->selected_count;i++){uint32_t slot=result->routing_slots[i];if(slot==0xFFu){/* pre-reduced sentinel — skip slot validation */}else if(slot>=FG_TOP_K||seen[slot]){fg_error_set(err,FG_ERR_FORMAT,"invalid expert result slot %u",slot);return FG_ERR_FORMAT;}else{seen[slot]=true;}for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++)if(!isfinite(result->outputs[i][j])){fg_error_set(err,FG_ERR_FORMAT,"non-finite expert result at slot %u element %u",slot,j);return FG_ERR_FORMAT;}}
    return FG_OK;
}

fg_status fg_expert_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_expert_result *result,fg_error *err){
    if(!output||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"expert result output is null");return FG_ERR_ARGUMENT;}fg_status status=validate_expert_result(result,err);if(status!=FG_OK)return status;uint32_t required=8u+result->selected_count*FG_EXPERT_RESULT_ENTRY_BYTES;if(capacity<required){fg_error_set(err,FG_ERR_LIMIT,"expert result buffer is too small");return FG_ERR_LIMIT;}memset(output,0,required);output[0]=result->layer;output[1]=result->source_rank;output[2]=result->destination_rank;output[3]=result->selected_count;put_u32_be(output+4,result->position);uint32_t offset=8u;for(uint32_t i=0;i<result->selected_count;i++){output[offset]=result->routing_slots[i];offset+=4u;for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++){union{float f;uint32_t u;}value={result->outputs[i][j]};put_u32_be(output+offset,value.u);offset+=4u;}}*bytes=required;return FG_OK;
}

fg_status fg_expert_result_decode(fg_expert_result *result,const uint8_t *payload,uint32_t bytes,fg_error *err){
    if(!result||!payload||bytes<8u){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert result input");return FG_ERR_ARGUMENT;}uint32_t count=payload[3],required=count<=FG_TOP_K?8u+count*FG_EXPERT_RESULT_ENTRY_BYTES:0u;if(!required||bytes!=required){fg_error_set(err,FG_ERR_FORMAT,"invalid expert result payload size");return FG_ERR_FORMAT;}memset(result,0,sizeof(*result));result->layer=payload[0];result->source_rank=payload[1];result->destination_rank=payload[2];result->selected_count=(uint8_t)count;result->position=get_u32_be(payload+4);uint32_t offset=8u;for(uint32_t i=0;i<count;i++){result->routing_slots[i]=payload[offset];if(payload[offset+1u]||payload[offset+2u]||payload[offset+3u]){fg_error_set(err,FG_ERR_FORMAT,"expert result reserved bytes are non-zero");return FG_ERR_FORMAT;}offset+=4u;for(uint32_t j=0;j<FG_HIDDEN_SIZE;j++){union{uint32_t u;float f;}value={get_u32_be(payload+offset)};result->outputs[i][j]=value.f;offset+=4u;}}return validate_expert_result(result,err);
}

static fg_status validate_prefill_work(const fg_prefill_work *work,fg_error *err){
    if(!work||work->layer>=FG_LAYER_COUNT||work->source_rank>=FG_RANK_COUNT||work->destination_rank>=FG_RANK_COUNT||
       !work->token_count||work->token_count>FG_PREFILL_MAX_TOKENS||!work->pair_count||
       work->pair_count>(uint32_t)work->token_count*FG_TOP_K||!work->activations_q8k||!work->pairs){
        fg_error_set(err,FG_ERR_FORMAT,"invalid prefill work header");return FG_ERR_FORMAT;
    }
    bool seen_slots[FG_PREFILL_MAX_TOKENS][FG_TOP_K]={{false}};
    uint64_t seen_experts[FG_PREFILL_MAX_TOKENS][FG_EXPERT_COUNT/64u]={{0}};
    for(uint32_t i=0;i<work->pair_count;i++){
        const fg_prefill_pair *pair=&work->pairs[i];
        if(pair->token_slot>=work->token_count||pair->expert_id>=FG_EXPERT_COUNT||
           pair->routing_slot>=FG_TOP_K||!isfinite(pair->gate)){
            fg_error_set(err,FG_ERR_FORMAT,"invalid prefill route pair %u",i);return FG_ERR_FORMAT;
        }
        uint64_t bit=UINT64_C(1)<<(pair->expert_id&63u);
        if(seen_slots[pair->token_slot][pair->routing_slot]||
           (seen_experts[pair->token_slot][pair->expert_id>>6u]&bit)){
            fg_error_set(err,FG_ERR_FORMAT,"duplicate prefill route for token %u",pair->token_slot);return FG_ERR_FORMAT;
        }
        seen_slots[pair->token_slot][pair->routing_slot]=true;
        seen_experts[pair->token_slot][pair->expert_id>>6u]|=bit;
    }
    return FG_OK;
}

fg_status fg_prefill_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_prefill_work *work,fg_error *err){
    if(!output||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"prefill work output is null");return FG_ERR_ARGUMENT;}
    fg_status status=validate_prefill_work(work,err);if(status!=FG_OK)return status;
    uint32_t activation_bytes=(uint32_t)work->token_count*FG_Q8K_ACTIVATION_BYTES;
    uint32_t required=FG_PREFILL_WORK_HEADER_BYTES+activation_bytes+(uint32_t)work->pair_count*FG_PREFILL_PAIR_BYTES;
    if(required>FG_MAX_FRAME_BYTES||capacity<required){fg_error_set(err,FG_ERR_LIMIT,"prefill work buffer is too small");return FG_ERR_LIMIT;}
    memset(output,0,required);output[0]=work->layer;output[1]=work->source_rank;output[2]=work->destination_rank;
    put_u32_be(output+4u,work->first_position);put_u16_be(output+8u,work->token_count);put_u16_be(output+10u,work->pair_count);put_u32_be(output+12u,activation_bytes);
    memcpy(output+FG_PREFILL_WORK_HEADER_BYTES,work->activations_q8k,activation_bytes);
    uint32_t offset=FG_PREFILL_WORK_HEADER_BYTES+activation_bytes;
    for(uint32_t i=0;i<work->pair_count;i++,offset+=FG_PREFILL_PAIR_BYTES){const fg_prefill_pair *pair=&work->pairs[i];put_u16_be(output+offset,pair->token_slot);put_u16_be(output+offset+2u,pair->expert_id);output[offset+4u]=pair->routing_slot;put_f32_be(output+offset+8u,pair->gate);}
    *bytes=required;return FG_OK;
}

fg_status fg_prefill_work_decode(fg_prefill_work *work,uint8_t *activation_storage,uint32_t activation_capacity,fg_prefill_pair *pair_storage,uint32_t pair_capacity,const uint8_t *payload,uint32_t bytes,fg_error *err){
    if(!work||!activation_storage||!pair_storage||!payload||bytes<FG_PREFILL_WORK_HEADER_BYTES){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill work input");return FG_ERR_ARGUMENT;}
    uint16_t token_count=get_u16_be(payload+8u),pair_count=get_u16_be(payload+10u);uint32_t activation_bytes=get_u32_be(payload+12u);
    uint64_t required=(uint64_t)FG_PREFILL_WORK_HEADER_BYTES+activation_bytes+(uint64_t)pair_count*FG_PREFILL_PAIR_BYTES;
    if(payload[3]||!token_count||token_count>FG_PREFILL_MAX_TOKENS||!pair_count||pair_count>(uint32_t)token_count*FG_TOP_K||
       activation_bytes!=(uint32_t)token_count*FG_Q8K_ACTIVATION_BYTES||required!=bytes||activation_capacity<activation_bytes||pair_capacity<pair_count){
        fg_error_set(err,FG_ERR_FORMAT,"invalid prefill work payload size or header");return FG_ERR_FORMAT;
    }
    memset(work,0,sizeof(*work));work->layer=payload[0];work->source_rank=payload[1];work->destination_rank=payload[2];work->first_position=get_u32_be(payload+4u);work->token_count=token_count;work->pair_count=pair_count;work->activations_q8k=activation_storage;work->pairs=pair_storage;
    memcpy(activation_storage,payload+FG_PREFILL_WORK_HEADER_BYTES,activation_bytes);uint32_t offset=FG_PREFILL_WORK_HEADER_BYTES+activation_bytes;
    for(uint32_t i=0;i<pair_count;i++,offset+=FG_PREFILL_PAIR_BYTES){fg_prefill_pair *pair=&pair_storage[i];pair->token_slot=get_u16_be(payload+offset);pair->expert_id=get_u16_be(payload+offset+2u);pair->routing_slot=payload[offset+4u];if(payload[offset+5u]||payload[offset+6u]||payload[offset+7u]){fg_error_set(err,FG_ERR_FORMAT,"non-zero prefill pair reserved bytes");return FG_ERR_FORMAT;}pair->gate=get_f32_be(payload+offset+8u);}
    return validate_prefill_work(work,err);
}

static fg_status validate_prefill_result(const fg_prefill_result *result,fg_error *err){
    if(!result||result->layer>=FG_LAYER_COUNT||result->source_rank>=FG_RANK_COUNT||result->destination_rank>=FG_RANK_COUNT||
       !result->token_count||result->token_count>FG_PREFILL_MAX_TOKENS||!result->pair_count||
       result->pair_count>(uint32_t)result->token_count*FG_TOP_K||!result->pairs||!result->outputs){
        fg_error_set(err,FG_ERR_FORMAT,"invalid prefill result header");return FG_ERR_FORMAT;
    }
    bool seen[FG_PREFILL_MAX_TOKENS][FG_TOP_K]={{false}};
    for(uint32_t i=0;i<result->pair_count;i++){
        const fg_prefill_result_pair *pair=&result->pairs[i];
        if(pair->token_slot>=result->token_count||pair->routing_slot>=FG_TOP_K||seen[pair->token_slot][pair->routing_slot]){
            fg_error_set(err,FG_ERR_FORMAT,"invalid prefill result pair %u",i);return FG_ERR_FORMAT;
        }
        seen[pair->token_slot][pair->routing_slot]=true;
        for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)if(!isfinite(result->outputs[(uint64_t)i*FG_HIDDEN_SIZE+element])){
            fg_error_set(err,FG_ERR_FORMAT,"non-finite prefill result pair %u element %u",i,element);return FG_ERR_FORMAT;
        }
    }
    return FG_OK;
}

fg_status fg_prefill_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_prefill_result *result,fg_error *err){
    if(!output||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"prefill result output is null");return FG_ERR_ARGUMENT;}fg_status status=validate_prefill_result(result,err);if(status!=FG_OK)return status;
    uint32_t required=FG_PREFILL_RESULT_HEADER_BYTES+(uint32_t)result->pair_count*FG_PREFILL_RESULT_PAIR_BYTES;if(required>FG_MAX_FRAME_BYTES||capacity<required){fg_error_set(err,FG_ERR_LIMIT,"prefill result buffer is too small");return FG_ERR_LIMIT;}
    memset(output,0,required);output[0]=result->layer;output[1]=result->source_rank;output[2]=result->destination_rank;put_u32_be(output+4u,result->first_position);put_u16_be(output+8u,result->token_count);put_u16_be(output+10u,result->pair_count);
    uint32_t offset=FG_PREFILL_RESULT_HEADER_BYTES;for(uint32_t i=0;i<result->pair_count;i++,offset+=FG_PREFILL_RESULT_PAIR_BYTES){put_u16_be(output+offset,result->pairs[i].token_slot);output[offset+2u]=result->pairs[i].routing_slot;for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)put_f32_be(output+offset+4u+element*4u,result->outputs[(uint64_t)i*FG_HIDDEN_SIZE+element]);}
    *bytes=required;return FG_OK;
}

fg_status fg_prefill_result_decode(fg_prefill_result *result,fg_prefill_result_pair *pair_storage,uint32_t pair_capacity,float *output_storage,uint64_t output_capacity_values,const uint8_t *payload,uint32_t bytes,fg_error *err){
    if(!result||!pair_storage||!output_storage||!payload||bytes<FG_PREFILL_RESULT_HEADER_BYTES){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill result input");return FG_ERR_ARGUMENT;}
    uint16_t token_count=get_u16_be(payload+8u),pair_count=get_u16_be(payload+10u);uint64_t required=(uint64_t)FG_PREFILL_RESULT_HEADER_BYTES+(uint64_t)pair_count*FG_PREFILL_RESULT_PAIR_BYTES;
    if(payload[3]||get_u32_be(payload+12u)||!token_count||token_count>FG_PREFILL_MAX_TOKENS||!pair_count||pair_count>(uint32_t)token_count*FG_TOP_K||required!=bytes||pair_capacity<pair_count||output_capacity_values<(uint64_t)pair_count*FG_HIDDEN_SIZE){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill result payload size or header");return FG_ERR_FORMAT;}
    memset(result,0,sizeof(*result));result->layer=payload[0];result->source_rank=payload[1];result->destination_rank=payload[2];result->first_position=get_u32_be(payload+4u);result->token_count=token_count;result->pair_count=pair_count;result->pairs=pair_storage;result->outputs=output_storage;
    uint32_t offset=FG_PREFILL_RESULT_HEADER_BYTES;for(uint32_t i=0;i<pair_count;i++,offset+=FG_PREFILL_RESULT_PAIR_BYTES){pair_storage[i].token_slot=get_u16_be(payload+offset);pair_storage[i].routing_slot=payload[offset+2u];if(payload[offset+3u]){fg_error_set(err,FG_ERR_FORMAT,"non-zero prefill result pair reserved byte");return FG_ERR_FORMAT;}for(uint32_t element=0;element<FG_HIDDEN_SIZE;element++)output_storage[(uint64_t)i*FG_HIDDEN_SIZE+element]=get_f32_be(payload+offset+4u+element*4u);}
    return validate_prefill_result(result,err);
}

static fg_status validate_prefill_layer_work(const fg_prefill_layer_work *work,fg_error *err){
    if(!work||work->layer>=FG_LAYER_COUNT||work->source_rank>=FG_RANK_COUNT||work->destination_rank>=FG_RANK_COUNT||(work->flags&~FG_LAYER_WORK_HAS_NGRAM)||((work->layer==1u)!=((work->flags&FG_LAYER_WORK_HAS_NGRAM)!=0))||work->position_mode>FG_POSITION_FOUR_AXIS||!work->token_count||work->token_count>FG_PREFILL_MAX_TOKENS||!work->positions||!work->hyper||((work->flags&FG_LAYER_WORK_HAS_NGRAM)&&!work->ngram_embeddings)){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill layer work header");return FG_ERR_FORMAT;}uint64_t hyper_values=(uint64_t)work->token_count*FG_HYPER_WIDTH;for(uint64_t i=0;i<hyper_values;i++)if(!isfinite(work->hyper[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite prefill layer input at %llu",(unsigned long long)i);return FG_ERR_FORMAT;}if(work->flags&FG_LAYER_WORK_HAS_NGRAM){uint64_t ngram_values=(uint64_t)work->token_count*FG_NGRAM_EMBED_VALUES;for(uint64_t i=0;i<ngram_values;i++)if(!isfinite(work->ngram_embeddings[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite batched n-gram embedding at %llu",(unsigned long long)i);return FG_ERR_FORMAT;}}return FG_OK;
}

fg_status fg_prefill_layer_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,uint16_t protocol_version,const fg_prefill_layer_work *work,fg_error *err){
    if(!output||!bytes||!fg_protocol_version_supported(protocol_version)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill layer work output or protocol");return FG_ERR_ARGUMENT;}fg_status status=validate_prefill_layer_work(work,err);if(status!=FG_OK)return status;if(protocol_version==FG_PROTOCOL_MIN_VERSION&&work->position_mode!=FG_POSITION_TEXT){fg_error_set(err,FG_ERR_MISMATCH,"protocol 5 prefill layer work requires text positions");return FG_ERR_MISMATCH;}uint32_t axes=work->position_mode==FG_POSITION_FOUR_AXIS?4u:3u;uint64_t position_bytes=(uint64_t)work->token_count*axes*4u,hyper_bytes=(uint64_t)work->token_count*FG_HYPER_WIDTH*4u,ngram_bytes=(work->flags&FG_LAYER_WORK_HAS_NGRAM)?(uint64_t)work->token_count*FG_NGRAM_EMBED_VALUES*4u:0u,required=FG_PREFILL_LAYER_HEADER_BYTES+position_bytes+hyper_bytes+ngram_bytes;if(required>FG_MAX_FRAME_BYTES||required>capacity){fg_error_set(err,FG_ERR_LIMIT,"prefill layer work buffer is too small");return FG_ERR_LIMIT;}memset(output,0,FG_PREFILL_LAYER_HEADER_BYTES);output[0]=work->layer;output[1]=work->source_rank;output[2]=work->destination_rank;output[3]=work->flags;put_u32_be(output+4u,work->first_token);put_u16_be(output+8u,work->token_count);if(protocol_version>=FG_PROTOCOL_VERSION){output[10]=(uint8_t)work->position_mode;output[11]=(uint8_t)axes;}uint32_t offset=FG_PREFILL_LAYER_HEADER_BYTES;for(uint32_t i=0;i<(uint32_t)work->token_count*axes;i++,offset+=4u)put_u32_be(output+offset,work->positions[i]);for(uint64_t i=0;i<(uint64_t)work->token_count*FG_HYPER_WIDTH;i++,offset+=4u)put_f32_be(output+offset,work->hyper[i]);if(work->flags&FG_LAYER_WORK_HAS_NGRAM)for(uint64_t i=0;i<(uint64_t)work->token_count*FG_NGRAM_EMBED_VALUES;i++,offset+=4u)put_f32_be(output+offset,work->ngram_embeddings[i]);*bytes=(uint32_t)required;return FG_OK;
}

fg_status fg_prefill_layer_work_decode(fg_prefill_layer_work *work,uint16_t protocol_version,uint32_t *position_storage,uint32_t position_capacity,float *hyper_storage,uint64_t hyper_capacity_values,float *ngram_storage,uint64_t ngram_capacity_values,const uint8_t *payload,uint32_t bytes,fg_error *err){
    if(!work||!position_storage||!hyper_storage||!payload||bytes<FG_PREFILL_LAYER_HEADER_BYTES||!fg_protocol_version_supported(protocol_version)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill layer work input or protocol");return FG_ERR_ARGUMENT;}uint8_t flags=payload[3];uint16_t tokens=get_u16_be(payload+8u);fg_position_mode mode=FG_POSITION_TEXT;uint32_t axes=3u;if(protocol_version==FG_PROTOCOL_MIN_VERSION){if(payload[10]||payload[11]){fg_error_set(err,FG_ERR_FORMAT,"protocol 5 prefill position header is not legacy");return FG_ERR_FORMAT;}}else{mode=(fg_position_mode)payload[10];axes=payload[11];if(mode>FG_POSITION_FOUR_AXIS||axes!=(mode==FG_POSITION_FOUR_AXIS?4u:3u)){fg_error_set(err,FG_ERR_FORMAT,"invalid protocol 6 prefill position contract");return FG_ERR_FORMAT;}}uint64_t position_values=(uint64_t)tokens*axes,hyper_values=(uint64_t)tokens*FG_HYPER_WIDTH,ngram_values=(flags&FG_LAYER_WORK_HAS_NGRAM)?(uint64_t)tokens*FG_NGRAM_EMBED_VALUES:0u,required=FG_PREFILL_LAYER_HEADER_BYTES+(position_values+hyper_values+ngram_values)*4u;if(get_u32_be(payload+12u)||!tokens||tokens>FG_PREFILL_MAX_TOKENS||required!=bytes||position_capacity<position_values||hyper_capacity_values<hyper_values||(ngram_values&&(!ngram_storage||ngram_capacity_values<ngram_values))){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill layer work size, reserved bytes, or storage capacity");return FG_ERR_FORMAT;}memset(work,0,sizeof(*work));work->layer=payload[0];work->source_rank=payload[1];work->destination_rank=payload[2];work->flags=flags;work->position_mode=mode;work->first_token=get_u32_be(payload+4u);work->token_count=tokens;work->positions=position_storage;work->hyper=hyper_storage;work->ngram_embeddings=ngram_values?ngram_storage:NULL;uint32_t offset=FG_PREFILL_LAYER_HEADER_BYTES;for(uint32_t i=0;i<(uint32_t)position_values;i++,offset+=4u)position_storage[i]=get_u32_be(payload+offset);for(uint64_t i=0;i<hyper_values;i++,offset+=4u)hyper_storage[i]=get_f32_be(payload+offset);for(uint64_t i=0;i<ngram_values;i++,offset+=4u)ngram_storage[i]=get_f32_be(payload+offset);return validate_prefill_layer_work(work,err);
}

static fg_status validate_prefill_layer_result(const fg_prefill_layer_result *result,fg_error *err){if(!result||result->layer>=FG_LAYER_COUNT||result->source_rank>=FG_RANK_COUNT||result->destination_rank>=FG_RANK_COUNT||!result->token_count||result->token_count>FG_PREFILL_MAX_TOKENS||!result->hyper){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill layer result header");return FG_ERR_FORMAT;}uint64_t values=(uint64_t)result->token_count*FG_HYPER_WIDTH;for(uint64_t i=0;i<values;i++)if(!isfinite(result->hyper[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite prefill layer result at %llu",(unsigned long long)i);return FG_ERR_FORMAT;}return FG_OK;}

fg_status fg_prefill_layer_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_prefill_layer_result *result,fg_error *err){if(!output||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"prefill layer result output is null");return FG_ERR_ARGUMENT;}fg_status status=validate_prefill_layer_result(result,err);if(status!=FG_OK)return status;uint64_t required=FG_PREFILL_LAYER_HEADER_BYTES+(uint64_t)result->token_count*FG_HYPER_WIDTH*4u;if(required>FG_MAX_FRAME_BYTES||required>capacity){fg_error_set(err,FG_ERR_LIMIT,"prefill layer result buffer is too small");return FG_ERR_LIMIT;}memset(output,0,FG_PREFILL_LAYER_HEADER_BYTES);output[0]=result->layer;output[1]=result->source_rank;output[2]=result->destination_rank;put_u32_be(output+4u,result->first_token);put_u16_be(output+8u,result->token_count);uint32_t offset=FG_PREFILL_LAYER_HEADER_BYTES;for(uint64_t i=0;i<(uint64_t)result->token_count*FG_HYPER_WIDTH;i++,offset+=4u)put_f32_be(output+offset,result->hyper[i]);*bytes=(uint32_t)required;return FG_OK;}

fg_status fg_prefill_layer_result_decode(fg_prefill_layer_result *result,float *hyper_storage,uint64_t hyper_capacity_values,const uint8_t *payload,uint32_t bytes,fg_error *err){if(!result||!hyper_storage||!payload||bytes<FG_PREFILL_LAYER_HEADER_BYTES){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill layer result input");return FG_ERR_ARGUMENT;}uint16_t tokens=get_u16_be(payload+8u);uint64_t values=(uint64_t)tokens*FG_HYPER_WIDTH,required=FG_PREFILL_LAYER_HEADER_BYTES+values*4u;if(payload[3]||payload[10]||payload[11]||get_u32_be(payload+12u)||!tokens||tokens>FG_PREFILL_MAX_TOKENS||required!=bytes||hyper_capacity_values<values){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill layer result size, reserved bytes, or storage capacity");return FG_ERR_FORMAT;}memset(result,0,sizeof(*result));result->layer=payload[0];result->source_rank=payload[1];result->destination_rank=payload[2];result->first_token=get_u32_be(payload+4u);result->token_count=tokens;result->hyper=hyper_storage;uint32_t offset=FG_PREFILL_LAYER_HEADER_BYTES;for(uint64_t i=0;i<values;i++,offset+=4u)hyper_storage[i]=get_f32_be(payload+offset);return validate_prefill_layer_result(result,err);}

static fg_status validate_layer_work(const fg_layer_work *work,fg_error *err){
    if(!work||work->layer>=FG_LAYER_COUNT||work->source_rank>=FG_RANK_COUNT||work->destination_rank>=FG_RANK_COUNT||(work->flags&~FG_LAYER_WORK_HAS_NGRAM)||((work->layer==1u)!=((work->flags&FG_LAYER_WORK_HAS_NGRAM)!=0))||work->position_mode>FG_POSITION_FOUR_AXIS||(work->position_mode==FG_POSITION_TEXT&&work->position[3])){fg_error_set(err,FG_ERR_FORMAT,"invalid layer work header");return FG_ERR_FORMAT;}for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)if(!isfinite(work->hyper[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite layer input at %u",i);return FG_ERR_FORMAT;}if(work->flags&FG_LAYER_WORK_HAS_NGRAM)for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)if(!isfinite(work->ngram_embedding[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite n-gram embedding at %u",i);return FG_ERR_FORMAT;}return FG_OK;
}

fg_status fg_layer_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,uint16_t protocol_version,const fg_layer_work *work,fg_error *err){
    if(!output||!bytes||!fg_protocol_version_supported(protocol_version)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid layer work output or protocol");return FG_ERR_ARGUMENT;}fg_status status=validate_layer_work(work,err);if(status!=FG_OK)return status;if(protocol_version==FG_PROTOCOL_MIN_VERSION&&work->position_mode!=FG_POSITION_TEXT){fg_error_set(err,FG_ERR_MISMATCH,"protocol 5 layer work requires text positions");return FG_ERR_MISMATCH;}uint32_t axes=work->position_mode==FG_POSITION_FOUR_AXIS?4u:3u,header=protocol_version==FG_PROTOCOL_MIN_VERSION?FG_LAYER_WORK_LEGACY_HEADER_BYTES:12u+axes*4u,position_offset=protocol_version==FG_PROTOCOL_MIN_VERSION?8u:12u,required=header+FG_HYPER_WIDTH*4u+((work->flags&FG_LAYER_WORK_HAS_NGRAM)?FG_NGRAM_EMBED_VALUES*4u:0u);if(capacity<required){fg_error_set(err,FG_ERR_LIMIT,"layer work buffer is too small");return FG_ERR_LIMIT;}memset(output,0,header);output[0]=work->layer;output[1]=work->source_rank;output[2]=work->destination_rank;output[3]=work->flags;put_u32_be(output+4u,work->token_index);if(protocol_version>=FG_PROTOCOL_VERSION){output[8]=(uint8_t)work->position_mode;output[9]=(uint8_t)axes;}for(uint32_t axis=0;axis<axes;axis++)put_u32_be(output+position_offset+axis*4u,work->position[axis]);uint32_t offset=header;for(uint32_t i=0;i<FG_HYPER_WIDTH;i++,offset+=4u)put_f32_be(output+offset,work->hyper[i]);if(work->flags&FG_LAYER_WORK_HAS_NGRAM)for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++,offset+=4u)put_f32_be(output+offset,work->ngram_embedding[i]);*bytes=required;return FG_OK;
}

fg_status fg_layer_work_decode(fg_layer_work *work,uint16_t protocol_version,const uint8_t *payload,uint32_t bytes,fg_error *err){
    if(!work||!payload||!fg_protocol_version_supported(protocol_version)){fg_error_set(err,FG_ERR_ARGUMENT,"invalid layer work input or protocol");return FG_ERR_ARGUMENT;}uint32_t minimum=protocol_version==FG_PROTOCOL_MIN_VERSION?FG_LAYER_WORK_LEGACY_HEADER_BYTES:FG_LAYER_WORK_TEXT_HEADER_BYTES;if(bytes<minimum+FG_HYPER_WIDTH*4u){fg_error_set(err,FG_ERR_FORMAT,"layer work is shorter than protocol %u minimum",protocol_version);return FG_ERR_FORMAT;}uint8_t flags=payload[3];uint32_t body=FG_HYPER_WIDTH*4u+((flags&FG_LAYER_WORK_HAS_NGRAM)?FG_NGRAM_EMBED_VALUES*4u:0u);if(bytes<body){fg_error_set(err,FG_ERR_FORMAT,"invalid layer work payload size");return FG_ERR_FORMAT;}uint32_t header=bytes-body,axes=3u,position_offset=8u;fg_position_mode mode=FG_POSITION_TEXT;if(protocol_version==FG_PROTOCOL_MIN_VERSION){if(header!=FG_LAYER_WORK_LEGACY_HEADER_BYTES){fg_error_set(err,FG_ERR_FORMAT,"protocol 5 layer work is not legacy layout");return FG_ERR_FORMAT;}}else{if(header!=FG_LAYER_WORK_TEXT_HEADER_BYTES&&header!=FG_LAYER_WORK_FOUR_AXIS_HEADER_BYTES){fg_error_set(err,FG_ERR_FORMAT,"invalid protocol 6 layer work size");return FG_ERR_FORMAT;}mode=(fg_position_mode)payload[8];axes=payload[9];position_offset=12u;if(payload[10]||payload[11]||mode>FG_POSITION_FOUR_AXIS||axes!=(mode==FG_POSITION_FOUR_AXIS?4u:3u)||header!=12u+axes*4u){fg_error_set(err,FG_ERR_FORMAT,"invalid protocol 6 layer position contract");return FG_ERR_FORMAT;}}memset(work,0,sizeof(*work));work->layer=payload[0];work->source_rank=payload[1];work->destination_rank=payload[2];work->flags=flags;work->position_mode=mode;work->token_index=get_u32_be(payload+4u);for(uint32_t axis=0;axis<axes;axis++)work->position[axis]=get_u32_be(payload+position_offset+axis*4u);uint32_t offset=header;for(uint32_t i=0;i<FG_HYPER_WIDTH;i++,offset+=4u)work->hyper[i]=get_f32_be(payload+offset);if(flags&FG_LAYER_WORK_HAS_NGRAM)for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++,offset+=4u)work->ngram_embedding[i]=get_f32_be(payload+offset);return validate_layer_work(work,err);
}

static fg_status validate_layer_result(const fg_layer_result *result,fg_error *err){if(!result||result->layer>=FG_LAYER_COUNT||result->source_rank>=FG_RANK_COUNT||result->destination_rank>=FG_RANK_COUNT){fg_error_set(err,FG_ERR_FORMAT,"invalid layer result header");return FG_ERR_FORMAT;}for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)if(!isfinite(result->hyper[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite layer result at %u",i);return FG_ERR_FORMAT;}return FG_OK;}

fg_status fg_layer_result_encode(uint8_t output[FG_LAYER_RESULT_BYTES],const fg_layer_result *result,fg_error *err){if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"layer result output is null");return FG_ERR_ARGUMENT;}fg_status status=validate_layer_result(result,err);if(status!=FG_OK)return status;output[0]=result->layer;output[1]=result->source_rank;output[2]=result->destination_rank;output[3]=0;put_u32_be(output+4u,result->token_index);for(uint32_t i=0,offset=8u;i<FG_HYPER_WIDTH;i++,offset+=4u)put_f32_be(output+offset,result->hyper[i]);return FG_OK;}

fg_status fg_layer_result_decode(fg_layer_result *result,const uint8_t *payload,uint32_t bytes,fg_error *err){if(!result||!payload){fg_error_set(err,FG_ERR_ARGUMENT,"invalid layer result input");return FG_ERR_ARGUMENT;}if(bytes!=FG_LAYER_RESULT_BYTES||payload[3]){fg_error_set(err,FG_ERR_FORMAT,"invalid layer result payload size or reserved byte");return FG_ERR_FORMAT;}memset(result,0,sizeof(*result));result->layer=payload[0];result->source_rank=payload[1];result->destination_rank=payload[2];result->token_index=get_u32_be(payload+4u);for(uint32_t i=0,offset=8u;i<FG_HYPER_WIDTH;i++,offset+=4u)result->hyper[i]=get_f32_be(payload+offset);return validate_layer_result(result,err);}

static fg_status validate_output_work(const fg_output_work *work,fg_error *err){if(!work||work->source_rank>=FG_RANK_COUNT||work->destination_rank!=4u){fg_error_set(err,FG_ERR_FORMAT,"invalid output work route");return FG_ERR_FORMAT;}for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)if(!isfinite(work->hyper[i])){fg_error_set(err,FG_ERR_FORMAT,"non-finite output input at %u",i);return FG_ERR_FORMAT;}return FG_OK;}

fg_status fg_output_work_encode(uint8_t output[FG_OUTPUT_WORK_BYTES],const fg_output_work *work,fg_error *err){if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"output work buffer is null");return FG_ERR_ARGUMENT;}fg_status status=validate_output_work(work,err);if(status!=FG_OK)return status;output[0]=work->source_rank;output[1]=work->destination_rank;output[2]=0;output[3]=0;put_u32_be(output+4u,work->token_index);for(uint32_t i=0,offset=8u;i<FG_HYPER_WIDTH;i++,offset+=4u)put_f32_be(output+offset,work->hyper[i]);return FG_OK;}

fg_status fg_output_work_decode(fg_output_work *work,const uint8_t *payload,uint32_t bytes,fg_error *err){if(!work||!payload){fg_error_set(err,FG_ERR_ARGUMENT,"invalid output work input");return FG_ERR_ARGUMENT;}if(bytes!=FG_OUTPUT_WORK_BYTES||payload[2]||payload[3]){fg_error_set(err,FG_ERR_FORMAT,"invalid output work size or reserved bytes");return FG_ERR_FORMAT;}memset(work,0,sizeof(*work));work->source_rank=payload[0];work->destination_rank=payload[1];work->token_index=get_u32_be(payload+4u);for(uint32_t i=0,offset=8u;i<FG_HYPER_WIDTH;i++,offset+=4u)work->hyper[i]=get_f32_be(payload+offset);return validate_output_work(work,err);}

static fg_status validate_output_result(const fg_output_result *result,fg_error *err){if(!result||result->source_rank!=4u||result->destination_rank>=FG_RANK_COUNT||result->token>=FG_Q38_VOCAB_SIZE||!isfinite(result->logit)){fg_error_set(err,FG_ERR_FORMAT,"invalid output result");return FG_ERR_FORMAT;}return FG_OK;}

fg_status fg_output_result_encode(uint8_t output[FG_OUTPUT_RESULT_BYTES],const fg_output_result *result,fg_error *err){if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"output result buffer is null");return FG_ERR_ARGUMENT;}fg_status status=validate_output_result(result,err);if(status!=FG_OK)return status;output[0]=result->source_rank;output[1]=result->destination_rank;output[2]=0;output[3]=0;put_u32_be(output+4u,result->token_index);put_u32_be(output+8u,result->token);put_f32_be(output+12u,result->logit);return FG_OK;}

fg_status fg_output_result_decode(fg_output_result *result,const uint8_t *payload,uint32_t bytes,fg_error *err){if(!result||!payload){fg_error_set(err,FG_ERR_ARGUMENT,"invalid output result input");return FG_ERR_ARGUMENT;}if(bytes!=FG_OUTPUT_RESULT_BYTES||payload[2]||payload[3]){fg_error_set(err,FG_ERR_FORMAT,"invalid output result size or reserved bytes");return FG_ERR_FORMAT;}memset(result,0,sizeof(*result));result->source_rank=payload[0];result->destination_rank=payload[1];result->token_index=get_u32_be(payload+4u);result->token=get_u32_be(payload+8u);result->logit=get_f32_be(payload+12u);return validate_output_result(result,err);}

static fg_status validate_ngram_items(uint8_t source,uint8_t destination,uint8_t item_count,const uint8_t heads[FG_NGRAM_SHARD_MAX_ITEMS],fg_error *err){if(source>=FG_RANK_COUNT||destination>=FG_RANK_COUNT||source==destination||!item_count||item_count>FG_NGRAM_SHARD_MAX_ITEMS){fg_error_set(err,FG_ERR_FORMAT,"invalid resident n-gram route");return FG_ERR_FORMAT;}bool seen[FG_NGRAM_HEAD_COUNT]={0};for(uint32_t i=0;i<item_count;i++){uint32_t head=heads[i];if(head>=FG_NGRAM_HEAD_COUNT||seen[head]){fg_error_set(err,FG_ERR_FORMAT,"invalid resident n-gram head %u",head);return FG_ERR_FORMAT;}seen[head]=true;}return FG_OK;}

fg_status fg_ngram_work_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_ngram_work *work,fg_error *err){if(!output||!bytes||!work){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram work output");return FG_ERR_ARGUMENT;}fg_status status=validate_ngram_items(work->source_rank,work->destination_rank,work->item_count,work->heads,err);uint32_t required=8u+(uint32_t)work->item_count*9u;if(status!=FG_OK)return status;if(capacity<required){fg_error_set(err,FG_ERR_LIMIT,"n-gram work buffer is too small");return FG_ERR_LIMIT;}output[0]=work->source_rank;output[1]=work->destination_rank;output[2]=work->item_count;output[3]=0u;put_u32_be(output+4u,work->token_index);for(uint32_t i=0;i<work->item_count;i++){uint32_t offset=8u+i*9u;output[offset]=work->heads[i];put_u64_be(output+offset+1u,work->rows[i]);}*bytes=required;return FG_OK;}

fg_status fg_ngram_work_decode(fg_ngram_work *work,const uint8_t *payload,uint32_t bytes,fg_error *err){if(!work||!payload||bytes<8u){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram work input");return FG_ERR_ARGUMENT;}uint8_t item_count=payload[2];uint32_t required=8u+(uint32_t)item_count*9u;if(payload[3]||!item_count||item_count>FG_NGRAM_SHARD_MAX_ITEMS||bytes!=required){fg_error_set(err,FG_ERR_FORMAT,"invalid n-gram work size");return FG_ERR_FORMAT;}memset(work,0,sizeof(*work));work->source_rank=payload[0];work->destination_rank=payload[1];work->item_count=item_count;work->token_index=get_u32_be(payload+4u);for(uint32_t i=0;i<item_count;i++){uint32_t offset=8u+i*9u;work->heads[i]=payload[offset];work->rows[i]=get_u64_be(payload+offset+1u);}return validate_ngram_items(work->source_rank,work->destination_rank,work->item_count,work->heads,err);}

fg_status fg_ngram_result_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,const fg_ngram_result *result,fg_error *err){if(!output||!bytes||!result){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram result output");return FG_ERR_ARGUMENT;}fg_status status=validate_ngram_items(result->source_rank,result->destination_rank,result->item_count,result->heads,err);uint32_t required=8u+(uint32_t)result->item_count*(1u+FG_NGRAM_WIRE_ROW_BYTES);if(status!=FG_OK)return status;if(capacity<required){fg_error_set(err,FG_ERR_LIMIT,"n-gram result buffer is too small");return FG_ERR_LIMIT;}output[0]=result->source_rank;output[1]=result->destination_rank;output[2]=result->item_count;output[3]=0u;put_u32_be(output+4u,result->token_index);for(uint32_t i=0;i<result->item_count;i++){uint32_t offset=8u+i*(1u+FG_NGRAM_WIRE_ROW_BYTES);output[offset]=result->heads[i];memcpy(output+offset+1u,result->packed+(uint64_t)i*FG_NGRAM_WIRE_ROW_BYTES,FG_NGRAM_WIRE_ROW_BYTES);}*bytes=required;return FG_OK;}

fg_status fg_ngram_result_decode(fg_ngram_result *result,const uint8_t *payload,uint32_t bytes,fg_error *err){if(!result||!payload||bytes<8u){fg_error_set(err,FG_ERR_ARGUMENT,"invalid n-gram result input");return FG_ERR_ARGUMENT;}uint8_t item_count=payload[2];uint32_t required=8u+(uint32_t)item_count*(1u+FG_NGRAM_WIRE_ROW_BYTES);if(payload[3]||!item_count||item_count>FG_NGRAM_SHARD_MAX_ITEMS||bytes!=required){fg_error_set(err,FG_ERR_FORMAT,"invalid n-gram result size");return FG_ERR_FORMAT;}memset(result,0,sizeof(*result));result->source_rank=payload[0];result->destination_rank=payload[1];result->item_count=item_count;result->token_index=get_u32_be(payload+4u);for(uint32_t i=0;i<item_count;i++){uint32_t offset=8u+i*(1u+FG_NGRAM_WIRE_ROW_BYTES);result->heads[i]=payload[offset];memcpy(result->packed+(uint64_t)i*FG_NGRAM_WIRE_ROW_BYTES,payload+offset+1u,FG_NGRAM_WIRE_ROW_BYTES);}return validate_ngram_items(result->source_rank,result->destination_rank,result->item_count,result->heads,err);}

static bool digest_zero(const uint8_t digest[32]){uint8_t value=0;for(uint32_t i=0;i<32u;i++)value|=digest[i];return value==0;}

static fg_status validate_owner_session_control(const fg_owner_session_control *control,fg_error *err){
    if(!control||control->version!=FG_OWNER_SESSION_CONTROL_VERSION||
       control->operation<FG_OWNER_SESSION_BEGIN||
       control->operation>FG_OWNER_SESSION_RESTORED||
       control->rank>=FG_RANK_COUNT||control->position_mode>FG_POSITION_FOUR_AXIS||
       control->flags||!control->session_nonce||digest_zero(control->identity_sha256)||
       digest_zero(control->state_format_sha256)){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid owner session control header");
        return FG_ERR_MISMATCH;
    }
    bool initial=control->operation==FG_OWNER_SESSION_BEGIN||
                 control->operation==FG_OWNER_SESSION_READY;
    bool transactional=control->operation>=FG_OWNER_SESSION_PREPARE;
    bool state_reply=control->operation==FG_OWNER_SESSION_PREPARED||
                     control->operation==FG_OWNER_SESSION_RESTORED;
    if(initial&&(control->generation||control->committed_tokens||
                 !digest_zero(control->frontier_sha256)||
                 !digest_zero(control->state_sha256))){
        fg_error_set(err,FG_ERR_MISMATCH,"initial owner session control has checkpoint state");
        return FG_ERR_MISMATCH;
    }
    if(transactional&&(!control->generation||digest_zero(control->frontier_sha256))){
        fg_error_set(err,FG_ERR_MISMATCH,"transactional owner session control lacks a frontier");
        return FG_ERR_MISMATCH;
    }
    if(state_reply!=!digest_zero(control->state_sha256)){
        fg_error_set(err,FG_ERR_MISMATCH,"owner session state fingerprint is invalid for operation");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_owner_session_control_encode(uint8_t output[FG_OWNER_SESSION_CONTROL_BYTES],
                                          const fg_owner_session_control *control,
                                          fg_error *err){
    if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"owner session control output is null");return FG_ERR_ARGUMENT;}
    fg_status status=validate_owner_session_control(control,err);if(status!=FG_OK)return status;
    memset(output,0,FG_OWNER_SESSION_CONTROL_BYTES);put_u16_be(output,control->version);
    output[2]=control->operation;output[3]=control->rank;
    output[4]=(uint8_t)control->position_mode;output[5]=control->flags;
    put_u64_be(output+8u,control->session_nonce);
    put_u64_be(output+16u,control->generation);
    put_u64_be(output+24u,control->committed_tokens);
    memcpy(output+32u,control->identity_sha256,32u);
    memcpy(output+64u,control->frontier_sha256,32u);
    memcpy(output+96u,control->state_format_sha256,32u);
    memcpy(output+128u,control->state_sha256,32u);
    return FG_OK;
}

fg_status fg_owner_session_control_decode(fg_owner_session_control *control,
                                          const uint8_t *payload,uint32_t bytes,
                                          fg_error *err){
    if(!control||!payload){fg_error_set(err,FG_ERR_ARGUMENT,"invalid owner session control input");return FG_ERR_ARGUMENT;}
    if(bytes!=FG_OWNER_SESSION_CONTROL_BYTES||payload[6]||payload[7]){
        fg_error_set(err,FG_ERR_FORMAT,"invalid owner session control size or reserved bytes");
        return FG_ERR_FORMAT;
    }
    memset(control,0,sizeof(*control));control->version=get_u16_be(payload);
    control->operation=payload[2];control->rank=payload[3];
    control->position_mode=(fg_position_mode)payload[4];control->flags=payload[5];
    control->session_nonce=get_u64_be(payload+8u);
    control->generation=get_u64_be(payload+16u);
    control->committed_tokens=get_u64_be(payload+24u);
    memcpy(control->identity_sha256,payload+32u,32u);
    memcpy(control->frontier_sha256,payload+64u,32u);
    memcpy(control->state_format_sha256,payload+96u,32u);
    memcpy(control->state_sha256,payload+128u,32u);
    return validate_owner_session_control(control,err);
}

fg_status fg_expert_results_validate_route(const fg_manifest *manifest,uint32_t layer,uint32_t position,uint32_t owner_rank,const uint16_t expert_ids[FG_TOP_K],const fg_expert_result *results,uint32_t result_count,fg_error *err){
    if(!manifest||!expert_ids||!results||layer>=FG_LAYER_COUNT||owner_rank>=FG_RANK_COUNT||result_count==0||result_count>FG_GROUP_SIZE){fg_error_set(err,FG_ERR_ARGUMENT,"invalid expert result route validation arguments");return FG_ERR_ARGUMENT;}
    if(owner_rank!=0u&&manifest->layer_owner[layer]!=owner_rank){fg_error_set(err,FG_ERR_MISMATCH,"expert result destination is not layer owner");return FG_ERR_MISMATCH;}
    bool seen_expert[FG_EXPERT_COUNT]={0},seen_rank[FG_RANK_COUNT]={0},seen_slot[FG_TOP_K]={0};
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){
        if(expert_ids[slot]>=FG_EXPERT_COUNT||seen_expert[expert_ids[slot]]){fg_error_set(err,FG_ERR_FORMAT,"invalid canonical expert at slot %u",slot);return FG_ERR_FORMAT;}
        seen_expert[expert_ids[slot]]=true;
    }
    uint32_t received=0;bool has_prereduced=false;
    for(uint32_t r=0;r<result_count;r++){
        const fg_expert_result *result=&results[r];
        if(result->layer!=layer||result->position!=position||result->destination_rank!=owner_rank||result->source_rank>=FG_RANK_COUNT||seen_rank[result->source_rank]||!fg_topology_rank_in_layer(manifest,layer,result->source_rank)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted expert result %u",r);return FG_ERR_MISMATCH;}
        if(result->selected_count==0||result->selected_count>FG_TOP_K){fg_error_set(err,FG_ERR_FORMAT,"invalid expert result count");return FG_ERR_FORMAT;}
        seen_rank[result->source_rank]=true;
        for(uint32_t i=0;i<result->selected_count;i++){
            uint32_t slot=result->routing_slots[i];
            if(slot==0xFFu){has_prereduced=true;continue;}
            if(slot>=FG_TOP_K||seen_slot[slot]||manifest->expert_rank[layer][expert_ids[slot]]!=result->source_rank){fg_error_set(err,FG_ERR_MISMATCH,"expert result rank does not match route slot %u",slot);return FG_ERR_MISMATCH;}
            seen_slot[slot]=true;received++;
        }
    }
    if(!has_prereduced&&received!=FG_TOP_K){fg_error_set(err,FG_ERR_MISMATCH,"received %u of %u routed expert slots",received,FG_TOP_K);return FG_ERR_MISMATCH;}
    return FG_OK;
}

fg_status fg_partition_route(const fg_manifest *m,uint32_t layer,const uint16_t ids[FG_TOP_K],const float gates[FG_TOP_K],
                             fg_expert_route routes[FG_GROUP_SIZE],uint32_t *route_count,fg_error *err){
    if(!m||!ids||!gates||!routes||!route_count||layer>=FG_LAYER_COUNT){fg_error_set(err,FG_ERR_ARGUMENT,"invalid route partition arguments");return FG_ERR_ARGUMENT;}
    memset(routes,0,sizeof(*routes)*FG_GROUP_SIZE);*route_count=0;bool seen[FG_EXPERT_COUNT]={0};
    for(uint32_t slot=0;slot<FG_TOP_K;slot++){
        uint32_t expert=ids[slot];if(expert>=FG_EXPERT_COUNT||seen[expert]||!isfinite(gates[slot])){fg_error_set(err,FG_ERR_FORMAT,"invalid routed expert at slot %u",slot);return FG_ERR_FORMAT;}seen[expert]=true;
        uint32_t rank=m->expert_rank[layer][expert];if(!fg_topology_rank_in_layer(m,layer,rank)){fg_error_set(err,FG_ERR_MISMATCH,"expert %u is outside layer %u group",expert,layer);return FG_ERR_MISMATCH;}
        uint32_t ri=0;while(ri<*route_count&&routes[ri].destination_rank!=rank)ri++;
        if(ri==*route_count){if(*route_count>=FG_GROUP_SIZE){fg_error_set(err,FG_ERR_LIMIT,"route exceeds layer group");return FG_ERR_LIMIT;}routes[ri].destination_rank=(uint8_t)rank;(*route_count)++;}
        fg_expert_route *route=&routes[ri];uint32_t j=route->selected_count;if(j>=FG_TOP_K){fg_error_set(err,FG_ERR_LIMIT,"too many destination routes");return FG_ERR_LIMIT;}
        uint32_t local=0;for(uint32_t e=0;e<expert;e++)if(m->expert_rank[layer][e]==rank)local++;
        if(local>=FG_EXPERTS_PER_RANK){fg_error_set(err,FG_ERR_MISMATCH,"expert %u local index exceeds rank shard",expert);return FG_ERR_MISMATCH;}
        route->global_expert_ids[j]=(uint16_t)expert;route->local_expert_ids[j]=(uint16_t)local;route->routing_slots[j]=(uint8_t)slot;route->gates[j]=gates[slot];route->selected_count++;
    }
    return FG_OK;
}

fg_status fg_partition_prefill_routes(const fg_manifest *manifest,uint32_t layer,uint16_t token_count,const uint16_t *expert_ids,const float *gates,fg_prefill_route routes[FG_GROUP_SIZE],uint32_t *route_count,fg_prefill_pair *pair_storage,uint32_t pair_capacity,fg_error *err){
    if(!manifest||layer>=FG_LAYER_COUNT||!token_count||token_count>FG_PREFILL_MAX_TOKENS||!expert_ids||!gates||!routes||!route_count||!pair_storage||pair_capacity<(uint32_t)token_count*FG_TOP_K){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill route partition arguments");return FG_ERR_ARGUMENT;}
    uint32_t counts[FG_RANK_COUNT]={0};for(uint32_t token=0;token<token_count;token++){uint64_t seen[FG_EXPERT_COUNT/64u]={0};for(uint32_t slot=0;slot<FG_TOP_K;slot++){uint32_t index=token*FG_TOP_K+slot,expert=expert_ids[index];if(expert>=FG_EXPERT_COUNT||!isfinite(gates[index])||(seen[expert>>6u]&(UINT64_C(1)<<(expert&63u)))){fg_error_set(err,FG_ERR_FORMAT,"invalid prefill expert token %u slot %u",token,slot);return FG_ERR_FORMAT;}seen[expert>>6u]|=UINT64_C(1)<<(expert&63u);uint32_t rank=manifest->expert_rank[layer][expert];if(!fg_topology_rank_in_layer(manifest,layer,rank)){fg_error_set(err,FG_ERR_MISMATCH,"prefill expert %u is outside layer %u group",expert,layer);return FG_ERR_MISMATCH;}counts[rank]++;}}
    memset(routes,0,FG_GROUP_SIZE*sizeof(*routes));*route_count=0;uint32_t used=0,route_for_rank[FG_RANK_COUNT];for(uint32_t rank=0;rank<FG_RANK_COUNT;rank++)route_for_rank[rank]=UINT32_MAX;
    for(uint32_t group_slot=0;group_slot<FG_GROUP_SIZE;group_slot++){uint32_t rank=manifest->layer_groups[layer][group_slot];if(!counts[rank])continue;uint32_t route=(*route_count)++;route_for_rank[rank]=route;routes[route].destination_rank=(uint8_t)rank;routes[route].pair_count=(uint16_t)counts[rank];routes[route].pairs=pair_storage+used;used+=counts[rank];}
    uint32_t written[FG_GROUP_SIZE]={0};for(uint32_t token=0;token<token_count;token++)for(uint32_t slot=0;slot<FG_TOP_K;slot++){uint32_t index=token*FG_TOP_K+slot,rank=manifest->expert_rank[layer][expert_ids[index]],route=route_for_rank[rank],at=written[route]++;routes[route].pairs[at]=(fg_prefill_pair){.token_slot=(uint16_t)token,.expert_id=expert_ids[index],.routing_slot=(uint8_t)slot,.gate=gates[index]};}
    for(uint32_t route=0;route<*route_count;route++)if(written[route]!=routes[route].pair_count){fg_error_set(err,FG_ERR_MISMATCH,"prefill route partition count mismatch");return FG_ERR_MISMATCH;}
    return FG_OK;
}

fg_status fg_prefill_results_validate_route(const fg_manifest *manifest,uint32_t layer,uint32_t first_position,uint32_t owner_rank,uint16_t token_count,const uint16_t *expert_ids,const fg_prefill_result *results,uint32_t result_count,fg_error *err){
    if(!manifest||layer>=FG_LAYER_COUNT||owner_rank>=FG_RANK_COUNT||(owner_rank!=0u&&manifest->layer_owner[layer]!=owner_rank)||!token_count||token_count>FG_PREFILL_MAX_TOKENS||!expert_ids||!results||!result_count||result_count>FG_GROUP_SIZE){fg_error_set(err,FG_ERR_ARGUMENT,"invalid prefill result route validation arguments");return FG_ERR_ARGUMENT;}
    for(uint32_t token=0;token<token_count;token++){uint64_t experts[FG_EXPERT_COUNT/64u]={0};for(uint32_t slot=0;slot<FG_TOP_K;slot++){uint32_t expert=expert_ids[token*FG_TOP_K+slot];if(expert>=FG_EXPERT_COUNT||(experts[expert>>6u]&(UINT64_C(1)<<(expert&63u)))){fg_error_set(err,FG_ERR_FORMAT,"invalid canonical prefill expert token %u slot %u",token,slot);return FG_ERR_FORMAT;}experts[expert>>6u]|=UINT64_C(1)<<(expert&63u);}}
    bool seen_rank[FG_RANK_COUNT]={0};bool seen_slot[FG_PREFILL_MAX_TOKENS][FG_TOP_K]={{false}};uint32_t received=0;
    for(uint32_t r=0;r<result_count;r++){const fg_prefill_result *result=&results[r];fg_status status=validate_prefill_result(result,err);if(status!=FG_OK)return status;if(result->layer!=layer||result->first_position!=first_position||result->destination_rank!=owner_rank||result->token_count!=token_count||result->source_rank>=FG_RANK_COUNT||seen_rank[result->source_rank]||!fg_topology_rank_in_layer(manifest,layer,result->source_rank)){fg_error_set(err,FG_ERR_MISMATCH,"stale or misrouted prefill expert result %u",r);return FG_ERR_MISMATCH;}seen_rank[result->source_rank]=true;for(uint32_t pair_index=0;pair_index<result->pair_count;pair_index++){uint32_t token=result->pairs[pair_index].token_slot,slot=result->pairs[pair_index].routing_slot;if(token>=token_count||slot>=FG_TOP_K||seen_slot[token][slot]||manifest->expert_rank[layer][expert_ids[token*FG_TOP_K+slot]]!=result->source_rank){fg_error_set(err,FG_ERR_MISMATCH,"prefill result rank does not match token %u route slot %u",token,slot);return FG_ERR_MISMATCH;}seen_slot[token][slot]=true;received++;}}
    uint32_t expected=(uint32_t)token_count*FG_TOP_K;if(received!=expected){fg_error_set(err,FG_ERR_MISMATCH,"received %u of %u prefill expert pairs",received,expected);return FG_ERR_MISMATCH;}return FG_OK;
}
