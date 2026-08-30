#include "fg_session.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"

#include <math.h>
#include <string.h>

#define FG_SESSION_IDENTITY_MAGIC UINT64_C(0x314449534746474c)
#define FG_SESSION_FRONTIER_MAGIC UINT64_C(0x315246534746474c)

static void put_u32_be(uint8_t *output,uint32_t value){
    output[0]=(uint8_t)(value>>24u);output[1]=(uint8_t)(value>>16u);
    output[2]=(uint8_t)(value>>8u);output[3]=(uint8_t)value;
}

static uint32_t get_u32_be(const uint8_t *input){
    return((uint32_t)input[0]<<24u)|((uint32_t)input[1]<<16u)|
           ((uint32_t)input[2]<<8u)|input[3];
}

static void put_u64_be(uint8_t *output,uint64_t value){
    put_u32_be(output,(uint32_t)(value>>32u));put_u32_be(output+4u,(uint32_t)value);
}

static uint64_t get_u64_be(const uint8_t *input){
    return((uint64_t)get_u32_be(input)<<32u)|get_u32_be(input+4u);
}

static bool digest_is_zero(const uint8_t digest[32]){
    uint8_t value=0;for(uint32_t i=0;i<32u;i++)value|=digest[i];return value==0;
}

static void identity_digest(const fg_session_identity *identity,uint8_t digest[32]){
    uint8_t wire[FG_SESSION_IDENTITY_WIRE_BYTES];
    memset(wire,0,sizeof(wire));put_u64_be(wire,FG_SESSION_IDENTITY_MAGIC);
    put_u32_be(wire+8u,identity->version);put_u32_be(wire+12u,sizeof(wire));
    uint32_t offset=16u;
    const uint8_t *fields[]={
        identity->model_sha256,identity->tokenizer_sha256,identity->quantization_sha256,
        identity->manifest_sha256,identity->rope_policy_sha256,
        identity->vision_projector_sha256,identity->state_format_sha256
    };
    for(uint32_t i=0;i<sizeof(fields)/sizeof(fields[0]);i++,offset+=32u)
        memcpy(wire+offset,fields[i],32u);
    fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,wire,offset);
    fg_sha256_final(&hash,digest);
}

static fg_status validate_identity(const fg_session_identity *identity,fg_error *err){
    if(!identity||identity->version!=FG_SESSION_IDENTITY_VERSION||
       digest_is_zero(identity->manifest_sha256)||digest_is_zero(identity->rope_policy_sha256)||
       digest_is_zero(identity->state_format_sha256)){
        fg_error_set(err,FG_ERR_MISMATCH,"session identity is incomplete or has an unsupported version");
        return FG_ERR_MISMATCH;
    }
    uint8_t digest[32];identity_digest(identity,digest);
    if(memcmp(digest,identity->identity_sha256,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"session identity fingerprint mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_session_identity_from_manifest(const fg_manifest *manifest,
                                            fg_session_identity *identity,
                                            fg_error *err){
    if(!manifest||!identity){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid session identity arguments");return FG_ERR_ARGUMENT;
    }
    fg_status status=fg_manifest_validate(manifest,err);
    if(status!=FG_OK)return status;
    if(manifest->session.version!=FG_MANIFEST_CONTRACT_VERSION||
       manifest->session.position_mode>FG_POSITION_FOUR_AXIS||
       digest_is_zero(manifest->manifest_sha256)){
        fg_error_set(err,FG_ERR_MISMATCH,"manifest has no compatible session identity contract");
        return FG_ERR_MISMATCH;
    }
    memset(identity,0,sizeof(*identity));identity->version=FG_SESSION_IDENTITY_VERSION;
    const uint8_t *model=digest_is_zero(manifest->source_sha256)?
        manifest->session.component_sha256[FG_COMPONENT_TEXT]:manifest->source_sha256;
    const uint8_t *quantization=digest_is_zero(manifest->quant_profile_sha256)?
        manifest->session.quantization_sha256:manifest->quant_profile_sha256;
    memcpy(identity->model_sha256,model,32u);
    memcpy(identity->tokenizer_sha256,
           manifest->session.component_sha256[FG_COMPONENT_TOKENIZER],32u);
    memcpy(identity->quantization_sha256,quantization,32u);
    memcpy(identity->manifest_sha256,manifest->manifest_sha256,32u);
    memcpy(identity->rope_policy_sha256,manifest->session.rope_policy_sha256,32u);
    memcpy(identity->vision_projector_sha256,
           manifest->session.component_sha256[FG_COMPONENT_VISION],32u);
    memcpy(identity->state_format_sha256,manifest->session.state_format_sha256,32u);
    identity_digest(identity,identity->identity_sha256);
    return validate_identity(identity,err);
}

fg_status fg_session_identity_encode(uint8_t output[FG_SESSION_IDENTITY_WIRE_BYTES],
                                     const fg_session_identity *identity,fg_error *err){
    if(!output){fg_error_set(err,FG_ERR_ARGUMENT,"session identity output is null");return FG_ERR_ARGUMENT;}
    fg_status status=validate_identity(identity,err);if(status!=FG_OK)return status;
    memset(output,0,FG_SESSION_IDENTITY_WIRE_BYTES);
    put_u64_be(output,FG_SESSION_IDENTITY_MAGIC);
    put_u32_be(output+8u,identity->version);
    put_u32_be(output+12u,FG_SESSION_IDENTITY_WIRE_BYTES);
    uint32_t offset=16u;
    const uint8_t *fields[]={
        identity->model_sha256,identity->tokenizer_sha256,identity->quantization_sha256,
        identity->manifest_sha256,identity->rope_policy_sha256,
        identity->vision_projector_sha256,identity->state_format_sha256,
        identity->identity_sha256
    };
    for(uint32_t i=0;i<sizeof(fields)/sizeof(fields[0]);i++,offset+=32u)
        memcpy(output+offset,fields[i],32u);
    return FG_OK;
}

fg_status fg_session_identity_decode(fg_session_identity *identity,const uint8_t *payload,
                                     uint32_t bytes,fg_error *err){
    if(!identity||!payload){fg_error_set(err,FG_ERR_ARGUMENT,"invalid session identity input");return FG_ERR_ARGUMENT;}
    if(bytes!=FG_SESSION_IDENTITY_WIRE_BYTES||
       get_u64_be(payload)!=FG_SESSION_IDENTITY_MAGIC||
       get_u32_be(payload+8u)!=FG_SESSION_IDENTITY_VERSION||
       get_u32_be(payload+12u)!=FG_SESSION_IDENTITY_WIRE_BYTES){
        fg_error_set(err,FG_ERR_MISMATCH,"unsupported session identity payload");return FG_ERR_MISMATCH;
    }
    memset(identity,0,sizeof(*identity));identity->version=get_u32_be(payload+8u);
    uint32_t offset=16u;
    uint8_t *fields[]={
        identity->model_sha256,identity->tokenizer_sha256,identity->quantization_sha256,
        identity->manifest_sha256,identity->rope_policy_sha256,
        identity->vision_projector_sha256,identity->state_format_sha256,
        identity->identity_sha256
    };
    for(uint32_t i=0;i<sizeof(fields)/sizeof(fields[0]);i++,offset+=32u)
        memcpy(fields[i],payload+offset,32u);
    return validate_identity(identity,err);
}

fg_status fg_session_identity_validate_compatible(const fg_session_identity *expected,
                                                  const fg_session_identity *actual,
                                                  fg_error *err){
    fg_status status=validate_identity(expected,err);if(status!=FG_OK)return status;
    status=validate_identity(actual,err);if(status!=FG_OK)return status;
    struct identity_field {const char *name;const uint8_t *expected;const uint8_t *actual;};
    const struct identity_field fields[]={
        {"model",expected->model_sha256,actual->model_sha256},
        {"tokenizer",expected->tokenizer_sha256,actual->tokenizer_sha256},
        {"quantization",expected->quantization_sha256,actual->quantization_sha256},
        {"manifest",expected->manifest_sha256,actual->manifest_sha256},
        {"RoPE policy",expected->rope_policy_sha256,actual->rope_policy_sha256},
        {"vision projector",expected->vision_projector_sha256,actual->vision_projector_sha256},
        {"state format",expected->state_format_sha256,actual->state_format_sha256}
    };
    for(uint32_t i=0;i<sizeof(fields)/sizeof(fields[0]);i++)if(
        memcmp(fields[i].expected,fields[i].actual,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"session %s fingerprint mismatch",fields[i].name);
        return FG_ERR_MISMATCH;
    }
    if(memcmp(expected->identity_sha256,actual->identity_sha256,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"session identity fingerprint mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

static fg_status validate_frontier(const fg_session_frontier *frontier,fg_error *err){
    union{float f;uint32_t u;}logit={frontier?frontier->next_logit:0.0f};
    if(!frontier||frontier->version!=FG_SESSION_FRONTIER_VERSION||
       frontier->position_mode>FG_POSITION_FOUR_AXIS||
       (frontier->position_mode==FG_POSITION_TEXT&&frontier->position[3])||
       frontier->token_count>FG_MAX_CONTEXT||
       frontier->committed_tokens>frontier->token_count||
       (frontier->token_count&&!frontier->tokens)||
       digest_is_zero(frontier->identity_sha256)||
       digest_is_zero(frontier->rendered_transcript_sha256)||
       (frontier->next_token_valid&&
        (frontier->next_token>=FG_Q38_VOCAB_SIZE||!isfinite(frontier->next_logit)))||
       (!frontier->next_token_valid&&(frontier->next_token||logit.u))){
        fg_error_set(err,FG_ERR_MISMATCH,"invalid or incompatible session frontier");
        return FG_ERR_MISMATCH;
    }
    if(frontier->next_token_valid&&digest_is_zero(frontier->next_token_state_sha256)){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier lacks next-token state");
        return FG_ERR_MISMATCH;
    }
    if(!frontier->next_token_valid&&!digest_is_zero(frontier->next_token_state_sha256)){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier has state without a next token");
        return FG_ERR_MISMATCH;
    }
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++)if(
        frontier->qsa_lengths[layer]>frontier->committed_tokens||
        frontier->gdn_lengths[layer]>frontier->committed_tokens||
        frontier->ple_lengths[layer]>frontier->committed_tokens){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier layer %u exceeds committed tokens",layer);
        return FG_ERR_MISMATCH;
    }
    for(uint64_t token=0;token<frontier->token_count;token++)if(
        frontier->tokens[token]<0||(uint32_t)frontier->tokens[token]>=FG_Q38_VOCAB_SIZE){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier token %llu is outside vocabulary",
                     (unsigned long long)token);return FG_ERR_MISMATCH;
    }
    return FG_OK;
}

fg_status fg_session_frontier_encode(uint8_t *output,uint32_t capacity,uint32_t *bytes,
                                     const fg_session_frontier *frontier,fg_error *err){
    if(!output||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"invalid session frontier output");return FG_ERR_ARGUMENT;}
    fg_status status=validate_frontier(frontier,err);if(status!=FG_OK)return status;
    uint64_t required=(uint64_t)FG_SESSION_FRONTIER_HEADER_BYTES+
                      frontier->token_count*4u+FG_SESSION_FRONTIER_DIGEST_BYTES;
    if(required>UINT32_MAX||required>capacity){
        fg_error_set(err,FG_ERR_LIMIT,"session frontier output buffer is too small");
        return FG_ERR_LIMIT;
    }
    memset(output,0,(size_t)required);put_u64_be(output,FG_SESSION_FRONTIER_MAGIC);
    put_u32_be(output+8u,frontier->version);
    put_u32_be(output+12u,FG_SESSION_FRONTIER_HEADER_BYTES);
    put_u32_be(output+16u,(uint32_t)required);
    put_u32_be(output+20u,(uint32_t)frontier->position_mode);
    put_u32_be(output+24u,frontier->next_token_valid?1u:0u);
    put_u64_be(output+28u,frontier->generation);
    put_u64_be(output+36u,frontier->committed_tokens);
    put_u64_be(output+44u,frontier->token_count);
    put_u32_be(output+52u,frontier->next_token);
    union{float f;uint32_t u;}logit={frontier->next_logit};
    put_u32_be(output+56u,logit.u);
    for(uint32_t axis=0;axis<4u;axis++)put_u32_be(output+60u+axis*4u,frontier->position[axis]);
    uint32_t offset=76u;
    memcpy(output+offset,frontier->identity_sha256,32u);offset+=32u;
    memcpy(output+offset,frontier->rendered_transcript_sha256,32u);offset+=32u;
    memcpy(output+offset,frontier->next_token_state_sha256,32u);offset+=32u;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++,offset+=8u)
        put_u64_be(output+offset,frontier->qsa_lengths[layer]);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++,offset+=8u)
        put_u64_be(output+offset,frontier->gdn_lengths[layer]);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++,offset+=8u)
        put_u64_be(output+offset,frontier->ple_lengths[layer]);
    uint32_t token_hash_offset=offset;offset+=32u;
    if(offset!=FG_SESSION_FRONTIER_HEADER_BYTES){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier header layout mismatch");
        return FG_ERR_MISMATCH;
    }
    fg_sha256 token_hash;fg_sha256_init(&token_hash);
    for(uint64_t token=0;token<frontier->token_count;token++,offset+=4u){
        put_u32_be(output+offset,(uint32_t)frontier->tokens[token]);
        fg_sha256_update(&token_hash,output+offset,4u);
    }
    fg_sha256_final(&token_hash,output+token_hash_offset);
    fg_sha256 frontier_hash;fg_sha256_init(&frontier_hash);
    fg_sha256_update(&frontier_hash,output,(size_t)required-FG_SESSION_FRONTIER_DIGEST_BYTES);
    fg_sha256_final(&frontier_hash,output+required-FG_SESSION_FRONTIER_DIGEST_BYTES);
    *bytes=(uint32_t)required;return FG_OK;
}

fg_status fg_session_frontier_decode(fg_session_frontier *frontier,int32_t *token_storage,
                                     uint64_t token_capacity,const uint8_t *payload,
                                     uint32_t bytes,fg_error *err){
    if(!frontier||!payload){fg_error_set(err,FG_ERR_ARGUMENT,"invalid session frontier input");return FG_ERR_ARGUMENT;}
    if(bytes<FG_SESSION_FRONTIER_HEADER_BYTES+FG_SESSION_FRONTIER_DIGEST_BYTES||
       get_u64_be(payload)!=FG_SESSION_FRONTIER_MAGIC||
       get_u32_be(payload+8u)!=FG_SESSION_FRONTIER_VERSION||
       get_u32_be(payload+12u)!=FG_SESSION_FRONTIER_HEADER_BYTES||
       get_u32_be(payload+16u)!=bytes){
        fg_error_set(err,FG_ERR_MISMATCH,"unsupported session frontier payload");
        return FG_ERR_MISMATCH;
    }
    uint64_t token_count=get_u64_be(payload+44u);
    uint64_t required=(uint64_t)FG_SESSION_FRONTIER_HEADER_BYTES+token_count*4u+
                      FG_SESSION_FRONTIER_DIGEST_BYTES;
    if(required!=bytes||token_count>token_capacity||(token_count&&!token_storage)){
        fg_error_set(err,FG_ERR_FORMAT,"invalid session frontier token storage or length");
        return FG_ERR_FORMAT;
    }
    fg_sha256 frontier_hash;uint8_t expected[32];fg_sha256_init(&frontier_hash);
    fg_sha256_update(&frontier_hash,payload,bytes-FG_SESSION_FRONTIER_DIGEST_BYTES);
    fg_sha256_final(&frontier_hash,expected);
    if(memcmp(expected,payload+bytes-FG_SESSION_FRONTIER_DIGEST_BYTES,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier fingerprint mismatch");
        return FG_ERR_MISMATCH;
    }
    memset(frontier,0,sizeof(*frontier));frontier->version=get_u32_be(payload+8u);
    frontier->position_mode=(fg_position_mode)get_u32_be(payload+20u);
    uint32_t flags=get_u32_be(payload+24u);
    if(flags&~1u){fg_error_set(err,FG_ERR_FORMAT,"session frontier has unknown flags");return FG_ERR_FORMAT;}
    frontier->next_token_valid=(flags&1u)!=0;
    frontier->generation=get_u64_be(payload+28u);
    frontier->committed_tokens=get_u64_be(payload+36u);
    frontier->token_count=token_count;frontier->next_token=get_u32_be(payload+52u);
    union{uint32_t u;float f;}logit={get_u32_be(payload+56u)};frontier->next_logit=logit.f;
    for(uint32_t axis=0;axis<4u;axis++)frontier->position[axis]=get_u32_be(payload+60u+axis*4u);
    uint32_t offset=76u;
    memcpy(frontier->identity_sha256,payload+offset,32u);offset+=32u;
    memcpy(frontier->rendered_transcript_sha256,payload+offset,32u);offset+=32u;
    memcpy(frontier->next_token_state_sha256,payload+offset,32u);offset+=32u;
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++,offset+=8u)
        frontier->qsa_lengths[layer]=get_u64_be(payload+offset);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++,offset+=8u)
        frontier->gdn_lengths[layer]=get_u64_be(payload+offset);
    for(uint32_t layer=0;layer<FG_LAYER_COUNT;layer++,offset+=8u)
        frontier->ple_lengths[layer]=get_u64_be(payload+offset);
    memcpy(frontier->token_history_sha256,payload+offset,32u);offset+=32u;
    fg_sha256 token_hash;fg_sha256_init(&token_hash);frontier->tokens=token_storage;
    for(uint64_t token=0;token<token_count;token++,offset+=4u){
        token_storage[token]=(int32_t)get_u32_be(payload+offset);
        fg_sha256_update(&token_hash,payload+offset,4u);
    }
    fg_sha256_final(&token_hash,expected);
    if(memcmp(expected,frontier->token_history_sha256,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier token history fingerprint mismatch");
        return FG_ERR_MISMATCH;
    }
    memcpy(frontier->frontier_sha256,payload+bytes-FG_SESSION_FRONTIER_DIGEST_BYTES,32u);
    return validate_frontier(frontier,err);
}

fg_status fg_session_frontier_validate_compatible(const fg_session_identity *identity,
                                                  const fg_session_frontier *frontier,
                                                  fg_position_mode position_mode,
                                                  fg_error *err){
    fg_status status=validate_identity(identity,err);if(status!=FG_OK)return status;
    status=validate_frontier(frontier,err);if(status!=FG_OK)return status;
    if(frontier->position_mode!=position_mode){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier position mode mismatch");
        return FG_ERR_MISMATCH;
    }
    if(memcmp(frontier->identity_sha256,identity->identity_sha256,32u)){
        fg_error_set(err,FG_ERR_MISMATCH,"session frontier identity mismatch");
        return FG_ERR_MISMATCH;
    }
    return FG_OK;
}
