#include "fg_protocol.h"
#include "fg_runtime.h"
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static float from_bits(uint32_t bits){float x;memcpy(&x,&bits,4u);return x;}
static uint32_t bits(float x){uint32_t u;memcpy(&u,&x,4u);return u;}

int main(void){
    unsetenv("FG_FP16_HOPS");unsetenv("FG_FP16_HOP_MASK");
    assert(unsetenv("FG_BF16_HOPS")==0);assert(!fg_bf16_hops_enabled());
    assert(setenv("FG_BF16_HOPS","0",1)==0);assert(!fg_bf16_hops_enabled());
    assert(setenv("FG_BF16_HOPS","yes",1)==0);assert(!fg_bf16_hops_enabled());
    assert(bits(fg_bf16_hop_round(from_bits(0x3f808000)))==0x3f800000);
    assert(bits(fg_bf16_hop_round(from_bits(0x3f818000)))==0x3f820000);
    assert(bits(fg_bf16_hop_round(from_bits(0xbf818000)))==0xbf820000);
    assert(bits(fg_bf16_hop_round(-0.0f))==0x80000000);
    assert(bits(fg_bf16_hop_round(from_bits(0x00008000)))==0);
    assert(bits(fg_bf16_hop_round(from_bits(0x00018000)))==0x00020000);
    assert(isnan(fg_bf16_hop_round(from_bits(0x7f800001))));
    assert(isinf(fg_bf16_hop_round(INFINITY)));
    fg_layer_work source={.layer=1,.source_rank=0,.destination_rank=1,
        .flags=FG_LAYER_WORK_HAS_NGRAM,.token_index=17,.position={17,17,17,0}},decoded;
    for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)source.hyper[i]=(float)((int)(i%71)-35)/7.0f;
    for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)
        source.ngram_embedding[i]=(float)((int)(i%97)-48)/11.0f;
    uint8_t wire[FG_DECODE_BATCH_WORK_MAX_BYTES],canonical[FG_LAYER_WORK_MAX_BYTES];
    uint32_t bytes=0,full=0;fg_error error={0};
    assert(setenv("FG_BF16_HOPS","0",1)==0);
    assert(fg_layer_work_encode(canonical,sizeof(canonical),&full,6,&source,&error)==FG_OK);
    assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
    setenv("FG_FP16_SCALAR","1",1);
    assert(fg_decode_layer_work_encode(canonical,sizeof(canonical),&full,6,&source,&error)==FG_OK);
    assert(full==bytes&&!memcmp(canonical,wire,bytes));
    unsetenv("FG_FP16_SCALAR");
    assert(bytes==full&&!memcmp(wire,canonical,bytes));
    for(uint32_t four=0;four<2;four++)for(uint32_t ngram=0;ngram<2;ngram++){
        source.position_mode=four?FG_POSITION_FOUR_AXIS:FG_POSITION_TEXT;
        source.position[3]=four?123u:0u;source.flags=ngram?FG_LAYER_WORK_HAS_NGRAM:0u;
        assert(setenv("FG_BF16_HOPS","1",1)==0);assert(fg_bf16_hops_enabled());
        assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
        uint32_t header=four?28:24;
        assert(bytes==header+(FG_HYPER_WIDTH+(ngram?FG_NGRAM_EMBED_VALUES:0))*2);
        assert(wire[3]==(source.flags|FG_LAYER_WORK_WIRE_BF16));
        assert(fg_decode_layer_work_encode(wire,bytes-1,&full,6,&source,&error)==FG_ERR_LIMIT);
        assert(setenv("FG_BF16_HOPS","0",1)==0);
        assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_OK);
        fg_layer_work portable;setenv("FG_FP16_SCALAR","1",1);
        assert(fg_decode_layer_work_decode(&portable,6,wire,bytes,&error)==FG_OK);
        assert(!memcmp(&portable,&decoded,sizeof(decoded)));unsetenv("FG_FP16_SCALAR");
        assert(decoded.flags==source.flags&&decoded.position_mode==source.position_mode);
        assert(!memcmp(decoded.position,source.position,sizeof(source.position)));
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
            assert(bits(decoded.hyper[i])==bits(fg_bf16_hop_round(source.hyper[i])));
        if(ngram)for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)
            assert(bits(decoded.ngram_embedding[i])==bits(fg_bf16_hop_round(source.ngram_embedding[i])));
        assert(fg_decode_layer_work_decode(&decoded,5,wire,bytes,&error)==FG_ERR_FORMAT);
        for(uint32_t short_bytes=0;short_bytes<bytes;short_bytes++)
            assert(fg_decode_layer_work_decode(&decoded,6,wire,short_bytes,&error)!=FG_OK);
        wire[10]=1;assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_ERR_FORMAT);
        wire[10]=0;wire[header]=0x7f;wire[header+1]=0x80;
        assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_ERR_FORMAT);
    }
    source.hyper[0]=FLT_MAX;assert(setenv("FG_BF16_HOPS","1",1)==0);
    assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_ERR_FORMAT);
    source.hyper[0]=NAN;
    assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_ERR_FORMAT);
    source.hyper[0]=1.0f;
    fg_decode_batch_work batch={.layer=1,.source_rank=0,.destination_rank=1,
        .flags=FG_LAYER_WORK_HAS_NGRAM},result;
    float hyper[FG_DECODE_BATCH_MAX_SLOTS*FG_HYPER_WIDTH];
    float ngram[FG_DECODE_BATCH_MAX_SLOTS*FG_NGRAM_EMBED_VALUES];
    for(uint32_t four=0;four<2;four++)for(uint32_t has_ngram=0;has_ngram<2;has_ngram++)
      for(uint32_t slots=1;slots<=FG_DECODE_BATCH_MAX_SLOTS;slots++){
        batch.position_mode=four?FG_POSITION_FOUR_AXIS:FG_POSITION_TEXT;
        batch.flags=has_ngram?FG_LAYER_WORK_HAS_NGRAM:0;
        batch.slot_count=(uint16_t)slots;
        for(uint32_t i=0;i<slots;i++)batch.slots[i]=(fg_decode_batch_slot_work){
            .token_index=17+i,.state_slot=i,.position={17+i,17+i,17+i,four?99u:0u},
            .hyper=source.hyper,.ngram_embedding=source.ngram_embedding};
        assert(setenv("FG_BF16_HOPS","1",1)==0);
        assert(fg_decode_batch_work_encode(wire,sizeof(wire),&bytes,6,&batch,&error)==FG_OK);
        assert(setenv("FG_BF16_HOPS","0",1)==0);
        assert(fg_decode_batch_work_decode(&result,6,hyper,
            sizeof(hyper)/sizeof(float),ngram,sizeof(ngram)/sizeof(float),
            wire,bytes,&error)==FG_OK);
        assert(result.flags==batch.flags&&result.slot_count==slots);
        for(uint32_t slot=0;slot<slots;slot++){
            assert(result.slots[slot].state_slot==slot);
            for(uint32_t j=0;j<FG_HYPER_WIDTH;j++)
                assert(bits(result.slots[slot].hyper[j])==bits(fg_bf16_hop_round(source.hyper[j])));
            if(has_ngram)for(uint32_t j=0;j<FG_NGRAM_EMBED_VALUES;j++)
                assert(bits(result.slots[slot].ngram_embedding[j])==
                       bits(fg_bf16_hop_round(source.ngram_embedding[j])));
        }
        assert(fg_decode_batch_work_decode(&result,6,hyper,
            sizeof(hyper)/sizeof(float),ngram,sizeof(ngram)/sizeof(float),
            wire,bytes-1,&error)==FG_ERR_FORMAT);
    }
    fg_manifest *m=malloc(sizeof(*m));assert(m);fg_manifest_init(m);
    assert(mkdir(".build",0700)==0||errno==EEXIST);
    char path[128];snprintf(path,sizeof(path),".build/bf16-options-%ld.fgm",(long)getpid());
    assert(fg_manifest_write(path,m,&error)==FG_OK);
    fg_runtime_options requested={.experimental_flags=FG_RUNTIME_EXPERIMENTAL_MTP},resolved;
    assert(setenv("FG_BF16_HOPS","0",1)==0);
    fg_status resolved_status=fg_runtime_options_resolve(&resolved,m,&requested,&error);
    if(resolved_status!=FG_ERR_ARGUMENT)fprintf(stderr,"options status=%u %s\n",resolved_status,error.message);
    assert(resolved_status==FG_ERR_ARGUMENT);
    assert(setenv("FG_BF16_HOPS","1",1)==0);
    assert(fg_runtime_options_resolve(&resolved,m,&requested,&error)==FG_ERR_UNAVAILABLE);
    assert(strstr(error.message,"BF16 decode hops"));
    unsetenv("FG_BF16_HOPS");
    assert(setenv("FG_FP16_HOPS","1",1)==0);
    assert(fg_runtime_options_resolve(&resolved,m,&requested,&error)==FG_ERR_UNAVAILABLE);
    unlink(path);free(m);
    for(uint32_t mode=0;mode<2;mode++){
        const char *setting=mode?"128":"1";source.hyper[0]=1.0f+0x1p-11f;
        source.hyper[1]=1.0f+3.0f*0x1p-11f;
        for(uint32_t four=0;four<2;four++)for(uint32_t ng=0;ng<2;ng++){
            source.position_mode=four?FG_POSITION_FOUR_AXIS:FG_POSITION_TEXT;
            source.position[3]=four?9u:0u;source.flags=ng?FG_LAYER_WORK_HAS_NGRAM:0;
            setenv("FG_FP16_HOPS",setting,1);
            assert(fg_fp16_hops_mode()==(mode?128u:1u));
            assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
            uint32_t values=FG_HYPER_WIDTH+(ng?FG_NGRAM_EMBED_VALUES:0);
            uint32_t header=four?28u:24u,scales=mode?(values+127u)/128u:1u;
            assert(bytes==header+values*2u+scales*4u);
            assert(fg_decode_layer_work_encode(wire,bytes-1u,&full,6,&source,&error)==FG_ERR_LIMIT);
            unsetenv("FG_FP16_HOPS");
            assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_OK);
            assert(bits(decoded.hyper[0])==bits(1.0f));
            assert(bits(decoded.hyper[1])==bits(1.0f+0x1p-9f));
            for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
                assert(fabsf(decoded.hyper[i]-source.hyper[i])<=0.0005f*fmaxf(1,fabsf(source.hyper[i])));
            if(ng)for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)
                assert(fabsf(decoded.ngram_embedding[i]-source.ngram_embedding[i])<=
                    0.0005f*fmaxf(1,fabsf(source.ngram_embedding[i])));
            for(uint32_t n=0;n<bytes;n++)
                assert(fg_decode_layer_work_decode(&decoded,6,wire,n,&error)!=FG_OK);
            wire[header]=0x7f;wire[header+1]=0x80;wire[header+2]=wire[header+3]=0;
            assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_ERR_FORMAT);
        }
        setenv("FG_FP16_HOPS",setting,1);source.flags=0;source.position_mode=FG_POSITION_TEXT;
        source.position[3]=0;
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)source.hyper[i]=(i&1u)?-FLT_MAX:FLT_MAX;
        assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
        assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_OK);
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)assert(bits(decoded.hyper[i])==bits(source.hyper[i]));
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)source.hyper[i]=from_bits(i%128u);
        source.hyper[0]=-0.0f;
        assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
        assert(fg_decode_layer_work_decode(&decoded,6,wire,bytes,&error)==FG_OK);
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)assert(bits(decoded.hyper[i])==bits(source.hyper[i]));
        for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)source.hyper[i]=(float)((int)(i%71)-35)/7.0f;
        for(uint32_t ng=0;ng<2;ng++)for(uint32_t four=0;four<2;four++)
          for(uint32_t slots=1;slots<=FG_DECODE_BATCH_MAX_SLOTS;slots++){
            batch.flags=ng?FG_LAYER_WORK_HAS_NGRAM:0;
            batch.position_mode=four?FG_POSITION_FOUR_AXIS:FG_POSITION_TEXT;batch.slot_count=(uint16_t)slots;
            for(uint32_t i=0;i<slots;i++)batch.slots[i]=(fg_decode_batch_slot_work){
                .token_index=i,.state_slot=i,.position={i,i,i,four?5u:0},
                .hyper=source.hyper,.ngram_embedding=source.ngram_embedding};
            assert(fg_decode_batch_work_encode(wire,sizeof(wire),&bytes,6,&batch,&error)==FG_OK);
            assert(fg_decode_batch_work_decode(&result,6,hyper,
                sizeof(hyper)/sizeof(float),ngram,sizeof(ngram)/sizeof(float),wire,bytes,&error)==FG_OK);
            for(uint32_t s=0;s<slots;s++){
                assert(result.slots[s].state_slot==s);
                for(uint32_t i=0;i<FG_HYPER_WIDTH;i++)
                    assert(fabsf(result.slots[s].hyper[i]-source.hyper[i])<=0.0005f*fmaxf(1,fabsf(source.hyper[i])));
                if(ng)for(uint32_t i=0;i<FG_NGRAM_EMBED_VALUES;i++)
                    assert(fabsf(result.slots[s].ngram_embedding[i]-source.ngram_embedding[i])<=
                        0.0005f*fmaxf(1,fabsf(source.ngram_embedding[i])));
            }
            assert(fg_decode_batch_work_decode(&result,6,hyper,
                sizeof(hyper)/sizeof(float),ngram,sizeof(ngram)/sizeof(float),wire,bytes-1u,&error)!=FG_OK);
        }
    }
    setenv("FG_FP16_HOPS","1",1);setenv("FG_FP16_HOP_MASK","1",1);
    assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
    assert(!(wire[3]&(FG_LAYER_WORK_WIRE_FP16|FG_LAYER_WORK_WIRE_FP16_128)));
    setenv("FG_FP16_HOP_MASK","2",1);
    assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_OK);
    assert(wire[3]&FG_LAYER_WORK_WIRE_FP16);
    setenv("FG_BF16_HOPS","1",1);
    assert(fg_decode_layer_work_encode(wire,sizeof(wire),&bytes,6,&source,&error)==FG_ERR_ARGUMENT);
    unsetenv("FG_FP16_HOPS");unsetenv("FG_FP16_HOP_MASK");
    unsetenv("FG_BF16_HOPS");puts("BF16_HOPS codec/RNE/default/mixed-peer/truncation/batch/MTP-guard PASS");
    return 0;
}
