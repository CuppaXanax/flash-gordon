#include "fg_qsa_state.h"
#include "fg_protocol.h"
#include "fg_quant.h"
#include "fg_uring.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define FG_QSA_PAGE_MAGIC UINT32_C(0x31534151) /* QAS1, little endian */
#define FG_QSA_PAGE_VERSION 2u
#define FG_QSA_PAGE_HEADER_BYTES 32u
#define FG_QSA_FILE_MAGIC UINT32_C(0x31534651) /* QFS1, little endian */
#define FG_QSA_FILE_VERSION 2u
#define FG_QSA_FILE_CRC_OFFSET 400u

#if FG_QSA_PAGE_HEADER_BYTES + FG_Q38_QSA_COMPRESS_RATIO * FG_Q38_QSA_TOKEN_RECORD_BYTES > FG_Q38_QSA_STATE_PAGE_BYTES
#error "QSA token records do not fit in one state page"
#endif

struct fg_qsa_state {int fd;fg_uring *ring;uint32_t slot,layer_count,max_context,blocks_per_layer;uint8_t layers[FG_LAYER_COUNT];uint32_t layer_tokens[FG_LAYER_COUNT];uint8_t *page_pool;};

static void put_u32_le(uint8_t *p,uint32_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8u);p[2]=(uint8_t)(v>>16u);p[3]=(uint8_t)(v>>24u);}
static uint32_t get_u32_le(const uint8_t *p){return (uint32_t)p[0]|((uint32_t)p[1]<<8u)|((uint32_t)p[2]<<16u)|((uint32_t)p[3]<<24u);}

uint64_t fg_qsa_state_required_bytes(uint32_t layers,uint32_t max_context){
    if(!layers||layers>FG_LAYER_COUNT||!max_context)return 0;
    return FG_Q38_QSA_STATE_PAGE_BYTES+(uint64_t)layers*((max_context+FG_Q38_QSA_COMPRESS_RATIO-1u)/FG_Q38_QSA_COMPRESS_RATIO)*FG_Q38_QSA_STATE_PAGE_BYTES;
}

void fg_qsa_encode_token_record(const float key[FG_Q38_ATTN_KV_WIDTH],const float value[FG_Q38_ATTN_KV_WIDTH],uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES]){
    if(!key||!value||!record)return;
    memset(record,0,FG_Q38_QSA_TOKEN_RECORD_BYTES);
    fg_quantize_q8_0(key,record,FG_Q38_ATTN_KV_WIDTH);fg_quantize_q8_0(value,record+FG_Q38_QSA_KEY_BYTES,FG_Q38_ATTN_KV_WIDTH);
}

void fg_qsa_encode_full_token_record(const float key[FG_Q38_ATTN_KV_WIDTH],const float value[FG_Q38_ATTN_KV_WIDTH],const float index_key[FG_Q38_INDEX_WIDTH],const uint32_t position[3],uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES]){
    if(!key||!value||!index_key||!position||!record)return;
    fg_quantize_q8_0(key,record,FG_Q38_ATTN_KV_WIDTH);
    fg_quantize_q8_0(value,record+FG_Q38_QSA_KEY_BYTES,FG_Q38_ATTN_KV_WIDTH);
    fg_quantize_q8_0(index_key,record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES,FG_Q38_INDEX_WIDTH);
    for(uint32_t axis=0;axis<3u;axis++)put_u32_le(record+FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES+FG_Q38_QSA_INDEX_KEY_BYTES+axis*4u,position[axis]);
}

void fg_qsa_decode_token_record(const uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES],float key[FG_Q38_ATTN_KV_WIDTH],float value[FG_Q38_ATTN_KV_WIDTH]){
    if(!key||!value||!record)return;
    fg_dequantize_q8_0(record,key,FG_Q38_ATTN_KV_WIDTH);fg_dequantize_q8_0(record+FG_Q38_QSA_KEY_BYTES,value,FG_Q38_ATTN_KV_WIDTH);
}

void fg_qsa_decode_token_metadata(const uint8_t record[FG_Q38_QSA_TOKEN_RECORD_BYTES],float index_key[FG_Q38_INDEX_WIDTH],uint32_t position[3]){
    if(!record||!index_key||!position)return;
    const uint32_t offset=FG_Q38_QSA_KEY_BYTES+FG_Q38_QSA_VALUE_BYTES;
    fg_dequantize_q8_0(record+offset,index_key,FG_Q38_INDEX_WIDTH);
    for(uint32_t axis=0;axis<3u;axis++)position[axis]=get_u32_le(record+offset+FG_Q38_QSA_INDEX_KEY_BYTES+axis*4u);
}

static uint64_t page_offset(const fg_qsa_state *state,uint32_t layer_slot,uint32_t block){return (1u+(uint64_t)layer_slot*state->blocks_per_layer+block)*FG_Q38_QSA_STATE_PAGE_BYTES;}

static void encode_file_header(const fg_qsa_state *state,uint8_t *page){
    memset(page,0,FG_Q38_QSA_STATE_PAGE_BYTES);put_u32_le(page,FG_QSA_FILE_MAGIC);put_u32_le(page+4u,FG_QSA_FILE_VERSION);put_u32_le(page+8u,state->layer_count);put_u32_le(page+12u,state->max_context);
    for(uint32_t i=0;i<state->layer_count;i++){put_u32_le(page+16u+i*8u,state->layers[i]);put_u32_le(page+20u+i*8u,state->layer_tokens[i]);}
    put_u32_le(page+FG_QSA_FILE_CRC_OFFSET,fg_crc32c(page,FG_QSA_FILE_CRC_OFFSET));
}

static fg_status write_file_header(fg_qsa_state *state,fg_error *err){encode_file_header(state,state->page_pool);return fg_uring_pwrite(state->ring,state->slot,state->page_pool,FG_Q38_QSA_STATE_PAGE_BYTES,0,err);}

static fg_status read_file_header(fg_qsa_state *state,fg_error *err){
    fg_status status=fg_uring_pread(state->ring,state->slot,state->page_pool,FG_Q38_QSA_STATE_PAGE_BYTES,0,err);if(status!=FG_OK)return status;
    const uint8_t *page=state->page_pool;if(get_u32_le(page)!=FG_QSA_FILE_MAGIC||get_u32_le(page+4u)!=FG_QSA_FILE_VERSION||get_u32_le(page+8u)!=state->layer_count||get_u32_le(page+12u)!=state->max_context||get_u32_le(page+FG_QSA_FILE_CRC_OFFSET)!=fg_crc32c(page,FG_QSA_FILE_CRC_OFFSET)){fg_error_set(err,FG_ERR_MISMATCH,"stale, torn, or corrupt QSA state header");return FG_ERR_MISMATCH;}
    for(uint32_t i=0;i<state->layer_count;i++){uint32_t layer=get_u32_le(page+16u+i*8u),tokens=get_u32_le(page+20u+i*8u);if(layer!=state->layers[i]||tokens>state->max_context){fg_error_set(err,FG_ERR_MISMATCH,"QSA state identity or committed length mismatch");return FG_ERR_MISMATCH;}state->layer_tokens[i]=tokens;}
    return FG_OK;
}

fg_status fg_qsa_state_open(fg_qsa_state **out,const char *path,const uint8_t *layers,uint32_t layer_count,uint32_t max_context,bool create,fg_error *err){
    if(!out||!path||!layers||!layer_count||layer_count>FG_LAYER_COUNT||!max_context){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state open arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    bool seen[FG_LAYER_COUNT]={0};for(uint32_t i=0;i<layer_count;i++)if(layers[i]>=FG_LAYER_COUNT||(layers[i]&3u)!=3u||seen[layers[i]]){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA layer list");return FG_ERR_ARGUMENT;}else seen[layers[i]]=true;
    uint64_t required=fg_qsa_state_required_bytes(layer_count,max_context);if(required>INT64_MAX){fg_error_set(err,FG_ERR_LIMIT,"QSA state file is too large");return FG_ERR_LIMIT;}
    int flags=O_RDWR|O_DIRECT|O_CLOEXEC|(create?(O_CREAT|O_EXCL):0);int fd=open(path,flags,0600);if(fd<0){fg_error_set(err,FG_ERR_IO,"open QSA state: %s",strerror(errno));return FG_ERR_IO;}
    fg_status status=FG_OK;if(create){int rc=posix_fallocate(fd,0,(off_t)required);if(rc){fg_error_set(err,FG_ERR_IO,"preallocate QSA state: %s",strerror(rc));status=FG_ERR_IO;}}else{struct stat st;if(fstat(fd,&st)!=0||(uint64_t)st.st_size!=required){fg_error_set(err,FG_ERR_MISMATCH,"QSA state file size mismatch");status=FG_ERR_MISMATCH;}}
    fg_qsa_state *state=status==FG_OK?calloc(1,sizeof(*state)):NULL;if(status==FG_OK&&!state){fg_error_set(err,FG_ERR_OOM,"allocate QSA state");status=FG_ERR_OOM;}
    const uint64_t pool_bytes=(uint64_t)FG_QSA_MAX_SELECTED_BLOCKS*FG_Q38_QSA_STATE_PAGE_BYTES;if(status==FG_OK&&posix_memalign((void **)&state->page_pool,FG_Q38_QSA_STATE_PAGE_BYTES,(size_t)pool_bytes)!=0){fg_error_set(err,FG_ERR_OOM,"allocate aligned QSA page pool");status=FG_ERR_OOM;}
    if(status==FG_OK){state->fd=fd;state->layer_count=layer_count;state->max_context=max_context;state->blocks_per_layer=(max_context+3u)/4u;memcpy(state->layers,layers,layer_count);status=fg_uring_create(&state->ring,FG_RING_STORAGE,1024u,err);}
    if(status==FG_OK)status=fg_uring_register_file(state->ring,fd,&state->slot,err);
    if(status==FG_OK)status=fg_uring_register_buffer(state->ring,state->page_pool,pool_bytes,err);
    if(status==FG_OK)status=create?write_file_header(state,err):read_file_header(state,err);
    if(status!=FG_OK){
        fg_error original={.code=status};if(err)original=*err;
        if(state){fg_uring_destroy(state->ring);free(state->page_pool);free(state);}
        close(fd);if(create)unlink(path);if(err)*err=original;return status;
    }
    *out=state;return FG_OK;
}

void fg_qsa_state_close(fg_qsa_state *state){if(!state)return;fg_uring_destroy(state->ring);close(state->fd);free(state->page_pool);free(state);}

fg_status fg_qsa_state_write_block(fg_qsa_state *state,uint32_t layer_slot,uint32_t block,const uint8_t *records,uint32_t committed,fg_error *err){
    if(!state||!records||layer_slot>=state->layer_count||block>=state->blocks_per_layer||committed==0||committed>FG_Q38_QSA_COMPRESS_RATIO){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state write");return FG_ERR_ARGUMENT;}
    uint32_t current=state->layer_tokens[layer_slot],end=block*FG_Q38_QSA_COMPRESS_RATIO+committed;if(block!=current/FG_Q38_QSA_COMPRESS_RATIO||end<current||end>state->max_context){fg_error_set(err,FG_ERR_MISMATCH,"non-contiguous QSA state write");return FG_ERR_MISMATCH;}
    uint8_t *page=state->page_pool;memset(page,0,FG_Q38_QSA_STATE_PAGE_BYTES);put_u32_le(page,FG_QSA_PAGE_MAGIC);put_u32_le(page+4u,FG_QSA_PAGE_VERSION);put_u32_le(page+8u,state->layers[layer_slot]);put_u32_le(page+12u,block);put_u32_le(page+16u,committed);uint32_t bytes=committed*FG_Q38_QSA_TOKEN_RECORD_BYTES;memcpy(page+FG_QSA_PAGE_HEADER_BYTES,records,bytes);put_u32_le(page+20u,fg_crc32c(page+FG_QSA_PAGE_HEADER_BYTES,bytes));fg_status status=fg_uring_pwrite(state->ring,state->slot,page,FG_Q38_QSA_STATE_PAGE_BYTES,page_offset(state,layer_slot,block),err);if(status!=FG_OK)return status;state->layer_tokens[layer_slot]=end;return write_file_header(state,err);
}

fg_status fg_qsa_state_write_blocks(fg_qsa_state *state,uint32_t layer_slot,
                                    const uint32_t *blocks,uint32_t count,
                                    const uint8_t *records,fg_error *err){
    if(!state||!blocks||!count||count>FG_QSA_MAX_SELECTED_BLOCKS||!records||
       layer_slot>=state->layer_count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state write batch");
        return FG_ERR_ARGUMENT;
    }
    uint32_t first=state->layer_tokens[layer_slot]/FG_Q38_QSA_COMPRESS_RATIO;
    if(state->layer_tokens[layer_slot]%FG_Q38_QSA_COMPRESS_RATIO||
       first>state->blocks_per_layer||count>state->blocks_per_layer-first){
        fg_error_set(err,FG_ERR_MISMATCH,"QSA state write batch frontier is not page aligned");
        return FG_ERR_MISMATCH;
    }
    fg_uring_read writes[FG_QSA_MAX_SELECTED_BLOCKS];
    for(uint32_t i=0;i<count;i++){
        if(blocks[i]!=first+i){
            fg_error_set(err,FG_ERR_MISMATCH,"non-contiguous QSA state write batch");
            return FG_ERR_MISMATCH;
        }
        uint8_t *page=state->page_pool+(uint64_t)i*FG_Q38_QSA_STATE_PAGE_BYTES;
        memset(page,0,FG_Q38_QSA_STATE_PAGE_BYTES);
        put_u32_le(page,FG_QSA_PAGE_MAGIC);put_u32_le(page+4u,FG_QSA_PAGE_VERSION);
        put_u32_le(page+8u,state->layers[layer_slot]);put_u32_le(page+12u,blocks[i]);
        put_u32_le(page+16u,FG_Q38_QSA_COMPRESS_RATIO);
        const uint8_t *source=records+(uint64_t)i*FG_QSA_PAGE_RECORD_BYTES;
        memcpy(page+FG_QSA_PAGE_HEADER_BYTES,source,FG_QSA_PAGE_RECORD_BYTES);
        put_u32_le(page+20u,fg_crc32c(source,FG_QSA_PAGE_RECORD_BYTES));
        writes[i]=(fg_uring_read){page,FG_Q38_QSA_STATE_PAGE_BYTES,
                                  page_offset(state,layer_slot,blocks[i])};
    }
    fg_status status=fg_uring_pwrite_batch(state->ring,state->slot,writes,count,err);
    if(status!=FG_OK)return status;
    state->layer_tokens[layer_slot]+=
        count*FG_Q38_QSA_COMPRESS_RATIO;
    return write_file_header(state,err);
}

uint32_t fg_qsa_state_layer_tokens(const fg_qsa_state *state,uint32_t layer_slot){return state&&layer_slot<state->layer_count?state->layer_tokens[layer_slot]:0;}
void fg_qsa_state_set_layer_tokens(fg_qsa_state *state,uint32_t layer_slot,uint32_t tokens){if(state&&layer_slot<state->layer_count)state->layer_tokens[layer_slot]=tokens;}
fg_status fg_qsa_state_reset(fg_qsa_state *state,fg_error *err){if(!state){fg_error_set(err,FG_ERR_ARGUMENT,"QSA state reset is null");return FG_ERR_ARGUMENT;}memset(state->layer_tokens,0,sizeof(state->layer_tokens));return write_file_header(state,err);}

static fg_status decode_page(const fg_qsa_state *state,const uint8_t *page,uint32_t layer_slot,uint32_t block,uint8_t *records,uint32_t *committed,fg_error *err){uint32_t count=get_u32_le(page+16u),bytes=count*FG_Q38_QSA_TOKEN_RECORD_BYTES;if(get_u32_le(page)!=FG_QSA_PAGE_MAGIC||get_u32_le(page+4u)!=FG_QSA_PAGE_VERSION||get_u32_le(page+8u)!=state->layers[layer_slot]||get_u32_le(page+12u)!=block||count==0||count>FG_Q38_QSA_COMPRESS_RATIO||get_u32_le(page+20u)!=fg_crc32c(page+FG_QSA_PAGE_HEADER_BYTES,bytes)){fg_error_set(err,FG_ERR_MISMATCH,"stale, torn, or corrupt QSA state page");return FG_ERR_MISMATCH;}memcpy(records,page+FG_QSA_PAGE_HEADER_BYTES,bytes);if(count<FG_Q38_QSA_COMPRESS_RATIO)memset(records+(uint64_t)count*FG_Q38_QSA_TOKEN_RECORD_BYTES,0,(FG_Q38_QSA_COMPRESS_RATIO-count)*FG_Q38_QSA_TOKEN_RECORD_BYTES);*committed=count;return FG_OK;}

fg_status fg_qsa_state_read_block(fg_qsa_state *state,uint32_t layer_slot,uint32_t block,uint8_t *records,uint32_t *committed,fg_error *err){
    if(!state||!records||!committed||layer_slot>=state->layer_count||block>=state->blocks_per_layer){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state read");return FG_ERR_ARGUMENT;}
    uint8_t *page=state->page_pool;fg_status status=fg_uring_pread(state->ring,state->slot,page,FG_Q38_QSA_STATE_PAGE_BYTES,page_offset(state,layer_slot,block),err);if(status!=FG_OK)return status;return decode_page(state,page,layer_slot,block,records,committed,err);
}

fg_status fg_qsa_state_read_blocks(fg_qsa_state *state,uint32_t layer_slot,const uint32_t *blocks,uint32_t count,uint8_t *records,uint32_t *committed,fg_error *err){if(!state||!blocks||!records||!committed||layer_slot>=state->layer_count||!count||count>FG_QSA_MAX_SELECTED_BLOCKS){fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA state read batch");return FG_ERR_ARGUMENT;}fg_uring_read reads[FG_QSA_MAX_SELECTED_BLOCKS];for(uint32_t i=0;i<count;i++){if(blocks[i]>=state->blocks_per_layer){fg_error_set(err,FG_ERR_ARGUMENT,"QSA state block %u is out of range",blocks[i]);return FG_ERR_ARGUMENT;}reads[i]=(fg_uring_read){state->page_pool+(uint64_t)i*FG_Q38_QSA_STATE_PAGE_BYTES,FG_Q38_QSA_STATE_PAGE_BYTES,page_offset(state,layer_slot,blocks[i])};}fg_status status=fg_uring_pread_batch(state->ring,state->slot,reads,count,err);for(uint32_t i=0;status==FG_OK&&i<count;i++)status=decode_page(state,reads[i].buffer,layer_slot,blocks[i],records+(uint64_t)i*FG_Q38_QSA_COMPRESS_RATIO*FG_Q38_QSA_TOKEN_RECORD_BYTES,&committed[i],err);return status;}
