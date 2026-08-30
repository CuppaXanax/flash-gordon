#include "fg_tokenizer.h"
#include "fg_q38_schema.h"
#include "fg_sha256.h"
#include "fg_uring.h"

#include <errno.h>
#include <fcntl.h>
#include <locale.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <wctype.h>

#define GGUF_MAGIC UINT32_C(0x46554747)
#define FG_TOKENIZER_MAGIC UINT64_C(0x314e4b4f544746) /* FGTOKN1 */
#define FG_TOKENIZER_VERSION 1u
#define FG_TOKENIZER_MAX_ITEMS 2000000u
#define FG_TOKENIZER_MAX_STRING (16u*1024u*1024u)

typedef struct string_array {char **items;uint32_t *lengths;uint64_t count;} string_array;
typedef struct tokenizer_metadata {string_array tokens,merges;int32_t *types;uint64_t type_count;uint32_t bos,eos;bool have_bos,have_eos,add_bos;} tokenizer_metadata;
typedef struct token_slice {const char *text;uint32_t bytes,type;} token_slice;
typedef struct token_hash_entry {uint64_t hash;uint32_t token;bool used;} token_hash_entry;
typedef struct merge_hash_entry {uint64_t hash;uint32_t rank;bool used;} merge_hash_entry;
struct fg_tokenizer {void *storage;uint64_t storage_bytes;token_slice *tokens,*merges;uint32_t *special_ids;uint32_t vocab_count,merge_count,special_count,bos,eos;bool add_bos;token_hash_entry *token_table;merge_hash_entry *merge_table;uint32_t table_capacity,merge_capacity;};

static bool read_exact(FILE *file,void *data,size_t bytes){return fread(data,1,bytes,file)==bytes;}
static bool read_u32(FILE *file,uint32_t *value){return read_exact(file,value,4u);}
static bool read_u64(FILE *file,uint64_t *value){return read_exact(file,value,8u);}
static char *read_string(FILE *file,uint32_t *length){uint64_t bytes;if(!read_u64(file,&bytes)||bytes>FG_TOKENIZER_MAX_STRING)return NULL;char *text=malloc((size_t)bytes+1u);if(!text)return NULL;if(!read_exact(file,text,(size_t)bytes)){free(text);return NULL;}text[bytes]=0;if(length)*length=(uint32_t)bytes;return text;}
static bool skip_bytes(FILE *file,uint64_t bytes){while(bytes){long step=bytes>0x3fffffffu?0x3fffffffu:(long)bytes;if(fseek(file,step,SEEK_CUR)!=0)return false;bytes-=(uint64_t)step;}return true;}
static bool skip_value(FILE *file,uint32_t type){static const uint8_t scalar[13]={1,1,2,2,4,4,4,1,0,0,8,8,8};if(type<13u&&scalar[type])return skip_bytes(file,scalar[type]);if(type==8u){char *value=read_string(file,NULL);free(value);return value!=NULL;}if(type==9u){uint32_t element;uint64_t count;if(!read_u32(file,&element)||!read_u64(file,&count)||count>FG_TOKENIZER_MAX_ITEMS)return false;for(uint64_t i=0;i<count;i++)if(!skip_value(file,element))return false;return true;}return false;}
static void free_strings(string_array *array){if(!array)return;for(uint64_t i=0;i<array->count;i++)free(array->items[i]);free(array->lengths);free(array->items);memset(array,0,sizeof(*array));}
static void metadata_free(tokenizer_metadata *metadata){free_strings(&metadata->tokens);free_strings(&metadata->merges);free(metadata->types);memset(metadata,0,sizeof(*metadata));}
static bool read_string_array(FILE *file,string_array *array){uint32_t element;uint64_t count;if(!read_u32(file,&element)||element!=8u||!read_u64(file,&count)||!count||count>FG_TOKENIZER_MAX_ITEMS)return false;array->items=calloc((size_t)count,sizeof(*array->items));array->lengths=calloc((size_t)count,sizeof(*array->lengths));if(!array->items||!array->lengths)return false;array->count=count;for(uint64_t i=0;i<count;i++){array->items[i]=read_string(file,&array->lengths[i]);if(!array->items[i])return false;}return true;}
static bool read_i32_array(FILE *file,int32_t **values,uint64_t *count){uint32_t element;uint64_t n;if(!read_u32(file,&element)||element!=5u||!read_u64(file,&n)||!n||n>FG_TOKENIZER_MAX_ITEMS)return false;int32_t *data=malloc((size_t)n*sizeof(*data));if(!data||!read_exact(file,data,(size_t)n*sizeof(*data))){free(data);return false;}*values=data;*count=n;return true;}

static fg_status read_metadata(const char *path,tokenizer_metadata *metadata,fg_error *err){
    FILE *file=fopen(path,"rb");if(!file){fg_error_set(err,FG_ERR_IO,"open tokenizer source %s: %s",path,strerror(errno));return FG_ERR_IO;}
    uint32_t magic,version;uint64_t tensors,kv;if(!read_u32(file,&magic)||!read_u32(file,&version)||!read_u64(file,&tensors)||!read_u64(file,&kv)||magic!=GGUF_MAGIC||version<2u||version>3u||kv>(1u<<20)){fclose(file);fg_error_set(err,FG_ERR_FORMAT,"invalid tokenizer GGUF header");return FG_ERR_FORMAT;}
    (void)tensors;bool ok=true;
    for(uint64_t i=0;ok&&i<kv;i++){
        char *key=read_string(file,NULL);uint32_t type;if(!key||!read_u32(file,&type)){free(key);ok=false;break;}
        if(strcmp(key,"tokenizer.ggml.tokens")==0&&type==9u)ok=read_string_array(file,&metadata->tokens);
        else if(strcmp(key,"tokenizer.ggml.merges")==0&&type==9u)ok=read_string_array(file,&metadata->merges);
        else if(strcmp(key,"tokenizer.ggml.token_type")==0&&type==9u)ok=read_i32_array(file,&metadata->types,&metadata->type_count);
        else if(strcmp(key,"tokenizer.ggml.bos_token_id")==0&&type==4u){ok=read_u32(file,&metadata->bos);metadata->have_bos=ok;}
        else if(strcmp(key,"tokenizer.ggml.eos_token_id")==0&&type==4u){ok=read_u32(file,&metadata->eos);metadata->have_eos=ok;}
        else if(strcmp(key,"tokenizer.ggml.add_bos_token")==0&&type==7u){uint8_t value;ok=read_exact(file,&value,1u);metadata->add_bos=value!=0;}
        else ok=skip_value(file,type);
        free(key);
    }
    fclose(file);
    if(!ok||metadata->tokens.count!=248320u||metadata->type_count!=metadata->tokens.count||!metadata->merges.count||!metadata->have_bos||!metadata->have_eos||metadata->bos>=metadata->tokens.count||metadata->eos>=metadata->tokens.count){metadata_free(metadata);fg_error_set(err,FG_ERR_MISMATCH,"Qwen tokenizer metadata is missing or inconsistent");return FG_ERR_MISMATCH;}
    return FG_OK;
}

static bool write_u32(FILE *file,uint32_t value){return fwrite(&value,1,4u,file)==4u;}
static bool write_u64(FILE *file,uint64_t value){return fwrite(&value,1,8u,file)==8u;}
static bool write_string(FILE *file,const char *text,uint32_t bytes){return write_u32(file,bytes)&&fwrite(text,1,bytes,file)==bytes;}

fg_status fg_tokenizer_pack_gguf(const char *source_path,const char *output_dir,fg_manifest *manifest,fg_error *err){
    if(!source_path||!output_dir||!manifest){fg_error_set(err,FG_ERR_ARGUMENT,"invalid tokenizer pack arguments");return FG_ERR_ARGUMENT;}
    tokenizer_metadata metadata={0};fg_status status=read_metadata(source_path,&metadata,err);if(status!=FG_OK)return status;
    char directory[1024],path[1200];if(snprintf(directory,sizeof(directory),"%s/tokenizer",output_dir)>=(int)sizeof(directory)||snprintf(path,sizeof(path),"%s/tokenizer.fgt",directory)>=(int)sizeof(path)){metadata_free(&metadata);fg_error_set(err,FG_ERR_LIMIT,"tokenizer output path is too long");return FG_ERR_LIMIT;}
    if(mkdir(directory,0755)!=0&&errno!=EEXIST){metadata_free(&metadata);fg_error_set(err,FG_ERR_IO,"create %s: %s",directory,strerror(errno));return FG_ERR_IO;}
    FILE *file=fopen(path,"wb");if(!file){metadata_free(&metadata);fg_error_set(err,FG_ERR_IO,"create %s: %s",path,strerror(errno));return FG_ERR_IO;}
    bool ok=write_u64(file,FG_TOKENIZER_MAGIC)&&write_u32(file,FG_TOKENIZER_VERSION)&&write_u32(file,(uint32_t)metadata.tokens.count)&&write_u32(file,(uint32_t)metadata.merges.count)&&write_u32(file,metadata.bos)&&write_u32(file,metadata.eos)&&write_u32(file,metadata.add_bos?1u:0u);
    for(uint64_t i=0;ok&&i<metadata.tokens.count;i++)ok=write_string(file,metadata.tokens.items[i],metadata.tokens.lengths[i])&&write_u32(file,(uint32_t)metadata.types[i]);
    for(uint64_t i=0;ok&&i<metadata.merges.count;i++)ok=write_string(file,metadata.merges.items[i],metadata.merges.lengths[i]);
    if(ok){long end=ftell(file);if(end<0)ok=false;else{uint64_t padded=fg_align_up_u64((uint64_t)end,FG_ALIGNMENT);static const uint8_t zeros[FG_ALIGNMENT]={0};while(ok&&(uint64_t)end<padded){size_t bytes=(size_t)(padded-(uint64_t)end);if(bytes>sizeof(zeros))bytes=sizeof(zeros);ok=fwrite(zeros,1,bytes,file)==bytes;end+=(long)bytes;}}}
    if(ok)ok=fflush(file)==0&&fsync(fileno(file))==0;
    int close_rc=fclose(file);if(close_rc!=0)ok=false;metadata_free(&metadata);
    if(!ok){unlink(path);fg_error_set(err,FG_ERR_IO,"write tokenizer asset %s: %s",path,strerror(errno));return FG_ERR_IO;}
    struct stat info;if(stat(path,&info)!=0){fg_error_set(err,FG_ERR_IO,"stat tokenizer asset: %s",strerror(errno));return FG_ERR_IO;}
    fg_tensor_record record={0};snprintf(record.name,sizeof(record.name),"tokenizer/tokenizer.fgt");record.bytes=(uint64_t)info.st_size;record.dims=1u;record.shape[0]=record.bytes;record.rank=UINT16_MAX;record.layer=UINT16_MAX;record.expert=UINT16_MAX;record.kind=FG_TENSOR_TOKENIZER;
    status=fg_sha256_file(path,record.sha256,err);if(status==FG_OK)status=fg_manifest_add_tensor(manifest,&record,err);if(status==FG_OK)manifest->flags|=FG_MANIFEST_HAS_TOKENIZER;return status;
}

typedef struct asset_cursor {const uint8_t *data;uint64_t bytes,offset;} asset_cursor;
static bool cursor_u32(asset_cursor *cursor,uint32_t *value){if(cursor->offset>cursor->bytes||4u>cursor->bytes-cursor->offset)return false;memcpy(value,cursor->data+cursor->offset,4u);cursor->offset+=4u;return true;}
static bool cursor_u64(asset_cursor *cursor,uint64_t *value){if(cursor->offset>cursor->bytes||8u>cursor->bytes-cursor->offset)return false;memcpy(value,cursor->data+cursor->offset,8u);cursor->offset+=8u;return true;}
static bool cursor_slice(asset_cursor *cursor,token_slice *slice,bool with_type){uint32_t bytes;if(!cursor_u32(cursor,&bytes)||cursor->offset>cursor->bytes||bytes>cursor->bytes-cursor->offset)return false;slice->text=(const char *)cursor->data+cursor->offset;slice->bytes=bytes;cursor->offset+=bytes;if(with_type&&!cursor_u32(cursor,&slice->type))return false;return true;}
static uint64_t string_hash(const char *text,size_t bytes){uint64_t hash=UINT64_C(1469598103934665603);for(size_t i=0;i<bytes;i++){hash^=(uint8_t)text[i];hash*=UINT64_C(1099511628211);}return hash;}
static bool token_equal(const token_slice *slice,const char *text,size_t bytes){return slice->bytes==bytes&&memcmp(slice->text,text,bytes)==0;}

fg_status fg_tokenizer_open(fg_tokenizer **out,const char *pack_dir,const fg_manifest *manifest,fg_error *err){
    if(!out||!pack_dir||!manifest){fg_error_set(err,FG_ERR_ARGUMENT,"invalid tokenizer open arguments");return FG_ERR_ARGUMENT;}*out=NULL;
    const fg_tensor_record *record=NULL;for(uint32_t i=0;i<manifest->tensor_count;i++)if(manifest->tensors[i].kind==FG_TENSOR_TOKENIZER&&strcmp(manifest->tensors[i].name,"tokenizer/tokenizer.fgt")==0){if(record){fg_error_set(err,FG_ERR_MISMATCH,"duplicate tokenizer artifact record");return FG_ERR_MISMATCH;}record=&manifest->tensors[i];}
    if(!record||record->rank!=UINT16_MAX||!record->bytes||record->bytes%FG_ALIGNMENT||record->bytes>UINT32_MAX){fg_error_set(err,FG_ERR_MISMATCH,"invalid sealed tokenizer artifact record");return FG_ERR_MISMATCH;}
    fg_tokenizer *tokenizer=calloc(1,sizeof(*tokenizer));if(!tokenizer){fg_error_set(err,FG_ERR_OOM,"allocate tokenizer");return FG_ERR_OOM;}
    char path[1200];if(snprintf(path,sizeof(path),"%s/%s",pack_dir,record->name)>=(int)sizeof(path)){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_LIMIT,"tokenizer path is too long");return FG_ERR_LIMIT;}
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_DIRECT);if(fd<0){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_IO,"open tokenizer asset %s: %s",path,strerror(errno));return FG_ERR_IO;}
    if(posix_memalign(&tokenizer->storage,FG_ALIGNMENT,(size_t)record->bytes)!=0){close(fd);fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_OOM,"allocate aligned tokenizer asset");return FG_ERR_OOM;}tokenizer->storage_bytes=record->bytes;
    fg_uring *ring=NULL;uint32_t slot=0;fg_status status=fg_uring_create(&ring,FG_RING_STORAGE,8u,err);if(status==FG_OK)status=fg_uring_register_file(ring,fd,&slot,err);if(status==FG_OK)status=fg_uring_register_buffer(ring,tokenizer->storage,record->bytes,err);if(status==FG_OK)status=fg_uring_pread(ring,slot,tokenizer->storage,(uint32_t)record->bytes,0,err);fg_uring_destroy(ring);close(fd);if(status!=FG_OK){fg_tokenizer_close(tokenizer);return status;}
    uint8_t digest[32];fg_sha256 hash;fg_sha256_init(&hash);fg_sha256_update(&hash,tokenizer->storage,(size_t)record->bytes);fg_sha256_final(&hash,digest);if(memcmp(digest,record->sha256,32u)!=0){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_MISMATCH,"tokenizer artifact SHA-256 mismatch");return FG_ERR_MISMATCH;}
    asset_cursor cursor={(const uint8_t *)tokenizer->storage,record->bytes,0};uint64_t magic;uint32_t version,add_bos;if(!cursor_u64(&cursor,&magic)||!cursor_u32(&cursor,&version)||!cursor_u32(&cursor,&tokenizer->vocab_count)||!cursor_u32(&cursor,&tokenizer->merge_count)||!cursor_u32(&cursor,&tokenizer->bos)||!cursor_u32(&cursor,&tokenizer->eos)||!cursor_u32(&cursor,&add_bos)||magic!=FG_TOKENIZER_MAGIC||version!=FG_TOKENIZER_VERSION||tokenizer->vocab_count!=248320u||!tokenizer->merge_count||tokenizer->bos>=tokenizer->vocab_count||tokenizer->eos>=tokenizer->vocab_count||add_bos>1u){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_FORMAT,"invalid tokenizer asset header");return FG_ERR_FORMAT;}tokenizer->add_bos=add_bos!=0;
    tokenizer->tokens=calloc(tokenizer->vocab_count,sizeof(*tokenizer->tokens));tokenizer->merges=calloc(tokenizer->merge_count,sizeof(*tokenizer->merges));if(!tokenizer->tokens||!tokenizer->merges){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_OOM,"allocate tokenizer tables");return FG_ERR_OOM;}
    for(uint32_t i=0;i<tokenizer->vocab_count;i++)if(!cursor_slice(&cursor,&tokenizer->tokens[i],true)){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_FORMAT,"truncated tokenizer token %u",i);return FG_ERR_FORMAT;}
    for(uint32_t i=0;i<tokenizer->merge_count;i++)if(!cursor_slice(&cursor,&tokenizer->merges[i],false)){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_FORMAT,"truncated tokenizer merge %u",i);return FG_ERR_FORMAT;}
    for(uint64_t i=cursor.offset;i<cursor.bytes;i++)if(cursor.data[i]){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_FORMAT,"non-zero tokenizer padding");return FG_ERR_FORMAT;}
    for(uint32_t i=0;i<tokenizer->vocab_count;i++)if(tokenizer->tokens[i].type==3u||tokenizer->tokens[i].type==4u)tokenizer->special_count++;
    tokenizer->special_ids=malloc((size_t)tokenizer->special_count*sizeof(*tokenizer->special_ids));if(tokenizer->special_count&&!tokenizer->special_ids){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_OOM,"allocate special-token index");return FG_ERR_OOM;}for(uint32_t i=0,at=0;i<tokenizer->vocab_count;i++)if(tokenizer->tokens[i].type==3u||tokenizer->tokens[i].type==4u)tokenizer->special_ids[at++]=i;
    if(!setlocale(LC_CTYPE,"C.UTF-8")){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_UNAVAILABLE,"C.UTF-8 locale is required for Qwen tokenization");return FG_ERR_UNAVAILABLE;}
    uint32_t capacity=1u;while(capacity<tokenizer->vocab_count*2u)capacity<<=1u;tokenizer->token_table=calloc(capacity,sizeof(*tokenizer->token_table));if(!tokenizer->token_table){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_OOM,"allocate tokenizer hash table");return FG_ERR_OOM;}tokenizer->table_capacity=capacity;
    for(uint32_t token=0;token<tokenizer->vocab_count;token++){uint64_t h=string_hash(tokenizer->tokens[token].text,tokenizer->tokens[token].bytes);uint32_t index=(uint32_t)h&(capacity-1u);while(tokenizer->token_table[index].used){if(token_equal(&tokenizer->tokens[tokenizer->token_table[index].token],tokenizer->tokens[token].text,tokenizer->tokens[token].bytes)){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_MISMATCH,"duplicate tokenizer token bytes at %u",token);return FG_ERR_MISMATCH;}index=(index+1u)&(capacity-1u);}tokenizer->token_table[index]=(token_hash_entry){h,token,true};}
    capacity=1u;while((uint64_t)capacity<(uint64_t)tokenizer->merge_count*2u){if(capacity>UINT32_MAX/2u){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_LIMIT,"tokenizer merge table is too large");return FG_ERR_LIMIT;}capacity<<=1u;}tokenizer->merge_table=calloc(capacity,sizeof(*tokenizer->merge_table));if(!tokenizer->merge_table){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_OOM,"allocate merge-rank table");return FG_ERR_OOM;}tokenizer->merge_capacity=capacity;
    for(uint32_t rank=0;rank<tokenizer->merge_count;rank++){uint64_t h=string_hash(tokenizer->merges[rank].text,tokenizer->merges[rank].bytes);uint32_t index=(uint32_t)h&(capacity-1u);while(tokenizer->merge_table[index].used){const token_slice *prior=&tokenizer->merges[tokenizer->merge_table[index].rank];if(tokenizer->merge_table[index].hash==h&&token_equal(prior,tokenizer->merges[rank].text,tokenizer->merges[rank].bytes)){fg_tokenizer_close(tokenizer);fg_error_set(err,FG_ERR_MISMATCH,"duplicate tokenizer merge at %u",rank);return FG_ERR_MISMATCH;}index=(index+1u)&(capacity-1u);}tokenizer->merge_table[index]=(merge_hash_entry){h,rank,true};}
    *out=tokenizer;return FG_OK;
}

void fg_tokenizer_close(fg_tokenizer *tokenizer){if(!tokenizer)return;free(tokenizer->merge_table);free(tokenizer->token_table);free(tokenizer->special_ids);free(tokenizer->merges);free(tokenizer->tokens);free(tokenizer->storage);free(tokenizer);}
uint32_t fg_tokenizer_vocab_size(const fg_tokenizer *tokenizer){return tokenizer?tokenizer->vocab_count:0u;}
uint32_t fg_tokenizer_bos(const fg_tokenizer *tokenizer){return tokenizer?tokenizer->bos:UINT32_MAX;}
uint32_t fg_tokenizer_eos(const fg_tokenizer *tokenizer){return tokenizer?tokenizer->eos:UINT32_MAX;}
bool fg_tokenizer_add_bos(const fg_tokenizer *tokenizer){return tokenizer&&tokenizer->add_bos;}
fg_status fg_tokenizer_lookup(const fg_tokenizer *tokenizer,const char *text,size_t bytes,uint32_t *token,fg_error *err){if(!tokenizer||(!text&&bytes)||!token){fg_error_set(err,FG_ERR_ARGUMENT,"invalid tokenizer lookup");return FG_ERR_ARGUMENT;}uint64_t hash=string_hash(text,bytes);uint32_t index=(uint32_t)hash&(tokenizer->table_capacity-1u);for(uint32_t probes=0;probes<tokenizer->table_capacity;probes++){const token_hash_entry *entry=&tokenizer->token_table[index];if(!entry->used)break;if(entry->hash==hash&&token_equal(&tokenizer->tokens[entry->token],text,bytes)){*token=entry->token;return FG_OK;}index=(index+1u)&(tokenizer->table_capacity-1u);}fg_error_set(err,FG_ERR_MISMATCH,"tokenizer piece is absent from vocabulary");return FG_ERR_MISMATCH;}
fg_status fg_tokenizer_token(const fg_tokenizer *tokenizer,uint32_t token,const char **text,size_t *bytes,uint32_t *type,fg_error *err){if(!tokenizer||token>=tokenizer->vocab_count||!text||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"invalid tokenizer token lookup");return FG_ERR_ARGUMENT;}*text=tokenizer->tokens[token].text;*bytes=tokenizer->tokens[token].bytes;if(type)*type=tokenizer->tokens[token].type;return FG_OK;}

void fg_tokens_free(fg_tokens *tokens){if(!tokens)return;free(tokens->data);memset(tokens,0,sizeof(*tokens));}
static fg_status tokens_push(fg_tokens *tokens,uint32_t token,fg_error *err){if(tokens->count==tokens->capacity){size_t capacity=tokens->capacity?tokens->capacity*2u:256u;if(capacity<tokens->capacity||capacity>SIZE_MAX/sizeof(*tokens->data)){fg_error_set(err,FG_ERR_LIMIT,"token output exceeds address space");return FG_ERR_LIMIT;}uint32_t *data=realloc(tokens->data,capacity*sizeof(*data));if(!data){fg_error_set(err,FG_ERR_OOM,"grow token output");return FG_ERR_OOM;}tokens->data=data;tokens->capacity=capacity;}tokens->data[tokens->count++]=token;return FG_OK;}

static uint32_t utf8_decode(const char *text,size_t bytes,size_t offset,size_t *next){const uint8_t *p=(const uint8_t *)text;uint8_t a=p[offset];if(a<0x80u){*next=offset+1u;return a;}uint32_t value;size_t length;if((a&0xe0u)==0xc0u){value=a&0x1fu;length=2u;}else if((a&0xf0u)==0xe0u){value=a&0x0fu;length=3u;}else if((a&0xf8u)==0xf0u){value=a&7u;length=4u;}else{*next=offset+1u;return 0xfffdu;}if(offset+length>bytes){*next=offset+1u;return 0xfffdu;}for(size_t i=1;i<length;i++){uint8_t b=p[offset+i];if((b&0xc0u)!=0x80u){*next=offset+1u;return 0xfffdu;}value=(value<<6u)|(b&0x3fu);}if((length==2u&&value<0x80u)||(length==3u&&value<0x800u)||(length==4u&&(value<0x10000u||value>0x10ffffu))||(value>=0xd800u&&value<=0xdfffu)){*next=offset+1u;return 0xfffdu;}*next=offset+length;return value;}
static bool code_letter(uint32_t code){return iswalpha((wint_t)code)!=0;}
static bool code_number(uint32_t code){return iswdigit((wint_t)code)!=0||(iswalnum((wint_t)code)!=0&&!code_letter(code));}
static bool code_space(uint32_t code){return iswspace((wint_t)code)!=0;}
static bool code_newline(uint32_t code){return code=='\r'||code=='\n';}
static bool code_symbol(uint32_t code){return !code_space(code)&&!code_letter(code)&&!code_number(code);}

static size_t match_contraction(const char *text,size_t bytes,size_t offset){static const char *suffix[]={"'s","'t","'re","'ve","'m","'ll","'d"};if(offset>=bytes||text[offset]!='\'')return offset;for(size_t i=0;i<sizeof(suffix)/sizeof(suffix[0]);i++){size_t n=strlen(suffix[i]);if(offset+n<=bytes){bool equal=true;for(size_t j=0;j<n;j++){char a=text[offset+j],b=suffix[i][j];if(a>='A'&&a<='Z')a=(char)(a-'A'+'a');if(a!=b){equal=false;break;}}if(equal)return offset+n;}}return offset;}
static size_t consume_letters(const char *text,size_t bytes,size_t offset){while(offset<bytes){size_t next;uint32_t code=utf8_decode(text,bytes,offset,&next);if(!code_letter(code))break;offset=next;}return offset;}

static size_t next_piece(const char *text,size_t bytes,size_t offset){
    size_t next;uint32_t code=utf8_decode(text,bytes,offset,&next),after=match_contraction(text,bytes,offset);if(after>offset)return after;
    if(code_letter(code))return consume_letters(text,bytes,offset);
    if(!code_newline(code)&&!code_letter(code)&&!code_number(code)&&next<bytes){size_t second_next;uint32_t second=utf8_decode(text,bytes,next,&second_next);if(code_letter(second))return consume_letters(text,bytes,next);}
    if(code_number(code)){size_t position=offset;for(uint32_t count=0;count<3u&&position<bytes;count++){size_t n;uint32_t one=utf8_decode(text,bytes,position,&n);if(!code_number(one))break;position=n;}return position;}
    size_t position=offset;if(code==' '&&next<bytes){size_t n;uint32_t second=utf8_decode(text,bytes,next,&n);if(code_symbol(second)){position=next;code=second;next=n;}}
    if(code_symbol(code)){position=next;while(position<bytes){size_t n;uint32_t one=utf8_decode(text,bytes,position,&n);if(!code_symbol(one))break;position=n;}while(position<bytes&&(text[position]=='\r'||text[position]=='\n'))position++;return position;}
    if(code_space(code)){size_t scan=offset,last_newline=0;while(scan<bytes){size_t n;uint32_t one=utf8_decode(text,bytes,scan,&n);if(!code_space(one))break;scan=n;if(code_newline(one))last_newline=scan;}if(last_newline)return last_newline;if(scan<bytes&&scan>next)return scan-1u;return scan;}
    return next;
}

static uint32_t byte_codepoint(uint8_t byte){if((byte>=33u&&byte<=126u)||(byte>=161u&&byte<=172u)||byte>=174u)return byte;uint32_t ordinal=0;for(uint32_t value=0;value<256u;value++){if((value>=33u&&value<=126u)||(value>=161u&&value<=172u)||value>=174u)continue;if(value==byte)return 256u+ordinal;ordinal++;}return byte;}
static size_t utf8_encode(uint32_t code,char output[4]){if(code<=0x7fu){output[0]=(char)code;return 1u;}if(code<=0x7ffu){output[0]=(char)(0xc0u|(code>>6u));output[1]=(char)(0x80u|(code&0x3fu));return 2u;}if(code<=0xffffu){output[0]=(char)(0xe0u|(code>>12u));output[1]=(char)(0x80u|((code>>6u)&0x3fu));output[2]=(char)(0x80u|(code&0x3fu));return 3u;}output[0]=(char)(0xf0u|(code>>18u));output[1]=(char)(0x80u|((code>>12u)&0x3fu));output[2]=(char)(0x80u|((code>>6u)&0x3fu));output[3]=(char)(0x80u|(code&0x3fu));return 4u;}
static char *byte_encode(const char *text,size_t bytes,size_t *output_bytes){char *output=malloc(bytes*4u+1u);if(!output)return NULL;size_t used=0;for(size_t i=0;i<bytes;i++){char encoded[4];size_t n=utf8_encode(byte_codepoint((uint8_t)text[i]),encoded);memcpy(output+used,encoded,n);used+=n;}output[used]=0;*output_bytes=used;return output;}

typedef struct bpe_symbol {char *text;size_t bytes;} bpe_symbol;
static int32_t merge_rank(const fg_tokenizer *tokenizer,const bpe_symbol *left,const bpe_symbol *right){size_t bytes=left->bytes+1u+right->bytes;char stack[512],*key=bytes<=sizeof(stack)?stack:malloc(bytes);if(!key)return -2;memcpy(key,left->text,left->bytes);key[left->bytes]=' ';memcpy(key+left->bytes+1u,right->text,right->bytes);uint64_t hash=string_hash(key,bytes);uint32_t index=(uint32_t)hash&(tokenizer->merge_capacity-1u);int32_t rank=-1;for(uint32_t probes=0;probes<tokenizer->merge_capacity;probes++){const merge_hash_entry *entry=&tokenizer->merge_table[index];if(!entry->used)break;if(entry->hash==hash&&token_equal(&tokenizer->merges[entry->rank],key,bytes)){rank=(int32_t)entry->rank;break;}index=(index+1u)&(tokenizer->merge_capacity-1u);}if(key!=stack)free(key);return rank;}

static fg_status encode_piece(const fg_tokenizer *tokenizer,const char *text,size_t bytes,fg_tokens *tokens,fg_error *err){
    size_t encoded_bytes=0;char *encoded=byte_encode(text,bytes,&encoded_bytes);if(!encoded){fg_error_set(err,FG_ERR_OOM,"byte-encode tokenizer piece");return FG_ERR_OOM;}size_t count=0,capacity=32u;bpe_symbol *symbols=calloc(capacity,sizeof(*symbols));if(!symbols){free(encoded);fg_error_set(err,FG_ERR_OOM,"allocate BPE symbols");return FG_ERR_OOM;}
    for(size_t offset=0;offset<encoded_bytes;){size_t next;utf8_decode(encoded,encoded_bytes,offset,&next);if(count==capacity){capacity*=2u;bpe_symbol *grown=realloc(symbols,capacity*sizeof(*symbols));if(!grown){for(size_t i=0;i<count;i++)free(symbols[i].text);free(symbols);free(encoded);fg_error_set(err,FG_ERR_OOM,"grow BPE symbols");return FG_ERR_OOM;}symbols=grown;}symbols[count].bytes=next-offset;symbols[count].text=malloc(symbols[count].bytes);if(!symbols[count].text){for(size_t i=0;i<count;i++)free(symbols[i].text);free(symbols);free(encoded);fg_error_set(err,FG_ERR_OOM,"copy BPE symbol");return FG_ERR_OOM;}memcpy(symbols[count].text,encoded+offset,symbols[count].bytes);count++;offset=next;}
    fg_status status=FG_OK;for(;;){size_t best=SIZE_MAX;int32_t best_rank=INT32_MAX;for(size_t i=0;i+1u<count;i++){int32_t rank=merge_rank(tokenizer,&symbols[i],&symbols[i+1u]);if(rank==-2){status=FG_ERR_OOM;fg_error_set(err,status,"allocate BPE merge key");break;}if(rank>=0&&rank<best_rank){best=i;best_rank=rank;}}if(status!=FG_OK||best==SIZE_MAX)break;size_t merged_bytes=symbols[best].bytes+symbols[best+1u].bytes;char *merged=malloc(merged_bytes);if(!merged){status=FG_ERR_OOM;fg_error_set(err,status,"allocate merged BPE symbol");break;}memcpy(merged,symbols[best].text,symbols[best].bytes);memcpy(merged+symbols[best].bytes,symbols[best+1u].text,symbols[best+1u].bytes);free(symbols[best].text);free(symbols[best+1u].text);symbols[best]=(bpe_symbol){merged,merged_bytes};memmove(symbols+best+1u,symbols+best+2u,(count-best-2u)*sizeof(*symbols));count--;}
    for(size_t i=0;status==FG_OK&&i<count;i++){uint32_t token;status=fg_tokenizer_lookup(tokenizer,symbols[i].text,symbols[i].bytes,&token,err);if(status==FG_OK)status=tokens_push(tokens,token,err);}for(size_t i=0;i<count;i++)free(symbols[i].text);free(symbols);free(encoded);return status;
}

static bool special_at(const fg_tokenizer *tokenizer,const char *text,size_t bytes,uint32_t *token,size_t *length){size_t best=0;uint32_t best_token=0;for(uint32_t index=0;index<tokenizer->special_count;index++){uint32_t i=tokenizer->special_ids[index];if(tokenizer->tokens[i].bytes>best&&tokenizer->tokens[i].bytes<=bytes&&memcmp(tokenizer->tokens[i].text,text,tokenizer->tokens[i].bytes)==0){best=tokenizer->tokens[i].bytes;best_token=i;}}if(!best)return false;*token=best_token;*length=best;return true;}

fg_status fg_tokenizer_encode(const fg_tokenizer *tokenizer,const char *text,bool allow_special,fg_tokens *tokens,fg_error *err){if(!tokenizer||!text||!tokens){fg_error_set(err,FG_ERR_ARGUMENT,"invalid tokenizer encode arguments");return FG_ERR_ARGUMENT;}size_t bytes=strlen(text),offset=0;if(tokenizer->add_bos&&tokens->count==0){fg_status status=tokens_push(tokens,tokenizer->bos,err);if(status!=FG_OK)return status;}while(offset<bytes){if(allow_special){uint32_t special;size_t length;if(special_at(tokenizer,text+offset,bytes-offset,&special,&length)){fg_status status=tokens_push(tokens,special,err);if(status!=FG_OK)return status;offset+=length;continue;}}size_t end=next_piece(text,bytes,offset);if(end<=offset||end>bytes){fg_error_set(err,FG_ERR_FORMAT,"Qwen pre-tokenizer made no progress");return FG_ERR_FORMAT;}fg_status status=encode_piece(tokenizer,text+offset,end-offset,tokens,err);if(status!=FG_OK)return status;offset=end;}return FG_OK;}

static bool codepoint_byte(uint32_t code,uint8_t *byte){if((code>=33u&&code<=126u)||(code>=161u&&code<=172u)||(code>=174u&&code<=255u)){*byte=(uint8_t)code;return true;}uint32_t ordinal=0;for(uint32_t value=0;value<256u;value++){if((value>=33u&&value<=126u)||(value>=161u&&value<=172u)||value>=174u)continue;if(code==256u+ordinal){*byte=(uint8_t)value;return true;}ordinal++;}return false;}
fg_status fg_tokenizer_decode_token(const fg_tokenizer *tokenizer,uint32_t token,char *output,size_t capacity,size_t *bytes,fg_error *err){if(!tokenizer||token>=tokenizer->vocab_count||!output||!bytes){fg_error_set(err,FG_ERR_ARGUMENT,"invalid token decode arguments");return FG_ERR_ARGUMENT;}const token_slice *slice=&tokenizer->tokens[token];if(slice->type==3u||slice->type==4u){if(capacity<slice->bytes){fg_error_set(err,FG_ERR_LIMIT,"token decode buffer is too small");return FG_ERR_LIMIT;}memcpy(output,slice->text,slice->bytes);*bytes=slice->bytes;return FG_OK;}size_t used=0;for(size_t offset=0;offset<slice->bytes;){size_t next;uint32_t code=utf8_decode(slice->text,slice->bytes,offset,&next);uint8_t byte;if(!codepoint_byte(code,&byte)){fg_error_set(err,FG_ERR_FORMAT,"token %u has invalid byte-level codepoint",token);return FG_ERR_FORMAT;}if(used>=capacity){fg_error_set(err,FG_ERR_LIMIT,"token decode buffer is too small");return FG_ERR_LIMIT;}output[used++]=(char)byte;offset=next;}*bytes=used;return FG_OK;}

fg_status fg_tokenizer_validate_qwen38(const fg_tokenizer *tokenizer,fg_error *err){
    static const uint32_t hello_ids[]={14556};
    static const uint32_t hello_world_ids[]={9419,11,1814,0};
    static const uint32_t contractions_ids[]={4660,1357,2677,914,220,16,17,18,19};
    static const uint32_t unicode_ids[]={48,16451,220,109924,10838,248,222,198,5394,1500};
    static const uint32_t chat_ids[]={248045,846,198,12675,248046,198,248045,74455,198};
    static const struct{const char *text;bool special;const uint32_t *ids;size_t count;}vectors[]={
        {"hello",false,hello_ids,sizeof(hello_ids)/sizeof(hello_ids[0])},
        {"Hello, world!",false,hello_world_ids,sizeof(hello_world_ids)/sizeof(hello_world_ids[0])},
        {"can’t won't 1234",false,contractions_ids,sizeof(contractions_ids)/sizeof(contractions_ids[0])},
        {"Qwen 火箭 🚀\nsecond line",false,unicode_ids,sizeof(unicode_ids)/sizeof(unicode_ids[0])},
        {"<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",true,chat_ids,sizeof(chat_ids)/sizeof(chat_ids[0])}
    };
    if(!tokenizer||tokenizer->vocab_count!=FG_Q38_VOCAB_SIZE||tokenizer->bos!=248044u||tokenizer->eos!=248046u||tokenizer->add_bos){fg_error_set(err,FG_ERR_MISMATCH,"Qwen3.8 tokenizer header mismatch");return FG_ERR_MISMATCH;}
    for(size_t vector=0;vector<sizeof(vectors)/sizeof(vectors[0]);vector++){fg_tokens tokens={0};fg_status status=fg_tokenizer_encode(tokenizer,vectors[vector].text,vectors[vector].special,&tokens,err);if(status==FG_OK&&(tokens.count!=vectors[vector].count||memcmp(tokens.data,vectors[vector].ids,vectors[vector].count*sizeof(*tokens.data))!=0)){fg_error_set(err,FG_ERR_MISMATCH,"Qwen3.8 tokenizer parity vector %zu failed",vector);status=FG_ERR_MISMATCH;}fg_tokens_free(&tokens);if(status!=FG_OK)return status;}
    return FG_OK;
}
