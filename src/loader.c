#include "fg_loader.h"
#include "fg_sha256.h"
#include "fg_uring.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

fg_status fg_verify_rank_arena(const fg_manifest *m,uint32_t rank,const void *arena,uint64_t arena_bytes,fg_error *err){
    if(!m||rank>=FG_RANK_COUNT||!arena){fg_error_set(err,FG_ERR_ARGUMENT,"invalid rank arena");return FG_ERR_ARGUMENT;}const uint8_t *base=arena;
    for(uint32_t i=0;i<m->tensor_count;i++){const fg_tensor_record *t=&m->tensors[i];if(t->rank!=rank||t->kind==FG_TENSOR_HOST_CACHE)continue;if(!fg_is_aligned_u64(t->offset,FG_ALIGNMENT)||t->offset>arena_bytes||t->bytes>arena_bytes-t->offset){fg_error_set(err,FG_ERR_FORMAT,"rank %u tensor %.96s is outside or misaligned in arena",rank,t->name);return FG_ERR_FORMAT;}fg_sha256 c;uint8_t digest[32];fg_sha256_init(&c);fg_sha256_update(&c,base+t->offset,(size_t)t->bytes);fg_sha256_final(&c,digest);if(memcmp(digest,t->sha256,32)!=0){fg_error_set(err,FG_ERR_MISMATCH,"rank %u tensor %.96s SHA-256 mismatch",rank,t->name);return FG_ERR_MISMATCH;}}
    return FG_OK;
}

fg_status fg_load_rank_weights(const fg_manifest *m,const char *dir,uint32_t rank,void *arena,uint64_t arena_bytes,fg_error *err){
    if(!m||!dir||rank>=FG_RANK_COUNT||!arena||!fg_is_aligned_u64((uintptr_t)arena,FG_ALIGNMENT)){fg_error_set(err,FG_ERR_ARGUMENT,"final Vulkan arena must be non-null and 4 KiB aligned");return FG_ERR_ARGUMENT;}char path[1024];if(snprintf(path,sizeof(path),"%s/rank-%02u.fgw",dir,rank)>=(int)sizeof(path)){fg_error_set(err,FG_ERR_ARGUMENT,"pack directory path is too long");return FG_ERR_ARGUMENT;}int fd=open(path,O_RDONLY|O_DIRECT|O_CLOEXEC);if(fd<0){fg_error_set(err,FG_ERR_IO,"open direct %s: %s",path,strerror(errno));return FG_ERR_IO;}struct stat st;if(fstat(fd,&st)!=0||st.st_size<0||!fg_is_aligned_u64((uint64_t)st.st_size,FG_ALIGNMENT)||(uint64_t)st.st_size>arena_bytes){fg_error_set(err,FG_ERR_FORMAT,"rank artifact size is invalid or exceeds final arena");close(fd);return FG_ERR_FORMAT;}
    const uint32_t chunk=8u*1024u*1024u;fg_uring *ring=NULL;fg_status rc=fg_uring_create(&ring,FG_RING_STORAGE,256,err);uint32_t slot=0;if(rc==FG_OK)rc=fg_uring_register_file(ring,fd,&slot,err);
    fg_status buf_rc=rc==FG_OK?fg_uring_register_buffer(ring,arena,(uint64_t)st.st_size,err):rc;
    if(buf_rc==FG_OK){for(uint64_t off=0;rc==FG_OK&&off<(uint64_t)st.st_size;off+=chunk){uint32_t n=(uint32_t)(((uint64_t)st.st_size-off)>chunk?chunk:(uint64_t)st.st_size-off);rc=fg_uring_pread(ring,slot,(uint8_t *)arena+off,n,off,err);}}
    else if(rc==FG_OK){
        fprintf(stderr,"[rank %u] Vulkan arena not pinnable for io_uring; using bounce-buffer pread\n",rank);
        void *bounce=aligned_alloc(FG_ALIGNMENT,chunk);
        if(!bounce){fg_error_set(err,FG_ERR_OOM,"allocate loader bounce buffer");rc=FG_ERR_OOM;}
        for(uint64_t off=0;rc==FG_OK&&off<(uint64_t)st.st_size;off+=chunk){
            uint32_t n=(uint32_t)(((uint64_t)st.st_size-off)>chunk?chunk:(uint64_t)st.st_size-off);
            ssize_t got=pread(fd,bounce,n,(off_t)off);
            if(got!=(ssize_t)n){fg_error_set(err,FG_ERR_IO,"pread rank %u at offset %llu: %s",rank,(unsigned long long)off,got<0?strerror(errno):"short read");rc=FG_ERR_IO;}
            else memcpy((uint8_t *)arena+off,bounce,n);
        }
        free(bounce);
    }
    if(rc==FG_OK)
        rc=fg_verify_rank_arena(m,rank,arena,(uint64_t)st.st_size,err);
    fg_uring_destroy(ring);close(fd);return rc;
}
