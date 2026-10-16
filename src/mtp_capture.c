#include "fg_mtp_capture.h"
#include "fg_protocol.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *capture_directory;

fg_status fg_mtp_capture_configure(const char *directory,fg_error *err){
    struct stat st;
    if(directory&&(!*directory||strlen(directory)>900u||stat(directory,&st)||
                   !S_ISDIR(st.st_mode))){
        fg_error_set(err,FG_ERR_ARGUMENT,"MTP capture requires an existing directory");
        return FG_ERR_ARGUMENT;
    }
    capture_directory=directory;
    return FG_OK;
}

static fg_status write_at(const char *path,uint64_t offset,const void *data,
                           uint64_t bytes,fg_error *err){
    int fd=open(path,O_WRONLY|O_CREAT|O_CLOEXEC,0600);
    uint64_t done=0u;
    if(fd<0)goto fail;
    while(done<bytes){
        ssize_t n=pwrite(fd,(const char *)data+done,(size_t)(bytes-done),
                         (off_t)(offset+done));
        if(n<0&&errno==EINTR)continue;
        if(n<=0){close(fd);goto fail;}
        done+=(uint64_t)n;
    }

    if(close(fd))goto fail;
    return FG_OK;
fail:
    fg_error_set(err,FG_ERR_IO,"MTP capture write %s: %s",path,strerror(errno));
    return FG_ERR_IO;
}

fg_status fg_mtp_capture_hyper(uint64_t session,uint32_t first,uint32_t count,
                              const float *hyper,fg_error *err){
    if(!capture_directory)return FG_OK;
    if(!session||!count||count>FG_PREFILL_MAX_TOKENS||!hyper||
       first>FG_NATIVE_CONTEXT-count){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid MTP capture range");
        return FG_ERR_ARGUMENT;
    }
    char path[1024];
    snprintf(path,sizeof(path),"%s/hc-%llu.f32",capture_directory,
             (unsigned long long)session);
    fg_status s=write_at(path,(uint64_t)first*FG_HYPER_WIDTH*4u,hyper,
        (uint64_t)count*FG_HYPER_WIDTH*4u,err);
    if(s!=FG_OK)return s;
    uint8_t present[FG_PREFILL_MAX_TOKENS];memset(present,1u,count);
    snprintf(path,sizeof(path),"%s/hc-%llu.coverage",capture_directory,
             (unsigned long long)session);
    return write_at(path,first,present,count,err);
}
