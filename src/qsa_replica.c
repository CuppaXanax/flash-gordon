#include "fg_qsa_replica.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define FG_QSA_REPLICA_DEPTH 2u

typedef struct replica_slot {
    uint8_t *payload;
    fg_qsa_replica_item item;
} replica_slot;

struct fg_qsa_replica {
    fg_qsa_replica_send_fn send;
    void *context;
    replica_slot slots[FG_QSA_REPLICA_DEPTH];
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t drained;
    fg_status status;
    fg_error error;
    uint32_t head,count,reserved;
    bool stop,thread_started,mutex_ready,ready_ready,drained_ready;
};

static void *replica_main(void *opaque){
    fg_qsa_replica *replica=opaque;
    pthread_mutex_lock(&replica->mutex);
    while(!replica->stop||replica->count){
        while(!replica->count&&!replica->stop)
            pthread_cond_wait(&replica->ready,&replica->mutex);
        if(!replica->count)continue;
        uint32_t slot=replica->head;
        fg_qsa_replica_item item=replica->slots[slot].item;
        uint8_t *payload=replica->slots[slot].payload;
        pthread_mutex_unlock(&replica->mutex);
        fg_error error={0};
        fg_status status=replica->send(replica->context,item.owner,item.session_id,
                                      item.batch_id,payload,item.bytes,&error);
        pthread_mutex_lock(&replica->mutex);
        if(status!=FG_OK){
            replica->status=status;replica->error=error;replica->count=0;
        }else{
            replica->head=(replica->head+1u)%FG_QSA_REPLICA_DEPTH;
            replica->count--;
        }
        pthread_cond_broadcast(&replica->drained);
    }
    pthread_mutex_unlock(&replica->mutex);
    return NULL;
}

static void replica_cleanup(fg_qsa_replica *replica){
    if(!replica)return;
    if(replica->thread_started){
        pthread_mutex_lock(&replica->mutex);replica->stop=true;
        pthread_cond_signal(&replica->ready);pthread_mutex_unlock(&replica->mutex);
        pthread_join(replica->thread,NULL);
    }
    if(replica->drained_ready)pthread_cond_destroy(&replica->drained);
    if(replica->ready_ready)pthread_cond_destroy(&replica->ready);
    if(replica->mutex_ready)pthread_mutex_destroy(&replica->mutex);
    for(uint32_t i=0;i<FG_QSA_REPLICA_DEPTH;i++)free(replica->slots[i].payload);
    free(replica);
}

fg_status fg_qsa_replica_create(fg_qsa_replica **out,fg_qsa_replica_send_fn send,
                                void *context,fg_error *err){
    if(!out||!send){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA replica queue");
        return FG_ERR_ARGUMENT;
    }
    *out=NULL;fg_qsa_replica *replica=calloc(1,sizeof(*replica));
    if(!replica){fg_error_set(err,FG_ERR_OOM,"allocate QSA replica queue");return FG_ERR_OOM;}
    replica->send=send;replica->context=context;replica->status=FG_OK;
    for(uint32_t i=0;i<FG_QSA_REPLICA_DEPTH;i++){
        replica->slots[i].payload=malloc(FG_QSA_PAGE_APPEND_MAX_BYTES);
        if(!replica->slots[i].payload){
            fg_error_set(err,FG_ERR_OOM,"allocate bounded QSA replica slot");
            replica_cleanup(replica);return FG_ERR_OOM;
        }
    }
    if(pthread_mutex_init(&replica->mutex,NULL)!=0){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"initialize QSA replica mutex");
        replica_cleanup(replica);return FG_ERR_UNAVAILABLE;
    }
    replica->mutex_ready=true;
    if(pthread_cond_init(&replica->ready,NULL)!=0){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"initialize QSA replica condition");
        replica_cleanup(replica);return FG_ERR_UNAVAILABLE;
    }
    replica->ready_ready=true;
    if(pthread_cond_init(&replica->drained,NULL)!=0){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"initialize QSA replica drain condition");
        replica_cleanup(replica);return FG_ERR_UNAVAILABLE;
    }
    replica->drained_ready=true;
    if(pthread_create(&replica->thread,NULL,replica_main,replica)!=0){
        fg_error_set(err,FG_ERR_UNAVAILABLE,"start QSA replica sender");
        replica_cleanup(replica);return FG_ERR_UNAVAILABLE;
    }
    replica->thread_started=true;*out=replica;return FG_OK;
}

fg_status fg_qsa_replica_reserve(fg_qsa_replica *replica,uint32_t count,
                                 uint8_t *buffers[2],fg_error *err){
    if(!replica||!buffers||!count||count>FG_QSA_REPLICA_DEPTH){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA replica reservation");
        return FG_ERR_ARGUMENT;
    }
    pthread_mutex_lock(&replica->mutex);
    if(replica->status!=FG_OK){
        fg_status status=replica->status;if(err)*err=replica->error;
        pthread_mutex_unlock(&replica->mutex);return status;
    }
    if(replica->reserved||replica->count+count>FG_QSA_REPLICA_DEPTH){
        pthread_mutex_unlock(&replica->mutex);
        fg_error_set(err,FG_ERR_LIMIT,
                     "QSA replica queue is backpressured; session must reset");
        return FG_ERR_LIMIT;
    }
    replica->reserved=count;
    for(uint32_t i=0;i<count;i++)
        buffers[i]=replica->slots[(replica->head+replica->count+i)%
                                  FG_QSA_REPLICA_DEPTH].payload;
    pthread_mutex_unlock(&replica->mutex);return FG_OK;
}

fg_status fg_qsa_replica_commit(fg_qsa_replica *replica,
                                const fg_qsa_replica_item *items,uint32_t count,
                                fg_error *err){
    if(!replica||!items||!count||count>FG_QSA_REPLICA_DEPTH){
        fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA replica commit");
        return FG_ERR_ARGUMENT;
    }
    pthread_mutex_lock(&replica->mutex);
    if(replica->status!=FG_OK){
        fg_status status=replica->status;replica->reserved=0;
        if(err)*err=replica->error;
        pthread_cond_broadcast(&replica->drained);
        pthread_mutex_unlock(&replica->mutex);return status;
    }
    if(replica->reserved!=count){
        pthread_mutex_unlock(&replica->mutex);
        fg_error_set(err,FG_ERR_MISMATCH,"QSA replica reservation mismatch");
        return FG_ERR_MISMATCH;
    }
    for(uint32_t i=0;i<count;i++){
        if((items[i].owner!=3u&&items[i].owner!=7u)||!items[i].session_id||
           !items[i].bytes||items[i].bytes>FG_QSA_PAGE_APPEND_MAX_BYTES){
            replica->reserved=0;pthread_mutex_unlock(&replica->mutex);
            fg_error_set(err,FG_ERR_ARGUMENT,"invalid QSA replica item");
            return FG_ERR_ARGUMENT;
        }
        replica->slots[(replica->head+replica->count+i)%FG_QSA_REPLICA_DEPTH].item=
            items[i];
    }
    replica->count+=count;replica->reserved=0;
    pthread_cond_signal(&replica->ready);
    pthread_mutex_unlock(&replica->mutex);return FG_OK;
}

void fg_qsa_replica_cancel(fg_qsa_replica *replica){
    if(!replica)return;
    pthread_mutex_lock(&replica->mutex);replica->reserved=0;
    pthread_cond_broadcast(&replica->drained);
    pthread_mutex_unlock(&replica->mutex);
}

fg_status fg_qsa_replica_status(fg_qsa_replica *replica,fg_error *err){
    if(!replica){fg_error_set(err,FG_ERR_ARGUMENT,"QSA replica queue is null");return FG_ERR_ARGUMENT;}
    pthread_mutex_lock(&replica->mutex);fg_status status=replica->status;
    if(status!=FG_OK&&err)*err=replica->error;
    pthread_mutex_unlock(&replica->mutex);return status;
}

fg_status fg_qsa_replica_drain(fg_qsa_replica *replica,fg_error *err){
    if(!replica){fg_error_set(err,FG_ERR_ARGUMENT,"QSA replica queue is null");return FG_ERR_ARGUMENT;}
    pthread_mutex_lock(&replica->mutex);
    while((replica->count||replica->reserved)&&replica->status==FG_OK)
        pthread_cond_wait(&replica->drained,&replica->mutex);
    fg_status status=replica->status;if(status!=FG_OK&&err)*err=replica->error;
    pthread_mutex_unlock(&replica->mutex);return status;
}

static uint64_t replica_payload_bytes(void){
    return (uint64_t)FG_QSA_REPLICA_DEPTH*FG_QSA_PAGE_APPEND_MAX_BYTES;
}

static uint64_t replica_metadata_bytes(void){
    return sizeof(fg_qsa_replica);
}

uint64_t fg_qsa_replica_host_bytes(const fg_qsa_replica *replica){
    return replica?replica_metadata_bytes()+replica_payload_bytes():0u;
}

uint64_t fg_qsa_replica_host_bytes_for_capacity(void){
    return replica_metadata_bytes()+replica_payload_bytes();
}

void fg_qsa_replica_destroy(fg_qsa_replica *replica){replica_cleanup(replica);}
