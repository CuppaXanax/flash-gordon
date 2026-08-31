#ifndef FLASH_GORDON_QSA_REPLICA_H
#define FLASH_GORDON_QSA_REPLICA_H

#include "fg_protocol.h"

typedef struct fg_qsa_replica fg_qsa_replica;
typedef fg_status (*fg_qsa_replica_send_fn)(
    void *context,uint32_t owner,uint64_t session_id,uint32_t batch_id,
    const void *payload,uint32_t bytes,fg_error *err);

typedef struct fg_qsa_replica_item {
    uint32_t owner;
    uint32_t batch_id;
    uint32_t bytes;
    uint64_t session_id;
} fg_qsa_replica_item;

fg_status fg_qsa_replica_create(fg_qsa_replica **out,fg_qsa_replica_send_fn send,
                                void *context,fg_error *err);
fg_status fg_qsa_replica_reserve(fg_qsa_replica *replica,uint32_t count,
                                 uint8_t *buffers[2],fg_error *err);
fg_status fg_qsa_replica_commit(fg_qsa_replica *replica,
                                const fg_qsa_replica_item *items,uint32_t count,
                                fg_error *err);
void fg_qsa_replica_cancel(fg_qsa_replica *replica);
fg_status fg_qsa_replica_status(fg_qsa_replica *replica,fg_error *err);
fg_status fg_qsa_replica_drain(fg_qsa_replica *replica,fg_error *err);
/* A lazy page transport has no queue to drain until its first page service. */
static inline fg_status fg_qsa_replica_drain_if_present(
    fg_qsa_replica *replica,fg_error *err){
    return replica?fg_qsa_replica_drain(replica,err):FG_OK;
}
uint64_t fg_qsa_replica_host_bytes(const fg_qsa_replica *replica);
uint64_t fg_qsa_replica_host_bytes_for_capacity(void);
void fg_qsa_replica_destroy(fg_qsa_replica *replica);

#endif
