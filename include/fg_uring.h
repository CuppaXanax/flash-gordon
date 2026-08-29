#ifndef FLASH_GORDON_URING_H
#define FLASH_GORDON_URING_H

#include "fg.h"

typedef enum fg_ring_class {
    FG_RING_FABRIC = 1,
    FG_RING_STORAGE = 2
} fg_ring_class;

typedef struct fg_uring fg_uring;
typedef struct fg_uring_read {void *buffer;uint32_t bytes;uint64_t offset;} fg_uring_read;

fg_status fg_uring_create(fg_uring **out, fg_ring_class ring_class, uint32_t entries, fg_error *err);
void fg_uring_destroy(fg_uring *ring);
fg_status fg_uring_register_file(fg_uring *ring, int fd, uint32_t *slot, fg_error *err);
fg_status fg_uring_register_buffer(fg_uring *ring,void *aligned_buffer,uint64_t bytes,fg_error *err);
fg_status fg_uring_pread(fg_uring *ring, uint32_t file_slot, void *aligned_buffer, uint32_t bytes, uint64_t offset, fg_error *err);
fg_status fg_uring_pread_batch(fg_uring *ring,uint32_t file_slot,const fg_uring_read *reads,
                               uint32_t count,fg_error *err);
fg_status fg_uring_pwrite(fg_uring *ring, uint32_t file_slot, const void *aligned_buffer, uint32_t bytes, uint64_t offset, fg_error *err);
fg_status fg_uring_send_all(fg_uring *ring, uint32_t file_slot, const void *buffer, uint32_t bytes, fg_error *err);
fg_status fg_uring_recv_all(fg_uring *ring, uint32_t file_slot, void *buffer, uint32_t bytes, fg_error *err);

/* Async I/O: prep SQEs without submitting, flush to kernel, reap CQEs. */
typedef struct fg_uring_cqe {uint64_t tag;int32_t result;} fg_uring_cqe;
fg_status fg_uring_prep_recv(fg_uring *ring,uint32_t file_slot,void *buffer,uint32_t bytes,uint64_t tag,fg_error *err);
fg_status fg_uring_flush(fg_uring *ring,uint32_t pending_count,fg_error *err);
fg_status fg_uring_reap(fg_uring *ring,uint32_t min_count,fg_uring_cqe *out,uint32_t capacity,uint32_t *completed,fg_error *err);

#endif
