#ifndef FLASH_GORDON_MTP_EMBEDDING_H
#define FLASH_GORDON_MTP_EMBEDDING_H

#include "fg_manifest.h"

typedef struct fg_mtp_embedding fg_mtp_embedding;

/* A raw production-Q8 sidecar, bound to the sealed source tensor digest and
 * independently checksummed. Read a row on demand; never pin another full
 * vocabulary-sized embedding on the final owner. */
fg_status fg_mtp_embedding_export(const fg_manifest *manifest,const char *pack_dir,
    const char *output,fg_error *err);
fg_status fg_mtp_embedding_open(fg_mtp_embedding **out,const fg_manifest *manifest,
    const char *path,fg_error *err);
fg_status fg_mtp_embedding_row(fg_mtp_embedding *embedding,uint32_t token,
    float *values,fg_error *err);
void fg_mtp_embedding_close(fg_mtp_embedding *embedding);

#endif
