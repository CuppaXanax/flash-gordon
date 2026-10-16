#ifndef FLASH_GORDON_MTP_CAPTURE_H
#define FLASH_GORDON_MTP_CAPTURE_H
#include "fg.h"

/* Test-only CLI configuration; NULL disables capture. */
fg_status fg_mtp_capture_configure(const char *directory,fg_error *err);
fg_status fg_mtp_capture_hyper(uint64_t session,uint32_t first,uint32_t count,
                              const float *hyper,fg_error *err);
#endif
