#ifndef FLASH_GORDON_MTP_CAPTURE_H
#define FLASH_GORDON_MTP_CAPTURE_H
#include "fg.h"

/* Test-only CLI configuration; NULL disables capture. */
fg_status fg_mtp_capture_configure(const char *directory,fg_error *err);
bool fg_mtp_capture_enabled(void);
fg_status fg_mtp_capture_routes(uint64_t session,uint32_t layer,uint32_t first,
    uint32_t count,const uint32_t *experts,fg_error *err);
fg_status fg_mtp_capture_verify_routes(uint64_t session,uint32_t layer,uint32_t first,
    uint32_t count,bool replay,const uint32_t *experts,fg_error *err);
fg_status fg_mtp_capture_hyper(uint64_t session,uint32_t first,uint32_t count,
                              const float *hyper,fg_error *err);
static inline double fg_mtp_capture_deadline_ms(double now_ms,uint32_t budget_seconds){
    return now_ms+1000.0*(double)budget_seconds;
}
#endif
