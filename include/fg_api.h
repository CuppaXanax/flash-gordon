#ifndef FLASH_GORDON_API_H
#define FLASH_GORDON_API_H

#include "fg_runtime.h"

fg_status fg_api_main(const char *manifest_path, const char *host, uint16_t port,
                      fg_error *err);
fg_status fg_api_main_with_options(const char *manifest_path, const char *host,
                                   uint16_t port,
                                   const fg_runtime_options *runtime_options,
                                   fg_error *err);

#endif
