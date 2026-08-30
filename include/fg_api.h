#ifndef FLASH_GORDON_API_H
#define FLASH_GORDON_API_H

#include "fg.h"

fg_status fg_api_main(const char *manifest_path, const char *host, uint16_t port,
                      fg_error *err);

#endif
