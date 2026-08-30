#ifndef FLASH_GORDON_TOPOLOGY_H
#define FLASH_GORDON_TOPOLOGY_H

#include "fg_manifest.h"

void fg_topology_build(fg_manifest *manifest);
fg_status fg_topology_assign_round_robin(fg_manifest *manifest, fg_error *err);
fg_status fg_topology_assign_profile(fg_manifest *manifest, const double frequency[FG_LAYER_COUNT][FG_EXPERT_COUNT], fg_error *err);
fg_status fg_topology_assign_map(fg_manifest *manifest, const uint16_t expert_rank[FG_LAYER_COUNT][FG_EXPERT_COUNT], fg_error *err);
fg_status fg_topology_assign_map_file(fg_manifest *manifest, const char *path, fg_error *err);
bool fg_topology_rank_in_layer(const fg_manifest *manifest, uint32_t layer, uint32_t rank);

#endif
