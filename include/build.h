/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// build.h — offline sharded-index build API (no MPI / ubs-mem / NUMA).
//
// Follows the faiss index_io.h pattern: a free function
// `build_sharded_index` that builds a global HNSW index and persists it to
// disk as N shard files. `Context::build` (gd_hnsw_api.h) is a thin delegate
// to this function, so build tools only need to link the build layer
// (gd_hnsw_build) rather than the full deploy layer (gd_hnsw_api).
//
// This header only depends on status.h and the standard library; it never
// includes gd_hnsw_api.h.

#pragma once

#include "status.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gd_hnsw {

// ============================================================
// build() options / result
// ============================================================

struct BuildOptions {
    std::string dataset_path; // HDF5 file or .bin directory/file
    std::string output_path;  // writes output_path.shard_0 .. shard_<N-1>
    uint32_t num_shards = 1;
    uint32_t M = 16;
    uint32_t ef_construction = 200;
    uint32_t ef_search = 64;
    uint32_t build_threads = 0;          // 0 = all cores
    bool cluster_partition = false;      // k-means shard partition (+ idmap/centroids)
    std::string dataset_format = "auto"; // auto|hdf5|bin
    std::string bin_base_path;           // .bin mode: override base vectors file
};

struct BuildResult {
    Status status = Status::Ok;
    uint64_t ntotal = 0;
    uint32_t dim = 0;
    uint32_t num_shards = 0;
    std::vector<uint64_t> shard_bytes; // per-shard file size
    std::string error;
};

// Build a sharded HNSW index from `opts.dataset_path` and persist it as
// `opts.output_path.shard_0 .. shard_<num_shards-1>` (plus idmap/centroids
// sidecars when opts.cluster_partition is set). Single-process; no MPI
// collectives, no ubs-mem, no NUMA. See README_API.md §4.
BuildResult build_sharded_index(const BuildOptions &opts);

} // namespace gd_hnsw
