/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include "gd_layout.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Forward declare faiss types to avoid header dependency in this header
namespace faiss {
struct IndexHNSWFlat;
}

namespace gd_hnsw {

// Build-time shard partition policy.
struct ShardBuildOptions {
    bool cluster_partition = false; // false = range split by global id (legacy)

    // Called after index->add() completes (FAISS holds its own vector copy).
    // Use this to release the original input vectors and reclaim memory before
    // the expensive extract_sharded phase begins.
    std::function<void()> post_add_callback;
};

// ============================================================
// FaissExtractor — convert faiss IndexHNSWFlat to flat gd buffer
// ============================================================

class FaissExtractor {
public:
    // Extract from a faiss index into per-shard buffers for NUMA distribution.
    // Returns num_shards buffers, each containing vectors/graph for its id range.
    // Neighbor IDs remain global.
    // shard_sizes: if provided, specifies per-shard vector count (non-uniform).
    //              If nullptr, shards are split uniformly.
    static std::vector<std::vector<char>> extract_sharded(const faiss::IndexHNSWFlat &index, uint32_t num_shards,
                                                          uint32_t ef_search = 0,
                                                          const std::vector<uint32_t> *old_gid_of_new = nullptr,
                                                          const std::vector<uint64_t> *shard_sizes = nullptr);

    // Build then extract sharded.
    // out_centroids: if non-null, receives k-means centroids (num_shards * dim floats).
    static std::vector<std::vector<char>>
    build_and_extract_sharded(const float *vectors, uint64_t n, uint32_t dim, uint32_t num_shards, uint32_t M = 16,
                              uint32_t ef_construction = 40, uint32_t ef_search = 16,
                              const ShardBuildOptions &opts = {}, std::vector<uint32_t> *new_to_old_idmap = nullptr,
                              std::vector<float> *out_centroids = nullptr);

    // Build then extract sharded with streaming write — each shard is written
    // to disk immediately after extraction and its buffer freed, so at most one
    // shard buffer is alive at a time.  Returns the per-shard byte sizes.
    // Caller must load shards back from disk for MPI distribution.
    static std::vector<uint64_t> build_and_extract_sharded_streaming(
        const float *vectors, uint64_t n, uint32_t dim, uint32_t num_shards, const std::string &save_path,
        uint32_t M = 16, uint32_t ef_construction = 40, uint32_t ef_search = 16, const ShardBuildOptions &opts = {},
        std::vector<uint32_t> *new_to_old_idmap = nullptr, std::vector<float> *out_centroids = nullptr);
};

} // namespace gd_hnsw
