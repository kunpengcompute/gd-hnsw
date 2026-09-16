/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include "gd_layout.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gd_hnsw {

// ============================================================
// Index file path helpers
// ============================================================

std::string index_file_for_shard(const std::string &base_path, uint32_t shard_id);
std::string idmap_file_for_index(const std::string &base_path);
std::string centroids_file_for_index(const std::string &base_path);

// Recursively create the parent directory of `path` (idempotent: success if
// it already exists; success if `path` has no directory component).
// On failure returns false and fills `err_out` with a human-readable reason.
bool ensure_parent_dir(const std::string &path, std::string &err_out);

// ============================================================
// IdMap file I/O
// ============================================================

struct IdMapFileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t reserved;
    uint64_t count;
};

static constexpr uint64_t IDMAP_MAGIC = 0x484E535749444D31ULL; // "HNSWIDM1"
static constexpr uint32_t IDMAP_VERSION = 1;

enum class IdMapLoadStatus {
    Loaded,
    NotFound,
    Error,
};

bool save_idmap_file(const std::string &base_path, const std::vector<uint32_t> &idmap);

IdMapLoadStatus load_idmap_file(const std::string &base_path, std::vector<uint32_t> &idmap_out);

// ============================================================
// Centroids file I/O
// ============================================================

struct CentroidsFileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t num_shards;
    uint32_t reserved;
};

static constexpr uint64_t CENTROIDS_MAGIC = 0x484E5357434E5431ULL; // "HNSWCNT1"
static constexpr uint32_t CENTROIDS_VERSION = 1;

bool save_centroids_file(const std::string &base_path, const std::vector<float> &centroids, uint32_t num_shards,
                         uint32_t dim);

IdMapLoadStatus load_centroids_file(const std::string &base_path, std::vector<float> &centroids_out,
                                    uint32_t &num_shards_out, uint32_t &dim_out);

// ============================================================
// Shard buffer I/O
// ============================================================

bool save_shard_buffers(const std::string &base_path, const std::vector<std::vector<char>> &shard_bufs);

bool load_shard_buffers(const std::string &base_path, uint32_t num_shards, std::vector<std::vector<char>> &shard_bufs);

} // namespace gd_hnsw
