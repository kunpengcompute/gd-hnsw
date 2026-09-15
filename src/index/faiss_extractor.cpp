/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include "faiss_extractor.h"
#include "index_io.h"

#include <faiss/Clustering.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

namespace gd_hnsw {

namespace {

std::vector<uint32_t> build_cluster_old_gid_of_new(const float *vectors, uint64_t n, uint32_t dim, uint32_t num_shards,
                                                   const ShardBuildOptions & /* opts */,
                                                   std::vector<uint64_t> &out_shard_sizes,
                                                   std::vector<float> *out_centroids)
{
    if (n == 0) {
        return {};
    }
    if (n > static_cast<uint64_t>(UINT32_MAX)) {
        throw std::runtime_error("cluster partition currently supports n <= UINT32_MAX");
    }

    // 1) Train k-means centroids.
    faiss::Clustering clus(static_cast<int>(dim), static_cast<int>(num_shards));
    clus.niter = 25;

    faiss::IndexFlatL2 train_index(static_cast<int>(dim));
    clus.train(static_cast<faiss::idx_t>(n), vectors, train_index);

    // Export centroids if requested.
    if (out_centroids) {
        *out_centroids = clus.centroids; // num_shards * dim floats
    }

    // 2) Assign each vector to its nearest centroid (natural cluster sizes).
    faiss::IndexFlatL2 centroid_index(static_cast<int>(dim));
    centroid_index.add(static_cast<faiss::idx_t>(num_shards), clus.centroids.data());

    std::vector<float> cand_dists(static_cast<size_t>(n));
    std::vector<faiss::idx_t> cand_ids(static_cast<size_t>(n));
    centroid_index.search(static_cast<faiss::idx_t>(n), vectors, 1, cand_dists.data(), cand_ids.data());

    std::vector<uint32_t> assignment(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; i++) {
        faiss::idx_t cid = cand_ids[i];
        if (cid < 0 || cid >= static_cast<faiss::idx_t>(num_shards)) {
            throw std::runtime_error("k-means assignment returned invalid centroid id");
        }
        assignment[i] = static_cast<uint32_t>(cid);
    }

    // 3) Build permutation new_gid -> old_gid, grouped by shard.
    std::vector<std::vector<uint32_t>> bucket(num_shards);
    for (uint32_t old_gid = 0; old_gid < static_cast<uint32_t>(n); old_gid++) {
        uint32_t s = assignment[old_gid];
        if (s >= num_shards) {
            throw std::runtime_error("invalid shard assignment generated");
        }
        bucket[s].push_back(old_gid);
    }

    // Log natural cluster sizes and output shard_sizes
    out_shard_sizes.resize(num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        out_shard_sizes[s] = bucket[s].size();
        if (bucket[s].empty()) {
            throw std::runtime_error("k-means produced empty cluster for shard " + std::to_string(s) +
                                     " — reduce num_shards or increase dataset size");
        }
        printf("[cluster] shard %u: %zu vectors (%.1f%%)\n", s, bucket[s].size(), 100.0 * bucket[s].size() / n);
    }

    std::vector<uint32_t> old_gid_of_new;
    old_gid_of_new.reserve(static_cast<size_t>(n));
    for (uint32_t s = 0; s < num_shards; s++) {
        old_gid_of_new.insert(old_gid_of_new.end(), bucket[s].begin(), bucket[s].end());
    }

    if (old_gid_of_new.size() != static_cast<size_t>(n)) {
        throw std::runtime_error("internal error: permutation size mismatch");
    }

    return old_gid_of_new;
}

} // namespace

// ============================================================
// Sharded extraction — split into per-NUMA buffers
// ============================================================

std::vector<std::vector<char>> FaissExtractor::extract_sharded(const faiss::IndexHNSWFlat &index, uint32_t num_shards,
                                                               uint32_t ef_search,
                                                               const std::vector<uint32_t> *old_gid_of_new,
                                                               const std::vector<uint64_t> *shard_sizes)
{

    const auto &hnsw = index.hnsw;
    const auto *storage = dynamic_cast<const faiss::IndexFlat *>(index.storage);
    if (!storage) {
        throw std::runtime_error("storage is not IndexFlat");
    }

    uint64_t ntotal = static_cast<uint64_t>(index.ntotal);
    uint32_t dim = static_cast<uint32_t>(index.d);

    if (ntotal == 0) {
        throw std::runtime_error("index is empty");
    }
    if (num_shards == 0 || num_shards > 32) {
        throw std::runtime_error("num_shards must be in [1, 32] (pushdown bitmask limit)");
    }
    if (ntotal > static_cast<uint64_t>(INT32_MAX)) {
        throw std::runtime_error("ntotal exceeds INT32_MAX");
    }

    std::vector<uint32_t> identity_map;
    if (!old_gid_of_new) {
        identity_map.resize(static_cast<size_t>(ntotal));
        std::iota(identity_map.begin(), identity_map.end(), 0);
        old_gid_of_new = &identity_map;
    }

    if (old_gid_of_new->size() != static_cast<size_t>(ntotal)) {
        throw std::runtime_error("old_gid_of_new size mismatch");
    }

    // Build inverse map: old_gid -> new_gid
    std::vector<uint32_t> new_gid_of_old(static_cast<size_t>(ntotal), UINT32_MAX);
    for (uint32_t new_gid = 0; new_gid < static_cast<uint32_t>(ntotal); new_gid++) {
        uint32_t old_gid = (*old_gid_of_new)[new_gid];
        if (old_gid >= static_cast<uint32_t>(ntotal)) {
            throw std::runtime_error("old_gid_of_new contains out-of-range id");
        }
        if (new_gid_of_old[old_gid] != UINT32_MAX) {
            throw std::runtime_error("old_gid_of_new is not a permutation (duplicate old id)");
        }
        new_gid_of_old[old_gid] = new_gid;
    }

    // Compute per-shard ranges in NEW id space.
    struct ShardRange {
        uint64_t global_begin;
        uint64_t count;
    };
    std::vector<ShardRange> ranges(num_shards);
    {
        uint64_t offset = 0;
        if (shard_sizes) {
            // Non-uniform: use provided sizes
            if (shard_sizes->size() != num_shards) {
                throw std::runtime_error("shard_sizes length mismatch");
            }
            uint64_t total_check = 0;
            for (uint32_t s = 0; s < num_shards; s++) {
                ranges[s].global_begin = offset;
                ranges[s].count = (*shard_sizes)[s];
                offset += ranges[s].count;
                total_check += (*shard_sizes)[s];
            }
            if (total_check != ntotal) {
                throw std::runtime_error("shard_sizes sum does not match ntotal");
            }
        } else {
            // Uniform split
            uint64_t base_per_shard = ntotal / num_shards;
            uint64_t remainder = ntotal % num_shards;
            for (uint32_t s = 0; s < num_shards; s++) {
                ranges[s].global_begin = offset;
                ranges[s].count = base_per_shard + (s < remainder ? 1 : 0);
                offset += ranges[s].count;
            }
        }
    }

    // Common fields
    uint32_t num_levels = static_cast<uint32_t>(hnsw.cum_nneighbor_per_level.size());
    if (num_levels < 2) {
        throw std::runtime_error("cum_nneighbor_per_level too small");
    }
    uint32_t M_val = static_cast<uint32_t>(hnsw.cum_nneighbor_per_level[1]) / 2;
    uint32_t ef = ef_search > 0 ? ef_search : static_cast<uint32_t>(hnsw.efSearch);

    const float *src_vectors = storage->get_xb();

    int32_t new_entry_point = -1;
    if (hnsw.entry_point >= 0) {
        uint32_t old_ep = static_cast<uint32_t>(hnsw.entry_point);
        if (old_ep >= static_cast<uint32_t>(ntotal)) {
            throw std::runtime_error("entry point out of range");
        }
        new_entry_point = static_cast<int32_t>(new_gid_of_old[old_ep]);
    }

    std::vector<std::vector<char>> result(num_shards);

    for (uint32_t s = 0; s < num_shards; s++) {
        uint64_t g_begin = ranges[s].global_begin;
        uint64_t n_local = ranges[s].count;

        // Compute local offsets and neighbors_size for this shard
        std::vector<uint64_t> local_offsets;
        local_offsets.reserve(static_cast<size_t>(n_local) + 1);
        local_offsets.push_back(0);
        for (uint64_t i = 0; i < n_local; i++) {
            uint64_t new_gid = g_begin + i;
            uint32_t old_gid = (*old_gid_of_new)[static_cast<size_t>(new_gid)];
            uint64_t slots = static_cast<uint64_t>(hnsw.offsets[old_gid + 1] - hnsw.offsets[old_gid]);
            local_offsets.push_back(local_offsets.back() + slots);
        }
        uint64_t local_neighbors_size = local_offsets.back();

        // Build SuperBlock for this shard
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::BUILDING);

#ifdef FAISS_VERSION_MAJOR
        sb.faiss_major = FAISS_VERSION_MAJOR;
        sb.faiss_minor = FAISS_VERSION_MINOR;
        sb.faiss_patch = FAISS_VERSION_PATCH;
#else
        sb.faiss_major = 1;
        sb.faiss_minor = 14;
        sb.faiss_patch = 1;
#endif

        sb.metric_type = METRIC_L2;
        sb.storage_idx_bits = 32;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = ntotal;
        sb.dim = dim;
        sb.M = M_val;
        sb.ef_search = ef;
        sb.max_level = static_cast<int32_t>(hnsw.max_level);
        sb.entry_point = new_entry_point;
        sb.num_levels = num_levels;

        // Shard fields
        sb.num_shards = num_shards;
        sb.shard_id = s;
        sb.ntotal_local = n_local;
        sb.global_id_begin = g_begin;

        sb.neighbors_size = local_neighbors_size;

        compute_layout(sb);

        sb.build_pid = static_cast<uint64_t>(getpid());
        auto now = std::chrono::system_clock::now();
        sb.build_unix_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());

        // Allocate buffer
        std::vector<char> buf(sb.total_bytes, 0);

        // Copy vectors for this shard
        auto *dst_vectors = reinterpret_cast<float *>(buf.data() + sb.blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < n_local; i++) {
            uint64_t new_gid = g_begin + i;
            uint32_t old_gid = (*old_gid_of_new)[static_cast<size_t>(new_gid)];
            std::memcpy(dst_vectors + i * dim, src_vectors + static_cast<uint64_t>(old_gid) * dim,
                        static_cast<size_t>(dim) * sizeof(float));
        }

        // Copy levels
        auto *dst_levels = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < n_local; i++) {
            uint64_t new_gid = g_begin + i;
            uint32_t old_gid = (*old_gid_of_new)[static_cast<size_t>(new_gid)];
            dst_levels[i] = static_cast<int32_t>(hnsw.levels[old_gid]);
        }

        // Copy local offsets
        std::memcpy(buf.data() + sb.blocks[BLK_OFFSETS].off, local_offsets.data(),
                    local_offsets.size() * sizeof(uint64_t));

        // Copy neighbors with old->new remap
        auto *dst_neighbors = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < n_local; i++) {
            uint64_t new_gid = g_begin + i;
            uint32_t old_gid = (*old_gid_of_new)[static_cast<size_t>(new_gid)];
            uint64_t src_begin = static_cast<uint64_t>(hnsw.offsets[old_gid]);
            uint64_t slots = static_cast<uint64_t>(hnsw.offsets[old_gid + 1]) - src_begin;
            uint64_t dst_begin = local_offsets[i];
            for (uint64_t j = 0; j < slots; j++) {
                auto val = hnsw.neighbors[src_begin + j];
                if (val < 0) {
                    dst_neighbors[dst_begin + j] = EMPTY_NEIGHBOR;
                } else {
                    uint32_t old_nid = static_cast<uint32_t>(val);
                    if (old_nid >= static_cast<uint32_t>(ntotal)) {
                        throw std::runtime_error("neighbor id out of range during remap");
                    }
                    uint32_t new_nid = new_gid_of_old[old_nid];
                    if (new_nid == UINT32_MAX) {
                        throw std::runtime_error("neighbor remap failed: missing old->new mapping");
                    }
                    dst_neighbors[dst_begin + j] = static_cast<int32_t>(new_nid);
                }
            }
        }

        // Copy cum_nneighbor_per_level (same for all shards)
        auto *dst_cum = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_CUM_NNEIGHBOR].off);
        for (uint32_t i = 0; i < num_levels; i++) {
            dst_cum[i] = static_cast<int32_t>(hnsw.cum_nneighbor_per_level[i]);
        }

        // Finalize
        sb.state = static_cast<uint32_t>(GdState::SEALED);
        sb.payload_crc64 = compute_payload_crc64(buf.data(), sb);
        sb.header_crc64 = compute_header_crc64(sb);
        std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));

        result[s] = std::move(buf);
    }

    return result;
}

std::vector<std::vector<char>> FaissExtractor::build_and_extract_sharded(const float *vectors, uint64_t n, uint32_t dim,
                                                                         uint32_t num_shards, uint32_t M,
                                                                         uint32_t ef_construction, uint32_t ef_search,
                                                                         const ShardBuildOptions &opts,
                                                                         std::vector<uint32_t> *new_to_old_idmap,
                                                                         std::vector<float> *out_centroids)
{

    // Use unique_ptr so we can explicitly destroy the index before returning
    // the shard buffers.  The FAISS index holds ~3x the vector data size
    // (vectors + int64 neighbors + offsets), so early destruction reclaims
    // ~145 GiB for 100M×128D×M64 before the shard buffers (~72 GiB) are
    // returned to the caller.
    auto index = std::make_unique<faiss::IndexHNSWFlat>(static_cast<int>(dim), static_cast<int>(M));
    index->hnsw.efConstruction = static_cast<int>(ef_construction);
    index->hnsw.efSearch = static_cast<int>(ef_search);

    index->add(static_cast<faiss::idx_t>(n), vectors);

    // After index.add(), FAISS holds its own copy of the vectors inside
    // IndexFlat storage.  From this point we use the internal copy so the
    // caller can free the original dataset.train immediately.
    if (opts.post_add_callback) {
        opts.post_add_callback();
    }

    const auto *storage = dynamic_cast<const faiss::IndexFlat *>(index->storage);
    if (!storage) {
        throw std::runtime_error("storage is not IndexFlat after add");
    }
    const float *indexed_vectors = storage->get_xb();

    std::vector<uint32_t> cluster_old_gid_of_new;
    const std::vector<uint32_t> *old_gid_of_new_ptr = nullptr;
    std::vector<uint64_t> cluster_shard_sizes;
    const std::vector<uint64_t> *shard_sizes_ptr = nullptr;

    if (opts.cluster_partition) {
        cluster_old_gid_of_new =
            build_cluster_old_gid_of_new(indexed_vectors, n, dim, num_shards, opts, cluster_shard_sizes, out_centroids);
        old_gid_of_new_ptr = &cluster_old_gid_of_new;
        shard_sizes_ptr = &cluster_shard_sizes;
    }

    if (new_to_old_idmap) {
        if (old_gid_of_new_ptr) {
            *new_to_old_idmap = *old_gid_of_new_ptr;
        } else {
            new_to_old_idmap->resize(static_cast<size_t>(n));
            std::iota(new_to_old_idmap->begin(), new_to_old_idmap->end(), 0);
        }
    }

    // Diagnostic: compute layer-0 edge cut ratio after clustering.
    if (opts.cluster_partition && old_gid_of_new_ptr) {
        const auto &hnsw = index->hnsw;
        // Build new_gid_of_old for shard lookup
        std::vector<uint32_t> diag_new_of_old(static_cast<size_t>(n));
        for (uint32_t ng = 0; ng < static_cast<uint32_t>(n); ng++) {
            diag_new_of_old[(*old_gid_of_new_ptr)[ng]] = ng;
        }
        // Shard ranges from actual cluster sizes
        std::vector<uint64_t> shard_begin(num_shards);
        {
            uint64_t off = 0;
            for (uint32_t s = 0; s < num_shards; s++) {
                shard_begin[s] = off;
                off += cluster_shard_sizes[s];
            }
        }
        auto shard_of = [&](uint32_t new_gid) -> uint32_t {
            // Binary search on shard_begin
            uint32_t lo = 0, hi = num_shards;
            while (lo + 1 < hi) {
                uint32_t mid = lo + (hi - lo) / 2;
                if (shard_begin[mid] <= new_gid)
                    lo = mid;
                else
                    hi = mid;
            }
            return lo;
        };

        uint64_t total_edges = 0;
        uint64_t cross_edges = 0;
        // Only count layer-0 edges (first cum_nneighbor[1] slots per node)
        int layer0_slots =
            hnsw.cum_nneighbor_per_level.size() >= 2 ? static_cast<int>(hnsw.cum_nneighbor_per_level[1]) : 0;
        for (uint64_t old_gid = 0; old_gid < n; old_gid++) {
            uint32_t new_gid = diag_new_of_old[static_cast<size_t>(old_gid)];
            uint32_t my_shard = shard_of(new_gid);
            uint64_t off_begin = static_cast<uint64_t>(hnsw.offsets[old_gid]);
            int slots = std::min(layer0_slots, static_cast<int>(hnsw.offsets[old_gid + 1] - hnsw.offsets[old_gid]));
            for (int j = 0; j < slots; j++) {
                auto nid = hnsw.neighbors[off_begin + j];
                if (nid < 0)
                    continue;
                uint32_t nb_new = diag_new_of_old[static_cast<size_t>(nid)];
                uint32_t nb_shard = shard_of(nb_new);
                total_edges++;
                if (nb_shard != my_shard)
                    cross_edges++;
            }
        }
        double cut_ratio = (total_edges > 0) ? (100.0 * cross_edges / total_edges) : 0.0;
        printf("[cluster-diag] layer-0 edge cut: %lu / %lu (%.1f%% cross-shard)\n", (unsigned long)cross_edges,
               (unsigned long)total_edges, cut_ratio);
    }

    // Extract shard buffers while index is still alive, then destroy the
    // index immediately to free ~145 GiB before returning.
    auto result = extract_sharded(*index, num_shards, ef_search, old_gid_of_new_ptr, shard_sizes_ptr);
    index.reset(); // free FAISS index memory NOW
    printf("[build] FAISS index released before returning shard buffers\n");

    return result;
}

// ---------------------------------------------------------------------------
// Streaming variant: extract one shard at a time, write to disk, free buffer.
// Peak memory = FAISS index + 1 shard buffer (instead of all shard buffers).
// ---------------------------------------------------------------------------

static std::string streaming_shard_path(const std::string &base_path, uint32_t shard_id)
{
    return base_path + ".shard_" + std::to_string(shard_id);
}

std::vector<uint64_t> FaissExtractor::build_and_extract_sharded_streaming(
    const float *vectors, uint64_t n, uint32_t dim, uint32_t num_shards, const std::string &save_path, uint32_t M,
    uint32_t ef_construction, uint32_t ef_search, const ShardBuildOptions &opts,
    std::vector<uint32_t> *new_to_old_idmap, std::vector<float> *out_centroids)
{

    auto index = std::make_unique<faiss::IndexHNSWFlat>(static_cast<int>(dim), static_cast<int>(M));
    index->hnsw.efConstruction = static_cast<int>(ef_construction);
    index->hnsw.efSearch = static_cast<int>(ef_search);

    index->add(static_cast<faiss::idx_t>(n), vectors);

    if (opts.post_add_callback) {
        opts.post_add_callback();
    }

    const auto *storage = dynamic_cast<const faiss::IndexFlat *>(index->storage);
    if (!storage) {
        throw std::runtime_error("storage is not IndexFlat after add");
    }

    std::vector<uint32_t> cluster_old_gid_of_new;
    const std::vector<uint32_t> *old_gid_of_new_ptr = nullptr;
    std::vector<uint64_t> cluster_shard_sizes;
    const std::vector<uint64_t> *shard_sizes_ptr = nullptr;

    if (opts.cluster_partition) {
        cluster_old_gid_of_new = build_cluster_old_gid_of_new(storage->get_xb(), n, dim, num_shards, opts,
                                                              cluster_shard_sizes, out_centroids);
        old_gid_of_new_ptr = &cluster_old_gid_of_new;
        shard_sizes_ptr = &cluster_shard_sizes;
    }

    if (new_to_old_idmap) {
        if (old_gid_of_new_ptr) {
            *new_to_old_idmap = *old_gid_of_new_ptr;
        } else {
            new_to_old_idmap->resize(static_cast<size_t>(n));
            std::iota(new_to_old_idmap->begin(), new_to_old_idmap->end(), 0);
        }
    }

    // Extract shards one at a time, write to disk, free buffer immediately.
    // This reuses extract_sharded internally but processes one shard at a time.
    // We call the single-shard extraction inline here.
    const auto &hnsw = index->hnsw;
    const float *src_vectors = storage->get_xb();
    uint64_t ntotal = static_cast<uint64_t>(index->ntotal);

    // --- Input boundary checks (same as extract_sharded) ---
    if (ntotal == 0) {
        throw std::runtime_error("index is empty");
    }
    if (num_shards == 0 || num_shards > 32) {
        throw std::runtime_error("num_shards must be in [1, 32] (pushdown bitmask limit)");
    }
    if (ntotal > static_cast<uint64_t>(INT32_MAX)) {
        throw std::runtime_error("ntotal exceeds INT32_MAX");
    }

    uint32_t M_val =
        static_cast<uint32_t>(hnsw.cum_nneighbor_per_level.size() >= 2 ? hnsw.cum_nneighbor_per_level[1] / 2 : 0);
    uint32_t num_levels = static_cast<uint32_t>(hnsw.cum_nneighbor_per_level.size());

    if (num_levels < 2) {
        throw std::runtime_error("HNSW has fewer than 2 levels (need at least layer 0)");
    }

    // Build identity or cluster mapping
    std::vector<uint32_t> identity_map;
    const std::vector<uint32_t> *mapping = old_gid_of_new_ptr;
    if (!mapping) {
        identity_map.resize(static_cast<size_t>(ntotal));
        std::iota(identity_map.begin(), identity_map.end(), 0);
        mapping = &identity_map;
    }

    if (mapping->size() != static_cast<size_t>(ntotal)) {
        throw std::runtime_error("old_gid_of_new size mismatch");
    }

    // Inverse map with permutation validation
    std::vector<uint32_t> new_gid_of_old(static_cast<size_t>(ntotal), UINT32_MAX);
    for (uint32_t new_gid = 0; new_gid < static_cast<uint32_t>(ntotal); new_gid++) {
        uint32_t old_gid = (*mapping)[new_gid];
        if (old_gid >= static_cast<uint32_t>(ntotal)) {
            throw std::runtime_error("old_gid_of_new contains out-of-range id");
        }
        if (new_gid_of_old[old_gid] != UINT32_MAX) {
            throw std::runtime_error("old_gid_of_new is not a permutation (duplicate old id)");
        }
        new_gid_of_old[old_gid] = new_gid;
    }

    // Compute shard ranges
    struct ShardRange {
        uint64_t global_begin;
        uint64_t count;
    };
    std::vector<ShardRange> ranges(num_shards);
    {
        uint64_t offset = 0;
        for (uint32_t s = 0; s < num_shards; s++) {
            uint64_t count =
                shard_sizes_ptr ? (*shard_sizes_ptr)[s] : (ntotal / num_shards + (s < ntotal % num_shards ? 1 : 0));
            ranges[s] = {offset, count};
            offset += count;
        }
    }

    int32_t new_entry_point = -1;
    if (hnsw.entry_point >= 0) {
        uint32_t old_ep = static_cast<uint32_t>(hnsw.entry_point);
        if (old_ep >= static_cast<uint32_t>(ntotal)) {
            throw std::runtime_error("entry point out of range");
        }
        new_entry_point = static_cast<int32_t>(new_gid_of_old[old_ep]);
    }

    std::vector<uint64_t> shard_byte_sizes(num_shards);

    for (uint32_t s = 0; s < num_shards; s++) {
        uint64_t g_begin = ranges[s].global_begin;
        uint64_t n_local = ranges[s].count;

        // Compute local offsets
        std::vector<uint64_t> local_offsets;
        local_offsets.reserve(static_cast<size_t>(n_local) + 1);
        local_offsets.push_back(0);
        for (uint64_t i = 0; i < n_local; i++) {
            uint32_t old_gid = (*mapping)[static_cast<size_t>(g_begin + i)];
            uint64_t slots = static_cast<uint64_t>(hnsw.offsets[old_gid + 1] - hnsw.offsets[old_gid]);
            local_offsets.push_back(local_offsets.back() + slots);
        }
        uint64_t local_neighbors_size = local_offsets.back();

        // Build SuperBlock
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::BUILDING);
#ifdef FAISS_VERSION_MAJOR
        sb.faiss_major = FAISS_VERSION_MAJOR;
        sb.faiss_minor = FAISS_VERSION_MINOR;
        sb.faiss_patch = FAISS_VERSION_PATCH;
#else
        sb.faiss_major = 1;
        sb.faiss_minor = 14;
        sb.faiss_patch = 1;
#endif
        sb.metric_type = METRIC_L2;
        sb.storage_idx_bits = 32;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = ntotal;
        sb.dim = dim;
        sb.M = M_val;
        sb.ef_search = ef_search;
        sb.max_level = static_cast<int32_t>(hnsw.max_level);
        sb.entry_point = new_entry_point;
        sb.num_levels = num_levels;
        sb.num_shards = num_shards;
        sb.shard_id = s;
        sb.ntotal_local = n_local;
        sb.global_id_begin = g_begin;
        sb.neighbors_size = local_neighbors_size;
        compute_layout(sb);
        sb.build_pid = static_cast<uint64_t>(getpid());
        auto now = std::chrono::system_clock::now();
        sb.build_unix_ns =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());

        // Allocate buffer for this shard
        std::vector<char> buf(sb.total_bytes, 0);

        // Copy vectors
        auto *dst_vectors = reinterpret_cast<float *>(buf.data() + sb.blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < n_local; i++) {
            uint32_t old_gid = (*mapping)[static_cast<size_t>(g_begin + i)];
            std::memcpy(dst_vectors + i * dim, src_vectors + static_cast<uint64_t>(old_gid) * dim,
                        static_cast<size_t>(dim) * sizeof(float));
        }

        // Copy levels
        auto *dst_levels = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < n_local; i++) {
            uint32_t old_gid = (*mapping)[static_cast<size_t>(g_begin + i)];
            dst_levels[i] = static_cast<int32_t>(hnsw.levels[old_gid]);
        }

        // Copy offsets
        std::memcpy(buf.data() + sb.blocks[BLK_OFFSETS].off, local_offsets.data(),
                    local_offsets.size() * sizeof(uint64_t));

        // Copy neighbors with remap
        auto *dst_neighbors = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < n_local; i++) {
            uint32_t old_gid = (*mapping)[static_cast<size_t>(g_begin + i)];
            uint64_t src_begin = static_cast<uint64_t>(hnsw.offsets[old_gid]);
            uint64_t slots = static_cast<uint64_t>(hnsw.offsets[old_gid + 1]) - src_begin;
            uint64_t dst_begin = local_offsets[i];
            for (uint64_t j = 0; j < slots; j++) {
                auto val = hnsw.neighbors[src_begin + j];
                if (val < 0) {
                    dst_neighbors[dst_begin + j] = EMPTY_NEIGHBOR;
                } else {
                    uint32_t old_nid = static_cast<uint32_t>(val);
                    if (old_nid >= static_cast<uint32_t>(ntotal)) {
                        throw std::runtime_error("neighbor id out of range during remap");
                    }
                    uint32_t new_nid = new_gid_of_old[old_nid];
                    if (new_nid == UINT32_MAX) {
                        throw std::runtime_error("neighbor remap failed: missing old->new mapping");
                    }
                    dst_neighbors[dst_begin + j] = static_cast<int32_t>(new_nid);
                }
            }
        }

        // Copy cum_nneighbor_per_level
        auto *dst_cum = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_CUM_NNEIGHBOR].off);
        for (uint32_t i = 0; i < num_levels; i++) {
            dst_cum[i] = static_cast<int32_t>(hnsw.cum_nneighbor_per_level[i]);
        }

        // Finalize
        sb.state = static_cast<uint32_t>(GdState::SEALED);
        sb.payload_crc64 = compute_payload_crc64(buf.data(), sb);
        sb.header_crc64 = compute_header_crc64(sb);
        std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));

        // Write to disk immediately
        std::string path = streaming_shard_path(save_path, s);
        std::string dir_err;
        if (!ensure_parent_dir(path, dir_err)) {
            throw std::runtime_error(std::string("cannot create parent directory for '") + path + "': " + dir_err);
        }
        FILE *fp = fopen(path.c_str(), "wb");
        if (!fp) {
            throw std::runtime_error(std::string("cannot open '") + path + "' for writing: " + strerror(errno));
        }
        size_t written = fwrite(buf.data(), 1, buf.size(), fp);
        fclose(fp);
        if (written != buf.size()) {
            throw std::runtime_error(std::string("short write to '") + path + "'");
        }
        shard_byte_sizes[s] = buf.size();
        printf("  [streaming] Saved shard %u: %s (%.1f MB)\n", s, path.c_str(), buf.size() / (1024.0 * 1024.0));
        // buf is freed here at end of loop iteration
    }

    index.reset();
    printf("[build] FAISS index released (streaming mode)\n");

    return shard_byte_sizes;
}

} // namespace gd_hnsw
