/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// gd_hnsw_api.h — public distributed HNSW retrieval library API.
//
// Encapsulates the MPI + ubs-mem shared-memory + pushdown distributed search
// flow behind a single stateful `Context` object (no globals). Users call a
// handful of methods in each container process' main() to do "everything
// before search", synchronize N containers with MPI, then run `search`.
//
// See README_API.md for usage and REFACTOR_PLAN.md for design background.
//
// This header is intentionally lean: it only depends on the C++ standard
// library and exposes no internal types (SuperBlockV1, GdHnswSearcher,
// DualDistService, ...). All state is hidden behind a pimpl.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "status.h"
#include "build.h"

namespace gd_hnsw {

// ============================================================
// initialize() options
// ============================================================

struct InitOptions {
    int num_threads = 0; // >0 -> omp_set_num_threads; 0 = omp default
    bool auto_mpi_init = true; // false: caller already called MPI_Init
};

// ============================================================
// load() result
// ============================================================

struct LoadResult {
    Status status = Status::Ok;
    uint64_t ntotal = 0;
    uint32_t dim = 0;
    uint32_t M = 0;
    uint32_t ef_search = 0;
    uint32_t num_shards = 0;
    uint32_t shard_id = 0; // this node's shard id
    uint64_t ntotal_local = 0;
    uint64_t global_id_begin = 0;
    std::string error;
};

// ============================================================
// node_init_and_sync() options (deployment-time, fixes service topology)
// ============================================================

struct DeployOptions {
    uint32_t search_threads = 0;  // 0 = auto (num_threads - service_threads)
    uint32_t service_threads = 0; // 0 = auto (cores/4, clamped to num_shards-1)
    uint32_t expand_batch = 1;    // multi-pop expansion (1 = single candidate)
    bool use_dot_norm = false;    // dot+norm distance kernel
};

// ============================================================
// search() params (per-query, may vary across calls)
// ============================================================

struct SearchParams {
    int32_t k = 10;        // top-K results
    int32_t ef_search = 0; // 0 = use the ef_search stored in the index

    SearchParams() = default;
    explicit SearchParams(int32_t k_) : k(k_), ef_search(0) {}
};

struct SearchResult {
    Status status = Status::Ok;
    std::string error;
};

// ============================================================
// Context — all state lives here (pimpl, non-copyable, non-movable)
// ============================================================

class Context {
public:
    Context();
    ~Context();

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // (1) Initialize MPI + ubs-mem + NUMA provider + job_id.
    //     If opts.auto_mpi_init is true, argc/argv are forwarded to MPI_Init
    //     (pass nullptr if you have no argv). If you already called MPI_Init,
    //     set auto_mpi_init=false (argc/argv ignored).
    //     Collective: MPI_Init + MPI_Bcast(job_id).
    Status initialize(int *argc = nullptr, char ***argv = nullptr, const InitOptions &opts = {});

    // (3) Build the global sharded index and persist to disk.
    //     Single-process; does NOT call MPI collectives. May be called without
    //     initialize() in a pure build tool, but calling initialize() first is
    //     recommended (and required if you later use the same Context online).
    BuildResult build(const BuildOptions &opts);

    // (4) Load this node's shard file into this node's ubs-mem GD
    //     (file -> GD, graph compaction for multi-shard, mark ACTIVE,
    //      release writable mapping). Caches the shard file path, idmap and
    //     centroids into the Context for node_init_and_sync(). node_id == rank.
    //     Non-collective (local I/O).
    LoadResult load(uint32_t node_id, const std::string &shard_base_path);

    // (5) Node init + sync: create task/result channel GD -> MPI Barrier ->
    //     map all remote shard GD read-only -> map remote channels -> wire
    //     DualDistProxy -> load local vectors + EP vector -> start service
    //     threads -> build GdHnswSearcher. Cross-shard consistency check here.
    //     Collective: several Barrier / Allgather / Bcast on MPI_COMM_WORLD.
    //     Single-shard (num_shards==1) takes a fast path (vectors in GD, no
    //     channels/service).
    Status node_init_and_sync(uint32_t node_id, const DeployOptions &opts = {});

    // (6) Search this node's queries; returns full top-K (pushdown fans out
    //     remote distance computation to other nodes' service threads).
    //     Non-collective. out_ids[nq*k] / out_dists[nq*k] allocated by caller.
    //     Unfilled slots: id=-1, dist=FLT_MAX.
    SearchResult search(const float *queries, uint64_t nq, const SearchParams &params, int32_t *out_ids,
                        float *out_dists);

    // Helper: route queries to nodes. With centroids -> nearest-centroid
    // assignment, returns indices owned by this ctx's rank. Without centroids
    // -> uniform split. Static, non-collective. centroids/idmap must have been
    // loaded via load() first; falls back to uniform split otherwise.
    static std::vector<uint32_t> route_queries(const Context &ctx, const float *queries, uint64_t nq);

    // (2) Release: stop service -> unmap channels -> unmap shards ->
    //     deallocate GD -> ubsmem_finalize -> (if auto_mpi_init) MPI_Finalize.
    //     Collective: Barriers + ordered deallocation.
    Status finalize();

    // ---- accessors ----
    bool initialized() const;
    int rank() const; // MPI rank (== node_id == shard_id), -1 if not init
    int size() const; // MPI world size
    uint32_t num_shards() const;
    uint32_t dim() const;
    uint64_t ntotal() const;
    bool has_centroids() const; // centroids loaded (cluster_partition build)
    bool has_idmap() const;     // idmap loaded (cluster_partition build)
    // idmap (new gid -> original gid). Empty if none. Reference into Context
    // state — valid until finalize(). Useful to remap search result ids back
    // to dataset-original ids.
    const std::vector<uint32_t> &idmap() const;
    Status last_status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace gd_hnsw
