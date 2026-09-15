/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// gd_hnsw_api.cpp — implementation of the gd_hnsw::Context API.
//
// Ports the deployment logic from the former monolithic MPI benchmark
// (src/main.cpp) into a stateful pimpl. The benchmark-only concerns
// (query/gt broadcast, warmup, timed rounds, QPS/latency/recall stats,
// profiling) are dropped; everything needed to "deploy then search" lives
// in initialize / load / node_init_and_sync / search / finalize.

#include "gd_hnsw_api.h"

#include "gd_layout.h"
#include "gd_hnsw_search.h"
#include "dist_service.h"
#include "index_io.h"

#include <mpi.h>
#include <sys/mman.h> // PROT_*, MAP_*, madvise
#include <unistd.h>   // getpid, gethostname, access

#ifdef __linux__
#include <sched.h> // sched_getcpu
#include <numa.h>  // numa_node_of_cpu
#endif

#include <ubs_mem_def.h>
#include <ubs_mem.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <omp.h>

#ifndef __linux__
#error "gd_hnsw::Context requires Linux (NUMA-local ubs-mem allocation)."
#endif

namespace gd_hnsw {

// ============================================================
// File-local helpers (pure — no Context state needed)
// ============================================================

namespace {

// ubs-mem minimum allocation granularity.
constexpr size_t UBSMEM_ALIGN = 4UL * 1024 * 1024;

// pushdown expand_batch upper bound, and num_shards cap (pending_mask is a
// uint32_t bitmask).
constexpr uint32_t kMaxExpandBatch = 64;
constexpr uint32_t kMaxShardsMask32 = 32;

inline size_t ubsmem_align_size(size_t sz)
{
    if (sz == 0)
        return UBSMEM_ALIGN;
    return (sz + UBSMEM_ALIGN - 1) & ~(UBSMEM_ALIGN - 1);
}

// Deallocate with retry on UBSM_ERR_IN_USING (6024).
inline int ubsmem_deallocate_retry(const char *name, int max_retries = 3)
{
    for (int i = 0; i <= max_retries; i++) {
        int ret = ubsmem_shmem_deallocate(name);
        if (ret == UBSM_OK || ret == UBSM_ERR_NOT_FOUND)
            return UBSM_OK;
        if (ret == UBSM_ERR_IN_USING && i < max_retries) {
            usleep(100000 * (1 << i)); // 100ms, 200ms, 400ms
            continue;
        }
        return ret;
    }
    return UBSM_ERR_IN_USING;
}

// Graph-only compaction: copy graph blocks into GD, skip BLK_VECTORS.
// Assumes SuperBlock header is already at dst. Updates dst SuperBlock
// in-place (blocks, total_bytes). Does NOT set state or recompute CRCs.
void compact_graph_to_gd(void *dst, const char *src_data, const SuperBlockV1 &src_sb)
{
    auto *gd_sb = static_cast<SuperBlockV1 *>(dst);
    char *d = static_cast<char *>(dst);
    uint64_t offset = SUPERBLOCK_SIZE;

    gd_sb->blocks[BLK_VECTORS].off = 0;
    gd_sb->blocks[BLK_VECTORS].bytes = 0;

    static constexpr int graph_blocks[] = {BLK_LEVELS, BLK_OFFSETS, BLK_NEIGHBORS, BLK_CUM_NNEIGHBOR};
    for (int blk : graph_blocks) {
        uint64_t new_off = align_up(offset, gd_sb->blocks[blk].alignment);
        std::memcpy(d + new_off, src_data + src_sb.blocks[blk].off, src_sb.blocks[blk].bytes);
        gd_sb->blocks[blk].off = new_off;
        offset = new_off + gd_sb->blocks[blk].bytes;
    }

    gd_sb->total_bytes = align_up(offset, PAGE_ALIGN);
}

// Map an existing ubs-mem gd region read-only.
// Aborts on failure (called inside collective setup where partial failure
// cannot be unwound per-rank).
void *open_gd_readonly(const std::string &name, uint64_t total_bytes)
{
    size_t alloc_size = ubsmem_align_size(static_cast<size_t>(total_bytes));
    void *ptr = nullptr;
    int ret = ubsmem_shmem_map(nullptr, alloc_size, PROT_READ, MAP_SHARED, name.c_str(), 0, &ptr);
    if (ret != UBSM_OK || ptr == nullptr) {
        fprintf(stderr, "ubsmem_shmem_map readonly '%s' failed: %d\n", name.c_str(), ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

// Per-shard / channel naming (no leading '/' — ubs-mem constraint).
inline std::string gd_name_for_shard(uint32_t job_id, uint32_t shard_id)
{
    return "gd_hnsw_" + std::to_string(job_id) + "_" + std::to_string(shard_id);
}
inline std::string task_inbox_name_for_rank(uint32_t job_id, uint32_t rank)
{
    return "gd_hnsw_task_" + std::to_string(job_id) + "_" + std::to_string(rank);
}
inline std::string result_inbox_name_for_rank(uint32_t job_id, uint32_t rank)
{
    return "gd_hnsw_result_" + std::to_string(job_id) + "_" + std::to_string(rank);
}

// Map shard_id to dense [0, num_remote-1] index, skipping my_shard.
inline uint32_t compressed_remote_idx(uint32_t shard, uint32_t my_shard)
{
    return (shard < my_shard) ? shard : shard - 1;
}

// Metadata produced by loading one shard file into GD.
struct LocalShardMeta {
    uint64_t ntotal = 0;
    uint32_t dim = 0;
    uint32_t M = 0;
    uint32_t ef_search = 0;
    int32_t max_level = 0;
    int32_t entry_point = 0;
    uint32_t num_shards = 0;
    uint32_t shard_id = 0;
    uint64_t ntotal_local = 0;
    uint64_t global_id_begin = 0;
    uint64_t file_size = 0;
    uint64_t gd_alloc_size = 0; // actual graph-only GD allocation size
    bool ok = false;
    Status status = Status::Ok; // failure category, set at each error return
    std::string error;
};

// POD mirror of LocalShardMeta for MPI_Allgather (cross-shard validation).
struct ShardMetaWire {
    uint64_t ntotal;
    uint32_t dim;
    uint32_t M;
    int32_t max_level;
    int32_t entry_point;
    uint32_t num_shards;
    uint32_t shard_id;
    uint64_t ntotal_local;
    uint64_t global_id_begin;
};
static_assert(std::is_trivially_copyable_v<ShardMetaWire>,
              "ShardMetaWire must be trivially copyable for MPI_BYTE transfer");

} // namespace

// ============================================================
// Context::Impl — all state and logic
// ============================================================

struct Context::Impl {
    // --- init state ---
    bool initialized_ = false;
    bool mpi_ready_ = false; // MPI is initialized (by us or user)
    bool ubsmem_ready_ = false;
    bool auto_mpi_init_ = true;
    int rank_ = -1;
    int size_ = 1;
    uint32_t job_id_ = 0;
    int num_threads_ = 0; // omp budget (0 = default)
    uint64_t ubsmem_flags_ = 0;
    uint64_t channel_flags_ = 0;
    int log_level_ = 4;
    ubs_mem_provider_t provider_{};
    Status last_status_ = Status::Ok;

    // --- load state ---
    std::string shard_base_path_; // cached for node_init_and_sync (local vectors + EP)
    LocalShardMeta load_meta_{};
    std::vector<uint64_t> shard_sizes_; // [num_shards], Allgather'd in node_init_and_sync
    std::vector<uint32_t> idmap_;
    bool has_idmap_ = false;
    std::vector<float> centroids_; // num_shards * dim
    bool has_centroids_ = false;

    // --- node_init_and_sync state ---
    std::unique_ptr<GdHnswSearcher> searcher_;
    // Mapped read-only shard pointers (own + remote). Kept for unmap in finalize.
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs_;
    // Pushdown state (multi-shard only)
    std::vector<std::unique_ptr<DualDistService>> dual_dist_services_;
    std::vector<std::vector<DualDistProxy>> dual_pushdown_proxies_; // [thread][shard]
    std::vector<std::string> owned_channel_gd_names_;
    std::vector<std::pair<void *, size_t>> requester_channel_maps_;
    std::vector<std::pair<void *, size_t>> service_channel_maps_;
    uint64_t run_gen_ = 0;
    uint32_t srch_threads_ = 0;
    uint32_t svc_threads_ = 0;
    uint32_t num_shards_ = 0; // from shard header (== size_)
    uint32_t dim_ = 0;
    uint64_t ntotal_ = 0;
    bool use_dot_norm_ = false;
    uint32_t expand_batch_ = 1;
    bool pushdown_active_ = false;

    // ---- methods ----
    Impl() = default;
    ~Impl() { /* cleanup is explicit via finalize(); avoid silent MPI use */ }

    Status initialize(int *argc, char ***argv, const InitOptions &opts);
    BuildResult build(const BuildOptions &opts);
    LoadResult load(uint32_t node_id, const std::string &shard_base_path);
    Status node_init_and_sync(uint32_t node_id, const DeployOptions &opts);
    SearchResult search(const float *queries, uint64_t nq, const SearchParams &params, int32_t *out_ids,
                        float *out_dists);
    Status finalize();

    // Query routing helper (centroid or uniform). Returns indices owned by
    // this rank. Accessed by the static Context::route_queries.
    std::vector<uint32_t> route_queries(const float *queries, uint64_t nq) const;

    // Load this node's shard file into ubs-mem GD (file -> GD, compaction,
    // ACTIVE, release writable mapping). Fills meta. Non-collective.
    LocalShardMeta load_local_shard_direct_to_gd(const std::string &base_path, uint32_t shard_id,
                                                 const std::string &gd_name, bool keep_vectors);
};

// ============================================================
// initialize
// ============================================================

Status Context::Impl::initialize(int *argc, char ***argv, const InitOptions &opts)
{
    if (initialized_) {
        last_status_ = Status::AlreadyInit;
        return last_status_;
    }

    num_threads_ = opts.num_threads;
    auto_mpi_init_ = opts.auto_mpi_init;

    // Default flags: half-noncache (only import noncache + write-delay-comp).
    ubsmem_flags_ = (UBSM_FLAG_ONLY_IMPORT_NONCACHE | UBSM_FLAG_WR_DELAY_COMP);
    channel_flags_ = (UBSM_FLAG_ONLY_IMPORT_NONCACHE | UBSM_FLAG_WR_DELAY_COMP);

    // --- MPI ---
    if (auto_mpi_init_) {
        int mret = MPI_Init(argc, argv);
        if (mret != MPI_SUCCESS) {
            last_status_ = Status::MpiError;
            return last_status_;
        }
    }
    int mpi_init_flag = 0;
    MPI_Initialized(&mpi_init_flag);
    mpi_ready_ = (mpi_init_flag != 0);

    if (mpi_ready_) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        // rank 0 generates job_id (pid) and broadcasts.
        uint32_t jid = 0;
        if (rank_ == 0)
            jid = static_cast<uint32_t>(getpid());
        MPI_Bcast(&jid, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        job_id_ = jid;
    } else {
        // No MPI (e.g. a pure build tool that never called MPI_Init).
        rank_ = -1;
        size_ = 1;
        job_id_ = static_cast<uint32_t>(getpid());
    }

    // --- OpenMP ---
    omp_set_dynamic(0);
    if (num_threads_ > 0)
        omp_set_num_threads(num_threads_);

    // --- ubs-mem ---
    {
        ubsmem_options_t ubopts{};
        int ret = ubsmem_init_attributes(&ubopts);
        if (ret != UBSM_OK) {
            fprintf(stderr, "[gd-hnsw] ubsmem_init_attributes failed: %d\n", ret);
            last_status_ = Status::UbsemError;
            return last_status_;
        }
        ubsmem_set_logger_level(log_level_);
        ret = ubsmem_initialize(&ubopts);
        if (ret != UBSM_OK) {
            fprintf(stderr, "[gd-hnsw] ubsmem_initialize failed: %d\n", ret);
            last_status_ = Status::UbsemError;
            return last_status_;
        }
        ubsmem_ready_ = true;
    }

    // --- NUMA-local provider ---
    {
        std::memset(&provider_, 0, sizeof(provider_));
        if (gethostname(provider_.host_name, sizeof(provider_.host_name) - 1) != 0) {
            fprintf(stderr, "[gd-hnsw] gethostname failed: %s\n", strerror(errno));
            last_status_ = Status::UbsemError;
            return last_status_;
        }
        provider_.socket_id = UINT32_MAX; // auto-detect
        provider_.port_id = UINT32_MAX;   // auto-detect
        int cpu = sched_getcpu();
        if (cpu < 0) {
            fprintf(stderr, "[gd-hnsw] sched_getcpu failed: %s\n", strerror(errno));
            last_status_ = Status::UbsemError;
            return last_status_;
        }
        int numa = numa_node_of_cpu(cpu);
        if (numa < 0) {
            fprintf(stderr, "[gd-hnsw] numa_node_of_cpu(%d) failed\n", cpu);
            last_status_ = Status::UbsemError;
            return last_status_;
        }
        provider_.numa_id = static_cast<uint32_t>(numa);
        if (rank_ <= 0) {
            printf("[gd-hnsw] provider: host=%s, numa_id=%u, ubsmem_flags=0x%lx, channel_flags=0x%lx\n",
                   provider_.host_name, provider_.numa_id, (unsigned long)ubsmem_flags_, (unsigned long)channel_flags_);
        }
    }

    initialized_ = true;
    last_status_ = Status::Ok;
    return last_status_;
}

// ============================================================
// build (single-process, no MPI collectives) — delegates to the
// build layer's free function build_sharded_index().
// ============================================================

BuildResult Context::Impl::build(const BuildOptions &opts) { return build_sharded_index(opts); }

// ============================================================
// load (file -> GD, non-collective)
// ============================================================

LocalShardMeta Context::Impl::load_local_shard_direct_to_gd(const std::string &base_path, uint32_t shard_id,
                                                            const std::string &gd_name, bool keep_vectors)
{
    LocalShardMeta meta{};

    std::string path = index_file_for_shard(base_path, shard_id);
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) {
        meta.status = Status::IoError;
        meta.error = std::string("cannot open '") + path + "': " + strerror(errno);
        return meta;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (file_size <= 0) {
        fclose(fp);
        meta.status = Status::IoError;
        meta.error = std::string("'") + path + "' is empty or unreadable";
        return meta;
    }

    SuperBlockV1 file_sb{};
    if (fread(&file_sb, sizeof(SuperBlockV1), 1, fp) != 1) {
        fclose(fp);
        meta.status = Status::IoError;
        meta.error = "cannot read SuperBlock from file";
        return meta;
    }

    // Read entire file into a temp buffer for CRC validation.
    fseek(fp, 0, SEEK_SET);
    std::vector<char> file_buf(static_cast<size_t>(file_size));
    size_t nread = fread(file_buf.data(), 1, static_cast<size_t>(file_size), fp);
    fclose(fp);
    if (nread != static_cast<size_t>(file_size)) {
        meta.status = Status::IoError;
        meta.error = "short read: " + std::to_string(nread) + " / " + std::to_string(file_size);
        return meta;
    }
    auto vr = validate_superblock(file_buf.data(), static_cast<uint64_t>(file_size));
    if (!vr.ok) {
        meta.status = Status::ValidationErr;
        meta.error = "validation failed: " + vr.error;
        return meta;
    }

    size_t gd_size;
    if (keep_vectors) {
        gd_size = static_cast<size_t>(file_sb.total_bytes);
    } else {
        gd_size = static_cast<size_t>(file_sb.total_bytes - file_sb.blocks[BLK_VECTORS].bytes);
        if (gd_size < SUPERBLOCK_SIZE)
            gd_size = file_sb.total_bytes;
    }

    ubsmem_deallocate_retry(gd_name.c_str());
    size_t alloc_size = ubsmem_align_size(gd_size);
    int ret = ubsmem_shmem_allocate_with_provider(&provider_, gd_name.c_str(), alloc_size, 0600, ubsmem_flags_);
    if (ret != UBSM_OK) {
        meta.status = Status::UbsemError;
        meta.error = "ubsmem_shmem_allocate_with_provider failed: " + std::to_string(ret);
        return meta;
    }
    void *ptr = nullptr;
    ret = ubsmem_shmem_map(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, gd_name.c_str(), 0, &ptr);
    if (ret != UBSM_OK || ptr == nullptr) {
        ubsmem_deallocate_retry(gd_name.c_str());
        meta.status = Status::UbsemError;
        meta.error = "ubsmem_shmem_map failed: " + std::to_string(ret);
        return meta;
    }

    std::memcpy(ptr, file_buf.data(), SUPERBLOCK_SIZE);
    auto *gd_sb = static_cast<SuperBlockV1 *>(ptr);

    if (keep_vectors) {
        std::memcpy(static_cast<char *>(ptr) + SUPERBLOCK_SIZE, file_buf.data() + SUPERBLOCK_SIZE,
                    static_cast<size_t>(file_sb.total_bytes) - SUPERBLOCK_SIZE);
        gd_sb->state = static_cast<uint32_t>(GdState::ACTIVE);
        gd_sb->payload_crc64 = compute_payload_crc64(ptr, *gd_sb);
        gd_sb->header_crc64 = compute_header_crc64(*gd_sb);
    } else {
        compact_graph_to_gd(ptr, file_buf.data(), file_sb);
        gd_sb->state = static_cast<uint32_t>(GdState::ACTIVE);
        gd_sb->payload_crc64 = compute_payload_crc64(ptr, *gd_sb);
        gd_sb->header_crc64 = compute_header_crc64(*gd_sb);
    }

    // file_buf is unused once the copy into GD completes; release it early to
    // narrow the window where file_buf and GD coexist.
    std::vector<char>().swap(file_buf);

    madvise(static_cast<char *>(ptr) + gd_sb->blocks[BLK_NEIGHBORS].off, gd_sb->blocks[BLK_NEIGHBORS].bytes,
            MADV_HUGEPAGE);

    meta.ok = true;
    meta.ntotal = gd_sb->ntotal;
    meta.dim = gd_sb->dim;
    meta.M = gd_sb->M;
    meta.ef_search = gd_sb->ef_search;
    meta.max_level = gd_sb->max_level;
    meta.entry_point = gd_sb->entry_point;
    meta.num_shards = gd_sb->num_shards;
    meta.shard_id = gd_sb->shard_id;
    meta.ntotal_local = gd_sb->ntotal_local;
    meta.global_id_begin = gd_sb->global_id_begin;
    meta.file_size = static_cast<uint64_t>(file_size);
    meta.gd_alloc_size = alloc_size;

    // Release the writable mapping — gd object persists; all ranks re-open
    // read-only in node_init_and_sync().
    ubsmem_shmem_unmap(ptr, alloc_size);
    return meta;
}

LoadResult Context::Impl::load(uint32_t node_id, const std::string &shard_base_path)
{
    LoadResult r;
    if (!ubsmem_ready_) {
        r.status = Status::NotInitialized;
        r.error = "initialize() not called";
        return r;
    }
    if (shard_base_path.empty()) {
        r.status = Status::InvalidArg;
        r.error = "shard_base_path empty";
        return r;
    }
    if (mpi_ready_ && static_cast<int>(node_id) != rank_) {
        r.status = Status::InvalidArg;
        r.error = "node_id (" + std::to_string(node_id) + ") != MPI rank (" + std::to_string(rank_) + ")";
        return r;
    }

    shard_base_path_ = shard_base_path;

    // Peek shard header to learn num_shards (decides keep_vectors) before
    // creating GD. (load_local_shard_direct_to_gd re-reads the file.)
    std::string peek_path = index_file_for_shard(shard_base_path, node_id);
    SuperBlockV1 peek_sb{};
    {
        FILE *fp = fopen(peek_path.c_str(), "rb");
        if (!fp) {
            r.status = Status::IoError;
            r.error = "cannot open '" + peek_path + "'";
            return r;
        }
        if (fread(&peek_sb, sizeof(peek_sb), 1, fp) != 1) {
            fclose(fp);
            r.status = Status::IoError;
            r.error = "cannot read header '" + peek_path + "'";
            return r;
        }
        fclose(fp);
    }
    if (peek_sb.magic != GD_MAGIC) {
        r.status = Status::ValidationErr;
        r.error = "bad magic in '" + peek_path + "'";
        return r;
    }

    bool keep_vectors = (peek_sb.num_shards == 1);
    std::string gd_name = gd_name_for_shard(job_id_, node_id);
    LocalShardMeta meta = load_local_shard_direct_to_gd(shard_base_path, node_id, gd_name, keep_vectors);
    if (!meta.ok) {
        r.status = meta.status;
        r.error = meta.error;
        return r;
    }
    load_meta_ = meta;

    num_shards_ = meta.num_shards;
    dim_ = meta.dim;
    ntotal_ = meta.ntotal;

    // Load idmap / centroids sidecars (shared files; each node reads its own).
    {
        std::vector<uint32_t> idmap;
        IdMapLoadStatus st = load_idmap_file(shard_base_path, idmap);
        if (st == IdMapLoadStatus::Loaded) {
            if (idmap.size() != static_cast<size_t>(meta.ntotal)) {
                r.status = Status::ValidationErr;
                r.error =
                    "idmap size (" + std::to_string(idmap.size()) + ") != ntotal (" + std::to_string(meta.ntotal) + ")";
                return r;
            }
            idmap_ = std::move(idmap);
            has_idmap_ = true;
        } else if (st == IdMapLoadStatus::Error) {
            r.status = Status::IoError;
            r.error = "idmap file read error";
            return r;
        } else {
            has_idmap_ = false;
        }
    }
    {
        std::vector<float> cents;
        uint32_t c_shards = 0, c_dim = 0;
        IdMapLoadStatus st = load_centroids_file(shard_base_path, cents, c_shards, c_dim);
        if (st == IdMapLoadStatus::Loaded) {
            if (c_shards != num_shards_ || c_dim != dim_) {
                r.status = Status::ValidationErr;
                r.error = "centroids mismatch (shards " + std::to_string(c_shards) +
                          "!=" + std::to_string(num_shards_) + " or dim " + std::to_string(c_dim) +
                          "!=" + std::to_string(dim_) + ")";
                return r;
            }
            centroids_ = std::move(cents);
            has_centroids_ = true;
        } else if (st == IdMapLoadStatus::Error) {
            r.status = Status::IoError;
            r.error = "centroids file read error";
            return r;
        } else {
            has_centroids_ = false;
        }
    }

    if (rank_ <= 0) {
        printf("[load] rank=%u shard=%u %s (%.1f MB, ntotal_local=%lu)\n", node_id, meta.shard_id,
               keep_vectors ? "full" : "graph-only", meta.file_size / (1024.0 * 1024.0),
               (unsigned long)meta.ntotal_local);
    }

    r.status = Status::Ok;
    r.ntotal = meta.ntotal;
    r.dim = meta.dim;
    r.M = meta.M;
    r.ef_search = meta.ef_search;
    r.num_shards = meta.num_shards;
    r.shard_id = meta.shard_id;
    r.ntotal_local = meta.ntotal_local;
    r.global_id_begin = meta.global_id_begin;
    return r;
}

// ============================================================
// node_init_and_sync (collective)
// ============================================================

Status Context::Impl::node_init_and_sync(uint32_t node_id, const DeployOptions &opts)
{
    if (!ubsmem_ready_) {
        last_status_ = Status::NotInitialized;
        return last_status_;
    }
    if (!mpi_ready_) {
        last_status_ = Status::NotInitialized;
        return last_status_;
    }
    if (load_meta_.num_shards == 0) {
        last_status_ = Status::NotInitialized;
        return last_status_;
    }
    // These checks can be data-dependent (shard files may differ across ranks),
    // so collect a local code and Allreduce it — every rank must proceed or
    // return together, or a partial return would deadlock the MPI_Allgather below.
    int local_err = 0;
    if (static_cast<int>(node_id) != rank_) {
        fprintf(stderr, "[gd-hnsw] node_init_and_sync: node_id (%u) != rank (%d)\n", node_id, rank_);
        local_err = 1;
    }
    if (num_shards_ != static_cast<uint32_t>(size_)) {
        fprintf(stderr, "[gd-hnsw] num_shards (%u) must equal MPI world size (%d)\n", num_shards_, size_);
        local_err = 1;
    }
    if (opts.expand_batch < 1 || opts.expand_batch > kMaxExpandBatch) {
        fprintf(stderr, "[gd-hnsw] expand_batch must be in [1, %u], got %u\n", kMaxExpandBatch, opts.expand_batch);
        local_err = 1;
    }
    int global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_err != 0) {
        last_status_ = Status::InvalidArg;
        return last_status_;
    }

    use_dot_norm_ = opts.use_dot_norm;
    expand_batch_ = opts.expand_batch;
    const uint32_t my_shard = node_id;

    // --- Allgather graph-only GD sizes so all ranks can open remote shards ---
    shard_sizes_.assign(num_shards_, 0);
    uint64_t my_size = load_meta_.gd_alloc_size;
    MPI_Allgather(&my_size, 1, MPI_UINT64_T, shard_sizes_.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    // --- Cross-shard consistency validation (ShardMetaWire Allgather) ---
    {
        ShardMetaWire my_wire{};
        my_wire.ntotal = load_meta_.ntotal;
        my_wire.dim = load_meta_.dim;
        my_wire.M = load_meta_.M;
        my_wire.max_level = load_meta_.max_level;
        my_wire.entry_point = load_meta_.entry_point;
        my_wire.num_shards = load_meta_.num_shards;
        my_wire.shard_id = load_meta_.shard_id;
        my_wire.ntotal_local = load_meta_.ntotal_local;
        my_wire.global_id_begin = load_meta_.global_id_begin;

        std::vector<ShardMetaWire> all_meta(num_shards_);
        MPI_Allgather(&my_wire, sizeof(ShardMetaWire), MPI_BYTE, all_meta.data(), sizeof(ShardMetaWire), MPI_BYTE,
                      MPI_COMM_WORLD);

        const auto &s0 = all_meta[0];
        uint64_t sum_ntotal_local = 0;
        bool consistent = true;
        for (uint32_t s = 0; s < num_shards_; s++) {
            const auto &sm = all_meta[s];
            if (sm.shard_id != s) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u has shard_id=%u\n", rank_, s, sm.shard_id);
                consistent = false;
            }
            if (sm.num_shards != num_shards_) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u has num_shards=%u (expected %u)\n", rank_, s,
                        sm.num_shards, num_shards_);
                consistent = false;
            }
            if (sm.ntotal != s0.ntotal || sm.dim != s0.dim || sm.M != s0.M || sm.max_level != s0.max_level ||
                sm.entry_point != s0.entry_point) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u param mismatch with shard 0\n", rank_, s);
                consistent = false;
            }
            if (sm.global_id_begin != sum_ntotal_local) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u global_id_begin=%lu, expected %lu\n", rank_, s,
                        (unsigned long)sm.global_id_begin, (unsigned long)sum_ntotal_local);
                consistent = false;
            }
            sum_ntotal_local += sm.ntotal_local;
        }
        if (sum_ntotal_local != s0.ntotal) {
            fprintf(stderr, "[rank %d] cross-shard: sum(ntotal_local)=%lu != ntotal=%lu\n", rank_,
                    (unsigned long)sum_ntotal_local, (unsigned long)s0.ntotal);
            consistent = false;
        }
        if (!consistent)
            MPI_Abort(MPI_COMM_WORLD, 1);
        if (rank_ == 0) {
            printf("[sync] Cross-shard consistency: OK (%u shards, ntotal=%lu, dim=%u, M=%u)\n", num_shards_,
                   (unsigned long)s0.ntotal, s0.dim, s0.M);
        }
    }

    // --- Map ALL shards read-only (own + remote) ---
    shard_ptrs_.assign(num_shards_, {nullptr, 0});
    for (uint32_t s = 0; s < num_shards_; s++) {
        std::string name = gd_name_for_shard(job_id_, s);
        shard_ptrs_[s] = {open_gd_readonly(name, shard_sizes_[s]), shard_sizes_[s]};
    }

    // --- Build searcher (validation already done during load/build) ---
    searcher_ = std::make_unique<GdHnswSearcher>(shard_ptrs_, /*skip_validation=*/true);
    searcher_->set_local_shard(my_shard);
    searcher_->set_expand_batch(expand_batch_);

    pushdown_active_ = (num_shards_ > 1);

    // -----------------------------------------------------------------
    // Single-shard fast path: vectors stay in GD, no channels / service.
    // -----------------------------------------------------------------
    if (!pushdown_active_) {
        // dot_norm is a multi-shard pushdown optimization (pre-computed norms +
        // dot-product kernel for remote distance services). It never activates
        // here: vectors stay in GD (vectors_localized_ == false), so
        // enable_dot_norm() would warn and silently self-disable. Explicitly
        // skip it instead and tell the user once — search_at_layer_local uses
        // l2_sqr, which computes the identical distances (dot_norm ≡ L2²).
        if (use_dot_norm_ && rank_ == 0) {
            printf("[sync] dot_norm ignored in single-shard mode (l2_sqr on GD vectors)\n");
        }
        srch_threads_ =
            (num_threads_ > 0) ? static_cast<uint32_t>(num_threads_) : static_cast<uint32_t>(omp_get_max_threads());
        svc_threads_ = 0;
        if (rank_ == 0)
            printf("[sync] Single-shard fast path (vectors in GD, no pushdown)\n");
        MPI_Barrier(MPI_COMM_WORLD); // match the multi-shard "services ready" barrier
        last_status_ = Status::Ok;
        return last_status_;
    }

    // -----------------------------------------------------------------
    // Multi-shard pushdown path.
    // -----------------------------------------------------------------

    // Load LOCAL shard's vector data to heap (remote vectors are null; pushdown
    // fetches them). Must happen BEFORE service setup (workers copy ShardView).
    {
        const auto &local_sv = searcher_->shards()[my_shard];
        int32_t ep = local_sv.sb->entry_point;
        uint64_t ep_gid = static_cast<uint64_t>(ep);
        bool ep_is_local =
            (ep_gid >= local_sv.global_id_begin && ep_gid < local_sv.global_id_begin + local_sv.ntotal_local);

        std::string my_path = index_file_for_shard(shard_base_path_, my_shard);
        searcher_->load_local_vectors(my_path);

        // If EP is on a remote shard, read its vector from the EP shard file.
        if (!ep_is_local) {
            uint32_t ep_shard = 0;
            for (uint32_t s = 0; s < num_shards_; s++) {
                const auto &sv = searcher_->shards()[s];
                if (ep_gid >= sv.global_id_begin && ep_gid < sv.global_id_begin + sv.ntotal_local) {
                    ep_shard = s;
                    break;
                }
            }
            std::string ep_path = index_file_for_shard(shard_base_path_, ep_shard);
            FILE *ep_fp = fopen(ep_path.c_str(), "rb");
            if (!ep_fp) {
                fprintf(stderr, "[rank %d] FATAL: cannot open EP shard file '%s': %s\n", rank_, ep_path.c_str(),
                        strerror(errno));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            SuperBlockV1 ep_sb{};
            if (fread(&ep_sb, sizeof(SuperBlockV1), 1, ep_fp) != 1) {
                fclose(ep_fp);
                fprintf(stderr, "[rank %d] FATAL: cannot read SuperBlock from EP shard '%s'\n", rank_, ep_path.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            uint64_t ep_local_id = ep_gid - ep_sb.global_id_begin;
            uint64_t vec_file_off = ep_sb.blocks[BLK_VECTORS].off + ep_local_id * dim_ * sizeof(float);
            std::vector<float> ep_vec(dim_);
            fseek(ep_fp, static_cast<long>(vec_file_off), SEEK_SET);
            if (fread(ep_vec.data(), sizeof(float), dim_, ep_fp) != dim_) {
                fclose(ep_fp);
                fprintf(stderr, "[rank %d] FATAL: short read on EP vector (gid %d) from '%s'\n", rank_, ep,
                        ep_path.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fclose(ep_fp);
            searcher_->set_ep_vector(ep_vec.data(), dim_);
            printf("[rank %d] EP vector (gid %d) loaded from shard %u file\n", rank_, ep, ep_shard);
        }
    }

    if (use_dot_norm_)
        searcher_->enable_dot_norm(true);

    // --- Resolve thread budgets ---
    auto resolve_service_threads = [&](uint32_t configured) -> uint32_t {
        if (configured != 0)
            return configured;
        uint32_t cores = static_cast<uint32_t>(std::max(1, num_threads_ > 0 ? num_threads_ : omp_get_max_threads()));
        uint32_t auto_svc = std::max<uint32_t>(1, cores / 4);
        uint32_t max_svc = std::max<uint32_t>(1, num_shards_ - 1);
        return std::min(auto_svc, max_svc);
    };

    svc_threads_ = resolve_service_threads(opts.service_threads);
    uint32_t svc_total_cores = svc_threads_;
    int thread_budget = (num_threads_ > 0) ? num_threads_ : omp_get_max_threads();

    if (svc_total_cores >= static_cast<uint32_t>(thread_budget)) {
        fprintf(stderr, "FATAL: service cores (%u) >= thread budget (%d). No threads left for search.\n",
                svc_total_cores, thread_budget);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (opts.search_threads == 0) {
        srch_threads_ = static_cast<uint32_t>(thread_budget) - svc_total_cores;
    } else {
        srch_threads_ = opts.search_threads;
        if (srch_threads_ + svc_total_cores > static_cast<uint32_t>(thread_budget)) {
            fprintf(stderr, "FATAL: search_threads (%u) + service cores (%u) > budget (%d).\n", srch_threads_,
                    svc_total_cores, thread_budget);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    omp_set_num_threads(static_cast<int>(srch_threads_));

    if (num_shards_ > kMaxShardsMask32) {
        fprintf(stderr, "FATAL: num_shards (%u) > %u. Pushdown pending_mask is uint32_t bitmask.\n", num_shards_,
                kMaxShardsMask32);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // max_batch = layer-0 neighbor count (cum_nn[1]), scaled by expand_batch.
    const auto *sb0 = static_cast<const SuperBlockV1 *>(shard_ptrs_[0].first);
    const int32_t *cum_nn0 = cum_nneighbor_ptr(shard_ptrs_[0].first, *sb0);
    uint32_t max_batch = static_cast<uint32_t>(cum_nn0[1]);
    for (uint32_t s = 1; s < num_shards_; s++) {
        const auto *sb_s = static_cast<const SuperBlockV1 *>(shard_ptrs_[s].first);
        const int32_t *cum_s = cum_nneighbor_ptr(shard_ptrs_[s].first, *sb_s);
        if (static_cast<uint32_t>(cum_s[1]) != max_batch) {
            fprintf(stderr, "FATAL: shard %u layer-0 neighbors=%d, shard 0 has %u. Must match.\n", s, cum_s[1],
                    max_batch);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    {
        uint64_t scaled = static_cast<uint64_t>(max_batch) * expand_batch_;
        if (scaled > UINT32_MAX) {
            fprintf(stderr, "FATAL: max_batch * expand_batch overflows uint32: %u * %u\n", max_batch, expand_batch_);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        max_batch = static_cast<uint32_t>(scaled);
    }

    // --- run_gen: rank 0 generates, broadcast to all ---
    if (rank_ == 0) {
        run_gen_ = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    MPI_Bcast(&run_gen_, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // --- Validate srch_threads consistency across participant ranks ---
    {
        std::vector<uint32_t> all_srch(size_);
        MPI_Allgather(&srch_threads_, 1, MPI_UINT32_T, all_srch.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD);
        for (uint32_t r = 0; r < num_shards_; r++) {
            if (all_srch[r] != srch_threads_) {
                fprintf(stderr, "FATAL: srch_threads mismatch: rank %d has %u, rank %u has %u\n", rank_, srch_threads_,
                        r, all_srch[r]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    const uint32_t num_remote = num_shards_ - 1;
    const uint32_t my_rank_u = static_cast<uint32_t>(rank_);

    TaskChannelLayout task_layout = TaskChannelLayout::build(max_batch, dim_);
    ResultChannelLayout result_layout = ResultChannelLayout::build(max_batch);

    // Own inbox base addresses, carried from Step 1 to Step 2 by name.
    // (Previously recovered via positional indexing into
    // requester_channel_maps_ — [size()-2]/[size()-1] — which silently
    // misroutes if the [task, result] push order ever changes.)
    void *own_task_base = nullptr;
    void *own_result_base = nullptr;

    // --- Step 1: create own task_inbox + result_inbox, init headers ---
    {
        std::string task_name = task_inbox_name_for_rank(job_id_, my_rank_u);
        ubsmem_deallocate_retry(task_name.c_str());
        size_t task_total = static_cast<size_t>(num_remote) * srch_threads_ * task_layout.bytes_total;
        size_t task_alloc = ubsmem_align_size(task_total);
        int ubret =
            ubsmem_shmem_allocate_with_provider(&provider_, task_name.c_str(), task_alloc, 0600, channel_flags_);
        if (ubret != UBSM_OK) {
            fprintf(stderr, "ubsmem_shmem_allocate_with_provider task_inbox '%s' failed: %d\n", task_name.c_str(),
                    ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        void *task_base = nullptr;
        ubret =
            ubsmem_shmem_map(nullptr, task_alloc, PROT_READ | PROT_WRITE, MAP_SHARED, task_name.c_str(), 0, &task_base);
        if (ubret != UBSM_OK || !task_base) {
            fprintf(stderr, "ubsmem_shmem_map task_inbox '%s' failed: %d\n", task_name.c_str(), ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::memset(task_base, 0, task_total);
        for (uint32_t ri = 0; ri < num_remote; ri++) {
            for (uint32_t t = 0; t < srch_threads_; t++) {
                size_t off = dual_channel_slot(ri, t, srch_threads_, task_layout.bytes_total);
                auto tv = TaskChannelView::from_raw(static_cast<char *>(task_base) + off, task_layout);
                tv.h->run_generation = run_gen_;
                tv.h->max_batch = max_batch;
                tv.h->dim = dim_;
                tv.h->channel_role = CHANNEL_ROLE_TASK;
                tv.h->dst_shard = static_cast<uint16_t>(my_shard);
                tv.h->identity_valid = IDENTITY_VALID_MAGIC;
            }
        }
        requester_channel_maps_.push_back({task_base, task_alloc});
        own_task_base = task_base;
        owned_channel_gd_names_.push_back(task_name);

        std::string result_name = result_inbox_name_for_rank(job_id_, my_rank_u);
        ubsmem_deallocate_retry(result_name.c_str());
        size_t result_total = static_cast<size_t>(num_remote) * srch_threads_ * result_layout.bytes_total;
        size_t result_alloc = ubsmem_align_size(result_total);
        ubret =
            ubsmem_shmem_allocate_with_provider(&provider_, result_name.c_str(), result_alloc, 0600, channel_flags_);
        if (ubret != UBSM_OK) {
            fprintf(stderr, "ubsmem_shmem_allocate_with_provider result_inbox '%s' failed: %d\n", result_name.c_str(),
                    ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        void *result_base = nullptr;
        ubret = ubsmem_shmem_map(nullptr, result_alloc, PROT_READ | PROT_WRITE, MAP_SHARED, result_name.c_str(), 0,
                                 &result_base);
        if (ubret != UBSM_OK || !result_base) {
            fprintf(stderr, "ubsmem_shmem_map result_inbox '%s' failed: %d\n", result_name.c_str(), ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::memset(result_base, 0, result_total);
        for (uint32_t ri = 0; ri < num_remote; ri++) {
            for (uint32_t t = 0; t < srch_threads_; t++) {
                size_t off = dual_channel_slot(ri, t, srch_threads_, result_layout.bytes_total);
                auto rv = ResultChannelView::from_raw(static_cast<char *>(result_base) + off, result_layout);
                rv.h->run_generation = run_gen_;
                rv.h->max_batch = max_batch;
                rv.h->dim = dim_;
                rv.h->channel_role = CHANNEL_ROLE_RESULT;
                rv.h->src_rank = static_cast<uint16_t>(my_rank_u);
                rv.h->identity_valid = IDENTITY_VALID_MAGIC;
            }
        }
        requester_channel_maps_.push_back({result_base, result_alloc});
        own_result_base = result_base;
        owned_channel_gd_names_.push_back(result_name);

        dual_pushdown_proxies_.assign(srch_threads_, std::vector<DualDistProxy>(num_shards_));
    }

    // --- Collective: wait for all ranks to finish channel creation ---
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Step 2: open remote channels, wire proxies, start service ---
    {
        size_t task_file_bytes = static_cast<size_t>(num_remote) * srch_threads_ * task_layout.bytes_total;
        size_t result_file_bytes = static_cast<size_t>(num_remote) * srch_threads_ * result_layout.bytes_total;

        struct RemoteMapping {
            void *task_base;
            void *result_base;
            size_t task_alloc;
            size_t result_alloc;
        };
        std::vector<RemoteMapping> remote_maps(num_shards_);

        for (uint32_t src = 0; src < num_shards_; src++) {
            if (src == my_shard)
                continue;
            std::string remote_task = task_inbox_name_for_rank(job_id_, src);
            size_t t_alloc = ubsmem_align_size(task_file_bytes);
            void *t_base = nullptr;
            int ubret =
                ubsmem_shmem_map(nullptr, t_alloc, PROT_READ | PROT_WRITE, MAP_SHARED, remote_task.c_str(), 0, &t_base);
            if (ubret != UBSM_OK || !t_base) {
                fprintf(stderr, "rank %d: ubsmem_shmem_map task_inbox '%s' failed: %d\n", rank_, remote_task.c_str(),
                        ubret);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            service_channel_maps_.push_back({t_base, t_alloc});
            remote_maps[src].task_base = t_base;
            remote_maps[src].task_alloc = t_alloc;

            std::string remote_result = result_inbox_name_for_rank(job_id_, src);
            size_t r_alloc = ubsmem_align_size(result_file_bytes);
            void *r_base = nullptr;
            ubret = ubsmem_shmem_map(nullptr, r_alloc, PROT_READ | PROT_WRITE, MAP_SHARED, remote_result.c_str(), 0,
                                     &r_base);
            if (ubret != UBSM_OK || !r_base) {
                fprintf(stderr, "rank %d: ubsmem_shmem_map result_inbox '%s' failed: %d\n", rank_,
                        remote_result.c_str(), ubret);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            service_channel_maps_.push_back({r_base, r_alloc});
            remote_maps[src].result_base = r_base;
            remote_maps[src].result_alloc = r_alloc;
        }

        // Wire DualDistProxy: task slot in remote task_inbox, result slot in own result_inbox.
        for (uint32_t t = 0; t < srch_threads_; t++) {
            for (uint32_t s = 0; s < num_shards_; s++) {
                if (s == my_shard)
                    continue;
                uint32_t task_ri = compressed_remote_idx(my_rank_u, s);
                size_t task_off = dual_channel_slot(task_ri, t, srch_threads_, task_layout.bytes_total);
                auto tv =
                    TaskChannelView::from_raw(static_cast<char *>(remote_maps[s].task_base) + task_off, task_layout);

                uint32_t result_ri = compressed_remote_idx(s, my_rank_u);
                size_t result_off = dual_channel_slot(result_ri, t, srch_threads_, result_layout.bytes_total);
                auto rv = ResultChannelView::from_raw(static_cast<char *>(own_result_base) + result_off, result_layout);

                dual_pushdown_proxies_[t][s].init(tv, rv, dim_, MAX_SPIN_ITERS_DEFAULT, max_batch,
                                                  static_cast<uint16_t>(my_rank_u), static_cast<uint16_t>(t));
            }
        }

        // Collect service channel pairs: own task_inbox + remote result_inboxes.
        std::vector<ServiceChannelPair> service_pairs;
        for (uint32_t src = 0; src < num_shards_; src++) {
            if (src == my_shard)
                continue;
            uint32_t src_ri = compressed_remote_idx(src, my_rank_u);
            for (uint32_t t = 0; t < srch_threads_; t++) {
                size_t task_off = dual_channel_slot(src_ri, t, srch_threads_, task_layout.bytes_total);
                auto tv = TaskChannelView::from_raw(static_cast<char *>(own_task_base) + task_off, task_layout);

                uint32_t result_ri = compressed_remote_idx(my_rank_u, src);
                size_t result_off = dual_channel_slot(result_ri, t, srch_threads_, result_layout.bytes_total);
                auto rv = ResultChannelView::from_raw(static_cast<char *>(remote_maps[src].result_base) + result_off,
                                                      result_layout);

                service_pairs.push_back({tv, rv});
            }
        }

        const auto &my_shard_view = searcher_->shard_by_id(my_shard);
        // core_ids omitted: make_unique forwards to the constructor, which
        // applies its default (empty -> no core pinning). A bare {} can't be
        // passed here because make_unique's forwarding parameter pack cannot
        // deduce a type from a braced-init-list.
        dual_dist_services_.push_back(std::make_unique<DualDistService>(my_shard, my_shard_view, dim_, run_gen_,
                                                                        svc_threads_, std::move(service_pairs)));

        if (use_dot_norm_) {
            dual_dist_services_.back()->enable_dot_norm(searcher_->vector_norms_data());
        }

        searcher_->set_dual_pushdown_proxies(&dual_pushdown_proxies_, srch_threads_);
    }

    if (rank_ == 0) {
        printf("[pushdown] Enabled (dual-channel): search_threads=%u, service_threads=%u, "
               "max_batch=%u, channels_per_rank=%u\n",
               srch_threads_, svc_threads_, max_batch, srch_threads_ * num_remote);
    }

    // --- Collective: all service threads ready before search may begin ---
    MPI_Barrier(MPI_COMM_WORLD);

    last_status_ = Status::Ok;
    return last_status_;
}

// ============================================================
// search (non-collective)
// ============================================================

SearchResult Context::Impl::search(const float *queries, uint64_t nq, const SearchParams &params, int32_t *out_ids,
                                   float *out_dists)
{
    SearchResult r;
    if (!searcher_) {
        r.status = Status::NotInitialized;
        r.error = "node_init_and_sync() not called";
        return r;
    }
    if (nq > 0 && queries == nullptr) {
        r.status = Status::InvalidArg;
        r.error = "null queries";
        return r;
    }
    if (params.k <= 0) {
        r.status = Status::InvalidArg;
        r.error = "k must be > 0";
        return r;
    }
    if (nq > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        r.status = Status::InvalidArg;
        r.error = "nq exceeds int32 max (2147483647)";
        return r;
    }
    if (nq > 0 && out_ids == nullptr) {
        r.status = Status::InvalidArg;
        r.error = "null out_ids";
        return r;
    }
    if (nq > 0 && out_dists == nullptr) {
        r.status = Status::InvalidArg;
        r.error = "null out_dists";
        return r;
    }
    if (params.ef_search < 0) {
        r.status = Status::InvalidArg;
        r.error = "ef_search must be >= 0";
        return r;
    }

    HnswQueryParams hp;
    hp.k = params.k;
    hp.ef_search = params.ef_search;
    searcher_->search_batch_flat(queries, static_cast<int32_t>(nq), hp, out_ids, out_dists);
    r.status = Status::Ok;
    return r;
}

// ============================================================
// finalize (collective)
// ============================================================

Status Context::Impl::finalize()
{
    // initialize() may fail after MPI/ubsmem are ready but before initialized_
    // is set; do not early-return on initialized_ — each step below is gated by
    // mpi_ready_/ubsmem_ready_, so partial-init cleanup still runs.
    if (mpi_ready_ && pushdown_active_)
        MPI_Barrier(MPI_COMM_WORLD);

    // Stop service workers (joins threads).
    dual_dist_services_.clear();
    dual_pushdown_proxies_.clear();

    // Unmap service-side channel gd.
    for (auto &[ptr, sz] : service_channel_maps_)
        ubsmem_shmem_unmap(ptr, sz);
    service_channel_maps_.clear();

    // Unmap requester-side channel gd.
    for (auto &[ptr, sz] : requester_channel_maps_)
        ubsmem_shmem_unmap(ptr, sz);
    requester_channel_maps_.clear();

    // Release searcher (drops references into shard GD).
    searcher_.reset();

    // Unmap read-only shard mappings.
    for (auto &[ptr, sz] : shard_ptrs_) {
        if (ptr)
            ubsmem_shmem_unmap(const_cast<void *>(ptr), ubsmem_align_size(sz));
    }
    shard_ptrs_.clear();

    if (mpi_ready_) {
        MPI_Barrier(MPI_COMM_WORLD); // all unmapped before any deallocates

        // Deallocate channel gd segments owned by this rank.
        for (const auto &name : owned_channel_gd_names_) {
            int dr = ubsmem_deallocate_retry(name.c_str());
            if (dr != UBSM_OK) {
                fprintf(stderr, "[rank %d] WARNING: deallocate('%s') failed: %d\n", rank_, name.c_str(), dr);
            }
        }
        owned_channel_gd_names_.clear();

        MPI_Barrier(MPI_COMM_WORLD); // channel dealloc done before shard cleanup

        // Rank 0 deallocates all shard gd segments.
        if (rank_ == 0) {
            for (uint32_t s = 0; s < num_shards_; s++) {
                std::string name = gd_name_for_shard(job_id_, s);
                int dr = ubsmem_deallocate_retry(name.c_str());
                if (dr != UBSM_OK) {
                    fprintf(stderr, "[rank 0] WARNING: deallocate('%s') failed: %d\n", name.c_str(), dr);
                }
            }
        }
    }

    if (ubsmem_ready_) {
        ubsmem_finalize();
        ubsmem_ready_ = false;
    }
    if (mpi_ready_ && auto_mpi_init_) {
        MPI_Finalize();
        mpi_ready_ = false;
    }

    initialized_ = false;
    last_status_ = Status::Ok;
    return last_status_;
}

// ============================================================
// Context forwarding (pimpl)
// ============================================================

Context::Context() : pimpl_(std::make_unique<Impl>()) {}
Context::~Context() = default;

Status Context::initialize(int *argc, char ***argv, const InitOptions &opts)
{
    return pimpl_->initialize(argc, argv, opts);
}
BuildResult Context::build(const BuildOptions &opts) { return pimpl_->build(opts); }
LoadResult Context::load(uint32_t node_id, const std::string &shard_base_path)
{
    return pimpl_->load(node_id, shard_base_path);
}
Status Context::node_init_and_sync(uint32_t node_id, const DeployOptions &opts)
{
    return pimpl_->node_init_and_sync(node_id, opts);
}
SearchResult Context::search(const float *queries, uint64_t nq, const SearchParams &params, int32_t *out_ids,
                             float *out_dists)
{
    return pimpl_->search(queries, nq, params, out_ids, out_dists);
}
Status Context::finalize() { return pimpl_->finalize(); }

bool Context::initialized() const { return pimpl_ && pimpl_->initialized_; }
int Context::rank() const { return pimpl_ ? pimpl_->rank_ : -1; }
int Context::size() const { return pimpl_ ? pimpl_->size_ : 1; }
uint32_t Context::num_shards() const { return pimpl_ ? pimpl_->num_shards_ : 0; }
uint32_t Context::dim() const { return pimpl_ ? pimpl_->dim_ : 0; }
uint64_t Context::ntotal() const { return pimpl_ ? pimpl_->ntotal_ : 0; }
bool Context::has_centroids() const { return pimpl_ && pimpl_->has_centroids_; }
bool Context::has_idmap() const { return pimpl_ && pimpl_->has_idmap_; }
const std::vector<uint32_t> &Context::idmap() const
{
    static const std::vector<uint32_t> empty;
    return pimpl_ ? pimpl_->idmap_ : empty;
}
Status Context::last_status() const { return pimpl_ ? pimpl_->last_status_ : Status::NotInitialized; }

// ============================================================
// route_queries (static, non-collective) — delegates to Impl
// ============================================================

std::vector<uint32_t> Context::route_queries(const Context &ctx, const float *queries, uint64_t nq)
{
    if (!ctx.pimpl_ || nq == 0)
        return {};
    return ctx.pimpl_->route_queries(queries, nq);
}

std::vector<uint32_t> Context::Impl::route_queries(const float *queries, uint64_t nq) const
{
    std::vector<uint32_t> idx;
    if (nq == 0)
        return idx;
    if (queries == nullptr) {
        fprintf(stderr, "Error: route_queries called with null queries (nq=%llu)\n",
                static_cast<unsigned long long>(nq));
        return idx;
    }
    const uint32_t dim = dim_;
    const uint32_t num_shards = num_shards_;
    const int my_rank = rank_;

    if (has_centroids_ && num_shards > 0 && dim > 0) {
        // Nearest-centroid routing: return indices owned by my rank.
        for (uint64_t q = 0; q < nq; q++) {
            const float *qvec = queries + q * dim;
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_shard = 0;
            for (uint32_t s = 0; s < num_shards; s++) {
                const float *cvec = centroids_.data() + static_cast<size_t>(s) * dim;
                float dist = 0.0f;
                for (uint32_t d = 0; d < dim; d++) {
                    float diff = qvec[d] - cvec[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_shard = s;
                }
            }
            if (static_cast<int>(best_shard) == my_rank)
                idx.push_back(static_cast<uint32_t>(q));
        }
    } else if (my_rank >= 0 && num_shards > 0) {
        // Uniform split.
        uint64_t per = (nq + num_shards - 1) / num_shards;
        uint64_t q_start = static_cast<uint64_t>(my_rank) * per;
        uint64_t q_end = std::min(q_start + per, nq);
        for (uint64_t i = q_start; i < q_end; i++)
            idx.push_back(static_cast<uint32_t>(i));
    }
    return idx;
}

} // namespace gd_hnsw
