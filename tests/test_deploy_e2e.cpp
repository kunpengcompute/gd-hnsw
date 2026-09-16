/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// test_deploy_e2e.cpp — Layer 4 deploy API end-to-end tests.
//
// CMake gate: GD_HNSW_DEPLOY_OK (Linux + MPI + ubs-mem + HDF5 + NUMA).
// Two ctest registrations sharing this binary:
//   test_deploy_e2e        — single-process, --gtest_filter=-*Mpi*
//   test_deploy_e2e_mpi    — mpirun -np 2, --gtest_filter=*Mpi*
//
// MPI deadlock safety rules (for DeployMpi* tests):
//   1. Use EXPECT_* only, never ASSERT_* — single-rank early return deadlocks
//      collective calls.
//   2. Guard entry: verify MPI_Comm_size == expected, else GTEST_SKIP.
//   3. If initialize() returns UbsemError, GTEST_SKIP (daemon not running).
//
// Data construction: ported from test_pushdown_fork's build_shard_buf() —
// hand-built SuperBlock written to temp files. The deploy layer's job is
// orchestration, not index quality (already covered by test_search_e2e).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>

#include <unistd.h> // getpid
#include <mpi.h>

#include "gd_hnsw_api.h"
#include "gd_layout.h"
#include "index_io.h"
#include "build.h"

using namespace gd_hnsw;

// ============================================================
// Helpers — ported from test_pushdown_fork::build_shard_buf
// ============================================================

static constexpr uint32_t DEPLOY_DIM = 8;
static constexpr uint32_t DEPLOY_M = 4;

static std::vector<char> build_shard_buf(uint64_t ntotal, uint64_t nlocal, uint32_t dim, uint32_t M, uint32_t shard_id,
                                         uint32_t num_shards, uint64_t gid_begin, int32_t entry_point)
{
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.state = static_cast<uint32_t>(GdState::ACTIVE);
    sb.metric_type = METRIC_L2;
    sb.endianness = ENDIAN_LITTLE;
    sb.ntotal = ntotal;
    sb.ntotal_local = nlocal;
    sb.dim = dim;
    sb.M = M;
    sb.ef_search = 32;
    sb.max_level = 0;
    sb.entry_point = entry_point;
    sb.num_levels = 2;
    sb.num_shards = num_shards;
    sb.shard_id = shard_id;
    sb.global_id_begin = gid_begin;
    sb.neighbors_size = nlocal * 2 * M;
    compute_layout(sb);

    std::vector<char> buf(sb.total_bytes, 0);
    std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));
    auto *sbp = reinterpret_cast<SuperBlockV1 *>(buf.data());

    // Vectors: 1D line — vec[gid] = gid / ntotal (all dims same value)
    float *vecs = reinterpret_cast<float *>(buf.data() + sb.blocks[BLK_VECTORS].off);
    for (uint64_t lid = 0; lid < nlocal; lid++) {
        uint64_t gid = gid_begin + lid;
        float val = static_cast<float>(gid) / static_cast<float>(ntotal);
        for (uint32_t d = 0; d < dim; d++)
            vecs[lid * dim + d] = val;
    }

    // Levels
    int32_t *levels = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_LEVELS].off);
    for (uint64_t lid = 0; lid < nlocal; lid++)
        levels[lid] = 1;

    // Offsets
    uint64_t *offsets = reinterpret_cast<uint64_t *>(buf.data() + sb.blocks[BLK_OFFSETS].off);
    uint64_t nn = 2 * M;
    for (uint64_t lid = 0; lid <= nlocal; lid++)
        offsets[lid] = lid * nn;

    // Neighbors: 1D-line topology, ±1..±M (global IDs)
    int32_t *neighbors = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_NEIGHBORS].off);
    for (uint64_t lid = 0; lid < nlocal; lid++) {
        int64_t gid = static_cast<int64_t>(gid_begin + lid);
        int32_t *nb = neighbors + lid * nn;
        for (uint64_t j = 0; j < nn; j++)
            nb[j] = EMPTY_NEIGHBOR;
        for (uint32_t r = 0; r < M; r++) {
            int64_t left = gid - 1 - static_cast<int64_t>(r);
            int64_t right = gid + 1 + static_cast<int64_t>(r);
            if (left >= 0)
                nb[2 * r] = static_cast<int32_t>(left);
            if (right < static_cast<int64_t>(ntotal))
                nb[2 * r + 1] = static_cast<int32_t>(right);
        }
    }

    // cum_nneighbor
    int32_t *cum_nn = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_CUM_NNEIGHBOR].off);
    cum_nn[0] = 0;
    cum_nn[1] = static_cast<int32_t>(nn);

    sbp->header_crc64 = compute_header_crc64(*sbp);
    sbp->payload_crc64 = compute_payload_crc64(buf.data(), *sbp);
    return buf;
}

// Write a single shard buffer to temp dir, return the base_path (without .shard_N suffix).
static std::string write_shard_file(const std::string &tmp_dir, uint32_t shard_id, const std::vector<char> &buf)
{
    std::string base = tmp_dir + "/deploy_test";
    std::string path = index_file_for_shard(base, shard_id);
    // Remove stale sidecars left by earlier tests/runs (all tests here share the
    // same base path): a leftover .idmap/.centroids would make load() fail with
    // IoError in tests that don't write sidecars themselves. Tests that DO write
    // sidecars always do so after this call, so ordering is safe.
    unlink(idmap_file_for_index(base).c_str());
    unlink(centroids_file_for_index(base).c_str());
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) {
        ADD_FAILURE() << "cannot create '" << path << "': " << strerror(errno);
        return {};
    }
    size_t written = fwrite(buf.data(), 1, buf.size(), fp);
    fclose(fp);
    if (written != buf.size()) {
        ADD_FAILURE() << "short write to '" << path << "'";
        return {};
    }
    return base;
}

// Brute-force L2 KNN for single-shard validation.
static void brute_force_knn(const float *vectors, uint64_t ntotal, uint32_t dim, const float *query, int32_t k,
                            std::vector<int32_t> &out_ids, std::vector<float> &out_dists)
{
    std::vector<std::pair<float, int32_t>> pairs;
    pairs.reserve(ntotal);
    for (uint64_t i = 0; i < ntotal; i++) {
        float dist = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float diff = vectors[i * dim + d] - query[d];
            dist += diff * diff;
        }
        pairs.push_back({dist, static_cast<int32_t>(i)});
    }
    std::sort(pairs.begin(), pairs.end());
    int32_t count = std::min(static_cast<int32_t>(pairs.size()), k);
    out_ids.resize(count);
    out_dists.resize(count);
    for (int32_t i = 0; i < count; i++) {
        out_ids[i] = pairs[i].second;
        out_dists[i] = pairs[i].first;
    }
}

// Write a corrupted sidecar file (bad magic) to trigger IoError in load().
static void write_corrupted_sidecar(const std::string &path)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
        return;
    uint64_t bad_magic = 0xDEADBEEFCAFEBABEULL;
    fwrite(&bad_magic, sizeof(bad_magic), 1, fp);
    fclose(fp);
}

// Write raw bytes to a file (for malformed-shard-file tests).
static bool write_raw_file(const std::string &path, const void *data, size_t len)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    size_t n = fwrite(data, 1, len, fp);
    fclose(fp);
    return n == len;
}

// Concatenate vectors from multiple shard buffers for multi-shard brute force.
static std::vector<float> extract_all_vectors(const std::vector<const std::vector<char> *> &bufs, uint32_t dim,
                                              uint64_t ntotal)
{
    std::vector<float> all_vecs(ntotal * dim);
    for (const auto *buf : bufs) {
        const auto *sbp = reinterpret_cast<const SuperBlockV1 *>(buf->data());
        const float *vecs = reinterpret_cast<const float *>(buf->data() + sbp->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < sbp->ntotal_local; lid++) {
            uint64_t gid = sbp->global_id_begin + lid;
            std::memcpy(&all_vecs[gid * dim], &vecs[lid * dim], dim * sizeof(float));
        }
    }
    return all_vecs;
}

// ============================================================
// Phase 1: Single-process tests (gtest filter: -*Mpi*)
// ============================================================

// --- DeployInitFinalizeRoundtrip ---
// initialize() → finalize() state machine; second initialize() → AlreadyInit.
TEST(Deploy, InitFinalizeRoundtrip)
{
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    EXPECT_EQ(s, Status::Ok);
    EXPECT_TRUE(ctx.initialized());

    // Second initialize must fail
    s = ctx.initialize(nullptr, nullptr);
    EXPECT_EQ(s, Status::AlreadyInit);

    s = ctx.finalize();
    EXPECT_EQ(s, Status::Ok);
    EXPECT_FALSE(ctx.initialized());
}

// --- DeployNegativeArgs ---
// Error paths: empty path, bad magic, search before sync.
TEST(Deploy, NegativeArgs)
{
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    // load() with empty path
    LoadResult lr = ctx.load(0, "");
    EXPECT_EQ(lr.status, Status::InvalidArg);

    // load() with non-existent file
    lr = ctx.load(0, "/tmp/gd_hnsw_nonexistent_test_path");
    EXPECT_EQ(lr.status, Status::IoError);

    // search() before node_init_and_sync()
    float query[DEPLOY_DIM] = {0};
    int32_t ids[5];
    float dists[5];
    SearchParams sp(5);
    SearchResult sr = ctx.search(query, 1, sp, ids, dists);
    EXPECT_EQ(sr.status, Status::NotInitialized);

    ctx.finalize();
}

// --- DeploySingleShardE2E ---
// Single-shard: initialize → load → node_init_and_sync → search → finalize.
// Validates: GD ACTIVE + CRC, search results vs brute-force, result ordering.
TEST(Deploy, SingleShardE2E)
{
    static constexpr uint64_t N = 200;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    // Build single-shard index file
    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string tmp_dir = ::testing::TempDir();
    std::string base = write_shard_file(tmp_dir, 0, buf);
    ASSERT_FALSE(base.empty());

    // load()
    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    ASSERT_EQ(lr.status, Status::Ok) << lr.error;
    EXPECT_EQ(lr.ntotal, N);
    EXPECT_EQ(lr.dim, DEPLOY_DIM);
    EXPECT_EQ(lr.num_shards, 1u);
    EXPECT_EQ(lr.shard_id, 0u);

    // node_init_and_sync() — single-shard fast path (no channels)
    s = ctx.node_init_and_sync(0);
    ASSERT_EQ(s, Status::Ok);
    EXPECT_EQ(ctx.num_shards(), 1u);
    EXPECT_EQ(ctx.dim(), DEPLOY_DIM);
    EXPECT_EQ(ctx.ntotal(), N);

    // Search — query near middle of the 1D line
    float query[DEPLOY_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = qval;

    int32_t k = 5;
    int32_t ids[5];
    float dists[5];
    SearchParams sp(k);
    SearchResult sr = ctx.search(query, 1, sp, ids, dists);
    ASSERT_EQ(sr.status, Status::Ok) << sr.error;

    // All results should be non-empty
    for (int i = 0; i < k; i++) {
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
    }

    // Distances should be sorted ascending
    for (int i = 1; i < k; i++) {
        EXPECT_LE(dists[i - 1], dists[i]) << "distances not sorted at " << i;
    }

    // Brute-force validation: extract vectors from the buffer
    const auto *sbp = reinterpret_cast<const SuperBlockV1 *>(buf.data());
    const float *vectors = reinterpret_cast<const float *>(buf.data() + sbp->blocks[BLK_VECTORS].off);
    std::vector<int32_t> bf_ids;
    std::vector<float> bf_dists;
    brute_force_knn(vectors, N, DEPLOY_DIM, query, k, bf_ids, bf_dists);

    // HNSW should find the same or better top-k (approximate — at least the best result)
    // For this simple 1D line graph, recall should be perfect or near-perfect.
    // Check: all returned IDs are within the brute-force top-k set
    std::vector<int32_t> bf_set(bf_ids.begin(), bf_ids.end());
    std::sort(bf_set.begin(), bf_set.end());
    int match_count = 0;
    for (int i = 0; i < k; i++) {
        if (std::binary_search(bf_set.begin(), bf_set.end(), ids[i]))
            match_count++;
    }
    // At least k-1 out of k should match (allow 1 miss for tie-breaking edge cases)
    EXPECT_GE(match_count, k - 1) << "recall too low: " << match_count << "/" << k;

    s = ctx.finalize();
    EXPECT_EQ(s, Status::Ok);

    // Clean up temp file
    std::string path = index_file_for_shard(base, 0);
    unlink(path.c_str());
}

// --- DeployLoadBeforeInit ---
// load() before initialize() → NotInitialized.
TEST(Deploy, LoadBeforeInit)
{
    Context ctx;
    LoadResult lr = ctx.load(0, "/tmp/gd_hnsw_test_no_init");
    EXPECT_EQ(lr.status, Status::NotInitialized);
}

// --- DeployFinalizeWithoutInit ---
// finalize() without initialize() → Ok (no-ops, partial-init cleanup paths).
TEST(Deploy, FinalizeWithoutInit)
{
    Context ctx;
    Status s = ctx.finalize();
    EXPECT_EQ(s, Status::Ok);
}

// --- DeploySearchNegativeArgs ---
// After full single-shard flow, test search() negative parameter paths.
TEST(Deploy, SearchNegativeArgs)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string tmp_dir = ::testing::TempDir();
    std::string base = write_shard_file(tmp_dir, 0, buf);
    ASSERT_FALSE(base.empty());

    {
        auto _lr = ctx.load(0, base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        ASSERT_EQ(_lr.status, Status::Ok);
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);

    // null queries, nq=1 → InvalidArg
    int32_t ids[5];
    float dists[5];
    SearchParams sp(5);
    SearchResult sr = ctx.search(nullptr, 1, sp, ids, dists);
    EXPECT_EQ(sr.status, Status::InvalidArg);

    // k<=0 → InvalidArg
    float q[DEPLOY_DIM] = {0};
    sr = ctx.search(q, 1, SearchParams(0), ids, dists);
    EXPECT_EQ(sr.status, Status::InvalidArg);

    // nq exceeds INT32_MAX → InvalidArg
    sr = ctx.search(q, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1, sp, ids, dists);
    EXPECT_EQ(sr.status, Status::InvalidArg);

    // null out_ids, nq=1 → InvalidArg
    sr = ctx.search(q, 1, sp, nullptr, dists);
    EXPECT_EQ(sr.status, Status::InvalidArg);

    // null out_dists, nq=1 → InvalidArg
    sr = ctx.search(q, 1, sp, ids, nullptr);
    EXPECT_EQ(sr.status, Status::InvalidArg);

    // ef_search<0 → InvalidArg
    SearchParams sp_neg(5);
    sp_neg.ef_search = -1;
    sr = ctx.search(q, 1, sp_neg, ids, dists);
    EXPECT_EQ(sr.status, Status::InvalidArg);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployLoadSidecars_* ---
// load() with idmap / centroids sidecar files: valid, size mismatch, corrupt.
// Split into individual tests for process isolation (1 Context per process).

TEST(Deploy, LoadSidecars_ValidIdmap)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    std::vector<uint32_t> idmap(N);
    for (uint64_t i = 0; i < N; i++)
        idmap[i] = static_cast<uint32_t>(i);
    save_idmap_file(base, idmap);

    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::Ok) << lr.error;
    EXPECT_TRUE(ctx.has_idmap());
    EXPECT_EQ(ctx.idmap().size(), N);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(idmap_file_for_index(base).c_str());
}

TEST(Deploy, LoadSidecars_IdmapMismatch)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    std::vector<uint32_t> idmap(N - 1);
    save_idmap_file(base, idmap);

    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::ValidationErr);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(idmap_file_for_index(base).c_str());
}

TEST(Deploy, LoadSidecars_CorruptIdmap)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    write_corrupted_sidecar(idmap_file_for_index(base));

    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::IoError);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(idmap_file_for_index(base).c_str());
}

TEST(Deploy, LoadSidecars_ValidCentroids)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    std::vector<float> cents(DEPLOY_DIM, 0.5f);
    save_centroids_file(base, cents, 1, DEPLOY_DIM);

    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::Ok) << lr.error;
    EXPECT_TRUE(ctx.has_centroids());

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(centroids_file_for_index(base).c_str());
}

TEST(Deploy, LoadSidecars_CentroidMismatch)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    std::vector<float> cents(DEPLOY_DIM * 2, 0.0f);
    save_centroids_file(base, cents, 2, DEPLOY_DIM);

    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::ValidationErr);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(centroids_file_for_index(base).c_str());
}

TEST(Deploy, LoadSidecars_CorruptCentroids)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    write_corrupted_sidecar(centroids_file_for_index(base));

    LoadResult lr = ctx.load(0, base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::IoError);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(centroids_file_for_index(base).c_str());
}

// --- DeployRouteQueries_* ---
// route_queries(): nq=0, null queries, uniform split, centroid routing.
// Split for process isolation (1 Context per process).

TEST(Deploy, RouteQueries_NullAndUniform)
{
    // nq=0 → empty (no Context needed, but need one for pimpl)
    Context ctx0;
    float dummy[DEPLOY_DIM] = {0};
    auto idx0 = Context::route_queries(ctx0, dummy, 0);
    EXPECT_TRUE(idx0.empty());

    // null queries, nq>0 → empty
    auto idx_null = Context::route_queries(ctx0, nullptr, 5);
    EXPECT_TRUE(idx_null.empty());

    // With initialized Context: uniform split (no centroids)
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    {
        auto _lr = ctx.load(0, base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        ASSERT_EQ(_lr.status, Status::Ok);
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);

    float queries[2 * DEPLOY_DIM];
    for (int i = 0; i < 2; i++)
        for (uint32_t d = 0; d < DEPLOY_DIM; d++)
            queries[i * DEPLOY_DIM + d] = 0.3f + 0.1f * i;

    auto idx = Context::route_queries(ctx, queries, 2);
    EXPECT_EQ(idx.size(), 2u);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
}

TEST(Deploy, RouteQueries_CentroidRouting)
{
    static constexpr uint64_t N = 50;
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem runtime unavailable";
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    std::vector<float> cents(DEPLOY_DIM, 0.5f);
    save_centroids_file(base, cents, 1, DEPLOY_DIM);

    {
        auto _lr = ctx.load(0, base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        ASSERT_EQ(_lr.status, Status::Ok);
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);
    ASSERT_TRUE(ctx.has_centroids());

    float query[DEPLOY_DIM];
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = 0.5f;
    auto idx = Context::route_queries(ctx, query, 1);
    EXPECT_EQ(idx.size(), 1u);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
    unlink(centroids_file_for_index(base).c_str());
}

// --- DeployDotNormSingleShard ---
// Single-shard with use_dot_norm=true: covers the "dot_norm ignored" branch.
TEST(Deploy, DotNormSingleShard)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string tmp_dir = ::testing::TempDir();
    std::string base = write_shard_file(tmp_dir, 0, buf);
    ASSERT_FALSE(base.empty());

    {
        auto _lr = ctx.load(0, base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        ASSERT_EQ(_lr.status, Status::Ok);
    }

    // DeployOptions with use_dot_norm=true (single-shard ignores it)
    DeployOptions opts;
    opts.use_dot_norm = true;
    s = ctx.node_init_and_sync(0, opts);
    EXPECT_EQ(s, Status::Ok);

    // Search should still work (l2_sqr on GD vectors)
    float query[DEPLOY_DIM];
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = 0.5f;
    int32_t ids[5];
    float dists[5];
    SearchParams sp(5);
    SearchResult sr = ctx.search(query, 1, sp, ids, dists);
    EXPECT_EQ(sr.status, Status::Ok);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployLoadNodeIdMismatch ---
// load() with node_id != MPI rank → InvalidArg (singleton world: rank 0).
TEST(Deploy, LoadNodeIdMismatch)
{
    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    // Rejected before any file I/O; path may be anything.
    LoadResult lr = ctx.load(1, "/tmp/gd_hnsw_node_id_mismatch");
    EXPECT_EQ(lr.status, Status::InvalidArg);
    EXPECT_NE(lr.error.find("MPI rank"), std::string::npos);

    ctx.finalize();
}

// --- DeployLoadBadFiles ---
// Malformed shard files: empty, truncated header, truncated payload,
// corrupted payload CRC.
TEST(Deploy, LoadBadFiles)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    std::string tmp_dir = ::testing::TempDir();
    std::string base = tmp_dir + "/deploy_test_bad";
    std::string path = index_file_for_shard(base, 0);

    // (a) Empty file → IoError (peek cannot read header)
    {
        char dummy = 0;
        ASSERT_TRUE(write_raw_file(path, &dummy, 0));
        LoadResult lr = ctx.load(0, base);
        EXPECT_EQ(lr.status, Status::IoError);
        unlink(path.c_str());
    }

    // (b) Truncated header (< sizeof(SuperBlockV1)) → IoError
    {
        auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
        ASSERT_TRUE(write_raw_file(path, buf.data(), 64));
        LoadResult lr = ctx.load(0, base);
        EXPECT_EQ(lr.status, Status::IoError);
        unlink(path.c_str());
    }

    // (c) Truncated payload (header ok, total_bytes > file size) → ValidationErr
    {
        auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
        const auto *sbp = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        size_t truncated = sbp->total_bytes / 2;
        ASSERT_TRUE(write_raw_file(path, buf.data(), truncated));
        LoadResult lr = ctx.load(0, base);
        EXPECT_EQ(lr.status, Status::ValidationErr);
        unlink(path.c_str());
    }

    // (d) Corrupted payload byte (CRC mismatch, magic intact) → ValidationErr
    {
        auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
        const auto *sbp = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        buf[sbp->blocks[BLK_VECTORS].off + 16] ^= 0xFF;
        ASSERT_TRUE(write_raw_file(path, buf.data(), buf.size()));
        LoadResult lr = ctx.load(0, base);
        EXPECT_EQ(lr.status, Status::ValidationErr);
        unlink(path.c_str());
    }

    // (e) Wrong magic (file size ok, header readable, magic mismatch) →
    // ValidationErr "bad magic" — rejected at the peek stage before
    // load_local_shard_direct_to_gd is called.
    {
        auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
        auto *sbp = reinterpret_cast<SuperBlockV1 *>(buf.data());
        sbp->magic = 0xDEADBEEFU; // wrong magic
        ASSERT_TRUE(write_raw_file(path, buf.data(), buf.size()));
        LoadResult lr = ctx.load(0, base);
        EXPECT_EQ(lr.status, Status::ValidationErr);
        EXPECT_NE(lr.error.find("bad magic"), std::string::npos);
        unlink(path.c_str());
    }

    ctx.finalize();
}

// --- DeploySyncBeforeInit ---
// node_init_and_sync() before initialize() and after initialize() but
// before load() → NotInitialized (two distinct guard branches).
TEST(Deploy, SyncBeforeInit)
{
    // Before initialize() → ubsmem not ready
    {
        Context ctx;
        EXPECT_EQ(ctx.node_init_and_sync(0), Status::NotInitialized);
    }

    // After initialize(), before load() → load_meta_ empty
    {
        Context ctx;
        Status s = ctx.initialize(nullptr, nullptr);
        if (s == Status::UbsemError) {
            GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
        }
        ASSERT_EQ(s, Status::Ok);
        EXPECT_EQ(ctx.node_init_and_sync(0), Status::NotInitialized);
        ctx.finalize();
    }
}

// --- DeploySyncNegativeArgs ---
// node_init_and_sync() argument validation: expand_batch bounds, node_id
// mismatch. Failure is decided via MPI_Allreduce consensus (singleton world
// here — each call is self-consistent).
TEST(Deploy, SyncNegativeArgs)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());
    {
        auto _lr = ctx.load(0, base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        ASSERT_EQ(_lr.status, Status::Ok);
    }

    // expand_batch < 1 → InvalidArg
    DeployOptions o_zero;
    o_zero.expand_batch = 0;
    EXPECT_EQ(ctx.node_init_and_sync(0, o_zero), Status::InvalidArg);

    // expand_batch > kMaxExpandBatch (64) → InvalidArg
    DeployOptions o_big;
    o_big.expand_batch = 65;
    EXPECT_EQ(ctx.node_init_and_sync(0, o_big), Status::InvalidArg);

    // node_id != rank → InvalidArg
    EXPECT_EQ(ctx.node_init_and_sync(1), Status::InvalidArg);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeploySearchEdgeCases ---
// search() boundary inputs: nq=0 (empty batch), k > ntotal (unfilled slots
// must stay id=-1 / dist=FLT_MAX).
TEST(Deploy, SearchEdgeCases)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    ASSERT_EQ(s, Status::Ok);

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());
    {
        auto _lr = ctx.load(0, base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        ASSERT_EQ(_lr.status, Status::Ok);
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);

    // nq=0 with null pointers → Ok, no-op batch
    SearchResult sr = ctx.search(nullptr, 0, SearchParams(5), nullptr, nullptr);
    EXPECT_EQ(sr.status, Status::Ok);

    // k > ntotal → Ok; unfilled slots id=-1, dist=FLT_MAX
    int32_t k = 60; // N == 50
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);
    float query[DEPLOY_DIM];
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = 0.5f;
    sr = ctx.search(query, 1, SearchParams(k), ids.data(), dists.data());
    EXPECT_EQ(sr.status, Status::Ok) << sr.error;

    int valid = 0;
    for (int i = 0; i < k; i++) {
        if (ids[i] >= 0) {
            valid++;
            EXPECT_LT(ids[i], static_cast<int32_t>(N)) << "result id out of range";
        } else {
            EXPECT_EQ(dists[i], std::numeric_limits<float>::max()) << "unfilled slot dist != FLT_MAX";
        }
    }
    // ef is bumped to k (60) >= N, and the 1D-line graph is fully reachable
    // from the entry point — expect (nearly) all N vectors returned.
    EXPECT_GE(valid, static_cast<int>(N) - 1);

    ctx.finalize();
    unlink(index_file_for_shard(base, 0).c_str());
}

// ============================================================
// Phase 2: Multi-rank MPI tests (gtest filter: *Mpi*)
// ============================================================

// --- DeployMpiTwoShardPushdownE2e ---
// 2 ranks, full pushdown flow: initialize → load → node_init_and_sync
// → search → finalize. Covers compact_graph_to_gd, open_gd_readonly,
// cross-shard validation, channel creation, proxy wiring, DualDistService,
// remote EP vector, multi-shard finalize cleanup.
TEST(DeployMpi, TwoShardPushdownE2e)
{
    static constexpr uint64_t N_TOTAL = 200;
    static constexpr uint64_t N_LOCAL = 100;
    static constexpr int32_t EP_GID = 50; // EP in shard 0

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    EXPECT_EQ(s, Status::Ok);

    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2) {
        GTEST_SKIP() << "Requires mpirun -np 2 (got " << comm_size << ")";
    }

    int rank = ctx.rank();
    EXPECT_EQ(rank >= 0 && rank < 2, true);

    // Build this rank's shard file
    auto buf = build_shard_buf(N_TOTAL, N_LOCAL, DEPLOY_DIM, DEPLOY_M, static_cast<uint32_t>(rank), 2,
                               static_cast<uint64_t>(rank) * N_LOCAL, EP_GID);
    std::string tmp_dir = ::testing::TempDir();
    std::string base = write_shard_file(tmp_dir, static_cast<uint32_t>(rank), buf);
    EXPECT_FALSE(base.empty());

    // load (keep_vectors=false since num_shards=2)
    LoadResult lr = ctx.load(static_cast<uint32_t>(rank), base);
    if (lr.status == Status::UbsemError)
        GTEST_SKIP() << "ubs-mem allocate failed: " << lr.error;
    EXPECT_EQ(lr.status, Status::Ok) << lr.error;
    EXPECT_EQ(lr.num_shards, 2u);
    EXPECT_EQ(lr.shard_id, static_cast<uint32_t>(rank));
    EXPECT_EQ(lr.ntotal, N_TOTAL);

    // node_init_and_sync — full pushdown path
    s = ctx.node_init_and_sync(static_cast<uint32_t>(rank));
    EXPECT_EQ(s, Status::Ok);
    EXPECT_EQ(ctx.num_shards(), 2u);
    EXPECT_EQ(ctx.dim(), DEPLOY_DIM);
    EXPECT_EQ(ctx.ntotal(), N_TOTAL);

    // Search — query near gid 100 (boundary of shard 0/1)
    float query[DEPLOY_DIM];
    float qval = 100.0f / static_cast<float>(N_TOTAL); // ≈ 0.5
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = qval;

    int32_t k = 5;
    int32_t ids[5];
    float dists[5];
    SearchParams sp(k);
    SearchResult sr = ctx.search(query, 1, sp, ids, dists);
    EXPECT_EQ(sr.status, Status::Ok) << sr.error;

    // All results should be non-empty
    for (int i = 0; i < k; i++) {
        EXPECT_GE(ids[i], 0) << "rank " << rank << " result " << i << " is empty";
    }

    // Distances should be sorted ascending
    for (int i = 1; i < k; i++) {
        EXPECT_LE(dists[i - 1], dists[i]) << "rank " << rank << " dist not sorted at " << i;
    }

    // Brute-force validation on full dataset
    auto buf0 = build_shard_buf(N_TOTAL, N_LOCAL, DEPLOY_DIM, DEPLOY_M, 0, 2, 0, EP_GID);
    auto buf1 = build_shard_buf(N_TOTAL, N_LOCAL, DEPLOY_DIM, DEPLOY_M, 1, 2, N_LOCAL, EP_GID);
    auto all_vecs = extract_all_vectors({&buf0, &buf1}, DEPLOY_DIM, N_TOTAL);
    std::vector<int32_t> bf_ids;
    std::vector<float> bf_dists;
    brute_force_knn(all_vecs.data(), N_TOTAL, DEPLOY_DIM, query, k, bf_ids, bf_dists);

    // Recall: at least k-1 out of k should match brute-force
    std::vector<int32_t> bf_set(bf_ids.begin(), bf_ids.end());
    std::sort(bf_set.begin(), bf_set.end());
    int match_count = 0;
    for (int i = 0; i < k; i++) {
        if (std::binary_search(bf_set.begin(), bf_set.end(), ids[i]))
            match_count++;
    }
    EXPECT_GE(match_count, k - 1) << "rank " << rank << " recall: " << match_count << "/" << k;

    s = ctx.finalize();
    EXPECT_EQ(s, Status::Ok);

    // Cleanup
    unlink(index_file_for_shard(base, static_cast<uint32_t>(rank)).c_str());
}

// --- DeployMpiDotNorm ---
// 2-shard pushdown with use_dot_norm=true: covers enable_dot_norm(true),
// DualDistService::enable_dot_norm(vector_norms_data()).
TEST(DeployMpi, DotNorm)
{
    static constexpr uint64_t N_TOTAL = 200;
    static constexpr uint64_t N_LOCAL = 100;
    static constexpr int32_t EP_GID = 50;

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    EXPECT_EQ(s, Status::Ok);

    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2) {
        GTEST_SKIP() << "Requires mpirun -np 2 (got " << comm_size << ")";
    }

    int rank = ctx.rank();

    auto buf = build_shard_buf(N_TOTAL, N_LOCAL, DEPLOY_DIM, DEPLOY_M, static_cast<uint32_t>(rank), 2,
                               static_cast<uint64_t>(rank) * N_LOCAL, EP_GID);
    std::string tmp_dir = ::testing::TempDir();
    std::string base = write_shard_file(tmp_dir, static_cast<uint32_t>(rank), buf);
    EXPECT_FALSE(base.empty());

    {
        auto _lr = ctx.load(static_cast<uint32_t>(rank), base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        EXPECT_EQ(_lr.status, Status::Ok);
    }

    // DeployOptions with use_dot_norm=true
    DeployOptions opts;
    opts.use_dot_norm = true;
    s = ctx.node_init_and_sync(static_cast<uint32_t>(rank), opts);
    EXPECT_EQ(s, Status::Ok);

    // Search
    float query[DEPLOY_DIM];
    float qval = 100.0f / static_cast<float>(N_TOTAL);
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = qval;

    int32_t k = 5;
    int32_t ids[5];
    float dists[5];
    SearchParams sp(k);
    SearchResult sr = ctx.search(query, 1, sp, ids, dists);
    EXPECT_EQ(sr.status, Status::Ok) << sr.error;

    for (int i = 0; i < k; i++) {
        EXPECT_GE(ids[i], 0) << "rank " << rank << " result " << i << " is empty";
    }
    for (int i = 1; i < k; i++) {
        EXPECT_LE(dists[i - 1], dists[i]) << "rank " << rank;
    }

    s = ctx.finalize();
    EXPECT_EQ(s, Status::Ok);

    unlink(index_file_for_shard(base, static_cast<uint32_t>(rank)).c_str());
}

// --- DeployMpiSyncValidation ---
// node_init_and_sync() failure consensus under MPI: a validation error on
// ANY rank must make ALL ranks return InvalidArg together (MPI_Allreduce
// MAX), so no rank proceeds while another fails — no deadlock, no hang.
//
// MPI deadlock rules: EXPECT only (never ASSERT before collective calls);
// both ranks always reach finalize() regardless of validation outcome.
TEST(DeployMpi, SyncValidation)
{
    static constexpr uint64_t N_TOTAL = 200;
    static constexpr uint64_t N_LOCAL = 100;
    static constexpr int32_t EP_GID = 50; // in shard 0's range [0, 100)

    Context ctx;
    Status s = ctx.initialize(nullptr, nullptr);
    if (s == Status::UbsemError) {
        GTEST_SKIP() << "ubs-mem runtime unavailable (daemon not running)";
    }
    EXPECT_EQ(s, Status::Ok);

    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2)
        GTEST_SKIP() << "Requires mpirun -np 2";

    int rank = ctx.rank();
    auto buf = build_shard_buf(N_TOTAL, N_LOCAL, DEPLOY_DIM, DEPLOY_M, static_cast<uint32_t>(rank), 2,
                               static_cast<uint64_t>(rank) * N_LOCAL, EP_GID);
    std::string base = write_shard_file(::testing::TempDir(), static_cast<uint32_t>(rank), buf);
    {
        auto _lr = ctx.load(static_cast<uint32_t>(rank), base);
        if (_lr.status == Status::UbsemError)
            GTEST_SKIP() << "ubs-mem allocate failed: " << _lr.error;
        EXPECT_EQ(_lr.status, Status::Ok);
    }

    // Sub-case A: both ranks pass bad expand_batch → both InvalidArg
    {
        DeployOptions bad;
        bad.expand_batch = 0;
        Status sa = ctx.node_init_and_sync(static_cast<uint32_t>(rank), bad);
        EXPECT_EQ(sa, Status::InvalidArg);
    }

    // Sub-case B: rank 1 passes wrong node_id → both InvalidArg via Allreduce
    {
        uint32_t node_id = (rank == 1) ? 0u : static_cast<uint32_t>(rank);
        Status sb = ctx.node_init_and_sync(node_id);
        EXPECT_EQ(sb, Status::InvalidArg);
    }

    EXPECT_EQ(ctx.finalize(), Status::Ok);
    unlink(index_file_for_shard(base, static_cast<uint32_t>(rank)).c_str());
}

// ============================================================
// Context::build forwarding (runs against stub AND real SDK — no
// ubs-mem interaction: build_sharded_index is a pure file-to-file call).
// ============================================================

TEST(Deploy, ContextBuild)
{
    // Tiny .bin dataset: [uint32 rows][uint32 cols] + row*col floats.
    // NOTE: without bin_base_path the loader resolves the sibling "base.bin"
    // next to dataset_path — pass the override so any filename works (and
    // exercise the opts.bin_base_path branch in build_sharded_index).
    std::string bin = ::testing::TempDir() + "/deploy_ctx_build.bin";
    constexpr uint32_t rows = 48;
    std::vector<float> data(static_cast<size_t>(rows) * DEPLOY_DIM);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = static_cast<float>((i * 37) % 251) / 251.0f;
    {
        FILE *fp = fopen(bin.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        ASSERT_EQ(fwrite(&rows, sizeof(rows), 1, fp), 1u);
        ASSERT_EQ(fwrite(&DEPLOY_DIM, sizeof(DEPLOY_DIM), 1, fp), 1u);
        ASSERT_EQ(fwrite(data.data(), sizeof(float), data.size(), fp), data.size());
        fclose(fp);
    }

    std::string out = ::testing::TempDir() + "/deploy_ctx_build_idx";
    BuildOptions o;
    o.dataset_path = bin;
    o.bin_base_path = bin; // override: file is not named base.bin
    o.output_path = out;
    o.num_shards = 2;
    o.M = 8;
    o.ef_construction = 40;
    o.ef_search = 32;
    o.build_threads = 2;

    Context ctx; // build() needs no initialize()
    BuildResult r = ctx.build(o);
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, rows);
    ASSERT_EQ(r.shard_bytes.size(), 2u);

    for (uint32_t s = 0; s < 2; s++)
        unlink(index_file_for_shard(out, s).c_str());
    unlink(bin.c_str());
}

// --- Deploy.InitWithOptions ---
// initialize() with InitOptions.num_threads > 0 (omp_set_num_threads path)
// plus accessor getters before load() and after finalize().
TEST(Deploy, InitWithOptions)
{
    InitOptions io;
    io.num_threads = 2;
    Context ctx;
    ASSERT_EQ(ctx.initialize(nullptr, nullptr, io), Status::Ok);
    EXPECT_TRUE(ctx.initialized());
    EXPECT_EQ(ctx.rank(), 0);
    EXPECT_EQ(ctx.size(), 1);
    EXPECT_EQ(ctx.num_shards(), 0u);
    EXPECT_EQ(ctx.dim(), 0u);
    EXPECT_EQ(ctx.ntotal(), 0u);
    EXPECT_FALSE(ctx.has_centroids());
    EXPECT_FALSE(ctx.has_idmap());
    EXPECT_EQ(ctx.last_status(), Status::Ok);
    EXPECT_EQ(ctx.finalize(), Status::Ok);
    EXPECT_FALSE(ctx.initialized());
}

#ifdef GD_HNSW_USE_UBSMEM_STUB
// ============================================================
// Failure-injection tests (stub build only — GD_HNSW_USE_UBSMEM_STUB)
//
// The POSIX-shm stub succeeds unconditionally, leaving the deploy layer's
// ubs-mem error branches (initialize/load/finalize failures) dead. The
// injection knobs (tests/ubs_mem_stub.cpp) flip individual API calls to
// error returns. Each test runs in its own process (one Context per run —
// MPI/ubs-mem are once-per-process resources).
// ============================================================

extern "C" void ubs_stub_inject(int fail_init_attributes, int fail_initialize, int fail_allocate, int fail_map,
                                int dealloc_inuse_times, int fail_deallocate);
extern "C" void ubs_stub_reset(void);
extern "C" int ubsmem_shmem_deallocate(const char *name);

// Reset injection on scope exit, including on ASSERT failure.
struct StubInjectGuard {
    ~StubInjectGuard() { ubs_stub_reset(); }
};

// --- DeployStubFailInitAttributes ---
// ubsmem_init_attributes fails → initialize() returns UbsemError; the
// follow-up finalize() exercises the partial-init cleanup (MPI finalized,
// ubs-mem never became ready).
TEST(Deploy, StubFailInitAttributes)
{
    Context ctx;
    ubs_stub_inject(/*init_attributes=*/1001, 0, 0, 0, 0, 0);
    EXPECT_EQ(ctx.initialize(nullptr, nullptr), Status::UbsemError);
    ubs_stub_reset();
    EXPECT_EQ(ctx.finalize(), Status::Ok);
}

// --- DeployStubFailInitialize ---
// ubsmem_initialize fails → initialize() returns UbsemError; finalize()
// still cleans up the already-initialized MPI.
TEST(Deploy, StubFailInitialize)
{
    Context ctx;
    ubs_stub_inject(0, /*initialize=*/1002, 0, 0, 0, 0);
    EXPECT_EQ(ctx.initialize(nullptr, nullptr), Status::UbsemError);
    ubs_stub_reset();
    EXPECT_EQ(ctx.finalize(), Status::Ok);
}

// --- DeployStubFailLoadAllocate ---
// GD allocate fails inside load_local_shard_direct_to_gd → load() returns
// UbsemError ("ubsmem_shmem_allocate_with_provider failed"); no GD object
// was created, so finalize() has nothing to clean.
TEST(Deploy, StubFailLoadAllocate)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    ASSERT_EQ(ctx.initialize(nullptr, nullptr), Status::Ok);
    StubInjectGuard guard;

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    ubs_stub_inject(0, 0, /*allocate=*/1003, 0, 0, 0);
    LoadResult lr = ctx.load(0, base);
    EXPECT_EQ(lr.status, Status::UbsemError);
    EXPECT_NE(lr.error.find("ubsmem_shmem_allocate_with_provider"), std::string::npos);

    ubs_stub_reset();
    EXPECT_EQ(ctx.finalize(), Status::Ok);
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployStubFailLoadMap ---
// GD allocate succeeds but the writable map fails → load() returns
// UbsemError ("ubsmem_shmem_map failed") and the error path deallocates
// the allocated GD object (no leftover in /dev/shm).
TEST(Deploy, StubFailLoadMap)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    ASSERT_EQ(ctx.initialize(nullptr, nullptr), Status::Ok);
    StubInjectGuard guard;

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());

    ubs_stub_inject(0, 0, 0, /*map=*/1004, 0, 0);
    LoadResult lr = ctx.load(0, base);
    EXPECT_EQ(lr.status, Status::UbsemError);
    EXPECT_NE(lr.error.find("ubsmem_shmem_map"), std::string::npos);

    ubs_stub_reset();
    EXPECT_EQ(ctx.finalize(), Status::Ok);
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployStubDeallocRetrySucceeds ---
// Full single-shard lifecycle, then the shard-GD deallocate reports
// UBSM_ERR_IN_USING twice before succeeding: exercises
// ubsmem_deallocate_retry's backoff loop (usleep + continue) and the
// retry-eventual-success path; finalize() still returns Ok.
TEST(Deploy, StubDeallocRetrySucceeds)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    ASSERT_EQ(ctx.initialize(nullptr, nullptr), Status::Ok);
    StubInjectGuard guard;

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());
    {
        auto _lr = ctx.load(0, base);
        ASSERT_EQ(_lr.status, Status::Ok) << _lr.error;
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);

    float q[DEPLOY_DIM];
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        q[d] = 0.5f;
    int32_t ids[4];
    float dists[4];
    EXPECT_EQ(ctx.search(q, 1, SearchParams(4), ids, dists).status, Status::Ok);

    ubs_stub_inject(0, 0, 0, 0, /*in_use times=*/2, 0);
    EXPECT_EQ(ctx.finalize(), Status::Ok);
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployStubDeallocRetryExhausted ---
// Deallocation stays IN_USE for longer than the retry budget →
// deallocate_retry exhausts its retries, finalize() prints the rank-0
// warning but still returns Ok. The leftover GD object is cleaned here
// (the "failed" deallocate never unlinked it).
TEST(Deploy, StubDeallocRetryExhausted)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    ASSERT_EQ(ctx.initialize(nullptr, nullptr), Status::Ok);
    StubInjectGuard guard;

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());
    {
        auto _lr = ctx.load(0, base);
        ASSERT_EQ(_lr.status, Status::Ok) << _lr.error;
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);

    ubs_stub_inject(0, 0, 0, 0, /*in_use times=*/99, 0);
    EXPECT_EQ(ctx.finalize(), Status::Ok);

    // job_id_ == getpid() (single rank); clean the leftover GD object.
    ubs_stub_reset();
    std::string gd_name = "gd_hnsw_" + std::to_string(static_cast<uint32_t>(getpid())) + "_0";
    ubsmem_shmem_deallocate(gd_name.c_str());
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployStubDeallocHardFail ---
// deallocate returns a hard (non-IN_USE) error → deallocate_retry returns
// immediately (no backoff) and finalize() prints the warning, returns Ok.
TEST(Deploy, StubDeallocHardFail)
{
    static constexpr uint64_t N = 50;

    Context ctx;
    ASSERT_EQ(ctx.initialize(nullptr, nullptr), Status::Ok);
    StubInjectGuard guard;

    auto buf = build_shard_buf(N, N, DEPLOY_DIM, DEPLOY_M, 0, 1, 0, 25);
    std::string base = write_shard_file(::testing::TempDir(), 0, buf);
    ASSERT_FALSE(base.empty());
    {
        auto _lr = ctx.load(0, base);
        ASSERT_EQ(_lr.status, Status::Ok) << _lr.error;
    }
    ASSERT_EQ(ctx.node_init_and_sync(0), Status::Ok);

    ubs_stub_inject(0, 0, 0, 0, 0, /*deallocate=*/1005);
    EXPECT_EQ(ctx.finalize(), Status::Ok);

    ubs_stub_reset();
    std::string gd_name = "gd_hnsw_" + std::to_string(static_cast<uint32_t>(getpid())) + "_0";
    ubsmem_shmem_deallocate(gd_name.c_str());
    unlink(index_file_for_shard(base, 0).c_str());
}

// --- DeployMpiStubDeallocWarning ---
// 2-rank pushdown lifecycle, then every deallocate hard-fails: covers the
// per-rank channel-deallocate warning ("[rank N] WARNING") and the rank-0
// shard-deallocate warning inside finalize(); finalize() still returns Ok.
TEST(DeployMpi, StubMpiDeallocWarning)
{
    static constexpr uint64_t N_TOTAL = 200;
    static constexpr uint64_t N_LOCAL = 100;
    static constexpr int32_t EP_GID = 50;

    Context ctx;
    EXPECT_EQ(ctx.initialize(nullptr, nullptr), Status::Ok);

    int comm_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2)
        GTEST_SKIP() << "Requires mpirun -np 2 (got " << comm_size << ")";

    int rank = ctx.rank();
    StubInjectGuard guard;

    // job_id_ == rank 0's pid (broadcast in initialize()); needed below to
    // clean the GD objects the "failed" deallocates leave behind.
    uint32_t jid = 0;
    if (rank == 0)
        jid = static_cast<uint32_t>(getpid());
    MPI_Bcast(&jid, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    auto buf = build_shard_buf(N_TOTAL, N_LOCAL, DEPLOY_DIM, DEPLOY_M, static_cast<uint32_t>(rank), 2,
                               static_cast<uint64_t>(rank) * N_LOCAL, EP_GID);
    std::string base = write_shard_file(::testing::TempDir(), static_cast<uint32_t>(rank), buf);
    EXPECT_FALSE(base.empty());
    {
        auto _lr = ctx.load(static_cast<uint32_t>(rank), base);
        EXPECT_EQ(_lr.status, Status::Ok) << _lr.error;
    }
    EXPECT_EQ(ctx.node_init_and_sync(static_cast<uint32_t>(rank)), Status::Ok);

    float query[DEPLOY_DIM];
    float qval = 100.0f / static_cast<float>(N_TOTAL);
    for (uint32_t d = 0; d < DEPLOY_DIM; d++)
        query[d] = qval;
    int32_t ids[5];
    float dists[5];
    EXPECT_EQ(ctx.search(query, 1, SearchParams(5), ids, dists).status, Status::Ok);

    // All channel + shard deallocates fail → warnings only, finalize Ok.
    ubs_stub_inject(0, 0, 0, 0, 0, /*deallocate=*/1005);
    EXPECT_EQ(ctx.finalize(), Status::Ok);

    // Clean leftovers: own task/result inboxes (+ rank 0: both shard GDs).
    ubs_stub_reset();
    std::string task_name = "gd_hnsw_task_" + std::to_string(jid) + "_" + std::to_string(rank);
    std::string result_name = "gd_hnsw_result_" + std::to_string(jid) + "_" + std::to_string(rank);
    ubsmem_shmem_deallocate(task_name.c_str());
    ubsmem_shmem_deallocate(result_name.c_str());
    if (rank == 0) {
        for (uint32_t s = 0; s < 2; s++) {
            std::string shard_name = "gd_hnsw_" + std::to_string(jid) + "_" + std::to_string(s);
            ubsmem_shmem_deallocate(shard_name.c_str());
        }
    }
    unlink(index_file_for_shard(base, static_cast<uint32_t>(rank)).c_str());
}
#endif // GD_HNSW_USE_UBSMEM_STUB
