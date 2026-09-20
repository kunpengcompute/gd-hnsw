/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include <gtest/gtest.h>

#if defined(__linux__) && defined(__aarch64__)

#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gd_hnsw_search.h"
#include "dist_service.h"

using namespace gd_hnsw;

static constexpr uint32_t FORK_DIM = 8;
static constexpr uint32_t FORK_M = 4;
static constexpr uint32_t FORK_MAX_BATCH = 2 * FORK_M;
static constexpr uint64_t FORK_RUN_GEN = 1;

// ============================================================
// Helpers
// ============================================================

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

    // Vectors: 1D line
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

    // Neighbors: 1D-line, ±1..±M (global IDs)
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

static ShardView make_shard_view(const std::vector<char> &buf)
{
    const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
    ShardView sv;
    sv.sb = sb;
    sv.vectors = reinterpret_cast<const float *>(buf.data() + sb->blocks[BLK_VECTORS].off);
    sv.levels = reinterpret_cast<const int32_t *>(buf.data() + sb->blocks[BLK_LEVELS].off);
    sv.offsets = reinterpret_cast<const uint64_t *>(buf.data() + sb->blocks[BLK_OFFSETS].off);
    sv.neighbors = reinterpret_cast<const int32_t *>(buf.data() + sb->blocks[BLK_NEIGHBORS].off);
    sv.cum_nneighbor = reinterpret_cast<const int32_t *>(buf.data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
    sv.global_id_begin = sb->global_id_begin;
    sv.ntotal_local = sb->ntotal_local;
    return sv;
}

// ============================================================
// Shared memory layout (mmap'd, visible to both processes):
//   [0..63]    stop flag (std::atomic<bool>, cacheline-aligned)
//   [64..]     task channel
//   [64+task]  result channel
// ============================================================

struct ForkGdLayout {
    TaskChannelLayout task_layout;
    ResultChannelLayout result_layout;
    size_t task_off;
    size_t result_off;
    size_t total;

    ForkGdLayout()
    {
        task_layout = TaskChannelLayout::build(FORK_MAX_BATCH, FORK_DIM);
        result_layout = ResultChannelLayout::build(FORK_MAX_BATCH);
        task_off = 64; // after stop flag cacheline
        result_off = task_off + task_layout.bytes_total;
        total = result_off + result_layout.bytes_total;
    }
};

// ============================================================
// ForkBasicPushdownRoundtrip
// Parent = searcher (owns shard 0), Child = service worker (shard 1)
// ============================================================

TEST(PushdownFork, ForkBasicPushdownRoundtrip)
{
    static constexpr uint64_t N = 100;
    static constexpr uint64_t N0 = 50;
    static constexpr uint64_t N1 = 50;
    static constexpr uint32_t NS = 2;

    ForkGdLayout gd_layout;

    // 1. Build shard buffers (heap memory, read-only after fork)
    auto b0 = build_shard_buf(N, N0, FORK_DIM, FORK_M, 0, NS, 0, 25);
    auto b1 = build_shard_buf(N, N1, FORK_DIM, FORK_M, 1, NS, N0, 25);
    ShardView sv1 = make_shard_view(b1);

    // 2. mmap shared memory
    void *gd = mmap(nullptr, gd_layout.total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(gd, MAP_FAILED) << "mmap failed";
    memset(gd, 0, gd_layout.total);

    auto *stop = new (gd) std::atomic<bool>(false);

    char *task_gd = static_cast<char *>(gd) + gd_layout.task_off;
    char *result_gd = static_cast<char *>(gd) + gd_layout.result_off;

    auto task_view = TaskChannelView::from_raw(task_gd, gd_layout.task_layout);
    auto result_view = ResultChannelView::from_raw(result_gd, gd_layout.result_layout);

    // Initialize channel headers
    task_view.h->channel_role = CHANNEL_ROLE_TASK;
    task_view.h->identity_valid = IDENTITY_VALID_MAGIC;
    task_view.h->max_batch = FORK_MAX_BATCH;
    task_view.h->dim = FORK_DIM;
    task_view.h->run_generation = FORK_RUN_GEN;
    task_view.h->req_seq.store(0, std::memory_order_relaxed);

    result_view.h->channel_role = CHANNEL_ROLE_RESULT;
    result_view.h->identity_valid = IDENTITY_VALID_MAGIC;
    result_view.h->max_batch = FORK_MAX_BATCH;
    result_view.h->dim = FORK_DIM;
    result_view.h->run_generation = FORK_RUN_GEN;
    result_view.h->resp_seq.store(0, std::memory_order_relaxed);

    // 3. Fork
    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        // === Child: Service Worker for shard 1 ===
        ServiceChannelPair pair{task_view, result_view};
        DualDistServiceWorker worker(1, sv1, FORK_DIM, FORK_RUN_GEN, {pair}, *stop);
        worker.run();
        _exit(0);
    }

    // === Parent: Searcher for shard 0 ===

    // Set up multi-shard searcher with pushdown proxy
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shard_ptrs, true);
    searcher.set_local_shard(0);

    // Load local vectors from buffer
    const auto *sb0 = reinterpret_cast<const SuperBlockV1 *>(b0.data());
    searcher.load_local_vectors(reinterpret_cast<const float *>(b0.data() + sb0->blocks[BLK_VECTORS].off));

    // Set up pushdown proxy for remote shard 1
    std::vector<std::vector<DualDistProxy>> proxies(1);
    proxies[0].resize(NS);
    proxies[0][1].init(task_view, result_view, FORK_DIM, MAX_SPIN_ITERS_DEFAULT, FORK_MAX_BATCH, 0, 0);
    searcher.set_dual_pushdown_proxies(&proxies, 1);

    // Search
    int32_t ids[5];
    float dists[5];
    float query[FORK_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < FORK_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    searcher.search_to(query, params, ids, dists);

    // Verify results
    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
    for (int i = 1; i < 5; i++)
        EXPECT_LE(dists[i - 1], dists[i]) << "distances not sorted at " << i;

    // Shutdown child
    stop->store(true, std::memory_order_relaxed);
    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status)) << "child terminated abnormally";
    EXPECT_EQ(WEXITSTATUS(status), 0);

    munmap(gd, gd_layout.total);
}

// ============================================================
// ForkWorkerIdentityValidation
// Corrupted identity fields → child worker must abort on first request
// ============================================================

TEST(PushdownFork, ForkWorkerIdentityValidation)
{
    static constexpr uint64_t N = 100;
    static constexpr uint64_t N0 = 50;
    static constexpr uint64_t N1 = 50;
    static constexpr uint32_t NS = 2;

    ForkGdLayout gd_layout;

    auto b0 = build_shard_buf(N, N0, FORK_DIM, FORK_M, 0, NS, 0, 25);
    auto b1 = build_shard_buf(N, N1, FORK_DIM, FORK_M, 1, NS, N0, 25);
    ShardView sv1 = make_shard_view(b1);

    void *gd = mmap(nullptr, gd_layout.total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(gd, MAP_FAILED);
    memset(gd, 0, gd_layout.total);

    auto *stop = new (gd) std::atomic<bool>(false);

    char *task_gd = static_cast<char *>(gd) + gd_layout.task_off;
    char *result_gd = static_cast<char *>(gd) + gd_layout.result_off;

    auto task_view = TaskChannelView::from_raw(task_gd, gd_layout.task_layout);
    auto result_view = ResultChannelView::from_raw(result_gd, gd_layout.result_layout);

    // Set up channels but corrupt task identity
    task_view.h->channel_role = CHANNEL_ROLE_TASK;
    task_view.h->identity_valid = 0; // INVALID — will trigger abort in worker
    task_view.h->max_batch = FORK_MAX_BATCH;
    task_view.h->dim = FORK_DIM;
    task_view.h->run_generation = FORK_RUN_GEN;
    task_view.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    result_view.h->channel_role = CHANNEL_ROLE_RESULT;
    result_view.h->identity_valid = IDENTITY_VALID_MAGIC;
    result_view.h->max_batch = FORK_MAX_BATCH;
    result_view.h->dim = FORK_DIM;
    result_view.h->run_generation = FORK_RUN_GEN;
    result_view.h->resp_seq.store(0, std::memory_order_relaxed);

    pid_t pid = fork();
    ASSERT_GE(pid, 0);

    if (pid == 0) {
        ServiceChannelPair pair{task_view, result_view};
        DualDistServiceWorker worker(1, sv1, FORK_DIM, FORK_RUN_GEN, {pair}, *stop);
        worker.run(); // should abort on identity check
        _exit(1);     // if we reach here, abort didn't fire
    }

    int status = 0;
    waitpid(pid, &status, 0);
    // Child should have been killed by SIGABRT
    EXPECT_FALSE(WIFEXITED(status)) << "worker should have aborted, but exited normally with code "
                                    << WEXITSTATUS(status);
    EXPECT_TRUE(WIFSIGNALED(status)) << "worker should have been killed by signal (SIGABRT)";
    EXPECT_EQ(WTERMSIG(status), SIGABRT);

    munmap(gd, gd_layout.total);
}

// ============================================================
// ForkRunGenerationMismatch
// Mismatched run_generation → worker skips the request
// ============================================================

TEST(PushdownFork, ForkRunGenerationMismatch)
{
    static constexpr uint64_t N = 100;
    static constexpr uint64_t N0 = 50;
    static constexpr uint64_t N1 = 50;
    static constexpr uint32_t NS = 2;

    ForkGdLayout gd_layout;

    auto b0 = build_shard_buf(N, N0, FORK_DIM, FORK_M, 0, NS, 0, 25);
    auto b1 = build_shard_buf(N, N1, FORK_DIM, FORK_M, 1, NS, N0, 25);
    ShardView sv1 = make_shard_view(b1);

    void *gd = mmap(nullptr, gd_layout.total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(gd, MAP_FAILED);
    memset(gd, 0, gd_layout.total);

    auto *stop = new (gd) std::atomic<bool>(false);

    char *task_gd = static_cast<char *>(gd) + gd_layout.task_off;
    char *result_gd = static_cast<char *>(gd) + gd_layout.result_off;

    auto task_view = TaskChannelView::from_raw(task_gd, gd_layout.task_layout);
    auto result_view = ResultChannelView::from_raw(result_gd, gd_layout.result_layout);

    task_view.h->channel_role = CHANNEL_ROLE_TASK;
    task_view.h->identity_valid = IDENTITY_VALID_MAGIC;
    task_view.h->max_batch = FORK_MAX_BATCH;
    task_view.h->dim = FORK_DIM;
    task_view.h->run_generation = 999; // WRONG gen — worker uses FORK_RUN_GEN=1
    task_view.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    result_view.h->channel_role = CHANNEL_ROLE_RESULT;
    result_view.h->identity_valid = IDENTITY_VALID_MAGIC;
    result_view.h->max_batch = FORK_MAX_BATCH;
    result_view.h->dim = FORK_DIM;
    result_view.h->run_generation = 999;
    result_view.h->resp_seq.store(0, std::memory_order_relaxed);

    pid_t pid = fork();
    ASSERT_GE(pid, 0);

    if (pid == 0) {
        ServiceChannelPair pair{task_view, result_view};
        // Worker has run_generation=1, channel has run_generation=999
        // → run() must skip the request at the gen check and spin until stop.
        DualDistServiceWorker worker(1, sv1, FORK_DIM, FORK_RUN_GEN, {pair}, *stop);
        worker.run();
        _exit(0);
    }

    // Parent: give the worker ample time to (wrongly) process the request if
    // the run_generation check in run() were missing — any real processing
    // here completes in the first spin iteration, so 200ms is generous.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // THE assertion: the request must have been skipped, so resp_seq is
    // untouched. If the gen check is deleted from run(), process_request()
    // runs and resp_seq becomes the doorbell seq (1) → this fails.
    EXPECT_EQ(result_view.h->resp_seq.load(std::memory_order_acquire), 0u)
        << "gen-mismatched request was processed (resp_seq advanced)";

    stop->store(true, std::memory_order_relaxed);

    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status)) << "child should have exited normally";
    EXPECT_EQ(WEXITSTATUS(status), 0);

    munmap(gd, gd_layout.total);
}

#else // !defined(__linux__) || !defined(__aarch64__)

// Dummy test for non-Linux or non-ARM64 platforms (avoids "no tests" error)
TEST(PushdownFork, ForkTestsRequireLinuxAndARM64) { GTEST_SKIP() << "Fork-based pushdown tests require Linux + ARM64"; }

#endif // defined(__linux__) && defined(__aarch64__)
