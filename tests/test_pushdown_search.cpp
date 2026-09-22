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

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <omp.h>

#include "gd_hnsw_search.h"
#include "dist_service.h"
#include "distance_kernel_dot_norm.h"
#include "test_distance_common.hpp"

#include <cstdio>
#include <cerrno>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <unistd.h>

// ============================================================
// gcov flush on abort: EXPECT_DEATH forks the process; the child
// calls abort() which bypasses gcov's atexit handler. We:
// 1. Use "threadsafe" death test style (fork without exec) so the
//    child inherits the parent's gcov runtime state.
// 2. Install a SIGABRT handler that flushes gcov counters then
//    _exit(1) — gcov merges (max) the child's counts into .gcda.
// ============================================================
extern "C" void __gcov_dump(void) __attribute__((weak));

static void gcov_flush_on_abort(int /*sig*/)
{
    if (&__gcov_dump)
        __gcov_dump();
    _exit(1);
}

class GcovDeathTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override
    {
        ::testing::FLAGS_gtest_death_test_style = "threadsafe";
        signal(SIGABRT, gcov_flush_on_abort);
    }
};

static auto *g_gcov_env = ::testing::AddGlobalTestEnvironment(new GcovDeathTestEnvironment);

using namespace gd_hnsw;

// ============================================================
// Pushdown test constants
// ============================================================

static constexpr uint32_t PD_DIM = 8;
static constexpr uint32_t PD_M = 4;
static constexpr uint32_t PD_MAX_BATCH = 2 * PD_M;
static constexpr uint64_t PD_RUN_GEN = 1;

// ============================================================
// Ground truth: brute-force top-K by global ID
// ============================================================

static std::vector<int32_t> brute_force_topk_global(const float *query, const float *vectors, uint64_t n, uint32_t dim,
                                                    int k)
{
    std::vector<std::pair<float, int32_t>> all(n);
    for (uint64_t i = 0; i < n; i++) {
        float d = scalar::l2_sqr(query, vectors + i * dim, dim);
        all[i] = {d, static_cast<int32_t>(i)};
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    std::vector<int32_t> ids(k);
    for (int i = 0; i < k; i++)
        ids[i] = all[i].second;
    return ids;
}

static double compute_recall(const int32_t *result, const std::vector<int32_t> &gt, int k)
{
    int hits = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (result[i] == gt[j]) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

// ============================================================
// Channel pair helper (same pattern as test_concurrency.cpp)
// ============================================================

struct PushdownChannelPair {
    std::vector<char> task_buf;
    std::vector<char> result_buf;
    TaskChannelView task;
    ResultChannelView result;
    TaskChannelLayout task_layout;
    ResultChannelLayout result_layout;

    PushdownChannelPair(uint32_t max_batch = PD_MAX_BATCH, uint32_t dim = PD_DIM)
    {
        task_layout = TaskChannelLayout::build(max_batch, dim);
        result_layout = ResultChannelLayout::build(max_batch);

        task_buf.resize(task_layout.bytes_total, 0);
        result_buf.resize(result_layout.bytes_total, 0);

        task = TaskChannelView::from_raw(task_buf.data(), task_layout);
        result = ResultChannelView::from_raw(result_buf.data(), result_layout);

        task.h->channel_role = CHANNEL_ROLE_TASK;
        task.h->identity_valid = IDENTITY_VALID_MAGIC;
        task.h->max_batch = max_batch;
        task.h->dim = dim;
        task.h->run_generation = PD_RUN_GEN;
        task.h->req_seq.store(0, std::memory_order_relaxed);

        result.h->channel_role = CHANNEL_ROLE_RESULT;
        result.h->identity_valid = IDENTITY_VALID_MAGIC;
        result.h->max_batch = max_batch;
        result.h->dim = dim;
        result.h->run_generation = PD_RUN_GEN;
        result.h->resp_seq.store(0, std::memory_order_relaxed);
    }
};

// ============================================================
// Multi-shard pushdown graph fixture
// Builds 2 shards with 1D-line neighbors spanning shard boundaries.
// Shard 0 is local (vectors loaded to heap), shard 1 is remote.
// ============================================================

struct PushdownFixture {
    static constexpr uint64_t N = 100;
    static constexpr uint64_t N_PER_SHARD = N / 2; // 50
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr uint32_t NUM_LEVELS = 2; // max_level=0
    static constexpr uint32_t NN_PER_NODE = 2 * PD_M;

    // Shard buffers (in-memory GD)
    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    // Full vector set for ground truth + local shard heap load
    std::vector<float> all_vectors;

    // Channel + proxy + worker for remote shard
    PushdownChannelPair channel;
    std::atomic<bool> stop{false};
    std::unique_ptr<DualDistServiceWorker> worker;
    std::thread worker_thread;

    // 2D proxy array for searcher
    std::vector<std::vector<DualDistProxy>> proxies_2d;

    // Searcher
    std::unique_ptr<GdHnswSearcher> searcher;

    PushdownFixture(int32_t entry_point = 25) : proxies_2d(1) // 1 search thread
    {
        omp_set_num_threads(1); // single-threaded for deterministic testing
        proxies_2d[0].resize(NUM_SHARDS);

        // --- Build shard buffers ---
        uint32_t nlevels = NUM_LEVELS;

        for (uint32_t s = 0; s < NUM_SHARDS; s++) {
            SuperBlockV1 sb{};
            sb.magic = GD_MAGIC;
            sb.schema_version = SCHEMA_VERSION;
            sb.state = static_cast<uint32_t>(GdState::ACTIVE);
            sb.metric_type = METRIC_L2;
            sb.endianness = ENDIAN_LITTLE;
            sb.ntotal = N;
            sb.ntotal_local = N_PER_SHARD;
            sb.dim = PD_DIM;
            sb.M = PD_M;
            sb.ef_search = 32;
            sb.max_level = 0;
            sb.entry_point = entry_point;
            sb.num_levels = nlevels;
            sb.num_shards = NUM_SHARDS;
            sb.shard_id = s;
            sb.global_id_begin = s * N_PER_SHARD;
            sb.neighbors_size = N_PER_SHARD * NN_PER_NODE;
            compute_layout(sb);

            shard_buf[s].resize(sb.total_bytes, 0);
            std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
            sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
        }

        // --- Fill per-shard data ---
        for (uint32_t s = 0; s < NUM_SHARDS; s++) {
            uint64_t gid_begin = s * N_PER_SHARD;
            auto *sb = sb_mut[s];

            // Vectors: 1D line (each global ID maps to a value along a line)
            float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
            for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
                uint64_t gid = gid_begin + lid;
                float val = static_cast<float>(gid) / static_cast<float>(N);
                for (uint32_t d = 0; d < PD_DIM; d++)
                    vecs[lid * PD_DIM + d] = val;
            }

            // Levels: all nodes at level 1 (= faiss level 0)
            int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
            for (uint64_t lid = 0; lid < N_PER_SHARD; lid++)
                levels[lid] = 1;

            // Offsets: each node has NN_PER_NODE neighbors at layer 0
            uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
            for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++)
                offsets[lid] = lid * NN_PER_NODE;

            // Neighbors: 1D-line, connect to ±1..±M (global IDs)
            int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
            for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
                int64_t gid = static_cast<int64_t>(gid_begin + lid);
                int32_t *nb = neighbors + lid * NN_PER_NODE;
                for (uint32_t j = 0; j < NN_PER_NODE; j++)
                    nb[j] = EMPTY_NEIGHBOR;
                for (uint32_t r = 0; r < PD_M; r++) {
                    int64_t left = gid - 1 - static_cast<int64_t>(r);
                    int64_t right = gid + 1 + static_cast<int64_t>(r);
                    if (left >= 0)
                        nb[2 * r] = static_cast<int32_t>(left);
                    if (right < static_cast<int64_t>(N))
                        nb[2 * r + 1] = static_cast<int32_t>(right);
                }
            }

            // cum_nneighbor
            int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
            cum_nn[0] = 0;                                 // layer > 0: no nodes
            cum_nn[1] = static_cast<int32_t>(NN_PER_NODE); // layer 0

            // CRCs
            sb->header_crc64 = compute_header_crc64(*sb);
            sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

            // Build ShardView
            shard_view[s].sb = sb;
            shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
            shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
            shard_view[s].offsets =
                reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
            shard_view[s].neighbors =
                reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
            shard_view[s].cum_nneighbor =
                reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
            shard_view[s].global_id_begin = gid_begin;
            shard_view[s].ntotal_local = N_PER_SHARD;
        }

        // Build full vector set for ground truth
        all_vectors.resize(N * PD_DIM);
        for (uint32_t s = 0; s < NUM_SHARDS; s++) {
            uint64_t gid_begin = s * N_PER_SHARD;
            const float *src = shard_view[s].vectors;
            std::memcpy(all_vectors.data() + gid_begin * PD_DIM, src, N_PER_SHARD * PD_DIM * sizeof(float));
        }

        // --- Set up channels, proxy, worker for remote shard ---
        // Local shard = 0, remote shard = 1

        // Init proxy for remote shard (shard 1)
        proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, PD_MAX_BATCH, 0, 0);

        // Start service worker for shard 1
        ServiceChannelPair pair{channel.task, channel.result};
        worker.reset(new DualDistServiceWorker(1, shard_view[1], PD_DIM, PD_RUN_GEN,
                                               std::vector<ServiceChannelPair>{pair}, stop));
        worker_thread = std::thread([this]() { worker->run(); });

        // --- Build searcher ---
        std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                     {shard_buf[1].data(), shard_buf[1].size()}};
        searcher.reset(new GdHnswSearcher(shard_ptrs, true));
        searcher->set_local_shard(0);
        searcher->load_local_vectors(
            reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
        searcher->set_dual_pushdown_proxies(&proxies_2d, 1);
    }

    ~PushdownFixture()
    {
        stop.store(true, std::memory_order_relaxed);
        if (worker_thread.joinable())
            worker_thread.join();
    }

    // Non-copyable
    PushdownFixture(const PushdownFixture &) = delete;
    PushdownFixture &operator=(const PushdownFixture &) = delete;
};

// ============================================================
// B1: Basic pushdown layer-0 search
// ============================================================

TEST(PushdownSearch, BasicSearchReturnsResults)
{
    PushdownFixture fx;

    int32_t k = 10;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    // Query near the boundary (gid=50) to force cross-shard traffic
    float query[PD_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    // Should return k valid results
    for (int i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
}

TEST(PushdownSearch, ResultsSortedByDistance)
{
    PushdownFixture fx;

    int32_t k = 10;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.35f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 1; i < k && ids[i] >= 0; i++)
        EXPECT_LE(dists[i - 1], dists[i] + 1e-5f);
}

TEST(PushdownSearch, SearchReturnsCorrectTopK)
{
    PushdownFixture fx;

    int32_t k = 5;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.30f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.95) << "Recall@" << k << " = " << recall;
}

TEST(PushdownSearch, QueryInRemoteShardRegion)
{
    // Query near gid=75 (shard 1) - forces classification of remote neighbors
    PushdownFixture fx;

    int32_t k = 5;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.75f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.95) << "Recall@" << k << " = " << recall;
}

// ============================================================
// B1: Pushdown search determinism
// ============================================================

TEST(PushdownSearch, Deterministic)
{
    PushdownFixture fx;

    int32_t k = 5;
    float query[PD_DIM];
    float qval = 0.45f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    std::vector<int32_t> ids1(k), ids2(k);
    std::vector<float> dists1(k), dists2(k);

    fx.searcher->search_to(query, params, ids1.data(), dists1.data());
    fx.searcher->search_to(query, params, ids2.data(), dists2.data());

    for (int i = 0; i < k; i++) {
        EXPECT_EQ(ids1[i], ids2[i]);
        EXPECT_FLOAT_EQ(dists1[i], dists2[i]);
    }
}

// ============================================================
// B1: Edge cases
// ============================================================

TEST(PushdownSearch, K1ReturnsNearestNeighbor)
{
    PushdownFixture fx;

    int32_t id;
    float dist;
    float query[PD_DIM];
    float qval = 0.20f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 1;
    params.ef_search = 16;

    fx.searcher->search_to(query, params, &id, &dist);
    EXPECT_GE(id, 0);
    EXPECT_GE(dist, 0.0f);

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, 1);
    EXPECT_EQ(id, gt[0]);
}

TEST(PushdownSearch, EfSearchLessThanK)
{
    PushdownFixture fx;

    int32_t k = 10;
    std::vector<int32_t> ids(k, -1);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.60f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 3; // less than k

    fx.searcher->search_to(query, params, ids.data(), dists.data());
    for (int i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
}

// ============================================================
// B1: Expand batch > 1 (multi-pop path)
// ============================================================

TEST(PushdownSearch, ExpandBatch4)
{
    PushdownFixture fx;
    fx.searcher->set_expand_batch(4);
    EXPECT_EQ(fx.searcher->expand_batch(), 4u);

    int32_t k = 10;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.95) << "Recall@" << k << " with expand_batch=4: " << recall;
}

// ============================================================
// B2: Multi-layer graph for greedy_search_upper_pushdown
// ============================================================

struct MultiLevelFixture {
    static constexpr uint64_t N = 60;
    static constexpr uint64_t N_PER_SHARD = N / 2; // 30
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr int32_t MAX_LEVEL = 2;
    static constexpr uint32_t NUM_LEVEL_SLOTS = 4; // MAX_LEVEL + 2
    static constexpr uint32_t NN_PER_NODE = 2 * PD_M;

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];
    std::vector<float> all_vectors;

    PushdownChannelPair channel;
    std::atomic<bool> stop{false};
    std::unique_ptr<DualDistServiceWorker> worker;
    std::thread worker_thread;

    std::vector<std::vector<DualDistProxy>> proxies_2d;
    std::unique_ptr<GdHnswSearcher> searcher;

    MultiLevelFixture() : proxies_2d(1)
    {
        omp_set_num_threads(1); // single-threaded for deterministic testing
        proxies_2d[0].resize(NUM_SHARDS);

        // Build shard buffers with max_level=2
        for (uint32_t s = 0; s < NUM_SHARDS; s++) {
            SuperBlockV1 sb{};
            sb.magic = GD_MAGIC;
            sb.schema_version = SCHEMA_VERSION;
            sb.state = static_cast<uint32_t>(GdState::ACTIVE);
            sb.metric_type = METRIC_L2;
            sb.endianness = ENDIAN_LITTLE;
            sb.ntotal = N;
            sb.ntotal_local = N_PER_SHARD;
            sb.dim = PD_DIM;
            sb.M = PD_M;
            sb.ef_search = 32;
            sb.max_level = MAX_LEVEL;
            sb.entry_point = 15; // in shard 0
            sb.num_levels = NUM_LEVEL_SLOTS;
            sb.num_shards = NUM_SHARDS;
            sb.shard_id = s;
            sb.global_id_begin = s * N_PER_SHARD;
            sb.neighbors_size = N_PER_SHARD * static_cast<uint64_t>(MAX_LEVEL + 1) * NN_PER_NODE;
            compute_layout(sb);

            shard_buf[s].resize(sb.total_bytes, 0);
            std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
            sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
        }

        for (uint32_t s = 0; s < NUM_SHARDS; s++) {
            uint64_t gid_begin = s * N_PER_SHARD;
            auto *sb = sb_mut[s];

            // Vectors
            float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
            for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
                uint64_t gid = gid_begin + lid;
                float val = static_cast<float>(gid) / static_cast<float>(N);
                for (uint32_t d = 0; d < PD_DIM; d++)
                    vecs[lid * PD_DIM + d] = val;
            }

            // Levels: stored level = faiss level + 1 (matching TestGraph convention)
            int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
            for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
                int64_t gid = static_cast<int64_t>(gid_begin + lid);
                if (gid == 15)
                    levels[lid] = 3; // faiss level 2 (top)
                else if (gid == 10 || gid == 20 || gid == 35 || gid == 45)
                    levels[lid] = 3;
                else if (gid == 5 || gid == 8 || gid == 12 || gid == 18 || gid == 22 || gid == 28 || gid == 32 ||
                         gid == 38 || gid == 42 || gid == 48 || gid == 52 || gid == 55)
                    levels[lid] = 2; // faiss level 1
                else
                    levels[lid] = 1; // faiss level 0
            }

            // Offsets[i] = start of node i's neighbor data (cumulative)
            uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
            uint64_t cum = 0;
            for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++) {
                offsets[lid] = cum;
                if (lid < N_PER_SHARD) {
                    int faiss_lv = levels[lid] - 1; // stored → faiss level
                    cum += static_cast<uint64_t>(faiss_lv + 1) * NN_PER_NODE;
                }
            }

            // Neighbors: populate each node's neighbors at its level and below
            int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
            // Fill all with EMPTY
            std::fill(neighbors, neighbors + offsets[N_PER_SHARD], EMPTY_NEIGHBOR);

            for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
                int64_t gid = static_cast<int64_t>(gid_begin + lid);
                int faiss_lv = levels[lid] - 1;
                uint64_t base = offsets[lid];

                // For each layer from faiss_lv down to 0
                for (int lv = faiss_lv; lv >= 0; lv--) {
                    uint64_t layer_base = base + static_cast<uint64_t>(lv) * NN_PER_NODE;
                    int32_t *nb = neighbors + layer_base;

                    // Connect to nearest neighbors at this level
                    int step = 1 << lv; // wider spacing at higher levels
                    int idx = 0;
                    for (uint32_t r = 0; r < PD_M; r++) {
                        int64_t left = gid - step * (1 + static_cast<int64_t>(r));
                        int64_t right = gid + step * (1 + static_cast<int64_t>(r));
                        if (left >= 0 && idx < static_cast<int>(NN_PER_NODE))
                            nb[idx++] = static_cast<int32_t>(left);
                        if (right < static_cast<int64_t>(N) && idx < static_cast<int>(NN_PER_NODE))
                            nb[idx++] = static_cast<int32_t>(right);
                    }
                }
            }

            // cum_nneighbor[layer] = cumulative neighbors below `layer`
            // NUM_LEVEL_SLOTS entries (0..MAX_LEVEL+1)
            int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
            for (uint32_t i = 0; i < NUM_LEVEL_SLOTS; i++)
                cum_nn[i] = static_cast<int32_t>(i * NN_PER_NODE);

            // CRCs
            sb->header_crc64 = compute_header_crc64(*sb);
            sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

            // ShardView
            shard_view[s].sb = sb;
            shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
            shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
            shard_view[s].offsets =
                reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
            shard_view[s].neighbors =
                reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
            shard_view[s].cum_nneighbor =
                reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
            shard_view[s].global_id_begin = gid_begin;
            shard_view[s].ntotal_local = N_PER_SHARD;
        }

        // Full vectors
        all_vectors.resize(N * PD_DIM);
        for (uint32_t s = 0; s < NUM_SHARDS; s++) {
            uint64_t gid_begin = s * N_PER_SHARD;
            std::memcpy(all_vectors.data() + gid_begin * PD_DIM, shard_view[s].vectors,
                        N_PER_SHARD * PD_DIM * sizeof(float));
        }

        // Channels, proxy, worker
        proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, PD_MAX_BATCH, 0, 0);
        ServiceChannelPair pair{channel.task, channel.result};
        worker.reset(new DualDistServiceWorker(1, shard_view[1], PD_DIM, PD_RUN_GEN,
                                               std::vector<ServiceChannelPair>{pair}, stop));
        worker_thread = std::thread([this]() { worker->run(); });

        // Searcher
        std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                     {shard_buf[1].data(), shard_buf[1].size()}};
        searcher.reset(new GdHnswSearcher(shard_ptrs, true));
        searcher->set_local_shard(0);
        searcher->load_local_vectors(
            reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
        searcher->set_dual_pushdown_proxies(&proxies_2d, 1);
    }

    ~MultiLevelFixture()
    {
        stop.store(true, std::memory_order_relaxed);
        if (worker_thread.joinable())
            worker_thread.join();
    }

    MultiLevelFixture(const MultiLevelFixture &) = delete;
    MultiLevelFixture &operator=(const MultiLevelFixture &) = delete;
};

TEST(PushdownSearch, MultiLevelGreedyUpperPushdown)
{
    MultiLevelFixture fx;

    int32_t k = 5;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.90) << "Recall@" << k << " = " << recall;
}

// ============================================================
// B1: Entry point on remote shard (test ep_vector_cache path)
// ============================================================

TEST(PushdownSearch, EntryPointOnRemoteShard)
{
    // Build with EP in shard 1 (gid=75)
    PushdownFixture fx(75);

    // Provide EP vector from the remote shard
    const float *ep_vec = fx.shard_view[1].vectors + (75 - fx.shard_view[1].global_id_begin) * PD_DIM;
    fx.searcher->set_ep_vector(ep_vec, PD_DIM);

    int32_t k = 5;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.90) << "Recall@" << k << " = " << recall;
}

// ============================================================
// B1: Non-uniform shards
// ============================================================

TEST(PushdownSearch, NonUniformShards)
{
    // Shard 0: 30 nodes, Shard 1: 70 nodes
    static constexpr uint64_t N_NU = 100;
    static constexpr uint64_t N0 = 30;
    static constexpr uint64_t N1 = 70;

    auto make_buf = [](uint64_t ntotal, uint64_t nlocal, uint32_t shard_id, uint64_t gid_begin) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = ntotal;
        sb.ntotal_local = nlocal;
        sb.dim = PD_DIM;
        sb.M = PD_M;
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = 15;
        sb.num_levels = 2;
        sb.num_shards = 2;
        sb.shard_id = shard_id;
        sb.global_id_begin = gid_begin;
        sb.neighbors_size = nlocal * 2 * PD_M;
        compute_layout(sb);

        std::vector<char> buf(sb.total_bytes, 0);
        std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));
        auto *sbp = reinterpret_cast<SuperBlockV1 *>(buf.data());

        // Vectors
        float *vecs = reinterpret_cast<float *>(buf.data() + sb.blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < nlocal; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(ntotal);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        // Levels
        int32_t *levels = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < nlocal; lid++)
            levels[lid] = 1;

        // Offsets
        uint64_t *offsets = reinterpret_cast<uint64_t *>(buf.data() + sb.blocks[BLK_OFFSETS].off);
        uint64_t nn = 2 * PD_M;
        for (uint64_t lid = 0; lid <= nlocal; lid++)
            offsets[lid] = lid * nn;

        // Neighbors
        int32_t *neighbors = reinterpret_cast<int32_t *>(buf.data() + sb.blocks[BLK_NEIGHBORS].off);
        for (uint64_t lid = 0; lid < nlocal; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int32_t *nb = neighbors + lid * nn;
            for (uint64_t j = 0; j < nn; j++)
                nb[j] = EMPTY_NEIGHBOR;
            for (uint32_t r = 0; r < PD_M; r++) {
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
    };

    auto b0 = make_buf(N_NU, N0, 0, 0);
    auto b1 = make_buf(N_NU, N1, 1, N0);

    // Build ShardView for remote shard
    const auto *sb1 = reinterpret_cast<const SuperBlockV1 *>(b1.data());
    ShardView sv1;
    sv1.sb = sb1;
    sv1.vectors = reinterpret_cast<const float *>(b1.data() + sb1->blocks[BLK_VECTORS].off);
    sv1.levels = reinterpret_cast<const int32_t *>(b1.data() + sb1->blocks[BLK_LEVELS].off);
    sv1.offsets = reinterpret_cast<const uint64_t *>(b1.data() + sb1->blocks[BLK_OFFSETS].off);
    sv1.neighbors = reinterpret_cast<const int32_t *>(b1.data() + sb1->blocks[BLK_NEIGHBORS].off);
    sv1.cum_nneighbor = reinterpret_cast<const int32_t *>(b1.data() + sb1->blocks[BLK_CUM_NNEIGHBOR].off);
    sv1.global_id_begin = N0;
    sv1.ntotal_local = N1;

    // Channels + worker
    PushdownChannelPair ch;
    std::atomic<bool> stop_nu{false};
    std::vector<std::vector<DualDistProxy>> proxies(1);
    proxies[0].resize(2);
    proxies[0][1].init(ch.task, ch.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, PD_MAX_BATCH, 0, 0);

    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(1, sv1, PD_DIM, PD_RUN_GEN, std::vector<ServiceChannelPair>{pair}, stop_nu);
    std::thread wt([&]() { worker.run(); });

    // Searcher
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher sr(shard_ptrs, true);
    sr.set_local_shard(0);
    const auto *sb0 = reinterpret_cast<const SuperBlockV1 *>(b0.data());
    sr.load_local_vectors(reinterpret_cast<const float *>(b0.data() + sb0->blocks[BLK_VECTORS].off));
    sr.set_dual_pushdown_proxies(&proxies, 1);

    // Full vectors for ground truth
    std::vector<float> all_vecs(N_NU * PD_DIM);
    {
        const float *v0 = reinterpret_cast<const float *>(b0.data() + sb0->blocks[BLK_VECTORS].off);
        std::memcpy(all_vecs.data(), v0, N0 * PD_DIM * sizeof(float));
        std::memcpy(all_vecs.data() + N0 * PD_DIM, sv1.vectors, N1 * PD_DIM * sizeof(float));
    }

    int32_t k = 5;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    sr.search_to(query, params, ids.data(), dists.data());

    auto gt = brute_force_topk_global(query, all_vecs.data(), N_NU, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.90) << "Recall@" << k << " = " << recall;

    stop_nu.store(true, std::memory_order_relaxed);
    wt.join();
}

// ============================================================
// Error path: multi-shard without proxies aborts
// ============================================================

static std::vector<char> make_minimal_shard_buf(uint64_t ntotal, uint64_t nlocal, uint32_t dim, uint32_t M,
                                                uint32_t shard_id, uint32_t num_shards, uint64_t gid_begin)
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
    sb.ef_search = 16;
    sb.max_level = 0;
    sb.entry_point = 0;
    sb.num_levels = 2;
    sb.num_shards = num_shards;
    sb.shard_id = shard_id;
    sb.global_id_begin = gid_begin;
    sb.neighbors_size = nlocal * M * 2;
    compute_layout(sb);
    std::vector<char> buf(sb.total_bytes, 0);
    std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));
    return buf;
}

TEST(PushdownSearch, MultiShardWithoutProxiesAborts)
{
    auto b0 = make_minimal_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_minimal_shard_buf(100, 50, 2, 4, 1, 2, 50);
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher sr(shard_ptrs, true);
    sr.set_local_shard(0);

    int32_t ids[5];
    float dists[5];
    float query[] = {0.5f, 0.5f};
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 8;

    EXPECT_DEATH(sr.search_to(query, params, ids, dists), "pushdown proxies");
}

// ============================================================
// Phase G1a: DotNorm + Pushdown combination tests
// ============================================================

// Fixture that extends PushdownFixture with dot_norm enabled on both
// searcher (local shard) and worker (remote shard).
struct DotNormPushdownFixture : public PushdownFixture {
    std::vector<float> remote_norms; // norms for remote shard vectors

    DotNormPushdownFixture(int32_t entry_point = 25) : PushdownFixture(entry_point)
    {
        // Enable dot_norm on searcher (computes norms for local shard 0)
        searcher->enable_dot_norm(true);

        // Compute norms for remote shard 1 and inject into worker
        auto &sv = shard_view[1];
        remote_norms.resize(sv.ntotal_local);
        for (uint64_t i = 0; i < sv.ntotal_local; i++) {
            remote_norms[i] = vec_norm_sq(sv.vectors + i * PD_DIM, PD_DIM);
        }
        worker->enable_dot_norm(remote_norms.data());
    }
};

TEST(PushdownSearch, DotNormBasicSearch)
{
    DotNormPushdownFixture fx;

    int32_t k = 10;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);

    float query[PD_DIM];
    float qval = 0.50f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    // DotNorm results must still produce valid sorted distances
    for (int i = 1; i < k; i++)
        EXPECT_LE(dists[i - 1], dists[i]) << "distances not sorted at index " << i;

    // Recall vs ground truth (L2 brute force — dot_norm approximates L2
    // so recall may be slightly lower but should still be reasonable)
    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, k);
    double recall = compute_recall(ids.data(), gt, k);
    EXPECT_GE(recall, 0.85) << "DotNorm Recall@" << k << " = " << recall;
}

TEST(PushdownSearch, DotNormPushdownRecall)
{
    DotNormPushdownFixture fx;

    int32_t ids[5];
    float dists[5];

    float query[PD_DIM];
    float qval = 0.30f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids, dists);

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.N, PD_DIM, 5);
    double recall = compute_recall(ids, gt, 5);
    EXPECT_GE(recall, 0.85) << "DotNormPushdown Recall@5 = " << recall;
}

// ============================================================
// Phase G3: File I/O path tests
// ============================================================

TEST(PushdownSearch, LoadLocalVectorsFromFile)
{
    // Build a minimal shard buffer and write to a temp file
    auto buf = make_minimal_shard_buf(100, 50, /*dim=*/2, /*M=*/4,
                                      /*shard_id=*/0, /*num_shards=*/1,
                                      /*gid_begin=*/0);

    // Write to temp file
    std::string tmpfile = "/tmp/test_gd_hnsw_load_vecs.bin";
    FILE *fp = fopen(tmpfile.c_str(), "wb");
    ASSERT_NE(fp, nullptr) << "Cannot create temp file: " << strerror(errno);
    size_t written = fwrite(buf.data(), 1, buf.size(), fp);
    fclose(fp);
    ASSERT_EQ(written, buf.size());

    // Load via file path
    GdHnswSearcher searcher(buf.data(), buf.size(), true);
    searcher.set_local_shard(0);
    EXPECT_NO_THROW(searcher.load_local_vectors(tmpfile));

    // Clean up
    std::remove(tmpfile.c_str());
}

TEST(PushdownSearch, LoadLocalVectorsFileNotFoundThrows)
{
    auto buf = make_minimal_shard_buf(100, 50, 2, 4, 0, 1, 0);
    GdHnswSearcher searcher(buf.data(), buf.size(), true);
    searcher.set_local_shard(0);
    EXPECT_THROW(searcher.load_local_vectors("/nonexistent/path_hsnw_test.bin"), std::runtime_error);
}

TEST(PushdownSearch, LoadLocalVectorsTruncatedFileThrows)
{
    auto buf = make_minimal_shard_buf(100, 50, 2, 4, 0, 1, 0);

    // Write only SuperBlock header — all data blocks (vectors/levels/etc.)
    // are absent, so fread on vectors will fail with short read.
    std::string tmpfile = "/tmp/test_gd_hnsw_truncated.bin";
    FILE *fp = fopen(tmpfile.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    size_t hdr_size = sizeof(SuperBlockV1);
    ASSERT_GT(hdr_size, 0u);
    fwrite(buf.data(), 1, hdr_size, fp);
    fclose(fp);

    GdHnswSearcher searcher(buf.data(), buf.size(), true);
    searcher.set_local_shard(0);
    EXPECT_THROW(searcher.load_local_vectors(tmpfile), std::runtime_error);

    std::remove(tmpfile.c_str());
}

TEST(PushdownSearch, LoadLocalVectorsCantReadSuperBlock)
{
    auto buf = make_minimal_shard_buf(100, 50, 2, 4, 0, 1, 0);

    // Write fewer bytes than sizeof(SuperBlockV1) — fread returns 0 records
    std::string tmpfile = "/tmp/test_gd_hnsw_short_hdr.bin";
    FILE *fp = fopen(tmpfile.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    char dummy[16] = {};
    fwrite(dummy, 1, sizeof(dummy), fp);
    fclose(fp);

    GdHnswSearcher searcher(buf.data(), buf.size(), true);
    searcher.set_local_shard(0);
    EXPECT_THROW(searcher.load_local_vectors(tmpfile), std::runtime_error);

    std::remove(tmpfile.c_str());
}

TEST(PushdownSearch, LoadLocalVectorsEmptyVectorsBlock)
{
    auto buf = make_minimal_shard_buf(100, 50, 2, 4, 0, 1, 0);

    // Write a valid SuperBlock but with BLK_VECTORS.bytes zeroed out
    SuperBlockV1 sb_copy;
    std::memcpy(&sb_copy, buf.data(), sizeof(SuperBlockV1));
    sb_copy.blocks[BLK_VECTORS].bytes = 0;

    std::string tmpfile = "/tmp/test_gd_hnsw_empty_vec.bin";
    FILE *fp = fopen(tmpfile.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&sb_copy, sizeof(SuperBlockV1), 1, fp);
    fclose(fp);

    GdHnswSearcher searcher(buf.data(), buf.size(), true);
    searcher.set_local_shard(0);
    EXPECT_THROW(searcher.load_local_vectors(tmpfile), std::runtime_error);

    std::remove(tmpfile.c_str());
}

// ============================================================
// Phase G5: Phase 1 scalar tail + dot_norm tail paths
// Uses M=2 so edge nodes have 3 valid neighbours (non-4-multiple)
// triggering the NEON scalar fallback and dot_norm tail paths.
// ============================================================

struct TailCoverageFixture {
    static constexpr uint32_t TC_DIM = 4;
    static constexpr uint32_t TC_M = 2;
    static constexpr uint32_t TC_MAX_BATCH = 4; // 2*M
    static constexpr uint64_t TC_N = 40;
    static constexpr uint64_t TC_N_PER = 20;
    static constexpr uint32_t TC_NS = 2;
    static constexpr uint64_t TC_RUN_GEN = 1;

    std::vector<char> shard_buf[2];
    ShardView shard_view[2];
    std::vector<float> all_vectors;

    PushdownChannelPair channel{TC_MAX_BATCH, TC_DIM};
    std::atomic<bool> stop{false};
    std::unique_ptr<DualDistServiceWorker> worker;
    std::thread worker_thread;
    std::vector<std::vector<DualDistProxy>> proxies_2d;
    std::unique_ptr<GdHnswSearcher> searcher;

    TailCoverageFixture() : proxies_2d(1)
    {
        omp_set_num_threads(1);
        proxies_2d[0].resize(TC_NS);

        for (uint32_t s = 0; s < TC_NS; s++) {
            SuperBlockV1 sb{};
            sb.magic = GD_MAGIC;
            sb.schema_version = SCHEMA_VERSION;
            sb.state = static_cast<uint32_t>(GdState::ACTIVE);
            sb.metric_type = METRIC_L2;
            sb.endianness = ENDIAN_LITTLE;
            sb.ntotal = TC_N;
            sb.ntotal_local = TC_N_PER;
            sb.dim = TC_DIM;
            sb.M = TC_M;
            sb.ef_search = 32;
            sb.max_level = 0;
            sb.entry_point = 10; // middle of shard 0
            sb.num_levels = 2;
            sb.num_shards = TC_NS;
            sb.shard_id = s;
            sb.global_id_begin = s * TC_N_PER;
            sb.neighbors_size = TC_N_PER * TC_MAX_BATCH;
            compute_layout(sb);

            shard_buf[s].resize(sb.total_bytes, 0);
            std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
            auto *sbp = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());

            // Vectors: 1D line
            float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sbp->blocks[BLK_VECTORS].off);
            for (uint64_t lid = 0; lid < TC_N_PER; lid++) {
                uint64_t gid = sbp->global_id_begin + lid;
                float val = static_cast<float>(gid) / static_cast<float>(TC_N);
                for (uint32_t d = 0; d < TC_DIM; d++)
                    vecs[lid * TC_DIM + d] = val;
            }

            // Levels: all at layer 1
            int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_LEVELS].off);
            for (uint64_t lid = 0; lid < TC_N_PER; lid++)
                levels[lid] = 1;

            // Offsets
            uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sbp->blocks[BLK_OFFSETS].off);
            for (uint64_t lid = 0; lid <= TC_N_PER; lid++)
                offsets[lid] = lid * TC_MAX_BATCH;

            // Neighbors: 1D line, ±1..±M, EMPTY for out-of-range
            int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
            for (uint64_t lid = 0; lid < TC_N_PER; lid++) {
                int64_t gid = static_cast<int64_t>(sbp->global_id_begin + lid);
                int32_t *nb = neighbors + lid * TC_MAX_BATCH;
                for (uint32_t j = 0; j < TC_MAX_BATCH; j++)
                    nb[j] = EMPTY_NEIGHBOR;
                for (uint32_t r = 0; r < TC_M; r++) {
                    int64_t left = gid - 1 - static_cast<int64_t>(r);
                    int64_t right = gid + 1 + static_cast<int64_t>(r);
                    if (left >= 0)
                        nb[2 * r] = static_cast<int32_t>(left);
                    if (right < static_cast<int64_t>(TC_N))
                        nb[2 * r + 1] = static_cast<int32_t>(right);
                }
            }

            // cum_nneighbor
            int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
            cum_nn[0] = 0;
            cum_nn[1] = static_cast<int32_t>(TC_MAX_BATCH);

            sbp->header_crc64 = compute_header_crc64(*sbp);
            sbp->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sbp);

            shard_view[s].sb = sbp;
            shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sbp->blocks[BLK_VECTORS].off);
            shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_LEVELS].off);
            shard_view[s].offsets =
                reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sbp->blocks[BLK_OFFSETS].off);
            shard_view[s].neighbors =
                reinterpret_cast<const int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
            shard_view[s].cum_nneighbor =
                reinterpret_cast<const int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
            shard_view[s].global_id_begin = sbp->global_id_begin;
            shard_view[s].ntotal_local = TC_N_PER;
        }

        // Full vectors for ground truth
        all_vectors.resize(TC_N * TC_DIM);
        for (uint32_t s = 0; s < TC_NS; s++) {
            uint64_t gb = s * TC_N_PER;
            std::memcpy(all_vectors.data() + gb * TC_DIM, shard_view[s].vectors, TC_N_PER * TC_DIM * sizeof(float));
        }

        proxies_2d[0][1].init(channel.task, channel.result, TC_DIM, MAX_SPIN_ITERS_DEFAULT, TC_MAX_BATCH, 0, 0);

        ServiceChannelPair pair{channel.task, channel.result};
        worker.reset(new DualDistServiceWorker(1, shard_view[1], TC_DIM, TC_RUN_GEN,
                                               std::vector<ServiceChannelPair>{pair}, stop));
        worker_thread = std::thread([this]() { worker->run(); });

        std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                     {shard_buf[1].data(), shard_buf[1].size()}};
        searcher.reset(new GdHnswSearcher(shard_ptrs, true));
        searcher->set_local_shard(0);
        const auto *sb0 = reinterpret_cast<const SuperBlockV1 *>(shard_buf[0].data());
        searcher->load_local_vectors(
            reinterpret_cast<const float *>(shard_buf[0].data() + sb0->blocks[BLK_VECTORS].off));
        searcher->set_dual_pushdown_proxies(&proxies_2d, 1);
    }

    ~TailCoverageFixture()
    {
        stop.store(true, std::memory_order_relaxed);
        if (worker_thread.joinable())
            worker_thread.join();
    }
};

TEST(PushdownSearch, Phase1ScalarTailWithEdgeNodes)
{
    // M=2 → edge nodes (gid=1) have 3 valid neighbours
    // NEON batch detects EMPTY and falls through to scalar tail.
    TailCoverageFixture fx;

    int32_t ids[5];
    float dists[5];

    // Query near gid=1 (edge of shard 0) forces traversal through edge nodes
    float query[4];
    float qval = 0.02f; // near gid=1 (1/40 ≈ 0.025)
    for (uint32_t d = 0; d < 4; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    // Verify basic correctness
    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.TC_N, 4, 5);
    double recall = compute_recall(ids, gt, 5);
    EXPECT_GE(recall, 0.80) << "Recall@5 = " << recall;
}

TEST(PushdownSearch, DotNormTailWithNonMultipleOf4)
{
    // Same M=2 graph with dot_norm enabled.
    // Edge nodes produce non-4-multiple local_ids → dot_norm tail paths.
    TailCoverageFixture fx;

    // Enable dot_norm on searcher
    fx.searcher->enable_dot_norm(true);

    // Compute norms for remote shard and inject into worker
    auto &sv = fx.shard_view[1];
    std::vector<float> remote_norms(sv.ntotal_local);
    for (uint64_t i = 0; i < sv.ntotal_local; i++)
        remote_norms[i] = vec_norm_sq(sv.vectors + i * 4, 4);
    fx.worker->enable_dot_norm(remote_norms.data());

    int32_t ids[5];
    float dists[5];

    float query[4];
    float qval = 0.02f; // near gid=1 — edge node with 3 valid neighbours
    for (uint32_t d = 0; d < 4; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    fx.searcher->search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0);
    for (int i = 1; i < 5; i++)
        EXPECT_LE(dists[i - 1], dists[i]);

    auto gt = brute_force_topk_global(query, fx.all_vectors.data(), fx.TC_N, 4, 5);
    double recall = compute_recall(ids, gt, 5);
    EXPECT_GE(recall, 0.80) << "DotNorm Recall@5 = " << recall;
}

// ============================================================
// Phase H: Final coverage push to 90%
// ============================================================

// --- Power-of-2 shard pushdown: covers find_shard shift path (gid >> shard_shift_) ---
// 64 = 2^6 → set_local_shard triggers power-of-2 detection, search_to uses shift.

TEST(PushdownSearch, PowerOfTwoShardPushdown)
{
    constexpr uint32_t DIM = 4;
    constexpr uint32_t M = 2;
    constexpr uint64_t N = 128;
    constexpr uint64_t N_PER = 64; // power of 2 → shard_shift_valid_
    constexpr uint32_t NS = 2;
    constexpr uint32_t NN = 2 * M;

    // Build 2 shard buffers with full graph data
    std::vector<char> bufs[2];
    ShardView sv[2];
    for (uint32_t s = 0; s < NS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER;
        sb.dim = DIM;
        sb.M = M;
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = 32;
        sb.num_levels = 2;
        sb.num_shards = NS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER;
        sb.neighbors_size = N_PER * NN;
        compute_layout(sb);
        bufs[s].resize(sb.total_bytes, 0);
        std::memcpy(bufs[s].data(), &sb, sizeof(SuperBlockV1));
        auto *sbp = reinterpret_cast<SuperBlockV1 *>(bufs[s].data());

        float *vecs = reinterpret_cast<float *>(bufs[s].data() + sbp->blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < N_PER; i++) {
            float val = static_cast<float>(s * N_PER + i) / static_cast<float>(N);
            for (uint32_t d = 0; d < DIM; d++)
                vecs[i * DIM + d] = val;
        }

        int32_t *lvls = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < N_PER; i++)
            lvls[i] = 1;

        uint64_t *offs = reinterpret_cast<uint64_t *>(bufs[s].data() + sbp->blocks[BLK_OFFSETS].off);
        for (uint64_t i = 0; i <= N_PER; i++)
            offs[i] = i * NN;

        int32_t *nbs = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < N_PER; i++) {
            int64_t gid = static_cast<int64_t>(s * N_PER + i);
            int32_t *nb = nbs + i * NN;
            for (uint32_t j = 0; j < NN; j++)
                nb[j] = EMPTY_NEIGHBOR;
            for (uint32_t r = 0; r < M; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0)
                    nb[2 * r] = static_cast<int32_t>(left);
                if (right < (int64_t)N)
                    nb[2 * r + 1] = static_cast<int32_t>(right);
            }
        }

        int32_t *cum = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
        cum[0] = 0;
        cum[1] = static_cast<int32_t>(NN);

        sv[s].sb = sbp;
        sv[s].vectors = vecs;
        sv[s].levels = lvls;
        sv[s].offsets = offs;
        sv[s].neighbors = nbs;
        sv[s].cum_nneighbor = cum;
        sv[s].global_id_begin = sbp->global_id_begin;
        sv[s].ntotal_local = N_PER;
    }

    // Pushdown channel + worker for remote shard 1
    PushdownChannelPair channel{NN, DIM};
    std::atomic<bool> stop{false};
    ServiceChannelPair pair{channel.task, channel.result};
    DualDistServiceWorker worker(1, sv[1], DIM, 1, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    // Proxy
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NS);
    proxies_2d[0][1].init(channel.task, channel.result, DIM, MAX_SPIN_ITERS_DEFAULT, NN, 0, 0);

    // Searcher
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{bufs[0].data(), bufs[0].size()},
                                                                 {bufs[1].data(), bufs[1].size()}};
    GdHnswSearcher searcher(shard_ptrs, true);
    searcher.set_local_shard(0); // power-of-2 detection: N_PER=64 → shard_shift_valid_=true
    searcher.load_local_vectors(reinterpret_cast<const float *>(
        bufs[0].data() + reinterpret_cast<SuperBlockV1 *>(bufs[0].data())->blocks[BLK_VECTORS].off));
    searcher.set_dual_pushdown_proxies(&proxies_2d, 1);

    // Search → find_shard uses gid >> shard_shift_ (line 354)
    float query[DIM];
    for (uint32_t d = 0; d < DIM; d++)
        query[d] = 0.5f;
    int32_t ids[5];
    float dists[5];
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;
    searcher.search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0);
    for (int i = 1; i < 5; i++)
        EXPECT_LE(dists[i - 1], dists[i]);

    stop.store(true);
    worker_thread.join();
}

// --- Non-4-multiple dimension dot_norm: covers scalar tail ---

TEST(PushdownSearch, DotNormNonMultipleOf4Dim)
{
    constexpr uint32_t D7 = 7; // not a multiple of 4 → scalar tail in dot_norm_dist
    constexpr uint32_t M = 2;
    constexpr uint64_t N = 20;
    constexpr uint64_t N_PER = N;

    auto buf = make_minimal_shard_buf(N, N_PER, D7, M, 0, 1, 0);

    // Fill vectors with deterministic data
    auto *sbp = reinterpret_cast<SuperBlockV1 *>(buf.data());
    // Entry point must be a mid-graph node: boundary nodes have a leading
    // EMPTY_NEIGHBOR hole (no left neighbor) and the layer-0 scan breaks at
    // the first EMPTY, so entry 0 would never expand.
    sbp->entry_point = 10;
    float *vecs = reinterpret_cast<float *>(buf.data() + sbp->blocks[BLK_VECTORS].off);
    for (uint64_t i = 0; i < N; i++) {
        float val = static_cast<float>(i) / static_cast<float>(N);
        for (uint32_t d = 0; d < D7; d++)
            vecs[i * D7 + d] = val;
    }

    // Levels
    int32_t *levels = reinterpret_cast<int32_t *>(buf.data() + sbp->blocks[BLK_LEVELS].off);
    for (uint64_t i = 0; i < N; i++)
        levels[i] = 1;

    // Offsets
    uint64_t *offsets = reinterpret_cast<uint64_t *>(buf.data() + sbp->blocks[BLK_OFFSETS].off);
    uint32_t nn = 2 * M;
    for (uint64_t i = 0; i <= N; i++)
        offsets[i] = i * nn;

    // Neighbors: 1D line ±1..±M
    int32_t *neighbors = reinterpret_cast<int32_t *>(buf.data() + sbp->blocks[BLK_NEIGHBORS].off);
    for (uint64_t i = 0; i < N; i++) {
        int64_t gid = static_cast<int64_t>(i);
        int32_t *nb = neighbors + i * nn;
        for (uint32_t j = 0; j < nn; j++)
            nb[j] = EMPTY_NEIGHBOR;
        for (uint32_t r = 0; r < M; r++) {
            int64_t left = gid - 1 - static_cast<int64_t>(r);
            int64_t right = gid + 1 + static_cast<int64_t>(r);
            if (left >= 0)
                nb[2 * r] = static_cast<int32_t>(left);
            if (right < (int64_t)N)
                nb[2 * r + 1] = static_cast<int32_t>(right);
        }
    }

    // cum_nneighbor
    int32_t *cum_nn = reinterpret_cast<int32_t *>(buf.data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
    cum_nn[0] = 0;
    cum_nn[1] = static_cast<int32_t>(nn);

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{buf.data(), buf.size()}};
    GdHnswSearcher searcher(shard_ptrs, true); // skip CRC validation
    searcher.set_local_shard(0);

    // Load vectors from buffer
    searcher.load_local_vectors(reinterpret_cast<const float *>(buf.data() + sbp->blocks[BLK_VECTORS].off));

    // Enable dot_norm → covers dot_norm_dist with DIM=7 (scalar tail)
    searcher.enable_dot_norm(true);

    float query[D7];
    for (uint32_t d = 0; d < D7; d++)
        query[d] = 0.5f;

    int32_t ids[5];
    float dists[5];
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 16;

    searcher.search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0);
    for (int i = 1; i < 5; i++)
        EXPECT_LE(dists[i - 1], dists[i]);
}

// --- Multi-shard file-based load_local_vectors: covers nullptr assignment ---

TEST(PushdownSearch, MultiShardFileLoadLocalVectors)
{
    constexpr uint32_t DIM = 4;
    constexpr uint32_t M = 2;
    constexpr uint64_t N = 40;
    constexpr uint64_t N_PER = 20;

    // Build 2 shard buffers with full data
    std::vector<char> bufs[2];
    for (uint32_t s = 0; s < 2; s++) {
        bufs[s] = make_minimal_shard_buf(N, N_PER, DIM, M, s, 2, s * N_PER);
        auto *sbp = reinterpret_cast<SuperBlockV1 *>(bufs[s].data());

        float *vecs = reinterpret_cast<float *>(bufs[s].data() + sbp->blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < N_PER; i++) {
            float val = static_cast<float>(s * N_PER + i) / static_cast<float>(N);
            for (uint32_t d = 0; d < DIM; d++)
                vecs[i * DIM + d] = val;
        }

        int32_t *lvls = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < N_PER; i++)
            lvls[i] = 1;

        uint64_t *offs = reinterpret_cast<uint64_t *>(bufs[s].data() + sbp->blocks[BLK_OFFSETS].off);
        uint32_t nn = 2 * M;
        for (uint64_t i = 0; i <= N_PER; i++)
            offs[i] = i * nn;

        int32_t *nbs = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < N_PER; i++) {
            int64_t gid = static_cast<int64_t>(s * N_PER + i);
            int32_t *nb = nbs + i * nn;
            for (uint32_t j = 0; j < nn; j++)
                nb[j] = EMPTY_NEIGHBOR;
            for (uint32_t r = 0; r < M; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0)
                    nb[2 * r] = static_cast<int32_t>(left);
                if (right < (int64_t)N)
                    nb[2 * r + 1] = static_cast<int32_t>(right);
            }
        }

        int32_t *cum = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
        cum[0] = 0;
        cum[1] = static_cast<int32_t>(nn);
    }

    // Write shard files to tmp
    std::string path0 = "/tmp/test_gd_hnsw_multishard_0.bin";
    std::string path1 = "/tmp/test_gd_hnsw_multishard_1.bin";

    {
        FILE *f = fopen(path0.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        fwrite(bufs[0].data(), 1, bufs[0].size(), f);
        fclose(f);
    }
    {
        FILE *f = fopen(path1.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        fwrite(bufs[1].data(), 1, bufs[1].size(), f);
        fclose(f);
    }

    // Load searcher with both shards (in-memory)
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{bufs[0].data(), bufs[0].size()},
                                                                 {bufs[1].data(), bufs[1].size()}};
    GdHnswSearcher searcher(shard_ptrs, true); // skip CRC validation
    searcher.set_local_shard(0);

    // File-based load → triggers nullptr assignment for remote shard (line 301)
    searcher.load_local_vectors(path0);

    // Cleanup
    std::remove(path0.c_str());
    std::remove(path1.c_str());
}

// --- dot_norm_dist scalar tail: dim=7 leaves 3 elements after NEON 4-batches ---

TEST(PushdownSearch, DotNormDistScalarTailDirect)
{
    constexpr uint32_t D = 7;
    float q[D], v[D];
    for (uint32_t i = 0; i < D; i++) {
        q[i] = 0.5f * static_cast<float>(i + 1);
        v[i] = 0.25f * static_cast<float>(i + 2);
    }
    float qn = 0.0f, vn = 0.0f, dot = 0.0f;
    for (uint32_t i = 0; i < D; i++) {
        qn += q[i] * q[i];
        vn += v[i] * v[i];
        dot += q[i] * v[i];
    }
    float expect = qn + vn - 2.0f * dot;
    if (expect < 0.0f)
        expect = 0.0f;

    float got = dot_norm_dist(q, v, D, qn, vn);
    EXPECT_NEAR(got, expect, 1e-4f);
    EXPECT_NEAR(got, scalar::l2_sqr(q, v, D), 1e-4f);
}

// --- MinimaxHeap capacity governance: cand_heap_ shrink after outlier query ---

TEST(PushdownSearch, MinimaxHeapCandHeapShrink)
{
    MinimaxHeap heap(1); // shrink_trigger = 32, retain_cap = 8
    // Each push improves on the current best, so every entry enters cand_heap_
    // (never popped) — both entries_ and cand_heap_ grow past 32 * n.
    for (int i = 0; i < 100; i++) {
        heap.push(i, 1000.0f - static_cast<float>(i));
    }
    EXPECT_EQ(heap.size(), 1);
    heap.clear(); // triggers shrink_to_fit + reserve on entries_ and cand_heap_
    EXPECT_EQ(heap.size(), 0);
    EXPECT_EQ(heap.candidates(), 0);

    // Heap still usable after shrink
    heap.push(7, 1.0f);
    auto [d, id] = heap.pop_min_candidate();
    EXPECT_EQ(id, 7);
    EXPECT_FLOAT_EQ(d, 1.0f);
}

// --- Non-uniform shard sizes: covers binary-search shard_of lambda ---

TEST(PushdownSearch, NonUniformShardPushdown)
{
    constexpr uint32_t DIM = 4;
    constexpr uint32_t M = 2;
    constexpr uint64_t N = 40;
    constexpr uint32_t NS = 2;
    constexpr uint32_t NN = 2 * M;
    constexpr uint64_t N_LOCAL[NS] = {24, 16}; // unequal → uniform_shards_ = false
    constexpr uint64_t GID_BEGIN[NS] = {0, 24};

    std::vector<char> bufs[NS];
    ShardView sv[NS];
    for (uint32_t s = 0; s < NS; s++) {
        const uint64_t nl = N_LOCAL[s];
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = nl;
        sb.dim = DIM;
        sb.M = M;
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = 30; // in shard 1 → remote for local_shard 0
        sb.num_levels = 2;
        sb.num_shards = NS;
        sb.shard_id = s;
        sb.global_id_begin = GID_BEGIN[s];
        sb.neighbors_size = nl * NN;
        compute_layout(sb);
        bufs[s].resize(sb.total_bytes, 0);
        std::memcpy(bufs[s].data(), &sb, sizeof(SuperBlockV1));
        auto *sbp = reinterpret_cast<SuperBlockV1 *>(bufs[s].data());

        float *vecs = reinterpret_cast<float *>(bufs[s].data() + sbp->blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < nl; i++) {
            float val = static_cast<float>(GID_BEGIN[s] + i) / static_cast<float>(N);
            for (uint32_t d = 0; d < DIM; d++)
                vecs[i * DIM + d] = val;
        }

        int32_t *lvls = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < nl; i++)
            lvls[i] = 1;

        uint64_t *offs = reinterpret_cast<uint64_t *>(bufs[s].data() + sbp->blocks[BLK_OFFSETS].off);
        for (uint64_t i = 0; i <= nl; i++)
            offs[i] = i * NN;

        int32_t *nbs = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < nl; i++) {
            int64_t gid = static_cast<int64_t>(GID_BEGIN[s] + i);
            int32_t *nb = nbs + i * NN;
            for (uint32_t j = 0; j < NN; j++)
                nb[j] = EMPTY_NEIGHBOR;
            int idx = 0;
            for (uint32_t r = 0; r < M; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0)
                    nb[idx++] = static_cast<int32_t>(left);
                if (right < (int64_t)N)
                    nb[idx++] = static_cast<int32_t>(right);
            }
        }

        int32_t *cum = reinterpret_cast<int32_t *>(bufs[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
        cum[0] = 0;
        cum[1] = static_cast<int32_t>(NN);

        sv[s].sb = sbp;
        sv[s].vectors = vecs;
        sv[s].levels = lvls;
        sv[s].offsets = offs;
        sv[s].neighbors = nbs;
        sv[s].cum_nneighbor = cum;
        sv[s].global_id_begin = sbp->global_id_begin;
        sv[s].ntotal_local = nl;
    }

    // Pushdown channel + worker for remote shard 1
    PushdownChannelPair channel{NN, DIM};
    std::atomic<bool> stop{false};
    ServiceChannelPair pair{channel.task, channel.result};
    DualDistServiceWorker worker(1, sv[1], DIM, 1, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NS);
    proxies_2d[0][1].init(channel.task, channel.result, DIM, MAX_SPIN_ITERS_DEFAULT, NN, 0, 0);

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{bufs[0].data(), bufs[0].size()},
                                                                 {bufs[1].data(), bufs[1].size()}};
    GdHnswSearcher searcher(shard_ptrs, true);
    searcher.set_local_shard(0);
    searcher.load_local_vectors(reinterpret_cast<const float *>(
        bufs[0].data() + reinterpret_cast<SuperBlockV1 *>(bufs[0].data())->blocks[BLK_VECTORS].off));
    // EP 30 lives on remote shard 1 whose vectors were nullptr'd by
    // load_local_vectors — provide its vector explicitly (production pattern).
    float ep_vec[DIM];
    for (uint32_t d = 0; d < DIM; d++)
        ep_vec[d] = 30.0f / static_cast<float>(N);
    searcher.set_ep_vector(ep_vec, DIM);
    searcher.set_dual_pushdown_proxies(&proxies_2d, 1);

    float query[DIM];
    for (uint32_t d = 0; d < DIM; d++)
        query[d] = 0.5f;
    int32_t ids[5];
    float dists[5];
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    // Non-profiled search → shard_of lambda takes binary-search branch
    searcher.search_to(query, params, ids, dists);
    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0);
    for (int i = 1; i < 5; i++)
        EXPECT_LE(dists[i - 1], dists[i]);
    EXPECT_EQ(ids[0], 20); // val 20/40 = 0.5 exactly matches query

    stop.store(true);
    worker_thread.join();
}

// ============================================================
// B3: Upper-layer remote_total == 0 fallback path
// Covers L701-715: greedy_search_upper_pushdown's inline fallback
// when all upper-layer neighbors are local (no remote to offload).
//
// Graph: 2 shards, max_level=2, but EP's upper-layer neighbors are
// all in shard 0 (local) → remote_total=0 → fallback branch.
// ============================================================

TEST(PushdownSearch, UpperLayerAllLocalFallback)
{
    // Reuse MultiLevelFixture but with a graph where upper-layer neighbors
    // of EP (gid=15) are all in shard 0 (local).
    // MultiLevelFixture already has EP=15 in shard 0 with upper-layer
    // neighbors at step distances. With N_PER_SHARD=30, neighbors at
    // gid 15±step are in [0,30) → all local.
    //
    // The existing MultiLevelGreedyUpperPushdown test triggers the
    // remote_total > 0 path (neighbors span shard boundary at step=4).
    // We need a variant where upper neighbors stay local.
    //
    // Solution: use a larger N_PER_SHARD so upper-layer steps don't cross.
    static constexpr uint64_t N = 200;
    static constexpr uint64_t N_PER_SHARD = 100;
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr int32_t MAX_LEVEL = 2;
    static constexpr uint32_t NUM_LEVEL_SLOTS = 4;
    static constexpr uint32_t BIG_M = 8; // need ≥16 neighbors for ldnp
    static constexpr uint32_t BIG_NN = 2 * BIG_M;

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER_SHARD;
        sb.dim = PD_DIM;
        sb.M = BIG_M;
        sb.ef_search = 32;
        sb.max_level = MAX_LEVEL;
        sb.entry_point = 50; // in shard 0, center of local range
        sb.num_levels = NUM_LEVEL_SLOTS;
        sb.num_shards = NUM_SHARDS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER_SHARD;
        sb.neighbors_size = N_PER_SHARD * static_cast<uint64_t>(MAX_LEVEL + 1) * BIG_NN;
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
    }

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        auto *sb = sb_mut[s];

        // Vectors: 1D line
        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        // Levels: EP=50 at level 2, others at level 1 or 0
        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            if (gid == 50)
                levels[lid] = 3; // faiss level 2 (top)
            else if (gid == 40 || gid == 60 || gid == 30 || gid == 70)
                levels[lid] = 2; // faiss level 1
            else
                levels[lid] = 1; // faiss level 0
        }

        // Offsets
        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        uint64_t cum = 0;
        for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++) {
            offsets[lid] = cum;
            if (lid < N_PER_SHARD) {
                int faiss_lv = levels[lid] - 1;
                cum += static_cast<uint64_t>(faiss_lv + 1) * BIG_NN;
            }
        }

        // Neighbors
        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        std::fill(neighbors, neighbors + offsets[N_PER_SHARD], EMPTY_NEIGHBOR);

        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int faiss_lv = levels[lid] - 1;
            uint64_t base = offsets[lid];

            for (int lv = faiss_lv; lv >= 0; lv--) {
                uint64_t layer_base = base + static_cast<uint64_t>(lv) * BIG_NN;
                int32_t *nb = neighbors + layer_base;

                // Upper layers (lv >= 1): large step, stay local in shard 0
                //   Layer 2 (lv=2): step=20 → EP 50 connects to 30, 70 (both local)
                //   Layer 1 (lv=1): step=10 → connects within [0,100)
                // Layer 0: step=1 → tight grid for good recall
                int step = (lv >= 1) ? (lv == 2 ? 20 : 10) : 1;
                int idx = 0;
                for (uint32_t r = 0; r < BIG_M; r++) {
                    int64_t left = gid - step * (1 + static_cast<int64_t>(r));
                    int64_t right = gid + step * (1 + static_cast<int64_t>(r));
                    if (left >= 0 && idx < static_cast<int>(BIG_NN))
                        nb[idx++] = static_cast<int32_t>(left);
                    if (right < static_cast<int64_t>(N) && idx < static_cast<int>(BIG_NN))
                        nb[idx++] = static_cast<int32_t>(right);
                }
            }
        }

        // cum_nneighbor
        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        for (uint32_t i = 0; i < NUM_LEVEL_SLOTS; i++)
            cum_nn[i] = static_cast<int32_t>(i * BIG_NN);

        // CRCs
        sb->header_crc64 = compute_header_crc64(*sb);
        sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

        // ShardView
        shard_view[s].sb = sb;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = gid_begin;
        shard_view[s].ntotal_local = N_PER_SHARD;
    }

    // Full vectors for ground truth
    std::vector<float> all_vectors(N * PD_DIM);
    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        std::memcpy(all_vectors.data() + gid_begin * PD_DIM, shard_view[s].vectors,
                    N_PER_SHARD * PD_DIM * sizeof(float));
    }

    // Channels, proxy, worker for remote shard 1
    PushdownChannelPair channel(BIG_NN, PD_DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NUM_SHARDS);
    proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, BIG_NN, 0, 0);

    ServiceChannelPair pair{channel.task, channel.result};
    DualDistServiceWorker worker(1, shard_view[1], PD_DIM, PD_RUN_GEN, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    // Searcher
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(
        reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    // Query near gid=50 → upper-layer greedy will explore local neighbors
    // EP=50 at level 2, neighbors at step 20 → gid 30, 70 (both local in shard 0)
    // → remote_total == 0 → triggers L701-715 fallback
    float query[PD_DIM];
    float qval = 50.0f / static_cast<float>(N);
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    int32_t ids[5];
    float dists[5];
    searcher->search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    auto gt = brute_force_topk_global(query, all_vectors.data(), N, PD_DIM, 5);
    double recall = compute_recall(ids, gt, 5);
    EXPECT_GE(recall, 0.60) << "Recall@5 = " << recall;

    stop.store(true, std::memory_order_relaxed);
    if (worker_thread.joinable())
        worker_thread.join();
}

// ============================================================
// B4: Remote shard with ≥16 neighbors → NEON ldnp path
// Covers L1099-1232: search_at_layer_pushdown's remote LDNP branch
//
// Need a node on the REMOTE shard whose neighbor count at layer 0
// is ≥16. With M=8, NN_PER_NODE=16 → exactly fills one ldnp batch.
// ============================================================

TEST(PushdownSearch, RemoteShardLdnpBatchPath)
{
    static constexpr uint64_t N = 200;
    static constexpr uint64_t N_PER_SHARD = 100;
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr uint32_t LDNP_M = 8;
    static constexpr uint32_t LDNP_NN = 2 * LDNP_M; // = 16, exactly one ldnp batch

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER_SHARD;
        sb.dim = PD_DIM;
        sb.M = LDNP_M;
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = 50; // shard 0
        sb.num_levels = 2;
        sb.num_shards = NUM_SHARDS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER_SHARD;
        sb.neighbors_size = N_PER_SHARD * LDNP_NN;
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
    }

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        auto *sb = sb_mut[s];

        // Vectors: 1D line
        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        // Levels: all at layer 0
        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++)
            levels[lid] = 1;

        // Offsets
        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++)
            offsets[lid] = lid * LDNP_NN;

        // Neighbors: ±1..±M (16 neighbors per node, all non-EMPTY)
        // This ensures remote nodes (shard 1) have exactly 16 neighbors
        // → triggers the ldnp batch loop (j + 16 <= n_nb)
        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int32_t *nb = neighbors + lid * LDNP_NN;
            for (uint32_t j = 0; j < LDNP_NN; j++)
                nb[j] = EMPTY_NEIGHBOR;
            int idx = 0;
            for (uint32_t r = 0; r < LDNP_M; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0 && idx < static_cast<int>(LDNP_NN))
                    nb[idx++] = static_cast<int32_t>(left);
                if (right < static_cast<int64_t>(N) && idx < static_cast<int>(LDNP_NN))
                    nb[idx++] = static_cast<int32_t>(right);
            }
        }

        // cum_nneighbor
        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(LDNP_NN);

        // CRCs
        sb->header_crc64 = compute_header_crc64(*sb);
        sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

        // ShardView
        shard_view[s].sb = sb;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = gid_begin;
        shard_view[s].ntotal_local = N_PER_SHARD;
    }

    // Full vectors
    std::vector<float> all_vectors(N * PD_DIM);
    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        std::memcpy(all_vectors.data() + gid_begin * PD_DIM, shard_view[s].vectors,
                    N_PER_SHARD * PD_DIM * sizeof(float));
    }

    // Channel, proxy, worker for remote shard 1
    PushdownChannelPair channel(LDNP_NN, PD_DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NUM_SHARDS);
    proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, LDNP_NN, 0, 0);

    ServiceChannelPair pair{channel.task, channel.result};
    DualDistServiceWorker worker(1, shard_view[1], PD_DIM, PD_RUN_GEN, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    // Searcher
    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(
        reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    // Query that will cause search to traverse into remote shard nodes.
    // EP=50 (shard 0), neighbors include remote nodes (gid >= 100).
    // When search expands a remote node, its neighbors are loaded via ldnp.
    float query[PD_DIM];
    float qval = 0.5f; // near boundary → search crosses shards
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 10;
    params.ef_search = 64; // large ef → more remote node expansions

    std::vector<int32_t> ids(params.k);
    std::vector<float> dists(params.k);
    searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 0; i < params.k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    auto gt = brute_force_topk_global(query, all_vectors.data(), N, PD_DIM, params.k);
    double recall = compute_recall(ids.data(), gt, params.k);
    EXPECT_GE(recall, 0.70) << "Recall@" << params.k << " = " << recall;

    stop.store(true, std::memory_order_relaxed);
    if (worker_thread.joinable())
        worker_thread.join();
}

// ============================================================
// B5a: Remote-shard LDNP with 20 neighbors — covers bridge prefetch
// (j+18<=n_nb branch) and all 14+2 lane classify calls.
// EP is in remote shard 1 → first candidate expansion enters LDNP path.
// ============================================================
TEST(PushdownSearch, RemoteShardLdnpBridgePrefetch)
{
    static constexpr uint64_t N = 200;
    static constexpr uint64_t N_PER_SHARD = 100;
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr uint32_t BIG_M = 10;
    static constexpr uint32_t BIG_NN = 2 * BIG_M; // = 20 > 18 → bridge prefetch

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER_SHARD;
        sb.dim = PD_DIM;
        sb.M = BIG_M;
        sb.ef_search = 32;
        sb.max_level = 0;
        // EP in remote shard 1 → first expansion enters LDNP path
        sb.entry_point = N_PER_SHARD; // gid=100, shard 1
        sb.num_levels = 2;
        sb.num_shards = NUM_SHARDS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER_SHARD;
        sb.neighbors_size = N_PER_SHARD * BIG_NN;
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
    }

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        auto *sb = sb_mut[s];

        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++)
            levels[lid] = 1;

        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++)
            offsets[lid] = lid * BIG_NN;

        // Neighbors: ±1..±M (all 20 slots filled, non-EMPTY)
        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int32_t *nb = neighbors + lid * BIG_NN;
            for (uint32_t j = 0; j < BIG_NN; j++)
                nb[j] = EMPTY_NEIGHBOR;
            int idx = 0;
            for (uint32_t r = 0; r < BIG_M; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0 && idx < static_cast<int>(BIG_NN))
                    nb[idx++] = static_cast<int32_t>(left);
                if (right < static_cast<int64_t>(N) && idx < static_cast<int>(BIG_NN))
                    nb[idx++] = static_cast<int32_t>(right);
            }
        }

        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(BIG_NN);

        sb->header_crc64 = compute_header_crc64(*sb);
        sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

        shard_view[s].sb = sb;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = gid_begin;
        shard_view[s].ntotal_local = N_PER_SHARD;
    }

    std::vector<float> all_vectors(N * PD_DIM);
    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        std::memcpy(all_vectors.data() + gid_begin * PD_DIM, shard_view[s].vectors,
                    N_PER_SHARD * PD_DIM * sizeof(float));
    }

    PushdownChannelPair channel(BIG_NN, PD_DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NUM_SHARDS);
    proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, BIG_NN, 0, 0);

    ServiceChannelPair pair{channel.task, channel.result};
    DualDistServiceWorker worker(1, shard_view[1], PD_DIM, PD_RUN_GEN, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(
        reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
    // EP=100 is on remote shard 1; load_local_vectors nulls remote vectors.
    // Cache EP vector so get_vector(entry_point_) doesn't dereference nullptr.
    searcher->set_ep_vector(reinterpret_cast<const float *>(shard_buf[1].data() + sb_mut[1]->blocks[BLK_VECTORS].off),
                            PD_DIM);
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    // Query near gid=100 (EP) → search immediately expands remote EP
    float query[PD_DIM];
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = static_cast<float>(N_PER_SHARD) / static_cast<float>(N);

    HnswQueryParams params;
    params.k = 10;
    params.ef_search = 64;

    std::vector<int32_t> ids(params.k);
    std::vector<float> dists(params.k);
    searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 0; i < params.k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    stop.store(true, std::memory_order_relaxed);
    if (worker_thread.joinable())
        worker_thread.join();
}

// ============================================================
// B5b: Remote-shard LDNP with 17 neighbors — covers else-if branch
// (j+17<=n_nb but j+18>n_nb). Custom graph: EP has exactly 17 neighbor
// slots instead of the standard 2*M.
// ============================================================
TEST(PushdownSearch, RemoteShardLdnpOddBatch)
{
    static constexpr uint64_t N = 200;
    static constexpr uint64_t N_PER_SHARD = 100;
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr uint32_t ODD_NN = 17; // exactly 17 → triggers else-if

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER_SHARD;
        sb.dim = PD_DIM;
        sb.M = 8; // standard M for most nodes
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = N_PER_SHARD; // gid=100, shard 1 (remote)
        sb.num_levels = 2;
        sb.num_shards = NUM_SHARDS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER_SHARD;
        // Use ODD_NN for all nodes in shard 1 (EP's shard), standard for shard 0
        if (s == 1) {
            sb.neighbors_size = N_PER_SHARD * ODD_NN;
        } else {
            sb.neighbors_size = N_PER_SHARD * 2 * sb.M;
        }
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
    }

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        auto *sb = sb_mut[s];
        uint32_t nn = (s == 1) ? ODD_NN : (2 * sb->M);

        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++)
            levels[lid] = 1;

        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++)
            offsets[lid] = lid * nn;

        // Neighbors: ±1..±(nn/2) — fill all nn slots
        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int32_t *nb = neighbors + lid * nn;
            for (uint32_t j = 0; j < nn; j++)
                nb[j] = EMPTY_NEIGHBOR;
            int idx = 0;
            uint32_t half = nn / 2;
            for (uint32_t r = 0; r < half; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0 && idx < static_cast<int>(nn))
                    nb[idx++] = static_cast<int32_t>(left);
                if (right < static_cast<int64_t>(N) && idx < static_cast<int>(nn))
                    nb[idx++] = static_cast<int32_t>(right);
            }
            // Fill remaining slot(s) with nearest valid neighbor
            while (idx < static_cast<int>(nn)) {
                int64_t extra = gid + 1 + static_cast<int64_t>(idx);
                if (extra < static_cast<int64_t>(N))
                    nb[idx++] = static_cast<int32_t>(extra);
                else
                    nb[idx++] = EMPTY_NEIGHBOR;
            }
        }

        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(nn);

        sb->header_crc64 = compute_header_crc64(*sb);
        sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

        shard_view[s].sb = sb;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = gid_begin;
        shard_view[s].ntotal_local = N_PER_SHARD;
    }

    std::vector<float> all_vectors(N * PD_DIM);
    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        std::memcpy(all_vectors.data() + gid_begin * PD_DIM, shard_view[s].vectors,
                    N_PER_SHARD * PD_DIM * sizeof(float));
    }

    // max_batch must accommodate the 17-neighbor batch
    PushdownChannelPair channel(ODD_NN, PD_DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NUM_SHARDS);
    proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, ODD_NN, 0, 0);

    ServiceChannelPair pair{channel.task, channel.result};
    DualDistServiceWorker worker(1, shard_view[1], PD_DIM, PD_RUN_GEN, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(
        reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
    // EP=100 is on remote shard 1; load_local_vectors nulls remote vectors.
    // Cache EP vector so get_vector(entry_point_) doesn't dereference nullptr.
    searcher->set_ep_vector(reinterpret_cast<const float *>(shard_buf[1].data() + sb_mut[1]->blocks[BLK_VECTORS].off),
                            PD_DIM);
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    float query[PD_DIM];
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = static_cast<float>(N_PER_SHARD) / static_cast<float>(N);

    HnswQueryParams params;
    params.k = 10;
    params.ef_search = 64;

    std::vector<int32_t> ids(params.k);
    std::vector<float> dists(params.k);
    searcher->search_to(query, params, ids.data(), dists.data());

    for (int i = 0; i < params.k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    stop.store(true, std::memory_order_relaxed);
    if (worker_thread.joinable())
        worker_thread.join();
}

// ============================================================
// B5: parallel_memcpy_f32 large-vector path (lines 250-260)
// Triggers the parallel chunked memcpy when vec_count > 4M elements.
// Uses a single-shard graph with ntotal_local * dim > CHUNK_THRESHOLD.
// ============================================================

TEST(PushdownSearch, ParallelMemcpyLargeVectors)
{
    // CHUNK_THRESHOLD = 4*1024*1024 = 4194304 floats.
    // Use dim=8 and ntotal_local=525000 → vec_count=4400000 > 4194304.
    static constexpr uint32_t DIM = 8;
    static constexpr uint64_t NLOCAL = 525000;
    static constexpr uint32_t M = 1;

    auto buf = make_minimal_shard_buf(NLOCAL, NLOCAL, DIM, M, 0, 1, 0);

    GdHnswSearcher searcher(buf.data(), buf.size(), true);
    searcher.set_local_shard(0);

    // Allocate a large vector array and fill with a simple pattern
    std::vector<float> vecs(static_cast<size_t>(NLOCAL) * DIM, 0.0f);
    for (uint64_t i = 0; i < NLOCAL; i++)
        for (uint32_t d = 0; d < DIM; d++)
            vecs[static_cast<size_t>(i) * DIM + d] = static_cast<float>(i);

    // load_local_vectors should use parallel_memcpy_f32 path (vec_count > 4M)
    EXPECT_NO_THROW(searcher.load_local_vectors(vecs.data()));
}

// ============================================================
// B6: get_vector multi-shard path (lines 483-486)
// Covers the multi-shard lookup in get_vector for a non-EP local node.
// Uses 2 shards; the layer-0 search internally calls get_vector for
// non-EP local neighbors, triggering the multi-shard path at L483-486.
// ============================================================

TEST(PushdownSearch, GetVectorMultiShardPath)
{
    // Standard PushdownFixture: 2 shards, EP=25 in shard 0 (local).
    // During layer-0 search, neighbors of EP (e.g., gid=24, 26) are local
    // but not the EP → get_vector skips the EP cache → multi-shard path.
    PushdownFixture fx;

    float query[PD_DIM];
    float qval = 0.25f;
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    int32_t ids[5];
    float dists[5];
    fx.searcher->search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
}

// ============================================================
// B7: Layer-0 local tail == 1 (lines ~1357-1359)
// Covers the tail==1 branch in Phase 3 local distance computation.
// Need local_ids.size() % 4 == 1 after batch-4 loop.
// ============================================================

TEST(PushdownSearch, LocalTailOneNeighbor)
{
    // Use PushdownFixture with M=4 (8 neighbors per node).
    // Node at shard boundary (gid=48 in shard 0, boundary at 50):
    // neighbors: 47(local),49(local),46(local),50(remote),45(local),51(remote),44(local),52(remote)
    // → 5 local, 3 remote → local_ids.size()=5, 5%4=1 → tail==1 ✓
    PushdownFixture fx;

    float query[PD_DIM];
    // Query near gid=48 to force expansion of node 48
    float qval = 48.0f / static_cast<float>(fx.N);
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 64; // larger ef to expand boundary nodes

    int32_t ids[5];
    float dists[5];
    fx.searcher->search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";

    // Also test with a node that has exactly 1 local neighbor.
    // Use M=1: gid=0 has 1 valid neighbor (gid=1).
    // local_ids.size()=1, 1%4=1 → tail==1 ✓
    static constexpr uint32_t DIM = 4;
    static constexpr uint64_t N = 40;
    static constexpr uint64_t N_PER = 20;
    static constexpr uint32_t M1 = 1;
    static constexpr uint32_t NN1 = 2 * M1;

    std::vector<char> shard_buf[2];
    ShardView shard_view[2];

    for (uint32_t s = 0; s < 2; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER;
        sb.dim = DIM;
        sb.M = M1;
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = 10;
        sb.num_levels = 2;
        sb.num_shards = 2;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER;
        sb.neighbors_size = N_PER * NN1;
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        auto *sbp = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());

        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sbp->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER; lid++) {
            uint64_t gid = sbp->global_id_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < DIM; d++)
                vecs[lid * DIM + d] = val;
        }

        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER; lid++)
            levels[lid] = 1;

        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sbp->blocks[BLK_OFFSETS].off);
        for (uint64_t lid = 0; lid <= N_PER; lid++)
            offsets[lid] = lid * NN1;

        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
        for (uint64_t lid = 0; lid < N_PER; lid++) {
            int64_t gid = static_cast<int64_t>(sbp->global_id_begin + lid);
            int32_t *nb = neighbors + lid * NN1;
            for (uint32_t j = 0; j < NN1; j++)
                nb[j] = EMPTY_NEIGHBOR;
            int64_t left = gid - 1;
            int64_t right = gid + 1;
            if (left >= 0)
                nb[0] = static_cast<int32_t>(left);
            if (right < static_cast<int64_t>(N))
                nb[1] = static_cast<int32_t>(right);
        }

        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(NN1);

        sbp->header_crc64 = compute_header_crc64(*sbp);
        sbp->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sbp);

        shard_view[s].sb = sbp;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sbp->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sbp->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sbp->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = sbp->global_id_begin;
        shard_view[s].ntotal_local = N_PER;
    }

    PushdownChannelPair channel_m1(NN1, DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(2);
    proxies_2d[0][1].init(channel_m1.task, channel_m1.result, DIM, MAX_SPIN_ITERS_DEFAULT, NN1, 0, 0);

    ServiceChannelPair pair{channel_m1.task, channel_m1.result};
    DualDistServiceWorker worker(1, shard_view[1], DIM, 1, std::vector<ServiceChannelPair>{pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(reinterpret_cast<const float *>(
        shard_buf[0].data() + reinterpret_cast<SuperBlockV1 *>(shard_buf[0].data())->blocks[BLK_VECTORS].off));
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    // Query near gid=0 which has 1 local neighbor (gid=1)
    float query2[DIM];
    float qval2 = 0.5f / static_cast<float>(N);
    for (uint32_t d = 0; d < DIM; d++)
        query2[d] = qval2;

    HnswQueryParams params2;
    params2.k = 3;
    params2.ef_search = 32;

    int32_t ids2[3];
    float dists2[3];
    searcher->search_to(query2, params2, ids2, dists2);

    for (int i = 0; i < 3; i++)
        EXPECT_GE(ids2[i], 0) << "result " << i << " is empty";

    stop.store(true, std::memory_order_relaxed);
    if (worker_thread.joinable())
        worker_thread.join();
}

// ============================================================
// B8: Scalar tail prefetch path (lines 1083-1089)
// Covers vt.prefetch(next2) in the scalar tail of Phase 1.
// Need a node whose first 4-neighbor batch contains an EMPTY,
// causing NEON break, then j+2 < n_nb with non-EMPTY at j+2.
// ============================================================

TEST(PushdownSearch, ScalarTailPrefetch)
{
    // TailCoverageFixture with M=2: gid=1 has neighbors [0, 2, EMPTY, 3].
    // NEON batch detects EMPTY → breaks. Scalar tail:
    //   j=0: nid=0, j+2=2 < 4, nb_ptr[2]=EMPTY → no prefetch
    //   j=1: nid=2, j+2=3 < 4, nb_ptr[3]=3 → prefetch(3) ✓
    TailCoverageFixture fx;

    // Query very close to gid=1 to ensure node 1 is expanded
    float query[4];
    float qval = 1.0f / static_cast<float>(fx.TC_N); // gid=1/N ≈ 0.025
    for (uint32_t d = 0; d < 4; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    int32_t ids[5];
    float dists[5];
    fx.searcher->search_to(query, params, ids, dists);

    for (int i = 0; i < 5; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
}

// ============================================================
// B9: Upper-layer pushdown poll timeout (lines 788-793)
// Covers the abort() path when upper-layer remote poll exceeds 5s.
// We set up a multi-shard graph with upper layers where EP has remote
// upper-layer neighbors, but DON'T start the service worker.
// The proxy submits a request that never gets answered → timeout.
// ============================================================

TEST(PushdownSearch, UpperLayerPollTimeoutDeath)
{
    static constexpr uint64_t N = 200;
    static constexpr uint64_t N_PER_SHARD = 100;
    static constexpr uint32_t NUM_SHARDS = 2;
    static constexpr int32_t MAX_LEVEL = 2;
    static constexpr uint32_t NUM_LEVEL_SLOTS = 4;
    static constexpr uint32_t BIG_M = 8;
    static constexpr uint32_t BIG_NN = 2 * BIG_M;

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER_SHARD;
        sb.dim = PD_DIM;
        sb.M = BIG_M;
        sb.ef_search = 32;
        sb.max_level = MAX_LEVEL;
        sb.entry_point = 80; // near boundary at 100, so upper-layer neighbors span shards
        sb.num_levels = NUM_LEVEL_SLOTS;
        sb.num_shards = NUM_SHARDS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER_SHARD;
        sb.neighbors_size = N_PER_SHARD * static_cast<uint64_t>(MAX_LEVEL + 1) * BIG_NN;
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
    }

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        auto *sb = sb_mut[s];

        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            if (gid == 80)
                levels[lid] = 3; // EP at faiss level 2
            else if (gid == 70 || gid == 90 || gid == 60 || gid == 100)
                levels[lid] = 2; // faiss level 1
            else
                levels[lid] = 1;
        }

        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        uint64_t cum = 0;
        for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++) {
            offsets[lid] = cum;
            if (lid < N_PER_SHARD) {
                int faiss_lv = levels[lid] - 1;
                cum += static_cast<uint64_t>(faiss_lv + 1) * BIG_NN;
            }
        }

        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        std::fill(neighbors, neighbors + offsets[N_PER_SHARD], EMPTY_NEIGHBOR);

        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int faiss_lv = levels[lid] - 1;
            uint64_t base = offsets[lid];

            for (int lv = faiss_lv; lv >= 0; lv--) {
                uint64_t layer_base = base + static_cast<uint64_t>(lv) * BIG_NN;
                int32_t *nb = neighbors + layer_base;

                // Level 2 (lv=2): step=20 → EP 80 connects to 60, 100(remote!)
                // Level 1 (lv=1): step=10 → connects to 70, 90, 60, 100(remote!)
                // Level 0: step=1 → tight grid
                int step = (lv >= 1) ? (lv == 2 ? 20 : 10) : 1;
                int idx = 0;
                for (uint32_t r = 0; r < BIG_M; r++) {
                    int64_t left = gid - step * (1 + static_cast<int64_t>(r));
                    int64_t right = gid + step * (1 + static_cast<int64_t>(r));
                    if (left >= 0 && idx < static_cast<int>(BIG_NN))
                        nb[idx++] = static_cast<int32_t>(left);
                    if (right < static_cast<int64_t>(N) && idx < static_cast<int>(BIG_NN))
                        nb[idx++] = static_cast<int32_t>(right);
                }
            }
        }

        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        for (uint32_t i = 0; i < NUM_LEVEL_SLOTS; i++)
            cum_nn[i] = static_cast<int32_t>(i * BIG_NN);

        sb->header_crc64 = compute_header_crc64(*sb);
        sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

        shard_view[s].sb = sb;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = gid_begin;
        shard_view[s].ntotal_local = N_PER_SHARD;
    }

    // Set up channel and proxy but DON'T start the worker thread.
    // The proxy will submit a request but never get a response.
    PushdownChannelPair channel(BIG_NN, PD_DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NUM_SHARDS);
    proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, BIG_NN, 0, 0);

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(
        reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    // Query near EP=80 → upper-layer greedy finds remote neighbors (gid 100)
    // → submits remote request → no worker → poll timeout after 5s → abort
    float query[PD_DIM];
    float qval = 80.0f / static_cast<float>(N);
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    int32_t ids[5];
    float dists[5];
    EXPECT_DEATH(searcher->search_to(query, params, ids, dists), "upper pushdown poll timeout");
}

// ============================================================
// B10: Layer-0 pushdown poll timeout (lines 1391-1408)
// Covers the abort() path when layer-0 remote poll exceeds 5s.
// We set up a multi-shard graph (max_level=0) with remote neighbors,
// but DON'T start the service worker.
// ============================================================

TEST(PushdownSearch, Layer0PollTimeoutDeath)
{
    // Reuse PushdownFixture structure but without worker thread.
    // EP=48 (near boundary at 50) so search immediately finds remote neighbors.
    static constexpr uint64_t N = 100;
    static constexpr uint64_t N_PER_SHARD = 50;
    static constexpr uint32_t NUM_SHARDS = 2;

    std::vector<char> shard_buf[2];
    SuperBlockV1 *sb_mut[2] = {nullptr, nullptr};
    ShardView shard_view[2];

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        SuperBlockV1 sb{};
        sb.magic = GD_MAGIC;
        sb.schema_version = SCHEMA_VERSION;
        sb.state = static_cast<uint32_t>(GdState::ACTIVE);
        sb.metric_type = METRIC_L2;
        sb.endianness = ENDIAN_LITTLE;
        sb.ntotal = N;
        sb.ntotal_local = N_PER_SHARD;
        sb.dim = PD_DIM;
        sb.M = PD_M;
        sb.ef_search = 32;
        sb.max_level = 0;
        sb.entry_point = 48;
        sb.num_levels = 2;
        sb.num_shards = NUM_SHARDS;
        sb.shard_id = s;
        sb.global_id_begin = s * N_PER_SHARD;
        sb.neighbors_size = N_PER_SHARD * 2 * PD_M;
        compute_layout(sb);

        shard_buf[s].resize(sb.total_bytes, 0);
        std::memcpy(shard_buf[s].data(), &sb, sizeof(SuperBlockV1));
        sb_mut[s] = reinterpret_cast<SuperBlockV1 *>(shard_buf[s].data());
    }

    for (uint32_t s = 0; s < NUM_SHARDS; s++) {
        uint64_t gid_begin = s * N_PER_SHARD;
        auto *sb = sb_mut[s];

        float *vecs = reinterpret_cast<float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            uint64_t gid = gid_begin + lid;
            float val = static_cast<float>(gid) / static_cast<float>(N);
            for (uint32_t d = 0; d < PD_DIM; d++)
                vecs[lid * PD_DIM + d] = val;
        }

        int32_t *levels = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++)
            levels[lid] = 1;

        uint64_t *offsets = reinterpret_cast<uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t lid = 0; lid <= N_PER_SHARD; lid++)
            offsets[lid] = lid * 2 * PD_M;

        int32_t *neighbors = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t lid = 0; lid < N_PER_SHARD; lid++) {
            int64_t gid = static_cast<int64_t>(gid_begin + lid);
            int32_t *nb = neighbors + lid * 2 * PD_M;
            for (uint32_t j = 0; j < 2 * PD_M; j++)
                nb[j] = EMPTY_NEIGHBOR;
            for (uint32_t r = 0; r < PD_M; r++) {
                int64_t left = gid - 1 - static_cast<int64_t>(r);
                int64_t right = gid + 1 + static_cast<int64_t>(r);
                if (left >= 0)
                    nb[2 * r] = static_cast<int32_t>(left);
                if (right < static_cast<int64_t>(N))
                    nb[2 * r + 1] = static_cast<int32_t>(right);
            }
        }

        int32_t *cum_nn = reinterpret_cast<int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(2 * PD_M);

        sb->header_crc64 = compute_header_crc64(*sb);
        sb->payload_crc64 = compute_payload_crc64(shard_buf[s].data(), *sb);

        shard_view[s].sb = sb;
        shard_view[s].vectors = reinterpret_cast<const float *>(shard_buf[s].data() + sb->blocks[BLK_VECTORS].off);
        shard_view[s].levels = reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_LEVELS].off);
        shard_view[s].offsets = reinterpret_cast<const uint64_t *>(shard_buf[s].data() + sb->blocks[BLK_OFFSETS].off);
        shard_view[s].neighbors =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view[s].cum_nneighbor =
            reinterpret_cast<const int32_t *>(shard_buf[s].data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view[s].global_id_begin = gid_begin;
        shard_view[s].ntotal_local = N_PER_SHARD;
    }

    // Set up channel and proxy but DON'T start the worker thread.
    PushdownChannelPair channel(PD_MAX_BATCH, PD_DIM);
    std::atomic<bool> stop{false};
    std::vector<std::vector<DualDistProxy>> proxies_2d(1);
    proxies_2d[0].resize(NUM_SHARDS);
    proxies_2d[0][1].init(channel.task, channel.result, PD_DIM, MAX_SPIN_ITERS_DEFAULT, PD_MAX_BATCH, 0, 0);

    std::vector<std::pair<const void *, uint64_t>> shard_ptrs = {{shard_buf[0].data(), shard_buf[0].size()},
                                                                 {shard_buf[1].data(), shard_buf[1].size()}};
    auto searcher = std::make_unique<GdHnswSearcher>(shard_ptrs, true);
    searcher->set_local_shard(0);
    searcher->load_local_vectors(
        reinterpret_cast<const float *>(shard_buf[0].data() + sb_mut[0]->blocks[BLK_VECTORS].off));
    searcher->set_dual_pushdown_proxies(&proxies_2d, 1);

    // Query near boundary → search expands remote neighbors → no worker → timeout
    float query[PD_DIM];
    float qval = 48.0f / static_cast<float>(N);
    for (uint32_t d = 0; d < PD_DIM; d++)
        query[d] = qval;

    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 32;

    int32_t ids[5];
    float dists[5];
    EXPECT_DEATH(searcher->search_to(query, params, ids, dists), "pushdown poll timeout");
}
