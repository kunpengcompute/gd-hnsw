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
#include <chrono>

#include "dist_service.h"
#include "test_distance_common.hpp"

using namespace gd_hnsw;

// 按截止时间自旋等待 worker 响应。固定次数(100000次)自旋预算不足1ms，
// 在 ctest --parallel 多测试进程争抢 CPU 时，worker 线程可能整个预算内
// 都得不到调度，导致偶发超时。改为时间预算(默认5s)。
static bool wait_for_response(const DualDistProxy &proxy, int timeout_ms = 5000)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!proxy.try_wait_batch()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        gd_hnsw::cpu_pause();
    }
    return true;
}

// ============================================================
// Fixture: builds an in-memory graph buffer and provides
// channel + worker setup for concurrency testing.
// ============================================================

static constexpr uint32_t TEST_DIM = 8;
static constexpr uint32_t TEST_M = 4;
static constexpr uint32_t TEST_MAX_BATCH = 2 * TEST_M; // 8
static constexpr uint64_t TEST_N = 100;

struct ConcurrencyFixture {
    std::vector<char> graph_buf;
    const SuperBlockV1 *sb = nullptr;
    ShardView shard_view;

    ConcurrencyFixture()
    {
        // Build a minimal 1D-line graph (same pattern as TestGraph)
        uint32_t nlevels = 2;
        uint32_t nn_per_node = 2 * TEST_M;
        uint64_t n = TEST_N;

        SuperBlockV1 tmp{};
        tmp.magic = GD_MAGIC;
        tmp.schema_version = SCHEMA_VERSION;
        tmp.state = static_cast<uint32_t>(GdState::ACTIVE);
        tmp.metric_type = METRIC_L2;
        tmp.endianness = ENDIAN_LITTLE;
        tmp.ntotal = n;
        tmp.ntotal_local = n;
        tmp.dim = TEST_DIM;
        tmp.M = TEST_M;
        tmp.ef_search = 16;
        tmp.max_level = 0;
        tmp.entry_point = static_cast<int32_t>(n / 2);
        tmp.num_levels = nlevels;
        tmp.neighbors_size = n * nn_per_node;
        tmp.num_shards = 0;
        compute_layout(tmp);

        graph_buf.resize(tmp.total_bytes, 0);
        std::memcpy(graph_buf.data(), &tmp, sizeof(SuperBlockV1));
        sb = reinterpret_cast<const SuperBlockV1 *>(graph_buf.data());

        // Fill vectors: 1D line
        float *vecs = reinterpret_cast<float *>(graph_buf.data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < n; i++) {
            float val = static_cast<float>(i) / static_cast<float>(n);
            for (uint32_t d = 0; d < TEST_DIM; d++)
                vecs[i * TEST_DIM + d] = val;
        }

        // Fill levels
        int32_t *levels = reinterpret_cast<int32_t *>(graph_buf.data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < n; i++)
            levels[i] = 1;

        // Fill offsets
        uint64_t *offsets = reinterpret_cast<uint64_t *>(graph_buf.data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t i = 0; i <= n; i++)
            offsets[i] = i * nn_per_node;

        // Fill neighbors
        int32_t *neighbors = reinterpret_cast<int32_t *>(graph_buf.data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < n; i++) {
            int32_t *nb = neighbors + i * nn_per_node;
            for (uint64_t s = 0; s < nn_per_node; s++)
                nb[s] = EMPTY_NEIGHBOR;
            for (uint32_t r = 0; r < TEST_M; r++) {
                int64_t left = static_cast<int64_t>(i) - 1 - static_cast<int64_t>(r);
                int64_t right = static_cast<int64_t>(i) + 1 + static_cast<int64_t>(r);
                if (left >= 0)
                    nb[2 * r] = static_cast<int32_t>(left);
                if (right < static_cast<int64_t>(n))
                    nb[2 * r + 1] = static_cast<int32_t>(right);
            }
        }

        // Fill cum_nneighbor
        int32_t *cum_nn = reinterpret_cast<int32_t *>(graph_buf.data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(nn_per_node);

        // CRCs
        auto *sb_mut = reinterpret_cast<SuperBlockV1 *>(graph_buf.data());
        sb_mut->header_crc64 = compute_header_crc64(*sb_mut);
        sb_mut->payload_crc64 = compute_payload_crc64(graph_buf.data(), *sb_mut);

        // Build ShardView
        shard_view.sb = sb;
        shard_view.vectors = reinterpret_cast<const float *>(graph_buf.data() + sb->blocks[BLK_VECTORS].off);
        shard_view.levels = reinterpret_cast<const int32_t *>(graph_buf.data() + sb->blocks[BLK_LEVELS].off);
        shard_view.offsets = reinterpret_cast<const uint64_t *>(graph_buf.data() + sb->blocks[BLK_OFFSETS].off);
        shard_view.neighbors = reinterpret_cast<const int32_t *>(graph_buf.data() + sb->blocks[BLK_NEIGHBORS].off);
        shard_view.cum_nneighbor =
            reinterpret_cast<const int32_t *>(graph_buf.data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        shard_view.global_id_begin = 0;
        shard_view.ntotal_local = n;
    }

    // Expected L2 distance between two vectors in the 1D-line graph
    static float expected_dist(int32_t id_a, int32_t id_b)
    {
        float diff = static_cast<float>(id_a - id_b) / static_cast<float>(TEST_N);
        return static_cast<float>(TEST_DIM) * diff * diff;
    }
};

// ============================================================
// Channel pair helper
// ============================================================

struct TestChannelPair {
    std::vector<char> task_buf;
    std::vector<char> result_buf;
    TaskChannelView task;
    ResultChannelView result;
    TaskChannelLayout task_layout;
    ResultChannelLayout result_layout;

    TestChannelPair()
    {
        task_layout = TaskChannelLayout::build(TEST_MAX_BATCH, TEST_DIM);
        result_layout = ResultChannelLayout::build(TEST_MAX_BATCH);

        task_buf.resize(task_layout.bytes_total, 0);
        result_buf.resize(result_layout.bytes_total, 0);

        task = TaskChannelView::from_raw(task_buf.data(), task_layout);
        result = ResultChannelView::from_raw(result_buf.data(), result_layout);

        task.h->channel_role = CHANNEL_ROLE_TASK;
        task.h->identity_valid = IDENTITY_VALID_MAGIC;
        task.h->max_batch = TEST_MAX_BATCH;
        task.h->dim = TEST_DIM;
        task.h->run_generation = 1; // must match worker's run_generation_
        task.h->req_seq.store(0, std::memory_order_relaxed);

        result.h->channel_role = CHANNEL_ROLE_RESULT;
        result.h->identity_valid = IDENTITY_VALID_MAGIC;
        result.h->max_batch = TEST_MAX_BATCH;
        result.h->dim = TEST_DIM;
        result.h->run_generation = 1;
        result.h->resp_seq.store(0, std::memory_order_relaxed);
    }
};

// ============================================================
// Single request roundtrip
// ============================================================

TEST(Concurrency, ServiceWorkerSingleRequest)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    // Submit a query from the test thread
    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    float query[TEST_DIM];
    for (uint32_t d = 0; d < TEST_DIM; d++)
        query[d] = 0.25f;

    int32_t ids[] = {0, 10, 50, 90};
    uint16_t count = 4;

    proxy.set_query(query);
    proxy.submit_batch(ids, count);

    // Wait for service to respond (with timeout)
    bool done = wait_for_response(proxy);
    ASSERT_TRUE(done) << "service worker did not respond within timeout";

    const float *dists = proxy.consume_batch();

    // Verify distances
    for (uint16_t i = 0; i < count; i++) {
        float expected = scalar::l2_sqr(query, fx.shard_view.vectors + ids[i] * TEST_DIM, TEST_DIM);
        EXPECT_NEAR(dists[i], expected, 1e-4f) << "wrong distance for id=" << ids[i];
    }

    // Shutdown
    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Multiple sequential roundtrips
// ============================================================

TEST(Concurrency, ServiceWorkerMultipleRoundtrips)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    for (int round = 0; round < 10; round++) {
        float query[TEST_DIM];
        float qval = 0.1f + static_cast<float>(round) * 0.08f;
        for (uint32_t d = 0; d < TEST_DIM; d++)
            query[d] = qval;

        int32_t ids[] = {static_cast<int32_t>(round * 3), static_cast<int32_t>(round * 3 + 1)};
        uint16_t count = 2;

        proxy.set_query(query);
        proxy.submit_batch(ids, count);

        bool done = wait_for_response(proxy);
        ASSERT_TRUE(done) << "round " << round << " timed out";

        const float *dists = proxy.consume_batch();
        for (uint16_t i = 0; i < count; i++) {
            float expected = scalar::l2_sqr(query, fx.shard_view.vectors + ids[i] * TEST_DIM, TEST_DIM);
            EXPECT_NEAR(dists[i], expected, 1e-4f) << "round " << round << " id=" << ids[i];
        }
    }

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Multiple proxies + single worker (concurrent submits)
// ============================================================

TEST(Concurrency, MultipleProxiesSingleWorker)
{
    ConcurrencyFixture fx;
    const uint32_t num_proxies = 4;

    std::vector<TestChannelPair> channels(num_proxies);
    std::vector<ServiceChannelPair> pairs;
    pairs.reserve(num_proxies);
    for (uint32_t i = 0; i < num_proxies; i++)
        pairs.push_back({channels[i].task, channels[i].result});

    std::atomic<bool> stop{false};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, pairs, stop);
    std::thread worker_thread([&]() { worker.run(); });

    // Each proxy thread submits queries concurrently
    std::vector<std::thread> proxy_threads;
    std::atomic<int> errors{0};

    for (uint32_t p = 0; p < num_proxies; p++) {
        proxy_threads.emplace_back([&, p]() {
            DualDistProxy proxy;
            proxy.init(channels[p].task, channels[p].result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, p);

            for (int round = 0; round < 5; round++) {
                float query[TEST_DIM];
                float qval = 0.05f + static_cast<float>(p) * 0.1f + static_cast<float>(round) * 0.02f;
                for (uint32_t d = 0; d < TEST_DIM; d++)
                    query[d] = qval;

                int32_t ids[] = {static_cast<int32_t>(p * 10 + round)};
                uint16_t count = 1;

                proxy.set_query(query);
                proxy.submit_batch(ids, count);

                bool done = wait_for_response(proxy);
                if (!done) {
                    errors.fetch_add(1);
                    return;
                }

                const float *dists = proxy.consume_batch();
                float expected = scalar::l2_sqr(query, fx.shard_view.vectors + ids[0] * TEST_DIM, TEST_DIM);
                if (std::fabs(dists[0] - expected) > 1e-4f) {
                    errors.fetch_add(1);
                }
            }
        });
    }

    for (auto &t : proxy_threads)
        t.join();

    EXPECT_EQ(errors.load(), 0) << "concurrent proxy requests had errors";

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Empty batch
// ============================================================

TEST(Concurrency, ServiceWorkerEmptyBatch)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    float query[TEST_DIM] = {};
    proxy.set_query(query);
    proxy.submit_batch(nullptr, 0); // count=0 is valid

    bool done = wait_for_response(proxy);
    ASSERT_TRUE(done);
    proxy.consume_batch();

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Capacity boundary
// ============================================================

TEST(Concurrency, ServiceWorkerMaxBatch)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    float query[TEST_DIM] = {};
    int32_t ids[TEST_MAX_BATCH];
    for (uint32_t i = 0; i < TEST_MAX_BATCH; i++)
        ids[i] = static_cast<int32_t>(i);

    proxy.set_query(query);
    proxy.submit_batch(ids, TEST_MAX_BATCH);

    bool done = wait_for_response(proxy);
    ASSERT_TRUE(done);

    const float *dists = proxy.consume_batch();
    for (uint32_t i = 0; i < TEST_MAX_BATCH; i++) {
        float expected = scalar::l2_sqr(query, fx.shard_view.vectors + ids[i] * TEST_DIM, TEST_DIM);
        EXPECT_NEAR(dists[i], expected, 1e-4f);
    }

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Phase C2: DualDistServiceWorker uncovered branches
// ============================================================

TEST(Concurrency, WorkerEmptyPairsNoop)
{
    ConcurrencyFixture fx;

    std::atomic<bool> stop{false};
    std::vector<ServiceChannelPair> empty_pairs;
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, empty_pairs, stop);

    // run() should return immediately when n_pairs == 0
    worker.run();
    // No crash = pass
}

TEST(Concurrency, WorkerDotNormEnabled)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    // Pre-compute vector norms for dot_norm path
    std::vector<float> norms(TEST_N);
    for (uint64_t i = 0; i < TEST_N; i++) {
        norms[i] = scalar::vec_norm_sq(fx.shard_view.vectors + i * TEST_DIM, TEST_DIM);
    }

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    worker.enable_dot_norm(norms.data());
    std::thread worker_thread([&]() { worker.run(); });

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    float query[TEST_DIM];
    for (uint32_t d = 0; d < TEST_DIM; d++)
        query[d] = 0.1f;

    int32_t ids[] = {5, 20, 50};
    uint16_t count = 3;

    proxy.set_query(query);
    proxy.submit_batch(ids, count);

    bool done = wait_for_response(proxy);
    ASSERT_TRUE(done) << "dot_norm worker did not respond";

    const float *dists = proxy.consume_batch();
    for (uint16_t i = 0; i < count; i++) {
        float expected = scalar::l2_sqr(query, fx.shard_view.vectors + ids[i] * TEST_DIM, TEST_DIM);
        EXPECT_NEAR(dists[i], expected, 1e-4f) << "dot_norm distance mismatch for id=" << ids[i];
    }

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

TEST(Concurrency, WorkerRunGenerationMismatch)
{
    ConcurrencyFixture fx;
    TestChannelPair ch; // run_generation = 1 in channel

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    // Worker has run_generation = 2, channel has 1 -> all requests skipped
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 2, {pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    float query[TEST_DIM] = {};
    int32_t ids[] = {0, 1, 2};

    proxy.set_query(query);
    proxy.submit_batch(ids, 3);

    // Give worker time to scan the channel
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Worker should NOT have processed the request due to generation mismatch:
    // the result channel resp_seq stays behind the request seq.
    EXPECT_EQ(proxy.debug_resp_seq(), 0u);

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Phase E2: dist_service.h non-FATAL paths
// ============================================================

// E2b: debug methods on DualDistProxy
TEST(Concurrency, ProxyDebugMethods)
{
    TestChannelPair ch;

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    EXPECT_EQ(proxy.debug_run_generation(), 1u);
    EXPECT_EQ(proxy.debug_query_epoch(), 0u);
    EXPECT_EQ(proxy.debug_count(), 0u); // no request submitted yet

    // Submit a batch and check debug_count
    float query[TEST_DIM] = {};
    proxy.set_query(query);
    int32_t ids[] = {0, 1, 2, 3};
    proxy.submit_batch(ids, 4);
    EXPECT_EQ(proxy.debug_count(), 4u);
}

// E2c: last_seen_seq_ = 0 path — odd req_seq with unmatched resp_seq
TEST(Concurrency, WorkerOddReqSeqAbsorption)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    // Set an odd req_seq that hasn't been consumed (resp_seq is 0)
    uint64_t odd_seq = 7;
    uint64_t doorbell = doorbell_encode(odd_seq, 2);
    ch.task.h->req_seq.store(doorbell, std::memory_order_relaxed);
    ch.result.h->resp_seq.store(0, std::memory_order_relaxed); // not matched

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    // Worker constructor should absorb the odd seq as last_seen_seq_ = 0
    // (neither condition met: seq is odd, resp != seq)

    std::thread worker_thread([&]() { worker.run(); });

    // Submit a normal request — worker should process it
    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    float query[TEST_DIM];
    for (uint32_t d = 0; d < TEST_DIM; d++)
        query[d] = 0.2f;
    int32_t ids[] = {10, 20};

    proxy.set_query(query);
    proxy.submit_batch(ids, 2);

    bool done = wait_for_response(proxy);
    ASSERT_TRUE(done);

    const float *dists = proxy.consume_batch();
    for (uint16_t i = 0; i < 2; i++) {
        float expected = scalar::l2_sqr(query, fx.shard_view.vectors + ids[i] * TEST_DIM, TEST_DIM);
        EXPECT_NEAR(dists[i], expected, 1e-4f);
    }

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// E2d: multi-round roundtrip with increasing batch sizes (1, 2, 3)
TEST(Concurrency, WorkerMultiBatchRoundtrip)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;

    std::atomic<bool> stop{false};
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    std::thread worker_thread([&]() { worker.run(); });

    DualDistProxy proxy;
    proxy.init(ch.task, ch.result, TEST_DIM, MAX_SPIN_ITERS_DEFAULT, TEST_MAX_BATCH, 0, 0);

    // Send multiple requests with various batch sizes
    float query[TEST_DIM];
    for (int round = 0; round < 3; round++) {
        for (uint32_t d = 0; d < TEST_DIM; d++)
            query[d] = 0.1f + static_cast<float>(round) * 0.1f;

        uint16_t count = static_cast<uint16_t>(round + 1); // 1, 2, 3
        int32_t ids[3];
        for (uint16_t j = 0; j < count; j++)
            ids[j] = static_cast<int32_t>(j * 10 + round);

        proxy.set_query(query);
        proxy.submit_batch(ids, count);

        bool done = wait_for_response(proxy);
        ASSERT_TRUE(done);
        proxy.consume_batch();
    }

    stop.store(true, std::memory_order_relaxed);
    worker_thread.join();
}

// ============================================================
// Phase F Tier B: DualDistServiceWorker DEATH tests
// ============================================================

TEST(Concurrency, WorkerMaxBatchMismatchDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch1;
    TestChannelPair ch2;
    ch2.task.h->max_batch = TEST_MAX_BATCH + 1; // different from ch1

    std::atomic<bool> stop{false};
    std::vector<ServiceChannelPair> pairs = {{ch1.task, ch1.result}, {ch2.task, ch2.result}};

    EXPECT_DEATH(DualDistServiceWorker(0, fx.shard_view, TEST_DIM, 1, pairs, stop), "max_batch mismatch");
}

// Helper: set up a channel with invalid identity and a pending request,
// then run the worker — deferred identity check should trigger abort.
static void RunWorkerWithBadChannel(ConcurrencyFixture &fx, TestChannelPair &ch, std::atomic<bool> &stop)
{
    ServiceChannelPair pair{ch.task, ch.result};
    DualDistServiceWorker worker(0, fx.shard_view, TEST_DIM, 1, {pair}, stop);
    worker.run();
}

TEST(Concurrency, WorkerTaskIdentityNotInitDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;
    ch.task.h->identity_valid = 0; // invalid
    // Submit a request so worker finds it (odd seq, resp != seq → last_seen=0)
    ch.task.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    EXPECT_DEATH(RunWorkerWithBadChannel(fx, ch, stop), "identity not initialized");
}

TEST(Concurrency, WorkerTaskRoleWrongDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;
    ch.task.h->channel_role = CHANNEL_ROLE_RESULT; // must be TASK
    ch.task.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    EXPECT_DEATH(RunWorkerWithBadChannel(fx, ch, stop), "wrong role");
}

TEST(Concurrency, WorkerResultIdentityNotInitDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;
    ch.result.h->identity_valid = 0; // invalid
    ch.task.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    EXPECT_DEATH(RunWorkerWithBadChannel(fx, ch, stop), "identity not initialized");
}

TEST(Concurrency, WorkerResultRoleWrongDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;
    ch.result.h->channel_role = CHANNEL_ROLE_TASK; // must be RESULT
    ch.task.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    EXPECT_DEATH(RunWorkerWithBadChannel(fx, ch, stop), "wrong role");
}

TEST(Concurrency, WorkerDimMismatchDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;
    ch.task.h->dim = TEST_DIM + 1; // different from worker dim
    ch.task.h->req_seq.store(doorbell_encode(1, 1), std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    EXPECT_DEATH(RunWorkerWithBadChannel(fx, ch, stop), "dim mismatch");
}

TEST(Concurrency, WorkerCountOverflowDies)
{
    ConcurrencyFixture fx;
    TestChannelPair ch;
    // Submit request with count > max_batch
    uint16_t overflow_count = TEST_MAX_BATCH + 1;
    ch.task.h->req_seq.store(doorbell_encode(1, overflow_count), std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    EXPECT_DEATH(RunWorkerWithBadChannel(fx, ch, stop), "count overflow");
}

// ============================================================
// Phase G1b note: WorkerProfiledIdleScans / WorkerProfiledTimingBreakdown
// were removed — they asserted on the split-only worker.stats() profiling
// interface (idle_scans / timing breakdown), which the api layer deleted.
// Idle-loop liveness is still exercised by WorkerRunGenerationMismatch and
// WorkerOddReqSeqAbsorption (worker spins without processing).
// ============================================================
