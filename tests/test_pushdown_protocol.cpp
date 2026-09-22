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

#include "dist_service.h"

using namespace gd_hnsw;

// ============================================================
// Helpers
// ============================================================

static constexpr uint32_t MAX_BATCH = 32;
static constexpr uint32_t DIM = 16;

struct MockChannels {
    std::vector<char> task_buf;
    std::vector<char> result_buf;
    TaskChannelView task;
    ResultChannelView result;
    TaskChannelLayout task_layout;
    ResultChannelLayout result_layout;

    MockChannels()
    {
        task_layout = TaskChannelLayout::build(MAX_BATCH, DIM);
        result_layout = ResultChannelLayout::build(MAX_BATCH);

        task_buf.resize(task_layout.bytes_total, 0);
        result_buf.resize(result_layout.bytes_total, 0);

        task = TaskChannelView::from_raw(task_buf.data(), task_layout);
        result = ResultChannelView::from_raw(result_buf.data(), result_layout);

        task.h->channel_role = CHANNEL_ROLE_TASK;
        task.h->identity_valid = IDENTITY_VALID_MAGIC;
        task.h->max_batch = MAX_BATCH;
        task.h->dim = DIM;
        task.h->req_seq.store(0, std::memory_order_relaxed);

        result.h->channel_role = CHANNEL_ROLE_RESULT;
        result.h->identity_valid = IDENTITY_VALID_MAGIC;
        result.h->max_batch = MAX_BATCH;
        result.h->dim = DIM;
        result.h->resp_seq.store(0, std::memory_order_relaxed);
    }

    DualDistProxy make_proxy()
    {
        DualDistProxy p;
        p.init(task, result, DIM, MAX_SPIN_ITERS_DEFAULT, MAX_BATCH, 0, 0);
        return p;
    }

    // Simulate service worker consuming task and writing results
    void mock_service_respond()
    {
        uint64_t db = task.h->req_seq.load(std::memory_order_acquire);
        uint64_t seq = doorbell_seq(db);
        uint16_t cnt = doorbell_count(db);
        for (uint16_t i = 0; i < cnt; i++)
            result.distances[i] = static_cast<float>(i) * 0.1f;
        result.h->resp_seq.store(seq, std::memory_order_release);
    }
};

// ============================================================
// Channel layout tests
// ============================================================

TEST(PushdownProtocol, TaskChannelLayoutOffsetsAligned)
{
    auto layout = TaskChannelLayout::build(MAX_BATCH, DIM);
    EXPECT_EQ(layout.header_off, 0u);
    // neighbor_ids_off must be cacheline-aligned (multiple of 64)
    EXPECT_EQ(layout.neighbor_ids_off % 64, 0u);
    EXPECT_GE(layout.neighbor_ids_off, sizeof(DistChannelHeader));
    // query_vec_off must be cacheline-aligned
    EXPECT_EQ(layout.query_vec_off % 64, 0u);
    EXPECT_GE(layout.query_vec_off, layout.neighbor_ids_off + MAX_BATCH * sizeof(int32_t));
    // bytes_total must be cacheline-aligned
    EXPECT_EQ(layout.bytes_total % 64, 0u);
    EXPECT_GE(layout.bytes_total, layout.query_vec_off + DIM * sizeof(float));
}

TEST(PushdownProtocol, TaskChannelLayoutCapacityMatchesSize)
{
    auto layout = TaskChannelLayout::build(MAX_BATCH, DIM);
    // Must have room for header + max_batch ids + dim floats
    EXPECT_GE(layout.bytes_total, sizeof(DistChannelHeader) + MAX_BATCH * sizeof(int32_t) + DIM * sizeof(float));
}

TEST(PushdownProtocol, ResultChannelLayoutOffsetsAligned)
{
    auto layout = ResultChannelLayout::build(MAX_BATCH);
    EXPECT_EQ(layout.header_off, 0u);
    EXPECT_EQ(layout.distances_off % 64, 0u);
    EXPECT_GE(layout.distances_off, sizeof(DistChannelHeader));
    EXPECT_EQ(layout.bytes_total % 64, 0u);
    EXPECT_GE(layout.bytes_total, layout.distances_off + MAX_BATCH * sizeof(float));
}

TEST(PushdownProtocol, LayoutVariesWithMaxBatch)
{
    auto small = TaskChannelLayout::build(8, DIM);
    auto large = TaskChannelLayout::build(256, DIM);
    EXPECT_LT(small.bytes_total, large.bytes_total);
    EXPECT_LT(small.neighbor_ids_off, large.neighbor_ids_off + 1); // may be same alignment
}

TEST(PushdownProtocol, LayoutVariesWithDim)
{
    auto small = TaskChannelLayout::build(MAX_BATCH, 8);
    auto large = TaskChannelLayout::build(MAX_BATCH, 256);
    EXPECT_LT(small.bytes_total, large.bytes_total);
}

// ============================================================
// DistChannelHeader tests
// ============================================================

TEST(PushdownProtocol, DistChannelHeaderSize)
{
    EXPECT_EQ(sizeof(DistChannelHeader), 192u); // exactly 3 cachelines
}

TEST(PushdownProtocol, DistChannelHeaderAlignment) { EXPECT_EQ(alignof(DistChannelHeader), 64u); }

TEST(PushdownProtocol, HeaderFieldsAtExpectedOffsets)
{
    EXPECT_EQ(offsetof(DistChannelHeader, req_seq), 0u);
    EXPECT_EQ(offsetof(DistChannelHeader, resp_seq), 64u);
    EXPECT_EQ(offsetof(DistChannelHeader, run_generation), 128u);
}

TEST(PushdownProtocol, HeaderInitialState)
{
    DistChannelHeader hdr{};
    EXPECT_EQ(hdr.req_seq.load(), 0u);
    EXPECT_EQ(hdr.resp_seq.load(), 0u);
    EXPECT_EQ(hdr.channel_role, 0u);
    EXPECT_EQ(hdr.identity_valid, 0u);
}

// ============================================================
// DualDistProxy init tests
// ============================================================

TEST(PushdownProtocol, InitWithValidChannels)
{
    MockChannels mc;
    DualDistProxy p;
    EXPECT_NO_THROW(p.init(mc.task, mc.result, DIM, MAX_SPIN_ITERS_DEFAULT, MAX_BATCH, 0, 0));
    EXPECT_EQ(p.debug_local_seq(), 0u);
}

TEST(PushdownProtocol, InitSetsIdentityFields)
{
    MockChannels mc;
    mc.make_proxy(); // init() writes src_rank/thread_id to channel headers
    EXPECT_EQ(mc.task.h->src_rank, 0u);
    EXPECT_EQ(mc.task.h->thread_id, 0u);
    EXPECT_EQ(mc.result.h->thread_id, 0u);
}

TEST(PushdownProtocol, InitRejectsInvalidIdentity)
{
    MockChannels mc;
    mc.task.h->identity_valid = 0; // invalid
    DualDistProxy p;
    EXPECT_DEATH(p.init(mc.task, mc.result, DIM, MAX_SPIN_ITERS_DEFAULT, MAX_BATCH, 0, 0), "identity not initialized");
}

TEST(PushdownProtocol, InitRejectsWrongChannelRole)
{
    MockChannels mc;
    mc.task.h->channel_role = CHANNEL_ROLE_RESULT; // task must be TASK
    DualDistProxy p;
    EXPECT_DEATH(p.init(mc.task, mc.result, DIM, MAX_SPIN_ITERS_DEFAULT, MAX_BATCH, 0, 0), "wrong role");
}

TEST(PushdownProtocol, InitRejectsResultChannelInvalidIdentity)
{
    MockChannels mc;
    mc.result.h->identity_valid = 0; // invalid
    DualDistProxy p;
    EXPECT_DEATH(p.init(mc.task, mc.result, DIM, MAX_SPIN_ITERS_DEFAULT, MAX_BATCH, 0, 0), "identity not initialized");
}

TEST(PushdownProtocol, InitRejectsResultChannelWrongRole)
{
    MockChannels mc;
    mc.result.h->channel_role = CHANNEL_ROLE_TASK; // result must be RESULT
    DualDistProxy p;
    EXPECT_DEATH(p.init(mc.task, mc.result, DIM, MAX_SPIN_ITERS_DEFAULT, MAX_BATCH, 0, 0), "wrong role");
}

// ============================================================
// submit / try_wait / consume tests
// ============================================================

TEST(PushdownProtocol, SubmitSetsOddSeq)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    int32_t ids[] = {1, 2, 3};
    p.submit_batch(ids, 3);
    // After submit, seq should be odd (1 for first submit)
    EXPECT_EQ(p.debug_local_seq(), 1u);
    EXPECT_EQ(p.debug_local_seq() & 1, 1u);
}

TEST(PushdownProtocol, SubmitWritesQueryToGd)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM];
    for (uint32_t i = 0; i < DIM; i++)
        query[i] = static_cast<float>(i + 1);
    p.set_query(query);
    int32_t ids[] = {5};
    p.submit_batch(ids, 1);
    // Query should have been copied to task.query_vec
    for (uint32_t i = 0; i < DIM; i++)
        EXPECT_FLOAT_EQ(mc.task.query_vec[i], static_cast<float>(i + 1));
}

TEST(PushdownProtocol, SubmitWritesIdsToGd)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    int32_t ids[] = {10, 20, 30};
    p.submit_batch(ids, 3);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(mc.task.neighbor_ids[i], ids[i]);
}

TEST(PushdownProtocol, SubmitDoorbellEncodesCount)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    int32_t ids[7] = {};
    p.submit_batch(ids, 7);
    uint64_t db = mc.task.h->req_seq.load(std::memory_order_acquire);
    EXPECT_EQ(doorbell_count(db), 7u);
    EXPECT_EQ(doorbell_seq(db), 1u);
}

TEST(PushdownProtocol, TryWaitReturnsFalseWhenServiceNotDone)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    p.submit_batch(nullptr, 0);
    EXPECT_FALSE(p.try_wait_batch());
}

TEST(PushdownProtocol, TryWaitReturnsTrueAfterServiceResponds)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    p.submit_batch(nullptr, 0);
    mc.mock_service_respond();
    EXPECT_TRUE(p.try_wait_batch());
}

TEST(PushdownProtocol, ConsumeAdvancesSeqToEven)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    int32_t ids[] = {1};
    p.submit_batch(ids, 1);
    mc.mock_service_respond();
    ASSERT_TRUE(p.try_wait_batch());
    p.consume_batch();
    EXPECT_EQ(p.debug_local_seq() & 1, 0u); // even = idle
}

TEST(PushdownProtocol, ConsumeReturnsDistancesPointer)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    int32_t ids[] = {1, 2};
    p.submit_batch(ids, 2);
    mc.mock_service_respond();
    ASSERT_TRUE(p.try_wait_batch());
    const float *dists = p.consume_batch();
    EXPECT_EQ(dists, mc.result.distances);
}

TEST(PushdownProtocol, ConsumeWithoutPendingRequestAborts)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    EXPECT_DEATH(p.consume_batch(), "consume_batch called without pending request");
}

TEST(PushdownProtocol, DoubleSubmitAborts)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    p.submit_batch(nullptr, 0);
    p.set_query(query);
    EXPECT_DEATH(p.submit_batch(nullptr, 0), "submit_batch called while request pending");
}

// ============================================================
// Roundtrip tests
// ============================================================

TEST(PushdownProtocol, SubmitWaitConsumeRoundtrip)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    for (uint32_t i = 0; i < DIM; i++)
        query[i] = static_cast<float>(i);

    int32_t ids[] = {100, 200, 300};
    p.set_query(query);
    p.submit_batch(ids, 3);
    mc.mock_service_respond();
    ASSERT_TRUE(p.try_wait_batch());
    const float *dists = p.consume_batch();

    // Verify distances were written by mock service
    EXPECT_FLOAT_EQ(dists[0], 0.0f);
    EXPECT_FLOAT_EQ(dists[1], 0.1f);
    EXPECT_FLOAT_EQ(dists[2], 0.2f);
}

TEST(PushdownProtocol, MultipleRoundtrips)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    for (int round = 0; round < 5; round++) {
        int32_t ids[] = {round};
        p.set_query(query);
        p.submit_batch(ids, 1);
        mc.mock_service_respond();
        ASSERT_TRUE(p.try_wait_batch());
        p.consume_batch();
    }
    // After 5 roundtrips: 5 submits with odd seqs 1,3,5,7,9
    // After last consume: seq = 10 (even)
    EXPECT_EQ(p.debug_local_seq(), 10u);
}

TEST(PushdownProtocol, CountZeroIsValid)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    p.submit_batch(nullptr, 0); // empty batch
    EXPECT_EQ(doorbell_count(mc.task.h->req_seq.load()), 0u);
    mc.mock_service_respond();
    ASSERT_TRUE(p.try_wait_batch());
    p.consume_batch();
    EXPECT_EQ(p.debug_local_seq(), 2u); // 0→1 (submit) → 2 (consume)
}

TEST(PushdownProtocol, SubmitAtCapacityBoundary)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    int32_t ids[MAX_BATCH];
    for (uint32_t i = 0; i < MAX_BATCH; i++)
        ids[i] = static_cast<int32_t>(i);
    p.set_query(query);
    p.submit_batch(ids, MAX_BATCH);
    EXPECT_EQ(doorbell_count(mc.task.h->req_seq.load()), MAX_BATCH);
}

TEST(PushdownProtocol, SubmitExceedsMaxBatchAborts)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    int32_t ids[MAX_BATCH + 1] = {};
    EXPECT_DEATH(p.submit_batch(ids, MAX_BATCH + 1), "count overflow");
}

TEST(PushdownProtocol, QueryCachedAcrossSubmits)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM];
    for (uint32_t i = 0; i < DIM; i++)
        query[i] = static_cast<float>(i * 3);

    p.set_query(query);
    // First submit copies query
    p.submit_batch(nullptr, 0);
    mc.mock_service_respond();
    p.try_wait_batch();
    p.consume_batch();

    // Second submit without set_query — must still have query in gd
    int32_t ids[] = {1};
    p.submit_batch(ids, 1);
    // query_vec should still hold the original values (not re-copied)
    for (uint32_t i = 0; i < DIM; i++)
        EXPECT_FLOAT_EQ(mc.task.query_vec[i], static_cast<float>(i * 3));
}

TEST(PushdownProtocol, QueryEpochIncrementsOnNewQuery)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float q1[DIM] = {};
    float q2[DIM] = {};
    p.set_query(q1);
    p.submit_batch(nullptr, 0);
    mc.mock_service_respond();
    p.try_wait_batch();
    p.consume_batch();

    uint32_t epoch1 = mc.task.h->query_epoch;

    p.set_query(q2);
    p.submit_batch(nullptr, 0);
    mc.mock_service_respond();
    p.try_wait_batch();
    p.consume_batch();

    uint32_t epoch2 = mc.task.h->query_epoch;
    EXPECT_GT(epoch2, epoch1);
}

TEST(PushdownProtocol, DebugDebugMethods)
{
    MockChannels mc;
    auto p = mc.make_proxy();
    float query[DIM] = {};
    p.set_query(query);
    p.submit_batch(nullptr, 0);
    EXPECT_EQ(p.debug_req_seq(), 1u);
    EXPECT_EQ(p.debug_resp_seq(), 0u); // not responded yet
    mc.mock_service_respond();
    EXPECT_EQ(p.debug_resp_seq(), 1u);
}

// Note: the split-only profiled submit_batch(ids, count, double&, double&,
// double&) overload (Phase F Tier B DEATH tests) was deleted in the api
// layer. The abort semantics it covered — double submit and count overflow —
// are already tested by DoubleSubmitAborts and SubmitExceedsMaxBatchAborts
// via the plain 2-arg overload above.
