/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/spsc_ring.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/runtime/thread_group.hpp"
#include "hnsw_urma/transport/transport.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>
using namespace hnsw_urma;

TEST(CoreExtended, CheckedArithmeticBoundaries) {
  using U = std::uint64_t;
  const U max = std::numeric_limits<U>::max();
  EXPECT_EQ(CheckedAdd(max, U{1}).status().code(), StatusCode::kOutOfRange);
  EXPECT_EQ(CheckedMultiply(max, U{2}).status().code(),
            StatusCode::kOutOfRange);
  ASSERT_TRUE(CheckedMultiply(max, U{0}).ok());
  EXPECT_EQ(CheckedMultiply(max, U{0}).value(), 0U);
  for (U alignment : {U{0}, U{3}, U{7}})
    EXPECT_FALSE(CheckedAlignUp(U{5}, alignment).ok());
  EXPECT_FALSE(CheckedAlignUp(max, U{8}).ok());
  EXPECT_EQ(CheckedAlignUp(U{4097}, U{4096}).value(), 8192U);
  EXPECT_TRUE(IsAligned(U{4096}, U{4096}));
  EXPECT_FALSE(IsAligned(U{4097}, U{4096}));
}
TEST(CoreExtended, QueryHashIsStableAndUsesFullIdentity) {
  const QueryKey base{1, 2, 3};
  EXPECT_EQ(StableHash(base, 4), StableHash(base, 4));
  const std::array<QueryKey, 3> changed{{{2, 2, 3}, {1, 3, 3}, {1, 2, 4}}};
  for (const auto &value : changed)
    EXPECT_NE(StableHash(base, 4), StableHash(value, 4));
  EXPECT_NE(StableHash(base, 4), StableHash(base, 5));
  EXPECT_EQ(StableHash(1U, 2U), StableHash(1U, 2U));
}
TEST(CoreExtended, SpscReservationsBackpressureAndWraparound) {
  EXPECT_FALSE(SpscRing<int>::Create(0).ok());
  auto created = SpscRing<int>::Create(2);
  ASSERT_TRUE(created.ok());
  auto ring = std::move(created).value();
  EXPECT_FALSE(ring.TryPop().has_value());
  EXPECT_FALSE(ring.Reserve(0).ok());
  EXPECT_FALSE(ring.PushReserved(1).ok());
  ASSERT_TRUE(ring.Reserve(2).ok());
  EXPECT_EQ(ring.producer_reserved(), 2U);
  EXPECT_FALSE(ring.Reserve(1).ok());
  EXPECT_FALSE(ring.CancelReservation(3).ok());
  ASSERT_TRUE(ring.CancelReservation(1).ok());
  ASSERT_TRUE(ring.PushReserved(10).ok());
  ASSERT_TRUE(ring.TryPush(20).ok());
  EXPECT_FALSE(ring.TryPush(30).ok());
  EXPECT_EQ(ring.TryPop().value(), 10);
  ASSERT_TRUE(ring.TryPush(30).ok());
  EXPECT_EQ(ring.TryPop().value(), 20);
  EXPECT_EQ(ring.TryPop().value(), 30);
  EXPECT_FALSE(ring.TryPop());
  EXPECT_EQ(ring.metrics().high_watermark, 2U);
  EXPECT_GT(ring.metrics().full_count, 0U);
}
TEST(CoreExtended, SpscMovePreservesPublishedItems) {
  auto original = SpscRing<int>::Create(3).value();
  ASSERT_TRUE(original.TryPush(7).ok());
  auto moved = std::move(original);
  EXPECT_EQ(moved.TryPop().value(), 7);
  auto destination = SpscRing<int>::Create(1).value();
  destination = std::move(moved);
  EXPECT_EQ(destination.capacity(), 3U);
  EXPECT_FALSE(destination.TryPop());
}
TEST(CoreExtended, SpscThreadedProducerConsumerPreserveOrder) {
  auto ring = SpscRing<std::uint32_t>::Create(17).value();
  constexpr std::uint32_t count = 2000;
  std::atomic<bool> wrong{false};
  std::jthread producer([&](std::stop_token stop) {
    for (std::uint32_t value = 0; value < count && !stop.stop_requested();
         ++value)
      while (!ring.TryPush(value).ok()) {
        if (stop.stop_requested())
          return;
        std::this_thread::yield();
      }
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  std::uint32_t next = 0;
  while (next < count && std::chrono::steady_clock::now() < deadline) {
    auto value = ring.TryPop();
    if (value) {
      if (*value != next)
        wrong.store(true);
      ++next;
    } else
      std::this_thread::yield();
  }
  producer.request_stop();
  producer.join();
  EXPECT_EQ(next, count);
  EXPECT_FALSE(wrong.load());
  EXPECT_EQ(ring.size(), 0U);
}
TEST(CoreExtended,
     DistanceMatchesIndependentDoubleReferenceAcrossTailDimensions) {
  for (std::size_t dim : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 31U, 128U, 129U}) {
    SCOPED_TRACE(dim);
    std::vector<float> a(dim), b(dim);
    double l2 = 0, dot = 0;
    for (std::size_t i = 0; i < dim; ++i) {
      a[i] = static_cast<float>(i % 7) * 0.25F;
      b[i] = -static_cast<float>(i % 5) * 0.5F;
      const double delta = double(a[i]) - b[i];
      l2 += delta * delta;
      dot += double(a[i]) * b[i];
    }
    for (auto implementation :
         {DistanceImplementation::kScalar, DistanceImplementation::kSimd,
          DistanceImplementation::kAuto}) {
      const float l = ComputeDistance(a, b, Metric::kL2, implementation);
      const float d =
          ComputeDistance(a, b, Metric::kInnerProduct, implementation);
      ASSERT_TRUE(std::isfinite(l));
      ASSERT_TRUE(std::isfinite(d));
      EXPECT_NEAR(l, l2, 1e-4 * dim);
      EXPECT_NEAR(d, -dot, 1e-4 * dim);
    }
  }
}
TEST(CoreExtended, DistanceInvalidBatchShapesAndNaN) {
  const std::array<float, 2> query{1, 2};
  const std::array<float, 4> rows{1, 2, 4, 6};
  std::array<float, 2> output{};
  EXPECT_FALSE(ComputeDistanceBatch({}, rows, 2, Metric::kL2, output).ok());
  EXPECT_FALSE(ComputeDistanceBatch(query, rows, 1, Metric::kL2, output).ok());
  EXPECT_FALSE(ComputeDistanceBatch(query, std::span(rows).first(3), 2,
                                    Metric::kL2, output)
                   .ok());
  EXPECT_TRUE(std::isnan(ScalarL2(query, std::span(query).first(1))));
  EXPECT_TRUE(std::isnan(ScalarDot(query, std::span(query).first(1))));
}
TEST(CoreExtended, FailedResultValueIsADeathContract) {
  EXPECT_DEATH(
      {
        Result<int> value(Status(StatusCode::kInvalidArgument, "bad"));
        (void)value.value();
      },
      "");
}
TEST(CoreExtended, BoundedQueueFullCloseAndStop) {
  BoundedQueue<int> queue(1);
  ASSERT_TRUE(queue.TryPush(7).ok());
  EXPECT_FALSE(queue.TryPush(8).ok());
  EXPECT_EQ(queue.size(), 1U);
  EXPECT_EQ(queue.high_watermark(), 1U);
  EXPECT_EQ(queue.Pop(std::stop_token{}).value(), 7);
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_FALSE(queue.Pop(stopped.get_token()).ok());
  queue.Close();
  EXPECT_FALSE(queue.TryPush(9).ok());
  EXPECT_FALSE(queue.Pop(std::stop_token{}).ok());
}
TEST(CoreExtended, CountingNotificationSupportsDeadlinesAndStop) {
  CountingNotification notification;
  notification.Notify(0);
  EXPECT_EQ(notification.pending(), 0U);
  EXPECT_FALSE(notification.TryConsume());
  EXPECT_FALSE(notification.WaitUntil({}, std::chrono::steady_clock::now()));
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_FALSE(notification.Wait(stopped.get_token()));
  notification.Notify(2);
  EXPECT_TRUE(notification.WaitUntil({}, std::chrono::steady_clock::now() +
                                             std::chrono::seconds(1)));
  EXPECT_TRUE(notification.Wait({}));
  EXPECT_EQ(notification.pending(), 0U);
  EXPECT_EQ(notification.notifications(), 2U);
  EXPECT_GT(notification.empty_wakeups(), 0U);
}
namespace {
EffectiveRuntimeConfig ThreadConfig() {
  EffectiveRuntimeConfig config;
  config.search_workers = 1;
  config.distance_workers = 1;
  config.read_progress_threads = 1;
  config.rpc_progress_threads = 1;
  config.search_cpus = {0};
  config.distance_cpus = {1};
  config.read_progress_cpus = {2};
  config.rpc_progress_cpus = {3};
  config.reserved_cpus = {4};
  return config;
}
} // namespace
TEST(CoreExtended, RuntimeThreadsStartStopAndJoinAllRoles) {
  std::atomic<unsigned> entered{0};
  const RuntimeLoop loop = [&](std::stop_token stop, std::uint32_t) {
    entered.fetch_add(1);
    while (!stop.stop_requested())
      std::this_thread::yield();
    return Status::Ok();
  };
  RuntimeCallbacks callbacks{loop, loop, loop, loop, loop};
  auto created =
      RuntimeThreadGroup::Start(ThreadConfig(), CpuBindMode::kNone, callbacks);
  ASSERT_TRUE(created.ok());
  auto group = std::move(created).value();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (entered.load() != 5 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  group->RequestStop();
  group->Join();
  EXPECT_EQ(entered.load(), 5U);
  EXPECT_TRUE(group->status().ok());
}
TEST(CoreExtended, RuntimeThreadFailureStopsTheGroupAndPreservesRole) {
  const RuntimeLoop wait = [](std::stop_token stop, std::uint32_t) {
    while (!stop.stop_requested())
      std::this_thread::yield();
    return Status::Ok();
  };
  RuntimeCallbacks callbacks{wait,
                             [](std::stop_token, std::uint32_t) {
                               return Status(StatusCode::kInternalError,
                                             "injected search failure");
                             },
                             wait, wait, wait};
  auto created =
      RuntimeThreadGroup::Start(ThreadConfig(), CpuBindMode::kNone, callbacks);
  ASSERT_TRUE(created.ok());
  auto group = std::move(created).value();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (group->status().ok() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  group->RequestStop();
  group->Join();
  EXPECT_EQ(group->status().code(), StatusCode::kInternalError);
  EXPECT_NE(group->status().ToString().find("search"), std::string::npos);
}
TEST(CoreExtended, RuntimeThreadsRejectIncompleteCallbacksAndCpuAssignments) {
  EXPECT_FALSE(
      RuntimeThreadGroup::Start(ThreadConfig(), CpuBindMode::kNone, {}).ok());
  RuntimeLoop done = [](std::stop_token, std::uint32_t) {
    return Status::Ok();
  };
  RuntimeCallbacks callbacks{done, done, done, done, done};
  auto bad = ThreadConfig();
  bad.search_cpus.clear();
  EXPECT_FALSE(
      RuntimeThreadGroup::Start(bad, CpuBindMode::kNone, callbacks).ok());
  EXPECT_FALSE(BindCurrentThread(UINT32_MAX).ok());
}
TEST(CoreExtended, FirstTouchWritesPagesAndTailWithoutOverrun) {
  std::vector<std::byte> bytes(8194, std::byte{0x7f});
  FirstTouch(bytes);
  EXPECT_EQ(bytes[0], std::byte{0});
  EXPECT_EQ(bytes[4096], std::byte{0});
  EXPECT_EQ(bytes[8192], std::byte{0});
  EXPECT_EQ(bytes.back(), std::byte{0});
  EXPECT_EQ(bytes[1], std::byte{0x7f});
  FirstTouch({});
}
TEST(CoreExtended, TransportDefaultBatchReportsAcceptedPrefix) {
  class Engine final : public LanePostEngine {
  public:
    int reads = 0, writes = 0;
    Status PostRead(const ReadOperation &) override {
      return ++reads == 2 ? Status(StatusCode::kResourceExhausted, "full")
                          : Status::Ok();
    }
    Status PostWriteNotify(const WriteNotifyOperation &) override {
      return ++writes == 2 ? Status(StatusCode::kTransportError, "failed")
                           : Status::Ok();
    }
    Result<std::size_t> Poll(std::span<TransportCompletion>) override {
      return std::size_t{0};
    }
  } engine;
  const std::array<ReadOperation, 3> reads{};
  auto read = engine.PostReads(reads);
  EXPECT_EQ(read.posted, 1U);
  EXPECT_EQ(read.post_calls, 2U);
  EXPECT_EQ(read.status.code(), StatusCode::kResourceExhausted);
  const std::array<WriteNotifyOperation, 3> writes{};
  auto write = engine.PostWriteNotifies(writes);
  EXPECT_EQ(write.posted, 1U);
  EXPECT_EQ(write.status.code(), StatusCode::kTransportError);
  EXPECT_TRUE(engine.PostReads({}).status.ok());
  EXPECT_TRUE(engine.PostWriteNotifies({}).status.ok());
}
