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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "hnsw_urma/search/query_context.hpp"
#include "hnsw_urma/search/query_slot_pool.hpp"

using hnsw_urma::QueryRunController;
using hnsw_urma::RunState;
using hnsw_urma::StatusCode;

TEST(ReferenceMockQueryState, QueryRunControllerPublishesExactlyOneWakeup) {
  QueryRunController controller(77);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_TRUE(controller.Park(64).ok());

  std::atomic<std::uint32_t> wakeups{0};
  std::atomic<std::uint32_t> completion_errors{0};
  std::vector<std::thread> threads;
  threads.reserve(64);
  for (std::uint32_t index = 0; index < 64; ++index) {
    threads.emplace_back([&] {
      auto completed = controller.CompleteOne(77);
      if (!completed.ok()) {
        completion_errors.fetch_add(1, std::memory_order_relaxed);
      } else if (completed.value()) {
        wakeups.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  UB_REQUIRE_EQ(wakeups.load(), 1U);
  UB_REQUIRE_EQ(completion_errors.load(), 0U);
  UB_REQUIRE_EQ(controller.pending(), 0U);
  UB_REQUIRE_EQ(controller.state(), RunState::kQueued);
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_TRUE(controller.MarkCompleted().ok());
}

TEST(ReferenceMockQueryState,
     QueryRunControllerRejectsGenerationMismatchAndDuplicate) {
  QueryRunController controller(9);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_TRUE(controller.Park(1).ok());
  auto stale = controller.CompleteOne(8);
  UB_REQUIRE_FALSE(stale.ok());
  UB_REQUIRE_EQ(controller.pending(), 1U);
  UB_REQUIRE_TRUE(controller.CompleteOne(9).value());
  auto duplicate = controller.CompleteOne(9);
  UB_REQUIRE_FALSE(duplicate.ok());
  UB_REQUIRE_EQ(duplicate.status().code(), StatusCode::kProtocolError);
}

TEST(ReferenceMockQueryState,
     QueryRunControllerPublishesReadyTimestampWhenTimingIsEnabled) {
  QueryRunController controller(10, 1, true);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_TRUE(controller.Park(1).ok());
  UB_REQUIRE_TRUE(controller.CompleteOne(10).value());
  UB_REQUIRE_TRUE(controller.TakeReadyTimestampNanoseconds() != 0U);
  UB_REQUIRE_EQ(controller.TakeReadyTimestampNanoseconds(), 0U);
}

TEST(ReferenceMockQueryState,
     QueryRunControllerConvergesLanesAndRejectsStaleWaitHandles) {
  QueryRunController controller(41, 2);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  const hnsw_urma::LaneOperationCount lanes[]{{0, 2}, {1, 1}};
  const auto expected_epoch = controller.next_wait_epoch();
  hnsw_urma::QueryOperationState operations[3];
  const hnsw_urma::QueryOperationHandle handles[]{
      {0, 0, 41, expected_epoch, 101},
      {1, 0, 41, expected_epoch, 102},
      {2, 1, 41, expected_epoch, 103},
  };
  for (std::size_t index = 0; index < 3; ++index) {
    UB_REQUIRE_TRUE(operations[index].Prepare(handles[index]).ok());
  }
  auto wait = controller.BeginWait(lanes, 41);
  UB_REQUIRE_TRUE(wait.ok());
  UB_REQUIRE_EQ(wait.value(), expected_epoch);
  for (std::size_t index = 0; index < 3; ++index) {
    UB_REQUIRE_TRUE(operations[index].MarkPublished(handles[index]).ok());
  }

  std::atomic<std::uint32_t> wakeups{0};
  std::thread lane0([&] {
    UB_REQUIRE_FALSE(
        controller
            .OnOperationTerminal(operations[0], handles[0],
                                 hnsw_urma::OperationTerminalOutcome::kComplete)
            .value());
    if (controller
            .OnOperationTerminal(operations[1], handles[1],
                                 hnsw_urma::OperationTerminalOutcome::kComplete)
            .value()) {
      wakeups.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::thread lane1([&] {
    if (controller
            .OnOperationTerminal(operations[2], handles[2],
                                 hnsw_urma::OperationTerminalOutcome::kComplete)
            .value()) {
      wakeups.fetch_add(1, std::memory_order_relaxed);
    }
  });
  lane0.join();
  lane1.join();
  UB_REQUIRE_EQ(wakeups.load(), 1U);
  UB_REQUIRE_EQ(controller.remaining_lanes(), 0U);
  UB_REQUIRE_EQ(controller.pending(), 0U);
  UB_REQUIRE_EQ(controller.state(), RunState::kQueued);
  UB_REQUIRE_TRUE(controller.ValidateReadyHandle({{0, 9, 41}, expected_epoch}));
  UB_REQUIRE_FALSE(
      controller.ValidateReadyHandle({{0, 9, 41}, expected_epoch + 1U}));
  UB_REQUIRE_FALSE(
      controller
          .OnOperationTerminal(operations[2], handles[2],
                               hnsw_urma::OperationTerminalOutcome::kComplete)
          .value());
}

TEST(ReferenceMockQueryState,
     QueryRunControllerKeepsReservationRetrySeparateFromWaitEpoch) {
  QueryRunController controller(55, 2);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  auto retry = controller.RequeueForRetry();
  UB_REQUIRE_TRUE(retry.ok());
  UB_REQUIRE_EQ(controller.wait_epoch(), 0U);
  UB_REQUIRE_TRUE(controller.ValidateRetryHandle(
      {{0, 3, 55}, retry.value(), std::chrono::steady_clock::now()}));
  UB_REQUIRE_FALSE(controller.ValidateRetryHandle(
      {{0, 3, 55}, retry.value() + 1U, std::chrono::steady_clock::now()}));
  UB_REQUIRE_TRUE(controller.TryAcquire());
}

TEST(ReferenceMockQueryState,
     QuerySlotPoolBoundsAdmissionAndAdvancesGeneration) {
  auto pool = hnsw_urma::QuerySlotPool::Create(3, 1).value();
  auto first = pool.Acquire();
  UB_REQUIRE_TRUE(first.ok());
  UB_REQUIRE_FALSE(pool.Acquire().ok());
  UB_REQUIRE_FALSE(pool.Release({3, 0, first.value().generation + 1}).ok());
  UB_REQUIRE_TRUE(pool.Release(first.value()).ok());
  auto second = pool.Acquire();
  UB_REQUIRE_TRUE(second.ok());
  UB_REQUIRE_TRUE(second.value().generation > first.value().generation);
  UB_REQUIRE_FALSE(pool.Release(first.value()).ok());
  UB_REQUIRE_TRUE(pool.Release(second.value()).ok());
}
