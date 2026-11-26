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
#include "scheduler_test_support.hpp"
#include "test_framework.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include "hnsw_urma/runtime/scheduler.hpp"
#include "hnsw_urma/search/query_context.hpp"

TEST(ReferenceRuntimeScheduler, SchedulerOwnerIsFixedAcrossWaitAndRetry) {
  auto queues = hnsw_urma::SchedulerQueues::Create(2, 1, 2, 1).value();
  auto publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 2, 1).value();
  const hnsw_urma::QueryKey key{0, 0, 91};
  hnsw_urma::QueryRunController controller(key.generation, 1);
  UB_REQUIRE_TRUE(controller.BindOwner(1).ok());
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_FALSE(controller.BindOwner(0).ok());
  const hnsw_urma::LaneOperationCount count{0, 1};
  const auto epoch =
      controller.BeginWait(std::span(&count, 1), key.generation).value();
  UB_REQUIRE_TRUE(controller.CompleteOne(key.generation).value());
  UB_REQUIRE_TRUE(publisher->PublishReady(0, {key, epoch, 0, 0, 1}).ok());
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(
                       *queues, 0, std::chrono::steady_clock::now())
                       .has_value());
  auto ready = hnsw_urma::test::TakeRunnable(*queues, 1,
                                             std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(ready.has_value());
  UB_REQUIRE_TRUE(controller.ValidateReadyHandle({key, epoch, 0, 0, 1}));
  UB_REQUIRE_FALSE(queues->PublishReady(0, 0, {key, epoch, 0, 0, 1}).ok());
  // A retired QuerySlot may have had a different owner. Reject its stale
  // generation before asserting the live Query's routing invariant.
  auto stale_key = key;
  --stale_key.generation;
  UB_REQUIRE_FALSE(controller.ValidateReadyHandle({stale_key, epoch, 0, 0, 0}));
  UB_REQUIRE_TRUE(controller.TryAcquire());
  auto retry = controller.RequeueForRetry().value();
  UB_REQUIRE_TRUE(queues->ScheduleRetry(1, {key, retry, {}}).ok());
  UB_REQUIRE_TRUE(hnsw_urma::test::TakeRunnable(
                      *queues, 1, std::chrono::steady_clock::now())
                      .has_value());
  UB_REQUIRE_EQ(controller.owner_search(), 1U);
  UB_REQUIRE_EQ(
      hnsw_urma::ClassifyYieldedQueryState(hnsw_urma::RunState::kRunning),
      hnsw_urma::YieldedQueryDisposition::kInvalid);
}

TEST(ReferenceRuntimeScheduler, SchedulerRunsAnotherQueryWhileFirstQueryWaits) {
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 1, 2, 2).value();
  hnsw_urma::QueryRunController first(10, 1);
  hnsw_urma::QueryRunController second(20, 1);
  const hnsw_urma::QueryKey first_key{0, 0, 10};
  const hnsw_urma::QueryKey second_key{0, 1, 20};
  UB_REQUIRE_TRUE(first.Admit().ok());
  UB_REQUIRE_TRUE(second.Admit().ok());
  UB_REQUIRE_TRUE(queues->PublishAdmission(0, first_key).ok());
  UB_REQUIRE_TRUE(queues->PublishAdmission(0, second_key).ok());

  auto first_run = hnsw_urma::test::TakeRunnable(
      *queues, 0, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(first_run.has_value());
  UB_REQUIRE_EQ(first_run->query, first_key);
  UB_REQUIRE_TRUE(first.TryAcquire());
  const auto epoch = first.next_wait_epoch();
  hnsw_urma::QueryOperationState operation;
  const hnsw_urma::QueryOperationHandle operation_handle{0, 0, 10, epoch, 1};
  auto ready_publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 1, 1).value();
  hnsw_urma::QueryOperationBinding binding;
  UB_REQUIRE_TRUE(binding
                      .Prepare(first_key, first, operation, *ready_publisher,
                               operation_handle)
                      .ok());
  const hnsw_urma::LaneOperationCount count{0, 1};
  UB_REQUIRE_TRUE(first.BeginWait(std::span(&count, 1), 10).ok());
  UB_REQUIRE_TRUE(binding.MarkPublished().ok());

  // The worker returns to its loop instead of polling the first query.
  auto second_run = hnsw_urma::test::TakeRunnable(
      *queues, 0, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(second_run.has_value());
  UB_REQUIRE_EQ(second_run->query, second_key);
  UB_REQUIRE_TRUE(second.TryAcquire());
  UB_REQUIRE_TRUE(second.MarkCompleted().ok());
  UB_REQUIRE_TRUE(
      queues->PublishDone(0, {second_key, hnsw_urma::RunState::kCompleted})
          .ok());
  UB_REQUIRE_EQ(queues->PopDone()->query, second_key);

  UB_REQUIRE_TRUE(
      binding.Complete(hnsw_urma::OperationTerminalOutcome::kComplete).ok());
  auto resumed = hnsw_urma::test::TakeRunnable(
      *queues, 0, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(resumed.has_value());
  UB_REQUIRE_EQ(resumed->source, hnsw_urma::RunnableSource::kReady);
  UB_REQUIRE_EQ(resumed->epoch, epoch);
  UB_REQUIRE_TRUE(first.ValidateReadyHandle({resumed->query, resumed->epoch}));
  UB_REQUIRE_TRUE(first.TryAcquire());
  UB_REQUIRE_TRUE(first.MarkCompleted().ok());
  UB_REQUIRE_TRUE(
      queues->PublishDone(0, {first_key, hnsw_urma::RunState::kCompleted})
          .ok());
  UB_REQUIRE_EQ(queues->PopDone()->query, first_key);
}

TEST(ReferenceRuntimeScheduler,
     FinalOwnerLanePublishesTheOnlyReadyHandleFromItsOwnLane) {
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 2, 2, 2).value();
  auto ready_publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 1, 2).value();
  const hnsw_urma::QueryKey key{0, 1, 41};
  hnsw_urma::QueryRunController controller(key.generation, 2);
  hnsw_urma::QueryOperationState operations[2];
  hnsw_urma::QueryOperationBinding bindings[2];
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  const auto epoch = controller.next_wait_epoch();
  const hnsw_urma::QueryOperationHandle handles[]{
      {0, 0, key.generation, epoch, 101},
      {1, 1, key.generation, epoch, 202},
  };
  for (std::uint32_t lane = 0; lane < 2; ++lane) {
    UB_REQUIRE_TRUE(bindings[lane]
                        .Prepare(key, controller, operations[lane],
                                 *ready_publisher, handles[lane])
                        .ok());
  }
  const hnsw_urma::LaneOperationCount counts[]{{0, 1}, {1, 1}};
  UB_REQUIRE_EQ(controller.BeginWait(counts, key.generation).value(), epoch);
  UB_REQUIRE_TRUE(bindings[0].MarkPublished().ok());
  UB_REQUIRE_TRUE(bindings[1].MarkPublished().ok());

  // Lane 0 completes first but cannot publish. Lane 1 is the final owner and
  // therefore produces the only ready handle to SearchEvents[0].
  UB_REQUIRE_TRUE(bindings[0]
                      .Complete(hnsw_urma::OperationTerminalOutcome::kComplete)
                      .ok());
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(
                       *queues, 0, std::chrono::steady_clock::now())
                       .has_value());
  UB_REQUIRE_TRUE(bindings[1]
                      .Complete(hnsw_urma::OperationTerminalOutcome::kComplete)
                      .ok());
  auto runnable = hnsw_urma::test::TakeRunnable(
      *queues, 0, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(runnable.has_value());
  UB_REQUIRE_EQ(runnable->query, key);
  UB_REQUIRE_EQ(runnable->epoch, epoch);
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(
                       *queues, 0, std::chrono::steady_clock::now())
                       .has_value());
}

TEST(ReferenceRuntimeScheduler,
     DeferredReadyWaitsForInlineRunnerBeforePublishing) {
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 1, 1, 1).value();
  auto ready_publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 1, 1).value();
  const hnsw_urma::QueryKey key{0, 0, 73};
  hnsw_urma::QueryRunController controller(key.generation, 1);
  hnsw_urma::QueryOperationState operation;
  hnsw_urma::QueryOperationBinding binding;
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_TRUE(controller.BeginDeferredReady().ok());
  const auto epoch = controller.next_wait_epoch();
  const hnsw_urma::QueryOperationHandle handle{0, 0, key.generation, epoch, 1};
  UB_REQUIRE_TRUE(
      binding.Prepare(key, controller, operation, *ready_publisher, handle)
          .ok());
  const hnsw_urma::LaneOperationCount count{0, 1};
  UB_REQUIRE_EQ(
      controller.BeginWait(std::span(&count, 1), key.generation).value(),
      epoch);
  UB_REQUIRE_TRUE(binding.MarkPublished().ok());
  UB_REQUIRE_TRUE(
      binding.Complete(hnsw_urma::OperationTerminalOutcome::kComplete).ok());
  UB_REQUIRE_EQ(controller.state(), hnsw_urma::RunState::kWaiting);
  UB_REQUIRE_EQ(ready_publisher->Flush(0, 1).value(), 0U);
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(
                       *queues, 0, std::chrono::steady_clock::now())
                       .has_value());

  controller.EndDeferredReady();
  UB_REQUIRE_EQ(ready_publisher->Flush(0, 1).value(), 1U);
  auto runnable = hnsw_urma::test::TakeRunnable(
      *queues, 0, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(runnable.has_value());
  UB_REQUIRE_EQ(runnable->query, key);
  UB_REQUIRE_EQ(runnable->epoch, epoch);
  UB_REQUIRE_TRUE(controller.TryAcquire());
  UB_REQUIRE_TRUE(controller.MarkCompleted().ok());
}

TEST(ReferenceRuntimeScheduler, SchedulerCapacityCoversEveryActiveQuery) {
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 1, 2, 1).value();
  const hnsw_urma::QueryReadyHandle first{{0, 0, 1}, 1};
  const hnsw_urma::QueryReadyHandle second{{0, 1, 2}, 1};
  UB_REQUIRE_TRUE(queues->PublishReady(0, 0, first).ok());
  UB_REQUIRE_TRUE(queues->PublishReady(0, 0, second).ok());
  UB_REQUIRE_EQ(queues->metrics().ready_published, 2U);
  UB_REQUIRE_EQ(queues->metrics().ready_queue_high_watermark, 2U);
  UB_REQUIRE_EQ(hnsw_urma::test::TakeRunnable(*queues, 0,
                                              std::chrono::steady_clock::now())
                    ->query,
                first.query);
  UB_REQUIRE_EQ(queues->metrics().ready_consumed, 1U);
  UB_REQUIRE_EQ(hnsw_urma::test::TakeRunnable(*queues, 0,
                                              std::chrono::steady_clock::now())
                    ->query,
                second.query);
}

TEST(ReferenceRuntimeScheduler, SchedulerProfilesReadyStagesAndWaitEvents) {
  auto queues = hnsw_urma::SchedulerQueues::Create(2, 2, 4, 4, true).value();
  const auto ready_at = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  const hnsw_urma::QueryReadyHandle handle{{0, 1, 7}, 3, ready_at, 0};
  UB_REQUIRE_TRUE(
      queues
          ->PublishReady(0, 1,
                         {handle.query, handle.wait_epoch,
                          handle.ready_timestamp_nanoseconds,
                          handle.published_timestamp_nanoseconds, 1})
          .ok());
  auto runnable = hnsw_urma::test::TakeRunnable(
      *queues, 1, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(runnable.has_value());
  UB_REQUIRE_TRUE(runnable->ready_published_nanoseconds >= ready_at);
  UB_REQUIRE_TRUE(runnable->ready_popped_nanoseconds >=
                  runnable->ready_published_nanoseconds);
  queues->RecordReadyStep(1, *runnable,
                          runnable->ready_popped_nanoseconds + 500);
  queues->RecordReadyRun(1, 700);
  queues->RecordWaitProfile({hnsw_urma::QueryWaitProfileKind::kLayer0Distance,
                             10'000, 1'000, 2'000, 7'000, 3, 64, 1});

  const auto metrics = queues->metrics();
  UB_REQUIRE_EQ(metrics.ready_published_per_worker,
                std::vector<std::uint64_t>({0, 1}));
  UB_REQUIRE_EQ(metrics.ready_consumed_per_worker,
                std::vector<std::uint64_t>({0, 1}));
  UB_REQUIRE_EQ(metrics.ready_lane_depth_sum_per_worker[1], 1U);
  UB_REQUIRE_EQ(metrics.ready_producer_worker_counts[1], 1U);
  UB_REQUIRE_EQ(metrics.ready_pop_to_step_ns_per_worker[1], 500U);
  UB_REQUIRE_EQ(metrics.ready_run_ns_per_worker[1], 700U);
  const auto kind = static_cast<std::size_t>(
      hnsw_urma::QueryWaitProfileKind::kLayer0Distance);
  UB_REQUIRE_EQ(metrics.wait_event_counts[kind], 1U);
  UB_REQUIRE_EQ(metrics.wait_nanoseconds[kind], 10'000U);
  UB_REQUIRE_EQ(metrics.wait_publish_nanoseconds[kind], 1'000U);
  UB_REQUIRE_EQ(metrics.wait_first_completion_nanoseconds[kind], 2'000U);
  UB_REQUIRE_EQ(metrics.wait_completion_spread_nanoseconds[kind], 7'000U);
  UB_REQUIRE_EQ(metrics.wait_remote_operations[kind], 3U);
  UB_REQUIRE_EQ(metrics.wait_distance_values[kind], 64U);
  UB_REQUIRE_EQ(metrics.wait_critical_lane_counts[kind * 2 + 1], 1U);
}

TEST(ReferenceRuntimeScheduler,
     SchedulerRetryHeapUsesDeadlineAndIndependentEpoch) {
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 1, 2, 2).value();
  const auto now = std::chrono::steady_clock::now();
  const hnsw_urma::QueryRetryHandle later{
      {0, 0, 1}, 1, now + std::chrono::seconds(2)};
  const hnsw_urma::QueryRetryHandle sooner{
      {0, 1, 2}, 3, now + std::chrono::seconds(1)};
  UB_REQUIRE_TRUE(queues->ScheduleRetry(0, later).ok());
  UB_REQUIRE_TRUE(queues->ScheduleRetry(0, sooner).ok());
  UB_REQUIRE_EQ(*queues->NextRetryDeadline(0), sooner.retry_deadline);
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(*queues, 0, now).has_value());
  auto runnable =
      hnsw_urma::test::TakeRunnable(*queues, 0, now + std::chrono::seconds(1));
  UB_REQUIRE_TRUE(runnable.has_value());
  UB_REQUIRE_EQ(runnable->query, sooner.query);
  UB_REQUIRE_EQ(runnable->source, hnsw_urma::RunnableSource::kRetry);
  UB_REQUIRE_EQ(runnable->epoch, sooner.retry_epoch);
}

TEST(ReferenceRuntimeScheduler,
     QueryDeadlineHeapExpiresActivelyAndRemovesCompletedQueries) {
  auto deadlines = hnsw_urma::QueryDeadlineHeap::Create(2).value();
  const auto now = std::chrono::steady_clock::now();
  const hnsw_urma::QueryDeadlineEntry later{{0, 0, 1},
                                            now + std::chrono::seconds(2)};
  const hnsw_urma::QueryDeadlineEntry sooner{{0, 1, 1},
                                             now + std::chrono::seconds(1)};
  UB_REQUIRE_TRUE(deadlines.Schedule(later).ok());
  UB_REQUIRE_TRUE(deadlines.Schedule(sooner).ok());
  UB_REQUIRE_EQ(*deadlines.next_deadline(), sooner.deadline);
  UB_REQUIRE_FALSE(deadlines.PopExpired(now).has_value());
  UB_REQUIRE_EQ(deadlines.PopExpired(sooner.deadline)->query, sooner.query);
  UB_REQUIRE_TRUE(deadlines.Cancel(later.query));
  UB_REQUIRE_EQ(deadlines.size(), 0U);
}

TEST(ReferenceRuntimeScheduler,
     SchedulerMetricsSupportConcurrentLaneAndWorkerOwners) {
  auto queues = hnsw_urma::SchedulerQueues::Create(2, 2, 8, 1, true).value();
  std::atomic<bool> start{false};
  std::atomic<std::uint32_t> failures{0};
  const auto await_start = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  };
  std::array<std::thread, 4> owners{
      std::thread([&] {
        await_start();
        if (!queues->PublishReady(0, 0, {{0, 0, 1}, 1}).ok() ||
            !queues->PublishReady(0, 0, {{0, 1, 1}, 1}).ok()) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }),
      std::thread([&] {
        await_start();
        if (!queues->PublishReady(1, 1, {{1, 0, 1}, 1, 0, 0, 1}).ok() ||
            !queues->PublishReady(1, 1, {{1, 1, 1}, 1, 0, 0, 1}).ok()) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }),
      std::thread([&] {
        await_start();
        if (!queues
                 ->ScheduleRetry(
                     0, {{2, 0, 1}, 1, std::chrono::steady_clock::now()})
                 .ok()) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }),
      std::thread([&] {
        await_start();
        if (!queues
                 ->ScheduleRetry(
                     1, {{2, 1, 1}, 1, std::chrono::steady_clock::now()})
                 .ok()) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      })};
  start.store(true, std::memory_order_release);
  for (auto &owner : owners) {
    owner.join();
  }
  UB_REQUIRE_EQ(failures.load(), 0U);
  const auto metrics = queues->metrics();
  UB_REQUIRE_EQ(metrics.retry_count, 2U);
  UB_REQUIRE_EQ(metrics.ready_published_per_worker,
                std::vector<std::uint64_t>({2, 2}));
  UB_REQUIRE_EQ(metrics.ready_producer_worker_counts,
                std::vector<std::uint64_t>({2, 0, 0, 2}));
}

TEST(ReferenceRuntimeScheduler,
     ReadyPublisherKeepsOwnerAcrossConcurrentProducerLanes) {
  auto queues = hnsw_urma::SchedulerQueues::Create(2, 2, 4, 4).value();
  auto publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 2, 2).value();
  const hnsw_urma::QueryKey query{0, 1, 9};
  std::atomic<bool> start{false};
  std::atomic<std::uint32_t> failures{0};
  const auto publish = [&](std::uint32_t lane, std::uint64_t wait_epoch) {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (!publisher->PublishReady(lane, {query, wait_epoch, 0, 0, 1}).ok()) {
      failures.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::thread first(publish, 0, 1);
  std::thread second(publish, 1, 2);
  start.store(true, std::memory_order_release);
  first.join();
  second.join();

  UB_REQUIRE_EQ(failures.load(std::memory_order_relaxed), 0U);
  auto first_worker = hnsw_urma::test::TakeRunnable(
      *queues, 1, std::chrono::steady_clock::now());
  auto second_worker = hnsw_urma::test::TakeRunnable(
      *queues, 1, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(first_worker.has_value());
  UB_REQUIRE_TRUE(second_worker.has_value());
  UB_REQUIRE_TRUE(first_worker->epoch != second_worker->epoch);
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(
                       *queues, 0, std::chrono::steady_clock::now())
                       .has_value());
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(
                       *queues, 1, std::chrono::steady_clock::now())
                       .has_value());
}

TEST(ReferenceRuntimeScheduler,
     SearchEventsKeepOwnerAndRetryCannotStarveReady) {
  auto queues = hnsw_urma::SchedulerQueues::Create(2, 1, 4, 1).value();
  auto publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 2, 1).value();
  const auto now = std::chrono::steady_clock::now();
  UB_REQUIRE_TRUE(publisher->PublishReady(0, {{0, 1, 1}, 1, 0, 0, 1}).ok());
  UB_REQUIRE_TRUE(publisher->PublishReady(0, {{0, 2, 1}, 1, 0, 0, 1}).ok());
  UB_REQUIRE_TRUE(queues->ScheduleRetry(1, {{0, 3, 1}, 1, now}).ok());
  UB_REQUIRE_EQ(hnsw_urma::test::TakeRunnable(*queues, 1, now)->source,
                hnsw_urma::RunnableSource::kReady);
  UB_REQUIRE_EQ(publisher->Flush(0, 8).value(), 0U);
  UB_REQUIRE_EQ(hnsw_urma::test::TakeRunnable(*queues, 1, now)->source,
                hnsw_urma::RunnableSource::kRetry);
  UB_REQUIRE_TRUE(queues->ScheduleRetry(1, {{0, 3, 1}, 2, now}).ok());
  UB_REQUIRE_EQ(hnsw_urma::test::TakeRunnable(*queues, 1, now)->source,
                hnsw_urma::RunnableSource::kReady);
  UB_REQUIRE_FALSE(hnsw_urma::test::TakeRunnable(*queues, 0, now).has_value());
}

TEST(ReferenceRuntimeScheduler,
     SearchEventRingHasLifecycleCapacityAndFailsClosedOnOverflow) {
  auto queues = hnsw_urma::SchedulerQueues::Create(32, 6, 1, 1).value();
  // A=1 still requires a two-cell sequence ring. Extra injected events violate
  // the Query lifecycle and must fail instead of disappearing in a backlog.
  UB_REQUIRE_TRUE(queues->PublishReady(0, 31, {{0, 0, 1}, 1, 0, 0, 31}).ok());
  UB_REQUIRE_TRUE(queues->PublishReady(1, 31, {{0, 0, 2}, 1, 0, 0, 31}).ok());
  UB_REQUIRE_EQ(queues->PublishReady(2, 31, {{0, 0, 3}, 1, 0, 0, 31}).code(),
                hnsw_urma::StatusCode::kInternalError);
}

TEST(ReferenceRuntimeScheduler,
     SearchInputRotatesAdmissionMixedEventsAndRetryWithBoundedFairness) {
  using namespace hnsw_urma;
  auto queues = SchedulerQueues::Create(1, 3, 4, 4, true, 2).value();
  const auto now = std::chrono::steady_clock::now();
  UB_REQUIRE_TRUE(queues->PublishAdmission(0, {0, 0, 1}).ok());
  UB_REQUIRE_TRUE(queues->PublishRpcReclaim(1, {0, 0, 100}).ok());
  UB_REQUIRE_TRUE(queues->PublishReady(2, 0, {{0, 1, 1}, 1}).ok());
  UB_REQUIRE_TRUE(queues->ScheduleRetry(0, {{0, 2, 1}, 1, now}).ok());
  UB_REQUIRE_EQ(
      std::get<QueryRunnableHandle>(*queues->PopSearchInput(0, now)).source,
      RunnableSource::kAdmission);
  UB_REQUIRE_TRUE(std::holds_alternative<RpcReclaimHandle>(
      *queues->PopSearchInput(0, now)));
  UB_REQUIRE_EQ(
      std::get<QueryRunnableHandle>(*queues->PopSearchInput(0, now)).source,
      RunnableSource::kRetry);
  UB_REQUIRE_EQ(
      std::get<QueryRunnableHandle>(*queues->PopSearchInput(0, now)).source,
      RunnableSource::kReady);
  auto metrics = queues->metrics();
  UB_REQUIRE_EQ(metrics.rpc_reclaim_consumed, 1U);
  UB_REQUIRE_EQ(metrics.ready_consumed, 1U);
  UB_REQUIRE_EQ(metrics.ready_producer_worker_counts,
                std::vector<std::uint64_t>({0, 0, 1}));
}
