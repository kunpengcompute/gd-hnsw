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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "scheduler_test_support.hpp"
#include "test_framework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hnsw_urma/transport/mock_lane_post_engine.hpp"
#include "hnsw_urma/transport/read_batch_lane.hpp"

namespace {

hnsw_urma::OperationTicket Ticket(std::uint64_t value, std::uint32_t lane = 0) {
  return {value, static_cast<std::uint32_t>(value), lane, 1};
}

} // namespace

TEST(ReferenceMockMockLane, MockLaneCompletesReadsOutOfOrderByTicket) {
  auto engine = hnsw_urma::MockLanePostEngine::Create(0, 4).value();
  std::array<std::byte, 16> remote{};
  for (std::size_t index = 0; index < remote.size(); ++index) {
    remote[index] = static_cast<std::byte>(index);
  }
  UB_REQUIRE_TRUE(
      engine.RegisterSegment(1, hnsw_urma::SegmentKind::kLayer0Graph, remote)
          .ok());
  std::array<std::byte, 4> first{};
  std::array<std::byte, 4> second{};
  UB_REQUIRE_TRUE(
      engine
          .PostRead(
              {Ticket(1), 1, hnsw_urma::SegmentKind::kLayer0Graph, first, 0})
          .ok());
  UB_REQUIRE_TRUE(
      engine
          .PostRead(
              {Ticket(2), 1, hnsw_urma::SegmentKind::kLayer0Graph, second, 8})
          .ok());
  UB_REQUIRE_TRUE(engine.Progress(1).ok());
  UB_REQUIRE_TRUE(engine.Progress(0).ok());
  std::array<hnsw_urma::TransportCompletion, 4> completions{};
  auto count = engine.Poll(completions);
  UB_REQUIRE_EQ(count.value(), 2U);
  UB_REQUIRE_EQ(completions[0].ticket.value, 2U);
  UB_REQUIRE_EQ(completions[1].ticket.value, 1U);
  UB_REQUIRE_EQ(std::to_integer<unsigned>(second[0]), 8U);
}

TEST(ReferenceMockMockLane,
     MockLaneWriteNotifyPublishesPayloadThenNotifyAsOneEvent) {
  auto engine = hnsw_urma::MockLanePostEngine::Create(0, 4).value();
  std::array<std::byte, 64> requests{};
  alignas(8) std::array<std::byte, 16> notify{};
  UB_REQUIRE_TRUE(
      engine.RegisterSegment(1, hnsw_urma::SegmentKind::kRequests, requests)
          .ok());
  UB_REQUIRE_TRUE(
      engine.RegisterSegment(1, hnsw_urma::SegmentKind::kNotify, notify).ok());

  std::array<std::byte, 4> inline_source{std::byte{1}, std::byte{2},
                                         std::byte{3}, std::byte{4}};
  hnsw_urma::WriteNotifyOperation inline_write{
      hnsw_urma::OperationClass::kRequestWrite,
      Ticket(1),
      1,
      hnsw_urma::SegmentKind::kRequests,
      inline_source,
      8,
      hnsw_urma::SegmentKind::kNotify,
      0,
      77,
      true};
  UB_REQUIRE_TRUE(engine.PostWriteNotify(inline_write).ok());
  inline_source[0] = std::byte{9};
  UB_REQUIRE_TRUE(engine.Progress().ok());
  UB_REQUIRE_EQ(std::to_integer<unsigned>(requests[8]), 1U);
  std::uint64_t observed = 0;
  std::memcpy(&observed, notify.data(), sizeof(observed));
  UB_REQUIRE_EQ(observed, 77U);

  std::array<std::byte, 2> non_inline{std::byte{5}, std::byte{6}};
  hnsw_urma::WriteNotifyOperation write{
      hnsw_urma::OperationClass::kResponseWrite,
      Ticket(2),
      1,
      hnsw_urma::SegmentKind::kRequests,
      non_inline,
      16,
      hnsw_urma::SegmentKind::kNotify,
      8,
      88,
      false};
  UB_REQUIRE_TRUE(engine.PostWriteNotify(write).ok());
  non_inline[0] = std::byte{7};
  UB_REQUIRE_TRUE(engine.Progress().ok());
  UB_REQUIRE_EQ(std::to_integer<unsigned>(requests[16]), 7U);
}

TEST(ReferenceMockMockLane,
     MockLaneInjectsQueueFullPostFailureDropAndDuplicateCompletion) {
  auto engine = hnsw_urma::MockLanePostEngine::Create(2, 1).value();
  std::array<std::byte, 8> remote{};
  std::array<std::byte, 1> destination{};
  UB_REQUIRE_TRUE(
      engine.RegisterSegment(3, hnsw_urma::SegmentKind::kVectors, remote).ok());
  hnsw_urma::ReadOperation operation{
      Ticket(1, 2), 3, hnsw_urma::SegmentKind::kVectors, destination, 0};
  UB_REQUIRE_TRUE(engine.PostRead(operation).ok());
  UB_REQUIRE_FALSE(engine.PostRead(operation).ok());
  UB_REQUIRE_TRUE(engine.Drop().ok());
  engine.FailNextPost(hnsw_urma::Status(hnsw_urma::StatusCode::kTransportError,
                                        "injected post failure"));
  UB_REQUIRE_FALSE(engine.PostRead(operation).ok());
  UB_REQUIRE_FALSE(
      engine
          .PostRead({Ticket(2, 1), 3, hnsw_urma::SegmentKind::kVectors,
                     destination, 0})
          .ok());
  hnsw_urma::TransportCompletion duplicate{Ticket(9, 2),
                                           hnsw_urma::OperationClass::kRead,
                                           hnsw_urma::Status::Ok(), 1};
  engine.InjectCompletion(duplicate);
  engine.InjectCompletion(duplicate);
  std::array<hnsw_urma::TransportCompletion, 2> completions{};
  UB_REQUIRE_EQ(engine.Poll(completions).value(), 2U);
  UB_REQUIRE_EQ(completions[0].ticket, completions[1].ticket);
}

TEST(ReferenceMockMockLane, LanePostBatchFallbackReportsAcceptedPrefix) {
  auto engine = hnsw_urma::MockLanePostEngine::Create(0, 2).value();
  std::array<std::byte, 8> remote{};
  std::array<std::byte, 3> destinations{};
  UB_REQUIRE_TRUE(
      engine.RegisterSegment(1, hnsw_urma::SegmentKind::kVectors, remote).ok());
  std::array<hnsw_urma::ReadOperation, 3> reads{
      hnsw_urma::ReadOperation{Ticket(1), 1, hnsw_urma::SegmentKind::kVectors,
                               std::span(destinations).subspan(0, 1), 0},
      hnsw_urma::ReadOperation{Ticket(2), 1, hnsw_urma::SegmentKind::kVectors,
                               std::span(destinations).subspan(1, 1), 1},
      hnsw_urma::ReadOperation{Ticket(3), 1, hnsw_urma::SegmentKind::kVectors,
                               std::span(destinations).subspan(2, 1), 2}};
  auto result = engine.PostReads(reads);
  UB_REQUIRE_FALSE(result.status.ok());
  UB_REQUIRE_EQ(result.posted, 2U);
  UB_REQUIRE_EQ(engine.pending_events(), 2U);
}

#include "read_test_support.hpp"

namespace {
using hnsw_urma::test::ControlledReadEngine;
using hnsw_urma::test::DrainReads;
using hnsw_urma::test::ReadTestBatch;
using hnsw_urma::test::ReadTestPool;
} // namespace

TEST(ReferenceMockMockLane,
     SearchResultPoolReservesAllSlotsAndSubmitRollbackIsLocal) {
  ControlledReadEngine engine;
  ReadTestPool pool(0, 3);
  auto lane = hnsw_urma::ReadBatchLane::Create(
                  engine, 0, {.peer_count = 2, .submit_depth = 1})
                  .value();
  std::array<hnsw_urma::ReadResultView, 4> too_many{};
  UB_REQUIRE_FALSE(pool.pool->Acquire(too_many, 4).ok());
  UB_REQUIRE_EQ(pool.pool->available(), 3U);
  ReadTestBatch first(*pool.pool, {0, 1});
  ReadTestBatch second(*pool.pool, {2});
  UB_REQUIRE_TRUE(lane->PrepareBatch(first.batch).ok());
  UB_REQUIRE_FALSE(lane->PrepareBatch(second.batch).ok());
  UB_REQUIRE_TRUE(lane->CancelPreparedBatch(first.batch).ok());
  first.Consume();
  UB_REQUIRE_EQ(pool.pool->available(), 2U);
  UB_REQUIRE_TRUE(lane->PrepareBatch(second.batch).ok());
  UB_REQUIRE_TRUE(lane->CancelPreparedBatch(second.batch).ok());
  second.Consume();
  UB_REQUIRE_TRUE(pool.pool->drained());
}

TEST(ReferenceMockMockLane,
     MixedSearchQueriesSharePostAndCompleteIndependently) {
  ControlledReadEngine engine;
  ReadTestPool s0(0, 4), s1(1, 1);
  auto lane =
      hnsw_urma::ReadBatchLane::Create(
          engine, 0, {.search_workers = 2, .peer_count = 2, .wr_list_max = 5})
          .value();
  ReadTestBatch q1(*s0.pool, {0, 1}), q2(*s1.pool, {2}), q3(*s0.pool, {3, 4});
  q1.Publish(*lane);
  q2.Publish(*lane);
  q3.Publish(*lane);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_EQ(engine.posts.size(), 1U);
  UB_REQUIRE_EQ(engine.posts.front().size(), 5U);
  UB_REQUIRE_EQ(engine.pending[0].remote_offset, 0U);
  UB_REQUIRE_EQ(engine.pending[1].remote_offset, 2U);
  UB_REQUIRE_EQ(engine.pending[2].remote_offset, 3U);
  engine.Complete(1);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_EQ(q2.terminal_calls, 1U);
  UB_REQUIRE_EQ(q1.terminal_calls, 0U);
  UB_REQUIRE_EQ(q3.terminal_calls, 0U);
  const auto *old_address = q2.results[0].data;
  q2.Consume();
  UB_REQUIRE_EQ(s1.pool->available(), 1U);
  ReadTestBatch q4(*s1.pool, {5});
  UB_REQUIRE_TRUE(q4.results[0].data == old_address);
  q4.Publish(*lane);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  DrainReads(engine, *lane);
  for (auto *query : {&q1, &q3, &q4}) {
    UB_REQUIRE_EQ(query->terminal_calls, 1U);
    for (std::size_t i = 0; i < query->results.size(); ++i)
      UB_REQUIRE_EQ(
          query->results[i].data[0],
          static_cast<std::byte>(query->requests[i].remote_offset + 1));
    query->Consume();
  }
  UB_REQUIRE_TRUE(s0.pool->drained());
  UB_REQUIRE_TRUE(s1.pool->drained());
}

TEST(ReferenceMockMockLane,
     MixedBatchAcceptedPrefixRetriesWithoutLosingOrRepeatingReads) {
  ControlledReadEngine engine;
  engine.accept_limit = 3;
  engine.capacity = 3;
  ReadTestPool s0(0, 4), s1(1, 1);
  auto lane =
      hnsw_urma::ReadBatchLane::Create(
          engine, 0, {.search_workers = 2, .peer_count = 2, .wr_list_max = 5})
          .value();
  ReadTestBatch q1(*s0.pool, {0, 1}), q2(*s1.pool, {2}), q3(*s0.pool, {3, 4});
  q1.Publish(*lane);
  q2.Publish(*lane);
  q3.Publish(*lane);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_EQ(engine.posts.front().size(), 5U);
  UB_REQUIRE_EQ(engine.accepted.size(), 3U);
  UB_REQUIRE_EQ(q1.batch.posted_reads, 1U);
  UB_REQUIRE_EQ(q2.batch.posted_reads, 1U);
  UB_REQUIRE_EQ(q3.batch.posted_reads, 1U);
  DrainReads(engine, *lane);
  std::vector<std::uint64_t> offsets;
  for (const auto &operation : engine.accepted)
    offsets.push_back(operation.remote_offset);
  std::sort(offsets.begin(), offsets.end());
  UB_REQUIRE_EQ(offsets, std::vector<std::uint64_t>({0, 1, 2, 3, 4}));
  for (auto *query : {&q1, &q2, &q3}) {
    UB_REQUIRE_EQ(query->terminal_calls, 1U);
    query->Consume();
  }
  UB_REQUIRE_EQ(lane->profile().wr_completion_samples, 0U);
}

TEST(ReferenceMockMockLane,
     OneBatchInterleavesPeerSegmentRangesWithSmallWrBudget) {
  ControlledReadEngine engine;
  engine.accept_limit = 1;
  ReadTestPool pool(0, 6);
  auto lane =
      hnsw_urma::ReadBatchLane::Create(
          engine, 0, {.peer_count = 3, .in_flight_limit = 2, .wr_list_max = 6})
          .value();
  ReadTestBatch batch(*pool.pool, {0, 1, 2, 3, 4, 5});
  for (std::size_t i = 2; i < 4; ++i)
    batch.requests[i].peer_rank = 2;
  for (std::size_t i = 4; i < 6; ++i)
    batch.requests[i].remote_segment = hnsw_urma::SegmentKind::kLayer0Graph;
  batch.Publish(*lane);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  DrainReads(engine, *lane);
  std::vector<std::uint64_t> offsets;
  for (const auto &operation : engine.accepted)
    offsets.push_back(operation.remote_offset);
  std::sort(offsets.begin(), offsets.end());
  UB_REQUIRE_EQ(offsets, std::vector<std::uint64_t>({0, 1, 2, 3, 4, 5}));
  for (const auto &post : engine.posts)
    for (const auto &wr : post) {
      UB_REQUIRE_EQ(wr.peer_rank, post.front().peer_rank);
      UB_REQUIRE_EQ(wr.remote_segment, post.front().remote_segment);
    }
  UB_REQUIRE_EQ(batch.terminal_calls, 1U);
  batch.Consume();
}

TEST(ReferenceMockMockLane,
     CancelledMixedListQueryRetainsDmaAddressUntilReclaimIsConsumed) {
  for (bool reclaim_first : {false, true}) {
    ControlledReadEngine engine;
    ReadTestPool pool(0, 3);
    auto lane =
        hnsw_urma::ReadBatchLane::Create(engine, 0, {.peer_count = 2}).value();
    auto *lane_view = lane.get();
    UB_REQUIRE_TRUE(pool.pool->BindLanes(std::span(&lane_view, 1)).ok());
    ReadTestBatch q1(*pool.pool, {0, 1}), q2(*pool.pool, {2});
    q1.Publish(*lane);
    q2.Publish(*lane);
    UB_REQUIRE_TRUE(lane->Progress().ok());
    q1.cancel.store(true);
    UB_REQUIRE_TRUE(lane->Progress().ok());
    UB_REQUIRE_EQ(q1.terminal_calls, 1U);
    UB_REQUIRE_TRUE(q1.batch.reclaim_required);
    UB_REQUIRE_FALSE(lane->drained());
    if (!reclaim_first)
      q1.Consume();
    UB_REQUIRE_EQ(pool.pool->available(), 0U);
    while (!engine.pending.empty())
      engine.Complete();
    UB_REQUIRE_TRUE(lane->Progress().ok());
    UB_REQUIRE_EQ(q2.terminal_calls, 1U);
    q2.Consume();
    UB_REQUIRE_EQ(pool.pool->available(), 1U);
    // A queued event is not permission to reuse the old record or addresses.
    UB_REQUIRE_FALSE(lane->drained());
    UB_REQUIRE_EQ(pool.pool->PollReclaims(), 1U);
    UB_REQUIRE_TRUE(lane->drained());
    if (reclaim_first) {
      UB_REQUIRE_EQ(pool.pool->available(), 1U);
      q1.Consume();
    }
    UB_REQUIRE_TRUE(pool.pool->drained());
    UB_REQUIRE_EQ(q1.terminal_calls, 1U);
    UB_REQUIRE_EQ(lane->profile().late_completion_count, 2U);
  }
}

TEST(ReferenceMockMockLane,
     FailedReadOnlyFailsItsBatchAndRetainsOtherAcceptedDma) {
  ControlledReadEngine engine;
  ReadTestPool pool(0, 3);
  auto lane =
      hnsw_urma::ReadBatchLane::Create(engine, 0, {.peer_count = 2}).value();
  auto *lane_view = lane.get();
  UB_REQUIRE_TRUE(pool.pool->BindLanes(std::span(&lane_view, 1)).ok());
  ReadTestBatch first(*pool.pool, {0, 1}), second(*pool.pool, {2});
  first.Publish(*lane);
  second.Publish(*lane);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  engine.Complete(0, hnsw_urma::Status(hnsw_urma::StatusCode::kTransportError,
                                       "injected CQE error"));
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_EQ(first.batch.status, hnsw_urma::StatusCode::kTransportError);
  first.Consume();
  while (!engine.pending.empty())
    engine.Complete();
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_EQ(second.batch.status, hnsw_urma::StatusCode::kOk);
  UB_REQUIRE_EQ(second.terminal_calls, 1U);
  second.Consume();
  UB_REQUIRE_EQ(pool.pool->PollReclaims(), 1U);
  UB_REQUIRE_TRUE(pool.pool->drained());
}

TEST(ReferenceMockMockLane, TerminalCallbackMayDestroyBatchImmediately) {
  ControlledReadEngine engine;
  ReadTestPool pool(0, 1);
  auto lane =
      hnsw_urma::ReadBatchLane::Create(engine, 0, {.peer_count = 2}).value();
  auto query = std::make_unique<ReadTestBatch>(
      *pool.pool, std::initializer_list<std::uint64_t>{0});
  query->batch.terminal_context = &query;
  query->batch.terminal_callback = [](void *context,
                                      hnsw_urma::ReadBatch &) noexcept {
    auto &owner = *static_cast<std::unique_ptr<ReadTestBatch> *>(context);
    owner->Consume();
    owner.reset();
    return hnsw_urma::Status::Ok();
  };
  query->Publish(*lane);
  UB_REQUIRE_TRUE(lane->Progress().ok());
  engine.Complete();
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_TRUE(query == nullptr);
  UB_REQUIRE_TRUE(lane->drained());
  UB_REQUIRE_TRUE(pool.pool->drained());
}

#include "hnsw_urma/runtime/scheduler.hpp"
#include <thread>

TEST(ReferenceMockMockLane,
     ConcurrentSearchAndProgressReuseResultsAfterFinalReadyPublication) {
  constexpr std::uint32_t searches = 2, progresses = 2, iterations = 200;
  std::array<ControlledReadEngine, progresses> engines;
  std::array<std::unique_ptr<hnsw_urma::ReadBatchLane>, progresses> lanes;
  for (std::uint32_t lane = 0; lane < progresses; ++lane)
    lanes[lane] = hnsw_urma::ReadBatchLane::Create(engines[lane], lane,
                                                   {.search_workers = searches,
                                                    .peer_count = 2,
                                                    .submit_depth = 2,
                                                    .records_per_search = 2,
                                                    .in_flight_limit = 2})
                      .value();
  auto queues =
      hnsw_urma::SchedulerQueues::Create(searches, progresses, searches, 1)
          .value();
  auto publisher =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, searches, progresses)
          .value();
  std::atomic<std::uint32_t> finished{0};
  std::atomic<bool> failed{false};
  std::array<std::exception_ptr, searches + progresses> errors{};
  std::vector<std::thread> threads;
  for (std::uint32_t lane = 0; lane < progresses; ++lane) {
    threads.emplace_back([&, lane] {
      try {
        while (finished.load(std::memory_order_acquire) != searches &&
               !failed.load()) {
          while (!engines[lane].pending.empty())
            engines[lane].Complete();
          UB_REQUIRE_TRUE(lanes[lane]->Progress().ok());
          UB_REQUIRE_TRUE(publisher->Flush(lane, 4).ok());
          std::this_thread::yield();
        }
      } catch (...) {
        errors[searches + lane] = std::current_exception();
        failed.store(true);
      }
    });
  }
  for (std::uint32_t search = 0; search < searches; ++search) {
    threads.emplace_back([&, search] {
      try {
        ReadTestPool pool(search, progresses);
        for (std::uint32_t iteration = 0;
             iteration < iterations && !failed.load(); ++iteration) {
          const hnsw_urma::QueryKey key{0, search, iteration + 1U};
          hnsw_urma::QueryRunController controller(key.generation, progresses);
          UB_REQUIRE_TRUE(controller.BindOwner(search).ok());
          UB_REQUIRE_TRUE(controller.Admit().ok());
          UB_REQUIRE_TRUE(controller.TryAcquire());
          std::array<hnsw_urma::ReadResultView, progresses> views{};
          UB_REQUIRE_TRUE(pool.pool->Acquire(views, 4).ok());
          std::array<hnsw_urma::ReadRequest, progresses> requests{};
          std::array<hnsw_urma::ReadBatch, progresses> batches;
          std::array<hnsw_urma::QueryOperationState, progresses> states;
          std::array<hnsw_urma::QueryOperationBinding, progresses> bindings;
          std::array<hnsw_urma::LaneOperationCount, progresses> counts{};
          for (std::uint32_t lane = 0; lane < progresses; ++lane) {
            counts[lane] = {lane, 1};
            requests[lane] = {
                1,
                hnsw_urma::SegmentKind::kVectors,
                iteration % 64,
                {const_cast<std::byte *>(views[lane].data), views[lane].bytes}};
            auto &batch = batches[lane];
            batch.owner_search = search;
            batch.requests = std::span(&requests[lane], 1);
            batch.ranges.reserve(1);
            UB_REQUIRE_TRUE(
                bindings[lane]
                    .Prepare(key, controller, states[lane], *publisher,
                             {lane, lane, key.generation, 1, iteration + 1U})
                    .ok());
            batch.terminal_context = &bindings[lane];
            batch.terminal_callback = [](void *context,
                                         hnsw_urma::ReadBatch &) noexcept {
              return static_cast<hnsw_urma::QueryOperationBinding *>(context)
                  ->Complete(hnsw_urma::OperationTerminalOutcome::kComplete);
            };
            UB_REQUIRE_TRUE(lanes[lane]->PrepareBatch(batch).ok());
          }
          UB_REQUIRE_TRUE(controller.BeginWait(counts, key.generation).ok());
          for (std::uint32_t lane = 0; lane < progresses; ++lane) {
            UB_REQUIRE_TRUE(bindings[lane].MarkPublished().ok());
            lanes[lane]->PublishBatch(batches[lane]);
          }
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(10);
          std::optional<hnsw_urma::QueryRunnableHandle> ready;
          while (!failed.load() &&
                 !(ready = hnsw_urma::test::TakeRunnable(
                       *queues, search, std::chrono::steady_clock::now()))) {
            UB_REQUIRE_TRUE(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
          }
          if (failed.load())
            break;
          UB_REQUIRE_EQ(ready->query, key);
          UB_REQUIRE_TRUE(controller.ValidateReadyHandle(
              {key, ready->epoch, 0, 0, search}));
          for (const auto &view : views)
            UB_REQUIRE_EQ(view.data[0],
                          static_cast<std::byte>(iteration % 64 + 1));
          pool.pool->Return(views);
          // All batch, binding and Query objects are destroyed immediately.
        }
        UB_REQUIRE_TRUE(pool.pool->drained());
      } catch (...) {
        errors[search] = std::current_exception();
        failed.store(true);
      }
      finished.fetch_add(1, std::memory_order_release);
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (auto &error : errors)
    if (error)
      std::rethrow_exception(error);
  UB_REQUIRE_FALSE(failed.load());
  for (auto &lane : lanes)
    UB_REQUIRE_TRUE(lane->drained());
}
