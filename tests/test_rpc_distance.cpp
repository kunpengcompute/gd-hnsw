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

#include "hnsw_urma/runtime/rpc/distance_oracle.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <thread>

namespace hnsw_urma {
namespace {
struct Wire {
  alignas(8) std::array<std::byte, 4096> bytes{};
};
struct Fixture {
  std::unique_ptr<SchedulerQueues> queues;
  std::unique_ptr<SchedulerReadyPublisher> publisher;
  std::unique_ptr<RpcDistanceOracle> oracle;
  std::array<std::unique_ptr<RpcClientOwner>, 2> clients;
  const std::array<std::uint32_t, 4> ranks{0, 1, 2, 3};
  const std::array<std::uint32_t, 4> owners{kNoCommunicationOwner, 0, 1, 0};
  Fixture(std::uint32_t searches = 4, std::uint32_t slots = 8,
          std::uint32_t submit = 8, std::uint32_t queries = 8) {
    queues = SchedulerQueues::Create(searches, 3, queries, queries, true, slots)
                 .value();
    publisher = SchedulerReadyPublisher::Create(*queues, searches, 3).value();
    oracle = RpcDistanceOracle::Create(searches, 2, submit, slots, 8, 2, 0,
                                       ranks, owners, 1, *queues, true)
                 .value();
    for (std::uint32_t owner = 0; owner < 2; ++owner)
      clients[owner] = RpcClientOwner::Create(owner, 4, searches * slots,
                                              oracle->submissions(), queries, 4,
                                              true, oracle.get())
                           .value();
  }
  void Drain() {
    for (auto &client : clients)
      UB_REQUIRE_TRUE(client->DrainSubmissions().ok());
  }
  PreparedBatch Send(std::uint32_t peer, std::uint32_t slot = 0,
                     std::uint64_t seq = 1) {
    auto &client = clients[owners[peer]];
    Wire wire;
    auto batch =
        client->PrepareRequestBatch(peer, slot, seq, 8, 4096, 4096, wire.bytes);
    UB_REQUIRE_TRUE(batch.ok());
    UB_REQUIRE_TRUE(batch.value().has_value());
    UB_REQUIRE_TRUE(client->ValidatePreparedRequest(*batch.value()).value());
    UB_REQUIRE_TRUE(client->CommitPreparedRequest(*batch.value()).ok());
    return *batch.value();
  }
  void Respond(PreparedBatch batch,
               std::span<const DistanceBatchResponseTaskInput> tasks) {
    Wire wire;
    auto bytes = EncodeDistanceBatchResponse(batch.sequence, tasks, wire.bytes);
    UB_REQUIRE_TRUE(bytes.ok());
    auto accepted = clients[owners[batch.peer_rank]]->AcceptResponse(
        batch.peer_rank, batch.slot_id, batch.sequence,
        std::span<const std::byte>(wire.bytes).first(bytes.value()));
    UB_REQUIRE_TRUE(accepted.ok());
  }
  void Audit(std::chrono::steady_clock::time_point now =
                 std::chrono::steady_clock::now()) {
    for (std::uint32_t owner = 0; owner < 2; ++owner)
      UB_REQUIRE_TRUE(
          oracle->AuditOwnerControl(owner, *clients[owner], 64, now).ok());
  }
  RpcReclaimHandle Reclaim(std::uint32_t search) {
    auto input =
        queues->PopSearchInput(search, std::chrono::steady_clock::now());
    UB_REQUIRE_TRUE(input.has_value());
    UB_REQUIRE_TRUE(std::holds_alternative<RpcReclaimHandle>(*input));
    auto handle = std::get<RpcReclaimHandle>(*input);
    SetCurrentDistanceSearchWorker(search);
    UB_REQUIRE_TRUE(oracle->HandleReclaim(search, handle).ok());
    return handle;
  }
};
struct Stage {
  Fixture &fixture;
  QueryKey key;
  std::uint32_t search;
  std::unique_ptr<QueryRunController> controller;
  QueryVectorLease lease;
  std::array<node_id_t, 2> ids{10, 11};
  std::array<std::array<float, 2>, 3> outputs{{{-1, -1}, {-1, -1}, {-1, -1}}};
  std::vector<PreparedDistanceOperation> operations;
  Stage(Fixture &f, QueryKey q = {0, 0, 42}, std::uint32_t owner = 3)
      : fixture(f), key(q), search(owner),
        controller(std::make_unique<QueryRunController>(q.generation, 3)),
        lease{q, std::make_shared<const std::vector<float>>(
                     std::initializer_list<float>{1, 2})} {
    UB_REQUIRE_TRUE(controller->BindOwner(search).ok());
    UB_REQUIRE_TRUE(controller->Admit().ok());
    UB_REQUIRE_TRUE(controller->TryAcquire());
  }
  Status Prepare(std::span<const shard_id_t> shards,
                 std::chrono::steady_clock::time_point deadline =
                     std::chrono::steady_clock::now() +
                     std::chrono::seconds(30),
                 QueryReadyPublisher *publisher = nullptr,
                 DistanceRequestOptions options = {}) {
    SetCurrentDistanceSearchWorker(search);
    for (auto shard : shards) {
      auto prepared = fixture.oracle->PrepareLeaseForWait(
          {key, controller->next_wait_epoch(), deadline, controller.get(),
           publisher ? publisher : fixture.publisher.get()},
          lease, shard, ids, outputs[operations.size()], options);
      if (!prepared.ok()) {
        for (auto prior : operations)
          UB_REQUIRE_TRUE(fixture.oracle->CancelPrepared(prior.ticket).ok());
        operations.clear();
        return prepared.status();
      }
      operations.push_back(prepared.value());
    }
    return Status::Ok();
  }
  void Publish(bool with_read = false, bool deferred = false) {
    std::array<std::uint32_t, 3> counts{};
    if (with_read)
      counts[0] = 1;
    for (auto op : operations)
      ++counts[*op.owner_lane];
    std::vector<LaneOperationCount> lanes;
    for (std::uint32_t lane = 0; lane < 3; ++lane)
      if (counts[lane])
        lanes.push_back({lane, counts[lane]});
    if (deferred)
      UB_REQUIRE_TRUE(controller->BeginDeferredReady().ok());
    UB_REQUIRE_TRUE(controller->BeginWait(lanes, key.generation).ok());
    for (auto op : operations)
      UB_REQUIRE_TRUE(fixture.oracle->PublishPrepared(op.ticket).ok());
  }
  void Ready() {
    auto runnable = test::TakeRunnable(*fixture.queues, search,
                                       std::chrono::steady_clock::now());
    UB_REQUIRE_TRUE(runnable.has_value());
    UB_REQUIRE_EQ(runnable->query, key);
    UB_REQUIRE_TRUE(
        controller->ValidateReadyHandle({key, runnable->epoch, 0, 0, search}));
    UB_REQUIRE_TRUE(controller->TryAcquire());
  }
  void Finish(StatusCode expected = StatusCode::kOk) {
    SetCurrentDistanceSearchWorker(search);
    for (auto op : operations)
      UB_REQUIRE_EQ(fixture.oracle->Finish(op.ticket).code(), expected);
  }
  DistanceBatchResponseTaskInput
  Response(std::size_t index, std::span<const float> distances) const {
    return {operations[index].ticket.generation, key.query_slot, distances,
            ids};
  }
  void Cancel() {
    UB_REQUIRE_TRUE(controller
                        ->RequestFailure(key.generation,
                                         TerminalOutcome::kCancelled,
                                         StatusCode::kCancelled)
                        .ok());
  }
};
const std::array<shard_id_t, 1> B{1};
const std::array<shard_id_t, 3> BCD{1, 2, 3};
const std::array<float, 2> distances{3, 4};
void RespondOne(Fixture &fixture, PreparedBatch batch, const Stage &stage,
                std::size_t op = 0) {
  auto response = stage.Response(op, distances);
  fixture.Respond(batch, std::span(&response, 1));
}
} // namespace

TEST(ReferenceRuntimeRpcDistance,
     RpcProgressAggregatesThreePeersInAllOrdersAndPreservesSearchOwner) {
  std::array<unsigned, 3> order{0, 1, 2};
  do {
    Fixture fixture;
    Stage stage(fixture);
    UB_REQUIRE_TRUE(stage.Prepare(BCD).ok());
    stage.Publish();
    fixture.Drain();
    UB_REQUIRE_EQ(*stage.operations[0].owner_lane, 1U);
    UB_REQUIRE_EQ(*stage.operations[1].owner_lane, 2U);
    UB_REQUIRE_EQ(*stage.operations[2].owner_lane, 1U);
    std::array<PreparedBatch, 3> batches{fixture.Send(1), fixture.Send(2),
                                         fixture.Send(3)};
    for (unsigned i = 0; i < 3; ++i) {
      RespondOne(fixture, batches[order[i]], stage, order[i]);
      UB_REQUIRE_EQ(fixture.queues->metrics().ready_published,
                    i == 2 ? 1U : 0U);
    }
    UB_REQUIRE_EQ(stage.outputs[0][0], -1.0F); // Copy belongs to Search Finish.
    stage.Ready();
    stage.Finish();
    for (auto output : stage.outputs)
      UB_REQUIRE_EQ(output, distances);
    UB_REQUIRE_TRUE(fixture.oracle->drained());
    UB_REQUIRE_EQ(fixture.queues->metrics().rpc_reclaim_published, 0U);
    for (auto &client : fixture.clients)
      UB_REQUIRE_TRUE(client->drained());
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(ReferenceRuntimeRpcDistance,
     RpcTwoOwnersAndReadConvergeOnceWhileInlineWorkDefersReady) {
  Fixture fixture;
  Stage stage(fixture);
  UB_REQUIRE_TRUE(stage.Prepare(BCD).ok());
  QueryOperationState state;
  QueryOperationBinding read;
  UB_REQUIRE_TRUE(read.Prepare(stage.key, *stage.controller, state,
                               *fixture.publisher,
                               {0, 0, stage.key.generation, 1, 99})
                      .ok());
  stage.Publish(true, true);
  UB_REQUIRE_TRUE(read.MarkPublished().ok());
  fixture.Drain();
  auto b = fixture.Send(1), c = fixture.Send(2), d = fixture.Send(3);
  std::array<std::exception_ptr, 3> errors{};
  std::thread owner0([&] {
    try {
      RespondOne(fixture, b, stage, 0);
      RespondOne(fixture, d, stage, 2);
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  std::thread owner1([&] {
    try {
      RespondOne(fixture, c, stage, 1);
    } catch (...) {
      errors[1] = std::current_exception();
    }
  });
  std::thread reader([&] {
    try {
      UB_REQUIRE_TRUE(read.Complete(OperationTerminalOutcome::kComplete).ok());
    } catch (...) {
      errors[2] = std::current_exception();
    }
  });
  owner0.join();
  owner1.join();
  reader.join();
  for (auto error : errors)
    if (error)
      std::rethrow_exception(error);
  UB_REQUIRE_EQ(fixture.queues->metrics().ready_published, 0U);
  for (std::uint32_t lane = 0; lane < 3; ++lane)
    UB_REQUIRE_EQ(fixture.publisher->Flush(lane, 8).value(), 0U);
  stage.controller->EndDeferredReady();
  for (std::uint32_t lane = 0; lane < 3; ++lane)
    UB_REQUIRE_TRUE(fixture.publisher->Flush(lane, 8).ok());
  UB_REQUIRE_EQ(fixture.queues->metrics().ready_published, 1U);
  stage.Ready();
  stage.Finish();
  UB_REQUIRE_TRUE(fixture.oracle->drained());
}

TEST(ReferenceRuntimeRpcDistance,
     RpcPoolIsPrivateTicketsAreGlobalAndWholePreparationRollsBack) {
  Fixture fixture(2, 2, 8);
  Stage first(fixture, {0, 0, 1}, 0);
  UB_REQUIRE_EQ(first.Prepare(BCD).code(), StatusCode::kResourceExhausted);
  UB_REQUIRE_TRUE(fixture.oracle->drained());
  const std::array<shard_id_t, 2> bd{1, 3};
  UB_REQUIRE_TRUE(first.Prepare(bd).ok());
  Stage same(fixture, {0, 1, 1}, 0), other(fixture, {0, 2, 1}, 1);
  UB_REQUIRE_EQ(same.Prepare(B).code(), StatusCode::kResourceExhausted);
  UB_REQUIRE_TRUE(other.Prepare(B).ok());
  UB_REQUIRE_TRUE(other.operations[0].ticket.generation !=
                  first.operations[0].ticket.generation);
  for (auto *stage : {&first, &other}) {
    SetCurrentDistanceSearchWorker(stage->search);
    for (auto op : stage->operations)
      UB_REQUIRE_TRUE(fixture.oracle->CancelPrepared(op.ticket).ok());
  }
  UB_REQUIRE_TRUE(fixture.oracle->drained());
  UB_REQUIRE_TRUE(fixture.oracle->profile().pool_backpressure >= 2);
  Fixture submit_limited(1, 8, 1);
  Stage limited(submit_limited, {0, 0, 1}, 0);
  UB_REQUIRE_EQ(limited.Prepare(bd).code(), StatusCode::kResourceExhausted);
  UB_REQUIRE_TRUE(submit_limited.oracle->drained());
  UB_REQUIRE_TRUE(limited.Prepare(B).ok());
  UB_REQUIRE_TRUE(
      submit_limited.oracle->CancelPrepared(limited.operations[0].ticket).ok());
  UB_REQUIRE_TRUE(submit_limited.oracle->profile().submit_backpressure != 0);
}

TEST(ReferenceRuntimeRpcDistance,
     RpcTimeoutAndCancellationBeforeSendNeverRequireReclaim) {
  // 0: unpublished rollback; 1: submit; 2: pending; 3: encoded/unaccepted.
  for (unsigned location = 0; location < 4; ++location) {
    for (bool cancel : {false, true}) {
      Fixture fixture;
      Stage stage(fixture);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(30);
      UB_REQUIRE_TRUE(stage.Prepare(B, deadline).ok());
      if (location == 0) {
        UB_REQUIRE_TRUE(
            fixture.oracle->CancelPrepared(stage.operations[0].ticket).ok());
        UB_REQUIRE_TRUE(fixture.oracle->drained());
        continue;
      }
      stage.Publish();
      if (location >= 2)
        fixture.Drain();
      Wire encoded;
      std::optional<PreparedBatch> batch;
      if (location == 3)
        batch = fixture.clients[0]
                    ->PrepareRequestBatch(1, 0, 1, 8, 4096, 4096, encoded.bytes)
                    .value();
      if (cancel)
        stage.Cancel();
      // Submit is checked at take-over; use actual expired deadline for that
      // case.
      if (location == 1 && !cancel) {
        // Cancellation at take-over is covered here; timeout at take-over is
        // tested below with an already expired prepared operation.
        stage.Cancel();
      }
      if (location == 1)
        fixture.Drain();
      else
        fixture.Audit(cancel ? std::chrono::steady_clock::now() : deadline);
      stage.Ready();
      stage.Finish(cancel || location == 1 ? StatusCode::kCancelled
                                           : StatusCode::kTimeout);
      UB_REQUIRE_TRUE(fixture.oracle->drained());
      UB_REQUIRE_TRUE(fixture.clients[0]->drained());
      UB_REQUIRE_EQ(fixture.queues->metrics().rpc_reclaim_published, 0U);
      if (batch)
        UB_REQUIRE_FALSE(
            fixture.clients[0]->CommitPreparedRequest(*batch).ok());
    }
  }
  Fixture fixture;
  Stage expired(fixture);
  UB_REQUIRE_TRUE(expired
                      .Prepare(B, std::chrono::steady_clock::now() -
                                      std::chrono::seconds(1))
                      .ok());
  expired.Publish();
  fixture.Drain();
  expired.Ready();
  expired.Finish(StatusCode::kTimeout);
  UB_REQUIRE_TRUE(fixture.oracle->drained());
}

TEST(ReferenceRuntimeRpcDistance,
     RpcPreparedCancellationReencodesSurvivingBatchMembers) {
  Fixture fixture;
  Stage first(fixture, {0, 0, 1}, 0), second(fixture, {0, 1, 1}, 1);
  UB_REQUIRE_TRUE(first.Prepare(B).ok());
  first.Publish();
  UB_REQUIRE_TRUE(second.Prepare(B).ok());
  second.Publish();
  fixture.Drain();
  Wire wire;
  auto batch = fixture.clients[0]
                   ->PrepareRequestBatch(1, 0, 1, 8, 4096, 4096, wire.bytes)
                   .value();
  UB_REQUIRE_EQ(batch->task_count, 2U);
  first.Cancel();
  UB_REQUIRE_FALSE(fixture.clients[0]->ValidatePreparedRequest(*batch).value());
  fixture.Audit();
  first.Ready();
  first.Finish(StatusCode::kCancelled);
  auto survivor = fixture.Send(1, 0, 2);
  UB_REQUIRE_EQ(survivor.task_count, 1U);
  RespondOne(fixture, survivor, second);
  second.Ready();
  second.Finish();
  UB_REQUIRE_TRUE(fixture.oracle->drained());
}

TEST(ReferenceRuntimeRpcDistance,
     RpcLateResponseReclaimsAfterQueryDestructionAndRejectsOldTicket) {
  Fixture fixture(1, 1, 1);
  auto stage = std::make_unique<Stage>(fixture, QueryKey{0, 0, 1}, 0);
  UB_REQUIRE_TRUE(stage->Prepare(B).ok());
  stage->Publish();
  fixture.Drain();
  auto batch = fixture.Send(1);
  auto response = stage->Response(0, distances);
  // Preserve wire input outside Query storage for the late packet.
  const auto ids = stage->ids;
  response.ids = ids;
  fixture.Audit(std::chrono::steady_clock::now() + std::chrono::seconds(60));
  stage->Ready();
  stage->Finish(StatusCode::kTimeout);
  UB_REQUIRE_EQ(stage->outputs[0][0], -1.0F);
  const auto key = stage->key;
  UB_REQUIRE_FALSE(fixture.oracle->QueryWireQuiesced(key));
  stage.reset();
  UB_REQUIRE_FALSE(fixture.oracle->drained());
  fixture.Respond(batch, std::span(&response, 1));
  UB_REQUIRE_FALSE(
      fixture.oracle->drained()); // Physical terminal is insufficient.
  auto reclaim = fixture.Reclaim(0);
  UB_REQUIRE_TRUE(fixture.oracle->drained());
  UB_REQUIRE_TRUE(fixture.oracle->QueryWireQuiesced(key));
  Stage next(fixture, {0, 0, 2}, 0);
  UB_REQUIRE_TRUE(next.Prepare(B).ok());
  UB_REQUIRE_EQ(next.operations[0].ticket.value, reclaim.operation_slot + 1U);
  UB_REQUIRE_TRUE(next.operations[0].ticket.generation != reclaim.ticket);
  UB_REQUIRE_FALSE(fixture.oracle->HandleReclaim(0, reclaim).ok());
  UB_REQUIRE_TRUE(
      fixture.oracle->CancelPrepared(next.operations[0].ticket).ok());
  UB_REQUIRE_EQ(fixture.oracle->profile().exceptional_reclaimed, 1U);
}

TEST(ReferenceRuntimeRpcDistance,
     RpcReclaimCanPrecedeReadyAndFinishWithoutReusingTheSlot) {
  Fixture fixture;
  Stage stage(fixture);
  // Only B expires, C keeps this stage Waiting until after Reclaim is consumed.
  UB_REQUIRE_TRUE(stage
                      .Prepare(B, std::chrono::steady_clock::now() +
                                      std::chrono::seconds(1))
                      .ok());
  const std::array<shard_id_t, 1> c{2};
  UB_REQUIRE_TRUE(stage.Prepare(c).ok());
  stage.Publish();
  fixture.Drain();
  auto b = fixture.Send(1), cb = fixture.Send(2);
  fixture.Audit(std::chrono::steady_clock::now() + std::chrono::seconds(2));
  RespondOne(fixture, b, stage);
  auto reclaim = fixture.Reclaim(3);
  UB_REQUIRE_EQ(fixture.queues->metrics().ready_published, 0U);
  UB_REQUIRE_EQ(fixture.oracle->profile().slots_in_use, 2U);
  UB_REQUIRE_FALSE(fixture.oracle->HandleReclaim(3, reclaim).ok());
  RespondOne(fixture, cb, stage, 1);
  stage.Ready();
  SetCurrentDistanceSearchWorker(3);
  UB_REQUIRE_EQ(fixture.oracle->Finish(stage.operations[0].ticket).code(),
                StatusCode::kTimeout);
  UB_REQUIRE_TRUE(fixture.oracle->Finish(stage.operations[1].ticket).ok());
  UB_REQUIRE_EQ(stage.outputs[0][0], -1.0F);
  UB_REQUIRE_EQ(stage.outputs[1], distances);
  UB_REQUIRE_TRUE(fixture.oracle->drained());
}

TEST(ReferenceRuntimeRpcDistance,
     RpcResponseFilteringKeepsOriginalOrderAndFrozenCancellation) {
  Fixture fixture;
  Stage stage(fixture);
  UB_REQUIRE_TRUE(
      stage
          .Prepare(B,
                   std::chrono::steady_clock::now() + std::chrono::seconds(30),
                   nullptr, {3.5F})
          .ok());
  stage.Publish();
  fixture.Drain();
  auto batch = fixture.Send(1);
  const std::array<node_id_t, 1> survivor{11};
  const std::array<float, 1> value{2};
  DistanceBatchResponseTaskInput response{stage.operations[0].ticket.generation,
                                          0, value, survivor};
  fixture.Respond(batch, std::span(&response, 1));
  stage.Ready();
  stage.Finish();
  UB_REQUIRE_TRUE(stage.outputs[0][0] > 3.5F);
  UB_REQUIRE_EQ(stage.outputs[0][1], 2.0F);
  UB_REQUIRE_TRUE(fixture.oracle->drained());
}

TEST(ReferenceRuntimeRpcDistance,
     RpcResponseCallbackCanConsumeDestroyAndReuseBeforeBatchHandoffReturns) {
  Fixture fixture(2, 1, 1);
  struct BlockingPublisher final : QueryReadyPublisher {
    SchedulerReadyPublisher &next;
    std::atomic<bool> entered{false}, released{false};
    explicit BlockingPublisher(SchedulerReadyPublisher &p) : next(p) {}
    Status PublishReady(std::uint32_t lane,
                        QueryReadyHandle handle) noexcept override {
      auto status = next.PublishReady(lane, handle);
      entered.store(true, std::memory_order_release);
      while (!released.load(std::memory_order_acquire))
        std::this_thread::yield();
      return status;
    }
  } blocking(*fixture.publisher);
  auto first = std::make_unique<Stage>(fixture, QueryKey{0, 0, 1}, 0);
  Stage second(fixture, {0, 1, 1}, 1);
  UB_REQUIRE_TRUE(
      first
          ->Prepare(B,
                    std::chrono::steady_clock::now() + std::chrono::seconds(30),
                    &blocking)
          .ok());
  first->Publish();
  UB_REQUIRE_TRUE(second.Prepare(B).ok());
  second.Publish();
  fixture.Drain();
  auto batch = fixture.Send(1);
  UB_REQUIRE_EQ(batch.task_count, 2U);
  std::array<DistanceBatchResponseTaskInput, 2> response{
      first->Response(0, distances), second.Response(0, distances)};
  // Encoding happens before the thread; callbacks are allowed to destroy IDs.
  Wire wire;
  auto bytes =
      EncodeDistanceBatchResponse(batch.sequence, response, wire.bytes).value();
  Status accepted;
  std::thread rpc([&] {
    accepted = fixture.clients[0]->AcceptResponse(
        1, 0, 1, std::span<const std::byte>(wire.bytes).first(bytes));
  });
  while (!blocking.entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  first->Ready();
  const auto previous = first->operations[0].ticket;
  first->Finish();
  first.reset();
  Stage reused(fixture, {0, 0, 2}, 0);
  UB_REQUIRE_TRUE(reused.Prepare(B).ok());
  UB_REQUIRE_EQ(reused.operations[0].ticket.value, previous.value);
  blocking.released.store(true, std::memory_order_release);
  rpc.join();
  UB_REQUIRE_TRUE(accepted.ok());
  second.Ready();
  second.Finish();
  SetCurrentDistanceSearchWorker(0);
  UB_REQUIRE_TRUE(
      fixture.oracle->CancelPrepared(reused.operations[0].ticket).ok());
  UB_REQUIRE_TRUE(fixture.oracle->drained());
  UB_REQUIRE_TRUE(fixture.clients[0]->drained());
}

} // namespace hnsw_urma

namespace hnsw_urma {
TEST(ReferenceRuntimeRpcDistance,
     RpcOneQueryOneSlotFillsAPlusKThenReusesOnlyAfterBothEvents) {
  Fixture fixture(32, 1, 1, 1);
  UB_REQUIRE_EQ(fixture.queues->event_ring_count(), 32U);
  UB_REQUIRE_EQ(fixture.queues->event_capacity(), 2U);
  for (std::uint64_t generation = 1; generation <= 20; ++generation) {
    Stage stage(fixture, {0, 0, generation}, 31);
    UB_REQUIRE_TRUE(stage.Prepare(B).ok());
    stage.Publish();
    fixture.Drain();
    const auto batch = fixture.Send(1, 0, generation);
    stage.Cancel();
    fixture.Audit(std::chrono::steady_clock::now() +
                  std::chrono::seconds(generation));
    RespondOne(fixture, batch, stage);
    UB_REQUIRE_EQ(fixture.queues->metrics().search_event_depth[31], 2U);
    stage.Ready();
    stage.Finish(StatusCode::kCancelled);
    UB_REQUIRE_FALSE(fixture.oracle->drained());
    fixture.Reclaim(31);
    UB_REQUIRE_TRUE(fixture.oracle->drained());
    UB_REQUIRE_EQ(fixture.queues->metrics().search_event_depth[31], 0U);
  }
}

TEST(ReferenceRuntimeRpcDistance,
     RpcOverlappingQuerySlotGenerationRollsBackWithoutLeakingPoolCapacity) {
  Fixture fixture;
  Stage first(fixture, {0, 0, 1}, 0), overlapping(fixture, {0, 0, 2}, 1);
  UB_REQUIRE_TRUE(first.Prepare(B).ok());
  UB_REQUIRE_EQ(overlapping.Prepare(B).code(), StatusCode::kProtocolError);
  UB_REQUIRE_EQ(fixture.oracle->profile().slots_in_use, 1U);
  SetCurrentDistanceSearchWorker(0);
  UB_REQUIRE_TRUE(
      fixture.oracle->CancelPrepared(first.operations[0].ticket).ok());
  UB_REQUIRE_TRUE(overlapping.Prepare(B).ok());
  UB_REQUIRE_TRUE(
      fixture.oracle->CancelPrepared(overlapping.operations[0].ticket).ok());
  UB_REQUIRE_TRUE(fixture.oracle->drained());
}
} // namespace hnsw_urma

#include "hnsw_urma/runtime/drain.hpp"
#include "hnsw_urma/search/search.hpp"
namespace hnsw_urma {
TEST(ReferenceRuntimeRpcDistance,
     HnswConsumesFailedRpcStageBeforeDoneAndDrainsLateLeaseThroughSearchEvent) {
  Fixture fixture;
  class UnusedGraph final : public GraphReader {
    bool SupportsPreparedWait() const noexcept override { return true; }
  } graph;
  GlobalIndexMeta meta;
  meta.ntotal = 40;
  meta.dim = 2;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 2;
  meta.ef_search_default = 2;
  meta.num_shards = 4;
  meta.entry_point = 10;
  meta.max_level = 0;
  const std::array<ShardRange, 4> ranges{
      {{0, 0, 10}, {1, 10, 10}, {2, 20, 10}, {3, 30, 10}}};
  auto map = ShardMap::Create(ranges, 40).value();
  SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  SearchRuntimeContext runtime{fixture.publisher.get(), 3};
  const std::array<float, 2> query{1, 2};
  const QueryKey key{0, 0, 1};
  auto task =
      HnswSearchTask::Create(key, meta, map, graph, *fixture.oracle, query,
                             params, std::chrono::steady_clock::now(), &runtime)
          .value();
  UB_REQUIRE_TRUE(task->BindOwner(3).ok());
  SetCurrentDistanceSearchWorker(3);
  UB_REQUIRE_EQ(task->Step(), SearchStepResult::kYielded);
  fixture.Drain();
  Wire request_wire;
  auto batch =
      fixture.clients[0]
          ->PrepareRequestBatch(1, 0, 1, 8, 4096, 4096, request_wire.bytes)
          .value();
  UB_REQUIRE_TRUE(fixture.clients[0]->CommitPreparedRequest(*batch).ok());
  auto request =
      DecodeDistanceBatchRequest(
          std::span<const std::byte>(request_wire.bytes).first(batch->bytes))
          .value();
  auto wire_task = request.Task(0).value();
  const std::array<float, 1> distance{2};
  DistanceBatchResponseTaskInput response{wire_task.descriptor.task_ticket, 0,
                                          distance, wire_task.ids};
  UB_REQUIRE_TRUE(
      task->RequestFailure(TerminalOutcome::kCancelled, StatusCode::kCancelled)
          .ok());
  UB_REQUIRE_EQ(
      task->Step(),
      SearchStepResult::kYielded); // Still Waiting; controller must survive.
  fixture.Audit();
  auto ready =
      test::TakeRunnable(*fixture.queues, 3, std::chrono::steady_clock::now());
  UB_REQUIRE_TRUE(ready.has_value());
  UB_REQUIRE_EQ(ready->query, key);
  UB_REQUIRE_EQ(task->Step(),
                SearchStepResult::kFailed); // Calls Finish on the entire stage.
  UB_REQUIRE_EQ(task->status().code(), StatusCode::kCancelled);
  task.reset();
  auto drain = DataPlaneDrainTracker::Create(1).value();
  UB_REQUIRE_TRUE(drain->StopAdmission().ok());
  UB_REQUIRE_TRUE(drain->MarkLocalQueriesComplete().ok());
  UB_REQUIRE_TRUE(drain->BeginDrainAfterRankConsensus().ok());
  LaneDrainSnapshot pending;
  pending.rpc_active = fixture.oracle->SearchDrained(3) ? 0U : 1U;
  drain->PublishLane(0, pending);
  UB_REQUIRE_FALSE(drain->TryMarkDrained().ok());
  fixture.Respond(*batch, std::span(&response, 1));
  UB_REQUIRE_FALSE(fixture.oracle->QueryWireQuiesced(key));
  fixture.Reclaim(3);
  UB_REQUIRE_TRUE(fixture.oracle->QueryWireQuiesced(key));
  UB_REQUIRE_TRUE(fixture.oracle->drained());
  drain->PublishLane(0, {});
  UB_REQUIRE_TRUE(drain->TryMarkDrained().ok());
}
} // namespace hnsw_urma
