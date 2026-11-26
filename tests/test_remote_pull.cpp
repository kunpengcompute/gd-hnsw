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
#include "test_framework.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <thread>
#include <vector>

#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/index/builder.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/search/remote_io.hpp"
#include "hnsw_urma/transport/mock_lane_post_engine.hpp"
#include "hnsw_urma/transport/read_batch_lane.hpp"

namespace {

class CapturingReadyPublisher final : public hnsw_urma::QueryReadyPublisher {
public:
  hnsw_urma::Status
  PublishReady(std::uint32_t,
               hnsw_urma::QueryReadyHandle handle) noexcept override {
    last = handle;
    calls.fetch_add(1, std::memory_order_relaxed);
    return hnsw_urma::Status::Ok();
  }

  std::atomic<std::uint32_t> calls{0};
  hnsw_urma::QueryReadyHandle last{};
};

class CollectingGraphConsumer final : public hnsw_urma::GraphNeighborConsumer {
public:
  hnsw_urma::Status
  Consume(std::size_t node_index,
          std::span<const hnsw_urma::node_id_t> neighbors) override {
    if (node_index != values.size()) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "graph nodes were consumed out of order");
    }
    values.emplace_back(neighbors.begin(), neighbors.end());
    return hnsw_urma::Status::Ok();
  }

  std::vector<std::vector<hnsw_urma::node_id_t>> values;
};

std::vector<std::byte> CopyBytes(std::span<const hnsw_urma::node_id_t> values) {
  std::vector<std::byte> bytes(values.size_bytes());
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

std::vector<std::byte> CopyBytes(std::span<const float> values) {
  std::vector<std::byte> bytes(values.size_bytes());
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void CompleteAll(hnsw_urma::MockLanePostEngine &engine,
                 hnsw_urma::ReadBatchLane &lane, bool reverse = false) {
  UB_REQUIRE_TRUE(lane.Progress().ok());
  while (engine.pending_events() != 0) {
    const auto index = reverse ? engine.pending_events() - 1U : 0U;
    UB_REQUIRE_TRUE(engine.Progress(index).ok());
  }
  UB_REQUIRE_TRUE(lane.Progress().ok());
}

} // namespace

TEST(ReferenceIntegrationRemotePull, GraphAndPullVectorShareOneReadBatchLane) {
  const auto root = std::filesystem::temp_directory_path() /
                    "hnsw_urma_read_batch_integration";
  std::filesystem::remove_all(root);
  const std::vector<float> vectors{
      0.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F,
      0.0F, 2.0F, 1.0F, 2.0F, 2.0F, 2.0F, 3.0F, 2.0F,
  };
  hnsw_urma::ReferenceBuildOptions build;
  build.num_shards = 2;
  build.M = 2;
  build.M0 = 7;
  build.ef_search = 8;
  UB_REQUIRE_TRUE(
      hnsw_urma::BuildReferenceIndex(root, vectors, 8, 2, build, 100).ok());
  auto global = hnsw_urma::LoadGlobalIndexFile(root).value();
  auto shard_map =
      hnsw_urma::ShardMap::Create(global.ranges, global.meta.ntotal).value();
  auto upper = hnsw_urma::LoadUpperGraphFile(root, global.meta).value();
  auto local =
      hnsw_urma::LoadShardFile(root, global.meta, global.ranges[0]).value();
  auto remote =
      hnsw_urma::LoadShardFile(root, global.meta, global.ranges[1]).value();
  auto graph_memory = CopyBytes(remote.layer0_slots());
  auto vector_memory = CopyBytes(remote.vectors());
  auto engine = hnsw_urma::MockLanePostEngine::Create(0, 128).value();
  UB_REQUIRE_TRUE(engine
                      .RegisterSegment(1, hnsw_urma::SegmentKind::kLayer0Graph,
                                       graph_memory)
                      .ok());
  UB_REQUIRE_TRUE(
      engine.RegisterSegment(1, hnsw_urma::SegmentKind::kVectors, vector_memory)
          .ok());
  auto lane = hnsw_urma::ReadBatchLane::Create(engine, 0,
                                               {.peer_count = 2,
                                                .submit_depth = 32,
                                                .in_flight_limit = 128,
                                                .wr_list_max = 16,
                                                .poll_batch = 64})
                  .value();
  hnsw_urma::ReadBatchLane *lane_view = lane.get();
  auto memory = hnsw_urma::PageAlignedBuffer::Create(128 * 64).value();
  auto pool = hnsw_urma::SearchReadResources::Create(0, memory.logical_memory(),
                                                     128, 64)
                  .value();
  auto *pool_view = pool.get();
  UB_REQUIRE_TRUE(pool->BindLanes(std::span(&lane_view, 1)).ok());
  const hnsw_urma::PeerDataSegments peer{1, 1, graph_memory.size(),
                                         vector_memory.size()};
  auto peers = hnsw_urma::PeerSegmentTable::Create(global.meta, shard_map, 0,
                                                   std::span(&peer, 1));
  UB_REQUIRE_TRUE(peers.ok());
  const std::array<std::uint32_t, 2> owners{
      std::numeric_limits<std::uint32_t>::max(), 0};
  auto graph = hnsw_urma::UrmaGraphReader::Create(
      global.meta, shard_map, upper, local, peers.value(),
      std::span(&lane_view, 1), owners, std::span(&pool_view, 1), 4);
  auto distance = hnsw_urma::PullVectorDistanceOracle::Create(
      global.meta, shard_map, local, peers.value(), std::span(&lane_view, 1),
      owners, std::span(&pool_view, 1),
      hnsw_urma::DistanceImplementation::kScalar, 4, 8);
  UB_REQUIRE_TRUE(graph.ok());
  UB_REQUIRE_TRUE(distance.ok());

  CapturingReadyPublisher graph_ready;
  hnsw_urma::QueryRunController graph_controller(11, 1);
  UB_REQUIRE_TRUE(graph_controller.Admit().ok());
  UB_REQUIRE_TRUE(graph_controller.TryAcquire());
  const std::array<hnsw_urma::NodeToExpand, 2> nodes{{{0, 0}, {4, 0}}};
  hnsw_urma::NeighborBatch unused_output;
  const auto graph_epoch = graph_controller.next_wait_epoch();
  auto graph_phase = graph.value().PrepareForWait(
      {{0, 0, 11},
       graph_epoch,
       std::chrono::steady_clock::now() + std::chrono::seconds(1),
       &graph_controller,
       &graph_ready},
      nodes, &unused_output);
  UB_REQUIRE_TRUE(graph_phase.ok());
  UB_REQUIRE_EQ(graph_phase.value().participating_lanes.size(), 1U);
  UB_REQUIRE_EQ(graph_phase.value().participating_lanes[0].operations, 1U);
  UB_REQUIRE_TRUE(
      graph_controller.BeginWait(graph_phase.value().participating_lanes, 11)
          .ok());
  UB_REQUIRE_TRUE(
      graph.value().PublishPrepared(graph_phase.value().ticket).ok());
  CompleteAll(engine, *lane, true);
  UB_REQUIRE_EQ(graph_ready.calls.load(), 1U);
  CollectingGraphConsumer consumer;
  UB_REQUIRE_TRUE(
      graph.value().FinishConsume(graph_phase.value().ticket, consumer).ok());
  UB_REQUIRE_EQ(consumer.values.size(), 2U);
  auto local_neighbors = local.TrustedNeighbors(0);
  auto remote_neighbors = remote.Neighbors(4).value();
  remote_neighbors = remote_neighbors.first(static_cast<std::size_t>(
      std::distance(remote_neighbors.begin(),
                    std::find(remote_neighbors.begin(), remote_neighbors.end(),
                              hnsw_urma::INVALID_NODE_ID))));
  UB_REQUIRE_EQ(consumer.values[0],
                std::vector<hnsw_urma::node_id_t>(local_neighbors.begin(),
                                                  local_neighbors.end()));
  UB_REQUIRE_EQ(consumer.values[1],
                std::vector<hnsw_urma::node_id_t>(remote_neighbors.begin(),
                                                  remote_neighbors.end()));
  UB_REQUIRE_TRUE(lane->Progress().ok());

  CapturingReadyPublisher distance_ready;
  hnsw_urma::QueryRunController distance_controller(12, 1);
  UB_REQUIRE_TRUE(distance_controller.Admit().ok());
  UB_REQUIRE_TRUE(distance_controller.TryAcquire());
  const std::vector<float> query{2.8F, 1.8F};
  const std::array<hnsw_urma::node_id_t, 2> ids{4, 7};
  std::array<float, 2> output{};
  const auto distance_epoch = distance_controller.next_wait_epoch();
  auto distance_phase = distance.value().PrepareForWait(
      {{0, 0, 12},
       distance_epoch,
       std::chrono::steady_clock::now() + std::chrono::seconds(1),
       &distance_controller,
       &distance_ready},
      query, 1, ids, output);
  UB_REQUIRE_TRUE(distance_phase.ok());
  const hnsw_urma::LaneOperationCount distance_lane{
      *distance_phase.value().owner_lane, 1};
  UB_REQUIRE_TRUE(
      distance_controller.BeginWait(std::span(&distance_lane, 1), 12).ok());
  UB_REQUIRE_TRUE(
      distance.value().PublishPrepared(distance_phase.value().ticket).ok());
  CompleteAll(engine, *lane, true);
  UB_REQUIRE_EQ(distance_ready.calls.load(), 1U);
  UB_REQUIRE_TRUE(distance.value().Finish(distance_phase.value().ticket).ok());
  UB_REQUIRE_EQ(output[0],
                hnsw_urma::ScalarL2(query, remote.Vector(4).value()));
  UB_REQUIRE_EQ(output[1],
                hnsw_urma::ScalarL2(query, remote.Vector(7).value()));
  UB_REQUIRE_TRUE(lane->Progress().ok());
  UB_REQUIRE_TRUE(lane->drained());
  UB_REQUIRE_TRUE(pool->drained());
  std::filesystem::remove_all(root);
}

namespace {
struct AdapterFixture {
  std::filesystem::path root;
  hnsw_urma::GlobalIndexFile global;
  hnsw_urma::ShardMap map;
  hnsw_urma::UpperGraphReplica upper;
  hnsw_urma::LocalShard local;
  std::vector<hnsw_urma::LocalShard> remote;
  std::vector<std::unique_ptr<hnsw_urma::MockLanePostEngine>> engines;
  std::vector<std::unique_ptr<hnsw_urma::ReadBatchLane>> lanes;
  std::vector<hnsw_urma::ReadBatchLane *> lane_views;
  hnsw_urma::PageAlignedBuffer memory;
  std::unique_ptr<hnsw_urma::SearchReadResources> pool;
  std::unique_ptr<hnsw_urma::UrmaGraphReader> graph;
  std::unique_ptr<hnsw_urma::PullVectorDistanceOracle> pull;
  hnsw_urma::PeerSegmentTable peers;

  AdapterFixture(std::uint32_t lane_count, std::uint32_t slots,
                 std::uint32_t submit_depth = 8, std::uint32_t records = 4) {
    static std::uint32_t serial = 0;
    root = std::filesystem::temp_directory_path() /
           ("hnsw_urma_search_read_adapter_" + std::to_string(++serial));
    std::filesystem::remove_all(root);
    const std::vector<float> vectors{0, 0, 1, 0, 2, 0, 3, 0, 4,
                                     0, 5, 0, 6, 0, 7, 0, 8, 0};
    hnsw_urma::ReferenceBuildOptions options;
    options.num_shards = 3;
    options.M = 2;
    options.M0 = 8;
    options.ef_search = 9;
    UB_REQUIRE_TRUE(
        hnsw_urma::BuildReferenceIndex(root, vectors, 9, 2, options, 100).ok());
    global = hnsw_urma::LoadGlobalIndexFile(root).value();
    map =
        hnsw_urma::ShardMap::Create(global.ranges, global.meta.ntotal).value();
    upper = hnsw_urma::LoadUpperGraphFile(root, global.meta).value();
    local =
        hnsw_urma::LoadShardFile(root, global.meta, global.ranges[0]).value();
    for (std::uint32_t rank = 1; rank < 3; ++rank)
      remote.push_back(
          hnsw_urma::LoadShardFile(root, global.meta, global.ranges[rank])
              .value());
    std::vector<hnsw_urma::PeerDataSegments> descriptors;
    std::array<std::uint32_t, 3> owners{
        std::numeric_limits<std::uint32_t>::max(), 0, lane_count - 1};
    for (std::uint32_t rank = 1; rank < 3; ++rank)
      descriptors.push_back({rank, rank,
                             remote[rank - 1].layer0_slots().size_bytes(),
                             remote[rank - 1].vectors().size_bytes()});
    peers =
        hnsw_urma::PeerSegmentTable::Create(global.meta, map, 0, descriptors)
            .value();
    memory = hnsw_urma::PageAlignedBuffer::Create(slots * 64).value();
    pool = hnsw_urma::SearchReadResources::Create(0, memory.logical_memory(),
                                                  slots, 64)
               .value();
    for (std::uint32_t index = 0; index < lane_count; ++index) {
      engines.push_back(std::make_unique<hnsw_urma::MockLanePostEngine>(
          hnsw_urma::MockLanePostEngine::Create(index, 2).value()));
      for (std::uint32_t rank = 1; rank < 3; ++rank) {
        UB_REQUIRE_TRUE(
            engines.back()
                ->RegisterSegment(rank, hnsw_urma::SegmentKind::kLayer0Graph,
                                  remote[rank - 1].graph_logical_memory())
                .ok());
        UB_REQUIRE_TRUE(
            engines.back()
                ->RegisterSegment(rank, hnsw_urma::SegmentKind::kVectors,
                                  remote[rank - 1].vector_logical_memory())
                .ok());
      }
      lanes.push_back(
          hnsw_urma::ReadBatchLane::Create(*engines.back(), index,
                                           {.peer_count = 3,
                                            .submit_depth = submit_depth,
                                            .in_flight_limit = 2})
              .value());
      lane_views.push_back(lanes.back().get());
    }
    UB_REQUIRE_TRUE(pool->BindLanes(lane_views).ok());
    auto *resource = pool.get();
    graph = std::make_unique<hnsw_urma::UrmaGraphReader>(
        hnsw_urma::UrmaGraphReader::Create(global.meta, map, upper, local,
                                           peers, lane_views, owners,
                                           std::span(&resource, 1), records)
            .value());
    pull = std::make_unique<hnsw_urma::PullVectorDistanceOracle>(
        hnsw_urma::PullVectorDistanceOracle::Create(
            global.meta, map, local, peers, lane_views, owners,
            std::span(&resource, 1), hnsw_urma::DistanceImplementation::kScalar,
            records, 8)
            .value());
  }
  ~AdapterFixture() { std::filesystem::remove_all(root); }
};

hnsw_urma::QueryWaitPreparation
Preparation(hnsw_urma::QueryRunController &controller,
            CapturingReadyPublisher &ready) {
  return {{0, 0, controller.generation()},
          controller.next_wait_epoch(),
          std::chrono::steady_clock::now() + std::chrono::seconds(10),
          &controller,
          &ready};
}
} // namespace

TEST(ReferenceIntegrationRemotePull,
     GraphMultiLaneRollsBackWholePreparationAndCompletesOutOfOrder) {
  AdapterFixture fixture(2, 2);
  CapturingReadyPublisher ready;
  hnsw_urma::QueryRunController controller(70, 2);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  std::array<hnsw_urma::ReadResultView, 1> held{};
  UB_REQUIRE_TRUE(fixture.pool->Acquire(held, 8).ok());
  const std::array<hnsw_urma::NodeToExpand, 2> nodes{{{3, 0}, {6, 0}}};
  hnsw_urma::NeighborBatch output;
  auto failed = fixture.graph->PrepareForWait(Preparation(controller, ready),
                                              nodes, &output);
  UB_REQUIRE_FALSE(failed.ok());
  UB_REQUIRE_EQ(fixture.pool->available(), 1U);
  UB_REQUIRE_EQ(controller.state(), hnsw_urma::RunState::kRunning);
  for (auto &lane : fixture.lanes)
    UB_REQUIRE_TRUE(lane->drained());
  fixture.pool->Return(held);
  auto phase =
      fixture.graph
          ->PrepareForWait(Preparation(controller, ready), nodes, &output)
          .value();
  UB_REQUIRE_TRUE(controller.BeginWait(phase.participating_lanes, 70).ok());
  UB_REQUIRE_TRUE(fixture.graph->PublishPrepared(phase.ticket).ok());
  CompleteAll(*fixture.engines[1], *fixture.lanes[1]);
  UB_REQUIRE_EQ(ready.calls.load(), 0U);
  CompleteAll(*fixture.engines[0], *fixture.lanes[0]);
  UB_REQUIRE_EQ(ready.calls.load(), 1U);
  CollectingGraphConsumer consumer;
  UB_REQUIRE_TRUE(fixture.graph->FinishConsume(phase.ticket, consumer).ok());
  UB_REQUIRE_EQ(consumer.values.size(), 2U);
  UB_REQUIRE_TRUE(fixture.pool->drained());
}

TEST(ReferenceIntegrationRemotePull,
     PullGroupsOnOneLaneRollBackSubmitReservationsAndShareGraphPool) {
  AdapterFixture fixture(1, 8, 2);
  CapturingReadyPublisher ready;
  hnsw_urma::QueryRunController controller(71, 1);
  UB_REQUIRE_TRUE(controller.Admit().ok());
  UB_REQUIRE_TRUE(controller.TryAcquire());
  const std::array<float, 2> query{1, 1};
  const std::array<hnsw_urma::node_id_t, 1> first_ids{3}, second_ids{6};
  std::array<float, 1> first_output{}, second_output{};
  hnsw_urma::NeighborBatch graph_output;
  const std::array<hnsw_urma::NodeToExpand, 1> nodes{{{3, 0}}};
  auto graph =
      fixture.graph
          ->PrepareForWait(Preparation(controller, ready), nodes, &graph_output)
          .value();
  auto first = fixture.pull
                   ->PrepareForWait(Preparation(controller, ready), query, 1,
                                    first_ids, first_output)
                   .value();
  auto second = fixture.pull->PrepareForWait(
      Preparation(controller, ready), query, 2, second_ids, second_output);
  UB_REQUIRE_FALSE(second.ok());
  UB_REQUIRE_EQ(second.status().code(),
                hnsw_urma::StatusCode::kResourceExhausted);
  UB_REQUIRE_EQ(fixture.pool->available(), 6U);
  // ScheduleDistances performs this rollback for every earlier group.
  UB_REQUIRE_TRUE(fixture.pull->CancelPrepared(first.ticket).ok());
  UB_REQUIRE_TRUE(fixture.graph->CancelPrepared(graph.ticket).ok());
  UB_REQUIRE_TRUE(fixture.pool->drained());
  first = fixture.pull
              ->PrepareForWait(Preparation(controller, ready), query, 1,
                               first_ids, first_output)
              .value();
  second = fixture.pull->PrepareForWait(Preparation(controller, ready), query,
                                        2, second_ids, second_output);
  UB_REQUIRE_TRUE(second.ok());
  const hnsw_urma::LaneOperationCount count{0, 2};
  UB_REQUIRE_TRUE(controller.BeginWait(std::span(&count, 1), 71).ok());
  UB_REQUIRE_TRUE(fixture.pull->PublishPrepared(first.ticket).ok());
  UB_REQUIRE_TRUE(fixture.pull->PublishPrepared(second.value().ticket).ok());
  CompleteAll(*fixture.engines[0], *fixture.lanes[0]);
  UB_REQUIRE_EQ(ready.calls.load(), 1U);
  UB_REQUIRE_TRUE(fixture.pull->Finish(first.ticket).ok());
  UB_REQUIRE_TRUE(fixture.pull->Finish(second.value().ticket).ok());
  UB_REQUIRE_TRUE(fixture.pool->drained());
}

TEST(ReferenceIntegrationRemotePull,
     PullTimeoutRetainsRecordAfterQueryDestructionUntilActualReclaim) {
  for (bool reclaim_first : {false, true}) {
    AdapterFixture fixture(1, 4, 4, 1);
    CapturingReadyPublisher ready;
    auto controller = std::make_unique<hnsw_urma::QueryRunController>(72, 1);
    UB_REQUIRE_TRUE(controller->Admit().ok());
    UB_REQUIRE_TRUE(controller->TryAcquire());
    const std::array<float, 2> query{1, 1};
    const std::array<hnsw_urma::node_id_t, 2> ids{3, 4};
    std::array<float, 2> output{-99, -99};
    auto phase = fixture.pull
                     ->PrepareForWait(Preparation(*controller, ready), query, 1,
                                      ids, output)
                     .value();
    const hnsw_urma::LaneOperationCount count{0, 1};
    UB_REQUIRE_TRUE(controller->BeginWait(std::span(&count, 1), 72).ok());
    UB_REQUIRE_TRUE(fixture.pull->PublishPrepared(phase.ticket).ok());
    UB_REQUIRE_TRUE(fixture.lanes[0]->Progress().ok());
    UB_REQUIRE_TRUE(controller
                        ->RequestFailure(72,
                                         hnsw_urma::TerminalOutcome::kTimedOut,
                                         hnsw_urma::StatusCode::kTimeout)
                        .ok());
    UB_REQUIRE_TRUE(fixture.lanes[0]->Progress().ok());
    UB_REQUIRE_EQ(ready.calls.load(), 1U);
    if (!reclaim_first) {
      UB_REQUIRE_FALSE(fixture.pull->Finish(phase.ticket).ok());
      controller.reset();
    }
    hnsw_urma::QueryRunController next(73, 1);
    UB_REQUIRE_TRUE(next.Admit().ok());
    UB_REQUIRE_TRUE(next.TryAcquire());
    auto blocked = fixture.pull->PrepareForWait(Preparation(next, ready), query,
                                                1, ids, output);
    UB_REQUIRE_FALSE(blocked.ok());
    UB_REQUIRE_EQ(fixture.pool->available(), 2U);
    CompleteAll(*fixture.engines[0], *fixture.lanes[0]);
    UB_REQUIRE_EQ(fixture.pool->PollReclaims(), 1U);
    if (reclaim_first) {
      UB_REQUIRE_EQ(fixture.pool->available(), 2U);
      UB_REQUIRE_FALSE(fixture.pull->Finish(phase.ticket).ok());
      controller.reset();
    }
    UB_REQUIRE_EQ(output[0], -99.0F);
    UB_REQUIRE_EQ(output[1], -99.0F);
    UB_REQUIRE_TRUE(fixture.pool->drained());
    auto reused =
        fixture.pull
            ->PrepareForWait(Preparation(next, ready), query, 1, ids, output)
            .value();
    UB_REQUIRE_EQ(reused.ticket.value, phase.ticket.value);
    UB_REQUIRE_TRUE(reused.ticket.generation != phase.ticket.generation);
    UB_REQUIRE_FALSE(fixture.pull->Finish(phase.ticket).ok());
    UB_REQUIRE_TRUE(fixture.pull->CancelPrepared(reused.ticket).ok());
  }
}

TEST(ReferenceIntegrationRemotePull,
     GraphFailureDoesNotConsumePartialResultsOrReleaseLateDma) {
  AdapterFixture fixture(2, 4, 4, 1);
  CapturingReadyPublisher ready;
  auto controller = std::make_unique<hnsw_urma::QueryRunController>(74, 2);
  UB_REQUIRE_TRUE(controller->Admit().ok());
  UB_REQUIRE_TRUE(controller->TryAcquire());
  const std::array<hnsw_urma::NodeToExpand, 3> nodes{{{3, 0}, {4, 0}, {6, 0}}};
  hnsw_urma::NeighborBatch output;
  auto phase =
      fixture.graph
          ->PrepareForWait(Preparation(*controller, ready), nodes, &output)
          .value();
  UB_REQUIRE_TRUE(controller->BeginWait(phase.participating_lanes, 74).ok());
  UB_REQUIRE_TRUE(fixture.graph->PublishPrepared(phase.ticket).ok());
  UB_REQUIRE_TRUE(fixture.lanes[0]->Progress().ok());
  UB_REQUIRE_TRUE(fixture.lanes[1]->Progress().ok());
  CompleteAll(*fixture.engines[1], *fixture.lanes[1]);
  UB_REQUIRE_TRUE(controller
                      ->RequestFailure(74,
                                       hnsw_urma::TerminalOutcome::kCancelled,
                                       hnsw_urma::StatusCode::kCancelled)
                      .ok());
  UB_REQUIRE_TRUE(fixture.lanes[0]->Progress().ok());
  UB_REQUIRE_EQ(ready.calls.load(), 1U);
  CollectingGraphConsumer consumer;
  UB_REQUIRE_FALSE(fixture.graph->FinishConsume(phase.ticket, consumer).ok());
  UB_REQUIRE_TRUE(consumer.values.empty());
  controller.reset();
  // The successful lane's slot is immediately reusable, the failed lane's
  // two slots stay occupied even after its Query controller was destroyed.
  UB_REQUIRE_EQ(fixture.pool->available(), 2U);
  CompleteAll(*fixture.engines[0], *fixture.lanes[0]);
  UB_REQUIRE_FALSE(fixture.pool->drained());
  UB_REQUIRE_EQ(fixture.pool->PollReclaims(), 1U);
  UB_REQUIRE_TRUE(fixture.pool->drained());
}
