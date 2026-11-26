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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "hnsw_urma/compute/dataset.hpp"
#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/index/builder.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/runtime/scheduler.hpp"
#include "hnsw_urma/search/local_io.hpp"
#include "hnsw_urma/search/mock_io.hpp"
#include "hnsw_urma/search/search.hpp"

namespace {

class ImmediateLookupGraphReader final : public hnsw_urma::GraphReader {
public:
  explicit ImmediateLookupGraphReader(hnsw_urma::MockGraphLookup lookup)
      : lookup_(std::move(lookup)) {}

  hnsw_urma::Result<hnsw_urma::GraphTicket>
  Submit(hnsw_urma::QueryKey query,
         std::span<const hnsw_urma::NodeToExpand> nodes,
         hnsw_urma::NeighborBatch *output) override {
    if (active_ticket_.has_value() || nodes.empty() || output == nullptr) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate lookup graph request is invalid");
    }
    output->Clear();
    for (const auto &node : nodes) {
      auto neighbors = lookup_(node.global_id, node.level);
      if (!neighbors.ok()) {
        return neighbors.status();
      }
      output->neighbors.insert(output->neighbors.end(),
                               neighbors.value().begin(),
                               neighbors.value().end());
      output->offsets.push_back(
          static_cast<std::uint32_t>(output->neighbors.size()));
    }
    active_ticket_ = hnsw_urma::GraphTicket{next_ticket_++, query.generation};
    return *active_ticket_;
  }

  hnsw_urma::Result<bool> Poll(hnsw_urma::GraphTicket ticket) override {
    if (!active_ticket_.has_value() || ticket != *active_ticket_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate lookup graph ticket is stale");
    }
    return true;
  }

  hnsw_urma::Status Finish(hnsw_urma::GraphTicket ticket) override {
    if (!active_ticket_.has_value() || ticket != *active_ticket_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate lookup graph ticket is stale");
    }
    active_ticket_.reset();
    return hnsw_urma::Status::Ok();
  }

  hnsw_urma::Result<std::span<const hnsw_urma::node_id_t>>
  ReadUpperNeighbors(hnsw_urma::node_id_t id,
                     std::int32_t level) const noexcept override {
    auto neighbors = lookup_(id, level);
    if (!neighbors.ok()) {
      return neighbors.status();
    }
    upper_neighbors_ = std::move(neighbors).value();
    return std::span<const hnsw_urma::node_id_t>(upper_neighbors_);
  }

private:
  hnsw_urma::MockGraphLookup lookup_;
  mutable std::vector<hnsw_urma::node_id_t> upper_neighbors_;
  std::optional<hnsw_urma::GraphTicket> active_ticket_;
  hnsw_urma::ticket_t next_ticket_{1};
};

class ImmediateGraphReader final : public hnsw_urma::GraphReader {
public:
  hnsw_urma::Result<hnsw_urma::GraphTicket>
  Submit(hnsw_urma::QueryKey, std::span<const hnsw_urma::NodeToExpand>,
         hnsw_urma::NeighborBatch *) override {
    return hnsw_urma::Status(hnsw_urma::StatusCode::kInvalidArgument,
                             "legacy graph path is disabled");
  }
  hnsw_urma::Result<bool> Poll(hnsw_urma::GraphTicket) override {
    return false;
  }
  hnsw_urma::Status Finish(hnsw_urma::GraphTicket ticket) override {
    if (!published_ || ticket != ticket_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate graph ticket is stale");
    }
    published_ = false;
    return hnsw_urma::Status::Ok();
  }
  bool SupportsPreparedWait() const noexcept override { return true; }
  hnsw_urma::Result<hnsw_urma::PreparedGraphPhase>
  PrepareForWait(hnsw_urma::QueryWaitPreparation,
                 std::span<const hnsw_urma::NodeToExpand> nodes,
                 hnsw_urma::NeighborBatch *output) override {
    output->neighbors.clear();
    output->offsets.clear();
    output->offsets.push_back(0);
    for (const auto &node : nodes) {
      output->neighbors.push_back(node.global_id == 0 ? 1 : 0);
      output->offsets.push_back(
          static_cast<std::uint32_t>(output->neighbors.size()));
    }
    ticket_ = {1, ++generation_};
    return hnsw_urma::PreparedGraphPhase{ticket_, {}, false};
  }
  hnsw_urma::Status PublishPrepared(hnsw_urma::GraphTicket ticket) override {
    if (ticket != ticket_ || published_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate graph publish is stale");
    }
    published_ = true;
    return hnsw_urma::Status::Ok();
  }
  hnsw_urma::Status CancelPrepared(hnsw_urma::GraphTicket) override {
    return hnsw_urma::Status::Ok();
  }

private:
  hnsw_urma::GraphTicket ticket_{};
  std::uint64_t generation_{0};
  bool published_{false};
};

class RetryOnceDistanceOracle final : public hnsw_urma::DistanceOracle {
public:
  hnsw_urma::Result<hnsw_urma::DistanceTicket>
  Submit(hnsw_urma::QueryKey, std::span<const float>, hnsw_urma::shard_id_t,
         std::span<const hnsw_urma::node_id_t>, std::span<float>,
         hnsw_urma::DistanceRequestOptions = {}) override {
    return hnsw_urma::Status(hnsw_urma::StatusCode::kInvalidArgument,
                             "legacy distance path is disabled");
  }
  hnsw_urma::Result<bool> Poll(hnsw_urma::DistanceTicket) override {
    return false;
  }
  hnsw_urma::Status Finish(hnsw_urma::DistanceTicket ticket) override {
    if (!published_ || ticket != ticket_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate distance ticket is stale");
    }
    published_ = false;
    return hnsw_urma::Status::Ok();
  }
  bool SupportsPreparedWait() const noexcept override { return true; }
  hnsw_urma::Result<hnsw_urma::PreparedDistanceOperation>
  PrepareForWait(hnsw_urma::QueryWaitPreparation, std::span<const float> query,
                 hnsw_urma::shard_id_t,
                 std::span<const hnsw_urma::node_id_t> ids,
                 std::span<float> output,
                 hnsw_urma::DistanceRequestOptions = {}) override {
    if (reject_once_) {
      reject_once_ = false;
      return hnsw_urma::Status(hnsw_urma::StatusCode::kResourceExhausted,
                               "injected reservation pressure");
    }
    for (std::size_t index = 0; index < ids.size(); ++index) {
      const auto delta = query[0] - static_cast<float>(ids[index]);
      output[index] = delta * delta;
    }
    ticket_ = {1, ++generation_};
    return hnsw_urma::PreparedDistanceOperation{ticket_, std::nullopt};
  }
  hnsw_urma::Status PublishPrepared(hnsw_urma::DistanceTicket ticket) override {
    if (ticket != ticket_ || published_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "immediate distance publish is stale");
    }
    published_ = true;
    return hnsw_urma::Status::Ok();
  }
  hnsw_urma::Status CancelPrepared(hnsw_urma::DistanceTicket) override {
    return hnsw_urma::Status::Ok();
  }

private:
  hnsw_urma::DistanceTicket ticket_{};
  std::uint64_t generation_{0};
  bool reject_once_{true};
  bool published_{false};
};

class InlineCompletionDistanceOracle final : public hnsw_urma::DistanceOracle {
public:
  hnsw_urma::Result<hnsw_urma::DistanceTicket>
  Submit(hnsw_urma::QueryKey, std::span<const float>, hnsw_urma::shard_id_t,
         std::span<const hnsw_urma::node_id_t>, std::span<float>,
         hnsw_urma::DistanceRequestOptions = {}) override {
    return hnsw_urma::Status(hnsw_urma::StatusCode::kInvalidArgument,
                             "legacy distance path is disabled");
  }
  hnsw_urma::Result<bool> Poll(hnsw_urma::DistanceTicket ticket) override {
    return ticket == ticket_ && terminal_;
  }
  hnsw_urma::Status Finish(hnsw_urma::DistanceTicket ticket) override {
    if (ticket != ticket_ || !terminal_) {
      return hnsw_urma::Status(
          hnsw_urma::StatusCode::kProtocolError,
          "inline-completion distance ticket is not terminal");
    }
    binding_.Reset();
    ticket_ = {};
    terminal_ = false;
    return hnsw_urma::Status::Ok();
  }
  bool SupportsPreparedWait() const noexcept override { return true; }
  hnsw_urma::Result<hnsw_urma::PreparedDistanceOperation>
  PrepareForWait(hnsw_urma::QueryWaitPreparation preparation,
                 std::span<const float> query, hnsw_urma::shard_id_t,
                 std::span<const hnsw_urma::node_id_t> ids,
                 std::span<float> output,
                 hnsw_urma::DistanceRequestOptions = {}) override {
    if (ticket_.value != 0 || preparation.controller == nullptr ||
        preparation.ready_publisher == nullptr) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "inline-completion distance operation overlaps");
    }
    for (std::size_t index = 0; index < ids.size(); ++index) {
      const auto delta = query[0] - static_cast<float>(ids[index]);
      output[index] = delta * delta;
    }
    ticket_ = {1, ++generation_};
    auto bound = binding_.Prepare(preparation.query, *preparation.controller,
                                  operation_, *preparation.ready_publisher,
                                  {0, 0, preparation.query.generation,
                                   preparation.wait_epoch, ticket_.value});
    if (!bound.ok()) {
      ticket_ = {};
      return bound;
    }
    return hnsw_urma::PreparedDistanceOperation{ticket_, 0};
  }
  hnsw_urma::Status PublishPrepared(hnsw_urma::DistanceTicket ticket) override {
    if (ticket != ticket_ || terminal_) {
      return hnsw_urma::Status(
          hnsw_urma::StatusCode::kProtocolError,
          "inline-completion distance publication is stale");
    }
    auto published = binding_.MarkPublished();
    if (!published.ok()) {
      return published;
    }
    terminal_ = true;
    return binding_.Complete(hnsw_urma::OperationTerminalOutcome::kComplete);
  }
  hnsw_urma::Status CancelPrepared(hnsw_urma::DistanceTicket ticket) override {
    if (ticket != ticket_) {
      return hnsw_urma::Status(
          hnsw_urma::StatusCode::kProtocolError,
          "inline-completion distance cancellation is stale");
    }
    binding_.Reset();
    ticket_ = {};
    terminal_ = false;
    return hnsw_urma::Status::Ok();
  }

private:
  hnsw_urma::DistanceTicket ticket_{};
  std::uint64_t generation_{0};
  hnsw_urma::QueryOperationState operation_;
  hnsw_urma::QueryOperationBinding binding_;
  bool terminal_{false};
};

class RecordingDistanceOracle final : public hnsw_urma::DistanceOracle {
public:
  hnsw_urma::Result<hnsw_urma::DistanceTicket>
  Submit(hnsw_urma::QueryKey query, std::span<const float> query_vector,
         hnsw_urma::shard_id_t, std::span<const hnsw_urma::node_id_t>,
         std::span<float> output,
         hnsw_urma::DistanceRequestOptions = {}) override {
    observed_query_data = query_vector.data();
    std::fill(output.begin(), output.end(), 0.0F);
    ticket_ = {1, query.generation};
    return ticket_;
  }

  hnsw_urma::Result<bool> Poll(hnsw_urma::DistanceTicket ticket) override {
    if (ticket != ticket_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "recording distance ticket is stale");
    }
    return false;
  }

  hnsw_urma::Status Finish(hnsw_urma::DistanceTicket ticket) override {
    if (ticket != ticket_) {
      return hnsw_urma::Status(hnsw_urma::StatusCode::kProtocolError,
                               "recording distance ticket is stale");
    }
    ticket_ = {};
    return hnsw_urma::Status::Ok();
  }

  const float *observed_query_data{nullptr};

private:
  hnsw_urma::DistanceTicket ticket_{};
};

std::vector<hnsw_urma::HeapEntry> BruteForce(const std::vector<float> &vectors,
                                             std::uint32_t dim,
                                             std::span<const float> query,
                                             std::size_t topk) {
  std::vector<hnsw_urma::HeapEntry> entries;
  for (std::size_t row = 0; row < vectors.size() / dim; ++row) {
    entries.push_back(
        {static_cast<hnsw_urma::node_id_t>(row),
         hnsw_urma::ScalarL2(
             query, std::span<const float>(vectors).subspan(row * dim, dim))});
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.distance < rhs.distance ||
                     (lhs.distance == rhs.distance && lhs.id < rhs.id);
            });
  entries.resize(topk);
  return entries;
}

} // namespace

TEST(ReferenceIntegrationLocalSearch,
     SearchBorrowsQueryWhenImmutableOwnerIsProvided) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 1;
  meta.dim = 2;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 1;
  meta.ef_search_default = 1;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 1;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 1}}, 1)
                 .value();
  ImmediateGraphReader graph;
  RecordingDistanceOracle distance;
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 1;
  params.expand_batch = 1;

  auto dataset = hnsw_urma::FloatDataset{1, 2, {0.25F, 0.5F}};
  const auto query = dataset.Row(0);
  const auto *query_data = query.data();
  auto owner = dataset.owner();
  std::weak_ptr<const void> lifetime = owner;
  auto created = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 77}, meta, map, graph, distance, query, params,
      std::chrono::steady_clock::now(), nullptr, nullptr, owner);
  UB_REQUIRE_TRUE(created.ok());
  auto task = std::move(created).value();
  owner.reset();
  dataset = {};

  UB_REQUIRE_FALSE(lifetime.expired());
  UB_REQUIRE_EQ(task->Step(), hnsw_urma::SearchStepResult::kYielded);
  UB_REQUIRE_EQ(distance.observed_query_data, query_data);

  task.reset();
  UB_REQUIRE_TRUE(lifetime.expired());
}

TEST(ReferenceIntegrationLocalSearch,
     SearchReservationPressureUsesWorkerLocalRetry) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 2;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 1;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 1;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 2}}, 2)
                 .value();
  ImmediateGraphReader graph;
  RetryOnceDistanceOracle distance;
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 1, 2, 2).value();
  auto ready =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 1, 1).value();
  const hnsw_urma::SearchRuntimeContext runtime{ready.get(), 1};
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.expand_batch = 1;
  const std::vector<float> query{0.2F};
  const hnsw_urma::QueryKey key{0, 0, 31};
  auto task = hnsw_urma::HnswSearchTask::Create(
                  key, meta, map, graph, distance, query, params,
                  std::chrono::steady_clock::now(), &runtime)
                  .value();

  UB_REQUIRE_EQ(task->Step(), hnsw_urma::SearchStepResult::kYielded);
  UB_REQUIRE_EQ(task->run_state(), hnsw_urma::RunState::kQueued);
  UB_REQUIRE_EQ(task->run_controller().wait_epoch(), 0U);
  UB_REQUIRE_EQ(task->run_controller().retry_epoch(), 1U);
  auto retry = task->TakeRetryHandle();
  UB_REQUIRE_TRUE(retry.has_value());
  UB_REQUIRE_TRUE(queues->ScheduleRetry(0, *retry).ok());
  auto runnable = hnsw_urma::test::TakeRunnable(
      *queues, 0, std::chrono::steady_clock::time_point::max());
  UB_REQUIRE_TRUE(runnable.has_value());
  UB_REQUIRE_EQ(runnable->source, hnsw_urma::RunnableSource::kRetry);
  UB_REQUIRE_TRUE(task->run_controller().ValidateRetryHandle(
      {runnable->query, runnable->epoch, {}}));
  UB_REQUIRE_EQ(task->Step(), hnsw_urma::SearchStepResult::kCompleted);
  UB_REQUIRE_EQ(task->result().neighbors.size(), 1U);
  UB_REQUIRE_EQ(task->result().neighbors.front().id, 0U);
}

TEST(ReferenceIntegrationLocalSearch,
     InlineCompletionPublishesReadyWithoutLocalRetry) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 2;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 1;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 1;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 2}}, 2)
                 .value();
  ImmediateGraphReader graph;
  InlineCompletionDistanceOracle distance;
  auto queues = hnsw_urma::SchedulerQueues::Create(1, 1, 2, 2).value();
  auto ready =
      hnsw_urma::SchedulerReadyPublisher::Create(*queues, 1, 1).value();
  const hnsw_urma::SearchRuntimeContext runtime{ready.get(), 1};
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.expand_batch = 1;
  const hnsw_urma::QueryKey key{0, 0, 73};
  const std::vector<float> query{0.2F};
  auto task = hnsw_urma::HnswSearchTask::Create(
                  key, meta, map, graph, distance, query, params,
                  std::chrono::steady_clock::now(), &runtime)
                  .value();

  for (std::uint32_t wakeup = 0; wakeup < 4; ++wakeup) {
    const auto step = task->Step();
    if (step == hnsw_urma::SearchStepResult::kCompleted) {
      break;
    }
    UB_REQUIRE_EQ(step, hnsw_urma::SearchStepResult::kYielded);
    UB_REQUIRE_EQ(task->run_state(), hnsw_urma::RunState::kQueued);
    UB_REQUIRE_FALSE(task->TakeRetryHandle().has_value());
    auto runnable = hnsw_urma::test::TakeRunnable(
        *queues, 0, std::chrono::steady_clock::now());
    UB_REQUIRE_TRUE(runnable.has_value());
    UB_REQUIRE_EQ(runnable->source, hnsw_urma::RunnableSource::kReady);
    UB_REQUIRE_TRUE(task->run_controller().ValidateReadyHandle(
        {runnable->query, runnable->epoch}));
  }
  UB_REQUIRE_EQ(task->run_state(), hnsw_urma::RunState::kCompleted);
  UB_REQUIRE_EQ(task->result().neighbors.size(), 1U);
  UB_REQUIRE_EQ(task->result().neighbors.front().id, 0U);
}

TEST(ReferenceIntegrationLocalSearch,
     ExternalQueryScratchSupportsSearchReadyProducerLanes) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 2;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 1;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 1;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 2}}, 2)
                 .value();
  ImmediateGraphReader graph;
  InlineCompletionDistanceOracle distance;
  constexpr std::uint32_t kCommunicationThreads = 2;
  constexpr std::uint32_t kSearchWorkers = 3;
  constexpr std::uint32_t kReadyProducerCount =
      kCommunicationThreads + kSearchWorkers;
  auto queues = hnsw_urma::SchedulerQueues::Create(kSearchWorkers,
                                                   kReadyProducerCount, 2, 2)
                    .value();
  auto ready = hnsw_urma::SchedulerReadyPublisher::Create(
                   *queues, kSearchWorkers, kReadyProducerCount)
                   .value();
  const hnsw_urma::SearchRuntimeContext runtime{ready.get(),
                                                kReadyProducerCount};
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.expand_batch = 1;
  auto scratch =
      hnsw_urma::QueryScratch::Create(meta, params, 2, kReadyProducerCount);
  UB_REQUIRE_TRUE(scratch.ok());
  const std::vector<float> query{0.2F};
  auto task = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 74}, meta, map, graph, distance, query, params,
      std::chrono::steady_clock::now(), &runtime, scratch.value().get());
  UB_REQUIRE_TRUE(task.ok());
}

TEST(ReferenceIntegrationLocalSearch,
     LocalHnswSearchMatchesBruteForceOnFullFixtureGraph) {
  const auto root =
      std::filesystem::temp_directory_path() / "hnsw_urma_local_search";
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

  auto global = hnsw_urma::LoadGlobalIndexFile(root);
  UB_REQUIRE_TRUE(global.ok());
  auto map = hnsw_urma::ShardMap::Create(global.value().ranges,
                                         global.value().meta.ntotal);
  auto upper = hnsw_urma::LoadUpperGraphFile(root, global.value().meta);
  UB_REQUIRE_TRUE(map.ok());
  UB_REQUIRE_TRUE(upper.ok());

  std::vector<hnsw_urma::LocalShard> shards;
  shards.reserve(2);
  for (const auto &range : global.value().ranges) {
    auto shard = hnsw_urma::LoadShardFile(root, global.value().meta, range);
    UB_REQUIRE_TRUE(shard.ok());
    shards.push_back(std::move(shard).value());
  }
  std::vector<const hnsw_urma::LocalShard *> shard_views{&shards[0],
                                                         &shards[1]};
  auto graph = hnsw_urma::LocalGraphReader::Create(map.value(), upper.value(),
                                                   shard_views);
  auto distance = hnsw_urma::LocalDistanceOracle::Create(
      map.value(), shard_views, hnsw_urma::Metric::kL2,
      hnsw_urma::DistanceImplementation::kScalar);
  UB_REQUIRE_TRUE(graph.ok());
  UB_REQUIRE_TRUE(distance.ok());

  const std::vector<float> query{1.1F, 0.2F};
  hnsw_urma::SearchParams params;
  params.topk = 3;
  params.ef_search = 8;
  params.expand_batch = 3;
  auto scratch =
      hnsw_urma::QueryScratch::Create(global.value().meta, params, 1).value();
  auto task = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 1}, global.value().meta, map.value(), graph.value(),
      distance.value(), query, params, std::chrono::steady_clock::now(),
      nullptr, scratch.get());
  UB_REQUIRE_TRUE(task.ok());
  auto result = hnsw_urma::RunToCompletion(std::move(task).value());
  UB_REQUIRE_TRUE(result.ok());
  UB_REQUIRE_EQ(result.value().neighbors, BruteForce(vectors, 2, query, 3));
  UB_REQUIRE_TRUE(result.value().profile.nodes_expanded > 0);
  UB_REQUIRE_FALSE(scratch->active());

  const std::vector<float> second_query{2.8F, 1.8F};
  params.layer0_sort_mode = hnsw_urma::Layer0SortMode::kUpperOnly;
  auto second_task = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 2}, global.value().meta, map.value(), graph.value(),
      distance.value(), second_query, params, std::chrono::steady_clock::now(),
      nullptr, scratch.get());
  UB_REQUIRE_TRUE(second_task.ok());
  auto second_result =
      hnsw_urma::RunToCompletion(std::move(second_task).value());
  UB_REQUIRE_TRUE(second_result.ok());
  UB_REQUIRE_EQ(second_result.value().neighbors,
                BruteForce(vectors, 2, second_query, 3));
  UB_REQUIRE_EQ(second_result.value().profile.routing_layer0_sorted_values, 0U);
  UB_REQUIRE_TRUE(second_result.value().profile.routing_layer0_unsorted_values >
                  0U);
  UB_REQUIRE_TRUE(second_result.value().profile.routing_group_owner_writes >
                  0U);
  UB_REQUIRE_TRUE(second_result.value().profile.routing_local_offset_writes >
                  0U);
  UB_REQUIRE_FALSE(scratch->active());
  std::filesystem::remove_all(root);
}

TEST(ReferenceIntegrationLocalSearch,
     UpperVisitedSkipsRepeatedDistancesAndProfilesEliminatedWork) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 6;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 2;
  meta.M0 = 2;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 3;
  meta.num_shards = 2;
  auto map =
      hnsw_urma::ShardMap::Create(
          std::vector<hnsw_urma::ShardRange>{{0, 0, 3}, {1, 3, 3}}, meta.ntotal)
          .value();
  ImmediateLookupGraphReader graph(
      [](hnsw_urma::node_id_t id, std::int32_t level)
          -> hnsw_urma::Result<std::vector<hnsw_urma::node_id_t>> {
        if (id == 0 && level == 3) {
          return std::vector<hnsw_urma::node_id_t>{1};
        }
        if (id == 1 && (level == 3 || level == 2)) {
          return std::vector<hnsw_urma::node_id_t>{0, 2};
        }
        return std::vector<hnsw_urma::node_id_t>{};
      });
  hnsw_urma::MockDistanceOracle distance(
      [](std::span<const float>,
         hnsw_urma::node_id_t id) -> hnsw_urma::Result<float> {
        switch (id) {
        case 0:
          return 9.0F;
        case 1:
          return 4.0F;
        case 2:
          return 5.0F;
        default:
          return 16.0F;
        }
      });
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.expand_batch = 1;
  params.profile_timing = true;
  const std::vector<float> query{0.0F};
  auto task = hnsw_urma::HnswSearchTask::Create({0, 0, 1}, meta, map, graph,
                                                distance, query, params)
                  .value();
  for (std::size_t step = 0; step < 100; ++step) {
    const auto outcome = task->Step();
    if (outcome == hnsw_urma::SearchStepResult::kCompleted) {
      break;
    }
    UB_REQUIRE_FALSE(outcome == hnsw_urma::SearchStepResult::kFailed);
    distance.CompleteAll();
  }
  UB_REQUIRE_EQ(task->query_state(), hnsw_urma::QueryState::kCompleted);
  UB_REQUIRE_EQ(task->result().neighbors.front().id, 1U);
  const auto &profile = task->result().profile;
  UB_REQUIRE_EQ(profile.upper_steps, 2U);
  UB_REQUIRE_EQ(profile.upper_same_level_moves, 1U);
  UB_REQUIRE_EQ(profile.upper_level_drops, 3U);
  UB_REQUIRE_EQ(profile.upper_empty_neighbor_level_drops, 1U);
  UB_REQUIRE_EQ(profile.upper_level_drop_runs, 1U);
  UB_REQUIRE_EQ(profile.upper_max_consecutive_level_drops, 3U);
  UB_REQUIRE_EQ(profile.upper_level_drop_runs_length_1, 0U);
  UB_REQUIRE_EQ(profile.upper_level_drop_runs_length_2, 0U);
  UB_REQUIRE_EQ(profile.upper_level_drop_runs_length_3, 1U);
  UB_REQUIRE_EQ(profile.upper_level_drop_runs_length_4_plus, 0U);
  UB_REQUIRE_EQ(profile.upper_distance_candidate_values, 5U);
  UB_REQUIRE_EQ(profile.upper_distance_visited_hits, 3U);
  UB_REQUIRE_EQ(profile.upper_distance_unique_values, 2U);
  UB_REQUIRE_EQ(profile.upper_distance_fully_visited_steps, 1U);
  UB_REQUIRE_EQ(profile.upper_remote_candidate_values, 5U);
  UB_REQUIRE_EQ(profile.upper_remote_visited_hits, 3U);
  UB_REQUIRE_EQ(profile.upper_remote_operations, 3U);
  UB_REQUIRE_EQ(profile.upper_eliminated_remote_operations, 1U);
  UB_REQUIRE_TRUE(task->result().timing.upper_elapsed_nanoseconds > 0U);
  UB_REQUIRE_TRUE(task->result().timing.layer0_elapsed_nanoseconds > 0U);
  UB_REQUIRE_TRUE(task->result().timing.upper_elapsed_nanoseconds >=
                  task->result().timing.upper_search_nanoseconds);
}

TEST(ReferenceIntegrationLocalSearch,
     MockSearchAllowsDelayedOutOfOrderPhasesAndStableCommit) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 6;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 2;
  meta.M0 = 5;
  meta.ef_search_default = 6;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 2;
  const std::vector<hnsw_urma::ShardRange> ranges{{0, 0, 3}, {1, 3, 3}};
  auto map = hnsw_urma::ShardMap::Create(ranges, 6).value();

  hnsw_urma::MockGraphReader graph(
      [](hnsw_urma::node_id_t id, std::int32_t level)
          -> hnsw_urma::Result<std::vector<hnsw_urma::node_id_t>> {
        static_cast<void>(level);
        std::vector<hnsw_urma::node_id_t> neighbors;
        for (hnsw_urma::node_id_t candidate = 0; candidate < 6; ++candidate) {
          if (candidate != id) {
            neighbors.push_back(candidate);
          }
        }
        return neighbors;
      });
  hnsw_urma::MockDistanceOracle distance(
      [](std::span<const float> query,
         hnsw_urma::node_id_t id) -> hnsw_urma::Result<float> {
        const float delta = query[0] - static_cast<float>(id);
        return delta * delta;
      });

  hnsw_urma::SearchParams params;
  params.topk = 2;
  params.ef_search = 6;
  params.expand_batch = 2;
  const std::vector<float> query{3.4F};
  auto task_result = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 11}, meta, map, graph, distance, query, params);
  UB_REQUIRE_TRUE(task_result.ok());
  auto task = std::move(task_result).value();
  for (std::size_t step = 0; step < 1000; ++step) {
    const auto outcome = task->Step();
    if (outcome == hnsw_urma::SearchStepResult::kCompleted) {
      break;
    }
    UB_REQUIRE_FALSE(outcome == hnsw_urma::SearchStepResult::kFailed);
    graph.CompleteAll();
    distance.CompleteAll(true);
  }
  UB_REQUIRE_EQ(task->query_state(), hnsw_urma::QueryState::kCompleted);
  const std::vector<hnsw_urma::HeapEntry> expected{{3, 0.16F}, {4, 0.36F}};
  UB_REQUIRE_EQ(task->result().neighbors.size(), expected.size());
  UB_REQUIRE_EQ(task->result().neighbors[0].id, expected[0].id);
  UB_REQUIRE_EQ(task->result().neighbors[1].id, expected[1].id);
  UB_REQUIRE_NEAR(task->result().neighbors[0].distance, expected[0].distance,
                  1e-5F);
  UB_REQUIRE_NEAR(task->result().neighbors[1].distance, expected[1].distance,
                  1e-5F);
}

TEST(ReferenceIntegrationLocalSearch,
     SearchStartsFromConfiguredSourceShardEntryPoint) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 6;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 2;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 2;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 3}, {1, 3, 3}}, 6)
                 .value();
  hnsw_urma::MockGraphReader graph(
      [](hnsw_urma::node_id_t,
         std::int32_t) -> hnsw_urma::Result<std::vector<hnsw_urma::node_id_t>> {
        return std::vector<hnsw_urma::node_id_t>{};
      });
  std::vector<hnsw_urma::node_id_t> distance_ids;
  hnsw_urma::MockDistanceOracle distance(
      [&distance_ids](std::span<const float> query,
                      hnsw_urma::node_id_t id) -> hnsw_urma::Result<float> {
        distance_ids.push_back(id);
        const float delta = query[0] - static_cast<float>(id);
        return delta * delta;
      });
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.expand_batch = 1;
  params.entry_point = 3;
  params.entry_level = 0;
  const std::vector<float> query{3.1F};

  auto task = hnsw_urma::HnswSearchTask::Create({1, 0, 1}, meta, map, graph,
                                                distance, query, params)
                  .value();
  for (std::size_t step = 0; step < 100; ++step) {
    const auto outcome = task->Step();
    if (outcome == hnsw_urma::SearchStepResult::kCompleted) {
      break;
    }
    UB_REQUIRE_FALSE(outcome == hnsw_urma::SearchStepResult::kFailed);
    graph.CompleteAll();
    distance.CompleteAll();
  }
  UB_REQUIRE_EQ(task->query_state(), hnsw_urma::QueryState::kCompleted);
  UB_REQUIRE_FALSE(distance_ids.empty());
  UB_REQUIRE_EQ(distance_ids.front(), 3U);
  UB_REQUIRE_EQ(task->result().neighbors.front().id, 3U);

  auto wrong_owner = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 2}, meta, map, graph, distance, query, params);
  UB_REQUIRE_FALSE(wrong_owner.ok());
}

TEST(ReferenceIntegrationLocalSearch, SearchPushdownUsesUpperDistanceFilter) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 4;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 2;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 1;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 4}}, 4)
                 .value();
  hnsw_urma::MockGraphReader graph(
      [](hnsw_urma::node_id_t id,
         std::int32_t) -> hnsw_urma::Result<std::vector<hnsw_urma::node_id_t>> {
        if (id == 0) {
          return std::vector<hnsw_urma::node_id_t>{1, 2};
        }
        if (id == 2) {
          return std::vector<hnsw_urma::node_id_t>{3};
        }
        return std::vector<hnsw_urma::node_id_t>{};
      });
  hnsw_urma::MockDistanceOracle distance(
      [](std::span<const float> query,
         hnsw_urma::node_id_t id) -> hnsw_urma::Result<float> {
        const float delta = query[0] - static_cast<float>(id);
        return delta * delta;
      });
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.expand_batch = 1;
  const std::vector<float> query{3.0F};
  const auto run = [&](std::uint64_t generation) {
    auto task =
        hnsw_urma::HnswSearchTask::Create({0, 0, generation}, meta, map, graph,
                                          distance, query, params)
            .value();
    for (std::size_t step = 0; step < 1000; ++step) {
      const auto outcome = task->Step();
      if (outcome == hnsw_urma::SearchStepResult::kCompleted) {
        break;
      }
      UB_REQUIRE_FALSE(outcome == hnsw_urma::SearchStepResult::kFailed);
      graph.CompleteAll();
      distance.CompleteAll(true);
    }
    UB_REQUIRE_EQ(task->query_state(), hnsw_urma::QueryState::kCompleted);
    return task->result().profile;
  };

  const auto enabled = run(1);
  UB_REQUIRE_TRUE(distance.saw_upper_distance());
  UB_REQUIRE_TRUE(enabled.nodes_expanded > 0U);
}

TEST(ReferenceIntegrationLocalSearch,
     MockSearchConvergesSubmitFailureAndOperationTimeout) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 2;
  meta.dim = 1;
  meta.metric = 0;
  meta.M = 1;
  meta.M0 = 1;
  meta.ef_search_default = 2;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 1;
  auto map = hnsw_urma::ShardMap::Create(
                 std::vector<hnsw_urma::ShardRange>{{0, 0, 2}}, 2)
                 .value();
  hnsw_urma::MockGraphReader graph([](auto, auto) {
    return hnsw_urma::Result<std::vector<hnsw_urma::node_id_t>>(
        std::vector<hnsw_urma::node_id_t>{1});
  });
  hnsw_urma::MockDistanceOracle distance([](auto query, auto id) {
    const float delta = query[0] - static_cast<float>(id);
    return hnsw_urma::Result<float>(delta * delta);
  });
  hnsw_urma::SearchParams params;
  params.topk = 1;
  params.ef_search = 2;
  params.operation_timeout = std::chrono::milliseconds(5);
  params.query_timeout = std::chrono::milliseconds(20);
  const std::vector<float> query{0.2F};

  distance.RejectNext(hnsw_urma::Status(
      hnsw_urma::StatusCode::kResourceExhausted, "descriptor queue full"));
  auto failed = hnsw_urma::HnswSearchTask::Create({0, 0, 1}, meta, map, graph,
                                                  distance, query, params)
                    .value();
  UB_REQUIRE_EQ(failed->Step(), hnsw_urma::SearchStepResult::kYielded);
  UB_REQUIRE_EQ(failed->Step(), hnsw_urma::SearchStepResult::kFailed);
  UB_REQUIRE_EQ(failed->status().code(),
                hnsw_urma::StatusCode::kResourceExhausted);

  auto cooperative = hnsw_urma::HnswSearchTask::Create(
                         {0, 2, 3}, meta, map, graph, distance, query, params)
                         .value();
  UB_REQUIRE_EQ(cooperative->Step(), hnsw_urma::SearchStepResult::kYielded);
  UB_REQUIRE_FALSE(cooperative->ProgressWaiting().value());
  distance.CompleteAll();
  UB_REQUIRE_TRUE(cooperative->ProgressWaiting().value());
  UB_REQUIRE_EQ(cooperative->run_state(), hnsw_urma::RunState::kQueued);
  UB_REQUIRE_EQ(cooperative->Step(), hnsw_urma::SearchStepResult::kYielded);

  const auto admission = std::chrono::steady_clock::now();
  auto timed =
      hnsw_urma::HnswSearchTask::Create({0, 1, 2}, meta, map, graph, distance,
                                        query, params, admission)
          .value();
  UB_REQUIRE_EQ(timed->Step(admission), hnsw_urma::SearchStepResult::kYielded);
  UB_REQUIRE_EQ(timed->Step(admission + std::chrono::milliseconds(6)),
                hnsw_urma::SearchStepResult::kFailed);
  UB_REQUIRE_EQ(timed->status().code(), hnsw_urma::StatusCode::kTimeout);
}
