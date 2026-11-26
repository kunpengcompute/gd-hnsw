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

#include "hnsw_urma/search/remote_io.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <utility>

#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/core/log.hpp"

namespace hnsw_urma {
namespace {

constexpr std::size_t kMaxGraphNodesPerOperation = 64;
constexpr auto kNoLane = std::numeric_limits<std::uint32_t>::max();

OperationTerminalOutcome ToOperationOutcome(StatusCode status) noexcept {
  if (status == StatusCode::kOk) {
    return OperationTerminalOutcome::kComplete;
  }
  if (status == StatusCode::kTimeout) {
    return OperationTerminalOutcome::kTimedOut;
  }
  if (status == StatusCode::kCancelled) {
    return OperationTerminalOutcome::kCancelled;
  }
  return OperationTerminalOutcome::kFailed;
}

template <typename Operation>
void RecordFirstFailure(Operation &operation, StatusCode status) noexcept {
  if (status == StatusCode::kOk) {
    return;
  }
  auto expected = StatusCode::kOk;
  static_cast<void>(operation.aggregate_status.compare_exchange_strong(
      expected, status, std::memory_order_relaxed, std::memory_order_relaxed));
}

} // namespace

Result<PeerSegmentTable>
PeerSegmentTable::Create(const GlobalIndexMeta &meta, const ShardMap &shard_map,
                         shard_id_t local_shard,
                         std::span<const PeerDataSegments> peers) {
  if (local_shard >= shard_map.ranges().size() ||
      meta.num_shards != shard_map.ranges().size()) {
    return Status(StatusCode::kInvalidArgument,
                  "peer segment table local shard is invalid");
  }
  try {
    PeerSegmentTable table;
    table.meta_ = meta;
    table.shard_map_ = &shard_map;
    table.local_shard_ = local_shard;
    table.peers_.assign(shard_map.ranges().size(), PeerDataSegments{});
    std::vector<bool> present(shard_map.ranges().size(), false);
    present[local_shard] = true;
    for (const auto &peer : peers) {
      if (peer.shard_id >= table.peers_.size() ||
          peer.shard_id == local_shard || peer.peer_rank != peer.shard_id ||
          present[peer.shard_id]) {
        return Status(
            StatusCode::kInvalidArgument,
            "peer segment descriptors must be unique and rank ordered");
      }
      auto range = shard_map.Range(peer.shard_id);
      auto graph_slots = range.ok() ? CheckedMultiply<std::uint64_t>(
                                          range.value().local_count, meta.M0)
                                    : Result<std::uint64_t>(range.status());
      auto graph_bytes =
          graph_slots.ok() ? CheckedMultiply<std::uint64_t>(graph_slots.value(),
                                                            sizeof(node_id_t))
                           : Result<std::uint64_t>(graph_slots.status());
      auto vector_slots = range.ok() ? CheckedMultiply<std::uint64_t>(
                                           range.value().local_count, meta.dim)
                                     : Result<std::uint64_t>(range.status());
      auto vector_bytes = vector_slots.ok()
                              ? CheckedMultiply<std::uint64_t>(
                                    vector_slots.value(), sizeof(float))
                              : Result<std::uint64_t>(vector_slots.status());
      if (!graph_bytes.ok() || !vector_bytes.ok()) {
        return Status(StatusCode::kOutOfRange,
                      "peer segment byte requirement overflows");
      }
      if (peer.graph_bytes < graph_bytes.value() ||
          peer.vector_bytes < vector_bytes.value()) {
        return Status(StatusCode::kInvalidArgument,
                      "peer segment is smaller than its shard shape");
      }
      table.peers_[peer.shard_id] = peer;
      present[peer.shard_id] = true;
    }
    if (std::find(present.begin(), present.end(), false) != present.end()) {
      return Status(StatusCode::kInvalidArgument,
                    "peer segment table is incomplete");
    }
    return table;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate peer segment table");
  }
}

Result<RemoteReadLocation>
PeerSegmentTable::Graph(node_id_t id) const noexcept {
  auto location = shard_map_->Locate(id);
  if (!location.ok()) {
    return location.status();
  }
  return Graph(location.value().shard_id, location.value().local_id);
}

Result<RemoteReadLocation>
PeerSegmentTable::Graph(shard_id_t shard,
                        std::uint64_t local_id) const noexcept {
  if (shard >= peers_.size() || shard == local_shard_) {
    return Status(StatusCode::kInvalidArgument,
                  "local graph node does not require remote READ");
  }
  auto slot = CheckedMultiply<std::uint64_t>(local_id, meta_.M0);
  auto offset = slot.ok() ? CheckedMultiply<std::uint64_t>(slot.value(),
                                                           sizeof(node_id_t))
                          : Result<std::uint64_t>(slot.status());
  auto bytes = CheckedMultiply<std::uint64_t>(meta_.M0, sizeof(node_id_t));
  if (!offset.ok() || !bytes.ok() ||
      bytes.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "remote graph READ offset overflows");
  }
  const auto &peer = peers_[shard];
  if (offset.value() > peer.graph_bytes ||
      bytes.value() > peer.graph_bytes - offset.value()) {
    return Status(StatusCode::kOutOfRange,
                  "remote graph READ exceeds the peer segment");
  }
  return RemoteReadLocation{peer.peer_rank, SegmentKind::kLayer0Graph,
                            offset.value(),
                            static_cast<std::size_t>(bytes.value())};
}

Result<RemoteReadLocation>
PeerSegmentTable::Vector(node_id_t id) const noexcept {
  auto location = shard_map_->Locate(id);
  if (!location.ok()) {
    return location.status();
  }
  return Vector(location.value().shard_id, location.value().local_id);
}

Result<RemoteReadLocation>
PeerSegmentTable::Vector(shard_id_t shard,
                         std::uint64_t local_id) const noexcept {
  if (shard >= peers_.size() || shard == local_shard_) {
    return Status(StatusCode::kInvalidArgument,
                  "local vector does not require remote READ");
  }
  auto slot = CheckedMultiply<std::uint64_t>(local_id, meta_.dim);
  auto offset =
      slot.ok() ? CheckedMultiply<std::uint64_t>(slot.value(), sizeof(float))
                : Result<std::uint64_t>(slot.status());
  auto bytes = CheckedMultiply<std::uint64_t>(meta_.dim, sizeof(float));
  if (!offset.ok() || !bytes.ok() ||
      bytes.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "remote vector READ offset overflows");
  }
  const auto &peer = peers_[shard];
  if (offset.value() > peer.vector_bytes ||
      bytes.value() > peer.vector_bytes - offset.value()) {
    return Status(StatusCode::kOutOfRange,
                  "remote vector READ exceeds the peer segment");
  }
  return RemoteReadLocation{peer.peer_rank, SegmentKind::kVectors,
                            offset.value(),
                            static_cast<std::size_t>(bytes.value())};
}

Result<UrmaGraphReader> UrmaGraphReader::Create(
    const GlobalIndexMeta &meta, const ShardMap &shard_map,
    const UpperGraphReplica &upper, const LocalShard &local_shard,
    const PeerSegmentTable &peers, std::span<ReadBatchLane *const> lanes,
    std::span<const std::uint32_t> read_owner_by_rank,
    std::span<SearchReadResources *const> resources,
    std::size_t max_operations_per_search) {
  if (local_shard.range().shard_id != peers.local_shard() || lanes.empty() ||
      max_operations_per_search == 0 ||
      max_operations_per_search > std::numeric_limits<std::uint32_t>::max() ||
      read_owner_by_rank.empty() || resources.empty() ||
      max_operations_per_search >
          std::numeric_limits<std::uint32_t>::max() / resources.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "remote graph reader configuration is invalid");
  }
  try {
    UrmaGraphReader reader;
    reader.meta_ = &meta;
    reader.shard_map_ = &shard_map;
    reader.upper_ = &upper;
    reader.local_shard_ = &local_shard;
    reader.peers_ = &peers;
    reader.lanes_.assign(lanes.begin(), lanes.end());
    reader.resources_.assign(resources.begin(), resources.end());
    reader.partitions_.resize(resources.size());
    for (std::size_t search = 0; search < resources.size(); ++search) {
      if (resources[search] == nullptr ||
          resources[search]->owner_search() != search ||
          resources[search]->slot_bytes() < meta.M0 * sizeof(node_id_t))
        return Status(StatusCode::kInvalidArgument,
                      "Search READ pool shape is invalid");
      reader.partitions_[search].free_records.reserve(
          max_operations_per_search);
    }
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
      if (lanes[lane] == nullptr || lanes[lane]->owner_lane() != lane ||
          lanes[lane]->config().search_workers != resources.size())
        return Status(StatusCode::kInvalidArgument,
                      "READ lane topology is invalid");
    }
    reader.read_owner_by_rank_.assign(read_owner_by_rank.begin(),
                                      read_owner_by_rank.end());
    for (const auto owner : reader.read_owner_by_rank_) {
      if (owner != kNoLane && owner >= reader.lanes_.size()) {
        return Status(StatusCode::kInvalidArgument,
                      "graph READ owner map references a missing lane");
      }
    }
    reader.prepared_operation_count_ = static_cast<std::uint32_t>(
        max_operations_per_search * resources.size());
    reader.prepared_operations_ =
        std::make_unique<PreparedOperation[]>(reader.prepared_operation_count_);
    for (std::uint32_t slot = 0; slot < reader.prepared_operation_count_;
         ++slot) {
      auto &operation = reader.prepared_operations_[slot];
      const auto search = slot / max_operations_per_search;
      operation.partition = &reader.partitions_[search];
      operation.resources = resources[search];
      operation.slot = slot;
      operation.partition->free_records.push_back(slot);
      operation.nodes.reserve(kMaxGraphNodesPerOperation);
      operation.lane_count = static_cast<std::uint32_t>(reader.lanes_.size());
      operation.lane_batches =
          std::make_unique<LaneBatch[]>(operation.lane_count);
      operation.participating_lanes.reserve(operation.lane_count);
      for (std::uint32_t lane = 0; lane < operation.lane_count; ++lane) {
        operation.lane_batches[lane].lane = reader.lanes_[lane];
        operation.lane_batches[lane].parent = &operation;
        operation.lane_batches[lane].batch.ranges.reserve(
            kMaxGraphNodesPerOperation);
        operation.lane_batches[lane].requests.reserve(
            kMaxGraphNodesPerOperation);
        operation.lane_batches[lane].results.reserve(
            kMaxGraphNodesPerOperation);
      }
    }
    return Result<UrmaGraphReader>(std::move(reader));
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate remote graph reader");
  }
}

UrmaGraphReader::UrmaGraphReader(UrmaGraphReader &&other) noexcept
    : meta_(other.meta_), shard_map_(other.shard_map_), upper_(other.upper_),
      local_shard_(other.local_shard_), peers_(other.peers_),
      lanes_(std::move(other.lanes_)),
      read_owner_by_rank_(std::move(other.read_owner_by_rank_)),
      prepared_operations_(std::move(other.prepared_operations_)),
      prepared_operation_count_(other.prepared_operation_count_),
      resources_(std::move(other.resources_)),
      partitions_(std::move(other.partitions_)) {}

Result<std::span<const node_id_t>>
UrmaGraphReader::ReadUpperNeighbors(node_id_t id,
                                    std::int32_t level) const noexcept {
  return upper_->Neighbors(id, level);
}

Result<ReadBatchLane *>
UrmaGraphReader::LaneFor(std::uint32_t peer_rank) const noexcept {
  if (peer_rank >= read_owner_by_rank_.size() ||
      read_owner_by_rank_[peer_rank] >= lanes_.size()) {
    return Status(StatusCode::kProtocolError,
                  "remote graph rank has no static READ owner");
  }
  return lanes_[read_owner_by_rank_[peer_rank]];
}

Status UrmaGraphReader::CompleteBatch(void *context,
                                      ReadBatch &batch) noexcept {
  auto &lane_batch = *static_cast<LaneBatch *>(context);
  auto *parent = lane_batch.parent;
  const auto status = batch.status;
  RecordFirstFailure(*parent, status);
  const auto outcome = ToOperationOutcome(status);
  return lane_batch.binding.Complete(outcome);
}

StatusCode UrmaGraphReader::CheckBatchControl(
    void *context, std::chrono::steady_clock::time_point) noexcept {
  return static_cast<LaneBatch *>(context)->binding.CancellationCode();
}

void UrmaGraphReader::ResetPreparedOperation(
    PreparedOperation &operation) noexcept {
  for (std::uint32_t lane = 0; lane < operation.lane_count; ++lane) {
    auto &lane_batch = operation.lane_batches[lane];
    if (lane_batch.binding.prepared()) {
      lane_batch.binding.Reset();
    }
    if (lane_batch.results_owned) {
      operation.resources->Return(lane_batch.results);
      lane_batch.results_owned = false;
    }
    lane_batch.batch.reclaim_required = lane_batch.batch.reclaim_received =
        false;
    lane_batch.active = false;
    lane_batch.requests.clear();
    lane_batch.results.clear();
    lane_batch.batch.ticket = {};
    lane_batch.batch.requests = {};
    lane_batch.batch.terminal_context = nullptr;
    lane_batch.batch.terminal_callback = nullptr;
    lane_batch.batch.control_context = nullptr;
    lane_batch.batch.control_check = nullptr;
    lane_batch.batch.state.store(ReadBatchState::kFree,
                                 std::memory_order_release);
  }
  operation.ticket = {};
  operation.nodes.clear();
  operation.participating_lanes.clear();
  operation.aggregate_status.store(StatusCode::kOk, std::memory_order_relaxed);
  operation.state.store(PreparedOperationState::kFree,
                        std::memory_order_release);
  operation.resources->ReturnRecord();
  operation.partition->free_records.push_back(operation.slot);
}

void UrmaGraphReader::ConsumeOperation(PreparedOperation &operation) noexcept {
  operation.consumed = true;
  bool retained = false;
  for (std::uint32_t lane = 0; lane < operation.lane_count; ++lane) {
    auto &entry = operation.lane_batches[lane];
    if (entry.binding.prepared())
      entry.binding.Reset();
    if (entry.batch.reclaim_required && !entry.batch.reclaim_received) {
      retained = true;
    } else if (entry.results_owned) {
      operation.resources->Return(entry.results);
      entry.results_owned = false;
    }
  }
  operation.nodes.clear();
  if (!retained)
    ResetPreparedOperation(operation);
}

void UrmaGraphReader::ReclaimBatch(void *context, ReadBatch &) noexcept {
  auto &entry = *static_cast<LaneBatch *>(context);
  if (entry.parent->consumed)
    ConsumeOperation(*entry.parent);
}

Result<PreparedGraphPhase>
UrmaGraphReader::PrepareForWait(QueryWaitPreparation preparation,
                                std::span<const NodeToExpand> nodes,
                                NeighborBatch *output) {
  if (preparation.controller == nullptr ||
      preparation.ready_publisher == nullptr ||
      preparation.query.generation == 0 || preparation.wait_epoch == 0 ||
      preparation.operation_deadline ==
          std::chrono::steady_clock::time_point{} ||
      preparation.controller->generation() != preparation.query.generation ||
      output == nullptr || nodes.empty() ||
      nodes.size() > kMaxGraphNodesPerOperation) {
    return Status(StatusCode::kInvalidArgument,
                  "prepared graph phase preparation is invalid");
  }

  const auto owner = preparation.controller->owner_search();
  if (owner >= resources_.size())
    return Status(StatusCode::kInvalidArgument,
                  "READ Search owner is out of range");
  auto &partition = partitions_[owner];
  if (partition.free_records.empty())
    return Status(StatusCode::kResourceExhausted,
                  "prepared graph Search records are full");
  const auto operation_slot = partition.free_records.back();
  partition.free_records.pop_back();
  auto *operation = &prepared_operations_[operation_slot];
  operation->resources->AcquireRecord();
  operation->consumed = false;
  operation->state.store(PreparedOperationState::kPreparing,
                         std::memory_order_relaxed);

  const auto fail = [&](Status status) -> Result<PreparedGraphPhase> {
    for (std::uint32_t lane = 0; lane < operation->lane_count; ++lane) {
      auto &lane_batch = operation->lane_batches[lane];
      if (lane_batch.batch.state.load(std::memory_order_acquire) ==
          ReadBatchState::kPrepared) {
        static_cast<void>(
            lane_batch.lane->CancelPreparedBatch(lane_batch.batch));
      }
    }
    ResetPreparedOperation(*operation);
    return status;
  };

  auto generation = partition.next_ticket++;
  if (generation == 0)
    generation = partition.next_ticket++;
  operation->ticket = {static_cast<ticket_t>(operation_slot) + 1U, generation};
  operation->nodes.assign(nodes.size(), NodeSource{});
  operation->participating_lanes.clear();
  operation->aggregate_status.store(StatusCode::kOk, std::memory_order_relaxed);
  for (std::uint32_t lane = 0; lane < operation->lane_count; ++lane) {
    auto &lane_batch = operation->lane_batches[lane];
    lane_batch.active = false;
    lane_batch.requests.clear();
    lane_batch.results.clear();
  }

  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const auto &node = nodes[index];
    auto &source = operation->nodes[index];
    if (node.level > 0) {
      auto neighbors = ReadUpperNeighbors(node.global_id, node.level);
      if (!neighbors.ok()) {
        return fail(neighbors.status());
      }
      source.local_neighbors = neighbors.value();
      continue;
    }
    auto location = shard_map_->Locate(node.global_id);
    if (!location.ok()) {
      return fail(location.status());
    }
    if (location.value().shard_id == peers_->local_shard()) {
      source.local_neighbors =
          local_shard_->TrustedNeighbors(location.value().local_id);
      continue;
    }
    auto remote =
        peers_->Graph(location.value().shard_id, location.value().local_id);
    if (!remote.ok()) {
      return fail(remote.status());
    }
    auto lane = LaneFor(remote.value().peer_rank);
    if (!lane.ok()) {
      return fail(lane.status());
    }
    const auto lane_index = lane.value()->owner_lane();
    if (lane_index >= operation->lane_count ||
        operation->lane_batches[lane_index].lane != lane.value()) {
      return fail(
          Status(StatusCode::kProtocolError,
                 "prepared graph READ owner lane is outside the phase table"));
    }
    auto &lane_batch = operation->lane_batches[lane_index];
    lane_batch.active = true;
    source.lane = lane_index;
    source.result_index = static_cast<std::uint32_t>(lane_batch.results.size());
    lane_batch.results.push_back({});
    lane_batch.requests.push_back({remote.value().peer_rank,
                                   remote.value().segment,
                                   remote.value().offset,
                                   {}});
  }

  for (std::uint32_t lane = 0; lane < operation->lane_count; ++lane) {
    auto &lane_batch = operation->lane_batches[lane];
    if (!lane_batch.active) {
      continue;
    }
    auto acquired = operation->resources->Acquire(
        lane_batch.results, meta_->M0 * sizeof(node_id_t));
    if (!acquired.ok())
      return fail(acquired);
    lane_batch.results_owned = true;
    for (std::size_t index = 0; index < lane_batch.requests.size(); ++index) {
      const auto &view = lane_batch.results[index];
      lane_batch.requests[index].destination = {
          const_cast<std::byte *>(view.data), view.bytes};
    }
    std::sort(lane_batch.requests.begin(), lane_batch.requests.end(),
              [](const ReadRequest &left, const ReadRequest &right) {
                if (left.peer_rank != right.peer_rank) {
                  return left.peer_rank < right.peer_rank;
                }
                return static_cast<std::uint8_t>(left.remote_segment) <
                       static_cast<std::uint8_t>(right.remote_segment);
              });
    auto query_ticket = partition.next_ticket++;
    if (query_ticket == 0)
      query_ticket = partition.next_ticket++;
    auto bound = lane_batch.binding.Prepare(
        preparation.query, *preparation.controller, lane_batch.query_operation,
        *preparation.ready_publisher,
        {operation_slot, lane, preparation.query.generation,
         preparation.wait_epoch, query_ticket});
    if (!bound.ok()) {
      return fail(bound);
    }
    lane_batch.batch.owner_search = owner;
    lane_batch.batch.reclaim_context = &lane_batch;
    lane_batch.batch.reclaim_callback = ReclaimBatch;
    lane_batch.batch.requests = lane_batch.requests;
    lane_batch.batch.deadline = preparation.operation_deadline;
    lane_batch.batch.terminal_context = &lane_batch;
    lane_batch.batch.terminal_callback = CompleteBatch;
    lane_batch.batch.control_context = &lane_batch;
    lane_batch.batch.control_check = CheckBatchControl;
    auto prepared = lane_batch.lane->PrepareBatch(lane_batch.batch);
    if (!prepared.ok()) {
      return fail(prepared.status());
    }
    operation->participating_lanes.push_back({lane, 1});
  }

  operation->state.store(PreparedOperationState::kPrepared,
                         std::memory_order_release);
  return PreparedGraphPhase{operation->ticket, operation->participating_lanes,
                            !operation->participating_lanes.empty()};
}

Status UrmaGraphReader::PublishPrepared(GraphTicket ticket) {
  if (ticket.value == 0 || ticket.value > prepared_operation_count_) {
    return Status(StatusCode::kProtocolError,
                  "prepared graph ticket is invalid");
  }
  auto &operation = prepared_operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) !=
          PreparedOperationState::kPrepared) {
    return Status(StatusCode::kProtocolError, "prepared graph ticket is stale");
  }
  for (std::uint32_t lane = 0; lane < operation.lane_count; ++lane) {
    auto &lane_batch = operation.lane_batches[lane];
    if (!lane_batch.active) {
      continue;
    }
    auto published = lane_batch.binding.MarkPublished();
    if (!published.ok()) {
      return published;
    }
  }
  operation.state.store(PreparedOperationState::kPublished,
                        std::memory_order_release);
  for (std::uint32_t lane = 0; lane < operation.lane_count; ++lane) {
    auto &lane_batch = operation.lane_batches[lane];
    if (lane_batch.active) {
      lane_batch.lane->PublishBatch(lane_batch.batch);
    }
  }
  return Status::Ok();
}

Status UrmaGraphReader::CancelPrepared(GraphTicket ticket) {
  if (ticket.value == 0 || ticket.value > prepared_operation_count_) {
    return Status(StatusCode::kProtocolError,
                  "prepared graph cancellation ticket is invalid");
  }
  auto &operation = prepared_operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) !=
          PreparedOperationState::kPrepared) {
    return Status(StatusCode::kProtocolError,
                  "prepared graph cancellation ticket is stale");
  }
  Status result;
  for (std::uint32_t lane = 0; lane < operation.lane_count; ++lane) {
    auto &lane_batch = operation.lane_batches[lane];
    if (!lane_batch.active) {
      continue;
    }
    auto cancelled = lane_batch.lane->CancelPreparedBatch(lane_batch.batch);
    if (result.ok() && !cancelled.ok()) {
      result = cancelled;
    }
  }
  ResetPreparedOperation(operation);
  return result;
}

Status UrmaGraphReader::FinishConsume(GraphTicket ticket,
                                      GraphNeighborConsumer &consumer) {
  if (ticket.value == 0 || ticket.value > prepared_operation_count_) {
    return Status(StatusCode::kProtocolError,
                  "graph consume ticket is invalid");
  }
  auto &operation = prepared_operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) !=
          PreparedOperationState::kPublished) {
    return Status(StatusCode::kProtocolError, "graph consume ticket is stale");
  }

  const auto aggregate =
      operation.aggregate_status.load(std::memory_order_acquire);
  Status result = aggregate == StatusCode::kOk
                      ? Status::Ok()
                      : Status(aggregate, "prepared graph READ batch failed");
  if (result.ok()) {
    for (std::size_t index = 0; index < operation.nodes.size(); ++index) {
      const auto &source = operation.nodes[index];
      std::span<const node_id_t> neighbors = source.local_neighbors;
      if (source.lane != kNoLane) {
        const auto &view =
            operation.lane_batches[source.lane].results[source.result_index];
        if (view.bytes !=
            static_cast<std::size_t>(meta_->M0) * sizeof(node_id_t)) {
          LOG_ERROR("UrmaGraphReader::FinishConsume view bytes {} != M0*{}",
                    view.bytes, sizeof(node_id_t));
          return Status(StatusCode::kInternalError,
                        "prepared graph READ batch view size is invalid");
        }
        neighbors = std::span<const node_id_t>(
            reinterpret_cast<const node_id_t *>(view.data), meta_->M0);
        const auto invalid =
            std::find(neighbors.begin(), neighbors.end(), INVALID_NODE_ID);
        neighbors = neighbors.first(static_cast<std::size_t>(
            std::distance(neighbors.begin(), invalid)));
      }
      result = consumer.Consume(index, neighbors);
      if (!result.ok()) {
        break;
      }
    }
  }

  ConsumeOperation(operation);
  return result;
}

Result<PullVectorDistanceOracle> PullVectorDistanceOracle::Create(
    const GlobalIndexMeta &meta, const ShardMap &shard_map,
    const LocalShard &local_shard, const PeerSegmentTable &peers,
    std::span<ReadBatchLane *const> lanes,
    std::span<const std::uint32_t> read_owner_by_rank,
    std::span<SearchReadResources *const> resources,
    DistanceImplementation implementation,
    std::size_t max_operations_per_search, std::size_t max_batch,
    bool use_dot_norm) {
  if (local_shard.range().shard_id != peers.local_shard() || lanes.empty() ||
      max_operations_per_search == 0 ||
      max_operations_per_search > std::numeric_limits<std::uint32_t>::max() ||
      read_owner_by_rank.empty() || resources.empty() ||
      max_operations_per_search >
          std::numeric_limits<std::uint32_t>::max() / resources.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "pull distance oracle configuration is invalid");
  }
  if (max_batch == 0) {
    auto resolved =
        CheckedMultiply<std::size_t>(meta.M0, kMaxGraphNodesPerOperation);
    if (!resolved.ok()) {
      return resolved.status();
    }
    max_batch = std::max<std::size_t>({1U, meta.M, resolved.value()});
  }
  try {
    PullVectorDistanceOracle oracle;
    oracle.meta_ = &meta;
    oracle.shard_map_ = &shard_map;
    oracle.local_shard_ = &local_shard;
    oracle.peers_ = &peers;
    oracle.lanes_.assign(lanes.begin(), lanes.end());
    oracle.resources_.assign(resources.begin(), resources.end());
    oracle.partitions_.resize(resources.size());
    for (std::size_t search = 0; search < resources.size(); ++search) {
      if (resources[search] == nullptr ||
          resources[search]->owner_search() != search ||
          resources[search]->slot_bytes() < meta.dim * sizeof(float))
        return Status(StatusCode::kInvalidArgument,
                      "Search READ pool shape is invalid");
      oracle.partitions_[search].free_records.reserve(
          max_operations_per_search);
    }
    for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
      if (lanes[lane] == nullptr || lanes[lane]->owner_lane() != lane ||
          lanes[lane]->config().search_workers != resources.size())
        return Status(StatusCode::kInvalidArgument,
                      "READ lane topology is invalid");
    }
    oracle.read_owner_by_rank_.assign(read_owner_by_rank.begin(),
                                      read_owner_by_rank.end());
    for (const auto owner : oracle.read_owner_by_rank_) {
      if (owner != kNoLane && owner >= oracle.lanes_.size()) {
        return Status(StatusCode::kInvalidArgument,
                      "vector READ owner map references a missing lane");
      }
    }
    oracle.metric_ = meta.metric == 0 ? Metric::kL2 : Metric::kInnerProduct;
    oracle.implementation_ = implementation;
    oracle.use_dot_norm_ = use_dot_norm && oracle.metric_ == Metric::kL2;
    oracle.max_batch_ = max_batch;
    oracle.prepared_operation_count_ = static_cast<std::uint32_t>(
        max_operations_per_search * resources.size());
    oracle.prepared_operations_ =
        std::make_unique<PreparedOperation[]>(oracle.prepared_operation_count_);
    for (std::uint32_t slot = 0; slot < oracle.prepared_operation_count_;
         ++slot) {
      auto &operation = oracle.prepared_operations_[slot];
      const auto search = slot / max_operations_per_search;
      operation.partition = &oracle.partitions_[search];
      operation.resources = resources[search];
      operation.slot = slot;
      operation.partition->free_records.push_back(slot);
      operation.requests.reserve(max_batch);
      operation.batch.ranges.reserve(max_batch);
      oracle.prepared_operations_[slot].results.reserve(max_batch);
    }
    return Result<PullVectorDistanceOracle>(std::move(oracle));
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate pull distance oracle");
  }
}

PullVectorDistanceOracle::PullVectorDistanceOracle(
    PullVectorDistanceOracle &&other) noexcept
    : meta_(other.meta_), shard_map_(other.shard_map_),
      local_shard_(other.local_shard_), peers_(other.peers_),
      lanes_(std::move(other.lanes_)),
      read_owner_by_rank_(std::move(other.read_owner_by_rank_)),
      metric_(other.metric_), implementation_(other.implementation_),
      use_dot_norm_(other.use_dot_norm_), max_batch_(other.max_batch_),
      prepared_operations_(std::move(other.prepared_operations_)),
      prepared_operation_count_(other.prepared_operation_count_),
      resources_(std::move(other.resources_)),
      partitions_(std::move(other.partitions_)) {}

Result<ReadBatchLane *>
PullVectorDistanceOracle::LaneFor(std::uint32_t peer_rank) const noexcept {
  if (peer_rank >= read_owner_by_rank_.size() ||
      read_owner_by_rank_[peer_rank] >= lanes_.size()) {
    return Status(StatusCode::kProtocolError,
                  "remote vector rank has no static READ owner");
  }
  return lanes_[read_owner_by_rank_[peer_rank]];
}

Status PullVectorDistanceOracle::CompleteBatch(void *context,
                                               ReadBatch &batch) noexcept {
  auto &operation = *static_cast<PreparedOperation *>(context);
  RecordFirstFailure(operation, batch.status);
  const auto outcome = ToOperationOutcome(batch.status);
  return operation.binding.Complete(outcome);
}

StatusCode PullVectorDistanceOracle::CheckBatchControl(
    void *context, std::chrono::steady_clock::time_point) noexcept {
  return static_cast<PreparedOperation *>(context)->binding.CancellationCode();
}

void PullVectorDistanceOracle::ResetPreparedOperation(
    PreparedOperation &operation) noexcept {
  if (operation.binding.prepared()) {
    operation.binding.Reset();
  }
  if (operation.results_owned) {
    operation.resources->Return(operation.results);
    operation.results_owned = false;
  }
  operation.batch.reclaim_required = operation.batch.reclaim_received = false;
  operation.ticket = {};
  operation.lane = nullptr;
  operation.query = {};
  operation.ids = {};
  operation.output = {};
  operation.requests.clear();
  operation.results.clear();
  operation.batch.ticket = {};
  operation.batch.requests = {};
  operation.batch.terminal_context = nullptr;
  operation.batch.terminal_callback = nullptr;
  operation.batch.control_context = nullptr;
  operation.batch.control_check = nullptr;
  operation.batch.state.store(ReadBatchState::kFree, std::memory_order_release);
  operation.aggregate_status.store(StatusCode::kOk, std::memory_order_relaxed);
  operation.local = false;
  operation.state.store(PreparedOperationState::kFree,
                        std::memory_order_release);
  operation.resources->ReturnRecord();
  operation.partition->free_records.push_back(operation.slot);
}

void PullVectorDistanceOracle::ConsumeOperation(
    PreparedOperation &operation) noexcept {
  operation.consumed = true;
  if (operation.binding.prepared())
    operation.binding.Reset();
  operation.query = {};
  operation.ids = {};
  operation.output = {};
  if (!operation.batch.reclaim_required || operation.batch.reclaim_received)
    ResetPreparedOperation(operation);
}

void PullVectorDistanceOracle::ReclaimBatch(void *context,
                                            ReadBatch &) noexcept {
  auto &operation = *static_cast<PreparedOperation *>(context);
  if (operation.consumed)
    ConsumeOperation(operation);
}

Result<PreparedDistanceOperation> PullVectorDistanceOracle::PrepareForWait(
    QueryWaitPreparation preparation, std::span<const float> query_vector,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  static_cast<void>(options);
  if (preparation.controller == nullptr ||
      preparation.ready_publisher == nullptr ||
      preparation.query.generation == 0 || preparation.wait_epoch == 0 ||
      preparation.operation_deadline ==
          std::chrono::steady_clock::time_point{} ||
      preparation.controller->generation() != preparation.query.generation ||
      query_vector.size() != meta_->dim || ids.empty() ||
      ids.size() != output.size() || ids.size() > max_batch_ ||
      owner_shard >= shard_map_->ranges().size()) {
    return Status(StatusCode::kInvalidArgument,
                  "prepared pull-distance phase preparation is invalid");
  }

  const auto owner = preparation.controller->owner_search();
  if (owner >= resources_.size())
    return Status(StatusCode::kInvalidArgument,
                  "READ Search owner is out of range");
  auto &partition = partitions_[owner];
  if (partition.free_records.empty())
    return Status(StatusCode::kResourceExhausted,
                  "prepared pull-distance Search records are full");
  const auto operation_slot = partition.free_records.back();
  partition.free_records.pop_back();
  auto *operation = &prepared_operations_[operation_slot];
  operation->resources->AcquireRecord();
  operation->consumed = false;
  operation->state.store(PreparedOperationState::kPreparing,
                         std::memory_order_relaxed);

  const auto fail = [&](Status status) -> Result<PreparedDistanceOperation> {
    if (operation->lane != nullptr &&
        operation->batch.state.load(std::memory_order_acquire) ==
            ReadBatchState::kPrepared) {
      static_cast<void>(operation->lane->CancelPreparedBatch(operation->batch));
    }
    ResetPreparedOperation(*operation);
    return status;
  };

  auto generation = partition.next_ticket++;
  if (generation == 0)
    generation = partition.next_ticket++;
  operation->ticket = {static_cast<ticket_t>(operation_slot) + 1U, generation};
  operation->query = query_vector;
  operation->ids = ids;
  operation->output = output;
  operation->requests.clear();
  operation->results.clear();
  operation->aggregate_status.store(StatusCode::kOk, std::memory_order_relaxed);
  operation->local = owner_shard == peers_->local_shard();

  if (operation->local) {
    std::size_t index = 0;
    for (; index + 4 <= ids.size(); index += 4) {
      std::array<const float *, 4> rows{};
      for (std::size_t row = 0; row < rows.size(); ++row) {
        auto location = shard_map_->Locate(ids[index + row]);
        if (!location.ok() || location.value().shard_id != owner_shard) {
          return fail(Status(StatusCode::kOutOfRange,
                             "prepared local distance ID has wrong owner"));
        }
        rows[row] =
            local_shard_->TrustedVector(location.value().local_id).data();
      }
      auto computed = ComputeDistanceBatch4(query_vector, rows, metric_,
                                            output.subspan(index, rows.size()),
                                            implementation_);
      if (!computed.ok()) {
        return fail(computed);
      }
    }
    for (; index < ids.size(); ++index) {
      auto location = shard_map_->Locate(ids[index]);
      if (!location.ok() || location.value().shard_id != owner_shard) {
        return fail(Status(StatusCode::kOutOfRange,
                           "prepared local distance ID has wrong owner"));
      }
      output[index] = ComputeDistance(
          query_vector, local_shard_->TrustedVector(location.value().local_id),
          metric_, implementation_);
    }
    operation->state.store(PreparedOperationState::kPrepared,
                           std::memory_order_release);
    return PreparedDistanceOperation{operation->ticket, std::nullopt};
  }

  std::uint32_t peer_rank = 0;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    auto location = shard_map_->Locate(ids[index]);
    if (!location.ok() || location.value().shard_id != owner_shard) {
      return fail(Status(StatusCode::kOutOfRange,
                         "prepared remote distance ID has wrong owner"));
    }
    auto remote =
        peers_->Vector(location.value().shard_id, location.value().local_id);
    if (!remote.ok()) {
      return fail(remote.status());
    }
    if (index == 0) {
      peer_rank = remote.value().peer_rank;
      auto lane = LaneFor(peer_rank);
      if (!lane.ok()) {
        return fail(lane.status());
      }
      operation->lane = lane.value();
    } else if (remote.value().peer_rank != peer_rank) {
      return fail(
          Status(StatusCode::kProtocolError,
                 "prepared pull-distance owner group spans multiple peers"));
    }
    operation->requests.push_back({remote.value().peer_rank,
                                   remote.value().segment,
                                   remote.value().offset,
                                   {}});
    operation->results.push_back({});
  }

  auto acquired = operation->resources->Acquire(operation->results,
                                                meta_->dim * sizeof(float));
  if (!acquired.ok())
    return fail(acquired);
  operation->results_owned = true;
  for (std::size_t index = 0; index < operation->requests.size(); ++index) {
    const auto &view = operation->results[index];
    operation->requests[index].destination = {
        const_cast<std::byte *>(view.data), view.bytes};
  }
  const auto lane = operation->lane->owner_lane();
  auto query_ticket = partition.next_ticket++;
  if (query_ticket == 0)
    query_ticket = partition.next_ticket++;
  auto bound = operation->binding.Prepare(
      preparation.query, *preparation.controller, operation->query_operation,
      *preparation.ready_publisher,
      {operation_slot, lane, preparation.query.generation,
       preparation.wait_epoch, query_ticket});
  if (!bound.ok()) {
    return fail(bound);
  }
  operation->batch.owner_search = owner;
  operation->batch.reclaim_context = operation;
  operation->batch.reclaim_callback = ReclaimBatch;
  operation->batch.requests = operation->requests;
  operation->batch.deadline = preparation.operation_deadline;
  operation->batch.terminal_context = operation;
  operation->batch.terminal_callback = CompleteBatch;
  operation->batch.control_context = operation;
  operation->batch.control_check = CheckBatchControl;
  auto prepared = operation->lane->PrepareBatch(operation->batch);
  if (!prepared.ok()) {
    return fail(prepared.status());
  }
  operation->state.store(PreparedOperationState::kPrepared,
                         std::memory_order_release);
  return PreparedDistanceOperation{operation->ticket, lane};
}

Status PullVectorDistanceOracle::PublishPrepared(DistanceTicket ticket) {
  if (ticket.value == 0 || ticket.value > prepared_operation_count_) {
    return Status(StatusCode::kProtocolError,
                  "prepared pull-distance ticket is invalid");
  }
  auto &operation = prepared_operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) !=
          PreparedOperationState::kPrepared) {
    return Status(StatusCode::kProtocolError,
                  "prepared pull-distance ticket is stale");
  }
  if (!operation.local) {
    auto published = operation.binding.MarkPublished();
    if (!published.ok()) {
      return published;
    }
  }
  operation.state.store(PreparedOperationState::kPublished,
                        std::memory_order_release);
  if (!operation.local) {
    operation.lane->PublishBatch(operation.batch);
  }
  return Status::Ok();
}

Status PullVectorDistanceOracle::CancelPrepared(DistanceTicket ticket) {
  if (ticket.value == 0 || ticket.value > prepared_operation_count_) {
    return Status(StatusCode::kProtocolError,
                  "prepared pull-distance cancellation ticket is invalid");
  }
  auto &operation = prepared_operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) !=
          PreparedOperationState::kPrepared) {
    return Status(StatusCode::kProtocolError,
                  "prepared pull-distance cancellation ticket is stale");
  }
  Status result;
  if (!operation.local) {
    result = operation.lane->CancelPreparedBatch(operation.batch);
  }
  ResetPreparedOperation(operation);
  return result;
}

Status PullVectorDistanceOracle::Finish(DistanceTicket ticket) {
  if (ticket.value == 0 || ticket.value > prepared_operation_count_) {
    return Status(StatusCode::kProtocolError,
                  "prepared pull-distance finish ticket is invalid");
  }
  auto &operation = prepared_operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) !=
          PreparedOperationState::kPublished) {
    return Status(StatusCode::kProtocolError,
                  "prepared pull-distance finish ticket is stale");
  }
  if (operation.local) {
    ResetPreparedOperation(operation);
    return Status::Ok();
  }

  const auto aggregate =
      operation.aggregate_status.load(std::memory_order_acquire);
  Status result = aggregate == StatusCode::kOk
                      ? Status::Ok()
                      : Status(aggregate, "prepared vector READ batch failed");
  if (result.ok()) {
    std::size_t index = 0;
    for (; index + 4 <= operation.results.size(); index += 4) {
      std::array<const float *, 4> rows{};
      for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto &view = operation.results[index + row];
        if (view.bytes != meta_->dim * sizeof(float)) {
          LOG_ERROR("PullVectorDistanceOracle::Finish view bytes {} != dim*{}",
                    view.bytes, sizeof(float));
          return Status(StatusCode::kInternalError,
                        "prepared vector READ batch view size is invalid");
        }
        rows[row] = reinterpret_cast<const float *>(view.data);
      }
      auto computed = ComputeDistanceBatch4(
          operation.query, rows, metric_,
          operation.output.subspan(index, rows.size()), implementation_);
      if (!computed.ok()) {
        result = computed;
        break;
      }
    }
    for (; result.ok() && index < operation.results.size(); ++index) {
      const auto &view = operation.results[index];
      if (view.bytes != meta_->dim * sizeof(float)) {
        LOG_ERROR(
            "PullVectorDistanceOracle::Finish tail view bytes {} != dim*{}",
            view.bytes, sizeof(float));
        return Status(StatusCode::kInternalError,
                      "prepared vector READ batch view size is invalid");
      }
      operation.output[index] = ComputeDistance(
          operation.query,
          std::span<const float>(reinterpret_cast<const float *>(view.data),
                                 meta_->dim),
          metric_, implementation_);
    }
  }
  ConsumeOperation(operation);
  return result;
}

Result<bool> PullVectorDistanceOracle::ComputeLocalTrusted(
    std::span<const float> query_vector, shard_id_t owner_shard,
    std::span<const std::uint64_t> local_offsets, std::span<float> output) {
  try {
    const QueryVectorLease lease{{},
                                 std::make_shared<const std::vector<float>>(
                                     query_vector.begin(), query_vector.end()),
                                 ScalarDot(query_vector, query_vector)};
    return ComputeLocalTrustedLease(lease, owner_shard, local_offsets, output);
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to snapshot trusted local query");
  }
}

Result<bool> PullVectorDistanceOracle::ComputeLocalTrustedLease(
    const QueryVectorLease &lease, shard_id_t owner_shard,
    std::span<const std::uint64_t> local_offsets, std::span<float> output) {
  if (owner_shard != peers_->local_shard()) {
    return false;
  }
  const auto query_vector = lease.values();
  if (query_vector.size() != meta_->dim || local_offsets.empty() ||
      local_offsets.size() != output.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "trusted pull-local distance request shape is invalid");
  }
  std::size_t index = 0;
  for (; index + 4 <= local_offsets.size(); index += 4) {
    for (std::size_t next = index + 4;
         next < std::min(index + 8, local_offsets.size()); ++next) {
      local_shard_->PrefetchVector(local_offsets[next]);
    }
    std::array<const float *, 4> rows{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
      rows[row] =
          local_shard_->TrustedVector(local_offsets[index + row]).data();
    }
    Status computed;
    if (use_dot_norm_) {
      std::array<float, 4> norms{};
      for (std::size_t row = 0; row < norms.size(); ++row) {
        norms[row] =
            local_shard_->TrustedVectorNorm(local_offsets[index + row]);
      }
      computed = ComputeL2DotNormBatch4(
          query_vector, lease.l2_norm, rows, norms,
          output.subspan(index, rows.size()), implementation_);
    } else {
      computed = ComputeDistanceBatch4(query_vector, rows, metric_,
                                       output.subspan(index, rows.size()),
                                       implementation_);
    }
    if (!computed.ok()) {
      return computed;
    }
  }
  for (; index < local_offsets.size(); ++index) {
    const auto vector = local_shard_->TrustedVector(local_offsets[index]);
    output[index] =
        use_dot_norm_
            ? std::max(0.0F, L2FromDotNorm(lease.l2_norm,
                                           local_shard_->TrustedVectorNorm(
                                               local_offsets[index]),
                                           ScalarDot(query_vector, vector)))
            : ComputeDistance(query_vector, vector, metric_, implementation_);
  }
  return true;
}

} // namespace hnsw_urma
