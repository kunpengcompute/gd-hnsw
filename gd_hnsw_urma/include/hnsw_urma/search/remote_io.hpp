#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/core/shard_map.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/search/io.hpp"
#include "hnsw_urma/search/read_resources.hpp"
#include "hnsw_urma/transport/read_batch_lane.hpp"

namespace hnsw_urma {

struct PeerDataSegments {
  shard_id_t shard_id{0};
  std::uint32_t peer_rank{0};
  std::uint64_t graph_bytes{0};
  std::uint64_t vector_bytes{0};
};

struct RemoteReadLocation {
  std::uint32_t peer_rank{0};
  SegmentKind segment{SegmentKind::kLayer0Graph};
  segment_offset_t offset{0};
  std::size_t bytes{0};
};

class PeerSegmentTable {
 public:
  [[nodiscard]] static Result<PeerSegmentTable> Create(
      const GlobalIndexMeta& meta, const ShardMap& shard_map,
      shard_id_t local_shard, std::span<const PeerDataSegments> peers);

  [[nodiscard]] Result<RemoteReadLocation> Graph(
      node_id_t id) const noexcept;
  [[nodiscard]] Result<RemoteReadLocation> Graph(
      shard_id_t shard, std::uint64_t local_id) const noexcept;
  [[nodiscard]] Result<RemoteReadLocation> Vector(
      node_id_t id) const noexcept;
  [[nodiscard]] Result<RemoteReadLocation> Vector(
      shard_id_t shard, std::uint64_t local_id) const noexcept;
  [[nodiscard]] shard_id_t local_shard() const noexcept {
    return local_shard_;
  }

 private:
  GlobalIndexMeta meta_{};
  const ShardMap* shard_map_{nullptr};
  shard_id_t local_shard_{0};
  std::vector<PeerDataSegments> peers_;
};

class UrmaGraphReader final : public GraphReader {
 public:
  UrmaGraphReader(UrmaGraphReader&& other) noexcept;
  UrmaGraphReader& operator=(UrmaGraphReader&&) = delete;
  [[nodiscard]] static Result<UrmaGraphReader> Create(
      const GlobalIndexMeta& meta, const ShardMap& shard_map,
      const UpperGraphReplica& upper, const LocalShard& local_shard,
      const PeerSegmentTable& peers,
      std::span<ReadBatchLane* const> lanes,
      std::span<const std::uint32_t> read_owner_by_rank,
      std::span<SearchReadResources* const> resources,
      std::size_t max_operations_per_search = 1024);

  Result<std::span<const node_id_t>> ReadUpperNeighbors(
      node_id_t id, std::int32_t level) const noexcept override;
  [[nodiscard]] bool SupportsDirectConsume() const noexcept override {
    return true;
  }
  Status FinishConsume(GraphTicket ticket,
                       GraphNeighborConsumer& consumer) override;
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override {
    return true;
  }
  Result<PreparedGraphPhase> PrepareForWait(
      QueryWaitPreparation preparation,
      std::span<const NodeToExpand> nodes,
      NeighborBatch* output) override;
  Status PublishPrepared(GraphTicket ticket) override;
  Status CancelPrepared(GraphTicket ticket) override;

 private:
  UrmaGraphReader() = default;
  enum class PreparedOperationState : std::uint8_t {
    kFree,
    kPreparing,
    kPrepared,
    kPublished,
  };
  struct PreparedOperation;
  struct LaneBatch {
    bool active{false};
    bool results_owned{false};
    PreparedOperation* parent{nullptr};
    ReadBatchLane* lane{nullptr};
    ReadBatch batch;
    QueryOperationState query_operation;
    QueryOperationBinding binding;
    std::vector<ReadRequest> requests;
    std::vector<ReadResultView> results;
  };
  struct NodeSource {
    std::span<const node_id_t> local_neighbors;
    std::uint32_t lane{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t result_index{0};
  };
  struct RecordPartition {
    std::vector<std::uint32_t> free_records;
    ticket_t next_ticket{1};
  };
  struct PreparedOperation {
    RecordPartition* partition{nullptr};
    SearchReadResources* resources{nullptr};
    std::uint32_t slot{0};
    bool consumed{false};
    std::atomic<PreparedOperationState> state{
        PreparedOperationState::kFree};
    GraphTicket ticket;
    std::vector<NodeSource> nodes;
    std::unique_ptr<LaneBatch[]> lane_batches;
    std::uint32_t lane_count{0};
    std::vector<LaneOperationCount> participating_lanes;
    std::atomic<StatusCode> aggregate_status{StatusCode::kOk};
  };

  [[nodiscard]] static Status CompleteBatch(
      void* context, ReadBatch& batch) noexcept;
  [[nodiscard]] static StatusCode CheckBatchControl(
      void* context,
      std::chrono::steady_clock::time_point now) noexcept;
  [[nodiscard]] Result<ReadBatchLane*> LaneFor(
      std::uint32_t peer_rank) const noexcept;
  static void ResetPreparedOperation(PreparedOperation& operation) noexcept;
  static void ConsumeOperation(PreparedOperation& operation) noexcept;
  static void ReclaimBatch(void* context, ReadBatch& batch) noexcept;

  const GlobalIndexMeta* meta_{nullptr};
  const ShardMap* shard_map_{nullptr};
  const UpperGraphReplica* upper_{nullptr};
  const LocalShard* local_shard_{nullptr};
  const PeerSegmentTable* peers_{nullptr};
  std::vector<ReadBatchLane*> lanes_;
  std::vector<std::uint32_t> read_owner_by_rank_;
  std::unique_ptr<PreparedOperation[]> prepared_operations_;
  std::uint32_t prepared_operation_count_{0};
  std::vector<SearchReadResources*> resources_;
  std::vector<RecordPartition> partitions_;
};

class PullVectorDistanceOracle final : public DistanceOracle {
 public:
  PullVectorDistanceOracle(PullVectorDistanceOracle&& other) noexcept;
  PullVectorDistanceOracle& operator=(PullVectorDistanceOracle&&) = delete;
  [[nodiscard]] static Result<PullVectorDistanceOracle> Create(
      const GlobalIndexMeta& meta, const ShardMap& shard_map,
      const LocalShard& local_shard, const PeerSegmentTable& peers,
      std::span<ReadBatchLane* const> lanes,
      std::span<const std::uint32_t> read_owner_by_rank,
      std::span<SearchReadResources* const> resources,
      DistanceImplementation implementation = DistanceImplementation::kAuto,
      std::size_t max_operations_per_search = 1024, std::size_t max_batch = 0,
      bool use_dot_norm = false);

  Status Finish(DistanceTicket ticket) override;
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override {
    return true;
  }
  Result<PreparedDistanceOperation> PrepareForWait(
      QueryWaitPreparation preparation,
      std::span<const float> query_vector, shard_id_t owner_shard,
      std::span<const node_id_t> ids, std::span<float> output,
      DistanceRequestOptions options = {}) override;
  Status PublishPrepared(DistanceTicket ticket) override;
  Status CancelPrepared(DistanceTicket ticket) override;
  [[nodiscard]] bool CanComputeLocalTrusted(
      shard_id_t owner_shard) const noexcept override {
    return peers_ != nullptr && owner_shard == peers_->local_shard();
  }
  Result<bool> ComputeLocalTrusted(
      std::span<const float> query_vector, shard_id_t owner_shard,
      std::span<const std::uint64_t> local_offsets,
      std::span<float> output) override;
  Result<bool> ComputeLocalTrustedLease(
      const QueryVectorLease& lease, shard_id_t owner_shard,
      std::span<const std::uint64_t> local_offsets,
      std::span<float> output) override;

 private:
  PullVectorDistanceOracle() = default;
  enum class PreparedOperationState : std::uint8_t {
    kFree,
    kPreparing,
    kPrepared,
    kPublished,
  };
  struct RecordPartition {
    std::vector<std::uint32_t> free_records;
    ticket_t next_ticket{1};
  };
  struct PreparedOperation {
    RecordPartition* partition{nullptr};
    SearchReadResources* resources{nullptr};
    std::uint32_t slot{0};
    bool consumed{false};
    std::atomic<PreparedOperationState> state{
        PreparedOperationState::kFree};
    DistanceTicket ticket;
    QueryOperationState query_operation;
    QueryOperationBinding binding;
    ReadBatchLane* lane{nullptr};
    ReadBatch batch;
    std::span<const float> query;
    std::span<const node_id_t> ids;
    std::span<float> output;
    std::vector<ReadRequest> requests;
    std::vector<ReadResultView> results;
    std::atomic<StatusCode> aggregate_status{StatusCode::kOk};
    bool local{false};
    bool results_owned{false};
  };

  [[nodiscard]] static Status CompleteBatch(
      void* context, ReadBatch& batch) noexcept;
  [[nodiscard]] static StatusCode CheckBatchControl(
      void* context,
      std::chrono::steady_clock::time_point now) noexcept;
  [[nodiscard]] Result<ReadBatchLane*> LaneFor(
      std::uint32_t peer_rank) const noexcept;
  static void ResetPreparedOperation(PreparedOperation& operation) noexcept;
  static void ConsumeOperation(PreparedOperation& operation) noexcept;
  static void ReclaimBatch(void* context, ReadBatch& batch) noexcept;

  const GlobalIndexMeta* meta_{nullptr};
  const ShardMap* shard_map_{nullptr};
  const LocalShard* local_shard_{nullptr};
  const PeerSegmentTable* peers_{nullptr};
  std::vector<ReadBatchLane*> lanes_;
  std::vector<std::uint32_t> read_owner_by_rank_;
  Metric metric_{Metric::kL2};
  DistanceImplementation implementation_{DistanceImplementation::kAuto};
  bool use_dot_norm_{false};
  std::size_t max_batch_{0};
  std::unique_ptr<PreparedOperation[]> prepared_operations_;
  std::uint32_t prepared_operation_count_{0};
  std::vector<SearchReadResources*> resources_;
  std::vector<RecordPartition> partitions_;
};

}  // namespace hnsw_urma
