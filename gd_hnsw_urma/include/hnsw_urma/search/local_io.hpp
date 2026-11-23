#pragma once

#include <span>
#include <vector>

#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/core/shard_map.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/search/io.hpp"

namespace hnsw_urma {

class LocalGraphReader final : public GraphReader {
public:
  [[nodiscard]] static Result<LocalGraphReader>
  Create(const ShardMap &shard_map, const UpperGraphReplica &upper,
         std::span<const LocalShard *const> shards);

  Result<GraphTicket> Submit(QueryKey query,
                             std::span<const NodeToExpand> nodes,
                             NeighborBatch *output) override;
  Result<bool> Poll(GraphTicket ticket) override;
  Status Finish(GraphTicket ticket) override;
  Result<std::span<const node_id_t>>
  ReadUpperNeighbors(node_id_t id, std::int32_t level) const noexcept override;
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override {
    return true;
  }
  Result<PreparedGraphPhase> PrepareForWait(QueryWaitPreparation preparation,
                                            std::span<const NodeToExpand> nodes,
                                            NeighborBatch *output) override;
  Status PublishPrepared(GraphTicket) override { return Status::Ok(); }
  Status CancelPrepared(GraphTicket) override { return Status::Ok(); }

private:
  const ShardMap *shard_map_{nullptr};
  const UpperGraphReplica *upper_{nullptr};
  std::vector<const LocalShard *> shards_;
  ticket_t next_ticket_{1};
};

class LocalDistanceOracle final : public DistanceOracle {
public:
  [[nodiscard]] static Result<LocalDistanceOracle>
  Create(const ShardMap &shard_map, std::span<const LocalShard *const> shards,
         Metric metric,
         DistanceImplementation implementation = DistanceImplementation::kAuto,
         bool use_dot_norm = false);

  Result<DistanceTicket>
  Submit(QueryKey query, std::span<const float> query_vector,
         shard_id_t owner_shard, std::span<const node_id_t> ids,
         std::span<float> output, DistanceRequestOptions options = {}) override;
  Result<bool> Poll(DistanceTicket ticket) override;
  Status Finish(DistanceTicket ticket) override;
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override {
    return true;
  }
  Result<PreparedDistanceOperation>
  PrepareForWait(QueryWaitPreparation preparation,
                 std::span<const float> query_vector, shard_id_t owner_shard,
                 std::span<const node_id_t> ids, std::span<float> output,
                 DistanceRequestOptions options) override;
  Status PublishPrepared(DistanceTicket) override { return Status::Ok(); }
  Status CancelPrepared(DistanceTicket) override { return Status::Ok(); }
  [[nodiscard]] bool
  CanComputeLocalTrusted(shard_id_t owner_shard) const noexcept override {
    return owner_shard < shards_.size();
  }
  Result<bool> ComputeLocalTrusted(std::span<const float> query_vector,
                                   shard_id_t owner_shard,
                                   std::span<const std::uint64_t> local_offsets,
                                   std::span<float> output) override;
  Result<bool>
  ComputeLocalTrustedLease(const QueryVectorLease &lease,
                           shard_id_t owner_shard,
                           std::span<const std::uint64_t> local_offsets,
                           std::span<float> output) override;

private:
  const ShardMap *shard_map_{nullptr};
  std::vector<const LocalShard *> shards_;
  Metric metric_{Metric::kL2};
  DistanceImplementation implementation_{DistanceImplementation::kAuto};
  bool use_dot_norm_{false};
  ticket_t next_ticket_{1};
};

} // namespace hnsw_urma
