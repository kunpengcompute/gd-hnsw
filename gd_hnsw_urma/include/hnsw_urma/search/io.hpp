#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"
#include "hnsw_urma/search/query_context.hpp"

namespace hnsw_urma {

struct NodeToExpand {
  node_id_t global_id{INVALID_NODE_ID};
  std::int32_t level{0};
};

struct GraphTicket {
  ticket_t value{0};
  std::uint64_t generation{0};
  friend bool operator==(const GraphTicket &, const GraphTicket &) = default;
};

struct DistanceTicket {
  ticket_t value{0};
  std::uint64_t generation{0};
  friend bool operator==(const DistanceTicket &,
                         const DistanceTicket &) = default;
};

struct QueryWaitPreparation {
  QueryKey query{};
  std::uint64_t wait_epoch{0};
  std::chrono::steady_clock::time_point operation_deadline{};
  QueryRunController *controller{nullptr};
  QueryReadyPublisher *ready_publisher{nullptr};
};

struct PreparedGraphPhase {
  GraphTicket ticket{};
  std::span<const LaneOperationCount> participating_lanes;
  bool asynchronous{false};
};

struct PreparedDistanceOperation {
  DistanceTicket ticket{};
  std::optional<std::uint32_t> owner_lane;
};

// One immutable query-vector snapshot shared by every asynchronous operation
// belonging to a query. Shared ownership deliberately outlives a retired
// QuerySlot when an RPC operation is quarantined or completes late.
struct QueryVectorLease {
  QueryVectorLease() = default;
  QueryVectorLease(QueryKey query_key,
                   std::shared_ptr<const std::vector<float>> vector_storage,
                   float norm = 0.0F)
      : query(query_key), owner(vector_storage),
        vector(vector_storage == nullptr
                   ? std::span<const float>{}
                   : std::span<const float>(*vector_storage)),
        l2_norm(norm) {}
  QueryVectorLease(QueryKey query_key, std::shared_ptr<const void> vector_owner,
                   std::span<const float> vector_values, float norm = 0.0F)
      : query(query_key), owner(std::move(vector_owner)), vector(vector_values),
        l2_norm(norm) {}

  QueryKey query{};
  std::shared_ptr<const void> owner;
  std::span<const float> vector;
  float l2_norm{0.0F};

  [[nodiscard]] std::span<const float> values() const noexcept {
    return vector;
  }
  [[nodiscard]] bool valid() const noexcept {
    return query.generation != 0 && owner != nullptr && !vector.empty();
  }
};

struct NeighborBatch {
  std::vector<node_id_t> neighbors;
  std::vector<std::uint32_t> offsets;

  void Clear() {
    neighbors.clear();
    offsets.clear();
    offsets.push_back(0);
  }
};

class GraphNeighborConsumer {
public:
  virtual ~GraphNeighborConsumer() = default;
  virtual Status Consume(std::size_t node_index,
                         std::span<const node_id_t> neighbors) = 0;
};

struct DistanceRequestOptions {
  std::optional<float> upper_distance;
};

class GraphReader {
public:
  virtual ~GraphReader() = default;
  virtual Result<GraphTicket> Submit(QueryKey, std::span<const NodeToExpand>,
                                     NeighborBatch *) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support polling submission");
  }
  virtual Result<bool> Poll(GraphTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support polling");
  }
  virtual Status Finish(GraphTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support polling finish");
  }
  virtual Result<std::span<const node_id_t>>
  ReadUpperNeighbors(node_id_t, std::int32_t) const noexcept {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not expose a local upper replica");
  }
  [[nodiscard]] virtual bool SupportsDirectConsume() const noexcept {
    return false;
  }
  virtual Status FinishConsume(GraphTicket, GraphNeighborConsumer &) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support direct consumption");
  }
  [[nodiscard]] virtual bool SupportsPreparedWait() const noexcept {
    return false;
  }
  virtual Result<PreparedGraphPhase>
  PrepareForWait(QueryWaitPreparation, std::span<const NodeToExpand>,
                 NeighborBatch *) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support prepared waits");
  }
  virtual Status PublishPrepared(GraphTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support prepared publication");
  }
  virtual Status CancelPrepared(GraphTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "graph reader does not support prepared cancellation");
  }
};

class DistanceOracle {
public:
  virtual ~DistanceOracle() = default;
  virtual Result<DistanceTicket> Submit(QueryKey, std::span<const float>,
                                        shard_id_t, std::span<const node_id_t>,
                                        std::span<float>,
                                        DistanceRequestOptions = {}) {
    return Status(StatusCode::kInvalidArgument,
                  "distance oracle does not support polling submission");
  }
  // Schedulers pass the immutable wait epoch to asynchronous backends.
  // Synchronous and legacy adapters may keep using Submit().
  virtual Result<DistanceTicket>
  SubmitForWait(QueryKey query, std::uint64_t wait_epoch,
                std::span<const float> query_vector, shard_id_t owner_shard,
                std::span<const node_id_t> ids, std::span<float> output,
                DistanceRequestOptions options = {}) {
    static_cast<void>(wait_epoch);
    return Submit(query, query_vector, owner_shard, ids, output, options);
  }
  virtual Result<DistanceTicket>
  SubmitLeaseForWait(const QueryVectorLease &lease, std::uint64_t wait_epoch,
                     shard_id_t owner_shard, std::span<const node_id_t> ids,
                     std::span<float> output,
                     DistanceRequestOptions options = {}) {
    return SubmitForWait(lease.query, wait_epoch, lease.values(), owner_shard,
                         ids, output, options);
  }
  virtual Result<bool> Poll(DistanceTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "distance oracle does not support polling");
  }
  virtual Status Finish(DistanceTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "distance oracle does not support polling finish");
  }
  [[nodiscard]] virtual bool SupportsPreparedWait() const noexcept {
    return false;
  }
  virtual Result<PreparedDistanceOperation>
  PrepareForWait(QueryWaitPreparation, std::span<const float>, shard_id_t,
                 std::span<const node_id_t>, std::span<float>,
                 DistanceRequestOptions = {}) {
    return Status(StatusCode::kInvalidArgument,
                  "distance oracle does not support prepared waits");
  }
  virtual Result<PreparedDistanceOperation>
  PrepareLeaseForWait(QueryWaitPreparation preparation,
                      const QueryVectorLease &lease, shard_id_t owner_shard,
                      std::span<const node_id_t> ids, std::span<float> output,
                      DistanceRequestOptions options = {}) {
    return PrepareForWait(preparation, lease.values(), owner_shard, ids, output,
                          options);
  }
  virtual Status PublishPrepared(DistanceTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "distance oracle does not support prepared publication");
  }
  virtual Status CancelPrepared(DistanceTicket) {
    return Status(StatusCode::kInvalidArgument,
                  "distance oracle does not support prepared cancellation");
  }
  [[nodiscard]] virtual bool CanComputeLocalTrusted(shard_id_t) const noexcept {
    return false;
  }
  // Search-internal synchronous fast path. local_offsets have already passed
  // owner/range validation in RouteIds. Returns true when output was computed.
  virtual Result<bool> ComputeLocalTrusted(std::span<const float>, shard_id_t,
                                           std::span<const std::uint64_t>,
                                           std::span<float>) {
    return false;
  }
  virtual Result<bool> ComputeLocalTrustedLease(
      const QueryVectorLease &lease, shard_id_t owner_shard,
      std::span<const std::uint64_t> local_offsets, std::span<float> output) {
    return ComputeLocalTrusted(lease.values(), owner_shard, local_offsets,
                               output);
  }
};

} // namespace hnsw_urma
