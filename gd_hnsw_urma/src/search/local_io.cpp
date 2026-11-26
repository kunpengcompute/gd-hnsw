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

#include "hnsw_urma/search/local_io.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <new>

namespace hnsw_urma {
namespace {
bool ValidPreparation(const QueryWaitPreparation &preparation) {
  return preparation.controller != nullptr &&
         preparation.ready_publisher != nullptr &&
         preparation.query.generation != 0 && preparation.wait_epoch != 0 &&
         preparation.operation_deadline !=
             std::chrono::steady_clock::time_point{} &&
         preparation.controller->generation() == preparation.query.generation;
}
} // namespace

Result<LocalGraphReader>
LocalGraphReader::Create(const ShardMap &shard_map,
                         const UpperGraphReplica &upper,
                         std::span<const LocalShard *const> shards) {
  if (shards.size() != shard_map.ranges().size()) {
    return Status(StatusCode::kInvalidArgument,
                  "local graph reader needs every shard view");
  }
  try {
    LocalGraphReader reader;
    reader.shard_map_ = &shard_map;
    reader.upper_ = &upper;
    reader.shards_.assign(shards.begin(), shards.end());
    for (std::size_t index = 0; index < reader.shards_.size(); ++index) {
      if (reader.shards_[index] == nullptr ||
          reader.shards_[index]->range().shard_id != index) {
        return Status(StatusCode::kInvalidArgument,
                      "local graph shard views are not rank ordered");
      }
    }
    return reader;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate local graph reader");
  }
}

Result<GraphTicket>
LocalGraphReader::Submit(QueryKey query, std::span<const NodeToExpand> nodes,
                         NeighborBatch *output) {
  if (output == nullptr || nodes.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "graph read requires nodes and output");
  }
  output->Clear();
  for (const auto &node : nodes) {
    Result<std::span<const node_id_t>> neighbors =
        Status(StatusCode::kInternalError, "graph source not selected");
    if (node.level > 0) {
      neighbors = upper_->Neighbors(node.global_id, node.level);
    } else {
      auto location = shard_map_->Locate(node.global_id);
      if (!location.ok()) {
        return location.status();
      }
      neighbors = shards_[location.value().shard_id]->Neighbors(node.global_id);
    }
    if (!neighbors.ok()) {
      return neighbors.status();
    }
    const auto valid_end = std::find(neighbors.value().begin(),
                                     neighbors.value().end(), INVALID_NODE_ID);
    output->neighbors.insert(output->neighbors.end(), neighbors.value().begin(),
                             valid_end);
    output->offsets.push_back(
        static_cast<std::uint32_t>(output->neighbors.size()));
  }
  return GraphTicket{
      std::atomic_ref(next_ticket_).fetch_add(1, std::memory_order_relaxed),
      query.generation};
}

Result<PreparedGraphPhase>
LocalGraphReader::PrepareForWait(QueryWaitPreparation preparation,
                                 std::span<const NodeToExpand> nodes,
                                 NeighborBatch *output) {
  if (!ValidPreparation(preparation)) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid local graph preparation");
  }
  auto submitted = Submit(preparation.query, nodes, output);
  if (!submitted.ok())
    return submitted.status();
  // The output is complete before returning. No owner lane, wait epoch or
  // ready publication is involved; cancellation owns no outstanding work.
  return PreparedGraphPhase{submitted.value(), {}, false};
}

Result<bool> LocalGraphReader::Poll(GraphTicket ticket) {
  static_cast<void>(ticket);
  return true;
}

Status LocalGraphReader::Finish(GraphTicket ticket) {
  static_cast<void>(ticket);
  return Status::Ok();
}

Result<std::span<const node_id_t>>
LocalGraphReader::ReadUpperNeighbors(node_id_t id,
                                     std::int32_t level) const noexcept {
  return upper_->Neighbors(id, level);
}

Result<LocalDistanceOracle> LocalDistanceOracle::Create(
    const ShardMap &shard_map, std::span<const LocalShard *const> shards,
    Metric metric, DistanceImplementation implementation, bool use_dot_norm) {
  if (shards.size() != shard_map.ranges().size()) {
    return Status(StatusCode::kInvalidArgument,
                  "local distance oracle needs every shard view");
  }
  try {
    LocalDistanceOracle oracle;
    oracle.shard_map_ = &shard_map;
    oracle.shards_.assign(shards.begin(), shards.end());
    oracle.metric_ = metric;
    oracle.implementation_ = implementation;
    oracle.use_dot_norm_ = use_dot_norm && metric == Metric::kL2;
    return oracle;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate local distance oracle");
  }
}

Result<DistanceTicket> LocalDistanceOracle::Submit(
    QueryKey query, std::span<const float> query_vector, shard_id_t owner_shard,
    std::span<const node_id_t> ids, std::span<float> output,
    DistanceRequestOptions options) {
  static_cast<void>(options);
  if (owner_shard >= shards_.size() || ids.empty() ||
      ids.size() != output.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid local distance request shape/owner");
  }
  std::size_t index = 0;
  const auto query_norm =
      use_dot_norm_ ? ScalarDot(query_vector, query_vector) : 0.0F;
  for (; index + 4 <= ids.size(); index += 4) {
    std::array<const float *, 4> rows{};
    std::array<float, 4> norms{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
      auto location = shard_map_->Locate(ids[index + row]);
      if (!location.ok() || location.value().shard_id != owner_shard) {
        return Status(StatusCode::kOutOfRange,
                      "distance ID does not belong to requested owner");
      }
      auto vector = shards_[owner_shard]->Vector(ids[index + row]);
      if (!vector.ok() || vector.value().size() != query_vector.size()) {
        return Status(StatusCode::kInvalidArgument,
                      "distance vector dimension mismatch");
      }
      rows[row] = vector.value().data();
      norms[row] =
          shards_[owner_shard]->TrustedVectorNorm(location.value().local_id);
    }
    auto computed =
        use_dot_norm_
            ? ComputeL2DotNormBatch4(query_vector, query_norm, rows, norms,
                                     output.subspan(index, rows.size()),
                                     implementation_)
            : ComputeDistanceBatch4(query_vector, rows, metric_,
                                    output.subspan(index, rows.size()),
                                    implementation_);
    if (!computed.ok()) {
      return computed;
    }
  }
  for (; index < ids.size(); ++index) {
    auto location = shard_map_->Locate(ids[index]);
    if (!location.ok() || location.value().shard_id != owner_shard) {
      return Status(StatusCode::kOutOfRange,
                    "distance ID does not belong to requested owner");
    }
    auto vector = shards_[owner_shard]->Vector(ids[index]);
    if (!vector.ok() || vector.value().size() != query_vector.size()) {
      return Status(StatusCode::kInvalidArgument,
                    "distance vector dimension mismatch");
    }
    output[index] =
        use_dot_norm_
            ? std::max(0.0F,
                       L2FromDotNorm(query_norm,
                                     shards_[owner_shard]->TrustedVectorNorm(
                                         location.value().local_id),
                                     ScalarDot(query_vector, vector.value())))
            : ComputeDistance(query_vector, vector.value(), metric_,
                              implementation_);
  }
  return DistanceTicket{
      std::atomic_ref(next_ticket_).fetch_add(1, std::memory_order_relaxed),
      query.generation};
}

Result<PreparedDistanceOperation> LocalDistanceOracle::PrepareForWait(
    QueryWaitPreparation preparation, std::span<const float> query_vector,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  if (!ValidPreparation(preparation)) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid local distance preparation");
  }
  auto submitted = Submit(preparation.query, query_vector, owner_shard, ids,
                          output, options);
  if (!submitted.ok())
    return submitted.status();
  return PreparedDistanceOperation{submitted.value(), std::nullopt};
}

Result<bool> LocalDistanceOracle::Poll(DistanceTicket ticket) {
  static_cast<void>(ticket);
  return true;
}

Status LocalDistanceOracle::Finish(DistanceTicket ticket) {
  static_cast<void>(ticket);
  return Status::Ok();
}

Result<bool> LocalDistanceOracle::ComputeLocalTrusted(
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

Result<bool> LocalDistanceOracle::ComputeLocalTrustedLease(
    const QueryVectorLease &lease, shard_id_t owner_shard,
    std::span<const std::uint64_t> local_offsets, std::span<float> output) {
  const auto query_vector = lease.values();
  if (owner_shard >= shards_.size() || local_offsets.empty() ||
      local_offsets.size() != output.size() ||
      query_vector.size() != shards_[owner_shard]->TrustedVector(0).size()) {
    return Status(StatusCode::kInvalidArgument,
                  "trusted local distance request shape is invalid");
  }
  std::size_t index = 0;
  for (; index + 4 <= local_offsets.size(); index += 4) {
    for (std::size_t next = index + 4;
         next < std::min(index + 8, local_offsets.size()); ++next) {
      shards_[owner_shard]->PrefetchVector(local_offsets[next]);
    }
    std::array<const float *, 4> rows{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
      rows[row] = shards_[owner_shard]
                      ->TrustedVector(local_offsets[index + row])
                      .data();
    }
    Status computed;
    if (use_dot_norm_) {
      std::array<float, 4> norms{};
      for (std::size_t row = 0; row < norms.size(); ++row) {
        norms[row] =
            shards_[owner_shard]->TrustedVectorNorm(local_offsets[index + row]);
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
    const auto vector =
        shards_[owner_shard]->TrustedVector(local_offsets[index]);
    output[index] =
        use_dot_norm_
            ? std::max(0.0F,
                       L2FromDotNorm(lease.l2_norm,
                                     shards_[owner_shard]->TrustedVectorNorm(
                                         local_offsets[index]),
                                     ScalarDot(query_vector, vector)))
            : ComputeDistance(query_vector, vector, metric_, implementation_);
  }
  return true;
}

} // namespace hnsw_urma
