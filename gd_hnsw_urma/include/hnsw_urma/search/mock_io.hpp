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

#pragma once

#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

#include "hnsw_urma/search/io.hpp"

namespace hnsw_urma {

using MockGraphLookup = std::function<Result<std::vector<node_id_t>>(
    node_id_t id, std::int32_t level)>;
using MockDistanceLookup =
    std::function<Result<float>(std::span<const float> query, node_id_t id)>;

class MockGraphReader final : public GraphReader {
public:
  explicit MockGraphReader(MockGraphLookup lookup)
      : lookup_(std::move(lookup)) {}

  Result<GraphTicket> Submit(QueryKey query,
                             std::span<const NodeToExpand> nodes,
                             NeighborBatch *output) override;
  Result<bool> Poll(GraphTicket ticket) override;
  Status Finish(GraphTicket ticket) override;
  Result<std::span<const node_id_t>>
  ReadUpperNeighbors(node_id_t id, std::int32_t level) const noexcept override;
  void Complete(GraphTicket ticket);
  void CompleteAll();
  void FailNext(Status status) { next_failure_ = std::move(status); }
  void RejectNext(Status status) { next_rejection_ = std::move(status); }
  [[nodiscard]] std::vector<GraphTicket> pending_tickets() const;

private:
  struct Operation {
    GraphTicket ticket;
    std::vector<NodeToExpand> nodes;
    NeighborBatch *output{nullptr};
    bool ready{false};
    Status status;
  };
  MockGraphLookup lookup_;
  mutable std::vector<node_id_t> upper_neighbors_;
  std::unordered_map<ticket_t, Operation> operations_;
  ticket_t next_ticket_{1};
  Status next_failure_;
  Status next_rejection_;
};

class MockDistanceOracle final : public DistanceOracle {
public:
  explicit MockDistanceOracle(MockDistanceLookup lookup)
      : lookup_(std::move(lookup)) {}

  Result<DistanceTicket>
  Submit(QueryKey query, std::span<const float> query_vector,
         shard_id_t owner_shard, std::span<const node_id_t> ids,
         std::span<float> output, DistanceRequestOptions options = {}) override;
  Result<bool> Poll(DistanceTicket ticket) override;
  Status Finish(DistanceTicket ticket) override;
  void Complete(DistanceTicket ticket);
  void CompleteAll(bool reverse = false);
  void FailNext(Status status) { next_failure_ = std::move(status); }
  void RejectNext(Status status) { next_rejection_ = std::move(status); }
  [[nodiscard]] std::vector<DistanceTicket> pending_tickets() const;
  [[nodiscard]] bool saw_upper_distance() const noexcept {
    return saw_upper_distance_;
  }
  void ResetRequestObservations() noexcept { saw_upper_distance_ = false; }

private:
  struct Operation {
    DistanceTicket ticket;
    std::vector<float> query;
    std::vector<node_id_t> ids;
    std::span<float> output;
    bool ready{false};
    Status status;
  };
  MockDistanceLookup lookup_;
  std::unordered_map<ticket_t, Operation> operations_;
  ticket_t next_ticket_{1};
  Status next_failure_;
  Status next_rejection_;
  bool saw_upper_distance_{false};
};

} // namespace hnsw_urma
