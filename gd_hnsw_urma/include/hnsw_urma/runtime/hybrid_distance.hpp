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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "hnsw_urma/search/io.hpp"

namespace hnsw_urma {

enum class DistancePolicy : std::uint8_t {
  kAuto,
  kPull,
  kPushdown,
};

struct HybridDistanceProfile {
  std::uint64_t local_batches{0};
  std::uint64_t pull_batches{0};
  std::uint64_t pushdown_batches{0};
};

// Runtime policy adapter. It does not reorder IDs: every backend writes into
// the caller's original output span in request order.
class HybridDistanceOracle final : public DistanceOracle {
public:
  [[nodiscard]] static Result<std::unique_ptr<HybridDistanceOracle>>
  Create(shard_id_t local_shard, DistancePolicy policy,
         std::uint32_t pull_threshold, DistanceOracle &pull,
         DistanceOracle &pushdown);
  ~HybridDistanceOracle() override;

  Result<DistanceTicket>
  Submit(QueryKey query, std::span<const float> query_vector,
         shard_id_t owner_shard, std::span<const node_id_t> ids,
         std::span<float> output, DistanceRequestOptions options = {}) override;
  Result<DistanceTicket>
  SubmitForWait(QueryKey query, std::uint64_t wait_epoch,
                std::span<const float> query_vector, shard_id_t owner_shard,
                std::span<const node_id_t> ids, std::span<float> output,
                DistanceRequestOptions options = {}) override;
  Result<DistanceTicket>
  SubmitLeaseForWait(const QueryVectorLease &lease, std::uint64_t wait_epoch,
                     shard_id_t owner_shard, std::span<const node_id_t> ids,
                     std::span<float> output,
                     DistanceRequestOptions options = {}) override;
  Result<bool> Poll(DistanceTicket ticket) override;
  Status Finish(DistanceTicket ticket) override;
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override;
  Result<PreparedDistanceOperation>
  PrepareForWait(QueryWaitPreparation preparation,
                 std::span<const float> query_vector, shard_id_t owner_shard,
                 std::span<const node_id_t> ids, std::span<float> output,
                 DistanceRequestOptions options = {}) override;
  Result<PreparedDistanceOperation>
  PrepareLeaseForWait(QueryWaitPreparation preparation,
                      const QueryVectorLease &lease, shard_id_t owner_shard,
                      std::span<const node_id_t> ids, std::span<float> output,
                      DistanceRequestOptions options = {}) override;
  Status PublishPrepared(DistanceTicket ticket) override;
  Status CancelPrepared(DistanceTicket ticket) override;
  [[nodiscard]] bool
  CanComputeLocalTrusted(shard_id_t owner_shard) const noexcept override;
  Result<bool> ComputeLocalTrusted(std::span<const float> query_vector,
                                   shard_id_t owner_shard,
                                   std::span<const std::uint64_t> local_offsets,
                                   std::span<float> output) override;
  Result<bool>
  ComputeLocalTrustedLease(const QueryVectorLease &lease,
                           shard_id_t owner_shard,
                           std::span<const std::uint64_t> local_offsets,
                           std::span<float> output) override;
  [[nodiscard]] HybridDistanceProfile profile() const;

private:
  class Impl;
  explicit HybridDistanceOracle(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace hnsw_urma
