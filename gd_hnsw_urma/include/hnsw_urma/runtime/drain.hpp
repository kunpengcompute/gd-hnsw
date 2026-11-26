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

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

enum class DrainPhase : std::uint8_t {
  kRunning,
  kAdmissionStopped,
  kLocalQueriesComplete,
  kDraining,
  kDrained,
  kGlobalConsensus,
  kStopped,
};

struct LaneDrainSnapshot {
  std::uint64_t read_active{0};
  std::uint64_t read_backlog{0};
  std::uint64_t rpc_active{0};
  std::uint64_t request_slots_active{0};
  std::uint64_t server_result_active{0};
  std::uint64_t reply_active{0};
  std::uint64_t completion_pending{0};
  std::uint64_t quarantine_records{0};

  [[nodiscard]] bool empty() const noexcept;
};

// Control-owned lifecycle plus lane-published drain snapshots. Snapshot fields
// are sequentially consistent atomics guarded by a sequence counter, so the
// control plane observes a consistent lane state without a data-plane mutex.
class DataPlaneDrainTracker {
public:
  [[nodiscard]] static Result<std::unique_ptr<DataPlaneDrainTracker>>
  Create(std::uint32_t communication_owner_count);

  void PublishLane(std::uint32_t lane, LaneDrainSnapshot snapshot) noexcept;
  [[nodiscard]] Result<LaneDrainSnapshot>
  ReadLane(std::uint32_t lane) const noexcept;
  void PublishActiveQueries(std::uint32_t active) noexcept;

  [[nodiscard]] Status StopAdmission() noexcept;
  [[nodiscard]] Status MarkLocalQueriesComplete() noexcept;
  [[nodiscard]] Status BeginDrainAfterRankConsensus() noexcept;
  [[nodiscard]] Status TryMarkDrained() noexcept;
  [[nodiscard]] Status
  WaitUntilDrained(std::chrono::steady_clock::time_point deadline,
                   const std::function<Status()> &health_check);
  [[nodiscard]] Status MarkGlobalDrainConsensus() noexcept;
  [[nodiscard]] Status MarkStopped() noexcept;

  [[nodiscard]] bool all_lanes_empty() const noexcept;
  [[nodiscard]] DrainPhase phase() const noexcept {
    return phase_.load(std::memory_order_acquire);
  }

private:
  static constexpr std::size_t kSnapshotFields = 8;
  struct alignas(64) AtomicLane {
    std::atomic<std::uint64_t> sequence{0};
    std::array<std::atomic<std::uint64_t>, kSnapshotFields> fields{};
  };

  DataPlaneDrainTracker() = default;
  std::uint32_t communication_owner_count_{0};
  std::unique_ptr<AtomicLane[]> lanes_;
  std::atomic<std::uint32_t> active_queries_{0};
  std::atomic<DrainPhase> phase_{DrainPhase::kRunning};
};

enum class FailureSeverity : std::uint8_t {
  kQuery = 0,
  kRankFatal = 1,
};

struct LaneFailureRecord {
  FailureSeverity severity{FailureSeverity::kQuery};
  std::uint32_t lane{0};
  std::uint32_t operation_slot{0};
  StatusCode code{StatusCode::kOk};
};

[[nodiscard]] Result<LaneFailureRecord>
SelectPrimaryFailure(std::span<const LaneFailureRecord> failures) noexcept;

} // namespace hnsw_urma
