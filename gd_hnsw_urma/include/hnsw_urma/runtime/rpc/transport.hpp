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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/runtime/rpc/mailbox.hpp"
#include "hnsw_urma/runtime/rpc_runtime.hpp"
#include "hnsw_urma/transport/transport.hpp"
#include "hnsw_urma/transport/urma_resources.hpp"

namespace hnsw_urma {

struct RpcTransportConfig {
  std::uint32_t owner_thread{0};
  std::uint32_t transport_lane{0};
  std::uint32_t local_rank{0};
  std::uint32_t request_batch_max_tasks{0};
  std::uint32_t request_batch_max_bytes{0};
  std::uint32_t response_progress_max_bytes{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t poll_batch{16};
  std::uint32_t phase_budget{64};
  bool profiling_enabled{false};
};

struct RpcTransportProfile {
  static constexpr std::size_t kProgressStepCount = 5;
  std::uint64_t progress_iterations{0};
  std::uint64_t progress_timing_samples{0};
  std::array<std::uint64_t, kProgressStepCount> progress_step_nanoseconds{};
  std::uint64_t request_physical_writes{0};
  std::uint64_t response_physical_writes{0};
  std::uint64_t request_completions{0};
  std::uint64_t response_completions{0};
  std::uint64_t request_write_completion_nanoseconds{0};
  std::uint64_t response_write_completion_nanoseconds{0};
  std::uint64_t response_batches_received{0};
  std::uint64_t request_post_peer_visits{0};
  std::uint64_t request_post_free_slot_waits{0};
  std::uint64_t request_post_staging_unavailable{0};
  std::uint64_t request_post_tx_unavailable{0};
  std::uint64_t request_post_provider_backpressure_batches{0};
  std::uint64_t response_scan_slots{0};
  std::uint64_t response_scan_hits{0};
  std::uint64_t result_slot_scans{0};
  std::uint64_t result_slot_hits{0};
  std::uint64_t request_notify_observed_to_response_post_nanoseconds{0};
  std::uint64_t request_notify_observed_to_response_post_samples{0};
  std::uint64_t result_ready_to_post_nanoseconds{0};
  std::uint64_t result_ready_to_post_samples{0};
  std::uint64_t result_post_retries{0};
  std::uint64_t result_post_to_completion_nanoseconds{0};
  std::uint64_t result_post_to_completion_samples{0};
  std::uint64_t request_slots_in_use{0};
  std::uint64_t request_slots_high_watermark{0};
};

struct RpcTransportPeerSnapshot {
  std::uint64_t request_posted{0};
  std::uint64_t request_completed{0};
  std::uint64_t response_posted{0};
  std::uint64_t response_completed{0};
  std::uint64_t response_received{0};
  std::uint32_t busy_request_slots{0};
  std::uint32_t waiting_slot_id{std::numeric_limits<std::uint32_t>::max()};
  std::uint64_t waiting_sequence{0};
  std::uint64_t response_notify_value{0};
  RpcTransportProfile owner_profile;
};

class RpcTransportOwner {
public:
  [[nodiscard]] static Result<std::unique_ptr<RpcTransportOwner>>
  Create(RpcTransportConfig config, std::span<const std::uint32_t> owned_peers,
         LanePostEngine &engine, DmaVisibilityDomain &visibility,
         RpcMailboxMemory &memory, RpcClientOwner &client,
         RpcServerRuntime &server);
  [[nodiscard]] Status Progress();
  [[nodiscard]] bool drained() const noexcept;
  [[nodiscard]] Status fatal_status() const { return fatal_; }
  [[nodiscard]] RpcTransportProfile profile() const noexcept {
    return profile_;
  }
  [[nodiscard]] Result<RpcTransportPeerSnapshot>
  PeerSnapshot(std::uint32_t peer_rank) const;

private:
  enum class TxKind : std::uint8_t { kRequest, kResponse };
  enum class ClientRequestSlotState : std::uint8_t {
    kFree,
    kPrepared,
    kWaitingResponse,
  };
  struct ClientRequestSlot {
    ClientRequestSlotState state{ClientRequestSlotState::kFree};
    std::uint64_t sequence{0};
  };
  struct PeerState {
    std::vector<ClientRequestSlot> request_slots;
    std::size_t request_cursor{0};
    std::uint64_t next_request_sequence{1};
    std::uint64_t request_posted{0};
    std::uint64_t request_completed{0};
    std::uint64_t response_received{0};
    std::uint64_t response_posted{0};
    std::uint64_t response_completed{0};
  };
  struct TxSlot {
    bool active{false};
    std::uint64_t generation{0};
    OperationTicket ticket{};
    TxKind kind{TxKind::kRequest};
    std::uint32_t staging_slot{std::numeric_limits<std::uint32_t>::max()};
    PreparedBatch batch{};
    RpcSlotId result_slot{};
    std::chrono::steady_clock::time_point posted_at{};
  };

  RpcTransportOwner() = default;
  [[nodiscard]] Status SetFatal(Status status);
  [[nodiscard]] Result<std::uint32_t> ReserveTx(TxKind kind);
  void ReleaseTx(std::uint32_t slot) noexcept;
  [[nodiscard]] std::optional<std::uint32_t> AcquireRequestStaging() noexcept;
  [[nodiscard]] std::optional<std::uint32_t>
  AcquireRequestSlot(std::uint32_t peer_rank) noexcept;
  [[nodiscard]] Status PollCompletions();
  [[nodiscard]] Status ScanResponses();
  [[nodiscard]] Status PostResponses();
  [[nodiscard]] Status DrainClientSubmissions();
  [[nodiscard]] Status PostRequests();

  RpcTransportConfig config_{};
  std::vector<std::uint32_t> owned_peers_;
  std::vector<PeerState> peers_;
  LanePostEngine *engine_{nullptr};
  DmaVisibilityDomain *visibility_{nullptr};
  RpcMailboxMemory *memory_{nullptr};
  RpcClientOwner *client_{nullptr};
  RpcServerRuntime *server_{nullptr};
  std::vector<TxSlot> tx_slots_;
  std::vector<bool> request_staging_busy_;
  std::vector<TransportCompletion> completion_scratch_;
  std::vector<std::uint32_t> active_peer_scratch_;
  std::size_t response_scan_cursor_{0};
  std::size_t request_post_cursor_{0};
  std::uint64_t next_transport_ticket_{1};
  std::uint64_t response_posted_{0};
  std::uint64_t response_completed_{0};
  Status fatal_;
  RpcTransportProfile profile_;
};

} // namespace hnsw_urma
