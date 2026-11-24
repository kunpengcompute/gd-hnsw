#pragma once

#include "hnsw_urma/runtime/rpc_runtime.hpp"
#include "hnsw_urma/runtime/search_event.hpp"
#include "hnsw_urma/search/io.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace hnsw_urma {
class SchedulerQueues;
void SetCurrentDistanceSearchWorker(std::uint32_t worker) noexcept;
[[nodiscard]] std::uint32_t CurrentDistanceSearchWorker() noexcept;

struct RpcDistanceProfile {
  std::vector<std::uint64_t> slots_in_use_per_search;
  std::vector<std::uint64_t> slots_high_watermark_per_search;
  std::vector<std::uint64_t> pool_backpressure_per_search;
  std::vector<std::uint64_t> submit_backpressure_per_search;
  std::vector<std::uint64_t> audit_operations_per_owner;
  std::uint64_t submitted{0};
  std::uint64_t slots_in_use{0};
  std::uint64_t slots_high_watermark{0};
  std::uint64_t pool_backpressure{0};
  std::uint64_t submit_backpressure{0};
  std::uint64_t ticket_block_refills{0};
  std::uint64_t exceptional_slots_in_use{0};
  std::uint64_t exceptional_slots_high_watermark{0};
  std::uint64_t exceptional_retained_nanoseconds{0};
  std::uint64_t exceptional_reclaimed{0};
  std::uint64_t completed{0};
  std::uint64_t timed_out{0};
  std::uint64_t cancelled{0};
  std::uint64_t failed{0};
  std::uint64_t detached{0};
  std::uint64_t active_audit_passes{0};
  std::uint64_t active_audit_operations{0};
  std::uint64_t response_to_stage_terminal_nanoseconds{0};
  std::uint64_t response_to_stage_terminal_samples{0};
};

// Search owns allocation, consumption and final reclamation. RPC borrows stable
// wire/ID/result storage until normal completion or an exceptional Reclaim.
class RpcDistanceOracle final : public DistanceOracle,
                                public RpcClientTaskObserver {
public:
  [[nodiscard]] static Result<std::unique_ptr<RpcDistanceOracle>>
  Create(std::uint32_t search_workers, std::uint32_t rpc_threads,
         std::uint32_t submit_depth_per_cell, std::uint32_t slots_per_search,
         std::uint32_t max_ids_per_task, std::uint32_t dim,
         std::uint32_t local_rank, std::span<const std::uint32_t> shard_to_rank,
         std::span<const std::uint32_t> rpc_owner_by_rank,
         std::uint32_t ready_lane_base, SchedulerQueues &events,
         bool use_dot_norm = false);
  [[nodiscard]] static Result<std::uint64_t>
  RequiredBytes(std::uint32_t searches, std::uint32_t owners,
                std::uint32_t slots_per_search, std::uint32_t max_ids,
                std::uint32_t queries, std::uint32_t shards,
                std::uint32_t ranks);

  Status Finish(DistanceTicket ticket) override;
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override {
    return true;
  }
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
  [[nodiscard]] Status HandleReclaim(std::uint32_t search,
                                     RpcReclaimHandle handle);
  [[nodiscard]] Result<std::uint32_t>
  AuditOwnerControl(std::uint32_t owner, RpcClientOwner &client,
                    std::uint32_t budget = 64,
                    std::chrono::steady_clock::time_point now =
                        std::chrono::steady_clock::now());
  [[nodiscard]] RpcSubmitMatrix &submissions() noexcept {
    return *submissions_;
  }
  [[nodiscard]] const RpcSubmitMatrix &submissions() const noexcept {
    return *submissions_;
  }
  [[nodiscard]] bool drained() const noexcept;
  [[nodiscard]] bool SearchDrained(std::uint32_t search) const noexcept;
  [[nodiscard]] bool QueryWireQuiesced(QueryKey query) const noexcept;
  [[nodiscard]] RpcDistanceProfile profile() const;
  // Owners stopped and drained: start a new benchmark sampling interval.
  [[nodiscard]] Status ResetProfile() noexcept;

private:
  enum class OperationState : std::uint8_t {
    kFree,
    kPrepared,
    kActive,
    kTerminal
  };
  struct OperationEvent {
    std::uint32_t slot;
    ticket_t ticket;
  };
  struct alignas(64) Operation {
    std::atomic<OperationState> state{OperationState::kFree};
    DistanceTicket ticket{};
    RpcClientTask task;
    QueryOperationState query_operation;
    QueryOperationBinding binding;
    QueryVectorLease query_lease;
    std::uint32_t search_worker{0};
    std::uint32_t rpc_owner{0};
    std::uint32_t destination_rank{0};
    std::chrono::steady_clock::time_point deadline{};
    std::span<float> user_output;
    // Frozen by RPC before the logical handoff. Search never inspects wire
    // fields while an exceptional task can still receive a response.
    StatusCode status{StatusCode::kOk};
    RpcClientTaskState terminal_wire_state{RpcClientTaskState::kFree};
    std::uint64_t terminal_sequence{0};
    std::uint64_t terminal_nanoseconds{0};
    bool reclaim_required{false};
    // Search-only flags.
    bool consumed{false};
    bool reclaim_received{false};
    bool exception_observed{false};
    bool wire_lease_held{false};
    // RPC-only membership, removed before any logical handoff.
    std::uint32_t audit_position{UINT32_MAX};
  };
  struct alignas(64) SearchPool {
    std::vector<std::uint32_t> free_slots;
    ticket_t next_ticket{0};
    ticket_t ticket_end{0};
    std::atomic<std::uint64_t> submitted{0};
    std::atomic<std::uint64_t> slots_in_use{0};
    std::atomic<std::uint64_t> slots_high_watermark{0};
    std::atomic<std::uint64_t> pool_backpressure{0};
    std::atomic<std::uint64_t> submit_backpressure{0};
    std::atomic<std::uint64_t> ticket_block_refills{0};
    std::atomic<std::uint64_t> exceptional_slots_in_use{0};
    std::atomic<std::uint64_t> exceptional_slots_high_watermark{0};
    std::atomic<std::uint64_t> exceptional_retained_nanoseconds{0};
    std::atomic<std::uint64_t> exceptional_reclaimed{0};
  };
  struct alignas(64) OwnerState {
    std::vector<OperationEvent> active;
    std::size_t cursor{0};
    std::chrono::steady_clock::time_point next_audit{};
    std::atomic<std::uint64_t> completed{0};
    std::atomic<std::uint64_t> timed_out{0};
    std::atomic<std::uint64_t> cancelled{0};
    std::atomic<std::uint64_t> failed{0};
    std::atomic<std::uint64_t> detached{0};
    std::atomic<std::uint64_t> active_audit_passes{0};
    std::atomic<std::uint64_t> active_audit_operations{0};
    std::atomic<std::uint64_t> response_to_stage_terminal_nanoseconds{0};
    std::atomic<std::uint64_t> response_to_stage_terminal_samples{0};
  };
  struct alignas(64) QueryWireLeaseSlot {
    std::atomic<std::uint64_t> generation{0};
    std::atomic<std::uint32_t> refs{0};
  };
  RpcDistanceOracle() = default;
  Result<bool> OnRpcTaskOwned(std::uint32_t owner,
                              RpcClientTask &task) noexcept override;
  Status OnRpcTaskTerminal(std::uint32_t owner,
                           RpcClientTask &task) noexcept override;
  StatusCode
  TaskControlStatus(std::uint32_t owner,
                    const RpcClientTask &task) const noexcept override;
  Result<Operation *>
  PrepareOperation(const QueryVectorLease &lease, std::uint64_t wait_epoch,
                   shard_id_t shard, std::span<const node_id_t> ids,
                   std::span<float> output,
                   std::chrono::steady_clock::time_point deadline,
                   DistanceRequestOptions options);
  Result<ticket_t> NextTicket(SearchPool &pool) noexcept;
  Result<Operation *> Find(DistanceTicket ticket) noexcept;
  Status CompleteOperation(Operation &operation, StatusCode status,
                           bool reclaim_required) noexcept;
  void RemoveOwnerActive(Operation &operation) noexcept;
  StatusCode
  ControlStatus(const Operation &operation,
                std::chrono::steady_clock::time_point now) const noexcept;
  void ObserveException(Operation &operation) noexcept;
  void Reset(Operation &operation) noexcept;
  Status AcquireQueryWireLease(QueryKey query) noexcept;
  void ReleaseQueryWireLease(QueryKey query) noexcept;

  std::uint32_t search_workers_{0}, rpc_threads_{0}, slots_per_search_{0};
  std::uint32_t operation_count_{0}, max_ids_per_task_{0}, dim_{0},
      local_rank_{0};
  std::uint32_t ready_lane_base_{0}, max_active_queries_{0};
  bool use_dot_norm_{false};
  SchedulerQueues *events_{nullptr};
  std::vector<std::uint32_t> shard_to_rank_, rpc_owner_by_rank_;
  std::unique_ptr<Operation[]> operations_;
  std::unique_ptr<node_id_t[]> id_storage_;
  std::unique_ptr<float[]> result_storage_;
  std::unique_ptr<RpcSubmitMatrix> submissions_;
  std::unique_ptr<OwnerState[]> owner_states_;
  std::unique_ptr<SearchPool[]> search_pools_;
  std::unique_ptr<QueryWireLeaseSlot[]> query_wire_leases_;
  std::atomic<ticket_t> next_ticket_{1};
};
} // namespace hnsw_urma
