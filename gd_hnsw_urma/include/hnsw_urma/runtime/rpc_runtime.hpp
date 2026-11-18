#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/spsc_ring.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"
#include "hnsw_urma/protocol/distance_batch_wire.hpp"
#include "hnsw_urma/runtime/counting_notification.hpp"
#include "hnsw_urma/runtime/rpc/mailbox.hpp"
#include "hnsw_urma/runtime/topology.hpp"

namespace hnsw_urma {

class DmaVisibilityDomain;
class LocalShard;

class ActivePeerSet {
 public:
  [[nodiscard]] static Result<ActivePeerSet> Create(std::uint32_t peer_count);
  [[nodiscard]] bool Activate(std::uint32_t peer) noexcept;
  [[nodiscard]] bool Deactivate(std::uint32_t peer) noexcept;
  [[nodiscard]] bool contains(std::uint32_t peer) const noexcept;
  [[nodiscard]] std::span<const std::uint32_t> peers() const noexcept { return peers_; }

 private:
  static constexpr std::uint32_t kInactive =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> peers_;
  std::vector<std::uint32_t> positions_;
};

enum class RpcClientTaskState : std::uint8_t {
  kFree,
  kPrepared,
  kQueued,
  kOwnerPending,
  kBatchPrepared,
  kWaitingResponse,
  kTerminal,
};

struct RpcClientTask {
  [[nodiscard]] Status Prepare(
      ticket_t immutable_ticket, QueryKey query_key,
      std::uint64_t immutable_wait_epoch,
      std::uint32_t immutable_destination_rank,
      shard_id_t immutable_owner_shard,
      std::span<const node_id_t> request_ids,
      std::span<const float> query_values,
      std::span<float> result_values,
      std::uint32_t request_flags = kBatchHasQueryVector,
      float immutable_upper_distance = 0.0F,
      std::chrono::steady_clock::time_point immutable_deadline =
          std::chrono::steady_clock::time_point::max()) noexcept;
  void Reset() noexcept;

  ticket_t ticket{0};
  QueryKey query{};
  std::uint64_t wait_epoch{0};
  std::uint64_t request_sequence{0};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
  std::uint32_t destination_rank{0};
  std::uint32_t request_slot_id{std::numeric_limits<std::uint32_t>::max()};
  shard_id_t owner_shard{0};
  std::uint32_t flags{0};
  std::span<const node_id_t> ids;
  std::span<const float> query_vector;
  std::span<float> output;
  float upper_distance{0.0F};
  StatusCode status{StatusCode::kOk};
  std::uint64_t owner_received_nanoseconds{0};
  std::uint64_t request_prepare_started_nanoseconds{0};
  std::uint64_t request_prepared_nanoseconds{0};
  std::uint64_t request_posted_nanoseconds{0};
  std::uint64_t terminal_nanoseconds{0};
  std::uint64_t response_received_nanoseconds{0};
  std::uint32_t operation_slot{std::numeric_limits<std::uint32_t>::max()};
  std::atomic<RpcClientTaskState> state{RpcClientTaskState::kFree};
};

class RpcClientTaskObserver {
 public:
  virtual ~RpcClientTaskObserver() = default;
  [[nodiscard]] virtual Result<bool> OnRpcTaskOwned(
      std::uint32_t rpc_owner, RpcClientTask& task) noexcept = 0;
  // false means the callback handed the task back; caller must not touch it.
  [[nodiscard]] virtual StatusCode TaskControlStatus(
      std::uint32_t rpc_owner, const RpcClientTask& task) const noexcept = 0;
  [[nodiscard]] virtual Status OnRpcTaskTerminal(
      std::uint32_t rpc_owner, RpcClientTask& task) noexcept = 0;
};

struct RpcSubmitHandle {
  RpcClientTask* task{nullptr};
  std::uint64_t query_generation{0};
  std::uint64_t wait_epoch{0};
  ticket_t ticket{0};
};
static_assert(sizeof(RpcSubmitHandle) == kRpcSubmitHandleBytes);

struct RpcSubmitProfile {
  std::uint64_t published{0};
  std::uint64_t consumed{0};
  std::uint64_t full_count{0};
  std::uint64_t high_watermark{0};
};

class RpcSubmitMatrix {
 public:
  // Excludes ring objects/elements, which RuntimeQueuePlan accounts separately.
  [[nodiscard]] static Result<std::uint64_t> AuxiliaryBytes(
      std::uint32_t searches, std::uint32_t owners, std::uint32_t ranks);
  [[nodiscard]] static Result<std::unique_ptr<RpcSubmitMatrix>> Create(
      std::uint32_t search_workers, std::uint32_t rpc_threads,
      std::uint32_t depth_per_cell,
      std::span<const std::uint32_t> rpc_owner_by_rank);
  [[nodiscard]] Status Publish(std::uint32_t search_worker,
                               RpcClientTask* task) noexcept;
  [[nodiscard]] Status Reserve(std::uint32_t search_worker,
                               std::uint32_t destination_rank) noexcept;
  [[nodiscard]] Status PublishReserved(std::uint32_t search_worker,
                                       RpcClientTask* task) noexcept;
  [[nodiscard]] Status CancelReservation(std::uint32_t search_worker,
                                         std::uint32_t destination_rank) noexcept;
  [[nodiscard]] Result<std::uint32_t> OwnerForRank(
      std::uint32_t destination_rank) const noexcept;
  [[nodiscard]] std::optional<RpcClientTask*> Pop(std::uint32_t rpc_owner) noexcept;
  [[nodiscard]] RpcSubmitProfile profile() const noexcept;
  [[nodiscard]] bool drained(std::uint32_t rpc_owner) const noexcept;
  void SetAdaptiveNotifications(bool enabled) noexcept {
    adaptive_notifications_enabled_ = enabled;
  }
  [[nodiscard]] CountingNotification& ready(std::uint32_t rpc_owner) noexcept {
    return owner_ready_[rpc_owner];
  }

 private:
  struct alignas(64) QueueBitmapWord { std::atomic<std::uint64_t> bits{0}; };
  struct alignas(64) QueueActivity { std::atomic<bool> nonempty{false}; };
  RpcSubmitMatrix() = default;
  [[nodiscard]] std::size_t Index(std::uint32_t worker,
                                  std::uint32_t owner) const noexcept;
  [[nodiscard]] bool MarkNonempty(std::uint32_t worker,
                                  std::uint32_t owner) noexcept;
  void ClearNonempty(std::uint32_t worker, std::uint32_t owner) noexcept;
  [[nodiscard]] std::optional<std::uint32_t> NextNonemptyWorker(
      std::uint32_t owner, std::uint32_t cursor) const noexcept;

  std::uint32_t search_workers_{0};
  std::uint32_t rpc_threads_{0};
  std::uint32_t bitmap_words_per_owner_{0};
  std::vector<std::uint32_t> rpc_owner_by_rank_;
  std::vector<std::size_t> owner_cursors_;
  std::vector<SpscRing<RpcSubmitHandle>> queues_;
  std::unique_ptr<QueueBitmapWord[]> nonempty_bitmap_;
  std::unique_ptr<QueueActivity[]> queue_activity_;
  std::unique_ptr<CountingNotification[]> owner_ready_;
  std::unique_ptr<RpcSubmitProfile[]> producer_profiles_;
  std::unique_ptr<RpcSubmitProfile[]> consumer_profiles_;
  bool adaptive_notifications_enabled_{false};
};

struct PreparedBatch {
  std::uint32_t peer_rank{0};
  std::uint32_t slot_id{0};
  std::uint64_t sequence{0};
  std::uint32_t task_count{0};
  std::uint32_t bytes{0};
};

struct RpcClientProfile {
  std::uint64_t logical_tasks{0};
  std::uint64_t request_batches{0};
  std::uint64_t request_batch_tasks{0};
  std::uint64_t request_batch_bytes{0};
  std::uint64_t response_batches{0};
  std::uint64_t response_batch_tasks{0};
  std::uint64_t response_batch_bytes{0};
  std::uint64_t terminal_tasks{0};
  std::uint64_t pending_high_watermark{0};
  std::uint64_t owner_queue_nanoseconds{0};
  std::uint64_t owner_queue_samples{0};
  std::uint64_t owner_pending_nanoseconds{0};
  std::uint64_t request_prepare_nanoseconds{0};
  std::uint64_t request_post_wait_nanoseconds{0};
  std::uint64_t round_trip_nanoseconds{0};
  std::uint64_t round_trip_samples{0};
  std::uint64_t response_notify_to_receive_nanoseconds{0};
  std::uint64_t response_notify_to_receive_samples{0};
  std::uint64_t response_notify_to_receive_task_nanoseconds{0};
  std::uint64_t response_notify_to_receive_task_samples{0};
  std::uint64_t response_received_to_task_terminal_nanoseconds{0};
  std::uint64_t response_received_to_task_terminal_samples{0};
  std::uint64_t response_decode_nanoseconds{0};
  std::uint64_t response_decode_samples{0};
};

class RpcClientOwner {
 public:
  [[nodiscard]] static Result<std::unique_ptr<RpcClientOwner>> Create(
      std::uint32_t owner_thread, std::uint32_t world_size,
      std::uint32_t pending_capacity, RpcSubmitMatrix& submissions,
      std::uint32_t query_slots, std::uint32_t request_ring_depth,
      bool profiling_enabled = false,
      RpcClientTaskObserver* task_observer = nullptr);
  [[nodiscard]] Result<std::uint32_t> DrainSubmissions(std::uint32_t budget = 64);
  [[nodiscard]] Result<std::optional<PreparedBatch>> PrepareRequestBatch(
      std::uint32_t peer_rank, std::uint32_t slot_id,
      std::uint64_t sequence, std::uint32_t max_tasks,
      std::uint32_t max_request_bytes, std::uint32_t max_response_bytes,
      std::span<std::byte> output);
  [[nodiscard]] Status CommitPreparedRequest(PreparedBatch batch) noexcept;
  [[nodiscard]] Status AbortPreparedRequest(PreparedBatch batch) noexcept;
  [[nodiscard]] Status AcceptResponse(
      std::uint32_t peer_rank, std::uint32_t slot_id,
      std::uint64_t expected_sequence,
      std::span<const std::byte> message,
      std::uint64_t notify_observed_nanoseconds = 0) noexcept;
  // Safe before provider acceptance. Cancels the complete prepared batch if
  // needed; surviving members return to pending and are encoded again.
  [[nodiscard]] Result<bool> CancelUnsent(RpcClientTask& task) noexcept;
  [[nodiscard]] Result<bool> ValidatePreparedRequest(PreparedBatch batch) noexcept;
  [[nodiscard]] static Result<std::uint64_t> RequiredBytes(
      std::uint32_t world_size, std::uint32_t pending_capacity,
      std::uint32_t query_slots, std::uint32_t request_ring_depth);
  [[nodiscard]] bool drained() const noexcept;
  [[nodiscard]] RpcClientProfile profile() const noexcept { return profile_; }
  [[nodiscard]] std::size_t pending_tasks(std::uint32_t peer_rank) const noexcept;
  [[nodiscard]] std::size_t waiting_response_tasks(
      std::uint32_t peer_rank) const noexcept;
  [[nodiscard]] std::span<const std::uint32_t> pending_peers() const noexcept {
    return pending_peers_.peers();
  }

 private:
  struct QueryPeerSendState {
    std::uint64_t local_generation{0};
    bool query_sent{false};
  };
  struct ClientInflightBatch {
    std::uint64_t sequence{0};
    std::vector<RpcClientTask*> tasks;
  };
  // RPC-private fixed ring: normal batch removal is O(batch), with no growth
  // or shifting of other peers' queued tasks. Cancellation is a rare O(M) erase.
  struct PendingTasks {
    std::vector<RpcClientTask*> entries;
    std::size_t head{0};
    std::size_t count{0};
    [[nodiscard]] std::size_t size() const noexcept { return count; }
    [[nodiscard]] bool empty() const noexcept { return count == 0; }
    RpcClientTask*& operator[](std::size_t index) noexcept {
      return entries[(head + index) % entries.size()];
    }
    void push_back(RpcClientTask* task) noexcept { (*this)[count++] = task; }
    void pop_front() noexcept {
      entries[head] = nullptr;
      head = (head + 1U) % entries.size();
      --count;
    }
    bool Remove(RpcClientTask* task) noexcept {
      for (std::size_t i = 0; i < count; ++i) {
        if ((*this)[i] != task) continue;
        for (std::size_t j = i + 1; j < count; ++j) (*this)[j - 1] = (*this)[j];
        (*this)[count - 1] = nullptr;
        --count;
        return true;
      }
      return false;
    }
  };
  struct PeerState {
    PendingTasks pending;
    std::vector<RpcClientTask*> prepared;
    std::vector<std::uint32_t> prepared_flags;
    std::vector<QueryPeerSendState> query_states;
    std::vector<ClientInflightBatch> inflight;
    std::size_t waiting_response_tasks{0};
  };
  struct ActiveEntry {
    std::uint32_t peer_rank{0};
    ticket_t ticket{0};
    RpcClientTask* task{nullptr};
  };

  RpcClientOwner() = default;
  [[nodiscard]] Result<QueryPeerSendState*> QueryState(
      std::uint32_t peer_rank, QueryKey query) noexcept;
  [[nodiscard]] RpcClientTask* FindActive(
      std::uint32_t peer_rank, ticket_t ticket) const noexcept;
  [[nodiscard]] bool InsertActive(std::uint32_t peer_rank, ticket_t ticket,
                                  RpcClientTask* task) noexcept;
  [[nodiscard]] bool EraseActive(std::uint32_t peer_rank, ticket_t ticket,
                                 const RpcClientTask* task) noexcept;

  std::uint32_t owner_thread_{0};
  std::uint32_t pending_capacity_{0};
  std::uint32_t request_ring_depth_{0};
  RpcSubmitMatrix* submissions_{nullptr};
  RpcClientTaskObserver* task_observer_{nullptr};
  std::vector<PeerState> peers_;
  ActivePeerSet pending_peers_;
  std::vector<ActiveEntry> active_index_;
  std::size_t active_count_{0};
  std::size_t pending_count_{0};
  std::vector<DistanceBatchRequestTaskInput> request_inputs_;
  std::vector<RpcClientTask*> terminal_work_;
  RpcClientProfile profile_;
  bool profiling_enabled_{false};
};

enum class ResultSlotState : std::uint8_t {
  kFree,
  kComputing,
  kReady,
  kInFlight,
};

struct ReadyResultSlot {
  RpcSlotId id{};
  std::uint64_t sequence{0};
  std::uint32_t task_count{0};
  std::uint32_t bytes{0};
  std::uint64_t request_notify_observed_nanoseconds{0};
  std::uint64_t ready_nanoseconds{0};
  std::span<const std::byte> payload;
};

struct DistanceWorkerProfile {
  std::uint64_t request_slot_scans{0};
  std::uint64_t request_slot_hits{0};
  std::uint64_t request_slot_blocked_by_result_inflight{0};
  std::uint64_t request_batches{0};
  std::uint64_t request_tasks{0};
  std::uint64_t distance_values{0};
  std::uint64_t request_poll_hit_to_claim_nanoseconds{0};
  std::uint64_t request_poll_hit_to_claim_samples{0};
  std::uint64_t request_decode_nanoseconds{0};
  std::uint64_t query_slot_store_nanoseconds{0};
  std::uint64_t query_slot_load_nanoseconds{0};
  std::uint64_t distance_compute_nanoseconds{0};
  std::uint64_t filter_and_response_encode_nanoseconds{0};
  std::uint64_t result_ready_publications{0};
  std::uint64_t busy_iterations{0};
  std::uint64_t idle_iterations{0};
};

class RpcServerRuntime {
 public:
  [[nodiscard]] static Result<std::unique_ptr<RpcServerRuntime>> Create(
      RpcMailboxMemory& memory, std::uint32_t distance_workers,
      std::uint32_t max_ids_per_task, std::uint32_t local_rank,
      std::span<const std::uint32_t> shard_to_rank,
      std::span<const std::uint32_t> rpc_owner_by_rank,
      std::uint32_t max_active_queries, std::uint32_t dim,
      bool profiling_enabled = false,
      const LocalShard* local_shard = nullptr);

  [[nodiscard]] Result<bool> PollAndProcessRequestSlot(
      std::uint32_t worker, Metric metric,
      DmaVisibilityDomain& visibility) noexcept;
  [[nodiscard]] Result<std::optional<ReadyResultSlot>> TryGetReadyResult(
      std::uint32_t rpc_owner) noexcept;
  [[nodiscard]] Status MarkResultPosted(RpcSlotId id,
                                        std::uint64_t sequence) noexcept;
  [[nodiscard]] Status AbortResultPost(RpcSlotId id,
                                       std::uint64_t sequence) noexcept;
  [[nodiscard]] Status CompleteResultSend(RpcSlotId id,
                                          std::uint64_t sequence) noexcept;
  [[nodiscard]] CountingNotification& result_ready(
      std::uint32_t rpc_owner) noexcept;
  void SetResultAdaptiveNotifications(bool enabled) noexcept {
    result_adaptive_notifications_enabled_ = enabled;
  }
  [[nodiscard]] bool drained(std::uint32_t rpc_owner) const noexcept;
  [[nodiscard]] DistanceWorkerProfile worker_profile(
      std::uint32_t worker) const noexcept;
  [[nodiscard]] std::span<const RpcSlotId> worker_slots(
      std::uint32_t worker) const noexcept;
  [[nodiscard]] ResultSlotState result_state(RpcSlotId id) const noexcept;

 private:
  struct alignas(64) ResultSlotControl {
    std::atomic<ResultSlotState> state{ResultSlotState::kFree};
    std::uint64_t request_sequence{0};
    std::uint32_t response_bytes{0};
    std::uint32_t task_count{0};
    std::uint32_t owner_worker{0};
    std::uint64_t request_notify_observed_nanoseconds{0};
    std::uint64_t ready_nanoseconds{0};
  };
  struct alignas(64) RemoteQuerySlot {
    std::atomic_flag local_access = ATOMIC_FLAG_INIT;
    float norm{0.0F};
  };
  struct alignas(64) AtomicWorkerProfile {
    std::atomic<std::uint64_t> request_slot_scans{0};
    std::atomic<std::uint64_t> request_slot_hits{0};
    std::atomic<std::uint64_t> request_slot_blocked_by_result_inflight{0};
    std::atomic<std::uint64_t> request_batches{0};
    std::atomic<std::uint64_t> request_tasks{0};
    std::atomic<std::uint64_t> distance_values{0};
    std::atomic<std::uint64_t> request_poll_hit_to_claim_nanoseconds{0};
    std::atomic<std::uint64_t> request_poll_hit_to_claim_samples{0};
    std::atomic<std::uint64_t> request_decode_nanoseconds{0};
    std::atomic<std::uint64_t> query_slot_store_nanoseconds{0};
    std::atomic<std::uint64_t> query_slot_load_nanoseconds{0};
    std::atomic<std::uint64_t> distance_compute_nanoseconds{0};
    std::atomic<std::uint64_t> filter_and_response_encode_nanoseconds{0};
    std::atomic<std::uint64_t> result_ready_publications{0};
    std::atomic<std::uint64_t> busy_iterations{0};
    std::atomic<std::uint64_t> idle_iterations{0};
  };

  RpcServerRuntime() = default;
  [[nodiscard]] std::size_t SlotIndex(RpcSlotId id) const noexcept;
  [[nodiscard]] std::size_t QueryIndex(
      std::uint32_t source_rank, std::uint32_t query_slot_id) const noexcept;
  [[nodiscard]] Status ComputeTask(
      Metric metric, std::span<const float> query, float query_norm,
      std::span<const node_id_t> ids,
      std::span<float> distances) const noexcept;

  RpcMailboxMemory* memory_{nullptr};
  std::uint32_t world_size_{0};
  std::uint32_t rpc_threads_{0};
  std::uint32_t distance_workers_{0};
  std::uint32_t max_ids_per_task_{0};
  std::uint32_t max_active_queries_{0};
  std::uint32_t dim_{0};
  std::uint32_t ring_depth_{0};
  std::uint32_t local_rank_{0};
  const LocalShard* local_shard_{nullptr};
  std::vector<std::uint32_t> shard_to_rank_;
  std::vector<std::uint32_t> rpc_owner_by_rank_;
  std::vector<std::vector<RpcSlotId>> request_slots_by_worker_;
  std::vector<std::size_t> request_scan_cursor_by_worker_;
  std::vector<std::vector<RpcSlotId>> result_slots_by_owner_;
  std::vector<std::size_t> result_scan_cursor_by_owner_;
  std::unique_ptr<ResultSlotControl[]> result_controls_;
  std::unique_ptr<RemoteQuerySlot[]> remote_query_slots_;
  std::unique_ptr<float[]> remote_query_vectors_;
  std::vector<std::vector<float>> worker_distance_scratch_;
  std::vector<std::vector<node_id_t>> worker_id_scratch_;
  std::unique_ptr<AtomicWorkerProfile[]> worker_profiles_;
  std::unique_ptr<CountingNotification[]> result_ready_;
  bool result_adaptive_notifications_enabled_{false};
  bool profiling_enabled_{false};
};

}  // namespace hnsw_urma
