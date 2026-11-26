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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include "hnsw_urma/core/minimax_heap.hpp"
#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"
#include "hnsw_urma/core/visited_table.hpp"

namespace hnsw_urma {

enum class QueryState : std::uint8_t {
  kInit,
  kWaitEntryDistance,
  kReadyUpperGraph,
  kReadyUpperDistance,
  kWaitUpperDistance,
  kReadyLayer0Expand,
  kWaitLayer0Graph,
  kReadyLayer0Distance,
  kWaitLayer0Distance,
  kReadyCommit,
  kCompleted,
  kFailed,
};

enum class Layer0SortMode : std::uint8_t {
  kAll,
  kLocalOnly,
  kUpperOnly,
};

enum class RunState : std::uint8_t {
  kIdle,
  kQueued,
  kRunning,
  kWaiting,
  kCompleted,
  kFailed,
};

enum class OperationTerminalOutcome : std::uint8_t {
  kComplete,
  kFailed,
  kTimedOut,
  kCancelled,
};

enum class TerminalOutcome : std::uint8_t {
  kNone,
  kFailed,
  kTimedOut,
  kCancelled,
};

struct QueryOperationHandle {
  std::uint32_t operation_slot{0};
  std::uint32_t owner_lane{0};
  std::uint64_t query_generation{0};
  std::uint64_t wait_epoch{0};
  ticket_t ticket{0};
};

class QueryOperationState {
public:
  [[nodiscard]] Status Prepare(QueryOperationHandle handle) noexcept;
  [[nodiscard]] Status MarkPublished(QueryOperationHandle handle) noexcept;
  [[nodiscard]] Result<bool>
  TryMarkTerminal(QueryOperationHandle handle,
                  OperationTerminalOutcome outcome) noexcept;
  void Reset() noexcept;

  [[nodiscard]] QueryOperationHandle handle() const noexcept { return handle_; }
  [[nodiscard]] OperationTerminalOutcome outcome() const noexcept {
    return outcome_;
  }

private:
  enum class State : std::uint8_t { kFree, kPrepared, kPublished, kTerminal };
  QueryOperationHandle handle_{};
  State state_{State::kFree};
  OperationTerminalOutcome outcome_{OperationTerminalOutcome::kFailed};
};

struct LaneOperationCount {
  std::uint32_t lane{0};
  std::uint32_t operations{0};
};

struct QueryReadyHandle {
  QueryKey query{};
  std::uint64_t wait_epoch{0};
  std::uint64_t ready_timestamp_nanoseconds{0};
  std::uint64_t published_timestamp_nanoseconds{0};
  std::uint32_t owner_search{0};
};

enum class QueryWaitProfileKind : std::uint8_t {
  kGraph,
  kEntryDistance,
  kUpperDistance,
  kLayer0Distance,
  kCount,
};

struct QueryWaitProfileSample {
  QueryWaitProfileKind kind{QueryWaitProfileKind::kGraph};
  std::uint64_t wait_nanoseconds{0};
  std::uint64_t publish_nanoseconds{0};
  std::uint64_t first_completion_nanoseconds{0};
  std::uint64_t completion_spread_nanoseconds{0};
  std::uint64_t remote_operations{0};
  std::uint64_t distance_values{0};
  std::uint32_t critical_lane{0};
};

class QueryRunController;

// Implemented by the runtime scheduler. The Progress Lane that performs the
// final operation transition is also the unique producer of the selected
// ready_queue[lane][search_worker] cell.
class QueryReadyPublisher {
public:
  virtual ~QueryReadyPublisher() = default;
  [[nodiscard]] virtual Status
  PublishReady(std::uint32_t producer_lane,
               QueryReadyHandle handle) noexcept = 0;
  [[nodiscard]] virtual Status
  DeferReady(std::uint32_t producer_lane, QueryReadyHandle handle,
             QueryRunController *controller) noexcept {
    static_cast<void>(producer_lane);
    static_cast<void>(handle);
    static_cast<void>(controller);
    return Status(StatusCode::kInvalidArgument,
                  "ready publisher does not support deferred publication");
  }
  virtual void RecordWaitProfile(QueryWaitProfileSample sample) noexcept {
    static_cast<void>(sample);
  }
};

struct QueryRetryHandle {
  QueryKey query{};
  std::uint64_t retry_epoch{0};
  std::chrono::steady_clock::time_point retry_deadline{};
};
static_assert(std::is_trivially_copyable_v<QueryRetryHandle>);

// Owns the single-runner state. Results must be written before CompleteOne();
// the final acq_rel decrement publishes them to the next acquire runner.
class QueryRunController {
public:
  explicit QueryRunController(std::uint64_t generation,
                              std::uint32_t max_lanes = 1,
                              bool timing_enabled = false);

  // Control may rebind only before publishing a successful admission.
  [[nodiscard]] Status BindOwner(std::uint32_t owner) noexcept;
  [[nodiscard]] std::uint32_t owner_search() const noexcept {
    return owner_search_;
  }
  [[nodiscard]] Status Admit() noexcept;
  [[nodiscard]] bool TryAcquire() noexcept;
  [[nodiscard]] Status Requeue() noexcept;
  [[nodiscard]] Result<std::uint64_t> RequeueForRetry() noexcept;
  [[nodiscard]] Result<std::uint64_t>
  BeginWait(std::span<const LaneOperationCount> lanes,
            std::uint64_t generation) noexcept;
  [[nodiscard]] Status BeginDeferredReady() noexcept;
  void EndDeferredReady() noexcept;
  [[nodiscard]] Result<bool>
  PublishDeferredReady(std::uint64_t generation,
                       std::uint64_t wait_epoch) noexcept;
  [[nodiscard]] bool ready_publication_deferred() const noexcept {
    return deferred_ready_mode_.load(std::memory_order_acquire);
  }
  [[nodiscard]] Result<bool>
  OnOperationTerminal(QueryOperationState &operation,
                      QueryOperationHandle handle,
                      OperationTerminalOutcome outcome) noexcept;
  [[nodiscard]] Status RequestFailure(std::uint64_t generation,
                                      TerminalOutcome outcome,
                                      StatusCode code) noexcept;
  // Checks freshness; owner routing is an internal invariant asserted in Debug.
  [[nodiscard]] bool
  ValidateReadyHandle(QueryReadyHandle handle) const noexcept;
  [[nodiscard]] bool
  ValidateRetryHandle(QueryRetryHandle handle) const noexcept;

  // Poll-compatible bridge for legacy GraphReader/DistanceOracle adapters.
  // The production distributed runner invokes completion from its Progress
  // owner and never keeps a Search Worker polling a Waiting query.
  [[nodiscard]] Status Park(std::uint32_t operation_count) noexcept;
  [[nodiscard]] Result<bool> CompleteOne(std::uint64_t generation) noexcept;
  [[nodiscard]] Status MarkCompleted() noexcept;
  void MarkFailed() noexcept;

  [[nodiscard]] RunState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint32_t pending() const noexcept {
    return remaining_operations_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint32_t remaining_lanes() const noexcept {
    return remaining_lanes_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t wait_epoch() const noexcept {
    return wait_epoch_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t next_wait_epoch() const noexcept {
    return wait_epoch() + 1U;
  }
  [[nodiscard]] std::uint64_t retry_epoch() const noexcept {
    return retry_epoch_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t TakeReadyTimestampNanoseconds() noexcept {
    return ready_timestamp_nanoseconds_.exchange(0, std::memory_order_acq_rel);
  }
  [[nodiscard]] std::uint64_t ready_timestamp_nanoseconds() const noexcept {
    return ready_timestamp_nanoseconds_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t
  TakeOperationReadyTimestampNanoseconds() noexcept {
    return operation_ready_timestamp_nanoseconds_.exchange(
        0, std::memory_order_acq_rel);
  }
  [[nodiscard]] std::uint64_t
  TakeFirstOperationTimestampNanoseconds() noexcept {
    return first_operation_timestamp_nanoseconds_.exchange(
        0, std::memory_order_acq_rel);
  }
  [[nodiscard]] std::uint32_t TakeCriticalReadyLane() noexcept {
    return critical_ready_lane_.exchange(
        std::numeric_limits<std::uint32_t>::max(), std::memory_order_acq_rel);
  }
  [[nodiscard]] TerminalOutcome terminal_outcome() const noexcept {
    return static_cast<TerminalOutcome>(
        failure_intent_.load(std::memory_order_acquire) & 0xffU);
  }
  [[nodiscard]] StatusCode failure_code() const noexcept {
    return static_cast<StatusCode>(
        (failure_intent_.load(std::memory_order_acquire) >> 8U) & 0xffU);
  }
  [[nodiscard]] StatusCode
  CancellationCode(std::uint64_t generation) const noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

private:
  std::uint32_t owner_search_{0};
  bool execution_started_{false};
  struct alignas(64) LanePending {
    std::atomic<std::uint32_t> operations{0};
    std::atomic<std::uint64_t> epoch{0};
  };

  [[nodiscard]] Result<bool>
  CompleteLaneOperation(std::uint32_t lane, std::uint64_t generation,
                        std::uint64_t wait_epoch) noexcept;

  std::uint64_t generation_{0};
  std::uint32_t max_lanes_{0};
  bool timing_enabled_{false};
  std::unique_ptr<LanePending[]> lane_pending_;
  std::atomic<RunState> state_{RunState::kIdle};
  std::atomic<std::uint32_t> remaining_lanes_{0};
  std::atomic<std::uint32_t> remaining_operations_{0};
  std::atomic<std::uint64_t> wait_epoch_{0};
  std::atomic<std::uint64_t> retry_epoch_{0};
  std::atomic<std::uint64_t> ready_timestamp_nanoseconds_{0};
  std::atomic<std::uint64_t> operation_ready_timestamp_nanoseconds_{0};
  std::atomic<std::uint64_t> first_operation_timestamp_nanoseconds_{0};
  std::atomic<std::uint32_t> critical_ready_lane_{
      std::numeric_limits<std::uint32_t>::max()};
  std::atomic<bool> deferred_ready_mode_{false};
  std::atomic<bool> inline_runner_active_{false};
  std::atomic<bool> operations_converged_{false};
  std::atomic<std::uint64_t> cancel_generation_{0};
  std::atomic<std::uint64_t> failure_intent_{0};
};

// Stable binding stored by a query phase operation. Search prepares and
// publishes it; exactly one owner Progress Lane completes it. If that
// operation also completes the final participating lane, the binding publishes
// the sole ready handle without returning to a polling Search/bridge thread.
class QueryOperationBinding {
public:
  [[nodiscard]] Status Prepare(QueryKey query, QueryRunController &controller,
                               QueryOperationState &operation,
                               QueryReadyPublisher &ready_publisher,
                               QueryOperationHandle handle) noexcept;
  [[nodiscard]] Status MarkPublished() noexcept;
  [[nodiscard]] Status Complete(OperationTerminalOutcome outcome) noexcept;
  [[nodiscard]] StatusCode CancellationCode() const noexcept;
  void Reset() noexcept;

  [[nodiscard]] QueryOperationHandle handle() const noexcept { return handle_; }
  [[nodiscard]] QueryKey query() const noexcept { return query_; }
  [[nodiscard]] bool prepared() const noexcept {
    return controller_ != nullptr;
  }

private:
  QueryKey query_{};
  QueryRunController *controller_{nullptr};
  QueryOperationState *operation_{nullptr};
  QueryReadyPublisher *ready_publisher_{nullptr};
  QueryOperationHandle handle_{};
};

struct SearchParams {
  std::uint32_t topk{10};
  std::uint32_t ef_search{64};
  std::uint32_t expand_batch{1};
  std::size_t visited_capacity{0};
  std::chrono::milliseconds operation_timeout{5000};
  std::chrono::milliseconds query_timeout{30000};
  bool profile_timing{false};
  Layer0SortMode layer0_sort_mode{Layer0SortMode::kAll};
  // INVALID/-1 preserves the global index entry point. Distributed runners
  // set these to the entry selected from the source rank's local shard.
  node_id_t entry_point{INVALID_NODE_ID};
  std::int32_t entry_level{-1};
};

struct SearchProfile {
  std::uint64_t nodes_expanded{0};
  std::uint64_t edges_scanned{0};
  std::uint64_t visited_hits{0};
  std::uint64_t heap_pushes{0};
  std::uint64_t heap_rejections{0};
  std::uint64_t heap_prefilter_rejections{0};
  std::uint64_t heap_suffix_rejections{0};
  std::uint64_t heap_full_pushes{0};
  std::uint64_t upper_steps{0};
  std::uint64_t upper_same_level_moves{0};
  std::uint64_t upper_level_drops{0};
  std::uint64_t upper_empty_neighbor_level_drops{0};
  std::uint64_t upper_level_drop_runs{0};
  std::uint64_t upper_max_consecutive_level_drops{0};
  std::uint64_t upper_level_drop_runs_length_1{0};
  std::uint64_t upper_level_drop_runs_length_2{0};
  std::uint64_t upper_level_drop_runs_length_3{0};
  std::uint64_t upper_level_drop_runs_length_4_plus{0};
  std::uint64_t upper_distance_candidate_values{0};
  std::uint64_t upper_distance_visited_hits{0};
  std::uint64_t upper_distance_unique_values{0};
  std::uint64_t upper_distance_fully_visited_steps{0};
  std::uint64_t upper_remote_candidate_values{0};
  std::uint64_t upper_remote_visited_hits{0};
  std::uint64_t upper_remote_operations{0};
  std::uint64_t upper_eliminated_remote_operations{0};
  std::uint64_t local_distance_count{0};
  std::uint64_t suspend_count{0};
  std::uint64_t resume_count{0};
  std::uint64_t routing_group_owner_writes{0};
  std::uint64_t routing_local_offset_writes{0};
  std::uint64_t routing_layer0_sorted_values{0};
  std::uint64_t routing_layer0_unsorted_values{0};
};

struct SearchTimingProfile {
  // End-to-end wall-clock stages. upper_elapsed_nanoseconds includes entry
  // distance initialization and ends immediately before layer 0 starts.
  std::uint64_t upper_elapsed_nanoseconds{0};
  std::uint64_t layer0_elapsed_nanoseconds{0};
  std::uint64_t graph_wait_nanoseconds{0};
  std::uint64_t distance_wait_nanoseconds{0};
  std::uint64_t ready_queue_nanoseconds{0};
  std::uint64_t upper_search_nanoseconds{0};
  std::uint64_t graph_plan_nanoseconds{0};
  std::uint64_t graph_publish_nanoseconds{0};
  std::uint64_t graph_consume_nanoseconds{0};
  std::uint64_t neighbor_scan_visited_nanoseconds{0};
  std::uint64_t route_group_sort_nanoseconds{0};
  std::uint64_t local_distance_nanoseconds{0};
  std::uint64_t remote_prepare_publish_nanoseconds{0};
  std::uint64_t remote_finish_nanoseconds{0};
  std::uint64_t commit_heap_nanoseconds{0};
  std::uint64_t controller_other_nanoseconds{0};

  // These phases are recorded by one exclusive timer. Their sum is the
  // Search Worker wall time spent actively executing this query; asynchronous
  // waits and ready-queue residence are separate intervals above.
  [[nodiscard]] std::uint64_t LocalExecutionNanoseconds() const noexcept {
    return upper_search_nanoseconds + graph_plan_nanoseconds +
           graph_publish_nanoseconds + graph_consume_nanoseconds +
           neighbor_scan_visited_nanoseconds + route_group_sort_nanoseconds +
           local_distance_nanoseconds + remote_prepare_publish_nanoseconds +
           remote_finish_nanoseconds + commit_heap_nanoseconds +
           controller_other_nanoseconds;
  }
};

struct SearchResult {
  std::vector<HeapEntry> neighbors;
  SearchProfile profile;
  SearchTimingProfile timing;
};

enum class SearchStepResult : std::uint8_t {
  kYielded,
  kCompleted,
  kFailed,
};

} // namespace hnsw_urma
