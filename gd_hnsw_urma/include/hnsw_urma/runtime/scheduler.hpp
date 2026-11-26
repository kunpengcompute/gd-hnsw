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

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hnsw_urma/core/mpmc_ring.hpp"
#include "hnsw_urma/core/spsc_ring.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/runtime/search_event.hpp"
#include "hnsw_urma/runtime/thread_group.hpp"
#include "hnsw_urma/search/query_context.hpp"

namespace hnsw_urma {

enum class RunnableSource : std::uint8_t {
  kAdmission,
  kReady,
  kRetry,
};

// After Step() yields the owner observes a queued retry/ready handle or a
// parked wait. Another Search never acquires this query.
enum class YieldedQueryDisposition : std::uint8_t {
  kInvalid,
  kQueued,
  kHandedOff,
};

[[nodiscard]] constexpr YieldedQueryDisposition
ClassifyYieldedQueryState(RunState state) noexcept {
  switch (state) {
  case RunState::kIdle:
    return YieldedQueryDisposition::kInvalid;
  case RunState::kQueued:
    return YieldedQueryDisposition::kQueued;
  case RunState::kWaiting:
    return YieldedQueryDisposition::kHandedOff;
  case RunState::kRunning:
  case RunState::kCompleted:
  case RunState::kFailed:
    return YieldedQueryDisposition::kInvalid;
  }
  return YieldedQueryDisposition::kInvalid;
}

struct QueryRunnableHandle {
  QueryKey query{};
  RunnableSource source{RunnableSource::kAdmission};
  std::uint64_t epoch{0};
  std::uint64_t ready_published_nanoseconds{0};
  std::uint64_t ready_popped_nanoseconds{0};
};

using SearchInput = std::variant<QueryRunnableHandle, RpcReclaimHandle>;

struct QueryDoneHandle {
  QueryKey query{};
  RunState outcome{RunState::kCompleted};
};

struct SchedulerQueueMetrics {
  std::uint64_t rpc_reclaim_published{0};
  std::uint64_t rpc_reclaim_consumed{0};
  std::vector<std::uint64_t> search_event_depth;
  std::vector<std::uint64_t> search_event_high_watermark;
  std::uint64_t ready_published{0};
  std::uint64_t ready_consumed{0};
  std::uint64_t ready_queue_high_watermark{0};
  std::uint64_t retry_count{0};
  std::uint64_t stale_query_handle{0};
  std::uint64_t stale_ready_handle{0};
  std::uint64_t stale_retry_handle{0};
  std::uint64_t duplicate_handle{0};

  std::vector<std::uint64_t> ready_published_per_worker;
  std::vector<std::uint64_t> ready_consumed_per_worker;
  std::vector<std::uint64_t> ready_lane_depth_sum_per_worker;
  std::vector<std::uint64_t> ready_lane_depth_high_watermark_per_worker;
  std::vector<std::uint64_t> ready_converged_to_publish_ns_per_worker;
  std::vector<std::uint64_t> ready_publish_to_pop_ns_per_worker;
  std::vector<std::uint64_t> ready_pop_to_step_ns_per_worker;
  std::vector<std::uint64_t> ready_run_ns_per_worker;
  std::vector<std::uint64_t> ready_producer_worker_counts;
  std::vector<std::uint64_t> ready_converged_to_publish_histograms;
  std::vector<std::uint64_t> ready_publish_to_pop_histograms;
  std::vector<std::uint64_t> ready_pop_to_step_histograms;
  std::vector<std::uint64_t> ready_run_histograms;

  std::vector<std::uint64_t> wait_event_counts;
  std::vector<std::uint64_t> wait_nanoseconds;
  std::vector<std::uint64_t> wait_publish_nanoseconds;
  std::vector<std::uint64_t> wait_first_completion_nanoseconds;
  std::vector<std::uint64_t> wait_completion_spread_nanoseconds;
  std::vector<std::uint64_t> wait_remote_operations;
  std::vector<std::uint64_t> wait_distance_values;
  std::vector<std::uint64_t> wait_remote_operations_high_watermark;
  std::vector<std::uint64_t> wait_distance_values_high_watermark;
  std::vector<std::uint64_t> wait_critical_lane_counts;
  std::vector<std::uint64_t> wait_histograms;
  std::vector<std::uint64_t> wait_publish_histograms;
  std::vector<std::uint64_t> wait_first_completion_histograms;
  std::vector<std::uint64_t> wait_completion_spread_histograms;
};

// Physical scheduler topology. Each method is documented by its role in the
// name; callers must keep the declared single producer/consumer ownership.
struct LocalRetryQueue {
  std::vector<QueryRetryHandle> heap;
  std::size_t capacity{0};
};

class SchedulerQueues {
public:
  static constexpr std::uint32_t kTimingHistogramBuckets = 64;
  static constexpr std::uint32_t kWaitProfileKinds =
      static_cast<std::uint32_t>(QueryWaitProfileKind::kCount);

  SchedulerQueues(const SchedulerQueues &) = delete;
  SchedulerQueues &operator=(const SchedulerQueues &) = delete;

  [[nodiscard]] static Result<std::unique_ptr<SchedulerQueues>>
  Create(std::uint32_t search_workers, std::uint32_t ready_producer_count,
         std::uint32_t max_active_queries, std::uint32_t queue_depth,
         bool profiling_enabled = false,
         std::uint32_t rpc_slots_per_search = 0);

  [[nodiscard]] static Result<std::uint64_t>
  ProfileStorageBytes(std::uint32_t searches, std::uint32_t producers);

  // Scheduler -> Search.
  [[nodiscard]] Status PublishAdmission(std::uint32_t worker,
                                        QueryKey query) noexcept;
  // Progress lanes -> owner Search. A full event ring is a fatal lifecycle
  // error.
  [[nodiscard]] Status PublishReady(std::uint32_t lane,
                                    std::uint32_t target_worker,
                                    QueryReadyHandle handle) noexcept;
  [[nodiscard]] Status PublishRpcReclaim(std::uint32_t lane,
                                         RpcReclaimHandle handle) noexcept;
  // Called between drained benchmark rounds, with every owner stopped.
  void ResetMetrics() noexcept;
  [[nodiscard]] std::size_t event_capacity() const noexcept {
    return ready_queues_.front().capacity();
  }
  [[nodiscard]] std::size_t event_ring_count() const noexcept {
    return ready_queues_.size();
  }
  // Only owner Search consumes its MPSC ring, Admission and private retries.
  [[nodiscard]] std::optional<SearchInput>
  PopSearchInput(std::uint32_t worker,
                 std::chrono::steady_clock::time_point now) noexcept;
  void RecordReadyStep(std::uint32_t worker, QueryRunnableHandle handle,
                       std::uint64_t step_nanoseconds) noexcept;
  void RecordReadyRun(std::uint32_t worker,
                      std::uint64_t run_nanoseconds) noexcept;
  void RecordWaitProfile(QueryWaitProfileSample sample) noexcept;
  [[nodiscard]] Status ScheduleRetry(std::uint32_t worker,
                                     QueryRetryHandle handle) noexcept;
  [[nodiscard]] std::optional<std::chrono::steady_clock::time_point>
  NextRetryDeadline(std::uint32_t worker) const noexcept;

  // Search -> Scheduler.
  [[nodiscard]] Status PublishDone(std::uint32_t worker,
                                   QueryDoneHandle handle) noexcept;
  [[nodiscard]] std::optional<QueryDoneHandle> PopDone() noexcept;

  [[nodiscard]] CountingNotification &scheduler_ready() noexcept {
    return scheduler_ready_;
  }
  [[nodiscard]] SchedulerQueueMetrics metrics() const;
  [[nodiscard]] std::uint32_t max_active_queries() const noexcept {
    return max_active_queries_;
  }

private:
  friend class SchedulerReadyPublisher;

  struct AtomicMetrics {
    std::atomic<std::uint64_t> rpc_reclaim_published{0};
    std::atomic<std::uint64_t> rpc_reclaim_consumed{0};
    std::atomic<std::uint64_t> ready_published{0};
    std::atomic<std::uint64_t> ready_consumed{0};
    std::atomic<std::uint64_t> ready_depth{0};
    std::atomic<std::uint64_t> ready_queue_high_watermark{0};
    std::atomic<std::uint64_t> retry_count{0};
    std::atomic<std::uint64_t> stale_query_handle{0};
    std::atomic<std::uint64_t> stale_ready_handle{0};
    std::atomic<std::uint64_t> stale_retry_handle{0};
    std::atomic<std::uint64_t> duplicate_handle{0};
  };
  struct alignas(64) ReadyProfileShard {
    std::unique_ptr<std::atomic<std::uint64_t>[]> worker_metrics;
    std::unique_ptr<std::atomic<std::uint64_t>[]> timing_histograms;
  };

  SchedulerQueues() = default;
  void RecordReadyPublished(std::uint32_t lane, std::uint32_t target_worker,
                            QueryReadyHandle handle,
                            std::uint64_t lane_depth) noexcept;
  void RecordReadyTiming(std::uint32_t profile_shard, std::uint32_t metric,
                         std::uint32_t worker,
                         std::uint64_t nanoseconds) noexcept;
  void RecordWaitTiming(std::uint32_t metric, std::uint32_t kind,
                        std::uint64_t nanoseconds) noexcept;
  std::uint32_t search_workers_{0};
  std::uint32_t ready_producer_count_{0};
  std::uint32_t max_active_queries_{0};
  std::size_t scheduler_done_cursor_{0};
  std::vector<std::size_t> search_input_cursors_;
  std::vector<SpscRing<QueryKey>> admission_queues_;
  std::vector<MpmcRing<SearchEvent>> ready_queues_;
  std::vector<SpscRing<QueryDoneHandle>> done_queues_;
  std::vector<LocalRetryQueue> retry_queues_;
  CountingNotification scheduler_ready_;
  AtomicMetrics metrics_;
  bool profiling_enabled_{false};
  std::unique_ptr<ReadyProfileShard[]> ready_profile_shards_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> wait_metrics_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> wait_timing_histograms_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> wait_critical_lane_counts_;
};

struct QueryDeferredReadyEntry {
  QueryReadyHandle handle;
  QueryRunController *controller{nullptr};
  bool transitioned{false};
};

struct DeferredReadyWork {
  std::vector<QueryDeferredReadyEntry> entries;
  std::size_t head{0};
  std::size_t size{0};
};

// Shared scheduler facade with strictly lane-local mutable scratch. It may be
// called concurrently by different Progress Lanes, but a lane must only call
// its own producer row.
class SchedulerReadyPublisher final : public QueryReadyPublisher {
public:
  [[nodiscard]] static Result<std::unique_ptr<SchedulerReadyPublisher>>
  Create(SchedulerQueues &queues, std::uint32_t search_workers,
         std::uint32_t ready_producer_count);

  [[nodiscard]] Status PublishReady(std::uint32_t producer_lane,
                                    QueryReadyHandle handle) noexcept override;
  [[nodiscard]] Status
  DeferReady(std::uint32_t producer_lane, QueryReadyHandle handle,
             QueryRunController *controller) noexcept override;
  void RecordWaitProfile(QueryWaitProfileSample sample) noexcept override;
  [[nodiscard]] Result<std::size_t> Flush(std::uint32_t producer_lane,
                                          std::size_t budget) noexcept;

  [[nodiscard]] std::size_t pending(std::uint32_t lane) const noexcept {
    return deferred_ready_[lane].size;
  }

private:
  SchedulerReadyPublisher() = default;
  SchedulerQueues *queues_{nullptr};
  std::uint32_t search_workers_{0};
  std::uint32_t ready_producer_count_{0};
  std::vector<DeferredReadyWork> deferred_ready_;
};

struct QueryDeadlineEntry {
  QueryKey query{};
  std::chrono::steady_clock::time_point deadline{};
};

// Scheduler-local bounded min heap. It is deliberately non-thread-safe because
// Scheduler/Control is its only owner. Cancel is O(max_active_queries); the
// configured bound is intentionally small and cancellation is not a hot path.
class QueryDeadlineHeap {
public:
  [[nodiscard]] static Result<QueryDeadlineHeap> Create(std::size_t capacity);
  [[nodiscard]] Status Schedule(QueryDeadlineEntry entry);
  [[nodiscard]] bool Cancel(QueryKey query) noexcept;
  [[nodiscard]] std::optional<QueryDeadlineEntry>
  PopExpired(std::chrono::steady_clock::time_point now) noexcept;
  [[nodiscard]] std::optional<std::chrono::steady_clock::time_point>
  next_deadline() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return heap_.size(); }

private:
  std::size_t capacity_{0};
  std::vector<QueryDeadlineEntry> heap_;
};

} // namespace hnsw_urma
