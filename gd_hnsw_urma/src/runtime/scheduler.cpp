#include "hnsw_urma/runtime/scheduler.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <utility>

namespace hnsw_urma {
namespace {

enum class ReadyWorkerMetric : std::uint32_t {
  kPublished,
  kConsumed,
  kLaneDepthSum,
  kLaneDepthHighWatermark,
  kConvergedToPublishNanoseconds,
  kPublishToPopNanoseconds,
  kPopToStepNanoseconds,
  kRunNanoseconds,
  kCount,
};

enum class ReadyTimingMetric : std::uint32_t {
  kConvergedToPublish,
  kPublishToPop,
  kPopToStep,
  kRun,
  kCount,
};

enum class WaitMetric : std::uint32_t {
  kCount,
  kNanoseconds,
  kPublishNanoseconds,
  kFirstCompletionNanoseconds,
  kCompletionSpreadNanoseconds,
  kRemoteOperations,
  kDistanceValues,
  kRemoteOperationsHighWatermark,
  kDistanceValuesHighWatermark,
  kMetricCount,
};

enum class WaitTimingMetric : std::uint32_t {
  kWait,
  kPublish,
  kFirstCompletion,
  kCompletionSpread,
  kCount,
};

constexpr std::uint32_t Count(ReadyWorkerMetric) noexcept {
  return static_cast<std::uint32_t>(ReadyWorkerMetric::kCount);
}

constexpr std::uint32_t Count(ReadyTimingMetric) noexcept {
  return static_cast<std::uint32_t>(ReadyTimingMetric::kCount);
}

constexpr std::uint32_t Count(WaitMetric) noexcept {
  return static_cast<std::uint32_t>(WaitMetric::kMetricCount);
}

constexpr std::uint32_t Count(WaitTimingMetric) noexcept {
  return static_cast<std::uint32_t>(WaitTimingMetric::kCount);
}

std::uint64_t SteadyNowNanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool RetryLater(const QueryRetryHandle &lhs,
                const QueryRetryHandle &rhs) noexcept {
  return lhs.retry_deadline > rhs.retry_deadline;
}

bool DeadlineLater(const QueryDeadlineEntry &lhs,
                   const QueryDeadlineEntry &rhs) noexcept {
  return lhs.deadline > rhs.deadline;
}

void UpdateMaximum(std::atomic<std::uint64_t> &target,
                   std::uint64_t value) noexcept {
  auto observed = target.load(std::memory_order_relaxed);
  while (observed < value) {
    if (target.compare_exchange_weak(observed, value, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
      break;
    }
    // observed is refreshed by the failed compare_exchange_weak above.
  }
}

} // namespace

Result<std::uint64_t>
SchedulerQueues::ProfileStorageBytes(std::uint32_t searches,
                                     std::uint32_t producers) {
  auto shard_count = static_cast<std::uint64_t>(searches) + producers;
  auto per_shard = CheckedMultiply<std::uint64_t>(
      searches, (Count(ReadyWorkerMetric{}) +
                 Count(ReadyTimingMetric{}) * kTimingHistogramBuckets) *
                    sizeof(std::atomic<std::uint64_t>));
  if (!per_shard.ok())
    return per_shard.status();
  auto shard_bytes = CheckedMultiply(
      shard_count, per_shard.value() + sizeof(ReadyProfileShard));
  if (!shard_bytes.ok())
    return shard_bytes.status();
  const auto wait_bytes =
      (static_cast<std::uint64_t>(Count(WaitMetric{})) +
       Count(WaitTimingMetric{}) * kTimingHistogramBuckets + producers) *
      kWaitProfileKinds * sizeof(std::atomic<std::uint64_t>);
  return CheckedAdd(shard_bytes.value(), wait_bytes);
}

Result<std::unique_ptr<SchedulerQueues>> SchedulerQueues::Create(
    std::uint32_t search_workers, std::uint32_t ready_producer_count,
    std::uint32_t max_active_queries, std::uint32_t queue_depth,
    bool profiling_enabled, std::uint32_t rpc_slots_per_search) {
  if (search_workers == 0 || ready_producer_count == 0 ||
      max_active_queries == 0 || queue_depth == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "scheduler queue dimensions must be non-zero");
  }
  auto profile_bytes =
      ProfileStorageBytes(search_workers, ready_producer_count);
  if (profiling_enabled &&
      (!profile_bytes.ok() || profile_bytes.value() > SIZE_MAX))
    return Status(StatusCode::kOutOfRange, "Ready profile storage overflows");
  auto capacity = CheckedAdd(max_active_queries, rpc_slots_per_search);
  if (!capacity.ok())
    return capacity.status();
  try {
    auto queues = std::unique_ptr<SchedulerQueues>(new SchedulerQueues());
    queues->search_workers_ = search_workers;
    queues->ready_producer_count_ = ready_producer_count;
    queues->max_active_queries_ = max_active_queries;
    queues->profiling_enabled_ = profiling_enabled;
    queues->search_input_cursors_.resize(search_workers);
    queues->admission_queues_.reserve(search_workers);
    queues->ready_queues_.reserve(search_workers);
    queues->done_queues_.reserve(search_workers);
    queues->retry_queues_.resize(search_workers);
    for (std::uint32_t worker = 0; worker < search_workers; ++worker) {
      auto admission = SpscRing<QueryKey>::Create(queue_depth);
      auto done = SpscRing<QueryDoneHandle>::Create(queue_depth);
      if (!admission.ok() || !done.ok()) {
        return admission.ok() ? done.status() : admission.status();
      }
      queues->admission_queues_.push_back(std::move(admission).value());
      queues->done_queues_.push_back(std::move(done).value());
      queues->retry_queues_[worker].capacity = max_active_queries;
      queues->retry_queues_[worker].heap.reserve(max_active_queries);
    }
    for (std::uint32_t worker = 0; worker < search_workers; ++worker) {
      auto ready =
          MpmcRing<SearchEvent>::Create(std::max(2U, capacity.value()));
      if (!ready.ok())
        return ready.status();
      queues->ready_queues_.push_back(std::move(ready).value());
    }
    if (profiling_enabled) {
      const auto ready_worker_values =
          static_cast<std::size_t>(Count(ReadyWorkerMetric{})) * search_workers;
      const auto ready_timing_values =
          static_cast<std::size_t>(Count(ReadyTimingMetric{})) *
          search_workers * kTimingHistogramBuckets;
      const auto ready_profile_shards =
          static_cast<std::size_t>(ready_producer_count) + search_workers;
      const auto wait_values =
          static_cast<std::size_t>(Count(WaitMetric{})) * kWaitProfileKinds;
      const auto wait_timing_values =
          static_cast<std::size_t>(Count(WaitTimingMetric{})) *
          kWaitProfileKinds * kTimingHistogramBuckets;
      const auto critical_lane_values =
          static_cast<std::size_t>(kWaitProfileKinds) * ready_producer_count;
      queues->ready_profile_shards_ =
          std::make_unique<ReadyProfileShard[]>(ready_profile_shards);
      queues->wait_metrics_ =
          std::make_unique<std::atomic<std::uint64_t>[]>(wait_values);
      queues->wait_timing_histograms_ =
          std::make_unique<std::atomic<std::uint64_t>[]>(wait_timing_values);
      queues->wait_critical_lane_counts_ =
          std::make_unique<std::atomic<std::uint64_t>[]>(critical_lane_values);
      for (std::size_t shard_index = 0; shard_index < ready_profile_shards;
           ++shard_index) {
        auto &shard = queues->ready_profile_shards_[shard_index];
        shard.worker_metrics =
            std::make_unique<std::atomic<std::uint64_t>[]>(ready_worker_values);
        shard.timing_histograms =
            std::make_unique<std::atomic<std::uint64_t>[]>(ready_timing_values);
        for (std::size_t index = 0; index < ready_worker_values; ++index) {
          shard.worker_metrics[index].store(0, std::memory_order_relaxed);
        }
        for (std::size_t index = 0; index < ready_timing_values; ++index) {
          shard.timing_histograms[index].store(0, std::memory_order_relaxed);
        }
      }
      for (std::size_t index = 0; index < wait_values; ++index) {
        queues->wait_metrics_[index].store(0, std::memory_order_relaxed);
      }
      for (std::size_t index = 0; index < wait_timing_values; ++index) {
        queues->wait_timing_histograms_[index].store(0,
                                                     std::memory_order_relaxed);
      }
      for (std::size_t index = 0; index < critical_lane_values; ++index) {
        queues->wait_critical_lane_counts_[index].store(
            0, std::memory_order_relaxed);
      }
    }
    return queues;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate scheduler topology");
  }
}

Status SchedulerQueues::PublishAdmission(std::uint32_t worker,
                                         QueryKey query) noexcept {
  if (worker >= search_workers_ || query.generation == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "admission target or query generation is invalid");
  }
  return admission_queues_[worker].TryPush(query);
}

Status SchedulerQueues::PublishReady(std::uint32_t lane,
                                     std::uint32_t target_worker,
                                     QueryReadyHandle handle) noexcept {
  if (lane >= ready_producer_count_ || target_worker >= search_workers_ ||
      handle.query.generation == 0 || handle.wait_epoch == 0 ||
      handle.owner_search != target_worker) {
    return Status(StatusCode::kInvalidArgument,
                  "ready target or handle is invalid");
  }
  if (profiling_enabled_) {
    handle.published_timestamp_nanoseconds = SteadyNowNanoseconds();
  }
  // Record before release: the consumer may immediately run and reuse Query.
  RecordReadyPublished(lane, target_worker, handle,
                       ready_queues_[target_worker].size() + 1U);
  auto published = ready_queues_[target_worker].TryPush(handle);
  if (!published.ok()) {
    return Status(StatusCode::kInternalError,
                  "Search event capacity/lifecycle invariant violated");
  }
  return Status::Ok();
}

Status SchedulerQueues::PublishRpcReclaim(std::uint32_t lane,
                                          RpcReclaimHandle handle) noexcept {
  if (lane >= ready_producer_count_ || handle.owner_search >= search_workers_ ||
      handle.ticket == 0) {
    return Status(StatusCode::kInvalidArgument, "invalid RPC Reclaim event");
  }
  metrics_.rpc_reclaim_published.fetch_add(1, std::memory_order_relaxed);
  auto status = ready_queues_[handle.owner_search].TryPush(handle);
  return status.ok()
             ? status
             : Status(StatusCode::kInternalError,
                      "Search event capacity/lifecycle invariant violated");
}

void SchedulerQueues::ResetMetrics() noexcept {
  metrics_.rpc_reclaim_published.store(0, std::memory_order_relaxed);
  metrics_.rpc_reclaim_consumed.store(0, std::memory_order_relaxed);
  metrics_.ready_published.store(0, std::memory_order_relaxed);
  metrics_.ready_consumed.store(0, std::memory_order_relaxed);
  metrics_.ready_depth.store(0, std::memory_order_relaxed);
  metrics_.ready_queue_high_watermark.store(0, std::memory_order_relaxed);
  metrics_.retry_count.store(0, std::memory_order_relaxed);
  metrics_.stale_query_handle.store(0, std::memory_order_relaxed);
  metrics_.stale_ready_handle.store(0, std::memory_order_relaxed);
  metrics_.stale_retry_handle.store(0, std::memory_order_relaxed);
  metrics_.duplicate_handle.store(0, std::memory_order_relaxed);
  for (auto &queue : ready_queues_)
    queue.ResetMetrics();
  if (!profiling_enabled_)
    return;
  const auto clear = [](auto *values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
      values[i].store(0, std::memory_order_relaxed);
  };
  for (std::size_t i = 0; i < ready_producer_count_ + search_workers_; ++i) {
    clear(ready_profile_shards_[i].worker_metrics.get(),
          Count(ReadyWorkerMetric{}) *
              static_cast<std::size_t>(search_workers_));
    clear(ready_profile_shards_[i].timing_histograms.get(),
          Count(ReadyTimingMetric{}) *
              static_cast<std::size_t>(search_workers_) *
              kTimingHistogramBuckets);
  }
  clear(wait_metrics_.get(), Count(WaitMetric{}) * kWaitProfileKinds);
  clear(wait_timing_histograms_.get(), Count(WaitTimingMetric{}) *
                                           kWaitProfileKinds *
                                           kTimingHistogramBuckets);
  clear(wait_critical_lane_counts_.get(),
        static_cast<std::size_t>(kWaitProfileKinds) * ready_producer_count_);
}

void SchedulerQueues::RecordReadyPublished(std::uint32_t lane,
                                           std::uint32_t target_worker,
                                           QueryReadyHandle handle,
                                           std::uint64_t lane_depth) noexcept {
  metrics_.ready_published.fetch_add(1, std::memory_order_relaxed);
  const auto depth =
      metrics_.ready_depth.fetch_add(1, std::memory_order_relaxed) + 1U;
  UpdateMaximum(metrics_.ready_queue_high_watermark, depth);
  if (!profiling_enabled_) {
    return;
  }
  const auto worker_index = [this, target_worker](ReadyWorkerMetric metric) {
    return static_cast<std::size_t>(metric) * search_workers_ + target_worker;
  };
  auto &shard = ready_profile_shards_[lane];
  const auto add = [&](ReadyWorkerMetric metric, std::uint64_t value) {
    auto &counter = shard.worker_metrics[worker_index(metric)];
    counter.store(counter.load(std::memory_order_relaxed) + value,
                  std::memory_order_relaxed);
  };
  const auto update_maximum = [&](ReadyWorkerMetric metric,
                                  std::uint64_t value) {
    auto &counter = shard.worker_metrics[worker_index(metric)];
    counter.store(std::max(counter.load(std::memory_order_relaxed), value),
                  std::memory_order_relaxed);
  };
  add(ReadyWorkerMetric::kPublished, 1);
  add(ReadyWorkerMetric::kLaneDepthSum, lane_depth);
  update_maximum(ReadyWorkerMetric::kLaneDepthHighWatermark, lane_depth);
  if (handle.ready_timestamp_nanoseconds != 0 &&
      handle.published_timestamp_nanoseconds >=
          handle.ready_timestamp_nanoseconds) {
    RecordReadyTiming(
        lane,
        static_cast<std::uint32_t>(ReadyTimingMetric::kConvergedToPublish),
        target_worker,
        handle.published_timestamp_nanoseconds -
            handle.ready_timestamp_nanoseconds);
  }
}

void SchedulerQueues::RecordReadyTiming(std::uint32_t profile_shard,
                                        std::uint32_t metric,
                                        std::uint32_t worker,
                                        std::uint64_t nanoseconds) noexcept {
  if (!profiling_enabled_ ||
      profile_shard >= ready_producer_count_ + search_workers_ ||
      metric >= Count(ReadyTimingMetric{}) || worker >= search_workers_) {
    return;
  }
  const ReadyWorkerMetric total_metric =
      metric ==
              static_cast<std::uint32_t>(ReadyTimingMetric::kConvergedToPublish)
          ? ReadyWorkerMetric::kConvergedToPublishNanoseconds
      : metric == static_cast<std::uint32_t>(ReadyTimingMetric::kPublishToPop)
          ? ReadyWorkerMetric::kPublishToPopNanoseconds
      : metric == static_cast<std::uint32_t>(ReadyTimingMetric::kPopToStep)
          ? ReadyWorkerMetric::kPopToStepNanoseconds
          : ReadyWorkerMetric::kRunNanoseconds;
  auto &shard = ready_profile_shards_[profile_shard];
  auto &total = shard.worker_metrics[static_cast<std::size_t>(total_metric) *
                                         search_workers_ +
                                     worker];
  total.store(total.load(std::memory_order_relaxed) + nanoseconds,
              std::memory_order_relaxed);
  const auto bucket = std::min<std::uint32_t>(
      nanoseconds == 0
          ? 0U
          : static_cast<std::uint32_t>(std::bit_width(nanoseconds)),
      kTimingHistogramBuckets - 1U);
  auto &samples = shard.timing_histograms[(static_cast<std::size_t>(metric) *
                                               search_workers_ +
                                           worker) *
                                              kTimingHistogramBuckets +
                                          bucket];
  samples.store(samples.load(std::memory_order_relaxed) + 1U,
                std::memory_order_relaxed);
}

std::optional<SearchInput> SchedulerQueues::PopSearchInput(
    std::uint32_t worker, std::chrono::steady_clock::time_point now) noexcept {
  if (worker >= search_workers_) {
    return std::nullopt;
  }
  constexpr std::size_t input_count = 3;
  const auto start = search_input_cursors_[worker] % input_count;
  for (std::size_t scanned = 0; scanned < input_count; ++scanned) {
    const auto input = (start + scanned) % input_count;
    if (input == input_count - 1U) {
      auto &retries = retry_queues_[worker].heap;
      if (!retries.empty() && retries.front().retry_deadline <= now) {
        std::pop_heap(retries.begin(), retries.end(), RetryLater);
        auto retry = retries.back();
        retries.pop_back();
        search_input_cursors_[worker] = input + 1U;
        return QueryRunnableHandle{retry.query, RunnableSource::kRetry,
                                   retry.retry_epoch};
      }
      continue;
    }
    if (input == 0) {
      auto admitted = admission_queues_[worker].TryPop();
      if (admitted.has_value()) {
        search_input_cursors_[worker] = input + 1U;
        return QueryRunnableHandle{*admitted, RunnableSource::kAdmission, 0};
      }
      continue;
    }
    auto event = ready_queues_[worker].TryPop();
    if (event.has_value()) {
      if (auto *reclaim = std::get_if<RpcReclaimHandle>(&*event)) {
        metrics_.rpc_reclaim_consumed.fetch_add(1, std::memory_order_relaxed);
        search_input_cursors_[worker] = input + 1U;
        return *reclaim;
      }
      const auto *ready = &std::get<QueryReadyHandle>(*event);
      metrics_.ready_consumed.fetch_add(1, std::memory_order_relaxed);
      metrics_.ready_depth.fetch_sub(1, std::memory_order_relaxed);
      const auto popped_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
      if (profiling_enabled_) {
        auto &consumed = ready_profile_shards_[ready_producer_count_ + worker]
                             .worker_metrics[static_cast<std::size_t>(
                                                 ReadyWorkerMetric::kConsumed) *
                                                 search_workers_ +
                                             worker];
        consumed.store(consumed.load(std::memory_order_relaxed) + 1U,
                       std::memory_order_relaxed);
        if (ready->published_timestamp_nanoseconds != 0 &&
            popped_at >= ready->published_timestamp_nanoseconds) {
          RecordReadyTiming(
              ready_producer_count_ + worker,
              static_cast<std::uint32_t>(ReadyTimingMetric::kPublishToPop),
              worker, popped_at - ready->published_timestamp_nanoseconds);
        }
      }
      search_input_cursors_[worker] = input + 1U;
      return QueryRunnableHandle{
          ready->query, RunnableSource::kReady, ready->wait_epoch,
          ready->published_timestamp_nanoseconds, popped_at};
    }
  }
  return std::nullopt;
}

void SchedulerQueues::RecordReadyStep(std::uint32_t worker,
                                      QueryRunnableHandle handle,
                                      std::uint64_t step_nanoseconds) noexcept {
  if (!profiling_enabled_ || worker >= search_workers_ ||
      handle.source != RunnableSource::kReady ||
      handle.ready_popped_nanoseconds == 0 ||
      step_nanoseconds < handle.ready_popped_nanoseconds) {
    return;
  }
  RecordReadyTiming(ready_producer_count_ + worker,
                    static_cast<std::uint32_t>(ReadyTimingMetric::kPopToStep),
                    worker, step_nanoseconds - handle.ready_popped_nanoseconds);
}

void SchedulerQueues::RecordReadyRun(std::uint32_t worker,
                                     std::uint64_t run_nanoseconds) noexcept {
  RecordReadyTiming(ready_producer_count_ + worker,
                    static_cast<std::uint32_t>(ReadyTimingMetric::kRun), worker,
                    run_nanoseconds);
}

void SchedulerQueues::RecordWaitTiming(std::uint32_t metric, std::uint32_t kind,
                                       std::uint64_t nanoseconds) noexcept {
  if (!profiling_enabled_ || metric >= Count(WaitTimingMetric{}) ||
      kind >= kWaitProfileKinds) {
    return;
  }
  const auto bucket = std::min<std::uint32_t>(
      nanoseconds == 0
          ? 0U
          : static_cast<std::uint32_t>(std::bit_width(nanoseconds)),
      kTimingHistogramBuckets - 1U);
  wait_timing_histograms_
      [(static_cast<std::size_t>(metric) * kWaitProfileKinds + kind) *
           kTimingHistogramBuckets +
       bucket]
          .fetch_add(1, std::memory_order_relaxed);
}

void SchedulerQueues::RecordWaitProfile(
    QueryWaitProfileSample sample) noexcept {
  const auto kind = static_cast<std::uint32_t>(sample.kind);
  if (!profiling_enabled_ || kind >= kWaitProfileKinds) {
    return;
  }
  const auto metric_index = [kind](WaitMetric metric) {
    return static_cast<std::size_t>(metric) * kWaitProfileKinds + kind;
  };
  wait_metrics_[metric_index(WaitMetric::kCount)].fetch_add(
      1, std::memory_order_relaxed);
  wait_metrics_[metric_index(WaitMetric::kNanoseconds)].fetch_add(
      sample.wait_nanoseconds, std::memory_order_relaxed);
  wait_metrics_[metric_index(WaitMetric::kPublishNanoseconds)].fetch_add(
      sample.publish_nanoseconds, std::memory_order_relaxed);
  wait_metrics_[metric_index(WaitMetric::kFirstCompletionNanoseconds)]
      .fetch_add(sample.first_completion_nanoseconds,
                 std::memory_order_relaxed);
  wait_metrics_[metric_index(WaitMetric::kCompletionSpreadNanoseconds)]
      .fetch_add(sample.completion_spread_nanoseconds,
                 std::memory_order_relaxed);
  wait_metrics_[metric_index(WaitMetric::kRemoteOperations)].fetch_add(
      sample.remote_operations, std::memory_order_relaxed);
  wait_metrics_[metric_index(WaitMetric::kDistanceValues)].fetch_add(
      sample.distance_values, std::memory_order_relaxed);
  UpdateMaximum(
      wait_metrics_[metric_index(WaitMetric::kRemoteOperationsHighWatermark)],
      sample.remote_operations);
  UpdateMaximum(
      wait_metrics_[metric_index(WaitMetric::kDistanceValuesHighWatermark)],
      sample.distance_values);
  if (sample.critical_lane < ready_producer_count_) {
    wait_critical_lane_counts_[static_cast<std::size_t>(kind) *
                                   ready_producer_count_ +
                               sample.critical_lane]
        .fetch_add(1, std::memory_order_relaxed);
  }
  RecordWaitTiming(static_cast<std::uint32_t>(WaitTimingMetric::kWait), kind,
                   sample.wait_nanoseconds);
  RecordWaitTiming(static_cast<std::uint32_t>(WaitTimingMetric::kPublish), kind,
                   sample.publish_nanoseconds);
  RecordWaitTiming(
      static_cast<std::uint32_t>(WaitTimingMetric::kFirstCompletion), kind,
      sample.first_completion_nanoseconds);
  RecordWaitTiming(
      static_cast<std::uint32_t>(WaitTimingMetric::kCompletionSpread), kind,
      sample.completion_spread_nanoseconds);
}

Status SchedulerQueues::ScheduleRetry(std::uint32_t worker,
                                      QueryRetryHandle handle) noexcept {
  if (worker >= search_workers_ || handle.query.generation == 0 ||
      handle.retry_epoch == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "local retry handle is invalid");
  }
  auto &queue = retry_queues_[worker];
  if (queue.heap.size() == queue.capacity) {
    return Status(StatusCode::kInternalError,
                  "worker-local retry capacity invariant was violated");
  }
  queue.heap.push_back(handle);
  std::push_heap(queue.heap.begin(), queue.heap.end(), RetryLater);
  metrics_.retry_count.fetch_add(1, std::memory_order_relaxed);
  return Status::Ok();
}

std::optional<std::chrono::steady_clock::time_point>
SchedulerQueues::NextRetryDeadline(std::uint32_t worker) const noexcept {
  if (worker >= search_workers_ || retry_queues_[worker].heap.empty()) {
    return std::nullopt;
  }
  return retry_queues_[worker].heap.front().retry_deadline;
}

Status SchedulerQueues::PublishDone(std::uint32_t worker,
                                    QueryDoneHandle handle) noexcept {
  if (worker >= search_workers_ || handle.query.generation == 0 ||
      (handle.outcome != RunState::kCompleted &&
       handle.outcome != RunState::kFailed)) {
    return Status(StatusCode::kInvalidArgument, "done handle is invalid");
  }
  auto published = done_queues_[worker].TryPush(handle);
  if (published.ok()) {
    scheduler_ready_.Notify();
  }
  return published;
}

std::optional<QueryDoneHandle> SchedulerQueues::PopDone() noexcept {
  const auto start = scheduler_done_cursor_;
  for (std::size_t scanned = 0; scanned < search_workers_; ++scanned) {
    const auto worker = (start + scanned) % search_workers_;
    auto done = done_queues_[worker].TryPop();
    if (done.has_value()) {
      scheduler_done_cursor_ = (worker + 1U) % search_workers_;
      (void)scheduler_ready_.TryConsume();
      return done;
    }
  }
  return std::nullopt;
}

SchedulerQueueMetrics SchedulerQueues::metrics() const {
  SchedulerQueueMetrics snapshot;
  snapshot.rpc_reclaim_published =
      metrics_.rpc_reclaim_published.load(std::memory_order_relaxed);
  snapshot.rpc_reclaim_consumed =
      metrics_.rpc_reclaim_consumed.load(std::memory_order_relaxed);
  for (const auto &queue : ready_queues_) {
    snapshot.search_event_depth.push_back(queue.size());
    snapshot.search_event_high_watermark.push_back(
        queue.metrics().high_watermark);
  }
  snapshot.ready_published =
      metrics_.ready_published.load(std::memory_order_relaxed);
  snapshot.ready_consumed =
      metrics_.ready_consumed.load(std::memory_order_relaxed);
  snapshot.ready_queue_high_watermark =
      metrics_.ready_queue_high_watermark.load(std::memory_order_relaxed);
  snapshot.retry_count = metrics_.retry_count.load(std::memory_order_relaxed);
  snapshot.stale_query_handle =
      metrics_.stale_query_handle.load(std::memory_order_relaxed);
  snapshot.stale_ready_handle =
      metrics_.stale_ready_handle.load(std::memory_order_relaxed);
  snapshot.stale_retry_handle =
      metrics_.stale_retry_handle.load(std::memory_order_relaxed);
  snapshot.duplicate_handle =
      metrics_.duplicate_handle.load(std::memory_order_relaxed);
  if (!profiling_enabled_) {
    return snapshot;
  }
  const auto copy_ready_metric = [&](ReadyWorkerMetric metric,
                                     std::vector<std::uint64_t> &output) {
    output.resize(search_workers_);
    const auto begin = static_cast<std::size_t>(metric) * search_workers_;
    const auto use_maximum =
        metric == ReadyWorkerMetric::kLaneDepthHighWatermark;
    for (std::uint32_t worker = 0; worker < search_workers_; ++worker) {
      std::uint64_t value = 0;
      for (std::uint32_t shard = 0;
           shard < ready_producer_count_ + search_workers_; ++shard) {
        const auto sample =
            ready_profile_shards_[shard].worker_metrics[begin + worker].load(
                std::memory_order_relaxed);
        value = use_maximum ? std::max(value, sample) : value + sample;
      }
      output[worker] = value;
    }
  };
  copy_ready_metric(ReadyWorkerMetric::kPublished,
                    snapshot.ready_published_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kConsumed,
                    snapshot.ready_consumed_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kLaneDepthSum,
                    snapshot.ready_lane_depth_sum_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kLaneDepthHighWatermark,
                    snapshot.ready_lane_depth_high_watermark_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kConvergedToPublishNanoseconds,
                    snapshot.ready_converged_to_publish_ns_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kPublishToPopNanoseconds,
                    snapshot.ready_publish_to_pop_ns_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kPopToStepNanoseconds,
                    snapshot.ready_pop_to_step_ns_per_worker);
  copy_ready_metric(ReadyWorkerMetric::kRunNanoseconds,
                    snapshot.ready_run_ns_per_worker);
  snapshot.ready_producer_worker_counts.resize(
      static_cast<std::size_t>(ready_producer_count_) * search_workers_);
  for (std::uint32_t lane = 0; lane < ready_producer_count_; ++lane) {
    const auto begin = static_cast<std::size_t>(ReadyWorkerMetric::kPublished) *
                       search_workers_;
    for (std::uint32_t worker = 0; worker < search_workers_; ++worker) {
      snapshot.ready_producer_worker_counts[static_cast<std::size_t>(lane) *
                                                search_workers_ +
                                            worker] =
          ready_profile_shards_[lane].worker_metrics[begin + worker].load(
              std::memory_order_relaxed);
    }
  }
  const auto copy_atomic_array = [](const std::atomic<std::uint64_t> *input,
                                    std::size_t count,
                                    std::vector<std::uint64_t> &output) {
    output.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
      output[index] = input[index].load(std::memory_order_relaxed);
    }
  };
  const auto copy_ready_histogram = [&](ReadyTimingMetric metric,
                                        std::vector<std::uint64_t> &output) {
    const auto count =
        static_cast<std::size_t>(search_workers_) * kTimingHistogramBuckets;
    output.assign(count, 0);
    const auto begin = static_cast<std::size_t>(metric) * count;
    for (std::uint32_t shard = 0;
         shard < ready_producer_count_ + search_workers_; ++shard) {
      for (std::size_t index = 0; index < count; ++index) {
        output[index] +=
            ready_profile_shards_[shard].timing_histograms[begin + index].load(
                std::memory_order_relaxed);
      }
    }
  };
  copy_ready_histogram(ReadyTimingMetric::kConvergedToPublish,
                       snapshot.ready_converged_to_publish_histograms);
  copy_ready_histogram(ReadyTimingMetric::kPublishToPop,
                       snapshot.ready_publish_to_pop_histograms);
  copy_ready_histogram(ReadyTimingMetric::kPopToStep,
                       snapshot.ready_pop_to_step_histograms);
  copy_ready_histogram(ReadyTimingMetric::kRun, snapshot.ready_run_histograms);
  const auto copy_wait_metric = [&](WaitMetric metric,
                                    std::vector<std::uint64_t> &output) {
    output.resize(kWaitProfileKinds);
    const auto begin = static_cast<std::size_t>(metric) * kWaitProfileKinds;
    for (std::uint32_t kind = 0; kind < kWaitProfileKinds; ++kind) {
      output[kind] =
          wait_metrics_[begin + kind].load(std::memory_order_relaxed);
    }
  };
  copy_wait_metric(WaitMetric::kCount, snapshot.wait_event_counts);
  copy_wait_metric(WaitMetric::kNanoseconds, snapshot.wait_nanoseconds);
  copy_wait_metric(WaitMetric::kPublishNanoseconds,
                   snapshot.wait_publish_nanoseconds);
  copy_wait_metric(WaitMetric::kFirstCompletionNanoseconds,
                   snapshot.wait_first_completion_nanoseconds);
  copy_wait_metric(WaitMetric::kCompletionSpreadNanoseconds,
                   snapshot.wait_completion_spread_nanoseconds);
  copy_wait_metric(WaitMetric::kRemoteOperations,
                   snapshot.wait_remote_operations);
  copy_wait_metric(WaitMetric::kDistanceValues, snapshot.wait_distance_values);
  copy_wait_metric(WaitMetric::kRemoteOperationsHighWatermark,
                   snapshot.wait_remote_operations_high_watermark);
  copy_wait_metric(WaitMetric::kDistanceValuesHighWatermark,
                   snapshot.wait_distance_values_high_watermark);
  copy_atomic_array(wait_critical_lane_counts_.get(),
                    static_cast<std::size_t>(kWaitProfileKinds) *
                        ready_producer_count_,
                    snapshot.wait_critical_lane_counts);
  const auto copy_wait_histogram = [&](WaitTimingMetric metric,
                                       std::vector<std::uint64_t> &output) {
    const auto count =
        static_cast<std::size_t>(kWaitProfileKinds) * kTimingHistogramBuckets;
    copy_atomic_array(wait_timing_histograms_.get() +
                          static_cast<std::size_t>(metric) * count,
                      count, output);
  };
  copy_wait_histogram(WaitTimingMetric::kWait, snapshot.wait_histograms);
  copy_wait_histogram(WaitTimingMetric::kPublish,
                      snapshot.wait_publish_histograms);
  copy_wait_histogram(WaitTimingMetric::kFirstCompletion,
                      snapshot.wait_first_completion_histograms);
  copy_wait_histogram(WaitTimingMetric::kCompletionSpread,
                      snapshot.wait_completion_spread_histograms);
  return snapshot;
}

Result<std::unique_ptr<SchedulerReadyPublisher>>
SchedulerReadyPublisher::Create(SchedulerQueues &queues,
                                std::uint32_t search_workers,
                                std::uint32_t ready_producer_count) {
  if (search_workers == 0 || ready_producer_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "ready publisher dimensions are invalid");
  }
  try {
    auto publisher =
        std::unique_ptr<SchedulerReadyPublisher>(new SchedulerReadyPublisher());
    publisher->queues_ = &queues;
    publisher->search_workers_ = search_workers;
    publisher->ready_producer_count_ = ready_producer_count;
    publisher->deferred_ready_.resize(ready_producer_count);
    for (auto &work : publisher->deferred_ready_) {
      work.entries.resize(queues.max_active_queries());
    }
    return publisher;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate ready publisher scratch");
  }
}

Status SchedulerReadyPublisher::PublishReady(std::uint32_t producer_lane,
                                             QueryReadyHandle handle) noexcept {
  if (producer_lane >= ready_producer_count_ || handle.query.generation == 0 ||
      handle.wait_epoch == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "ready publication metadata is invalid");
  }
  return queues_->PublishReady(producer_lane, handle.owner_search, handle);
}

Status
SchedulerReadyPublisher::DeferReady(std::uint32_t producer_lane,
                                    QueryReadyHandle handle,
                                    QueryRunController *controller) noexcept {
  if (producer_lane >= ready_producer_count_ || controller == nullptr ||
      handle.query.generation == 0 || handle.wait_epoch == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "deferred ready publication metadata is invalid");
  }
  auto &work = deferred_ready_[producer_lane];
  if (work.size == work.entries.size()) {
    return Status(StatusCode::kInternalError,
                  "deferred Ready capacity violated");
  }
  work.entries[(work.head + work.size) % work.entries.size()] = {
      handle, controller, false};
  ++work.size;
  return Status::Ok();
}

Result<std::size_t>
SchedulerReadyPublisher::Flush(std::uint32_t producer_lane,
                               std::size_t budget) noexcept {
  if (producer_lane >= ready_producer_count_) {
    return Status(StatusCode::kInvalidArgument, "invalid deferred Ready lane");
  }
  auto &work = deferred_ready_[producer_lane];
  const auto pending = work.size;
  std::size_t published = 0;
  for (std::size_t scanned = 0; scanned < std::min(pending, budget);
       ++scanned) {
    auto entry = work.entries[work.head];
    if (!entry.transitioned) {
      auto transitioned = entry.controller->PublishDeferredReady(
          entry.handle.query.generation, entry.handle.wait_epoch);
      if (!transitioned.ok())
        return transitioned.status();
      entry.transitioned = transitioned.value();
      if (entry.transitioned) {
        entry.handle.ready_timestamp_nanoseconds =
            entry.controller->ready_timestamp_nanoseconds();
      }
    }
    // Remove the private reference before releasing the only Ready to Search.
    work.entries[work.head] = {};
    work.head = (work.head + 1U) % work.entries.size();
    --work.size;
    if (!entry.transitioned) {
      work.entries[(work.head + work.size) % work.entries.size()] = entry;
      ++work.size;
      continue;
    }
    auto ready = PublishReady(producer_lane, entry.handle);
    if (!ready.ok()) {
      work.entries[(work.head + work.size) % work.entries.size()] = entry;
      ++work.size;
      return ready;
    }
    ++published;
  }
  return published;
}

void SchedulerReadyPublisher::RecordWaitProfile(
    QueryWaitProfileSample sample) noexcept {
  if (queues_ != nullptr) {
    queues_->RecordWaitProfile(sample);
  }
}

Result<QueryDeadlineHeap> QueryDeadlineHeap::Create(std::size_t capacity) {
  if (capacity == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "query deadline heap capacity must be non-zero");
  }
  try {
    QueryDeadlineHeap heap;
    heap.capacity_ = capacity;
    heap.heap_.reserve(capacity);
    return heap;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate query deadline heap");
  }
}

Status QueryDeadlineHeap::Schedule(QueryDeadlineEntry entry) {
  if (entry.query.generation == 0 || heap_.size() == capacity_) {
    return Status(heap_.size() == capacity_ ? StatusCode::kResourceExhausted
                                            : StatusCode::kInvalidArgument,
                  "query deadline heap cannot accept entry");
  }
  heap_.push_back(entry);
  std::push_heap(heap_.begin(), heap_.end(), DeadlineLater);
  return Status::Ok();
}

bool QueryDeadlineHeap::Cancel(QueryKey query) noexcept {
  const auto found =
      std::find_if(heap_.begin(), heap_.end(),
                   [query](const auto &entry) { return entry.query == query; });
  if (found == heap_.end()) {
    return false;
  }
  heap_.erase(found);
  std::make_heap(heap_.begin(), heap_.end(), DeadlineLater);
  return true;
}

std::optional<QueryDeadlineEntry> QueryDeadlineHeap::PopExpired(
    std::chrono::steady_clock::time_point now) noexcept {
  if (heap_.empty() || heap_.front().deadline > now) {
    return std::nullopt;
  }
  std::pop_heap(heap_.begin(), heap_.end(), DeadlineLater);
  auto entry = heap_.back();
  heap_.pop_back();
  return entry;
}

std::optional<std::chrono::steady_clock::time_point>
QueryDeadlineHeap::next_deadline() const noexcept {
  return heap_.empty() ? std::optional<std::chrono::steady_clock::time_point>{}
                       : std::optional<std::chrono::steady_clock::time_point>(
                             heap_.front().deadline);
}

} // namespace hnsw_urma
