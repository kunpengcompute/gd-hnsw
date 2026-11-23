#include "hnsw_urma/milvus_query_executor.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <new>
#include <thread>
#include <variant>

namespace hnsw_urma {
using Clock = std::chrono::steady_clock;

void MilvusQueryExecutor::Completion::Finish(Status error, SearchResult value) {
  std::lock_guard lock(mutex);
  if (done) return;
  status = std::move(error);
  result = std::move(value);
  done = true;
  ready.notify_one();
}

Result<std::unique_ptr<MilvusQueryExecutor>> MilvusQueryExecutor::Create(
    const GlobalIndexMeta& meta, const ShardMap& map, ShardEntryPoint entry,
    GraphReader& graph, DistanceOracle& distance, const RuntimeConfig& config,
    const EffectiveRuntimeConfig& effective, std::uint32_t rank,
    SchedulerQueues& queues, RpcDistanceOracle* rpc,
    std::span<SearchReadResources* const> reads, DataPlaneDrainTracker* drain,
    ProfileReporter reporter) {
  auto service = std::unique_ptr<MilvusQueryExecutor>(new MilvusQueryExecutor);
  service->meta_ = &meta;
  service->map_ = &map;
  service->entry_ = entry;
  service->graph_ = &graph;
  service->distance_ = &distance;
  service->config_ = config;
  service->effective_ = effective;
  service->queues_ = &queues;
  service->rpc_ = rpc;
  service->reads_.assign(reads.begin(), reads.end());
  service->drain_ = drain;
  service->report_profile_ = std::move(reporter);
  auto pool = QuerySlotPool::Create(rank, config.max_active_queries);
  if (!pool.ok()) return pool.status();
  service->pool_.emplace(std::move(pool).value());
  auto deadlines = QueryDeadlineHeap::Create(config.max_active_queries);
  if (!deadlines.ok()) return deadlines.status();
  service->deadlines_.emplace(std::move(deadlines).value());
  service->slots_ = std::make_unique<Slot[]>(config.max_active_queries);
  const auto producers = effective.read_progress_threads + effective.rpc_progress_threads;
  auto publisher = SchedulerReadyPublisher::Create(queues, effective.search_workers, producers);
  if (!publisher.ok()) return publisher.status();
  service->publisher_ = std::move(publisher).value();
  service->context_ = {service->publisher_.get(), producers};
  return service;
}

Result<MilvusQueryExecutor::BatchResult> MilvusQueryExecutor::Submit(
    std::span<const float> queries, std::uint32_t nq, std::uint32_t topk,
    std::uint32_t ef, Clock::time_point request_deadline) {
  // Request boundary. Internal tasks receive validated geometry and values.
  if (nq == 0 || nq > 4096 || queries.size() != std::size_t(nq) * meta_->dim ||
      topk == 0 || ef < topk ||
      !std::all_of(queries.begin(), queries.end(), [](float v) { return std::isfinite(v); })) {
    return Status(StatusCode::kInvalidArgument, "invalid UB query shape, topk, ef or non-finite vector");
  }
  SearchParams params;
  params.topk = topk;
  params.ef_search = ef;
  params.expand_batch = config_.expand_batch;
  params.layer0_sort_mode = config_.layer0_sort_mode;
  params.entry_point = entry_.global_id;
  params.entry_level = entry_.level;
  const auto now = Clock::now();
  const auto deadline = std::min(request_deadline,
      now + std::chrono::milliseconds(config_.query_timeout_ms));
  if (deadline <= now) {
    return Status(StatusCode::kTimeout, "UB request deadline expired before submission");
  }
  // Keep SearchParams' millisecond geometry valid even for sub-ms budgets.
  // The exact deadline is retained separately, including time spent pending.
  params.query_timeout = std::chrono::ceil<std::chrono::milliseconds>(deadline - now);
  params.operation_timeout = std::min(params.query_timeout,
      std::chrono::milliseconds(config_.operation_timeout_ms));
  params.profile_timing = config_.profile;
  // Same auto rule as pinned ResolveRuntimeConfig, applied to each request's ef.
  const auto visited = std::bit_ceil(std::max<std::uint64_t>(1024, std::uint64_t(ef) * meta_->M0 * 2));
  if (visited > UINT32_MAX) return Status(StatusCode::kInvalidArgument, "ef exceeds visited capacity range");
  params.visited_capacity = config_.visited_capacity.automatic
      ? static_cast<std::uint32_t>(visited) : config_.visited_capacity.value;
  std::vector<std::shared_ptr<Completion>> completions;
  completions.reserve(nq);
  {
    std::lock_guard lock(mutex_);
    if (!accepting_) return Status(StatusCode::kCancelled, "UB service is stopping");
    // Normal late responses/Reclaims may briefly occupy every slot. Let those
    // requests use the existing pending queue and deadline, just as when every
    // slot is active. Fail fast only after all slots exceeded the drain budget.
    if (quarantined_wire_slots_ >= config_.max_active_queries) {
      return Status(StatusCode::kResourceExhausted,
                    "all UB query slots are awaiting wire retirement (drain timeout exceeded)");
    }
    // Reserve the complete batch before copying. Bound queue entries and bytes
    // even when all wire slots are temporarily retiring.
    constexpr std::size_t kPendingRows = 4096;
    constexpr std::size_t kPendingBytes = 64U * 1024U * 1024U;
    if (pending_.size() > kPendingRows - nq ||
        meta_->dim > kPendingBytes / sizeof(float) / (pending_.size() + nq))
      return Status(StatusCode::kResourceExhausted, "UB pending queue capacity exhausted");
    const auto original_size = pending_.size();
    try {
    for (std::uint32_t row = 0; row < nq; ++row) {
      auto completion = std::make_shared<Completion>();
      auto vector = queries.subspan(std::size_t(row) * meta_->dim, meta_->dim);
      pending_.push_back({{vector.begin(), vector.end()}, params, completion, deadline});
      completions.push_back(std::move(completion));
    }
    } catch (const std::bad_alloc&) {
      while (pending_.size() > original_size) pending_.pop_back();
      return Status(StatusCode::kResourceExhausted, "unable to reserve complete UB query batch");
    }
  }
  queues_->scheduler_ready().Notify();
  std::vector<SearchResult> results;
  results.reserve(nq);
  for (auto& completion : completions) {
    std::unique_lock lock(completion->mutex);
    if (!completion->ready.wait_until(lock, deadline, [&] { return completion->done; })) {
      return Status(StatusCode::kTimeout, "UB query deadline exceeded in service queue/search");
    }
    if (!completion->status.ok()) return completion->status;
    results.push_back(std::move(completion->result));
  }
  return BatchResult{params, std::move(results)};
}

bool MilvusQueryExecutor::WireQuiesced(QueryKey key) const {
  return rpc_ == nullptr || rpc_->QueryWireQuiesced(key);
}
void MilvusQueryExecutor::StopAdmission() {
  std::lock_guard lock(mutex_);
  accepting_ = false;
  queues_->scheduler_ready().Notify();
}
void MilvusQueryExecutor::FailPending(Status status) {
  // Called after Control exits; no other thread writes completions.
  std::lock_guard lock(mutex_);
  accepting_ = false;
  for (auto& pending : pending_) pending.completion->Finish(status);
  pending_.clear();
  for (std::uint32_t i = 0; i < config_.max_active_queries; ++i) {
    if (slots_[i].completion) {
      slots_[i].completion->Finish(status);
      slots_[i].completion.reset();
    }
  }
}

Status MilvusQueryExecutor::Control(std::stop_token stop) {
  struct Deferred {
    QueryKey key;
    Clock::time_point deadline;
    bool quarantined{false};
  };
  std::deque<Deferred> deferred;
  std::uint32_t active = 0, cursor = 0;
  Profile profile;
  auto next_report = Clock::time_point{};
  while (!stop.stop_requested()) {
    bool quarantine_changed = false;
    for (auto it = deferred.begin(); it != deferred.end();) {
      if (!WireQuiesced(it->key)) {
        if (!it->quarantined && Clock::now() >= it->deadline) {
          // Keep the slot and task storage while late wire work can still use it.
          std::lock_guard lock(mutex_);
          it->quarantined = true;
          ++quarantined_wire_slots_;
          quarantine_changed = true;
        }
        ++it;
      } else {
        {
          std::lock_guard lock(mutex_);
          slots_[it->key.query_slot].storage.reset();
          auto released = pool_->Release(it->key);
          if (!released.ok()) return released;
          if (it->quarantined) --quarantined_wire_slots_;
        }
        it = deferred.erase(it);
      }
    }
    while (auto done = queues_->PopDone()) {
      auto& slot = slots_[done->query.query_slot];
      auto* task = slot.task.exchange(nullptr, std::memory_order_acq_rel);
      while (slot.readers.load(std::memory_order_acquire)) std::this_thread::yield();
      if (!slot.deadline_expired) (void)deadlines_->Cancel(done->query);
      if (config_.profile) {
        if (task->status().ok()) {
          const auto& result = task->result();
          ++profile.completed;
          profile.pre_admit_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
              slot.admitted_at - slot.submitted_at).count();
          profile.active_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now() - slot.admitted_at).count();
          profile.execution_ns += result.timing.LocalExecutionNanoseconds();
          profile.graph_wait_ns += result.timing.graph_wait_nanoseconds;
          profile.distance_wait_ns += result.timing.distance_wait_nanoseconds;
          profile.ready_wait_ns += result.timing.ready_queue_nanoseconds;
          profile.upper_ns += result.timing.upper_elapsed_nanoseconds;
          profile.layer0_ns += result.timing.layer0_elapsed_nanoseconds;
          profile.nodes += result.profile.nodes_expanded;
          profile.edges += result.profile.edges_scanned;
        } else {
          ++profile.errors;
        }
      }
      // Copy before destruction: result/status belong to the task.
      slot.completion->Finish(task->status(), task->result());
      slot.completion.reset();
      if (WireQuiesced(done->query)) {
        slot.storage.reset();
        auto released = pool_->Release(done->query);
        if (!released.ok()) return released;
      } else {
        deferred.push_back({done->query,
                            Clock::now() + std::chrono::milliseconds(config_.drain_timeout_ms),
                            false});
      }
      --active;
    }
    bool finished = false;
    {
      std::lock_guard lock(mutex_);
      while (!pending_.empty() && active < config_.max_active_queries) {
        auto& pending = pending_.front();
        if (Clock::now() >= pending.deadline) {
          pending.completion->Finish(Status(StatusCode::kTimeout, "UB query expired before admission"));
          pending_.pop_front();
          continue;
        }
        auto key = pool_->Acquire();
        if (!key.ok()) break;  // Slots awaiting wire retirement are unavailable.
        auto& slot = slots_[key.value().query_slot];
        if (!slot.scratch || slot.topk != pending.params.topk || slot.ef != pending.params.ef_search) {
          auto scratch = QueryScratch::Create(*meta_, pending.params, config_.max_active_queries, context_.ready_producer_count);
          if (!scratch.ok()) {
            (void)pool_->Release(key.value());
            pending.completion->Finish(scratch.status());
            pending_.pop_front();
            continue;
          }
          slot.scratch = std::move(scratch).value();
          slot.topk = pending.params.topk;
          slot.ef = pending.params.ef_search;
        }
        auto created = HnswSearchTask::Create(key.value(), *meta_, *map_, *graph_, *distance_, pending.query,
            pending.params, pending.deadline - pending.params.query_timeout, &context_, slot.scratch.get());
        if (!created.ok()) {
          (void)pool_->Release(key.value());
          pending.completion->Finish(created.status());
          pending_.pop_front();
          continue;
        }
        slot.storage = std::move(created).value();
        slot.completion = pending.completion;
        slot.deadline_expired = false;
        const auto rollback = [&] {
          slot.task.store(nullptr, std::memory_order_release);
          slot.storage.reset();
          slot.completion.reset();
          (void)pool_->Release(key.value());
        };
        auto bound = slot.storage->BindOwner(cursor);
        if (!bound.ok()) {
          rollback();
          return bound;
        }
        if (config_.profile) {
          slot.submitted_at = pending.deadline - pending.params.query_timeout;
          slot.admitted_at = Clock::now();
        }
        // Reserve the deadline before the worker can observe the task. Any
        // failure up to publication can then roll back without a live reader.
        auto scheduled = deadlines_->Schedule({key.value(), pending.deadline});
        if (!scheduled.ok()) {
          rollback();
          return scheduled;
        }
        slot.task.store(slot.storage.get(), std::memory_order_release);
        auto admitted = queues_->PublishAdmission(cursor, key.value());
        if (!admitted.ok()) {
          (void)deadlines_->Cancel(key.value());
          rollback();
          if (admitted.code() != StatusCode::kResourceExhausted) return admitted;
          break;
        }
        cursor = (cursor + 1) % effective_.search_workers;
        pending_.pop_front();
        ++active;
      }
      profile.pending = pending_.size();
      finished = !accepting_ && pending_.empty() && active == 0 && deferred.empty();
    }
    const auto now = Clock::now();
    if (report_profile_ && (config_.profile || !deferred.empty()) &&
        (quarantine_changed || now >= next_report || finished)) {
      profile.active = active;
      profile.retiring = deferred.size();
      profile.quarantined = std::count_if(
          deferred.begin(), deferred.end(),
          [](const auto& entry) { return entry.quarantined; });
      report_profile_(profile);
      next_report = now + std::chrono::seconds(1);
    }
    if (finished) {
      drained_.store(true);
      return Status::Ok();
    }
    while (auto expired = deadlines_->PopExpired(now)) {
      auto& slot = slots_[expired->query.query_slot];
      TaskRef task(slot);
      auto failed = task->RequestFailure(TerminalOutcome::kTimedOut, StatusCode::kTimeout);
      if (!failed.ok()) return failed;
      slot.deadline_expired = true;
    }
    (void)queues_->scheduler_ready().WaitUntil(stop, now + std::chrono::milliseconds(1));
  }
  return Status(StatusCode::kCancelled, "UB scheduler stopped");
}

Status MilvusQueryExecutor::Search(std::stop_token stop, std::uint32_t worker) {
  SetCurrentDistanceSearchWorker(worker);
  while (!stop.stop_requested()) {
    if (!reads_.empty()) (void)reads_[worker]->PollReclaims();
    if (drain_) {
      LaneDrainSnapshot snapshot;
      snapshot.read_active = reads_[worker]->drained() ? 0 : 1;
      snapshot.rpc_active = rpc_->SearchDrained(worker) ? 0 : 1;
      drain_->PublishLane(context_.ready_producer_count + worker, snapshot);
    }
    auto input = queues_->PopSearchInput(worker, Clock::now());
    if (!input) continue;
    if (std::holds_alternative<RpcReclaimHandle>(*input)) {
      auto status = rpc_->HandleReclaim(worker, std::get<RpcReclaimHandle>(*input));
      if (!status.ok()) return status;
      continue;
    }
    const auto runnable = std::get<QueryRunnableHandle>(*input);
    TaskRef task(slots_[runnable.query.query_slot]);
    // Keep the pinned scheduler's epoch checks: these enforce slot reuse.
    if (!task || task->key() != runnable.query ||
        (runnable.source == RunnableSource::kReady && !task->run_controller().ValidateReadyHandle(
            {runnable.query, runnable.epoch, 0, 0, worker})) ||
        (runnable.source == RunnableSource::kRetry && !task->run_controller().ValidateRetryHandle(
            {runnable.query, runnable.epoch, {}}))) {
      return Status(StatusCode::kProtocolError, "stale UB runnable handle");
    }
    const auto step = task->Step();
    if (task->requires_runtime_abort()) {
      // A failed publication/rollback can leave READ callbacks holding task
      // state. Stop the thread group and retain its slot instead of sending
      // Done, which would let Control destroy the task before Progress stops.
      return task->status();
    }
    if (step == SearchStepResult::kCompleted || step == SearchStepResult::kFailed) {
      auto done = queues_->PublishDone(worker, {runnable.query, task->run_state()});
      if (!done.ok()) return done;
    } else {
      const auto disposition = ClassifyYieldedQueryState(task->run_state());
      if (disposition == YieldedQueryDisposition::kQueued) {
        if (auto retry = task->TakeRetryHandle()) {
          auto scheduled = queues_->ScheduleRetry(worker, *retry);
          if (!scheduled.ok()) return scheduled;
        }
      } else if (disposition != YieldedQueryDisposition::kHandedOff) {
        return Status(StatusCode::kProtocolError, "UB query yielded without handoff");
      }
    }
  }
  return Status::Ok();
}
}  // namespace hnsw_urma
