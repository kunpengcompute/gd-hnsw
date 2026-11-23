#include "hnsw_urma/transport/read_batch_lane.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <new>
#include <utility>

namespace hnsw_urma {
namespace {
constexpr std::size_t kSegmentCount = 6;
std::uint64_t Elapsed(std::chrono::steady_clock::time_point start,
                      std::chrono::steady_clock::time_point end) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}
}  // namespace

Result<std::unique_ptr<ReadBatchLane>> ReadBatchLane::Create(
    LanePostEngine& engine, std::uint32_t owner_lane,
    ReadBatchLaneConfig config) {
  if (config.search_workers == 0 || config.peer_count == 0 ||
      config.submit_depth == 0 || config.records_per_search == 0 ||
      config.in_flight_limit == 0 || config.wr_list_max == 0 ||
      config.poll_batch == 0 || config.work_budget == 0) {
    return Status(StatusCode::kInvalidArgument, "READ lane bounds must be non-zero");
  }
  try {
    auto lane = std::unique_ptr<ReadBatchLane>(new ReadBatchLane());
    lane->engine_ = &engine;
    lane->owner_lane_ = owner_lane;
    lane->config_ = config;
    for (std::uint32_t search = 0; search < config.search_workers; ++search) {
      auto submit = SpscRing<ReadBatch*>::Create(config.submit_depth);
      auto reclaim = SpscRing<ReadBatch*>::Create(config.records_per_search);
      if (!submit.ok() || !reclaim.ok())
        return submit.ok() ? reclaim.status() : submit.status();
      lane->submit_queues_.push_back(std::move(submit).value());
      lane->reclaim_queues_.push_back(std::move(reclaim).value());
    }
    lane->next_batch_tickets_.resize(config.search_workers, 1);
    lane->active_batches_.reserve(
        static_cast<std::size_t>(config.records_per_search) * config.search_workers);
    lane->groups_.resize(static_cast<std::size_t>(config.peer_count) * kSegmentCount);
    lane->in_flight_.resize(config.in_flight_limit);
    lane->free_records_.reserve(config.in_flight_limit);
    for (std::uint32_t i = config.in_flight_limit; i != 0; --i)
      lane->free_records_.push_back(i - 1U);
    lane->post_operations_.reserve(config.wr_list_max);
    lane->posting_reads_.reserve(config.wr_list_max);
    lane->completion_batch_.resize(config.poll_batch);
    return lane;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate READ lane");
  }
}

Result<ReadBatchTicket> ReadBatchLane::PrepareBatch(ReadBatch& batch) noexcept {
  assert(batch.state.load(std::memory_order_relaxed) == ReadBatchState::kFree);
  assert(batch.owner_search < config_.search_workers);
  assert(!batch.requests.empty() && batch.terminal_callback != nullptr);
  assert(batch.ranges.capacity() >= batch.requests.size());
  auto reserved = submit_queues_[batch.owner_search].Reserve(1);
  if (!reserved.ok()) return reserved;
  batch.ranges.clear();
  for (std::size_t begin = 0; begin < batch.requests.size();) {
    const auto& request = batch.requests[begin];
    assert(request.peer_rank < config_.peer_count && !request.destination.empty());
    std::size_t end = begin + 1;
    while (end < batch.requests.size() &&
           batch.requests[end].peer_rank == request.peer_rank &&
           batch.requests[end].remote_segment == request.remote_segment) ++end;
    batch.ranges.push_back({&batch, static_cast<std::uint32_t>(begin),
        static_cast<std::uint32_t>(end),
        request.peer_rank * kSegmentCount + static_cast<std::size_t>(request.remote_segment)});
    begin = end;
  }
  auto& next = next_batch_tickets_[batch.owner_search];
  if (next == 0) ++next;
  batch.ticket = {next++, owner_lane_};
  batch.completed_reads = batch.posted_reads = batch.in_flight = 0;
  batch.status = StatusCode::kOk;
  batch.reclaim_required = batch.reclaim_received = false;
  batch.state.store(ReadBatchState::kPrepared, std::memory_order_relaxed);
  return batch.ticket;
}

Status ReadBatchLane::CancelPreparedBatch(ReadBatch& batch) noexcept {
  assert(batch.state.load(std::memory_order_relaxed) == ReadBatchState::kPrepared);
  auto status = submit_queues_[batch.owner_search].CancelReservation(1);
  batch.state.store(ReadBatchState::kFree, std::memory_order_relaxed);
  return status;
}

void ReadBatchLane::NotifyWork() noexcept {
  if (adaptive_notifications_enabled_ &&
      !work_pending_.exchange(true, std::memory_order_acq_rel)) ready_.Notify();
}

void ReadBatchLane::PublishBatch(ReadBatch& batch) noexcept {
  if (config_.profiling_enabled) batch.submitted_at = std::chrono::steady_clock::now();
  batch.state.store(ReadBatchState::kQueued, std::memory_order_release);
  const auto published = submit_queues_[batch.owner_search].PushReserved(&batch);
  assert(published.ok());
  if (!published.ok()) std::terminate();
  NotifyWork();
}

ReadBatch* ReadBatchLane::PopReclaim(std::uint32_t search) noexcept {
  auto event = reclaim_queues_[search].TryPop();
  return event ? *event : nullptr;
}

void ReadBatchLane::LinkRange(ReadRequestRange& range) noexcept {
  auto*& head = groups_[range.group];
  if (head == nullptr) {
    head = range.previous = range.following = &range;
  } else {
    range.previous = head->previous;
    range.following = head;
    head->previous->following = &range;
    head->previous = &range;
  }
}

void ReadBatchLane::UnlinkRange(ReadRequestRange& range) noexcept {
  if (range.following == nullptr) return;
  auto*& head = groups_[range.group];
  if (range.following == &range) {
    head = nullptr;
  } else {
    if (head == &range) head = range.following;
    range.previous->following = range.following;
    range.following->previous = range.previous;
  }
  range.previous = range.following = nullptr;
}

void ReadBatchLane::RemoveActive(ReadBatch& batch) noexcept {
  for (auto& range : batch.ranges) UnlinkRange(range);
  const auto position = batch.active_position;
  active_batches_[position] = active_batches_.back();
  active_batches_[position]->active_position = position;
  active_batches_.pop_back();
}

Status ReadBatchLane::FinishBatch(ReadBatch& batch, StatusCode status) noexcept {
  RemoveActive(batch);
  batch.status = status;
  auto terminal = ReadBatchState::kSucceeded;
  if (status != StatusCode::kOk) {
    terminal = ReadBatchState::kFailed;
    if (status == StatusCode::kTimeout) {
      terminal = ReadBatchState::kTimedOut;
      ++profile_.batches_timed_out;
    } else if (status == StatusCode::kCancelled) {
      terminal = ReadBatchState::kCancelled;
      ++profile_.batches_cancelled;
    } else ++profile_.batches_failed;
    batch.reclaim_required = batch.in_flight != 0;
    if (batch.reclaim_required) {
      ++profile_.exceptional_batches;
      profile_.exceptional_batches_high_watermark = std::max(
          profile_.exceptional_batches_high_watermark, profile_.exceptional_batches);
      if (config_.profiling_enabled) batch.failed_at = std::chrono::steady_clock::now();
    }
  }
  batch.control_check = nullptr;
  batch.control_context = nullptr;
  const auto callback = batch.terminal_callback;
  auto* const context = batch.terminal_context;
  batch.state.store(terminal, std::memory_order_release);
  // The callback may synchronously recycle this batch. All lane references
  // (apart from stable late-DMA records) and statistics are settled first.
  return callback(context, batch);
}

Status ReadBatchLane::PollCompletions() {
  auto polled = engine_->Poll(completion_batch_);
  if (!polled.ok()) return polled.status();
  const auto now = config_.profiling_enabled ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
  for (std::size_t i = 0; i < polled.value(); ++i) {
    const auto& completion = completion_batch_[i];
    if (completion.operation_class != OperationClass::kRead ||
        completion.ticket.operation_slot >= in_flight_.size())
      return Status(StatusCode::kProtocolError, "READ CQE has invalid WR context");
    const auto record_index = completion.ticket.operation_slot;
    auto& record = in_flight_[record_index];
    // Native providers validate their contexts; generic engines use the same
    // minimum index/generation boundary before a record can be dereferenced.
    if (record.batch == nullptr || completion.ticket.generation != record.generation)
      return Status(StatusCode::kProtocolError, "READ CQE has stale WR context");
    auto* batch = record.batch;
    if (config_.profiling_enabled) {
      const auto elapsed = Elapsed(record.posted_at, now);
      ++profile_.wr_completion_samples;
      profile_.wr_completion_nanoseconds += elapsed;
      ++profile_.wr_completion_histogram[std::min<std::size_t>(static_cast<std::size_t>(std::bit_width(elapsed)), 63)];
    }
    const auto bytes = record.bytes;
    record.batch = nullptr;
    free_records_.push_back(record_index);
    --batch->in_flight;
    if (batch->reclaim_required) {
      ++profile_.late_completion_count;
      if (batch->in_flight == 0) {
        --profile_.exceptional_batches;
        if (config_.profiling_enabled)
          profile_.reclaim_wait_nanoseconds += Elapsed(batch->failed_at, now);
        auto& queue = reclaim_queues_[batch->owner_search];
        const auto published = queue.TryPush(batch);
        // Every event owns a non-reusable Search record, so the configured
        // record bound proves capacity. Never drop an event on overflow.
        if (!published.ok()) std::terminate();
        // No dereference after publishing to Search.
      }
      continue;
    }
    if (!completion.status.ok()) {
      ++profile_.cqe_failure_count;
      auto failed = FinishBatch(*batch, completion.status.code());
      if (!failed.ok()) return failed;
    } else {
      ++batch->completed_reads;
      ++profile_.requests_completed;
      profile_.bytes_completed += bytes;
      if (batch->completed_reads == batch->requests.size()) {
        auto succeeded = FinishBatch(*batch, StatusCode::kOk);
        if (!succeeded.ok()) return succeeded;
      }
    }
  }
  return Status::Ok();
}

void ReadBatchLane::DrainSubmissions() {
  std::size_t empty = 0;
  for (std::size_t work = 0; work < config_.work_budget &&
       empty < submit_queues_.size() && active_batches_.size() < active_batches_.capacity(); ++work) {
    auto& queue = submit_queues_[submit_cursor_++ % submit_queues_.size()];
    auto submitted = queue.TryPop();
    if (!submitted) { ++empty; continue; }
    empty = 0;
    auto* batch = *submitted;
    batch->active_position = active_batches_.size();
    active_batches_.push_back(batch);
    batch->state.store(ReadBatchState::kActive, std::memory_order_relaxed);
    for (auto& range : batch->ranges) LinkRange(range);
    ++profile_.batches_submitted;
    profile_.active_batch_high_watermark = std::max<std::uint64_t>(
        profile_.active_batch_high_watermark, active_batches_.size());
  }
}

Status ReadBatchLane::HandleControl() {
  const auto now = std::chrono::steady_clock::now();
  const auto budget = std::min(config_.work_budget, active_batches_.size());
  for (std::size_t work = 0; work < budget && !active_batches_.empty(); ++work) {
    control_cursor_ %= active_batches_.size();
    auto* batch = active_batches_[control_cursor_];
    auto status = transport_failure_.ok() ? StatusCode::kOk : transport_failure_.code();
    if (status == StatusCode::kOk && batch->control_check != nullptr)
      status = batch->control_check(batch->control_context, now);
    if (status == StatusCode::kOk && now >= batch->deadline) status = StatusCode::kTimeout;
    if (status != StatusCode::kOk) {
      auto failed = FinishBatch(*batch, status);
      if (!failed.ok()) return failed;
    } else ++control_cursor_;
  }
  return Status::Ok();
}

Status ReadBatchLane::PostPendingReads() {
  if (!transport_failure_.ok()) return transport_failure_;
  std::size_t work = 0;
  std::size_t scanned_groups = 0;
  while (work < config_.work_budget && scanned_groups < std::min(groups_.size(), config_.work_budget) && !free_records_.empty()) {
    const auto group_index = group_cursor_++ % groups_.size();
    ++scanned_groups;
    auto*& head = groups_[group_index];
    if (head == nullptr) continue;
    post_operations_.clear();
    posting_reads_.clear();
    // A bounded cyclic walk across batches in this compatible group. Selected
    // offsets are speculative until the provider reports its accepted prefix.
    auto* const start = head;
    bool any = false;
    while (head != nullptr && post_operations_.size() < config_.wr_list_max &&
           !free_records_.empty() && work < config_.work_budget) {
      auto* range = head;
      head = range->following;
      ++work;
      if (range->next + range->selected < range->end) {
        const auto index = range->next + range->selected++;
        const auto& request = range->batch->requests[index];
        const auto record_index = free_records_.back();
        free_records_.pop_back();
        auto& record = in_flight_[record_index];
        if (++record.generation == 0) ++record.generation;
        record.batch = range->batch;
        record.bytes = request.destination.size();
        post_operations_.push_back({
            {range->batch->ticket.value, record_index, owner_lane_, record.generation},
            request.peer_rank, request.remote_segment, request.destination, request.remote_offset});
        posting_reads_.push_back({range, record_index});
        any = true;
      }
      if (head == start) {
        if (!any) break;
        any = false;
      }
    }
    if (post_operations_.empty()) continue;
    const auto posted = engine_->PostReads(post_operations_);
    const auto now = config_.profiling_enabled ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
    if (posted.posted > posting_reads_.size())
      return Status(StatusCode::kProtocolError, "provider accepted more READs than submitted");
    profile_.post_calls += posted.post_calls;
    profile_.post_list_wrs += posted.posted;
    profile_.post_list_high_watermark = std::max<std::uint64_t>(profile_.post_list_high_watermark, posted.posted);
    for (std::size_t i = 0; i < posting_reads_.size(); ++i) {
      const auto item = posting_reads_[i];
      auto& record = in_flight_[item.record];
      --item.range->selected;
      if (i < posted.posted) {
        auto& batch = *record.batch;
        if (config_.profiling_enabled && batch.posted_reads == 0)
          profile_.submit_wait_nanoseconds += Elapsed(batch.submitted_at, now);
        record.posted_at = now;
        ++item.range->next;
        ++batch.posted_reads;
        ++batch.in_flight;
        ++profile_.requests_posted;
      } else {
        record.batch = nullptr;
        free_records_.push_back(item.record);
      }
    }
    for (const auto item : posting_reads_)
      if (item.range->next == item.range->end) UnlinkRange(*item.range);
    profile_.in_flight_high_watermark = std::max<std::uint64_t>(
        profile_.in_flight_high_watermark, in_flight_.size() - free_records_.size());
    if (!posted.status.ok() && posted.status.code() != StatusCode::kResourceExhausted) {
      // The generic provider result gives no reliable per-WR fault scope.
      // Stop this transport; accepted WR records remain live for safe drain.
      transport_failure_ = posted.status;
      return transport_failure_;
    }
    if (posted.posted == 0) break;
  }
  return Status::Ok();
}

Status ReadBatchLane::Progress() {
  if (progress_thread_ == std::thread::id{}) progress_thread_ = std::this_thread::get_id();
  else if (progress_thread_ != std::this_thread::get_id())
    return Status(StatusCode::kInternalError, "READ lane progressed by a non-owner thread");
  auto status = PollCompletions();
  if (!status.ok()) return status;
  DrainSubmissions();
  status = HandleControl();
  if (!status.ok()) return status;
  status = PostPendingReads();
  if (adaptive_notifications_enabled_) {
    work_pending_.store(false, std::memory_order_release);
    for (const auto& queue : submit_queues_) if (queue.size() != 0) { NotifyWork(); break; }
  }
  return status;
}

ReadBatchProfile ReadBatchLane::profile() const noexcept {
  auto snapshot = profile_;
  for (const auto& queue : submit_queues_) {
    const auto metrics = queue.metrics();
    snapshot.submit_queue_high_watermark = std::max(snapshot.submit_queue_high_watermark, metrics.high_watermark);
    snapshot.submit_queue_full_count += metrics.full_count;
  }
  return snapshot;
}

bool ReadBatchLane::drained() const noexcept {
  if (!active_batches_.empty() || free_records_.size() != in_flight_.size()) return false;
  for (const auto& queue : submit_queues_) if (queue.size() != 0) return false;
  for (const auto& queue : reclaim_queues_) if (queue.size() != 0) return false;
  return true;
}

Status ReadBatchLane::ReleaseProgressOwner() {
  if (!drained()) return Status(StatusCode::kProtocolError, "READ owner cannot change before drain");
  progress_thread_ = {};
  return Status::Ok();
}
}  // namespace hnsw_urma
