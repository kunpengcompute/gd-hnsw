#include "hnsw_urma/search/query_context.hpp"

#include <cassert>
#include <chrono>
#include <limits>

namespace hnsw_urma {

namespace {

bool SameOperation(QueryOperationHandle lhs,
                   QueryOperationHandle rhs) noexcept {
  return lhs.operation_slot == rhs.operation_slot &&
         lhs.owner_lane == rhs.owner_lane &&
         lhs.query_generation == rhs.query_generation &&
         lhs.wait_epoch == rhs.wait_epoch && lhs.ticket == rhs.ticket;
}

std::uint64_t SteadyNowNanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

} // namespace

Status QueryOperationState::Prepare(QueryOperationHandle handle) noexcept {
  if (state_ != State::kFree || handle.query_generation == 0 ||
      handle.wait_epoch == 0 || handle.ticket == 0) {
    return Status(StatusCode::kProtocolError,
                  "query operation cannot be prepared");
  }
  handle_ = handle;
  state_ = State::kPrepared;
  return Status::Ok();
}

Status
QueryOperationState::MarkPublished(QueryOperationHandle handle) noexcept {
  if (state_ != State::kPrepared || !SameOperation(handle_, handle)) {
    return Status(StatusCode::kProtocolError,
                  "query operation publish metadata mismatch");
  }
  state_ = State::kPublished;
  return Status::Ok();
}

Result<bool> QueryOperationState::TryMarkTerminal(
    QueryOperationHandle handle, OperationTerminalOutcome outcome) noexcept {
  if (!SameOperation(handle_, handle)) {
    return Status(StatusCode::kProtocolError,
                  "query operation terminal metadata mismatch");
  }
  if (state_ == State::kTerminal) {
    return false;
  }
  if (state_ != State::kPublished) {
    return Status(StatusCode::kProtocolError,
                  "unpublished query operation cannot become terminal");
  }
  outcome_ = outcome;
  state_ = State::kTerminal;
  return true;
}

void QueryOperationState::Reset() noexcept {
  handle_ = {};
  state_ = State::kFree;
  outcome_ = OperationTerminalOutcome::kFailed;
}

QueryRunController::QueryRunController(std::uint64_t generation,
                                       std::uint32_t max_lanes,
                                       bool timing_enabled)
    : generation_(generation), max_lanes_(max_lanes),
      timing_enabled_(timing_enabled),
      lane_pending_(std::make_unique<LanePending[]>(max_lanes)) {}

Status QueryRunController::BindOwner(std::uint32_t owner) noexcept {
  if (execution_started_) {
    return Status(StatusCode::kProtocolError,
                  "query owner is fixed after admission");
  }
  owner_search_ = owner;
  return Status::Ok();
}

Status QueryRunController::Admit() noexcept {
  RunState expected = RunState::kIdle;
  if (!state_.compare_exchange_strong(expected, RunState::kQueued,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
    return Status(StatusCode::kProtocolError,
                  "query can only be admitted from Idle");
  }
  return Status::Ok();
}

bool QueryRunController::TryAcquire() noexcept {
  RunState expected = RunState::kQueued;
  const bool acquired = state_.compare_exchange_strong(
      expected, RunState::kRunning, std::memory_order_acquire,
      std::memory_order_relaxed);
  if (acquired)
    execution_started_ = true;
  return acquired;
}

Status QueryRunController::Requeue() noexcept {
  RunState expected = RunState::kRunning;
  if (!state_.compare_exchange_strong(expected, RunState::kQueued,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
    return Status(StatusCode::kProtocolError,
                  "query can only requeue from Running");
  }
  return Status::Ok();
}

Result<std::uint64_t> QueryRunController::RequeueForRetry() noexcept {
  const auto epoch = retry_epoch_.fetch_add(1, std::memory_order_relaxed) + 1U;
  auto status = Requeue();
  if (!status.ok()) {
    return status;
  }
  return epoch;
}

Result<std::uint64_t>
QueryRunController::BeginWait(std::span<const LaneOperationCount> lanes,
                              std::uint64_t generation) noexcept {
  if (generation != generation_ || lanes.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "wait generation or participating lanes are invalid");
  }
  const auto epoch = wait_epoch_.load(std::memory_order_relaxed) + 1U;
  if (epoch == 0) {
    return Status(StatusCode::kResourceExhausted, "query wait epoch exhausted");
  }
  std::uint32_t operation_total = 0;
  for (std::size_t index = 0; index < lanes.size(); ++index) {
    const auto &entry = lanes[index];
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (lanes[previous].lane == entry.lane) {
        return Status(StatusCode::kProtocolError,
                      "wait contains a duplicate lane");
      }
    }
    if (entry.lane >= max_lanes_ || entry.operations == 0 ||
        lane_pending_[entry.lane].operations.load(std::memory_order_relaxed) !=
            0) {
      return Status(StatusCode::kProtocolError,
                    "wait lane is invalid, duplicated, or still active");
    }
    if (entry.operations >
        std::numeric_limits<std::uint32_t>::max() - operation_total) {
      return Status(StatusCode::kOutOfRange, "wait operation count overflows");
    }
    operation_total += entry.operations;
  }
  for (const auto &entry : lanes) {
    lane_pending_[entry.lane].operations.store(entry.operations,
                                               std::memory_order_relaxed);
    lane_pending_[entry.lane].epoch.store(epoch, std::memory_order_relaxed);
  }
  remaining_operations_.store(operation_total, std::memory_order_relaxed);
  remaining_lanes_.store(static_cast<std::uint32_t>(lanes.size()),
                         std::memory_order_relaxed);
  ready_timestamp_nanoseconds_.store(0, std::memory_order_relaxed);
  operation_ready_timestamp_nanoseconds_.store(0, std::memory_order_relaxed);
  first_operation_timestamp_nanoseconds_.store(0, std::memory_order_relaxed);
  critical_ready_lane_.store(std::numeric_limits<std::uint32_t>::max(),
                             std::memory_order_relaxed);
  operations_converged_.store(false, std::memory_order_relaxed);
  wait_epoch_.store(epoch, std::memory_order_relaxed);
  RunState expected = RunState::kRunning;
  if (!state_.compare_exchange_strong(expected, RunState::kWaiting,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
    for (const auto &entry : lanes) {
      lane_pending_[entry.lane].operations.store(0, std::memory_order_relaxed);
    }
    remaining_operations_.store(0, std::memory_order_relaxed);
    remaining_lanes_.store(0, std::memory_order_relaxed);
    return Status(StatusCode::kProtocolError,
                  "query can only begin a wait from Running");
  }
  return epoch;
}

Status QueryRunController::BeginDeferredReady() noexcept {
  if (state_.load(std::memory_order_acquire) != RunState::kRunning) {
    return Status(StatusCode::kProtocolError,
                  "deferred ready requires the active runner");
  }
  bool expected = false;
  if (!deferred_ready_mode_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "deferred ready is already active");
  }
  inline_runner_active_.store(true, std::memory_order_release);
  return Status::Ok();
}

void QueryRunController::EndDeferredReady() noexcept {
  inline_runner_active_.store(false, std::memory_order_release);
}

Result<bool>
QueryRunController::PublishDeferredReady(std::uint64_t generation,
                                         std::uint64_t wait_epoch) noexcept {
  if (generation != generation_ ||
      wait_epoch != wait_epoch_.load(std::memory_order_acquire) ||
      !deferred_ready_mode_.load(std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "deferred ready metadata is stale");
  }
  if (inline_runner_active_.load(std::memory_order_acquire) ||
      !operations_converged_.load(std::memory_order_acquire)) {
    return false;
  }
  if (timing_enabled_) {
    ready_timestamp_nanoseconds_.store(SteadyNowNanoseconds(),
                                       std::memory_order_relaxed);
  }
  RunState expected = RunState::kWaiting;
  if (!state_.compare_exchange_strong(expected, RunState::kQueued,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    ready_timestamp_nanoseconds_.store(0, std::memory_order_relaxed);
    return Status(StatusCode::kProtocolError,
                  "deferred ready observed query outside Waiting");
  }
  deferred_ready_mode_.store(false, std::memory_order_release);
  return true;
}

Result<bool> QueryRunController::OnOperationTerminal(
    QueryOperationState &operation, QueryOperationHandle handle,
    OperationTerminalOutcome outcome) noexcept {
  auto terminal = operation.TryMarkTerminal(handle, outcome);
  if (!terminal.ok()) {
    return terminal.status();
  }
  if (!terminal.value()) {
    return false;
  }
  return CompleteLaneOperation(handle.owner_lane, handle.query_generation,
                               handle.wait_epoch);
}

Result<bool>
QueryRunController::CompleteLaneOperation(std::uint32_t lane,
                                          std::uint64_t generation,
                                          std::uint64_t wait_epoch) noexcept {
  if (generation != generation_ || lane >= max_lanes_ ||
      wait_epoch != wait_epoch_.load(std::memory_order_acquire) ||
      lane_pending_[lane].epoch.load(std::memory_order_relaxed) != wait_epoch) {
    return Status(StatusCode::kProtocolError,
                  "operation completion generation/epoch/lane mismatch");
  }
  auto lane_operations =
      lane_pending_[lane].operations.load(std::memory_order_relaxed);
  while (lane_operations != 0) {
    if (lane_pending_[lane].operations.compare_exchange_weak(
            lane_operations, lane_operations - 1U, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      const auto completed_at = timing_enabled_ ? SteadyNowNanoseconds() : 0;
      if (timing_enabled_) {
        std::uint64_t first = 0;
        (void)first_operation_timestamp_nanoseconds_.compare_exchange_strong(
            first, completed_at, std::memory_order_relaxed,
            std::memory_order_relaxed);
      }
      // Metrics/debugging only: this aggregate may transiently lead the final
      // lane decrement and must not be used as a convergence barrier.
      remaining_operations_.fetch_sub(1, std::memory_order_relaxed);
      if (lane_operations != 1) {
        return false;
      }
      const auto remaining =
          remaining_lanes_.fetch_sub(1, std::memory_order_acq_rel);
      if (remaining == 0) {
        return Status(StatusCode::kProtocolError,
                      "lane completed after wait convergence");
      }
      if (remaining != 1) {
        return false;
      }
      operations_converged_.store(true, std::memory_order_release);
      const auto ready_at = completed_at;
      critical_ready_lane_.store(lane, std::memory_order_relaxed);
      operation_ready_timestamp_nanoseconds_.store(ready_at,
                                                   std::memory_order_relaxed);
      if (deferred_ready_mode_.load(std::memory_order_acquire)) {
        return true;
      }
      if (timing_enabled_) {
        ready_timestamp_nanoseconds_.store(ready_at, std::memory_order_relaxed);
      }
      RunState expected = RunState::kWaiting;
      if (!state_.compare_exchange_strong(expected, RunState::kQueued,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
        ready_timestamp_nanoseconds_.store(0, std::memory_order_relaxed);
        return Status(StatusCode::kProtocolError,
                      "last lane observed query outside Waiting");
      }
      return true;
    }
  }
  return Status(StatusCode::kProtocolError,
                "duplicate completion after lane pending reached zero");
}

Status QueryRunController::RequestFailure(std::uint64_t generation,
                                          TerminalOutcome outcome,
                                          StatusCode code) noexcept {
  if (generation != generation_ || outcome == TerminalOutcome::kNone ||
      code == StatusCode::kOk) {
    return Status(StatusCode::kInvalidArgument,
                  "query failure intent is invalid");
  }
  const auto encoded = static_cast<std::uint64_t>(outcome) |
                       (static_cast<std::uint64_t>(code) << 8U);
  std::uint64_t expected = 0;
  if (failure_intent_.compare_exchange_strong(expected, encoded,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
    cancel_generation_.store(generation, std::memory_order_release);
  }
  return Status::Ok();
}

StatusCode
QueryRunController::CancellationCode(std::uint64_t generation) const noexcept {
  if (cancel_generation_.load(std::memory_order_acquire) != generation) {
    return StatusCode::kOk;
  }
  return failure_code();
}

bool QueryRunController::ValidateReadyHandle(
    QueryReadyHandle handle) const noexcept {
  if (handle.query.generation != generation_ ||
      handle.wait_epoch != wait_epoch_.load(std::memory_order_acquire)) {
    return false;
  }
  assert(handle.owner_search == owner_search_);
  return true;
}

bool QueryRunController::ValidateRetryHandle(
    QueryRetryHandle handle) const noexcept {
  return handle.query.generation == generation_ &&
         handle.retry_epoch == retry_epoch_.load(std::memory_order_acquire);
}

Status QueryRunController::Park(std::uint32_t operation_count) noexcept {
  const LaneOperationCount lane{0, operation_count};
  auto waited = BeginWait(std::span(&lane, 1), generation_);
  return waited.ok() ? Status::Ok() : waited.status();
}

Result<bool>
QueryRunController::CompleteOne(std::uint64_t generation) noexcept {
  return CompleteLaneOperation(0, generation, wait_epoch());
}

Status QueryRunController::MarkCompleted() noexcept {
  RunState expected = RunState::kRunning;
  if (!state_.compare_exchange_strong(expected, RunState::kCompleted,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
    return Status(StatusCode::kProtocolError,
                  "query can only complete from Running");
  }
  return Status::Ok();
}

void QueryRunController::MarkFailed() noexcept {
  state_.store(RunState::kFailed, std::memory_order_release);
}

Status QueryOperationBinding::Prepare(QueryKey query,
                                      QueryRunController &controller,
                                      QueryOperationState &operation,
                                      QueryReadyPublisher &ready_publisher,
                                      QueryOperationHandle handle) noexcept {
  if (controller_ != nullptr || query.generation == 0 ||
      query.generation != handle.query_generation ||
      controller.generation() != query.generation || handle.wait_epoch == 0 ||
      handle.ticket == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "query operation binding metadata is invalid");
  }
  auto prepared = operation.Prepare(handle);
  if (!prepared.ok()) {
    return prepared;
  }
  query_ = query;
  controller_ = &controller;
  operation_ = &operation;
  ready_publisher_ = &ready_publisher;
  handle_ = handle;
  return Status::Ok();
}

Status QueryOperationBinding::MarkPublished() noexcept {
  if (controller_ == nullptr || operation_ == nullptr) {
    return Status(StatusCode::kProtocolError,
                  "query operation binding is not prepared");
  }
  return operation_->MarkPublished(handle_);
}

Status
QueryOperationBinding::Complete(OperationTerminalOutcome outcome) noexcept {
  if (controller_ == nullptr || operation_ == nullptr ||
      ready_publisher_ == nullptr) {
    return Status(StatusCode::kProtocolError,
                  "query operation binding is not prepared");
  }
  // Copy everything needed after the terminal transition. Once PublishReady()
  // succeeds the owner Search may finish and retire the query immediately.
  auto *const controller = controller_;
  auto *const operation = operation_;
  auto *const publisher = ready_publisher_;
  const auto handle = handle_;
  const auto query = query_;
  auto terminal = controller->OnOperationTerminal(*operation, handle, outcome);
  if (!terminal.ok()) {
    return terminal.status();
  }
  if (!terminal.value()) {
    return Status::Ok();
  }
  const QueryReadyHandle ready{query, handle.wait_epoch,
                               controller->ready_timestamp_nanoseconds(), 0,
                               controller->owner_search()};
  if (controller->ready_publication_deferred()) {
    return publisher->DeferReady(handle.owner_lane, ready, controller);
  }
  return publisher->PublishReady(handle.owner_lane, ready);
}

StatusCode QueryOperationBinding::CancellationCode() const noexcept {
  return controller_ == nullptr
             ? StatusCode::kProtocolError
             : controller_->CancellationCode(handle_.query_generation);
}

void QueryOperationBinding::Reset() noexcept {
  if (operation_ != nullptr) {
    operation_->Reset();
  }
  query_ = {};
  controller_ = nullptr;
  operation_ = nullptr;
  ready_publisher_ = nullptr;
  handle_ = {};
}

} // namespace hnsw_urma
