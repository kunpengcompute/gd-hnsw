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

#include "hnsw_urma/runtime/rpc/distance_oracle.hpp"
#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/core/log.hpp"
#include "hnsw_urma/runtime/scheduler.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>

namespace hnsw_urma {
namespace {
constexpr auto kActiveAuditInterval = std::chrono::microseconds(50);
constexpr ticket_t kTicketBlockSize = 256;
thread_local std::uint32_t g_distance_search_worker = UINT32_MAX;
std::uint64_t SteadyNowNanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void AddSingleWriterCounter(std::atomic<std::uint64_t> &counter,
                            std::uint64_t value) noexcept {
  counter.store(counter.load(std::memory_order_relaxed) + value,
                std::memory_order_relaxed);
}

OperationTerminalOutcome OutcomeFor(StatusCode status) noexcept {
  if (status == StatusCode::kOk) {
    return OperationTerminalOutcome::kComplete;
  }
  if (status == StatusCode::kTimeout) {
    return OperationTerminalOutcome::kTimedOut;
  }
  if (status == StatusCode::kCancelled) {
    return OperationTerminalOutcome::kCancelled;
  }
  return OperationTerminalOutcome::kFailed;
}

} // namespace
void SetCurrentDistanceSearchWorker(std::uint32_t worker) noexcept {
  g_distance_search_worker = worker;
}

std::uint32_t CurrentDistanceSearchWorker() noexcept {
  return g_distance_search_worker;
}

Result<std::uint64_t>
RpcDistanceOracle::RequiredBytes(std::uint32_t searches, std::uint32_t owners,
                                 std::uint32_t slots_per_search,
                                 std::uint32_t max_ids, std::uint32_t queries,
                                 std::uint32_t shards, std::uint32_t ranks) {
  auto total = CheckedMultiply<std::uint64_t>(searches, slots_per_search);
  if (!total.ok() || total.value() == 0 ||
      total.value() >= (std::uint64_t{1} << 31U))
    return Status(StatusCode::kOutOfRange,
                  "RPC slots exceed Hybrid ticket encoding");
  std::uint64_t bytes = sizeof(RpcDistanceOracle);
  const auto add = [&](std::uint64_t count, std::uint64_t size) -> Status {
    auto product = CheckedMultiply(count, size);
    if (!product.ok())
      return product.status();
    auto sum = CheckedAdd(bytes, product.value());
    if (!sum.ok())
      return sum.status();
    bytes = sum.value();
    return Status::Ok();
  };
  for (const auto &status :
       {add(total.value(), sizeof(Operation)),
        add(total.value() * max_ids, sizeof(node_id_t) + sizeof(float)),
        add(total.value(), sizeof(std::uint32_t)),
        add(searches, sizeof(SearchPool)), add(owners, sizeof(OwnerState)),
        add(total.value() * owners, sizeof(OperationEvent)),
        add(queries, sizeof(QueryWireLeaseSlot)),
        add(static_cast<std::uint64_t>(shards) + ranks,
            sizeof(std::uint32_t))}) {
    if (!status.ok())
      return status;
  }
  return bytes;
}

Result<std::unique_ptr<RpcDistanceOracle>> RpcDistanceOracle::Create(
    std::uint32_t search_workers, std::uint32_t rpc_threads,
    std::uint32_t submit_depth_per_cell, std::uint32_t slots_per_search,
    std::uint32_t max_ids_per_task, std::uint32_t dim, std::uint32_t local_rank,
    std::span<const std::uint32_t> shard_to_rank,
    std::span<const std::uint32_t> rpc_owner_by_rank,
    std::uint32_t ready_lane_base, SchedulerQueues &events, bool use_dot_norm) {
  if (search_workers == 0 || rpc_threads == 0 || slots_per_search == 0 ||
      submit_depth_per_cell == 0 || max_ids_per_task == 0 || dim == 0 ||
      local_rank >= rpc_owner_by_rank.size() || shard_to_rank.empty() ||
      ready_lane_base > UINT32_MAX - rpc_threads ||
      events.event_ring_count() != search_workers) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC pool dimensions are invalid");
  }
  const auto capacity =
      CheckedAdd(events.max_active_queries(), slots_per_search);
  if (!capacity.ok() ||
      events.event_capacity() < std::max(2U, capacity.value()))
    return Status(StatusCode::kInvalidArgument,
                  "SearchEvents cannot cover RPC pool lifetime");
  for (auto rank : shard_to_rank) {
    if (rank >= rpc_owner_by_rank.size())
      return Status(StatusCode::kInvalidArgument, "invalid shard rank");
  }
  auto bytes =
      RequiredBytes(search_workers, rpc_threads, slots_per_search,
                    max_ids_per_task, events.max_active_queries(),
                    static_cast<std::uint32_t>(shard_to_rank.size()),
                    static_cast<std::uint32_t>(rpc_owner_by_rank.size()));
  if (!bytes.ok() || bytes.value() > SIZE_MAX)
    return bytes.ok() ? Status(StatusCode::kOutOfRange,
                               "RPC pool exceeds address space")
                      : bytes.status();
  auto submissions = RpcSubmitMatrix::Create(
      search_workers, rpc_threads, submit_depth_per_cell, rpc_owner_by_rank);
  if (!submissions.ok())
    return submissions.status();
  try {
    auto oracle = std::unique_ptr<RpcDistanceOracle>(new RpcDistanceOracle());
    oracle->search_workers_ = search_workers;
    oracle->rpc_threads_ = rpc_threads;
    oracle->slots_per_search_ = slots_per_search;
    oracle->operation_count_ = search_workers * slots_per_search;
    oracle->max_ids_per_task_ = max_ids_per_task;
    oracle->dim_ = dim;
    oracle->local_rank_ = local_rank;
    oracle->ready_lane_base_ = ready_lane_base;
    oracle->max_active_queries_ = events.max_active_queries();
    oracle->events_ = &events;
    oracle->use_dot_norm_ = use_dot_norm;
    oracle->shard_to_rank_.assign(shard_to_rank.begin(), shard_to_rank.end());
    oracle->rpc_owner_by_rank_.assign(rpc_owner_by_rank.begin(),
                                      rpc_owner_by_rank.end());
    oracle->operations_ =
        std::make_unique<Operation[]>(oracle->operation_count_);
    const auto values =
        static_cast<std::size_t>(oracle->operation_count_) * max_ids_per_task;
    oracle->id_storage_ = std::make_unique<node_id_t[]>(values);
    oracle->result_storage_ = std::make_unique<float[]>(values);
    oracle->search_pools_ = std::make_unique<SearchPool[]>(search_workers);
    for (std::uint32_t worker = 0; worker < search_workers; ++worker) {
      auto &free = oracle->search_pools_[worker].free_slots;
      free.reserve(slots_per_search);
      for (std::uint32_t slot = slots_per_search; slot > 0; --slot)
        free.push_back(worker * slots_per_search + slot - 1U);
    }
    oracle->owner_states_ = std::make_unique<OwnerState[]>(rpc_threads);
    for (std::uint32_t owner = 0; owner < rpc_threads; ++owner)
      oracle->owner_states_[owner].active.reserve(oracle->operation_count_);
    oracle->query_wire_leases_ =
        std::make_unique<QueryWireLeaseSlot[]>(events.max_active_queries());
    oracle->submissions_ = std::move(submissions).value();
    return oracle;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate Search RPC pools");
  }
}

Result<ticket_t> RpcDistanceOracle::NextTicket(SearchPool &shard) noexcept {
  if (shard.next_ticket != shard.ticket_end) {
    return shard.next_ticket++;
  }
  auto block_begin = next_ticket_.load(std::memory_order_relaxed);
  while (block_begin != 0 &&
         block_begin < std::numeric_limits<ticket_t>::max()) {
    const auto block_size = std::min(
        kTicketBlockSize, std::numeric_limits<ticket_t>::max() - block_begin);
    const auto block_end = block_begin + block_size;
    if (next_ticket_.compare_exchange_weak(block_begin, block_end,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
      shard.next_ticket = block_begin + 1U;
      shard.ticket_end = block_end;
      shard.ticket_block_refills.store(
          shard.ticket_block_refills.load(std::memory_order_relaxed) + 1U,
          std::memory_order_relaxed);
      return block_begin;
    }
  }
  return Status(StatusCode::kResourceExhausted,
                "distance task ticket space exhausted");
}

Result<RpcDistanceOracle::Operation *> RpcDistanceOracle::PrepareOperation(
    const QueryVectorLease &lease, std::uint64_t wait_epoch,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, std::chrono::steady_clock::time_point deadline,
    DistanceRequestOptions options) {
  const auto query = lease.query;
  const auto query_vector = lease.values();
  const auto search_worker = CurrentDistanceSearchWorker();
  if (search_worker >= search_workers_ || query.generation == 0 ||
      query.query_slot >= max_active_queries_ || wait_epoch == 0 ||
      query_vector.size() != dim_ || ids.empty() ||
      ids.size() != output.size() || ids.size() > max_ids_per_task_ ||
      owner_shard >= shard_to_rank_.size() ||
      (options.upper_distance.has_value() &&
       !std::isfinite(*options.upper_distance))) {
    return Status(StatusCode::kInvalidArgument,
                  "distance submission shape is invalid");
  }
  const auto destination = shard_to_rank_[owner_shard];
  if (destination == local_rank_) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC distance cannot target the local rank");
  }
  auto owner = submissions_->OwnerForRank(destination);
  if (!owner.ok()) {
    return owner.status();
  }
  auto &pool = search_pools_[search_worker];
  if (pool.free_slots.empty()) {
    AddSingleWriterCounter(pool.pool_backpressure, 1);
    return Status(StatusCode::kResourceExhausted, "Search RPC pool is full");
  }
  auto reserved = submissions_->Reserve(search_worker, destination);
  if (!reserved.ok()) {
    AddSingleWriterCounter(pool.submit_backpressure, 1);
    return reserved;
  }
  auto ticket = NextTicket(pool);
  if (!ticket.ok()) {
    (void)submissions_->CancelReservation(search_worker, destination);
    return ticket.status();
  }
  const auto claimed_index = pool.free_slots.back();
  pool.free_slots.pop_back();
  auto *claimed = &operations_[claimed_index];
  if (claimed->state.load(std::memory_order_relaxed) != OperationState::kFree) {
    LOG_ERROR("RpcDistanceOracle::PrepareOperation claimed slot {} not free",
              claimed_index);
    return Status(StatusCode::kInternalError,
                  "RPC distance operation slot was not free");
  }
  const auto in_use = pool.slots_in_use.load(std::memory_order_relaxed) + 1U;
  pool.slots_in_use.store(in_use, std::memory_order_relaxed);
  pool.slots_high_watermark.store(
      std::max(in_use,
               pool.slots_high_watermark.load(std::memory_order_relaxed)),
      std::memory_order_relaxed);
  auto &operation = *claimed;
  auto ids_storage = std::span<node_id_t>(
      id_storage_.get() +
          static_cast<std::size_t>(claimed_index) * max_ids_per_task_,
      ids.size());
  auto result_storage = std::span<float>(
      result_storage_.get() +
          static_cast<std::size_t>(claimed_index) * max_ids_per_task_,
      ids.size());
  std::copy(ids.begin(), ids.end(), ids_storage.begin());
  operation.ticket = {static_cast<ticket_t>(claimed_index) + 1U,
                      ticket.value()};
  operation.search_worker = search_worker;
  operation.rpc_owner = owner.value();
  operation.destination_rank = destination;
  operation.deadline = deadline;
  operation.user_output = output;
  operation.status = StatusCode::kOk;
  operation.consumed = false;
  operation.reclaim_received = false;
  operation.reclaim_required = false;
  operation.exception_observed = false;
  operation.audit_position = UINT32_MAX;
  operation.query_lease = lease;
  auto wire_lease = AcquireQueryWireLease(query);
  if (!wire_lease.ok()) {
    Reset(operation);
    (void)submissions_->CancelReservation(search_worker, destination);
    return wire_lease;
  }
  operation.wire_lease_held = true;
  const auto request_flags =
      kBatchHasQueryVector | (use_dot_norm_ ? kBatchUseDotNorm : 0U) |
      (options.upper_distance.has_value() ? kBatchHasUpperDistance : 0U);
  auto prepared = operation.task.Prepare(
      ticket.value(), query, wait_epoch, destination, owner_shard, ids_storage,
      operation.query_lease.values(), result_storage, request_flags,
      options.upper_distance.value_or(0.0F), deadline);
  if (!prepared.ok()) {
    Reset(operation);
    (void)submissions_->CancelReservation(search_worker, destination);
    return prepared;
  }
  operation.task.operation_slot = claimed_index;
  operation.state.store(OperationState::kPrepared, std::memory_order_relaxed);
  return &operation;
}

Result<PreparedDistanceOperation> RpcDistanceOracle::PrepareForWait(
    QueryWaitPreparation preparation, std::span<const float> query_vector,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  try {
    auto storage = std::make_shared<const std::vector<float>>(
        query_vector.begin(), query_vector.end());
    return PrepareLeaseForWait(preparation,
                               {preparation.query, std::move(storage)},
                               owner_shard, ids, output, options);
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to snapshot query vector");
  }
}

Result<PreparedDistanceOperation> RpcDistanceOracle::PrepareLeaseForWait(
    QueryWaitPreparation preparation, const QueryVectorLease &lease,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  if (preparation.controller == nullptr ||
      preparation.ready_publisher == nullptr ||
      preparation.operation_deadline ==
          std::chrono::steady_clock::time_point{} ||
      preparation.controller->generation() != preparation.query.generation ||
      lease.query != preparation.query || !lease.valid()) {
    return Status(StatusCode::kInvalidArgument,
                  "prepared distance wait metadata is invalid");
  }
  if (preparation.controller->owner_search() != CurrentDistanceSearchWorker()) {
    LOG_ERROR("RpcDistanceOracle::PrepareLeaseForWait owner_search mismatch "
              "controller={} worker={}",
              preparation.controller->owner_search(),
              CurrentDistanceSearchWorker());
    return Status(StatusCode::kInvalidArgument,
                  "prepared distance wait owner search mismatch");
  }
  auto operation =
      PrepareOperation(lease, preparation.wait_epoch, owner_shard, ids, output,
                       preparation.operation_deadline, options);
  if (!operation.ok()) {
    return operation.status();
  }
  auto *value = operation.value();
  const auto slot = static_cast<std::uint32_t>(value->ticket.value - 1U);
  auto bound = value->binding.Prepare(
      preparation.query, *preparation.controller, value->query_operation,
      *preparation.ready_publisher,
      {slot, ready_lane_base_ + value->rpc_owner, preparation.query.generation,
       preparation.wait_epoch, value->task.ticket});
  if (!bound.ok()) {
    (void)submissions_->CancelReservation(value->search_worker,
                                          value->destination_rank);
    Reset(*value);
    return bound;
  }
  value->state.store(OperationState::kPrepared, std::memory_order_release);
  return PreparedDistanceOperation{value->ticket,
                                   ready_lane_base_ + value->rpc_owner};
}

Result<RpcDistanceOracle::Operation *>
RpcDistanceOracle::Find(DistanceTicket ticket) noexcept {
  if (ticket.value == 0 || ticket.value > operation_count_ ||
      ticket.generation == 0 ||
      CurrentDistanceSearchWorker() >= search_workers_ ||
      (ticket.value - 1U) / slots_per_search_ !=
          CurrentDistanceSearchWorker()) {
    return Status(StatusCode::kProtocolError,
                  "distance ticket is outside the fixed table");
  }
  auto &operation = operations_[ticket.value - 1U];
  if (operation.ticket != ticket ||
      operation.state.load(std::memory_order_acquire) ==
          OperationState::kFree) {
    return Status(StatusCode::kProtocolError, "distance ticket is stale");
  }
  return &operation;
}

Status RpcDistanceOracle::PublishPrepared(DistanceTicket ticket) {
  auto found = Find(ticket);
  if (!found.ok())
    return found.status();
  auto &operation = *found.value();
  if (operation.state.load(std::memory_order_relaxed) !=
      OperationState::kPrepared)
    return Status(StatusCode::kProtocolError, "RPC operation is not prepared");
  auto bound = operation.binding.MarkPublished();
  if (!bound.ok())
    return bound;
  operation.state.store(OperationState::kActive, std::memory_order_relaxed);
  AddSingleWriterCounter(search_pools_[operation.search_worker].submitted, 1);
  // Publication is the last access: even an immediately rejected submission
  // can publish Ready while this stack is returning.
  return submissions_->PublishReserved(operation.search_worker,
                                       &operation.task);
}

Status RpcDistanceOracle::CancelPrepared(DistanceTicket ticket) {
  auto found = Find(ticket);
  if (!found.ok())
    return found.status();
  auto &operation = *found.value();
  if (operation.state.load(std::memory_order_relaxed) !=
      OperationState::kPrepared)
    return Status(StatusCode::kProtocolError,
                  "cannot rollback a published RPC operation");
  auto cancelled = submissions_->CancelReservation(operation.search_worker,
                                                   operation.destination_rank);
  if (!cancelled.ok())
    return cancelled;
  Reset(operation);
  return Status::Ok();
}

void RpcDistanceOracle::ObserveException(Operation &operation) noexcept {
  if (!operation.reclaim_required || operation.exception_observed)
    return;
  operation.exception_observed = true;
  auto &pool = search_pools_[operation.search_worker];
  const auto retained =
      pool.exceptional_slots_in_use.load(std::memory_order_relaxed) + 1U;
  pool.exceptional_slots_in_use.store(retained, std::memory_order_relaxed);
  pool.exceptional_slots_high_watermark.store(
      std::max(retained, pool.exceptional_slots_high_watermark.load(
                             std::memory_order_relaxed)),
      std::memory_order_relaxed);
}

Status RpcDistanceOracle::Finish(DistanceTicket ticket) {
  auto found = Find(ticket);
  if (!found.ok())
    return found.status();
  auto &operation = *found.value();
  if (operation.state.load(std::memory_order_acquire) !=
          OperationState::kTerminal ||
      operation.consumed)
    return Status(StatusCode::kProtocolError, "RPC result is not consumable");
  Status result;
  if (operation.status == StatusCode::kOk) {
    // Only successful wire completion writes output; delayed failure responses
    // can touch the stable backing but never this Query-owned destination.
    const auto *values =
        result_storage_.get() + (ticket.value - 1U) * max_ids_per_task_;
    std::copy_n(values, operation.user_output.size(),
                operation.user_output.begin());
  } else {
    result = Status(operation.status, "RPC distance task failed");
    result.AddContext("rpc_owner", std::to_string(operation.rpc_owner))
        .AddContext("destination_rank",
                    std::to_string(operation.destination_rank))
        .AddContext("task_ticket", std::to_string(ticket.generation))
        .AddContext("request_sequence",
                    std::to_string(operation.terminal_sequence))
        .AddContext("wire_state", std::to_string(static_cast<unsigned>(
                                      operation.terminal_wire_state)))
        .AddContext("reclaim_required",
                    operation.reclaim_required ? "true" : "false");
  }
  ObserveException(operation);
  operation.consumed = true;
  operation.user_output = {};
  operation.binding.Reset();
  if (!operation.reclaim_required || operation.reclaim_received)
    Reset(operation);
  return result;
}

Status RpcDistanceOracle::HandleReclaim(std::uint32_t search,
                                        RpcReclaimHandle handle) {
  if (search != CurrentDistanceSearchWorker() || handle.owner_search != search)
    return Status(StatusCode::kProtocolError,
                  "RPC Reclaim has wrong Search owner");
  auto found =
      Find({static_cast<ticket_t>(handle.operation_slot) + 1U, handle.ticket});
  if (!found.ok())
    return found.status();
  auto &operation = *found.value();
  if (operation.state.load(std::memory_order_acquire) !=
          OperationState::kTerminal ||
      !operation.reclaim_required || operation.reclaim_received)
    return Status(StatusCode::kProtocolError,
                  "duplicate or premature RPC Reclaim");
  ObserveException(operation);
  operation.reclaim_received = true;
  if (operation.consumed)
    Reset(operation);
  return Status::Ok();
}

StatusCode RpcDistanceOracle::ControlStatus(
    const Operation &operation,
    std::chrono::steady_clock::time_point now) const noexcept {
  const auto cancellation = operation.binding.CancellationCode();
  return cancellation != StatusCode::kOk ? cancellation
         : now >= operation.deadline     ? StatusCode::kTimeout
                                         : StatusCode::kOk;
}

StatusCode
RpcDistanceOracle::TaskControlStatus(std::uint32_t owner,
                                     const RpcClientTask &task) const noexcept {
  if (task.operation_slot >= operation_count_)
    return StatusCode::kProtocolError;
  const auto &operation = operations_[task.operation_slot];
  if (operation.rpc_owner != owner ||
      operation.ticket.generation != task.ticket)
    return StatusCode::kProtocolError;
  return ControlStatus(operation, std::chrono::steady_clock::now());
}

Result<bool> RpcDistanceOracle::OnRpcTaskOwned(std::uint32_t owner,
                                               RpcClientTask &task) noexcept {
  if (owner >= rpc_threads_ || task.operation_slot >= operation_count_)
    return Status(StatusCode::kProtocolError, "invalid RPC ownership callback");
  auto &operation = operations_[task.operation_slot];
  if (operation.rpc_owner != owner ||
      operation.ticket.generation != task.ticket ||
      operation.state.load(std::memory_order_relaxed) !=
          OperationState::kActive)
    return Status(StatusCode::kProtocolError, "stale RPC ownership callback");
  const auto status =
      ControlStatus(operation, std::chrono::steady_clock::now());
  if (status != StatusCode::kOk) {
    task.state.store(RpcClientTaskState::kTerminal, std::memory_order_relaxed);
    auto completed = CompleteOperation(operation, status, false);
    if (!completed.ok())
      return completed;
    return false; // Already handed off; DrainSubmissions must not touch task.
  }
  auto &active = owner_states_[owner].active;
  if (active.size() == active.capacity())
    return Status(StatusCode::kInternalError, "RPC audit capacity violated");
  operation.audit_position = static_cast<std::uint32_t>(active.size());
  active.push_back({task.operation_slot, task.ticket});
  return true;
}

void RpcDistanceOracle::RemoveOwnerActive(Operation &operation) noexcept {
  auto &owner = owner_states_[operation.rpc_owner];
  const auto position = operation.audit_position;
  if (position >= owner.active.size()) {
    LOG_ERROR(
        "RpcDistanceOracle::RemoveOwnerActive audit position {} >= active {}",
        position, owner.active.size());
    return;
  }
  const auto last = owner.active.back();
  owner.active[position] = last;
  operations_[last.slot].audit_position = position;
  owner.active.pop_back();
  operation.audit_position = UINT32_MAX;
  // Preserve a bounded rotating audit even when another task completes.
  if (position < owner.cursor && owner.cursor != 0)
    --owner.cursor;
  if (owner.cursor >= owner.active.size())
    owner.cursor = 0;
}

Status RpcDistanceOracle::CompleteOperation(Operation &operation,
                                            StatusCode status,
                                            bool reclaim_required) noexcept {
  operation.status = status;
  operation.reclaim_required = reclaim_required;
  operation.terminal_wire_state =
      operation.task.state.load(std::memory_order_relaxed);
  operation.terminal_sequence = operation.task.request_sequence;
  operation.terminal_nanoseconds = SteadyNowNanoseconds();
  auto &owner = owner_states_[operation.rpc_owner];
  if (reclaim_required)
    AddSingleWriterCounter(owner.detached, 1);
  if (status == StatusCode::kOk)
    AddSingleWriterCounter(owner.completed, 1);
  else if (status == StatusCode::kTimeout)
    AddSingleWriterCounter(owner.timed_out, 1);
  else if (status == StatusCode::kCancelled)
    AddSingleWriterCounter(owner.cancelled, 1);
  else
    AddSingleWriterCounter(owner.failed, 1);
  if (operation.task.response_received_nanoseconds != 0) {
    AddSingleWriterCounter(owner.response_to_stage_terminal_nanoseconds,
                           operation.terminal_nanoseconds -
                               operation.task.response_received_nanoseconds);
    AddSingleWriterCounter(owner.response_to_stage_terminal_samples, 1);
  }
  operation.state.store(OperationState::kTerminal, std::memory_order_release);
  // Complete copies its controller/operation/publisher before convergence.
  // Any other lane can publish Ready now: absolutely no subsequent slot reads.
  return operation.binding.Complete(OutcomeFor(status));
}

Status RpcDistanceOracle::OnRpcTaskTerminal(std::uint32_t owner,
                                            RpcClientTask &task) noexcept {
  if (owner >= rpc_threads_ || task.operation_slot >= operation_count_)
    return Status(StatusCode::kProtocolError, "invalid RPC terminal callback");
  auto &operation = operations_[task.operation_slot];
  if (operation.rpc_owner != owner ||
      operation.ticket.generation != task.ticket)
    return Status(StatusCode::kProtocolError, "stale RPC terminal callback");
  if (operation.state.load(std::memory_order_relaxed) ==
      OperationState::kTerminal) {
    if (!operation.reclaim_required)
      return Status(StatusCode::kProtocolError,
                    "duplicate RPC terminal callback");
    const RpcReclaimHandle reclaim{operation.search_worker, task.operation_slot,
                                   task.ticket};
    return events_->PublishRpcReclaim(ready_lane_base_ + owner, reclaim);
  }
  if (operation.state.load(std::memory_order_relaxed) !=
      OperationState::kActive)
    return Status(StatusCode::kProtocolError,
                  "RPC terminal operation is not active");
  RemoveOwnerActive(operation);
  const auto control =
      ControlStatus(operation, std::chrono::steady_clock::now());
  return CompleteOperation(
      operation, control == StatusCode::kOk ? task.status : control, false);
}

Result<std::uint32_t> RpcDistanceOracle::AuditOwnerControl(
    std::uint32_t owner_index, RpcClientOwner &client, std::uint32_t budget,
    std::chrono::steady_clock::time_point now) {
  if (owner_index >= rpc_threads_)
    return Status(StatusCode::kInvalidArgument, "invalid RPC audit owner");
  auto &owner = owner_states_[owner_index];
  if (now < owner.next_audit || budget == 0)
    return 0U;
  owner.next_audit = now + kActiveAuditInterval;
  AddSingleWriterCounter(owner.active_audit_passes, 1);
  const auto count = std::min<std::size_t>(owner.active.size(), budget);
  std::uint32_t completed = 0;
  for (std::size_t scanned = 0; scanned < count && !owner.active.empty();
       ++scanned) {
    owner.cursor %= owner.active.size();
    const auto event = owner.active[owner.cursor];
    auto &operation = operations_[event.slot];
    if (operation.ticket.generation != event.ticket) {
      LOG_ERROR("RpcDistanceOracle::AuditOwnerControl generation mismatch "
                "operation={} event={}",
                operation.ticket.generation, event.ticket);
      return Status(StatusCode::kInternalError,
                    "RPC distance audit generation mismatch");
    }
    AddSingleWriterCounter(owner.active_audit_operations, 1);
    const auto status = ControlStatus(operation, now);
    if (status == StatusCode::kOk) {
      ++owner.cursor;
      continue;
    }
    auto removed = client.CancelUnsent(operation.task);
    if (!removed.ok())
      return removed.status();
    RemoveOwnerActive(operation);
    auto terminal = CompleteOperation(operation, status, !removed.value());
    if (!terminal.ok())
      return terminal;
    ++completed;
  }
  return completed;
}

Status RpcDistanceOracle::AcquireQueryWireLease(QueryKey query) noexcept {
  if (query.generation == 0 || query.query_slot >= max_active_queries_) {
    return Status(StatusCode::kProtocolError,
                  "query wire lease key is invalid");
  }
  auto &slot = query_wire_leases_[query.query_slot];
  const auto refs = slot.refs.load(std::memory_order_acquire);
  if (refs == 0) {
    slot.generation.store(query.generation, std::memory_order_release);
  } else if (slot.generation.load(std::memory_order_acquire) !=
             query.generation) {
    return Status(StatusCode::kProtocolError,
                  "query slot generation overlaps active wire work");
  }
  if (slot.refs.fetch_add(1, std::memory_order_acq_rel) ==
      std::numeric_limits<std::uint32_t>::max()) {
    slot.refs.fetch_sub(1, std::memory_order_release);
    return Status(StatusCode::kResourceExhausted,
                  "query wire lease reference count overflowed");
  }
  return Status::Ok();
}

void RpcDistanceOracle::ReleaseQueryWireLease(QueryKey query) noexcept {
  if (query.query_slot >= max_active_queries_)
    return;
  auto &slot = query_wire_leases_[query.query_slot];
  if (slot.generation.load(std::memory_order_acquire) != query.generation) {
    std::abort();
  }
  const auto previous = slot.refs.fetch_sub(1, std::memory_order_acq_rel);
  if (previous == 0)
    std::abort();
}

bool RpcDistanceOracle::QueryWireQuiesced(QueryKey query) const noexcept {
  if (query.generation == 0 || query.query_slot >= max_active_queries_) {
    return false;
  }
  const auto &slot = query_wire_leases_[query.query_slot];
  const auto refs = slot.refs.load(std::memory_order_acquire);
  return refs == 0;
}

void RpcDistanceOracle::Reset(Operation &operation) noexcept {
  auto &pool = search_pools_[operation.search_worker];
  const auto slot = static_cast<std::uint32_t>(operation.ticket.value - 1U);
  if (operation.exception_observed) {
    pool.exceptional_slots_in_use.fetch_sub(1, std::memory_order_relaxed);
    AddSingleWriterCounter(pool.exceptional_retained_nanoseconds,
                           SteadyNowNanoseconds() -
                               operation.terminal_nanoseconds);
    AddSingleWriterCounter(pool.exceptional_reclaimed, 1);
  }
  if (operation.wire_lease_held) {
    ReleaseQueryWireLease(operation.query_lease.query);
    operation.wire_lease_held = false;
  }
  operation.binding.Reset();
  operation.task.Reset();
  operation.query_lease = {};
  operation.user_output = {};
  operation.ticket = {};
  operation.state.store(OperationState::kFree, std::memory_order_release);
  pool.free_slots.push_back(slot);
  pool.slots_in_use.fetch_sub(1, std::memory_order_release);
}

bool RpcDistanceOracle::SearchDrained(std::uint32_t search) const noexcept {
  return search < search_workers_ && search_pools_[search].slots_in_use.load(
                                         std::memory_order_acquire) == 0;
}

bool RpcDistanceOracle::drained() const noexcept {
  // Slot ownership, not a transient empty queue read, is the drain barrier.
  for (std::uint32_t search = 0; search < search_workers_; ++search)
    if (!SearchDrained(search))
      return false;
  for (std::uint32_t slot = 0; slot < max_active_queries_; ++slot)
    if (query_wire_leases_[slot].refs.load(std::memory_order_acquire) != 0)
      return false;
  return true;
}

Status RpcDistanceOracle::ResetProfile() noexcept {
  if (!drained())
    return Status(StatusCode::kProtocolError,
                  "cannot reset live RPC pool metrics");
  for (std::uint32_t search = 0; search < search_workers_; ++search) {
    auto &pool = search_pools_[search];
    pool.submitted.store(0, std::memory_order_relaxed);
    pool.slots_in_use.store(0, std::memory_order_relaxed);
    pool.slots_high_watermark.store(0, std::memory_order_relaxed);
    pool.pool_backpressure.store(0, std::memory_order_relaxed);
    pool.submit_backpressure.store(0, std::memory_order_relaxed);
    pool.ticket_block_refills.store(0, std::memory_order_relaxed);
    pool.exceptional_slots_in_use.store(0, std::memory_order_relaxed);
    pool.exceptional_slots_high_watermark.store(0, std::memory_order_relaxed);
    pool.exceptional_retained_nanoseconds.store(0, std::memory_order_relaxed);
    pool.exceptional_reclaimed.store(0, std::memory_order_relaxed);
  }
  for (std::uint32_t owner = 0; owner < rpc_threads_; ++owner) {
    auto &state = owner_states_[owner];
    state.completed.store(0, std::memory_order_relaxed);
    state.timed_out.store(0, std::memory_order_relaxed);
    state.cancelled.store(0, std::memory_order_relaxed);
    state.failed.store(0, std::memory_order_relaxed);
    state.detached.store(0, std::memory_order_relaxed);
    state.active_audit_passes.store(0, std::memory_order_relaxed);
    state.active_audit_operations.store(0, std::memory_order_relaxed);
    state.response_to_stage_terminal_nanoseconds.store(
        0, std::memory_order_relaxed);
    state.response_to_stage_terminal_samples.store(0,
                                                   std::memory_order_relaxed);
  }
  return Status::Ok();
}

RpcDistanceProfile RpcDistanceOracle::profile() const {
  RpcDistanceProfile result;
  for (std::uint32_t search = 0; search < search_workers_; ++search) {
    const auto &pool = search_pools_[search];
    result.slots_in_use_per_search.push_back(
        pool.slots_in_use.load(std::memory_order_relaxed));
    result.slots_high_watermark_per_search.push_back(
        pool.slots_high_watermark.load(std::memory_order_relaxed));
    result.pool_backpressure_per_search.push_back(
        pool.pool_backpressure.load(std::memory_order_relaxed));
    result.submit_backpressure_per_search.push_back(
        pool.submit_backpressure.load(std::memory_order_relaxed));
    result.submitted += pool.submitted.load(std::memory_order_relaxed);
    result.slots_in_use += pool.slots_in_use.load(std::memory_order_relaxed);
    result.slots_high_watermark +=
        pool.slots_high_watermark.load(std::memory_order_relaxed);
    result.pool_backpressure +=
        pool.pool_backpressure.load(std::memory_order_relaxed);
    result.submit_backpressure +=
        pool.submit_backpressure.load(std::memory_order_relaxed);
    result.ticket_block_refills +=
        pool.ticket_block_refills.load(std::memory_order_relaxed);
    result.exceptional_slots_in_use +=
        pool.exceptional_slots_in_use.load(std::memory_order_relaxed);
    result.exceptional_slots_high_watermark +=
        pool.exceptional_slots_high_watermark.load(std::memory_order_relaxed);
    result.exceptional_retained_nanoseconds +=
        pool.exceptional_retained_nanoseconds.load(std::memory_order_relaxed);
    result.exceptional_reclaimed +=
        pool.exceptional_reclaimed.load(std::memory_order_relaxed);
  }
  for (std::uint32_t owner = 0; owner < rpc_threads_; ++owner) {
    const auto &state = owner_states_[owner];
    result.audit_operations_per_owner.push_back(
        state.active_audit_operations.load(std::memory_order_relaxed));
    result.completed += state.completed.load(std::memory_order_relaxed);
    result.timed_out += state.timed_out.load(std::memory_order_relaxed);
    result.cancelled += state.cancelled.load(std::memory_order_relaxed);
    result.failed += state.failed.load(std::memory_order_relaxed);
    result.detached += state.detached.load(std::memory_order_relaxed);
    result.active_audit_passes +=
        state.active_audit_passes.load(std::memory_order_relaxed);
    result.active_audit_operations +=
        state.active_audit_operations.load(std::memory_order_relaxed);
    result.response_to_stage_terminal_nanoseconds +=
        state.response_to_stage_terminal_nanoseconds.load(
            std::memory_order_relaxed);
    result.response_to_stage_terminal_samples +=
        state.response_to_stage_terminal_samples.load(
            std::memory_order_relaxed);
  }
  return result;
}
} // namespace hnsw_urma
