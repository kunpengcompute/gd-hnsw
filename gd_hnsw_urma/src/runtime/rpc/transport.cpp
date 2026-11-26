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

#include "hnsw_urma/runtime/rpc/transport.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>

namespace hnsw_urma {
namespace {

bool IsBackpressure(const Status &status) noexcept {
  return status.code() == StatusCode::kResourceExhausted;
}

std::uint64_t
ElapsedNanoseconds(std::chrono::steady_clock::time_point begin,
                   std::chrono::steady_clock::time_point end) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

} // namespace

Result<std::unique_ptr<RpcTransportOwner>> RpcTransportOwner::Create(
    RpcTransportConfig config, std::span<const std::uint32_t> owned_peers,
    LanePostEngine &engine, DmaVisibilityDomain &visibility,
    RpcMailboxMemory &memory, RpcClientOwner &client,
    RpcServerRuntime &server) {
  const auto &mailbox = memory.config();
  if (config.owner_thread >= mailbox.rpc_threads ||
      config.local_rank >= mailbox.world_size ||
      config.request_batch_max_tasks == 0 ||
      config.request_batch_max_bytes == 0 ||
      config.request_batch_max_bytes > mailbox.request_capacity ||
      config.poll_batch == 0 || config.phase_budget == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC transport configuration is invalid");
  }
  try {
    auto owner = std::unique_ptr<RpcTransportOwner>(new RpcTransportOwner());
    owner->config_ = config;
    owner->owned_peers_.assign(owned_peers.begin(), owned_peers.end());
    owner->peers_.resize(mailbox.world_size);
    std::vector<bool> seen(mailbox.world_size, false);
    for (const auto peer : owned_peers) {
      if (peer >= mailbox.world_size || peer == config.local_rank ||
          seen[peer]) {
        return Status(StatusCode::kInvalidArgument,
                      "RPC transport owned peer set is invalid");
      }
      seen[peer] = true;
      owner->peers_[peer].request_slots.resize(mailbox.ring_depth);
    }
    owner->engine_ = &engine;
    owner->visibility_ = &visibility;
    owner->memory_ = &memory;
    owner->client_ = &client;
    owner->server_ = &server;
    owner->request_staging_busy_.assign(
        mailbox.request_staging_slots_per_thread, false);
    const auto result_slots =
        static_cast<std::uint64_t>(owned_peers.size()) * mailbox.ring_depth;
    const auto tx_count =
        static_cast<std::uint64_t>(mailbox.request_staging_slots_per_thread) +
        result_slots;
    if (tx_count == 0 || tx_count > std::numeric_limits<std::size_t>::max()) {
      return Status(StatusCode::kOutOfRange,
                    "RPC transport TX slot count overflows");
    }
    owner->tx_slots_.resize(static_cast<std::size_t>(tx_count));
    owner->completion_scratch_.resize(config.poll_batch);
    owner->active_peer_scratch_.reserve(owned_peers.size());
    return owner;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate RPC transport owner");
  }
}

Status RpcTransportOwner::SetFatal(Status status) {
  if (fatal_.ok() && !status.ok())
    fatal_ = std::move(status);
  return fatal_;
}

Result<std::uint32_t> RpcTransportOwner::ReserveTx(TxKind kind) {
  for (std::uint32_t slot = 0; slot < tx_slots_.size(); ++slot) {
    if (tx_slots_[slot].active)
      continue;
    auto &tx = tx_slots_[slot];
    tx.active = true;
    ++tx.generation;
    if (tx.generation == 0)
      ++tx.generation;
    if (next_transport_ticket_ == 0) {
      return Status(StatusCode::kResourceExhausted,
                    "RPC transport ticket space is exhausted");
    }
    tx.kind = kind;
    tx.ticket = OperationTicket{next_transport_ticket_++, slot,
                                config_.transport_lane, tx.generation};
    return slot;
  }
  return Status(StatusCode::kResourceExhausted,
                "RPC transport TX slots are exhausted");
}

void RpcTransportOwner::ReleaseTx(std::uint32_t slot) noexcept {
  const auto generation = tx_slots_[slot].generation;
  tx_slots_[slot] = {};
  tx_slots_[slot].generation = generation;
}

std::optional<std::uint32_t>
RpcTransportOwner::AcquireRequestStaging() noexcept {
  for (std::uint32_t slot = 0; slot < request_staging_busy_.size(); ++slot) {
    if (!request_staging_busy_[slot]) {
      request_staging_busy_[slot] = true;
      return slot;
    }
  }
  return std::nullopt;
}

std::optional<std::uint32_t>
RpcTransportOwner::AcquireRequestSlot(std::uint32_t peer_rank) noexcept {
  auto &peer = peers_[peer_rank];
  for (std::size_t visited = 0; visited < peer.request_slots.size();
       ++visited) {
    const auto slot = static_cast<std::uint32_t>(
        (peer.request_cursor + visited) % peer.request_slots.size());
    if (peer.request_slots[slot].state != ClientRequestSlotState::kFree)
      continue;
    peer.request_slots[slot].state = ClientRequestSlotState::kPrepared;
    peer.request_cursor = (slot + 1U) % peer.request_slots.size();
    return slot;
  }
  return std::nullopt;
}

Status RpcTransportOwner::PollCompletions() {
  auto polled = engine_->Poll(completion_scratch_);
  if (!polled.ok())
    return SetFatal(polled.status());
  if (polled.value() > completion_scratch_.size()) {
    return SetFatal(Status(StatusCode::kProtocolError,
                           "transport returned too many completions"));
  }
  const auto completed_at = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < polled.value(); ++index) {
    const auto &completion = completion_scratch_[index];
    if (completion.ticket.operation_slot >= tx_slots_.size()) {
      return SetFatal(Status(StatusCode::kProtocolError,
                             "transport completion has invalid TX slot"));
    }
    auto &tx = tx_slots_[completion.ticket.operation_slot];
    if (!tx.active || tx.ticket != completion.ticket ||
        !completion.status.ok()) {
      return SetFatal(
          completion.status.ok()
              ? Status(StatusCode::kProtocolError,
                       "transport completion does not match active TX")
              : completion.status);
    }
    const auto elapsed = ElapsedNanoseconds(tx.posted_at, completed_at);
    if (tx.kind == TxKind::kRequest) {
      if (completion.operation_class != OperationClass::kRequestWrite ||
          tx.staging_slot >= request_staging_busy_.size() ||
          !request_staging_busy_[tx.staging_slot]) {
        return SetFatal(Status(StatusCode::kProtocolError,
                               "request completion metadata is invalid"));
      }
      request_staging_busy_[tx.staging_slot] = false;
      ++peers_[tx.batch.peer_rank].request_completed;
      ++profile_.request_completions;
      profile_.request_write_completion_nanoseconds += elapsed;
    } else {
      if (completion.operation_class != OperationClass::kResponseWrite) {
        return SetFatal(Status(StatusCode::kProtocolError,
                               "response completion class is invalid"));
      }
      auto status =
          server_->CompleteResultSend(tx.result_slot, tx.batch.sequence);
      if (!status.ok())
        return SetFatal(status);
      ++peers_[tx.result_slot.peer_rank].response_completed;
      ++response_completed_;
      ++profile_.response_completions;
      profile_.response_write_completion_nanoseconds += elapsed;
      profile_.result_post_to_completion_nanoseconds += elapsed;
      ++profile_.result_post_to_completion_samples;
    }
    ReleaseTx(completion.ticket.operation_slot);
  }
  return Status::Ok();
}

Status RpcTransportOwner::ScanResponses() {
  const auto ring_depth = memory_->config().ring_depth;
  const auto total_slots = owned_peers_.size() * ring_depth;
  if (total_slots == 0)
    return Status::Ok();
  std::uint32_t accepted = 0;
  for (std::size_t visited = 0;
       visited < total_slots && accepted < config_.phase_budget; ++visited) {
    const auto ordinal = (response_scan_cursor_ + visited) % total_slots;
    const auto peer = owned_peers_[ordinal / ring_depth];
    const auto slot_id = static_cast<std::uint32_t>(ordinal % ring_depth);
    auto &slot = peers_[peer].request_slots[slot_id];
    if (slot.state != ClientRequestSlotState::kWaitingResponse)
      continue;
    ++profile_.response_scan_slots;
    auto *notify =
        memory_->NotifyAddress(memory_->ResponseNotifyOffset(peer, slot_id));
    auto observed = DmaLoadNotifyAcquire(notify, *visibility_);
    if (!observed.ok())
      return SetFatal(observed.status());
    if (observed.value() == 0)
      continue;
    ++profile_.response_scan_hits;
    if (observed.value() != slot.sequence) {
      return SetFatal(Status(StatusCode::kProtocolError,
                             "response notify sequence is stale or wrong"));
    }
    const auto observed_at =
        config_.profiling_enabled
            ? static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count())
            : 0;
    auto status = client_->AcceptResponse(peer, slot_id, slot.sequence,
                                          memory_->ResponseSlot(peer, slot_id),
                                          observed_at);
    if (!status.ok())
      return SetFatal(status);
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile std::uint64_t *>(notify) = 0;
    slot.state = ClientRequestSlotState::kFree;
    slot.sequence = 0;
    ++peers_[peer].response_received;
    ++profile_.response_batches_received;
    --profile_.request_slots_in_use;
    ++accepted;
    response_scan_cursor_ = (ordinal + 1U) % total_slots;
  }
  return Status::Ok();
}

Status RpcTransportOwner::PostResponses() {
  std::uint64_t bytes = 0;
  for (std::uint32_t posted = 0; posted < config_.phase_budget; ++posted) {
    ++profile_.result_slot_scans;
    auto ready = server_->TryGetReadyResult(config_.owner_thread);
    if (!ready.ok())
      return SetFatal(ready.status());
    if (!ready.value().has_value())
      return Status::Ok();
    ++profile_.result_slot_hits;
    const auto &result = *ready.value();
    // This is a per-round work budget, not a message size limit. Always allow
    // one response: otherwise a valid batch larger than the budget stays Ready
    // forever and its peer can never retire the corresponding request slot.
    if (posted != 0 &&
        bytes + result.bytes > config_.response_progress_max_bytes)
      return Status::Ok();
    auto tx_slot = ReserveTx(TxKind::kResponse);
    if (!tx_slot.ok())
      return Status::Ok();
    auto &tx = tx_slots_[tx_slot.value()];
    tx.batch = PreparedBatch{result.id.peer_rank, result.id.slot_id,
                             result.sequence, result.task_count, result.bytes};
    tx.result_slot = result.id;
    const WriteNotifyOperation operation{
        OperationClass::kResponseWrite,
        tx.ticket,
        result.id.peer_rank,
        SegmentKind::kResponses,
        result.payload,
        memory_->ResponseOffset(config_.local_rank, result.id.slot_id),
        SegmentKind::kNotify,
        memory_->ResponseNotifyOffset(config_.local_rank, result.id.slot_id),
        result.sequence,
        false};
    // The peer can receive the response and reuse this slot before post
    // returns.
    auto status = server_->MarkResultPosted(result.id, result.sequence);
    if (!status.ok())
      return SetFatal(status);
    status = engine_->PostWriteNotify(operation);
    if (!status.ok()) {
      if (IsBackpressure(status)) {
        auto aborted = server_->AbortResultPost(result.id, result.sequence);
        if (!aborted.ok())
          return SetFatal(aborted);
        ReleaseTx(tx_slot.value());
        ++profile_.result_post_retries;
        return Status::Ok();
      }
      // A hard provider failure can follow acceptance; retain the send lease.
      return SetFatal(status);
    }
    tx.posted_at = std::chrono::steady_clock::now();
    if (config_.profiling_enabled) {
      const auto posted_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              tx.posted_at.time_since_epoch())
              .count());
      if (result.request_notify_observed_nanoseconds != 0 &&
          posted_ns >= result.request_notify_observed_nanoseconds) {
        const auto elapsed =
            posted_ns - result.request_notify_observed_nanoseconds;
        profile_.request_notify_observed_to_response_post_nanoseconds +=
            elapsed * result.task_count;
        profile_.request_notify_observed_to_response_post_samples +=
            result.task_count;
      }
      if (result.ready_nanoseconds != 0 &&
          posted_ns >= result.ready_nanoseconds) {
        profile_.result_ready_to_post_nanoseconds +=
            posted_ns - result.ready_nanoseconds;
        ++profile_.result_ready_to_post_samples;
      }
    }
    ++peers_[result.id.peer_rank].response_posted;
    ++response_posted_;
    ++profile_.response_physical_writes;
    bytes += result.bytes;
  }
  return Status::Ok();
}

Status RpcTransportOwner::DrainClientSubmissions() {
  auto drained = client_->DrainSubmissions(config_.phase_budget);
  return drained.ok() ? Status::Ok() : SetFatal(drained.status());
}

Status RpcTransportOwner::PostRequests() {
  const auto pending_peers = client_->pending_peers();
  active_peer_scratch_.assign(pending_peers.begin(), pending_peers.end());
  if (active_peer_scratch_.empty())
    return Status::Ok();
  const auto visit_count =
      std::min<std::size_t>(active_peer_scratch_.size(), config_.phase_budget);
  for (std::size_t visited = 0; visited < visit_count; ++visited) {
    const auto peer = active_peer_scratch_[request_post_cursor_ %
                                           active_peer_scratch_.size()];
    request_post_cursor_ =
        (request_post_cursor_ + 1U) % active_peer_scratch_.size();
    ++profile_.request_post_peer_visits;
    auto slot_id = AcquireRequestSlot(peer);
    if (!slot_id.has_value()) {
      ++profile_.request_post_free_slot_waits;
      continue;
    }
    auto staging = AcquireRequestStaging();
    if (!staging.has_value()) {
      peers_[peer].request_slots[*slot_id].state =
          ClientRequestSlotState::kFree;
      ++profile_.request_post_staging_unavailable;
      return Status::Ok();
    }
    auto tx_slot = ReserveTx(TxKind::kRequest);
    if (!tx_slot.ok()) {
      request_staging_busy_[*staging] = false;
      peers_[peer].request_slots[*slot_id].state =
          ClientRequestSlotState::kFree;
      ++profile_.request_post_tx_unavailable;
      return Status::Ok();
    }
    auto &peer_state = peers_[peer];
    const auto sequence = peer_state.next_request_sequence++;
    if (sequence == 0 || peer_state.next_request_sequence == 0) {
      return SetFatal(Status(StatusCode::kResourceExhausted,
                             "request sequence space is exhausted"));
    }
    auto prepared = client_->PrepareRequestBatch(
        peer, *slot_id, sequence, config_.request_batch_max_tasks,
        config_.request_batch_max_bytes, memory_->config().response_capacity,
        memory_->RequestStaging(config_.owner_thread, *staging));
    if (!prepared.ok())
      return SetFatal(prepared.status());
    if (!prepared.value().has_value()) {
      request_staging_busy_[*staging] = false;
      peer_state.request_slots[*slot_id].state = ClientRequestSlotState::kFree;
      ReleaseTx(tx_slot.value());
      continue;
    }
    auto &tx = tx_slots_[tx_slot.value()];
    tx.staging_slot = *staging;
    tx.batch = *prepared.value();
    auto valid = client_->ValidatePreparedRequest(tx.batch);
    if (!valid.ok())
      return SetFatal(valid.status());
    if (!valid.value()) {
      request_staging_busy_[*staging] = false;
      peer_state.request_slots[*slot_id].state = ClientRequestSlotState::kFree;
      ReleaseTx(tx_slot.value());
      continue;
    }
    const auto payload = memory_->RequestStaging(config_.owner_thread, *staging)
                             .first(tx.batch.bytes);
    const WriteNotifyOperation operation{
        OperationClass::kRequestWrite,
        tx.ticket,
        peer,
        SegmentKind::kRequests,
        payload,
        memory_->RequestOffset(config_.local_rank, *slot_id),
        SegmentKind::kNotify,
        memory_->RequestNotifyOffset(config_.local_rank, *slot_id),
        sequence,
        false};
    auto status = engine_->PostWriteNotify(operation);
    if (!status.ok()) {
      auto aborted = client_->AbortPreparedRequest(tx.batch);
      request_staging_busy_[*staging] = false;
      peer_state.request_slots[*slot_id].state = ClientRequestSlotState::kFree;
      ReleaseTx(tx_slot.value());
      if (!aborted.ok())
        return SetFatal(aborted);
      if (IsBackpressure(status)) {
        ++profile_.request_post_provider_backpressure_batches;
        return Status::Ok();
      }
      return SetFatal(status);
    }
    status = client_->CommitPreparedRequest(tx.batch);
    if (!status.ok())
      return SetFatal(status);
    auto &request_slot = peer_state.request_slots[*slot_id];
    request_slot.sequence = sequence;
    request_slot.state = ClientRequestSlotState::kWaitingResponse;
    tx.posted_at = std::chrono::steady_clock::now();
    ++peer_state.request_posted;
    ++profile_.request_physical_writes;
    ++profile_.request_slots_in_use;
    profile_.request_slots_high_watermark = std::max(
        profile_.request_slots_high_watermark, profile_.request_slots_in_use);
  }
  return Status::Ok();
}

Status RpcTransportOwner::Progress() {
  if (!fatal_.ok())
    return fatal_;
  const auto timed = config_.profiling_enabled;
  auto step_started = std::chrono::steady_clock::now();
  auto status = PollCompletions();
  if (!status.ok())
    return status;
  auto now = std::chrono::steady_clock::now();
  if (timed)
    profile_.progress_step_nanoseconds[0] +=
        ElapsedNanoseconds(step_started, now);
  step_started = now;
  status = ScanResponses();
  if (!status.ok())
    return status;
  now = std::chrono::steady_clock::now();
  if (timed)
    profile_.progress_step_nanoseconds[1] +=
        ElapsedNanoseconds(step_started, now);
  step_started = now;
  status = PostResponses();
  if (!status.ok())
    return status;
  now = std::chrono::steady_clock::now();
  if (timed)
    profile_.progress_step_nanoseconds[2] +=
        ElapsedNanoseconds(step_started, now);
  step_started = now;
  status = DrainClientSubmissions();
  if (!status.ok())
    return status;
  now = std::chrono::steady_clock::now();
  if (timed)
    profile_.progress_step_nanoseconds[3] +=
        ElapsedNanoseconds(step_started, now);
  step_started = now;
  status = PostRequests();
  now = std::chrono::steady_clock::now();
  if (timed) {
    profile_.progress_step_nanoseconds[4] +=
        ElapsedNanoseconds(step_started, now);
    ++profile_.progress_timing_samples;
  }
  ++profile_.progress_iterations;
  return status;
}

bool RpcTransportOwner::drained() const noexcept {
  if (!fatal_.ok() || !client_->drained() ||
      !server_->drained(config_.owner_thread))
    return false;
  for (const auto &tx : tx_slots_) {
    if (tx.active)
      return false;
  }
  for (const auto peer : owned_peers_) {
    for (const auto &slot : peers_[peer].request_slots) {
      if (slot.state != ClientRequestSlotState::kFree)
        return false;
    }
  }
  return true;
}

Result<RpcTransportPeerSnapshot>
RpcTransportOwner::PeerSnapshot(std::uint32_t peer_rank) const {
  if (peer_rank >= peers_.size() || peers_[peer_rank].request_slots.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "peer is not owned by this RPC transport");
  }
  RpcTransportPeerSnapshot snapshot;
  const auto &peer = peers_[peer_rank];
  snapshot.request_posted = peer.request_posted;
  snapshot.request_completed = peer.request_completed;
  snapshot.response_posted = peer.response_posted;
  snapshot.response_completed = peer.response_completed;
  snapshot.response_received = peer.response_received;
  for (std::uint32_t slot_id = 0; slot_id < peer.request_slots.size();
       ++slot_id) {
    const auto &slot = peer.request_slots[slot_id];
    if (slot.state == ClientRequestSlotState::kFree)
      continue;
    ++snapshot.busy_request_slots;
    if (snapshot.waiting_sequence == 0) {
      snapshot.waiting_slot_id = slot_id;
      snapshot.waiting_sequence = slot.sequence;
      const auto *notify = memory_->NotifyAddress(
          memory_->ResponseNotifyOffset(peer_rank, slot_id));
      snapshot.response_notify_value =
          *reinterpret_cast<const volatile std::uint64_t *>(notify);
    }
  }
  snapshot.owner_profile = profile_;
  return snapshot;
}

} // namespace hnsw_urma
