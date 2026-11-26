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

#include "hnsw_urma/transport/shared_memory_transport.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace hnsw_urma {

SharedMemoryPeer::Segment *SharedMemoryPeer::Find(SegmentKind kind) noexcept {
  for (auto &segment : segments_) {
    if (segment.kind == kind) {
      return &segment;
    }
  }
  return nullptr;
}

Result<std::shared_ptr<SharedMemoryPeer>> SharedMemoryPeer::Open(
    std::uint32_t rank,
    std::span<const SharedMemorySegmentDescriptor> segments) {
  if (segments.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "shared-memory peer has no segments");
  }
  try {
    auto peer = std::shared_ptr<SharedMemoryPeer>(new SharedMemoryPeer());
    peer->rank_ = rank;
    peer->segments_.reserve(segments.size());
    for (const auto &descriptor : segments) {
      if (descriptor.name.empty() || descriptor.bytes == 0 ||
          peer->Find(descriptor.kind) != nullptr) {
        return Status(StatusCode::kInvalidArgument,
                      "shared-memory peer segment metadata is invalid")
            .AddContext("peer_rank", std::to_string(rank))
            .AddContext(
                "segment_kind",
                std::to_string(static_cast<std::uint32_t>(descriptor.kind)));
      }
      auto mapping =
          PageAlignedBuffer::OpenShared(descriptor.name, descriptor.bytes);
      if (!mapping.ok()) {
        auto status = mapping.status();
        status.AddContext("peer_rank", std::to_string(rank))
            .AddContext(
                "segment_kind",
                std::to_string(static_cast<std::uint32_t>(descriptor.kind)));
        return status;
      }
      peer->segments_.push_back({descriptor.kind, std::move(mapping).value()});
    }
    return peer;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to map shared-memory peer");
  }
}

struct SharedMemoryLanePostEngine::Impl {
  std::uint32_t owner_lane{0};
  std::vector<std::shared_ptr<SharedMemoryPeer>> peers;
  std::unique_ptr<LanePostEngine> fallback;
  std::vector<TransportCompletion> completions;
  std::size_t completion_head{0};
  std::size_t completion_count{0};
  std::size_t batch_max{0};
  bool local_poll_first{true};

  SharedMemoryPeer *FindPeer(std::uint32_t rank) noexcept {
    for (const auto &peer : peers) {
      if (peer->rank() == rank) {
        return peer.get();
      }
    }
    return nullptr;
  }

  Status ValidateTicket(const OperationTicket &ticket) const {
    if (ticket.value == 0 || ticket.generation == 0 ||
        ticket.owner_lane != owner_lane) {
      return Status(StatusCode::kProtocolError,
                    "ticket does not belong to this shared-memory lane");
    }
    return Status::Ok();
  }

  void PushCompletion(const OperationTicket &ticket,
                      OperationClass operation_class,
                      std::uint64_t bytes) noexcept {
    const auto tail = (completion_head + completion_count) % completions.size();
    completions[tail] = {ticket, operation_class, Status::Ok(), bytes};
    ++completion_count;
  }

  std::size_t PollLocal(std::span<TransportCompletion> output) noexcept {
    const auto count = std::min(output.size(), completion_count);
    for (std::size_t index = 0; index < count; ++index) {
      output[index] = std::move(completions[completion_head]);
      completions[completion_head] = {};
      completion_head = (completion_head + 1U) % completions.size();
    }
    completion_count -= count;
    return count;
  }
};

SharedMemoryLanePostEngine::SharedMemoryLanePostEngine(
    std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}

SharedMemoryLanePostEngine::~SharedMemoryLanePostEngine() = default;

Result<std::unique_ptr<LanePostEngine>>
SharedMemoryLanePostEngine::Create(SharedMemoryLaneOptions options) {
  if (!options.fallback || options.peers.empty() ||
      options.max_outstanding == 0 || options.batch_max == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "shared-memory lane options are invalid");
  }
  try {
    if (std::any_of(options.peers.begin(), options.peers.end(),
                    [](const auto &peer) { return !peer; })) {
      return Status(StatusCode::kInvalidArgument,
                    "shared-memory lane peer set is invalid");
    }
    std::sort(options.peers.begin(), options.peers.end(),
              [](const auto &left, const auto &right) {
                return left->rank() < right->rank();
              });
    if (std::adjacent_find(options.peers.begin(), options.peers.end(),
                           [](const auto &left, const auto &right) {
                             return left->rank() == right->rank();
                           }) != options.peers.end()) {
      return Status(StatusCode::kInvalidArgument,
                    "shared-memory lane peer set is invalid");
    }
    auto implementation = std::unique_ptr<Impl>(new Impl());
    implementation->owner_lane = options.owner_lane;
    implementation->peers = std::move(options.peers);
    implementation->fallback = std::move(options.fallback);
    implementation->completions.resize(options.max_outstanding);
    implementation->batch_max = options.batch_max;
    return std::unique_ptr<LanePostEngine>(
        new SharedMemoryLanePostEngine(std::move(implementation)));
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate shared-memory lane");
  }
}

Status SharedMemoryLanePostEngine::PostRead(const ReadOperation &operation) {
  return PostReads(std::span(&operation, 1)).status;
}

Status SharedMemoryLanePostEngine::PostWriteNotify(
    const WriteNotifyOperation &operation) {
  return PostWriteNotifies(std::span(&operation, 1)).status;
}

LanePostBatchResult SharedMemoryLanePostEngine::PostReads(
    std::span<const ReadOperation> operations) {
  LanePostBatchResult result;
  if (operations.empty()) {
    return result;
  }
  auto *peer = impl_->FindPeer(operations.front().peer_rank);
  if (peer == nullptr) {
    return impl_->fallback->PostReads(operations);
  }
  auto count = std::min({operations.size(), impl_->batch_max,
                         impl_->completions.size() - impl_->completion_count});
  if (count == 0) {
    result.status = Status(StatusCode::kResourceExhausted,
                           "shared-memory lane has no READ completion slots");
    return result;
  }
  const auto segment_kind = operations.front().remote_segment;
  for (std::size_t index = 1; index < count; ++index) {
    if (operations[index].peer_rank != peer->rank() ||
        operations[index].remote_segment != segment_kind) {
      count = index;
      break;
    }
  }
  auto *segment = peer->Find(segment_kind);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &operation = operations[index];
    auto ticket_status = impl_->ValidateTicket(operation.ticket);
    const auto memory = segment == nullptr ? std::span<const std::byte>{}
                                           : segment->memory.logical_memory();
    if (!ticket_status.ok() || operation.destination.empty() ||
        segment == nullptr || operation.remote_offset > memory.size() ||
        operation.destination.size() >
            memory.size() - operation.remote_offset) {
      result.status =
          ticket_status.ok()
              ? Status(StatusCode::kInvalidArgument,
                       "shared-memory READ binding, size, or offset is invalid")
              : ticket_status;
      return result;
    }
  }
  for (std::size_t index = 0; index < count; ++index) {
    const auto &operation = operations[index];
    const auto memory = segment->memory.logical_memory();
    std::memcpy(operation.destination.data(),
                memory.data() + operation.remote_offset,
                operation.destination.size());
    impl_->PushCompletion(operation.ticket, OperationClass::kRead,
                          operation.destination.size());
  }
  result.post_calls = 1;
  result.posted = count;
  return result;
}

LanePostBatchResult SharedMemoryLanePostEngine::PostWriteNotifies(
    std::span<const WriteNotifyOperation> operations) {
  LanePostBatchResult result;
  if (operations.empty()) {
    return result;
  }
  auto *peer = impl_->FindPeer(operations.front().peer_rank);
  if (peer == nullptr) {
    return impl_->fallback->PostWriteNotifies(operations);
  }
  auto count = std::min({operations.size(), impl_->batch_max,
                         impl_->completions.size() - impl_->completion_count});
  if (count == 0) {
    result.status =
        Status(StatusCode::kResourceExhausted,
               "shared-memory lane has no WRITE_NOTIFY completion slots");
    return result;
  }
  const auto operation_class = operations.front().operation_class;
  const auto payload_kind = operations.front().payload_segment;
  const auto notify_kind = operations.front().notify_segment;
  for (std::size_t index = 1; index < count; ++index) {
    const auto &operation = operations[index];
    if (operation.peer_rank != peer->rank() ||
        operation.operation_class != operation_class ||
        operation.payload_segment != payload_kind ||
        operation.notify_segment != notify_kind) {
      count = index;
      break;
    }
  }
  auto *payload_segment = peer->Find(payload_kind);
  auto *notify_segment = peer->Find(notify_kind);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &operation = operations[index];
    auto ticket_status = impl_->ValidateTicket(operation.ticket);
    const auto payload_memory = payload_segment == nullptr
                                    ? std::span<const std::byte>{}
                                    : payload_segment->memory.logical_memory();
    const auto notify_memory = notify_segment == nullptr
                                   ? std::span<const std::byte>{}
                                   : notify_segment->memory.logical_memory();
    if (!ticket_status.ok() || operation_class == OperationClass::kRead ||
        operation.payload.empty() || operation.notify_value == 0 ||
        payload_segment == nullptr || notify_segment == nullptr ||
        operation.remote_payload_offset > payload_memory.size() ||
        operation.payload.size() >
            payload_memory.size() - operation.remote_payload_offset ||
        operation.remote_notify_offset % alignof(std::uint64_t) != 0 ||
        operation.remote_notify_offset > notify_memory.size() ||
        sizeof(std::uint64_t) >
            notify_memory.size() - operation.remote_notify_offset) {
      result.status = ticket_status.ok()
                          ? Status(StatusCode::kInvalidArgument,
                                   "shared-memory WRITE_NOTIFY binding, size, "
                                   "or offset is invalid")
                          : ticket_status;
      return result;
    }
  }
  static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
  for (std::size_t index = 0; index < count; ++index) {
    const auto &operation = operations[index];
    auto payload_memory = payload_segment->memory.logical_memory();
    auto notify_memory = notify_segment->memory.logical_memory();
    std::memcpy(payload_memory.data() + operation.remote_payload_offset,
                operation.payload.data(), operation.payload.size());
    auto *notify = reinterpret_cast<std::uint64_t *>(
        notify_memory.data() + operation.remote_notify_offset);
    std::atomic_ref<std::uint64_t>(*notify).store(operation.notify_value,
                                                  std::memory_order_release);
    impl_->PushCompletion(operation.ticket, operation_class,
                          operation.payload.size());
  }
  result.posted = count;
  return result;
}

Result<std::size_t>
SharedMemoryLanePostEngine::Poll(std::span<TransportCompletion> completions) {
  if (completions.empty()) {
    return std::size_t{0};
  }
  if (impl_->local_poll_first && impl_->completion_count != 0) {
    impl_->local_poll_first = false;
    return impl_->PollLocal(completions);
  }
  auto remote = impl_->fallback->Poll(completions);
  if (!remote.ok()) {
    return remote.status();
  }
  const auto local = impl_->PollLocal(completions.subspan(remote.value()));
  impl_->local_poll_first = local == 0;
  return remote.value() + local;
}

} // namespace hnsw_urma
