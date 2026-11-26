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

#include "hnsw_urma/transport/mock_lane_post_engine.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>

namespace hnsw_urma {

Result<MockLanePostEngine>
MockLanePostEngine::Create(std::uint32_t owner_lane,
                           std::uint32_t max_outstanding) {
  if (max_outstanding == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "mock lane depth must be non-zero");
  }
  try {
    MockLanePostEngine engine;
    engine.owner_lane_ = owner_lane;
    engine.max_outstanding_ = max_outstanding;
    engine.events_.reserve(max_outstanding);
    engine.completions_.reserve(max_outstanding * 2U);
    return engine;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate mock lane engine");
  }
}

Status MockLanePostEngine::RegisterSegment(std::uint32_t peer_rank,
                                           SegmentKind kind,
                                           std::span<std::byte> memory) {
  if (memory.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "mock segment must be non-empty");
  }
  for (const auto &segment : segments_) {
    if (segment.peer_rank == peer_rank && segment.kind == kind) {
      return Status(StatusCode::kInvalidArgument,
                    "mock segment is already registered");
    }
  }
  try {
    segments_.push_back({peer_rank, kind, memory});
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to register mock segment");
  }
}

Status MockLanePostEngine::ValidateTicket(
    const OperationTicket &ticket) const noexcept {
  if (ticket.value == 0 || ticket.generation == 0 ||
      ticket.owner_lane != owner_lane_) {
    return Status(StatusCode::kProtocolError,
                  "operation ticket is invalid for this owner lane");
  }
  return Status::Ok();
}

Status MockLanePostEngine::PostRead(const ReadOperation &operation) {
  if (!next_post_failure_.ok()) {
    Status failure = std::move(next_post_failure_);
    next_post_failure_ = Status::Ok();
    return failure;
  }
  auto status = ValidateTicket(operation.ticket);
  if (!status.ok() || operation.destination.empty()) {
    return status.ok() ? Status(StatusCode::kInvalidArgument,
                                "READ destination is empty")
                       : status;
  }
  if (events_.size() >= max_outstanding_) {
    return Status(StatusCode::kResourceExhausted,
                  "mock lane event queue is full");
  }
  Event event;
  event.is_read = true;
  event.read = operation;
  events_.push_back(std::move(event));
  return Status::Ok();
}

Status
MockLanePostEngine::PostWriteNotify(const WriteNotifyOperation &operation) {
  if (!next_post_failure_.ok()) {
    Status failure = std::move(next_post_failure_);
    next_post_failure_ = Status::Ok();
    return failure;
  }
  auto status = ValidateTicket(operation.ticket);
  if (!status.ok() || operation.operation_class == OperationClass::kRead ||
      operation.payload.empty() || operation.notify_value == 0) {
    return status.ok() ? Status(StatusCode::kInvalidArgument,
                                "invalid WRITE_NOTIFY operation")
                       : status;
  }
  if (events_.size() >= max_outstanding_) {
    return Status(StatusCode::kResourceExhausted,
                  "mock lane event queue is full");
  }
  Event event;
  event.is_read = false;
  event.write = operation;
  if (operation.inline_source) {
    ++inline_write_notify_posts_;
    event.inline_payload.assign(operation.payload.begin(),
                                operation.payload.end());
    event.write.payload = event.inline_payload;
  }
  events_.push_back(std::move(event));
  if (events_.back().write.inline_source) {
    events_.back().write.payload = events_.back().inline_payload;
  }
  return Status::Ok();
}

LanePostBatchResult MockLanePostEngine::PostWriteNotifies(
    std::span<const WriteNotifyOperation> operations) {
  ++write_notify_batch_calls_;
  maximum_write_notify_batch_ =
      std::max(maximum_write_notify_batch_, operations.size());
  return LanePostEngine::PostWriteNotifies(operations);
}

Result<std::span<std::byte>>
MockLanePostEngine::FindSegment(std::uint32_t peer_rank,
                                SegmentKind kind) noexcept {
  for (auto &segment : segments_) {
    if (segment.peer_rank == peer_rank && segment.kind == kind) {
      return segment.memory;
    }
  }
  return Status(StatusCode::kTransportError,
                "mock peer segment is not registered");
}

Status MockLanePostEngine::Progress(std::size_t event_index) {
  if (event_index >= events_.size()) {
    return Status(StatusCode::kOutOfRange, "mock event index is invalid");
  }
  Event event = std::move(events_[event_index]);
  events_.erase(events_.begin() + static_cast<std::ptrdiff_t>(event_index));
  TransportCompletion completion;
  if (event.is_read) {
    completion.ticket = event.read.ticket;
    completion.operation_class = OperationClass::kRead;
    completion.bytes = event.read.destination.size();
    auto segment = FindSegment(event.read.peer_rank, event.read.remote_segment);
    if (!segment.ok() || event.read.remote_offset > segment.value().size() ||
        event.read.destination.size() >
            segment.value().size() - event.read.remote_offset) {
      completion.status = segment.ok() ? Status(StatusCode::kTransportError,
                                                "mock READ is out of bounds")
                                       : segment.status();
    } else {
      std::memcpy(event.read.destination.data(),
                  segment.value().data() + event.read.remote_offset,
                  event.read.destination.size());
    }
  } else {
    completion.ticket = event.write.ticket;
    completion.operation_class = event.write.operation_class;
    completion.bytes = event.write.payload.size();
    auto payload_segment =
        FindSegment(event.write.peer_rank, event.write.payload_segment);
    auto notify_segment =
        FindSegment(event.write.peer_rank, event.write.notify_segment);
    if (!payload_segment.ok() || !notify_segment.ok() ||
        event.write.remote_payload_offset > payload_segment.value().size() ||
        event.write.payload.size() > payload_segment.value().size() -
                                         event.write.remote_payload_offset ||
        event.write.remote_notify_offset > notify_segment.value().size() ||
        sizeof(std::uint64_t) >
            notify_segment.value().size() - event.write.remote_notify_offset ||
        event.write.remote_notify_offset % alignof(std::uint64_t) != 0) {
      completion.status =
          Status(StatusCode::kTransportError,
                 "mock WRITE_NOTIFY target is out of bounds or unaligned");
    } else {
      std::memcpy(payload_segment.value().data() +
                      event.write.remote_payload_offset,
                  event.write.payload.data(), event.write.payload.size());
      std::atomic_thread_fence(std::memory_order_release);
      std::memcpy(notify_segment.value().data() +
                      event.write.remote_notify_offset,
                  &event.write.notify_value, sizeof(event.write.notify_value));
    }
  }
  completions_.push_back(std::move(completion));
  return Status::Ok();
}

Status MockLanePostEngine::Drop(std::size_t event_index) {
  if (event_index >= events_.size()) {
    return Status(StatusCode::kOutOfRange, "mock event index is invalid");
  }
  events_.erase(events_.begin() + static_cast<std::ptrdiff_t>(event_index));
  return Status::Ok();
}

void MockLanePostEngine::InjectCompletion(TransportCompletion completion) {
  completions_.push_back(std::move(completion));
}

Result<std::size_t>
MockLanePostEngine::Poll(std::span<TransportCompletion> completions) {
  const std::size_t count = std::min(completions.size(), completions_.size());
  for (std::size_t index = 0; index < count; ++index) {
    completions[index] = std::move(completions_[index]);
  }
  completions_.erase(completions_.begin(),
                     completions_.begin() + static_cast<std::ptrdiff_t>(count));
  return count;
}

} // namespace hnsw_urma
