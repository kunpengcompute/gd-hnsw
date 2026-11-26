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

#include <cstddef>
#include <cstdint>
#include <span>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

enum class OperationClass : std::uint8_t {
  kRead,
  kRequestWrite,
  kResponseWrite,
};

enum class SegmentKind : std::uint8_t {
  kLayer0Graph,
  kVectors,
  kRequests,
  kResponses,
  kNotify,
  kRpcPool,
};

struct OperationTicket {
  ticket_t value{0};
  std::uint32_t operation_slot{0};
  std::uint32_t owner_lane{0};
  std::uint64_t generation{0};

  friend bool operator==(const OperationTicket &,
                         const OperationTicket &) = default;
};

struct ReadOperation {
  OperationTicket ticket;
  std::uint32_t peer_rank{0};
  SegmentKind remote_segment{SegmentKind::kLayer0Graph};
  std::span<std::byte> destination;
  segment_offset_t remote_offset{0};
};

struct WriteNotifyOperation {
  OperationClass operation_class{OperationClass::kRequestWrite};
  OperationTicket ticket;
  std::uint32_t peer_rank{0};
  SegmentKind payload_segment{SegmentKind::kRequests};
  std::span<const std::byte> payload;
  segment_offset_t remote_payload_offset{0};
  SegmentKind notify_segment{SegmentKind::kNotify};
  segment_offset_t remote_notify_offset{0};
  std::uint64_t notify_value{0};
  bool inline_source{false};
};

struct TransportCompletion {
  OperationTicket ticket;
  OperationClass operation_class{OperationClass::kRead};
  Status status;
  std::uint64_t bytes{0};
};

struct LanePostBatchResult {
  std::size_t posted{0};
  Status status;
  // Time spent inside the native provider post call. Generic and mock
  // engines leave this at zero.
  std::uint64_t provider_post_nanoseconds{0};
  // Actual submission calls; a native engine returns zero for software-only
  // backpressure. The generic fallback counts its single-READ calls.
  std::uint64_t post_calls{0};
};

// Internal transport execution boundary. Only a Progress Lane may call an
// implementation. Search code receives higher-level reader/oracle interfaces.
class LanePostEngine {
public:
  virtual ~LanePostEngine() = default;
  virtual Status PostRead(const ReadOperation &operation) = 0;
  virtual Status PostWriteNotify(const WriteNotifyOperation &operation) = 0;
  // Accept a possibly shorter prefix using current WR/descriptor credits.
  // Only posted entries may DMA; retryable suffixes remain unsubmitted.
  virtual LanePostBatchResult
  PostReads(std::span<const ReadOperation> operations);
  virtual LanePostBatchResult
  PostWriteNotifies(std::span<const WriteNotifyOperation> operations);
  virtual Result<std::size_t>
  Poll(std::span<TransportCompletion> completions) = 0;
};

} // namespace hnsw_urma
