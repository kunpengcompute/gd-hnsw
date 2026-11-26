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

#include <cstddef>
#include <cstdint>
#include <span>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

inline constexpr std::uint32_t DISTANCE_BATCH_REQUEST_MAGIC = 0x34514248U;
inline constexpr std::uint32_t DISTANCE_BATCH_RESPONSE_MAGIC = 0x34534248U;
inline constexpr std::uint16_t DISTANCE_BATCH_WIRE_VERSION = 8;

enum class DistanceBatchKind : std::uint8_t {
  kRequest = 1,
  kResponse = 2,
};

struct DistanceBatchHeader {
  std::uint32_t magic{0};
  std::uint16_t version{DISTANCE_BATCH_WIRE_VERSION};
  DistanceBatchKind kind{DistanceBatchKind::kRequest};
  std::uint8_t flags{0};
  std::uint32_t header_bytes{40};
  std::uint32_t task_count{0};
  std::uint64_t batch_sequence{0};
  std::uint32_t task_table_offset{40};
  std::uint32_t payload_offset{0};
  std::uint32_t payload_bytes{0};
  std::uint32_t total_bytes{0};
};
static_assert(sizeof(DistanceBatchHeader) == 40);
static_assert(alignof(DistanceBatchHeader) == 8);

enum DistanceBatchRequestFlag : std::uint32_t {
  kBatchHasQueryVector = 1U << 0U,
  kBatchUseDotNorm = 1U << 1U,
  kBatchHasUpperDistance = 1U << 2U,
};
inline constexpr std::uint32_t DISTANCE_BATCH_REQUEST_KNOWN_FLAGS =
    kBatchHasQueryVector | kBatchUseDotNorm | kBatchHasUpperDistance;

struct DistanceBatchRequestTask {
  ticket_t task_ticket{0};
  std::uint32_t query_slot_id{0};
  shard_id_t owner_shard{0};
  std::uint32_t count{0};
  std::uint32_t dim{0};
  std::uint32_t flags{0};
  std::uint32_t ids_offset{0};
  std::uint32_t query_offset{0};
  std::uint32_t payload_end{0};
  float upper_distance{0.0F};
  std::uint32_t reserved{0};
};
static_assert(sizeof(DistanceBatchRequestTask) == 48);
static_assert(alignof(DistanceBatchRequestTask) == 8);

struct DistanceBatchResponseTask {
  ticket_t task_ticket{0};
  std::uint32_t query_slot_id{0};
  std::uint32_t count{0};
  std::uint32_t ids_offset{0};
  std::uint32_t distances_offset{0};
  std::uint32_t payload_end{0};
  std::uint32_t reserved{0};
};
static_assert(sizeof(DistanceBatchResponseTask) == 32);
static_assert(alignof(DistanceBatchResponseTask) == 8);

struct DistanceBatchRequestTaskInput {
  ticket_t task_ticket{0};
  std::uint32_t query_slot_id{0};
  shard_id_t owner_shard{0};
  std::uint32_t flags{0};
  std::span<const node_id_t> ids;
  std::span<const float> query_vector;
  std::uint32_t dim{0};
  float upper_distance{0.0F};
};

struct DistanceBatchResponseTaskInput {
  ticket_t task_ticket{0};
  std::uint32_t query_slot_id{0};
  std::span<const float> distances;
  std::span<const node_id_t> ids;
};

struct DistanceBatchRequestTaskView {
  DistanceBatchRequestTask descriptor{};
  std::span<const node_id_t> ids;
  std::span<const float> query_vector;
};

struct DistanceBatchResponseTaskView {
  DistanceBatchResponseTask descriptor{};
  std::span<const node_id_t> ids;
  std::span<const float> distances;
};

class DistanceBatchRequestView {
public:
  [[nodiscard]] const DistanceBatchHeader &header() const noexcept {
    return header_;
  }
  [[nodiscard]] std::size_t task_count() const noexcept {
    return descriptors_.size();
  }
  [[nodiscard]] Result<DistanceBatchRequestTaskView>
  Task(std::size_t index) const noexcept;

private:
  friend Result<DistanceBatchRequestView>
      DecodeDistanceBatchRequest(std::span<const std::byte>) noexcept;
  DistanceBatchHeader header_{};
  std::span<const DistanceBatchRequestTask> descriptors_;
  std::span<const std::byte> message_;
};

class DistanceBatchResponseView {
public:
  [[nodiscard]] const DistanceBatchHeader &header() const noexcept {
    return header_;
  }
  [[nodiscard]] std::size_t task_count() const noexcept {
    return descriptors_.size();
  }
  [[nodiscard]] Result<DistanceBatchResponseTaskView>
  Task(std::size_t index) const noexcept;

private:
  friend Result<DistanceBatchResponseView>
      DecodeDistanceBatchResponse(std::span<const std::byte>) noexcept;
  DistanceBatchHeader header_{};
  std::span<const DistanceBatchResponseTask> descriptors_;
  std::span<const std::byte> message_;
};

[[nodiscard]] Result<std::size_t> DistanceBatchRequestBytesForShape(
    std::uint32_t task_count, std::uint64_t total_id_count,
    std::uint64_t total_query_value_count) noexcept;
[[nodiscard]] Result<std::size_t> EncodedDistanceBatchRequestBytes(
    std::span<const DistanceBatchRequestTaskInput> tasks) noexcept;
[[nodiscard]] Result<std::size_t>
EncodeDistanceBatchRequest(std::uint64_t batch_sequence,
                           std::span<const DistanceBatchRequestTaskInput> tasks,
                           std::span<std::byte> output) noexcept;
[[nodiscard]] Result<DistanceBatchHeader>
DecodeDistanceBatchRequestEnvelope(std::span<const std::byte> message) noexcept;
[[nodiscard]] Result<DistanceBatchRequestView>
DecodeDistanceBatchRequest(std::span<const std::byte> message) noexcept;

[[nodiscard]] Result<std::size_t>
DistanceBatchResponseBytesForShape(std::uint32_t task_count,
                                   std::uint64_t total_distance_count) noexcept;
[[nodiscard]] Result<std::size_t> EncodedDistanceBatchResponseBytes(
    std::span<const DistanceBatchResponseTaskInput> tasks) noexcept;
[[nodiscard]] Result<std::size_t> EncodeDistanceBatchResponse(
    std::uint64_t batch_sequence,
    std::span<const DistanceBatchResponseTaskInput> tasks,
    std::span<std::byte> output) noexcept;
[[nodiscard]] Result<DistanceBatchResponseView>
DecodeDistanceBatchResponse(std::span<const std::byte> message) noexcept;

class DistanceBatchResponseWriter {
public:
  [[nodiscard]] static Result<DistanceBatchResponseWriter>
  Create(std::uint64_t batch_sequence, std::uint32_t task_count,
         std::span<std::byte> output) noexcept;
  [[nodiscard]] Status Append(ticket_t task_ticket, std::uint32_t query_slot_id,
                              std::span<const node_id_t> ids,
                              std::span<const float> distances) noexcept;
  [[nodiscard]] Result<std::size_t> Finish() noexcept;

private:
  std::span<std::byte> output_;
  DistanceBatchHeader header_{};
  std::uint32_t next_task_{0};
  std::size_t payload_cursor_{0};
};

} // namespace hnsw_urma
