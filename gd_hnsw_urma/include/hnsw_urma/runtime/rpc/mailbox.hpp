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
#include <memory>
#include <span>
#include <string_view>

#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

struct RpcSlotId {
  std::uint32_t peer_rank{0};
  std::uint32_t slot_id{0};
  friend bool operator==(const RpcSlotId &, const RpcSlotId &) = default;
};

struct RpcMailboxConfig {
  std::uint32_t world_size{0};
  std::uint32_t rpc_threads{0};
  std::uint32_t ring_depth{0};
  std::uint32_t request_capacity{0};
  std::uint32_t response_capacity{0};
  std::uint32_t request_staging_slots_per_thread{0};
};

// Three registered segments: inbound requests plus request TX staging,
// inbound responses plus the outbound result ring, and cache-line-isolated
// request/response notifications. Every address API uses an explicit slot ID.
class RpcMailboxMemory {
public:
  [[nodiscard]] static Result<std::unique_ptr<RpcMailboxMemory>>
  Create(RpcMailboxConfig config, bool shared = false);

  [[nodiscard]] const RpcMailboxConfig &config() const noexcept {
    return config_;
  }
  [[nodiscard]] std::uint64_t request_stride() const noexcept {
    return request_stride_;
  }
  [[nodiscard]] std::uint64_t response_stride() const noexcept {
    return response_stride_;
  }
  [[nodiscard]] std::span<std::byte> requests() noexcept {
    return request_storage_.logical_memory();
  }
  [[nodiscard]] std::span<std::byte> responses() noexcept {
    return response_storage_.logical_memory();
  }
  [[nodiscard]] std::span<std::byte> notify() noexcept {
    return notify_storage_.logical_memory();
  }
  [[nodiscard]] std::span<const std::byte> requests() const noexcept {
    return request_storage_.logical_memory();
  }
  [[nodiscard]] std::span<const std::byte> responses() const noexcept {
    return response_storage_.logical_memory();
  }
  [[nodiscard]] std::span<const std::byte> notify() const noexcept {
    return notify_storage_.logical_memory();
  }
  [[nodiscard]] std::span<std::byte> request_registration() noexcept {
    return request_storage_.registration_memory();
  }
  [[nodiscard]] std::span<std::byte> response_registration() noexcept {
    return response_storage_.registration_memory();
  }
  [[nodiscard]] std::span<std::byte> notify_registration() noexcept {
    return notify_storage_.registration_memory();
  }
  [[nodiscard]] bool is_shared() const noexcept {
    return request_storage_.is_shared() && response_storage_.is_shared() &&
           notify_storage_.is_shared();
  }
  [[nodiscard]] std::string_view request_shared_memory_name() const noexcept {
    return request_storage_.shared_memory_name();
  }
  [[nodiscard]] std::string_view response_shared_memory_name() const noexcept {
    return response_storage_.shared_memory_name();
  }
  [[nodiscard]] std::string_view notify_shared_memory_name() const noexcept {
    return notify_storage_.shared_memory_name();
  }
  [[nodiscard]] Status UnlinkSharedMemory();

  [[nodiscard]] segment_offset_t
  RequestOffset(std::uint32_t source_rank,
                std::uint32_t slot_id) const noexcept;
  [[nodiscard]] segment_offset_t
  ResponseOffset(std::uint32_t producer_rank,
                 std::uint32_t slot_id) const noexcept;
  [[nodiscard]] segment_offset_t
  ResultOffset(std::uint32_t destination_rank,
               std::uint32_t slot_id) const noexcept;
  [[nodiscard]] segment_offset_t
  RequestNotifyOffset(std::uint32_t source_rank,
                      std::uint32_t slot_id) const noexcept;
  [[nodiscard]] segment_offset_t
  ResponseNotifyOffset(std::uint32_t producer_rank,
                       std::uint32_t slot_id) const noexcept;

  [[nodiscard]] std::span<std::byte>
  RequestSlot(std::uint32_t source_rank, std::uint32_t slot_id) noexcept;
  [[nodiscard]] std::span<std::byte>
  ResponseSlot(std::uint32_t producer_rank, std::uint32_t slot_id) noexcept;
  [[nodiscard]] std::span<std::byte> ResultSlot(std::uint32_t destination_rank,
                                                std::uint32_t slot_id) noexcept;
  [[nodiscard]] std::span<std::byte>
  RequestStaging(std::uint32_t rpc_thread, std::uint32_t slot_id) noexcept;
  [[nodiscard]] std::span<std::byte>
  RequestStagingRegion(std::uint32_t rpc_thread) noexcept;
  [[nodiscard]] std::uint64_t *NotifyAddress(segment_offset_t offset) noexcept;
  [[nodiscard]] const std::uint64_t *
  NotifyAddress(segment_offset_t offset) const noexcept;
  [[nodiscard]] std::uint64_t logical_bytes() const noexcept {
    return logical_bytes_;
  }

private:
  RpcMailboxMemory() = default;

  RpcMailboxConfig config_{};
  std::uint64_t request_stride_{0};
  std::uint64_t response_stride_{0};
  std::uint64_t request_offset_{0};
  std::uint64_t request_staging_offset_{0};
  std::uint64_t response_offset_{0};
  std::uint64_t result_offset_{0};
  std::uint64_t request_notify_offset_{0};
  std::uint64_t response_notify_offset_{0};
  std::uint64_t logical_bytes_{0};
  PageAlignedBuffer request_storage_;
  PageAlignedBuffer response_storage_;
  PageAlignedBuffer notify_storage_;
};

} // namespace hnsw_urma
