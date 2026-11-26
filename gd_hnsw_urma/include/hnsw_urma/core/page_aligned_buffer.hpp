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
#include <span>
#include <string>
#include <string_view>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

// Owns one allocation whose base address and allocated length are both aligned
// to the host page size. logical_memory() is the application-visible prefix;
// registration_memory() includes any zero-filled tail padding required by URMA.
class PageAlignedBuffer {
public:
  PageAlignedBuffer() = default;
  ~PageAlignedBuffer();
  PageAlignedBuffer(const PageAlignedBuffer &) = delete;
  PageAlignedBuffer &operator=(const PageAlignedBuffer &) = delete;
  PageAlignedBuffer(PageAlignedBuffer &&other) noexcept;
  PageAlignedBuffer &operator=(PageAlignedBuffer &&other) noexcept;

  [[nodiscard]] static Result<std::size_t> SystemPageSize();
  [[nodiscard]] static Result<PageAlignedBuffer>
  Create(std::size_t logical_bytes);
  [[nodiscard]] static Result<PageAlignedBuffer>
  CopyFrom(std::span<const std::byte> source);
  [[nodiscard]] static Result<PageAlignedBuffer>
  CreateShared(std::size_t logical_bytes);
  [[nodiscard]] static Result<PageAlignedBuffer>
  CopyFromShared(std::span<const std::byte> source);
  [[nodiscard]] static Result<PageAlignedBuffer>
  OpenShared(std::string_view name, std::size_t logical_bytes);

  [[nodiscard]] std::span<std::byte> logical_memory() noexcept;
  [[nodiscard]] std::span<const std::byte> logical_memory() const noexcept;
  [[nodiscard]] std::span<std::byte> registration_memory() noexcept;
  [[nodiscard]] std::span<const std::byte> registration_memory() const noexcept;
  // Call only after every peer has opened the exported name. Existing
  // mappings remain valid and the kernel reclaims the object after the last
  // mapping is released, including on abnormal process exit.
  [[nodiscard]] Status UnlinkSharedMemory();

  [[nodiscard]] std::size_t page_size() const noexcept { return page_size_; }
  [[nodiscard]] std::size_t logical_bytes() const noexcept {
    return logical_bytes_;
  }
  [[nodiscard]] std::size_t registered_bytes() const noexcept {
    return registered_bytes_;
  }
  [[nodiscard]] bool is_shared() const noexcept {
    return storage_ == Storage::kShared;
  }
  [[nodiscard]] std::string_view shared_memory_name() const noexcept {
    return shared_name_;
  }

private:
  enum class Storage : unsigned char { kEmpty, kHeap, kShared };

  void Reset() noexcept;

  std::byte *data_{nullptr};
  std::size_t logical_bytes_{0};
  std::size_t registered_bytes_{0};
  std::size_t page_size_{0};
  Storage storage_{Storage::kEmpty};
  bool shared_owner_{false};
  std::string shared_name_;
};

} // namespace hnsw_urma
