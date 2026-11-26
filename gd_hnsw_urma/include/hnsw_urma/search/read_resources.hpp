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
#include <memory>
#include <span>
#include <vector>

#include "hnsw_urma/transport/read_batch_lane.hpp"

namespace hnsw_urma {

struct SearchReadResourceProfile {
  std::uint64_t slots_in_use{0};
  std::uint64_t slots_high_watermark{0};
  std::uint64_t allocation_retries{0};
  std::uint64_t reclaim_events{0};
};

// Its Search is the sole allocator and reclaimer. The caller owns registration
// and must retain memory until this pool and all transport engines are drained.
class SearchReadResources {
public:
  [[nodiscard]] static Result<std::size_t>
  RequiredBytes(std::uint32_t slots, std::size_t slot_bytes) noexcept;
  [[nodiscard]] static Result<std::unique_ptr<SearchReadResources>>
  Create(std::uint32_t owner_search, std::span<std::byte> registered_memory,
         std::uint32_t slots, std::size_t slot_bytes);
  [[nodiscard]] Status BindLanes(std::span<ReadBatchLane *const> lanes);
  [[nodiscard]] Status Acquire(std::span<ReadResultView> results,
                               std::size_t bytes) noexcept;
  void Return(std::span<const ReadResultView> results) noexcept;
  [[nodiscard]] std::size_t PollReclaims(std::size_t budget = 64) noexcept;
  void AcquireRecord() noexcept { ++records_in_use_; }
  void ReturnRecord() noexcept;
  [[nodiscard]] bool drained() const noexcept {
    return free_slots_.size() == slot_count_ && records_in_use_ == 0;
  }
  [[nodiscard]] std::size_t available() const noexcept {
    return free_slots_.size();
  }
  [[nodiscard]] std::uint32_t owner_search() const noexcept {
    return owner_search_;
  }
  [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
  [[nodiscard]] SearchReadResourceProfile profile() const noexcept {
    return profile_;
  }

private:
  std::uint32_t owner_search_{0};
  std::uint32_t slot_count_{0};
  std::size_t slot_bytes_{0};
  std::size_t records_in_use_{0};
  std::span<std::byte> memory_;
  std::vector<std::uint32_t> free_slots_;
  std::vector<ReadBatchLane *> lanes_;
  std::size_t reclaim_cursor_{0};
  SearchReadResourceProfile profile_;
};

} // namespace hnsw_urma
