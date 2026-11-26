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

#include <cstdint>
#include <vector>

#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

// Scheduler/Control is the sole owner. Keeping allocation single-threaded
// avoids a global free-list mutex on admission and completion.
class QuerySlotPool {
public:
  [[nodiscard]] static Result<QuerySlotPool>
  Create(std::uint32_t source_rank, std::uint32_t capacity,
         std::uint64_t generation_base = 0);
  QuerySlotPool(QuerySlotPool &&other) noexcept;
  QuerySlotPool &operator=(QuerySlotPool &&other) noexcept;
  QuerySlotPool(const QuerySlotPool &) = delete;
  QuerySlotPool &operator=(const QuerySlotPool &) = delete;

  [[nodiscard]] Result<QueryKey> Acquire();
  [[nodiscard]] Status Release(QueryKey key);
  [[nodiscard]] std::uint32_t capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }

private:
  QuerySlotPool() = default;

  struct Slot {
    std::uint64_t generation{0};
    bool active{false};
  };

  std::uint32_t source_rank_{0};
  std::vector<Slot> slots_;
  std::uint32_t cursor_{0};
};

} // namespace hnsw_urma
