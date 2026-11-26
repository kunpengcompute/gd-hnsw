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

#include "hnsw_urma/search/query_slot_pool.hpp"

#include <limits>
#include <new>

namespace hnsw_urma {

Result<QuerySlotPool> QuerySlotPool::Create(std::uint32_t source_rank,
                                            std::uint32_t capacity,
                                            std::uint64_t generation_base) {
  if (capacity == 0 ||
      generation_base == std::numeric_limits<std::uint64_t>::max()) {
    return Status(StatusCode::kInvalidArgument,
                  "query slot pool capacity/base generation is invalid");
  }
  try {
    QuerySlotPool pool;
    pool.source_rank_ = source_rank;
    pool.slots_.resize(capacity);
    for (auto &slot : pool.slots_) {
      slot.generation = generation_base;
    }
    return pool;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate query slot pool");
  }
}

QuerySlotPool::QuerySlotPool(QuerySlotPool &&other) noexcept {
  source_rank_ = other.source_rank_;
  slots_ = std::move(other.slots_);
  cursor_ = other.cursor_;
}

QuerySlotPool &QuerySlotPool::operator=(QuerySlotPool &&other) noexcept {
  if (this != &other) {
    source_rank_ = other.source_rank_;
    slots_ = std::move(other.slots_);
    cursor_ = other.cursor_;
  }
  return *this;
}

Result<QueryKey> QuerySlotPool::Acquire() {
  for (std::uint32_t offset = 0; offset < slots_.size(); ++offset) {
    const std::uint32_t index =
        (cursor_ + offset) % static_cast<std::uint32_t>(slots_.size());
    auto &slot = slots_[index];
    if (slot.active) {
      continue;
    }
    if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
      return Status(StatusCode::kResourceExhausted,
                    "query slot generation exhausted");
    }
    ++slot.generation;
    slot.active = true;
    cursor_ = (index + 1) % static_cast<std::uint32_t>(slots_.size());
    return QueryKey{source_rank_, index, slot.generation};
  }
  return Status(StatusCode::kResourceExhausted, "no query slot is available");
}

Status QuerySlotPool::Release(QueryKey key) {
  if (key.source_rank != source_rank_ || key.query_slot >= slots_.size()) {
    return Status(StatusCode::kProtocolError,
                  "query slot release key is outside this pool");
  }
  auto &slot = slots_[key.query_slot];
  if (!slot.active || slot.generation != key.generation) {
    return Status(StatusCode::kProtocolError,
                  "stale or duplicate query slot release");
  }
  slot.active = false;
  return Status::Ok();
}

} // namespace hnsw_urma
