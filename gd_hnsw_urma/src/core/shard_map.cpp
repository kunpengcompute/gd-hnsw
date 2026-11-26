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

#include "hnsw_urma/core/shard_map.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {

Result<ShardMap> ShardMap::Create(std::span<const ShardRange> ranges,
                                  std::uint64_t ntotal) {
  if (ranges.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "shard map requires at least one range");
  }
  if (ntotal == 0 || ntotal > std::numeric_limits<node_id_t>::max()) {
    return Status(StatusCode::kInvalidArgument,
                  "ntotal must be in [1, UINT32_MAX] for IndexFormatV2");
  }

  std::uint64_t expected_begin = 0;
  bool uniform = true;
  const std::uint64_t first_count = ranges.front().local_count;
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const auto &range = ranges[index];
    if (range.shard_id != index) {
      return Status(StatusCode::kInvalidArgument,
                    "shard IDs must be dense and rank ordered")
          .AddContext("range_index", std::to_string(index));
    }
    if (range.local_count == 0 || range.global_begin != expected_begin) {
      return Status(StatusCode::kInvalidArgument,
                    "shard ranges must be non-empty, contiguous, and start at "
                    "zero")
          .AddContext("shard", std::to_string(range.shard_id));
    }
    auto next = CheckedAdd(expected_begin, range.local_count);
    if (!next.ok() || next.value() > ntotal) {
      return Status(StatusCode::kOutOfRange, "shard range exceeds ntotal")
          .AddContext("shard", std::to_string(range.shard_id));
    }
    expected_begin = next.value();
    uniform = uniform && range.local_count == first_count;
  }
  if (expected_begin != ntotal) {
    return Status(StatusCode::kInvalidArgument,
                  "shard ranges do not fully cover ntotal");
  }

  try {
    ShardMap map;
    map.ranges_.assign(ranges.begin(), ranges.end());
    map.ntotal_ = ntotal;
    map.uniform_count_ = uniform ? first_count : 0;
    return map;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate shard map");
  }
}

Result<NodeLocation> ShardMap::Locate(node_id_t global_id) const noexcept {
  if (global_id == INVALID_NODE_ID ||
      static_cast<std::uint64_t>(global_id) >= ntotal_) {
    return Status(StatusCode::kOutOfRange,
                  "global node ID is outside the index");
  }

  if (uniform_count_ != 0) {
    const auto shard = static_cast<std::uint64_t>(global_id) / uniform_count_;
    const auto local = static_cast<std::uint64_t>(global_id) % uniform_count_;
    return NodeLocation{static_cast<shard_id_t>(shard), local};
  }

  const auto it = std::upper_bound(ranges_.begin(), ranges_.end(), global_id,
                                   [](node_id_t id, const ShardRange &range) {
                                     return id < range.global_begin;
                                   });
  const auto owner = std::prev(it);
  return NodeLocation{owner->shard_id, static_cast<std::uint64_t>(global_id) -
                                           owner->global_begin};
}

Result<ShardRange> ShardMap::Range(shard_id_t shard_id) const noexcept {
  if (static_cast<std::size_t>(shard_id) >= ranges_.size()) {
    return Status(StatusCode::kOutOfRange, "shard ID is outside the shard map");
  }
  return ranges_[shard_id];
}

} // namespace hnsw_urma
