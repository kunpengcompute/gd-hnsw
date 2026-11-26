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
#include <span>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

struct ShardRange {
  shard_id_t shard_id{0};
  node_id_t global_begin{0};
  std::uint64_t local_count{0};

  friend bool operator==(const ShardRange &, const ShardRange &) = default;
};

struct NodeLocation {
  shard_id_t shard_id{0};
  std::uint64_t local_id{0};

  friend bool operator==(const NodeLocation &, const NodeLocation &) = default;
};

class ShardMap {
public:
  [[nodiscard]] static Result<ShardMap>
  Create(std::span<const ShardRange> ranges, std::uint64_t ntotal);

  [[nodiscard]] Result<NodeLocation> Locate(node_id_t global_id) const noexcept;
  [[nodiscard]] Result<ShardRange> Range(shard_id_t shard_id) const noexcept;
  [[nodiscard]] std::span<const ShardRange> ranges() const noexcept {
    return ranges_;
  }
  [[nodiscard]] std::uint64_t ntotal() const noexcept { return ntotal_; }
  [[nodiscard]] bool uniform() const noexcept { return uniform_count_ != 0; }
  [[nodiscard]] std::uint64_t uniform_count() const noexcept {
    return uniform_count_;
  }

private:
  std::vector<ShardRange> ranges_;
  std::uint64_t ntotal_{0};
  std::uint64_t uniform_count_{0};
};

} // namespace hnsw_urma
