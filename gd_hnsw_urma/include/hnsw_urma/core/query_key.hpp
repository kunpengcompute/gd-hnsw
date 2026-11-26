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

namespace hnsw_urma {

struct QueryKey {
  std::uint32_t source_rank{0};
  std::uint32_t query_slot{0};
  std::uint64_t generation{0};

  friend bool operator==(const QueryKey &, const QueryKey &) = default;
};
static_assert(sizeof(QueryKey) == 16);
static_assert(alignof(QueryKey) == 8);

[[nodiscard]] std::uint64_t StableHash(const QueryKey &query,
                                       std::uint32_t destination_rank) noexcept;
[[nodiscard]] std::uint64_t StableHash(std::uint32_t source_rank,
                                       std::uint32_t source_lane) noexcept;

} // namespace hnsw_urma
