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

#include "hnsw_urma/core/query_key.hpp"

namespace hnsw_urma {
namespace {

std::uint64_t Mix(std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

} // namespace

std::uint64_t StableHash(const QueryKey &query,
                         std::uint32_t destination_rank) noexcept {
  std::uint64_t hash =
      Mix((static_cast<std::uint64_t>(query.source_rank) << 32U) |
          query.query_slot);
  hash ^= Mix(query.generation + 0x9e3779b97f4a7c15ULL);
  hash ^=
      Mix(static_cast<std::uint64_t>(destination_rank) + 0xd6e8feb86659fd93ULL);
  return Mix(hash);
}

std::uint64_t StableHash(std::uint32_t source_rank,
                         std::uint32_t source_lane) noexcept {
  return Mix((static_cast<std::uint64_t>(source_rank) << 32U) | source_lane);
}

} // namespace hnsw_urma
