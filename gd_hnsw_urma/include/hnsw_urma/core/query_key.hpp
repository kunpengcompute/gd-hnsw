#pragma once

#include <cstddef>
#include <cstdint>

namespace hnsw_urma {

struct QueryKey {
  std::uint32_t source_rank{0};
  std::uint32_t query_slot{0};
  std::uint64_t generation{0};

  friend bool operator==(const QueryKey&, const QueryKey&) = default;
};
static_assert(sizeof(QueryKey) == 16);
static_assert(alignof(QueryKey) == 8);

[[nodiscard]] std::uint64_t StableHash(const QueryKey& query,
                                       std::uint32_t destination_rank) noexcept;
[[nodiscard]] std::uint64_t StableHash(std::uint32_t source_rank,
                                       std::uint32_t source_lane) noexcept;

}  // namespace hnsw_urma
