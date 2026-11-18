#pragma once

#include <cstdint>
#include <limits>

namespace hnsw_urma {

using node_id_t = std::uint32_t;
inline constexpr node_id_t INVALID_NODE_ID = std::numeric_limits<node_id_t>::max();

using shard_id_t = std::uint32_t;
using file_offset_t = std::uint64_t;
using segment_offset_t = std::uint64_t;
using ticket_t = std::uint64_t;

}  // namespace hnsw_urma
