#pragma once

#include "hnsw_urma/search/query_context.hpp"
#include <cstdint>
#include <type_traits>
#include <variant>

namespace hnsw_urma {

struct RpcReclaimHandle {
  std::uint32_t owner_search{0};
  std::uint32_t operation_slot{0};
  ticket_t ticket{0};
};
using SearchEvent = std::variant<QueryReadyHandle, RpcReclaimHandle>;
static_assert(std::is_nothrow_copy_assignable_v<SearchEvent>);

} // namespace hnsw_urma
