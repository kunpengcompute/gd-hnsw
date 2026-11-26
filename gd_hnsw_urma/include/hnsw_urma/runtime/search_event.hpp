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
