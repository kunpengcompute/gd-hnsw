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

#pragma once

#include <cstdint>
#include <limits>

namespace hnsw_urma {

using node_id_t = std::uint32_t;
inline constexpr node_id_t INVALID_NODE_ID =
    std::numeric_limits<node_id_t>::max();

using shard_id_t = std::uint32_t;
using file_offset_t = std::uint64_t;
using segment_offset_t = std::uint64_t;
using ticket_t = std::uint64_t;

} // namespace hnsw_urma
