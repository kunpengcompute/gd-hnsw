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
#include "hnsw_urma/runtime/scheduler.hpp"
#include "test_framework.hpp"
namespace hnsw_urma::test {
inline std::optional<QueryRunnableHandle>
TakeRunnable(SchedulerQueues &queues, std::uint32_t worker,
             std::chrono::steady_clock::time_point now) {
  auto input = queues.PopSearchInput(worker, now);
  if (!input)
    return std::nullopt;
  UB_REQUIRE_TRUE(std::holds_alternative<QueryRunnableHandle>(*input));
  return std::get<QueryRunnableHandle>(*input);
}
} // namespace hnsw_urma::test
