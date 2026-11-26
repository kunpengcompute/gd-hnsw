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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <semaphore>
#include <stop_token>

namespace hnsw_urma {

class CountingNotification {
public:
  CountingNotification() = default;
  CountingNotification(const CountingNotification &) = delete;
  CountingNotification &operator=(const CountingNotification &) = delete;

  void Notify(std::uint32_t count = 1) noexcept;
  [[nodiscard]] bool TryConsume() noexcept;
  [[nodiscard]] bool Wait(std::stop_token stop);
  [[nodiscard]] bool WaitUntil(std::stop_token stop,
                               std::chrono::steady_clock::time_point deadline);
  [[nodiscard]] std::uint64_t pending() const noexcept {
    return pending_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t notifications() const noexcept {
    return notifications_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t empty_wakeups() const noexcept {
    return empty_wakeups_.load(std::memory_order_relaxed);
  }

private:
  std::counting_semaphore<> semaphore_{0};
  std::atomic<std::uint64_t> pending_{0};
  std::atomic<std::uint64_t> notifications_{0};
  std::atomic<std::uint64_t> empty_wakeups_{0};
};

} // namespace hnsw_urma
