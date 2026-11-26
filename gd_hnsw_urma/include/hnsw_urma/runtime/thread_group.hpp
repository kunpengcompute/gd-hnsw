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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/runtime/counting_notification.hpp"

namespace hnsw_urma {

template <typename T> class BoundedQueue {
public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  [[nodiscard]] Status TryPush(T value) {
    std::lock_guard lock(mutex_);
    if (closed_) {
      return Status(StatusCode::kInvalidArgument, "bounded queue is closed");
    }
    if (queue_.size() >= capacity_) {
      return Status(StatusCode::kResourceExhausted, "bounded queue is full");
    }
    queue_.push_back(std::move(value));
    high_watermark_ = std::max(high_watermark_, queue_.size());
    ready_.notify_one();
    return Status::Ok();
  }

  [[nodiscard]] Result<T> Pop(std::stop_token stop) {
    std::unique_lock lock(mutex_);
    std::stop_callback wake(stop, [this] { ready_.notify_all(); });
    ready_.wait(lock, [this, &stop] {
      return closed_ || !queue_.empty() || stop.stop_requested();
    });
    if (queue_.empty()) {
      return Status(StatusCode::kResourceExhausted,
                    stop.stop_requested() ? "bounded queue pop stopped"
                                          : "bounded queue is closed");
    }
    T value = std::move(queue_.front());
    queue_.pop_front();
    return value;
  }

  void Close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    ready_.notify_all();
  }

  [[nodiscard]] std::size_t size() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
  }
  [[nodiscard]] std::size_t high_watermark() const {
    std::lock_guard lock(mutex_);
    return high_watermark_;
  }

private:
  std::size_t capacity_{0};
  mutable std::mutex mutex_;
  std::condition_variable_any ready_;
  std::deque<T> queue_;
  std::size_t high_watermark_{0};
  bool closed_{false};
};

using RuntimeLoop =
    std::function<Status(std::stop_token stop, std::uint32_t worker_id)>;

struct RuntimeCallbacks {
  RuntimeLoop control;
  RuntimeLoop search;
  RuntimeLoop distance;
  RuntimeLoop read_progress;
  RuntimeLoop rpc_progress;
};

class RuntimeThreadGroup {
public:
  RuntimeThreadGroup(const RuntimeThreadGroup &) = delete;
  RuntimeThreadGroup &operator=(const RuntimeThreadGroup &) = delete;
  ~RuntimeThreadGroup();

  [[nodiscard]] static Result<std::unique_ptr<RuntimeThreadGroup>>
  Start(const EffectiveRuntimeConfig &config, CpuBindMode bind_mode,
        RuntimeCallbacks callbacks);
  void RequestStop();
  void Join();
  [[nodiscard]] Status status() const;

private:
  RuntimeThreadGroup() = default;
  void Launch(std::string name, std::uint32_t worker_id, std::uint32_t cpu,
              RuntimeLoop loop);
  void RecordFailure(Status status, std::string role, std::uint32_t worker_id);

  CpuBindMode bind_mode_{CpuBindMode::kNone};
  std::stop_source stop_;
  std::vector<std::jthread> threads_;
  mutable std::mutex mutex_;
  Status status_;
};

[[nodiscard]] Status BindCurrentThread(std::uint32_t cpu);
void FirstTouch(std::span<std::byte> memory) noexcept;

} // namespace hnsw_urma
