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

#include "hnsw_urma/runtime/thread_group.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <new>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace hnsw_urma {

void CountingNotification::Notify(std::uint32_t count) noexcept {
  if (count == 0) {
    return;
  }
  pending_.fetch_add(count, std::memory_order_release);
  notifications_.fetch_add(count, std::memory_order_relaxed);
  semaphore_.release(static_cast<std::ptrdiff_t>(count));
}

bool CountingNotification::TryConsume() noexcept {
  if (!semaphore_.try_acquire()) {
    empty_wakeups_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  pending_.fetch_sub(1, std::memory_order_acq_rel);
  return true;
}

bool CountingNotification::Wait(std::stop_token stop) {
  while (!stop.stop_requested()) {
    if (semaphore_.try_acquire_for(std::chrono::milliseconds(10))) {
      pending_.fetch_sub(1, std::memory_order_acq_rel);
      return true;
    }
  }
  return false;
}

bool CountingNotification::WaitUntil(
    std::stop_token stop, std::chrono::steady_clock::time_point deadline) {
  while (!stop.stop_requested()) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return false;
    }
    const auto poll_deadline =
        std::min(deadline, now + std::chrono::milliseconds(10));
    if (semaphore_.try_acquire_until(poll_deadline)) {
      pending_.fetch_sub(1, std::memory_order_acq_rel);
      return true;
    }
  }
  return false;
}

Status BindCurrentThread(std::uint32_t cpu) {
#if defined(__linux__)
  const auto cpu_index = static_cast<std::size_t>(cpu);
  if (cpu_index >= static_cast<std::size_t>(CPU_SETSIZE)) {
    return Status(StatusCode::kOutOfRange, "CPU ID exceeds CPU_SETSIZE");
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_index, &set);
  const int result = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (result != 0) {
    return Status(StatusCode::kInternalError, "pthread_setaffinity_np failed")
        .AddContext("pthread_status", std::to_string(result))
        .AddContext("cpu", std::to_string(cpu));
  }
#else
  (void)cpu;
#endif
  return Status::Ok();
}

void FirstTouch(std::span<std::byte> memory) noexcept {
  constexpr std::size_t page_bytes = 4096;
  for (std::size_t offset = 0; offset < memory.size(); offset += page_bytes) {
    memory[offset] = std::byte{0};
  }
  if (!memory.empty()) {
    memory.back() = std::byte{0};
  }
  std::atomic_thread_fence(std::memory_order_release);
}

RuntimeThreadGroup::~RuntimeThreadGroup() {
  RequestStop();
  Join();
}

Result<std::unique_ptr<RuntimeThreadGroup>>
RuntimeThreadGroup::Start(const EffectiveRuntimeConfig &config,
                          CpuBindMode bind_mode, RuntimeCallbacks callbacks) {
  const bool common_complete =
      callbacks.control && callbacks.search && callbacks.distance &&
      config.search_cpus.size() == config.search_workers &&
      config.distance_cpus.size() == config.distance_workers &&
      !config.reserved_cpus.empty();
  const bool communication_complete =
      callbacks.read_progress && callbacks.rpc_progress &&
      config.read_progress_cpus.size() == config.read_progress_threads &&
      config.rpc_progress_cpus.size() == config.rpc_progress_threads;
  if (!common_complete || !communication_complete) {
    return Status(StatusCode::kInvalidArgument,
                  "thread callbacks or CPU assignments are incomplete");
  }
  try {
    auto group = std::unique_ptr<RuntimeThreadGroup>(new RuntimeThreadGroup());
    group->bind_mode_ = bind_mode;
    const std::size_t thread_count =
        1U + config.search_workers + config.distance_workers +
        config.read_progress_threads + config.rpc_progress_threads;
    group->threads_.reserve(thread_count);
    group->Launch("control", 0, config.reserved_cpus.front(),
                  std::move(callbacks.control));
    for (std::uint32_t worker = 0; worker < config.search_workers; ++worker) {
      group->Launch("search", worker, config.search_cpus[worker],
                    callbacks.search);
    }
    for (std::uint32_t worker = 0; worker < config.distance_workers; ++worker) {
      group->Launch("distance", worker, config.distance_cpus[worker],
                    callbacks.distance);
    }
    for (std::uint32_t thread = 0; thread < config.read_progress_threads;
         ++thread) {
      group->Launch("read-progress", thread, config.read_progress_cpus[thread],
                    callbacks.read_progress);
    }
    for (std::uint32_t thread = 0; thread < config.rpc_progress_threads;
         ++thread) {
      group->Launch("rpc-progress", thread, config.rpc_progress_cpus[thread],
                    callbacks.rpc_progress);
    }
    return group;
  } catch (const std::system_error &error) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to start runtime thread")
        .AddContext("system_error", error.what());
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate runtime thread group");
  }
}

void RuntimeThreadGroup::Launch(std::string name, std::uint32_t worker_id,
                                std::uint32_t cpu, RuntimeLoop loop) {
  threads_.emplace_back([this, name = std::move(name), worker_id, cpu,
                         loop = std::move(loop)](std::stop_token) mutable {
    if (bind_mode_ == CpuBindMode::kAuto) {
      const auto binding = BindCurrentThread(cpu);
      if (!binding.ok()) {
        // A failed pin must not take the runtime down. RecordFailure stops
        // EVERY thread of the group, so one unusable CPU id (e.g. the
        // container cpuset is narrower than the computed layout) used to
        // turn into a silent, total data-plane outage: every lane printed
        // its "started" line and then exited, the runtime still reported
        // rpc_ready=true and searches waited for distances that no thread
        // could ever post. Keep the thread unpinned and say why.
        fprintf(stderr,
                "[UB thread group] bind failed role=%s worker=%u cpu=%u: %s;"
                " continuing unpinned\n",
                name.c_str(), worker_id, cpu, binding.ToString().c_str());
      }
    }
    auto result = loop(stop_.get_token(), worker_id);
    if (!result.ok()) {
      RecordFailure(std::move(result), name, worker_id);
    }
  });
}

void RuntimeThreadGroup::RecordFailure(Status status, std::string role,
                                       std::uint32_t worker_id) {
  // Nothing else reports why the runtime went quiet: without this line a
  // stopped thread group looks exactly like a healthy one that never gets work.
  fprintf(stderr,
          "[UB thread group] thread failed role=%s worker=%u: %s;"
          " stopping every runtime thread\n",
          role.c_str(), worker_id, status.ToString().c_str());
  {
    std::lock_guard lock(mutex_);
    if (status_.ok()) {
      status.AddContext("thread_role", std::move(role));
      status.AddContext("worker_id", std::to_string(worker_id));
      status_ = std::move(status);
    }
  }
  stop_.request_stop();
}

void RuntimeThreadGroup::RequestStop() { stop_.request_stop(); }

void RuntimeThreadGroup::Join() {
  for (auto &thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  threads_.clear();
}

Status RuntimeThreadGroup::status() const {
  std::lock_guard lock(mutex_);
  return status_;
}

} // namespace hnsw_urma
