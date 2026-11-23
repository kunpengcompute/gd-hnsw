#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <thread>
#include <utility>

#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

struct MpmcRingMetrics {
  std::uint64_t depth{0};
  std::uint64_t high_watermark{0};
  std::uint64_t full_count{0};
};

// Fixed-capacity bounded MPMC ring. Producers and consumers claim monotonically
// increasing positions; the per-cell sequence prevents a later item from being
// consumed before an earlier claimed position has been published.
template <typename T> class MpmcRing {
public:
  MpmcRing() = default;
  MpmcRing(const MpmcRing &) = delete;
  MpmcRing &operator=(const MpmcRing &) = delete;

  MpmcRing(MpmcRing &&other) noexcept { MoveFrom(std::move(other)); }
  MpmcRing &operator=(MpmcRing &&other) noexcept {
    if (this != &other) {
      MoveFrom(std::move(other));
    }
    return *this;
  }

  [[nodiscard]] static Result<std::size_t> StorageBytes(std::size_t capacity) {
    return CheckedMultiply(capacity, sizeof(Cell));
  }
  [[nodiscard]] static Result<std::size_t> RequiredBytes(std::size_t capacity) {
    auto storage = StorageBytes(capacity);
    if (!storage.ok())
      return storage.status();
    return CheckedAdd(sizeof(MpmcRing), storage.value());
  }

  [[nodiscard]] static Result<MpmcRing> Create(std::size_t capacity) {
    if (capacity < 2) {
      return Status(StatusCode::kInvalidArgument,
                    "MPMC ring capacity must be at least two");
    }
    auto bytes = RequiredBytes(capacity);
    if (!bytes.ok())
      return bytes.status();
    try {
      MpmcRing ring;
      ring.capacity_ = capacity;
      ring.cells_ = std::make_unique<Cell[]>(capacity);
      for (std::size_t index = 0; index < capacity; ++index) {
        ring.cells_[index].sequence.store(index, std::memory_order_relaxed);
      }
      return ring;
    } catch (const std::bad_alloc &) {
      return Status(StatusCode::kResourceExhausted,
                    "unable to allocate MPMC ring");
    }
  }

  template <typename U> [[nodiscard]] Status TryPush(U &&value) noexcept {
    if (cells_ == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "MPMC ring is not initialized");
    }
    std::uint64_t position = 0;
    if (!ClaimEnqueuePosition(position, false)) {
      full_count_.fetch_add(1, std::memory_order_relaxed);
      return Status(StatusCode::kResourceExhausted, "MPMC ring is full");
    }
    PublishClaimed(position, std::forward<U>(value));
    return Status::Ok();
  }

  // The caller must already own one unit of capacity tracked outside this
  // ring. With multiple consumers, a later cell can be released before the
  // next FIFO cell, so an aggregate capacity credit does not guarantee that
  // TryPush can immediately reuse the next physical cell. Wait for that
  // transient release window instead of turning a valid reservation into a
  // false queue-full failure.
  template <typename U>
  [[nodiscard]] Status PushWithReservedCapacity(U &&value) noexcept {
    if (cells_ == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "MPMC ring is not initialized");
    }
    std::uint64_t position = 0;
    static_cast<void>(ClaimEnqueuePosition(position, true));
    PublishClaimed(position, std::forward<U>(value));
    return Status::Ok();
  }

  [[nodiscard]] std::optional<T> TryPop() noexcept {
    if (cells_ == nullptr) {
      return std::nullopt;
    }
    auto position = dequeue_position_.load(std::memory_order_relaxed);
    Cell *cell = nullptr;
    while (true) {
      cell = &cells_[position % capacity_];
      const auto sequence = cell->sequence.load(std::memory_order_acquire);
      const auto difference = static_cast<std::intptr_t>(sequence) -
                              static_cast<std::intptr_t>(position + 1U);
      if (difference == 0) {
        if (dequeue_position_.compare_exchange_weak(
                position, position + 1U, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          break;
        }
      } else if (difference < 0) {
        return std::nullopt;
      } else {
        position = dequeue_position_.load(std::memory_order_relaxed);
      }
    }
    T value = std::move(cell->value);
    cell->sequence.store(position + capacity_, std::memory_order_release);
    return value;
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept {
    const auto tail = enqueue_position_.load(std::memory_order_acquire);
    const auto head = dequeue_position_.load(std::memory_order_acquire);
    return tail >= head ? static_cast<std::size_t>(tail - head) : 0;
  }
  // Only while all producers/consumers are stopped and the ring is drained.
  void ResetMetrics() noexcept {
    high_watermark_.store(0, std::memory_order_relaxed);
    full_count_.store(0, std::memory_order_relaxed);
  }
  [[nodiscard]] MpmcRingMetrics metrics() const noexcept {
    return {size(), high_watermark_.load(std::memory_order_relaxed),
            full_count_.load(std::memory_order_relaxed)};
  }

private:
  struct Cell {
    std::atomic<std::uint64_t> sequence{0};
    T value{};
  };

  [[nodiscard]] bool
  ClaimEnqueuePosition(std::uint64_t &position,
                       bool wait_for_reserved_capacity) noexcept {
    position = enqueue_position_.load(std::memory_order_relaxed);
    while (true) {
      auto &cell = cells_[position % capacity_];
      const auto sequence = cell.sequence.load(std::memory_order_acquire);
      const auto difference = static_cast<std::intptr_t>(sequence) -
                              static_cast<std::intptr_t>(position);
      if (difference == 0) {
        if (enqueue_position_.compare_exchange_weak(
                position, position + 1U, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          return true;
        }
      } else if (difference < 0) {
        if (!wait_for_reserved_capacity) {
          return false;
        }
        std::this_thread::yield();
        position = enqueue_position_.load(std::memory_order_relaxed);
      } else {
        position = enqueue_position_.load(std::memory_order_relaxed);
      }
    }
  }

  template <typename U>
  void PublishClaimed(std::uint64_t position, U &&value) noexcept {
    auto &cell = cells_[position % capacity_];
    cell.value = std::forward<U>(value);
    cell.sequence.store(position + 1U, std::memory_order_release);
    UpdateHighWatermark(position + 1U);
  }

  void UpdateHighWatermark(std::uint64_t published_tail) noexcept {
    const auto head = dequeue_position_.load(std::memory_order_acquire);
    // Another producer may publish a later claimed position first, allowing
    // consumers to advance past this producer before it records the metric.
    // In that case this older publication cannot establish a new high-water
    // mark; avoid unsigned depth underflow.
    if (published_tail <= head) {
      return;
    }
    const auto depth = published_tail - head;
    auto observed = high_watermark_.load(std::memory_order_relaxed);
    while (observed < depth && !high_watermark_.compare_exchange_weak(
                                   observed, depth, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
    }
  }

  void MoveFrom(MpmcRing &&other) noexcept {
    cells_ = std::move(other.cells_);
    capacity_ = std::exchange(other.capacity_, 0);
    enqueue_position_.store(
        other.enqueue_position_.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    dequeue_position_.store(
        other.dequeue_position_.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    high_watermark_.store(other.high_watermark_.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
    full_count_.store(other.full_count_.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
    other.enqueue_position_.store(0, std::memory_order_relaxed);
    other.dequeue_position_.store(0, std::memory_order_relaxed);
    other.high_watermark_.store(0, std::memory_order_relaxed);
    other.full_count_.store(0, std::memory_order_relaxed);
  }

  std::unique_ptr<Cell[]> cells_;
  std::size_t capacity_{0};
  alignas(64) std::atomic<std::uint64_t> enqueue_position_{0};
  alignas(64) std::atomic<std::uint64_t> dequeue_position_{0};
  alignas(64) std::atomic<std::uint64_t> high_watermark_{0};
  std::atomic<std::uint64_t> full_count_{0};
};

} // namespace hnsw_urma
