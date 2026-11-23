#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

inline constexpr std::size_t kCacheLineBytes = 64;

struct SpscRingMetrics {
  std::uint64_t depth{0};
  std::uint64_t high_watermark{0};
  std::uint64_t full_count{0};
};

// Fixed-capacity SPSC ring. Reserve/CancelReservation/PushReserved are
// producer- only operations; TryPop is consumer-only. The cache-line separated
// monotonic cursors make every slot reusable only after the consumer publishes
// its head.
template <typename T> class SpscRing {
public:
  SpscRing() = default;
  SpscRing(const SpscRing &) = delete;
  SpscRing &operator=(const SpscRing &) = delete;

  SpscRing(SpscRing &&other) noexcept { MoveFrom(std::move(other)); }
  SpscRing &operator=(SpscRing &&other) noexcept {
    if (this != &other) {
      MoveFrom(std::move(other));
    }
    return *this;
  }

  [[nodiscard]] static Result<SpscRing> Create(std::size_t capacity) {
    if (capacity == 0) {
      return Status(StatusCode::kInvalidArgument,
                    "SPSC ring capacity must be non-zero");
    }
    try {
      SpscRing ring;
      ring.capacity_ = capacity;
      ring.entries_ = std::make_unique<T[]>(capacity);
      return ring;
    } catch (const std::bad_alloc &) {
      return Status(StatusCode::kResourceExhausted,
                    "unable to allocate SPSC ring");
    }
  }

  // Capacity reservation is local to the sole producer. It prevents a phase
  // from publishing only a prefix of its descriptors.
  [[nodiscard]] Status Reserve(std::size_t count) noexcept {
    if (count == 0 || entries_ == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "SPSC reservation must be non-zero");
    }
    const auto tail = producer_.tail.load(std::memory_order_relaxed);
    const auto head = consumer_.head.load(std::memory_order_acquire);
    const auto used = tail - head;
    if (used > capacity_ || count > capacity_ - used - producer_.reserved) {
      producer_.full_count.fetch_add(1, std::memory_order_relaxed);
      return Status(StatusCode::kResourceExhausted,
                    "SPSC ring reservation exceeds available capacity");
    }
    producer_.reserved += count;
    return Status::Ok();
  }

  [[nodiscard]] Status CancelReservation(std::size_t count) noexcept {
    if (count == 0 || count > producer_.reserved) {
      return Status(StatusCode::kProtocolError,
                    "SPSC reservation rollback exceeds reserved capacity");
    }
    producer_.reserved -= count;
    return Status::Ok();
  }

  template <typename U> [[nodiscard]] Status PushReserved(U &&value) noexcept {
    if (producer_.reserved == 0 || entries_ == nullptr) {
      return Status(StatusCode::kProtocolError,
                    "SPSC push has no producer reservation");
    }
    const auto tail = producer_.tail.load(std::memory_order_relaxed);
    entries_[tail % capacity_] = std::forward<U>(value);
    --producer_.reserved;
    producer_.tail.store(tail + 1, std::memory_order_release);
    UpdateHighWatermark(tail + 1);
    return Status::Ok();
  }

  template <typename U> [[nodiscard]] Status TryPush(U &&value) noexcept {
    auto reserved = Reserve(1);
    if (!reserved.ok()) {
      return reserved;
    }
    return PushReserved(std::forward<U>(value));
  }

  [[nodiscard]] std::optional<T> TryPop() noexcept {
    if (entries_ == nullptr) {
      return std::nullopt;
    }
    const auto head = consumer_.head.load(std::memory_order_relaxed);
    const auto tail = producer_.tail.load(std::memory_order_acquire);
    if (head == tail) {
      return std::nullopt;
    }
    T value = std::move(entries_[head % capacity_]);
    consumer_.head.store(head + 1, std::memory_order_release);
    return value;
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t producer_reserved() const noexcept {
    return producer_.reserved;
  }
  [[nodiscard]] std::size_t size() const noexcept {
    const auto tail = producer_.tail.load(std::memory_order_acquire);
    const auto head = consumer_.head.load(std::memory_order_acquire);
    return static_cast<std::size_t>(tail - head);
  }
  [[nodiscard]] std::size_t available() const noexcept {
    const auto depth = size();
    return depth > capacity_ ? 0 : capacity_ - depth;
  }
  [[nodiscard]] SpscRingMetrics metrics() const noexcept {
    return {size(), producer_.high_watermark.load(std::memory_order_relaxed),
            producer_.full_count.load(std::memory_order_relaxed)};
  }

private:
  struct alignas(kCacheLineBytes) ProducerState {
    std::atomic<std::uint64_t> tail{0};
    std::size_t reserved{0};
    std::atomic<std::uint64_t> high_watermark{0};
    std::atomic<std::uint64_t> full_count{0};
  };

  struct alignas(kCacheLineBytes) ConsumerState {
    std::atomic<std::uint64_t> head{0};
  };

  void UpdateHighWatermark(std::uint64_t published_tail) noexcept {
    const auto head = consumer_.head.load(std::memory_order_acquire);
    const auto depth = published_tail - head;
    auto observed = producer_.high_watermark.load(std::memory_order_relaxed);
    while (observed < depth && !producer_.high_watermark.compare_exchange_weak(
                                   observed, depth, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
    }
  }

  void MoveFrom(SpscRing &&other) noexcept {
    entries_ = std::move(other.entries_);
    capacity_ = std::exchange(other.capacity_, 0);
    producer_.tail.store(other.producer_.tail.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    producer_.reserved = std::exchange(other.producer_.reserved, 0);
    producer_.high_watermark.store(
        other.producer_.high_watermark.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    producer_.full_count.store(
        other.producer_.full_count.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    consumer_.head.store(other.consumer_.head.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    other.producer_.tail.store(0, std::memory_order_relaxed);
    other.producer_.high_watermark.store(0, std::memory_order_relaxed);
    other.producer_.full_count.store(0, std::memory_order_relaxed);
    other.consumer_.head.store(0, std::memory_order_relaxed);
  }

  std::unique_ptr<T[]> entries_;
  std::size_t capacity_{0};
  ProducerState producer_;
  ConsumerState consumer_;
};

} // namespace hnsw_urma
