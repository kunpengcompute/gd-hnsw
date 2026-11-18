#include "hnsw_urma/core/visited_table.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

namespace hnsw_urma {

Result<VisitedTable> VisitedTable::Create(std::uint64_t ntotal, std::size_t capacity,
                                          std::size_t max_probe, double max_load_factor) {
  if (ntotal == 0 || ntotal > std::numeric_limits<node_id_t>::max()) {
    return Status(StatusCode::kInvalidArgument, "visited ntotal is outside node_id_t range");
  }
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    return Status(StatusCode::kInvalidArgument, "visited capacity must be a non-zero power of two");
  }
  if (!(max_load_factor > 0.0 && max_load_factor <= 1.0)) {
    return Status(StatusCode::kInvalidArgument, "visited max load factor must be in (0, 1]");
  }
  if (max_probe == 0) {
    max_probe = capacity;
  }
  if (max_probe > capacity) {
    return Status(StatusCode::kInvalidArgument, "visited max probe cannot exceed capacity");
  }
  const auto max_entries = static_cast<std::size_t>(
      std::floor(static_cast<double>(capacity) * max_load_factor));
  if (max_entries == 0) {
    return Status(StatusCode::kInvalidArgument, "visited load factor permits no entries");
  }

  try {
    VisitedTable table(ntotal, VisitedLayout::kSparseEpoch, max_probe,
                       max_entries);
    table.sparse_slots_.assign(capacity, 0);
    return table;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate visited table");
  }
}

Result<VisitedTable> VisitedTable::CreateAdaptive(
    std::uint64_t ntotal, std::size_t sparse_capacity,
    std::uint32_t active_slots, std::uint64_t dense_total_budget_bytes,
    std::size_t max_probe, double max_load_factor) {
  if (ntotal == 0 || ntotal > std::numeric_limits<node_id_t>::max() ||
      active_slots == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "adaptive visited dimensions are invalid");
  }
  const bool dense_fits =
      ntotal <= dense_total_budget_bytes / active_slots;
  if (!dense_fits) {
    return Create(ntotal, sparse_capacity, max_probe, max_load_factor);
  }
  if (ntotal > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kResourceExhausted,
                  "dense visited table exceeds addressable memory");
  }
  try {
    VisitedTable table(
        ntotal, VisitedLayout::kDenseGeneration, 1,
        static_cast<std::size_t>(ntotal));
    table.dense_stamps_.assign(static_cast<std::size_t>(ntotal), 0);
    return table;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate dense visited table");
  }
}

Result<bool> VisitedTable::Mark(node_id_t id) noexcept {
  if (id == INVALID_NODE_ID || static_cast<std::uint64_t>(id) >= ntotal_) {
    return Status(StatusCode::kOutOfRange, "visited ID is outside the index");
  }

  if (layout_ == VisitedLayout::kDenseGeneration) {
    auto& stamp = dense_stamps_[id];
    if (stamp == dense_generation_) {
      return false;
    }
    stamp = dense_generation_;
    ++size_;
    probe_high_watermark_ = 1;
    return true;
  }

  const std::size_t initial = InitialSlot(id);
  for (std::size_t probe = 0; probe < max_probe_; ++probe) {
    const std::size_t slot =
        (initial + probe) & (sparse_slots_.size() - 1);
    const auto value = sparse_slots_[slot];
    if (value == Packed(id)) {
      probe_high_watermark_ = std::max(probe_high_watermark_, probe + 1);
      return false;
    }
    if (static_cast<std::uint32_t>(value >> 32U) != sparse_epoch_) {
      if (size_ >= max_entries_) {
        return Status(StatusCode::kResourceExhausted, "visited table load limit reached");
      }
      sparse_slots_[slot] = Packed(id);
      ++size_;
      probe_high_watermark_ = std::max(probe_high_watermark_, probe + 1);
      return true;
    }
  }
  return Status(StatusCode::kResourceExhausted, "visited maximum probe length reached");
}

Result<bool> VisitedTable::Contains(node_id_t id) const noexcept {
  if (id == INVALID_NODE_ID || static_cast<std::uint64_t>(id) >= ntotal_) {
    return Status(StatusCode::kOutOfRange, "visited ID is outside the index");
  }
  if (layout_ == VisitedLayout::kDenseGeneration) {
    return dense_stamps_[id] == dense_generation_;
  }
  const std::size_t initial = InitialSlot(id);
  for (std::size_t probe = 0; probe < max_probe_; ++probe) {
    const std::size_t slot =
        (initial + probe) & (sparse_slots_.size() - 1);
    const auto value = sparse_slots_[slot];
    if (value == Packed(id)) {
      return true;
    }
    if (static_cast<std::uint32_t>(value >> 32U) != sparse_epoch_) {
      return false;
    }
  }
  return false;
}

void VisitedTable::Prefetch(node_id_t id) const noexcept {
#if defined(__GNUC__) || defined(__clang__)
  if (id == INVALID_NODE_ID ||
      static_cast<std::uint64_t>(id) >= ntotal_) {
    return;
  }
  if (layout_ == VisitedLayout::kDenseGeneration) {
    __builtin_prefetch(dense_stamps_.data() + id, 1, 1);
  } else {
    __builtin_prefetch(sparse_slots_.data() + InitialSlot(id), 1, 1);
  }
#else
  static_cast<void>(id);
#endif
}

void VisitedTable::Clear() noexcept {
  if (layout_ == VisitedLayout::kDenseGeneration) {
    ++dense_generation_;
    if (dense_generation_ == 0) {
      std::fill(dense_stamps_.begin(), dense_stamps_.end(), 0);
      dense_generation_ = 1;
    }
  } else {
    ++sparse_epoch_;
    if (sparse_epoch_ == 0) {
      std::fill(sparse_slots_.begin(), sparse_slots_.end(), 0);
      sparse_epoch_ = 1;
    }
  }
  size_ = 0;
  probe_high_watermark_ = 0;
}

std::size_t VisitedTable::capacity() const noexcept {
  return layout_ == VisitedLayout::kDenseGeneration
             ? dense_stamps_.size()
             : sparse_slots_.size();
}

std::size_t VisitedTable::InitialSlot(node_id_t id) const noexcept {
  constexpr std::uint32_t kFibonacciHash = 2654435769U;
  const auto value = static_cast<std::uint32_t>(id) * kFibonacciHash;
  return static_cast<std::size_t>(value) & (sparse_slots_.size() - 1);
}

std::uint64_t VisitedTable::Packed(node_id_t id) const noexcept {
  return (static_cast<std::uint64_t>(sparse_epoch_) << 32U) |
         static_cast<std::uint64_t>(id);
}

}  // namespace hnsw_urma
