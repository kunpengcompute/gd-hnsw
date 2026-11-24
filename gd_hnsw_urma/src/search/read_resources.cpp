#include "hnsw_urma/search/read_resources.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/core/log.hpp"

namespace hnsw_urma {
Result<std::size_t>
SearchReadResources::RequiredBytes(std::uint32_t slots,
                                   std::size_t slot_bytes) noexcept {
  if (slots == 0 || slot_bytes == 0)
    return Status(StatusCode::kInvalidArgument,
                  "Search READ pool bounds must be non-zero");
  auto stride = CheckedAlignUp<std::size_t>(slot_bytes, 64);
  if (!stride.ok())
    return stride.status();
  return CheckedMultiply<std::size_t>(stride.value(), slots);
}

Result<std::unique_ptr<SearchReadResources>>
SearchReadResources::Create(std::uint32_t owner_search,
                            std::span<std::byte> registered_memory,
                            std::uint32_t slots, std::size_t slot_bytes) {
  auto required = RequiredBytes(slots, slot_bytes);
  if (!required.ok())
    return required.status();
  if (registered_memory.size() < required.value() ||
      reinterpret_cast<std::uintptr_t>(registered_memory.data()) % 64 != 0)
    return Status(StatusCode::kInvalidArgument,
                  "Search READ memory is too small or misaligned");
  // Defensive guard: RequiredBytes already rejects slots == 0, but make the
  // non-zero divisor explicit here so the division cannot divide by zero
  // regardless of future refactor (G.EXP.22-CPP).
  if (slots == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "Search READ pool slots must be non-zero");
  }
  try {
    auto pool = std::make_unique<SearchReadResources>();
    pool->owner_search_ = owner_search;
    pool->slot_count_ = slots;
    pool->slot_bytes_ = required.value() / slots;
    pool->memory_ = registered_memory.first(required.value());
    pool->free_slots_.reserve(slots);
    for (auto slot = slots; slot != 0; --slot)
      pool->free_slots_.push_back(slot - 1);
    return pool;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate Search READ pool");
  }
}

Status SearchReadResources::BindLanes(std::span<ReadBatchLane *const> lanes) {
  for (auto *lane : lanes)
    if (lane == nullptr || owner_search_ >= lane->config().search_workers)
      return Status(StatusCode::kInvalidArgument,
                    "READ pool owner is outside lane topology");
  lanes_.assign(lanes.begin(), lanes.end());
  return Status::Ok();
}

Status SearchReadResources::Acquire(std::span<ReadResultView> results,
                                    std::size_t bytes) noexcept {
  if (bytes == 0 || bytes > slot_bytes_) {
    LOG_ERROR("SearchReadResources::Acquire invalid view size {} slot_bytes {}",
              bytes, slot_bytes_);
    return Status(StatusCode::kInvalidArgument,
                  "Search READ result view size is invalid");
  }
  if (free_slots_.size() < results.size()) {
    ++profile_.allocation_retries;
    return Status(StatusCode::kResourceExhausted,
                  "Search READ result pool is full");
  }
  for (auto &view : results) {
    const auto slot = free_slots_.back();
    free_slots_.pop_back();
    view = {memory_.data() + slot * slot_bytes_, bytes, slot};
  }
  profile_.slots_in_use = slot_count_ - free_slots_.size();
  profile_.slots_high_watermark =
      std::max(profile_.slots_high_watermark, profile_.slots_in_use);
  return Status::Ok();
}

void SearchReadResources::Return(
    std::span<const ReadResultView> results) noexcept {
  for (const auto &view : results) {
    if (view.slot >= slot_count_ ||
        view.data != memory_.data() + view.slot * slot_bytes_) {
      LOG_ERROR(
          "SearchReadResources::Return invalid view slot {} slot_count {}",
          view.slot, slot_count_);
      continue;
    }
    free_slots_.push_back(view.slot);
  }
  if (free_slots_.size() > slot_count_) {
    LOG_ERROR("SearchReadResources::Return free_slots {} exceeds slot_count {}",
              free_slots_.size(), slot_count_);
    return;
  }
  profile_.slots_in_use = slot_count_ - free_slots_.size();
}

void SearchReadResources::ReturnRecord() noexcept {
  if (records_in_use_ == 0) {
    LOG_ERROR(
        "SearchReadResources::ReturnRecord called with no records in use");
    return;
  }
  --records_in_use_;
}

std::size_t SearchReadResources::PollReclaims(std::size_t budget) noexcept {
  std::size_t received = 0, empty = 0;
  while (received < budget && empty < lanes_.size()) {
    auto *lane = lanes_[reclaim_cursor_++ % lanes_.size()];
    auto *batch = lane->PopReclaim(owner_search_);
    if (batch == nullptr) {
      ++empty;
      continue;
    }
    empty = 0;
    ++received;
    ++profile_.reclaim_events;
    if (!batch->reclaim_required || batch->reclaim_received) {
      LOG_ERROR("SearchReadResources::PollReclaims invalid reclaim state "
                "required={} received={}",
                batch->reclaim_required, batch->reclaim_received);
      continue;
    }
    batch->reclaim_received = true;
    const auto callback = batch->reclaim_callback;
    auto *context = batch->reclaim_context;
    if (callback == nullptr) {
      LOG_ERROR("SearchReadResources::PollReclaims reclaim callback is null");
      continue;
    }
    callback(context, *batch);
  }
  return received;
}
} // namespace hnsw_urma
