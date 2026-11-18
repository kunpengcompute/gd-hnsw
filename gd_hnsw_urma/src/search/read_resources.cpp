#include "hnsw_urma/search/read_resources.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <new>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
Result<std::size_t> SearchReadResources::RequiredBytes(
    std::uint32_t slots, std::size_t slot_bytes) noexcept {
  if (slots == 0 || slot_bytes == 0)
    return Status(StatusCode::kInvalidArgument, "Search READ pool bounds must be non-zero");
  auto stride = CheckedAlignUp<std::size_t>(slot_bytes, 64);
  if (!stride.ok()) return stride.status();
  return CheckedMultiply<std::size_t>(stride.value(), slots);
}

Result<std::unique_ptr<SearchReadResources>> SearchReadResources::Create(
    std::uint32_t owner_search, std::span<std::byte> registered_memory,
    std::uint32_t slots, std::size_t slot_bytes) {
  auto required = RequiredBytes(slots, slot_bytes);
  if (!required.ok()) return required.status();
  if (registered_memory.size() < required.value() ||
      reinterpret_cast<std::uintptr_t>(registered_memory.data()) % 64 != 0)
    return Status(StatusCode::kInvalidArgument, "Search READ memory is too small or misaligned");
  try {
    auto pool = std::make_unique<SearchReadResources>();
    pool->owner_search_ = owner_search;
    pool->slot_count_ = slots;
    pool->slot_bytes_ = required.value() / slots;
    pool->memory_ = registered_memory.first(required.value());
    pool->free_slots_.reserve(slots);
    for (auto slot = slots; slot != 0; --slot) pool->free_slots_.push_back(slot - 1);
    return pool;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate Search READ pool");
  }
}

Status SearchReadResources::BindLanes(std::span<ReadBatchLane* const> lanes) {
  for (auto* lane : lanes)
    if (lane == nullptr || owner_search_ >= lane->config().search_workers)
      return Status(StatusCode::kInvalidArgument, "READ pool owner is outside lane topology");
  lanes_.assign(lanes.begin(), lanes.end());
  return Status::Ok();
}

Status SearchReadResources::Acquire(std::span<ReadResultView> results,
                                    std::size_t bytes) noexcept {
  assert(bytes != 0 && bytes <= slot_bytes_);
  if (free_slots_.size() < results.size()) {
    ++profile_.allocation_retries;
    return Status(StatusCode::kResourceExhausted, "Search READ result pool is full");
  }
  for (auto& view : results) {
    const auto slot = free_slots_.back();
    free_slots_.pop_back();
    view = {memory_.data() + slot * slot_bytes_, bytes, slot};
  }
  profile_.slots_in_use = slot_count_ - free_slots_.size();
  profile_.slots_high_watermark = std::max(profile_.slots_high_watermark, profile_.slots_in_use);
  return Status::Ok();
}

void SearchReadResources::Return(std::span<const ReadResultView> results) noexcept {
  for (const auto& view : results) {
    assert(view.slot < slot_count_ && view.data == memory_.data() + view.slot * slot_bytes_);
    free_slots_.push_back(view.slot);
  }
  assert(free_slots_.size() <= slot_count_);
  profile_.slots_in_use = slot_count_ - free_slots_.size();
}

void SearchReadResources::ReturnRecord() noexcept {
  assert(records_in_use_ != 0);
  --records_in_use_;
}

std::size_t SearchReadResources::PollReclaims(std::size_t budget) noexcept {
  std::size_t received = 0, empty = 0;
  while (received < budget && empty < lanes_.size()) {
    auto* lane = lanes_[reclaim_cursor_++ % lanes_.size()];
    auto* batch = lane->PopReclaim(owner_search_);
    if (batch == nullptr) { ++empty; continue; }
    empty = 0;
    ++received;
    ++profile_.reclaim_events;
    assert(batch->reclaim_required && !batch->reclaim_received);
    batch->reclaim_received = true;
    const auto callback = batch->reclaim_callback;
    auto* context = batch->reclaim_context;
    assert(callback != nullptr);
    callback(context, *batch);
  }
  return received;
}
}  // namespace hnsw_urma
