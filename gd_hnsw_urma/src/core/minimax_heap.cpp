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

#include "hnsw_urma/core/minimax_heap.hpp"

#include <algorithm>
#include <cmath>
#include <new>

namespace hnsw_urma {
namespace {

bool Better(const HeapEntry &lhs, const HeapEntry &rhs) noexcept {
  return lhs.distance < rhs.distance ||
         (lhs.distance == rhs.distance && lhs.id < rhs.id);
}

bool Worse(const HeapEntry &lhs, const HeapEntry &rhs) noexcept {
  return Better(rhs, lhs);
}

} // namespace

Result<MinimaxHeap> MinimaxHeap::Create(std::uint32_t capacity) {
  try {
    MinimaxHeap heap(capacity);
    heap.entries_.resize(capacity);
    heap.result_heap_.reserve(capacity);
    heap.candidate_heap_.reserve(static_cast<std::size_t>(capacity) * 2U + 1U);
    return heap;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate minimax heap");
  }
}

Result<bool> MinimaxHeap::Push(node_id_t id, float distance) {
  return PushImpl(id, distance, true);
}

Result<bool> MinimaxHeap::PushUnique(node_id_t id, float distance) {
  return PushImpl(id, distance, false);
}

Result<bool> MinimaxHeap::PushImpl(node_id_t id, float distance,
                                   bool check_duplicate) {
  if (id == INVALID_NODE_ID || !std::isfinite(distance)) {
    return Status(StatusCode::kInvalidArgument,
                  "heap entry requires a valid ID and finite distance");
  }
  if (capacity_ == 0) {
    return false;
  }

  const HeapEntry candidate{id, distance};
  if (result_heap_.size() == capacity_ &&
      !Better(candidate, entries_[result_heap_.front()].value)) {
    return false;
  }

  if (check_duplicate) {
    for (const EntryIndex index : result_heap_) {
      if (entries_[index].value.id == id) {
        return false;
      }
    }
  }

  const auto result_better = [this](EntryIndex lhs, EntryIndex rhs) {
    return Better(entries_[lhs].value, entries_[rhs].value);
  };
  const auto candidate_worse = [](const CandidateHandle &lhs,
                                  const CandidateHandle &rhs) {
    return Worse(lhs.value, rhs.value);
  };

  if (candidate_heap_.size() >= static_cast<std::size_t>(capacity_) * 2U) {
    CompactCandidates();
  }

  EntryIndex entry_index = 0;
  if (result_heap_.size() == capacity_) {
    std::pop_heap(result_heap_.begin(), result_heap_.end(), result_better);
    entry_index = result_heap_.back();
    result_heap_.pop_back();

    EntryState &evicted = entries_[entry_index];
    if (!evicted.expanded) {
      --current_candidate_count_;
    }
    evicted.in_result = false;
  } else {
    entry_index = static_cast<EntryIndex>(result_heap_.size());
  }

  EntryState &inserted = entries_[entry_index];
  ++inserted.generation;
  if (inserted.generation == 0) {
    ++inserted.generation;
  }
  inserted.value = candidate;
  inserted.in_result = true;
  inserted.expanded = false;

  result_heap_.push_back(entry_index);
  std::push_heap(result_heap_.begin(), result_heap_.end(), result_better);

  candidate_heap_.push_back(
      CandidateHandle{candidate, entry_index, inserted.generation});
  std::push_heap(candidate_heap_.begin(), candidate_heap_.end(),
                 candidate_worse);
  ++current_candidate_count_;
  return true;
}

std::optional<HeapEntry> MinimaxHeap::PopMinCandidate() {
  const auto candidate_worse = [](const CandidateHandle &lhs,
                                  const CandidateHandle &rhs) {
    return Worse(lhs.value, rhs.value);
  };
  while (!candidate_heap_.empty()) {
    std::pop_heap(candidate_heap_.begin(), candidate_heap_.end(),
                  candidate_worse);
    const CandidateHandle candidate = candidate_heap_.back();
    candidate_heap_.pop_back();
    if (IsCurrentCandidate(candidate)) {
      entries_[candidate.index].expanded = true;
      --current_candidate_count_;
      return candidate.value;
    }
  }
  return std::nullopt;
}

std::optional<HeapEntry> MinimaxHeap::max() const noexcept {
  if (result_heap_.empty()) {
    return std::nullopt;
  }
  return entries_[result_heap_.front()].value;
}

std::vector<HeapEntry> MinimaxHeap::ExtractSorted() const {
  std::vector<HeapEntry> sorted;
  sorted.reserve(result_heap_.size());
  for (const EntryIndex index : result_heap_) {
    sorted.push_back(entries_[index].value);
  }
  std::sort(sorted.begin(), sorted.end(), Better);
  return sorted;
}

void MinimaxHeap::Clear() noexcept {
  for (const EntryIndex index : result_heap_) {
    entries_[index].in_result = false;
    entries_[index].expanded = false;
  }
  result_heap_.clear();
  candidate_heap_.clear();
  current_candidate_count_ = 0;
}

bool MinimaxHeap::IsCurrentCandidate(
    const CandidateHandle &candidate) const noexcept {
  const EntryState &entry = entries_[candidate.index];
  return entry.generation == candidate.generation && entry.in_result &&
         !entry.expanded;
}

void MinimaxHeap::CompactCandidates() {
  auto end = std::remove_if(candidate_heap_.begin(), candidate_heap_.end(),
                            [this](const CandidateHandle &candidate) {
                              return !IsCurrentCandidate(candidate);
                            });
  candidate_heap_.erase(end, candidate_heap_.end());
  const auto candidate_worse = [](const CandidateHandle &lhs,
                                  const CandidateHandle &rhs) {
    return Worse(lhs.value, rhs.value);
  };
  std::make_heap(candidate_heap_.begin(), candidate_heap_.end(),
                 candidate_worse);
  current_candidate_count_ = candidate_heap_.size();
}

} // namespace hnsw_urma
