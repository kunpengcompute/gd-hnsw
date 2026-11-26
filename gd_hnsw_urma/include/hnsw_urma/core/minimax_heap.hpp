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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

struct HeapEntry {
  node_id_t id{INVALID_NODE_ID};
  float distance{0.0F};

  friend bool operator==(const HeapEntry &, const HeapEntry &) = default;
};

// Fixed-capacity HNSW candidate/result pair. The caller's VisitedTable owns
// global duplicate suppression; this type also rejects a duplicate still in
// the bounded result set. Candidate handles use generation checks so evicted
// entries can be discarded lazily without scanning the result heap.
class MinimaxHeap {
public:
  [[nodiscard]] static Result<MinimaxHeap> Create(std::uint32_t capacity);

  [[nodiscard]] Result<bool> Push(node_id_t id, float distance);
  // The caller must guarantee that id is not already in the current result
  // set. This avoids the linear duplicate scan performed by Push().
  [[nodiscard]] Result<bool> PushUnique(node_id_t id, float distance);
  [[nodiscard]] std::optional<HeapEntry> PopMinCandidate();
  [[nodiscard]] std::optional<HeapEntry> max() const noexcept;
  [[nodiscard]] std::vector<HeapEntry> ExtractSorted() const;
  void Clear() noexcept;

  [[nodiscard]] std::size_t result_size() const noexcept {
    return result_heap_.size();
  }
  [[nodiscard]] std::size_t candidate_size() const noexcept {
    return current_candidate_count_;
  }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

private:
  using EntryIndex = std::uint32_t;

  struct EntryState {
    HeapEntry value{};
    std::uint64_t generation{0};
    bool in_result{false};
    bool expanded{false};
  };

  struct CandidateHandle {
    HeapEntry value{};
    EntryIndex index{0};
    std::uint64_t generation{0};
  };

  explicit MinimaxHeap(std::uint32_t capacity) : capacity_(capacity) {}

  [[nodiscard]] Result<bool> PushImpl(node_id_t id, float distance,
                                      bool check_duplicate);
  [[nodiscard]] bool
  IsCurrentCandidate(const CandidateHandle &candidate) const noexcept;
  void CompactCandidates();

  std::uint32_t capacity_{0};
  std::size_t current_candidate_count_{0};
  std::vector<EntryState> entries_;
  std::vector<EntryIndex> result_heap_;
  std::vector<CandidateHandle> candidate_heap_;
};

} // namespace hnsw_urma
