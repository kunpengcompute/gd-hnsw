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
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

enum class VisitedLayout : std::uint8_t {
  kDenseGeneration,
  kSparseEpoch,
};

// QuerySlot-owned visited set. Dense mode directly indexes one generation byte
// per global ID. Sparse mode stores packed key+epoch values in a fixed hash
// table. Clear() advances a generation/epoch and only touches the full table
// when the counter wraps.
class VisitedTable {
public:
  [[nodiscard]] static Result<VisitedTable>
  Create(std::uint64_t ntotal, std::size_t capacity, std::size_t max_probe = 0,
         double max_load_factor = 0.70);
  [[nodiscard]] static Result<VisitedTable>
  CreateAdaptive(std::uint64_t ntotal, std::size_t sparse_capacity,
                 std::uint32_t active_slots,
                 std::uint64_t dense_total_budget_bytes,
                 std::size_t max_probe = 0, double max_load_factor = 0.70);

  // Returns true for a new ID and false when it was already present.
  [[nodiscard]] Result<bool> Mark(node_id_t id) noexcept;
  [[nodiscard]] Result<bool> Contains(node_id_t id) const noexcept;
  void Prefetch(node_id_t id) const noexcept;
  void Clear() noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t max_probe() const noexcept { return max_probe_; }
  [[nodiscard]] std::size_t probe_high_watermark() const noexcept {
    return probe_high_watermark_;
  }
  [[nodiscard]] std::uint64_t ntotal() const noexcept { return ntotal_; }
  [[nodiscard]] VisitedLayout layout() const noexcept { return layout_; }

private:
  VisitedTable(std::uint64_t ntotal, VisitedLayout layout,
               std::size_t max_probe, std::size_t max_entries)
      : ntotal_(ntotal), layout_(layout), max_probe_(max_probe),
        max_entries_(max_entries) {}

  [[nodiscard]] std::size_t InitialSlot(node_id_t id) const noexcept;
  [[nodiscard]] std::uint64_t Packed(node_id_t id) const noexcept;

  std::uint64_t ntotal_{0};
  VisitedLayout layout_{VisitedLayout::kSparseEpoch};
  std::size_t max_probe_{0};
  std::size_t max_entries_{0};
  std::size_t size_{0};
  std::size_t probe_high_watermark_{0};
  std::vector<std::uint8_t> dense_stamps_;
  std::vector<std::uint64_t> sparse_slots_;
  std::uint8_t dense_generation_{1};
  std::uint32_t sparse_epoch_{1};
};

} // namespace hnsw_urma
