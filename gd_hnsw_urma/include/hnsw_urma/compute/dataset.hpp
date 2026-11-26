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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

// Portable local fixture format: uint32 row count, uint32 dimension, followed
// by row-major IEEE-754 float32 values in host little-endian order.
struct FloatDataset {
  FloatDataset() = default;
  FloatDataset(std::uint64_t row_count, std::uint32_t dimension,
               std::vector<float> data)
      : rows(row_count), dim(dimension),
        values_(std::make_shared<std::vector<float>>(std::move(data))) {}

  std::uint64_t rows{0};
  std::uint32_t dim{0};

  // Returns empty if the row is outside either the shape or backing storage.
  [[nodiscard]] std::span<const float> Row(std::uint64_t row) const noexcept;
  [[nodiscard]] const std::vector<float> &values() const noexcept;
  [[nodiscard]] std::shared_ptr<const void> owner() const noexcept {
    return values_;
  }

private:
  std::shared_ptr<const std::vector<float>> values_;
};

[[nodiscard]] Result<FloatDataset>
LoadFloatBin(const std::filesystem::path &path);
[[nodiscard]] Status WriteFloatBin(const std::filesystem::path &path,
                                   const FloatDataset &dataset);

} // namespace hnsw_urma
