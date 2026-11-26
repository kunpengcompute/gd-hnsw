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

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

enum class Metric : std::uint32_t {
  kL2 = 0,
  kInnerProduct = 1,
};

enum class DistanceImplementation : std::uint8_t {
  kScalar,
  kSimd,
  kAuto,
};

[[nodiscard]] float ScalarL2(std::span<const float> lhs,
                             std::span<const float> rhs) noexcept;
[[nodiscard]] float ScalarDot(std::span<const float> lhs,
                              std::span<const float> rhs) noexcept;
[[nodiscard]] float L2FromDotNorm(float query_norm, float vector_norm,
                                  float dot) noexcept;

[[nodiscard]] float ComputeDistance(std::span<const float> lhs,
                                    std::span<const float> rhs, Metric metric,
                                    DistanceImplementation implementation =
                                        DistanceImplementation::kAuto) noexcept;

[[nodiscard]] Status ComputeDistanceBatch(
    std::span<const float> query, std::span<const float> row_major_vectors,
    std::uint64_t vector_count, Metric metric, std::span<float> output,
    DistanceImplementation implementation =
        DistanceImplementation::kAuto) noexcept;

// Computes up to four independent rows while loading each query block once.
// Row pointers must each reference query.size() floats.
[[nodiscard]] Status
ComputeDistanceBatch4(std::span<const float> query,
                      std::span<const float *const> vector_rows, Metric metric,
                      std::span<float> output,
                      DistanceImplementation implementation =
                          DistanceImplementation::kAuto) noexcept;

// Experimental L2 kernel using ||q-v||^2 = ||q||^2 + ||v||^2 - 2(q.v).
// vector_norms and output must contain four elements.
[[nodiscard]] Status
ComputeL2DotNormBatch4(std::span<const float> query, float query_norm,
                       std::span<const float *const> vector_rows,
                       std::span<const float> vector_norms,
                       std::span<float> output,
                       DistanceImplementation implementation =
                           DistanceImplementation::kAuto) noexcept;

[[nodiscard]] bool SimdDistanceAvailable() noexcept;
[[nodiscard]] std::string_view ActiveDistanceImplementationName() noexcept;

} // namespace hnsw_urma
