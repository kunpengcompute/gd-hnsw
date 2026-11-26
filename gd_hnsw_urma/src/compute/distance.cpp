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

#include "hnsw_urma/compute/distance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define HNSW_URMA_HAS_NEON 1
#else
#define HNSW_URMA_HAS_NEON 0
#endif

namespace hnsw_urma {
namespace {

#if HNSW_URMA_HAS_NEON
float NeonL2(std::span<const float> lhs, std::span<const float> rhs) noexcept {
  float32x4_t sum = vdupq_n_f32(0.0F);
  std::size_t index = 0;
  for (; index + 4 <= lhs.size(); index += 4) {
    const float32x4_t left = vld1q_f32(lhs.data() + index);
    const float32x4_t right = vld1q_f32(rhs.data() + index);
    const float32x4_t delta = vsubq_f32(left, right);
    sum = vfmaq_f32(sum, delta, delta);
  }
  float value = vaddvq_f32(sum);
  for (; index < lhs.size(); ++index) {
    const float delta = lhs[index] - rhs[index];
    value += delta * delta;
  }
  return value;
}

float NeonDot(std::span<const float> lhs, std::span<const float> rhs) noexcept {
  float32x4_t sum = vdupq_n_f32(0.0F);
  std::size_t index = 0;
  for (; index + 4 <= lhs.size(); index += 4) {
    sum = vfmaq_f32(sum, vld1q_f32(lhs.data() + index),
                    vld1q_f32(rhs.data() + index));
  }
  float value = vaddvq_f32(sum);
  for (; index < lhs.size(); ++index) {
    value += lhs[index] * rhs[index];
  }
  return value;
}

void NeonL2Batch4(std::span<const float> query,
                  std::span<const float *const> vectors,
                  std::span<float> output) noexcept {
  std::array<float32x4_t, 8> sums;
  for (auto &sum : sums) {
    sum = vdupq_n_f32(0.0F);
  }
  std::size_t index = 0;
  for (; index + 8 <= query.size(); index += 8) {
    const auto query0 = vld1q_f32(query.data() + index);
    const auto query1 = vld1q_f32(query.data() + index + 4);
    for (std::size_t row = 0; row < 4; ++row) {
      const auto vector0 = vld1q_f32(vectors[row] + index);
      const auto vector1 = vld1q_f32(vectors[row] + index + 4);
      const auto delta0 = vsubq_f32(query0, vector0);
      const auto delta1 = vsubq_f32(query1, vector1);
      sums[row * 2] = vfmaq_f32(sums[row * 2], delta0, delta0);
      sums[row * 2 + 1] = vfmaq_f32(sums[row * 2 + 1], delta1, delta1);
    }
  }
  for (std::size_t row = 0; row < 4; ++row) {
    float value = vaddvq_f32(vaddq_f32(sums[row * 2], sums[row * 2 + 1]));
    for (std::size_t tail = index; tail < query.size(); ++tail) {
      const float delta = query[tail] - vectors[row][tail];
      value += delta * delta;
    }
    output[row] = value;
  }
}

void NeonDotBatch4(std::span<const float> query,
                   std::span<const float *const> vectors,
                   std::span<float> output) noexcept {
  std::array<float32x4_t, 8> sums;
  for (auto &sum : sums) {
    sum = vdupq_n_f32(0.0F);
  }
  std::size_t index = 0;
  for (; index + 8 <= query.size(); index += 8) {
    const auto query0 = vld1q_f32(query.data() + index);
    const auto query1 = vld1q_f32(query.data() + index + 4);
    for (std::size_t row = 0; row < 4; ++row) {
      sums[row * 2] =
          vfmaq_f32(sums[row * 2], query0, vld1q_f32(vectors[row] + index));
      sums[row * 2 + 1] = vfmaq_f32(sums[row * 2 + 1], query1,
                                    vld1q_f32(vectors[row] + index + 4));
    }
  }
  for (std::size_t row = 0; row < 4; ++row) {
    float value = vaddvq_f32(vaddq_f32(sums[row * 2], sums[row * 2 + 1]));
    for (std::size_t tail = index; tail < query.size(); ++tail) {
      value += query[tail] * vectors[row][tail];
    }
    output[row] = value;
  }
}
#endif

void ScalarL2Batch4(std::span<const float> query,
                    std::span<const float *const> vectors,
                    std::span<float> output) noexcept {
  std::array<float, 4> sums{};
  for (std::size_t index = 0; index < query.size(); ++index) {
    const float query_value = query[index];
    for (std::size_t row = 0; row < 4; ++row) {
      const float delta = query_value - vectors[row][index];
      sums[row] += delta * delta;
    }
  }
  std::copy(sums.begin(), sums.end(), output.begin());
}

void ScalarDotBatch4(std::span<const float> query,
                     std::span<const float *const> vectors,
                     std::span<float> output) noexcept {
  std::array<float, 4> sums{};
  for (std::size_t index = 0; index < query.size(); ++index) {
    const float query_value = query[index];
    for (std::size_t row = 0; row < 4; ++row) {
      sums[row] += query_value * vectors[row][index];
    }
  }
  std::copy(sums.begin(), sums.end(), output.begin());
}

} // namespace

float ScalarL2(std::span<const float> lhs,
               std::span<const float> rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float sum = 0.0F;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    const float delta = lhs[index] - rhs[index];
    sum += delta * delta;
  }
  return sum;
}

float ScalarDot(std::span<const float> lhs,
                std::span<const float> rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float sum = 0.0F;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    sum += lhs[index] * rhs[index];
  }
  return sum;
}

float L2FromDotNorm(float query_norm, float vector_norm, float dot) noexcept {
  return query_norm + vector_norm - 2.0F * dot;
}

float ComputeDistance(std::span<const float> lhs, std::span<const float> rhs,
                      Metric metric,
                      DistanceImplementation implementation) noexcept {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
#if HNSW_URMA_HAS_NEON
  if (implementation != DistanceImplementation::kScalar) {
    return metric == Metric::kL2 ? NeonL2(lhs, rhs) : -NeonDot(lhs, rhs);
  }
#else
  static_cast<void>(implementation);
#endif
  return metric == Metric::kL2 ? ScalarL2(lhs, rhs) : -ScalarDot(lhs, rhs);
}

Status ComputeDistanceBatch(std::span<const float> query,
                            std::span<const float> row_major_vectors,
                            std::uint64_t vector_count, Metric metric,
                            std::span<float> output,
                            DistanceImplementation implementation) noexcept {
  if (query.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "distance query dimension must be non-zero");
  }
  if (vector_count != output.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "distance output count mismatch");
  }
  if (vector_count > row_major_vectors.size() / query.size() ||
      vector_count * query.size() != row_major_vectors.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "distance vector matrix size mismatch");
  }
  std::size_t row = 0;
  for (; row + 4 <= output.size(); row += 4) {
    std::array<const float *, 4> rows{};
    for (std::size_t batch_row = 0; batch_row < rows.size(); ++batch_row) {
      rows[batch_row] =
          row_major_vectors.data() + (row + batch_row) * query.size();
    }
    auto status = ComputeDistanceBatch4(
        query, rows, metric, output.subspan(row, rows.size()), implementation);
    if (!status.ok()) {
      return status;
    }
  }
  for (; row < output.size(); ++row) {
    const auto begin = row * query.size();
    output[row] =
        ComputeDistance(query, row_major_vectors.subspan(begin, query.size()),
                        metric, implementation);
  }
  return Status::Ok();
}

Status ComputeDistanceBatch4(std::span<const float> query,
                             std::span<const float *const> vector_rows,
                             Metric metric, std::span<float> output,
                             DistanceImplementation implementation) noexcept {
  if (query.empty() || vector_rows.size() != 4 || output.size() != 4 ||
      std::any_of(vector_rows.begin(), vector_rows.end(),
                  [](const float *row) { return row == nullptr; })) {
    return Status(StatusCode::kInvalidArgument,
                  "distance batch-4 shape is invalid");
  }
  if (metric != Metric::kL2) {
    for (std::size_t row = 0; row < vector_rows.size(); ++row) {
      output[row] = ComputeDistance(
          query, std::span<const float>(vector_rows[row], query.size()), metric,
          implementation);
    }
    return Status::Ok();
  }
#if HNSW_URMA_HAS_NEON
  if (implementation != DistanceImplementation::kScalar) {
    NeonL2Batch4(query, vector_rows, output);
    return Status::Ok();
  }
#endif
  ScalarL2Batch4(query, vector_rows, output);
  return Status::Ok();
}

Status ComputeL2DotNormBatch4(std::span<const float> query, float query_norm,
                              std::span<const float *const> vector_rows,
                              std::span<const float> vector_norms,
                              std::span<float> output,
                              DistanceImplementation implementation) noexcept {
  if (query.empty() || vector_rows.size() != 4 || vector_norms.size() != 4 ||
      output.size() != 4 ||
      std::any_of(vector_rows.begin(), vector_rows.end(),
                  [](const float *row) { return row == nullptr; })) {
    return Status(StatusCode::kInvalidArgument,
                  "dot-norm batch-4 shape is invalid");
  }
#if HNSW_URMA_HAS_NEON
  if (implementation != DistanceImplementation::kScalar) {
    NeonDotBatch4(query, vector_rows, output);
  } else {
    ScalarDotBatch4(query, vector_rows, output);
  }
#else
  static_cast<void>(implementation);
  ScalarDotBatch4(query, vector_rows, output);
#endif
  for (std::size_t row = 0; row < output.size(); ++row) {
    output[row] = std::max(
        0.0F, L2FromDotNorm(query_norm, vector_norms[row], output[row]));
  }
  return Status::Ok();
}

bool SimdDistanceAvailable() noexcept {
#if HNSW_URMA_HAS_NEON
  return true;
#else
  return false;
#endif
}

std::string_view ActiveDistanceImplementationName() noexcept {
  return SimdDistanceAvailable() ? "neon" : "scalar";
}

} // namespace hnsw_urma
