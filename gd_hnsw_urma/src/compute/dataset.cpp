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

#include "hnsw_urma/compute/dataset.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <new>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {

std::span<const float> FloatDataset::Row(std::uint64_t row) const noexcept {
  if (row >= rows || dim == 0 || values_ == nullptr) {
    return {};
  }
  const auto begin = static_cast<std::size_t>(row) * dim;
  return std::span<const float>(*values_).subspan(begin, dim);
}

const std::vector<float> &FloatDataset::values() const noexcept {
  static const std::vector<float> empty;
  return values_ == nullptr ? empty : *values_;
}

Result<FloatDataset> LoadFloatBin(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return Status(StatusCode::kInvalidArgument, "unable to open float dataset")
        .AddContext("path", path.string());
  }
  const auto end = input.tellg();
  if (end < static_cast<std::streamoff>(sizeof(std::uint32_t) * 2U)) {
    return Status(StatusCode::kInvalidFormat,
                  "float dataset header is truncated");
  }
  const auto file_bytes = static_cast<std::uint64_t>(end);
  input.seekg(0);

  std::array<std::uint32_t, 2> header{};
  input.read(reinterpret_cast<char *>(header.data()),
             static_cast<std::streamsize>(sizeof(header)));
  const std::uint64_t rows = header[0];
  const std::uint32_t dim = header[1];
  if (rows == 0 || dim == 0) {
    return Status(StatusCode::kInvalidFormat,
                  "float dataset shape must be non-zero");
  }

  auto count = CheckedMultiply(rows, static_cast<std::uint64_t>(dim));
  if (!count.ok() || count.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "float dataset element count overflows memory size");
  }
  auto payload_bytes =
      CheckedMultiply(count.value(), static_cast<std::uint64_t>(sizeof(float)));
  if (!payload_bytes.ok()) {
    return payload_bytes.status();
  }
  auto expected_bytes = CheckedAdd(payload_bytes.value(),
                                   static_cast<std::uint64_t>(sizeof(header)));
  if (!expected_bytes.ok() || expected_bytes.value() != file_bytes) {
    return Status(StatusCode::kInvalidFormat,
                  "float dataset size does not match its shape");
  }

  try {
    std::vector<float> values(static_cast<std::size_t>(count.value()));
    input.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(payload_bytes.value()));
    if (!input) {
      return Status(StatusCode::kInvalidFormat,
                    "float dataset payload is truncated");
    }
    return FloatDataset(rows, dim, std::move(values));
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate float dataset");
  }
}

Status WriteFloatBin(const std::filesystem::path &path,
                     const FloatDataset &dataset) {
  const auto &values = dataset.values();
  if (dataset.rows == 0 ||
      dataset.rows > std::numeric_limits<std::uint32_t>::max() ||
      dataset.dim == 0 || dataset.rows > values.size() / dataset.dim ||
      dataset.rows * dataset.dim != values.size()) {
    return Status(StatusCode::kInvalidArgument, "invalid float dataset shape");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status(StatusCode::kInvalidArgument,
                  "unable to create float dataset")
        .AddContext("path", path.string());
  }
  const std::array<std::uint32_t, 2> header{
      static_cast<std::uint32_t>(dataset.rows), dataset.dim};
  output.write(reinterpret_cast<const char *>(header.data()),
               static_cast<std::streamsize>(sizeof(header)));
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output) {
    return Status(StatusCode::kInternalError, "failed to write float dataset")
        .AddContext("path", path.string());
  }
  return Status::Ok();
}

} // namespace hnsw_urma
