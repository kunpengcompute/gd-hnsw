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

#include <concepts>
#include <limits>
#include <type_traits>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

template <typename T>
concept UnsignedInteger = std::unsigned_integral<T> && !std::same_as<T, bool>;

template <UnsignedInteger T>
[[nodiscard]] constexpr Result<T> CheckedAdd(T lhs, T rhs) {
  if (rhs > std::numeric_limits<T>::max() - lhs) {
    return Status(StatusCode::kOutOfRange, "unsigned addition overflow");
  }
  return static_cast<T>(lhs + rhs);
}

template <UnsignedInteger T>
[[nodiscard]] constexpr Result<T> CheckedMultiply(T lhs, T rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<T>::max() / lhs) {
    return Status(StatusCode::kOutOfRange, "unsigned multiplication overflow");
  }
  return static_cast<T>(lhs * rhs);
}

template <UnsignedInteger T>
[[nodiscard]] constexpr Result<T> CheckedAlignUp(T value, T alignment) {
  if (alignment == 0 || (alignment & static_cast<T>(alignment - 1)) != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "alignment must be a non-zero power of two");
  }
  const T mask = static_cast<T>(alignment - 1);
  auto added = CheckedAdd(value, mask);
  if (!added.ok()) {
    return added.status();
  }
  return static_cast<T>(added.value() & static_cast<T>(~mask));
}

template <UnsignedInteger T>
[[nodiscard]] constexpr bool IsAligned(T value, T alignment) noexcept {
  return alignment != 0 && (alignment & static_cast<T>(alignment - 1)) == 0 &&
         (value & static_cast<T>(alignment - 1)) == 0;
}

} // namespace hnsw_urma
