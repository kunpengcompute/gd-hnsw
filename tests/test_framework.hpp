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
#include <cmath>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string_view>

// The reference's requirements were fatal, including in non-void helpers.
// Keep that contract instead of continuing into Result::value() after failure.
// Native GoogleTest registers/runs every TEST and catches/report exceptions.
namespace hnsw_urma::test {
[[noreturn]] inline void Fail(std::string_view expression,
                              std::string_view file, int line,
                              std::string_view detail = {}) {
  std::ostringstream message;
  message << file << ':' << line << ": " << expression;
  if (!detail.empty())
    message << " (" << detail << ')';
  throw std::runtime_error(message.str());
}
} // namespace hnsw_urma::test
#define UB_REQUIRE_TRUE(expression)                                            \
  do {                                                                         \
    if (!(expression))                                                         \
      ::hnsw_urma::test::Fail(#expression, __FILE__, __LINE__);                \
  } while (false)
#define UB_REQUIRE_FALSE(expression) UB_REQUIRE_TRUE(!(expression))
#define UB_REQUIRE_EQ(lhs, rhs)                                                \
  do {                                                                         \
    const auto &ub_left = (lhs);                                               \
    const auto &ub_right = (rhs);                                              \
    if (!(ub_left == ub_right))                                                \
      ::hnsw_urma::test::Fail(#lhs " == " #rhs, __FILE__, __LINE__,            \
                              ::testing::PrintToString(ub_left) + " vs " +     \
                                  ::testing::PrintToString(ub_right));         \
  } while (false)
#define UB_REQUIRE_NEAR(lhs, rhs, tolerance)                                   \
  do {                                                                         \
    const double ub_left = (lhs);                                              \
    const double ub_right = (rhs);                                             \
    const double ub_tolerance = (tolerance);                                   \
    if (!(ub_tolerance >= 0) || std::isnan(ub_left) || std::isnan(ub_right) || \
        !(ub_left == ub_right ||                                               \
          std::fabs(ub_left - ub_right) <= ub_tolerance))                      \
      ::hnsw_urma::test::Fail(#lhs " ~= " #rhs, __FILE__, __LINE__);           \
  } while (false)
