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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <cstdint>
#include <limits>

#include "hnsw_urma/core/checked_math.hpp"

using hnsw_urma::CheckedAdd;
using hnsw_urma::CheckedAlignUp;
using hnsw_urma::CheckedMultiply;
using hnsw_urma::StatusCode;

TEST(ReferenceUnitCheckedMath, CheckedAddAcceptsBoundary) {
  auto result = CheckedAdd<std::uint64_t>(
      std::numeric_limits<std::uint64_t>::max() - 1, 1);
  UB_REQUIRE_TRUE(result.ok());
  UB_REQUIRE_EQ(result.value(), std::numeric_limits<std::uint64_t>::max());
}

TEST(ReferenceUnitCheckedMath, CheckedAddRejectsOverflow) {
  auto result =
      CheckedAdd<std::uint64_t>(std::numeric_limits<std::uint64_t>::max(), 1);
  UB_REQUIRE_FALSE(result.ok());
  UB_REQUIRE_EQ(result.status().code(), StatusCode::kOutOfRange);
}

TEST(ReferenceUnitCheckedMath, CheckedMultiplyHandlesZeroAndOverflow) {
  UB_REQUIRE_EQ(CheckedMultiply<std::uint32_t>(
                    0, std::numeric_limits<std::uint32_t>::max())
                    .value(),
                0U);
  auto result = CheckedMultiply<std::uint32_t>(
      std::numeric_limits<std::uint32_t>::max(), 2);
  UB_REQUIRE_FALSE(result.ok());
  UB_REQUIRE_EQ(result.status().code(), StatusCode::kOutOfRange);
}

TEST(ReferenceUnitCheckedMath, CheckedAlignUpValidatesPowerOfTwoAndOverflow) {
  UB_REQUIRE_EQ(CheckedAlignUp<std::uint64_t>(4097, 4096).value(), 8192U);
  UB_REQUIRE_FALSE(CheckedAlignUp<std::uint64_t>(17, 3).ok());
  UB_REQUIRE_FALSE(CheckedAlignUp<std::uint64_t>(
                       std::numeric_limits<std::uint64_t>::max(), 8)
                       .ok());
}
