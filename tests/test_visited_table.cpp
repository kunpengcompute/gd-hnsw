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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <cstdint>
#include <vector>

#include "hnsw_urma/core/visited_table.hpp"

using hnsw_urma::StatusCode;
using hnsw_urma::VisitedLayout;
using hnsw_urma::VisitedTable;

TEST(ReferenceUnitVisitedTable, VisitedTableMarksChecksAndClears) {
  auto table = VisitedTable::Create(100, 16).value();
  UB_REQUIRE_TRUE(table.Mark(42).value());
  UB_REQUIRE_FALSE(table.Mark(42).value());
  UB_REQUIRE_TRUE(table.Contains(42).value());
  UB_REQUIRE_FALSE(table.Contains(43).value());
  table.Clear();
  UB_REQUIRE_EQ(table.size(), 0U);
  UB_REQUIRE_FALSE(table.Contains(42).value());
}

TEST(ReferenceUnitVisitedTable,
     VisitedTableReportsLoadExhaustionAndIllegalIds) {
  auto table = VisitedTable::Create(100, 4, 4, 0.5).value();
  UB_REQUIRE_TRUE(table.Mark(1).value());
  UB_REQUIRE_TRUE(table.Mark(2).value());
  auto full = table.Mark(3);
  UB_REQUIRE_FALSE(full.ok());
  UB_REQUIRE_EQ(full.status().code(), StatusCode::kResourceExhausted);
  UB_REQUIRE_FALSE(table.Mark(100).ok());
  UB_REQUIRE_FALSE(table.Mark(hnsw_urma::INVALID_NODE_ID).ok());
}

TEST(ReferenceUnitVisitedTable, VisitedTableEnforcesMaximumProbe) {
  auto table = VisitedTable::Create(10000, 8, 1, 1.0).value();
  std::vector<std::uint32_t> by_slot(8, hnsw_urma::INVALID_NODE_ID);
  bool found_collision = false;
  for (std::uint32_t id = 0; id < 10000 && !found_collision; ++id) {
    auto single = VisitedTable::Create(10000, 8, 1, 1.0).value();
    UB_REQUIRE_TRUE(single.Mark(id).value());
    for (std::uint32_t other = id + 1; other < 10000; ++other) {
      auto probe = single.Mark(other);
      if (!probe.ok() &&
          probe.status().code() == StatusCode::kResourceExhausted) {
        UB_REQUIRE_TRUE(table.Mark(id).value());
        auto collision = table.Mark(other);
        UB_REQUIRE_FALSE(collision.ok());
        found_collision = true;
        break;
      }
    }
  }
  UB_REQUIRE_TRUE(found_collision);
}

TEST(ReferenceUnitVisitedTable,
     VisitedTableSelectsDenseLayoutWithinAggregateBudget) {
  auto dense = VisitedTable::CreateAdaptive(1024, 128, 64, 64 * 1024).value();
  UB_REQUIRE_EQ(dense.layout(), VisitedLayout::kDenseGeneration);
  UB_REQUIRE_EQ(dense.capacity(), 1024U);
  UB_REQUIRE_TRUE(dense.Mark(1023).value());
  UB_REQUIRE_FALSE(dense.Mark(1023).value());

  for (std::uint32_t generation = 0; generation < 300; ++generation) {
    dense.Clear();
    UB_REQUIRE_FALSE(dense.Contains(1023).value());
    UB_REQUIRE_TRUE(dense.Mark(1023).value());
  }
}

TEST(ReferenceUnitVisitedTable,
     VisitedTableFallsBackToSparseLayoutOverAggregateBudget) {
  auto sparse = VisitedTable::CreateAdaptive(1024, 128, 65, 64 * 1024).value();
  UB_REQUIRE_EQ(sparse.layout(), VisitedLayout::kSparseEpoch);
  UB_REQUIRE_EQ(sparse.capacity(), 128U);
  UB_REQUIRE_TRUE(sparse.Mark(17).value());
  sparse.Clear();
  UB_REQUIRE_FALSE(sparse.Contains(17).value());
  UB_REQUIRE_TRUE(sparse.Mark(17).value());
}
