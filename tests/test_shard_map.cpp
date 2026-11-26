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

#include <array>

#include "hnsw_urma/core/shard_map.hpp"

using hnsw_urma::NodeLocation;
using hnsw_urma::ShardMap;
using hnsw_urma::ShardRange;

TEST(ReferenceUnitShardMap, ShardMapLocatesUniformBoundaries) {
  const std::array ranges{ShardRange{0, 0, 4}, ShardRange{1, 4, 4},
                          ShardRange{2, 8, 4}};
  auto map = ShardMap::Create(ranges, 12);
  UB_REQUIRE_TRUE(map.ok());
  UB_REQUIRE_TRUE(map.value().uniform());
  UB_REQUIRE_EQ(map.value().Locate(0).value(), (NodeLocation{0, 0}));
  UB_REQUIRE_EQ(map.value().Locate(3).value(), (NodeLocation{0, 3}));
  UB_REQUIRE_EQ(map.value().Locate(4).value(), (NodeLocation{1, 0}));
  UB_REQUIRE_EQ(map.value().Locate(11).value(), (NodeLocation{2, 3}));
  UB_REQUIRE_FALSE(map.value().Locate(12).ok());
}

TEST(ReferenceUnitShardMap, ShardMapLocatesNonUniformBoundaries) {
  const std::array ranges{ShardRange{0, 0, 2}, ShardRange{1, 2, 5},
                          ShardRange{2, 7, 3}};
  auto map = ShardMap::Create(ranges, 10);
  UB_REQUIRE_TRUE(map.ok());
  UB_REQUIRE_FALSE(map.value().uniform());
  UB_REQUIRE_EQ(map.value().Locate(2).value(), (NodeLocation{1, 0}));
  UB_REQUIRE_EQ(map.value().Locate(6).value(), (NodeLocation{1, 4}));
  UB_REQUIRE_EQ(map.value().Locate(7).value(), (NodeLocation{2, 0}));
}

TEST(ReferenceUnitShardMap, ShardMapRejectsGapsOverlapsAndBadIds) {
  const std::array gap{ShardRange{0, 0, 2}, ShardRange{1, 3, 2}};
  UB_REQUIRE_FALSE(ShardMap::Create(gap, 5).ok());
  const std::array overlap{ShardRange{0, 0, 3}, ShardRange{1, 2, 2}};
  UB_REQUIRE_FALSE(ShardMap::Create(overlap, 4).ok());
  const std::array bad_id{ShardRange{1, 0, 4}};
  UB_REQUIRE_FALSE(ShardMap::Create(bad_id, 4).ok());
}
