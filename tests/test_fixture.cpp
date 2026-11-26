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

#include <filesystem>
#include <vector>

#include "hnsw_urma/compute/dataset.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/search/local_io.hpp"
#include "hnsw_urma/search/search.hpp"

TEST(ReferenceIntegrationFixture,
     CommittedFixtureLoadsAndMatchesGoldenNeighbors) {
  const std::filesystem::path root =
      std::filesystem::path(HNSW_URMA_SOURCE_DIR) / "fixtures/local_v1";
  auto queries = hnsw_urma::LoadFloatBin(root / "queries.bin");
  auto global = hnsw_urma::LoadGlobalIndexFile(root / "index");
  UB_REQUIRE_TRUE(queries.ok());
  UB_REQUIRE_TRUE(global.ok());
  auto map = hnsw_urma::ShardMap::Create(global.value().ranges,
                                         global.value().meta.ntotal);
  auto upper =
      hnsw_urma::LoadUpperGraphFile(root / "index", global.value().meta);
  UB_REQUIRE_TRUE(map.ok());
  UB_REQUIRE_TRUE(upper.ok());

  std::vector<hnsw_urma::LocalShard> shards;
  shards.reserve(global.value().ranges.size());
  for (const auto &range : global.value().ranges) {
    auto shard =
        hnsw_urma::LoadShardFile(root / "index", global.value().meta, range);
    UB_REQUIRE_TRUE(shard.ok());
    shards.push_back(std::move(shard).value());
  }
  std::vector<const hnsw_urma::LocalShard *> views;
  for (const auto &shard : shards) {
    views.push_back(&shard);
  }
  auto graph =
      hnsw_urma::LocalGraphReader::Create(map.value(), upper.value(), views);
  auto distance = hnsw_urma::LocalDistanceOracle::Create(
      map.value(), views, hnsw_urma::Metric::kL2,
      hnsw_urma::DistanceImplementation::kScalar);
  UB_REQUIRE_TRUE(graph.ok());
  UB_REQUIRE_TRUE(distance.ok());

  hnsw_urma::SearchParams params;
  params.topk = 3;
  params.ef_search = 8;
  params.expand_batch = 3;
  auto task = hnsw_urma::HnswSearchTask::Create(
      {0, 0, 1}, global.value().meta, map.value(), graph.value(),
      distance.value(), queries.value().Row(0), params);
  UB_REQUIRE_TRUE(task.ok());
  auto result = hnsw_urma::RunToCompletion(std::move(task).value());
  UB_REQUIRE_TRUE(result.ok());
  UB_REQUIRE_EQ(result.value().neighbors.size(), 3U);
  UB_REQUIRE_EQ(result.value().neighbors[0].id, 1U);
  UB_REQUIRE_EQ(result.value().neighbors[1].id, 2U);
  UB_REQUIRE_EQ(result.value().neighbors[2].id, 0U);
  UB_REQUIRE_NEAR(result.value().neighbors[0].distance, 0.0500000045F, 1e-5F);
  UB_REQUIRE_NEAR(result.value().neighbors[1].distance, 0.849999964F, 1e-5F);
  UB_REQUIRE_NEAR(result.value().neighbors[2].distance, 1.25F, 1e-5F);
}
