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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "hnsw_urma/search/local_io.hpp"

namespace hnsw_urma {
namespace {

GlobalIndexMeta Meta() {
  GlobalIndexMeta meta;
  meta.ntotal = 8;
  meta.dim = 2;
  meta.metric = 0;
  meta.M = 2;
  meta.M0 = 4;
  meta.ef_search_default = 8;
  meta.entry_point = 1;
  meta.max_level = 1;
  meta.num_shards = 2;
  return meta;
}

QueryKey Key() { return QueryKey{1, 2, 7}; }

Result<LocalShard> MakeShard(const GlobalIndexMeta &meta, shard_id_t shard_id) {
  const ShardRange range{shard_id, static_cast<node_id_t>(shard_id * 4U), 4};
  std::vector<node_id_t> layer0(4U * meta.M0, INVALID_NODE_ID);
  for (std::uint64_t row = 0; row < 4; ++row) {
    const node_id_t first = static_cast<node_id_t>(shard_id * 4U + row);
    layer0[static_cast<std::size_t>(row) * meta.M0] =
        row % 2U == 0U ? static_cast<node_id_t>(first + 1U)
                       : static_cast<node_id_t>(first - 1U);
  }
  std::vector<float> vectors(4U * meta.dim);
  for (std::size_t index = 0; index < vectors.size(); ++index) {
    vectors[index] = static_cast<float>(index + 1U) * 0.25F;
  }
  return LocalShard::Create(meta, range, std::move(layer0), std::move(vectors));
}

} // namespace

TEST(SearchLocalIo, GraphReaderValidatesAndReads) {
  const auto meta = Meta();
  const std::array<ShardRange, 2> ranges{ShardRange{0, 0, 4},
                                         ShardRange{1, 4, 4}};
  auto map = ShardMap::Create(ranges, meta.ntotal);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  auto shard0_result = MakeShard(meta, 0);
  ASSERT_TRUE(shard0_result.ok()) << shard0_result.status().ToString();
  auto shard1_result = MakeShard(meta, 1);
  ASSERT_TRUE(shard1_result.ok()) << shard1_result.status().ToString();
  LocalShard shard0 = std::move(shard0_result).value();
  LocalShard shard1 = std::move(shard1_result).value();
  auto upper = UpperGraphReplica::Create(
      meta, std::vector<UpperNodeMeta>{{1U, 1U, 0U, 0U}},
      std::vector<node_id_t>{2U, INVALID_NODE_ID});
  ASSERT_TRUE(upper.ok()) << upper.status().ToString();

  const std::array<const LocalShard *, 1> one{&shard0};
  EXPECT_EQ(
      LocalGraphReader::Create(map.value(), upper.value(), one).status().code(),
      StatusCode::kInvalidArgument);
  const std::array<const LocalShard *, 2> missing{nullptr, nullptr};
  EXPECT_FALSE(
      LocalGraphReader::Create(map.value(), upper.value(), missing).ok());
  const std::array<const LocalShard *, 2> reversed{&shard1, &shard0};
  EXPECT_FALSE(
      LocalGraphReader::Create(map.value(), upper.value(), reversed).ok());

  const std::array<const LocalShard *, 2> views{&shard0, &shard1};
  auto reader = LocalGraphReader::Create(map.value(), upper.value(), views);
  ASSERT_TRUE(reader.ok()) << reader.status().ToString();
  auto &graph = reader.value();

  NeighborBatch batch;
  EXPECT_EQ(graph.Submit(Key(), std::span<const NodeToExpand>{}, &batch)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  const NodeToExpand local{0, 0};
  const std::array<NodeToExpand, 1> local_nodes{local};
  EXPECT_EQ(graph.Submit(Key(), local_nodes, nullptr).status().code(),
            StatusCode::kInvalidArgument);

  auto ticket = graph.Submit(Key(), local_nodes, &batch);
  ASSERT_TRUE(ticket.ok()) << ticket.status().ToString();
  ASSERT_EQ(batch.neighbors.size(), 1U);
  EXPECT_EQ(batch.neighbors[0], 1U);
  EXPECT_EQ(batch.offsets.size(), 2U);

  const NodeToExpand upper_node{1, 1};
  const std::array<NodeToExpand, 1> upper_nodes{upper_node};
  auto upper_ticket = graph.Submit(Key(), upper_nodes, &batch);
  ASSERT_TRUE(upper_ticket.ok()) << upper_ticket.status().ToString();
  const NodeToExpand absent_upper{5, 1};
  const std::array<NodeToExpand, 1> absent_upper_nodes{absent_upper};
  EXPECT_EQ(graph.Submit(Key(), absent_upper_nodes, &batch).status().code(),
            StatusCode::kOutOfRange);
  const NodeToExpand absent{99, 0};
  const std::array<NodeToExpand, 1> absent_nodes{absent};
  EXPECT_EQ(graph.Submit(Key(), absent_nodes, &batch).status().code(),
            StatusCode::kOutOfRange);

  EXPECT_TRUE(graph.Poll(ticket.value()).value());
  EXPECT_TRUE(graph.Finish(ticket.value()).ok());
  EXPECT_FALSE(graph.ReadUpperNeighbors(0, 1).ok());
  EXPECT_TRUE(graph.SupportsPreparedWait());
  EXPECT_TRUE(graph.PublishPrepared(ticket.value()).ok());
  EXPECT_TRUE(graph.CancelPrepared(ticket.value()).ok());
  QueryWaitPreparation preparation;
  EXPECT_EQ(
      graph.PrepareForWait(preparation, local_nodes, &batch).status().code(),
      StatusCode::kInvalidArgument);
}

TEST(SearchLocalIo, DistanceOracleCoversBatchesAndTrustedPaths) {
  const auto meta = Meta();
  const std::array<ShardRange, 2> ranges{ShardRange{0, 0, 4},
                                         ShardRange{1, 4, 4}};
  auto map = ShardMap::Create(ranges, meta.ntotal);
  ASSERT_TRUE(map.ok()) << map.status().ToString();
  auto shard0_result = MakeShard(meta, 0);
  ASSERT_TRUE(shard0_result.ok()) << shard0_result.status().ToString();
  auto shard1_result = MakeShard(meta, 1);
  ASSERT_TRUE(shard1_result.ok()) << shard1_result.status().ToString();
  LocalShard shard0 = std::move(shard0_result).value();
  LocalShard shard1 = std::move(shard1_result).value();
  const std::array<const LocalShard *, 2> views{&shard0, &shard1};

  const std::array<const LocalShard *, 1> one{&shard0};
  EXPECT_EQ(LocalDistanceOracle::Create(map.value(), one, Metric::kL2)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_FALSE(LocalDistanceOracle::Create(map.value(),
                                           std::span<const LocalShard *const>{},
                                           Metric::kL2)
                   .ok());

  auto oracle_result =
      LocalDistanceOracle::Create(map.value(), views, Metric::kL2);
  ASSERT_TRUE(oracle_result.ok()) << oracle_result.status().ToString();
  auto &oracle = oracle_result.value();
  const std::array<float, 2> query{1.0F, 2.0F};
  const std::array<node_id_t, 4> four{0, 1, 2, 3};
  const std::array<node_id_t, 2> two{4, 5};
  const std::array<node_id_t, 1> one_id{0};
  std::array<float, 4> out4{};
  std::array<float, 2> out2{};
  std::array<float, 1> out1{};

  EXPECT_EQ(oracle.Submit(Key(), query, 9, four, out4).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(oracle.Submit(Key(), query, 0, std::span<const node_id_t>{}, out4)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(oracle.Submit(Key(), query, 0, four, out2).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(oracle.Submit(Key(), query, 0, two, out2).status().code(),
            StatusCode::kOutOfRange);
  const std::array<node_id_t, 1> absent{99};
  EXPECT_EQ(oracle.Submit(Key(), query, 0, absent, out1).status().code(),
            StatusCode::kOutOfRange);

  auto batch4 = oracle.Submit(Key(), query, 0, four, out4);
  ASSERT_TRUE(batch4.ok()) << batch4.status().ToString();
  auto single = oracle.Submit(Key(), query, 1, two, out2);
  ASSERT_TRUE(single.ok()) << single.status().ToString();
  auto one_result = oracle.Submit(Key(), query, 0, one_id, out1);
  ASSERT_TRUE(one_result.ok()) << one_result.status().ToString();
  EXPECT_TRUE(oracle.Poll(batch4.value()).value());
  EXPECT_TRUE(oracle.Finish(batch4.value()).ok());
  EXPECT_TRUE(oracle.CanComputeLocalTrusted(0));
  EXPECT_FALSE(oracle.CanComputeLocalTrusted(9));

  QueryWaitPreparation preparation;
  EXPECT_EQ(oracle.PrepareForWait(preparation, query, 0, one_id, out1, {})
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  const std::array<std::uint64_t, 4> offsets4{0, 1, 2, 3};
  const std::array<std::uint64_t, 2> offsets2{0, 1};
  EXPECT_EQ(
      oracle.ComputeLocalTrusted(query, 9, offsets4, out4).status().code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      oracle
          .ComputeLocalTrusted(query, 0, std::span<const std::uint64_t>{}, out4)
          .status()
          .code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      oracle.ComputeLocalTrusted(query, 0, offsets4, out2).status().code(),
      StatusCode::kInvalidArgument);
  EXPECT_TRUE(oracle.ComputeLocalTrusted(query, 0, offsets4, out4).value());
  EXPECT_TRUE(oracle.ComputeLocalTrusted(query, 1, offsets2, out2).value());

  const QueryVectorLease lease(
      Key(),
      std::make_shared<const std::vector<float>>(query.begin(), query.end()),
      5.0F);
  EXPECT_TRUE(
      oracle.ComputeLocalTrustedLease(lease, 0, offsets4, out4).value());
  EXPECT_TRUE(
      oracle.ComputeLocalTrustedLease(lease, 1, offsets2, out2).value());
  EXPECT_EQ(
      oracle.ComputeLocalTrustedLease(lease, 9, offsets4, out4).status().code(),
      StatusCode::kInvalidArgument);

  auto dot_result = LocalDistanceOracle::Create(
      map.value(), views, Metric::kL2, DistanceImplementation::kAuto, true);
  ASSERT_TRUE(dot_result.ok()) << dot_result.status().ToString();
  auto &dot = dot_result.value();
  auto dot_batch = dot.Submit(Key(), query, 0, four, out4);
  ASSERT_TRUE(dot_batch.ok()) << dot_batch.status().ToString();
  auto dot_single = dot.Submit(Key(), query, 0, one_id, out1);
  ASSERT_TRUE(dot_single.ok()) << dot_single.status().ToString();
  EXPECT_TRUE(dot.ComputeLocalTrusted(query, 0, offsets4, out4).value());
  EXPECT_TRUE(dot.ComputeLocalTrustedLease(lease, 0, offsets2, out2).value());
}

} // namespace hnsw_urma
