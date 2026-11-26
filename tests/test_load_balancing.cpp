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
#include <atomic>
#include <thread>
#include <vector>

#include "hnsw_urma/runtime/load_balancing.hpp"

namespace {

hnsw_urma::LaneLoadSample LaneLoad(std::uint32_t backlog,
                                   std::uint32_t reply_capacity = 1) {
  hnsw_urma::LaneLoadSample sample;
  sample.submit_backlog = backlog;
  sample.reply_capacity_remaining = reply_capacity;
  return sample;
}

} // namespace

TEST(ReferenceRuntimeLoadBalancing,
     LanePowerOfTwoAvoidsHotOrCapacityExhaustedCandidate) {
  auto snapshots = hnsw_urma::LaneLoadSnapshots::Create(2).value();
  snapshots->Publish(0, LaneLoad(100));
  snapshots->Publish(1, LaneLoad(1));
  const hnsw_urma::QueryKey query{3, 9, 1};
  UB_REQUIRE_EQ(
      hnsw_urma::SelectLanePowerOfTwo(query, 7, snapshots.get(), 2).value(),
      1U);

  snapshots->Publish(0, LaneLoad(0, 0));
  snapshots->Publish(1, LaneLoad(500, 1));
  UB_REQUIRE_EQ(
      hnsw_urma::SelectLanePowerOfTwo(query, 7, snapshots.get(), 2).value(),
      1U);
  snapshots->Publish(1, LaneLoad(0, 0));
  auto rejected = hnsw_urma::SelectLanePowerOfTwo(query, 7, snapshots.get(), 2);
  UB_REQUIRE_FALSE(rejected.ok());
  UB_REQUIRE_EQ(rejected.status().code(),
                hnsw_urma::StatusCode::kResourceExhausted);
}

TEST(ReferenceRuntimeLoadBalancing,
     QueryLaneAffinityDoesNotMigrateUntilCleared) {
  auto snapshots = hnsw_urma::LaneLoadSnapshots::Create(2).value();
  snapshots->Publish(0, LaneLoad(100));
  snapshots->Publish(1, LaneLoad(0));
  auto affinity = hnsw_urma::QueryLaneAffinity::Create(4).value();
  const hnsw_urma::QueryKey query{1, 2, 3};
  UB_REQUIRE_EQ(affinity.SelectOrGet(query, 2, snapshots.get(), 2).value(), 1U);
  snapshots->Publish(0, LaneLoad(0));
  snapshots->Publish(1, LaneLoad(100));
  UB_REQUIRE_EQ(affinity.SelectOrGet(query, 2, snapshots.get(), 2).value(), 1U);
  affinity.Clear();
  UB_REQUIRE_EQ(affinity.SelectOrGet(query, 2, snapshots.get(), 2).value(), 0U);
}

TEST(ReferenceRuntimeLoadBalancing,
     ConcurrentLaneAffinityPublishesOneStickyMappingAndReusesGeneration) {
  auto snapshots = hnsw_urma::LaneLoadSnapshots::Create(2).value();
  snapshots->Publish(0, LaneLoad(100));
  snapshots->Publish(1, LaneLoad(0));
  auto table = hnsw_urma::ConcurrentLaneAffinityTable::Create(8).value();
  const hnsw_urma::QueryKey query{1, 4, 7};
  std::atomic<bool> start{false};
  std::atomic<std::uint32_t> failures{0};
  std::array<std::uint32_t, 8> selected{};
  std::vector<std::thread> threads;
  for (std::uint32_t index = 0; index < selected.size(); ++index) {
    threads.emplace_back([&, index] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto lane = table->SelectOrGet(query, 3, snapshots.get(), 2);
      if (!lane.ok()) {
        failures.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      selected[index] = lane.value();
    });
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }
  UB_REQUIRE_EQ(failures.load(), 0U);
  for (const auto lane : selected) {
    UB_REQUIRE_EQ(lane, 1U);
  }
  UB_REQUIRE_EQ(table->inserts(), 1U);
  UB_REQUIRE_EQ(table->hits(), 7U);

  snapshots->Publish(0, LaneLoad(0));
  snapshots->Publish(1, LaneLoad(100));
  const hnsw_urma::QueryKey next_generation{1, 4, 8};
  UB_REQUIRE_EQ(
      table->SelectOrGet(next_generation, 3, snapshots.get(), 2).value(), 0U);
  UB_REQUIRE_EQ(table->replacements(), 1U);
  auto stale = table->SelectOrGet(query, 3, snapshots.get(), 2);
  UB_REQUIRE_FALSE(stale.ok());
  UB_REQUIRE_EQ(stale.status().code(), hnsw_urma::StatusCode::kProtocolError);
}

TEST(ReferenceRuntimeLoadBalancing, LaneLoadSnapshotsNeverExposeTornSamples) {
  auto snapshots = hnsw_urma::LaneLoadSnapshots::Create(1).value();
  snapshots->Publish(0, LaneLoad(0));
  std::atomic<bool> done{false};
  std::atomic<std::uint32_t> mismatches{0};
  std::thread publisher([&] {
    for (std::uint32_t value = 1; value < 10000; ++value) {
      auto sample = LaneLoad(value);
      sample.outstanding = value;
      sample.active_replies = value;
      snapshots->Publish(0, sample);
    }
    done.store(true, std::memory_order_release);
  });
  while (!done.load(std::memory_order_acquire)) {
    auto sample = snapshots->Read(0).value();
    if (sample.submit_backlog != sample.outstanding ||
        sample.submit_backlog != sample.active_replies) {
      mismatches.fetch_add(1, std::memory_order_relaxed);
    }
  }
  publisher.join();
  UB_REQUIRE_EQ(mismatches.load(), 0U);
}
