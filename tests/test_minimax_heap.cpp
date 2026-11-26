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

#include "hnsw_urma/core/minimax_heap.hpp"

using hnsw_urma::HeapEntry;
using hnsw_urma::MinimaxHeap;

TEST(ReferenceUnitMinimaxHeap, MinimaxHeapHandlesCapacityZeroAndOne) {
  auto zero = MinimaxHeap::Create(0).value();
  UB_REQUIRE_FALSE(zero.Push(1, 1.0F).value());
  UB_REQUIRE_FALSE(zero.PopMinCandidate().has_value());

  auto one = MinimaxHeap::Create(1).value();
  UB_REQUIRE_TRUE(one.Push(7, 7.0F).value());
  UB_REQUIRE_TRUE(one.Push(3, 3.0F).value());
  const auto one_sorted = one.ExtractSorted();
  UB_REQUIRE_EQ(one_sorted.front(), (HeapEntry{3, 3.0F}));
}

TEST(ReferenceUnitMinimaxHeap, MinimaxHeapUsesStableDistanceIdOrder) {
  auto heap = MinimaxHeap::Create(3).value();
  UB_REQUIRE_TRUE(heap.Push(9, 1.0F).value());
  UB_REQUIRE_TRUE(heap.Push(4, 1.0F).value());
  UB_REQUIRE_TRUE(heap.Push(5, 0.5F).value());
  UB_REQUIRE_FALSE(heap.Push(4, 1.0F).value());
  const auto sorted = heap.ExtractSorted();
  UB_REQUIRE_EQ(sorted[0], (HeapEntry{5, 0.5F}));
  UB_REQUIRE_EQ(sorted[1], (HeapEntry{4, 1.0F}));
  UB_REQUIRE_EQ(sorted[2], (HeapEntry{9, 1.0F}));
  UB_REQUIRE_EQ(heap.PopMinCandidate().value(), (HeapEntry{5, 0.5F}));
}

TEST(ReferenceUnitMinimaxHeap, MinimaxHeapEvictsWorstAndCanBeReused) {
  auto heap = MinimaxHeap::Create(2).value();
  UB_REQUIRE_TRUE(heap.Push(1, 10.0F).value());
  UB_REQUIRE_TRUE(heap.Push(2, 20.0F).value());
  UB_REQUIRE_TRUE(heap.Push(3, 5.0F).value());
  UB_REQUIRE_FALSE(heap.Push(4, 20.0F).value());
  UB_REQUIRE_EQ(heap.max().value(), (HeapEntry{1, 10.0F}));
  heap.Clear();
  UB_REQUIRE_EQ(heap.result_size(), 0U);
  UB_REQUIRE_TRUE(heap.Push(8, 2.0F).value());
  const auto reused_sorted = heap.ExtractSorted();
  UB_REQUIRE_EQ(reused_sorted.front().id, 8U);
}

TEST(ReferenceUnitMinimaxHeap, MinimaxHeapBetterTieReplacesWorseNodeId) {
  auto heap = MinimaxHeap::Create(2).value();
  UB_REQUIRE_TRUE(heap.Push(9, 1.0F).value());
  UB_REQUIRE_TRUE(heap.Push(4, 1.0F).value());
  UB_REQUIRE_TRUE(heap.PushUnique(3, 1.0F).value());

  const auto sorted = heap.ExtractSorted();
  UB_REQUIRE_EQ(sorted.size(), 2U);
  UB_REQUIRE_EQ(sorted[0].id, 3U);
  UB_REQUIRE_EQ(sorted[1].id, 4U);
}

TEST(ReferenceUnitMinimaxHeap,
     MinimaxHeapLazyInvalidationNeverReturnsEvictedCandidate) {
  auto heap = MinimaxHeap::Create(2).value();
  UB_REQUIRE_TRUE(heap.PushUnique(1, 10.0F).value());
  UB_REQUIRE_TRUE(heap.PushUnique(2, 20.0F).value());
  UB_REQUIRE_TRUE(heap.PushUnique(3, 5.0F).value());

  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 3U);
  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 1U);
  UB_REQUIRE_FALSE(heap.PopMinCandidate().has_value());
  UB_REQUIRE_EQ(heap.candidate_size(), 0U);
}

TEST(ReferenceUnitMinimaxHeap,
     MinimaxHeapCandidateCountTracksExpansionAndEviction) {
  auto heap = MinimaxHeap::Create(2).value();
  UB_REQUIRE_TRUE(heap.PushUnique(1, 1.0F).value());
  UB_REQUIRE_TRUE(heap.PushUnique(2, 2.0F).value());
  UB_REQUIRE_EQ(heap.candidate_size(), 2U);

  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 1U);
  UB_REQUIRE_EQ(heap.candidate_size(), 1U);

  UB_REQUIRE_TRUE(heap.PushUnique(3, 0.5F).value());
  UB_REQUIRE_EQ(heap.candidate_size(), 1U);
  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 3U);
  UB_REQUIRE_EQ(heap.candidate_size(), 0U);
  UB_REQUIRE_FALSE(heap.PopMinCandidate().has_value());
}

TEST(ReferenceUnitMinimaxHeap,
     MinimaxHeapCompactsStaleHandlesWithoutChangingResults) {
  auto heap = MinimaxHeap::Create(3).value();
  UB_REQUIRE_TRUE(heap.PushUnique(1, 100.0F).value());
  UB_REQUIRE_TRUE(heap.PushUnique(2, 90.0F).value());
  UB_REQUIRE_TRUE(heap.PushUnique(3, 80.0F).value());
  for (std::uint32_t id = 4; id < 40; ++id) {
    UB_REQUIRE_TRUE(
        heap.PushUnique(id, 80.0F - static_cast<float>(id)).value());
  }

  const auto sorted = heap.ExtractSorted();
  UB_REQUIRE_EQ(sorted.size(), 3U);
  UB_REQUIRE_EQ(sorted[0].id, 39U);
  UB_REQUIRE_EQ(sorted[1].id, 38U);
  UB_REQUIRE_EQ(sorted[2].id, 37U);

  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 39U);
  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 38U);
  UB_REQUIRE_EQ(heap.PopMinCandidate().value().id, 37U);
  UB_REQUIRE_FALSE(heap.PopMinCandidate().has_value());
}
