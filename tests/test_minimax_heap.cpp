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
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "gd_hnsw_search.h"

using namespace gd_hnsw;

TEST(MinimaxHeap, PushWithinCapacity)
{
    MinimaxHeap h(10);
    for (int i = 0; i < 5; i++)
        h.push(i, static_cast<float>(10 - i));
    EXPECT_EQ(h.size(), 5);
    EXPECT_EQ(h.candidates(), 5);
}

TEST(MinimaxHeap, PushBeyondCapacityEvictsWorst)
{
    MinimaxHeap h(5);
    for (int i = 0; i < 10; i++)
        h.push(i, static_cast<float>(i));
    EXPECT_EQ(h.size(), 5);
    EXPECT_LE(h.max(), 4.0f);
}

TEST(MinimaxHeap, TieBreakingEvictsLargerId)
{
    MinimaxHeap h(3);
    h.push(10, 1.0f);
    h.push(20, 2.0f);
    h.push(30, 1.5f);
    h.push(5, 0.5f); // forces eviction of worst

    std::vector<std::pair<float, int32_t>> sorted;
    h.extract_sorted(sorted);
    ASSERT_EQ(sorted.size(), 3u);
    bool found_20 = false, found_5 = false;
    for (auto &[d, id] : sorted) {
        if (id == 20)
            found_20 = true;
        if (id == 5)
            found_5 = true;
    }
    EXPECT_FALSE(found_20);
    EXPECT_TRUE(found_5);
}

TEST(MinimaxHeap, CandidateTieBreakingSmallerIdFirst)
{
    MinimaxHeap h(10);
    h.push(300, 0.5f);
    h.push(100, 0.5f);
    h.push(200, 0.5f);

    EXPECT_EQ(h.pop_min_candidate().second, 100);
    EXPECT_EQ(h.pop_min_candidate().second, 200);
    EXPECT_EQ(h.pop_min_candidate().second, 300);
}

TEST(MinimaxHeap, PopMinCandidateReturnsClosest)
{
    MinimaxHeap h(5);
    h.push(10, 1.0f);
    h.push(20, 3.0f);
    h.push(30, 2.0f);
    auto c = h.pop_min_candidate();
    EXPECT_EQ(c.second, 10);
    EXPECT_FLOAT_EQ(c.first, 1.0f);
}

TEST(MinimaxHeap, PopMinCandidateSkipsEvicted)
{
    MinimaxHeap h(2);
    h.push(10, 1.0f);
    h.push(20, 2.0f);
    h.push(30, 0.5f); // evicts worst

    std::set<int32_t> popped;
    while (true) {
        auto c = h.pop_min_candidate();
        if (c.second < 0)
            break;
        popped.insert(c.second);
    }
    EXPECT_EQ(popped.count(20), 0u);
    EXPECT_EQ(popped.count(10), 1u);
    EXPECT_EQ(popped.count(30), 1u);
}

TEST(MinimaxHeap, PopMinCandidateEmptyReturnsInf)
{
    MinimaxHeap h(5);
    auto c = h.pop_min_candidate();
    EXPECT_EQ(c.first, std::numeric_limits<float>::max());
    EXPECT_EQ(c.second, -1);
}

TEST(MinimaxHeap, MaxReturnsWorstDistance)
{
    MinimaxHeap h(5);
    h.push(1, 1.0f);
    h.push(2, 5.0f);
    h.push(3, 3.0f);
    EXPECT_FLOAT_EQ(h.max(), 5.0f);
}

TEST(MinimaxHeap, ClearResetsState)
{
    MinimaxHeap h(5);
    h.push(1, 1.0f);
    h.push(2, 2.0f);
    h.clear();
    EXPECT_EQ(h.size(), 0);
    EXPECT_EQ(h.candidates(), 0);
    h.push(10, 0.5f);
    EXPECT_EQ(h.size(), 1);
}

TEST(MinimaxHeap, CapacityGovernanceShrinks)
{
    MinimaxHeap h(10);
    for (int i = 0; i < 500; i++)
        h.push(i, static_cast<float>(rand() % 1000) / 1000.0f);
    h.clear();
    for (int i = 0; i < 10; i++)
        h.push(i, static_cast<float>(i));
    EXPECT_EQ(h.size(), 10);
}

TEST(MinimaxHeap, DeterministicOutput)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    auto run = [&]() {
        MinimaxHeap h(10);
        for (int i = 0; i < 50; i++)
            h.push(i, dist(rng));
        std::vector<std::pair<float, int32_t>> sorted;
        h.extract_sorted(sorted);
        return sorted;
    };

    rng.seed(42);
    auto r1 = run();
    rng.seed(42);
    auto r2 = run();
    EXPECT_EQ(r1, r2);
}

TEST(MinimaxHeap, LargeBatchInvariants)
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    MinimaxHeap h(100);
    for (int i = 0; i < 100000; i++)
        h.push(i, dist(rng));
    std::vector<std::pair<float, int32_t>> sorted;
    h.extract_sorted(sorted);
    ASSERT_EQ(sorted.size(), 100u);
    for (size_t i = 1; i < sorted.size(); i++)
        EXPECT_LE(sorted[i - 1].first, sorted[i].first);
}

TEST(MinimaxHeap, PoppedCandidateNotReturnedTwice)
{
    MinimaxHeap h(10);
    for (int i = 0; i < 10; i++)
        h.push(i, static_cast<float>(i));
    std::set<int32_t> popped;
    while (true) {
        auto c = h.pop_min_candidate();
        if (c.second < 0)
            break;
        EXPECT_EQ(popped.count(c.second), 0u);
        popped.insert(c.second);
    }
    EXPECT_EQ(popped.size(), 10u);
}

TEST(MinimaxHeap, ShrinkToFitBothVectors)
{
    MinimaxHeap h(10);
    // Push enough entries to inflate capacity well past 32*n (320)
    for (int i = 0; i < 5000; i++)
        h.push(i, static_cast<float>(rand() % 1000) / 1000.0f);

    h.clear(); // triggers shrink_to_fit on entries_ and cand_heap_

    // After shrink, heap must still function correctly
    for (int i = 0; i < 10; i++)
        h.push(i, static_cast<float>(10 - i));
    EXPECT_EQ(h.size(), 10);

    std::vector<std::pair<float, int32_t>> sorted;
    h.extract_sorted(sorted);
    ASSERT_EQ(sorted.size(), 10u);
    for (size_t i = 1; i < sorted.size(); i++)
        EXPECT_LE(sorted[i - 1].first, sorted[i].first);

    // Repeated clear+refill cycle
    for (int cycle = 0; cycle < 20; cycle++) {
        for (int i = 0; i < 200; i++)
            h.push(i, static_cast<float>(rand() % 1000) / 1000.0f);
        h.clear();
        h.push(0, 0.0f);
        EXPECT_EQ(h.size(), 1);
    }
}
