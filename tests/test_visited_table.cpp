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
#include <vector>

#include "gd_hnsw_search.h"

using namespace gd_hnsw;

static constexpr uint64_t SPARSE_TRIGGER = 16ULL * 1024 * 1024 + 1;

// ============================================================
// Dense mode
// ============================================================

TEST(VisitedTable, DenseGetFalseForUnvisited)
{
    VisitedTable vt(100);
    EXPECT_FALSE(vt.get(50));
}

TEST(VisitedTable, DenseSetReturnsTrueFirstVisit)
{
    VisitedTable vt(100);
    EXPECT_TRUE(vt.set(50));
}

TEST(VisitedTable, DenseSetReturnsFalseSecondVisit)
{
    VisitedTable vt(100);
    vt.set(50);
    EXPECT_FALSE(vt.set(50));
}

TEST(VisitedTable, DenseGetTrueAfterSet)
{
    VisitedTable vt(100);
    vt.set(50);
    EXPECT_TRUE(vt.get(50));
}

TEST(VisitedTable, DenseAdvanceClearsAll)
{
    VisitedTable vt(100);
    vt.set(10);
    vt.set(50);
    vt.advance();
    EXPECT_FALSE(vt.get(10));
    EXPECT_FALSE(vt.get(50));
}

TEST(VisitedTable, DenseSetWorksAfterAdvance)
{
    VisitedTable vt(100);
    vt.set(10);
    vt.advance();
    EXPECT_TRUE(vt.set(10));
}

TEST(VisitedTable, DenseGenerationWraparound)
{
    VisitedTable vt(100);
    for (int i = 0; i < 255; i++)
        vt.advance();
    EXPECT_FALSE(vt.get(0));
    vt.set(0);
    EXPECT_TRUE(vt.get(0));
}

// ============================================================
// Sparse mode
// ============================================================

TEST(VisitedTable, SparseTriggeredForLargeNtotal)
{
    VisitedTable vt(SPARSE_TRIGGER);
    EXPECT_FALSE(vt.get(42));
}

TEST(VisitedTable, SparseGetSetBasic)
{
    VisitedTable vt(SPARSE_TRIGGER);
    EXPECT_FALSE(vt.get(100));
    EXPECT_TRUE(vt.set(100));
    EXPECT_TRUE(vt.get(100));
    EXPECT_FALSE(vt.set(100));
}

TEST(VisitedTable, SparseAdvanceClearsOld)
{
    VisitedTable vt(SPARSE_TRIGGER);
    vt.set(100);
    vt.set(200);
    vt.advance();
    EXPECT_FALSE(vt.get(100));
    EXPECT_FALSE(vt.get(200));
    EXPECT_TRUE(vt.set(100));
}

TEST(VisitedTable, SparseHashCollisionResolved)
{
    VisitedTable vt(SPARSE_TRIGGER);
    const int N = 1000;
    for (int i = 0; i < N; i++)
        EXPECT_TRUE(vt.set(i * 1000));
    for (int i = 0; i < N; i++)
        EXPECT_TRUE(vt.get(i * 1000));
}

TEST(VisitedTable, SparseStampWraparound)
{
    VisitedTable vt(SPARSE_TRIGGER);
    vt.set(42);
    for (uint64_t i = 0; i < 0x100000001ULL; i++) {
        vt.advance();
        if (i > 0x100000000ULL - 5) {
            EXPECT_FALSE(vt.get(42));
            break;
        }
    }
    EXPECT_TRUE(vt.set(99));
    EXPECT_TRUE(vt.get(99));
}

// ============================================================
// Threshold boundaries
// ============================================================

TEST(VisitedTable, Exactly16MUsesDense)
{
    VisitedTable vt(16ULL * 1024 * 1024);
    EXPECT_FALSE(vt.get(0));
    EXPECT_TRUE(vt.set(0));
    EXPECT_TRUE(vt.get(0));
}

TEST(VisitedTable, Above16MUsesSparse)
{
    VisitedTable vt(SPARSE_TRIGGER);
    EXPECT_FALSE(vt.get(0));
    EXPECT_TRUE(vt.set(0));
}

// ============================================================
// Stress
// ============================================================

TEST(VisitedTable, MaxProbeLimitTerminates)
{
    VisitedTable vt(SPARSE_TRIGGER);
    const int N = 30000;
    for (int i = 0; i < N; i++)
        ASSERT_TRUE(vt.set(i));
    for (int i = 0; i < N; i++)
        EXPECT_TRUE(vt.get(i));
}

// ============================================================
// Prefetch
// ============================================================

TEST(VisitedTable, PrefetchDenseDoesNotCrash)
{
    VisitedTable vt(100);
    EXPECT_NO_THROW(vt.prefetch(50));
    EXPECT_NO_THROW(vt.prefetch(99));
}

TEST(VisitedTable, PrefetchSparseDoesNotCrash)
{
    VisitedTable vt(SPARSE_TRIGGER);
    EXPECT_NO_THROW(vt.prefetch(500));
}

TEST(VisitedTable, SparseProbeLimitAborts)
{
    // All multiples of 65536 map to the same hash slot (gcd(31153,65536)=1),
    // so linear probing fills consecutive slots.  At i=512, slots 0..511
    // are all occupied by different keys → MAX_PROBE=512 exceeded → abort.
    EXPECT_DEATH(
        {
            VisitedTable vt(SPARSE_TRIGGER);
            for (int i = 0; i < 600; i++)
                vt.set(i * 65536);
        },
        "probe limit");
}

// ============================================================
// Phase F Tier A: defensive error handling tests
// ============================================================

TEST(VisitedTable, SparseGetReturnsFalseOnProbeExhausted)
{
    // Fill 512 slots with same-hash keys so that get() for a 513th key
    // that hashes to the same slot exhausts MAX_PROBE (512) and returns false.
    // Keys of form k*65536 all hash to slot 0 (table size = 65536).
    VisitedTable vt(SPARSE_TRIGGER);
    for (int i = 0; i < 512; i++)
        ASSERT_TRUE(vt.set(i * 65536));
    // Query key 512*65536 — hashes to slot 0, not in table,
    // probes slots 0..511 all occupied by different keys → returns false
    EXPECT_FALSE(vt.get(512 * 65536));
}
