/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include <gtest/gtest.h>
#include <cstdint>

// Doorbell encoding replicated from dist_service.h
static constexpr uint64_t DOORBELL_SEQ_MASK = 0x0000FFFFFFFFFFFF;

static inline uint64_t doorbell_encode(uint64_t seq, uint16_t count)
{
    return (static_cast<uint64_t>(count) << 48) | (seq & DOORBELL_SEQ_MASK);
}
static inline uint64_t doorbell_seq(uint64_t doorbell) { return doorbell & DOORBELL_SEQ_MASK; }
static inline uint16_t doorbell_count(uint64_t doorbell) { return static_cast<uint16_t>(doorbell >> 48); }

TEST(Doorbell, EncodeCountInHigh16)
{
    uint64_t db = doorbell_encode(0x1234, 0xABCD);
    EXPECT_EQ(doorbell_count(db), 0xABCD);
}

TEST(Doorbell, EncodeSeqInLow48)
{
    uint64_t db = doorbell_encode(0x123456789ABC, 0);
    EXPECT_EQ(doorbell_seq(db), 0x123456789ABC);
}

TEST(Doorbell, SeqAndCountDoNotOverlap)
{
    uint64_t db = doorbell_encode(0xFFFFFFFFFFFF, 0xFFFF);
    EXPECT_EQ(doorbell_seq(db), 0xFFFFFFFFFFFF);
    EXPECT_EQ(doorbell_count(db), 0xFFFF);
}

TEST(Doorbell, DoorbellSeqExtractsLow48)
{
    uint64_t db = (0xAAAAULL << 48) | 0x123456789ABCULL;
    EXPECT_EQ(doorbell_seq(db), 0x123456789ABC);
}

TEST(Doorbell, DoorbellCountExtractsHigh16)
{
    uint64_t db = (0xBBBBULL << 48) | 0x123456789ABCULL;
    EXPECT_EQ(doorbell_count(db), 0xBBBB);
}

TEST(Doorbell, Roundtrip)
{
    auto check = [](uint64_t seq, uint16_t count) {
        uint64_t db = doorbell_encode(seq, count);
        EXPECT_EQ(doorbell_seq(db), seq);
        EXPECT_EQ(doorbell_count(db), count);
    };
    check(0, 0);
    check(1, 1);
    check(0xFFFFFFFFFFFF, 0xFFFF);
    check(0x123456789ABC, 0x5678);
    check(0, 0xFFFF);
    check(0xFFFFFFFFFFFF, 0);
}

TEST(Doorbell, SeqWraparoundNear48BitLimit)
{
    uint64_t seq = 0xFFFFFFFFFFFF;
    uint64_t db = doorbell_encode(seq, 42);
    EXPECT_EQ(doorbell_seq(db), seq);
    EXPECT_EQ(doorbell_count(db), 42);
}

TEST(Doorbell, CountZeroIsValid)
{
    uint64_t db = doorbell_encode(1, 0);
    EXPECT_EQ(doorbell_seq(db), 1);
    EXPECT_EQ(doorbell_count(db), 0);
}

TEST(Doorbell, OddSeqIsRequest)
{
    EXPECT_EQ(doorbell_seq(doorbell_encode(1, 1)) & 1, 1u);
    EXPECT_EQ(doorbell_seq(doorbell_encode(3, 1)) & 1, 1u);
}

TEST(Doorbell, EvenSeqIsIdle) { EXPECT_EQ(doorbell_seq(doorbell_encode(0, 0)) & 1, 0u); }

TEST(Doorbell, SeqMaskLimitsTo48Bits)
{
    uint64_t db = doorbell_encode(0x0001FFFFFFFFFFFFULL, 0);
    EXPECT_EQ(doorbell_seq(db), 0x0000FFFFFFFFFFFFULL);
}
