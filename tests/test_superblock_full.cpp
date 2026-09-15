/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gd_layout.h"

using namespace gd_hnsw;

// ============================================================
// State transition test helper
// ============================================================
// Builds a minimal valid SuperBlock buffer at a given state,
// with optional structural data fill for SEALED/ACTIVE states.

struct StateTestBuf {
    std::vector<char> buf;
    SuperBlockV1 *sb;

    explicit StateTestBuf(uint64_t n = 100, uint32_t dim = 16, uint32_t M = 16, GdState state = GdState::BUILDING)
    {
        SuperBlockV1 tmp{};
        tmp.magic = GD_MAGIC;
        tmp.schema_version = SCHEMA_VERSION;
        tmp.state = static_cast<uint32_t>(state);
        tmp.metric_type = METRIC_L2;
        tmp.endianness = ENDIAN_LITTLE;
        tmp.ntotal = n;
        tmp.ntotal_local = n;
        tmp.dim = dim;
        tmp.M = M;
        tmp.neighbors_size = n * M * 2;
        tmp.num_levels = 2;
        tmp.entry_point = 0;
        tmp.max_level = 0;
        tmp.num_shards = 0;

        if (state >= GdState::SEALED) {
            // compute_layout needs neighbors_size first
            compute_layout(tmp);
            buf.resize(tmp.total_bytes, 0);
            std::memcpy(buf.data(), &tmp, sizeof(SuperBlockV1));
            sb = reinterpret_cast<SuperBlockV1 *>(buf.data());
            fill_structural_data();
            sb->payload_crc64 = compute_payload_crc64(buf.data(), *sb);
            sb->header_crc64 = compute_header_crc64(*sb);
        } else {
            compute_layout(tmp);
            buf.resize(tmp.total_bytes, 0);
            std::memcpy(buf.data(), &tmp, sizeof(SuperBlockV1));
            sb = reinterpret_cast<SuperBlockV1 *>(buf.data());
        }
    }

    void fill_structural_data()
    {
        uint32_t nn_per_node = sb->M * 2;
        uint64_t n = sb->ntotal_local;

        int32_t *levels = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < n; i++)
            levels[i] = static_cast<int32_t>(sb->max_level + 1);

        uint64_t *offsets = reinterpret_cast<uint64_t *>(buf.data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t i = 0; i <= n; i++)
            offsets[i] = i * nn_per_node;

        int32_t *cum_nn = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(nn_per_node);

        int32_t *neighbors = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < sb->neighbors_size; i++)
            neighbors[i] = EMPTY_NEIGHBOR;
    }
};

// ============================================================
// Building state
// ============================================================

TEST(SuperBlockFull, BuildingStatePassesValidation)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(SuperBlockFull, BuildingStateSkipsCrcChecks)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();
    stb.sb->header_crc64 = 0xDEAD;
    stb.sb->payload_crc64 = 0xBEEF;
    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

// ============================================================
// BUILDING -> SEALED transition
// ============================================================

TEST(SuperBlockFull, TransitionBuildingToSealed)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(SuperBlockFull, SealedStateFailsWithBadHeaderCrc)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    // Corrupt header CRC
    stb.sb->header_crc64 ^= 1;
    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(SuperBlockFull, SealedStateFailsWithBadPayloadCrc)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    // Corrupt payload CRC
    stb.sb->payload_crc64 ^= 1;
    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_FALSE(r.ok);
}

// ============================================================
// SEALED -> ACTIVE transition
// ============================================================

TEST(SuperBlockFull, TransitionSealedToActive)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    // Seal
    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    // Activate: only header changed, payload unchanged
    stb.sb->state = static_cast<uint32_t>(GdState::ACTIVE);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(SuperBlockFull, ActiveStatePreservesPayloadCrcAfterSeal)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    uint64_t payload_crc = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->payload_crc64 = payload_crc;
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    // Transition to ACTIVE
    stb.sb->state = static_cast<uint32_t>(GdState::ACTIVE);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    // Payload CRC should still be valid (payload data didn't change)
    EXPECT_EQ(stb.sb->payload_crc64, payload_crc);
    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(SuperBlockFull, ActiveStateWithoutCrcFails)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    // Jump directly to ACTIVE without computing CRCs
    stb.sb->state = static_cast<uint32_t>(GdState::ACTIVE);
    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_FALSE(r.ok);
}

// ============================================================
// Edge cases
// ============================================================

TEST(SuperBlockFull, SingleNodeGraph)
{
    StateTestBuf stb(1, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();

    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(SuperBlockFull, MultipleShardsShareGlobalParams)
{
    const uint64_t ntotal = 200;
    const uint32_t dim = 16;
    const uint32_t M = 16;

    for (uint32_t s = 0; s < 2; s++) {
        SuperBlockV1 tmp{};
        tmp.magic = GD_MAGIC;
        tmp.schema_version = SCHEMA_VERSION;
        tmp.state = static_cast<uint32_t>(GdState::BUILDING);
        tmp.metric_type = METRIC_L2;
        tmp.endianness = ENDIAN_LITTLE;
        tmp.ntotal = ntotal;
        tmp.ntotal_local = 100;
        tmp.dim = dim;
        tmp.M = M;
        tmp.neighbors_size = 100 * M * 2;
        tmp.num_levels = 2;
        tmp.entry_point = 50;
        tmp.max_level = 0;
        tmp.num_shards = 2;
        tmp.shard_id = s;
        tmp.global_id_begin = s * 100;
        compute_layout(tmp);

        std::vector<char> buf(tmp.total_bytes, 0);
        std::memcpy(buf.data(), &tmp, sizeof(SuperBlockV1));

        auto r = validate_superblock(buf.data(), buf.size());
        EXPECT_TRUE(r.ok) << "shard " << s << ": " << r.error;
    }
}

TEST(SuperBlockFull, SealedStateFailsWithCorruptedPayloadData)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();
    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    // Flip a payload byte (neighbors block) — NOT the stored CRC field. The
    // recomputed payload CRC must mismatch and be reported before structure checks.
    uint64_t off = stb.sb->blocks[BLK_NEIGHBORS].off;
    stb.buf[off] ^= 0x01;

    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("payload_crc64"), std::string::npos);
}

TEST(SuperBlockFull, SealedStateFailsWithNonMonotonicOffsets)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();
    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);

    // Break offsets monotonicity, then recompute CRCs so the structure check
    // (not the CRC check) is what fires.
    uint64_t *offsets = reinterpret_cast<uint64_t *>(stb.buf.data() + stb.sb->blocks[BLK_OFFSETS].off);
    offsets[51] = offsets[50] - 1;

    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("not monotonic"), std::string::npos);
}

TEST(SuperBlockFull, SealedStateFailsWithOutOfRangeNeighbor)
{
    StateTestBuf stb(100, 16, 16, GdState::BUILDING);
    stb.fill_structural_data();
    stb.sb->state = static_cast<uint32_t>(GdState::SEALED);

    int32_t *neighbors = reinterpret_cast<int32_t *>(stb.buf.data() + stb.sb->blocks[BLK_NEIGHBORS].off);
    neighbors[0] = static_cast<int32_t>(stb.sb->ntotal) + 10; // >= ntotal

    stb.sb->payload_crc64 = compute_payload_crc64(stb.buf.data(), *stb.sb);
    stb.sb->header_crc64 = compute_header_crc64(*stb.sb);

    auto r = validate_superblock(stb.buf.data(), stb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("neighbor id out of range"), std::string::npos);
}
