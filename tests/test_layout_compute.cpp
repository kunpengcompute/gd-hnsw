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
#include <cstring>
#include <vector>

#include "gd_layout.h"

using namespace gd_hnsw;

// ============================================================
// compute_layout tests
// ============================================================

static SuperBlockV1 make_sb(uint64_t n, uint32_t dim, uint32_t M, uint64_t neighbors_size, uint32_t num_levels)
{
    SuperBlockV1 sb{};
    sb.ntotal = n;
    sb.ntotal_local = n;
    sb.dim = dim;
    sb.M = M;
    sb.neighbors_size = neighbors_size;
    sb.num_levels = num_levels;
    sb.num_shards = 0;
    return sb;
}

TEST(LayoutCompute, ProducesFiveBlocks)
{
    auto sb = make_sb(1000, 128, 16, 30000, 8);
    compute_layout(sb);
    for (uint32_t i = 0; i < BLK_COUNT; i++)
        EXPECT_GT(sb.blocks[i].bytes, 0u);
}

TEST(LayoutCompute, VectorsAndNeighborsAlignedTo2MB)
{
    auto sb = make_sb(1000, 128, 16, 30000, 8);
    compute_layout(sb);
    EXPECT_EQ(sb.blocks[BLK_VECTORS].off % HUGEPAGE_ALIGN, 0u);
    EXPECT_EQ(sb.blocks[BLK_VECTORS].alignment, HUGEPAGE_ALIGN);
    EXPECT_EQ(sb.blocks[BLK_NEIGHBORS].off % HUGEPAGE_ALIGN, 0u);
    EXPECT_EQ(sb.blocks[BLK_NEIGHBORS].alignment, HUGEPAGE_ALIGN);
}

TEST(LayoutCompute, LevelsOffsetsCumNnAlignedTo4KB)
{
    auto sb = make_sb(1000, 128, 16, 30000, 8);
    compute_layout(sb);
    EXPECT_EQ(sb.blocks[BLK_LEVELS].off % PAGE_ALIGN, 0u);
    EXPECT_EQ(sb.blocks[BLK_LEVELS].alignment, PAGE_ALIGN);
    EXPECT_EQ(sb.blocks[BLK_OFFSETS].off % PAGE_ALIGN, 0u);
    EXPECT_EQ(sb.blocks[BLK_OFFSETS].alignment, PAGE_ALIGN);
    EXPECT_EQ(sb.blocks[BLK_CUM_NNEIGHBOR].off % PAGE_ALIGN, 0u);
    EXPECT_EQ(sb.blocks[BLK_CUM_NNEIGHBOR].alignment, PAGE_ALIGN);
}

TEST(LayoutCompute, NoBlockOverlap)
{
    auto sb = make_sb(1000, 128, 16, 30000, 8);
    compute_layout(sb);
    for (uint32_t i = 0; i < BLK_COUNT; i++) {
        uint64_t end_i = sb.blocks[i].off + sb.blocks[i].bytes;
        for (uint32_t j = i + 1; j < BLK_COUNT; j++) {
            uint64_t end_j = sb.blocks[j].off + sb.blocks[j].bytes;
            EXPECT_TRUE(end_i <= sb.blocks[j].off || end_j <= sb.blocks[i].off)
                << "blocks " << i << " and " << j << " overlap";
        }
    }
}

TEST(LayoutCompute, TotalBytesCoversLastBlock)
{
    auto sb = make_sb(1000, 128, 16, 30000, 8);
    compute_layout(sb);
    uint64_t last_end = sb.blocks[BLK_CUM_NNEIGHBOR].off + sb.blocks[BLK_CUM_NNEIGHBOR].bytes;
    EXPECT_GE(sb.total_bytes, last_end);
    EXPECT_EQ(sb.total_bytes % PAGE_ALIGN, 0u);
}

TEST(LayoutCompute, FirstBlockStartsAfterSuperBlock)
{
    auto sb = make_sb(1000, 128, 16, 30000, 8);
    compute_layout(sb);
    EXPECT_GE(sb.blocks[BLK_VECTORS].off, SUPERBLOCK_SIZE);
}

TEST(LayoutCompute, UsesNtotalLocalWhenSharded)
{
    SuperBlockV1 sb{};
    sb.ntotal = 10000;
    sb.ntotal_local = 1000;
    sb.dim = 64;
    sb.M = 16;
    sb.neighbors_size = 5000;
    sb.num_levels = 8;
    sb.num_shards = 4;
    compute_layout(sb);
    // vectors block should use ntotal_local (1000), not ntotal (10000)
    EXPECT_EQ(sb.blocks[BLK_VECTORS].bytes, 1000u * 64 * sizeof(float));
    EXPECT_EQ(sb.blocks[BLK_LEVELS].bytes, 1000u * sizeof(int32_t));
}

TEST(LayoutCompute, BlockSizesCorrect)
{
    auto sb = make_sb(500, 32, 8, 2000, 5);
    compute_layout(sb);
    EXPECT_EQ(sb.blocks[BLK_VECTORS].bytes, 500u * 32 * sizeof(float));
    EXPECT_EQ(sb.blocks[BLK_LEVELS].bytes, 500u * sizeof(int32_t));
    EXPECT_EQ(sb.blocks[BLK_OFFSETS].bytes, (500u + 1) * sizeof(uint64_t));
    EXPECT_EQ(sb.blocks[BLK_NEIGHBORS].bytes, 2000u * sizeof(int32_t));
    EXPECT_EQ(sb.blocks[BLK_CUM_NNEIGHBOR].bytes, 5u * sizeof(int32_t));
}

// ============================================================
// validate_superblock tests
// ============================================================

// Helper: build a minimal valid superblock in a buffer
struct ValidSB {
    std::vector<char> buf;
    SuperBlockV1 *sb;

    explicit ValidSB(uint64_t n = 100, uint32_t dim = 16, uint32_t M = 16)
    {
        // Compute layout first
        SuperBlockV1 tmp{};
        tmp.magic = GD_MAGIC;
        tmp.schema_version = SCHEMA_VERSION;
        tmp.state = static_cast<uint32_t>(GdState::BUILDING);
        tmp.metric_type = METRIC_L2;
        tmp.endianness = ENDIAN_LITTLE;
        tmp.ntotal = n;
        tmp.dim = dim;
        tmp.M = M;
        tmp.ntotal_local = n;
        tmp.neighbors_size = n * M * 2; // typical: n * 2M
        tmp.num_levels = 2;
        tmp.entry_point = 0;
        tmp.max_level = 0;
        tmp.num_shards = 0;
        compute_layout(tmp);

        buf.resize(tmp.total_bytes, 0);
        std::memcpy(buf.data(), &tmp, sizeof(SuperBlockV1));
        sb = reinterpret_cast<SuperBlockV1 *>(buf.data());
    }

    // Fill minimal valid structural data so SEALED validation passes
    void fill_structural_data()
    {
        uint32_t nn_per_node = sb->M * 2;
        uint64_t n = sb->ntotal_local;

        // levels: all nodes at max_level+1 (=1, since max_level=0, num_levels=2)
        int32_t *levels = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < n; i++)
            levels[i] = static_cast<int32_t>(sb->max_level + 1);

        // offsets: monotonic, offsets[n] == neighbors_size
        uint64_t *offsets = reinterpret_cast<uint64_t *>(buf.data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t i = 0; i <= n; i++)
            offsets[i] = i * nn_per_node;

        // cum_nneighbor: cum_nn[0]=0, cum_nn[1]=nn_per_node
        int32_t *cum_nn = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;
        cum_nn[1] = static_cast<int32_t>(nn_per_node);

        // neighbors: fill with EMPTY_NEIGHBOR (-1)
        int32_t *neighbors = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < sb->neighbors_size; i++)
            neighbors[i] = EMPTY_NEIGHBOR;
    }
};

TEST(ValidateSuperblock, PassesOnValidBuildingState)
{
    ValidSB vsb;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(ValidateSuperblock, FailsOnBadMagic)
{
    ValidSB vsb;
    vsb.sb->magic = 0xDEADBEEF;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("magic"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnWrongSchemaVersion)
{
    ValidSB vsb;
    vsb.sb->schema_version = 99;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("schema"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnInvalidState)
{
    ValidSB vsb;
    vsb.sb->state = 99;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("state"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnDimZero)
{
    ValidSB vsb(100, 0);
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnDimTooLarge)
{
    ValidSB vsb(100, 65537);
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnNtotalZero)
{
    ValidSB vsb(0);
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("ntotal"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnGdTooSmall)
{
    ValidSB vsb(100, 16, 16);
    auto r = validate_superblock(vsb.buf.data(), SUPERBLOCK_SIZE - 1);
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnTotalBytesExceedsGdSize)
{
    ValidSB vsb;
    vsb.sb->total_bytes = vsb.buf.size() + 4096; // inflate total_bytes
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("total_bytes"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnBlockOutOfBounds)
{
    ValidSB vsb;
    vsb.sb->blocks[BLK_VECTORS].bytes = vsb.buf.size() + 1;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("block exceeds"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnBlockAlignmentViolated)
{
    ValidSB vsb;
    vsb.sb->blocks[BLK_VECTORS].off = 64; // not aligned to 2MB
    vsb.sb->blocks[BLK_VECTORS].alignment = HUGEPAGE_ALIGN;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("alignment"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnMOutOfRange)
{
    ValidSB vsb;
    vsb.sb->M = 0;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("M out of range"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnEntryPointOutOfRange)
{
    ValidSB vsb(100);
    vsb.sb->entry_point = 999; // >= ntotal
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnShardIdOutOfRange)
{
    ValidSB vsb;
    vsb.sb->num_shards = 2;
    vsb.sb->shard_id = 5;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("shard_id"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnNtotalLocalZero)
{
    ValidSB vsb;
    vsb.sb->num_shards = 2;
    vsb.sb->shard_id = 0;
    vsb.sb->ntotal_local = 0;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("ntotal_local"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnHeadersCrcMismatchWhenSealed)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    // Set state to SEALED with intentionally wrong CRC
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = 0xBAD; // wrong
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("header_crc64"), std::string::npos);
}

TEST(ValidateSuperblock, PassesWithCorrectCRCWhenSealed)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

TEST(ValidateSuperblock, FailsOnPayloadCrcMismatchWhenSealed)
{
    ValidSB vsb;
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = 0xCAFE; // wrong
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("payload_crc64"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnEndianness)
{
    ValidSB vsb;
    vsb.sb->endianness = 0xFF;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnNumLevelsLessThanTwo)
{
    ValidSB vsb;
    vsb.sb->num_levels = 1;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnNonL2Metric)
{
    ValidSB vsb;
    vsb.sb->metric_type = 99; // not METRIC_L2
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("L2"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnNtotalExceedsInt32Max)
{
    ValidSB vsb;
    vsb.sb->ntotal = static_cast<uint64_t>(INT32_MAX) + 1;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
}

TEST(ValidateSuperblock, FailsOnMaxLevelOutOfRange)
{
    ValidSB vsb(100, 16, 16);
    vsb.sb->num_levels = 3;
    vsb.sb->max_level = 5; // max_level+1 (6) >= num_levels (3)
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("max_level"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnMaxLevelNegative)
{
    ValidSB vsb(100, 16, 16);
    vsb.sb->max_level = -1;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("max_level"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnNeighborsBlockSizeMismatch)
{
    ValidSB vsb;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error; // baseline passes
    // Inflate neighbors block bytes without updating neighbors_size
    vsb.sb->blocks[BLK_NEIGHBORS].bytes += 4;
    r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("neighbors block size mismatch"), std::string::npos);
}

TEST(ValidateSuperblock, PassesOnNumShardsZero)
{
    // num_shards=0 means unsharded, ntotal_local not checked separately
    ValidSB vsb;
    vsb.sb->num_shards = 0;
    vsb.sb->ntotal_local = 0; // ignored when num_shards=0
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
}

// ============================================================
// Phase F Tier A: defensive error handling tests
// ============================================================

TEST(ValidateSuperblock, FailsOnShardRangeExceedsNtotal)
{
    ValidSB vsb(100, 16, 16);
    vsb.sb->num_shards = 2;
    vsb.sb->shard_id = 0;
    vsb.sb->ntotal_local = 50;
    vsb.sb->global_id_begin = 80; // 80 + 50 = 130 > ntotal (100)
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("shard range"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnVectorsBlockSizeMismatch)
{
    ValidSB vsb;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
    vsb.sb->blocks[BLK_VECTORS].bytes += 4; // inflate without changing ntotal_local
    r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("vectors block size mismatch"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnLevelsBlockSizeMismatch)
{
    ValidSB vsb;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
    vsb.sb->blocks[BLK_LEVELS].bytes += 4;
    r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("levels block size mismatch"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnOffsetsBlockSizeMismatch)
{
    ValidSB vsb;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
    vsb.sb->blocks[BLK_OFFSETS].bytes += 8;
    r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("offsets block size mismatch"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnCumNneighborBlockSizeMismatch)
{
    ValidSB vsb;
    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_TRUE(r.ok) << r.error;
    vsb.sb->blocks[BLK_CUM_NNEIGHBOR].bytes += 4;
    r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("cum_nneighbor block size mismatch"), std::string::npos);
}

// ============================================================
// Deep structural validation (state >= SEALED) — Phase D2
// ============================================================

TEST(ValidateSuperblock, FailsOnNonMonotonicOffsets)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    // Corrupt offsets: make them non-monotonic at position 40
    uint64_t *offsets = reinterpret_cast<uint64_t *>(vsb.buf.data() + vsb.sb->blocks[BLK_OFFSETS].off);
    offsets[40] = offsets[39] - 1; // decreasing
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("offsets not monotonic"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnOffsetsEndNotEqualNeighborsSize)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    // Corrupt last offset
    uint64_t *offsets = reinterpret_cast<uint64_t *>(vsb.buf.data() + vsb.sb->blocks[BLK_OFFSETS].off);
    offsets[vsb.sb->ntotal_local] = vsb.sb->neighbors_size - 1;
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("offsets[n_local]"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnInvalidLevelForNode)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    // Set a node's level to 0 (invalid: must be > 0)
    int32_t *levels = reinterpret_cast<int32_t *>(vsb.buf.data() + vsb.sb->blocks[BLK_LEVELS].off);
    levels[10] = 0;
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("invalid level"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnLevelExceedsNumLevels)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    // Set a node's level >= num_levels
    int32_t *levels = reinterpret_cast<int32_t *>(vsb.buf.data() + vsb.sb->blocks[BLK_LEVELS].off);
    levels[5] = static_cast<int32_t>(vsb.sb->num_levels);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("invalid level"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnOffsetsSlotCountMismatch)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    // Make offsets span not match expected cum_nn[level]
    uint64_t *offsets = reinterpret_cast<uint64_t *>(vsb.buf.data() + vsb.sb->blocks[BLK_OFFSETS].off);
    // Node 0: original spans from offsets[0]=0 to offsets[1]=nn_per_node
    // Enlarge it to break the match
    offsets[1] = offsets[0] + 1; // now 1 slot, but cum_nn[level] = nn_per_node
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("slot count mismatch"), std::string::npos);
}

TEST(ValidateSuperblock, FailsOnNeighborIdOutOfRange)
{
    ValidSB vsb;
    vsb.fill_structural_data();
    vsb.sb->state = static_cast<uint32_t>(GdState::SEALED);
    vsb.sb->header_crc64 = compute_header_crc64(*vsb.sb);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    // Put an out-of-range neighbor id
    int32_t *neighbors = reinterpret_cast<int32_t *>(vsb.buf.data() + vsb.sb->blocks[BLK_NEIGHBORS].off);
    neighbors[0] = static_cast<int32_t>(vsb.sb->ntotal + 10);
    vsb.sb->payload_crc64 = compute_payload_crc64(vsb.buf.data(), *vsb.sb);

    auto r = validate_superblock(vsb.buf.data(), vsb.buf.size());
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("neighbor id"), std::string::npos);
}
