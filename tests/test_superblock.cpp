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

TEST(SuperBlockCRC, Fnv1a64EmptyInputReturnsOffset)
{
    uint64_t h = fnv1a64_update(FNV1A64_OFFSET, "", 0);
    EXPECT_EQ(h, FNV1A64_OFFSET);
}

TEST(SuperBlockCRC, Fnv1a64NonEmptyProducesDifferentHash)
{
    uint64_t h0 = fnv1a64_update(FNV1A64_OFFSET, "", 0);
    uint64_t h1 = fnv1a64_update(FNV1A64_OFFSET, "hello", 5);
    EXPECT_NE(h0, h1);
}

TEST(SuperBlockCRC, Fnv1a64IncrementalEqualsOneShot)
{
    const char *data = "the quick brown fox jumps over the lazy dog";
    size_t len = std::strlen(data);

    uint64_t oneshot = fnv1a64_update(FNV1A64_OFFSET, data, len);

    uint64_t state = FNV1A64_OFFSET;
    size_t half = len / 2;
    state = fnv1a64_update(state, data, half);
    state = fnv1a64_update(state, data + half, len - half);
    EXPECT_EQ(oneshot, state);
}

TEST(SuperBlockCRC, Fnv1a64DifferentDataDifferentHash)
{
    uint64_t a = fnv1a64_update(FNV1A64_OFFSET, "abc", 3);
    uint64_t b = fnv1a64_update(FNV1A64_OFFSET, "abd", 3);
    EXPECT_NE(a, b);
}

TEST(SuperBlockCRC, HeaderCRCZeroesHeaderCrc64Field)
{
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.dim = 128;
    sb.M = 16;
    sb.header_crc64 = 0xDEADBEEFCAFE0001; // garbage — should be ignored

    uint64_t crc1 = compute_header_crc64(sb);

    // Setting header_crc64 to a different value must produce same CRC
    sb.header_crc64 = 0x1234567890ABCDEF;
    uint64_t crc2 = compute_header_crc64(sb);

    EXPECT_EQ(crc1, crc2);
}

TEST(SuperBlockCRC, HeaderCRCDifferentFieldsDifferentResult)
{
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    uint64_t crc_a = compute_header_crc64(sb);

    sb.dim = 256;
    uint64_t crc_b = compute_header_crc64(sb);
    EXPECT_NE(crc_a, crc_b);
}

TEST(SuperBlockCRC, HeaderCRCDeterministic)
{
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.ntotal = 1000000;
    sb.dim = 128;
    sb.M = 32;
    sb.max_level = 5;

    uint64_t crc1 = compute_header_crc64(sb);
    uint64_t crc2 = compute_header_crc64(sb);
    EXPECT_EQ(crc1, crc2);
}

TEST(SuperBlockCRC, PayloadCRCEqualForIdenticalBlocks)
{
    std::vector<char> buf1(4096, 0);
    std::vector<char> buf2(4096, 0);
    // fill with identical payloads
    for (size_t i = 0; i < 4096; i++)
        buf1[i] = buf2[i] = static_cast<char>(i & 0xFF);

    SuperBlockV1 sb{};
    sb.blocks[0].off = 0;
    sb.blocks[0].bytes = 4096;

    uint64_t crc1 = compute_payload_crc64(buf1.data(), sb);
    uint64_t crc2 = compute_payload_crc64(buf2.data(), sb);
    EXPECT_EQ(crc1, crc2);
}

TEST(SuperBlockCRC, PayloadCRCDifferentForDifferentBlocks)
{
    std::vector<char> buf1(4096, 0xAA);
    std::vector<char> buf2(4096, 0xBB);

    SuperBlockV1 sb{};
    sb.blocks[0].off = 0;
    sb.blocks[0].bytes = 4096;

    uint64_t crc1 = compute_payload_crc64(buf1.data(), sb);
    uint64_t crc2 = compute_payload_crc64(buf2.data(), sb);
    EXPECT_NE(crc1, crc2);
}

TEST(SuperBlockCRC, PayloadCRCEmptyBlockProducesValidCRC)
{
    std::vector<char> buf(4096, 0);
    SuperBlockV1 sb{};
    // All blocks with zero length
    for (uint32_t i = 0; i < BLK_COUNT; i++) {
        sb.blocks[i].off = 0;
        sb.blocks[i].bytes = 0;
    }
    uint64_t crc = compute_payload_crc64(buf.data(), sb);
    // Hashing zero bytes for each block → accumulates FNV updates of empty strings
    EXPECT_NE(crc, 0ULL);
}

TEST(SuperBlockCRC, PayloadCRCOrderDependent)
{
    const size_t blk = 1024;
    std::vector<char> buf(blk * 2, 0);
    for (size_t i = 0; i < blk; i++) {
        buf[i] = 0x11;
        buf[blk + i] = 0x22;
    }

    SuperBlockV1 sb1{};
    sb1.blocks[0].off = 0;
    sb1.blocks[0].bytes = blk;
    sb1.blocks[1].off = blk;
    sb1.blocks[1].bytes = blk;
    uint64_t crc1 = compute_payload_crc64(buf.data(), sb1);

    // Swap block order: blk1 first, blk0 second
    std::vector<char> buf2(blk * 2, 0);
    for (size_t i = 0; i < blk; i++) {
        buf2[i] = 0x22;
        buf2[blk + i] = 0x11;
    }
    uint64_t crc2 = compute_payload_crc64(buf2.data(), sb1);
    EXPECT_NE(crc1, crc2);
}

TEST(SuperBlockCRC, FullRoundtrip)
{
    // Build a SuperBlock, compute both CRCs, verify self-consistency
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.state = static_cast<uint32_t>(GdState::BUILDING);
    sb.ntotal = 5000;
    sb.dim = 64;
    sb.M = 16;
    sb.ef_search = 128;
    sb.max_level = 3;
    sb.entry_point = 42;
    sb.num_shards = 1;
    sb.shard_id = 0;
    sb.ntotal_local = 5000;
    sb.global_id_begin = 0;

    // Simulate one block of payload data
    std::vector<char> payload(8192);
    for (size_t i = 0; i < payload.size(); i++)
        payload[i] = static_cast<char>((i * 7 + 13) & 0xFF);

    sb.blocks[BLK_VECTORS].off = 0;
    sb.blocks[BLK_VECTORS].bytes = payload.size();

    sb.payload_crc64 = compute_payload_crc64(payload.data(), sb);
    sb.header_crc64 = compute_header_crc64(sb);

    // Verify header CRC is non-zero and self-consistent
    EXPECT_NE(sb.header_crc64, 0ULL);
    EXPECT_NE(sb.payload_crc64, 0ULL);
    EXPECT_EQ(compute_header_crc64(sb), sb.header_crc64);
}

TEST(SuperBlockCRC, HeaderCRCIndependentOfPayloadCRC)
{
    // compute_header_crc64 zeroes both header_crc64 AND payload_crc64.
    // This avoids a circular dependency: header CRC must not depend on
    // payload CRC, otherwise updating the payload CRC would invalidate
    // the header CRC.
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.payload_crc64 = 0xAAAAAAAAAAAAAAAA;
    uint64_t crc1 = compute_header_crc64(sb);

    sb.payload_crc64 = 0xBBBBBBBBBBBBBBBB;
    uint64_t crc2 = compute_header_crc64(sb);
    EXPECT_EQ(crc1, crc2);
}

TEST(SuperBlockCRC, PayloadCRCIgnoresPaddingBetweenBlocks)
{
    // payload CRC hashes only each block's [off, off+bytes) range and must
    // skip alignment padding between blocks. A streaming implementation must
    // preserve this — i.e. it must not hash the whole file contiguously.
    std::vector<char> buf(4096, 0);
    SuperBlockV1 sb{}; // zero-init: blocks 2..BLK_COUNT-1 are empty (off=0, bytes=0)
    sb.blocks[0].off = 0;
    sb.blocks[0].bytes = 1024;
    sb.blocks[1].off = 2048; // bytes [1024, 2048) are padding
    sb.blocks[1].bytes = 1024;

    for (size_t i = 0; i < 1024; i++) {
        buf[i] = 0x11;
        buf[2048 + i] = 0x22;
    }
    uint64_t crc1 = compute_payload_crc64(buf.data(), sb);

    // Changing padding bytes must not change the CRC.
    for (size_t i = 1024; i < 2048; i++)
        buf[i] = 0xFF;
    uint64_t crc2 = compute_payload_crc64(buf.data(), sb);

    EXPECT_EQ(crc1, crc2);
}
