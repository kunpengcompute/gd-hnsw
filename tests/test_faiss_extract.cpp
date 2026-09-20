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
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

#include <faiss/IndexHNSW.h>

#include "faiss_extractor.h"
#include "test_distance_common.hpp"

using namespace gd_hnsw;

// ============================================================
// Single shard extraction
// ============================================================

TEST(FaissExtract, BuildAndExtractSingleShard)
{
    uint64_t n = 200;
    uint32_t dim = 16;
    uint32_t M = 16;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    auto shards = FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, 1, M, 20, 16);
    ASSERT_EQ(shards.size(), 1u);

    const auto &buf = shards[0];
    ASSERT_GE(buf.size(), SUPERBLOCK_SIZE);

    const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
    EXPECT_EQ(sb->magic, GD_MAGIC);
    EXPECT_EQ(sb->schema_version, SCHEMA_VERSION);
    EXPECT_EQ(sb->state, static_cast<uint32_t>(GdState::SEALED));
    EXPECT_EQ(sb->metric_type, METRIC_L2);
    EXPECT_EQ(sb->endianness, ENDIAN_LITTLE);
    EXPECT_EQ(sb->ntotal, n);
    EXPECT_EQ(sb->ntotal_local, n);
    EXPECT_EQ(sb->dim, dim);
    EXPECT_EQ(sb->M, M);
    EXPECT_EQ(sb->num_shards, 1u);
    EXPECT_EQ(sb->shard_id, 0u);
    EXPECT_EQ(sb->global_id_begin, 0u);
    EXPECT_GT(sb->num_levels, 1u);

    auto vr = validate_superblock(buf.data(), buf.size());
    EXPECT_TRUE(vr.ok) << vr.error;
}

// ============================================================
// Multiple shards
// ============================================================

TEST(FaissExtract, BuildAndExtractMultipleShards)
{
    uint64_t n = 200;
    uint32_t dim = 16;
    uint32_t num_shards = 2;
    uint32_t M = 16;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    auto shards = FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, num_shards, M, 20, 16);
    ASSERT_EQ(shards.size(), num_shards);

    uint64_t total_local = 0;
    for (uint32_t s = 0; s < num_shards; s++) {
        const auto &buf = shards[s];
        ASSERT_GE(buf.size(), SUPERBLOCK_SIZE);

        const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        EXPECT_EQ(sb->shard_id, s);
        EXPECT_EQ(sb->num_shards, num_shards);
        EXPECT_GT(sb->ntotal_local, 0u);
        total_local += sb->ntotal_local;

        // All shards share the same global params
        EXPECT_EQ(sb->ntotal, n);
        EXPECT_EQ(sb->dim, dim);
        EXPECT_EQ(sb->M, M);

        auto vr = validate_superblock(buf.data(), buf.size());
        EXPECT_TRUE(vr.ok) << "shard " << s << ": " << vr.error;
    }
    EXPECT_EQ(total_local, n);
}

TEST(FaissExtract, ShardRangesDoNotOverlap)
{
    uint64_t n = 200;
    uint32_t dim = 16;
    uint32_t num_shards = 3;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    auto shards = FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, num_shards, 12, 20, 16);
    ASSERT_EQ(shards.size(), num_shards);

    uint64_t expected_begin = 0;
    for (uint32_t s = 0; s < num_shards; s++) {
        const auto *sb = reinterpret_cast<const SuperBlockV1 *>(shards[s].data());
        EXPECT_EQ(sb->global_id_begin, expected_begin) << "shard " << s << " has wrong global_id_begin";
        expected_begin += sb->ntotal_local;
    }
    EXPECT_EQ(expected_begin, n);
}

// ============================================================
// ID mapping
// ============================================================

TEST(FaissExtract, IdmapIsIdentityForRangePartition)
{
    uint64_t n = 100;
    uint32_t dim = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    std::vector<uint32_t> idmap;
    auto shards =
        FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, 1, 8, 20, 16, ShardBuildOptions{}, &idmap);

    ASSERT_EQ(idmap.size(), n);
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); i++)
        EXPECT_EQ(idmap[i], i);
    ASSERT_EQ(shards.size(), 1u);
}

// ============================================================
// Vectors are preserved
// ============================================================

TEST(FaissExtract, VectorsCopiedCorrectly)
{
    uint64_t n = 50;
    uint32_t dim = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    auto shards = FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, 1, 8, 20, 16);
    ASSERT_EQ(shards.size(), 1u);

    const auto &buf = shards[0];
    const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
    const float *extracted_vecs = reinterpret_cast<const float *>(buf.data() + sb->blocks[BLK_VECTORS].off);

    // Spot check: verify vectors match the input
    // Range partition preserves order, so first and last vectors should match
    for (uint32_t d = 0; d < dim; d++) {
        EXPECT_FLOAT_EQ(extracted_vecs[d], vectors[d]);
        EXPECT_FLOAT_EQ(extracted_vecs[(n - 1) * dim + d], vectors[(n - 1) * dim + d]);
    }
}

// ============================================================
// Entry point
// ============================================================

TEST(FaissExtract, EntryPointInRange)
{
    uint64_t n = 100;
    uint32_t dim = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    auto shards = FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, 1, 8, 20, 16);
    ASSERT_EQ(shards.size(), 1u);

    const auto *sb = reinterpret_cast<const SuperBlockV1 *>(shards[0].data());
    EXPECT_GE(sb->entry_point, 0);
    EXPECT_LT(sb->entry_point, static_cast<int32_t>(n));
}

// ============================================================
// Cluster partition
// ============================================================

TEST(FaissExtract, ClusterPartitionProducesValidShards)
{
    uint64_t n = 200;
    uint32_t dim = 8;
    uint32_t num_shards = 2;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    ShardBuildOptions opts;
    opts.cluster_partition = true;

    std::vector<uint32_t> idmap;
    auto shards =
        FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, num_shards, 8, 20, 16, opts, &idmap);
    ASSERT_EQ(shards.size(), num_shards);

    // idmap should be a permutation
    ASSERT_EQ(idmap.size(), n);
    std::vector<bool> seen(n, false);
    for (uint64_t i = 0; i < n; i++) {
        ASSERT_LT(idmap[i], n);
        EXPECT_FALSE(seen[idmap[i]]) << "duplicate at position " << i;
        seen[idmap[i]] = true;
    }

    // Each shard should validate
    uint64_t total_local = 0;
    for (uint32_t s = 0; s < num_shards; s++) {
        const auto &buf = shards[s];
        const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        EXPECT_GT(sb->ntotal_local, 0u);
        total_local += sb->ntotal_local;

        auto vr = validate_superblock(buf.data(), buf.size());
        EXPECT_TRUE(vr.ok) << "shard " << s << ": " << vr.error;
    }
    EXPECT_EQ(total_local, n);
}

// ============================================================
// Centroid output
// ============================================================

// ============================================================
// Error branch tests (A6)
// ============================================================

TEST(FaissExtract, EmptyVectorsThrows)
{
    uint32_t dim = 8;
    EXPECT_THROW(FaissExtractor::build_and_extract_sharded(nullptr, 0, dim, 1, 8, 20, 16), std::runtime_error);
}

TEST(FaissExtract, NumShardsZeroThrows)
{
    uint64_t n = 100;
    uint32_t dim = 8;
    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);
    EXPECT_THROW(FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, 0, 8, 20, 16), std::runtime_error);
}

TEST(FaissExtract, NumShardsExceeds64Throws)
{
    uint64_t n = 100;
    uint32_t dim = 8;
    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);
    EXPECT_THROW(FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, 65, 8, 20, 16), std::runtime_error);
}

TEST(FaissExtract, ClusterPartitionMoreShardsThanVectors)
{
    uint64_t n = 5;
    uint32_t dim = 8;
    uint32_t num_shards = 10; // more shards than vectors → empty clusters
    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);
    ShardBuildOptions opts;
    opts.cluster_partition = true;
    // faiss throws faiss::FaissException for "nx >= k" before our code
    // gets a chance to check for empty clusters; either exception is valid.
    EXPECT_ANY_THROW(FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, num_shards, 8, 20, 16, opts));
}

TEST(FaissExtract, ClusterPartitionOutputsCentroids)
{
    uint64_t n = 100;
    uint32_t dim = 8;
    uint32_t num_shards = 2;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    ShardBuildOptions opts;
    opts.cluster_partition = true;

    std::vector<float> centroids;
    auto shards = FaissExtractor::build_and_extract_sharded(vectors.data(), n, dim, num_shards, 8, 20, 16, opts,
                                                            nullptr, &centroids);

    ASSERT_EQ(shards.size(), num_shards);
    EXPECT_EQ(centroids.size(), num_shards * dim);
    // Centroids should not be all zeros
    bool non_zero = false;
    for (size_t i = 0; i < centroids.size(); i++) {
        if (centroids[i] != 0.0f) {
            non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(non_zero);
}

// ============================================================
// Streaming extraction tests (Phase C1)
// ============================================================

namespace {

std::string streaming_tmp_base(const char *tag)
{
    return "/tmp/gd_hnsw_test_" + std::string(tag) + "_" + std::to_string(getpid());
}

bool streaming_file_exists(const std::string &path)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    fclose(fp);
    return true;
}

size_t streaming_file_size(const std::string &path)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp)
        return 0;
    fseek(fp, 0, SEEK_END);
    size_t sz = static_cast<size_t>(ftell(fp));
    fclose(fp);
    return sz;
}

void streaming_cleanup(const std::string &base, uint32_t num_shards)
{
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = base + ".shard_" + std::to_string(s);
        std::remove(path.c_str());
    }
}

} // namespace

TEST(FaissExtract, StreamingBasicExtract)
{
    uint64_t n = 100;
    uint32_t dim = 8;
    uint32_t num_shards = 2;
    uint32_t M = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    std::string base = streaming_tmp_base("basic");
    auto byte_sizes =
        FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, num_shards, base, M, 20, 16);

    ASSERT_EQ(byte_sizes.size(), num_shards);

    uint64_t total_local = 0;
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = base + ".shard_" + std::to_string(s);
        EXPECT_TRUE(streaming_file_exists(path));
        EXPECT_EQ(byte_sizes[s], streaming_file_size(path));
        EXPECT_GT(byte_sizes[s], 0u);

        std::vector<char> buf(byte_sizes[s]);
        FILE *fp = fopen(path.c_str(), "rb");
        ASSERT_NE(fp, nullptr);
        fread(buf.data(), 1, byte_sizes[s], fp);
        fclose(fp);

        const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        EXPECT_EQ(sb->shard_id, s);
        EXPECT_EQ(sb->num_shards, num_shards);
        EXPECT_EQ(sb->ntotal, n);
        total_local += sb->ntotal_local;

        auto vr = validate_superblock(buf.data(), buf.size());
        EXPECT_TRUE(vr.ok) << "shard " << s << ": " << vr.error;
    }
    EXPECT_EQ(total_local, n);

    streaming_cleanup(base, num_shards);
}

TEST(FaissExtract, StreamingSingleShard)
{
    uint64_t n = 80;
    uint32_t dim = 8;
    uint32_t M = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    std::string base = streaming_tmp_base("single");
    auto byte_sizes = FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, 1, base, M, 20, 16);

    ASSERT_EQ(byte_sizes.size(), 1u);
    std::string path = base + ".shard_0";
    EXPECT_TRUE(streaming_file_exists(path));

    std::vector<char> buf(byte_sizes[0]);
    FILE *fp = fopen(path.c_str(), "rb");
    ASSERT_NE(fp, nullptr);
    fread(buf.data(), 1, byte_sizes[0], fp);
    fclose(fp);

    const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
    EXPECT_EQ(sb->ntotal_local, n);
    auto vr = validate_superblock(buf.data(), buf.size());
    EXPECT_TRUE(vr.ok) << vr.error;

    streaming_cleanup(base, 1);
}

TEST(FaissExtract, StreamingWithClusterPartition)
{
    uint64_t n = 100;
    uint32_t dim = 8;
    uint32_t num_shards = 2;
    uint32_t M = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    ShardBuildOptions opts;
    opts.cluster_partition = true;

    std::string base = streaming_tmp_base("cluster");
    auto byte_sizes =
        FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, num_shards, base, M, 20, 16, opts);

    ASSERT_EQ(byte_sizes.size(), num_shards);

    uint64_t total_local = 0;
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = base + ".shard_" + std::to_string(s);
        EXPECT_TRUE(streaming_file_exists(path));

        std::vector<char> buf(byte_sizes[s]);
        FILE *fp = fopen(path.c_str(), "rb");
        ASSERT_NE(fp, nullptr);
        fread(buf.data(), 1, byte_sizes[s], fp);
        fclose(fp);

        const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        EXPECT_GT(sb->ntotal_local, 0u);
        total_local += sb->ntotal_local;
    }
    EXPECT_EQ(total_local, n);

    streaming_cleanup(base, num_shards);
}

TEST(FaissExtract, StreamingOutputsIdmap)
{
    uint64_t n = 80;
    uint32_t dim = 8;
    uint32_t M = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    std::vector<uint32_t> idmap;
    std::string base = streaming_tmp_base("idmap");
    auto byte_sizes = FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, 2, base, M, 20, 16,
                                                                          ShardBuildOptions{}, &idmap);

    ASSERT_EQ(idmap.size(), n);
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); i++)
        EXPECT_EQ(idmap[i], i);

    streaming_cleanup(base, 2);
}

TEST(FaissExtract, StreamingOutputsCentroids)
{
    uint64_t n = 100;
    uint32_t dim = 8;
    uint32_t num_shards = 2;
    uint32_t M = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    ShardBuildOptions opts;
    opts.cluster_partition = true;

    std::vector<float> centroids;
    std::string base = streaming_tmp_base("centroids");
    auto byte_sizes = FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, num_shards, base, M,
                                                                          20, 16, opts, nullptr, &centroids);

    EXPECT_EQ(centroids.size(), num_shards * dim);
    bool non_zero = false;
    for (size_t i = 0; i < centroids.size(); i++) {
        if (centroids[i] != 0.0f) {
            non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(non_zero);

    streaming_cleanup(base, num_shards);
}

TEST(FaissExtract, StreamingPostAddCallback)
{
    uint64_t n = 50;
    uint32_t dim = 8;
    uint32_t M = 8;

    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);

    bool callback_called = false;
    ShardBuildOptions opts;
    opts.post_add_callback = [&]() { callback_called = true; };

    std::string base = streaming_tmp_base("callback");
    auto byte_sizes =
        FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, 1, base, M, 20, 16, opts);

    EXPECT_TRUE(callback_called);
    streaming_cleanup(base, 1);
}

TEST(FaissExtract, StreamingEmptyVectorsThrows)
{
    EXPECT_THROW(
        FaissExtractor::build_and_extract_sharded_streaming(nullptr, 0, 8, 1, "/tmp/nonexistent_stream", 8, 20, 16),
        std::runtime_error);
}

TEST(FaissExtract, StreamingNumShardsZeroThrows)
{
    uint64_t n = 50;
    uint32_t dim = 8;
    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);
    EXPECT_THROW(FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, 0,
                                                                     "/tmp/nonexistent_stream", 8, 20, 16),
                 std::runtime_error);
}

TEST(FaissExtract, StreamingNumShardsExceeds64Throws)
{
    uint64_t n = 50;
    uint32_t dim = 8;
    std::vector<float> vectors(n * dim);
    fill_random_vec(vectors.data(), n * dim);
    EXPECT_THROW(FaissExtractor::build_and_extract_sharded_streaming(vectors.data(), n, dim, 65,
                                                                     "/tmp/nonexistent_stream", 8, 20, 16),
                 std::runtime_error);
}

// ============================================================
// extract_sharded parameter validation (P2 coverage boost)
//
// Constructs a real faiss::IndexHNSWFlat, then calls extract_sharded
// with invalid old_gid_of_new / shard_sizes to trigger the throw
// branches that build_and_extract cannot reach.
// ============================================================

namespace {

// Build a small HNSW index for extract_sharded parameter tests.
// Returns the index and the vectors used to build it.
struct TestIndex {
    std::unique_ptr<faiss::IndexHNSWFlat> index;
    std::vector<float> vectors;
    uint64_t n;
    uint32_t dim;
};

TestIndex make_test_index(uint64_t n = 100, uint32_t dim = 8, uint32_t M = 8)
{
    TestIndex ti;
    ti.n = n;
    ti.dim = dim;
    ti.vectors.resize(n * dim);
    fill_random_vec(ti.vectors.data(), n * dim);
    ti.index = std::make_unique<faiss::IndexHNSWFlat>(static_cast<int>(dim), static_cast<int>(M));
    ti.index->hnsw.efConstruction = 20;
    ti.index->hnsw.efSearch = 16;
    ti.index->add(static_cast<faiss::idx_t>(n), ti.vectors.data());
    return ti;
}

} // namespace

// --- old_gid_of_new size mismatch ---

TEST(FaissExtract, ExtractOldGidSizeMismatch)
{
    auto ti = make_test_index();
    std::vector<uint32_t> bad_map(ti.n - 1); // size != ntotal
    EXPECT_THROW(FaissExtractor::extract_sharded(*ti.index, 1, 16, &bad_map), std::runtime_error);
}

// --- old_gid_of_new contains out-of-range id ---

TEST(FaissExtract, ExtractOldGidOutOfRange)
{
    auto ti = make_test_index();
    std::vector<uint32_t> bad_map(ti.n);
    std::iota(bad_map.begin(), bad_map.end(), 0);
    bad_map[0] = static_cast<uint32_t>(ti.n); // out of range
    EXPECT_THROW(FaissExtractor::extract_sharded(*ti.index, 1, 16, &bad_map), std::runtime_error);
}

// --- old_gid_of_new is not a permutation (duplicate old id) ---

TEST(FaissExtract, ExtractOldGidNotPermutation)
{
    auto ti = make_test_index();
    std::vector<uint32_t> bad_map(ti.n, 0); // all zeros → duplicate
    EXPECT_THROW(FaissExtractor::extract_sharded(*ti.index, 1, 16, &bad_map), std::runtime_error);
}

// --- shard_sizes length mismatch ---

TEST(FaissExtract, ExtractShardSizesLengthMismatch)
{
    auto ti = make_test_index();
    std::vector<uint32_t> idmap(ti.n);
    std::iota(idmap.begin(), idmap.end(), 0);
    std::vector<uint64_t> bad_sizes = {ti.n}; // only 1 entry, but num_shards=2
    EXPECT_THROW(FaissExtractor::extract_sharded(*ti.index, 2, 16, &idmap, &bad_sizes), std::runtime_error);
}

// --- shard_sizes sum does not match ntotal ---

TEST(FaissExtract, ExtractShardSizesSumMismatch)
{
    auto ti = make_test_index();
    std::vector<uint32_t> idmap(ti.n);
    std::iota(idmap.begin(), idmap.end(), 0);
    std::vector<uint64_t> bad_sizes = {ti.n / 2, ti.n / 2 + 1}; // sum = n+1
    EXPECT_THROW(FaissExtractor::extract_sharded(*ti.index, 2, 16, &idmap, &bad_sizes), std::runtime_error);
}
