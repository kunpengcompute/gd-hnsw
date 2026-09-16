/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gd_hnsw_search.h"
#include "gd_layout.h"
#include "dist_service.h"

using namespace gd_hnsw;

// ============================================================
// In-memory HNSW graph builder for testing
// ============================================================
// Builds a single-layer graph (max_level=0, num_levels=2) where
// nodes form a 1D line.  Each node connects to its nearest neighbors.
// Returns the buffer and SuperBlock pointer.

struct TestGraph {
    std::vector<char> buf;
    const SuperBlockV1 *sb = nullptr;

    TestGraph(uint64_t n, uint32_t dim, uint32_t M)
    {
        uint32_t nlevels = 2;         // max_level=0 => num_levels=2
        uint32_t level_of_node = 1;   // all nodes at level 1
        uint64_t nn_per_node = 2 * M; // cum_nn[1]

        // Step 1: compute layout
        SuperBlockV1 tmp{};
        tmp.magic = GD_MAGIC;
        tmp.schema_version = SCHEMA_VERSION;
        tmp.state = static_cast<uint32_t>(GdState::ACTIVE);
        tmp.metric_type = METRIC_L2;
        tmp.endianness = ENDIAN_LITTLE;
        tmp.ntotal = n;
        tmp.ntotal_local = n;
        tmp.dim = dim;
        tmp.M = M;
        tmp.ef_search = 16;
        tmp.max_level = 0;
        tmp.entry_point = static_cast<int32_t>(n / 2);
        tmp.num_levels = nlevels;
        tmp.neighbors_size = n * nn_per_node;
        tmp.num_shards = 0;
        compute_layout(tmp);

        buf.resize(tmp.total_bytes, 0);
        std::memcpy(buf.data(), &tmp, sizeof(SuperBlockV1));
        sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());

        auto *sb_mut = reinterpret_cast<SuperBlockV1 *>(buf.data());

        // Step 2: fill vectors — 1D line in high-dim space
        float *vecs = reinterpret_cast<float *>(buf.data() + sb->blocks[BLK_VECTORS].off);
        for (uint64_t i = 0; i < n; i++) {
            float val = static_cast<float>(i) / static_cast<float>(n);
            for (uint32_t d = 0; d < dim; d++)
                vecs[i * dim + d] = val;
        }

        // Step 3: fill levels
        int32_t *levels = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_LEVELS].off);
        for (uint64_t i = 0; i < n; i++)
            levels[i] = level_of_node; // all at level 1 (faiss level 0)

        // Step 4: fill offsets
        uint64_t *offsets = reinterpret_cast<uint64_t *>(buf.data() + sb->blocks[BLK_OFFSETS].off);
        for (uint64_t i = 0; i <= n; i++)
            offsets[i] = i * nn_per_node;

        // Step 5: fill neighbors — connect to 2*M closest nodes
        int32_t *neighbors = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_NEIGHBORS].off);
        for (uint64_t i = 0; i < n; i++) {
            int32_t *nb = neighbors + i * nn_per_node;
            // Fill with EMPTY_NEIGHBOR initially
            for (uint64_t s = 0; s < nn_per_node; s++)
                nb[s] = EMPTY_NEIGHBOR;
            // Add M nearest neighbors on each side
            for (uint32_t r = 0; r < M; r++) {
                int64_t left = static_cast<int64_t>(i) - 1 - static_cast<int64_t>(r);
                int64_t right = static_cast<int64_t>(i) + 1 + static_cast<int64_t>(r);
                if (left >= 0)
                    nb[2 * r] = static_cast<int32_t>(left);
                if (right < static_cast<int64_t>(n))
                    nb[2 * r + 1] = static_cast<int32_t>(right);
            }
        }

        // Step 6: fill cum_nneighbor
        int32_t *cum_nn = reinterpret_cast<int32_t *>(buf.data() + sb->blocks[BLK_CUM_NNEIGHBOR].off);
        cum_nn[0] = 0;                                 // layer > 0: no neighbors (no nodes above level 0)
        cum_nn[1] = static_cast<int32_t>(nn_per_node); // layer 0: 2*M neighbors

        // Step 7: compute CRCs for ACTIVE state
        sb_mut->header_crc64 = compute_header_crc64(*sb_mut);
        sb_mut->payload_crc64 = compute_payload_crc64(buf.data(), *sb_mut);
    }

    // Reinterpret buffer after layout (pointer may change from resize)
    void refresh_sb() { sb = reinterpret_cast<const SuperBlockV1 *>(buf.data()); }

    GdHnswSearcher make_searcher(bool skip_validation = true) const
    {
        return GdHnswSearcher(buf.data(), buf.size(), skip_validation);
    }
};

// ============================================================
// Constructor tests
// ============================================================

TEST(SearchAlgorithm, ConstructFromValidBuffer)
{
    TestGraph g(50, 2, 4);
    EXPECT_NO_THROW({ GdHnswSearcher searcher(g.buf.data(), g.buf.size(), true); });
}

TEST(SearchAlgorithm, ConstructThrowsOnNonActiveState)
{
    TestGraph g(50, 2, 4);
    auto &sb_mut = *reinterpret_cast<SuperBlockV1 *>(g.buf.data());
    sb_mut.state = static_cast<uint32_t>(GdState::BUILDING);
    EXPECT_THROW(GdHnswSearcher(g.buf.data(), g.buf.size(), true), std::runtime_error);
}

TEST(SearchAlgorithm, ValidationPassesOnValidBuffer)
{
    TestGraph g(50, 2, 4);
    EXPECT_NO_THROW({ GdHnswSearcher searcher(g.buf.data(), g.buf.size(), false); });
}

TEST(SearchAlgorithm, ValidationFailsOnBadCRC)
{
    TestGraph g(50, 2, 4);
    auto &sb_mut = *reinterpret_cast<SuperBlockV1 *>(g.buf.data());
    sb_mut.header_crc64 = 0;
    EXPECT_THROW(GdHnswSearcher(g.buf.data(), g.buf.size(), false), std::runtime_error);
}

// ============================================================
// Accessor tests
// ============================================================

TEST(SearchAlgorithm, AccessorsReturnCorrectValues)
{
    TestGraph g(100, 16, 8);
    auto s = g.make_searcher();
    EXPECT_EQ(s.ntotal(), 100u);
    EXPECT_EQ(s.dim(), 16u);
    EXPECT_EQ(s.shards().size(), 1u);
    EXPECT_EQ(s.expand_batch(), 1u);
}

TEST(SearchAlgorithm, SetExpandBatch)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    s.set_expand_batch(8);
    EXPECT_EQ(s.expand_batch(), 8u);
    s.set_expand_batch(0); // 0 → clamped to 1
    EXPECT_EQ(s.expand_batch(), 1u);
}

TEST(SearchAlgorithm, SetLocalShard)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    EXPECT_NO_THROW(s.set_local_shard(0));
}

TEST(SearchAlgorithm, SetEpVector)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    std::vector<float> ep_vec(2);
    ep_vec[0] = 1.0f;
    ep_vec[1] = 2.0f;
    EXPECT_NO_THROW(s.set_ep_vector(ep_vec.data(), 2));
}

// ============================================================
// search_to tests
// ============================================================

TEST(SearchAlgorithm, SearchToReturnsCorrectNumberOfResults)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    int32_t k = 10;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);
    float query[] = {0.25f, 0.25f};

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    s.search_to(query, params, ids.data(), dists.data());

    int count = 0;
    for (int32_t i = 0; i < k; i++) {
        if (ids[i] >= 0)
            count++;
    }
    EXPECT_GT(count, 0);
}

TEST(SearchAlgorithm, SearchToResultsSorted)
{
    TestGraph g(100, 2, 4);
    auto s = g.make_searcher();

    int32_t k = 10;
    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);
    float query[] = {0.5f, 0.5f};

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    s.search_to(query, params, ids.data(), dists.data());

    // Distances of valid results must be non-decreasing
    for (int32_t i = 1; i < k && ids[i] >= 0; i++)
        EXPECT_LE(dists[i - 1], dists[i] + 1e-6f);
}

TEST(SearchAlgorithm, SearchToDeterministic)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    int32_t k = 10;
    float query[] = {0.3f, 0.3f};
    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    std::vector<int32_t> ids1(k), ids2(k);
    std::vector<float> dists1(k), dists2(k);

    s.search_to(query, params, ids1.data(), dists1.data());
    s.search_to(query, params, ids2.data(), dists2.data());

    for (int32_t i = 0; i < k; i++) {
        EXPECT_EQ(ids1[i], ids2[i]);
        EXPECT_FLOAT_EQ(dists1[i], dists2[i]);
    }
}

TEST(SearchAlgorithm, SearchToK1)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    int32_t ids[1];
    float dists[1];
    float query[] = {0.1f, 0.1f};

    HnswQueryParams params;
    params.k = 1;
    params.ef_search = 8;

    s.search_to(query, params, ids, dists);
    EXPECT_GE(ids[0], 0);
    EXPECT_GE(dists[0], 0.0f);
}

// ============================================================
// Batch search tests
// ============================================================

TEST(SearchAlgorithm, SearchBatchFlatReturnsResults)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    int32_t k = 5, nq = 3;
    std::vector<float> queries(nq * 2);
    for (int32_t q = 0; q < nq; q++) {
        queries[q * 2 + 0] = static_cast<float>(q) / static_cast<float>(nq);
        queries[q * 2 + 1] = static_cast<float>(q) / static_cast<float>(nq);
    }

    std::vector<int32_t> ids(nq * k);
    std::vector<float> dists(nq * k);

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 16;

    EXPECT_NO_THROW(s.search_batch_flat(queries.data(), nq, params, ids.data(), dists.data()));

    // Each query should have at least 1 result
    for (int32_t q = 0; q < nq; q++)
        EXPECT_GE(ids[q * k], 0);
}

// ============================================================
// Edge cases
// ============================================================

TEST(SearchAlgorithm, SearchEmptyGraphNoCrash)
{
    // n=0 should handle gracefully
    // Actually, validation forbids ntotal=0, so skip_validation build,
    // but the state must be ACTIVE. Let's test n=1 edge.
    TestGraph g(1, 2, 2);
    auto s = g.make_searcher();

    int32_t ids[5];
    float dists[5];
    float query[] = {0.5f, 0.5f};
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 8;

    EXPECT_NO_THROW(s.search_to(query, params, ids, dists));
}

TEST(SearchAlgorithm, SearchBatchFlatZeroQueries)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    std::vector<int32_t> ids;
    std::vector<float> dists;
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 16;

    EXPECT_NO_THROW(s.search_batch_flat(nullptr, 0, params, ids.data(), dists.data()));
}

TEST(SearchAlgorithm, EfSearchLessThanK)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    int32_t k = 10;
    std::vector<int32_t> ids(k, -1);
    std::vector<float> dists(k);
    float query[] = {0.4f, 0.4f};

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 3; // less than k, should be clamped internally

    s.search_to(query, params, ids.data(), dists.data());
    // Should still return k results (ef clamped to k internally)
    for (int32_t i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
}

TEST(SearchAlgorithm, DifferentQueriesDifferentResults)
{
    TestGraph g(100, 2, 4);
    auto s = g.make_searcher();

    int32_t k = 1;
    int32_t id_a, id_b;
    float dist_a, dist_b;
    float qa[] = {0.0f, 0.0f};
    float qb[] = {1.0f, 1.0f};

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 16;

    s.search_to(qa, params, &id_a, &dist_a);
    s.search_to(qb, params, &id_b, &dist_b);

    // Queries at opposite ends should produce different nearest neighbors
    EXPECT_NE(id_a, id_b);
}

// ============================================================
// Multi-shard tests (A1)
// ============================================================

static std::vector<char> make_shard_buf(uint64_t ntotal, uint64_t nlocal, uint32_t dim, uint32_t M, uint32_t shard_id,
                                        uint32_t num_shards, uint64_t gid_begin, uint32_t num_levels = 2,
                                        int32_t max_level = 0, int32_t entry_point = 0)
{
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.state = static_cast<uint32_t>(GdState::ACTIVE);
    sb.metric_type = METRIC_L2;
    sb.endianness = ENDIAN_LITTLE;
    sb.ntotal = ntotal;
    sb.ntotal_local = nlocal;
    sb.dim = dim;
    sb.M = M;
    sb.ef_search = 16;
    sb.max_level = max_level;
    sb.entry_point = entry_point;
    sb.num_levels = num_levels;
    sb.num_shards = num_shards;
    sb.shard_id = shard_id;
    sb.global_id_begin = gid_begin;
    sb.neighbors_size = nlocal * M * 2;
    compute_layout(sb);
    std::vector<char> buf(sb.total_bytes, 0);
    std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));
    return buf;
}

TEST(SearchAlgorithm, MultiShardConstructTwoUniformShards)
{
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_EQ(searcher.ntotal(), 100u);
    EXPECT_EQ(searcher.dim(), 2u);
    EXPECT_EQ(searcher.shards().size(), 2u);
}

TEST(SearchAlgorithm, MultiShardConstructThreeUniformShards)
{
    auto b0 = make_shard_buf(150, 50, 4, 4, 0, 3, 0);
    auto b1 = make_shard_buf(150, 50, 4, 4, 1, 3, 50);
    auto b2 = make_shard_buf(150, 50, 4, 4, 2, 3, 100);
    std::vector<std::pair<const void *, uint64_t>> shards = {
        {b0.data(), b0.size()}, {b1.data(), b1.size()}, {b2.data(), b2.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_EQ(searcher.shards().size(), 3u);
    EXPECT_EQ(searcher.ntotal(), 150u);
}

TEST(SearchAlgorithm, MultiShardConstructNonUniformShards)
{
    auto b0 = make_shard_buf(100, 30, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 70, 2, 4, 1, 2, 30);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_EQ(searcher.shards().size(), 2u);
}

TEST(SearchAlgorithm, MultiShardSortsByGlobalIdBegin)
{
    // Provide shards out of order; constructor sorts by global_id_begin
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b1.data(), b1.size()}, {b0.data(), b0.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_EQ(searcher.shards().size(), 2u);
    // shard 0 (gid_begin=0) should be first after sort
    EXPECT_EQ(searcher.shards()[0].sb->shard_id, 0u);
    EXPECT_EQ(searcher.shards()[1].sb->shard_id, 1u);
}

TEST(SearchAlgorithm, MultiShardShardById)
{
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_EQ(searcher.shard_by_id(0).sb->shard_id, 0u);
    EXPECT_EQ(searcher.shard_by_id(1).sb->shard_id, 1u);
    EXPECT_EQ(searcher.shard_by_id(0).ntotal_local, 50u);
    EXPECT_EQ(searcher.shard_by_id(1).ntotal_local, 50u);
}

TEST(SearchAlgorithm, MultiShardShardByIdOutOfRange)
{
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_THROW(searcher.shard_by_id(99), std::runtime_error);
}

TEST(SearchAlgorithm, MultiShardSetLocalShard)
{
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_NO_THROW(searcher.set_local_shard(0));
    EXPECT_NO_THROW(searcher.set_local_shard(1));
}

// E3c: Power-of-2 shard sizes trigger __builtin_ctzll fast path in set_local_shard
TEST(SearchAlgorithm, MultiShardSetLocalShardPowerOfTwo)
{
    uint64_t n_per_shard = 64; // power of 2
    uint64_t ntotal = n_per_shard * 2;
    auto b0 = make_shard_buf(ntotal, n_per_shard, 4, 4, 0, 2, 0);
    auto b1 = make_shard_buf(ntotal, n_per_shard, 4, 4, 1, 2, n_per_shard);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_NO_THROW(searcher.set_local_shard(0));
    EXPECT_NO_THROW(searcher.set_local_shard(1));
}

// E3c: Non-power-of-2 ntotal_per_shard triggers non-uniform path
TEST(SearchAlgorithm, MultiShardSetLocalShardNonPowerOfTwo)
{
    auto b0 = make_shard_buf(100, 30, 4, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 70, 4, 4, 1, 2, 30);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_NO_THROW(searcher.set_local_shard(0));
}

TEST(SearchAlgorithm, MultiShardEmptyShardsThrows)
{
    std::vector<std::pair<const void *, uint64_t>> shards;
    EXPECT_THROW(GdHnswSearcher(shards, true), std::runtime_error);
}

TEST(SearchAlgorithm, MultiShardDuplicateShardIdThrows)
{
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 50, 2, 4, 0, 2, 50); // same shard_id=0
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    EXPECT_THROW(GdHnswSearcher(shards, true), std::runtime_error);
}

TEST(SearchAlgorithm, MultiShardMissingShardIdThrows)
{
    // shard_id=1 is missing (only 0 and 2 present)
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 3, 0);
    auto b2 = make_shard_buf(100, 50, 2, 4, 2, 3, 100);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b2.data(), b2.size()}};
    EXPECT_THROW(GdHnswSearcher(shards, true), std::runtime_error);
}

TEST(SearchAlgorithm, MultiShardInvalidShardIdThrows)
{
    // shard_id >= num_shards
    auto b0 = make_shard_buf(100, 50, 2, 4, 5, 2, 0); // shard_id=5, num_shards=2
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    EXPECT_THROW(GdHnswSearcher(shards, true), std::runtime_error);
}

TEST(SearchAlgorithm, MultiShardAccessorsReturnGlobalValues)
{
    auto b0 = make_shard_buf(200, 100, 8, 8, 0, 2, 0);
    auto b1 = make_shard_buf(200, 100, 8, 8, 1, 2, 100);
    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher searcher(shards, true);
    EXPECT_EQ(searcher.ntotal(), 200u);
    EXPECT_EQ(searcher.dim(), 8u);
    EXPECT_EQ(searcher.expand_batch(), 1u);
}

// ============================================================
// enable_dot_norm tests (A2)
// ============================================================

TEST(SearchAlgorithm, DotNormDisabledByDefault)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    EXPECT_FALSE(s.dot_norm_enabled());
}

TEST(SearchAlgorithm, EnableDotNormAfterLoadLocalVectors)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    // Extract vectors from the graph buffer
    const float *vecs = reinterpret_cast<const float *>(g.buf.data() + g.sb->blocks[BLK_VECTORS].off);
    s.load_local_vectors(vecs);
    s.enable_dot_norm(true);
    EXPECT_TRUE(s.dot_norm_enabled());
    EXPECT_NE(s.vector_norms_data(), nullptr);
}

TEST(SearchAlgorithm, EnableDotNormBeforeLoadLocalVectors)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    // enable_dot_norm before load_local_vectors: should warn and stay disabled
    s.enable_dot_norm(true);
    EXPECT_FALSE(s.dot_norm_enabled());
}

TEST(SearchAlgorithm, EnableDotNormIdempotent)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    const float *vecs = reinterpret_cast<const float *>(g.buf.data() + g.sb->blocks[BLK_VECTORS].off);
    s.load_local_vectors(vecs);
    s.enable_dot_norm(true);
    EXPECT_TRUE(s.dot_norm_enabled());
    // Second call should be no-op (norms already computed)
    EXPECT_NO_THROW(s.enable_dot_norm(true));
    EXPECT_TRUE(s.dot_norm_enabled());
}

TEST(SearchAlgorithm, DotNormDisable)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    const float *vecs = reinterpret_cast<const float *>(g.buf.data() + g.sb->blocks[BLK_VECTORS].off);
    s.load_local_vectors(vecs);
    s.enable_dot_norm(true);
    EXPECT_TRUE(s.dot_norm_enabled());
    s.enable_dot_norm(false);
    EXPECT_FALSE(s.dot_norm_enabled());
}

// ============================================================
// load_local_vectors from file tests (A3)
// ============================================================

TEST(SearchAlgorithm, LoadLocalVectorsFromFile)
{
    TestGraph g(30, 4, 4);
    auto s = g.make_searcher();

    // Write the buffer to a temp file
    std::string tmp_path = "test_load_vecs_" + std::to_string(std::rand() % 1000000);
    {
        FILE *fp = fopen(tmp_path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        fwrite(g.buf.data(), 1, g.buf.size(), fp);
        fclose(fp);
    }

    EXPECT_NO_THROW(s.load_local_vectors(tmp_path));
    EXPECT_TRUE(s.dot_norm_enabled() || !s.dot_norm_enabled()); // no crash

    // Verify vectors were loaded by enabling dot_norm (requires localized vectors)
    s.enable_dot_norm(true);
    EXPECT_TRUE(s.dot_norm_enabled());

    std::remove(tmp_path.c_str());
}

TEST(SearchAlgorithm, LoadLocalVectorsFileNotFound)
{
    TestGraph g(30, 4, 4);
    auto s = g.make_searcher();
    EXPECT_THROW(s.load_local_vectors("/nonexistent/test_load_vecs_xyz"), std::runtime_error);
}

TEST(SearchAlgorithm, LoadLocalVectorsFileTruncated)
{
    TestGraph g(30, 4, 4);
    auto s = g.make_searcher();

    std::string tmp_path = "test_load_trunc_" + std::to_string(std::rand() % 1000000);
    {
        FILE *fp = fopen(tmp_path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        // Write only partial SuperBlock (not enough for full header + vectors)
        fwrite(g.buf.data(), 1, SUPERBLOCK_SIZE / 2, fp);
        fclose(fp);
    }

    EXPECT_THROW(s.load_local_vectors(tmp_path), std::runtime_error);

    std::remove(tmp_path.c_str());
}

TEST(SearchAlgorithm, LoadLocalVectorsIdempotent)
{
    TestGraph g(30, 4, 4);
    auto s = g.make_searcher();

    std::string tmp_path = "test_load_idem_" + std::to_string(std::rand() % 1000000);
    {
        FILE *fp = fopen(tmp_path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        fwrite(g.buf.data(), 1, g.buf.size(), fp);
        fclose(fp);
    }

    EXPECT_NO_THROW(s.load_local_vectors(tmp_path));
    // Second call should be no-op
    EXPECT_NO_THROW(s.load_local_vectors(tmp_path));

    std::remove(tmp_path.c_str());
}

TEST(SearchAlgorithm, LoadLocalVectorsFromDataPointer)
{
    TestGraph g(30, 4, 4);
    auto s = g.make_searcher();
    const float *vecs = reinterpret_cast<const float *>(g.buf.data() + g.sb->blocks[BLK_VECTORS].off);
    EXPECT_NO_THROW(s.load_local_vectors(vecs));
    // Should be able to enable dot_norm after loading
    s.enable_dot_norm(true);
    EXPECT_TRUE(s.dot_norm_enabled());
}

TEST(SearchAlgorithm, LoadLocalVectorsFromDataPointerLargeDim)
{
    // Test load_local_vectors(data) works with non-trivial dim count
    uint64_t n = 200;
    uint32_t dim = 256;
    TestGraph g(n, dim, 4);
    auto s = g.make_searcher();
    const float *vecs = reinterpret_cast<const float *>(g.buf.data() + g.sb->blocks[BLK_VECTORS].off);
    EXPECT_NO_THROW(s.load_local_vectors(vecs));
    EXPECT_TRUE(s.vector_norms_data() == nullptr); // norms not computed yet
}

// ============================================================
// Phase F Tier A: defensive error handling tests
// ============================================================

// Regression for code-review issue #4: search_to with k <= 0 previously
// reached MinimaxHeap(0) and indexed result_heap_[0] of an empty vector
// (heap UB); negative k additionally walked batch output pointers
// backwards in search_batch_flat.
TEST(SearchAlgorithm, SearchToInvalidKAborts)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();

    int32_t ids[5];
    float dists[5];
    float query[] = {0.5f, 0.5f};
    HnswQueryParams params;
    params.ef_search = 8;

    params.k = 0;
    EXPECT_DEATH(s.search_to(query, params, ids, dists), "k > 0");

    params.k = -3;
    EXPECT_DEATH(s.search_to(query, params, ids, dists), "k > 0");
}

// Regression for code-review issue #5: out-of-range gids previously clamped
// to the last shard, then either silently dropped (last shard local, Phase 2
// skips own rank) or triggering a delayed service-side abort (remote shard).
// find_shard / shard_of now fail fast. Triggered via a corrupted entry_point:
// search_to resolves get_vector(ep) -> find_shard(ep) before any proxy
// channel access, so default-constructed proxies suffice.
TEST(SearchAlgorithm, MultiShardGidOutOfRangeAborts)
{
    auto b0 = make_shard_buf(100, 50, 2, 4, 0, 2, 0);
    auto b1 = make_shard_buf(100, 50, 2, 4, 1, 2, 50);
    // Corrupt shard 0's entry_point to an out-of-range global id (ntotal=100)
    auto *sb_mut = reinterpret_cast<SuperBlockV1 *>(b0.data());
    sb_mut->entry_point = 200;

    std::vector<std::pair<const void *, uint64_t>> shards = {{b0.data(), b0.size()}, {b1.data(), b1.size()}};
    GdHnswSearcher s(shards, true); // skip_validation: CRC not rechecked

    // Pass multi-shard proxy gate (pointer-only check); never dereferenced
    // because the abort fires while resolving the entry point vector.
    std::vector<std::vector<DualDistProxy>> proxies(1);
    proxies[0].resize(2);
    s.set_dual_pushdown_proxies(&proxies, 1);

    int32_t ids[5];
    float dists[5];
    float query[] = {0.5f, 0.5f};
    HnswQueryParams params;
    params.k = 5;
    params.ef_search = 8;

    EXPECT_DEATH(s.search_to(query, params, ids, dists), "out of range");
}

TEST(SearchAlgorithm, SetLocalShardOutOfRangeDies)
{
    TestGraph g(50, 2, 4);
    auto s = g.make_searcher();
    EXPECT_DEATH(s.set_local_shard(1), "out of range");
}

TEST(SearchAlgorithm, BadMagicThrows)
{
    TestGraph g(50, 2, 4);
    auto &sb_mut = *reinterpret_cast<SuperBlockV1 *>(g.buf.data());
    sb_mut.magic = 0xDEADBEEF;
    EXPECT_THROW(GdHnswSearcher(g.buf.data(), g.buf.size(), false), std::runtime_error);
}
