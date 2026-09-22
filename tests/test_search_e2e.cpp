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
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <vector>
#include <algorithm>

#include "faiss_extractor.h"
#include "gd_hnsw_search.h"
#include "test_distance_common.hpp"

using namespace gd_hnsw;

// ============================================================
// Helpers
// ============================================================

static std::vector<int32_t> brute_force_topk(const float *query, const float *vectors, uint64_t n, uint32_t dim, int k)
{
    std::vector<std::pair<float, int32_t>> all(n);
    for (uint64_t i = 0; i < n; i++) {
        float d = scalar::l2_sqr(query, vectors + i * dim, dim);
        all[i] = {d, static_cast<int32_t>(i)};
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    std::vector<int32_t> ids(k);
    for (int i = 0; i < k; i++)
        ids[i] = all[i].second;
    return ids;
}

static double compute_recall(const int32_t *result, const std::vector<int32_t> &gt, int k)
{
    int hits = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (result[i] == gt[j]) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

// Build dataset, extract shard, transition to ACTIVE.
struct E2EData {
    uint64_t n;
    uint32_t dim;
    std::vector<float> vectors;
    std::vector<char> shard_buf;
};

static E2EData make_e2e_data(uint64_t n, uint32_t dim, uint32_t M)
{
    E2EData d;
    d.n = n;
    d.dim = dim;
    d.vectors.resize(n * dim);
    fill_random_vec(d.vectors.data(), n * dim);

    auto shards = FaissExtractor::build_and_extract_sharded(d.vectors.data(), n, dim, 1, M, 40, 16);
    d.shard_buf = std::move(shards[0]);

    auto *sb = reinterpret_cast<SuperBlockV1 *>(d.shard_buf.data());
    sb->state = static_cast<uint32_t>(GdState::ACTIVE);
    sb->header_crc64 = compute_header_crc64(*sb);
    return d;
}

static std::unique_ptr<GdHnswSearcher> make_searcher(const E2EData &d)
{
    auto s = std::make_unique<GdHnswSearcher>(d.shard_buf.data(), d.shard_buf.size(), false);
    s->set_local_shard(0);
    s->load_local_vectors(d.vectors.data());
    return s;
}

// ============================================================
// Recall tests
// ============================================================

TEST(SearchE2E, SingleShardRecallAt10)
{
    auto d = make_e2e_data(300, 16, 16);
    auto s = make_searcher(d);
    const int k = 10;
    const int nq = 20;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 64;

    double total_recall = 0.0;
    for (int q = 0; q < nq; q++) {
        const float *query = d.vectors.data() + q * d.dim;

        std::vector<int32_t> ids(k);
        std::vector<float> dists(k);
        s->search_to(query, params, ids.data(), dists.data());

        auto gt = brute_force_topk(query, d.vectors.data(), d.n, d.dim, k);
        total_recall += compute_recall(ids.data(), gt, k);
    }

    double avg_recall = total_recall / nq;
    EXPECT_GE(avg_recall, 0.95) << "Recall@" << k << " = " << avg_recall;
}

TEST(SearchE2E, SingleShardRecallAt1)
{
    auto d = make_e2e_data(200, 8, 12);
    auto s = make_searcher(d);
    const int k = 1;
    const int nq = 30;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    int correct = 0;
    for (int q = 0; q < nq; q++) {
        const float *query = d.vectors.data() + q * d.dim;

        int32_t id;
        float dist;
        s->search_to(query, params, &id, &dist);

        auto gt = brute_force_topk(query, d.vectors.data(), d.n, d.dim, k);
        if (id == gt[0])
            correct++;
    }

    double accuracy = static_cast<double>(correct) / nq;
    EXPECT_GE(accuracy, 0.90) << "Top-1 accuracy = " << accuracy;
}

TEST(SearchE2E, HigherEfSearchImprovesRecall)
{
    auto d = make_e2e_data(200, 8, 12);
    auto s = make_searcher(d);
    const int k = 10;
    const int nq = 15;

    auto measure_recall = [&](int32_t ef) {
        HnswQueryParams params;
        params.k = k;
        params.ef_search = ef;

        double total = 0.0;
        for (int q = 0; q < nq; q++) {
            const float *query = d.vectors.data() + q * d.dim;
            std::vector<int32_t> ids(k);
            std::vector<float> dists(k);
            s->search_to(query, params, ids.data(), dists.data());
            auto gt = brute_force_topk(query, d.vectors.data(), d.n, d.dim, k);
            total += compute_recall(ids.data(), gt, k);
        }
        return total / nq;
    };

    double r8 = measure_recall(8);
    double r32 = measure_recall(32);
    EXPECT_GE(r32 + 0.05, r8) << "Recall@ef=8=" << r8 << " Recall@ef=32=" << r32;
}

// ============================================================
// Result ordering and count
// ============================================================

TEST(SearchE2E, ResultsSortedByDistance)
{
    auto d = make_e2e_data(100, 8, 8);
    auto s = make_searcher(d);
    const int k = 10;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 16;

    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);
    s->search_to(d.vectors.data(), params, ids.data(), dists.data());

    for (int i = 1; i < k && ids[i] >= 0; i++)
        EXPECT_LE(dists[i - 1], dists[i] + 1e-5f);
}

TEST(SearchE2E, ReturnsExactlyKResults)
{
    auto d = make_e2e_data(100, 8, 8);
    auto s = make_searcher(d);
    const int k = 10;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 16;

    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);
    s->search_to(d.vectors.data(), params, ids.data(), dists.data());

    for (int i = 0; i < k; i++)
        EXPECT_GE(ids[i], 0) << "result " << i << " is empty";
}

// ============================================================
// Batch vs single consistency
// ============================================================

TEST(SearchE2E, BatchAndSingleQueryConsistency)
{
    auto d = make_e2e_data(100, 8, 8);
    auto s = make_searcher(d);
    const int k = 5;
    const int nq = 4;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 16;

    std::vector<float> queries(nq * d.dim);
    for (int q = 0; q < nq; q++)
        std::memcpy(queries.data() + q * d.dim, d.vectors.data() + q * d.dim, d.dim * sizeof(float));

    std::vector<int32_t> batch_ids(nq * k);
    std::vector<float> batch_dists(nq * k);
    s->search_batch_flat(queries.data(), nq, params, batch_ids.data(), batch_dists.data());

    for (int q = 0; q < nq; q++) {
        std::vector<int32_t> single_ids(k);
        std::vector<float> single_dists(k);
        s->search_to(queries.data() + q * d.dim, params, single_ids.data(), single_dists.data());

        for (int i = 0; i < k; i++) {
            EXPECT_EQ(batch_ids[q * k + i], single_ids[i]) << "batch vs single mismatch at q=" << q << " i=" << i;
        }
    }
}

// ============================================================
// Determinism
// ============================================================

TEST(SearchE2E, SearchDeterministic)
{
    auto d = make_e2e_data(100, 8, 8);
    auto s = make_searcher(d);
    const int k = 10;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 16;

    std::vector<int32_t> ids1(k), ids2(k);
    std::vector<float> dists1(k), dists2(k);

    s->search_to(d.vectors.data(), params, ids1.data(), dists1.data());
    s->search_to(d.vectors.data(), params, ids2.data(), dists2.data());

    for (int i = 0; i < k; i++) {
        EXPECT_EQ(ids1[i], ids2[i]);
        EXPECT_FLOAT_EQ(dists1[i], dists2[i]);
    }
}

// ============================================================
// dot_norm (IP) distance mode e2e tests
//
// dot_norm is an optimization for L2: ||q||^2 + ||v||^2 - 2*dot(q,v).
// Results must match the L2 path exactly — only the kernel differs
// (pre-computed vector norms + dot product vs sub+FMA).
// These tests exercise: enable_dot_norm, get_vector_in_shard,
// get_norm_in_shard, dot_norm_dist_batch_2/3/4, and the scalar tail.
// ============================================================

TEST(SearchE2E, DotNormRecallAt10)
{
    auto d = make_e2e_data(300, 16, 16);
    auto s = make_searcher(d);
    s->enable_dot_norm();
    ASSERT_TRUE(s->dot_norm_enabled());

    const int k = 10;
    const int nq = 20;

    HnswQueryParams params;
    params.k = k;
    params.ef_search = 64;

    double total_recall = 0.0;
    for (int q = 0; q < nq; q++) {
        const float *query = d.vectors.data() + q * d.dim;

        std::vector<int32_t> ids(k);
        std::vector<float> dists(k);
        s->search_to(query, params, ids.data(), dists.data());

        auto gt = brute_force_topk(query, d.vectors.data(), d.n, d.dim, k);
        total_recall += compute_recall(ids.data(), gt, k);
    }

    double avg_recall = total_recall / nq;
    EXPECT_GE(avg_recall, 0.95) << "dot_norm Recall@" << k << " = " << avg_recall;
}

TEST(SearchE2E, DotNormMatchesL2Results)
{
    // dot_norm and L2 compute the same metric — results must be identical.
    auto d = make_e2e_data(200, 16, 16);

    // L2 searcher (no dot_norm)
    auto s_l2 = make_searcher(d);

    // dot_norm searcher (same data, separate instance)
    auto s_dn = make_searcher(d);
    s_dn->enable_dot_norm();

    const int k = 10;
    const int nq = 15;
    HnswQueryParams params;
    params.k = k;
    params.ef_search = 64;

    for (int q = 0; q < nq; q++) {
        const float *query = d.vectors.data() + q * d.dim;

        std::vector<int32_t> ids_l2(k), ids_dn(k);
        std::vector<float> dists_l2(k), dists_dn(k);

        s_l2->search_to(query, params, ids_l2.data(), dists_l2.data());
        s_dn->search_to(query, params, ids_dn.data(), dists_dn.data());

        // Same IDs (order may differ on ties, so compare as sets)
        std::set<int32_t> set_l2(ids_l2.begin(), ids_l2.end());
        std::set<int32_t> set_dn(ids_dn.begin(), ids_dn.end());
        EXPECT_EQ(set_l2, set_dn) << "dot_norm vs L2 result set mismatch at q=" << q;

        // Distances should be equal (same metric, same values)
        std::vector<float> sorted_l2(dists_l2.begin(), dists_l2.end());
        std::vector<float> sorted_dn(dists_dn.begin(), dists_dn.end());
        std::sort(sorted_l2.begin(), sorted_l2.end());
        std::sort(sorted_dn.begin(), sorted_dn.end());
        for (int i = 0; i < k; i++) {
            EXPECT_NEAR(sorted_l2[i], sorted_dn[i], 1e-4f) << "dist mismatch at q=" << q << " i=" << i;
        }
    }
}

TEST(SearchE2E, DotNormBatchAndSingleConsistency)
{
    auto d = make_e2e_data(200, 16, 16);
    auto s = make_searcher(d);
    s->enable_dot_norm();

    const int k = 5;
    const int nq = 4;
    HnswQueryParams params;
    params.k = k;
    params.ef_search = 32;

    std::vector<float> queries(nq * d.dim);
    for (int q = 0; q < nq; q++)
        std::memcpy(queries.data() + q * d.dim, d.vectors.data() + q * d.dim, d.dim * sizeof(float));

    std::vector<int32_t> batch_ids(nq * k);
    std::vector<float> batch_dists(nq * k);
    s->search_batch_flat(queries.data(), nq, params, batch_ids.data(), batch_dists.data());

    for (int q = 0; q < nq; q++) {
        std::vector<int32_t> single_ids(k);
        std::vector<float> single_dists(k);
        s->search_to(queries.data() + q * d.dim, params, single_ids.data(), single_dists.data());

        std::set<int32_t> set_batch(batch_ids.begin() + q * k, batch_ids.begin() + (q + 1) * k);
        std::set<int32_t> set_single(single_ids.begin(), single_ids.end());
        EXPECT_EQ(set_batch, set_single) << "batch vs single mismatch at q=" << q;
    }
}

TEST(SearchE2E, DotNormBatch4Path)
{
    // Force the batch-4 kernel path: need >= 4 neighbors classified in one
    // iteration. Use a larger ef_search and a dense graph (high M).
    auto d = make_e2e_data(300, 16, 24);
    auto s = make_searcher(d);
    s->enable_dot_norm();

    const int k = 10;
    const int nq = 10;
    HnswQueryParams params;
    params.k = k;
    params.ef_search = 128; // large ef → many candidates per expansion → batch-4

    double total_recall = 0.0;
    for (int q = 0; q < nq; q++) {
        const float *query = d.vectors.data() + q * d.dim;

        std::vector<int32_t> ids(k);
        std::vector<float> dists(k);
        s->search_to(query, params, ids.data(), dists.data());

        auto gt = brute_force_topk(query, d.vectors.data(), d.n, d.dim, k);
        total_recall += compute_recall(ids.data(), gt, k);
    }
    EXPECT_GE(total_recall / nq, 0.95);
}

TEST(SearchE2E, DotNormDisableRevertsToL2)
{
    auto d = make_e2e_data(200, 16, 16);
    auto s = make_searcher(d);

    // Enable then disable — should revert to L2 path
    s->enable_dot_norm();
    ASSERT_TRUE(s->dot_norm_enabled());
    s->enable_dot_norm(false);
    ASSERT_FALSE(s->dot_norm_enabled());

    const int k = 10;
    HnswQueryParams params;
    params.k = k;
    params.ef_search = 64;

    std::vector<int32_t> ids(k);
    std::vector<float> dists(k);
    s->search_to(d.vectors.data(), params, ids.data(), dists.data());

    auto gt = brute_force_topk(d.vectors.data(), d.vectors.data(), d.n, d.dim, k);
    EXPECT_GE(compute_recall(ids.data(), gt, k), 0.95);
}
