/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef HAS_ARM_NEON
#include "distance_kernel_dot_norm.h"
#endif

#include "test_distance_common.hpp"

using namespace gd_hnsw;

// ============================================================
// Scalar self-consistency tests (run on all platforms)
// ============================================================

TEST(DotNormScalar, VecNormSqZeroVector)
{
    std::vector<float> z(128, 0.0f);
    EXPECT_FLOAT_EQ(scalar::vec_norm_sq(z.data(), 128), 0.0f);
}

TEST(DotNormScalar, VecNormSqKnownValue)
{
    float a[] = {3.0f, 4.0f, 0.0f};
    EXPECT_FLOAT_EQ(scalar::vec_norm_sq(a, 3), 25.0f);
}

TEST(DotNormScalar, DotNormVsL2Sqr)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v.data(), dim);

    float q_norm = scalar::vec_norm_sq(q.data(), dim);
    float v_norm = scalar::vec_norm_sq(v.data(), dim);
    float dn = scalar::dot_norm_dist(q.data(), v.data(), dim, q_norm, v_norm);
    float l2 = scalar::l2_sqr(q.data(), v.data(), dim);

    EXPECT_LE(std::fabs(dn - l2), 1e-4f * (1.0f + dim));
}

TEST(DotNormScalar, DotNormSameVectorZero)
{
    uint32_t dim = 64;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);
    float norm = scalar::vec_norm_sq(a.data(), dim);
    float d = scalar::dot_norm_dist(a.data(), a.data(), dim, norm, norm);
    EXPECT_NEAR(d, 0.0f, 1e-4f);
}

TEST(DotNormScalar, Batch4VsSingle)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);
    float n2 = scalar::vec_norm_sq(v2.data(), dim);
    float n3 = scalar::vec_norm_sq(v3.data(), dim);

    float d0, d1, d2, d3;
    scalar::dot_norm_dist_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, qn, n0, n1, n2, n3, d0, d1,
                                  d2, d3);
    EXPECT_FLOAT_EQ(d0, scalar::dot_norm_dist(q.data(), v0.data(), dim, qn, n0));
    EXPECT_FLOAT_EQ(d1, scalar::dot_norm_dist(q.data(), v1.data(), dim, qn, n1));
    EXPECT_FLOAT_EQ(d2, scalar::dot_norm_dist(q.data(), v2.data(), dim, qn, n2));
    EXPECT_FLOAT_EQ(d3, scalar::dot_norm_dist(q.data(), v3.data(), dim, qn, n3));
}

// ============================================================
// NEON vs scalar accuracy tests (ARM64 only)
// ============================================================

#ifdef HAS_ARM_NEON

static constexpr float MAX_ERR = 1e-4f;

TEST(DotNormNEON, VecNormVsScalarDim128)
{
    uint32_t dim = 128;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);

    float nr = vec_norm_sq(a.data(), dim);
    float sr = scalar::vec_norm_sq(a.data(), dim);
    EXPECT_LE(std::fabs(nr - sr), MAX_ERR);
}

TEST(DotNormNEON, VecNormVsScalarDim1)
{
    float a[] = {5.0f};
    EXPECT_FLOAT_EQ(vec_norm_sq(a, 1), scalar::vec_norm_sq(a, 1));
}

TEST(DotNormNEON, VecNormVsScalarDim7)
{
    uint32_t dim = 7;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);

    float nr = vec_norm_sq(a.data(), dim);
    float sr = scalar::vec_norm_sq(a.data(), dim);
    EXPECT_LE(std::fabs(nr - sr), MAX_ERR);
}

TEST(DotNormNEON, VecNormVsScalarDim5)
{
    uint32_t dim = 5;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);

    float nr = vec_norm_sq(a.data(), dim);
    float sr = scalar::vec_norm_sq(a.data(), dim);
    EXPECT_LE(std::fabs(nr - sr), MAX_ERR);
}

TEST(DotNormNEON, VecNormVsScalarDim15)
{
    uint32_t dim = 15;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);

    float nr = vec_norm_sq(a.data(), dim);
    float sr = scalar::vec_norm_sq(a.data(), dim);
    EXPECT_LE(std::fabs(nr - sr), MAX_ERR);
}

TEST(DotNormNEON, VecNormVsScalarDim1000)
{
    uint32_t dim = 1000;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);

    float nr = vec_norm_sq(a.data(), dim);
    float sr = scalar::vec_norm_sq(a.data(), dim);
    EXPECT_LE(std::fabs(nr - sr), MAX_ERR * dim);
}

TEST(DotNormNEON, VecNormZeroVector)
{
    std::vector<float> z(128, 0.0f);
    EXPECT_FLOAT_EQ(vec_norm_sq(z.data(), 128), 0.0f);
}

TEST(DotNormNEON, SingleVsScalarDim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float vn = scalar::vec_norm_sq(v.data(), dim);
    float nr = dot_norm_dist(q.data(), v.data(), dim, qn, vn);
    float l2 = scalar::l2_sqr(q.data(), v.data(), dim);

    EXPECT_LE(std::fabs(nr - l2), MAX_ERR * (1.0f + dim));
}

TEST(DotNormNEON, SingleSameVectorNearZero)
{
    uint32_t dim = 64;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);

    float norm = scalar::vec_norm_sq(a.data(), dim);
    float d = dot_norm_dist(a.data(), a.data(), dim, norm, norm);
    EXPECT_NEAR(d, 0.0f, MAX_ERR * (1.0f + dim));
}

TEST(DotNormNEON, SingleClampNegative)
{
    // Use vectors that would produce slight negative from cancellation
    uint32_t dim = 4;
    float q[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v[] = {1.0f + 1e-7f, 0.0f, 0.0f, 0.0f};
    float qn = 1.0f;
    float vn = 1.0f + 2e-7f; // ≈ ||v||^2

    float d = dot_norm_dist(q, v, dim, qn, vn);
    EXPECT_GE(d, 0.0f);
}

TEST(DotNormNEON, Batch4VsScalarL2Dim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);
    float n2 = scalar::vec_norm_sq(v2.data(), dim);
    float n3 = scalar::vec_norm_sq(v3.data(), dim);

    float d0, d1, d2, d3;
    dot_norm_dist_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, qn, n0, n1, n2, n3, d0, d1, d2,
                          d3);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);
    float l2 = scalar::l2_sqr(q.data(), v2.data(), dim);
    float l3 = scalar::l2_sqr(q.data(), v3.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d2 - l2), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d3 - l3), MAX_ERR * (1.0f + dim));
}

TEST(DotNormNEON, Batch3VsScalarL2Dim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);
    float n2 = scalar::vec_norm_sq(v2.data(), dim);

    float d0, d1, d2;
    dot_norm_dist_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, qn, n0, n1, n2, d0, d1, d2);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);
    float l2 = scalar::l2_sqr(q.data(), v2.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d2 - l2), MAX_ERR * (1.0f + dim));
}

TEST(DotNormNEON, Batch2VsScalarL2Dim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);

    float d0, d1;
    dot_norm_dist_batch_2(q.data(), v0.data(), v1.data(), dim, qn, n0, n1, d0, d1);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
}

// Batch-4: dim=5 triggers i+4 secondary loop + scalar tail
TEST(DotNormNEON, Batch4Dim5)
{
    uint32_t dim = 5;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);
    float n2 = scalar::vec_norm_sq(v2.data(), dim);
    float n3 = scalar::vec_norm_sq(v3.data(), dim);

    float d0, d1, d2, d3;
    dot_norm_dist_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, qn, n0, n1, n2, n3, d0, d1, d2,
                          d3);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);
    float l2 = scalar::l2_sqr(q.data(), v2.data(), dim);
    float l3 = scalar::l2_sqr(q.data(), v3.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d2 - l2), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d3 - l3), MAX_ERR * (1.0f + dim));
}

// Batch-4: dim=20 triggers i+4 secondary loop (no scalar tail)
TEST(DotNormNEON, Batch4Dim20)
{
    uint32_t dim = 20;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);
    float n2 = scalar::vec_norm_sq(v2.data(), dim);
    float n3 = scalar::vec_norm_sq(v3.data(), dim);

    float d0, d1, d2, d3;
    dot_norm_dist_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, qn, n0, n1, n2, n3, d0, d1, d2,
                          d3);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);
    float l2 = scalar::l2_sqr(q.data(), v2.data(), dim);
    float l3 = scalar::l2_sqr(q.data(), v3.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d2 - l2), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d3 - l3), MAX_ERR * (1.0f + dim));
}

// Batch-3: dim=5 triggers i+4 secondary loop + scalar tail
TEST(DotNormNEON, Batch3Dim5)
{
    uint32_t dim = 5;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);
    float n2 = scalar::vec_norm_sq(v2.data(), dim);

    float d0, d1, d2;
    dot_norm_dist_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, qn, n0, n1, n2, d0, d1, d2);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);
    float l2 = scalar::l2_sqr(q.data(), v2.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d2 - l2), MAX_ERR * (1.0f + dim));
}

// Batch-2: dim=5 triggers i+4 secondary loop + scalar tail
TEST(DotNormNEON, Batch2Dim5)
{
    uint32_t dim = 5;
    std::vector<float> q(dim), v0(dim), v1(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);

    float qn = scalar::vec_norm_sq(q.data(), dim);
    float n0 = scalar::vec_norm_sq(v0.data(), dim);
    float n1 = scalar::vec_norm_sq(v1.data(), dim);

    float d0, d1;
    dot_norm_dist_batch_2(q.data(), v0.data(), v1.data(), dim, qn, n0, n1, d0, d1);

    float l0 = scalar::l2_sqr(q.data(), v0.data(), dim);
    float l1 = scalar::l2_sqr(q.data(), v1.data(), dim);

    EXPECT_LE(std::fabs(d0 - l0), MAX_ERR * (1.0f + dim));
    EXPECT_LE(std::fabs(d1 - l1), MAX_ERR * (1.0f + dim));
}

#endif // HAS_ARM_NEON
