/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef HAS_ARM_NEON
#include "distance_kernel.h"
#include "ls64_copy.h"
#endif

#include "test_distance_common.hpp"

using namespace gd_hnsw;

// ============================================================
// Scalar self-consistency tests (run on all platforms)
// ============================================================

TEST(L2KernelScalar, SameVectorZeroDistance)
{
    std::vector<float> a(128);
    fill_random_vec(a.data(), 128);
    EXPECT_FLOAT_EQ(scalar::l2_sqr(a.data(), a.data(), 128), 0.0f);
}

TEST(L2KernelScalar, AllZeros)
{
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    EXPECT_FLOAT_EQ(scalar::l2_sqr(a.data(), b.data(), 16), 0.0f);
}

TEST(L2KernelScalar, KnownValues)
{
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {3.0f, 4.0f, 0.0f};
    EXPECT_FLOAT_EQ(scalar::l2_sqr(a, b, 3), 25.0f);
}

TEST(L2KernelScalar, Batch4VsSingle)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float d0, d1, d2, d3;
    scalar::l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, d0, d1, d2, d3);
    EXPECT_FLOAT_EQ(d0, scalar::l2_sqr(q.data(), v0.data(), dim));
    EXPECT_FLOAT_EQ(d1, scalar::l2_sqr(q.data(), v1.data(), dim));
    EXPECT_FLOAT_EQ(d2, scalar::l2_sqr(q.data(), v2.data(), dim));
    EXPECT_FLOAT_EQ(d3, scalar::l2_sqr(q.data(), v3.data(), dim));
}

TEST(L2KernelScalar, Batch3VsSingle)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);

    float d0, d1, d2;
    scalar::l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, d0, d1, d2);
    EXPECT_FLOAT_EQ(d0, scalar::l2_sqr(q.data(), v0.data(), dim));
    EXPECT_FLOAT_EQ(d1, scalar::l2_sqr(q.data(), v1.data(), dim));
    EXPECT_FLOAT_EQ(d2, scalar::l2_sqr(q.data(), v2.data(), dim));
}

TEST(L2KernelScalar, Batch2VsSingle)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);

    float d0, d1;
    scalar::l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, d0, d1);
    EXPECT_FLOAT_EQ(d0, scalar::l2_sqr(q.data(), v0.data(), dim));
    EXPECT_FLOAT_EQ(d1, scalar::l2_sqr(q.data(), v1.data(), dim));
}

TEST(L2KernelScalar, Dim1)
{
    float a[] = {2.0f};
    float b[] = {5.0f};
    EXPECT_FLOAT_EQ(scalar::l2_sqr(a, b, 1), 9.0f);
}

TEST(L2KernelScalar, NaNInputPropagates)
{
    float nan_val[] = {std::nanf(""), std::nanf(""), std::nanf(""), std::nanf("")};
    float normal[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result = scalar::l2_sqr(nan_val, normal, 1);
    EXPECT_TRUE(std::isnan(result));
    // Also test batch functions don't crash
    float d0, d1;
    scalar::l2_sqr_batch_2(normal, normal, nan_val, 4, d0, d1);
    EXPECT_FALSE(std::isnan(d0));
    EXPECT_TRUE(std::isnan(d1));
}

// ============================================================
// NEON vs scalar accuracy tests (ARM64 only)
// ============================================================

#ifdef HAS_ARM_NEON

static constexpr float MAX_ERR = 1e-5f;

TEST(L2KernelNEON, SingleVsScalarDim128)
{
    uint32_t dim = 128;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR * (1.0f + std::fabs(scalar_r)));
}

TEST(L2KernelNEON, SingleVsScalarDim1)
{
    float a[] = {2.5f};
    float b[] = {1.5f};
    float neon_r = l2_sqr(a, b, 1);
    float scalar_r = scalar::l2_sqr(a, b, 1);
    EXPECT_FLOAT_EQ(neon_r, scalar_r);
}

TEST(L2KernelNEON, SingleVsScalarDim3)
{
    uint32_t dim = 3;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR);
}

TEST(L2KernelNEON, SingleVsScalarDim5)
{
    uint32_t dim = 5;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR);
}

TEST(L2KernelNEON, SingleVsScalarDim15)
{
    uint32_t dim = 15;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR);
}

TEST(L2KernelNEON, SingleVsScalarDim7)
{
    uint32_t dim = 7;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR);
}

TEST(L2KernelNEON, SingleVsScalarDim17)
{
    uint32_t dim = 17;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR);
}

TEST(L2KernelNEON, SingleVsScalarDim1000)
{
    uint32_t dim = 1000;
    std::vector<float> a(dim), b(dim);
    fill_random_vec(a.data(), dim);
    fill_random_vec(b.data(), dim);

    float neon_r = l2_sqr(a.data(), b.data(), dim);
    float scalar_r = scalar::l2_sqr(a.data(), b.data(), dim);
    EXPECT_LE(std::fabs(neon_r - scalar_r), MAX_ERR * dim);
}

TEST(L2KernelNEON, SingleSameVector)
{
    uint32_t dim = 128;
    std::vector<float> a(dim);
    fill_random_vec(a.data(), dim);
    EXPECT_FLOAT_EQ(l2_sqr(a.data(), a.data(), dim), 0.0f);
}

TEST(L2KernelNEON, SingleAllZeros)
{
    std::vector<float> z(128, 0.0f);
    EXPECT_FLOAT_EQ(l2_sqr(z.data(), z.data(), 128), 0.0f);
}

TEST(L2KernelNEON, Batch4VsScalarDim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float n0, n1, n2, n3;
    l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, n0, n1, n2, n3);
    float s0, s1, s2, s3;
    scalar::l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, s0, s1, s2, s3);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR);
    EXPECT_LE(std::fabs(n2 - s2), MAX_ERR);
    EXPECT_LE(std::fabs(n3 - s3), MAX_ERR);
}

TEST(L2KernelNEON, Batch4Dim7)
{
    uint32_t dim = 7;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float n0, n1, n2, n3;
    l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, n0, n1, n2, n3);
    float s0, s1, s2, s3;
    scalar::l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, s0, s1, s2, s3);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR);
    EXPECT_LE(std::fabs(n2 - s2), MAX_ERR);
    EXPECT_LE(std::fabs(n3 - s3), MAX_ERR);
}

TEST(L2KernelNEON, Batch4Dim1000)
{
    uint32_t dim = 1000;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim), v3(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);
    fill_random_vec(v3.data(), dim);

    float n0, n1, n2, n3;
    l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, n0, n1, n2, n3);
    float s0, s1, s2, s3;
    scalar::l2_sqr_batch_4(q.data(), v0.data(), v1.data(), v2.data(), v3.data(), dim, s0, s1, s2, s3);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR * dim);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR * dim);
    EXPECT_LE(std::fabs(n2 - s2), MAX_ERR * dim);
    EXPECT_LE(std::fabs(n3 - s3), MAX_ERR * dim);
}

TEST(L2KernelNEON, Batch3VsScalarDim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);

    float n0, n1, n2;
    l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, n0, n1, n2);
    float s0, s1, s2;
    scalar::l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, s0, s1, s2);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR * 2);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR * 2);
    EXPECT_LE(std::fabs(n2 - s2), MAX_ERR * 2);
}

TEST(L2KernelNEON, Batch2VsScalarDim128)
{
    uint32_t dim = 128;
    std::vector<float> q(dim), v0(dim), v1(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);

    float n0, n1;
    l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, n0, n1);
    float s0, s1;
    scalar::l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, s0, s1);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR);
}

// Batch-2: dim=5 triggers i+4 secondary loop + scalar tail
TEST(L2KernelNEON, Batch2Dim5)
{
    uint32_t dim = 5;
    std::vector<float> q(dim), v0(dim), v1(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);

    float n0, n1;
    l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, n0, n1);
    float s0, s1;
    scalar::l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, s0, s1);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR);
}

// Batch-2: dim=12 triggers i+4 secondary loop (no scalar tail)
TEST(L2KernelNEON, Batch2Dim12)
{
    uint32_t dim = 12;
    std::vector<float> q(dim), v0(dim), v1(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);

    float n0, n1;
    l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, n0, n1);
    float s0, s1;
    scalar::l2_sqr_batch_2(q.data(), v0.data(), v1.data(), dim, s0, s1);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR);
}

// Batch-3: dim=5 triggers i+4 secondary loop + scalar tail
TEST(L2KernelNEON, Batch3Dim5)
{
    uint32_t dim = 5;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);

    float n0, n1, n2;
    l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, n0, n1, n2);
    float s0, s1, s2;
    scalar::l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, s0, s1, s2);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR * 2);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR * 2);
    EXPECT_LE(std::fabs(n2 - s2), MAX_ERR * 2);
}

// Batch-3: dim=12 triggers i+4 secondary loop (no scalar tail)
TEST(L2KernelNEON, Batch3Dim12)
{
    uint32_t dim = 12;
    std::vector<float> q(dim), v0(dim), v1(dim), v2(dim);
    fill_random_vec(q.data(), dim);
    fill_random_vec(v0.data(), dim);
    fill_random_vec(v1.data(), dim);
    fill_random_vec(v2.data(), dim);

    float n0, n1, n2;
    l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, n0, n1, n2);
    float s0, s1, s2;
    scalar::l2_sqr_batch_3(q.data(), v0.data(), v1.data(), v2.data(), dim, s0, s1, s2);
    EXPECT_LE(std::fabs(n0 - s0), MAX_ERR * 2);
    EXPECT_LE(std::fabs(n1 - s1), MAX_ERR * 2);
    EXPECT_LE(std::fabs(n2 - s2), MAX_ERR * 2);
}

TEST(L2KernelNEON, NaNInputPropagates)
{
    float nan_val = std::nanf("");
    float normal[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result = l2_sqr(&nan_val, normal, 1);
    EXPECT_TRUE(std::isnan(result));
}

// ============================================================
// ls64_copy tests (ARM64 only)
// ============================================================

TEST(Ls64Copy, FromGdAligned64)
{
    std::vector<char> src(64);
    for (int i = 0; i < 64; i++)
        src[i] = static_cast<char>(i);
    std::vector<char> dst(64, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 64);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 64), 0);
}

TEST(Ls64Copy, FromGd128Bytes)
{
    std::vector<char> src(128);
    for (int i = 0; i < 128; i++)
        src[i] = static_cast<char>(i + 1);
    std::vector<char> dst(128, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 128);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 128), 0);
}

TEST(Ls64Copy, FromGd256Bytes)
{
    std::vector<char> src(256);
    for (int i = 0; i < 256; i++)
        src[i] = static_cast<char>(i * 3);
    std::vector<char> dst(256, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 256);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 256), 0);
}

TEST(Ls64Copy, FromGd1024Bytes)
{
    std::vector<char> src(1024);
    for (int i = 0; i < 1024; i++)
        src[i] = static_cast<char>(i & 0xFF);
    std::vector<char> dst(1024, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 1024);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 1024), 0);
}

TEST(Ls64Copy, FromGdUnaligned65)
{
    std::vector<char> src(65);
    for (int i = 0; i < 65; i++)
        src[i] = static_cast<char>(i);
    std::vector<char> dst(65, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 65);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 65), 0);
}

TEST(Ls64Copy, FromGdUnaligned127)
{
    std::vector<char> src(127);
    for (int i = 0; i < 127; i++)
        src[i] = static_cast<char>(i + 3);
    std::vector<char> dst(127, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 127);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 127), 0);
}

TEST(Ls64Copy, FromGdUnaligned129)
{
    std::vector<char> src(129);
    for (int i = 0; i < 129; i++)
        src[i] = static_cast<char>(i + 5);
    std::vector<char> dst(129, 0);
    ls64_copy_from_gd(dst.data(), src.data(), 129);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 129), 0);
}

TEST(Ls64Copy, FromGdZeroBytes)
{
    char c = 'x';
    char d = 0;
    ls64_copy_from_gd(&d, &c, 0);
    EXPECT_EQ(d, 0); // no-op
}

TEST(Ls64Copy, ToGdAligned64)
{
    std::vector<char> src(64);
    for (int i = 0; i < 64; i++)
        src[i] = static_cast<char>(i);
    std::vector<char> dst(64, 0);
    ls64_copy_to_gd(dst.data(), src.data(), 64);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 64), 0);
}

TEST(Ls64Copy, ToGdUnaligned65)
{
    std::vector<char> src(65);
    for (int i = 0; i < 65; i++)
        src[i] = static_cast<char>(i + 7);
    std::vector<char> dst(65, 0);
    ls64_copy_to_gd(dst.data(), src.data(), 65);
    EXPECT_EQ(std::memcmp(dst.data(), src.data(), 65), 0);
}

TEST(Ls64Copy, Roundtrip)
{
    std::vector<char> orig(256);
    for (int i = 0; i < 256; i++)
        orig[i] = static_cast<char>(i);
    std::vector<char> mid(256, 0);
    std::vector<char> back(256, 0);

    ls64_copy_to_gd(mid.data(), orig.data(), 256);
    ls64_copy_from_gd(back.data(), mid.data(), 256);
    EXPECT_EQ(std::memcmp(orig.data(), back.data(), 256), 0);
}

#endif // HAS_ARM_NEON
