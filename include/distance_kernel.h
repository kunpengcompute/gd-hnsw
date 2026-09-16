/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include <cstdint>

#include <arm_neon.h>

namespace gd_hnsw {

// ============================================================
// Single-vector L2 squared distance
// Priority: NEON > scalar
// ============================================================

inline float l2_sqr(const float *a, const float *b, uint32_t d)
{
    // NEON path: 4 accumulators x 4 lanes = 16 floats/iter
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t da0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t da1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        float32x4_t da2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        float32x4_t da3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        sum0 = vfmaq_f32(sum0, da0, da0);
        sum1 = vfmaq_f32(sum1, da1, da1);
        sum2 = vfmaq_f32(sum2, da2, da2);
        sum3 = vfmaq_f32(sum3, da3, da3);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t da = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        sum0 = vfmaq_f32(sum0, da, da);
    }
    float result = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (; i < d; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

// ============================================================
// Batch-4 L2 squared distance: compute distances from one query
// to 4 target vectors simultaneously.
// Priority: NEON > scalar
// NEON: 2x unrolled with dual accumulator chains per target to hide
//       FMA latency. Uses 16 NEON accumulators (4 targets x 2 chains).
// ============================================================

inline void l2_sqr_batch_4(const float *q, const float *v0, const float *v1, const float *v2, const float *v3,
                           uint32_t d, float &dis0, float &dis1, float &dis2, float &dis3)
{
    // Dual accumulator chains: a/b interleave to hide FMA latency
    float32x4_t s0a = vdupq_n_f32(0.0f), s0b = vdupq_n_f32(0.0f);
    float32x4_t s1a = vdupq_n_f32(0.0f), s1b = vdupq_n_f32(0.0f);
    float32x4_t s2a = vdupq_n_f32(0.0f), s2b = vdupq_n_f32(0.0f);
    float32x4_t s3a = vdupq_n_f32(0.0f), s3b = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    // Main loop: 8 floats per iteration (2x unrolled)
    for (; i + 8 <= d; i += 8) {
        float32x4_t qi0 = vld1q_f32(q + i);
        float32x4_t qi1 = vld1q_f32(q + i + 4);

        float32x4_t da0 = vsubq_f32(qi0, vld1q_f32(v0 + i));
        float32x4_t db0 = vsubq_f32(qi1, vld1q_f32(v0 + i + 4));
        s0a = vfmaq_f32(s0a, da0, da0);
        s0b = vfmaq_f32(s0b, db0, db0);

        float32x4_t da1 = vsubq_f32(qi0, vld1q_f32(v1 + i));
        float32x4_t db1 = vsubq_f32(qi1, vld1q_f32(v1 + i + 4));
        s1a = vfmaq_f32(s1a, da1, da1);
        s1b = vfmaq_f32(s1b, db1, db1);

        float32x4_t da2 = vsubq_f32(qi0, vld1q_f32(v2 + i));
        float32x4_t db2 = vsubq_f32(qi1, vld1q_f32(v2 + i + 4));
        s2a = vfmaq_f32(s2a, da2, da2);
        s2b = vfmaq_f32(s2b, db2, db2);

        float32x4_t da3 = vsubq_f32(qi0, vld1q_f32(v3 + i));
        float32x4_t db3 = vsubq_f32(qi1, vld1q_f32(v3 + i + 4));
        s3a = vfmaq_f32(s3a, da3, da3);
        s3b = vfmaq_f32(s3b, db3, db3);
    }
    // Tail: 4 floats (handles dims not multiple of 8, e.g. dim=12)
    for (; i + 4 <= d; i += 4) {
        float32x4_t qi = vld1q_f32(q + i);
        float32x4_t da0 = vsubq_f32(qi, vld1q_f32(v0 + i));
        float32x4_t da1 = vsubq_f32(qi, vld1q_f32(v1 + i));
        float32x4_t da2 = vsubq_f32(qi, vld1q_f32(v2 + i));
        float32x4_t da3 = vsubq_f32(qi, vld1q_f32(v3 + i));
        s0a = vfmaq_f32(s0a, da0, da0);
        s1a = vfmaq_f32(s1a, da1, da1);
        s2a = vfmaq_f32(s2a, da2, da2);
        s3a = vfmaq_f32(s3a, da3, da3);
    }
    // Merge dual chains and reduce
    dis0 = vaddvq_f32(vaddq_f32(s0a, s0b));
    dis1 = vaddvq_f32(vaddq_f32(s1a, s1b));
    dis2 = vaddvq_f32(vaddq_f32(s2a, s2b));
    dis3 = vaddvq_f32(vaddq_f32(s3a, s3b));
    // Scalar tail for non-multiple-of-4 dimensions
    for (; i < d; i++) {
        float qi = q[i];
        float t0 = qi - v0[i];
        dis0 += t0 * t0;
        float t1 = qi - v1[i];
        dis1 += t1 * t1;
        float t2 = qi - v2[i];
        dis2 += t2 * t2;
        float t3 = qi - v3[i];
        dis3 += t3 * t3;
    }
}

// ============================================================
// Batch-2 L2 squared distance: compute distances from one query
// to 2 target vectors simultaneously (query loaded once).
// Priority: NEON > scalar
// ============================================================

inline void l2_sqr_batch_2(const float *q, const float *v0, const float *v1, uint32_t d, float &dis0, float &dis1)
{
    float32x4_t s0a = vdupq_n_f32(0.0f), s0b = vdupq_n_f32(0.0f);
    float32x4_t s1a = vdupq_n_f32(0.0f), s1b = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qi0 = vld1q_f32(q + i);
        float32x4_t qi1 = vld1q_f32(q + i + 4);

        float32x4_t da0 = vsubq_f32(qi0, vld1q_f32(v0 + i));
        float32x4_t db0 = vsubq_f32(qi1, vld1q_f32(v0 + i + 4));
        s0a = vfmaq_f32(s0a, da0, da0);
        s0b = vfmaq_f32(s0b, db0, db0);

        float32x4_t da1 = vsubq_f32(qi0, vld1q_f32(v1 + i));
        float32x4_t db1 = vsubq_f32(qi1, vld1q_f32(v1 + i + 4));
        s1a = vfmaq_f32(s1a, da1, da1);
        s1b = vfmaq_f32(s1b, db1, db1);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qi = vld1q_f32(q + i);
        float32x4_t da0 = vsubq_f32(qi, vld1q_f32(v0 + i));
        float32x4_t da1 = vsubq_f32(qi, vld1q_f32(v1 + i));
        s0a = vfmaq_f32(s0a, da0, da0);
        s1a = vfmaq_f32(s1a, da1, da1);
    }
    dis0 = vaddvq_f32(vaddq_f32(s0a, s0b));
    dis1 = vaddvq_f32(vaddq_f32(s1a, s1b));
    for (; i < d; i++) {
        float qi = q[i];
        float t0 = qi - v0[i];
        dis0 += t0 * t0;
        float t1 = qi - v1[i];
        dis1 += t1 * t1;
    }
}

// ============================================================
// Batch-3 L2 squared distance: compute distances from one query
// to 3 target vectors simultaneously (query loaded once).
// Priority: NEON > scalar
// ============================================================

inline void l2_sqr_batch_3(const float *q, const float *v0, const float *v1, const float *v2, uint32_t d, float &dis0,
                           float &dis1, float &dis2)
{
    float32x4_t s0a = vdupq_n_f32(0.0f), s0b = vdupq_n_f32(0.0f);
    float32x4_t s1a = vdupq_n_f32(0.0f), s1b = vdupq_n_f32(0.0f);
    float32x4_t s2a = vdupq_n_f32(0.0f), s2b = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qi0 = vld1q_f32(q + i);
        float32x4_t qi1 = vld1q_f32(q + i + 4);

        float32x4_t da0 = vsubq_f32(qi0, vld1q_f32(v0 + i));
        float32x4_t db0 = vsubq_f32(qi1, vld1q_f32(v0 + i + 4));
        s0a = vfmaq_f32(s0a, da0, da0);
        s0b = vfmaq_f32(s0b, db0, db0);

        float32x4_t da1 = vsubq_f32(qi0, vld1q_f32(v1 + i));
        float32x4_t db1 = vsubq_f32(qi1, vld1q_f32(v1 + i + 4));
        s1a = vfmaq_f32(s1a, da1, da1);
        s1b = vfmaq_f32(s1b, db1, db1);

        float32x4_t da2 = vsubq_f32(qi0, vld1q_f32(v2 + i));
        float32x4_t db2 = vsubq_f32(qi1, vld1q_f32(v2 + i + 4));
        s2a = vfmaq_f32(s2a, da2, da2);
        s2b = vfmaq_f32(s2b, db2, db2);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qi = vld1q_f32(q + i);
        float32x4_t da0 = vsubq_f32(qi, vld1q_f32(v0 + i));
        float32x4_t da1 = vsubq_f32(qi, vld1q_f32(v1 + i));
        float32x4_t da2 = vsubq_f32(qi, vld1q_f32(v2 + i));
        s0a = vfmaq_f32(s0a, da0, da0);
        s1a = vfmaq_f32(s1a, da1, da1);
        s2a = vfmaq_f32(s2a, da2, da2);
    }
    dis0 = vaddvq_f32(vaddq_f32(s0a, s0b));
    dis1 = vaddvq_f32(vaddq_f32(s1a, s1b));
    dis2 = vaddvq_f32(vaddq_f32(s2a, s2b));
    for (; i < d; i++) {
        float qi = q[i];
        float t0 = qi - v0[i];
        dis0 += t0 * t0;
        float t1 = qi - v1[i];
        dis1 += t1 * t1;
        float t2 = qi - v2[i];
        dis2 += t2 * t2;
    }
}

} // namespace gd_hnsw
