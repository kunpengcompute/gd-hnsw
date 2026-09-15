/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

// ============================================================
// Dot+Norm distance kernels for L2 approximation.
//
// L2(a,b)^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
//
// When vector norms are precomputed, this reduces the inner loop
// to a dot product (one FMA per element vs sub+FMA for direct L2).
// Caller provides: query_norm_sq = ||q||^2, vec_norms[] = ||v_i||^2.
//
// Priority: NEON > scalar (same as l2_sqr kernels).
// ============================================================

#include <cstdint>

#include <arm_neon.h>

namespace gd_hnsw {

// ============================================================
// Compute squared norm of a vector (used to precompute norms)
// ============================================================

inline float vec_norm_sq(const float *a, uint32_t d)
{
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        s0 = vfmaq_f32(s0, a0, a0);
        s1 = vfmaq_f32(s1, a1, a1);
        s2 = vfmaq_f32(s2, a2, a2);
        s3 = vfmaq_f32(s3, a3, a3);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        s0 = vfmaq_f32(s0, a0, a0);
    }
    float result = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; i++) {
        result += a[i] * a[i];
    }
    return result;
}

// ============================================================
// Single dot+norm distance:
// Returns q_norm_sq + v_norm_sq - 2*dot(q, v)
// ============================================================

inline float dot_norm_dist(const float *q, const float *v, uint32_t d, float q_norm_sq, float v_norm_sq)
{
    float32x4_t d0 = vdupq_n_f32(0.0f);
    float32x4_t d1 = vdupq_n_f32(0.0f);
    float32x4_t d2 = vdupq_n_f32(0.0f);
    float32x4_t d3 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        d0 = vfmaq_f32(d0, vld1q_f32(q + i), vld1q_f32(v + i));
        d1 = vfmaq_f32(d1, vld1q_f32(q + i + 4), vld1q_f32(v + i + 4));
        d2 = vfmaq_f32(d2, vld1q_f32(q + i + 8), vld1q_f32(v + i + 8));
        d3 = vfmaq_f32(d3, vld1q_f32(q + i + 12), vld1q_f32(v + i + 12));
    }
    for (; i + 4 <= d; i += 4) {
        d0 = vfmaq_f32(d0, vld1q_f32(q + i), vld1q_f32(v + i));
    }
    float dot_val = vaddvq_f32(vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3)));
    for (; i < d; i++) {
        dot_val += q[i] * v[i];
    }
    float result_neon = q_norm_sq + v_norm_sq - 2.0f * dot_val;
    return result_neon > 0.0f ? result_neon : 0.0f;
}

// ============================================================
// Batch-4 dot+norm distance: compute distances from one query
// to 4 target vectors using precomputed norms.
// NEON: dual accumulator chains per target to hide FMA latency.
// ============================================================

inline void dot_norm_dist_batch_4(const float *q, const float *v0, const float *v1, const float *v2, const float *v3,
                                  uint32_t d, float q_norm_sq, float vn0, float vn1, float vn2, float vn3, float &dis0,
                                  float &dis1, float &dis2, float &dis3)
{
    // Dual accumulator chains: a/b interleave to hide FMA latency
    float32x4_t s0a = vdupq_n_f32(0.0f), s0b = vdupq_n_f32(0.0f);
    float32x4_t s1a = vdupq_n_f32(0.0f), s1b = vdupq_n_f32(0.0f);
    float32x4_t s2a = vdupq_n_f32(0.0f), s2b = vdupq_n_f32(0.0f);
    float32x4_t s3a = vdupq_n_f32(0.0f), s3b = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qi0 = vld1q_f32(q + i);
        float32x4_t qi1 = vld1q_f32(q + i + 4);

        s0a = vfmaq_f32(s0a, qi0, vld1q_f32(v0 + i));
        s0b = vfmaq_f32(s0b, qi1, vld1q_f32(v0 + i + 4));

        s1a = vfmaq_f32(s1a, qi0, vld1q_f32(v1 + i));
        s1b = vfmaq_f32(s1b, qi1, vld1q_f32(v1 + i + 4));

        s2a = vfmaq_f32(s2a, qi0, vld1q_f32(v2 + i));
        s2b = vfmaq_f32(s2b, qi1, vld1q_f32(v2 + i + 4));

        s3a = vfmaq_f32(s3a, qi0, vld1q_f32(v3 + i));
        s3b = vfmaq_f32(s3b, qi1, vld1q_f32(v3 + i + 4));
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qi = vld1q_f32(q + i);
        s0a = vfmaq_f32(s0a, qi, vld1q_f32(v0 + i));
        s1a = vfmaq_f32(s1a, qi, vld1q_f32(v1 + i));
        s2a = vfmaq_f32(s2a, qi, vld1q_f32(v2 + i));
        s3a = vfmaq_f32(s3a, qi, vld1q_f32(v3 + i));
    }
    // Merge and compute final distances
    float dot0 = vaddvq_f32(vaddq_f32(s0a, s0b));
    float dot1 = vaddvq_f32(vaddq_f32(s1a, s1b));
    float dot2 = vaddvq_f32(vaddq_f32(s2a, s2b));
    float dot3 = vaddvq_f32(vaddq_f32(s3a, s3b));
    // Scalar tail
    for (; i < d; i++) {
        float qi = q[i];
        dot0 += qi * v0[i];
        dot1 += qi * v1[i];
        dot2 += qi * v2[i];
        dot3 += qi * v3[i];
    }
    dis0 = q_norm_sq + vn0 - 2.0f * dot0;
    dis1 = q_norm_sq + vn1 - 2.0f * dot1;
    dis2 = q_norm_sq + vn2 - 2.0f * dot2;
    dis3 = q_norm_sq + vn3 - 2.0f * dot3;
    // Clamp negative distances from floating-point cancellation
    if (dis0 < 0.0f)
        dis0 = 0.0f;
    if (dis1 < 0.0f)
        dis1 = 0.0f;
    if (dis2 < 0.0f)
        dis2 = 0.0f;
    if (dis3 < 0.0f)
        dis3 = 0.0f;
}

// ============================================================
// Batch-2 dot+norm distance
// ============================================================

inline void dot_norm_dist_batch_2(const float *q, const float *v0, const float *v1, uint32_t d, float q_norm_sq,
                                  float vn0, float vn1, float &dis0, float &dis1)
{
    float32x4_t s0a = vdupq_n_f32(0.0f), s0b = vdupq_n_f32(0.0f);
    float32x4_t s1a = vdupq_n_f32(0.0f), s1b = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qi0 = vld1q_f32(q + i);
        float32x4_t qi1 = vld1q_f32(q + i + 4);
        s0a = vfmaq_f32(s0a, qi0, vld1q_f32(v0 + i));
        s0b = vfmaq_f32(s0b, qi1, vld1q_f32(v0 + i + 4));
        s1a = vfmaq_f32(s1a, qi0, vld1q_f32(v1 + i));
        s1b = vfmaq_f32(s1b, qi1, vld1q_f32(v1 + i + 4));
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qi = vld1q_f32(q + i);
        s0a = vfmaq_f32(s0a, qi, vld1q_f32(v0 + i));
        s1a = vfmaq_f32(s1a, qi, vld1q_f32(v1 + i));
    }
    float dot0 = vaddvq_f32(vaddq_f32(s0a, s0b));
    float dot1 = vaddvq_f32(vaddq_f32(s1a, s1b));
    for (; i < d; i++) {
        float qi = q[i];
        dot0 += qi * v0[i];
        dot1 += qi * v1[i];
    }
    dis0 = q_norm_sq + vn0 - 2.0f * dot0;
    dis1 = q_norm_sq + vn1 - 2.0f * dot1;
    if (dis0 < 0.0f)
        dis0 = 0.0f;
    if (dis1 < 0.0f)
        dis1 = 0.0f;
}

// ============================================================
// Batch-3 dot+norm distance
// ============================================================

inline void dot_norm_dist_batch_3(const float *q, const float *v0, const float *v1, const float *v2, uint32_t d,
                                  float q_norm_sq, float vn0, float vn1, float vn2, float &dis0, float &dis1,
                                  float &dis2)
{
    float32x4_t s0a = vdupq_n_f32(0.0f), s0b = vdupq_n_f32(0.0f);
    float32x4_t s1a = vdupq_n_f32(0.0f), s1b = vdupq_n_f32(0.0f);
    float32x4_t s2a = vdupq_n_f32(0.0f), s2b = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t qi0 = vld1q_f32(q + i);
        float32x4_t qi1 = vld1q_f32(q + i + 4);
        s0a = vfmaq_f32(s0a, qi0, vld1q_f32(v0 + i));
        s0b = vfmaq_f32(s0b, qi1, vld1q_f32(v0 + i + 4));
        s1a = vfmaq_f32(s1a, qi0, vld1q_f32(v1 + i));
        s1b = vfmaq_f32(s1b, qi1, vld1q_f32(v1 + i + 4));
        s2a = vfmaq_f32(s2a, qi0, vld1q_f32(v2 + i));
        s2b = vfmaq_f32(s2b, qi1, vld1q_f32(v2 + i + 4));
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t qi = vld1q_f32(q + i);
        s0a = vfmaq_f32(s0a, qi, vld1q_f32(v0 + i));
        s1a = vfmaq_f32(s1a, qi, vld1q_f32(v1 + i));
        s2a = vfmaq_f32(s2a, qi, vld1q_f32(v2 + i));
    }
    float dot0 = vaddvq_f32(vaddq_f32(s0a, s0b));
    float dot1 = vaddvq_f32(vaddq_f32(s1a, s1b));
    float dot2 = vaddvq_f32(vaddq_f32(s2a, s2b));
    for (; i < d; i++) {
        float qi = q[i];
        dot0 += qi * v0[i];
        dot1 += qi * v1[i];
        dot2 += qi * v2[i];
    }
    dis0 = q_norm_sq + vn0 - 2.0f * dot0;
    dis1 = q_norm_sq + vn1 - 2.0f * dot1;
    dis2 = q_norm_sq + vn2 - 2.0f * dot2;
    if (dis0 < 0.0f)
        dis0 = 0.0f;
    if (dis1 < 0.0f)
        dis1 = 0.0f;
    if (dis2 < 0.0f)
        dis2 = 0.0f;
}

} // namespace gd_hnsw
