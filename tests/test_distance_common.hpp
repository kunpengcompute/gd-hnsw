/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace gd_hnsw {
namespace scalar {

// Pure C++ scalar L2 squared distance (reference for NEON validation)
inline float l2_sqr(const float *a, const float *b, uint32_t d)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline void l2_sqr_batch_4(const float *q, const float *v0, const float *v1, const float *v2, const float *v3,
                           uint32_t d, float &dis0, float &dis1, float &dis2, float &dis3)
{
    dis0 = 0.0f;
    dis1 = 0.0f;
    dis2 = 0.0f;
    dis3 = 0.0f;
    for (uint32_t i = 0; i < d; i++) {
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

inline void l2_sqr_batch_3(const float *q, const float *v0, const float *v1, const float *v2, uint32_t d, float &dis0,
                           float &dis1, float &dis2)
{
    dis0 = 0.0f;
    dis1 = 0.0f;
    dis2 = 0.0f;
    for (uint32_t i = 0; i < d; i++) {
        float qi = q[i];
        float t0 = qi - v0[i];
        dis0 += t0 * t0;
        float t1 = qi - v1[i];
        dis1 += t1 * t1;
        float t2 = qi - v2[i];
        dis2 += t2 * t2;
    }
}

inline void l2_sqr_batch_2(const float *q, const float *v0, const float *v1, uint32_t d, float &dis0, float &dis1)
{
    dis0 = 0.0f;
    dis1 = 0.0f;
    for (uint32_t i = 0; i < d; i++) {
        float qi = q[i];
        float t0 = qi - v0[i];
        dis0 += t0 * t0;
        float t1 = qi - v1[i];
        dis1 += t1 * t1;
    }
}

inline float vec_norm_sq(const float *a, uint32_t d)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < d; i++)
        sum += a[i] * a[i];
    return sum;
}

inline float dot_norm_dist(const float *q, const float *v, uint32_t d, float q_norm_sq, float v_norm_sq)
{
    float dot_val = 0.0f;
    for (uint32_t i = 0; i < d; i++)
        dot_val += q[i] * v[i];
    float result = q_norm_sq + v_norm_sq - 2.0f * dot_val;
    return result > 0.0f ? result : 0.0f;
}

inline void dot_norm_dist_batch_4(const float *q, const float *v0, const float *v1, const float *v2, const float *v3,
                                  uint32_t d, float q_norm_sq, float vn0, float vn1, float vn2, float vn3, float &dis0,
                                  float &dis1, float &dis2, float &dis3)
{
    float dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
    for (uint32_t i = 0; i < d; i++) {
        float qi = q[i];
        dot0 += qi * v0[i];
        dot1 += qi * v1[i];
        dot2 += qi * v2[i];
        dot3 += qi * v3[i];
    }
    dis0 = q_norm_sq + vn0 - 2.0f * dot0;
    if (dis0 < 0)
        dis0 = 0;
    dis1 = q_norm_sq + vn1 - 2.0f * dot1;
    if (dis1 < 0)
        dis1 = 0;
    dis2 = q_norm_sq + vn2 - 2.0f * dot2;
    if (dis2 < 0)
        dis2 = 0;
    dis3 = q_norm_sq + vn3 - 2.0f * dot3;
    if (dis3 < 0)
        dis3 = 0;
}

inline void dot_norm_dist_batch_3(const float *q, const float *v0, const float *v1, const float *v2, uint32_t d,
                                  float q_norm_sq, float vn0, float vn1, float vn2, float &dis0, float &dis1,
                                  float &dis2)
{
    float dot0 = 0, dot1 = 0, dot2 = 0;
    for (uint32_t i = 0; i < d; i++) {
        float qi = q[i];
        dot0 += qi * v0[i];
        dot1 += qi * v1[i];
        dot2 += qi * v2[i];
    }
    dis0 = q_norm_sq + vn0 - 2.0f * dot0;
    if (dis0 < 0)
        dis0 = 0;
    dis1 = q_norm_sq + vn1 - 2.0f * dot1;
    if (dis1 < 0)
        dis1 = 0;
    dis2 = q_norm_sq + vn2 - 2.0f * dot2;
    if (dis2 < 0)
        dis2 = 0;
}

inline void dot_norm_dist_batch_2(const float *q, const float *v0, const float *v1, uint32_t d, float q_norm_sq,
                                  float vn0, float vn1, float &dis0, float &dis1)
{
    float dot0 = 0, dot1 = 0;
    for (uint32_t i = 0; i < d; i++) {
        float qi = q[i];
        dot0 += qi * v0[i];
        dot1 += qi * v1[i];
    }
    dis0 = q_norm_sq + vn0 - 2.0f * dot0;
    if (dis0 < 0)
        dis0 = 0;
    dis1 = q_norm_sq + vn1 - 2.0f * dot1;
    if (dis1 < 0)
        dis1 = 0;
}

} // namespace scalar
} // namespace gd_hnsw

// Generate a random float vector of length dim
inline void fill_random_vec(float *v, uint32_t dim, float scale = 1.0f)
{
    for (uint32_t i = 0; i < dim; i++) {
        v[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * scale;
    }
}
