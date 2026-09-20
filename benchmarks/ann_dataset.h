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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gd_hnsw {

// Common dataset structure for ANN benchmarks.
// Used by both HDF5 and binary loaders.
struct AnnDataset {
    std::vector<float> train;       // base vectors [N * dim]
    std::vector<float> test;        // query vectors [nq * dim]
    std::vector<int32_t> neighbors; // ground truth ids [nq * k]
    std::vector<float> distances;   // ground truth dists [nq * k]
    uint64_t n_train = 0;
    uint64_t n_test = 0;
    uint32_t dim = 0;
    uint32_t k_gt = 0; // ground truth k
};

} // namespace gd_hnsw
