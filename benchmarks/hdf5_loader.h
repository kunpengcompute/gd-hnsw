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
#include <stdexcept>

#include <H5Cpp.h>

#include "ann_dataset.h"

namespace gd_hnsw {

// ============================================================
// HDF5 dataset loader for ANN benchmark format (SIFT1M etc.)
// Expected HDF5 layout:
//   /train    — float32[N, dim]  (base vectors)
//   /test     — float32[nq, dim] (query vectors)
//   /neighbors — int32[nq, k]   (ground truth neighbor ids)
//   /distances — float32[nq, k] (ground truth distances)
// ============================================================

inline AnnDataset load_hdf5_dataset(const std::string &path, bool eval_only = false, bool train_only = false)
{
    AnnDataset ds;

    H5::H5File file(path, H5F_ACC_RDONLY);

    // Helper to read a 2D float dataset
    auto read_float2d = [&](const std::string &name, std::vector<float> &out, uint64_t &nrows, uint32_t &ncols) {
        H5::DataSet dataset = file.openDataSet(name);
        H5::DataSpace space = dataset.getSpace();
        if (space.getSimpleExtentNdims() != 2) {
            throw std::runtime_error(name + " is not a 2D dataset");
        }
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);
        nrows = dims[0];
        ncols = static_cast<uint32_t>(dims[1]);
        out.resize(nrows * ncols);
        dataset.read(out.data(), H5::PredType::NATIVE_FLOAT);
    };

    // Helper to read a 2D int dataset
    auto read_int2d = [&](const std::string &name, std::vector<int32_t> &out, uint64_t &nrows, uint32_t &ncols) {
        H5::DataSet dataset = file.openDataSet(name);
        H5::DataSpace space = dataset.getSpace();
        if (space.getSimpleExtentNdims() != 2) {
            throw std::runtime_error(name + " is not a 2D dataset");
        }
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);
        nrows = dims[0];
        ncols = static_cast<uint32_t>(dims[1]);
        out.resize(nrows * ncols);
        dataset.read(out.data(), H5::PredType::NATIVE_INT32);
    };

    // Helper to read shape only (no data copy)
    auto read_float2d_shape = [&](const std::string &name, uint64_t &nrows, uint32_t &ncols) {
        H5::DataSet dataset = file.openDataSet(name);
        H5::DataSpace space = dataset.getSpace();
        if (space.getSimpleExtentNdims() != 2) {
            throw std::runtime_error(name + " is not a 2D dataset");
        }
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);
        nrows = dims[0];
        ncols = static_cast<uint32_t>(dims[1]);
    };

    // Read train vectors (or shape-only in eval mode to get n_train/dim)
    uint64_t n_train;
    uint32_t dim_train;
    if (eval_only) {
        read_float2d_shape("train", n_train, dim_train);
    } else {
        read_float2d("train", ds.train, n_train, dim_train);
    }
    ds.n_train = n_train;
    ds.dim = dim_train;

    // train_only mode: skip query/gt loading (for standalone index building)
    if (train_only) {
        return ds;
    }

    // Read test vectors
    uint64_t n_test;
    uint32_t dim_test;
    read_float2d("test", ds.test, n_test, dim_test);
    ds.n_test = n_test;
    if (dim_test != ds.dim) {
        throw std::runtime_error("train/test dimension mismatch");
    }

    // Read ground truth neighbors
    uint64_t n_gt;
    uint32_t k_gt;
    read_int2d("neighbors", ds.neighbors, n_gt, k_gt);
    ds.k_gt = k_gt;
    if (n_gt != ds.n_test) {
        throw std::runtime_error("ground truth row count != n_test");
    }

    // Read ground truth distances (optional in some datasets).
    // If missing, fill with zeros to keep shape contracts consistent.
    try {
        uint64_t n_dist;
        uint32_t k_dist;
        read_float2d("distances", ds.distances, n_dist, k_dist);
        if (n_dist != ds.n_test || k_dist != ds.k_gt) {
            throw std::runtime_error("distances shape mismatch");
        }
    } catch (const H5::Exception &) {
        ds.distances.assign(ds.n_test * ds.k_gt, 0.0f);
    }

    return ds;
}

} // namespace gd_hnsw
