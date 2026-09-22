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

#include "gd_hnsw_api.h"

#include <mpi.h>
#include <cstdint>
#include <vector>

namespace gd_hnsw {

// ============================================================
// Benchmark result for one rank
// ============================================================

struct BenchResult {
    std::vector<double> round_qps;          // total QPS per round
    std::vector<double> round_e2e_qps;      // e2e QPS per round
    std::vector<double> query_latencies_us; // per-query latency (last round)
    std::vector<int32_t> out_ids;           // flat result ids [my_nq * K]
    std::vector<float> out_dists;           // flat result distances [my_nq * K]
};

// ============================================================
// Run benchmark rounds: warmup (untimed) + N timed rounds.
// Non-last rounds use batch search for throughput; the last round uses
// per-query search for the latency distribution and final results.
// All MPI_Reduce calls are internal.
// ============================================================

BenchResult run_benchmark_rounds(Context &ctx, const float *my_queries, uint64_t my_nq, uint32_t dim,
                                 const SearchParams &params, int32_t K, int num_rounds, int num_warmup,
                                 uint64_t nq_global, int world_rank, MPI_Comm comm);

} // namespace gd_hnsw
