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

#include <mpi.h>
#include <cstdint>
#include <vector>

namespace gd_hnsw {

// ============================================================
// Recall@K computation
// ============================================================

struct RecallStats {
    uint64_t hits = 0;
    uint64_t total = 0;
};

// result_ids: flat [nq * k], gt: flat [nq * gt_k] in the SAME query order.
// new_to_old_idmap: if set, remaps result ids (internal gids) to original ids
// before comparing against ground truth.
RecallStats compute_recall_stats(const int32_t *result_ids, uint64_t nq, const int32_t *gt, uint32_t gt_k, int32_t k,
                                 const std::vector<uint32_t> *new_to_old_idmap);

// ============================================================
// Latency percentiles (per-query, last round; worst rank via MPI_MAX)
// ============================================================

struct LatencyPercentiles {
    double p50 = 0;
    double p95 = 0;
    double p99 = 0;
    double max = 0;
};

LatencyPercentiles compute_latency_percentiles(std::vector<double> &query_latencies_us, // sorted in-place
                                               uint64_t my_nq, MPI_Comm comm);

// ============================================================
// Recall aggregate (MPI_Reduce)
// ============================================================

double aggregate_recall(const RecallStats &local_stats, int world_rank, MPI_Comm comm);

// ============================================================
// Benchmark summary printing (rank 0 only)
// ============================================================

struct BenchReportArgs {
    const char *dataset_path;
    uint64_t n_train;
    uint64_t nq;
    int world_size;
    int num_threads;
    uint32_t num_shards;
    int num_rounds;
    uint32_t M;
    uint32_t ef_search;
    int32_t K;
};

void print_benchmark_summary(const BenchReportArgs &args, const std::vector<double> &round_qps,
                             const std::vector<double> &round_e2e_qps, const LatencyPercentiles &lat,
                             double avg_recall);

} // namespace gd_hnsw
