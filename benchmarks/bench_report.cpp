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

#include "bench_report.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace gd_hnsw {

RecallStats compute_recall_stats(const int32_t *result_ids, uint64_t nq, const int32_t *gt, uint32_t gt_k, int32_t k,
                                 const std::vector<uint32_t> *new_to_old_idmap)
{
    RecallStats stats;
    stats.total = nq * static_cast<uint64_t>(std::min(k, static_cast<int32_t>(gt_k)));
    for (uint64_t q = 0; q < nq; q++) {
        const int32_t *res_row = result_ids + q * k;
        const int32_t *gt_row = gt + q * gt_k;
        for (int32_t i = 0; i < k; i++) {
            int32_t id = res_row[i];
            if (id < 0)
                continue;
            if (new_to_old_idmap) {
                if (static_cast<size_t>(id) >= new_to_old_idmap->size())
                    continue;
                id = static_cast<int32_t>((*new_to_old_idmap)[static_cast<size_t>(id)]);
            }
            for (int32_t j = 0; j < k && j < static_cast<int32_t>(gt_k); j++) {
                if (gt_row[j] == id) {
                    stats.hits++;
                    break;
                }
            }
        }
    }
    return stats;
}

LatencyPercentiles compute_latency_percentiles(std::vector<double> &query_latencies_us, uint64_t my_nq, MPI_Comm comm)
{
    LatencyPercentiles lp;
    std::sort(query_latencies_us.begin(), query_latencies_us.end());
    auto percentile = [&](double pct) -> double {
        if (my_nq == 0)
            return 0.0;
        size_t idx = static_cast<size_t>(pct / 100.0 * (my_nq - 1));
        if (idx >= static_cast<size_t>(my_nq))
            idx = my_nq - 1;
        return query_latencies_us[idx];
    };
    double local_p50 = percentile(50);
    double local_p95 = percentile(95);
    double local_p99 = percentile(99);
    double local_max_lat = (my_nq > 0) ? query_latencies_us.back() : 0.0;

    MPI_Reduce(&local_p50, &lp.p50, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_p95, &lp.p95, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_p99, &lp.p99, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_max_lat, &lp.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return lp;
}

double aggregate_recall(const RecallStats &local_stats, int world_rank, MPI_Comm comm)
{
    uint64_t total_hits = 0;
    uint64_t total_pairs = 0;
    MPI_Reduce(&local_stats.hits, &total_hits, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    MPI_Reduce(&local_stats.total, &total_pairs, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
    if (world_rank != 0)
        return 0.0;
    return (total_pairs > 0) ? (static_cast<double>(total_hits) / total_pairs) : 0.0;
}

void print_benchmark_summary(const BenchReportArgs &args, const std::vector<double> &round_qps,
                             const std::vector<double> &round_e2e_qps, const LatencyPercentiles &lat, double avg_recall)
{
    int num_rounds = args.num_rounds;
    // QPS stats: exclude last round (per-query timing overhead)
    int qps_rounds = (num_rounds > 1) ? (num_rounds - 1) : num_rounds;
    std::vector<double> qps_sample(round_qps.begin(), round_qps.begin() + qps_rounds);
    std::vector<double> e2e_qps_sample(round_e2e_qps.begin(), round_e2e_qps.begin() + qps_rounds);
    std::sort(qps_sample.begin(), qps_sample.end());
    std::sort(e2e_qps_sample.begin(), e2e_qps_sample.end());

    printf("\n========== BENCHMARK RESULTS ==========\n");
    printf("Dataset:          %s\n", args.dataset_path);
    printf("N vectors:        %lu\n", (unsigned long)args.n_train);
    printf("Queries:          %lu\n", (unsigned long)args.nq);
    printf("Workers:          %d\n", args.world_size);
    printf("Threads/worker:   %d\n", args.num_threads);
    printf("Shards:           %u\n", args.num_shards);
    printf("Rounds:           %d\n", num_rounds);
    printf("HNSW M:           %u\n", args.M);
    printf("ef_search:        %u\n", args.ef_search);
    printf("top-K:            %d\n", args.K);
    printf("---------------------------------------\n");
    printf("Recall@%d:         %.4f\n", args.K, avg_recall);

    double median_qps = qps_sample[qps_rounds / 2];
    double median_e2e_qps = e2e_qps_sample[qps_rounds / 2];
    printf("Aggregate QPS:    %.1f (median), %.1f (min), %.1f (max)\n", median_qps, qps_sample.front(),
           qps_sample.back());
    printf("E2E QPS:          %.1f (median), %.1f (min), %.1f (max)\n", median_e2e_qps, e2e_qps_sample.front(),
           e2e_qps_sample.back());
    printf("---------------------------------------\n");
    printf("Latency (per-query, last round, worst-rank):\n");
    printf("  p50:            %.3f ms\n", lat.p50 / 1000.0);
    printf("  p95:            %.3f ms\n", lat.p95 / 1000.0);
    printf("  p99:            %.3f ms\n", lat.p99 / 1000.0);
    printf("  max:            %.3f ms\n", lat.max / 1000.0);
    printf("=======================================\n");
}

} // namespace gd_hnsw
