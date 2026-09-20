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

#include "bench_runner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include <mpi.h>

namespace gd_hnsw {

BenchResult run_benchmark_rounds(Context &ctx, const float *my_queries, uint64_t my_nq, uint32_t dim,
                                 const SearchParams &params, int32_t K, int num_rounds, int num_warmup,
                                 uint64_t nq_global, int world_rank, MPI_Comm comm)
{
    BenchResult result;

    uint64_t buf_size = static_cast<uint64_t>(my_nq) * K;
    result.out_ids.resize(buf_size, -1);
    result.out_dists.resize(buf_size, 0.0f);
    result.round_qps.resize(num_rounds);
    result.round_e2e_qps.resize(num_rounds);
    result.query_latencies_us.resize(my_nq);

    // Warmup rounds (untimed, batch search).
    for (int w = 0; w < num_warmup; w++) {
        if (world_rank == 0) {
            printf("[bench] Warmup round %d/%d...\n", w + 1, num_warmup);
        }
        SearchResult sr = ctx.search(my_queries, my_nq, params, result.out_ids.data(), result.out_dists.data());
        if (sr.status != Status::Ok && world_rank == 0) {
            fprintf(stderr, "[bench] warmup search failed: %s\n", sr.error.c_str());
        }
        MPI_Barrier(comm);
    }

    // Timed rounds.
    for (int r = 0; r < num_rounds; r++) {
        MPI_Barrier(comm);
        const bool last_round = (r == num_rounds - 1);

        double search_sec = 0.0;
        if (last_round) {
            // Per-query search for the latency distribution + final results.
            // Parallel per-query timing (matches the old bench's concurrency
            // model): each OMP search thread times its own queries, so
            // latencies include concurrent contention (pushdown service
            // queues, NUMA traffic) that dominates tail latency. Relies on
            // search_batch_flat's omp_in_parallel guard to avoid a nested
            // OMP region; team size == search_threads set by
            // node_init_and_sync().
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(my_nq); i++) {
                auto qt0 = std::chrono::steady_clock::now();
                SearchResult sr = ctx.search(my_queries + i * dim, 1, params, result.out_ids.data() + i * K,
                                             result.out_dists.data() + i * K);
                auto qt1 = std::chrono::steady_clock::now();
                if (sr.status != Status::Ok) {
                    result.out_ids[i * K] = -1;
                }
                result.query_latencies_us[i] = std::chrono::duration<double, std::micro>(qt1 - qt0).count();
            }
            auto t1 = std::chrono::steady_clock::now();
            search_sec = std::chrono::duration<double>(t1 - t0).count();
        } else {
            auto t0 = std::chrono::steady_clock::now();
            SearchResult sr = ctx.search(my_queries, my_nq, params, result.out_ids.data(), result.out_dists.data());
            auto t1 = std::chrono::steady_clock::now();
            if (sr.status != Status::Ok && world_rank == 0) {
                fprintf(stderr, "[bench] search failed: %s\n", sr.error.c_str());
            }
            search_sec = std::chrono::duration<double>(t1 - t0).count();
        }

        double local_qps = (search_sec > 0.0) ? (static_cast<double>(my_nq) / search_sec) : 0.0;
        double total_qps = 0;
        MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        result.round_qps[r] = total_qps;

        double max_sec = 0;
        MPI_Reduce(&search_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        double e2e_qps = (max_sec > 0.0) ? (static_cast<double>(nq_global) / max_sec) : 0.0;
        result.round_e2e_qps[r] = e2e_qps;

        if (world_rank == 0) {
            if (last_round) {
                printf("[bench] Round %d/%d: %.1f QPS (agg), %.1f QPS (e2e)\n", r + 1, num_rounds, total_qps, e2e_qps);
            } else {
                printf("[bench] Round %d/%d: %.1f QPS\n", r + 1, num_rounds, total_qps);
            }
        }
    }

    return result;
}

} // namespace gd_hnsw
