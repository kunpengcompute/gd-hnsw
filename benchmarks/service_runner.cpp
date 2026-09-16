/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include "service_runner.h"

#include <algorithm>
#include <cstdio>
#include <pthread.h>
#include <sched.h>

namespace gd_hnsw {

DualDistService::DualDistService(uint32_t my_shard, const ShardView &shard, uint32_t dim, uint64_t run_generation,
                                 uint32_t service_threads, std::vector<ServiceChannelPair> all_pairs,
                                 std::vector<int> core_ids, bool profile_enabled)
    : stop_(false)
{
    std::vector<std::vector<ServiceChannelPair>> per_thread(service_threads);
    for (size_t i = 0; i < all_pairs.size(); i++) {
        per_thread[i % service_threads].push_back(all_pairs[i]);
    }

    for (uint32_t t = 0; t < service_threads; t++) {
        workers_.emplace_back(my_shard, shard, dim, run_generation, std::move(per_thread[t]), stop_, profile_enabled);
    }

    for (uint32_t t = 0; t < service_threads; t++) {
        auto &w = workers_[t];
        int pin_core = (!core_ids.empty()) ? core_ids[t % core_ids.size()] : -1;
        threads_.emplace_back([&w, t, pin_core]() {
#ifdef __linux__
            char name[16];
            snprintf(name, sizeof(name), "dual_svc_%u", t);
            pthread_setname_np(pthread_self(), name);
            if (pin_core >= 0) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(pin_core, &cpuset);
                if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
                    fprintf(stderr, "WARN: dual_svc_%u failed to pin to core %d\n", t, pin_core);
                }
            }
#endif
            try {
                w.run();
            } catch (const std::exception &e) {
                fprintf(stderr, "FATAL: dual service worker exception: %s\n", e.what());
                abort();
            } catch (...) {
                fprintf(stderr, "FATAL: dual service worker unknown exception\n");
                abort();
            }
        });
    }
}

DualDistService::~DualDistService() { shutdown(); }

void DualDistService::shutdown()
{
    if (stop_.exchange(true))
        return;
    for (auto &t : threads_) {
        if (t.joinable())
            t.join();
    }
}

std::vector<ServiceWorkerStats> DualDistService::get_all_stats() const
{
    std::vector<ServiceWorkerStats> result;
    result.reserve(workers_.size());
    for (const auto &w : workers_) {
        result.push_back(w.stats());
    }
    return result;
}

void DualDistService::enable_dot_norm(const float *norms)
{
    for (auto &w : workers_) {
        w.enable_dot_norm(norms);
    }
}

void DualDistService::print_stats(int rank, uint32_t shard) const
{
    auto all = get_all_stats();
    uint64_t agg_requests = 0, agg_ids = 0, agg_query_refreshes = 0;
    uint64_t agg_process_ns = 0, agg_compute_ns = 0, max_process_ns = 0;
    uint64_t agg_dist_kernel_ns = 0, agg_memcpy_wb_ns = 0;
    uint64_t agg_validate_ns = 0, agg_cache_ns = 0, agg_resp_ns = 0;
    uint64_t agg_poll_scans = 0, agg_idle_scans = 0, agg_yield_count = 0;
    uint64_t agg_batch_hist[8] = {};
    uint32_t total_channels = 0;

    for (size_t w = 0; w < all.size(); w++) {
        const auto &s = all[w];
        agg_requests += s.total_requests;
        agg_ids += s.total_ids;
        agg_query_refreshes += s.total_query_refreshes;
        agg_process_ns += s.total_process_ns;
        agg_compute_ns += s.total_compute_ns;
        agg_dist_kernel_ns += s.total_dist_kernel_ns;
        agg_memcpy_wb_ns += s.total_memcpy_wb_ns;
        agg_validate_ns += s.total_validate_ns;
        agg_cache_ns += s.total_cache_ns;
        agg_resp_ns += s.total_resp_ns;
        if (s.max_process_ns > max_process_ns)
            max_process_ns = s.max_process_ns;
        agg_poll_scans += s.poll_scans;
        agg_idle_scans += s.idle_scans;
        agg_yield_count += s.yield_count;
        for (int b = 0; b < 8; b++)
            agg_batch_hist[b] += s.batch_hist[b];
        total_channels += s.num_channels;
    }

    uint64_t max_run_ns = 0;
    for (const auto &s : all) {
        if (s.run_ns > max_run_ns)
            max_run_ns = s.run_ns;
    }

    auto safe_div = [](double num, double den) -> double { return (den > 0.0) ? (num / den) : 0.0; };
    auto safe_pct = [](double part, double total) -> double { return (total > 0.0) ? (100.0 * part / total) : 0.0; };

    double avg_ids_per_req = safe_div(agg_ids, agg_requests);
    double avg_process_ns_per_req = safe_div(agg_process_ns, agg_requests);
    double avg_compute_ns_per_req = safe_div(agg_compute_ns, agg_requests);
    double avg_dist_kernel_ns_per_req = safe_div(agg_dist_kernel_ns, agg_requests);
    double avg_memcpy_wb_ns_per_req = safe_div(agg_memcpy_wb_ns, agg_requests);
    double avg_compute_ns_per_id = safe_div(agg_compute_ns, agg_ids);
    double overhead_ns_per_req = avg_process_ns_per_req - avg_compute_ns_per_req;
    double avg_validate_ns_per_req = safe_div(agg_validate_ns, agg_requests);
    double avg_cache_ns_per_req = safe_div(agg_cache_ns, agg_requests);
    double avg_resp_ns_per_req = safe_div(agg_resp_ns, agg_requests);
    double overhead_gap = overhead_ns_per_req - avg_validate_ns_per_req - avg_cache_ns_per_req - avg_resp_ns_per_req;
    double scan_hit_pct = safe_pct(agg_poll_scans - agg_idle_scans, agg_poll_scans);
    double service_util =
        (max_run_ns > 0 && !all.empty()) ? safe_pct(agg_process_ns, static_cast<double>(all.size()) * max_run_ns) : 0.0;

    printf("\n--- Service Profile [rank %d, shard %u] (%zu workers, %u channels) ---\n", rank, shard, all.size(),
           total_channels);
    printf("  [Throughput]\n");
    printf("    total_requests:      %lu\n", (unsigned long)agg_requests);
    printf("    total_ids:           %lu (%.2f ids/req)\n", (unsigned long)agg_ids, avg_ids_per_req);
    printf("    query_refreshes:     %lu (%.1f%% of requests)\n", (unsigned long)agg_query_refreshes,
           safe_pct(agg_query_refreshes, agg_requests));
    printf("  [Timing]\n");
    printf("    avg_process_ns/req:  %.1f ns (%.3f us)\n", avg_process_ns_per_req, avg_process_ns_per_req / 1e3);
    printf("      compute:          %.1f ns (%.1f%%)\n", avg_compute_ns_per_req,
           safe_pct(avg_compute_ns_per_req, avg_process_ns_per_req));
    printf("        dist_kernel:    %.1f ns (%.1f%% of compute)\n", avg_dist_kernel_ns_per_req,
           safe_pct(avg_dist_kernel_ns_per_req, avg_compute_ns_per_req));
    printf("        memcpy_wb:      %.1f ns (%.1f%% of compute)\n", avg_memcpy_wb_ns_per_req,
           safe_pct(avg_memcpy_wb_ns_per_req, avg_compute_ns_per_req));
    printf("      overhead:         %.1f ns (%.1f%%)\n", overhead_ns_per_req > 0 ? overhead_ns_per_req : 0,
           safe_pct(overhead_ns_per_req > 0 ? overhead_ns_per_req : 0, avg_process_ns_per_req));
    printf("        validate:       %.1f ns (dim+count check)\n", avg_validate_ns_per_req);
    printf("        cache:          %.1f ns (epoch check+memcpy)\n", avg_cache_ns_per_req);
    printf("        resp_store:     %.1f ns (atomic release)\n", avg_resp_ns_per_req);
    if (overhead_gap > 1.0)
        printf("        gap:            %.1f ns (unaccounted)\n", overhead_gap);
    printf("    avg_compute_ns/id:   %.1f ns\n", avg_compute_ns_per_id);
    printf("    max_process_ns:      %lu ns (%.3f us)\n", (unsigned long)max_process_ns, max_process_ns / 1e3);
    printf("  [Poll Loop]\n");
    printf("    poll_scans:          %lu\n", (unsigned long)agg_poll_scans);
    printf("    idle_scans:          %lu (%.1f%%)\n", (unsigned long)agg_idle_scans,
           safe_pct(agg_idle_scans, agg_poll_scans));
    printf("    scan_hit_rate:       %.1f%%\n", scan_hit_pct);
    printf("    service_util:        %.1f%%\n", service_util);
    printf("    yield_count:         %lu\n", (unsigned long)agg_yield_count);
    printf("  [Batch Size Distribution]\n");
    printf("    [1]:    %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[0], safe_pct(agg_batch_hist[0], agg_requests));
    printf("    [2]:    %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[1], safe_pct(agg_batch_hist[1], agg_requests));
    printf("    [3]:    %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[2], safe_pct(agg_batch_hist[2], agg_requests));
    printf("    [4]:    %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[3], safe_pct(agg_batch_hist[3], agg_requests));
    printf("    [5-8]:  %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[4], safe_pct(agg_batch_hist[4], agg_requests));
    printf("    [9-16]: %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[5], safe_pct(agg_batch_hist[5], agg_requests));
    printf("    [17-32]:%lu (%.1f%%)\n", (unsigned long)agg_batch_hist[6], safe_pct(agg_batch_hist[6], agg_requests));
    printf("    [33+]:  %lu (%.1f%%)\n", (unsigned long)agg_batch_hist[7], safe_pct(agg_batch_hist[7], agg_requests));

    printf("  [Per-Worker]\n");
    for (size_t w = 0; w < all.size(); w++) {
        const auto &s = all[w];
        double w_avg_ns = safe_div(s.total_process_ns, s.total_requests);
        double w_util = (s.run_ns > 0) ? safe_pct(s.total_process_ns, s.run_ns) : 0.0;
        printf("    worker[%zu]: %lu reqs, %lu ids, %.1f ns/req, %.1f%% util, %u ch\n", w,
               (unsigned long)s.total_requests, (unsigned long)s.total_ids, w_avg_ns, w_util, s.num_channels);
    }

    if (total_channels > 0) {
        struct ChInfo {
            size_t worker_idx;
            size_t ch_idx;
            uint64_t requests;
            uint64_t ids;
            uint16_t src_rank;
            uint16_t thread_id;
            bool has_identity;
        };
        std::vector<ChInfo> ch_all;
        for (size_t w = 0; w < all.size(); w++) {
            for (size_t c = 0; c < all[w].channel_stats.size(); c++) {
                const auto &cs = all[w].channel_stats[c];
                ch_all.push_back({w, c, cs.requests, cs.ids_total, cs.src_rank, cs.thread_id, cs.has_identity});
            }
        }
        std::sort(ch_all.begin(), ch_all.end(),
                  [](const ChInfo &a, const ChInfo &b) { return a.requests > b.requests; });

        size_t show = std::min(ch_all.size(), size_t(5));
        printf("  [Channel Hotspots] (top %zu / %zu)\n", show, ch_all.size());
        for (size_t k = 0; k < show; k++) {
            const auto &c = ch_all[k];
            if (c.has_identity) {
                printf("    w[%zu]ch[%zu] (rank%u:t%u): %lu reqs, %lu ids\n", c.worker_idx, c.ch_idx, c.src_rank,
                       c.thread_id, (unsigned long)c.requests, (unsigned long)c.ids);
            } else {
                printf("    w[%zu]ch[%zu]: %lu reqs, %lu ids\n", c.worker_idx, c.ch_idx, (unsigned long)c.requests,
                       (unsigned long)c.ids);
            }
        }
        if (ch_all.size() > 5) {
            printf("  [Channel Coldspots] (bottom %zu / %zu)\n", show, ch_all.size());
            for (size_t k = ch_all.size() - show; k < ch_all.size(); k++) {
                const auto &c = ch_all[k];
                if (c.has_identity) {
                    printf("    w[%zu]ch[%zu] (rank%u:t%u): %lu reqs, %lu ids\n", c.worker_idx, c.ch_idx, c.src_rank,
                           c.thread_id, (unsigned long)c.requests, (unsigned long)c.ids);
                } else {
                    printf("    w[%zu]ch[%zu]: %lu reqs, %lu ids\n", c.worker_idx, c.ch_idx, (unsigned long)c.requests,
                           (unsigned long)c.ids);
                }
            }
        }
    }
}

} // namespace gd_hnsw
