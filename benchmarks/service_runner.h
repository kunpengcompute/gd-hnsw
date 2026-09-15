/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include "dist_service.h"

#include <atomic>
#include <thread>
#include <vector>

namespace gd_hnsw {

class DualDistService {
public:
    DualDistService(uint32_t my_shard, const ShardView &shard, uint32_t dim, uint64_t run_generation,
                    uint32_t service_threads, std::vector<ServiceChannelPair> all_pairs, std::vector<int> core_ids = {},
                    bool profile_enabled = false);

    ~DualDistService();

    void shutdown();

    std::vector<ServiceWorkerStats> get_all_stats() const;

    void enable_dot_norm(const float *norms);

    void print_stats(int rank, uint32_t shard) const;

private:
    std::atomic<bool> stop_;
    std::vector<DualDistServiceWorker> workers_;
    std::vector<std::thread> threads_;
};

} // namespace gd_hnsw
