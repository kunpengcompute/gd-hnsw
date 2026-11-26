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
#include <span>
#include <vector>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

inline constexpr std::uint32_t kNoCommunicationOwner = UINT32_MAX;

enum class CpuLayout : std::uint8_t {
  kAuto,
  kContiguous,
  kLlcIsolated,
  kLlcSpread,
};

struct CpuTopologyEntry {
  std::uint32_t cpu{0};
  std::uint32_t physical_core{0};
  std::uint32_t llc_id{0};
};

struct CpuRoleCounts {
  std::uint32_t search{0};
  std::uint32_t distance{0};
  std::uint32_t read_progress{0};
  std::uint32_t rpc_progress{0};
  std::uint32_t reserved{0};
};

struct CpuRoleLayout {
  std::vector<std::uint32_t> search;
  std::vector<std::uint32_t> distance;
  std::vector<std::uint32_t> read_progress;
  std::vector<std::uint32_t> rpc_progress;
  std::vector<std::uint32_t> reserved;
};

[[nodiscard]] Result<std::vector<CpuTopologyEntry>>
DiscoverCpuTopology(std::span<const std::uint32_t> allowed_cpus);
[[nodiscard]] Result<std::vector<std::uint32_t>>
DiscoverNumaNodes(std::span<const std::uint32_t> allowed_cpus);
[[nodiscard]] Result<CpuRoleLayout>
BuildContiguousCpuLayout(std::span<const std::uint32_t> allowed_cpus,
                         CpuRoleCounts counts);
[[nodiscard]] Result<CpuRoleLayout>
BuildLlcIsolatedCpuLayout(std::span<const std::uint32_t> allowed_cpus,
                          std::span<const CpuTopologyEntry> topology,
                          CpuRoleCounts counts);
[[nodiscard]] Result<CpuRoleLayout>
BuildLlcSpreadCpuLayout(std::span<const std::uint32_t> allowed_cpus,
                        std::span<const CpuTopologyEntry> topology,
                        CpuRoleCounts counts);

struct CommunicationOwnerMap {
  std::vector<std::uint32_t> read_owner_by_rank;
  std::vector<std::uint32_t> rpc_owner_by_rank;
  std::vector<std::vector<std::uint32_t>> read_owned_ranks;
  std::vector<std::vector<std::uint32_t>> rpc_owned_ranks;
};

// Builds a stable, balanced mapping over remote ranks in ascending rank order.
// The local rank has kNoCommunicationOwner because local work never enters a
// remote communication pool.
[[nodiscard]] Result<CommunicationOwnerMap>
BuildCommunicationOwnerMap(std::uint32_t world_size, std::uint32_t local_rank,
                           std::uint32_t read_threads,
                           std::uint32_t rpc_threads);

struct RuntimeQueueDepths {
  std::uint32_t admission{0};
  std::uint32_t ready{0};
  std::uint32_t read_submit{0};
  std::uint32_t read_reclaim{0};
  std::uint32_t rpc_submit{0};
  std::uint32_t done{0};
};

struct RuntimeQueuePlan {
  std::uint64_t admission_rings{0};
  std::uint64_t done_rings{0};
  std::uint64_t read_submit_rings{0};
  std::uint64_t read_reclaim_rings{0};
  std::uint64_t rpc_submit_rings{0};
  std::uint64_t ready_rings{0};
  std::uint64_t deferred_ready_entries{0};
  std::uint64_t private_work_bytes{0};
  std::uint64_t mpsc_ring_count{0};
  std::uint64_t retry_entries{0};
  std::uint64_t spsc_ring_count{0};
  std::uint64_t ring_metadata_bytes{0};
  std::uint64_t ring_storage_bytes{0};
  std::uint64_t queue_bytes_total{0};
  RuntimeQueueDepths depths;
};

inline constexpr std::uint64_t kRpcSubmitHandleBytes = 32;

[[nodiscard]] Result<RuntimeQueuePlan>
BuildRuntimeQueuePlan(std::uint32_t search_workers, std::uint32_t read_threads,
                      std::uint32_t rpc_threads, RuntimeQueueDepths depths,
                      bool profiling_enabled = false,
                      std::uint32_t world_size = 1);

[[nodiscard]] std::string
FormatCommunicationOwnerMap(const CommunicationOwnerMap &owners);

} // namespace hnsw_urma
