/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hnsw_urma/control/mpi_control.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/runtime/hybrid_distance.hpp"
#include "hnsw_urma/runtime/topology.hpp"
#include "hnsw_urma/transport/urma_resources.hpp"

namespace hnsw_urma {

enum class BenchmarkAction : std::uint8_t { kRun, kHelp, kVersion };
enum class CpuBindMode : std::uint8_t { kAuto, kNone };
enum class ProgressMode : std::uint8_t { kBusy, kAdaptive };
enum class TpType : std::uint8_t { kCtp, kRtp };
enum class IntraNodeTransport : std::uint8_t { kUrma, kSharedMemory };
enum class OutputFormat : std::uint8_t { kText, kJson, kCsv };

struct AutoU32 {
  bool automatic{true};
  std::uint32_t value{0};
};

struct RuntimeConfig {
  std::filesystem::path index;
  std::filesystem::path queries;
  std::filesystem::path ground_truth;
  std::uint64_t query_offset{0};
  std::uint64_t query_count{0};

  std::uint32_t topk{10};
  AutoU32 ef_search;
  std::uint32_t expand_batch{1};
  DistancePolicy distance_policy{DistancePolicy::kAuto};
  std::uint32_t pull_threshold{4};
  bool use_dot_norm{false};
  Layer0SortMode layer0_sort_mode{Layer0SortMode::kAll};

  AutoU32 search_workers;
  AutoU32 distance_workers;
  AutoU32 read_progress_threads;
  AutoU32 rpc_progress_threads;
  std::uint32_t max_active_queries{16};
  AutoU32 visited_capacity;
  CpuBindMode cpu_bind{CpuBindMode::kAuto};
  CpuLayout cpu_layout{CpuLayout::kAuto};
  ProgressMode read_progress_mode{ProgressMode::kBusy};
  ProgressMode rpc_progress_mode{ProgressMode::kBusy};
  AutoU32 read_progress_spin_us;
  AutoU32 rpc_progress_spin_us;

  AutoU32 read_submit_depth;
  AutoU32 read_result_slots_per_search;
  AutoU32 rpc_submit_depth;
  AutoU32 rpc_operation_slots_per_search;
  AutoU32 rpc_request_batch_max_tasks;
  AutoU32 rpc_request_batch_max_bytes;
  AutoU32 rpc_response_batch_max_bytes;
  AutoU32 rpc_response_progress_max_bytes;

  std::string urma_device;
  std::map<std::uint32_t, std::string> urma_device_map;
  AutoU32 eid_index;
  TpType tp_type{TpType::kCtp};
  IntraNodeTransport intra_node_transport{IntraNodeTransport::kUrma};
  AutoU32 read_jfs_depth;
  AutoU32 request_jfs_depth;
  AutoU32 response_jfs_depth;
  std::uint32_t request_jfs_per_rpc_lane{1};
  std::uint32_t response_jfs_per_rpc_lane{1};
  AutoU32 read_jfc_depth;
  AutoU32 request_jfc_depth;
  AutoU32 response_jfc_depth;
  std::uint32_t wr_list_max{8};
  std::uint32_t request_ring_depth{8};
  // Deterministic startup ceilings. Providers do not expose a portable
  // registered-memory byte limit, so operators may lower these per host.
  std::uint64_t queue_memory_budget_bytes{1ULL << 30U};
  std::uint64_t registered_rpc_memory_budget_bytes{8ULL << 30U};

  std::uint32_t operation_timeout_ms{5000};
  std::uint32_t query_timeout_ms{30000};
  std::uint32_t drain_timeout_ms{30000};
  std::uint32_t warmup{2};
  std::uint32_t rounds{5};
  bool profile{false};
  std::filesystem::path profile_output;
  std::filesystem::path output{"-"};
  OutputFormat output_format{OutputFormat::kText};
  bool dry_run{false};
};

struct BenchmarkCommand {
  BenchmarkAction action{BenchmarkAction::kRun};
  RuntimeConfig config;
};

struct RpcMemoryPlan {
  std::uint32_t request_staging_slots_per_thread{0};
  std::uint64_t request_stride{0};
  std::uint64_t response_stride{0};
  std::uint64_t request_mailbox_bytes{0};
  std::uint64_t response_mailbox_bytes{0};
  std::uint64_t result_ring_bytes{0};
  std::uint64_t request_staging_bytes{0};
  std::uint64_t notify_bytes{0};
  std::uint64_t request_segment_bytes{0};
  std::uint64_t response_segment_bytes{0};
  std::uint64_t notify_segment_bytes{0};
  std::uint64_t registered_rpc_bytes{0};
};

struct RpcOperationPlan {
  std::uint32_t slots_per_search{0};
  std::uint32_t total_slots{0};
  std::uint32_t search_event_capacity{0};
  std::uint32_t submit_depth{0};
  std::uint32_t minimum_submit_depth{0};
  std::uint64_t pool_bytes{0};
  std::uint64_t client_owner_bytes{0};
};

[[nodiscard]] Result<RpcOperationPlan>
BuildRpcOperationPlan(std::uint32_t queries, std::uint32_t searches,
                      std::uint32_t rpc_threads, std::uint32_t maximum_batch,
                      std::uint32_t local_rank,
                      std::span<const std::uint32_t> shard_to_rank,
                      std::span<const std::uint32_t> rpc_owner_by_rank,
                      std::uint32_t request_ring_depth, AutoU32 slots = {},
                      AutoU32 submit_depth = {});

struct EffectiveRuntimeConfig {
  std::uint32_t maximum_distance_batch{0};
  RpcOperationPlan rpc_operations;
  std::uint32_t read_result_slots_per_search{0};
  std::uint32_t read_graph_records_per_search{0};
  std::uint32_t read_pull_records_per_search{0};
  std::uint64_t registered_read_result_bytes{0};
  std::uint32_t ef_search{0};
  std::uint32_t search_workers{0};
  std::uint32_t distance_workers{0};
  std::uint32_t read_progress_threads{0};
  std::uint32_t rpc_progress_threads{0};
  std::uint32_t read_progress_spin_us{0};
  std::uint32_t rpc_progress_spin_us{0};
  std::uint32_t rpc_request_batch_max_tasks{0};
  std::uint32_t rpc_request_batch_max_bytes{0};
  std::uint32_t rpc_response_batch_max_bytes{0};
  std::uint32_t rpc_response_progress_max_bytes{0};
  std::uint32_t visited_capacity{0};
  std::optional<std::uint32_t> eid_index;
  // Ordered with the NUMA-local device first. Every listed device is opened
  // so a rank can publish matching target resources for remote peers whose
  // primary device differs from its own.
  std::vector<std::string> urma_devices;
  std::string urma_device;
  std::optional<std::uint32_t> numa_node;
  UrmaResourceOptions urma;
  RpcMemoryPlan rpc_memory;
  CommunicationOwnerMap owners;
  RuntimeQueuePlan queues;
  std::vector<std::uint32_t> allowed_cpus;
  std::vector<std::uint32_t> search_cpus;
  std::vector<std::uint32_t> distance_cpus;
  std::vector<std::uint32_t> read_progress_cpus;
  std::vector<std::uint32_t> rpc_progress_cpus;
  std::vector<std::uint32_t> reserved_cpus;
  CpuLayout cpu_layout{CpuLayout::kContiguous};
  std::vector<CpuTopologyEntry> cpu_topology;
};

[[nodiscard]] Result<RpcMemoryPlan>
BuildRpcMemoryPlan(std::uint32_t world_size, std::uint32_t rpc_threads,
                   std::uint32_t mailbox_depth,
                   std::uint32_t request_send_depth,
                   std::uint32_t request_batch_max_bytes,
                   std::uint32_t response_batch_max_bytes) noexcept;

[[nodiscard]] Result<BenchmarkCommand>
ParseBenchmarkCommand(std::span<const std::string_view> arguments);
[[nodiscard]] Result<EffectiveRuntimeConfig>
ResolveRuntimeConfig(const RuntimeConfig &config, const GlobalIndexMeta &meta,
                     std::uint32_t world_size,
                     std::span<const std::uint32_t> allowed_cpus,
                     std::uint32_t local_rank = 0,
                     std::span<const std::uint32_t> numa_nodes = {});
[[nodiscard]] Result<CanonicalConfig>
BuildCanonicalRuntimeConfig(const RuntimeConfig &requested,
                            const EffectiveRuntimeConfig &effective);
[[nodiscard]] std::string BenchmarkHelp();
[[nodiscard]] std::string BenchmarkVersion();
[[nodiscard]] std::string_view GitCommit() noexcept;
[[nodiscard]] std::string_view ToString(DistancePolicy value) noexcept;
[[nodiscard]] std::string_view ToString(OutputFormat value) noexcept;
[[nodiscard]] std::string_view ToString(CpuLayout value) noexcept;
[[nodiscard]] std::string_view ToString(Layer0SortMode value) noexcept;
[[nodiscard]] std::string_view ToString(IntraNodeTransport value) noexcept;

} // namespace hnsw_urma
