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

#include "hnsw_urma/milvus_ub_runtime.hpp"
#include "hnsw_urma/runtime/config.hpp"
#include <set>
#include <sstream>

namespace hnsw_urma {
Result<RuntimeConfig>
BuildMilvusRuntimeConfig(const MilvusUbRuntime::Knobs &k) {
  // Reuse the pinned parser and resolver; no second set of enum/range rules.
  std::vector<std::string> args{"--index", "milvus", "--dry-run"};
  const auto add = [&](const char *name, auto value) {
    args.emplace_back(name);
    args.push_back(std::to_string(value));
  };
  if (k.search_workers)
    add("--search-workers", k.search_workers);
  if (k.distance_workers)
    add("--distance-workers", k.distance_workers);
  if (k.read_progress_workers)
    add("--read-progress-threads", k.read_progress_workers);
  if (k.rpc_progress_workers)
    add("--rpc-progress-threads", k.rpc_progress_workers);
  add("--max-active-queries-per-rank", k.max_active_queries);
  if (k.read_jfs_depth)
    add("--read-jfs-depth", k.read_jfs_depth);
  if (k.queue_memory_budget_bytes)
    add("--queue-memory-budget-bytes", k.queue_memory_budget_bytes);
  if (k.registered_rpc_memory_budget_bytes)
    add("--registered-rpc-memory-budget-bytes",
        k.registered_rpc_memory_budget_bytes);
  if (k.operation_timeout_ms)
    add("--operation-timeout-ms", k.operation_timeout_ms);
  if (k.query_timeout_ms)
    add("--query-timeout-ms", k.query_timeout_ms);
  if (k.expand_batch)
    add("--expand-batch", k.expand_batch);
  args.insert(args.end(),
              {"--cpu-bind", k.cpu_bind_mode, "--cpu-layout", k.cpu_layout,
               "--read-progress-mode", k.progress_mode, "--rpc-progress-mode",
               k.progress_mode});
  const std::set<std::string> value_options{
      "--expand-batch",
      "--distance-policy",
      "--pull-threshold",
      "--layer0-sort",
      "--visited-capacity",
      "--search-workers",
      "--distance-workers",
      "--read-progress-threads",
      "--rpc-progress-threads",
      "--max-active-queries-per-rank",
      "--cpu-bind",
      "--cpu-layout",
      "--read-progress-mode",
      "--rpc-progress-mode",
      "--read-progress-spin-us",
      "--rpc-progress-spin-us",
      "--read-submit-depth",
      "--read-result-slots-per-search",
      "--rpc-submit-depth",
      "--rpc-operation-slots-per-search",
      "--rpc-request-batch-max-tasks",
      "--rpc-request-batch-max-bytes",
      "--rpc-response-batch-max-bytes",
      "--rpc-response-progress-max-bytes",
      "--read-jfs-depth",
      "--request-jfs-depth",
      "--response-jfs-depth",
      "--read-jfc-depth",
      "--request-jfc-depth",
      "--response-jfc-depth",
      "--request-jfs-per-rpc-lane",
      "--response-jfs-per-rpc-lane",
      "--wr-list-max",
      "--request-ring-depth",
      "--queue-memory-budget-bytes",
      "--registered-rpc-memory-budget-bytes",
      "--operation-timeout-ms",
      "--query-timeout-ms",
      "--drain-timeout-ms"};
  std::istringstream options(k.runtime_options);
  for (std::string name; options >> name;) {
    if (name == "--use-dot-norm" || name == "--profile") {
      args.push_back(name);
    } else if (value_options.count(name)) {
      std::string value;
      if (!(options >> value))
        return Status(StatusCode::kInvalidArgument,
                      "missing runtime option value: " + name);
      args.push_back(name);
      args.push_back(value);
    } else {
      return Status(StatusCode::kInvalidArgument,
                    "unsupported Milvus runtime option: " + name);
    }
  }
  // Device/EID belong to the shared UrmaManager context, configured by
  // queryNode.urma. They cannot be independently overridden by this adapter.
  args.insert(args.end(), {"--urma-device", k.urma_device});
  std::vector<std::string_view> views(args.begin(), args.end());
  auto parsed = ParseBenchmarkCommand(views);
  if (!parsed.ok())
    return parsed.status();
  auto config = std::move(parsed).value().config;
  // UrmaManager accepts EID 0. The reference CLI rejects numeric zero, so
  // take the already selected context EID directly, never its CLI auto rule.
  config.eid_index = {false, k.eid_index};
  return config;
}
Result<RuntimeConfig> MilvusUbRuntime::RequestedRuntimeConfig() const {
  return BuildMilvusRuntimeConfig(knobs_);
}
} // namespace hnsw_urma
