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

#include <chrono>
#include <memory>
#include <optional>
#include <span>

#include "hnsw_urma/core/shard_map.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/search/io.hpp"
#include "hnsw_urma/search/query_context.hpp"

namespace hnsw_urma {

inline constexpr std::uint64_t kDefaultDenseVisitedBudgetBytes =
    256ULL * 1024ULL * 1024ULL;

struct SearchRuntimeContext {
  QueryReadyPublisher *ready_publisher{nullptr};
  std::uint32_t ready_producer_count{0};
};

// Reusable storage whose lifetime follows a fixed QuerySlot rather than a
// Search Worker. A scratch instance may only be active for one slot generation
// at a time; HnswSearchTask releases it after all query operations retire.
class QueryScratch {
public:
  ~QueryScratch();

  [[nodiscard]] static Result<std::unique_ptr<QueryScratch>>
  Create(const GlobalIndexMeta &meta, const SearchParams &params,
         std::uint32_t active_slots, std::uint32_t ready_producer_count = 1,
         std::uint64_t dense_visited_budget_bytes =
             kDefaultDenseVisitedBudgetBytes);

  [[nodiscard]] VisitedLayout visited_layout() const noexcept;
  [[nodiscard]] bool active() const noexcept;

private:
  QueryScratch() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class HnswSearchTask;
};

class HnswSearchTask {
public:
  ~HnswSearchTask();

  [[nodiscard]] static Result<std::unique_ptr<HnswSearchTask>>
  Create(QueryKey key, const GlobalIndexMeta &meta, const ShardMap &shard_map,
         GraphReader &graph_reader, DistanceOracle &distance_oracle,
         std::span<const float> query, const SearchParams &params,
         std::chrono::steady_clock::time_point admission_time =
             std::chrono::steady_clock::now(),
         const SearchRuntimeContext *runtime_context = nullptr,
         QueryScratch *scratch = nullptr,
         // A non-null owner keeps query.data() alive and enables zero-copy.
         // Empty owners retain the safe QueryScratch snapshot fallback.
         std::shared_ptr<const void> query_owner = {});

  [[nodiscard]] SearchStepResult
  Step(std::chrono::steady_clock::time_point now =
           std::chrono::steady_clock::now());

  // Legacy/test compatibility hook for polling GraphReader/DistanceOracle
  // adapters. The production distributed runner never calls it: owner-lane
  // terminal callbacks converge the phase and publish the ready handle
  // directly.
  [[nodiscard]] Result<bool>
  ProgressWaiting(std::chrono::steady_clock::time_point now =
                      std::chrono::steady_clock::now());

  [[nodiscard]] QueryState query_state() const noexcept;
  [[nodiscard]] RunState run_state() const noexcept;
  [[nodiscard]] const Status &status() const noexcept;
  [[nodiscard]] const SearchResult &result() const noexcept;
  [[nodiscard]] const QueryRunController &run_controller() const noexcept;
  [[nodiscard]] Status BindOwner(std::uint32_t owner) noexcept;
  [[nodiscard]] Status RequestFailure(TerminalOutcome outcome,
                                      StatusCode code) noexcept;
  [[nodiscard]] std::optional<QueryRetryHandle> TakeRetryHandle() noexcept;
  [[nodiscard]] QueryKey key() const noexcept;

private:
  HnswSearchTask() = default;
  struct Impl;
  std::unique_ptr<QueryScratch> owned_scratch_;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Result<SearchResult>
RunToCompletion(std::unique_ptr<HnswSearchTask> task,
                std::uint64_t max_steps = 1000000);

} // namespace hnsw_urma
