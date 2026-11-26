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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/search/query_context.hpp"

namespace hnsw_urma {

struct QueryBenchmarkRecord {
  std::uint64_t query_id{0};
  std::uint64_t latency_nanoseconds{0};
  std::optional<double> recall;
  bool success{false};
  SearchProfile search;
  std::uint64_t queue_wait_nanoseconds{0};
  std::uint64_t execution_nanoseconds{0};
  std::uint64_t upper_elapsed_nanoseconds{0};
  std::uint64_t layer0_elapsed_nanoseconds{0};
  std::uint64_t local_execution_nanoseconds{0};
  std::uint64_t graph_wait_nanoseconds{0};
  std::uint64_t distance_wait_nanoseconds{0};
  std::uint64_t ready_queue_nanoseconds{0};
  std::uint64_t upper_search_nanoseconds{0};
  std::uint64_t graph_plan_nanoseconds{0};
  std::uint64_t graph_publish_nanoseconds{0};
  std::uint64_t graph_consume_nanoseconds{0};
  std::uint64_t neighbor_scan_visited_nanoseconds{0};
  std::uint64_t route_group_sort_nanoseconds{0};
  std::uint64_t local_distance_nanoseconds{0};
  std::uint64_t remote_prepare_publish_nanoseconds{0};
  std::uint64_t remote_finish_nanoseconds{0};
  std::uint64_t commit_heap_nanoseconds{0};
  std::uint64_t controller_other_nanoseconds{0};
};

struct LatencySummary {
  double mean_ms{0.0};
  double p50_ms{0.0};
  double p95_ms{0.0};
  double p99_ms{0.0};
};

struct BenchmarkSummary {
  std::uint64_t query_count{0};
  std::uint64_t succeeded{0};
  std::uint64_t failed{0};
  double elapsed_seconds{0.0};
  double qps{0.0};
  double latency_mean_ms{0.0};
  double latency_p50_ms{0.0};
  double latency_p95_ms{0.0};
  double latency_p99_ms{0.0};
  LatencySummary queue_wait_latency;
  LatencySummary execution_latency;
  LatencySummary upper_elapsed_latency;
  LatencySummary layer0_elapsed_latency;
  LatencySummary local_execution_latency;
  LatencySummary graph_wait_latency;
  LatencySummary distance_wait_latency;
  LatencySummary ready_queue_latency;
  LatencySummary upper_search_latency;
  LatencySummary graph_plan_latency;
  LatencySummary graph_publish_latency;
  LatencySummary graph_consume_latency;
  LatencySummary neighbor_scan_visited_latency;
  LatencySummary route_group_sort_latency;
  LatencySummary local_distance_latency;
  LatencySummary remote_prepare_publish_latency;
  LatencySummary remote_finish_latency;
  LatencySummary commit_heap_latency;
  LatencySummary controller_other_latency;
  std::optional<double> recall_mean;
  SearchProfile search;
};

// Schema v6 separates Search result pools, WR lists and transport resources.
struct RuntimeProfileSnapshot {
  std::uint64_t read_batches_submitted{0};
  std::uint64_t read_batches_failed{0};
  std::uint64_t read_batches_timed_out{0};
  std::uint64_t read_batches_cancelled{0};
  std::uint64_t read_requests_posted{0};
  std::uint64_t read_requests_completed{0};
  std::uint64_t read_bytes_completed{0};
  std::uint64_t read_wr_completion_samples{0};
  std::uint64_t read_wr_completion_nanoseconds{0};
  std::uint64_t read_cqe_failure_count{0};
  std::uint64_t pushdown_requests{0};
  std::uint64_t pushdown_responses{0};
  std::uint64_t request_bytes{0};
  std::uint64_t response_bytes{0};
  std::uint64_t admission_high_watermark{0};
  std::uint64_t lane_affinity_hits{0};
  std::uint64_t lane_affinity_inserts{0};
  std::uint64_t lane_affinity_replacements{0};
  std::uint64_t lane_candidate_capacity_rejections{0};
  std::uint64_t read_exceptional_batches_high_watermark{0};
  std::uint64_t read_late_completion_count{0};
  std::uint64_t rpc_quarantine_high_watermark{0};
  std::uint64_t rpc_detached_records{0};
  std::uint64_t rpc_late_completion_count{0};
  std::uint64_t rpc_late_response_count{0};
  std::uint64_t read_thread_count{0};
  std::uint64_t rpc_thread_count{0};
  std::uint64_t request_jfs_per_rpc_lane{1};
  std::uint64_t response_jfs_per_rpc_lane{1};
  std::uint64_t read_result_slots_high_watermark{0};
  std::uint64_t read_in_flight_high_watermark{0};
  std::uint64_t read_native_descriptors_high_watermark{0};
  std::uint64_t read_post_calls{0};
  std::uint64_t read_post_list_wrs{0};
  std::uint64_t read_post_list_high_watermark{0};
  std::uint64_t read_submit_wait_nanoseconds{0};
  std::uint64_t read_reclaim_wait_nanoseconds{0};
  std::uint64_t read_result_allocation_retries{0};
  std::uint64_t read_result_registered_bytes{0};

  std::uint64_t rpc_submit_queue_high_watermark{0};
  std::uint64_t rpc_operation_slots_in_use{0};
  std::uint64_t rpc_operation_slots_high_watermark{0};
  std::uint64_t rpc_active_audit_passes{0};
  std::uint64_t rpc_active_audit_operations{0};
  std::vector<std::uint64_t> rpc_slots_in_use_per_search;
  std::vector<std::uint64_t> rpc_slots_high_watermark_per_search;
  std::vector<std::uint64_t> rpc_pool_backpressure_per_search;
  std::vector<std::uint64_t> rpc_submit_backpressure_per_search;
  std::vector<std::uint64_t> rpc_audit_operations_per_owner;
  std::uint64_t rpc_pool_backpressure{0};
  std::uint64_t rpc_submit_backpressure{0};
  std::uint64_t rpc_exceptional_reclaimed{0};
  std::uint64_t rpc_ticket_block_refills{0};
  std::uint64_t logical_distance_tasks{0};
  std::uint64_t completed_distance_tasks{0};
  std::uint64_t request_batches{0};
  std::uint64_t request_batch_tasks{0};
  std::uint64_t request_batch_bytes{0};
  std::uint64_t response_batches{0};
  std::uint64_t response_batch_tasks{0};
  std::uint64_t response_batch_bytes{0};
  std::uint64_t request_physical_writes{0};
  std::uint64_t response_physical_writes{0};
  std::uint64_t request_completions{0};
  std::uint64_t response_completions{0};
  std::uint64_t request_write_completion_nanoseconds{0};
  std::uint64_t response_write_completion_nanoseconds{0};
  std::uint64_t rpc_client_owner_queue_nanoseconds{0};
  std::uint64_t rpc_client_owner_queue_samples{0};
  std::uint64_t rpc_client_owner_pending_nanoseconds{0};
  std::uint64_t rpc_client_request_prepare_nanoseconds{0};
  std::uint64_t rpc_client_request_post_wait_nanoseconds{0};
  std::uint64_t rpc_client_round_trip_nanoseconds{0};
  std::uint64_t rpc_client_round_trip_samples{0};
  std::uint64_t rpc_client_response_notify_to_receive_nanoseconds{0};
  std::uint64_t rpc_client_response_notify_to_receive_samples{0};
  std::uint64_t rpc_client_response_notify_to_receive_task_nanoseconds{0};
  std::uint64_t rpc_client_response_notify_to_receive_task_samples{0};
  std::uint64_t cluster_bidirectional_post_to_notify_observed_nanoseconds{0};
  std::uint64_t cluster_bidirectional_post_to_notify_observed_samples{0};
  std::uint64_t rpc_client_response_received_to_task_terminal_nanoseconds{0};
  std::uint64_t rpc_client_response_received_to_task_terminal_samples{0};
  std::uint64_t rpc_exceptional_retained_nanoseconds{0};
  std::uint64_t rpc_client_response_to_stage_terminal_nanoseconds{0};
  std::uint64_t rpc_client_response_to_stage_terminal_samples{0};
  std::uint64_t rpc_client_response_decode_nanoseconds{0};
  std::uint64_t rpc_client_response_decode_samples{0};
  std::uint64_t request_slot_scans{0};
  std::uint64_t request_slot_hits{0};
  std::uint64_t request_slot_blocked_by_result_inflight{0};
  std::uint64_t server_request_batches{0};
  std::uint64_t server_request_tasks{0};
  std::uint64_t server_distance_values{0};
  std::uint64_t request_poll_hit_to_claim_nanoseconds{0};
  std::uint64_t request_poll_hit_to_claim_samples{0};
  std::uint64_t request_decode_nanoseconds{0};
  std::uint64_t query_slot_store_nanoseconds{0};
  std::uint64_t query_slot_load_nanoseconds{0};
  std::uint64_t distance_compute_nanoseconds{0};
  std::uint64_t filter_and_response_encode_nanoseconds{0};
  std::uint64_t result_ready_publications{0};
  std::uint64_t distance_busy_iterations{0};
  std::uint64_t distance_idle_iterations{0};
  std::uint64_t result_slot_scans{0};
  std::uint64_t result_slot_hits{0};
  std::uint64_t request_notify_observed_to_response_post_nanoseconds{0};
  std::uint64_t request_notify_observed_to_response_post_samples{0};
  std::uint64_t result_ready_to_post_nanoseconds{0};
  std::uint64_t result_ready_to_post_samples{0};
  std::uint64_t result_post_retries{0};
  std::uint64_t result_post_to_completion_nanoseconds{0};
  std::uint64_t result_post_to_completion_samples{0};
  std::uint64_t response_scan_slots{0};
  std::uint64_t response_scan_hits{0};
  std::uint64_t request_slots_in_use{0};
  std::uint64_t request_slots_high_watermark{0};
  std::uint64_t request_post_free_slot_waits{0};
  std::uint64_t request_post_staging_unavailable{0};
  std::uint64_t request_post_tx_unavailable{0};
  std::uint64_t request_post_provider_backpressure_batches{0};
  std::uint64_t rpc_reclaim_events_published{0};
  std::uint64_t rpc_reclaim_events_consumed{0};
  std::uint64_t rpc_exceptional_slots_in_use{0};
  std::uint64_t rpc_exceptional_slots_high_watermark{0};
  std::vector<std::uint64_t> ready_published_per_worker;
  std::vector<std::uint64_t> ready_consumed_per_worker;
  std::vector<std::uint64_t> ready_producer_worker_counts;
  std::vector<std::uint64_t> ready_converged_to_publish_ns_per_worker;
  std::vector<std::uint64_t> ready_publish_to_pop_ns_per_worker;
  std::vector<std::uint64_t> ready_pop_to_step_ns_per_worker;
  std::vector<std::uint64_t> ready_run_ns_per_worker;
  std::vector<std::uint64_t> search_event_depth;
  std::vector<std::uint64_t> search_event_high_watermark;
  std::uint64_t ready_events_published{0};
  std::uint64_t ready_events_consumed{0};
  std::uint64_t ready_queue_high_watermark{0};

  std::vector<std::uint64_t> read_owned_ranks_per_thread;
  std::vector<std::uint64_t> rpc_owned_ranks_per_thread;
  std::vector<std::uint64_t> search_busy_nanoseconds_per_thread;
  std::vector<std::uint64_t> search_idle_nanoseconds_per_thread;
  std::vector<std::uint64_t> search_iterations_per_thread;
  std::vector<std::uint64_t> search_new_queries_per_thread;
  std::vector<std::uint64_t> search_ready_resumes_per_thread;
  std::vector<std::uint64_t> search_retry_runs_per_thread;
  std::vector<std::uint64_t> distance_busy_nanoseconds_per_thread;
  std::vector<std::uint64_t> distance_idle_nanoseconds_per_thread;
  std::vector<std::uint64_t> distance_iterations_per_thread;
  std::vector<std::uint64_t> read_busy_nanoseconds_per_thread;
  std::vector<std::uint64_t> read_idle_nanoseconds_per_thread;
  std::vector<std::uint64_t> read_progress_iterations_per_thread;
  std::vector<std::uint64_t> rpc_busy_nanoseconds_per_thread;
  std::vector<std::uint64_t> rpc_idle_nanoseconds_per_thread;
  std::vector<std::uint64_t> rpc_progress_iterations_per_thread;
  std::vector<std::uint64_t> rpc_poll_completions_nanoseconds_per_thread;
  std::vector<std::uint64_t> rpc_scan_responses_nanoseconds_per_thread;
  std::vector<std::uint64_t> rpc_post_responses_nanoseconds_per_thread;
  std::vector<std::uint64_t>
      rpc_drain_client_submissions_nanoseconds_per_thread;
  std::vector<std::uint64_t> rpc_post_requests_nanoseconds_per_thread;
  std::vector<std::uint64_t> distance_request_slot_scans_per_thread;
  std::vector<std::uint64_t> distance_request_slot_hits_per_thread;
  std::vector<std::uint64_t> distance_request_batches_per_thread;
  std::vector<std::uint64_t> distance_request_tasks_per_thread;
  std::vector<std::uint64_t> distance_values_per_thread;
  std::vector<std::uint64_t> wait_event_counts;
  std::vector<std::uint64_t> wait_nanoseconds;
  std::vector<std::uint64_t> wait_publish_nanoseconds;
  std::vector<std::uint64_t> wait_first_completion_nanoseconds;
  std::vector<std::uint64_t> wait_completion_spread_nanoseconds;
  std::vector<std::uint64_t> wait_remote_operations;
  std::vector<std::uint64_t> wait_distance_values;
  std::vector<std::uint64_t> wait_remote_operations_high_watermark;
  std::vector<std::uint64_t> wait_distance_values_high_watermark;
  std::vector<std::uint64_t> wait_critical_lane_counts;
  std::vector<std::uint64_t> wait_histograms;
  std::vector<std::uint64_t> wait_publish_histograms;
  std::vector<std::uint64_t> wait_first_completion_histograms;
  std::vector<std::uint64_t> wait_completion_spread_histograms;
};

struct DistributedBenchmarkPayload {
  std::vector<QueryBenchmarkRecord> records;
  double elapsed_seconds{0.0};
  RuntimeProfileSnapshot profile;
};

[[nodiscard]] BenchmarkSummary
SummarizeBenchmark(std::span<const QueryBenchmarkRecord> records,
                   double elapsed_seconds);
[[nodiscard]] double RecallAtK(std::span<const HeapEntry> result,
                               std::span<const node_id_t> truth,
                               std::uint32_t topk) noexcept;
[[nodiscard]] std::string
RenderBenchmarkSummary(const BenchmarkSummary &summary, OutputFormat format);
[[nodiscard]] std::string RenderProfileJson(
    const BenchmarkSummary &summary, const RuntimeProfileSnapshot &profile,
    std::string_view git_commit, std::uint32_t rank, std::uint32_t world_size);
[[nodiscard]] std::string RenderProfileCsv(
    const BenchmarkSummary &summary, const RuntimeProfileSnapshot &profile,
    std::string_view git_commit, std::uint32_t rank, std::uint32_t world_size);
[[nodiscard]] Result<std::vector<std::byte>>
EncodeDistributedBenchmarkPayload(std::span<const QueryBenchmarkRecord> records,
                                  double elapsed_seconds,
                                  const RuntimeProfileSnapshot &profile);
[[nodiscard]] Result<DistributedBenchmarkPayload>
DecodeDistributedBenchmarkPayload(std::span<const std::byte> input);
[[nodiscard]] Status WriteOutput(const std::filesystem::path &path,
                                 std::string_view content);

} // namespace hnsw_urma
