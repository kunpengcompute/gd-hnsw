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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "hnsw_urma/protocol/distance_batch_wire.hpp"
#include "hnsw_urma/runtime/benchmark.hpp"
#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/runtime/scheduler.hpp"
#include "hnsw_urma/runtime/topology.hpp"

namespace {

TEST(ReferenceUnitRuntime, RuntimeCliParsesDirectSlotSurface) {
  const std::string_view arguments[]{
      "--index",   "/index",
      "--dry-run", "--search-workers",
      "8",         "--distance-workers",
      "4",         "--read-progress-threads",
      "2",         "--rpc-progress-threads",
      "2",         "--request-ring-depth",
      "16",        "--rpc-request-batch-max-tasks",
      "12",        "--rpc-request-batch-max-bytes",
      "4096",      "--rpc-response-batch-max-bytes",
      "8192",      "--rpc-response-progress-max-bytes",
      "16384",     "--profile"};
  auto command = hnsw_urma::ParseBenchmarkCommand(arguments);
  UB_REQUIRE_TRUE(command.ok());
  UB_REQUIRE_EQ(command.value().config.distance_workers.value, 4U);
  UB_REQUIRE_EQ(command.value().config.request_ring_depth, 16U);
  UB_REQUIRE_EQ(command.value().config.rpc_request_batch_max_tasks.value, 12U);
  UB_REQUIRE_EQ(command.value().config.rpc_request_batch_max_bytes.value,
                4096U);
  UB_REQUIRE_EQ(command.value().config.rpc_response_batch_max_bytes.value,
                8192U);
  UB_REQUIRE_TRUE(command.value().config.profile);
}

TEST(ReferenceUnitRuntime, RuntimeCliRejectsRemovedSplitPipelineOptions) {
  const std::array<std::string_view, 9> removed{
      {"--distance-progress-mode", "--rpc-service-depth",
       "--rpc-remote-task-slots", "--rpc-batching",
       "--rpc-response-batch-max-tasks", "--query-cache-entries",
       "--remote-adjacency-writeback", "--read-batch-depth",
       "--direct-graph-consume"}};
  for (const auto option : removed) {
    const std::vector<std::string_view> arguments{"--index", "/index",
                                                  "--dry-run", option, "1"};
    auto parsed = hnsw_urma::ParseBenchmarkCommand(arguments);
    UB_REQUIRE_FALSE(parsed.ok());
    UB_REQUIRE_EQ(parsed.status().code(),
                  hnsw_urma::StatusCode::kInvalidArgument);
  }
}

TEST(ReferenceUnitRuntime, RpcMemoryPlanContainsOnlyThreeRegisteredSegments) {
  auto plan = hnsw_urma::BuildRpcMemoryPlan(4, 2, 8, 16, 4096, 2048);
  UB_REQUIRE_TRUE(plan.ok());
  UB_REQUIRE_EQ(plan.value().request_staging_slots_per_thread, 16U);
  UB_REQUIRE_EQ(plan.value().request_stride, 4096U);
  UB_REQUIRE_EQ(plan.value().response_stride, 2048U);
  UB_REQUIRE_EQ(plan.value().request_mailbox_bytes, 4U * 8U * 4096U);
  UB_REQUIRE_EQ(plan.value().response_mailbox_bytes, 4U * 8U * 2048U);
  UB_REQUIRE_EQ(plan.value().result_ring_bytes, 4U * 8U * 2048U);
  UB_REQUIRE_EQ(plan.value().request_staging_bytes, 2U * 16U * 4096U);
  UB_REQUIRE_EQ(plan.value().notify_bytes, 2U * 4U * 8U * 64U);
  UB_REQUIRE_EQ(plan.value().registered_rpc_bytes,
                plan.value().request_segment_bytes +
                    plan.value().response_segment_bytes +
                    plan.value().notify_segment_bytes);
}

TEST(ReferenceUnitRuntime, OwnerAndQueuePlansUseOnlyLiveDirectSlotQueues) {
  auto owners = hnsw_urma::BuildCommunicationOwnerMap(5, 2, 2, 2);
  UB_REQUIRE_TRUE(owners.ok());
  UB_REQUIRE_EQ(owners.value().rpc_owner_by_rank[2],
                hnsw_urma::kNoCommunicationOwner);
  UB_REQUIRE_EQ(owners.value().rpc_owned_ranks[0].size(), 2U);
  UB_REQUIRE_EQ(owners.value().rpc_owned_ranks[1].size(), 2U);

  const hnsw_urma::RuntimeQueueDepths depths{16, 16, 64, 128, 32, 16};
  auto queues = hnsw_urma::BuildRuntimeQueuePlan(8, 2, 2, depths);
  UB_REQUIRE_TRUE(queues.ok());
  UB_REQUIRE_EQ(queues.value().admission_rings, 8U);
  UB_REQUIRE_EQ(queues.value().done_rings, 8U);
  UB_REQUIRE_EQ(queues.value().read_submit_rings, 16U);
  UB_REQUIRE_EQ(queues.value().read_reclaim_rings, 16U);
  UB_REQUIRE_EQ(queues.value().rpc_submit_rings, 16U);
  UB_REQUIRE_EQ(queues.value().ready_rings, 8U);
  UB_REQUIRE_EQ(queues.value().spsc_ring_count, 64U);
}

TEST(ReferenceUnitRuntime, StartupSeparatesReadResultCapacityFromWrDepth) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 128;
  meta.dim = 4;
  meta.M = 4;
  meta.M0 = 8;
  meta.ef_search_default = 16;
  meta.num_shards = 2;
  hnsw_urma::RuntimeConfig config;
  config.topk = 1;
  config.expand_batch = 4;
  config.max_active_queries = 2;
  config.search_workers = {false, 2};
  config.distance_workers = {false, 1};
  config.read_progress_threads = {false, 1};
  config.rpc_progress_threads = {false, 1};
  config.cpu_layout = hnsw_urma::CpuLayout::kContiguous;
  config.read_jfs_depth = {false, 1};
  config.read_result_slots_per_search = {false, 31};
  const std::array<std::uint32_t, 8> cpus{0, 1, 2, 3, 4, 5, 6, 7};
  auto rejected = hnsw_urma::ResolveRuntimeConfig(config, meta, 2, cpus, 0);
  UB_REQUIRE_FALSE(rejected.ok());
  UB_REQUIRE_EQ(rejected.status().code(),
                hnsw_urma::StatusCode::kInvalidArgument);

  config.read_result_slots_per_search.value = 32;
  auto accepted = hnsw_urma::ResolveRuntimeConfig(config, meta, 2, cpus, 0);
  UB_REQUIRE_TRUE(accepted.ok());
  UB_REQUIRE_TRUE(accepted.value().queues.depths.read_submit > 0U);

  // Upper-layer stages can exceed M0 * expand_batch. Their result pool and
  // RPC task capacities must use the same maximum as the Pull adapter.
  meta.M = 40;
  UB_REQUIRE_FALSE(
      hnsw_urma::ResolveRuntimeConfig(config, meta, 2, cpus, 0).ok());
  config.read_result_slots_per_search = {true, 0};
  config.rpc_response_batch_max_bytes = {
      false, static_cast<std::uint32_t>(
                 hnsw_urma::DistanceBatchResponseBytesForShape(1, 32).value())};
  UB_REQUIRE_FALSE(
      hnsw_urma::ResolveRuntimeConfig(config, meta, 2, cpus, 0).ok());
  config.rpc_response_batch_max_bytes.value = static_cast<std::uint32_t>(
      hnsw_urma::DistanceBatchResponseBytesForShape(1, 40).value());
  auto upper = hnsw_urma::ResolveRuntimeConfig(config, meta, 2, cpus, 0);
  UB_REQUIRE_TRUE(upper.ok());
  UB_REQUIRE_EQ(upper.value().read_result_slots_per_search, 40U);
  UB_REQUIRE_EQ(upper.value().urma.read_jfs_depth, 1U);
}

TEST(ReferenceUnitRuntime, CountingNotificationPreservesMultipleWakeups) {
  hnsw_urma::CountingNotification notification;
  notification.Notify();
  notification.Notify();
  UB_REQUIRE_TRUE(notification.TryConsume());
  UB_REQUIRE_TRUE(notification.TryConsume());
  UB_REQUIRE_FALSE(notification.TryConsume());
}

TEST(ReferenceUnitRuntime, ProfileSchemaIsV7AndContainsSearchOwnedReadMetrics) {
  hnsw_urma::BenchmarkSummary summary;
  summary.query_count = 3;
  summary.succeeded = 2;
  summary.execution_latency.mean_ms = 10.0;
  summary.local_execution_latency.mean_ms = 4.0;
  summary.graph_wait_latency.mean_ms = 1.25;
  summary.upper_elapsed_latency.mean_ms = 2.0;
  summary.layer0_elapsed_latency.mean_ms = 8.0;
  summary.remote_finish_latency.mean_ms = 0.75;
  hnsw_urma::RuntimeProfileSnapshot profile;
  profile.read_batches_submitted = 9;
  profile.read_requests_posted = 27;
  profile.request_slot_scans = 17;
  profile.result_ready_publications = 4;
  profile.response_scan_hits = 3;
  profile.request_notify_observed_to_response_post_nanoseconds = 19;
  profile.request_notify_observed_to_response_post_samples = 5;
  profile.rpc_client_response_notify_to_receive_task_nanoseconds = 17;
  profile.rpc_client_response_notify_to_receive_task_samples = 5;
  profile.cluster_bidirectional_post_to_notify_observed_nanoseconds = 31;
  profile.cluster_bidirectional_post_to_notify_observed_samples = 5;
  profile.rpc_exceptional_retained_nanoseconds = 23;
  profile.rpc_client_response_to_stage_terminal_nanoseconds = 29;
  profile.rpc_post_requests_nanoseconds_per_thread = {11, 12};
  profile.distance_request_slot_scans_per_thread = {9, 8};
  profile.read_progress_iterations_per_thread = {7};
  profile.wait_event_counts = {2, 0, 0, 0};
  profile.wait_nanoseconds = {600, 0, 0, 0};
  profile.wait_publish_nanoseconds = {120, 0, 0, 0};
  profile.wait_first_completion_nanoseconds = {300, 0, 0, 0};
  profile.wait_completion_spread_nanoseconds = {180, 0, 0, 0};
  profile.wait_remote_operations = {4, 0, 0, 0};
  profile.wait_distance_values = {0, 0, 0, 0};
  profile.wait_remote_operations_high_watermark = {2, 0, 0, 0};
  profile.wait_distance_values_high_watermark = {0, 0, 0, 0};
  profile.wait_critical_lane_counts = {1, 1, 0, 0, 0, 0, 0, 0};
  profile.wait_histograms.assign(4U * 64U, 0);
  profile.wait_publish_histograms.assign(4U * 64U, 0);
  profile.wait_first_completion_histograms.assign(4U * 64U, 0);
  profile.wait_completion_spread_histograms.assign(4U * 64U, 0);
  profile.wait_histograms[9] = 2;
  profile.wait_publish_histograms[6] = 2;
  profile.wait_first_completion_histograms[8] = 2;
  profile.wait_completion_spread_histograms[7] = 2;
  const auto json =
      hnsw_urma::RenderProfileJson(summary, profile, "abc123", 0, 2);
  UB_REQUIRE_TRUE(json.find("\"schema_version\":7") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"query_timing_ms\"") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"graph_wait\":{\"mean_ms\":1.250000") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"search_stage_ms\"") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"local_phase_ms\"") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"remote_finish\":{\"mean_ms\":0.750000") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"wait_events\"") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"critical_lane_count\":2") != std::string::npos);
  UB_REQUIRE_TRUE(
      json.find("\"publish_to_first_completion\":{\"samples\":[2,0,0,0]") !=
      std::string::npos);
  UB_REQUIRE_TRUE(
      json.find("\"completion_spread\":{\"samples\":[2,0,0,0],"
                "\"total_ns\":[180,0,0,0],\"mean_ns\":[90.000000") !=
      std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"direct_slot\"") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"request_slot_scans\":17") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"request_slot_scans_per_worker\":[9,8]") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"post_requests_ns_per_thread\":[11,12]") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"exceptional_retained_ns\":23") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"response_to_stage_terminal_ns\":29") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(
      json.find("\"request_notify_observed_to_response_post_ns\":19") !=
      std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"response_notify_to_receive_task_ns\":17") !=
                  std::string::npos);
  UB_REQUIRE_TRUE(
      json.find("\"cluster_bidirectional_post_to_notify_observed_ns\":31") !=
      std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"read\"") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"batches_submitted\":9") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"requests_posted\":27") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("query_cache") == std::string::npos);
  UB_REQUIRE_TRUE(json.find("credit") == std::string::npos);
  UB_REQUIRE_TRUE(json.find("adjacency") == std::string::npos);
}

TEST(ReferenceUnitRuntime,
     DistributedBenchmarkPayloadRoundTripsDirectSlotProfile) {
  hnsw_urma::QueryBenchmarkRecord record;
  record.query_id = 7;
  record.latency_nanoseconds = 9000;
  record.success = true;
  hnsw_urma::RuntimeProfileSnapshot profile;
  profile.request_slot_hits = 13;
  profile.result_post_retries = 2;
  profile.request_notify_observed_to_response_post_nanoseconds = 37;
  profile.rpc_client_response_notify_to_receive_task_nanoseconds = 41;
  profile.cluster_bidirectional_post_to_notify_observed_nanoseconds = 43;
  profile.distance_request_tasks_per_thread = {4, 5};
  profile.wait_completion_spread_nanoseconds = {11, 22, 33, 44};
  profile.wait_completion_spread_histograms = {1, 2, 3, 4};
  auto encoded = hnsw_urma::EncodeDistributedBenchmarkPayload(
      std::span<const hnsw_urma::QueryBenchmarkRecord>(&record, 1), 1.5,
      profile);
  UB_REQUIRE_TRUE(encoded.ok());
  auto decoded = hnsw_urma::DecodeDistributedBenchmarkPayload(encoded.value());
  UB_REQUIRE_TRUE(decoded.ok());
  UB_REQUIRE_EQ(decoded.value().records.size(), 1U);
  UB_REQUIRE_EQ(decoded.value().records[0].query_id, 7U);
  UB_REQUIRE_EQ(decoded.value().profile.request_slot_hits, 13U);
  UB_REQUIRE_EQ(decoded.value().profile.result_post_retries, 2U);
  UB_REQUIRE_EQ(
      decoded.value()
          .profile.request_notify_observed_to_response_post_nanoseconds,
      37U);
  UB_REQUIRE_EQ(
      decoded.value()
          .profile.rpc_client_response_notify_to_receive_task_nanoseconds,
      41U);
  UB_REQUIRE_EQ(
      decoded.value()
          .profile.cluster_bidirectional_post_to_notify_observed_nanoseconds,
      43U);
  UB_REQUIRE_EQ(decoded.value().profile.distance_request_tasks_per_thread[1],
                5U);
  UB_REQUIRE_EQ(decoded.value().profile.wait_completion_spread_nanoseconds[3],
                44U);
  UB_REQUIRE_EQ(decoded.value().profile.wait_completion_spread_histograms[2],
                3U);
}

TEST(ReferenceUnitRuntime, RecallAtKCountsMissingResultsAsMisses) {
  const std::array<hnsw_urma::HeapEntry, 2> result{{{7, 1.0F}, {9, 2.0F}}};
  const std::array<hnsw_urma::node_id_t, 3> truth{7, 8, 9};
  UB_REQUIRE_EQ(hnsw_urma::RecallAtK(result, truth, 3), 2.0 / 3.0);
}

} // namespace

TEST(ReferenceUnitRuntime,
     RpcCapacityCountsLogicalShardsAndRequiresWholeStageResources) {
  using namespace hnsw_urma;
  const std::array<std::uint32_t, 5> shards{0, 1, 1, 2, 3};
  const std::array<std::uint32_t, 4> owners{kNoCommunicationOwner, 0, 1, 0};
  auto plan = BuildRpcOperationPlan(17, 4, 2, 8, 0, shards, owners, 4);
  UB_REQUIRE_TRUE(plan.ok());
  UB_REQUIRE_EQ(plan.value().slots_per_search, 20U); // ceil(17/4) * 4 shards.
  UB_REQUIRE_EQ(plan.value().total_slots, 80U);
  UB_REQUIRE_EQ(plan.value().search_event_capacity, 37U);
  UB_REQUIRE_EQ(plan.value().submit_depth, 20U);
  UB_REQUIRE_EQ(plan.value().minimum_submit_depth, 3U);
  UB_REQUIRE_TRUE(plan.value().pool_bytes >
                  80U * 8U * (sizeof(node_id_t) + sizeof(float)));
  UB_REQUIRE_FALSE(
      BuildRpcOperationPlan(17, 4, 2, 8, 0, shards, owners, 4, {false, 3})
          .ok());
  UB_REQUIRE_FALSE(
      BuildRpcOperationPlan(17, 4, 2, 8, 0, shards, owners, 4, {}, {false, 2})
          .ok());
  UB_REQUIRE_TRUE(
      BuildRpcOperationPlan(17, 4, 2, 8, 0, shards, owners, 4, {}, {false, 3})
          .ok());
  auto one = BuildRpcOperationPlan(1, 32, 2, 1, 0, shards, owners, 4);
  UB_REQUIRE_TRUE(one.ok());
  UB_REQUIRE_EQ(one.value().slots_per_search, 1U);
  UB_REQUIRE_EQ(one.value().search_event_capacity, 2U);
  UB_REQUIRE_FALSE(
      BuildRpcOperationPlan(UINT32_MAX, 1, 2, 8, 0, shards, owners, 4).ok());
  UB_REQUIRE_FALSE(BuildRpcOperationPlan(1, 4, 2, 8, 0, shards, owners, 4,
                                         {false, UINT32_MAX})
                       .ok());
}

TEST(ReferenceUnitRuntime,
     UnifiedQueuePlanHas32MpscAnd320SpscRingsWithExactCellStorage) {
  using namespace hnsw_urma;
  RuntimeQueueDepths depths{16, 24, 8, 8, 8, 16};
  auto plan = BuildRuntimeQueuePlan(32, 2, 4, depths).value();
  UB_REQUIRE_EQ(plan.mpsc_ring_count, 32U);
  UB_REQUIRE_EQ(plan.ready_rings, 32U);
  UB_REQUIRE_EQ(plan.spsc_ring_count, 320U);
  UB_REQUIRE_EQ(plan.deferred_ready_entries, 6U * 16U);
  UB_REQUIRE_TRUE(plan.ring_storage_bytes >=
                  32U * MpmcRing<SearchEvent>::StorageBytes(24).value());
  UB_REQUIRE_EQ(plan.queue_bytes_total, plan.ring_storage_bytes +
                                            plan.ring_metadata_bytes +
                                            plan.private_work_bytes);
  UB_REQUIRE_FALSE(BuildRuntimeQueuePlan(1, UINT32_MAX, 1, depths).ok());
}

TEST(ReferenceUnitRuntime, RpcPoolCliAndProfileSchemaExposeFinalNames) {
  using namespace hnsw_urma;
  const std::array<std::string_view, 7> args{
      "--index",   "test",
      "--dry-run", "--rpc-operation-slots-per-search",
      "7",         "--rpc-submit-depth",
      "3"};
  auto command = ParseBenchmarkCommand(args);
  UB_REQUIRE_TRUE(command.ok());
  UB_REQUIRE_EQ(command.value().config.rpc_operation_slots_per_search.value,
                7U);
  UB_REQUIRE_TRUE(BenchmarkHelp().find("rpc-operation-slots-per-search") !=
                  std::string::npos);
  RuntimeProfileSnapshot profile;
  profile.rpc_reclaim_events_published = 2;
  profile.rpc_pool_backpressure = 3;
  profile.rpc_exceptional_reclaimed = 4;
  profile.search_event_high_watermark = {5};
  profile.ready_producer_worker_counts = {0, 3, 4};
  auto payload = EncodeDistributedBenchmarkPayload({}, 0, profile).value();
  auto decoded = DecodeDistributedBenchmarkPayload(payload).value();
  UB_REQUIRE_EQ(decoded.profile.rpc_reclaim_events_published, 2U);
  UB_REQUIRE_EQ(decoded.profile.rpc_pool_backpressure, 3U);
  UB_REQUIRE_EQ(decoded.profile.search_event_high_watermark,
                std::vector<std::uint64_t>({5}));
  UB_REQUIRE_EQ(decoded.profile.ready_producer_worker_counts,
                std::vector<std::uint64_t>({0, 3, 4}));
  const auto json = RenderProfileJson({}, profile, "test", 0, 1);
  UB_REQUIRE_TRUE(json.find("\"pool_backpressure\":3") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("\"reclaim_published\":2") != std::string::npos);
  UB_REQUIRE_TRUE(json.find("completion_consumed") == std::string::npos);
  const auto csv = RenderProfileCsv({}, profile, "test", 0, 1);
  UB_REQUIRE_TRUE(csv.find("\n7,\"test\"") != std::string::npos);
  UB_REQUIRE_TRUE(csv.find("rpc_pool_backpressure") != std::string::npos);
}
