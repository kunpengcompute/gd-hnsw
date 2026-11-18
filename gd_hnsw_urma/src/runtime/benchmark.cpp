#include "hnsw_urma/runtime/benchmark.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <unordered_set>

namespace hnsw_urma {
namespace {

constexpr std::uint64_t kPayloadMagic = 0x344D42414D525548ULL;
constexpr std::uint64_t kPayloadVersion = 9;
constexpr std::size_t kTimingHistogramBuckets = 64;

#define SEARCH_PROFILE_FIELDS(X) \
  X(nodes_expanded) X(edges_scanned) X(visited_hits) X(heap_pushes) \
  X(heap_rejections) X(heap_prefilter_rejections) X(heap_suffix_rejections) \
  X(heap_full_pushes) X(upper_steps) X(upper_same_level_moves) \
  X(upper_level_drops) X(upper_empty_neighbor_level_drops) \
  X(upper_level_drop_runs) X(upper_max_consecutive_level_drops) \
  X(upper_level_drop_runs_length_1) X(upper_level_drop_runs_length_2) \
  X(upper_level_drop_runs_length_3) X(upper_level_drop_runs_length_4_plus) \
  X(upper_distance_candidate_values) X(upper_distance_visited_hits) \
  X(upper_distance_unique_values) X(upper_distance_fully_visited_steps) \
  X(upper_remote_candidate_values) X(upper_remote_visited_hits) \
  X(upper_remote_operations) X(upper_eliminated_remote_operations) \
  X(local_distance_count) X(suspend_count) X(resume_count) \
  X(routing_group_owner_writes) X(routing_local_offset_writes) \
  X(routing_layer0_sorted_values) X(routing_layer0_unsorted_values)

#define RECORD_TIMING_FIELDS(X) \
  X(queue_wait_nanoseconds) X(execution_nanoseconds) \
  X(upper_elapsed_nanoseconds) X(layer0_elapsed_nanoseconds) \
  X(local_execution_nanoseconds) X(graph_wait_nanoseconds) \
  X(distance_wait_nanoseconds) X(ready_queue_nanoseconds) \
  X(upper_search_nanoseconds) X(graph_plan_nanoseconds) \
  X(graph_publish_nanoseconds) X(graph_consume_nanoseconds) \
  X(neighbor_scan_visited_nanoseconds) X(route_group_sort_nanoseconds) \
  X(local_distance_nanoseconds) X(remote_prepare_publish_nanoseconds) \
  X(remote_finish_nanoseconds) X(commit_heap_nanoseconds) \
  X(controller_other_nanoseconds)

#define PROFILE_SCALAR_FIELDS(X) \
  X(rpc_reclaim_events_published) X(rpc_reclaim_events_consumed) \
  X(rpc_exceptional_slots_in_use) X(rpc_exceptional_slots_high_watermark) \
  X(read_batches_submitted) X(read_batches_failed) \
  X(read_batches_timed_out) X(read_batches_cancelled) \
  X(read_requests_posted) X(read_requests_completed) \
  X(read_bytes_completed) X(read_wr_completion_samples) \
  X(read_wr_completion_nanoseconds) X(read_cqe_failure_count) \
  X(pushdown_requests) \
  X(pushdown_responses) X(request_bytes) X(response_bytes) \
  X(admission_high_watermark) X(lane_affinity_hits) \
  X(lane_affinity_inserts) X(lane_affinity_replacements) \
  X(lane_candidate_capacity_rejections) X(read_exceptional_batches_high_watermark) \
  X(read_late_completion_count) X(rpc_quarantine_high_watermark) \
  X(rpc_detached_records) X(rpc_late_completion_count) \
  X(rpc_late_response_count) X(read_thread_count) X(rpc_thread_count) \
  X(request_jfs_per_rpc_lane) X(response_jfs_per_rpc_lane) \
  X(read_result_slots_high_watermark) \
  X(read_in_flight_high_watermark) X(read_native_descriptors_high_watermark) \
  X(read_post_calls) \
  X(read_post_list_wrs) \
  X(read_post_list_high_watermark) \
  X(read_submit_wait_nanoseconds) \
  X(read_reclaim_wait_nanoseconds) \
  X(read_result_allocation_retries) \
  X(read_result_registered_bytes) \
  X(rpc_submit_queue_high_watermark) X(rpc_operation_slots_in_use) \
  X(rpc_operation_slots_high_watermark) X(rpc_active_audit_passes) \
  X(rpc_active_audit_operations) X(rpc_pool_backpressure) \
  X(rpc_submit_backpressure) X(rpc_exceptional_reclaimed) \
  X(rpc_ticket_block_refills) X(logical_distance_tasks) \
  X(completed_distance_tasks) X(request_batches) X(request_batch_tasks) \
  X(request_batch_bytes) X(response_batches) X(response_batch_tasks) \
  X(response_batch_bytes) X(request_physical_writes) \
  X(response_physical_writes) X(request_completions) \
  X(response_completions) X(request_write_completion_nanoseconds) \
  X(response_write_completion_nanoseconds) \
  X(rpc_client_owner_queue_nanoseconds) X(rpc_client_owner_queue_samples) \
  X(rpc_client_owner_pending_nanoseconds) \
  X(rpc_client_request_prepare_nanoseconds) \
  X(rpc_client_request_post_wait_nanoseconds) \
  X(rpc_client_round_trip_nanoseconds) X(rpc_client_round_trip_samples) \
  X(rpc_client_response_notify_to_receive_nanoseconds) \
  X(rpc_client_response_notify_to_receive_samples) \
  X(rpc_client_response_notify_to_receive_task_nanoseconds) \
  X(rpc_client_response_notify_to_receive_task_samples) \
  X(cluster_bidirectional_post_to_notify_observed_nanoseconds) \
  X(cluster_bidirectional_post_to_notify_observed_samples) \
  X(rpc_client_response_received_to_task_terminal_nanoseconds) \
  X(rpc_client_response_received_to_task_terminal_samples) \
  X(rpc_exceptional_retained_nanoseconds) \
  X(rpc_client_response_to_stage_terminal_nanoseconds) \
  X(rpc_client_response_to_stage_terminal_samples) \
  X(rpc_client_response_decode_nanoseconds) \
  X(rpc_client_response_decode_samples) X(request_slot_scans) \
  X(request_slot_hits) X(request_slot_blocked_by_result_inflight) \
  X(server_request_batches) X(server_request_tasks) \
  X(server_distance_values) X(request_poll_hit_to_claim_nanoseconds) \
  X(request_poll_hit_to_claim_samples) X(request_decode_nanoseconds) \
  X(query_slot_store_nanoseconds) X(query_slot_load_nanoseconds) \
  X(distance_compute_nanoseconds) X(filter_and_response_encode_nanoseconds) \
  X(result_ready_publications) X(distance_busy_iterations) \
  X(distance_idle_iterations) X(result_slot_scans) X(result_slot_hits) \
  X(request_notify_observed_to_response_post_nanoseconds) \
  X(request_notify_observed_to_response_post_samples) \
  X(result_ready_to_post_nanoseconds) X(result_ready_to_post_samples) \
  X(result_post_retries) X(result_post_to_completion_nanoseconds) \
  X(result_post_to_completion_samples) X(response_scan_slots) \
  X(response_scan_hits) X(request_slots_in_use) \
  X(request_slots_high_watermark) X(request_post_free_slot_waits) \
  X(request_post_staging_unavailable) X(request_post_tx_unavailable) \
  X(request_post_provider_backpressure_batches) X(ready_events_published) \
  X(ready_events_consumed) X(ready_queue_high_watermark)


#define PROFILE_VECTOR_FIELDS(X) \
  X(rpc_slots_in_use_per_search) \
  X(rpc_slots_high_watermark_per_search) \
  X(rpc_pool_backpressure_per_search) \
  X(rpc_submit_backpressure_per_search) \
  X(rpc_audit_operations_per_owner) \
  X(ready_published_per_worker) \
  X(ready_consumed_per_worker) \
  X(ready_producer_worker_counts) \
  X(ready_converged_to_publish_ns_per_worker) \
  X(ready_publish_to_pop_ns_per_worker) \
  X(ready_pop_to_step_ns_per_worker) \
  X(ready_run_ns_per_worker) \
  X(search_event_depth) X(search_event_high_watermark) \
  X(read_owned_ranks_per_thread) X(rpc_owned_ranks_per_thread) \
  X(search_busy_nanoseconds_per_thread) X(search_idle_nanoseconds_per_thread) \
  X(search_iterations_per_thread) X(search_new_queries_per_thread) \
  X(search_ready_resumes_per_thread) X(search_retry_runs_per_thread) \
  X(distance_busy_nanoseconds_per_thread) \
  X(distance_idle_nanoseconds_per_thread) X(distance_iterations_per_thread) \
  X(read_busy_nanoseconds_per_thread) X(read_idle_nanoseconds_per_thread) \
  X(read_progress_iterations_per_thread) X(rpc_busy_nanoseconds_per_thread) \
  X(rpc_idle_nanoseconds_per_thread) X(rpc_progress_iterations_per_thread) \
  X(rpc_poll_completions_nanoseconds_per_thread) \
  X(rpc_scan_responses_nanoseconds_per_thread) \
  X(rpc_post_responses_nanoseconds_per_thread) \
  X(rpc_drain_client_submissions_nanoseconds_per_thread) \
  X(rpc_post_requests_nanoseconds_per_thread) \
  X(distance_request_slot_scans_per_thread) \
  X(distance_request_slot_hits_per_thread) \
  X(distance_request_batches_per_thread) \
  X(distance_request_tasks_per_thread) X(distance_values_per_thread) \
  X(wait_event_counts) X(wait_nanoseconds) X(wait_publish_nanoseconds) \
  X(wait_first_completion_nanoseconds) \
  X(wait_completion_spread_nanoseconds) X(wait_remote_operations) \
  X(wait_distance_values) X(wait_remote_operations_high_watermark) \
  X(wait_distance_values_high_watermark) X(wait_critical_lane_counts) \
  X(wait_histograms) X(wait_publish_histograms) \
  X(wait_first_completion_histograms) \
  X(wait_completion_spread_histograms)

double Percentile(std::vector<std::uint64_t> values, double percentile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(std::ceil(
      percentile * static_cast<double>(values.size()))) - 1U;
  return static_cast<double>(values[std::min(index, values.size() - 1U)]) /
         1.0e6;
}

LatencySummary MakeLatency(std::vector<std::uint64_t> values) {
  LatencySummary result;
  if (values.empty()) return result;
  long double total = 0;
  for (const auto value : values) total += static_cast<long double>(value);
  result.mean_ms = static_cast<double>(
      total / static_cast<long double>(values.size())) / 1.0e6;
  result.p50_ms = Percentile(values, 0.50);
  result.p95_ms = Percentile(values, 0.95);
  result.p99_ms = Percentile(std::move(values), 0.99);
  return result;
}

void AddSearchProfile(SearchProfile& total, const SearchProfile& value) {
  const auto previous_max = total.upper_max_consecutive_level_drops;
#define ADD_SEARCH(name) total.name += value.name;
  SEARCH_PROFILE_FIELDS(ADD_SEARCH)
#undef ADD_SEARCH
  total.upper_max_consecutive_level_drops = std::max(
      previous_max,
      value.upper_max_consecutive_level_drops);
}

std::string EscapeJson(std::string_view value) {
  std::string output;
  output.reserve(value.size());
  for (const char ch : value) {
    switch (ch) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default: output += ch; break;
    }
  }
  return output;
}

void AppendU64(std::vector<std::byte>& output, std::uint64_t value) {
  const auto offset = output.size();
  output.resize(offset + sizeof(value));
  std::memcpy(output.data() + offset, &value, sizeof(value));
}

class PayloadReader {
 public:
  explicit PayloadReader(std::span<const std::byte> input) : input_(input) {}
  Result<std::uint64_t> U64() {
    if (offset_ > input_.size() || sizeof(std::uint64_t) > input_.size() - offset_) {
      return Status(StatusCode::kInvalidArgument,
                    "distributed benchmark payload is truncated");
    }
    std::uint64_t value = 0;
    std::memcpy(&value, input_.data() + offset_, sizeof(value));
    offset_ += sizeof(value);
    return value;
  }
  [[nodiscard]] bool done() const noexcept { return offset_ == input_.size(); }

 private:
  std::span<const std::byte> input_;
  std::size_t offset_{0};
};

void AppendVector(std::vector<std::byte>& output,
                  const std::vector<std::uint64_t>& values) {
  AppendU64(output, values.size());
  for (const auto value : values) AppendU64(output, value);
}

Result<std::vector<std::uint64_t>> ReadVector(PayloadReader& reader) {
  auto size = reader.U64();
  if (!size.ok()) return size.status();
  if (size.value() > (1U << 20U)) {
    return Status(StatusCode::kOutOfRange,
                  "distributed profile vector is too large");
  }
  try {
    std::vector<std::uint64_t> result;
    result.reserve(static_cast<std::size_t>(size.value()));
    for (std::uint64_t index = 0; index < size.value(); ++index) {
      auto value = reader.U64();
      if (!value.ok()) return value.status();
      result.push_back(value.value());
    }
    return result;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to decode distributed profile vector");
  }
}

double Ratio(double part, double total) noexcept {
  return total == 0.0 ? 0.0 : part / total;
}

void RenderLatencyJson(std::ostringstream& output, std::string_view name,
                       const LatencySummary& latency, bool comma = true) {
  output << "\"" << name << "\":{\"mean_ms\":" << latency.mean_ms
         << ",\"p50_ms\":" << latency.p50_ms
         << ",\"p95_ms\":" << latency.p95_ms
         << ",\"p99_ms\":" << latency.p99_ms << "}";
  if (comma) output << ',';
}

void RenderU64Values(std::ostringstream& output,
                     std::span<const std::uint64_t> values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void RenderMeanValues(std::ostringstream& output,
                      std::span<const std::uint64_t> totals,
                      std::span<const std::uint64_t> counts) {
  output << '[';
  const auto size = std::min(totals.size(), counts.size());
  for (std::size_t index = 0; index < size; ++index) {
    if (index != 0) output << ',';
    output << (counts[index] == 0
                   ? 0.0
                   : static_cast<double>(totals[index]) /
                         static_cast<double>(counts[index]));
  }
  output << ']';
}

std::uint64_t HistogramPercentile(
    std::span<const std::uint64_t> histogram, std::size_t group,
    std::size_t buckets, std::uint32_t numerator,
    std::uint32_t denominator) noexcept {
  if (buckets == 0 || numerator == 0 || denominator == 0 ||
      numerator > denominator || group >= histogram.size() / buckets) {
    return 0;
  }
  const auto begin = group * buckets;
  std::uint64_t samples = 0;
  for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
    samples += histogram[begin + bucket];
  }
  if (samples == 0) return 0;
  const auto quotient = samples / denominator;
  const auto remainder = samples % denominator;
  const auto target = quotient * numerator +
                      (remainder * numerator + denominator - 1U) /
                          denominator;
  std::uint64_t cumulative = 0;
  for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
    cumulative += histogram[begin + bucket];
    if (cumulative >= target) {
      return bucket == 0 ? 0 : (std::uint64_t{1} << bucket) - 1U;
    }
  }
  return std::numeric_limits<std::uint64_t>::max();
}

void RenderHistogramPercentiles(
    std::ostringstream& output, std::span<const std::uint64_t> histogram,
    std::size_t groups, std::size_t buckets, std::uint32_t numerator,
    std::uint32_t denominator) {
  output << '[';
  for (std::size_t group = 0; group < groups; ++group) {
    if (group != 0) output << ',';
    output << HistogramPercentile(histogram, group, buckets, numerator,
                                  denominator);
  }
  output << ']';
}

void RenderEventTiming(std::ostringstream& output,
                       std::span<const std::uint64_t> totals,
                       std::span<const std::uint64_t> groups,
                       std::span<const std::uint64_t> histogram,
                       std::size_t buckets) {
  std::vector<std::uint64_t> samples(groups.size());
  for (std::size_t group = 0; group < groups.size(); ++group) {
    const auto begin = group * buckets;
    if (begin > histogram.size() || buckets > histogram.size() - begin) break;
    for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
      samples[group] += histogram[begin + bucket];
    }
  }
  output << "{\"samples\":";
  RenderU64Values(output, samples);
  output << ",\"total_ns\":";
  RenderU64Values(output, totals);
  output << ",\"mean_ns\":";
  RenderMeanValues(output, totals, samples);
  output << ",\"p50_ns\":";
  RenderHistogramPercentiles(output, histogram, groups.size(), buckets, 50,
                             100);
  output << ",\"p95_ns\":";
  RenderHistogramPercentiles(output, histogram, groups.size(), buckets, 95,
                             100);
  output << ",\"p99_ns\":";
  RenderHistogramPercentiles(output, histogram, groups.size(), buckets, 99,
                             100);
  output << '}';
}

void RenderCounterArray(std::ostringstream& output, std::string_view name,
                        std::span<const std::uint64_t> values,
                        bool comma = true) {
  output << '"' << name << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
  if (comma) output << ',';
}

}  // namespace

BenchmarkSummary SummarizeBenchmark(
    std::span<const QueryBenchmarkRecord> records, double elapsed_seconds) {
  BenchmarkSummary summary;
  summary.query_count = records.size();
  summary.elapsed_seconds = elapsed_seconds;
  std::vector<std::uint64_t> latency;
  std::vector<std::uint64_t> queue_wait;
  std::vector<std::uint64_t> execution;
  std::vector<std::uint64_t> upper_elapsed;
  std::vector<std::uint64_t> layer0_elapsed;
  std::vector<std::uint64_t> local_execution;
  std::vector<std::uint64_t> graph_wait;
  std::vector<std::uint64_t> distance_wait;
  std::vector<std::uint64_t> ready_queue;
  std::vector<std::uint64_t> upper_search;
  std::vector<std::uint64_t> graph_plan;
  std::vector<std::uint64_t> graph_publish;
  std::vector<std::uint64_t> graph_consume;
  std::vector<std::uint64_t> neighbor_scan;
  std::vector<std::uint64_t> route_sort;
  std::vector<std::uint64_t> local_distance;
  std::vector<std::uint64_t> remote_prepare;
  std::vector<std::uint64_t> remote_finish;
  std::vector<std::uint64_t> commit_heap;
  std::vector<std::uint64_t> controller_other;
  long double recall_total = 0;
  std::uint64_t recall_count = 0;
  for (const auto& record : records) {
    if (record.success) ++summary.succeeded; else ++summary.failed;
    latency.push_back(record.latency_nanoseconds);
    queue_wait.push_back(record.queue_wait_nanoseconds);
    execution.push_back(record.execution_nanoseconds);
    upper_elapsed.push_back(record.upper_elapsed_nanoseconds);
    layer0_elapsed.push_back(record.layer0_elapsed_nanoseconds);
    local_execution.push_back(record.local_execution_nanoseconds);
    graph_wait.push_back(record.graph_wait_nanoseconds);
    distance_wait.push_back(record.distance_wait_nanoseconds);
    ready_queue.push_back(record.ready_queue_nanoseconds);
    upper_search.push_back(record.upper_search_nanoseconds);
    graph_plan.push_back(record.graph_plan_nanoseconds);
    graph_publish.push_back(record.graph_publish_nanoseconds);
    graph_consume.push_back(record.graph_consume_nanoseconds);
    neighbor_scan.push_back(record.neighbor_scan_visited_nanoseconds);
    route_sort.push_back(record.route_group_sort_nanoseconds);
    local_distance.push_back(record.local_distance_nanoseconds);
    remote_prepare.push_back(record.remote_prepare_publish_nanoseconds);
    remote_finish.push_back(record.remote_finish_nanoseconds);
    commit_heap.push_back(record.commit_heap_nanoseconds);
    controller_other.push_back(record.controller_other_nanoseconds);
    AddSearchProfile(summary.search, record.search);
    if (record.recall.has_value()) {
      recall_total += *record.recall;
      ++recall_count;
    }
  }
  const auto overall = MakeLatency(std::move(latency));
  summary.latency_mean_ms = overall.mean_ms;
  summary.latency_p50_ms = overall.p50_ms;
  summary.latency_p95_ms = overall.p95_ms;
  summary.latency_p99_ms = overall.p99_ms;
  summary.queue_wait_latency = MakeLatency(std::move(queue_wait));
  summary.execution_latency = MakeLatency(std::move(execution));
  summary.upper_elapsed_latency = MakeLatency(std::move(upper_elapsed));
  summary.layer0_elapsed_latency = MakeLatency(std::move(layer0_elapsed));
  summary.local_execution_latency = MakeLatency(std::move(local_execution));
  summary.graph_wait_latency = MakeLatency(std::move(graph_wait));
  summary.distance_wait_latency = MakeLatency(std::move(distance_wait));
  summary.ready_queue_latency = MakeLatency(std::move(ready_queue));
  summary.upper_search_latency = MakeLatency(std::move(upper_search));
  summary.graph_plan_latency = MakeLatency(std::move(graph_plan));
  summary.graph_publish_latency = MakeLatency(std::move(graph_publish));
  summary.graph_consume_latency = MakeLatency(std::move(graph_consume));
  summary.neighbor_scan_visited_latency = MakeLatency(std::move(neighbor_scan));
  summary.route_group_sort_latency = MakeLatency(std::move(route_sort));
  summary.local_distance_latency = MakeLatency(std::move(local_distance));
  summary.remote_prepare_publish_latency = MakeLatency(std::move(remote_prepare));
  summary.remote_finish_latency = MakeLatency(std::move(remote_finish));
  summary.commit_heap_latency = MakeLatency(std::move(commit_heap));
  summary.controller_other_latency = MakeLatency(std::move(controller_other));
  if (recall_count != 0) {
    summary.recall_mean = static_cast<double>(
        recall_total / static_cast<long double>(recall_count));
  }
  if (elapsed_seconds > 0.0) {
    summary.qps = static_cast<double>(summary.succeeded) / elapsed_seconds;
  }
  return summary;
}

double RecallAtK(std::span<const HeapEntry> result,
                 std::span<const node_id_t> truth,
                 std::uint32_t topk) noexcept {
  if (topk == 0) return 0.0;
  const auto result_count = std::min<std::size_t>(result.size(), topk);
  const auto truth_count = std::min<std::size_t>(truth.size(), topk);
  std::uint32_t matches = 0;
  for (std::size_t index = 0; index < result_count; ++index) {
    if (result[index].id == INVALID_NODE_ID) continue;
    bool duplicate = false;
    for (std::size_t prior = 0; prior < index; ++prior) {
      duplicate |= result[prior].id == result[index].id;
    }
    const auto truth_end = truth.begin() +
        static_cast<std::ptrdiff_t>(truth_count);
    if (!duplicate && std::find(truth.begin(), truth_end,
                                result[index].id) != truth_end) {
      ++matches;
    }
  }
  return static_cast<double>(matches) / static_cast<double>(topk);
}

std::string RenderBenchmarkSummary(const BenchmarkSummary& summary,
                                   OutputFormat format) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6);
  if (format == OutputFormat::kJson) {
    output << "{\"query_count\":" << summary.query_count
           << ",\"succeeded\":" << summary.succeeded
           << ",\"failed\":" << summary.failed
           << ",\"elapsed_seconds\":" << summary.elapsed_seconds
           << ",\"qps\":" << summary.qps
           << ",\"latency_ms\":{\"mean\":" << summary.latency_mean_ms
           << ",\"p50\":" << summary.latency_p50_ms
           << ",\"p95\":" << summary.latency_p95_ms
           << ",\"p99\":" << summary.latency_p99_ms << '}';
    if (summary.recall_mean.has_value()) {
      output << ",\"recall_mean\":" << *summary.recall_mean;
    }
    output << "}\n";
  } else if (format == OutputFormat::kCsv) {
    output << "query_count,succeeded,failed,elapsed_seconds,qps,latency_mean_ms,"
              "latency_p50_ms,latency_p95_ms,latency_p99_ms,recall_mean\n"
           << summary.query_count << ',' << summary.succeeded << ','
           << summary.failed << ',' << summary.elapsed_seconds << ','
           << summary.qps << ',' << summary.latency_mean_ms << ','
           << summary.latency_p50_ms << ',' << summary.latency_p95_ms << ','
           << summary.latency_p99_ms << ',';
    if (summary.recall_mean.has_value()) output << *summary.recall_mean;
    output << '\n';
  } else {
    output << "queries=" << summary.query_count
           << " succeeded=" << summary.succeeded
           << " failed=" << summary.failed
           << " elapsed_s=" << summary.elapsed_seconds
           << " qps=" << summary.qps
           << " latency_ms(mean/p50/p95/p99)=" << summary.latency_mean_ms
           << '/' << summary.latency_p50_ms << '/' << summary.latency_p95_ms
           << '/' << summary.latency_p99_ms;
    if (summary.recall_mean.has_value()) {
      output << " recall=" << *summary.recall_mean;
    }
    output << '\n';
  }
  return output.str();
}

std::string RenderProfileJson(const BenchmarkSummary& summary,
                              const RuntimeProfileSnapshot& profile,
                              std::string_view git_commit, std::uint32_t rank,
                              std::uint32_t world_size) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "{\n  \"schema_version\":7,\n  \"git_commit\":\""
         << EscapeJson(git_commit) << "\",\n  \"rank\":" << rank
         << ",\n  \"world_size\":" << world_size
         << ",\n  \"summary\":{\"query_count\":" << summary.query_count
         << ",\"succeeded\":" << summary.succeeded
         << ",\"failed\":" << summary.failed
         << ",\"elapsed_seconds\":" << summary.elapsed_seconds
         << ",\"qps\":" << summary.qps << "},\n  \"latency\":{";
  RenderLatencyJson(output, "end_to_end", {summary.latency_mean_ms,
                    summary.latency_p50_ms, summary.latency_p95_ms,
                    summary.latency_p99_ms});
  RenderLatencyJson(output, "queue_wait", summary.queue_wait_latency);
  RenderLatencyJson(output, "execution", summary.execution_latency, false);
  output << "},\n  \"query_timing_ms\":{";
  RenderLatencyJson(output, "execution", summary.execution_latency);
  RenderLatencyJson(output, "local_execution",
                    summary.local_execution_latency);
  RenderLatencyJson(output, "graph_wait", summary.graph_wait_latency);
  RenderLatencyJson(output, "distance_wait", summary.distance_wait_latency);
  RenderLatencyJson(output, "ready_queue", summary.ready_queue_latency);
  output << "\"mean_ratio_of_execution\":{\"local_execution\":"
         << Ratio(summary.local_execution_latency.mean_ms,
                  summary.execution_latency.mean_ms)
         << ",\"graph_wait\":"
         << Ratio(summary.graph_wait_latency.mean_ms,
                  summary.execution_latency.mean_ms)
         << ",\"distance_wait\":"
         << Ratio(summary.distance_wait_latency.mean_ms,
                  summary.execution_latency.mean_ms)
         << ",\"ready_queue\":"
         << Ratio(summary.ready_queue_latency.mean_ms,
                  summary.execution_latency.mean_ms)
         << "}},\n  \"search_stage_ms\":{";
  RenderLatencyJson(output, "upper", summary.upper_elapsed_latency);
  RenderLatencyJson(output, "layer0", summary.layer0_elapsed_latency);
  const auto stage_mean_ms = summary.upper_elapsed_latency.mean_ms +
                             summary.layer0_elapsed_latency.mean_ms;
  output << "\"mean_ratio_of_stage_total\":{\"upper\":"
         << Ratio(summary.upper_elapsed_latency.mean_ms, stage_mean_ms)
         << ",\"layer0\":"
         << Ratio(summary.layer0_elapsed_latency.mean_ms, stage_mean_ms)
         << "},\"mean_ratio_of_execution\":{\"upper\":"
         << Ratio(summary.upper_elapsed_latency.mean_ms,
                  summary.execution_latency.mean_ms)
         << ",\"layer0\":"
         << Ratio(summary.layer0_elapsed_latency.mean_ms,
                  summary.execution_latency.mean_ms)
         << ",\"accounted\":"
         << Ratio(stage_mean_ms, summary.execution_latency.mean_ms)
         << "}},\n  \"local_phase_ms\":{";
  RenderLatencyJson(output, "upper_search", summary.upper_search_latency);
  RenderLatencyJson(output, "graph_plan", summary.graph_plan_latency);
  RenderLatencyJson(output, "graph_publish", summary.graph_publish_latency);
  RenderLatencyJson(output, "graph_consume", summary.graph_consume_latency);
  RenderLatencyJson(output, "neighbor_scan_visited",
                    summary.neighbor_scan_visited_latency);
  RenderLatencyJson(output, "route_group_sort",
                    summary.route_group_sort_latency);
  RenderLatencyJson(output, "local_distance",
                    summary.local_distance_latency);
  RenderLatencyJson(output, "remote_prepare_publish",
                    summary.remote_prepare_publish_latency);
  RenderLatencyJson(output, "remote_finish", summary.remote_finish_latency);
  RenderLatencyJson(output, "commit_heap", summary.commit_heap_latency);
  RenderLatencyJson(output, "controller_other",
                    summary.controller_other_latency);
  const auto local_phase_accounted_ms =
      summary.upper_search_latency.mean_ms +
      summary.graph_plan_latency.mean_ms +
      summary.graph_publish_latency.mean_ms +
      summary.graph_consume_latency.mean_ms +
      summary.neighbor_scan_visited_latency.mean_ms +
      summary.route_group_sort_latency.mean_ms +
      summary.local_distance_latency.mean_ms +
      summary.remote_prepare_publish_latency.mean_ms +
      summary.remote_finish_latency.mean_ms +
      summary.commit_heap_latency.mean_ms +
      summary.controller_other_latency.mean_ms;
  output << "\"mean_ratio_of_local_execution\":{\"upper_search\":"
         << Ratio(summary.upper_search_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"graph_plan\":"
         << Ratio(summary.graph_plan_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"graph_publish\":"
         << Ratio(summary.graph_publish_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"graph_consume\":"
         << Ratio(summary.graph_consume_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"neighbor_scan_visited\":"
         << Ratio(summary.neighbor_scan_visited_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"route_group_sort\":"
         << Ratio(summary.route_group_sort_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"local_distance\":"
         << Ratio(summary.local_distance_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"remote_prepare_publish\":"
         << Ratio(summary.remote_prepare_publish_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"remote_finish\":"
         << Ratio(summary.remote_finish_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"commit_heap\":"
         << Ratio(summary.commit_heap_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"controller_other\":"
         << Ratio(summary.controller_other_latency.mean_ms,
                  summary.local_execution_latency.mean_ms)
         << ",\"accounted\":"
         << Ratio(local_phase_accounted_ms,
                  summary.local_execution_latency.mean_ms)
         << "}},\n  \"wait_events\":{\"kinds\":"
            "[\"graph\",\"entry_distance\",\"upper_distance\","
            "\"layer0_distance\"],\"critical_lane_count\":";
  const auto wait_kind_count = profile.wait_event_counts.size();
  const auto critical_lane_count =
      wait_kind_count == 0
          ? 0U
          : profile.wait_critical_lane_counts.size() / wait_kind_count;
  output << critical_lane_count << ",\"count\":";
  RenderU64Values(output, profile.wait_event_counts);
  output << ",\"remote_operations_mean\":";
  RenderMeanValues(output, profile.wait_remote_operations,
                   profile.wait_event_counts);
  output << ",\"remote_operations_high_watermark\":";
  RenderU64Values(output, profile.wait_remote_operations_high_watermark);
  output << ",\"distance_values_mean\":";
  RenderMeanValues(output, profile.wait_distance_values,
                   profile.wait_event_counts);
  output << ",\"distance_values_high_watermark\":";
  RenderU64Values(output, profile.wait_distance_values_high_watermark);
  output << ",\"critical_lane_counts\":";
  RenderU64Values(output, profile.wait_critical_lane_counts);
  output << ",\"total_wait\":";
  RenderEventTiming(output, profile.wait_nanoseconds,
                    profile.wait_event_counts, profile.wait_histograms,
                    kTimingHistogramBuckets);
  output << ",\"publish\":";
  RenderEventTiming(output, profile.wait_publish_nanoseconds,
                    profile.wait_event_counts,
                    profile.wait_publish_histograms,
                    kTimingHistogramBuckets);
  output << ",\"publish_to_first_completion\":";
  RenderEventTiming(output, profile.wait_first_completion_nanoseconds,
                    profile.wait_event_counts,
                    profile.wait_first_completion_histograms,
                    kTimingHistogramBuckets);
  output << ",\"completion_spread\":";
  RenderEventTiming(output, profile.wait_completion_spread_nanoseconds,
                    profile.wait_event_counts,
                    profile.wait_completion_spread_histograms,
                    kTimingHistogramBuckets);
  output << "},\n  \"direct_slot\":{"
         << "\"request_slot_scans\":" << profile.request_slot_scans
         << ",\"request_slot_hits\":" << profile.request_slot_hits
         << ",\"request_slot_blocked_by_result_inflight\":"
         << profile.request_slot_blocked_by_result_inflight
         << ",\"request_batches\":" << profile.server_request_batches
         << ",\"request_tasks\":" << profile.server_request_tasks
         << ",\"distance_values\":" << profile.server_distance_values
         << ",\"request_poll_hit_to_claim_ns\":"
         << profile.request_poll_hit_to_claim_nanoseconds
         << ",\"request_poll_hit_to_claim_samples\":"
         << profile.request_poll_hit_to_claim_samples
         << ",\"request_decode_ns\":" << profile.request_decode_nanoseconds
         << ",\"query_slot_store_ns\":" << profile.query_slot_store_nanoseconds
         << ",\"query_slot_load_ns\":" << profile.query_slot_load_nanoseconds
         << ",\"distance_compute_ns\":" << profile.distance_compute_nanoseconds
         << ",\"filter_and_response_encode_ns\":"
         << profile.filter_and_response_encode_nanoseconds
         << ",\"result_ready_publications\":"
         << profile.result_ready_publications
         << ",\"result_slot_scans\":" << profile.result_slot_scans
         << ",\"result_slot_hits\":" << profile.result_slot_hits
         << ",\"request_notify_observed_to_response_post_ns\":"
         << profile.request_notify_observed_to_response_post_nanoseconds
         << ",\"request_notify_observed_to_response_post_samples\":"
         << profile.request_notify_observed_to_response_post_samples
         << ",\"result_ready_to_post_ns\":"
         << profile.result_ready_to_post_nanoseconds
         << ",\"result_ready_to_post_samples\":"
         << profile.result_ready_to_post_samples
         << ",\"result_post_retries\":" << profile.result_post_retries
         << ",\"result_post_to_completion_ns\":"
         << profile.result_post_to_completion_nanoseconds
         << ",\"result_post_to_completion_samples\":"
         << profile.result_post_to_completion_samples
         << ",\"request_slots_in_use\":" << profile.request_slots_in_use
         << ",\"request_slots_high_watermark\":"
         << profile.request_slots_high_watermark
         << ",\"distance_busy_iterations\":"
         << profile.distance_busy_iterations
         << ",\"distance_idle_iterations\":"
         << profile.distance_idle_iterations << ',';
  RenderCounterArray(output, "request_slot_scans_per_worker",
                     profile.distance_request_slot_scans_per_thread);
  RenderCounterArray(output, "request_slot_hits_per_worker",
                     profile.distance_request_slot_hits_per_thread);
  RenderCounterArray(output, "request_batches_per_worker",
                     profile.distance_request_batches_per_thread);
  RenderCounterArray(output, "request_tasks_per_worker",
                     profile.distance_request_tasks_per_thread);
  RenderCounterArray(output, "distance_values_per_worker",
                     profile.distance_values_per_thread);
  RenderCounterArray(output, "busy_ns_per_worker",
                     profile.distance_busy_nanoseconds_per_thread);
  RenderCounterArray(output, "idle_ns_per_worker",
                     profile.distance_idle_nanoseconds_per_thread);
  RenderCounterArray(output, "iterations_per_worker",
                     profile.distance_iterations_per_thread, false);
  output << "},\n  \"rpc\":{\"logical_tasks\":"
         << profile.logical_distance_tasks
          << ",\"completed_tasks\":" << profile.completed_distance_tasks
         << ",\"request_batches\":" << profile.request_batches
         << ",\"request_batch_tasks\":" << profile.request_batch_tasks
         << ",\"request_batch_bytes\":" << profile.request_batch_bytes
         << ",\"response_batches\":" << profile.response_batches
         << ",\"response_batch_tasks\":" << profile.response_batch_tasks
         << ",\"response_batch_bytes\":" << profile.response_batch_bytes
          << ",\"request_physical_writes\":"
          << profile.request_physical_writes
          << ",\"response_physical_writes\":"
          << profile.response_physical_writes
         << ",\"request_completions\":" << profile.request_completions
         << ",\"response_completions\":" << profile.response_completions
         << ",\"request_write_completion_ns\":"
         << profile.request_write_completion_nanoseconds
         << ",\"response_write_completion_ns\":"
         << profile.response_write_completion_nanoseconds
         << ",\"client_owner_queue_ns\":"
         << profile.rpc_client_owner_queue_nanoseconds
         << ",\"client_owner_queue_samples\":"
         << profile.rpc_client_owner_queue_samples
         << ",\"client_owner_pending_ns\":"
         << profile.rpc_client_owner_pending_nanoseconds
         << ",\"client_request_prepare_ns\":"
         << profile.rpc_client_request_prepare_nanoseconds
         << ",\"client_request_post_wait_ns\":"
         << profile.rpc_client_request_post_wait_nanoseconds
          << ",\"round_trip_ns\":" << profile.rpc_client_round_trip_nanoseconds
          << ",\"round_trip_samples\":" << profile.rpc_client_round_trip_samples
         << ",\"response_notify_to_receive_ns\":"
         << profile.rpc_client_response_notify_to_receive_nanoseconds
         << ",\"response_notify_to_receive_samples\":"
         << profile.rpc_client_response_notify_to_receive_samples
         << ",\"response_notify_to_receive_task_ns\":"
         << profile.rpc_client_response_notify_to_receive_task_nanoseconds
         << ",\"response_notify_to_receive_task_samples\":"
         << profile.rpc_client_response_notify_to_receive_task_samples
         << ",\"cluster_bidirectional_post_to_notify_observed_ns\":"
         << profile.cluster_bidirectional_post_to_notify_observed_nanoseconds
         << ",\"cluster_bidirectional_post_to_notify_observed_samples\":"
         << profile.cluster_bidirectional_post_to_notify_observed_samples
         << ",\"response_decode_ns\":"
         << profile.rpc_client_response_decode_nanoseconds
         << ",\"response_decode_samples\":"
         << profile.rpc_client_response_decode_samples
         << ",\"response_received_to_task_terminal_ns\":"
         << profile.rpc_client_response_received_to_task_terminal_nanoseconds
         << ",\"response_received_to_task_terminal_samples\":"
         << profile.rpc_client_response_received_to_task_terminal_samples
         << ",\"exceptional_retained_ns\":"
         << profile
                .rpc_exceptional_retained_nanoseconds
         << ",\"response_to_stage_terminal_ns\":"
         << profile
                .rpc_client_response_to_stage_terminal_nanoseconds
         << ",\"response_to_stage_terminal_samples\":"
         << profile
                .rpc_client_response_to_stage_terminal_samples
         << ",\"response_scan_slots\":" << profile.response_scan_slots
         << ",\"response_scan_hits\":" << profile.response_scan_hits
         << ",\"request_post_free_slot_waits\":"
         << profile.request_post_free_slot_waits
         << ",\"request_post_staging_unavailable\":"
         << profile.request_post_staging_unavailable
         << ",\"request_post_tx_unavailable\":"
         << profile.request_post_tx_unavailable
         << ",\"request_post_provider_backpressure_batches\":"
         << profile.request_post_provider_backpressure_batches << ',';
  RenderCounterArray(output, "poll_completions_ns_per_thread",
                     profile.rpc_poll_completions_nanoseconds_per_thread);
  RenderCounterArray(output, "scan_responses_ns_per_thread",
                     profile.rpc_scan_responses_nanoseconds_per_thread);
  RenderCounterArray(output, "post_responses_ns_per_thread",
                     profile.rpc_post_responses_nanoseconds_per_thread);
  RenderCounterArray(output, "drain_submissions_ns_per_thread",
                     profile.rpc_drain_client_submissions_nanoseconds_per_thread);
  RenderCounterArray(output, "post_requests_ns_per_thread",
                     profile.rpc_post_requests_nanoseconds_per_thread);
  RenderCounterArray(output, "busy_ns_per_thread",
                     profile.rpc_busy_nanoseconds_per_thread);
  RenderCounterArray(output, "idle_ns_per_thread",
                     profile.rpc_idle_nanoseconds_per_thread);
  RenderCounterArray(output, "iterations_per_thread",
                     profile.rpc_progress_iterations_per_thread, false);
  output << "},\n  \"search\":{\"nodes_expanded\":"
         << summary.search.nodes_expanded << ",\"edges_scanned\":"
         << summary.search.edges_scanned << ',';
  RenderCounterArray(output, "busy_ns_per_thread",
                     profile.search_busy_nanoseconds_per_thread);
  RenderCounterArray(output, "idle_ns_per_thread",
                     profile.search_idle_nanoseconds_per_thread);
  RenderCounterArray(output, "iterations_per_thread",
                     profile.search_iterations_per_thread, false);
  output << "},\n  \"read\":{\"batches_submitted\":"
         << profile.read_batches_submitted << ",\"batches_failed\":"
         << profile.read_batches_failed << ",\"batches_timed_out\":"
         << profile.read_batches_timed_out << ",\"batches_cancelled\":"
         << profile.read_batches_cancelled << ",\"requests_posted\":"
         << profile.read_requests_posted << ",\"requests_completed\":"
         << profile.read_requests_completed << ",\"bytes_completed\":"
         << profile.read_bytes_completed << ",\"cqe_failure_count\":"
         << profile.read_cqe_failure_count << ',';
  output << "\"native_descriptors_high_watermark\":" << profile.read_native_descriptors_high_watermark << ',';
  output << "\"in_flight_high_watermark\":" << profile.read_in_flight_high_watermark << ',';
  output << "\"post_calls\":" << profile.read_post_calls << ',';
  output << "\"post_list_wrs\":" << profile.read_post_list_wrs << ',';
  output << "\"post_list_high_watermark\":" << profile.read_post_list_high_watermark << ',';
  output << "\"submit_wait_nanoseconds\":" << profile.read_submit_wait_nanoseconds << ',';
  output << "\"reclaim_wait_nanoseconds\":" << profile.read_reclaim_wait_nanoseconds << ',';
  output << "\"result_allocation_retries\":" << profile.read_result_allocation_retries << ',';
  output << "\"result_registered_bytes\":" << profile.read_result_registered_bytes << ',';
  output << "\"result_slots_high_watermark\":" << profile.read_result_slots_high_watermark << ',';
  output << "\"exceptional_batches_high_watermark\":" << profile.read_exceptional_batches_high_watermark << ',';
  RenderCounterArray(output, "busy_ns_per_thread",
                     profile.read_busy_nanoseconds_per_thread);
  RenderCounterArray(output, "idle_ns_per_thread",
                     profile.read_idle_nanoseconds_per_thread);
  RenderCounterArray(output, "iterations_per_thread",
                     profile.read_progress_iterations_per_thread, false);
  output << "},\n  \"search_events\":{"
         << "\"ready_published\":" << profile.ready_events_published
         << ",\"ready_consumed\":" << profile.ready_events_consumed
         << ",\"reclaim_published\":" << profile.rpc_reclaim_events_published
         << ",\"reclaim_consumed\":" << profile.rpc_reclaim_events_consumed << ',';
  RenderCounterArray(output, "depth_per_search", profile.search_event_depth);
  RenderCounterArray(output, "high_watermark_per_search", profile.search_event_high_watermark);
  RenderCounterArray(output, "ready_published_per_worker", profile.ready_published_per_worker);
  RenderCounterArray(output, "ready_consumed_per_worker", profile.ready_consumed_per_worker);
  RenderCounterArray(output, "ready_producer_worker_counts", profile.ready_producer_worker_counts);
  RenderCounterArray(output, "ready_converged_to_publish_ns_per_worker", profile.ready_converged_to_publish_ns_per_worker);
  RenderCounterArray(output, "ready_publish_to_pop_ns_per_worker", profile.ready_publish_to_pop_ns_per_worker);
  RenderCounterArray(output, "ready_pop_to_step_ns_per_worker", profile.ready_pop_to_step_ns_per_worker);
  RenderCounterArray(output, "ready_run_ns_per_worker", profile.ready_run_ns_per_worker, false);
  output << "},\n  \"rpc_pool\":{"
         << "\"slots_in_use\":" << profile.rpc_operation_slots_in_use
         << ",\"slots_high_watermark\":" << profile.rpc_operation_slots_high_watermark
         << ",\"pool_backpressure\":" << profile.rpc_pool_backpressure
         << ",\"submit_backpressure\":" << profile.rpc_submit_backpressure
         << ",\"exceptional_slots_in_use\":" << profile.rpc_exceptional_slots_in_use
         << ",\"exceptional_slots_high_watermark\":" << profile.rpc_exceptional_slots_high_watermark
         << ",\"exceptional_reclaimed\":" << profile.rpc_exceptional_reclaimed
         << ",\"exceptional_retained_ns\":" << profile.rpc_exceptional_retained_nanoseconds
         << ",\"audit_passes\":" << profile.rpc_active_audit_passes
         << ",\"audit_operations\":" << profile.rpc_active_audit_operations
         << ',';
  RenderCounterArray(output, "slots_in_use_per_search", profile.rpc_slots_in_use_per_search, true);
  RenderCounterArray(output, "slots_high_watermark_per_search", profile.rpc_slots_high_watermark_per_search, true);
  RenderCounterArray(output, "pool_backpressure_per_search", profile.rpc_pool_backpressure_per_search, true);
  RenderCounterArray(output, "submit_backpressure_per_search", profile.rpc_submit_backpressure_per_search, true);
  RenderCounterArray(output, "audit_operations_per_owner", profile.rpc_audit_operations_per_owner, false);
  output << "}\n}\n";
  return output.str();
}

std::string RenderProfileCsv(const BenchmarkSummary& summary,
                             const RuntimeProfileSnapshot& profile,
                             std::string_view git_commit, std::uint32_t rank,
                             std::uint32_t world_size) {
  std::ostringstream output;
  output << "schema_version,git_commit,rank,world_size,query_count,qps,"
            "request_slot_scans,request_slot_hits,server_request_batches,"
            "server_request_tasks,server_distance_values,result_slot_scans,"
            "result_slot_hits,result_post_retries,request_slots_high_watermark,"
            "query_ready,reclaim_events,rpc_pool_backpressure,rpc_submit_backpressure,"
            "rpc_exceptional_reclaimed,rpc_exceptional_retained_ns,rpc_audit_operations\n"
         << "7,\"" << EscapeJson(git_commit) << "\"," << rank << ','
         << world_size << ',' << summary.query_count << ',' << summary.qps
         << ',' << profile.request_slot_scans << ',' << profile.request_slot_hits
         << ',' << profile.server_request_batches << ','
         << profile.server_request_tasks << ',' << profile.server_distance_values
         << ',' << profile.result_slot_scans << ',' << profile.result_slot_hits
         << ',' << profile.result_post_retries << ','
         << profile.request_slots_high_watermark << ',' << profile.ready_events_published
         << ',' << profile.rpc_reclaim_events_published << ',' << profile.rpc_pool_backpressure
         << ',' << profile.rpc_submit_backpressure << ',' << profile.rpc_exceptional_reclaimed
         << ',' << profile.rpc_exceptional_retained_nanoseconds << ',' << profile.rpc_active_audit_operations << '\n';
  return output.str();
}

Result<std::vector<std::byte>> EncodeDistributedBenchmarkPayload(
    std::span<const QueryBenchmarkRecord> records, double elapsed_seconds,
    const RuntimeProfileSnapshot& profile) {
  if (!std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0 ||
      records.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kInvalidArgument,
                  "distributed benchmark payload inputs are invalid");
  }
  try {
    std::vector<std::byte> output;
    output.reserve(64U + records.size() * 512U);
    AppendU64(output, kPayloadMagic);
    AppendU64(output, kPayloadVersion);
    AppendU64(output, records.size());
    AppendU64(output, std::bit_cast<std::uint64_t>(elapsed_seconds));
    for (const auto& record : records) {
      AppendU64(output, record.query_id);
      AppendU64(output, record.latency_nanoseconds);
      AppendU64(output, record.recall.has_value() ? 1U : 0U);
      AppendU64(output, record.recall.has_value()
                            ? std::bit_cast<std::uint64_t>(*record.recall)
                            : 0U);
      AppendU64(output, record.success ? 1U : 0U);
#define APPEND_SEARCH(name) AppendU64(output, record.search.name);
      SEARCH_PROFILE_FIELDS(APPEND_SEARCH)
#undef APPEND_SEARCH
#define APPEND_TIMING(name) AppendU64(output, record.name);
      RECORD_TIMING_FIELDS(APPEND_TIMING)
#undef APPEND_TIMING
    }
#define APPEND_PROFILE(name) AppendU64(output, profile.name);
    PROFILE_SCALAR_FIELDS(APPEND_PROFILE)
#undef APPEND_PROFILE
#define APPEND_PROFILE_VECTOR(name) AppendVector(output, profile.name);
    PROFILE_VECTOR_FIELDS(APPEND_PROFILE_VECTOR)
#undef APPEND_PROFILE_VECTOR
    return output;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to encode distributed benchmark payload");
  }
}

Result<DistributedBenchmarkPayload> DecodeDistributedBenchmarkPayload(
    std::span<const std::byte> input) {
  PayloadReader reader(input);
  auto magic = reader.U64();
  auto version = reader.U64();
  auto count = reader.U64();
  auto elapsed = reader.U64();
  if (!magic.ok() || !version.ok() || !count.ok() || !elapsed.ok()) {
    return Status(StatusCode::kInvalidArgument,
                  "distributed benchmark payload header is truncated");
  }
  if (magic.value() != kPayloadMagic || version.value() != kPayloadVersion) {
    return Status(StatusCode::kProtocolError,
                  "distributed benchmark payload version is incompatible");
  }
  if (count.value() > (1U << 30U)) {
    return Status(StatusCode::kOutOfRange,
                  "distributed benchmark record count is too large");
  }
  DistributedBenchmarkPayload payload;
  payload.elapsed_seconds = std::bit_cast<double>(elapsed.value());
  if (!std::isfinite(payload.elapsed_seconds) || payload.elapsed_seconds < 0.0) {
    return Status(StatusCode::kProtocolError,
                  "distributed benchmark elapsed time is invalid");
  }
  try {
    payload.records.resize(static_cast<std::size_t>(count.value()));
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate distributed benchmark records");
  }
  for (auto& record : payload.records) {
#define READ_RECORD_FIELD(target, field) \
    do { auto value = reader.U64(); if (!value.ok()) return value.status(); \
         (target) = value.value(); } while (false)
    READ_RECORD_FIELD(record.query_id, query_id);
    READ_RECORD_FIELD(record.latency_nanoseconds, latency_nanoseconds);
    std::uint64_t has_recall = 0;
    std::uint64_t recall_bits = 0;
    std::uint64_t success = 0;
    READ_RECORD_FIELD(has_recall, has_recall);
    READ_RECORD_FIELD(recall_bits, recall_bits);
    READ_RECORD_FIELD(success, success);
    if (has_recall > 1U || success > 1U) {
      return Status(StatusCode::kProtocolError,
                    "distributed benchmark record flags are invalid");
    }
    if (has_recall != 0) {
      const auto recall = std::bit_cast<double>(recall_bits);
      if (!std::isfinite(recall)) {
        return Status(StatusCode::kProtocolError,
                      "distributed benchmark recall is invalid");
      }
      record.recall = recall;
    }
    record.success = success != 0;
#define READ_SEARCH(name) \
    do { auto value = reader.U64(); if (!value.ok()) return value.status(); \
         record.search.name = value.value(); } while (false);
    SEARCH_PROFILE_FIELDS(READ_SEARCH)
#undef READ_SEARCH
#define READ_TIMING(name) \
    do { auto value = reader.U64(); if (!value.ok()) return value.status(); \
         record.name = value.value(); } while (false);
    RECORD_TIMING_FIELDS(READ_TIMING)
#undef READ_TIMING
#undef READ_RECORD_FIELD
  }
#define READ_PROFILE(name) \
  do { auto value = reader.U64(); if (!value.ok()) return value.status(); \
       payload.profile.name = value.value(); } while (false);
  PROFILE_SCALAR_FIELDS(READ_PROFILE)
#undef READ_PROFILE
#define READ_PROFILE_VECTOR(name) \
  do { auto value = ReadVector(reader); if (!value.ok()) return value.status(); \
       payload.profile.name = std::move(value).value(); } while (false);
  PROFILE_VECTOR_FIELDS(READ_PROFILE_VECTOR)
#undef READ_PROFILE_VECTOR
  if (!reader.done()) {
    return Status(StatusCode::kProtocolError,
                  "distributed benchmark payload has trailing bytes");
  }
  return payload;
}

Status WriteOutput(const std::filesystem::path& path,
                   std::string_view content) {
  if (path == "-") {
    std::cout << content;
    return std::cout.good()
        ? Status::Ok()
        : Status(StatusCode::kInternalError, "unable to write stdout");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status(StatusCode::kInvalidArgument,
                  "unable to open benchmark output: " + path.string());
  }
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  if (!output) {
    return Status(StatusCode::kInternalError,
                  "unable to write benchmark output: " + path.string());
  }
  return Status::Ok();
}

}  // namespace hnsw_urma
