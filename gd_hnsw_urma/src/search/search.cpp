#include "hnsw_urma/search/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
namespace {

std::uint64_t SteadyNowNanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool Better(const HeapEntry &lhs, const HeapEntry &rhs) noexcept {
  return lhs.distance < rhs.distance ||
         (lhs.distance == rhs.distance && lhs.id < rhs.id);
}

using SearchClock = std::chrono::steady_clock;

Result<SearchClock::time_point>
CheckedQueryDeadline(SearchClock::time_point admission,
                     std::chrono::milliseconds timeout) {
  // Check the conversion as well as the addition: milliseconds::max() may
  // overflow the clock's finer duration before time_point addition begins.
  const auto max_timeout =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          SearchClock::duration::max());
  if (timeout > max_timeout) {
    return Status(StatusCode::kOutOfRange,
                  "query timeout exceeds the steady clock duration range");
  }
  const auto duration =
      std::chrono::duration_cast<SearchClock::duration>(timeout);
  if (admission > SearchClock::time_point::max() - duration) {
    return Status(StatusCode::kOutOfRange,
                  "query deadline exceeds the steady clock time point range");
  }
  return admission + duration;
}

SearchClock::time_point
CappedDeadline(SearchClock::time_point now, SearchClock::duration timeout,
               SearchClock::time_point query_deadline) noexcept {
  if (now > SearchClock::time_point::max() - timeout) {
    return query_deadline;
  }
  return std::min(now + timeout, query_deadline);
}

class ExclusiveProfileTimerState {
public:
  [[nodiscard]] std::uint64_t *Enter(std::uint64_t &destination) noexcept {
    const auto now = SteadyNowNanoseconds();
    if (current_ != nullptr) {
      *current_ += now - started_;
    }
    auto *previous = current_;
    current_ = &destination;
    started_ = now;
    return previous;
  }

  void Leave(std::uint64_t *previous) noexcept {
    const auto now = SteadyNowNanoseconds();
    if (current_ != nullptr) {
      *current_ += now - started_;
    }
    current_ = previous;
    started_ = now;
  }

private:
  std::uint64_t *current_{nullptr};
  std::uint64_t started_{0};
};

class ScopedProfileTimer {
public:
  ScopedProfileTimer(ExclusiveProfileTimerState &state, bool enabled,
                     std::uint64_t &destination) noexcept
      : state_(enabled ? &state : nullptr),
        previous_(enabled ? state.Enter(destination) : nullptr) {}

  ~ScopedProfileTimer() {
    if (state_ != nullptr) {
      state_->Leave(previous_);
    }
  }

private:
  ExclusiveProfileTimerState *state_{nullptr};
  std::uint64_t *previous_{nullptr};
};

Result<std::size_t> ResolveVisitedCapacity(const SearchParams &params,
                                           const GlobalIndexMeta &meta) {
  if (params.visited_capacity != 0) {
    if ((params.visited_capacity & (params.visited_capacity - 1)) != 0) {
      return Status(StatusCode::kInvalidArgument,
                    "visited capacity must be a power of two");
    }
    return params.visited_capacity;
  }
  auto work = CheckedMultiply<std::uint64_t>(params.ef_search, meta.M0);
  if (!work.ok()) {
    return work.status();
  }
  auto scaled = CheckedMultiply<std::uint64_t>(work.value(), 2);
  if (!scaled.ok()) {
    return scaled.status();
  }
  std::uint64_t capacity = 1;
  while (capacity < scaled.value()) {
    if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
      return Status(StatusCode::kResourceExhausted,
                    "visited capacity overflows size_t");
    }
    capacity *= 2;
  }
  return static_cast<std::size_t>(capacity);
}

} // namespace

struct ScratchDistanceOperation {
  std::optional<DistanceTicket> ticket;
  bool local{false};
  bool harvested{false};
  bool terminal_notified{false};
  bool asynchronous{false};
};

struct QueryScratch::Impl {
  Impl(GlobalIndexMeta global_meta, SearchParams search_params,
       std::uint32_t lanes, MinimaxHeap search_heap, VisitedTable visited_table)
      : meta(global_meta), params(search_params), ready_producer_count(lanes),
        heap(std::move(search_heap)), visited(std::move(visited_table)) {}

  [[nodiscard]] bool Compatible(const GlobalIndexMeta &global_meta,
                                const SearchParams &search_params,
                                std::uint32_t lanes) const noexcept {
    return meta.ntotal == global_meta.ntotal && meta.dim == global_meta.dim &&
           meta.M == global_meta.M && meta.M0 == global_meta.M0 &&
           meta.num_shards == global_meta.num_shards &&
           params.ef_search == search_params.ef_search &&
           params.expand_batch == search_params.expand_batch &&
           params.visited_capacity == search_params.visited_capacity &&
           ready_producer_count == lanes;
  }

  [[nodiscard]] Status Acquire(std::uint64_t query_generation,
                               std::span<const float> input_query,
                               std::shared_ptr<const void> input_owner) {
    if (active || query_generation == 0 || input_query.size() != meta.dim) {
      return Status(StatusCode::kProtocolError,
                    "query scratch is active or incompatible");
    }
    if (input_owner == nullptr) {
      if (query_snapshot == nullptr || query_snapshot.use_count() != 1) {
        query_snapshot = std::make_shared<std::vector<float>>(meta.dim);
      }
      std::copy(input_query.begin(), input_query.end(),
                query_snapshot->begin());
      query_owner = query_snapshot;
      query_values = *query_snapshot;
    } else {
      query_owner = std::move(input_owner);
      query_values = input_query;
    }
    active = true;
    generation = query_generation;
    query_norm = ScalarDot(query_values, query_values);
    heap.Clear();
    visited.Clear();
    neighbor_batch.Clear();
    expansion_nodes.clear();
    routed_ids.clear();
    group_owners.clear();
    local_offsets.clear();
    routed_distances.clear();
    commit_entries.clear();
    group_offsets.clear();
    phase_lane_counts.clear();
    distance_operations.clear();
    std::fill(route_counts.begin(), route_counts.end(), 0U);
    std::fill(route_next.begin(), route_next.end(), 0U);
    return Status::Ok();
  }

  void Release(std::uint64_t query_generation) noexcept {
    if (active && generation == query_generation) {
      active = false;
      generation = 0;
      query_owner.reset();
      query_values = {};
    }
  }

  GlobalIndexMeta meta;
  SearchParams params;
  std::uint32_t ready_producer_count{1};
  bool active{false};
  std::uint64_t generation{0};
  std::shared_ptr<std::vector<float>> query_snapshot;
  std::shared_ptr<const void> query_owner;
  std::span<const float> query_values;
  float query_norm{0.0F};
  MinimaxHeap heap;
  VisitedTable visited;
  NeighborBatch neighbor_batch;
  std::vector<NodeToExpand> expansion_nodes;
  std::vector<node_id_t> routed_ids;
  std::vector<shard_id_t> group_owners;
  std::vector<std::uint64_t> local_offsets;
  std::vector<float> routed_distances;
  std::vector<HeapEntry> commit_entries;
  std::vector<std::uint32_t> group_offsets;
  std::vector<LaneOperationCount> phase_lane_counts;
  std::vector<ScratchDistanceOperation> distance_operations;
  std::vector<std::uint32_t> route_counts;
  std::vector<std::uint32_t> route_next;
};

QueryScratch::~QueryScratch() = default;

Result<std::unique_ptr<QueryScratch>>
QueryScratch::Create(const GlobalIndexMeta &meta, const SearchParams &params,
                     std::uint32_t active_slots,
                     std::uint32_t ready_producer_count,
                     std::uint64_t dense_visited_budget_bytes) {
  auto meta_status = ValidateGlobalMeta(meta);
  if (!meta_status.ok()) {
    return meta_status;
  }
  if (params.topk == 0 || params.ef_search < params.topk ||
      params.expand_batch == 0 || params.expand_batch > 64 ||
      active_slots == 0 || ready_producer_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "query scratch configuration is invalid");
  }
  auto capacity = ResolveVisitedCapacity(params, meta);
  if (!capacity.ok()) {
    return capacity.status();
  }
  auto heap = MinimaxHeap::Create(params.ef_search);
  if (!heap.ok()) {
    return heap.status();
  }
  auto visited = VisitedTable::CreateAdaptive(
      meta.ntotal, capacity.value(), active_slots, dense_visited_budget_bytes);
  if (!visited.ok()) {
    return visited.status();
  }
  try {
    auto scratch = std::unique_ptr<QueryScratch>(new QueryScratch());
    scratch->impl_ = std::make_unique<QueryScratch::Impl>(
        meta, params, ready_producer_count, std::move(heap).value(),
        std::move(visited).value());
    auto &storage = *scratch->impl_;
    const std::size_t graph_slots =
        static_cast<std::size_t>(params.expand_batch) * meta.M0;
    const auto batch_capacity = std::max<std::size_t>(meta.M, graph_slots);
    storage.query_snapshot = std::make_shared<std::vector<float>>(meta.dim);
    storage.neighbor_batch.neighbors.reserve(batch_capacity);
    storage.neighbor_batch.offsets.reserve(params.expand_batch + 1U);
    storage.expansion_nodes.reserve(params.expand_batch);
    storage.routed_ids.reserve(batch_capacity);
    storage.group_owners.reserve(meta.num_shards);
    storage.local_offsets.reserve(batch_capacity);
    storage.routed_distances.reserve(batch_capacity);
    storage.commit_entries.reserve(batch_capacity);
    storage.group_offsets.reserve(meta.num_shards + 1U);
    storage.distance_operations.reserve(meta.num_shards);
    storage.phase_lane_counts.reserve(ready_producer_count);
    storage.route_counts.resize(meta.num_shards);
    storage.route_next.resize(meta.num_shards);
    return scratch;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate query scratch");
  }
}

VisitedLayout QueryScratch::visited_layout() const noexcept {
  return impl_->visited.layout();
}

bool QueryScratch::active() const noexcept { return impl_->active; }

struct HnswSearchTask::Impl {
  enum class RetryPhase : std::uint8_t { kNone, kGraph, kDistance };

  Impl(QueryKey query_key, const GlobalIndexMeta &global_meta,
       const ShardMap &map, GraphReader &graph, DistanceOracle &distance,
       QueryScratch &query_scratch, SearchParams search_params,
       std::chrono::steady_clock::time_point deadline,
       const SearchRuntimeContext *runtime_context)
      : key(query_key), meta(global_meta), shard_map(&map),
        graph_reader(&graph), distance_oracle(&distance),
        scratch(&query_scratch),
        query_lease{query_key, query_scratch.impl_->query_owner,
                    query_scratch.impl_->query_values,
                    query_scratch.impl_->query_norm},
        query(query_lease.values()), params(search_params),
        heap(query_scratch.impl_->heap), visited(query_scratch.impl_->visited),
        neighbor_batch(query_scratch.impl_->neighbor_batch),
        expansion_nodes(query_scratch.impl_->expansion_nodes),
        routed_ids(query_scratch.impl_->routed_ids),
        group_owners(query_scratch.impl_->group_owners),
        local_offsets(query_scratch.impl_->local_offsets),
        routed_distances(query_scratch.impl_->routed_distances),
        route_counts(query_scratch.impl_->route_counts),
        route_next(query_scratch.impl_->route_next),
        commit_entries(query_scratch.impl_->commit_entries),
        group_offsets(query_scratch.impl_->group_offsets),
        phase_lane_counts(query_scratch.impl_->phase_lane_counts),
        distance_operations(query_scratch.impl_->distance_operations),
        controller(query_key.generation,
                   runtime_context == nullptr
                       ? 1U
                       : runtime_context->ready_producer_count,
                   search_params.profile_timing),
        ready_publisher(runtime_context == nullptr
                            ? nullptr
                            : runtime_context->ready_publisher),
        entry_point(search_params.entry_point == INVALID_NODE_ID
                        ? global_meta.entry_point
                        : search_params.entry_point),
        entry_level(search_params.entry_point == INVALID_NODE_ID
                        ? global_meta.max_level
                        : search_params.entry_level),
        query_deadline(deadline) {}

  ~Impl() {
    if (scratch_active) {
      scratch->impl_->Release(key.generation);
    }
  }

  QueryKey key;
  GlobalIndexMeta meta;
  const ShardMap *shard_map;
  GraphReader *graph_reader;
  DistanceOracle *distance_oracle;
  QueryScratch *scratch;
  QueryVectorLease query_lease;
  std::span<const float> query;
  SearchParams params;
  MinimaxHeap &heap;
  VisitedTable &visited;
  NeighborBatch &neighbor_batch;
  std::vector<NodeToExpand> &expansion_nodes;
  std::vector<node_id_t> &routed_ids;
  std::vector<shard_id_t> &group_owners;
  std::vector<std::uint64_t> &local_offsets;
  std::vector<float> &routed_distances;
  std::vector<std::uint32_t> &route_counts;
  std::vector<std::uint32_t> &route_next;
  std::vector<HeapEntry> &commit_entries;
  std::vector<std::uint32_t> &group_offsets;
  std::vector<LaneOperationCount> &phase_lane_counts;
  std::vector<ScratchDistanceOperation> &distance_operations;
  bool scratch_active{false};
  QueryRunController controller;
  QueryReadyPublisher *ready_publisher{nullptr};
  QueryState state{QueryState::kInit};
  bool runtime_abort_required{false};
  Status operation_status;
  Status final_status;
  SearchResult result;
  ExclusiveProfileTimerState local_phase_timer;

  node_id_t entry_point{INVALID_NODE_ID};
  std::int32_t entry_level{-1};
  node_id_t current_id{INVALID_NODE_ID};
  float current_distance{0.0F};
  std::int32_t current_level{0};
  std::uint64_t consecutive_upper_level_drops{0};
  bool upper_had_candidates_before_visited{false};

  RetryPhase retry_phase{RetryPhase::kNone};
  QueryState retry_wait_state{QueryState::kInit};
  std::optional<QueryRetryHandle> retry_handle;

  std::optional<GraphTicket> graph_ticket;
  bool graph_harvested{false};
  bool graph_terminal_notified{false};
  bool graph_direct_consumed{false};
  std::chrono::steady_clock::time_point query_deadline;
  std::chrono::steady_clock::time_point attempt_deadline;
  std::uint64_t wait_started_nanoseconds{0};
  std::uint64_t wait_all_published_nanoseconds{0};
  std::uint64_t wait_remote_operations{0};
  std::uint64_t wait_distance_values{0};
  std::atomic<bool> wait_published{false};
  bool resume_pending{false};
  std::uint64_t stage_started_nanoseconds{0};
  bool layer0_started{false};

  void StartStageTiming(std::uint64_t now_nanoseconds) noexcept {
    if (params.profile_timing && stage_started_nanoseconds == 0) {
      stage_started_nanoseconds = now_nanoseconds;
    }
  }

  void StartLayer0Timing() noexcept {
    if (!params.profile_timing || layer0_started) {
      return;
    }
    const auto now = SteadyNowNanoseconds();
    if (stage_started_nanoseconds != 0) {
      result.timing.upper_elapsed_nanoseconds +=
          now - stage_started_nanoseconds;
    }
    stage_started_nanoseconds = now;
    layer0_started = true;
  }

  void FinishStageTiming() noexcept {
    if (!params.profile_timing || stage_started_nanoseconds == 0) {
      return;
    }
    const auto elapsed = SteadyNowNanoseconds() - stage_started_nanoseconds;
    if (layer0_started) {
      result.timing.layer0_elapsed_nanoseconds += elapsed;
    } else {
      result.timing.upper_elapsed_nanoseconds += elapsed;
    }
    stage_started_nanoseconds = 0;
  }

  void RecordUpperLevelDrop(bool empty_neighbors) noexcept {
    ++result.profile.upper_level_drops;
    if (empty_neighbors) {
      ++result.profile.upper_empty_neighbor_level_drops;
    }
    if (consecutive_upper_level_drops == 0) {
      ++result.profile.upper_level_drop_runs;
    }
    ++consecutive_upper_level_drops;
    result.profile.upper_max_consecutive_level_drops =
        std::max(result.profile.upper_max_consecutive_level_drops,
                 consecutive_upper_level_drops);
  }

  void FlushUpperLevelDropRun() noexcept {
    switch (consecutive_upper_level_drops) {
    case 0:
      return;
    case 1:
      ++result.profile.upper_level_drop_runs_length_1;
      break;
    case 2:
      ++result.profile.upper_level_drop_runs_length_2;
      break;
    case 3:
      ++result.profile.upper_level_drop_runs_length_3;
      break;
    default:
      ++result.profile.upper_level_drop_runs_length_4_plus;
      break;
    }
    consecutive_upper_level_drops = 0;
  }

  void RecordUpperSameLevelMove() noexcept {
    FlushUpperLevelDropRun();
    ++result.profile.upper_same_level_moves;
  }

  Status FilterUpperVisitedCandidates() {
    upper_had_candidates_before_visited = !routed_ids.empty();
    if (routed_ids.empty()) {
      return Status::Ok();
    }
    // Upper search follows one greedy current node whose (distance, id) key
    // only improves. Once a candidate has been evaluated, a later current
    // node is at least as good, so evaluating that candidate again cannot
    // change the greedy choice.
    bool step_fully_visited = true;
    std::size_t write_index = 0;
    std::size_t retained_groups = 0;
    std::size_t source_begin = group_offsets.front();
    for (std::size_t group = 0; group + 1U < group_offsets.size(); ++group) {
      const auto source_end = group_offsets[group + 1U];
      const bool remote =
          !distance_oracle->CanComputeLocalTrusted(group_owners[group]);
      const auto retained_begin = write_index;
      if (remote) {
        ++result.profile.upper_remote_operations;
        result.profile.upper_remote_candidate_values +=
            source_end - source_begin;
      }
      for (std::size_t index = source_begin; index < source_end; ++index) {
        ++result.profile.upper_distance_candidate_values;
        auto inserted = visited.Mark(routed_ids[index]);
        if (!inserted.ok()) {
          return inserted.status();
        }
        if (inserted.value()) {
          ++result.profile.upper_distance_unique_values;
          step_fully_visited = false;
          routed_ids[write_index] = routed_ids[index];
          ++write_index;
        } else {
          ++result.profile.upper_distance_visited_hits;
          if (remote) {
            ++result.profile.upper_remote_visited_hits;
          }
        }
      }
      if (write_index != retained_begin) {
        group_owners[retained_groups] = group_owners[group];
        group_offsets[++retained_groups] =
            static_cast<std::uint32_t>(write_index);
      } else if (remote) {
        ++result.profile.upper_eliminated_remote_operations;
      }
      source_begin = source_end;
    }
    routed_ids.resize(write_index);
    routed_distances.resize(write_index);
    group_offsets.resize(retained_groups + 1U);
    group_owners.resize(retained_groups);
    if (step_fully_visited) {
      ++result.profile.upper_distance_fully_visited_steps;
    }
    return Status::Ok();
  }

  void RecordReadyTiming(std::uint64_t resumed_at) noexcept {
    if (resume_pending) {
      ++result.profile.resume_count;
      resume_pending = false;
    }
    if (!params.profile_timing) {
      return;
    }
    const auto ready_at = controller.TakeReadyTimestampNanoseconds();
    const auto operation_ready_at =
        controller.TakeOperationReadyTimestampNanoseconds();
    const auto first_operation_at =
        controller.TakeFirstOperationTimestampNanoseconds();
    const auto critical_lane = controller.TakeCriticalReadyLane();
    const auto wait_started = std::exchange(wait_started_nanoseconds, 0);
    const auto all_published = std::exchange(wait_all_published_nanoseconds, 0);
    const auto remote_operations = std::exchange(wait_remote_operations, 0);
    const auto distance_values = std::exchange(wait_distance_values, 0);
    if (ready_at == 0 || operation_ready_at == 0 || wait_started == 0 ||
        operation_ready_at < wait_started || resumed_at < ready_at) {
      return;
    }
    const auto wait_nanoseconds = operation_ready_at - wait_started;
    if (state == QueryState::kWaitLayer0Graph) {
      result.timing.graph_wait_nanoseconds += wait_nanoseconds;
    } else if (state == QueryState::kWaitEntryDistance ||
               state == QueryState::kWaitUpperDistance ||
               state == QueryState::kWaitLayer0Distance) {
      result.timing.distance_wait_nanoseconds += wait_nanoseconds;
    }
    result.timing.ready_queue_nanoseconds += resumed_at - ready_at;
    if (ready_publisher == nullptr) {
      return;
    }
    std::optional<QueryWaitProfileKind> kind;
    if (state == QueryState::kWaitLayer0Graph) {
      kind = QueryWaitProfileKind::kGraph;
    } else if (state == QueryState::kWaitEntryDistance) {
      kind = QueryWaitProfileKind::kEntryDistance;
    } else if (state == QueryState::kWaitUpperDistance) {
      kind = QueryWaitProfileKind::kUpperDistance;
    } else if (state == QueryState::kWaitLayer0Distance) {
      kind = QueryWaitProfileKind::kLayer0Distance;
    }
    if (!kind.has_value()) {
      return;
    }
    const auto publish_finished =
        std::clamp(all_published, wait_started, operation_ready_at);
    const auto first_completed =
        std::clamp(first_operation_at, publish_finished, operation_ready_at);
    ready_publisher->RecordWaitProfile({
        *kind,
        operation_ready_at - wait_started,
        publish_finished - wait_started,
        first_completed - publish_finished,
        operation_ready_at - first_completed,
        remote_operations,
        distance_values,
        critical_lane,
    });
  }

  void Fail(Status status) {
    FinishStageTiming();
    status.AddContext("source_rank", std::to_string(key.source_rank))
        .AddContext("query_slot", std::to_string(key.query_slot))
        .AddContext("generation", std::to_string(key.generation))
        .AddContext("query_state",
                    std::to_string(static_cast<unsigned>(state)));
    final_status = std::move(status);
    state = QueryState::kFailed;
    controller.MarkFailed();
  }

  bool OperationFailed() {
    if (!operation_status.ok()) {
      Fail(operation_status);
      return true;
    }
    return false;
  }

  void SetAttemptDeadline(std::chrono::steady_clock::time_point now) {
    attempt_deadline =
        CappedDeadline(now,
                       std::chrono::duration_cast<SearchClock::duration>(
                           params.operation_timeout),
                       query_deadline);
  }

  [[nodiscard]] std::optional<SearchStepResult>
  HandleScheduleResult(Result<bool> scheduled, RetryPhase phase,
                       QueryState wait_state,
                       std::chrono::steady_clock::time_point now) {
    if (!scheduled.ok()) {
      if (scheduled.status().code() == StatusCode::kResourceExhausted) {
        auto epoch = controller.RequeueForRetry();
        if (!epoch.ok()) {
          Fail(epoch.status());
          return SearchStepResult::kFailed;
        }
        retry_phase = phase;
        retry_wait_state = wait_state;
        retry_handle = QueryRetryHandle{
            key, epoch.value(),
            CappedDeadline(now,
                           std::chrono::duration_cast<SearchClock::duration>(
                               std::chrono::microseconds(50)),
                           query_deadline)};
        return SearchStepResult::kYielded;
      }
      Fail(scheduled.status());
      return SearchStepResult::kFailed;
    }
    if (scheduled.value()) {
      return SearchStepResult::kYielded;
    }
    return std::nullopt;
  }

  void CompleteControllerOperation() {
    auto completed = controller.CompleteOne(key.generation);
    if (!completed.ok() && controller.state() != RunState::kFailed) {
      Fail(completed.status());
    }
  }

  Status RollbackGraphPreparation(Status failure) {
    auto cancelled = graph_reader->CancelPrepared(*graph_ticket);
    if (cancelled.ok()) {
      graph_ticket.reset();
    } else {
      runtime_abort_required = true;
      failure.AddContext("cancel_prepared", cancelled.ToString());
    }
    return failure;
  }

  Result<bool> ScheduleGraph(QueryState wait_state,
                             std::chrono::steady_clock::time_point now) {
    wait_published.store(false, std::memory_order_release);
    wait_all_published_nanoseconds = 0;
    wait_remote_operations = 0;
    wait_distance_values = 0;
    operation_status = Status::Ok();
    graph_ticket.reset();
    graph_harvested = false;
    graph_terminal_notified = false;
    graph_direct_consumed = false;
    state = wait_state;
    SetAttemptDeadline(now);
    if (ready_publisher != nullptr) {
      const auto expected_epoch = controller.next_wait_epoch();
      auto prepared = [&]() {
        ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                                 result.timing.graph_plan_nanoseconds);
        return graph_reader->PrepareForWait({key, expected_epoch,
                                             attempt_deadline, &controller,
                                             ready_publisher},
                                            expansion_nodes, &neighbor_batch);
      }();
      if (!prepared.ok()) {
        return prepared.status();
      }
      graph_ticket = prepared.value().ticket;
      {
        ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                                 result.timing.graph_publish_nanoseconds);
        if (prepared.value().asynchronous) {
          if (params.profile_timing) {
            wait_started_nanoseconds = SteadyNowNanoseconds();
          }
          auto waiting = controller.BeginWait(
              prepared.value().participating_lanes, key.generation);
          if (!waiting.ok()) {
            wait_started_nanoseconds = 0;
            return RollbackGraphPreparation(waiting.status());
          }
          if (waiting.value() != expected_epoch) {
            wait_started_nanoseconds = 0;
            return RollbackGraphPreparation(
                Status(StatusCode::kProtocolError,
                       "prepared graph wait epoch changed during preparation"));
          }
          ++result.profile.suspend_count;
          resume_pending = true;
          for (const auto &lane : prepared.value().participating_lanes) {
            wait_remote_operations += lane.operations;
          }
        }
        auto published = graph_reader->PublishPrepared(*graph_ticket);
        if (!published.ok()) {
          wait_started_nanoseconds = 0;
          auto failure = Status(StatusCode::kInternalError,
                                "prepared graph publication failed after wait "
                                "registration");
          failure.AddContext("cause", published.ToString());
          // Graph publication failures retain an unpublished, cancellable
          // phase. Never discard the ticket if rollback itself is rejected.
          return RollbackGraphPreparation(std::move(failure));
        }
      }
      if (prepared.value().asynchronous) {
        if (params.profile_timing) {
          wait_all_published_nanoseconds = SteadyNowNanoseconds();
        }
        return true;
      }
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.graph_consume_nanoseconds);
      auto finished = FinishGraphTicket();
      graph_harvested = true;
      graph_ticket.reset();
      if (!finished.status.ok()) {
        operation_status = std::move(finished.status);
      }
      return false;
    }
    auto parked = controller.Park(1);
    if (!parked.ok()) {
      return parked;
    }
    ++result.profile.suspend_count;
    resume_pending = true;
    if (params.profile_timing) {
      wait_started_nanoseconds = SteadyNowNanoseconds();
    }
    auto submitted = [&]() {
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.graph_plan_nanoseconds);
      return graph_reader->Submit(key, expansion_nodes, &neighbor_batch);
    }();
    if (!submitted.ok()) {
      operation_status = submitted.status();
      graph_harvested = true;
      CompleteControllerOperation();
    } else {
      graph_ticket = submitted.value();
    }
    wait_published.store(true, std::memory_order_release);
    return true;
  }

  void BeginRouteIds() {
    routed_ids.clear();
    group_owners.clear();
    local_offsets.clear();
    group_offsets.clear();
    group_offsets.push_back(0);
    std::fill(route_counts.begin(), route_counts.end(), 0U);
  }

  Result<shard_id_t> RouteOwner(node_id_t id) const noexcept {
    const auto uniform_count = shard_map->uniform_count();
    if (uniform_count != 0) {
      if (static_cast<std::uint64_t>(id) >= shard_map->ntotal()) {
        return Status(StatusCode::kOutOfRange,
                      "global node ID is outside the index");
      }
      return static_cast<shard_id_t>(static_cast<std::uint64_t>(id) /
                                     uniform_count);
    }
    auto location = shard_map->Locate(id);
    if (!location.ok()) {
      return location.status();
    }
    return location.value().shard_id;
  }

  Status ConsumeRouteIds(std::span<const node_id_t> ids, bool use_visited) {
    {
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.neighbor_scan_visited_nanoseconds);
      for (std::size_t input_index = 0; input_index < ids.size();
           ++input_index) {
        constexpr std::size_t kVisitedPrefetchDistance = 4;
        if (use_visited &&
            input_index + kVisitedPrefetchDistance < ids.size()) {
          visited.Prefetch(ids[input_index + kVisitedPrefetchDistance]);
        }
        const auto id = ids[input_index];
        if (id == INVALID_NODE_ID) {
          continue;
        }
        if (use_visited) {
          auto marked = visited.Mark(id);
          if (!marked.ok()) {
            return marked.status();
          }
          if (!marked.value()) {
            ++result.profile.visited_hits;
            continue;
          }
        }
        auto owner = RouteOwner(id);
        if (!owner.ok()) {
          return owner.status();
        }
        if (route_counts[owner.value()] ==
                std::numeric_limits<std::uint32_t>::max() ||
            routed_ids.size() == std::numeric_limits<std::uint32_t>::max()) {
          return Status(StatusCode::kOutOfRange,
                        "routed ID count exceeds uint32");
        }
        ++route_counts[owner.value()];
        routed_ids.push_back(id);
      }
    }
    return Status::Ok();
  }

  Status FinalizeRouteIds(bool use_visited) {
    {
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.route_group_sort_nanoseconds);
      std::uint32_t cursor = 0;
      for (std::size_t owner = 0; owner < route_counts.size(); ++owner) {
        const auto count = route_counts[owner];
        route_next[owner] = cursor;
        cursor += count;
        route_counts[owner] = cursor;
        if (count != 0) {
          group_owners.push_back(static_cast<shard_id_t>(owner));
          ++result.profile.routing_group_owner_writes;
          group_offsets.push_back(cursor);
        }
      }
      for (std::size_t owner = 0; owner < route_next.size(); ++owner) {
        while (route_next[owner] < route_counts[owner]) {
          const auto source = route_next[owner];
          auto destination_owner = RouteOwner(routed_ids[source]);
          if (!destination_owner.ok()) {
            return destination_owner.status();
          }
          if (destination_owner.value() == owner) {
            ++route_next[owner];
            continue;
          }
          const auto destination = route_next[destination_owner.value()]++;
          std::swap(routed_ids[source], routed_ids[destination]);
        }
      }

      if (!use_visited) {
        group_owners.clear();
        group_offsets.clear();
        group_offsets.push_back(0);
      }
      std::uint32_t write = 0;
      for (std::size_t owner = 0; owner < route_counts.size(); ++owner) {
        const auto begin = owner == 0 ? 0U : route_counts[owner - 1U];
        const auto end = route_counts[owner];
        if (begin == end) {
          continue;
        }
        const auto shard = static_cast<shard_id_t>(owner);
        const bool sort_bucket =
            !use_visited || params.layer0_sort_mode == Layer0SortMode::kAll ||
            (params.layer0_sort_mode == Layer0SortMode::kLocalOnly &&
             distance_oracle->CanComputeLocalTrusted(shard));
        auto bucket =
            std::span<node_id_t>(routed_ids).subspan(begin, end - begin);
        if (sort_bucket) {
          std::sort(bucket.begin(), bucket.end());
        }
        if (use_visited) {
          if (sort_bucket) {
            result.profile.routing_layer0_sorted_values += bucket.size();
          } else {
            result.profile.routing_layer0_unsorted_values += bucket.size();
          }
          continue;
        }
        const auto unique_end = std::unique(bucket.begin(), bucket.end());
        const auto unique_count = static_cast<std::uint32_t>(
            std::distance(bucket.begin(), unique_end));
        if (write != begin) {
          std::move(bucket.begin(), unique_end, routed_ids.begin() + write);
        }
        write += unique_count;
        group_owners.push_back(shard);
        group_offsets.push_back(write);
      }
      if (!use_visited) {
        routed_ids.resize(write);
      }
    }
    routed_distances.resize(routed_ids.size());
    return Status::Ok();
  }

  Status RouteIds(std::span<const node_id_t> ids, bool use_visited) {
    BeginRouteIds();
    auto consumed = ConsumeRouteIds(ids, use_visited);
    return consumed.ok() ? FinalizeRouteIds(use_visited) : consumed;
  }

  struct GraphFinishResult {
    Status status;
  };

  GraphFinishResult FinishGraphTicket() {
    if (!graph_ticket.has_value()) {
      return {Status(StatusCode::kProtocolError,
                     "graph finish has no active ticket")};
    }
    if (ready_publisher == nullptr || state != QueryState::kWaitLayer0Graph ||
        !graph_reader->SupportsDirectConsume()) {
      return {graph_reader->Finish(*graph_ticket)};
    }

    class RouteAccumulator final : public GraphNeighborConsumer {
    public:
      explicit RouteAccumulator(Impl &search) : search_(search) {}

      Status Consume(std::size_t node_index,
                     std::span<const node_id_t> neighbors) override {
        if (node_index != next_node_ ||
            node_index >= search_.expansion_nodes.size()) {
          return Status(StatusCode::kProtocolError,
                        "direct graph consumer received an invalid node index");
        }
        ++next_node_;
        search_.result.profile.edges_scanned += neighbors.size();
        auto consumed = search_.ConsumeRouteIds(neighbors, true);
        return consumed;
      }

    private:
      Impl &search_;
      std::size_t next_node_{0};
    };

    BeginRouteIds();
    RouteAccumulator accumulator(*this);
    auto status = graph_reader->FinishConsume(*graph_ticket, accumulator);
    if (status.ok()) {
      status = FinalizeRouteIds(true);
    }
    graph_direct_consumed = status.ok();
    return {std::move(status)};
  }

  Result<bool> ScheduleDistances(QueryState wait_state,
                                 std::chrono::steady_clock::time_point now) {
    wait_published.store(false, std::memory_order_release);
    wait_all_published_nanoseconds = 0;
    wait_remote_operations = 0;
    wait_distance_values = 0;
    if (group_offsets.size() < 2 || group_offsets.back() != routed_ids.size() ||
        group_owners.size() + 1U != group_offsets.size()) {
      return Status(StatusCode::kProtocolError,
                    "routed distance groups are inconsistent");
    }
    const std::uint32_t operation_count =
        static_cast<std::uint32_t>(group_offsets.size() - 1);
    operation_status = Status::Ok();
    distance_operations.clear();
    distance_operations.resize(operation_count);
    DistanceRequestOptions request_options;
    if (wait_state == QueryState::kWaitLayer0Distance &&
        heap.result_size() == heap.capacity()) {
      const auto upper = heap.max();
      if (upper.has_value()) {
        request_options.upper_distance = upper->distance;
      }
    }
    state = wait_state;
    SetAttemptDeadline(now);
    auto compute_local = [&](std::size_t operation) -> Result<bool> {
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.local_distance_nanoseconds);
      const auto begin = group_offsets[operation];
      const auto count = group_offsets[operation + 1] - begin;
      const auto owner = group_owners[operation];
      const auto &range = shard_map->ranges()[owner];
      local_offsets.resize(count);
      result.profile.routing_local_offset_writes += count;
      for (std::size_t index = 0; index < count; ++index) {
        const auto id = routed_ids[begin + index];
        if (id < range.global_begin ||
            static_cast<std::uint64_t>(id - range.global_begin) >=
                range.local_count) {
          return Status(
              StatusCode::kProtocolError,
              "local distance group contains an ID outside its owner range");
        }
        local_offsets[index] =
            static_cast<std::uint64_t>(id - range.global_begin);
      }
      return distance_oracle->ComputeLocalTrustedLease(
          query_lease, owner, local_offsets,
          std::span<float>(routed_distances).subspan(begin, count));
    };
    bool has_local = false;
    for (std::size_t operation = 0; operation < operation_count; ++operation) {
      distance_operations[operation].local =
          distance_oracle->CanComputeLocalTrusted(group_owners[operation]);
      has_local |= distance_operations[operation].local;
    }
    if (ready_publisher != nullptr) {
      phase_lane_counts.clear();
      const auto expected_epoch = controller.next_wait_epoch();
      for (std::size_t operation = 0; operation < operation_count;
           ++operation) {
        if (distance_operations[operation].local) {
          continue;
        }
        const auto begin = group_offsets[operation];
        const auto count = group_offsets[operation + 1] - begin;
        ScopedProfileTimer timer(
            local_phase_timer, params.profile_timing,
            result.timing.remote_prepare_publish_nanoseconds);
        auto prepared = distance_oracle->PrepareLeaseForWait(
            {key, expected_epoch, attempt_deadline, &controller,
             ready_publisher},
            query_lease, group_owners[operation],
            std::span<const node_id_t>(routed_ids).subspan(begin, count),
            std::span<float>(routed_distances).subspan(begin, count),
            request_options);
        if (!prepared.ok()) {
          for (auto &prior : distance_operations) {
            if (prior.ticket.has_value()) {
              (void)distance_oracle->CancelPrepared(*prior.ticket);
              prior.ticket.reset();
            }
          }
          return prepared.status();
        }
        auto &record = distance_operations[operation];
        record.ticket = prepared.value().ticket;
        record.asynchronous = prepared.value().owner_lane.has_value();
        if (!record.asynchronous) {
          continue;
        }
        ++wait_remote_operations;
        wait_distance_values += count;
        const auto lane = *prepared.value().owner_lane;
        auto found = std::find_if(
            phase_lane_counts.begin(), phase_lane_counts.end(),
            [lane](const auto &entry) { return entry.lane == lane; });
        if (found == phase_lane_counts.end()) {
          phase_lane_counts.push_back({lane, 1});
        } else {
          ++found->operations;
        }
      }
      std::sort(
          phase_lane_counts.begin(), phase_lane_counts.end(),
          [](const auto &lhs, const auto &rhs) { return lhs.lane < rhs.lane; });
      const bool defer_ready = has_local && !phase_lane_counts.empty();
      if (defer_ready) {
        auto deferred = controller.BeginDeferredReady();
        if (!deferred.ok()) {
          for (auto &record : distance_operations) {
            if (record.ticket.has_value()) {
              (void)distance_oracle->CancelPrepared(*record.ticket);
            }
          }
          return deferred;
        }
      }
      if (!phase_lane_counts.empty()) {
        if (params.profile_timing) {
          wait_started_nanoseconds = SteadyNowNanoseconds();
        }
        auto waiting = controller.BeginWait(phase_lane_counts, key.generation);
        if (!waiting.ok()) {
          wait_started_nanoseconds = 0;
          if (defer_ready) {
            controller.EndDeferredReady();
          }
          for (auto &record : distance_operations) {
            if (record.ticket.has_value()) {
              (void)distance_oracle->CancelPrepared(*record.ticket);
            }
          }
          return waiting.status();
        }
        if (waiting.value() != expected_epoch) {
          wait_started_nanoseconds = 0;
          auto failure =
              Status(StatusCode::kProtocolError,
                     "distance wait epoch changed during preparation");
          if (defer_ready) {
            controller.EndDeferredReady();
          }
          for (auto &record : distance_operations) {
            if (!record.ticket.has_value())
              continue;
            auto cancelled = distance_oracle->CancelPrepared(*record.ticket);
            if (cancelled.ok()) {
              record.ticket.reset();
            } else {
              runtime_abort_required = true;
              failure.AddContext("cancel_prepared", cancelled.ToString());
            }
          }
          return failure;
        }
        ++result.profile.suspend_count;
        resume_pending = true;
      }
      for (std::size_t index = 0; index < distance_operations.size(); ++index) {
        auto &record = distance_operations[index];
        if (!record.ticket.has_value())
          continue;
        ScopedProfileTimer timer(
            local_phase_timer, params.profile_timing,
            result.timing.remote_prepare_publish_nanoseconds);
        auto published = distance_oracle->PublishPrepared(*record.ticket);
        if (!published.ok()) {
          runtime_abort_required = true;
          // Only later operations are certainly unpublished. Keep the current
          // and earlier slots stable until rank-fatal shutdown stops Progress.
          for (std::size_t later = index + 1;
               later < distance_operations.size(); ++later) {
            auto &pending = distance_operations[later];
            if (!pending.ticket)
              continue;
            auto cancelled = distance_oracle->CancelPrepared(*pending.ticket);
            if (cancelled.ok())
              pending.ticket.reset();
          }
          wait_started_nanoseconds = 0;
          if (defer_ready)
            controller.EndDeferredReady();
          return Status(StatusCode::kInternalError,
                        "prepared distance publication failed after wait "
                        "registration")
              .AddContext("cause", published.ToString());
        }
      }
      if (!phase_lane_counts.empty() && params.profile_timing) {
        wait_all_published_nanoseconds = SteadyNowNanoseconds();
      }
      for (std::size_t operation = 0; operation < operation_count;
           ++operation) {
        auto &record = distance_operations[operation];
        if (!record.local) {
          continue;
        }
        auto local = compute_local(operation);
        if (!local.ok() || !local.value()) {
          const auto status =
              local.ok()
                  ? Status(StatusCode::kProtocolError,
                           "trusted local distance path declined its group")
                  : local.status();
          if (defer_ready) {
            operation_status = status;
            static_cast<void>(controller.RequestFailure(
                key.generation, TerminalOutcome::kFailed, status.code()));
            controller.EndDeferredReady();
            return true;
          }
          return status;
        }
        record.harvested = true;
      }
      if (defer_ready) {
        controller.EndDeferredReady();
      }
      if (!phase_lane_counts.empty()) {
        return true;
      }
      for (auto &record : distance_operations) {
        if (!record.ticket.has_value()) {
          continue;
        }
        ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                                 result.timing.remote_finish_nanoseconds);
        auto finished = distance_oracle->Finish(*record.ticket);
        if (!finished.ok() && operation_status.ok()) {
          operation_status = finished;
        }
        record.harvested = true;
      }
      return false;
    }
    std::uint32_t pending_operations = 0;
    for (std::size_t operation = 0; operation < operation_count; ++operation) {
      if (distance_operations[operation].local) {
        auto local = compute_local(operation);
        if (!local.ok()) {
          return local.status();
        }
        if (!local.value()) {
          return Status(StatusCode::kProtocolError,
                        "trusted local distance path declined its group");
        }
        distance_operations[operation].harvested = true;
      } else {
        ++pending_operations;
      }
    }
    if (pending_operations == 0) {
      return false;
    }
    auto parked = controller.Park(pending_operations);
    if (!parked.ok()) {
      return parked;
    }
    ++result.profile.suspend_count;
    resume_pending = true;
    if (params.profile_timing) {
      wait_started_nanoseconds = SteadyNowNanoseconds();
    }
    for (std::size_t operation = 0; operation < operation_count; ++operation) {
      if (distance_operations[operation].harvested) {
        continue;
      }
      const auto begin = group_offsets[operation];
      const auto count = group_offsets[operation + 1] - begin;
      auto submitted = distance_oracle->SubmitLeaseForWait(
          query_lease, controller.wait_epoch(), group_owners[operation],
          std::span<const node_id_t>(routed_ids).subspan(begin, count),
          std::span<float>(routed_distances).subspan(begin, count),
          request_options);
      if (!submitted.ok()) {
        if (operation_status.ok()) {
          operation_status = submitted.status();
        }
        distance_operations[operation].harvested = true;
        CompleteControllerOperation();
      } else {
        distance_operations[operation].ticket = submitted.value();
      }
    }
    wait_published.store(true, std::memory_order_release);
    return true;
  }

  void ProbeGraph() {
    if (graph_harvested || graph_terminal_notified ||
        !graph_ticket.has_value()) {
      return;
    }
    auto ready = graph_reader->Poll(*graph_ticket);
    if (!ready.ok()) {
      operation_status = ready.status();
    } else if (!ready.value()) {
      return;
    } else {
      graph_terminal_notified = true;
    }
    if (!ready.ok()) {
      graph_harvested = true;
    }
    CompleteControllerOperation();
  }

  void ProbeDistances() {
    for (auto &operation : distance_operations) {
      if (operation.harvested || operation.terminal_notified ||
          !operation.ticket.has_value()) {
        continue;
      }
      auto ready = distance_oracle->Poll(*operation.ticket);
      if (!ready.ok()) {
        if (operation_status.ok()) {
          operation_status = ready.status();
        }
      } else if (!ready.value()) {
        continue;
      } else {
        operation.terminal_notified = true;
      }
      if (!ready.ok()) {
        operation.harvested = true;
      }
      CompleteControllerOperation();
    }
  }

  void ProbeWaiting(std::chrono::steady_clock::time_point now) {
    if (controller.state() != RunState::kWaiting) {
      return;
    }
    switch (state) {
    case QueryState::kWaitLayer0Graph:
      ProbeGraph();
      break;
    case QueryState::kWaitEntryDistance:
    case QueryState::kWaitUpperDistance:
    case QueryState::kWaitLayer0Distance:
      ProbeDistances();
      break;
    default:
      Fail(Status(StatusCode::kInternalError,
                  "waiting run state has no asynchronous phase"));
      return;
    }
    if (controller.state() == RunState::kWaiting && now >= attempt_deadline) {
      Fail(Status(StatusCode::kTimeout, "search operation deadline exceeded"));
    }
  }

  void ConsumeTerminalResults() {
    if (ready_publisher != nullptr) {
      if (!graph_harvested && graph_ticket.has_value()) {
        ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                                 result.timing.graph_consume_nanoseconds);
        auto finished = FinishGraphTicket();
        if (!finished.status.ok() && operation_status.ok()) {
          operation_status = std::move(finished.status);
        }
        graph_harvested = true;
        graph_ticket.reset();
      }
      for (auto &operation : distance_operations) {
        if (operation.harvested || !operation.ticket.has_value()) {
          continue;
        }
        ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                                 result.timing.remote_finish_nanoseconds);
        auto finished = distance_oracle->Finish(*operation.ticket);
        if (!finished.ok() && operation_status.ok()) {
          operation_status = std::move(finished);
        }
        operation.harvested = true;
      }
      return;
    }
    if (graph_terminal_notified && !graph_harvested &&
        graph_ticket.has_value()) {
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.graph_consume_nanoseconds);
      auto finished = FinishGraphTicket();
      if (!finished.status.ok() && operation_status.ok()) {
        operation_status = std::move(finished.status);
      }
      graph_harvested = true;
      graph_terminal_notified = false;
    }
    for (auto &operation : distance_operations) {
      if (!operation.terminal_notified || operation.harvested ||
          !operation.ticket.has_value()) {
        continue;
      }
      ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                               result.timing.remote_finish_nanoseconds);
      auto finished = distance_oracle->Finish(*operation.ticket);
      if (!finished.ok() && operation_status.ok()) {
        operation_status = std::move(finished);
      }
      operation.harvested = true;
      operation.terminal_notified = false;
    }
  }

  Result<bool> ScheduleEntry(std::chrono::steady_clock::time_point now) {
    routed_ids = {entry_point};
    auto location = shard_map->Locate(entry_point);
    if (!location.ok()) {
      return location.status();
    }
    group_owners = {location.value().shard_id};
    ++result.profile.routing_group_owner_writes;
    group_offsets = {0, 1};
    routed_distances.resize(1);
    return ScheduleDistances(QueryState::kWaitEntryDistance, now);
  }

  Result<std::span<const node_id_t>> ReadUpperGraphImmediately() {
    ScopedProfileTimer timer(local_phase_timer, params.profile_timing,
                             result.timing.upper_search_nanoseconds);
    return graph_reader->ReadUpperNeighbors(current_id, current_level);
  }

  void StartLayer0() {
    StartLayer0Timing();
    FlushUpperLevelDropRun();
    // Upper and layer-0 search are strictly serial. Advance the shared
    // generation/epoch so upper-level marks do not leak into layer-0.
    visited.Clear();
    auto marked = visited.Mark(current_id);
    if (!marked.ok()) {
      Fail(marked.status());
      return;
    }
    if (!marked.value()) {
      Fail(Status(StatusCode::kInternalError,
                  "unable to mark layer-0 entry point"));
      return;
    }
    auto pushed = heap.PushUnique(current_id, current_distance);
    if (!pushed.ok() || !pushed.value()) {
      Fail(pushed.ok() ? Status(StatusCode::kInternalError,
                                "unable to seed layer-0 heap")
                       : pushed.status());
      return;
    }
    state = QueryState::kReadyLayer0Expand;
  }

  SearchStepResult FinishQuery() {
    result.neighbors = heap.ExtractSorted();
    if (result.neighbors.size() > params.topk) {
      result.neighbors.resize(params.topk);
    }
    state = QueryState::kCompleted;
    auto completed = controller.MarkCompleted();
    if (!completed.ok()) {
      Fail(completed);
      return SearchStepResult::kFailed;
    }
    final_status = Status::Ok();
    FinishStageTiming();
    return SearchStepResult::kCompleted;
  }
};

HnswSearchTask::~HnswSearchTask() = default;

Result<std::unique_ptr<HnswSearchTask>> HnswSearchTask::Create(
    QueryKey key, const GlobalIndexMeta &meta, const ShardMap &shard_map,
    GraphReader &graph_reader, DistanceOracle &distance_oracle,
    std::span<const float> query, const SearchParams &params,
    std::chrono::steady_clock::time_point admission_time,
    const SearchRuntimeContext *runtime_context, QueryScratch *scratch,
    std::shared_ptr<const void> query_owner) {
  auto meta_status = ValidateGlobalMeta(meta);
  if (!meta_status.ok()) {
    return meta_status;
  }
  if (query.size() != meta.dim || params.topk == 0 ||
      params.ef_search < params.topk || params.expand_batch == 0 ||
      params.expand_batch > 64 || params.operation_timeout.count() <= 0 ||
      params.query_timeout < params.operation_timeout ||
      (runtime_context != nullptr &&
       (runtime_context->ready_publisher == nullptr ||
        runtime_context->ready_producer_count == 0 ||
        !graph_reader.SupportsPreparedWait() ||
        !distance_oracle.SupportsPreparedWait()))) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid query shape or search parameters");
  }
  const bool custom_entry =
      params.entry_point != INVALID_NODE_ID || params.entry_level != -1;
  if (key.source_rank >= meta.num_shards ||
      (custom_entry &&
       (params.entry_point == INVALID_NODE_ID || params.entry_level < 0 ||
        params.entry_level > meta.max_level))) {
    return Status(StatusCode::kInvalidArgument,
                  "query source rank or entry point is invalid");
  }
  auto deadline = CheckedQueryDeadline(admission_time, params.query_timeout);
  if (!deadline.ok()) {
    return deadline.status();
  }
  if (custom_entry) {
    auto location = shard_map.Locate(params.entry_point);
    if (!location.ok() || location.value().shard_id != key.source_rank) {
      return Status(StatusCode::kInvalidArgument,
                    "query entry point is not owned by the source rank");
    }
  }
  try {
    auto task = std::unique_ptr<HnswSearchTask>(new HnswSearchTask());
    if (scratch == nullptr) {
      auto owned = QueryScratch::Create(
          meta, params, 1,
          runtime_context == nullptr ? 1U
                                     : runtime_context->ready_producer_count);
      if (!owned.ok()) {
        return owned.status();
      }
      task->owned_scratch_ = std::move(owned).value();
      scratch = task->owned_scratch_.get();
    }
    const auto ready_producer_count =
        runtime_context == nullptr ? 1U : runtime_context->ready_producer_count;
    if (!scratch->impl_->Compatible(meta, params, ready_producer_count)) {
      return Status(StatusCode::kInvalidArgument,
                    "query scratch does not match the search configuration");
    }
    auto acquired =
        scratch->impl_->Acquire(key.generation, query, std::move(query_owner));
    if (!acquired.ok()) {
      return acquired;
    }
    try {
      task->impl_ = std::make_unique<Impl>(key, meta, shard_map, graph_reader,
                                           distance_oracle, *scratch, params,
                                           deadline.value(), runtime_context);
    } catch (...) {
      scratch->impl_->Release(key.generation);
      throw;
    }
    task->impl_->scratch_active = true;
    task->impl_->result.neighbors.reserve(params.topk);
    auto admitted = task->impl_->controller.Admit();
    if (!admitted.ok()) {
      return admitted;
    }
    return task;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate query context");
  }
}

SearchStepResult
HnswSearchTask::Step(std::chrono::steady_clock::time_point now) {
  const auto resumed_at =
      impl_->params.profile_timing ? SteadyNowNanoseconds() : 0;
  if (impl_->state == QueryState::kCompleted) {
    return SearchStepResult::kCompleted;
  }
  if (impl_->state == QueryState::kFailed) {
    return SearchStepResult::kFailed;
  }
  ScopedProfileTimer local_timer(
      impl_->local_phase_timer, impl_->params.profile_timing,
      impl_->result.timing.controller_other_nanoseconds);
  impl_->StartStageTiming(resumed_at);
  if (now >= impl_->query_deadline) {
    if (impl_->ready_publisher == nullptr) {
      impl_->Fail(
          Status(StatusCode::kTimeout, "absolute query deadline exceeded"));
      return SearchStepResult::kFailed;
    }
    auto failed = impl_->controller.RequestFailure(impl_->key.generation,
                                                   TerminalOutcome::kTimedOut,
                                                   StatusCode::kTimeout);
    if (!failed.ok()) {
      impl_->Fail(failed);
      return SearchStepResult::kFailed;
    }
  }

  if (impl_->ready_publisher == nullptr) {
    impl_->ProbeWaiting(now);
  }
  if (impl_->state == QueryState::kFailed) {
    return SearchStepResult::kFailed;
  }
  if (impl_->controller.state() == RunState::kWaiting) {
    return SearchStepResult::kYielded;
  }
  if (!impl_->controller.TryAcquire()) {
    impl_->Fail(Status(StatusCode::kProtocolError,
                       "query step could not acquire single-runner token"));
    return SearchStepResult::kFailed;
  }
  impl_->RecordReadyTiming(resumed_at);
  impl_->ConsumeTerminalResults();

  if (impl_->controller.terminal_outcome() != TerminalOutcome::kNone) {
    impl_->Fail(Status(impl_->controller.failure_code(),
                       "query terminated by scheduler/control"));
    return SearchStepResult::kFailed;
  }
  if (impl_->retry_phase != Impl::RetryPhase::kNone) {
    const auto phase = impl_->retry_phase;
    const auto wait_state = impl_->retry_wait_state;
    // Clear retry state before publishing: a fast completion may already
    // enqueue the owner Search's Ready handle before scheduling returns.
    impl_->retry_phase = Impl::RetryPhase::kNone;
    impl_->retry_handle.reset();
    auto scheduled = phase == Impl::RetryPhase::kGraph
                         ? impl_->ScheduleGraph(wait_state, now)
                         : impl_->ScheduleDistances(wait_state, now);
    auto resolved = impl_->HandleScheduleResult(std::move(scheduled), phase,
                                                wait_state, now);
    if (resolved.has_value()) {
      return *resolved;
    }
  }

  while (impl_->controller.state() == RunState::kRunning) {
    if (impl_->controller.terminal_outcome() != TerminalOutcome::kNone) {
      impl_->Fail(Status(impl_->controller.failure_code(),
                         "query terminated by scheduler/control"));
      return SearchStepResult::kFailed;
    }
    switch (impl_->state) {
    case QueryState::kInit: {
      auto scheduled = impl_->ScheduleEntry(now);
      auto resolved = impl_->HandleScheduleResult(
          std::move(scheduled), Impl::RetryPhase::kDistance,
          QueryState::kWaitEntryDistance, now);
      if (resolved.has_value()) {
        return *resolved;
      }
      break;
    }

    case QueryState::kWaitEntryDistance:
      if (impl_->OperationFailed()) {
        return SearchStepResult::kFailed;
      }
      impl_->current_id = impl_->entry_point;
      impl_->current_distance = impl_->routed_distances[0];
      impl_->current_level = impl_->entry_level;
      if (impl_->current_level > 0) {
        auto marked = impl_->visited.Mark(impl_->entry_point);
        if (!marked.ok() || !marked.value()) {
          impl_->Fail(marked.ok() ? Status(StatusCode::kInternalError,
                                           "unable to seed upper visited table")
                                  : marked.status());
          return SearchStepResult::kFailed;
        }
        impl_->state = QueryState::kReadyUpperGraph;
      } else {
        impl_->StartLayer0();
      }
      break;

    case QueryState::kReadyUpperGraph: {
      auto neighbors = impl_->ReadUpperGraphImmediately();
      if (!neighbors.ok()) {
        impl_->Fail(neighbors.status());
        return SearchStepResult::kFailed;
      }
      auto status = impl_->RouteIds(neighbors.value(), false);
      if (!status.ok()) {
        impl_->Fail(status);
        return SearchStepResult::kFailed;
      }
      status = impl_->FilterUpperVisitedCandidates();
      if (!status.ok()) {
        impl_->Fail(status);
        return SearchStepResult::kFailed;
      }
      impl_->state = QueryState::kReadyUpperDistance;
      break;
    }

    case QueryState::kReadyUpperDistance:
      if (impl_->routed_ids.empty()) {
        impl_->RecordUpperLevelDrop(
            !impl_->upper_had_candidates_before_visited);
        --impl_->current_level;
        if (impl_->current_level > 0) {
          impl_->state = QueryState::kReadyUpperGraph;
        } else {
          impl_->StartLayer0();
        }
        break;
      }
      {
        auto scheduled =
            impl_->ScheduleDistances(QueryState::kWaitUpperDistance, now);
        auto resolved = impl_->HandleScheduleResult(
            std::move(scheduled), Impl::RetryPhase::kDistance,
            QueryState::kWaitUpperDistance, now);
        if (resolved.has_value()) {
          return *resolved;
        }
        break;
      }

    case QueryState::kWaitUpperDistance: {
      ScopedProfileTimer timer(impl_->local_phase_timer,
                               impl_->params.profile_timing,
                               impl_->result.timing.upper_search_nanoseconds);
      if (impl_->OperationFailed()) {
        return SearchStepResult::kFailed;
      }
      HeapEntry best{impl_->current_id, impl_->current_distance};
      for (std::size_t index = 0; index < impl_->routed_ids.size(); ++index) {
        HeapEntry candidate{impl_->routed_ids[index],
                            impl_->routed_distances[index]};
        if (Better(candidate, best)) {
          best = candidate;
        }
      }
      ++impl_->result.profile.upper_steps;
      if (best.id != impl_->current_id) {
        impl_->RecordUpperSameLevelMove();
        impl_->current_id = best.id;
        impl_->current_distance = best.distance;
      } else {
        impl_->RecordUpperLevelDrop(false);
        --impl_->current_level;
      }
      if (impl_->current_level > 0) {
        impl_->state = QueryState::kReadyUpperGraph;
      } else {
        impl_->StartLayer0();
      }
      break;
    }

    case QueryState::kReadyLayer0Expand:
      impl_->expansion_nodes.clear();
      for (std::uint32_t count = 0; count < impl_->params.expand_batch;
           ++count) {
        auto candidate = impl_->heap.PopMinCandidate();
        if (!candidate.has_value()) {
          break;
        }
        impl_->expansion_nodes.push_back({candidate->id, 0});
      }
      if (impl_->expansion_nodes.empty()) {
        return impl_->FinishQuery();
      }
      impl_->result.profile.nodes_expanded += impl_->expansion_nodes.size();
      {
        auto scheduled =
            impl_->ScheduleGraph(QueryState::kWaitLayer0Graph, now);
        auto resolved = impl_->HandleScheduleResult(
            std::move(scheduled), Impl::RetryPhase::kGraph,
            QueryState::kWaitLayer0Graph, now);
        if (resolved.has_value()) {
          return *resolved;
        }
        break;
      }

    case QueryState::kWaitLayer0Graph: {
      if (impl_->OperationFailed()) {
        return SearchStepResult::kFailed;
      }
      if (!impl_->graph_direct_consumed) {
        if (impl_->neighbor_batch.offsets.size() !=
            impl_->expansion_nodes.size() + 1U) {
          impl_->Fail(Status(StatusCode::kProtocolError,
                             "graph result boundaries mismatch"));
          return SearchStepResult::kFailed;
        }
        impl_->result.profile.edges_scanned +=
            impl_->neighbor_batch.neighbors.size();
        auto status = impl_->RouteIds(impl_->neighbor_batch.neighbors, true);
        if (!status.ok()) {
          impl_->Fail(status);
          return SearchStepResult::kFailed;
        }
      }
      if (impl_->routed_ids.empty()) {
        impl_->state = QueryState::kReadyLayer0Expand;
      } else {
        impl_->state = QueryState::kReadyLayer0Distance;
      }
      break;
    }

    case QueryState::kReadyLayer0Distance: {
      auto scheduled =
          impl_->ScheduleDistances(QueryState::kWaitLayer0Distance, now);
      auto resolved = impl_->HandleScheduleResult(
          std::move(scheduled), Impl::RetryPhase::kDistance,
          QueryState::kWaitLayer0Distance, now);
      if (resolved.has_value()) {
        return *resolved;
      }
      break;
    }

    case QueryState::kWaitLayer0Distance:
      if (impl_->OperationFailed()) {
        return SearchStepResult::kFailed;
      }
      impl_->state = QueryState::kReadyCommit;
      break;

    case QueryState::kReadyCommit: {
      ScopedProfileTimer timer(impl_->local_phase_timer,
                               impl_->params.profile_timing,
                               impl_->result.timing.commit_heap_nanoseconds);
      constexpr std::size_t kCommitBatch = 8;
      // Keep validation separate from filtering so the compiler can
      // vectorize the finite checks without the heap-bound branch.
      std::size_t checked = 0;
      for (; checked + kCommitBatch <= impl_->routed_distances.size();
           checked += kCommitBatch) {
        bool finite = true;
        for (std::size_t lane = 0; lane < kCommitBatch; ++lane) {
          finite &= std::isfinite(impl_->routed_distances[checked + lane]);
        }
        if (!finite) {
          impl_->Fail(
              Status(StatusCode::kInvalidArgument,
                     "heap entry requires a valid ID and finite distance"));
          return SearchStepResult::kFailed;
        }
      }
      for (; checked < impl_->routed_distances.size(); ++checked) {
        if (!std::isfinite(impl_->routed_distances[checked])) {
          impl_->Fail(
              Status(StatusCode::kInvalidArgument,
                     "heap entry requires a valid ID and finite distance"));
          return SearchStepResult::kFailed;
        }
      }

      impl_->commit_entries.resize(impl_->routed_ids.size());
      std::size_t competitive_count = 0;
      const auto initial_worst = impl_->heap.max();
      const bool heap_was_full =
          impl_->heap.result_size() == impl_->heap.capacity();
      const auto filter_batch = [&](std::size_t begin, std::size_t count) {
        for (std::size_t lane = 0; lane < count; ++lane) {
          const auto index = begin + lane;
          const HeapEntry entry{impl_->routed_ids[index],
                                impl_->routed_distances[index]};
          // An accepted entry can only improve the current worst
          // result. Reject against the stable pre-commit bound before
          // the deterministic sort.
          if (heap_was_full && initial_worst.has_value() &&
              !Better(entry, *initial_worst)) {
            ++impl_->result.profile.heap_rejections;
            ++impl_->result.profile.heap_prefilter_rejections;
          } else {
            impl_->commit_entries[competitive_count++] = entry;
          }
        }
      };
      std::size_t filtered = 0;
      for (; filtered + kCommitBatch <= impl_->routed_ids.size();
           filtered += kCommitBatch) {
        filter_batch(filtered, kCommitBatch);
      }
      if (filtered < impl_->routed_ids.size()) {
        filter_batch(filtered, impl_->routed_ids.size() - filtered);
      }
      impl_->commit_entries.resize(competitive_count);
      std::sort(impl_->commit_entries.begin(), impl_->commit_entries.end(),
                Better);
      for (std::size_t entry_index = 0;
           entry_index < impl_->commit_entries.size(); ++entry_index) {
        const auto &entry = impl_->commit_entries[entry_index];
        const auto worst = impl_->heap.max();
        if (impl_->heap.result_size() == impl_->heap.capacity() &&
            worst.has_value() && !Better(entry, *worst)) {
          impl_->result.profile.heap_rejections +=
              impl_->commit_entries.size() - entry_index;
          impl_->result.profile.heap_suffix_rejections +=
              impl_->commit_entries.size() - entry_index;
          break;
        }

        // RouteIds() already marked these IDs in VisitedTable. Therefore
        // this sorted commit batch contains no duplicate heap IDs.
        const bool pushed_into_full_heap =
            impl_->heap.result_size() == impl_->heap.capacity();
        auto pushed = impl_->heap.PushUnique(entry.id, entry.distance);
        if (!pushed.ok()) {
          impl_->Fail(pushed.status());
          return SearchStepResult::kFailed;
        }
        if (pushed.value()) {
          ++impl_->result.profile.heap_pushes;
          if (pushed_into_full_heap) {
            ++impl_->result.profile.heap_full_pushes;
          }
        } else {
          ++impl_->result.profile.heap_rejections;
        }
      }
      impl_->result.profile.local_distance_count += impl_->routed_ids.size();
      impl_->state = QueryState::kReadyLayer0Expand;
      break;
    }

    case QueryState::kCompleted:
      return SearchStepResult::kCompleted;
    case QueryState::kFailed:
      return SearchStepResult::kFailed;
    }
  }
  return SearchStepResult::kYielded;
}

Result<bool>
HnswSearchTask::ProgressWaiting(std::chrono::steady_clock::time_point now) {
  // Compatibility polling is gated until the Search Worker has published all
  // legacy tickets and phase metadata.
  if (!impl_->wait_published.load(std::memory_order_acquire) ||
      impl_->controller.state() != RunState::kWaiting) {
    return false;
  }
  if (now >= impl_->query_deadline) {
    impl_->Fail(
        Status(StatusCode::kTimeout, "absolute query deadline exceeded"));
    return impl_->final_status;
  }
  impl_->ProbeWaiting(now);
  if (impl_->state == QueryState::kFailed) {
    return impl_->final_status;
  }
  const bool ready = impl_->controller.state() == RunState::kQueued;
  if (ready) {
    impl_->wait_published.store(false, std::memory_order_release);
  }
  return ready;
}

QueryState HnswSearchTask::query_state() const noexcept { return impl_->state; }
RunState HnswSearchTask::run_state() const noexcept {
  return impl_->controller.state();
}
bool HnswSearchTask::requires_runtime_abort() const noexcept {
  return impl_->runtime_abort_required;
}
const Status &HnswSearchTask::status() const noexcept {
  return impl_->final_status;
}
const SearchResult &HnswSearchTask::result() const noexcept {
  return impl_->result;
}
Status HnswSearchTask::BindOwner(std::uint32_t owner) noexcept {
  return impl_->controller.BindOwner(owner);
}

const QueryRunController &HnswSearchTask::run_controller() const noexcept {
  return impl_->controller;
}
Status HnswSearchTask::RequestFailure(TerminalOutcome outcome,
                                      StatusCode code) noexcept {
  return impl_->controller.RequestFailure(impl_->key.generation, outcome, code);
}
std::optional<QueryRetryHandle> HnswSearchTask::TakeRetryHandle() noexcept {
  auto handle = impl_->retry_handle;
  impl_->retry_handle.reset();
  return handle;
}
QueryKey HnswSearchTask::key() const noexcept { return impl_->key; }

Result<SearchResult> RunToCompletion(std::unique_ptr<HnswSearchTask> task,
                                     std::uint64_t max_steps) {
  if (!task) {
    return Status(StatusCode::kInvalidArgument, "search task is null");
  }
  for (std::uint64_t step = 0; step < max_steps; ++step) {
    const auto state = task->Step();
    if (state == SearchStepResult::kCompleted) {
      return task->result();
    }
    if (state == SearchStepResult::kFailed) {
      return task->status();
    }
  }
  return Status(StatusCode::kTimeout,
                "search exceeded local maximum step count");
}

} // namespace hnsw_urma
