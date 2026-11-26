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

#include "hnsw_urma/core/log.hpp"
#include "hnsw_urma/milvus_ub_runtime_impl.hpp"
#include <array>
#include <numeric>
#include <thread>

namespace hnsw_urma {
namespace {
#if HNSW_URMA_ENABLE_URMA
std::uint64_t ReadActivity(const ReadBatchProfile &p) {
  return p.batches_submitted + p.requests_completed + p.batches_failed +
         p.batches_timed_out + p.batches_cancelled + p.late_completion_count;
}
std::uint64_t RpcActivity(const RpcTransportProfile &p) {
  return p.request_physical_writes + p.response_physical_writes +
         p.request_completions + p.response_completions +
         p.response_batches_received + p.result_slot_hits +
         p.response_scan_hits;
}
#endif
} // namespace

Status MilvusUbRuntime::EnableDistributedSearch(
    std::chrono::steady_clock::time_point deadline) {
  std::unique_lock lock(state_mutex_, std::defer_lock);
  while (!lock.try_lock()) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline)
      return Status(StatusCode::kTimeout,
                    "UB startup expired waiting for runtime state");
    std::this_thread::sleep_for(std::min(
        deadline - now,
        std::chrono::steady_clock::duration(std::chrono::milliseconds(1))));
  }
  auto status = EnableDistributedSearchLocked(deadline);
  if (!status.ok())
    LOG_ERROR("UB service startup failed: {}", status.ToString());
  return status;
}

Status MilvusUbRuntime::EnableDistributedSearchLocked(
    std::chrono::steady_clock::time_point deadline) {
  const auto expired = [&] {
    return std::chrono::steady_clock::now() >= deadline;
  };
  const auto timeout = [] {
    return Status(StatusCode::kTimeout,
                  "UB distributed service startup expired");
  };
  if (expired())
    return timeout();
#if HNSW_URMA_ENABLE_URMA
  if (distributed_runtime_)
    return distributed_runtime_->threads->status();
  if (!meta_ || local_shards_.size() != 1 || !rpc_memory_) {
    return Status(StatusCode::kInvalidArgument,
                  "UB distributed service requires one local shard and its "
                  "published RPC mailbox");
  }
  auto requested = RequestedRuntimeConfig();
  if (!requested.ok())
    return requested.status();
  const auto rank = local_shards_.front()->range().shard_id;
  const auto world = meta_->num_shards;
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    if (peer == rank)
      continue;
    for (std::uint32_t role = 0; role < 5; ++role) {
      if (!peer_handles_.count({peer, role})) {
        return Status(StatusCode::kInvalidArgument,
                      "UB peer resources are incomplete")
            .AddContext("shard", std::to_string(peer))
            .AddContext("role", std::to_string(role));
      }
    }
  }
  auto resolved = ResolveRuntimeConfig(requested.value(), *meta_, world,
                                       knobs_.allowed_cpus, rank);
  if (!resolved.ok())
    return resolved.status();
  auto state = std::make_unique<DistributedState>();
  state->requested = requested.value();
  state->effective = std::move(resolved).value();
  const auto &config = state->requested;
  const auto &effective = state->effective;
  const auto communication =
      effective.read_progress_threads + effective.rpc_progress_threads;
  const auto maximum_batch = effective.maximum_distance_batch;
  const auto &local = *local_shards_.front();

  UmdkBackendOptions options;
  options.device_name = effective.urma_device;
  options.eid_index = effective.eid_index;
  options.tp_type =
      config.tp_type == TpType::kCtp ? UmdkTpType::kCtp : UmdkTpType::kRtp;
  options.configure_bonding = true;
  options.external_context = injected_urma_context_;
  if (!options.external_context)
    return Status(StatusCode::kInvalidArgument,
                  "UrmaManager context is unavailable");
  auto backend = UmdkResourceBackend::Create(options);
  if (!backend.ok())
    return backend.status();
  state->backend = std::move(backend).value();
  if (expired())
    return timeout();
  auto resource_options = effective.urma;
  resource_options.duplex_jetty = true;
  auto plan =
      PlanUrmaResources(state->backend->capabilities(), resource_options);
  if (!plan.ok())
    return plan.status();
  const auto slot_bytes = std::max<std::size_t>(meta_->M0 * sizeof(node_id_t),
                                                meta_->dim * sizeof(float));
  auto pool_bytes = SearchReadResources::RequiredBytes(
      effective.read_result_slots_per_search, slot_bytes);
  if (!pool_bytes.ok())
    return pool_bytes.status();
  auto buffer =
      PageAlignedBuffer::Create(effective.registered_read_result_bytes);
  if (!buffer.ok())
    return buffer.status();
  state->read_buffer.emplace(std::move(buffer).value());
  const auto writable = [](std::span<const std::byte> span) {
    return std::span<std::byte>(const_cast<std::byte *>(span.data()),
                                span.size());
  };
  const std::array<std::span<std::byte>, 6> logical{
      writable(local.graph_logical_memory()),
      writable(local.vector_logical_memory()),
      rpc_memory_->requests(),
      rpc_memory_->responses(),
      rpc_memory_->notify(),
      state->read_buffer->logical_memory()};
  const std::array<std::span<std::byte>, 6> registered{
      writable(local.graph_registration_memory()),
      writable(local.vector_registration_memory()),
      rpc_memory_->request_registration(),
      rpc_memory_->response_registration(),
      rpc_memory_->notify_registration(),
      state->read_buffer->registration_memory()};
  constexpr std::array<std::string_view, 6> names{
      "layer0_graph",  "vectors",    "rpc_requests",
      "rpc_responses", "rpc_notify", "read_results"};
  std::array<UrmaSegmentRegistration, 6> registrations;
  for (std::size_t i = 0; i < registrations.size(); ++i) {
    registrations[i] = {registered[i].data(), registered[i].size(), names[i],
                        logical[i].size()};
  }
  auto bundle = UrmaResourceBundle::CreateLocal(*state->backend, plan.value(),
                                                registrations);
  if (!bundle.ok())
    return bundle.status();
  state->bundle.emplace(std::move(bundle).value());
  if (expired())
    return timeout();

  std::vector<std::vector<UmdkPeerBinding>> bindings(communication);
  std::vector<PeerDataSegments> peer_data;
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    if (expired())
      return timeout();
    if (peer == rank)
      continue;
    const auto range = shard_map_->Range(peer).value();
    const auto graph_bytes =
        std::uint64_t(range.local_count) * meta_->M0 * sizeof(node_id_t);
    const auto vector_bytes =
        std::uint64_t(range.local_count) * meta_->dim * sizeof(float);
    const std::array<std::uint64_t, 5> sizes{
        graph_bytes, vector_bytes, logical[2].size(), logical[3].size(),
        logical[4].size()};
    const std::array<SegmentKind, 5> kinds{
        SegmentKind::kLayer0Graph, SegmentKind::kVectors,
        SegmentKind::kRequests, SegmentKind::kResponses, SegmentKind::kNotify};
    std::vector<UmdkRemoteSegmentBinding> segments;
    for (std::uint32_t role = 0; role < 5; ++role) {
      if (expired())
        return timeout();
      auto imported = state->backend->ImportExternal(
          UrmaResourceKind::kImportedSegment,
          peer_handles_.at({peer, role}).segment);
      if (!imported.ok())
        return imported.status();
      segments.push_back({kinds[role], imported.value(), sizes[role]});
    }
    const auto rpc_owner = effective.owners.rpc_owner_by_rank[peer];
    const auto read_lane = effective.owners.read_owner_by_rank[peer];
    const auto rpc_lane = effective.read_progress_threads + rpc_owner;
    bindings[read_lane].push_back({peer, segments, {}});
    bindings[rpc_lane].push_back({peer, segments, {}});
    for (auto operation : {OperationClass::kRead, OperationClass::kRequestWrite,
                           OperationClass::kResponseWrite}) {
      if (expired())
        return timeout();
      const bool read = operation == OperationClass::kRead;
      const bool request = operation == OperationClass::kRequestWrite;
      const auto lane = read ? read_lane : rpc_lane;
      const auto role = read ? 0U : request ? 2U : 3U;
      const auto &handle = peer_handles_.at({peer, role});
      const auto controls = state->bundle->ControlLaneResources(lane).value();
      std::uint32_t queue = 0;
      if (!read) {
        const auto &peers = effective.owners.rpc_owned_ranks[rpc_owner];
        queue = std::distance(peers.begin(),
                              std::find(peers.begin(), peers.end(), peer)) %
                (request ? effective.urma.request_jfs_per_rpc_lane
                         : effective.urma.response_jfs_per_rpc_lane);
      }
      UrmaResourceRequest import;
      import.kind = UrmaResourceKind::kImportedEndpoint;
      import.operation_class = operation;
      import.descriptor = std::span<const std::byte>(handle.rjetty_descriptor);
      import.token = handle.jetty_token;
      import.secondary_dependency = read      ? controls.read_jfs
                                    : request ? controls.request_jfs[queue]
                                              : controls.response_jfs[queue];
      auto endpoint = state->bundle->CreatePeerResource(import);
      if (!endpoint.ok())
        return endpoint.status();
      bindings[lane].back().endpoints.push_back(
          {operation, endpoint.value(), queue});
      if (state->backend->requires_endpoint_advice()) {
        UrmaResourceRequest advice;
        advice.kind = UrmaResourceKind::kAdvice;
        advice.dependency = endpoint.value();
        advice.secondary_dependency = import.secondary_dependency;
        auto advised = state->bundle->CreatePeerResource(advice);
        if (!advised.ok())
          return advised.status();
      }
    }
    peer_data.push_back({peer, peer, graph_bytes, vector_bytes});
  }

  const auto handles = state->bundle->registered_segments();
  std::vector<ReadBatchLane *> read_views;
  for (std::uint32_t lane = 0; lane < communication; ++lane) {
    if (expired())
      return timeout();
    const bool read = lane < effective.read_progress_threads;
    auto resources = state->bundle->ClaimLane(lane);
    if (!resources.ok())
      return resources.status();
    UmdkLaneOptions lane_options;
    lane_options.owner_lane = lane;
    lane_options.resources = resources.value();
    lane_options.peers = std::move(bindings[lane]);
    lane_options.max_outstanding =
        read ? effective.urma.read_jfs_depth
             : std::size_t(
                   effective.rpc_memory.request_staging_slots_per_thread) +
                   effective.urma.response_jfs_depth;
    lane_options.wr_list_max = config.wr_list_max;
    lane_options.poll_batch = 16;
    lane_options.profiling_enabled = config.profile;
    if (read)
      lane_options.local_segments.push_back({handles[5], logical[5]});
    else
      for (std::size_t i = 2; i <= 4; ++i)
        lane_options.local_segments.push_back({handles[i], logical[i]});
    auto engine =
        UmdkLanePostEngine::Create(*state->backend, std::move(lane_options));
    if (!engine.ok())
      return engine.status();
    state->engines.push_back(std::move(engine).value());
    if (read) {
      ReadBatchLaneConfig lane_config;
      lane_config.search_workers = effective.search_workers;
      lane_config.peer_count = world;
      lane_config.submit_depth = effective.queues.depths.read_submit;
      lane_config.records_per_search = effective.queues.depths.read_reclaim;
      lane_config.in_flight_limit = effective.urma.read_jfs_depth;
      lane_config.wr_list_max = config.wr_list_max;
      lane_config.profiling_enabled = config.profile;
      auto dispatcher =
          ReadBatchLane::Create(*state->engines.back(), lane, lane_config);
      if (!dispatcher.ok())
        return dispatcher.status();
      dispatcher.value()->SetAdaptiveNotifications(config.read_progress_mode ==
                                                   ProgressMode::kAdaptive);
      read_views.push_back(dispatcher.value().get());
      state->read_lanes.push_back(std::move(dispatcher).value());
    }
  }
  std::vector<SearchReadResources *> pool_views;
  for (std::uint32_t search = 0; search < effective.search_workers; ++search) {
    if (expired())
      return timeout();
    auto pool = SearchReadResources::Create(
        search,
        logical[5].subspan(search * pool_bytes.value(), pool_bytes.value()),
        effective.read_result_slots_per_search, slot_bytes);
    if (!pool.ok())
      return pool.status();
    auto bound = pool.value()->BindLanes(read_views);
    if (!bound.ok())
      return bound;
    pool_views.push_back(pool.value().get());
    state->read_pools.push_back(std::move(pool).value());
  }
  auto table = PeerSegmentTable::Create(*meta_, *shard_map_, rank, peer_data);
  if (!table.ok())
    return table.status();
  state->segments.emplace(std::move(table).value());
  auto graph = UrmaGraphReader::Create(
      *meta_, *shard_map_, *upper_, local, *state->segments, read_views,
      effective.owners.read_owner_by_rank, pool_views,
      effective.read_graph_records_per_search);
  if (!graph.ok())
    return graph.status();
  state->graph.emplace(std::move(graph).value());
  auto pull = PullVectorDistanceOracle::Create(
      *meta_, *shard_map_, local, *state->segments, read_views,
      effective.owners.read_owner_by_rank, pool_views,
      DistanceImplementation::kAuto, effective.read_pull_records_per_search,
      maximum_batch, config.use_dot_norm);
  if (!pull.ok())
    return pull.status();
  state->pull.emplace(std::move(pull).value());
  auto queues = SchedulerQueues::Create(
      effective.search_workers, communication, config.max_active_queries,
      config.max_active_queries, config.profile,
      effective.rpc_operations.slots_per_search);
  if (!queues.ok())
    return queues.status();
  state->queues = std::move(queues).value();
  std::vector<std::uint32_t> shard_to_rank(world);
  std::iota(shard_to_rank.begin(), shard_to_rank.end(), 0);
  auto rpc = RpcDistanceOracle::Create(
      effective.search_workers, effective.rpc_progress_threads,
      effective.queues.depths.rpc_submit,
      effective.rpc_operations.slots_per_search, maximum_batch, meta_->dim,
      rank, shard_to_rank, effective.owners.rpc_owner_by_rank,
      effective.read_progress_threads, *state->queues, config.use_dot_norm);
  if (!rpc.ok())
    return rpc.status();
  state->rpc = std::move(rpc).value();
  state->rpc->submissions().SetAdaptiveNotifications(config.rpc_progress_mode ==
                                                     ProgressMode::kAdaptive);
  auto server = RpcServerRuntime::Create(
      *rpc_memory_, effective.distance_workers, maximum_batch, rank,
      shard_to_rank, effective.owners.rpc_owner_by_rank,
      config.max_active_queries, meta_->dim, config.profile, &local);
  if (!server.ok())
    return server.status();
  state->server = std::move(server).value();
  state->server->SetResultAdaptiveNotifications(config.rpc_progress_mode ==
                                                ProgressMode::kAdaptive);
  for (std::uint32_t owner = 0; owner < effective.rpc_progress_threads;
       ++owner) {
    if (expired())
      return timeout();
    auto client = RpcClientOwner::Create(
        owner, world, effective.rpc_operations.total_slots,
        state->rpc->submissions(), config.max_active_queries,
        config.request_ring_depth, config.profile, state->rpc.get());
    if (!client.ok())
      return client.status();
    state->clients.push_back(std::move(client).value());
    RpcTransportConfig transport_config;
    transport_config.owner_thread = owner;
    transport_config.transport_lane = effective.read_progress_threads + owner;
    transport_config.local_rank = rank;
    transport_config.request_batch_max_tasks =
        effective.rpc_request_batch_max_tasks;
    transport_config.request_batch_max_bytes =
        effective.rpc_request_batch_max_bytes;
    transport_config.response_progress_max_bytes =
        effective.rpc_response_progress_max_bytes;
    transport_config.profiling_enabled = config.profile;
    transport_config.poll_batch = 16;
    transport_config.phase_budget = 64;
    auto transport = RpcTransportOwner::Create(
        transport_config, effective.owners.rpc_owned_ranks[owner],
        *state->engines[transport_config.transport_lane], *state->backend,
        *rpc_memory_, *state->clients.back(), *state->server);
    if (!transport.ok())
      return transport.status();
    state->transports.push_back(std::move(transport).value());
  }
  auto distance = HybridDistanceOracle::Create(rank, config.distance_policy,
                                               config.pull_threshold,
                                               *state->pull, *state->rpc);
  if (!distance.ok())
    return distance.status();
  state->distance = std::move(distance).value();
  auto drain =
      DataPlaneDrainTracker::Create(communication + effective.search_workers);
  if (!drain.ok())
    return drain.status();
  state->drain = std::move(drain).value();
  for (std::uint32_t lane = 0; lane < communication + effective.search_workers;
       ++lane) {
    LaneDrainSnapshot pending;
    pending.read_active = 1;
    state->drain->PublishLane(lane, pending);
  }
  auto executor = MilvusQueryExecutor::Create(
      *meta_, *shard_map_, shard_entries_.at(rank), *state->graph,
      *state->distance, config, effective, rank, *state->queues,
      state->rpc.get(), pool_views, state->drain.get(),
      [s = state.get()](const MilvusQueryExecutor::Profile &p) {
        // Worker profiles expose atomic counters. No polling of another lane's
        // non-atomic transport profile and no per-query logging.
        std::uint64_t distance_values = 0, compute_ns = 0, distance_tasks = 0;
        for (std::uint32_t worker = 0; worker < s->effective.distance_workers;
             ++worker) {
          const auto d = s->server->worker_profile(worker);
          distance_values += d.distance_values;
          compute_ns += d.distance_compute_nanoseconds;
          distance_tasks += d.request_tasks;
        }
        LOG_INFO(
            "[UB service_profile] completed={} errors={} pre_admit_ns={} "
            "active_ns={} execution_ns={} "
            "graph_wait_ns={} distance_wait_ns={} ready_wait_ns={} upper_ns={} "
            "layer0_ns={} "
            "nodes={} edges={} active={} pending={} retiring={} quarantined={} "
            "distance_values={} compute_ns={} distance_tasks={}",
            p.completed, p.errors, p.pre_admit_ns, p.active_ns, p.execution_ns,
            p.graph_wait_ns, p.distance_wait_ns, p.ready_wait_ns, p.upper_ns,
            p.layer0_ns, p.nodes, p.edges, p.active, p.pending, p.retiring,
            p.quarantined, distance_values, compute_ns, distance_tasks);
      });
  if (!executor.ok())
    return executor.status();
  state->executor = std::move(executor).value();

  auto *s = state.get();
  RuntimeCallbacks callbacks;
  callbacks.control = [s](std::stop_token stop, std::uint32_t) {
    auto status = s->executor->Control(stop);
    if (!status.ok())
      s->executor->FailPending(status);
    return status;
  };
  callbacks.search = [s](std::stop_token stop, std::uint32_t worker) {
    return s->executor->Search(stop, worker);
  };
  callbacks.distance = [s, metric = static_cast<Metric>(meta_->metric)](
                           std::stop_token stop, std::uint32_t worker) {
    while (!stop.stop_requested()) {
      auto processed =
          s->server->PollAndProcessRequestSlot(worker, metric, *s->backend);
      if (!processed.ok())
        return processed.status();
    }
    return Status::Ok();
  };
  callbacks.read_progress = [s](std::stop_token stop, std::uint32_t lane) {
    auto last = std::chrono::steady_clock::now();
    auto &publisher = s->executor->publisher();
    auto &read = *s->read_lanes[lane];
    auto next_report = std::chrono::steady_clock::time_point{};
    while (!stop.stop_requested()) {
      const auto before = ReadActivity(read.profile());
      auto progressed = read.Progress();
      if (!progressed.ok())
        return progressed;
      auto flushed = publisher.Flush(lane, s->effective.search_workers);
      if (!flushed.ok())
        return flushed.status();
      const auto now = std::chrono::steady_clock::now();
      if (before != ReadActivity(read.profile()) || flushed.value())
        last = now;
      LaneDrainSnapshot snapshot;
      snapshot.read_active =
          read.drained() && publisher.pending(lane) == 0 ? 0 : 1;
      snapshot.quarantine_records = read.profile().exceptional_batches;
      s->drain->PublishLane(lane, snapshot);
      if (now >= next_report) {
        const auto p = read.profile();
        LOG_INFO("[UB read_progress {}] hb submitted={} completed={} bytes={} "
                 "cqe_error={} timeout_count={}",
                 lane, p.requests_posted, p.requests_completed,
                 p.bytes_completed, p.cqe_failure_count, p.batches_timed_out);
        if (s->requested.profile) {
          LOG_INFO("[UB read_profile {}] batches={} wr_samples={} wr_ns={} "
                   "submit_ns={} queue_full={}",
                   lane, p.batches_submitted, p.wr_completion_samples,
                   p.wr_completion_nanoseconds, p.submit_wait_nanoseconds,
                   p.submit_queue_full_count);
        }
        next_report = now + std::chrono::seconds(1);
      }
      if (s->requested.read_progress_mode == ProgressMode::kAdaptive &&
          publisher.pending(lane) == 0 &&
          now - last >=
              std::chrono::microseconds(s->effective.read_progress_spin_us)) {
        (void)read.adaptive_ready().WaitUntil(
            stop, now + std::chrono::microseconds(100));
      }
    }
    return Status::Ok();
  };
  callbacks.rpc_progress = [s](std::stop_token stop, std::uint32_t owner) {
    auto last = std::chrono::steady_clock::now();
    const auto producer = s->effective.read_progress_threads + owner;
    auto &publisher = s->executor->publisher();
    auto &transport = *s->transports[owner];
    auto next_report = std::chrono::steady_clock::time_point{};
    while (!stop.stop_requested()) {
      const auto before = RpcActivity(transport.profile());
      auto progressed = transport.Progress();
      if (!progressed.ok())
        return progressed;
      auto audited = s->rpc->AuditOwnerControl(owner, *s->clients[owner]);
      if (!audited.ok())
        return audited.status();
      auto flushed = publisher.Flush(producer, s->effective.search_workers);
      if (!flushed.ok())
        return flushed.status();
      const auto now = std::chrono::steady_clock::now();
      if (before != RpcActivity(transport.profile()) || audited.value() ||
          flushed.value())
        last = now;
      LaneDrainSnapshot snapshot;
      snapshot.rpc_active =
          transport.drained() && publisher.pending(producer) == 0 ? 0 : 1;
      s->drain->PublishLane(producer, snapshot);
      if (now >= next_report) {
        const auto p = transport.profile();
        LOG_INFO("[UB rpc_progress {}] profile req_post={} req_comp={} "
                 "result_slot_hits={} resp_post={} resp_comp={} "
                 "resp_batch_recv={} fatal={}",
                 owner, p.request_physical_writes, p.request_completions,
                 p.result_slot_hits, p.response_physical_writes,
                 p.response_completions, p.response_batches_received,
                 static_cast<int>(transport.fatal_status().code()));
        if (s->requested.profile) {
          const auto c = s->clients[owner]->profile();
          LOG_INFO("[UB rpc_profile {}] tasks={} queue_samples={} queue_ns={} "
                   "pending_ns={} post_wait_ns={} "
                   "rtt_samples={} rtt_ns={} slot_wait={} staging_wait={} "
                   "tx_wait={} provider_wait={} "
                   "ready_samples={} ready_ns={}",
                   owner, c.logical_tasks, c.owner_queue_samples,
                   c.owner_queue_nanoseconds, c.owner_pending_nanoseconds,
                   c.request_post_wait_nanoseconds, c.round_trip_samples,
                   c.round_trip_nanoseconds, p.request_post_free_slot_waits,
                   p.request_post_staging_unavailable,
                   p.request_post_tx_unavailable,
                   p.request_post_provider_backpressure_batches,
                   p.result_ready_to_post_samples,
                   p.result_ready_to_post_nanoseconds);
        }
        next_report = now + std::chrono::seconds(1);
      }
      if (s->requested.rpc_progress_mode == ProgressMode::kAdaptive &&
          publisher.pending(producer) == 0 && s->clients[owner]->drained() &&
          now - last >=
              std::chrono::microseconds(s->effective.rpc_progress_spin_us)) {
        auto &ready = s->rpc->submissions().ready(owner);
        if (!ready.TryConsume() &&
            !s->server->result_ready(owner).TryConsume()) {
          (void)ready.WaitUntil(stop, now + std::chrono::microseconds(100));
        }
      }
    }
    return Status::Ok();
  };
  if (expired())
    return timeout();
  auto threads = RuntimeThreadGroup::Start(effective, config.cpu_bind,
                                           std::move(callbacks));
  if (!threads.ok())
    return threads.status();
  state->threads = std::move(threads).value();
  auto canonical = BuildCanonicalRuntimeConfig(config, effective);
  if (!canonical.ok())
    return canonical.status();
  for (const auto &[key, value] : canonical.value().values())
    LOG_INFO("UB effective {}={}", key, value);
  LOG_INFO("UB aligned plan expand_batch={} maximum_batch={} dim={} "
           "distance_policy={} use_dot_norm={} layer0_sort={} rpc_layout={}",
           config.expand_batch, maximum_batch, meta_->dim,
           ToString(config.distance_policy), config.use_dot_norm,
           ToString(config.layer0_sort_mode),
           std::to_string(DISTANCE_BATCH_WIRE_VERSION) + ":" +
               std::to_string(rpc_memory_->config().ring_depth));
  LOG_INFO("UB aligned service ready reference={} rank={} entry={} "
           "entry_level={} eid_index={}",
           GitCommit(), rank, shard_entries_.at(rank).global_id,
           shard_entries_.at(rank).level, state->backend->selected_eid_index());
  distributed_runtime_ = std::move(state);
  // Once threads start, retain their resources on timeout so normal Stop/drain
  // can retire peer work safely. Go never publishes Ready for this result.
  if (expired())
    return timeout();
  return Status::Ok();
#else
  return Status(StatusCode::kInvalidArgument,
                "Milvus was built without URMA support");
#endif
}
} // namespace hnsw_urma
