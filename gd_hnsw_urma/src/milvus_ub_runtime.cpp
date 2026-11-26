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

// Milvus ownership, configuration and request boundary for pinned HNSW.
#include "hnsw_urma/core/log.hpp"

#include "hnsw_urma/milvus_ub_runtime.hpp"
#include "hnsw_urma/milvus_ub_runtime_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/runtime/topology.hpp"
#include "hnsw_urma/search/local_io.hpp"
#include "hnsw_urma/search/query_context.hpp"
#include "hnsw_urma/search/search.hpp"

namespace hnsw_urma {

MilvusUbRuntime &MilvusUbRuntime::Instance() {
  // Explicit QueryNode cleanup orders registrations before their backing data.
  static auto *instance = new MilvusUbRuntime;
  return *instance;
}

MilvusUbRuntime::Knobs MilvusUbRuntime::GetKnobs() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return knobs_;
}

void MilvusUbRuntime::SetKnobs(const Knobs &knobs) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  knobs_ = knobs;
  LOG_INFO("UB knobs staged: search_workers={} distance_workers={} "
           "read_progress_workers={} rpc_progress_workers={} cpu_bind={} "
           "cpu_layout={} progress_mode={} allowed_cpus={}",
           knobs_.search_workers, knobs_.distance_workers,
           knobs_.read_progress_workers, knobs_.rpc_progress_workers,
           knobs_.cpu_bind_mode, knobs_.cpu_layout, knobs_.progress_mode,
           knobs_.allowed_cpus.size());
}

void MilvusUbRuntime::ResetLocked() {
  local_shards_.clear();
  loaded_shard_ids_.clear();
  shard_entries_.clear();
  upper_.reset();
  shard_map_.reset();
  meta_.reset();
  rpc_memory_.reset();
  graph_id_.clear();
  graph_epoch_ = 0;
}

Status MilvusUbRuntime::BindGlobalGraphIdentity(std::string graph_id,
                                                std::uint64_t epoch) {
  std::lock_guard lock(state_mutex_);
  if (!meta_ || local_shards_.empty() || graph_id.empty() || epoch == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "UB graph identity requires a seeded graph, id and epoch");
  }
  if (!graph_id_.empty() && (graph_id_ != graph_id || graph_epoch_ != epoch)) {
    return Status(StatusCode::kInvalidArgument,
                  "UB graph identity differs; successful shutdown is required");
  }
  graph_id_ = std::move(graph_id);
  graph_epoch_ = epoch;
  return Status::Ok();
}

Result<std::optional<MilvusUbRuntime::SeededShardInfo>>
MilvusUbRuntime::ReuseGlobalGraphShard(const std::string &graph_id,
                                       std::uint64_t epoch,
                                       shard_id_t shard) const {
  std::lock_guard lock(state_mutex_);
  if (graph_id_.empty())
    return std::optional<SeededShardInfo>{};
  if (graph_id_ != graph_id || graph_epoch_ != epoch) {
    return Status(
        StatusCode::kInvalidArgument,
        "UB segment artifact graph/epoch differs from the seeded graph");
  }
  for (const auto &local : local_shards_) {
    if (local->range().shard_id == shard) {
      return std::optional<SeededShardInfo>{
          {meta_->dim, local->range().local_count}};
    }
  }
  return Status(StatusCode::kInvalidArgument,
                "UB segment target shard is not seeded locally");
}

Status MilvusUbRuntime::LoadShard(std::unique_ptr<LoadedIndex> loaded) {
  if (!loaded) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::LoadShard: null loaded index");
  }
  std::lock_guard<std::mutex> lock(state_mutex_);

  // First shard seeds the shared meta/shard_map/upper.
  if (!meta_) {
    meta_ = std::unique_ptr<GlobalIndexMeta>(new GlobalIndexMeta(loaded->meta));
    shard_map_ =
        std::unique_ptr<ShardMap>(new ShardMap(std::move(loaded->shard_map)));
    upper_ = std::unique_ptr<UpperGraphReplica>(
        new UpperGraphReplica(std::move(loaded->upper_graph)));
  } else {
    // Subsequent shards must agree with the seeded meta/shard_map/upper
    // (they are shared across segments of one collection).
    if (!(loaded->meta == *meta_)) {
      return Status(StatusCode::kInvalidArgument,
                    "MilvusUbRuntime::LoadShard: meta mismatch across shards");
    }
    // ShardMap has no operator==; compare ranges() span + ntotal.
    if (shard_map_->ntotal() != loaded->shard_map.ntotal() ||
        shard_map_->ranges().size() != loaded->shard_map.ranges().size()) {
      return Status(StatusCode::kInvalidArgument,
                    "MilvusUbRuntime::LoadShard: shard_map size mismatch");
    }
    for (std::size_t i = 0; i < shard_map_->ranges().size(); ++i) {
      if (!(shard_map_->ranges()[i] == loaded->shard_map.ranges()[i])) {
        return Status(StatusCode::kInvalidArgument,
                      "MilvusUbRuntime::LoadShard: shard_map range mismatch");
      }
    }
    // UpperGraphReplica has no public equality; skip deep compare. The
    // external build guarantees identical upper graphs across shards.
  }

  // Record shard_id and move the LocalShard into our vector.
  shard_id_t sid = loaded->local_shard.range().shard_id;
  for (auto id : loaded_shard_ids_) {
    if (id == sid) {
      // Idempotent: this shard is already registered (e.g. re-load after a
      // release, or a duplicate LoadSegment for the same segment). The index
      // data was already checked to match meta/shard_map above, so just
      // acknowledge.
      return Status::Ok();
    }
  }

  if (distributed_runtime_) {
    return Status(StatusCode::kInvalidArgument,
                  "stop the UB service before adding a shard");
  }

  std::unique_ptr<LocalShard> shard(
      new LocalShard(std::move(loaded->local_shard)));
  shard_entries_[sid] = loaded->local_entry_point;
  local_shards_.push_back(std::move(shard));
  loaded_shard_ids_.push_back(sid);

  return Status::Ok();
}

Status MilvusUbRuntime::SetLocalShard(shard_id_t shard_id) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  const bool loaded =
      std::find(loaded_shard_ids_.begin(), loaded_shard_ids_.end(), shard_id) !=
      loaded_shard_ids_.end();
  if (!loaded) {
    return Status(
        StatusCode::kInvalidArgument,
        "MilvusUbRuntime::SetLocalShard: shard is not loaded locally");
  }
  if (distributed_runtime_) {
    return Status::Ok();
  }
  // One worker owns one shard. A segment load may have registered the shards
  // its index files carried before this process selected its own; drop those
  // so EnableDistributedSearch starts with exactly one local shard. The
  // metadata/shard map are shared and stay; only the vector data of the
  // dropped shards is released (peers serve it over URMA).
  if (loaded_shard_ids_.size() > 1) {
    std::vector<std::unique_ptr<LocalShard>> kept;
    std::vector<shard_id_t> kept_ids;
    kept.reserve(1);
    kept_ids.reserve(1);
    for (std::size_t index = 0; index < loaded_shard_ids_.size(); ++index) {
      const auto candidate = loaded_shard_ids_[index];
      if (candidate == shard_id) {
        kept.push_back(std::move(local_shards_[index]));
        kept_ids.push_back(candidate);
        continue;
      }
      shard_entries_.erase(candidate);
      LOG_INFO("UB local shard pinned to {}; dropping locally loaded shard {}",
               shard_id, candidate);
    }
    local_shards_ = std::move(kept);
    loaded_shard_ids_ = std::move(kept_ids);
  }
  return Status::Ok();
}

std::optional<shard_id_t> MilvusUbRuntime::LocalShardId() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (loaded_shard_ids_.size() == 1) {
    return loaded_shard_ids_.front();
  }
  return std::nullopt;
}

void MilvusUbRuntime::SetInjectedUrmaContext(void *urma_context) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  injected_urma_context_ = urma_context;
}

void *MilvusUbRuntime::InjectedUrmaContext() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return injected_urma_context_;
}

bool MilvusUbRuntime::RpcDistanceReady() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return distributed_runtime_ != nullptr;
}

Status MilvusUbRuntime::RetryEnableForPushdown() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (distributed_runtime_) {
    return Status::Ok();
  }
  return EnableDistributedSearchLocked();
}

Status MilvusUbRuntime::Search(
    std::span<const float> query, std::uint32_t nq, std::uint32_t dim,
    std::uint32_t topk, std::uint32_t ef, Metric metric,
    std::vector<std::int64_t> &ids, std::vector<float> &distances,
    std::chrono::steady_clock::time_point request_deadline) {
#if HNSW_URMA_ENABLE_URMA
  // Shutdown waits for in-flight callers; different searches share this lock.
  std::shared_lock lifetime(search_lifetime_mutex_, std::defer_lock);
  if (request_deadline == std::chrono::steady_clock::time_point::max()) {
    lifetime.lock();
  } else if (!lifetime.try_lock_until(request_deadline)) {
    return Status(StatusCode::kTimeout,
                  "UB request expired waiting for runtime lifecycle");
  }
  DistributedState *state;
  {
    std::unique_lock lock(state_mutex_, std::defer_lock);
    if (request_deadline == std::chrono::steady_clock::time_point::max()) {
      lock.lock();
    } else {
      // Runtime startup may hold state_mutex_ while provisioning transports.
      // A search's remaining budget also bounds that lifecycle wait.
      while (!lock.try_lock()) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= request_deadline) {
          return Status(StatusCode::kTimeout,
                        "UB request expired waiting for runtime state");
        }
        std::this_thread::sleep_for(std::min(
            request_deadline - now,
            std::chrono::steady_clock::duration(std::chrono::milliseconds(1))));
      }
    }
    ids.clear();
    distances.clear();
    if (!meta_ || dim != meta_->dim ||
        metric != static_cast<Metric>(meta_->metric)) {
      return Status(
          StatusCode::kInvalidArgument,
          "UB request dimension/metric does not match the loaded index");
    }
    state = distributed_runtime_.get();
    if (!state)
      return Status(StatusCode::kInternalError,
                    "UB distributed service is not ready");
    auto health = state->threads->status();
    if (!health.ok())
      return health;
  }
  auto result = state->executor->Submit(query, nq, topk, ef, request_deadline);
  if (!result.ok()) {
    auto health = state->threads->status();
    return health.ok() ? result.status() : health;
  }
  const auto &params = result.value().params;
  LOG_INFO(
      "UB request plan nq={} topk={} ef={} visited_capacity={} expand_batch={}",
      nq, params.topk, params.ef_search, params.visited_capacity,
      params.expand_batch);
  ids.assign(std::size_t(nq) * topk, -1);
  distances.assign(ids.size(), std::numeric_limits<float>::infinity());
  for (std::uint32_t row = 0; row < nq; ++row) {
    const auto &neighbors = result.value().rows[row].neighbors;
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
      ids[std::size_t(row) * topk + i] = neighbors[i].id;
      distances[std::size_t(row) * topk + i] = neighbors[i].distance;
    }
  }
  return Status::Ok();
#else
  return Status(StatusCode::kInvalidArgument,
                "Milvus was built without URMA support");
#endif
}

Status MilvusUbRuntime::EnsureRpcMailbox() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (rpc_memory_) {
    return Status::Ok();
  }
  if (!meta_ || !shard_map_ || local_shards_.empty()) {
    return Status(StatusCode::kInternalError,
                  "EnsureRpcMailbox: runtime not seeded");
  }
  const auto world_size =
      static_cast<std::uint32_t>(shard_map_->ranges().size());
  if (world_size == 0) {
    return Status(StatusCode::kInternalError,
                  "EnsureRpcMailbox: shard map is empty");
  }
  // The mailbox is registered and published to the URMA directory here, long
  // before EnableDistributedSearchLocked builds the transport; that transport
  // then reuses this memory. Sizing it from hard-coded 4 KiB cells made every
  // distance request larger than the staging slot and produced
  // "request fragment batch output is too small" on the wire (a single task
  // already carries a dim*4 query vector: 3840 B for dim=960, plus ids and the
  // fragment table). Resolve the real plan instead, exactly as
  // EnableDistributedSearchLocked does.
  auto parsed = RequestedRuntimeConfig();
  if (!parsed.ok())
    return parsed.status();
  const auto &requested = parsed.value();
  // The Milvus build has no MPI placement detection (DetectPlacement is a
  // stub returning an empty list); Go always stages the CPU set through
  // UbSetConfig (cgroup / yaml / numa). Without it the plan cannot be
  // resolved, so fail with a clear message instead of guessing.
  std::vector<std::uint32_t> allowed_cpus = knobs_.allowed_cpus;
  if (allowed_cpus.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "EnsureRpcMailbox: queryNode.ub allowedCpus is empty; "
                  "cannot resolve the RPC mailbox plan");
  }
  std::uint32_t local_rank = std::numeric_limits<std::uint32_t>::max();
  for (auto sid : loaded_shard_ids_) {
    local_rank = std::min(local_rank, static_cast<std::uint32_t>(sid));
  }
  auto effective = ResolveRuntimeConfig(requested, *meta_, world_size,
                                        allowed_cpus, local_rank);
  if (!effective.ok()) {
    return effective.status();
  }
  const auto &plan = effective.value();
  RpcMailboxConfig config;
  config.world_size = world_size;
  config.rpc_threads = plan.rpc_progress_threads;
  config.ring_depth = requested.request_ring_depth;
  config.request_capacity = plan.rpc_request_batch_max_bytes;
  config.response_capacity = plan.rpc_response_batch_max_bytes;
  config.request_staging_slots_per_thread =
      plan.rpc_memory.request_staging_slots_per_thread;
  auto mailbox = RpcMailboxMemory::Create(config);
  if (!mailbox.ok()) {
    return mailbox.status();
  }
  rpc_memory_ = std::move(mailbox).value();
  return Status::Ok();
}

Result<std::uint64_t>
MilvusUbRuntime::EstimateRuntimeMemory(const std::filesystem::path &root,
                                       shard_id_t local_shard) const {
  auto global = LoadGlobalIndexFile(root);
  if (!global.ok())
    return global.status();
  std::lock_guard lock(state_mutex_);
  auto requested = RequestedRuntimeConfig();
  if (!requested.ok())
    return requested.status();
  auto resolved = ResolveRuntimeConfig(requested.value(), global.value().meta,
                                       global.value().meta.num_shards,
                                       knobs_.allowed_cpus, local_shard);
  if (!resolved.ok())
    return resolved.status();
  const auto &plan = resolved.value();
  std::uint64_t total = 0;
  for (const auto bytes :
       {plan.queues.queue_bytes_total, plan.rpc_memory.registered_rpc_bytes,
        plan.registered_read_result_bytes, plan.rpc_operations.pool_bytes,
        plan.rpc_operations.client_owner_bytes}) {
    if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
      return Status(StatusCode::kResourceExhausted,
                    "UB runtime memory plan overflows");
    total += bytes;
  }
  // Each active query owns a visited hash table; include it before seeding.
  const auto slots = requested.value().max_active_queries;
  if (slots == 0)
    return Status(StatusCode::kInvalidArgument,
                  "UB active query count must be positive");
  const auto visited =
      global.value().meta.ntotal <= kDefaultDenseVisitedBudgetBytes / slots
          ? global.value().meta.ntotal
          : std::uint64_t(plan.visited_capacity) * sizeof(std::uint64_t);
  if (visited > (std::numeric_limits<std::uint64_t>::max() - total) / slots)
    return Status(StatusCode::kResourceExhausted,
                  "UB visited memory plan overflows");
  return total + visited * slots;
}

std::string MilvusUbRuntime::RpcLayout() const {
  std::lock_guard lock(state_mutex_);
  if (!rpc_memory_)
    return {};
  const auto &c = rpc_memory_->config();
  return std::to_string(DISTANCE_BATCH_WIRE_VERSION) + ":" +
         std::to_string(c.world_size) + ":" + std::to_string(c.rpc_threads) +
         ":" + std::to_string(c.ring_depth) + ":" +
         std::to_string(c.request_capacity) + ":" +
         std::to_string(c.response_capacity) + ":" +
         std::to_string(c.request_staging_slots_per_thread);
}

Status MilvusUbRuntime::GetRpcMailboxRegions(RpcMailboxRegion &out) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!rpc_memory_) {
    return Status(StatusCode::kInternalError,
                  "GetRpcMailboxRegions: mailbox not constructed");
  }
  // URMA driver (urma_register_seg) requires page-aligned VA + page-multiple
  // length. requests()/responses()/notify() return logical_memory() which may
  // have a non-page-multiple length (the registration region adds zero-filled
  // tail padding to satisfy the driver). Use request_registration() etc. so
  // the Go layer hands the driver a buffer it will actually pin. Without this
  // the worker QN publish of role 2/3 fails with segcoreCode=2001.
  auto req_reg = rpc_memory_->request_registration();
  auto resp_reg = rpc_memory_->response_registration();
  auto not_reg = rpc_memory_->notify_registration();
  out.requests_address = req_reg.data();
  out.requests_bytes = req_reg.size();
  out.responses_address = resp_reg.data();
  out.responses_bytes = resp_reg.size();
  out.notify_address = not_reg.data();
  out.notify_bytes = not_reg.size();
  out.ring_depth = rpc_memory_->config().ring_depth;
  return Status::Ok();
}

Status MilvusUbRuntime::StopDistributedSearch() {
  std::unique_lock lifetime(search_lifetime_mutex_);
  std::lock_guard lock(state_mutex_);
#if HNSW_URMA_ENABLE_URMA
  if (distributed_runtime_) {
    auto &state = *distributed_runtime_;
    state.executor->StopAdmission();
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(state.requested.drain_timeout_ms);
    while (state.threads->status().ok() &&
           std::chrono::steady_clock::now() < deadline &&
           (!state.executor->queries_drained() ||
            !state.drain->all_lanes_empty())) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!state.executor->queries_drained() || !state.drain->all_lanes_empty()) {
      // Do not invalidate memory still referenced by READ/RPC work. The caller
      // must retain its borrowed UrmaManager links when drain fails.
      auto health = state.threads->status();
      return health.ok()
                 ? Status(StatusCode::kTimeout, "UB data plane did not drain")
                 : health;
    }
    state.threads->RequestStop();
    state.threads->Join();
    LOG_INFO("UB shutdown queries_drained={} lanes_drained={} status={}",
             state.executor->queries_drained(), state.drain->all_lanes_empty(),
             state.threads->status().ToString());
    distributed_runtime_.reset();
  }
#endif
  return Status::Ok();
}

Status MilvusUbRuntime::Shutdown() {
  auto stopped = StopDistributedSearch();
  if (!stopped.ok())
    return stopped;
  std::lock_guard lock(state_mutex_);
  peer_handles_.clear();
  ResetLocked();
  return Status::Ok();
}

Status
MilvusUbRuntime::ImportPeerHandles(shard_id_t shard_id, std::uint32_t role,
                                   std::span<const std::byte> rjetty_descriptor,
                                   std::uint32_t jetty_token,
                                   void *target_segment) {
  if (target_segment == nullptr || rjetty_descriptor.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::ImportPeerHandles: null native handles");
  }
  if (role > 4) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::ImportPeerHandles: role must be 0..4");
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  PeerHandle handle;
  handle.rjetty_descriptor.assign(rjetty_descriptor.begin(),
                                  rjetty_descriptor.end());
  handle.jetty_token = jetty_token;
  handle.segment = target_segment;
  peer_handles_[{shard_id, role}] = std::move(handle);
  LOG_INFO("UB ImportPeerHandles: shard={} role={} token={} "
           "descriptor_bytes={} segment={} total_handles={}",
           static_cast<unsigned>(shard_id), role, jetty_token,
           rjetty_descriptor.size(), target_segment, peer_handles_.size());
  return Status::Ok();
}

void MilvusUbRuntime::RemovePeerHandles(shard_id_t shard_id,
                                        std::uint32_t role) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  peer_handles_.erase({shard_id, role});
}

std::size_t MilvusUbRuntime::PeerShardCount() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  std::set<shard_id_t> shards;
  for (const auto &entry : peer_handles_) {
    shards.insert(entry.first.first);
  }
  return shards.size();
}

std::vector<MilvusUbRuntime::ShardMemory>
MilvusUbRuntime::ShardMemoryRegions() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  std::vector<ShardMemory> regions;
  regions.reserve(local_shards_.size());
  for (const auto &shard : local_shards_) {
    ShardMemory region;
    region.shard_id = shard->range().shard_id;
    auto graph = shard->graph_registration_memory();
    auto vector = shard->vector_registration_memory();
    region.graph_address = graph.data();
    region.graph_bytes = graph.size();
    region.vector_address = vector.data();
    region.vector_bytes = vector.size();
    regions.push_back(region);
  }
  return regions;
}

Status MilvusUbRuntime::GetEntryPoint(node_id_t &out_entry_vertex_id,
                                      std::int32_t &out_entry_level) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!meta_) {
    return Status(StatusCode::kInternalError,
                  "MilvusUbRuntime::GetEntryPoint: runtime not seeded");
  }
  out_entry_vertex_id = meta_->entry_point;
  out_entry_level = meta_->max_level;
  return Status::Ok();
}

Status MilvusUbRuntime::GetEntryShard(shard_id_t &out_entry_shard_id) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!meta_ || !shard_map_) {
    return Status(StatusCode::kInternalError,
                  "MilvusUbRuntime::GetEntryShard: runtime not seeded");
  }
  auto located = shard_map_->Locate(meta_->entry_point);
  if (!located.ok()) {
    return located.status();
  }
  out_entry_shard_id = located.value().shard_id;
  return Status::Ok();
}

Status MilvusUbRuntime::ValidateShardLayout(shard_id_t shard_id,
                                            std::uint32_t shard_count,
                                            shard_id_t entry_shard_id,
                                            node_id_t entry_vertex_id) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!meta_ || !shard_map_) {
    return Status(StatusCode::kInternalError,
                  "MilvusUbRuntime::ValidateShardLayout: runtime not seeded");
  }
  if (shard_count != meta_->num_shards) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::ValidateShardLayout: shard count " +
                      std::to_string(shard_count) + " != global.meta " +
                      std::to_string(meta_->num_shards));
  }
  if (static_cast<std::uint64_t>(shard_id) >= shard_count) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::ValidateShardLayout: shard id " +
                      std::to_string(shard_id) + " out of range");
  }
  if (static_cast<std::uint64_t>(entry_shard_id) >= shard_count) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::ValidateShardLayout: entry shard id " +
                      std::to_string(entry_shard_id) + " out of range");
  }
  // The published entry vertex must live inside the shard the record claims
  // as the entry shard: the local shard map is the same source of truth
  // (global.meta) the publisher used, so any mismatch means the directory
  // record does not belong to this graph build.
  auto located = shard_map_->Locate(entry_vertex_id);
  if (!located.ok()) {
    return Status(StatusCode::kInvalidArgument,
                  "MilvusUbRuntime::ValidateShardLayout: entry vertex " +
                      std::to_string(entry_vertex_id) +
                      " not inside the local shard map");
  }
  if (located.value().shard_id != entry_shard_id) {
    return Status(
        StatusCode::kInvalidArgument,
        "MilvusUbRuntime::ValidateShardLayout: entry vertex belongs to shard " +
            std::to_string(located.value().shard_id) + " but record claims " +
            std::to_string(entry_shard_id));
  }
  return Status::Ok();
}

bool MilvusUbRuntime::Ready() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
#if HNSW_URMA_ENABLE_URMA
  return distributed_runtime_ && distributed_runtime_->threads->status().ok();
#else
  return false;
#endif
}

std::size_t MilvusUbRuntime::LoadedShardCount() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return local_shards_.size();
}

std::size_t MilvusUbRuntime::ExpectedShardCount() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return shard_map_ ? shard_map_->ranges().size() : 0;
}

MilvusUbRuntime::ShardLoadSnapshot
MilvusUbRuntime::GetShardLoadSnapshot() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  ShardLoadSnapshot snap;
  // Mirrors LocalShardId(): the single owned shard, if exactly one is loaded.
  if (loaded_shard_ids_.size() == 1) {
    snap.local_shard_id = loaded_shard_ids_.front();
  }
  snap.loaded_count = local_shards_.size();
  snap.expected_count = shard_map_ ? shard_map_->ranges().size() : 0;
  return snap;
}

// Unsupported metrics remain invalid at the request boundary; never reinterpret
// as L2.
Metric MilvusUbRuntime_MetricFromName(const char *name) {
  const std::string_view metric = name ? name : "";
  if (metric == "L2" || metric == "l2")
    return Metric::kL2;
  if (metric == "IP" || metric == "ip" || metric == "INNER_PRODUCT")
    return Metric::kInnerProduct;
  return static_cast<Metric>(0xff);
}
} // namespace hnsw_urma
