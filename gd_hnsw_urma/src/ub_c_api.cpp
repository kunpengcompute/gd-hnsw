#include "hnsw_urma/ub_c_api.h"
#include "hnsw_urma/core/log.hpp"
#include "fmt/format.h"

#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/compute/distance.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/milvus_ub_runtime.hpp"

namespace hnsw_urma {
// Defined in milvus_ub_runtime.cpp. Exposed here so the C ABI trampoline
// can convert milvus metric names to URMA Metric without leaking the
// singleton header into ub_c_api.h.
Metric MilvusUbRuntime_MetricFromName(const char* name);
}  // namespace hnsw_urma

int UbPing(int x) {
  return x + 1;
}

// ---- Thread-local last error (UbLastError) -------------------------------
//
// Every Ub* C ABI entry that can fail writes a human-readable status string
// here on failure (and clears it on success). The pointer is valid until the
// next Ub* call on the same thread; callers must copy if they need it across
// calls. This preserves the diagnostics the prior ThrowInfo paths surfaced,
// which a bare int StatusCode would have lost.
namespace {
thread_local std::string g_last_error;
void ClearUbLastError() { g_last_error.clear(); }
void SetUbLastError(std::string_view msg) { g_last_error.assign(msg); }
}  // namespace

int UbEstimateRuntimeMemory(const char* root, uint32_t shard, uint64_t* bytes) {
  using namespace hnsw_urma;
  if (!root || !bytes || shard > std::numeric_limits<shard_id_t>::max())
    return static_cast<int>(StatusCode::kInvalidArgument);
  *bytes = 0;
  try {
    auto estimate = MilvusUbRuntime::Instance().EstimateRuntimeMemory(root, static_cast<shard_id_t>(shard));
    if (!estimate.ok()) return static_cast<int>(estimate.status().code());
    *bytes = estimate.value();
    return 0;
  } catch (const std::bad_alloc&) {
    return static_cast<int>(StatusCode::kResourceExhausted);
  } catch (const std::exception&) {
    return static_cast<int>(StatusCode::kInternalError);
  }
}

const char* UbLastError(void) {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

int UbConfigure(const UbConfig* cfg) {
  try {
    if (cfg == nullptr) {
      SetUbLastError("UbConfigure: cfg is null");
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
    if (hnsw_urma::MilvusUbRuntime::Instance().RpcDistanceReady()) {
      SetUbLastError("UbConfigure: runtime already enabled; configure before enable");
      return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
    }
    auto knobs = hnsw_urma::MilvusUbRuntime::Instance().GetKnobs();
    // 0/NULL everywhere means "keep built-in default" — only overwrite when the
    // host actually set a value.
    knobs.search_workers = cfg->search_workers;
    knobs.distance_workers = cfg->distance_workers;
    knobs.read_progress_workers = cfg->read_progress_workers;
    knobs.rpc_progress_workers = cfg->rpc_progress_workers;
    knobs.cpu_bind_mode = cfg->cpu_bind_mode ? cfg->cpu_bind_mode : "auto";
    knobs.cpu_layout = cfg->cpu_layout ? cfg->cpu_layout : "contiguous";
    knobs.progress_mode = cfg->progress_mode ? cfg->progress_mode : "busy";
    if (cfg->allowed_cpu_count && cfg->allowed_cpus) {
      knobs.allowed_cpus.assign(cfg->allowed_cpus,
                                cfg->allowed_cpus + cfg->allowed_cpu_count);
    } else if (cfg->allowed_cpu_count && !cfg->allowed_cpus) {
      // Non-zero count but NULL array: illegal input. Returning here avoids
      // pointer arithmetic / dereference on NULL (UB, crash). Early return
      // before SetKnobs so knobs stays unmodified.
      SetUbLastError("UbConfigure: allowed_cpu_count>0 but allowed_cpus is NULL");
      return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
    } else {
      knobs.allowed_cpus.clear();
    }
    knobs.max_active_queries = cfg->max_active_queries;
    knobs.read_jfs_depth = cfg->read_jfs_depth;
    knobs.queue_memory_budget_bytes = cfg->queue_memory_budget_bytes;
    knobs.registered_rpc_memory_budget_bytes = cfg->registered_rpc_memory_budget_bytes;
    knobs.query_timeout_ms = cfg->query_timeout_ms;
    knobs.operation_timeout_ms = cfg->operation_timeout_ms;
    knobs.expand_batch = cfg->expand_batch;
    knobs.runtime_options = cfg->runtime_options ? cfg->runtime_options : "";
    knobs.urma_device = cfg->urma_device_name ? cfg->urma_device_name : std::string();
    knobs.eid_index = cfg->urma_eid_index;
    hnsw_urma::MilvusUbRuntime::Instance().SetKnobs(knobs);
    ClearUbLastError();
    return static_cast<int>(hnsw_urma::StatusCode::kOk);
  } catch (const std::bad_alloc&) {
    SetUbLastError("UbConfigure: out of memory");
    return static_cast<int>(hnsw_urma::StatusCode::kResourceExhausted);
  } catch (const std::exception& e) {
    SetUbLastError(e.what());
    return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
  }
}

int UbSetUrmaContext(void* urma_context) {
  // Setting the context after the distributed runtime is already enabled
  // would race with the UMDK backend that already bonded to the old context.
  if (hnsw_urma::MilvusUbRuntime::Instance().RpcDistanceReady()) {
    SetUbLastError("UbSetUrmaContext: runtime already enabled");
    return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
  }
  hnsw_urma::MilvusUbRuntime::Instance().SetInjectedUrmaContext(urma_context);
  ClearUbLastError();
  return static_cast<int>(hnsw_urma::StatusCode::kOk);
}

int UbSearch(const float* query, uint32_t nq, uint32_t dim, uint32_t topk,
             uint32_t ef_search, const char* metric_name,
             int64_t* out_ids, float* out_dists) {
  return UbSearchWithTimeout(query, nq, dim, topk, ef_search, metric_name,
                             -1, out_ids, out_dists);
}

int UbSearchWithTimeout(const float* query, uint32_t nq, uint32_t dim, uint32_t topk,
                        uint32_t ef_search, const char* metric_name,
                        int64_t timeout_ns, int64_t* out_ids, float* out_dists) {
  try {
    if (timeout_ns < -1) {
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
    if (timeout_ns == 0) return static_cast<int>(hnsw_urma::StatusCode::kTimeout);
    using Clock = std::chrono::steady_clock;
    const auto now = Clock::now();
    const auto deadline = timeout_ns < 0 ? Clock::time_point::max()
        : now + std::min(std::chrono::nanoseconds(timeout_ns),
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - now));
    if (!query || !out_ids || !out_dists) {
      SetUbLastError("UbSearch: null query/out_ids/out_dists");
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
    auto metric = hnsw_urma::MilvusUbRuntime_MetricFromName(metric_name);
    std::span<const float> qspan(query, static_cast<std::size_t>(nq) * dim);
    std::vector<std::int64_t> ids;
    std::vector<float> dists;
    auto st = hnsw_urma::MilvusUbRuntime::Instance().Search(
        qspan, nq, dim, topk, ef_search, metric, ids, dists, deadline);
    if (!st.ok()) {
      SetUbLastError(st.ToString());
      LOG_ERROR("UB service search failed: {}", st.ToString());
      return static_cast<int>(st.code());
    }
    std::memcpy(out_ids, ids.data(),
                static_cast<std::size_t>(nq) * topk * sizeof(std::int64_t));
    std::memcpy(out_dists, dists.data(),
                static_cast<std::size_t>(nq) * topk * sizeof(float));
    ClearUbLastError();
    return static_cast<int>(hnsw_urma::StatusCode::kOk);
  } catch (const std::bad_alloc&) {
    SetUbLastError("UbSearchWithTimeout: out of memory");
    return static_cast<int>(hnsw_urma::StatusCode::kResourceExhausted);
  } catch (const std::exception& e) {
    SetUbLastError(e.what());
    return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
  }
}

int UbStopDistributedSearch(void) {
  return static_cast<int>(hnsw_urma::MilvusUbRuntime::Instance().StopDistributedSearch().code());
}
int UbShutdown(void) {
  return static_cast<int>(hnsw_urma::MilvusUbRuntime::Instance().Shutdown().code());
}

int UbSetLocalShard(uint32_t shard_id) {
  auto st = hnsw_urma::MilvusUbRuntime::Instance()
                .SetLocalShard(static_cast<hnsw_urma::shard_id_t>(shard_id));
  if (!st.ok()) SetUbLastError(st.ToString());
  else ClearUbLastError();
  return static_cast<int>(st.code());
}

int UbLoadShardFromDir(const char* triple_dir, uint32_t shard_id,
                       UbIndexMeta* out_meta) {
  try {
    if (!triple_dir || !*triple_dir) {
      SetUbLastError("UbLoadShardFromDir: triple_dir is null/empty");
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
    namespace fs = std::filesystem;
    fs::path root(triple_dir);
    auto loaded = hnsw_urma::LoadIndexForShard(
        root, static_cast<hnsw_urma::shard_id_t>(shard_id),
        hnsw_urma::LocalShardStorage::kPrivate);
    if (!loaded.ok()) {
      SetUbLastError(loaded.status().ToString());
      return static_cast<int>(loaded.status().code());
    }
    // C4: sum(local_count over all ranges) must equal ntotal, and the requested
    // shard must be present. Pushed down from UBHNSWIndex so the host no longer
    // reads algorithm internals.
    std::uint64_t sum_counts = 0;
    bool shard_found = false;
    std::uint64_t local_count = 0;
    for (const auto& range : loaded.value().shard_map.ranges()) {
      sum_counts += range.local_count;
      if (range.shard_id == static_cast<hnsw_urma::shard_id_t>(shard_id)) {
        shard_found = true;
        local_count = range.local_count;
      }
    }
    if (!shard_found || sum_counts != loaded.value().meta.ntotal) {
      auto msg = fmt::format(
          "UbLoadShardFromDir: C4 check failed for shard {} (found={}, "
          "sum(local_count)={}, ntotal={})",
          shard_id, shard_found, sum_counts, loaded.value().meta.ntotal);
      SetUbLastError(msg);
      return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
    }
    // v2: project the meta back so the host skips a follow-up UbReadIndexMeta.
    if (out_meta != nullptr) {
      out_meta->dim = loaded.value().meta.dim;
      out_meta->ntotal = loaded.value().meta.ntotal;
      out_meta->shard_count =
          static_cast<std::uint32_t>(loaded.value().shard_map.ranges().size());
      out_meta->local_count = local_count;
      out_meta->reserved = 0;
    }
    auto owned = std::unique_ptr<hnsw_urma::LoadedIndex>(
        new hnsw_urma::LoadedIndex(std::move(loaded).value()));
    auto st = hnsw_urma::MilvusUbRuntime::Instance().LoadShard(std::move(owned));
    if (!st.ok()) {
      SetUbLastError(st.ToString());
      return static_cast<int>(st.code());
    }
    ClearUbLastError();
    return static_cast<int>(hnsw_urma::StatusCode::kOk);
  } catch (const std::bad_alloc&) {
    SetUbLastError("UbLoadShardFromDir: out of memory");
    return static_cast<int>(hnsw_urma::StatusCode::kResourceExhausted);
  } catch (const std::exception& e) {
    SetUbLastError(e.what());
    return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
  }
}

int UbBindGlobalGraphIdentity(const char* graph_id, uint64_t epoch) {
  return static_cast<int>(hnsw_urma::MilvusUbRuntime::Instance()
      .BindGlobalGraphIdentity(graph_id ? graph_id : "", epoch).code());
}

int UbRpcDistanceReady(void) {
  return hnsw_urma::MilvusUbRuntime::Instance().RpcDistanceReady() ? 1 : 0;
}

int UbReadIndexMeta(const char* triple_dir,
                    uint32_t shard_id,
                    UbIndexMeta* out_meta) {
  if (!triple_dir || !*triple_dir || !out_meta) {
    SetUbLastError("UbReadIndexMeta: null argument");
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  namespace fs = std::filesystem;
  fs::path root(triple_dir);
  auto global = hnsw_urma::LoadGlobalIndexFile(root);
  if (!global.ok()) {
    SetUbLastError(global.status().ToString());
    return static_cast<int>(global.status().code());
  }
  out_meta->dim = global.value().meta.dim;
  out_meta->ntotal = global.value().meta.ntotal;
  out_meta->shard_count = static_cast<std::uint32_t>(global.value().ranges.size());
  std::uint64_t local = 0;
  for (const auto& range : global.value().ranges) {
    if (range.shard_id == static_cast<hnsw_urma::shard_id_t>(shard_id)) {
      local = range.local_count;
      break;
    }
  }
  out_meta->local_count = local;
  out_meta->reserved = 0;
  ClearUbLastError();
  return static_cast<int>(hnsw_urma::StatusCode::kOk);
}

// v2: one atomic snapshot of shard-load status. The runtime reads local
// shard id, loaded count, and expected count under its own lock so a
// distributed-enable ready gate cannot observe a torn (loaded bumped but
// local_id not yet updated) state. Replaces UbGetLocalShardId +
// UbLoadedShardCount + UbExpectedShardCount.
int UbGetShardStatus(UbShardStatus* out) {
  if (out == nullptr) {
    SetUbLastError("UbGetShardStatus: out is null");
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  auto snap = hnsw_urma::MilvusUbRuntime::Instance().GetShardLoadSnapshot();
  out->local_shard_id =
      snap.local_shard_id.has_value()
          ? static_cast<std::uint32_t>(*snap.local_shard_id)
          : UB_NO_LOCAL_SHARD;
  out->loaded_count = static_cast<std::uint32_t>(snap.loaded_count);
  out->expected_count = static_cast<std::uint32_t>(snap.expected_count);
  out->reserved = 0;
  ClearUbLastError();
  return static_cast<int>(hnsw_urma::StatusCode::kOk);
}

int UbReuseGlobalGraphShard(const char* graph_id, uint64_t epoch,
                            uint32_t shard, int* out_reused,
                            UbSeededShardInfo* out_info) {
  if (out_reused == nullptr) {
    SetUbLastError("UbReuseGlobalGraphShard: out_reused is null");
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  *out_reused = 0;
  if (graph_id == nullptr || graph_id[0] == 0 || out_info == nullptr) {
    SetUbLastError("UbReuseGlobalGraphShard: graph_id/out_info is null/empty");
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  try {
    auto result = hnsw_urma::MilvusUbRuntime::Instance().ReuseGlobalGraphShard(
        graph_id, epoch, static_cast<hnsw_urma::shard_id_t>(shard));
    if (!result.ok()) {
      SetUbLastError(result.status().ToString());
      return static_cast<int>(result.status().code());
    }
    const auto& seeded = result.value();
    if (seeded.has_value()) {
      out_info->dim = seeded->dim;
      out_info->count = seeded->count;
      out_info->reserved = 0;
      *out_reused = 1;
    }
    ClearUbLastError();
    return static_cast<int>(hnsw_urma::StatusCode::kOk);
  } catch (const std::bad_alloc&) {
    return static_cast<int>(hnsw_urma::StatusCode::kResourceExhausted);
  } catch (const std::exception& e) {
    SetUbLastError(e.what());
    return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
  }
}

static int UbSeedShards(const char* triple_dir,
                        const std::vector<std::uint32_t>& shard_ids) {
  if (!triple_dir || !*triple_dir) {
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  namespace fs = std::filesystem;
  fs::path root(triple_dir);
  auto global = hnsw_urma::LoadGlobalIndexFile(root);
  if (!global.ok()) {
    return static_cast<int>(global.status().code());
  }
  const bool runtime_was_empty =
      hnsw_urma::MilvusUbRuntime::Instance().LoadedShardCount() == 0;
  const auto shard_count = global.value().ranges.size();
  for (auto sid : shard_ids) {
    if (static_cast<std::size_t>(sid) >= shard_count) {
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
  }
  for (auto sid : shard_ids) {
    auto loaded = hnsw_urma::LoadIndexForShard(
        root, static_cast<hnsw_urma::shard_id_t>(sid),
        hnsw_urma::LocalShardStorage::kPrivate);
    if (!loaded.ok()) {
      if (runtime_was_empty) {
        (void)hnsw_urma::MilvusUbRuntime::Instance().Shutdown();
      }
      return static_cast<int>(loaded.status().code());
    }
    auto owned = std::unique_ptr<hnsw_urma::LoadedIndex>(
        new hnsw_urma::LoadedIndex(std::move(loaded).value()));
    auto st = hnsw_urma::MilvusUbRuntime::Instance().LoadShard(
        std::move(owned));
    if (!st.ok()) {
      // LoadShard can have installed shared metadata before a later shard
      // consistency check fails. Roll back only a seed that began from an
      // empty runtime; never tear down a runtime that was already serving.
      if (runtime_was_empty) {
        (void)hnsw_urma::MilvusUbRuntime::Instance().Shutdown();
      }
      return static_cast<int>(st.code());
    }
  }
  return static_cast<int>(hnsw_urma::StatusCode::kOk);
}

// v2: merged UbSeedFromDir (full) + UbSeedFromDirWithShards (subset).
// shard_ids==NULL or count==0 = seed every shard range advertised by
// global.meta; otherwise seed only the listed subset.
int UbSeedFromDir(const char* triple_dir,
                  const uint32_t* shard_ids,
                  uint32_t count) {
  try {
    if (!triple_dir || !*triple_dir) {
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
    if (shard_ids == nullptr || count == 0) {
      // Full-seed: enumerate every shard range from global.meta.
      namespace fs = std::filesystem;
      fs::path root(triple_dir);
      auto global = hnsw_urma::LoadGlobalIndexFile(root);
      if (!global.ok()) {
        return static_cast<int>(global.status().code());
      }
      std::vector<std::uint32_t> all(global.value().ranges.size());
      for (std::size_t i = 0; i < all.size(); ++i) {
        all[i] = static_cast<std::uint32_t>(i);
      }
      return UbSeedShards(triple_dir, all);
    }
    std::vector<std::uint32_t> ids(shard_ids, shard_ids + count);
    return UbSeedShards(triple_dir, ids);
  } catch (const std::bad_alloc&) {
    SetUbLastError("UbSeedFromDir: out of memory");
    return static_cast<int>(hnsw_urma::StatusCode::kResourceExhausted);
  } catch (const std::exception& e) {
    SetUbLastError(e.what());
    return static_cast<int>(hnsw_urma::StatusCode::kInternalError);
  }
}

int UbImportPeerHandles(uint32_t shard_id, uint32_t role,
                        const void* rjetty_descriptor,
                        uint32_t descriptor_bytes,
                        uint32_t jetty_token,
                        void* target_segment) {
  // Validate the pointer itself before constructing the span: the downstream
  // ImportPeerHandles null-check is rjetty_descriptor.empty() (size-based), so
  // "nullptr + descriptor_bytes>0" slips through (size!=0 => empty()==false),
  // and the downstream assign(begin(), end()) then dereferences nullptr and
  // crashes. span bundles pointer + length and drops the "pointer is null"
  // dimension, so this layer must check it. Downstream already rejects empty
  // descriptors (empty() => kInvalidArgument), so rejecting nullptr here
  // agrees with that.
  if (rjetty_descriptor == nullptr) {
    SetUbLastError("UbImportPeerHandles: rjetty_descriptor is null");
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  auto bytes = std::span<const std::byte>(
      static_cast<const std::byte*>(rjetty_descriptor), descriptor_bytes);
  auto st = hnsw_urma::MilvusUbRuntime::Instance().ImportPeerHandles(
      shard_id, role, bytes, jetty_token, target_segment);
  if (!st.ok()) SetUbLastError(st.ToString());
  else ClearUbLastError();
  return st.ok() ? static_cast<int>(hnsw_urma::StatusCode::kOk)
                 : static_cast<int>(st.code());
}

void UbRemovePeerHandles(uint32_t shard_id, uint32_t role) {
  hnsw_urma::MilvusUbRuntime::Instance().RemovePeerHandles(shard_id, role);
}

// v4: one atomic-ish snapshot of everything publishUbShardGraphResources
// needs to build a directory record: per-shard registration memory + global
// shard count + entry point (vertex id, level, owning shard). Replaces the
// prior UbGetShardRegions + UbGetEntryPoint + UbGetEntryShard triple (and
// absorbs the publish-site UbExpectedShardCount call).
//
// Two-phase use: pass out_regions==NULL / capacity==0 to query only the
// count (out_snapshot->regions then points at the runtime-owned internal
// array, valid until UbShutdown); pass a non-NULL out_regions of length
// >= region_count to receive a caller-owned copy of the per-shard records
// (out_snapshot->regions then points at out_regions).
int UbGetGraphPublishSnapshot(uint32_t capacity,
                              UbShardRegion* out_regions,
                              UbGraphPublishSnapshot* out_snapshot) {
  if (out_snapshot == nullptr) {
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  auto& rt = hnsw_urma::MilvusUbRuntime::Instance();
  auto regions = rt.ShardMemoryRegions();

  // Global entry point + its owning shard. Failing here means no shard is
  // loaded yet; publish is not meaningful, so surface the StatusCode.
  hnsw_urma::node_id_t entry_vertex = 0;
  std::int32_t entry_level = -1;
  auto ep_st = rt.GetEntryPoint(entry_vertex, entry_level);
  if (!ep_st.ok()) {
    SetUbLastError(ep_st.ToString());
    return static_cast<int>(ep_st.code());
  }
  hnsw_urma::shard_id_t entry_shard = 0;
  auto es_st = rt.GetEntryShard(entry_shard);
  if (!es_st.ok()) {
    SetUbLastError(es_st.ToString());
    return static_cast<int>(es_st.code());
  }

  // Fill caller-provided region array when present.
  //
  // NOTE: we deliberately do NOT write out_snapshot->regions = out_regions.
  // out_regions may be a Go-heap pointer (CGO caller), and out_snapshot is
  // also Go-heap memory; having C store a Go pointer into Go memory is the
  // pattern cgo's pointer checker rejects ("Go pointer to Go pointer"). The
  // caller already owns out_regions and knows how many entries (region_count)
  // were filled, so it does not need the back-pointer. Leave regions=nullptr
  // in both phases and let callers use the (caller-owned) buffer they passed.
  if (out_regions != nullptr) {
    if (regions.size() > capacity) {
      SetUbLastError("UbGetGraphPublishSnapshot: capacity too small for regions");
      return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
    }
    for (std::size_t i = 0; i < regions.size(); ++i) {
      out_regions[i].shard_id =
          static_cast<std::uint32_t>(regions[i].shard_id);
      out_regions[i].graph_address = regions[i].graph_address;
      out_regions[i].graph_bytes = regions[i].graph_bytes;
      out_regions[i].vector_address = regions[i].vector_address;
      out_regions[i].vector_bytes = regions[i].vector_bytes;
    }
  }
  out_snapshot->regions = nullptr;
  out_snapshot->region_count = static_cast<std::uint32_t>(regions.size());
  out_snapshot->shard_count =
      static_cast<std::uint32_t>(rt.ExpectedShardCount());
  out_snapshot->entry_shard_id = static_cast<std::uint32_t>(entry_shard);
  out_snapshot->entry_vertex_id = static_cast<std::uint64_t>(entry_vertex);
  out_snapshot->entry_level = entry_level;
  out_snapshot->reserved = 0;
  ClearUbLastError();
  return static_cast<int>(hnsw_urma::StatusCode::kOk);
}

int UbValidateShardLayout(uint32_t shard_id,
                          uint32_t shard_count,
                          uint32_t entry_shard_id,
                          uint64_t entry_vertex_id) {
  auto st = hnsw_urma::MilvusUbRuntime::Instance().ValidateShardLayout(
      shard_id, shard_count, entry_shard_id, entry_vertex_id);
  return st.ok() ? static_cast<int>(hnsw_urma::StatusCode::kOk)
                 : static_cast<int>(st.code());
}

// v3: merged UbRetryEnableForPushdown's optional msg buffer into the enable
// entry. The runtime EnableDistributedSearch is idempotent (returns the
// running status when already up), so "retry" is just calling enable again.
// On failure, a nul-terminated reason is written to msg (truncated to
// msg_capacity) when msg is non-NULL and msg_capacity > 0.
int UbEnableDistributedSearch(char* msg, int msg_capacity) {
  auto st = hnsw_urma::MilvusUbRuntime::Instance().EnableDistributedSearch();
  if (st.ok()) {
    ClearUbLastError();
    return static_cast<int>(hnsw_urma::StatusCode::kOk);
  }
  const auto text = st.ToString();
  SetUbLastError(text);
  if (msg != nullptr && msg_capacity > 0) {
    std::snprintf(msg, static_cast<std::size_t>(msg_capacity), "%s", text.c_str());
  }
  return static_cast<int>(st.code());
}

int UbEnableDistributedSearchWithTimeout(int64_t timeout_ns) {
  using namespace hnsw_urma;
  if (timeout_ns < -1) return static_cast<int>(StatusCode::kInvalidArgument);
  if (timeout_ns == 0) return static_cast<int>(StatusCode::kTimeout);
  using Clock = std::chrono::steady_clock;
  const auto now = Clock::now();
  const auto deadline = timeout_ns < 0 ? Clock::time_point::max()
      : now + std::min(std::chrono::nanoseconds(timeout_ns),
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - now));
  try {
    auto st = MilvusUbRuntime::Instance().EnableDistributedSearch(deadline);
    return static_cast<int>(st.code());
  } catch (const std::bad_alloc&) {
    return static_cast<int>(StatusCode::kResourceExhausted);
  } catch (const std::exception&) {
    return static_cast<int>(StatusCode::kInternalError);
  }}

const char* UbRpcLayout(void) {
  thread_local std::string layout;
  layout = hnsw_urma::MilvusUbRuntime::Instance().RpcLayout();
  return layout.c_str();
}

// v3: merged UbEnsureRpcMailbox (void) + UbGetRpcMailboxRegions. Builds the
// mailbox and returns its three segment addresses/sizes + ring_depth in one
// round-trip. out must be non-null.
int UbEnsureRpcMailbox(UbRpcMailboxRegions* out) {
  if (out == nullptr) {
    return static_cast<int>(hnsw_urma::StatusCode::kInvalidArgument);
  }
  auto ensure_st = hnsw_urma::MilvusUbRuntime::Instance().EnsureRpcMailbox();
  if (!ensure_st.ok()) {
    SetUbLastError(ensure_st.ToString());
    return static_cast<int>(ensure_st.code());
  }
  hnsw_urma::MilvusUbRuntime::RpcMailboxRegion region;
  auto get_st = hnsw_urma::MilvusUbRuntime::Instance().GetRpcMailboxRegions(region);
  if (!get_st.ok()) {
    SetUbLastError(get_st.ToString());
    return static_cast<int>(get_st.code());
  }
  out->requests_address = region.requests_address;
  out->requests_bytes = region.requests_bytes;
  out->responses_address = region.responses_address;
  out->responses_bytes = region.responses_bytes;
  out->notify_address = region.notify_address;
  out->notify_bytes = region.notify_bytes;
  out->ring_depth = region.ring_depth;
  out->reserved = 0;
  ClearUbLastError();
  return static_cast<int>(hnsw_urma::StatusCode::kOk);
}
