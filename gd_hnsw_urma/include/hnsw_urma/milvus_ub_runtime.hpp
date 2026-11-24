// Milvus adapter for the pinned distributed HNSW runtime. One worker owns
// one shard; the Go router selects the worker and its shard-local entry.
#ifndef HNSW_URMA_MILVUS_UB_RUNTIME_H_
#define HNSW_URMA_MILVUS_UB_RUNTIME_H_

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/runtime/rpc/mailbox.hpp"
#include "hnsw_urma/runtime/thread_group.hpp"
#include "hnsw_urma/search/io.hpp"
#include "hnsw_urma/search/local_io.hpp"
#include "hnsw_urma/search/query_context.hpp"

namespace hnsw_urma {

class MilvusUbRuntime {
public:
  static MilvusUbRuntime &Instance();

  // Called from UBHNSWIndex::LoadV2Index for each segment. Moves ownership
  // of the LoadedIndex into the singleton. The first call seeds meta_,
  // shard_map_, upper_; subsequent calls only append local_shard.
  // Returns an error if the new index's meta/shard_map/upper disagree with
  // the already-loaded ones (they are shared across segments of one
  // collection and must be byte-identical).
  Status LoadShard(std::unique_ptr<LoadedIndex> loaded);

  // Bind only after successful seed. A different graph/epoch requires a
  // successful Shutdown; equal metadata alone never establishes identity.
  Status BindGlobalGraphIdentity(std::string graph_id, std::uint64_t epoch);
  struct SeededShardInfo {
    std::uint32_t dim;
    std::uint64_t count;
  };
  Result<std::optional<SeededShardInfo>>
  ReuseGlobalGraphShard(const std::string &graph_id, std::uint64_t epoch,
                        shard_id_t shard) const;

  // Blocking service search, padded with -1 / +inf when fewer than topk
  // neighbors exist. All callers use the same distributed executor.
  Status Search(std::span<const float> query, std::uint32_t nq,
                std::uint32_t dim, std::uint32_t topk, std::uint32_t ef_search,
                Metric metric, std::vector<std::int64_t> &out_ids,
                std::vector<float> &out_dists,
                std::chrono::steady_clock::time_point request_deadline =
                    std::chrono::steady_clock::time_point::max());

  // Releases all state. Safe to call multiple times. After Shutdown, the
  // singleton can be re-seeded by the next LoadShard.
  Status StopDistributedSearch();
  Status Shutdown();

  // ---- M-3 bridge: peer handle injection -------------------------------
  //
  // Borrows a self-described peer endpoint imported by the host's URMA
  // manager for the peer owning `shard_id`. To keep the algorithm library
  // free of the host's `UrmaPeerEndpointInfo` type, the host passes the raw
  // rjetty descriptor bytes and the peer jetty token directly; the runtime
  // copies the bytes (UMDK needs a stable address for the descriptor
  // lifetime). The segment handle (`target_segment`) is opaque to the
  // algorithm and forwarded to UMDK verbatim.
  //
  // The handles are NOT owned by the runtime: the caller must keep the
  // host URMA link alive for as long as the runtime uses them (i.e. until
  // Shutdown or RemovePeerHandles).
  //
  // role selects which memory region of the peer shard the segment handle
  // refers to:
  //   0 = topology (layer-0 graph slots)
  //   1 = vectors
  //   2 = distance-tasks queue
  //   3 = distance-results queue
  //   4 = distance-notify (WRITE_NOTIFY target for RPC mailbox credits)
  //
  // The runtime stores the entry keyed by (shard_id, role); a repeated call
  // with the same key replaces the previous entry (idempotent re-activation).
  Status ImportPeerHandles(shard_id_t shard_id, std::uint32_t role,
                           std::span<const std::byte> rjetty_descriptor,
                           std::uint32_t jetty_token, void *target_segment);

  // Removes the stored pair for (shard_id, role) without touching any
  // native resource (ownership stays with UrmaManager).
  void RemovePeerHandles(shard_id_t shard_id, std::uint32_t role);

  // Number of distinct shard_ids with at least one stored pair (diagnostic).
  std::size_t PeerShardCount() const;

  // ---- M-3 bridge: publish-side memory regions ---------------------------
  //
  // Describes one locally loaded shard's graph/vector memory so the caller
  // (Go/CGO) can register and publish it via UrmaManager's 4-role model.
  struct ShardMemory {
    shard_id_t shard_id{0};
    void *graph_address{nullptr};
    std::uint64_t graph_bytes{0};
    void *vector_address{nullptr};
    std::uint64_t vector_bytes{0};
  };

  // Collects the registration memory of every loaded shard. The memory is
  // owned by the runtime; callers must not free it and must not use it
  // after Shutdown.
  std::vector<ShardMemory> ShardMemoryRegions() const;

  // Publish-side identity: the global graph's entry point (meta_.entry_point)
  // and its level (meta_.max_level). Fails when no shard has been loaded.
  Status GetEntryPoint(node_id_t &out_entry_vertex_id,
                       std::int32_t &out_entry_level) const;

  // Resolves which shard owns the global entry point. This is the correct
  // EntryShardID to publish in the URMA resource directory (the entry
  // point's level is NOT a shard id).
  Status GetEntryShard(shard_id_t &out_entry_shard_id) const;

  // ---- M-4: shard numbering same-source validation -----------------------
  //
  // Validates an externally published shard identity (etcd ResourceRecord)
  // against the locally seeded global.meta topology. Rejects:
  //   - shard_count != meta_.num_shards (records from a different build)
  //   - shard_id / entry_shard_id outside [0, num_shards)
  //   - entry_vertex_id not resolvable inside the local shard map, or
  //     owned by a shard other than entry_shard_id
  // Fails when the runtime has not been seeded.
  Status ValidateShardLayout(shard_id_t shard_id, std::uint32_t shard_count,
                             shard_id_t entry_shard_id,
                             node_id_t entry_vertex_id) const;

  bool Ready() const;

  // Start the pinned scheduler and READ/RPC/distance workers after discovery.
  Status
  EnableDistributedSearch(std::chrono::steady_clock::time_point deadline =
                              std::chrono::steady_clock::time_point::max());

  Result<std::uint64_t> EstimateRuntimeMemory(const std::filesystem::path &root,
                                              shard_id_t local_shard) const;

  // Pins the one shard served from local memory in distributed mode. The shard
  // must already be loaded; changing the assignment requires Shutdown first.
  Status SetLocalShard(shard_id_t shard_id);

  // Returns the single shard this process serves locally. Empty when no shard
  // has been loaded yet, or when more than one shard is registered: one worker
  // owns one shard, so a caller must not add a second one.
  [[nodiscard]] std::optional<shard_id_t> LocalShardId() const;

  // ---- R1: host-injected URMA context -----------------------------------
  //
  // The algorithm library must not call urma_init itself in service mode;
  // the host (Milvus UrmaManager) owns the provider/context and injects the
  // native urma_context_t* here before EnableDistributedSearch. A nullptr
  // (the default) makes EnableDistributedSearch fail with kInvalidArgument
  // in service mode; portable tests that legitimately self-manage URMA may
  // call AcquireSelfManagedUrmaRuntime instead.
  void SetInjectedUrmaContext(void *urma_context);
  [[nodiscard]] void *InjectedUrmaContext() const;

  // Reports whether the distributed runtime is up and its RPC distance data
  // plane (mailbox) has been built. False before EnableDistributedSearch.
  bool RpcDistanceReady() const;

  // Re-runs the enable for an enabled runtime. The pinned runtime builds the
  // full RPC distance tree on success and refuses when peer resources are
  // incomplete, so this is safe to call from the late-mailbox upgrade path.
  Status RetryEnableForPushdown();

  // Diagnostic. how many of shard_map's declared shards are loaded.
  std::size_t LoadedShardCount() const;
  std::size_t ExpectedShardCount() const;

  // Atomic snapshot of the shard-load state for the distributed-enable ready
  // gate. Unlike calling LocalShardId()/LoadedShardCount()/ExpectedShardCount()
  // separately (each takes and releases state_mutex_ on its own, so a
  // concurrent LoadShard can tear the triple), this reads all three under a
  // single state_mutex_ acquisition so a caller cannot observe an inconsistent
  // combination (e.g. loaded bumped but local_id not yet updated).
  struct ShardLoadSnapshot {
    std::optional<shard_id_t> local_shard_id;
    std::size_t loaded_count;
    std::size_t expected_count;
  };
  ShardLoadSnapshot GetShardLoadSnapshot() const;

  // ---- M-6: RPC distance peer-handle path -------------------------------
  //
  // Ensures the process-local RpcMailboxMemory is constructed. Idempotent:
  // returns Ok immediately when the mailbox already exists. The mailbox
  // owns three page-aligned registered segments (requests / responses /
  // notify) that back this QN's role as both a distance-service worker
  // (peer WRITEs distance tasks into the requests segment) and as the
  // distance-service client (peer WRITEs responses into the responses
  // segment, notify segment carries WRITE_NOTIFY credits). Constructed
  // early so publishUbShardResources can advertise role 2/3 memory to the
  // URMA directory before any peer activates a link. Requires the runtime
  // to be seeded (meta_/shard_map_) so world_size is known.
  Status EnsureRpcMailbox();
  std::string RpcLayout() const;

  // Describes the three mailbox segments so the Go/CGO layer can register
  // and publish them via the URMA resource directory (role 2 = TaskQueue
  // backed by requests segment, role 3 = ResultQueue backed by responses
  // segment; notify is published as peer role 4).
  // Fails when EnsureRpcMailbox has not succeeded.
  struct RpcMailboxRegion {
    void *requests_address{nullptr};
    std::uint64_t requests_bytes{0};
    void *responses_address{nullptr};
    std::uint64_t responses_bytes{0};
    void *notify_address{nullptr};
    std::uint64_t notify_bytes{0};
    std::uint32_t ring_depth{0};
  };
  // Non-const because request_registration()/response_registration()/
  // notify_registration() (the page-aligned registration memory the URMA
  // driver needs) are non-const accessors on RpcMailboxMemory. The logical
  // memory accessors requests()/responses() have const overloads, but the
  // registration accessors do not.
  Status GetRpcMailboxRegions(RpcMailboxRegion &out);

  // Startup configuration staged from Go; runtimeOptions extends these fields.
  struct Knobs {
    std::uint32_t search_workers{0};
    std::uint32_t distance_workers{0};
    std::uint32_t read_progress_workers{0};
    std::uint32_t rpc_progress_workers{0};
    std::string cpu_bind_mode{"auto"};
    std::string cpu_layout{"contiguous"};
    std::string progress_mode{"busy"};
    std::vector<std::uint32_t> allowed_cpus;
    std::string urma_device;
    std::uint32_t eid_index{0};
    std::string runtime_options;
    // Cap on in-flight admitted HNSW tasks. Default 16 preserves the
    // pre-existing behavior (config.hpp:55 default) when Go does not
    // override. Larger values expand slot pool / queue / URMA lane
    // buffer capacity — verify device memory before raising.
    std::uint32_t max_active_queries{16};
    // URMA READ JFS (WR slot pool) depth. 0 = auto (config.cpp resolves
    // to max(1024, max_active_queries * expand_batch * 2)). Override when
    // raising max_active_queries so the slot pool has headroom.
    std::uint32_t read_jfs_depth{0};
    // Startup memory ceilings for the plan BuildRuntimeQueuePlan /
    // BuildRpcMemoryPlan produce (config.hpp queue_memory_budget_bytes and
    // registered_rpc_memory_budget_bytes). 0 keeps the built-in default.
    // Both scale with maximum_batch = M0 * expand_batch, so raising
    // expand_batch needs these raised as well.
    std::uint64_t queue_memory_budget_bytes{0};
    std::uint64_t registered_rpc_memory_budget_bytes{0};
    // Per-query wall clock and per-attempt (phase) budget. 0 keeps the
    // built-in defaults (30s / 5s). A large nq batch is only bounded by
    // query_timeout: with maxActiveQueries=1 a 125-row slice costs
    // 125 * per-query latency, so raise this before concluding that the
    // runtime is stuck.
    std::uint32_t query_timeout_ms{0};
    std::uint32_t operation_timeout_ms{0};
    // HNSW layer-0 beam width (nodes expanded per round). 0 keeps the
    // SearchParams default from query_context.hpp. Applied to BOTH the search
    // task and the startup plan sizing (maximum_batch = M0 * expand_batch).
    std::uint32_t expand_batch{0};
  };

  // Returns a snapshot of the staged knobs (thread-safe). Values are those
  // set by the most recent UbSetConfig call; defaults when Go never called.
  // SetKnobs is called by the C ABI layer (UbSetConfig); T5 topology
  // construction reads the snapshot when building thread groups.
  Knobs GetKnobs() const;
  void SetKnobs(const Knobs &knobs);

private:
  MilvusUbRuntime() = default;
  ~MilvusUbRuntime() = default;
  MilvusUbRuntime(const MilvusUbRuntime &) = delete;
  MilvusUbRuntime &operator=(const MilvusUbRuntime &) = delete;

  // Maps the Go-staged knobs onto a RuntimeConfig. Shared by
  // EnsureRpcMailbox (which publishes the mailbox before the distributed
  // runtime exists) and EnableDistributedSearchLocked, so both size the RPC
  // mailbox/mailbox plan identically. Caller holds state_mutex_.
  Result<RuntimeConfig> RequestedRuntimeConfig() const;

  // Reset everything to empty. Caller holds state_mutex_.
  void ResetLocked();

  // M-5: build+start (or return existing) distributed runtime under
  // state_mutex_. Implemented in milvus_distributed_runtime.cpp.
  Status EnableDistributedSearchLocked(
      std::chrono::steady_clock::time_point deadline =
          std::chrono::steady_clock::time_point::max());

  struct DistributedState;

  mutable std::mutex state_mutex_;
  std::shared_timed_mutex search_lifetime_mutex_;

  // Stage 5 thread-group knobs staged from Go via UbSetConfig.
  Knobs knobs_;

  // R1: host-injected native urma_context_t* (Milvus UrmaManager owns it).
  // Accessed under state_mutex_ during EnableDistributedSearch; set via
  // SetInjectedUrmaContext before enable. nullptr until the host injects.
  void *injected_urma_context_{nullptr};

  // Seeded from the first LoadedIndex; subsequent loads must agree.
  std::unique_ptr<GlobalIndexMeta> meta_;
  std::unique_ptr<ShardMap> shard_map_;
  std::unique_ptr<UpperGraphReplica> upper_;
  std::string graph_id_;
  std::uint64_t graph_epoch_{0};

  // One entry per loaded shard, in load order. Owned by the singleton.
  std::vector<std::unique_ptr<LocalShard>> local_shards_;
  std::vector<shard_id_t> loaded_shard_ids_;
  std::map<shard_id_t, ShardEntryPoint> shard_entries_;

  // M-3 bridge: peer (shard, role) -> imported native handle. The rjetty
  // descriptor bytes are owned by the runtime (copied on import so UMDK has
  // a stable address); the segment pointer is borrowed from the host URMA
  // manager and not freed by the runtime.
  struct PeerHandle {
    std::vector<std::byte> rjetty_descriptor;
    std::uint32_t jetty_token{0};
    void *segment{nullptr};
  };
  std::map<std::pair<shard_id_t, std::uint32_t>, PeerHandle> peer_handles_;

  // M-5: scheduler-backed runtime; null until EnableDistributedSearch.
  std::unique_ptr<DistributedState> distributed_runtime_;

  // M-6: process-local RPC mailbox, constructed by EnsureRpcMailbox before
  // the URMA directory publishes role 2/3 (distance-tasks/distance-results)
  // so peers can bond to it. Reused by EnableDistributedSearchLocked when
  // building the bundle (segments 2/3/4). Null until EnsureRpcMailbox.
  std::unique_ptr<RpcMailboxMemory> rpc_memory_;
};

Result<RuntimeConfig>
BuildMilvusRuntimeConfig(const MilvusUbRuntime::Knobs &knobs);

} // namespace hnsw_urma

#endif // HNSW_URMA_MILVUS_UB_RUNTIME_H_
