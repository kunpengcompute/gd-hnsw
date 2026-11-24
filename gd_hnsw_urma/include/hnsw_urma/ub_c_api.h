// Minimal C ABI surface exposed by the hnsw_urma runtime for Milvus (Go/CGO).
// Keep this header C-compatible: no C++ constructs, no STL in signatures.
#ifndef HNSW_URMA_UB_C_API_H_
#define HNSW_URMA_UB_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Smoke test: returns x + 1.
int UbPing(int x);

// R1: inject the host-owned native urma_context_t* before
// UbEnableDistributedSearch. The algorithm library borrows the context and
// must NOT call urma_init itself in service mode. Pass NULL only in
// self-managed portable tests. Returns 0 on success, non-zero hnsw_urma
// StatusCode if the runtime is already enabled (context must be set before
// enable).
int UbSetUrmaContext(void *urma_context);

// Stage UB runtime configuration in one call (from Milvus yaml via CGO).
// All fields are stashed into the runtime singleton's Knobs and consumed
// when the first shard is loaded (ResolveRuntimeConfig needs the index meta).
//
// Semantics: every field with a 0 / NULL value keeps the algorithm library's
// built-in default (the C++ side treats 0 as "unset", not as 0). String
// fields accept NULL (= keep default). This single call replaces the prior
// UbSetConfig / UbSetMemoryBudget / UbSetTimeouts / UbSetExpandBatch /
// UbSetRuntimeOptions / UbSetUrmaEndpoint scatter-set ABI.
//
// Must be called BEFORE the first shard load / UbEnableDistributedSearch.
// Calling after the distributed runtime is enabled returns kInternalError
// and leaves the staged knobs unchanged.
//
// Forward compatibility: only append fields to the tail of UbConfig; never
// reorder, rename, or remove. The C++ side reads fields by offset, so a
// newer struct with extra trailing fields is read correctly by an older
// library (the older library ignores the tail).
typedef struct UbConfig {
  uint32_t search_workers;        // 0 = auto (upstream formula)
  uint32_t distance_workers;      // 0 = auto
  uint32_t read_progress_workers; // 0 = auto
  uint32_t rpc_progress_workers;  // 0 = auto
  const char *cpu_bind_mode;      // NULL="auto"; "auto"|"none"
  const char *
      cpu_layout; // NULL="contiguous"; "contiguous"|"llc-isolated"|"llc-spread"
  const char *progress_mode; // NULL="busy"; "busy"|"adaptive"
  const uint32_t
      *allowed_cpus; // explicit CPU list; allowed_cpu_count==0 = not set
                    // non-NULL when allowed_cpu_count > 0
  uint32_t allowed_cpu_count;
  uint32_t max_active_queries;                 // 0 = built-in default (16)
  uint32_t read_jfs_depth;                     // 0 = auto
  uint64_t queue_memory_budget_bytes;          // 0 = built-in default
  uint64_t registered_rpc_memory_budget_bytes; // 0 = built-in default
  uint32_t query_timeout_ms;                   // 0 = built-in default (30s)
  uint32_t operation_timeout_ms;               // 0 = built-in default (5s)
  uint32_t expand_batch;                       // 0 = SearchParams default
  const char *runtime_options;  // NULL=""; free-form key=value overrides
  const char *urma_device_name; // NULL = cleared; URMA device for
                                // EnableDistributedSearch
  uint32_t urma_eid_index;      // EID index on the URMA device
  // Forward-compat tail: only append, never reorder/remove.
} UbConfig;

int UbConfigure(const UbConfig *cfg);

// Resolve the configured runtime memory plan before loading/registering a
// shard.
int UbEstimateRuntimeMemory(const char *root, uint32_t shard, uint64_t *bytes);

// Stage 4: shard loading + synchronous local search.
//
// Synchronous service entry for nq query vectors. The worker's shard-local
// entry is used; results have nq*topk entries padded with -1/+inf.
// metric_name must match the loaded index (L2 or IP); unknown metrics fail.
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
int UbSearch(const float *query, uint32_t nq, uint32_t dim, uint32_t topk,
             uint32_t ef_search, const char *metric_name, int64_t *out_ids,
             float *out_dists);

// Request remaining wall-clock budget in nanoseconds, including native queue
// time. -1 retains the configured timeout; 0 fails immediately. The call owns
// all asynchronous query storage and never retains caller input/output
// pointers.
int UbSearchWithTimeout(const float *query, uint32_t nq, uint32_t dim,
                        uint32_t topk, uint32_t ef_search,
                        const char *metric_name, int64_t timeout_ns,
                        int64_t *out_ids, float *out_dists);

// Releases all singleton state. Safe to call multiple times.
int UbStopDistributedSearch(void);
int UbShutdown(void);

// Pins the shard this process serves from local memory. One worker owns one
// shard; the call validates that the shard is loaded locally.
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
int UbSetLocalShard(uint32_t shard_id);

// Reports whether the distributed runtime is up and its distance data plane
// (RPC mailbox) is available for this process. False before
// EnableDistributedSearch and after a stop.
int UbRpcDistanceReady(void);

// v2 (PROTOCOL.md ABI evolution v2): shard-load status snapshot struct.
// Atomically read by UbGetShardStatus; replaces the prior
// UbGetLocalShardId + UbLoadedShardCount + UbExpectedShardCount triple
// (which had a TOCTOU window in the distributed-enable ready gate).
//
// Forward compatibility: only append fields to the tail; never reorder,
// rename, or remove. An older library reading a newer struct ignores the
// trailing fields; a newer library reading an older struct sees 0 in the
// tail (zero-initialized by the caller).
#define UB_NO_LOCAL_SHARD 0xFFFFFFFFu // local_shard_id not pinned

typedef struct UbShardStatus {
  uint32_t local_shard_id; // UB_NO_LOCAL_SHARD = not pinned; else
                           // UbSetLocalShard value
  uint32_t loaded_count;   // replaces UbLoadedShardCount
  uint32_t expected_count; // replaces UbExpectedShardCount
  uint32_t reserved;       // forward-compat, tail-only append
} UbShardStatus;

// One atomic snapshot of all shard-load status. Replaces the prior
// UbGetLocalShardId + UbLoadedShardCount + UbExpectedShardCount triple.
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
int UbGetShardStatus(UbShardStatus *out);

// v2: global-graph shard reuse result. When UbReuseGlobalGraphShard reports a
// hit (out_reused set to 1), the caller skips re-staging/re-ingesting the
// shard and adopts these seeded dimensions. Mirrors
// MilvusUbRuntime::SeededShardInfo via the C ABI so the host (UBHNSWIndex)
// can reuse an already-seeded process-wide graph without touching the runtime
// object directly. Forward-compat: tail-only append, same rule as
// UbShardStatus.
typedef struct UbSeededShardInfo {
  uint32_t dim;      // dimensionality of the seeded local graph shard
  uint64_t count;    // row count of the seeded local graph shard
  uint32_t reserved; // forward-compat, tail-only append
} UbSeededShardInfo;

// Probe whether the process-wide graph already holds the (graph_id, epoch,
// shard) triple from a prior segment seed. On a hit the host reuses the
// seeded shard and skips re-ingest (sets out_reused=1, fills *out_info).
// On a miss (graph not seeded yet, or shard not local) returns 0 with
// out_reused=0 so the host proceeds with the normal ingest path. A non-zero
// return is an hnsw_urma StatusCode error (graph/epoch mismatch etc.) and
// must be surfaced as a load failure.
int UbReuseGlobalGraphShard(const char *graph_id, uint64_t epoch,
                            uint32_t shard, int *out_reused,
                            UbSeededShardInfo *out_info);

// v2: index-metadata projection struct. Shared by UbLoadShardFromDir's
// optional out_meta and UbReadIndexMeta (replaces the prior 4-scalar out
// form, so callers cannot swap argument order). Forward-compat: tail-only
// append, same rule as UbConfig / UbShardStatus.
typedef struct UbIndexMeta {
  uint32_t dim;
  uint64_t ntotal;
  uint32_t shard_count;
  uint64_t local_count; // rows owned by shard_id
  uint64_t reserved;    // forward-compat
} UbIndexMeta;

// Load exactly one shard (shard_id) from a local triple directory
// (global.meta + upper_graph.index + shard_N.index). Performs the full
// LoadIndexForShard → C4 (sum(local_count)==ntotal) consistency check →
// runtime LoadShard pipeline; the C4 check is pushed down from Milvus so the
// host no longer needs to read algorithm internals. Idempotent per shard.
// On failure, UbLastError carries the status detail (the int return alone is
// only the StatusCode category).
//
// v2: when out_meta is non-NULL, the loaded shard's metadata projection is
// written back so the host no longer needs a follow-up UbReadIndexMeta to
// keep its index object coherent (saves one ABI round-trip at the
// ub_hnsw_index.cpp LoadV2Index call site). out_meta==NULL = caller does not
// need the meta read-back (behavior unchanged from the prior no-out_meta
// signature).
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
int UbLoadShardFromDir(const char *triple_dir, uint32_t shard_id,
                       UbIndexMeta *out_meta);

// Read global.meta from a triple directory without loading any shard. Used by
// the host when a segment does not carry this process's shard: it still needs
// dim / ntotal / shard_count, and the row count owned by `shard_id`
// (out_local_count), to keep the index object coherent.
// shard_id selects which shard's local_count is reported in
// out_meta->local_count; pass a shard_id outside [0, shard_count) to leave it
// unset (0).
//
// v2: out is a single UbIndexMeta struct (replaces the prior 4-scalar out
// form so callers cannot swap argument order). out_meta must be non-null.
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure (UbLastError
// carries the detail).
int UbReadIndexMeta(const char *triple_dir, uint32_t shard_id,
                    UbIndexMeta *out_meta);

// Returns a human-readable detail string for the most recent Ub* failure on
// the calling thread, or NULL when the last call succeeded. The pointer is
// valid only until the next Ub* call on the same thread; callers must copy.
const char *UbLastError(void);

// Stage 5: seed the singleton directly from a local triple directory
// (global.meta + upper_graph.index + shard_N.index).
//
// v2: merged the prior UbSeedFromDir (full) + UbSeedFromDirWithShards (subset)
// into one entry. shard_ids==NULL or count==0 = seed every shard range
// advertised by global.meta; otherwise seed only the listed subset (true
// distributed mode: each QN holds its own shard subset, peers read over
// URMA). A shard id outside [0, num_shards) is an error. Idempotent per
// shard (already-loaded shards are skipped).
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
int UbSeedFromDir(const char *triple_dir, const uint32_t *shard_ids,
                  uint32_t count);

// Bind a successfully seeded global graph to its immutable artifact identity.
// Idempotent for the same graph/epoch; a different identity requires shutdown.
int UbBindGlobalGraphIdentity(const char *graph_id, uint64_t epoch);

// M-3 bridge: peer handle injection. Adopts the native UMDK handles behind
// an activated host URMA link for the peer that owns shard `shard_id`.
// The host exports a self-described endpoint (raw rjetty descriptor bytes +
// jetty token) plus the opaque target segment pointer; the algorithm library
// copies the descriptor bytes so UMDK has a stable address and forwards the
// segment verbatim. This keeps the algorithm free of the host's
// UrmaPeerEndpointInfo type. The runtime does NOT take ownership of the
// segment; the host link must stay alive until UbShutdown or
// UbRemovePeerHandles. rjetty_descriptor may be NULL only when
// descriptor_bytes==0 (rejected). role: 0=topology, 1=vectors,
// 2=distance-tasks, 3=distance-results, 4=notify. Returns 0 on success,
// non-zero hnsw_urma StatusCode on failure.
int UbImportPeerHandles(uint32_t shard_id, uint32_t role,
                        const void *rjetty_descriptor,
                        uint32_t descriptor_bytes, uint32_t jetty_token,
                        void *target_segment);

// Removes the stored handle pair for (shard_id, role) without freeing any
// native resource (ownership stays with UrmaManager).
void UbRemovePeerHandles(uint32_t shard_id, uint32_t role);

// v4 (PROTOCOL.md ABI evolution v4): publish-snapshot structs.
// UbGetGraphPublishSnapshot replaces the prior UbGetShardRegions +
// UbGetEntryPoint + UbGetEntryShard triple (and absorbs the publish-site
// UbExpectedShardCount call). server.go's publishUbShardGraphResources
// always called those four together to build one directory record; the
// snapshot makes that one atomic read so a topology change mid-publish
// cannot produce an inconsistent record (the prior triple had a TOCTOU
// window). Forward-compat: tail-only append, same rule as UbConfig.

typedef struct UbShardRegion {
  uint32_t shard_id;
  void *graph_address;
  uint64_t graph_bytes;
  void *vector_address;
  uint64_t vector_bytes;
} UbShardRegion;

typedef struct UbGraphPublishSnapshot {
  // Per-shard registration memory of this node's locally loaded shards
  // (M-3 publish-side). The C side always leaves this nullptr: the caller
  // owns the out_regions buffer it passed to UbGetGraphPublishSnapshot and
  // reads region_count entries from it. (Storing the caller's possibly
  // Go-heap pointer back into this Go-heap struct would trip cgo's pointer
  // checker, so the back-pointer is intentionally not written.)
  const UbShardRegion *regions; // always nullptr; use the caller-owned buffer
  uint32_t region_count;
  // Global topology (M-3/M-4 publish fields).
  uint32_t shard_count; // global shard total (replaces UbExpectedShardCount at
                        // publish)
  uint32_t entry_shard_id;  // shard owning the global entry point (replaces
                            // UbGetEntryShard)
  uint64_t entry_vertex_id; // global entry-point vertex id (replaces
                            // UbGetEntryPoint)
  int32_t entry_level; // entry point's level (UbGetEntryPoint also returned it)
  uint32_t reserved;   // forward-compat, tail-only append
} UbGraphPublishSnapshot;

// One atomic snapshot of everything publishUbShardGraphResources needs to
// build a directory record. Replaces UbGetShardRegions + UbGetEntryPoint +
// UbGetEntryShard (+ the publish-site UbExpectedShardCount call).
//
// capacity = caller-provided out_regions array length. Two-phase use is
// supported: pass out_regions==NULL / capacity==0 to only query
// out_snapshot->region_count, then allocate and call again to fill it.
// out_snapshot->regions is always left nullptr (the caller owns the buffer
// it passed and reads region_count entries from it).
// Returns 0 on success, negative hnsw_urma StatusCode on error.
int UbGetGraphPublishSnapshot(uint32_t capacity, UbShardRegion *out_regions,
                              UbGraphPublishSnapshot *out_snapshot);

// M-4 shard numbering same-source validation. Checks the identity of a
// published etcd ResourceRecord (shard_count, shard_id, entry point) against
// the locally seeded global.meta topology. Use before activating links for a
// shard: a mismatch means the directory record does not belong to this graph
// build (e.g. a stale or mis-tagged publication).
// Returns 0 when the record matches the local topology, non-zero
// hnsw_urma StatusCode otherwise.
int UbValidateShardLayout(uint32_t shard_id, uint32_t shard_count,
                          uint32_t entry_shard_id, uint64_t entry_vertex_id);

// M-5: build and start the distributed thread-group runtime (scheduler +
// control/search/distance/read_progress/rpc_progress threads). Requires
// the runtime to be seeded (all shards loaded). Idempotent: returns 0
// when the runtime is already up.
//
// v3: merged UbRetryEnableForPushdown's optional msg buffer into this entry.
// msg==NULL or msg_capacity==0 = caller does not need a reason string;
// non-NULL = on failure, a nul-terminated reason is written (truncated to
// msg_capacity). The runtime builds the full RPC distance tree on success
// and refuses when peer resources are incomplete.
// Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
int UbEnableDistributedSearch(char *msg, int msg_capacity);

// Remaining startup budget, including native lock waits. -1 keeps legacy
// behavior; 0 fails before provisioning. Synchronous driver calls must return
// before their deadline can be checked; no abandoned background calls are used.
int UbEnableDistributedSearchWithTimeout(int64_t timeout_ns);
// M-6: construct the process-local RPC distance mailbox (idempotent).
// Must be called after UbSeedFromDir/UbLoadShard so world_size is known.
// The mailbox backs role 2 (distance-tasks) + role 3 (distance-results)
// publish and is later reused by UbEnableDistributedSearch as the bundle's
// requests/responses/notify segments.
//
// v3: merged the prior UbEnsureRpcMailbox (void) + UbGetRpcMailboxRegions
// two-call sequence into one entry. server.go always called Ensure then
// Get back-to-back; the merged call builds the mailbox and returns its
// three segment addresses/sizes + ring_depth in one round-trip. out must
// be non-null. Returns 0 on success, non-zero hnsw_urma StatusCode on failure.
typedef struct UbRpcMailboxRegions {
  void *requests_address;
  uint64_t requests_bytes; // role 2 TaskQueue
  void *responses_address;
  uint64_t responses_bytes; // role 3 ResultQueue
  void *notify_address;
  uint64_t notify_bytes; // role 4 peer-facing
  uint32_t ring_depth;
  uint32_t reserved; // forward-compat, tail-only append
} UbRpcMailboxRegions;

int UbEnsureRpcMailbox(UbRpcMailboxRegions *out);

// Thread-local string, valid until the next call on this thread.
const char *UbRpcLayout(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HNSW_URMA_UB_C_API_H_
