/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include "gd_layout.h"
#include "distance_kernel_dot_norm.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace gd_hnsw {
class DualDistProxy;
} // namespace gd_hnsw

namespace gd_hnsw {

// ============================================================
// VisitedTable — per-thread, NOT in gd
// Dual-mode: dense array for small graphs (<= VISITED_SPARSE_THRESHOLD),
//            flat open-addressing hash table for large graphs.
// The flat hash table uses epoch stamping for O(1) advance() and
// provides cache-friendly linear probing (fits in L2 cache).
// ============================================================

class VisitedTable {
public:
    // max_visits: expected upper bound of distinct ids visited per query
    // (e.g. ef_search * level0_neighbors). 0 = default (FLAT_CAPACITY).
    explicit VisitedTable(uint64_t ntotal, uint32_t max_visits = 0);

    // Capacity for a visit budget: max(next_pow2(2 * max_visits), FLAT_CAPACITY).
    // 0 budget -> default capacity.
    static uint32_t flat_capacity_for(uint32_t max_visits);

    void advance(); // start a new query
    bool get(int32_t id) const;
    // Returns true if newly inserted (was not visited), false if already visited.
    // Combines check + mark in one operation to avoid double lookup.
    bool set(int32_t id);

    // Prefetch visited entry into cache.
    // Dense mode: prefetch visited_[id] (direct index).
    // Sparse mode: prefetch the hash slot for id (flat hash table).
    // Uses write hint (1) since set() will write on miss.
    void prefetch(int32_t id) const
    {
        if (!use_sparse_) {
            __builtin_prefetch(&visited_[id], 1, 2);
        } else {
            __builtin_prefetch(&flat_table_[flat_hash(id) & flat_mask_], 1, 2);
        }
    }

private:
    uint64_t ntotal_;
    bool use_sparse_;

    // dense mode (uint8_t like faiss for cache efficiency)
    std::vector<uint8_t> visited_;
    uint8_t generation_ = 1;

    // sparse mode: flat open-addressing hash table with epoch stamp
    // Packed uint64_t: high 32 bits = stamp, low 32 bits = key (as uint32_t).
    // Single 8-byte load/store per probe — no struct field splitting.
    static constexpr uint32_t FLAT_CAPACITY = 65536; // minimum capacity: 512KB, fits in L2
    static constexpr uint32_t MAX_PROBE = 512;       // relaxed probe limit for large ef_search
    std::vector<uint64_t> flat_table_;
    uint32_t flat_mask_ = 0; // capacity - 1 (power of 2)
    uint32_t flat_stamp_ = 0;
    uint32_t flat_capacity_ = FLAT_CAPACITY; // runtime capacity (>= FLAT_CAPACITY)

    static uint32_t flat_hash(int32_t id)
    {
        return static_cast<uint32_t>(id) * 2654435761u; // fibonacci hashing
    }

    // Pack/unpack helpers
    static uint64_t flat_pack(int32_t key, uint32_t stamp)
    {
        return (static_cast<uint64_t>(stamp) << 32) | static_cast<uint32_t>(key);
    }
    static uint32_t flat_stamp_of(uint64_t slot) { return static_cast<uint32_t>(slot >> 32); }
    static int32_t flat_key_of(uint64_t slot) { return static_cast<int32_t>(slot & 0xFFFFFFFF); }
};

// ============================================================
// MinimaxHeap — combined candidate + result heap
// Dual-heap design: result max-heap (bounded ef_search) +
// candidate min-heap with lazy invalidation.
// pop_min_candidate() is O(log n) amortized instead of O(n).
// Public API unchanged from the original linear-scan version.
// ============================================================

class MinimaxHeap {
public:
    int n; // capacity (ef_search)

    explicit MinimaxHeap(int capacity) : n(capacity)
    {
        entries_.reserve(capacity * 2); // may accumulate more than n due to lazy deletion
        result_heap_.reserve(capacity + 1);
        cand_heap_.reserve(capacity * 2);
    }

    void push(int32_t id, float d)
    {
        int eidx = static_cast<int>(entries_.size());
        entries_.push_back({d, id, true, false}); // in_result=true, expanded=false

        if (result_size_ < n) {
            // Result not full yet — add directly
            result_size_++;
            result_heap_push(eidx);
            cand_heap_push(eidx);
            ncand_++;
        } else {
            // Result full — only insert if better than current max
            int worst = result_heap_[0];
            if (d >= entries_[worst].dist) {
                // Not good enough, discard
                entries_.back().in_result = false;
                return;
            }
            // Evict the worst result
            if (!entries_[worst].expanded)
                ncand_--;
            entries_[worst].in_result = false;
            result_heap_pop();
            // Insert new entry
            result_size_++;
            result_heap_push(eidx);
            cand_heap_push(eidx);
            ncand_++;
        }
    }

    float max() const { return entries_[result_heap_[0]].dist; }

    int size() const { return result_size_; }

    int candidates() const { return ncand_; }

    // Pop the closest unexpanded candidate. O(log n) amortized.
    // Marks it as expanded but keeps it in results.
    std::pair<float, int32_t> pop_min_candidate()
    {
        while (!cand_heap_.empty()) {
            int eidx = cand_heap_[0];
            auto &e = entries_[eidx];
            if (!e.in_result || e.expanded) {
                // Lazy skip: entry was evicted from results or already expanded
                cand_heap_pop();
                continue;
            }
            // Valid candidate found
            e.expanded = true;
            ncand_--;
            cand_heap_pop();
            return {e.dist, e.id};
        }
        return {std::numeric_limits<float>::max(), -1};
    }

    // Extract all result entries sorted by distance (ascending).
    void extract_sorted(std::vector<std::pair<float, int32_t>> &out) const
    {
        out.clear();
        out.reserve(result_size_);
        for (const auto &e : entries_) {
            if (e.in_result) {
                out.push_back({e.dist, e.id});
            }
        }
        std::sort(out.begin(), out.end());
    }

    void clear()
    {
        entries_.clear();
        result_heap_.clear();
        cand_heap_.clear();
        result_size_ = 0;
        ncand_ = 0;
        // Capacity governance with hysteresis: only shrink after extreme outliers,
        // and retain enough headroom to avoid shrink/regrow churn on normal queries.
        // Trigger: capacity > 32*n (extreme outlier).  Retain: 8*n after shrink.
        size_t shrink_trigger = static_cast<size_t>(n) * 32;
        size_t retain_cap = static_cast<size_t>(n) * 8;
        if (entries_.capacity() > shrink_trigger) {
            entries_.shrink_to_fit();
            entries_.reserve(retain_cap);
        }
        if (cand_heap_.capacity() > shrink_trigger) {
            cand_heap_.shrink_to_fit();
            cand_heap_.reserve(retain_cap);
        }
        // result_heap_ is bounded by n, no governance needed
    }

private:
    struct Entry {
        float dist;
        int32_t id;
        bool in_result;
        bool expanded;
    };

    std::vector<Entry> entries_;
    std::vector<int> result_heap_; // max-heap of entry indices (by dist)
    std::vector<int> cand_heap_;   // min-heap of entry indices (by dist)
    int result_size_ = 0;
    int ncand_ = 0;

    // --- Helpers for deterministic tie-breaking ---
    // When distances are equal, break ties by id for deterministic behavior.
    // result max-heap: "a > b" means a should be closer to root
    bool result_greater(int a, int b) const
    {
        float da = entries_[a].dist, db = entries_[b].dist;
        return da > db || (da == db && entries_[a].id < entries_[b].id);
    }
    bool result_geq(int a, int b) const
    {
        float da = entries_[a].dist, db = entries_[b].dist;
        return da > db || (da == db && entries_[a].id <= entries_[b].id);
    }
    // cand min-heap: "a < b" means a should be closer to root
    bool cand_less(int a, int b) const
    {
        float da = entries_[a].dist, db = entries_[b].dist;
        return da < db || (da == db && entries_[a].id < entries_[b].id);
    }
    bool cand_leq(int a, int b) const
    {
        float da = entries_[a].dist, db = entries_[b].dist;
        return da < db || (da == db && entries_[a].id <= entries_[b].id);
    }

    // --- Result max-heap operations ---
    void result_heap_push(int eidx)
    {
        result_heap_.push_back(eidx);
        // sift up (max-heap: parent >= child)
        int pos = static_cast<int>(result_heap_.size()) - 1;
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (result_geq(result_heap_[parent], result_heap_[pos]))
                break;
            std::swap(result_heap_[parent], result_heap_[pos]);
            pos = parent;
        }
    }

    void result_heap_pop()
    {
        result_heap_[0] = result_heap_.back();
        result_heap_.pop_back();
        result_size_--;
        if (result_heap_.empty())
            return;
        // sift down
        int pos = 0;
        int sz = static_cast<int>(result_heap_.size());
        int child = 2 * pos + 1;
        while (child < sz) {
            if (child + 1 < sz && result_greater(result_heap_[child + 1], result_heap_[child]))
                child++;
            if (result_geq(result_heap_[pos], result_heap_[child]))
                break;
            std::swap(result_heap_[pos], result_heap_[child]);
            pos = child;
            child = 2 * pos + 1;
        }
    }

    // --- Candidate min-heap operations ---
    void cand_heap_push(int eidx)
    {
        cand_heap_.push_back(eidx);
        // sift up (min-heap: parent <= child)
        int pos = static_cast<int>(cand_heap_.size()) - 1;
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (cand_leq(cand_heap_[parent], cand_heap_[pos]))
                break;
            std::swap(cand_heap_[parent], cand_heap_[pos]);
            pos = parent;
        }
    }

    void cand_heap_pop()
    {
        cand_heap_[0] = cand_heap_.back();
        cand_heap_.pop_back();
        if (cand_heap_.empty())
            return;
        // sift down
        int pos = 0;
        int sz = static_cast<int>(cand_heap_.size());
        int child = 2 * pos + 1;
        while (child < sz) {
            if (child + 1 < sz && cand_less(cand_heap_[child + 1], cand_heap_[child]))
                child++;
            if (cand_leq(cand_heap_[pos], cand_heap_[child]))
                break;
            std::swap(cand_heap_[pos], cand_heap_[child]);
            pos = child;
            child = 2 * pos + 1;
        }
    }
};

// ============================================================
// HnswQueryParams — per-query parameters for the internal searcher.
// (Renamed from SearchParams to free the name for the public API in
//  gd_hnsw_api.h, which also lives in namespace gd_hnsw.)
// ============================================================

struct HnswQueryParams {
    int32_t ef_search = 0; // 0 = use sb.ef_search
    int32_t k = 10;        // top-k results
};

// ============================================================
// ShardView — cached pointers for one shard
// ============================================================

struct ShardView {
    const SuperBlockV1 *sb;
    const float *vectors;
    const int32_t *levels;
    const uint64_t *offsets;
    const int32_t *neighbors;
    const int32_t *cum_nneighbor;
    uint64_t global_id_begin;
    uint64_t ntotal_local;
};

// ============================================================
// GdHnswSearcher — stateful searcher bound to gd shards
// ============================================================

class GdHnswSearcher {
public:
    // Single-shard constructor (backward compatible)
    // skip_validation: if true, skip full CRC + structural validation
    // (caller guarantees data was already validated, e.g. during load)
    GdHnswSearcher(const void *gd_base, uint64_t gd_size, bool skip_validation = false);

    // Multi-shard constructor: takes array of (base, size) pairs.
    // Throws std::runtime_error if shards.size() > 32: pushdown search paths
    // key shard masks by bit position (uint32_t touched/pending_mask, 1u << s,
    // shard_order[32]), so shard count must fit in 32 bits.
    GdHnswSearcher(const std::vector<std::pair<const void *, uint64_t>> &shards, bool skip_validation = false);

    // Non-copyable, non-movable (ShardView contains raw pointers)
    GdHnswSearcher(const GdHnswSearcher &) = delete;
    GdHnswSearcher &operator=(const GdHnswSearcher &) = delete;
    GdHnswSearcher(GdHnswSearcher &&) = delete;
    GdHnswSearcher &operator=(GdHnswSearcher &&) = delete;

    // Load LOCAL shard's vector data directly from file to process-local heap.
    // This replaces the old localize_vectors() flow (GD → heap copy).
    // After this call, ShardView.vectors for the local shard points to heap memory.
    // Remote shard ShardView.vectors remain nullptr (pushdown mode doesn't need them).
    // Also caches the entry_point vector if it resides on a remote shard.
    // Requires: set_local_shard() must have been called first.
    // Must be called BEFORE pushdown service setup (service workers copy ShardView).
    // Idempotent: second call is a no-op.
    //
    // Parameters:
    //   file_path: path to the shard file (v1 format, contains vectors at BLK_VECTORS offset)
    //   vectors_data: alternatively, pointer to pre-loaded vector data (e.g. from build buffer).
    //                 If non-null, data is memcpy'd from here instead of reading from file.
    //                 Must contain ntotal_local * dim floats for the local shard.
    void load_local_vectors(const std::string &file_path);
    void load_local_vectors(const float *vectors_data);

    // Inject cached entry_point vector (for when EP is on a remote shard).
    // Must be called after load_local_vectors() if EP was not in the local shard.
    void set_ep_vector(const float *ep_vec, uint32_t dim) { ep_vector_cache_.assign(ep_vec, ep_vec + dim); }

    // Zero-allocation search: write results directly to caller buffers.
    // out_ids and out_dists must point to arrays of size >= params.k.
    void search_to(const float *query, const HnswQueryParams &params, int32_t *out_ids, float *out_dists) const;

    // Flat-buffer batch search: results written to pre-allocated arrays.
    // out_ids[nq*k], out_dists[nq*k] must be pre-allocated by caller.
    // Parallelism: spawns an OMP team over queries when called from a serial
    // context. Inside an existing OMP parallel region the loop runs inline on
    // the calling thread (serial) — a nested region would reset
    // omp_get_thread_num() to 0 and corrupt the [tid][shard] DualDistProxy
    // indexing in search_to(). Concurrent calls with disjoint output buffers
    // are safe (state is thread_local; pushdown proxies are per tid).
    void search_batch_flat(const float *queries, int32_t nq, const HnswQueryParams &params, int32_t *out_ids,
                           float *out_dists) const;

    // Which shard does this process own (for local/remote classification)
    void set_local_shard(uint32_t shard_id)
    {
        if (shard_id >= num_shards_) {
            fprintf(stderr, "FATAL: set_local_shard(%u) out of range (num_shards=%u)\n", shard_id, num_shards_);
            abort();
        }
        local_shard_id_ = shard_id;
        // Precompute local ID range for fast is_local check
        uint32_t sorted_idx = shard_id_to_sorted_idx_[shard_id];
        local_id_begin_ = shards_[sorted_idx].global_id_begin;
        local_id_end_ = local_id_begin_ + shards_[sorted_idx].ntotal_local;
        // Precompute shift for power-of-2 shard sizes (replaces integer division)
        shard_shift_valid_ = false;
        if (uniform_shards_ && ntotal_per_shard_ > 0 && (ntotal_per_shard_ & (ntotal_per_shard_ - 1)) == 0) {
            shard_shift_ = __builtin_ctzll(ntotal_per_shard_);
            shard_shift_valid_ = true;
        }
    }

    // Inject dual-channel pushdown proxies (split task/result channels).
    void set_dual_pushdown_proxies(std::vector<std::vector<DualDistProxy>> *proxies_2d, uint32_t num_search_threads)
    {
        dual_pushdown_proxies_2d_ = proxies_2d;
        num_search_threads_ = num_search_threads;
    }

    // Set multi-pop expansion batch size: pop up to N candidates per iteration.
    // 1 = original single-candidate (default). Higher values merge remote batches.
    void set_expand_batch(uint32_t n) { expand_batch_ = (n > 0) ? n : 1; }
    uint32_t expand_batch() const { return expand_batch_; }

    // Accessors
    uint64_t ntotal() const { return ntotal_; }
    uint32_t dim() const { return dim_; }
    const std::vector<ShardView> &shards() const { return shards_; }
    const ShardView &shard_by_id(uint32_t shard_id) const;

private:
    // Get vector pointer for a global id
    const float *get_vector(int32_t global_id) const;

    // Fast path: get vector when sorted_idx is already known (skips find_shard)
    inline const float *get_vector_in_shard(int32_t global_id, uint32_t sorted_idx) const
    {
        uint64_t local_id = static_cast<uint64_t>(global_id) - shards_[sorted_idx].global_id_begin;
        return shards_[sorted_idx].vectors + local_id * dim_;
    }

    // Prefetch vector data into L1 cache when sorted_idx is already known.
    inline void prefetch_vector_in_shard(int32_t global_id, uint32_t sorted_idx) const
    {
        uint64_t local_id = static_cast<uint64_t>(global_id) - shards_[sorted_idx].global_id_begin;
        __builtin_prefetch(shards_[sorted_idx].vectors + local_id * dim_, 0, 3);
    }

    // Prefetch vector data for any global id (uses find_shard).
    inline void prefetch_vector(int32_t global_id) const
    {
        const float *v = get_vector(global_id);
        __builtin_prefetch(v, 0, 3);
    }

    // Get neighbor range for a global id at a given layer
    // Returns begin/end indices and the neighbors array pointer
    void get_neighbor_range(int32_t global_id, int layer, uint64_t &begin_out, uint64_t &end_out,
                            const int32_t *&neighbors_out) const;

    // Local search: greedy descent through upper layers (single-shard, no pushdown)
    int32_t greedy_search_upper_local(const float *query, int32_t ep, int level, int target_level,
                                      float &ep_dist) const;

    // Local search: beam search at layer 0 (single-shard, no pushdown)
    void search_at_layer_local(const float *query, int32_t ep, float ep_dist, int level, int ef_search,
                               std::vector<std::pair<float, int32_t>> &results, VisitedTable &vt) const;

    // Pushdown beam search: remote distances computed by DualDistService
    // proxies: this thread's proxy array (one per shard), indexed by shard_id
    template <typename ProxyT>
    void search_at_layer_pushdown(const float *query, int32_t ep, float ep_dist, int level, int ef_search,
                                  std::vector<std::pair<float, int32_t>> &results, VisitedTable &vt,
                                  std::vector<ProxyT> &proxies) const;

    void init_shard(const void *gd_base, uint64_t gd_size, bool skip_validation = false);

    // Global index params
    uint64_t ntotal_;
    uint32_t dim_;
    uint32_t ef_search_;
    int32_t max_level_;
    int32_t entry_point_;
    uint32_t num_shards_;
    uint64_t ntotal_per_shard_;   // used for fast O(1) lookup when shards are uniform
    bool uniform_shards_;         // true if all shards have same ntotal_local
    uint32_t local_shard_id_ = 0; // this process's shard (for local/remote classification)
    // Precomputed local shard range for fast is_local check (avoids find_shard + division)
    uint64_t local_id_begin_ = 0; // inclusive
    uint64_t local_id_end_ = 0;   // exclusive
    // Shift-based shard lookup (valid when uniform_shards_ && ntotal_per_shard_ is power of 2)
    uint32_t shard_shift_ = 0;
    bool shard_shift_valid_ = false;
    // Max level-0 neighbors per node across shards (VisitedTable sizing input)
    uint32_t level0_nn_max_ = 0;

    // Pushdown state (set via set_dual_pushdown_proxies, nullptr = disabled)
    // 2D: [search_thread][shard], each search thread has its own proxy per shard
    std::vector<std::vector<DualDistProxy>> *dual_pushdown_proxies_2d_ = nullptr;
    uint32_t num_search_threads_ = 0;

    // Multi-pop expansion: pop up to N candidates per iteration and classify
    // all their neighbors before submitting. Merges small remote batches
    // naturally. 1 = original single-candidate expansion (default).
    uint32_t expand_batch_ = 1;
    // Find shard index for a global id
    uint32_t find_shard(int32_t global_id) const;

    // Mapping between sorted shard vector index and logical shard_id.
    // sorted_idx -> shard_id (used by find_shard routing)
    std::vector<uint32_t> sorted_to_shard_id_;
    // shard_id -> sorted_idx (used by data access + DualDistService binding)
    std::vector<uint32_t> shard_id_to_sorted_idx_;

    // Pushdown-aware greedy search at upper levels (layers > 0)
    // Offloads remote-shard distance computations when batch threshold is met.
    template <typename ProxyT>
    int32_t greedy_search_upper_pushdown(const float *query, int32_t ep, int level, int target_level, float &ep_dist,
                                         std::vector<ProxyT> &proxies) const;

    std::vector<ShardView> shards_;

    // Per-shard local copy of vector data (heap-allocated)
    struct LocalVectorsCopy {
        std::vector<float> vectors;
    };
    std::vector<LocalVectorsCopy> local_vectors_; // only local shard populated
    bool vectors_localized_ = false;

    // Cached entry_point vector (for when EP is on a remote shard with vectors=nullptr)
    std::vector<float> ep_vector_cache_; // dim_ floats, populated by load_local_vectors

    // --- dot_norm optimization (experimental) ---
    // Pre-computed ||v||^2 for each local vector, indexed by local_id
    std::vector<float> vector_norms_;
    bool use_dot_norm_ = false;

public:
    // Enable dot_norm distance mode (must be called after load_local_vectors)
    // Computes vector norms on first enable.
    void enable_dot_norm(bool enable = true);
    bool dot_norm_enabled() const { return use_dot_norm_; }
    const float *vector_norms_data() const { return vector_norms_.data(); }

private:
    // Get pre-computed norm for a global_id on the local shard
    inline float get_norm_in_shard(int32_t global_id, uint32_t sorted_idx) const
    {
        uint64_t local_id = static_cast<uint64_t>(global_id) - shards_[sorted_idx].global_id_begin;
        return vector_norms_[local_id];
    }
};

} // namespace gd_hnsw
