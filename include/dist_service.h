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

#include "gd_layout.h"
#include "gd_hnsw_search.h"
#include "distance_kernel.h"
#include "distance_kernel_dot_norm.h"
#include "ls64_copy.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace gd_hnsw {

inline void cpu_pause() { asm volatile("yield" ::: "memory"); }

// Cross-process gd requires lock-free atomics (lock-based would use process-private mutex)
static_assert(std::atomic<uint64_t>::is_always_lock_free, "atomic<uint64_t> must be lock-free for cross-process gd");

// ============================================================
// Constants
// ============================================================

static constexpr uint32_t MAX_SPIN_ITERS_DEFAULT = 100000;
static constexpr uint32_t SERVICE_SPIN_YIELD = 1024;

// Service-side vector prefetch tuning constants
// PREFETCH_INIT_VECS: number of vectors to prefetch upfront (before compute loop)
// PREFETCH_DISTANCE: how far ahead to prefetch during compute loop (in vectors)
// PREFETCH_LINES: cache lines per vector to prefetch (64B each)
// PREFETCH_LOCALITY: 0=NTA/streaming, 3=keep in all cache levels
static constexpr uint32_t SERVICE_PREFETCH_INIT_VECS = 8;
static constexpr uint32_t SERVICE_PREFETCH_DISTANCE = 8;
static constexpr uint32_t SERVICE_PREFETCH_LINES = 2;
static constexpr uint32_t SERVICE_PREFETCH_LOCALITY = 3;

// Channel identity roles (dual-channel mode)
static constexpr uint8_t CHANNEL_ROLE_TASK = 0;
static constexpr uint8_t CHANNEL_ROLE_RESULT = 1;
static constexpr uint8_t IDENTITY_VALID_MAGIC = 0xAA;

// ============================================================
// Doorbell encoding: pack count into req_seq to merge two noncache
// stores into one atomic write.
//   High 16 bits = neighbor count (0 when idle)
//   Low  48 bits = sequence number (odd=request, even=idle)
// 48-bit seq space: ~4.5 years at 1M submits/sec before wrap.
// ============================================================

static constexpr uint64_t DOORBELL_SEQ_MASK = 0x0000FFFFFFFFFFFF;

static inline uint64_t doorbell_encode(uint64_t seq, uint16_t count)
{
    return (static_cast<uint64_t>(count) << 48) | (seq & DOORBELL_SEQ_MASK);
}
static inline uint64_t doorbell_seq(uint64_t doorbell) { return doorbell & DOORBELL_SEQ_MASK; }
static inline uint16_t doorbell_count(uint64_t doorbell) { return static_cast<uint16_t>(doorbell >> 48); }

// ============================================================
// DistChannelHeader — control fields in shared memory
// Split into 3 cachelines by writer identity to avoid false sharing:
//   Line 0 (search-writer): req_seq (doorbell-encoded: high16=count, low48=seq),
//          query_epoch.  count field is legacy (not written on hot path).
//   Line 1 (service-writer, hot): resp_seq  (search reads on every poll)
//   Line 2 (init-only): run_generation, max_batch, dim
//
// Protocol note: consume_batch() does NOT write idle (even) back to req_seq.
// After consume, req_seq stays at the last odd request value until the next
// submit_batch().  Service worker deduplicates via seq != last_seen_seq.
// Consequence: during idle periods, req_seq/count may hold stale request
// values — do not rely on even/odd to determine idle state externally.
// ============================================================

struct alignas(64) DistChannelHeader {
    // --- Cacheline 0: search thread writes ---
    // Doorbell: high16=count, low48=seq.  Odd seq = request submitted.
    // Note: after consume, req_seq remains at last odd value (no idle write-back).
    std::atomic<uint64_t> req_seq;
    uint32_t query_epoch; // bumped on new query, triggers cache refresh
    uint16_t count;       // legacy (not written on hot path; use doorbell_count)
    uint16_t reserved;

    // --- Cacheline 1: service thread writes (hot path) ---
    alignas(64) std::atomic<uint64_t> resp_seq; // service echoes decoded seq (low 48 bits of doorbell)

    // --- Cacheline 2: written once at init ---
    alignas(64) uint64_t run_generation; // process generation, prevents stale gd
    uint32_t max_batch;                  // = level0 max neighbors (2*M)
    uint32_t dim;                        // vector dimension
    // Identity metadata (dual-channel: owner writes at init, importer asserts)
    uint16_t src_rank;      // source search rank (or owner rank)
    uint16_t dst_shard;     // destination service shard
    uint16_t thread_id;     // search thread id within src_rank
    uint8_t channel_role;   // CHANNEL_ROLE_TASK / RESULT / UNIFIED
    uint8_t identity_valid; // IDENTITY_VALID_MAGIC when initialized
};

static_assert(offsetof(DistChannelHeader, req_seq) == 0, "req_seq must be at cacheline 0");
static_assert(offsetof(DistChannelHeader, resp_seq) == 64, "resp_seq must be at cacheline 1 (alone)");
static_assert(offsetof(DistChannelHeader, run_generation) == 128, "run_generation must be at cacheline 2");
static_assert(sizeof(DistChannelHeader) == 192, "DistChannelHeader must be exactly 3 cachelines (192 bytes)");

// ============================================================
// QueryCache — service-thread-private, avoids repeated cross-NUMA reads
// ============================================================

struct QueryCache {
    std::vector<float> q;
    uint32_t epoch = UINT32_MAX;
    float q_norm_sq = 0.0f; // cached ||q||^2 for dot_norm path
};

// ============================================================
// Dual-channel slot index — unified for task_inbox and result_inbox
// Both files use identical slot layout: (remote_rank_idx, thread_id)
// ============================================================

inline size_t dual_channel_slot(uint32_t remote_rank_idx, uint32_t thread_id, uint32_t srch_threads,
                                size_t bytes_per_slot)
{
    return (static_cast<size_t>(remote_rank_idx) * srch_threads + thread_id) * bytes_per_slot;
}

// ============================================================
// TaskChannelLayout — task channel: header + neighbor_ids + query_vec
// ============================================================

struct TaskChannelLayout {
    size_t header_off;
    size_t neighbor_ids_off;
    size_t query_vec_off;
    size_t bytes_total;

    static TaskChannelLayout build(uint32_t max_batch, uint32_t dim)
    {
        TaskChannelLayout l;
        l.header_off = 0;
        size_t cursor = sizeof(DistChannelHeader);

        cursor = (cursor + 63) & ~size_t(63);
        l.neighbor_ids_off = cursor;
        cursor += max_batch * sizeof(int32_t);

        cursor = (cursor + 63) & ~size_t(63);
        l.query_vec_off = cursor;
        cursor += dim * sizeof(float);

        l.bytes_total = (cursor + 63) & ~size_t(63);
        return l;
    }
};

// ============================================================
// ResultChannelLayout — result channel: header + distances
// ============================================================

struct ResultChannelLayout {
    size_t header_off;
    size_t distances_off;
    size_t bytes_total;

    static ResultChannelLayout build(uint32_t max_batch)
    {
        ResultChannelLayout l;
        l.header_off = 0;
        size_t cursor = sizeof(DistChannelHeader);

        cursor = (cursor + 63) & ~size_t(63);
        l.distances_off = cursor;
        cursor += max_batch * sizeof(float);

        l.bytes_total = (cursor + 63) & ~size_t(63);
        return l;
    }
};

// ============================================================
// TaskChannelView / ResultChannelView — typed pointers
// ============================================================

struct TaskChannelView {
    DistChannelHeader *h;
    int32_t *neighbor_ids;
    float *query_vec;

    static TaskChannelView from_raw(void *base, const TaskChannelLayout &layout)
    {
        auto *raw = static_cast<char *>(base);
        TaskChannelView v;
        v.h = reinterpret_cast<DistChannelHeader *>(raw + layout.header_off);
        v.neighbor_ids = reinterpret_cast<int32_t *>(raw + layout.neighbor_ids_off);
        v.query_vec = reinterpret_cast<float *>(raw + layout.query_vec_off);
        return v;
    }
};

struct ResultChannelView {
    DistChannelHeader *h;
    float *distances;

    static ResultChannelView from_raw(void *base, const ResultChannelLayout &layout)
    {
        auto *raw = static_cast<char *>(base);
        ResultChannelView v;
        v.h = reinterpret_cast<DistChannelHeader *>(raw + layout.header_off);
        v.distances = reinterpret_cast<float *>(raw + layout.distances_off);
        return v;
    }
};

// ============================================================
// ServiceChannelPair — paired task+result views for service worker
// ============================================================

struct ServiceChannelPair {
    TaskChannelView task;     // own task_inbox (export/cache — service reads)
    ResultChannelView result; // remote result_inbox (import/noncache — service writes)
};

// ============================================================
// DualDistProxy — search thread side, dual-channel mode
// task_ is in remote service rank's task_inbox (import/noncache — search writes)
// result_ is in own result_inbox (export/cache — search reads)
// ============================================================

class DualDistProxy {
public:
    DualDistProxy() = default;

    void init(TaskChannelView task, ResultChannelView result, uint32_t dim, uint32_t max_spin_iters, uint32_t max_batch,
              uint16_t src_rank = 0, uint16_t thread_id = 0)
    {
        task_ = task;
        result_ = result;
        dim_ = dim;
        max_batch_ = max_batch;
        max_spin_iters_ = max_spin_iters;
        seq_ = 0;
        query_epoch_ = 0;
        cached_query_ = nullptr;

        // Validate identity on both channels (owner wrote these before barrier)
        // Hard checks — must survive NDEBUG/Release builds
        if (task_.h->identity_valid != IDENTITY_VALID_MAGIC) {
            fprintf(stderr, "FATAL: task channel identity not initialized (got 0x%02x)\n", task_.h->identity_valid);
            abort();
        }
        if (task_.h->channel_role != CHANNEL_ROLE_TASK) {
            fprintf(stderr, "FATAL: task channel has wrong role (got %u, expected %u)\n", task_.h->channel_role,
                    CHANNEL_ROLE_TASK);
            abort();
        }
        if (result_.h->identity_valid != IDENTITY_VALID_MAGIC) {
            fprintf(stderr, "FATAL: result channel identity not initialized (got 0x%02x)\n", result_.h->identity_valid);
            abort();
        }
        if (result_.h->channel_role != CHANNEL_ROLE_RESULT) {
            fprintf(stderr, "FATAL: result channel has wrong role (got %u, expected %u)\n", result_.h->channel_role,
                    CHANNEL_ROLE_RESULT);
            abort();
        }

        // Importer writes src_rank/thread_id on task channel (owner left these for us)
        task_.h->src_rank = src_rank;
        task_.h->thread_id = thread_id;
        // Also set dst_shard/thread_id on result channel for diagnostics
        result_.h->thread_id = thread_id;

        // Dynamic fields: zero-init
        task_.h->req_seq.store(0, std::memory_order_relaxed);
        result_.h->resp_seq.store(0, std::memory_order_relaxed);
    }

    void set_query(const float *query) { cached_query_ = query; }

    void submit_batch(const int32_t *ids, uint16_t count)
    {
        if ((seq_ & 1) != 0) {
            fprintf(stderr, "FATAL: submit_batch called while request pending (seq=%llu)\n", (unsigned long long)seq_);
            abort();
        }
        if (count > max_batch_) {
            fprintf(stderr, "FATAL: submit_batch count overflow: count=%u max_batch=%u\n", static_cast<unsigned>(count),
                    static_cast<unsigned>(max_batch_));
            abort();
        }
        if (cached_query_) {
            query_epoch_++;
            ls64_copy_to_gd(task_.query_vec, cached_query_, dim_ * sizeof(float));
            task_.h->query_epoch = query_epoch_;
            cached_query_ = nullptr;
        }
        // Pre-compute doorbell (register ALU) so memcpy + release store
        // are back-to-back.  Pending check above uses old seq_ (even).
        const uint64_t next_seq = (seq_ + 1) & DOORBELL_SEQ_MASK;
        const uint64_t doorbell = doorbell_encode(next_seq, count);
        seq_ = next_seq;

        ls64_copy_to_gd(task_.neighbor_ids, ids, count * sizeof(int32_t));
        task_.h->req_seq.store(doorbell, std::memory_order_release);
    }

    bool try_wait_batch() const { return result_.h->resp_seq.load(std::memory_order_acquire) == seq_; }

    // Consume results after try_wait_batch() returns true.
    // Note: no idle write-back to task req_seq — the channel stays at the
    // last odd request seq until the next submit_batch().  Service worker
    // uses seq != last_seen_seq to deduplicate, so this is safe.
    const float *consume_batch()
    {
        if ((seq_ & 1) != 1) {
            fprintf(stderr, "FATAL: consume_batch called without pending request (seq=%llu)\n",
                    (unsigned long long)seq_);
            abort();
        }
        seq_ = (seq_ + 1) & DOORBELL_SEQ_MASK; // even = idle (local only)
        return result_.distances;
    }

    // Debug snapshot
    uint64_t debug_local_seq() const { return seq_; }
    uint64_t debug_req_seq() const { return doorbell_seq(task_.h->req_seq.load(std::memory_order_relaxed)); }
    uint64_t debug_resp_seq() const { return result_.h->resp_seq.load(std::memory_order_relaxed); }
    uint64_t debug_run_generation() const { return task_.h->run_generation; }
    uint32_t debug_query_epoch() const { return task_.h->query_epoch; }
    uint16_t debug_count() const { return doorbell_count(task_.h->req_seq.load(std::memory_order_relaxed)); }

private:
    TaskChannelView task_{};
    ResultChannelView result_{};
    uint32_t dim_ = 0;
    uint32_t max_batch_ = 0;
    uint64_t seq_ = 0;
    uint32_t query_epoch_ = 0;
    uint32_t max_spin_iters_ = MAX_SPIN_ITERS_DEFAULT;
    const float *cached_query_ = nullptr;
};

// ============================================================
// compute_distances_range — shared distance kernel for service worker
// Computes distances from q_ptr to ids[0..count) with prefetch + batch-4 + tail.
// ============================================================

inline void compute_distances_range(const float *q_ptr, const int32_t *ids, float *dists, uint32_t count, uint32_t dim,
                                    const ShardView &shard, bool use_dot_norm, float q_norm_sq,
                                    const float *vector_norms)
{
    const uint32_t pf_lines = std::min(SERVICE_PREFETCH_LINES, (dim * static_cast<uint32_t>(sizeof(float)) + 63) / 64);

    // Early prefetch
    const uint32_t init_pf = std::min(count, SERVICE_PREFETCH_INIT_VECS);
    for (uint32_t p = 0; p < init_pf; p++) {
        uint64_t lid = static_cast<uint64_t>(ids[p]) - shard.global_id_begin;
        if (lid < shard.ntotal_local) {
            const char *base = reinterpret_cast<const char *>(shard.vectors + lid * dim);
            for (uint32_t line = 0; line < pf_lines; line++) {
                __builtin_prefetch(base + line * 64, 0, SERVICE_PREFETCH_LOCALITY);
            }
        }
    }

    uint32_t i = 0;
    for (; i + 4 <= count; i += 4) {
        // Lookahead prefetch
        const uint32_t pf_start = i + SERVICE_PREFETCH_DISTANCE;
        const uint32_t pf_end = std::min(pf_start + 4, count);
        for (uint32_t p = pf_start; p < pf_end; p++) {
            uint64_t lid = static_cast<uint64_t>(ids[p]) - shard.global_id_begin;
            if (lid < shard.ntotal_local) {
                const char *base = reinterpret_cast<const char *>(shard.vectors + lid * dim);
                for (uint32_t line = 0; line < pf_lines; line++) {
                    __builtin_prefetch(base + line * 64, 0, SERVICE_PREFETCH_LOCALITY);
                }
            }
        }
        const float *vecs[4];
        for (int bi = 0; bi < 4; bi++) {
            int32_t gid = ids[i + bi];
            uint64_t local_id = static_cast<uint64_t>(gid) - shard.global_id_begin;
            if (local_id >= shard.ntotal_local) {
                fprintf(stderr, "FATAL: compute_distances_range gid=%d out of shard [%lu, %lu)\n", gid,
                        (unsigned long)shard.global_id_begin,
                        (unsigned long)(shard.global_id_begin + shard.ntotal_local));
                abort();
            }
            vecs[bi] = shard.vectors + local_id * dim;
        }
        if (use_dot_norm) {
            dot_norm_dist_batch_4(q_ptr, vecs[0], vecs[1], vecs[2], vecs[3], dim, q_norm_sq,
                                  vector_norms[static_cast<uint64_t>(ids[i]) - shard.global_id_begin],
                                  vector_norms[static_cast<uint64_t>(ids[i + 1]) - shard.global_id_begin],
                                  vector_norms[static_cast<uint64_t>(ids[i + 2]) - shard.global_id_begin],
                                  vector_norms[static_cast<uint64_t>(ids[i + 3]) - shard.global_id_begin], dists[i],
                                  dists[i + 1], dists[i + 2], dists[i + 3]);
        } else {
            l2_sqr_batch_4(q_ptr, vecs[0], vecs[1], vecs[2], vecs[3], dim, dists[i], dists[i + 1], dists[i + 2],
                           dists[i + 3]);
        }
    }
    // Tail: batch-3 / batch-2 / single
    const uint32_t tail = count - i;
    if (tail == 3) {
        const float *tv[3];
        for (int ti = 0; ti < 3; ti++) {
            int32_t gid = ids[i + ti];
            uint64_t local_id = static_cast<uint64_t>(gid) - shard.global_id_begin;
            if (local_id >= shard.ntotal_local) {
                fprintf(stderr, "FATAL: compute_distances_range gid=%d out of shard [%lu, %lu)\n", gid,
                        (unsigned long)shard.global_id_begin,
                        (unsigned long)(shard.global_id_begin + shard.ntotal_local));
                abort();
            }
            tv[ti] = shard.vectors + local_id * dim;
        }
        if (use_dot_norm) {
            dot_norm_dist_batch_3(q_ptr, tv[0], tv[1], tv[2], dim, q_norm_sq,
                                  vector_norms[static_cast<uint64_t>(ids[i]) - shard.global_id_begin],
                                  vector_norms[static_cast<uint64_t>(ids[i + 1]) - shard.global_id_begin],
                                  vector_norms[static_cast<uint64_t>(ids[i + 2]) - shard.global_id_begin], dists[i],
                                  dists[i + 1], dists[i + 2]);
        } else {
            l2_sqr_batch_3(q_ptr, tv[0], tv[1], tv[2], dim, dists[i], dists[i + 1], dists[i + 2]);
        }
    } else if (tail == 2) {
        const float *tv[2];
        for (int ti = 0; ti < 2; ti++) {
            int32_t gid = ids[i + ti];
            uint64_t local_id = static_cast<uint64_t>(gid) - shard.global_id_begin;
            if (local_id >= shard.ntotal_local) {
                fprintf(stderr, "FATAL: compute_distances_range gid=%d out of shard [%lu, %lu)\n", gid,
                        (unsigned long)shard.global_id_begin,
                        (unsigned long)(shard.global_id_begin + shard.ntotal_local));
                abort();
            }
            tv[ti] = shard.vectors + local_id * dim;
        }
        if (use_dot_norm) {
            dot_norm_dist_batch_2(q_ptr, tv[0], tv[1], dim, q_norm_sq,
                                  vector_norms[static_cast<uint64_t>(ids[i]) - shard.global_id_begin],
                                  vector_norms[static_cast<uint64_t>(ids[i + 1]) - shard.global_id_begin], dists[i],
                                  dists[i + 1]);
        } else {
            l2_sqr_batch_2(q_ptr, tv[0], tv[1], dim, dists[i], dists[i + 1]);
        }
    } else if (tail == 1) {
        int32_t gid = ids[i];
        uint64_t local_id = static_cast<uint64_t>(gid) - shard.global_id_begin;
        if (local_id >= shard.ntotal_local) {
            fprintf(stderr, "FATAL: compute_distances_range gid=%d out of shard [%lu, %lu)\n", gid,
                    (unsigned long)shard.global_id_begin, (unsigned long)(shard.global_id_begin + shard.ntotal_local));
            abort();
        }
        const float *vec = shard.vectors + local_id * dim;
        if (use_dot_norm) {
            dists[i] = dot_norm_dist(q_ptr, vec, dim, q_norm_sq, vector_norms[local_id]);
        } else {
            dists[i] = l2_sqr(q_ptr, vec, dim);
        }
    }
}

// ============================================================
// DualDistServiceWorker — service thread, dual-channel mode
// Reads from task channel (export/cache), writes to result channel (import/noncache)
// ============================================================

class DualDistServiceWorker {
public:
    DualDistServiceWorker(uint32_t my_shard, ShardView shard, uint32_t dim, uint64_t run_generation,
                          std::vector<ServiceChannelPair> pairs, std::atomic<bool> &stop)
        : my_shard_(my_shard), shard_(shard), dim_(dim), run_generation_(run_generation), pairs_(std::move(pairs)),
          stop_(stop)
    {
        query_caches_.resize(pairs_.size());
        for (auto &qc : query_caches_)
            qc.q.resize(dim);
        // Pre-allocate local buffers (avoids per-element noncache reads/stores)
        uint32_t max_b = pairs_.empty() ? 0 : pairs_[0].task.h->max_batch;
        for (size_t pi = 1; pi < pairs_.size(); pi++) {
            if (pairs_[pi].task.h->max_batch != max_b) {
                fprintf(stderr,
                        "FATAL: DualDistServiceWorker max_batch mismatch across channels: "
                        "%u vs %u (shard=%u)\n",
                        max_b, pairs_[pi].task.h->max_batch, my_shard_);
                abort();
            }
        }
        max_batch_ = max_b;
        local_dists_.resize(max_b);
        local_ids_.resize(max_b);
        // Absorb already-completed requests to prevent replay after restart.
        // consume_batch() no longer writes idle (even) back to req_seq, so
        // the channel may sit at a stale odd seq.  We must not re-process it.
        last_seen_seq_.resize(pairs_.size());
        for (size_t i = 0; i < pairs_.size(); i++) {
            uint64_t raw = pairs_[i].task.h->req_seq.load(std::memory_order_relaxed);
            uint64_t seq = doorbell_seq(raw);
            uint64_t resp = pairs_[i].result.h->resp_seq.load(std::memory_order_relaxed);
            if ((seq & 1) == 0 || resp == seq) {
                last_seen_seq_[i] = seq;
            } else {
                last_seen_seq_[i] = 0;
            }
        }
    }

    void enable_dot_norm(const float *norms)
    {
        use_dot_norm_ = (norms != nullptr);
        vector_norms_ = norms;
    }

    void run()
    {
        uint32_t spin_count = 0;
        const size_t n_pairs = pairs_.size();
        if (n_pairs == 0)
            return; // no channels assigned to this worker
        while (!stop_.load(std::memory_order_relaxed)) {
            bool did_work = false;
            for (size_t iter = 0; iter < n_pairs; iter++) {
                size_t i = (scan_cursor_ + iter) % n_pairs;
                auto &pair = pairs_[i];
                uint64_t raw = pair.task.h->req_seq.load(std::memory_order_acquire);
                uint64_t seq = doorbell_seq(raw);

                if ((seq & 1) == 0)
                    continue;
                if (seq == last_seen_seq_[i])
                    continue; // already processed (wrap-safe)
                if (pair.task.h->run_generation != run_generation_)
                    continue;

                // Identity validation on first request per channel.
                // Deferred to here because remote rank may not have written
                // identity at construct time.
                if (last_seen_seq_[i] == 0) {
                    if (pair.task.h->identity_valid != IDENTITY_VALID_MAGIC) {
                        fprintf(stderr, "FATAL: service: task channel[%zu] identity not initialized (got 0x%02x)\n", i,
                                pair.task.h->identity_valid);
                        abort();
                    }
                    if (pair.task.h->channel_role != CHANNEL_ROLE_TASK) {
                        fprintf(stderr, "FATAL: service: task channel[%zu] has wrong role (got %u)\n", i,
                                pair.task.h->channel_role);
                        abort();
                    }
                    if (pair.result.h->identity_valid != IDENTITY_VALID_MAGIC) {
                        fprintf(stderr, "FATAL: service: result channel[%zu] identity not initialized (got 0x%02x)\n",
                                i, pair.result.h->identity_valid);
                        abort();
                    }
                    if (pair.result.h->channel_role != CHANNEL_ROLE_RESULT) {
                        fprintf(stderr, "FATAL: service: result channel[%zu] has wrong role (got %u)\n", i,
                                pair.result.h->channel_role);
                        abort();
                    }
                }

                uint16_t count = doorbell_count(raw);
                process_request(pair, i, seq, count);
                last_seen_seq_[i] = seq;
                did_work = true;
            }
            scan_cursor_ = (scan_cursor_ + 1) % n_pairs;
            if (did_work) {
                spin_count = 0;
            } else {
                ++spin_count;
                cpu_pause();
            }
        }
    }

private:
    void process_request(ServiceChannelPair &pair, size_t ch_idx, uint64_t seq, uint16_t count)
    {
        auto &cache = query_caches_[ch_idx];

        if (pair.task.h->dim != dim_) {
            fprintf(stderr, "FATAL: DualDistService dim mismatch: ch_dim=%u worker_dim=%u\n", pair.task.h->dim, dim_);
            abort();
        }
        if (count > max_batch_) {
            fprintf(stderr,
                    "FATAL: DualDistService count overflow: count=%u max_batch=%u "
                    "(shard=%u ch=%zu req_seq=%llu)\n",
                    static_cast<unsigned>(count), max_batch_, my_shard_, ch_idx, static_cast<unsigned long long>(seq));
            abort();
        }

        // Bulk-copy neighbor_ids to local buffer first (avoids repeated cross-NUMA reads)
        const uint32_t count32d = count;
        std::memcpy(local_ids_.data(), pair.task.neighbor_ids, count * sizeof(int32_t));

        // Refresh query cache on epoch change
        if (pair.task.h->query_epoch != cache.epoch) {
            std::memcpy(cache.q.data(), pair.task.query_vec, dim_ * sizeof(float));
            cache.epoch = pair.task.h->query_epoch;
            if (use_dot_norm_) {
                cache.q_norm_sq = vec_norm_sq(cache.q.data(), dim_);
            }
        }

        // Distance computation
        const float *q_ptr = cache.q.data();
        compute_distances_range(q_ptr, local_ids_.data(), local_dists_.data(), count32d, dim_, shard_, use_dot_norm_,
                                cache.q_norm_sq, vector_norms_);

        // Bulk write-back + release store back-to-back
        ls64_copy_to_gd(pair.result.distances, local_dists_.data(), count * sizeof(float));
        pair.result.h->resp_seq.store(seq, std::memory_order_release);
    }

    uint32_t my_shard_;
    ShardView shard_;
    uint32_t dim_;
    uint64_t run_generation_;
    std::vector<ServiceChannelPair> pairs_;
    std::atomic<bool> &stop_;
    std::vector<QueryCache> query_caches_;
    std::vector<uint64_t> last_seen_seq_;
    size_t scan_cursor_ = 0;         // round-robin start for fair channel scheduling
    std::vector<float> local_dists_; // local buffer to batch cross-NUMA write-back
    std::vector<int32_t> local_ids_; // local buffer to avoid per-element noncache reads
    uint32_t max_batch_;             // cached from gd header (immutable after init)
    bool use_dot_norm_ = false;
    const float *vector_norms_ = nullptr; // pointer to pre-computed norms (owned by searcher)
};

// ============================================================
// DualDistService — manages service threads for dual-channel mode
// ============================================================

class DualDistService {
public:
    DualDistService(uint32_t my_shard, const ShardView &shard, uint32_t dim, uint64_t run_generation,
                    uint32_t service_threads, std::vector<ServiceChannelPair> all_pairs, std::vector<int> core_ids = {})
        : stop_(false)
    {
        if (service_threads == 0) {
            return;
        }
        std::vector<std::vector<ServiceChannelPair>> per_thread(service_threads);
        for (size_t i = 0; i < all_pairs.size(); i++) {
            per_thread[i % service_threads].push_back(all_pairs[i]);
        }

        for (uint32_t t = 0; t < service_threads; t++) {
            workers_.emplace_back(my_shard, shard, dim, run_generation, std::move(per_thread[t]), stop_);
        }

        for (uint32_t t = 0; t < service_threads; t++) {
            auto &w = workers_[t];
            int pin_core = (!core_ids.empty()) ? core_ids[t % core_ids.size()] : -1;
            threads_.emplace_back([&w, t, pin_core]() {
#ifdef __linux__
                char name[16];
                snprintf(name, sizeof(name), "dual_svc_%u", t);
                pthread_setname_np(pthread_self(), name);
                if (pin_core >= 0) {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(pin_core, &cpuset);
                    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
                        fprintf(stderr, "WARN: dual_svc_%u failed to pin to core %d\n", t, pin_core);
                    }
                }
#endif
                try {
                    w.run();
                } catch (const std::exception &e) {
                    fprintf(stderr, "FATAL: dual service worker exception: %s\n", e.what());
                    abort();
                } catch (...) {
                    fprintf(stderr, "FATAL: dual service worker unknown exception\n");
                    abort();
                }
            });
        }
    }

    ~DualDistService() { shutdown(); }

    void shutdown()
    {
        if (stop_.exchange(true))
            return;
        for (auto &t : threads_) {
            if (t.joinable())
                t.join();
        }
    }

    // Enable dot_norm distance kernel for all workers.
    // Must be called BEFORE threads start processing (i.e., before any queries arrive).
    // norms: pointer to pre-computed ||v||^2 array indexed by local_id.
    void enable_dot_norm(const float *norms)
    {
        for (auto &w : workers_) {
            w.enable_dot_norm(norms);
        }
    }

private:
    std::atomic<bool> stop_;
    std::vector<DualDistServiceWorker> workers_;
    std::vector<std::thread> threads_;
};

// ============================================================
// Pushdown config — passed around during setup
// ============================================================

struct PushdownConfig {
    uint32_t search_threads = 0;  // 0 = auto
    uint32_t service_threads = 0; // 0 = auto
    uint32_t max_spin_iters = MAX_SPIN_ITERS_DEFAULT;
};

} // namespace gd_hnsw
