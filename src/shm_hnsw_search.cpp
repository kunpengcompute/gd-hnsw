#include "shm_hnsw_search.h"
#include "distance_kernel.h"
#include "dist_service.h"
#include "ls64_copy.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <omp.h>

namespace shm_hnsw {

// ============================================================
// consume_dists_neon — bulk-load distances from (possibly noncache) memory
// Uses NEON 128-bit loads to reduce bus transactions, then processes
// directly from registers via lane extraction.
// ============================================================
template <typename InsertFn>
static inline void consume_dists_neon(const int32_t* ids, const float* dists,
                                      size_t n, InsertFn&& insert) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(dists + i);
        insert(ids[i],     vgetq_lane_f32(d, 0));
        insert(ids[i + 1], vgetq_lane_f32(d, 1));
        insert(ids[i + 2], vgetq_lane_f32(d, 2));
        insert(ids[i + 3], vgetq_lane_f32(d, 3));
    }
    // Scalar tail — safe, no over-read
    for (; i < n; i++) {
        insert(ids[i], dists[i]);
    }
}

// ============================================================
// VisitedTable
// ============================================================

VisitedTable::VisitedTable(uint64_t ntotal)
    : ntotal_(ntotal),
      use_sparse_(ntotal > VISITED_SPARSE_THRESHOLD) {
    if (!use_sparse_) {
        visited_.resize(ntotal, 0);
    } else {
        flat_table_.resize(FLAT_CAPACITY, 0);
        flat_mask_ = FLAT_CAPACITY - 1;
        flat_stamp_ = 0;
    }
}

void VisitedTable::advance() {
    if (use_sparse_) {
        flat_stamp_++;
        if (flat_stamp_ == 0) {
            // uint32 overflow (~4 billion queries) — full reset
            std::fill(flat_table_.begin(), flat_table_.end(), uint64_t(0));
            flat_stamp_ = 1;
        }
    } else {
        generation_++;
        if (generation_ == 0) {
            std::fill(visited_.begin(), visited_.end(), 0);
            generation_ = 1;
        }
    }
}

bool VisitedTable::get(int32_t id) const {
    if (use_sparse_) {
        uint32_t h = flat_hash(id);
        for (uint32_t i = 0; i < MAX_PROBE; i++) {
            uint64_t slot = flat_table_[(h + i) & flat_mask_];
            if (flat_stamp_of(slot) != flat_stamp_) return false;  // empty slot
            if (flat_key_of(slot) == id) return true;
        }
        return false;
    }
    return visited_[id] == generation_;
}

bool VisitedTable::set(int32_t id) {
    if (use_sparse_) {
        uint32_t h = flat_hash(id);
        for (uint32_t i = 0; i < MAX_PROBE; i++) {
            uint64_t& slot = flat_table_[(h + i) & flat_mask_];
            if (flat_stamp_of(slot) != flat_stamp_) {
                // empty slot — insert (single 8-byte store)
                slot = flat_pack(id, flat_stamp_);
                return true;
            }
            if (flat_key_of(slot) == id) {
                return false;  // already visited
            }
        }
        // Should never reach here at load factor < 0.5
        fprintf(stderr, "FATAL: FlatVisitedTable probe limit reached "
                "(id=%d, stamp=%u, capacity=%u, max_probe=%u)\n",
                id, flat_stamp_, FLAT_CAPACITY, MAX_PROBE);
        abort();
    }
    if (visited_[id] == generation_) return false;
    visited_[id] = generation_;
    return true;
}

// ============================================================
// ShmHnswSearcher — init helpers
// ============================================================

void ShmHnswSearcher::init_shard(const void* shm_base, uint64_t shm_size,
                                  bool skip_validation) {
    if (!skip_validation) {
        auto vr = validate_superblock(shm_base, shm_size);
        if (!vr.ok) {
            throw std::runtime_error("shm validation failed: " + vr.error);
        }
    }

    const auto* sb_ptr = static_cast<const SuperBlockV1*>(shm_base);
    if (sb_ptr->state != static_cast<uint32_t>(ShmState::ACTIVE)) {
        throw std::runtime_error("shm is not in ACTIVE state");
    }

    ShardView sv;
    sv.sb = sb_ptr;
    sv.vectors      = vectors_ptr(shm_base, *sb_ptr);
    sv.levels       = levels_ptr(shm_base, *sb_ptr);
    sv.offsets      = offsets_ptr(shm_base, *sb_ptr);
    sv.neighbors    = neighbors_ptr(shm_base, *sb_ptr);
    sv.cum_nneighbor = cum_nneighbor_ptr(shm_base, *sb_ptr);

    if (sb_ptr->num_shards > 0) {
        sv.global_id_begin = sb_ptr->global_id_begin;
        sv.ntotal_local = sb_ptr->ntotal_local;
    } else {
        sv.global_id_begin = 0;
        sv.ntotal_local = sb_ptr->ntotal;
    }

    shards_.push_back(sv);
}

// Single-shard constructor (backward compatible)
ShmHnswSearcher::ShmHnswSearcher(const void* shm_base, uint64_t shm_size,
                                  bool skip_validation) {
    init_shard(shm_base, shm_size, skip_validation);

    const auto* sb = shards_[0].sb;
    ntotal_ = sb->ntotal;
    dim_ = sb->dim;
    ef_search_ = sb->ef_search;
    max_level_ = sb->max_level;
    entry_point_ = sb->entry_point;
    num_shards_ = 1;
    ntotal_per_shard_ = ntotal_;
    uniform_shards_ = true;
    sorted_to_shard_id_ = {0};
    shard_id_to_sorted_idx_ = {0};
}

// Multi-shard constructor
ShmHnswSearcher::ShmHnswSearcher(
        const std::vector<std::pair<const void*, uint64_t>>& shards,
        bool skip_validation) {
    if (shards.empty()) {
        throw std::runtime_error("no shards provided");
    }

    for (const auto& [base, size] : shards) {
        init_shard(base, size, skip_validation);
    }

    // Sort shards by global_id_begin
    std::sort(shards_.begin(), shards_.end(),
              [](const ShardView& a, const ShardView& b) {
                  return a.global_id_begin < b.global_id_begin;
              });

    const auto* sb0 = shards_[0].sb;
    ntotal_ = sb0->ntotal;
    dim_ = sb0->dim;
    ef_search_ = sb0->ef_search;
    max_level_ = sb0->max_level;
    entry_point_ = sb0->entry_point;
    num_shards_ = static_cast<uint32_t>(shards_.size());
    ntotal_per_shard_ = shards_[0].ntotal_local;

    // Build logical shard-id mapping (channel/proxy topology is keyed by shard_id,
    // not by sorted position in this vector).
    sorted_to_shard_id_.resize(num_shards_);
    shard_id_to_sorted_idx_.assign(num_shards_, UINT32_MAX);
    for (uint32_t sorted_idx = 0; sorted_idx < num_shards_; sorted_idx++) {
        uint32_t shard_id = shards_[sorted_idx].sb->shard_id;
        if (shard_id >= num_shards_) {
            throw std::runtime_error("invalid shard_id in superblock");
        }
        if (shard_id_to_sorted_idx_[shard_id] != UINT32_MAX) {
            throw std::runtime_error("duplicate shard_id in loaded shards");
        }
        sorted_to_shard_id_[sorted_idx] = shard_id;
        shard_id_to_sorted_idx_[shard_id] = sorted_idx;
    }
    for (uint32_t shard_id = 0; shard_id < num_shards_; shard_id++) {
        if (shard_id_to_sorted_idx_[shard_id] == UINT32_MAX) {
            throw std::runtime_error("missing shard_id in loaded shards");
        }
    }

    // Check if all shards are uniform size (enables O(1) lookup)
    uniform_shards_ = true;
    for (uint32_t i = 1; i < num_shards_; i++) {
        if (shards_[i].ntotal_local != ntotal_per_shard_) {
            uniform_shards_ = false;
            break;
        }
    }
}


// ============================================================
// load_local_vectors — load vectors directly to heap (no SHM)
// ============================================================

// Helper: parallel chunked memcpy for large float arrays.
static void parallel_memcpy_f32(float* dst, const float* src, uint64_t count) {
    constexpr uint64_t CHUNK_SIZE = 1024 * 1024;  // 1M elements = 4 MB
    uint64_t nchunks = (count + CHUNK_SIZE - 1) / CHUNK_SIZE;
    #pragma omp parallel for schedule(static)
    for (uint64_t c = 0; c < nchunks; c++) {
        uint64_t start = c * CHUNK_SIZE;
        uint64_t end = std::min(start + CHUNK_SIZE, count);
        std::memcpy(dst + start, src + start, (end - start) * sizeof(float));
    }
}

// ============================================================
// load_local_vectors — load vectors directly to heap (no SHM)
// ============================================================

void ShmHnswSearcher::load_local_vectors(const float* vectors_data) {
    if (vectors_localized_) return;  // idempotent

    auto t0 = std::chrono::steady_clock::now();
    const size_t nshard = shards_.size();
    local_vectors_.resize(nshard);

    uint32_t sorted_idx = shard_id_to_sorted_idx_[local_shard_id_];
    const auto& sv = shards_[sorted_idx];
    auto& lv = local_vectors_[sorted_idx];

    uint64_t vec_count = sv.ntotal_local * dim_;
    lv.vectors.resize(vec_count);

    // Parallel chunked copy from provided data
    constexpr uint64_t CHUNK_THRESHOLD = 4ULL * 1024 * 1024;
    if (vec_count > CHUNK_THRESHOLD) {
        parallel_memcpy_f32(lv.vectors.data(), vectors_data, vec_count);
    } else {
        std::memcpy(lv.vectors.data(), vectors_data, vec_count * sizeof(float));
    }

    // Redirect local shard's ShardView.vectors pointer to heap
    shards_[sorted_idx].vectors = lv.vectors.data();

    // Set remote shard vectors to nullptr (pushdown mode doesn't need them)
    for (size_t i = 0; i < nshard; i++) {
        if (i != sorted_idx) {
            shards_[i].vectors = nullptr;
        }
    }

    // Cache entry_point vector if it's on the local shard
    if (entry_point_ >= 0) {
        uint64_t ep_gid = static_cast<uint64_t>(entry_point_);
        if (ep_gid >= sv.global_id_begin &&
            ep_gid < sv.global_id_begin + sv.ntotal_local) {
            // EP is local — cache from our heap copy
            uint64_t ep_local = ep_gid - sv.global_id_begin;
            ep_vector_cache_.assign(
                lv.vectors.data() + ep_local * dim_,
                lv.vectors.data() + (ep_local + 1) * dim_);
        }
        // If EP is remote, caller must provide EP vector separately
        // (via the file-based overload which reads it from the EP's shard file)
    }

    vectors_localized_ = true;
    uint64_t total_bytes = vec_count * sizeof(float);

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("[load_local_vectors] local shard %u, %.1f MB loaded to heap, %.2f sec\n",
           local_shard_id_, total_bytes / (1024.0 * 1024.0), sec);
}

void ShmHnswSearcher::load_local_vectors(const std::string& file_path) {
    if (vectors_localized_) return;  // idempotent

    auto t0 = std::chrono::steady_clock::now();

    // Read SuperBlock to find vectors offset/size
    FILE* fp = fopen(file_path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error(
            std::string("load_local_vectors: cannot open '") + file_path + "': " + strerror(errno));
    }

    SuperBlockV1 sb{};
    if (fread(&sb, sizeof(SuperBlockV1), 1, fp) != 1) {
        fclose(fp);
        throw std::runtime_error("load_local_vectors: cannot read SuperBlock");
    }

    uint64_t vec_offset = sb.blocks[BLK_VECTORS].off;
    uint64_t vec_bytes = sb.blocks[BLK_VECTORS].bytes;

    if (vec_bytes == 0) {
        fclose(fp);
        throw std::runtime_error("load_local_vectors: BLK_VECTORS is empty in file");
    }

    // Allocate heap and pread vectors directly
    const size_t nshard = shards_.size();
    local_vectors_.resize(nshard);

    uint32_t sorted_idx = shard_id_to_sorted_idx_[local_shard_id_];
    auto& lv = local_vectors_[sorted_idx];
    uint64_t vec_count = shards_[sorted_idx].ntotal_local * dim_;
    lv.vectors.resize(vec_count);

    // Seek and read vectors block
    if (fseek(fp, static_cast<long>(vec_offset), SEEK_SET) != 0) {
        fclose(fp);
        throw std::runtime_error("load_local_vectors: fseek failed");
    }
    size_t nread = fread(lv.vectors.data(), sizeof(float), vec_count, fp);
    if (nread != vec_count) {
        fclose(fp);
        throw std::runtime_error("load_local_vectors: short read on vectors");
    }

    // Cache entry_point vector if EP is in this shard file
    const auto& sv = shards_[sorted_idx];
    if (entry_point_ >= 0) {
        uint64_t ep_gid = static_cast<uint64_t>(entry_point_);
        if (ep_gid >= sv.global_id_begin &&
            ep_gid < sv.global_id_begin + sv.ntotal_local) {
            uint64_t ep_local = ep_gid - sv.global_id_begin;
            ep_vector_cache_.assign(
                lv.vectors.data() + ep_local * dim_,
                lv.vectors.data() + (ep_local + 1) * dim_);
        }
    }

    fclose(fp);

    // Redirect local shard's ShardView.vectors pointer to heap
    shards_[sorted_idx].vectors = lv.vectors.data();

    // Set remote shard vectors to nullptr (pushdown mode doesn't need them)
    for (size_t i = 0; i < nshard; i++) {
        if (i != sorted_idx) {
            shards_[i].vectors = nullptr;
        }
    }

    vectors_localized_ = true;
    uint64_t total_bytes = vec_count * sizeof(float);

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("[load_local_vectors] local shard %u from file, %.1f MB loaded to heap, %.2f sec\n",
           local_shard_id_, total_bytes / (1024.0 * 1024.0), sec);
}

// ============================================================
// enable_dot_norm — compute norms lazily on first enable
// ============================================================

void ShmHnswSearcher::enable_dot_norm(bool enable) {
    use_dot_norm_ = enable;
    if (!enable || !vector_norms_.empty()) return;  // already computed or disabling

    if (!vectors_localized_) {
        fprintf(stderr, "WARNING: enable_dot_norm called before load_local_vectors, norms not computed\n");
        use_dot_norm_ = false;
        return;
    }

    uint32_t sorted_idx = shard_id_to_sorted_idx_[local_shard_id_];
    uint64_t nvecs = shards_[sorted_idx].ntotal_local;
    const float* vecs = shards_[sorted_idx].vectors;

    auto t0 = std::chrono::steady_clock::now();
    vector_norms_.resize(nvecs);
    #pragma omp parallel for schedule(static)
    for (uint64_t vid = 0; vid < nvecs; vid++) {
        vector_norms_[vid] = vec_norm_sq(vecs + vid * dim_, dim_);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[dot_norm] Pre-computed %lu vector norms (%.1f ms)\n",
           static_cast<unsigned long>(nvecs), ms);
}

// ============================================================
// Shard-aware accessors
// ============================================================

uint32_t ShmHnswSearcher::find_shard(int32_t global_id) const {
    uint64_t gid = static_cast<uint64_t>(global_id);
    if (uniform_shards_) {
        // O(1) fast path: shift when power-of-2, otherwise division
        uint32_t sorted_idx;
        if (shard_shift_valid_) {
            sorted_idx = static_cast<uint32_t>(gid >> shard_shift_);
        } else {
            sorted_idx = static_cast<uint32_t>(gid / ntotal_per_shard_);
        }
        if (sorted_idx >= num_shards_) sorted_idx = num_shards_ - 1;
        return sorted_to_shard_id_[sorted_idx];
    }
    // Binary search fallback for non-uniform shards
    uint32_t lo = 0, hi = num_shards_;
    while (lo + 1 < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (shards_[mid].global_id_begin <= gid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return sorted_to_shard_id_[lo];
}

const float* ShmHnswSearcher::get_vector(int32_t global_id) const {
    // Entry point may be on a remote shard with vectors=nullptr; use cache.
    if (global_id == entry_point_ && !ep_vector_cache_.empty()) {
        return ep_vector_cache_.data();
    }
    if (num_shards_ == 1) {
        return shards_[0].vectors + static_cast<uint64_t>(global_id) * dim_;
    }
    uint32_t shard_id = find_shard(global_id);
    uint32_t sorted_idx = shard_id_to_sorted_idx_[shard_id];
    uint64_t local_id = static_cast<uint64_t>(global_id) - shards_[sorted_idx].global_id_begin;
    return shards_[sorted_idx].vectors + local_id * dim_;
}

void ShmHnswSearcher::get_neighbor_range(
        int32_t global_id, int layer,
        uint64_t& begin_out, uint64_t& end_out,
        const int32_t*& neighbors_out) const {
    uint32_t sorted_idx = 0;
    uint64_t local_id = static_cast<uint64_t>(global_id);

    if (num_shards_ > 1) {
        uint32_t shard_id = find_shard(global_id);
        sorted_idx = shard_id_to_sorted_idx_[shard_id];
        local_id = static_cast<uint64_t>(global_id) - shards_[sorted_idx].global_id_begin;
    }

    const auto& sv = shards_[sorted_idx];
    uint64_t o = sv.offsets[local_id];
    begin_out = o + static_cast<uint64_t>(sv.cum_nneighbor[layer]);
    end_out   = o + static_cast<uint64_t>(sv.cum_nneighbor[layer + 1]);
    neighbors_out = sv.neighbors;
}

const ShardView& ShmHnswSearcher::shard_by_id(uint32_t shard_id) const {
    if (shard_id >= shard_id_to_sorted_idx_.size()) {
        throw std::runtime_error("shard_id out of range");
    }
    uint32_t sorted_idx = shard_id_to_sorted_idx_[shard_id];
    if (sorted_idx == UINT32_MAX) {
        throw std::runtime_error("shard_id mapping missing");
    }
    return shards_[sorted_idx];
}

// ============================================================
// greedy_search_upper — local upper-layer greedy descent
// All distances computed locally by reading vectors from SHM.
// ============================================================

int32_t ShmHnswSearcher::greedy_search_upper_local(
        const float* query, int32_t ep,
        int level, int target_level,
        float& ep_dist, SearchProfile* prof) const {
    using Clock = std::chrono::steady_clock;

    for (int l = level; l > target_level; l--) {
        bool changed = true;
        while (changed) {
            changed = false;

            uint64_t begin, end;
            const int32_t* nb;
            get_neighbor_range(ep, l, begin, end, nb);

            for (uint64_t j = begin; j < end; j++) {
                int32_t nid = nb[j];
                if (nid == EMPTY_NEIGHBOR) break;

                if (j + 1 < end) {
                    int32_t next = nb[j + 1];
                    if (next != EMPTY_NEIGHBOR) {
                        prefetch_vector(next);
                    }
                }

                float d;
                if (prof) {
                    uint32_t s = (num_shards_ > 1) ? find_shard(nid) : 0;
                    bool is_local = (s == local_shard_id_);
                    auto t0 = Clock::now();
                    d = l2_sqr(query, get_vector(nid), dim_);
                    auto t1 = Clock::now();
                    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                    if (is_local) { prof->dist_calls_local++; prof->dist_ns_local += ns; }
                    else          { prof->dist_calls_remote++; prof->dist_ns_remote += ns; }
                } else {
                    d = l2_sqr(query, get_vector(nid), dim_);
                }

                if (d < ep_dist) {
                    ep_dist = d;
                    ep = nid;
                    changed = true;
                }
            }
        }
    }
    return ep;
}

// ============================================================
// search_at_layer — local beam search at layer 0
// All distances computed locally with batch-4 NEON kernel.
// ============================================================

void ShmHnswSearcher::search_at_layer_local(
        const float* query, int32_t ep, float ep_dist,
        int level, int ef_search,
        std::vector<std::pair<float, int32_t>>& results,
        VisitedTable& vt) const {

    thread_local MinimaxHeap heap(0);
    if (heap.n != ef_search) heap = MinimaxHeap(ef_search);
    heap.clear();

    heap.push(ep, ep_dist);
    vt.set(ep);

    float lower_bound = (heap.size() >= ef_search) ? heap.max()
                                                     : std::numeric_limits<float>::max();

    while (heap.candidates() > 0) {
        auto [cd, cid] = heap.pop_min_candidate();
        if (cid < 0) break;

        if (cd > lower_bound) break;

        uint64_t begin, end;
        const int32_t* nb;
        get_neighbor_range(cid, level, begin, end, nb);

        // Batch-4 distance computation (FAISS pattern)
        int32_t buf_ids[4];
        const float* buf_vecs[4];
        int buf_cnt = 0;

        for (uint64_t j = begin; j < end; j++) {
            int32_t nid = nb[j];
            if (nid == EMPTY_NEIGHBOR) break;

            if (j + 1 < end) {
                int32_t next = nb[j + 1];
                if (next != EMPTY_NEIGHBOR) {
                    vt.prefetch(next);
                    prefetch_vector(next);
                }
            }

            if (!vt.set(nid)) continue;

            buf_ids[buf_cnt] = nid;
            buf_vecs[buf_cnt] = get_vector(nid);
            buf_cnt++;

            if (buf_cnt == 4) {
                float d0, d1, d2, d3;
                l2_sqr_batch_4(query, buf_vecs[0], buf_vecs[1],
                               buf_vecs[2], buf_vecs[3], dim_,
                               d0, d1, d2, d3);
                float dists[4] = {d0, d1, d2, d3};
                for (int bi = 0; bi < 4; bi++) {
                    if (dists[bi] < lower_bound || heap.size() < ef_search) {
                        heap.push(buf_ids[bi], dists[bi]);
                        if (heap.size() >= ef_search) {
                            lower_bound = heap.max();
                        }
                    }
                }
                buf_cnt = 0;
            }
        }
        // Tail: remaining < 4 neighbors
        for (int bi = 0; bi < buf_cnt; bi++) {
            float d = l2_sqr(query, buf_vecs[bi], dim_);
            if (d < lower_bound || heap.size() < ef_search) {
                heap.push(buf_ids[bi], d);
                if (heap.size() >= ef_search) {
                    lower_bound = heap.max();
                }
            }
        }
    }

    heap.extract_sorted(results);
}

// ============================================================
// search_at_layer_profiled — local beam search with profiling
// ============================================================

void ShmHnswSearcher::search_at_layer_local_profiled(
        const float* query, int32_t ep, float ep_dist,
        int level, int ef_search,
        std::vector<std::pair<float, int32_t>>& results,
        VisitedTable& vt, SearchProfile& prof) const {
    using Clock = std::chrono::steady_clock;

    thread_local MinimaxHeap heap(0);
    if (heap.n != ef_search) heap = MinimaxHeap(ef_search);
    heap.clear();

    heap.push(ep, ep_dist);
    vt.set(ep);
    prof.heap_pushes++;

    float lower_bound = (heap.size() >= ef_search) ? heap.max()
                                                     : std::numeric_limits<float>::max();

    while (heap.candidates() > 0) {
        auto [cd, cid] = heap.pop_min_candidate();
        if (cid < 0) break;

        if (cd > lower_bound) break;
        prof.nodes_expanded++;

        uint64_t begin, end;
        const int32_t* nb;
        get_neighbor_range(cid, level, begin, end, nb);

        for (uint64_t j = begin; j < end; j++) {
            int32_t nid = nb[j];
            if (nid == EMPTY_NEIGHBOR) break;
            prof.edges_scanned++;

            if (j + 1 < end) {
                int32_t next = nb[j + 1];
                if (next != EMPTY_NEIGHBOR) {
                    vt.prefetch(next);
                    prefetch_vector(next);
                }
            }

            if (!vt.set(nid)) continue;

            uint32_t s = (num_shards_ > 1) ? find_shard(nid) : 0;
            bool is_local = (s == local_shard_id_);

            auto t0 = Clock::now();
            float d = l2_sqr(query, get_vector(nid), dim_);
            auto t1 = Clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();

            if (is_local) { prof.dist_calls_local++; prof.dist_ns_local += ns; }
            else          { prof.dist_calls_remote++; prof.dist_ns_remote += ns; }

            if (d < lower_bound || heap.size() < ef_search) {
                heap.push(nid, d);
                prof.heap_pushes++;
                if (heap.size() >= ef_search) {
                    lower_bound = heap.max();
                }
            } else {
                prof.heap_rejections++;
            }
        }
    }

    heap.extract_sorted(results);
}

// ============================================================
// Pushdown-aware greedy search at upper levels (layers > 0)
// Offloads remote-shard distance computations via proxies.
// Guard: if remote_total == 0, falls back to inline computation.
// ============================================================

template <typename ProxyT>
int32_t ShmHnswSearcher::greedy_search_upper_pushdown(
        const float* query, int32_t ep,
        int level, int target_level,
        float& ep_dist,
        std::vector<ProxyT>& proxies,
        SearchProfile* prof) const {
    using Clock = std::chrono::steady_clock;

    // Per-shard remote ID buffer (reused across iterations)
    thread_local std::vector<std::vector<int32_t>> upper_remote_ids;
    if (upper_remote_ids.size() < num_shards_) {
        upper_remote_ids.resize(num_shards_);
    }

    for (int l = level; l > target_level; l--) {
        bool changed = true;
        while (changed) {
            changed = false;

            uint64_t begin, end;
            const int32_t* nb;
            get_neighbor_range(ep, l, begin, end, nb);

            // Classify neighbors into local vs remote
            thread_local std::vector<int32_t> upper_local_ids;
            upper_local_ids.clear();
            for (uint32_t s = 0; s < num_shards_; s++) {
                upper_remote_ids[s].clear();
            }

            uint32_t remote_total = 0;
            for (uint64_t j = begin; j < end; j++) {
                int32_t nid = nb[j];
                if (prof) prof->edges_scanned++;
                if (nid == EMPTY_NEIGHBOR) break;

                uint64_t gid = static_cast<uint64_t>(nid);
                if (gid >= local_id_begin_ && gid < local_id_end_) {
                    upper_local_ids.push_back(nid);
                } else {
                    uint32_t s = find_shard(nid);
                    upper_remote_ids[s].push_back(nid);
                    remote_total++;
                }
            }

            // Guard: fallback to inline if not enough remote or no local overlap
            if (remote_total == 0) {
                for (auto nid : upper_local_ids) {
                    float d;
                    if (prof) {
                        auto t0 = Clock::now();
                        d = l2_sqr(query, get_vector(nid), dim_);
                        auto t1 = Clock::now();
                        prof->dist_calls_local++;
                        prof->dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
                    } else {
                        d = l2_sqr(query, get_vector(nid), dim_);
                    }
                    if (d < ep_dist) { ep_dist = d; ep = nid; changed = true; }
                }
                for (uint32_t s = 0; s < num_shards_; s++) {
                    for (auto nid : upper_remote_ids[s]) {
                        float d;
                        if (prof) {
                            auto t0 = Clock::now();
                            d = l2_sqr(query, get_vector(nid), dim_);
                            auto t1 = Clock::now();
                            prof->dist_calls_remote++;
                            prof->dist_ns_remote += std::chrono::duration<double, std::nano>(t1 - t0).count();
                        } else {
                            d = l2_sqr(query, get_vector(nid), dim_);
                        }
                        if (d < ep_dist) { ep_dist = d; ep = nid; changed = true; }
                    }
                }
            } else {
                // Phase 2: submit remote batches (largest batch first for early service start)
                uint32_t pending_mask = 0;
                // Build shard order sorted by batch size descending
                uint32_t shard_order[32];
                uint32_t shard_count = 0;
                for (uint32_t i = 1; i < num_shards_; i++) {
                    uint32_t s = (local_shard_id_ + i) % num_shards_;
                    if (upper_remote_ids[s].empty()) continue;
                    shard_order[shard_count++] = s;
                }
                std::sort(shard_order, shard_order + shard_count,
                          [&](uint32_t a, uint32_t b) {
                              return upper_remote_ids[a].size() > upper_remote_ids[b].size();
                          });
                for (uint32_t i = 0; i < shard_count; i++) {
                    uint32_t s = shard_order[i];
                    const size_t rs = upper_remote_ids[s].size();

                    if (rs > UINT16_MAX) {
                        fprintf(stderr, "FATAL: upper remote batch size %zu exceeds uint16_t max\n", rs);
                        abort();
                    }
                    proxies[s].set_query(query);
                    proxies[s].submit_batch(upper_remote_ids[s].data(),
                                           static_cast<uint16_t>(rs));
                    pending_mask |= (1u << s);
                    if (prof) prof->dist_calls_pushed += rs;
                }

                // Phase 3: compute local distances (overlaps with remote service)
                const uint32_t local_sorted_idx = shard_id_to_sorted_idx_[local_shard_id_];
                for (auto nid : upper_local_ids) {
                    float d;
                    if (prof) {
                        auto t0 = Clock::now();
                        d = l2_sqr(query, get_vector_in_shard(nid, local_sorted_idx), dim_);
                        auto t1 = Clock::now();
                        prof->dist_calls_local++;
                        prof->dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
                    } else {
                        d = l2_sqr(query, get_vector_in_shard(nid, local_sorted_idx), dim_);
                    }
                    if (d < ep_dist) { ep_dist = d; ep = nid; changed = true; }
                }

                // Phase 4: poll remote results (with timeout protection)
                static constexpr auto UPPER_MAX_POLL_WAIT = std::chrono::seconds(5);
                uint32_t spins = 0;
                auto poll_start = std::chrono::steady_clock::now();
                while (pending_mask) {
                    bool made_progress = false;
                    for (uint32_t tmp = pending_mask; tmp; ) {
                        uint32_t s = __builtin_ctz(tmp);
                        tmp &= tmp - 1;
                        if (!proxies[s].try_wait_batch()) continue;

                        const float* dists = proxies[s].consume_batch();
                        for (size_t i = 0; i < upper_remote_ids[s].size(); i++) {
                            float d = dists[i];
                            if (d < ep_dist) {
                                ep_dist = d;
                                ep = upper_remote_ids[s][i];
                                changed = true;
                            }
                        }
                        pending_mask &= ~(1u << s);
                        made_progress = true;
                    }
                    if (!made_progress) {
                        spins++;
                        if (spins < 64) {
                            cpu_pause();
                        } else if (spins < 1024) {
                            std::this_thread::yield();
                        } else if ((spins & 0xFFF) == 0) {
                            auto now = std::chrono::steady_clock::now();
                            if (now - poll_start > UPPER_MAX_POLL_WAIT) {
                                fprintf(stderr, "FATAL: upper pushdown poll timeout (5s), pending_mask=0x%x\n",
                                        pending_mask);
                                abort();
                            }
                        }
                    }
                }
            }
        }
    }
    return ep;
}

// Explicit template instantiations
template int32_t ShmHnswSearcher::greedy_search_upper_pushdown<DualDistProxy>(
        const float*, int32_t, int, int, float&, std::vector<DualDistProxy>&, SearchProfile*) const;

// ============================================================
// search_to — zero-allocation single query (optionally profiled)
// ============================================================

void ShmHnswSearcher::search_to(const float* query, const SearchParams& params,
                                 int32_t* out_ids, float* out_dists,
                                 SearchProfile* prof) const {
    using Clock = std::chrono::steady_clock;

    if (prof) *prof = SearchProfile{};

    int32_t k = params.k;
    int32_t ef = params.ef_search > 0 ? params.ef_search : static_cast<int32_t>(ef_search_);
    if (ef < k) ef = k;

    // Initialize output
    for (int32_t i = 0; i < k; i++) {
        out_ids[i] = -1;
        out_dists[i] = std::numeric_limits<float>::max();
    }

    if (ntotal_ == 0 || entry_point_ < 0) return;

    Clock::time_point t_total;
    if (prof) t_total = Clock::now();

    if (num_shards_ > 1 && !dual_pushdown_proxies_2d_) {
        fprintf(stderr, "FATAL: search_to requires pushdown proxies for multi-shard\n");
        abort();
    }

    thread_local std::unique_ptr<VisitedTable> tl_vt;
    thread_local uint64_t tl_vt_ntotal = 0;
    if (!tl_vt || tl_vt_ntotal != ntotal_) {
        tl_vt = std::make_unique<VisitedTable>(ntotal_);
        tl_vt_ntotal = ntotal_;
    }
    tl_vt->advance();

    thread_local std::vector<std::pair<float, int32_t>> tl_candidates;
    tl_candidates.clear();

    int32_t ep = entry_point_;
    float ep_dist;

    if (prof) {
        uint32_t s = (num_shards_ > 1) ? find_shard(ep) : 0;
        bool is_local = (s == local_shard_id_);
        auto t0 = Clock::now();
        ep_dist = l2_sqr(query, get_vector(ep), dim_);
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (is_local) { prof->dist_calls_local++; prof->dist_ns_local += ns; }
        else          { prof->dist_calls_remote++; prof->dist_ns_remote += ns; }
    } else {
        ep_dist = l2_sqr(query, get_vector(ep), dim_);
    }

    Clock::time_point t_upper, t_upper_end, t_layer0, t_layer0_end;
    if (prof) t_upper = Clock::now();

    if (num_shards_ <= 1) {
        ep = greedy_search_upper_local(query, ep, max_level_, 0, ep_dist, prof);
        if (prof) {
            t_upper_end = Clock::now();
            t_layer0 = Clock::now();
            search_at_layer_local_profiled(query, ep, ep_dist, 0, ef, tl_candidates, *tl_vt, *prof);
            t_layer0_end = Clock::now();
        } else {
            search_at_layer_local(query, ep, ep_dist, 0, ef, tl_candidates, *tl_vt);
        }
    } else {
        int tid = omp_get_thread_num();
        if (tid < 0 || static_cast<uint32_t>(tid) >= num_search_threads_) {
            fprintf(stderr, "FATAL: OMP thread %d out of range [0, %u)\n",
                    tid, num_search_threads_);
            abort();
        }
        ep = greedy_search_upper_pushdown(query, ep, max_level_, 0, ep_dist,
                                         (*dual_pushdown_proxies_2d_)[tid], prof);
        if (prof) {
            t_upper_end = Clock::now();
            t_layer0 = Clock::now();
            search_at_layer_pushdown_profiled(query, ep, ep_dist, 0, ef,
                                              tl_candidates, *tl_vt,
                                              (*dual_pushdown_proxies_2d_)[tid], *prof);
            t_layer0_end = Clock::now();
        } else {
            search_at_layer_pushdown(query, ep, ep_dist, 0, ef, tl_candidates, *tl_vt,
                                     (*dual_pushdown_proxies_2d_)[tid]);
        }
    }

    if (prof) {
        prof->upper_ns = std::chrono::duration<double, std::nano>(t_upper_end - t_upper).count();
        prof->layer0_ns = std::chrono::duration<double, std::nano>(t_layer0_end - t_layer0).count();
    }

    int32_t n = std::min(k, static_cast<int32_t>(tl_candidates.size()));
    for (int32_t i = 0; i < n; i++) {
        out_dists[i] = tl_candidates[i].first;
        out_ids[i]   = tl_candidates[i].second;
    }

    if (prof) {
        auto t_total_end = Clock::now();
        prof->total_ns = std::chrono::duration<double, std::nano>(t_total_end - t_total).count();
    }
}

// ============================================================
// search_batch_flat — zero-allocation batch
// ============================================================

void ShmHnswSearcher::search_batch_flat(
        const float* queries, int32_t nq,
        const SearchParams& params,
        int32_t* out_ids, float* out_dists) const {
    int32_t k = params.k;
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < nq; i++) {
        search_to(queries + static_cast<uint64_t>(i) * dim_, params,
                  out_ids + static_cast<uint64_t>(i) * k,
                  out_dists + static_cast<uint64_t>(i) * k);
    }
}


// ============================================================
// search_batch_profiled — batch with aggregated profile
// ============================================================

SearchProfileSummary ShmHnswSearcher::search_batch_profiled(
        const float* queries, int32_t nq,
        const SearchParams& params,
        int32_t* out_ids, float* out_dists) const {
    int32_t k = params.k;

    std::vector<SearchProfile> profiles(nq);

    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < nq; i++) {
        search_to(
            queries + static_cast<uint64_t>(i) * dim_, params,
            out_ids + static_cast<uint64_t>(i) * k,
            out_dists + static_cast<uint64_t>(i) * k,
            &profiles[i]);
    }

    // Aggregate
    SearchProfileSummary s;
    s.nq = nq;
    uint64_t total_dist = 0;
    uint64_t total_remote = 0;
    for (int32_t i = 0; i < nq; i++) {
        const auto& p = profiles[i];
        s.avg_dist_local += p.dist_calls_local;
        s.avg_dist_remote += p.dist_calls_remote;
        s.avg_dist_pushed += p.dist_calls_pushed;
        s.avg_nodes_expanded += p.nodes_expanded;
        s.avg_edges_scanned += p.edges_scanned;
        s.avg_heap_pushes += p.heap_pushes;
        s.avg_heap_rejections += p.heap_rejections;
        s.avg_dist_ns_local += p.dist_ns_local;
        s.avg_dist_ns_remote += p.dist_ns_remote;
        s.avg_dist_mem_ns_local += p.dist_mem_ns_local;
        s.avg_dist_compute_ns_local += p.dist_compute_ns_local;
        s.avg_dist_mem_ns_remote += p.dist_mem_ns_remote;
        s.avg_dist_compute_ns_remote += p.dist_compute_ns_remote;
        s.avg_dist_split_calls_local += p.dist_split_calls_local;
        s.avg_dist_split_calls_remote += p.dist_split_calls_remote;
        s.avg_upper_ns += p.upper_ns;
        s.avg_layer0_ns += p.layer0_ns;
        s.avg_total_ns += p.total_ns;
        total_dist += p.dist_calls_local + p.dist_calls_remote + p.dist_calls_pushed;
        total_remote += p.dist_calls_remote + p.dist_calls_pushed;
    }
    double inv = (nq > 0) ? (1.0 / nq) : 0.0;
    s.avg_dist_local *= inv;
    s.avg_dist_remote *= inv;
    s.avg_dist_pushed *= inv;
    s.avg_nodes_expanded *= inv;
    s.avg_edges_scanned *= inv;
    s.avg_heap_pushes *= inv;
    s.avg_heap_rejections *= inv;
    s.avg_dist_ns_local *= inv;
    s.avg_dist_ns_remote *= inv;
    s.avg_dist_mem_ns_local *= inv;
    s.avg_dist_compute_ns_local *= inv;
    s.avg_dist_mem_ns_remote *= inv;
    s.avg_dist_compute_ns_remote *= inv;
    s.avg_dist_split_calls_local *= inv;
    s.avg_dist_split_calls_remote *= inv;
    s.avg_upper_ns *= inv;
    s.avg_layer0_ns *= inv;
    s.avg_total_ns *= inv;
    s.pct_remote = (total_dist > 0) ? (100.0 * total_remote / total_dist) : 0.0;

    // Pushdown phase aggregation
    if (dual_pushdown_proxies_2d_ && num_shards_ > 1 && nq > 0) {
        s.pushdown_active = true;
        double max_poll = 0;
        for (int32_t i = 0; i < nq; i++) {
            const auto& pd = profiles[i].pushdown;
            s.avg_phase1_ns += pd.phase1_ns;
            s.avg_graph_access_ns += pd.graph_access_ns;
            s.avg_classify_loop_ns += pd.classify_loop_ns;
            s.avg_vt_check_ns += pd.vt_check_ns;
            s.avg_find_shard_ns += pd.find_shard_ns;
            s.avg_route_push_ns += pd.route_push_ns;
            s.avg_phase2_ns += pd.phase2_ns;
            s.avg_phase2_query_ns += pd.phase2_query_ns;
            s.avg_phase2_ids_ns += pd.phase2_ids_ns;
            s.avg_phase2_doorbell_ns += pd.phase2_doorbell_ns;
            s.avg_phase3_ns += pd.phase3_ns;
            s.avg_poll_wait_ns += pd.poll_wait_ns;
            s.avg_poll_spin_ns += pd.poll_spin_ns;
            s.avg_poll_consume_ns += pd.poll_consume_ns;
            s.avg_poll_overhead_ns += pd.poll_overhead_ns;
            s.avg_poll_spins_pause += pd.poll_spins_pause;
            s.avg_poll_spins_yield += pd.poll_spins_yield;
            s.avg_poll_spins_sleep += pd.poll_spins_sleep;
            s.avg_batches_pushed += pd.batches_pushed;
            s.avg_batches_fallback += pd.batches_fallback;
            s.avg_ids_pushed += pd.ids_pushed;
            s.avg_ids_fallback += pd.ids_fallback;
            s.avg_expansions_with_remote += pd.expansions_with_remote;
            s.avg_neighbors_local += pd.neighbors_local;
            s.avg_neighbors_remote += pd.neighbors_remote;
            s.avg_vt_dedup_hits += pd.vt_dedup_hits;
            if (pd.max_poll_wait_ns > max_poll) max_poll = pd.max_poll_wait_ns;
        }
        s.avg_phase1_ns *= inv;
        s.avg_graph_access_ns *= inv;
        s.avg_classify_loop_ns *= inv;
        s.avg_vt_check_ns *= inv;
        s.avg_find_shard_ns *= inv;
        s.avg_route_push_ns *= inv;
        s.avg_phase2_ns *= inv;
        s.avg_phase2_query_ns *= inv;
        s.avg_phase2_ids_ns *= inv;
        s.avg_phase2_doorbell_ns *= inv;
        s.avg_phase3_ns *= inv;
        s.avg_poll_wait_ns *= inv;
        s.avg_poll_spin_ns *= inv;
        s.avg_poll_consume_ns *= inv;
        s.avg_poll_overhead_ns *= inv;
        s.avg_poll_spins_pause *= inv;
        s.avg_poll_spins_yield *= inv;
        s.avg_poll_spins_sleep *= inv;
        s.avg_batches_pushed *= inv;
        s.avg_batches_fallback *= inv;
        s.avg_ids_pushed *= inv;
        s.avg_ids_fallback *= inv;
        s.avg_expansions_with_remote *= inv;
        s.avg_neighbors_local *= inv;
        s.avg_neighbors_remote *= inv;
        s.avg_vt_dedup_hits *= inv;
        s.max_poll_wait_ns = max_poll;
    }

    return s;
}

// ============================================================
// search_at_layer_pushdown — NUMA-aware beam search
// Remote neighbors batched to DualDistProxy, local computed inline.
// ============================================================

template <typename ProxyT>
void ShmHnswSearcher::search_at_layer_pushdown(
        const float* query, int32_t ep, float ep_dist,
        int level, int ef_search,
        std::vector<std::pair<float, int32_t>>& results,
        VisitedTable& vt,
        std::vector<ProxyT>& proxies) const {

    thread_local MinimaxHeap heap(0);
    if (heap.n != ef_search) heap = MinimaxHeap(ef_search);
    heap.clear();

    heap.push(ep, ep_dist);
    vt.set(ep);

    float lower_bound = (heap.size() >= ef_search) ? heap.max()
                                                     : std::numeric_limits<float>::max();

    // Per-shard batch buffers (thread-local to avoid alloc per query)
    thread_local std::vector<std::vector<int32_t>> remote_ids;
    if (remote_ids.size() < num_shards_) {
        remote_ids.resize(num_shards_);
    }
    // Full clear on function entry to flush any residue from previous query
    for (uint32_t s = 0; s < num_shards_; s++) {
        remote_ids[s].clear();
    }

    // Cache local shard's sorted_idx to skip find_shard in Phase 3
    const uint32_t local_sorted_idx = shard_id_to_sorted_idx_[local_shard_id_];

    // Branchless shard lookup: resolve strategy once outside hot loop.
    // Eliminates 2 branches per find_shard call (uniform_shards_ + shard_shift_valid_).
    const auto shard_of = [this](int32_t nid) __attribute__((always_inline)) -> uint32_t {
        uint64_t gid = static_cast<uint64_t>(nid);
        if (uniform_shards_) {
            uint32_t sorted_idx = shard_shift_valid_
                ? static_cast<uint32_t>(gid >> shard_shift_)
                : static_cast<uint32_t>(gid / ntotal_per_shard_);
            if (__builtin_expect(sorted_idx >= num_shards_, 0)) sorted_idx = num_shards_ - 1;
            return sorted_to_shard_id_[sorted_idx];
        }
        // Binary search fallback (rare: non-uniform shards)
        uint32_t lo = 0, hi = num_shards_;
        while (lo + 1 < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (shards_[mid].global_id_begin <= gid) lo = mid;
            else hi = mid;
        }
        return sorted_to_shard_id_[lo];
    };

    // Pre-compute query norm for dot_norm path (once per query, not per expansion)
    const float q_norm = use_dot_norm_ ? vec_norm_sq(query, dim_) : 0.0f;

    // Register query with all remote proxies (lazy — actual memcpy deferred to submit)
    for (uint32_t s = 0; s < num_shards_; s++) {
        if (s != local_shard_id_) {
            proxies[s].set_query(query);
        }
    }

    // Helper: try insert into heap and update lower_bound
    auto try_insert = [&](int32_t nid, float d) {
        if (d < lower_bound || heap.size() < ef_search) {
            heap.push(nid, d);
            if (heap.size() >= ef_search) {
                lower_bound = heap.max();
            }
        }
    };

    while (heap.candidates() > 0) {
        // Phase 1: classify neighbors from up to expand_batch_ candidates
        thread_local std::vector<int32_t> local_ids;
        local_ids.clear();
        uint32_t touched_mask = 0;
        bool search_done = false;

        for (uint32_t eb = 0; eb < expand_batch_ && heap.candidates() > 0; eb++) {
        auto [cd, cid] = heap.pop_min_candidate();
        if (cid < 0) { search_done = true; break; }

        if (cd > lower_bound) { search_done = true; break; }

        uint64_t begin, end;
        const int32_t* nb;
        get_neighbor_range(cid, level, begin, end, nb);

        uint64_t n_nb = end - begin;

        // Reserve capacity to avoid realloc during classify (no-op in steady state)
        local_ids.reserve(local_ids.size() + n_nb);
        for (uint32_t s = 0; s < num_shards_; s++) {
            remote_ids[s].reserve(remote_ids[s].size() + n_nb);
        }

        uint64_t cid_u = static_cast<uint64_t>(cid);

        if (cid_u >= local_id_begin_ && cid_u < local_id_end_) {
            // Local shard: data already in local NUMA memory, iterate directly
            const int32_t* nb_ptr = &nb[begin];
            uint64_t j = 0;

            // NEON batch: load 4 neighbors, vectorized empty check, scalar classify.
            // Benefit: single vector compare replaces 4 scalar branches for
            // EMPTY_NEIGHBOR detection; prefetch covers next batch uniformly.
            const int32x4_t empty_vec = vdupq_n_s32(EMPTY_NEIGHBOR);
            for (; j + 4 <= n_nb; j += 4) {
                int32x4_t v = vld1q_s32(nb_ptr + j);
                // Vectorized empty check: any lane == EMPTY_NEIGHBOR?
                uint32x4_t cmp = vceqq_s32(v, empty_vec);
                if (vmaxvq_u32(cmp) != 0) {
                    // At least one EMPTY — fall through to scalar tail
                    break;
                }

                // Prefetch vt entries: lookahead to next batch
                if (j + 6 < n_nb) {
                    vt.prefetch(nb_ptr[j + 4]);
                    vt.prefetch(nb_ptr[j + 5]);
                }

                // All 4 are valid neighbors — classify each
                int32_t n0 = vgetq_lane_s32(v, 0);
                int32_t n1 = vgetq_lane_s32(v, 1);
                int32_t n2 = vgetq_lane_s32(v, 2);
                int32_t n3 = vgetq_lane_s32(v, 3);

#define LOCAL_CLASSIFY(nid) do { \
    if (vt.set(nid)) { \
        uint64_t gid = static_cast<uint64_t>(nid); \
        if (gid >= local_id_begin_ && gid < local_id_end_) { \
            local_ids.push_back(nid); \
        } else { \
            uint32_t s = shard_of(nid); \
            remote_ids[s].push_back(nid); \
            touched_mask |= (1u << s); \
        } \
    } \
} while(0)
                LOCAL_CLASSIFY(n0);
                LOCAL_CLASSIFY(n1);
                LOCAL_CLASSIFY(n2);
                LOCAL_CLASSIFY(n3);
#undef LOCAL_CLASSIFY
            }
            // Scalar tail (< 4 remaining, or batch hit EMPTY_NEIGHBOR)
            for (; j < n_nb; j++) {
                int32_t nid = nb_ptr[j];
                if (nid == EMPTY_NEIGHBOR) break;

                if (j + 2 < n_nb) {
                    int32_t next2 = nb_ptr[j + 2];
                    if (next2 != EMPTY_NEIGHBOR) vt.prefetch(next2);
                }

                if (!vt.set(nid)) continue;

                uint64_t gid = static_cast<uint64_t>(nid);
                if (gid >= local_id_begin_ && gid < local_id_end_) {
                    local_ids.push_back(nid);
                } else {
                    uint32_t s = shard_of(nid);
                    remote_ids[s].push_back(nid);
                    touched_mask |= (1u << s);
                }
            }
        } else {
            // Remote shard: LDNP from SHM → NEON registers → extract lanes
            // directly. Data never written to memory buffer — shortest path.
            const int32_t* src = &nb[begin];
            uint64_t j = 0;

            // Process 16 int32s (64 bytes) per iteration via LDNP
            for (; j + 16 <= n_nb; j += 16) {
                // Prefetch next batch's SHM address to warm TLB (page walk)
                if (j + 32 <= n_nb) {
                    __builtin_prefetch(src + j + 16, 0, 0);
                }

                int32x4_t v0, v1, v2, v3;
                __asm__ __volatile__(
                    "ldnp %q[a], %q[b], [%[s]]      \n"
                    "ldnp %q[c], %q[d], [%[s], #32] \n"
                    : [a]"=w"(v0), [b]"=w"(v1), [c]"=w"(v2), [d]"=w"(v3)
                    : [s]"r"(src + j)
                    : "memory"
                );

// Macro: extract lane from NEON register and classify
// prefetch_vec/prefetch_lane provide lookahead=2 for vt hash table
#define PHASE1_CLASSIFY_LANE(vec, lane, pf_vec, pf_lane) do { \
    int32_t nid = vgetq_lane_s32(vec, lane); \
    if (nid == EMPTY_NEIGHBOR) goto phase1_done; \
    { int32_t pf_nid = vgetq_lane_s32(pf_vec, pf_lane); \
      if (pf_nid != EMPTY_NEIGHBOR) vt.prefetch(pf_nid); } \
    if (vt.set(nid)) { \
        uint64_t gid = static_cast<uint64_t>(nid); \
        if (gid >= local_id_begin_ && gid < local_id_end_) { \
            local_ids.push_back(nid); \
        } else { \
            uint32_t s = shard_of(nid); \
            remote_ids[s].push_back(nid); \
            touched_mask |= (1u << s); \
        } \
    } \
} while(0)

// No-prefetch variant for last 2 lanes in a batch
#define PHASE1_CLASSIFY_LANE_NOPF(vec, lane) do { \
    int32_t nid = vgetq_lane_s32(vec, lane); \
    if (nid == EMPTY_NEIGHBOR) goto phase1_done; \
    if (vt.set(nid)) { \
        uint64_t gid = static_cast<uint64_t>(nid); \
        if (gid >= local_id_begin_ && gid < local_id_end_) { \
            local_ids.push_back(nid); \
        } else { \
            uint32_t s = shard_of(nid); \
            remote_ids[s].push_back(nid); \
            touched_mask |= (1u << s); \
        } \
    } \
} while(0)

                // Lookahead=2: when processing lane N, prefetch lane N+2
                PHASE1_CLASSIFY_LANE(v0, 0, v0, 2); PHASE1_CLASSIFY_LANE(v0, 1, v0, 3);
                PHASE1_CLASSIFY_LANE(v0, 2, v1, 0); PHASE1_CLASSIFY_LANE(v0, 3, v1, 1);
                PHASE1_CLASSIFY_LANE(v1, 0, v1, 2); PHASE1_CLASSIFY_LANE(v1, 1, v1, 3);
                PHASE1_CLASSIFY_LANE(v1, 2, v2, 0); PHASE1_CLASSIFY_LANE(v1, 3, v2, 1);
                PHASE1_CLASSIFY_LANE(v2, 0, v2, 2); PHASE1_CLASSIFY_LANE(v2, 1, v2, 3);
                PHASE1_CLASSIFY_LANE(v2, 2, v3, 0); PHASE1_CLASSIFY_LANE(v2, 3, v3, 1);
                PHASE1_CLASSIFY_LANE(v3, 0, v3, 2); PHASE1_CLASSIFY_LANE(v3, 1, v3, 3);
                // Last 2 lanes: bridge prefetch gap to next batch
                if (j + 18 <= n_nb) {
                    int32_t pf0 = src[j + 16];
                    int32_t pf1 = src[j + 17];
                    if (pf0 != EMPTY_NEIGHBOR) vt.prefetch(pf0);
                    if (pf1 != EMPTY_NEIGHBOR) vt.prefetch(pf1);
                } else if (j + 17 <= n_nb) {
                    int32_t pf0 = src[j + 16];
                    if (pf0 != EMPTY_NEIGHBOR) vt.prefetch(pf0);
                }
                PHASE1_CLASSIFY_LANE_NOPF(v3, 2); PHASE1_CLASSIFY_LANE_NOPF(v3, 3);

#undef PHASE1_CLASSIFY_LANE
#undef PHASE1_CLASSIFY_LANE_NOPF
            }

            // Tail: remaining elements (< 16), scalar read with prefetch
            for (; j < n_nb; j++) {
                int32_t nid = src[j];
                if (nid == EMPTY_NEIGHBOR) break;

                if (j + 2 < n_nb) {
                    int32_t next2 = src[j + 2];
                    if (next2 != EMPTY_NEIGHBOR) vt.prefetch(next2);
                }

                if (!vt.set(nid)) continue;

                uint64_t gid = static_cast<uint64_t>(nid);
                if (gid >= local_id_begin_ && gid < local_id_end_) {
                    local_ids.push_back(nid);
                } else {
                    uint32_t s = shard_of(nid);
                    remote_ids[s].push_back(nid);
                    touched_mask |= (1u << s);
                }
            }
        }
        phase1_done: (void)0;
        } // end expand_batch_ loop

        // Phase 2: submit remote batches FIRST (largest batch first for early service start)
        // Sorted by batch size descending so largest batches start processing earliest
        uint32_t pending_mask = 0;

        // Build shard order sorted by batch size descending
        uint32_t shard_order[32];
        uint32_t shard_count = 0;
        for (uint32_t i = 1; i < num_shards_; i++) {
            uint32_t s = (local_shard_id_ + i) % num_shards_;
            if (!(touched_mask & (1u << s))) continue;
            shard_order[shard_count++] = s;
        }
        std::sort(shard_order, shard_order + shard_count,
                  [&](uint32_t a, uint32_t b) {
                      return remote_ids[a].size() > remote_ids[b].size();
                  });

        for (uint32_t i = 0; i < shard_count; i++) {
            uint32_t s = shard_order[i];
            const size_t rs = remote_ids[s].size();

            if (rs > UINT16_MAX) {
                fprintf(stderr, "FATAL: remote batch size %zu exceeds uint16_t max\n", rs);
                abort();
            }
            // Sort remote_ids to improve cache locality on service worker side
            // (vectors are stored contiguously by id in remote SHM)
            if (rs >= 64) {
                std::sort(remote_ids[s].begin(), remote_ids[s].end());
            }
            proxies[s].submit_batch(remote_ids[s].data(),
                                    static_cast<uint16_t>(rs));
            pending_mask |= (1u << s);
        }

        // Phase 3: compute LOCAL distances (overlaps with remote service)
        // Sort local_ids to improve cache locality when expand_batch > 1
        // (vectors are stored contiguously by id in SHM)
        if (local_ids.size() >= 64) {
            std::sort(local_ids.begin(), local_ids.end());
        }
        // Batch-4 over local_ids — use get_vector_in_shard to skip redundant find_shard
        {
            size_t n = local_ids.size();
            // Prefetch first batch of vectors
            for (size_t p = 0; p < std::min(n, size_t(4)); p++) {
                prefetch_vector_in_shard(local_ids[p], local_sorted_idx);
            }
            size_t i = 0;
            if (use_dot_norm_) {
                // dot_norm path: uses pre-computed norms and hoisted q_norm
                for (; i + 4 <= n; i += 4) {
                    for (size_t p = i + 4; p < std::min(i + 8, n); p++) {
                        prefetch_vector_in_shard(local_ids[p], local_sorted_idx);
                    }
                    const float* v0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* v1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    const float* v2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                    const float* v3 = get_vector_in_shard(local_ids[i+3], local_sorted_idx);
                    float d0, d1, d2, d3;
                    dot_norm_dist_batch_4(query, v0, v1, v2, v3, dim_, q_norm,
                                          get_norm_in_shard(local_ids[i],   local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+1], local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+2], local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+3], local_sorted_idx),
                                          d0, d1, d2, d3);
                    try_insert(local_ids[i],   d0);
                    try_insert(local_ids[i+1], d1);
                    try_insert(local_ids[i+2], d2);
                    try_insert(local_ids[i+3], d3);
                }
                // Tail
                const size_t tail = n - i;
                if (tail == 3) {
                    const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    const float* tv2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                    float td0, td1, td2;
                    dot_norm_dist_batch_3(query, tv0, tv1, tv2, dim_, q_norm,
                                          get_norm_in_shard(local_ids[i],   local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+1], local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+2], local_sorted_idx),
                                          td0, td1, td2);
                    try_insert(local_ids[i],   td0);
                    try_insert(local_ids[i+1], td1);
                    try_insert(local_ids[i+2], td2);
                } else if (tail == 2) {
                    const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    float td0, td1;
                    dot_norm_dist_batch_2(query, tv0, tv1, dim_, q_norm,
                                          get_norm_in_shard(local_ids[i],   local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+1], local_sorted_idx),
                                          td0, td1);
                    try_insert(local_ids[i],   td0);
                    try_insert(local_ids[i+1], td1);
                } else if (tail == 1) {
                    float d = dot_norm_dist(query, get_vector_in_shard(local_ids[i], local_sorted_idx),
                                            dim_, q_norm,
                                            get_norm_in_shard(local_ids[i], local_sorted_idx));
                    try_insert(local_ids[i], d);
                }
            } else {
                // Original L2 path
                for (; i + 4 <= n; i += 4) {
                    for (size_t p = i + 4; p < std::min(i + 8, n); p++) {
                        prefetch_vector_in_shard(local_ids[p], local_sorted_idx);
                    }
                    const float* v0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* v1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    const float* v2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                    const float* v3 = get_vector_in_shard(local_ids[i+3], local_sorted_idx);
                    float d0, d1, d2, d3;
                    l2_sqr_batch_4(query, v0, v1, v2, v3, dim_, d0, d1, d2, d3);
                    try_insert(local_ids[i],   d0);
                    try_insert(local_ids[i+1], d1);
                    try_insert(local_ids[i+2], d2);
                    try_insert(local_ids[i+3], d3);
                }
                // Tail
                const size_t tail = n - i;
                if (tail == 3) {
                    const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    const float* tv2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                    float td0, td1, td2;
                    l2_sqr_batch_3(query, tv0, tv1, tv2, dim_, td0, td1, td2);
                    try_insert(local_ids[i],   td0);
                    try_insert(local_ids[i+1], td1);
                    try_insert(local_ids[i+2], td2);
                } else if (tail == 2) {
                    const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    float td0, td1;
                    l2_sqr_batch_2(query, tv0, tv1, dim_, td0, td1);
                    try_insert(local_ids[i],   td0);
                    try_insert(local_ids[i+1], td1);
                } else if (tail == 1) {
                    float d = l2_sqr(query, get_vector_in_shard(local_ids[i], local_sorted_idx), dim_);
                    try_insert(local_ids[i], d);
                }
            }
        }
        // Phase 4: poll and consume remote results.
        static constexpr auto MAX_POLL_WAIT = std::chrono::seconds(5);
        uint32_t no_progress = 0;
        auto poll_start = std::chrono::steady_clock::now();
        while (pending_mask) {
            bool made_progress = false;
            for (uint32_t tmp = pending_mask; tmp; ) {
                uint32_t s = __builtin_ctz(tmp);
                tmp &= tmp - 1;  // clear lowest set bit
                if (!proxies[s].try_wait_batch()) continue;

                const float* dists = proxies[s].consume_batch();
                consume_dists_neon(remote_ids[s].data(), dists,
                                   remote_ids[s].size(), try_insert);
                pending_mask &= ~(1u << s);
                made_progress = true;
            }
            if (!made_progress) {
                // Three-level backoff (pause → yield → sleep)
                ++no_progress;
                if (no_progress < 64) {
                    cpu_pause();
                } else if (no_progress < 1024) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                auto elapsed = std::chrono::steady_clock::now() - poll_start;
                if (elapsed > MAX_POLL_WAIT) {
                    auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - poll_start).count();
                    fprintf(stderr,
                            "FATAL: pushdown poll timeout after %lld ms, pending_mask=0x%x\n",
                            static_cast<long long>(wait_ms), pending_mask);
                    for (uint32_t s = 0; s < num_shards_; s++) {
                        if (!(pending_mask & (1u << s))) continue;
                        fprintf(stderr,
                                "  shard=%u local_seq=%llu req_seq=%llu resp_seq=%llu "
                                "query_epoch=%u count=%u run_gen=%llu\n",
                                s,
                                static_cast<unsigned long long>(proxies[s].debug_local_seq()),
                                static_cast<unsigned long long>(proxies[s].debug_req_seq()),
                                static_cast<unsigned long long>(proxies[s].debug_resp_seq()),
                                proxies[s].debug_query_epoch(),
                                static_cast<unsigned>(proxies[s].debug_count()),
                                static_cast<unsigned long long>(proxies[s].debug_run_generation()));
                    }
                    abort();
                }
            } else {
                no_progress = 0;
                poll_start = std::chrono::steady_clock::now();
            }
        }

        // End of expansion: clear touched shard buffers for next iteration
        for (uint32_t tmp = touched_mask; tmp; tmp &= (tmp - 1)) {
            remote_ids[__builtin_ctz(tmp)].clear();
        }
        if (search_done) break;
    }

    heap.extract_sorted(results);
}

// ============================================================
// search_at_layer_pushdown_profiled — instrumented NUMA-aware beam search
// Same logic as search_at_layer_pushdown, with Phase 4 profiling.
// Templated on ProxyT (DualDistProxy).
// ============================================================

template <typename ProxyT>
void ShmHnswSearcher::search_at_layer_pushdown_profiled(
        const float* query, int32_t ep, float ep_dist,
        int level, int ef_search,
        std::vector<std::pair<float, int32_t>>& results,
        VisitedTable& vt,
        std::vector<ProxyT>& proxies,
        SearchProfile& prof) const {

    using Clock = std::chrono::steady_clock;
    auto& pd = prof.pushdown;

    thread_local MinimaxHeap heap(0);
    if (heap.n != ef_search) heap = MinimaxHeap(ef_search);
    heap.clear();

    heap.push(ep, ep_dist);
    vt.set(ep);

    float lower_bound = (heap.size() >= ef_search) ? heap.max()
                                                     : std::numeric_limits<float>::max();

    thread_local std::vector<std::vector<int32_t>> remote_ids;
    if (remote_ids.size() < num_shards_) {
        remote_ids.resize(num_shards_);
    }
    // Full clear on function entry to flush any residue from previous query
    for (uint32_t s = 0; s < num_shards_; s++) {
        remote_ids[s].clear();
    }

    // Cache local shard's sorted_idx to skip find_shard in Phase 3
    const uint32_t local_sorted_idx = shard_id_to_sorted_idx_[local_shard_id_];

    // Branchless shard lookup: resolve strategy once outside hot loop.
    const auto shard_of = [this](int32_t nid) __attribute__((always_inline)) -> uint32_t {
        uint64_t gid = static_cast<uint64_t>(nid);
        if (uniform_shards_) {
            uint32_t sorted_idx = shard_shift_valid_
                ? static_cast<uint32_t>(gid >> shard_shift_)
                : static_cast<uint32_t>(gid / ntotal_per_shard_);
            if (__builtin_expect(sorted_idx >= num_shards_, 0)) sorted_idx = num_shards_ - 1;
            return sorted_to_shard_id_[sorted_idx];
        }
        uint32_t lo = 0, hi = num_shards_;
        while (lo + 1 < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (shards_[mid].global_id_begin <= gid) lo = mid;
            else hi = mid;
        }
        return sorted_to_shard_id_[lo];
    };

    // Pre-compute query norm for dot_norm path (once per query, not per expansion)
    const float q_norm = use_dot_norm_ ? vec_norm_sq(query, dim_) : 0.0f;

    for (uint32_t s = 0; s < num_shards_; s++) {
        if (s != local_shard_id_) {
            proxies[s].set_query(query);
        }
    }

    auto try_insert = [&](int32_t nid, float d) {
        if (d < lower_bound || heap.size() < ef_search) {
            heap.push(nid, d);
            prof.heap_pushes++;
            if (heap.size() >= ef_search) {
                lower_bound = heap.max();
            }
        } else {
            prof.heap_rejections++;
        }
    };

    while (heap.candidates() > 0) {
        // Phase 1: classify neighbors from up to expand_batch_ candidates
        thread_local std::vector<int32_t> local_ids;
        local_ids.clear();
        uint32_t touched_mask = 0;
        bool search_done = false;

        for (uint32_t eb = 0; eb < expand_batch_ && heap.candidates() > 0; eb++) {
        auto [cd, cid] = heap.pop_min_candidate();
        if (cid < 0) { search_done = true; break; }

        if (cd > lower_bound) { search_done = true; break; }

        prof.nodes_expanded++;

        uint64_t begin, end;
        const int32_t* nb;
        auto t_graph = Clock::now();
        get_neighbor_range(cid, level, begin, end, nb);
        pd.graph_access_ns += std::chrono::duration<double, std::nano>(Clock::now() - t_graph).count();

        // Phase 1: classify neighbors
        auto t_phase1 = Clock::now();

#ifdef HNSW_PROFILE_LIGHTWEIGHT
        // Lightweight: phase-level timing only, no per-edge Clock::now()
        {
        uint64_t n_nb = end - begin;

        // Reserve capacity to avoid realloc during classify (no-op in steady state)
        local_ids.reserve(local_ids.size() + n_nb);
        for (uint32_t s = 0; s < num_shards_; s++) {
            remote_ids[s].reserve(remote_ids[s].size() + n_nb);
        }

        uint64_t cid_u = static_cast<uint64_t>(cid);

        if (cid_u >= local_id_begin_ && cid_u < local_id_end_) {
            // Local shard: iterate directly
            const int32_t* nb_ptr = &nb[begin];
            uint64_t j = 0;

            const int32x4_t empty_vec = vdupq_n_s32(EMPTY_NEIGHBOR);
            for (; j + 4 <= n_nb; j += 4) {
                int32x4_t v = vld1q_s32(nb_ptr + j);
                uint32x4_t cmp = vceqq_s32(v, empty_vec);
                if (vmaxvq_u32(cmp) != 0) break;

                prof.edges_scanned += 4;

                if (j + 6 < n_nb) {
                    vt.prefetch(nb_ptr[j + 4]);
                    vt.prefetch(nb_ptr[j + 5]);
                }

                int32_t n0 = vgetq_lane_s32(v, 0);
                int32_t n1 = vgetq_lane_s32(v, 1);
                int32_t n2 = vgetq_lane_s32(v, 2);
                int32_t n3 = vgetq_lane_s32(v, 3);

#define LOCAL_LW_CLASSIFY(nid) do { \
    if (!vt.set(nid)) { pd.vt_dedup_hits++; } else { \
        uint64_t gid = static_cast<uint64_t>(nid); \
        if (gid >= local_id_begin_ && gid < local_id_end_) { \
            local_ids.push_back(nid); pd.neighbors_local++; \
        } else { \
            uint32_t s = shard_of(nid); \
            remote_ids[s].push_back(nid); \
            touched_mask |= (1u << s); pd.neighbors_remote++; \
        } \
    } \
} while(0)
                LOCAL_LW_CLASSIFY(n0);
                LOCAL_LW_CLASSIFY(n1);
                LOCAL_LW_CLASSIFY(n2);
                LOCAL_LW_CLASSIFY(n3);
#undef LOCAL_LW_CLASSIFY
            }
            for (; j < n_nb; j++) {
                int32_t nid = nb_ptr[j];
                prof.edges_scanned++;
                if (nid == EMPTY_NEIGHBOR) break;

                if (j + 2 < n_nb) {
                    int32_t next2 = nb_ptr[j + 2];
                    if (next2 != EMPTY_NEIGHBOR) vt.prefetch(next2);
                }

                if (!vt.set(nid)) { pd.vt_dedup_hits++; continue; }

                uint64_t gid = static_cast<uint64_t>(nid);
                if (gid >= local_id_begin_ && gid < local_id_end_) {
                    local_ids.push_back(nid);
                    pd.neighbors_local++;
                } else {
                    uint32_t s = shard_of(nid);
                    remote_ids[s].push_back(nid);
                    touched_mask |= (1u << s);
                    pd.neighbors_remote++;
                }
            }
        } else {
            // Remote shard: LDNP → register → extract lanes directly
            const int32_t* src = &nb[begin];
            uint64_t j = 0;

            for (; j + 16 <= n_nb; j += 16) {
                // Prefetch next batch's SHM address to warm TLB (page walk)
                if (j + 32 <= n_nb) {
                    __builtin_prefetch(src + j + 16, 0, 0);
                }

                int32x4_t v0, v1, v2, v3;
                __asm__ __volatile__(
                    "ldnp %q[a], %q[b], [%[s]]      \n"
                    "ldnp %q[c], %q[d], [%[s], #32] \n"
                    : [a]"=w"(v0), [b]"=w"(v1), [c]"=w"(v2), [d]"=w"(v3)
                    : [s]"r"(src + j)
                    : "memory"
                );

#define PHASE1_LW_LANE(vec, lane, pf_vec, pf_lane) do { \
    int32_t nid = vgetq_lane_s32(vec, lane); \
    prof.edges_scanned++; \
    if (nid == EMPTY_NEIGHBOR) goto phase1_lw_done; \
    { int32_t pf_nid = vgetq_lane_s32(pf_vec, pf_lane); \
      if (pf_nid != EMPTY_NEIGHBOR) vt.prefetch(pf_nid); } \
    if (!vt.set(nid)) { pd.vt_dedup_hits++; break; } \
    uint64_t gid = static_cast<uint64_t>(nid); \
    if (gid >= local_id_begin_ && gid < local_id_end_) { \
        local_ids.push_back(nid); pd.neighbors_local++; \
    } else { \
        uint32_t s = shard_of(nid); \
        remote_ids[s].push_back(nid); \
        touched_mask |= (1u << s); pd.neighbors_remote++; \
    } \
} while(0)

#define PHASE1_LW_LANE_NOPF(vec, lane) do { \
    int32_t nid = vgetq_lane_s32(vec, lane); \
    prof.edges_scanned++; \
    if (nid == EMPTY_NEIGHBOR) goto phase1_lw_done; \
    if (!vt.set(nid)) { pd.vt_dedup_hits++; break; } \
    uint64_t gid = static_cast<uint64_t>(nid); \
    if (gid >= local_id_begin_ && gid < local_id_end_) { \
        local_ids.push_back(nid); pd.neighbors_local++; \
    } else { \
        uint32_t s = shard_of(nid); \
        remote_ids[s].push_back(nid); \
        touched_mask |= (1u << s); pd.neighbors_remote++; \
    } \
} while(0)

                PHASE1_LW_LANE(v0, 0, v0, 2); PHASE1_LW_LANE(v0, 1, v0, 3);
                PHASE1_LW_LANE(v0, 2, v1, 0); PHASE1_LW_LANE(v0, 3, v1, 1);
                PHASE1_LW_LANE(v1, 0, v1, 2); PHASE1_LW_LANE(v1, 1, v1, 3);
                PHASE1_LW_LANE(v1, 2, v2, 0); PHASE1_LW_LANE(v1, 3, v2, 1);
                PHASE1_LW_LANE(v2, 0, v2, 2); PHASE1_LW_LANE(v2, 1, v2, 3);
                PHASE1_LW_LANE(v2, 2, v3, 0); PHASE1_LW_LANE(v2, 3, v3, 1);
                PHASE1_LW_LANE(v3, 0, v3, 2); PHASE1_LW_LANE(v3, 1, v3, 3);
                // Last 2 lanes: bridge prefetch gap to next batch
                if (j + 18 <= n_nb) {
                    int32_t pf0 = src[j + 16];
                    int32_t pf1 = src[j + 17];
                    if (pf0 != EMPTY_NEIGHBOR) vt.prefetch(pf0);
                    if (pf1 != EMPTY_NEIGHBOR) vt.prefetch(pf1);
                } else if (j + 17 <= n_nb) {
                    int32_t pf0 = src[j + 16];
                    if (pf0 != EMPTY_NEIGHBOR) vt.prefetch(pf0);
                }
                PHASE1_LW_LANE_NOPF(v3, 2); PHASE1_LW_LANE_NOPF(v3, 3);

#undef PHASE1_LW_LANE
#undef PHASE1_LW_LANE_NOPF
            }
            // Tail
            for (; j < n_nb; j++) {
                int32_t nid = src[j];
                prof.edges_scanned++;
                if (nid == EMPTY_NEIGHBOR) break;

                if (j + 2 < n_nb) {
                    int32_t next2 = src[j + 2];
                    if (next2 != EMPTY_NEIGHBOR) vt.prefetch(next2);
                }

                if (!vt.set(nid)) { pd.vt_dedup_hits++; continue; }
                uint64_t gid = static_cast<uint64_t>(nid);
                if (gid >= local_id_begin_ && gid < local_id_end_) {
                    local_ids.push_back(nid); pd.neighbors_local++;
                } else {
                    uint32_t s = shard_of(nid);
                    remote_ids[s].push_back(nid);
                    touched_mask |= (1u << s); pd.neighbors_remote++;
                }
            }
        }
        phase1_lw_done: (void)0;
        }
        auto t_phase1_end = Clock::now();
        pd.classify_loop_ns += std::chrono::duration<double, std::nano>(t_phase1_end - t_phase1).count();
        pd.phase1_ns += std::chrono::duration<double, std::nano>(t_phase1_end - t_graph).count();
#else
        // Full: per-edge sub-phase timing (vt_check, find_shard, route_push)
        double vt_ns_acc = 0, shard_ns_acc = 0, route_ns_acc = 0;

        {
        uint64_t n_nb_full = end - begin;

        // Reserve capacity to avoid realloc during classify (no-op in steady state)
        local_ids.reserve(local_ids.size() + n_nb_full);
        for (uint32_t s = 0; s < num_shards_; s++) {
            remote_ids[s].reserve(remote_ids[s].size() + n_nb_full);
        }

        const int32_t* src_full;
        thread_local std::vector<int32_t> nb_local_full;

        // For full profile, still distinguish local/remote to avoid
        // per-element cache misses on remote SHM reads
        uint64_t cid_u_full = static_cast<uint64_t>(cid);
        if (cid_u_full >= local_id_begin_ && cid_u_full < local_id_end_) {
            src_full = &nb[begin];
        } else {
            nb_local_full.resize(n_nb_full);
            ls64_copy_from_shm(nb_local_full.data(), &nb[begin], n_nb_full * sizeof(int32_t));
            src_full = nb_local_full.data();
        }

        for (uint64_t j = 0; j < n_nb_full; j++) {
            int32_t nid = src_full[j];
            prof.edges_scanned++;
            if (nid == EMPTY_NEIGHBOR) break;

            // Prefetch vt entry with lookahead=2
            if (j + 2 < n_nb_full) {
                int32_t next2 = src_full[j + 2];
                if (next2 != EMPTY_NEIGHBOR) vt.prefetch(next2);
            }

            auto t_vt = Clock::now();
            if (!vt.set(nid)) {
                pd.vt_dedup_hits++;
                vt_ns_acc += std::chrono::duration<double, std::nano>(Clock::now() - t_vt).count();
                continue;
            }
            auto t_vt_end = Clock::now();
            vt_ns_acc += std::chrono::duration<double, std::nano>(t_vt_end - t_vt).count();

            auto t_shard = Clock::now();
            uint64_t gid = static_cast<uint64_t>(nid);
            bool is_local = (gid >= local_id_begin_ && gid < local_id_end_);
            uint32_t s = is_local ? local_shard_id_ : shard_of(nid);
            shard_ns_acc += std::chrono::duration<double, std::nano>(Clock::now() - t_shard).count();

            auto t_route = Clock::now();
            if (is_local) {
                local_ids.push_back(nid);
                pd.neighbors_local++;
            } else {
                remote_ids[s].push_back(nid);
                touched_mask |= (1u << s);
                pd.neighbors_remote++;
            }
            route_ns_acc += std::chrono::duration<double, std::nano>(Clock::now() - t_route).count();
        }
        }
        auto t_phase1_end = Clock::now();
        pd.vt_check_ns += vt_ns_acc;
        pd.find_shard_ns += shard_ns_acc;
        pd.route_push_ns += route_ns_acc;
        pd.classify_loop_ns += std::chrono::duration<double, std::nano>(t_phase1_end - t_phase1).count();
        pd.phase1_ns += std::chrono::duration<double, std::nano>(t_phase1_end - t_graph).count();
#endif
        } // end expand_batch_ loop

        // Phase 2: submit remote batches — largest batch first for early service start
        auto t_phase2 = Clock::now();
        uint32_t pending_mask = 0;

        // Build shard order sorted by batch size descending
        uint32_t shard_order[32];
        uint32_t shard_count = 0;
        for (uint32_t i = 1; i < num_shards_; i++) {
            uint32_t s = (local_shard_id_ + i) % num_shards_;
            if (!(touched_mask & (1u << s))) continue;
            shard_order[shard_count++] = s;
        }
        std::sort(shard_order, shard_order + shard_count,
                  [&](uint32_t a, uint32_t b) {
                      return remote_ids[a].size() > remote_ids[b].size();
                  });

        for (uint32_t i = 0; i < shard_count; i++) {
            uint32_t s = shard_order[i];
            const size_t rs = remote_ids[s].size();

            if (rs > UINT16_MAX) {
                fprintf(stderr, "FATAL: remote batch size %zu exceeds uint16_t max\n", rs);
                abort();
            }
            // Sort remote_ids to improve cache locality on service worker side
            // (vectors are stored contiguously by id in remote SHM)
            if (rs >= 64) {
                std::sort(remote_ids[s].begin(), remote_ids[s].end());
            }
#ifdef HNSW_PROFILE_LIGHTWEIGHT
            proxies[s].submit_batch(remote_ids[s].data(),
                                    static_cast<uint16_t>(rs));
#else
            double q_ns = 0, id_ns = 0, db_ns = 0;
            proxies[s].submit_batch(remote_ids[s].data(),
                                    static_cast<uint16_t>(rs),
                                    q_ns, id_ns, db_ns);
            pd.phase2_query_ns += q_ns;
            pd.phase2_ids_ns += id_ns;
            pd.phase2_doorbell_ns += db_ns;
#endif
            pending_mask |= (1u << s);
            pd.batches_pushed++;
            pd.ids_pushed += rs;
        }
        pd.phase2_ns += std::chrono::duration<double, std::nano>(Clock::now() - t_phase2).count();

        // Phase 3: compute LOCAL distances
        auto t_phase3 = Clock::now();
        // Sort local_ids to improve cache locality when expand_batch > 1
        // (vectors are stored contiguously by id in SHM)
        if (local_ids.size() >= 64) {
            std::sort(local_ids.begin(), local_ids.end());
        }
        // Batch-4 over local_ids with profiling — use get_vector_in_shard to skip redundant find_shard
        {
            size_t n = local_ids.size();
            // Prefetch first batch of vectors
            for (size_t p = 0; p < std::min(n, size_t(4)); p++) {
                prefetch_vector_in_shard(local_ids[p], local_sorted_idx);
            }
            size_t i = 0;
            if (use_dot_norm_) {
                for (; i + 4 <= n; i += 4) {
                    for (size_t p = i + 4; p < std::min(i + 8, n); p++) {
                        prefetch_vector_in_shard(local_ids[p], local_sorted_idx);
                    }
                    const float* v0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* v1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    const float* v2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                    const float* v3 = get_vector_in_shard(local_ids[i+3], local_sorted_idx);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    auto t0 = Clock::now();
#endif
                    float d0, d1, d2, d3;
                    dot_norm_dist_batch_4(query, v0, v1, v2, v3, dim_, q_norm,
                                          get_norm_in_shard(local_ids[i],   local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+1], local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+2], local_sorted_idx),
                                          get_norm_in_shard(local_ids[i+3], local_sorted_idx),
                                          d0, d1, d2, d3);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    auto t1 = Clock::now();
                    prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                    prof.dist_calls_local += 4;
                    try_insert(local_ids[i],   d0);
                    try_insert(local_ids[i+1], d1);
                    try_insert(local_ids[i+2], d2);
                    try_insert(local_ids[i+3], d3);
                }
                // Tail
                {
                    const size_t tail = n - i;
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    auto t0 = Clock::now();
#endif
                    if (tail == 3) {
                        const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                        const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                        const float* tv2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                        float td0, td1, td2;
                        dot_norm_dist_batch_3(query, tv0, tv1, tv2, dim_, q_norm,
                                              get_norm_in_shard(local_ids[i],   local_sorted_idx),
                                              get_norm_in_shard(local_ids[i+1], local_sorted_idx),
                                              get_norm_in_shard(local_ids[i+2], local_sorted_idx),
                                              td0, td1, td2);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                        auto t1 = Clock::now();
                        prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                        prof.dist_calls_local += 3;
                        try_insert(local_ids[i],   td0);
                        try_insert(local_ids[i+1], td1);
                        try_insert(local_ids[i+2], td2);
                    } else if (tail == 2) {
                        const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                        const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                        float td0, td1;
                        dot_norm_dist_batch_2(query, tv0, tv1, dim_, q_norm,
                                              get_norm_in_shard(local_ids[i],   local_sorted_idx),
                                              get_norm_in_shard(local_ids[i+1], local_sorted_idx),
                                              td0, td1);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                        auto t1 = Clock::now();
                        prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                        prof.dist_calls_local += 2;
                        try_insert(local_ids[i],   td0);
                        try_insert(local_ids[i+1], td1);
                    } else if (tail == 1) {
                        float d = dot_norm_dist(query, get_vector_in_shard(local_ids[i], local_sorted_idx),
                                                dim_, q_norm,
                                                get_norm_in_shard(local_ids[i], local_sorted_idx));
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                        auto t1 = Clock::now();
                        prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                        prof.dist_calls_local++;
                        try_insert(local_ids[i], d);
                    }
                }
            } else {
                // Original L2 path
                for (; i + 4 <= n; i += 4) {
                    for (size_t p = i + 4; p < std::min(i + 8, n); p++) {
                        prefetch_vector_in_shard(local_ids[p], local_sorted_idx);
                    }
                    const float* v0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                    const float* v1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                    const float* v2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                    const float* v3 = get_vector_in_shard(local_ids[i+3], local_sorted_idx);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    auto t0 = Clock::now();
#endif
                    float d0, d1, d2, d3;
                    l2_sqr_batch_4(query, v0, v1, v2, v3, dim_, d0, d1, d2, d3);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    auto t1 = Clock::now();
                    prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                    prof.dist_calls_local += 4;
                    try_insert(local_ids[i],   d0);
                    try_insert(local_ids[i+1], d1);
                    try_insert(local_ids[i+2], d2);
                    try_insert(local_ids[i+3], d3);
                }
                // Tail
                {
                    const size_t tail = n - i;
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    auto t0 = Clock::now();
#endif
                    if (tail == 3) {
                        const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                        const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                        const float* tv2 = get_vector_in_shard(local_ids[i+2], local_sorted_idx);
                        float td0, td1, td2;
                        l2_sqr_batch_3(query, tv0, tv1, tv2, dim_, td0, td1, td2);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                        auto t1 = Clock::now();
                        prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                        prof.dist_calls_local += 3;
                        try_insert(local_ids[i],   td0);
                        try_insert(local_ids[i+1], td1);
                        try_insert(local_ids[i+2], td2);
                    } else if (tail == 2) {
                        const float* tv0 = get_vector_in_shard(local_ids[i],   local_sorted_idx);
                        const float* tv1 = get_vector_in_shard(local_ids[i+1], local_sorted_idx);
                        float td0, td1;
                        l2_sqr_batch_2(query, tv0, tv1, dim_, td0, td1);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                        auto t1 = Clock::now();
                        prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                        prof.dist_calls_local += 2;
                        try_insert(local_ids[i],   td0);
                        try_insert(local_ids[i+1], td1);
                    } else if (tail == 1) {
                        float d = l2_sqr(query, get_vector_in_shard(local_ids[i], local_sorted_idx), dim_);
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                        auto t1 = Clock::now();
                        prof.dist_ns_local += std::chrono::duration<double, std::nano>(t1 - t0).count();
#endif
                        prof.dist_calls_local++;
                        try_insert(local_ids[i], d);
                    }
                }
            }
        }
        pd.phase3_ns += std::chrono::duration<double, std::nano>(Clock::now() - t_phase3).count();

        // Phase 4: poll and consume remote results (instrumented)
        if (pending_mask) {
            pd.expansions_with_remote++;
            auto t_poll_start = Clock::now();

#ifdef HNSW_PROFILE_LIGHTWEIGHT
            // Lightweight: only record total poll_wait_ns, no per-iteration breakdown
            static constexpr uint32_t TIMEOUT_CHECK_INTERVAL = 4096;
            static constexpr auto MAX_POLL_WAIT = std::chrono::seconds(5);
            uint32_t no_progress = 0;
            auto poll_start = Clock::now();
            while (pending_mask) {
                bool made_progress = false;
                for (uint32_t tmp = pending_mask; tmp; ) {
                    uint32_t s = __builtin_ctz(tmp);
                    tmp &= tmp - 1;
                    if (!proxies[s].try_wait_batch()) continue;

                    const float* dists = proxies[s].consume_batch();
                    consume_dists_neon(remote_ids[s].data(), dists,
                                       remote_ids[s].size(), try_insert);
                    pending_mask &= ~(1u << s);
                    made_progress = true;
                }
                if (!made_progress) {
                    ++no_progress;
                    if (no_progress < 64) {
                        cpu_pause();
                    } else if (no_progress < 1024) {
                        std::this_thread::yield();
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                    if (Clock::now() - poll_start > MAX_POLL_WAIT) {
                        auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            Clock::now() - poll_start).count();
                        fprintf(stderr,
                                "FATAL: pushdown poll timeout after %lld ms, pending_mask=0x%x\n",
                                static_cast<long long>(wait_ms), pending_mask);
                        for (uint32_t s = 0; s < num_shards_; s++) {
                            if (!(pending_mask & (1u << s))) continue;
                            fprintf(stderr,
                                    "  shard=%u local_seq=%llu req_seq=%llu resp_seq=%llu "
                                    "query_epoch=%u count=%u run_gen=%llu\n",
                                    s,
                                    static_cast<unsigned long long>(proxies[s].debug_local_seq()),
                                    static_cast<unsigned long long>(proxies[s].debug_req_seq()),
                                    static_cast<unsigned long long>(proxies[s].debug_resp_seq()),
                                    proxies[s].debug_query_epoch(),
                                    static_cast<unsigned>(proxies[s].debug_count()),
                                    static_cast<unsigned long long>(proxies[s].debug_run_generation()));
                        }
                        abort();
                    }
                } else {
                    no_progress = 0;
                    poll_start = Clock::now();
                }
            }

            double this_poll_ns = std::chrono::duration<double, std::nano>(
                Clock::now() - t_poll_start).count();
            pd.poll_wait_ns += this_poll_ns;
            if (this_poll_ns > pd.max_poll_wait_ns) {
                pd.max_poll_wait_ns = this_poll_ns;
            }
#else
            double this_consume_ns = 0;  // consume time for this expansion only

            static constexpr uint32_t TIMEOUT_CHECK_INTERVAL = 4096;
            static constexpr uint32_t SPIN_SAMPLE_INTERVAL = 64;  // sample every N spins
            static constexpr auto MAX_POLL_WAIT = std::chrono::seconds(5);
            uint32_t no_progress = 0;
            uint32_t spin_sample_counter = 0;
            uint32_t spin_samples_taken = 0;
            uint32_t spins_this_expansion = 0;  // local spin count for extrapolation
            double sampled_spin_ns = 0;
            auto poll_start = Clock::now();
            while (pending_mask) {
                bool made_progress = false;
                for (uint32_t tmp = pending_mask; tmp; ) {
                    uint32_t s = __builtin_ctz(tmp);
                    tmp &= tmp - 1;  // clear lowest set bit
                    if (!proxies[s].try_wait_batch()) continue;

                    auto t_consume = Clock::now();
                    const float* dists = proxies[s].consume_batch();
                    consume_dists_neon(remote_ids[s].data(), dists,
                                       remote_ids[s].size(), try_insert);
                    double consume_ns = std::chrono::duration<double, std::nano>(
                        Clock::now() - t_consume).count();
                    pd.poll_consume_ns += consume_ns;
                    this_consume_ns += consume_ns;
                    pending_mask &= ~(1u << s);
                    made_progress = true;
                }
                if (!made_progress) {
                    // Shared core: three-level backoff (pause → yield → sleep)
                    // Sampled timing: only Clock::now() every SPIN_SAMPLE_INTERVAL iters
                    ++no_progress;
                    ++spins_this_expansion;
                    bool do_sample = (++spin_sample_counter >= SPIN_SAMPLE_INTERVAL);
                    auto t_spin = Clock::now();

                    if (no_progress < 64) {
                        cpu_pause();
                        pd.poll_spins_pause++;
                    } else if (no_progress < 1024) {
                        std::this_thread::yield();
                        pd.poll_spins_yield++;
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                        pd.poll_spins_sleep++;
                    }

                    if (do_sample) {
                        sampled_spin_ns += std::chrono::duration<double, std::nano>(
                            Clock::now() - t_spin).count();
                        spin_samples_taken++;
                        spin_sample_counter = 0;
                    }

                    // Timeout check: interval-based (1 Clock::now per 4096 iters)
                    if (no_progress % TIMEOUT_CHECK_INTERVAL == 0) {
                        if (Clock::now() - poll_start > MAX_POLL_WAIT) {
                            auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                Clock::now() - poll_start).count();
                            fprintf(stderr,
                                    "FATAL: pushdown poll timeout after %lld ms, pending_mask=0x%x\n",
                                    static_cast<long long>(wait_ms), pending_mask);
                            for (uint32_t s = 0; s < num_shards_; s++) {
                                if (!(pending_mask & (1u << s))) continue;
                                fprintf(stderr,
                                        "  shard=%u local_seq=%llu req_seq=%llu resp_seq=%llu "
                                        "query_epoch=%u count=%u run_gen=%llu\n",
                                        s,
                                        static_cast<unsigned long long>(proxies[s].debug_local_seq()),
                                        static_cast<unsigned long long>(proxies[s].debug_req_seq()),
                                        static_cast<unsigned long long>(proxies[s].debug_resp_seq()),
                                        proxies[s].debug_query_epoch(),
                                        static_cast<unsigned>(proxies[s].debug_count()),
                                        static_cast<unsigned long long>(proxies[s].debug_run_generation()));
                            }
                            abort();
                        }
                    }
                } else {
                    no_progress = 0;
                    poll_start = Clock::now();
                }
            }

            double this_poll_ns = std::chrono::duration<double, std::nano>(
                Clock::now() - t_poll_start).count();
            pd.poll_wait_ns += this_poll_ns;
            if (this_poll_ns > pd.max_poll_wait_ns) {
                pd.max_poll_wait_ns = this_poll_ns;
            }

            // Compute spin time for this expansion
            double this_spin_ns = 0;
            // Shared core: extrapolate from sampled iterations
            if (spin_samples_taken > 0) {
                double avg_spin_per_sample = sampled_spin_ns / spin_samples_taken;
                // Each sample covers SPIN_SAMPLE_INTERVAL iterations worth of one spin
                // Extrapolate to total spins this expansion
                this_spin_ns = avg_spin_per_sample * (static_cast<double>(spins_this_expansion) / SPIN_SAMPLE_INTERVAL);
                // Cap: extrapolated spin cannot exceed available time
                double available = this_poll_ns - this_consume_ns;
                if (this_spin_ns > available) this_spin_ns = available;
            } else {
                // No samples taken (very few spins) — derive from total poll time
                this_spin_ns = this_poll_ns - this_consume_ns;
            }
            pd.poll_spin_ns += this_spin_ns;

            // Overhead = total - consume - spin (loop control, try_wait_batch probes)
            // May be slightly negative due to extrapolation jitter — not clamped
            // so negative values surface as a diagnostic signal.
            pd.poll_overhead_ns += this_poll_ns - this_consume_ns - this_spin_ns;
#endif
        }

        // End of expansion: clear touched shard buffers for next iteration
        for (uint32_t tmp = touched_mask; tmp; tmp &= (tmp - 1)) {
            remote_ids[__builtin_ctz(tmp)].clear();
        }
        if (search_done) break;
    }

    prof.dist_calls_pushed += pd.ids_pushed;
    heap.extract_sorted(results);
}

// Explicit template instantiations for pushdown proxy types
template void ShmHnswSearcher::search_at_layer_pushdown<DualDistProxy>(
        const float*, int32_t, float, int, int,
        std::vector<std::pair<float, int32_t>>&, VisitedTable&,
        std::vector<DualDistProxy>&) const;

template void ShmHnswSearcher::search_at_layer_pushdown_profiled<DualDistProxy>(
        const float*, int32_t, float, int, int,
        std::vector<std::pair<float, int32_t>>&, VisitedTable&,
        std::vector<DualDistProxy>&, SearchProfile&) const;

} // namespace shm_hnsw
