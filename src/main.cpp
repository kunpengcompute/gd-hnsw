#include "shm_layout.h"
#include "shm_hnsw_search.h"
#include "dist_service.h"
#include "faiss_extractor.h"
#include "index_io.h"
#include "hdf5_loader.h"
#include "bin_loader.h"

#include <mpi.h>
#include <sys/mman.h>   // PROT_*, MAP_*, madvise
#include <unistd.h>     // getpid, gethostname

#ifdef __linux__
#include <sched.h>      // sched_getcpu
#include <numa.h>       // numa_node_of_cpu
#include <pthread.h>    // pthread_setaffinity_np
#endif

#include <ubs_mem_def.h>
#include <ubs_mem.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <omp.h>

using namespace shm_hnsw;

// ============================================================
// ubs-mem shared memory helpers
// ============================================================

// Job-unique ID (rank 0 PID, broadcast to all ranks) to avoid name collisions
static uint32_t g_job_id = 0;

// ubs-mem flags for index shard shm (hardcoded half-noncache)
static uint64_t g_ubsmem_flags = UBSM_FLAG_CACHE;
// ubs-mem flags for pushdown channel shm (always noncache in cache mode)
static uint64_t g_channel_flags = UBSM_FLAG_NONCACHE;

// Provider for NUMA-local allocation (initialized at runtime)
static ubs_mem_provider_t g_provider{};

// 4MB alignment (ubs-mem minimum allocation granularity)
static constexpr size_t UBSMEM_ALIGN = 4UL * 1024 * 1024;
static size_t ubsmem_align_size(size_t sz) {
    if (sz == 0) return UBSMEM_ALIGN;
    return (sz + UBSMEM_ALIGN - 1) & ~(UBSMEM_ALIGN - 1);
}

// Deallocate with retry on UBSM_ERR_IN_USING (6024)
static int ubsmem_deallocate_retry(const char* name, int max_retries = 3) {
    for (int i = 0; i <= max_retries; i++) {
        int ret = ubsmem_shmem_deallocate(name);
        if (ret == UBSM_OK || ret == UBSM_ERR_NOT_FOUND) return UBSM_OK;
        if (ret == UBSM_ERR_IN_USING && i < max_retries) {
            usleep(100000 * (1 << i));  // 100ms, 200ms, 400ms
            continue;
        }
        return ret;
    }
    return UBSM_ERR_IN_USING;
}

// Per-shard naming (no leading '/' — ubs-mem constraint)
static std::string shm_name_for_shard(uint32_t shard_id) {
    return "shm_hnsw_" + std::to_string(g_job_id) + "_" + std::to_string(shard_id);
}

// Dual-channel mode: task_inbox (service rank creates, service reads tasks)
static std::string task_inbox_name_for_rank(uint32_t rank) {
    return "shm_hnsw_task_" + std::to_string(g_job_id) + "_" + std::to_string(rank);
}

// Dual-channel mode: result_inbox (search rank creates, search reads results)
static std::string result_inbox_name_for_rank(uint32_t rank) {
    return "shm_hnsw_result_" + std::to_string(g_job_id) + "_" + std::to_string(rank);
}

// Map shard_id to dense [0, num_remote-1] index, skipping my_shard
static uint32_t compressed_remote_idx(uint32_t shard, uint32_t my_shard) {
    return (shard < my_shard) ? shard : shard - 1;
}

// ============================================================
// Index file I/O — now in index_io.h / index_io.cpp
// ============================================================
using shm_hnsw::index_file_for_shard;
using shm_hnsw::IdMapLoadStatus;
using shm_hnsw::save_idmap_file;
using shm_hnsw::load_idmap_file;
using shm_hnsw::save_centroids_file;
using shm_hnsw::load_centroids_file;
using shm_hnsw::save_shard_buffers;
using shm_hnsw::load_shard_buffers;

// ============================================================
// Graph-only compaction: copy graph blocks into SHM, skip BLK_VECTORS.
// Assumes SuperBlock header is already at dst. Updates dst SuperBlock
// in-place (blocks, total_bytes). Does NOT set state or recompute CRCs.
// ============================================================
static void compact_graph_to_shm(void* dst, const char* src_data,
                                  const SuperBlockV1& src_sb) {
    auto* shm_sb = static_cast<SuperBlockV1*>(dst);
    char* d = static_cast<char*>(dst);
    uint64_t offset = SUPERBLOCK_SIZE;

    shm_sb->blocks[BLK_VECTORS].off = 0;
    shm_sb->blocks[BLK_VECTORS].bytes = 0;

    // Copy each graph block with proper alignment
    static constexpr int graph_blocks[] = {
        BLK_LEVELS, BLK_OFFSETS, BLK_NEIGHBORS, BLK_CUM_NNEIGHBOR
    };
    for (int blk : graph_blocks) {
        uint64_t new_off = align_up(offset, shm_sb->blocks[blk].alignment);
        std::memcpy(d + new_off,
                    src_data + src_sb.blocks[blk].off,
                    src_sb.blocks[blk].bytes);
        shm_sb->blocks[blk].off = new_off;
        offset = new_off + shm_sb->blocks[blk].bytes;
    }

    shm_sb->total_bytes = align_up(offset, PAGE_ALIGN);
}


static void* create_shm(const std::string& name, const std::vector<char>& buf,
                         size_t& out_alloc_size, bool keep_vectors = false) {
    ubsmem_deallocate_retry(name.c_str());

    const auto& src_sb = *reinterpret_cast<const SuperBlockV1*>(buf.data());

    // Single-shard: keep full buffer (including vectors). Pushdown: graph-only.
    size_t shm_size;
    if (keep_vectors) {
        shm_size = static_cast<size_t>(src_sb.total_bytes);
    } else {
        shm_size = static_cast<size_t>(src_sb.total_bytes - src_sb.blocks[BLK_VECTORS].bytes);
        if (shm_size < SUPERBLOCK_SIZE) shm_size = src_sb.total_bytes;
    }

    size_t alloc_size = ubsmem_align_size(shm_size);
    int ret = ubsmem_shmem_allocate_with_provider(&g_provider, name.c_str(), alloc_size, 0600, g_ubsmem_flags);
    if (ret != UBSM_OK) {
        fprintf(stderr, "ubsmem_shmem_allocate_with_provider('%s', %zu) failed: %d\n",
                name.c_str(), alloc_size, ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    void* ptr = nullptr;
    ret = ubsmem_shmem_map(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, name.c_str(), 0, &ptr);
    if (ret != UBSM_OK || ptr == nullptr) {
        fprintf(stderr, "ubsmem_shmem_map('%s', %zu) failed: %d\n",
                name.c_str(), alloc_size, ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (keep_vectors) {
        // Full copy: preserve all blocks including vectors
        std::memcpy(ptr, buf.data(), static_cast<size_t>(src_sb.total_bytes));
    } else {
        // Graph-only compaction: skip BLK_VECTORS
        std::memcpy(ptr, buf.data(), SUPERBLOCK_SIZE);
        compact_graph_to_shm(ptr, buf.data(), src_sb);

        auto* shm_sb = static_cast<SuperBlockV1*>(ptr);
        shm_sb->payload_crc64 = compute_payload_crc64(ptr, *shm_sb);
        shm_sb->header_crc64 = compute_header_crc64(*shm_sb);
    }

#ifdef __linux__
    auto* final_sb = static_cast<SuperBlockV1*>(ptr);
    madvise(static_cast<char*>(ptr) + final_sb->blocks[BLK_NEIGHBORS].off,
            final_sb->blocks[BLK_NEIGHBORS].bytes, MADV_HUGEPAGE);
#endif

    out_alloc_size = alloc_size;
    return ptr;
}

// ============================================================
// Direct-load: each rank reads its own shard file into shm
// ============================================================

struct LocalShardMeta {
    uint64_t ntotal;
    uint32_t dim;
    uint32_t M;
    uint32_t ef_search;
    int32_t  max_level;
    int32_t  entry_point;
    uint32_t num_shards;
    uint32_t shard_id;
    uint64_t ntotal_local;
    uint64_t global_id_begin;
    uint64_t file_size;
    uint64_t shm_alloc_size;  // actual graph-only SHM allocation size
    bool ok;
    std::string error;
};

// POD mirror of LocalShardMeta for MPI_Allgather (cross-shard validation)
struct ShardMetaWire {
    uint64_t ntotal;
    uint32_t dim;
    uint32_t M;
    int32_t  max_level;
    int32_t  entry_point;
    uint32_t num_shards;
    uint32_t shard_id;
    uint64_t ntotal_local;
    uint64_t global_id_begin;
};
static_assert(std::is_trivially_copyable_v<ShardMetaWire>,
              "ShardMetaWire must be trivially copyable for MPI_BYTE transfer");

static LocalShardMeta load_local_shard_direct_to_shm(
        const std::string& base_path, uint32_t shard_id,
        const std::string& shm_name, bool keep_vectors = false) {
    LocalShardMeta meta{};
    meta.ok = false;

    std::string path = index_file_for_shard(base_path, shard_id);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        meta.error = std::string("cannot open '") + path + "': " + strerror(errno);
        return meta;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (file_size <= 0) {
        fclose(fp);
        meta.error = std::string("'") + path + "' is empty or unreadable";
        return meta;
    }

    // Read SuperBlock from file to determine layout
    SuperBlockV1 file_sb{};
    if (fread(&file_sb, sizeof(SuperBlockV1), 1, fp) != 1) {
        fclose(fp);
        meta.error = "cannot read SuperBlock from file";
        return meta;
    }

    // Validate the file (read entire file into temp buffer for CRC validation)
    fseek(fp, 0, SEEK_SET);
    std::vector<char> file_buf(static_cast<size_t>(file_size));
    size_t nread = fread(file_buf.data(), 1, static_cast<size_t>(file_size), fp);
    fclose(fp);
    if (nread != static_cast<size_t>(file_size)) {
        meta.error = "short read: " + std::to_string(nread) + " / " + std::to_string(file_size);
        return meta;
    }

    auto vr = validate_superblock(file_buf.data(), static_cast<uint64_t>(file_size));
    if (!vr.ok) {
        meta.error = "validation failed: " + vr.error;
        return meta;
    }

    // Single-shard: keep full buffer (including vectors). Pushdown: graph-only.
    size_t shm_size;
    if (keep_vectors) {
        shm_size = static_cast<size_t>(file_sb.total_bytes);
    } else {
        shm_size = static_cast<size_t>(file_sb.total_bytes - file_sb.blocks[BLK_VECTORS].bytes);
        if (shm_size < SUPERBLOCK_SIZE) shm_size = file_sb.total_bytes;
    }

    // Create ubs-mem shared memory region
    ubsmem_deallocate_retry(shm_name.c_str());

    size_t alloc_size = ubsmem_align_size(shm_size);
    int ret = ubsmem_shmem_allocate_with_provider(&g_provider, shm_name.c_str(), alloc_size, 0600, g_ubsmem_flags);
    if (ret != UBSM_OK) {
        meta.error = "ubsmem_shmem_allocate_with_provider failed: " + std::to_string(ret);
        return meta;
    }

    void* ptr = nullptr;
    ret = ubsmem_shmem_map(nullptr, alloc_size,
                           PROT_READ | PROT_WRITE, MAP_SHARED,
                           shm_name.c_str(), 0, &ptr);
    if (ret != UBSM_OK || ptr == nullptr) {
        ubsmem_deallocate_retry(shm_name.c_str());
        meta.error = "ubsmem_shmem_map failed: " + std::to_string(ret);
        return meta;
    }

    // Copy SuperBlock header
    std::memcpy(ptr, file_buf.data(), SUPERBLOCK_SIZE);
    auto* shm_sb = static_cast<SuperBlockV1*>(ptr);

    if (keep_vectors) {
        // Full copy: preserve all blocks including vectors
        std::memcpy(static_cast<char*>(ptr) + SUPERBLOCK_SIZE,
                    file_buf.data() + SUPERBLOCK_SIZE,
                    static_cast<size_t>(file_sb.total_bytes) - SUPERBLOCK_SIZE);

        // Transition to ACTIVE and recompute CRCs
        shm_sb->state = static_cast<uint32_t>(ShmState::ACTIVE);
        shm_sb->payload_crc64 = compute_payload_crc64(ptr, *shm_sb);
        shm_sb->header_crc64 = compute_header_crc64(*shm_sb);
    } else {
        // Graph-only compaction: skip BLK_VECTORS
        compact_graph_to_shm(ptr, file_buf.data(), file_sb);
        shm_sb->state = static_cast<uint32_t>(ShmState::ACTIVE);
        shm_sb->payload_crc64 = compute_payload_crc64(ptr, *shm_sb);
        shm_sb->header_crc64 = compute_header_crc64(*shm_sb);
    }

#ifdef __linux__
    madvise(static_cast<char*>(ptr) + shm_sb->blocks[BLK_NEIGHBORS].off,
            shm_sb->blocks[BLK_NEIGHBORS].bytes, MADV_HUGEPAGE);
#endif

    meta.ok = true;
    meta.ntotal = shm_sb->ntotal;
    meta.dim = shm_sb->dim;
    meta.M = shm_sb->M;
    meta.ef_search = shm_sb->ef_search;
    meta.max_level = shm_sb->max_level;
    meta.entry_point = shm_sb->entry_point;
    meta.num_shards = shm_sb->num_shards;
    meta.shard_id = shm_sb->shard_id;
    meta.ntotal_local = shm_sb->ntotal_local;
    meta.global_id_begin = shm_sb->global_id_begin;
    meta.file_size = static_cast<uint64_t>(file_size);
    meta.shm_alloc_size = alloc_size;

    // Release the writable mapping — shm object persists;
    // all ranks will re-open read-only later via open_shm_readonly().
    ubsmem_shmem_unmap(ptr, alloc_size);

    return meta;
}

static const void* open_shm_readonly(const std::string& name, uint64_t total_bytes) {
    size_t alloc_size = ubsmem_align_size(static_cast<size_t>(total_bytes));
    void* ptr = nullptr;
    int ret = ubsmem_shmem_map(nullptr, alloc_size, PROT_READ,
                               MAP_SHARED, name.c_str(), 0, &ptr);
    if (ret != UBSM_OK || ptr == nullptr) {
        fprintf(stderr, "ubsmem_shmem_map readonly '%s' failed: %d\n",
                name.c_str(), ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return ptr;
}

// ============================================================
// MPI chunked broadcast
// ============================================================

static void chunked_bcast_bytes(void* data, uint64_t total, int root, MPI_Comm comm) {
    uint64_t offset = 0;
    auto* p = static_cast<char*>(data);
    while (offset < total) {
        int chunk = static_cast<int>(
            std::min(static_cast<uint64_t>(MPI_CHUNK_BYTES), total - offset));
        MPI_Bcast(p + offset, chunk, MPI_BYTE, root, comm);
        offset += chunk;
    }
}

// ============================================================
// Recall@K computation
// ============================================================

struct RecallStats {
    uint64_t hits = 0;
    uint64_t total = 0;
};

static RecallStats compute_recall_stats(const int32_t* result_ids, uint64_t nq,
                                        const int32_t* gt, uint32_t gt_k,
                                        int32_t k,
                                        const std::vector<uint32_t>* new_to_old_idmap) {
    RecallStats stats;
    stats.total = nq * static_cast<uint64_t>(std::min(k, static_cast<int32_t>(gt_k)));
    for (uint64_t q = 0; q < nq; q++) {
        const int32_t* res_row = result_ids + q * k;
        const int32_t* gt_row = gt + q * gt_k;
        for (int32_t i = 0; i < k; i++) {
            int32_t id = res_row[i];
            if (id < 0) continue;
            if (new_to_old_idmap) {
                if (static_cast<size_t>(id) >= new_to_old_idmap->size()) continue;
                id = static_cast<int32_t>((*new_to_old_idmap)[static_cast<size_t>(id)]);
            }
            for (int32_t j = 0; j < k && j < static_cast<int32_t>(gt_k); j++) {
                if (gt_row[j] == id) {
                    stats.hits++;
                    break;
                }
            }
        }
    }
    return stats;
}

// ============================================================
// Usage
// ============================================================

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -d, --dataset PATH    Dataset path (HDF5 file, .bin file, or directory with base/query/gt.bin)\n"
        "  -M, --M INT           HNSW M parameter (default: 16)\n"
        "  -c, --ef-construction INT  ef_construction (default: 200)\n"
        "  -e, --ef-search INT   ef_search (default: 64)\n"
        "  -k, --topk INT        top-K results (default: 10)\n"
        "  -t, --threads INT     threads per process (default: 24)\n"
        "  -r, --rounds INT      search rounds for stable results (default: 5)\n"
        "  -s, --shards INT      number of shards (default: world_size, 1 per rank)\n"
        "  -p, --profile         enable search profiling output\n"
        "      --search-threads INT  search threads (pushdown, default: auto)\n"
        "      --service-threads INT service threads per shard (pushdown, default: auto)\n"
        "      --cluster         build with k-means clustered shard partition\n"
        "      --save-index PATH save built index to files (PATH.shard_N)\n"
        "      --load-index PATH load index from files (skip build, still needs --dataset)\n"
        "      --dataset-format FMT  dataset format: auto|hdf5|bin (default: auto)\n"
        "      --base-bin PATH  override base vectors file (default: base.bin in dataset dir)\n"
        "      --query-bin PATH override query vectors file (default: query.bin in dataset dir)\n"
        "      --gt-bin PATH    override ground truth file (default: gt.bin in dataset dir)\n"
        "      --use-dot-norm   use dot+norm distance kernel (experimental, local path only)\n"
        "      --expand-batch INT    multi-pop expansion batch (default: 1). Pop N candidates per iteration\n"
        "                            to merge small remote batches before submitting.\n"
        "  -h, --help            show this help\n",
        prog);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Broadcast job-unique ID for shm naming (rank 0 PID)
    if (world_rank == 0) g_job_id = static_cast<uint32_t>(getpid());
    MPI_Bcast(&g_job_id, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // --- Parse arguments ---
    std::string dataset_path;
    std::string save_index_path;
    std::string load_index_path;
    uint32_t M = 16;
    uint32_t ef_construction = 200;
    uint32_t ef_search = 64;
    bool ef_search_from_cli = false;
    int32_t  K = 10;
    int      num_threads = 24;
    int      num_rounds = 5;
    int      num_shards_arg = 0; // 0 = auto-detect from world_size
    bool     profile_mode = false;
    bool     use_dot_norm = false;
    uint32_t expand_batch = 1;   // default: 1 = single-candidate expansion. Set >1 for multi-pop.
    ShardBuildOptions shard_build_opts;
    PushdownConfig pushdown_cfg;
    std::string dataset_format_str = "auto";
    std::string bin_base_path, bin_query_path, bin_gt_path;

    static struct option long_opts[] = {
        {"dataset",         required_argument, nullptr, 'd'},
        {"M",               required_argument, nullptr, 'M'},
        {"ef-construction", required_argument, nullptr, 'c'},
        {"ef-search",       required_argument, nullptr, 'e'},
        {"topk",            required_argument, nullptr, 'k'},
        {"threads",         required_argument, nullptr, 't'},
        {"rounds",          required_argument, nullptr, 'r'},
        {"shards",          required_argument, nullptr, 's'},
        {"profile",         no_argument,       nullptr, 'p'},
        {"search-threads",  required_argument, nullptr, 1002},
        {"service-threads", required_argument, nullptr, 1003},
        {"save-index",      required_argument, nullptr, 1005},
        {"load-index",      required_argument, nullptr, 1006},
        {"cluster",         no_argument,       nullptr, 1007},
        {"dataset-format",  required_argument, nullptr, 1015},
        {"base-bin",        required_argument, nullptr, 1016},
        {"query-bin",       required_argument, nullptr, 1017},
        {"gt-bin",          required_argument, nullptr, 1018},
        {"use-dot-norm",    no_argument,       nullptr, 1019},
        {"expand-batch",    required_argument, nullptr, 1022},
        {"help",            no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:M:c:e:k:t:r:s:ph", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': dataset_path = optarg; break;
            case 'M': M = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 'c': ef_construction = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 'e': ef_search = static_cast<uint32_t>(std::stoi(optarg)); ef_search_from_cli = true; break;
            case 'k': K = std::stoi(optarg); break;
            case 't': num_threads = std::stoi(optarg); break;
            case 'r': num_rounds = std::stoi(optarg); break;
            case 's': num_shards_arg = std::stoi(optarg); break;
            case 'p': profile_mode = true; break;
            case 1002: pushdown_cfg.search_threads = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 1003: pushdown_cfg.service_threads = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 1005: save_index_path = optarg; break;
            case 1006: load_index_path = optarg; break;
            case 1007: shard_build_opts.cluster_partition = true; break;
            case 1015: dataset_format_str = optarg; break;
            case 1016: bin_base_path = optarg; break;
            case 1017: bin_query_path = optarg; break;
            case 1018: bin_gt_path = optarg; break;
            case 1019: use_dot_norm = true; break;
            case 1022: {
                int val = std::stoi(optarg);
                if (val < 1 || val > 64) {
                    if (world_rank == 0)
                        fprintf(stderr, "Error: --expand-batch must be in [1, 64], got %d\n", val);
                    MPI_Finalize(); return 1;
                }
                expand_batch = static_cast<uint32_t>(val);
                break;
            }
            case 'h': print_usage(argv[0]); MPI_Finalize(); return 0;
            default:  print_usage(argv[0]); MPI_Finalize(); return 1;
        }
    }

    if (dataset_path.empty()) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: --dataset is required (provides queries and ground truth)\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Resolve dataset format
    DatasetFormat ds_format;
    bool has_bin_overrides = !bin_base_path.empty() || !bin_query_path.empty() || !bin_gt_path.empty();
    if (dataset_format_str == "auto") {
        if (has_bin_overrides) {
            // Override paths imply bin format
            ds_format = DatasetFormat::BIN;
        } else {
            ds_format = detect_dataset_format(dataset_path);
            if (ds_format == DatasetFormat::UNKNOWN) {
                // Default to HDF5 for backward compatibility
                ds_format = DatasetFormat::HDF5;
            }
        }
    } else if (dataset_format_str == "hdf5") {
        if (has_bin_overrides && world_rank == 0) {
            fprintf(stderr, "Warning: --base-bin/--query-bin/--gt-bin ignored with --dataset-format=hdf5\n");
        }
        ds_format = DatasetFormat::HDF5;
    } else if (dataset_format_str == "bin") {
        ds_format = DatasetFormat::BIN;
    } else {
        if (world_rank == 0) {
            fprintf(stderr, "Error: --dataset-format must be 'auto', 'hdf5', or 'bin'\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (world_rank == 0) {
        printf("[config] dataset format: %s\n",
               ds_format == DatasetFormat::BIN ? "bin" : "hdf5");
    }

    if (!load_index_path.empty() && shard_build_opts.cluster_partition && world_rank == 0) {
        printf("[config] --cluster is ignored when --load-index is used\n");
    }

    // Hardcoded half-noncache mode
    g_ubsmem_flags = UBSM_FLAG_ONLY_IMPORT_NONCACHE | UBSM_FLAG_WR_DELAY_COMP;
    g_channel_flags = UBSM_FLAG_ONLY_IMPORT_NONCACHE | UBSM_FLAG_WR_DELAY_COMP;

    if (world_rank == 0) {
        printf("[ubs-mem] shm-mode=half-noncache, index_flags=0x%lx, channel_flags=0x%lx\n",
               (unsigned long)g_ubsmem_flags, (unsigned long)g_channel_flags);
    }

    // Determine number of shards (default: one per MPI rank)
    uint32_t num_shards = (num_shards_arg > 0)
        ? static_cast<uint32_t>(num_shards_arg)
        : static_cast<uint32_t>(world_size);

    if (num_shards != static_cast<uint32_t>(world_size)) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: num_shards (%u) must equal world_size (%d). "
                    "One rank per shard is required.\n", num_shards, world_size);
        }
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) {
        printf("[config] %d MPI processes, %u shards\n", world_size, num_shards);
    }

    // Honor --threads explicitly; CPU/NUMA binding is controlled by launcher flags.
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    if (world_rank == 0) {
        printf("[config] Threads per process: %d\n", num_threads);
    }

    // Initialize ubs-mem SDK (after all validation, before any shm operations)
    {
        ubsmem_options_t opts{};
        int ubsm_ret = ubsmem_init_attributes(&opts);
        if (ubsm_ret != UBSM_OK) {
            fprintf(stderr, "ubsmem_init_attributes failed: %d\n", ubsm_ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Set log level to CRITICAL(4) before initialize to suppress init-phase logs
        ubsmem_set_logger_level(4);
        ubsm_ret = ubsmem_initialize(&opts);
        if (ubsm_ret != UBSM_OK) {
            fprintf(stderr, "ubsmem_initialize failed: %d\n", ubsm_ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Initialize provider for NUMA-local allocation
    {
        memset(&g_provider, 0, sizeof(g_provider));
        if (gethostname(g_provider.host_name, sizeof(g_provider.host_name) - 1) != 0) {
            fprintf(stderr, "[rank %d] gethostname failed: %s\n", world_rank, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        g_provider.socket_id = UINT32_MAX;  // auto-detect
        g_provider.port_id   = UINT32_MAX;  // auto-detect
#ifdef __linux__
        int cpu = sched_getcpu();
        if (cpu < 0) {
            fprintf(stderr, "[rank %d] sched_getcpu failed: %s\n", world_rank, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int numa = numa_node_of_cpu(cpu);
        if (numa < 0) {
            fprintf(stderr, "[rank %d] numa_node_of_cpu(%d) failed\n", world_rank, cpu);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        g_provider.numa_id = static_cast<uint32_t>(numa);
#else
        #error "NUMA-local allocation requires Linux"
#endif
        printf("[rank %d] ubs-mem provider: host=%s, numa_id=%u\n",
               world_rank, g_provider.host_name, g_provider.numa_id);
    }

    // ========================================
    // Route decision: direct-load vs legacy MPI distribute
    // In direct-load, each rank reads its own shard file into shm,
    // bypassing rank-0 bottleneck and eliminating double copy.
    // Requires all ranks to have read access to shard files.
    // ========================================
    uint32_t my_shard = static_cast<uint32_t>(world_rank);
    bool use_direct_load = false;
    if (!load_index_path.empty()) {
        std::string my_path = index_file_for_shard(load_index_path, my_shard);
        int local_ok = (access(my_path.c_str(), R_OK) == 0) ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        use_direct_load = (all_ok != 0);
        if (world_rank == 0) {
            printf("[init] load-index mode: direct-load=%s%s\n",
                   use_direct_load ? "ON" : "OFF",
                   use_direct_load ? "" : " (fallback: not all ranks can access shard files)");
        }
    }

    // ========================================
    // Phase 1: Load dataset + Build + Shard (rank 0)
    //          OR load pre-built index from disk
    // ========================================
    std::vector<std::vector<char>> shard_bufs;
    std::vector<uint32_t> new_to_old_idmap; // new gid -> original gid (for clustered build)
    bool has_idmap = false;
    std::vector<float> cluster_centroids;  // num_shards * dim floats (for query routing)
    bool has_centroids = false;
    AnnDataset dataset;

    if (world_rank == 0) {
        uint32_t peeked_dim = 0;  // set by direct-load peek for dim validation

        if (!load_index_path.empty() && !use_direct_load) {
            // --- Load pre-built index from disk (legacy MPI path) ---
            printf("[rank 0] Loading index from: %s (shards=%u)\n",
                   load_index_path.c_str(), num_shards);
            auto t0 = std::chrono::steady_clock::now();
            if (!load_shard_buffers(load_index_path, num_shards, shard_bufs)) {
                fprintf(stderr, "Error: failed to load index\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            auto t1 = std::chrono::steady_clock::now();
            double load_sec = std::chrono::duration<double>(t1 - t0).count();

            // Read back index params from shard 0 header
            const auto* sb0 = reinterpret_cast<const SuperBlockV1*>(shard_bufs[0].data());
            M = sb0->M;
            if (!ef_search_from_cli) ef_search = sb0->ef_search;

            uint64_t total_shm = 0;
            for (const auto& b : shard_bufs) total_shm += b.size();
            printf("[rank 0] Index loaded: %.2f sec, total: %.1f MB (%u shards, M=%u, ef=%u)\n",
                   load_sec, total_shm / (1024.0 * 1024.0), num_shards, M, ef_search);

            IdMapLoadStatus idmap_status = load_idmap_file(load_index_path, new_to_old_idmap);
            has_idmap = (idmap_status == IdMapLoadStatus::Loaded);
            if (has_idmap) {
                if (new_to_old_idmap.size() != static_cast<size_t>(sb0->ntotal)) {
                    fprintf(stderr, "Error: idmap size (%zu) != ntotal (%lu)\n",
                            new_to_old_idmap.size(), static_cast<unsigned long>(sb0->ntotal));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else if (idmap_status == IdMapLoadStatus::NotFound) {
                printf("  idmap not found, search results will use internal index ids\n");
            } else {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Load centroids for query routing
            uint32_t cent_shards = 0, cent_dim = 0;
            IdMapLoadStatus cent_status = load_centroids_file(
                load_index_path, cluster_centroids, cent_shards, cent_dim);
            has_centroids = (cent_status == IdMapLoadStatus::Loaded);
            if (has_centroids) {
                if (cent_shards != num_shards || cent_dim != sb0->dim) {
                    fprintf(stderr, "Error: centroids mismatch (shards %u!=%u or dim %u!=%u)\n",
                            cent_shards, num_shards, cent_dim, sb0->dim);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else if (cent_status == IdMapLoadStatus::NotFound) {
                printf("  centroids not found, queries will be distributed uniformly\n");
            } else {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else if (!load_index_path.empty() && use_direct_load) {
            // --- Direct-load: peek shard 0 header for M/ef_search/ntotal ---
            printf("[rank 0] Direct-load mode: peeking shard 0 header\n");
            std::string shard0_path = index_file_for_shard(load_index_path, 0);
            FILE* fp = fopen(shard0_path.c_str(), "rb");
            if (!fp) {
                fprintf(stderr, "Error: cannot open shard 0: %s\n", shard0_path.c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            SuperBlockV1 sb0_header{};
            if (fread(&sb0_header, sizeof(sb0_header), 1, fp) != 1) {
                fclose(fp);
                fprintf(stderr, "Error: cannot read shard 0 header\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fclose(fp);

            if (sb0_header.magic != SHM_MAGIC) {
                fprintf(stderr, "Error: shard 0 bad magic\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            M = sb0_header.M;
            if (!ef_search_from_cli) ef_search = sb0_header.ef_search;
            peeked_dim = sb0_header.dim;
            uint64_t peeked_ntotal = sb0_header.ntotal;

            printf("[rank 0] Direct-load peek: ntotal=%lu, dim=%u, M=%u, ef=%u\n",
                   (unsigned long)peeked_ntotal, peeked_dim, M, ef_search);

            // Load idmap (same as legacy path)
            IdMapLoadStatus idmap_status = load_idmap_file(load_index_path, new_to_old_idmap);
            has_idmap = (idmap_status == IdMapLoadStatus::Loaded);
            if (has_idmap) {
                if (new_to_old_idmap.size() != static_cast<size_t>(peeked_ntotal)) {
                    fprintf(stderr, "Error: idmap size (%zu) != ntotal (%lu)\n",
                            new_to_old_idmap.size(), (unsigned long)peeked_ntotal);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else if (idmap_status == IdMapLoadStatus::NotFound) {
                printf("  idmap not found, search results will use internal index ids\n");
            } else {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Load centroids for query routing
            uint32_t cent_shards = 0, cent_dim = 0;
            IdMapLoadStatus cent_status = load_centroids_file(
                load_index_path, cluster_centroids, cent_shards, cent_dim);
            has_centroids = (cent_status == IdMapLoadStatus::Loaded);
            if (has_centroids) {
                if (cent_shards != num_shards || cent_dim != peeked_dim) {
                    fprintf(stderr, "Error: centroids mismatch (shards %u!=%u or dim %u!=%u)\n",
                            cent_shards, num_shards, cent_dim, peeked_dim);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else if (cent_status == IdMapLoadStatus::NotFound) {
                printf("  centroids not found, queries will be distributed uniformly\n");
            } else {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            // --- Build from dataset ---
            printf("[rank 0] Loading dataset: %s\n", dataset_path.c_str());
            dataset = (ds_format == DatasetFormat::BIN)
                      ? load_bin_dataset(dataset_path, false, bin_base_path, bin_query_path, bin_gt_path)
                      : load_hdf5_dataset(dataset_path);
            printf("[rank 0] Dataset loaded: N=%lu, nq=%lu, dim=%u, gt_k=%u\n",
                   (unsigned long)dataset.n_train, (unsigned long)dataset.n_test,
                   dataset.dim, dataset.k_gt);

            printf("[rank 0] Building HNSW: M=%u, ef_construction=%u, shards=%u\n",
                   M, ef_construction, num_shards);
            if (shard_build_opts.cluster_partition) {
                printf("[rank 0] Cluster partition: ON\n");
            } else {
                printf("[rank 0] Cluster partition: OFF (legacy range split)\n");
            }
            auto t0 = std::chrono::steady_clock::now();
            shard_build_opts.post_add_callback = [&]() {
                size_t freed_bytes = dataset.train.size() * sizeof(float);
                dataset.train.clear();
                dataset.train.shrink_to_fit();
                printf("[rank 0] Released dataset.train after FAISS add (~%.1f GiB)\n",
                       freed_bytes / (1024.0 * 1024.0 * 1024.0));
            };

            if (!save_index_path.empty()) {
                // Streaming mode: extract one shard at a time, write to disk
                // immediately, free buffer.  Peak memory = FAISS index + 1 shard.
                printf("[rank 0] Using streaming extraction (save_path: %s)\n",
                       save_index_path.c_str());
                auto shard_sizes = FaissExtractor::build_and_extract_sharded_streaming(
                    dataset.train.data(), dataset.n_train, dataset.dim,
                    num_shards, save_index_path,
                    M, ef_construction, ef_search,
                    shard_build_opts,
                    shard_build_opts.cluster_partition ? &new_to_old_idmap : nullptr,
                    shard_build_opts.cluster_partition ? &cluster_centroids : nullptr);
                auto t1 = std::chrono::steady_clock::now();
                double build_sec = std::chrono::duration<double>(t1 - t0).count();
                uint64_t total_shm = 0;
                for (auto sz : shard_sizes) total_shm += sz;
                printf("[rank 0] Build+save complete (streaming): %.2f sec, total: %.1f MB (%u shards)\n",
                       build_sec, total_shm / (1024.0 * 1024.0), num_shards);

                has_idmap = shard_build_opts.cluster_partition;
                has_centroids = shard_build_opts.cluster_partition
                               && !cluster_centroids.empty();
                if (has_idmap && new_to_old_idmap.size() != static_cast<size_t>(dataset.n_train)) {
                    fprintf(stderr, "Error: generated idmap size mismatch (%zu != %lu)\n",
                            new_to_old_idmap.size(), static_cast<unsigned long>(dataset.n_train));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (has_idmap) {
                    if (!save_idmap_file(save_index_path, new_to_old_idmap)) {
                        fprintf(stderr, "Warning: failed to save idmap (continuing)\n");
                    }
                }
                if (has_centroids) {
                    if (!save_centroids_file(save_index_path, cluster_centroids,
                                             num_shards, dataset.dim)) {
                        fprintf(stderr, "Warning: failed to save centroids (continuing)\n");
                    }
                }

                // Load shard buffers back for MPI distribution (one at a time
                // would be ideal, but current MPI distribute expects shard_bufs).
                if (!load_shard_buffers(save_index_path, num_shards, shard_bufs)) {
                    fprintf(stderr, "Error: failed to reload shard buffers after streaming save\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            } else {
                // In-memory mode: all shard buffers held simultaneously.
                shard_bufs = FaissExtractor::build_and_extract_sharded(
                    dataset.train.data(), dataset.n_train, dataset.dim,
                    num_shards, M, ef_construction, ef_search,
                    shard_build_opts,
                    shard_build_opts.cluster_partition ? &new_to_old_idmap : nullptr,
                    shard_build_opts.cluster_partition ? &cluster_centroids : nullptr);
                auto t1 = std::chrono::steady_clock::now();
                double build_sec = std::chrono::duration<double>(t1 - t0).count();
                uint64_t total_shm = 0;
                for (const auto& b : shard_bufs) total_shm += b.size();
                printf("[rank 0] Build complete: %.2f sec, total shm: %.1f MB (%u shards)\n",
                       build_sec, total_shm / (1024.0 * 1024.0), num_shards);

                has_idmap = shard_build_opts.cluster_partition;
                has_centroids = shard_build_opts.cluster_partition
                               && !cluster_centroids.empty();
                if (has_idmap && new_to_old_idmap.size() != static_cast<size_t>(dataset.n_train)) {
                    fprintf(stderr, "Error: generated idmap size mismatch (%zu != %lu)\n",
                            new_to_old_idmap.size(), static_cast<unsigned long>(dataset.n_train));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                // --- Save index to disk if requested ---
                if (!save_index_path.empty()) {
                    printf("[rank 0] Saving index to: %s\n", save_index_path.c_str());
                    if (!save_shard_buffers(save_index_path, shard_bufs)) {
                        fprintf(stderr, "Warning: failed to save index (continuing)\n");
                    }
                    if (has_idmap) {
                        if (!save_idmap_file(save_index_path, new_to_old_idmap)) {
                            fprintf(stderr, "Warning: failed to save idmap (continuing)\n");
                        }
                    }
                    if (has_centroids) {
                        if (!save_centroids_file(save_index_path, cluster_centroids,
                                                 num_shards, dataset.dim)) {
                            fprintf(stderr, "Warning: failed to save centroids (continuing)\n");
                        }
                    }
                }
            }
        }

        // Load dataset for queries/ground-truth if not already loaded.
        // In load-index mode, skip reading /train vectors (eval_only=true)
        // to avoid ~51 GiB of unnecessary I/O for large datasets.
        if (dataset.n_test == 0) {
            auto t_hdf5_start = std::chrono::steady_clock::now();
            bool eval_only = !load_index_path.empty();
            dataset = (ds_format == DatasetFormat::BIN)
                      ? load_bin_dataset(dataset_path, eval_only, bin_base_path, bin_query_path, bin_gt_path)
                      : load_hdf5_dataset(dataset_path, eval_only);
            auto t_hdf5_end = std::chrono::steady_clock::now();
            double hdf5_sec = std::chrono::duration<double>(t_hdf5_end - t_hdf5_start).count();
            printf("[rank 0] Dataset load: %.2f sec%s\n",
                   hdf5_sec, eval_only ? " (eval-only, base skipped)" : "");
        }

        // Validate index dim matches dataset dim (prevents OOB reads in search)
        if (!load_index_path.empty() && dataset.n_test > 0) {
            uint32_t idx_dim = use_direct_load
                ? peeked_dim
                : reinterpret_cast<const SuperBlockV1*>(shard_bufs[0].data())->dim;
            if (idx_dim != dataset.dim) {
                fprintf(stderr, "Error: index dim (%u) != dataset dim (%u)\n",
                        idx_dim, dataset.dim);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Release training vectors — already freed via post_add_callback in build
    // path, but clear again defensively for the load-index path where the
    // callback is never set.
    if (world_rank == 0) {
        dataset.train.clear();
        dataset.train.shrink_to_fit();
    }

    // ========================================
    // Phase 2: Distribute shard buffers (point-to-point)
    //
    // Old approach: Bcast ALL shards to ALL ranks → peak memory = N × total_index.
    // New approach: rank 0 sends each rank ONLY its own shard via MPI_Send,
    // then broadcasts the small shard_sizes array so every rank can open
    // remote shards via shm later.  Peak memory per rank ≈ 1 shard buffer.
    // ========================================
    MPI_Bcast(&num_shards, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // Sync ef_search across all ranks (rank 0 may have overridden from index header or CLI)
    MPI_Bcast(&ef_search, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // my_shard already defined above (route decision block)

    // Broadcast shard sizes so all ranks know how large each shm segment is.
    std::vector<uint64_t> shard_sizes(num_shards, 0);
    std::vector<char> my_shard_buf;  // only used in legacy (non-direct-load) path

    if (!use_direct_load) {
    // --- Legacy Phase 2: rank 0 distributes shard buffers via MPI ---
    if (world_rank == 0) {
        for (uint32_t s = 0; s < num_shards; s++) {
            shard_sizes[s] = shard_bufs[s].size();
        }
    }
    MPI_Bcast(shard_sizes.data(), static_cast<int>(num_shards),
              MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Point-to-point distribution: rank 0 sends each shard to its owner rank.
    // Each non-zero rank receives only its own shard buffer.

    auto t_dist_start = std::chrono::steady_clock::now();

    if (world_rank == 0) {
        printf("[rank 0] Distributing %u shards to %d processes (point-to-point)...\n",
               num_shards, world_size);

        // Keep shard 0 for ourselves
        my_shard_buf = std::move(shard_bufs[0]);

        // Send each other shard to its owner rank, then free immediately
        for (uint32_t s = 1; s < num_shards; s++) {
            uint64_t offset = 0;
            uint64_t total = shard_bufs[s].size();
            while (offset < total) {
                int chunk = static_cast<int>(
                    std::min(static_cast<uint64_t>(MPI_CHUNK_BYTES), total - offset));
                MPI_Send(shard_bufs[s].data() + offset, chunk, MPI_BYTE,
                         static_cast<int>(s), static_cast<int>(s),
                         MPI_COMM_WORLD);
                offset += chunk;
            }
            // Free this shard buffer immediately after sending
            shard_bufs[s].clear();
            shard_bufs[s].shrink_to_fit();
        }
        shard_bufs.clear();
        shard_bufs.shrink_to_fit();
    } else if (my_shard < num_shards) {
        // Receive only our own shard from rank 0
        my_shard_buf.resize(static_cast<size_t>(shard_sizes[my_shard]));
        uint64_t offset = 0;
        uint64_t total = shard_sizes[my_shard];
        while (offset < total) {
            int chunk = static_cast<int>(
                std::min(static_cast<uint64_t>(MPI_CHUNK_BYTES), total - offset));
            MPI_Recv(my_shard_buf.data() + offset, chunk, MPI_BYTE,
                     0, static_cast<int>(my_shard), MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            offset += chunk;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        auto t_dist_end = std::chrono::steady_clock::now();
        double dist_sec = std::chrono::duration<double>(t_dist_end - t_dist_start).count();
        printf("[rank %d] Phase 2 (MPI distribute): %.2f sec\n", world_rank, dist_sec);
    }
    } // end legacy Phase 2 (!use_direct_load)

    // Broadcast optional id remap (new gid -> original gid) used for recall/output.
    uint32_t idmap_flag = (world_rank == 0 && has_idmap) ? 1u : 0u;
    MPI_Bcast(&idmap_flag, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    has_idmap = (idmap_flag != 0u);
    if (has_idmap) {
        uint64_t idmap_count = (world_rank == 0)
            ? static_cast<uint64_t>(new_to_old_idmap.size()) : 0;
        MPI_Bcast(&idmap_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        if (world_rank != 0) {
            new_to_old_idmap.resize(static_cast<size_t>(idmap_count));
        }
        if (idmap_count > 0) {
            chunked_bcast_bytes(new_to_old_idmap.data(),
                                idmap_count * sizeof(uint32_t),
                                0, MPI_COMM_WORLD);
        }
    }

    // ========================================
    // Phase 3: Each rank creates its own shard shm (1:1 rank-to-shard).
    // Then all processes open all shards read-only.
    // ========================================

    auto t_shm_start = std::chrono::steady_clock::now();

    // Track owner's writable mapping for cleanup (direct-load unmaps inside the function)
    void* owner_write_ptr = nullptr;
    size_t owner_write_alloc = 0;

    // For build path: extract vectors from shard buffer before SHM compaction discards them.
    // These will be passed to load_local_vectors() later.
    std::vector<float> build_path_vectors;

    if (!use_direct_load) {
    // --- Legacy Phase 3: copy buffer into shm ---
    if (my_shard < num_shards) {
        // Mark as ACTIVE and create shm
        auto* sb = reinterpret_cast<SuperBlockV1*>(my_shard_buf.data());
        sb->state = static_cast<uint32_t>(ShmState::ACTIVE);
        sb->header_crc64 = compute_header_crc64(*sb);

        // Extract vectors from buffer BEFORE create_shm (which discards them)
        // Only needed for pushdown path (load_local_vectors).
        const bool keep_vectors = (num_shards == 1);
        if (!keep_vectors) {
            uint64_t vec_off = sb->blocks[BLK_VECTORS].off;
            uint64_t vec_bytes = sb->blocks[BLK_VECTORS].bytes;
            uint64_t vec_count = vec_bytes / sizeof(float);
            build_path_vectors.resize(vec_count);
            std::memcpy(build_path_vectors.data(),
                        my_shard_buf.data() + vec_off, vec_bytes);
        }

        std::string name = shm_name_for_shard(my_shard);
        size_t shm_alloc_size = 0;
        owner_write_ptr = create_shm(name, my_shard_buf, shm_alloc_size, keep_vectors);
        owner_write_alloc = shm_alloc_size;
        shard_sizes[my_shard] = shm_alloc_size;
        printf("[rank %d] Created SHM shard %u: %s (%.1f MB%s)\n",
               world_rank, my_shard, name.c_str(),
               shm_alloc_size / (1024.0 * 1024.0),
               keep_vectors ? "" : " graph-only");

        // Free the shard buffer — data is now in shm
        my_shard_buf.clear();
        my_shard_buf.shrink_to_fit();
    }
    // Sync SHM sizes across all ranks
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  shard_sizes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    } else {
    // --- Direct-load Phase 2+3: each rank reads its own shard file into shm ---
    const bool keep_vectors = (num_shards == 1);
    std::string shm_name = shm_name_for_shard(my_shard);
    LocalShardMeta meta = load_local_shard_direct_to_shm(
            load_index_path, my_shard, shm_name, keep_vectors);

    // All-or-nothing: if any rank fails, abort collectively
    int local_fail = meta.ok ? 0 : 1;
    int any_fail = 0;
    MPI_Allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_fail) {
        if (!meta.ok) {
            fprintf(stderr, "[rank %d] direct-load failed: %s\n",
                    world_rank, meta.error.c_str());
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allgather graph-only SHM sizes so all ranks can open remote shards
    uint64_t my_size = meta.shm_alloc_size;
    MPI_Allgather(&my_size, 1, MPI_UINT64_T,
                  shard_sizes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    // Cross-shard consistency validation (mirrors legacy load_shard_buffers checks)
    ShardMetaWire my_wire{};
    my_wire.ntotal         = meta.ntotal;
    my_wire.dim            = meta.dim;
    my_wire.M              = meta.M;
    my_wire.max_level      = meta.max_level;
    my_wire.entry_point    = meta.entry_point;
    my_wire.num_shards     = meta.num_shards;
    my_wire.shard_id       = meta.shard_id;
    my_wire.ntotal_local   = meta.ntotal_local;
    my_wire.global_id_begin = meta.global_id_begin;

    std::vector<ShardMetaWire> all_meta(num_shards);
    MPI_Allgather(&my_wire, sizeof(ShardMetaWire), MPI_BYTE,
                  all_meta.data(), sizeof(ShardMetaWire), MPI_BYTE,
                  MPI_COMM_WORLD);

    // Validate on every rank (cheap integer comparisons)
    {
        const auto& s0 = all_meta[0];
        uint64_t sum_ntotal_local = 0;
        bool consistent = true;
        for (uint32_t s = 0; s < num_shards; s++) {
            const auto& sm = all_meta[s];
            if (sm.shard_id != s) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u has shard_id=%u\n",
                        world_rank, s, sm.shard_id);
                consistent = false;
            }
            if (sm.num_shards != num_shards) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u has num_shards=%u (expected %u)\n",
                        world_rank, s, sm.num_shards, num_shards);
                consistent = false;
            }
            if (sm.ntotal != s0.ntotal || sm.dim != s0.dim ||
                sm.M != s0.M || sm.max_level != s0.max_level ||
                sm.entry_point != s0.entry_point) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u param mismatch with shard 0\n",
                        world_rank, s);
                consistent = false;
            }
            if (sm.global_id_begin != sum_ntotal_local) {
                fprintf(stderr, "[rank %d] cross-shard: shard %u global_id_begin=%lu, expected %lu\n",
                        world_rank, s, (unsigned long)sm.global_id_begin,
                        (unsigned long)sum_ntotal_local);
                consistent = false;
            }
            sum_ntotal_local += sm.ntotal_local;
        }
        if (sum_ntotal_local != s0.ntotal) {
            fprintf(stderr, "[rank %d] cross-shard: sum(ntotal_local)=%lu != ntotal=%lu\n",
                    world_rank, (unsigned long)sum_ntotal_local, (unsigned long)s0.ntotal);
            consistent = false;
        }
        if (!consistent) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (world_rank == 0) {
            printf("  Cross-shard consistency: OK (%u shards, ntotal=%lu, dim=%u, M=%u)\n",
                   num_shards, (unsigned long)s0.ntotal, s0.dim, s0.M);
        }
    }

    printf("[rank %d] Direct-load shard %u: %.1f MB\n",
           world_rank, my_shard, meta.file_size / (1024.0 * 1024.0));
    } // end direct-load

    MPI_Barrier(MPI_COMM_WORLD);

    // Release owner's writable mapping (legacy path only; direct-load unmaps in-function)
    if (owner_write_ptr) {
        ubsmem_shmem_unmap(owner_write_ptr, owner_write_alloc);
        owner_write_ptr = nullptr;
    }

    // All processes open all shards read-only
    std::vector<std::pair<const void*, uint64_t>> shard_ptrs(num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string name = shm_name_for_shard(s);
        shard_ptrs[s] = {open_shm_readonly(name, shard_sizes[s]), shard_sizes[s]};
    }

    {
        auto t_shm_end = std::chrono::steady_clock::now();
        double shm_sec = std::chrono::duration<double>(t_shm_end - t_shm_start).count();
        printf("[rank %d] Phase 3 (SHM %s+open): %.2f sec\n",
               world_rank, use_direct_load ? "direct-load" : "create", shm_sec);
    }

    // ========================================
    // Phase 4: Distribute queries + Search
    // ========================================
    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t nq = 0;
    uint64_t n_train = 0;
    uint32_t dim = 0;
    uint32_t gt_k = 0;
    if (world_rank == 0) {
        nq = dataset.n_test;
        n_train = dataset.n_train;
        dim = dataset.dim;
        gt_k = dataset.k_gt;
    }
    MPI_Bcast(&nq, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_train, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gt_k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Broadcast cluster centroids for query routing.
    uint32_t cent_flag = (world_rank == 0 && has_centroids) ? 1u : 0u;
    MPI_Bcast(&cent_flag, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    has_centroids = (cent_flag != 0u);
    if (has_centroids) {
        uint64_t cent_count = static_cast<uint64_t>(num_shards) * dim;
        if (world_rank != 0) {
            cluster_centroids.resize(static_cast<size_t>(cent_count));
        }
        if (cent_count > 0) {
            chunked_bcast_bytes(cluster_centroids.data(),
                                cent_count * sizeof(float),
                                0, MPI_COMM_WORLD);
        }
    }

    if (has_idmap && new_to_old_idmap.size() != static_cast<size_t>(n_train)) {
        fprintf(stderr, "Error: rank %d idmap size mismatch (%zu != %lu)\n",
                world_rank, new_to_old_idmap.size(), static_cast<unsigned long>(n_train));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Broadcast all queries
    std::vector<float> all_queries(nq * dim);
    if (world_rank == 0) {
        std::memcpy(all_queries.data(), dataset.test.data(), nq * dim * sizeof(float));
    }
    {
        uint64_t total = nq * dim * sizeof(float);
        uint64_t offset = 0;
        while (offset < total) {
            int chunk = static_cast<int>(
                std::min(static_cast<uint64_t>(MPI_CHUNK_BYTES), total - offset));
            MPI_Bcast(reinterpret_cast<char*>(all_queries.data()) + offset,
                      chunk, MPI_BYTE, 0, MPI_COMM_WORLD);
            offset += chunk;
        }
    }

    // Broadcast ground truth
    std::vector<int32_t> all_gt(nq * gt_k);
    if (world_rank == 0) {
        std::memcpy(all_gt.data(), dataset.neighbors.data(), nq * gt_k * sizeof(int32_t));
    }
    {
        uint64_t total = nq * gt_k * sizeof(int32_t);
        uint64_t offset = 0;
        while (offset < total) {
            int chunk = static_cast<int>(
                std::min(static_cast<uint64_t>(MPI_CHUNK_BYTES), total - offset));
            MPI_Bcast(reinterpret_cast<char*>(all_gt.data()) + offset,
                      chunk, MPI_BYTE, 0, MPI_COMM_WORLD);
            offset += chunk;
        }
    }

    // Route queries to ranks by nearest cluster centroid (or uniform split).
    std::vector<uint32_t> my_query_indices;
    uint64_t my_nq = 0;

    if (has_centroids) {
        for (uint64_t q = 0; q < nq; q++) {
            const float* qvec = all_queries.data() + q * dim;
            float best_dist = std::numeric_limits<float>::max();
            uint32_t best_shard = 0;
            for (uint32_t s = 0; s < num_shards; s++) {
                const float* cvec = cluster_centroids.data() + static_cast<size_t>(s) * dim;
                float dist = 0.0f;
                for (uint32_t d = 0; d < dim; d++) {
                    float diff = qvec[d] - cvec[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_shard = s;
                }
            }
            if (best_shard == my_shard) {
                my_query_indices.push_back(static_cast<uint32_t>(q));
            }
        }
        my_nq = my_query_indices.size();
        printf("[cluster-route] rank %d: %lu queries (of %lu total)\n",
               world_rank, (unsigned long)my_nq, (unsigned long)nq);
    } else {
        uint64_t queries_per_worker = (nq + world_size - 1) / world_size;
        uint64_t q_start = static_cast<uint64_t>(world_rank) * queries_per_worker;
        uint64_t q_end = std::min(q_start + queries_per_worker, nq);
        my_nq = (q_start < nq) ? (q_end - q_start) : 0;
        my_query_indices.resize(static_cast<size_t>(my_nq));
        for (uint64_t i = 0; i < my_nq; i++) {
            my_query_indices[i] = static_cast<uint32_t>(q_start + i);
        }
    }

    // Build contiguous query/gt buffers for this rank's queries
    std::vector<float> my_queries_buf(static_cast<size_t>(my_nq) * dim);
    for (uint64_t i = 0; i < my_nq; i++) {
        std::memcpy(my_queries_buf.data() + i * dim,
                    all_queries.data() + static_cast<size_t>(my_query_indices[i]) * dim,
                    dim * sizeof(float));
    }
    std::vector<int32_t> my_gt(static_cast<size_t>(my_nq) * gt_k);
    for (uint64_t i = 0; i < my_nq; i++) {
        std::memcpy(my_gt.data() + i * gt_k,
                    all_gt.data() + static_cast<size_t>(my_query_indices[i]) * gt_k,
                    gt_k * sizeof(int32_t));
    }

    // Create multi-shard searcher
    // In load-index mode, rank 0 already validated all shards during
    // load_shard_buffers — skip redundant per-rank full CRC + structural walk.
    auto t_searcher_start = std::chrono::steady_clock::now();
    // Always skip validation: SHM was validated before loading.
    // In compact mode BLK_VECTORS is zeroed; full buffer is preserved for single-shard.
    // Full validation was already done on the original file/buffer before compaction.
    bool skip_val = true;
    ShmHnswSearcher searcher(shard_ptrs, skip_val);
    {
        auto t_searcher_end = std::chrono::steady_clock::now();
        double searcher_sec = std::chrono::duration<double>(t_searcher_end - t_searcher_start).count();
        printf("[rank %d] Searcher init: %.2f sec%s\n",
               world_rank, searcher_sec, skip_val ? " (validation skipped)" : "");
    }
    // Each rank owns exactly one shard (1:1 mapping)
    searcher.set_local_shard(my_shard);

    searcher.set_expand_batch(expand_batch);
    if (world_rank == 0 && expand_batch > 1) {
        printf("[rank 0] Multi-pop expansion: %u candidates per iteration\n", expand_batch);
    }

    // Load LOCAL shard's vector data directly to heap.
    // In full-pushdown mode, only local shard vectors are needed on the hot path.
    // Remote shard ShardView.vectors are set to nullptr (pushdown handles remote distances).
    // The entry_point vector is cached separately if EP is on a remote shard.
    // Must be called BEFORE pushdown service setup (service workers copy ShardView).
    if (num_shards > 1) {
        auto t_lv_start = std::chrono::steady_clock::now();

        // Before loading (which nulls remote pointers), cache EP vector if remote.
        const auto& local_sv = searcher.shards()[my_shard];
        int32_t ep = local_sv.sb->entry_point;
        uint64_t ep_gid = static_cast<uint64_t>(ep);
        bool ep_is_local = (ep_gid >= local_sv.global_id_begin &&
                            ep_gid < local_sv.global_id_begin + local_sv.ntotal_local);

        if (use_direct_load && !load_index_path.empty()) {
            // Direct-load path: each rank can access shard files directly
            std::string my_path = index_file_for_shard(load_index_path, my_shard);
            searcher.load_local_vectors(my_path);

            // If EP is remote, read its vector from the EP's shard file (MUST succeed)
            if (!ep_is_local) {
                uint32_t ep_shard = 0;
                for (uint32_t s = 0; s < num_shards; s++) {
                    const auto& sv = searcher.shards()[s];
                    if (ep_gid >= sv.global_id_begin &&
                        ep_gid < sv.global_id_begin + sv.ntotal_local) {
                        ep_shard = s;
                        break;
                    }
                }
                std::string ep_path = index_file_for_shard(load_index_path, ep_shard);
                FILE* ep_fp = fopen(ep_path.c_str(), "rb");
                if (!ep_fp) {
                    fprintf(stderr, "[rank %d] FATAL: cannot open EP shard file '%s': %s\n",
                            world_rank, ep_path.c_str(), strerror(errno));
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                SuperBlockV1 ep_sb{};
                if (fread(&ep_sb, sizeof(SuperBlockV1), 1, ep_fp) != 1) {
                    fclose(ep_fp);
                    fprintf(stderr, "[rank %d] FATAL: cannot read SuperBlock from EP shard '%s'\n",
                            world_rank, ep_path.c_str());
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                uint64_t ep_local_id = ep_gid - ep_sb.global_id_begin;
                uint64_t vec_file_off = ep_sb.blocks[BLK_VECTORS].off
                                        + ep_local_id * dim * sizeof(float);
                std::vector<float> ep_vec(dim);
                fseek(ep_fp, static_cast<long>(vec_file_off), SEEK_SET);
                if (fread(ep_vec.data(), sizeof(float), dim, ep_fp) != dim) {
                    fclose(ep_fp);
                    fprintf(stderr, "[rank %d] FATAL: short read on EP vector (gid %d) from '%s'\n",
                            world_rank, ep, ep_path.c_str());
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                fclose(ep_fp);
                searcher.set_ep_vector(ep_vec.data(), dim);
                printf("[rank %d] EP vector (gid %d) loaded from shard %u file\n",
                       world_rank, ep, ep_shard);
            }
        } else {
            // Build path: broadcast EP vector from owning rank to all ranks.
            uint32_t ep_owner_rank = UINT32_MAX;
            for (uint32_t s = 0; s < num_shards; s++) {
                const auto& sv = searcher.shards()[s];
                if (ep_gid >= sv.global_id_begin &&
                    ep_gid < sv.global_id_begin + sv.ntotal_local) {
                    ep_owner_rank = s;  // 1:1 rank-to-shard
                    break;
                }
            }
            assert(ep_owner_rank < num_shards);

            std::vector<float> ep_vec(dim);
            if (static_cast<uint32_t>(world_rank) == ep_owner_rank) {
                const auto& sv = searcher.shards()[ep_owner_rank];
                uint64_t ep_local_id = ep_gid - sv.global_id_begin;
                if (!build_path_vectors.empty()) {
                    // Graph-only compact: vectors were extracted before SHM creation
                    std::memcpy(ep_vec.data(),
                                build_path_vectors.data() + ep_local_id * dim,
                                dim * sizeof(float));
                } else {
                    // keep_vectors mode: read directly from SHM
                    std::memcpy(ep_vec.data(),
                                sv.vectors + ep_local_id * dim,
                                dim * sizeof(float));
                }
            }
            MPI_Bcast(ep_vec.data(), static_cast<int>(dim), MPI_FLOAT,
                       static_cast<int>(ep_owner_rank), MPI_COMM_WORLD);

            if (!ep_is_local) {
                searcher.set_ep_vector(ep_vec.data(), dim);
                printf("[rank %d] EP vector (gid %d) broadcast from rank %u\n",
                       world_rank, ep, ep_owner_rank);
            }

            if (!build_path_vectors.empty()) {
                // Graph-only compact: load from extracted buffer to heap
                searcher.load_local_vectors(build_path_vectors.data());
                build_path_vectors.clear();
                build_path_vectors.shrink_to_fit();
            } else {
                // keep_vectors mode (single-shard): vectors in SHM, copy local to heap
                const auto& sv = searcher.shards()[my_shard];
                searcher.load_local_vectors(sv.vectors);
            }
        }

        auto t_lv_end = std::chrono::steady_clock::now();
        double lv_sec = std::chrono::duration<double>(t_lv_end - t_lv_start).count();
        printf("[rank %d] Vector data loaded to local heap: %.2f sec\n",
               world_rank, lv_sec);
    } else {
        // Single-shard: vectors remain in SHM, no remote distance computation needed
        printf("[rank %d] Single-shard: vectors served directly from SHM\n", world_rank);

        // Free build_path_vectors if populated (legacy path extracted them before SHM creation)
        if (!build_path_vectors.empty()) {
            build_path_vectors.clear();
            build_path_vectors.shrink_to_fit();
        }
    }

    // Enable dot_norm distance optimization if requested
    if (use_dot_norm) {
        searcher.enable_dot_norm(true);
        if (world_rank == 0) {
            if (searcher.dot_norm_enabled()) {
                printf("[dot_norm] Enabled — using dot+norm distance kernel for local search\n");
            } else {
                printf("[dot_norm] WARNING: requested but not active (norms not precomputed)\n");
            }
        }
    }

    // ========================================
    // Pushdown setup: cross-process channel shm, proxies, service threads
    //
    // Topology: each rank creates channels for its search threads to reach
    // remote shards via ubs-mem shm. The owner rank of each shard maps those
    // channels and runs DistService to compute distances locally (near-data).
    // ========================================
    auto t_pushdown_start = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<DualDistService>> dual_dist_services;
    std::vector<std::vector<DualDistProxy>> dual_pushdown_proxies;  // [thread][shard]
    // Track channel shm names for cleanup
    std::vector<std::string> owned_channel_shm_names;
    // Track mapped pointers for requester-side channels (for unmap on cleanup)
    std::vector<std::pair<void*, size_t>> requester_channel_maps;
    // Track mapped pointers for service-side channels (for unmap on cleanup)
    std::vector<std::pair<void*, size_t>> service_channel_maps;

    // Pushdown uses 3 MPI collectives on MPI_COMM_WORLD (Bcast + 2 Barriers).
    // When world_size > num_shards, ranks with world_rank >= num_shards don't
    // do pushdown work but MUST participate in collectives to avoid deadlock.
    const bool pushdown_active = num_shards > 1;
    const bool pushdown_participant = pushdown_active &&
        static_cast<uint32_t>(world_rank) < num_shards;

    auto resolve_service_threads = [&](uint32_t configured) -> uint32_t {
        if (configured != 0) return configured;
        uint32_t cores = static_cast<uint32_t>(std::max(1, num_threads));
        uint32_t auto_svc = std::max<uint32_t>(1, cores / 4);
        uint32_t max_svc = std::max<uint32_t>(1, num_shards - 1);
        return std::min(auto_svc, max_svc);
    };

    // Total service-side cores
    auto resolve_service_cores = [&](uint32_t svc_threads) -> uint32_t {
        return svc_threads;
    };

    // Derive search threads from total budget minus service cores
    auto resolve_search_threads = [&](uint32_t svc_threads) -> uint32_t {
        uint32_t srch = pushdown_cfg.search_threads;
        if (srch == 0) {
            srch = static_cast<uint32_t>(num_threads) - resolve_service_cores(svc_threads);
        }
        return srch;
    };

    // run_gen must be declared outside the if-block so all ranks can participate
    // in MPI_Bcast even if they don't use the value.
    uint64_t run_gen = 0;

    if (pushdown_participant) {

        // Thread budget validation.
        // New topology: each rank runs ONE DistService for its own shard,
        // with svc_threads worker threads. Total service threads = svc_threads.
        // so total service-side cores = svc_threads * 2.
        uint32_t svc_threads = resolve_service_threads(pushdown_cfg.service_threads);
        const uint32_t svc_total_cores = resolve_service_cores(svc_threads);
        uint32_t srch_threads = pushdown_cfg.search_threads;

        if (svc_total_cores >= static_cast<uint32_t>(num_threads)) {
            fprintf(stderr, "FATAL: service cores (%u) >= num_threads (%d). "
                    "No threads left for search.\n",
                    svc_total_cores, num_threads);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (srch_threads == 0) {
            srch_threads = static_cast<uint32_t>(num_threads) - svc_total_cores;
        } else {
            if (srch_threads + svc_total_cores > static_cast<uint32_t>(num_threads)) {
                fprintf(stderr, "FATAL: search_threads (%u) + service cores (%u) "
                        "> num_threads (%d). Reduce --search-threads or --service-threads.\n",
                        srch_threads, svc_total_cores, num_threads);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        omp_set_num_threads(static_cast<int>(srch_threads));

        if (num_shards > 32) {
            fprintf(stderr, "FATAL: num_shards (%u) > 32. "
                    "Pushdown pending_mask is uint32_t bitmask.\n", num_shards);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const auto* sb0 = static_cast<const SuperBlockV1*>(shard_ptrs[0].first);
        const int32_t* cum_nn0 = cum_nneighbor_ptr(shard_ptrs[0].first, *sb0);
        uint32_t max_batch = static_cast<uint32_t>(cum_nn0[1]);

        // Verify all shards have the same layer-0 neighbor count
        for (uint32_t s = 1; s < num_shards; s++) {
            const auto* sb_s = static_cast<const SuperBlockV1*>(shard_ptrs[s].first);
            const int32_t* cum_s = cum_nneighbor_ptr(shard_ptrs[s].first, *sb_s);
            if (static_cast<uint32_t>(cum_s[1]) != max_batch) {
                fprintf(stderr, "FATAL: shard %u has layer-0 neighbors=%d, "
                        "shard 0 has %u. All shards must match.\n",
                        s, cum_s[1], max_batch);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        // Scale channel capacity for multi-pop expansion (after validation)
        {
            uint64_t scaled = static_cast<uint64_t>(max_batch) * expand_batch;
            if (scaled > UINT32_MAX) {
                fprintf(stderr, "FATAL: max_batch * expand_batch overflows uint32: %u * %u = %llu\n",
                        max_batch, expand_batch, (unsigned long long)scaled);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            max_batch = static_cast<uint32_t>(scaled);
        }

        // Rank 0 generates run_gen; broadcast happens below outside the if-block
        if (world_rank == 0) {
            run_gen = static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
        }
    }

    // --- Collective 1: Broadcast run_gen so all ranks agree ---
    // All MPI_COMM_WORLD members participate (including non-pushdown ranks).
    if (pushdown_active) {
        MPI_Bcast(&run_gen, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

    // --- Collective 1.5: Validate srch_threads consistency across all ranks ---
    // Must be outside pushdown_participant guard: MPI_COMM_WORLD requires all ranks.
    uint32_t srch_threads_validated = 0;
    if (pushdown_active) {
        uint32_t svc_tmp = resolve_service_threads(pushdown_cfg.service_threads);
        uint32_t srch_tmp = resolve_search_threads(svc_tmp);
        srch_threads_validated = srch_tmp;

        std::vector<uint32_t> all_srch(world_size);
        MPI_Allgather(&srch_tmp, 1, MPI_UINT32_T,
                      all_srch.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD);
        // Only validate among participant ranks (rank < num_shards)
        for (uint32_t r = 0; r < num_shards; r++) {
            if (all_srch[r] != srch_tmp) {
                fprintf(stderr, "FATAL: srch_threads mismatch: rank %d has %u, rank %u has %u\n",
                        world_rank, srch_tmp, r, all_srch[r]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    if (pushdown_participant) {
        const auto* sb0 = static_cast<const SuperBlockV1*>(shard_ptrs[0].first);
        const int32_t* cum_nn0 = cum_nneighbor_ptr(shard_ptrs[0].first, *sb0);
        uint32_t max_batch = static_cast<uint32_t>(cum_nn0[1]);
        {
            uint64_t scaled = static_cast<uint64_t>(max_batch) * expand_batch;
            if (scaled > UINT32_MAX) {
                fprintf(stderr, "FATAL: max_batch * expand_batch overflows uint32: %u * %u\n",
                        max_batch, expand_batch);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            max_batch = static_cast<uint32_t>(scaled);
        }

        uint32_t srch_threads = srch_threads_validated;

        uint32_t num_remote = num_shards - 1;

        // --- Dual-channel path: task_inbox + result_inbox ---
        // task_inbox: this rank's service reads, remote search writes
        // result_inbox: this rank's search reads, remote service writes
        // Channel flags determined by g_channel_flags (noncache in cache mode)
        TaskChannelLayout task_layout = TaskChannelLayout::build(max_batch, dim);
        ResultChannelLayout result_layout = ResultChannelLayout::build(max_batch);

        uint32_t my_rank_u = static_cast<uint32_t>(world_rank);

        // Create task_inbox (service reads from here)
        std::string task_name = task_inbox_name_for_rank(my_rank_u);
        ubsmem_deallocate_retry(task_name.c_str());
        size_t task_total = static_cast<size_t>(num_remote) * srch_threads * task_layout.bytes_total;
        size_t task_alloc = ubsmem_align_size(task_total);
        int ubret = ubsmem_shmem_allocate_with_provider(
            &g_provider, task_name.c_str(), task_alloc, 0600, g_channel_flags);
        if (ubret != UBSM_OK) {
            fprintf(stderr, "ubsmem_shmem_allocate_with_provider task_inbox '%s' failed: %d\n",
                    task_name.c_str(), ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        void* task_base = nullptr;
        ubret = ubsmem_shmem_map(nullptr, task_alloc, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, task_name.c_str(), 0, &task_base);
        if (ubret != UBSM_OK || !task_base) {
            fprintf(stderr, "ubsmem_shmem_map task_inbox '%s' failed: %d\n",
                    task_name.c_str(), ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::memset(task_base, 0, task_total);
        // Initialize task_inbox headers (this rank is owner/service reader)
        for (uint32_t ri = 0; ri < num_remote; ri++) {
            for (uint32_t t = 0; t < srch_threads; t++) {
                size_t off = dual_channel_slot(ri, t, srch_threads, task_layout.bytes_total);
                auto tv = TaskChannelView::from_raw(
                    static_cast<char*>(task_base) + off, task_layout);
                tv.h->run_generation = run_gen;
                tv.h->max_batch = max_batch;
                tv.h->dim = dim;
                tv.h->channel_role = CHANNEL_ROLE_TASK;
                tv.h->dst_shard = static_cast<uint16_t>(my_shard);
                tv.h->identity_valid = IDENTITY_VALID_MAGIC;
            }
        }
        requester_channel_maps.push_back({task_base, task_alloc});
        owned_channel_shm_names.push_back(task_name);

        // Create result_inbox (search threads read from here)
        std::string result_name = result_inbox_name_for_rank(my_rank_u);
        ubsmem_deallocate_retry(result_name.c_str());
        size_t result_total = static_cast<size_t>(num_remote) * srch_threads * result_layout.bytes_total;
        size_t result_alloc = ubsmem_align_size(result_total);
        ubret = ubsmem_shmem_allocate_with_provider(
            &g_provider, result_name.c_str(), result_alloc, 0600, g_channel_flags);
        if (ubret != UBSM_OK) {
            fprintf(stderr, "ubsmem_shmem_allocate_with_provider result_inbox '%s' failed: %d\n",
                    result_name.c_str(), ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        void* result_base = nullptr;
        ubret = ubsmem_shmem_map(nullptr, result_alloc, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, result_name.c_str(), 0, &result_base);
        if (ubret != UBSM_OK || !result_base) {
            fprintf(stderr, "ubsmem_shmem_map result_inbox '%s' failed: %d\n",
                    result_name.c_str(), ubret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::memset(result_base, 0, result_total);
        // Initialize result_inbox headers
        for (uint32_t ri = 0; ri < num_remote; ri++) {
            for (uint32_t t = 0; t < srch_threads; t++) {
                size_t off = dual_channel_slot(ri, t, srch_threads, result_layout.bytes_total);
                auto rv = ResultChannelView::from_raw(
                    static_cast<char*>(result_base) + off, result_layout);
                rv.h->run_generation = run_gen;
                rv.h->max_batch = max_batch;
                rv.h->dim = dim;
                rv.h->channel_role = CHANNEL_ROLE_RESULT;
                rv.h->src_rank = static_cast<uint16_t>(my_rank_u);
                rv.h->identity_valid = IDENTITY_VALID_MAGIC;
            }
        }
        requester_channel_maps.push_back({result_base, result_alloc});
        owned_channel_shm_names.push_back(result_name);

        // Wire up DualDistProxy: each proxy holds a task slot (remote) + result slot (own)
        // Result slots are in our own result_inbox
        dual_pushdown_proxies.resize(srch_threads);
        for (uint32_t t = 0; t < srch_threads; t++) {
            dual_pushdown_proxies[t].resize(num_shards);
            // result slots for thread t are in our result_inbox
            // (will be paired with remote task slots after barrier)
        }
    }

    // --- Collective 2: Wait for all ranks to finish channel creation ---
    if (pushdown_active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (pushdown_participant) {
        const auto* sb0 = static_cast<const SuperBlockV1*>(shard_ptrs[0].first);
        const int32_t* cum_nn0 = cum_nneighbor_ptr(shard_ptrs[0].first, *sb0);
        uint32_t max_batch = static_cast<uint32_t>(cum_nn0[1]);
        {
            uint64_t scaled = static_cast<uint64_t>(max_batch) * expand_batch;
            if (scaled > UINT32_MAX) {
                fprintf(stderr, "FATAL: max_batch * expand_batch overflows uint32: %u * %u\n",
                        max_batch, expand_batch);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            max_batch = static_cast<uint32_t>(scaled);
        }

        uint32_t svc_threads = resolve_service_threads(pushdown_cfg.service_threads);
        uint32_t srch_threads = resolve_search_threads(svc_threads);
        uint32_t num_remote = num_shards - 1;
        uint32_t my_rank_u = static_cast<uint32_t>(world_rank);

        std::vector<int> svc_core_ids;  // no core pinning

        const auto& my_shard_view = searcher.shard_by_id(my_shard);

        // --- Dual-channel Step 2: open remote files, wire up pairs ---
        TaskChannelLayout task_layout = TaskChannelLayout::build(max_batch, dim);
        ResultChannelLayout result_layout = ResultChannelLayout::build(max_batch);

        size_t task_file_bytes = static_cast<size_t>(num_remote) * srch_threads
                                 * task_layout.bytes_total;
        size_t result_file_bytes = static_cast<size_t>(num_remote) * srch_threads
                                   * result_layout.bytes_total;

        // For each remote rank, open its task_inbox (search writes) and result_inbox (service writes)
        // Also collect service channel pairs from our own task_inbox + remote result_inboxes
        struct RemoteMapping {
            void* task_base;    // remote rank's task_inbox (we write tasks)
            void* result_base;  // remote rank's result_inbox (we write results)
            size_t task_alloc;
            size_t result_alloc;
        };
        std::vector<RemoteMapping> remote_maps(num_shards);

        for (uint32_t src = 0; src < num_shards; src++) {
            if (src == my_shard) continue;

            // Open remote task_inbox (our search threads write tasks to src's service)
            std::string remote_task = task_inbox_name_for_rank(src);
            size_t t_alloc = ubsmem_align_size(task_file_bytes);
            void* t_base = nullptr;
            int ubret = ubsmem_shmem_map(nullptr, t_alloc, PROT_READ | PROT_WRITE,
                                         MAP_SHARED, remote_task.c_str(), 0, &t_base);
            if (ubret != UBSM_OK || !t_base) {
                fprintf(stderr, "rank %d: ubsmem_shmem_map task_inbox '%s' failed: %d\n",
                        world_rank, remote_task.c_str(), ubret);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            service_channel_maps.push_back({t_base, t_alloc});
            remote_maps[src].task_base = t_base;
            remote_maps[src].task_alloc = t_alloc;

            // Open remote result_inbox (our service writes results to src's search threads)
            std::string remote_result = result_inbox_name_for_rank(src);
            size_t r_alloc = ubsmem_align_size(result_file_bytes);
            void* r_base = nullptr;
            ubret = ubsmem_shmem_map(nullptr, r_alloc, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, remote_result.c_str(), 0, &r_base);
            if (ubret != UBSM_OK || !r_base) {
                fprintf(stderr, "rank %d: ubsmem_shmem_map result_inbox '%s' failed: %d\n",
                        world_rank, remote_result.c_str(), ubret);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            service_channel_maps.push_back({r_base, r_alloc});
            remote_maps[src].result_base = r_base;
            remote_maps[src].result_alloc = r_alloc;
        }

        // Retrieve our own task_inbox and result_inbox bases from requester_channel_maps
        // (pushed in Step 1: [0]=task_inbox, [1]=result_inbox)
        void* own_task_base = requester_channel_maps[requester_channel_maps.size() - 2].first;
        void* own_result_base = requester_channel_maps[requester_channel_maps.size() - 1].first;

        // Wire DualDistProxy: task slot in remote task_inbox, result slot in own result_inbox
        for (uint32_t t = 0; t < srch_threads; t++) {
            for (uint32_t s = 0; s < num_shards; s++) {
                if (s == my_shard) continue;
                // Task slot: remote task_inbox_{s}[slot(compressed_remote_idx(my_rank, s), t)]
                uint32_t task_ri = compressed_remote_idx(my_rank_u, s);
                size_t task_off = dual_channel_slot(task_ri, t, srch_threads,
                                                    task_layout.bytes_total);
                auto tv = TaskChannelView::from_raw(
                    static_cast<char*>(remote_maps[s].task_base) + task_off, task_layout);

                // Result slot: own result_inbox[slot(compressed_remote_idx(s, my_rank), t)]
                uint32_t result_ri = compressed_remote_idx(s, my_rank_u);
                size_t result_off = dual_channel_slot(result_ri, t, srch_threads,
                                                      result_layout.bytes_total);
                auto rv = ResultChannelView::from_raw(
                    static_cast<char*>(own_result_base) + result_off, result_layout);

                dual_pushdown_proxies[t][s].init(tv, rv, dim,
                                                  pushdown_cfg.max_spin_iters,
                                                  max_batch,
                                                  static_cast<uint16_t>(my_rank_u),
                                                  static_cast<uint16_t>(t));
            }
        }

        // Collect service channel pairs: own task_inbox + remote result_inboxes
        // Iteration order: (src_rank, thread) for RR distribution to service threads
        std::vector<ServiceChannelPair> service_pairs;
        for (uint32_t src = 0; src < num_shards; src++) {
            if (src == my_shard) continue;
            uint32_t src_ri = compressed_remote_idx(src, my_rank_u);
            for (uint32_t t = 0; t < srch_threads; t++) {
                // Task slot: own task_inbox[slot(src_ri, t)]
                size_t task_off = dual_channel_slot(src_ri, t, srch_threads,
                                                    task_layout.bytes_total);
                auto tv = TaskChannelView::from_raw(
                    static_cast<char*>(own_task_base) + task_off, task_layout);

                // Result slot: remote result_inbox_{src}[slot(compressed_remote_idx(my_shard, src), t)]
                uint32_t result_ri = compressed_remote_idx(my_rank_u, src);
                size_t result_off = dual_channel_slot(result_ri, t, srch_threads,
                                                      result_layout.bytes_total);
                auto rv = ResultChannelView::from_raw(
                    static_cast<char*>(remote_maps[src].result_base) + result_off,
                    result_layout);

                service_pairs.push_back({tv, rv});
            }
        }

        dual_dist_services.push_back(std::make_unique<DualDistService>(
            my_shard, my_shard_view, dim, run_gen,
            svc_threads, std::move(service_pairs), svc_core_ids,
            profile_mode));

        // Enable dot_norm on service workers if requested
        if (use_dot_norm) {
            dual_dist_services.back()->enable_dot_norm(searcher.vector_norms_data());
        }

        searcher.set_dual_pushdown_proxies(&dual_pushdown_proxies, srch_threads);
    }

    {
        auto t_pushdown_end = std::chrono::steady_clock::now();
        double pushdown_sec = std::chrono::duration<double>(t_pushdown_end - t_pushdown_start).count();
        if (pushdown_active) {
            printf("[rank %d] Pushdown setup: %.2f sec\n", world_rank, pushdown_sec);
        }
    }

    // --- Collective 3: All ranks synchronize before search begins ---
    if (pushdown_active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (pushdown_participant && world_rank == 0) {
        uint32_t svc_threads = resolve_service_threads(pushdown_cfg.service_threads);
        uint32_t srch_threads = resolve_search_threads(svc_threads);
        uint32_t num_remote = num_shards - 1;
        const auto* sb0 = static_cast<const SuperBlockV1*>(shard_ptrs[0].first);
        const int32_t* cum_nn0 = cum_nneighbor_ptr(shard_ptrs[0].first, *sb0);
        uint32_t max_batch = static_cast<uint32_t>(cum_nn0[1]);
        {
            uint64_t scaled = static_cast<uint64_t>(max_batch) * expand_batch;
            if (scaled > UINT32_MAX) {
                fprintf(stderr, "FATAL: max_batch * expand_batch overflows uint32: %u * %u\n",
                        max_batch, expand_batch);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            max_batch = static_cast<uint32_t>(scaled);
        }
        printf("[pushdown] Enabled (cross-process, dual-channel): search_threads=%u, "
               "service_threads=%u (serving own shard), "
               "max_batch=%u, channels_per_rank=%u, shm_files=%u (2/rank: task+result)",
               srch_threads, svc_threads,
               max_batch, srch_threads * num_remote, num_shards * 2);
        printf("\n");
    }

    SearchParams params;
    params.k = K;
    params.ef_search = static_cast<int32_t>(ef_search);

    const float* my_queries = my_queries_buf.data();

    // Pre-allocate flat output buffers (zero per-query allocation)
    uint64_t buf_size = static_cast<uint64_t>(my_nq) * K;
    std::vector<int32_t> out_ids(buf_size);
    std::vector<float> out_dists(buf_size);

    // Warmup round
    if (world_rank == 0) {
        printf("[bench] Warmup round...\n");
    }
    searcher.search_batch_flat(my_queries, static_cast<int32_t>(my_nq), params,
                               out_ids.data(), out_dists.data());
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed rounds
    std::vector<double> round_qps(num_rounds);
    std::vector<double> round_e2e_qps(num_rounds);

    // Per-query latency collection (last round only)
    std::vector<double> query_latencies_us(my_nq);

    for (int r = 0; r < num_rounds; r++) {
        MPI_Barrier(MPI_COMM_WORLD);

        const bool last_round = (r == num_rounds - 1);

        if (last_round) {
            // Last round: per-query timing for latency distribution
            auto t0 = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (int32_t i = 0; i < static_cast<int32_t>(my_nq); i++) {
                auto qt0 = std::chrono::steady_clock::now();
                searcher.search_to(
                    my_queries + static_cast<uint64_t>(i) * dim,
                    params,
                    out_ids.data() + static_cast<uint64_t>(i) * K,
                    out_dists.data() + static_cast<uint64_t>(i) * K);
                auto qt1 = std::chrono::steady_clock::now();
                query_latencies_us[i] = std::chrono::duration<double, std::micro>(qt1 - qt0).count();
            }
            auto t1 = std::chrono::steady_clock::now();
            double search_sec = std::chrono::duration<double>(t1 - t0).count();

            double local_qps = (search_sec > 0.0) ? (my_nq / search_sec) : 0.0;
            double total_qps = 0;
            MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            round_qps[r] = total_qps;

            double max_sec = 0;
            MPI_Reduce(&search_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // E2E QPS: total queries / max wall-time across all ranks
            double e2e_qps = (max_sec > 0.0) ? (nq / max_sec) : 0.0;
            round_e2e_qps[r] = e2e_qps;

            if (world_rank == 0) {
                printf("[bench] Round %d/%d: %.1f QPS (agg), %.1f QPS (e2e)\n",
                       r + 1, num_rounds, total_qps, e2e_qps);
            }
        } else {
            auto t0 = std::chrono::steady_clock::now();
            searcher.search_batch_flat(my_queries, static_cast<int32_t>(my_nq), params,
                                       out_ids.data(), out_dists.data());
            auto t1 = std::chrono::steady_clock::now();

            double search_sec = std::chrono::duration<double>(t1 - t0).count();

            double local_qps = (search_sec > 0.0) ? (my_nq / search_sec) : 0.0;
            double total_qps = 0;
            MPI_Reduce(&local_qps, &total_qps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            round_qps[r] = total_qps;

            double max_sec = 0;
            MPI_Reduce(&search_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            double e2e_qps = (max_sec > 0.0) ? (nq / max_sec) : 0.0;
            round_e2e_qps[r] = e2e_qps;

            if (world_rank == 0) {
                printf("[bench] Round %d/%d: %.1f QPS\n", r + 1, num_rounds, total_qps);
            }
        }
    }

    // Final-round safety barrier (pushdown only):
    // non-root ranks can return from MPI_Reduce early, and without a post-loop
    // sync they may advance to teardown and stop DistService while other ranks
    // are still inside the last round's search and need remote service.
    if (pushdown_active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Compute recall from last round's flat buffer
    RecallStats local_stats = compute_recall_stats(
        out_ids.data(), my_nq, my_gt.data(), gt_k, K,
        has_idmap ? &new_to_old_idmap : nullptr);

    // Profile run (separate from timed benchmark)
    SearchProfileSummary prof_summary;
    if (profile_mode) {
        prof_summary = searcher.search_batch_profiled(
            my_queries, static_cast<int32_t>(my_nq), params,
            out_ids.data(), out_dists.data());
    }

    // ========================================
    // Phase 5: Latency percentiles (per-query from last round)
    // ========================================
    std::sort(query_latencies_us.begin(), query_latencies_us.end());
    auto percentile = [&](double pct) -> double {
        if (my_nq == 0) return 0.0;
        size_t idx = static_cast<size_t>(pct / 100.0 * (my_nq - 1));
        if (idx >= static_cast<size_t>(my_nq)) idx = my_nq - 1;
        return query_latencies_us[idx];
    };
    double local_p50 = percentile(50);
    double local_p95 = percentile(95);
    double local_p99 = percentile(99);
    double local_max_lat = (my_nq > 0) ? query_latencies_us.back() : 0.0;

    // Take max across ranks (worst-case per-query latency at each percentile)
    double global_p50 = 0, global_p95 = 0, global_p99 = 0, global_max_lat = 0;
    MPI_Reduce(&local_p50, &global_p50, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_p95, &global_p95, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_p99, &global_p99, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_lat, &global_max_lat, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ========================================
    // Phase 5b: Aggregate and report
    // ========================================
    uint64_t total_hits = 0;
    uint64_t total_pairs = 0;
    MPI_Reduce(&local_stats.hits, &total_hits, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_stats.total, &total_pairs, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // QPS stats: exclude last round (it uses per-query timing with overhead)
        int qps_rounds = (num_rounds > 1) ? (num_rounds - 1) : num_rounds;
        std::vector<double> qps_sample(round_qps.begin(), round_qps.begin() + qps_rounds);
        std::vector<double> e2e_qps_sample(round_e2e_qps.begin(), round_e2e_qps.begin() + qps_rounds);
        std::sort(qps_sample.begin(), qps_sample.end());
        std::sort(e2e_qps_sample.begin(), e2e_qps_sample.end());
        double median_qps = qps_sample[qps_rounds / 2];
        double median_e2e_qps = e2e_qps_sample[qps_rounds / 2];

        uint64_t total_shm = 0;
        for (uint32_t s = 0; s < num_shards; s++) total_shm += shard_sizes[s];

        double avg_recall = (total_pairs > 0)
                                ? (static_cast<double>(total_hits) / total_pairs)
                                : 0.0;
        printf("\n========== BENCHMARK RESULTS ==========\n");
        printf("Dataset:          %s\n", dataset_path.c_str());
        printf("N vectors:        %lu\n", (unsigned long)n_train);
        printf("Queries:          %lu\n", (unsigned long)nq);
        printf("Workers:          %d\n", world_size);
        printf("Threads/worker:   %d\n", num_threads);
        printf("Shards:           %u\n", num_shards);
        printf("Rounds:           %d (+ 1 warmup)\n", num_rounds);
        printf("HNSW M:           %u\n", M);
        printf("ef_construction:  %u\n", ef_construction);
        const char* ef_source = ef_search_from_cli ? "cli" :
                               (!load_index_path.empty() ? "index" : "default");
        printf("ef_search:        %u (%s)\n", ef_search, ef_source);
        printf("top-K:            %d\n", K);
        printf("---------------------------------------\n");
        printf("Recall@%d:         %.4f\n", K, avg_recall);
        printf("Aggregate QPS:    %.1f (median), %.1f (min), %.1f (max)\n",
               median_qps, qps_sample.front(), qps_sample.back());
        printf("E2E QPS:          %.1f (median), %.1f (min), %.1f (max)\n",
               median_e2e_qps, e2e_qps_sample.front(), e2e_qps_sample.back());
        printf("---------------------------------------\n");
        printf("Latency (per-query, last round, worst-rank):\n");
        printf("  p50:            %.3f ms\n", global_p50 / 1000.0);
        printf("  p95:            %.3f ms\n", global_p95 / 1000.0);
        printf("  p99:            %.3f ms\n", global_p99 / 1000.0);
        printf("  max:            %.3f ms\n", global_max_lat / 1000.0);
        printf("SHM total:        %.1f MB (%u shards)\n",
               total_shm / (1024.0 * 1024.0), num_shards);
        printf("=======================================\n");
    }

    // Profile output (each rank prints its own, rank 0 first)
    if (profile_mode) {
        for (int r = 0; r < world_size; r++) {
            if (r == world_rank) {
                const auto& s = prof_summary;
                auto safe_div = [](double num, double den) {
                    return (den > 0.0) ? (num / den) : 0.0;
                };
                auto safe_pct = [](double part, double total) {
                    return (total > 0.0) ? (100.0 * part / total) : 0.0;
                };

                double dist_calls_total = s.avg_dist_local + s.avg_dist_remote + s.avg_dist_pushed;
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                double dist_ns_total = s.avg_dist_ns_local + s.avg_dist_ns_remote;
                double avg_ns_per_local = safe_div(s.avg_dist_ns_local, s.avg_dist_local);
                double avg_ns_per_remote = safe_div(s.avg_dist_ns_remote, s.avg_dist_remote);
                double avg_ns_per_dist = safe_div(dist_ns_total, s.avg_dist_local + s.avg_dist_remote);
                double avg_mem_ns_per_local = safe_div(s.avg_dist_mem_ns_local, s.avg_dist_split_calls_local);
                double avg_compute_ns_per_local = safe_div(s.avg_dist_compute_ns_local, s.avg_dist_split_calls_local);
                double avg_mem_ns_per_remote = safe_div(s.avg_dist_mem_ns_remote, s.avg_dist_split_calls_remote);
                double avg_compute_ns_per_remote = safe_div(s.avg_dist_compute_ns_remote, s.avg_dist_split_calls_remote);
#endif
                double upper_pct_total = safe_pct(s.avg_upper_ns, s.avg_total_ns);
                double layer0_pct_total = safe_pct(s.avg_layer0_ns, s.avg_total_ns);
                double edges_per_expand = safe_div(s.avg_edges_scanned, s.avg_nodes_expanded);
                double pushes_per_expand = safe_div(s.avg_heap_pushes, s.avg_nodes_expanded);
                double pushes_per_edge = safe_div(s.avg_heap_pushes, s.avg_edges_scanned);

                printf("\n--- Profile [rank %d, owns shard %u] (avg per query, nq=%lu) ---\n",
                       world_rank, my_shard, (unsigned long)s.nq);
                printf("  [Distance]\n");
                printf("    dist_calls_total:   %.1f (local %.1f + fallback %.1f + pushed %.1f)\n",
                       dist_calls_total, s.avg_dist_local, s.avg_dist_remote, s.avg_dist_pushed);
                printf("    dist_local:         %.1f (%.1f%%)\n",
                       s.avg_dist_local, safe_pct(s.avg_dist_local, dist_calls_total));
                printf("    dist_remote_fallback: %.1f (%.1f%%, computed by search thread)\n",
                       s.avg_dist_remote, safe_pct(s.avg_dist_remote, dist_calls_total));
                printf("    dist_pushed:        %.1f (%.1f%%, offloaded to service)\n",
                       s.avg_dist_pushed, safe_pct(s.avg_dist_pushed, dist_calls_total));
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                double dist_pct_total = safe_pct(dist_ns_total, s.avg_total_ns);
                printf("    dist_ns_total:      %.1f ns (%.1f%% of total_time, local-thread only)\n",
                       dist_ns_total, dist_pct_total);
                printf("    avg_dist_ns/call:   %.1f ns (local-thread)\n", avg_ns_per_dist);
                printf("    avg_local_ns/call:  %.1f ns\n", avg_ns_per_local);
                if (s.avg_dist_split_calls_local > 0) {
                    printf("      layer0 mem/compute: mem %.1f + compute %.1f = %.1f ns\n",
                           avg_mem_ns_per_local, avg_compute_ns_per_local,
                           avg_mem_ns_per_local + avg_compute_ns_per_local);
                }
                printf("    avg_remote_ns/call: %.1f ns (fallback path)\n", avg_ns_per_remote);
                if (s.avg_dist_split_calls_remote > 0) {
                    printf("      layer0 mem/compute: mem %.1f + compute %.1f = %.1f ns\n",
                           avg_mem_ns_per_remote, avg_compute_ns_per_remote,
                           avg_mem_ns_per_remote + avg_compute_ns_per_remote);
                }
#endif

                printf("  [Traversal]\n");
                printf("    nodes_expanded:     %.1f\n", s.avg_nodes_expanded);
                printf("    edges_scanned:      %.1f (%.2f / expanded)\n",
                       s.avg_edges_scanned, edges_per_expand);
                printf("    heap_pushes:        %.1f (%.2f / expanded, %.3f / edge)\n",
                       s.avg_heap_pushes, pushes_per_expand, pushes_per_edge);
                printf("    heap_rejections:    %.1f (%.1f%% reject rate)\n",
                       s.avg_heap_rejections,
                       safe_pct(s.avg_heap_rejections, s.avg_heap_pushes + s.avg_heap_rejections));

                printf("  [Timeline]\n");
                printf("    upper_level_time:   %.1f ns (%.1f%%)\n",
                       s.avg_upper_ns, upper_pct_total);
                printf("    layer0_time:        %.1f ns (%.1f%%)\n",
                       s.avg_layer0_ns, layer0_pct_total);
                printf("    total_time:         %.1f ns (%.3f ms)\n",
                       s.avg_total_ns, s.avg_total_ns / 1e6);
                if (s.pushdown_active) {
                    double total_batches = s.avg_batches_pushed + s.avg_batches_fallback;
                    double total_ids = s.avg_ids_pushed + s.avg_ids_fallback;
                    double pushdown_batch_pct = safe_pct(s.avg_batches_pushed, total_batches);
                    double pushdown_id_pct = safe_pct(s.avg_ids_pushed, total_ids);
                    double avg_ids_per_push_batch = safe_div(s.avg_ids_pushed, s.avg_batches_pushed);
                    double avg_ids_per_fallback_batch = safe_div(s.avg_ids_fallback, s.avg_batches_fallback);
                    double remote_expand_pct = safe_pct(s.avg_expansions_with_remote, s.avg_nodes_expanded);
                    double poll_wait_us = s.avg_poll_wait_ns / 1e3;
                    double poll_wait_us_per_expand = safe_div(s.avg_poll_wait_ns, s.avg_expansions_with_remote) / 1e3;
                    double poll_wait_us_per_push_batch = safe_div(s.avg_poll_wait_ns, s.avg_batches_pushed) / 1e3;
                    double total_spins = s.avg_poll_spins_pause + s.avg_poll_spins_yield + s.avg_poll_spins_sleep;
                    double pause_spin_pct = safe_pct(s.avg_poll_spins_pause, total_spins);
                    double yield_spin_pct = safe_pct(s.avg_poll_spins_yield, total_spins);
                    double sleep_spin_pct = safe_pct(s.avg_poll_spins_sleep, total_spins);

                    // Phase timing breakdown
                    double phase_sum = s.avg_phase1_ns + s.avg_phase2_ns
                                     + s.avg_phase3_ns + s.avg_poll_wait_ns;
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    double p3_overhead = s.avg_phase3_ns - s.avg_dist_ns_local - s.avg_dist_ns_remote;
#endif
                    printf("  [Pushdown Phases]\n");
                    printf("    P1 classify:         %.1f ns (%.3f us, %.1f%%)\n",
                           s.avg_phase1_ns, s.avg_phase1_ns / 1e3,
                           safe_pct(s.avg_phase1_ns, phase_sum));
                    printf("      graph_access:      %.1f ns (%.1f%% of P1)\n",
                           s.avg_graph_access_ns,
                           safe_pct(s.avg_graph_access_ns, s.avg_phase1_ns));
                    printf("      classify_loop:     %.1f ns (%.1f%% of P1, vt+shard+route)\n",
                           s.avg_classify_loop_ns,
                           safe_pct(s.avg_classify_loop_ns, s.avg_phase1_ns));
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    printf("        vt_check:        %.1f ns (%.1f%% of loop)\n",
                           s.avg_vt_check_ns,
                           safe_pct(s.avg_vt_check_ns, s.avg_classify_loop_ns));
                    printf("        find_shard:      %.1f ns (%.1f%% of loop)\n",
                           s.avg_find_shard_ns,
                           safe_pct(s.avg_find_shard_ns, s.avg_classify_loop_ns));
                    printf("        route_push:      %.1f ns (%.1f%% of loop)\n",
                           s.avg_route_push_ns,
                           safe_pct(s.avg_route_push_ns, s.avg_classify_loop_ns));
#endif
                    printf("    P2 submit:           %.1f ns (%.3f us, %.1f%%)\n",
                           s.avg_phase2_ns, s.avg_phase2_ns / 1e3,
                           safe_pct(s.avg_phase2_ns, phase_sum));
                    printf("      query_memcpy:      %.1f ns (%.1f%% of P2)\n",
                           s.avg_phase2_query_ns,
                           safe_pct(s.avg_phase2_query_ns, s.avg_phase2_ns));
                    printf("      ids_memcpy:        %.1f ns (%.1f%% of P2)\n",
                           s.avg_phase2_ids_ns,
                           safe_pct(s.avg_phase2_ids_ns, s.avg_phase2_ns));
                    printf("      doorbell:          %.1f ns (%.1f%% of P2)\n",
                           s.avg_phase2_doorbell_ns,
                           safe_pct(s.avg_phase2_doorbell_ns, s.avg_phase2_ns));
                    double loop_other = std::max(0.0, s.avg_phase2_ns - s.avg_phase2_query_ns - s.avg_phase2_ids_ns - s.avg_phase2_doorbell_ns);
                    printf("      loop+other:        %.1f ns (%.1f%% of P2)\n",
                           loop_other, safe_pct(loop_other, s.avg_phase2_ns));
                    printf("    P3 local_compute:    %.1f ns (%.3f us, %.1f%%)\n",
                           s.avg_phase3_ns, s.avg_phase3_ns / 1e3,
                           safe_pct(s.avg_phase3_ns, phase_sum));
#ifndef HNSW_PROFILE_LIGHTWEIGHT
                    printf("      dist_kernel:       %.1f ns (%.1f%% of P3)\n",
                           s.avg_dist_ns_local + s.avg_dist_ns_remote,
                           safe_pct(s.avg_dist_ns_local + s.avg_dist_ns_remote, s.avg_phase3_ns));
                    printf("      overhead:          %.1f ns (%.1f%% of P3, prefetch+lookup+heap)\n",
                           p3_overhead > 0 ? p3_overhead : 0,
                           safe_pct(p3_overhead > 0 ? p3_overhead : 0, s.avg_phase3_ns));
#endif
                    printf("    P4 poll(total):      %.1f ns (%.3f us, %.1f%%)\n",
                           s.avg_poll_wait_ns, s.avg_poll_wait_ns / 1e3,
                           safe_pct(s.avg_poll_wait_ns, phase_sum));
                    printf("      spin_wait:         %.1f ns (%.1f%% of P4)\n",
                           s.avg_poll_spin_ns,
                           safe_pct(s.avg_poll_spin_ns, s.avg_poll_wait_ns));
                    printf("      consume+insert:    %.1f ns (%.1f%% of P4)\n",
                           s.avg_poll_consume_ns,
                           safe_pct(s.avg_poll_consume_ns, s.avg_poll_wait_ns));
                    printf("      overhead(probe+ctrl): %.1f ns (%.1f%% of P4)\n",
                           s.avg_poll_overhead_ns,
                           safe_pct(s.avg_poll_overhead_ns, s.avg_poll_wait_ns));
                    printf("    phase_sum:           %.1f ns (%.1f%% of layer0_time)\n",
                           phase_sum, safe_pct(phase_sum, s.avg_layer0_ns));

                    // Neighbor routing
                    double total_neighbors = s.avg_neighbors_local + s.avg_neighbors_remote;
                    printf("  [Neighbor Routing]\n");
                    printf("    neighbors_local:     %.1f (%.1f%%)\n",
                           s.avg_neighbors_local, safe_pct(s.avg_neighbors_local, total_neighbors));
                    printf("    neighbors_remote:    %.1f (%.1f%%)\n",
                           s.avg_neighbors_remote, safe_pct(s.avg_neighbors_remote, total_neighbors));
                    printf("    vt_dedup_hits:       %.1f (%.1f%% of edges scanned)\n",
                           s.avg_vt_dedup_hits,
                           safe_pct(s.avg_vt_dedup_hits, s.avg_edges_scanned));
                    printf("    unique_neighbors:    %.1f / edge_scanned\n",
                           safe_div(total_neighbors, s.avg_edges_scanned));

                    printf("  [Pushdown Routing]\n");
                    printf("    batches_pushed:      %.1f (%.1f ids, %.2f ids/batch)\n",
                           s.avg_batches_pushed, s.avg_ids_pushed, avg_ids_per_push_batch);
                    printf("    batches_fallback:    %.1f (%.1f ids, %.2f ids/batch)\n",
                           s.avg_batches_fallback, s.avg_ids_fallback, avg_ids_per_fallback_batch);
                    printf("    pushdown_hit_rate:   %.1f%% by batches, %.1f%% by ids\n",
                           pushdown_batch_pct, pushdown_id_pct);
                    printf("    expansions_w_remote: %.1f (%.1f%% of expanded nodes)\n",
                           s.avg_expansions_with_remote, remote_expand_pct);

                    printf("  [Pushdown Poll Detail]\n");
                    printf("    poll_wait_avg:       %.1f ns (%.3f us)\n",
                           s.avg_poll_wait_ns, poll_wait_us);
                    printf("    poll_wait_max:       %.1f ns (%.3f us)\n",
                           s.max_poll_wait_ns, s.max_poll_wait_ns / 1e3);
                    printf("    poll_wait/remote_expand: %.3f us\n", poll_wait_us_per_expand);
                    printf("    poll_wait/push_batch:    %.3f us\n", poll_wait_us_per_push_batch);
                    printf("    poll_spins_total:    %.1f (pause %.1f%%, yield %.1f%%, sleep %.1f%%)\n",
                           total_spins, pause_spin_pct, yield_spin_pct, sleep_spin_pct);
                    printf("    poll_spins_raw:      pause=%.1f yield=%.1f sleep=%.1f\n",
                           s.avg_poll_spins_pause, s.avg_poll_spins_yield, s.avg_poll_spins_sleep);
                }
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ========================================
    // Cleanup
    // ========================================
    // Ensure all ranks have finished all search operations (including profiled
    // runs) before any rank shuts down its DistService.  Without this barrier,
    // a fast rank can destroy its service workers while a slower rank still has
    // in-flight pushdown requests targeting that shard, causing poll timeout.
    if (pushdown_active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // Print service-side profile before destroying workers (rank-ordered)
    if (profile_mode && pushdown_active) {
        // Shutdown all workers first so stats are final
        for (auto& svc : dual_dist_services) svc->shutdown();
        for (int r = 0; r < world_size; r++) {
            if (r == world_rank) {
                for (auto& svc : dual_dist_services)
                    svc->print_stats(world_rank, my_shard);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    // Shutdown pushdown services before unmapping shm
    dual_dist_services.clear();
    dual_pushdown_proxies.clear();

    // Unmap service-side channel shm
    for (auto& [ptr, sz] : service_channel_maps) {
        ubsmem_shmem_unmap(ptr, sz);
    }
    service_channel_maps.clear();

    // Unmap requester-side channel shm
    for (auto& [ptr, sz] : requester_channel_maps) {
        ubsmem_shmem_unmap(ptr, sz);
    }
    requester_channel_maps.clear();

    // Unmap read-only shard mappings (were never unmapped in POSIX version)
    for (auto& [ptr, sz] : shard_ptrs) {
        if (ptr) ubsmem_shmem_unmap(const_cast<void*>(ptr), ubsmem_align_size(sz));
    }

    // Barrier: all ranks must finish unmapping before any rank deallocates
    MPI_Barrier(MPI_COMM_WORLD);

    // Deallocate channel shm segments owned by this rank
    for (const auto& name : owned_channel_shm_names) {
        int dr = ubsmem_deallocate_retry(name.c_str());
        if (dr != UBSM_OK) {
            fprintf(stderr, "[rank %d] WARNING: ubsmem_deallocate_retry('%s') failed: %d\n",
                    world_rank, name.c_str(), dr);
        }
    }
    owned_channel_shm_names.clear();

    // Barrier: all channel deallocations must complete before shard cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 deallocates all shard shm segments
    if (world_rank == 0) {
        for (uint32_t s = 0; s < num_shards; s++) {
            std::string name = shm_name_for_shard(s);
            int dr = ubsmem_deallocate_retry(name.c_str());
            if (dr != UBSM_OK) {
                fprintf(stderr, "[rank 0] WARNING: ubsmem_deallocate_retry('%s') failed: %d\n",
                        name.c_str(), dr);
            }
        }
    }

    ubsmem_finalize();
    MPI_Finalize();
    return 0;
}
