#include <mpi.h>
#include <omp.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/index_io.h>
#include <hdf5.h>

#include <ubs_mem_def.h>
#include <ubs_mem.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_set>
#include <cstring>
#include <cmath>
#include <climits>
#include <atomic>
#include <sys/mman.h>

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <unistd.h>
#endif

// ===================================================
// UBS shared memory helpers
// ===================================================

static uint32_t g_job_id = 0;
static ubs_mem_provider_t g_provider{};

static constexpr size_t UBSMEM_ALIGN = 4UL * 1024 * 1024;
static size_t ubsmem_align_size(size_t sz) {
    if (sz == 0) return UBSMEM_ALIGN;
    return (sz + UBSMEM_ALIGN - 1) & ~(UBSMEM_ALIGN - 1);
}

static int ubsmem_deallocate_retry(const char* name, int max_retries = 3) {
    for (int i = 0; i <= max_retries; i++) {
        int ret = ubsmem_shmem_deallocate(name);
        if (ret == UBSM_OK || ret == UBSM_ERR_NOT_FOUND) return UBSM_OK;
        if (ret == UBSM_ERR_IN_USING && i < max_retries) {
            usleep(100000 * (1 << i));
            continue;
        }
        return ret;
    }
    return UBSM_ERR_IN_USING;
}

static std::string gather_shm_name() {
    return "cyytest_hnsw_base_" + std::to_string(g_job_id);
}

// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 

struct alignas(64) QuerySlotHeader {
    std::atomic<uint64_t> ready_count;
    uint8_t padding[56];
};

static constexpr size_t SLOT_HEADER_SIZE = 64;

static size_t compute_rank_result_size(int top_k) {
    size_t raw = (size_t)top_k * sizeof(faiss::idx_t) + (size_t)top_k * sizeof(float);
    return (raw + 7) & ~(size_t)7;  // 8-byte aligned
}

static size_t compute_slot_size(int top_k, int world_size) {
    return SLOT_HEADER_SIZE + (size_t)world_size * compute_rank_result_size(top_k);
}

static uint8_t* slot_ptr(uint8_t* base, int q, size_t slot_size) {
    return base + (size_t)q * slot_size;
}

static uint8_t* rank_result_ptr(uint8_t* slot_base, int rank, int top_k) {
    return slot_base + SLOT_HEADER_SIZE + (size_t)rank * compute_rank_result_size(top_k);
}

// ===================================================
// Timer & stats
// ===================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    static double duration_ms(std::chrono::high_resolution_clock::time_point start,
                              std::chrono::high_resolution_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

struct PerformanceStats {
    double qps = 0;
    double avg_latency = 0;
    double p50_latency = 0;
    double p95_latency = 0;
    double p99_latency = 0;
    double recall = 0;

    void print(const std::string& name) const {
        std::cout << "\n========== " << name << " ==========" << std::endl;
        std::cout << "QPS:           " << qps << std::endl;
        std::cout << "Avg latency:   " << avg_latency << " ms" << std::endl;
        std::cout << "P50 latency:   " << p50_latency << " ms" << std::endl;
        std::cout << "P95 latency:   " << p95_latency << " ms" << std::endl;
        std::cout << "P99 latency:   " << p99_latency << " ms" << std::endl;
        std::cout << "Recall@k:      " << (recall * 100) << "%" << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
};

struct ThreadSearchContext {
    std::unique_ptr<faiss::VisitedTable> vt;
    std::unique_ptr<faiss::DistanceComputer> dis;
    std::vector<faiss::idx_t> local_I;
    std::vector<float> local_D;

    void init(int64_t ntotal, faiss::IndexHNSWFlat& index, int top_k) {
        vt = std::make_unique<faiss::VisitedTable>(ntotal);
        dis.reset(index.get_distance_computer());
        local_I.resize(top_k);
        local_D.resize(top_k);
    }
};

void search_one_prealloc(
    const faiss::IndexHNSWFlat* index,
    ThreadSearchContext& ctx,
    const float* query,
    int k,
    float* distances,
    faiss::idx_t* labels)
{
    ctx.vt->advance();
    ctx.dis->set_query(query);

    faiss::HeapBlockResultHandler<faiss::HNSW::C> block_handler(1, distances, labels, k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res(block_handler);
    res.begin(0);

    index->hnsw.search(*ctx.dis, index, res, *ctx.vt, nullptr);

    res.end();
}

// =====================================
// fbin
// =====================================

void get_fbin_info(const std::string& filename, int64_t& num_vectors, int& dim) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open fbin file: " + filename);

    uint32_t n, d;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    ifs.read(reinterpret_cast<char*>(&d), sizeof(d));

    num_vectors = n;
    dim = d;
}

void load_fbin_shard(
    const std::string& filename,
    std::vector<float>& data,
    int64_t start_vector,
    int64_t num_vectors_to_load,
    int dim,
    int rank)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open fbin file: " + filename);

    size_t offset = 8 + (size_t)start_vector * dim * sizeof(float);
    ifs.seekg(offset, std::ios::beg);

    if (!ifs) {
        throw std::runtime_error("Seek failed for rank " + std::to_string(rank));
    }

    data.resize((size_t)num_vectors_to_load * dim);
    ifs.read(reinterpret_cast<char*>(data.data()),
             (size_t)num_vectors_to_load * dim * sizeof(float));

    if (!ifs) {
        throw std::runtime_error("Read failed for rank " + std::to_string(rank) +
                                ", expected " + std::to_string(num_vectors_to_load) + " vectors");
    }
}

void load_fbin_queries(const std::string& filename,
                       std::vector<float>& queries,
                       int& num_queries, int& dim) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open query fbin file: " + filename);

    uint32_t n, d;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    ifs.read(reinterpret_cast<char*>(&d), sizeof(d));

    num_queries = n;
    dim = d;

    queries.resize((size_t)n * d);
    ifs.read(reinterpret_cast<char*>(queries.data()), (size_t)n * d * sizeof(float));
}

void load_groundtruth_bin(const std::string& filename,
                          std::vector<int>& groundtruth,
                          int& num_queries, int& gt_k) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open groundtruth file: " + filename);

    uint32_t nq, k;
    ifs.read(reinterpret_cast<char*>(&nq), sizeof(nq));
    ifs.read(reinterpret_cast<char*>(&k), sizeof(k));

    num_queries = nq;
    gt_k = k;

    groundtruth.resize((size_t)nq * k);
    ifs.read(reinterpret_cast<char*>(groundtruth.data()), (size_t)nq * k * sizeof(int32_t));
}

// ==============================
// HDF5
// ==============================

void load_hdf5_full(const std::string& filename,
                    std::vector<float>& train,
                    std::vector<float>& queries,
                    std::vector<int>& groundtruth,
                    int64_t& num_vectors, int& num_queries, int& dim, int& gt_k) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Cannot open HDF5 file: " + filename);

    auto read_float_dataset = [&](const char* name, std::vector<float>& data, int64_t& rows, int& cols) {
        hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
        hid_t space = H5Dget_space(dataset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        rows = dims[0];
        cols = dims[1];
        data.resize((size_t)rows * cols);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Sclose(space);
        H5Dclose(dataset);
    };

    int train_cols, query_cols;
    int64_t num_queries_64;
    read_float_dataset("train", train, num_vectors, train_cols);
    read_float_dataset("test", queries, num_queries_64, query_cols);
    num_queries = static_cast<int>(num_queries_64);
    dim = train_cols;

    hid_t dataset = H5Dopen2(file, "neighbors", H5P_DEFAULT);
    hid_t space = H5Dget_space(dataset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    gt_k = dims[1];
    groundtruth.resize(dims[0] * dims[1]);
    H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, groundtruth.data());
    H5Sclose(space);
    H5Dclose(dataset);

    H5Fclose(file);
}

void load_hdf5_shard(const std::string& filename,
                     std::vector<float>& data,
                     int64_t start_vector,
                     int64_t num_vectors_to_load,
                     int dim) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Cannot open HDF5 file: " + filename);

    hid_t dataset = H5Dopen2(file, "train", H5P_DEFAULT);
    hid_t file_space = H5Dget_space(dataset);

    hsize_t offset[2] = {(hsize_t)start_vector, 0};
    hsize_t count[2] = {(hsize_t)num_vectors_to_load, (hsize_t)dim};
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, nullptr, count, nullptr);

    hid_t mem_space = H5Screate_simple(2, count, nullptr);

    data.resize((size_t)num_vectors_to_load * dim);
    H5Dread(dataset, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, data.data());

    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dataset);
    H5Fclose(file);
}

void load_hdf5_queries_and_gt(const std::string& filename,
                              std::vector<float>& queries,
                              std::vector<int>& groundtruth,
                              int& num_queries, int& dim, int& gt_k) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Cannot open HDF5 file: " + filename);

    {
        hid_t dataset = H5Dopen2(file, "test", H5P_DEFAULT);
        hid_t space = H5Dget_space(dataset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        num_queries = dims[0];
        dim = dims[1];
        queries.resize((size_t)dims[0] * dims[1]);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, queries.data());
        H5Sclose(space);
        H5Dclose(dataset);
    }

    {
        hid_t dataset = H5Dopen2(file, "neighbors", H5P_DEFAULT);
        hid_t space = H5Dget_space(dataset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        gt_k = dims[1];
        groundtruth.resize(dims[0] * dims[1]);
        H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, groundtruth.data());
        H5Sclose(space);
        H5Dclose(dataset);
    }

    H5Fclose(file);
}

void get_hdf5_info(const std::string& filename, int64_t& num_vectors, int& dim) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("Cannot open HDF5 file: " + filename);

    hid_t dataset = H5Dopen2(file, "train", H5P_DEFAULT);
    hid_t space = H5Dget_space(dataset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    num_vectors = dims[0];
    dim = static_cast<int>(dims[1]);

    H5Sclose(space);
    H5Dclose(dataset);
    H5Fclose(file);
}

template<typename T>
void chunked_bcast(T* data, size_t total_count, int root, MPI_Comm comm, MPI_Datatype dtype) {
    const size_t max_chunk = 500000000ULL;

    size_t offset = 0;
    while (offset < total_count) {
        size_t chunk_size = std::min(max_chunk, total_count - offset);
        MPI_Bcast(data + offset, static_cast<int>(chunk_size), dtype, root, comm);
        offset += chunk_size;
    }
}

void bcast_large_float(float* data, size_t total_count, int root, MPI_Comm comm) {
    chunked_bcast(data, total_count, root, comm, MPI_FLOAT);
}

void bcast_large_int(int* data, size_t total_count, int root, MPI_Comm comm) {
    chunked_bcast(data, total_count, root, comm, MPI_INT);
}

// ==================================================
// performance statistics
// ==================================================

PerformanceStats calculate_stats(const std::vector<double>& latencies, double total_time) {
    PerformanceStats stats;
    if (latencies.empty()) return stats;

    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    int n = sorted.size();
    stats.qps = n * 1000.0 / total_time;
    stats.avg_latency = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
    stats.p50_latency = sorted[n * 50 / 100];
    stats.p95_latency = sorted[n * 95 / 100];
    stats.p99_latency = sorted[n * 99 / 100];

    return stats;
}

double calculate_recall(const std::vector<faiss::idx_t>& results,
                        const std::vector<int>& groundtruth,
                        int num_queries, int k, int gt_k) {
    int64_t hits = 0;
    for (int q = 0; q < num_queries; ++q) {
        std::unordered_set<int> gt_set;
        for (int i = 0; i < std::min(k, gt_k); ++i) {
            gt_set.insert(groundtruth[q * gt_k + i]);
        }
        for (int i = 0; i < k; ++i) {
            if (gt_set.count(results[q * k + i])) {
                hits++;
            }
        }
    }
    return (double)hits / (num_queries * std::min(k, gt_k));
}

// ======================================================
// search - UBS shared memory gather
// ======================================================

PerformanceStats run_single_benchmark(
    const std::vector<float>& queries,
    int num_queries, int dim, int top_k,
    int rank, int world_size, int64_t local_start,
    faiss::IndexHNSWFlat& local_index,
    std::vector<ThreadSearchContext>& thread_contexts,
    uint8_t* gather_shm_base,
    size_t slot_size,
    std::vector<faiss::idx_t>& final_indices,
    std::vector<float>& final_distances,
    std::vector<double>& latencies)
{
    Timer total_timer;
    total_timer.start();

    size_t rank_result_size = compute_rank_result_size(top_k);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadSearchContext& ctx = thread_contexts[tid];

        #pragma omp for schedule(static)
        for (int q = 0; q < num_queries; ++q) {
            auto start_time = std::chrono::high_resolution_clock::now();

            const float* query = queries.data() + q * dim;

            search_one_prealloc(&local_index, ctx, query, top_k,
                                ctx.local_D.data(), ctx.local_I.data());

            // Adjust local indices to global IDs
            for (int j = 0; j < top_k; ++j) {
                if (ctx.local_I[j] >= 0) {
                    ctx.local_I[j] += local_start;
                }
            }

            // Write result to gather shared memory
            uint8_t* slot_base = slot_ptr(gather_shm_base, q, slot_size);
            uint8_t* my_result = rank_result_ptr(slot_base, rank, top_k);

            memcpy(my_result, ctx.local_I.data(), top_k * sizeof(faiss::idx_t));
            memcpy(my_result + top_k * sizeof(faiss::idx_t),
                   ctx.local_D.data(), top_k * sizeof(float));

            // Signal completion: release-store ensures memcpy is visible before ready_count
            auto* header = reinterpret_cast<QuerySlotHeader*>(slot_base);
            header->ready_count.fetch_add(1, std::memory_order_release);

            // Rank 0: wait for all ranks, merge results
            if (rank == 0) {
                // Spin-wait with pause for all ranks to complete
                while (header->ready_count.load(std::memory_order_acquire)
                       < static_cast<uint64_t>(world_size)) {
#if defined(__i386__) || defined(__x86_64__)
                    _mm_pause();
#elif defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
#else
                    std::this_thread::yield();
#endif
                }

                // Merge all ranks' results
                std::vector<std::pair<float, faiss::idx_t>> candidates;
                candidates.reserve(top_k * world_size);

                for (int p = 0; p < world_size; ++p) {
                    const uint8_t* ptr = rank_result_ptr(slot_base, p, top_k);
                    const faiss::idx_t* pI = reinterpret_cast<const faiss::idx_t*>(ptr);
                    const float* pD = reinterpret_cast<const float*>(
                        ptr + top_k * sizeof(faiss::idx_t));
                    for (int j = 0; j < top_k; ++j) {
                        if (pI[j] >= 0) {
                            candidates.emplace_back(pD[j], pI[j]);
                        }
                    }
                }

                std::partial_sort(candidates.begin(),
                                  candidates.begin() + std::min(top_k, (int)candidates.size()),
                                  candidates.end());

                for (int j = 0; j < top_k; ++j) {
                    if (j < (int)candidates.size()) {
                        final_distances[q * top_k + j] = candidates[j].first;
                        final_indices[q * top_k + j] = candidates[j].second;
                    } else {
                        final_distances[q * top_k + j] = std::numeric_limits<float>::max();
                        final_indices[q * top_k + j] = -1;
                    }
                }

            }
            auto end_time = std::chrono::high_resolution_clock::now();
            latencies[q] = Timer::duration_ms(start_time, end_time);
        }
    }

    double total_time = total_timer.elapsed_ms();
    return calculate_stats(latencies, total_time);
}

// =====================================================
// main
// =====================================================

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  HDF5 mode:  mpirun -np N " << prog << " -f <data.hdf5> [options]\n"
              << "  fbin mode:  mpirun -np N " << prog << " -base <base.fbin> -query <query.fbin> -gt <groundtruth.bin> [options]\n"
              << "\nOptions:\n"
              << "  -tpp <num>     Threads per process (default: 24)\n"
              << "  -M <num>       HNSW M parameter (default: 32)\n"
              << "  -efC <num>     efConstruction (default: 40)\n"
              << "  -efS <num>     efSearch (default: 64)\n"
              << "  -k <num>       Top-k (default: 10)\n"
              << "  -runs <num>    Number of benchmark runs (default: 5)\n"
              << "  -index <path>  Index file prefix (enables save/load, skip rebuild)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // FUNNELED is sufficient — MPI calls only from main thread, not from OpenMP workers
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error: MPI doesn't support MPI_THREAD_FUNNELED" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string hdf5_file;
    std::string base_file;
    std::string query_file;
    std::string gt_file;
    int tpp = 24;
    int hnsw_M = 32;
    int hnsw_efConstruction = 40;
    int hnsw_efSearch = 64;
    int top_k = 10;
    int num_runs = 5;
    std::string index_prefix;

    printf("Rank %d: 开始加载 HNSW 分片...\n", rank); fflush(stdout);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i+1 < argc) hdf5_file = argv[++i];
        else if (arg == "-base" && i+1 < argc) base_file = argv[++i];
        else if (arg == "-query" && i+1 < argc) query_file = argv[++i];
        else if (arg == "-gt" && i+1 < argc) gt_file = argv[++i];
        else if (arg == "-tpp" && i+1 < argc) tpp = std::stoi(argv[++i]);
        else if (arg == "-M" && i+1 < argc) hnsw_M = std::stoi(argv[++i]);
        else if (arg == "-efC" && i+1 < argc) hnsw_efConstruction = std::stoi(argv[++i]);
        else if (arg == "-efS" && i+1 < argc) hnsw_efSearch = std::stoi(argv[++i]);
        else if (arg == "-k" && i+1 < argc) top_k = std::stoi(argv[++i]);
        else if (arg == "-runs" && i+1 < argc) num_runs = std::stoi(argv[++i]);
        else if (arg == "-index" && i+1 < argc) index_prefix = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    bool use_fbin = !base_file.empty();
    bool use_hdf5 = !hdf5_file.empty();

    if (!use_fbin && !use_hdf5) {
        if (rank == 0) {
            std::cerr << "Error: Must specify either -f (HDF5) or -base/-query/-gt (fbin)\n";
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (use_fbin && (query_file.empty() || gt_file.empty())) {
        if (rank == 0) {
            std::cerr << "Error: fbin mode requires -base, -query, and -gt\n";
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    omp_set_num_threads(tpp);
    omp_set_nested(0);

    // ========================================================================
    // UBS initialization
    // ========================================================================
    {
        ubsmem_options_t opts{};
        int ubsm_ret = ubsmem_init_attributes(&opts);
        if (ubsm_ret != UBSM_OK) {
            fprintf(stderr, "Rank %d: ubsmem_init_attributes failed: %d\n", rank, ubsm_ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ubsm_ret = ubsmem_initialize(&opts);
        if (ubsm_ret != UBSM_OK) {
            fprintf(stderr, "Rank %d: ubsmem_initialize failed: %d\n", rank, ubsm_ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Provider for NUMA-local allocation
    {
        memset(&g_provider, 0, sizeof(g_provider));
        if (gethostname(g_provider.host_name, sizeof(g_provider.host_name) - 1) != 0) {
            fprintf(stderr, "Rank %d: gethostname failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        g_provider.socket_id = UINT32_MAX;  // auto-detect
        g_provider.port_id   = UINT32_MAX;  // auto-detect
#ifdef __linux__
        int cpu = sched_getcpu();
        if (cpu < 0) {
            fprintf(stderr, "Rank %d: sched_getcpu failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int numa = numa_node_of_cpu(cpu);
        if (numa < 0) {
            fprintf(stderr, "Rank %d: numa_node_of_cpu(%d) failed\n", rank, cpu);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        g_provider.numa_id = static_cast<uint32_t>(numa);
#else
        g_provider.numa_id = 0;
#endif
        printf("  Rank %d: UBS provider: host=%s, numa_id=%u\n",
               rank, g_provider.host_name, g_provider.numa_id);
    }

    // Broadcast job_id (rank 0 PID) for unique shm naming
    {
        uint32_t my_pid = static_cast<uint32_t>(getpid());
        MPI_Bcast(&my_pid, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        g_job_id = my_pid;
    }

    // ========================================================================
    // Data loading (unchanged from original)
    // ========================================================================

    int64_t num_vectors = 0;
    int dim = 0;
    int num_queries = 0;
    int gt_k = 0;

    if (rank == 0) {
        if (use_fbin) {
            get_fbin_info(base_file, num_vectors, dim);
        } else {
            get_hdf5_info(hdf5_file, num_vectors, dim);
        }

        std::cout << "\n====== Sharded HNSW Benchmark (UBS Gather) ======" << std::endl;
        std::cout << "Mode: " << (use_fbin ? "fbin (BigANN)" : "HDF5") << std::endl;
        std::cout << "Dataset: " << num_vectors << " vectors, dim=" << dim << std::endl;

        size_t db_size_gb = (size_t)num_vectors * dim * sizeof(float) / (1024ULL * 1024 * 1024);
        std::cout << "Database size: ~" << db_size_gb << " GB" << std::endl;
        std::cout << "Processes: " << world_size << ", Threads/proc: " << tpp << std::endl;
        std::cout << "HNSW: M=" << hnsw_M << ", efC=" << hnsw_efConstruction
                  << ", efS=" << hnsw_efSearch << std::endl;
        std::cout << "Gather: UBS shared memory (no MPI_Gather in hot path)" << std::endl;
    }

    MPI_Bcast(&num_vectors, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int64_t vectors_per_shard = num_vectors / world_size;
    int64_t local_start = rank * vectors_per_shard;
    int64_t local_count = (rank == world_size - 1) ? (num_vectors - local_start) : vectors_per_shard;

    if (rank == 0) {
        std::cout << "\nShard distribution:" << std::endl;
        for (int r = 0; r < world_size; ++r) {
            int64_t r_start = r * vectors_per_shard;
            int64_t r_count = (r == world_size - 1) ? (num_vectors - r_start) : vectors_per_shard;
            size_t shard_size_mb = (size_t)r_count * dim * sizeof(float) / (1024 * 1024);
            std::cout << "  Rank " << r << ": vectors [" << r_start << ", "
                      << (r_start + r_count) << "), count=" << r_count
                      << " (~" << shard_size_mb << " MB)" << std::endl;
        }
    }

    std::vector<float> local_database;
    std::vector<float> queries;
    std::vector<int> groundtruth;

    Timer load_timer;
    load_timer.start();

    if (use_fbin) {
        if (rank == 0) {
            std::cout << "\nLoading data (sharded direct read)..." << std::endl;
        }

        load_fbin_shard(base_file, local_database, local_start, local_count, dim, rank);

        std::cout << "  Rank " << rank << ": loaded " << local_count << " vectors directly" << std::endl;

        if (rank == 0) {
            int query_dim;
            load_fbin_queries(query_file, queries, num_queries, query_dim);
            load_groundtruth_bin(gt_file, groundtruth, num_queries, gt_k);
            std::cout << "  Rank 0: loaded " << num_queries << " queries, gt_k=" << gt_k << std::endl;
        }

        MPI_Bcast(&num_queries, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&gt_k, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            queries.resize((size_t)num_queries * dim);
            groundtruth.resize((size_t)num_queries * gt_k);
        }

        bcast_large_float(queries.data(), (size_t)num_queries * dim, 0, MPI_COMM_WORLD);
        bcast_large_int(groundtruth.data(), (size_t)num_queries * gt_k, 0, MPI_COMM_WORLD);

    } else {
        if (rank == 0) {
            std::cout << "\nLoading data (HDF5 sharded read)..." << std::endl;
        }

        load_hdf5_shard(hdf5_file, local_database, local_start, local_count, dim);

        std::cout << "  Rank " << rank << ": loaded " << local_count << " vectors" << std::endl;

        if (rank == 0) {
            load_hdf5_queries_and_gt(hdf5_file, queries, groundtruth, num_queries, dim, gt_k);
            std::cout << "  Rank 0: loaded " << num_queries << " queries, gt_k=" << gt_k << std::endl;
        }

        MPI_Bcast(&num_queries, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&gt_k, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            queries.resize((size_t)num_queries * dim);
            groundtruth.resize((size_t)num_queries * gt_k);
        }

        bcast_large_float(queries.data(), (size_t)num_queries * dim, 0, MPI_COMM_WORLD);
        bcast_large_int(groundtruth.data(), (size_t)num_queries * gt_k, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double load_time = load_timer.elapsed_ms();
    if (rank == 0) {
        std::cout << "Data loading time: " << load_time / 1000.0 << " seconds" << std::endl;
    }

    // =====================================================
    // HNSW index building (unchanged)
    // =====================================================

    std::string index_file;
    bool use_index_file = !index_prefix.empty();
    if (use_index_file) {
        index_file = index_prefix + "_rank" + std::to_string(rank) + ".index";
    }

    std::unique_ptr<faiss::IndexHNSWFlat> local_index;

    Timer build_timer;
    build_timer.start();

    if (use_index_file) {
        std::ifstream test_ifs(index_file, std::ios::binary);
        if (test_ifs.good()) {
            test_ifs.close();
            if (rank == 0) std::cout << "\nLoading local HNSW indexes from disk..." << std::endl;

            faiss::Index* loaded = faiss::read_index(index_file.c_str());
            faiss::IndexHNSWFlat* loaded_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(loaded);
            if (!loaded_hnsw) {
                std::cerr << "Rank " << rank << ": loaded index is not IndexHNSWFlat, falling back to build" << std::endl;
                delete loaded;
                local_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_M);
                local_index->hnsw.efConstruction = hnsw_efConstruction;
                local_index->hnsw.efSearch = hnsw_efSearch;
                local_index->add(local_count, local_database.data());
            } else {
                local_index.reset(loaded_hnsw);
                local_index->hnsw.efSearch = hnsw_efSearch;
                std::cout << "  Rank " << rank << ": loaded index from " << index_file
                          << " (" << local_index->ntotal << " vectors)" << std::endl;
            }
        } else {
            test_ifs.close();
            if (rank == 0) std::cout << "\nIndex file not found, building local HNSW indexes..." << std::endl;

            local_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_M);
            local_index->hnsw.efConstruction = hnsw_efConstruction;
            local_index->hnsw.efSearch = hnsw_efSearch;
            local_index->add(local_count, local_database.data());

            faiss::write_index(local_index.get(), index_file.c_str());
            std::cout << "  Rank " << rank << ": built and saved index to " << index_file << std::endl;
        }
    } else {
        if (rank == 0) std::cout << "\nBuilding local HNSW indexes..." << std::endl;

        local_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_M);
        local_index->hnsw.efConstruction = hnsw_efConstruction;
        local_index->hnsw.efSearch = hnsw_efSearch;
        local_index->add(local_count, local_database.data());
    }

    double local_build_time = build_timer.elapsed_ms();

    double max_build_time;
    MPI_Reduce(&local_build_time, &max_build_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::cout << "  Rank " << rank << ": " << local_count << " vectors, "
              << "build time=" << local_build_time / 1000.0 << " s" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Max build time: " << max_build_time / 1000.0 << " s" << std::endl;
    }

    local_database.clear();
    local_database.shrink_to_fit();

    // ========================================================================
    // Thread search contexts (simplified — no MPI gather buffers)
    // ========================================================================

    if (rank == 0) {
        std::cout << "\nAllocating thread search contexts..." << std::endl;
        std::cout << "  VisitedTable per thread: " << (local_count / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total for " << tpp << " threads: " << (tpp * local_count / 1024.0 / 1024.0) << " MB" << std::endl;
    }

    std::vector<ThreadSearchContext> thread_contexts(tpp);
    for (int t = 0; t < tpp; ++t) {
        thread_contexts[t].init(local_count, *local_index, top_k);
    }

    if (rank == 0) std::cout << "Thread contexts initialized." << std::endl;

    // ========================================================================
    // Allocate and map UBS gather shared memory
    // ========================================================================

    size_t slot_size = compute_slot_size(top_k, world_size);
    size_t total_gather_bytes = (size_t)num_queries * slot_size;
    size_t gather_alloc_size = ubsmem_align_size(total_gather_bytes);

    std::string gather_name = gather_shm_name();
    uint8_t* gather_shm_base = nullptr;

    if (rank == 0) {
        // Rank 0 allocates the gather shm
        ubsmem_deallocate_retry(gather_name.c_str());

        int ret = ubsmem_shmem_allocate_with_provider(
            &g_provider, gather_name.c_str(), gather_alloc_size, 0600,
            UBSM_FLAG_NONCACHE | UBSM_FLAG_WR_DELAY_COMP);
        if (ret != UBSM_OK) {
            fprintf(stderr, "Rank 0: ubsmem_shmem_allocate_with_provider('%s', %zu) failed: %d\n",
                    gather_name.c_str(), gather_alloc_size, ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        ret = ubsmem_shmem_map(nullptr, gather_alloc_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, gather_name.c_str(), 0,
                               reinterpret_cast<void**>(&gather_shm_base));
        if (ret != UBSM_OK || gather_shm_base == nullptr) {
            fprintf(stderr, "Rank 0: ubsmem_shmem_map('%s', %zu) failed: %d\n",
                    gather_name.c_str(), gather_alloc_size, ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Zero the entire region
        memset(gather_shm_base, 0, gather_alloc_size);

        printf("  Rank 0: UBS gather shm '%s' allocated (%zu MB, %d queries, %d ranks, slot=%zu bytes)\n",
               gather_name.c_str(), gather_alloc_size / (1024 * 1024),
               num_queries, world_size, slot_size);
    }

    // Barrier: ensure rank 0 has created the shm before other ranks map it
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0) {
        // Other ranks map the gather shm
        int ret = ubsmem_shmem_map(nullptr, gather_alloc_size, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, gather_name.c_str(), 0,
                                   reinterpret_cast<void**>(&gather_shm_base));
        if (ret != UBSM_OK || gather_shm_base == nullptr) {
            fprintf(stderr, "Rank %d: ubsmem_shmem_map('%s', %zu) failed: %d\n",
                    rank, gather_name.c_str(), gather_alloc_size, ret);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("  Rank %d: UBS gather shm '%s' mapped (%zu MB)\n",
               rank, gather_name.c_str(), gather_alloc_size / (1024 * 1024));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ========================================================================
    // Benchmark runs
    // ========================================================================

    if (rank == 0) std::cout << "\nRunning " << num_runs << " benchmark iterations..." << std::endl;

    std::vector<double> all_qps(num_runs);
    std::vector<double> all_p50(num_runs);
    std::vector<double> all_p99(num_runs);
    std::vector<double> all_recall(num_runs);

    std::vector<faiss::idx_t> final_indices((size_t)num_queries * top_k);
    std::vector<float> final_distances((size_t)num_queries * top_k);
    std::vector<double> latencies(num_queries);

    for (int run = 0; run < num_runs; ++run) {
        // Reset all ready_counts to 0 before each run
        if (rank == 0) {
            for (int q = 0; q < num_queries; ++q) {
                auto* header = reinterpret_cast<QuerySlotHeader*>(
                    slot_ptr(gather_shm_base, q, slot_size));
                header->ready_count.store(0, std::memory_order_relaxed);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        PerformanceStats stats = run_single_benchmark(
            queries, num_queries, dim, top_k,
            rank, world_size, local_start,
            *local_index, thread_contexts,
            gather_shm_base, slot_size,
            final_indices, final_distances, latencies);

        double recall = calculate_recall(final_indices, groundtruth, num_queries, top_k, gt_k);

        all_qps[run] = stats.qps;
        all_p50[run] = stats.p50_latency;
        all_p99[run] = stats.p99_latency;
        all_recall[run] = recall;

        if (rank == 0) {
            std::cout << "  Run " << run << ": QPS=" << stats.qps
                      << ", P50=" << stats.p50_latency << " ms"
                      << ", P99=" << stats.p99_latency << " ms"
                      << ", Recall=" << (recall * 100) << "%" << std::endl;
        }
    }

        std::sort(all_qps.begin(), all_qps.end());
        std::sort(all_p50.begin(), all_p50.end());
        std::sort(all_p99.begin(), all_p99.end());
        std::sort(all_recall.begin(), all_recall.end());

        double qps_median = all_qps[num_runs / 2];
        double qps_min = all_qps[0];
        double qps_max = all_qps[num_runs - 1];
        double qps_mean = std::accumulate(all_qps.begin(), all_qps.end(), 0.0) / num_runs;

        double variance = 0;
        for (double q : all_qps) variance += (q - qps_mean) * (q - qps_mean);
        double qps_stddev = (num_runs > 1) ? std::sqrt(variance / (num_runs - 1)) : 0;

        double recall_mean = std::accumulate(all_recall.begin(), all_recall.end(), 0.0) / num_runs;
        double recall_median = all_recall[num_runs / 2];

    if (rank == 0) {
        std::cout << "\n========== Final Statistics ==========" << std::endl;
        std::cout << "QPS:" << std::endl;
        std::cout << "  Min:    " << qps_min << std::endl;
        std::cout << "  Max:    " << qps_max << std::endl;
        std::cout << "  Mean:   " << qps_mean << std::endl;
        std::cout << "  Median: " << qps_median << " (recommended)" << std::endl;
        std::cout << "  StdDev: " << qps_stddev << std::endl;
        std::cout << "\nLatency:" << std::endl;
        std::cout << "  P50 (median run): " << all_p50[num_runs / 2] << " ms" << std::endl;
        std::cout << "  P99 (median run): " << all_p99[num_runs / 2] << " ms" << std::endl;
        std::cout << "\nRecall@" << top_k << ":" << std::endl;
        std::cout << "  Mean:   " << (recall_mean * 100) << "%" << std::endl;
        std::cout << "  Median: " << (recall_median * 100) << "%" << std::endl;
        std::cout << "=========================================\n" << std::endl;
    }
        std::cout << "rank " << rank << "  P50 (median run): " << all_p50[num_runs / 2] << " ms" << std::endl;
        std::cout << "rank " << rank  << "  P99 (median run): " << all_p99[num_runs / 2] << " ms" << std::endl;
    // ========================================================================
    // Cleanup UBS gather shm
    // ========================================================================

    MPI_Barrier(MPI_COMM_WORLD);

    ubsmem_shmem_unmap(gather_shm_base, gather_alloc_size);

    if (rank == 0) {
        ubsmem_deallocate_retry(gather_name.c_str());
        printf("  Rank 0: UBS gather shm deallocated.\n");
    }

    ubsmem_finalize();
    MPI_Finalize();
    return 0;
}
