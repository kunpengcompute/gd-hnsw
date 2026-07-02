#include "faiss_extractor.h"
#include "index_io.h"
#include "hdf5_loader.h"
#include "bin_loader.h"
#include "ann_dataset.h"

#include <omp.h>
#include <getopt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace shm_hnsw;

// ============================================================
// Usage
// ============================================================

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Standalone HNSW index builder. Builds and saves sharded index files\n"
        "that can be loaded by shm_hnsw_bench via --load-index.\n"
        "\n"
        "  -d, --dataset PATH        Dataset path (HDF5 or .bin directory)\n"
        "  -o, --output PATH         Output index path (writes PATH.shard_N files)\n"
        "  -M, --M INT               HNSW M parameter (default: 16)\n"
        "  -c, --ef-construction INT  ef_construction (default: 200)\n"
        "  -e, --ef-search INT        ef_search stored in index (default: 64)\n"
        "  -s, --shards INT           Number of shards to build (default: 1)\n"
        "  -t, --threads INT          OpenMP threads for building (default: all cores)\n"
        "      --cluster              Use k-means clustered shard partition\n"
        "      --dataset-format FMT   Dataset format: auto|hdf5|bin (default: auto)\n"
        "      --base-bin PATH        Override base vectors file (bin format)\n"
        "  -h, --help                 Show this help\n"
        "\n"
        "Example:\n"
        "  %s -d /data/sift1b.h5 -M 32 -c 200 -s 4 -t 192 -o /data/idx\n"
        "  mpirun -np 4 ./shm_hnsw_bench --load-index /data/idx -d /data/sift1b.h5\n",
        prog, prog);
}

// ============================================================
// Dataset format detection
// ============================================================

static DatasetFormat detect_format(const std::string& path, const std::string& fmt_str) {
    if (fmt_str == "hdf5") return DatasetFormat::HDF5;
    if (fmt_str == "bin")  return DatasetFormat::BIN;
    if (fmt_str != "auto") {
        fprintf(stderr, "Error: unknown --dataset-format '%s' (expected: auto|hdf5|bin)\n",
                fmt_str.c_str());
        exit(1);
    }
    DatasetFormat detected = detect_dataset_format(path);
    // Default to HDF5 if unknown
    return (detected == DatasetFormat::UNKNOWN) ? DatasetFormat::HDF5 : detected;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    std::string dataset_path;
    std::string output_path;
    uint32_t M = 16;
    uint32_t ef_construction = 200;
    uint32_t ef_search = 64;
    int num_shards = 1;
    int num_threads = 0; // 0 = use all available
    bool cluster_partition = false;
    std::string dataset_format_str = "auto";
    std::string bin_base_path;

    static struct option long_opts[] = {
        {"dataset",         required_argument, nullptr, 'd'},
        {"output",          required_argument, nullptr, 'o'},
        {"M",               required_argument, nullptr, 'M'},
        {"ef-construction", required_argument, nullptr, 'c'},
        {"ef-search",       required_argument, nullptr, 'e'},
        {"shards",          required_argument, nullptr, 's'},
        {"threads",         required_argument, nullptr, 't'},
        {"cluster",         no_argument,       nullptr, 1007},
        {"dataset-format",  required_argument, nullptr, 1015},
        {"base-bin",        required_argument, nullptr, 1016},
        {"help",            no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:o:M:c:e:s:t:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': dataset_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 'M': M = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 'c': ef_construction = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 'e': ef_search = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 's': num_shards = std::stoi(optarg); break;
            case 't': num_threads = std::stoi(optarg); break;
            case 1007: cluster_partition = true; break;
            case 1015: dataset_format_str = optarg; break;
            case 1016: bin_base_path = optarg; break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    // --- Validate arguments ---
    if (dataset_path.empty()) {
        fprintf(stderr, "Error: --dataset is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (output_path.empty()) {
        fprintf(stderr, "Error: --output is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (num_shards < 1 || num_shards > 64) {
        fprintf(stderr, "Error: --shards must be in [1, 64]\n");
        return 1;
    }

    // --- Set thread count ---
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);  // Prevent runtime from reducing thread count

    printf("=== HNSW Index Builder ===\n");
    printf("  Dataset:          %s\n", dataset_path.c_str());
    printf("  Output:           %s\n", output_path.c_str());
    printf("  M:                %u\n", M);
    printf("  ef_construction:  %u\n", ef_construction);
    printf("  ef_search:        %u\n", ef_search);
    printf("  Shards:           %d\n", num_shards);
    printf("  Threads:          %d\n", num_threads);
    printf("  Cluster:          %s\n", cluster_partition ? "ON" : "OFF");
    printf("\n");

    // --- Load dataset (train vectors only) ---
    printf("[build] Loading dataset (train vectors only)...\n");
    auto t_load_start = std::chrono::steady_clock::now();

    AnnDataset dataset;
    DatasetFormat fmt = detect_format(dataset_path, dataset_format_str);

    if (fmt == DatasetFormat::HDF5) {
        dataset = load_hdf5_dataset(dataset_path, /*eval_only=*/false, /*train_only=*/true);
    } else {
        dataset = load_bin_dataset(dataset_path, /*eval_only=*/false,
                                   bin_base_path, "", "", /*train_only=*/true);
    }

    auto t_load_end = std::chrono::steady_clock::now();
    double load_sec = std::chrono::duration<double>(t_load_end - t_load_start).count();
    printf("[build] Loaded %lu vectors, dim=%u (%.2f sec)\n",
           (unsigned long)dataset.n_train, dataset.dim, load_sec);

    if (dataset.n_train == 0 || dataset.dim == 0) {
        fprintf(stderr, "Error: dataset is empty\n");
        return 1;
    }

    // --- Build index ---
    printf("[build] Building HNSW index (M=%u, ef_construction=%u, shards=%d, threads=%d)...\n",
           M, ef_construction, num_shards, num_threads);
    auto t_build_start = std::chrono::steady_clock::now();

    ShardBuildOptions shard_build_opts;
    shard_build_opts.cluster_partition = cluster_partition;

    // Free train vectors after FAISS copies them internally
    shard_build_opts.post_add_callback = [&]() {
        size_t freed_bytes = dataset.train.size() * sizeof(float);
        dataset.train.clear();
        dataset.train.shrink_to_fit();
        printf("[build] Released dataset.train after FAISS add (~%.1f GiB)\n",
               freed_bytes / (1024.0 * 1024.0 * 1024.0));
    };

    std::vector<uint32_t> new_to_old_idmap;
    std::vector<float> cluster_centroids;

    // Use streaming mode: extract one shard at a time, write to disk immediately
    printf("[build] Using streaming extraction (save_path: %s)\n", output_path.c_str());
    auto shard_sizes = FaissExtractor::build_and_extract_sharded_streaming(
        dataset.train.data(), dataset.n_train, dataset.dim,
        static_cast<uint32_t>(num_shards), output_path,
        M, ef_construction, ef_search,
        shard_build_opts,
        cluster_partition ? &new_to_old_idmap : nullptr,
        cluster_partition ? &cluster_centroids : nullptr);

    auto t_build_end = std::chrono::steady_clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();

    uint64_t total_bytes = 0;
    for (auto sz : shard_sizes) total_bytes += sz;
    printf("[build] Build complete: %.2f sec, total: %.1f MB (%d shards)\n",
           build_sec, total_bytes / (1024.0 * 1024.0), num_shards);

    // --- Save sidecars ---
    if (cluster_partition && !new_to_old_idmap.empty()) {
        if (!save_idmap_file(output_path, new_to_old_idmap)) {
            fprintf(stderr, "Warning: failed to save idmap\n");
        }
    }
    if (cluster_partition && !cluster_centroids.empty()) {
        if (!save_centroids_file(output_path, cluster_centroids,
                                  static_cast<uint32_t>(num_shards), dataset.dim)) {
            fprintf(stderr, "Warning: failed to save centroids\n");
        }
    }

    // --- Lightweight verification (one shard at a time to avoid OOM) ---
    printf("[build] Verifying saved shards...\n");
    for (int s = 0; s < num_shards; s++) {
        std::string path = index_file_for_shard(output_path, static_cast<uint32_t>(s));
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "Error: cannot open '%s' for verification\n", path.c_str());
            return 1;
        }
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        if (file_size <= 0) {
            fprintf(stderr, "Error: '%s' is empty\n", path.c_str());
            fclose(fp);
            return 1;
        }
        std::vector<char> buf(static_cast<size_t>(file_size));
        size_t nread = fread(buf.data(), 1, buf.size(), fp);
        fclose(fp);
        if (nread != buf.size()) {
            fprintf(stderr, "Error: short read from '%s'\n", path.c_str());
            return 1;
        }
        auto result = validate_superblock(buf.data(), static_cast<uint64_t>(file_size));
        if (!result.ok) {
            fprintf(stderr, "Error: '%s' failed validation: %s\n",
                    path.c_str(), result.error.c_str());
            return 1;
        }
        printf("  Shard %d: OK (%.1f MB)\n", s, file_size / (1024.0 * 1024.0));
        // buf freed here — only 1 shard in memory at a time
    }

    // Cross-shard consistency check (streaming — only headers kept)
    if (num_shards > 1) {
        printf("[build] Cross-shard consistency check...\n");
        uint64_t sum_ntotal_local = 0;
        uint32_t expected_dim = 0, expected_M = 0, expected_num_shards = 0;
        uint64_t expected_ntotal = 0;
        int32_t expected_max_level = 0, expected_entry_point = 0;

        for (int s = 0; s < num_shards; s++) {
            std::string path = index_file_for_shard(output_path, static_cast<uint32_t>(s));
            FILE* fp = fopen(path.c_str(), "rb");
            if (!fp) { fprintf(stderr, "Error: cannot reopen '%s'\n", path.c_str()); return 1; }
            SuperBlockV1 sb{};
            if (fread(&sb, sizeof(sb), 1, fp) != 1) {
                fprintf(stderr, "Error: cannot read header from '%s'\n", path.c_str());
                fclose(fp); return 1;
            }
            fclose(fp);

            if (s == 0) {
                expected_ntotal = sb.ntotal;
                expected_dim = sb.dim;
                expected_M = sb.M;
                expected_num_shards = sb.num_shards;
                expected_max_level = sb.max_level;
                expected_entry_point = sb.entry_point;
            } else {
                if (sb.ntotal != expected_ntotal || sb.dim != expected_dim ||
                    sb.M != expected_M || sb.num_shards != expected_num_shards ||
                    sb.max_level != expected_max_level || sb.entry_point != expected_entry_point) {
                    fprintf(stderr, "Error: shard %d has inconsistent parameters with shard 0\n", s);
                    return 1;
                }
            }
            if (sb.shard_id != static_cast<uint32_t>(s)) {
                fprintf(stderr, "Error: shard %d has shard_id=%u\n", s, sb.shard_id);
                return 1;
            }
            if (sb.global_id_begin != sum_ntotal_local) {
                fprintf(stderr, "Error: shard %d global_id_begin=%lu, expected %lu\n",
                        s, (unsigned long)sb.global_id_begin, (unsigned long)sum_ntotal_local);
                return 1;
            }
            sum_ntotal_local += sb.ntotal_local;
        }
        if (sum_ntotal_local != expected_ntotal) {
            fprintf(stderr, "Error: sum of ntotal_local (%lu) != ntotal (%lu)\n",
                    (unsigned long)sum_ntotal_local, (unsigned long)expected_ntotal);
            return 1;
        }
        printf("  Cross-shard: OK (ntotal=%lu, dim=%u, M=%u)\n",
               (unsigned long)expected_ntotal, expected_dim, expected_M);
    }

    printf("[build] Verification passed.\n");

    printf("\n=== Done ===\n");
    printf("  Index saved to: %s.shard_0 .. %s.shard_%d\n",
           output_path.c_str(), output_path.c_str(), num_shards - 1);
    printf("  Total build time: %.2f sec\n", build_sec);
    printf("  To use: mpirun -np %d ./shm_hnsw_bench --load-index %s -d %s ...\n",
           num_shards, output_path.c_str(), dataset_path.c_str());

    return 0;
}
