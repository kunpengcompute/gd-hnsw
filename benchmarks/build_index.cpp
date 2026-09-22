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

// build_index.cpp — standalone CLI wrapper around build_sharded_index().
//
// Builds and persists a sharded HNSW index to disk (output_path.shard_N) so
// each container can later load its own shard via Context::load.  No MPI,
// no ubs-mem, no NUMA — links only the build layer (gd_hnsw_build).
//
// Configuration can come from the same key=value file as gd_hnsw_bench
// (--config, CLI flags override). Shared keys: dataset / index / ef-search /
// dataset-format / base-bin. Build-only keys carry a build.* prefix
// (build.M, build.ef-construction, build.shards, build.threads, build.cluster).
//
// Usage:
//   ./build_index -d /data/sift.h5 -o /data/idx -M 16 -c 200 -e 64 -s 4 -t 192
//   ./build_index --config configs/gist-960-euclidean.config
//   # add --cluster for k-means shard partition (also writes idmap + centroids)

#include "build.h"
#include "index_io.h"
#include "config_parser.h"

#include <getopt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace gd_hnsw;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Standalone HNSW index builder. Builds and saves sharded index files\n"
            "that can be loaded by gd_hnsw_bench / Context::load.\n"
            "\n"
            "  -d, --dataset PATH        Dataset path (HDF5 or .bin directory/file)\n"
            "  -o, --output PATH         Output index path (writes PATH.shard_N files)\n"
            "  -M, --M INT               HNSW M parameter (default: 16)\n"
            "  -c, --ef-construction INT  ef_construction (default: 200)\n"
            "  -e, --ef-search INT        ef_search stored in index (default: 64)\n"
            "  -s, --shards INT           Number of shards, in [1, 32] (default: 1)\n"
            "  -t, --threads INT          OpenMP threads for building (default: all cores)\n"
            "      --cluster              Use k-means clustered shard partition\n"
            "      --dataset-format FMT   auto|hdf5|bin (default: auto)\n"
            "      --base-bin PATH        Override base vectors file (bin format)\n"
            "      --config PATH          key=value config file (CLI flags override;\n"
            "                              shared with gd_hnsw_bench, see configs/)\n"
            "  -h, --help                 Show this help\n"
            "\n"
            "Example:\n"
            "  %s -d /data/sift1b.h5 -M 32 -c 200 -s 4 -t 192 -o /data/idx\n"
            "  mpirun -np 4 ./gd_hnsw_bench --load-index /data/idx -d /data/sift1b.h5\n",
            prog, prog);
}

// Parse a uint32 CLI argument. Rejects garbage and negatives instead of
// wrapping: static_cast<uint32_t>(std::stoi("-1")) == 4294967295 would
// silently bypass the ef_search >= 1 check below. Throws std::runtime_error.
static uint32_t parse_u32_arg(const char *opt, const char *arg)
{
    long long v = 0;
    size_t pos = 0;
    try {
        v = std::stoll(arg, &pos);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(opt) + " expects an integer, got '" + arg + "'");
    }
    if (pos != std::strlen(arg) || v < 0 || v > UINT32_MAX)
        throw std::runtime_error(std::string(opt) + " expects an integer in [0, 4294967295], got '" + arg + "'");
    return static_cast<uint32_t>(v);
}

static void apply_config(const ConfigParser &cfg, BuildOptions &opts, int &num_shards, int &num_threads)
{
    // Shared keys (also read by gd_hnsw_bench).
    if (cfg.has("dataset"))
        opts.dataset_path = cfg.get_str("dataset");
    if (cfg.has("index"))
        opts.output_path = cfg.get_str("index");
    if (cfg.has("ef-search"))
        opts.ef_search = cfg.get_uint("ef-search", opts.ef_search);
    if (cfg.has("dataset-format"))
        opts.dataset_format = cfg.get_str("dataset-format", opts.dataset_format);
    if (cfg.has("base-bin"))
        opts.bin_base_path = cfg.get_str("base-bin", opts.bin_base_path);
    // Build-only keys.
    if (cfg.has("build.M"))
        opts.M = cfg.get_uint("build.M", opts.M);
    if (cfg.has("build.ef-construction"))
        opts.ef_construction = cfg.get_uint("build.ef-construction", opts.ef_construction);
    if (cfg.has("build.shards"))
        num_shards = cfg.get_int("build.shards", num_shards);
    if (cfg.has("build.threads"))
        num_threads = cfg.get_int("build.threads", num_threads);
    if (cfg.has("build.cluster"))
        opts.cluster_partition = cfg.get_bool("build.cluster", opts.cluster_partition);
}

int main(int argc, char **argv)
{
    BuildOptions opts;
    int num_shards = 1;
    int num_threads = 0; // 0 = use all available

    // ---- (0) locate --config before full parsing (config is baseline, CLI overrides) ----
    std::string config_path;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[i + 1];
            break;
        }
        if (std::strncmp(argv[i], "--config=", 9) == 0) {
            config_path = argv[i] + 9;
            break;
        }
    }
    if (!config_path.empty()) {
        ConfigParser cfg(config_path);
        if (!cfg.ok()) {
            fprintf(stderr, "Error: cannot read config file '%s'\n", config_path.c_str());
            return 1;
        }
        try {
            apply_config(cfg, opts, num_shards, num_threads);
        } catch (const std::exception &e) {
            fprintf(stderr, "Error: %s\n", e.what());
            return 1;
        }
    }

    static struct option long_opts[] = {{"dataset", required_argument, nullptr, 'd'},
                                        {"output", required_argument, nullptr, 'o'},
                                        {"M", required_argument, nullptr, 'M'},
                                        {"ef-construction", required_argument, nullptr, 'c'},
                                        {"ef-search", required_argument, nullptr, 'e'},
                                        {"shards", required_argument, nullptr, 's'},
                                        {"threads", required_argument, nullptr, 't'},
                                        {"cluster", no_argument, nullptr, 1007},
                                        {"dataset-format", required_argument, nullptr, 1015},
                                        {"base-bin", required_argument, nullptr, 1016},
                                        {"config", required_argument, nullptr, 1017},
                                        {"help", no_argument, nullptr, 'h'},
                                        {nullptr, 0, nullptr, 0}};

    int opt;
    try {
        while ((opt = getopt_long(argc, argv, "d:o:M:c:e:s:t:h", long_opts, nullptr)) != -1) {
            switch (opt) {
                case 'd': opts.dataset_path = optarg; break;
                case 'o': opts.output_path = optarg; break;
                case 'M': opts.M = parse_u32_arg("--M", optarg); break;
                case 'c': opts.ef_construction = parse_u32_arg("--ef-construction", optarg); break;
                case 'e': opts.ef_search = parse_u32_arg("--ef-search", optarg); break;
                case 's': num_shards = std::stoi(optarg); break;
                case 't': num_threads = std::stoi(optarg); break;
                case 1007: opts.cluster_partition = true; break;
                case 1015: opts.dataset_format = optarg; break;
                case 1016: opts.bin_base_path = optarg; break;
                case 1017: break; // already applied above
                case 'h': print_usage(argv[0]); return 0;
                default: print_usage(argv[0]); return 1;
            }
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    if (opts.dataset_path.empty()) {
        fprintf(stderr, "Error: --dataset is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (opts.output_path.empty()) {
        fprintf(stderr, "Error: --output is required\n");
        print_usage(argv[0]);
        return 1;
    }
    // 32 = pushdown pending_mask (uint32_t bitmask) hard limit; must stay
    // in sync with kMaxShards (gd_hnsw_search.cpp / faiss_extractor.cpp).
    if (num_shards < 1 || num_shards > 32) {
        fprintf(stderr, "Error: --shards must be in [1, 32] (pushdown bitmask limit)\n");
        return 1;
    }
    // ef_search = 0 is the bench-side "use index default" sentinel; storing it
    // in the index would be meaningless, so fail fast here.
    if (opts.ef_search < 1) {
        fprintf(stderr, "Error: ef-search must be >= 1\n");
        return 1;
    }
    opts.num_shards = static_cast<uint32_t>(num_shards);
    opts.build_threads = (num_threads > 0) ? static_cast<uint32_t>(num_threads) : 0;

    // Fail fast: create the output directory up front so a bad -o path is
    // reported before hours of building, not at the final save step.
    {
        std::string dir_err;
        if (!ensure_parent_dir(opts.output_path, dir_err)) {
            fprintf(stderr, "Error: cannot create output directory for '%s': %s\n", opts.output_path.c_str(),
                    dir_err.c_str());
            return 1;
        }
    }

    printf("=== HNSW Index Builder ===\n");
    printf("  Dataset:          %s\n", opts.dataset_path.c_str());
    printf("  Output:           %s\n", opts.output_path.c_str());
    printf("  M:                %u\n", opts.M);
    printf("  ef_construction:  %u\n", opts.ef_construction);
    printf("  ef_search:        %u\n", opts.ef_search);
    printf("  Shards:           %u\n", opts.num_shards);
    printf("  Threads:          %u\n", opts.build_threads);
    printf("  Cluster:          %s\n", opts.cluster_partition ? "ON" : "OFF");
    printf("  Format:           %s\n", opts.dataset_format.c_str());
    printf("\n");

    BuildResult r = build_sharded_index(opts);
    if (r.status != Status::Ok) {
        fprintf(stderr, "Error: build failed: %s\n", r.error.c_str());
        return 1;
    }

    uint64_t total_bytes = 0;
    for (auto sz : r.shard_bytes)
        total_bytes += sz;
    printf("\n=== Done ===\n");
    printf("  Index saved to: %s.shard_0 .. %s.shard_%u\n", opts.output_path.c_str(), opts.output_path.c_str(),
           r.num_shards - 1);
    printf("  ntotal=%lu, dim=%u, total=%.1f MB\n", (unsigned long)r.ntotal, r.dim, total_bytes / (1024.0 * 1024.0));
    printf("  To use: mpirun -np %u ./gd_hnsw_bench --load-index %s -d %s ...\n", r.num_shards,
           opts.output_path.c_str(), opts.dataset_path.c_str());
    return 0;
}
