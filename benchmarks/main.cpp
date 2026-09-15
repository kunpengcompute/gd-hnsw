/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// main.cpp — gd_hnsw_bench: MPI benchmark using the gd_hnsw::Context API.
//
// Deploys N containers (mpirun -np N), loads each node's shard, syncs, routes
// queries, runs warmup + timed rounds, and prints aggregate recall@k, QPS and
// per-query latency percentiles. Configuration can come from a key=value file
// (--config) and/or CLI flags (CLI overrides config). The config file is
// shared with build_index: dataset / index / ef-search / dataset-format /
// base-bin are common keys, build.*-prefixed keys are build-only.
//
// Usage:
//   mpirun -np 4 ./gd_hnsw_bench --config configs/gist-960-euclidean.config
//   mpirun -np 4 ./gd_hnsw_bench --load-index /data/idx --dataset /data/sift.h5 \
//       --k 10 --ef-search 64 --rounds 3 --warmup 1

#include "gd_hnsw_api.h"
#include "hdf5_loader.h"
#include "bin_loader.h"
#include "bench_runner.h"
#include "bench_report.h"
#include "config_parser.h"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <string>
#include <vector>

using namespace gd_hnsw;

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --config PATH        key=value config file (CLI flags override)\n"
            "  --load-index PATH    load sharded index from PATH.shard_<rank> (required)\n"
            "  --dataset PATH       dataset for queries + ground truth (HDF5 or .bin) (required)\n"
            "  --k INT              top-K (default 10)\n"
            "  --ef-search INT      ef_search (0 = use index default)\n"
            "  --num-threads INT    OpenMP threads per process (0 = default)\n"
            "  --search-threads INT  pushdown search threads (0 = auto)\n"
            "  --service-threads INT pushdown service threads (0 = auto)\n"
            "  --expand-batch INT   multi-pop expansion (1..64, default 1)\n"
            "  --use-dot-norm       use dot+norm distance kernel\n"
            "  --cluster-route      route queries by nearest centroid (needs cluster build)\n"
            "  --dataset-format FMT auto|hdf5|bin (default auto)\n"
            "  --base-bin PATH      override base.bin path (bin format)\n"
            "  --rounds INT         timed benchmark rounds (default 1)\n"
            "  --warmup INT         untimed warmup rounds (default 0)\n"
            "  --help               show this help\n",
            prog);
}

static void apply_config(const ConfigParser &cfg, std::string &load_index_path, std::string &dataset_path, int32_t &K,
                         int32_t &ef_search, int &num_threads, uint32_t &search_threads, uint32_t &service_threads,
                         uint32_t &expand_batch, bool &use_dot_norm, bool &cluster_route,
                         std::string &dataset_format_str, std::string &bin_base_path, int &num_rounds, int &num_warmup)
{
    if (cfg.has("index"))
        load_index_path = cfg.get_str("index");
    if (cfg.has("dataset"))
        dataset_path = cfg.get_str("dataset");
    if (cfg.has("k"))
        K = cfg.get_int("k", K);
    if (cfg.has("ef-search"))
        ef_search = cfg.get_int("ef-search", ef_search);
    if (cfg.has("num-threads"))
        num_threads = cfg.get_int("num-threads", num_threads);
    if (cfg.has("search-threads"))
        search_threads = cfg.get_uint("search-threads", search_threads);
    if (cfg.has("service-threads"))
        service_threads = cfg.get_uint("service-threads", service_threads);
    if (cfg.has("expand-batch"))
        expand_batch = cfg.get_uint("expand-batch", expand_batch);
    if (cfg.has("use-dot-norm"))
        use_dot_norm = cfg.get_bool("use-dot-norm", use_dot_norm);
    if (cfg.has("cluster-route"))
        cluster_route = cfg.get_bool("cluster-route", cluster_route);
    if (cfg.has("dataset-format"))
        dataset_format_str = cfg.get_str("dataset-format", dataset_format_str);
    if (cfg.has("base-bin"))
        bin_base_path = cfg.get_str("base-bin", bin_base_path);
    if (cfg.has("rounds"))
        num_rounds = cfg.get_int("rounds", num_rounds);
    if (cfg.has("warmup"))
        num_warmup = cfg.get_int("warmup", num_warmup);
}

int main(int argc, char **argv)
{
    std::string load_index_path;
    std::string dataset_path;
    int32_t K = 10;
    int32_t ef_search = 0;
    int num_threads = 0;
    uint32_t search_threads = 0;
    uint32_t service_threads = 0;
    uint32_t expand_batch = 1;
    bool use_dot_norm = false;
    bool cluster_route = false;
    std::string dataset_format_str = "auto";
    std::string bin_base_path;
    int num_rounds = 1;
    int num_warmup = 0;

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
        apply_config(cfg, load_index_path, dataset_path, K, ef_search, num_threads, search_threads, service_threads,
                     expand_batch, use_dot_norm, cluster_route, dataset_format_str, bin_base_path, num_rounds,
                     num_warmup);
    }

    // ---- (1) CLI parse (overrides config) ----
    static struct option long_opts[] = {{"config", required_argument, nullptr, 1000},
                                        {"load-index", required_argument, nullptr, 1006},
                                        {"dataset", required_argument, nullptr, 'd'},
                                        {"k", required_argument, nullptr, 'k'},
                                        {"ef-search", required_argument, nullptr, 'e'},
                                        {"num-threads", required_argument, nullptr, 't'},
                                        {"search-threads", required_argument, nullptr, 1002},
                                        {"service-threads", required_argument, nullptr, 1003},
                                        {"expand-batch", required_argument, nullptr, 1022},
                                        {"use-dot-norm", no_argument, nullptr, 1019},
                                        {"cluster-route", no_argument, nullptr, 1100},
                                        {"dataset-format", required_argument, nullptr, 1015},
                                        {"base-bin", required_argument, nullptr, 1016},
                                        {"rounds", required_argument, nullptr, 1001},
                                        {"warmup", required_argument, nullptr, 1004},
                                        {"help", no_argument, nullptr, 'h'},
                                        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "d:k:e:t:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 1000: config_path = optarg; break; // already applied above
            case 1006: load_index_path = optarg; break;
            case 'd': dataset_path = optarg; break;
            case 'k': K = std::stoi(optarg); break;
            case 'e': ef_search = std::stoi(optarg); break;
            case 't': num_threads = std::stoi(optarg); break;
            case 1002: search_threads = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 1003: service_threads = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 1022: expand_batch = static_cast<uint32_t>(std::stoi(optarg)); break;
            case 1019: use_dot_norm = true; break;
            case 1100: cluster_route = true; break;
            case 1015: dataset_format_str = optarg; break;
            case 1016: bin_base_path = optarg; break;
            case 1001: num_rounds = std::stoi(optarg); break;
            case 1004: num_warmup = std::stoi(optarg); break;
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (load_index_path.empty() || dataset_path.empty()) {
        fprintf(stderr, "Error: --load-index and --dataset are required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (K <= 0) {
        fprintf(stderr, "Error: --k must be > 0\n");
        return 1;
    }
    if (num_rounds < 1)
        num_rounds = 1;
    if (num_warmup < 0)
        num_warmup = 0;

    // ---- (2) initialize ----
    Context ctx;
    InitOptions iopts;
    iopts.num_threads = num_threads;
    if (ctx.initialize(&argc, &argv, iopts) != Status::Ok) {
        fprintf(stderr, "[rank ?] initialize failed\n");
        return 1;
    }
    const int rank = ctx.rank();
    const int world_size = ctx.size();

    // ---- (3) load this node's shard into GD ----
    auto lr = ctx.load(static_cast<uint32_t>(rank), load_index_path);
    if (lr.status != Status::Ok) {
        fprintf(stderr, "[rank %d] load failed: %s\n", rank, lr.error.c_str());
        ctx.finalize();
        return 1;
    }
    const uint32_t dim = lr.dim;
    const uint64_t ntotal = lr.ntotal;

    // ---- (4) deploy: channels + sync + service threads ----
    DeployOptions deploy;
    deploy.search_threads = search_threads;
    deploy.service_threads = service_threads;
    deploy.expand_batch = expand_batch;
    deploy.use_dot_norm = use_dot_norm;
    if (ctx.node_init_and_sync(static_cast<uint32_t>(rank), deploy) != Status::Ok) {
        fprintf(stderr, "[rank %d] node_init_and_sync failed\n", rank);
        ctx.finalize();
        return 1;
    }

    // ---- (5) load queries + ground truth (each rank; eval-only skips base) ----
    DatasetFormat fmt;
    if (dataset_format_str == "hdf5") {
        fmt = DatasetFormat::HDF5;
    } else if (dataset_format_str == "bin") {
        fmt = DatasetFormat::BIN;
    } else {
        DatasetFormat detected = detect_dataset_format(dataset_path);
        fmt = (detected == DatasetFormat::UNKNOWN) ? DatasetFormat::HDF5 : detected;
    }

    AnnDataset dataset;
    try {
        if (fmt == DatasetFormat::HDF5) {
            dataset = load_hdf5_dataset(dataset_path, /*eval_only=*/true);
        } else {
            dataset = load_bin_dataset(dataset_path, /*eval_only=*/true, bin_base_path);
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "[rank %d] dataset load failed: %s\n", rank, e.what());
        ctx.finalize();
        return 1;
    }
    if (dataset.dim != dim) {
        fprintf(stderr, "[rank %d] dataset dim (%u) != index dim (%u)\n", rank, dataset.dim, dim);
        ctx.finalize();
        return 1;
    }
    const uint64_t nq = dataset.n_test;
    const uint32_t gt_k = dataset.k_gt;
    const float *all_queries = dataset.test.data();
    const int32_t *all_gt = dataset.neighbors.data();

    // ---- (6) route queries to this rank ----
    std::vector<uint32_t> my_idx;
    if (cluster_route) {
        my_idx = Context::route_queries(ctx, all_queries, nq);
    } else {
        uint64_t per = (nq + world_size - 1) / world_size;
        uint64_t qs = static_cast<uint64_t>(rank) * per;
        uint64_t qe = std::min(qs + per, nq);
        my_idx.reserve(qe - qs);
        for (uint64_t i = qs; i < qe; i++)
            my_idx.push_back(static_cast<uint32_t>(i));
    }
    const uint64_t my_nq = my_idx.size();
    if (rank == 0) {
        printf("[bench] ntotal=%lu, nq=%lu, dim=%u, k=%d, ef_search=%d, ranks=%d, rounds=%d, warmup=%d\n",
               (unsigned long)ntotal, (unsigned long)nq, dim, K,
               ef_search > 0 ? ef_search : static_cast<int>(lr.ef_search), world_size, num_rounds, num_warmup);
    }

    // Build contiguous query + ground-truth buffers for this rank.
    std::vector<float> my_queries(my_nq * dim);
    std::vector<int32_t> my_gt;
    if (gt_k > 0)
        my_gt.resize(my_nq * gt_k);
    for (uint64_t i = 0; i < my_nq; i++) {
        std::memcpy(my_queries.data() + i * dim, all_queries + static_cast<uint64_t>(my_idx[i]) * dim,
                    dim * sizeof(float));
        if (gt_k > 0) {
            std::memcpy(my_gt.data() + i * gt_k, all_gt + static_cast<uint64_t>(my_idx[i]) * gt_k,
                        gt_k * sizeof(int32_t));
        }
    }

    // ---- (7) benchmark rounds ----
    SearchParams sp(K);
    sp.ef_search = ef_search;
    BenchResult bres = run_benchmark_rounds(ctx, my_queries.data(), my_nq, dim, sp, K, num_rounds, num_warmup, nq, rank,
                                            MPI_COMM_WORLD);

    // ---- (8) recall + latency + summary ----
    RecallStats rstats;
    if (gt_k > 0) {
        const std::vector<uint32_t> *idmap_ptr = ctx.has_idmap() ? &ctx.idmap() : nullptr;
        rstats = compute_recall_stats(bres.out_ids.data(), my_nq, my_gt.data(), gt_k, K, idmap_ptr);
    }
    double avg_recall = aggregate_recall(rstats, rank, MPI_COMM_WORLD);
    LatencyPercentiles lat = compute_latency_percentiles(bres.query_latencies_us, my_nq, MPI_COMM_WORLD);

    if (rank == 0) {
        BenchReportArgs bargs;
        bargs.dataset_path = dataset_path.c_str();
        bargs.n_train = ntotal;
        bargs.nq = nq;
        bargs.world_size = world_size;
        bargs.num_threads = num_threads;
        bargs.num_shards = lr.num_shards;
        bargs.num_rounds = num_rounds;
        bargs.M = lr.M;
        bargs.ef_search = (ef_search > 0) ? static_cast<uint32_t>(ef_search) : lr.ef_search;
        bargs.K = K;
        print_benchmark_summary(bargs, bres.round_qps, bres.round_e2e_qps, lat, avg_recall);
    }

    // ---- (9) finalize ----
    ctx.finalize();
    return 0;
}
