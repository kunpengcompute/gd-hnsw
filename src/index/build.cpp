/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// build.cpp — offline sharded-index build (no MPI / ubs-mem / NUMA).
//
// `build_sharded_index()` is the faiss `write_index`-style free function that
// builds a global HNSW index and persists it as N shard files. It lives in its
// own translation unit so build tools only link the build layer
// (gd_hnsw_build) rather than the full deploy layer (gd_hnsw_api), which
// would drag in MPI / ubs-mem / NUMA symbols at link time.

#include "build.h"

#include "ann_dataset.h"
#include "bin_loader.h"
#include "faiss_extractor.h"
#include "hdf5_loader.h"
#include "index_io.h"
#include "gd_layout.h"

#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace gd_hnsw {

namespace {

DatasetFormat detect_format(const std::string &path, const std::string &fmt_str)
{
    if (fmt_str == "hdf5")
        return DatasetFormat::HDF5;
    if (fmt_str == "bin")
        return DatasetFormat::BIN;
    if (fmt_str != "auto")
        return DatasetFormat::UNKNOWN;
    DatasetFormat detected = detect_dataset_format(path);
    return (detected == DatasetFormat::UNKNOWN) ? DatasetFormat::HDF5 : detected;
}

} // namespace

BuildResult build_sharded_index(const BuildOptions &opts)
{
    BuildResult r;
    if (opts.dataset_path.empty()) {
        r.status = Status::InvalidArg;
        r.error = "dataset_path is empty";
        return r;
    }
    if (opts.output_path.empty()) {
        r.status = Status::InvalidArg;
        r.error = "output_path is empty";
        return r;
    }
    if (opts.num_shards == 0) {
        r.status = Status::InvalidArg;
        r.error = "num_shards must be > 0";
        return r;
    }

    if (opts.build_threads > 0) {
        omp_set_num_threads(static_cast<int>(opts.build_threads));
        omp_set_dynamic(0);
    }

    DatasetFormat fmt = detect_format(opts.dataset_path, opts.dataset_format);
    if (fmt == DatasetFormat::UNKNOWN) {
        r.status = Status::InvalidArg;
        r.error = "unknown dataset_format '" + opts.dataset_format + "'";
        return r;
    }
    bool has_bin_override = !opts.bin_base_path.empty();
    if (fmt == DatasetFormat::HDF5 && has_bin_override) {
        fprintf(stderr, "[build] Warning: bin_base_path ignored with dataset_format=hdf5\n");
    }

    printf("[build] Loading dataset (train vectors only): %s\n", opts.dataset_path.c_str());
    auto t0 = std::chrono::steady_clock::now();
    AnnDataset dataset;
    try {
        if (fmt == DatasetFormat::HDF5) {
            dataset = load_hdf5_dataset(opts.dataset_path, /*eval_only=*/false, /*train_only=*/true);
        } else {
            dataset = load_bin_dataset(opts.dataset_path, /*eval_only=*/false, opts.bin_base_path, "", "",
                                       /*train_only=*/true);
        }
    } catch (const std::exception &e) {
        r.status = Status::IoError;
        r.error = std::string("dataset load failed: ") + e.what();
        return r;
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("[build] Loaded %lu vectors, dim=%u (%.2f sec)\n", (unsigned long)dataset.n_train, dataset.dim,
           std::chrono::duration<double>(t1 - t0).count());

    if (dataset.n_train == 0 || dataset.dim == 0) {
        r.status = Status::InvalidArg;
        r.error = "dataset is empty";
        return r;
    }

    ShardBuildOptions shard_build_opts;
    shard_build_opts.cluster_partition = opts.cluster_partition;
    shard_build_opts.post_add_callback = [&]() {
        size_t freed_bytes = dataset.train.size() * sizeof(float);
        dataset.train.clear();
        dataset.train.shrink_to_fit();
        printf("[build] Released dataset.train after FAISS add (~%.1f GiB)\n",
               freed_bytes / (1024.0 * 1024.0 * 1024.0));
    };

    std::vector<uint32_t> new_to_old_idmap;
    std::vector<float> cluster_centroids;

    printf("[build] Building HNSW: M=%u, ef_construction=%u, shards=%u, cluster=%s\n", opts.M, opts.ef_construction,
           opts.num_shards, opts.cluster_partition ? "ON" : "OFF");
    auto tb0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> shard_sizes;
    try {
        shard_sizes = FaissExtractor::build_and_extract_sharded_streaming(
            dataset.train.data(), dataset.n_train, dataset.dim, opts.num_shards, opts.output_path, opts.M,
            opts.ef_construction, opts.ef_search, shard_build_opts,
            opts.cluster_partition ? &new_to_old_idmap : nullptr,
            opts.cluster_partition ? &cluster_centroids : nullptr);
    } catch (const std::exception &e) {
        r.status = Status::IoError;
        r.error = std::string("build failed: ") + e.what();
        return r;
    }
    auto tb1 = std::chrono::steady_clock::now();
    uint64_t total_bytes = 0;
    for (auto sz : shard_sizes)
        total_bytes += sz;
    printf("[build] Build complete: %.2f sec, total: %.1f MB (%u shards)\n",
           std::chrono::duration<double>(tb1 - tb0).count(), total_bytes / (1024.0 * 1024.0), opts.num_shards);

    // Sidecars.
    if (opts.cluster_partition && !new_to_old_idmap.empty()) {
        if (!save_idmap_file(opts.output_path, new_to_old_idmap)) {
            r.status = Status::IoError;
            r.error = "failed to save idmap sidecar";
            return r;
        }
    }
    if (opts.cluster_partition && !cluster_centroids.empty()) {
        if (!save_centroids_file(opts.output_path, cluster_centroids, opts.num_shards, dataset.dim)) {
            r.status = Status::IoError;
            r.error = "failed to save centroids sidecar";
            return r;
        }
    }

    // Per-shard verification (one at a time to bound memory).
    printf("[build] Verifying saved shards...\n");
    try {
        for (uint32_t s = 0; s < opts.num_shards; s++) {
            std::string path = index_file_for_shard(opts.output_path, s);
            FILE *fp = fopen(path.c_str(), "rb");
            if (!fp) {
                r.status = Status::IoError;
                r.error = "cannot open '" + path + "'";
                return r;
            }
            fseek(fp, 0, SEEK_END);
            long fsz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (fsz <= 0) {
                fclose(fp);
                r.status = Status::IoError;
                r.error = "'" + path + "' empty";
                return r;
            }
            std::vector<char> buf(static_cast<size_t>(fsz));
            size_t nread = fread(buf.data(), 1, buf.size(), fp);
            fclose(fp);
            if (nread != buf.size()) {
                r.status = Status::IoError;
                r.error = "short read '" + path + "'";
                return r;
            }
            auto vr = validate_superblock(buf.data(), static_cast<uint64_t>(fsz));
            if (!vr.ok) {
                r.status = Status::ValidationErr;
                r.error = "'" + path + "' validation: " + vr.error;
                return r;
            }
            printf("  Shard %u: OK (%.1f MB)\n", s, fsz / (1024.0 * 1024.0));
        }
    } catch (const std::exception &e) {
        r.status = Status::IoError;
        r.error = std::string("shard verification failed: ") + e.what();
        return r;
    }

    // Cross-shard consistency check (headers only).
    if (opts.num_shards > 1) {
        printf("[build] Cross-shard consistency check...\n");
        uint64_t sum_ntotal_local = 0;
        uint64_t expected_ntotal = 0;
        uint32_t expected_dim = 0, expected_M = 0, expected_num_shards = 0;
        uint32_t expected_num_levels = 0, expected_ef_search = 0;
        int32_t expected_max_level = 0, expected_entry_point = 0;
        for (uint32_t s = 0; s < opts.num_shards; s++) {
            std::string path = index_file_for_shard(opts.output_path, s);
            FILE *fp = fopen(path.c_str(), "rb");
            if (!fp) {
                r.status = Status::IoError;
                r.error = "cannot reopen '" + path + "'";
                return r;
            }
            SuperBlockV1 sb{};
            if (fread(&sb, sizeof(sb), 1, fp) != 1) {
                fclose(fp);
                r.status = Status::IoError;
                r.error = "cannot read header '" + path + "'";
                return r;
            }
            fclose(fp);
            if (s == 0) {
                expected_ntotal = sb.ntotal;
                expected_dim = sb.dim;
                expected_M = sb.M;
                expected_num_shards = sb.num_shards;
                expected_num_levels = sb.num_levels;
                expected_ef_search = sb.ef_search;
                expected_max_level = sb.max_level;
                expected_entry_point = sb.entry_point;
            } else if (sb.ntotal != expected_ntotal || sb.dim != expected_dim || sb.M != expected_M ||
                       sb.num_shards != expected_num_shards || sb.max_level != expected_max_level ||
                       sb.entry_point != expected_entry_point) {
                r.status = Status::ValidationErr;
                r.error = "shard " + std::to_string(s) + " params inconsistent with shard 0";
                return r;
            }
            if (s > 0 && sb.num_levels != expected_num_levels) {
                r.status = Status::ValidationErr;
                r.error = "shard " + std::to_string(s) + " num_levels=" + std::to_string(sb.num_levels) +
                          ", expected " + std::to_string(expected_num_levels);
                return r;
            }
            if (s > 0 && sb.ef_search != expected_ef_search) {
                r.status = Status::ValidationErr;
                r.error = "shard " + std::to_string(s) + " ef_search=" + std::to_string(sb.ef_search) + ", expected " +
                          std::to_string(expected_ef_search);
                return r;
            }
            if (sb.shard_id != s) {
                r.status = Status::ValidationErr;
                r.error = "shard " + std::to_string(s) + " has shard_id=" + std::to_string(sb.shard_id);
                return r;
            }
            if (sb.global_id_begin != sum_ntotal_local) {
                r.status = Status::ValidationErr;
                r.error = "shard " + std::to_string(s) + " global_id_begin mismatch";
                return r;
            }
            sum_ntotal_local += sb.ntotal_local;
        }
        if (sum_ntotal_local != expected_ntotal) {
            r.status = Status::ValidationErr;
            r.error = "sum(ntotal_local) != ntotal";
            return r;
        }
        printf("  Cross-shard: OK (ntotal=%lu, dim=%u, M=%u)\n", (unsigned long)expected_ntotal, expected_dim,
               expected_M);
    }
    printf("[build] Verification passed.\n");

    r.status = Status::Ok;
    r.ntotal = dataset.n_train;
    r.dim = dataset.dim;
    r.num_shards = opts.num_shards;
    r.shard_bytes = shard_sizes;
    return r;
}

} // namespace gd_hnsw
