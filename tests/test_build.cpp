/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// test_build.cpp — Layer 3 build API coverage.
//
// Covers (previously 0%):
//   - include/bin_loader.h    : load_bin_dataset full / eval_only / train_only
//                               / overrides / every error branch,
//                               detect_dataset_format
//   - include/hdf5_loader.h  : load_hdf5_dataset full / missing distances
//                               / train_only / eval_only / every error branch
//   - include/ann_dataset.h  : AnnDataset aggregate init + accessors
//   - include/build.h        : build_sharded_index invalid args, bin/hdf5
//                               datasets, multi-shard, cluster partition
//                               sidecars, bin_base_path override
//   - include/faiss_extractor.h: ShardBuildOptions post_add_callback wiring
//
// Uses tiny datasets (dim 8, <= 96 vectors) so the Faiss HNSW build inside
// build_sharded_index completes in milliseconds.

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <H5Cpp.h>

#include "ann_dataset.h"
#include "bin_loader.h"
#include "build.h"
#include "faiss_extractor.h"
#include "gd_layout.h"
#include "hdf5_loader.h"
#include "index_io.h"

using namespace gd_hnsw;

namespace {

constexpr uint32_t B_DIM = 8;

std::string make_temp_dir(const std::string &tag)
{
    std::string dir = ::testing::TempDir() + "/gd_hnsw_build_test_" + tag;
    std::error_code ec;
    // wipe leftovers from earlier runs — stale files would break "missing file" cases
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

std::vector<float> make_vecs(uint64_t n, uint32_t dim)
{
    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (size_t i = 0; i < v.size(); i++)
        v[i] = static_cast<float>((i * 37) % 251) / 251.0f;
    return v;
}

bool write_bytes(const std::string &path, const void *data, size_t len)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    const size_t w = fwrite(data, 1, len, fp);
    fclose(fp);
    return w == len;
}

// .bin layout: [uint32 rows][uint32 cols] + row*col elements
bool write_bin_f32(const std::string &path, uint32_t rows, uint32_t cols, const std::vector<float> &data)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    const size_t h = fwrite(&rows, sizeof(rows), 1, fp) + fwrite(&cols, sizeof(cols), 1, fp);
    const size_t p = data.empty() ? 1 : fwrite(data.data(), sizeof(float), data.size(), fp);
    fclose(fp);
    return h == 2 && p >= 1;
}

bool write_bin_i32(const std::string &path, uint32_t rows, uint32_t cols, const std::vector<int32_t> &data)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    const size_t h = fwrite(&rows, sizeof(rows), 1, fp) + fwrite(&cols, sizeof(cols), 1, fp);
    const size_t p = data.empty() ? 1 : fwrite(data.data(), sizeof(int32_t), data.size(), fp);
    fclose(fp);
    return h == 2 && p >= 1;
}

// header-only .bin (for size-mismatch / invalid-header cases)
bool write_bin_raw_header(const std::string &path, uint32_t a, uint32_t b)
{
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    fwrite(&a, sizeof(a), 1, fp);
    fwrite(&b, sizeof(b), 1, fp);
    fclose(fp);
    return true;
}

// Write a 2D float dataset into an open HDF5 file.
void h5_add_f32(const H5::H5File &f, const char *name, hsize_t r, hsize_t c)
{
    hsize_t dims[2] = {r, c};
    H5::DataSpace s(2, dims);
    H5::DataSet d = f.createDataSet(name, H5::PredType::NATIVE_FLOAT, s);
    if (r > 0 && c > 0) {
        std::vector<float> v = make_vecs(r, static_cast<uint32_t>(c));
        d.write(v.data(), H5::PredType::NATIVE_FLOAT);
    }
}

void h5_add_i32(const H5::H5File &f, const char *name, hsize_t r, hsize_t c, int32_t pattern)
{
    hsize_t dims[2] = {r, c};
    H5::DataSpace s(2, dims);
    H5::DataSet d = f.createDataSet(name, H5::PredType::NATIVE_INT32, s);
    if (r > 0 && c > 0) {
        std::vector<int32_t> v(static_cast<size_t>(r) * c, pattern);
        d.write(v.data(), H5::PredType::NATIVE_INT32);
    }
}

// Standard benchmark-shaped HDF5: /train [+ /test /neighbors (/distances)]
void make_hdf5(const std::string &path, uint64_t n_train, uint32_t dim, uint64_t n_test, uint32_t k, bool with_dist)
{
    H5::H5File f(path, H5F_ACC_TRUNC);
    h5_add_f32(f, "train", static_cast<hsize_t>(n_train), dim);
    if (n_test > 0) {
        h5_add_f32(f, "test", static_cast<hsize_t>(n_test), dim);
        h5_add_i32(f, "neighbors", static_cast<hsize_t>(n_test), k, 3);
        if (with_dist)
            h5_add_f32(f, "distances", static_cast<hsize_t>(n_test), k);
    }
}

// Remove the outputs build_sharded_index produces under `base`.
void cleanup_build_output(const std::string &base, uint32_t num_shards, bool cluster)
{
    for (uint32_t s = 0; s < num_shards; s++)
        unlink(index_file_for_shard(base, s).c_str());
    if (cluster) {
        unlink(idmap_file_for_index(base).c_str());
        unlink(centroids_file_for_index(base).c_str());
    }
}

BuildOptions make_build_opts(const std::string &dataset, const std::string &out, uint32_t shards)
{
    BuildOptions o;
    o.dataset_path = dataset;
    o.output_path = out;
    o.num_shards = shards;
    o.M = 8;
    o.ef_construction = 40;
    o.ef_search = 32;
    o.build_threads = 2; // exercise the omp_set_num_threads branch
    return o;
}

// Read a shard file back and validate it as a SuperBlock.
std::vector<char> read_shard_ok(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return {};
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp)
        return {};
    std::vector<char> buf(static_cast<size_t>(st.st_size));
    const size_t rd = fread(buf.data(), 1, buf.size(), fp);
    fclose(fp);
    if (rd != buf.size())
        return {};
    return buf;
}

} // namespace

// ============================================================
// AnnDataset
// ============================================================

TEST(AnnDataset, InitZeroedAndAssign)
{
    AnnDataset ds; // default-init (covers the in-class initializers)
    EXPECT_EQ(ds.n_train, 0u);
    EXPECT_EQ(ds.dim, 0u);
    EXPECT_EQ(ds.n_test, 0u);
    EXPECT_EQ(ds.k_gt, 0u);
    EXPECT_TRUE(ds.train.empty());
    EXPECT_TRUE(ds.test.empty());
    EXPECT_TRUE(ds.neighbors.empty());
    EXPECT_TRUE(ds.distances.empty());

    ds.n_train = 3;
    ds.dim = 4;
    ds.train = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    EXPECT_EQ(ds.train.size(), ds.n_train * ds.dim);
}

// ============================================================
// bin_loader.h — load_bin_dataset
// ============================================================

TEST(BinLoader, FullLoad)
{
    const std::string dir = make_temp_dir("bin_full");
    constexpr uint32_t n = 20, nq = 5, k = 4;
    const auto base = make_vecs(n, B_DIM);
    const auto query = make_vecs(nq, B_DIM);
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, base));
    ASSERT_TRUE(write_bin_f32(dir + "/query.bin", nq, B_DIM, query));
    ASSERT_TRUE(write_bin_i32(dir + "/gt.bin", nq, k, std::vector<int32_t>(nq * k, 3)));

    AnnDataset ds = load_bin_dataset(dir);
    EXPECT_EQ(ds.n_train, n);
    EXPECT_EQ(ds.dim, B_DIM);
    ASSERT_EQ(ds.train.size(), static_cast<size_t>(n) * B_DIM);
    EXPECT_EQ(ds.train, base);
    EXPECT_EQ(ds.n_test, nq);
    ASSERT_EQ(ds.test.size(), static_cast<size_t>(nq) * B_DIM);
    EXPECT_EQ(ds.test, query);
    EXPECT_EQ(ds.k_gt, k);
    ASSERT_EQ(ds.neighbors.size(), static_cast<size_t>(nq) * k);
    EXPECT_EQ(ds.neighbors.front(), 3);
    // .bin format has no distances — loader fills zeros
    ASSERT_EQ(ds.distances.size(), static_cast<size_t>(nq) * k);
    for (float d : ds.distances)
        EXPECT_EQ(d, 0.0f);
}

TEST(BinLoader, EvalOnlyHeader)
{
    const std::string dir = make_temp_dir("bin_evalonly");
    constexpr uint32_t n = 7, nq = 2, k = 2;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    // eval_only only skips the base payload; query/gt are still loaded
    ASSERT_TRUE(write_bin_f32(dir + "/query.bin", nq, B_DIM, make_vecs(nq, B_DIM)));
    ASSERT_TRUE(write_bin_i32(dir + "/gt.bin", nq, k, std::vector<int32_t>(nq * k, 3)));

    AnnDataset ds = load_bin_dataset(dir, /*eval_only=*/true);
    EXPECT_EQ(ds.n_train, n);
    EXPECT_EQ(ds.dim, B_DIM);
    EXPECT_TRUE(ds.train.empty()); // base payload must not be read
    EXPECT_EQ(ds.n_test, nq);
    EXPECT_EQ(ds.k_gt, k);
    EXPECT_FALSE(ds.neighbors.empty());
}

TEST(BinLoader, TrainOnlySkipsQueryGt)
{
    const std::string dir = make_temp_dir("bin_trainonly");
    constexpr uint32_t n = 11;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));

    // no query.bin / gt.bin present — must not be touched in train_only mode
    AnnDataset ds = load_bin_dataset(dir, /*eval_only=*/false, "", "", "", /*train_only=*/true);
    EXPECT_EQ(ds.n_train, n);
    EXPECT_EQ(ds.dim, B_DIM);
    ASSERT_EQ(ds.train.size(), static_cast<size_t>(n) * B_DIM);
    EXPECT_EQ(ds.n_test, 0u);
    EXPECT_TRUE(ds.neighbors.empty());
}

TEST(BinLoader, BaseGivenAsFilePath)
{
    const std::string dir = make_temp_dir("bin_filepath");
    constexpr uint32_t n = 9, nq = 2, k = 2;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    ASSERT_TRUE(write_bin_f32(dir + "/query.bin", nq, B_DIM, make_vecs(nq, B_DIM)));
    ASSERT_TRUE(write_bin_i32(dir + "/gt.bin", nq, k, std::vector<int32_t>(nq * k, 1)));

    // passing the base.bin file itself — siblings resolved next to it
    AnnDataset ds = load_bin_dataset(dir + "/base.bin");
    EXPECT_EQ(ds.n_train, n);
    EXPECT_EQ(ds.n_test, nq);
    EXPECT_EQ(ds.k_gt, k);
}

TEST(BinLoader, OverridePaths)
{
    const std::string data = make_temp_dir("bin_ovr_data");
    const std::string work = make_temp_dir("bin_ovr_work"); // holds no bin files
    constexpr uint32_t n = 6, nq = 2, k = 3;
    ASSERT_TRUE(write_bin_f32(data + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    ASSERT_TRUE(write_bin_f32(data + "/query.bin", nq, B_DIM, make_vecs(nq, B_DIM)));
    ASSERT_TRUE(write_bin_i32(data + "/gt.bin", nq, k, std::vector<int32_t>(nq * k, 2)));

    AnnDataset ds =
        load_bin_dataset(work, /*eval_only=*/false, data + "/base.bin", data + "/query.bin", data + "/gt.bin");
    EXPECT_EQ(ds.n_train, n);
    EXPECT_EQ(ds.n_test, nq);
    EXPECT_EQ(ds.k_gt, k);
}

TEST(BinLoader, BaseErrors)
{
    const std::string dir = make_temp_dir("bin_base_err");

    // missing base.bin
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // too small (< 8 bytes)
    ASSERT_TRUE(write_bytes(dir + "/base.bin", "12345", 5));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // header n == 0
    ASSERT_TRUE(write_bin_raw_header(dir + "/base.bin", 0, B_DIM));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // header dim == 0
    ASSERT_TRUE(write_bin_raw_header(dir + "/base.bin", 4, 0));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // header dim > 10000
    ASSERT_TRUE(write_bin_raw_header(dir + "/base.bin", 4, 10001));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // size mismatch (header only, payload missing)
    ASSERT_TRUE(write_bin_raw_header(dir + "/base.bin", 4, B_DIM));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);
}

TEST(BinLoader, EvalOnlyErrors)
{
    const std::string dir = make_temp_dir("bin_eval_err");

    // missing
    EXPECT_THROW(load_bin_dataset(dir, /*eval_only=*/true), std::runtime_error);

    // too small
    ASSERT_TRUE(write_bytes(dir + "/base.bin", "1234", 4));
    EXPECT_THROW(load_bin_dataset(dir, true), std::runtime_error);

    // invalid header
    ASSERT_TRUE(write_bin_raw_header(dir + "/base.bin", 0, B_DIM));
    EXPECT_THROW(load_bin_dataset(dir, true), std::runtime_error);

    // size mismatch (eval path checks the full expected size too)
    ASSERT_TRUE(write_bin_raw_header(dir + "/base.bin", 4, B_DIM));
    EXPECT_THROW(load_bin_dataset(dir, true), std::runtime_error);
}

TEST(BinLoader, QueryAndGtErrors)
{
    const std::string dir = make_temp_dir("bin_q_err");
    constexpr uint32_t n = 8, nq = 2, k = 2;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));

    // missing query.bin
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // query dim mismatch
    ASSERT_TRUE(write_bin_f32(dir + "/query.bin", nq, B_DIM + 2, make_vecs(nq, B_DIM + 2)));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // query invalid header (nq == 0)
    ASSERT_TRUE(write_bin_raw_header(dir + "/query.bin", 0, B_DIM));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // query size mismatch
    ASSERT_TRUE(write_bin_raw_header(dir + "/query.bin", nq, B_DIM));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // valid query, missing gt.bin
    ASSERT_TRUE(write_bin_f32(dir + "/query.bin", nq, B_DIM, make_vecs(nq, B_DIM)));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // gt too small
    ASSERT_TRUE(write_bytes(dir + "/gt.bin", "12", 2));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // gt invalid header (k == 0)
    ASSERT_TRUE(write_bin_raw_header(dir + "/gt.bin", nq, 0));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // gt nq mismatch
    ASSERT_TRUE(write_bin_i32(dir + "/gt.bin", nq + 1, k, std::vector<int32_t>((nq + 1) * k, 1)));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // gt size mismatch
    ASSERT_TRUE(write_bin_raw_header(dir + "/gt.bin", nq, k));
    EXPECT_THROW(load_bin_dataset(dir), std::runtime_error);

    // valid trio — sanity
    ASSERT_TRUE(write_bin_i32(dir + "/gt.bin", nq, k, std::vector<int32_t>(nq * k, 1)));
    AnnDataset ds = load_bin_dataset(dir);
    EXPECT_EQ(ds.k_gt, k);
}

TEST(BinLoader, DetectFormat)
{
    EXPECT_EQ(detect_dataset_format("some/dir/ds.h5"), DatasetFormat::HDF5);
    EXPECT_EQ(detect_dataset_format("ds.hdf5"), DatasetFormat::HDF5);
    EXPECT_EQ(detect_dataset_format("some/dir/ds.bin"), DatasetFormat::BIN);

    // directory containing base.bin
    const std::string dir = make_temp_dir("detect");
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", 2, B_DIM, make_vecs(2, B_DIM)));
    EXPECT_EQ(detect_dataset_format(dir), DatasetFormat::BIN);

    // directory without base.bin / nonexistent path
    EXPECT_EQ(detect_dataset_format(::testing::TempDir()), DatasetFormat::UNKNOWN);
    EXPECT_EQ(detect_dataset_format("/nonexistent_gd_hnsw_dir"), DatasetFormat::UNKNOWN);
}

// ============================================================
// hdf5_loader.h — load_hdf5_dataset
// ============================================================

TEST(Hdf5Loader, FullLoad)
{
    const std::string dir = make_temp_dir("h5_full");
    const std::string path = dir + "/ds.h5";
    constexpr uint64_t n = 12, nq = 4;
    constexpr uint32_t k = 3;
    make_hdf5(path, n, B_DIM, nq, k, /*with_dist=*/true);

    AnnDataset ds = load_hdf5_dataset(path);
    EXPECT_EQ(ds.n_train, n);
    EXPECT_EQ(ds.dim, B_DIM);
    ASSERT_EQ(ds.train.size(), static_cast<size_t>(n) * B_DIM);
    EXPECT_EQ(ds.train, make_vecs(n, B_DIM));
    EXPECT_EQ(ds.n_test, nq);
    ASSERT_EQ(ds.test.size(), static_cast<size_t>(nq) * B_DIM);
    EXPECT_EQ(ds.k_gt, k);
    ASSERT_EQ(ds.neighbors.size(), static_cast<size_t>(nq) * k);
    EXPECT_EQ(ds.neighbors.front(), 3);
    ASSERT_EQ(ds.distances.size(), static_cast<size_t>(nq) * k);
    // distances written by make_hdf5 (patterned) are preserved
    EXPECT_EQ(ds.distances.front(), make_vecs(nq, k).front());
}

TEST(Hdf5Loader, MissingDistancesFillsZeros)
{
    const std::string dir = make_temp_dir("h5_nodist");
    const std::string path = dir + "/ds.h5";
    constexpr uint64_t n = 6, nq = 2;
    constexpr uint32_t k = 2;
    make_hdf5(path, n, B_DIM, nq, k, /*with_dist=*/false);

    AnnDataset ds = load_hdf5_dataset(path);
    ASSERT_EQ(ds.distances.size(), static_cast<size_t>(nq) * k);
    for (float d : ds.distances)
        EXPECT_EQ(d, 0.0f);
}

TEST(Hdf5Loader, TrainOnly)
{
    const std::string dir = make_temp_dir("h5_trainonly");
    const std::string path = dir + "/ds.h5";
    make_hdf5(path, 6, B_DIM, /*n_test=*/0, /*k=*/0, false);

    AnnDataset ds = load_hdf5_dataset(path, /*eval_only=*/false, /*train_only=*/true);
    EXPECT_EQ(ds.n_train, 6u);
    EXPECT_EQ(ds.dim, B_DIM);
    EXPECT_EQ(ds.n_test, 0u);
    EXPECT_TRUE(ds.neighbors.empty());
}

TEST(Hdf5Loader, EvalOnly)
{
    const std::string dir = make_temp_dir("h5_evalonly");
    const std::string path = dir + "/ds.h5";
    make_hdf5(path, 9, B_DIM, 2, 2, true);

    AnnDataset ds = load_hdf5_dataset(path, /*eval_only=*/true, /*train_only=*/false);
    EXPECT_EQ(ds.n_train, 9u);
    EXPECT_EQ(ds.dim, B_DIM);
    EXPECT_TRUE(ds.train.empty()); // train payload must not be read
    // eval_only only skips the train payload; query/gt are still loaded
    EXPECT_EQ(ds.n_test, 2u);
    ASSERT_EQ(ds.test.size(), static_cast<size_t>(2) * B_DIM);
    EXPECT_EQ(ds.k_gt, 2u);
    EXPECT_FALSE(ds.neighbors.empty());
}

TEST(Hdf5Loader, Errors)
{
    const std::string dir = make_temp_dir("h5_err");

    // nonexistent file
    EXPECT_ANY_THROW(load_hdf5_dataset(dir + "/missing.h5"));

    // /train dim != /test dim
    {
        H5::H5File f(dir + "/dim.h5", H5F_ACC_TRUNC);
        h5_add_f32(f, "train", 6, 8);
        h5_add_f32(f, "test", 4, 4);
    }
    EXPECT_ANY_THROW(load_hdf5_dataset(dir + "/dim.h5"));

    // /neighbors rows != /test rows
    {
        H5::H5File f(dir + "/gt.h5", H5F_ACC_TRUNC);
        h5_add_f32(f, "train", 6, 8);
        h5_add_f32(f, "test", 2, 8);
        h5_add_i32(f, "neighbors", 3, 5, 1);
    }
    EXPECT_ANY_THROW(load_hdf5_dataset(dir + "/gt.h5"));

    // /distances shape mismatch
    {
        H5::H5File f(dir + "/dist.h5", H5F_ACC_TRUNC);
        h5_add_f32(f, "train", 6, 8);
        h5_add_f32(f, "test", 2, 8);
        h5_add_i32(f, "neighbors", 2, 5, 1);
        h5_add_f32(f, "distances", 2, 6);
    }
    EXPECT_ANY_THROW(load_hdf5_dataset(dir + "/dist.h5"));

    // /train not 2D
    {
        H5::H5File f(dir + "/nd1.h5", H5F_ACC_TRUNC);
        hsize_t dims[1] = {6};
        H5::DataSpace s(1, dims);
        H5::DataSet d = f.createDataSet("train", H5::PredType::NATIVE_FLOAT, s);
    }
    EXPECT_ANY_THROW(load_hdf5_dataset(dir + "/nd1.h5"));
}

// ============================================================
// build.h — build_sharded_index
// ============================================================

TEST(Build, InvalidArgs)
{
    BuildOptions o;
    o.output_path = ::testing::TempDir() + "/gd_hnsw_build_test_inv";
    o.num_shards = 1;

    o.dataset_path = ""; // empty dataset path
    BuildResult r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::InvalidArg);
    EXPECT_FALSE(r.error.empty());

    o.dataset_path = "whatever.bin";
    o.output_path = ""; // empty output path
    r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::InvalidArg);

    o.output_path = ::testing::TempDir() + "/gd_hnsw_build_test_inv";
    o.num_shards = 0; // zero shards
    r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::InvalidArg);

    o.num_shards = 1;
    o.dataset_format = "xml"; // unknown format
    r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::InvalidArg);
}

TEST(Build, BinSingleShard)
{
    const std::string dir = make_temp_dir("b_bin1");
    constexpr uint32_t n = 64;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    BuildResult r = build_sharded_index(make_build_opts(dir, out, 1));
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, n);
    EXPECT_EQ(r.dim, B_DIM);
    ASSERT_EQ(r.shard_bytes.size(), 1u);

    // shard file exists, has the reported size, and validates
    const std::string p = index_file_for_shard(out, 0);
    const std::vector<char> buf = read_shard_ok(p);
    ASSERT_FALSE(buf.empty());
    EXPECT_EQ(buf.size(), r.shard_bytes[0]);
    const auto vr = validate_superblock(buf.data(), buf.size());
    EXPECT_TRUE(vr.ok) << vr.error;
    const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
    EXPECT_EQ(sb->ntotal, n);
    EXPECT_EQ(sb->dim, B_DIM);
    EXPECT_EQ(sb->num_shards, 1u);
    EXPECT_EQ(sb->ntotal_local, static_cast<uint64_t>(n));
    EXPECT_EQ(sb->global_id_begin, 0u);

    cleanup_build_output(out, 1, false);
}

TEST(Build, BinMultiShard)
{
    const std::string dir = make_temp_dir("b_bin3");
    constexpr uint32_t n = 96;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    BuildResult r = build_sharded_index(make_build_opts(dir, out, 3));
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, n);
    ASSERT_EQ(r.shard_bytes.size(), 3u);

    // per-shard headers: ids ascending, global_id_begin chains to ntotal
    uint64_t gid = 0;
    for (uint32_t s = 0; s < 3; s++) {
        const std::vector<char> buf = read_shard_ok(index_file_for_shard(out, s));
        ASSERT_FALSE(buf.empty());
        const auto vr = validate_superblock(buf.data(), buf.size());
        EXPECT_TRUE(vr.ok) << vr.error;
        const auto *sb = reinterpret_cast<const SuperBlockV1 *>(buf.data());
        EXPECT_EQ(sb->shard_id, s);
        EXPECT_EQ(sb->ntotal, n);
        EXPECT_EQ(sb->global_id_begin, gid);
        gid += sb->ntotal_local;
    }
    EXPECT_EQ(gid, static_cast<uint64_t>(n));

    cleanup_build_output(out, 3, false);
}

TEST(Build, Hdf5SingleShard)
{
    const std::string dir = make_temp_dir("b_h5");
    const std::string path = dir + "/ds.h5";
    make_hdf5(path, 48, B_DIM, 0, 0, false);
    const std::string out = dir + "/idx";

    BuildResult r = build_sharded_index(make_build_opts(path, out, 1));
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, 48u);
    EXPECT_EQ(r.dim, B_DIM);
    ASSERT_EQ(r.shard_bytes.size(), 1u);

    const std::vector<char> buf = read_shard_ok(index_file_for_shard(out, 0));
    ASSERT_FALSE(buf.empty());
    const auto vr = validate_superblock(buf.data(), buf.size());
    EXPECT_TRUE(vr.ok) << vr.error;

    cleanup_build_output(out, 1, false);
}

TEST(Build, ClusterPartitionSidecars)
{
    const std::string dir = make_temp_dir("b_cluster");
    constexpr uint32_t n = 80;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    BuildOptions o = make_build_opts(dir, out, 2);
    o.cluster_partition = true;
    BuildResult r = build_sharded_index(o);
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, n);

    // idmap sidecar: a permutation of [0, n)
    std::vector<uint32_t> idmap;
    ASSERT_EQ(load_idmap_file(out, idmap), IdMapLoadStatus::Loaded);
    ASSERT_EQ(idmap.size(), static_cast<size_t>(n));
    std::vector<uint32_t> sorted = idmap;
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t i = 0; i < n; i++)
        ASSERT_EQ(sorted[i], i) << "idmap is not a permutation";

    // centroids sidecar: num_shards * dim floats
    std::vector<float> cents;
    uint32_t ns = 0, dm = 0;
    ASSERT_EQ(load_centroids_file(out, cents, ns, dm), IdMapLoadStatus::Loaded);
    EXPECT_EQ(ns, 2u);
    EXPECT_EQ(dm, B_DIM);
    EXPECT_EQ(cents.size(), static_cast<size_t>(2) * B_DIM);

    cleanup_build_output(out, 2, true);
}

TEST(Build, BinBasePathOverride)
{
    const std::string data = make_temp_dir("b_ovr_data");
    const std::string work = make_temp_dir("b_ovr_work"); // no base.bin inside
    constexpr uint32_t n = 40;
    ASSERT_TRUE(write_bin_f32(data + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = work + "/idx";

    BuildOptions o = make_build_opts(work, out, 1);
    o.dataset_format = "bin";
    o.bin_base_path = data + "/base.bin";
    BuildResult r = build_sharded_index(o);
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, n);
    EXPECT_EQ(r.dim, B_DIM);

    cleanup_build_output(out, 1, false);
}

TEST(Build, Hdf5WithBinOverrideWarns)
{
    // explicit hdf5 + stray bin_base_path: build must still succeed (warn only)
    const std::string dir = make_temp_dir("b_h5_ovr");
    const std::string path = dir + "/ds.h5";
    make_hdf5(path, 32, B_DIM, 0, 0, false);
    const std::string out = dir + "/idx";

    BuildOptions o = make_build_opts(path, out, 1);
    o.dataset_format = "hdf5";
    o.bin_base_path = "/tmp/gd_hnsw_ignored_base.bin";
    BuildResult r = build_sharded_index(o);
    ASSERT_EQ(r.status, Status::Ok) << r.error;
    EXPECT_EQ(r.ntotal, 32u);

    cleanup_build_output(out, 1, false);
}

TEST(Build, DatasetLoadFailures)
{
    const std::string dir = make_temp_dir("b_fail");
    const std::string out = dir + "/idx";

    // nonexistent hdf5 file (suffix-based format detection) — H5::Exception
    // does not derive from std::exception, so it escapes build_sharded_index
    EXPECT_ANY_THROW(build_sharded_index(make_build_opts(dir + "/missing.h5", out, 1)));

    // empty train set: hdf5 /train with 0 rows
    const std::string empty = dir + "/empty.h5";
    {
        H5::H5File f(empty, H5F_ACC_TRUNC);
        h5_add_f32(f, "train", 0, B_DIM);
    }
    BuildResult r = build_sharded_index(make_build_opts(empty, out, 1));
    EXPECT_EQ(r.status, Status::InvalidArg);
}

// ============================================================
// build.h — build_sharded_index error paths
// ============================================================

// dataset load throws std::runtime_error → caught → IoError
TEST(Build, DatasetLoadFailureBin)
{
    const std::string dir = make_temp_dir("b_loadfail");
    const std::string out = dir + "/idx";

    // .bin suffix → BIN format; base.bin missing next to it
    BuildResult r = build_sharded_index(make_build_opts(dir + "/missing.bin", out, 1));
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("dataset load failed"), std::string::npos);
}

// streaming shard write fails (shard path pre-created as a directory) → build catch → IoError
TEST(Build, ShardWriteFailure)
{
    const std::string dir = make_temp_dir("b_writefail");
    constexpr uint32_t n = 16;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    // pre-create the shard-0 path as a directory: fopen(path, "wb") fails with EISDIR
    std::error_code ec;
    std::filesystem::create_directories(index_file_for_shard(out, 0), ec);

    BuildResult r = build_sharded_index(make_build_opts(dir, out, 1));
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("build failed"), std::string::npos);
}

// cluster partition with fewer vectors than shards → k-means empty cluster → build catch → IoError
TEST(Build, ClusterEmptyShardFails)
{
    const std::string dir = make_temp_dir("b_clusterempty");
    constexpr uint32_t n = 2;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    BuildOptions o = make_build_opts(dir, out, 3);
    o.cluster_partition = true;
    BuildResult r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("build failed"), std::string::npos);
}

// idmap sidecar save fails (path pre-created as a directory)
TEST(Build, IdmapSidecarSaveFailure)
{
    const std::string dir = make_temp_dir("b_idmapfail");
    constexpr uint32_t n = 64;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    std::error_code ec;
    std::filesystem::create_directories(idmap_file_for_index(out), ec);

    BuildOptions o = make_build_opts(dir, out, 2);
    o.cluster_partition = true;
    BuildResult r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("idmap sidecar"), std::string::npos);
}

// centroids sidecar save fails (path pre-created as a directory); idmap sidecar saves fine
TEST(Build, CentroidsSidecarSaveFailure)
{
    const std::string dir = make_temp_dir("b_centroidfail");
    constexpr uint32_t n = 64;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    std::error_code ec;
    std::filesystem::create_directories(centroids_file_for_index(out), ec);

    BuildOptions o = make_build_opts(dir, out, 2);
    o.cluster_partition = true;
    BuildResult r = build_sharded_index(o);
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("centroids sidecar"), std::string::npos);
}

// verification fopen("rb") fails: shard file pre-created write-only (0222) —
// build's "wb" open succeeds (write perm), verify's "rb" gets EACCES.
// Skipped when running as root (permission bits are not enforced for root).
TEST(Build, VerifyShardOpenFailure)
{
    if (::geteuid() == 0)
        GTEST_SKIP() << "chmod-based open failure does not apply to root";

    const std::string dir = make_temp_dir("b_vopenfail");
    constexpr uint32_t n = 16;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    const std::string shard0 = index_file_for_shard(out, 0);
    ASSERT_TRUE(write_bytes(shard0, "x", 1));
    ASSERT_EQ(chmod(shard0.c_str(), 0222), 0);

    BuildResult r = build_sharded_index(make_build_opts(dir, out, 1));
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("cannot open"), std::string::npos);

    chmod(shard0.c_str(), 0644); // restore so cleanup can remove it
}

// verification sees an empty shard: shard path is a symlink to /dev/null —
// the streaming write succeeds (kernel discards the bytes) but ftell reports
// size 0, so the "empty" branch fires. Works as root too.
TEST(Build, VerifyShardEmptyFailure)
{
    const std::string dir = make_temp_dir("b_vempty");
    constexpr uint32_t n = 16;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    const std::string shard0 = index_file_for_shard(out, 0);
    ASSERT_EQ(symlink("/dev/null", shard0.c_str()), 0);

    BuildResult r = build_sharded_index(make_build_opts(dir, out, 1));
    EXPECT_EQ(r.status, Status::IoError);
    EXPECT_NE(r.error.find("empty"), std::string::npos);
}

// validation failure: n=1 with 2 shards → uniform split leaves an empty tail
// shard (ntotal_local=0), which validate_superblock rejects → ValidationErr
TEST(Build, VerifyShardValidationFailure)
{
    const std::string dir = make_temp_dir("b_vvalid");
    constexpr uint32_t n = 1;
    ASSERT_TRUE(write_bin_f32(dir + "/base.bin", n, B_DIM, make_vecs(n, B_DIM)));
    const std::string out = dir + "/idx";

    BuildResult r = build_sharded_index(make_build_opts(dir, out, 2));
    EXPECT_EQ(r.status, Status::ValidationErr);
    EXPECT_NE(r.error.find("validation"), std::string::npos);
}

// ============================================================
// faiss_extractor.h — ShardBuildOptions wiring
// ============================================================

TEST(FaissExtractorOptions, PostAddCallbackRuns)
{
    constexpr uint64_t n = 32;
    const auto vecs = make_vecs(n, B_DIM);

    ShardBuildOptions o;
    bool called = false;
    o.post_add_callback = [&called]() { called = true; };

    const auto bufs = FaissExtractor::build_and_extract_sharded(vecs.data(), n, B_DIM, /*num_shards=*/2,
                                                                /*M=*/8, /*ef_construction=*/40, /*ef_search=*/32, o);
    EXPECT_TRUE(called) << "post_add_callback was not invoked after index->add";
    ASSERT_EQ(bufs.size(), 2u);
    for (const auto &b : bufs) {
        const auto vr = validate_superblock(b.data(), b.size());
        EXPECT_TRUE(vr.ok) << vr.error;
    }
}
