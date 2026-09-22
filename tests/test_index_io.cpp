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

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#if defined(__linux__)
#include <csignal>
#include <sys/resource.h>
#endif

#include "index_io.h"

using namespace gd_hnsw;

static std::string rand_path(const char *tag)
{
    return "test_io_" + std::string(tag) + "_" + std::to_string(std::rand() % 1000000);
}

// ============================================================
// Path helpers
// ============================================================

TEST(IndexIO, IndexFileForShard)
{
    auto p = index_file_for_shard("/tmp/idx", 3);
    EXPECT_NE(p.find("shard_3"), std::string::npos);
}

TEST(IndexIO, IdmapFileForIndex)
{
    auto p = idmap_file_for_index("/tmp/idx");
    EXPECT_NE(p.find(".idmap"), std::string::npos);
}

TEST(IndexIO, CentroidsFileForIndex)
{
    auto p = centroids_file_for_index("/tmp/idx");
    EXPECT_NE(p.find(".centroids"), std::string::npos);
}

// ============================================================
// Parent directory creation
// ============================================================

TEST(IndexIO, EnsureParentDirCreatesNested)
{
    auto root = "test_io_pdir_" + std::to_string(std::rand() % 1000000);
    auto base = root + "/a/b/c/idx";

    std::string err;
    EXPECT_TRUE(ensure_parent_dir(base, err)) << err;
    EXPECT_TRUE(std::filesystem::is_directory(root + "/a/b/c"));

    // Idempotent: directory already exists is still success
    EXPECT_TRUE(ensure_parent_dir(base, err));

    std::filesystem::remove_all(root);
}

TEST(IndexIO, EnsureParentDirNoDirComponentSucceeds)
{
    std::string err;
    EXPECT_TRUE(ensure_parent_dir("plain_file_name", err));
}

TEST(IndexIO, EnsureParentDirFailsUnderRegularFile)
{
    auto file = rand_path("pdir_file");
    FILE *fp = fopen(file.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    std::string err;
    EXPECT_FALSE(ensure_parent_dir(file + "/sub/x", err));
    EXPECT_FALSE(err.empty());

    std::remove(file.c_str());
}

// ============================================================
// IdMap I/O
// ============================================================

TEST(IndexIO, IdmapSaveLoadRoundtrip)
{
    auto base = rand_path("idmap_rtt");
    std::vector<uint32_t> orig = {10, 20, 30, 40, 50};

    ASSERT_TRUE(save_idmap_file(base, orig));

    std::vector<uint32_t> loaded;
    EXPECT_EQ(load_idmap_file(base, loaded), IdMapLoadStatus::Loaded);
    EXPECT_EQ(loaded, orig);

    std::remove(idmap_file_for_index(base).c_str());
}

TEST(IndexIO, IdmapSaveCreatesMissingParentDirs)
{
    // Regression for code-review issue #2: save must create missing parent
    // directories instead of failing with an open error.
    auto root = "test_io_idmap_dirs_" + std::to_string(std::rand() % 1000000);
    auto base = root + "/x/y/idx";
    std::vector<uint32_t> orig = {7, 8, 9};

    ASSERT_TRUE(save_idmap_file(base, orig));

    std::vector<uint32_t> loaded;
    EXPECT_EQ(load_idmap_file(base, loaded), IdMapLoadStatus::Loaded);
    EXPECT_EQ(loaded, orig);

    std::filesystem::remove_all(root);
}

TEST(IndexIO, IdmapEmptySucceeds)
{
    auto base = rand_path("idmap_empty");
    std::vector<uint32_t> empty;
    ASSERT_TRUE(save_idmap_file(base, empty));

    std::vector<uint32_t> loaded;
    EXPECT_EQ(load_idmap_file(base, loaded), IdMapLoadStatus::Loaded);
    EXPECT_TRUE(loaded.empty());

    std::remove(idmap_file_for_index(base).c_str());
}

TEST(IndexIO, IdmapNotFound)
{
    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file("/nonexistent/path_xyz", out), IdMapLoadStatus::NotFound);
}

TEST(IndexIO, IdmapBadMagic)
{
    // Write a file with wrong magic, verify load returns Error
    auto base = rand_path("idmap_badmagic");
    std::string path = idmap_file_for_index(base);

    IdMapFileHeader hdr{};
    hdr.magic = 0xDEADBEEF;
    hdr.version = IDMAP_VERSION;
    hdr.count = 0;

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file(base, out), IdMapLoadStatus::Error);
    EXPECT_TRUE(out.empty());

    std::remove(path.c_str());
}

TEST(IndexIO, IdmapFileNotOverwrittenOnError)
{
    // Load from a non-existent file — should return NotFound without crashing
    std::vector<uint32_t> out;
    auto status = load_idmap_file("/nonexistent/idmap_file_test", out);
    EXPECT_TRUE(status == IdMapLoadStatus::NotFound || status == IdMapLoadStatus::Error);
    EXPECT_TRUE(out.empty());
}

TEST(IndexIO, IdmapLargeRoundtrip)
{
    auto base = rand_path("idmap_large");
    std::vector<uint32_t> orig(10000);
    for (size_t i = 0; i < orig.size(); i++)
        orig[i] = static_cast<uint32_t>(i * 3);

    ASSERT_TRUE(save_idmap_file(base, orig));
    std::vector<uint32_t> loaded;
    EXPECT_EQ(load_idmap_file(base, loaded), IdMapLoadStatus::Loaded);
    EXPECT_EQ(loaded, orig);

    std::remove(idmap_file_for_index(base).c_str());
}

// ============================================================
// Centroids I/O
// ============================================================

TEST(IndexIO, CentroidsSaveLoadRoundtrip)
{
    auto base = rand_path("cnt_rtt");
    uint32_t dim = 8, num_shards = 3;
    std::vector<float> orig(num_shards * dim);
    for (size_t i = 0; i < orig.size(); i++)
        orig[i] = static_cast<float>(i) * 1.5f;

    ASSERT_TRUE(save_centroids_file(base, orig, num_shards, dim));

    std::vector<float> loaded;
    uint32_t loaded_num_shards = 0, loaded_dim = 0;
    EXPECT_EQ(load_centroids_file(base, loaded, loaded_num_shards, loaded_dim), IdMapLoadStatus::Loaded);
    EXPECT_EQ(loaded_num_shards, num_shards);
    EXPECT_EQ(loaded_dim, dim);
    EXPECT_EQ(loaded.size(), orig.size());
    for (size_t i = 0; i < orig.size(); i++)
        EXPECT_FLOAT_EQ(loaded[i], orig[i]);

    std::remove(centroids_file_for_index(base).c_str());
}

TEST(IndexIO, CentroidsSingleShard)
{
    auto base = rand_path("cnt_one");
    uint32_t dim = 16, num_shards = 1;
    std::vector<float> orig(dim);
    for (uint32_t i = 0; i < dim; i++)
        orig[i] = static_cast<float>(i);

    ASSERT_TRUE(save_centroids_file(base, orig, num_shards, dim));
    std::vector<float> loaded;
    uint32_t ld_ns = 0, ld_dim = 0;
    EXPECT_EQ(load_centroids_file(base, loaded, ld_ns, ld_dim), IdMapLoadStatus::Loaded);
    EXPECT_EQ(ld_ns, 1u);
    EXPECT_EQ(ld_dim, 16u);
    EXPECT_FLOAT_EQ(loaded[0], 0.0f);

    std::remove(centroids_file_for_index(base).c_str());
}

TEST(IndexIO, CentroidsNotFound)
{
    std::vector<float> out;
    uint32_t ns = 0, dim = 0;
    EXPECT_EQ(load_centroids_file("/nonexistent/cnt_xyz", out, ns, dim), IdMapLoadStatus::NotFound);
}

TEST(IndexIO, CentroidsBadMagic)
{
    // Write a file with wrong magic, verify load returns Error
    auto base = rand_path("cnt_badmagic");
    std::string path = centroids_file_for_index(base);

    CentroidsFileHeader hdr{};
    hdr.magic = 0xCAFEBABE;
    hdr.version = CENTROIDS_VERSION;
    hdr.dim = 1;
    hdr.num_shards = 1;

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<float> out;
    uint32_t ns = 0, dim = 0;
    EXPECT_EQ(load_centroids_file(base, out, ns, dim), IdMapLoadStatus::Error);
    EXPECT_TRUE(out.empty());

    std::remove(path.c_str());
}

// ============================================================
// Shard buffer I/O
// ============================================================

TEST(IndexIO, ShardBuffersSaveLoadRoundtrip)
{
    auto base = rand_path("shard_rtt");
    const uint32_t num_shards = 2;

    // Build two minimal valid shard buffers
    SuperBlockV1 sb_proto{};
    sb_proto.magic = GD_MAGIC;
    sb_proto.schema_version = SCHEMA_VERSION;
    sb_proto.state = static_cast<uint32_t>(GdState::BUILDING);
    sb_proto.metric_type = METRIC_L2;
    sb_proto.endianness = ENDIAN_LITTLE;
    sb_proto.ntotal = 200;
    sb_proto.dim = 16;
    sb_proto.M = 16;
    sb_proto.num_shards = num_shards;
    sb_proto.num_levels = 2;
    sb_proto.entry_point = 0;
    sb_proto.max_level = 0;
    sb_proto.neighbors_size = 200 * 16;

    std::vector<std::vector<char>> orig(num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        auto tmp = sb_proto;
        tmp.shard_id = s;
        tmp.ntotal_local = 100;
        tmp.global_id_begin = s * 100;
        compute_layout(tmp);
        tmp.header_crc64 = compute_header_crc64(tmp);
        orig[s].resize(tmp.total_bytes);
        std::memcpy(orig[s].data(), &tmp, sizeof(SuperBlockV1));
    }

    ASSERT_TRUE(save_shard_buffers(base, orig));
    std::vector<std::vector<char>> loaded;
    ASSERT_TRUE(load_shard_buffers(base, num_shards, loaded));
    EXPECT_EQ(loaded.size(), num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        const auto *sb_ld = reinterpret_cast<const SuperBlockV1 *>(loaded[s].data());
        EXPECT_EQ(sb_ld->shard_id, s);
        EXPECT_EQ(sb_ld->num_shards, num_shards);
    }

    for (uint32_t s = 0; s < num_shards; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBuffersMissingFile)
{
    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers("/nonexistent/shards_xyz", 1, bufs));
}

// ============================================================
// I/O error path tests (A7)
// ============================================================

TEST(IndexIO, IdmapTruncatedFile)
{
    auto base = rand_path("idmap_trunc");
    std::string path = idmap_file_for_index(base);

    // Write only the header, no data — file is shorter than header+count*uint32
    IdMapFileHeader hdr{};
    hdr.magic = IDMAP_MAGIC;
    hdr.version = IDMAP_VERSION;
    hdr.count = 100; // claim 100 entries but write none

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file(base, out), IdMapLoadStatus::Error);

    std::remove(path.c_str());
}

TEST(IndexIO, CentroidsTruncatedFile)
{
    auto base = rand_path("cnt_trunc");
    std::string path = centroids_file_for_index(base);

    CentroidsFileHeader hdr{};
    hdr.magic = CENTROIDS_MAGIC;
    hdr.version = CENTROIDS_VERSION;
    hdr.dim = 16;
    hdr.num_shards = 4; // claim 4*16=64 floats but write none

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<float> out;
    uint32_t ns = 0, dim = 0;
    EXPECT_EQ(load_centroids_file(base, out, ns, dim), IdMapLoadStatus::Error);

    std::remove(path.c_str());
}

TEST(IndexIO, ShardBufferTruncatedFile)
{
    auto base = rand_path("shard_trunc");
    std::string path = index_file_for_shard(base, 0);

    // Write a valid SuperBlock header but truncate before the full buffer
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.state = static_cast<uint32_t>(GdState::BUILDING);
    sb.metric_type = METRIC_L2;
    sb.endianness = ENDIAN_LITTLE;
    sb.ntotal = 100;
    sb.ntotal_local = 100;
    sb.dim = 16;
    sb.M = 16;
    sb.num_shards = 1;
    sb.shard_id = 0;
    sb.num_levels = 2;
    sb.entry_point = 0;
    sb.max_level = 0;
    sb.neighbors_size = 100 * 16;
    compute_layout(sb);
    sb.header_crc64 = compute_header_crc64(sb);

    // Write only half of the expected buffer
    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&sb, sizeof(SuperBlockV1), 1, fp);
    // Don't write the block data — file is truncated
    fclose(fp);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 1, bufs));

    std::remove(path.c_str());
}

TEST(IndexIO, CentroidsWrongVersion)
{
    auto base = rand_path("cnt_badver");
    std::string path = centroids_file_for_index(base);

    CentroidsFileHeader hdr{};
    hdr.magic = CENTROIDS_MAGIC;
    hdr.version = 999; // unsupported version
    hdr.dim = 1;
    hdr.num_shards = 1;

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<float> out;
    uint32_t ns = 0, dim = 0;
    EXPECT_EQ(load_centroids_file(base, out, ns, dim), IdMapLoadStatus::Error);
    EXPECT_TRUE(out.empty());

    std::remove(path.c_str());
}

TEST(IndexIO, IdmapWrongVersion)
{
    auto base = rand_path("idmap_badver");
    std::string path = idmap_file_for_index(base);

    IdMapFileHeader hdr{};
    hdr.magic = IDMAP_MAGIC;
    hdr.version = 999; // unsupported version
    hdr.count = 0;

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file(base, out), IdMapLoadStatus::Error);
    EXPECT_TRUE(out.empty());

    std::remove(path.c_str());
}

TEST(IndexIO, ShardBufferSingleShard)
{
    auto base = rand_path("shard_single");
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.state = static_cast<uint32_t>(GdState::BUILDING);
    sb.metric_type = METRIC_L2;
    sb.endianness = ENDIAN_LITTLE;
    sb.ntotal = 50;
    sb.ntotal_local = 50;
    sb.dim = 8;
    sb.M = 8;
    sb.num_shards = 1;
    sb.shard_id = 0;
    sb.num_levels = 2;
    sb.entry_point = 0;
    sb.max_level = 0;
    sb.neighbors_size = 50 * 8;
    compute_layout(sb);
    sb.header_crc64 = compute_header_crc64(sb);

    std::vector<std::vector<char>> orig(1);
    orig[0].resize(sb.total_bytes);
    std::memcpy(orig[0].data(), &sb, sizeof(SuperBlockV1));

    ASSERT_TRUE(save_shard_buffers(base, orig));
    std::vector<std::vector<char>> loaded;
    ASSERT_TRUE(load_shard_buffers(base, 1, loaded));
    EXPECT_EQ(loaded.size(), 1u);

    std::remove(index_file_for_shard(base, 0).c_str());
}

// ============================================================
// Cross-shard validation error paths (coverage gap fill)
// ============================================================

// Helper: build a minimal shard buffer (BUILDING state skips CRC
// checks in validate_superblock, so we don't need payload CRCs).
static std::vector<char> make_shard_buf_for_io(uint64_t ntotal, uint64_t ntotal_local, uint32_t dim, uint32_t M,
                                               uint32_t shard_id, uint32_t num_shards, uint64_t gid_begin,
                                               int32_t entry_point = 0, int32_t max_level = 0)
{
    SuperBlockV1 sb{};
    sb.magic = GD_MAGIC;
    sb.schema_version = SCHEMA_VERSION;
    sb.state = static_cast<uint32_t>(GdState::BUILDING);
    sb.metric_type = METRIC_L2;
    sb.endianness = ENDIAN_LITTLE;
    sb.ntotal = ntotal;
    sb.ntotal_local = ntotal_local;
    sb.dim = dim;
    sb.M = M;
    sb.num_shards = num_shards;
    sb.shard_id = shard_id;
    sb.num_levels = 2;
    sb.entry_point = entry_point;
    sb.max_level = max_level;
    sb.global_id_begin = gid_begin;
    sb.neighbors_size = ntotal_local * M * 2;
    compute_layout(sb);
    std::vector<char> buf(sb.total_bytes, 0);
    std::memcpy(buf.data(), &sb, sizeof(SuperBlockV1));
    return buf;
}

static void write_shard_file(const std::string &base, uint32_t shard_id, const std::vector<char> &buf)
{
    std::string path = index_file_for_shard(base, shard_id);
    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr) << "Cannot create " << path;
    fwrite(buf.data(), 1, buf.size(), fp);
    fclose(fp);
}

TEST(IndexIO, ShardBufferNumShardsMismatch)
{
    // sb0.num_shards in header != num_shards passed to load_shard_buffers
    auto base = rand_path("shard_ns_mm");
    const uint32_t num_files = 2;

    for (uint32_t s = 0; s < num_files; s++) {
        auto buf = make_shard_buf_for_io(100, 50, 8, 4, s,
                                         3, // claim 3 shards in header
                                         s * 50);
        write_shard_file(base, s, buf);
    }

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < num_files; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBufferShardIdMismatch)
{
    // A shard file has shard_id that doesn't match its file index
    auto base = rand_path("shard_sid_mm");

    auto buf0 = make_shard_buf_for_io(100, 50, 8, 4, 1, // shard_id=1 in file index 0
                                      2, 0);
    auto buf1 = make_shard_buf_for_io(100, 50, 8, 4, 1, 2, 50);
    write_shard_file(base, 0, buf0);
    write_shard_file(base, 1, buf1);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < 2; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBufferPerShardNumShardsMismatch)
{
    // A non-zero shard has different num_shards from shard 0
    auto base = rand_path("shard_psns_mm");

    auto buf0 = make_shard_buf_for_io(100, 50, 8, 4, 0, 2, 0);
    auto buf1 = make_shard_buf_for_io(100, 50, 8, 4, 1, 3, 50); // num_shards=3
    write_shard_file(base, 0, buf0);
    write_shard_file(base, 1, buf1);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < 2; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBufferInconsistentParams)
{
    // Shards with different ntotal / dim / M / max_level / entry_point
    auto base = rand_path("shard_params");

    auto buf0 = make_shard_buf_for_io(100, 50, 8, 4, 0, 2, 0, 0, 0);
    auto buf1 = make_shard_buf_for_io(200, 150, 16, 8, 1, 2, 50, 10, 1); // all different
    write_shard_file(base, 0, buf0);
    write_shard_file(base, 1, buf1);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < 2; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBufferGlobalIdBeginMismatch)
{
    // Non-contiguous global_id_begin (gap between shards)
    auto base = rand_path("shard_gid_mm");

    auto buf0 = make_shard_buf_for_io(100, 50, 8, 4, 0, 2, 0);
    auto buf1 = make_shard_buf_for_io(100, 50, 8, 4, 1, 2,
                                      60); // should be 50
    write_shard_file(base, 0, buf0);
    write_shard_file(base, 1, buf1);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < 2; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBufferSumNtotalMismatch)
{
    // ntotal_local values that don't sum to ntotal
    auto base = rand_path("shard_sum_mm");

    auto buf0 = make_shard_buf_for_io(100, 40, 8, 4, 0, 2, 0);  // 40
    auto buf1 = make_shard_buf_for_io(100, 70, 8, 4, 1, 2, 40); // 40+70=110 != 100
    write_shard_file(base, 0, buf0);
    write_shard_file(base, 1, buf1);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < 2; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

TEST(IndexIO, ShardBufferEmptyFile)
{
    auto base = rand_path("shard_empty");
    std::string path = index_file_for_shard(base, 0);

    // Create an empty file
    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 1, bufs));

    std::remove(path.c_str());
}

TEST(IndexIO, ShardBufferInvalidMagic)
{
    // File with valid size but corrupted magic
    auto base = rand_path("shard_badmag");
    auto buf = make_shard_buf_for_io(100, 50, 8, 4, 0, 2, 0);

    // Corrupt magic
    auto *sb = reinterpret_cast<SuperBlockV1 *>(buf.data());
    sb->magic = 0xDEADBEEFBAADF00DULL;

    write_shard_file(base, 0, buf);
    write_shard_file(base, 1, make_shard_buf_for_io(100, 50, 8, 4, 1, 2, 50));

    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers(base, 2, bufs));

    for (uint32_t s = 0; s < 2; s++)
        std::remove(index_file_for_shard(base, s).c_str());
}

// ============================================================
// I/O error path tests (P1 coverage boost)
//
// Targets the remaining uncovered branches in index_io.cpp:
//   - fopen for write fails (EISDIR: derived path is a directory)
//   - fopen for read fails with non-ENOENT (ENOTDIR: regular file in path)
//   - fread of header fails (empty file)
//   - idmap count overflow guard
//   - num_shards == 0 guard
//   - fwrite failures via RLIMIT_FSIZE (Linux only)
// ============================================================

#if defined(__linux__)
// RAII guard: cap the max writable file size and ignore SIGXFSZ so that
// writes beyond the cap fail with EFBIG instead of killing the process.
// Restores both on destruction.
class FileSizeLimitGuard {
public:
    explicit FileSizeLimitGuard(rlim_t limit)
    {
        getrlimit(RLIMIT_FSIZE, &old_);
        rlimit capped{limit, old_.rlim_max};
        set_ok_ = setrlimit(RLIMIT_FSIZE, &capped) == 0;
        old_handler_ = signal(SIGXFSZ, SIG_IGN);
    }
    ~FileSizeLimitGuard()
    {
        signal(SIGXFSZ, old_handler_);
        setrlimit(RLIMIT_FSIZE, &old_);
    }
    FileSizeLimitGuard(const FileSizeLimitGuard &) = delete;
    FileSizeLimitGuard &operator=(const FileSizeLimitGuard &) = delete;

    bool set_ok() const { return set_ok_; }

private:
    rlimit old_{};
    bool set_ok_ = false;
    void (*old_handler_)(int) = SIG_DFL;
};
#endif // defined(__linux__)

// --- save_*: fopen for writing fails (derived path is a directory) ---

TEST(IndexIO, IdmapSaveOpenFailsWhenPathIsDirectory)
{
    auto base = rand_path("idmap_dirsave");
    std::string path = idmap_file_for_index(base);
    ASSERT_TRUE(std::filesystem::create_directory(path)); // dir where the file should go

    EXPECT_FALSE(save_idmap_file(base, {1, 2, 3}));

    std::filesystem::remove(path);
}

TEST(IndexIO, CentroidsSaveOpenFailsWhenPathIsDirectory)
{
    auto base = rand_path("cnt_dirsave");
    std::string path = centroids_file_for_index(base);
    ASSERT_TRUE(std::filesystem::create_directory(path));

    EXPECT_FALSE(save_centroids_file(base, {1.f, 2.f}, 1, 2));

    std::filesystem::remove(path);
}

TEST(IndexIO, ShardSaveOpenFailsWhenPathIsDirectory)
{
    auto base = rand_path("shard_dirsave");
    std::string path = index_file_for_shard(base, 0);
    ASSERT_TRUE(std::filesystem::create_directory(path));

    std::vector<std::vector<char>> bufs(1);
    bufs[0].assign(64, 0xAB);
    EXPECT_FALSE(save_shard_buffers(base, bufs));

    std::filesystem::remove(path);
}

// --- save_*: parent directory creation fails (regular file in path) ---

TEST(IndexIO, IdmapSaveFailsWhenParentUncreatable)
{
    auto file = rand_path("idmap_pblock");
    FILE *fp = fopen(file.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    // "file/sub/idx.idmap" — create_directories fails: a regular file sits in the path
    EXPECT_FALSE(save_idmap_file(file + "/sub/idx", {1, 2, 3}));

    std::remove(file.c_str());
}

TEST(IndexIO, CentroidsSaveFailsWhenParentUncreatable)
{
    auto file = rand_path("cnt_pblock");
    FILE *fp = fopen(file.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    EXPECT_FALSE(save_centroids_file(file + "/sub/idx", {1.f}, 1, 1));

    std::remove(file.c_str());
}

TEST(IndexIO, ShardSaveFailsWhenParentUncreatable)
{
    auto file = rand_path("shard_pblock");
    FILE *fp = fopen(file.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    std::vector<std::vector<char>> bufs(1);
    EXPECT_FALSE(save_shard_buffers(file + "/sub/idx", bufs));

    std::remove(file.c_str());
}

// --- load_*: fopen for reading fails with non-ENOENT errno (ENOTDIR) ---

TEST(IndexIO, IdmapLoadOpenFailsWithEnotdir)
{
    auto file = rand_path("idmap_enotdir");
    FILE *fp = fopen(file.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    // "file/sub.idmap" — a regular file where a directory is required -> ENOTDIR
    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file(file + "/sub", out), IdMapLoadStatus::Error);

    std::remove(file.c_str());
}

TEST(IndexIO, CentroidsLoadOpenFailsWithEnotdir)
{
    auto file = rand_path("cnt_enotdir");
    FILE *fp = fopen(file.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp);

    std::vector<float> out;
    uint32_t ns = 0, dim = 0;
    EXPECT_EQ(load_centroids_file(file + "/sub", out, ns, dim), IdMapLoadStatus::Error);

    std::remove(file.c_str());
}

// --- load_*: header read fails (empty file: fread of header returns 0) ---

TEST(IndexIO, IdmapEmptyFileHeaderReadFails)
{
    auto base = rand_path("idmap_hdr0");
    std::string path = idmap_file_for_index(base);

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp); // zero-length file

    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file(base, out), IdMapLoadStatus::Error);

    std::remove(path.c_str());
}

TEST(IndexIO, CentroidsEmptyFileHeaderReadFails)
{
    auto base = rand_path("cnt_hdr0");
    std::string path = centroids_file_for_index(base);

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fclose(fp); // zero-length file

    std::vector<float> out;
    uint32_t ns = 0, dim = 0;
    EXPECT_EQ(load_centroids_file(base, out, ns, dim), IdMapLoadStatus::Error);

    std::remove(path.c_str());
}

// --- load_idmap_file: count overflow guard (count > INT32_MAX) ---

TEST(IndexIO, IdmapCountTooLarge)
{
    auto base = rand_path("idmap_huge");
    std::string path = idmap_file_for_index(base);

    IdMapFileHeader hdr{};
    hdr.magic = IDMAP_MAGIC;
    hdr.version = IDMAP_VERSION;
    hdr.count = 2147483648ULL; // INT32_MAX + 1

    FILE *fp = fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fclose(fp);

    std::vector<uint32_t> out;
    EXPECT_EQ(load_idmap_file(base, out), IdMapLoadStatus::Error);

    std::remove(path.c_str());
}

// --- load_shard_buffers: num_shards == 0 guard ---

TEST(IndexIO, ShardLoadZeroShardsFails)
{
    std::vector<std::vector<char>> bufs;
    EXPECT_FALSE(load_shard_buffers("unused_base_path", 0, bufs));
}

// --- fwrite failure paths via RLIMIT_FSIZE (Linux only) ---
//
// Note: stdio uses a ~4KB internal buffer (BUFSIZ). fwrite calls that
// don't fill the buffer are absorbed by memcpy and never trigger a
// write() syscall, so RLIMIT_FSIZE has no effect on them. The header
// writes (24/28 bytes) are too small to trigger a flush — they only
// hit the kernel at fclose(), whose return value is not checked.
//
// Therefore:
// - "header write fails" branch cannot be triggered via RLIMIT_FSIZE
//   (would need setvbuf(fp, NULL, _IONBF, 0) inside the production code).
// - "short write" branch requires data large enough to fill the stdio
//   buffer (~4096 bytes), forcing a write() syscall mid-fwrite.

#if defined(__linux__)

TEST(IndexIO, IdmapShortWriteUnderFsizeLimit)
{
    // 1100 entries = 4400 bytes.  Header (24) + 4072 bytes of data fill the
    // 4096-byte stdio buffer -> flush -> write() hits RLIMIT_FSIZE cap ->
    // short write.  Cap = header + 2 bytes so only 0 complete uint32_t
    // items make it through.
    FileSizeLimitGuard guard(sizeof(IdMapFileHeader) + 2);
    ASSERT_TRUE(guard.set_ok());

    auto base = rand_path("idmap_fs_data");
    std::vector<uint32_t> idmap(1100, 0);
    EXPECT_FALSE(save_idmap_file(base, idmap));

    std::remove(idmap_file_for_index(base).c_str());
}

TEST(IndexIO, CentroidsShortWriteUnderFsizeLimit)
{
    // 1100 floats (ns=1, dim=1100) = 4400 bytes.  Same buffer-flush logic
    // as the idmap test above.  Cap = header + 1 byte.
    FileSizeLimitGuard guard(sizeof(CentroidsFileHeader) + 1);
    ASSERT_TRUE(guard.set_ok());

    auto base = rand_path("cnt_fs_data");
    std::vector<float> data(1100, 1.0f);
    EXPECT_FALSE(save_centroids_file(base, data, 1, 1100));

    std::remove(centroids_file_for_index(base).c_str());
}

TEST(IndexIO, ShardShortWriteUnderFsizeLimit)
{
    // Cap far below the buffer size: fwrite writes 16 of 4096 bytes.
    FileSizeLimitGuard guard(16);
    ASSERT_TRUE(guard.set_ok());

    auto base = rand_path("shard_fs");
    std::vector<std::vector<char>> bufs(1);
    bufs[0].assign(4096, 0xAB);
    EXPECT_FALSE(save_shard_buffers(base, bufs));

    std::remove(index_file_for_shard(base, 0).c_str());
}

#endif // defined(__linux__)
