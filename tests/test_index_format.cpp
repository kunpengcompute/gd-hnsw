// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "hnsw_urma/index/builder.hpp"
#include "hnsw_urma/index/format.hpp"

namespace {

std::filesystem::path BuildFixture(std::string_view name) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("hnsw_urma_index_" + std::string(name));
  std::filesystem::remove_all(root);
  const std::vector<float> vectors{
      0.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F, 0.0F, 2.0F, 1.0F, 2.0F, 2.0F, 2.0F,
  };
  hnsw_urma::ReferenceBuildOptions options;
  options.num_shards = 2;
  options.M = 2;
  options.M0 = 3;
  options.ef_search = 4;
  const auto status =
      hnsw_urma::BuildReferenceIndex(root, vectors, 6, 2, options, 100);
  if (!status.ok()) {
    hnsw_urma::test::Fail("fixture build", __FILE__, __LINE__,
                          status.ToString());
  }
  return root;
}

hnsw_urma::IndexHeaderV2 ReadHeader(const std::filesystem::path &path) {
  hnsw_urma::IndexHeaderV2 header;
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  return header;
}

void WriteHeader(const std::filesystem::path &path,
                 hnsw_urma::IndexHeaderV2 header) {
  header.header_checksum = hnsw_urma::IndexHeaderChecksum(header);
  std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
  output.write(reinterpret_cast<const char *>(&header), sizeof(header));
}

} // namespace

TEST(ReferenceFormatIndexFormat, IndexFormatRoundTripsAllFiles) {
  const auto root = BuildFixture("roundtrip");
  auto global = hnsw_urma::LoadGlobalIndexFile(root);
  UB_REQUIRE_TRUE(global.ok());
  UB_REQUIRE_EQ(global.value().meta.ntotal, 6U);
  UB_REQUIRE_EQ(global.value().ranges.size(), 2U);

  auto shard0 = hnsw_urma::LoadIndexForShard(root, 0);
  auto shard1 = hnsw_urma::LoadIndexForShard(root, 1);
  UB_REQUIRE_TRUE(shard0.ok());
  UB_REQUIRE_TRUE(shard1.ok());
  UB_REQUIRE_EQ(shard0.value().local_shard.Neighbors(0).value().size(), 3U);
  const auto trusted_neighbors = shard0.value().local_shard.TrustedNeighbors(0);
  UB_REQUIRE_TRUE(trusted_neighbors.size() <= 3U);
  UB_REQUIRE_TRUE(std::find(trusted_neighbors.begin(), trusted_neighbors.end(),
                            hnsw_urma::INVALID_NODE_ID) ==
                  trusted_neighbors.end());
  UB_REQUIRE_EQ(shard1.value().local_shard.Vector(5).value()[0], 2.0F);
  UB_REQUIRE_TRUE(shard0.value().upper_graph.persistent_bytes() > 0);
  UB_REQUIRE_TRUE(shard0.value().local_entry_point.global_id < 3U);
  UB_REQUIRE_TRUE(shard1.value().local_entry_point.global_id >= 3U);
  std::filesystem::remove_all(root);
}

TEST(ReferenceFormatIndexFormat, UpperGraphLoadOwnsResidentStorage) {
  const auto root = BuildFixture("resident_upper_graph");
  auto global = hnsw_urma::LoadGlobalIndexFile(root);
  UB_REQUIRE_TRUE(global.ok());
  auto upper = hnsw_urma::LoadUpperGraphFile(root, global.value().meta);
  UB_REQUIRE_TRUE(upper.ok());
  const std::vector<hnsw_urma::node_id_t> original(
      upper.value().neighbor_slots().begin(),
      upper.value().neighbor_slots().end());
  if (original.empty()) {
    hnsw_urma::test::Fail("upper neighbor storage is non-empty", __FILE__,
                          __LINE__);
  }

  const auto path = root / "upper_graph.index";
  const auto header = ReadHeader(path);
  const auto replacement = original.front() ^ 1U;
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekp(static_cast<std::streamoff>(header.blocks[1].offset));
  file.write(reinterpret_cast<const char *>(&replacement), sizeof(replacement));
  file.flush();
  UB_REQUIRE_TRUE(file.good());
  file.close();

  UB_REQUIRE_EQ(upper.value().neighbor_slots().front(), original.front());
  std::filesystem::remove_all(root);
}

TEST(ReferenceFormatIndexFormat,
     LocalShardUsesCanonicalAlignedGraphAndVectorBuffers) {
  const auto root = BuildFixture("canonical_shard_storage");
  auto loaded = hnsw_urma::LoadIndexForShard(root, 0);
  UB_REQUIRE_TRUE(loaded.ok());
  auto &shard = loaded.value().local_shard;
  const auto graph = shard.layer0_slots();
  const auto graph_logical = shard.graph_logical_memory();
  const auto graph_registration = shard.graph_registration_memory();
  const auto vectors = shard.vectors();
  const auto vector_logical = shard.vector_logical_memory();
  const auto vector_registration = shard.vector_registration_memory();
  auto page_size = hnsw_urma::PageAlignedBuffer::SystemPageSize();
  UB_REQUIRE_TRUE(page_size.ok());
  UB_REQUIRE_EQ(graph_logical.size(), graph.size_bytes());
  UB_REQUIRE_TRUE(graph_logical.data() ==
                  reinterpret_cast<const std::byte *>(graph.data()));
  UB_REQUIRE_EQ(graph_registration.data(), graph_logical.data());
  UB_REQUIRE_TRUE(graph_registration.size() >= graph_logical.size());
  UB_REQUIRE_EQ(reinterpret_cast<std::uintptr_t>(graph_logical.data()) %
                    page_size.value(),
                0U);
  UB_REQUIRE_FALSE(shard.graph_storage_shared());
  UB_REQUIRE_EQ(vector_logical.size(), vectors.size_bytes());
  UB_REQUIRE_TRUE(vector_logical.data() ==
                  reinterpret_cast<const std::byte *>(vectors.data()));
  UB_REQUIRE_EQ(vector_registration.data(), vector_logical.data());
  UB_REQUIRE_TRUE(vector_registration.size() >= vector_logical.size());
  UB_REQUIRE_EQ(reinterpret_cast<std::uintptr_t>(vector_logical.data()) %
                    page_size.value(),
                0U);
  UB_REQUIRE_FALSE(shard.vector_storage_shared());

  auto shared = hnsw_urma::LoadIndexForShard(
      root, 0, hnsw_urma::LocalShardStorage::kShared);
  if (!shared.ok()) {
    hnsw_urma::test::Fail("shared.ok()", __FILE__, __LINE__,
                          shared.status().ToString());
  }
  auto &shared_shard = shared.value().local_shard;
  const auto shared_graph = shared_shard.layer0_slots();
  const auto shared_graph_registration =
      shared_shard.graph_registration_memory();
  const auto shared_vectors = shared_shard.vectors();
  const auto shared_vector_registration =
      shared_shard.vector_registration_memory();
  UB_REQUIRE_TRUE(shared_shard.graph_storage_shared());
  UB_REQUIRE_FALSE(shared_shard.graph_shared_memory_name().empty());
  UB_REQUIRE_TRUE(shared_graph_registration.data() ==
                  reinterpret_cast<const std::byte *>(shared_graph.data()));
  UB_REQUIRE_EQ(shared_graph_registration.size() % page_size.value(), 0U);
  UB_REQUIRE_TRUE(shared_shard.vector_storage_shared());
  UB_REQUIRE_FALSE(shared_shard.vector_shared_memory_name().empty());
  UB_REQUIRE_TRUE(shared_vector_registration.data() ==
                  reinterpret_cast<const std::byte *>(shared_vectors.data()));
  UB_REQUIRE_EQ(shared_vector_registration.size() % page_size.value(), 0U);
  UB_REQUIRE_TRUE(shared_shard.UnlinkGraphSharedMemory().ok());
  UB_REQUIRE_TRUE(shared_shard.UnlinkVectorSharedMemory().ok());
  std::filesystem::remove_all(root);
}

TEST(ReferenceFormatIndexFormat,
     UpperGraphSelectsHighestLocalEntryPointPerShard) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 8;
  meta.dim = 2;
  meta.metric = 0;
  meta.M = 2;
  meta.M0 = 3;
  meta.ef_search_default = 4;
  meta.entry_point = 0;
  meta.max_level = 3;
  meta.num_shards = 3;
  const std::vector<hnsw_urma::UpperNodeMeta> nodes{
      {0, 3, 0, 0},
      {2, 2, 0, 6},
      {3, 2, 0, 10},
  };
  const std::vector<hnsw_urma::node_id_t> neighbors(14,
                                                    hnsw_urma::INVALID_NODE_ID);
  auto upper = hnsw_urma::UpperGraphReplica::Create(meta, nodes, neighbors);
  UB_REQUIRE_TRUE(upper.ok());

  auto shard0 = upper.value().EntryPointForShard({0, 0, 2});
  auto shard1 = upper.value().EntryPointForShard({1, 2, 2});
  auto shard2 = upper.value().EntryPointForShard({2, 4, 4});
  UB_REQUIRE_TRUE(shard0.ok());
  UB_REQUIRE_TRUE(shard1.ok());
  UB_REQUIRE_TRUE(shard2.ok());
  UB_REQUIRE_EQ(shard0.value(), (hnsw_urma::ShardEntryPoint{0, 3}));
  UB_REQUIRE_EQ(shard1.value(), (hnsw_urma::ShardEntryPoint{2, 2}));
  UB_REQUIRE_EQ(shard2.value(), (hnsw_urma::ShardEntryPoint{4, 0}));
}

TEST(ReferenceFormatIndexFormat, IndexFormatRejectsHeaderAndPayloadCorruption) {
  const auto root = BuildFixture("corruption");
  const auto global_path = root / "global.meta";
  auto header = ReadHeader(global_path);
  header.version = 99;
  WriteHeader(global_path, header);
  UB_REQUIRE_FALSE(hnsw_urma::LoadGlobalIndexFile(root).ok());

  std::filesystem::remove_all(root);
  const auto payload_root = BuildFixture("payload");
  const auto shard_path = hnsw_urma::ShardFilePath(payload_root, 0);
  const auto shard_header = ReadHeader(shard_path);
  std::fstream file(shard_path,
                    std::ios::binary | std::ios::in | std::ios::out);
  file.seekp(static_cast<std::streamoff>(shard_header.blocks[0].offset));
  const char byte = static_cast<char>(0x7f);
  file.write(&byte, 1);
  file.close();
  UB_REQUIRE_FALSE(hnsw_urma::LoadIndexForShard(payload_root, 0).ok());
  std::filesystem::remove_all(payload_root);
}

TEST(ReferenceFormatIndexFormat,
     IndexFormatRejectsTruncationAlignmentOverlapAndOverflow) {
  const auto trunc_root = BuildFixture("truncate");
  const auto trunc_path = hnsw_urma::ShardFilePath(trunc_root, 0);
  std::filesystem::resize_file(trunc_path, 4097);
  UB_REQUIRE_FALSE(hnsw_urma::LoadIndexForShard(trunc_root, 0).ok());
  std::filesystem::remove_all(trunc_root);

  const auto align_root = BuildFixture("align");
  const auto align_path = align_root / "upper_graph.index";
  auto align_header = ReadHeader(align_path);
  align_header.blocks[0].offset += 1;
  WriteHeader(align_path, align_header);
  auto global = hnsw_urma::LoadGlobalIndexFile(align_root);
  UB_REQUIRE_TRUE(global.ok());
  UB_REQUIRE_FALSE(
      hnsw_urma::LoadUpperGraphFile(align_root, global.value().meta).ok());
  std::filesystem::remove_all(align_root);

  const auto overlap_root = BuildFixture("overlap");
  const auto overlap_path = overlap_root / "upper_graph.index";
  auto overlap_header = ReadHeader(overlap_path);
  overlap_header.blocks[1].offset = overlap_header.blocks[0].offset;
  WriteHeader(overlap_path, overlap_header);
  global = hnsw_urma::LoadGlobalIndexFile(overlap_root);
  UB_REQUIRE_FALSE(
      hnsw_urma::LoadUpperGraphFile(overlap_root, global.value().meta).ok());
  std::filesystem::remove_all(overlap_root);

  const auto overflow_root = BuildFixture("overflow");
  const auto overflow_path = hnsw_urma::ShardFilePath(overflow_root, 0);
  auto overflow_header = ReadHeader(overflow_path);
  overflow_header.blocks[0].offset =
      std::numeric_limits<std::uint64_t>::max() - 4095U;
  overflow_header.blocks[0].bytes = 8192;
  WriteHeader(overflow_path, overflow_header);
  UB_REQUIRE_FALSE(hnsw_urma::LoadIndexForShard(overflow_root, 0).ok());
  std::filesystem::remove_all(overflow_root);
}

TEST(ReferenceFormatIndexFormat,
     IndexFormatRejectsInvalidUpperOffsetsAndNeighbors) {
  hnsw_urma::GlobalIndexMeta meta;
  meta.ntotal = 4;
  meta.dim = 2;
  meta.metric = 0;
  meta.M = 2;
  meta.M0 = 3;
  meta.ef_search_default = 4;
  meta.entry_point = 0;
  meta.max_level = 1;
  meta.num_shards = 1;
  std::vector<hnsw_urma::UpperNodeMeta> nodes{{0, 1, 0, 1}};
  std::vector<hnsw_urma::node_id_t> neighbors{1, 2};
  UB_REQUIRE_FALSE(
      hnsw_urma::UpperGraphReplica::Create(meta, nodes, neighbors).ok());
  nodes[0].neighbor_offset = 0;
  neighbors[1] = 4;
  UB_REQUIRE_FALSE(
      hnsw_urma::UpperGraphReplica::Create(meta, nodes, neighbors).ok());
  neighbors = {hnsw_urma::INVALID_NODE_ID, 1};
  UB_REQUIRE_FALSE(
      hnsw_urma::UpperGraphReplica::Create(meta, nodes, neighbors).ok());

  std::vector<hnsw_urma::node_id_t> layer0(
      static_cast<std::size_t>(meta.ntotal) * meta.M0,
      hnsw_urma::INVALID_NODE_ID);
  layer0[1] = 1;
  std::vector<float> vectors(static_cast<std::size_t>(meta.ntotal) * meta.dim,
                             0.0F);
  UB_REQUIRE_FALSE(
      hnsw_urma::LocalShard::Create(meta, {0, 0, meta.ntotal}, layer0, vectors)
          .ok());
}
