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

#include "fixture.hpp"
#include "hnsw_urma/compute/dataset.hpp"
#include "hnsw_urma/protocol/distance_batch_wire.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
using namespace hnsw_urma;
namespace {
class IndexExtended : public ub_complete::IndexFixture {};
IndexHeaderV2 Header(const std::filesystem::path &file) {
  IndexHeaderV2 h;
  std::ifstream input(file, std::ios::binary);
  input.read(reinterpret_cast<char *>(&h), sizeof(h));
  return h;
}
void WriteHeader(const std::filesystem::path &file, IndexHeaderV2 h) {
  h.header_checksum = IndexHeaderChecksum(h);
  std::fstream output(file, std::ios::binary | std::ios::in | std::ios::out);
  output.write(reinterpret_cast<const char *>(&h), sizeof(h));
}
} // namespace
TEST_F(IndexExtended, BuilderRejectsInvalidShapesAndEveryOptionBoundary) {
  ReferenceBuildOptions valid;
  valid.num_shards = 2;
  valid.M = 2;
  valid.M0 = 7;
  valid.ef_search = 8;
  EXPECT_FALSE(BuildReferenceIndex(root_, {}, 0, 2, valid).ok());
  EXPECT_FALSE(
      BuildReferenceIndex(root_, ub_complete::kVectors, 8, 0, valid).ok());
  EXPECT_FALSE(
      BuildReferenceIndex(root_, ub_complete::kVectors, 8, 3, valid).ok());
  EXPECT_FALSE(
      BuildReferenceIndex(root_, ub_complete::kVectors, 8, 2, valid, 7).ok());
  for (int field = 0; field < 6; ++field) {
    auto bad = valid;
    switch (field) {
    case 0:
      bad.num_shards = 0;
      break;
    case 1:
      bad.num_shards = 9;
      break;
    case 2:
      bad.M = 0;
      break;
    case 3:
      bad.M0 = 0;
      break;
    case 4:
      bad.ef_search = 0;
      break;
    case 5:
      bad.metric = 2;
      break;
    }
    EXPECT_FALSE(
        BuildReferenceIndex(root_, ub_complete::kVectors, 8, 2, bad).ok());
  }
}
TEST_F(IndexExtended, SingleRowAndUnevenShardsRemainReadable) {
  const std::vector<float> one{2, 3};
  ReferenceBuildOptions options;
  options.M = 2;
  options.M0 = 2;
  options.ef_search = 2;
  auto singleton = root_ / "one";
  ASSERT_TRUE(BuildReferenceIndex(singleton, one, 1, 2, options).ok());
  auto loaded = LoadIndexForShard(singleton, 0);
  ASSERT_TRUE(loaded.ok());
  EXPECT_EQ(loaded.value().meta.ntotal, 1U);
  EXPECT_FLOAT_EQ(loaded.value().local_shard.Vector(0).value()[0], 2.0F);
  options.num_shards = 3;
  ASSERT_TRUE(
      BuildReferenceIndex(root_, ub_complete::kVectors, 8, 2, options).ok());
  auto global = LoadGlobalIndexFile(root_);
  ASSERT_TRUE(global.ok());
  ASSERT_EQ(global.value().ranges.size(), 3U);
  EXPECT_EQ(global.value().ranges[0].local_count, 3U);
  EXPECT_EQ(global.value().ranges[2].local_count, 2U);
}
TEST_F(IndexExtended, GlobalHeaderRejectsEachStructuralCorruption) {
  const auto file = root_ / "global.meta";
  const auto original = Header(file);
  using Mutation = std::function<void(IndexHeaderV2 &)>;
  const std::vector<Mutation> mutations{
      [](auto &h) { h.magic = 0; },
      [](auto &h) { ++h.version; },
      [](auto &h) { h.header_bytes = 0; },
      [](auto &h) { h.kind = IndexFileKind::kShard; },
      [](auto &h) { h.flags = 1; },
      [](auto &h) { h.block_count = 0; },
      [](auto &h) { h.block_count = INDEX_MAX_BLOCKS + 1; },
      [](auto &h) { h.reserved32 = 1; },
      [](auto &h) { h.reserved[0] = std::byte{1}; },
      [](auto &h) { h.blocks[0].offset = 1; },
      [](auto &h) { h.blocks[0].bytes = UINT64_MAX; },
      [](auto &h) { h.blocks[0].checksum ^= 1; },
      [](auto &h) { h.meta.dim = 0; },
      [](auto &h) { h.meta.ntotal = 0; },
      [](auto &h) { h.meta.M = 0; },
      [](auto &h) { h.meta.M0 = 0; },
      [](auto &h) { h.meta.ef_search_default = 0; },
      [](auto &h) { h.meta.metric = 2; },
      [](auto &h) { h.meta.id_width = 8; },
      [](auto &h) { h.meta.entry_point = INVALID_NODE_ID; },
      [](auto &h) { h.meta.reserved[0] = 1; }};
  for (std::size_t index = 0; index < mutations.size(); ++index) {
    SCOPED_TRACE(index);
    auto bad = original;
    mutations[index](bad);
    WriteHeader(file, bad);
    EXPECT_FALSE(LoadGlobalIndexFile(root_).ok());
  }
  WriteHeader(file, original);
  EXPECT_TRUE(LoadGlobalIndexFile(root_).ok());
}
TEST_F(IndexExtended, MissingTruncatedAndChangedPayloadAreRejected) {
  EXPECT_FALSE(LoadGlobalIndexFile(root_ / "missing").ok());
  EXPECT_FALSE(LoadIndexForShard(root_, 99).ok());
  const auto file = root_ / "global.meta";
  const auto original = Header(file);
  std::filesystem::resize_file(file, 64);
  EXPECT_FALSE(LoadGlobalIndexFile(root_).ok());
  Build();
  std::fstream stream(file, std::ios::binary | std::ios::in | std::ios::out);
  stream.seekp(static_cast<std::streamoff>(original.blocks[0].offset));
  const char value = 127;
  stream.write(&value, 1);
  stream.close();
  EXPECT_FALSE(LoadGlobalIndexFile(root_).ok());
}
TEST_F(IndexExtended, LocalShardRejectsWrongShapesAndInvalidIds) {
  auto loaded = LoadIndexForShard(root_, 0);
  ASSERT_TRUE(loaded.ok());
  auto &index = loaded.value();
  EXPECT_FALSE(index.local_shard.Vector(INVALID_NODE_ID).ok());
  EXPECT_FALSE(index.local_shard.Neighbors(INVALID_NODE_ID).ok());
  EXPECT_FALSE(LocalShard::Create(index.meta, index.local_shard.range(),
                                  std::vector<node_id_t>{},
                                  std::vector<float>{})
                   .ok());
  EXPECT_FALSE(index.upper_graph.Neighbors(INVALID_NODE_ID, 1).ok());
}
TEST_F(IndexExtended, DatasetRejectsTruncationAndInvalidShape) {
  const auto file = root_ / "dataset.bin";
  FloatDataset data(2, 2, {1, 2, 3, 4});
  ASSERT_TRUE(WriteFloatBin(file, data).ok());
  auto loaded = LoadFloatBin(file);
  ASSERT_TRUE(loaded.ok());
  EXPECT_EQ(loaded.value().values(), data.values());
  EXPECT_TRUE(loaded.value().Row(2).empty());
  EXPECT_TRUE(FloatDataset{}.Row(0).empty());
  EXPECT_TRUE(FloatDataset{}.values().empty());
  std::filesystem::resize_file(file, 7);
  EXPECT_FALSE(LoadFloatBin(file).ok());
  ASSERT_TRUE(WriteFloatBin(file, data).ok());
  std::filesystem::resize_file(file, 12);
  EXPECT_FALSE(LoadFloatBin(file).ok());
  EXPECT_FALSE(LoadFloatBin(root_ / "missing.bin").ok());
  EXPECT_FALSE(WriteFloatBin(root_ / "absent" / "file", data).ok());
}
TEST(WireExtended, RequestHeaderAndDescriptorCorruptionAreRejected) {
  const std::array<node_id_t, 2> ids{1, 2};
  const std::array<float, 2> query{1, 2};
  DistanceBatchRequestTaskInput input;
  input.task_ticket = 1;
  input.query_slot_id = 2;
  input.owner_shard = 0;
  input.flags = kBatchHasQueryVector;
  input.ids = ids;
  input.query_vector = query;
  input.dim = 2;
  const std::span<const DistanceBatchRequestTaskInput> tasks(&input, 1);
  auto size = EncodedDistanceBatchRequestBytes(tasks);
  ASSERT_TRUE(size.ok());
  std::vector<std::byte> message(size.value());
  ASSERT_TRUE(EncodeDistanceBatchRequest(1, tasks, message).ok());
  ASSERT_TRUE(DecodeDistanceBatchRequest(message).ok());
  const auto pristine = message;
  using Mutation = std::function<void(DistanceBatchHeader &)>;
  const std::vector<Mutation> mutations{
      [](auto &h) { h.magic = 0; },
      [](auto &h) { ++h.version; },
      [](auto &h) { h.kind = DistanceBatchKind::kResponse; },
      [](auto &h) { h.flags = 1; },
      [](auto &h) { h.header_bytes = 0; },
      [](auto &h) { h.batch_sequence = 0; },
      [](auto &h) { h.task_table_offset = 0; },
      [](auto &h) { h.payload_offset = 0; },
      [](auto &h) { h.payload_bytes = UINT32_MAX; },
      [](auto &h) { h.total_bytes = UINT32_MAX; }};
  for (std::size_t i = 0; i < mutations.size(); ++i) {
    SCOPED_TRACE(i);
    message = pristine;
    DistanceBatchHeader header;
    std::memcpy(&header, message.data(), sizeof(header));
    mutations[i](header);
    std::memcpy(message.data(), &header, sizeof(header));
    EXPECT_FALSE(DecodeDistanceBatchRequest(message).ok());
  }
  for (int field = 0; field < 6; ++field) {
    message = pristine;
    DistanceBatchRequestTask task;
    std::memcpy(&task, message.data() + sizeof(DistanceBatchHeader),
                sizeof(task));
    switch (field) {
    case 0:
      task.task_ticket = 0;
      break;
    case 1:
      task.reserved = 1;
      break;
    case 2:
      task.flags = UINT32_MAX;
      break;
    case 3:
      task.ids_offset = 0;
      break;
    case 4:
      task.query_offset = 0;
      break;
    case 5:
      task.payload_end = UINT32_MAX;
      break;
    }
    std::memcpy(message.data() + sizeof(DistanceBatchHeader), &task,
                sizeof(task));
    EXPECT_FALSE(DecodeDistanceBatchRequest(message).ok());
  }
  EXPECT_FALSE(DecodeDistanceBatchRequest(std::span(pristine).first(10)).ok());
}
TEST(WireExtended, RequestInputShapeAndCapacityAreChecked) {
  const std::array<node_id_t, 2> ids{1, 2};
  const std::array<float, 2> query{1, 2};
  DistanceBatchRequestTaskInput valid;
  valid.task_ticket = 1;
  valid.ids = ids;
  valid.query_vector = query;
  valid.dim = 2;
  valid.flags = kBatchHasQueryVector;
  for (int field = 0; field < 6; ++field) {
    auto bad = valid;
    switch (field) {
    case 0:
      bad.task_ticket = 0;
      break;
    case 1:
      bad.ids = {};
      break;
    case 2:
      bad.dim = 0;
      break;
    case 3:
      bad.flags = UINT32_MAX;
      break;
    case 4:
      bad.query_vector = std::span(query).first(1);
      break;
    case 5:
      bad.flags = 0;
      break;
    }
    EXPECT_FALSE(EncodedDistanceBatchRequestBytes(std::span(&bad, 1)).ok());
  }
  std::array<std::byte, 8> short_output{};
  EXPECT_FALSE(
      EncodeDistanceBatchRequest(1, std::span(&valid, 1), short_output).ok());
  EXPECT_FALSE(
      DistanceBatchRequestBytesForShape(UINT32_MAX, UINT64_MAX, UINT64_MAX)
          .ok());
  EXPECT_FALSE(DistanceBatchResponseBytesForShape(UINT32_MAX, UINT64_MAX).ok());
}
TEST(WireExtended, ResponseRoundtripAndCorruptionAreChecked) {
  const std::array<node_id_t, 2> ids{1, 2};
  const std::array<float, 2> values{0.25F, 1.5F};
  DistanceBatchResponseTaskInput input;
  input.task_ticket = 1;
  input.query_slot_id = 2;
  input.ids = ids;
  input.distances = values;
  auto size = EncodedDistanceBatchResponseBytes(std::span(&input, 1));
  ASSERT_TRUE(size.ok());
  std::vector<std::byte> message(size.value());
  ASSERT_TRUE(
      EncodeDistanceBatchResponse(7, std::span(&input, 1), message).ok());
  auto view = DecodeDistanceBatchResponse(message);
  ASSERT_TRUE(view.ok());
  auto task = view.value().Task(0);
  ASSERT_TRUE(task.ok());
  EXPECT_EQ(task.value().ids[1], 2U);
  EXPECT_FLOAT_EQ(task.value().distances[1], 1.5F);
  EXPECT_FALSE(view.value().Task(1).ok());
  const auto pristine = message;
  for (int field = 0; field < 7; ++field) {
    message = pristine;
    DistanceBatchResponseTask descriptor;
    std::memcpy(&descriptor, message.data() + sizeof(DistanceBatchHeader),
                sizeof(descriptor));
    switch (field) {
    case 0:
      descriptor.task_ticket = 0;
      break;
    case 1:
      descriptor.reserved = 1;
      break;
    case 2:
      descriptor.ids_offset = 0;
      break;
    case 3:
      descriptor.distances_offset = 0;
      break;
    case 4:
      descriptor.payload_end = UINT32_MAX;
      break;
    case 5:
      descriptor.count = UINT32_MAX;
      break;
    case 6:
      descriptor.distances_offset += 4;
      break;
    }
    std::memcpy(message.data() + sizeof(DistanceBatchHeader), &descriptor,
                sizeof(descriptor));
    EXPECT_FALSE(DecodeDistanceBatchResponse(message).ok());
  }
  EXPECT_FALSE(DecodeDistanceBatchResponse(std::span(pristine).first(10)).ok());
}
TEST(WireExtended, ResponseWriterRejectsInvalidLifecycleAndShapes) {
  std::vector<std::byte> memory(512);
  EXPECT_FALSE(DistanceBatchResponseWriter::Create(0, 1, memory).ok());
  auto writer = DistanceBatchResponseWriter::Create(1, 1, memory);
  ASSERT_TRUE(writer.ok());
  const std::array<node_id_t, 1> ids{2};
  const std::array<float, 1> values{1};
  EXPECT_FALSE(writer.value().Finish().ok());
  EXPECT_FALSE(writer.value().Append(0, 0, ids, values).ok());
  EXPECT_FALSE(writer.value().Append(1, 0, ids, {}).ok());
  ASSERT_TRUE(writer.value().Append(1, 0, ids, values).ok());
  EXPECT_FALSE(writer.value().Append(2, 0, ids, values).ok());
  auto size = writer.value().Finish();
  ASSERT_TRUE(size.ok());
  EXPECT_TRUE(
      DecodeDistanceBatchResponse(std::span(memory).first(size.value())).ok());
}
