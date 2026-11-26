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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "hnsw_urma/protocol/distance_batch_wire.hpp"

namespace hnsw_urma {
namespace {

template <std::size_t N> struct AlignedBytes {
  alignas(8) std::array<std::byte, N> bytes{};
};

DistanceBatchHeader ReadHeader(std::span<const std::byte> bytes) {
  DistanceBatchHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  return header;
}

void WriteHeader(std::span<std::byte> bytes,
                 const DistanceBatchHeader &header) {
  std::memcpy(bytes.data(), &header, sizeof(header));
}

} // namespace

TEST(ReferenceFormatDistanceBatchWire, DistanceBatchVNextLayoutIsFixed) {
  UB_REQUIRE_EQ(sizeof(DistanceBatchHeader), 40U);
  UB_REQUIRE_EQ(sizeof(DistanceBatchRequestTask), 48U);
  UB_REQUIRE_EQ(sizeof(DistanceBatchResponseTask), 32U);
  UB_REQUIRE_EQ(DISTANCE_BATCH_WIRE_VERSION, 8U);
}

TEST(ReferenceFormatDistanceBatchWire,
     DistanceBatchRequestRoundTripsCanonicalTaskOrder) {
  const std::array<node_id_t, 2> first_ids{7, 9};
  const std::array<float, 3> first_query{1.0F, 2.0F, 3.0F};
  const std::array<node_id_t, 1> second_ids{11};
  const std::array<DistanceBatchRequestTaskInput, 2> tasks{{
      {41, 3, 2, kBatchHasQueryVector | kBatchUseDotNorm, first_ids,
       first_query, 3, 0.0F},
      {42, 8, 2, kBatchHasUpperDistance, second_ids, {}, 3, 1.25F},
  }};
  AlignedBytes<1024> storage;
  auto encoded = EncodeDistanceBatchRequest(17, tasks, storage.bytes);
  UB_REQUIRE_TRUE(encoded.ok());
  UB_REQUIRE_EQ(encoded.value(),
                EncodedDistanceBatchRequestBytes(tasks).value());

  auto decoded = DecodeDistanceBatchRequest(
      std::span<const std::byte>(storage.bytes).first(encoded.value()));
  UB_REQUIRE_TRUE(decoded.ok());
  UB_REQUIRE_EQ(decoded.value().header().batch_sequence, 17U);
  UB_REQUIRE_EQ(decoded.value().task_count(), 2U);
  auto first = decoded.value().Task(0);
  auto second = decoded.value().Task(1);
  UB_REQUIRE_TRUE(first.ok());
  UB_REQUIRE_TRUE(second.ok());
  UB_REQUIRE_EQ(first.value().descriptor.task_ticket, 41U);
  UB_REQUIRE_EQ(first.value().descriptor.query_slot_id, 3U);
  UB_REQUIRE_EQ(first.value().descriptor.owner_shard, 2U);
  UB_REQUIRE_EQ(first.value().ids[1], 9U);
  UB_REQUIRE_EQ(first.value().query_vector[2], 3.0F);
  UB_REQUIRE_EQ(second.value().descriptor.query_slot_id, 8U);
  UB_REQUIRE_EQ(second.value().descriptor.upper_distance, 1.25F);
  UB_REQUIRE_TRUE(second.value().query_vector.empty());
  UB_REQUIRE_FALSE(decoded.value().Task(2).ok());
}

TEST(ReferenceFormatDistanceBatchWire,
     DistanceBatchResponseWriterPreservesEmptyTaskPosition) {
  const std::array<node_id_t, 2> ids{7, 9};
  const std::array<float, 2> distances{0.5F, 1.5F};
  AlignedBytes<1024> storage;
  auto writer = DistanceBatchResponseWriter::Create(33, 2, storage.bytes);
  UB_REQUIRE_TRUE(writer.ok());
  UB_REQUIRE_TRUE(writer.value().Append(91, 7, {}, {}).ok());
  UB_REQUIRE_TRUE(writer.value().Append(92, 8, ids, distances).ok());
  auto bytes = writer.value().Finish();
  UB_REQUIRE_TRUE(bytes.ok());

  auto decoded = DecodeDistanceBatchResponse(
      std::span<const std::byte>(storage.bytes).first(bytes.value()));
  UB_REQUIRE_TRUE(decoded.ok());
  UB_REQUIRE_EQ(decoded.value().header().batch_sequence, 33U);
  UB_REQUIRE_EQ(decoded.value().task_count(), 2U);
  auto empty = decoded.value().Task(0);
  auto full = decoded.value().Task(1);
  UB_REQUIRE_TRUE(empty.ok());
  UB_REQUIRE_TRUE(full.ok());
  UB_REQUIRE_EQ(empty.value().descriptor.task_ticket, 91U);
  UB_REQUIRE_EQ(empty.value().descriptor.query_slot_id, 7U);
  UB_REQUIRE_TRUE(empty.value().ids.empty());
  UB_REQUIRE_TRUE(empty.value().distances.empty());
  UB_REQUIRE_EQ(full.value().ids[1], 9U);
  UB_REQUIRE_EQ(full.value().distances[1], 1.5F);
}

TEST(ReferenceFormatDistanceBatchWire,
     DistanceBatchRejectsStaleAndNonCanonicalEnvelopes) {
  const std::array<node_id_t, 1> ids{7};
  const std::array<float, 2> query{1.0F, 2.0F};
  const DistanceBatchRequestTaskInput task{41,  3,     2, kBatchHasQueryVector,
                                           ids, query, 2, 0.0F};
  AlignedBytes<256> storage;
  auto encoded = EncodeDistanceBatchRequest(
      9, std::span<const DistanceBatchRequestTaskInput>(&task, 1),
      storage.bytes);
  UB_REQUIRE_TRUE(encoded.ok());

  auto header = ReadHeader(storage.bytes);
  header.version = DISTANCE_BATCH_WIRE_VERSION - 1;
  WriteHeader(storage.bytes, header);
  UB_REQUIRE_FALSE(DecodeDistanceBatchRequest(storage.bytes).ok());

  header.version = DISTANCE_BATCH_WIRE_VERSION;
  header.batch_sequence = 0;
  WriteHeader(storage.bytes, header);
  UB_REQUIRE_FALSE(DecodeDistanceBatchRequest(storage.bytes).ok());

  header.batch_sequence = 9;
  header.payload_bytes += 1;
  WriteHeader(storage.bytes, header);
  UB_REQUIRE_FALSE(DecodeDistanceBatchRequest(storage.bytes).ok());
}

TEST(ReferenceFormatDistanceBatchWire,
     DistanceBatchShapeRejectsEmptyBatchesAndSmallBuffers) {
  UB_REQUIRE_FALSE(DistanceBatchRequestBytesForShape(0, 0, 0).ok());
  UB_REQUIRE_FALSE(DistanceBatchResponseBytesForShape(0, 0).ok());
  const std::array<node_id_t, 1> ids{7};
  const std::array<float, 1> distances{1.0F};
  const DistanceBatchResponseTaskInput task{1, 2, distances, ids};
  AlignedBytes<40> storage;
  UB_REQUIRE_FALSE(
      EncodeDistanceBatchResponse(
          1, std::span<const DistanceBatchResponseTaskInput>(&task, 1),
          storage.bytes)
          .ok());
}

} // namespace hnsw_urma
