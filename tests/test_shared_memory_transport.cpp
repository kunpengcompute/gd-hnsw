/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>

#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/transport/shared_memory_transport.hpp"

namespace hnsw_urma {
namespace {

OperationTicket SharedTicket(std::uint64_t value, std::uint32_t lane) {
  return {value, static_cast<std::uint32_t>(value - 1U), lane, 1};
}

class RecordingFallback final : public LanePostEngine {
public:
  Status PostRead(const ReadOperation &) override {
    ++read_posts;
    return Status::Ok();
  }
  Status PostWriteNotify(const WriteNotifyOperation &) override {
    ++write_posts;
    return Status::Ok();
  }
  Result<std::size_t> Poll(std::span<TransportCompletion>) override {
    return std::size_t{0};
  }

  std::size_t read_posts{0};
  std::size_t write_posts{0};
};

} // namespace

TEST(ReferenceMockSharedMemoryTransport,
     SharedMemoryLaneReadsAndPublishesWriteNotify) {
  const std::array<std::byte, 8> vector_source{
      std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
      std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
  auto vectors = PageAlignedBuffer::CopyFromShared(vector_source);
  auto requests = PageAlignedBuffer::CreateShared(64);
  auto notify = PageAlignedBuffer::CreateShared(64);
  UB_REQUIRE_TRUE(vectors.ok());
  UB_REQUIRE_TRUE(requests.ok());
  UB_REQUIRE_TRUE(notify.ok());

  const std::array<SharedMemorySegmentDescriptor, 3> descriptors{
      SharedMemorySegmentDescriptor{
          SegmentKind::kVectors,
          std::string(vectors.value().shared_memory_name()),
          vectors.value().logical_memory().size()},
      SharedMemorySegmentDescriptor{
          SegmentKind::kRequests,
          std::string(requests.value().shared_memory_name()),
          requests.value().logical_memory().size()},
      SharedMemorySegmentDescriptor{
          SegmentKind::kNotify,
          std::string(notify.value().shared_memory_name()),
          notify.value().logical_memory().size()}};
  auto peer = SharedMemoryPeer::Open(2, descriptors);
  UB_REQUIRE_TRUE(peer.ok());

  auto fallback = std::make_unique<RecordingFallback>();
  auto *fallback_view = fallback.get();
  SharedMemoryLaneOptions options;
  options.owner_lane = 3;
  options.peers.push_back(peer.value());
  options.fallback = std::move(fallback);
  options.max_outstanding = 8;
  options.batch_max = 4;
  auto engine = SharedMemoryLanePostEngine::Create(std::move(options));
  UB_REQUIRE_TRUE(engine.ok());

  std::array<std::byte, 3> read_destination{};
  ReadOperation read{SharedTicket(1, 3), 2, SegmentKind::kVectors,
                     read_destination, 2};
  UB_REQUIRE_TRUE(engine.value()->PostRead(read).ok());
  UB_REQUIRE_EQ(read_destination[0], std::byte{3});
  UB_REQUIRE_EQ(read_destination[2], std::byte{5});

  const std::array<std::byte, 4> payload{std::byte{9}, std::byte{10},
                                         std::byte{11}, std::byte{12}};
  WriteNotifyOperation write{OperationClass::kRequestWrite,
                             SharedTicket(2, 3),
                             2,
                             SegmentKind::kRequests,
                             payload,
                             8,
                             SegmentKind::kNotify,
                             0,
                             41,
                             true};
  UB_REQUIRE_TRUE(engine.value()->PostWriteNotify(write).ok());
  UB_REQUIRE_EQ(requests.value().logical_memory()[8], std::byte{9});
  std::uint64_t observed = 0;
  std::memcpy(&observed, notify.value().logical_memory().data(),
              sizeof(observed));
  UB_REQUIRE_EQ(observed, 41U);

  std::array<TransportCompletion, 4> completions{};
  auto completed = engine.value()->Poll(completions);
  UB_REQUIRE_TRUE(completed.ok());
  UB_REQUIRE_EQ(completed.value(), 2U);
  UB_REQUIRE_EQ(completions[0].operation_class, OperationClass::kRead);
  UB_REQUIRE_EQ(completions[1].operation_class, OperationClass::kRequestWrite);

  // Partial capacity accepts a shorter prefix into explicitly supplied
  // destinations, including when fewer credits remain than one list.
  std::array<std::byte, 8> prefix_destinations{};
  std::array<ReadOperation, 8> prefix_reads{};
  for (std::size_t index = 0; index < prefix_reads.size(); ++index)
    prefix_reads[index] = {
        SharedTicket(index + 10, 3), 2, SegmentKind::kVectors,
        std::span(prefix_destinations).subspan(index, 1), index};
  auto first_prefix = engine.value()->PostReads(prefix_reads);
  UB_REQUIRE_EQ(first_prefix.posted, 4U);
  auto second_prefix =
      engine.value()->PostReads(std::span(prefix_reads).subspan(4, 3));
  UB_REQUIRE_EQ(second_prefix.posted, 3U);
  auto last_credit = engine.value()->PostReads(prefix_reads);
  UB_REQUIRE_EQ(last_credit.posted, 1U);
  UB_REQUIRE_EQ(engine.value()->PostReads(prefix_reads).posted, 0U);
  UB_REQUIRE_EQ(engine.value()->Poll(completions).value(), 4U);
  UB_REQUIRE_EQ(engine.value()->Poll(completions).value(), 4U);
  for (std::size_t index = 0; index < 7; ++index)
    UB_REQUIRE_EQ(prefix_destinations[index], vector_source[index]);

  ReadOperation remote_read{SharedTicket(3, 3), 9, SegmentKind::kVectors,
                            read_destination, 0};
  UB_REQUIRE_TRUE(engine.value()->PostRead(remote_read).ok());
  UB_REQUIRE_EQ(fallback_view->read_posts, 1U);
  UB_REQUIRE_EQ(fallback_view->write_posts, 0U);
}

} // namespace hnsw_urma
