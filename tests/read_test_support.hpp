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

#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <vector>

#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/search/read_resources.hpp"
#include "test_framework.hpp"

namespace hnsw_urma::test {

struct ReadTestPool {
  PageAlignedBuffer memory;
  std::unique_ptr<SearchReadResources> pool;
  ReadTestPool(std::uint32_t owner, std::uint32_t slots)
      : memory(PageAlignedBuffer::Create(slots * 64).value()),
        pool(SearchReadResources::Create(owner, memory.logical_memory(), slots,
                                         64)
                 .value()) {}
};

// Controllable DMA and accepted-prefix boundary. A completed READ writes its
// offset pattern only when Complete is called, including after cancellation.
class ControlledReadEngine final : public LanePostEngine {
public:
  Status PostRead(const ReadOperation &op) override {
    return PostReads(std::span(&op, 1)).status;
  }
  Status PostWriteNotify(const WriteNotifyOperation &) override {
    return Status(StatusCode::kInvalidArgument, "READ-only test engine");
  }
  LanePostBatchResult
  PostReads(std::span<const ReadOperation> operations) override {
    posts.emplace_back(operations.begin(), operations.end());
    const auto count =
        std::min({operations.size(), accept_limit, capacity - pending.size()});
    for (const auto &operation : operations.first(count)) {
      pending.push_back(operation);
      accepted.push_back(operation);
    }
    return {count,
            count == operations.size() ? Status::Ok()
                                       : Status(StatusCode::kResourceExhausted,
                                                "injected prefix capacity"),
            0, count == 0 ? 0U : 1U};
  }
  Result<std::size_t> Poll(std::span<TransportCompletion> output) override {
    const auto count = std::min(output.size(), completions.size());
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = completions.front();
      completions.pop_front();
    }
    return count;
  }
  void Complete(std::size_t index = 0, Status status = Status::Ok()) {
    const auto operation = pending.at(index);
    pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(index));
    if (status.ok())
      std::fill(operation.destination.begin(), operation.destination.end(),
                static_cast<std::byte>(operation.remote_offset + 1));
    completions.push_back({operation.ticket, OperationClass::kRead, status,
                           operation.destination.size()});
  }
  std::size_t accept_limit{1024};
  std::size_t capacity{1024};
  std::vector<std::vector<ReadOperation>> posts;
  std::vector<ReadOperation> accepted;
  std::vector<ReadOperation> pending;
  std::deque<TransportCompletion> completions;
};

struct ReadTestBatch {
  SearchReadResources *pool;
  ReadBatch batch;
  std::vector<ReadRequest> requests;
  std::vector<ReadResultView> results;
  std::atomic<bool> cancel{false};
  std::size_t terminal_calls{0};
  bool consumed{false};
  bool returned{false};
  explicit ReadTestBatch(SearchReadResources &resource,
                         std::initializer_list<std::uint64_t> offsets,
                         std::uint32_t peer = 1)
      : pool(&resource), results(offsets.size()) {
    auto acquired = pool->Acquire(results, 4);
    UB_REQUIRE_TRUE(acquired.ok());
    std::size_t index = 0;
    for (const auto offset : offsets) {
      auto view = results[index++];
      requests.push_back({peer,
                          SegmentKind::kVectors,
                          offset,
                          {const_cast<std::byte *>(view.data), view.bytes}});
    }
    batch.owner_search = pool->owner_search();
    batch.requests = requests;
    batch.ranges.reserve(requests.size());
    batch.terminal_context = batch.control_context = batch.reclaim_context =
        this;
    batch.terminal_callback = [](void *context, ReadBatch &) noexcept {
      ++static_cast<ReadTestBatch *>(context)->terminal_calls;
      return Status::Ok();
    };
    batch.control_check = [](void *context,
                             std::chrono::steady_clock::time_point) noexcept {
      return static_cast<ReadTestBatch *>(context)->cancel.load()
                 ? StatusCode::kCancelled
                 : StatusCode::kOk;
    };
    batch.reclaim_callback = [](void *context, ReadBatch &) noexcept {
      auto &self = *static_cast<ReadTestBatch *>(context);
      if (self.consumed)
        self.Consume();
    };
  }
  void Publish(ReadBatchLane &lane) {
    UB_REQUIRE_TRUE(lane.PrepareBatch(batch).ok());
    lane.PublishBatch(batch);
  }
  void Consume() noexcept {
    consumed = true;
    if (!returned && (!batch.reclaim_required || batch.reclaim_received)) {
      pool->Return(results);
      returned = true;
    }
  }
};

inline void DrainReads(ControlledReadEngine &engine, ReadBatchLane &lane) {
  for (int iteration = 0; iteration < 1000 && !lane.drained(); ++iteration) {
    while (!engine.pending.empty())
      engine.Complete(engine.pending.size() - 1);
    UB_REQUIRE_TRUE(lane.Progress().ok());
  }
  UB_REQUIRE_TRUE(lane.drained());
}
} // namespace hnsw_urma::test
