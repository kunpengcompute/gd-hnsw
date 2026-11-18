#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {

struct SharedMemorySegmentDescriptor {
  SegmentKind kind{SegmentKind::kLayer0Graph};
  std::string name;
  std::size_t bytes{0};
};

// One mapped same-host peer. Instances are shared by all local progress lanes
// so each peer segment is mapped only once per process.
class SharedMemoryPeer {
 public:
  SharedMemoryPeer(const SharedMemoryPeer&) = delete;
  SharedMemoryPeer& operator=(const SharedMemoryPeer&) = delete;

  [[nodiscard]] static Result<std::shared_ptr<SharedMemoryPeer>> Open(
      std::uint32_t rank,
      std::span<const SharedMemorySegmentDescriptor> segments);

  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }

 private:
  struct Segment {
    SegmentKind kind{SegmentKind::kLayer0Graph};
    PageAlignedBuffer memory;
  };

  SharedMemoryPeer() = default;
  [[nodiscard]] Segment* Find(SegmentKind kind) noexcept;

  std::uint32_t rank_{0};
  std::vector<Segment> segments_;

  friend class SharedMemoryLanePostEngine;
};

struct SharedMemoryLaneOptions {
  std::uint32_t owner_lane{0};
  std::vector<std::shared_ptr<SharedMemoryPeer>> peers;
  std::unique_ptr<LanePostEngine> fallback;
  std::size_t max_outstanding{1024};
  std::size_t batch_max{8};
};

// Routes same-host peers through mapped memory and all other peers through the
// existing transport. The class remains single-owner, matching LanePostEngine.
class SharedMemoryLanePostEngine final : public LanePostEngine {
 public:
  SharedMemoryLanePostEngine(const SharedMemoryLanePostEngine&) = delete;
  SharedMemoryLanePostEngine& operator=(const SharedMemoryLanePostEngine&) =
      delete;
  ~SharedMemoryLanePostEngine() override;

  [[nodiscard]] static Result<std::unique_ptr<LanePostEngine>> Create(
      SharedMemoryLaneOptions options);

  Status PostRead(const ReadOperation& operation) override;
  Status PostWriteNotify(const WriteNotifyOperation& operation) override;
  LanePostBatchResult PostReads(
      std::span<const ReadOperation> operations) override;
  LanePostBatchResult PostWriteNotifies(
      std::span<const WriteNotifyOperation> operations) override;
  Result<std::size_t> Poll(
      std::span<TransportCompletion> completions) override;

 private:
  struct Impl;
  explicit SharedMemoryLanePostEngine(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> impl_;
};

}  // namespace hnsw_urma
