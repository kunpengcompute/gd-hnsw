#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {

class MockLanePostEngine final : public LanePostEngine {
 public:
  [[nodiscard]] static Result<MockLanePostEngine> Create(std::uint32_t owner_lane,
                                                         std::uint32_t max_outstanding);
  [[nodiscard]] Status RegisterSegment(std::uint32_t peer_rank, SegmentKind kind,
                                       std::span<std::byte> memory);

  Status PostRead(const ReadOperation& operation) override;
  Status PostWriteNotify(const WriteNotifyOperation& operation) override;
  LanePostBatchResult PostWriteNotifies(
      std::span<const WriteNotifyOperation> operations) override;
  Result<std::size_t> Poll(std::span<TransportCompletion> completions) override;

  [[nodiscard]] Status Progress(std::size_t event_index = 0);
  [[nodiscard]] Status Drop(std::size_t event_index = 0);
  void FailNextPost(Status status) { next_post_failure_ = std::move(status); }
  void InjectCompletion(TransportCompletion completion);

  [[nodiscard]] std::size_t pending_events() const noexcept { return events_.size(); }
  [[nodiscard]] std::size_t queued_completions() const noexcept { return completions_.size(); }
  [[nodiscard]] std::size_t write_notify_batch_calls() const noexcept {
    return write_notify_batch_calls_;
  }
  [[nodiscard]] std::size_t maximum_write_notify_batch() const noexcept {
    return maximum_write_notify_batch_;
  }
  [[nodiscard]] std::size_t inline_write_notify_posts() const noexcept {
    return inline_write_notify_posts_;
  }

 private:
  struct Segment {
    std::uint32_t peer_rank{0};
    SegmentKind kind{SegmentKind::kLayer0Graph};
    std::span<std::byte> memory;
  };

  struct Event {
    bool is_read{true};
    ReadOperation read;
    WriteNotifyOperation write;
    std::vector<std::byte> inline_payload;
  };

  [[nodiscard]] Result<std::span<std::byte>> FindSegment(std::uint32_t peer_rank,
                                                         SegmentKind kind) noexcept;
  [[nodiscard]] Status ValidateTicket(const OperationTicket& ticket) const noexcept;

  std::uint32_t owner_lane_{0};
  std::uint32_t max_outstanding_{0};
  std::vector<Segment> segments_;
  std::vector<Event> events_;
  std::vector<TransportCompletion> completions_;
  Status next_post_failure_;
  std::size_t write_notify_batch_calls_{0};
  std::size_t maximum_write_notify_batch_{0};
  std::size_t inline_write_notify_posts_{0};
};

}  // namespace hnsw_urma
