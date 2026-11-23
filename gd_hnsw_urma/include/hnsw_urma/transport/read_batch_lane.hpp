#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "hnsw_urma/core/spsc_ring.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/runtime/counting_notification.hpp"
#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {

struct ReadBatchTicket {
  ticket_t value{0};
  std::uint32_t owner_lane{0};
  friend bool operator==(const ReadBatchTicket&, const ReadBatchTicket&) = default;
};

struct ReadRequest {
  std::uint32_t peer_rank{0};
  SegmentKind remote_segment{SegmentKind::kLayer0Graph};
  segment_offset_t remote_offset{0};
  std::span<std::byte> destination;
};

struct ReadResultView {
  const std::byte* data{nullptr};
  std::size_t bytes{0};
  std::uint32_t slot{0};
};

enum class ReadBatchState : std::uint8_t {
  kFree, kPrepared, kQueued, kActive, kSucceeded, kFailed, kTimedOut, kCancelled,
};

struct ReadBatch;
using ReadBatchTerminalCallback = Status (*)(void*, ReadBatch&) noexcept;
using ReadBatchReclaimCallback = void (*)(void*, ReadBatch&) noexcept;
using ReadBatchControlCheck = StatusCode (*)(
    void*, std::chrono::steady_clock::time_point) noexcept;

// Intrusive cursors in the batch's preallocated storage. Progress links each
// compatible interval directly into its peer/segment group; no payload copy.
struct ReadRequestRange {
  ReadBatch* batch{nullptr};
  std::uint32_t next{0};
  std::uint32_t end{0};
  std::size_t group{0};
  std::uint32_t selected{0};
  ReadRequestRange* previous{nullptr};
  ReadRequestRange* following{nullptr};
};

struct ReadBatch {
  ReadBatchTicket ticket{};
  std::uint32_t owner_search{0};
  std::span<const ReadRequest> requests;
  std::vector<ReadRequestRange> ranges;
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
  void* terminal_context{nullptr};
  ReadBatchTerminalCallback terminal_callback{nullptr};
  void* control_context{nullptr};
  ReadBatchControlCheck control_check{nullptr};
  void* reclaim_context{nullptr};
  ReadBatchReclaimCallback reclaim_callback{nullptr};
  std::atomic<ReadBatchState> state{ReadBatchState::kFree};
  StatusCode status{StatusCode::kOk};
  // Written before terminal publication. Only receipt of the actual reclaim
  // event lets Search release a logically completed batch with late DMA.
  bool reclaim_required{false};
  bool reclaim_received{false};  // owner Search only

  // Owner Progress only after publication.
  std::uint32_t completed_reads{0};
  std::uint32_t posted_reads{0};
  std::uint32_t in_flight{0};
  std::size_t active_position{0};
  std::chrono::steady_clock::time_point submitted_at{};
  std::chrono::steady_clock::time_point failed_at{};
};

struct ReadBatchProfile {
  std::uint64_t batches_submitted{0};
  std::uint64_t batches_failed{0};
  std::uint64_t batches_timed_out{0};
  std::uint64_t batches_cancelled{0};
  std::uint64_t requests_posted{0};
  std::uint64_t requests_completed{0};
  std::uint64_t bytes_completed{0};
  std::uint64_t wr_completion_samples{0};
  std::uint64_t wr_completion_nanoseconds{0};
  std::array<std::uint64_t, 64> wr_completion_histogram{};
  std::uint64_t in_flight_high_watermark{0};
  std::uint64_t cqe_failure_count{0};
  std::uint64_t exceptional_batches{0};
  std::uint64_t exceptional_batches_high_watermark{0};
  std::uint64_t late_completion_count{0};
  std::uint64_t submit_queue_high_watermark{0};
  std::uint64_t submit_queue_full_count{0};
  std::uint64_t active_batch_high_watermark{0};
  std::uint64_t post_calls{0};
  std::uint64_t post_list_wrs{0};
  std::uint64_t post_list_high_watermark{0};
  std::uint64_t submit_wait_nanoseconds{0};
  std::uint64_t reclaim_wait_nanoseconds{0};
};

struct ReadBatchLaneConfig {
  std::uint32_t search_workers{1};
  std::uint32_t peer_count{1};
  std::uint32_t submit_depth{64};
  // Sum of Graph and Pull record bounds per Search, also bounds Reclaim[P,S].
  std::uint32_t records_per_search{128};
  std::uint32_t in_flight_limit{128};
  std::size_t wr_list_max{8};
  std::size_t poll_batch{16};
  std::size_t work_budget{64};
  bool profiling_enabled{false};
};

// Submit[S,P] and Reclaim[P,S] are separate SPSC cells. Search owns result
// memory; Progress owns only pending intervals, WR records and CQE handling.
class ReadBatchLane {
 public:
  ReadBatchLane(const ReadBatchLane&) = delete;
  ReadBatchLane& operator=(const ReadBatchLane&) = delete;
  [[nodiscard]] static Result<std::unique_ptr<ReadBatchLane>> Create(
      LanePostEngine& engine, std::uint32_t owner_lane,
      ReadBatchLaneConfig config);
  [[nodiscard]] Result<ReadBatchTicket> PrepareBatch(ReadBatch& batch) noexcept;
  [[nodiscard]] Status CancelPreparedBatch(ReadBatch& batch) noexcept;
  void PublishBatch(ReadBatch& batch) noexcept;
  // Called exclusively by the Search owning this column.
  [[nodiscard]] ReadBatch* PopReclaim(std::uint32_t search) noexcept;
  [[nodiscard]] Status Progress();
  [[nodiscard]] ReadBatchProfile profile() const noexcept;
  [[nodiscard]] bool drained() const noexcept;
  [[nodiscard]] Status ReleaseProgressOwner();
  void SetAdaptiveNotifications(bool enabled) noexcept {
    adaptive_notifications_enabled_ = enabled;
  }
  [[nodiscard]] CountingNotification& adaptive_ready() noexcept { return ready_; }
  [[nodiscard]] std::uint32_t owner_lane() const noexcept { return owner_lane_; }
  [[nodiscard]] const ReadBatchLaneConfig& config() const noexcept { return config_; }

 private:
  struct InFlightRead {
    ReadBatch* batch{nullptr};
    std::uint64_t generation{0};
    std::size_t bytes{0};
    std::chrono::steady_clock::time_point posted_at{};
  };
  struct PostingRead {
    ReadRequestRange* range{nullptr};
    std::uint32_t record{0};
  };
  ReadBatchLane() = default;
  void NotifyWork() noexcept;
  void LinkRange(ReadRequestRange& range) noexcept;
  void UnlinkRange(ReadRequestRange& range) noexcept;
  void RemoveActive(ReadBatch& batch) noexcept;
  [[nodiscard]] Status FinishBatch(ReadBatch& batch, StatusCode status) noexcept;
  [[nodiscard]] Status PollCompletions();
  void DrainSubmissions();
  [[nodiscard]] Status HandleControl();
  [[nodiscard]] Status PostPendingReads();

  LanePostEngine* engine_{nullptr};
  std::uint32_t owner_lane_{0};
  ReadBatchLaneConfig config_;
  std::vector<SpscRing<ReadBatch*>> submit_queues_;
  std::vector<SpscRing<ReadBatch*>> reclaim_queues_;
  std::vector<ticket_t> next_batch_tickets_;  // each cell's producer only
  std::atomic<bool> work_pending_{false};
  CountingNotification ready_;
  bool adaptive_notifications_enabled_{false};
  std::vector<ReadBatch*> active_batches_;
  std::vector<ReadRequestRange*> groups_;
  std::size_t submit_cursor_{0};
  std::size_t group_cursor_{0};
  std::size_t control_cursor_{0};
  std::vector<InFlightRead> in_flight_;
  std::vector<std::uint32_t> free_records_;
  std::vector<ReadOperation> post_operations_;
  std::vector<PostingRead> posting_reads_;
  std::vector<TransportCompletion> completion_batch_;
  ReadBatchProfile profile_;
  Status transport_failure_;
  std::thread::id progress_thread_;
};

}  // namespace hnsw_urma
