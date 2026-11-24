#include "hnsw_urma/runtime/drain.hpp"

#include <algorithm>
#include <array>
#include <new>
#include <thread>

namespace hnsw_urma {

bool LaneDrainSnapshot::empty() const noexcept {
  return read_active == 0 && read_backlog == 0 && rpc_active == 0 &&
         request_slots_active == 0 && server_result_active == 0 &&
         reply_active == 0 && completion_pending == 0 &&
         quarantine_records == 0;
}

Result<std::unique_ptr<DataPlaneDrainTracker>>
DataPlaneDrainTracker::Create(std::uint32_t communication_owner_count) {
  if (communication_owner_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "drain tracker requires progress lanes");
  }
  try {
    auto tracker =
        std::unique_ptr<DataPlaneDrainTracker>(new DataPlaneDrainTracker());
    tracker->communication_owner_count_ = communication_owner_count;
    tracker->lanes_ = std::make_unique<AtomicLane[]>(communication_owner_count);
    return tracker;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate data-plane drain tracker");
  }
}

void DataPlaneDrainTracker::PublishLane(std::uint32_t lane,
                                        LaneDrainSnapshot snapshot) noexcept {
  if (lane >= communication_owner_count_) {
    return;
  }
  auto &target = lanes_[lane];
  // These samples are published only during shutdown. Sequential consistency
  // keeps the sequence and all field atomics in one total order, which is
  // required for a portable multi-word snapshot (acquire/release alone does
  // not prevent a reader from observing mixed field generations).
  target.sequence.fetch_add(1, std::memory_order_seq_cst);
  const std::uint64_t values[kSnapshotFields]{
      snapshot.read_active,          snapshot.read_backlog,
      snapshot.rpc_active,           snapshot.request_slots_active,
      snapshot.server_result_active, snapshot.reply_active,
      snapshot.completion_pending,   snapshot.quarantine_records,
  };
  for (std::size_t index = 0; index < kSnapshotFields; ++index) {
    target.fields[index].store(values[index], std::memory_order_seq_cst);
  }
  target.sequence.fetch_add(1, std::memory_order_seq_cst);
}

Result<LaneDrainSnapshot>
DataPlaneDrainTracker::ReadLane(std::uint32_t lane) const noexcept {
  if (lane >= communication_owner_count_) {
    return Status(StatusCode::kOutOfRange,
                  "drain snapshot lane is out of range");
  }
  const auto &source = lanes_[lane];
  std::array<std::uint64_t, kSnapshotFields> values{};
  std::uint32_t retries = 0;
  for (bool consistent = false; !consistent;) {
    const auto before = source.sequence.load(std::memory_order_seq_cst);
    if ((before & 1U) != 0) {
      if ((++retries & 63U) == 0) {
        std::this_thread::yield();
      }
      continue;
    }
    for (std::size_t index = 0; index < kSnapshotFields; ++index) {
      values[index] = source.fields[index].load(std::memory_order_seq_cst);
    }
    const auto after = source.sequence.load(std::memory_order_seq_cst);
    if (before == after) {
      consistent = true;
    } else if ((++retries & 63U) == 0) {
      std::this_thread::yield();
    }
  }
  return LaneDrainSnapshot{values[0], values[1], values[2], values[3],
                           values[4], values[5], values[6], values[7]};
}

void DataPlaneDrainTracker::PublishActiveQueries(
    std::uint32_t active) noexcept {
  active_queries_.store(active, std::memory_order_release);
}

Status DataPlaneDrainTracker::StopAdmission() noexcept {
  DrainPhase expected = DrainPhase::kRunning;
  if (!phase_.compare_exchange_strong(expected, DrainPhase::kAdmissionStopped,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "admission can only stop from Running");
  }
  return Status::Ok();
}

Status DataPlaneDrainTracker::MarkLocalQueriesComplete() noexcept {
  if (active_queries_.load(std::memory_order_acquire) != 0) {
    return Status(StatusCode::kResourceExhausted,
                  "local queries are still active");
  }
  DrainPhase expected = DrainPhase::kAdmissionStopped;
  if (!phase_.compare_exchange_strong(
          expected, DrainPhase::kLocalQueriesComplete,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "local completion requires stopped admission");
  }
  return Status::Ok();
}

Status DataPlaneDrainTracker::BeginDrainAfterRankConsensus() noexcept {
  DrainPhase expected = DrainPhase::kLocalQueriesComplete;
  if (!phase_.compare_exchange_strong(expected, DrainPhase::kDraining,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "global drain requires local-query consensus");
  }
  return Status::Ok();
}

bool DataPlaneDrainTracker::all_lanes_empty() const noexcept {
  for (std::uint32_t lane = 0; lane < communication_owner_count_; ++lane) {
    auto snapshot = ReadLane(lane);
    if (!snapshot.ok() || !snapshot.value().empty()) {
      return false;
    }
  }
  return true;
}

Status DataPlaneDrainTracker::TryMarkDrained() noexcept {
  if (!all_lanes_empty()) {
    return Status(StatusCode::kResourceExhausted,
                  "data plane still owns resources");
  }
  DrainPhase expected = DrainPhase::kDraining;
  if (!phase_.compare_exchange_strong(expected, DrainPhase::kDrained,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "data plane can only drain from Draining");
  }
  return Status::Ok();
}

Status DataPlaneDrainTracker::WaitUntilDrained(
    std::chrono::steady_clock::time_point deadline,
    const std::function<Status()> &health_check) {
  if (!health_check) {
    return Status(StatusCode::kInvalidArgument,
                  "data-plane drain health check is unavailable");
  }
  for (bool drained = false; !drained;) {
    auto status = TryMarkDrained();
    if (status.ok() || status.code() != StatusCode::kResourceExhausted) {
      return status;
    }
    status = health_check();
    if (!status.ok()) {
      return status;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return Status(StatusCode::kTimeout,
                    "local data plane did not drain before deadline");
    }
    std::this_thread::yield();
  }
  // Unreachable: the loop above only exits via one of its return statements.
  return Status(StatusCode::kInternalError, "drain loop exited unexpectedly");
}

Status DataPlaneDrainTracker::MarkGlobalDrainConsensus() noexcept {
  DrainPhase expected = DrainPhase::kDrained;
  if (!phase_.compare_exchange_strong(expected, DrainPhase::kGlobalConsensus,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "global consensus requires local drain");
  }
  return Status::Ok();
}

Status DataPlaneDrainTracker::MarkStopped() noexcept {
  DrainPhase expected = DrainPhase::kGlobalConsensus;
  if (!phase_.compare_exchange_strong(expected, DrainPhase::kStopped,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "runtime stop requires global drain consensus");
  }
  return Status::Ok();
}

Result<LaneFailureRecord>
SelectPrimaryFailure(std::span<const LaneFailureRecord> failures) noexcept {
  if (failures.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "cannot select from an empty failure set");
  }
  const auto selected = std::min_element(
      failures.begin(), failures.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.severity != rhs.severity) {
          return lhs.severity > rhs.severity;
        }
        if (lhs.lane != rhs.lane) {
          return lhs.lane < rhs.lane;
        }
        return lhs.operation_slot < rhs.operation_slot;
      });
  return *selected;
}

} // namespace hnsw_urma
