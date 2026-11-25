// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include "hnsw_urma/runtime/drain.hpp"

TEST(ReferenceRuntimeDrain,
     DataPlaneDrainRequiresTwoConsensusPhasesAndZeroOwnership) {
  auto tracker = hnsw_urma::DataPlaneDrainTracker::Create(2).value();
  tracker->PublishActiveQueries(1);
  UB_REQUIRE_TRUE(tracker->StopAdmission().ok());
  UB_REQUIRE_FALSE(tracker->MarkLocalQueriesComplete().ok());
  tracker->PublishActiveQueries(0);
  UB_REQUIRE_TRUE(tracker->MarkLocalQueriesComplete().ok());
  UB_REQUIRE_TRUE(tracker->BeginDrainAfterRankConsensus().ok());

  hnsw_urma::LaneDrainSnapshot busy;
  busy.request_slots_active = 1;
  busy.quarantine_records = 1;
  tracker->PublishLane(0, busy);
  tracker->PublishLane(1, {});
  UB_REQUIRE_FALSE(tracker->TryMarkDrained().ok());
  UB_REQUIRE_EQ(tracker->phase(), hnsw_urma::DrainPhase::kDraining);
  tracker->PublishLane(0, {});
  UB_REQUIRE_TRUE(tracker->TryMarkDrained().ok());
  UB_REQUIRE_TRUE(tracker->MarkGlobalDrainConsensus().ok());
  UB_REQUIRE_TRUE(tracker->MarkStopped().ok());
  UB_REQUIRE_EQ(tracker->phase(), hnsw_urma::DrainPhase::kStopped);
}

TEST(ReferenceRuntimeDrain, DataPlaneDrainWaitRetriesTransientOwnership) {
  auto tracker = hnsw_urma::DataPlaneDrainTracker::Create(1).value();
  UB_REQUIRE_TRUE(tracker->StopAdmission().ok());
  UB_REQUIRE_TRUE(tracker->MarkLocalQueriesComplete().ok());
  UB_REQUIRE_TRUE(tracker->BeginDrainAfterRankConsensus().ok());

  hnsw_urma::LaneDrainSnapshot busy;
  busy.rpc_active = 1;
  tracker->PublishLane(0, busy);
  std::uint32_t health_checks = 0;
  auto status = tracker->WaitUntilDrained(
      std::chrono::steady_clock::now() + std::chrono::seconds(1), [&] {
        ++health_checks;
        if (health_checks == 2) {
          tracker->PublishLane(0, {});
        }
        return hnsw_urma::Status::Ok();
      });
  UB_REQUIRE_TRUE(status.ok());
  UB_REQUIRE_EQ(health_checks, 2U);
  UB_REQUIRE_EQ(tracker->phase(), hnsw_urma::DrainPhase::kDrained);
}

TEST(ReferenceRuntimeDrain,
     DataPlaneDrainSnapshotsAreConsistentDuringLanePublication) {
  auto tracker = hnsw_urma::DataPlaneDrainTracker::Create(1).value();
  std::atomic<bool> finished{false};
  std::atomic<std::uint32_t> mismatches{0};
  std::thread lane([&] {
    for (std::uint64_t value = 1; value < 10000; ++value) {
      hnsw_urma::LaneDrainSnapshot snapshot;
      snapshot.read_active = value;
      snapshot.server_result_active = value;
      snapshot.quarantine_records = value;
      tracker->PublishLane(0, snapshot);
    }
    finished.store(true, std::memory_order_release);
  });
  while (!finished.load(std::memory_order_acquire)) {
    auto snapshot = tracker->ReadLane(0).value();
    if (snapshot.read_active != snapshot.server_result_active ||
        snapshot.read_active != snapshot.quarantine_records) {
      mismatches.fetch_add(1, std::memory_order_relaxed);
    }
  }
  lane.join();
  UB_REQUIRE_EQ(mismatches.load(), 0U);
}

TEST(ReferenceRuntimeDrain,
     LaneFailuresChooseStableSeverityLaneAndOperationOrder) {
  const std::array<hnsw_urma::LaneFailureRecord, 4> failures{{
      {hnsw_urma::FailureSeverity::kQuery, 0, 1,
       hnsw_urma::StatusCode::kTimeout},
      {hnsw_urma::FailureSeverity::kRankFatal, 2, 1,
       hnsw_urma::StatusCode::kProtocolError},
      {hnsw_urma::FailureSeverity::kRankFatal, 1, 9,
       hnsw_urma::StatusCode::kTransportError},
      {hnsw_urma::FailureSeverity::kRankFatal, 1, 3,
       hnsw_urma::StatusCode::kInternalError},
  }};
  auto selected = hnsw_urma::SelectPrimaryFailure(failures).value();
  UB_REQUIRE_EQ(selected.severity, hnsw_urma::FailureSeverity::kRankFatal);
  UB_REQUIRE_EQ(selected.lane, 1U);
  UB_REQUIRE_EQ(selected.operation_slot, 3U);
}
