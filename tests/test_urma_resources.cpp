// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "hnsw_urma/transport/urma_resources.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "test_framework.hpp"

namespace hnsw_urma {
namespace {

UrmaCapabilities Capabilities(std::uint32_t lanes) {
  return {.max_jfs = lanes * 3U,
          .max_jfr = lanes * 3U,
          .max_jfc = lanes * 3U + 1U,
          .max_jfs_depth = 128,
          .max_jfr_depth = 8,
          .max_jfc_depth = 128,
          .max_inline_bytes = 256,
          .max_read_bytes = 1U << 20U,
          .max_write_bytes = 1U << 20U};
}

UrmaResourceOptions Options(std::uint32_t lanes) {
  UrmaResourceOptions options;
  options.communication_lanes = lanes;
  options.read_progress_threads = lanes / 2U;
  options.rpc_progress_threads = lanes - options.read_progress_threads;
  return options;
}

class FakeResourceBackend final : public UrmaResourceBackend {
public:
  Result<UrmaResourceHandle>
  Create(const UrmaResourceRequest &request) override {
    ++attempts;
    if (attempts == fail_at) {
      return Status(StatusCode::kTransportError,
                    "injected resource creation failure");
    }
    const UrmaResourceHandle handle{next_handle++};
    created.push_back({request.kind, handle});
    requests.push_back(request);
    return handle;
  }

  void Destroy(UrmaResourceKind kind,
               UrmaResourceHandle handle) noexcept override {
    destroyed.push_back({kind, handle});
  }

  std::uint64_t next_handle{1};
  std::size_t attempts{0};
  std::size_t fail_at{std::numeric_limits<std::size_t>::max()};
  std::vector<std::pair<UrmaResourceKind, UrmaResourceHandle>> created;
  std::vector<std::pair<UrmaResourceKind, UrmaResourceHandle>> destroyed;
  std::vector<UrmaResourceRequest> requests;
};

class FakePoster final : public UrmaWorkRequestPoster {
public:
  ProviderPostResult
  Post(std::uint64_t jfs_key,
       std::span<const OpportunisticPostDescriptor> descriptors) override {
    seen_key = jfs_key;
    seen.assign(descriptors.begin(), descriptors.end());
    return result;
  }

  std::uint64_t seen_key{0};
  std::vector<OpportunisticPostDescriptor> seen;
  ProviderPostResult result;
};

class FakeVisibility final : public DmaVisibilityDomain {
public:
  Status AcquireDeviceWrites() override {
    ++calls;
    if (fail_on_call == calls) {
      return Status(StatusCode::kTransportError, "barrier failed");
    }
    return next_status;
  }
  std::uint32_t calls{0};
  std::uint32_t fail_on_call{0};
  Status next_status;
};

} // namespace

TEST(ReferenceMockUrmaResources, UrmaResourcePlanCountsForSupportedLaneCounts) {
  for (const auto lanes : {2U, 4U, 6U, 8U}) {
    auto plan = PlanUrmaResources(Capabilities(lanes), Options(lanes));
    UB_REQUIRE_TRUE(plan.ok());
    const auto object_count = lanes / 2U + 2U * (lanes - lanes / 2U);
    UB_REQUIRE_EQ(plan.value().send_jfs_count, object_count);
    UB_REQUIRE_EQ(plan.value().send_jfc_count, object_count);
    UB_REQUIRE_EQ(plan.value().target_jfr_count, object_count);
    UB_REQUIRE_EQ(plan.value().target_jfc_count, 1U);
    UB_REQUIRE_EQ(plan.value().target_jfc_depth, object_count);
  }
}

TEST(ReferenceMockUrmaResources,
     UrmaSplitResourcePlanCreatesReadPlusTwoRpcQueuePairs) {
  auto options = Options(6);
  options.read_progress_threads = 2;
  options.rpc_progress_threads = 4;
  auto capabilities = Capabilities(6);
  auto plan = PlanUrmaResources(capabilities, options);
  UB_REQUIRE_TRUE(plan.ok());
  UB_REQUIRE_EQ(plan.value().send_jfs_count, 10U);
  UB_REQUIRE_EQ(plan.value().send_jfc_count, 10U);
  UB_REQUIRE_EQ(plan.value().target_jfr_count, 10U);
  UB_REQUIRE_EQ(plan.value().target_jfc_depth, 10U);

  FakeResourceBackend backend;
  auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
  UB_REQUIRE_TRUE(bundle.ok());
  auto read = bundle.value().ControlLaneResources(0);
  auto rpc = bundle.value().ControlLaneResources(2);
  UB_REQUIRE_TRUE(read.ok());
  UB_REQUIRE_TRUE(rpc.ok());
  UB_REQUIRE_TRUE(read.value().read_jfs.value != 0);
  UB_REQUIRE_TRUE(read.value().read_target_jfr.value != 0);
  UB_REQUIRE_TRUE(read.value().request_jfs.empty());
  UB_REQUIRE_EQ(read.value().request_target_jfr.value, 0U);
  UB_REQUIRE_TRUE(read.value().response_jfs.empty());
  UB_REQUIRE_EQ(read.value().response_target_jfr.value, 0U);
  UB_REQUIRE_EQ(rpc.value().read_jfs.value, 0U);
  UB_REQUIRE_EQ(rpc.value().read_target_jfr.value, 0U);
  UB_REQUIRE_EQ(rpc.value().request_jfs.size(), 1U);
  UB_REQUIRE_TRUE(rpc.value().request_jfs.front().value != 0);
  UB_REQUIRE_TRUE(rpc.value().request_target_jfr.value != 0);
  UB_REQUIRE_EQ(rpc.value().response_jfs.size(), 1U);
  UB_REQUIRE_TRUE(rpc.value().response_jfs.front().value != 0);
  UB_REQUIRE_TRUE(rpc.value().response_target_jfr.value != 0);
}

TEST(ReferenceMockUrmaResources,
     UrmaResourcePlanCreatesMultipleJfsPerRpcLaneWithSharedJfc) {
  auto options = Options(4);
  options.request_jfs_per_rpc_lane = 2;
  options.response_jfs_per_rpc_lane = 3;
  options.request_jfs_depth = 16;
  options.response_jfs_depth = 16;
  options.request_jfc_depth = 33;
  options.response_jfc_depth = 49;
  auto capabilities = Capabilities(4);
  auto plan = PlanUrmaResources(capabilities, options);
  UB_REQUIRE_TRUE(plan.ok());
  UB_REQUIRE_EQ(plan.value().send_jfs_count, 12U);
  UB_REQUIRE_EQ(plan.value().send_jfc_count, 6U);
  UB_REQUIRE_EQ(plan.value().target_jfr_count, 6U);

  FakeResourceBackend backend;
  auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
  UB_REQUIRE_TRUE(bundle.ok());
  auto rpc = bundle.value().ControlLaneResources(2);
  UB_REQUIRE_TRUE(rpc.ok());
  UB_REQUIRE_EQ(rpc.value().request_jfs.size(), 2U);
  UB_REQUIRE_EQ(rpc.value().response_jfs.size(), 3U);

  std::size_t request_jfs_count = 0;
  std::size_t response_jfs_count = 0;
  for (const auto &request : backend.requests) {
    if (request.kind != UrmaResourceKind::kSendJfs || request.lane != 2) {
      continue;
    }
    if (request.operation_class == OperationClass::kRequestWrite) {
      ++request_jfs_count;
      UB_REQUIRE_EQ(request.dependency, rpc.value().request_jfc);
    } else if (request.operation_class == OperationClass::kResponseWrite) {
      ++response_jfs_count;
      UB_REQUIRE_EQ(request.dependency, rpc.value().response_jfc);
    }
  }
  UB_REQUIRE_EQ(request_jfs_count, 2U);
  UB_REQUIRE_EQ(response_jfs_count, 3U);
}

TEST(ReferenceMockUrmaResources,
     UrmaResourcePlanRejectsCountAndDepthOvercommit) {
  auto capabilities = Capabilities(4);
  capabilities.max_jfs = 5;
  auto count_failure = PlanUrmaResources(capabilities, Options(4));
  UB_REQUIRE_FALSE(count_failure.ok());
  UB_REQUIRE_EQ(count_failure.status().code(), StatusCode::kResourceExhausted);

  capabilities = Capabilities(4);
  auto options = Options(4);
  options.request_jfc_depth = 8;
  auto moderation_failure = PlanUrmaResources(capabilities, options);
  UB_REQUIRE_FALSE(moderation_failure.ok());
  UB_REQUIRE_EQ(moderation_failure.status().code(),
                StatusCode::kInvalidArgument);
  UB_REQUIRE_EQ(
      moderation_failure.status().FindContext("minimum_request_jfc").value(),
      "65");

  options = Options(4);
  options.target_jfr_depth = 0;
  UB_REQUIRE_FALSE(PlanUrmaResources(capabilities, options).ok());
}

TEST(ReferenceMockUrmaResources,
     UrmaResourceBundleDestroysInExactReverseOrder) {
  auto plan = PlanUrmaResources(Capabilities(2), Options(2));
  UB_REQUIRE_TRUE(plan.ok());
  FakeResourceBackend backend;
  {
    auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
    UB_REQUIRE_TRUE(bundle.ok());
    auto lane0 = bundle.value().ClaimLane(0);
    auto lane1 = bundle.value().ClaimLane(1);
    UB_REQUIRE_TRUE(lane0.ok());
    UB_REQUIRE_TRUE(lane1.ok());
    UB_REQUIRE_TRUE(bundle.value().all_lanes_claimed());
    UB_REQUIRE_FALSE(bundle.value().ClaimLane(1).ok());
  }
  UB_REQUIRE_EQ(backend.created.size(), backend.destroyed.size());
  for (std::size_t index = 0; index < backend.created.size(); ++index) {
    UB_REQUIRE_EQ(backend.destroyed[index],
                  backend.created[backend.created.size() - index - 1U]);
  }
}

TEST(ReferenceMockUrmaResources, UrmaResourceBundleCleansPartialConstruction) {
  auto plan = PlanUrmaResources(Capabilities(4), Options(4));
  UB_REQUIRE_TRUE(plan.ok());
  FakeResourceBackend backend;
  backend.fail_at = 7;
  auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
  UB_REQUIRE_FALSE(bundle.ok());
  UB_REQUIRE_EQ(backend.created.size(), 6U);
  UB_REQUIRE_EQ(backend.destroyed.size(), 6U);
  for (std::size_t index = 0; index < backend.created.size(); ++index) {
    UB_REQUIRE_EQ(backend.destroyed[index],
                  backend.created[backend.created.size() - index - 1U]);
  }
}

TEST(ReferenceMockUrmaResources,
     UrmaDuplexPlanUsesJettyCapabilityInsteadOfStandaloneJfsLimit) {
  auto capabilities = Capabilities(2);
  auto options = Options(2);
  options.duplex_jetty = true;
  capabilities.max_jfs = 0;
  capabilities.max_jetty = 3;
  UB_REQUIRE_TRUE(PlanUrmaResources(capabilities, options).ok());
  capabilities.max_jetty = 2;
  UB_REQUIRE_FALSE(PlanUrmaResources(capabilities, options).ok());
}

TEST(ReferenceMockUrmaResources,
     UrmaDuplexQueuesKeepSharedJfrAliveAndExportJettyEndpoints) {
  auto capabilities = Capabilities(2);
  capabilities.max_jetty = 8;
  auto options = Options(2);
  options.duplex_jetty = true;
  options.request_jfs_per_rpc_lane = 2;
  options.request_jfs_depth = 16;
  auto plan = PlanUrmaResources(capabilities, options);
  UB_REQUIRE_TRUE(plan.ok());
  FakeResourceBackend backend;
  {
    auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
    UB_REQUIRE_TRUE(bundle.ok());
    auto read = bundle.value().ControlLaneResources(0);
    auto rpc = bundle.value().ControlLaneResources(1);
    UB_REQUIRE_TRUE(read.ok());
    UB_REQUIRE_TRUE(rpc.ok());
    UB_REQUIRE_EQ(read.value().TargetEndpoint(OperationClass::kRead),
                  read.value().read_jfs);
    UB_REQUIRE_EQ(rpc.value().TargetEndpoint(OperationClass::kRequestWrite),
                  rpc.value().request_jfs.front());
    UB_REQUIRE_EQ(rpc.value().TargetEndpoint(OperationClass::kResponseWrite),
                  rpc.value().response_jfs.front());
    std::size_t jettys = 0;
    for (std::size_t i = 0; i < backend.requests.size(); ++i) {
      const auto &request = backend.requests[i];
      UB_REQUIRE_FALSE(request.kind == UrmaResourceKind::kSendJfs);
      if (request.kind != UrmaResourceKind::kSendJetty) {
        continue;
      }
      ++jettys;
      const auto dependency_exists = [&](UrmaResourceKind kind,
                                         UrmaResourceHandle handle) {
        const auto before =
            backend.created.begin() + static_cast<std::ptrdiff_t>(i);
        return std::find(backend.created.begin(), before,
                         std::pair{kind, handle}) != before;
      };
      UB_REQUIRE_TRUE(
          dependency_exists(UrmaResourceKind::kSendJfc, request.dependency));
      UB_REQUIRE_TRUE(dependency_exists(UrmaResourceKind::kTargetJfr,
                                        request.secondary_dependency));
      if (request.operation_class == OperationClass::kRequestWrite) {
        UB_REQUIRE_EQ(request.secondary_dependency,
                      rpc.value().request_target_jfr);
      }
    }
    UB_REQUIRE_EQ(jettys, 4U);
  }
  UB_REQUIRE_EQ(backend.destroyed.size(), backend.created.size());
  for (std::size_t i = 0; i < backend.created.size(); ++i) {
    UB_REQUIRE_EQ(backend.destroyed[i],
                  backend.created[backend.created.size() - i - 1U]);
  }
}

TEST(ReferenceMockUrmaResources,
     UrmaDuplexConstructionRollsBackEveryFailurePoint) {
  auto capabilities = Capabilities(2);
  capabilities.max_jetty = 8;
  auto options = Options(2);
  options.duplex_jetty = true;
  auto plan = PlanUrmaResources(capabilities, options);
  UB_REQUIRE_TRUE(plan.ok());
  FakeResourceBackend successful;
  {
    auto bundle = UrmaResourceBundle::CreateLocal(successful, plan.value(), {});
    UB_REQUIRE_TRUE(bundle.ok());
  }
  for (std::size_t fail_at = 1; fail_at <= successful.attempts; ++fail_at) {
    FakeResourceBackend backend;
    backend.fail_at = fail_at;
    auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
    UB_REQUIRE_FALSE(bundle.ok());
    UB_REQUIRE_EQ(backend.destroyed.size(), backend.created.size());
    for (std::size_t i = 0; i < backend.created.size(); ++i) {
      UB_REQUIRE_EQ(backend.destroyed[i],
                    backend.created[backend.created.size() - i - 1U]);
    }
  }
}

TEST(ReferenceMockUrmaResources,
     UrmaResourceBundleAddsSegmentContextToRegistrationFailure) {
  auto plan = PlanUrmaResources(Capabilities(2), Options(2));
  UB_REQUIRE_TRUE(plan.ok());
  FakeResourceBackend backend;
  backend.fail_at = 2;
  std::array<std::byte, 16> storage{};
  const std::array<UrmaSegmentRegistration, 1> segments{{
      {storage.data(), storage.size(), "vectors", 7},
  }};

  auto bundle =
      UrmaResourceBundle::CreateLocal(backend, plan.value(), segments);
  UB_REQUIRE_FALSE(bundle.ok());
  UB_REQUIRE_EQ(bundle.status().FindContext("segment_index").value(), "0");
  UB_REQUIRE_EQ(bundle.status().FindContext("segment_name").value(), "vectors");
  UB_REQUIRE_EQ(bundle.status().FindContext("logical_bytes").value(), "7");
  UB_REQUIRE_EQ(bundle.status().FindContext("registered_bytes").value(), "16");
}

TEST(ReferenceMockUrmaResources, UrmaResourceBundleEnforcesTwoPhaseTeardown) {
  auto plan = PlanUrmaResources(Capabilities(2), Options(2));
  FakeResourceBackend backend;
  auto bundle = UrmaResourceBundle::CreateLocal(backend, plan.value(), {});
  UB_REQUIRE_TRUE(bundle.ok());
  std::array<std::byte, 2> provider_descriptor{std::byte{1}, std::byte{2}};
  UrmaResourceRequest import;
  import.kind = UrmaResourceKind::kImportedSegment;
  import.descriptor = provider_descriptor;
  auto imported = bundle.value().CreatePeerResource(import);
  UB_REQUIRE_TRUE(imported.ok());
  UB_REQUIRE_TRUE(bundle.value().ControlLaneResources(0).ok());
  UB_REQUIRE_TRUE(bundle.value().ClaimLane(0).ok());
  UB_REQUIRE_FALSE(bundle.value().ControlLaneResources(0).ok());
  UB_REQUIRE_FALSE(bundle.value().CreatePeerResource(import).ok());
  UB_REQUIRE_FALSE(bundle.value().MarkDrained().ok());
  UB_REQUIRE_TRUE(bundle.value().StopAdmission().ok());
  UB_REQUIRE_TRUE(bundle.value().MarkDrained().ok());
  UB_REQUIRE_TRUE(bundle.value().MarkFirstBarrier().ok());
  UB_REQUIRE_TRUE(bundle.value().ReleasePeerResources().ok());
  UB_REQUIRE_TRUE(bundle.value().MarkSecondBarrier().ok());
  UB_REQUIRE_TRUE(bundle.value().ReleaseLocalResources().ok());
  UB_REQUIRE_EQ(bundle.value().teardown_phase(),
                UrmaTeardownPhase::kLocalReleased);
  UB_REQUIRE_EQ(backend.created.size(), backend.destroyed.size());
}

TEST(ReferenceMockUrmaResources,
     OpportunisticPostBatchesExistingWorkersOnOneJfs) {
  std::array<OpportunisticPostDescriptor, 5> ready{};
  for (std::size_t index = 0; index < ready.size(); ++index) {
    ready[index].jfs_key = index < 4 ? 17 : 18;
    ready[index].worker_id = static_cast<std::uint32_t>(index % 2U);
    ready[index].ticket.value = index + 1U;
  }
  FakePoster poster;
  auto outcome = PostOpportunisticBatch(ready, 8, poster);
  UB_REQUIRE_TRUE(outcome.status.ok());
  UB_REQUIRE_EQ(outcome.attempted, 4U);
  UB_REQUIRE_EQ(outcome.posted, 4U);
  UB_REQUIRE_EQ(poster.seen.size(), 4U);
  UB_REQUIRE_EQ(poster.seen[0].worker_id, 0U);
  UB_REQUIRE_EQ(poster.seen[1].worker_id, 1U);

  poster.seen.clear();
  outcome = PostOpportunisticBatch(std::span(ready).first(1), 8, poster);
  UB_REQUIRE_TRUE(outcome.status.ok());
  UB_REQUIRE_EQ(poster.seen.size(), 1U);
}

TEST(ReferenceMockUrmaResources, OpportunisticPostAccountsBadWrPrefix) {
  std::array<OpportunisticPostDescriptor, 4> ready{};
  for (std::size_t index = 0; index < ready.size(); ++index) {
    ready[index].jfs_key = 7;
    ready[index].ticket.value = index + 1U;
  }
  FakePoster poster;
  poster.result.status = Status(StatusCode::kTransportError, "injected bad_wr");
  poster.result.bad_wr_index = 2;
  auto outcome = PostOpportunisticBatch(ready, 4, poster);
  UB_REQUIRE_FALSE(outcome.status.ok());
  UB_REQUIRE_EQ(outcome.attempted, 4U);
  UB_REQUIRE_EQ(outcome.posted, 2U);
  UB_REQUIRE_EQ(outcome.unposted, 2U);
}

TEST(ReferenceMockUrmaResources,
     DmaNotifyAcquiresProviderWritesOnlyAfterObservingNotification) {
  alignas(std::uint64_t) std::uint64_t notify = 0;
  FakeVisibility visibility;
  auto empty = DmaLoadNotifyAcquire(&notify, visibility);
  UB_REQUIRE_TRUE(empty.ok());
  UB_REQUIRE_EQ(empty.value(), 0U);
  UB_REQUIRE_EQ(visibility.calls, 0U);

  notify = 9;
  auto published = DmaLoadNotifyAcquire(&notify, visibility);
  UB_REQUIRE_TRUE(published.ok());
  UB_REQUIRE_EQ(published.value(), 9U);
  UB_REQUIRE_EQ(visibility.calls, 1U);

  visibility.fail_on_call = visibility.calls + 1U;
  notify = 10;
  UB_REQUIRE_FALSE(DmaLoadNotifyAcquire(&notify, visibility).ok());
  UB_REQUIRE_EQ(visibility.calls, 2U);
}

TEST(ReferenceMockUrmaResources,
     UrmaProviderErrorMappingPreservesNativeStatus) {
  auto status = MapUrmaProviderStatus(-17, "urma_poll_jfc");
  UB_REQUIRE_EQ(status.code(), StatusCode::kTransportError);
  UB_REQUIRE_TRUE(status.FindContext("provider_status").has_value());
  UB_REQUIRE_TRUE(MapUrmaProviderStatus(0, "ok").ok());
}

} // namespace hnsw_urma
