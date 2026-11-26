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

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {

enum class UrmaResourceKind : std::uint8_t {
  kContext,
  kRegisteredSegment,
  kSendJfc,
  kSendJfs,
  kTargetJfc,
  kTargetJfr,
  kImportedSegment,
  kImportedEndpoint,
  kAdvice,
  kSendJetty,
};

struct UrmaCapabilities {
  std::uint32_t max_jfs{0};
  std::uint32_t max_jfr{0};
  std::uint32_t max_jfc{0};
  std::uint32_t max_jfs_depth{0};
  std::uint32_t max_jfr_depth{0};
  std::uint32_t max_jfc_depth{0};
  std::uint32_t max_inline_bytes{0};
  std::uint32_t max_read_bytes{0};
  std::uint32_t max_write_bytes{0};
  std::uint32_t max_jetty{0};
};

struct UrmaResourceOptions {
  std::uint32_t communication_lanes{2};
  std::uint32_t read_progress_threads{1};
  std::uint32_t rpc_progress_threads{1};
  std::uint32_t read_jfs_depth{64};
  std::uint32_t request_jfs_depth{64};
  std::uint32_t response_jfs_depth{64};
  std::uint32_t request_jfs_per_rpc_lane{1};
  std::uint32_t response_jfs_per_rpc_lane{1};
  std::uint32_t read_jfc_depth{64};
  // The production transport requests one completion for every
  // request/response WRITE_NOTIFY. Keep room for every in-flight WR plus one
  // provider safety slot.
  std::uint32_t request_jfc_depth{65};
  std::uint32_t response_jfc_depth{65};
  std::uint32_t target_jfr_depth{1};
  std::uint32_t target_error_completions_per_jfr{1};
  std::uint32_t wr_list_max{8};
  // Duplex queues share the role's target JFR and export a Jetty endpoint.
  bool duplex_jetty{false};
};

struct UrmaResourcePlan {
  UrmaResourceOptions options;
  std::uint32_t send_jfs_count{0};
  std::uint32_t send_jfc_count{0};
  std::uint32_t target_jfr_count{0};
  std::uint32_t target_jfc_count{1};
  std::uint32_t target_jfc_depth{0};
};

[[nodiscard]] Result<UrmaResourcePlan>
PlanUrmaResources(const UrmaCapabilities &capabilities,
                  const UrmaResourceOptions &options);

struct UrmaResourceHandle {
  std::uint64_t value{0};
  friend bool operator==(const UrmaResourceHandle &,
                         const UrmaResourceHandle &) = default;
};

struct UrmaResourceRequest {
  UrmaResourceKind kind{UrmaResourceKind::kContext};
  std::uint32_t lane{0};
  OperationClass operation_class{OperationClass::kRead};
  std::uint32_t depth{0};
  UrmaResourceHandle dependency;
  UrmaResourceHandle secondary_dependency;
  void *address{nullptr};
  std::uint64_t bytes{0};
  std::span<const std::byte> descriptor;
  std::uint64_t token{0};
};

class UrmaResourceBackend {
public:
  virtual ~UrmaResourceBackend() = default;
  virtual Result<UrmaResourceHandle>
  Create(const UrmaResourceRequest &request) = 0;
  virtual void Destroy(UrmaResourceKind kind,
                       UrmaResourceHandle handle) noexcept = 0;
};

struct UrmaSegmentRegistration {
  void *address{nullptr};
  std::uint64_t bytes{0};
  std::string_view name;
  std::uint64_t logical_bytes{0};
};

struct UrmaLaneResources {
  std::uint32_t lane{0};
  UrmaResourceHandle read_jfc;
  UrmaResourceHandle read_jfs;
  UrmaResourceHandle read_target_jfr;
  UrmaResourceHandle request_jfc;
  std::vector<UrmaResourceHandle> request_jfs;
  UrmaResourceHandle request_target_jfr;
  UrmaResourceHandle response_jfc;
  std::vector<UrmaResourceHandle> response_jfs;
  UrmaResourceHandle response_target_jfr;
  // The *_jfs handles name kSendJetty resources in duplex mode. JFR handles
  // still name the shared receive resources, with independent lifetimes.
  bool duplex_jetty{false};

  [[nodiscard]] UrmaResourceHandle
  TargetEndpoint(OperationClass role) const noexcept {
    if (role == OperationClass::kRead) {
      return duplex_jetty ? read_jfs : read_target_jfr;
    }
    if (role == OperationClass::kRequestWrite) {
      return duplex_jetty ? (request_jfs.empty() ? UrmaResourceHandle{}
                                                 : request_jfs.front())
                          : request_target_jfr;
    }
    return duplex_jetty ? (response_jfs.empty() ? UrmaResourceHandle{}
                                                : response_jfs.front())
                        : response_target_jfr;
  }
};

enum class UrmaTeardownPhase : std::uint8_t {
  kRunning,
  kAdmissionStopped,
  kDrained,
  kFirstBarrier,
  kPeersReleased,
  kSecondBarrier,
  kLocalReleased,
};

// The control thread owns a bundle until ClaimLane() permanently transfers hot-
// path authority to a progress lane. The bundle still owns object lifetime so
// context/JFC dependencies cannot disappear while a lane is running.
class UrmaResourceBundle {
public:
  UrmaResourceBundle() = default;
  UrmaResourceBundle(const UrmaResourceBundle &) = delete;
  UrmaResourceBundle &operator=(const UrmaResourceBundle &) = delete;
  UrmaResourceBundle(UrmaResourceBundle &&other) noexcept;
  UrmaResourceBundle &operator=(UrmaResourceBundle &&other) noexcept;
  ~UrmaResourceBundle();

  [[nodiscard]] static Result<UrmaResourceBundle>
  CreateLocal(UrmaResourceBackend &backend, const UrmaResourcePlan &plan,
              std::span<const UrmaSegmentRegistration> segments);

  [[nodiscard]] Result<UrmaLaneResources> ClaimLane(std::uint32_t lane);
  [[nodiscard]] Result<UrmaLaneResources>
  ControlLaneResources(std::uint32_t lane) const;
  [[nodiscard]] std::span<const UrmaResourceHandle>
  registered_segments() const noexcept {
    return registered_segments_;
  }
  [[nodiscard]] Result<UrmaResourceHandle>
  CreatePeerResource(const UrmaResourceRequest &request);
  [[nodiscard]] bool all_lanes_claimed() const noexcept;
  [[nodiscard]] UrmaTeardownPhase teardown_phase() const noexcept {
    return phase_;
  }

  Status StopAdmission();
  Status MarkDrained();
  Status MarkFirstBarrier();
  Status ReleasePeerResources();
  Status MarkSecondBarrier();
  Status ReleaseLocalResources();

private:
  struct OwnedResource {
    UrmaResourceKind kind{UrmaResourceKind::kContext};
    UrmaResourceHandle handle;
  };

  void DestroyReverse(std::vector<OwnedResource> &resources) noexcept;
  void Reset() noexcept;

  UrmaResourceBackend *backend_{nullptr};
  UrmaResourcePlan plan_;
  UrmaResourceHandle context_;
  std::vector<OwnedResource> local_resources_;
  std::vector<OwnedResource> peer_resources_;
  std::vector<UrmaResourceHandle> registered_segments_;
  std::vector<UrmaLaneResources> lane_resources_;
  std::vector<bool> lane_claimed_;
  UrmaTeardownPhase phase_{UrmaTeardownPhase::kRunning};
};

struct OpportunisticPostDescriptor {
  std::uint64_t jfs_key{0};
  std::uint32_t worker_id{0};
  OperationTicket ticket;
};

struct ProviderPostResult {
  Status status;
  // The first WR not accepted by the provider. It must be set on a partial
  // failure; SIZE_MAX means that no safe prefix can be inferred.
  std::size_t bad_wr_index{static_cast<std::size_t>(-1)};
};

class UrmaWorkRequestPoster {
public:
  virtual ~UrmaWorkRequestPoster() = default;
  virtual ProviderPostResult
  Post(std::uint64_t jfs_key,
       std::span<const OpportunisticPostDescriptor> descriptors) = 0;
};

struct OpportunisticPostOutcome {
  std::size_t attempted{0};
  std::size_t posted{0};
  std::size_t unposted{0};
  Status status;
};

// Forms a bounded batch only from descriptors already present at the head and
// only while they target one JFS. It never waits for a fuller list.
[[nodiscard]] OpportunisticPostOutcome
PostOpportunisticBatch(std::span<const OpportunisticPostDescriptor> ready,
                       std::size_t wr_list_max, UrmaWorkRequestPoster &poster);

class DmaVisibilityDomain {
public:
  virtual ~DmaVisibilityDomain() = default;
  virtual Status AcquireDeviceWrites() = 0;
};

// Force an actual device-visible notify load. WRITE_NOTIFY publishes its
// payload before the notify value; after observing a non-zero notification,
// acquire device writes before accessing the associated payload. Empty polls
// deliberately avoid the device-read barrier.
[[nodiscard]] Result<std::uint64_t>
DmaLoadNotifyAcquire(const std::uint64_t *notify,
                     DmaVisibilityDomain &visibility);

[[nodiscard]] Status MapUrmaProviderStatus(int provider_status,
                                           std::string operation);

} // namespace hnsw_urma
