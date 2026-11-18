#include "hnsw_urma/transport/urma_resources.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <tuple>
#include <utility>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
namespace {

constexpr std::size_t kNoBadWorkRequest = static_cast<std::size_t>(-1);

Result<std::uint32_t> PerOperationCompletionDepth(
    std::uint32_t send_depth) {
  auto depth = CheckedAdd(send_depth, 1U);
  if (!depth.ok()) {
    return Status(StatusCode::kOutOfRange,
                  "completion depth overflows uint32");
  }
  return depth.value();
}

Status RequireLimit(std::uint32_t requested, std::uint32_t maximum, const char* name) {
  if (requested > maximum) {
    return Status(StatusCode::kResourceExhausted,
                  std::string(name) + " exceeds the provider capability")
        .AddContext("requested", std::to_string(requested))
        .AddContext("maximum", std::to_string(maximum));
  }
  return Status::Ok();
}

UrmaResourceRequest QueueRequest(UrmaResourceKind kind, std::uint32_t lane,
                                 OperationClass operation_class, std::uint32_t depth,
                                 UrmaResourceHandle dependency = {}) {
  UrmaResourceRequest request;
  request.kind = kind;
  request.lane = lane;
  request.operation_class = operation_class;
  request.depth = depth;
  request.dependency = dependency;
  return request;
}

}  // namespace

Result<UrmaResourcePlan> PlanUrmaResources(const UrmaCapabilities& capabilities,
                                           const UrmaResourceOptions& options) {
  if (options.communication_lanes == 0 || options.wr_list_max == 0 ||
      options.read_jfs_depth == 0 || options.request_jfs_depth == 0 ||
      options.response_jfs_depth == 0 ||
      options.request_jfs_per_rpc_lane == 0 ||
      options.response_jfs_per_rpc_lane == 0 ||
      options.read_jfc_depth == 0 ||
      options.request_jfc_depth == 0 || options.response_jfc_depth == 0 ||
      options.target_jfr_depth == 0 || options.target_error_completions_per_jfr == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "URMA lane counts, queue depths, and batching limits must be non-zero");
  }
  if (options.read_progress_threads == 0 ||
      options.rpc_progress_threads == 0 ||
      static_cast<std::uint64_t>(options.read_progress_threads) +
              options.rpc_progress_threads !=
          options.communication_lanes) {
    return Status(StatusCode::kInvalidArgument,
                  "URMA thread counts do not match communication lanes");
  }
  auto rpc_queue_classes =
      CheckedMultiply<std::uint32_t>(options.rpc_progress_threads, 2U);
  Result<std::uint32_t> queue_class_count =
      rpc_queue_classes.ok()
          ? CheckedAdd(options.read_progress_threads,
                       rpc_queue_classes.value())
          : Result<std::uint32_t>(rpc_queue_classes.status());
  if (!queue_class_count.ok()) {
    return queue_class_count.status();
  }
  auto jfs_per_rpc_lane = CheckedAdd<std::uint32_t>(
      options.request_jfs_per_rpc_lane,
      options.response_jfs_per_rpc_lane);
  auto rpc_jfs_count = jfs_per_rpc_lane.ok()
                           ? CheckedMultiply<std::uint32_t>(
                                 options.rpc_progress_threads,
                                 jfs_per_rpc_lane.value())
                           : Result<std::uint32_t>(jfs_per_rpc_lane.status());
  auto send_jfs_count = rpc_jfs_count.ok()
                            ? CheckedAdd(options.read_progress_threads,
                                         rpc_jfs_count.value())
                            : Result<std::uint32_t>(rpc_jfs_count.status());
  if (!send_jfs_count.ok()) {
    return send_jfs_count.status();
  }
  auto total_jfc = CheckedAdd<std::uint32_t>(queue_class_count.value(), 1U);
  if (!total_jfc.ok()) {
    return total_jfc.status();
  }
  auto target_jfc_depth = CheckedMultiply<std::uint32_t>(
      queue_class_count.value(), options.target_error_completions_per_jfr);
  if (!target_jfc_depth.ok()) {
    return target_jfc_depth.status();
  }

  for (const auto& [requested, maximum, name] : {
           std::tuple{send_jfs_count.value(),
                      options.duplex_jetty ? capabilities.max_jetty : capabilities.max_jfs,
                      options.duplex_jetty ? "send Jetty count" : "send JFS count"},
           std::tuple{queue_class_count.value(), capabilities.max_jfr,
                      "target JFR count"},
           std::tuple{total_jfc.value(), capabilities.max_jfc, "JFC count"},
           std::tuple{options.read_jfs_depth, capabilities.max_jfs_depth, "read JFS depth"},
           std::tuple{options.request_jfs_depth, capabilities.max_jfs_depth,
                      "request JFS depth"},
           std::tuple{options.response_jfs_depth, capabilities.max_jfs_depth,
                      "response JFS depth"},
           std::tuple{options.read_jfc_depth, capabilities.max_jfc_depth, "read JFC depth"},
           std::tuple{options.request_jfc_depth, capabilities.max_jfc_depth,
                      "request JFC depth"},
           std::tuple{options.response_jfc_depth, capabilities.max_jfc_depth,
                      "response JFC depth"},
           std::tuple{options.target_jfr_depth, capabilities.max_jfr_depth,
                      "target JFR depth"},
           std::tuple{target_jfc_depth.value(), capabilities.max_jfc_depth,
                      "target JFC depth"},
       }) {
    auto status = RequireLimit(requested, maximum, name);
    if (!status.ok()) {
      return status;
    }
  }

  // UmdkLanePostEngine sets complete_enable=1 on every WRITE_NOTIFY, so the
  // JFC must cover every in-flight JFS entry plus the safety slot below.
  auto total_request_depth = CheckedMultiply<std::uint32_t>(
      options.request_jfs_depth, options.request_jfs_per_rpc_lane);
  auto total_response_depth = CheckedMultiply<std::uint32_t>(
      options.response_jfs_depth, options.response_jfs_per_rpc_lane);
  auto minimum_request_jfc = total_request_depth.ok()
                                 ? PerOperationCompletionDepth(
                                       total_request_depth.value())
                                 : Result<std::uint32_t>(
                                       total_request_depth.status());
  auto minimum_response_jfc = total_response_depth.ok()
                                  ? PerOperationCompletionDepth(
                                        total_response_depth.value())
                                  : Result<std::uint32_t>(
                                        total_response_depth.status());
  if (!minimum_request_jfc.ok()) {
    return minimum_request_jfc.status();
  }
  if (!minimum_response_jfc.ok()) {
    return minimum_response_jfc.status();
  }
  if (options.read_jfc_depth < options.read_jfs_depth ||
      options.request_jfc_depth < minimum_request_jfc.value() ||
      options.response_jfc_depth < minimum_response_jfc.value()) {
    return Status(StatusCode::kInvalidArgument,
                  "JFC depth is too small for queue depth and completion moderation")
        .AddContext("minimum_request_jfc", std::to_string(minimum_request_jfc.value()))
        .AddContext("minimum_response_jfc", std::to_string(minimum_response_jfc.value()));
  }

  UrmaResourcePlan plan;
  plan.options = options;
  plan.send_jfs_count = send_jfs_count.value();
  plan.send_jfc_count = queue_class_count.value();
  plan.target_jfr_count = queue_class_count.value();
  plan.target_jfc_depth = target_jfc_depth.value();
  return plan;
}

UrmaResourceBundle::UrmaResourceBundle(UrmaResourceBundle&& other) noexcept {
  *this = std::move(other);
}

UrmaResourceBundle& UrmaResourceBundle::operator=(UrmaResourceBundle&& other) noexcept {
  if (this != &other) {
    Reset();
    backend_ = std::exchange(other.backend_, nullptr);
    plan_ = other.plan_;
    context_ = other.context_;
    local_resources_ = std::move(other.local_resources_);
    peer_resources_ = std::move(other.peer_resources_);
    registered_segments_ = std::move(other.registered_segments_);
    lane_resources_ = std::move(other.lane_resources_);
    lane_claimed_ = std::move(other.lane_claimed_);
    phase_ = other.phase_;
  }
  return *this;
}

UrmaResourceBundle::~UrmaResourceBundle() { Reset(); }

void UrmaResourceBundle::DestroyReverse(std::vector<OwnedResource>& resources) noexcept {
  if (backend_ != nullptr) {
    for (auto iterator = resources.rbegin(); iterator != resources.rend(); ++iterator) {
      backend_->Destroy(iterator->kind, iterator->handle);
    }
  }
  resources.clear();
}

void UrmaResourceBundle::Reset() noexcept {
  DestroyReverse(peer_resources_);
  DestroyReverse(local_resources_);
  backend_ = nullptr;
  context_ = {};
  registered_segments_.clear();
  lane_resources_.clear();
  lane_claimed_.clear();
}

Result<UrmaResourceBundle> UrmaResourceBundle::CreateLocal(
    UrmaResourceBackend& backend, const UrmaResourcePlan& plan,
    std::span<const UrmaSegmentRegistration> segments) {
  UrmaResourceBundle bundle;
  bundle.backend_ = &backend;
  bundle.plan_ = plan;
  try {
    bundle.local_resources_.reserve(2U + segments.size() + plan.send_jfc_count +
                                    plan.send_jfs_count + plan.target_jfr_count);
    bundle.registered_segments_.reserve(segments.size());
    bundle.lane_resources_.resize(plan.options.communication_lanes);
    bundle.lane_claimed_.assign(plan.options.communication_lanes, false);
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted, "unable to allocate URMA resource ledger");
  }

  const auto create = [&](const UrmaResourceRequest& request)
      -> Result<UrmaResourceHandle> {
    auto result = backend.Create(request);
    if (result.ok()) {
      bundle.local_resources_.push_back({request.kind, result.value()});
    }
    return result;
  };

  auto context = create(QueueRequest(UrmaResourceKind::kContext, 0,
                                     OperationClass::kRead, 0));
  if (!context.ok()) {
    return context.status();
  }
  bundle.context_ = context.value();
  for (std::size_t segment_index = 0; segment_index < segments.size();
       ++segment_index) {
    const auto& segment = segments[segment_index];
    UrmaResourceRequest request;
    request.kind = UrmaResourceKind::kRegisteredSegment;
    request.dependency = context.value();
    request.address = segment.address;
    request.bytes = segment.bytes;
    auto registered = create(request);
    if (!registered.ok()) {
      auto status = registered.status();
      status.AddContext("segment_index", std::to_string(segment_index))
          .AddContext("segment_name",
                      segment.name.empty() ? "unnamed" : std::string(segment.name))
          .AddContext("address",
                      std::to_string(
                          reinterpret_cast<std::uintptr_t>(segment.address)))
          .AddContext("logical_bytes",
                      std::to_string(segment.logical_bytes == 0
                                         ? segment.bytes
                                         : segment.logical_bytes))
          .AddContext("registered_bytes", std::to_string(segment.bytes));
      return status;
    }
    bundle.registered_segments_.push_back(registered.value());
  }
  auto target_jfc = create(QueueRequest(UrmaResourceKind::kTargetJfc, 0,
                                        OperationClass::kRead, plan.target_jfc_depth,
                                        context.value()));
  if (!target_jfc.ok()) {
    return target_jfc.status();
  }

  const OperationClass classes[] = {
      OperationClass::kRead, OperationClass::kRequestWrite,
      OperationClass::kResponseWrite};
  for (std::uint32_t lane = 0; lane < plan.options.communication_lanes; ++lane) {
    bundle.lane_resources_[lane].lane = lane;
    bundle.lane_resources_[lane].duplex_jetty = plan.options.duplex_jetty;
    for (const auto operation_class : classes) {
      const bool read_lane = lane < plan.options.read_progress_threads;
      if (read_lane != (operation_class == OperationClass::kRead)) {
        continue;
      }
      std::uint32_t jfs_depth = plan.options.read_jfs_depth;
      std::uint32_t jfc_depth = plan.options.read_jfc_depth;
      if (operation_class == OperationClass::kRequestWrite) {
        jfs_depth = plan.options.request_jfs_depth;
        jfc_depth = plan.options.request_jfc_depth;
      } else if (operation_class == OperationClass::kResponseWrite) {
        jfs_depth = plan.options.response_jfs_depth;
        jfc_depth = plan.options.response_jfc_depth;
      }
      auto jfc = create(QueueRequest(UrmaResourceKind::kSendJfc, lane, operation_class,
                                     jfc_depth, context.value()));
      if (!jfc.ok()) {
        return jfc.status();
      }
      // UB duplex Jettys require a shared JFR. Create it before the Jettys so
      // reverse teardown releases every Jetty before its receive resource.
      const auto create_target = [&]() {
        return create(QueueRequest(UrmaResourceKind::kTargetJfr, lane, operation_class,
                                   plan.options.target_jfr_depth, target_jfc.value()));
      };
      UrmaResourceHandle target;
      if (plan.options.duplex_jetty) {
        auto jfr = create_target();
        if (!jfr.ok()) {
          return jfr.status();
        }
        target = jfr.value();
      }
      std::uint32_t jfs_count = 1;
      if (operation_class == OperationClass::kRequestWrite) {
        jfs_count = plan.options.request_jfs_per_rpc_lane;
      } else if (operation_class == OperationClass::kResponseWrite) {
        jfs_count = plan.options.response_jfs_per_rpc_lane;
      }
      std::vector<UrmaResourceHandle> jfs_handles;
      try {
        jfs_handles.reserve(jfs_count);
      } catch (const std::bad_alloc&) {
        return Status(StatusCode::kResourceExhausted,
                      "unable to allocate JFS handle list");
      }
      for (std::uint32_t jfs_index = 0; jfs_index < jfs_count;
           ++jfs_index) {
        auto request = QueueRequest(plan.options.duplex_jetty
                                        ? UrmaResourceKind::kSendJetty
                                        : UrmaResourceKind::kSendJfs,
                                    lane, operation_class, jfs_depth, jfc.value());
        request.secondary_dependency = target;
        auto jfs = create(request);
        if (!jfs.ok()) {
          return jfs.status();
        }
        jfs_handles.push_back(jfs.value());
      }
      if (!plan.options.duplex_jetty) {
        auto jfr = create_target();
        if (!jfr.ok()) {
          return jfr.status();
        }
        target = jfr.value();
      }
      auto& resources = bundle.lane_resources_[lane];
      if (operation_class == OperationClass::kRead) {
        resources.read_jfc = jfc.value();
        resources.read_jfs = jfs_handles.front();
        resources.read_target_jfr = target;
      } else if (operation_class == OperationClass::kRequestWrite) {
        resources.request_jfc = jfc.value();
        resources.request_jfs = std::move(jfs_handles);
        resources.request_target_jfr = target;
      } else {
        resources.response_jfc = jfc.value();
        resources.response_jfs = std::move(jfs_handles);
        resources.response_target_jfr = target;
      }
    }
  }
  return bundle;
}

Result<UrmaLaneResources> UrmaResourceBundle::ClaimLane(std::uint32_t lane) {
  if (phase_ != UrmaTeardownPhase::kRunning) {
    return Status(StatusCode::kInvalidArgument, "cannot claim a lane during teardown");
  }
  if (lane >= lane_resources_.size()) {
    return Status(StatusCode::kOutOfRange, "progress lane is out of range");
  }
  if (lane_claimed_[lane]) {
    return Status(StatusCode::kInvalidArgument, "progress lane was already claimed");
  }
  lane_claimed_[lane] = true;
  return lane_resources_[lane];
}

Result<UrmaLaneResources> UrmaResourceBundle::ControlLaneResources(std::uint32_t lane) const {
  if (phase_ != UrmaTeardownPhase::kRunning) {
    return Status(StatusCode::kInvalidArgument,
                  "control resources are unavailable during teardown");
  }
  if (lane >= lane_resources_.size()) {
    return Status(StatusCode::kOutOfRange, "progress lane is out of range");
  }
  if (lane_claimed_[lane]) {
    return Status(StatusCode::kInvalidArgument,
                  "control thread no longer owns this progress lane");
  }
  return lane_resources_[lane];
}

Result<UrmaResourceHandle> UrmaResourceBundle::CreatePeerResource(
    const UrmaResourceRequest& input) {
  if (phase_ != UrmaTeardownPhase::kRunning || backend_ == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "peer resources can only be imported while the bundle is running");
  }
  if (std::any_of(lane_claimed_.begin(), lane_claimed_.end(),
                  [](bool claimed) { return claimed; })) {
    return Status(StatusCode::kInvalidArgument,
                  "peer imports must finish before progress lane ownership transfer");
  }
  if (input.kind != UrmaResourceKind::kImportedSegment &&
      input.kind != UrmaResourceKind::kImportedEndpoint &&
      input.kind != UrmaResourceKind::kAdvice) {
    return Status(StatusCode::kInvalidArgument, "resource is not a peer import or advice");
  }
  UrmaResourceRequest request = input;
  if (request.dependency.value == 0) {
    request.dependency = context_;
  }
  auto created = backend_->Create(request);
  if (!created.ok()) {
    return created.status();
  }
  try {
    peer_resources_.push_back({request.kind, created.value()});
  } catch (const std::bad_alloc&) {
    backend_->Destroy(request.kind, created.value());
    return Status(StatusCode::kResourceExhausted,
                  "unable to record imported peer resource");
  }
  return created.value();
}

bool UrmaResourceBundle::all_lanes_claimed() const noexcept {
  return std::all_of(lane_claimed_.begin(), lane_claimed_.end(), [](bool value) { return value; });
}

Status UrmaResourceBundle::StopAdmission() {
  if (phase_ != UrmaTeardownPhase::kRunning) {
    return Status(StatusCode::kInvalidArgument, "admission can only stop from running state");
  }
  phase_ = UrmaTeardownPhase::kAdmissionStopped;
  return Status::Ok();
}

Status UrmaResourceBundle::MarkDrained() {
  if (phase_ != UrmaTeardownPhase::kAdmissionStopped) {
    return Status(StatusCode::kInvalidArgument, "in-flight operations must drain after admission stops");
  }
  phase_ = UrmaTeardownPhase::kDrained;
  return Status::Ok();
}

Status UrmaResourceBundle::MarkFirstBarrier() {
  if (phase_ != UrmaTeardownPhase::kDrained) {
    return Status(StatusCode::kInvalidArgument, "first barrier requires drained lanes");
  }
  phase_ = UrmaTeardownPhase::kFirstBarrier;
  return Status::Ok();
}

Status UrmaResourceBundle::ReleasePeerResources() {
  if (phase_ != UrmaTeardownPhase::kFirstBarrier) {
    return Status(StatusCode::kInvalidArgument, "peer imports require the first global barrier");
  }
  DestroyReverse(peer_resources_);
  phase_ = UrmaTeardownPhase::kPeersReleased;
  return Status::Ok();
}

Status UrmaResourceBundle::MarkSecondBarrier() {
  if (phase_ != UrmaTeardownPhase::kPeersReleased) {
    return Status(StatusCode::kInvalidArgument, "second barrier requires peer imports to be released");
  }
  phase_ = UrmaTeardownPhase::kSecondBarrier;
  return Status::Ok();
}

Status UrmaResourceBundle::ReleaseLocalResources() {
  if (phase_ != UrmaTeardownPhase::kSecondBarrier) {
    return Status(StatusCode::kInvalidArgument, "local resources require the second global barrier");
  }
  DestroyReverse(local_resources_);
  phase_ = UrmaTeardownPhase::kLocalReleased;
  return Status::Ok();
}

OpportunisticPostOutcome PostOpportunisticBatch(
    std::span<const OpportunisticPostDescriptor> ready, std::size_t wr_list_max,
    UrmaWorkRequestPoster& poster) {
  OpportunisticPostOutcome outcome;
  if (ready.empty()) {
    return outcome;
  }
  if (wr_list_max == 0) {
    outcome.status = Status(StatusCode::kInvalidArgument, "WR-list maximum must be non-zero");
    return outcome;
  }
  const auto jfs_key = ready.front().jfs_key;
  while (outcome.attempted < ready.size() && outcome.attempted < wr_list_max &&
         ready[outcome.attempted].jfs_key == jfs_key) {
    ++outcome.attempted;
  }
  auto result = poster.Post(jfs_key, ready.first(outcome.attempted));
  outcome.status = std::move(result.status);
  if (outcome.status.ok()) {
    outcome.posted = outcome.attempted;
  } else if (result.bad_wr_index != kNoBadWorkRequest &&
             result.bad_wr_index <= outcome.attempted) {
    outcome.posted = result.bad_wr_index;
  }
  outcome.unposted = outcome.attempted - outcome.posted;
  return outcome;
}

Result<std::uint64_t> DmaLoadNotifyAcquire(const std::uint64_t* notify,
                                           DmaVisibilityDomain& visibility) {
  if (notify == nullptr || reinterpret_cast<std::uintptr_t>(notify) % alignof(std::uint64_t) != 0) {
    return Status(StatusCode::kInvalidArgument, "notify address must be non-null and aligned");
  }
  const auto* device_notify =
      reinterpret_cast<const volatile std::uint64_t*>(notify);
  const auto value = *device_notify;
  if (value == 0) {
    return value;
  }
  auto status = visibility.AcquireDeviceWrites();
  if (!status.ok()) {
    return status;
  }
  return value;
}

Status MapUrmaProviderStatus(int provider_status, std::string operation) {
  if (provider_status == 0) {
    return Status::Ok();
  }
  return Status(StatusCode::kTransportError, std::move(operation) + " failed")
      .AddContext("provider_status", std::to_string(provider_status));
}

}  // namespace hnsw_urma
