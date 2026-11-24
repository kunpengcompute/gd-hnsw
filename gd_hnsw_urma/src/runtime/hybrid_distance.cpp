#include "hnsw_urma/runtime/hybrid_distance.hpp"

#include <atomic>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace hnsw_urma {

class HybridDistanceOracle::Impl {
public:
  static constexpr ticket_t kPullTicket = ticket_t{1} << 62U;
  static constexpr ticket_t kPushdownTicket = ticket_t{1} << 63U;
  static constexpr ticket_t kBackendMask = kPullTicket | kPushdownTicket;

  [[nodiscard]] DistanceOracle *Select(shard_id_t owner_shard,
                                       std::size_t id_count, bool &local,
                                       bool &pushdown_selected) const noexcept {
    local = owner_shard == local_shard;
    pushdown_selected =
        !local &&
        (policy == DistancePolicy::kPushdown ||
         (policy == DistancePolicy::kAuto && id_count >= pull_threshold));
    return pushdown_selected ? pushdown : pull;
  }

  [[nodiscard]] Result<DistanceTicket>
  Encode(DistanceTicket child, bool pushdown_selected) const noexcept {
    if (child.value == 0 || child.generation == 0 ||
        (child.value & kBackendMask) != 0) {
      return Status(StatusCode::kProtocolError,
                    "hybrid backend ticket cannot be tagged");
    }
    child.value |= pushdown_selected ? kPushdownTicket : kPullTicket;
    return child;
  }

  [[nodiscard]] Result<std::pair<DistanceOracle *, DistanceTicket>>
  Decode(DistanceTicket tagged) const noexcept {
    const auto backend_tag = tagged.value & kBackendMask;
    if (tagged.generation == 0 ||
        (backend_tag != kPullTicket && backend_tag != kPushdownTicket)) {
      return Status(StatusCode::kProtocolError,
                    "hybrid distance ticket is invalid");
    }
    tagged.value &= ~kBackendMask;
    if (tagged.value == 0) {
      return Status(StatusCode::kProtocolError,
                    "hybrid distance ticket is empty");
    }
    return std::pair{backend_tag == kPushdownTicket ? pushdown : pull, tagged};
  }

  shard_id_t local_shard{0};
  DistancePolicy policy{DistancePolicy::kAuto};
  std::uint32_t pull_threshold{0};
  DistanceOracle *pull{nullptr};
  DistanceOracle *pushdown{nullptr};
  std::atomic<std::uint64_t> local_batches{0};
  std::atomic<std::uint64_t> pull_batches{0};
  std::atomic<std::uint64_t> pushdown_batches{0};
};

HybridDistanceOracle::HybridDistanceOracle(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

HybridDistanceOracle::~HybridDistanceOracle() = default;

Result<std::unique_ptr<HybridDistanceOracle>>
HybridDistanceOracle::Create(shard_id_t local_shard, DistancePolicy policy,
                             std::uint32_t pull_threshold, DistanceOracle &pull,
                             DistanceOracle &pushdown) {
  if (pull_threshold == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid hybrid distance configuration");
  }
  try {
    auto impl = std::make_unique<Impl>();
    impl->local_shard = local_shard;
    impl->policy = policy;
    impl->pull_threshold = pull_threshold;
    impl->pull = &pull;
    impl->pushdown = &pushdown;
    return std::unique_ptr<HybridDistanceOracle>(
        new HybridDistanceOracle(std::move(impl)));
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate hybrid distance oracle");
  }
}

Result<DistanceTicket> HybridDistanceOracle::Submit(
    QueryKey query, std::span<const float> query_vector, shard_id_t owner_shard,
    std::span<const node_id_t> ids, std::span<float> output,
    DistanceRequestOptions options) {
  return SubmitForWait(query, 1, query_vector, owner_shard, ids, output,
                       options);
}

Result<DistanceTicket> HybridDistanceOracle::SubmitForWait(
    QueryKey query, std::uint64_t wait_epoch,
    std::span<const float> query_vector, shard_id_t owner_shard,
    std::span<const node_id_t> ids, std::span<float> output,
    DistanceRequestOptions options) {
  try {
    auto storage = std::make_shared<const std::vector<float>>(
        query_vector.begin(), query_vector.end());
    return SubmitLeaseForWait({query, std::move(storage)}, wait_epoch,
                              owner_shard, ids, output, options);
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to snapshot hybrid query vector");
  }
}

Result<DistanceTicket> HybridDistanceOracle::SubmitLeaseForWait(
    const QueryVectorLease &lease, std::uint64_t wait_epoch,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  bool local = false;
  bool use_pushdown = false;
  auto *backend = impl_->Select(owner_shard, ids.size(), local, use_pushdown);
  if (!use_pushdown) {
    options = {};
  }
  auto submitted = backend->SubmitLeaseForWait(lease, wait_epoch, owner_shard,
                                               ids, output, options);
  if (!submitted.ok()) {
    return submitted.status();
  }
  auto tagged = impl_->Encode(submitted.value(), use_pushdown);
  if (!tagged.ok()) {
    return tagged.status();
  }
  auto &counter = local          ? impl_->local_batches
                  : use_pushdown ? impl_->pushdown_batches
                                 : impl_->pull_batches;
  counter.fetch_add(1, std::memory_order_relaxed);
  return tagged;
}

Result<bool> HybridDistanceOracle::Poll(DistanceTicket ticket) {
  auto decoded = impl_->Decode(ticket);
  if (!decoded.ok()) {
    return decoded.status();
  }
  return decoded.value().first->Poll(decoded.value().second);
}

bool HybridDistanceOracle::SupportsPreparedWait() const noexcept {
  return impl_->pull->SupportsPreparedWait() &&
         impl_->pushdown->SupportsPreparedWait();
}

Result<PreparedDistanceOperation> HybridDistanceOracle::PrepareForWait(
    QueryWaitPreparation preparation, std::span<const float> query_vector,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  try {
    auto storage = std::make_shared<const std::vector<float>>(
        query_vector.begin(), query_vector.end());
    return PrepareLeaseForWait(preparation,
                               {preparation.query, std::move(storage)},
                               owner_shard, ids, output, options);
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to snapshot hybrid query vector");
  }
}

Result<PreparedDistanceOperation> HybridDistanceOracle::PrepareLeaseForWait(
    QueryWaitPreparation preparation, const QueryVectorLease &lease,
    shard_id_t owner_shard, std::span<const node_id_t> ids,
    std::span<float> output, DistanceRequestOptions options) {
  bool local = false;
  bool use_pushdown = false;
  auto *backend = impl_->Select(owner_shard, ids.size(), local, use_pushdown);
  if (!use_pushdown) {
    options = {};
  }
  if (!backend->SupportsPreparedWait()) {
    return Status(StatusCode::kProtocolError,
                  "hybrid backend does not support prepared waits");
  }
  auto prepared = backend->PrepareLeaseForWait(preparation, lease, owner_shard,
                                               ids, output, options);
  if (!prepared.ok()) {
    return prepared.status();
  }
  auto tagged = impl_->Encode(prepared.value().ticket, use_pushdown);
  if (!tagged.ok()) {
    (void)backend->CancelPrepared(prepared.value().ticket);
    return tagged.status();
  }
  auto &counter = local          ? impl_->local_batches
                  : use_pushdown ? impl_->pushdown_batches
                                 : impl_->pull_batches;
  counter.fetch_add(1, std::memory_order_relaxed);
  return PreparedDistanceOperation{tagged.value(), prepared.value().owner_lane};
}

Status HybridDistanceOracle::PublishPrepared(DistanceTicket ticket) {
  auto decoded = impl_->Decode(ticket);
  if (!decoded.ok()) {
    return decoded.status();
  }
  return decoded.value().first->PublishPrepared(decoded.value().second);
}

Status HybridDistanceOracle::CancelPrepared(DistanceTicket ticket) {
  auto decoded = impl_->Decode(ticket);
  if (!decoded.ok()) {
    return decoded.status();
  }
  return decoded.value().first->CancelPrepared(decoded.value().second);
}

Status HybridDistanceOracle::Finish(DistanceTicket ticket) {
  auto decoded = impl_->Decode(ticket);
  if (!decoded.ok()) {
    return decoded.status();
  }
  return decoded.value().first->Finish(decoded.value().second);
}

bool HybridDistanceOracle::CanComputeLocalTrusted(
    shard_id_t owner_shard) const noexcept {
  return owner_shard == impl_->local_shard;
}

Result<bool> HybridDistanceOracle::ComputeLocalTrusted(
    std::span<const float> query_vector, shard_id_t owner_shard,
    std::span<const std::uint64_t> local_offsets, std::span<float> output) {
  if (owner_shard != impl_->local_shard) {
    return false;
  }
  auto computed = impl_->pull->ComputeLocalTrusted(query_vector, owner_shard,
                                                   local_offsets, output);
  if (computed.ok() && computed.value()) {
    impl_->local_batches.fetch_add(1, std::memory_order_relaxed);
  }
  return computed;
}

Result<bool> HybridDistanceOracle::ComputeLocalTrustedLease(
    const QueryVectorLease &lease, shard_id_t owner_shard,
    std::span<const std::uint64_t> local_offsets, std::span<float> output) {
  if (owner_shard != impl_->local_shard) {
    return false;
  }
  auto computed = impl_->pull->ComputeLocalTrustedLease(lease, owner_shard,
                                                        local_offsets, output);
  if (computed.ok() && computed.value()) {
    impl_->local_batches.fetch_add(1, std::memory_order_relaxed);
  }
  return computed;
}

HybridDistanceProfile HybridDistanceOracle::profile() const {
  return {impl_->local_batches.load(std::memory_order_relaxed),
          impl_->pull_batches.load(std::memory_order_relaxed),
          impl_->pushdown_batches.load(std::memory_order_relaxed)};
}

} // namespace hnsw_urma
