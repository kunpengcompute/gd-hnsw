#include "hnsw_urma/runtime/rpc_runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/transport/urma_resources.hpp"

namespace hnsw_urma {
namespace {

constexpr std::uint32_t kQueueBitmapWordBits = 64;

std::uint64_t SteadyNowNanoseconds() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}

template <typename Word>
std::optional<std::uint32_t> FindNextSetBit(
    const Word* words,
    std::uint32_t bit_count, std::uint32_t cursor) noexcept {
  if (bit_count == 0) return std::nullopt;
  for (std::uint32_t offset = 0; offset < bit_count; ++offset) {
    const auto bit = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(cursor) + offset) % bit_count);
    const auto value = words[bit / kQueueBitmapWordBits].bits.load(
        std::memory_order_acquire);
    if ((value & (std::uint64_t{1} << (bit % kQueueBitmapWordBits))) != 0) {
      return bit;
    }
  }
  return std::nullopt;
}

Result<std::size_t> ActiveIndexCapacity(std::size_t maximum_entries) {
  auto doubled = CheckedMultiply(maximum_entries, std::size_t{2});
  if (!doubled.ok()) return doubled.status();
  std::size_t capacity = 2;
  while (capacity < doubled.value()) {
    auto next = CheckedMultiply(capacity, std::size_t{2});
    if (!next.ok()) return next.status();
    capacity = next.value();
  }
  return capacity;
}

std::size_t ActiveTicketHash(std::uint32_t rank, ticket_t ticket) noexcept {
  auto value = static_cast<std::uint64_t>(ticket) ^
               (static_cast<std::uint64_t>(rank) << 32U);
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return static_cast<std::size_t>(value ^ (value >> 31U));
}

}  // namespace

Result<ActivePeerSet> ActivePeerSet::Create(std::uint32_t peer_count) {
  if (peer_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "active peer set must contain peers");
  }
  try {
    ActivePeerSet result;
    result.positions_.assign(peer_count, kInactive);
    result.peers_.reserve(peer_count);
    return result;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate active peer set");
  }
}

bool ActivePeerSet::Activate(std::uint32_t peer) noexcept {
  if (peer >= positions_.size() || positions_[peer] != kInactive) return false;
  positions_[peer] = static_cast<std::uint32_t>(peers_.size());
  peers_.push_back(peer);
  return true;
}

bool ActivePeerSet::Deactivate(std::uint32_t peer) noexcept {
  if (peer >= positions_.size() || positions_[peer] == kInactive) return false;
  const auto position = positions_[peer];
  const auto last = peers_.back();
  peers_[position] = last;
  positions_[last] = position;
  peers_.pop_back();
  positions_[peer] = kInactive;
  return true;
}

bool ActivePeerSet::contains(std::uint32_t peer) const noexcept {
  return peer < positions_.size() && positions_[peer] != kInactive;
}

Status RpcClientTask::Prepare(
    ticket_t immutable_ticket, QueryKey query_key,
    std::uint64_t immutable_wait_epoch,
    std::uint32_t immutable_destination_rank,
    shard_id_t immutable_owner_shard,
    std::span<const node_id_t> request_ids,
    std::span<const float> query_values,
    std::span<float> result_values,
    std::uint32_t request_flags,
    float immutable_upper_distance,
    std::chrono::steady_clock::time_point immutable_deadline) noexcept {
  if (immutable_ticket == 0 || query_key.generation == 0 ||
      immutable_wait_epoch == 0 || request_ids.empty() || query_values.empty() ||
      result_values.size() != request_ids.size() ||
      (request_flags & ~DISTANCE_BATCH_REQUEST_KNOWN_FLAGS) != 0 ||
      ((request_flags & kBatchHasUpperDistance) != 0 &&
       !std::isfinite(immutable_upper_distance))) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC client task parameters are invalid");
  }
  auto expected = RpcClientTaskState::kFree;
  if (!state.compare_exchange_strong(expected, RpcClientTaskState::kPrepared,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "RPC client task is already in use");
  }
  ticket = immutable_ticket;
  query = query_key;
  wait_epoch = immutable_wait_epoch;
  destination_rank = immutable_destination_rank;
  owner_shard = immutable_owner_shard;
  ids = request_ids;
  query_vector = query_values;
  output = result_values;
  flags = request_flags & ~kBatchHasQueryVector;
  upper_distance = immutable_upper_distance;
  deadline = immutable_deadline;
  status = StatusCode::kOk;
  return Status::Ok();
}

void RpcClientTask::Reset() noexcept {
  ticket = 0;
  query = {};
  wait_epoch = 0;
  request_sequence = 0;
  deadline = std::chrono::steady_clock::time_point::max();
  destination_rank = 0;
  request_slot_id = std::numeric_limits<std::uint32_t>::max();
  owner_shard = 0;
  flags = 0;
  ids = {};
  query_vector = {};
  output = {};
  upper_distance = 0.0F;
  status = StatusCode::kOk;
  owner_received_nanoseconds = 0;
  request_prepare_started_nanoseconds = 0;
  request_prepared_nanoseconds = 0;
  request_posted_nanoseconds = 0;
  terminal_nanoseconds = 0;
  response_received_nanoseconds = 0;
  operation_slot = std::numeric_limits<std::uint32_t>::max();
  state.store(RpcClientTaskState::kFree, std::memory_order_release);
}

Result<std::uint64_t> RpcSubmitMatrix::AuxiliaryBytes(
    std::uint32_t searches, std::uint32_t owners, std::uint32_t ranks) {
  const auto cells = static_cast<std::uint64_t>(searches) * owners;
  const auto words = (static_cast<std::uint64_t>(searches) + 63U) / 64U * owners;
  auto activity = CheckedMultiply(cells, std::uint64_t{sizeof(QueueActivity)});
  auto bitmap = CheckedMultiply(words, std::uint64_t{sizeof(QueueBitmapWord)});
  if (!activity.ok() || !bitmap.ok()) return Status(StatusCode::kOutOfRange, "RPC submit metadata overflows");
  auto storage = CheckedAdd(activity.value(), bitmap.value());
  if (!storage.ok()) return storage.status();
  const auto fixed = sizeof(RpcSubmitMatrix) +
      static_cast<std::uint64_t>(ranks) * sizeof(std::uint32_t) +
      static_cast<std::uint64_t>(owners) * (sizeof(std::size_t) + sizeof(CountingNotification)) +
      (static_cast<std::uint64_t>(searches) + owners) * sizeof(RpcSubmitProfile);
  return CheckedAdd(storage.value(), fixed);
}

Result<std::unique_ptr<RpcSubmitMatrix>> RpcSubmitMatrix::Create(
    std::uint32_t search_workers, std::uint32_t rpc_threads,
    std::uint32_t depth_per_cell,
    std::span<const std::uint32_t> rpc_owner_by_rank) {
  if (search_workers == 0 || rpc_threads == 0 || depth_per_cell == 0 ||
      rpc_owner_by_rank.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC submit matrix dimensions must be non-zero");
  }
  try {
    auto matrix = std::unique_ptr<RpcSubmitMatrix>(new RpcSubmitMatrix());
    matrix->search_workers_ = search_workers;
    matrix->rpc_threads_ = rpc_threads;
    matrix->bitmap_words_per_owner_ =
        search_workers / kQueueBitmapWordBits +
        (search_workers % kQueueBitmapWordBits != 0 ? 1U : 0U);
    matrix->rpc_owner_by_rank_.assign(rpc_owner_by_rank.begin(),
                                      rpc_owner_by_rank.end());
    for (const auto owner : matrix->rpc_owner_by_rank_) {
      if (owner != kNoCommunicationOwner && owner >= rpc_threads) {
        return Status(StatusCode::kInvalidArgument,
                      "RPC owner map references a missing thread");
      }
    }
    auto cell_count = CheckedMultiply<std::uint64_t>(search_workers, rpc_threads);
    auto bitmap_cells = CheckedMultiply<std::uint64_t>(
        rpc_threads, matrix->bitmap_words_per_owner_);
    if (!cell_count.ok() || !bitmap_cells.ok() ||
        cell_count.value() > std::numeric_limits<std::size_t>::max() ||
        bitmap_cells.value() > std::numeric_limits<std::size_t>::max()) {
      return Status(StatusCode::kOutOfRange,
                    "RPC submit matrix cell count overflows");
    }
    matrix->queues_.reserve(static_cast<std::size_t>(cell_count.value()));
    for (std::uint64_t cell = 0; cell < cell_count.value(); ++cell) {
      auto queue = SpscRing<RpcSubmitHandle>::Create(depth_per_cell);
      if (!queue.ok()) return queue.status();
      matrix->queues_.push_back(std::move(queue).value());
    }
    matrix->owner_cursors_.assign(rpc_threads, 0);
    matrix->nonempty_bitmap_ =
        std::make_unique<QueueBitmapWord[]>(bitmap_cells.value());
    matrix->queue_activity_ =
        std::make_unique<QueueActivity[]>(cell_count.value());
    matrix->producer_profiles_ =
        std::make_unique<RpcSubmitProfile[]>(search_workers);
    matrix->consumer_profiles_ =
        std::make_unique<RpcSubmitProfile[]>(rpc_threads);
    matrix->owner_ready_ = std::make_unique<CountingNotification[]>(rpc_threads);
    return matrix;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate RPC submit matrix");
  }
}

std::size_t RpcSubmitMatrix::Index(std::uint32_t worker,
                                   std::uint32_t owner) const noexcept {
  return static_cast<std::size_t>(worker) * rpc_threads_ + owner;
}

bool RpcSubmitMatrix::MarkNonempty(std::uint32_t worker,
                                   std::uint32_t owner) noexcept {
  const auto was_nonempty = queue_activity_[Index(worker, owner)].nonempty.exchange(
      true, std::memory_order_acq_rel);
  if (was_nonempty) return true;
  const auto index = static_cast<std::size_t>(owner) * bitmap_words_per_owner_ +
                     worker / kQueueBitmapWordBits;
  nonempty_bitmap_[index].bits.fetch_or(
      std::uint64_t{1} << (worker % kQueueBitmapWordBits),
      std::memory_order_release);
  return false;
}

void RpcSubmitMatrix::ClearNonempty(std::uint32_t worker,
                                    std::uint32_t owner) noexcept {
  const auto index = static_cast<std::size_t>(owner) * bitmap_words_per_owner_ +
                     worker / kQueueBitmapWordBits;
  nonempty_bitmap_[index].bits.fetch_and(
      ~(std::uint64_t{1} << (worker % kQueueBitmapWordBits)),
      std::memory_order_acq_rel);
  queue_activity_[Index(worker, owner)].nonempty.store(false,
                                                       std::memory_order_release);
}

std::optional<std::uint32_t> RpcSubmitMatrix::NextNonemptyWorker(
    std::uint32_t owner, std::uint32_t cursor) const noexcept {
  const auto* words = nonempty_bitmap_.get() +
      static_cast<std::size_t>(owner) * bitmap_words_per_owner_;
  return FindNextSetBit(words, search_workers_, cursor);
}

Result<std::uint32_t> RpcSubmitMatrix::OwnerForRank(
    std::uint32_t destination_rank) const noexcept {
  if (destination_rank >= rpc_owner_by_rank_.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC destination rank is outside the owner map");
  }
  const auto owner = rpc_owner_by_rank_[destination_rank];
  if (owner == kNoCommunicationOwner || owner >= rpc_threads_) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC destination is local or has no owner");
  }
  return owner;
}

Status RpcSubmitMatrix::Reserve(std::uint32_t search_worker,
                                std::uint32_t destination_rank) noexcept {
  auto owner = OwnerForRank(destination_rank);
  if (search_worker >= search_workers_ || !owner.ok()) {
    return owner.ok() ? Status(StatusCode::kInvalidArgument,
                               "RPC submit producer is invalid")
                      : owner.status();
  }
  auto result = queues_[Index(search_worker, owner.value())].Reserve(1);
  if (!result.ok()) ++producer_profiles_[search_worker].full_count;
  return result;
}

Status RpcSubmitMatrix::CancelReservation(
    std::uint32_t search_worker, std::uint32_t destination_rank) noexcept {
  auto owner = OwnerForRank(destination_rank);
  if (search_worker >= search_workers_ || !owner.ok()) {
    return owner.ok() ? Status(StatusCode::kInvalidArgument,
                               "RPC reservation producer is invalid")
                      : owner.status();
  }
  return queues_[Index(search_worker, owner.value())].CancelReservation(1);
}

Status RpcSubmitMatrix::PublishReserved(std::uint32_t search_worker,
                                        RpcClientTask* task) noexcept {
  if (search_worker >= search_workers_ || task == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "reserved RPC submit handle is invalid");
  }
  auto owner = OwnerForRank(task->destination_rank);
  if (!owner.ok()) return owner.status();
  auto expected = RpcClientTaskState::kPrepared;
  if (!task->state.compare_exchange_strong(expected, RpcClientTaskState::kQueued,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "reserved RPC task is not prepared");
  }
  auto& queue = queues_[Index(search_worker, owner.value())];
  auto result = queue.PushReserved(RpcSubmitHandle{
      task, task->query.generation, task->wait_epoch, task->ticket});
  if (!result.ok()) {
    task->state.store(RpcClientTaskState::kPrepared, std::memory_order_release);
    return result;
  }
  auto& profile = producer_profiles_[search_worker];
  ++profile.published;
  profile.high_watermark = std::max(profile.high_watermark,
                                    queue.metrics().high_watermark);
  const bool was_nonempty = MarkNonempty(search_worker, owner.value());
  if (adaptive_notifications_enabled_ && !was_nonempty) {
    owner_ready_[owner.value()].Notify();
  }
  return Status::Ok();
}

Status RpcSubmitMatrix::Publish(std::uint32_t search_worker,
                                RpcClientTask* task) noexcept {
  if (search_worker >= search_workers_ || task == nullptr) {
    return Status(StatusCode::kInvalidArgument, "RPC submit handle is invalid");
  }
  auto owner = OwnerForRank(task->destination_rank);
  if (!owner.ok()) return owner.status();
  auto expected = RpcClientTaskState::kPrepared;
  if (!task->state.compare_exchange_strong(expected, RpcClientTaskState::kQueued,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "RPC task is not prepared for publish");
  }
  auto& queue = queues_[Index(search_worker, owner.value())];
  auto result = queue.TryPush(RpcSubmitHandle{
      task, task->query.generation, task->wait_epoch, task->ticket});
  if (!result.ok()) {
    task->state.store(RpcClientTaskState::kPrepared, std::memory_order_release);
    ++producer_profiles_[search_worker].full_count;
    return result;
  }
  auto& profile = producer_profiles_[search_worker];
  ++profile.published;
  profile.high_watermark = std::max(profile.high_watermark,
                                    queue.metrics().high_watermark);
  const bool was_nonempty = MarkNonempty(search_worker, owner.value());
  if (adaptive_notifications_enabled_ && !was_nonempty) {
    owner_ready_[owner.value()].Notify();
  }
  return Status::Ok();
}

std::optional<RpcClientTask*> RpcSubmitMatrix::Pop(
    std::uint32_t rpc_owner) noexcept {
  if (rpc_owner >= rpc_threads_) return std::nullopt;
  auto& cursor = owner_cursors_[rpc_owner];
  while (true) {
    auto next = NextNonemptyWorker(rpc_owner,
        static_cast<std::uint32_t>(cursor));
    if (!next.has_value()) return std::nullopt;
    const auto worker = *next;
    auto handle = queues_[Index(worker, rpc_owner)].TryPop();
    if (!handle.has_value()) {
      ClearNonempty(worker, rpc_owner);
      handle = queues_[Index(worker, rpc_owner)].TryPop();
      if (handle.has_value()) (void)MarkNonempty(worker, rpc_owner);
    }
    if (!handle.has_value()) continue;
    cursor = (worker + 1U) % search_workers_;
    auto* task = handle->task;
    if (task == nullptr || handle->query_generation != task->query.generation ||
        handle->wait_epoch != task->wait_epoch || handle->ticket != task->ticket) {
      return nullptr;
    }
    auto expected = RpcClientTaskState::kQueued;
    if (!task->state.compare_exchange_strong(
            expected, RpcClientTaskState::kOwnerPending,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return nullptr;
    }
    ++consumer_profiles_[rpc_owner].consumed;
    return task;
  }
}

RpcSubmitProfile RpcSubmitMatrix::profile() const noexcept {
  RpcSubmitProfile result;
  for (std::uint32_t worker = 0; worker < search_workers_; ++worker) {
    result.published += producer_profiles_[worker].published;
    result.full_count += producer_profiles_[worker].full_count;
    result.high_watermark = std::max(result.high_watermark,
                                     producer_profiles_[worker].high_watermark);
  }
  for (std::uint32_t owner = 0; owner < rpc_threads_; ++owner) {
    result.consumed += consumer_profiles_[owner].consumed;
  }
  return result;
}

bool RpcSubmitMatrix::drained(std::uint32_t rpc_owner) const noexcept {
  if (rpc_owner >= rpc_threads_) return false;
  for (std::uint32_t worker = 0; worker < search_workers_; ++worker) {
    if (queues_[Index(worker, rpc_owner)].size() != 0) return false;
  }
  return true;
}

Result<std::uint64_t> RpcClientOwner::RequiredBytes(
    std::uint32_t world_size, std::uint32_t pending_capacity,
    std::uint32_t query_slots, std::uint32_t request_ring_depth) {
  auto index_capacity = ActiveIndexCapacity(pending_capacity);
  if (!index_capacity.ok()) return index_capacity.status();
  std::uint64_t bytes = sizeof(RpcClientOwner);
  const auto add = [&](std::uint64_t count, std::uint64_t size) -> Status {
    auto product = CheckedMultiply(count, size);
    if (!product.ok()) return product.status();
    auto sum = CheckedAdd(bytes, product.value());
    if (!sum.ok()) return sum.status();
    bytes = sum.value();
    return Status::Ok();
  };
  auto batch_entries = CheckedMultiply<std::uint64_t>(world_size, request_ring_depth);
  auto inflight_entries = batch_entries.ok()
      ? CheckedMultiply(batch_entries.value(), std::uint64_t{pending_capacity})
      : Result<std::uint64_t>(batch_entries.status());
  if (!inflight_entries.ok()) return inflight_entries.status();
  for (auto status : {
      add(world_size, sizeof(PeerState) + 2 * sizeof(std::uint32_t)),
      add(static_cast<std::uint64_t>(world_size) * pending_capacity,
          2 * sizeof(RpcClientTask*) + sizeof(std::uint32_t)),
      add(static_cast<std::uint64_t>(world_size) * query_slots, sizeof(QueryPeerSendState)),
      add(batch_entries.value(), sizeof(ClientInflightBatch)),
      add(inflight_entries.value(), sizeof(RpcClientTask*)),
      add(index_capacity.value(), sizeof(ActiveEntry)),
      add(pending_capacity, sizeof(DistanceBatchRequestTaskInput) + sizeof(RpcClientTask*))}) {
    if (!status.ok()) return status;
  }
  return bytes;
}

Result<std::unique_ptr<RpcClientOwner>> RpcClientOwner::Create(
    std::uint32_t owner_thread, std::uint32_t world_size,
    std::uint32_t pending_capacity, RpcSubmitMatrix& submissions,
    std::uint32_t query_slots, std::uint32_t request_ring_depth,
    bool profiling_enabled, RpcClientTaskObserver* task_observer) {
  if (world_size == 0 || pending_capacity == 0 || query_slots == 0 ||
      request_ring_depth == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC client owner dimensions must be non-zero");
  }
  auto bytes = RequiredBytes(world_size, pending_capacity, query_slots, request_ring_depth);
  if (!bytes.ok() || bytes.value() > SIZE_MAX)
    return bytes.ok() ? Status(StatusCode::kOutOfRange, "RPC owner exceeds address space") : bytes.status();
  try {
    auto owner = std::unique_ptr<RpcClientOwner>(new RpcClientOwner());
    owner->owner_thread_ = owner_thread;
    owner->pending_capacity_ = pending_capacity;
    owner->request_ring_depth_ = request_ring_depth;
    owner->submissions_ = &submissions;
    owner->task_observer_ = task_observer;
    owner->profiling_enabled_ = profiling_enabled;
    owner->peers_.resize(world_size);
    auto active = ActivePeerSet::Create(world_size);
    if (!active.ok()) return active.status();
    owner->pending_peers_ = std::move(active).value();
    for (auto& peer : owner->peers_) {
      peer.pending.entries.resize(pending_capacity);
      peer.prepared.reserve(pending_capacity);
      peer.prepared_flags.reserve(pending_capacity);
      peer.query_states.resize(query_slots);
      peer.inflight.resize(request_ring_depth);
      for (auto& batch : peer.inflight) {
        batch.tasks.reserve(pending_capacity);
      }
    }
    owner->active_index_.resize(ActiveIndexCapacity(pending_capacity).value());
    owner->terminal_work_.reserve(pending_capacity);
    owner->request_inputs_.reserve(pending_capacity);
    return owner;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate RPC client owner");
  }
}

Result<RpcClientOwner::QueryPeerSendState*> RpcClientOwner::QueryState(
    std::uint32_t peer_rank, QueryKey query) noexcept {
  if (peer_rank >= peers_.size() || query.generation == 0 ||
      query.query_slot >= peers_[peer_rank].query_states.size()) {
    return Status(StatusCode::kProtocolError,
                  "client query slot is outside the fixed table");
  }
  auto& state = peers_[peer_rank].query_states[query.query_slot];
  if (state.local_generation != query.generation) {
    state.local_generation = query.generation;
    state.query_sent = false;
  }
  return &state;
}

RpcClientTask* RpcClientOwner::FindActive(
    std::uint32_t peer_rank, ticket_t ticket) const noexcept {
  if (ticket == 0 || active_index_.empty()) return nullptr;
  const auto mask = active_index_.size() - 1U;
  auto index = ActiveTicketHash(peer_rank, ticket) & mask;
  for (std::size_t probe = 0; probe < active_index_.size(); ++probe) {
    const auto& entry = active_index_[index];
    if (entry.ticket == 0) return nullptr;
    if (entry.peer_rank == peer_rank && entry.ticket == ticket) return entry.task;
    index = (index + 1U) & mask;
  }
  return nullptr;
}

bool RpcClientOwner::InsertActive(std::uint32_t peer_rank, ticket_t ticket,
                                  RpcClientTask* task) noexcept {
  if (ticket == 0 || task == nullptr || active_index_.empty()) return false;
  const auto mask = active_index_.size() - 1U;
  auto index = ActiveTicketHash(peer_rank, ticket) & mask;
  for (std::size_t probe = 0; probe < active_index_.size(); ++probe) {
    auto& entry = active_index_[index];
    if (entry.ticket == 0) {
      entry = {peer_rank, ticket, task};
      ++active_count_;
      return true;
    }
    if (entry.peer_rank == peer_rank && entry.ticket == ticket) return false;
    index = (index + 1U) & mask;
  }
  return false;
}

bool RpcClientOwner::EraseActive(std::uint32_t peer_rank, ticket_t ticket,
                                 const RpcClientTask* task) noexcept {
  if (ticket == 0 || task == nullptr || active_index_.empty()) return false;
  const auto mask = active_index_.size() - 1U;
  auto index = ActiveTicketHash(peer_rank, ticket) & mask;
  for (std::size_t probe = 0; probe < active_index_.size(); ++probe) {
    auto& entry = active_index_[index];
    if (entry.ticket == 0) return false;
    if (entry.task == task && entry.peer_rank == peer_rank &&
        entry.ticket == ticket) {
      auto hole = index;
      auto scan = (index + 1U) & mask;
      while (active_index_[scan].ticket != 0) {
        const auto home = ActiveTicketHash(active_index_[scan].peer_rank,
                                           active_index_[scan].ticket) & mask;
        const auto scan_distance = (scan - home) & mask;
        const auto hole_distance = (hole - home) & mask;
        if (hole_distance < scan_distance) {
          active_index_[hole] = active_index_[scan];
          hole = scan;
        }
        scan = (scan + 1U) & mask;
      }
      active_index_[hole] = {};
      --active_count_;
      return true;
    }
    index = (index + 1U) & mask;
  }
  return false;
}

Result<std::uint32_t> RpcClientOwner::DrainSubmissions(std::uint32_t budget) {
  std::uint32_t drained_count = 0;
  while (drained_count < budget) {
    auto task = submissions_->Pop(owner_thread_);
    if (!task.has_value()) break;
    if (*task == nullptr) {
      return Status(StatusCode::kProtocolError,
                    "RPC submit matrix returned a stale handle");
    }
    auto* value = *task;
    if (active_count_ + pending_count_ >= pending_capacity_ ||
        value->destination_rank >= peers_.size() ||
        FindActive(value->destination_rank, value->ticket) != nullptr) {
      return Status(StatusCode::kProtocolError,
                    "RPC submission violates owner capacity or identity");
    }
    auto& peer = peers_[value->destination_rank];
    for (std::size_t i = 0; i < peer.pending.size(); ++i) {
      const auto* pending = peer.pending[i];
      if (pending->ticket == value->ticket) {
        return Status(StatusCode::kProtocolError,
                      "RPC pending ticket is duplicated");
      }
    }
    if (profiling_enabled_) value->owner_received_nanoseconds = SteadyNowNanoseconds();
    if (task_observer_ != nullptr) {
      auto observed = task_observer_->OnRpcTaskOwned(owner_thread_, *value);
      if (!observed.ok()) return observed.status();
      if (!observed.value()) { ++drained_count; ++profile_.logical_tasks; continue; }
    }
    peer.pending.push_back(value);
    ++pending_count_;
    (void)pending_peers_.Activate(value->destination_rank);
    ++profile_.logical_tasks;
    ++drained_count;
    profile_.pending_high_watermark = std::max<std::uint64_t>(
        profile_.pending_high_watermark, active_count_ + pending_count_);
  }
  return drained_count;
}

Result<std::optional<PreparedBatch>> RpcClientOwner::PrepareRequestBatch(
    std::uint32_t peer_rank, std::uint32_t slot_id,
    std::uint64_t sequence, std::uint32_t max_tasks,
    std::uint32_t max_request_bytes, std::uint32_t max_response_bytes,
    std::span<std::byte> output) {
  if (peer_rank >= peers_.size() || slot_id >= request_ring_depth_ ||
      sequence == 0 || max_tasks == 0 || max_request_bytes == 0 ||
      max_response_bytes == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "request batch parameters are invalid");
  }
  auto& peer = peers_[peer_rank];
  if (!peer.prepared.empty() || peer.inflight[slot_id].sequence != 0) {
    return Status(StatusCode::kProtocolError,
                  "request slot or prepare state is already active");
  }
  if (peer.pending.empty()) return std::optional<PreparedBatch>{};
  const auto prepare_started_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
  request_inputs_.clear();
  std::uint64_t total_ids = 0;
  for (std::size_t pending_index = 0; pending_index < peer.pending.size(); ++pending_index) {
    auto* task = peer.pending[pending_index];
    if (request_inputs_.size() >= max_tasks) break;
    auto query_state = QueryState(peer_rank, task->query);
    if (!query_state.ok()) return query_state.status();
    bool query_already_selected = false;
    for (auto* selected : peer.prepared) {
      if (selected->query == task->query) {
        query_already_selected = true;
        break;
      }
    }
    const bool attach_query = !query_state.value()->query_sent &&
                              !query_already_selected;
    const auto wire_flags = attach_query
        ? task->flags | kBatchHasQueryVector
        : task->flags & ~kBatchHasQueryVector;
    DistanceBatchRequestTaskInput input{
        task->ticket, task->query.query_slot, task->owner_shard, wire_flags,
        task->ids, attach_query ? task->query_vector : std::span<const float>{},
        static_cast<std::uint32_t>(task->query_vector.size()),
        task->upper_distance};
    request_inputs_.push_back(input);
    auto bytes = EncodedDistanceBatchRequestBytes(request_inputs_);
    auto next_total_ids = CheckedAdd<std::uint64_t>(total_ids, task->ids.size());
    auto response_bytes = next_total_ids.ok()
        ? DistanceBatchResponseBytesForShape(
              static_cast<std::uint32_t>(request_inputs_.size()),
              next_total_ids.value())
        : Result<std::size_t>(next_total_ids.status());
    const auto request_limit =
        std::min<std::size_t>(max_request_bytes, output.size());
    if (!bytes.ok() || bytes.value() > request_limit || !response_bytes.ok() ||
        response_bytes.value() > max_response_bytes) {
      request_inputs_.pop_back();
      if (request_inputs_.empty()) {
        if (!bytes.ok()) return bytes.status();
        if (!response_bytes.ok()) return response_bytes.status();
        request_inputs_.clear();
        return Status(StatusCode::kInvalidArgument,
                      "single request task exceeds request or response slot");
      }
      break;
    }
    total_ids = next_total_ids.value();
    peer.prepared.push_back(task);
    peer.prepared_flags.push_back(wire_flags);
  }
  if (request_inputs_.empty()) return std::optional<PreparedBatch>{};
  auto encoded = EncodeDistanceBatchRequest(sequence, request_inputs_, output);
  request_inputs_.clear();
  if (!encoded.ok()) {
    peer.prepared.clear();
    peer.prepared_flags.clear();
    return encoded.status();
  }
  std::size_t transitioned = 0;
  for (auto* task : peer.prepared) {
    auto expected = RpcClientTaskState::kOwnerPending;
    if (!task->state.compare_exchange_strong(expected,
                                              RpcClientTaskState::kBatchPrepared,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      for (std::size_t index = 0; index < transitioned; ++index) {
        peer.prepared[index]->state.store(RpcClientTaskState::kOwnerPending,
                                           std::memory_order_release);
      }
      peer.prepared.clear();
      peer.prepared_flags.clear();
      return Status(StatusCode::kProtocolError,
                    "request task changed state during batch prepare");
    }
    ++transitioned;
  }
  if (profiling_enabled_) {
    const auto prepared_at = SteadyNowNanoseconds();
    for (auto* task : peer.prepared) {
      task->request_prepare_started_nanoseconds = prepare_started_at;
      task->request_prepared_nanoseconds = prepared_at;
    }
  }
  return std::optional<PreparedBatch>(PreparedBatch{
      peer_rank, slot_id, sequence,
      static_cast<std::uint32_t>(peer.prepared.size()),
      static_cast<std::uint32_t>(encoded.value())});
}

Status RpcClientOwner::CommitPreparedRequest(PreparedBatch batch) noexcept {
  if (batch.peer_rank >= peers_.size() || batch.slot_id >= request_ring_depth_) {
    return Status(StatusCode::kInvalidArgument,
                  "prepared request slot is invalid");
  }
  auto& peer = peers_[batch.peer_rank];
  auto& inflight = peer.inflight[batch.slot_id];
  if (batch.sequence == 0 || inflight.sequence != 0 ||
      batch.task_count != peer.prepared.size() ||
      batch.task_count != peer.prepared_flags.size() ||
      batch.task_count > peer.pending.size() || batch.task_count > pending_count_) {
    return Status(StatusCode::kProtocolError,
                  "prepared request token is stale");
  }
  for (std::size_t index = 0; index < peer.prepared.size(); ++index) {
    if (peer.pending[index] != peer.prepared[index] ||
        peer.prepared[index]->state.load(std::memory_order_acquire) !=
            RpcClientTaskState::kBatchPrepared) {
      return Status(StatusCode::kProtocolError,
                    "prepared request no longer matches pending prefix");
    }
  }
  const auto posted_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
  inflight.sequence = batch.sequence;
  inflight.tasks.clear();
  for (std::size_t index = 0; index < peer.prepared.size(); ++index) {
    auto* task = peer.prepared[index];
    if (profiling_enabled_ && task->owner_received_nanoseconds != 0 &&
        task->request_prepare_started_nanoseconds >=
            task->owner_received_nanoseconds &&
        task->request_prepared_nanoseconds >=
            task->request_prepare_started_nanoseconds &&
        posted_at >= task->request_prepared_nanoseconds) {
      profile_.owner_queue_nanoseconds +=
          posted_at - task->owner_received_nanoseconds;
      ++profile_.owner_queue_samples;
      profile_.owner_pending_nanoseconds +=
          task->request_prepare_started_nanoseconds -
          task->owner_received_nanoseconds;
      profile_.request_prepare_nanoseconds +=
          task->request_prepared_nanoseconds -
          task->request_prepare_started_nanoseconds;
      profile_.request_post_wait_nanoseconds +=
          posted_at - task->request_prepared_nanoseconds;
    }
    if (!InsertActive(batch.peer_rank, task->ticket, task)) {
      return Status(StatusCode::kProtocolError,
                    "request active ticket index rejected a task");
    }
    auto query_state = QueryState(batch.peer_rank, task->query);
    if (!query_state.ok()) return query_state.status();
    if ((peer.prepared_flags[index] & kBatchHasQueryVector) != 0) {
      query_state.value()->query_sent = true;
    }
    task->flags = peer.prepared_flags[index];
    task->request_sequence = batch.sequence;
    task->request_slot_id = batch.slot_id;
    task->request_posted_nanoseconds = posted_at;
    task->state.store(RpcClientTaskState::kWaitingResponse,
                      std::memory_order_release);
    inflight.tasks.push_back(task);
  }
  for (std::uint32_t i = 0; i < batch.task_count; ++i) peer.pending.pop_front();
  pending_count_ -= batch.task_count;
  peer.waiting_response_tasks += batch.task_count;
  if (peer.pending.empty()) (void)pending_peers_.Deactivate(batch.peer_rank);
  ++profile_.request_batches;
  profile_.request_batch_tasks += batch.task_count;
  profile_.request_batch_bytes += batch.bytes;
  peer.prepared.clear();
  peer.prepared_flags.clear();
  return Status::Ok();
}

Status RpcClientOwner::AbortPreparedRequest(PreparedBatch batch) noexcept {
  if (batch.peer_rank >= peers_.size() || batch.slot_id >= request_ring_depth_) {
    return Status(StatusCode::kInvalidArgument,
                  "prepared request slot is invalid");
  }
  auto& peer = peers_[batch.peer_rank];
  if (batch.sequence == 0 || batch.task_count != peer.prepared.size() ||
      peer.inflight[batch.slot_id].sequence != 0) {
    return Status(StatusCode::kProtocolError,
                  "prepared request abort token is stale");
  }
  for (auto* task : peer.prepared) {
    auto expected = RpcClientTaskState::kBatchPrepared;
    if (!task->state.compare_exchange_strong(expected,
                                              RpcClientTaskState::kOwnerPending,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return Status(StatusCode::kProtocolError,
                    "request task changed state before abort");
    }
    task->request_prepare_started_nanoseconds = 0;
    task->request_prepared_nanoseconds = 0;
  }
  request_inputs_.clear();
  peer.prepared.clear();
  peer.prepared_flags.clear();
  return Status::Ok();
}

Status RpcClientOwner::AcceptResponse(
    std::uint32_t peer_rank, std::uint32_t slot_id,
    std::uint64_t expected_sequence, std::span<const std::byte> message,
    std::uint64_t notify_observed_nanoseconds) noexcept {
  if (peer_rank >= peers_.size() || slot_id >= request_ring_depth_ ||
      expected_sequence == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "response physical slot is invalid");
  }
  const auto decode_started_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
  auto decoded = DecodeDistanceBatchResponse(message);
  if (!decoded.ok()) return decoded.status();
  const auto response_received_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
  if (profiling_enabled_) {
    profile_.response_decode_nanoseconds += response_received_at - decode_started_at;
    ++profile_.response_decode_samples;
  }
  auto& peer = peers_[peer_rank];
  auto& inflight = peer.inflight[slot_id];
  if (inflight.sequence != expected_sequence ||
      decoded.value().header().batch_sequence != expected_sequence ||
      decoded.value().task_count() != inflight.tasks.size()) {
    return Status(StatusCode::kProtocolError,
                  "response does not match the request slot generation");
  }
  for (std::size_t index = 0; index < inflight.tasks.size(); ++index) {
    auto response = decoded.value().Task(index);
    auto* task = inflight.tasks[index];
    if (!response.ok()) return response.status();
    if (task == nullptr ||
        task->state.load(std::memory_order_acquire) !=
            RpcClientTaskState::kWaitingResponse ||
        FindActive(peer_rank, task->ticket) != task ||
        task->request_slot_id != slot_id ||
        task->request_sequence != expected_sequence ||
        response.value().descriptor.task_ticket != task->ticket ||
        response.value().descriptor.query_slot_id != task->query.query_slot ||
        response.value().ids.size() != response.value().distances.size() ||
        ((task->flags & kBatchHasUpperDistance) == 0 &&
         response.value().ids.size() != task->ids.size()) ||
        response.value().ids.size() > task->ids.size()) {
      return Status(StatusCode::kProtocolError,
                    "response task does not match its request position");
    }
    std::size_t request_index = 0;
    for (std::size_t survivor = 0; survivor < response.value().ids.size(); ++survivor) {
      while (request_index < task->ids.size() &&
             task->ids[request_index] != response.value().ids[survivor]) {
        ++request_index;
      }
      if (request_index == task->ids.size() ||
          ((task->flags & kBatchHasUpperDistance) != 0 &&
           response.value().distances[survivor] > task->upper_distance)) {
        return Status(StatusCode::kProtocolError,
                      "response survivor is not an ordered request member");
      }
      ++request_index;
    }
  }
  if (profiling_enabled_ && notify_observed_nanoseconds != 0 &&
      response_received_at >= notify_observed_nanoseconds) {
    const auto elapsed = response_received_at - notify_observed_nanoseconds;
    profile_.response_notify_to_receive_nanoseconds += elapsed;
    ++profile_.response_notify_to_receive_samples;
    profile_.response_notify_to_receive_task_nanoseconds +=
        elapsed * inflight.tasks.size();
    profile_.response_notify_to_receive_task_samples += inflight.tasks.size();
  }
  terminal_work_.clear();
  for (std::size_t index = 0; index < inflight.tasks.size(); ++index) {
    auto response = decoded.value().Task(index).value();
    auto* task = inflight.tasks[index];
    if ((task->flags & kBatchHasUpperDistance) != 0 &&
        response.ids.size() != task->ids.size()) {
      const auto rejected = std::nextafter(
          task->upper_distance, std::numeric_limits<float>::infinity());
      if (!std::isfinite(rejected)) {
        return Status(StatusCode::kProtocolError,
                      "filtered response has no finite rejection marker");
      }
      std::fill(task->output.begin(), task->output.end(), rejected);
    }
    std::size_t request_index = 0;
    for (std::size_t survivor = 0; survivor < response.ids.size(); ++survivor) {
      while (task->ids[request_index] != response.ids[survivor]) ++request_index;
      task->output[request_index++] = response.distances[survivor];
    }
    task->status = StatusCode::kOk;
    task->response_received_nanoseconds = response_received_at;
    if (profiling_enabled_) {
      if (task->request_posted_nanoseconds != 0 &&
          response_received_at >= task->request_posted_nanoseconds) {
        profile_.round_trip_nanoseconds +=
            response_received_at - task->request_posted_nanoseconds;
        ++profile_.round_trip_samples;
      }
      task->terminal_nanoseconds = SteadyNowNanoseconds();
      profile_.response_received_to_task_terminal_nanoseconds +=
          task->terminal_nanoseconds - response_received_at;
      ++profile_.response_received_to_task_terminal_samples;
    }
    task->state.store(RpcClientTaskState::kTerminal, std::memory_order_release);
    if (!EraseActive(peer_rank, task->ticket, task))
      return Status(StatusCode::kProtocolError, "response active identity vanished");
    terminal_work_.push_back(task);
  }
  peer.waiting_response_tasks -= inflight.tasks.size();
  ++profile_.response_batches;
  profile_.response_batch_tasks += inflight.tasks.size();
  profile_.response_batch_bytes += decoded.value().header().total_bytes;
  profile_.terminal_tasks += inflight.tasks.size();
  inflight.tasks.clear();
  inflight.sequence = 0;
  // Every active/inflight reference and span has been removed. Pop before
  // callback: Search may immediately reset/reuse this task before it returns.
  std::reverse(terminal_work_.begin(), terminal_work_.end());
  while (!terminal_work_.empty()) {
    auto* task = terminal_work_.back();
    terminal_work_.back() = nullptr;
    terminal_work_.pop_back();
    if (task_observer_ != nullptr) {
      auto status = task_observer_->OnRpcTaskTerminal(owner_thread_, *task);
      if (!status.ok()) return status;
    }
  }
  return Status::Ok();
}

Result<bool> RpcClientOwner::ValidatePreparedRequest(PreparedBatch batch) noexcept {
  if (batch.peer_rank >= peers_.size())
    return Status(StatusCode::kInvalidArgument, "invalid prepared peer");
  auto& peer = peers_[batch.peer_rank];
  if (peer.prepared.empty()) return false;
  if (peer.prepared.size() != batch.task_count)
    return Status(StatusCode::kProtocolError, "prepared request changed");
  if (task_observer_) {
    for (auto* task : peer.prepared) {
      if (task_observer_->TaskControlStatus(owner_thread_, *task) != StatusCode::kOk) {
        auto aborted = AbortPreparedRequest(batch);
        if (!aborted.ok()) return aborted;
        return false;
      }
    }
  }
  return true;
}

Result<bool> RpcClientOwner::CancelUnsent(RpcClientTask& task) noexcept {
  const auto state = task.state.load(std::memory_order_relaxed);
  if (state == RpcClientTaskState::kWaitingResponse) return false;
  if (task.destination_rank >= peers_.size() ||
      (state != RpcClientTaskState::kOwnerPending && state != RpcClientTaskState::kBatchPrepared))
    return Status(StatusCode::kProtocolError, "RPC cancellation is not owner pending");
  auto& peer = peers_[task.destination_rank];
  if (state == RpcClientTaskState::kBatchPrepared) {
    for (auto* prepared : peer.prepared)
      prepared->state.store(RpcClientTaskState::kOwnerPending, std::memory_order_relaxed);
    peer.prepared.clear();
    peer.prepared_flags.clear();
    request_inputs_.clear();
  }
  if (!peer.pending.Remove(&task))
    return Status(StatusCode::kProtocolError, "RPC cancellation lost pending identity");
  --pending_count_;
  if (peer.pending.empty()) (void)pending_peers_.Deactivate(task.destination_rank);
  task.state.store(RpcClientTaskState::kTerminal, std::memory_order_relaxed);
  return true;
}

bool RpcClientOwner::drained() const noexcept {
  if (!terminal_work_.empty() || active_count_ != 0 || pending_count_ != 0 ||
      !submissions_->drained(owner_thread_)) return false;
  for (const auto& peer : peers_) {
    if (!peer.pending.empty() || !peer.prepared.empty() ||
        peer.waiting_response_tasks != 0) return false;
    for (const auto& batch : peer.inflight) {
      if (batch.sequence != 0 || !batch.tasks.empty()) return false;
    }
  }
  return true;
}

std::size_t RpcClientOwner::pending_tasks(std::uint32_t peer_rank) const noexcept {
  return peer_rank < peers_.size() ? peers_[peer_rank].pending.size() : 0;
}

std::size_t RpcClientOwner::waiting_response_tasks(
    std::uint32_t peer_rank) const noexcept {
  return peer_rank < peers_.size() ? peers_[peer_rank].waiting_response_tasks : 0;
}

Result<std::unique_ptr<RpcServerRuntime>> RpcServerRuntime::Create(
    RpcMailboxMemory& memory, std::uint32_t distance_workers,
    std::uint32_t max_ids_per_task, std::uint32_t local_rank,
    std::span<const std::uint32_t> shard_to_rank,
    std::span<const std::uint32_t> rpc_owner_by_rank,
    std::uint32_t max_active_queries, std::uint32_t dim,
    bool profiling_enabled, const LocalShard* local_shard) {
  const auto& config = memory.config();
  const auto remote_slot_count =
      static_cast<std::uint64_t>(config.world_size - 1U) * config.ring_depth;
  if (config.world_size < 2 || local_rank >= config.world_size ||
      distance_workers == 0 || distance_workers > remote_slot_count ||
      max_ids_per_task == 0 || max_active_queries == 0 || dim == 0 ||
      shard_to_rank.empty() || rpc_owner_by_rank.size() != config.world_size ||
      local_shard == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "direct-slot server dimensions are invalid");
  }
  std::uint32_t rpc_threads = 0;
  for (std::uint32_t rank = 0; rank < config.world_size; ++rank) {
    const auto owner = rpc_owner_by_rank[rank];
    if (rank == local_rank) {
      if (owner != kNoCommunicationOwner) {
        return Status(StatusCode::kInvalidArgument,
                      "local rank must not have an RPC owner");
      }
      continue;
    }
    if (owner == kNoCommunicationOwner || owner >= config.rpc_threads) {
      return Status(StatusCode::kInvalidArgument,
                    "remote rank has no valid RPC owner");
    }
    rpc_threads = std::max(rpc_threads, owner + 1U);
  }
  if (rpc_threads != config.rpc_threads) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC owner map does not cover configured threads");
  }
  for (const auto rank : shard_to_rank) {
    if (rank >= config.world_size) {
      return Status(StatusCode::kInvalidArgument,
                    "shard map references an invalid rank");
    }
  }
  auto result_count = CheckedMultiply<std::uint64_t>(
      config.world_size, config.ring_depth);
  auto query_count = CheckedMultiply<std::uint64_t>(
      config.world_size, max_active_queries);
  auto query_values = query_count.ok()
      ? CheckedMultiply<std::uint64_t>(query_count.value(), dim)
      : Result<std::uint64_t>(query_count.status());
  if (!result_count.ok() || !query_count.ok() || !query_values.ok() ||
      result_count.value() > std::numeric_limits<std::size_t>::max() ||
      query_count.value() > std::numeric_limits<std::size_t>::max() ||
      query_values.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "direct-slot server storage shape overflows");
  }
  try {
    auto server = std::unique_ptr<RpcServerRuntime>(new RpcServerRuntime());
    server->memory_ = &memory;
    server->world_size_ = config.world_size;
    server->rpc_threads_ = config.rpc_threads;
    server->distance_workers_ = distance_workers;
    server->max_ids_per_task_ = max_ids_per_task;
    server->max_active_queries_ = max_active_queries;
    server->dim_ = dim;
    server->ring_depth_ = config.ring_depth;
    server->local_rank_ = local_rank;
    server->local_shard_ = local_shard;
    server->shard_to_rank_.assign(shard_to_rank.begin(), shard_to_rank.end());
    server->rpc_owner_by_rank_.assign(rpc_owner_by_rank.begin(),
                                      rpc_owner_by_rank.end());
    server->request_slots_by_worker_.resize(distance_workers);
    server->request_scan_cursor_by_worker_.assign(distance_workers, 0);
    server->result_slots_by_owner_.resize(config.rpc_threads);
    server->result_scan_cursor_by_owner_.assign(config.rpc_threads, 0);
    server->result_controls_ =
        std::make_unique<ResultSlotControl[]>(result_count.value());
    server->remote_query_slots_ =
        std::make_unique<RemoteQuerySlot[]>(query_count.value());
    server->remote_query_vectors_ =
        std::make_unique<float[]>(query_values.value());
    server->worker_profiles_ =
        std::make_unique<AtomicWorkerProfile[]>(distance_workers);
    server->result_ready_ =
        std::make_unique<CountingNotification[]>(config.rpc_threads);
    server->worker_distance_scratch_.resize(distance_workers);
    server->worker_id_scratch_.resize(distance_workers);
    for (std::uint32_t worker = 0; worker < distance_workers; ++worker) {
      server->worker_distance_scratch_[worker].resize(max_ids_per_task);
      server->worker_id_scratch_[worker].resize(max_ids_per_task);
    }
    const auto remote_peer_count = config.world_size - 1U;
    const bool has_disjoint_peer_groups =
        distance_workers >= remote_peer_count;
    const auto workers_per_peer = has_disjoint_peer_groups
        ? distance_workers / remote_peer_count
        : 1U;
    const auto peers_with_extra_worker = has_disjoint_peer_groups
        ? distance_workers % remote_peer_count
        : 0U;
    std::uint32_t peer_ordinal = 0;
    for (std::uint32_t peer = 0; peer < config.world_size; ++peer) {
      if (peer == local_rank) continue;
      const auto owner = rpc_owner_by_rank[peer];

      // Keep each peer's request ring on a stable Distance Worker group so
      // repeated work from the same source retains query/cache affinity. When
      // there are enough workers, split them as evenly as possible into
      // disjoint peer groups and stripe consecutive slots within each group.
      // With fewer workers than peers, each peer remains affine to one worker
      // and workers serve multiple peers.
      const auto worker_count = workers_per_peer +
          (peer_ordinal < peers_with_extra_worker ? 1U : 0U);
      const auto worker_begin = has_disjoint_peer_groups
          ? peer_ordinal * workers_per_peer +
                std::min(peer_ordinal, peers_with_extra_worker)
          : peer_ordinal % distance_workers;
      for (std::uint32_t slot = 0; slot < config.ring_depth; ++slot) {
        const RpcSlotId id{peer, slot};
        const auto worker = worker_begin + slot % worker_count;
        server->request_slots_by_worker_[worker].push_back(id);
        server->result_slots_by_owner_[owner].push_back(id);
        server->result_controls_[server->SlotIndex(id)].owner_worker = worker;
      }
      ++peer_ordinal;
    }
    server->profiling_enabled_ = profiling_enabled;
    return server;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate direct-slot server");
  }
}

std::size_t RpcServerRuntime::SlotIndex(RpcSlotId id) const noexcept {
  return static_cast<std::size_t>(id.peer_rank) * ring_depth_ + id.slot_id;
}

std::size_t RpcServerRuntime::QueryIndex(
    std::uint32_t source_rank, std::uint32_t query_slot_id) const noexcept {
  return static_cast<std::size_t>(source_rank) * max_active_queries_ + query_slot_id;
}

Status RpcServerRuntime::ComputeTask(
    Metric metric, std::span<const float> query, float query_norm,
    std::span<const node_id_t> ids,
    std::span<float> distances) const noexcept {
  if (query.size() != dim_ || ids.size() != distances.size()) {
    return Status(StatusCode::kProtocolError,
                  "remote distance task shape is invalid");
  }
  const auto global_begin = local_shard_->range().global_begin;
  const auto local_count = local_shard_->range().local_count;
  std::size_t index = 0;
  for (; index + 4 <= ids.size(); index += 4) {
    std::array<const float*, 4> rows{};
    std::array<std::uint64_t, 4> offsets{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
      const auto id = ids[index + row];
      if (id < global_begin ||
          static_cast<std::uint64_t>(id - global_begin) >= local_count) {
        return Status(StatusCode::kOutOfRange,
                      "distance ID is outside the local shard");
      }
      offsets[row] = id - global_begin;
      local_shard_->PrefetchVector(offsets[row]);
      rows[row] = local_shard_->TrustedVector(offsets[row]).data();
    }
    Status status;
    if (metric == Metric::kL2) {
      std::array<float, 4> norms{};
      for (std::size_t row = 0; row < norms.size(); ++row) {
        norms[row] = local_shard_->TrustedVectorNorm(offsets[row]);
      }
      status = ComputeL2DotNormBatch4(
          query, query_norm, rows, norms,
          distances.subspan(index, rows.size()), DistanceImplementation::kAuto);
    } else {
      status = ComputeDistanceBatch4(
          query, rows, metric, distances.subspan(index, rows.size()),
          DistanceImplementation::kAuto);
    }
    if (!status.ok()) return status;
  }
  for (; index < ids.size(); ++index) {
    const auto id = ids[index];
    if (id < global_begin ||
        static_cast<std::uint64_t>(id - global_begin) >= local_count) {
      return Status(StatusCode::kOutOfRange,
                    "distance ID is outside the local shard");
    }
    const auto offset = static_cast<std::uint64_t>(id - global_begin);
    const auto row = local_shard_->TrustedVector(offset);
    distances[index] = metric == Metric::kL2
        ? std::max(0.0F, L2FromDotNorm(
              query_norm, local_shard_->TrustedVectorNorm(offset),
              ScalarDot(query, row)))
        : ComputeDistance(query, row, metric, DistanceImplementation::kAuto);
  }
  return Status::Ok();
}

Result<bool> RpcServerRuntime::PollAndProcessRequestSlot(
    std::uint32_t worker, Metric metric,
    DmaVisibilityDomain& visibility) noexcept {
  if (worker >= distance_workers_) {
    return Status(StatusCode::kInvalidArgument,
                  "distance worker index is invalid");
  }
  auto& slots = request_slots_by_worker_[worker];
  auto& cursor = request_scan_cursor_by_worker_[worker];
  auto& profile = worker_profiles_[worker];
  for (std::size_t visited = 0; visited < slots.size(); ++visited) {
    const auto index = (cursor + visited) % slots.size();
    const auto id = slots[index];
    auto* notify = memory_->NotifyAddress(
        memory_->RequestNotifyOffset(id.peer_rank, id.slot_id));
    auto observed = DmaLoadNotifyAcquire(notify, visibility);
    if (!observed.ok()) return observed.status();
    if (observed.value() == 0) continue;
    const auto hit_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
    profile.request_slot_hits.fetch_add(1, std::memory_order_relaxed);
    auto& control = result_controls_[SlotIndex(id)];
    auto state = control.state.load(std::memory_order_acquire);
    if (state == ResultSlotState::kInFlight) {
      profile.request_slot_blocked_by_result_inflight.fetch_add(
          1, std::memory_order_relaxed);
      continue;
    }
    if (state != ResultSlotState::kFree) {
      return Status(StatusCode::kProtocolError,
                    "request arrived while its result slot is active");
    }
    if (!control.state.compare_exchange_strong(
            state, ResultSlotState::kComputing,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return Status(StatusCode::kProtocolError,
                    "request result slot claim raced unexpectedly");
    }
    cursor = (index + 1U) % slots.size();
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile std::uint64_t*>(notify) = 0;
    if (profiling_enabled_) {
      control.request_notify_observed_nanoseconds = hit_at;
      const auto claimed_at = SteadyNowNanoseconds();
      profile.request_poll_hit_to_claim_nanoseconds.fetch_add(
          claimed_at - hit_at, std::memory_order_relaxed);
      profile.request_poll_hit_to_claim_samples.fetch_add(
          1, std::memory_order_relaxed);
    }
    const auto decode_started_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
    auto request = DecodeDistanceBatchRequest(memory_->RequestSlot(
        id.peer_rank, id.slot_id));
    if (!request.ok()) return request.status();
    if (request.value().header().batch_sequence != observed.value()) {
      return Status(StatusCode::kProtocolError,
                    "request notify and batch sequence disagree");
    }
    if (profiling_enabled_) {
      profile.request_decode_nanoseconds.fetch_add(
          SteadyNowNanoseconds() - decode_started_at,
          std::memory_order_relaxed);
    }
    std::uint64_t total_ids = 0;
    for (std::size_t task_index = 0;
         task_index < request.value().task_count(); ++task_index) {
      auto task = request.value().Task(task_index);
      if (!task.ok()) return task.status();
      total_ids += task.value().ids.size();
      if (task.value().descriptor.owner_shard >= shard_to_rank_.size() ||
          shard_to_rank_[task.value().descriptor.owner_shard] != local_rank_ ||
          task.value().descriptor.query_slot_id >= max_active_queries_ ||
          task.value().descriptor.dim != dim_ ||
          task.value().ids.size() > max_ids_per_task_) {
        return Status(StatusCode::kProtocolError,
                      "request task routing or shape is invalid");
      }
    }
    auto maximum_response = DistanceBatchResponseBytesForShape(
        static_cast<std::uint32_t>(request.value().task_count()), total_ids);
    if (!maximum_response.ok() ||
        maximum_response.value() > memory_->config().response_capacity) {
      return maximum_response.ok()
          ? Status(StatusCode::kResourceExhausted,
                   "request worst-case response exceeds result slot")
          : maximum_response.status();
    }
    auto writer = DistanceBatchResponseWriter::Create(
        observed.value(), static_cast<std::uint32_t>(request.value().task_count()),
        memory_->ResultSlot(id.peer_rank, id.slot_id));
    if (!writer.ok()) return writer.status();
    for (std::size_t task_index = 0;
         task_index < request.value().task_count(); ++task_index) {
      auto task = request.value().Task(task_index);
      if (!task.ok()) return task.status();
      const auto query_index = QueryIndex(
          id.peer_rank, task.value().descriptor.query_slot_id);
      auto& query_slot = remote_query_slots_[query_index];
      const auto query_started_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
      if (query_slot.local_access.test_and_set(std::memory_order_acquire)) {
        return Status(StatusCode::kProtocolError,
                      "remote query slot has concurrent access");
      }
      auto query = std::span<float>(
          remote_query_vectors_.get() + query_index * dim_, dim_);
      const bool stores_query =
          (task.value().descriptor.flags & kBatchHasQueryVector) != 0;
      if (stores_query) {
        std::copy(task.value().query_vector.begin(),
                  task.value().query_vector.end(), query.begin());
        query_slot.norm = ScalarDot(query, query);
      }
      if (profiling_enabled_) {
        const auto elapsed = SteadyNowNanoseconds() - query_started_at;
        (stores_query ? profile.query_slot_store_nanoseconds
                      : profile.query_slot_load_nanoseconds)
            .fetch_add(elapsed, std::memory_order_relaxed);
      }
      auto distances = std::span<float>(worker_distance_scratch_[worker])
          .first(task.value().ids.size());
      const auto compute_started_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
      auto computed = ComputeTask(metric, query, query_slot.norm,
                                  task.value().ids, distances);
      query_slot.local_access.clear(std::memory_order_release);
      if (!computed.ok()) return computed;
      if (profiling_enabled_) {
        profile.distance_compute_nanoseconds.fetch_add(
            SteadyNowNanoseconds() - compute_started_at,
            std::memory_order_relaxed);
      }
      const auto encode_started_at = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
      auto& survivor_ids = worker_id_scratch_[worker];
      std::size_t survivor_count = 0;
      if ((task.value().descriptor.flags & kBatchHasUpperDistance) != 0) {
        for (std::size_t value = 0; value < distances.size(); ++value) {
          if (distances[value] <= task.value().descriptor.upper_distance) {
            survivor_ids[survivor_count] = task.value().ids[value];
            distances[survivor_count] = distances[value];
            ++survivor_count;
          }
        }
      } else {
        survivor_count = task.value().ids.size();
        std::copy(task.value().ids.begin(), task.value().ids.end(),
                  survivor_ids.begin());
      }
      auto appended = writer.value().Append(
          task.value().descriptor.task_ticket,
          task.value().descriptor.query_slot_id,
          std::span<const node_id_t>(survivor_ids).first(survivor_count),
          std::span<const float>(distances).first(survivor_count));
      if (!appended.ok()) return appended;
      if (profiling_enabled_) {
        profile.filter_and_response_encode_nanoseconds.fetch_add(
            SteadyNowNanoseconds() - encode_started_at,
            std::memory_order_relaxed);
      }
    }
    auto response_bytes = writer.value().Finish();
    if (!response_bytes.ok()) return response_bytes.status();
    control.request_sequence = observed.value();
    control.response_bytes = static_cast<std::uint32_t>(response_bytes.value());
    control.task_count = static_cast<std::uint32_t>(request.value().task_count());
    control.ready_nanoseconds = profiling_enabled_ ? SteadyNowNanoseconds() : 0;
    control.state.store(ResultSlotState::kReady, std::memory_order_release);
    const auto owner = rpc_owner_by_rank_[id.peer_rank];
    if (result_adaptive_notifications_enabled_) result_ready_[owner].Notify();
    profile.request_batches.fetch_add(1, std::memory_order_relaxed);
    profile.request_tasks.fetch_add(request.value().task_count(),
                                    std::memory_order_relaxed);
    profile.distance_values.fetch_add(total_ids, std::memory_order_relaxed);
    profile.result_ready_publications.fetch_add(1, std::memory_order_relaxed);
    profile.busy_iterations.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  return false;
}

Result<std::optional<ReadyResultSlot>> RpcServerRuntime::TryGetReadyResult(
    std::uint32_t rpc_owner) noexcept {
  if (rpc_owner >= rpc_threads_) {
    return Status(StatusCode::kInvalidArgument, "RPC owner index is invalid");
  }
  auto& slots = result_slots_by_owner_[rpc_owner];
  auto& cursor = result_scan_cursor_by_owner_[rpc_owner];
  for (std::size_t visited = 0; visited < slots.size(); ++visited) {
    const auto index = (cursor + visited) % slots.size();
    const auto id = slots[index];
    auto& control = result_controls_[SlotIndex(id)];
    if (control.state.load(std::memory_order_acquire) != ResultSlotState::kReady) {
      continue;
    }
    if (control.request_sequence == 0 || control.response_bytes == 0 ||
        control.response_bytes > memory_->config().response_capacity ||
        control.task_count == 0) {
      return Status(StatusCode::kProtocolError,
                    "ready result control is invalid");
    }
    cursor = (index + 1U) % slots.size();
    return std::optional<ReadyResultSlot>(ReadyResultSlot{
        id, control.request_sequence, control.task_count,
        control.response_bytes, control.request_notify_observed_nanoseconds,
        control.ready_nanoseconds,
        memory_->ResultSlot(id.peer_rank, id.slot_id)
            .first(control.response_bytes)});
  }
  return std::optional<ReadyResultSlot>{};
}

Status RpcServerRuntime::MarkResultPosted(
    RpcSlotId id, std::uint64_t sequence) noexcept {
  if (id.peer_rank >= world_size_ || id.slot_id >= ring_depth_ ||
      id.peer_rank == local_rank_) {
    return Status(StatusCode::kInvalidArgument, "result slot is invalid");
  }
  auto& control = result_controls_[SlotIndex(id)];
  if (control.request_sequence != sequence) {
    return Status(StatusCode::kProtocolError,
                  "result sequence changed before post");
  }
  auto expected = ResultSlotState::kReady;
  if (!control.state.compare_exchange_strong(
          expected, ResultSlotState::kInFlight,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError, "result slot is not ready");
  }
  return Status::Ok();
}

Status RpcServerRuntime::AbortResultPost(
    RpcSlotId id, std::uint64_t sequence) noexcept {
  if (id.peer_rank >= world_size_ || id.slot_id >= ring_depth_ ||
      id.peer_rank == local_rank_) {
    return Status(StatusCode::kInvalidArgument, "result slot is invalid");
  }
  auto& control = result_controls_[SlotIndex(id)];
  if (control.request_sequence != sequence) {
    return Status(StatusCode::kProtocolError,
                  "result sequence changed before post rollback");
  }
  auto expected = ResultSlotState::kInFlight;
  if (!control.state.compare_exchange_strong(
          expected, ResultSlotState::kReady,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    return Status(StatusCode::kProtocolError,
                  "result slot is not in flight for rollback");
  }
  return Status::Ok();
}

Status RpcServerRuntime::CompleteResultSend(
    RpcSlotId id, std::uint64_t sequence) noexcept {
  if (id.peer_rank >= world_size_ || id.slot_id >= ring_depth_ ||
      id.peer_rank == local_rank_) {
    return Status(StatusCode::kInvalidArgument, "result slot is invalid");
  }
  auto& control = result_controls_[SlotIndex(id)];
  if (control.state.load(std::memory_order_acquire) != ResultSlotState::kInFlight) {
    return Status(StatusCode::kProtocolError,
                  "result slot is not in flight at completion");
  }
  if (control.request_sequence != sequence) {
    return Status(StatusCode::kProtocolError,
                  "result sequence changed before completion");
  }
  control.request_sequence = 0;
  control.response_bytes = 0;
  control.task_count = 0;
  if (profiling_enabled_) control.request_notify_observed_nanoseconds = 0;
  control.ready_nanoseconds = 0;
  // This RPC owner alone releases InFlight; the distance worker may claim Free.
  control.state.store(ResultSlotState::kFree, std::memory_order_release);
  return Status::Ok();
}

CountingNotification& RpcServerRuntime::result_ready(
    std::uint32_t rpc_owner) noexcept {
  return result_ready_[rpc_owner];
}

bool RpcServerRuntime::drained(std::uint32_t rpc_owner) const noexcept {
  if (rpc_owner >= rpc_threads_) return false;
  for (const auto id : result_slots_by_owner_[rpc_owner]) {
    if (result_controls_[SlotIndex(id)].state.load(std::memory_order_acquire) !=
        ResultSlotState::kFree) return false;
  }
  return true;
}

DistanceWorkerProfile RpcServerRuntime::worker_profile(
    std::uint32_t worker) const noexcept {
  if (worker >= distance_workers_) return {};
  const auto& source = worker_profiles_[worker];
  DistanceWorkerProfile result;
#define LOAD_PROFILE(field) result.field = source.field.load(std::memory_order_relaxed)
  LOAD_PROFILE(request_slot_scans);
  LOAD_PROFILE(request_slot_hits);
  LOAD_PROFILE(request_slot_blocked_by_result_inflight);
  LOAD_PROFILE(request_batches);
  LOAD_PROFILE(request_tasks);
  LOAD_PROFILE(distance_values);
  LOAD_PROFILE(request_poll_hit_to_claim_nanoseconds);
  LOAD_PROFILE(request_poll_hit_to_claim_samples);
  LOAD_PROFILE(request_decode_nanoseconds);
  LOAD_PROFILE(query_slot_store_nanoseconds);
  LOAD_PROFILE(query_slot_load_nanoseconds);
  LOAD_PROFILE(distance_compute_nanoseconds);
  LOAD_PROFILE(filter_and_response_encode_nanoseconds);
  LOAD_PROFILE(result_ready_publications);
  LOAD_PROFILE(busy_iterations);
  LOAD_PROFILE(idle_iterations);
#undef LOAD_PROFILE
  return result;
}

std::span<const RpcSlotId> RpcServerRuntime::worker_slots(
    std::uint32_t worker) const noexcept {
  return worker < request_slots_by_worker_.size()
      ? std::span<const RpcSlotId>(request_slots_by_worker_[worker])
      : std::span<const RpcSlotId>{};
}

ResultSlotState RpcServerRuntime::result_state(RpcSlotId id) const noexcept {
  if (id.peer_rank >= world_size_ || id.slot_id >= ring_depth_) {
    return ResultSlotState::kInFlight;
  }
  return result_controls_[SlotIndex(id)].state.load(std::memory_order_acquire);
}

}  // namespace hnsw_urma
