#include "hnsw_urma/runtime/rpc/mailbox.hpp"

#include <initializer_list>
#include <limits>
#include <new>
#include <utility>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
namespace {

Result<std::uint64_t> Product(std::initializer_list<std::uint64_t> values) {
  std::uint64_t result = 1;
  for (const auto value : values) {
    auto next = CheckedMultiply(result, value);
    if (!next.ok()) return next.status();
    result = next.value();
  }
  return result;
}

Status AppendSection(std::uint64_t bytes, std::uint64_t& cursor,
                     std::uint64_t& section_offset) {
  auto aligned = CheckedAlignUp<std::uint64_t>(cursor, 64);
  if (!aligned.ok()) return aligned.status();
  section_offset = aligned.value();
  auto next = CheckedAdd(section_offset, bytes);
  if (!next.ok()) return next.status();
  cursor = next.value();
  return Status::Ok();
}

Result<std::uint64_t> FinishSegment(std::uint64_t cursor) {
  return CheckedAlignUp<std::uint64_t>(cursor, 64);
}

}  // namespace

Result<std::unique_ptr<RpcMailboxMemory>> RpcMailboxMemory::Create(
    RpcMailboxConfig config, bool shared) {
  if (config.world_size == 0 || config.rpc_threads == 0 ||
      config.ring_depth == 0 ||
      (config.ring_depth & (config.ring_depth - 1U)) != 0 ||
      config.request_capacity == 0 || config.response_capacity == 0 ||
      config.request_staging_slots_per_thread == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC mailbox dimensions are invalid");
  }
  auto request_stride = CheckedAlignUp<std::uint64_t>(config.request_capacity, 64);
  auto response_stride = CheckedAlignUp<std::uint64_t>(config.response_capacity, 64);
  auto cells = Product({config.world_size, config.ring_depth});
  auto request_bytes = request_stride.ok() && cells.ok()
      ? CheckedMultiply(request_stride.value(), cells.value())
      : Result<std::uint64_t>(request_stride.ok() ? cells.status()
                                                  : request_stride.status());
  auto response_bytes = response_stride.ok() && cells.ok()
      ? CheckedMultiply(response_stride.value(), cells.value())
      : Result<std::uint64_t>(response_stride.ok() ? cells.status()
                                                   : response_stride.status());
  auto notify_bytes = cells.ok()
      ? CheckedMultiply<std::uint64_t>(cells.value(), 64)
      : Result<std::uint64_t>(cells.status());
  auto staging_cells = Product(
      {config.rpc_threads, config.request_staging_slots_per_thread});
  auto staging_bytes = staging_cells.ok() && request_stride.ok()
      ? CheckedMultiply(staging_cells.value(), request_stride.value())
      : Result<std::uint64_t>(staging_cells.ok() ? request_stride.status()
                                                : staging_cells.status());
  if (!request_bytes.ok() || !response_bytes.ok() || !notify_bytes.ok() ||
      !staging_bytes.ok()) {
    return Status(StatusCode::kOutOfRange,
                  "RPC mailbox size calculation overflows");
  }

  try {
    auto memory = std::unique_ptr<RpcMailboxMemory>(new RpcMailboxMemory());
    memory->config_ = config;
    memory->request_stride_ = request_stride.value();
    memory->response_stride_ = response_stride.value();

    std::uint64_t request_cursor = 0;
    auto status = AppendSection(request_bytes.value(), request_cursor,
                                memory->request_offset_);
    if (status.ok()) {
      status = AppendSection(staging_bytes.value(), request_cursor,
                             memory->request_staging_offset_);
    }
    auto request_total = status.ok()
        ? FinishSegment(request_cursor)
        : Result<std::uint64_t>(status);

    std::uint64_t response_cursor = 0;
    status = AppendSection(response_bytes.value(), response_cursor,
                           memory->response_offset_);
    if (status.ok()) {
      status = AppendSection(response_bytes.value(), response_cursor,
                             memory->result_offset_);
    }
    auto response_total = status.ok()
        ? FinishSegment(response_cursor)
        : Result<std::uint64_t>(status);

    std::uint64_t notify_cursor = 0;
    status = AppendSection(notify_bytes.value(), notify_cursor,
                           memory->request_notify_offset_);
    if (status.ok()) {
      status = AppendSection(notify_bytes.value(), notify_cursor,
                             memory->response_notify_offset_);
    }
    auto notify_total = status.ok()
        ? FinishSegment(notify_cursor)
        : Result<std::uint64_t>(status);
    if (!request_total.ok() || !response_total.ok() || !notify_total.ok()) {
      return Status(StatusCode::kOutOfRange, "RPC segment layout overflows");
    }
    auto payload_total = CheckedAdd(request_total.value(), response_total.value());
    auto total = payload_total.ok()
        ? CheckedAdd(payload_total.value(), notify_total.value())
        : Result<std::uint64_t>(payload_total.status());
    if (!total.ok() ||
        request_total.value() > std::numeric_limits<std::size_t>::max() ||
        response_total.value() > std::numeric_limits<std::size_t>::max() ||
        notify_total.value() > std::numeric_limits<std::size_t>::max()) {
      return Status(StatusCode::kOutOfRange,
                    "RPC mailbox exceeds addressable memory");
    }
    memory->logical_bytes_ = total.value();
    const auto allocate = [shared](std::size_t bytes) {
      return shared ? PageAlignedBuffer::CreateShared(bytes)
                    : PageAlignedBuffer::Create(bytes);
    };
    auto requests = allocate(static_cast<std::size_t>(request_total.value()));
    if (!requests.ok()) return requests.status();
    auto responses = allocate(static_cast<std::size_t>(response_total.value()));
    if (!responses.ok()) return responses.status();
    auto notify = allocate(static_cast<std::size_t>(notify_total.value()));
    if (!notify.ok()) return notify.status();
    memory->request_storage_ = std::move(requests).value();
    memory->response_storage_ = std::move(responses).value();
    memory->notify_storage_ = std::move(notify).value();
    return memory;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate RPC mailbox");
  }
}

Status RpcMailboxMemory::UnlinkSharedMemory() {
  auto status = request_storage_.UnlinkSharedMemory();
  if (!status.ok()) return status;
  status = response_storage_.UnlinkSharedMemory();
  if (!status.ok()) return status;
  return notify_storage_.UnlinkSharedMemory();
}

segment_offset_t RpcMailboxMemory::RequestOffset(
    std::uint32_t source_rank, std::uint32_t slot_id) const noexcept {
  return request_offset_ +
         (static_cast<std::uint64_t>(source_rank) * config_.ring_depth + slot_id) *
             request_stride_;
}

segment_offset_t RpcMailboxMemory::ResponseOffset(
    std::uint32_t producer_rank, std::uint32_t slot_id) const noexcept {
  return response_offset_ +
         (static_cast<std::uint64_t>(producer_rank) * config_.ring_depth + slot_id) *
             response_stride_;
}

segment_offset_t RpcMailboxMemory::ResultOffset(
    std::uint32_t destination_rank, std::uint32_t slot_id) const noexcept {
  return result_offset_ +
         (static_cast<std::uint64_t>(destination_rank) * config_.ring_depth + slot_id) *
             response_stride_;
}

segment_offset_t RpcMailboxMemory::RequestNotifyOffset(
    std::uint32_t source_rank, std::uint32_t slot_id) const noexcept {
  return request_notify_offset_ +
         (static_cast<std::uint64_t>(source_rank) * config_.ring_depth + slot_id) * 64U;
}

segment_offset_t RpcMailboxMemory::ResponseNotifyOffset(
    std::uint32_t producer_rank, std::uint32_t slot_id) const noexcept {
  return response_notify_offset_ +
         (static_cast<std::uint64_t>(producer_rank) * config_.ring_depth + slot_id) * 64U;
}

std::span<std::byte> RpcMailboxMemory::RequestSlot(
    std::uint32_t source_rank, std::uint32_t slot_id) noexcept {
  if (source_rank >= config_.world_size || slot_id >= config_.ring_depth) return {};
  return requests().subspan(static_cast<std::size_t>(RequestOffset(source_rank, slot_id)),
                            config_.request_capacity);
}

std::span<std::byte> RpcMailboxMemory::ResponseSlot(
    std::uint32_t producer_rank, std::uint32_t slot_id) noexcept {
  if (producer_rank >= config_.world_size || slot_id >= config_.ring_depth) return {};
  return responses().subspan(
      static_cast<std::size_t>(ResponseOffset(producer_rank, slot_id)),
      config_.response_capacity);
}

std::span<std::byte> RpcMailboxMemory::ResultSlot(
    std::uint32_t destination_rank, std::uint32_t slot_id) noexcept {
  if (destination_rank >= config_.world_size || slot_id >= config_.ring_depth) return {};
  return responses().subspan(
      static_cast<std::size_t>(ResultOffset(destination_rank, slot_id)),
      config_.response_capacity);
}

std::span<std::byte> RpcMailboxMemory::RequestStaging(
    std::uint32_t rpc_thread, std::uint32_t slot_id) noexcept {
  if (rpc_thread >= config_.rpc_threads ||
      slot_id >= config_.request_staging_slots_per_thread) return {};
  const auto index = static_cast<std::uint64_t>(rpc_thread) *
                         config_.request_staging_slots_per_thread +
                     slot_id;
  return requests().subspan(
      static_cast<std::size_t>(request_staging_offset_ + index * request_stride_),
      config_.request_capacity);
}

std::span<std::byte> RpcMailboxMemory::RequestStagingRegion(
    std::uint32_t rpc_thread) noexcept {
  if (rpc_thread >= config_.rpc_threads) return {};
  const auto bytes = static_cast<std::uint64_t>(
                         config_.request_staging_slots_per_thread) * request_stride_;
  const auto offset = request_staging_offset_ +
                      static_cast<std::uint64_t>(rpc_thread) * bytes;
  return requests().subspan(static_cast<std::size_t>(offset),
                            static_cast<std::size_t>(bytes));
}

std::uint64_t* RpcMailboxMemory::NotifyAddress(segment_offset_t offset) noexcept {
  if (offset > notify().size() || sizeof(std::uint64_t) > notify().size() - offset) {
    return nullptr;
  }
  return reinterpret_cast<std::uint64_t*>(notify().data() + offset);
}

const std::uint64_t* RpcMailboxMemory::NotifyAddress(
    segment_offset_t offset) const noexcept {
  if (offset > notify().size() || sizeof(std::uint64_t) > notify().size() - offset) {
    return nullptr;
  }
  return reinterpret_cast<const std::uint64_t*>(notify().data() + offset);
}

}  // namespace hnsw_urma
