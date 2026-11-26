/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "hnsw_urma/protocol/distance_batch_wire.hpp"

#include <cstring>
#include <limits>
#include <type_traits>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
namespace {

constexpr std::size_t kWireAlignment = 8;

template <typename Descriptor>
Result<std::size_t> PayloadStart(std::uint64_t task_count) noexcept {
  auto table_bytes =
      CheckedMultiply<std::uint64_t>(task_count, sizeof(Descriptor));
  if (!table_bytes.ok())
    return table_bytes.status();
  auto end = CheckedAdd<std::uint64_t>(sizeof(DistanceBatchHeader),
                                       table_bytes.value());
  if (!end.ok())
    return end.status();
  auto aligned = CheckedAlignUp<std::uint64_t>(end.value(), kWireAlignment);
  if (!aligned.ok() ||
      aligned.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "distance batch descriptor table overflows");
  }
  return static_cast<std::size_t>(aligned.value());
}

Result<std::size_t> AddBytes(std::size_t cursor, std::uint64_t bytes) noexcept {
  auto next = CheckedAdd<std::uint64_t>(cursor, bytes);
  if (!next.ok() || next.value() > std::numeric_limits<std::uint32_t>::max() ||
      next.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "distance batch exceeds wire offset range");
  }
  return static_cast<std::size_t>(next.value());
}

template <typename T>
void Store(std::span<std::byte> output, std::size_t offset,
           const T &value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  std::memcpy(output.data() + offset, &value, sizeof(value));
}

template <typename T> bool Aligned(const T *pointer) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0;
}

Result<DistanceBatchHeader>
DecodeEnvelope(std::span<const std::byte> message, std::uint32_t magic,
               DistanceBatchKind kind, std::size_t descriptor_bytes) noexcept {
  if (message.size() < sizeof(DistanceBatchHeader)) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch header is truncated");
  }
  DistanceBatchHeader header;
  std::memcpy(&header, message.data(), sizeof(header));
  if (header.magic != magic || header.version != DISTANCE_BATCH_WIRE_VERSION ||
      header.kind != kind || header.flags != 0 ||
      header.header_bytes != sizeof(header) ||
      header.task_table_offset != sizeof(header) || header.task_count == 0 ||
      header.batch_sequence == 0 || header.total_bytes > message.size() ||
      header.total_bytes < sizeof(header)) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch envelope is invalid");
  }
  auto table_bytes =
      CheckedMultiply<std::uint64_t>(header.task_count, descriptor_bytes);
  if (!table_bytes.ok())
    return table_bytes.status();
  auto table_end =
      CheckedAdd<std::uint64_t>(header.task_table_offset, table_bytes.value());
  if (!table_end.ok())
    return table_end.status();
  auto payload_offset =
      CheckedAlignUp<std::uint64_t>(table_end.value(), kWireAlignment);
  if (!payload_offset.ok() || header.payload_offset != payload_offset.value() ||
      header.payload_offset > header.total_bytes ||
      header.payload_bytes != header.total_bytes - header.payload_offset) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch table or payload bounds are invalid");
  }
  return header;
}

template <typename T>
Result<std::span<const T>> ViewArray(std::span<const std::byte> message,
                                     std::uint32_t offset,
                                     std::uint32_t count) noexcept {
  auto bytes = CheckedMultiply<std::uint64_t>(count, sizeof(T));
  if (!bytes.ok() || offset > message.size() ||
      bytes.value() > message.size() - offset) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch descriptor array is truncated");
  }
  const auto *data = reinterpret_cast<const T *>(message.data() + offset);
  if (!Aligned(data)) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch descriptor array is misaligned");
  }
  return std::span<const T>(data, count);
}

template <typename T>
Result<std::span<const T>>
ViewPayload(std::span<const std::byte> message, std::uint32_t offset,
            std::uint32_t count, std::uint32_t payload_begin,
            std::uint32_t payload_end) noexcept {
  auto bytes = CheckedMultiply<std::uint64_t>(count, sizeof(T));
  if (!bytes.ok() || offset < payload_begin || offset > payload_end ||
      bytes.value() > payload_end - offset || payload_end > message.size()) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch task payload is invalid");
  }
  const auto *data = reinterpret_cast<const T *>(message.data() + offset);
  if (!Aligned(data)) {
    return Status(StatusCode::kInvalidFormat,
                  "distance batch task payload is misaligned");
  }
  return std::span<const T>(data, count);
}

Status ValidateRequestInput(const DistanceBatchRequestTaskInput &task) {
  const bool has_query = (task.flags & kBatchHasQueryVector) != 0;
  if (task.task_ticket == 0 || task.ids.empty() || task.dim == 0 ||
      (task.flags & ~DISTANCE_BATCH_REQUEST_KNOWN_FLAGS) != 0 ||
      (has_query && task.query_vector.size() != task.dim) ||
      (!has_query && !task.query_vector.empty())) {
    return Status(StatusCode::kInvalidArgument,
                  "distance request task is invalid");
  }
  return Status::Ok();
}

Status ValidateResponseInput(const DistanceBatchResponseTaskInput &task) {
  if (task.task_ticket == 0 || task.ids.size() != task.distances.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "distance response task is invalid");
  }
  return Status::Ok();
}

} // namespace

Result<std::size_t> DistanceBatchRequestBytesForShape(
    std::uint32_t task_count, std::uint64_t total_id_count,
    std::uint64_t total_query_value_count) noexcept {
  if (task_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "distance request batch must contain a task");
  }
  auto cursor = PayloadStart<DistanceBatchRequestTask>(task_count);
  if (!cursor.ok())
    return cursor.status();
  auto ids = CheckedMultiply<std::uint64_t>(total_id_count, sizeof(node_id_t));
  auto queries =
      CheckedMultiply<std::uint64_t>(total_query_value_count, sizeof(float));
  if (!ids.ok() || !queries.ok()) {
    return Status(StatusCode::kOutOfRange,
                  "distance request payload shape overflows");
  }
  auto with_ids = AddBytes(cursor.value(), ids.value());
  if (!with_ids.ok())
    return with_ids.status();
  return AddBytes(with_ids.value(), queries.value());
}

Result<std::size_t> EncodedDistanceBatchRequestBytes(
    std::span<const DistanceBatchRequestTaskInput> tasks) noexcept {
  std::uint64_t ids = 0;
  std::uint64_t query_values = 0;
  for (const auto &task : tasks) {
    auto status = ValidateRequestInput(task);
    if (!status.ok())
      return status;
    auto next_ids = CheckedAdd<std::uint64_t>(ids, task.ids.size());
    auto next_queries =
        CheckedAdd<std::uint64_t>(query_values, task.query_vector.size());
    if (!next_ids.ok() || !next_queries.ok()) {
      return Status(StatusCode::kOutOfRange,
                    "distance request payload shape overflows");
    }
    ids = next_ids.value();
    query_values = next_queries.value();
  }
  if (tasks.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "distance request task count overflows");
  }
  return DistanceBatchRequestBytesForShape(
      static_cast<std::uint32_t>(tasks.size()), ids, query_values);
}

Result<std::size_t>
EncodeDistanceBatchRequest(std::uint64_t batch_sequence,
                           std::span<const DistanceBatchRequestTaskInput> tasks,
                           std::span<std::byte> output) noexcept {
  if (batch_sequence == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "distance request sequence must be non-zero");
  }
  auto required = EncodedDistanceBatchRequestBytes(tasks);
  if (!required.ok())
    return required.status();
  if (required.value() > output.size()) {
    return Status(StatusCode::kResourceExhausted,
                  "distance request output is too small");
  }
  auto payload_start = PayloadStart<DistanceBatchRequestTask>(tasks.size());
  if (!payload_start.ok())
    return payload_start.status();
  std::size_t cursor = payload_start.value();
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    const auto &input = tasks[index];
    DistanceBatchRequestTask descriptor;
    descriptor.task_ticket = input.task_ticket;
    descriptor.query_slot_id = input.query_slot_id;
    descriptor.owner_shard = input.owner_shard;
    descriptor.count = static_cast<std::uint32_t>(input.ids.size());
    descriptor.dim = input.dim;
    descriptor.flags = input.flags;
    descriptor.ids_offset = static_cast<std::uint32_t>(cursor);
    std::memcpy(output.data() + cursor, input.ids.data(),
                input.ids.size_bytes());
    cursor += input.ids.size_bytes();
    if (!input.query_vector.empty()) {
      descriptor.query_offset = static_cast<std::uint32_t>(cursor);
      std::memcpy(output.data() + cursor, input.query_vector.data(),
                  input.query_vector.size_bytes());
      cursor += input.query_vector.size_bytes();
    }
    descriptor.payload_end = static_cast<std::uint32_t>(cursor);
    descriptor.upper_distance = input.upper_distance;
    Store(output,
          sizeof(DistanceBatchHeader) +
              index * sizeof(DistanceBatchRequestTask),
          descriptor);
  }
  DistanceBatchHeader header;
  header.magic = DISTANCE_BATCH_REQUEST_MAGIC;
  header.kind = DistanceBatchKind::kRequest;
  header.task_count = static_cast<std::uint32_t>(tasks.size());
  header.batch_sequence = batch_sequence;
  header.payload_offset = static_cast<std::uint32_t>(payload_start.value());
  header.payload_bytes =
      static_cast<std::uint32_t>(cursor - payload_start.value());
  header.total_bytes = static_cast<std::uint32_t>(cursor);
  Store(output, 0, header);
  return cursor;
}

Result<DistanceBatchHeader> DecodeDistanceBatchRequestEnvelope(
    std::span<const std::byte> message) noexcept {
  return DecodeEnvelope(message, DISTANCE_BATCH_REQUEST_MAGIC,
                        DistanceBatchKind::kRequest,
                        sizeof(DistanceBatchRequestTask));
}

Result<DistanceBatchRequestView>
DecodeDistanceBatchRequest(std::span<const std::byte> message) noexcept {
  auto envelope = DecodeDistanceBatchRequestEnvelope(message);
  if (!envelope.ok())
    return envelope.status();
  const auto &header = envelope.value();
  auto descriptors = ViewArray<DistanceBatchRequestTask>(
      message, header.task_table_offset, header.task_count);
  if (!descriptors.ok())
    return descriptors.status();
  std::uint32_t expected_offset = header.payload_offset;
  for (const auto &task : descriptors.value()) {
    if (task.task_ticket == 0 || task.count == 0 || task.dim == 0 ||
        task.reserved != 0 ||
        (task.flags & ~DISTANCE_BATCH_REQUEST_KNOWN_FLAGS) != 0 ||
        task.ids_offset != expected_offset) {
      return Status(StatusCode::kInvalidFormat,
                    "distance request descriptor is invalid");
    }
    auto ids_end = CheckedAdd<std::uint64_t>(
        task.ids_offset,
        static_cast<std::uint64_t>(task.count) * sizeof(node_id_t));
    if (!ids_end.ok() || ids_end.value() > task.payload_end) {
      return Status(StatusCode::kInvalidFormat,
                    "distance request IDs are invalid");
    }
    const bool has_query = (task.flags & kBatchHasQueryVector) != 0;
    const auto expected_query = static_cast<std::uint32_t>(ids_end.value());
    const auto query_bytes =
        static_cast<std::uint64_t>(task.dim) * sizeof(float);
    if ((has_query && (task.query_offset != expected_query ||
                       query_bytes > task.payload_end - task.query_offset)) ||
        (!has_query && task.query_offset != 0) ||
        task.payload_end > header.total_bytes ||
        (has_query && task.query_offset + query_bytes != task.payload_end) ||
        (!has_query && ids_end.value() != task.payload_end)) {
      return Status(StatusCode::kInvalidFormat,
                    "distance request query payload is invalid");
    }
    expected_offset = task.payload_end;
  }
  if (expected_offset != header.total_bytes) {
    return Status(StatusCode::kInvalidFormat,
                  "distance request payload is not canonical");
  }
  DistanceBatchRequestView view;
  view.header_ = header;
  view.descriptors_ = descriptors.value();
  view.message_ = message.first(header.total_bytes);
  return view;
}

Result<DistanceBatchRequestTaskView>
DistanceBatchRequestView::Task(std::size_t index) const noexcept {
  if (index >= descriptors_.size()) {
    return Status(StatusCode::kOutOfRange,
                  "distance request task index is out of range");
  }
  const auto descriptor = descriptors_[index];
  auto ids =
      ViewPayload<node_id_t>(message_, descriptor.ids_offset, descriptor.count,
                             header_.payload_offset, descriptor.payload_end);
  if (!ids.ok())
    return ids.status();
  std::span<const float> query;
  if ((descriptor.flags & kBatchHasQueryVector) != 0) {
    auto values =
        ViewPayload<float>(message_, descriptor.query_offset, descriptor.dim,
                           header_.payload_offset, descriptor.payload_end);
    if (!values.ok())
      return values.status();
    query = values.value();
  }
  return DistanceBatchRequestTaskView{descriptor, ids.value(), query};
}

Result<std::size_t> DistanceBatchResponseBytesForShape(
    std::uint32_t task_count, std::uint64_t total_distance_count) noexcept {
  if (task_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "distance response batch must contain a task");
  }
  auto cursor = PayloadStart<DistanceBatchResponseTask>(task_count);
  if (!cursor.ok())
    return cursor.status();
  auto values = CheckedMultiply<std::uint64_t>(
      total_distance_count, sizeof(node_id_t) + sizeof(float));
  if (!values.ok())
    return values.status();
  return AddBytes(cursor.value(), values.value());
}

Result<std::size_t> EncodedDistanceBatchResponseBytes(
    std::span<const DistanceBatchResponseTaskInput> tasks) noexcept {
  std::uint64_t count = 0;
  for (const auto &task : tasks) {
    auto status = ValidateResponseInput(task);
    if (!status.ok())
      return status;
    auto next = CheckedAdd<std::uint64_t>(count, task.ids.size());
    if (!next.ok())
      return next.status();
    count = next.value();
  }
  if (tasks.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "distance response task count overflows");
  }
  return DistanceBatchResponseBytesForShape(
      static_cast<std::uint32_t>(tasks.size()), count);
}

Result<DistanceBatchResponseWriter>
DistanceBatchResponseWriter::Create(std::uint64_t batch_sequence,
                                    std::uint32_t task_count,
                                    std::span<std::byte> output) noexcept {
  if (batch_sequence == 0 || task_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "distance response envelope is invalid");
  }
  auto payload = PayloadStart<DistanceBatchResponseTask>(task_count);
  if (!payload.ok())
    return payload.status();
  if (payload.value() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(
        StatusCode::kOutOfRange,
        "distance response descriptor table exceeds wire offset range");
  }
  if (payload.value() > output.size()) {
    return Status(StatusCode::kResourceExhausted,
                  "distance response descriptor table does not fit");
  }
  DistanceBatchResponseWriter writer;
  writer.output_ = output;
  writer.payload_cursor_ = payload.value();
  writer.header_.magic = DISTANCE_BATCH_RESPONSE_MAGIC;
  writer.header_.kind = DistanceBatchKind::kResponse;
  writer.header_.task_count = task_count;
  writer.header_.batch_sequence = batch_sequence;
  writer.header_.payload_offset = static_cast<std::uint32_t>(payload.value());
  return writer;
}

Status DistanceBatchResponseWriter::Append(
    ticket_t task_ticket, std::uint32_t query_slot_id,
    std::span<const node_id_t> ids, std::span<const float> distances) noexcept {
  if (next_task_ >= header_.task_count || task_ticket == 0 ||
      ids.size() != distances.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "distance response task append is invalid");
  }
  auto ids_end = AddBytes(payload_cursor_, ids.size_bytes());
  if (!ids_end.ok())
    return ids_end.status();
  auto distances_end = AddBytes(ids_end.value(), distances.size_bytes());
  if (!distances_end.ok())
    return distances_end.status();
  if (distances_end.value() > output_.size()) {
    return Status(StatusCode::kResourceExhausted,
                  "distance response result slot is too small");
  }
  DistanceBatchResponseTask descriptor;
  descriptor.task_ticket = task_ticket;
  descriptor.query_slot_id = query_slot_id;
  descriptor.count = static_cast<std::uint32_t>(ids.size());
  descriptor.ids_offset = static_cast<std::uint32_t>(payload_cursor_);
  descriptor.distances_offset = static_cast<std::uint32_t>(ids_end.value());
  descriptor.payload_end = static_cast<std::uint32_t>(distances_end.value());
  if (!ids.empty()) {
    std::memcpy(output_.data() + payload_cursor_, ids.data(), ids.size_bytes());
    std::memcpy(output_.data() + ids_end.value(), distances.data(),
                distances.size_bytes());
  }
  Store(output_,
        sizeof(DistanceBatchHeader) +
            next_task_ * sizeof(DistanceBatchResponseTask),
        descriptor);
  payload_cursor_ = distances_end.value();
  ++next_task_;
  return Status::Ok();
}

Result<std::size_t> DistanceBatchResponseWriter::Finish() noexcept {
  if (next_task_ != header_.task_count) {
    return Status(StatusCode::kInvalidArgument,
                  "distance response batch is incomplete");
  }
  header_.payload_bytes =
      static_cast<std::uint32_t>(payload_cursor_ - header_.payload_offset);
  header_.total_bytes = static_cast<std::uint32_t>(payload_cursor_);
  Store(output_, 0, header_);
  return payload_cursor_;
}

Result<std::size_t> EncodeDistanceBatchResponse(
    std::uint64_t batch_sequence,
    std::span<const DistanceBatchResponseTaskInput> tasks,
    std::span<std::byte> output) noexcept {
  auto required = EncodedDistanceBatchResponseBytes(tasks);
  if (!required.ok())
    return required.status();
  if (required.value() > output.size()) {
    return Status(StatusCode::kResourceExhausted,
                  "distance response output is too small");
  }
  auto writer = DistanceBatchResponseWriter::Create(
      batch_sequence, static_cast<std::uint32_t>(tasks.size()), output);
  if (!writer.ok())
    return writer.status();
  for (const auto &task : tasks) {
    auto status = writer.value().Append(task.task_ticket, task.query_slot_id,
                                        task.ids, task.distances);
    if (!status.ok())
      return status;
  }
  return writer.value().Finish();
}

Result<DistanceBatchResponseView>
DecodeDistanceBatchResponse(std::span<const std::byte> message) noexcept {
  auto envelope = DecodeEnvelope(message, DISTANCE_BATCH_RESPONSE_MAGIC,
                                 DistanceBatchKind::kResponse,
                                 sizeof(DistanceBatchResponseTask));
  if (!envelope.ok())
    return envelope.status();
  const auto &header = envelope.value();
  auto descriptors = ViewArray<DistanceBatchResponseTask>(
      message, header.task_table_offset, header.task_count);
  if (!descriptors.ok())
    return descriptors.status();
  std::uint32_t expected_offset = header.payload_offset;
  for (const auto &task : descriptors.value()) {
    if (task.task_ticket == 0 || task.reserved != 0 ||
        task.ids_offset != expected_offset) {
      return Status(StatusCode::kInvalidFormat,
                    "distance response descriptor is invalid");
    }
    auto ids_end = CheckedAdd<std::uint64_t>(
        task.ids_offset,
        static_cast<std::uint64_t>(task.count) * sizeof(node_id_t));
    auto distances_end =
        ids_end.ok()
            ? CheckedAdd<std::uint64_t>(ids_end.value(),
                                        static_cast<std::uint64_t>(task.count) *
                                            sizeof(float))
            : Result<std::uint64_t>(ids_end.status());
    if (!distances_end.ok() || task.distances_offset != ids_end.value() ||
        task.payload_end != distances_end.value() ||
        task.payload_end > header.total_bytes) {
      return Status(StatusCode::kInvalidFormat,
                    "distance response payload is invalid");
    }
    expected_offset = task.payload_end;
  }
  if (expected_offset != header.total_bytes) {
    return Status(StatusCode::kInvalidFormat,
                  "distance response payload is not canonical");
  }
  DistanceBatchResponseView view;
  view.header_ = header;
  view.descriptors_ = descriptors.value();
  view.message_ = message.first(header.total_bytes);
  return view;
}

Result<DistanceBatchResponseTaskView>
DistanceBatchResponseView::Task(std::size_t index) const noexcept {
  if (index >= descriptors_.size()) {
    return Status(StatusCode::kOutOfRange,
                  "distance response task index is out of range");
  }
  const auto descriptor = descriptors_[index];
  auto ids =
      ViewPayload<node_id_t>(message_, descriptor.ids_offset, descriptor.count,
                             header_.payload_offset, descriptor.payload_end);
  if (!ids.ok())
    return ids.status();
  auto distances = ViewPayload<float>(message_, descriptor.distances_offset,
                                      descriptor.count, header_.payload_offset,
                                      descriptor.payload_end);
  if (!distances.ok())
    return distances.status();
  return DistanceBatchResponseTaskView{descriptor, ids.value(),
                                       distances.value()};
}

} // namespace hnsw_urma
