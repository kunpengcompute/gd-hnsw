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

#include "hnsw_urma/control/mpi_control.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif
#include <unistd.h>

#if HNSW_URMA_HAS_MPI
#include <mpi.h>
#endif

namespace hnsw_urma {
namespace {

constexpr std::uint32_t kConfigMagic = 0x43464731U; // CFG1
constexpr std::uint32_t kPeerMagic = 0x50454552U;   // PEER
constexpr std::size_t kMaxConfigBytes = 1U << 20U;
constexpr std::size_t kMaxPeerDescriptorBytes = 16U << 20U;
constexpr std::size_t kMaxProviderDescriptorBytes = 4096;
constexpr std::size_t kMaxHostnameBytes = 255;
constexpr std::size_t kMaxDeviceNameBytes = 255;
constexpr std::size_t kMaxSharedMemoryNameBytes = 255;
constexpr std::size_t kMaxDevices = 16;
constexpr std::size_t kMaxSegments = 32;
constexpr std::size_t kMaxEndpoints = 256;

void AppendU32(std::vector<std::byte> &output, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

void AppendU64(std::vector<std::byte> &output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

Result<std::uint32_t> ReadU32(std::span<const std::byte> bytes,
                              std::size_t &offset) {
  if (offset > bytes.size() || sizeof(std::uint32_t) > bytes.size() - offset) {
    return Status(StatusCode::kInvalidFormat, "truncated uint32 field");
  }
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(bytes[offset++]) << shift;
  }
  return value;
}

Result<std::uint64_t> ReadU64(std::span<const std::byte> bytes,
                              std::size_t &offset) {
  if (offset > bytes.size() || sizeof(std::uint64_t) > bytes.size() - offset) {
    return Status(StatusCode::kInvalidFormat, "truncated uint64 field");
  }
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
  }
  return value;
}

Status AppendBlob(std::vector<std::byte> &output,
                  std::span<const std::byte> blob, std::size_t maximum) {
  if (blob.size() > maximum ||
      blob.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "metadata blob exceeds its fixed bound");
  }
  AppendU32(output, static_cast<std::uint32_t>(blob.size()));
  output.insert(output.end(), blob.begin(), blob.end());
  return Status::Ok();
}

Result<std::span<const std::byte>> ReadBlob(std::span<const std::byte> bytes,
                                            std::size_t &offset,
                                            std::size_t maximum) {
  auto length = ReadU32(bytes, offset);
  if (!length.ok()) {
    return length.status();
  }
  if (length.value() > maximum || offset > bytes.size() ||
      length.value() > bytes.size() - offset) {
    return Status(StatusCode::kInvalidFormat,
                  "metadata blob length is invalid");
  }
  auto blob = bytes.subspan(offset, length.value());
  offset += length.value();
  return blob;
}

ProcessPlacement DetectPlacement() {
  ProcessPlacement placement;
  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) == 0) {
    hostname.back() = '\0';
    placement.hostname = hostname.data();
  } else {
    placement.hostname = "unknown";
  }
#if defined(__linux__)
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) {
    placement.affinity_known = true;
    for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE);
         ++cpu) {
      if (CPU_ISSET(cpu, &cpus)) {
        placement.allowed_cpus.push_back(static_cast<std::uint32_t>(cpu));
      }
    }
  }
#endif
  if (placement.allowed_cpus.empty()) {
    const auto count = std::max(1U, std::thread::hardware_concurrency());
    placement.allowed_cpus.reserve(count);
    for (std::uint32_t cpu = 0; cpu < count; ++cpu) {
      placement.allowed_cpus.push_back(cpu);
    }
  }
  return placement;
}

#if HNSW_URMA_HAS_MPI
Status MpiStatus(int code, std::string operation) {
  if (code == MPI_SUCCESS) {
    return Status::Ok();
  }
  std::array<char, MPI_MAX_ERROR_STRING> message{};
  int length = 0;
  MPI_Error_string(code, message.data(), &length);
  return Status(StatusCode::kTransportError, std::move(operation) + " failed")
      .AddContext("mpi_error",
                  std::string(message.data(), static_cast<std::size_t>(length)))
      .AddContext("mpi_status", std::to_string(code));
}
#endif

} // namespace

Status CanonicalConfig::Set(std::string key, std::string value) {
  if (key.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "canonical config key must not be empty");
  }
  if (key.size() > std::numeric_limits<std::uint32_t>::max() ||
      value.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "canonical config value is too large");
  }
  values_.insert_or_assign(std::move(key), std::move(value));
  return Status::Ok();
}

Result<std::vector<std::byte>> CanonicalConfig::Encode() const {
  try {
    std::vector<std::byte> output;
    AppendU32(output, kConfigMagic);
    AppendU32(output, static_cast<std::uint32_t>(values_.size()));
    for (const auto &[key, value] : values_) {
      auto status =
          AppendBlob(output, std::as_bytes(std::span(key.data(), key.size())),
                     kMaxConfigBytes);
      if (!status.ok()) {
        return status;
      }
      status = AppendBlob(output,
                          std::as_bytes(std::span(value.data(), value.size())),
                          kMaxConfigBytes);
      if (!status.ok()) {
        return status;
      }
      if (output.size() > kMaxConfigBytes) {
        return Status(StatusCode::kOutOfRange,
                      "canonical config exceeds 1 MiB");
      }
    }
    return output;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to encode canonical config");
  }
}

std::uint64_t CanonicalConfig::Hash() const {
  auto encoded = Encode();
  if (!encoded.ok()) {
    return 0;
  }
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto value : encoded.value()) {
    hash ^= static_cast<std::uint8_t>(value);
    hash *= 1099511628211ULL;
  }
  return hash;
}

Result<CanonicalConfig>
CanonicalConfig::Decode(std::span<const std::byte> bytes) {
  if (bytes.size() > kMaxConfigBytes) {
    return Status(StatusCode::kInvalidFormat, "canonical config exceeds 1 MiB");
  }
  std::size_t offset = 0;
  auto magic = ReadU32(bytes, offset);
  auto count = ReadU32(bytes, offset);
  if (!magic.ok() || !count.ok() || magic.value() != kConfigMagic) {
    return Status(StatusCode::kInvalidFormat,
                  "canonical config header is invalid");
  }
  CanonicalConfig config;
  for (std::uint32_t index = 0; index < count.value(); ++index) {
    auto key = ReadBlob(bytes, offset, kMaxConfigBytes);
    auto value = ReadBlob(bytes, offset, kMaxConfigBytes);
    if (!key.ok() || !value.ok() || key.value().empty()) {
      return Status(StatusCode::kInvalidFormat,
                    "canonical config entry is invalid");
    }
    std::string key_string(reinterpret_cast<const char *>(key.value().data()),
                           key.value().size());
    if (config.values_.contains(key_string)) {
      return Status(StatusCode::kInvalidFormat,
                    "canonical config contains a duplicate key");
    }
    config.values_.emplace(
        std::move(key_string),
        std::string(reinterpret_cast<const char *>(value.value().data()),
                    value.value().size()));
  }
  if (offset != bytes.size()) {
    return Status(StatusCode::kInvalidFormat,
                  "canonical config has trailing bytes");
  }
  return config;
}

Result<std::vector<std::byte>>
EncodePeerDescriptor(const PeerDescriptor &descriptor) {
  if (descriptor.version != kPeerDescriptorVersion ||
      descriptor.devices.empty() || descriptor.devices.size() > kMaxDevices ||
      descriptor.hostname.empty() ||
      descriptor.hostname.size() > kMaxHostnameBytes) {
    return Status(StatusCode::kInvalidArgument,
                  "peer descriptor version or count is invalid");
  }
  for (std::size_t index = 0; index < descriptor.devices.size(); ++index) {
    const auto &device = descriptor.devices[index];
    if (device.device_name.empty() ||
        device.device_name.size() > kMaxDeviceNameBytes ||
        device.segments.size() > kMaxSegments ||
        device.target_endpoints.size() > kMaxEndpoints ||
        std::any_of(descriptor.devices.begin(),
                    descriptor.devices.begin() +
                        static_cast<std::ptrdiff_t>(index),
                    [&](const auto &prior) {
                      return prior.device_name == device.device_name;
                    })) {
      return Status(StatusCode::kInvalidArgument,
                    "peer device descriptor is invalid");
    }
  }
  try {
    std::vector<std::byte> output;
    AppendU32(output, kPeerMagic);
    AppendU32(output, descriptor.version);
    AppendU32(output, descriptor.rank);
    AppendU64(output, descriptor.token);
    auto status =
        AppendBlob(output,
                   std::as_bytes(std::span(descriptor.hostname.data(),
                                           descriptor.hostname.size())),
                   kMaxHostnameBytes);
    if (!status.ok()) {
      return status;
    }
    AppendU32(output, static_cast<std::uint32_t>(descriptor.devices.size()));
    for (const auto &device : descriptor.devices) {
      status = AppendBlob(output,
                          std::as_bytes(std::span(device.device_name.data(),
                                                  device.device_name.size())),
                          kMaxDeviceNameBytes);
      if (!status.ok()) {
        return status;
      }
      status = AppendBlob(output, device.eid, kMaxProviderDescriptorBytes);
      if (!status.ok()) {
        return status;
      }
      AppendU32(output, static_cast<std::uint32_t>(device.segments.size()));
      for (const auto &segment : device.segments) {
        AppendU32(output, static_cast<std::uint32_t>(segment.kind));
        AppendU64(output, segment.bytes);
        AppendU32(output, segment.permissions);
        status = AppendBlob(output, segment.provider_descriptor,
                            kMaxProviderDescriptorBytes);
        if (!status.ok()) {
          return status;
        }
        status = AppendBlob(
            output,
            std::as_bytes(std::span(segment.shared_memory_name.data(),
                                    segment.shared_memory_name.size())),
            kMaxSharedMemoryNameBytes);
        if (!status.ok()) {
          return status;
        }
      }
      AppendU32(output,
                static_cast<std::uint32_t>(device.target_endpoints.size()));
      for (const auto &endpoint : device.target_endpoints) {
        AppendU32(output, endpoint.lane);
        AppendU32(output, static_cast<std::uint32_t>(endpoint.operation_class));
        status = AppendBlob(output, endpoint.provider_descriptor,
                            kMaxProviderDescriptorBytes);
        if (!status.ok()) {
          return status;
        }
      }
    }
    if (output.size() > kMaxPeerDescriptorBytes) {
      return Status(StatusCode::kOutOfRange, "peer descriptor exceeds 16 MiB");
    }
    return output;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to encode peer descriptor");
  }
}

Result<PeerDescriptor> DecodePeerDescriptor(std::span<const std::byte> bytes) {
  if (bytes.size() > kMaxPeerDescriptorBytes) {
    return Status(StatusCode::kInvalidFormat, "peer descriptor exceeds 16 MiB");
  }
  std::size_t offset = 0;
  auto magic = ReadU32(bytes, offset);
  auto version = ReadU32(bytes, offset);
  auto rank = ReadU32(bytes, offset);
  auto token = ReadU64(bytes, offset);
  if (!magic.ok() || !version.ok() || !rank.ok() || !token.ok() ||
      magic.value() != kPeerMagic ||
      version.value() != kPeerDescriptorVersion) {
    return Status(StatusCode::kInvalidFormat,
                  "peer descriptor header is invalid");
  }
  auto hostname = ReadBlob(bytes, offset, kMaxHostnameBytes);
  auto device_count = ReadU32(bytes, offset);
  if (!hostname.ok() || hostname.value().empty() || !device_count.ok() ||
      device_count.value() == 0 || device_count.value() > kMaxDevices) {
    return Status(StatusCode::kInvalidFormat,
                  "peer hostname or device count is invalid");
  }
  PeerDescriptor descriptor;
  descriptor.version = version.value();
  descriptor.rank = rank.value();
  descriptor.token = token.value();
  descriptor.hostname.assign(
      reinterpret_cast<const char *>(hostname.value().data()),
      hostname.value().size());
  for (std::uint32_t device_index = 0; device_index < device_count.value();
       ++device_index) {
    auto device_name = ReadBlob(bytes, offset, kMaxDeviceNameBytes);
    auto eid = ReadBlob(bytes, offset, kMaxProviderDescriptorBytes);
    auto segment_count = ReadU32(bytes, offset);
    if (!device_name.ok() || device_name.value().empty() || !eid.ok() ||
        !segment_count.ok() || segment_count.value() > kMaxSegments) {
      return Status(StatusCode::kInvalidFormat,
                    "peer device metadata is invalid");
    }
    PeerDeviceDescriptor device;
    device.device_name.assign(
        reinterpret_cast<const char *>(device_name.value().data()),
        device_name.value().size());
    if (std::any_of(descriptor.devices.begin(), descriptor.devices.end(),
                    [&](const auto &prior) {
                      return prior.device_name == device.device_name;
                    })) {
      return Status(StatusCode::kInvalidFormat,
                    "peer descriptor contains a duplicate device");
    }
    device.eid.assign(eid.value().begin(), eid.value().end());
    for (std::uint32_t index = 0; index < segment_count.value(); ++index) {
      auto kind = ReadU32(bytes, offset);
      auto length = ReadU64(bytes, offset);
      auto permissions = ReadU32(bytes, offset);
      auto provider = ReadBlob(bytes, offset, kMaxProviderDescriptorBytes);
      auto shared_memory = ReadBlob(bytes, offset, kMaxSharedMemoryNameBytes);
      if (!kind.ok() || !length.ok() || !permissions.ok() || !provider.ok() ||
          !shared_memory.ok() ||
          kind.value() > static_cast<std::uint32_t>(SegmentKind::kRpcPool) ||
          length.value() == 0) {
        return Status(StatusCode::kInvalidFormat,
                      "exported segment descriptor is invalid");
      }
      device.segments.push_back(
          {static_cast<SegmentKind>(kind.value()), length.value(),
           permissions.value(),
           std::vector<std::byte>(provider.value().begin(),
                                  provider.value().end()),
           std::string(
               reinterpret_cast<const char *>(shared_memory.value().data()),
               shared_memory.value().size())});
    }
    auto endpoint_count = ReadU32(bytes, offset);
    if (!endpoint_count.ok() || endpoint_count.value() > kMaxEndpoints) {
      return Status(StatusCode::kInvalidFormat,
                    "target endpoint count is invalid");
    }
    for (std::uint32_t index = 0; index < endpoint_count.value(); ++index) {
      auto lane = ReadU32(bytes, offset);
      auto operation_class = ReadU32(bytes, offset);
      auto provider = ReadBlob(bytes, offset, kMaxProviderDescriptorBytes);
      if (!lane.ok() || !operation_class.ok() || !provider.ok() ||
          operation_class.value() >
              static_cast<std::uint32_t>(OperationClass::kResponseWrite)) {
        return Status(StatusCode::kInvalidFormat,
                      "target endpoint descriptor is invalid");
      }
      device.target_endpoints.push_back(
          {lane.value(), static_cast<OperationClass>(operation_class.value()),
           std::vector<std::byte>(provider.value().begin(),
                                  provider.value().end())});
    }
    descriptor.devices.push_back(std::move(device));
  }
  if (offset != bytes.size()) {
    return Status(StatusCode::kInvalidFormat,
                  "peer descriptor has trailing bytes");
  }
  return descriptor;
}

MpiControl::~MpiControl() {
#if HNSW_URMA_HAS_MPI
  if (owns_mpi_) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) {
      MPI_Finalize();
    }
  }
#endif
}

Result<std::unique_ptr<MpiControl>> MpiControl::Initialize(int *argc,
                                                           char ***argv) {
  auto control = std::unique_ptr<MpiControl>(new (std::nothrow) MpiControl());
  if (!control) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate MPI control plane");
  }
  control->control_thread_ = std::this_thread::get_id();
  control->placement_ = DetectPlacement();
#if HNSW_URMA_HAS_MPI
  int initialized = 0;
  auto status = MpiStatus(MPI_Initialized(&initialized), "MPI_Initialized");
  if (!status.ok()) {
    return status;
  }
  if (initialized == 0) {
    int provided = 0;
    status =
        MpiStatus(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided),
                  "MPI_Init_thread");
    if (!status.ok()) {
      return status;
    }
    control->owns_mpi_ = true;
    control->provided_thread_level_ = provided;
  } else {
    MPI_Query_thread(&control->provided_thread_level_);
  }
  if (control->provided_thread_level_ < MPI_THREAD_FUNNELED) {
    return Status(StatusCode::kTransportError,
                  "MPI implementation did not provide MPI_THREAD_FUNNELED");
  }
  status = MpiStatus(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
                     "MPI_Comm_set_errhandler");
  if (!status.ok()) {
    return status;
  }
  status = MpiStatus(MPI_Comm_rank(MPI_COMM_WORLD, &control->rank_),
                     "MPI_Comm_rank");
  if (!status.ok()) {
    return status;
  }
  status = MpiStatus(MPI_Comm_size(MPI_COMM_WORLD, &control->world_size_),
                     "MPI_Comm_size");
  if (!status.ok()) {
    return status;
  }
#else
  (void)argc;
  (void)argv;
#endif
  return control;
}

Status MpiControl::CheckControlThread() const {
  if (std::this_thread::get_id() != control_thread_) {
    return Status(StatusCode::kInternalError,
                  "MPI control operation called from a data-plane thread");
  }
  return Status::Ok();
}

Status MpiControl::ValidateShardWorld(std::uint32_t shard_count) const {
  if (shard_count != static_cast<std::uint32_t>(world_size_)) {
    return Status(StatusCode::kInvalidArgument,
                  "index shard count must equal MPI world size")
        .AddContext("shards", std::to_string(shard_count))
        .AddContext("world_size", std::to_string(world_size_));
  }
  return Status::Ok();
}

Result<CanonicalConfig>
MpiControl::SynchronizeConfig(const CanonicalConfig &local, int root) const {
  auto thread_status = CheckControlThread();
  if (!thread_status.ok()) {
    return thread_status;
  }
  if (root < 0 || root >= world_size_) {
    return Status(StatusCode::kInvalidArgument,
                  "canonical config root rank is invalid");
  }
  auto encoded = local.Encode();
  if (!encoded.ok()) {
    return encoded.status();
  }
#if HNSW_URMA_HAS_MPI
  const std::uint64_t local_hash = local.Hash();
  std::vector<std::uint64_t> hashes(static_cast<std::size_t>(world_size_));
  auto status =
      MpiStatus(MPI_Allgather(&local_hash, 1, MPI_UINT64_T, hashes.data(), 1,
                              MPI_UINT64_T, MPI_COMM_WORLD),
                "MPI_Allgather(config hash)");
  if (!status.ok()) {
    return status;
  }
  for (int rank = 0; rank < world_size_; ++rank) {
    if (hashes[static_cast<std::size_t>(rank)] !=
        hashes[static_cast<std::size_t>(root)]) {
      const auto commit = local.values().find("git_commit");
      return Status(StatusCode::kProtocolError,
                    "cluster-wide runtime config or build identity disagree")
          .AddContext("rank", std::to_string(rank))
          .AddContext("local_git_commit", commit == local.values().end()
                                              ? "<missing>"
                                              : commit->second)
          .AddContext("local_config_hash", std::to_string(local_hash))
          .AddContext("root_config_hash",
                      std::to_string(hashes[static_cast<std::size_t>(root)]));
    }
  }
  std::uint64_t length = encoded.value().size();
  status = MpiStatus(MPI_Bcast(&length, 1, MPI_UINT64_T, root, MPI_COMM_WORLD),
                     "MPI_Bcast(config length)");
  if (!status.ok()) {
    return status;
  }
  if (length > kMaxConfigBytes ||
      length > static_cast<std::uint64_t>(INT_MAX)) {
    return Status(StatusCode::kProtocolError,
                  "broadcast canonical config length is invalid");
  }
  if (rank_ != root) {
    encoded.value().resize(static_cast<std::size_t>(length));
  }
  status = MpiStatus(MPI_Bcast(encoded.value().data(), static_cast<int>(length),
                               MPI_BYTE, root, MPI_COMM_WORLD),
                     "MPI_Bcast(config bytes)");
  if (!status.ok()) {
    return status;
  }
#endif
  return CanonicalConfig::Decode(encoded.value());
}

Result<std::vector<PeerDescriptor>>
MpiControl::AllGatherPeerDescriptors(const PeerDescriptor &local) const {
  auto thread_status = CheckControlThread();
  if (!thread_status.ok()) {
    return thread_status;
  }
  if (local.rank != static_cast<std::uint32_t>(rank_)) {
    return Status(StatusCode::kInvalidArgument,
                  "local peer descriptor rank is inconsistent");
  }
  auto encoded = EncodePeerDescriptor(local);
  if (!encoded.ok()) {
    return encoded.status();
  }
#if HNSW_URMA_HAS_MPI
  std::uint64_t local_length = encoded.value().size();
  std::vector<std::uint64_t> lengths(static_cast<std::size_t>(world_size_));
  auto status =
      MpiStatus(MPI_Allgather(&local_length, 1, MPI_UINT64_T, lengths.data(), 1,
                              MPI_UINT64_T, MPI_COMM_WORLD),
                "MPI_Allgather(peer descriptor lengths)");
  if (!status.ok()) {
    return status;
  }
  std::vector<int> counts(static_cast<std::size_t>(world_size_));
  std::vector<int> displacements(static_cast<std::size_t>(world_size_));
  std::size_t total = 0;
  for (int rank = 0; rank < world_size_; ++rank) {
    const auto length = lengths[static_cast<std::size_t>(rank)];
    if (length > kMaxPeerDescriptorBytes ||
        length > static_cast<std::uint64_t>(INT_MAX) ||
        total > static_cast<std::size_t>(INT_MAX) - length) {
      return Status(StatusCode::kProtocolError,
                    "peer descriptor collective exceeds MPI bounds");
    }
    counts[static_cast<std::size_t>(rank)] = static_cast<int>(length);
    displacements[static_cast<std::size_t>(rank)] = static_cast<int>(total);
    total += static_cast<std::size_t>(length);
  }
  std::vector<std::byte> gathered(total);
  status =
      MpiStatus(MPI_Allgatherv(encoded.value().data(),
                               static_cast<int>(encoded.value().size()),
                               MPI_BYTE, gathered.data(), counts.data(),
                               displacements.data(), MPI_BYTE, MPI_COMM_WORLD),
                "MPI_Allgatherv(peer descriptors)");
  if (!status.ok()) {
    return status;
  }
  std::vector<PeerDescriptor> peers;
  peers.reserve(static_cast<std::size_t>(world_size_));
  for (int rank = 0; rank < world_size_; ++rank) {
    auto peer = DecodePeerDescriptor(std::span<const std::byte>(
        gathered.data() + displacements[static_cast<std::size_t>(rank)],
        static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)])));
    if (!peer.ok()) {
      return peer.status();
    }
    if (peer.value().rank != static_cast<std::uint32_t>(rank)) {
      return Status(StatusCode::kProtocolError,
                    "gathered peer descriptor rank is inconsistent");
    }
    peers.push_back(std::move(peer.value()));
  }
  return peers;
#else
  std::vector<PeerDescriptor> peers;
  peers.push_back(local);
  return peers;
#endif
}

Result<std::vector<std::vector<std::byte>>>
MpiControl::AllGatherBytes(std::span<const std::byte> local,
                           std::size_t maximum_per_rank) const {
  auto thread_status = CheckControlThread();
  if (!thread_status.ok()) {
    return thread_status;
  }
  if (maximum_per_rank == 0 || local.size() > maximum_per_rank ||
      maximum_per_rank > static_cast<std::size_t>(INT_MAX)) {
    return Status(StatusCode::kInvalidArgument,
                  "all-gather byte payload exceeds its configured bound");
  }
#if HNSW_URMA_HAS_MPI
  std::uint64_t local_length = local.size();
  std::vector<std::uint64_t> lengths(static_cast<std::size_t>(world_size_));
  auto status =
      MpiStatus(MPI_Allgather(&local_length, 1, MPI_UINT64_T, lengths.data(), 1,
                              MPI_UINT64_T, MPI_COMM_WORLD),
                "MPI_Allgather(byte lengths)");
  if (!status.ok()) {
    return status;
  }
  std::vector<int> counts(static_cast<std::size_t>(world_size_));
  std::vector<int> displacements(static_cast<std::size_t>(world_size_));
  std::size_t total = 0;
  for (int rank = 0; rank < world_size_; ++rank) {
    const auto length = lengths[static_cast<std::size_t>(rank)];
    if (length > maximum_per_rank ||
        length > static_cast<std::uint64_t>(INT_MAX) ||
        total > static_cast<std::size_t>(INT_MAX) - length) {
      return Status(StatusCode::kProtocolError,
                    "all-gather byte collective exceeds MPI bounds");
    }
    counts[static_cast<std::size_t>(rank)] = static_cast<int>(length);
    displacements[static_cast<std::size_t>(rank)] = static_cast<int>(total);
    total += static_cast<std::size_t>(length);
  }
  std::vector<std::byte> gathered(total);
  status =
      MpiStatus(MPI_Allgatherv(local.data(), static_cast<int>(local.size()),
                               MPI_BYTE, gathered.data(), counts.data(),
                               displacements.data(), MPI_BYTE, MPI_COMM_WORLD),
                "MPI_Allgatherv(bytes)");
  if (!status.ok()) {
    return status;
  }
  std::vector<std::vector<std::byte>> output;
  output.reserve(static_cast<std::size_t>(world_size_));
  for (int rank = 0; rank < world_size_; ++rank) {
    const auto displacement =
        static_cast<std::size_t>(displacements[static_cast<std::size_t>(rank)]);
    const auto count =
        static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)]);
    output.emplace_back(
        gathered.begin() + static_cast<std::ptrdiff_t>(displacement),
        gathered.begin() + static_cast<std::ptrdiff_t>(displacement + count));
  }
  return output;
#else
  return std::vector<std::vector<std::byte>>{
      std::vector<std::byte>(local.begin(), local.end())};
#endif
}

Status MpiControl::ConvergeStatus(const Status &local,
                                  std::string_view phase) const {
  auto thread_status = CheckControlThread();
  if (!thread_status.ok()) {
    return thread_status;
  }
#if HNSW_URMA_HAS_MPI
  int candidate = local.ok() ? world_size_ : rank_;
  int first_failure = world_size_;
  auto status = MpiStatus(MPI_Allreduce(&candidate, &first_failure, 1, MPI_INT,
                                        MPI_MIN, MPI_COMM_WORLD),
                          "MPI_Allreduce(phase status)");
  if (!status.ok()) {
    return status;
  }
  if (first_failure == world_size_) {
    return Status::Ok();
  }
  std::string message =
      rank_ == first_failure ? local.ToString() : std::string();
  std::uint64_t length = message.size();
  status = MpiStatus(
      MPI_Bcast(&length, 1, MPI_UINT64_T, first_failure, MPI_COMM_WORLD),
      "MPI_Bcast(phase error length)");
  if (!status.ok()) {
    return status;
  }
  if (length > kMaxConfigBytes ||
      length > static_cast<std::uint64_t>(INT_MAX)) {
    return Status(StatusCode::kProtocolError,
                  "collective phase error message exceeds its bound");
  }
  message.resize(static_cast<std::size_t>(length));
  status = MpiStatus(MPI_Bcast(message.data(), static_cast<int>(length),
                               MPI_BYTE, first_failure, MPI_COMM_WORLD),
                     "MPI_Bcast(phase error)");
  if (!status.ok()) {
    return status;
  }
  return Status(StatusCode::kInternalError, "distributed phase failed")
      .AddContext("phase", std::string(phase))
      .AddContext("first_failure_rank", std::to_string(first_failure))
      .AddContext("cause", std::move(message));
#else
  if (!local.ok()) {
    Status output = local;
    output.AddContext("phase", std::string(phase));
    return output;
  }
  return Status::Ok();
#endif
}

Status MpiControl::Barrier() const {
  auto thread_status = CheckControlThread();
  if (!thread_status.ok()) {
    return thread_status;
  }
#if HNSW_URMA_HAS_MPI
  return MpiStatus(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
#else
  return Status::Ok();
#endif
}

bool MpiSupportCompiled() {
#if HNSW_URMA_HAS_MPI
  return true;
#else
  return false;
#endif
}

} // namespace hnsw_urma
