#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {

inline constexpr std::uint32_t kPeerDescriptorVersion = 3;

class CanonicalConfig {
 public:
  Status Set(std::string key, std::string value);
  [[nodiscard]] const std::map<std::string, std::string>& values() const noexcept {
    return values_;
  }
  [[nodiscard]] Result<std::vector<std::byte>> Encode() const;
  [[nodiscard]] std::uint64_t Hash() const;
  [[nodiscard]] static Result<CanonicalConfig> Decode(std::span<const std::byte> bytes);

 private:
  std::map<std::string, std::string> values_;
};

struct ExportedSegmentDescriptor {
  SegmentKind kind{SegmentKind::kLayer0Graph};
  std::uint64_t bytes{0};
  std::uint32_t permissions{0};
  std::vector<std::byte> provider_descriptor;
  std::string shared_memory_name;
};

struct TargetEndpointDescriptor {
  std::uint32_t lane{0};
  OperationClass operation_class{OperationClass::kRead};
  std::vector<std::byte> provider_descriptor;
};

struct PeerDeviceDescriptor {
  std::string device_name;
  std::vector<std::byte> eid;
  std::vector<ExportedSegmentDescriptor> segments;
  std::vector<TargetEndpointDescriptor> target_endpoints;
};

struct PeerDescriptor {
  std::uint32_t version{kPeerDescriptorVersion};
  std::uint32_t rank{0};
  std::string hostname;
  std::uint64_t token{0};
  std::vector<PeerDeviceDescriptor> devices;
};

[[nodiscard]] Result<std::vector<std::byte>> EncodePeerDescriptor(
    const PeerDescriptor& descriptor);
[[nodiscard]] Result<PeerDescriptor> DecodePeerDescriptor(std::span<const std::byte> bytes);

struct ProcessPlacement {
  std::string hostname;
  std::vector<std::uint32_t> allowed_cpus;
  bool affinity_known{false};
};

class MpiControl {
 public:
  MpiControl(const MpiControl&) = delete;
  MpiControl& operator=(const MpiControl&) = delete;
  ~MpiControl();

  [[nodiscard]] static Result<std::unique_ptr<MpiControl>> Initialize(int* argc, char*** argv);

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int world_size() const noexcept { return world_size_; }
  [[nodiscard]] int provided_thread_level() const noexcept { return provided_thread_level_; }
  [[nodiscard]] const ProcessPlacement& placement() const noexcept { return placement_; }

  [[nodiscard]] Status ValidateShardWorld(std::uint32_t shard_count) const;
  [[nodiscard]] Result<CanonicalConfig> SynchronizeConfig(const CanonicalConfig& local,
                                                           int root = 0) const;
  [[nodiscard]] Result<std::vector<PeerDescriptor>> AllGatherPeerDescriptors(
      const PeerDescriptor& local) const;
  [[nodiscard]] Result<std::vector<std::vector<std::byte>>> AllGatherBytes(
      std::span<const std::byte> local, std::size_t maximum_per_rank) const;
  [[nodiscard]] Status ConvergeStatus(const Status& local,
                                      std::string_view phase) const;
  [[nodiscard]] Status Barrier() const;

 private:
  MpiControl() = default;
  [[nodiscard]] Status CheckControlThread() const;

  bool owns_mpi_{false};
  int rank_{0};
  int world_size_{1};
  int provided_thread_level_{0};
  ProcessPlacement placement_;
  std::thread::id control_thread_;
};

[[nodiscard]] bool MpiSupportCompiled();

}  // namespace hnsw_urma
