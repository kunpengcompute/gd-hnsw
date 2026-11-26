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
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/transport/transport.hpp"
#include "hnsw_urma/transport/urma_resources.hpp"

namespace hnsw_urma {

enum class UmdkTpType : std::uint8_t { kRtp, kCtp };
enum class UmdkBondingMode : std::uint8_t {
  kStandalone,
  kActiveBackup,
  kBalance
};

struct UmdkBackendOptions {
  std::string device_name;
  std::optional<std::uint32_t> eid_index;
  std::uint32_t token{0x51a7U};
  UmdkTpType tp_type{UmdkTpType::kCtp};
  bool configure_bonding{false};
  // Borrowed from Milvus UrmaManager; must outlive this backend.
  void *external_context{nullptr};
  // Match B129 urma_perftest's default bonding mode.
  UmdkBondingMode bonding_mode{UmdkBondingMode::kStandalone};
};

struct UmdkLocalSegmentBinding {
  UrmaResourceHandle resource;
  std::span<std::byte> memory;
};

struct UmdkRemoteSegmentBinding {
  SegmentKind kind{SegmentKind::kLayer0Graph};
  UrmaResourceHandle resource;
  std::uint64_t bytes{0};
};

struct UmdkRemoteEndpointBinding {
  OperationClass operation_class{OperationClass::kRead};
  UrmaResourceHandle resource;
  std::uint32_t send_queue_index{0};
};

struct UmdkPeerBinding {
  std::uint32_t rank{0};
  std::vector<UmdkRemoteSegmentBinding> segments;
  std::vector<UmdkRemoteEndpointBinding> endpoints;
};

struct UmdkBounceBufferBinding {
  UrmaResourceHandle resource;
  std::span<std::byte> memory;
  std::size_t slot_bytes{0};
};

struct UmdkLaneOptions {
  std::uint32_t owner_lane{0};
  UrmaLaneResources resources;
  std::vector<UmdkLocalSegmentBinding> local_segments;
  std::vector<UmdkPeerBinding> peers;
  std::optional<UmdkBounceBufferBinding> write_bounce;
  std::size_t max_outstanding{1024};
  std::size_t wr_list_max{8};
  std::size_t poll_batch{16};
  bool profiling_enabled{false};
};

class UmdkLanePostEngine;

class UmdkResourceBackend final : public UrmaResourceBackend,
                                  public DmaVisibilityDomain {
public:
  UmdkResourceBackend(const UmdkResourceBackend &) = delete;
  UmdkResourceBackend &operator=(const UmdkResourceBackend &) = delete;
  ~UmdkResourceBackend() override;

  [[nodiscard]] static Result<std::unique_ptr<UmdkResourceBackend>>
  Create(const UmdkBackendOptions &options);

  [[nodiscard]] const UrmaCapabilities &capabilities() const noexcept;
  [[nodiscard]] bool requires_endpoint_advice() const noexcept;
  [[nodiscard]] std::uint32_t selected_eid_index() const noexcept;
  [[nodiscard]] Result<std::vector<std::byte>> LocalEid() const;
  [[nodiscard]] Result<std::vector<std::byte>>
  ExportDescriptor(UrmaResourceHandle handle) const;

  Result<UrmaResourceHandle>
  Create(const UrmaResourceRequest &request) override;
  void Destroy(UrmaResourceKind kind,
               UrmaResourceHandle handle) noexcept override;
  Status AcquireDeviceWrites() override;

  // Registers a borrowed imported segment. UrmaManager retains ownership.
  [[nodiscard]] Result<UrmaResourceHandle>
  ImportExternal(UrmaResourceKind kind, void *native_handle) noexcept;

private:
  struct Impl;
  explicit UmdkResourceBackend(std::unique_ptr<Impl> implementation);
  [[nodiscard]] void *Native(UrmaResourceHandle handle,
                             UrmaResourceKind expected_kind) const noexcept;
  std::unique_ptr<Impl> impl_;
  friend class UmdkLanePostEngine;
};

class UmdkLanePostEngine final : public LanePostEngine {
public:
  UmdkLanePostEngine(const UmdkLanePostEngine &) = delete;
  UmdkLanePostEngine &operator=(const UmdkLanePostEngine &) = delete;
  ~UmdkLanePostEngine() override;

  [[nodiscard]] static Result<std::unique_ptr<UmdkLanePostEngine>>
  Create(UmdkResourceBackend &backend, UmdkLaneOptions options);

  Status PostRead(const ReadOperation &operation) override;
  Status PostWriteNotify(const WriteNotifyOperation &operation) override;
  LanePostBatchResult
  PostReads(std::span<const ReadOperation> operations) override;
  LanePostBatchResult
  PostWriteNotifies(std::span<const WriteNotifyOperation> operations) override;
  [[nodiscard]] std::uint64_t
  native_descriptors_high_watermark() const noexcept;
  Result<std::size_t> Poll(std::span<TransportCompletion> completions) override;

private:
  struct Impl;
  explicit UmdkLanePostEngine(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> impl_;
};

} // namespace hnsw_urma
