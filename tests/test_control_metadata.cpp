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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "hnsw_urma/control/mpi_control.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "test_framework.hpp"

namespace hnsw_urma {

TEST(ReferenceFormatControlMetadata,
     CanonicalConfigEncodingIsStableAndRejectsCorruption) {
  CanonicalConfig first;
  CanonicalConfig second;
  UB_REQUIRE_TRUE(first.Set("topk", "10").ok());
  UB_REQUIRE_TRUE(first.Set("ef_search", "64").ok());
  UB_REQUIRE_TRUE(second.Set("ef_search", "64").ok());
  UB_REQUIRE_TRUE(second.Set("topk", "10").ok());
  UB_REQUIRE_EQ(first.Hash(), second.Hash());

  auto encoded = first.Encode();
  UB_REQUIRE_TRUE(encoded.ok());
  auto decoded = CanonicalConfig::Decode(encoded.value());
  UB_REQUIRE_TRUE(decoded.ok());
  UB_REQUIRE_EQ(decoded.value().values(), first.values());
  encoded.value().push_back(std::byte{0});
  UB_REQUIRE_FALSE(CanonicalConfig::Decode(encoded.value()).ok());
}

TEST(ReferenceFormatControlMetadata, PeerDescriptorRoundTripAndBounds) {
  PeerDescriptor peer;
  peer.rank = 3;
  peer.hostname = "node-a";
  peer.token = 0x12345678ULL;
  PeerDeviceDescriptor device;
  device.device_name = "udma0";
  device.eid = {std::byte{1}, std::byte{2}, std::byte{3}};
  device.segments.push_back({SegmentKind::kVectors,
                             4096,
                             3,
                             {std::byte{4}, std::byte{5}},
                             "/hu0000000100000002"});
  device.target_endpoints.push_back(
      {2, OperationClass::kResponseWrite, {std::byte{6}, std::byte{7}}});
  peer.devices.push_back(device);
  PeerDeviceDescriptor second = device;
  second.device_name = "udma1";
  peer.devices.push_back(std::move(second));
  auto encoded = EncodePeerDescriptor(peer);
  UB_REQUIRE_TRUE(encoded.ok());
  auto decoded = DecodePeerDescriptor(encoded.value());
  UB_REQUIRE_TRUE(decoded.ok());
  UB_REQUIRE_EQ(decoded.value().rank, 3U);
  UB_REQUIRE_EQ(decoded.value().hostname, "node-a");
  UB_REQUIRE_EQ(decoded.value().token, peer.token);
  UB_REQUIRE_EQ(decoded.value().devices.size(), 2U);
  UB_REQUIRE_EQ(decoded.value().devices[0].device_name, "udma0");
  UB_REQUIRE_EQ(decoded.value().devices[0].segments[0].kind,
                SegmentKind::kVectors);
  UB_REQUIRE_EQ(decoded.value().devices[0].segments[0].bytes, 4096U);
  UB_REQUIRE_EQ(decoded.value().devices[0].segments[0].shared_memory_name,
                "/hu0000000100000002");
  UB_REQUIRE_EQ(decoded.value().devices[0].target_endpoints[0].lane, 2U);

  PeerDescriptor duplicate = peer;
  duplicate.devices[1].device_name = duplicate.devices[0].device_name;
  UB_REQUIRE_FALSE(EncodePeerDescriptor(duplicate).ok());

  encoded.value().resize(encoded.value().size() - 1U);
  UB_REQUIRE_FALSE(DecodePeerDescriptor(encoded.value()).ok());
}

TEST(ReferenceFormatControlMetadata,
     MpiByteAllGatherIsBoundedInTheSingleRankFallback) {
  int argc = 1;
  char name[] = "control_metadata_test";
  char *values[]{name, nullptr};
  char **argv = values;
  auto mpi = MpiControl::Initialize(&argc, &argv);
  UB_REQUIRE_TRUE(mpi.ok());
  const std::array<std::byte, 3> local{std::byte{1}, std::byte{2},
                                       std::byte{3}};
  auto gathered = mpi.value()->AllGatherBytes(local, 16);
  UB_REQUIRE_TRUE(gathered.ok());
  UB_REQUIRE_EQ(gathered.value().size(),
                static_cast<std::size_t>(mpi.value()->world_size()));
  UB_REQUIRE_EQ(gathered.value()[static_cast<std::size_t>(mpi.value()->rank())],
                std::vector<std::byte>(local.begin(), local.end()));
  UB_REQUIRE_FALSE(mpi.value()->AllGatherBytes(local, 2).ok());
  UB_REQUIRE_TRUE(mpi.value()->ConvergeStatus(Status::Ok(), "unit").ok());
}

TEST(ReferenceFormatControlMetadata,
     PeerMetadataPreservesVariableLengthProviderExtensions) {
  PeerDescriptor peer;
  peer.hostname = "bonding-node";
  PeerDeviceDescriptor device;
  device.device_name = "bonding_dev_0";
  device.eid = {std::byte{1}};
  std::vector<std::byte> full_descriptor(2048);
  for (std::size_t i = 0; i < full_descriptor.size(); ++i) {
    full_descriptor[i] = static_cast<std::byte>(i % 251U);
  }
  device.segments.push_back(
      {SegmentKind::kVectors, 4096, 3, full_descriptor, {}});
  device.target_endpoints.push_back(
      {0, OperationClass::kRead, full_descriptor});
  peer.devices.push_back(std::move(device));
  auto encoded = EncodePeerDescriptor(peer);
  UB_REQUIRE_TRUE(encoded.ok());
  auto decoded = DecodePeerDescriptor(encoded.value());
  UB_REQUIRE_TRUE(decoded.ok());
  UB_REQUIRE_EQ(decoded.value().devices[0].segments[0].provider_descriptor,
                full_descriptor);
  UB_REQUIRE_EQ(
      decoded.value().devices[0].target_endpoints[0].provider_descriptor,
      full_descriptor);
}

} // namespace hnsw_urma
