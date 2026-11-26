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

// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <filesystem>
#include <fstream>

#include "hnsw_urma/compute/dataset.hpp"

TEST(ReferenceUnitDataset, FloatDatasetRoundTripsAndRejectsTruncation) {
  const auto directory =
      std::filesystem::temp_directory_path() / "hnsw_urma_dataset_test";
  std::filesystem::create_directories(directory);
  const auto valid_path = directory / "valid.bin";
  const auto bad_path = directory / "bad.bin";

  const hnsw_urma::FloatDataset expected{
      2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}};
  UB_REQUIRE_TRUE(hnsw_urma::WriteFloatBin(valid_path, expected).ok());
  auto loaded = hnsw_urma::LoadFloatBin(valid_path);
  UB_REQUIRE_TRUE(loaded.ok());
  UB_REQUIRE_EQ(loaded.value().rows, 2U);
  UB_REQUIRE_EQ(loaded.value().dim, 3U);
  UB_REQUIRE_EQ(loaded.value().values(), expected.values());

  std::ofstream truncated(bad_path, std::ios::binary | std::ios::trunc);
  const std::uint32_t header[2] = {2, 3};
  truncated.write(reinterpret_cast<const char *>(header), sizeof(header));
  truncated.close();
  UB_REQUIRE_FALSE(hnsw_urma::LoadFloatBin(bad_path).ok());

  std::filesystem::remove_all(directory);
}

TEST(ReferenceUnitDataset, FloatDatasetOwnerKeepsRowStorageAlive) {
  auto dataset = hnsw_urma::FloatDataset{2, 2, {1.0F, 2.0F, 3.0F, 4.0F}};
  const auto row = dataset.Row(1);
  auto owner = dataset.owner();
  std::weak_ptr<const void> lifetime = owner;

  dataset = {};
  UB_REQUIRE_FALSE(lifetime.expired());
  UB_REQUIRE_EQ(row[0], 3.0F);
  UB_REQUIRE_EQ(row[1], 4.0F);

  owner.reset();
  UB_REQUIRE_TRUE(lifetime.expired());
}
