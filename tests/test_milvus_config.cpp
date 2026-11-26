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
#include "hnsw_urma/milvus_ub_runtime.hpp"
#include "hnsw_urma/runtime/config.hpp"
#include "test_framework.hpp"

using namespace hnsw_urma;
TEST(ReferenceUnitMilvusConfig, MilvusRuntimeOptionsOverrideWithoutClamping) {
  MilvusUbRuntime::Knobs k;
  k.urma_device = "bonding_dev_0";
  k.eid_index = 3;
  k.search_workers = 12;
  k.distance_workers = 18;
  k.read_progress_workers = 2;
  k.rpc_progress_workers = 7;
  k.expand_batch = 1;
  k.runtime_options =
      "--distance-policy pushdown --use-dot-norm --layer0-sort local-only "
      "--read-progress-mode adaptive --read-progress-spin-us 13 "
      "--request-jfs-per-rpc-lane 2 --rpc-submit-depth auto";
  auto parsed = BuildMilvusRuntimeConfig(k);
  UB_REQUIRE_TRUE(parsed.ok());
  const auto &c = parsed.value();
  UB_REQUIRE_EQ(c.search_workers.value, 12U);
  UB_REQUIRE_EQ(c.distance_workers.value, 18U);
  UB_REQUIRE_EQ(c.read_progress_threads.value, 2U);
  UB_REQUIRE_EQ(c.rpc_progress_threads.value, 7U);
  UB_REQUIRE_EQ(c.expand_batch, 1U);
  UB_REQUIRE_EQ(c.eid_index.value, 3U);
  UB_REQUIRE_EQ(c.distance_policy, DistancePolicy::kPushdown);
  UB_REQUIRE_TRUE(c.use_dot_norm);
  UB_REQUIRE_EQ(c.layer0_sort_mode, Layer0SortMode::kLocalOnly);
  UB_REQUIRE_EQ(c.read_progress_mode, ProgressMode::kAdaptive);
  UB_REQUIRE_EQ(c.rpc_progress_mode, ProgressMode::kBusy);
  UB_REQUIRE_EQ(c.read_progress_spin_us.value, 13U);
  UB_REQUIRE_EQ(c.request_jfs_per_rpc_lane, 2U);
  UB_REQUIRE_TRUE(c.rpc_submit_depth.automatic);
  UB_REQUIRE_EQ(c.cpu_layout, CpuLayout::kContiguous);
  k.runtime_options =
      "--expand-batch 20 --distance-policy pull --layer0-sort upper-only";
  auto changed = BuildMilvusRuntimeConfig(k).value();
  UB_REQUIRE_EQ(changed.expand_batch, 20U);
  UB_REQUIRE_FALSE(changed.use_dot_norm);
  UB_REQUIRE_EQ(changed.distance_policy, DistancePolicy::kPull);
}
TEST(ReferenceUnitMilvusConfig,
     MilvusRejectsInvalidOptionsAtConfigurationBoundary) {
  MilvusUbRuntime::Knobs k;
  k.urma_device = "bonding_dev_0";
  for (const auto *text :
       {"--expand-batch 65", "--distance-policy invalid", "--topk 10",
        "--rounds 5", "--tp-type rtp", "--urma-device other",
        "--request-jfs-depth", "--unknown 1"}) {
    k.runtime_options = text;
    UB_REQUIRE_FALSE(BuildMilvusRuntimeConfig(k).ok());
  }
}
TEST(ReferenceUnitMilvusConfig,
     MilvusRuntimeResolutionUsesConfiguredThreadsAndQueues) {
  MilvusUbRuntime::Knobs k;
  k.urma_device = "bonding_dev_0";
  k.search_workers = 12;
  k.distance_workers = 18;
  k.read_progress_workers = 2;
  k.rpc_progress_workers = 7;
  k.expand_batch = 20;
  k.runtime_options =
      "--distance-policy pushdown --use-dot-norm --layer0-sort local-only "
      "--read-submit-depth 256 --read-jfs-depth 2048 "
      "--request-jfs-per-rpc-lane 2 --request-ring-depth 32 "
      "--read-progress-mode adaptive --read-progress-spin-us 13";
  GlobalIndexMeta meta;
  meta.num_shards = 8;
  meta.dim = 960;
  meta.M = 32;
  meta.M0 = 64;
  meta.ntotal = 1000000;
  meta.ef_search_default = 350;
  std::vector<std::uint32_t> cpus(48);
  for (std::uint32_t i = 0; i < cpus.size(); ++i)
    cpus[i] = i;
  auto parsed = BuildMilvusRuntimeConfig(k);
  if (!parsed.ok())
    test::Fail("parse", __FILE__, __LINE__, parsed.status().ToString());
  auto requested = std::move(parsed).value();
  UB_REQUIRE_EQ(requested.eid_index.value, 0U);
  auto plan = ResolveRuntimeConfig(requested, meta, 8, cpus, 0);
  if (!plan.ok())
    test::Fail("resolve", __FILE__, __LINE__, plan.status().ToString());
  const auto &e = plan.value();
  UB_REQUIRE_EQ(e.search_workers, 12U);
  UB_REQUIRE_EQ(e.distance_workers, 18U);
  UB_REQUIRE_EQ(e.read_progress_threads, 2U);
  UB_REQUIRE_EQ(e.rpc_progress_threads, 7U);
  UB_REQUIRE_EQ(e.maximum_distance_batch, 1280U);
  UB_REQUIRE_EQ(e.queues.depths.read_submit, 256U);
  UB_REQUIRE_EQ(e.urma.read_jfs_depth, 2048U);
  UB_REQUIRE_EQ(e.urma.request_jfs_per_rpc_lane, 2U);
  UB_REQUIRE_EQ(e.read_progress_spin_us, 13U);
  auto canonical = BuildCanonicalRuntimeConfig(requested, e);
  UB_REQUIRE_TRUE(canonical.ok());
}
