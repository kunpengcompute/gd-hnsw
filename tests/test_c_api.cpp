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

#include "fixture.hpp"
#include "hnsw_urma/milvus_query_executor.hpp"
#include "hnsw_urma/milvus_ub_runtime.hpp"
#include "hnsw_urma/search/mock_io.hpp"
#include "hnsw_urma/ub_c_api.h"
#include <array>
#include <limits>
#include <thread>
using namespace hnsw_urma;
namespace {
class CApi : public ub_complete::IndexFixture {
protected:
  void SetUp() override {
    ASSERT_EQ(UbShutdown(), 0);
    MilvusUbRuntime::Instance().SetKnobs(MilvusUbRuntime::Knobs{});
    IndexFixture::SetUp();
  }
  void TearDown() override {
    EXPECT_EQ(UbShutdown(), 0);
    MilvusUbRuntime::Instance().SetKnobs(MilvusUbRuntime::Knobs{});
    IndexFixture::TearDown();
  }
  void ConfigureForMailbox() {
    const std::array<std::uint32_t, 8> cpus{0, 1, 2, 3, 4, 5, 6, 7};
    UbConfig config{};
    config.search_workers = 1;
    config.distance_workers = 1;
    config.read_progress_workers = 1;
    config.rpc_progress_workers = 1;
    config.cpu_bind_mode = "none";
    config.cpu_layout = "contiguous";
    config.allowed_cpus = cpus.data();
    config.allowed_cpu_count = cpus.size();
    config.max_active_queries = 2;
    ASSERT_EQ(UbConfigure(&config), 0);
  }
};
} // namespace
TEST_F(CApi, PingAndRepeatedShutdown) {
  EXPECT_EQ(UbPing(0), 1);
  EXPECT_EQ(UbPing(-2), -1);
  EXPECT_EQ(UbStopDistributedSearch(), 0);
  EXPECT_EQ(UbShutdown(), 0);
  EXPECT_EQ(UbShutdown(), 0);
  EXPECT_EQ(UbRpcDistanceReady(), 0);
}
TEST_F(CApi, ConfigurationCopiesFieldsAndClearsLastError) {
  EXPECT_NE(UbConfigure(nullptr), 0);
  ASSERT_NE(UbLastError(), nullptr);
  ConfigureForMailbox();
  EXPECT_EQ(UbLastError(), nullptr);
  const auto knobs = MilvusUbRuntime::Instance().GetKnobs();
  EXPECT_EQ(knobs.search_workers, 1U);
  EXPECT_EQ(knobs.allowed_cpus.size(), 8U);
  EXPECT_EQ(knobs.cpu_bind_mode, "none");
  EXPECT_EQ(UbSetUrmaContext(nullptr), 0);
}
TEST_F(CApi, LastErrorIsThreadLocal) {
  ASSERT_NE(UbConfigure(nullptr), 0);
  const std::string original = UbLastError();
  bool initially_empty = false, has_error = false;
  std::jthread worker([&] {
    initially_empty = UbLastError() == nullptr;
    (void)UbReadIndexMeta(nullptr, 0, nullptr);
    has_error = UbLastError() != nullptr;
  });
  worker.join();
  EXPECT_TRUE(initially_empty);
  EXPECT_TRUE(has_error);
  ASSERT_NE(UbLastError(), nullptr);
  EXPECT_EQ(std::string(UbLastError()), original);
}
TEST_F(CApi, MetadataProjectionAndNullArguments) {
  UbIndexMeta meta{};
  EXPECT_NE(UbReadIndexMeta(nullptr, 0, &meta), 0);
  EXPECT_NE(UbReadIndexMeta("", 0, &meta), 0);
  EXPECT_NE(UbReadIndexMeta(root_.c_str(), 0, nullptr), 0);
  ASSERT_EQ(UbReadIndexMeta(root_.c_str(), 0, &meta), 0);
  EXPECT_EQ(meta.dim, 2U);
  EXPECT_EQ(meta.ntotal, 8U);
  EXPECT_EQ(meta.shard_count, 2U);
  EXPECT_EQ(meta.local_count, 4U);
  EXPECT_EQ(meta.reserved, 0U);
  ASSERT_EQ(UbReadIndexMeta(root_.c_str(), 99, &meta), 0);
  EXPECT_EQ(meta.local_count, 0U);
  EXPECT_NE(UbReadIndexMeta((root_ / "missing").c_str(), 0, &meta), 0);
}
TEST_F(CApi, LoadShardValidatesInputAndIsIdempotent) {
  UbIndexMeta meta{};
  EXPECT_NE(UbLoadShardFromDir(nullptr, 0, &meta), 0);
  EXPECT_NE(UbLoadShardFromDir("", 0, &meta), 0);
  EXPECT_NE(UbLoadShardFromDir(root_.c_str(), 99, &meta), 0);
  ASSERT_EQ(UbLoadShardFromDir(root_.c_str(), 0, &meta), 0);
  EXPECT_EQ(meta.local_count, 4U);
  ASSERT_EQ(UbLoadShardFromDir(root_.c_str(), 0, nullptr), 0);
  UbShardStatus status{};
  ASSERT_EQ(UbGetShardStatus(&status), 0);
  EXPECT_EQ(status.loaded_count, 1U);
  EXPECT_EQ(status.expected_count, 2U);
  EXPECT_EQ(status.local_shard_id, 0U);
}
TEST_F(CApi, SeedAllSubsetAndPinLocalShard) {
  EXPECT_NE(UbSeedFromDir(nullptr, nullptr, 0), 0);
  const std::uint32_t invalid = 9;
  EXPECT_NE(UbSeedFromDir(root_.c_str(), &invalid, 1), 0);
  ASSERT_EQ(UbSeedFromDir(root_.c_str(), nullptr, 0), 0);
  UbShardStatus status{};
  ASSERT_EQ(UbGetShardStatus(&status), 0);
  EXPECT_EQ(status.loaded_count, 2U);
  EXPECT_EQ(status.local_shard_id, UB_NO_LOCAL_SHARD);
  EXPECT_NE(UbSetLocalShard(9), 0);
  ASSERT_EQ(UbSetLocalShard(1), 0);
  ASSERT_EQ(UbGetShardStatus(&status), 0);
  EXPECT_EQ(status.loaded_count, 1U);
  EXPECT_EQ(status.local_shard_id, 1U);
  ASSERT_EQ(UbSeedFromDir(root_.c_str(), &invalid, 0), 0);
}
TEST_F(CApi, ShardStatusBeforeSeedAndNullOutput) {
  EXPECT_NE(UbGetShardStatus(nullptr), 0);
  UbShardStatus status{};
  ASSERT_EQ(UbGetShardStatus(&status), 0);
  EXPECT_EQ(status.loaded_count, 0U);
  EXPECT_EQ(status.local_shard_id, UB_NO_LOCAL_SHARD);
  EXPECT_NE(UbSetLocalShard(0), 0);
}
TEST_F(CApi, GlobalIdentityAndReuseAreValidated) {
  EXPECT_NE(UbBindGlobalGraphIdentity("g", 1), 0);
  ASSERT_EQ(UbSeedFromDir(root_.c_str(), nullptr, 0), 0);
  EXPECT_NE(UbBindGlobalGraphIdentity(nullptr, 1), 0);
  EXPECT_NE(UbBindGlobalGraphIdentity("g", 0), 0);
  ASSERT_EQ(UbBindGlobalGraphIdentity("g", 1), 0);
  EXPECT_EQ(UbBindGlobalGraphIdentity("g", 1), 0);
  EXPECT_NE(UbBindGlobalGraphIdentity("other", 1), 0);
  int reused = 0;
  UbSeededShardInfo info{};
  ASSERT_EQ(UbReuseGlobalGraphShard("g", 1, 0, &reused, &info), 0);
  EXPECT_EQ(reused, 1);
  EXPECT_EQ(info.dim, 2U);
  EXPECT_EQ(info.count, 4U);
  EXPECT_NE(UbReuseGlobalGraphShard("g", 2, 0, &reused, &info), 0);
  EXPECT_NE(UbReuseGlobalGraphShard("g", 1, 99, &reused, &info), 0);
  EXPECT_NE(UbReuseGlobalGraphShard(nullptr, 1, 0, &reused, &info), 0);
  EXPECT_NE(UbReuseGlobalGraphShard("g", 1, 0, nullptr, &info), 0);
  EXPECT_NE(UbReuseGlobalGraphShard("g", 1, 0, &reused, nullptr), 0);
}
TEST_F(CApi, ReuseBeforeSeedIsACleanMiss) {
  int reused = 8;
  UbSeededShardInfo info{};
  ASSERT_EQ(UbReuseGlobalGraphShard("g", 1, 0, &reused, &info), 0);
  EXPECT_EQ(reused, 0);
}
TEST_F(CApi, PublishSnapshotSupportsCapacityQueryAndRejectsSmallBuffer) {
  UbGraphPublishSnapshot snapshot{};
  EXPECT_NE(UbGetGraphPublishSnapshot(0, nullptr, nullptr), 0);
  EXPECT_NE(UbGetGraphPublishSnapshot(0, nullptr, &snapshot), 0);
  ASSERT_EQ(UbSeedFromDir(root_.c_str(), nullptr, 0), 0);
  ASSERT_EQ(UbGetGraphPublishSnapshot(0, nullptr, &snapshot), 0);
  EXPECT_EQ(snapshot.region_count, 2U);
  EXPECT_EQ(snapshot.shard_count, 2U);
  EXPECT_EQ(snapshot.regions, nullptr);
  std::array<UbShardRegion, 2> regions{};
  EXPECT_NE(UbGetGraphPublishSnapshot(1, regions.data(), &snapshot), 0);
  ASSERT_EQ(UbGetGraphPublishSnapshot(2, regions.data(), &snapshot), 0);
  for (const auto &region : regions) {
    EXPECT_NE(region.graph_address, nullptr);
    EXPECT_GT(region.graph_bytes, 0U);
    EXPECT_NE(region.vector_address, nullptr);
    EXPECT_GT(region.vector_bytes, 0U);
  }
  EXPECT_EQ(UbValidateShardLayout(0, 2, snapshot.entry_shard_id,
                                  snapshot.entry_vertex_id),
            0);
  EXPECT_NE(UbValidateShardLayout(0, 3, snapshot.entry_shard_id,
                                  snapshot.entry_vertex_id),
            0);
  EXPECT_NE(UbValidateShardLayout(9, 2, snapshot.entry_shard_id,
                                  snapshot.entry_vertex_id),
            0);
  EXPECT_NE(UbValidateShardLayout(0, 2, snapshot.entry_shard_id, UINT64_MAX),
            0);
}
TEST_F(CApi, RpcMailboxRequiresSeedAndCpuConfigurationThenIsStable) {
  UbRpcMailboxRegions a{}, b{};
  EXPECT_NE(UbEnsureRpcMailbox(nullptr), 0);
  EXPECT_NE(UbEnsureRpcMailbox(&a), 0);
  ASSERT_EQ(UbSeedFromDir(root_.c_str(), nullptr, 0), 0);
  EXPECT_NE(UbEnsureRpcMailbox(&a), 0);
  ConfigureForMailbox();
  ASSERT_EQ(UbEnsureRpcMailbox(&a), 0) << (UbLastError() ? UbLastError() : "");
  ASSERT_EQ(UbEnsureRpcMailbox(&b), 0);
  EXPECT_EQ(a.requests_address, b.requests_address);
  EXPECT_GT(a.requests_bytes, 0U);
  EXPECT_EQ(a.responses_address, b.responses_address);
  EXPECT_GT(a.responses_bytes, 0U);
  EXPECT_NE(a.notify_address, nullptr);
  EXPECT_GT(a.notify_bytes, 0U);
  ASSERT_NE(UbRpcLayout(), nullptr);
  EXPECT_FALSE(std::string(UbRpcLayout()).empty());
}
TEST_F(CApi, MemoryEstimationChecksArgumentsAndProducesPositivePlan) {
  std::uint64_t bytes = 0;
  EXPECT_NE(UbEstimateRuntimeMemory(nullptr, 0, &bytes), 0);
  EXPECT_NE(UbEstimateRuntimeMemory(root_.c_str(), 0, nullptr), 0);
  ConfigureForMailbox();
  ASSERT_EQ(UbEstimateRuntimeMemory(root_.c_str(), 0, &bytes), 0);
  EXPECT_GT(bytes, 0U);
  EXPECT_NE(UbEstimateRuntimeMemory(root_.c_str(), 99, &bytes), 0);
}
TEST_F(CApi, SearchRejectsNullBuffersAndTimeoutBoundaries) {
  const std::array<float, 2> q{1, 2};
  std::int64_t id = 0;
  float distance = 0;
  EXPECT_NE(UbSearch(nullptr, 1, 2, 1, 8, "L2", &id, &distance), 0);
  EXPECT_NE(UbSearch(q.data(), 1, 2, 1, 8, "L2", nullptr, &distance), 0);
  EXPECT_NE(UbSearch(q.data(), 1, 2, 1, 8, "L2", &id, nullptr), 0);
  EXPECT_EQ(UbSearchWithTimeout(q.data(), 1, 2, 1, 8, "L2", 0, &id, &distance),
            int(StatusCode::kTimeout));
  EXPECT_EQ(UbSearchWithTimeout(q.data(), 1, 2, 1, 8, "L2", -2, &id, &distance),
            int(StatusCode::kInvalidArgument));
  EXPECT_NE(UbSearch(q.data(), 1, 2, 1, 8, "L2", &id, &distance), 0);
  EXPECT_EQ(UbEnableDistributedSearchWithTimeout(0), int(StatusCode::kTimeout));
  EXPECT_EQ(UbEnableDistributedSearchWithTimeout(-2),
            int(StatusCode::kInvalidArgument));
  char message[64]{};
  EXPECT_NE(UbEnableDistributedSearch(message, sizeof(message)), 0);
  EXPECT_NE(message[0], '\0');
}
TEST_F(CApi, BorrowedPeerHandlesValidateArgumentsAndAreRemoved) {
  const std::array<std::byte, 4> descriptor{};
  EXPECT_NE(UbImportPeerHandles(1, 0, nullptr, 0, 0, nullptr), 0);
  EXPECT_NE(UbImportPeerHandles(1, 99, descriptor.data(), descriptor.size(), 1,
                                reinterpret_cast<void *>(1)),
            0);
  ASSERT_EQ(UbImportPeerHandles(1, 0, descriptor.data(), descriptor.size(), 1,
                                reinterpret_cast<void *>(1)),
            0);
  EXPECT_EQ(MilvusUbRuntime::Instance().PeerShardCount(), 1U);
  UbRemovePeerHandles(1, 0);
  EXPECT_EQ(MilvusUbRuntime::Instance().PeerShardCount(), 0U);
}
TEST_F(CApi, RuntimeRejectsNullAndInconsistentLoadedIndex) {
  auto &runtime = MilvusUbRuntime::Instance();
  EXPECT_FALSE(runtime.LoadShard(nullptr).ok());
  ASSERT_EQ(UbLoadShardFromDir(root_.c_str(), 0, nullptr), 0);
  auto loaded = LoadIndexForShard(root_, 1);
  ASSERT_TRUE(loaded.ok());
  auto owned = std::make_unique<LoadedIndex>(std::move(loaded).value());
  owned->meta.dim = 3;
  EXPECT_FALSE(runtime.LoadShard(std::move(owned)).ok());
  auto second = LoadIndexForShard(root_, 1);
  ASSERT_TRUE(second.ok());
  auto changed = std::make_unique<LoadedIndex>(std::move(second).value());
  const std::vector<ShardRange> ranges{{0, 0, 5}, {1, 5, 3}};
  auto map = ShardMap::Create(ranges, 8);
  ASSERT_TRUE(map.ok());
  changed->shard_map = std::move(map).value();
  EXPECT_FALSE(runtime.LoadShard(std::move(changed)).ok());
}
TEST_F(CApi, LayoutValidationChecksUnseededWrongOwnerAndEntryBounds) {
  EXPECT_NE(UbValidateShardLayout(0, 2, 0, 0), 0);
  ASSERT_EQ(UbSeedFromDir(root_.c_str(), nullptr, 0), 0);
  EXPECT_NE(UbValidateShardLayout(0, 2, 2, 0), 0);
  EXPECT_NE(UbValidateShardLayout(0, 2, 1, 0), 0);
  EXPECT_NE(UbValidateShardLayout(0, 2, 0, 8), 0);
  EXPECT_EQ(UbValidateShardLayout(0, 2, 0, 0), 0);
}
TEST_F(CApi, KnobsStageOptionalFieldsAndNativeContextWithoutOwnership) {
  const std::array<std::uint32_t, 4> cpus{3, 4, 5, 6};
  UbConfig config{};
  config.cpu_bind_mode = "none";
  config.cpu_layout = "contiguous";
  config.progress_mode = "adaptive";
  config.allowed_cpus = cpus.data();
  config.allowed_cpu_count = cpus.size();
  config.max_active_queries = 7;
  config.read_jfs_depth = 32;
  config.queue_memory_budget_bytes = 1000000;
  config.registered_rpc_memory_budget_bytes = 2000000;
  config.query_timeout_ms = 1000;
  config.operation_timeout_ms = 100;
  config.expand_batch = 4;
  config.runtime_options = "--distance-policy pull";
  config.urma_device_name = "test-device";
  config.urma_eid_index = 3;
  ASSERT_EQ(UbConfigure(&config), 0);
  auto knobs = MilvusUbRuntime::Instance().GetKnobs();
  EXPECT_EQ(knobs.progress_mode, "adaptive");
  EXPECT_EQ(knobs.max_active_queries, 7U);
  EXPECT_EQ(knobs.read_jfs_depth, 32U);
  EXPECT_EQ(knobs.queue_memory_budget_bytes, 1000000U);
  EXPECT_EQ(knobs.registered_rpc_memory_budget_bytes, 2000000U);
  EXPECT_EQ(knobs.query_timeout_ms, 1000U);
  EXPECT_EQ(knobs.operation_timeout_ms, 100U);
  EXPECT_EQ(knobs.expand_batch, 4U);
  EXPECT_EQ(knobs.eid_index, 3U);
  EXPECT_EQ(knobs.runtime_options, "--distance-policy pull");
  EXPECT_EQ(knobs.urma_device, "test-device");
  int borrowed = 0;
  ASSERT_EQ(UbSetUrmaContext(&borrowed), 0);
  EXPECT_EQ(MilvusUbRuntime::Instance().InjectedUrmaContext(), &borrowed);
  EXPECT_FALSE(MilvusUbRuntime::Instance().Ready());
  EXPECT_EQ(UbSetUrmaContext(nullptr), 0);
}
TEST_F(CApi, QueryExecutorValidatesAdmissionBeforePublishingTasks) {
  auto global = LoadGlobalIndexFile(root_);
  ASSERT_TRUE(global.ok());
  auto map =
      ShardMap::Create(global.value().ranges, global.value().meta.ntotal);
  ASSERT_TRUE(map.ok());
  MockGraphReader graph(
      [](node_id_t, std::int32_t) -> Result<std::vector<node_id_t>> {
        return std::vector<node_id_t>{};
      });
  MockDistanceOracle distance(
      [](std::span<const float>, node_id_t) -> Result<float> { return 0.0F; });
  auto queues = SchedulerQueues::Create(1, 2, 2, 2, false, 2);
  ASSERT_TRUE(queues.ok());
  RuntimeConfig config;
  config.max_active_queries = 2;
  EffectiveRuntimeConfig effective;
  effective.search_workers = 1;
  effective.read_progress_threads = 1;
  effective.rpc_progress_threads = 1;
  auto create = [&](const RuntimeConfig &requested) {
    return MilvusQueryExecutor::Create(global.value().meta, map.value(), {0, 1},
                                       graph, distance, requested, effective, 0,
                                       *queues.value(), nullptr, {}, nullptr);
  };
  auto bad = config;
  bad.max_active_queries = 0;
  EXPECT_FALSE(create(bad).ok());
  auto made = create(config);
  ASSERT_TRUE(made.ok());
  auto executor = std::move(made).value();
  const std::array<float, 2> q{1, 2};
  EXPECT_EQ(executor->Submit({}, 0, 1, 8).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_FALSE(executor->Submit(q, 4097, 1, 8).ok());
  EXPECT_FALSE(executor->Submit(std::span(q).first(1), 1, 1, 8).ok());
  EXPECT_FALSE(executor->Submit(q, 1, 0, 8).ok());
  EXPECT_FALSE(executor->Submit(q, 1, 3, 2).ok());
  const std::array<float, 2> nonfinite{std::numeric_limits<float>::quiet_NaN(),
                                       1};
  EXPECT_FALSE(executor->Submit(nonfinite, 1, 1, 8).ok());
  EXPECT_FALSE(executor->Submit(q, 1, 1, UINT32_MAX).ok());
  EXPECT_EQ(executor->Submit(q, 1, 1, 8, std::chrono::steady_clock::now())
                .status()
                .code(),
            StatusCode::kTimeout);
  executor->StopAdmission();
  EXPECT_EQ(executor->Submit(q, 1, 1, 8).status().code(),
            StatusCode::kCancelled);
  executor->FailPending(Status(StatusCode::kCancelled, "test cleanup"));
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_TRUE(executor->Search(stopped.get_token(), 0).ok());
}
