// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "hnsw_urma/index/builder.hpp"
#include "hnsw_urma/milvus_query_executor.hpp"
#include "hnsw_urma/runtime/rpc/transport.hpp"
#include "hnsw_urma/search/remote_io.hpp"
#include "hnsw_urma/transport/mock_lane_post_engine.hpp"
#include "test_framework.hpp"
#include <array>
#include <atomic>
#include <future>
#include <numeric>
#include <thread>

using namespace hnsw_urma;
namespace {
class HostVisibility final : public DmaVisibilityDomain {
public:
  Status AcquireDeviceWrites() override {
    std::atomic_thread_fence(std::memory_order_acquire);
    return Status::Ok();
  }
};

// Real service executor + real READ/RPC protocol and search, with only NIC
// posting replaced by the pinned in-memory transport. Both ranks use the same
// shard/index fixture. This does not claim hardware verification.
void CheckService(DistancePolicy policy, bool dot_norm, Layer0SortMode sort,
                  std::uint32_t topk = 3, std::uint32_t ef = 8,
                  std::uint32_t expand = 3, bool profiling = false,
                  bool delay_first_rpc_response = false) {
  const auto root =
      std::filesystem::temp_directory_path() / "milvus_ub_service_alignment";
  std::filesystem::remove_all(root);
  const std::vector<float> vectors{0, 0, 1, 0, 2, 0, 3, 0,
                                   0, 2, 1, 2, 2, 2, 3, 2};
  ReferenceBuildOptions build;
  build.num_shards = 2;
  build.M = 2;
  build.M0 = 7;
  build.ef_search = 8;
  UB_REQUIRE_TRUE(BuildReferenceIndex(root, vectors, 8, 2, build, 100).ok());
  auto loaded = LoadIndexForShard(root, 0).value();
  auto remote = LoadIndexForShard(root, 1).value();
  RuntimeConfig config;
  config.search_workers = {false, 1};
  config.distance_workers = {false, 1};
  config.read_progress_threads = {false, 1};
  config.rpc_progress_threads = {false, 1};
  config.max_active_queries = 2;
  config.topk = topk;
  config.ef_search = {false, ef};
  config.expand_batch = expand;
  config.use_dot_norm = dot_norm;
  config.distance_policy = policy;
  config.layer0_sort_mode = sort;
  config.cpu_bind = CpuBindMode::kNone;
  config.query_timeout_ms = 2000;
  config.operation_timeout_ms = 1000;
  config.drain_timeout_ms = delay_first_rpc_response ? 50 : 1000;
  config.profile = profiling;
  const std::array<std::uint32_t, 8> cpus{0, 1, 2, 3, 4, 5, 6, 7};
  auto effective =
      ResolveRuntimeConfig(config, loaded.meta, 2, cpus, 0).value();
  auto queues = SchedulerQueues::Create(
                    1, 2, 2, 2, true, effective.rpc_operations.slots_per_search)
                    .value();
  const auto max_batch = effective.maximum_distance_batch;
  const std::array<std::uint32_t, 2> owners0{kNoCommunicationOwner, 0},
      owners1{0, kNoCommunicationOwner}, ranks{0, 1};
  auto rpc = RpcDistanceOracle::Create(
                 1, 1, effective.queues.depths.rpc_submit,
                 effective.rpc_operations.slots_per_search, max_batch, 2, 0,
                 ranks, owners0, 1, *queues, dot_norm)
                 .value();
  const RpcMailboxConfig mailbox{
      2,
      1,
      config.request_ring_depth,
      effective.rpc_request_batch_max_bytes,
      effective.rpc_response_batch_max_bytes,
      effective.rpc_memory.request_staging_slots_per_thread};
  auto mem0 = RpcMailboxMemory::Create(mailbox).value();
  auto mem1 = RpcMailboxMemory::Create(mailbox).value();
  auto rpc0 = MockLanePostEngine::Create(1, 4096).value();
  auto rpc1 = MockLanePostEngine::Create(1, 4096).value();
  for (auto kind : {SegmentKind::kRequests, SegmentKind::kResponses,
                    SegmentKind::kNotify}) {
    auto a = kind == SegmentKind::kRequests    ? mem0->requests()
             : kind == SegmentKind::kResponses ? mem0->responses()
                                               : mem0->notify();
    auto b = kind == SegmentKind::kRequests    ? mem1->requests()
             : kind == SegmentKind::kResponses ? mem1->responses()
                                               : mem1->notify();
    UB_REQUIRE_TRUE(rpc0.RegisterSegment(1, kind, b).ok());
    UB_REQUIRE_TRUE(rpc1.RegisterSegment(0, kind, a).ok());
  }
  auto remote_submissions =
      RpcSubmitMatrix::Create(1, 1, effective.queues.depths.rpc_submit, owners1)
          .value();
  auto client0 =
      RpcClientOwner::Create(0, 2, effective.rpc_operations.total_slots,
                             rpc->submissions(), 2, config.request_ring_depth,
                             true, rpc.get())
          .value();
  auto client1 = RpcClientOwner::Create(
                     0, 2, effective.rpc_operations.total_slots,
                     *remote_submissions, 2, config.request_ring_depth, true)
                     .value();
  auto server0 =
      RpcServerRuntime::Create(*mem0, 1, max_batch, 0, ranks, owners0, 2, 2,
                               true, &loaded.local_shard)
          .value();
  auto server1 =
      RpcServerRuntime::Create(*mem1, 1, max_batch, 1, ranks, owners1, 2, 2,
                               true, &remote.local_shard)
          .value();
  HostVisibility visibility;
  RpcTransportConfig transport;
  transport.transport_lane = 1;
  transport.request_batch_max_tasks = effective.rpc_request_batch_max_tasks;
  transport.request_batch_max_bytes = effective.rpc_request_batch_max_bytes;
  transport.response_progress_max_bytes =
      effective.rpc_response_progress_max_bytes;
  transport.profiling_enabled = true;
  std::array<std::uint32_t, 1> peer0{1}, peer1{0};
  auto tx0 = RpcTransportOwner::Create(transport, peer0, rpc0, visibility,
                                       *mem0, *client0, *server0)
                 .value();
  transport.local_rank = 1;
  auto tx1 = RpcTransportOwner::Create(transport, peer1, rpc1, visibility,
                                       *mem1, *client1, *server1)
                 .value();
  auto read_engine = MockLanePostEngine::Create(0, 4096).value();
  const auto writable = [](std::span<const std::byte> v) {
    return std::span<std::byte>(const_cast<std::byte *>(v.data()), v.size());
  };
  UB_REQUIRE_TRUE(
      read_engine
          .RegisterSegment(1, SegmentKind::kLayer0Graph,
                           writable(remote.local_shard.graph_logical_memory()))
          .ok());
  UB_REQUIRE_TRUE(
      read_engine
          .RegisterSegment(1, SegmentKind::kVectors,
                           writable(remote.local_shard.vector_logical_memory()))
          .ok());
  ReadBatchLaneConfig lane_config;
  lane_config.peer_count = 2;
  lane_config.search_workers = 1;
  lane_config.submit_depth = effective.queues.depths.read_submit;
  lane_config.records_per_search = effective.queues.depths.read_reclaim;
  lane_config.in_flight_limit = effective.urma.read_jfs_depth;
  auto lane = ReadBatchLane::Create(read_engine, 0, lane_config).value();
  std::array<ReadBatchLane *, 1> lanes{lane.get()};
  auto memory =
      PageAlignedBuffer::Create(effective.registered_read_result_bytes).value();
  auto pool =
      SearchReadResources::Create(0, memory.logical_memory(),
                                  effective.read_result_slots_per_search, 28)
          .value();
  UB_REQUIRE_TRUE(pool->BindLanes(lanes).ok());
  std::array<SearchReadResources *, 1> pools{pool.get()};
  const std::array<PeerDataSegments, 1> peers{
      {{1, 1, remote.local_shard.graph_logical_memory().size(),
        remote.local_shard.vector_logical_memory().size()}}};
  auto table =
      PeerSegmentTable::Create(loaded.meta, loaded.shard_map, 0, peers).value();
  auto graph =
      UrmaGraphReader::Create(loaded.meta, loaded.shard_map, loaded.upper_graph,
                              loaded.local_shard, table, lanes, owners0, pools,
                              effective.read_graph_records_per_search)
          .value();
  auto pull = PullVectorDistanceOracle::Create(
                  loaded.meta, loaded.shard_map, loaded.local_shard, table,
                  lanes, owners0, pools, DistanceImplementation::kAuto,
                  effective.read_pull_records_per_search, max_batch, dot_norm)
                  .value();
  auto distance =
      HybridDistanceOracle::Create(0, policy, config.pull_threshold, pull, *rpc)
          .value();
  std::vector<MilvusQueryExecutor::Profile> snapshots;
  auto service =
      MilvusQueryExecutor::Create(
          loaded.meta, loaded.shard_map, loaded.local_entry_point, graph,
          *distance, config, effective, 0, *queues, rpc.get(), pools, nullptr,
          [&](const auto &snapshot) { snapshots.push_back(snapshot); })
          .value();
  std::atomic<bool> process_remote_requests{true};
  const auto pump = [&]() -> Status {
    auto status = lane->Progress();
    if (!status.ok())
      return status;
    // Deliberately complete the latest read first to test stable consumption.
    while (read_engine.pending_events()) {
      status = read_engine.Progress(read_engine.pending_events() - 1);
      if (!status.ok())
        return status;
    }
    status = lane->Progress();
    if (!status.ok())
      return status;
    status = tx0->Progress();
    if (!status.ok())
      return status;
    status = tx1->Progress();
    if (!status.ok())
      return status;
    while (rpc0.pending_events()) {
      status = rpc0.Progress(0);
      if (!status.ok())
        return status;
    }
    while (rpc1.pending_events()) {
      status = rpc1.Progress(0);
      if (!status.ok())
        return status;
    }
    if (process_remote_requests.load(std::memory_order_acquire)) {
      auto processed =
          server1->PollAndProcessRequestSlot(0, Metric::kL2, visibility);
      if (!processed.ok())
        return processed.status();
    }
    auto audited = rpc->AuditOwnerControl(0, *client0);
    if (!audited.ok())
      return audited.status();
    for (std::uint32_t producer = 0; producer < 2; ++producer) {
      auto flushed = service->publisher().Flush(producer, 1);
      if (!flushed.ok())
        return flushed.status();
    }
    return Status::Ok();
  };

  // First run the pinned search directly, with its fixed owner/ready contract.
  const std::vector<float> query{1.1F, 0.2F};
  SearchParams params;
  params.topk = topk;
  params.ef_search = ef;
  params.expand_batch = expand;
  params.visited_capacity = effective.visited_capacity;
  params.layer0_sort_mode = sort;
  params.entry_point = loaded.local_entry_point.global_id;
  params.entry_level = loaded.local_entry_point.level;
  SearchRuntimeContext context{&service->publisher(), 2};
  SetCurrentDistanceSearchWorker(0);
  auto reference =
      HnswSearchTask::Create({0, 0, 100}, loaded.meta, loaded.shard_map, graph,
                             *distance, query, params,
                             std::chrono::steady_clock::now(), &context)
          .value();
  UB_REQUIRE_TRUE(reference->BindOwner(0).ok());
  bool runnable = true;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (reference->run_state() != RunState::kCompleted &&
         reference->run_state() != RunState::kFailed &&
         std::chrono::steady_clock::now() < deadline) {
    if (runnable) {
      (void)reference->Step();
      runnable = false;
      if (auto retry = reference->TakeRetryHandle())
        UB_REQUIRE_TRUE(queues->ScheduleRetry(0, *retry).ok());
    }
    UB_REQUIRE_TRUE(pump().ok());
    (void)pool->PollReclaims();
    while (auto input =
               queues->PopSearchInput(0, std::chrono::steady_clock::now())) {
      if (std::holds_alternative<RpcReclaimHandle>(*input))
        UB_REQUIRE_TRUE(
            rpc->HandleReclaim(0, std::get<RpcReclaimHandle>(*input)).ok());
      else
        runnable = true;
    }
  }
  UB_REQUIRE_EQ(reference->run_state(), RunState::kCompleted);
  const auto expected = reference->result();
  reference.reset();
  while (!rpc->QueryWireQuiesced({0, 0, 100}) || !pool->drained()) {
    UB_REQUIRE_TRUE(std::chrono::steady_clock::now() < deadline);
    UB_REQUIRE_TRUE(pump().ok());
    (void)pool->PollReclaims();
    while (auto input =
               queues->PopSearchInput(0, std::chrono::steady_clock::now())) {
      UB_REQUIRE_TRUE(
          rpc->HandleReclaim(0, std::get<RpcReclaimHandle>(*input)).ok());
    }
  }

  UB_REQUIRE_TRUE(lane->ReleaseProgressOwner().ok());
  Status control_status, search_status, progress_status;
  std::atomic<bool> control_finished{false};
  std::jthread control([&](std::stop_token stop) {
    control_status = service->Control(stop);
    control_finished.store(true, std::memory_order_release);
  });
  std::jthread search(
      [&](std::stop_token stop) { search_status = service->Search(stop, 0); });
  std::jthread progress([&](std::stop_token stop) {
    while (!stop.stop_requested()) {
      progress_status = pump();
      if (!progress_status.ok())
        return;
    }
  });
  if (delay_first_rpc_response) {
    process_remote_requests.store(false, std::memory_order_release);
    auto timed_out = service->Submit(query, 1, topk, ef);
    UB_REQUIRE_FALSE(timed_out.ok());
    UB_REQUIRE_EQ(timed_out.status().code(), StatusCode::kTimeout);
    UB_REQUIRE_TRUE(tx0->profile().request_physical_writes > 0);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config.drain_timeout_ms * 3));
    UB_REQUIRE_FALSE(control_finished.load(std::memory_order_acquire));

    auto second_timed_out = service->Submit(query, 1, topk, ef);
    UB_REQUIRE_FALSE(second_timed_out.ok());
    UB_REQUIRE_EQ(second_timed_out.status().code(), StatusCode::kTimeout);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(config.drain_timeout_ms * 3));
    UB_REQUIRE_FALSE(control_finished.load(std::memory_order_acquire));

    const auto exhausted_started = std::chrono::steady_clock::now();
    auto exhausted = service->Submit(query, 1, topk, ef);
    UB_REQUIRE_FALSE(exhausted.ok());
    UB_REQUIRE_EQ(exhausted.status().code(), StatusCode::kResourceExhausted);
    UB_REQUIRE_TRUE(std::chrono::steady_clock::now() - exhausted_started <
                    std::chrono::milliseconds(100));
    UB_REQUIRE_FALSE(control_finished.load(std::memory_order_acquire));
    process_remote_requests.store(true, std::memory_order_release);
    const auto recovery_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!rpc->drained() &&
           std::chrono::steady_clock::now() < recovery_deadline) {
      std::this_thread::yield();
    }
    UB_REQUIRE_TRUE(rpc->drained());
  }
  auto actual = service->Submit(query, 1, topk, ef);
  if (delay_first_rpc_response && !actual.ok() &&
      actual.status().code() == StatusCode::kResourceExhausted) {
    const auto admission_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!actual.ok() &&
           actual.status().code() == StatusCode::kResourceExhausted &&
           std::chrono::steady_clock::now() < admission_deadline) {
      std::this_thread::yield();
      actual = service->Submit(query, 1, topk, ef);
    }
  }
  // Reuse slots, including scratch replacement when request topk/ef changes.
  auto larger = service->Submit(query, 1, 4, 16);
  auto repeated = service->Submit(query, 1, topk, ef);
  service->StopAdmission();
  const auto stop_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!service->queries_drained() &&
         std::chrono::steady_clock::now() < stop_deadline)
    std::this_thread::yield();
  control.request_stop();
  control.join();
  search.request_stop();
  search.join();
  progress.request_stop();
  progress.join();
  if (!progress_status.ok())
    test::Fail("progress", __FILE__, __LINE__, progress_status.ToString());
  if (!search_status.ok())
    test::Fail("search", __FILE__, __LINE__, search_status.ToString());
  if (!control_status.ok())
    test::Fail("control", __FILE__, __LINE__, control_status.ToString());
  UB_REQUIRE_TRUE(actual.ok());
  UB_REQUIRE_TRUE(larger.ok());
  UB_REQUIRE_TRUE(repeated.ok());
  UB_REQUIRE_EQ(actual.value().params.visited_capacity,
                params.visited_capacity);
  UB_REQUIRE_EQ(actual.value().rows[0].neighbors, expected.neighbors);
  UB_REQUIRE_EQ(repeated.value().rows[0].neighbors, expected.neighbors);
  UB_REQUIRE_EQ(actual.value().rows[0].profile.nodes_expanded,
                expected.profile.nodes_expanded);
  UB_REQUIRE_EQ(actual.value().rows[0].profile.edges_scanned,
                expected.profile.edges_scanned);
  UB_REQUIRE_EQ(actual.value().rows[0].profile.visited_hits,
                expected.profile.visited_hits);
  UB_REQUIRE_EQ(actual.value().rows[0].profile.heap_pushes,
                expected.profile.heap_pushes);
  UB_REQUIRE_EQ(actual.value().rows[0].profile.upper_steps,
                expected.profile.upper_steps);
  UB_REQUIRE_TRUE(lane->profile().requests_completed > 0);
  if (policy == DistancePolicy::kPushdown) {
    UB_REQUIRE_TRUE(tx0->profile().request_physical_writes > 0);
    UB_REQUIRE_TRUE(tx0->profile().response_batches_received > 0);
  }
  UB_REQUIRE_TRUE(service->queries_drained());
  if (delay_first_rpc_response) {
    bool observed_quarantined_slots = false;
    for (const auto &snapshot : snapshots) {
      observed_quarantined_slots =
          observed_quarantined_slots || snapshot.quarantined == 2;
    }
    UB_REQUIRE_TRUE(observed_quarantined_slots);
  }
  if (profiling) {
    UB_REQUIRE_TRUE(!snapshots.empty());
    const auto &snapshot = snapshots.back();
    UB_REQUIRE_EQ(snapshot.completed, 3U);
    UB_REQUIRE_EQ(snapshot.errors, 0U);
    UB_REQUIRE_EQ(snapshot.active + snapshot.pending + snapshot.retiring, 0U);
    std::uint64_t execution = 0, graph_wait = 0, distance_wait = 0, nodes = 0;
    for (const auto *batch :
         {&actual.value(), &larger.value(), &repeated.value()}) {
      const auto &result = batch->rows[0];
      execution += result.timing.LocalExecutionNanoseconds();
      graph_wait += result.timing.graph_wait_nanoseconds;
      distance_wait += result.timing.distance_wait_nanoseconds;
      nodes += result.profile.nodes_expanded;
    }
    UB_REQUIRE_EQ(snapshot.execution_ns, execution);
    UB_REQUIRE_EQ(snapshot.graph_wait_ns, graph_wait);
    UB_REQUIRE_EQ(snapshot.distance_wait_ns, distance_wait);
    UB_REQUIRE_EQ(snapshot.nodes, nodes);
    UB_REQUIRE_TRUE(snapshot.pre_admit_ns > 0);
    UB_REQUIRE_TRUE(snapshot.execution_ns > 0);
    UB_REQUIRE_TRUE(snapshot.active_ns >=
                    snapshot.upper_ns + snapshot.layer0_ns);
  } else if (!delay_first_rpc_response) {
    UB_REQUIRE_TRUE(snapshots.empty());
  }
  std::filesystem::remove_all(root);
}
} // namespace

TEST(ReferenceIntegrationMilvusService,
     MilvusServiceMatchesPinnedDistributedPushdown) {
  CheckService(DistancePolicy::kPushdown, true, Layer0SortMode::kLocalOnly);
}
TEST(ReferenceIntegrationMilvusService, MilvusServiceHonorsPullAndSortOptions) {
  CheckService(DistancePolicy::kPull, false, Layer0SortMode::kAll);
}

TEST(ReferenceIntegrationMilvusService, MilvusServiceReferenceParameters) {
  CheckService(DistancePolicy::kPushdown, true, Layer0SortMode::kLocalOnly, 10,
               350, 20);
}

TEST(ReferenceIntegrationMilvusService,
     MilvusServiceProfilesRealSearchWithoutChangingResults) {
  CheckService(DistancePolicy::kPushdown, true, Layer0SortMode::kLocalOnly, 10,
               350, 20, true);
}

TEST(ReferenceIntegrationMilvusService,
     MilvusServiceQuarantinesExhaustedSlotsUntilWireQuiesces) {
  CheckService(DistancePolicy::kPushdown, true, Layer0SortMode::kLocalOnly, 3,
               8, 3, false, true);
}
