// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/runtime/rpc/transport.hpp"
#include "hnsw_urma/transport/mock_lane_post_engine.hpp"

namespace hnsw_urma {
namespace {

class HostVisibility final : public DmaVisibilityDomain {
public:
  Status AcquireDeviceWrites() override {
    std::atomic_thread_fence(std::memory_order_acquire);
    return Status::Ok();
  }
};

Result<LocalShard> MakeShard(std::uint32_t rank) {
  GlobalIndexMeta meta;
  meta.ntotal = 14;
  meta.dim = 2;
  meta.metric = static_cast<std::uint32_t>(Metric::kL2);
  meta.M = 1;
  meta.M0 = 2;
  meta.ef_search_default = 4;
  meta.entry_point = 0;
  meta.max_level = 0;
  meta.num_shards = 2;
  std::vector<node_id_t> layer0(8, INVALID_NODE_ID);
  std::vector<float> vectors =
      rank == 0
          ? std::vector<float>{0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F}
          : std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2.0F, 0.0F};
  const ShardRange range{rank, rank == 0 ? 0U : 10U, 4};
  return LocalShard::Create(meta, range, std::move(layer0), std::move(vectors));
}

bool ProgressOneOutOfOrder(MockLanePostEngine &engine) {
  if (engine.pending_events() == 0)
    return true;
  const auto index = engine.pending_events() > 1 ? 1U : 0U;
  return engine.Progress(index).ok();
}

// Let the peer and distance worker progress before a provider post returns.
class PostingWindowEngine final : public LanePostEngine {
public:
  MockLanePostEngine *backend{};
  std::function<void()> after_response;

  Status PostRead(const ReadOperation &operation) override {
    return backend->PostRead(operation);
  }
  Status PostWriteNotify(const WriteNotifyOperation &operation) override {
    auto status = backend->PostWriteNotify(operation);
    if (status.ok() &&
        operation.operation_class == OperationClass::kResponseWrite &&
        after_response) {
      after_response();
    }
    return status;
  }
  Result<std::size_t>
  Poll(std::span<TransportCompletion> completions) override {
    return backend->Poll(completions);
  }
};

struct TwoRankFixture {
  std::unique_ptr<RpcMailboxMemory> memory0;
  std::unique_ptr<RpcMailboxMemory> memory1;
  MockLanePostEngine engine0;
  MockLanePostEngine engine1;
  PostingWindowEngine posting_window;
  HostVisibility visibility0;
  HostVisibility visibility1;
  LocalShard shard0;
  LocalShard shard1;
  std::unique_ptr<RpcSubmitMatrix> submissions0;
  std::unique_ptr<RpcSubmitMatrix> submissions1;
  std::unique_ptr<RpcClientOwner> client0;
  std::unique_ptr<RpcClientOwner> client1;
  std::unique_ptr<RpcServerRuntime> server0;
  std::unique_ptr<RpcServerRuntime> server1;
  std::unique_ptr<RpcTransportOwner> transport0;
  std::unique_ptr<RpcTransportOwner> transport1;
};

Result<std::unique_ptr<TwoRankFixture>>
MakeTwoRankFixture(std::uint32_t ring_depth = 2) {
  auto fixture = std::make_unique<TwoRankFixture>();
  const auto capacity = std::max(8U, 2U * ring_depth);
  const RpcMailboxConfig mailbox_config{2, 1, ring_depth, 1024, 1024, 2};
  auto memory0 = RpcMailboxMemory::Create(mailbox_config);
  auto memory1 = RpcMailboxMemory::Create(mailbox_config);
  auto engine0 = MockLanePostEngine::Create(0, std::max(32U, capacity));
  auto engine1 = MockLanePostEngine::Create(0, std::max(32U, capacity));
  auto shard0 = MakeShard(0);
  auto shard1 = MakeShard(1);
  if (!memory0.ok())
    return memory0.status();
  if (!memory1.ok())
    return memory1.status();
  if (!engine0.ok())
    return engine0.status();
  if (!engine1.ok())
    return engine1.status();
  if (!shard0.ok())
    return shard0.status();
  if (!shard1.ok())
    return shard1.status();
  fixture->memory0 = std::move(memory0).value();
  fixture->memory1 = std::move(memory1).value();
  fixture->engine0 = std::move(engine0).value();
  fixture->engine1 = std::move(engine1).value();
  fixture->shard0 = std::move(shard0).value();
  fixture->shard1 = std::move(shard1).value();
  for (const auto kind : {SegmentKind::kRequests, SegmentKind::kResponses,
                          SegmentKind::kNotify}) {
    const auto remote1 =
        kind == SegmentKind::kRequests    ? fixture->memory1->requests()
        : kind == SegmentKind::kResponses ? fixture->memory1->responses()
                                          : fixture->memory1->notify();
    const auto remote0 =
        kind == SegmentKind::kRequests    ? fixture->memory0->requests()
        : kind == SegmentKind::kResponses ? fixture->memory0->responses()
                                          : fixture->memory0->notify();
    auto status = fixture->engine0.RegisterSegment(1, kind, remote1);
    if (!status.ok())
      return status;
    status = fixture->engine1.RegisterSegment(0, kind, remote0);
    if (!status.ok())
      return status;
  }
  const std::array<std::uint32_t, 2> owner0{kNoCommunicationOwner, 0};
  const std::array<std::uint32_t, 2> owner1{0, kNoCommunicationOwner};
  const std::array<std::uint32_t, 2> shard_to_rank{0, 1};
  auto submissions0 = RpcSubmitMatrix::Create(1, 1, capacity, owner0);
  auto submissions1 = RpcSubmitMatrix::Create(1, 1, capacity, owner1);
  if (!submissions0.ok())
    return submissions0.status();
  if (!submissions1.ok())
    return submissions1.status();
  fixture->submissions0 = std::move(submissions0).value();
  fixture->submissions1 = std::move(submissions1).value();
  auto client0 = RpcClientOwner::Create(0, 2, capacity, *fixture->submissions0,
                                        capacity, ring_depth, true);
  auto client1 = RpcClientOwner::Create(0, 2, capacity, *fixture->submissions1,
                                        capacity, ring_depth, true);
  if (!client0.ok())
    return client0.status();
  if (!client1.ok())
    return client1.status();
  fixture->client0 = std::move(client0).value();
  fixture->client1 = std::move(client1).value();
  auto server0 =
      RpcServerRuntime::Create(*fixture->memory0, 1, 8, 0, shard_to_rank,
                               owner0, capacity, 2, true, &fixture->shard0);
  auto server1 =
      RpcServerRuntime::Create(*fixture->memory1, 1, 8, 1, shard_to_rank,
                               owner1, capacity, 2, true, &fixture->shard1);
  if (!server0.ok())
    return server0.status();
  if (!server1.ok())
    return server1.status();
  fixture->server0 = std::move(server0).value();
  fixture->server1 = std::move(server1).value();
  RpcTransportConfig config;
  config.owner_thread = 0;
  config.transport_lane = 0;
  config.request_batch_max_tasks = 1;
  config.request_batch_max_bytes = 1024;
  config.poll_batch = 16;
  config.phase_budget = 16;
  config.profiling_enabled = true;
  const std::array<std::uint32_t, 1> peer0{1};
  const std::array<std::uint32_t, 1> peer1{0};
  config.local_rank = 0;
  auto transport0 = RpcTransportOwner::Create(
      config, peer0, fixture->engine0, fixture->visibility0, *fixture->memory0,
      *fixture->client0, *fixture->server0);
  config.local_rank = 1;
  fixture->posting_window.backend = &fixture->engine1;
  auto transport1 = RpcTransportOwner::Create(
      config, peer1, fixture->posting_window, fixture->visibility1,
      *fixture->memory1, *fixture->client1, *fixture->server1);
  if (!transport0.ok())
    return transport0.status();
  if (!transport1.ok())
    return transport1.status();
  fixture->transport0 = std::move(transport0).value();
  fixture->transport1 = std::move(transport1).value();
  return fixture;
}

bool Drive(TwoRankFixture &fixture, std::span<RpcClientTask> tasks) {
  for (std::uint32_t iteration = 0; iteration < 1000; ++iteration) {
    if (!fixture.transport0->Progress().ok() ||
        !fixture.transport1->Progress().ok() ||
        !ProgressOneOutOfOrder(fixture.engine0) ||
        !ProgressOneOutOfOrder(fixture.engine1)) {
      return false;
    }
    auto rank0 = fixture.server0->PollAndProcessRequestSlot(
        0, Metric::kL2, fixture.visibility0);
    auto rank1 = fixture.server1->PollAndProcessRequestSlot(
        0, Metric::kL2, fixture.visibility1);
    if (!rank0.ok() || !rank1.ok())
      return false;
    bool done = true;
    for (const auto &task : tasks) {
      done = done && task.state.load() == RpcClientTaskState::kTerminal;
    }
    if (done)
      return true;
  }
  return false;
}

} // namespace

TEST(ReferenceRuntimeRpcTransport,
     TransportUsesAnyFreeSlotAndAcceptsOutOfOrderCompletions) {
  auto fixture_result = MakeTwoRankFixture();
  UB_REQUIRE_TRUE(fixture_result.ok());
  auto fixture = std::move(fixture_result).value();
  const std::array<float, 2> query{1.0F, 1.0F};
  const std::array<std::array<node_id_t, 1>, 4> ids{
      {{{10}}, {{11}}, {{12}}, {{13}}}};
  std::array<std::array<float, 1>, 4> outputs{};
  std::array<RpcClientTask, 4> tasks;
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    UB_REQUIRE_TRUE(tasks[index]
                        .Prepare(static_cast<ticket_t>(index + 1),
                                 {0, static_cast<std::uint32_t>(index), 1},
                                 index + 1, 1, 1, ids[index], query,
                                 outputs[index])
                        .ok());
    UB_REQUIRE_TRUE(fixture->submissions0->Publish(0, &tasks[index]).ok());
  }
  UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
  const auto first_progress_profile = fixture->transport0->profile();
  UB_REQUIRE_EQ(first_progress_profile.request_post_peer_visits, 1U);
  UB_REQUIRE_EQ(first_progress_profile.request_physical_writes, 1U);
  UB_REQUIRE_EQ(first_progress_profile.request_post_free_slot_waits, 0U);
  UB_REQUIRE_TRUE(Drive(*fixture, tasks));
  UB_REQUIRE_NEAR(outputs[0][0], 1.0F, 1e-6F);
  UB_REQUIRE_NEAR(outputs[1][0], 1.0F, 1e-6F);
  UB_REQUIRE_NEAR(outputs[2][0], 0.0F, 1e-6F);
  UB_REQUIRE_NEAR(outputs[3][0], 2.0F, 1e-6F);
  for (auto &task : tasks)
    task.Reset();
  for (std::uint32_t iteration = 0; iteration < 20; ++iteration) {
    UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
    UB_REQUIRE_TRUE(fixture->transport1->Progress().ok());
    (void)ProgressOneOutOfOrder(fixture->engine0);
    (void)ProgressOneOutOfOrder(fixture->engine1);
  }
  UB_REQUIRE_TRUE(fixture->transport0->drained());
  UB_REQUIRE_TRUE(fixture->transport1->drained());
  const auto profile = fixture->transport0->profile();
  UB_REQUIRE_EQ(profile.request_physical_writes, 4U);
  UB_REQUIRE_EQ(profile.response_batches_received, 4U);
  UB_REQUIRE_EQ(profile.request_slots_high_watermark, 2U);
  const auto server_profile = fixture->transport1->profile();
  UB_REQUIRE_EQ(server_profile.request_notify_observed_to_response_post_samples,
                4U);
  UB_REQUIRE_TRUE(
      server_profile.request_notify_observed_to_response_post_nanoseconds > 0);
  const auto client_profile = fixture->client0->profile();
  UB_REQUIRE_EQ(client_profile.response_notify_to_receive_task_samples, 4U);
}

TEST(ReferenceRuntimeRpcTransport,
     TransportTreatsWrongResponseSequenceAsFatal) {
  auto fixture_result = MakeTwoRankFixture();
  UB_REQUIRE_TRUE(fixture_result.ok());
  auto fixture = std::move(fixture_result).value();
  const std::array<float, 2> query{1.0F, 1.0F};
  const std::array<node_id_t, 1> ids{10};
  std::array<float, 1> output{};
  RpcClientTask task;
  UB_REQUIRE_TRUE(task.Prepare(1, {0, 0, 1}, 1, 1, 1, ids, query, output).ok());
  UB_REQUIRE_TRUE(fixture->submissions0->Publish(0, &task).ok());
  UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
  auto *notify = fixture->memory0->NotifyAddress(
      fixture->memory0->ResponseNotifyOffset(1, 0));
  *notify = 99;
  auto status = fixture->transport0->Progress();
  UB_REQUIRE_FALSE(status.ok());
  UB_REQUIRE_EQ(status.code(), StatusCode::kProtocolError);
  UB_REQUIRE_FALSE(fixture->transport0->fatal_status().ok());
}

TEST(ReferenceRuntimeRpcTransport,
     ResponseVisibleBeforePostReturnsAllowsSlotReuse) {
  for (const auto depth : {2U, 32U}) {
    auto created = MakeTwoRankFixture(depth);
    UB_REQUIRE_TRUE(created.ok());
    auto fixture = std::move(created).value();
    const std::array<float, 2> query{1.0F, 1.0F};
    const std::array<node_id_t, 1> ids{10};
    std::vector<std::array<float, 1>> outputs(depth + 1);
    std::vector<RpcClientTask> tasks(depth + 1);
    for (std::uint32_t i = 0; i < tasks.size(); ++i) {
      UB_REQUIRE_TRUE(
          tasks[i]
              .Prepare(i + 1, {0, i, 1}, i + 1, 1, 1, ids, query, outputs[i])
              .ok());
      UB_REQUIRE_TRUE(fixture->submissions0->Publish(0, &tasks[i]).ok());
    }
    // Occupy every physical slot; the extra task must reuse a returned slot.
    for (std::uint32_t i = 0; i < depth; ++i) {
      UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
      UB_REQUIRE_TRUE(fixture->engine0.Progress().ok());
      auto processed = fixture->server1->PollAndProcessRequestSlot(
          0, Metric::kL2, fixture->visibility1);
      UB_REQUIRE_TRUE(processed.ok() && processed.value());
    }
    bool reused_before_post_returned = false;
    fixture->posting_window.after_response = [&] {
      if (reused_before_post_returned)
        return;
      reused_before_post_returned = true;
      UB_REQUIRE_TRUE(fixture->engine1.Progress().ok());
      UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
      UB_REQUIRE_EQ(tasks[0].state.load(), RpcClientTaskState::kTerminal);
      UB_REQUIRE_EQ(fixture->engine0.pending_events(), 1U);
      UB_REQUIRE_TRUE(fixture->engine0.Progress().ok());
      auto blocked = fixture->server1->PollAndProcessRequestSlot(
          0, Metric::kL2, fixture->visibility1);
      UB_REQUIRE_TRUE(blocked.ok());
      UB_REQUIRE_FALSE(blocked.value());
      UB_REQUIRE_EQ(fixture->server1->result_state({0, 0}),
                    ResultSlotState::kInFlight);
      UB_REQUIRE_TRUE(*fixture->memory1->NotifyAddress(
                          fixture->memory1->RequestNotifyOffset(0, 0)) != 0);
    };
    UB_REQUIRE_TRUE(fixture->transport1->Progress().ok());
    UB_REQUIRE_TRUE(reused_before_post_returned);
    UB_REQUIRE_TRUE(Drive(*fixture, tasks));
    for (const auto &output : outputs)
      UB_REQUIRE_NEAR(output[0], 1.0F, 1e-6F);
    for (std::uint32_t i = 0; i < depth + 2; ++i) {
      UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
      UB_REQUIRE_TRUE(fixture->transport1->Progress().ok());
      UB_REQUIRE_TRUE(ProgressOneOutOfOrder(fixture->engine0));
      UB_REQUIRE_TRUE(ProgressOneOutOfOrder(fixture->engine1));
    }
    UB_REQUIRE_TRUE(fixture->transport0->drained());
    UB_REQUIRE_TRUE(fixture->transport1->drained());
    UB_REQUIRE_EQ(fixture->transport1->profile().response_physical_writes,
                  depth + 1U);
  }
}

TEST(ReferenceRuntimeRpcTransport,
     ResponsePostBackpressureRestoresReadyAndRetriesOnceAccepted) {
  auto created = MakeTwoRankFixture();
  UB_REQUIRE_TRUE(created.ok());
  auto fixture = std::move(created).value();
  const std::array<float, 2> query{1.0F, 1.0F};
  const std::array<node_id_t, 1> ids{10};
  std::array<float, 1> output{};
  RpcClientTask task;
  UB_REQUIRE_TRUE(task.Prepare(1, {0, 0, 1}, 1, 1, 1, ids, query, output).ok());
  UB_REQUIRE_TRUE(fixture->submissions0->Publish(0, &task).ok());
  UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
  UB_REQUIRE_TRUE(fixture->engine0.Progress().ok());
  auto processed = fixture->server1->PollAndProcessRequestSlot(
      0, Metric::kL2, fixture->visibility1);
  UB_REQUIRE_TRUE(processed.ok() && processed.value());
  for (int attempt = 0; attempt < 4; ++attempt) {
    fixture->engine1.FailNextPost(
        Status(StatusCode::kResourceExhausted, "full send queue"));
    UB_REQUIRE_TRUE(fixture->transport1->Progress().ok());
    UB_REQUIRE_EQ(fixture->server1->result_state({0, 0}),
                  ResultSlotState::kReady);
    auto ready = fixture->server1->TryGetReadyResult(0);
    UB_REQUIRE_TRUE(ready.ok() && ready.value().has_value());
    UB_REQUIRE_EQ(ready.value()->sequence, 1U);
    UB_REQUIRE_EQ(fixture->engine1.pending_events(), 0U);
  }
  UB_REQUIRE_EQ(fixture->transport1->profile().result_post_retries, 4U);
  UB_REQUIRE_TRUE(Drive(*fixture, std::span(&task, 1)));
  UB_REQUIRE_NEAR(output[0], 1.0F, 1e-6F);
  for (int i = 0; i < 4; ++i) {
    UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
    UB_REQUIRE_TRUE(fixture->transport1->Progress().ok());
    UB_REQUIRE_TRUE(ProgressOneOutOfOrder(fixture->engine0));
    UB_REQUIRE_TRUE(ProgressOneOutOfOrder(fixture->engine1));
  }
  UB_REQUIRE_TRUE(fixture->transport0->drained());
  UB_REQUIRE_TRUE(fixture->transport1->drained());
  UB_REQUIRE_EQ(fixture->transport1->profile().response_physical_writes, 1U);
  UB_REQUIRE_EQ(fixture->transport0->profile().response_batches_received, 1U);
}

TEST(ReferenceRuntimeRpcTransport,
     ResponsePostFailurePreservesFatalStatusAndSendLease) {
  for (const auto code :
       {StatusCode::kTransportError, StatusCode::kProtocolError}) {
    auto created = MakeTwoRankFixture();
    UB_REQUIRE_TRUE(created.ok());
    auto fixture = std::move(created).value();
    const std::array<float, 2> query{1.0F, 1.0F};
    const std::array<node_id_t, 1> ids{10};
    std::array<float, 1> output{};
    RpcClientTask task;
    UB_REQUIRE_TRUE(
        task.Prepare(1, {0, 0, 1}, 1, 1, 1, ids, query, output).ok());
    UB_REQUIRE_TRUE(fixture->submissions0->Publish(0, &task).ok());
    UB_REQUIRE_TRUE(fixture->transport0->Progress().ok());
    UB_REQUIRE_TRUE(fixture->engine0.Progress().ok());
    auto processed = fixture->server1->PollAndProcessRequestSlot(
        0, Metric::kL2, fixture->visibility1);
    UB_REQUIRE_TRUE(processed.ok() && processed.value());
    fixture->engine1.FailNextPost(
        Status(code, "injected response post failure"));
    const auto failed = fixture->transport1->Progress();
    UB_REQUIRE_EQ(failed.code(), code);
    UB_REQUIRE_EQ(failed.message(), "injected response post failure");
    UB_REQUIRE_EQ(fixture->server1->result_state({0, 0}),
                  ResultSlotState::kInFlight);
    UB_REQUIRE_EQ(fixture->transport1->Progress().code(), code);
    UB_REQUIRE_FALSE(fixture->transport1->drained());
    UB_REQUIRE_EQ(fixture->transport1->profile().response_physical_writes, 0U);
  }
}

} // namespace hnsw_urma
