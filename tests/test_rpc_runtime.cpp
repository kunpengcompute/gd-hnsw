// Migrated from Milvus ub_alignment; source identity in
// manifests/reference-tests.json.
#include "test_framework.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include "hnsw_urma/index/format.hpp"
#include "hnsw_urma/runtime/rpc_runtime.hpp"
#include "hnsw_urma/transport/urma_resources.hpp"

namespace hnsw_urma {
namespace {

template <std::size_t N> struct AlignedBuffer {
  alignas(8) std::array<std::byte, N> bytes{};
};

class HostVisibility final : public DmaVisibilityDomain {
public:
  Status AcquireDeviceWrites() override {
    std::atomic_thread_fence(std::memory_order_acquire);
    return Status::Ok();
  }
};

Result<LocalShard> MakeShard() {
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
  std::vector<float> vectors{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2.0F, 0.0F};
  return LocalShard::Create(meta, {1, 10, 4}, std::move(layer0),
                            std::move(vectors));
}

struct ClientFixture {
  std::unique_ptr<RpcSubmitMatrix> submissions;
  std::unique_ptr<RpcClientOwner> client;
};

Result<ClientFixture> MakeClient(std::uint32_t ring_depth = 4) {
  const std::array<std::uint32_t, 2> owners{kNoCommunicationOwner, 0};
  auto submissions = RpcSubmitMatrix::Create(1, 1, 8, owners);
  if (!submissions.ok())
    return submissions.status();
  auto client = RpcClientOwner::Create(0, 2, 8, *submissions.value(), 8,
                                       ring_depth, true);
  if (!client.ok())
    return client.status();
  return ClientFixture{std::move(submissions).value(),
                       std::move(client).value()};
}

Status PublishTask(ClientFixture &fixture, RpcClientTask &task) {
  auto published = fixture.submissions->Publish(0, &task);
  if (!published.ok())
    return published;
  auto drained = fixture.client->DrainSubmissions();
  return drained.ok() ? Status::Ok() : drained.status();
}

} // namespace

TEST(ReferenceRuntimeRpcRuntime, ActivePeerSetMaintainsDenseMembership) {
  auto active = ActivePeerSet::Create(8);
  UB_REQUIRE_TRUE(active.ok());
  UB_REQUIRE_TRUE(active.value().Activate(2));
  UB_REQUIRE_TRUE(active.value().Activate(5));
  UB_REQUIRE_FALSE(active.value().Activate(2));
  UB_REQUIRE_TRUE(active.value().Deactivate(2));
  UB_REQUIRE_FALSE(active.value().contains(2));
  UB_REQUIRE_TRUE(active.value().contains(5));
  UB_REQUIRE_EQ(active.value().peers().size(), 1U);
}

TEST(ReferenceRuntimeRpcRuntime, RpcClientCompletesPhysicalSlotsOutOfOrder) {
  auto fixture_result = MakeClient();
  UB_REQUIRE_TRUE(fixture_result.ok());
  auto fixture = std::move(fixture_result).value();
  const std::array<node_id_t, 1> ids0{10};
  const std::array<node_id_t, 1> ids1{11};
  const std::array<node_id_t, 1> ids2{12};
  const std::array<float, 2> query{1.0F, 1.0F};
  std::array<float, 1> out0{};
  std::array<float, 1> out1{};
  std::array<float, 1> out2{};
  std::array<RpcClientTask, 3> tasks;
  UB_REQUIRE_TRUE(
      tasks[0].Prepare(11, {0, 0, 1}, 1, 1, 1, ids0, query, out0).ok());
  UB_REQUIRE_TRUE(
      tasks[1].Prepare(12, {0, 1, 1}, 1, 1, 1, ids1, query, out1).ok());
  UB_REQUIRE_TRUE(
      tasks[2].Prepare(13, {0, 2, 1}, 1, 1, 1, ids2, query, out2).ok());
  for (auto &task : tasks) {
    UB_REQUIRE_TRUE(fixture.submissions->Publish(0, &task).ok());
  }
  UB_REQUIRE_EQ(fixture.client->DrainSubmissions().value(), 3U);

  std::array<AlignedBuffer<512>, 3> requests;
  for (std::uint32_t slot = 0; slot < 3; ++slot) {
    auto prepared = fixture.client->PrepareRequestBatch(
        1, slot, 100 + slot, 1, 512, 512, requests[slot].bytes);
    UB_REQUIRE_TRUE(prepared.ok() && prepared.value().has_value());
    UB_REQUIRE_EQ(prepared.value()->slot_id, slot);
    UB_REQUIRE_TRUE(
        fixture.client->CommitPreparedRequest(*prepared.value()).ok());
  }

  const std::array<std::uint32_t, 3> completion_order{2, 0, 1};
  for (const auto slot : completion_order) {
    const std::array<float, 1> distance{static_cast<float>(slot + 1)};
    const DistanceBatchResponseTaskInput response{tasks[slot].ticket,
                                                  tasks[slot].query.query_slot,
                                                  distance, tasks[slot].ids};
    AlignedBuffer<512> wire;
    auto encoded = EncodeDistanceBatchResponse(
        100 + slot,
        std::span<const DistanceBatchResponseTaskInput>(&response, 1),
        wire.bytes);
    UB_REQUIRE_TRUE(encoded.ok());
    UB_REQUIRE_TRUE(
        fixture.client
            ->AcceptResponse(
                1, slot, 100 + slot,
                std::span<const std::byte>(wire.bytes).first(encoded.value()))
            .ok());
  }
  UB_REQUIRE_EQ(out0[0], 1.0F);
  UB_REQUIRE_EQ(out1[0], 2.0F);
  UB_REQUIRE_EQ(out2[0], 3.0F);
  UB_REQUIRE_EQ(fixture.client->waiting_response_tasks(1), 0U);
  for (auto &task : tasks) {
    task.Reset();
    task.Reset();
  }
  UB_REQUIRE_TRUE(fixture.client->drained());
}

TEST(ReferenceRuntimeRpcRuntime,
     RpcClientProfilesResponseNotifyDelayAtBatchAndTaskGranularity) {
  auto fixture_result = MakeClient();
  UB_REQUIRE_TRUE(fixture_result.ok());
  auto fixture = std::move(fixture_result).value();
  const std::array<float, 2> query{1.0F, 1.0F};
  const std::array<node_id_t, 1> ids0{10};
  const std::array<node_id_t, 1> ids1{11};
  std::array<float, 1> output0{};
  std::array<float, 1> output1{};
  std::array<RpcClientTask, 2> tasks;
  UB_REQUIRE_TRUE(
      tasks[0].Prepare(31, {0, 0, 1}, 1, 1, 1, ids0, query, output0).ok());
  UB_REQUIRE_TRUE(
      tasks[1].Prepare(32, {0, 1, 1}, 1, 1, 1, ids1, query, output1).ok());
  for (auto &task : tasks) {
    UB_REQUIRE_TRUE(fixture.submissions->Publish(0, &task).ok());
  }
  UB_REQUIRE_EQ(fixture.client->DrainSubmissions().value(), 2U);

  AlignedBuffer<512> request_wire;
  auto prepared = fixture.client->PrepareRequestBatch(1, 0, 80, 2, 512, 512,
                                                      request_wire.bytes);
  UB_REQUIRE_TRUE(prepared.ok() && prepared.value().has_value());
  UB_REQUIRE_EQ(prepared.value()->task_count, 2U);
  UB_REQUIRE_TRUE(
      fixture.client->CommitPreparedRequest(*prepared.value()).ok());

  const std::array<float, 1> distance0{1.0F};
  const std::array<float, 1> distance1{2.0F};
  const std::array<DistanceBatchResponseTaskInput, 2> responses{{
      {31, 0, distance0, ids0},
      {32, 1, distance1, ids1},
  }};
  AlignedBuffer<512> response_wire;
  auto response_bytes =
      EncodeDistanceBatchResponse(80, responses, response_wire.bytes);
  UB_REQUIRE_TRUE(response_bytes.ok());
  const auto notify_observed = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  UB_REQUIRE_TRUE(
      fixture.client
          ->AcceptResponse(1, 0, 80,
                           std::span<const std::byte>(response_wire.bytes)
                               .first(response_bytes.value()),
                           notify_observed)
          .ok());
  const auto profile = fixture.client->profile();
  UB_REQUIRE_EQ(profile.response_notify_to_receive_samples, 1U);
  UB_REQUIRE_EQ(profile.response_notify_to_receive_task_samples, 2U);
  UB_REQUIRE_EQ(profile.response_notify_to_receive_task_nanoseconds,
                profile.response_notify_to_receive_nanoseconds * 2U);
  for (auto &task : tasks)
    task.Reset();
  UB_REQUIRE_TRUE(fixture.client->drained());
}

TEST(ReferenceRuntimeRpcRuntime,
     RpcClientRejectsWrongSlotAndCapsWorstCaseResponse) {
  auto fixture_result = MakeClient();
  UB_REQUIRE_TRUE(fixture_result.ok());
  auto fixture = std::move(fixture_result).value();
  const std::array<node_id_t, 4> ids{10, 11, 12, 13};
  const std::array<float, 2> query{1.0F, 1.0F};
  std::array<float, 4> output{};
  RpcClientTask task;
  UB_REQUIRE_TRUE(
      task.Prepare(11, {0, 0, 1}, 1, 1, 1, ids, query, output).ok());
  UB_REQUIRE_TRUE(PublishTask(fixture, task).ok());
  AlignedBuffer<512> request;
  const auto response_bytes = DistanceBatchResponseBytesForShape(1, 4).value();
  auto too_small = fixture.client->PrepareRequestBatch(
      1, 1, 50, 1, 512, static_cast<std::uint32_t>(response_bytes - 1),
      request.bytes);
  UB_REQUIRE_FALSE(too_small.ok());

  auto prepared = fixture.client->PrepareRequestBatch(
      1, 1, 50, 1, 512, static_cast<std::uint32_t>(response_bytes),
      request.bytes);
  UB_REQUIRE_TRUE(prepared.ok() && prepared.value().has_value());
  UB_REQUIRE_TRUE(
      fixture.client->CommitPreparedRequest(*prepared.value()).ok());
  const std::array<float, 4> distances{1.0F, 2.0F, 3.0F, 4.0F};
  const DistanceBatchResponseTaskInput response{11, 0, distances, ids};
  AlignedBuffer<512> wire;
  auto encoded = EncodeDistanceBatchResponse(
      50, std::span<const DistanceBatchResponseTaskInput>(&response, 1),
      wire.bytes);
  UB_REQUIRE_TRUE(encoded.ok());
  const auto payload =
      std::span<const std::byte>(wire.bytes).first(encoded.value());
  UB_REQUIRE_FALSE(fixture.client->AcceptResponse(1, 0, 50, payload).ok());
  UB_REQUIRE_TRUE(fixture.client->AcceptResponse(1, 1, 50, payload).ok());
}

TEST(ReferenceRuntimeRpcRuntime,
     RpcClientAllowsSameQueryAcrossInflightBatchesAndSendsVectorOnce) {
  auto fixture_result = MakeClient();
  UB_REQUIRE_TRUE(fixture_result.ok());
  auto fixture = std::move(fixture_result).value();
  const std::array<node_id_t, 1> ids0{10};
  const std::array<node_id_t, 1> ids1{11};
  const std::array<float, 2> query{1.0F, 1.0F};
  std::array<float, 1> output0{};
  std::array<float, 1> output1{};
  std::array<RpcClientTask, 2> tasks;
  UB_REQUIRE_TRUE(
      tasks[0].Prepare(21, {0, 3, 7}, 1, 1, 1, ids0, query, output0).ok());
  UB_REQUIRE_TRUE(
      tasks[1].Prepare(22, {0, 3, 7}, 1, 1, 1, ids1, query, output1).ok());
  for (auto &task : tasks) {
    UB_REQUIRE_TRUE(fixture.submissions->Publish(0, &task).ok());
  }
  UB_REQUIRE_EQ(fixture.client->DrainSubmissions().value(), 2U);

  AlignedBuffer<512> first_wire;
  auto first = fixture.client->PrepareRequestBatch(1, 0, 70, 1, 512, 512,
                                                   first_wire.bytes);
  UB_REQUIRE_TRUE(first.ok() && first.value().has_value());
  auto first_request = DecodeDistanceBatchRequest(
      std::span<const std::byte>(first_wire.bytes).first(first.value()->bytes));
  UB_REQUIRE_TRUE(first_request.ok());
  auto first_task = first_request.value().Task(0);
  UB_REQUIRE_TRUE(first_task.ok());
  UB_REQUIRE_TRUE(
      (first_task.value().descriptor.flags & kBatchHasQueryVector) != 0);
  UB_REQUIRE_EQ(first_task.value().query_vector.size(), query.size());
  UB_REQUIRE_TRUE(fixture.client->CommitPreparedRequest(*first.value()).ok());

  AlignedBuffer<512> second_wire;
  auto second = fixture.client->PrepareRequestBatch(1, 1, 71, 1, 512, 512,
                                                    second_wire.bytes);
  UB_REQUIRE_TRUE(second.ok() && second.value().has_value());
  auto second_request =
      DecodeDistanceBatchRequest(std::span<const std::byte>(second_wire.bytes)
                                     .first(second.value()->bytes));
  UB_REQUIRE_TRUE(second_request.ok());
  auto second_task = second_request.value().Task(0);
  UB_REQUIRE_TRUE(second_task.ok());
  UB_REQUIRE_FALSE(
      (second_task.value().descriptor.flags & kBatchHasQueryVector) != 0);
  UB_REQUIRE_TRUE(second_task.value().query_vector.empty());
  UB_REQUIRE_TRUE(fixture.client->CommitPreparedRequest(*second.value()).ok());

  const std::array<float, 1> distance0{1.0F};
  const DistanceBatchResponseTaskInput response0{21, 3, distance0, ids0};
  AlignedBuffer<512> response_wire0;
  auto response_bytes0 = EncodeDistanceBatchResponse(
      70, std::span<const DistanceBatchResponseTaskInput>(&response0, 1),
      response_wire0.bytes);
  UB_REQUIRE_TRUE(response_bytes0.ok());

  const std::array<float, 1> distance1{2.0F};
  const DistanceBatchResponseTaskInput response1{22, 3, distance1, ids1};
  AlignedBuffer<512> response_wire1;
  auto response_bytes1 = EncodeDistanceBatchResponse(
      71, std::span<const DistanceBatchResponseTaskInput>(&response1, 1),
      response_wire1.bytes);
  UB_REQUIRE_TRUE(response_bytes1.ok());
  UB_REQUIRE_TRUE(
      fixture.client
          ->AcceptResponse(1, 1, 71,
                           std::span<const std::byte>(response_wire1.bytes)
                               .first(response_bytes1.value()))
          .ok());
  UB_REQUIRE_TRUE(
      fixture.client
          ->AcceptResponse(1, 0, 70,
                           std::span<const std::byte>(response_wire0.bytes)
                               .first(response_bytes0.value()))
          .ok());
  tasks[0].Reset();
  tasks[1].Reset();
  UB_REQUIRE_TRUE(fixture.client->drained());
}

TEST(ReferenceRuntimeRpcRuntime, RpcServerAssignsEveryRemoteSlotExactlyOnce) {
  RpcMailboxConfig config{3, 2, 4, 1024, 1024, 2};
  auto memory = RpcMailboxMemory::Create(config);
  auto shard = MakeShard();
  UB_REQUIRE_TRUE(memory.ok());
  UB_REQUIRE_TRUE(shard.ok());
  const std::array<std::uint32_t, 3> owners{0, kNoCommunicationOwner, 1};
  const std::array<std::uint32_t, 2> shard_to_rank{0, 1};
  auto server =
      RpcServerRuntime::Create(*memory.value(), 3, 8, 1, shard_to_rank, owners,
                               8, 2, true, &shard.value());
  UB_REQUIRE_TRUE(server.ok());
  std::array<bool, 12> seen{};
  std::array<std::uint32_t, 12> assigned_worker{};
  assigned_worker.fill(kNoCommunicationOwner);
  std::size_t assigned = 0;
  for (std::uint32_t worker = 0; worker < 3; ++worker) {
    for (const auto id : server.value()->worker_slots(worker)) {
      UB_REQUIRE_TRUE(id.peer_rank != 1);
      const auto index = id.peer_rank * 4U + id.slot_id;
      UB_REQUIRE_FALSE(seen[index]);
      seen[index] = true;
      assigned_worker[index] = worker;
      ++assigned;
    }
  }
  UB_REQUIRE_EQ(assigned, 8U);
  for (std::uint32_t slot = 0; slot < 4; ++slot) {
    UB_REQUIRE_EQ(assigned_worker[slot], slot % 2U);
    UB_REQUIRE_EQ(assigned_worker[8U + slot], 2U);
  }
}

TEST(ReferenceRuntimeRpcRuntime,
     RpcServerKeepsPeerRingsAffineWhenWorkersAreScarce) {
  RpcMailboxConfig config{4, 3, 2, 1024, 1024, 2};
  auto memory = RpcMailboxMemory::Create(config);
  auto shard = MakeShard();
  UB_REQUIRE_TRUE(memory.ok());
  UB_REQUIRE_TRUE(shard.ok());
  const std::array<std::uint32_t, 4> owners{0, kNoCommunicationOwner, 1, 2};
  const std::array<std::uint32_t, 2> shard_to_rank{0, 1};
  auto server =
      RpcServerRuntime::Create(*memory.value(), 2, 8, 1, shard_to_rank, owners,
                               8, 2, true, &shard.value());
  UB_REQUIRE_TRUE(server.ok());

  std::array<std::uint32_t, 8> assigned_worker{};
  assigned_worker.fill(kNoCommunicationOwner);
  for (std::uint32_t worker = 0; worker < 2; ++worker) {
    for (const auto id : server.value()->worker_slots(worker)) {
      const auto index = id.peer_rank * 2U + id.slot_id;
      UB_REQUIRE_EQ(assigned_worker[index], kNoCommunicationOwner);
      assigned_worker[index] = worker;
    }
  }
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    UB_REQUIRE_EQ(assigned_worker[slot], 0U);
    UB_REQUIRE_EQ(assigned_worker[4U + slot], 1U);
    UB_REQUIRE_EQ(assigned_worker[6U + slot], 0U);
  }
}

TEST(ReferenceRuntimeRpcRuntime,
     RpcServerDirectPollOwnsResultUntilSendCompletion) {
  RpcMailboxConfig config{2, 1, 2, 1024, 1024, 2};
  auto memory = RpcMailboxMemory::Create(config);
  auto shard = MakeShard();
  UB_REQUIRE_TRUE(memory.ok());
  UB_REQUIRE_TRUE(shard.ok());
  const std::array<std::uint32_t, 2> owners{0, kNoCommunicationOwner};
  const std::array<std::uint32_t, 2> shard_to_rank{0, 1};
  auto server =
      RpcServerRuntime::Create(*memory.value(), 1, 8, 1, shard_to_rank, owners,
                               8, 2, true, &shard.value());
  UB_REQUIRE_TRUE(server.ok());
  HostVisibility visibility;
  const std::array<node_id_t, 2> ids{10, 12};
  const std::array<float, 2> query{1.0F, 1.0F};
  const DistanceBatchRequestTaskInput task{
      77, 3, 1, kBatchHasQueryVector | kBatchUseDotNorm, ids, query, 2, 0.0F};
  auto request_slot = memory.value()->RequestSlot(0, 0);
  auto encoded = EncodeDistanceBatchRequest(
      9, std::span<const DistanceBatchRequestTaskInput>(&task, 1),
      request_slot);
  UB_REQUIRE_TRUE(encoded.ok());
  *memory.value()->NotifyAddress(memory.value()->RequestNotifyOffset(0, 0)) = 9;
  auto progressed =
      server.value()->PollAndProcessRequestSlot(0, Metric::kL2, visibility);
  UB_REQUIRE_TRUE(progressed.ok() && progressed.value());
  UB_REQUIRE_EQ(server.value()->result_state({0, 0}), ResultSlotState::kReady);
  auto ready = server.value()->TryGetReadyResult(0);
  UB_REQUIRE_TRUE(ready.ok() && ready.value().has_value());
  UB_REQUIRE_EQ(ready.value()->id, (RpcSlotId{0, 0}));
  UB_REQUIRE_TRUE(ready.value()->request_notify_observed_nanoseconds != 0);
  auto response = DecodeDistanceBatchResponse(ready.value()->payload);
  UB_REQUIRE_TRUE(response.ok());
  auto response_task = response.value().Task(0);
  UB_REQUIRE_TRUE(response_task.ok());
  UB_REQUIRE_NEAR(response_task.value().distances[0], 1.0F, 1e-6F);
  UB_REQUIRE_NEAR(response_task.value().distances[1], 0.0F, 1e-6F);

  UB_REQUIRE_EQ(server.value()->CompleteResultSend({0, 0}, 9).code(),
                StatusCode::kProtocolError);
  UB_REQUIRE_EQ(server.value()->result_state({0, 0}), ResultSlotState::kReady);
  UB_REQUIRE_TRUE(server.value()->MarkResultPosted({0, 0}, 9).ok());
  UB_REQUIRE_EQ(server.value()->result_state({0, 0}),
                ResultSlotState::kInFlight);
  UB_REQUIRE_EQ(server.value()->CompleteResultSend({0, 0}, 8).code(),
                StatusCode::kProtocolError);
  UB_REQUIRE_EQ(server.value()->result_state({0, 0}),
                ResultSlotState::kInFlight);
  auto second_encoded = EncodeDistanceBatchRequest(
      10, std::span<const DistanceBatchRequestTaskInput>(&task, 1),
      request_slot);
  UB_REQUIRE_TRUE(second_encoded.ok());
  *memory.value()->NotifyAddress(memory.value()->RequestNotifyOffset(0, 0)) =
      10;
  auto blocked =
      server.value()->PollAndProcessRequestSlot(0, Metric::kL2, visibility);
  UB_REQUIRE_TRUE(blocked.ok());
  UB_REQUIRE_FALSE(blocked.value());
  UB_REQUIRE_EQ(
      *memory.value()->NotifyAddress(memory.value()->RequestNotifyOffset(0, 0)),
      10U);
  UB_REQUIRE_TRUE(server.value()->CompleteResultSend({0, 0}, 9).ok());
  auto resumed =
      server.value()->PollAndProcessRequestSlot(0, Metric::kL2, visibility);
  UB_REQUIRE_TRUE(resumed.ok() && resumed.value());
  UB_REQUIRE_EQ(
      server.value()->worker_profile(0).request_slot_blocked_by_result_inflight,
      1U);
}

TEST(ReferenceRuntimeRpcRuntime,
     RpcServerCompletionHandsCleanSlotToDistanceThread) {
  auto memory = RpcMailboxMemory::Create({2, 1, 2, 1024, 1024, 2});
  auto shard = MakeShard();
  UB_REQUIRE_TRUE(memory.ok() && shard.ok());
  const std::array<std::uint32_t, 2> owners{0, kNoCommunicationOwner};
  const std::array<std::uint32_t, 2> shard_to_rank{0, 1};
  auto server =
      RpcServerRuntime::Create(*memory.value(), 1, 8, 1, shard_to_rank, owners,
                               8, 2, true, &shard.value());
  UB_REQUIRE_TRUE(server.ok());
  HostVisibility visibility;
  const std::array<node_id_t, 1> ids{10};
  const std::array<float, 2> query{1.0F, 1.0F};
  const DistanceBatchRequestTaskInput task{
      77, 0, 1, kBatchHasQueryVector | kBatchUseDotNorm, ids, query, 2, 0.0F};
  constexpr std::uint64_t iterations = 5000;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::atomic<std::uint64_t> response_seen{0};
  std::atomic<bool> failed{false};
  std::string sender_error;
  std::jthread sender([&](std::stop_token stop) {
    auto fail = [&](std::string message) {
      sender_error = std::move(message);
      failed.store(true, std::memory_order_release);
    };
    for (std::uint64_t sequence = 1; sequence <= iterations;) {
      if (stop.stop_requested())
        return;
      auto ready = server.value()->TryGetReadyResult(0);
      if (!ready.ok()) {
        fail(ready.status().ToString());
        return;
      }
      if (!ready.value()) {
        std::this_thread::yield();
        continue;
      }
      if (ready.value()->sequence != sequence) {
        fail("result sequence overwritten");
        return;
      }
      auto decoded = DecodeDistanceBatchResponse(ready.value()->payload);
      if (!decoded.ok()) {
        fail(decoded.status().ToString());
        return;
      }
      auto row = decoded.value().Task(0);
      if (!row.ok() || row.value().distances[0] != 1.0F) {
        fail("result distance changed during slot handoff");
        return;
      }
      auto posted = server.value()->MarkResultPosted({0, 0}, sequence);
      if (!posted.ok()) {
        fail(posted.ToString());
        return;
      }
      // The peer may deliver the next request before this local CQ completion.
      response_seen.store(sequence, std::memory_order_release);
      auto completed = server.value()->CompleteResultSend({0, 0}, sequence);
      if (!completed.ok()) {
        fail(completed.ToString());
        return;
      }
      ++sequence;
    }
  });
  std::uint64_t completed_requests = 0;
  std::string worker_error;
  for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
    auto encoded = EncodeDistanceBatchRequest(
        sequence, std::span(&task, 1), memory.value()->RequestSlot(0, 0));
    if (!encoded.ok()) {
      worker_error = encoded.status().ToString();
      break;
    }
    *memory.value()->NotifyAddress(memory.value()->RequestNotifyOffset(0, 0)) =
        sequence;
    bool processed = false;
    while (!failed.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      auto result =
          server.value()->PollAndProcessRequestSlot(0, Metric::kL2, visibility);
      if (!result.ok()) {
        worker_error = result.status().ToString();
        break;
      }
      if (result.value()) {
        processed = true;
        break;
      }
      std::this_thread::yield();
    }
    if (!processed)
      break;
    while (response_seen.load(std::memory_order_acquire) < sequence &&
           !failed.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    if (response_seen.load(std::memory_order_acquire) < sequence)
      break;
    ++completed_requests;
  }
  sender.request_stop();
  sender.join();
  if (!sender_error.empty() || !worker_error.empty()) {
    test::Fail("slot handoff", __FILE__, __LINE__,
               sender_error + " " + worker_error);
  }
  UB_REQUIRE_EQ(completed_requests, iterations);
  UB_REQUIRE_TRUE(server.value()->drained(0));
}

} // namespace hnsw_urma
