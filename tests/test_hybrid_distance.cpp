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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hnsw_urma/runtime/hybrid_distance.hpp"

namespace hnsw_urma {
namespace {

constexpr ticket_t kPullTag = ticket_t{1} << 62U;
constexpr ticket_t kPushTag = ticket_t{1} << 63U;

class FakeOracle final : public DistanceOracle {
public:
  Result<DistanceTicket> Submit(QueryKey, std::span<const float>, shard_id_t,
                                std::span<const node_id_t>, std::span<float>,
                                DistanceRequestOptions) override {
    ++submit_calls;
    return next;
  }
  Result<DistanceTicket>
  SubmitLeaseForWait(const QueryVectorLease &, std::uint64_t, shard_id_t,
                     std::span<const node_id_t>, std::span<float>,
                     DistanceRequestOptions options) override {
    ++lease_calls;
    last_upper = options.upper_distance;
    if (!submit_ok) {
      return Status(StatusCode::kInternalError, "fake submit failure");
    }
    return next;
  }
  Result<bool> Poll(DistanceTicket) override {
    ++poll_calls;
    return poll_value;
  }
  Status Finish(DistanceTicket) override {
    ++finish_calls;
    return Status::Ok();
  }
  [[nodiscard]] bool SupportsPreparedWait() const noexcept override {
    return prepared_support;
  }
  Result<PreparedDistanceOperation>
  PrepareLeaseForWait(QueryWaitPreparation, const QueryVectorLease &,
                      shard_id_t, std::span<const node_id_t>, std::span<float>,
                      DistanceRequestOptions) override {
    ++prepare_calls;
    if (!prepare_ok) {
      return Status(StatusCode::kInvalidArgument, "fake prepare failure");
    }
    return PreparedDistanceOperation{next, std::uint32_t{2}};
  }
  Status PublishPrepared(DistanceTicket) override {
    ++publish_calls;
    return Status::Ok();
  }
  Status CancelPrepared(DistanceTicket) override {
    ++cancel_calls;
    return Status::Ok();
  }
  Result<bool> ComputeLocalTrusted(std::span<const float>, shard_id_t,
                                   std::span<const std::uint64_t>,
                                   std::span<float>) override {
    return compute_value;
  }
  Result<bool> ComputeLocalTrustedLease(const QueryVectorLease &, shard_id_t,
                                        std::span<const std::uint64_t>,
                                        std::span<float>) override {
    return compute_value;
  }

  DistanceTicket next{1, 1};
  bool submit_ok{true};
  bool prepare_ok{true};
  bool prepared_support{false};
  bool poll_value{true};
  bool compute_value{true};
  std::optional<float> last_upper;
  int submit_calls{0};
  int lease_calls{0};
  int poll_calls{0};
  int finish_calls{0};
  int prepare_calls{0};
  int publish_calls{0};
  int cancel_calls{0};
};

QueryKey Key(std::uint64_t generation = 7) {
  return QueryKey{1, 2, generation};
}

QueryVectorLease Lease(QueryKey query) {
  auto storage = std::make_shared<const std::vector<float>>(
      std::initializer_list<float>{1.0F, 2.0F});
  return QueryVectorLease(query, std::move(storage));
}

} // namespace

TEST(RuntimeHybridDistance, CreateValidatesThresholdAndRoutesByPolicy) {
  FakeOracle pull;
  FakeOracle push;
  EXPECT_EQ(
      HybridDistanceOracle::Create(0, DistancePolicy::kAuto, 0, pull, push)
          .status()
          .code(),
      StatusCode::kInvalidArgument);

  auto created =
      HybridDistanceOracle::Create(0, DistancePolicy::kAuto, 4, pull, push);
  ASSERT_TRUE(created.ok()) << created.status().ToString();
  auto &oracle = *created.value();

  const std::array<float, 2> query{1.0F, 2.0F};
  const std::array<node_id_t, 2> small{1U, 2U};
  const std::array<node_id_t, 4> large{1U, 2U, 3U, 4U};
  std::array<float, 2> small_out{};
  std::array<float, 4> large_out{};
  DistanceRequestOptions options;
  options.upper_distance = 3.0F;

  auto pull_ticket = oracle.Submit(Key(), query, 9, small, small_out, options);
  ASSERT_TRUE(pull_ticket.ok()) << pull_ticket.status().ToString();
  EXPECT_EQ(pull_ticket.value().value, ticket_t{1} | kPullTag);
  EXPECT_FALSE(pull.last_upper.has_value());

  auto push_ticket = oracle.Submit(Key(), query, 9, large, large_out, options);
  ASSERT_TRUE(push_ticket.ok()) << push_ticket.status().ToString();
  EXPECT_EQ(push_ticket.value().value, ticket_t{1} | kPushTag);
  ASSERT_TRUE(push.last_upper.has_value());
  EXPECT_FLOAT_EQ(push.last_upper.value(), 3.0F);

  auto local_ticket = oracle.Submit(Key(), query, 0, small, small_out, options);
  ASSERT_TRUE(local_ticket.ok()) << local_ticket.status().ToString();
  EXPECT_EQ(local_ticket.value().value, ticket_t{1} | kPullTag);
  EXPECT_FALSE(pull.last_upper.has_value());

  const auto profile = oracle.profile();
  EXPECT_EQ(profile.local_batches, 1U);
  EXPECT_EQ(profile.pull_batches, 1U);
  EXPECT_EQ(profile.pushdown_batches, 1U);

  auto forced_push =
      HybridDistanceOracle::Create(0, DistancePolicy::kPushdown, 4, pull, push);
  ASSERT_TRUE(forced_push.ok());
  auto forced_ticket =
      forced_push.value()->Submit(Key(), query, 9, small, small_out);
  ASSERT_TRUE(forced_ticket.ok());
  EXPECT_EQ(forced_ticket.value().value, ticket_t{1} | kPushTag);

  auto forced_pull =
      HybridDistanceOracle::Create(0, DistancePolicy::kPull, 4, pull, push);
  ASSERT_TRUE(forced_pull.ok());
  auto forced_pull_ticket =
      forced_pull.value()->Submit(Key(), query, 9, large, large_out);
  ASSERT_TRUE(forced_pull_ticket.ok());
  EXPECT_EQ(forced_pull_ticket.value().value, ticket_t{1} | kPullTag);
}

TEST(RuntimeHybridDistance,
     TicketEncodingRejectsForeignTicketsAndDecodesFailures) {
  FakeOracle pull;
  FakeOracle push;
  auto created =
      HybridDistanceOracle::Create(0, DistancePolicy::kAuto, 4, pull, push);
  ASSERT_TRUE(created.ok());
  auto &oracle = *created.value();
  const std::array<float, 2> query{1.0F, 2.0F};
  const std::array<node_id_t, 2> ids{1U, 2U};
  std::array<float, 2> out{};

  pull.next = DistanceTicket{0, 1};
  EXPECT_EQ(oracle.Submit(Key(), query, 9, ids, out).status().code(),
            StatusCode::kProtocolError);
  pull.next = DistanceTicket{1, 0};
  EXPECT_EQ(oracle.Submit(Key(), query, 9, ids, out).status().code(),
            StatusCode::kProtocolError);
  pull.next = DistanceTicket{1 | kPullTag, 1};
  EXPECT_EQ(oracle.Submit(Key(), query, 9, ids, out).status().code(),
            StatusCode::kProtocolError);

  pull.next = DistanceTicket{1, 1};
  auto ticket = oracle.Submit(Key(), query, 9, ids, out);
  ASSERT_TRUE(ticket.ok());

  EXPECT_EQ(oracle.Poll(DistanceTicket{0, 1}).status().code(),
            StatusCode::kProtocolError);
  EXPECT_EQ(oracle.Poll(DistanceTicket{1 | kPullTag, 0}).status().code(),
            StatusCode::kProtocolError);
  EXPECT_EQ(oracle.Poll(DistanceTicket{kPullTag, 1}).status().code(),
            StatusCode::kProtocolError);
  EXPECT_TRUE(oracle.Poll(ticket.value()).ok());
  EXPECT_EQ(pull.poll_calls, 1);

  EXPECT_EQ(oracle.Finish(ticket.value()).code(), StatusCode::kOk);
  EXPECT_EQ(oracle.PublishPrepared(ticket.value()).code(), StatusCode::kOk);
  EXPECT_EQ(oracle.CancelPrepared(ticket.value()).code(), StatusCode::kOk);
  EXPECT_EQ(pull.finish_calls, 1);
  EXPECT_EQ(pull.publish_calls, 1);
  EXPECT_EQ(pull.cancel_calls, 1);

  EXPECT_EQ(oracle.Finish(DistanceTicket{0, 1}).code(),
            StatusCode::kProtocolError);
  EXPECT_EQ(oracle.PublishPrepared(DistanceTicket{kPullTag, 1}).code(),
            StatusCode::kProtocolError);
  EXPECT_EQ(oracle.CancelPrepared(DistanceTicket{1 | kPushTag, 0}).code(),
            StatusCode::kProtocolError);
}

TEST(RuntimeHybridDistance,
     PreparedPathsCheckBackendSupportAndRethrowFailures) {
  FakeOracle pull;
  FakeOracle push;
  auto created =
      HybridDistanceOracle::Create(0, DistancePolicy::kAuto, 4, pull, push);
  ASSERT_TRUE(created.ok());
  auto &oracle = *created.value();
  const QueryKey query = Key();
  const std::array<float, 2> vector{1.0F, 2.0F};
  const std::array<node_id_t, 2> ids{1U, 2U};
  std::array<float, 2> out{};
  QueryWaitPreparation preparation;
  preparation.query = query;
  preparation.wait_epoch = 1;
  const auto lease = Lease(query);

  EXPECT_FALSE(oracle.SupportsPreparedWait());
  EXPECT_EQ(oracle.PrepareLeaseForWait(preparation, lease, 0, ids, out, {})
                .status()
                .code(),
            StatusCode::kProtocolError);

  pull.prepared_support = true;
  EXPECT_FALSE(oracle.SupportsPreparedWait());
  push.prepared_support = true;
  EXPECT_TRUE(oracle.SupportsPreparedWait());

  pull.prepare_ok = false;
  EXPECT_EQ(oracle.PrepareLeaseForWait(preparation, lease, 0, ids, out, {})
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  pull.prepare_ok = true;
  auto prepared =
      oracle.PrepareLeaseForWait(preparation, lease, 0, ids, out, {});
  ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
  EXPECT_EQ(prepared.value().ticket.value, ticket_t{1} | kPullTag);
  ASSERT_TRUE(prepared.value().owner_lane.has_value());
  EXPECT_EQ(prepared.value().owner_lane.value(), 2U);
  EXPECT_EQ(oracle.profile().local_batches, 1U);

  auto prepared_remote =
      oracle.PrepareLeaseForWait(preparation, lease, 9, ids, out, {});
  ASSERT_TRUE(prepared_remote.ok());
  EXPECT_EQ(prepared_remote.value().ticket.value, ticket_t{1} | kPullTag);
  EXPECT_EQ(oracle.profile().pull_batches, 1U);

  auto prepared_for_wait =
      oracle.PrepareForWait(preparation, vector, 9, ids, out, {});
  ASSERT_TRUE(prepared_for_wait.ok()) << prepared_for_wait.status().ToString();
  EXPECT_EQ(prepared_for_wait.value().ticket.value, ticket_t{1} | kPullTag);
  EXPECT_EQ(oracle.profile().pull_batches, 2U);

  pull.next = DistanceTicket{1 | kPullTag, 1};
  EXPECT_EQ(oracle.PrepareLeaseForWait(preparation, lease, 0, ids, out, {})
                .status()
                .code(),
            StatusCode::kProtocolError);
  EXPECT_EQ(pull.cancel_calls, 1);
}

TEST(RuntimeHybridDistance, LocalTrustedComputationHonorsOwnerAndCounts) {
  FakeOracle pull;
  FakeOracle push;
  auto created =
      HybridDistanceOracle::Create(0, DistancePolicy::kAuto, 4, pull, push);
  ASSERT_TRUE(created.ok());
  auto &oracle = *created.value();
  const std::array<float, 2> query{1.0F, 2.0F};
  const std::array<std::uint64_t, 2> offsets{0U, 1U};
  std::array<float, 2> out{};

  EXPECT_TRUE(oracle.CanComputeLocalTrusted(0));
  EXPECT_FALSE(oracle.CanComputeLocalTrusted(9));

  auto remote = oracle.ComputeLocalTrusted(query, 9, offsets, out);
  ASSERT_TRUE(remote.ok());
  EXPECT_FALSE(remote.value());
  EXPECT_EQ(oracle.profile().local_batches, 0U);

  auto local = oracle.ComputeLocalTrusted(query, 0, offsets, out);
  ASSERT_TRUE(local.ok());
  EXPECT_TRUE(local.value());
  EXPECT_EQ(oracle.profile().local_batches, 1U);

  const auto lease = Lease(Key());
  auto local_lease = oracle.ComputeLocalTrustedLease(lease, 0, offsets, out);
  ASSERT_TRUE(local_lease.ok());
  EXPECT_TRUE(local_lease.value());
  EXPECT_EQ(oracle.profile().local_batches, 2U);

  auto remote_lease = oracle.ComputeLocalTrustedLease(lease, 9, offsets, out);
  ASSERT_TRUE(remote_lease.ok());
  EXPECT_FALSE(remote_lease.value());

  pull.compute_value = false;
  auto skipped = oracle.ComputeLocalTrusted(query, 0, offsets, out);
  ASSERT_TRUE(skipped.ok());
  EXPECT_FALSE(skipped.value());
  EXPECT_EQ(oracle.profile().local_batches, 2U);
}

TEST(RuntimeHybridDistance, SubmitFailurePropagatesWithoutCounting) {
  FakeOracle pull;
  FakeOracle push;
  auto created =
      HybridDistanceOracle::Create(0, DistancePolicy::kAuto, 4, pull, push);
  ASSERT_TRUE(created.ok());
  auto &oracle = *created.value();
  const std::array<float, 2> query{1.0F, 2.0F};
  const std::array<node_id_t, 2> ids{1U, 2U};
  std::array<float, 2> out{};

  pull.submit_ok = false;
  EXPECT_EQ(oracle.Submit(Key(), query, 9, ids, out).status().code(),
            StatusCode::kInternalError);
  EXPECT_EQ(oracle.profile().local_batches, 0U);
  EXPECT_EQ(oracle.profile().pull_batches, 0U);
  EXPECT_EQ(oracle.profile().pushdown_batches, 0U);
}

} // namespace hnsw_urma
