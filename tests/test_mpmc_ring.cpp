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

#include <atomic>
#include <cstdint>
#include <optional>
#include <thread>

#include "hnsw_urma/core/mpmc_ring.hpp"

namespace hnsw_urma {
namespace {

struct BlockingMoveState {
  std::atomic<bool> consumer_blocked{false};
  std::atomic<bool> release_consumer{false};
};

struct BlockingMoveValue {
  std::uint32_t value{0};
  BlockingMoveState *state{nullptr};

  BlockingMoveValue() = default;
  BlockingMoveValue(std::uint32_t input, BlockingMoveState *shared)
      : value(input), state(shared) {}

  BlockingMoveValue(BlockingMoveValue &&other) noexcept
      : value(other.value), state(other.state) {
    if (value == 1 && state != nullptr &&
        !state->consumer_blocked.exchange(true, std::memory_order_acq_rel)) {
      while (!state->release_consumer.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
  }

  BlockingMoveValue &operator=(BlockingMoveValue &&other) noexcept {
    value = other.value;
    state = other.state;
    return *this;
  }
};

} // namespace

TEST(ReferenceUnitMpmcRing, MpmcReservedPushWaitsForOutOfOrderConsumerRelease) {
  auto created = MpmcRing<BlockingMoveValue>::Create(2);
  UB_REQUIRE_TRUE(created.ok());
  if (!created.ok()) {
    return;
  }
  auto ring = std::move(created).value();
  BlockingMoveState state;
  UB_REQUIRE_TRUE(ring.TryPush(BlockingMoveValue(1, &state)).ok());
  UB_REQUIRE_TRUE(ring.TryPush(BlockingMoveValue(2, &state)).ok());

  std::thread slow_consumer([&] { (void)ring.TryPop(); });
  while (!state.consumer_blocked.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // The later consumer completes first. An external aggregate credit is now
  // available, but the next FIFO cell still belongs to slow_consumer.
  auto later = ring.TryPop();
  UB_REQUIRE_TRUE(later.has_value());
  UB_REQUIRE_EQ(later->value, 2U);
  UB_REQUIRE_FALSE(ring.TryPush(BlockingMoveValue(3, &state)).ok());

  std::optional<Status> reserved_push;
  std::thread producer([&] {
    reserved_push = ring.PushWithReservedCapacity(BlockingMoveValue(3, &state));
  });
  state.release_consumer.store(true, std::memory_order_release);
  slow_consumer.join();
  producer.join();

  UB_REQUIRE_TRUE(reserved_push.has_value());
  UB_REQUIRE_TRUE(reserved_push->ok());
  auto published = ring.TryPop();
  UB_REQUIRE_TRUE(published.has_value());
  UB_REQUIRE_EQ(published->value, 3U);
}

} // namespace hnsw_urma

TEST(ReferenceUnitMpmcRing,
     MpmcRejectsSingleCellAndAccountsForSequenceAndPadding) {
  using Ring = hnsw_urma::MpmcRing<std::uint8_t>;
  UB_REQUIRE_FALSE(Ring::Create(1).ok());
  UB_REQUIRE_TRUE(Ring::Create(2).ok());
  UB_REQUIRE_TRUE(Ring::StorageBytes(2).value() >=
                  2 * (sizeof(std::uint64_t) + 1));
  UB_REQUIRE_EQ(Ring::RequiredBytes(2).value(),
                sizeof(Ring) + Ring::StorageBytes(2).value());
  UB_REQUIRE_FALSE(Ring::RequiredBytes(SIZE_MAX).ok());
}

TEST(ReferenceUnitMpmcRing,
     MpscCannotConsumePastUnpublishedProducerAndReusesReleasedCells) {
  struct PausedValue {
    int value{0};
    std::atomic<bool> *entered{nullptr};
    std::atomic<bool> *release{nullptr};
    PausedValue() = default;
    PausedValue(int input, std::atomic<bool> *e = nullptr,
                std::atomic<bool> *r = nullptr)
        : value(input), entered(e), release(r) {}
    PausedValue(const PausedValue &) = default;
    PausedValue &operator=(const PausedValue &other) noexcept {
      value = other.value;
      if (other.entered) {
        other.entered->store(true, std::memory_order_release);
        while (!other.release->load(std::memory_order_acquire))
          std::this_thread::yield();
      }
      return *this;
    }
  };
  auto ring = hnsw_urma::MpmcRing<PausedValue>::Create(2).value();
  std::atomic<bool> entered{false}, release{false};
  hnsw_urma::Status first_status;
  std::thread first(
      [&] { first_status = ring.TryPush(PausedValue{1, &entered, &release}); });
  while (!entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  UB_REQUIRE_TRUE(ring.TryPush(PausedValue{2}).ok());
  UB_REQUIRE_FALSE(ring.TryPop().has_value());
  release.store(true, std::memory_order_release);
  first.join();
  UB_REQUIRE_TRUE(first_status.ok());
  UB_REQUIRE_EQ(ring.TryPop()->value, 1);
  UB_REQUIRE_TRUE(ring.TryPush(PausedValue{3}).ok());
  UB_REQUIRE_EQ(ring.TryPop()->value, 2);
  UB_REQUIRE_EQ(ring.TryPop()->value, 3);
}
