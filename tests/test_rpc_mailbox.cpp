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
#include "test_framework.hpp"

#include <cstddef>
#include <cstdint>

#include "hnsw_urma/runtime/rpc/mailbox.hpp"

namespace hnsw_urma {

TEST(ReferenceRuntimeRpcMailbox, MailboxPartitionsThreeRegisteredSegments) {
  RpcMailboxConfig config;
  config.world_size = 4;
  config.rpc_threads = 2;
  config.ring_depth = 8;
  config.request_capacity = 1000;
  config.response_capacity = 500;
  config.request_staging_slots_per_thread = 4;
  auto memory = RpcMailboxMemory::Create(config);
  UB_REQUIRE_TRUE(memory.ok());
  UB_REQUIRE_EQ(memory.value()->request_stride(), 1024U);
  UB_REQUIRE_EQ(memory.value()->response_stride(), 512U);
  UB_REQUIRE_EQ(memory.value()->RequestOffset(1, 1) -
                    memory.value()->RequestOffset(0, 1),
                8U * 1024U);
  UB_REQUIRE_EQ(memory.value()->ResponseOffset(2, 1) -
                    memory.value()->ResponseOffset(1, 1),
                8U * 512U);
  UB_REQUIRE_TRUE(memory.value()->ResponseOffset(1, 1) !=
                  memory.value()->ResultOffset(1, 1));
  UB_REQUIRE_TRUE(memory.value()->RequestNotifyOffset(0, 1) !=
                  memory.value()->ResponseNotifyOffset(0, 1));
  UB_REQUIRE_EQ(memory.value()->RequestSlot(1, 1).size(), 1000U);
  UB_REQUIRE_EQ(memory.value()->ResponseSlot(1, 1).size(), 500U);
  UB_REQUIRE_EQ(memory.value()->ResultSlot(1, 1).size(), 500U);
  UB_REQUIRE_EQ(memory.value()->RequestStaging(1, 3).size(), 1000U);
  UB_REQUIRE_TRUE(memory.value()->requests().data() !=
                  memory.value()->responses().data());
  UB_REQUIRE_TRUE(memory.value()->requests().data() !=
                  memory.value()->notify().data());
  UB_REQUIRE_EQ(memory.value()->logical_bytes(),
                memory.value()->requests().size() +
                    memory.value()->responses().size() +
                    memory.value()->notify().size());
  UB_REQUIRE_TRUE(memory.value()->NotifyAddress(
                      memory.value()->RequestNotifyOffset(0, 1)) != nullptr);
  UB_REQUIRE_TRUE(memory.value()->NotifyAddress(
                      memory.value()->ResponseNotifyOffset(0, 1)) != nullptr);
}

TEST(ReferenceRuntimeRpcMailbox, MailboxUsesExplicitSlotIdsWithoutWrapping) {
  RpcMailboxConfig config{2, 1, 2, 128, 128, 1};
  auto memory = RpcMailboxMemory::Create(config);
  UB_REQUIRE_TRUE(memory.ok());
  UB_REQUIRE_TRUE(memory.value()->RequestSlot(0, 2).empty());
  UB_REQUIRE_TRUE(memory.value()->ResponseSlot(2, 0).empty());
  UB_REQUIRE_TRUE(memory.value()->ResultSlot(0, 2).empty());
  UB_REQUIRE_TRUE(memory.value()->RequestStaging(1, 0).empty());
}

TEST(ReferenceRuntimeRpcMailbox, MailboxRejectsNonPowerOfTwoDepth) {
  RpcMailboxConfig config{2, 1, 3, 1024, 1024, 1};
  UB_REQUIRE_FALSE(RpcMailboxMemory::Create(config).ok());
}

TEST(ReferenceRuntimeRpcMailbox, MailboxCanBackThreeSegmentsWithSharedMemory) {
  RpcMailboxConfig config{2, 1, 2, 128, 128, 1};
  auto memory = RpcMailboxMemory::Create(config, true);
  UB_REQUIRE_TRUE(memory.ok());
  UB_REQUIRE_TRUE(memory.value()->is_shared());
  UB_REQUIRE_FALSE(memory.value()->request_shared_memory_name().empty());
  UB_REQUIRE_FALSE(memory.value()->response_shared_memory_name().empty());
  UB_REQUIRE_FALSE(memory.value()->notify_shared_memory_name().empty());
  UB_REQUIRE_TRUE(memory.value()->UnlinkSharedMemory().ok());
}

} // namespace hnsw_urma
