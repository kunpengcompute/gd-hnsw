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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "hnsw_urma/core/page_aligned_buffer.hpp"

using hnsw_urma::PageAlignedBuffer;
using hnsw_urma::StatusCode;

TEST(ReferenceUnitPageAlignedBuffer,
     PageAlignedBufferAlignsBaseAndRegistrationLength) {
  auto page_size = PageAlignedBuffer::SystemPageSize();
  UB_REQUIRE_TRUE(page_size.ok());

  auto buffer = PageAlignedBuffer::Create(page_size.value() + 1U);
  UB_REQUIRE_TRUE(buffer.ok());
  UB_REQUIRE_EQ(buffer.value().logical_memory().size(), page_size.value() + 1U);
  UB_REQUIRE_EQ(buffer.value().registration_memory().size(),
                page_size.value() * 2U);
  UB_REQUIRE_EQ(reinterpret_cast<std::uintptr_t>(
                    buffer.value().registration_memory().data()) %
                    page_size.value(),
                0U);
  UB_REQUIRE_EQ(buffer.value().registration_memory().size() % page_size.value(),
                0U);

  for (const auto value : buffer.value().registration_memory()) {
    UB_REQUIRE_EQ(value, std::byte{0});
  }
}

TEST(ReferenceUnitPageAlignedBuffer,
     PageAlignedBufferCopiesLogicalBytesAndZeroFillsPadding) {
  const std::vector<std::byte> source{
      std::byte{0x12},
      std::byte{0x34},
      std::byte{0x56},
  };
  auto buffer = PageAlignedBuffer::CopyFrom(source);
  UB_REQUIRE_TRUE(buffer.ok());
  UB_REQUIRE_EQ(buffer.value().logical_memory().size(), source.size());
  for (std::size_t index = 0; index < source.size(); ++index) {
    UB_REQUIRE_EQ(buffer.value().logical_memory()[index], source[index]);
  }
  for (std::size_t index = source.size();
       index < buffer.value().registration_memory().size(); ++index) {
    UB_REQUIRE_EQ(buffer.value().registration_memory()[index], std::byte{0});
  }
}

TEST(ReferenceUnitPageAlignedBuffer, PageAlignedBufferRejectsZeroAndOverflow) {
  auto zero = PageAlignedBuffer::Create(0);
  UB_REQUIRE_FALSE(zero.ok());
  UB_REQUIRE_EQ(zero.status().code(), StatusCode::kInvalidArgument);

  auto overflow =
      PageAlignedBuffer::Create(std::numeric_limits<std::size_t>::max());
  UB_REQUIRE_FALSE(overflow.ok());
  UB_REQUIRE_EQ(overflow.status().code(), StatusCode::kOutOfRange);
}

TEST(ReferenceUnitPageAlignedBuffer,
     PageAlignedBufferSharedMappingObservesOwnerWrites) {
  auto owner = PageAlignedBuffer::CreateShared(128);
  UB_REQUIRE_TRUE(owner.ok());
  UB_REQUIRE_TRUE(owner.value().is_shared());
  UB_REQUIRE_FALSE(owner.value().shared_memory_name().empty());

  auto peer =
      PageAlignedBuffer::OpenShared(owner.value().shared_memory_name(),
                                    owner.value().logical_memory().size());
  UB_REQUIRE_TRUE(peer.ok());
  owner.value().logical_memory()[17] = std::byte{0x5a};
  UB_REQUIRE_EQ(peer.value().logical_memory()[17], std::byte{0x5a});
  UB_REQUIRE_TRUE(owner.value().UnlinkSharedMemory().ok());
  UB_REQUIRE_TRUE(owner.value().shared_memory_name().empty());
  peer.value().logical_memory()[23] = std::byte{0x33};
  UB_REQUIRE_EQ(owner.value().logical_memory()[23], std::byte{0x33});
}
