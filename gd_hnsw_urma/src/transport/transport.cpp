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

#include "hnsw_urma/transport/transport.hpp"

namespace hnsw_urma {
void TransportModuleAnchor() {}

LanePostBatchResult
LanePostEngine::PostReads(std::span<const ReadOperation> operations) {
  LanePostBatchResult result;
  for (const auto &operation : operations) {
    ++result.post_calls;
    result.status = PostRead(operation);
    if (!result.status.ok()) {
      return result;
    }
    ++result.posted;
  }
  return result;
}

LanePostBatchResult LanePostEngine::PostWriteNotifies(
    std::span<const WriteNotifyOperation> operations) {
  LanePostBatchResult result;
  for (const auto &operation : operations) {
    result.status = PostWriteNotify(operation);
    if (!result.status.ok()) {
      return result;
    }
    ++result.posted;
  }
  return result;
}
} // namespace hnsw_urma
