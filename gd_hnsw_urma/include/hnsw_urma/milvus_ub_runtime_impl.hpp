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

#pragma once
#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/milvus_query_executor.hpp"
#include "hnsw_urma/milvus_ub_runtime.hpp"
#include "hnsw_urma/runtime/rpc/transport.hpp"
#include "hnsw_urma/runtime/rpc_runtime.hpp"
#include "hnsw_urma/search/remote_io.hpp"
#include "hnsw_urma/transport/urma/urma_transport.hpp"

namespace hnsw_urma {
struct MilvusUbRuntime::DistributedState {
#if HNSW_URMA_ENABLE_URMA
  RuntimeConfig requested;
  EffectiveRuntimeConfig effective;
  // Reverse destruction: stop workers, destroy transport registrations, then
  // release backing storage. Published mailbox and local shard belong to
  // runtime.
  std::optional<PageAlignedBuffer> read_buffer;
  std::unique_ptr<UmdkResourceBackend> backend;
  std::optional<UrmaResourceBundle> bundle;
  std::vector<std::unique_ptr<LanePostEngine>> engines;
  std::vector<std::unique_ptr<ReadBatchLane>> read_lanes;
  std::vector<std::unique_ptr<SearchReadResources>> read_pools;
  std::optional<PeerSegmentTable> segments;
  std::optional<UrmaGraphReader> graph;
  std::optional<PullVectorDistanceOracle> pull;
  std::unique_ptr<SchedulerQueues> queues;
  std::unique_ptr<RpcDistanceOracle> rpc;
  std::unique_ptr<RpcServerRuntime> server;
  std::vector<std::unique_ptr<RpcClientOwner>> clients;
  std::vector<std::unique_ptr<RpcTransportOwner>> transports;
  std::unique_ptr<HybridDistanceOracle> distance;
  std::unique_ptr<DataPlaneDrainTracker> drain;
  std::unique_ptr<MilvusQueryExecutor> executor;
  std::unique_ptr<RuntimeThreadGroup> threads;
#endif
};
} // namespace hnsw_urma
