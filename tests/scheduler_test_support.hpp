#pragma once
#include "hnsw_urma/runtime/scheduler.hpp"
#include "test_framework.hpp"
namespace hnsw_urma::test {
inline std::optional<QueryRunnableHandle>
TakeRunnable(SchedulerQueues &queues, std::uint32_t worker,
             std::chrono::steady_clock::time_point now) {
  auto input = queues.PopSearchInput(worker, now);
  if (!input)
    return std::nullopt;
  UB_REQUIRE_TRUE(std::holds_alternative<QueryRunnableHandle>(*input));
  return std::get<QueryRunnableHandle>(*input);
}
} // namespace hnsw_urma::test
