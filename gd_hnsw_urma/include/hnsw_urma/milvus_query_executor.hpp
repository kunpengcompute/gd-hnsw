#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/runtime/drain.hpp"
#include "hnsw_urma/runtime/rpc/distance_oracle.hpp"
#include "hnsw_urma/runtime/scheduler.hpp"
#include "hnsw_urma/search/query_slot_pool.hpp"
#include "hnsw_urma/search/read_resources.hpp"
#include "hnsw_urma/search/search.hpp"

namespace hnsw_urma {

// Service admission replaces the demo's finite query list. Search, ready,
// retry, READ reclamation and RPC slot retirement retain the pinned runner's
// owner model. This is also the entry exercised by the portable service test.
class MilvusQueryExecutor {
 public:
  // Owned and reported by Control only; successful-query timings are cumulative.
  struct Profile {
    std::uint64_t completed{0}, errors{0};
    std::uint64_t pre_admit_ns{0}, active_ns{0}, execution_ns{0};
    std::uint64_t graph_wait_ns{0}, distance_wait_ns{0}, ready_wait_ns{0};
    std::uint64_t upper_ns{0}, layer0_ns{0}, nodes{0}, edges{0};
    std::uint64_t active{0}, pending{0}, retiring{0}, quarantined{0};
  };
  using ProfileReporter = std::function<void(const Profile&)>;
  static Result<std::unique_ptr<MilvusQueryExecutor>> Create(
      const GlobalIndexMeta&, const ShardMap&, ShardEntryPoint,
      GraphReader&, DistanceOracle&, const RuntimeConfig&,
      const EffectiveRuntimeConfig&, std::uint32_t rank, SchedulerQueues&,
      RpcDistanceOracle*, std::span<SearchReadResources* const>,
      DataPlaneDrainTracker*, ProfileReporter = {});
  struct BatchResult {
    SearchParams params;
    std::vector<SearchResult> rows;
  };
  Result<BatchResult> Submit(std::span<const float> queries,
                                          std::uint32_t nq,
                                          std::uint32_t topk,
                                          std::uint32_t ef,
      std::chrono::steady_clock::time_point request_deadline =
          std::chrono::steady_clock::time_point::max());
  Status Control(std::stop_token);
  Status Search(std::stop_token, std::uint32_t worker);
  void StopAdmission();
  void FailPending(Status);
  bool queries_drained() const { return drained_.load(); }
  SchedulerReadyPublisher& publisher() { return *publisher_; }

 private:
  struct Completion {
    std::mutex mutex;
    std::condition_variable ready;
    bool done{false};
    Status status;
    SearchResult result;
    void Finish(Status, SearchResult = {});
  };
  struct Pending {
    std::vector<float> query;
    SearchParams params;
    std::shared_ptr<Completion> completion;
    std::chrono::steady_clock::time_point deadline;
  };
  struct Slot {
    std::chrono::steady_clock::time_point submitted_at{}, admitted_at{};
    std::unique_ptr<QueryScratch> scratch;
    std::uint32_t topk{0}, ef{0};
    std::unique_ptr<HnswSearchTask> storage;
    std::atomic<HnswSearchTask*> task{nullptr};
    std::atomic<std::uint32_t> readers{0};
    bool deadline_expired{false};
    std::shared_ptr<Completion> completion;
  };
  class TaskRef {
   public:
    explicit TaskRef(Slot& slot) : slot_(slot) {
      slot_.readers.fetch_add(1, std::memory_order_acq_rel);
      task_ = slot_.task.load(std::memory_order_acquire);
    }
    ~TaskRef() { slot_.readers.fetch_sub(1, std::memory_order_release); }
    HnswSearchTask* operator->() const { return task_; }
    explicit operator bool() const { return task_ != nullptr; }
   private:
    Slot& slot_;
    HnswSearchTask* task_;
  };
  bool WireQuiesced(QueryKey key) const;
  const GlobalIndexMeta* meta_{};
  const ShardMap* map_{};
  ShardEntryPoint entry_{};
  GraphReader* graph_{};
  DistanceOracle* distance_{};
  RuntimeConfig config_;
  EffectiveRuntimeConfig effective_;
  SchedulerQueues* queues_{};
  RpcDistanceOracle* rpc_{};
  std::vector<SearchReadResources*> reads_;
  DataPlaneDrainTracker* drain_{};
  std::optional<QuerySlotPool> pool_;
  std::optional<QueryDeadlineHeap> deadlines_;
  std::unique_ptr<Slot[]> slots_;
  std::unique_ptr<SchedulerReadyPublisher> publisher_;
  SearchRuntimeContext context_;
  std::mutex mutex_;
  std::deque<Pending> pending_;
  bool accepting_{true};
  // Protected by mutex_. Only prolonged retirement rejects new submissions;
  // the pool keeps every deferred slot unavailable until its wire quiesces.
  std::uint32_t quarantined_wire_slots_{0};
  ProfileReporter report_profile_;
  std::atomic<bool> drained_{false};
};
}  // namespace hnsw_urma
