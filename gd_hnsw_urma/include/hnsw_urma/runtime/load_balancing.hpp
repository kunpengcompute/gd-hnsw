#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

struct LaneLoadSample {
  std::uint32_t submit_backlog{0};
  std::uint32_t outstanding{0};
  std::uint32_t active_replies{0};
  std::uint32_t reply_capacity_remaining{0};
  std::uint32_t credit_pressure{0};
  std::uint32_t timeout_pressure{0};
  std::uint64_t recent_latency_ns{0};
};

template <typename Sample>
class LoadSnapshotArray {
 public:
  static_assert(std::is_trivially_copyable_v<Sample>);

  [[nodiscard]] static Result<std::unique_ptr<LoadSnapshotArray>> Create(
      std::uint32_t count) {
    if (count == 0) {
      return Status(StatusCode::kInvalidArgument,
                    "load snapshot count must be non-zero");
    }
    try {
      auto snapshots =
          std::unique_ptr<LoadSnapshotArray>(new LoadSnapshotArray());
      snapshots->count_ = count;
      snapshots->entries_ = std::make_unique<Entry[]>(count);
      return snapshots;
    } catch (const std::bad_alloc&) {
      return Status(StatusCode::kResourceExhausted,
                    "unable to allocate load snapshots");
    }
  }

  void Publish(std::uint32_t index, Sample sample) noexcept {
    if (index >= count_) {
      return;
    }
    auto& entry = entries_[index];
    entry.sequence.fetch_add(1, std::memory_order_seq_cst);
    std::array<std::uint64_t, kWords> words{};
    std::memcpy(words.data(), &sample, sizeof(Sample));
    for (std::size_t word = 0; word < kWords; ++word) {
      entry.words[word].store(words[word], std::memory_order_seq_cst);
    }
    entry.sequence.fetch_add(1, std::memory_order_seq_cst);
    entry.valid.store(true, std::memory_order_seq_cst);
  }

  [[nodiscard]] Result<Sample> Read(std::uint32_t index) const noexcept {
    if (index >= count_ ||
        !entries_[index].valid.load(std::memory_order_seq_cst)) {
      return Status(StatusCode::kResourceExhausted,
                    "load snapshot is unavailable");
    }
    const auto& entry = entries_[index];
    std::array<std::uint64_t, kWords> words{};
    while (true) {
      const auto before = entry.sequence.load(std::memory_order_seq_cst);
      if ((before & 1U) != 0) {
        continue;
      }
      for (std::size_t word = 0; word < kWords; ++word) {
        words[word] = entry.words[word].load(std::memory_order_seq_cst);
      }
      const auto after = entry.sequence.load(std::memory_order_seq_cst);
      if (before == after) {
        Sample sample{};
        // Sample is trivially copyable; the explicit void pointer marks this as
        // an intentional restoration from raw storage for GCC's diagnostics.
        std::memcpy(static_cast<void*>(&sample), words.data(), sizeof(Sample));
        return sample;
      }
    }
  }

  [[nodiscard]] std::uint32_t size() const noexcept { return count_; }

 private:
  static constexpr std::size_t kWords =
      (sizeof(Sample) + sizeof(std::uint64_t) - 1U) /
      sizeof(std::uint64_t);
  struct alignas(64) Entry {
    std::atomic<std::uint64_t> sequence{0};
    std::atomic<bool> valid{false};
    std::array<std::atomic<std::uint64_t>, kWords> words{};
  };

  LoadSnapshotArray() = default;
  std::uint32_t count_{0};
  std::unique_ptr<Entry[]> entries_;
};

using LaneLoadSnapshots = LoadSnapshotArray<LaneLoadSample>;

[[nodiscard]] std::uint64_t LaneLoadScore(LaneLoadSample sample) noexcept;
[[nodiscard]] Result<std::uint32_t> SelectLanePowerOfTwo(
    QueryKey query, std::uint32_t destination_rank,
    const LaneLoadSnapshots* snapshots, std::uint32_t lane_count) noexcept;

class QueryLaneAffinity {
 public:
  [[nodiscard]] static Result<QueryLaneAffinity> Create(
      std::uint32_t destination_count);
  [[nodiscard]] Result<std::uint32_t> SelectOrGet(
      QueryKey query, std::uint32_t destination_rank,
      const LaneLoadSnapshots* snapshots, std::uint32_t lane_count) noexcept;
  void Clear() noexcept;

 private:
  std::vector<std::uint32_t> lanes_;
};

// Fixed, lock-free affinity used by the shared DistanceOracle adapter. Entries
// for a reused QuerySlot are replaced only by a newer generation; active
// generation mappings never migrate because of changing snapshots.
class ConcurrentLaneAffinityTable {
 public:
  [[nodiscard]] static Result<std::unique_ptr<ConcurrentLaneAffinityTable>>
  Create(std::uint32_t capacity);
  [[nodiscard]] Result<std::uint32_t> SelectOrGet(
      QueryKey query, std::uint32_t destination_rank,
      const LaneLoadSnapshots* snapshots, std::uint32_t lane_count) noexcept;
  [[nodiscard]] std::uint64_t hits() const noexcept {
    return hits_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t inserts() const noexcept {
    return inserts_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t replacements() const noexcept {
    return replacements_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t capacity_rejections() const noexcept {
    return capacity_rejections_.load(std::memory_order_relaxed);
  }

 private:
  enum class EntryState : std::uint8_t { kFree, kWriting, kReady };
  struct alignas(64) Entry {
    std::atomic<EntryState> state{EntryState::kFree};
    std::uint64_t source_slot{0};
    std::uint32_t destination_rank{0};
    std::uint64_t generation{0};
    std::uint32_t lane{0};
  };

  ConcurrentLaneAffinityTable() = default;
  std::uint32_t capacity_{0};
  std::unique_ptr<Entry[]> entries_;
  std::atomic<std::uint64_t> hits_{0};
  std::atomic<std::uint64_t> inserts_{0};
  std::atomic<std::uint64_t> replacements_{0};
  std::atomic<std::uint64_t> capacity_rejections_{0};
};

}  // namespace hnsw_urma
