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

#include "hnsw_urma/runtime/load_balancing.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <thread>
#include <utility>

namespace hnsw_urma {
namespace {

constexpr std::uint32_t kHashA = 0x9e3779b9U;
constexpr std::uint32_t kHashB = 0xd1b54a32U;
constexpr std::uint32_t kHashC = 0x94d049bbU;

std::uint32_t Candidate(QueryKey query, std::uint32_t salt,
                        std::uint32_t count) noexcept {
  // Defensive guard against division by zero. Callers (SelectLanePowerOfTwo)
  // already reject lane_count == 0, but keep the divisor non-zero locally so
  // the modulo cannot trap regardless of future callers (G.EXP.22-CPP).
  if (count == 0)
    return 0;
  return static_cast<std::uint32_t>(StableHash(query, salt) % count);
}

std::pair<std::uint32_t, std::uint32_t>
TwoCandidates(QueryKey query, std::uint32_t salt,
              std::uint32_t count) noexcept {
  const auto first = Candidate(query, salt ^ kHashA, count);
  auto second = Candidate(query, salt ^ kHashB, count);
  if (count > 1 && second == first) {
    second = (second + 1U) % count;
  }
  return {first, second};
}

std::uint64_t SaturatingAdd(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
             ? std::numeric_limits<std::uint64_t>::max()
             : lhs + rhs;
}

} // namespace

std::uint64_t LaneLoadScore(LaneLoadSample sample) noexcept {
  std::uint64_t score = sample.submit_backlog;
  score =
      SaturatingAdd(score, static_cast<std::uint64_t>(sample.outstanding) * 2U);
  score = SaturatingAdd(score,
                        static_cast<std::uint64_t>(sample.active_replies) * 2U);
  score = SaturatingAdd(
      score, static_cast<std::uint64_t>(sample.credit_pressure) * 4U);
  score = SaturatingAdd(
      score, static_cast<std::uint64_t>(sample.timeout_pressure) * 16U);
  score = SaturatingAdd(score, sample.recent_latency_ns / 1000U);
  return score;
}

Result<std::uint32_t> SelectLanePowerOfTwo(QueryKey query,
                                           std::uint32_t destination_rank,
                                           const LaneLoadSnapshots *snapshots,
                                           std::uint32_t lane_count) noexcept {
  if (lane_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "lane selection requires lanes");
  }
  const auto fallback = Candidate(query, destination_rank ^ kHashA, lane_count);
  if (lane_count == 1 || snapshots == nullptr ||
      snapshots->size() < lane_count) {
    return fallback;
  }
  const auto [first, second] =
      TwoCandidates(query, destination_rank, lane_count);
  auto first_load = snapshots->Read(first);
  auto second_load = snapshots->Read(second);
  if (!first_load.ok() || !second_load.ok()) {
    return fallback;
  }
  const bool first_available = first_load.value().reply_capacity_remaining != 0;
  const bool second_available =
      second_load.value().reply_capacity_remaining != 0;
  if (!first_available && !second_available) {
    return Status(StatusCode::kResourceExhausted,
                  "both lane candidates have no reply capacity");
  }
  if (first_available != second_available) {
    return first_available ? first : second;
  }
  const auto first_score = LaneLoadScore(first_load.value());
  const auto second_score = LaneLoadScore(second_load.value());
  if (first_score != second_score) {
    return first_score < second_score ? first : second;
  }
  return (StableHash(query, destination_rank ^ kHashC) & 1U) == 0 ? first
                                                                  : second;
}

Result<QueryLaneAffinity>
QueryLaneAffinity::Create(std::uint32_t destination_count) {
  if (destination_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "query lane affinity requires destinations");
  }
  try {
    QueryLaneAffinity affinity;
    affinity.lanes_.assign(destination_count,
                           std::numeric_limits<std::uint32_t>::max());
    return affinity;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate query lane affinity");
  }
}

Result<std::uint32_t>
QueryLaneAffinity::SelectOrGet(QueryKey query, std::uint32_t destination_rank,
                               const LaneLoadSnapshots *snapshots,
                               std::uint32_t lane_count) noexcept {
  if (destination_rank >= lanes_.size()) {
    return Status(StatusCode::kOutOfRange,
                  "affinity destination rank is out of range");
  }
  if (lanes_[destination_rank] != std::numeric_limits<std::uint32_t>::max()) {
    return lanes_[destination_rank];
  }
  auto selected =
      SelectLanePowerOfTwo(query, destination_rank, snapshots, lane_count);
  if (selected.ok()) {
    lanes_[destination_rank] = selected.value();
  }
  return selected;
}

void QueryLaneAffinity::Clear() noexcept {
  std::fill(lanes_.begin(), lanes_.end(),
            std::numeric_limits<std::uint32_t>::max());
}

Result<std::unique_ptr<ConcurrentLaneAffinityTable>>
ConcurrentLaneAffinityTable::Create(std::uint32_t capacity) {
  if (capacity == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "concurrent lane affinity capacity must be non-zero");
  }
  try {
    auto table = std::unique_ptr<ConcurrentLaneAffinityTable>(
        new ConcurrentLaneAffinityTable());
    table->capacity_ = capacity;
    table->entries_ = std::make_unique<Entry[]>(capacity);
    return table;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate concurrent lane affinity");
  }
}

Result<std::uint32_t> ConcurrentLaneAffinityTable::SelectOrGet(
    QueryKey query, std::uint32_t destination_rank,
    const LaneLoadSnapshots *snapshots, std::uint32_t lane_count) noexcept {
  const auto source_slot =
      (static_cast<std::uint64_t>(query.source_rank) << 32U) | query.query_slot;
  const QueryKey stable_slot_key{query.source_rank, query.query_slot, 0};
  const auto start = StableHash(stable_slot_key, destination_rank) % capacity_;
  const auto retry_limit = static_cast<std::uint64_t>(capacity_) * 8U + 64U;
  for (std::uint64_t attempt = 0; attempt < retry_limit; ++attempt) {
    Entry *first_free = nullptr;
    bool writer_present = false;
    for (std::uint32_t probe = 0; probe < capacity_; ++probe) {
      auto &entry = entries_[(start + probe) % capacity_];
      const auto state = entry.state.load(std::memory_order_acquire);
      if (state == EntryState::kWriting) {
        writer_present = true;
        continue;
      }
      if (state == EntryState::kReady && entry.source_slot == source_slot &&
          entry.destination_rank == destination_rank) {
        if (entry.generation == query.generation) {
          hits_.fetch_add(1, std::memory_order_relaxed);
          return entry.lane;
        }
        if (query.generation < entry.generation) {
          return Status(StatusCode::kProtocolError,
                        "lane affinity rejected a stale query generation");
        }
        EntryState expected = EntryState::kReady;
        if (!entry.state.compare_exchange_strong(expected, EntryState::kWriting,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
          writer_present = true;
          break;
        }
        auto selected = SelectLanePowerOfTwo(query, destination_rank, snapshots,
                                             lane_count);
        if (!selected.ok()) {
          entry.state.store(EntryState::kReady, std::memory_order_release);
          capacity_rejections_.fetch_add(1, std::memory_order_relaxed);
          return selected.status();
        }
        entry.generation = query.generation;
        entry.lane = selected.value();
        entry.state.store(EntryState::kReady, std::memory_order_release);
        replacements_.fetch_add(1, std::memory_order_relaxed);
        return selected;
      }
      if (state == EntryState::kFree && first_free == nullptr) {
        first_free = &entry;
      }
    }
    // A writer may be publishing this same key. Restarting before claiming a
    // free cell guarantees one mapping for a concurrent first submission.
    // This bounded yield only arbitrates a short metadata publication; it is
    // not a wait for network credit, a transport slot, or completion.
    if (writer_present) {
      std::this_thread::yield();
      continue;
    }
    if (first_free == nullptr) {
      break;
    }
    EntryState expected = EntryState::kFree;
    if (!first_free->state.compare_exchange_strong(
            expected, EntryState::kWriting, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      continue;
    }
    auto selected =
        SelectLanePowerOfTwo(query, destination_rank, snapshots, lane_count);
    if (!selected.ok()) {
      first_free->state.store(EntryState::kFree, std::memory_order_release);
      capacity_rejections_.fetch_add(1, std::memory_order_relaxed);
      return selected.status();
    }
    first_free->source_slot = source_slot;
    first_free->destination_rank = destination_rank;
    first_free->generation = query.generation;
    first_free->lane = selected.value();
    first_free->state.store(EntryState::kReady, std::memory_order_release);
    inserts_.fetch_add(1, std::memory_order_relaxed);
    return selected;
  }
  capacity_rejections_.fetch_add(1, std::memory_order_relaxed);
  return Status(StatusCode::kResourceExhausted,
                "concurrent lane affinity table is full");
}

} // namespace hnsw_urma
