#include "hnsw_urma/runtime/topology.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>

#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/core/query_key.hpp"
#include "hnsw_urma/core/spsc_ring.hpp"
#include "hnsw_urma/runtime/scheduler.hpp"
#include "hnsw_urma/runtime/rpc_runtime.hpp"
#include "hnsw_urma/transport/read_batch_lane.hpp"

namespace hnsw_urma {
namespace {

Result<std::uint64_t> Product(std::initializer_list<std::uint64_t> values) {
  std::uint64_t result = 1;
  for (const auto value : values) {
    auto next = CheckedMultiply(result, value);
    if (!next.ok()) {
      return next.status();
    }
    result = next.value();
  }
  return result;
}

Status Add(std::uint64_t value, std::uint64_t& total) {
  auto next = CheckedAdd(total, value);
  if (!next.ok()) {
    return next.status();
  }
  total = next.value();
  return Status::Ok();
}

Status AddStorage(std::uint64_t count, std::uint64_t depth,
                  std::uint64_t element_bytes, std::uint64_t& total) {
  auto bytes = Product({count, depth, element_bytes});
  if (!bytes.ok()) {
    return bytes.status();
  }
  return Add(bytes.value(), total);
}

Status AssignOwners(std::uint32_t world_size, std::uint32_t local_rank,
                    std::uint32_t thread_count,
                    std::vector<std::uint32_t>& owner_by_rank,
                    std::vector<std::vector<std::uint32_t>>& owned_ranks) {
  if (thread_count == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "communication thread count must be non-zero");
  }
  try {
    owner_by_rank.assign(world_size, kNoCommunicationOwner);
    owned_ranks.resize(thread_count);
    std::uint32_t remote_index = 0;
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      if (rank == local_rank) {
        continue;
      }
      const auto owner = remote_index % thread_count;
      owner_by_rank[rank] = owner;
      owned_ranks[owner].push_back(rank);
      ++remote_index;
    }
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate communication owner map");
  }
  return Status::Ok();
}

Result<std::vector<std::uint32_t>> ValidateAndSortCpus(
    std::span<const std::uint32_t> allowed_cpus,
    CpuRoleCounts counts) {
  const auto requested =
      static_cast<std::uint64_t>(counts.search) + counts.distance +
      counts.read_progress + counts.rpc_progress + counts.reserved;
  if (allowed_cpus.empty() || counts.search == 0 ||
      counts.distance == 0 || counts.read_progress == 0 ||
      counts.rpc_progress == 0 || counts.reserved == 0 ||
      requested != allowed_cpus.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "CPU role counts do not cover the allowed CPU set");
  }
  try {
    std::vector<std::uint32_t> sorted(
        allowed_cpus.begin(), allowed_cpus.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      return Status(StatusCode::kInvalidArgument,
                    "allowed CPU set contains duplicates");
    }
    return sorted;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate CPU layout input");
  }
}

#if defined(__linux__)
Result<std::uint32_t> ReadSysfsU32(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  std::uint64_t value = 0;
  if (!input || !(input >> value) ||
      value > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kInternalError,
                  "unable to read CPU topology value")
        .AddContext("path", path.string());
  }
  return static_cast<std::uint32_t>(value);
}

Result<std::uint32_t> ReadSharedCpuListId(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string value;
  if (!input || !(input >> value)) {
    return Status(StatusCode::kInternalError,
                  "unable to read LLC shared CPU list")
        .AddContext("path", path.string());
  }
  const auto delimiter = value.find_first_of("-,");
  const auto first = value.substr(0, delimiter);
  try {
    const auto parsed = std::stoull(first);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
      throw std::out_of_range("LLC CPU ID");
    }
    return static_cast<std::uint32_t>(parsed);
  } catch (const std::exception&) {
    return Status(StatusCode::kInternalError,
                  "LLC shared CPU list is malformed")
        .AddContext("path", path.string())
        .AddContext("value", value);
  }
}

Result<std::uint32_t> DiscoverLlcId(std::uint32_t cpu) {
  const auto cache_root = std::filesystem::path(
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
      "/cache");
  std::error_code error;
  std::vector<std::filesystem::path> indexes;
  for (std::filesystem::directory_iterator iterator(cache_root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    if (iterator->is_directory(error)) {
      indexes.push_back(iterator->path());
    }
  }
  if (error) {
    return Status(StatusCode::kInternalError,
                  "unable to enumerate CPU cache topology")
        .AddContext("cpu", std::to_string(cpu))
        .AddContext("path", cache_root.string());
  }
  std::sort(indexes.begin(), indexes.end());
  for (const auto& index : indexes) {
    auto level = ReadSysfsU32(index / "level");
    if (!level.ok() || level.value() != 3U) {
      continue;
    }
    auto id = ReadSysfsU32(index / "id");
    if (id.ok()) {
      return id;
    }
    return ReadSharedCpuListId(index / "shared_cpu_list");
  }
  return Status(StatusCode::kInternalError,
                "CPU has no discoverable shared L3 cache")
      .AddContext("cpu", std::to_string(cpu));
}

Result<std::uint32_t> DiscoverNumaNode(std::uint32_t cpu) {
  const auto cpu_root = std::filesystem::path(
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu));
  std::error_code error;
  std::optional<std::uint32_t> node;
  for (std::filesystem::directory_iterator iterator(cpu_root, error), end;
       !error && iterator != end; iterator.increment(error)) {
    const auto name = iterator->path().filename().string();
    if (!name.starts_with("node") || name.size() == 4U) {
      continue;
    }
    std::uint32_t parsed = 0;
    const auto first = name.data() + 4;
    const auto last = name.data() + name.size();
    const auto result = std::from_chars(first, last, parsed);
    if (result.ec != std::errc{} || result.ptr != last || node.has_value()) {
      return Status(StatusCode::kInternalError,
                    "CPU has an invalid NUMA topology")
          .AddContext("cpu", std::to_string(cpu))
          .AddContext("path", iterator->path().string());
    }
    node = parsed;
  }
  if (error || !node.has_value()) {
    return Status(StatusCode::kInternalError,
                  "unable to discover CPU NUMA node")
        .AddContext("cpu", std::to_string(cpu))
        .AddContext("path", cpu_root.string());
  }
  return *node;
}
#endif

}  // namespace

Result<std::vector<CpuTopologyEntry>> DiscoverCpuTopology(
    std::span<const std::uint32_t> allowed_cpus) {
#if !defined(__linux__)
  (void)allowed_cpus;
  return Status(StatusCode::kInternalError,
                "CPU LLC topology discovery requires Linux sysfs");
#else
  if (allowed_cpus.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "allowed CPU set is empty");
  }
  try {
    std::vector<CpuTopologyEntry> topology;
    topology.reserve(allowed_cpus.size());
    std::set<std::uint32_t> seen;
    for (const auto cpu : allowed_cpus) {
      if (!seen.insert(cpu).second) {
        return Status(StatusCode::kInvalidArgument,
                      "allowed CPU set contains duplicates");
      }
      auto core = ReadSysfsU32(
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
          "/topology/core_id");
      auto llc = DiscoverLlcId(cpu);
      if (!core.ok() || !llc.ok()) {
        auto status = core.ok() ? llc.status() : core.status();
        status.AddContext("cpu", std::to_string(cpu));
        return status;
      }
      topology.push_back({cpu, core.value(), llc.value()});
    }
    std::sort(topology.begin(), topology.end(),
              [](const auto& left, const auto& right) {
                return left.cpu < right.cpu;
              });
    return topology;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate CPU topology");
  }
#endif
}

Result<std::vector<std::uint32_t>> DiscoverNumaNodes(
    std::span<const std::uint32_t> allowed_cpus) {
#if !defined(__linux__)
  (void)allowed_cpus;
  return Status(StatusCode::kInternalError,
                "CPU NUMA discovery requires Linux sysfs");
#else
  if (allowed_cpus.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "allowed CPU set is empty");
  }
  try {
    std::set<std::uint32_t> cpus;
    std::set<std::uint32_t> nodes;
    for (const auto cpu : allowed_cpus) {
      if (!cpus.insert(cpu).second) {
        return Status(StatusCode::kInvalidArgument,
                      "allowed CPU set contains duplicates");
      }
      auto node = DiscoverNumaNode(cpu);
      if (!node.ok()) {
        return node.status();
      }
      nodes.insert(node.value());
    }
    return std::vector<std::uint32_t>(nodes.begin(), nodes.end());
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate NUMA topology");
  }
#endif
}

Result<CpuRoleLayout> BuildContiguousCpuLayout(
    std::span<const std::uint32_t> allowed_cpus,
    CpuRoleCounts counts) {
  auto sorted = ValidateAndSortCpus(allowed_cpus, counts);
  if (!sorted.ok()) {
    return sorted.status();
  }
  CpuRoleLayout layout;
  try {
    std::size_t cursor = 0;
    const auto assign = [&](std::uint32_t count,
                            std::vector<std::uint32_t>& output) {
      output.insert(
          output.end(), sorted.value().begin() +
                            static_cast<std::ptrdiff_t>(cursor),
          sorted.value().begin() +
              static_cast<std::ptrdiff_t>(cursor + count));
      cursor += count;
    };
    assign(counts.search, layout.search);
    assign(counts.distance, layout.distance);
    assign(counts.read_progress, layout.read_progress);
    assign(counts.rpc_progress, layout.rpc_progress);
    assign(counts.reserved, layout.reserved);
    return layout;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate contiguous CPU layout");
  }
}

Result<CpuRoleLayout> BuildLlcIsolatedCpuLayout(
    std::span<const std::uint32_t> allowed_cpus,
    std::span<const CpuTopologyEntry> topology,
    CpuRoleCounts counts) {
  auto sorted = ValidateAndSortCpus(allowed_cpus, counts);
  if (!sorted.ok()) {
    return sorted.status();
  }
  if (topology.size() != sorted.value().size()) {
    return Status(StatusCode::kInvalidArgument,
                  "CPU topology does not cover the allowed CPU set");
  }
  try {
    std::set<std::uint32_t> allowed(
        sorted.value().begin(), sorted.value().end());
    std::set<std::uint32_t> topology_cpus;
    std::map<std::uint32_t, std::vector<std::uint32_t>> domains;
    for (const auto& entry : topology) {
      if (!allowed.contains(entry.cpu) ||
          !topology_cpus.insert(entry.cpu).second) {
        return Status(StatusCode::kInvalidArgument,
                      "CPU topology contains an unknown or duplicate CPU");
      }
      domains[entry.llc_id].push_back(entry.cpu);
    }
    for (auto& [llc, cpus] : domains) {
      (void)llc;
      std::sort(cpus.begin(), cpus.end());
    }
    if (domains.empty()) {
      return Status(StatusCode::kInvalidArgument,
                    "CPU topology contains no LLC domains");
    }

    const auto rpc_on_search = std::min(counts.rpc_progress, 1U);
    const auto search_base = static_cast<std::size_t>(counts.search) +
        counts.read_progress + rpc_on_search + 1U;
    const auto extra_reserve = counts.reserved - 1U;
    std::size_t search_side_size = search_base + extra_reserve;
    std::size_t prefix = 0;
    for (const auto& [llc, cpus] : domains) {
      (void)llc;
      prefix += cpus.size();
      if (prefix >= search_base &&
          prefix <= search_base + extra_reserve) {
        search_side_size = prefix;
      }
    }

    struct DomainSlice {
      std::vector<std::uint32_t> search_side;
      std::vector<std::uint32_t> distance_side;
    };
    std::vector<DomainSlice> slices;
    slices.reserve(domains.size());
    std::size_t search_remaining = search_side_size;
    for (const auto& [llc, cpus] : domains) {
      (void)llc;
      DomainSlice slice;
      const auto search_count =
          std::min(search_remaining, cpus.size());
      slice.search_side.insert(
          slice.search_side.end(), cpus.begin(),
          cpus.begin() + static_cast<std::ptrdiff_t>(search_count));
      slice.distance_side.insert(
          slice.distance_side.end(),
          cpus.begin() + static_cast<std::ptrdiff_t>(search_count),
          cpus.end());
      search_remaining -= search_count;
      slices.push_back(std::move(slice));
    }
    if (search_remaining != 0) {
      return Status(StatusCode::kInternalError,
                    "LLC planner could not fill the Search side");
    }

    std::vector<std::uint32_t> search_cpus;
    search_cpus.reserve(search_side_size);
    std::vector<std::vector<std::uint32_t>> distance_domains;
    for (const auto& slice : slices) {
      search_cpus.insert(search_cpus.end(),
                         slice.search_side.begin(),
                         slice.search_side.end());
      if (!slice.distance_side.empty()) {
        distance_domains.push_back(slice.distance_side);
      }
    }

    CpuRoleLayout layout;
    std::size_t cursor = 0;
    const auto assign_search = [&](std::uint32_t count,
                                   std::vector<std::uint32_t>& output) {
      output.insert(
          output.end(), search_cpus.begin() +
                            static_cast<std::ptrdiff_t>(cursor),
          search_cpus.begin() +
              static_cast<std::ptrdiff_t>(cursor + count));
      cursor += count;
    };
    assign_search(counts.search, layout.search);
    assign_search(counts.read_progress, layout.read_progress);
    assign_search(rpc_on_search, layout.rpc_progress);
    const auto search_reserve = static_cast<std::uint32_t>(
        search_side_size - search_base + 1U);
    assign_search(search_reserve, layout.reserved);

    const auto rpc_on_distance = counts.rpc_progress - rpc_on_search;
    std::vector<std::size_t> rpc_quota(distance_domains.size(), 0);
    std::size_t remaining_rpc = rpc_on_distance;
    while (remaining_rpc != 0) {
      bool assigned = false;
      for (std::size_t offset = 0;
           offset < distance_domains.size() && remaining_rpc != 0;
           ++offset) {
        const auto domain = distance_domains.size() - 1U - offset;
        if (rpc_quota[domain] < distance_domains[domain].size()) {
          ++rpc_quota[domain];
          --remaining_rpc;
          assigned = true;
        }
      }
      if (!assigned) {
        return Status(StatusCode::kResourceExhausted,
                      "LLC planner cannot place RPC progress threads");
      }
    }
    std::vector<std::size_t> distance_quota(
        distance_domains.size(), 0);
    std::size_t remaining_distance = counts.distance;
    while (remaining_distance != 0) {
      bool assigned = false;
      for (std::size_t domain = 0;
           domain < distance_domains.size() && remaining_distance != 0;
           ++domain) {
        const auto capacity =
            distance_domains[domain].size() - rpc_quota[domain];
        if (distance_quota[domain] < capacity) {
          ++distance_quota[domain];
          --remaining_distance;
          assigned = true;
        }
      }
      if (!assigned) {
        return Status(StatusCode::kResourceExhausted,
                      "LLC planner cannot place Distance workers");
      }
    }
    for (std::size_t domain = 0;
         domain < distance_domains.size(); ++domain) {
      const auto& cpus = distance_domains[domain];
      const auto distance_count = distance_quota[domain];
      const auto rpc_count = rpc_quota[domain];
      layout.distance.insert(
          layout.distance.end(), cpus.begin(),
          cpus.begin() +
              static_cast<std::ptrdiff_t>(distance_count));
      layout.rpc_progress.insert(
          layout.rpc_progress.end(),
          cpus.begin() +
              static_cast<std::ptrdiff_t>(distance_count),
          cpus.begin() + static_cast<std::ptrdiff_t>(
                             distance_count + rpc_count));
      layout.reserved.insert(
          layout.reserved.end(),
          cpus.begin() + static_cast<std::ptrdiff_t>(
                             distance_count + rpc_count),
          cpus.end());
    }
    if (layout.search.size() != counts.search ||
        layout.distance.size() != counts.distance ||
        layout.read_progress.size() != counts.read_progress ||
        layout.rpc_progress.size() != counts.rpc_progress ||
        layout.reserved.size() != counts.reserved) {
      return Status(StatusCode::kInternalError,
                    "LLC planner produced incorrect role counts");
    }
    return layout;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate LLC-aware CPU layout");
  }
}

Result<CpuRoleLayout> BuildLlcSpreadCpuLayout(
    std::span<const std::uint32_t> allowed_cpus,
    std::span<const CpuTopologyEntry> topology,
    CpuRoleCounts counts) {
  auto sorted = ValidateAndSortCpus(allowed_cpus, counts);
  if (!sorted.ok()) {
    return sorted.status();
  }
  if (topology.size() != sorted.value().size()) {
    return Status(StatusCode::kInvalidArgument,
                  "CPU topology does not cover the allowed CPU set");
  }
  try {
    std::set<std::uint32_t> allowed(
        sorted.value().begin(), sorted.value().end());
    std::set<std::uint32_t> topology_cpus;
    std::map<std::uint32_t, std::vector<std::uint32_t>> grouped;
    for (const auto& entry : topology) {
      if (!allowed.contains(entry.cpu) ||
          !topology_cpus.insert(entry.cpu).second) {
        return Status(StatusCode::kInvalidArgument,
                      "CPU topology contains an unknown or duplicate CPU");
      }
      grouped[entry.llc_id].push_back(entry.cpu);
    }
    if (grouped.empty()) {
      return Status(StatusCode::kInvalidArgument,
                    "CPU topology contains no LLC domains");
    }

    std::vector<std::vector<std::uint32_t>> domains;
    domains.reserve(grouped.size());
    for (auto& [llc, cpus] : grouped) {
      (void)llc;
      std::sort(cpus.begin(), cpus.end());
      domains.push_back(std::move(cpus));
    }

    // Equal-size LLCs differ by at most one Distance worker. Incomplete
    // allowed domains receive a proportional share. The worker vector is
    // interleaved below so low worker ids, including Service ingress owners,
    // land on distinct LLCs before any LLC receives a second worker.
    std::vector<std::size_t> distance_quota(domains.size(), 0);
    for (std::uint32_t remaining = counts.distance; remaining != 0;
         --remaining) {
      std::size_t selected = domains.size();
      for (std::size_t domain = 0; domain < domains.size(); ++domain) {
        if (distance_quota[domain] >= domains[domain].size()) {
          continue;
        }
        if (selected == domains.size()) {
          selected = domain;
          continue;
        }
        const auto left =
            static_cast<std::uint64_t>(distance_quota[domain]) *
            domains[selected].size();
        const auto right =
            static_cast<std::uint64_t>(distance_quota[selected]) *
            domains[domain].size();
        if (left < right ||
            (left == right &&
             (distance_quota[domain] < distance_quota[selected] ||
              (distance_quota[domain] == distance_quota[selected] &&
               domain < selected)))) {
          selected = domain;
        }
      }
      if (selected == domains.size()) {
        return Status(StatusCode::kResourceExhausted,
                      "LLC spread planner cannot place Distance workers");
      }
      ++distance_quota[selected];
    }

    std::vector<std::vector<std::uint32_t>> distance_by_domain(
        domains.size());
    std::vector<std::vector<std::uint32_t>> available_by_domain(
        domains.size());
    for (std::size_t domain = 0; domain < domains.size(); ++domain) {
      const auto split = domains[domain].size() - distance_quota[domain];
      available_by_domain[domain].insert(
          available_by_domain[domain].end(),
          domains[domain].begin(),
          domains[domain].begin() + static_cast<std::ptrdiff_t>(split));
      distance_by_domain[domain].insert(
          distance_by_domain[domain].end(),
          domains[domain].begin() + static_cast<std::ptrdiff_t>(split),
          domains[domain].end());
    }

    // Spread Search workers with the same normalized LLC-capacity rule used
    // for Distance workers, bounded by the capacity left after the Distance
    // allocation. Interleave the output vector below so low Search worker ids
    // land on distinct LLCs before an LLC receives a second worker.
    std::vector<std::size_t> search_by_domain(domains.size(), 0);
    for (std::uint32_t remaining = counts.search; remaining != 0;
         --remaining) {
      std::size_t selected = domains.size();
      for (std::size_t domain = 0; domain < domains.size(); ++domain) {
        if (search_by_domain[domain] >=
            available_by_domain[domain].size()) {
          continue;
        }
        if (selected == domains.size()) {
          selected = domain;
          continue;
        }
        const auto left =
            static_cast<std::uint64_t>(search_by_domain[domain]) *
            domains[selected].size();
        const auto right =
            static_cast<std::uint64_t>(search_by_domain[selected]) *
            domains[domain].size();
        if (left < right ||
            (left == right &&
             (search_by_domain[domain] < search_by_domain[selected] ||
              (search_by_domain[domain] == search_by_domain[selected] &&
               domain < selected)))) {
          selected = domain;
        }
      }
      if (selected == domains.size()) {
        return Status(StatusCode::kResourceExhausted,
                      "LLC spread planner cannot place Search workers");
      }
      ++search_by_domain[selected];
    }

    CpuRoleLayout layout;
    std::vector<std::size_t> available_cursor = search_by_domain;
    std::size_t maximum_search_quota = 0;
    for (const auto quota : search_by_domain) {
      maximum_search_quota = std::max(maximum_search_quota, quota);
    }
    for (std::size_t offset = 0; offset < maximum_search_quota; ++offset) {
      for (std::size_t domain = 0; domain < domains.size(); ++domain) {
        if (offset < search_by_domain[domain]) {
          layout.search.push_back(available_by_domain[domain][offset]);
        }
      }
    }

    std::vector<std::size_t> read_by_domain(domains.size(), 0);
    std::vector<std::size_t> rpc_by_domain(domains.size(), 0);
    std::size_t packed_domain = 0;
    const auto assign_packed = [&](std::uint32_t count,
                                   std::vector<std::uint32_t>& output,
                                   std::vector<std::size_t>& per_domain) {
      while (count != 0) {
        while (packed_domain < domains.size() &&
               available_cursor[packed_domain] >=
                   available_by_domain[packed_domain].size()) {
          ++packed_domain;
        }
        if (packed_domain == domains.size()) {
          return false;
        }
        output.push_back(
            available_by_domain[packed_domain]
                               [available_cursor[packed_domain]++]);
        ++per_domain[packed_domain];
        --count;
      }
      return true;
    };
    if (!assign_packed(
            counts.read_progress, layout.read_progress,
            read_by_domain)) {
      return Status(StatusCode::kResourceExhausted,
                    "LLC spread planner cannot place Read workers");
    }

    // Prefer LLCs without Search/Read work for RPC progress, then balance
    // combined Distance+RPC pressure normalized by LLC capacity.
    for (std::uint32_t rpc = 0; rpc < counts.rpc_progress; ++rpc) {
      std::size_t selected = domains.size();
      for (std::size_t domain = 0; domain < domains.size(); ++domain) {
        if (available_cursor[domain] >=
            available_by_domain[domain].size()) {
          continue;
        }
        if (selected == domains.size()) {
          selected = domain;
          continue;
        }
        const bool mixed =
            search_by_domain[domain] != 0 || read_by_domain[domain] != 0;
        const bool selected_mixed =
            search_by_domain[selected] != 0 ||
            read_by_domain[selected] != 0;
        const auto pressure = distance_quota[domain] + rpc_by_domain[domain];
        const auto selected_pressure =
            distance_quota[selected] + rpc_by_domain[selected];
        const auto left = static_cast<std::uint64_t>(pressure) *
                          domains[selected].size();
        const auto right = static_cast<std::uint64_t>(selected_pressure) *
                           domains[domain].size();
        if (mixed < selected_mixed ||
            (mixed == selected_mixed &&
             (left < right ||
              (left == right &&
               (rpc_by_domain[domain] < rpc_by_domain[selected] ||
                (rpc_by_domain[domain] == rpc_by_domain[selected] &&
                 domain < selected)))))) {
          selected = domain;
        }
      }
      if (selected == domains.size()) {
        return Status(StatusCode::kResourceExhausted,
                      "LLC spread planner cannot place RPC workers");
      }
      layout.rpc_progress.push_back(
          available_by_domain[selected][available_cursor[selected]++]);
      ++rpc_by_domain[selected];
    }

    std::size_t maximum_distance_quota = 0;
    for (const auto quota : distance_quota) {
      maximum_distance_quota = std::max(maximum_distance_quota, quota);
    }
    for (std::size_t offset = 0; offset < maximum_distance_quota; ++offset) {
      for (const auto& cpus : distance_by_domain) {
        if (offset < cpus.size()) {
          layout.distance.push_back(cpus[offset]);
        }
      }
    }

    for (std::size_t domain = 0; domain < domains.size(); ++domain) {
      layout.reserved.insert(
          layout.reserved.end(),
          available_by_domain[domain].begin() +
              static_cast<std::ptrdiff_t>(available_cursor[domain]),
          available_by_domain[domain].end());
    }
    if (layout.search.size() != counts.search ||
        layout.distance.size() != counts.distance ||
        layout.read_progress.size() != counts.read_progress ||
        layout.rpc_progress.size() != counts.rpc_progress ||
        layout.reserved.size() != counts.reserved) {
      return Status(StatusCode::kInternalError,
                    "LLC spread planner produced incorrect role counts");
    }
    return layout;
  } catch (const std::bad_alloc&) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate LLC spread CPU layout");
  }
}

Result<CommunicationOwnerMap> BuildCommunicationOwnerMap(
    std::uint32_t world_size, std::uint32_t local_rank,
    std::uint32_t read_threads, std::uint32_t rpc_threads) {
  if (world_size == 0 || local_rank >= world_size) {
    return Status(StatusCode::kInvalidArgument,
                  "owner map world/local rank is invalid");
  }
  CommunicationOwnerMap result;
  auto read = AssignOwners(world_size, local_rank, read_threads,
                           result.read_owner_by_rank,
                           result.read_owned_ranks);
  if (!read.ok()) {
    return read;
  }
  auto rpc = AssignOwners(world_size, local_rank, rpc_threads,
                          result.rpc_owner_by_rank,
                          result.rpc_owned_ranks);
  if (!rpc.ok()) {
    return rpc;
  }
  return result;
}

Result<RuntimeQueuePlan> BuildRuntimeQueuePlan(
    std::uint32_t search_workers, std::uint32_t read_threads,
    std::uint32_t rpc_threads, RuntimeQueueDepths depths,
    bool profiling_enabled, std::uint32_t world_size) {
  if (search_workers == 0 || read_threads == 0 ||
      rpc_threads == 0 || depths.admission == 0 || depths.ready == 0 ||
      depths.read_submit == 0 || depths.read_reclaim == 0 ||
      depths.rpc_submit == 0 ||
      depths.done == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "queue plan dimensions must be non-zero");
  }

  RuntimeQueuePlan plan;
  plan.depths = depths;
  plan.admission_rings = search_workers;
  plan.done_rings = search_workers;
  auto rpc_submit = Product({search_workers, rpc_threads});
  if (!rpc_submit.ok()) {
    return Status(StatusCode::kOutOfRange,
                  "runtime queue dimensions overflow");
  }
  plan.read_submit_rings = static_cast<std::uint64_t>(search_workers) * read_threads;
  plan.read_reclaim_rings = plan.read_submit_rings;
  const auto producer_count = static_cast<std::uint64_t>(read_threads) + rpc_threads;
  if (producer_count > UINT32_MAX)
    return Status(StatusCode::kOutOfRange, "producer count overflows");
  plan.deferred_ready_entries = producer_count * depths.admission;
  plan.retry_entries = static_cast<std::uint64_t>(search_workers) * depths.admission;
  plan.rpc_submit_rings = rpc_submit.value();
  plan.ready_rings = plan.mpsc_ring_count = search_workers;
  plan.depths.ready = std::max(2U, depths.ready);

  for (const auto count : {
           plan.admission_rings, plan.done_rings,
           plan.rpc_submit_rings, plan.read_submit_rings,
           plan.read_reclaim_rings}) {
    auto status = Add(count, plan.spsc_ring_count);
    if (!status.ok()) {
      return status;
    }
  }
  auto mpsc_bytes = MpmcRing<SearchEvent>::StorageBytes(plan.depths.ready);
  if (!mpsc_bytes.ok()) return mpsc_bytes.status();
  auto submit_auxiliary = RpcSubmitMatrix::AuxiliaryBytes(search_workers, rpc_threads, world_size);
  if (!submit_auxiliary.ok()) return submit_auxiliary.status();
  plan.private_work_bytes = sizeof(SchedulerQueues) + sizeof(SchedulerReadyPublisher);
  auto added = Add(submit_auxiliary.value(), plan.private_work_bytes);
  if (!added.ok()) return added;
  if (profiling_enabled) {
    auto profile_bytes = SchedulerQueues::ProfileStorageBytes(search_workers, static_cast<std::uint32_t>(producer_count));
    if (!profile_bytes.ok()) return profile_bytes.status();
    auto status = Add(profile_bytes.value(), plan.private_work_bytes);
    if (!status.ok()) return status;
  }
  for (auto status : {
      AddStorage(producer_count, 1, sizeof(DeferredReadyWork), plan.private_work_bytes),
      AddStorage(1, plan.deferred_ready_entries, sizeof(QueryDeferredReadyEntry), plan.private_work_bytes),
      AddStorage(search_workers, 1, sizeof(LocalRetryQueue) + sizeof(std::size_t), plan.private_work_bytes),
      AddStorage(1, plan.retry_entries, sizeof(QueryRetryHandle), plan.private_work_bytes)}) {
    if (!status.ok()) return status;
  }
  struct QueueGroup {
    std::uint64_t count;
    std::uint64_t depth;
    std::uint64_t element_bytes;
    std::uint64_t object_bytes;
  };
  const QueueGroup groups[]{
      {plan.admission_rings, depths.admission, sizeof(QueryKey), sizeof(SpscRing<QueryKey>)},
      {plan.done_rings, depths.done, sizeof(QueryDoneHandle), sizeof(SpscRing<QueryDoneHandle>)},
      {plan.read_submit_rings, depths.read_submit,
       sizeof(ReadBatch*), sizeof(SpscRing<ReadBatch*>)},
      {plan.read_reclaim_rings, depths.read_reclaim,
       sizeof(ReadBatch*), sizeof(SpscRing<ReadBatch*>)},
      {plan.rpc_submit_rings, depths.rpc_submit, sizeof(RpcSubmitHandle), sizeof(SpscRing<RpcSubmitHandle>)},
      {plan.ready_rings, 1, mpsc_bytes.value(), sizeof(MpmcRing<SearchEvent>)},
  };
  for (const auto& group : groups) {
    auto metadata = AddStorage(group.count, 1, group.object_bytes, plan.ring_metadata_bytes);
    if (!metadata.ok()) return metadata;
    auto status = AddStorage(group.count, group.depth, group.element_bytes,
                             plan.ring_storage_bytes);
    if (!status.ok()) {
      return status;
    }
  }
  for (const auto bytes : {plan.ring_metadata_bytes,
                           plan.ring_storage_bytes, plan.private_work_bytes}) {
    auto status = Add(bytes, plan.queue_bytes_total);
    if (!status.ok()) {
      return status;
    }
  }
  return plan;
}

std::string FormatCommunicationOwnerMap(const CommunicationOwnerMap& owners) {
  std::ostringstream output;
  for (std::size_t rank = 0; rank < owners.read_owner_by_rank.size(); ++rank) {
    if (owners.read_owner_by_rank[rank] == kNoCommunicationOwner) {
      continue;
    }
    output << "rank=" << rank
           << " read_owner=" << owners.read_owner_by_rank[rank]
           << " rpc_owner=" << owners.rpc_owner_by_rank[rank] << '\n';
  }
  return output.str();
}

}  // namespace hnsw_urma
