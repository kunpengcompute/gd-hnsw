#include "hnsw_urma/runtime/config.hpp"
#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/runtime/rpc/distance_oracle.hpp"
#include "hnsw_urma/search/read_resources.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <limits>
#include <set>
#include <sstream>

#include "hnsw_urma/core/checked_math.hpp"
#include "hnsw_urma/protocol/distance_batch_wire.hpp"
#include "hnsw_urma/runtime/git_commit.hpp"

namespace hnsw_urma {
namespace {

Result<std::uint64_t> ParseU64(std::string_view text, std::string_view option) {
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "invalid unsigned value for " + std::string(option));
  }
  return value;
}

Status AssignU32(std::string_view text, std::string_view option,
                 std::uint32_t &destination, bool allow_zero = false) {
  auto parsed = ParseU64(text, option);
  if (!parsed.ok()) {
    return parsed.status();
  }
  if (parsed.value() > std::numeric_limits<std::uint32_t>::max() ||
      (!allow_zero && parsed.value() == 0)) {
    return Status(StatusCode::kOutOfRange,
                  "numeric value is outside the range for " +
                      std::string(option));
  }
  destination = static_cast<std::uint32_t>(parsed.value());
  return Status::Ok();
}

Status AssignU64(std::string_view text, std::string_view option,
                 std::uint64_t &destination) {
  auto parsed = ParseU64(text, option);
  if (!parsed.ok()) {
    return parsed.status();
  }
  if (parsed.value() == 0) {
    return Status(StatusCode::kOutOfRange,
                  "numeric value must be non-zero for " + std::string(option));
  }
  destination = parsed.value();
  return Status::Ok();
}

Status AssignAutoU32(std::string_view text, std::string_view option,
                     AutoU32 &destination, bool allow_zero = false) {
  if (text == "auto") {
    destination = {};
    return Status::Ok();
  }
  destination.automatic = false;
  return AssignU32(text, option, destination.value, allow_zero);
}

Status AssignUrmaDeviceMap(std::string_view text,
                           std::map<std::uint32_t, std::string> &destination) {
  if (text.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "--urma-device-map must not be empty");
  }
  std::map<std::uint32_t, std::string> parsed;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const auto comma = text.find(',', offset);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto entry = text.substr(offset, end - offset);
    const auto equals = entry.find('=');
    if (entry.empty() || equals == std::string_view::npos || equals == 0 ||
        equals + 1U == entry.size() ||
        entry.find('=', equals + 1U) != std::string_view::npos) {
      return Status(StatusCode::kInvalidArgument,
                    "invalid --urma-device-map entry")
          .AddContext("entry", std::string(entry));
    }
    auto node = ParseU64(entry.substr(0, equals), "--urma-device-map");
    if (!node.ok() ||
        node.value() > std::numeric_limits<std::uint32_t>::max()) {
      return node.ok() ? Status(StatusCode::kOutOfRange,
                                "NUMA node in --urma-device-map exceeds uint32")
                       : node.status();
    }
    const auto device = entry.substr(equals + 1U);
    if (std::any_of(device.begin(), device.end(), [](char character) {
          return std::isspace(static_cast<unsigned char>(character)) != 0;
        })) {
      return Status(StatusCode::kInvalidArgument,
                    "URMA device name in --urma-device-map contains whitespace")
          .AddContext("device", std::string(device));
    }
    if (!parsed
             .emplace(static_cast<std::uint32_t>(node.value()),
                      std::string(device))
             .second) {
      return Status(StatusCode::kInvalidArgument,
                    "duplicate NUMA node in --urma-device-map")
          .AddContext("numa_node", std::to_string(node.value()));
    }
    if (comma == std::string_view::npos) {
      break;
    }
    offset = comma + 1U;
    if (offset == text.size()) {
      return Status(StatusCode::kInvalidArgument,
                    "--urma-device-map has an empty trailing entry");
    }
  }
  destination = std::move(parsed);
  return Status::Ok();
}

template <typename Enum>
Status
ParseEnum(std::string_view text, std::string_view option,
          std::initializer_list<std::pair<std::string_view, Enum>> values,
          Enum &destination) {
  for (const auto &[name, value] : values) {
    if (text == name) {
      destination = value;
      return Status::Ok();
    }
  }
  return Status(StatusCode::kInvalidArgument,
                "invalid enum value for " + std::string(option));
}

bool IsPowerOfTwo(std::uint32_t value) noexcept {
  return value != 0 && (value & (value - 1U)) == 0;
}

Result<std::uint32_t> NextPowerOfTwo(std::uint64_t value) noexcept {
  if (value == 0 || value > (std::uint64_t{1} << 31U)) {
    return Status(StatusCode::kOutOfRange,
                  "automatic power-of-two value is too large");
  }
  std::uint32_t result = 1;
  while (result < value) {
    result <<= 1U;
  }
  return result;
}

std::uint32_t Resolve(const AutoU32 &option, std::uint32_t automatic) noexcept {
  return option.automatic ? automatic : option.value;
}

Result<std::uint32_t>
PerOperationCompletionDepth(std::uint32_t send_depth) noexcept {
  return CheckedAdd(send_depth, 1U);
}

std::string AutoText(const AutoU32 &value) {
  return value.automatic ? "auto" : std::to_string(value.value);
}

std::string
UrmaDeviceMapText(const std::map<std::uint32_t, std::string> &devices) {
  std::string output;
  for (const auto &[node, device] : devices) {
    if (!output.empty()) {
      output += ',';
    }
    output += std::to_string(node) + '=' + device;
  }
  return output;
}

std::string NumaNodesText(std::span<const std::uint32_t> nodes) {
  std::string output;
  for (const auto node : nodes) {
    if (!output.empty()) {
      output += ',';
    }
    output += std::to_string(node);
  }
  return output;
}

Status Set(CanonicalConfig &canonical, std::string key, std::string value) {
  return canonical.Set(std::move(key), std::move(value));
}

Result<std::uint64_t> Product(std::initializer_list<std::uint64_t> values) {
  std::uint64_t result = 1;
  for (const auto value : values) {
    auto multiplied = CheckedMultiply(result, value);
    if (!multiplied.ok()) {
      return multiplied.status();
    }
    result = multiplied.value();
  }
  return result;
}

} // namespace

Result<RpcMemoryPlan>
BuildRpcMemoryPlan(std::uint32_t world_size, std::uint32_t rpc_threads,
                   std::uint32_t mailbox_depth,
                   std::uint32_t request_send_depth,
                   std::uint32_t request_batch_max_bytes,
                   std::uint32_t response_batch_max_bytes) noexcept {
  if (world_size == 0 || rpc_threads == 0 || mailbox_depth == 0 ||
      request_send_depth == 0 || request_batch_max_bytes == 0 ||
      response_batch_max_bytes == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "RPC memory plan dimensions must be non-zero");
  }
  auto request_stride =
      CheckedAlignUp<std::uint64_t>(request_batch_max_bytes, 64);
  auto response_stride =
      CheckedAlignUp<std::uint64_t>(response_batch_max_bytes, 64);
  auto mailbox_cells = Product({world_size, mailbox_depth});
  auto request_staging_cells = Product({rpc_threads, request_send_depth});
  if (!request_stride.ok() || !response_stride.ok() || !mailbox_cells.ok() ||
      !request_staging_cells.ok()) {
    return Status(StatusCode::kOutOfRange,
                  "RPC memory cell/stride calculation overflows");
  }
  auto request_mailbox =
      CheckedMultiply(mailbox_cells.value(), request_stride.value());
  auto response_mailbox =
      CheckedMultiply(mailbox_cells.value(), response_stride.value());
  auto request_staging =
      CheckedMultiply(request_staging_cells.value(), request_stride.value());
  auto notify_cells = CheckedMultiply<std::uint64_t>(mailbox_cells.value(), 2U);
  auto notify = notify_cells.ok()
                    ? CheckedMultiply<std::uint64_t>(notify_cells.value(), 64)
                    : Result<std::uint64_t>(notify_cells.status());
  if (!request_mailbox.ok() || !response_mailbox.ok() ||
      !request_staging.ok() || !notify.ok()) {
    return Status(StatusCode::kOutOfRange,
                  "registered RPC memory calculation overflows");
  }
  auto request_segment =
      CheckedAdd(request_mailbox.value(), request_staging.value());
  auto response_segment =
      CheckedMultiply<std::uint64_t>(response_mailbox.value(), 2U);
  auto payload_segments =
      request_segment.ok() && response_segment.ok()
          ? CheckedAdd(request_segment.value(), response_segment.value())
          : Result<std::uint64_t>(request_segment.ok()
                                      ? response_segment.status()
                                      : request_segment.status());
  auto registered = payload_segments.ok()
                        ? CheckedAdd(payload_segments.value(), notify.value())
                        : Result<std::uint64_t>(payload_segments.status());
  if (!registered.ok())
    return registered.status();
  RpcMemoryPlan plan;
  plan.request_staging_slots_per_thread = request_send_depth;
  plan.request_stride = request_stride.value();
  plan.response_stride = response_stride.value();
  plan.request_mailbox_bytes = request_mailbox.value();
  plan.response_mailbox_bytes = response_mailbox.value();
  plan.result_ring_bytes = response_mailbox.value();
  plan.request_staging_bytes = request_staging.value();
  plan.notify_bytes = notify.value();
  plan.request_segment_bytes = request_segment.value();
  plan.response_segment_bytes = response_segment.value();
  plan.notify_segment_bytes = notify.value();
  plan.registered_rpc_bytes = registered.value();
  return plan;
}

Result<BenchmarkCommand>
ParseBenchmarkCommand(std::span<const std::string_view> arguments) {
  BenchmarkCommand command;
  auto &config = command.config;
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const auto option = arguments[index];
    if (option == "-h" || option == "--help") {
      command.action = BenchmarkAction::kHelp;
      return command;
    }
    if (option == "--version") {
      command.action = BenchmarkAction::kVersion;
      return command;
    }
    if (option == "--dry-run") {
      config.dry_run = true;
      continue;
    }
    if (option == "--profile") {
      config.profile = true;
      continue;
    }
    if (option == "--use-dot-norm") {
      config.use_dot_norm = true;
      continue;
    }
    if (index + 1U >= arguments.size()) {
      return Status(StatusCode::kInvalidArgument,
                    "missing value for " + std::string(option));
    }
    const auto value = arguments[++index];
    Status status;
    if (option == "--index") {
      config.index = value;
    } else if (option == "--queries") {
      config.queries = value;
    } else if (option == "--ground-truth") {
      config.ground_truth = value;
    } else if (option == "--query-offset") {
      auto parsed = ParseU64(value, option);
      if (!parsed.ok()) {
        status = parsed.status();
      } else {
        config.query_offset = parsed.value();
      }
    } else if (option == "--query-count") {
      auto parsed = ParseU64(value, option);
      if (!parsed.ok()) {
        status = parsed.status();
      } else {
        config.query_count = parsed.value();
      }
    } else if (option == "--topk") {
      status = AssignU32(value, option, config.topk);
    } else if (option == "--ef-search") {
      status = AssignAutoU32(value, option, config.ef_search);
    } else if (option == "--expand-batch") {
      status = AssignU32(value, option, config.expand_batch);
    } else if (option == "--distance-policy") {
      status = ParseEnum(value, option,
                         {{"auto", DistancePolicy::kAuto},
                          {"pull", DistancePolicy::kPull},
                          {"pushdown", DistancePolicy::kPushdown}},
                         config.distance_policy);
    } else if (option == "--pull-threshold") {
      status = AssignU32(value, option, config.pull_threshold);
    } else if (option == "--layer0-sort") {
      status = ParseEnum(value, option,
                         {{"all", Layer0SortMode::kAll},
                          {"local-only", Layer0SortMode::kLocalOnly},
                          {"upper-only", Layer0SortMode::kUpperOnly}},
                         config.layer0_sort_mode);
    } else if (option == "--search-workers") {
      status = AssignAutoU32(value, option, config.search_workers);
    } else if (option == "--distance-workers") {
      status = AssignAutoU32(value, option, config.distance_workers);
    } else if (option == "--read-progress-threads") {
      status = AssignAutoU32(value, option, config.read_progress_threads);
    } else if (option == "--rpc-progress-threads") {
      status = AssignAutoU32(value, option, config.rpc_progress_threads);
    } else if (option == "--max-active-queries-per-rank") {
      status = AssignU32(value, option, config.max_active_queries);
    } else if (option == "--visited-capacity") {
      status = AssignAutoU32(value, option, config.visited_capacity);
    } else if (option == "--cpu-bind") {
      status = ParseEnum(
          value, option,
          {{"auto", CpuBindMode::kAuto}, {"none", CpuBindMode::kNone}},
          config.cpu_bind);
    } else if (option == "--cpu-layout") {
      status = ParseEnum(value, option,
                         {{"auto", CpuLayout::kAuto},
                          {"contiguous", CpuLayout::kContiguous},
                          {"llc-isolated", CpuLayout::kLlcIsolated},
                          {"llc-spread", CpuLayout::kLlcSpread}},
                         config.cpu_layout);
    } else if (option == "--read-progress-mode") {
      status = ParseEnum(value, option,
                         {{"busy", ProgressMode::kBusy},
                          {"adaptive", ProgressMode::kAdaptive}},
                         config.read_progress_mode);
    } else if (option == "--rpc-progress-mode") {
      status = ParseEnum(value, option,
                         {{"busy", ProgressMode::kBusy},
                          {"adaptive", ProgressMode::kAdaptive}},
                         config.rpc_progress_mode);
    } else if (option == "--read-progress-spin-us") {
      status = AssignAutoU32(value, option, config.read_progress_spin_us);
    } else if (option == "--rpc-progress-spin-us") {
      status = AssignAutoU32(value, option, config.rpc_progress_spin_us);
    } else if (option == "--read-result-slots-per-search") {
      status =
          AssignAutoU32(value, option, config.read_result_slots_per_search);
    } else if (option == "--read-submit-depth") {
      status = AssignAutoU32(value, option, config.read_submit_depth);
    } else if (option == "--rpc-operation-slots-per-search") {
      status =
          AssignAutoU32(value, option, config.rpc_operation_slots_per_search);
    } else if (option == "--rpc-submit-depth") {
      status = AssignAutoU32(value, option, config.rpc_submit_depth);
    } else if (option == "--rpc-request-batch-max-tasks") {
      status = AssignAutoU32(value, option, config.rpc_request_batch_max_tasks);
    } else if (option == "--rpc-request-batch-max-bytes") {
      status = AssignAutoU32(value, option, config.rpc_request_batch_max_bytes);
    } else if (option == "--rpc-response-batch-max-bytes") {
      status =
          AssignAutoU32(value, option, config.rpc_response_batch_max_bytes);
    } else if (option == "--rpc-response-progress-max-bytes") {
      status =
          AssignAutoU32(value, option, config.rpc_response_progress_max_bytes);
    } else if (option == "--urma-device") {
      config.urma_device = value;
    } else if (option == "--urma-device-map") {
      status = AssignUrmaDeviceMap(value, config.urma_device_map);
    } else if (option == "--eid-index") {
      status = AssignAutoU32(value, option, config.eid_index);
    } else if (option == "--tp-type") {
      status = ParseEnum(value, option,
                         {{"ctp", TpType::kCtp}, {"rtp", TpType::kRtp}},
                         config.tp_type);
    } else if (option == "--intra-node-transport") {
      status = ParseEnum(value, option,
                         {{"urma", IntraNodeTransport::kUrma},
                          {"shm", IntraNodeTransport::kSharedMemory}},
                         config.intra_node_transport);
    } else if (option == "--read-jfs-depth") {
      status = AssignAutoU32(value, option, config.read_jfs_depth);
    } else if (option == "--request-jfs-depth") {
      status = AssignAutoU32(value, option, config.request_jfs_depth);
    } else if (option == "--response-jfs-depth") {
      status = AssignAutoU32(value, option, config.response_jfs_depth);
    } else if (option == "--request-jfs-per-rpc-lane") {
      status = AssignU32(value, option, config.request_jfs_per_rpc_lane);
    } else if (option == "--response-jfs-per-rpc-lane") {
      status = AssignU32(value, option, config.response_jfs_per_rpc_lane);
    } else if (option == "--read-jfc-depth") {
      status = AssignAutoU32(value, option, config.read_jfc_depth);
    } else if (option == "--request-jfc-depth") {
      status = AssignAutoU32(value, option, config.request_jfc_depth);
    } else if (option == "--response-jfc-depth") {
      status = AssignAutoU32(value, option, config.response_jfc_depth);
    } else if (option == "--wr-list-max") {
      status = AssignU32(value, option, config.wr_list_max);
    } else if (option == "--request-ring-depth") {
      status = AssignU32(value, option, config.request_ring_depth);
    } else if (option == "--queue-memory-budget-bytes") {
      status = AssignU64(value, option, config.queue_memory_budget_bytes);
    } else if (option == "--registered-rpc-memory-budget-bytes") {
      status =
          AssignU64(value, option, config.registered_rpc_memory_budget_bytes);
    } else if (option == "--operation-timeout-ms") {
      status = AssignU32(value, option, config.operation_timeout_ms);
    } else if (option == "--query-timeout-ms") {
      status = AssignU32(value, option, config.query_timeout_ms);
    } else if (option == "--drain-timeout-ms") {
      status = AssignU32(value, option, config.drain_timeout_ms);
    } else if (option == "--warmup") {
      status = AssignU32(value, option, config.warmup, true);
    } else if (option == "--rounds") {
      status = AssignU32(value, option, config.rounds);
    } else if (option == "--profile-output") {
      config.profile_output = value;
    } else if (option == "--output") {
      config.output = value;
    } else if (option == "--output-format") {
      status = ParseEnum(value, option,
                         {{"text", OutputFormat::kText},
                          {"json", OutputFormat::kJson},
                          {"csv", OutputFormat::kCsv}},
                         config.output_format);
    } else {
      return Status(StatusCode::kInvalidArgument,
                    "unknown option: " + std::string(option));
    }
    if (!status.ok()) {
      return status;
    }
  }
  if (config.index.empty() || (!config.dry_run && config.queries.empty())) {
    return Status(StatusCode::kInvalidArgument,
                  config.dry_run ? "--index is required"
                                 : "--index and --queries are required");
  }
  if (config.query_timeout_ms < config.operation_timeout_ms) {
    return Status(StatusCode::kInvalidArgument,
                  "query timeout must not be smaller than operation timeout");
  }
  if (config.expand_batch > 64) {
    return Status(StatusCode::kOutOfRange, "--expand-batch must be in [1,64]");
  }
  if (!IsPowerOfTwo(config.request_ring_depth)) {
    return Status(StatusCode::kInvalidArgument,
                  "--request-ring-depth must be a power of two");
  }
  if (!config.visited_capacity.automatic &&
      !IsPowerOfTwo(config.visited_capacity.value)) {
    return Status(StatusCode::kInvalidArgument,
                  "--visited-capacity must be a power of two");
  }
  if (!config.urma_device.empty() && !config.urma_device_map.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "--urma-device and --urma-device-map are mutually exclusive");
  }
  return command;
}

Result<RpcOperationPlan> BuildRpcOperationPlan(
    std::uint32_t queries, std::uint32_t searches, std::uint32_t rpc_threads,
    std::uint32_t maximum_batch, std::uint32_t local_rank,
    std::span<const std::uint32_t> shard_to_rank,
    std::span<const std::uint32_t> rpc_owner_by_rank,
    std::uint32_t request_ring_depth, AutoU32 slots, AutoU32 submit_depth) {
  if (queries == 0 || searches == 0 || rpc_threads == 0 || maximum_batch == 0 ||
      local_rank >= rpc_owner_by_rank.size() || shard_to_rank.empty())
    return Status(StatusCode::kInvalidArgument,
                  "invalid RPC capacity dimensions");
  std::vector<std::uint32_t> shards_per_owner(rpc_threads, 0);
  std::uint64_t remote_shards = 0;
  for (auto rank : shard_to_rank) {
    if (rank >= rpc_owner_by_rank.size())
      return Status(StatusCode::kInvalidArgument,
                    "invalid RPC capacity shard rank");
    if (rank == local_rank)
      continue;
    if (rpc_owner_by_rank[rank] >= rpc_threads)
      return Status(StatusCode::kInvalidArgument, "invalid RPC capacity owner");
    ++shards_per_owner[rpc_owner_by_rank[rank]];
    ++remote_shards;
  }
  const auto fanout = std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(maximum_batch, remote_shards));
  const auto per_search =
      (static_cast<std::uint64_t>(queries) + searches - 1U) / searches;
  const auto automatic = per_search * fanout;
  if (automatic > UINT32_MAX)
    return Status(StatusCode::kOutOfRange, "RPC automatic pool exceeds uint32");
  RpcOperationPlan plan;
  plan.slots_per_search = Resolve(slots, static_cast<std::uint32_t>(automatic));
  if (plan.slots_per_search < fanout)
    return Status(StatusCode::kInvalidArgument,
                  "RPC pool cannot hold one complete stage");
  auto total = CheckedMultiply(searches, plan.slots_per_search);
  auto capacity = CheckedAdd(queries, plan.slots_per_search);
  if (!total.ok() || !capacity.ok())
    return Status(StatusCode::kOutOfRange, "RPC pool/event capacity overflows");
  plan.total_slots = total.value();
  plan.search_event_capacity = std::max(2U, capacity.value());
  plan.minimum_submit_depth = 1;
  for (auto count : shards_per_owner)
    plan.minimum_submit_depth =
        std::max(plan.minimum_submit_depth, std::min(maximum_batch, count));
  plan.submit_depth = Resolve(submit_depth, plan.slots_per_search);
  if (plan.submit_depth < plan.minimum_submit_depth)
    return Status(StatusCode::kInvalidArgument,
                  "RPC Submit cell cannot hold one complete stage");
  auto pool = RpcDistanceOracle::RequiredBytes(
      searches, rpc_threads, plan.slots_per_search, maximum_batch, queries,
      static_cast<std::uint32_t>(shard_to_rank.size()),
      static_cast<std::uint32_t>(rpc_owner_by_rank.size()));
  if (!pool.ok())
    return pool.status();
  plan.pool_bytes = pool.value();
  auto client = RpcClientOwner::RequiredBytes(
      static_cast<std::uint32_t>(rpc_owner_by_rank.size()), plan.total_slots,
      queries, request_ring_depth);
  if (!client.ok())
    return client.status();
  auto clients = CheckedMultiply<std::uint64_t>(rpc_threads, client.value());
  if (!clients.ok())
    return clients.status();
  plan.client_owner_bytes = clients.value();
  return plan;
}

Result<EffectiveRuntimeConfig> ResolveRuntimeConfig(
    const RuntimeConfig &config, const GlobalIndexMeta &meta,
    std::uint32_t world_size, std::span<const std::uint32_t> allowed_cpus,
    std::uint32_t local_rank, std::span<const std::uint32_t> numa_nodes) {
  if (world_size == 0 || local_rank >= world_size ||
      world_size != meta.num_shards || allowed_cpus.empty() ||
      config.max_active_queries == 0) {
    return Status(
        StatusCode::kInvalidArgument,
        "world/local rank, shard count, and allowed CPU set are inconsistent");
  }
  if (!config.urma_device.empty() && !config.urma_device_map.empty()) {
    return Status(StatusCode::kInvalidArgument,
                  "--urma-device and --urma-device-map are mutually exclusive");
  }
  EffectiveRuntimeConfig effective;
  effective.allowed_cpus.assign(allowed_cpus.begin(), allowed_cpus.end());
  std::sort(effective.allowed_cpus.begin(), effective.allowed_cpus.end());
  if (std::adjacent_find(effective.allowed_cpus.begin(),
                         effective.allowed_cpus.end()) !=
      effective.allowed_cpus.end()) {
    return Status(StatusCode::kInvalidArgument,
                  "allowed CPU set contains duplicates");
  }
  effective.urma_device = config.urma_device;
  if (!config.urma_device_map.empty()) {
    std::vector<std::uint32_t> discovered_nodes;
    if (numa_nodes.empty()) {
      auto discovered = DiscoverNumaNodes(effective.allowed_cpus);
      if (!discovered.ok()) {
        auto status = discovered.status();
        status.AddContext("option", "--urma-device-map");
        return status;
      }
      discovered_nodes = std::move(discovered).value();
      numa_nodes = discovered_nodes;
    }
    if (numa_nodes.size() != 1U) {
      return Status(StatusCode::kInvalidArgument,
                    "--urma-device-map requires the process affinity to cover "
                    "exactly one NUMA node")
          .AddContext("numa_nodes", NumaNodesText(numa_nodes));
    }
    effective.numa_node = numa_nodes.front();
    const auto device =
        config.urma_device_map.find(effective.numa_node.value());
    if (device == config.urma_device_map.end()) {
      return Status(StatusCode::kInvalidArgument,
                    "--urma-device-map has no entry for the process NUMA node")
          .AddContext("numa_node", std::to_string(effective.numa_node.value()))
          .AddContext("urma_device_map",
                      UrmaDeviceMapText(config.urma_device_map));
    }
    effective.urma_device = device->second;
  }
  if (!effective.urma_device.empty()) {
    effective.urma_devices.push_back(effective.urma_device);
  }
  for (const auto &[node, device] : config.urma_device_map) {
    (void)node;
    if (std::find(effective.urma_devices.begin(), effective.urma_devices.end(),
                  device) == effective.urma_devices.end()) {
      effective.urma_devices.push_back(device);
    }
  }
  const auto cpu_count =
      static_cast<std::uint32_t>(effective.allowed_cpus.size());
  const std::uint32_t automatic_distance = std::max(1U, cpu_count / 6U);
  const std::uint32_t reserved = std::max(1U, cpu_count / 12U);
  effective.distance_workers =
      Resolve(config.distance_workers, automatic_distance);
  if (world_size > 1U) {
    auto remote_slots = CheckedMultiply<std::uint64_t>(
        world_size - 1U, config.request_ring_depth);
    if (!remote_slots.ok() ||
        effective.distance_workers > remote_slots.value()) {
      return Status(
                 StatusCode::kInvalidArgument,
                 "distance workers exceed the number of remote request slots")
          .AddContext("distance_workers",
                      std::to_string(effective.distance_workers))
          .AddContext("remote_request_slots",
                      remote_slots.ok() ? std::to_string(remote_slots.value())
                                        : "overflow");
    }
  }
  const std::uint32_t automatic_read =
      std::max(1U, std::min(2U, cpu_count / 24U));
  const std::uint32_t automatic_rpc =
      std::max(1U, std::min(4U, cpu_count / 12U));
  effective.read_progress_threads =
      Resolve(config.read_progress_threads, automatic_read);
  effective.rpc_progress_threads =
      Resolve(config.rpc_progress_threads, automatic_rpc);
  auto communication_total = CheckedAdd(effective.read_progress_threads,
                                        effective.rpc_progress_threads);
  if (!communication_total.ok()) {
    return communication_total.status();
  }
  const std::uint32_t communication_threads = communication_total.value();
  const auto non_search = static_cast<std::uint64_t>(communication_threads) +
                          effective.distance_workers + reserved;
  if (non_search >= cpu_count) {
    return Status(StatusCode::kResourceExhausted,
                  "allowed CPU set is too small for requested runtime roles")
        .AddContext("distance_workers",
                    std::to_string(effective.distance_workers))
        .AddContext("communication_threads",
                    std::to_string(communication_threads))
        .AddContext("control_reserve", std::to_string(reserved))
        .AddContext("allowed_cpus", std::to_string(cpu_count));
  }
  const auto automatic_search =
      cpu_count - static_cast<std::uint32_t>(non_search);
  effective.search_workers = Resolve(config.search_workers, automatic_search);
  const std::uint64_t data_threads =
      static_cast<std::uint64_t>(communication_threads) +
      effective.distance_workers + effective.search_workers;
  if (data_threads > cpu_count - 1U) {
    return Status(StatusCode::kResourceExhausted,
                  "search+distance+communication threads exceed allowed CPUs "
                  "minus control reserve")
        .AddContext("search_workers", std::to_string(effective.search_workers))
        .AddContext("distance_workers",
                    std::to_string(effective.distance_workers))
        .AddContext("read_progress_threads",
                    std::to_string(effective.read_progress_threads))
        .AddContext("rpc_progress_threads",
                    std::to_string(effective.rpc_progress_threads))
        .AddContext("allowed_cpus", std::to_string(cpu_count));
  }

  effective.ef_search = Resolve(config.ef_search, meta.ef_search_default);
  if (effective.ef_search < config.topk) {
    return Status(StatusCode::kInvalidArgument,
                  "topk must not exceed ef-search");
  }
  auto visited_work = Product({static_cast<std::uint64_t>(effective.ef_search),
                               static_cast<std::uint64_t>(meta.M0), 2U});
  if (!visited_work.ok()) {
    return visited_work.status();
  }
  auto automatic_visited =
      NextPowerOfTwo(std::max<std::uint64_t>(1024, visited_work.value()));
  if (!automatic_visited.ok()) {
    return automatic_visited.status();
  }
  effective.visited_capacity =
      Resolve(config.visited_capacity, automatic_visited.value());
  auto active_world =
      CheckedMultiply<std::uint64_t>(config.max_active_queries, world_size);
  auto distance_four =
      CheckedMultiply<std::uint64_t>(effective.distance_workers, 4U);
  if (!active_world.ok() || !distance_four.ok() ||
      active_world.value() > std::numeric_limits<std::uint32_t>::max() ||
      distance_four.value() > std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "automatic response JFS depth overflows uint32");
  }
  const std::uint32_t automatic_response_jfs =
      std::max(static_cast<std::uint32_t>(distance_four.value()),
               std::max(effective.distance_workers,
                        static_cast<std::uint32_t>(active_world.value())));
  if (!config.eid_index.automatic) {
    effective.eid_index = config.eid_index.value;
  }

  auto layer0_batch =
      CheckedMultiply<std::uint64_t>(meta.M0, config.expand_batch);
  auto active_graph_reads = CheckedMultiply<std::uint64_t>(
      config.max_active_queries, config.expand_batch);
  auto automatic_read_floor =
      active_graph_reads.ok()
          ? CheckedMultiply<std::uint64_t>(active_graph_reads.value(), 2U)
          : Result<std::uint64_t>(active_graph_reads.status());
  if (!layer0_batch.ok() || !automatic_read_floor.ok() ||
      layer0_batch.value() > std::numeric_limits<std::uint32_t>::max() ||
      automatic_read_floor.value() >
          std::numeric_limits<std::uint32_t>::max()) {
    return Status(StatusCode::kOutOfRange,
                  "maximum READ batch exceeds supported bounds");
  }
  const auto maximum_batch =
      std::max<std::uint64_t>({1U, meta.M, layer0_batch.value()});
  effective.maximum_distance_batch = static_cast<std::uint32_t>(maximum_batch);

  auto &urma = effective.urma;
  urma.communication_lanes = communication_threads;
  urma.read_progress_threads = effective.read_progress_threads;
  urma.rpc_progress_threads = effective.rpc_progress_threads;
  urma.read_jfs_depth = Resolve(
      config.read_jfs_depth,
      std::max({64U, static_cast<std::uint32_t>(maximum_batch),
                static_cast<std::uint32_t>(automatic_read_floor.value())}));
  // A full wait stage includes all Pull groups, entry and upper-layer reads.
  const auto stage_slots =
      std::max<std::uint64_t>({1U, config.expand_batch, maximum_batch});
  const auto per_search_queries = std::max<std::uint64_t>(
      1, (static_cast<std::uint64_t>(config.max_active_queries) +
          effective.search_workers - 1) /
             effective.search_workers);
  auto automatic_slots =
      CheckedMultiply<std::uint64_t>(stage_slots, per_search_queries);
  auto graph_records = CheckedMultiply<std::uint64_t>(per_search_queries, 2U);
  auto pull_records =
      graph_records.ok()
          ? CheckedMultiply<std::uint64_t>(graph_records.value(), world_size)
          : Result<std::uint64_t>(graph_records.status());
  if (!automatic_slots.ok() || !graph_records.ok() || !pull_records.ok() ||
      automatic_slots.value() > std::numeric_limits<std::uint32_t>::max() ||
      graph_records.value() + pull_records.value() >
          std::numeric_limits<std::uint32_t>::max())
    return Status(StatusCode::kOutOfRange,
                  "Search READ capacity overflows uint32");
  effective.read_result_slots_per_search =
      Resolve(config.read_result_slots_per_search,
              static_cast<std::uint32_t>(automatic_slots.value()));
  effective.read_graph_records_per_search =
      static_cast<std::uint32_t>(graph_records.value());
  effective.read_pull_records_per_search =
      static_cast<std::uint32_t>(pull_records.value());
  if (effective.read_result_slots_per_search < stage_slots)
    return Status(StatusCode::kInvalidArgument,
                  "Search READ result pool cannot hold a complete wait stage")
        .AddContext("minimum_slots", std::to_string(stage_slots));
  auto result_bytes = SearchReadResources::RequiredBytes(
      effective.read_result_slots_per_search,
      std::max<std::size_t>(meta.M0 * sizeof(node_id_t),
                            meta.dim * sizeof(float)));
  auto total_result_bytes =
      result_bytes.ok() ? CheckedMultiply<std::uint64_t>(
                              result_bytes.value(), effective.search_workers)
                        : Result<std::uint64_t>(result_bytes.status());
  if (!total_result_bytes.ok())
    return total_result_bytes.status();
  auto page_size = PageAlignedBuffer::SystemPageSize();
  if (!page_size.ok())
    return page_size.status();
  auto registered_result_bytes = CheckedAlignUp<std::uint64_t>(
      total_result_bytes.value(), page_size.value());
  if (!registered_result_bytes.ok())
    return registered_result_bytes.status();
  effective.registered_read_result_bytes = registered_result_bytes.value();
  urma.request_jfs_depth = Resolve(
      config.request_jfs_depth,
      std::max(8U, config.request_ring_depth * std::max(1U, world_size - 1U)));
  urma.response_jfs_depth =
      Resolve(config.response_jfs_depth, std::max(8U, automatic_response_jfs));
  urma.request_jfs_per_rpc_lane = config.request_jfs_per_rpc_lane;
  urma.response_jfs_per_rpc_lane = config.response_jfs_per_rpc_lane;
  if (urma.request_jfs_per_rpc_lane == 0 ||
      urma.response_jfs_per_rpc_lane == 0) {
    return Status(StatusCode::kInvalidArgument,
                  "JFS count per RPC lane must be non-zero");
  }
  auto total_request_jfs_depth = CheckedMultiply<std::uint32_t>(
      urma.request_jfs_depth, urma.request_jfs_per_rpc_lane);
  auto total_response_jfs_depth = CheckedMultiply<std::uint32_t>(
      urma.response_jfs_depth, urma.response_jfs_per_rpc_lane);
  if (!total_request_jfs_depth.ok() || !total_response_jfs_depth.ok()) {
    return total_request_jfs_depth.ok() ? total_response_jfs_depth.status()
                                        : total_request_jfs_depth.status();
  }
  auto request_jfc_depth =
      PerOperationCompletionDepth(total_request_jfs_depth.value());
  auto response_jfc_depth =
      PerOperationCompletionDepth(total_response_jfs_depth.value());
  if (!request_jfc_depth.ok() || !response_jfc_depth.ok()) {
    return request_jfc_depth.ok() ? response_jfc_depth.status()
                                  : request_jfc_depth.status();
  }
  urma.read_jfc_depth = Resolve(config.read_jfc_depth, urma.read_jfs_depth);
  urma.request_jfc_depth =
      Resolve(config.request_jfc_depth, request_jfc_depth.value());
  urma.response_jfc_depth =
      Resolve(config.response_jfc_depth, response_jfc_depth.value());
  urma.wr_list_max = config.wr_list_max;

  {
    auto owners = BuildCommunicationOwnerMap(world_size, local_rank,
                                             effective.read_progress_threads,
                                             effective.rpc_progress_threads);
    if (!owners.ok()) {
      return owners.status();
    }
    effective.owners = std::move(owners).value();
    std::size_t maximum_remote_peers = 1;
    for (const auto &owned : effective.owners.read_owned_ranks) {
      maximum_remote_peers = std::max(maximum_remote_peers, owned.size());
    }
    auto automatic_read_batch = CheckedMultiply<std::uint64_t>(
        per_search_queries,
        std::min<std::uint64_t>(maximum_remote_peers, stage_slots));
    if (!automatic_read_batch.ok() || automatic_read_batch.value() > UINT32_MAX)
      return Status(StatusCode::kOutOfRange,
                    "automatic READ submit depth exceeds uint32");
    // Current index loading maps one shard to each rank. The plan itself uses
    // logical shard counts and also supports multiple shards at one peer.
    std::vector<std::uint32_t> shard_ranks(meta.num_shards);
    for (std::uint32_t shard = 0; shard < meta.num_shards; ++shard)
      shard_ranks[shard] = shard;
    auto rpc_plan = BuildRpcOperationPlan(
        config.max_active_queries, effective.search_workers,
        effective.rpc_progress_threads, effective.maximum_distance_batch,
        local_rank, shard_ranks, effective.owners.rpc_owner_by_rank,
        config.request_ring_depth, config.rpc_operation_slots_per_search,
        config.rpc_submit_depth);
    if (!rpc_plan.ok())
      return rpc_plan.status();
    effective.rpc_operations = rpc_plan.value();
    RuntimeQueueDepths depths;
    depths.admission = config.max_active_queries;
    depths.ready = effective.rpc_operations.search_event_capacity;
    depths.read_submit =
        Resolve(config.read_submit_depth,
                static_cast<std::uint32_t>(automatic_read_batch.value()));
    if (depths.read_submit <
        std::min<std::uint64_t>(maximum_remote_peers, stage_slots))
      return Status(StatusCode::kInvalidArgument,
                    "READ Submit cell cannot hold a complete wait stage");
    depths.read_reclaim = effective.read_graph_records_per_search +
                          effective.read_pull_records_per_search;
    depths.rpc_submit = effective.rpc_operations.submit_depth;
    depths.done = config.max_active_queries;
    auto queue_plan = BuildRuntimeQueuePlan(
        effective.search_workers, effective.read_progress_threads,
        effective.rpc_progress_threads, depths, config.profile, world_size);
    if (!queue_plan.ok()) {
      return queue_plan.status();
    }
    effective.queues = queue_plan.value();
    if (effective.queues.queue_bytes_total > config.queue_memory_budget_bytes) {
      return Status(StatusCode::kResourceExhausted,
                    "queue memory plan exceeds the configured startup budget")
          .AddContext("required_bytes",
                      std::to_string(effective.queues.queue_bytes_total))
          .AddContext("budget_bytes",
                      std::to_string(config.queue_memory_budget_bytes));
    }

    effective.read_progress_spin_us = Resolve(config.read_progress_spin_us, 50);
    effective.rpc_progress_spin_us = Resolve(config.rpc_progress_spin_us, 50);
    effective.rpc_request_batch_max_tasks =
        Resolve(config.rpc_request_batch_max_tasks,
                std::numeric_limits<std::uint32_t>::max());
    constexpr std::uint32_t kAutomaticBatchBytes = 64U << 10U;
    auto minimum_request_bytes =
        DistanceBatchRequestBytesForShape(1, maximum_batch, meta.dim);
    auto minimum_response_bytes =
        DistanceBatchResponseBytesForShape(1, maximum_batch);
    if (!minimum_request_bytes.ok() || !minimum_response_bytes.ok() ||
        minimum_request_bytes.value() >
            std::numeric_limits<std::uint32_t>::max() ||
        minimum_response_bytes.value() >
            std::numeric_limits<std::uint32_t>::max()) {
      if (!minimum_request_bytes.ok()) {
        return minimum_request_bytes.status();
      }
      return minimum_response_bytes.status();
    }
    const auto automatic_request_bytes =
        std::max(kAutomaticBatchBytes,
                 static_cast<std::uint32_t>(minimum_request_bytes.value()));
    const auto automatic_response_bytes =
        std::max(kAutomaticBatchBytes,
                 static_cast<std::uint32_t>(minimum_response_bytes.value()));
    effective.rpc_request_batch_max_bytes =
        Resolve(config.rpc_request_batch_max_bytes, automatic_request_bytes);
    effective.rpc_response_batch_max_bytes =
        Resolve(config.rpc_response_batch_max_bytes, automatic_response_bytes);
    effective.rpc_response_progress_max_bytes =
        Resolve(config.rpc_response_progress_max_bytes,
                effective.rpc_response_batch_max_bytes);
    if (effective.rpc_request_batch_max_bytes < minimum_request_bytes.value() ||
        effective.rpc_response_batch_max_bytes <
            minimum_response_bytes.value()) {
      return Status(StatusCode::kInvalidArgument,
                    "batch byte limit cannot hold one maximum-size task")
          .AddContext("minimum_request_bytes",
                      std::to_string(minimum_request_bytes.value()))
          .AddContext("minimum_response_bytes",
                      std::to_string(minimum_response_bytes.value()));
    }
  }
  {
    auto rpc_plan = BuildRpcMemoryPlan(
        world_size, effective.rpc_progress_threads, config.request_ring_depth,
        total_request_jfs_depth.value(), effective.rpc_request_batch_max_bytes,
        effective.rpc_response_batch_max_bytes);
    if (!rpc_plan.ok()) {
      return rpc_plan.status();
    }
    effective.rpc_memory = rpc_plan.value();
    if (effective.rpc_memory.registered_rpc_bytes >
        config.registered_rpc_memory_budget_bytes) {
      return Status(StatusCode::kResourceExhausted,
                    "registered RPC memory plan exceeds the configured "
                    "startup budget")
          .AddContext("required_bytes",
                      std::to_string(effective.rpc_memory.registered_rpc_bytes))
          .AddContext(
              "budget_bytes",
              std::to_string(config.registered_rpc_memory_budget_bytes));
    }
  }

  const CpuRoleCounts role_counts{
      effective.search_workers,
      effective.distance_workers,
      effective.read_progress_threads,
      effective.rpc_progress_threads,
      cpu_count - static_cast<std::uint32_t>(data_threads),
  };
  Result<CpuRoleLayout> cpu_layout =
      Status(StatusCode::kInternalError, "CPU layout was not planned");
  if (config.cpu_layout == CpuLayout::kContiguous) {
    cpu_layout = BuildContiguousCpuLayout(effective.allowed_cpus, role_counts);
    effective.cpu_layout = CpuLayout::kContiguous;
  } else {
    auto topology = DiscoverCpuTopology(effective.allowed_cpus);
    if (topology.ok()) {
      effective.cpu_topology = topology.value();
      std::set<std::uint32_t> llcs;
      for (const auto &entry : effective.cpu_topology) {
        llcs.insert(entry.llc_id);
      }
      if (llcs.size() > 1U) {
        cpu_layout =
            config.cpu_layout == CpuLayout::kLlcSpread
                ? BuildLlcSpreadCpuLayout(effective.allowed_cpus,
                                          effective.cpu_topology, role_counts)
                : BuildLlcIsolatedCpuLayout(effective.allowed_cpus,
                                            effective.cpu_topology,
                                            role_counts);
        if (cpu_layout.ok()) {
          effective.cpu_layout = config.cpu_layout == CpuLayout::kLlcSpread
                                     ? CpuLayout::kLlcSpread
                                     : CpuLayout::kLlcIsolated;
        }
      }
    }
    if (!cpu_layout.ok()) {
      if (config.cpu_layout != CpuLayout::kAuto) {
        auto status = topology.ok() ? cpu_layout.status() : topology.status();
        status.AddContext("cpu_layout",
                          std::string(ToString(config.cpu_layout)));
        return status;
      }
      cpu_layout =
          BuildContiguousCpuLayout(effective.allowed_cpus, role_counts);
      effective.cpu_layout = CpuLayout::kContiguous;
    }
  }
  if (!cpu_layout.ok()) {
    return cpu_layout.status();
  }
  effective.search_cpus = std::move(cpu_layout.value().search);
  effective.distance_cpus = std::move(cpu_layout.value().distance);
  effective.read_progress_cpus = std::move(cpu_layout.value().read_progress);
  effective.rpc_progress_cpus = std::move(cpu_layout.value().rpc_progress);
  effective.reserved_cpus = std::move(cpu_layout.value().reserved);
  return effective;
}

Result<CanonicalConfig>
BuildCanonicalRuntimeConfig(const RuntimeConfig &requested,
                            const EffectiveRuntimeConfig &effective) {
  CanonicalConfig canonical;
  const auto add = [&](std::string key, std::string value) -> Status {
    return Set(canonical, std::move(key), std::move(value));
  };
  for (const auto &[key, value] :
       std::initializer_list<std::pair<std::string, std::string>>{
           {"git_commit", HNSW_URMA_GIT_COMMIT},
           {"index", requested.index.string()},
           {"queries", requested.queries.string()},
           {"ground_truth", requested.ground_truth.string()},
           {"query_offset", std::to_string(requested.query_offset)},
           {"query_count", std::to_string(requested.query_count)},
           {"topk", std::to_string(requested.topk)},
           {"ef_search.requested", AutoText(requested.ef_search)},
           {"ef_search.effective", std::to_string(effective.ef_search)},
           {"expand_batch", std::to_string(requested.expand_batch)},
           {"distance_policy",
            std::string(ToString(requested.distance_policy))},
           {"pull_threshold", std::to_string(requested.pull_threshold)},
           {"use_dot_norm", requested.use_dot_norm ? "true" : "false"},
           {"layer0_sort", std::string(ToString(requested.layer0_sort_mode))},
           {"search_workers", std::to_string(effective.search_workers)},
           {"distance_workers", std::to_string(effective.distance_workers)},
           {"cpu_bind",
            requested.cpu_bind == CpuBindMode::kAuto ? "auto" : "none"},
           {"cpu_layout.requested",
            std::string(ToString(requested.cpu_layout))},
           {"cpu_layout.effective",
            std::string(ToString(effective.cpu_layout))},
           {"read_progress_threads",
            std::to_string(effective.read_progress_threads)},
           {"rpc_progress_threads",
            std::to_string(effective.rpc_progress_threads)},
           {"read_progress_mode",
            requested.read_progress_mode == ProgressMode::kBusy ? "busy"
                                                                : "adaptive"},
           {"rpc_progress_mode",
            requested.rpc_progress_mode == ProgressMode::kBusy ? "busy"
                                                               : "adaptive"},
           {"read_progress_spin_us",
            std::to_string(effective.read_progress_spin_us)},
           {"rpc_progress_spin_us",
            std::to_string(effective.rpc_progress_spin_us)},
           {"rpc_request_batch_max_tasks",
            std::to_string(effective.rpc_request_batch_max_tasks)},
           {"rpc_request_batch_max_bytes",
            std::to_string(effective.rpc_request_batch_max_bytes)},
           {"rpc_response_batch_max_bytes",
            std::to_string(effective.rpc_response_batch_max_bytes)},
           {"rpc_response_progress_max_bytes",
            std::to_string(effective.rpc_response_progress_max_bytes)},
           {"read_result_slots_per_search",
            std::to_string(effective.read_result_slots_per_search)},
           {"registered_read_result_bytes",
            std::to_string(effective.registered_read_result_bytes)},
           {"read_graph_records_per_search",
            std::to_string(effective.read_graph_records_per_search)},
           {"read_pull_records_per_search",
            std::to_string(effective.read_pull_records_per_search)},
           {"read_submit_depth",
            std::to_string(effective.queues.depths.read_submit)},
           {"rpc_operation_slots_per_search",
            std::to_string(effective.rpc_operations.slots_per_search)},
           {"rpc_operation_slots_total",
            std::to_string(effective.rpc_operations.total_slots)},
           {"rpc_operation_pool_bytes",
            std::to_string(effective.rpc_operations.pool_bytes)},
           {"rpc_client_owner_bytes",
            std::to_string(effective.rpc_operations.client_owner_bytes)},
           {"search_event_capacity",
            std::to_string(effective.rpc_operations.search_event_capacity)},
           {"rpc_submit_depth",
            std::to_string(effective.queues.depths.rpc_submit)},
           {"max_active_queries", std::to_string(requested.max_active_queries)},
           {"visited_capacity", std::to_string(effective.visited_capacity)},
           {"urma_device", requested.urma_device},
           {"urma_device_map", UrmaDeviceMapText(requested.urma_device_map)},
           {"tp_type", requested.tp_type == TpType::kCtp ? "ctp" : "rtp"},
           {"intra_node_transport",
            std::string(ToString(requested.intra_node_transport))},
           {"read_jfs_depth", std::to_string(effective.urma.read_jfs_depth)},
           {"request_jfs_depth",
            std::to_string(effective.urma.request_jfs_depth)},
           {"response_jfs_depth",
            std::to_string(effective.urma.response_jfs_depth)},
           {"request_jfs_per_rpc_lane",
            std::to_string(effective.urma.request_jfs_per_rpc_lane)},
           {"response_jfs_per_rpc_lane",
            std::to_string(effective.urma.response_jfs_per_rpc_lane)},
           {"read_jfc_depth", std::to_string(effective.urma.read_jfc_depth)},
           {"request_jfc_depth",
            std::to_string(effective.urma.request_jfc_depth)},
           {"response_jfc_depth",
            std::to_string(effective.urma.response_jfc_depth)},
           {"wr_list_max", std::to_string(requested.wr_list_max)},
           {"request_ring_depth", std::to_string(requested.request_ring_depth)},
           {"queue_memory_budget_bytes",
            std::to_string(requested.queue_memory_budget_bytes)},
           {"registered_rpc_memory_budget_bytes",
            std::to_string(requested.registered_rpc_memory_budget_bytes)},
           {"operation_timeout_ms",
            std::to_string(requested.operation_timeout_ms)},
           {"query_timeout_ms", std::to_string(requested.query_timeout_ms)},
           {"drain_timeout_ms", std::to_string(requested.drain_timeout_ms)},
           {"v3.spsc_ring_count",
            std::to_string(effective.queues.spsc_ring_count)},
           {"v3.mpsc_ring_count",
            std::to_string(effective.queues.mpsc_ring_count)},
           {"v3.private_work_bytes",
            std::to_string(effective.queues.private_work_bytes)},
           {"v3.queue_bytes_total",
            std::to_string(effective.queues.queue_bytes_total)},
           {"v3.ring_metadata_bytes",
            std::to_string(effective.queues.ring_metadata_bytes)},
           {"v3.ring_storage_bytes",
            std::to_string(effective.queues.ring_storage_bytes)},
           {"v3.request_mailbox_bytes",
            std::to_string(effective.rpc_memory.request_mailbox_bytes)},
           {"v3.response_mailbox_bytes",
            std::to_string(effective.rpc_memory.response_mailbox_bytes)},
           {"v3.result_ring_bytes",
            std::to_string(effective.rpc_memory.result_ring_bytes)},
           {"v3.request_staging_bytes",
            std::to_string(effective.rpc_memory.request_staging_bytes)},
           {"v3.rpc_notify_bytes",
            std::to_string(effective.rpc_memory.notify_bytes)},
           {"v3.request_segment_bytes",
            std::to_string(effective.rpc_memory.request_segment_bytes)},
           {"v3.response_segment_bytes",
            std::to_string(effective.rpc_memory.response_segment_bytes)},
           {"v3.notify_segment_bytes",
            std::to_string(effective.rpc_memory.notify_segment_bytes)},
           {"v3.registered_rpc_bytes",
            std::to_string(effective.rpc_memory.registered_rpc_bytes)},
           {"batch_wire_version", std::to_string(DISTANCE_BATCH_WIRE_VERSION)},
       }) {
    auto status = add(key, value);
    if (!status.ok()) {
      return status;
    }
  }
  return canonical;
}

std::string_view ToString(DistancePolicy value) noexcept {
  switch (value) {
  case DistancePolicy::kAuto:
    return "auto";
  case DistancePolicy::kPull:
    return "pull";
  case DistancePolicy::kPushdown:
    return "pushdown";
  }
  return "unknown";
}

std::string_view ToString(OutputFormat value) noexcept {
  switch (value) {
  case OutputFormat::kText:
    return "text";
  case OutputFormat::kJson:
    return "json";
  case OutputFormat::kCsv:
    return "csv";
  }
  return "unknown";
}

std::string_view ToString(CpuLayout value) noexcept {
  switch (value) {
  case CpuLayout::kAuto:
    return "auto";
  case CpuLayout::kContiguous:
    return "contiguous";
  case CpuLayout::kLlcIsolated:
    return "llc-isolated";
  case CpuLayout::kLlcSpread:
    return "llc-spread";
  }
  return "unknown";
}

std::string_view ToString(Layer0SortMode value) noexcept {
  switch (value) {
  case Layer0SortMode::kAll:
    return "all";
  case Layer0SortMode::kLocalOnly:
    return "local-only";
  case Layer0SortMode::kUpperOnly:
    return "upper-only";
  }
  return "unknown";
}

std::string_view ToString(IntraNodeTransport value) noexcept {
  switch (value) {
  case IntraNodeTransport::kUrma:
    return "urma";
  case IntraNodeTransport::kSharedMemory:
    return "shm";
  }
  return "unknown";
}

std::string BenchmarkVersion() {
  std::ostringstream output;
  output << "hnsw_urma " HNSW_URMA_PROJECT_VERSION "\n"
         << "git_commit=" << HNSW_URMA_GIT_COMMIT << '\n'
         << "index_format=" << INDEX_FORMAT_V2_VERSION << '\n'
         << "distance_batch_wire=" << DISTANCE_BATCH_WIRE_VERSION << '\n';
  return output.str();
}

std::string_view GitCommit() noexcept { return HNSW_URMA_GIT_COMMIT; }

std::string BenchmarkHelp() {
  return R"HELP(Usage: hnsw_urma_bench [options]

Input:
  --index PATH                    IndexFormatV2 root (required)
  --queries PATH                  Float-bin query vectors (required unless --dry-run)
  --ground-truth PATH             Optional bin ground truth; enables recall
  --query-offset N                Default: 0
  --query-count N                 Default: 0 (all remaining)

Search:
  --topk N                        Default: 10
  --ef-search auto|N              Default: index value
  --expand-batch N                Range: 1..64; default: 1
  --distance-policy auto|pull|pushdown
  --pull-threshold N              Default: 4
  --layer0-sort all|local-only|upper-only
                                  Default: all
Threads and admission:
  --search-workers auto|N
  --distance-workers auto|N
  --read-progress-threads auto|N
  --rpc-progress-threads auto|N
  --max-active-queries-per-rank N  Default: 16
  --visited-capacity auto|N       N must be a power of two
  --queue-memory-budget-bytes N   Default: 1073741824
  --registered-rpc-memory-budget-bytes N
                                  Default: 8589934592
  --cpu-bind auto|none
  --cpu-layout auto|contiguous|llc-isolated|llc-spread
                                  Default: auto; LLC-aware when discoverable
  --read-progress-mode busy|adaptive
  --rpc-progress-mode busy|adaptive
  --read-progress-spin-us auto|N
  --rpc-progress-spin-us auto|N

Queues and batching:
  --read-submit-depth auto|N (batches per Search/Progress SPSC cell)
  --read-result-slots-per-search auto|N
  --rpc-operation-slots-per-search auto|N
  --rpc-submit-depth auto|N
  --rpc-request-batch-max-tasks auto|N
  --rpc-request-batch-max-bytes auto|N
  --rpc-response-batch-max-bytes auto|N
  --rpc-response-progress-max-bytes auto|N
                                  Soft byte budget per progress iteration

URMA:
  --urma-device DEV               Required for multi-rank and UMDK dry-run
  --urma-device-map MAP           NUMA-to-device map, e.g. 0=udma0,1=udma0,
                                  2=udma1,3=udma1; mutually exclusive with
                                  --urma-device and requires one bound NUMA;
                                  every unique device is opened per rank
  --eid-index auto|N
  --tp-type ctp|rtp
  --intra-node-transport urma|shm
                                  Same-host peer data path; default: urma
  --read-jfs-depth auto|N
  --request-jfs-depth auto|N
  --response-jfs-depth auto|N
  --request-jfs-per-rpc-lane N   Default: 1
  --response-jfs-per-rpc-lane N  Default: 1
  --read-jfc-depth auto|N
  --request-jfc-depth auto|N
  --response-jfc-depth auto|N
  --wr-list-max N                 Default: 8
  --request-ring-depth N          Power of two; default: 8

Timeout, benchmark, and output:
  --operation-timeout-ms N        Default: 5000
  --query-timeout-ms N            Default: 30000
  --drain-timeout-ms N            Local drain deadline; default: 30000
  --warmup N                      Default: 2
  --rounds N                      Default: 5
  --use-dot-norm                  Experimental dot/norm L2 kernel
  --profile                       Enable detailed counters
  --profile-output PATH           Base path for timestamped JSON and CSV files
  --output PATH|-                 Default: -
  --output-format text|json|csv

Control:
  -h, --help
  --version
  --dry-run                       Validate config/index/device/resource plan
)HELP";
}

} // namespace hnsw_urma
