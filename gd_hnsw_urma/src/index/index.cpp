#include "hnsw_urma/index/format.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <new>
#include <string_view>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {
namespace {

Status ValidateCanonicalAdjacency(std::span<const node_id_t> neighbors,
                                  std::uint32_t width, std::uint64_t ntotal,
                                  std::string_view label) {
  if (width == 0 || neighbors.size() % width != 0) {
    return Status(StatusCode::kInvalidFormat,
                  std::string(label) + " adjacency shape is invalid");
  }
  for (std::size_t begin = 0; begin < neighbors.size(); begin += width) {
    bool padding = false;
    for (std::size_t index = begin; index < begin + width; ++index) {
      const auto neighbor = neighbors[index];
      if (neighbor == INVALID_NODE_ID) {
        padding = true;
      } else if (static_cast<std::uint64_t>(neighbor) >= ntotal) {
        return Status(StatusCode::kInvalidFormat,
                      std::string(label) + " neighbor ID is outside the index");
      } else if (padding) {
        return Status(StatusCode::kInvalidFormat,
                      std::string(label) +
                          " invalid padding is not a contiguous suffix");
      }
    }
  }
  return Status::Ok();
}

std::uint64_t UpperLookupHash(node_id_t id) noexcept {
  std::uint64_t value = id;
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

} // namespace

Status ValidateGlobalMeta(const GlobalIndexMeta &meta) noexcept {
  if (meta.ntotal == 0 || meta.ntotal > std::numeric_limits<node_id_t>::max()) {
    return Status(StatusCode::kInvalidFormat,
                  "index ntotal is outside node_id_t range");
  }
  if (meta.id_width != INDEX_FORMAT_V2_ID_WIDTH) {
    return Status(StatusCode::kInvalidFormat,
                  "IndexFormatV2 requires 32-bit node IDs");
  }
  if (meta.dim == 0 || meta.M == 0 || meta.M0 == 0 ||
      meta.ef_search_default == 0 || meta.num_shards == 0) {
    return Status(StatusCode::kInvalidFormat,
                  "index dimensions and graph parameters must be non-zero");
  }
  if (meta.entry_point == INVALID_NODE_ID || meta.entry_point >= meta.ntotal) {
    return Status(StatusCode::kInvalidFormat, "index entry point is invalid");
  }
  if (meta.max_level < 0 || (meta.metric != 0 && meta.metric != 1)) {
    return Status(StatusCode::kInvalidFormat,
                  "index level or metric is unsupported");
  }
  if (std::any_of(meta.reserved.begin(), meta.reserved.end(),
                  [](std::uint8_t value) { return value != 0; })) {
    return Status(StatusCode::kInvalidFormat,
                  "index metadata reserved bytes must be zero");
  }
  return Status::Ok();
}

Result<UpperGraphReplica>
UpperGraphReplica::Create(const GlobalIndexMeta &meta,
                          std::vector<UpperNodeMeta> nodes,
                          std::vector<node_id_t> neighbors) {
  try {
    UpperGraphReplica replica;
    replica.owned_nodes_ = std::move(nodes);
    replica.owned_neighbors_ = std::move(neighbors);
    auto initialized = replica.Initialize(meta, replica.owned_nodes_,
                                          replica.owned_neighbors_);
    if (!initialized.ok()) {
      return initialized;
    }
    return replica;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate upper graph replica");
  }
}

Status UpperGraphReplica::Initialize(const GlobalIndexMeta &meta,
                                     std::span<const UpperNodeMeta> nodes,
                                     std::span<const node_id_t> neighbors) {
  auto status = ValidateGlobalMeta(meta);
  if (!status.ok()) {
    return status;
  }
  std::uint64_t expected_offset = 0;
  node_id_t previous = INVALID_NODE_ID;
  bool found_entry_point = false;
  for (const auto &node : nodes) {
    if (node.global_id >= meta.ntotal || node.max_level == 0 ||
        node.max_level > static_cast<std::uint32_t>(meta.max_level) ||
        node.reserved16 != 0 || node.neighbor_offset != expected_offset ||
        (previous != INVALID_NODE_ID && node.global_id <= previous)) {
      return Status(StatusCode::kInvalidFormat,
                    "upper node metadata is invalid or not canonical")
          .AddContext("node", std::to_string(node.global_id));
    }
    auto slots = CheckedMultiply(static_cast<std::uint64_t>(node.max_level),
                                 static_cast<std::uint64_t>(meta.M));
    auto next = slots.ok() ? CheckedAdd(expected_offset, slots.value())
                           : Result<std::uint64_t>(slots.status());
    if (!next.ok()) {
      return next.status();
    }
    expected_offset = next.value();
    previous = node.global_id;
    if (node.global_id == meta.entry_point &&
        node.max_level == meta.max_level) {
      found_entry_point = true;
    }
  }
  if ((meta.max_level == 0 && !nodes.empty()) ||
      (meta.max_level > 0 && (nodes.empty() || !found_entry_point))) {
    return Status(
        StatusCode::kInvalidFormat,
        "upper graph does not match the global entry point/max level");
  }
  if (expected_offset != neighbors.size()) {
    return Status(StatusCode::kInvalidFormat,
                  "upper neighbor slot count mismatch");
  }
  auto adjacency =
      ValidateCanonicalAdjacency(neighbors, meta.M, meta.ntotal, "upper graph");
  if (!adjacency.ok()) {
    return adjacency;
  }

  if (nodes.size() > std::numeric_limits<std::uint32_t>::max() ||
      nodes.size() > (std::uint64_t{1} << 30U)) {
    return Status(StatusCode::kResourceExhausted,
                  "upper graph lookup exceeds supported size");
  }
  try {
    meta_ = meta;
    nodes_ = nodes;
    neighbors_ = neighbors;
    if (nodes.empty()) {
      lookup_.clear();
      return Status::Ok();
    }
    std::uint32_t lookup_size = 1;
    const auto minimum = static_cast<std::uint64_t>(nodes.size()) * 2U;
    while (lookup_size < minimum) {
      lookup_size <<= 1U;
    }
    lookup_.assign(lookup_size, std::numeric_limits<std::uint32_t>::max());
    const auto mask = lookup_.size() - 1U;
    for (std::uint32_t index = 0; index < nodes.size(); ++index) {
      auto bucket =
          static_cast<std::size_t>(UpperLookupHash(nodes[index].global_id)) &
          mask;
      while (lookup_[bucket] != std::numeric_limits<std::uint32_t>::max()) {
        bucket = (bucket + 1U) & mask;
      }
      lookup_[bucket] = index;
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate upper graph lookup");
  }
}

std::optional<std::uint32_t>
UpperGraphReplica::FindNode(node_id_t id) const noexcept {
  if (lookup_.empty()) {
    return std::nullopt;
  }
  const auto mask = lookup_.size() - 1U;
  auto bucket = static_cast<std::size_t>(UpperLookupHash(id)) & mask;
  for (std::size_t probe = 0; probe < lookup_.size(); ++probe) {
    const auto index = lookup_[bucket];
    if (index == std::numeric_limits<std::uint32_t>::max()) {
      return std::nullopt;
    }
    if (nodes_[index].global_id == id) {
      return index;
    }
    bucket = (bucket + 1U) & mask;
  }
  return std::nullopt;
}

Result<std::span<const node_id_t>>
UpperGraphReplica::Neighbors(node_id_t id, std::int32_t level) const noexcept {
  const auto found = FindNode(id);
  if (!found.has_value()) {
    return Status(StatusCode::kOutOfRange, "node has no upper graph record");
  }
  const auto &node = nodes_[*found];
  if (level <= 0 || level > node.max_level) {
    return Status(StatusCode::kOutOfRange, "requested upper level is absent");
  }
  const std::uint64_t offset =
      node.neighbor_offset + static_cast<std::uint64_t>(level - 1) * meta_.M;
  return neighbors_.subspan(static_cast<std::size_t>(offset), meta_.M);
}

Result<ShardEntryPoint>
UpperGraphReplica::EntryPointForShard(const ShardRange &range) const noexcept {
  const auto range_end = CheckedAdd(
      static_cast<std::uint64_t>(range.global_begin), range.local_count);
  if (range.local_count == 0 || range.shard_id >= meta_.num_shards ||
      !range_end.ok() || range_end.value() > meta_.ntotal) {
    return Status(StatusCode::kInvalidArgument,
                  "entry-point shard range is invalid");
  }

  ShardEntryPoint selected{range.global_begin, 0};
  auto node =
      std::lower_bound(nodes_.begin(), nodes_.end(), range.global_begin,
                       [](const UpperNodeMeta &candidate, node_id_t begin) {
                         return candidate.global_id < begin;
                       });
  for (; node != nodes_.end() &&
         static_cast<std::uint64_t>(node->global_id) < range_end.value();
       ++node) {
    if (static_cast<std::int32_t>(node->max_level) > selected.level) {
      selected = {node->global_id, static_cast<std::int32_t>(node->max_level)};
    }
  }
  return selected;
}

std::uint64_t UpperGraphReplica::persistent_bytes() const noexcept {
  return nodes_.size() * sizeof(UpperNodeMeta) +
         neighbors_.size() * sizeof(node_id_t);
}

std::uint64_t UpperGraphReplica::estimated_runtime_bytes() const noexcept {
  return persistent_bytes() + lookup_.size() * sizeof(std::uint32_t);
}

Status LocalShard::Validate(const GlobalIndexMeta &meta, ShardRange range,
                            std::span<const node_id_t> layer0,
                            std::size_t vector_count) {
  auto status = ValidateGlobalMeta(meta);
  if (!status.ok()) {
    return status;
  }
  if (range.shard_id >= meta.num_shards || range.local_count == 0 ||
      static_cast<std::uint64_t>(range.global_begin) + range.local_count >
          meta.ntotal) {
    return Status(StatusCode::kInvalidFormat, "local shard range is invalid");
  }
  auto graph_slots =
      CheckedMultiply(range.local_count, static_cast<std::uint64_t>(meta.M0));
  auto vector_slots =
      CheckedMultiply(range.local_count, static_cast<std::uint64_t>(meta.dim));
  if (!graph_slots.ok() || graph_slots.value() != layer0.size() ||
      !vector_slots.ok() || vector_slots.value() != vector_count) {
    return Status(StatusCode::kInvalidFormat,
                  "local shard block shape mismatch");
  }
  return ValidateCanonicalAdjacency(layer0, meta.M0, meta.ntotal,
                                    "layer-0 graph");
}

Result<LocalShard> LocalShard::Create(const GlobalIndexMeta &meta,
                                      ShardRange range,
                                      std::vector<node_id_t> layer0,
                                      std::vector<float> vectors,
                                      LocalShardStorage storage) {
  const auto graph_bytes = std::as_bytes(std::span<const node_id_t>(layer0));
  const auto vector_bytes = std::as_bytes(std::span<const float>(vectors));
  auto graph_storage = storage == LocalShardStorage::kShared
                           ? PageAlignedBuffer::CopyFromShared(graph_bytes)
                           : PageAlignedBuffer::CopyFrom(graph_bytes);
  if (!graph_storage.ok()) {
    return graph_storage.status();
  }
  auto vector_storage = storage == LocalShardStorage::kShared
                            ? PageAlignedBuffer::CopyFromShared(vector_bytes)
                            : PageAlignedBuffer::CopyFrom(vector_bytes);
  if (!vector_storage.ok()) {
    return vector_storage.status();
  }
  return Create(meta, range, std::move(graph_storage).value(),
                std::move(vector_storage).value());
}

Result<LocalShard> LocalShard::Create(const GlobalIndexMeta &meta,
                                      ShardRange range,
                                      PageAlignedBuffer graph_storage,
                                      PageAlignedBuffer vector_storage) {
  if (graph_storage.logical_bytes() % sizeof(node_id_t) != 0) {
    return Status(StatusCode::kInvalidFormat,
                  "local shard graph byte size is invalid");
  }
  if (vector_storage.logical_bytes() % sizeof(float) != 0) {
    return Status(StatusCode::kInvalidFormat,
                  "local shard vector byte size is invalid");
  }
  const auto graph_memory = graph_storage.logical_memory();
  const auto graph = std::span<const node_id_t>(
      reinterpret_cast<const node_id_t *>(graph_memory.data()),
      graph_memory.size() / sizeof(node_id_t));
  auto status = Validate(meta, range, graph,
                         vector_storage.logical_bytes() / sizeof(float));
  if (!status.ok()) {
    return status;
  }
  LocalShard shard;
  shard.meta_ = meta;
  shard.range_ = range;
  shard.graph_storage_ = std::move(graph_storage);
  shard.vector_storage_ = std::move(vector_storage);
  try {
    shard.layer0_degrees_.resize(range.local_count);
    shard.vector_norms_.resize(range.local_count);
    const auto stored_graph = shard.layer0_slots();
    const auto stored_vectors = shard.vectors();
    for (std::size_t row = 0; row < range.local_count; ++row) {
      const auto adjacency_begin = row * meta.M0;
      const auto row_adjacency = std::span<const node_id_t>(stored_graph)
                                     .subspan(adjacency_begin, meta.M0);
      shard.layer0_degrees_[row] = static_cast<std::uint32_t>(
          std::distance(row_adjacency.begin(),
                        std::find(row_adjacency.begin(), row_adjacency.end(),
                                  INVALID_NODE_ID)));
      float norm = 0.0F;
      const auto begin = row * meta.dim;
      for (std::size_t column = 0; column < meta.dim; ++column) {
        const auto value = stored_vectors[begin + column];
        norm += value * value;
      }
      shard.vector_norms_[row] = norm;
    }
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate local vector norms");
  }
  return shard;
}

Result<std::span<const node_id_t>>
LocalShard::Neighbors(node_id_t id) const noexcept {
  if (id < range_.global_begin ||
      static_cast<std::uint64_t>(id - range_.global_begin) >=
          range_.local_count) {
    return Status(StatusCode::kOutOfRange, "node is not owned by this shard");
  }
  const auto local = static_cast<std::uint64_t>(id - range_.global_begin);
  return layer0_slots().subspan(static_cast<std::size_t>(local * meta_.M0),
                                meta_.M0);
}

std::span<const node_id_t>
LocalShard::TrustedNeighbors(std::uint64_t local_offset) const noexcept {
  assert(local_offset < range_.local_count);
  const auto begin = static_cast<std::size_t>(local_offset * meta_.M0);
  return layer0_slots().subspan(begin, layer0_degrees_[local_offset]);
}

std::span<const node_id_t> LocalShard::layer0_slots() const noexcept {
  const auto memory = graph_storage_.logical_memory();
  assert(memory.size() % sizeof(node_id_t) == 0);
  return {reinterpret_cast<const node_id_t *>(memory.data()),
          memory.size() / sizeof(node_id_t)};
}

Result<std::span<const float>> LocalShard::Vector(node_id_t id) const noexcept {
  if (id < range_.global_begin ||
      static_cast<std::uint64_t>(id - range_.global_begin) >=
          range_.local_count) {
    return Status(StatusCode::kOutOfRange, "vector is not owned by this shard");
  }
  const auto local = static_cast<std::uint64_t>(id - range_.global_begin);
  return vectors().subspan(static_cast<std::size_t>(local * meta_.dim),
                           meta_.dim);
}

std::span<const float>
LocalShard::TrustedVector(std::uint64_t local_offset) const noexcept {
  assert(local_offset < range_.local_count);
  const auto begin = static_cast<std::size_t>(local_offset * meta_.dim);
  return vectors().subspan(begin, meta_.dim);
}

std::span<const float> LocalShard::vectors() const noexcept {
  const auto memory = vector_storage_.logical_memory();
  assert(memory.size() % sizeof(float) == 0);
  return {reinterpret_cast<const float *>(memory.data()),
          memory.size() / sizeof(float)};
}

float LocalShard::TrustedVectorNorm(std::uint64_t local_offset) const noexcept {
  assert(local_offset < range_.local_count);
  return vector_norms_[local_offset];
}

void LocalShard::PrefetchVector(std::uint64_t local_offset) const noexcept {
#if defined(__GNUC__) || defined(__clang__)
  if (local_offset < range_.local_count) {
    const auto begin = static_cast<std::size_t>(local_offset * meta_.dim);
    __builtin_prefetch(vectors().data() + begin, 0, 1);
  }
#else
  static_cast<void>(local_offset);
#endif
}

} // namespace hnsw_urma
