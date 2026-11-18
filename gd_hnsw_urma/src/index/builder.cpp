#include "hnsw_urma/index/builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "hnsw_urma/index/format.hpp"

namespace hnsw_urma {
namespace {

struct ScoredId {
  float distance;
  node_id_t id;
};

std::vector<node_id_t> Nearest(std::span<const float> vectors, std::uint64_t rows,
                               std::uint32_t dim, node_id_t source,
                               std::span<const node_id_t> candidates, std::uint32_t count,
                               std::uint32_t metric) {
  std::vector<ScoredId> scored;
  scored.reserve(candidates.size());
  const auto source_vector = vectors.subspan(static_cast<std::size_t>(source) * dim, dim);
  for (node_id_t candidate : candidates) {
    if (candidate == source) {
      continue;
    }
    const auto target = vectors.subspan(static_cast<std::size_t>(candidate) * dim, dim);
    float value = 0.0F;
    for (std::uint32_t component = 0; component < dim; ++component) {
      if (metric == 0) {
        const float delta = source_vector[component] - target[component];
        value += delta * delta;
      } else {
        value -= source_vector[component] * target[component];
      }
    }
    scored.push_back({value, candidate});
  }
  std::sort(scored.begin(), scored.end(), [](const ScoredId& lhs, const ScoredId& rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  });
  std::vector<node_id_t> result(count, INVALID_NODE_ID);
  for (std::size_t index = 0; index < std::min(result.size(), scored.size()); ++index) {
    result[index] = scored[index].id;
  }
  static_cast<void>(rows);
  return result;
}

std::uint16_t DeterministicLevel(node_id_t id, std::uint32_t M, std::uint64_t rows) {
  if (rows <= 1) {
    return 0;
  }
  std::uint16_t level = 0;
  std::uint64_t value = static_cast<std::uint64_t>(id) + 1;
  const std::uint64_t divisor = std::max<std::uint32_t>(2, M);
  while (value % divisor == 0 && level < 32) {
    ++level;
    value /= divisor;
  }
  return level;
}

}  // namespace

Status BuildReferenceIndex(const std::filesystem::path& root, std::span<const float> vectors,
                           std::uint64_t rows, std::uint32_t dim,
                           const ReferenceBuildOptions& options, std::uint64_t max_rows) {
  if (rows == 0 || rows > std::numeric_limits<node_id_t>::max() || dim == 0 ||
      options.num_shards == 0 || options.num_shards > rows || options.M == 0 ||
      options.M0 == 0 || options.ef_search == 0 || rows > max_rows ||
      options.metric > 1 ||
      rows > vectors.size() / dim || rows * dim != vectors.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "reference builder shape/options invalid or exceeds its explicit row limit");
  }

  std::vector<ShardRange> ranges;
  ranges.reserve(options.num_shards);
  const std::uint64_t base = rows / options.num_shards;
  const std::uint64_t remainder = rows % options.num_shards;
  std::uint64_t begin = 0;
  for (shard_id_t shard = 0; shard < options.num_shards; ++shard) {
    const std::uint64_t count = base + (shard < remainder ? 1U : 0U);
    ranges.push_back({shard, static_cast<node_id_t>(begin), count});
    begin += count;
  }

  std::vector<node_id_t> all_ids(static_cast<std::size_t>(rows));
  std::iota(all_ids.begin(), all_ids.end(), node_id_t{0});
  std::vector<node_id_t> layer0(static_cast<std::size_t>(rows) * options.M0, INVALID_NODE_ID);
  for (node_id_t id = 0; id < rows; ++id) {
    auto nearest = Nearest(vectors, rows, dim, id, all_ids, options.M0, options.metric);
    std::copy(nearest.begin(), nearest.end(),
              layer0.begin() + static_cast<std::ptrdiff_t>(static_cast<std::uint64_t>(id) * options.M0));
  }

  std::vector<std::uint16_t> levels(static_cast<std::size_t>(rows));
  std::uint16_t max_level = 0;
  node_id_t entry_point = 0;
  for (node_id_t id = 0; id < rows; ++id) {
    levels[id] = DeterministicLevel(id, options.M, rows);
    if (levels[id] > max_level) {
      max_level = levels[id];
      entry_point = id;
    }
  }
  if (max_level == 0 && rows > 1) {
    levels[0] = 1;
    max_level = 1;
    entry_point = 0;
  }

  std::vector<UpperNodeMeta> upper_nodes;
  std::vector<node_id_t> upper_neighbors;
  for (node_id_t id = 0; id < rows; ++id) {
    if (levels[id] == 0) {
      continue;
    }
    upper_nodes.push_back({id, levels[id], 0, upper_neighbors.size()});
    for (std::uint16_t level = 1; level <= levels[id]; ++level) {
      std::vector<node_id_t> candidates;
      for (node_id_t candidate = 0; candidate < rows; ++candidate) {
        if (levels[candidate] >= level) {
          candidates.push_back(candidate);
        }
      }
      auto nearest = Nearest(vectors, rows, dim, id, candidates, options.M, options.metric);
      upper_neighbors.insert(upper_neighbors.end(), nearest.begin(), nearest.end());
    }
  }

  GlobalIndexMeta meta;
  meta.ntotal = rows;
  meta.dim = dim;
  meta.metric = options.metric;
  meta.M = options.M;
  meta.M0 = options.M0;
  meta.ef_search_default = options.ef_search;
  meta.entry_point = entry_point;
  meta.max_level = max_level;
  meta.num_shards = options.num_shards;

  auto status = WriteGlobalIndexFile(root, meta, ranges);
  if (!status.ok()) {
    return status;
  }
  status = WriteUpperGraphFile(root, meta, upper_nodes, upper_neighbors);
  if (!status.ok()) {
    return status;
  }
  for (const auto& range : ranges) {
    const auto graph_begin = static_cast<std::size_t>(range.global_begin) * options.M0;
    const auto vector_begin = static_cast<std::size_t>(range.global_begin) * dim;
    status = WriteShardFile(
        root, meta, range,
        std::span<const node_id_t>(layer0).subspan(graph_begin,
                                                   static_cast<std::size_t>(range.local_count) * options.M0),
        vectors.subspan(vector_begin, static_cast<std::size_t>(range.local_count) * dim));
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

}  // namespace hnsw_urma
