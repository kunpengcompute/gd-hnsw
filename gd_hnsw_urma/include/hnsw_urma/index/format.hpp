#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "hnsw_urma/core/page_aligned_buffer.hpp"
#include "hnsw_urma/core/shard_map.hpp"
#include "hnsw_urma/core/status.hpp"
#include "hnsw_urma/core/types.hpp"

namespace hnsw_urma {

inline constexpr std::uint64_t INDEX_FORMAT_V2_MAGIC = 0x3258444E494D5248ULL;
inline constexpr std::uint16_t INDEX_FORMAT_V2_VERSION = 2;
inline constexpr std::uint16_t INDEX_FORMAT_V2_HEADER_BYTES = 4096;
inline constexpr std::uint32_t INDEX_FORMAT_V2_ID_WIDTH = 4;
inline constexpr std::uint64_t INDEX_BLOCK_ALIGNMENT = 4096;
inline constexpr std::size_t INDEX_MAX_BLOCKS = 4;

enum class IndexFileKind : std::uint16_t {
  kGlobalMeta = 1,
  kUpperGraph = 2,
  kShard = 3,
};

struct GlobalIndexMeta {
  std::uint64_t ntotal{0};
  std::uint32_t dim{0};
  std::uint32_t metric{0};
  std::uint32_t M{0};
  std::uint32_t M0{0};
  std::uint32_t ef_search_default{0};
  node_id_t entry_point{INVALID_NODE_ID};
  std::int32_t max_level{-1};
  std::uint32_t num_shards{0};
  std::uint8_t id_width{INDEX_FORMAT_V2_ID_WIDTH};
  std::array<std::uint8_t, 7> reserved{};

  friend bool operator==(const GlobalIndexMeta&, const GlobalIndexMeta&) = default;
};
static_assert(sizeof(GlobalIndexMeta) == 48);

struct BlockDescV2 {
  file_offset_t offset{0};
  std::uint64_t bytes{0};
  std::uint64_t checksum{0};
  std::uint64_t alignment{0};

  friend bool operator==(const BlockDescV2&, const BlockDescV2&) = default;
};
static_assert(sizeof(BlockDescV2) == 32);

struct UpperNodeMeta {
  node_id_t global_id{INVALID_NODE_ID};
  std::uint16_t max_level{0};
  std::uint16_t reserved16{0};
  std::uint64_t neighbor_offset{0};  // node_id_t slot index, not a byte offset

  friend bool operator==(const UpperNodeMeta&, const UpperNodeMeta&) = default;
};
static_assert(sizeof(UpperNodeMeta) == 16);
static_assert(sizeof(ShardRange) == 16);

struct ShardEntryPoint {
  node_id_t global_id{INVALID_NODE_ID};
  std::int32_t level{-1};

  friend bool operator==(const ShardEntryPoint&,
                         const ShardEntryPoint&) = default;
};

struct IndexHeaderV2 {
  std::uint64_t magic{INDEX_FORMAT_V2_MAGIC};
  std::uint16_t version{INDEX_FORMAT_V2_VERSION};
  std::uint16_t header_bytes{INDEX_FORMAT_V2_HEADER_BYTES};
  IndexFileKind kind{IndexFileKind::kGlobalMeta};
  std::uint16_t flags{0};
  std::uint64_t header_checksum{0};
  GlobalIndexMeta meta{};
  shard_id_t shard_id{0};
  node_id_t global_begin{0};
  std::uint64_t local_count{0};
  std::uint64_t upper_node_count{0};
  std::uint64_t upper_slot_count{0};
  std::uint32_t block_count{0};
  std::uint32_t reserved32{0};
  std::array<BlockDescV2, INDEX_MAX_BLOCKS> blocks{};
  std::array<std::byte, 3856> reserved{};
};
static_assert(sizeof(IndexHeaderV2) == INDEX_FORMAT_V2_HEADER_BYTES);
static_assert(alignof(IndexHeaderV2) == 8);

[[nodiscard]] std::uint64_t IndexChecksum(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] std::uint64_t IndexHeaderChecksum(const IndexHeaderV2& header) noexcept;
[[nodiscard]] Status ValidateGlobalMeta(const GlobalIndexMeta& meta) noexcept;

// The loader materializes upper graph blocks into owned memory before search;
// no search-time access depends on the backing index file.
class UpperGraphReplica {
 public:
  UpperGraphReplica();
  ~UpperGraphReplica();
  UpperGraphReplica(const UpperGraphReplica&) = delete;
  UpperGraphReplica& operator=(const UpperGraphReplica&) = delete;
  UpperGraphReplica(UpperGraphReplica&& other) noexcept;
  UpperGraphReplica& operator=(UpperGraphReplica&& other) noexcept;

  [[nodiscard]] static Result<UpperGraphReplica> Create(
      const GlobalIndexMeta& meta, std::vector<UpperNodeMeta> nodes,
      std::vector<node_id_t> neighbors);

  [[nodiscard]] Result<std::span<const node_id_t>> Neighbors(node_id_t id,
                                                             std::int32_t level) const noexcept;
  // Selects the highest-level node owned by the shard. Ties are resolved by
  // the lowest global ID. A shard without upper-level nodes starts directly
  // from its first layer-0 node.
  [[nodiscard]] Result<ShardEntryPoint> EntryPointForShard(
      const ShardRange& range) const noexcept;
  [[nodiscard]] std::span<const UpperNodeMeta> nodes() const noexcept { return nodes_; }
  [[nodiscard]] std::span<const node_id_t> neighbor_slots() const noexcept { return neighbors_; }
  [[nodiscard]] std::uint64_t persistent_bytes() const noexcept;
  [[nodiscard]] std::uint64_t estimated_runtime_bytes() const noexcept;

 private:
  [[nodiscard]] Status Initialize(
      const GlobalIndexMeta& meta,
      std::span<const UpperNodeMeta> nodes,
      std::span<const node_id_t> neighbors);
  [[nodiscard]] std::optional<std::uint32_t> FindNode(
      node_id_t id) const noexcept;

  GlobalIndexMeta meta_{};
  std::vector<UpperNodeMeta> owned_nodes_;
  std::vector<node_id_t> owned_neighbors_;
  std::span<const UpperNodeMeta> nodes_;
  std::span<const node_id_t> neighbors_;
  std::vector<std::uint32_t> lookup_;
};

enum class LocalShardStorage : std::uint8_t {
  kPrivate,
  kShared,
};

class LocalShard {
 public:
  [[nodiscard]] static Status Validate(
      const GlobalIndexMeta& meta, ShardRange range,
      std::span<const node_id_t> layer0,
      std::size_t vector_count);
  [[nodiscard]] static Result<LocalShard> Create(const GlobalIndexMeta& meta, ShardRange range,
                                                 std::vector<node_id_t> layer0,
                                                 std::vector<float> vectors,
                                                 LocalShardStorage storage =
                                                     LocalShardStorage::kPrivate);
  [[nodiscard]] static Result<LocalShard> Create(
      const GlobalIndexMeta& meta, ShardRange range,
      PageAlignedBuffer graph_storage,
      PageAlignedBuffer vector_storage);

  [[nodiscard]] Result<std::span<const node_id_t>> Neighbors(node_id_t id) const noexcept;
  // Internal response fast path. The caller must have validated the owner
  // and converted the global ID to this shard's local offset.
  [[nodiscard]] std::span<const node_id_t> TrustedNeighbors(
      std::uint64_t local_offset) const noexcept;
  [[nodiscard]] Result<std::span<const float>> Vector(node_id_t id) const noexcept;
  // Internal search fast path. The caller must have validated the owner and
  // converted the global ID to this shard's local offset.
  [[nodiscard]] std::span<const float> TrustedVector(
      std::uint64_t local_offset) const noexcept;
  [[nodiscard]] float TrustedVectorNorm(
      std::uint64_t local_offset) const noexcept;
  void PrefetchVector(std::uint64_t local_offset) const noexcept;
  [[nodiscard]] const ShardRange& range() const noexcept { return range_; }
  [[nodiscard]] std::span<const node_id_t> layer0_slots() const noexcept;
  [[nodiscard]] std::span<std::byte> graph_logical_memory() noexcept {
    return graph_storage_.logical_memory();
  }
  [[nodiscard]] std::span<const std::byte> graph_logical_memory() const noexcept {
    return graph_storage_.logical_memory();
  }
  [[nodiscard]] std::span<std::byte> graph_registration_memory() noexcept {
    return graph_storage_.registration_memory();
  }
  [[nodiscard]] std::span<const std::byte> graph_registration_memory() const noexcept {
    return graph_storage_.registration_memory();
  }
  [[nodiscard]] bool graph_storage_shared() const noexcept {
    return graph_storage_.is_shared();
  }
  [[nodiscard]] std::string_view graph_shared_memory_name() const noexcept {
    return graph_storage_.shared_memory_name();
  }
  [[nodiscard]] Status UnlinkGraphSharedMemory() {
    return graph_storage_.UnlinkSharedMemory();
  }
  [[nodiscard]] std::span<const float> vectors() const noexcept;
  [[nodiscard]] std::span<std::byte> vector_logical_memory() noexcept {
    return vector_storage_.logical_memory();
  }
  [[nodiscard]] std::span<const std::byte> vector_logical_memory() const noexcept {
    return vector_storage_.logical_memory();
  }
  [[nodiscard]] std::span<std::byte> vector_registration_memory() noexcept {
    return vector_storage_.registration_memory();
  }
  [[nodiscard]] std::span<const std::byte> vector_registration_memory() const noexcept {
    return vector_storage_.registration_memory();
  }
  [[nodiscard]] bool vector_storage_shared() const noexcept {
    return vector_storage_.is_shared();
  }
  [[nodiscard]] std::string_view vector_shared_memory_name() const noexcept {
    return vector_storage_.shared_memory_name();
  }
  [[nodiscard]] Status UnlinkVectorSharedMemory() {
    return vector_storage_.UnlinkSharedMemory();
  }
  [[nodiscard]] std::span<const float> vector_norms() const noexcept {
    return vector_norms_;
  }

 private:
  GlobalIndexMeta meta_{};
  ShardRange range_{};
  PageAlignedBuffer graph_storage_;
  std::vector<std::uint32_t> layer0_degrees_;
  PageAlignedBuffer vector_storage_;
  std::vector<float> vector_norms_;
};

struct GlobalIndexFile {
  GlobalIndexMeta meta;
  std::vector<ShardRange> ranges;
};

struct LoadedIndex {
  GlobalIndexMeta meta;
  ShardMap shard_map;
  UpperGraphReplica upper_graph;
  LocalShard local_shard;
  ShardEntryPoint local_entry_point;
};

[[nodiscard]] Status WriteGlobalIndexFile(const std::filesystem::path& root,
                                          const GlobalIndexMeta& meta,
                                          std::span<const ShardRange> ranges);
[[nodiscard]] Status WriteUpperGraphFile(const std::filesystem::path& root,
                                         const GlobalIndexMeta& meta,
                                         std::span<const UpperNodeMeta> nodes,
                                         std::span<const node_id_t> neighbors);
[[nodiscard]] Status WriteShardFile(const std::filesystem::path& root,
                                    const GlobalIndexMeta& meta, const ShardRange& range,
                                    std::span<const node_id_t> layer0,
                                    std::span<const float> vectors);

[[nodiscard]] Result<GlobalIndexFile> LoadGlobalIndexFile(const std::filesystem::path& root);
[[nodiscard]] Result<UpperGraphReplica> LoadUpperGraphFile(const std::filesystem::path& root,
                                                           const GlobalIndexMeta& expected_meta);
[[nodiscard]] Result<LocalShard> LoadShardFile(const std::filesystem::path& root,
                                               const GlobalIndexMeta& expected_meta,
                                               const ShardRange& expected_range,
                                               LocalShardStorage storage =
                                                   LocalShardStorage::kPrivate);
[[nodiscard]] Result<LoadedIndex> LoadIndexForShard(const std::filesystem::path& root,
                                                   shard_id_t shard_id,
                                                   LocalShardStorage storage =
                                                       LocalShardStorage::kPrivate);

[[nodiscard]] std::filesystem::path ShardFilePath(const std::filesystem::path& root,
                                                  shard_id_t shard_id);

}  // namespace hnsw_urma
