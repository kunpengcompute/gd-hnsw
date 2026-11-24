#include "hnsw_urma/index/format.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>

#include "hnsw_urma/core/checked_math.hpp"

namespace hnsw_urma {

UpperGraphReplica::UpperGraphReplica() = default;
UpperGraphReplica::~UpperGraphReplica() = default;

UpperGraphReplica::UpperGraphReplica(UpperGraphReplica &&other) noexcept {
  *this = std::move(other);
}

UpperGraphReplica &
UpperGraphReplica::operator=(UpperGraphReplica &&other) noexcept {
  if (this != &other) {
    meta_ = other.meta_;
    owned_nodes_ = std::move(other.owned_nodes_);
    owned_neighbors_ = std::move(other.owned_neighbors_);
    lookup_ = std::move(other.lookup_);
    nodes_ = owned_nodes_;
    neighbors_ = owned_neighbors_;
    other.nodes_ = {};
    other.neighbors_ = {};
  }
  return *this;
}

namespace {

constexpr std::size_t kHeaderChecksumOffset =
    offsetof(IndexHeaderV2, header_checksum);

template <typename T>
std::span<const std::byte> AsBytes(std::span<const T> values) {
  return std::as_bytes(values);
}

std::uint32_t ExpectedBlockCount(IndexFileKind kind) {
  switch (kind) {
  case IndexFileKind::kGlobalMeta:
    return 1;
  case IndexFileKind::kUpperGraph:
  case IndexFileKind::kShard:
    return 2;
  }
  return 0;
}

Status ValidateHeader(const IndexHeaderV2 &header, IndexFileKind expected_kind,
                      std::uint64_t file_bytes) {
  if constexpr (std::endian::native != std::endian::little) {
    return Status(StatusCode::kInvalidFormat,
                  "IndexFormatV2 supports little-endian hosts only");
  }
  if (header.magic != INDEX_FORMAT_V2_MAGIC ||
      header.version != INDEX_FORMAT_V2_VERSION ||
      header.header_bytes != INDEX_FORMAT_V2_HEADER_BYTES ||
      header.kind != expected_kind || header.flags != 0 ||
      header.reserved32 != 0) {
    return Status(StatusCode::kInvalidFormat,
                  "index header magic, version, kind, or flags mismatch");
  }
  if (header.header_checksum != IndexHeaderChecksum(header)) {
    return Status(StatusCode::kInvalidFormat, "index header checksum mismatch");
  }
  auto meta_status = ValidateGlobalMeta(header.meta);
  if (!meta_status.ok()) {
    return meta_status;
  }
  if (header.block_count != ExpectedBlockCount(expected_kind)) {
    return Status(StatusCode::kInvalidFormat, "index block count mismatch");
  }
  if (std::any_of(header.reserved.begin(), header.reserved.end(),
                  [](std::byte value) { return value != std::byte{0}; })) {
    return Status(StatusCode::kInvalidFormat,
                  "index header reserved bytes must be zero");
  }

  std::uint64_t previous_end = INDEX_FORMAT_V2_HEADER_BYTES;
  for (std::size_t index = 0; index < header.block_count; ++index) {
    const auto &block = header.blocks[index];
    if (block.alignment < alignof(std::uint64_t) ||
        (block.alignment & (block.alignment - 1)) != 0 ||
        !IsAligned(block.offset, block.alignment) ||
        block.offset < previous_end) {
      return Status(StatusCode::kInvalidFormat,
                    "index block offset/alignment is invalid")
          .AddContext("block", std::to_string(index));
    }
    auto end = CheckedAdd(block.offset, block.bytes);
    if (!end.ok() || end.value() > file_bytes) {
      return Status(StatusCode::kInvalidFormat,
                    "index block exceeds file bounds")
          .AddContext("block", std::to_string(index));
    }
    previous_end = end.value();
  }
  for (std::size_t index = header.block_count; index < header.blocks.size();
       ++index) {
    if (header.blocks[index] != BlockDescV2{}) {
      return Status(StatusCode::kInvalidFormat,
                    "unused index block descriptors must be zero");
    }
  }
  return Status::Ok();
}

Result<std::pair<IndexHeaderV2, std::uint64_t>>
ReadHeader(const std::filesystem::path &path, IndexFileKind kind) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return Status(StatusCode::kInvalidArgument, "unable to open index file")
        .AddContext("path", path.string());
  }
  const auto end = input.tellg();
  if (end < static_cast<std::streamoff>(sizeof(IndexHeaderV2))) {
    return Status(StatusCode::kInvalidFormat,
                  "index file is truncated before its header")
        .AddContext("path", path.string());
  }
  const auto bytes = static_cast<std::uint64_t>(end);
  input.seekg(0);
  IndexHeaderV2 header;
  input.read(reinterpret_cast<char *>(&header),
             static_cast<std::streamsize>(sizeof(header)));
  if (!input) {
    return Status(StatusCode::kInvalidFormat, "unable to read index header");
  }
  auto status = ValidateHeader(header, kind, bytes);
  if (!status.ok()) {
    status.AddContext("path", path.string());
    return status;
  }
  return std::make_pair(header, bytes);
}

template <typename T>
Result<std::vector<T>> ReadArrayBlock(const std::filesystem::path &path,
                                      const BlockDescV2 &block,
                                      std::string_view block_type) {
  if (block.bytes % sizeof(T) != 0) {
    return Status(StatusCode::kInvalidFormat,
                  "index block element size mismatch")
        .AddContext("block_type", std::string(block_type));
  }
  if (block.bytes > std::numeric_limits<std::size_t>::max() ||
      block.bytes > static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamsize>::max())) {
    return Status(StatusCode::kResourceExhausted,
                  "index block is too large for this process")
        .AddContext("block_type", std::string(block_type));
  }
  try {
    std::vector<T> values(static_cast<std::size_t>(block.bytes / sizeof(T)));
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(block.offset));
    if (!values.empty()) {
      input.read(reinterpret_cast<char *>(values.data()),
                 static_cast<std::streamsize>(block.bytes));
    }
    if (!input) {
      return Status(StatusCode::kInvalidFormat, "index block short read")
          .AddContext("block_type", std::string(block_type));
    }
    if (IndexChecksum(AsBytes(std::span<const T>(values))) != block.checksum) {
      return Status(StatusCode::kInvalidFormat, "index block checksum mismatch")
          .AddContext("block_type", std::string(block_type));
    }
    return values;
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to allocate index block")
        .AddContext("block_type", std::string(block_type));
  }
}

Result<PageAlignedBuffer> ReadAlignedBlock(const std::filesystem::path &path,
                                           const BlockDescV2 &block,
                                           LocalShardStorage storage,
                                           std::string_view block_type) {
  if (block.bytes == 0) {
    return Status(StatusCode::kInvalidFormat, "index aligned block is empty")
        .AddContext("block_type", std::string(block_type));
  }
  if (block.bytes > std::numeric_limits<std::size_t>::max() ||
      block.bytes > static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamsize>::max())) {
    return Status(StatusCode::kResourceExhausted,
                  "index aligned block is too large for this process")
        .AddContext("block_type", std::string(block_type));
  }
  auto buffer =
      storage == LocalShardStorage::kShared
          ? PageAlignedBuffer::CreateShared(
                static_cast<std::size_t>(block.bytes))
          : PageAlignedBuffer::Create(static_cast<std::size_t>(block.bytes));
  if (!buffer.ok()) {
    return buffer.status();
  }
  auto memory = buffer.value().logical_memory();
  std::ifstream input(path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(block.offset));
  input.read(reinterpret_cast<char *>(memory.data()),
             static_cast<std::streamsize>(memory.size()));
  if (!input) {
    return Status(StatusCode::kInvalidFormat, "index aligned block short read")
        .AddContext("block_type", std::string(block_type));
  }
  if (IndexChecksum(memory) != block.checksum) {
    return Status(StatusCode::kInvalidFormat,
                  "index aligned block checksum mismatch")
        .AddContext("block_type", std::string(block_type));
  }
  return std::move(buffer).value();
}

Status WriteFile(const std::filesystem::path &path, IndexHeaderV2 header,
                 std::span<const std::span<const std::byte>> blocks) {
  if (blocks.size() != header.block_count) {
    return Status(StatusCode::kInternalError, "writer block count mismatch");
  }
  std::uint64_t offset = INDEX_FORMAT_V2_HEADER_BYTES;
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    auto aligned = CheckedAlignUp(offset, INDEX_BLOCK_ALIGNMENT);
    if (!aligned.ok()) {
      return aligned.status();
    }
    header.blocks[index] = {aligned.value(), blocks[index].size(),
                            IndexChecksum(blocks[index]),
                            INDEX_BLOCK_ALIGNMENT};
    auto next = CheckedAdd(aligned.value(),
                           static_cast<std::uint64_t>(blocks[index].size()));
    if (!next.ok()) {
      return next.status();
    }
    offset = next.value();
  }
  header.header_checksum = IndexHeaderChecksum(header);

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Status(StatusCode::kInvalidArgument, "unable to create index file")
        .AddContext("path", path.string());
  }
  output.write(reinterpret_cast<const char *>(&header),
               static_cast<std::streamsize>(sizeof(header)));
  std::uint64_t written = sizeof(header);
  static constexpr std::array<char, 4096> zeros{};
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    const auto padding = header.blocks[index].offset - written;
    if (padding != 0) {
      output.write(zeros.data(), static_cast<std::streamsize>(padding));
    }
    output.write(reinterpret_cast<const char *>(blocks[index].data()),
                 static_cast<std::streamsize>(blocks[index].size()));
    written = header.blocks[index].offset + blocks[index].size();
  }
  if (!output) {
    return Status(StatusCode::kInternalError, "failed to write index file")
        .AddContext("path", path.string());
  }
  return Status::Ok();
}

bool SameMeta(const GlobalIndexMeta &lhs, const GlobalIndexMeta &rhs) {
  return lhs == rhs;
}

} // namespace

std::uint64_t IndexChecksum(std::span<const std::byte> bytes) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::byte byte : bytes) {
    hash ^= std::to_integer<std::uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t IndexHeaderChecksum(const IndexHeaderV2 &header) noexcept {
  std::array<std::byte, sizeof(IndexHeaderV2)> bytes{};
  std::memcpy(bytes.data(), &header, sizeof(header));
  std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderChecksumOffset),
            bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderChecksumOffset +
                                                        sizeof(std::uint64_t)),
            std::byte{0});
  return IndexChecksum(bytes);
}

std::filesystem::path ShardFilePath(const std::filesystem::path &root,
                                    shard_id_t shard_id) {
  return root / ("shard_" + std::to_string(shard_id) + ".index");
}

Status WriteGlobalIndexFile(const std::filesystem::path &root,
                            const GlobalIndexMeta &meta,
                            std::span<const ShardRange> ranges) {
  auto status = ValidateGlobalMeta(meta);
  if (!status.ok()) {
    return status;
  }
  auto map = ShardMap::Create(ranges, meta.ntotal);
  if (!map.ok() || ranges.size() != meta.num_shards) {
    return map.ok() ? Status(StatusCode::kInvalidArgument,
                             "global shard count mismatch")
                    : map.status();
  }
  std::error_code directory_error;
  std::filesystem::create_directories(root, directory_error);
  if (directory_error) {
    return Status(StatusCode::kInvalidArgument,
                  "unable to create index directory")
        .AddContext("path", root.string())
        .AddContext("error", directory_error.message());
  }
  IndexHeaderV2 header;
  header.kind = IndexFileKind::kGlobalMeta;
  header.meta = meta;
  header.block_count = 1;
  const auto range_bytes = AsBytes(ranges);
  const std::array blocks{range_bytes};
  return WriteFile(root / "global.meta", header, blocks);
}

Status WriteUpperGraphFile(const std::filesystem::path &root,
                           const GlobalIndexMeta &meta,
                           std::span<const UpperNodeMeta> nodes,
                           std::span<const node_id_t> neighbors) {
  Result<UpperGraphReplica> replica =
      Status(StatusCode::kInternalError, "not initialized");
  try {
    replica = UpperGraphReplica::Create(
        meta, std::vector<UpperNodeMeta>(nodes.begin(), nodes.end()),
        std::vector<node_id_t>(neighbors.begin(), neighbors.end()));
  } catch (const std::bad_alloc &) {
    return Status(StatusCode::kResourceExhausted,
                  "unable to validate upper graph blocks");
  }
  if (!replica.ok()) {
    return replica.status();
  }
  IndexHeaderV2 header;
  header.kind = IndexFileKind::kUpperGraph;
  header.meta = meta;
  header.upper_node_count = nodes.size();
  header.upper_slot_count = neighbors.size();
  header.block_count = 2;
  const std::array blocks{AsBytes(nodes), AsBytes(neighbors)};
  return WriteFile(root / "upper_graph.index", header, blocks);
}

Status WriteShardFile(const std::filesystem::path &root,
                      const GlobalIndexMeta &meta, const ShardRange &range,
                      std::span<const node_id_t> layer0,
                      std::span<const float> vectors) {
  auto validation = LocalShard::Validate(meta, range, layer0, vectors.size());
  if (!validation.ok()) {
    return validation;
  }
  IndexHeaderV2 header;
  header.kind = IndexFileKind::kShard;
  header.meta = meta;
  header.shard_id = range.shard_id;
  header.global_begin = range.global_begin;
  header.local_count = range.local_count;
  header.block_count = 2;
  const std::array blocks{AsBytes(layer0), AsBytes(vectors)};
  return WriteFile(ShardFilePath(root, range.shard_id), header, blocks);
}

Result<GlobalIndexFile> LoadGlobalIndexFile(const std::filesystem::path &root) {
  const auto path = root / "global.meta";
  auto read = ReadHeader(path, IndexFileKind::kGlobalMeta);
  if (!read.ok()) {
    return read.status();
  }
  auto ranges = ReadArrayBlock<ShardRange>(path, read.value().first.blocks[0],
                                           "shard-ranges");
  if (!ranges.ok() ||
      ranges.value().size() != read.value().first.meta.num_shards) {
    return ranges.ok() ? Status(StatusCode::kInvalidFormat,
                                "global shard range count mismatch")
                       : ranges.status();
  }
  auto map = ShardMap::Create(ranges.value(), read.value().first.meta.ntotal);
  if (!map.ok()) {
    return Status(StatusCode::kInvalidFormat, map.status().ToString());
  }
  return GlobalIndexFile{read.value().first.meta, std::move(ranges).value()};
}

Result<UpperGraphReplica>
LoadUpperGraphFile(const std::filesystem::path &root,
                   const GlobalIndexMeta &expected_meta) {
  const auto path = root / "upper_graph.index";
  auto read = ReadHeader(path, IndexFileKind::kUpperGraph);
  if (!read.ok()) {
    return read.status();
  }
  const auto &header = read.value().first;
  if (!SameMeta(header.meta, expected_meta)) {
    return Status(StatusCode::kInvalidFormat,
                  "upper graph metadata differs from global.meta");
  }
  auto nodes =
      ReadArrayBlock<UpperNodeMeta>(path, header.blocks[0], "upper-nodes");
  if (!nodes.ok()) {
    return nodes.status();
  }
  auto neighbors =
      ReadArrayBlock<node_id_t>(path, header.blocks[1], "upper-neighbors");
  if (!neighbors.ok()) {
    return neighbors.status();
  }
  if (nodes.value().size() != header.upper_node_count ||
      neighbors.value().size() != header.upper_slot_count) {
    return Status(StatusCode::kInvalidFormat,
                  "upper graph header count mismatch");
  }
  return UpperGraphReplica::Create(header.meta, std::move(nodes).value(),
                                   std::move(neighbors).value());
}

Result<LocalShard> LoadShardFile(const std::filesystem::path &root,
                                 const GlobalIndexMeta &expected_meta,
                                 const ShardRange &expected_range,
                                 LocalShardStorage storage) {
  const auto path = ShardFilePath(root, expected_range.shard_id);
  auto read = ReadHeader(path, IndexFileKind::kShard);
  if (!read.ok()) {
    return read.status();
  }
  const auto &header = read.value().first;
  if (!SameMeta(header.meta, expected_meta) ||
      header.shard_id != expected_range.shard_id ||
      header.global_begin != expected_range.global_begin ||
      header.local_count != expected_range.local_count) {
    return Status(StatusCode::kInvalidFormat,
                  "shard header differs from global.meta");
  }
  if (header.blocks[0].bytes % sizeof(node_id_t) != 0) {
    return Status(StatusCode::kInvalidFormat,
                  "index graph block element size mismatch");
  }
  if (header.blocks[1].bytes % sizeof(float) != 0) {
    return Status(StatusCode::kInvalidFormat,
                  "index vector block element size mismatch");
  }
  auto graph_storage =
      ReadAlignedBlock(path, header.blocks[0], storage, "layer0");
  if (!graph_storage.ok()) {
    return graph_storage.status();
  }
  auto vector_storage =
      ReadAlignedBlock(path, header.blocks[1], storage, "vectors");
  if (!vector_storage.ok()) {
    return vector_storage.status();
  }
  return LocalShard::Create(header.meta, expected_range,
                            std::move(graph_storage).value(),
                            std::move(vector_storage).value());
}

Result<LoadedIndex> LoadIndexForShard(const std::filesystem::path &root,
                                      shard_id_t shard_id,
                                      LocalShardStorage storage) {
  auto global = LoadGlobalIndexFile(root);
  if (!global.ok()) {
    return global.status();
  }
  if (shard_id >= global.value().ranges.size()) {
    return Status(StatusCode::kOutOfRange,
                  "requested shard is outside global metadata");
  }
  auto map =
      ShardMap::Create(global.value().ranges, global.value().meta.ntotal);
  auto upper = LoadUpperGraphFile(root, global.value().meta);
  auto shard = LoadShardFile(root, global.value().meta,
                             global.value().ranges[shard_id], storage);
  if (!map.ok()) {
    return map.status();
  }
  if (!upper.ok()) {
    return upper.status();
  }
  if (!shard.ok()) {
    return shard.status();
  }
  auto local_entry =
      upper.value().EntryPointForShard(global.value().ranges[shard_id]);
  if (!local_entry.ok()) {
    return local_entry.status();
  }
  return LoadedIndex{global.value().meta, std::move(map).value(),
                     std::move(upper).value(), std::move(shard).value(),
                     local_entry.value()};
}

} // namespace hnsw_urma
