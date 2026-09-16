/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace gd_hnsw {

// ============================================================
// Constants
// ============================================================

static constexpr uint64_t GD_MAGIC = 0x484E535747440000ULL; // "HNSWGD\0"
static constexpr uint32_t SCHEMA_VERSION = 1;
static constexpr uint32_t ENDIAN_LITTLE = 1;
static constexpr uint32_t METRIC_L2 = 1;
static constexpr uint64_t FNV1A64_OFFSET = 14695981039346656037ULL;
static constexpr uint64_t FNV1A64_PRIME = 1099511628211ULL;
static constexpr size_t SUPERBLOCK_SIZE = 4096; // one page
static constexpr size_t PAGE_ALIGN = 4096;
static constexpr size_t HUGEPAGE_ALIGN = 2 * 1024 * 1024;    // 2MB
static constexpr size_t MPI_CHUNK_BYTES = 128 * 1024 * 1024; // 128MiB
static constexpr int32_t EMPTY_NEIGHBOR = -1;

// VisitedTable: switch to sparse set above this threshold
// Dense mode: O(1) direct-index byte array (ntotal bytes, no hash overhead).
// Sparse mode: epoch-stamped flat hash table (512KB fixed, hash probe cost).
// At 10M nodes dense = 10MB (exceeds L2 but avoids probe overhead).
static constexpr uint64_t VISITED_SPARSE_THRESHOLD = 16 * 1024 * 1024; // 16M

// ============================================================
// Enums
// ============================================================

enum class GdState : uint32_t {
    BUILDING = 0,
    SEALED = 1,
    ACTIVE = 2,
};

enum BlockIndex : uint32_t {
    BLK_VECTORS = 0,
    BLK_LEVELS = 1,
    BLK_OFFSETS = 2,
    BLK_NEIGHBORS = 3,
    BLK_CUM_NNEIGHBOR = 4,
    BLK_COUNT = 5,
};

// ============================================================
// BlockDesc — describes one contiguous block in gd
// ============================================================

struct BlockDesc {
    uint64_t off;       // byte offset from gd base
    uint64_t bytes;     // payload size in bytes
    uint64_t crc64;     // xxhash64 of payload
    uint64_t alignment; // required alignment (PAGE_ALIGN or HUGEPAGE_ALIGN)
};

// ============================================================
// SuperBlockV1 — fixed 4KB header at offset 0
// ============================================================

struct SuperBlockV1 {
    // --- identity ---
    uint64_t magic;          // GD_MAGIC
    uint32_t schema_version; // SCHEMA_VERSION
    uint32_t state;          // GdState

    // --- faiss version used to build ---
    uint32_t faiss_major;
    uint32_t faiss_minor;
    uint32_t faiss_patch;

    // --- format metadata ---
    uint32_t metric_type;      // METRIC_L2
    uint32_t storage_idx_bits; // 32
    uint32_t endianness;       // ENDIAN_LITTLE

    // --- index parameters ---
    uint64_t ntotal; // global total vectors across all shards
    uint32_t dim;
    uint32_t M;
    uint32_t ef_search;
    int32_t max_level;
    int32_t entry_point; // global entry point id
    uint32_t num_levels; // length of cum_nneighbor_per_level array

    // --- sharding ---
    uint32_t num_shards;      // total number of NUMA shards (0 = legacy single-shard)
    uint32_t shard_id;        // this shard's index [0, num_shards)
    uint64_t ntotal_local;    // vectors in this shard
    uint64_t global_id_begin; // first global vector id in this shard

    // --- sizes ---
    uint64_t neighbors_size; // total neighbor elements in THIS shard
    uint64_t total_bytes;    // total gd region size for this shard

    // --- diagnostics ---
    uint64_t build_pid;
    uint64_t build_unix_ns;

    // --- integrity ---
    uint64_t header_crc64;  // crc of this struct (with header_crc64=0)
    uint64_t payload_crc64; // crc of all block payloads concatenated

    // --- block descriptors ---
    BlockDesc blocks[BLK_COUNT];

    // --- padding to 4KB ---
    // sizeof up to here is ~320 bytes, padded to SUPERBLOCK_SIZE
};

static_assert(sizeof(SuperBlockV1) <= SUPERBLOCK_SIZE, "SuperBlockV1 must fit in one page");

// ============================================================
// Inline accessors for gd-mapped data
// ============================================================

// Get pointer to a block's data given gd base address
inline const void *block_ptr(const void *gd_base, const SuperBlockV1 &sb, BlockIndex idx)
{
    return static_cast<const char *>(gd_base) + sb.blocks[idx].off;
}

inline const float *vectors_ptr(const void *gd_base, const SuperBlockV1 &sb)
{
    return static_cast<const float *>(block_ptr(gd_base, sb, BLK_VECTORS));
}

inline const int32_t *levels_ptr(const void *gd_base, const SuperBlockV1 &sb)
{
    return static_cast<const int32_t *>(block_ptr(gd_base, sb, BLK_LEVELS));
}

inline const uint64_t *offsets_ptr(const void *gd_base, const SuperBlockV1 &sb)
{
    return static_cast<const uint64_t *>(block_ptr(gd_base, sb, BLK_OFFSETS));
}

inline const int32_t *neighbors_ptr(const void *gd_base, const SuperBlockV1 &sb)
{
    return static_cast<const int32_t *>(block_ptr(gd_base, sb, BLK_NEIGHBORS));
}

inline const int32_t *cum_nneighbor_ptr(const void *gd_base, const SuperBlockV1 &sb)
{
    return static_cast<const int32_t *>(block_ptr(gd_base, sb, BLK_CUM_NNEIGHBOR));
}

// Get the neighbor range for node `no` at `layer` in the neighbors array
// NOTE: `no` must be a LOCAL id within the shard

// ============================================================
// Validation
// ============================================================

struct ValidationResult {
    bool ok = true;
    std::string error;
};

// Validate SuperBlock integrity and invariants.
// Full implementation in gd_layout.cpp
ValidationResult validate_superblock(const void *gd_base, uint64_t gd_size);

// ============================================================
// Layout computation — used by builder to compute block offsets
// ============================================================

// Align `offset` up to `alignment`
inline uint64_t align_up(uint64_t offset, uint64_t alignment) { return (offset + alignment - 1) & ~(alignment - 1); }

// Compute block offsets and total size for given index parameters.
// Fills sb.blocks[] and sb.total_bytes.
void compute_layout(SuperBlockV1 &sb);

// ============================================================
// Checksum helpers (FNV-1a 64-bit)
// ============================================================

inline uint64_t fnv1a64_update(uint64_t state, const void *data, size_t bytes)
{
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; i++) {
        state ^= static_cast<uint64_t>(p[i]);
        state *= FNV1A64_PRIME;
    }
    return state;
}

inline uint64_t compute_header_crc64(const SuperBlockV1 &sb)
{
    SuperBlockV1 tmp = sb;
    tmp.header_crc64 = 0;
    tmp.payload_crc64 = 0; // header CRC must not depend on payload CRC
    return fnv1a64_update(FNV1A64_OFFSET, &tmp, sizeof(SuperBlockV1));
}

inline uint64_t compute_payload_crc64(const void *gd_base, const SuperBlockV1 &sb)
{
    uint64_t state = FNV1A64_OFFSET;
    for (uint32_t i = 0; i < BLK_COUNT; i++) {
        const auto &blk = sb.blocks[i];
        const void *ptr = static_cast<const char *>(gd_base) + blk.off;
        state = fnv1a64_update(state, ptr, static_cast<size_t>(blk.bytes));
    }
    return state;
}

} // namespace gd_hnsw
