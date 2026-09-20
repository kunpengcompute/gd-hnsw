/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

#include "gd_layout.h"
#include <cstdio>
#include <cinttypes>

namespace gd_hnsw {

// ============================================================
// compute_layout — fill block offsets given index parameters
// ============================================================

void compute_layout(SuperBlockV1 &sb)
{
    // Use ntotal_local for block sizes if sharded, otherwise ntotal
    uint64_t n = (sb.num_shards > 0) ? sb.ntotal_local : sb.ntotal;

    uint64_t offset = SUPERBLOCK_SIZE; // first block starts after header

    // BLK_VECTORS: float[n * dim], 2MB aligned
    sb.blocks[BLK_VECTORS].off = align_up(offset, HUGEPAGE_ALIGN);
    sb.blocks[BLK_VECTORS].bytes = n * sb.dim * sizeof(float);
    sb.blocks[BLK_VECTORS].alignment = HUGEPAGE_ALIGN;
    offset = sb.blocks[BLK_VECTORS].off + sb.blocks[BLK_VECTORS].bytes;

    // BLK_LEVELS: int32_t[n], 4KB aligned
    sb.blocks[BLK_LEVELS].off = align_up(offset, PAGE_ALIGN);
    sb.blocks[BLK_LEVELS].bytes = n * sizeof(int32_t);
    sb.blocks[BLK_LEVELS].alignment = PAGE_ALIGN;
    offset = sb.blocks[BLK_LEVELS].off + sb.blocks[BLK_LEVELS].bytes;

    // BLK_OFFSETS: uint64_t[n + 1], 4KB aligned
    sb.blocks[BLK_OFFSETS].off = align_up(offset, PAGE_ALIGN);
    sb.blocks[BLK_OFFSETS].bytes = (n + 1) * sizeof(uint64_t);
    sb.blocks[BLK_OFFSETS].alignment = PAGE_ALIGN;
    offset = sb.blocks[BLK_OFFSETS].off + sb.blocks[BLK_OFFSETS].bytes;

    // BLK_NEIGHBORS: int32_t[neighbors_size], 2MB aligned
    sb.blocks[BLK_NEIGHBORS].off = align_up(offset, HUGEPAGE_ALIGN);
    sb.blocks[BLK_NEIGHBORS].bytes = sb.neighbors_size * sizeof(int32_t);
    sb.blocks[BLK_NEIGHBORS].alignment = HUGEPAGE_ALIGN;
    offset = sb.blocks[BLK_NEIGHBORS].off + sb.blocks[BLK_NEIGHBORS].bytes;

    // BLK_CUM_NNEIGHBOR: int32_t[num_levels], 4KB aligned
    sb.blocks[BLK_CUM_NNEIGHBOR].off = align_up(offset, PAGE_ALIGN);
    sb.blocks[BLK_CUM_NNEIGHBOR].bytes = sb.num_levels * sizeof(int32_t);
    sb.blocks[BLK_CUM_NNEIGHBOR].alignment = PAGE_ALIGN;
    offset = sb.blocks[BLK_CUM_NNEIGHBOR].off + sb.blocks[BLK_CUM_NNEIGHBOR].bytes;

    sb.total_bytes = align_up(offset, PAGE_ALIGN);
}

// ============================================================
// validate_superblock
// ============================================================

ValidationResult validate_superblock(const void *gd_base, uint64_t gd_size)
{
    ValidationResult r;
    auto fail = [&](const char *msg) -> ValidationResult & {
        r.ok = false;
        r.error = msg;
        return r;
    };

    if (gd_size < SUPERBLOCK_SIZE) {
        return fail("gd region smaller than SuperBlock");
    }

    const auto &sb = *static_cast<const SuperBlockV1 *>(gd_base);

    // Magic
    if (sb.magic != GD_MAGIC) {
        return fail("bad magic number");
    }

    // Schema version
    if (sb.schema_version != SCHEMA_VERSION) {
        return fail("unsupported schema version");
    }

    // State
    if (sb.state > static_cast<uint32_t>(GdState::ACTIVE)) {
        return fail("invalid state");
    }

    // Endianness
    if (sb.endianness != ENDIAN_LITTLE) {
        return fail("only little-endian supported");
    }

    // Metric
    if (sb.metric_type != METRIC_L2) {
        return fail("only L2 metric supported in V1");
    }

    // Basic bounds
    if (sb.ntotal == 0) {
        return fail("ntotal is zero");
    }
    if (sb.ntotal > INT32_MAX) {
        return fail("ntotal exceeds INT32_MAX");
    }

    // Shard validation
    uint64_t n_local = sb.ntotal; // default: unsharded
    if (sb.num_shards > 0) {
        if (sb.shard_id >= sb.num_shards) {
            return fail("shard_id >= num_shards");
        }
        if (sb.ntotal_local == 0) {
            return fail("ntotal_local is zero");
        }
        if (sb.global_id_begin + sb.ntotal_local > sb.ntotal) {
            return fail("shard range exceeds ntotal");
        }
        n_local = sb.ntotal_local;
    }

    if (sb.dim == 0 || sb.dim > 65536) {
        return fail("dim out of range");
    }
    if (sb.M == 0 || sb.M > 256) {
        return fail("M out of range");
    }
    if (sb.num_levels < 2) {
        return fail("num_levels must be >= 2");
    }
    if (sb.entry_point < 0 || static_cast<uint64_t>(sb.entry_point) >= sb.ntotal) {
        return fail("entry_point out of range");
    }
    if (sb.max_level < 0 || static_cast<uint64_t>(sb.max_level) + 1 >= static_cast<uint64_t>(sb.num_levels)) {
        return fail("max_level out of range (need max_level+1 < num_levels)");
    }

    // Total size
    if (sb.total_bytes > gd_size) {
        return fail("total_bytes exceeds gd region size");
    }

    // Block bounds and alignment checks
    for (uint32_t i = 0; i < BLK_COUNT; i++) {
        const auto &blk = sb.blocks[i];
        if (blk.off > sb.total_bytes || blk.bytes > sb.total_bytes - blk.off) {
            return fail("block exceeds total_bytes");
        }
        if (blk.alignment > 0 && (blk.off % blk.alignment) != 0) {
            return fail("block alignment violated");
        }
    }

    // Block size consistency
    if (sb.blocks[BLK_VECTORS].bytes != n_local * sb.dim * sizeof(float)) {
        return fail("vectors block size mismatch");
    }
    if (sb.blocks[BLK_LEVELS].bytes != n_local * sizeof(int32_t)) {
        return fail("levels block size mismatch");
    }
    if (sb.blocks[BLK_OFFSETS].bytes != (n_local + 1) * sizeof(uint64_t)) {
        return fail("offsets block size mismatch");
    }
    if (sb.blocks[BLK_NEIGHBORS].bytes != sb.neighbors_size * sizeof(int32_t)) {
        return fail("neighbors block size mismatch");
    }
    if (sb.blocks[BLK_CUM_NNEIGHBOR].bytes != sb.num_levels * sizeof(int32_t)) {
        return fail("cum_nneighbor block size mismatch");
    }

    // Integrity checksum validation for sealed/active payloads.
    if (sb.state >= static_cast<uint32_t>(GdState::SEALED)) {
        if (sb.header_crc64 != compute_header_crc64(sb)) {
            return fail("header_crc64 mismatch");
        }
        if (sb.payload_crc64 != compute_payload_crc64(gd_base, sb)) {
            return fail("payload_crc64 mismatch");
        }
    }

    // Only do deep structural validation if state >= SEALED.
    if (sb.state >= static_cast<uint32_t>(GdState::SEALED)) {
        const auto *offsets = offsets_ptr(gd_base, sb);
        const auto *levels = levels_ptr(gd_base, sb);
        const auto *cum_nn = cum_nneighbor_ptr(gd_base, sb);

        // offsets must be monotonically non-decreasing
        for (uint64_t i = 0; i < n_local; i++) {
            if (offsets[i + 1] < offsets[i]) {
                return fail("offsets not monotonic");
            }
        }

        // offsets[n_local] == neighbors_size
        if (offsets[n_local] != sb.neighbors_size) {
            return fail("offsets[n_local] != neighbors_size");
        }

        // per-node: offsets[i+1] - offsets[i] == cum_nneighbor[levels[i]]
        for (uint64_t i = 0; i < n_local; i++) {
            int32_t lvl = levels[i];
            if (lvl <= 0 || lvl >= static_cast<int32_t>(sb.num_levels)) {
                return fail("invalid level for node");
            }
            uint64_t expected_slots = static_cast<uint64_t>(cum_nn[lvl]);
            uint64_t actual_slots = offsets[i + 1] - offsets[i];
            if (actual_slots != expected_slots) {
                return fail("offsets slot count mismatch for node");
            }
        }

        // entry_point must sit at max_level (the top of the navigable
        // graph). With multiple shards the entry point may live on another
        // shard — only the owning shard can verify this. int64 arithmetic
        // sidesteps the max_level+1 signed overflow at INT32_MAX.
        uint64_t gid_begin = sb.num_shards > 0 ? sb.global_id_begin : 0;
        if (static_cast<uint64_t>(sb.entry_point) >= gid_begin &&
            static_cast<uint64_t>(sb.entry_point) < gid_begin + n_local) {
            uint64_t ep_local = static_cast<uint64_t>(sb.entry_point) - gid_begin;
            if (static_cast<int64_t>(levels[ep_local]) != static_cast<int64_t>(sb.max_level) + 1) {
                return fail("entry_point is not at max_level");
            }
        }

        // neighbor ids in [-1, ntotal) — note: global ids, not local
        const auto *neighbors = neighbors_ptr(gd_base, sb);
        for (uint64_t i = 0; i < sb.neighbors_size; i++) {
            int32_t nid = neighbors[i];
            if (nid != EMPTY_NEIGHBOR && (nid < 0 || static_cast<uint64_t>(nid) >= sb.ntotal)) {
                return fail("neighbor id out of range");
            }
        }
    }

    return r;
}

} // namespace gd_hnsw
