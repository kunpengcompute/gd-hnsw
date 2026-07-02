#pragma once
// ============================================================
// ls64_copy.h — NEON LDP/STP based bulk copy for AArch64
//
// Provides high-throughput copy primitives using NEON 128-bit
// load/store pairs (32 bytes per LDP/STP). Each iteration of
// the main loop copies 64 bytes (2x LDP + 2x STP).
//
// For cross-NUMA SHM channels where data won't be re-read locally,
// non-temporal variants (LDNP/STNP) are used to avoid cache pollution.
//
// Non-AArch64: compile-time error (ARM64-only project).
// ============================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <arm_neon.h>

// ============================================================
// Phase 1: SHM → local (read from remote shared memory)
// Uses LDNP (non-temporal load, no cache alloc) + STP (normal store
// to local buffer that will be iterated immediately).
// ============================================================

inline void ls64_copy_from_shm(void* __restrict dst, const void* __restrict src, size_t bytes) {
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(dst);

    // Main loop: 64 bytes per iteration (2x LDNP + 2x STP)
    while (bytes >= 64) {
        __asm__ __volatile__(
            "ldnp q0, q1, [%[s]]       \n"
            "ldnp q2, q3, [%[s], #32]  \n"
            "stp  q0, q1, [%[d]]       \n"
            "stp  q2, q3, [%[d], #32]  \n"
            : : [s]"r"(s), [d]"r"(d)
            : "v0", "v1", "v2", "v3", "memory"
        );
        s += 64; d += 64; bytes -= 64;
    }
    // Tail: 16 bytes at a time
    while (bytes >= 16) {
        __asm__ __volatile__(
            "ldr q0, [%[s]]  \n"
            "str q0, [%[d]]  \n"
            : : [s]"r"(s), [d]"r"(d) : "v0", "memory"
        );
        s += 16; d += 16; bytes -= 16;
    }
    if (bytes > 0) {
        std::memcpy(d, s, bytes);
    }
}

// ============================================================
// Phase 2 & 4: local → SHM (write to remote shared memory)
// Uses LDP (normal load from local) + STNP (non-temporal store,
// bypasses cache — avoids RFO on remote NUMA node).
// ============================================================

inline void ls64_copy_to_shm(void* __restrict dst, const void* __restrict src, size_t bytes) {
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(dst);

    // Main loop: 64 bytes per iteration (2x LDP + 2x STNP)
    while (bytes >= 64) {
        __asm__ __volatile__(
            "ldp  q0, q1, [%[s]]       \n"
            "ldp  q2, q3, [%[s], #32]  \n"
            "stnp q0, q1, [%[d]]       \n"
            "stnp q2, q3, [%[d], #32]  \n"
            : : [s]"r"(s), [d]"r"(d)
            : "v0", "v1", "v2", "v3", "memory"
        );
        s += 64; d += 64; bytes -= 64;
    }
    // Tail: 16 bytes at a time
    while (bytes >= 16) {
        __asm__ __volatile__(
            "ldr q0, [%[s]]  \n"
            "str q0, [%[d]]  \n"
            : : [s]"r"(s), [d]"r"(d) : "v0", "memory"
        );
        s += 16; d += 16; bytes -= 16;
    }
    if (bytes > 0) {
        std::memcpy(d, s, bytes);
    }
}
