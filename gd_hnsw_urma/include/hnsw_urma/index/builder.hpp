#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include "hnsw_urma/core/status.hpp"

namespace hnsw_urma {

struct ReferenceBuildOptions {
  std::uint32_t num_shards{1};
  std::uint32_t M{16};
  std::uint32_t M0{32};
  std::uint32_t ef_search{64};
  std::uint32_t metric{0}; // 0=L2, 1=negative inner product
};

// Deterministic O(N^2) builder used for fixtures and local correctness. Large
// production indexes should use the Faiss extractor adapter; this path refuses
// inputs above max_rows unless the caller explicitly raises the bound.
[[nodiscard]] Status BuildReferenceIndex(const std::filesystem::path &root,
                                         std::span<const float> vectors,
                                         std::uint64_t rows, std::uint32_t dim,
                                         const ReferenceBuildOptions &options,
                                         std::uint64_t max_rows = 100000);

} // namespace hnsw_urma
