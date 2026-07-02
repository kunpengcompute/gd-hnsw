#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "ann_dataset.h"  // reuse AnnDataset

namespace shm_hnsw {

// ============================================================
// Binary dataset loader for ANN benchmark format
// Supports the common .bin layout used by bigann, DEEP, SPACEV, etc.
//
// File format (per file):
//   [uint32_t num_vectors][uint32_t dim] followed by
//   num_vectors * dim * sizeof(element) raw data
//
// Expected files in a directory:
//   base.bin  — float32[N, dim]  (base vectors)
//   query.bin — float32[nq, dim] (query vectors)
//   gt.bin    — int32[nq, k]     (ground truth neighbor ids)
// ============================================================

namespace detail {

struct MmapGuard {
    void* ptr = MAP_FAILED;
    size_t len = 0;
    int fd = -1;

    MmapGuard() = default;
    MmapGuard(const MmapGuard&) = delete;
    MmapGuard& operator=(const MmapGuard&) = delete;

    ~MmapGuard() {
        if (ptr != MAP_FAILED) munmap(ptr, len);
        if (fd >= 0) close(fd);
    }

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0) return false;
        len = static_cast<size_t>(st.st_size);
        ptr = mmap(nullptr, len, PROT_READ, MAP_PRIVATE
#ifdef MAP_POPULATE
                   | MAP_POPULATE
#endif
                   , fd, 0);
        return ptr != MAP_FAILED;
    }
};

inline void read_bin_header(const void* data, uint32_t& n, uint32_t& d) {
    std::memcpy(&n, data, sizeof(uint32_t));
    std::memcpy(&d, static_cast<const char*>(data) + sizeof(uint32_t), sizeof(uint32_t));
}

inline std::string resolve_bin_path(const std::string& dir_or_file,
                                    const std::string& filename) {
    // If dir_or_file is a directory, append filename
    struct stat st;
    if (stat(dir_or_file.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        std::string path = dir_or_file;
        if (!path.empty() && path.back() != '/') path += '/';
        return path + filename;
    }
    // If it's a file (e.g. base.bin), derive sibling path
    size_t slash = dir_or_file.rfind('/');
    std::string dir = (slash != std::string::npos)
                      ? dir_or_file.substr(0, slash + 1) : "./";
    return dir + filename;
}

} // namespace detail

// Load a binary-format ANN dataset.
// path: either a directory containing {base.bin, query.bin, gt.bin},
//       or the path to base.bin (siblings resolved from same directory).
// eval_only: if true, only read base.bin header (skip payload) to get n_train/dim.
// override_base/query/gt: if non-empty, use these paths instead of auto-resolved ones.
inline AnnDataset load_bin_dataset(const std::string& path,
                                   bool eval_only = false,
                                   const std::string& override_base = "",
                                   const std::string& override_query = "",
                                   const std::string& override_gt = "",
                                   bool train_only = false) {
    AnnDataset ds;

    std::string base_path  = override_base.empty()  ? detail::resolve_bin_path(path, "base.bin")  : override_base;
    std::string query_path = override_query.empty() ? detail::resolve_bin_path(path, "query.bin") : override_query;
    std::string gt_path    = override_gt.empty()    ? detail::resolve_bin_path(path, "gt.bin")    : override_gt;

    // --- Load base vectors ---
    {
        if (eval_only) {
            // Only read 8-byte header to get n_train/dim — no mmap, no I/O on payload
            int fd = ::open(base_path.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("Cannot open base file: " + base_path);
            }
            struct stat st;
            if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("Cannot stat base file: " + base_path); }
            size_t file_len = static_cast<size_t>(st.st_size);
            if (file_len < 8) { close(fd); throw std::runtime_error("base.bin too small (< 8 bytes): " + base_path); }

            uint8_t hdr[8];
            if (pread(fd, hdr, 8, 0) != 8) { close(fd); throw std::runtime_error("Cannot read base.bin header: " + base_path); }
            close(fd);

            uint32_t n, dim;
            std::memcpy(&n, hdr, sizeof(uint32_t));
            std::memcpy(&dim, hdr + sizeof(uint32_t), sizeof(uint32_t));

            if (n == 0 || dim == 0 || dim > 10000) {
                throw std::runtime_error("base.bin invalid header: n=" +
                    std::to_string(n) + " dim=" + std::to_string(dim));
            }
            size_t expected = 8ULL + static_cast<size_t>(n) * dim * sizeof(float);
            if (file_len != expected) {
                throw std::runtime_error(
                    "base.bin size mismatch: expected " + std::to_string(expected) +
                    " bytes, got " + std::to_string(file_len));
            }
            ds.n_train = n;
            ds.dim = dim;
        } else {
            detail::MmapGuard mm;
            if (!mm.open(base_path)) {
                throw std::runtime_error("Cannot open base file: " + base_path);
            }
            if (mm.len < 8) {
                throw std::runtime_error("base.bin too small (< 8 bytes): " + base_path);
            }

            uint32_t n, dim;
            detail::read_bin_header(mm.ptr, n, dim);

            if (n == 0 || dim == 0 || dim > 10000) {
                throw std::runtime_error("base.bin invalid header: n=" +
                    std::to_string(n) + " dim=" + std::to_string(dim));
            }

            size_t expected = 8ULL + static_cast<size_t>(n) * dim * sizeof(float);
            if (mm.len != expected) {
                throw std::runtime_error(
                    "base.bin size mismatch: expected " + std::to_string(expected) +
                    " bytes, got " + std::to_string(mm.len));
            }

            ds.n_train = n;
            ds.dim = dim;
            ds.train.resize(static_cast<size_t>(n) * dim);
            std::memcpy(ds.train.data(),
                        static_cast<const char*>(mm.ptr) + 8,
                        static_cast<size_t>(n) * dim * sizeof(float));
        }
    }

    // train_only mode: skip query/gt loading (for standalone index building)
    if (train_only) {
        return ds;
    }

    // --- Load query vectors ---
    {
        detail::MmapGuard mm;
        if (!mm.open(query_path)) {
            throw std::runtime_error("Cannot open query file: " + query_path);
        }
        if (mm.len < 8) {
            throw std::runtime_error("query.bin too small: " + query_path);
        }

        uint32_t nq, dim;
        detail::read_bin_header(mm.ptr, nq, dim);

        if (nq == 0 || dim == 0 || dim > 10000) {
            throw std::runtime_error("query.bin invalid header: nq=" +
                std::to_string(nq) + " dim=" + std::to_string(dim));
        }
        if (dim != ds.dim) {
            throw std::runtime_error("query.bin dim (" + std::to_string(dim) +
                ") != base dim (" + std::to_string(ds.dim) + ")");
        }

        size_t expected = 8ULL + static_cast<size_t>(nq) * dim * sizeof(float);
        if (mm.len != expected) {
            throw std::runtime_error(
                "query.bin size mismatch: expected " + std::to_string(expected) +
                " bytes, got " + std::to_string(mm.len));
        }

        ds.n_test = nq;
        ds.test.resize(static_cast<size_t>(nq) * dim);
        std::memcpy(ds.test.data(),
                    static_cast<const char*>(mm.ptr) + 8,
                    static_cast<size_t>(nq) * dim * sizeof(float));
    }

    // --- Load ground truth ---
    {
        detail::MmapGuard mm;
        if (!mm.open(gt_path)) {
            throw std::runtime_error("Cannot open ground truth file: " + gt_path);
        }
        if (mm.len < 8) {
            throw std::runtime_error("gt.bin too small: " + gt_path);
        }

        uint32_t nq, k;
        detail::read_bin_header(mm.ptr, nq, k);

        if (nq == 0 || k == 0) {
            throw std::runtime_error("gt.bin invalid header: nq=" +
                std::to_string(nq) + " k=" + std::to_string(k));
        }
        if (nq != ds.n_test) {
            throw std::runtime_error("gt.bin nq (" + std::to_string(nq) +
                ") != query nq (" + std::to_string(ds.n_test) + ")");
        }

        size_t expected = 8ULL + static_cast<size_t>(nq) * k * sizeof(int32_t);
        if (mm.len != expected) {
            throw std::runtime_error(
                "gt.bin size mismatch: expected " + std::to_string(expected) +
                " bytes, got " + std::to_string(mm.len));
        }

        ds.k_gt = k;
        ds.neighbors.resize(static_cast<size_t>(nq) * k);
        std::memcpy(ds.neighbors.data(),
                    static_cast<const char*>(mm.ptr) + 8,
                    static_cast<size_t>(nq) * k * sizeof(int32_t));

        // distances not available in .bin format — fill zeros
        ds.distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    }

    return ds;
}

// Detect dataset format from path
enum class DatasetFormat { HDF5, BIN, UNKNOWN };

inline DatasetFormat detect_dataset_format(const std::string& path) {
    // Check file extension for HDF5
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = path.substr(dot);
        if (ext == ".h5" || ext == ".hdf5") return DatasetFormat::HDF5;
        if (ext == ".bin") return DatasetFormat::BIN;
    }

    // Check if it's a directory containing base.bin
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        std::string base_check = path;
        if (!base_check.empty() && base_check.back() != '/') base_check += '/';
        base_check += "base.bin";
        if (stat(base_check.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            return DatasetFormat::BIN;
        }
    }

    return DatasetFormat::UNKNOWN;
}

} // namespace shm_hnsw
