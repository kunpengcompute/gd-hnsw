#include "index_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace shm_hnsw {

// ============================================================
// Path helpers
// ============================================================

std::string index_file_for_shard(const std::string& base_path, uint32_t shard_id) {
    return base_path + ".shard_" + std::to_string(shard_id);
}

std::string idmap_file_for_index(const std::string& base_path) {
    return base_path + ".idmap";
}

std::string centroids_file_for_index(const std::string& base_path) {
    return base_path + ".centroids";
}

// ============================================================
// IdMap file I/O
// ============================================================

bool save_idmap_file(const std::string& base_path,
                     const std::vector<uint32_t>& idmap) {
    std::string path = idmap_file_for_index(base_path);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for writing: %s\n",
                path.c_str(), strerror(errno));
        return false;
    }

    IdMapFileHeader hdr{};
    hdr.magic = IDMAP_MAGIC;
    hdr.version = IDMAP_VERSION;
    hdr.count = static_cast<uint64_t>(idmap.size());

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to write idmap header '%s'\n", path.c_str());
        fclose(fp);
        return false;
    }

    if (!idmap.empty()) {
        size_t written = fwrite(idmap.data(), sizeof(uint32_t), idmap.size(), fp);
        if (written != idmap.size()) {
            fprintf(stderr, "Error: short write to '%s': %zu / %zu entries\n",
                    path.c_str(), written, idmap.size());
            fclose(fp);
            return false;
        }
    }

    fclose(fp);
    printf("  Saved idmap: %s (%zu entries)\n", path.c_str(), idmap.size());
    return true;
}

IdMapLoadStatus load_idmap_file(const std::string& base_path,
                                std::vector<uint32_t>& idmap_out) {
    std::string path = idmap_file_for_index(base_path);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        if (errno == ENOENT) {
            idmap_out.clear();
            return IdMapLoadStatus::NotFound;
        }
        fprintf(stderr, "Error: cannot open '%s' for reading: %s\n",
                path.c_str(), strerror(errno));
        return IdMapLoadStatus::Error;
    }

    IdMapFileHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to read idmap header '%s'\n", path.c_str());
        fclose(fp);
        return IdMapLoadStatus::Error;
    }
    if (hdr.magic != IDMAP_MAGIC || hdr.version != IDMAP_VERSION) {
        fprintf(stderr, "Error: invalid idmap header '%s' (magic/version mismatch)\n",
                path.c_str());
        fclose(fp);
        return IdMapLoadStatus::Error;
    }
    if (hdr.count > static_cast<uint64_t>(INT32_MAX)) {
        fprintf(stderr, "Error: idmap '%s' too large: %lu entries\n",
                path.c_str(), static_cast<unsigned long>(hdr.count));
        fclose(fp);
        return IdMapLoadStatus::Error;
    }

    idmap_out.resize(static_cast<size_t>(hdr.count));
    if (!idmap_out.empty()) {
        size_t nread = fread(idmap_out.data(), sizeof(uint32_t), idmap_out.size(), fp);
        if (nread != idmap_out.size()) {
            fprintf(stderr, "Error: short read from '%s': %zu / %zu entries\n",
                    path.c_str(), nread, idmap_out.size());
            fclose(fp);
            return IdMapLoadStatus::Error;
        }
    }

    fclose(fp);
    printf("  Loaded idmap: %s (%zu entries)\n", path.c_str(), idmap_out.size());
    return IdMapLoadStatus::Loaded;
}

// ============================================================
// Centroids file I/O
// ============================================================

bool save_centroids_file(const std::string& base_path,
                         const std::vector<float>& centroids,
                         uint32_t num_shards, uint32_t dim) {
    std::string path = centroids_file_for_index(base_path);
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for writing: %s\n",
                path.c_str(), strerror(errno));
        return false;
    }

    CentroidsFileHeader hdr{};
    hdr.magic = CENTROIDS_MAGIC;
    hdr.version = CENTROIDS_VERSION;
    hdr.dim = dim;
    hdr.num_shards = num_shards;

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to write centroids header '%s'\n", path.c_str());
        fclose(fp);
        return false;
    }

    size_t count = static_cast<size_t>(num_shards) * dim;
    if (fwrite(centroids.data(), sizeof(float), count, fp) != count) {
        fprintf(stderr, "Error: short write to '%s'\n", path.c_str());
        fclose(fp);
        return false;
    }

    fclose(fp);
    printf("  Saved centroids: %s (%u shards x %u dim)\n", path.c_str(), num_shards, dim);
    return true;
}

IdMapLoadStatus load_centroids_file(const std::string& base_path,
                                    std::vector<float>& centroids_out,
                                    uint32_t& num_shards_out,
                                    uint32_t& dim_out) {
    std::string path = centroids_file_for_index(base_path);
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        if (errno == ENOENT) return IdMapLoadStatus::NotFound;
        fprintf(stderr, "Error: cannot open '%s': %s\n", path.c_str(), strerror(errno));
        return IdMapLoadStatus::Error;
    }

    CentroidsFileHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to read centroids header '%s'\n", path.c_str());
        fclose(fp);
        return IdMapLoadStatus::Error;
    }
    if (hdr.magic != CENTROIDS_MAGIC || hdr.version != CENTROIDS_VERSION) {
        fprintf(stderr, "Error: invalid centroids file '%s'\n", path.c_str());
        fclose(fp);
        return IdMapLoadStatus::Error;
    }

    num_shards_out = hdr.num_shards;
    dim_out = hdr.dim;
    size_t count = static_cast<size_t>(hdr.num_shards) * hdr.dim;
    centroids_out.resize(count);
    if (fread(centroids_out.data(), sizeof(float), count, fp) != count) {
        fprintf(stderr, "Error: short read from '%s'\n", path.c_str());
        fclose(fp);
        return IdMapLoadStatus::Error;
    }

    fclose(fp);
    printf("  Loaded centroids: %s (%u shards x %u dim)\n", path.c_str(), hdr.num_shards, hdr.dim);
    return IdMapLoadStatus::Loaded;
}

// ============================================================
// Shard buffer I/O
// ============================================================

bool save_shard_buffers(const std::string& base_path,
                        const std::vector<std::vector<char>>& shard_bufs) {
    uint32_t num_shards = static_cast<uint32_t>(shard_bufs.size());
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = index_file_for_shard(base_path, s);
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) {
            fprintf(stderr, "Error: cannot open '%s' for writing: %s\n",
                    path.c_str(), strerror(errno));
            return false;
        }
        size_t written = fwrite(shard_bufs[s].data(), 1, shard_bufs[s].size(), fp);
        fclose(fp);
        if (written != shard_bufs[s].size()) {
            fprintf(stderr, "Error: short write to '%s': %zu / %zu bytes\n",
                    path.c_str(), written, shard_bufs[s].size());
            return false;
        }
        printf("  Saved shard %u: %s (%.1f MB)\n",
               s, path.c_str(), shard_bufs[s].size() / (1024.0 * 1024.0));
    }
    return true;
}

bool load_shard_buffers(const std::string& base_path,
                        uint32_t num_shards,
                        std::vector<std::vector<char>>& shard_bufs) {
    shard_bufs.resize(num_shards);
    for (uint32_t s = 0; s < num_shards; s++) {
        std::string path = index_file_for_shard(base_path, s);
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "Error: cannot open '%s' for reading: %s\n",
                    path.c_str(), strerror(errno));
            return false;
        }
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        if (file_size <= 0) {
            fprintf(stderr, "Error: '%s' is empty or unreadable\n", path.c_str());
            fclose(fp);
            return false;
        }
        shard_bufs[s].resize(static_cast<size_t>(file_size));
        size_t nread = fread(shard_bufs[s].data(), 1, static_cast<size_t>(file_size), fp);
        fclose(fp);
        if (nread != static_cast<size_t>(file_size)) {
            fprintf(stderr, "Error: short read from '%s': %zu / %ld bytes\n",
                    path.c_str(), nread, file_size);
            return false;
        }
        // Validate superblock integrity
        auto result = validate_superblock(shard_bufs[s].data(),
                                          static_cast<uint64_t>(file_size));
        if (!result.ok) {
            fprintf(stderr, "Error: '%s' failed validation: %s\n",
                    path.c_str(), result.error.c_str());
            return false;
        }
        printf("  Loaded shard %u: %s (%.1f MB, validated)\n",
               s, path.c_str(), file_size / (1024.0 * 1024.0));
    }

    // Cross-shard consistency validation
    const auto* sb0 = reinterpret_cast<const SuperBlockV1*>(shard_bufs[0].data());
    uint64_t expected_ntotal = sb0->ntotal;
    uint32_t expected_dim = sb0->dim;
    uint32_t expected_M = sb0->M;
    int32_t  expected_max_level = sb0->max_level;
    int32_t  expected_entry_point = sb0->entry_point;

    if (sb0->num_shards != num_shards) {
        fprintf(stderr, "Error: index was built with %u shards, but --shards=%u\n",
                sb0->num_shards, num_shards);
        return false;
    }

    uint64_t sum_ntotal_local = 0;
    for (uint32_t s = 0; s < num_shards; s++) {
        const auto* sb = reinterpret_cast<const SuperBlockV1*>(shard_bufs[s].data());
        if (sb->shard_id != s) {
            fprintf(stderr, "Error: shard file %u has shard_id=%u (expected %u)\n",
                    s, sb->shard_id, s);
            return false;
        }
        if (sb->num_shards != num_shards) {
            fprintf(stderr, "Error: shard %u has num_shards=%u (expected %u)\n",
                    s, sb->num_shards, num_shards);
            return false;
        }
        if (sb->ntotal != expected_ntotal || sb->dim != expected_dim ||
            sb->M != expected_M || sb->max_level != expected_max_level ||
            sb->entry_point != expected_entry_point) {
            fprintf(stderr, "Error: shard %u has inconsistent index parameters "
                    "(ntotal/dim/M/max_level/entry_point mismatch with shard 0)\n", s);
            return false;
        }
        if (sb->global_id_begin != sum_ntotal_local) {
            fprintf(stderr, "Error: shard %u has global_id_begin=%lu, expected %lu\n",
                    s, (unsigned long)sb->global_id_begin,
                    (unsigned long)sum_ntotal_local);
            return false;
        }
        sum_ntotal_local += sb->ntotal_local;
    }
    if (sum_ntotal_local != expected_ntotal) {
        fprintf(stderr, "Error: sum of ntotal_local (%lu) != ntotal (%lu)\n",
                (unsigned long)sum_ntotal_local, (unsigned long)expected_ntotal);
        return false;
    }
    printf("  Cross-shard consistency: OK (%u shards, ntotal=%lu, dim=%u, M=%u)\n",
           num_shards, (unsigned long)expected_ntotal, expected_dim, expected_M);

    return true;
}

} // namespace shm_hnsw
