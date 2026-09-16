/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */

// ubs_mem_stub.cpp — file-backed ubs-mem SDK stub for the deploy e2e tests.
//
// Rationale: test_deploy_e2e must also run on machines where the ubs-mem
// engine is unavailable (daemon dead / engine wedged). There the real SDK
// blocks forever inside ubsmem_shmem_allocate_with_provider(), and its
// library constructor additionally calls MPI_Init(), which aborts under
// mpirun (double init). This stub implements exactly the API surface used
// by gd_hnsw_api.cpp on top of POSIX shared memory (shm_open + ftruncate +
// mmap), which is sufficient for the cross-process (mpirun -np 2) flows:
//
//   ubsmem_init_attributes / ubsmem_initialize / ubsmem_finalize
//   ubsmem_set_logger_level
//   ubsmem_shmem_allocate_with_provider / ubsmem_shmem_map
//   ubsmem_shmem_unmap / ubsmem_shmem_deallocate
//
// Semantics kept compatible with the real API where the deploy layer
// depends on them:
//   - gd object names carry no leading '/' (POSIX shm requires one, so it
//     is prepended here);
//   - map() with offset != 0 or a fixed addr is not used by the deploy
//     layer and is rejected;
//   - deallocate on a missing object reports success (the deploy retry
//     helper treats UBSM_ERR_NOT_FOUND as ok as well);
//   - allocate on an existing name reports UBSM_ERR_ALREADY_EXIST (the
//     deploy layer always deallocate-retries names before allocating).

#include <ubs_mem.h>
#include <ubs_mem_def.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <string>

namespace {

// ubs-mem object names have no leading slash; POSIX shm names must start
// with one (and contain no further slashes — gd names only use '_').
std::string to_posix_shm_name(const char *name)
{
    std::string s = name ? name : "";
    if (!s.empty() && s.rfind('/', 0) != 0)
        s.insert(s.begin(), '/');
    return s;
}

// ---- test failure injection -------------------------------------------
// The stub always succeeds, which leaves the deploy layer's ubs-mem error
// branches (initialize/load/finalize failures) unreachable. These knobs let
// the failure-injection tests (Deploy.StubFail* / Deploy.StubDealloc*) flip
// individual API calls to an error return. Injection is process-wide and
// must be reset via ubs_stub_reset() before the next phase of a test.
//   fail_init_attributes / fail_initialize / fail_allocate / fail_map:
//       non-zero → that API returns the value verbatim.
//   dealloc_inuse_times: deallocate returns UBSM_ERR_IN_USING the first N
//       calls (exercises ubsmem_deallocate_retry's sleep/retry loop).
//   fail_deallocate: non-zero → deallocate returns the value verbatim.
int g_fail_init_attributes = 0;
int g_fail_initialize = 0;
int g_fail_allocate = 0;
int g_fail_map = 0;
int g_dealloc_inuse_times = 0;
int g_fail_deallocate = 0;

} // namespace

// Stub-only test controls (declared in test_deploy_e2e.cpp behind
// GD_HNSW_USE_UBSMEM_STUB; not present when linking the real SDK).
extern "C" void ubs_stub_inject(int fail_init_attributes, int fail_initialize, int fail_allocate, int fail_map,
                                int dealloc_inuse_times, int fail_deallocate)
{
    g_fail_init_attributes = fail_init_attributes;
    g_fail_initialize = fail_initialize;
    g_fail_allocate = fail_allocate;
    g_fail_map = fail_map;
    g_dealloc_inuse_times = dealloc_inuse_times;
    g_fail_deallocate = fail_deallocate;
}

extern "C" void ubs_stub_reset(void) { ubs_stub_inject(0, 0, 0, 0, 0, 0); }

// ---- lifecycle ----

int ubsmem_init_attributes(ubsmem_options_t *opts)
{
    if (g_fail_init_attributes)
        return g_fail_init_attributes;
    if (opts == nullptr)
        return UBSM_ERR_PARAM_INVALID;
    std::memset(opts, 0, sizeof(*opts));
    return UBSM_OK;
}

int ubsmem_initialize(const ubsmem_options_t *)
{
    if (g_fail_initialize)
        return g_fail_initialize;
    return UBSM_OK;
}

int ubsmem_finalize(void) { return UBSM_OK; }

int ubsmem_set_logger_level(int) { return UBSM_OK; }

// ---- named shared-memory objects ----

int ubsmem_shmem_allocate_with_provider(const ubs_mem_provider_t *, const char *name, size_t size, mode_t mode,
                                        uint64_t)
{
    if (g_fail_allocate)
        return g_fail_allocate;
    if (name == nullptr || name[0] == '\0' || size == 0)
        return UBSM_ERR_PARAM_INVALID;
    const std::string sname = to_posix_shm_name(name);
    const int fd = shm_open(sname.c_str(), O_CREAT | O_EXCL | O_RDWR, mode);
    if (fd < 0)
        return (errno == EEXIST) ? UBSM_ERR_ALREADY_EXIST : UBSM_ERR_MEMORY;
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        close(fd);
        shm_unlink(sname.c_str());
        return UBSM_ERR_MEMORY;
    }
    close(fd);
    return UBSM_OK;
}

int ubsmem_shmem_map(void *addr, size_t length, int prot, int, const char *name, off_t offset, void **local_ptr)
{
    if (g_fail_map)
        return g_fail_map;
    if (name == nullptr || name[0] == '\0' || local_ptr == nullptr || length == 0 || addr != nullptr || offset != 0)
        return UBSM_ERR_PARAM_INVALID;
    const std::string sname = to_posix_shm_name(name);
    const int oflag = (prot & PROT_WRITE) ? O_RDWR : O_RDONLY;
    const int fd = shm_open(sname.c_str(), oflag, 0);
    if (fd < 0)
        return (errno == ENOENT) ? UBSM_ERR_NOT_FOUND : UBSM_ERR_MEMORY;
    void *p = mmap(nullptr, length, prot, MAP_SHARED, fd, 0);
    const int saved = errno;
    close(fd);
    if (p == MAP_FAILED) {
        errno = saved;
        return UBSM_ERR_MEMORY;
    }
    *local_ptr = p;
    return UBSM_OK;
}

int ubsmem_shmem_unmap(void *local_ptr, size_t length)
{
    if (local_ptr == nullptr || length == 0)
        return UBSM_ERR_PARAM_INVALID;
    return (munmap(local_ptr, length) == 0) ? UBSM_OK : UBSM_ERR_MEMORY;
}

int ubsmem_shmem_deallocate(const char *name)
{
    if (g_dealloc_inuse_times > 0) {
        g_dealloc_inuse_times--;
        return UBSM_ERR_IN_USING;
    }
    if (g_fail_deallocate)
        return g_fail_deallocate;
    if (name == nullptr || name[0] == '\0')
        return UBSM_ERR_PARAM_INVALID;
    const std::string sname = to_posix_shm_name(name);
    if (shm_unlink(sname.c_str()) == 0 || errno == ENOENT)
        return UBSM_OK; // already gone — deletion is idempotent
    return UBSM_ERR_MEMORY;
}
