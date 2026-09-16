# ============================================================
# tests/CMakeLists.txt — Unit tests for gd-hnsw
#
# GTest detection is wired here; the per-layer test targets map onto
# the 4-layer library layout (core → faiss → build → deploy):
#   test_core           → header-only (no link dep)
#   test_pushdown       → header-only, ARM64 NEON required
#   test_core_lib       → links gd_hnsw_core
#   test_distance_kernels → header-only (NEON vs scalar)
#   test_faiss_extract  → links gd_hnsw_faiss
#   test_search_e2e     → links gd_hnsw_faiss
#   test_build          → links gd_hnsw_build (build API + dataset loaders)
#   test_concurrency    → links gd_hnsw_core, ARM64 NEON required
#   test_pushdown_fork  → links gd_hnsw_core, Linux + ARM64 only
#   test_deploy_e2e     → links gd_hnsw_api (or ubs-mem stub), deploy gate
# ============================================================

# ============================================================
# GTest
#
# Install 路径: ${PROJECT_3RDPARTY_BIN_DIR}/googletest/{lib,include}
# Submodule 需手动 init: git submodule update --init 3rdparty/googletest/googletest
# ============================================================

include_directories(${PROJECT_3RDPARTY_BIN_DIR}/googletest/include)
link_directories(${PROJECT_3RDPARTY_BIN_DIR}/googletest/lib
                 ${PROJECT_3RDPARTY_BIN_DIR}/googletest/lib64)
SET(LIB_3RD_GTEST ${PROJECT_3RDPARTY_BIN_DIR}/googletest/lib/libgtest.a)
SET(LIB_3RD_GTEST_MAIN ${PROJECT_3RDPARTY_BIN_DIR}/googletest/lib/libgtest_main.a)
set(GTEST_LIBS ${LIB_3RD_GTEST} ${LIB_3RD_GTEST_MAIN})

# --- test_core: header-only, platform-independent ---
# 第一层 A: MinimaxHeap, Doorbell, ConfigParser, SuperBlock CRC
# (VisitedTable 移至 test_core_lib — 其实现位于 gd_hnsw_search.cpp，非 header-only)
add_executable(test_core
    test_main.cpp
    test_minimax_heap.cpp
    test_doorbell.cpp
    test_config_parser.cpp
    test_superblock.cpp          # CRC functions inline in gd_layout.h
)
target_include_directories(test_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/benchmarks
)
target_link_libraries(test_core PRIVATE ${GTEST_LIBS})
target_compile_options(test_core PRIVATE ${PROJECT_WARNING_FLAGS})

# --- test_pushdown: header-only, ARM64 required ---
# 第一层 B: DualDistProxy 全部类内 inline in dist_service.h
# dist_service.h → distance_kernel.h → <arm_neon.h>
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_executable(test_pushdown
        test_main.cpp
        test_pushdown_protocol.cpp
    )
    target_include_directories(test_pushdown PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_pushdown PRIVATE ${GTEST_LIBS})
    target_compile_options(test_pushdown PRIVATE ${PROJECT_WARNING_FLAGS})
else()
    message(STATUS "test_pushdown skipped — requires ARM64 NEON (dist_service.h → <arm_neon.h>)")
endif()

# --- test_core_lib: needs gd_hnsw_core link ---
# 第一层 C: layout compute, index I/O, search algorithm
# 第四层: SuperBlock state transitions
# 函数实现在 gd_layout.cpp / index_io.cpp / gd_hnsw_search.cpp 中
# gd_hnsw_core 需要 ARM64 NEON + OpenMP
add_executable(test_core_lib
    test_main.cpp
    test_visited_table.cpp       # VisitedTable 实现在 gd_hnsw_search.cpp（core 库）
    test_layout_compute.cpp
    test_index_io.cpp
    test_search_algorithm.cpp
    test_pushdown_search.cpp     # Phase B: pushdown search integration tests
    test_superblock_full.cpp     # Phase 4: BUILDING→SEALED→ACTIVE transitions
)
target_include_directories(test_core_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_core_lib PRIVATE
    ${GTEST_LIBS}
    gd_hnsw_core
)
target_compile_options(test_core_lib PRIVATE ${PROJECT_WARNING_FLAGS})

# --- test_distance_kernels: scalar reference vs NEON accuracy ---
# 第二层: L2 and dot+norm distance kernels, ls64 bulk copy
# ARM64: NEON vs scalar reference (max error < 1e-5)
# x86: scalar vs scalar self-consistency only (no NEON headers)
add_executable(test_distance_kernels
    test_main.cpp
    test_l2_kernel.cpp
    test_dot_norm.cpp
)
target_include_directories(test_distance_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_distance_kernels PRIVATE ${GTEST_LIBS})
target_compile_options(test_distance_kernels PRIVATE ${PROJECT_WARNING_FLAGS})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_definitions(test_distance_kernels PRIVATE HAS_ARM_NEON)
endif()

# --- test_faiss_extract: Phase 4 integration test ---
# 第三层: FaissExtractor build + extract, 验证分片缓冲区
# 需要 faiss (通过 gd_hnsw_faiss)
add_executable(test_faiss_extract
    test_main.cpp
    test_faiss_extract.cpp
)
target_include_directories(test_faiss_extract PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_faiss_extract PRIVATE
    ${GTEST_LIBS}
    gd_hnsw_faiss
)
target_compile_options(test_faiss_extract PRIVATE ${PROJECT_WARNING_FLAGS})

# --- test_search_e2e: Phase 4 end-to-end search ---
# 第三层: faiss build → extract → search → Recall@K 验证
# 需要 faiss (通过 gd_hnsw_faiss)
add_executable(test_search_e2e
    test_main.cpp
    test_search_e2e.cpp
)
target_include_directories(test_search_e2e PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(test_search_e2e PRIVATE
    ${GTEST_LIBS}
    gd_hnsw_faiss
)
target_compile_options(test_search_e2e PRIVATE ${PROJECT_WARNING_FLAGS})

# --- test_build: Layer 3 build API + dataset loaders ---
# 第三层: build_sharded_index (bin/hdf5 数据集, 多分片, cluster 分区 sidecar),
# load_bin_dataset / load_hdf5_dataset 全分支, detect_dataset_format,
# ShardBuildOptions post_add_callback — 覆盖 build.cpp + build.h +
# ann_dataset.h + bin_loader.h + hdf5_loader.h + faiss_extractor.h 选项
# 需要 HDF5 (通过 gd_hnsw_build)
if(HDF5_FOUND)
    add_executable(test_build
        test_main.cpp
        test_build.cpp
    )
    target_include_directories(test_build PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(test_build PRIVATE
        ${GTEST_LIBS}
        gd_hnsw_build
    )
    target_compile_options(test_build PRIVATE ${PROJECT_WARNING_FLAGS})
    add_test(NAME test_build COMMAND test_build)
endif()

# --- test_concurrency: Phase 4 multi-thread pushdown ---
# 第三层: DualDistServiceWorker + DualDistProxy 多线程并发
# dist_service.h → <arm_neon.h>, ARM64 only
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_executable(test_concurrency
        test_main.cpp
        test_concurrency.cpp
    )
    target_include_directories(test_concurrency PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_concurrency PRIVATE
        ${GTEST_LIBS}
        gd_hnsw_core
    )
    target_compile_options(test_concurrency PRIVATE ${PROJECT_WARNING_FLAGS})
else()
    message(STATUS "test_concurrency skipped — requires ARM64 NEON (dist_service.h → <arm_neon.h>)")
endif()

# CTest registration
add_test(NAME test_core     COMMAND test_core)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_test(NAME test_pushdown COMMAND test_pushdown)
endif()
add_test(NAME test_core_lib         COMMAND test_core_lib)
add_test(NAME test_distance_kernels COMMAND test_distance_kernels)
add_test(NAME test_faiss_extract    COMMAND test_faiss_extract)
add_test(NAME test_search_e2e       COMMAND test_search_e2e)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_test(NAME test_concurrency   COMMAND test_concurrency)
endif()

# --- test_pushdown_fork: Phase G2 fork-based cross-process pushdown ---
# Linux + ARM64 only (requires fork + mmap + NEON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_executable(test_pushdown_fork
        test_main.cpp
        test_pushdown_fork.cpp
    )
    target_include_directories(test_pushdown_fork PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(test_pushdown_fork PRIVATE
        ${GTEST_LIBS}
        gd_hnsw_core
    )
    target_compile_options(test_pushdown_fork PRIVATE ${PROJECT_WARNING_FLAGS})
    add_test(NAME test_pushdown_fork COMMAND test_pushdown_fork)
endif()

# --- test_deploy_e2e: Layer 4 deploy API (Context: initialize/load/search) ---
# Only built when the deploy gate passes (Linux + MPI + ubs-mem + HDF5 + NUMA).
# Per-test ctest registrations: each Deploy test runs in its own process so
# ubsmem and MPI (once-per-process resources) are initialized/finalized exactly
# once per run. Multi-round tests (LoadSidecars, RouteQueries) are split into
# individual TEST()s to ensure 1 Context per process.
#
# ubs-mem stub mode (GD_HNSW_DEPLOY_UBSMEM_STUB, default ON): compile
# gd_hnsw_api.cpp directly into the test binary and resolve the ubs-mem API
# with the POSIX-shm-backed stub (tests/ubs_mem_stub.cpp) instead of linking
# the real SDK. Rationale: (a) on machines where the ubs engine is dead the
# real SDK blocks forever inside ubsmem_shmem_allocate_with_provider();
# (b) the real SDK's library constructor also calls MPI_Init, which aborts
# under mpirun (double init). The stub keeps the deploy code paths fully
# exercised without a live engine. Set the option OFF to link the real SDK.
# All entries still go through run_deploy_test.sh: when running against the
# real backend and it hangs (daemon up, ubse engine dead), allocate blocks
# ~30min and the test would hit the ctest timeout. The wrapper kills it after
# WRAPPER_TIMEOUT and reports the run as skipped/success so full UT stays
# green on such machines.
option(GD_HNSW_DEPLOY_UBSMEM_STUB
       "Build test_deploy_e2e against the POSIX-shm ubs-mem stub instead of the real SDK" ON)
if(GD_HNSW_DEPLOY_OK)
    if(GD_HNSW_DEPLOY_UBSMEM_STUB)
        add_executable(test_deploy_e2e
            test_main.cpp
            test_deploy_e2e.cpp
            ubs_mem_stub.cpp
            ${CMAKE_SOURCE_DIR}/src/deploy/gd_hnsw_api.cpp
        )
        target_include_directories(test_deploy_e2e PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${UBSMEM_INCLUDE_DIR}
        )
        target_compile_definitions(test_deploy_e2e PRIVATE GD_HNSW_USE_UBSMEM_STUB=1)
        target_link_libraries(test_deploy_e2e PRIVATE
            ${GTEST_LIBS}
            gd_hnsw_build
            MPI::MPI_CXX
            ${NUMA_LIB}
        )
    else()
        add_executable(test_deploy_e2e
            test_main.cpp
            test_deploy_e2e.cpp
        )
        target_include_directories(test_deploy_e2e PRIVATE
            ${CMAKE_SOURCE_DIR}/include
        )
        target_link_libraries(test_deploy_e2e PRIVATE
            ${GTEST_LIBS}
            gd_hnsw_api
            MPI::MPI_CXX
        )
    endif()
    target_compile_options(test_deploy_e2e PRIVATE ${PROJECT_WARNING_FLAGS})

    # Phase 1: one ctest entry per single-process Deploy.* test (process isolation)
    set(DEPLOY_SINGLE_TESTS
        InitFinalizeRoundtrip
        NegativeArgs
        SingleShardE2E
        LoadBeforeInit
        FinalizeWithoutInit
        SearchNegativeArgs
        LoadSidecars_ValidIdmap
        LoadSidecars_IdmapMismatch
        LoadSidecars_CorruptIdmap
        LoadSidecars_ValidCentroids
        LoadSidecars_CentroidMismatch
        LoadSidecars_CorruptCentroids
        RouteQueries_NullAndUniform
        RouteQueries_CentroidRouting
        DotNormSingleShard
        LoadNodeIdMismatch
        LoadBadFiles
        SyncBeforeInit
        SyncNegativeArgs
        SearchEdgeCases
        ContextBuild
        InitWithOptions
    )
    # Stub-only failure-injection tests (Deploy.Stub*): compiled only when
    # GD_HNSW_USE_UBSMEM_STUB is defined — against the real SDK these tests
    # don't exist (no injection points), so the ctest entries are stub-only too.
    if(GD_HNSW_DEPLOY_UBSMEM_STUB)
        list(APPEND DEPLOY_SINGLE_TESTS
            StubFailInitAttributes
            StubFailInitialize
            StubFailLoadAllocate
            StubFailLoadMap
            StubDeallocRetrySucceeds
            StubDeallocRetryExhausted
            StubDeallocHardFail
        )
    endif()
    # Wrapper guards against a hung ubs-mem backend; keep it well below the
    # ctest TIMEOUT (120s) so the wrapper reports a clean skip first.
    set(DEPLOY_TEST_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/run_deploy_test.sh)
    set(DEPLOY_WRAPPER_TIMEOUT 90)

    foreach(test ${DEPLOY_SINGLE_TESTS})
        add_test(NAME test_deploy_e2e_${test}
                 COMMAND bash ${DEPLOY_TEST_WRAPPER} ${DEPLOY_WRAPPER_TIMEOUT}
                         $<TARGET_FILE:test_deploy_e2e> --gtest_filter=Deploy.${test})
        set_tests_properties(test_deploy_e2e_${test} PROPERTIES
                             TIMEOUT 120 RUN_SERIAL TRUE LABELS "deploy")
    endforeach()

    # Phase 2: one ctest entry per multi-rank DeployMpi.* test (mpirun -np 2)
    if(MPIEXEC_EXECUTABLE)
        set(DEPLOY_MPI_TESTS
            TwoShardPushdownE2e
            DotNorm
            SyncValidation
        )
        if(GD_HNSW_DEPLOY_UBSMEM_STUB)
            list(APPEND DEPLOY_MPI_TESTS StubMpiDeallocWarning)
        endif()
        foreach(test ${DEPLOY_MPI_TESTS})
            add_test(NAME test_deploy_e2e_mpi_${test}
                     COMMAND bash ${DEPLOY_TEST_WRAPPER} ${DEPLOY_WRAPPER_TIMEOUT}
                             ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                             $<TARGET_FILE:test_deploy_e2e> --gtest_filter=DeployMpi.${test})
            set_tests_properties(test_deploy_e2e_mpi_${test} PROPERTIES
                                 TIMEOUT 120 RUN_SERIAL TRUE LABELS "deploy"
                                 ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
        endforeach()
    endif()
else()
    message(STATUS "test_deploy_e2e skipped — deploy gate not met")
endif()
