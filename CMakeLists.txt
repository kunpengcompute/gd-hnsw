cmake_minimum_required(VERSION 3.18)
project(gd-hnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_COVERAGE "Enable code coverage instrumentation (GCC/clang --coverage)" OFF)
if(ENABLE_COVERAGE)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "ENABLE_COVERAGE works best with -DCMAKE_BUILD_TYPE=Debug")
    endif()
    add_compile_options(--coverage)
    add_link_options(--coverage)
    message(STATUS "Coverage instrumentation enabled (--coverage)")
endif()

option(ENABLE_ASAN "Enable AddressSanitizer + LeakSanitizer (GCC/clang)" OFF)
if(ENABLE_ASAN)
    if(ENABLE_COVERAGE)
        message(FATAL_ERROR "ENABLE_ASAN and ENABLE_COVERAGE are mutually exclusive — both instrument the binary; pick one.")
    endif()
    if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        message(FATAL_ERROR "ENABLE_ASAN requires GCC or Clang.")
    endif()
    add_compile_options(-fsanitize=address -fsanitize-recover=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
    # Linux: LSan is on by default with ASan. Other platforms need ASAN_OPTIONS=detect_leaks=1 at runtime.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "ASan/LSan enabled — LeakSanitizer active by default")
    else()
        message(WARNING "ASan enabled — set ASAN_OPTIONS=detect_leaks=1 at runtime to activate LSan on this platform")
    endif()
endif()

option(STRICT_WARNINGS "Enable strict warnings (-Wconversion -Wsign-conversion -Wdouble-promotion)" OFF)

option(BUILD_TESTING "Build tests" ON)

# Project-wide warning flags (high signal-to-noise by default)
set(PROJECT_WARNING_FLAGS
    -Wall -Wextra -Wshadow -Wnull-dereference
    -Wformat=2 -Wunused -Werror=return-type -Werror=format-security)
if(STRICT_WARNINGS)
    list(APPEND PROJECT_WARNING_FLAGS -Wconversion -Wsign-conversion -Wdouble-promotion)
    message(STATUS "Strict warnings enabled")
endif()

# Layer optimization flags: coverage builds need line-accurate gcov data —
# -O3/-ffast-math produce inconsistent line records that lcov 2.x drops even
# with --ignore-errors, so downgrade to -O0 there. ASan builds prefer -O1 -g
# for friendlier stack traces and to avoid -ffast-math floating-point false
# positives. Release builds unchanged.
if(ENABLE_COVERAGE)
    set(GD_HNSW_LAYER_OPT_FLAGS -O0 -g -march=native)
    message(STATUS "Coverage build: layer optimization downgraded to -O0 -g")
elseif(ENABLE_ASAN)
    set(GD_HNSW_LAYER_OPT_FLAGS -O1 -g -march=native)
    message(STATUS "ASan build: layer optimization set to -O1 -g for friendly stack traces")
else()
    set(GD_HNSW_LAYER_OPT_FLAGS -O3 -march=native -ffast-math)
endif()

# ============================================================
# Dependencies (all optional/QUIET except OpenMP and faiss, which
# have a guaranteed fallback).  Each layer gates on what it needs:
#   core → OpenMP
#   faiss → faiss (find/local/FetchContent, always available)
#   build → HDF5
#   deploy → Linux + MPI + ubs-mem + HDF5 + NUMA
# ============================================================

# MPI (optional — only needed for the deploy layer / bench)
find_package(MPI QUIET)
if(MPI_FOUND)
    message(STATUS "MPI found — deploy layer will be built")
else()
    message(STATUS "MPI not found — deploy layer will be skipped")
endif()

# OpenMP (required — core layer depends on it)
find_package(OpenMP REQUIRED)

# BLAS/LAPACK (required — faiss libfaiss.a references sgemm_ / sgelsd_)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# HDF5 (C++ bindings, optional — only the build/deploy layers need it)
find_package(HDF5 QUIET COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found — build layer will be built")
else()
    message(STATUS "HDF5 not found — build/deploy layers will be skipped")
endif()

# NUMA (Linux only, optional — only needed for the deploy layer)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        set(NUMA_FOUND TRUE)
    else()
        set(NUMA_FOUND FALSE)
        message(STATUS "libnuma not found — NUMA support disabled")
    endif()
else()
    set(NUMA_FOUND FALSE)
endif()

# ubs-mem SDK (optional — only needed for the deploy layer)
set(UBSMEM_ROOT "" CACHE PATH "ubs-mem SDK install root (contains include/ and lib/)")
if(UBSMEM_ROOT)
    find_path(UBSMEM_INCLUDE_DIR ubs_mem.h PATHS ${UBSMEM_ROOT}/include NO_DEFAULT_PATH)
    find_library(UBSMEM_LIBRARY ubsm_sdk PATHS ${UBSMEM_ROOT}/lib NO_DEFAULT_PATH)
else()
    find_path(UBSMEM_INCLUDE_DIR ubs_mem.h)
    find_library(UBSMEM_LIBRARY ubsm_sdk)
endif()
if(UBSMEM_INCLUDE_DIR AND UBSMEM_LIBRARY)
    set(UBSMEM_FOUND TRUE)
    message(STATUS "ubs-mem SDK: ${UBSMEM_LIBRARY}")
else()
    set(UBSMEM_FOUND FALSE)
    message(STATUS "ubs-mem SDK not found — deploy layer will be skipped")
endif()

# ============================================================
# 3rdparty submodule dispatch
# ============================================================

set(PROJECT_3RDPARTY_SRC_DIR ${PROJECT_SOURCE_DIR}/3rdparty)
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/dist)
set(PROJECT_3RDPARTY_BIN_DIR ${PROJECT_OUTPUT_PATH}/3rdparty)

# faiss install artifacts
include_directories(${PROJECT_3RDPARTY_BIN_DIR}/faiss/include)
link_directories(${PROJECT_3RDPARTY_BIN_DIR}/faiss/lib
                 ${PROJECT_3RDPARTY_BIN_DIR}/faiss/lib64)
SET(LIB_3RD_FAISS ${PROJECT_3RDPARTY_BIN_DIR}/faiss/lib/libfaiss.a)

add_subdirectory(3rdparty)

# Deploy-layer gate: only buildable on Linux with all runtime deps present.
set(GD_HNSW_DEPLOY_OK FALSE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MPI_FOUND AND UBSMEM_FOUND AND HDF5_FOUND AND NUMA_FOUND)
    set(GD_HNSW_DEPLOY_OK TRUE)
endif()

# ============================================================
# Layer 1: gd_hnsw_core (layout + search, no faiss/HDF5/MPI dep)
# ============================================================

add_library(gd_hnsw_core STATIC
    src/core/gd_layout.cpp
    src/core/gd_hnsw_search.cpp
    src/core/index_io.cpp
)
target_include_directories(gd_hnsw_core PUBLIC include)
target_link_libraries(gd_hnsw_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(gd_hnsw_core PRIVATE ${GD_HNSW_LAYER_OPT_FLAGS} ${PROJECT_WARNING_FLAGS})

# ============================================================
# Layer 2: gd_hnsw_faiss (extractor, depends on faiss; no HDF5)
# ============================================================

add_library(gd_hnsw_faiss STATIC
    src/index/faiss_extractor.cpp
)
target_include_directories(gd_hnsw_faiss PUBLIC include)
target_link_libraries(gd_hnsw_faiss PUBLIC
    gd_hnsw_core
    ${LIB_3RD_FAISS}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES})
target_compile_options(gd_hnsw_faiss PRIVATE ${GD_HNSW_LAYER_OPT_FLAGS} ${PROJECT_WARNING_FLAGS})

# ============================================================
# Layer 3: gd_hnsw_build (offline index build, +HDF5)
# ============================================================

if(HDF5_FOUND)
    add_library(gd_hnsw_build STATIC
        src/index/build.cpp
    )
    target_include_directories(gd_hnsw_build PUBLIC include
                                              PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(gd_hnsw_build PUBLIC gd_hnsw_faiss
                                           PRIVATE ${HDF5_CXX_LIBRARIES})
    target_compile_options(gd_hnsw_build PRIVATE ${GD_HNSW_LAYER_OPT_FLAGS} ${PROJECT_WARNING_FLAGS})
else()
    message(STATUS "gd_hnsw_build skipped — HDF5 not found")
endif()

# ============================================================
# Layer 4: gd_hnsw_api (public Context API, +MPI+ubs-mem+NUMA)
# Links build (transitively faiss/core/HDF5) + MPI + ubs-mem + NUMA.
# ============================================================

if(GD_HNSW_DEPLOY_OK)
    add_library(gd_hnsw_api STATIC
        src/deploy/gd_hnsw_api.cpp
    )
    target_include_directories(gd_hnsw_api PUBLIC include
                                              PRIVATE ${UBSMEM_INCLUDE_DIR})
    target_link_libraries(gd_hnsw_api PUBLIC gd_hnsw_build
                                         PRIVATE MPI::MPI_CXX
                                                 ${UBSMEM_LIBRARY}
                                                 ${NUMA_LIB})
    target_compile_options(gd_hnsw_api PRIVATE ${GD_HNSW_LAYER_OPT_FLAGS} ${PROJECT_WARNING_FLAGS})
else()
    message(STATUS "gd_hnsw_api skipped — deploy gate not met (Linux + MPI + ubs-mem + HDF5 + NUMA)")
endif()

# ============================================================
# Benchmarks and tools
# ============================================================

add_subdirectory(benchmarks)

# ============================================================
# Tests
# ============================================================

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
