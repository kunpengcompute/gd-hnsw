cmake_minimum_required(VERSION 3.18)
project(shm_hnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PROFILE_LIGHTWEIGHT "Use lightweight profiling (phase-level only, no per-edge timing)" OFF)
if(PROFILE_LIGHTWEIGHT)
    add_compile_definitions(HNSW_PROFILE_LIGHTWEIGHT)
    message(STATUS "Lightweight profiling enabled — per-edge sub-phase timing disabled")
endif()

option(STRICT_WARNINGS "Enable strict warnings (-Wconversion -Wsign-conversion -Wdouble-promotion)" OFF)

# Project-wide warning flags (high signal-to-noise by default)
set(PROJECT_WARNING_FLAGS
    -Wall -Wextra -Wshadow -Wnull-dereference
    -Wformat=2 -Wunused -Werror=return-type -Werror=format-security)
if(STRICT_WARNINGS)
    list(APPEND PROJECT_WARNING_FLAGS -Wconversion -Wsign-conversion -Wdouble-promotion)
    message(STATUS "Strict warnings enabled")
endif()

# ============================================================
# Dependencies
# ============================================================

# MPI
find_package(MPI REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# HDF5 (C++ bindings)
find_package(HDF5 REQUIRED COMPONENTS CXX)

# NUMA (Linux only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(NUMA_LIB numa REQUIRED)
endif()

# ubs-mem SDK
set(UBSMEM_ROOT "" CACHE PATH "ubs-mem SDK install root (contains include/ and lib/)")
if(UBSMEM_ROOT)
    find_path(UBSMEM_INCLUDE_DIR ubs_mem.h PATHS ${UBSMEM_ROOT}/include NO_DEFAULT_PATH)
    find_library(UBSMEM_LIBRARY ubsm_sdk PATHS ${UBSMEM_ROOT}/lib NO_DEFAULT_PATH)
    if(NOT UBSMEM_INCLUDE_DIR OR NOT UBSMEM_LIBRARY)
        message(FATAL_ERROR "ubs-mem SDK not found at ${UBSMEM_ROOT}")
    endif()
else()
    find_path(UBSMEM_INCLUDE_DIR ubs_mem.h)
    find_library(UBSMEM_LIBRARY ubsm_sdk)
    if(NOT UBSMEM_INCLUDE_DIR OR NOT UBSMEM_LIBRARY)
        message(FATAL_ERROR "ubs-mem SDK not found. Set -DUBSMEM_ROOT=<path>")
    endif()
endif()
message(STATUS "ubs-mem SDK: ${UBSMEM_LIBRARY}")

# faiss
# Priority: 1) system/user install (find_package)
#           2) local source (-DFAISS_LOCAL_PATH=...)
#           3) FetchContent from GitHub (needs network)
set(FAISS_LOCAL_PATH "" CACHE PATH "Local path to faiss source tree")

find_package(faiss QUIET)
if(faiss_FOUND)
    message(STATUS "Using system-installed faiss: ${faiss_DIR}")
elseif(FAISS_LOCAL_PATH)
    message(STATUS "Using local faiss source: ${FAISS_LOCAL_PATH}")
    set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(${FAISS_LOCAL_PATH} ${CMAKE_BINARY_DIR}/_deps/faiss-build)
else()
    message(STATUS "Fetching faiss v1.14.1 from GitHub...")
    set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(
        faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG        v1.14.1
    )
    FetchContent_MakeAvailable(faiss)
endif()

# ============================================================
# Library: shm_hnsw_core (layout + search, no faiss/HDF5 dep)
# ============================================================

add_library(shm_hnsw_core STATIC
    src/shm_layout.cpp
    src/shm_hnsw_search.cpp
    src/index_io.cpp
)
target_include_directories(shm_hnsw_core PUBLIC include)
target_link_libraries(shm_hnsw_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(shm_hnsw_core PRIVATE -O3 -march=native -ffast-math ${PROJECT_WARNING_FLAGS})

# ============================================================
# Library: shm_hnsw_faiss (extractor, depends on faiss)
# ============================================================

add_library(shm_hnsw_faiss STATIC
    src/faiss_extractor.cpp
)
target_include_directories(shm_hnsw_faiss PUBLIC include)
target_link_libraries(shm_hnsw_faiss PUBLIC shm_hnsw_core faiss)
target_compile_options(shm_hnsw_faiss PRIVATE -O3 -march=native -ffast-math ${PROJECT_WARNING_FLAGS})

# ============================================================
# Executable: shm_hnsw_bench (MPI benchmark with HDF5)
# ============================================================

add_executable(shm_hnsw_bench src/main.cpp)
target_include_directories(shm_hnsw_bench PRIVATE ${HDF5_INCLUDE_DIRS} ${UBSMEM_INCLUDE_DIR})
target_link_libraries(shm_hnsw_bench PRIVATE
    shm_hnsw_core
    shm_hnsw_faiss
    MPI::MPI_CXX
    ${HDF5_CXX_LIBRARIES}
    ${UBSMEM_LIBRARY}
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(shm_hnsw_bench PRIVATE ${NUMA_LIB})
endif()
target_compile_options(shm_hnsw_bench PRIVATE -O3 -march=native -ffast-math ${PROJECT_WARNING_FLAGS})

# ============================================================
# Executable: build_index (standalone index builder, no MPI)
# ============================================================

add_executable(build_index src/build_index.cpp)
target_include_directories(build_index PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(build_index PRIVATE
    shm_hnsw_core
    shm_hnsw_faiss
    ${HDF5_CXX_LIBRARIES}
)
target_compile_options(build_index PRIVATE -O3 -march=native -ffast-math ${PROJECT_WARNING_FLAGS})
