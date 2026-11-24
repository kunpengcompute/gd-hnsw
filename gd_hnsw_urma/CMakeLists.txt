# Standalone build for the ub_hnsw (hnsw_urma) distributed retrieval algorithm.
#
# Produces libub_hnsw.so with a C ABI (ub_c_api.h) and the public algorithm
# headers, decoupled from Milvus. The library owns its own logging backend
# (hnsw_urma/core/log.hpp) and receives the URMA context from the host via
# UbSetUrmaContext (R1); it does not call urma_init itself in service mode.
#
# This target is built in parallel with the legacy Milvus-embedded path during
# phases A-C; Milvus switches to find_package(ub_hnsw) in phase D.
#
# Sources are the pinned hnsw_urma tree under this directory. The builder
# (src/index/builder.cpp) is included: it has no faiss dependency and is
# consumed by Milvus (UBVectorIndex / UrmaFaissBuilder / the urma_builder
# plugin). Benchmark and mock sources are excluded.

cmake_minimum_required(VERSION 3.22)
project(ub_hnsw LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Numeric package version (used for write_basic_package_version_file and
# SOVERSION). The Milvus-embedded build defines HNSW_URMA_PROJECT_VERSION as a
# string tag; for the standalone build we fall back to a numeric 0.1.0.
if(NOT DEFINED UB_HNSW_VERSION)
    set(UB_HNSW_VERSION 0.1.0 CACHE STRING "ub_hnsw package version" FORCE)
endif()


set(UB_HNSW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB_RECURSE UB_HNSW_SOURCES CONFIGURE_DEPENDS
    "${UB_HNSW_ROOT}/src/*.cpp")

# Exclude portable-test-only sources from the service library.
list(FILTER UB_HNSW_SOURCES EXCLUDE REGEX
    "/src/runtime/benchmark\\.cpp$")
list(FILTER UB_HNSW_SOURCES EXCLUDE REGEX
    "/src/search/mock_io\\.cpp$")
list(FILTER UB_HNSW_SOURCES EXCLUDE REGEX
    "/src/transport/mock_lane_post_engine\\.cpp$")
list(FILTER UB_HNSW_SOURCES EXCLUDE REGEX
    "/src/transport/shared_memory_transport\\.cpp$")

# URMA transport is only compiled when the URMA (UMDK) backend is available.
# ENABLE_URMA is normally inherited from the parent Milvus build; default OFF
# for a standalone build.
if(NOT DEFINED ENABLE_URMA)
    set(ENABLE_URMA OFF CACHE BOOL "Enable the URMA (UMDK) data-plane transport" FORCE)
endif()
if(NOT ENABLE_URMA)
    list(FILTER UB_HNSW_SOURCES EXCLUDE REGEX "/src/transport/urma/")
endif()

add_library(ub_hnsw SHARED ${UB_HNSW_SOURCES})

target_include_directories(ub_hnsw
    PUBLIC
        $<BUILD_INTERFACE:${UB_HNSW_ROOT}/include>
        $<INSTALL_INTERFACE:include>
    )

# Version macros expected by src/runtime/config.cpp.
# fmt: the log.hpp backend uses fmt::format with a runtime format string.
# When this target is built via add_subdirectory from the Milvus tree, the
# conan fmt::fmt target already exists and we just link it. For a standalone
# build, locate fmt via find_package (CMakeDeps config produced by conan, or a
# system fmt-config).
# Resolve the fmt target name: the Milvus conan build produces fmt::fmt
# (static lib); a standalone conan CMakeDeps install may produce
# fmt::fmt-header-only instead. Accept either.
if(TARGET fmt::fmt)
    set(_ub_hnsw_fmt_target fmt::fmt)
elseif(TARGET fmt::fmt-header-only)
    set(_ub_hnsw_fmt_target fmt::fmt-header-only)
else()
    find_package(fmt CONFIG QUIET)
    if(TARGET fmt::fmt)
        set(_ub_hnsw_fmt_target fmt::fmt)
    elseif(TARGET fmt::fmt-header-only)
        set(_ub_hnsw_fmt_target fmt::fmt-header-only)
    endif()
endif()
if(_ub_hnsw_fmt_target)
    target_link_libraries(ub_hnsw PUBLIC ${_ub_hnsw_fmt_target})
else()
    message(FATAL_ERROR "ub_hnsw requires fmt (log.hpp backend). Point -DCMAKE_PREFIX_PATH at a conan CMakeDeps output (fmt-config.cmake) or install fmt-config.")
endif()
target_compile_definitions(ub_hnsw PUBLIC
    HNSW_URMA_PROJECT_VERSION="milvus-alignment" HNSW_URMA_HAS_MPI=0)

if(ENABLE_URMA)
    find_path(HNSW_URMA_UMDK_INCLUDE_DIR urma_api.h
        HINTS ENV UMDK_ROOT
        PATHS /usr/include/ub/umdk/urma /usr/include/ub/umdk /usr/local/include/ub/umdk/urma
        PATH_SUFFIXES include include/urma src/urma/lib/urma/core/include)
    find_path(HNSW_URMA_UMDK_BONDING_INCLUDE_DIR urma_ubagg.h
        HINTS ENV UMDK_ROOT
        PATHS /usr/include/ub/umdk/urma /usr/include/ub/umdk /usr/local/include/ub/umdk/urma
        PATH_SUFFIXES include include/urma src/urma/lib/urma/bond/include)
    find_library(HNSW_URMA_UMDK_LIBRARY NAMES urma
        HINTS ENV UMDK_ROOT PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
        PATH_SUFFIXES lib lib64 build/lib)
    if(NOT HNSW_URMA_UMDK_INCLUDE_DIR OR NOT HNSW_URMA_UMDK_LIBRARY OR NOT HNSW_URMA_UMDK_BONDING_INCLUDE_DIR)
        message(FATAL_ERROR "ENABLE_URMA requires the URMA library, headers and bonding header")
    endif()
    target_compile_definitions(ub_hnsw PUBLIC
        HNSW_URMA_ENABLE_URMA=1 HNSW_URMA_HAS_BONDING_HEADER=1)
    target_include_directories(ub_hnsw PRIVATE
        ${HNSW_URMA_UMDK_INCLUDE_DIR} ${HNSW_URMA_UMDK_BONDING_INCLUDE_DIR})
    target_link_libraries(ub_hnsw PUBLIC ${HNSW_URMA_UMDK_LIBRARY} ${CMAKE_DL_LIBS})
else()
    target_compile_definitions(ub_hnsw PUBLIC HNSW_URMA_ENABLE_URMA=0)
endif()

set_target_properties(ub_hnsw PROPERTIES
    VERSION "${UB_HNSW_VERSION}"
    SOVERSION 0
    POSITION_INDEPENDENT_CODE ON)

# Install the library, the C ABI header (flat under include/ub_hnsw), and the
# public algorithm headers (under include/hnsw_urma) so a Milvus build using
# find_package(ub_hnsw) can compile UBVectorIndex / UrmaFaissBuilder.
install(TARGETS ub_hnsw
    EXPORT ub_hnsw-targets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(DIRECTORY "${UB_HNSW_ROOT}/include/hnsw_urma"
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

install(FILES "${UB_HNSW_ROOT}/include/hnsw_urma/ub_c_api.h"
    DESTINATION include/ub_hnsw)

# pkg-config template (consumed by ubcgo via `#cgo pkg-config: ub_hnsw`).
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ub_hnsw.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ub_hnsw.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ub_hnsw.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

install(EXPORT ub_hnsw-targets
    NAMESPACE ub_hnsw::
    FILE ub_hnsw-targets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ub_hnsw")

# CMake package config so find_package(ub_hnsw) works from the Milvus tree.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ub_hnsw-config-version.cmake"
    VERSION ${UB_HNSW_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ub_hnsw-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ub_hnsw-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ub_hnsw")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ub_hnsw-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ub_hnsw-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ub_hnsw")
