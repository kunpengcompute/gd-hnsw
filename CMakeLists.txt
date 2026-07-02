cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0144 NEW)
project(hnsw_shard_l2ing_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# FAISS
find_package(faiss CONFIG QUIET)
if(NOT faiss_FOUND)
    find_library(FAISS_LIBRARY NAMES libfaiss.a faiss 
        PATHS 
        /usr/local/lib 
        /usr/lib 
        /usr/lib/x86_64-linux-gnu
        $ENV{FAISS_ROOT}/lib
        NO_DEFAULT_PATH)
    
    if(NOT FAISS_LIBRARY)
        find_library(FAISS_LIBRARY NAMES libfaiss.a faiss)
    endif()
    
    find_path(FAISS_INCLUDE_DIR faiss/IndexHNSW.h PATHS 
        /usr/local/include 
        /usr/include
        $ENV{FAISS_ROOT}/include)
    
    if(FAISS_LIBRARY AND FAISS_INCLUDE_DIR)
        message(STATUS "Found FAISS: ${FAISS_LIBRARY}")
        
        if(FAISS_LIBRARY MATCHES "\\.a$")
            message(STATUS "FAISS is static library, adding dependencies...")
            find_package(BLAS REQUIRED)
            find_package(LAPACK REQUIRED)
            set(FAISS_LIBRARIES 
                ${FAISS_LIBRARY}
                ${BLAS_LIBRARIES}
                ${LAPACK_LIBRARIES}
            )
        else()
            set(FAISS_LIBRARIES ${FAISS_LIBRARY})
        endif()
        
        set(FAISS_INCLUDE_DIRS ${FAISS_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "FAISS not found. Please install FAISS or set FAISS_ROOT environment variable.")
    endif()
else()
    set(FAISS_LIBRARIES faiss)
    get_target_property(FAISS_INCLUDE_DIRS faiss INTERFACE_INCLUDE_DIRECTORIES)
endif()

# UBSMEM (Unified Backend Memory)
# Priority: 1) system install (find_path + find_library)
#           2) local path via -DUBSMEM_ROOT=...
set(UBSMEM_ROOT "" CACHE PATH "Local path to UBSMEM installation")
if(UBSMEM_ROOT)
    message(STATUS "Using local UBSMEM from: ${UBSMEM_ROOT}")
    set(UBSMEM_INCLUDE_DIRS "${UBSMEM_ROOT}/include")
    #set(UBSMEM_LIBRARIES "${UBSMEM_ROOT}/lib/libubsmd.so" "${UBSMEM_ROOT}/lib/libubsm_sdk.so")
    set(UBSMEM_LIBRARIES  "${UBSMEM_ROOT}/lib/libubsm_sdk.so")
    if(NOT EXISTS "${UBSMEM_INCLUDE_DIRS}/ubs_mem.h")
        message(FATAL_ERROR "UBSMEM headers not found in ${UBSMEM_INCLUDE_DIRS}")
    endif()
else()
    # Try to find UBSMEM in standard locations
    find_path(UBSMEM_INCLUDE_DIRS ubs_mem.h
        /usr/local/include
        /usr/include
        $ENV{UBSMEM_ROOT}/include
    )
#    find_library(UBSMEM_LIBRARY NAMES ubsmd ubsm_sdk
    find_library(UBSMEM_LIBRARY NAMES ubsm_sdk
        /usr/local/lib
        /usr/lib
        $ENV{UBSMEM_ROOT}/lib
    )
    if(UBSMEM_INCLUDE_DIRS AND UBSMEM_LIBRARY)
        message(STATUS "Found UBSMEM: include=${UBSMEM_INCLUDE_DIRS}, lib=${UBSMEM_LIBRARY}")
        set(UBSMEM_LIBRARIES ${UBSMEM_LIBRARY})
    else()
        message(WARNING "UBSMEM not found. Set -DUBSMEM_ROOT=... or ensure it's installed.")
        set(UBSMEM_INCLUDE_DIRS "")
        set(UBSMEM_LIBRARIES "")
    endif()
endif()

include_directories(
    ${MPI_CXX_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${FAISS_INCLUDE_DIRS}
    ${UBSMEM_INCLUDE_DIRS}
)

set(COMMON_LIBS
    ${MPI_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${FAISS_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    rt  # for shm_open
)

add_executable(hnsw_shard_ub src/hnsw_shard_ub.cpp)
target_link_libraries(hnsw_shard_ub ${COMMON_LIBS} ${UBSMEM_LIBRARIES} -lnuma)
target_compile_definitions(hnsw_shard_ub PRIVATE _GNU_SOURCE)
install(TARGETS hnsw_shard_ub RUNTIME DESTINATION bin)

install(DIRECTORY scripts/ DESTINATION scripts
        FILES_MATCHING PATTERN "*.sh"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

install(DIRECTORY docs/ DESTINATION docs
        FILES_MATCHING PATTERN "*.md")

message(STATUS "")
message(STATUS "========== HNSW Sharding Benchmark Configuration ==========")
message(STATUS "CMAKE_BUILD_TYPE:    ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER:  ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS:     ${CMAKE_CXX_FLAGS}")
message(STATUS "MPI:                 ${MPI_CXX_LIBRARIES}")
message(STATUS "OpenMP:              ${OpenMP_CXX_FLAGS}")
message(STATUS "HDF5:                ${HDF5_LIBRARIES}")
message(STATUS "FAISS:               ${FAISS_LIBRARIES}")
message(STATUS "============================================================")
message(STATUS "")
