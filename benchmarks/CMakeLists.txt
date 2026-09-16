# ============================================================
# benchmarks/CMakeLists.txt — benchmark main + standalone build tool
# ============================================================

# --- gd_hnsw_bench: full deploy benchmark (MPI, links deploy layer) ---
if(GD_HNSW_DEPLOY_OK)
    add_executable(gd_hnsw_bench
        main.cpp
        bench_runner.cpp
        bench_report.cpp
    )
    target_include_directories(gd_hnsw_bench PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HDF5_INCLUDE_DIRS}
        ${UBSMEM_INCLUDE_DIR}
    )
    target_link_libraries(gd_hnsw_bench PRIVATE
        gd_hnsw_api
        MPI::MPI_CXX
        ${HDF5_CXX_LIBRARIES}
    )
    target_compile_options(gd_hnsw_bench PRIVATE -O3 -march=native -ffast-math ${PROJECT_WARNING_FLAGS})
else()
    message(STATUS "gd_hnsw_bench skipped — deploy gate not met (Linux + MPI + ubs-mem + HDF5 + NUMA)")
endif()

# --- build_index: standalone index builder (no MPI/ubs-mem; links build layer only) ---
if(HDF5_FOUND)
    add_executable(build_index
        build_index.cpp
    )
    target_include_directories(build_index PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(build_index PRIVATE
        gd_hnsw_build
    )
    target_compile_options(build_index PRIVATE -O3 -march=native -ffast-math ${PROJECT_WARNING_FLAGS})
else()
    message(STATUS "build_index skipped — HDF5 not found")
endif()
