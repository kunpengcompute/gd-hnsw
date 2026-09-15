set(SOURCE_DIR ${PROJECT_3RDPARTY_SRC_DIR}/faiss/faiss)
set(BUILD_DIR ${PROJECT_BUILD_PATH}/faiss)
set(INSTALL_DIR ${PROJECT_3RDPARTY_BIN_DIR}/faiss)

message("-- faiss: check if need to build ")

# 检查 submodule 是否已 init（源码目录非空）
if(NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
    message("-- faiss: submodule not initialized, skipped (run "
            "`git submodule update --init 3rdparty/faiss/faiss` "
            "to build)")
    return()
endif()

# 检查是否已经构建
find_file(LIBS_BUILT_FAISS libfaiss.a ${INSTALL_DIR}/lib/ ${INSTALL_DIR}/lib64/)
if(NOT LIBS_BUILT_FAISS)

    # create build dir
    exec_program(mkdir ./ ARGS -p ${BUILD_DIR})
    exec_program(mkdir ./ ARGS -p ${INSTALL_DIR})

    # configure — 关闭 faiss 的 GPU/Python/测试，仅构建 CPU 静态库
    exec_program(cmake ${BUILD_DIR} ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DFAISS_ENABLE_GPU=OFF
        -DFAISS_ENABLE_PYTHON=OFF
        -DBUILD_TESTING=OFF
        ${SOURCE_DIR})

    # execute make && make install
    exec_program(make ${BUILD_DIR} ARGS clean)
    exec_program(make ${BUILD_DIR} ARGS -j8)
    exec_program("make install" ${BUILD_DIR})
else()
    message("-- faiss: ${LIBS_BUILT_FAISS}")
    message("-- faiss: has been built, ignored")
endif(NOT LIBS_BUILT_FAISS)
