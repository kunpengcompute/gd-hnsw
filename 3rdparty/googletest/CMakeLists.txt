set(SOURCE_DIR ${PROJECT_3RDPARTY_SRC_DIR}/googletest/googletest)
set(BUILD_DIR ${PROJECT_BUILD_PATH}/googletest)
set(INSTALL_DIR ${PROJECT_3RDPARTY_BIN_DIR}/googletest)

message("-- gtest: check if need to build ")

# 检查 submodule 是否已 init（源码目录非空）
if(NOT EXISTS "${SOURCE_DIR}/CMakeLists.txt")
    message("-- gtest: submodule not initialized, skipped (run "
            "`git submodule update --init 3rdparty/googletest/googletest` "
            "to build)")
    return()
endif()

# 检查是否已经构建
find_file(LIBS_BUILT_GTEST libgtest.a ${INSTALL_DIR}/lib/)
if(NOT LIBS_BUILT_GTEST)

    # create build dir
    exec_program(mkdir ./ ARGS -p ${BUILD_DIR})
    exec_program(mkdir ./ ARGS -p ${INSTALL_DIR})

    # configure
    exec_program(cmake ${BUILD_DIR} ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        ${SOURCE_DIR})

    # execute make && make install
    exec_program(make ${BUILD_DIR} ARGS clean)
    exec_program(make ${BUILD_DIR} ARGS -j8)
    exec_program("make install" ${BUILD_DIR})
else()
    message("-- gtest: ${LIBS_BUILT_GTEST}")
    message("-- gtest: has been built, ignored")
endif(NOT LIBS_BUILT_GTEST)
