#!/usr/bin/env bash
#
# build.sh — 编译脚本 for ub_hnsw (hnsw_urma) 分布式检索算法
#
# 用法:
#   sh scripts/build.sh -t release [--ut]
#   sh scripts/build.sh -t debug
#
# 选项:
#   -t, --type <type>   编译类型: release(默认) | debug
#   --ut                编译完成后执行 ut 用例(当前为占位实现，仅打印日志)
#   -h, --help          打印帮助信息
#
# 示例:
#   sh scripts/build.sh -t release --ut
#

set -euo pipefail

# ----------------------------------------------------------------------------
# 默认参数
# ----------------------------------------------------------------------------
BUILD_TYPE="Release"
RUN_UT=0

# 项目根目录(脚本所在目录的上一级)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../gd_hnsw_urma" && pwd)"

# URMA cmake prefix 路径(与原编译命令保持一致)
CMAKE_PREFIX_PATH="/usr/local/bin/conan"

# 安装前缀
INSTALL_PREFIX="${PROJECT_ROOT}/build_ind/install"

# 构建/安装目录名
BUILD_DIR="build_ind"

# ----------------------------------------------------------------------------
# 帮助信息
# ----------------------------------------------------------------------------
usage() {
    echo "用法: sh scripts/build.sh -t <release|debug> [--ut]"
    echo ""
    echo "选项:"
    echo "  -t, --type <type>   编译类型: release(默认) | debug"
    echo "      --ut            编译完成后执行 ut 用例(占位实现，仅打印日志)"
    echo "  -h, --help          打印帮助信息"
    echo ""
    echo "示例:"
    echo "  sh scripts/build.sh -t release --ut"
    echo "  sh scripts/build.sh -t debug"
}

# ----------------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--type)
            shift
            case "$1" in
                release|Release|RELEASE)
                    BUILD_TYPE="Release"
                    ;;
                debug|Debug|DEBUG)
                    BUILD_TYPE="Debug"
                    ;;
                *)
                    echo "[ERROR] 不支持的编译类型: $1 (仅支持 release|debug)"
                    usage
                    exit 1
                    ;;
            esac
            shift
            ;;
        --ut)
            RUN_UT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# UT 占位函数: 不执行真正的 ut 测试，仅打印日志
# ----------------------------------------------------------------------------
run_ut() {
    local ut_script="${SCRIPT_DIR}/../tests/run.sh"
    local gtest_source="${UB_GTEST_SOURCE_DIR:-/opt/compile_rely/googletest}"
    local gtest_build="${PROJECT_ROOT}/${BUILD_DIR}/googletest"
    local gtest_prefix="${PROJECT_ROOT}/${BUILD_DIR}/ut-deps"
    local deps_prefix="${UB_DEPS_PREFIX:-${CMAKE_PREFIX_PATH}}"
    local ut_status=0

    echo ""
    echo "============================================================"
    echo "[UT] 测试脚本 : ${ut_script}"
    echo "============================================================"

    if [[ ! -f "${ut_script}" ]]; then
        echo "[UT][ERROR] 未找到测试脚本: ${ut_script}"
        return 1
    fi

    if [[ ! -f "${gtest_source}/CMakeLists.txt" ]]; then
        echo "[UT][ERROR] 未找到GoogleTest源码: ${gtest_source}，请通过UB_GTEST_SOURCE_DIR指定"
        return 1
    fi

    echo "[UT] 配置GoogleTest: ${gtest_source}"
    cmake -S "${gtest_source}" -B "${gtest_build}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_INSTALL_PREFIX="${gtest_prefix}" \
        -DBUILD_GMOCK=OFF -DINSTALL_GTEST=ON || return $?
    echo "[UT] 编译GoogleTest"
    cmake --build "${gtest_build}" --parallel "${JOBS:-4}" || return $?
    echo "[UT] 安装GoogleTest: ${gtest_prefix}"
    cmake --install "${gtest_build}" || return $?

    deps_prefix="${gtest_prefix}${deps_prefix:+;${deps_prefix}}"
    ALGO_ROOT="${PROJECT_ROOT}" \
    UB_DEPS_PREFIX="${deps_prefix}" \
    BUILD_TYPE="${BUILD_TYPE}" \
    ENABLE_LCOV="${ENABLE_LCOV:-0}" \
        bash "${ut_script}" || ut_status=$?

    if [[ "${ut_status}" -ne 0 ]]; then
        echo "[UT][ERROR] 单元测试执行失败，退出码: ${ut_status}"
        return "${ut_status}"
    fi

    echo "[UT] 单元测试执行完成 ✓"
    return 0
}

# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
echo "============================================================"
echo "[BUILD] 项目根目录 : ${PROJECT_ROOT}"
echo "[BUILD] 编译类型   : ${BUILD_TYPE}"
echo "[BUILD] 构建目录   : ${BUILD_DIR}"
echo "[BUILD] 安装前缀   : ${INSTALL_PREFIX}"
echo "[BUILD] URMA 开关  : OFF"
echo "[BUILD] CMAKE_PREFIX_PATH : ${CMAKE_PREFIX_PATH}"
echo "============================================================"

# Step 1: cmake 配置
echo "[BUILD] Step 1/3: cmake 配置 ..."
cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
    -DENABLE_URMA=OFF

# Step 2: 编译
echo "[BUILD] Step 2/3: cmake 编译 (-j) ..."
cmake --build "${PROJECT_ROOT}/${BUILD_DIR}" -j

# Step 3: 安装
echo "[BUILD] Step 3/3: cmake 安装 ..."
cmake --install "${PROJECT_ROOT}/${BUILD_DIR}"

echo "============================================================"
echo "[BUILD] 编译完成 ✓"
echo "============================================================"

# ----------------------------------------------------------------------------
# 可选: 执行 ut
# ----------------------------------------------------------------------------
if [[ "${RUN_UT}" -eq 1 ]]; then
    run_ut
fi

exit 0
