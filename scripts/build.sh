#!/bin/bash

# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# script for build and ut gd-hnsw
# ***********************************************************************

set -o errtrace
set -o errexit

usage() {
    echo "Usage: $0 [ -h | -help ] [ -t | -type <build_type> ] [--ut] [--full_ut] [--coverage] [--asan] [--clean] [--ubs_mem [path]] "
    echo "       build_type 也可作为位置参数传入: $0 release --full_ut"
    echo "build_type: [debug, release]"
    echo "--ut: build + run unit tests (exclude deploy e2e)"
    echo "--full_ut: build + run ALL tests (include deploy e2e; auto-enable deploy layer)"
    echo "--coverage: lcov instrumentation + HTML coverage report after tests (needs lcov + genhtml)"
    echo "--asan: AddressSanitizer + LeakSanitizer instrumentation (GCC/clang only; Linux auto-enables leak detection)"
    echo "--ubs_mem: enable deploy layer, ubs-mem SDK path defaults to /usr/local/ubs_mem"
    echo "Examples:"
    echo " 1 ./build.sh -t release          # build only"
    echo " 2 ./build.sh -t release --ut    # build + run unit tests"
    echo " 3 ./build.sh -t release --full_ut   # build deploy layer + run ALL tests (incl. deploy e2e)"
    echo " 4 ./build.sh -t release --ubs_mem --ut   # build deploy layer(default SDK path) + run unit tests"
    echo " 5 ./build.sh -t release --ubs_mem /path/to/ubs_mem --ut   # build deploy layer with specified SDK path"
    echo " 6 ./build.sh -t debug --clean   # clean build/ and dist/ first, then build"
    echo " 7 ./build.sh -t debug --full_ut --coverage   # full tests + coverage report (build/coverage_report/)"
    echo " 8 ./build.sh -t debug --ut --asan   # unit tests + ASan/LSan memory leak detection"
    echo
    exit 1;
}

CMAKE_FLAGS=""
CURRENT_PATH="$(dirname "${BASH_SOURCE[0]}")"
PROJ_DIR="$(realpath "${CURRENT_PATH}/..")"
OUTPUT_DIR=${PROJ_DIR}/dist

BUILD_DIR=${PROJ_DIR}/build
LOG_FILE=${PROJ_DIR}/scripts/build.log

# FAISS 依赖 OpenBLAS
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p ${BUILD_DIR}
fi

BUILD_TYPE=release
BUILD_UT=OFF
RUN_FULL_UT=OFF
BUILD_COVERAGE=OFF
BUILD_ASAN=OFF
BUILD_CLEAN=OFF
BUILD_UBSMEM=OFF
UBSMEM_PATH=/usr/local/ubs_mem

while true; do
    case "$1" in
        -t | --type )
            [[ -z "$2" || "$2" == -* ]] && echo "Missing <build_type> after $1" && usage
            BUILD_TYPE=${2,,}
            shift 2
            ;;
        debug | release )
            # 位置参数形式：./build.sh release --full_ut 等价于 -t release --full_ut
            BUILD_TYPE=$1
            shift
            ;;
        --ut )
            BUILD_UT=ON
            shift ;;
        --full_ut )
            BUILD_UT=ON
            RUN_FULL_UT=ON
            shift ;;
        --coverage )
            BUILD_COVERAGE=ON
            shift ;;
        --asan )
            BUILD_ASAN=ON
            shift ;;
        --ubs_mem )
            BUILD_UBSMEM=ON
            # 可选路径参数：下一个参数若不以'-'开头，则视为 SDK 安装路径
            if [[ -n "$2" && "$2" != -* ]]; then
                UBSMEM_PATH=$2
                shift
            fi
            shift ;;
        --clean )
            BUILD_CLEAN=ON
            shift ;;
        -h | -help | --help )
            usage
            exit 0
            ;;
        * )
            # 参数解析完毕退出；未识别的参数直接报错，避免被静默丢弃
            # （旧版 "release --full_ut" 会在 release 处停止解析，--full_ut 被忽略）
            [[ -z "$1" ]] && break
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# 校验构建类型并生成 CMake flag（未指定时默认 release）
case "${BUILD_TYPE}" in
    debug)
        CMAKE_FLAGS+='-DCMAKE_BUILD_TYPE=Debug '
        ;;
    release)
        CMAKE_FLAGS+='-DCMAKE_BUILD_TYPE=Release '
        ;;
    *)
        echo "Invalid build type: ${BUILD_TYPE}"
        usage
        ;;
esac

# 未指定 --ut/--full_ut 时显式关闭测试，避免构建 gtest 与测试目标
if [[ "${BUILD_UT}" != "ON" ]]; then
    CMAKE_FLAGS+='-DBUILD_TESTING=OFF '
fi

# --coverage 依赖测试运行产生 .gcda 数据；插桩开关见根 CMakeLists 的 ENABLE_COVERAGE
if [[ "${BUILD_COVERAGE}" == "ON" ]]; then
    if [[ "${BUILD_UT}" != "ON" ]]; then
        echo "Error: --coverage requires tests to run, add --ut or --full_ut"
        usage
    fi
    CMAKE_FLAGS+='-DENABLE_COVERAGE=ON '
fi

# --asan 与 --coverage 互斥(两者都给二进制插桩,叠加使用会冲突)
# 与 --ubs_mem / --full_ut 同时使用时仅警告不阻止(MPI + ASan 兼容性差,deploy 层建议单独测)
if [[ "${BUILD_ASAN}" == "ON" ]]; then
    if [[ "${BUILD_COVERAGE}" == "ON" ]]; then
        echo "Error: --asan and --coverage are mutually exclusive (both instrument the binary)"
        usage
    fi
    CMAKE_FLAGS+='-DENABLE_ASAN=ON '
    if [[ "${BUILD_UBSMEM}" == "ON" || "${RUN_FULL_UT}" == "ON" ]]; then
        echo "Warning: --asan with deploy layer (--ubs_mem/--full_ut) may have MPI compatibility issues;"
        echo "         recommend running core/faiss/build tests separately: ./build.sh -t debug --ut --asan"
    fi
fi

# --full_ut 运行全部测试(含 deploy e2e)，需要 deploy 层；未显式指定 --ubs_mem 时自动启用默认 SDK 路径
if [[ "${RUN_FULL_UT}" == "ON" && "${BUILD_UBSMEM}" != "ON" ]]; then
    BUILD_UBSMEM=ON
    echo "--full_ut: auto-enable deploy layer (ubs-mem SDK path: ${UBSMEM_PATH})"
fi

# 指定 --ubs_mem 时启用 deploy 层，SDK 路径默认 /usr/local/ubs_mem，可通过传参覆盖
if [[ "${BUILD_UBSMEM}" == "ON" ]]; then
    CMAKE_FLAGS+="-DUBSMEM_ROOT=${UBSMEM_PATH} "
fi

CMAKE_CMD="cmake .. ${CMAKE_FLAGS}-DCMAKE_INSTALL_PREFIX=../dist"

# $LINENO是发生错误的行号，${FUNCNAME}是发生错误的函数名，$BASH_LINENO是调用该函数的行号
trap 'trap_error $LINENO ${FUNCNAME} $BASH_LINENO' ERR
trap 'cd $PROJ_DIR' EXIT

SUCCESS='[  \033[1;32mOK\033[0m  ]'
FAILURE='[\033[1;31mFAILED\033[0m]'

function log_info()
{
    if [ $# -lt 1 ]; then
        return
    fi
    echo "$(date +"%F %T") [INFO] $*" >> $LOG_FILE
    echo "$(date +"%F %T") [INFO] $*"
}

function trap_error()
{
    local err=$?
    local lineno=$1
    local funcname=$2
    local bash_lineno=$3

    log_info "Error occurred in function '$funcname' at line $lineno \
        Bash internal line: ${bash_lineno} \
        Command exit status: $err"
}

function echo_failure()
{
    echo -e "gd-hnsw project : $FAILURE"
}

function build_cmake()
{
  log_info "***** Start build_cmake with dir: ${BUILD_DIR} *****"
  log_info "CMAKE_CMD = ${CMAKE_CMD}"
  cd ${BUILD_DIR}
  # configure — 3rdparty 调度脚本(faiss/gtest)在此阶段构建并安装到 dist/3rdparty/
  ${CMAKE_CMD}
  cmake --build . --parallel $(nproc)

  local ret=$?
  cd ${PROJ_DIR}

  if [ $ret -ne 0 ]; then
    log_info "build_cmake failed"
    echo_failure
    exit 1
  fi
}

# 从 ut.log 提取 LeakSanitizer 报告，区分生产代码(src/)和 ut 自身(tests/)
# 只打印栈帧含 src/ 路径的泄漏块；ut 自身的泄漏只统计数量不打印
# 输入：$1 = ut.log 路径
function filter_asan_leaks()
{
  local ut_log=$1
  log_info "***** ASan leak filter — production code (src/) only *****"

  # 无 LeakSanitizer 报告则直接返回
  local total_leaks
  total_leaks=$(grep -c "ERROR: LeakSanitizer: detected memory leaks" ${ut_log} 2>/dev/null || echo 0)
  if [[ "${total_leaks}" == "0" ]]; then
    log_info "No LeakSanitizer reports — no memory leaks detected in production code"
    return
  fi
  log_info "LeakSanitizer report blocks found: ${total_leaks}"

  # 提取每个 leak 块(Direct/Indirect leak ... 到下一个 leak 或 SUMMARY)，
  # 若块内任意栈帧含 src/ 路径则打印整个块；否则跳过(ut 自身泄漏)
  awk '
    /^==[0-9]+==ERROR: LeakSanitizer/ { in_lsan=1; next }
    in_lsan && /^(Direct|Indirect) leak/ {
      if (blk != "" && blk ~ /src\//) { print blk; print_sep=1 }
      blk=$0 "\n"; next
    }
    in_lsan && /^    #[0-9]/ { blk=blk $0 "\n"; next }
    in_lsan && /^SUMMARY:/ {
      if (blk != "" && blk ~ /src\//) { print blk; print_sep=1 }
      in_lsan=0; next
    }
    END { if (print_sep) print "--------------------------------------------------------" }
  ' ${ut_log}

  local prod_leaks
  prod_leaks=$(awk '
    /^==[0-9]+==ERROR: LeakSanitizer/ { in_lsan=1; next }
    in_lsan && /^(Direct|Indirect) leak/ {
      if (blk != "" && blk ~ /src\//) cnt++
      blk=$0 "\n"; next
    }
    in_lsan && /^    #[0-9]/ { blk=blk $0 "\n"; next }
    in_lsan && /^SUMMARY:/ {
      if (blk != "" && blk ~ /src\//) cnt++
      in_lsan=0; next
    }
    END { print cnt+0 }
  ' ${ut_log})

  log_info "Production code (src/) leak blocks: ${prod_leaks}"
  log_info "Full ASan/LSan report (incl. tests/ internal): ${ut_log}"
}

function run_ut()
{
  if [[ "${BUILD_UT}" != "ON" ]]; then
    return
  fi

  log_info "***** Start run_ut with dir: ${BUILD_DIR} *****"
  local ut_log=${BUILD_DIR}/ut.log
  local ret=0

  # 默认排除需 deploy 层的端到端测试；--full_ut 模式运行全部测试
  local exclude_args=(-E "test_deploy_e2e.*")
  if [[ "${RUN_FULL_UT}" == "ON" ]]; then
    exclude_args=()
    log_info "Full UT mode: run all tests including deploy e2e"
  fi

  cd ${BUILD_DIR}
  # 覆盖率模式：清除历史 .gcda，确保报告只反映本次测试
  if [[ "${BUILD_COVERAGE}" == "ON" ]]; then
    find ${BUILD_DIR} -name '*.gcda' -delete
  fi
  set +e
  # ASan 模式：让 ut 自身越界/use-after-free 不 abort，继续跑完整个测试套
  # 编译期加 -fsanitize-recover=address 后，halt_on_error=0 可让 ASan 报告后继续执行
  # 跑完后由 filter_asan_leaks 提取生产代码(src/)的泄漏报告，过滤 tests/ 自身报告
  if [[ "${BUILD_ASAN}" == "ON" ]]; then
    ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:abort_on_error=0" \
      ctest --output-on-failure --parallel $(nproc) "${exclude_args[@]}" 2>&1 | tee ${ut_log}
  else
    ctest --output-on-failure --parallel $(nproc) "${exclude_args[@]}" 2>&1 | tee ${ut_log}
  fi
  ret=${PIPESTATUS[0]}
  set -e
  cd ${PROJ_DIR}

  # ASan 模式允许 ctest 失败（ut 自身 ASan 错误会导致部分 test fail），
  # 生产代码的泄漏报告会在 filter_asan_leaks 里单独提取并展示
  if [[ "${BUILD_ASAN}" != "ON" && $ret -ne 0 ]]; then
    log_info "run_ut failed"
    echo_failure
    exit 1
  fi

  local summary
  summary=$(grep -E "[0-9]+% tests passed" ${ut_log} | tail -1)
  log_info "UT summary: ${summary}"

  # ASan 模式：提取生产代码(src/)的泄漏报告，过滤 tests/ 自身报告
  if [[ "${BUILD_ASAN}" == "ON" ]]; then
    filter_asan_leaks ${ut_log}
  fi
}

# 生成覆盖率报告：lcov 捕获 + genhtml 渲染；仅统计 src/ 业务代码
# (tests/benchmarks 自身代码与 3rdparty 不计入)
function gen_coverage_report()
{
  if [[ "${BUILD_COVERAGE}" != "ON" ]]; then
    return
  fi

  log_info "***** Start gen_coverage_report *****"
  local report_dir=${BUILD_DIR}/coverage_report
  mkdir -p ${report_dir}
  local ret=0

  set +e
  # GCC 12+ 优化生成的行号数据与 lcov 2.x 严格校验不兼容(inconsistent/mismatch)，
  # 属已知的 gcov 数据精度问题，忽略后不影响其余统计；旧版 lcov 无此选项则不加
  local lcov_compat_args=()
  if lcov --help 2>&1 | grep -q -- '--ignore-errors'; then
      lcov_compat_args=(--ignore-errors inconsistent,mismatch)
  fi
  lcov --capture --directory ${BUILD_DIR} --output-file ${report_dir}/coverage.info \
       "${lcov_compat_args[@]}"
  ret=$?
  if [ $ret -eq 0 ]; then
    # 保留业务代码：src/ 实现文件 + include/ 头文件(header-only 实现也在其中)
    # 用 PROJ_DIR 绝对路径匹配，避免误匹配 dist/3rdparty/*/include 下的第三方头文件
    # extract/genhtml 同样执行 check_data_consistency 校验，需带同样的忽略参数
    lcov --extract ${report_dir}/coverage.info \
         "${PROJ_DIR}/src/*" "${PROJ_DIR}/include/*" \
         --output-file ${report_dir}/coverage.filtered.info \
         "${lcov_compat_args[@]}"
    ret=$?
  fi
  if [ $ret -eq 0 ]; then
    genhtml ${report_dir}/coverage.filtered.info --output-directory ${report_dir} \
         "${lcov_compat_args[@]}"
    ret=$?
  fi
  if [ $ret -eq 0 ]; then
    log_info "Coverage report (lcov): ${report_dir}/index.html"
  fi
  set -e

  if [ $ret -ne 0 ]; then
    log_info "gen_coverage_report failed"
    echo_failure
    exit 1
  fi
}

function parse_args()
{
    build_cmake && run_ut && gen_coverage_report
}

function clean() {
  rm -rf ${BUILD_DIR}/*
  rm -rf ${OUTPUT_DIR}/*
}

function check_glibc() {
    MIN_MAJOR=2
    MIN_MINOR=10

    # 可靠获取 glibc 版本
    GLIBC_VERSION=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
    if [[ -z "$GLIBC_VERSION" ]]; then
        GLIBC_VERSION=$(ldd --version 2>&1 | awk '/GNU libc/{print $NF; exit}')
    fi

    # 提取主次版本号
    CURRENT_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
    CURRENT_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)

    # 检查是否为数字
    if ! [[ "$CURRENT_MAJOR" =~ ^[0-9]+$ ]] || ! [[ "$CURRENT_MINOR" =~ ^[0-9]+$ ]]; then
        echo "Error: Failed to parse Glibc version: $GLIBC_VERSION"
        exit 1
    fi

    # 版本比较
    if [[ "$CURRENT_MAJOR" -lt "$MIN_MAJOR" ]] || \
       [[ "$CURRENT_MAJOR" -eq "$MIN_MAJOR" && "$CURRENT_MINOR" -lt "$MIN_MINOR" ]]; then
        echo "Error: Glibc version must be >= $MIN_MAJOR.$MIN_MINOR. Current version: $GLIBC_VERSION"
        exit 1
    fi
}

#### MAIN ####
echo $(date +"[%Y-%m-%d %H:%M]"): $0 $@

# check glibc version >=2.10
check_glibc

# check cmake available (CMakeLists.txt requires >= 3.18)
# 提前到 submodule 更新之前失败，避免无谓的耗时克隆
if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: cmake not found in PATH. Install it first, e.g.:"
    echo "  sudo yum install cmake           # distro package (check version >= 3.18)"
    echo "  pip3 install --user cmake        # latest version, installed to ~/.local/bin"
    exit 1
fi

# 覆盖率模式下预检查报告工具(lcov + genhtml)，避免构建+跑完测试后才发现缺工具
if [[ "${BUILD_COVERAGE}" == "ON" ]]; then
    if ! command -v lcov >/dev/null 2>&1 || ! command -v genhtml >/dev/null 2>&1; then
        echo "Error: --coverage needs lcov + genhtml to generate the report."
        exit 1
    fi
fi

# CI_BUILD是一个环境变量
if [ -z "${CI_BUILD}" ];then
    echo "update submodules ... "
    cd $PROJ_DIR && git submodule update --init --recursive
fi

if [[ "${BUILD_CLEAN}" == "ON" ]]; then
    clean
fi

cd ${PROJ_DIR} && parse_args $*
