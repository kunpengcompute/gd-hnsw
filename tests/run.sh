#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ALGO="${ALGO_ROOT:-$(cd .. && pwd)}"
JOBS="${JOBS:-4}"
BUILD="$PWD/build"
# 覆盖率报告开关: 1=使用 lcov/genhtml, 0=跳过（默认 0）
ENABLE_LCOV="${ENABLE_LCOV:-0}"
DEPS_PREFIX="${UB_DEPS_PREFIX:-$(cd "${ALGO}/.." && pwd)/boostdb-ci/cmake_build/conan}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

cmake -S . -B "$BUILD" -DUB_ALGORITHM_ROOT="$ALGO" -DUB_TEST_COVERAGE=ON -DCMAKE_PREFIX_PATH="${DEPS_PREFIX}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "$BUILD" -j"$JOBS"

echo "==== UT-only ===="
find "$BUILD" -name '*.gcda' -delete
ctest --test-dir "$BUILD" -L unit --output-on-failure -j"$JOBS"

if [[ "${ENABLE_LCOV}" -eq 1 ]]; then
  if ! command -v lcov >/dev/null 2>&1 || ! command -v genhtml >/dev/null 2>&1; then
    echo "[WARN] ENABLE_LCOV=1 但未找到 lcov/genhtml，跳过覆盖率报告"
  else
    lcov --capture --directory "$BUILD" --output-file unit.info \
      --rc branch_coverage=1 --rc geninfo_unexecuted_blocks=1 \
      --ignore-errors mismatch,source,gcov,negative,empty >/dev/null
    lcov --remove unit.info '/usr/*' '*/test/*' \
      --output-file unit.info --ignore-errors unused >/dev/null
    lcov --summary unit.info | grep -E 'lines|functions'
    rm -rf coverage-html-unit
    genhtml unit.info --output-directory coverage-html-unit >/dev/null
    echo "HTML report: $PWD/coverage-html-unit/index.html"
  fi
else
  echo "[INFO] ENABLE_LCOV=0，跳过 lcov 覆盖率报告（UT 已执行）"
fi

echo "==== combined ===="
find "$BUILD" -name '*.gcda' -delete
ctest --test-dir "$BUILD" --output-on-failure -j"$JOBS"
