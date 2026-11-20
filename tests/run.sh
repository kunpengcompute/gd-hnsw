#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ALGO="${ALGO_ROOT:-$(cd .. && pwd)}"
JOBS="${JOBS:-4}"
BUILD="$PWD/build"

cmake -S . -B "$BUILD" -DUB_ALGORITHM_ROOT="$ALGO" -DUB_TEST_COVERAGE=ON
cmake --build "$BUILD" -j"$JOBS"

echo "==== UT-only ===="
find "$BUILD" -name '*.gcda' -delete
ctest --test-dir "$BUILD" -L unit --output-on-failure -j"$JOBS"

lcov --capture --directory "$BUILD" --output-file unit.info \
  --rc branch_coverage=1 --rc geninfo_unexecuted_blocks=1 \
  --ignore-errors mismatch,source,gcov,negative,empty >/dev/null

lcov --remove unit.info '/usr/*' '*/test/*' \
  --output-file unit.info --ignore-errors unused >/dev/null

lcov --summary unit.info | grep -E 'lines|functions'

rm -rf coverage-html-unit
genhtml unit.info --output-directory coverage-html-unit >/dev/null
echo "HTML report: $PWD/coverage-html-unit/index.html"

echo "==== combined ===="
find "$BUILD" -name '*.gcda' -delete
ctest --test-dir "$BUILD" --output-on-failure -j"$JOBS"
