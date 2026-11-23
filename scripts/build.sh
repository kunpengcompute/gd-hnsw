#!/bin/bash
# 门禁调度入口：sh scripts/build.sh -t release [--ut]，开发者将 __TODO__ 替换为真实命令
set -eu

cd "$(dirname "$0")/.."

build_type=
run_ut=0
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--type) [ $# -ge 2 ] || { echo "[ERROR] $1 缺少取值" >&2; exit 1; }; build_type=$2; shift 2 ;;
        --ut) run_ut=1; shift ;;
        *) echo "[ERROR] 未识别的参数: $1" >&2; exit 1 ;;
    esac
done

[ "$build_type" = release ] || { echo "[ERROR] 仅支持 -t release" >&2; exit 1; }

BUILD_CMD="__TODO__"
UT_CMD="__TODO__"

run_stage() {
    [ "$2" != "__TODO__" ] || { echo "[TODO] $1 未补齐，请替换 scripts/build.sh 中的 __TODO__" >&2; exit 1; }
    echo "[INFO] $1: $2"
    sh -c "$2"
}

run_stage "release 编译" "$BUILD_CMD"
if [ "$run_ut" -eq 1 ]; then run_stage "单元测试" "$UT_CMD"; fi
