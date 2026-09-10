#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
用法:
  ./frog.sh -i                         安装依赖和本地 frog 包
  ./frog.sh -l                         列出所有 FrogLab 任务
  ./frog.sh -t [训练参数...]            执行 scripts/frog_rl/train.py
  ./frog.sh -p [推理参数...]            执行 scripts/frog_rl/play.py

环境变量:
  ISAAC_PATH  Isaac Sim 安装目录，使用其中的 python.sh
  PYTHON      覆盖默认 Python 解释器

示例:
  ./frog.sh -t --task FrogLab-Isaac-AMP-Flat-Unitree-G1-v0 --headless
  ./frog.sh -p --task FrogLab-Isaac-AMP-Flat-Unitree-G1-v0 --checkpoint /path/to/model.pt
EOF
}

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
elif [[ -n "${ISAAC_PATH:-}" ]]; then
    PYTHON_CMD=("$ISAAC_PATH/python.sh")
else
    PYTHON_CMD=(python)
fi

run_python() {
    "${PYTHON_CMD[@]}" "$@"
}

install_dependencies() {
    run_python -m pip install -e "$ROOT_DIR/source/frog_lab"
    run_python -m pip install -e "$ROOT_DIR/source/frog_rl"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

mode="$1"
shift

case "$mode" in
    -i)
        [[ $# -eq 0 ]] || { echo "-i 不接受额外参数。" >&2; usage >&2; exit 2; }
        install_dependencies
        ;;
    -l)
        [[ $# -eq 0 ]] || { echo "-l 不接受额外参数。" >&2; usage >&2; exit 2; }
        run_python "$ROOT_DIR/scripts/list_envs.py"
        ;;
    -t)
        run_python "$ROOT_DIR/scripts/frog_rl/train.py" --headless "$@"
        ;;
    -p)
        run_python "$ROOT_DIR/scripts/frog_rl/play.py"  "$@"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "未知选项: $mode" >&2
        usage >&2
        exit 2
        ;;
esac