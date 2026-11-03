#!/bin/bash
# Run a test file first, and only run the compute script if tests pass.
# Usage:
#   bash scripts/run_checked.sh tests/test_score_parser.py scripts/compute_games.py
# If tests fail, this prints a clear message and does NOT run the compute
# script, and exits with code 1. If tests pass, it runs the compute script
# and prints its output.
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "用法: bash scripts/run_checked.sh <测试文件> <计算脚本>" >&2
  echo "例如: bash scripts/run_checked.sh tests/test_score_parser.py scripts/compute_games.py" >&2
  exit 1
fi
TEST_FILE="$1"
COMPUTE_SCRIPT="$2"
echo "第一步: 运行测试 $TEST_FILE"
if python -m pytest "$TEST_FILE" -v; then
  echo "测试通过，开始计算 $COMPUTE_SCRIPT"
  python "$COMPUTE_SCRIPT"
else
  echo "测试未通过，已停止，未执行计算。" >&2
  exit 1
fi
