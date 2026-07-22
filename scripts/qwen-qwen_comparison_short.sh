#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# qwen_qwen_comparison.py samples max_rows randomly with the fixed seed 42.
# Keep this run isolated from the full-catalogue experiment's output files.
export AION_MAX_ROWS=300000
export AION_OUTPUT_DIR="${AION_SHORT_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-qwen-comparison-short-n300000-random-s42}"

exec "$SCRIPT_DIR/qwen-qwen_comparison.sh" "$@"
