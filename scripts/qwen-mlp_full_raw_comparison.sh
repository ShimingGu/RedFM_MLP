#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Raw/terse magnitudes go through frozen Qwen. AION image tokens bypass Qwen
# and are learned by the downstream FSQ-factor image encoder before fusion.
# This keeps the comparison MLP branch unchanged and avoids treating arbitrary
# AION image-token IDs as language tokens.
export AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-mlp_full_raw_fusion_comparison}"
export AION_MAX_ROWS="${AION_MAX_ROWS:-200000}"
export QWEN_EMBEDDING_BATCH_SIZE="${QWEN_EMBEDDING_BATCH_SIZE:-1}"
export QWEN_MAX_LENGTH="${QWEN_MAX_LENGTH:-2048}"
export QWEN_POOLING="${QWEN_POOLING:-last}"

exec "$SCRIPT_DIR/qwen-mlp_full_comparison.sh" "$@"
