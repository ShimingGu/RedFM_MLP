#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Validation control: keep the FM_Qwen3 magnitude+image pipeline but remove
# physical band descriptions. Image compactification remains configurable via
# QWEN_IMAGE_INPUT_MODE and QWEN_IMAGE_CROP_SIZE.
export AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-mlp_full_raw_comparison}"
export QWEN_PHYSICAL_CONTEXT=0

exec "$SCRIPT_DIR/qwen-mlp_full_image_comparison.sh" "$@"
