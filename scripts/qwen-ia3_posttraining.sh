#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
elif [[ -f "${PIXI_MANIFEST:-$REPO_ROOT/pixi.toml}" ]]; then
    PYTHON_CMD=(pixi run --manifest-path "${PIXI_MANIFEST:-$REPO_ROOT/pixi.toml}" python)
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
else
    echo "No project Python environment found." >&2
    exit 1
fi

if ! "${PYTHON_CMD[@]}" -c 'from peft import IA3Config; import torch, transformers' >/dev/null; then
    echo "The selected environment is missing IA3/PEFT dependencies." >&2
    exit 1
fi

GPU_DEVICE="${IA3_GPU_DEVICE:-2}"
OUTPUT_DIR="${IA3_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-ia3_posttraining-e10}"
CHECKPOINT_DIR="${IA3_CHECKPOINT_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/ia3_checkpoints/qwen-ia3-posttraining-e10}"
mkdir -p -- "$OUTPUT_DIR"

FLAGS=()
[[ "${QWEN_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-qwen-download)
[[ "${QWEN_FORCE_RECOMPUTE:-0}" == 1 ]] && FLAGS+=(--force-recompute-qwen)
[[ "${QWEN_USE_MORPHOLOGY:-0}" == 1 ]] && FLAGS+=(--use-morphology)

COMMON_ARGS=(
    --catalogue "${AION_CATALOGUE:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits}"
    --output-dir "$OUTPUT_DIR"
    --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"
    --max-rows "${AION_MAX_ROWS:-300000}"
    --seed "${AION_SEED:-42}"
    --train-fraction "${AION_TRAIN_FRACTION:-0.20}"
    --test-fraction "${AION_TEST_FRACTION:-0.75}"
    --val-fraction "${AION_VAL_FRACTION:-0.05}"
    --n-z-bins "${AION_N_Z_BINS:-300}"
    --qwen-model "${QWEN_MODEL:-Qwen3.5-4B-Base}"
    --qwen-max-length "${QWEN_MAX_LENGTH:-2048}"
    --qwen-pooling last
    --eval-batch-size "${POSTTRAIN_EVAL_BATCH_SIZE:-8}"
    --head-learning-rate "${HEAD_LEARNING_RATE:-2e-4}"
    --ia3-epochs "${IA3_EPOCHS:-10}"
    --ia3-batch-size "${IA3_BATCH_SIZE:-1}"
    --ia3-gradient-accumulation-steps "${IA3_GRADIENT_ACCUMULATION_STEPS:-16}"
    --ia3-learning-rate "${IA3_LEARNING_RATE:-1e-5}"
    --ia3-head-warmup-epochs "${IA3_HEAD_WARMUP_EPOCHS:-3}"
    --ia3-max-grad-norm "${IA3_MAX_GRAD_NORM:-0.1}"
    --ia3-target-modules "${IA3_TARGET_MODULES:-k_proj,v_proj,down_proj}"
    --ia3-feedforward-modules "${IA3_FEEDFORWARD_MODULES:-down_proj}"
    --ia3-checkpoint-dir "$CHECKPOINT_DIR"
    --ia3-checkpoint-steps "${IA3_CHECKPOINT_STEPS:-100}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
echo "IA3 GPU slice: $GPU_DEVICE"
echo "IA3 output: $OUTPUT_DIR"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_alternative_posttraining.py" \
    --stage prepare "${COMMON_ARGS[@]}" "$@"

env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_alternative_posttraining.py" \
    --stage ia3 "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$OUTPUT_DIR/ia3.log"
