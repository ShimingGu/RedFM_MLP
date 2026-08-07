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

if ! "${PYTHON_CMD[@]}" -c 'from peft import LoraConfig; assert "use_dora" in LoraConfig.__dataclass_fields__' >/dev/null; then
    echo "The selected environment does not support DoRA." >&2
    exit 1
fi

GPU_DEVICE="${DORA_GPU_DEVICE:-2}"
OUTPUT_DIR="${DORA_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-dora_posttraining-e10}"
CHECKPOINT_DIR="${DORA_CHECKPOINT_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/dora_checkpoints/qwen-dora-posttraining-e10}"
mkdir -p -- "$OUTPUT_DIR"

FLAGS=()
[[ "${QWEN_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-qwen-download)
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
    --dora-epochs "${DORA_EPOCHS:-10}"
    --dora-batch-size "${DORA_BATCH_SIZE:-1}"
    --dora-gradient-accumulation-steps "${DORA_GRADIENT_ACCUMULATION_STEPS:-16}"
    --dora-learning-rate "${DORA_LEARNING_RATE:-1e-5}"
    --dora-head-warmup-epochs "${DORA_HEAD_WARMUP_EPOCHS:-3}"
    --dora-max-grad-norm "${DORA_MAX_GRAD_NORM:-0.1}"
    --dora-rank "${DORA_RANK:-8}"
    --dora-alpha "${DORA_ALPHA:-16}"
    --dora-dropout "${DORA_DROPOUT:-0.05}"
    --dora-target-modules "${DORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
    --dora-checkpoint-dir "$CHECKPOINT_DIR"
    --dora-checkpoint-steps "${DORA_CHECKPOINT_STEPS:-100}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
echo "DoRA GPU slice: $GPU_DEVICE"
echo "DoRA output: $OUTPUT_DIR"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_dora_rlvr_posttraining.py" \
    --stage prepare "${COMMON_ARGS[@]}" "$@"

env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_dora_rlvr_posttraining.py" \
    --stage dora "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$OUTPUT_DIR/dora.log"
