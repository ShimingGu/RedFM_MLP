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

GPU_DEVICE="${RLVR_GPU_DEVICE:-3}"
OUTPUT_DIR="${RLVR_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-qlora_rlvr-e1}"
HEAD_ONLY_RESULT="${HEAD_ONLY_QWEN_RESULT:-/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10/frozen/result.pt}"
CHECKPOINT_DIR="${RLVR_CHECKPOINT_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/rlvr_checkpoints/qwen-qlora-rlvr-e1}"
SOURCE_DIR="${RLVR_SOURCE_DIR:-/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10/qlora}"
SOURCE_ADAPTER="${RLVR_SOURCE_ADAPTER_DIR:-$SOURCE_DIR/adapter}"
SOURCE_HEAD="${RLVR_SOURCE_HEAD_PATH:-$SOURCE_DIR/photoz_head.pt}"
mkdir -p -- "$OUTPUT_DIR"

if [[ ! -d "$SOURCE_ADAPTER" || ! -f "$SOURCE_HEAD" ]]; then
    echo "Completed QLoRA-SFT adapter/head not found under: $SOURCE_DIR" >&2
    exit 2
fi

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
    --rlvr-epochs "${RLVR_EPOCHS:-1}"
    --rlvr-batch-size "${RLVR_BATCH_SIZE:-1}"
    --rlvr-gradient-accumulation-steps "${RLVR_GRADIENT_ACCUMULATION_STEPS:-16}"
    --rlvr-head-learning-rate "${RLVR_HEAD_LEARNING_RATE:-1e-5}"
    --rlvr-adapter-learning-rate "${RLVR_ADAPTER_LEARNING_RATE:-1e-6}"
    --rlvr-group-samples "${RLVR_GROUP_SAMPLES:-8}"
    --rlvr-reward-scale "${RLVR_REWARD_SCALE:-0.05}"
    --rlvr-outlier-threshold "${RLVR_OUTLIER_THRESHOLD:-0.15}"
    --rlvr-outlier-penalty "${RLVR_OUTLIER_PENALTY:-0.5}"
    --rlvr-kl-beta "${RLVR_KL_BETA:-0.02}"
    --rlvr-entropy-coefficient "${RLVR_ENTROPY_COEFFICIENT:-0.001}"
    --rlvr-source-adapter-dir "$SOURCE_ADAPTER"
    --rlvr-source-head-path "$SOURCE_HEAD"
    --rlvr-checkpoint-dir "$CHECKPOINT_DIR"
    --rlvr-checkpoint-steps "${RLVR_CHECKPOINT_STEPS:-100}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
echo "RLVR GPU slice: $GPU_DEVICE"
echo "RLVR source SFT policy: $SOURCE_DIR"
echo "RLVR output: $OUTPUT_DIR"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_dora_rlvr_posttraining.py" \
    --stage prepare "${COMMON_ARGS[@]}" "$@"

env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_dora_rlvr_posttraining.py" \
    --stage rlvr "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$OUTPUT_DIR/rlvr.log"

"${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/plot_qwen_posttraining.py" \
    --baseline-result-path "$HEAD_ONLY_RESULT" \
    --result-path "$OUTPUT_DIR/rlvr/result.pt" \
    --output-dir "$OUTPUT_DIR" \
    --prefix qwen_rlvr_comparison \
    --label "QLoRA+RLVR-Qwen+photo-z-head" \
    --summary-path "$OUTPUT_DIR/rlvr_run.json"
