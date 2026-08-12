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

GPU_DEVICE="${EMBEDDING_ADAPTER_GPU_DEVICE:-3}"
OUTPUT_DIR="${EMBEDDING_ADAPTER_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-residual_embedding_adapter-e10}"
HEAD_ONLY_RESULT="${HEAD_ONLY_QWEN_RESULT:-/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10/frozen/result.pt}"
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
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-8}"
    --embedding-adapter-epochs "${EMBEDDING_ADAPTER_EPOCHS:-10}"
    --embedding-adapter-batch-size "${EMBEDDING_ADAPTER_BATCH_SIZE:-16}"
    --embedding-adapter-eval-batch-size "${EMBEDDING_ADAPTER_EVAL_BATCH_SIZE:-8}"
    --embedding-adapter-bottleneck-dim "${EMBEDDING_ADAPTER_BOTTLENECK_DIM:-256}"
    --embedding-adapter-learning-rate "${EMBEDDING_ADAPTER_LEARNING_RATE:-2e-4}"
    --embedding-adapter-adapter-learning-rate "${EMBEDDING_ADAPTER_ADAPTER_LEARNING_RATE:-1e-5}"
    --embedding-adapter-weight-decay "${EMBEDDING_ADAPTER_WEIGHT_DECAY:-0.01}"
    --embedding-adapter-head-warmup-epochs "${EMBEDDING_ADAPTER_HEAD_WARMUP_EPOCHS:-3}"
    --embedding-adapter-max-grad-norm "${EMBEDDING_ADAPTER_MAX_GRAD_NORM:-0.1}"
    --embedding-adapter-dropout "${EMBEDDING_ADAPTER_DROPOUT:-0.05}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
echo "Residual embedding-adapter GPU slice: $GPU_DEVICE"
echo "Residual embedding-adapter output: $OUTPUT_DIR"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_alternative_posttraining.py" \
    --stage prepare "${COMMON_ARGS[@]}" "$@"

env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_alternative_posttraining.py" \
    --stage embedding-adapter "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$OUTPUT_DIR/embedding_adapter.log"

"${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/plot_qwen_posttraining.py" \
    --baseline-result-path "$HEAD_ONLY_RESULT" \
    --result-path "$OUTPUT_DIR/embedding_adapter/result.pt" \
    --output-dir "$OUTPUT_DIR" \
    --prefix qwen_embedding_adapter_comparison \
    --label "residual-embedding-adapter-Qwen+photo-z-head" \
    --summary-path "$OUTPUT_DIR/embedding_adapter_run.json"
