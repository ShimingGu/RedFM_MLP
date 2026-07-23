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

if ! "${PYTHON_CMD[@]}" -c 'import bitsandbytes, peft, torch, transformers' >/dev/null; then
    echo "The selected environment is missing QLoRA dependencies." >&2
    echo "Run 'pixi install' before starting this comparison." >&2
    exit 1
fi

if [[ "${QWEN_POOLING:-last}" != "last" ]]; then
    echo "This controlled comparison requires QWEN_POOLING=last." >&2
    exit 2
fi

DEVICE_LIST="${AION_QWEN_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
IFS=',' read -r -a GPU_IDS <<< "$DEVICE_LIST"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "Need two GPUs; set AION_QWEN_GPU_DEVICES to two comma-separated IDs." >&2
    exit 2
fi
FROZEN_GPU="${GPU_IDS[0]}"
QLORA_GPU="${GPU_IDS[1]}"

OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison}"
LOG_DIR="$OUTPUT_DIR/multigpu_logs"
mkdir -p -- "$LOG_DIR"

FLAGS=()
[[ "${QWEN_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-qwen-download)
[[ "${AION_FORCE_REBUILD_TOKENS:-0}" == 1 ]] && FLAGS+=(--force-rebuild-tokens)
[[ "${AION_FORCE_REBUILD_PHOTOMETRY:-0}" == 1 ]] && FLAGS+=(--force-rebuild-photometry)
[[ "${QWEN_FORCE_RECOMPUTE:-0}" == 1 ]] && FLAGS+=(--force-recompute-qwen)

COMMON_ARGS=(
    --catalogue "${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
    --morphology-dir "${AION_MORPHOLOGY_DIR:-$REPO_ROOT/data/clauds/images/tilesv5}"
    --output-dir "$OUTPUT_DIR"
    --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"
    --max-rows "${AION_MAX_ROWS:-300000}"
    --seed "${AION_SEED:-42}"
    --token-batch-size "${AION_TOKEN_BATCH_SIZE:-64}"
    --image-flux-scale "${AION_IMAGE_FLUX_SCALE:-1.0}"
    --min-cutout-weight-coverage "${AION_MIN_CUTOUT_WEIGHT_COVERAGE:-0.90}"
    --n-z-bins "${AION_N_Z_BINS:-300}"
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}"
    --qwen-model "${QWEN_MODEL:-Qwen3.5-4B-Base}"
    --qwen-max-length "${QWEN_MAX_LENGTH:-2048}"
    --qwen-pooling last
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-8}"
    --frozen-epochs "${FROZEN_EPOCHS:-10}"
    --frozen-train-batch-size "${FROZEN_TRAIN_BATCH_SIZE:-256}"
    --eval-batch-size "${POSTTRAIN_EVAL_BATCH_SIZE:-8}"
    --head-learning-rate "${HEAD_LEARNING_RATE:-2e-4}"
    --qlora-epochs "${QLORA_EPOCHS:-3}"
    --qlora-batch-size "${QLORA_BATCH_SIZE:-1}"
    --gradient-accumulation-steps "${QLORA_GRADIENT_ACCUMULATION_STEPS:-16}"
    --qlora-learning-rate "${QLORA_LEARNING_RATE:-2e-4}"
    --lora-rank "${QLORA_RANK:-8}"
    --lora-alpha "${QLORA_ALPHA:-16}"
    --lora-dropout "${QLORA_DROPOUT:-0.05}"
    --qlora-checkpoint-dir "${QLORA_CHECKPOINT_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/qlora_checkpoints/qwen-qwen_posttraining_comparison}"
    --qlora-checkpoint-steps "${QLORA_CHECKPOINT_STEPS:-100}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
echo "Preparing the shared seeded-random cohort on GPU $FROZEN_GPU."
env CUDA_VISIBLE_DEVICES="$FROZEN_GPU" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_posttraining_comparison.py" \
    --stage prepare "${COMMON_ARGS[@]}" "$@"

echo "Frozen-Qwen control GPU: $FROZEN_GPU"
echo "QLoRA post-training GPU:   $QLORA_GPU"
echo "Worker logs: $LOG_DIR/frozen.log and $LOG_DIR/qlora.log"

env CUDA_VISIBLE_DEVICES="$FROZEN_GPU" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_posttraining_comparison.py" \
    --stage frozen "${COMMON_ARGS[@]}" "$@" \
    >"$LOG_DIR/frozen.log" 2>&1 &
frozen_pid=$!

env CUDA_VISIBLE_DEVICES="$QLORA_GPU" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_posttraining_comparison.py" \
    --stage qlora "${COMMON_ARGS[@]}" "$@" \
    >"$LOG_DIR/qlora.log" 2>&1 &
qlora_pid=$!

cleanup_workers() {
    kill "$frozen_pid" "$qlora_pid" 2>/dev/null || true
}
trap cleanup_workers INT TERM

progress_seconds="${POSTTRAIN_PROGRESS_SECONDS:-30}"
while kill -0 "$frozen_pid" 2>/dev/null || kill -0 "$qlora_pid" 2>/dev/null; do
    sleep "$progress_seconds"
    echo "----- post-training progress -----"
    printf 'frozen: '
    tail -n 1 "$LOG_DIR/frozen.log" 2>/dev/null || echo "starting"
    printf 'qlora:  '
    tail -n 1 "$LOG_DIR/qlora.log" 2>/dev/null || echo "starting"
done

frozen_status=0
qlora_status=0
wait "$frozen_pid" || frozen_status=$?
wait "$qlora_pid" || qlora_status=$?
trap - INT TERM
if (( frozen_status != 0 || qlora_status != 0 )); then
    echo "Post-training comparison failed (frozen=$frozen_status, qlora=$qlora_status)." >&2
    echo "Inspect $LOG_DIR/frozen.log and $LOG_DIR/qlora.log." >&2
    exit 1
fi

echo "Both arms complete; collecting paired diagnostics."
env CUDA_VISIBLE_DEVICES="$FROZEN_GPU" PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_posttraining_comparison.py" \
    --stage collect "${COMMON_ARGS[@]}" "$@"
