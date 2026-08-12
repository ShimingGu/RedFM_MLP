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

GPU_DEVICE="${AION_RLVR_GPU_DEVICE:-3}"
OUTPUT_ROOT="${AION_RLVR_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/aion-rlvr-e1}"
read -r -a INPUT_MODES <<< "${AION_INPUT_MODES:-photometry photometry-images}"

FLAGS=()
[[ "${AION_FORCE_RECOMPUTE_PRODUCT:-0}" == 1 ]] && FLAGS+=(--force-recompute-product)
[[ "${AION_RLVR_RESUME:-1}" == 0 ]] && FLAGS+=(--no-rlvr-resume)

COMMON_ARGS=(
    --catalogue "${AION_CATALOGUE:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
    --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"
    --hsc-image-dir "${AION_HSC_IMAGE_DIR:-/arc/projects/ots/pdr3_dud}"
    --image-assignment-cache-dir "${AION_IMAGE_ASSIGNMENT_CACHE_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/cache/aion_multiband_morphology_catalogue_updated}"
    --fits-backend "${AION_FITS_BACKEND:-auto}"
    --min-cutout-coverage "${AION_MIN_CUTOUT_COVERAGE:-0.90}"
    --max-rows "${AION_MAX_ROWS:-300000}"
    --sample-mode "${AION_SAMPLE_MODE:-head}"
    --sample-seed "${AION_SAMPLE_SEED:-42}"
    --seed "${AION_SEED:-42}"
    --train-fraction "${AION_TRAIN_FRACTION:-0.20}"
    --test-fraction "${AION_TEST_FRACTION:-0.75}"
    --val-fraction "${AION_VAL_FRACTION:-0.05}"
    --n-z-bins "${AION_N_Z_BINS:-300}"
    --embedding-batch-size "${AION_EMBEDDING_BATCH_SIZE:-512}"
    --image-embedding-batch-size "${AION_IMAGE_EMBEDDING_BATCH_SIZE:-8}"
    --head-only-epochs "${AION_HEAD_ONLY_EPOCHS:-10}"
    --head-only-batch-size "${AION_HEAD_ONLY_BATCH_SIZE:-256}"
    --head-only-eval-batch-size "${AION_HEAD_ONLY_EVAL_BATCH_SIZE:-512}"
    --embedding-adapter-epochs "${AION_ADAPTER_EPOCHS:-10}"
    --embedding-adapter-batch-size "${AION_ADAPTER_BATCH_SIZE:-256}"
    --embedding-adapter-eval-batch-size "${AION_ADAPTER_EVAL_BATCH_SIZE:-512}"
    --embedding-adapter-bottleneck-dim "${AION_ADAPTER_BOTTLENECK_DIM:-256}"
    --embedding-adapter-learning-rate "${AION_ADAPTER_HEAD_LEARNING_RATE:-2e-4}"
    --embedding-adapter-adapter-learning-rate "${AION_ADAPTER_LEARNING_RATE:-1e-5}"
    --embedding-adapter-weight-decay "${AION_ADAPTER_WEIGHT_DECAY:-0.01}"
    --embedding-adapter-head-warmup-epochs "${AION_ADAPTER_HEAD_WARMUP_EPOCHS:-3}"
    --embedding-adapter-max-grad-norm "${AION_ADAPTER_MAX_GRAD_NORM:-0.1}"
    --embedding-adapter-dropout "${AION_ADAPTER_DROPOUT:-0.05}"
    --rlvr-epochs "${AION_RLVR_EPOCHS:-1}"
    --rlvr-batch-size "${AION_RLVR_BATCH_SIZE:-64}"
    --rlvr-eval-batch-size "${AION_RLVR_EVAL_BATCH_SIZE:-512}"
    --rlvr-gradient-accumulation-steps "${AION_RLVR_GRADIENT_ACCUMULATION_STEPS:-4}"
    --rlvr-head-learning-rate "${AION_RLVR_HEAD_LEARNING_RATE:-1e-5}"
    --rlvr-adapter-learning-rate "${AION_RLVR_ADAPTER_LEARNING_RATE:-1e-6}"
    --rlvr-group-samples "${AION_RLVR_GROUP_SAMPLES:-8}"
    --rlvr-reward-scale "${AION_RLVR_REWARD_SCALE:-0.05}"
    --rlvr-outlier-threshold "${AION_RLVR_OUTLIER_THRESHOLD:-0.15}"
    --rlvr-outlier-penalty "${AION_RLVR_OUTLIER_PENALTY:-0.5}"
    --rlvr-kl-beta "${AION_RLVR_KL_BETA:-0.02}"
    --rlvr-entropy-coefficient "${AION_RLVR_ENTROPY_COEFFICIENT:-0.001}"
    --rlvr-checkpoint-steps "${AION_RLVR_CHECKPOINT_STEPS:-100}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
for INPUT_MODE in "${INPUT_MODES[@]}"; do
    MODE_TAG="${INPUT_MODE//-/_}"
    OUTPUT_DIR="$OUTPUT_ROOT/$MODE_TAG"
    mkdir -p -- "$OUTPUT_DIR"
    echo "AION RLVR GPU slice: $GPU_DEVICE"
    echo "AION input mode: $INPUT_MODE"
    echo "AION output: $OUTPUT_DIR"

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
        --stage prepare --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
        "${COMMON_ARGS[@]}" "$@" \
        2>&1 | tee "$OUTPUT_DIR/prepare.log"

    if [[ ! -f "$OUTPUT_DIR/head_only/result.pt" || "${AION_RETRAIN_HEAD_ONLY:-0}" == 1 ]]; then
        env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
            "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
            --stage head-only --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
            "${COMMON_ARGS[@]}" "$@" \
            2>&1 | tee "$OUTPUT_DIR/head_only.log"
    fi

    if [[ ! -f "$OUTPUT_DIR/embedding_adapter/result.pt" || "${AION_RETRAIN_SUPERVISED_ADAPTER:-0}" == 1 ]]; then
        env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
            "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
            --stage embedding-adapter --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
            "${COMMON_ARGS[@]}" "$@" \
            2>&1 | tee "$OUTPUT_DIR/embedding_adapter.log"
    fi

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
        --stage rlvr --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
        --rlvr-checkpoint-dir "$OUTPUT_DIR/rlvr_checkpoints" \
        "${COMMON_ARGS[@]}" "$@" \
        2>&1 | tee "$OUTPUT_DIR/rlvr.log"

    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/plot_aion_posttraining.py" \
        --baseline-result-path "$OUTPUT_DIR/head_only/result.pt" \
        --result-path "$OUTPUT_DIR/rlvr/result.pt" \
        --output-dir "$OUTPUT_DIR" \
        --prefix "aion_${MODE_TAG}_rlvr_comparison" \
        --baseline-label "mean-pooled AION head control ($INPUT_MODE)" \
        --label "post-encoder vector adapter + RLVR control ($INPUT_MODE)" \
        --summary-path "$OUTPUT_DIR/rlvr_run.json"
done
