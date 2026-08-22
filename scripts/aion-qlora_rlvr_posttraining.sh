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

GPU_DEVICE="${AION_QLORA_RLVR_GPU_DEVICE:-${AION_RLVR_GPU_DEVICE:-0}}"
SOURCE_ROOT="${AION_QLORA_RLVR_SOURCE_DIR:-/arc/home/gsm/aion_output/figures/aion-qlora-e10}"
OUTPUT_ROOT="${AION_QLORA_RLVR_OUTPUT_DIR:-${AION_RLVR_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/aion-qlora_rlvr-e1}}"
CHECKPOINT_ROOT="${AION_QLORA_RLVR_CHECKPOINT_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/rlvr_checkpoints/aion-qlora-rlvr-e1}"
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
    --rlvr-epochs "${AION_RLVR_EPOCHS:-1}"
    --rlvr-batch-size "${AION_RLVR_BATCH_SIZE:-1}"
    --rlvr-eval-batch-size "${AION_RLVR_EVAL_BATCH_SIZE:-2}"
    --rlvr-gradient-accumulation-steps "${AION_RLVR_GRADIENT_ACCUMULATION_STEPS:-16}"
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
    SOURCE_MODE_DIR="$SOURCE_ROOT/$MODE_TAG"
    SOURCE_RESULT="$SOURCE_MODE_DIR/qlora/result.pt"
    BASELINE_RESULT="$SOURCE_MODE_DIR/attentive_head_only/result.pt"
    OUTPUT_DIR="$OUTPUT_ROOT/$MODE_TAG"
    CHECKPOINT_DIR="$CHECKPOINT_ROOT/$MODE_TAG"
    if [[ ! -f "$SOURCE_RESULT" ]]; then
        echo "Completed encoder-level AION QLoRA result not found: $SOURCE_RESULT" >&2
        exit 2
    fi
    if [[ ! -f "$BASELINE_RESULT" ]]; then
        echo "Matched frozen AION baseline not found: $BASELINE_RESULT" >&2
        exit 2
    fi
    mkdir -p -- "$OUTPUT_DIR"
    echo "AION QLoRA+RLVR GPU slice: $GPU_DEVICE"
    echo "AION input mode: $INPUT_MODE"
    echo "AION supervised QLoRA source: $SOURCE_RESULT"
    echo "AION QLoRA+RLVR output: $OUTPUT_DIR"

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
        --stage prepare --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
        "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/prepare.log"

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_qlora_rlvr_posttraining.py" \
        --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
        --rlvr-source-result-path "$SOURCE_RESULT" \
        --rlvr-checkpoint-dir "$CHECKPOINT_DIR" \
        "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/rlvr.log"

    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/plot_aion_posttraining.py" \
        --baseline-result-path "$BASELINE_RESULT" \
        --result-path "$OUTPUT_DIR/rlvr/result.pt" \
        --output-dir "$OUTPUT_DIR" \
        --prefix "aion_${MODE_TAG}_qlora_rlvr_comparison" \
        --baseline-label "frozen AION + attentive head ($INPUT_MODE)" \
        --label "AION encoder QLoRA+RLVR ($INPUT_MODE)" \
        --summary-path "$OUTPUT_DIR/rlvr_run.json"
done
