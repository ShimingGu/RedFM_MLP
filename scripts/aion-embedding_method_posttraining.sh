#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
METHOD="${AION_EMBEDDING_METHOD:?Set AION_EMBEDDING_METHOD to ia3, dora, or qlora}"

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

case "$METHOD" in
    ia3)
        METHOD_LABEL="IA3"
        GPU_DEVICE="${AION_IA3_GPU_DEVICE:-3}"
        OUTPUT_ROOT="${AION_IA3_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/aion-ia3-e10}"
        METHOD_ARGS=(
            --ia3-epochs "${AION_IA3_EPOCHS:-10}"
            --ia3-batch-size "${AION_IA3_BATCH_SIZE:-1}"
            --ia3-eval-batch-size "${AION_IA3_EVAL_BATCH_SIZE:-2}"
            --ia3-gradient-accumulation-steps "${AION_IA3_GRADIENT_ACCUMULATION_STEPS:-16}"
            --ia3-head-learning-rate "${AION_IA3_HEAD_LEARNING_RATE:-2e-4}"
            --ia3-learning-rate "${AION_IA3_LEARNING_RATE:-1e-5}"
            --ia3-weight-decay "${AION_IA3_WEIGHT_DECAY:-0.01}"
            --ia3-head-warmup-epochs "${AION_IA3_HEAD_WARMUP_EPOCHS:-1}"
            --ia3-max-grad-norm "${AION_IA3_MAX_GRAD_NORM:-0.1}"
        )
        ;;
    dora)
        METHOD_LABEL="DoRA"
        GPU_DEVICE="${AION_DORA_GPU_DEVICE:-3}"
        OUTPUT_ROOT="${AION_DORA_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/aion-dora-e10}"
        METHOD_ARGS=(
            --dora-epochs "${AION_DORA_EPOCHS:-10}"
            --dora-batch-size "${AION_DORA_BATCH_SIZE:-1}"
            --dora-eval-batch-size "${AION_DORA_EVAL_BATCH_SIZE:-2}"
            --dora-gradient-accumulation-steps "${AION_DORA_GRADIENT_ACCUMULATION_STEPS:-16}"
            --dora-head-learning-rate "${AION_DORA_HEAD_LEARNING_RATE:-2e-4}"
            --dora-learning-rate "${AION_DORA_LEARNING_RATE:-1e-5}"
            --dora-weight-decay "${AION_DORA_WEIGHT_DECAY:-0.01}"
            --dora-head-warmup-epochs "${AION_DORA_HEAD_WARMUP_EPOCHS:-1}"
            --dora-max-grad-norm "${AION_DORA_MAX_GRAD_NORM:-0.1}"
            --dora-rank "${AION_DORA_RANK:-8}"
            --dora-alpha "${AION_DORA_ALPHA:-16}"
            --dora-dropout "${AION_DORA_DROPOUT:-0.05}"
        )
        ;;
    qlora)
        METHOD_LABEL="QLoRA"
        GPU_DEVICE="${AION_QLORA_GPU_DEVICE:-3}"
        OUTPUT_ROOT="${AION_QLORA_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/aion-qlora-e10}"
        METHOD_ARGS=(
            --qlora-epochs "${AION_QLORA_EPOCHS:-10}"
            --qlora-batch-size "${AION_QLORA_BATCH_SIZE:-1}"
            --qlora-eval-batch-size "${AION_QLORA_EVAL_BATCH_SIZE:-2}"
            --qlora-gradient-accumulation-steps "${AION_QLORA_GRADIENT_ACCUMULATION_STEPS:-16}"
            --qlora-head-learning-rate "${AION_QLORA_HEAD_LEARNING_RATE:-2e-4}"
            --qlora-learning-rate "${AION_QLORA_LEARNING_RATE:-1e-5}"
            --qlora-weight-decay "${AION_QLORA_WEIGHT_DECAY:-0.01}"
            --qlora-head-warmup-epochs "${AION_QLORA_HEAD_WARMUP_EPOCHS:-1}"
            --qlora-max-grad-norm "${AION_QLORA_MAX_GRAD_NORM:-0.1}"
            --qlora-rank "${AION_QLORA_RANK:-8}"
            --qlora-alpha "${AION_QLORA_ALPHA:-16}"
            --qlora-dropout "${AION_QLORA_DROPOUT:-0.05}"
            --qlora-quantization-block-size "${AION_QLORA_QUANTIZATION_BLOCK_SIZE:-64}"
        )
        ;;
    *)
        echo "Unsupported AION embedding method: $METHOD" >&2
        exit 2
        ;;
esac

read -r -a INPUT_MODES <<< "${AION_INPUT_MODES:-photometry photometry-images}"
FLAGS=()
[[ "${AION_FORCE_RECOMPUTE_PRODUCT:-0}" == 1 ]] && FLAGS+=(--force-recompute-product)
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
    --encoder-head-epochs "${AION_ENCODER_HEAD_EPOCHS:-10}"
    --encoder-head-batch-size "${AION_ENCODER_HEAD_BATCH_SIZE:-1}"
    --encoder-head-eval-batch-size "${AION_ENCODER_HEAD_EVAL_BATCH_SIZE:-2}"
    --encoder-head-gradient-accumulation-steps "${AION_ENCODER_HEAD_GRADIENT_ACCUMULATION_STEPS:-16}"
    --encoder-head-learning-rate "${AION_ENCODER_HEAD_LEARNING_RATE:-2e-4}"
    --encoder-head-weight-decay "${AION_ENCODER_HEAD_WEIGHT_DECAY:-0.01}"
    "${METHOD_ARGS[@]}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
for INPUT_MODE in "${INPUT_MODES[@]}"; do
    MODE_TAG="${INPUT_MODE//-/_}"
    OUTPUT_DIR="$OUTPUT_ROOT/$MODE_TAG"
    mkdir -p -- "$OUTPUT_DIR"
    echo "AION $METHOD_LABEL GPU slice: $GPU_DEVICE"
    echo "AION input mode: $INPUT_MODE"
    echo "AION output: $OUTPUT_DIR"

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
        --stage prepare --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
        "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/prepare.log"

    if [[ ! -f "$OUTPUT_DIR/attentive_head_only/result.pt" || "${AION_RETRAIN_ENCODER_HEAD:-0}" == 1 ]]; then
        env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
            "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
            --stage attentive-head-only --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
            "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/attentive_head_only.log"
    fi

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_posttraining.py" \
        --stage "$METHOD" --input-mode "$INPUT_MODE" --output-dir "$OUTPUT_DIR" \
        "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/$METHOD.log"

    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/plot_aion_posttraining.py" \
        --baseline-result-path "$OUTPUT_DIR/attentive_head_only/result.pt" \
        --result-path "$OUTPUT_DIR/$METHOD/result.pt" \
        --output-dir "$OUTPUT_DIR" \
        --prefix "aion_${MODE_TAG}_${METHOD}_comparison" \
        --baseline-label "frozen AION + attentive head ($INPUT_MODE)" \
        --label "$METHOD_LABEL AION encoder ($INPUT_MODE)" \
        --summary-path "$OUTPUT_DIR/${METHOD}_run.json"
done
