#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
METHOD="${1:-}"
if [[ -z "$METHOD" ]]; then
    echo "Usage: $0 {qlora|dora|ia3|embedding-adapter|rlvr} [driver options]" >&2
    exit 2
fi
shift
case "$METHOD" in
    qlora|dora|ia3|embedding-adapter|rlvr) ;;
    *) echo "Unknown GLM post-training method: $METHOD" >&2; exit 2 ;;
esac

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

if ! "${PYTHON_CMD[@]}" -c 'import accelerate, bitsandbytes, peft, torch, transformers, aion_magnitude' >/dev/null; then
    echo "The selected environment is missing GLM post-training dependencies." >&2
    exit 1
fi

GPU_DEVICE="${GLM52_GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_ROOT="${GLM52_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/glm52-posttraining-e10}"
CHECKPOINT_ROOT="${GLM52_CHECKPOINT_ROOT:-/arc/projects/ots/Cosmic_Imprint_of_Time/glm52_posttraining_checkpoints}"
INPUT_MODES="${GLM52_INPUT_MODES:-photometry photometry-morphology}"
MODEL="${GLM52_MODEL:-/arc/home/gsm/hf_models/GLM-5.2-0.8B-A0.8B}"
CATALOGUE="${AION_CATALOGUE:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits}"

FLAGS=()
[[ "${GLM52_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-glm-download)
[[ "${GLM52_FORCE_RECOMPUTE:-0}" == 1 ]] && FLAGS+=(--force-recompute-glm)

COMMON_ARGS=(
    --catalogue "$CATALOGUE"
    --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"
    --max-rows "${AION_MAX_ROWS:-300000}"
    --seed "${AION_SEED:-42}"
    --train-fraction "${AION_TRAIN_FRACTION:-0.20}"
    --test-fraction "${AION_TEST_FRACTION:-0.75}"
    --val-fraction "${AION_VAL_FRACTION:-0.05}"
    --n-z-bins "${AION_N_Z_BINS:-300}"
    --glm-model "$MODEL"
    --glm-max-length "${GLM52_MAX_LENGTH:-512}"
    --glm-embedding-batch-size "${GLM52_EMBEDDING_BATCH_SIZE:-8}"
    --frozen-epochs "${GLM52_FROZEN_EPOCHS:-10}"
    --frozen-train-batch-size "${GLM52_FROZEN_TRAIN_BATCH_SIZE:-256}"
    --eval-batch-size "${GLM52_EVAL_BATCH_SIZE:-8}"
    --head-learning-rate "${GLM52_HEAD_LEARNING_RATE:-2e-4}"
    --qlora-epochs "${GLM52_QLORA_EPOCHS:-10}"
    --qlora-batch-size "${GLM52_QLORA_BATCH_SIZE:-1}"
    --gradient-accumulation-steps "${GLM52_QLORA_GRADIENT_ACCUMULATION_STEPS:-16}"
    --qlora-learning-rate "${GLM52_QLORA_LEARNING_RATE:-1e-5}"
    --head-warmup-epochs "${GLM52_QLORA_HEAD_WARMUP_EPOCHS:-3}"
    --lora-max-grad-norm "${GLM52_QLORA_MAX_GRAD_NORM:-0.1}"
    --lora-rank "${GLM52_LORA_RANK:-8}"
    --lora-alpha "${GLM52_LORA_ALPHA:-16}"
    --lora-dropout "${GLM52_LORA_DROPOUT:-0.05}"
    "${FLAGS[@]}"
)

cd -- "$REPO_ROOT"
for input_mode in $INPUT_MODES; do
    MODE_FLAGS=()
    case "$input_mode" in
        photometry)
            input_tag="photometry"
            ;;
        photometry-morphology)
            input_tag="photometry_morphology"
            MODE_FLAGS+=(--use-morphology)
            ;;
        *)
            echo "Unknown GLM52_INPUT_MODES entry: $input_mode" >&2
            exit 2
            ;;
    esac

    MODE_OUTPUT="$OUTPUT_ROOT/$input_tag"
    MODE_CHECKPOINT="$CHECKPOINT_ROOT/$input_tag/$METHOD"
    mkdir -p -- "$MODE_OUTPUT" "$MODE_CHECKPOINT"
    ARGS=(
        --output-dir "$MODE_OUTPUT"
        "${COMMON_ARGS[@]}"
        "${MODE_FLAGS[@]}"
    )

    echo "GLM method: $METHOD; input: $input_mode; GPU slice: $GPU_DEVICE"
    echo "GLM output: $MODE_OUTPUT"

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/glm52_posttraining.py" \
        --stage prepare "${ARGS[@]}" "$@"

    (
        flock -x 9
        if [[ "${GLM52_FORCE_FROZEN:-0}" == 1 || ! -f "$MODE_OUTPUT/head_only/result.pt" ]]; then
            env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
                "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/glm52_posttraining.py" \
                --stage frozen "${ARGS[@]}" "$@" \
                2>&1 | tee "$MODE_OUTPUT/head_only.log"
        else
            echo "Reusing matched GLM-5.2 head-only control: $MODE_OUTPUT/head_only/result.pt"
        fi
    ) 9>"$MODE_OUTPUT/.head_only.lock"

    METHOD_ARGS=()
    case "$METHOD" in
        qlora)
            METHOD_ARGS=(
                --qlora-checkpoint-dir "$MODE_CHECKPOINT"
                --qlora-checkpoint-steps "${GLM52_QLORA_CHECKPOINT_STEPS:-100}"
            )
            [[ "${GLM52_QLORA_RESUME:-1}" == 1 ]] || METHOD_ARGS+=(--no-qlora-resume)
            result_dir="qlora"
            label="GLM-5.2-QLoRA"
            ;;
        dora)
            METHOD_ARGS=(
                --dora-epochs "${GLM52_DORA_EPOCHS:-10}"
                --dora-batch-size "${GLM52_DORA_BATCH_SIZE:-1}"
                --dora-gradient-accumulation-steps "${GLM52_DORA_GRADIENT_ACCUMULATION_STEPS:-16}"
                --dora-learning-rate "${GLM52_DORA_LEARNING_RATE:-1e-5}"
                --dora-head-warmup-epochs "${GLM52_DORA_HEAD_WARMUP_EPOCHS:-3}"
                --dora-max-grad-norm "${GLM52_DORA_MAX_GRAD_NORM:-0.1}"
                --dora-rank "${GLM52_DORA_RANK:-8}"
                --dora-alpha "${GLM52_DORA_ALPHA:-16}"
                --dora-dropout "${GLM52_DORA_DROPOUT:-0.05}"
                --dora-checkpoint-dir "$MODE_CHECKPOINT"
                --dora-checkpoint-steps "${GLM52_DORA_CHECKPOINT_STEPS:-100}"
            )
            [[ "${GLM52_DORA_RESUME:-1}" == 1 ]] || METHOD_ARGS+=(--no-dora-resume)
            result_dir="dora"
            label="GLM-5.2-DoRA"
            ;;
        ia3)
            METHOD_ARGS=(
                --ia3-epochs "${GLM52_IA3_EPOCHS:-10}"
                --ia3-batch-size "${GLM52_IA3_BATCH_SIZE:-1}"
                --ia3-gradient-accumulation-steps "${GLM52_IA3_GRADIENT_ACCUMULATION_STEPS:-16}"
                --ia3-learning-rate "${GLM52_IA3_LEARNING_RATE:-1e-5}"
                --ia3-head-warmup-epochs "${GLM52_IA3_HEAD_WARMUP_EPOCHS:-3}"
                --ia3-max-grad-norm "${GLM52_IA3_MAX_GRAD_NORM:-0.1}"
                --ia3-checkpoint-dir "$MODE_CHECKPOINT"
                --ia3-checkpoint-steps "${GLM52_IA3_CHECKPOINT_STEPS:-100}"
            )
            [[ "${GLM52_IA3_RESUME:-1}" == 1 ]] || METHOD_ARGS+=(--no-ia3-resume)
            result_dir="ia3"
            label="GLM-5.2-IA3"
            ;;
        embedding-adapter)
            METHOD_ARGS=(
                --embedding-adapter-epochs "${GLM52_ADAPTER_EPOCHS:-10}"
                --embedding-adapter-batch-size "${GLM52_ADAPTER_BATCH_SIZE:-16}"
                --embedding-adapter-eval-batch-size "${GLM52_ADAPTER_EVAL_BATCH_SIZE:-8}"
                --embedding-adapter-bottleneck-dim "${GLM52_ADAPTER_BOTTLENECK_DIM:-256}"
                --embedding-adapter-learning-rate "${GLM52_ADAPTER_HEAD_LEARNING_RATE:-2e-4}"
                --embedding-adapter-adapter-learning-rate "${GLM52_ADAPTER_LEARNING_RATE:-1e-5}"
                --embedding-adapter-head-warmup-epochs "${GLM52_ADAPTER_HEAD_WARMUP_EPOCHS:-3}"
                --embedding-adapter-max-grad-norm "${GLM52_ADAPTER_MAX_GRAD_NORM:-0.1}"
                --embedding-adapter-dropout "${GLM52_ADAPTER_DROPOUT:-0.05}"
            )
            result_dir="embedding_adapter"
            label="GLM-5.2-residual-embedding-adapter"
            ;;
        rlvr)
            source_dir="$MODE_OUTPUT/qlora"
            if [[ ! -d "$source_dir/adapter" || ! -f "$source_dir/photoz_head.pt" ]]; then
                echo "RLVR requires the completed GLM QLoRA arm first: $source_dir" >&2
                exit 2
            fi
            METHOD_ARGS=(
                --rlvr-epochs "${GLM52_RLVR_EPOCHS:-1}"
                --rlvr-batch-size "${GLM52_RLVR_BATCH_SIZE:-1}"
                --rlvr-gradient-accumulation-steps "${GLM52_RLVR_GRADIENT_ACCUMULATION_STEPS:-16}"
                --rlvr-head-learning-rate "${GLM52_RLVR_HEAD_LEARNING_RATE:-1e-5}"
                --rlvr-adapter-learning-rate "${GLM52_RLVR_ADAPTER_LEARNING_RATE:-1e-6}"
                --rlvr-group-samples "${GLM52_RLVR_GROUP_SAMPLES:-8}"
                --rlvr-reward-scale "${GLM52_RLVR_REWARD_SCALE:-0.05}"
                --rlvr-outlier-threshold "${GLM52_RLVR_OUTLIER_THRESHOLD:-0.15}"
                --rlvr-outlier-penalty "${GLM52_RLVR_OUTLIER_PENALTY:-0.5}"
                --rlvr-kl-beta "${GLM52_RLVR_KL_BETA:-0.02}"
                --rlvr-entropy-coefficient "${GLM52_RLVR_ENTROPY_COEFFICIENT:-0.001}"
                --rlvr-source-adapter-dir "$source_dir/adapter"
                --rlvr-source-head-path "$source_dir/photoz_head.pt"
                --rlvr-checkpoint-dir "$MODE_CHECKPOINT"
                --rlvr-checkpoint-steps "${GLM52_RLVR_CHECKPOINT_STEPS:-100}"
            )
            [[ "${GLM52_RLVR_RESUME:-1}" == 1 ]] || METHOD_ARGS+=(--no-rlvr-resume)
            result_dir="rlvr"
            label="GLM-5.2-QLoRA+RLVR"
            ;;
    esac

    env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" PYTHONUNBUFFERED=1 \
        "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/glm52_posttraining.py" \
        --stage "$METHOD" "${ARGS[@]}" "${METHOD_ARGS[@]}" "$@" \
        2>&1 | tee "$MODE_OUTPUT/$result_dir.log"

    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/plot_qwen_posttraining.py" \
        --baseline-result-path "$MODE_OUTPUT/head_only/result.pt" \
        --result-path "$MODE_OUTPUT/$result_dir/result.pt" \
        --output-dir "$MODE_OUTPUT" \
        --prefix "glm52_${input_tag}_${result_dir}_comparison" \
        --baseline-label "GLM-5.2-head-only" \
        --label "$label" \
        --summary-path "$MODE_OUTPUT/${result_dir}_run.json" \
        --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-10}"
done


