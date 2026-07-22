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

if ! "${PYTHON_CMD[@]}" -c 'import aion_magnitude, timm, torch, transformers, bitsandbytes' >/dev/null; then
    echo "The selected environment is missing Qwen/timm dependencies." >&2
    echo "Install the vision and qwen-cuda dependencies before running this comparison." >&2
    exit 1
fi

DEVICE_LIST="${AION_QWEN_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
IFS=',' read -r -a GPU_IDS <<< "$DEVICE_LIST"
if (( ${#GPU_IDS[@]} < 2 )); then
    echo "Need two GPUs; set AION_QWEN_GPU_DEVICES to two comma-separated device IDs." >&2
    exit 2
fi
QWEN_GPU="${GPU_IDS[0]}"
TIMM_GPU="${GPU_IDS[1]}"

FLAGS=()
[[ "${QWEN_LOAD_IN_4BIT:-1}" == 0 ]] && FLAGS+=(--no-qwen-4bit)
[[ "${QWEN_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-qwen-download)
[[ "${QWEN_NORMALIZE:-0}" == 1 ]] && FLAGS+=(--qwen-normalize)
[[ "${AION_FORCE_REBUILD_TOKENS:-0}" == 1 ]] && FLAGS+=(--force-rebuild-tokens)
[[ "${AION_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == 1 ]] && FLAGS+=(--force-rebuild-photometry --force-recompute-qwen)
[[ "${TIMM_FORCE_RECOMPUTE:-0}" == 1 ]] && FLAGS+=(--force-recompute-timm)
[[ "${TIMM_PRETRAINED:-1}" == 0 ]] && FLAGS+=(--no-timm-pretrained)
[[ -n "${QWEN_CACHE_PATH:-}" ]] && FLAGS+=(--qwen-cache-path "$QWEN_CACHE_PATH")
[[ -n "${TIMM_CACHE_PATH:-}" ]] && FLAGS+=(--timm-cache-path "$TIMM_CACHE_PATH")

OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen_aion-timm_raw_comparison}"
echo "Qwen magnitude encoder GPU: $QWEN_GPU"
echo "timm raw-image encoder GPU: $TIMM_GPU"
echo "output: $OUTPUT_DIR"

cd -- "$REPO_ROOT"
exec env \
    CUDA_VISIBLE_DEVICES="$QWEN_GPU" \
    AION_TIMM_GPU_DEVICE="$TIMM_GPU" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_aion_timm_raw_comparison.py" \
    --catalogue "${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}" \
    --morphology-dir "${AION_MORPHOLOGY_DIR:-$REPO_ROOT/data/clauds/images/tilesv5/}" \
    --output-dir "$OUTPUT_DIR" \
    --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}" \
    --max-rows "${AION_MAX_ROWS:-300000}" \
    --epochs "${AION_EPOCHS:-10}" \
    --token-batch-size "${AION_TOKEN_BATCH_SIZE:-64}" \
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-8}" \
    --timm-batch-size "${TIMM_EMBEDDING_BATCH_SIZE:-64}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
    --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --device cuda \
    --n-z-bins "${AION_N_Z_BINS:-300}" \
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
    --image-flux-scale "${AION_IMAGE_FLUX_SCALE:-1.0}" \
    --min-cutout-weight-coverage "${AION_MIN_CUTOUT_WEIGHT_COVERAGE:-0.90}" \
    --feature-scaling minmax \
    --qwen-model "${QWEN_MODEL:-Qwen3.5-4B-Base}" \
    --qwen-max-length "${QWEN_MAX_LENGTH:-2048}" \
    --qwen-pooling "${QWEN_POOLING:-last}" \
    --timm-model "${TIMM_MODEL:-hf-hub:timm/convnext_tiny.dinov3_lvd1689m}" \
    --timm-input-size "${TIMM_INPUT_SIZE:-224}" \
    --timm-percentile "${TIMM_PERCENTILE:-99}" \
    "${FLAGS[@]}" \
    "$@"
