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
    echo "No project Python environment found. Expected $REPO_ROOT/pixi.toml or $REPO_ROOT/.venv/bin/python." >&2
    exit 1
fi

AION_CATALOGUE="${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
AION_MORPHOLOGY_DIR="${AION_MORPHOLOGY_DIR:-$REPO_ROOT/data/clauds/images/tilesv5/}"
AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-mlp_full_image_comparison}"
AION_CACHE_ROOT="${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"

FLAGS=()
[[ "${QWEN_LOAD_IN_4BIT:-1}" == 0 ]] && FLAGS+=(--no-qwen-4bit)
[[ "${QWEN_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-qwen-download)
[[ "${QWEN_NORMALIZE:-0}" == 1 ]] && FLAGS+=(--qwen-normalize)
[[ "${AION_FORCE_REBUILD_TOKENS:-0}" == 1 ]] && FLAGS+=(--force-rebuild-tokens)
[[ "${AION_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == 1 ]] && FLAGS+=(--force-rebuild-photometry --force-recompute-qwen)

cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_mlp_full_image_comparison.py" \
    --catalogue "$AION_CATALOGUE" --morphology-dir "$AION_MORPHOLOGY_DIR" \
    --output-dir "$AION_OUTPUT_DIR" --cache-root "$AION_CACHE_ROOT" \
    --max-rows "${AION_MAX_ROWS:-200000}" --epochs "${AION_EPOCHS:-10}" \
    --token-batch-size "${AION_TOKEN_BATCH_SIZE:-64}" \
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-1}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --device "${AION_DEVICE:-auto}" --n-z-bins "${AION_N_Z_BINS:-300}" \
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
    --image-flux-scale "${AION_IMAGE_FLUX_SCALE:-1.0}" \
    --min-cutout-weight-coverage "${AION_MIN_CUTOUT_WEIGHT_COVERAGE:-0.90}" \
    --feature-scaling minmax --qwen-model "${QWEN_MODEL:-qwen3_8b_base}" \
    --qwen-max-length "${QWEN_MAX_LENGTH:-2048}" --qwen-pooling "${QWEN_POOLING:-mean}" \
    "${FLAGS[@]}" "$@"
