#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
else
    DEFAULT_PIXI_MANIFEST="$REPO_ROOT/../RedFM_original/pixi.toml"
    PIXI_MANIFEST="${PIXI_MANIFEST:-$DEFAULT_PIXI_MANIFEST}"
    if [[ -f "$PIXI_MANIFEST" ]]; then
        PYTHON_CMD=(pixi run --manifest-path "$PIXI_MANIFEST" python)
    else
        PYTHON_CMD=(pixi run python)
    fi
fi

AION_CATALOGUE="${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures}"
AION_CACHE_ROOT="${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"
QWEN_MODEL="${QWEN_MODEL:-qwen3_8b_base}"

QWEN_FLAGS=()
if [[ "${QWEN_LOAD_IN_4BIT:-1}" == "0" ]]; then
    QWEN_FLAGS+=(--no-qwen-4bit)
fi
if [[ "${QWEN_ALLOW_DOWNLOAD:-0}" == "1" ]]; then
    QWEN_FLAGS+=(--allow-qwen-download)
fi
if [[ "${QWEN_NORMALIZE:-0}" == "1" ]]; then
    QWEN_FLAGS+=(--qwen-normalize)
fi

# Both frozen encoders receive only HSC grizy. Their embeddings feed the same
# photo-z head and use identical catalogue rows, splits, loss, and optimizer.
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_aion_comparison.py" \
    --catalogue "$AION_CATALOGUE" \
    --output-dir "$AION_OUTPUT_DIR" \
    --cache-root "$AION_CACHE_ROOT" \
    --max-rows "${AION_MAX_ROWS:-none}" \
    --epochs "${AION_EPOCHS:-10}" \
    --aion-embedding-batch-size "${AION_EMBEDDING_BATCH_SIZE:-512}" \
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-8}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
    --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --device "${AION_DEVICE:-auto}" \
    --n-z-bins "${AION_N_Z_BINS:-300}" \
    --z-max "${AION_Z_MAX:-6.0}" \
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
    --qwen-model "$QWEN_MODEL" \
    --qwen-max-length "${QWEN_MAX_LENGTH:-256}" \
    --qwen-pooling "${QWEN_POOLING:-mean}" \
    "${QWEN_FLAGS[@]}" \
    "$@"
