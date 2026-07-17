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
    elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
    else
        echo "No project Python environment found." >&2
        echo "Expected $PIXI_MANIFEST or $REPO_ROOT/.venv/bin/python." >&2
        exit 1
    fi
fi

if ! "${PYTHON_CMD[@]}" -c 'import numpy, torch, astropy, matplotlib, safetensors, aion, transformers, accelerate, bitsandbytes' >/dev/null; then
    echo "The selected Python environment is missing Qwen/image-token dependencies." >&2
    echo "Install the qwen-cuda extra with: .venv/bin/pip install -e \".[qwen-cuda]\"" >&2
    exit 1
fi

if ! "${PYTHON_CMD[@]}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null; then
    echo "CUDA is not available in the selected Python environment." >&2
    exit 1
fi

AION_CATALOGUE="${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
AION_MORPHOLOGY_DIR="${AION_MORPHOLOGY_DIR:-$REPO_ROOT/data/clauds/images/tilesv5/}"
AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-mlp_full_comparison}"
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
if [[ "${AION_FORCE_REBUILD_TOKENS:-0}" == "1" ]]; then
    QWEN_FLAGS+=(--force-rebuild-tokens)
fi
if [[ "${AION_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == "1" ]]; then
    QWEN_FLAGS+=(--force-rebuild-photometry --force-recompute-qwen)
fi

# Compare all-magnitude Qwen + AION-tokenized u images against the all-magnitude
# MLP + the same tokens. Faint-end magnitude and redshift-range cuts are disabled.
# Only the AION image tokenizer is used; its image-to-redshift embedding is not.
cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_mlp_full_comparison.py" \
    --catalogue "$AION_CATALOGUE" \
    --morphology-dir "$AION_MORPHOLOGY_DIR" \
    --output-dir "$AION_OUTPUT_DIR" \
    --cache-root "$AION_CACHE_ROOT" \
    --max-rows "${AION_MAX_ROWS:-none}" \
    --epochs "${AION_EPOCHS:-10}" \
    --token-batch-size "${AION_TOKEN_BATCH_SIZE:-64}" \
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-8}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
    --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --device "${AION_DEVICE:-auto}" \
    --n-z-bins "${AION_N_Z_BINS:-300}" \
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
    --image-flux-scale "${AION_IMAGE_FLUX_SCALE:-1.0}" \
    --min-cutout-weight-coverage "${AION_MIN_CUTOUT_WEIGHT_COVERAGE:-0.90}" \
    --feature-scaling minmax \
    --qwen-model "$QWEN_MODEL" \
    --qwen-max-length "${QWEN_MAX_LENGTH:-256}" \
    --qwen-pooling "${QWEN_POOLING:-mean}" \
    "${QWEN_FLAGS[@]}" \
    "$@"

