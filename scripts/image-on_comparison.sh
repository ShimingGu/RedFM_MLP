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
        echo "Create one with: python3 -m venv .venv && .venv/bin/pip install -e . polymathic-aion" >&2
        exit 1
    fi
fi

if ! "${PYTHON_CMD[@]}" -c 'import numpy, torch, astropy, matplotlib, safetensors, aion' >/dev/null; then
    echo "The selected Python environment is missing image-on comparison dependencies." >&2
    echo "Install them with: .venv/bin/pip install -e . polymathic-aion" >&2
    exit 1
fi

AION_CATALOGUE="${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
AION_MORPHOLOGY_DIR="${AION_MORPHOLOGY_DIR:-$REPO_ROOT/data/clauds/morphology}"
AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/image-on_comparison}"
AION_CACHE_ROOT="${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"

REBUILD_FLAGS=()
if [[ "${AION_FORCE_REBUILD_TOKENS:-0}" == "1" ]]; then
    REBUILD_FLAGS+=(--force-rebuild-tokens)
fi
if [[ "${AION_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == "1" ]]; then
    REBUILD_FLAGS+=(--force-rebuild-photometry)
fi

# Compare the standard frozen grizy-AION embedding against the same embedding
# plus CLAUDS u-image tokens. Images use only the AION codec; their decoded FSQ
# token factors enter the trainable image MLP, never AION's image/redshift model.
cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" -m aion_magnitude.morphology train \
    --catalogue-path "$AION_CATALOGUE" \
    --morphology-dir "$AION_MORPHOLOGY_DIR" \
    --output-dir "$AION_OUTPUT_DIR" \
    --cache-root "$AION_CACHE_ROOT" \
    --max-rows "${AION_MAX_ROWS:-20000}" \
    --epochs "${AION_EPOCHS:-20}" \
    --aion-embedding-batch-size "${AION_EMBEDDING_BATCH_SIZE:-512}" \
    --token-batch-size "${AION_TOKEN_BATCH_SIZE:-64}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
    --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --device "${AION_DEVICE:-auto}" \
    --n-z-bins "${AION_N_Z_BINS:-100}" \
    --z-max "${AION_Z_MAX:-2.5}" \
    --image-flux-scale "${AION_IMAGE_FLUX_SCALE:-1.0}" \
    --min-cutout-weight-coverage "${AION_MIN_CUTOUT_WEIGHT_COVERAGE:-0.90}" \
    --use-aion-magnitude-embedding \
    --model-kinds aion,aion_morphology \
    "${REBUILD_FLAGS[@]}" \
    "$@"
