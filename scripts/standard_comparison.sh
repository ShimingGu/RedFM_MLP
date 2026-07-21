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
        PYTHON_CMD=(pixi run python)
    fi
fi
AION_CATALOGUE="${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}"
AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures}"
AION_CACHE_ROOT="${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"

# These defaults match the active comparison cells in aion_mlp_test.ipynb:
# full catalogue, 10 epochs, grizy MLP-only versus grizy AION-only.
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/aion_mlp_test.py" \
    --mode standard-comparison \
    --catalogue "$AION_CATALOGUE" \
    --output-dir "$AION_OUTPUT_DIR" \
    --cache-root "$AION_CACHE_ROOT" \
    --max-rows "${AION_MAX_ROWS:-none}" \
    --epochs "${AION_EPOCHS:-10}" \
    --embedding-batch-size "${AION_EMBEDDING_BATCH_SIZE:-512}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
    --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --device "${AION_DEVICE:-auto}" \
    --n-z-bins "${AION_N_Z_BINS:-300}" \
    --z-max "${AION_Z_MAX:-6.0}" \
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
    "$@"
