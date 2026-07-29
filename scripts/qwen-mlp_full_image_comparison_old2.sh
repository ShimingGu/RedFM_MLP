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

AION_CATALOGUE="${AION_CATALOGUE:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits}"
AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/qwen-mlp_full_image_comparison}"
AION_CACHE_ROOT="${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}"

FLAGS=()
[[ "${QWEN_LOAD_IN_4BIT:-1}" == 0 ]] && FLAGS+=(--no-qwen-4bit)
[[ "${QWEN_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-qwen-download)
[[ "${QWEN_NORMALIZE:-0}" == 1 ]] && FLAGS+=(--qwen-normalize)
[[ "${QWEN_ALLOW_TRUNCATION:-0}" == 1 ]] && FLAGS+=(--allow-qwen-truncation)
[[ "${QWEN_PHYSICAL_CONTEXT:-1}" == 0 ]] && FLAGS+=(--no-qwen-physical-context)
[[ "${AION_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == 1 ]] && FLAGS+=(--force-recompute-qwen)

# Frozen Qwen and the numerical MLP receive the same available magnitudes and
# the same 42 u,g,r,i,z,y morphology-catalogue features. The cohort requires
# morphology_available_* in all six bands. No image cutouts or tokens are read.
cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/qwen_mlp_full_image_comparison.py" \
    --catalogue "$AION_CATALOGUE" \
    --output-dir "$AION_OUTPUT_DIR" \
    --cache-root "$AION_CACHE_ROOT" \
    --max-rows "${AION_MAX_ROWS:-200000}" \
    --epochs "${AION_EPOCHS:-10}" \
    --qwen-embedding-batch-size "${QWEN_EMBEDDING_BATCH_SIZE:-1}" \
    --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
    --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
    --train-fraction "${AION_TRAIN_FRACTION:-0.63}" \
    --test-fraction "${AION_TEST_FRACTION:-0.32}" \
    --val-fraction "${AION_VAL_FRACTION:-0.05}" \
    --seed "${AION_SEED:-42}" \
    --device "${AION_DEVICE:-auto}" \
    --n-z-bins "${AION_N_Z_BINS:-300}" \
    --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
    --feature-scaling "${AION_FEATURE_SCALING:-minmax}" \
    --qwen-model "${QWEN_MODEL:-Qwen3-8B-Base}" \
    --qwen-max-length "${QWEN_MAX_LENGTH:-2048}" \
    --qwen-pooling "${QWEN_POOLING:-last}" \
    "${FLAGS[@]}" "$@"
