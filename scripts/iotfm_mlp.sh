#!/usr/bin/env bash
# IoTFM = inference-optimized transformer feature mapping.
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
    echo "No project Python environment found." >&2; exit 1
fi
if ! "${PYTHON_CMD[@]}" -c 'import numpy, torch, astropy, transformers, aion_magnitude' >/dev/null; then
    echo "The selected Python environment is missing IoTFM/MLP dependencies." >&2; exit 1
fi
FLAGS=()
[[ "${IOTFM_INCLUDE_ID:-0}" == 1 ]] && FLAGS+=(--include-id)
[[ "${IOTFM_INCLUDE_LOCATION:-0}" == 1 ]] && FLAGS+=(--include-location)
[[ "${IOTFM_LOAD_IN_4BIT:-0}" == 1 ]] && FLAGS+=(--load-in-4bit)
[[ "${IOTFM_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-download)
[[ "${IOTFM_NORMALIZE:-0}" == 1 ]] && FLAGS+=(--normalize)
[[ "${IOTFM_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == 1 ]] && FLAGS+=(--force-recompute-embeddings)
cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/iotfm_mlp.py" \
  --catalogue "${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}" \
  --output-dir "${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/iotfm_mlp}" \
  --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp}" \
  --max-rows "${AION_MAX_ROWS:-200000}" --model "${IOTFM_MODEL:-GLM-5.2-0.8B-A0.8B}" \
  --embedding-batch-size "${IOTFM_EMBEDDING_BATCH_SIZE:-8}" \
  --torch-dtype "${IOTFM_TORCH_DTYPE:-float32}" --max-length "${IOTFM_MAX_LENGTH:-2048}" \
  --pooling "${IOTFM_POOLING:-mean}" --epochs "${AION_EPOCHS:-10}" \
  --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
  --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
  --n-z-bins "${AION_N_Z_BINS:-300}" --train-fraction "${AION_TRAIN_FRACTION:-0.63}" \
  --test-fraction "${AION_TEST_FRACTION:-0.32}" --val-fraction "${AION_VAL_FRACTION:-0.05}" \
  --seed "${AION_SEED:-42}" --device "${AION_DEVICE:-auto}" "${FLAGS[@]}" "$@"
