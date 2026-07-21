#!/usr/bin/env bash
# GNLLL = Gaussian negative log-likelihood loss.
# IoTFM = inference-optimized transformer feature mapping.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
FEATURE_SCALING="${GNLLL_FEATURE_SCALING:-none}"
case "$FEATURE_SCALING" in
    none) EXPERIMENT_SUFFIX="" ;;
    physical_robust) EXPERIMENT_SUFFIX="_scaling" ;;
    *) echo "Unknown GNLLL_FEATURE_SCALING: $FEATURE_SCALING" >&2; exit 2 ;;
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

if ! "${PYTHON_CMD[@]}" -c 'import numpy, torch, astropy, transformers, aion_magnitude' >/dev/null; then
    echo "The selected Python environment is missing IoTFM/GNLLL dependencies." >&2
    exit 1
fi

FLAGS=()
[[ "${GNLLL_FORCE_RECOMPUTE_INPUT_CACHE:-0}" == 1 ]] && FLAGS+=(--force-recompute-input-cache)
[[ "${IOTFM_LOAD_IN_4BIT:-0}" == 1 ]] && FLAGS+=(--load-in-4bit)
[[ "${IOTFM_ALLOW_DOWNLOAD:-0}" == 1 ]] && FLAGS+=(--allow-download)
[[ "${IOTFM_NORMALIZE:-0}" == 1 ]] && FLAGS+=(--normalize)
[[ "${IOTFM_FORCE_RECOMPUTE_EMBEDDINGS:-0}" == 1 ]] && FLAGS+=(--force-recompute-embeddings)

cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" "$REPO_ROOT/notebooks/iotfm_mlp_gnlll.py" \
  --catalogue "${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}" \
  --output-dir "${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/iotfm_mlp_gnlll${EXPERIMENT_SUFFIX}}" \
  --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp_gnlll${EXPERIMENT_SUFFIX}}" \
  --input-cache-root "${AION_INPUT_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache/iotfm_mlp_gnlll_input}" \
  --max-rows "${AION_MAX_ROWS:-200000}" --model "${IOTFM_MODEL:-GLM-5.2-0.8B-A0.8B}" \
  --embedding-batch-size "${IOTFM_EMBEDDING_BATCH_SIZE:-8}" \
  --max-length "${IOTFM_MAX_LENGTH:-2048}" --pooling "${IOTFM_POOLING:-mean}" \
  --feature-scaling "$FEATURE_SCALING" \
  --mean-warmup-epochs "${GNLLL_MEAN_WARMUP_EPOCHS:-10}" \
  --gnlll-epochs "${GNLLL_EPOCHS:-20}" \
  --learning-rate "${GNLLL_LEARNING_RATE:-0.001}" \
  --weight-decay "${GNLLL_WEIGHT_DECAY:-0.0001}" \
  --mean-hidden-dim "${GNLLL_MEAN_HIDDEN_DIM:-256}" \
  --variance-hidden-dim "${GNLLL_VARIANCE_HIDDEN_DIM:-128}" \
  --dropout "${GNLLL_DROPOUT:-0.1}" --variance-floor "${GNLLL_VARIANCE_FLOOR:-1e-6}" \
  --train-batch-size "${AION_TRAIN_BATCH_SIZE:-256}" \
  --eval-batch-size "${AION_EVAL_BATCH_SIZE:-512}" \
  --n-z-bins "${AION_N_Z_BINS:-300}" \
  --tomographic-samples "${AION_TOMOGRAPHIC_SAMPLES:-100}" \
  --train-fraction "${AION_TRAIN_FRACTION:-0.63}" \
  --test-fraction "${AION_TEST_FRACTION:-0.32}" \
  --val-fraction "${AION_VAL_FRACTION:-0.05}" \
  --seed "${AION_SEED:-42}" --device "${AION_DEVICE:-auto}" "${FLAGS[@]}" "$@"
