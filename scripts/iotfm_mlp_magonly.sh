#!/usr/bin/env bash
# IoTFM = inference-optimized transformer feature mapping.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Validation control: use only u, u*, g, r, i, z, y, Y, J, H, and Ks AB
# magnitudes. Keep the parent pipeline's rows, splits, training, and figures.
export AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/iotfm_mlp_magonly}"
export IOTFM_INCLUDE_ID=0
export IOTFM_INCLUDE_LOCATION=0

exec "$SCRIPT_DIR/iotfm_mlp.sh" --magnitudes-only "$@"
