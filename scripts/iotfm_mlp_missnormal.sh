#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Missing values carry no predictive flag in this validation run: absent fields
# are omitted from IoTFM text and train-median-imputed for the MLP.
export AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-/arc/home/gsm/aion_output/figures/iotfm_mlp_missnormal}"
exec "$SCRIPT_DIR/iotfm_mlp.sh" --ignore-missingness "$@"
