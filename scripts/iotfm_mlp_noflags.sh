#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Validation ablation: exclude every detection/quality and mask/field flag
# classified as a flag in column_types.md; retain every other standard input.
DEFAULT_OUTPUT_DIR="/arc/home/gsm/aion_output/figures/iotfm_mlp_noflags"
for arg in "$@"; do
    if [[ "$arg" == "--no-classification" ]]; then
        DEFAULT_OUTPUT_DIR="/arc/home/gsm/aion_output/figures/iotfm_mlp_noflags_noclassification"
        break
    fi
done
export AION_OUTPUT_DIR="${AION_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
exec "$SCRIPT_DIR/iotfm_mlp.sh" --exclude-flags --ignore-missingness "$@"
