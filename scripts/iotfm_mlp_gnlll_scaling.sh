#!/usr/bin/env bash
# Identical to iotfm_mlp_gnlll.sh except that physical robust scaling is enabled.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GNLLL_FEATURE_SCALING=physical_robust
exec "$SCRIPT_DIR/iotfm_mlp_gnlll.sh" "$@"
