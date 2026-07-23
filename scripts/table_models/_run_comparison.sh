#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 COMPARISON [table-model arguments...]" >&2
    exit 2
fi

COMPARISON="$1"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

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

cd -- "$REPO_ROOT"
exec "${PYTHON_CMD[@]}" -m aion_magnitude.table_models \
    --comparison "$COMPARISON" \
    --catalogue "${AION_CATALOGUE:-$REPO_ROOT/data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}" \
    --morphology-dir "${AION_MORPHOLOGY_DIR:-$REPO_ROOT/data/clauds/images/tilesv5}" \
    --output-root "${AION_OUTPUT_ROOT:-/arc/home/gsm/aion_output/figures/table_models}" \
    --cache-root "${AION_CACHE_ROOT:-/scratch/.tmp-gsm/aion_output/cache}" \
    "$@"
