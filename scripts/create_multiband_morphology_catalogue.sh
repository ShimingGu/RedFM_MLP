#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The default command is `all`. Pass `manifest`, `train-head`, `features`, or
# `catalogue` as the first argument to run one resumable stage only.
exec pixi run python -u -m aion_magnitude.multiband_morphology_catalogue \
    --catalogue-path "${AION_CATALOGUE:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits}" \
    --output-catalogue-path "${AION_OUTPUT_CATALOGUE:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits}" \
    --u-image-dir "${AION_MORPHOLOGY_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/images/tilesv5}" \
    --cache-dir "${AION_MULTIBAND_CACHE_DIR:-/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/cache/aion_multiband_morphology_catalogue_updated}" \
    "$@"
