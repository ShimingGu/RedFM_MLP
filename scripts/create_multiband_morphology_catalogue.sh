#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The default command is `all`. Pass `manifest`, `train-head`, `features`, or
# `catalogue` as the first argument to run one resumable stage only.
exec pixi run python -m aion_magnitude.multiband_morphology_catalogue "$@"
