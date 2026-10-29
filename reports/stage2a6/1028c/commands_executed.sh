#!/usr/bin/env bash
# Original launch sequence in a fresh output directory; prepare intentionally
# refuses to overwrite a prior protocol. Existing arrays/checkpoints are cached.
set -euo pipefail
cd /scratch/RedFM_MLP
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
"$PYTHON" scripts/stage2a6_identifiability.py prepare
"$PYTHON" scripts/stage2a6_identifiability.py tests
"$PYTHON" scripts/stage2a6_identifiability.py run
"$PYTHON" scripts/stage2a6_identifiability.py assessment
"$PYTHON" scripts/stage2a6_identifiability.py posterior_moments
