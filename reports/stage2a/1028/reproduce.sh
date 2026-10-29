#!/usr/bin/env bash
set -euo pipefail
cd /scratch/RedFM_MLP
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
"$PYTHON" scripts/stage2_data.py
"$PYTHON" scripts/stage2_physics.py build
"$PYTHON" scripts/stage2_physics.py evaluate
