#!/usr/bin/env bash
set -euo pipefail
cd /scratch/RedFM_MLP
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
"$PYTHON" scripts/stage2_numerical.py tests
"$PYTHON" scripts/stage2_numerical.py prepare
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e "$PYTHON" scripts/stage2_numerical.py run
"$PYTHON" scripts/report_stage2_physics.py
