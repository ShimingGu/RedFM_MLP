#!/usr/bin/env bash
set -euo pipefail
cd /scratch/RedFM_MLP
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
"$PYTHON" scripts/stage2b5_learnability.py prepare
"$PYTHON" scripts/stage2b5_learnability.py run
"$PYTHON" scripts/report_stage2_1028b.py controls
"$PYTHON" scripts/report_stage2_1028b.py source_convergence
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/report_stage2_1028b.py finalize
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/report_stage2_1028b.py verify
