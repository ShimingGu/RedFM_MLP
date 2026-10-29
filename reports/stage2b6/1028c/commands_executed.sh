#!/usr/bin/env bash
# Run after the completed Stage2A.6 handoff; prepare refuses overwriting.
set -euo pipefail
cd /scratch/RedFM_MLP
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
"$PYTHON" scripts/stage2b6_trust.py prepare
"$PYTHON" scripts/stage2b6_trust.py run
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/report_stage2_1028c.py calibration_controls
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/report_stage2_1028c.py finalize
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/report_stage2_1028c.py benchmark
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/report_stage2_1028c.py refresh
