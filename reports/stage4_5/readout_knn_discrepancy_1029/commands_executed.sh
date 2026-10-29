#!/usr/bin/env bash
set -euo pipefail
# Retrospective study; output directory and configs are versioned separately from Stage 4.
# Pick an available single MIG UUID; reserve_one_gpu enforces an idle locked slice.
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
# Native extraction and the CPU audit/diagnostics originally ran concurrently.
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e "$PYTHON" scripts/knn_check_native_1029.py
"$PYTHON" scripts/knn_check_cpu_1029.py audit
"$PYTHON" scripts/knn_check_cpu_1029.py compute
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e "$PYTHON" scripts/knn_check_native_heads_1029.py
"$PYTHON" scripts/knn_check_report_1029.py
"$PYTHON" scripts/knn_check_writeup_1029.py
"$PYTHON" scripts/knn_check_verify_1029.py
