#!/usr/bin/env bash
# Run from /scratch/RedFM_MLP in a fresh output copy. Do not overwrite frozen results.
# Environment: /scratch/.tmp-gsm/stage0-venv/bin/python; exact packages in Stage3A.
# The original sampler now also preserves test membership metadata for exclusions.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_recover_1029.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_data_1029.py target
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_data_1029.py background
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_data_1029.py legacy
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_data_1029.py finalize
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_support_1029.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_models_1029.py
# Stage3A readiness and benchmark locks are written before these fits.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_train_1029.py freeze
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_train_1029.py run
CUDA_VISIBLE_DEVICES=MIG-dd002e82-c996-5c41-a8fd-0220ae1e9c32 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_qwen_1029.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_evaluate_1029.py physical
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_diagnostics_1029.py physical_controls
# Only after all20 candidate checkpoints and the source gate are locked:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_evaluate_1029.py evaluate
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_diagnostics_1029.py inference_runtime
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_report_1029.py finalize
/scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_interpret_1029.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_diagnostics_1029.py verify

# Actual first collection hit the archived ID dtype guard; it was completed with:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/.tmp-gsm/stage0-venv/bin/python scripts/stage3_finish_1029.py
# Current evaluator has only the two ID guard corrections for fresh reproductions.
