#!/usr/bin/env bash
# Execution history; frozen outputs must not be overwritten.
# Python:/scratch/.tmp-gsm/stage0-venv/bin/python
# Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
# Use an independently reserved idle MIG UUID for each CUDA command.
python scripts/anecdote_audit_1029.py
python scripts/anecdote_data_1029.py
python scripts/anecdote_features_1029.py prepare
# The following three frozen extractions ran concurrently on separate MIGs:
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e python scripts/anecdote_features_1029.py B0
CUDA_VISIBLE_DEVICES=MIG-dd002e82-c996-5c41-a8fd-0220ae1e9c32 python scripts/anecdote_features_1029.py BS
CUDA_VISIBLE_DEVICES=MIG-96b155fa-58a8-55f6-b1e3-0486e7c783c0 python scripts/anecdote_features_1029.py BC
# Initial K64/256 pilot:subset101/head11; source-only failures retained.
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e python scripts/anecdote_heads_1029.py pilot
# The saved source_only_sparse_amendment.json enables the fixed C_ZPHOT sigma10
# control. The archived original code/failed pilot preserve the pre-amendment run.
# All primary conditions remain unchanged; no query results chose this control.
CUDA_VISIBLE_DEVICES=MIG-96b155fa-58a8-55f6-b1e3-0486e7c783c0 python scripts/anecdote_heads_1029.py pilot
# Full task curves ran concurrently; existing identical pilot fits are reused:
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e python scripts/anecdote_heads_1029.py GMS
CUDA_VISIBLE_DEVICES=MIG-dd002e82-c996-5c41-a8fd-0220ae1e9c32 python scripts/anecdote_heads_1029.py C_SPEC
CUDA_VISIBLE_DEVICES=MIG-96b155fa-58a8-55f6-b1e3-0486e7c783c0 python scripts/anecdote_heads_1029.py C_ZPHOT
python scripts/anecdote_verify_1029.py prequery
python scripts/anecdote_evaluate_1029.py
python scripts/anecdote_statistics_1029.py
python scripts/anecdote_report_1029.py finalize
python scripts/anecdote_interpret_1029.py
python scripts/anecdote_verify_1029.py final
