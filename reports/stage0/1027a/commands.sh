# Actual successful diagnostic execution, from /scratch/RedFM_MLP.
# The first preparation attempt required a NumPy row_stack -> vstack compatibility
# fix; unfinished preparation was retried. Source-control metadata slicing was
# corrected before its successful retry. The corresponding logs are retained.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/scratch/.tmp-gsm/stage0-mpl
stage0_python=/scratch/.tmp-gsm/stage0-venv/bin/python
"$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage prepare --output-dir reports/stage0/1027a
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e "$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage run --arm original --output-dir reports/stage0/1027a
CUDA_VISIBLE_DEVICES=MIG-dd002e82-c996-5c41-a8fd-0220ae1e9c32 "$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage run --arm B --output-dir reports/stage0/1027a
CUDA_VISIBLE_DEVICES=MIG-96b155fa-58a8-55f6-b1e3-0486e7c783c0 "$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage run --arm A --output-dir reports/stage0/1027a
CUDA_VISIBLE_DEVICES=MIG-a3be3c07-49d1-5c06-88e8-27eb177c5e31 "$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage run --arm C --output-dir reports/stage0/1027a
CUDA_VISIBLE_DEVICES=MIG-2996269f-ca01-5507-9a8c-0519fb48fdff "$stage0_python" scripts/stage0_source_ebv_control.py --output-dir reports/stage0/1027a
"$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage collect --output-dir reports/stage0/1027a
"$stage0_python" scripts/link_filter_aware_storage.py
"$stage0_python" -m pytest -q tests/test_filter_photometry.py tests/test_clauds_common_flux.py -k 'not trainer'
"$stage0_python" scripts/report_hsc_clauds_stage0.py --output-dir reports/stage0/1027a
