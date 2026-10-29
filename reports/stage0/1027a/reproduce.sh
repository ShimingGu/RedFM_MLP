#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
stage0_output=${1:-reports/stage0/1027a-reproduced}
stage0_python=${STAGE0_PYTHON:-/scratch/.tmp-gsm/stage0-venv/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/scratch/.tmp-gsm/stage0-mpl
# If the temporary runtime no longer exists:
# uv venv /scratch/.tmp-gsm/stage0-venv --python /usr/local/bin/python
# uv pip install --python "$stage0_python" --cache-dir /scratch/.tmp-gsm/stage0-uv -r reports/stage0/1027a/requirements-executed.txt
# Choose any currently idle accessible MIG slice using CUDA_VISIBLE_DEVICES.
"$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage prepare --output-dir "$stage0_output"
for stage0_arm in original A B C; do
  "$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage run --arm "$stage0_arm" --output-dir "$stage0_output"
done
"$stage0_python" scripts/stage0_source_ebv_control.py --output-dir "$stage0_output"
"$stage0_python" scripts/diagnose_hsc_clauds_stage0.py --stage collect --output-dir "$stage0_output"
"$stage0_python" scripts/report_hsc_clauds_stage0.py --output-dir "$stage0_output"
