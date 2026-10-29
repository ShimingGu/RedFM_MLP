#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
stage05_output=${1:-reports/stage05/1027-reproduced}
stage05_python=${STAGE05_PYTHON:-/scratch/.tmp-gsm/stage0-venv/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/scratch/.tmp-gsm/stage05-mpl
# Uses one currently idle slice sequentially. The completed run used six idle
# locked slices concurrently; neither mode interrupts other jobs.
"$stage05_python" scripts/stage05_interface_diagnostics.py --stage prepare --output-dir "$stage05_output"
for stage05_worker in 0 1 2 3 4 5; do
  "$stage05_python" scripts/stage05_interface_diagnostics.py --stage run --scope screen --worker "$stage05_worker" --output-dir "$stage05_output"
done
"$stage05_python" scripts/stage05_interface_diagnostics.py --stage validate-screen --output-dir "$stage05_output"
for stage05_worker in 0 1 2 3 4 5; do
  "$stage05_python" scripts/stage05_interface_diagnostics.py --stage run --scope full --worker "$stage05_worker" --output-dir "$stage05_output"
done
"$stage05_python" scripts/stage05_interface_diagnostics.py --stage collect --output-dir "$stage05_output"
