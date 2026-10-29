#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
stage1_output=${1:-reports/stage1/1027-reproduced}
stage1_python=${STAGE1_PYTHON:-/scratch/.tmp-gsm/stage0-venv/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/scratch/.tmp-gsm/stage1-mpl
# Requires the completed Stage-0.5 correctness gate. Refuse an existing result.
test ! -e "$stage1_output"
mkdir -p "$stage1_output/templates"
cp reports/stage1/1027/templates/* "$stage1_output/templates/"
"$stage1_python" -m pytest -q tests/test_stage1_sed_forward.py > "$stage1_output/forward-tests.log"
"$stage1_python" scripts/stage1_physics_benchmark.py --stage prepare --output-dir "$stage1_output"
for stage1_model in base instruct; do
  HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage reason --model "$stage1_model" --output-dir "$stage1_output"
done
"$stage1_python" scripts/stage1_physics_benchmark.py --stage init-random --output-dir "$stage1_output"
HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage reason --model random --output-dir "$stage1_output"
for stage1_model in base random; do
  HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage extract --model "$stage1_model" --output-dir "$stage1_output"
done
"$stage1_python" scripts/stage1_physics_benchmark.py --stage physical --output-dir "$stage1_output"
"$stage1_python" scripts/stage1_physics_benchmark.py --stage fit --output-dir "$stage1_output"
stage1_tabicl_revision=$("$stage1_python" -c 'import json;print(json.load(open("reports/stage1/1027/tabicl_protocol.json"))["revision"])')
HF_HUB_OFFLINE=0 "$stage1_python" scripts/stage1_tabicl_control.py --checkpoint-revision "$stage1_tabicl_revision" --output-dir "$stage1_output"
"$stage1_python" scripts/report_stage1_physics.py --output-dir "$stage1_output"
