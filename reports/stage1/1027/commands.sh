# Successful commands, from /scratch/RedFM_MLP. Independent inference and
# physical-baseline calls ran concurrently after the Stage-0.5 completion gate.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
stage1_python=/scratch/.tmp-gsm/stage0-venv/bin/python
"$stage1_python" -m pytest -q tests/test_stage1_sed_forward.py
"$stage1_python" scripts/stage1_physics_benchmark.py --stage prepare
"$stage1_python" -m pytest -q tests/test_stage1_sed_forward.py tests/test_stage1_benchmark_integrity.py
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage reason --model base
CUDA_VISIBLE_DEVICES=MIG-dd002e82-c996-5c41-a8fd-0220ae1e9c32 HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage reason --model instruct
"$stage1_python" scripts/stage1_physics_benchmark.py --stage init-random
CUDA_VISIBLE_DEVICES=MIG-a3be3c07-49d1-5c06-88e8-27eb177c5e31 HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage reason --model random
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage extract --model base
CUDA_VISIBLE_DEVICES=MIG-96b155fa-58a8-55f6-b1e3-0486e7c783c0 HF_HUB_OFFLINE=1 "$stage1_python" scripts/stage1_physics_benchmark.py --stage extract --model random
"$stage1_python" scripts/stage1_physics_benchmark.py --stage physical
CUDA_VISIBLE_DEVICES=MIG-2996269f-ca01-5507-9a8c-0519fb48fdff HF_HUB_OFFLINE=0 "$stage1_python" scripts/stage1_tabicl_control.py
"$stage1_python" scripts/stage1_physics_benchmark.py --stage fit
"$stage1_python" scripts/report_stage1_physics.py
