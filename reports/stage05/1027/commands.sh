# Successful execution, from /scratch/RedFM_MLP. Workers were invoked concurrently
# through independent tool calls; these loops describe the exact commands.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
stage05_python=/scratch/.tmp-gsm/stage0-venv/bin/python
stage05_devices=(MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e MIG-dd002e82-c996-5c41-a8fd-0220ae1e9c32 MIG-96b155fa-58a8-55f6-b1e3-0486e7c783c0 MIG-a3be3c07-49d1-5c06-88e8-27eb177c5e31 MIG-2996269f-ca01-5507-9a8c-0519fb48fdff MIG-f4c40dce-2b3c-567f-98b5-3d715ec2b5d7)
"$stage05_python" scripts/stage05_interface_diagnostics.py --stage prepare
"$stage05_python" -m pytest -q tests/test_stage05_physics_controls.py
for stage05_worker in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${stage05_devices[$stage05_worker]} "$stage05_python" scripts/stage05_interface_diagnostics.py --stage run --scope screen --worker "$stage05_worker"
done
# The first screening outputs were superseded when BF16 replay failed. After
# matching the historical autocast context, all screening was repeated.
"$stage05_python" scripts/stage05_interface_diagnostics.py --stage validate-screen
for stage05_worker in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${stage05_devices[$stage05_worker]} "$stage05_python" scripts/stage05_interface_diagnostics.py --stage run --scope full --worker "$stage05_worker"
done
"$stage05_python" scripts/stage05_interface_diagnostics.py --stage collect
