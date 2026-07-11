# RedFM_MLP staging Codex work log

Date: 2026-07-10

Workspace: `/scratch/RedFM`

Nested package repo: `/scratch/RedFM/RedFM_MLP`

## Scope

This session staged `RedFM_MLP` inside the larger `/scratch/RedFM` workspace, added a Qwen embedding path, converted the AION notebook test to an executable script, and changed the pixi environment from CPU-only PyTorch to GPU-enabled PyTorch for the visible H100 MIG slice.

## Repository setup

- Cloned `https://github.com/ShimingGu/RedFM_MLP` into `./RedFM_MLP`.
- Kept `RedFM_MLP/` isolated from the parent `/scratch/RedFM` git repository:
  - parent `.gitignore` includes `RedFM_MLP/`
  - `github_worker.py` excludes `RedFM_MLP`
- Installed `RedFM_MLP` as an editable package in the pixi environment:

```bash
pixi run python -m pip install -e ./RedFM_MLP --no-deps
```

After the later Python/PyTorch environment rebuild, this editable install had to be repeated.

## Local model checkpoints

Downloaded and verified local Hugging Face checkpoints under:

- `/scratch/.tmp-gsm/hf_models/Qwen3-8B-Base`
- `/scratch/.tmp-gsm/hf_models/Qwen2.5-Math-7B`

The HF CLI needed telemetry/update checks disabled to avoid startup hangs:

```bash
HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_UPDATE_CHECK=1 hf ...
```

## Qwen embedding module

Added:

- `aion_magnitude/FM_Qwen.py`

Main functionality:

- Local default model paths for Qwen3-8B-Base and Qwen2.5-Math-7B.
- `load_frozen_qwen(...)` for frozen Qwen causal LM loading.
- 4-bit loading through `BitsAndBytesConfig`.
- Catalogue-row text serialization for grizy magnitudes plus optional extra features.
- Hidden-state pooling modes:
  - `mean`
  - `last`
  - `mean_last`
- Dataset helper mirroring the AION embedding extraction flow.

Important design note:

Qwen does not have AION's redshift-aware embedding. The Qwen path serializes the tabular catalogue row into text and pools general LM hidden states. It should be treated as a general frozen representation to be combined with an external MLP, not as a drop-in semantic equivalent of AION redshift tokens.

Final Qwen verification:

```text
Qwen2.5-Math-7B 4-bit load: OK
device: cuda:0
model dtype: torch.bfloat16
GPU allocation: about 5.56 GB
embedding shape: (1, 3584)
embedding dtype: torch.float32
normalized embedding norm: about 1.0
```

Also updated `FM_Qwen.py` to use the current Transformers `dtype` kwarg instead of deprecated `torch_dtype`.

## AION executable test script

Added:

- `notebooks/aion_mlp_test.py`

Purpose:

- Executable equivalent of the first AION notebook test.
- Defaults to AION-only mode:
  - `use_aion_embedding=True`
  - `use_mlp_features=False`
  - `model_kinds=("aion",)`
- Saves figures to:
  - `/arc/home/gsm/aion_output/figures`
- Uses scratch cache/checkpoint output by default:
  - `/scratch/.tmp-gsm/aion_output/cache`

Default command:

```bash
pixi run python RedFM_MLP/notebooks/aion_mlp_test.py
```

Small smoke command used during verification:

```bash
pixi run python RedFM_MLP/notebooks/aion_mlp_test.py \
  --max-rows 50 \
  --epochs 1 \
  --embedding-batch-size 8 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --tomographic-samples 5 \
  --force-recompute-embeddings
```

Output figures written:

- `/arc/home/gsm/aion_output/figures/aion_test_zp50_vs_zphot.jpeg`
- `/arc/home/gsm/aion_output/figures/aion_test_pit_histogram.jpeg`
- `/arc/home/gsm/aion_output/figures/aion_test_z_mean_vs_zphot.jpeg`
- `/arc/home/gsm/aion_output/figures/aion_test_z_mode_vs_zphot.jpeg`
- `/arc/home/gsm/aion_output/figures/aion_test_redshift_probability_distribution.jpeg`
- `/arc/home/gsm/aion_output/figures/aion_test_tomographic_nz.jpeg`
- `/arc/home/gsm/aion_output/figures/aion_test_metrics.json`

## Package compatibility fixes

The cloned package did not run the notebook path as-is. Fixes made in the nested `RedFM_MLP` repo:

- `config.py`
  - Added missing `@dataclass` for `AIONMagnitudeConfig`.
  - Added missing imports used by config normalization/path resolution.
  - Added local `_make_cache_run_tag(...)` to avoid an unresolved reference.

- `dataset.py`
  - Added dataclasses for `CLAUDSPhotoZBatch` and `CachedFusionBatch`.
  - Added missing imports.
  - Added `load_clauds_catalogue_from_fits(...)` that reuses or rebuilds the split `.npy` cache.
  - Fixed `dataset_for_split(...)` to avoid an unresolved `split_cached_product` reference.

- `caching.py`
  - Added missing imports for DataLoader, config helpers, split helpers, redshift grid helpers, and AION adjustment metadata validation.

- `models.py`
  - Added missing `nullcontext`, device resolver, and table helpers.

- `metrics.py`
  - Added missing `torch.nn.functional as F`.
  - Added missing table/tensor helpers.

- `training.py`
  - Added missing `torch.nn as nn`, `matplotlib.pyplot as plt`, config helpers, device resolver, cached batch type, plotting helpers, caching entry point, and metrics helpers.

- `plotting.py`
  - Added missing device/tensor helpers and metric sampling/binning helpers.
  - Imported `make_magnitude_config` and `run_training_and_evaluation` locally inside `run_config_pair(...)` to avoid top-level circular imports.

## GPU-enabled PyTorch environment

Initial state:

```text
torch 2.12.0
torch.version.cuda None
torch.cuda.is_available() False
```

Hardware visible through `nvidia-smi`:

```text
NVIDIA H100 NVL
MIG 1g.12gb
driver CUDA capability 13.2
```

Final pixi stack:

```text
python 3.13.14
torch 2.9.1
torch.version.cuda 12.9
torch.cuda.is_available() True
device NVIDIA H100 NVL MIG 1g.12gb
bitsandbytes 0.49.2
transformers 5.13.0
tokenizers 0.22.2
```

Reason for using PyTorch 2.9.1:

- Current conda-forge `bitsandbytes 0.49.2` available in this environment is CUDA 12.9-only.
- Newer conda-forge PyTorch 2.12 GPU builds resolved to CUDA 13.0.
- To keep Qwen 4-bit inference working, the environment was pinned to a CUDA 12.9-compatible PyTorch GPU stack.

Changes made in parent `pixi.toml`:

```toml
[activation.env]
PYTHONNOUSERSITE = "1"

[dependencies]
python = ">=3.13,<3.14"
pytorch = "2.9.1.*"
pytorch-gpu = "2.9.1.*"
cuda-version = "12.9.*"
bitsandbytes = ">=0.49.2,<0.50"
transformers = ">=5.13.0,<6"
accelerate = ">=1.14.0,<2"
safetensors = ">=0.8.0,<0.9"
pip = ">=26.1.2,<27"

[system-requirements]
cuda = "13.2"
```

The `PYTHONNOUSERSITE=1` activation setting is important. Without it, user-site packages under `/arc/home/gsm/.local/lib/python3.13/site-packages` overrode the pixi environment and caused Transformers to see `tokenizers==0.23.1`, which is outside its required `<=0.23.0` range.

## Verification commands

Core CUDA check:

```bash
pixi run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn((1024,1024), device='cuda'); print((x@x).shape)"
```

Observed:

```text
2.9.1 12.9 True NVIDIA H100 NVL MIG 1g.12gb
torch.Size([1024, 1024])
```

Qwen tokenizer/config check:

```text
/scratch/.tmp-gsm/hf_models/Qwen3-8B-Base qwen3 hidden_size=4096 Qwen2Tokenizer
/scratch/.tmp-gsm/hf_models/Qwen2.5-Math-7B qwen2 hidden_size=3584 Qwen2Tokenizer
```

AION executable smoke test:

```text
available devices: ['cuda', 'cpu']
Baseline training completed successfully.
Figures saved under /arc/home/gsm/aion_output/figures.
```

## Current git status notes

Parent `/scratch/RedFM` has modifications from this work:

- `.gitignore`
- `github_worker.py`
- `pixi.toml`
- `pixi.lock`

The parent repo still has pre-existing unrelated untracked/modified items:

- `.vscode/`
- `AiImTok/`

Nested `RedFM_MLP` status includes:

- modified package modules listed in the compatibility fixes section
- untracked `aion_magnitude/FM_Qwen.py`
- untracked `notebooks/aion_mlp_test.py`
- this log file

## Continuation notes

- Qwen3-8B-Base was installed locally but only Qwen2.5-Math-7B was fully 4-bit loaded in the final smoke test.
- The visible MIG slice has about 11 GiB memory. Qwen2.5-Math-7B 4-bit fits with room for short batches; larger batch sizes or sequence lengths may OOM.
- Full-catalogue AION or Qwen embedding runs should now use GPU automatically through `device_choice="auto"` or explicit `device="cuda"`.
- If pixi recreates the environment again, reinstall the nested package editable:

```bash
pixi run python -m pip install -e ./RedFM_MLP --no-deps
```
