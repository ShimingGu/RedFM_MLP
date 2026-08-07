# Qwen photo-z post-training breakthrough — 2026-08-07

This note records the completed magnitudes-only Qwen post-training result, the
meaning of each comparison arm, the controlled follow-up methods added on
2026-08-07, and the proposed DoRA/RLVR extensions.

## Completed controlled run

Output:

`/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10/`

The run completed on 2026-08-05. It used:

- Qwen3.5-4B-Base with final non-padding-token pooling
- catalogue magnitudes only; no morphology, image cutouts, or image tokens
- the fallback catalogue recorded in `run.json`:
  `/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits`
- 300,000 objects selected with seed 42
- train/validation/test split: 60,000 / 15,000 / 225,000
- 300 redshift bins
- 10 epochs
- head learning rate `2e-4`
- adapter learning rate `1e-5`
- three head-only warm-up epochs
- microbatch size 1 with 16-step gradient accumulation
- effective batch size 16
- 3,750 optimizer updates per epoch and 37,500 total updates

## What the two original arms mean

### Frozen-Qwen control

The control is a frozen-feature supervised probe:

1. Qwen is completely frozen.
2. Its final-token representation is extracted and cached for every object.
3. A nonlinear `PhotoZHead` MLP is trained with cross-entropy over the 300
   redshift bins.

This is supervised head-only training, or a frozen-Qwen nonlinear probe. It is
not Qwen supervised fine-tuning because no Qwen parameter is updated.

### QLoRA arm

The QLoRA arm uses the same photo-z head and supervised redshift-bin
cross-entropy, with staged training:

- epochs 1–3: head-only supervised warm-up with the Qwen representation detached
- epochs 4–10: joint supervised training of the head and LoRA adapters on
  `q_proj,k_proj,v_proj,o_proj`

This is task-specific supervised fine-tuning of Qwen. It is not language-model
instruction SFT, but it is SFT in the general sense: labeled examples directly
define the optimization target. QLoRA describes the parameter-efficient
adapter mechanism; SFT describes the objective.

## Final metrics

| Arm | Test NMAD | Test outlier fraction | Test CE | Test CRPS | Test bias | 16–84% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Qwen + MLP probe | 0.15638 | 0.36552 | 4.55419 | 0.32966 | 0.00949 | 0.66461 |
| QLoRA SFT + matched head | **0.07294** | **0.18164** | **3.93014** | **0.22491** | 0.01087 | 0.69085 |

Validation and test results agree closely:

- QLoRA validation NMAD: 0.07376
- QLoRA test NMAD: 0.07294
- QLoRA validation CE: 3.93053
- QLoRA test CE: 3.93014

The large magnitudes-only experiment therefore gives a clear positive result:
task-specific QLoRA substantially outperforms the matched frozen-Qwen probe.
This result should replace the earlier six-band-complete morphology run when
judging whether staged QLoRA is useful. The earlier morphology cohort contained
only 332 training objects and was not a fair test.

## Runtime explanation

The completed run was long but was not stalled. With 60,000 training objects,
batch size 1, accumulation 16, and 10 epochs, it performed 600,000 individual
Qwen forward/backward passes. Gradient accumulation reduced optimizer updates
but did not batch those model passes.

The launcher called “multigpu” assigns the frozen and QLoRA arms to separate
GPU slices; it does not distribute the QLoRA model over multiple slices. The
QLoRA arm therefore ran on one 12 GB MIG slice.

## Controlled follow-up methods implemented

The following files were added:

- `aion_magnitude/qwen_alternative_posttraining.py`
- `notebooks/qwen_alternative_posttraining.py`
- `scripts/qwen-ia3_posttraining.sh`
- `scripts/qwen-residual_embedding_adapter_posttraining.sh`
- `tests/test_qwen_alternative_posttraining.py`

Exact launch commands are maintained at the very bottom of
`notebooks/recent_runs.txt`.

### IA3 SFT

IA3 applies learned multiplicative vectors to:

- attention `k_proj`
- attention `v_proj`
- feed-forward `down_proj`

It matches the QLoRA experiment on the cohort, split, epochs, effective batch,
head architecture, warm-up, head and adapter learning rates, clipping,
redshift-bin loss, validation selection, and final metrics. The intended
treatment difference is the parameter-efficient transformer adapter.

### Residual embedding-adapter SFT

This method reuses the cached frozen-Qwen final-token embeddings and trains:

`LayerNorm -> zero-initialized residual bottleneck -> matched PhotoZHead`

It is now matched to the QLoRA controls:

- 10 epochs
- effective batch size 16
- evaluation batch size 8
- three head-only warm-up epochs
- head learning rate `2e-4`
- adapter learning rate `1e-5`
- head/adapter clipping `1.0` / `0.1`
- matched warm-up/decay schedule and weight decay
- the same dataset, split, 300-bin objective, and metrics

It processes cached embeddings in batches of 16 rather than performing 16
single-object Qwen passes, so it is much faster while preserving the same
number of examples and optimizer updates per epoch. Its intended treatment
difference is that adaptation occurs after Qwen instead of inside Qwen.

## DoRA and RLVR implementations

DoRA and RLVR are not interchangeable categories:

- QLoRA, IA3, and DoRA specify how transformer parameters are adapted.
- SFT and RLVR specify the training objective.
- The residual embedding adapter specifies a post-Qwen adaptation location.

The following files now implement and launch the additional methods:

- `aion_magnitude/qwen_rlvr.py`
- `notebooks/qwen_dora_rlvr_posttraining.py`
- `scripts/qwen-dora_posttraining.sh`
- `scripts/qwen-rlvr_posttraining.sh`
- `tests/test_qwen_dora_rlvr.py`

### DoRA-SFT

The supervised QLoRA trainer now has a backward-compatible `use_dora` switch.
The DoRA arm keeps rank 8, alpha 16, dropout 0.05,
`q_proj,k_proj,v_proj,o_proj` targets, and every data/training control from
QLoRA. Only PEFT's weight-decomposed update is enabled.

The default launcher uses logical CUDA slice 2, a new output directory, and a
new resumable checkpoint directory.

### QLoRA-SFT to RLVR

RLVR restores the completed supervised QLoRA adapter and photo-z head as a
trainable policy. It first caches the fixed SFT reference log-probabilities for
the training cohort, then performs a one-epoch continuation by default.

For each object it samples eight redshift bins and computes

`delta = abs(z_sample - z_spec) / (1 + z_spec)`

with verifier reward

`exp(-delta / 0.05) - 0.5 * I(delta > 0.15)`.

The group-relative policy loss standardizes rewards within the eight samples.
An exact categorical KL penalty with coefficient 0.02 anchors the policy to
the completed SFT model, and an entropy coefficient of 0.001 discourages early
collapse. Default continuation learning rates are `1e-5` for the head and
`1e-6` for the QLoRA adapter. Checkpoints are written every 100 updates.

RLVR selects the saved epoch by validation expected verifier reward and still
reports the same CE, CRPS, NMAD, outlier, coverage, PIT, and bias metrics as the
SFT arms.

RLVR is a continuation of SFT rather than a from-scratch matched arm. To
separate the effect of RLVR from extra optimizer steps, a rigorous study should
also compare an equal-update supervised CE continuation.

All four launch commands are at the bottom of `notebooks/recent_runs.txt`:
IA3 on logical slice 0, the residual adapter on slice 1, DoRA on slice 2, and
RLVR on slice 3.

Verification now includes 18 passing focused tests, shell/Python syntax checks,
successful real 4-bit model restoration, and finite real forward/backward
adapter gradients for both DoRA and RLVR on 12 GB slices.

## GPU-slice audit

PyTorch currently exposes four logical CUDA devices:

| Logical CUDA index | Physical GPU UUID | Slice | Free memory at audit |
|---:|---|---|---:|
| 0 | GPU-0603f6a3-c514-fb3f-6e7f-3628af93c795 | H100 NVL MIG 1g.12gb | about 10.6 GiB |
| 1 | GPU-0603f6a3-c514-fb3f-6e7f-3628af93c795 | H100 NVL MIG 1g.12gb | about 10.6 GiB |
| 2 | GPU-7beb5988-45d3-2072-872e-be6e2047de40 | H100 NVL MIG 1g.12gb | about 10.6 GiB |
| 3 | GPU-7beb5988-45d3-2072-872e-be6e2047de40 | H100 NVL MIG 1g.12gb | about 10.6 GiB |

Thus logical CUDA devices 0, 1, and 2 are available, and logical device 3 is
also visible. Logical devices 0–1 map to physical GPU 0; logical devices 2–3
map to physical GPU 1. A slice from physical GPU 2 is not visible to PyTorch,
even though the system-wide `nvidia-smi -L` topology lists MIG instances on
that GPU.

At the time of the audit, no Qwen training process was active and all four
logical slices were essentially empty.
