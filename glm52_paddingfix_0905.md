# GLM-5.2-0.8B-A0.8B padding-fix experiment

Date: 2026-09-05

## Outcome

The earlier GLM failure is not reproduced after correcting last-token pooling.
GLM QLoRA now improves over its matched frozen-head baseline on the untouched
225,000-object test split. The failure investigation can stop after the
corrected Stage 0 and matched reruns; the later layer, prompt, target-module,
and capacity stages remain optional optimization experiments.

## Root cause and correction

The shared `last` pooling code used `attention_mask.sum(1) - 1` as an absolute
sequence index. That is valid for right padding but not for the GLM tokenizer's
left-padded batches. It therefore pooled a padding or earlier-token position,
and the representation changed with batch padding shape.

The fix selects the greatest absolute position whose attention mask is set. It
is applied in both shared pooling implementations. A regression test covers
mixed left- and right-padded rows.

The rebuilt frozen cache also uses `prepare_model_for_kbit_training` before
embedding extraction. This matches the model state used by the PEFT path;
preparation promotes the checkpoint's non-quantized tensors from BF16 to FP32
and measurably changes a small number of MoE routing decisions. The corrected
cache filename and metadata distinguish it from the old cache.

## Stage 0 invariants

Stage 0 used the first 2,048 validation objects from the fixed seed-42 split.
All automatic checks pass:

- cache row order matches the product;
- no row is truncated;
- live, prepared, and PEFT tokenization is identical;
- cached versus prepared mean embedding cosine is 0.999770;
- prepared versus zero-adapter mean cosine is 1.000000;
- saved LoRA-B tensors are exactly zero;
- cached and prepared frozen-head CE differs by only 0.000208.

The prepared live model and the zero-initialized PEFT wrapper are exactly
identical at the representation level. This clears the online QLoRA path.

## Corrected full-photometry results

Both runs use 60,000 train, 15,000 validation, and 225,000 test objects with
seed 42. QLoRA uses physical batch 8, accumulation 2, and effective batch 16.
The saved QLoRA model is selected by minimum validation cross-entropy at epoch
4, the second joint-adapter epoch.

| Method | Test CE | Test NMAD | Test catastrophic fraction | Test CRPS |
|---|---:|---:|---:|---:|
| Corrected frozen head | 4.876984 | 0.238536 | 0.527218 | 0.405133 |
| Corrected QLoRA | **4.853008** | **0.226336** | **0.507569** | **0.393105** |
| QLoRA relative to frozen | **-0.49%** | **-5.11%** | **-3.73%** | **-2.97%** |

Relative to the old broken QLoRA result, corrected QLoRA improves test CE by
6.58%, NMAD by 38.70%, and the catastrophic fraction by 23.90%.

Validation CE reaches its minimum at epoch 4. Later epochs improve point-error
metrics further (validation NMAD reaches about 0.204) while cross-entropy
worsens, so validation-CE checkpoint selection correctly restores epoch 4.

## Conclusion and optional continuation

This checkpoint does work as a meaningful GLM architecture test: its corrected
QLoRA adapter produces a modest probabilistic improvement and a clear point-
estimate improvement over the matched frozen representation. The original
failure was primarily a representation-interface bug, not evidence of absent
GLM capacity or unusable adapter targets. The initial prediction in
`target_glm_0828.md` that the head optimizer would be the primary cause is
therefore superseded by the Stage 0 result.

Stages 1-5 are not required to explain the failure. If more improvement is
desired, the most useful next experiments are checkpoint selection with a
predeclared CE/NMAD trade-off, then layer/pooling probes and GLM-specific target
coverage. DoRA, IA3, RLVR, and a capacity-ceiling run should wait until one of
those cheaper probes justifies it.

## Artifacts

- Stage 0 report: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage0_invariants/stage0_report.md`
- Corrected cache: `/scratch/.tmp-gsm/aion_output/cache_glm52_paddingfix_0905/glm52_posttraining/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated/n300000_seed42_magnitudes_60fd8c1fd845/GLM_5.2_0.8B_A0.8B_last_len512_kbit_prepared_nf4.pt`
- Frozen result: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/head_only/result.pt`
- QLoRA result: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/qlora/result.pt`
- Adapter: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/qlora/adapter/`
- Prediction head: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/qlora/photoz_head.pt`
- Run summary: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/qlora_run.json`
- Comparison plots: `/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/glm52_photometry_qlora_comparison_*.jpeg`

The previous cache, results, and experimental checkpoints were preserved.

## Reproduce the corrected run

The command below reuses the corrected prepared cache but starts fresh frozen
and QLoRA training artifacts. Change the output and checkpoint suffixes for a
new independent run.

```bash
CUDA_VISIBLE_DEVICES=0 \
GLM52_GPU_DEVICE=0 \
GLM52_INPUT_MODES=photometry \
AION_MAX_ROWS=300000 \
AION_CACHE_ROOT=/scratch/.tmp-gsm/aion_output/cache_glm52_paddingfix_0905 \
GLM52_OUTPUT_DIR=/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-rerun \
GLM52_CHECKPOINT_ROOT=/arc/projects/ots/Cosmic_Imprint_of_Time/glm52_posttraining_checkpoints_paddingfix_rerun \
GLM52_FORCE_FROZEN=1 \
GLM52_QLORA_RESUME=0 \
GLM52_QLORA_EPOCHS=10 \
GLM52_QLORA_BATCH_SIZE=8 \
GLM52_QLORA_GRADIENT_ACCUMULATION_STEPS=2 \
GLM52_EVAL_BATCH_SIZE=8 \
./scripts/glm52-qlora_posttraining.sh
```

## Verification

- `pixi run python -m pytest tests/test_inference_opt_tfm.py`: 8 passed.
- `pixi run python -m py_compile notebooks/glm52_posttraining.py notebooks/glm52_stage0.py`: passed.
- `git diff --check`: passed.

## Stage 1 matched frozen-head controls

Stage 1 completed on the full 60,000-object training and 15,000-object
validation splits. It did not evaluate the test split. The four arms used
identical head initialization and row order; each cached/live pair also used
identical dropout masks. Baseline arms used effective batch 256, constant
`2e-4` learning rate, and `1e-4` weight decay. Direct arms used effective batch
16, the QLoRA head's linear schedule, `1e-2` weight decay, BF16 head autocast,
and gradient clipping.

| Arm | Validation CE | NMAD | Catastrophic fraction | Selected epoch |
|---|---:|---:|---:|---:|
| A: cached + baseline | 4.883122 | 0.236718 | 0.523667 | 5 |
| B: cached + direct | **4.868454** | **0.231249** | **0.513067** | 3 |
| C: live + baseline | 4.883069 | 0.236819 | 0.523333 | 5 |
| D: live + direct | **4.868832** | **0.232367** | **0.513600** | 3 |

The required A/C gate passes: live versus cached differs by -0.0011% in CE and
+0.0426% in NMAD. The B/D pair also passes, differing by +0.0077% in CE and
+0.4836% in NMAD. Relative to live baseline C, live direct D improves CE by
0.292%, NMAD by 1.88%, and catastrophic fraction by 1.86%.

The newly paired A arm agrees with the earlier corrected A reference to 0.051%
in CE. Its NMAD is 1.38% better, which makes the report's symmetric non-gating
1% reference check show `FAIL`; this is benign and does not affect the A/C
continuation gate.

Stage 1 therefore clears both live frozen execution and the direct head
protocol. The old GLM failure cannot be attributed to the live model path or
head optimizer. The already completed corrected QLoRA result improves further
over live direct D by 0.154% CE and 2.42% NMAD, isolating a modest positive
adapter contribution.

- Stage 1 report: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage1_head_controls/stage1_report.md`
- Stage 1 result: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage1_head_controls/stage1_result.pt`
- Validation curves: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage1_head_controls/stage1_validation_curves.png`
- Resumable checkpoint: `/arc/home/gsm/aion_output/figures/glm52-failure-analysis-paddingfix-0905/stage1_head_controls/checkpoint-latest.pt`
- Driver: `notebooks/glm52_stage1.py`
