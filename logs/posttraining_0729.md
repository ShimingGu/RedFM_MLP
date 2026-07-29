# Qwen post-training investigation — 2026-07-29

This file records the state of the frozen-Qwen versus QLoRA photo-z
post-training work before the cluster upgrade.

## Current implementation

Launcher:

- `scripts/qwen-qwen_posttraining_comparison_multigpu.sh`

Main comparison program:

- `notebooks/qwen_posttraining_comparison.py`

QLoRA implementation:

- `aion_magnitude/qwen_posttraining.py`

The comparison no longer uses image cutouts or AION image-token inputs.
By default it uses catalogue magnitudes. Setting
`QWEN_USE_MORPHOLOGY=1` adds the multiband morphological catalogue
features through `--use-morphology`.

The staged QLoRA defaults are:

- 10 total epochs
- 3 head-only warm-up epochs, with the Qwen representation detached
- head learning rate: `2e-4`
- LoRA learning rate: `1e-5`
- LoRA targets: `q_proj,k_proj,v_proj,o_proj`
- head gradient clipping: `1.0`
- adapter gradient clipping: `0.1`
- LoRA rank 8, alpha 16, dropout 0.05

These settings replaced the earlier joint-training setup after finding
that the old adapter collapsed the variation in the Qwen representation.

## Output investigated

Current overwritten output directory:

`/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison-e10/`

The current result files were generated on 2026-07-29. Some files under
`.ipynb_checkpoints` are older 2026-07-28 products and must not be treated
as results from the current run.

The run used morphology:

- input mode: `magnitudes_plus_catalogue_multiband_morphology`
- catalogue actually recorded by `prepared.json`:
  `/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits`
- requested maximum rows: 300,000
- random seed: 42
- usable objects after requiring all six morphology bands: **1,658**
- train: **332**
- validation: **83**
- test: **1,243**
- redshift range: 0.0371–5.9402
- feature count: 49

The 49 catalogue inputs were 7 finite magnitude columns
(`u,u_star,g,r,i,z,y`) plus 42 morphology columns. The `Y,J,H,K`
magnitudes were omitted because they had no finite training values in
this six-band-complete cohort.

This cohort reduction is the dominant cause of the bad result. Requiring
all six morphology bands simultaneously left only 332 training objects,
which is not enough for this 4B-model/posterior-head experiment.

## Current metrics

Frozen-Qwen control, test set:

- bias: 0.5387
- outlier fraction: 0.8278
- cross entropy: 5.3856
- CRPS: 0.7725
- NMAD: **0.4536**
- coverage: 0.6742
- PIT mean: 0.2760

Frozen validation:

- cross entropy: 5.3402
- NMAD: 0.4080

QLoRA, test set:

- bias: 0.1520
- outlier fraction: 0.6243
- cross entropy: 5.1196
- CRPS: 0.4117
- NMAD: **0.3396**
- coverage: 0.7506
- PIT mean: 0.4273

QLoRA validation:

- cross entropy: 5.1021
- NMAD: 0.3054
- outlier fraction: 0.5904

For context, the older frozen-Qwen run had test NMAD about **0.2126**.
Therefore the new morphology run is worse overall even though its QLoRA
arm is better than its matching frozen control.

## Training behaviour

Frozen training improved only slowly over 10 epochs:

- validation loss: 5.6186 -> 5.3402
- validation NMAD: 0.5348 -> 0.4080

QLoRA:

- epoch 0, warm-up: validation CE 5.5019, NMAD 0.4862
- epoch 1, warm-up: validation CE 5.2949, NMAD 0.3891
- epoch 2, warm-up: validation CE 5.1558, NMAD 0.3405
- epoch 3, joint: validation CE 5.1171, NMAD 0.3189
- epoch 4, joint: validation CE 5.1021, NMAD 0.3054
- epochs 5–9: training loss continued falling while validation CE worsened
  to about 5.114, indicating overfitting

There were only 210 optimizer updates. The best validation CE occurred
at epoch 4.

## Posterior-collapse diagnostic

The saved predictions from both current arms are nearly independent of
the object.

Frozen:

- standard deviation of per-object posterior mean redshift: 0.000738
- posterior median has one unique value: about 1.9753
- mean total-variation distance of a row from the aggregate posterior:
  0.00179

QLoRA:

- standard deviation of per-object posterior mean redshift: 0.001004
- posterior median has one unique value: about 1.2276
- posterior mode has one unique value
- mean total-variation distance of a row from the aggregate posterior:
  0.00195

Thus the final redshift head is effectively producing a learned prior.

## Direct adapter check

A direct GPU reconstruction compared 256 objects using the frozen cached
embeddings, the base Qwen model, the saved adapter, and the saved
posterior head.

Frozen/base Qwen representation:

- mean element-wise standard deviation: 0.27370
- mean row norm: 152.7743
- row-norm standard deviation: 0.2438
- mean paired-object distance: 19.3574

With the new staged adapter:

- mean element-wise standard deviation: 0.27724
- mean row norm: 152.7735
- row-norm standard deviation: 0.2574
- mean paired-object distance: 19.6013
- mean adapter delta norm: 2.2188
- adapter delta / base norm: 0.01452

This proves that the staged LoRA settings fixed the earlier
**representation-level adapter collapse**. The Qwen embeddings remain
diverse after applying the adapter.

However, the saved posterior head maps those healthy embeddings to
almost constant logits:

- base-input head-logit paired distance: 0.0984
- adapter-input head-logit paired distance: 0.1000
- adapter-input posterior-mean redshift standard deviation: 0.000758

Therefore the regression is not caused by the new adapter collapsing
Qwen. It is caused by the tiny and highly selected 332-object training
cohort, followed by head underfitting/prior prediction and then
overfitting.

## Conclusion

Do not use the current morphology-run figures as evidence that staged
QLoRA is worse. The staged LoRA change succeeded at preserving
representation diversity, but it was tested on a different and
drastically smaller sample.

There are two scientifically meaningful next runs:

1. Run magnitudes-only on the large catalogue to make a fair staged-QLoRA
   comparison with the earlier experiment.
2. For the intended morphology experiment, use morphology from all six
   bands without requiring every object to have all six measurements.
   Preserve objects with missing bands by using per-band availability
   masks plus an explicit imputation strategy. Requiring the six-band
   intersection should not be used unless a much larger updated
   catalogue actually supplies it.

The second option is recommended for the intended all-six-band
morphology model.

## Restart commands

Fair magnitudes-only rerun, with a new output and checkpoint directory:

```bash
AION_OUTPUT_DIR=/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_staged-magnitudes \
QLORA_CHECKPOINT_DIR=/arc/projects/ots/Cosmic_Imprint_of_Time/qlora_checkpoints/qwen-qwen_posttraining-staged-magnitudes-fresh \
QWEN_FORCE_RECOMPUTE=1 \
./scripts/qwen-qwen_posttraining_comparison_multigpu.sh --no-qlora-resume
```

`QWEN_USE_MORPHOLOGY` is intentionally omitted above. The launcher
defaults to magnitudes-only.

For any fresh morphology rerun, use a new output directory and a new
checkpoint directory, set `QWEN_USE_MORPHOLOGY=1`, and pass
`--no-qlora-resume`. Do not reuse the `e10` output directory because it
has already mixed old and new timestamps.

The launcher currently defaults to the updated catalogue:

`/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits`

Verify the availability counts in that file after the cluster returns.
The investigated run fell back to or explicitly used the older
`..._morphological_multiband.fits` catalogue, as recorded in
`prepared.json`.

