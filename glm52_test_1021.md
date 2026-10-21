# GLM-5.2-0.8B-A0.8B: training-fit sanity check and matched numeric MLP

Date: 2026-10-21

Scope: this report records the completed **CLAUDS / ZPHOT** checks. Subsequent commands in `notebooks/recent_runs.txt` were switched to **GalaxiesML-Spec / DESI spectroscopic z**, with separate output directories under `glm52-capacity-galaxiesml-spec-1021`. The numbers below remain the historical CLAUDS results, not results of the new spectroscopic checks.

**Both runs completed 100 epochs. The small-sample sanity check passed: GLM QLoRA can nearly memorize 512 training objects. The matched numeric MLP substantially outperforms the corrected GLM QLoRA reference on validation. These results establish neither a posttraining ceiling nor a need for full fine-tuning.**

## Shared data and verification

The checks follow [target_glm_1021.md](target_glm_1021.md), using the corrected CLAUDS photometry-only reference from the earlier [implementation diagnosis](diagnosis_glm_0906.md).

- Catalogue: `COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits`.
- Inputs: 11 magnitudes, `u`, `u_star`, `g`, `r`, `i`, `z`, `y`, `Y`, `J`, `H`, and `Ks`. No morphology or images.
- Target: catalogue **ZPHOT**, inherited from the CLAUDS pipeline. Although tensors are named `z_spec`, these checks do not measure accuracy against spectroscopic redshifts.
- Reference cohort: 300,000 objects; seed 42; 60,000 train, 15,000 validation, 225,000 test. Median imputation is fitted on the reference training split.
- Shared output grid: 300 redshift bins spanning approximately 0.00500003–5.99249983, inherited from the corrected reference.

Both saved cohort manifests have identical object IDs and split assignments. All 512 sanity-check objects belong to the training split, and their evaluation IDs match the selected sample. The MLP validation IDs match the reference validation cohort. Its copied GLM comparison metrics agree with the original corrected QLoRA result.

**Neither new check evaluated the held-out test split.** The driver checks saved test IDs and labels when reconstructing the reference cohort, but computes no new test predictions or metrics. The historical reference already contains test results; those are not used in this comparison.

## 1. Small-sample overfitting sanity check

The run selected 512 objects from the reference training split and used those same objects for fitting, checkpoint selection, and final evaluation. It reused the reference imputation and current attention/indexer QLoRA targets, leaving dense MLPs and routed experts frozen.

Configuration: 100 epochs, comprising 3 head-only warmup epochs and 97 joint epochs; microbatch 8; gradient accumulation 2; head LR `2e-4`; adapter LR `1e-5`; rank 8; alpha 16; adapter dropout 0.05; last-token pooling; maximum length 512. The selected checkpoint is epoch **87**, counting epochs from one.

| Training-sample metric | Selected checkpoint |
|---|---:|
| Cross-entropy | 0.052827 |
| NMAD | 0.002740 |
| Catastrophic outlier fraction | 1.5625% (8/512) |
| Mean CRPS | 0.019808 |
| Correct target-bin classification | 98.0469% (502/512) |

Training-sample evaluation CE fell from **5.612206 after epoch 1** to **0.052827** at the selected checkpoint. The final epoch had CE 0.052897, consistent with the fit having largely settled.

**Interpretation:** the current pipeline and restricted adapter setup can learn this small mapping very closely. A complete inability to fit training data is not the explanation for the poor full-cohort validation result. This is a memorization check, not evidence of generalization, calibrated uncertainty, or the model's maximum achievable performance.

## 2. Matched numeric MLP

The MLP used the same input information, targets, cohort, split assignments, and redshift bins as the corrected full-cohort GLM QLoRA reference. Numeric inputs were standardized using training-only means and standard deviations. It used the repository's tabular MLP with binned cross-entropy, AdamW, LR `1e-3`, weight decay `1e-4`, and batch size 256.

The MLP trained for 100 epochs. Checkpoint selection used validation cross-entropy and selected epoch **9**. The GLM comparison is the existing corrected **10-epoch** QLoRA run, not a new full-cohort GLM training run. Data are matched; optimization protocols and training budgets differ.

| Validation metric | Corrected GLM QLoRA | Numeric MLP, epoch 9 | Relative reduction |
|---|---:|---:|---:|
| Cross-entropy | 4.861322 | 4.013014 | 17.45% |
| NMAD | 0.226749 | 0.077974 | 65.61% |
| Catastrophic outlier fraction | 50.6933% | 21.0667% | 58.44% |
| Mean CRPS | 0.394698 | 0.241211 | 38.89% |

Point metrics use the PDF median, with catastrophic outliers defined by `abs(z_pred - ZPHOT) / (1 + ZPHOT) > 0.15`. Lower values are better for all four table metrics. The outlier reduction is **29.63 percentage points**.

The MLP's training CE continued falling, from **3.886468 at epoch 9** to **2.852326 at epoch 100**, while validation CE worsened from **4.013014** to **4.874993**. Its late-epoch NMAD improved somewhat, but the reported model remains the checkpoint selected by validation CE. This shows why additional epochs alone should not determine the selected model.

**Interpretation:** substantially better validation predictions are achievable from the same catalogue information. The poor corrected GLM result is not an unavoidable error floor imposed by these inputs. This comparison does not identify which combination of checkpoint initialization, numeric representation, optimization, or adaptation restrictions causes the gap.

## Decision

The two immediate checks are complete. The pipeline can fit a small training sample, and the matched MLP establishes a much stronger validation reference. Broader QLoRA or final-block tuning remains **very optional**, only to investigate a specific remaining question about adaptation restrictions. **Full fine-tuning is not currently necessary.**

These are single-seed results. Any configuration chosen for a stronger performance claim should be repeated across seeds, with test evaluation reserved for final selection as planned. Neither the memorization result nor the MLP comparison establishes the tiny GLM checkpoint's posttraining limit.

## Artifacts

- [Sanity-check summary](/arc/home/gsm/aion_output/figures/glm52-capacity-1021/overfit-n512/summary.json), [result and training history](/arc/home/gsm/aion_output/figures/glm52-capacity-1021/overfit-n512/result.pt), [selected training objects](/arc/home/gsm/aion_output/figures/glm52-capacity-1021/overfit-n512/training_sample.json).
- [Numeric MLP summary](/arc/home/gsm/aion_output/figures/glm52-capacity-1021/matched-numeric-mlp/summary.json), [result, selected weights, scaling, and history](/arc/home/gsm/aion_output/figures/glm52-capacity-1021/matched-numeric-mlp/result.pt).
- [Corrected GLM reference](/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry/qlora/result.pt).
- Driver: [glm52_capacity_checks.py](notebooks/glm52_capacity_checks.py). Commands: [recent_runs.txt](notebooks/recent_runs.txt:259).
