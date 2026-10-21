# GLM-5.2-0.8B-A0.8B: GalaxiesML-Spec sanity check and numeric MLP

Date: 2026-10-21

**Both GalaxiesML-Spec runs completed 100 epochs. GLM QLoRA memorized all 512 training objects' redshift bins, with zero catastrophic outliers. The numeric MLP reached validation NMAD 0.024848 and an outlier fraction of 1.1614%. The GLM run evaluated training fit only, so these two runs do not establish a GLM-versus-MLP validation ranking.**

This report covers the spectroscopic counterparts of the two CLAUDS checks in [glm52_test_1021.md](glm52_test_1021.md). Numerical-sensitivity and toy-pretraining-versus-random-initialization experiments are separate; their results are not included here.

## Data and verification

- Source: `GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5`, matched to the DESI DR1 quality catalogue.
- **True redshift: `DESI_fibermap/DESI_redshift`, a spectroscopic label**, checked against the primary DR1 redshift. No CLAUDS ZPHOT labels are used.
- Inputs: HSC PDR2 grizy cModel photometry converted to AB nanomaggy fluxes and one-sigma flux errors, plus MW EBV: 11 physical features. No images, spectral pixels, or morphology.
- Selection: primary galaxy redshifts, `ZWARN=0`, `COADD_FIBERSTATUS=0`, `DELTACHI2>40`, finite redshift in [0, 4.5], and at least three valid photometric bands. This is a uniform training-quality selection, not a tracer-specific LSS cut.
- Cohort: 112,002 eligible rows, with 22 duplicate imaging rows removed, leaving **111,980 objects**. No native/reference redshift disagreements were retained or required removal in this cohort.
- Grouped seed-42 splits: **22,444 training; 5,683 validation; 83,853 held-out test**. The split rule hashes the imaging group, preserving separation across splits.
- Prediction grid: 300 bins over [0, 4.5], with bin width 0.015. Point metrics use the PDF median; catastrophic outliers satisfy `abs(z_pred - spec_z)/(1 + spec_z) > 0.15`.

The saved cohort object IDs and split assignments agree between both runs. All 512 selected GLM objects belong to the training split; their saved sample IDs and evaluation IDs match. The MLP evaluated exactly the 5,683 validation IDs. Both result files agree with their JSON summaries. **Neither check produced held-out test predictions or metrics.**

## 1. Small-sample overfitting sanity check

The run selected 512 objects from the GalaxiesML-Spec training split and used those same objects for fitting, checkpoint selection, and final evaluation. Inputs retained explicit `NA` values and the existing five-decimal numerical formatting. The tokenizer audit passed: the longest training input was 204 tokens against a maximum length of 512.

The current restricted QLoRA configuration trained attention/indexer adapters and the prediction head; dense MLPs and routed experts remained frozen. Settings: 100 epochs, comprising 3 head-only warmup epochs and 97 joint epochs; microbatch 8; accumulation 2; head LR `2e-4`; adapter LR `1e-5`; rank 8; alpha 16; adapter dropout 0.05; last-token pooling. Seed: 42.

The checkpoint selected by training-sample evaluation CE was **epoch 99**, counting from one.

| Training-sample metric | Selected checkpoint |
|---|---:|
| Cross-entropy | 0.00046654 |
| NMAD | 0.00437010 |
| Catastrophic outlier fraction | 0% (0/512) |
| Mean CRPS | 0.00749651 |
| Correct target-bin classification | 100% (512/512) |
| Normalized mean bias | -0.00005146 |

Training-sample evaluation CE fell from **5.532453 after epoch 1** to **0.00046654** at the selected checkpoint. Epoch 100 had CE 0.00046682. Both saved histories contain all 100 epochs.

**Interpretation:** the corrected GLM pipeline and existing adapters can fit this small spectroscopic mapping extremely closely. This rules out a complete inability to learn these training examples. It does not demonstrate validation generalization or establish a posttraining limit.

Perfect bin classification does not imply exact continuous redshifts: the PDF median is a bin center, so nonzero NMAD remains. The nearly one-hot training PDFs also give zero reported p16–p84 coverage under the bin-center quantile definition; this memorization result is not evidence of calibrated continuous-redshift uncertainty.

## 2. Matched numeric MLP

The MLP used all **22,444 training objects** and selected its checkpoint using the **5,683 validation objects**. It received the same 11 physical inputs and spectroscopic labels. Training-only median imputation, 11 missingness indicators, and training-only standardization produced 22 numeric input columns. Imputation values, means, scales, feature order, and selected weights are saved with the result.

Configuration: repository tabular MLP, binned cross-entropy, AdamW, LR `1e-3`, weight decay `1e-4`, batch size 256, 100 epochs, seed 42. Validation cross-entropy selected **epoch 29**.

| Validation metric | Selected checkpoint, epoch 29 |
|---|---:|
| Cross-entropy | 2.372149 |
| NMAD | 0.024848 |
| Catastrophic outlier fraction | 1.1614% (66/5,683) |
| Mean CRPS | 0.026556 |
| Normalized mean bias | -0.00088782 |
| p16–p84 coverage | 67.8691% |

| Epoch | Training CE | Validation CE | Validation NMAD | Validation outliers |
|---|---:|---:|---:|---:|
| 1 | 3.465240 | 2.848918 | 0.041856 | 2.0764% |
| 29, selected | 2.244671 | 2.372149 | 0.024848 | 1.1614% |
| 100 | 1.484324 | 2.867019 | 0.023189 | 1.2669% |

Training CE kept improving after epoch 29 while validation CE worsened. The final epoch has slightly better NMAD but worse CE and outlier fraction than the selected checkpoint. The reported result consistently uses validation CE for selection; it does not substitute a different epoch for each metric.

**Interpretation:** the native GalaxiesML-Spec inputs support a useful validation baseline. The MLP demonstrates generalization under this split, while also showing that longer training need not improve every validation metric.

## What these checks establish

Both datasets' small-sample GLM checks passed: the earlier CLAUDS run nearly memorized its sample, and the GalaxiesML-Spec run correctly classified every training redshift bin. This strengthens the evidence that the corrected implementation can learn a small mapping.

The MLP is matched to the GalaxiesML-Spec input information, labels, and reference split definitions. Its actual training set is larger than the GLM sanity-check sample, and its reported evaluation split is different. **Do not compare GLM training CE/NMAD with MLP validation CE/NMAD as a model-performance ranking.** Unlike the CLAUDS report, these artifacts contain no full-cohort GLM validation comparator.

CLAUDS and GalaxiesML-Spec also differ in photometric inputs, selection, target provenance, and redshift bins. Their absolute metrics do not isolate an effect of changing the dataset or using spectroscopic labels.

These single-seed checks do not establish why GLM generalizes poorly, whether toy pretraining helps or harms learning, or whether numerical discontinuities exist. They do not establish a need for full fine-tuning. A matched GLM validation result and the separate planned diagnostics are needed before making those stronger claims.

## Artifacts

- [GLM sanity-check summary](/arc/home/gsm/aion_output/figures/glm52-capacity-galaxiesml-spec-1021/overfit-n512/summary.json), [result and history](/arc/home/gsm/aion_output/figures/glm52-capacity-galaxiesml-spec-1021/overfit-n512/result.pt), [training sample IDs](/arc/home/gsm/aion_output/figures/glm52-capacity-galaxiesml-spec-1021/overfit-n512/training_sample.json).
- [Numeric MLP summary](/arc/home/gsm/aion_output/figures/glm52-capacity-galaxiesml-spec-1021/matched-numeric-mlp/summary.json), [result, history, weights, and preprocessing](/arc/home/gsm/aion_output/figures/glm52-capacity-galaxiesml-spec-1021/matched-numeric-mlp/result.pt).
- [Survey preparation and quality selection](/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/prepared.json).
- Driver: [glm52_capacity_checks.py](notebooks/glm52_capacity_checks.py). Commands: [recent_runs.txt](notebooks/recent_runs.txt:259).
