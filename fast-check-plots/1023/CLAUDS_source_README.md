# CLAUDS-source raw zero-shot transfer — 2026-10-23

Source: original 10-epoch CLAUDS Qwen3.5-4B-Base QLoRA adapter and matching PDF head. Weights remain fixed; no target calibration examples or target-label fitting.

| Evaluation | N | NMAD | Outliers | Bias | CE | CRPS | 68% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLAUDS: source validation | 15,000 | 0.0738 | 18.39% | 0.0100 | 3.9305 | 0.2271 | 68.96% |
| GalaxiesML-Spec: zero shot | 256 | 0.7615 | 100.00% | 2.7814 | 7.9744 | 2.8042 | 0.00% |
| DESI–Legacy: zero shot | 256 | 1.0408 | 100.00% | 2.3175 | 7.6213 | 2.5523 | 0.00% |

Both target samples fail under this raw interface: predicted medians concentrate near z=3.8–3.9, all 256 queries in each sample are catastrophic outliers, and nominal 68% interval coverage is zero. The source replay exactly reproduces the archived model, so this is not a checkpoint-restoration mismatch.

Source reference labels are CLAUDS ZPHOT; both target labels are DESI spec-z. Target inputs retain their native observed flux/error/EBV serializers; the source was trained on CLAUDS AB magnitudes. This deliberately tests raw transfer with both input-system/format and label-definition shifts, without harmonizing bands or units through remapping.

The original 300-bin source grid is retained: 0.005000030156224966–5.992499828338623. Cross-entropy values should not be compared directly against the earlier HSC-source pilot's different grid.

Queries are 256 validation objects per target, within the source grid and the target catalogue's 0–4.5 range. Exact HSC object matches are excluded where available, and queries are farther than 1 arcsec from every object in the entire original CLAUDS source cohort. Source validation is shown only as context; it was used for checkpoint selection and is a different population.

Source replay: {"max_abs_logit_error": 0.0, "max_abs_pdf_error": 0.0, "n": 32, "passed": true}

Predictions: [PNG](clauds_qlora_raw_transfer_predictions.png), [PDF](clauds_qlora_raw_transfer_predictions.pdf).

Calibration: [PNG](clauds_qlora_raw_transfer_calibration.png), [PDF](clauds_qlora_raw_transfer_calibration.pdf).

Run artifacts: `/scratch/RedFM_MLP/fast-check-runs/1023/clauds-qlora-to-hsc-legacy`.
