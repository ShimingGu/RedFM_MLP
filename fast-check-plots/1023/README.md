# Raw zero-shot transfer sanity check — 2026-10-23

Source: completed GalaxiesML-Spec HSC grizy Qwen3.5-4B-Base QLoRA adapter and its matching PDF head.

All weights fixed. No support examples, target-label fitting, calibration, or band remapping. Source redshift grid remains 300 bins over 0–4.5.

| Evaluation | N | NMAD | Outliers | Bias | CE | CRPS | 68% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| HSC: source validation | 5,683 | 0.0268 | 1.07% | 0.0007 | 2.4530 | 0.0274 | 73.02% |
| DESI–Legacy: zero shot | 256 | 0.1537 | 32.42% | 0.0239 | 5.1516 | 0.1547 | 31.25% |
| CLAUDS: zero shot | 256 | 0.1345 | 96.48% | -0.5198 | 19.4920 | 1.4895 | 1.56% |
| Legacy-trained frozen-Qwen head, same Legacy queries (reference) | 256 | 0.0699 | 10.16% | -0.0055 | 3.4468 | 0.0849 | 71.09% |

Raw transfer is substantially degraded on Legacy relative to its own target-trained frozen-head reference. CLAUDS predictions collapse to approximately z=0.13–0.14, with a large negative bias and 96.5% catastrophic disagreement. Its median-centered NMAD alone does not describe this failure.

Source validation is shown for context and was used for checkpoint selection; it is not a matched target population.

Both target samples use validation objects with reference redshifts in 0–4.5, fixed seed 1023, exact-ID exclusions, and >1 arcsec separation from every galaxy in the entire source cohort. CLAUDS is scored against ZPHOT, not spec-z; its original AB-magnitude serializer and training-only imputation are preserved. This also changes numerical representation relative to source fluxes. These small pilot samples do not establish a scientific transfer benchmark.

Source replay: {"max_abs_logit_error": 0.0, "max_abs_pdf_error": 0.0, "n": 32, "passed": true}

Prediction figure: [PNG](hsc_qlora_raw_transfer_predictions.png), [PDF](hsc_qlora_raw_transfer_predictions.pdf).

Calibration figure: [PNG](hsc_qlora_raw_transfer_calibration.png), [PDF](hsc_qlora_raw_transfer_calibration.pdf).

Full run artifacts: `/scratch/RedFM_MLP/fast-check-runs/1023/hsc-qlora-to-legacy-clauds`.

Additional sanity check: [CLAUDS-source transfer to HSC and Legacy](CLAUDS_source_README.md).
