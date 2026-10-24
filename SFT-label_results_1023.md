# Filter-aware input pilot — 2026-10-23

Implemented from `SFT-label_1023.md`. All results below are from a small source-only pilot, with no held-out test evaluation and no target-label fitting.

All three arms remain severely underfit in-domain. These runs verify the implementation, but do not establish that filter metadata improve cross-survey generalisation.

## Matched HSC-source training

Qwen3.5-4B-Base, fresh rank-8 QLoRA adapters and a newly initialized instance of the existing 256-hidden-unit PDF head; 300 bins over 0.0–4.5. All arms use the same 128 source-training galaxies, 64 source-validation galaxies, seed 1023, 5 epochs, 3 head-warmup epochs, head LR 0.0002, adapter LR 1e-05, microbatch 1 and accumulation 16. The best validation-CE checkpoint is evaluated. These short runs must not be compared as matched-budget counterparts of the archived 10-epoch/full-cohort models.

| Level | Best epoch | Val CE | NMAD | Outliers | Bias | CRPS | 68% coverage | Peak allocated GiB | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 5.1887 | 0.3905 | 100.00% | 1.1654 | 0.8587 | 28.12% | 5.45 | 488.8 |
| 1 | 5 | 5.1630 | 0.3884 | 100.00% | 1.1537 | 0.8632 | 28.12% | 5.45 | 454.2 |
| 2 | 5 | 5.1249 | 0.3820 | 100.00% | 1.1184 | 0.8422 | 28.12% | 5.45 | 503.3 |

Query bootstrap intervals are descriptive and conditional on this one training seed and selected checkpoints; they do not quantify training-seed, checkpoint-selection or survey variation. These validation queries also selected checkpoints, so the intervals are not independent confirmation. Negative differences favour Level 2.

```json
{
  "level2_minus_level1_ce": {
    "mean": -0.038019970059394836,
    "conditional_query_bootstrap_95": [
      -0.058770418912172315,
      -0.017752872593700886
    ]
  },
  "level2_minus_level1_crps": {
    "mean": -0.020970601588487625,
    "conditional_query_bootstrap_95": [
      -0.02148843677714467,
      -0.020404971297830345
    ]
  }
}
```

## Fixed-weight cross-survey checks

Targets reuse the same 256 validation queries per catalogue from the HSC-source fast-check. That audit excluded the entire original source cohort by IDs and a 1-arcsec positional check; the new source-training subset is contained within that cohort. Legacy uses DESI spec-z; CLAUDS uses ZPHOT and unchanged, training-imputed AB magnitudes. The input-format and label-definition shifts for CLAUDS remain explicit confounders.

| Level | Target | N | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | clauds | 256 | 5.7827 | 0.3299 | 64.06% | 0.1999 | 0.5691 | 93.36% |
| 0 | desi-ls | 256 | 5.3362 | 0.5573 | 96.48% | 0.8823 | 0.7407 | 53.12% |
| 1 | clauds | 256 | 5.8087 | 0.3188 | 64.06% | 0.1684 | 0.5645 | 93.75% |
| 1 | desi-ls | 256 | 5.3272 | 0.5543 | 96.48% | 0.8721 | 0.7420 | 53.12% |
| 2 | clauds | 256 | 5.8140 | 0.3154 | 61.33% | 0.1557 | 0.5614 | 93.36% |
| 2 | desi-ls | 256 | 5.3089 | 0.5440 | 94.53% | 0.8471 | 0.7248 | 53.12% |

## Interpretation and limits

Level 1 changes the structure and spelling of already band-labelled Level-0 inputs. Level 2 differs from Level 1 only through pivot wavelength and rectangular equivalent width. All three native arms retain the same photometric values. The optional CLAUDS common-flux representation was implemented and token-audited separately; it was not mixed into this native-input comparison.

The pilot has 40 optimizer updates, including 16 after head warmup. It tests that the complete training and transfer paths work. Single-survey filter metadata are constant across galaxies; even an improvement would not establish that the model learned a transferable physical relationship. A promising result needs repeated seeds, adequate source learning, and matched-length/shuffled-metadata controls before a physics or novelty claim.

All three arms remain severely underfit in-domain. These runs verify the implementation, but do not establish that filter metadata improve cross-survey generalisation.

Legacy-source and CLAUDS-source Level-0/1/2 training, DoRA pilots, common-flux training/transfer, mixed-survey training and final test evaluation were not run. The full Legacy cohort cache was unavailable in this session; only existing audited query photometry was used, with actual filter identities recovered from Tractor records. The existing loader remains available when that cache is restored.

The report uses historical binned CE/CRPS and bin-centre interval coverage for comparability. PIT figures interpolate within each bin; historical upper-bin PIT and finite-bin coverage effects should not be interpreted as exact continuous calibration.

## Implementation and validation

All three serializers, optional raw-flux harmonisation for CLAUDS, QLoRA/DoRA integration and validation-only training are implemented. The offline registry contains 21 pinned response profiles with hashes, nm units, reproducible pivot/width definitions and documented nominal-response approximations. Legacy query filters are resolved from actual RELEASE values and matching saved fluxes. HSC r/r2 and i/i2 exposure mixtures are absent from the original metadata; original r/i profiles remain explicit approximations.

The relevant test suite passes 43 tests. Level-0 serialization exactly matches the 512 archived Legacy/CLAUDS query strings; the 32-object GPU HSC source replay has maximum logit error 0.0. Complete actual inputs are token-audited, with IDs and missing masks retained. Native/common-flux CLAUDS bundles share one portable cohort fingerprint and match archived validation IDs/labels. Source test evaluation is disabled, and target evaluations use validation objects only.

Measured token lengths, filter provenance, the corrected CLAUDS split-string storage, full launch commands and unexecuted follow-ups are documented in [the implementation notes](filter-aware-runs/1023/README.md).

## Files and reproduction

- Serializer: `aion_magnitude/filter_photometry.py`.
- Offline response registry: `configs/filter_registry_v1.json`; vendored originals, normalized curves and provenance notes: `configs/filter_responses_v1/`.
- Validation-only trainer integration: `aion_magnitude/qwen_posttraining.py` (`test_dataset=None`).
- Prepare/train/replay/transfer: `notebooks/qwen_filter_aware.py`.
- Sequential safe launcher: `scripts/run_filter_aware_pilot.py`.
- Tests: `tests/test_filter_photometry.py`.
- Run manifest, shared cohort, actual tokenizer inputs and per-arm results: `filter-aware-runs/1023/hsc-pilot`.
- Plot generation: `notebooks/plot_filter_aware.py`.
- Full implementation and launch notes: `filter-aware-runs/1023/README.md`.

![Learning curves](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-pilot/plots/learning_curves.png)

![Validation predictions](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-pilot/plots/validation_predictions.png)

![Cross-survey predictions and PDFs](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-pilot/plots/transfer_predictions.png)
