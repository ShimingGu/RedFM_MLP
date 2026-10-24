# Experiment B — Warm-start Level-1 Band-Aware QLoRA

Implement and execute continued SFT starting from the **same original completed HSC Qwen3.5-4B-Base QLoRA checkpoint used in Experiment A**.

Do not initialise from Experiment A's newly trained checkpoint.

## Objective

Determine whether an already trained photo-z foundation model can adapt to a new input representation that explicitly identifies the survey and physical photometric bands.

This is Level 1: **explicit band identity without wavelength or bandwidth metadata**.

We want to determine whether the previously learned photo-z capability can be retained while improving cross-survey input interpretability.

## Checkpoint and data

- Load the archived HSC QLoRA adapter and trained PDF head.
- Preserve Qwen3.5-4B-Base, adapter rank, trainable modules and PDF head architecture.
- Use the same 22,444 HSC training and 5,683 validation objects as Experiment A.
- Retain all original photometric values and supervision labels.
- PDF grid: 300 bins spanning z = 0–4.5.

Do not reinitialise the prediction head.

## Input representation

Use the existing Level-1 serializer implemented in `aion_magnitude/filter_photometry.py`.

The input must explicitly identify the observational system and actual physical band.

Example structure:

```text
Survey: HSC
HSC_g: flux=..., sigma=...
HSC_r: flux=..., sigma=...
HSC_i: flux=..., sigma=...
HSC_z: flux=..., sigma=...
HSC_y: flux=..., sigma=...
EBV: ...
```

Retain the exact numerical values, measurement units, validity masks and missing-value conventions from the corresponding native HSC data.

Do not include effective wavelength, bandwidth or full transmission-curve information.

For target-domain evaluation, use the correct actual band identities:

- DESI–Legacy: grzW1W2.
- CLAUDS: inspect and use its actual band definitions.

Never relabel measurements according to their array position.

## Initial diagnostics

Before training:

1. Load the archived adapter and PDF head.
2. Run zero-update predictions using Level-1 HSC inputs.
3. Quantify the initial performance change caused solely by changing the serializer.
4. Verify actual tokenizer input strings and ensure no truncation.
5. Check that the same training objects, validation objects and redshift labels are used as in Experiment A.

## Training

Use the same continued-SFT protocol as Experiment A:

- Maximum 2 additional epochs.
- No head-only warmup.
- Adapter LR = 3e-6.
- Head LR = 1e-5.
- Effective batch size = 16.
- Seed = 1023.
- Same adapter coverage, rank, training schedule and checkpoint-selection metric.
- Validation approximately every 250 optimizer updates.
- Include the zero-update checkpoint among candidates.

Select checkpoints using source-validation CE, never target-domain performance.

## Evaluation

Report source-validation CE, NMAD, outliers, bias, CRPS and 68% coverage, both before and after adaptation.

Then evaluate the selected checkpoint on the identical Legacy and CLAUDS target-validation objects used in Experiment A, without target fitting or calibration.

Separate:

- Initial damage caused by the changed serializer.
- Recovery through continued SFT.
- Final source-domain performance.
- Cross-survey transfer performance.

CLAUDS uses ZPHOT as its reference, unlike DESI spectroscopic redshifts. Preserve and report this distinction.

## Outputs

Save results under:

`filter-aware-runs/1023/hsc-warmstart/level1/`

Preserve the original checkpoint and previously completed experiments.

Use the existing tested infrastructure and registry. Do not introduce unrelated model architecture or PEFT changes.

Do not interrupt running GPU jobs.

The final report should answer:

**Can the previously trained Qwen adapter acquire explicit photometric-band identity information without sacrificing its existing photo-z performance, and does this improve cross-survey transfer relative to the matched Level-0 control?**

Execute the experiment and save all reproducible artifacts.