# Stage 0 — Diagnose HSC-to-CLAUDS Failure Before Further Training

## Scientific objective

We are studying whether Qwen3.5-4B-Base post-trained for photometric redshifts can generalise across genuinely different survey datasets.

The ultimate target is CLAUDS. Transfer between GalaxiesML-Spec and DESI–Legacy alone does not constitute success.

The existing HSC-source QLoRA models achieve approximately NMAD = 0.027 in-domain but catastrophically fail on CLAUDS.

Recent common-flux and filter-aware evaluations established:

- Approximately 96–97% catastrophic outliers on 256 CLAUDS validation objects.
- Predictions collapse near z = 0.08–0.13.
- Common-flux conversion does not rescue transfer.
- Restricting CLAUDS to actual HSC grizy bands does not rescue transfer.
- 237 of the 256 CLAUDS galaxies possess all five HSC bands.
- The reference redshifts lie within the HSC model's 0–4.5 output grid.
- An historical CLAUDS magnitude zero-point inconsistency was found and corrected in the optional flux interface.

The physically correct conversion for raw CLAUDS microJy fluxes is:

`f_nMgy = 0.2754228703 * f_microJy`

The archived native serializer must remain unchanged for reproducibility.

## Your task

**Determine whether the observed failure can be explained by an input-interface discrepancy or by source/target photometric distribution shift.**

This is a diagnostic experiment.

Do not train or modify any model.

## 1. Inspect existing artifacts

Locate and inspect:

- The original completed HSC QLoRA adapter and trained PDF head.
- The selected Level-0/1/2 continued-SFT checkpoints.
- Existing CLAUDS and GalaxiesML-Spec loaders.
- `aion_magnitude/filter_photometry.py`
- `configs/filter_registry_v1.json`
- `notebooks/qwen_filter_aware.py`
- The completed CLAUDS common-flux report and input manifests.

Use the repository's actual paths and configurations.

Do not invent dataset fields or preprocessing conventions.

## 2. Audit photometric equivalence

Compare the HSC-source grizy observations with CLAUDS's actual HSC grizy observations.

For each band, inspect:

- Original catalogue quantity and unit.
- Conversion into nanomaggies.
- Measurement uncertainty.
- Sign of flux and treatment of non-detections.
- Missingness and quality flags.
- Extinction treatment.
- Photometric measurement definition, including possible pipeline/version differences.
- Whether corresponding bandpass identifiers refer to the same physical system.

Verify the CLAUDS microJy conversion and the historical 0.9-mag offset.

Preserve genuine signed flux measurements.

Do not infer unavailable uncertainty or EBV values.

Distinguish filter-system equivalence from measurement-pipeline equivalence: two catalogues may use HSC grizy but have different selection, calibration, aperture or flux measurement conventions.

## 3. Source-exact serialization experiment

The previous Level-1/2 evaluations used their own filter-aware schemas.

Now construct an additional diagnostic in which CLAUDS's actual HSC grizy photometry is serialized using the original source HSC model's exact input schema.

Match:

- Field names.
- Band order.
- Flux units.
- Numerical precision.
- Missing-value conventions.
- Uncertainty representation.
- Token ordering.
- Final prediction/query token.

Use the true observed CLAUDS measurements. Do not substitute HSC-source measurements or invent metadata.

Run separate controlled variants if the original schema requires fields unavailable for CLAUDS.

If the source schema contains an explicit survey identifier, test its effect separately. Replacing a target identifier with a source identifier is allowed only as a clearly labelled counterfactual diagnostic, not as a scientifically truthful final input representation.

Compare against the archived Level-0 baseline and the existing Level-1/2 common-flux results.

Keep the model weights and PDF head fixed.

## 4. Photometric support analysis

Construct a common source/target feature space using actual HSC grizy observations.

Include suitable combinations of:

- Log or asinh-transformed fluxes, with a transformation valid for negative measurements.
- Colours or flux ratios where mathematically valid.
- Apparent magnitude or brightness scale.
- Relative and absolute uncertainties.
- S/N.
- Missingness and quality information.

Do not naively standardise an unbounded mixture of fluxes, magnitudes and non-detections.

Fit all preprocessing statistics on the HSC source training split only.

Use photometry-only methods to assess whether CLAUDS galaxies lie within the source distribution.

Implement at least two complementary diagnostics, for example:

1. Source-standardised k-nearest-neighbour distances.
2. A source-versus-target classifier with an independent validation split and a held-out target sample.

For density-ratio or support estimates, avoid treating a perfectly discriminating domain classifier as proof of literal zero support.

Use source cross-validation to choose or assess thresholds without using CLAUDS redshift labels.

Stratify by photometric completeness, brightness and S/N.

Do not filter target galaxies by their prediction errors.

## 5. Conditional evaluation

Use the existing 256 CLAUDS diagnostic objects.

Partition them into pre-specified common-support and out-of-support strata based only on photometric features.

Evaluate frozen Qwen predictions in each stratum.

Report:

- Number of objects.
- CE, NMAD, outlier fraction, bias and CRPS.
- 68% coverage.
- Predicted redshift distributions.
- Collapse/concentration statistics.
- Relative location within the source photometric distribution.

The CLAUDS labels are ZPHOT, not spectroscopic redshifts.

Do not use them for determining support thresholds, feature selection, model configuration or checkpoint selection.

Include plots comparing source and target feature distributions and a few representative, pre-selected input records.

## 6. Decision rules

Answer these questions:

A. Are CLAUDS HSC grizy measurements numerically and semantically compatible with the source HSC representation?

B. Does source-exact serialization reduce the collapse?

C. What fraction of CLAUDS appears to have meaningful photometric support in the HSC training set?

D. Do common-support CLAUDS objects still fail?

E. Are the remaining failures more consistent with unresolved measurement differences, population shift or learned representation limitations?

Do not claim that source/target support matching proves conditional invariance of redshift given photometry.

Do not assume a failure of current transfer proves that the foundation model lacks astrophysical knowledge.

## 7. Deliverables

Create a new reproducible Stage-0 directory.

Save:

- Input and unit audit.
- Serializer comparisons.
- Source/target support diagnostics.
- Fixed-weight performance tables.
- Conditional scatter and PDF plots.
- Representative actual tokenizer inputs.
- Object IDs, hashes, configurations and commands.
- A concise scientific report.

Use no target-label training or calibration.

Do not touch held-out test objects.

**Execute all feasible no-training diagnostics rather than stopping at a proposal.**

Conclude by stating whether Stage 1 and Stage 2 can proceed with a trustworthy, physically consistent photometric interface and what uncertainties remain.