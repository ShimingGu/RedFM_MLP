# HSC ↔ DESI–Legacy Matched Cross-Survey Transfer Benchmark

## Scientific objective

We are studying the generalisation of post-trained foundation models for photometric-redshift inference.

Our immediate objective is to establish a scientifically controlled, bidirectional transfer benchmark between:

- **GalaxiesML-Spec:** HSC grizy photometry with DESI spectroscopic-redshift labels.
- **DESI–Legacy:** Legacy grzW1W2 photometry with DESI spectroscopic-redshift labels.

These are different photometric measurement systems, but both use DESI spectroscopy.

This experiment should isolate observational-domain transfer from changes in galaxy population and redshift distribution as far as the available data permit.

**The primary experiment should use the same physical galaxies, observed through both HSC and Legacy photometric systems.**

This is a frozen-model evaluation and dataset-audit task, not a new training exercise.

CLAUDS is outside the scope of this experiment.

## 1. Inspect available checkpoints and data

Identify the actual completed checkpoints for:

1. Original GalaxiesML-Spec HSC Qwen3.5-4B-Base QLoRA and its trained PDF head.
2. HSC warm-start Level-0, Level-1 and Level-2 QLoRA checkpoints.
3. Original DESI–Legacy Qwen3.5-4B-Base QLoRA and its trained PDF head.
4. Any completed, compatible HSC/Legacy DoRA models, if available.

Do not assume a checkpoint exists merely because a run was previously planned.

For each available model, record:

- Exact source catalogue and training cohort.
- Training, validation and any test object IDs.
- Adapter/head checkpoint identities and hashes.
- PDF redshift grid and bin definitions.
- Source serializer, feature definitions and units.
- Actual model-selection procedure.
- Whether its training and validation objects overlap the other survey.

Inspect the GalaxiesML-Spec HDF5 and original DESI–Legacy Tractor data.

The GalaxiesML-Spec data may contain both HSC and Legacy photometry for the same galaxies. Verify that the Legacy measurements have the fluxes, uncertainties, EBV information, filter provenance and release metadata necessary to reproduce the original Legacy-model input.

Do not silently treat differently processed photometric columns as equivalent.

## 2. Build the paired-galaxy cohort

Construct a dataset where every row corresponds to one physical galaxy, with:

```text
galaxy_id
coordinates
DESI_TARGETID
spec_z

HSC:
    g, r, i, z, y
    fluxes and uncertainties
    validity masks

Legacy:
    g, r, z, W1, W2
    fluxes and uncertainties
    EBV
    RELEASE / instrument metadata
    validity masks
```

Use the actual raw measurements and preserve their source-specific definitions.

Verify:

- Unique and unambiguous cross-identifications.
- Exact TARGETID matching where available.
- Positional matches with a recorded angular tolerance otherwise.
- No ambiguous many-to-one crossmatches.
- Consistency of the DESI spectroscopic-redshift labels.
- Consistency of spectroscopic quality flags and redshift validity.
- Whether labels originate from identical DESI reductions.

If the two catalogues carry conflicting redshift labels, audit the discrepancy instead of arbitrarily choosing one.

The primary cohort must contain objects with usable photometry in both systems, but avoid unnecessarily restrictive quality cuts.

Report sample counts lost through every filtering step.

## 3. Audit source-model exposure and leakage

This is a mandatory prerequisite for the primary scientific comparison.

For each candidate evaluation galaxy, determine whether it appears in any model's:

- Source training cohort.
- Source validation/checkpoint-selection cohort.
- Source held-out test cohort, if present.

Check both catalogue identifiers and positional matches, including objects appearing in one catalogue under a different identifier.

Construct separate membership flags for every model.

Create the following disjoint evaluation categories:

**Category P — Strict independent paired holdout**

Objects with usable HSC and Legacy observations that were never used in training or checkpoint selection for any model included in the corresponding comparison.

This is the preferred primary benchmark.

**Category V — Validation-exposed but training-clean**

Objects absent from training, but used in validation or checkpoint selection for at least one evaluated model.

These may be used for explicitly labelled exploratory diagnostics, not claimed as independent test evidence.

**Category T — Training-exposed**

Objects appearing in a relevant model's training data.

Exclude these from claims of unseen-object transfer.

If sufficient Category P objects are unavailable, do not invent a clean benchmark.

Report exactly how many are available, identify the limiting model/cohort, and run whatever honest exploratory comparisons are feasible.

Do not use target redshift labels or prediction errors to select the paired cohort.

Do not silently enlarge the cohort by including contaminated objects.

## 4. Construct the matched 2 × 2 evaluation

For each available source model, run frozen inference with both observational representations.

The ideal primary matrix is:

| Trained model | HSC input | Legacy input |
|---|---|---|
| HSC-trained Qwen | In-domain | HSC → Legacy |
| Legacy-trained Qwen | Legacy → HSC | In-domain |

**Every cell must use the exact same paired physical galaxies and spectroscopic reference labels.**

This comparison is valid only when the same objects satisfy the relevant training-exposure constraints.

For the HSC-trained models, evaluate the existing original, Level-0 A, Level-1 B and Level-2 C checkpoints as distinct conditions.

For the Legacy-trained models, evaluate the actual available QLoRA/DoRA configurations. Do not train new Level-1/2 Legacy models merely to fill an empty table.

Use each model's proper tokenizer, serializer, trained PDF head and redshift grid.

Apply the correct physical filter identities to target measurements.

Do not substitute HSC i/y measurements into Legacy W1/W2 positional fields, or vice versa.

Keep genuine native flux/error values and missingness conventions; do not invent absent measurements.

Where input serializers are incompatible, report this explicitly rather than claiming that identical feature positions imply identical physical meaning.

## 5. Primary transfer measurements

For each model and each galaxy, calculate:

\[
\delta_i^{(D)}
=
\frac{\hat z_i^{(D)}-z_{i,\mathrm{spec}}}
{1+z_{i,\mathrm{spec}}},
\]

where \(D\) identifies the HSC or Legacy observational representation.

Report:

- CE;
- NMAD;
- Catastrophic-outlier fraction, using the existing \(|\delta|>0.15\) definition;
- Mean and median bias;
- CRPS;
- Nominal 68% coverage;
- Predicted-redshift distributions;
- PIT diagnostics where supported by the existing evaluator.

Calculate the same-model paired transfer penalty:

\[
\Delta {\rm CE}_{H\rightarrow L}
=
{\rm CE}(M_H, X_L)
-
{\rm CE}(M_H, X_H),
\]

and analogously for Legacy → HSC.

Also calculate paired changes in absolute normalised residuals:

\[
\Delta |\delta_i|
=
|\delta_i^{\rm target}|
-
|\delta_i^{\rm source}|.
\]

Report how many objects change from non-outlier to outlier and from outlier to non-outlier.

These object-level differences are particularly important because the test galaxies and their spec-z labels are identical.

Use paired bootstrap resampling at the galaxy level for descriptive confidence intervals. Clearly state that they are conditional on the existing trained checkpoints and do not include training-seed variation.

Do not compare cross-model CE values unless the prediction grids and probability-mass conventions have been verified compatible.

## 6. Redshift-stratified transfer

Use the exact same paired evaluation galaxies.

Before inspecting new prediction results, specify redshift bins using the available spectroscopic-redshift distribution.

Avoid bins with extremely small samples. Record bin populations and any predetermined bin merges.

For each bin, evaluate:

- In-domain NMAD and outlier fraction.
- Cross-domain NMAD and outlier fraction.
- Mean and median bias.
- CE and CRPS.
- Paired transfer penalties.

Report both the native evaluation redshift distribution and a redshift-standardised aggregate, using one predefined weighting scheme.

Do not replace the full-cohort result with a favourable redshift-selected subset.

The objective is to answer:

**Does observational-system transfer degrade differently across redshift?**

Do not present this as strict high-z extrapolation unless the model's source training genuinely excluded those redshifts.

## 7. Photometric common-support diagnostic

Evaluate whether the paired galaxies occupy the photometric distributions represented in the original source-training cohorts.

Inspect the physically common optical bands, especially g, r and z, along with:

- Flux distributions.
- Colour distributions.
- Photometric uncertainties.
- Signal-to-noise ratios.
- Missingness patterns.
- Apparent-magnitude/depth distributions.
- Available quality and extinction information.

Use physically verified units. Remember that nominally identically named HSC and Legacy filters need not have identical transmission curves.

Develop a simple, reproducible common-support diagnostic using only photometric measurements and source-cohort statistics.

Do not use target spec-z labels to optimise the support estimator.

Report results for:

1. All eligible paired galaxies.
2. A predefined photometric common-support subset, if meaningful.

Keep the full-sample results as the main reference.

Explain whether failures predominantly occur outside the source photometric support or also among well-supported objects.

Do not claim a causal isolation of bandpass effects: paired HSC and Legacy observations also differ in photometric pipelines, depths, noise properties and calibration.

## 8. Existing Level-0/1/2 representation comparison

For HSC-trained warm-start QLoRA, use the available:

- A: Level 0.
- B: Level 1.
- C: Level 2.

These models must be evaluated on the **same new paired-object cohort**, including their in-domain HSC observations.

Do not compare the new Legacy transfer result with the old HSC source-validation metric as if those populations were matched.

The relevant comparisons are:

- Cross-domain transfer penalty for A, B and C.
- Level 1 versus Level 0.
- Level 2 versus Level 1.
- Dependence of the transfer penalty on redshift and photometric support.

Remember that Level 2 retains slightly worse HSC source-validation performance than Level 1 in the archived warm-start experiment.

Consequently, report absolute and relative transfer performance separately.

An apparent Level-2 improvement in one target metric is not proof that wavelength/bandwidth metadata were interpreted physically.

## 9. Execution rules

- Do not launch new training.
- Do not modify existing checkpoints or PDF heads.
- No target-label fitting, calibration or target-informed checkpoint selection.
- No held-out test evaluation until membership and leakage audits establish an appropriate independent cohort and evaluation policy.
- Preserve archived results.
- Use existing inference code and GPU resources when safely available.
- Do not silently alter source data definitions or spectroscopic quality cuts.
- Keep exact input strings, object IDs, hashes, configuration files and evaluation outputs.

If a scientifically clean paired benchmark is impossible with existing model training splits, complete the exposure audit and exploratory analyses, then supply a minimal, reproducible plan for creating a new mutually held-out evaluation cohort.

Do not independently start expensive retraining merely to resolve that limitation.

## 10. Deliverables

Write a report containing:

1. Available models, source cohorts and actual training/validation exposure.
2. HSC–Legacy crossmatch and paired-galaxy construction audit.
3. Counts for independent, validation-exposed and training-exposed categories.
4. A matched 2 × 2 transfer matrix on every scientifically eligible cohort.
5. HSC Level-0/1/2 comparison on identical paired objects.
6. Paired galaxy-level prediction differences and confidence intervals.
7. Redshift-stratified transfer results.
8. Photometric common-support diagnostics.
9. Source and target prediction plots.
10. A reproducible manifest and commands.

Prioritise the integrity of the paired cohort over the number of reported models or sample size.

**Final scientific question:**

Does a photo-z model trained on one photometric survey retain predictive accuracy when the *same galaxies* are observed through a different photometric system, and how does that transfer penalty depend on the model's post-training method and input representation?

The result must explicitly distinguish observation-system transfer from changes in the evaluated galaxy population.