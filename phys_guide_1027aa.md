# Stage 0.5 — Causal Diagnostics of Qwen Input Fragility and Photometric Distribution Extrapolation

## Scientific objective

We are investigating whether the existing Qwen3.5-4B-Base photometric-redshift model has learned physically meaningful and transferable representations, or is functioning primarily as a source-specific conditional estimator.

Stage 0 revealed two major findings:

**1. Extreme HSC-source / CLAUDS-target photometric distribution shift**

- HSC-source median i-band magnitude: 18.45.
- CLAUDS median i-band magnitude: 25.51.
- HSC-source median i-band S/N: 1351.8.
- CLAUDS median i-band S/N: 13.8.
- Only 2/256 CLAUDS diagnostic galaxies pass the pre-specified joint photometric-support criterion.
- Independent source/target domain classifiers have AUC approximately 0.997–0.999.

**2. Severe input-interface sensitivity**

On 256 HSC-source validation galaxies, using the original frozen model and unchanged photometry:

- Native input: NMAD 0.0303; catastrophic outliers 5.47%.
- Replace EBV with NA: NMAD 0.1854; outliers 43.75%.
- Omit EBV: NMAD 0.1901; outliers 52.34%.
- Replace survey schema: NMAD 0.1593; outliers 43.36%.

These interventions conflate potentially meaningful EBV information, input syntax, token position and last-token hidden-state behaviour.

The goal of Stage 0.5 is to disentangle these mechanisms and determine how the model responds to controlled brightness and noise changes.

This is a diagnostic study, not another photo-z training experiment.

## Hard constraints

- No SFT, QLoRA, DoRA or other weight updates.
- Do not initialise new prediction heads.
- Do not modify archived model weights.
- Do not use CLAUDS target labels for training, calibration, configuration selection or tuning.
- Do not inspect held-out test objects.
- Preserve original source-validation splits.
- Preserve the original redshift PDF grid.
- Do not overwrite Stage 0 or historical results.
- Preserve every physical unit and measurement convention.
- Record exact actual tokenizer inputs and model outputs.

Use the original completed HSC QLoRA checkpoint and matching trained PDF head as the primary model.

Additional selected Level-1/2 checkpoints may be used for limited follow-up diagnostics, but do not expand into a large model matrix.

## Part A — EBV and last-token sensitivity

### A1. Inspect the current input architecture

Determine the exact original HSC serialization, including:

- Field order.
- Photometry fields.
- Uncertainty fields.
- MW_EBV position.
- Actual final non-padding token.
- Tokeniser output IDs.
- Last-token pooling implementation.
- Prediction-head input dimensionality.

Inspect whether the final hidden state is pooled directly from the final numerical EBV token or from a stable query/sentinel token.

Do not assume the implementation behaves as intended; verify actual token IDs and attention masks.

### A2. Frozen-weight numerical EBV counterfactuals

Evaluate fixed HSC source-validation galaxies with identical photometry but different EBV values.

Pre-specify the following:

1. Original true EBV.
2. EBV replaced by zero, preserving the numeric-field schema.
3. EBV replaced by the source-training median.
4. EBV values shuffled among source-validation galaxies using a fixed seed.
5. Small controlled perturbations of EBV around its true value.
6. EBV replaced with the existing NA sentinel.
7. EBV field omitted entirely.

Use physically meaningful perturbation ranges based on the source-training EBV distribution. Do not extrapolate to arbitrary unrealistic values merely to increase the effect size.

Retain all other fields unchanged.

Where feasible, include formatting-matched numerical controls to minimise token-length differences.

These controls still cannot perfectly isolate semantics because changing a number also changes its tokenisation. Measure and report token-length and final-token-ID changes explicitly.

### A3. Position and terminal-token counterfactuals

Construct a small set of controlled serialization variations:

- Original format with EBV in its original position.
- EBV moved before photometry.
- EBV placed after photometry but followed by a fixed terminal query marker.
- EBV retained but replaced with a fixed neutral number.
- The same fixed terminal query marker with true, neutral and missing EBV.

The terminal marker must be an ordinary valid textual token sequence supported by the existing tokenizer.

Do not add a new trainable embedding.

Keep the same frozen model and PDF head.

Critically, the original model was not trained on these alternative formats.

Therefore, any observed improvement or degradation is a **frozen-model interface sensitivity diagnostic**, not a fair comparison of separately adapted architectures.

Do not claim that a new terminal token is intrinsically better unless a later matched-training experiment supports it.

### A4. Representation diagnostics

For a representative paired subset and selected full-validation configurations, collect:

- Actual serialized text.
- Token IDs.
- Attention masks.
- Final non-padding token IDs.
- Pooled hidden states.
- PDF-head logits.
- Predicted PDFs.
- Redshift point estimates.

Measure paired representation changes using:

- Cosine similarity of pooled representations.
- Normalised embedding displacement.
- Logit differences.
- KL or Jensen–Shannon divergence between prediction PDFs.
- Redshift prediction shifts.

Where feasible, inspect whether position changes alter predictions even when the underlying physical photometry and EBV value remain unchanged.

Use identical objects and fixed seeds.

### A5. Evaluation

Use a small initial screening subset of 256–512 HSC source-validation galaxies.

Then evaluate a pre-specified subset of the most informative controls on the complete existing 5,683-object HSC source-validation cohort.

The initial screening may be used for computational feasibility and bug checks, but not to optimise configurations for improved accuracy.

Report:

- CE.
- NMAD.
- Catastrophic-outlier fraction.
- Bias.
- CRPS.
- 68% coverage.
- Median predicted redshift.
- Distribution concentration.
- Per-object prediction changes.

Do not select a new model checkpoint.

## Part B — Controlled brightness and noise extrapolation

### B1. Hypotheses

We want to distinguish:

**H1 — Absolute-brightness dependence**

The model is highly sensitive to overall flux normalisation even when colours and relative measurement precision remain unchanged.

**H2 — S/N dependence**

The model becomes unreliable when measurement uncertainty increases, independently of absolute flux changes.

**H3 — Combined numerical distribution shift**

The interaction between faint fluxes and increased noise drives the model into a prediction-collapse regime.

These hypotheses are not mutually exclusive.

### B2. Source-validation intervention dataset

Use the same original HSC source-validation galaxies and spectroscopic redshifts.

Preserve intrinsic object identity and the original redshift references throughout the interventions.

All transformations are counterfactual changes to observed photometric inputs.

Do not pretend that changing apparent flux normalisation physically moves a galaxy to a new redshift.

Preserve correct flux and uncertainty units.

### B3. Experiment B1 — Brightness scaling at fixed S/N

Multiply all valid measured HSC fluxes and their uncertainties by the same positive factor:

`flux_new = a * flux_original`

`sigma_new = a * sigma_original`

This approximately preserves per-band S/N and all flux ratios for valid bands.

Use a pre-specified magnitude-shift grid:

`delta_mag = [0, 1, 2, 4, 6, 7]`

with:

`a = 10 ** (-0.4 * delta_mag)`

This spans the approximate source-to-CLAUDS brightness contrast.

Missing values remain missing.

Do not modify redshift labels or EBV.

Evaluate whether the model's predicted redshift distribution responds strongly to an overall flux-scale change despite approximately preserved colour information.

Note that luminosity and apparent magnitude can carry legitimate population-prior information. Brightness dependence alone is not proof of nonphysical inference.

### B4. Experiment B2 — Uncertainty degradation at fixed measured flux

Keep the measured flux vector unchanged.

Increase its reported uncertainty using predetermined factors, such as:

`sigma_new = b * sigma_original`

with:

`b = [1, 3, 10, 30, 100]`

This is an uncertainty-reporting intervention, not a statistically self-consistent noisy reobservation.

It isolates model sensitivity to reported S/N and uncertainty tokens.

Report it separately from actual noise injection.

### B5. Experiment B3 — Simulated noisy faint observations

Construct controlled combinations of flux dimming and increased uncertainty.

For each object, begin from the measured source flux and error.

Let:

`f_scaled = a * f_original`

`sigma_scaled = a * sigma_original`

Choose new target uncertainties satisfying:

`sigma_target >= sigma_scaled`

Then generate a new noisy measurement by adding Gaussian noise with variance:

`sigma_target**2 - sigma_scaled**2`

to the scaled measurement.

This preserves the approximate existing measurement-noise variance rather than counting it twice.

Use only positive additional variance.

Use fixed random seeds and multiple noise realisations.

Document the approximation: the original measured flux is already a noisy observation, and the physical decomposition of shot noise, sky noise and calibration error is not generally available.

Do not describe this construction as a fully realistic survey image-noise simulation.

Use a limited, pre-specified grid sufficient to probe the observed source-to-CLAUDS contrast, rather than a large hyperparameter search.

Retain missingness, filter identities and EBV consistently.

### B6. Factorial evaluation

Separate the effects of:

- Brightness scaling.
- Reported uncertainty scaling.
- Actual injected noise.
- Combined faintness and noise degradation.

Evaluate the original frozen HSC QLoRA model on each intervention.

Report metrics versus:

- Magnitude shift.
- Median S/N.
- Effective flux scale.
- Source-to-target photometric distance, where compatible with Stage 0 definitions.

Track:

- CE, NMAD, outliers, bias, CRPS and coverage.
- Distribution of predicted median redshifts.
- Prediction collapse statistics.
- Per-object prediction shifts.
- Whether predictions systematically approach low redshift under extreme dimming.
- Dependence on the original source redshift, brightness and S/N.

Distinguish a learned brightness-dependent redshift prior from pathological loss of photometric information.

If computationally feasible, compare with a simple fixed physical/template-fitting baseline or archived MLP baseline on the same modified measurements. Do not train a new large baseline for this diagnostic.

## Part C — Interpretation

Answer five specific questions.

1. Is the catastrophic EBV-missingness effect predominantly associated with numerical EBV information, altered input syntax, final-token identity or some combination?
2. Can the model retain physically identical predictions when input field order changes?
3. Does reducing brightness at constant S/N cause substantial prediction drift?
4. Does decreasing S/N independently cause prediction collapse?
5. Can the observed direction of CLAUDS failure be approximately reproduced using controlled interventions on source-validation photometry?

Do not claim unique causal identification when interventions change multiple token-level properties.

Do not infer that the model lacks astrophysical knowledge merely because it is sensitive to formatting or extrapolation.

Do not claim that a galaxy moved to a lower flux scale is a realistic substitute for a truly different high-redshift galaxy population.

## Part D — Computational constraints

This stage should require inference only.

Begin with the minimum set of configurations needed to establish correctness and inspect effects.

Expand to the full source-validation cohort only for pre-specified interpretable comparisons.

Use existing trained checkpoints.

No expensive model training or adapter sweeps.

Do not interrupt running GPU jobs.

If memory or compute limits prevent the full grid, prioritise:

1. EBV numeric-versus-syntax controls.
2. Fixed-S/N brightness scaling.
3. Fixed-flux uncertainty changes.
4. A small set of combined dimming/noise conditions.

Clearly mark unexecuted experiments.

## Part E — Deliverables

Create a new directory:

`filter-aware-runs/1023/stage05-interface-robustness/`

or another suitable non-overwriting project path.

Produce:

1. Exact checkpoint and data manifests.
2. Verified tokenizer/last-token audit.
3. EBV numerical, positional and missingness controls.
4. Brightness and S/N response curves.
5. Paired prediction and representation diagnostics.
6. PDF-collapse comparisons.
7. Reproducible random seeds and perturbation definitions.
8. All evaluation tables.
9. A concise scientific report.

End with a verdict distinguishing:

- Genuine sensitivity to astrophysical EBV information.
- Fragility associated with token position or serialization.
- Absolute-brightness dependence.
- Noise/SNR dependence.
- Remaining unexplained CLAUDS domain shift.

**Complete this Stage 0.5 diagnostic without training. Preserve all historical results and do not evaluate held-out test objects.**