# Stage 2B — Mask-Aware Photometric Set Encoder with a Differentiable SED Forward Model

## Scientific objective

Develop a numerical, physics-aware neural architecture for photometric-redshift inference across heterogeneous survey photometry.

The central question is:

**Can an explicit filter-aware observational representation and a differentiable SED forward model improve cross-survey photo-z generalisation beyond ordinary supervised numerical regression?**

The ultimate transfer objective remains:

**GalaxiesML-Spec + DESI–Legacy → CLAUDS**

The complementary CLAUDS-source → spectroscopic-target direction may be prepared later but is not part of the primary training experiment.

This stage deliberately excludes pretrained Qwen and other large language models.

We must first establish whether an appropriate numerical inductive bias can provide the desired scientific generalisation.

## 1. Prerequisites and scientific context

Read the completed Stage 0, Stage 0.5, Stage 1 and Stage 2A reports, if available.

Relevant existing results:

- HSC-trained Qwen succeeds in-domain but fails catastrophically on CLAUDS.
- Source/target brightness and noise distributions differ severely.
- The previous Qwen representation is sensitive to token order and terminal fields.
- Frozen pretrained Qwen showed no robust advantage over numerical controls in the small synthetic cross-filter benchmark.
- Explicit SED-template physics performed better in synthetic point-redshift inference, but its PDF calibration was inadequate.
- The target CLAUDS catalogue has different filters, observational selection and redshift-reference conventions.

Do not assume that Stage 2A succeeds.

Inspect its results and reuse verified filter-response integration, catalogue processing, redshift-PDF conventions and template provenance.

If Stage 2A identified unresolved physical-input errors, resolve them or delimit their effect before training.

## 2. Dataset and label protocol

Use the verified spectroscopic source datasets:

- GalaxiesML-Spec: HSC grizy, DESI spectroscopic redshifts.
- DESI–Legacy: Legacy grzW1W2, DESI spectroscopic redshifts.

Use native observed fluxes and uncertainties in consistent physical units.

Retain actual filter identities and response curves.

Inspect spectroscopic quality indicators, source selections and redshift ranges.

Prevent object-level leakage:

- Audit shared DESI identifiers.
- Audit positional matches.
- Keep all photometric observations of the same underlying galaxy in the same split.
- Prevent source-validation objects from entering training through the other catalogue.
- Apply the existing CLAUDS overlap exclusions and extend them to the combined source cohort.

Do not label a single-survey training run as mixed-survey learning if the Legacy cache is unavailable.

If complete source data cannot be recovered, execute infrastructure tests and a small source pilot, but explicitly leave the definitive mixed-source experiment pending.

### Redshift grid

Start with 300 bins over z = 0–4.5 for compatibility with existing experiments.

Define an explicit policy for source spectroscopic objects outside this grid.

Do not silently clip high-redshift labels into the final bin.

Inspect the source redshift distribution, particularly whether source galaxies adequately cover redshifts characteristic of faint observational targets.

Do not assume brightness augmentation provides missing high-redshift supervision.

## 3. Physically consistent inputs

Every galaxy is represented as an unordered set of band observations.

For band b, provide:

`(flux_b, sigma_b, availability_b, quality_b, filter_response_b)`

Where available, include:

- Corrected or documented extinction information.
- Instrument/filter identity.
- Reliable measurement-quality information.

The model must not depend on a fixed positional meaning of band index.

### Flux normalisation

Separate observed relative spectral shape from absolute luminosity scale.

Use a robust, source-defined per-object flux scale to normalise the band measurements.

The transformation must:

- Support valid negative fluxes.
- Handle missing observations.
- Avoid division by zero.
- Apply consistently to flux and uncertainty.
- Preserve the actual relative errors/S/N.
- Record normalisation as an optional separate brightness feature.

The primary physics-aware representation should treat overall luminosity or flux amplitude as a nuisance variable.

Include a separate, explicitly labelled experiment in which brightness is supplied as a possible population-prior feature.

Do not silently allow the model to rely on absolute brightness as an uncontrolled source-specific shortcut.

### Missing extinction data

Separate:

`EBV_value`

from:

`EBV_available`

Do not use a numerical placeholder as if it represented a measured EBV value.

Ensure missing EBV cannot alter the final aggregation mechanism through an untrained sentinel-token convention.

## 4. Model architecture N0 — Numerical regression baseline

Construct a compact conventional numerical MLP for redshift-PDF prediction.

Inputs may use a fixed, padded numerical representation with:

- Fluxes.
- Uncertainties.
- Masks.
- Filter identities or numerical filter summaries.
- Optional explicit brightness feature.

Do not assume missing data equal zero measurements.

Use a matched train/validation cohort and redshift-PDF head.

This baseline is intended to answer:

**How much transfer performance can ordinary supervised numerical learning achieve?**

Retain existing suitable numerical baselines where their input and evaluation contracts match.

## 5. Model architecture N1 — Shared band encoder

Construct a shared numerical encoder applied independently to each observed band.

Schematic form:

`h_b = phi(flux_b, sigma_b, mask_b, response_b)`

Every valid band passes through the same encoder weights.

The filter response must be represented numerically rather than only as a textual band name.

Possible choices:

- Normalised response curve sampled on a fixed wavelength grid.
- Compact response-curve basis coefficients.
- A small shared response-curve encoder.

Choose the simplest representation that preserves meaningful wavelength location and passband structure.

Use a documented wavelength grid covering the actual ultraviolet, optical and infrared filters.

Do not truncate W1/W2 or CLAUDS NIR response curves to an optical-only wavelength interval.

Use permutation-invariant or permutation-equivariant aggregation, such as masked mean/sum pooling or a compact Set Transformer.

The final galaxy representation is:

`h_gal = Aggregate({h_b for valid bands})`

A fixed learned query for attention pooling is acceptable if trained consistently.

Band order permutations must not alter predictions beyond numerical tolerance.

### Output

Predict a normalised redshift PDF on the common grid.

Use source-supervised cross-entropy.

Retain the same evaluation metrics as Stage 2A.

### Crucial control

Test the same architecture with:

- Correct filter responses.
- A compact response-summary representation.
- Incorrect or shuffled filter-response information.

Wrong-response experiments are diagnostics, not a new source of target-based hyperparameter tuning.

## 6. Model architecture N2 — Shared encoder plus differentiable SED decoder

Use the same numerical encoder and aggregation architecture as N1.

Introduce a physically constrained decoder that connects redshift, intrinsic galaxy SED and actual observed filter responses.

### Rest-frame SED basis

Represent an intrinsic galaxy SED as a nonnegative combination of a small number of documented rest-frame basis spectra:

`SED_rest(lambda) = sum_k w_k T_k(lambda)`

with:

`w_k >= 0`

The basis may start from verified templates used in Stage 2A.

Keep the initial basis compact.

Record template coverage and representational limitations.

Do not let the model construct arbitrary unconstrained per-filter flux vectors that bypass the physical forward calculation.

### Redshifted filter integration

At candidate redshift z, project the SED through each actual response curve:

`f_model_b(z, w, A) = A * IntegralResponse(SED_rest, z, R_b)`

where A is a nonnegative nuisance amplitude.

Use physically correct response weighting and AB flux-density conventions.

Prefer precomputing a verified basis-response kernel:

`K[filter, redshift_bin, sed_component]`

when it improves efficiency.

The kernel must preserve the physical filter identities and support new filters without retraining a separate filter-position-specific output head.

The decoder must be differentiable with respect to relevant learnable SED coefficients and other trainable parameters.

Redshift may initially be handled as a discrete grid.

### Amplitude profiling

For a fixed candidate redshift and model spectral shape, profile the nonnegative overall amplitude using valid observed bands.

For example, with effective model band fluxes t_b:

`A_star = max(0, sum_b(f_b*t_b/sigma_b**2) / sum_b(t_b**2/sigma_b**2))`

using an appropriate positive denominator floor and valid-data mask.

If additional numerical treatment is necessary, document it.

Do not calculate weighted likelihood terms using fabricated uncertainties.

Do not treat missing data as independent high-precision zero-flux measurements.

### Physical likelihood

Use the actual measured fluxes and available uncertainties to compute a redshift-dependent physical reconstruction score.

A schematic expression is:

`chi2(z) = sum_valid_b ((f_obs_b - A_star(z)*f_template_b(z)) / sigma_b)**2`

Convert it to an appropriately normalised likelihood or score.

Document omitted normalisation terms and the distinction between a profile likelihood and a fully marginalised posterior.

### Model prediction

The architecture may combine:

- A neural redshift score from the shared set encoder.
- A redshift-dependent physical score from the SED decoder.

Use a minimal, explicit combination rule.

For example:

`p(z) = softmax(logits_neural(z) + beta * score_physical(z))`

where beta is fixed or calibrated using source validation only.

Prefer a constrained combination over a large flexible correction network.

Do not allow the neural network to bypass the physical score silently.

### Training losses

Compare at least:

**N1 — Redshift-only training**

`L = CE(p_neural, z_spec)`

**N2 — Physics-aware training**

`L = CE(p_hybrid, z_spec) + alpha * L_photometry + gamma * L_SED_regularisation`

Use a physically motivated photometric reconstruction loss, accounting for measured errors and the nuisance amplitude.

If supervising reconstruction at known source spectroscopic redshift, ensure the decoder does not receive true redshift at inference.

At inference, score candidate redshifts using observed photometry only.

For ambiguous or multimodal redshift distributions, do not equate reconstructing an average flux vector with correctly evaluating the photometric likelihood.

Avoid a reconstruction loss that can be minimised by choosing an arbitrary nuisance amplitude or unphysical SED coefficients.

Select alpha, gamma and any score-combination coefficient using source validation only, from a small pre-specified set.

## 7. Physics and architecture invariance tests

Before expensive training, verify:

1. Band-permutation invariance.
2. Correct masked aggregation.
3. No contribution from unavailable measurements.
4. Correct handling of negative fluxes.
5. Correct treatment of absent uncertainties.
6. Consistent S/N under common flux/uncertainty rescaling.
7. Correct AB normalisation.
8. Agreement between cached kernel integration and direct numerical response integration.
9. Finite and correctly directed gradients through the decoder.
10. Normalised redshift PDF output.
11. No redshift-label leakage into decoder inputs.
12. No hidden dependence on fixed filter positions.

Generate physically paired counterfactual observations using the Stage-1 SED benchmark.

Check that identical underlying SED/redshift instances observed through different filter systems lead to appropriately consistent predictions, allowing for genuine degeneracy and noise.

Do not interpret incorrect-metadata sensitivity alone as physical reasoning.

## 8. Training design and budget

Start with one small deterministic pilot.

Do not immediately train all models on all available data.

Recommended sequence:

1. N0 numerical MLP.
2. N1 shared band encoder.
3. N2 shared encoder plus physical decoder.

Use the same source cohorts, redshift grid, train/validation groups and training-label budget.

Match trainable parameter counts and optimisation budgets where practical.

For N1 versus N2, keep the shared encoder and neural redshift head architecture identical; additional physical-decoder components must be identified and counted separately.

Do not interpret equal parameter counts as guaranteeing equal effective capacity or optimisation quality.

Use source validation to assess convergence.

If one model is severely underfit, report that limitation before interpreting an architectural ranking.

After the initial pilot, repeat promising configurations with multiple seeds.

Do not expand to a large sweep without an interpretable pilot result.

### Training-domain diversity

Train on both spec-z source datasets when possible.

Use sensible survey balancing so that one catalogue does not dominate optimisation through sample count.

If the same spectroscopic galaxy has HSC and Legacy measurements, use their multi-view observations consistently and keep all views in the same split.

Distinguish:

- Same-galaxy multi-view learning.
- Multiple-source-survey learning.
- Transfer to an entirely unseen target population.

These are different experimental settings.

### Controlled photometric augmentation

Because source galaxies are much brighter than CLAUDS, include a pre-specified physically cautious augmentation option.

For example:

- Random common flux-amplitude scaling.
- Consistent uncertainty scaling.
- A limited noise-degradation model.

Use source labels only.

Do not infer target redshift labels or optimise augmentation to CLAUDS ZPHOT.

Do not pretend that dimming a bright low-redshift galaxy creates a physically realistic high-redshift galaxy.

Keep non-augmented and augmented conditions separate.

### Source redshift support

Inspect source spectroscopic coverage before deciding the final training protocol.

If high-redshift labelled objects are insufficient, record that supervised generalisation to that range is unsupported.

Do not conceal this limitation through an apparently successful synthetic augmentation.

## 9. Synthetic evaluation

Reuse Stage 1's independently split synthetic SED benchmark where possible.

Compare N0, N1 and N2 on:

- Source filters.
- CLAUDS-like filters.
- Strictly held-out filter responses.
- Held-out SED families.
- Controlled brightness and S/N changes.

Keep synthetic source-training and evaluation SED identities disjoint.

Report:

- NMAD and outlier fraction.
- Bias.
- CE and CRPS.
- 68% coverage.
- Paired cross-filter prediction changes.
- Known-redshift-change response.
- Physical reconstruction residuals.

Retain the Stage-1 physical/template baseline for context.

Do not compare models with access to different physical information as though they received identical inputs.

## 10. Real-survey source evaluation

Evaluate N0, N1 and N2 on the identical spectroscopic source-validation objects.

Report separate metrics for HSC and Legacy, not just pooled performance.

Use CE for source-only checkpoint selection.

Report:

- NMAD.
- Outliers.
- Bias.
- CRPS.
- Coverage.
- Conditional metrics by redshift, brightness, S/N and completeness.
- Predicted redshift distributions.
- Reconstruction-likelihood diagnostics for N2.

Do not use held-out source test data during development.

A model that cannot learn the source task sufficiently well is not ready for a meaningful cross-domain comparison.

## 11. Real CLAUDS zero-shot evaluation

CLAUDS is the primary target.

Keep all CLAUDS redshift labels excluded from training, checkpoint selection, hyperparameter tuning and calibration.

Reuse the existing 256 CLAUDS development queries for pre-specified exploratory evaluation only.

Before any new independent confirmation evaluation, freeze the models and all settings.

Compare:

- Stage 2A EAZY baseline.
- N0 numerical MLP.
- N1 shared band encoder.
- N2 shared encoder plus physical decoder.
- Archived HSC Qwen diagnostic results.

Report all standard redshift metrics and calibration diagnostics.

Also report:

- Predicted median-redshift distribution.
- Fraction of outputs concentrated in a narrow redshift interval.
- Whether the previous near-constant redshift collapse persists.
- Results conditional on photometric support.
- Results conditional on target brightness and S/N.
- Results with the full available CLAUDS band set versus HSC-compatible bands.
- Failed or insufficient-photometry cases.

CLAUDS reference labels are ZPHOT, not independent spectroscopic truth.

Explicitly distinguish agreement with ZPHOT from true-redshift accuracy.

If N2 improves source performance but does not transfer to CLAUDS, do not claim successful physical generalisation.

## 12. Scientific attribution

Separate three hypotheses:

### Hypothesis A — Better input architecture

The shared band encoder improves performance because it handles heterogeneous, unordered band sets more appropriately.

Primary comparison: N1 versus N0.

### Hypothesis B — Explicit physics improves generalisation

The differentiable SED decoder improves transfer to unseen filters or populations beyond the matched N1 architecture.

Primary comparison: N2 versus N1.

### Hypothesis C — Large-scale pretraining is necessary

This cannot be demonstrated in Stage 2B because Qwen is intentionally absent.

Leave this claim for Stage 2C.

Do not describe N2 success as evidence for pretrained astrophysical knowledge.

A successful physics-aware numerical model establishes a useful inductive bias, not a foundation-model advantage.

## 13. Execution gates

### Gate 1 — Physical correctness

All unit, bandpass, masking, forward-integration and gradient tests must pass.

### Gate 2 — Numerical learnability

N0/N1/N2 must demonstrate reasonable convergence on spectroscopic source-validation objects.

If the pilot is severely underfit, fix demonstrable implementation or optimisation problems using source-only evidence.

### Gate 3 — Physical improvement

Check whether N2 improves on N1 in controlled synthetic held-out-filter tests and source-validation diagnostics.

A lack of synthetic improvement does not automatically prohibit a small real-data evaluation, but it does not justify an expensive large-scale run.

### Gate 4 — CLAUDS transfer

Evaluate the selected fixed configurations on the pre-specified CLAUDS diagnostic cohort.

Only scientifically interpretable improvements justify further scaling.

Do not use CLAUDS results to iteratively tune the model.

## 14. Deliverables

Create a new directory such as:

`filter-aware-runs/1028/stage2b-physics-aware-numerical/`

Include:

1. Full architecture specifications.
2. Dataset/split/overlap audit.
3. Shared band encoder implementation.
4. Physical response-kernel construction.
5. Differentiable SED decoder.
6. Invariance and gradient tests.
7. N0/N1/N2 source-training histories.
8. Synthetic cross-filter results.
9. Real-survey source-validation results.
10. Frozen CLAUDS exploratory evaluations.
11. Model-size, runtime and GPU-memory comparisons.
12. Reproduction commands, hashes, seeds and configurations.
13. Concise scientific interpretation.

The final report must answer:

- Does a mask-aware set representation improve generalisation over conventional numerical regression?
- Does explicit SED reconstruction physics provide additional benefit?
- Can either approach infer meaningful redshifts in real CLAUDS without CLAUDS-label fitting?
- Which failures are associated with missing redshift support, photometric distribution shift, model architecture or physical-model mismatch?
- Is there a scientifically justified reason to proceed to Stage 2C and reintroduce pretrained Qwen?

**Execute implementation, unit tests, a controlled pilot and all justified evaluations. Do not stop after describing an architecture. Do not launch an expensive hyperparameter sweep or Qwen training.**