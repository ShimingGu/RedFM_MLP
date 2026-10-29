# Stage 2A.6 — SED Dimensionality, Profile-Likelihood Degeneracy and Redshift Identifiability

## Scientific objective

Investigate whether excessive SED flexibility and profile-likelihood degeneracy are major causes of catastrophic photometric-redshift failure in the existing physical inference pipeline.

The central question is:

**Can physically constrained SED representations and appropriately marginalised latent variables restore redshift identifiability across heterogeneous filter systems without relying on target-domain redshift labels?**

This is a controlled physical-inference experiment, not a broad template-fitting optimisation exercise.

## 1. Existing evidence

Read the completed Stage 2A, Stage 2A.5, Stage 2B and Stage 2B.5 reports and their manifests before implementation.

The Stage 2A.5 audit established:

- EAZY 0.8.7 with 12 FSPS templates and NNLS is numerically reproducible.
- A refined union-knot response integrator passes independent convergence checks.
- HSC validation P0 catastrophic outliers remain approximately 80%.
- HSC median chi-square at continuous spectroscopic redshift is approximately 311.
- The median global minimum chi-square is effectively zero.
- About 63% of HSC validation objects have incorrect high-redshift global minima above z=2.
- About 8.7% of redshift bins can produce almost exact fits in the typical HSC object.
- HSC has five optical bands but the free SED mixture has twelve nonnegative coefficients.
- HSC formal measurement uncertainties can be extremely small, amplifying small model mismatches.
- On the same galaxies, HSC grizy has 86.7% outliers whereas Legacy grzW1W2 has 7.0%.
- Removing Legacy W1/W2 increases its outliers to 97.7%.
- Numerical integration, nanomaggy conversion, column ordering and NNLS correctness do not explain the discrepancy.
- Increasing template errors or imposing a source redshift prior can change posterior behaviour without repairing incorrect physical likelihood minima.

The existing full 12-component profile likelihood must remain a reference.

Do not assume that decreasing the number of components automatically improves physical accuracy: reduced flexibility can also make the true-redshift spectrum impossible to fit.

## 2. Hard constraints

- No Qwen, QLoRA, DoRA or large-model training.
- No CLAUDS ZPHOT used for fitting, hyperparameter selection, calibration or architecture choices.
- No held-out real source test objects.
- Preserve all historical code, checkpoints, predictions, configurations and manifests.
- Use the independently verified Stage 2A.5 response integrator and signed-flux/masked-error interface.
- Preserve genuine negative fluxes and measurement errors.
- Do not introduce arbitrary source-to-target zero-point corrections.
- Do not infer unavailable foreground extinction values.
- Do not use a large grid search over templates, priors and error floors.
- All configuration choices must be source-defined or fixed from independently justified physical assumptions.

Use source training for diagnostic development, source validation for fixed comparison and a distinct source confirmation partition where possible.

Respect repeated HSC/Legacy galaxy identities when constructing partitions.

## 3. Define comparable SED models

Start from the same verified FSPS template library and forward-response integration.

Implement four primary models.

### Model D12 — Existing unrestricted 12-template NNLS

At candidate redshift z:

`f_model(z) = A * K(z) @ w`

with:

- `w >= 0`
- `sum(w) = 1`
- `A >= 0`

This is an identifiable reparameterisation of the existing nonnegative template amplitudes, not an additional restriction on SED shape.

Profile the amplitude and template coefficients as in the original method.

Recover the archived reference likelihood and predictions before changing the SED constraints.

### Model D1 — Single-template model

Choose one discrete template from the pinned library at each candidate redshift.

Profile only its nonnegative amplitude.

Compare:

- The best-template profile score.
- A properly weighted template-marginalised likelihood using fixed discrete template probabilities.

A single-template fit is deliberately restrictive.

Measure whether it reduces false high-redshift fits by suppressing excessive flexibility, or simply causes poor true-redshift reconstruction.

### Model D2/D3 — Physically constrained low-dimensional mixtures

Construct a small SED representation with two or three effective components.

Possible forms include:

- A physically selected two/three-template mixture.
- A low-dimensional nonnegative basis derived exclusively from source-training spectra or independently defined templates.
- A constrained simplex with explicitly documented coefficient priors.

Keep the mixture amplitude separate from spectral shape.

Do not allow a learned arbitrary per-band correction that bypasses the physical SED.

Do not select the basis by minimising errors on CLAUDS.

The central comparison must isolate reduced SED flexibility rather than differences in unrelated filter or photometric conventions.

### Model DM — Properly marginalised SED coefficients

Test a low-dimensional model in which template coefficients are integrated out rather than profiled.

For example:

`p(f | z) = integral p(f | z, A, w) p(A, w | z) dA dw`

Use a proper, normalisable coefficient prior and explicitly specified amplitude treatment.

A practical pilot may use deterministic quadrature, quasi-Monte Carlo or a validated low-dimensional approximation.

Record convergence and numerical integration error.

If marginalisation over an improper amplitude prior is not mathematically defined, choose a proper prior or explicitly condition/profile that amplitude and label the resulting quantity correctly.

Do not call a profiled chi-square curve a marginal likelihood.

Avoid comparing marginal evidences across models with inconsistent prior normalisation.

Begin with a small number of independently justified coefficient-prior strengths.

The goal is to test whether accounting for latent SED volume suppresses pathological exact-fit plateaus.

## 4. Fixed-spectroscopic-redshift diagnostics

For every model, evaluate selected HSC and Legacy spectroscopic galaxies at their known continuous z_spec.

Calculate:

- Minimum chi-square at z_spec.
- Residual per observed band.
- Profiled nuisance amplitude.
- Inferred or marginalised SED coefficients.
- Effective degrees of freedom, with clear treatment of constraints and non-identifiable parameters.
- Template reconstruction adequacy.
- Fraction of sources for which the true-redshift fit is incompatible with the assumed error model.

Compare these quantities against the global redshift solution.

Do not equate a low chi-square with recovery of the true redshift.

Do not interpret reduced chi-square using Nbands minus nominal parameters when the model is underdetermined or constrained in a way that invalidates that counting.

## 5. Characterise redshift identifiability

For each galaxy, evaluate the complete redshift-dependent score on the established grid.

Measure:

1. Global best-fit redshift.
2. Continuous-z_spec score.
3. Delta chi-square or analogous log-evidence difference.
4. Number of distinct competitive redshift modes.
5. Fraction of grid bins within predefined score differences of the optimum.
6. Width of low-chi-square or high-evidence plateaus.
7. Whether the preferred high-z solution disappears under reduced dimensionality.
8. Whether the correct low-z solution remains representable.
9. Predicted posterior concentration and multimodality.

Distinguish:

- Wrong-redshift exact fitting.
- Weakly informative or flat likelihood.
- Multiple physically plausible solutions.
- True-redshift template mismatch.
- Incorrect posterior weighting.

A smaller SED space is successful only if it improves redshift discrimination without making the correct-redshift physical model systematically worse.

## 6. Band-count versus wavelength-coverage experiment

Use exactly matched galaxy identities wherever possible.

Evaluate:

- HSC grizy.
- Legacy grz.
- Legacy grzW1.
- Legacy grzW1W2.
- Selected HSC grizy subsets controlling number of bands.
- HSC grizy plus W1/W2 where genuine audited same-object Legacy/WISE photometry is available.
- Full HSC+Legacy combinations only where photometric compatibility is sufficiently understood.

Do not invent missing WISE measurements.

Keep object identity, spectroscopic redshift, source selection and model configuration fixed across comparisons.

The experiment should distinguish:

**Hypothesis A:** Additional bands improve identifiability because they provide more independent constraints.

**Hypothesis B:** Longer-wavelength leverage improves identifiability because it breaks optical SED/redshift degeneracies.

**Hypothesis C:** Differences in measurement errors, templates or photometric pipelines explain part of the apparent improvement.

Band count and wavelength coverage may not be perfectly separable with the existing catalogues.

State which comparisons identify which effects and which remain confounded.

In particular, the contrast between HSC five-band optical and Legacy five-band optical+WISE is useful but is not a perfectly controlled filter-only intervention.

## 7. Source-only model adequacy and prior controls

Use a limited set of fixed likelihood constructions:

- Original measured-error likelihood.
- The previously source-selected fractional template-error model.
- A separately defined physically motivated error model if Stage 2A.5 justifies it.

Do not tune template-error magnitude to minimise HSC outliers without preserving the likelihood diagnostic.

Keep a primary likelihood-only result without a redshift population prior.

An optional source-defined redshift prior may be reported separately.

A prior suppressing an incorrect high-redshift posterior mode is not the same as repairing the likelihood.

For each model, report both redshift point statistics and probabilistic quality.

## 8. Physical simulation controls

Build small forward-model recovery tests using the verified SED basis.

Include:

- A single known SED.
- Two-component mixtures.
- Three-component mixtures.
- Random mixtures of all twelve components.
- Known simulated redshift.
- Known or profiled amplitude.
- Real HSC and Legacy filter responses.
- Noise-free observations.
- Noisy observations with explicitly controlled uncertainty.

Include both exact model specification and deliberate SED mismatch.

Separate intrinsic non-identifiability from implementation defects.

A model that recovers simulated redshifts when assumptions match but fails on real HSC provides evidence for model/measurement mismatch rather than an incorrect forward integrator.

Do not use simulated performance as a substitute for independent real-survey validation.

## 9. Statistical comparison

For every model and observational system report:

- NMAD.
- Catastrophic-outlier fraction.
- Bias and median bias.
- CE and CRPS where probability conventions permit.
- Nominal 68% coverage.
- PIT or probability calibration diagnostics.
- Fraction of wrong high-z solutions.
- Number and widths of competitive redshift minima.
- True-redshift fit residuals.
- Failed and unsupported fit counts.
- Runtime and computational cost.

Use paired-object bootstrap with galaxy identity as the resampling unit.

Preserve the same galaxy groups when they have multiple survey observations.

Do not infer a universal model advantage from one small sample or one prior specification.

## 10. Execution sequence and gates

### Gate A — Numerical correctness

All new likelihood/profile/marginal calculations must reproduce analytic or independently integrated small test cases.

The original D12 replay must pass.

### Gate B — Identifiability

Verify which SED restrictions remove incorrect exact-fit plateaus and which merely exclude true spectra.

### Gate C — Spectroscopic source confirmation

Compare pre-specified candidate models using an independent source confirmation subset, with no target-label input.

If no constrained model improves both meaningful redshift discrimination and acceptable true-redshift fit adequacy, explicitly report that negative result.

### Gate D — Decoder handoff

Identify which physical model is suitable for Stage 2B.6.

Do not automatically select the model with the narrowest posterior or lowest source NMAD.

Prefer a physically justified likelihood with controlled misspecification behaviour.

If no new model is demonstrably reliable, provide the original and best-characterised constrained models as separate diagnostic inputs, without declaring either universally trustworthy.

## 11. Resource discipline

This is primarily CPU/low-GPU physical fitting.

Start on predetermined source-training and source-validation subsets.

Use targeted, interpretable interventions rather than combinatorial sweeps.

Increase sample counts only for fixed confirmation comparisons.

No Qwen training, population-scale mixed-survey training or real CLAUDS optimisation.

Do not interrupt active GPU jobs.

## 12. Deliverables

Create:

`filter-aware-runs/1028/stage2a6-sed-identifiability/`

or an equivalent new non-overwriting directory.

Save:

- Numerical reproduction and correctness tests.
- D12/D1/D2/D3/DM implementations.
- SED basis and coefficient-prior definitions.
- Source-only selection records.
- Full redshift profile/evidence arrays.
- z_spec-constrained fit diagnostics.
- Per-band residuals.
- Band-count/wavelength-coverage comparisons.
- Simulation and identifiability controls.
- Paired uncertainty estimates.
- Plots and model comparison tables.
- Reproduction commands, hashes and manifest.
- A scientific report.

Create an explicit `stage2a6_handoff.json` specifying:

- Verified response integrator.
- SED basis.
- Nuisance-amplitude treatment.
- Selected or retained physical-score definitions.
- Uncertainty model.
- Prior provenance.
- Numerical accuracy.
- Identifiability limitations.
- Allowed source-only trust/calibration features.
- Interfaces and paths for Stage 2B.6.

End by answering:

1. Are HSC high-z failures substantially caused by excessive SED coefficient freedom?
2. Does a lower-dimensional SED model restore correct-redshift identifiability?
3. Does marginalising coefficients outperform profiling them?
4. How much do WISE wavelength coverage and the number of measured bands affect identifiability?
5. Can the preferred physical likelihood be trusted in a downstream physics-aware neural model?
6. What limitations remain under realistic SED/photometry mismatch?

**Execute the feasible experiments and produce empirical conclusions, not merely an implementation proposal. Do not use CLAUDS labels for any decision.**