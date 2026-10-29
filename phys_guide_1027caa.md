# Stage 2A.5 — Root-Cause Audit of the HSC EAZY Failure

## Scientific objective

Determine why the verified EAZY 0.8.7 physical-template inference pipeline performs reasonably on DESI–Legacy but catastrophically fails on GalaxiesML-Spec HSC photometry.

This is a **physical-model, numerical-integration and photometric-consistency diagnosis**, not a template hyperparameter optimisation project.

### Existing results

Stage 2A used:

- EAZY 0.8.7 NNLS.
- 12 pinned FSPS QSF templates.
- Inoue14 IGM attenuation.
- Actual response curves covering HSC, Legacy and CLAUDS.
- A profile-likelihood pseudo-posterior without a population prior.
- Redshift grid of 300 bins spanning 0–4.5.

Key P0 source-validation results:

| Dataset | N | NMAD | Outliers | CE | Median predicted z |
|---|---:|---:|---:|---:|---:|
| HSC | 5,683 | 0.3650 | 80.24% | 929.29 | 3.1575 |
| Legacy | 15,053 | 0.0702 | 14.67% | 35.88 | 0.4575 |

The HSC P2 confirmation cohort also performs poorly: 91.02% outliers, NMAD 0.4183.

Meanwhile, an independently trained Qwen photo-z model achieves NMAD approximately 0.0268 on the same HSC source domain.

Stage 2A also found a 1.40% discrepancy between its numerical response integration and the independent Stage-1 implementation at one comparison redshift. This discrepancy was attributed to different spectral interpolation schemes, but it has not been demonstrated to be negligible relative to the exceptionally small HSC errors.

The goal is to determine whether the HSC failure arises predominantly from:

1. Numerical integration or unit errors.
2. Filter-response or photometric-column mismatches.
3. Template-family incompleteness.
4. Extremely restrictive observational uncertainties.
5. Astrophysical degeneracies and absent population priors.
6. A combination of these mechanisms.

Do not presume which hypothesis is correct.

## 1. Hard constraints

- No Qwen, N0, N1 or N2 training.
- Preserve all archived Stage 2A outputs, template files, filters and manifests.
- Preserve existing source-data splits and cross-catalogue identity mappings.
- Do not access held-out source test objects.
- Do not use CLAUDS ZPHOT labels for model selection or calibration.
- Do not change the physical assumptions of P0 silently.
- Do not introduce arbitrary photometric zero-point corrections fitted to improve results.
- Do not hide incompatible objects through undocumented clipping, probability floors or selection cuts.
- Separate numerical correctness tests from astrophysical-model adequacy tests.
- Prefer source-training objects for diagnostic intervention development and source validation for fixed comparisons.

This stage is allowed to use existing source spectroscopic redshifts as known physical references for residual analysis. Clearly distinguish diagnostics involving those labels from unsupervised inference.

## 2. Recover and reproduce the exact Stage 2A pipeline

Inspect:

- `scripts/stage2_data.py`
- `scripts/stage2_physics.py`
- Stage 2A report and manifests.
- Filter-response registry.
- EAZY configuration and templates.
- Saved template-response kernels.
- Existing per-object PDFs and chi-square grids.
- Source HSC and Legacy photometry loaders.

Before proceeding, verify that a deterministic subset reproduces the archived P0 predictions, fitted chi-square grids and redshift PDFs.

Record the exact units, filter identifiers, observed flux/error arrays, effective uncertainty model, template normalisations and redshift-grid definitions.

Do not begin scientific interpretation if replay fails.

## 3. Numerical forward-model audit

This is the first priority.

### 3.1 Verify response integration independently

Compare the existing cached kernel with a separately evaluated high-resolution or adaptive numerical convolution.

Test:

- All 21 registered filter responses.
- Multiple redshifts from approximately 0 to 4.5.
- Every FSPS template.
- Smooth spectra and spectra with strong breaks/emission features.
- Constant-f_nu spectra.
- Wavelength-grid boundaries and interpolation behaviour.
- Photon-counting response weighting.
- AB flux-density normalisation.

The existing single-redshift 1.40% discrepancy is not sufficient validation for high-S/N HSC inference.

For each test, report:

`relative_flux_difference`

and, on representative actual galaxies:

`absolute_model_flux_difference / measured_sigma`

and the induced change in chi-square.

Do not impose a universal tolerance without considering observational precision.

Where numerical discrepancies are significant, correct the integration consistently, rerun the physical unit tests, and compare against the original archived results.

Preserve both versions.

### 3.2 Verify the NNLS/profile-likelihood implementation

For fixed redshift and template-response matrix, verify that:

- Nonnegative template coefficients are optimised correctly.
- The objective uses the intended inverse-variance weighting.
- Negative observed fluxes are handled consistently.
- Missing observations contribute no likelihood term.
- Zero/invalid uncertainties are properly excluded or handled according to documented rules.
- Template columns are not inadvertently reordered.
- Normalisation is consistent across redshift.
- Chi-square values match an independent small-matrix NNLS implementation.
- Redshift-PDF bin masses are normalised correctly.

Check whether numerical conditioning of the template matrix causes solver instability.

Test a small fixed collection of representative low-z and high-z cases using double-precision calculations.

### 3.3 Audit HSC source photometry

The HSC loader reconstructs nanomaggy flux and uncertainty from archived cmodel magnitude/error measurements and inherits magnitude-limit censoring.

Independently verify:

- Magnitude-to-flux conversion.
- Propagation of uncertainties.
- Treatment of missing or censored measurements.
- Source catalogue zero-point conventions.
- Whether uncertainties are genuine measurement errors or transformed approximations.
- Photometric quality and selection.
- HSC grizy band-column mapping.
- r/r2 and i/i2 throughput approximations.
- Galactic extinction history.
- Whether cmodel magnitudes correspond to the assumed flux measurement definition.

Compare analogous quantities against Legacy, including the difference between HSC cmodel and Tractor fluxes.

Do not treat those measurement definitions as interchangeable without evidence.

## 4. Redshift-conditioned chi-square diagnosis

Use spectroscopic source data, with the existing HSC/Legacy validation objects for descriptive evaluation.

For each object compute:

`chi2_at_zspec`

`chi2_global_min`

`delta_chi2 = chi2_at_zspec - chi2_global_min`

and the best-fit redshifts.

Use interpolation or exact evaluation at the continuous z_spec rather than attributing a large grid-rounding error to the physical model.

At the spectroscopic redshift, refit the nonnegative template coefficients and calculate each band's residual:

`r_b = (f_obs_b - f_model_b(z_spec)) / sigma_effective_b`

Save:

- Per-band signed and absolute residuals.
- Chi-square contributions.
- Best-fit template coefficients.
- Active NNLS template sets.
- Reduced chi-square where meaningful.
- Alternative redshift minima.
- Dominant redshift modes.
- Data quality and uncertainty flags.

Also repeat these diagnostics at the unconstrained minimum-chi-square redshift.

### Conditional analyses

Stratify by:

- True spectroscopic redshift.
- Observed i magnitude.
- S/N.
- HSC/Legacy view.
- Spectral colour.
- Photometric completeness.
- Filter configuration.
- Relative uncertainty.
- Quality flags.

Look for systematic tendencies such as low-z HSC galaxies receiving dramatically better high-z template fits.

Quantify the fraction for which the global chi-square minimum lies at z > 2 or z > 3 despite low spectroscopic redshift.

Distinguish:

- A strong incorrect high-z minimum.
- Multiple nearly degenerate solutions.
- A numerical failure.
- An apparently good chi-square fit at the wrong redshift.

Plot representative chi-square-versus-redshift curves selected by predeclared physical criteria, not by manually picking visually favourable examples.

## 5. Paired HSC–Legacy comparison

Where the same galaxy has both HSC and Legacy photometry, use the audited same-object multi-view cohort.

Keep exactly the same spectroscopic redshift.

Compare:

1. HSC-only photometry and HSC response curves.
2. Legacy-only photometry and Legacy response curves.
3. Their respective best-fit SEDs and coefficients at the same fixed z_spec.
4. Their best-fit unconstrained solutions.
5. Their per-band residuals in physically consistent flux units.

This is especially valuable because galaxy population and true redshift are controlled by object identity.

Where possible, construct a controlled joint HSC+Legacy likelihood for the same object, with documented assumptions about correlated errors and photometric calibration.

Do not assume independent measurements of overlapping objects are necessarily statistically independent.

Use this joint fit only as a diagnostic.

Determine whether the same intrinsic SED can consistently fit both observational views.

If not, characterise the wavelength-dependent discrepancy.

Do not fit source/target photometric corrections using CLAUDS labels.

## 6. Minimal hypothesis-discriminating interventions

After completing the unmodified baseline diagnosis, run a small fixed set of interventions.

### Intervention A — More accurate integration

Replace the original integration with the independently verified improved calculation if a significant discrepancy is confirmed.

Change nothing else.

### Intervention B — Realistic effective uncertainties

Compare the original measured-error P0 fit against the existing source-selected template-error or fractional-uncertainty model.

Do not simply raise the error floor until outliers improve.

Report separately:

- Changes in best-fit redshifts.
- Changes in chi-square.
- Changes in CE and coverage.
- Whether the posterior merely broadens while the incorrect minimum persists.

A likelihood-temperature change alone cannot repair a systematically incorrect chi-square minimum.

### Intervention C — Template-family coverage

Test whether the fixed 12-template basis can reconstruct the source SEDs at z_spec.

Where warranted, use a small independently documented alternative or extended template basis.

Prefer a source-agnostic physical choice or source-training-defined choice.

Do not choose templates based on CLAUDS performance.

Check whether improvement comes from genuine SED coverage rather than increased overfitting.

### Intervention D — Filter and extinction conventions

Test only documented physically plausible alternatives where independent evidence supports them.

If per-object HSC r/r2/i/i2 exposure fractions are unavailable, retain that uncertainty explicitly.

If extinction corrections are tested, use independent foreground information and record the adopted law.

Do not permit arbitrary label-optimised zero-point shifts to masquerade as a physical correction.

### Intervention E — Redshift prior sensitivity

If needed, use a separately declared source-defined prior as a diagnostic.

Keep the prior-free likelihood result alongside it.

An improved posterior obtained by downweighting high redshifts is not evidence that the original high-z chi-square minima were physically corrected.

## 7. Quantify attribution carefully

For each intervention report the matched-object change in:

- z_spec-constrained chi-square.
- Global chi-square minimum.
- Delta chi-square.
- Median predicted redshift.
- NMAD.
- Catastrophic-outlier fraction.
- CE, CRPS and coverage.
- Fraction of wrong high-z minima.
- Numerical or fit failures.

Where interventions overlap, do not add their improvements as independent causal contributions.

Distinguish corrections to the forward model from source-supervised model selection and posterior calibration.

If no intervention identifies a clear mechanism, say so.

## 8. Execution order and resource limits

1. Archived replay.
2. Independent forward-model integration audit.
3. HSC photometric mapping/uncertainty audit.
4. NNLS and fixed-z_spec diagnostics.
5. Same-object HSC–Legacy comparison.
6. Small mechanism-specific interventions.
7. Fixed source-validation confirmation of justified corrections.

Start with pre-specified representative subsets.

Scale expensive EAZY refits only when necessary.

No broad template, prior or error-floor sweep.

Use source training for any diagnostic configuration development and keep confirmation results separately identifiable.

Do not interrupt existing jobs or overwrite archived artifacts.

## 9. Deliverables

Create a new directory such as:

`filter-aware-runs/1028/stage2a5-hsc-eazy-audit/`

Save:

- Baseline replay and manifest.
- Independent integration tests.
- HSC/Legacy photometry audit.
- NNLS numerical checks.
- Per-object chi-square(z) and z_spec diagnostics.
- Per-band residual tables.
- Same-object HSC–Legacy comparisons.
- Predefined intervention results.
- Chi-square curves, residual distributions and redshift-mode plots.
- Exact commands and code hashes.
- A scientific root-cause report.

End with direct answers:

1. Why does EAZY P0 catastrophically fail on HSC while performing much better on Legacy?
2. How much of the difference is attributable to numerical integration, template coverage, photometric conventions or error modelling?
3. Does any correction genuinely restore the correct redshift likelihood minima, rather than simply broadening the PDFs?
4. Which source-verified physical likelihood, if any, can be safely used as an N2 decoder constraint?
5. What unresolved measurement/physics issues must remain explicit in the next cross-survey experiment?

**Execute the diagnostic work and feasible source-only confirmation. Do not stop at a proposal. Do not use CLAUDS target labels for corrections or model selection.**