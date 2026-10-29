# Stage 2A — Physical SED-Template Baseline for Cross-Survey Photo-z

## Scientific objective

Establish a physically consistent, reproducible SED-template photometric-redshift baseline across three real astronomical catalogues:

1. GalaxiesML-Spec — HSC grizy, DESI spectroscopic-redshift labels.
2. DESI–Legacy — Legacy grzW1W2, DESI spectroscopic-redshift labels.
3. CLAUDS — ultraviolet/optical/NIR photometry, ZPHOT reference labels.

**CLAUDS is the primary scientific target.**

The objective is to determine whether explicit SED physics and full filter-response integration can infer meaningful redshifts in CLAUDS without training on CLAUDS redshift labels.

Transfer between GalaxiesML-Spec and DESI–Legacy alone is not a successful scientific endpoint.

## Background

Previous experiments established:

- HSC-trained Qwen reaches in-domain NMAD approximately 0.0268.
- The same model fails on CLAUDS with approximately 96–97% catastrophic outliers.
- Converting CLAUDS photometry to common nanomaggy flux units does not rescue it.
- HSC source median i magnitude is approximately 18.45; CLAUDS median is approximately 25.51.
- Source/target S/N differs by approximately two orders of magnitude.
- Qwen is highly sensitive to input order, terminal-token identity and flux scale.
- A small synthetic SED-template baseline substantially outperformed frozen Qwen representations on unseen filter systems, although its PDF calibration was poor.

This does not establish that physical template fitting will work on real CLAUDS.

**Test that proposition directly.**

## 1. Inspect the repository and existing reports

Read the completed Stage 0, Stage 0.5 and Stage 1 reports and their implementation manifests.

Inspect:

- Existing GalaxiesML-Spec photometry loader.
- Existing DESI–Legacy loader.
- Corrected CLAUDS common-flux interface.
- `configs/filter_registry_v1.json`
- `configs/filter_responses_v1/`
- Existing redshift-PDF evaluation utilities.
- Existing source train/validation partitions and CLAUDS diagnostic IDs.

Identify missing dependencies and data caches before starting.

Do not alter archived model checkpoints or earlier experiment outputs.

## 2. Establish physical photometric consistency

Use common linear flux-density units, preferably AB nanomaggies.

For CLAUDS hscPipe cmodel fluxes reported in microJy, use the established conversion:

`f_nMgy = 0.2754228703 * f_microJy`

Do not reuse the historical 23-mag zero-point convention as if it were physically correct.

Audit independently for every catalogue:

- Instrument and survey.
- Filter identity and physical throughput.
- Flux-density units.
- Photometric uncertainty units.
- Extinction conventions.
- Measurement type and aperture/cmodel semantics.
- Availability masks and quality flags.
- Signed and non-detected fluxes.
- Possible systematic calibration differences.

Preserve valid negative fluxes.

Do not replace missing observations with fabricated zero flux and zero uncertainty.

Do not interpret missing EBV as EBV=0.

The primary configuration must not depend on CLAUDS having an unavailable EBV value.

Where extinction correction is uncertain, explicitly compare a pre-specified consistently untreated-photometry configuration against a documented correction strategy using independently available information. Do not fit target extinction from redshift labels.

If exact source/target processing equivalence cannot be established, record that as a limitation rather than silently declaring it solved.

## 3. Implement a maintained EAZY baseline

Prefer the existing maintained `eazy-py` implementation.

Inspect its installed API and confirm the available functions rather than assuming signatures.

Use an appropriate documented galaxy SED template library with suitable UV, optical and near-IR coverage.

Document the templates, versions, file hashes, dust assumptions and IGM treatment.

Integrate SEDs through the actual relevant filter response curves.

Use full response curves, not just pivot wavelength and effective width.

Generate or verify any EAZY filter files and photometric column translation tables required for each instrument configuration.

Explicitly distinguish DECam and BASS/MzLS Legacy responses where release metadata requires it.

Keep documented HSC r/r2 and i/i2 approximations explicit.

### Physical verification tests

Before fitting real galaxies, test:

1. AB magnitude and nanomaggy normalisation.
2. Constant-f_nu filter integration.
3. Redshifted SED integration through each survey's filters.
4. Conversion between spectral flux densities.
5. Invariance under consistently rescaling source flux and its uncertainty where only a free luminosity amplitude is relevant.
6. Signed flux measurements and masked observations.
7. Missing-band configurations.
8. Filter-curve wavelength units and response normalisation.
9. Agreement with independent synthetic-photometry calculations from Stage 1 where compatible.

Do not assume a fitted redshift is meaningful merely because the optimiser returns a finite result.

## 4. Pre-specify the physical inference configurations

Start with a minimal configuration matrix.

### Configuration P0 — Template likelihood without population prior

Fit physical templates and their nonnegative amplitudes to observed photometry.

No externally fitted redshift prior.

Treat overall luminosity or template normalisation as a nuisance quantity.

This is the primary physics-only diagnostic.

### Configuration P1 — Source-defined population prior

Optionally include a documented redshift/magnitude prior appropriate for the available photometry.

The prior must be specified or selected using the spectroscopic source data or independently established external information, never CLAUDS ZPHOT performance.

If the standard EAZY prior uses a filter unavailable in a catalogue, do not silently substitute another band or its calibration.

Make prior-free and prior-based performance separately visible.

### Configuration P2 — Fixed source-calibrated uncertainty model

Investigate whether a source-calibrated template-error prescription or fixed uncertainty broadening improves PDF calibration.

Source spectroscopic validation alone may be used to select this uncertainty model.

Do not tune uncertainty broadening against CLAUDS ZPHOT.

Avoid a large template/prior/temperature hyperparameter search.

The minimum viable experiment is P0; do not allow optional P1/P2 to block completion.

## 5. Redshift PDF conventions

Initially use a common evaluation grid covering 0 <= z <= 4.5, compatible with the previous Qwen experiment.

If EAZY uses a different native fitting grid, preserve the native inference calculation and rebin probability masses correctly for matched evaluation.

Do not interpolate discrete posterior masses as if they were probability densities without accounting for bin widths.

Distinguish:

- Profile chi-square versus redshift.
- Template likelihood.
- Magnitude prior.
- Template-error correction.
- Final normalised posterior or pseudo-posterior.

Do not identify `exp(-chi2/2)` automatically with EAZY's fully configured posterior.

Ensure:

- Finite and normalised bin probabilities.
- Correct interval definitions.
- Correct redshift-bin centres and boundaries.
- Correct CE/CRPS calculations.
- Explicit handling of insufficient valid bands or failed fits.

Any evaluation outside 0–4.5 must be reported separately, with a compatible model grid.

## 6. Source evaluation

Use existing spectroscopic source-validation cohorts.

Start with a modest deterministic screening sample to verify implementation and runtime.

Then scale to the available full validation cohorts where practical.

Evaluate both GalaxiesML-Spec and DESI–Legacy.

Perform an overlap audit when combining or comparing source samples so repeated galaxies are not treated as independent.

Report:

- NMAD.
- Catastrophic-outlier fraction, using the historical |delta z|/(1+z) > 0.15 definition.
- Bias and median bias.
- CE and CRPS.
- Nominal 68% interval coverage.
- PIT diagnostics where appropriate.
- Predicted redshift distribution.
- Fraction of failed or unsupported fits.
- Performance by redshift, magnitude, S/N and photometric completeness.

Do not optimise only aggregate NMAD while ignoring catastrophic failures and PDF calibration.

Any source-based template-error or prior selection must use source validation only, with separate untouched confirmation data reserved where possible.

## 7. Primary evaluation: CLAUDS

Use the existing 256 CLAUDS development/diagnostic validation queries for direct comparison with previous frozen Qwen experiments.

Their redshift references are ZPHOT, not spectroscopy.

Do not use these labels to select templates, priors, zero points, error floors, model versions or checkpoint configurations.

Include:

- Full valid CLAUDS UV/optical/NIR photometry.
- HSC grizy-only configuration using actual matching measurements and correct bandpasses.
- A documented missing-data treatment.
- Pre-specified quality/completeness strata.
- The existing Stage-0 photometric-support strata.

Evaluate EAZY with fixed source-defined or prior-free settings.

The primary question is whether a physical template baseline avoids the near-constant low-z collapse observed in Qwen.

Report both accuracy and redshift-distribution diagnostics:

- NMAD, outliers, bias, CE, CRPS, coverage.
- Median predicted redshift and its 5th–95th percentile range.
- Fraction concentrated in narrow redshift intervals.
- Prediction-versus-reference scatter.
- PDFs for pre-selected objects.
- High-z and faint-object behaviour.
- Failure rate versus number of usable bands.

**A low NMAD from a nearly constant predictor is not success.**

Because CLAUDS ZPHOT may itself come from template fitting, agreement with it is not independent proof of true-redshift accuracy.

Explicitly investigate the provenance of the ZPHOT catalogue and any shared template assumptions.

If an independent CLAUDS spectroscopic subset is available, identify it but reserve it for a frozen, pre-registered confirmation protocol. Do not use it for exploratory tuning.

## 8. Brightness and noise stress tests

Use the Stage-0.5 findings to construct a limited physics-baseline robustness comparison.

On source spectroscopic validation objects, predefine controlled:

- Common flux/uncertainty rescaling.
- Increased reported errors.
- Consistent injected-noise cases.

Use matching object identities and physical conventions.

Determine whether the physical estimator is less sensitive than archived Qwen to overall flux scale at fixed S/N.

Distinguish likelihood-only behaviour from effects introduced by a magnitude prior.

Do not portray simulated flux dimming as equivalent to a realistic high-redshift population.

## 9. Decision criteria

The final report must answer:

A. Can physical template fitting provide object-dependent CLAUDS redshift predictions without CLAUDS-label fitting?

B. Does it avoid the catastrophic low-z collapse seen in archived Qwen models?

C. How good are the redshift point estimates and PDFs?

D. Do full UV/NIR bands improve inference relative to grizy alone?

E. How sensitive are the results to priors, extinction, photometric completeness and measurement uncertainty?

F. Is the physical baseline credible enough to guide Stage 2B?

Do not claim success solely because the predicted redshift distribution looks broader.

Do not claim failure of astrophysical physics when poor performance could reflect inadequate templates, catalogue corrections or intrinsic photometric degeneracies.

## 10. Resource and execution policy

No Qwen training or adapter sweeps.

Start with CPU or modest GPU fitting if supported.

Validate a small cohort before scaling.

Avoid unnecessary repeated expensive fitting across large configuration matrices.

Do not overwrite historical artifacts or interrupt active jobs.

If complete Legacy data are unavailable, report the limitation and proceed with verified HSC and CLAUDS photometry; leave a reproducible Legacy launch command.

## 11. Deliverables

Create a separate output directory such as:

`filter-aware-runs/1028/stage2a-physical-baseline/`

Produce:

1. Physical input and unit audit.
2. Filter and template provenance.
3. EAZY configuration and dependency versions.
4. Passing physical forward-model tests.
5. Source-validation results.
6. Fixed CLAUDS diagnostic results.
7. PDF-calibration analysis.
8. Brightness/noise robustness comparison.
9. Per-object predictions and representative PDFs.
10. Exact commands, manifests and reproducibility instructions.
11. Concise scientific assessment.

End by providing a clear **Stage 2B handoff** describing the verified filter-response interface, available physical likelihood products, known limitations and the minimal baselines that must be retained.

**Execute the feasible physical-fitting experiments. Do not stop after a design report. Do not use CLAUDS redshift labels to improve EAZY's configuration.**