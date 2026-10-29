# Stage 2 — Physics-Guided Qwen for Cross-Survey Photometric Redshifts

## Scientific objective

Develop and test a physics-guided foundation-model approach to photometric-redshift inference.

The ultimate scientific objective is successful transfer between genuinely different survey datasets, especially:

**GalaxiesML-Spec + DESI–Legacy → CLAUDS**

and, as a complementary direction:

**CLAUDS → independently labelled DESI spectroscopic catalogues.**

Previous HSC-trained Qwen models achieved strong in-domain performance but failed on CLAUDS even after numerical flux harmonisation and filter-aware serialization.

The present hypothesis is that a direct Qwen-to-redshift mapping does not sufficiently exploit physical spectral constraints.

We will therefore test a hybrid approach combining:

1. An explicit astrophysical SED-based redshift likelihood.
2. A Qwen-based learned correction.
3. Matched non-foundation-model controls.

**The experiment must distinguish the value of adding physical modelling from any extra value contributed by Qwen's pretrained representation.**

## 1. Prerequisites and inspection

First inspect the Stage-0 and Stage-1 reports, if available.

Verify:

- Photometric units and measurement semantics are sufficiently understood.
- Filter response mappings are reproducible.
- Source/target common-support diagnostics are documented.
- Stage-1 physics-probe conclusions are available, or explicitly record their absence.

Do not assume that Stage 1 has demonstrated successful astrophysical reasoning.

Inspect the repository for:

- Existing Qwen3.5-4B-Base QLoRA checkpoints and PDF heads.
- GalaxiesML-Spec loader.
- DESI–Legacy loader.
- Corrected CLAUDS common-flux loader.
- Filter response registry.
- Existing MLP baselines.
- Redshift grid and PDF evaluation utilities.

If Stage 0 has identified unresolved fatal unit/filter mismatches, resolve or isolate them before expensive model training.

## 2. Establish a physically consistent photometric interface

Use measured flux densities and uncertainties in consistent physical units, preferably nanomaggies.

The CLAUDS raw microJy conversion must follow the verified factor:

`f_nMgy = 0.2754228703 * f_microJy`

Preserve:

- Signed fluxes.
- Actual measurement errors.
- Missingness and quality masks.
- Correct survey/instrument/filter identities.
- Available Galactic extinction information.
- True observational band sets.

Do not convert missing data into artificial zero-valued measurements.

Handle different photometric measurement conventions explicitly.

Avoid using spectroscopic labels, ZPHOT or previously inferred redshifts as input features.

Any catalogue-specific zero-point adjustment must be fixed using permitted source calibration information, never inferred from CLAUDS target redshift performance.

## 3. Construct an EAZY-based physical baseline

Use a maintained, documented implementation of EAZY/eazy-py if compatible with the environment.

Avoid reimplementing template fitting unless necessary.

Configure EAZY with:

- A documented galaxy SED template library.
- Correct filter response curves.
- A common redshift grid.
- Consistent observational units.
- An explicitly specified treatment of dust and IGM absorption.
- Explicit measurement-error handling.
- Explicit template-error and redshift-prior settings.

Inspect the actual installed EAZY version and APIs rather than assuming specific function signatures.

EAZY may expose redshift-dependent chi-square fits, log likelihoods, priors and other terms. Distinguish these carefully.

Do not silently equate a profiled `-chi2/2` score with EAZY's fully corrected posterior probability.

Use a common redshift grid for controlled comparisons, initially 300 bins over 0–4.5 where data support permits.

Evaluate bin masses consistently with the existing Qwen PDF conventions.

Verify PDF normalisation, redshift-bin alignment, finite values and handling of failed fits.

Do not artificially force a high-quality result by imposing CLAUDS-derived redshift priors.

### Baseline evaluation

Evaluate the frozen EAZY configuration on:

- GalaxiesML-Spec source validation.
- DESI–Legacy source validation.
- Existing CLAUDS diagnostic validation objects.

Report:

- NMAD.
- Catastrophic-outlier fraction.
- Bias.
- CE and CRPS.
- 68% coverage.
- Redshift-PDF diagnostics.
- Failures caused by missing photometry or invalid error information.

For CLAUDS, the existing reference labels are ZPHOT rather than spectroscopic redshifts.

Since ZPHOT may itself derive from template fitting, agreement with those labels does not constitute independent proof of redshift accuracy.

If an independent CLAUDS spectroscopic subset can be constructed without contaminating training or selection, reserve it for final confirmation.

## 4. Hybrid redshift inference

Construct a physics-guided redshift PDF.

Let the template baseline provide a normalised physical-model redshift mass vector:

`p_phys[z_bin]`

Let Qwen provide a learned redshift-dependent correction:

`delta_theta[z_bin]`

Implement:

`p_hybrid = softmax(log(p_phys + eps) + delta_theta)`

where the precise handling of zero probabilities and invalid bins must be documented and tested.

The formula is schematic: use a numerically stable implementation, and apply a consistent redshift-bin convention.

The learned correction must be regularised so that it cannot trivially erase the physical baseline everywhere.

Possible regularisation includes:

- A penalty on the magnitude of the correction.
- A KL-based consistency term relative to the physical PDF.
- A constrained or scaled correction amplitude.

Choose the simplest viable mechanism.

Any regularisation strength or correction scale must be selected using source validation only.

Do not tune hyperparameters using CLAUDS target-label scores.

## 5. Construct matched model comparisons

Implement four primary models:

### Model A — EAZY only

The fixed physical template-fitting baseline.

No Qwen or learned residual.

### Model B — Existing Qwen-only estimator

The existing direct Qwen photo-z system, with a physically consistent input interface.

Where possible, reuse archived checkpoints as diagnostic baselines.

If a new mixed-survey Qwen-only model is needed for a fair comparison, train it using the same source cohort and supervision budget as the hybrid.

### Model C — EAZY + MLP residual

Use a lightweight numerical MLP to predict a correction to the same physical-model redshift PDF.

The MLP should receive the relevant photometric information, validity masks and filter metadata through a numerical encoding.

Train under the same source/validation constraints as the Qwen residual.

This baseline determines whether adding an ordinary supervised residual is sufficient.

### Model D — EAZY + Qwen residual

Use Qwen3.5-4B-Base to generate a redshift-dependent correction to the physical-model PDF.

Prefer starting from an existing well-performing QLoRA checkpoint where compatible, rather than randomly initialising the entire system.

Preserve the trained PDF head when its redshift grid and architecture are compatible.

Use a new task-specific head only when necessary and clearly document why.

Train conservatively, initially with limited additional updates.

A fixed physical baseline can be precomputed and cached; gradients need not propagate through EAZY.

Do not launch a large adapter-method sweep.

### Additional low-cost control

Where technically feasible, test a fixed combination of archived Qwen and EAZY PDFs without retraining Qwen.

This can establish whether the two existing estimators provide complementary information before introducing a learned correction.

Do not treat a fitted combination weight as zero-shot unless it was selected exclusively using source data.

## 6. Training protocol

For the primary transfer direction, use both:

- GalaxiesML-Spec, native HSC grizy with DESI spectroscopic redshifts.
- DESI–Legacy, native grzW1W2 with DESI spectroscopic redshifts.

Use a common physically consistent input representation.

Allow survey identity and genuine filter metadata to vary between objects.

Check object overlap carefully, including matching DESI identifiers and sky coordinates where appropriate.

Do not place observations of the same galaxy across source train and source validation boundaries.

Use pre-defined source training/validation cohorts and source-only checkpoint selection.

Be careful that differing sample counts, redshift distributions and selection functions do not allow one survey to dominate training unintentionally.

Document any source sampling or weighting strategy.

Do not use any CLAUDS ZPHOT labels for:

- Training.
- Checkpoint selection.
- Hyperparameter tuning.
- Calibration.
- Learning-rate decisions.
- Input representation selection.

The primary transfer evaluation must remain zero-shot with respect to CLAUDS labels.

## 7. Evaluate genuine cross-survey transfer

The main target is CLAUDS.

Use previously inspected CLAUDS objects only for diagnostic comparisons.

Before examining a fresh confirmation set, freeze:

- Model architecture.
- Filter mappings.
- Source training protocol.
- Hyperparameters.
- Selection metric.
- Checkpoint.
- Primary evaluation metrics.

Where a genuinely untouched CLAUDS validation or test partition is available, reserve it for a single pre-specified confirmation run.

Otherwise explicitly classify the evidence as exploratory.

Evaluate:

- EAZY-only.
- Qwen-only.
- EAZY + MLP.
- EAZY + Qwen.

For each model report:

- Source-domain performance.
- CLAUDS transfer performance.
- Conditional performance versus redshift and brightness.
- Catastrophic-outlier fractions.
- Bias and probability calibration.
- Model behaviour on common-support and low-support target objects.
- Predicted PDF concentration or collapse.
- Computational cost.

The principal outcome is meaningful CLAUDS redshift inference, not HSC-to-Legacy improvement alone.

Do not mistake a low NMAD from nearly constant biased predictions for successful photo-z inference.

## 8. Complementary CLAUDS-source direction

If the spec-source-to-CLAUDS experiment fails or produces ambiguous results, prepare the reverse-transfer experiment:

**Corrected CLAUDS-source → HSC/Legacy spec-z targets.**

Use physically correct CLAUDS photometry and its existing ZPHOT supervision, with the known limitations of photometric rather than spectroscopic training labels.

Evaluate the resulting model against independent DESI spectroscopic references.

Do not recycle the historical incorrect-zero-point CLAUDS checkpoint as if its input conventions were physically correct.

Keep the reverse direction separate from the main experiment.

Do not combine the two directions into an undifferentiated performance score.

## 9. Distinguish explicit physics from pretrained knowledge

The hybrid architecture guarantees that an explicit physical model enters the inference.

It does **not** guarantee that Qwen uses pretrained astrophysical knowledge.

Interpret results as follows:

- If EAZY-only outperforms direct Qwen on CLAUDS, physical modelling is valuable for transfer under the tested setup.
- If EAZY + MLP performs as well as EAZY + Qwen, the Qwen backbone provides no demonstrated advantage in that comparison.
- If EAZY + Qwen reliably outperforms both EAZY and EAZY + MLP, Qwen contributes useful additional predictive ability.
- If Stage-1 pretraining controls also support a benefit from pretrained astrophysical representations, the argument for exploiting pretrained knowledge becomes stronger.

Avoid treating these claims as interchangeable.

Include a small pretrained-versus-random-representation or frozen-backbone control if Stage 1 has not already established a relevant result.

Do not claim physical reasoning from performance improvements alone.

## 10. Resource and execution policy

Proceed in stages:

1. Establish a correct physical/template-only baseline.
2. Validate common PDF conventions and source inputs.
3. Test a fixed EAZY/Qwen combination without training.
4. Train the EAZY + MLP residual.
5. Train a small EAZY + Qwen residual pilot.
6. Expand training only if source validation and model-integrity checks justify it.
7. Conduct the frozen target evaluation according to the pre-specified protocol.

Avoid a large hyperparameter or adapter sweep.

Do not interrupt running jobs or overwrite archived outputs.

If prerequisites fail, document the blocker and complete the safe diagnostic components instead of launching scientifically uninterpretable training.

## 11. Deliverables

Produce a reproducible Stage-2 experiment directory containing:

- Physically verified photometry inputs.
- EAZY configuration, templates and filter provenance.
- Source/target matching and split audit.
- Cached physical redshift likelihoods.
- Hybrid model implementation and tests.
- Source-only training and checkpoint records.
- Full four-model comparison.
- CLAUDS redshift-PDF and calibration diagnostics.
- Per-object paired predictions.
- Exact configurations, dependency versions, hashes and commands.
- Scientific report distinguishing physical-model contributions, Qwen contributions and pretrained-knowledge claims.

End with direct answers to:

1. Can a physically informed method infer meaningful redshifts for CLAUDS without CLAUDS-label training?
2. Does Qwen improve on the physical baseline?
3. Does pretrained Qwen offer an advantage over an ordinary supervised residual?
4. Which evidence supports actual use of pretrained astrophysical knowledge?
5. Is the result strong enough to become a central claim of Paper I?

**Execute the feasible stages in order. Prioritise empirical evidence over architectural complexity. Do not launch expensive training until the simpler controls are validated.**