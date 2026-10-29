# Stage 2B.5 — Synthetic Learnability and Physics-Decoder Audit

## Scientific objective

Diagnose why the Stage 2B numerical models fail to demonstrate learning on the controlled synthetic photometric-redshift benchmark, despite successful optimisation on a small real HSC+Legacy paired-source pilot.

The goal is to distinguish:

1. Implementation or optimisation failure.
2. Unstable PDF-head training.
3. Insufficient or inappropriate synthetic supervision.
4. Train/validation distribution mismatch.
5. Genuine photometric-redshift non-identifiability.
6. Incorrect or excessively restrictive physical-decoder assumptions.
7. Failure of cross-filter inductive bias despite successful source learning.

**Do not conclude that physics-aware neural modelling is ineffective until the models have demonstrated valid learnability on a controlled task.**

This is a small diagnostic experiment. Do not launch large-scale training.

## 1. Existing evidence

Stage 2B implemented three main numerical architectures:

- N0: padded numerical MLP.
- N1: shared band encoder with response-curve representation and masked set pooling.
- N2: the same encoder/head plus a differentiable SED-basis decoder and physical profile score.

The real paired-source pilot used 1,024 unique training galaxies and 512 validation galaxies, each with actual HSC and Legacy observations.

Real-source results:

| Model | HSC NMAD | Legacy NMAD |
|---|---:|---:|
| N0 | 0.0811 | 0.0967 |
| N1 | 0.0386 | 0.0359 |
| N2 | 0.0492 | 0.0409 |

However, all five synthetic model arms selected **update zero** after 800 attempted optimiser updates because source-validation CE did not improve sufficiently.

On strictly held-out filters:

- N1: 76.56% catastrophic outliers, predictions concentrated in one redshift bin.
- N2: 84.38% catastrophic outliers, more variable predictions but still poor accuracy.

The synthetic training set contains only 512 labelled source observations from 128 independent SED instances, each evaluated at two redshifts and under HSC/Legacy responses.

The synthetic SED generator and the Stage 2A physical basis are not identical.

Stage 2A uses FSPS QSF templates and Inoue14 IGM; Stage 1 includes analytic/CWW families and simplified attenuation.

The small real paired-source training population contains no z >= 2 galaxies, but that limitation must not be conflated with synthetic-data learnability.

**The current synthetic comparison does not provide a reliable architecture ranking.**

## 2. Hard constraints

- No Qwen, DoRA, QLoRA or large-backbone training.
- Do not change archived Stage 1/2A/2B results.
- No use of CLAUDS ZPHOT for training, model selection or debugging.
- Do not inspect held-out real-survey test data.
- Preserve original synthetic evaluation splits and truth files.
- Do not tune against strict held-out-filter or held-out-family results.
- Diagnose training behaviour using synthetic source-training and source-validation data.
- Keep all new diagnostic experiments separate from the original fixed Stage 2B benchmark.
- Do not interpret selecting update zero as proof that no learning occurred during the training trajectory.

The synthetic benchmark may be extended with a new pre-specified source-only training distribution, provided the original benchmark and its independent evaluation sets remain identifiable.

## 3. Reproduce the update-zero failure

Inspect:

- `scripts/stage2_numerical.py`
- Existing N0/N1/N2 implementations.
- Stage 2B training histories.
- Stage 1 synthetic generator and split manifests.
- Stage 2A verified response kernel.
- N2 differentiable SED decoder.
- Source-validation checkpoint-selection logic.

For each synthetic arm, reconstruct:

- Train loss by update.
- Source-validation CE by update.
- Gradient norms.
- Parameter update norms.
- Initial and final logits.
- PDF entropy and concentration.
- Prediction-head bias and weight changes.
- Fraction of redshift bins receiving meaningful probability mass.

Check:

- Actual targets and redshift-bin indices.
- Redshift-bin boundaries and label conversion.
- Normalisation and numerical stability of PDF masses.
- Whether gradients reach the encoder and head.
- Whether source and validation examples are serialised identically.
- Whether missing-band masks are applied correctly.
- Whether learning-rate scheduling and weight decay are actually active.
- Whether training and evaluation model modes are correct.
- Whether physical and neural scores are accidentally detached.
- Whether optimizer parameter groups include the intended parameters.

Compare train CE at update zero, intermediate updates and the final update.

A decreasing training loss with worsening validation CE indicates a different problem from an optimisation failure where even training loss cannot decrease.

Do not merge these explanations.

## 4. Controlled learnability ladder

Construct a sequence of increasingly difficult synthetic tasks.

Every stage must have independent train and validation SED identities.

All source distributions, seeds, supervision counts and evaluation criteria must be fixed before training.

### Task L0 — Elementary redshift mapping

Generate a simple rest-frame spectrum with one prominent, physically defined break.

Use exact and reproducible filter integration.

Choose a redshift range and filter set that make the break identifiable.

Begin with:

- No measurement noise.
- No dust.
- A fixed intrinsic SED family.
- Fixed or profiled luminosity amplitude.
- Correct physical filter responses.
- Uniformly sampled source redshifts with adequate coverage.

Verify identifiability with an independent grid-search or analytic physical baseline.

Train N0 and N1 on this elementary task.

The models must demonstrate:

- Reduction of training CE from initialization.
- Reduction of independent in-distribution validation CE.
- Object-dependent predictions.
- Approximate tracking of the known redshift.

If they fail, diagnose and repair the training implementation before proceeding.

Do not use the held-out filter task to tune the solver.

### Task L1 — Multiple identifiable SED families

Add several distinguishable spectra with controlled break strengths and continuum slopes.

Keep noise initially absent.

Use independently sampled intrinsic parameters.

Check whether the model learns redshift rather than memorizing a fixed spectral identity.

Hold out complete SED parameter realizations.

### Task L2 — Dust, luminosity and photometric noise

Add separately:

1. Independent luminosity normalisation.
2. Dust attenuation.
3. Moderate observational noise.
4. Missing filters.
5. Weak-break or featureless SEDs.

Record how redshift identifiability changes.

Do not demand exact redshift recovery for fundamentally degenerate spectra.

Include an independently computed physical likelihood to identify objects for which the photometry contains little redshift information.

### Task L3 — Source filter diversity

Train on multiple actual source response systems.

Use the same intrinsic SED/redshift population under different source filters.

Then evaluate on independent source-validation SEDs.

Measure whether learned predictions depend appropriately on actual bandpass responses.

Only after source validation succeeds should the locked models be evaluated on the original strict held-out filters.

### Task L4 — Original Stage-1 benchmark

Return to the original synthetic benchmark without altering its frozen evaluation membership or truth.

Use the validated architecture and optimisation procedure selected solely from the earlier source-only learnability tasks.

Report whether the original update-zero failure persists.

Do not retrospectively describe this new training protocol as the originally registered Stage 2B experiment.

## 5. Optimisation and capacity controls

Begin with a tiny overfitting sanity check.

Choose a deterministic set of approximately 16–32 simple labelled source examples.

Train each architecture to fit these examples under a sufficiently simple identifiable task.

This is a debugging test, not generalisation evidence.

Verify that:

- Target PDF bins are reachable.
- Training CE decreases.
- Prediction-head weights change.
- Gradients are finite.
- Model outputs differ meaningfully between distinct inputs.
- Label-shuffled controls do not produce genuine held-out predictive performance.

Then use a small source-only optimisation screen to diagnose whether the earlier failure was due to:

- Learning rate.
- Weight decay.
- Head output bias.
- Training duration.
- Class/bin sparsity.
- Sample count.
- Architectural bottlenecks.
- Numerical normalisation.

Prefer correcting concrete bugs or clear undertraining rather than undertaking a broad hyperparameter search.

Where possible, preserve approximately matched parameter and example budgets between N0, N1 and N2.

If larger datasets are required, generate additional unique source-training SEDs.

Do not increase training diversity by repeatedly duplicating the same SED identities while describing the samples as independent.

## 6. N2 physical decoder audit

This is a separate, crucial component.

The current N2 uses:

- Twelve nonnegative SED coefficients.
- The Stage 2A physical response kernel.
- A profiled nonnegative luminosity amplitude.
- A redshift-dependent chi-square score.
- Bounded neural redshift logits.
- A source-fixed combination coefficient.
- A photometric reconstruction objective evaluated at the known source spectroscopic redshift during training.

Audit each component independently.

### 6.1 Matched-physics oracle test

Generate synthetic photometry directly from the same twelve SED basis templates, response integration and noise assumptions used by N2.

Use known basis mixtures and redshifts.

Verify that a correctly implemented template inference calculation recovers appropriate likelihood minima at the known redshifts.

Include non-identifiable cases explicitly.

Compare:

- Known mixture at true redshift.
- NNLS-optimised mixture at true redshift.
- Profiled mixture across candidate redshifts.
- Neural-predicted mixture across candidate redshifts.

This distinguishes an inadequate learned coefficient representation from a malfunctioning physical kernel.

### 6.2 Mismatched-physics test

Repeat using Stage 1 analytic/CWW synthetic SEDs.

Compare performance against the matched-template case.

Quantify:

- True-redshift reconstruction residuals.
- Minimum chi-square at true and alternative redshifts.
- Error associated with template incompleteness.
- Sensitivity to different SED families.
- Whether an incorrect physical likelihood dominates the neural prediction.

Do not blame the neural encoder for a fundamentally mismatched physical basis without demonstrating that mismatch.

### 6.3 Gradient and loss-scale audit

Inspect the actual N2 combination rule:

`logits = 4*tanh(raw/4) + 0.2*(-0.5*(chi2-min_chi2)/Nvalid)`

Verify the intended gradients and scaling.

Determine whether:

- The physical term overwhelms the neural logits.
- The bounded neural term prevents recovery from an incorrect physical minimum.
- The physical score is too flat after tempering.
- The SED coefficients receive useful gradients.
- The reconstruction term conflicts with redshift CE.
- The nonnegative amplitude profile is stable.
- Effective errors are dominated by the source-selected fractional error term.
- Masking produces finite results for empty or partial band sets.

Test limiting cases such as:

- Neural-only.
- Physical-only.
- Matched oracle template fitting.
- The original combined model.

Use source-only validation to determine whether a combination is numerically coherent.

Do not claim physics improves transfer merely because the combined model produces a wider predicted-redshift distribution.

### 6.4 Physical correctness invariants

Verify:

- Band-permutation invariance.
- Masked-set behaviour.
- Negative-flux handling.
- Overall amplitude scaling symmetry.
- Correct redshift-bin indexing.
- Finite gradients at empty/near-empty inputs.
- No z_true leakage at inference.
- Agreement between cached response kernels and direct integration.

Preserve Stage 2A.5 integration corrections where verified.

If Stage 2A.5 identified unresolved physical-kernel inconsistencies, do not reuse that kernel without explicitly delimiting the resulting error.

## 7. Separate learning from transfer

Evaluate every model at three distinct levels:

**Level A — Training learnability**

Can the model fit an elementary identifiable source problem?

**Level B — Source-family generalisation**

Can it infer redshifts for independent SED instances from the same physical family and filter systems?

**Level C — Cross-filter/SED generalisation**

Can it infer redshifts under unseen filter responses and held-out SED families?

Passing Level A does not imply Level B.

Passing Level B does not imply Level C.

Report the progression explicitly.

Do not allow a model to pass the physics-generalisation gate based only on memorising its training examples.

## 8. Model comparisons

Retain:

- N0 conventional numerical MLP.
- N1 shared band encoder.
- N2 shared encoder plus physical decoder.
- Physical-template-only baseline.

For meaningful physical tests, ensure the numerical models receive consistent observational information.

If N2 uses full filter curves through an external physical kernel while N1 receives an approximate response encoding, identify that information advantage.

Include compact-response versus full-response controls when relevant.

Maintain separate matched-template and mismatched-template experiments.

Use identical source-training and source-validation labels for fair comparisons.

Do not attribute superiority to physical inductive bias where training data, optimisation convergence or access to filter information differ.

## 9. Quantitative evaluation

For each learnability stage report:

- Source train CE.
- Independent source validation CE.
- NMAD.
- Catastrophic-outlier fraction.
- Bias.
- CRPS.
- 68% coverage.
- Predicted-redshift distribution.
- Narrow-interval concentration.
- PDF entropy.
- Gradient and update diagnostics.
- Physical reconstruction chi-square where applicable.

For paired counterfactual experiments report:

- Same-SED, same-redshift cross-filter prediction changes.
- Correct redshift changes when the SED is regenerated at a new redshift.
- Wrong-response metadata sensitivity.
- Noise and missingness sensitivity.

Use the existing redshift-PDF conventions consistently.

The previous Stage 1 benchmark used a 90-bin physical baseline while Stage 2B used 300 bins. Do not compare CE across incompatible discretisations without correct rebinning and documented probability conventions.

Use independent SED-instance bootstrap units rather than treating repeated observations of one SED as independent draws.

## 10. Gates and execution policy

### Gate 1 — Implementation correctness

Pass the optimisation, masking, gradient, PDF, response-integration and label-index tests.

### Gate 2 — Elementary learnability

N0 and N1 must learn a deliberately identifiable synthetic source problem.

N2's physical-only and oracle controls must recover the expected physical likelihood behaviour.

If Gate 2 fails, stop architectural comparisons and diagnose the cause.

### Gate 3 — In-distribution validation learning

At least one numerical architecture must improve meaningfully from initialization on independent source-validation SED instances.

Record any persistent overfitting or underfitting.

### Gate 4 — Physics benefit

N2 must be compared against a properly trained N1 and an independent physical baseline.

Only consider a physics benefit established if it persists under meaningful source and held-out physical controls, with appropriate uncertainty.

### Gate 5 — Real-survey expansion

Do not initiate population-scale real-data experiments solely because an easy synthetic task was learned.

Before expanding, require:

- Valid source-learning behaviour.
- Interpretable held-out-filter results.
- A verified physical decoder.
- Adequate source spectroscopic coverage.
- A viable independent Legacy training population if claiming mixed-survey population learning.

Do not rerun exploratory CLAUDS target evaluation to guide each new synthetic modification.

Existing CLAUDS results may be used as historical context, not a tuning oracle.

## 11. Resource budget

This stage should primarily use lightweight numerical models.

Start with CPU or modest GPU execution.

Use a small number of deterministic source-only configurations.

Run additional seeds only for promising configurations that pass the elementary correctness gates.

Do not initiate Qwen training, Stage 2C, or a large architecture sweep.

Do not interrupt other experiments.

## 12. Deliverables

Create a new directory such as:

`filter-aware-runs/1028/stage2b5-synthetic-learnability/`

Save:

1. Original update-zero failure reproduction.
2. Training/validation loss and gradient audits.
3. Elementary synthetic learnability tests.
4. Source-only optimisation corrections.
5. N2 matched-template oracle experiment.
6. N2 mismatched-physics analysis.
7. Physical/neural loss-scale diagnostics.
8. Source-family generalisation results.
9. Held-out-filter and held-out-family results where gates pass.
10. Updated architecture comparisons with uncertainty.
11. Exact manifests, seeds, dependency versions and reproduction commands.
12. A concise scientific root-cause report.

End with explicit answers:

1. Why did all original synthetic arms select update zero?
2. Can N0/N1 learn a simple identifiable photometric-redshift problem?
3. Can N2 reproduce the expected physical-template likelihood when its generative assumptions are correct?
4. Does the SED decoder contribute useful generalisation beyond N1 once optimisation is valid?
5. Is performance limited by data quantity, SED-family mismatch, architecture or optimisation?
6. Does any result justify a larger real-source experiment?
7. What evidence, if any, would justify reintroducing pretrained Qwen in Stage 2C?

**Execute the diagnostic and small controlled training experiments. Do not stop at a design proposal. Do not claim architectural superiority from an undertrained or update-zero model.**